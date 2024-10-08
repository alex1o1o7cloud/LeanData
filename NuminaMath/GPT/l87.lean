import Mathlib

namespace middle_admitted_is_correct_l87_87992

-- Define the total number of admitted people.
def total_admitted := 100

-- Define the proportions of South, North, and Middle volumes.
def south_ratio := 11
def north_ratio := 7
def middle_ratio := 2

-- Calculating the total ratio.
def total_ratio := south_ratio + north_ratio + middle_ratio

-- Hypothesis that we are dealing with the correct ratio and total.
def middle_admitted (total_admitted : ℕ) (total_ratio : ℕ) (middle_ratio : ℕ) : ℕ :=
  total_admitted * middle_ratio / total_ratio

-- Proof statement
theorem middle_admitted_is_correct :
  middle_admitted total_admitted total_ratio middle_ratio = 10 :=
by
  -- This line would usually contain the detailed proof steps, which are omitted here.
  sorry

end middle_admitted_is_correct_l87_87992


namespace seokgi_share_is_67_l87_87156

-- The total length of the wire
def length_of_wire := 150

-- Seokgi's share is 16 cm shorter than Yeseul's share
def is_shorter_by (Y S : ℕ) := S = Y - 16

-- The sum of Yeseul's and Seokgi's shares equals the total length
def total_share (Y S : ℕ) := Y + S = length_of_wire

-- Prove that Seokgi's share is 67 cm
theorem seokgi_share_is_67 (Y S : ℕ) (h1 : is_shorter_by Y S) (h2 : total_share Y S) : 
  S = 67 :=
sorry

end seokgi_share_is_67_l87_87156


namespace face_value_of_share_l87_87727

-- Let FV be the face value of each share.
-- Given conditions:
-- Dividend rate is 9%
-- Market value of each share is Rs. 42
-- Desired interest rate is 12%

theorem face_value_of_share (market_value : ℝ) (dividend_rate : ℝ) (interest_rate : ℝ) (FV : ℝ) :
  market_value = 42 ∧ dividend_rate = 0.09 ∧ interest_rate = 0.12 →
  0.09 * FV = 0.12 * market_value →
  FV = 56 :=
by
  sorry

end face_value_of_share_l87_87727


namespace remainder_of_2n_div4_l87_87490

theorem remainder_of_2n_div4 (n : ℕ) (h : ∃ k : ℕ, n = 4 * k + 3) : (2 * n) % 4 = 2 := 
by
  sorry

end remainder_of_2n_div4_l87_87490


namespace projection_of_point_onto_xOy_plane_l87_87117

def point := (ℝ × ℝ × ℝ)

def projection_onto_xOy_plane (P : point) : point :=
  let (x, y, z) := P
  (x, y, 0)

theorem projection_of_point_onto_xOy_plane : 
  projection_onto_xOy_plane (2, 3, 4) = (2, 3, 0) :=
by
  -- proof steps would go here
  sorry

end projection_of_point_onto_xOy_plane_l87_87117


namespace no_third_number_for_lcm_l87_87362

theorem no_third_number_for_lcm (a : ℕ) : ¬ (Nat.lcm (Nat.lcm 23 46) a = 83) :=
sorry

end no_third_number_for_lcm_l87_87362


namespace panteleimon_twos_l87_87320

-- Define the variables
variables (P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2 : ℕ)

-- Define the conditions
def conditions :=
  P_5 + P_4 + P_3 + P_2 = 20 ∧
  G_5 + G_4 + G_3 + G_2 = 20 ∧
  P_5 = G_4 ∧
  P_4 = G_3 ∧
  P_3 = G_2 ∧
  P_2 = G_5 ∧
  (5 * P_5 + 4 * P_4 + 3 * P_3 + 2 * P_2 = 5 * G_5 + 4 * G_4 + 3 * G_3 + 2 * G_2)

-- The proof goal
theorem panteleimon_twos (h : conditions P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2) : P_2 = 5 :=
sorry

end panteleimon_twos_l87_87320


namespace negation_proposition_l87_87056

theorem negation_proposition:
  ¬(∃ x : ℝ, x^2 - x + 1 > 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≤ 0 :=
by
  sorry -- Proof not required as per instructions

end negation_proposition_l87_87056


namespace range_of_m_l87_87594

def P (m : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 ^ 2 + m * x1 + 1 = 0) ∧ (x2 ^ 2 + m * x2 + 1 = 0) ∧ (x1 < 0) ∧ (x2 < 0)

def Q (m : ℝ) : Prop :=
  ∀ (x : ℝ), 4 * x ^ 2 + 4 * (m - 2) * x + 1 ≠ 0

def P_or_Q (m : ℝ) : Prop :=
  P m ∨ Q m

def P_and_Q (m : ℝ) : Prop :=
  P m ∧ Q m

theorem range_of_m (m : ℝ) : P_or_Q m ∧ ¬P_and_Q m ↔ m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3 :=
by {
  sorry
}

end range_of_m_l87_87594


namespace one_over_a_plus_one_over_b_eq_neg_one_l87_87339

theorem one_over_a_plus_one_over_b_eq_neg_one
  (a b : ℝ) (h_distinct : a ≠ b)
  (h_eq : a / b + a = b / a + b) :
  1 / a + 1 / b = -1 :=
by
  sorry

end one_over_a_plus_one_over_b_eq_neg_one_l87_87339


namespace mean_days_correct_l87_87595

noncomputable def mean_days (a1 a2 a3 a4 a5 d1 d2 d3 d4 d5 : ℕ) : ℚ :=
  (a1 * d1 + a2 * d2 + a3 * d3 + a4 * d4 + a5 * d5 : ℚ) / (a1 + a2 + a3 + a4 + a5)

theorem mean_days_correct : mean_days 2 4 5 7 4 1 2 4 5 6 = 4.05 := by
  sorry

end mean_days_correct_l87_87595


namespace maximum_value_ratio_l87_87498

theorem maximum_value_ratio (a b : ℝ) (h1 : a + b - 2 ≥ 0) (h2 : b - a - 1 ≤ 0) (h3 : a ≤ 1) :
  ∃ x, x = (a + 2 * b) / (2 * a + b) ∧ x ≤ 7/5 := sorry

end maximum_value_ratio_l87_87498


namespace total_cost_is_21_l87_87400

-- Definitions of the costs
def cost_almond_croissant : Float := 4.50
def cost_salami_and_cheese_croissant : Float := 4.50
def cost_plain_croissant : Float := 3.00
def cost_focaccia : Float := 4.00
def cost_latte : Float := 2.50

-- Theorem stating the total cost
theorem total_cost_is_21 :
  (cost_almond_croissant + cost_salami_and_cheese_croissant) + (2 * cost_latte) + cost_plain_croissant + cost_focaccia = 21.00 :=
by
  sorry

end total_cost_is_21_l87_87400


namespace total_people_on_hike_l87_87437

-- Definitions of the conditions
def n_cars : ℕ := 3
def n_people_per_car : ℕ := 4
def n_taxis : ℕ := 6
def n_people_per_taxi : ℕ := 6
def n_vans : ℕ := 2
def n_people_per_van : ℕ := 5

-- Statement of the problem
theorem total_people_on_hike : 
  n_cars * n_people_per_car + n_taxis * n_people_per_taxi + n_vans * n_people_per_van = 58 :=
by sorry

end total_people_on_hike_l87_87437


namespace five_segments_acute_angle_l87_87208

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_obtuse (a b c : ℝ) : Prop :=
  c^2 > a^2 + b^2

def is_acute (a b c : ℝ) : Prop :=
  c^2 < a^2 + b^2

theorem five_segments_acute_angle (a b c d e : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (T1 : is_triangle a b c) (T2 : is_triangle a b d) (T3 : is_triangle a b e)
  (T4 : is_triangle a c d) (T5 : is_triangle a c e) (T6 : is_triangle a d e)
  (T7 : is_triangle b c d) (T8 : is_triangle b c e) (T9 : is_triangle b d e)
  (T10 : is_triangle c d e) : 
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
           is_triangle x y z ∧ is_acute x y z :=
by
  sorry

end five_segments_acute_angle_l87_87208


namespace danielle_money_for_supplies_l87_87127

-- Define the conditions
def cost_of_molds := 3
def cost_of_sticks_pack := 1
def sticks_in_pack := 100
def cost_of_juice_bottle := 2
def popsicles_per_bottle := 20
def remaining_sticks := 40
def used_sticks := sticks_in_pack - remaining_sticks

-- Define number of juice bottles used
def bottles_of_juice_used : ℕ := used_sticks / popsicles_per_bottle

-- Define the total cost
def total_cost : ℕ := cost_of_molds + cost_of_sticks_pack + bottles_of_juice_used * cost_of_juice_bottle

-- Prove that Danielle had $10 for supplies
theorem danielle_money_for_supplies : total_cost = 10 := by {
  sorry
}

end danielle_money_for_supplies_l87_87127


namespace determine_k_l87_87916

theorem determine_k (k r s : ℝ) (h1 : r + s = -k) (h2 : (r + 3) + (s + 3) = k) : k = 3 :=
by
  sorry

end determine_k_l87_87916


namespace find_initial_principal_amount_l87_87533

noncomputable def compound_interest (initial_principal : ℝ) : ℝ :=
  let year1 := initial_principal * 1.09
  let year2 := (year1 + 500) * 1.10
  let year3 := (year2 - 300) * 1.08
  let year4 := year3 * 1.08
  let year5 := year4 * 1.09
  year5

theorem find_initial_principal_amount :
  ∃ (P : ℝ), (|compound_interest P - 1120| < 0.01) :=
sorry

end find_initial_principal_amount_l87_87533


namespace number_of_tens_in_sum_l87_87805

theorem number_of_tens_in_sum (n : ℕ) : (100^n) / 10 = 10^(2*n - 1) :=
by sorry

end number_of_tens_in_sum_l87_87805


namespace max_value_of_x_l87_87936

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem max_value_of_x 
  (x : ℤ) 
  (h : log_base (1 / 4 : ℝ) (2 * x + 1) < log_base (1 / 2 : ℝ) (x - 1)) : x ≤ 3 :=
sorry

end max_value_of_x_l87_87936


namespace remainder_3_pow_17_mod_5_l87_87060

theorem remainder_3_pow_17_mod_5 :
  (3^17) % 5 = 3 :=
by
  have h : 3^4 % 5 = 1 := by norm_num
  sorry

end remainder_3_pow_17_mod_5_l87_87060


namespace average_goals_l87_87656

theorem average_goals (c s j : ℕ) (h1 : c = 4) (h2 : s = c / 2) (h3 : j = 2 * s - 3) :
  c + s + j = 7 :=
sorry

end average_goals_l87_87656


namespace probability_intersection_interval_l87_87472

theorem probability_intersection_interval (PA PB p : ℝ) (hPA : PA = 5 / 6) (hPB : PB = 3 / 4) :
  0 ≤ p ∧ p ≤ 3 / 4 :=
sorry

end probability_intersection_interval_l87_87472


namespace solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l87_87408

-- Given function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 1) * x - a

-- Problem 1: for a = 2, solution to f(x) < 0
theorem solution_set_f_lt_zero_a_two :
  { x : ℝ | f x 2 < 0 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Problem 2: for any a in ℝ, solution to f(x) > 0
theorem solution_set_f_gt_zero (a : ℝ) :
  { x : ℝ | f x a > 0 } =
  if a > -1 then
    {x : ℝ | x < -1} ∪ {x : ℝ | x > a}
  else if a = -1 then
    {x : ℝ | x ≠ -1}
  else
    {x : ℝ | x < a} ∪ {x : ℝ | x > -1} :=
sorry

end solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l87_87408


namespace product_469160_9999_l87_87185

theorem product_469160_9999 :
  469160 * 9999 = 4690696840 :=
by
  sorry

end product_469160_9999_l87_87185


namespace correct_word_is_tradition_l87_87685

-- Definitions of the words according to the problem conditions
def tradition : String := "custom, traditional practice"
def balance : String := "equilibrium"
def concern : String := "worry, care about"
def relationship : String := "relation"

-- The sentence to be filled
def sentence (word : String) : String :=
"There’s a " ++ word ++ " in our office that when it’s somebody’s birthday, they bring in a cake for us all to share."

-- The proof problem statement
theorem correct_word_is_tradition :
  ∀ word, (word ≠ tradition) → (sentence word ≠ "There’s a tradition in our office that when it’s somebody’s birthday, they bring in a cake for us all to share.") :=
by sorry

end correct_word_is_tradition_l87_87685


namespace quadratic_does_not_pass_third_quadrant_l87_87410

-- Definitions of the functions
def linear_function (a b x : ℝ) : ℝ := -a * x + b
def quadratic_function (a b x : ℝ) : ℝ := -a * x^2 + b * x

-- Conditions
variables (a b : ℝ)
axiom a_nonzero : a ≠ 0
axiom passes_first_third_fourth : ∀ x, (linear_function a b x > 0 ∧ x > 0) ∨ (linear_function a b x < 0 ∧ x < 0) ∨ (linear_function a b x < 0 ∧ x > 0)

-- Theorem stating the problem
theorem quadratic_does_not_pass_third_quadrant :
  ¬ (∃ x, quadratic_function a b x < 0 ∧ x < 0) := 
sorry

end quadratic_does_not_pass_third_quadrant_l87_87410


namespace sum_of_x_y_l87_87985

theorem sum_of_x_y (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 48) : x + y = 2 :=
sorry

end sum_of_x_y_l87_87985


namespace modulus_of_complex_raised_to_eight_l87_87224

-- Define the complex number 2 + i in Lean
def z : Complex := Complex.mk 2 1

-- State the proof problem with conditions
theorem modulus_of_complex_raised_to_eight : Complex.abs (z ^ 8) = 625 := by
  sorry

end modulus_of_complex_raised_to_eight_l87_87224


namespace sum_of_solutions_eq_minus_2_l87_87307

-- Defining the equation and the goal
theorem sum_of_solutions_eq_minus_2 (x1 x2 : ℝ) (floor : ℝ → ℤ) (h1 : floor (3 * x1 + 1) = 2 * x1 - 1 / 2)
(h2 : floor (3 * x2 + 1) = 2 * x2 - 1 / 2) :
  x1 + x2 = -2 :=
sorry

end sum_of_solutions_eq_minus_2_l87_87307


namespace find_a_b_l87_87461

noncomputable def f (a b x: ℝ) : ℝ := x / (a * x + b)

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : f a b (-4) = 4) (h₃ : ∀ x, f a b x = f b a x) :
  a + b = 3 / 2 :=
sorry

end find_a_b_l87_87461


namespace simplify_expression_l87_87716

noncomputable def simplify_expr (a b : ℝ) : ℝ :=
  (3 * a^5 * b^3 + a^4 * b^2) / (-(a^2 * b)^2) - (2 + a) * (2 - a) - a * (a - 5 * b)

theorem simplify_expression (a b : ℝ) :
  simplify_expr a b = 8 * a * b - 3 := 
by
  sorry

end simplify_expression_l87_87716


namespace scores_greater_than_18_l87_87118

theorem scores_greater_than_18 (scores : Fin 20 → ℝ) 
  (h_unique : Function.Injective scores)
  (h_sum : ∀ i j k : Fin 20, i ≠ j → i ≠ k → j ≠ k → scores i < scores j + scores k) :
  ∀ i : Fin 20, scores i > 18 := 
by
  sorry

end scores_greater_than_18_l87_87118


namespace equation_pattern_l87_87207
open Nat

theorem equation_pattern (n : ℕ) (h_pos : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end equation_pattern_l87_87207


namespace div_gcd_iff_div_ab_gcd_mul_l87_87204

variable (a b n c : ℕ)
variables (h₀ : a ≠ 0) (d : ℕ)
variable (hd : d = Nat.gcd a b)

theorem div_gcd_iff_div_ab : (n ∣ a ∧ n ∣ b) ↔ n ∣ d :=
by
  sorry

theorem gcd_mul (h₁ : c > 0) : Nat.gcd (a * c) (b * c) = c * Nat.gcd a b :=
by
  sorry

end div_gcd_iff_div_ab_gcd_mul_l87_87204


namespace distinct_integers_sum_l87_87376

theorem distinct_integers_sum {p q r s t : ℤ} 
    (h1 : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 120)
    (h2 : p ≠ q) (h3 : p ≠ r) (h4 : p ≠ s) (h5 : p ≠ t) 
    (h6 : q ≠ r) (h7 : q ≠ s) (h8 : q ≠ t) 
    (h9 : r ≠ s) (h10 : r ≠ t) (h11 : s ≠ t) : 
  p + q + r + s + t = 35 := 
sorry

end distinct_integers_sum_l87_87376


namespace NorrisSavings_l87_87568

theorem NorrisSavings : 
  let saved_september := 29
  let saved_october := 25
  let saved_november := 31
  let saved_december := 35
  let saved_january := 40
  saved_september + saved_october + saved_november + saved_december + saved_january = 160 :=
by
  sorry

end NorrisSavings_l87_87568


namespace base_satisfying_eq_l87_87163

theorem base_satisfying_eq : ∃ a : ℕ, (11 < a) ∧ (293 * a^2 + 9 * a + 3 + (4 * a^2 + 6 * a + 8) = 7 * a^2 + 3 * a + 11) ∧ (a = 12) :=
by
  sorry

end base_satisfying_eq_l87_87163


namespace max_value_m_l87_87539

variable {a b m : ℝ}

theorem max_value_m (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, (3 / a) + (1 / b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
by 
  sorry

end max_value_m_l87_87539


namespace min_distance_sum_well_l87_87011

theorem min_distance_sum_well (A B C : ℝ) (h1 : B = A + 50) (h2 : C = B + 50) :
  ∃ X : ℝ, X = B ∧ (∀ Y : ℝ, (dist Y A + dist Y B + dist Y C) ≥ (dist B A + dist B B + dist B C)) :=
sorry

end min_distance_sum_well_l87_87011


namespace gcd_gt_one_l87_87365

-- Defining the given conditions and the statement to prove
theorem gcd_gt_one (a b x y : ℕ) (h : (a^2 + b^2) ∣ (a * x + b * y)) : 
  Nat.gcd (x^2 + y^2) (a^2 + b^2) > 1 := 
sorry

end gcd_gt_one_l87_87365


namespace total_songs_purchased_is_162_l87_87332

variable (c_country : ℕ) (c_pop : ℕ) (c_jazz : ℕ) (c_rock : ℕ)
variable (s_country : ℕ) (s_pop : ℕ) (s_jazz : ℕ) (s_rock : ℕ)

-- Setting up the conditions
def num_country_albums := 6
def num_pop_albums := 2
def num_jazz_albums := 4
def num_rock_albums := 3

-- Number of songs per album
def country_album_songs := 9
def pop_album_songs := 9
def jazz_album_songs := 12
def rock_album_songs := 14

theorem total_songs_purchased_is_162 :
  num_country_albums * country_album_songs +
  num_pop_albums * pop_album_songs +
  num_jazz_albums * jazz_album_songs +
  num_rock_albums * rock_album_songs = 162 := by
  sorry

end total_songs_purchased_is_162_l87_87332


namespace simplify_expression1_simplify_expression2_l87_87770

-- Define variables as real numbers or appropriate domains
variables {a b x y: ℝ}

-- Problem 1
theorem simplify_expression1 : (2 * a - b) - (2 * b - 3 * a) - 2 * (a - 2 * b) = 3 * a + b :=
by sorry

-- Problem 2
theorem simplify_expression2 : (4 * x^2 - 5 * x * y) - (1 / 3 * y^2 + 2 * x^2) + 2 * (3 * x * y - 1 / 4 * y^2 - 1 / 12 * y^2) = 2 * x^2 + x * y - y^2 :=
by sorry

end simplify_expression1_simplify_expression2_l87_87770


namespace average_mark_of_all_three_boys_is_432_l87_87865

noncomputable def max_score : ℝ := 900
noncomputable def get_score (percent : ℝ) : ℝ := (percent / 100) * max_score

noncomputable def amar_score : ℝ := get_score 64
noncomputable def bhavan_score : ℝ := get_score 36
noncomputable def chetan_score : ℝ := get_score 44

noncomputable def total_score : ℝ := amar_score + bhavan_score + chetan_score
noncomputable def average_score : ℝ := total_score / 3

theorem average_mark_of_all_three_boys_is_432 : average_score = 432 := 
by
  sorry

end average_mark_of_all_three_boys_is_432_l87_87865


namespace laptop_sticker_price_l87_87205

theorem laptop_sticker_price (x : ℝ) (h₁ : 0.70 * x = 0.80 * x - 50 - 30) : x = 800 := 
  sorry

end laptop_sticker_price_l87_87205


namespace solution_set_f_gt_0_l87_87043

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2*x - 3 else - (x^2 - 2*x - 3)

theorem solution_set_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x > 3 ∨ (-3 < x ∧ x < 0)} :=
by
  sorry

end solution_set_f_gt_0_l87_87043


namespace luke_plays_14_rounds_l87_87087

theorem luke_plays_14_rounds (total_points : ℕ) (points_per_round : ℕ)
  (h1 : total_points = 154) (h2 : points_per_round = 11) : 
  total_points / points_per_round = 14 := by
  sorry

end luke_plays_14_rounds_l87_87087


namespace initial_boys_l87_87449

-- Define the initial conditions
def initial_girls := 4
def final_children := 8
def boys_left := 3
def girls_entered := 2

-- Define the statement to be proved
theorem initial_boys : 
  ∃ (B : ℕ), (B - boys_left) + (initial_girls + girls_entered) = final_children ∧ B = 5 :=
by
  -- Placeholder for the proof
  sorry

end initial_boys_l87_87449


namespace cohen_saw_1300_fish_eater_birds_l87_87261

theorem cohen_saw_1300_fish_eater_birds :
  let day1 := 300
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3 = 1300 :=
by
  sorry

end cohen_saw_1300_fish_eater_birds_l87_87261


namespace combined_tax_rate_l87_87069

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (Mork_tax_rate Mindy_tax_rate : ℝ)
  (h1 : Mork_tax_rate = 0.4) (h2 : Mindy_tax_rate = 0.3) (h3 : Mindy_income = 4 * Mork_income) :
  ((Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income)) * 100 = 32 :=
by
  sorry

end combined_tax_rate_l87_87069


namespace find_original_wage_l87_87445

-- Defining the conditions
variables (W : ℝ) (W_new : ℝ) (h : W_new = 35) (h2 : W_new = 1.40 * W)

-- Statement that needs to be proved
theorem find_original_wage : W = 25 :=
by
  -- proof omitted
  sorry

end find_original_wage_l87_87445


namespace cubes_sum_identity_l87_87606

variable {a b : ℝ}

theorem cubes_sum_identity (h : (a / (1 + b) + b / (1 + a) = 1)) : a^3 + b^3 = a + b :=
sorry

end cubes_sum_identity_l87_87606


namespace x_is_perfect_square_l87_87293

theorem x_is_perfect_square (x y : ℕ) (hx_pos : x > 0) (hy_pos : y > 0) (hdiv : 2 * x * y ∣ x^2 + y^2 - x) : ∃ (n : ℕ), x = n^2 :=
by
  sorry

end x_is_perfect_square_l87_87293


namespace Paula_needs_52_tickets_l87_87580

theorem Paula_needs_52_tickets :
  let g := 2
  let b := 4
  let r := 3
  let f := 1
  let t_g := 4
  let t_b := 5
  let t_r := 7
  let t_f := 3
  g * t_g + b * t_b + r * t_r + f * t_f = 52 := by
  intros
  sorry

end Paula_needs_52_tickets_l87_87580


namespace largest_possible_number_of_red_socks_l87_87157

noncomputable def max_red_socks (t : ℕ) (r : ℕ) : Prop :=
  t ≤ 1991 ∧
  ((r * (r - 1) + (t - r) * (t - r - 1)) / (t * (t - 1)) = 1 / 2) ∧
  ∀ r', r' ≤ 990 → (t ≤ 1991 ∧
    ((r' * (r' - 1) + (t - r') * (t - r' - 1)) / (t * (t - 1)) = 1 / 2) → r ≤ r')

theorem largest_possible_number_of_red_socks :
  ∃ t r, max_red_socks t r ∧ r = 990 :=
by
  sorry

end largest_possible_number_of_red_socks_l87_87157


namespace jane_reads_105_pages_in_a_week_l87_87592

-- Define the pages read in the morning and evening
def pages_morning := 5
def pages_evening := 10

-- Define the number of pages read in a day
def pages_per_day := pages_morning + pages_evening

-- Define the number of days in a week
def days_per_week := 7

-- Define the total number of pages read in a week
def pages_per_week := pages_per_day * days_per_week

-- The theorem that sums up the proof
theorem jane_reads_105_pages_in_a_week : pages_per_week = 105 := by
  sorry

end jane_reads_105_pages_in_a_week_l87_87592


namespace min_species_needed_l87_87037

theorem min_species_needed (num_birds : ℕ) (h1 : num_birds = 2021)
  (h2 : ∀ (s : ℤ) (x y : ℕ), x ≠ y → (between_same_species : ℕ) → (h3 : between_same_species = y - x - 1) → between_same_species % 2 = 0) :
  ∃ (species : ℕ), num_birds ≤ 2 * species ∧ species = 1011 :=
by
  sorry

end min_species_needed_l87_87037


namespace power_inequality_l87_87326

theorem power_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^5 + b^5 > a^2 * b^3 + a^3 * b^2 :=
sorry

end power_inequality_l87_87326


namespace range_of_a_l87_87337

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l87_87337


namespace find_integer_pairs_l87_87312

theorem find_integer_pairs (x y : ℤ) (h_xy : x ≤ y) (h_eq : (1 : ℚ)/x + (1 : ℚ)/y = 1/4) :
  (x, y) = (5, 20) ∨ (x, y) = (6, 12) ∨ (x, y) = (8, 8) ∨ (x, y) = (-4, 2) ∨ (x, y) = (-12, 3) :=
sorry

end find_integer_pairs_l87_87312


namespace cake_eaten_after_four_trips_l87_87099

-- Define the fraction of the cake eaten on each trip
def fraction_eaten (n : Nat) : ℚ :=
  (1 / 3) ^ n

-- Define the total cake eaten after four trips
def total_eaten_after_four_trips : ℚ :=
  fraction_eaten 1 + fraction_eaten 2 + fraction_eaten 3 + fraction_eaten 4

-- The mathematical statement we want to prove
theorem cake_eaten_after_four_trips : total_eaten_after_four_trips = 40 / 81 := 
by
  sorry

end cake_eaten_after_four_trips_l87_87099


namespace smallest_even_number_l87_87215

theorem smallest_even_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6)) = 140) : x = 32 :=
by
  sorry

end smallest_even_number_l87_87215


namespace problem_solution_l87_87407

-- Define the variables and the conditions
variable (a b c : ℝ)
axiom h1 : a^2 + 2 * b = 7
axiom h2 : b^2 - 2 * c = -1
axiom h3 : c^2 - 6 * a = -17

-- State the theorem to be proven
theorem problem_solution : a + b + c = 3 := 
by sorry

end problem_solution_l87_87407


namespace chord_angle_measure_l87_87488

theorem chord_angle_measure (AB_ratio : ℕ) (circ : ℝ) (h : AB_ratio = 1 + 5) : 
  ∃ θ : ℝ, θ = (1 / 6) * circ ∧ θ = 60 :=
by
  sorry

end chord_angle_measure_l87_87488


namespace circle_passing_given_points_l87_87671

theorem circle_passing_given_points :
  ∃ (D E F : ℚ), (F = 0) ∧ (E = - (9 / 5)) ∧ (D = 19 / 5) ∧
  (∀ (x y : ℚ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 3) ∨ (x = -4 ∧ y = 1)) :=
by
  sorry

end circle_passing_given_points_l87_87671


namespace union_complement_A_B_eq_l87_87460

-- Define the universal set U, set A, and set B
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- The statement to be proved
theorem union_complement_A_B_eq {U A B : Set ℕ} (hU : U = {0, 1, 2, 3, 4}) 
  (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) :
  (complement_U_A) ∪ B = {2, 3, 4} := 
by
  sorry

end union_complement_A_B_eq_l87_87460


namespace problem1_l87_87274

theorem problem1 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
sorry

end problem1_l87_87274


namespace current_failing_rate_l87_87507

def failing_student_rate := 28

def is_failing_student_rate (V : Prop) (n : ℕ) (rate : ℕ) : Prop :=
  (V ∧ rate = 24 ∧ n = 25) ∨ (¬V ∧ rate = 25 ∧ n - 1 = 24)

theorem current_failing_rate (V : Prop) (n : ℕ) (rate : ℕ) :
  is_failing_student_rate V n rate → rate = failing_student_rate :=
by
  sorry

end current_failing_rate_l87_87507


namespace no_four_points_with_all_odd_distances_l87_87643

theorem no_four_points_with_all_odd_distances :
  ∀ (A B C D : ℝ × ℝ),
    (∃ (x y z p q r : ℕ),
      (x = dist A B ∧ x % 2 = 1) ∧
      (y = dist B C ∧ y % 2 = 1) ∧
      (z = dist C D ∧ z % 2 = 1) ∧
      (p = dist D A ∧ p % 2 = 1) ∧
      (q = dist A C ∧ q % 2 = 1) ∧
      (r = dist B D ∧ r % 2 = 1))
    → false :=
by
  sorry

end no_four_points_with_all_odd_distances_l87_87643


namespace ratio_length_to_width_l87_87268

-- Define the given conditions and values
def width : ℕ := 75
def perimeter : ℕ := 360

-- Define the proof problem statement
theorem ratio_length_to_width (L : ℕ) (P_eq : perimeter = 2 * L + 2 * width) :
  (L / width : ℚ) = 7 / 5 :=
sorry

end ratio_length_to_width_l87_87268


namespace pick_three_cards_in_order_l87_87145

theorem pick_three_cards_in_order (deck_size : ℕ) (first_card_ways : ℕ) (second_card_ways : ℕ) (third_card_ways : ℕ) 
  (total_combinations : ℕ) (h1 : deck_size = 52) (h2 : first_card_ways = 52) 
  (h3 : second_card_ways = 51) (h4 : third_card_ways = 50) (h5 : total_combinations = first_card_ways * second_card_ways * third_card_ways) : 
  total_combinations = 132600 := 
by 
  sorry

end pick_three_cards_in_order_l87_87145


namespace no_stromino_covering_of_5x5_board_l87_87256

-- Define the conditions
def isStromino (r : ℕ) (c : ℕ) : Prop := 
  (r = 3 ∧ c = 1) ∨ (r = 1 ∧ c = 3)

def is5x5Board (r c : ℕ) : Prop := 
  r = 5 ∧ c = 5

-- The main goal is to show this proposition
theorem no_stromino_covering_of_5x5_board : 
  ∀ (board_size : ℕ × ℕ),
    is5x5Board board_size.1 board_size.2 →
    ∀ (stromino_count : ℕ),
      stromino_count = 16 →
      (∀ (stromino : ℕ × ℕ), 
        isStromino stromino.1 stromino.2 →
        ∀ (cover : ℕ), 
          3 = cover) →
      ¬(∃ (cover_fn : ℕ × ℕ → ℕ), 
          (∀ (pos : ℕ × ℕ), pos.fst < 5 ∧ pos.snd < 5 →
            cover_fn pos = 1 ∨ cover_fn pos = 2) ∧
          (∀ (i : ℕ), i < 25 → 
            ∃ (stromino_pos : ℕ × ℕ), 
              stromino_pos.fst < 5 ∧ stromino_pos.snd < 5 ∧ 
              -- Each stromino must cover exactly 3 squares, 
              -- which implies that the covering function must work appropriately.
              (cover_fn (stromino_pos.fst, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 1, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 2, stromino_pos.snd) = 3 ∨
               cover_fn (stromino_pos.fst, stromino_pos.snd + 1) +
               cover_fn (stromino_pos.fst, stromino_pos.snd + 2) = 3))) :=
by sorry

end no_stromino_covering_of_5x5_board_l87_87256


namespace exact_time_now_l87_87126

/-- Given that it is between 9:00 and 10:00 o'clock,
and nine minutes from now, the minute hand of a watch
will be exactly opposite the place where the hour hand
was six minutes ago, show that the exact time now is 9:06
-/
theorem exact_time_now 
  (t : ℕ)
  (h1 : t < 60)
  (h2 : ∃ t, 6 * (t + 9) - (270 + 0.5 * (t - 6)) = 180 ∨ 6 * (t + 9) - (270 + 0.5 * (t - 6)) = -180) :
  t = 6 := 
sorry

end exact_time_now_l87_87126


namespace stacy_height_last_year_l87_87839

-- Definitions for the conditions
def brother_growth := 1
def stacy_growth := brother_growth + 6
def stacy_current_height := 57
def stacy_last_years_height := stacy_current_height - stacy_growth

-- Proof statement
theorem stacy_height_last_year : stacy_last_years_height = 50 :=
by
  -- proof steps will go here
  sorry

end stacy_height_last_year_l87_87839


namespace initial_sheep_count_l87_87765

theorem initial_sheep_count (S : ℕ) :
  let S1 := S - (S / 3 + 1 / 3)
  let S2 := S1 - (S1 / 4 + 1 / 4)
  let S3 := S2 - (S2 / 5 + 3 / 5)
  S3 = 409
  → S = 1025 := 
by 
  sorry

end initial_sheep_count_l87_87765


namespace seq_solution_l87_87733

theorem seq_solution {a b : ℝ} (h1 : a - b = 8) (h2 : a + b = 11) : 2 * a = 19 ∧ 2 * b = 3 := by
  sorry

end seq_solution_l87_87733


namespace sum_of_squares_500_l87_87020

theorem sum_of_squares_500 : (Finset.range 500).sum (λ x => (x + 1) ^ 2) = 41841791750 := by
  sorry

end sum_of_squares_500_l87_87020


namespace max_value_fraction_l87_87095

theorem max_value_fraction (a b : ℝ) 
  (h1 : a + b - 2 ≥ 0)
  (h2 : b - a - 1 ≤ 0)
  (h3 : a ≤ 1) :
  ∃ max_val, max_val = 7 / 5 ∧ ∀ (x y : ℝ), 
    (x + y - 2 ≥ 0) → (y - x - 1 ≤ 0) → (x ≤ 1) → (x + 2*y) / (2*x + y) ≤ max_val :=
sorry

end max_value_fraction_l87_87095


namespace number_of_ways_split_2000_cents_l87_87907

theorem number_of_ways_split_2000_cents : 
  ∃ n : ℕ, n = 357 ∧ (∃ (nick d q : ℕ), 
    nick > 0 ∧ d > 0 ∧ q > 0 ∧ 5 * nick + 10 * d + 25 * q = 2000) :=
sorry

end number_of_ways_split_2000_cents_l87_87907


namespace find_original_number_l87_87116

theorem find_original_number (x : ℝ) : 1.5 * x = 525 → x = 350 := by
  sorry

end find_original_number_l87_87116


namespace trig_inequalities_l87_87918

theorem trig_inequalities :
  let sin_168 := Real.sin (168 * Real.pi / 180)
  let cos_10 := Real.cos (10 * Real.pi / 180)
  let tan_58 := Real.tan (58 * Real.pi / 180)
  let tan_45 := Real.tan (45 * Real.pi / 180)
  sin_168 < cos_10 ∧ cos_10 < tan_58 :=
  sorry

end trig_inequalities_l87_87918


namespace min_sum_of_dimensions_l87_87161

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 2310) :
  a + b + c = 42 :=
sorry

end min_sum_of_dimensions_l87_87161


namespace count_valid_numbers_l87_87266

theorem count_valid_numbers : 
  let count_A := 10 
  let count_B := 2 
  count_A * count_B = 20 :=
by 
  let count_A := 10
  let count_B := 2
  have : count_A * count_B = 20 := by norm_num
  exact this

end count_valid_numbers_l87_87266


namespace ratio_books_donated_l87_87886

theorem ratio_books_donated (initial_books: ℕ) (books_given_nephew: ℕ) (books_after_nephew: ℕ) 
  (books_final: ℕ) (books_purchased: ℕ) (books_donated_library: ℕ) (ratio: ℕ):
    initial_books = 40 → 
    books_given_nephew = initial_books / 4 → 
    books_after_nephew = initial_books - books_given_nephew →
    books_final = 23 →
    books_purchased = 3 →
    books_donated_library = books_after_nephew - (books_final - books_purchased) →
    ratio = books_donated_library / books_after_nephew →
    ratio = 1 / 3 := sorry

end ratio_books_donated_l87_87886


namespace gain_percentage_is_15_l87_87499

-- Initial conditions
def CP_A : ℤ := 100
def CP_B : ℤ := 200
def CP_C : ℤ := 300
def SP_A : ℤ := 110
def SP_B : ℤ := 250
def SP_C : ℤ := 330

-- Definitions for total values
def Total_CP : ℤ := CP_A + CP_B + CP_C
def Total_SP : ℤ := SP_A + SP_B + SP_C
def Overall_gain : ℤ := Total_SP - Total_CP
def Gain_percentage : ℚ := (Overall_gain * 100) / Total_CP

-- Theorem to prove the overall gain percentage
theorem gain_percentage_is_15 :
  Gain_percentage = 15 := 
by
  -- Proof placeholder
  sorry

end gain_percentage_is_15_l87_87499


namespace distance_centers_triangle_l87_87966

noncomputable def distance_between_centers (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  let circumradius := (a * b * c) / (4 * K)
  let hypotenuse := by
    by_cases hc : a * a + b * b = c * c
    exact c
    by_cases hb : a * a + c * c = b * b
    exact b
    by_cases ha : b * b + c * c = a * a
    exact a
    exact 0
  let oc := hypotenuse / 2
  Real.sqrt (oc * oc + r * r)

theorem distance_centers_triangle :
  distance_between_centers 7 24 25 = Real.sqrt 165.25 := sorry

end distance_centers_triangle_l87_87966


namespace total_tickets_l87_87728

theorem total_tickets (O B : ℕ) (h1 : 12 * O + 8 * B = 3320) (h2 : B = O + 90) : O + B = 350 := by
  sorry

end total_tickets_l87_87728


namespace ages_of_children_l87_87483

theorem ages_of_children : ∃ (a1 a2 a3 a4 : ℕ),
  a1 + a2 + a3 + a4 = 33 ∧
  (a1 - 3) + (a2 - 3) + (a3 - 3) + (a4 - 3) = 22 ∧
  (a1 - 7) + (a2 - 7) + (a3 - 7) + (a4 - 7) = 11 ∧
  (a1 - 13) + (a2 - 13) + (a3 - 13) + (a4 - 13) = 1 ∧
  a1 = 14 ∧ a2 = 11 ∧ a3 = 6 ∧ a4 = 2 :=
by
  sorry

end ages_of_children_l87_87483


namespace cds_per_rack_l87_87218

theorem cds_per_rack (total_cds : ℕ) (racks_per_shelf : ℕ) (cds_per_rack : ℕ) 
  (h1 : total_cds = 32) 
  (h2 : racks_per_shelf = 4) : 
  cds_per_rack = total_cds / racks_per_shelf :=
by 
  sorry

end cds_per_rack_l87_87218


namespace arithmetic_seq_condition_l87_87375

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := 
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_seq_condition (a2 : ℕ) (S3 S9 : ℕ) :
  a2 = 1 → 
  (∃ d, (d > 4 ∧ S3 = 3 * a2 + (3 * (3 - 1) / 2) * d ∧ S9 = 9 * a2 + (9 * (9 - 1) / 2) * d) → (S3 + S9) > 93) ↔ 
  (∃ d, (S3 + S9 = sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9 ∧ (sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9) > 93 → d > 3 ∧ a2 + d > 5)) :=
by 
  sorry

end arithmetic_seq_condition_l87_87375


namespace coins_in_box_l87_87908

theorem coins_in_box (n : ℕ) 
    (h1 : n % 8 = 7) 
    (h2 : n % 7 = 5) : 
    n = 47 ∧ (47 % 9 = 2) :=
sorry

end coins_in_box_l87_87908


namespace triangle_area_l87_87752

/-- Given a triangle ABC with BC = 12 cm and AD perpendicular to BC with AD = 15 cm,
    prove that the area of triangle ABC is 90 square centimeters. -/
theorem triangle_area {BC AD : ℝ} (hBC : BC = 12) (hAD : AD = 15) :
  (1 / 2) * BC * AD = 90 := by
  sorry

end triangle_area_l87_87752


namespace average_payment_l87_87471

theorem average_payment (n m : ℕ) (p1 p2 : ℕ) (h1 : n = 20) (h2 : m = 45) (h3 : p1 = 410) (h4 : p2 = 475) :
  (20 * p1 + 45 * p2) / 65 = 455 :=
by
  sorry

end average_payment_l87_87471


namespace find_num_chickens_l87_87833

-- Definitions based on problem conditions
def num_dogs : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2
def total_legs_seen : ℕ := 12

-- Proof problem: Prove the number of chickens Mrs. Hilt saw
theorem find_num_chickens (C : ℕ) (h1 : num_dogs * legs_per_dog + C * legs_per_chicken = total_legs_seen) : C = 2 := 
sorry

end find_num_chickens_l87_87833


namespace probability_sqrt_two_digit_less_than_seven_l87_87863

noncomputable def prob_sqrt_less_than_seven : ℚ := 
  let favorable := 39
  let total := 90
  favorable / total

theorem probability_sqrt_two_digit_less_than_seven : 
  prob_sqrt_less_than_seven = 13 / 30 := by
  sorry

end probability_sqrt_two_digit_less_than_seven_l87_87863


namespace composite_prop_true_l87_87285

def p : Prop := ∀ (x : ℝ), x > 0 → x + (1/(2*x)) ≥ 1

def q : Prop := ∀ (x : ℝ), x > 1 → (x^2 + 2*x - 3 > 0)

theorem composite_prop_true : p ∨ q :=
by
  sorry

end composite_prop_true_l87_87285


namespace sign_up_ways_l87_87555

theorem sign_up_ways : (3 ^ 4) = 81 :=
by
  sorry

end sign_up_ways_l87_87555


namespace slower_train_crosses_faster_in_36_seconds_l87_87786

-- Define the conditions of the problem
def speed_fast_train_kmph : ℚ := 110
def speed_slow_train_kmph : ℚ := 90
def length_fast_train_km : ℚ := 1.10
def length_slow_train_km : ℚ := 0.90

-- Convert speeds to m/s
def speed_fast_train_mps : ℚ := speed_fast_train_kmph * (1000 / 3600)
def speed_slow_train_mps : ℚ := speed_slow_train_kmph * (1000 / 3600)

-- Relative speed when moving in opposite directions
def relative_speed_mps : ℚ := speed_fast_train_mps + speed_slow_train_mps

-- Convert lengths to meters
def length_fast_train_m : ℚ := length_fast_train_km * 1000
def length_slow_train_m : ℚ := length_slow_train_km * 1000

-- Combined length of both trains in meters
def combined_length_m : ℚ := length_fast_train_m + length_slow_train_m

-- Time taken for the slower train to cross the faster train
def crossing_time : ℚ := combined_length_m / relative_speed_mps

theorem slower_train_crosses_faster_in_36_seconds :
  crossing_time = 36 := by
  sorry

end slower_train_crosses_faster_in_36_seconds_l87_87786


namespace range_of_a_l87_87890

theorem range_of_a (a : ℝ) : (1 ∉ {x : ℝ | (x - a) / (x + a) < 0}) → ( -1 ≤ a ∧ a ≤ 1 ) := 
by
  intro h
  sorry

end range_of_a_l87_87890


namespace adjacent_zero_point_range_l87_87725

def f (x : ℝ) : ℝ := x - 1
def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a*x - a + 3

theorem adjacent_zero_point_range (a : ℝ) :
  (∀ β, (∃ x, g x a = 0) → (|1 - β| ≤ 1 → (∃ x, f x = 0 → |x - β| ≤ 1))) →
  (2 ≤ a ∧ a ≤ 7 / 3) :=
sorry

end adjacent_zero_point_range_l87_87725


namespace fraction_of_7000_l87_87213

theorem fraction_of_7000 (x : ℝ) 
  (h1 : (1 / 10 / 100) * 7000 = 7) 
  (h2 : x * 7000 - 7 = 700) : 
  x = 0.101 :=
by
  sorry

end fraction_of_7000_l87_87213


namespace ratio_of_c_and_d_l87_87708

theorem ratio_of_c_and_d (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 3 * x + 2 * y = c) 
  (h2 : 4 * y - 6 * x = d) : c / d = -1 / 3 := 
sorry

end ratio_of_c_and_d_l87_87708


namespace traveler_meets_truck_at_15_48_l87_87001

noncomputable def timeTravelerMeetsTruck : ℝ := 15 + 48 / 60

theorem traveler_meets_truck_at_15_48 {S Vp Vm Vg : ℝ}
  (h_travel_covered : Vp = S / 4)
  (h_motorcyclist_catch : 1 = (S / 4) / (Vm - Vp))
  (h_motorcyclist_meet_truck : 1.5 = S / (Vm + Vg)) :
  (S / 4 + (12 / 5) * (Vg + Vp)) / (12 / 5) = timeTravelerMeetsTruck := sorry

end traveler_meets_truck_at_15_48_l87_87001


namespace valentines_initial_l87_87374

theorem valentines_initial (gave_away : ℕ) (left_over : ℕ) (initial : ℕ) : 
  gave_away = 8 → left_over = 22 → initial = gave_away + left_over → initial = 30 :=
by
  intros h1 h2 h3
  sorry

end valentines_initial_l87_87374


namespace area_ratio_of_squares_l87_87077

theorem area_ratio_of_squares (s L : ℝ) 
  (H : 4 * L = 4 * 4 * s) : (L^2) = 16 * (s^2) :=
by
  -- assuming the utilization of the given condition
  sorry

end area_ratio_of_squares_l87_87077


namespace product_of_complex_conjugates_l87_87788

theorem product_of_complex_conjugates (i : ℂ) (h : i^2 = -1) : (1 + i) * (1 - i) = 2 :=
by
  sorry

end product_of_complex_conjugates_l87_87788


namespace find_ab_l87_87951

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 := 
by 
  sorry

end find_ab_l87_87951


namespace rational_sum_l87_87262

theorem rational_sum (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : x + y = 7 ∨ x + y = 3 := 
sorry

end rational_sum_l87_87262


namespace smallest_class_size_l87_87399

theorem smallest_class_size :
  ∀ (x : ℕ), 4 * x + 3 > 50 → 4 * x + 3 = 51 :=
by
  sorry

end smallest_class_size_l87_87399


namespace rebecca_charge_for_dye_job_l87_87893

def charges_for_services (haircuts per perms per dye_jobs hair_dye_per_dye_job tips : ℕ) : ℕ := 
  4 * 30 + 1 * 40 + 2 * (dye_jobs - hair_dye_per_dye_job) + tips

theorem rebecca_charge_for_dye_job 
  (haircuts: ℕ) (perms: ℕ) (hair_dye_per_dye_job: ℕ) (tips: ℕ) (end_of_day_amount: ℕ) : 
  haircuts = 4 → perms = 1 → hair_dye_per_dye_job = 10 → tips = 50 → 
  end_of_day_amount = 310 → 
  ∃ D: ℕ, D = 60 := 
by
  sorry

end rebecca_charge_for_dye_job_l87_87893


namespace math_problem_l87_87971

theorem math_problem :
  (-1:ℤ) ^ 2023 - |(-3:ℤ)| + ((-1/3:ℚ) ^ (-2:ℤ)) + ((Real.pi - 3.14)^0) = 6 := 
by 
  sorry

end math_problem_l87_87971


namespace value_of_f_at_5_l87_87910

theorem value_of_f_at_5 (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = - f x) 
  (h_period : ∀ x, f (x + 4) = f x)
  (h_func : ∀ x, -2 ≤ x ∧ x < 0 → f x = 3 * x + 1) : 
  f 5 = 2 :=
  sorry

end value_of_f_at_5_l87_87910


namespace band_formation_max_l87_87953

-- Define the conditions provided in the problem
theorem band_formation_max (m r x : ℕ) (h1 : m = r * x + 5)
  (h2 : (r - 3) * (x + 2) = m) (h3 : m < 100) :
  m = 70 :=
sorry

end band_formation_max_l87_87953


namespace exp_eval_l87_87269

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end exp_eval_l87_87269


namespace prob_no_infection_correct_prob_one_infection_correct_l87_87528

-- Probability that no chicken is infected
def prob_no_infection (p_not_infected : ℚ) (n : ℕ) : ℚ := p_not_infected^n

-- Given
def p_not_infected : ℚ := 4 / 5
def n : ℕ := 5

-- Expected answer for no chicken infected
def expected_prob_no_infection : ℚ := 1024 / 3125

-- Lean statement
theorem prob_no_infection_correct : 
  prob_no_infection p_not_infected n = expected_prob_no_infection := by
  sorry

-- Probability that exactly one chicken is infected
def prob_one_infection (p_infected : ℚ) (p_not_infected : ℚ) (n : ℕ) : ℚ := 
  (n * p_not_infected^(n-1) * p_infected)

-- Given
def p_infected : ℚ := 1 / 5

-- Expected answer for exactly one chicken infected
def expected_prob_one_infection : ℚ := 256 / 625

-- Lean statement
theorem prob_one_infection_correct : 
  prob_one_infection p_infected p_not_infected n = expected_prob_one_infection := by
  sorry

end prob_no_infection_correct_prob_one_infection_correct_l87_87528


namespace probability_of_bug9_is_zero_l87_87864

-- Definitions based on conditions provided
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def non_vowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits_or_vowels : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'E', 'I', 'O', 'U']

-- Defining the number of choices for each position
def first_symbol_choices : Nat := 5
def second_symbol_choices : Nat := 21
def third_symbol_choices : Nat := 20
def fourth_symbol_choices : Nat := 15

-- Total number of possible license plates
def total_plates : Nat := first_symbol_choices * second_symbol_choices * third_symbol_choices * fourth_symbol_choices

-- Probability calculation for the specific license plate "BUG9"
def probability_bug9 : Nat := 0

theorem probability_of_bug9_is_zero : probability_bug9 = 0 := by sorry

end probability_of_bug9_is_zero_l87_87864


namespace ratio_of_girls_who_like_pink_l87_87856

theorem ratio_of_girls_who_like_pink 
  (total_students : ℕ) (answered_green : ℕ) (answered_yellow : ℕ) (total_girls : ℕ) (answered_yellow_students : ℕ)
  (portion_girls_pink : ℕ) 
  (h1 : total_students = 30)
  (h2 : answered_green = total_students / 2)
  (h3 : total_girls = 18)
  (h4 : answered_yellow_students = 9)
  (answered_pink := total_students - answered_green - answered_yellow_students)
  (ratio_pink : ℚ := answered_pink / total_girls) : 
  ratio_pink = 1 / 3 :=
sorry

end ratio_of_girls_who_like_pink_l87_87856


namespace probability_to_buy_ticket_l87_87305

def p : ℝ := 0.1
def q : ℝ := 0.9
def initial_money : ℝ := 20
def target_money : ℝ := 45
def ticket_cost : ℝ := 10
def prize : ℝ := 30

noncomputable def equation_lhs : ℝ := p^2 * (1 + 2 * q)
noncomputable def equation_rhs : ℝ := 1 - 2 * p * q^2

noncomputable def x2 : ℝ := equation_lhs / equation_rhs

theorem probability_to_buy_ticket : x2 = 0.033 := sorry

end probability_to_buy_ticket_l87_87305


namespace nancy_kept_chips_l87_87340

def nancy_initial_chips : ℕ := 22
def chips_given_to_brother : ℕ := 7
def chips_given_to_sister : ℕ := 5

theorem nancy_kept_chips : nancy_initial_chips - (chips_given_to_brother + chips_given_to_sister) = 10 :=
by
  sorry

end nancy_kept_chips_l87_87340


namespace total_legs_l87_87028

theorem total_legs 
  (johnny_legs : ℕ := 2) 
  (son_legs : ℕ := 2) 
  (dog_legs_per_dog : ℕ := 4) 
  (number_of_dogs : ℕ := 2) :
  johnny_legs + son_legs + dog_legs_per_dog * number_of_dogs = 12 := 
sorry

end total_legs_l87_87028


namespace problem_l87_87799

def operation (a b : ℤ) (h : a ≠ 0) : ℤ := (b - a) ^ 2 / a ^ 2

theorem problem : 
  operation (-1) (operation 1 (-1) (by decide)) (by decide) = 25 := 
by
  sorry

end problem_l87_87799


namespace john_reaching_floor_pushups_l87_87628

-- Definitions based on conditions
def john_train_days_per_week : ℕ := 5
def reps_to_progress : ℕ := 20
def variations : ℕ := 3  -- wall, incline, knee

-- Mathematical statement
theorem john_reaching_floor_pushups : 
  (reps_to_progress * variations) / john_train_days_per_week = 12 := 
by
  sorry

end john_reaching_floor_pushups_l87_87628


namespace ant_probability_after_10_minutes_l87_87614

-- Definitions based on the conditions given in the problem
def ant_start_at_A := true
def moves_each_minute (n : ℕ) := n == 10
def blue_dots (x y : ℤ) : Prop := 
  (x == 0 ∨ y == 0) ∧ (x + y) % 2 == 0
def A_at_center (x y : ℤ) : Prop := x == 0 ∧ y == 0
def B_north_of_A (x y : ℤ) : Prop := x == 0 ∧ y == 1

-- The probability we need to prove
def probability_ant_at_B_after_10_minutes := 1 / 9

-- We state our proof problem
theorem ant_probability_after_10_minutes :
  ant_start_at_A ∧ moves_each_minute 10 ∧ blue_dots 0 0 ∧ blue_dots 0 1 ∧ A_at_center 0 0 ∧ B_north_of_A 0 1
  → probability_ant_at_B_after_10_minutes = 1 / 9 := 
sorry

end ant_probability_after_10_minutes_l87_87614


namespace probability_greater_than_4_l87_87849

-- Given conditions
def die_faces : ℕ := 6
def favorable_outcomes : Finset ℕ := {5, 6}

-- Probability calculation
def probability (total : ℕ) (favorable : Finset ℕ) : ℚ :=
  favorable.card / total

theorem probability_greater_than_4 :
  probability die_faces favorable_outcomes = 1 / 3 :=
by
  sorry

end probability_greater_than_4_l87_87849


namespace borrowed_amount_l87_87101

theorem borrowed_amount (P : ℝ) (h1 : (9 / 100) * P - (8 / 100) * P = 200) : P = 20000 :=
  by sorry

end borrowed_amount_l87_87101


namespace probability_two_doors_open_l87_87755

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_doors_open :
  let total_doors := 5
  let total_combinations := 2 ^ total_doors
  let favorable_combinations := binomial total_doors 2
  let probability := favorable_combinations / total_combinations
  probability = 5 / 16 :=
by
  sorry

end probability_two_doors_open_l87_87755


namespace shifted_parabola_sum_constants_l87_87903

theorem shifted_parabola_sum_constants :
  let a := 2
  let b := -17
  let c := 43
  a + b + c = 28 := sorry

end shifted_parabola_sum_constants_l87_87903


namespace muffin_to_banana_ratio_l87_87767

-- Definitions of costs
def elaine_cost (m b : ℝ) : ℝ := 5 * m + 4 * b
def derek_cost (m b : ℝ) : ℝ := 3 * m + 18 * b

-- The problem statement
theorem muffin_to_banana_ratio (m b : ℝ) (h : derek_cost m b = 3 * elaine_cost m b) : m / b = 2 :=
by
  sorry

end muffin_to_banana_ratio_l87_87767


namespace find_sin_expression_l87_87874

noncomputable def trigonometric_identity (γ : ℝ) : Prop :=
  3 * (Real.tan γ)^2 + 3 * (1 / (Real.tan γ))^2 + 2 / (Real.sin γ)^2 + 2 / (Real.cos γ)^2 = 19

theorem find_sin_expression (γ : ℝ) (h : trigonometric_identity γ) : 
  (Real.sin γ)^4 - (Real.sin γ)^2 = -1 / 5 :=
sorry

end find_sin_expression_l87_87874


namespace find_k_l87_87972

def vector := (ℝ × ℝ)

def a : vector := (3, 1)
def b : vector := (1, 3)
def c (k : ℝ) : vector := (k, 2)

def subtract (v1 v2 : vector) : vector :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k (k : ℝ) (h : dot_product (subtract a (c k)) b = 0) : k = 0 := by
  sorry

end find_k_l87_87972


namespace emily_sold_toys_l87_87572

theorem emily_sold_toys (initial_toys : ℕ) (remaining_toys : ℕ) (sold_toys : ℕ) 
  (h_initial : initial_toys = 7) 
  (h_remaining : remaining_toys = 4) 
  (h_sold : sold_toys = initial_toys - remaining_toys) :
  sold_toys = 3 :=
by sorry

end emily_sold_toys_l87_87572


namespace turner_total_tickets_l87_87198

-- Definition of conditions
def days := 3
def rollercoaster_rides_per_day := 3
def catapult_rides_per_day := 2
def ferris_wheel_rides_per_day := 1

def rollercoaster_ticket_cost := 4
def catapult_ticket_cost := 4
def ferris_wheel_ticket_cost := 1

-- Proof statement
theorem turner_total_tickets : 
  days * (rollercoaster_rides_per_day * rollercoaster_ticket_cost 
  + catapult_rides_per_day * catapult_ticket_cost 
  + ferris_wheel_rides_per_day * ferris_wheel_ticket_cost) 
  = 63 := 
by
  sorry

end turner_total_tickets_l87_87198


namespace log_product_l87_87211

theorem log_product : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end log_product_l87_87211


namespace carvings_per_shelf_l87_87549

def total_wood_carvings := 56
def num_shelves := 7

theorem carvings_per_shelf : total_wood_carvings / num_shelves = 8 := by
  sorry

end carvings_per_shelf_l87_87549


namespace A_time_240m_race_l87_87964

theorem A_time_240m_race (t : ℕ) :
  (∀ t, (240 / t) = (184 / t) * (t + 7) ∧ 240 = 184 + ((184 * 7) / t)) → t = 23 :=
by
  sorry

end A_time_240m_race_l87_87964


namespace least_prime_P_with_integer_roots_of_quadratic_l87_87276

theorem least_prime_P_with_integer_roots_of_quadratic :
  ∃ P : ℕ, P.Prime ∧ (∃ m : ℤ,  m^2 = 12 * P + 60) ∧ P = 7 :=
by
  sorry

end least_prime_P_with_integer_roots_of_quadratic_l87_87276


namespace range_of_a_l87_87209

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a ≥ 0) → a ≥ 4 :=
by
  sorry

end range_of_a_l87_87209


namespace simplify_fraction_l87_87714

theorem simplify_fraction : (45 / (7 - 3 / 4)) = (36 / 5) :=
by
  sorry

end simplify_fraction_l87_87714


namespace integer_triplet_solution_l87_87139

def circ (a b : ℤ) : ℤ := a + b - a * b

theorem integer_triplet_solution (x y z : ℤ) :
  circ (circ x y) z + circ (circ y z) x + circ (circ z x) y = 0 ↔
  (x = 0 ∧ y = 0 ∧ z = 2) ∨ (x = 0 ∧ y = 2 ∧ z = 0) ∨ (x = 2 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end integer_triplet_solution_l87_87139


namespace mixed_operations_with_decimals_false_l87_87973

-- Definitions and conditions
def operations_same_level_with_decimals : Prop :=
  ∀ (a b c : ℝ), a + b - c = (a + b) - c

def calculate_left_to_right_with_decimals : Prop :=
  ∀ (a b c : ℝ), (a - b + c) = a - b + c ∧ (a + b - c) = a + b - c

-- Proposition we're proving
theorem mixed_operations_with_decimals_false :
  ¬ ∀ (a b c : ℝ), (a + b - c) ≠ (a - b + c) :=
by
  intro h
  sorry

end mixed_operations_with_decimals_false_l87_87973


namespace shadow_projection_height_l87_87993

theorem shadow_projection_height :
  ∃ (x : ℝ), (∃ (shadow_area : ℝ), shadow_area = 192) ∧ 1000 * x = 25780 :=
by
  sorry

end shadow_projection_height_l87_87993


namespace vince_bus_ride_distance_l87_87573

/-- 
  Vince's bus ride to school is 0.625 mile, 
  given that Zachary's bus ride is 0.5 mile 
  and Vince's bus ride is 0.125 mile longer than Zachary's.
--/
theorem vince_bus_ride_distance (zachary_ride : ℝ) (vince_longer : ℝ) 
  (h1 : zachary_ride = 0.5) (h2 : vince_longer = 0.125) 
  : zachary_ride + vince_longer = 0.625 :=
by sorry

end vince_bus_ride_distance_l87_87573


namespace sum_cubes_eq_neg_27_l87_87063

theorem sum_cubes_eq_neg_27 (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
 (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
 a^3 + b^3 + c^3 = -27 :=
by
  sorry

end sum_cubes_eq_neg_27_l87_87063


namespace B_knit_time_l87_87625

theorem B_knit_time (x : ℕ) (hA : 3 > 0) (h_combined_rate : 1/3 + 1/x = 1/2) : x = 6 := sorry

end B_knit_time_l87_87625


namespace translation_preserves_parallel_and_equal_length_l87_87301

theorem translation_preserves_parallel_and_equal_length
    (A B C D : ℝ)
    (after_translation : (C - A) = (D - B))
    (connecting_parallel : C - A = D - B) :
    (C - A = D - B) ∧ (C - A = D - B) :=
by
  sorry

end translation_preserves_parallel_and_equal_length_l87_87301


namespace cos_45_degree_l87_87751

theorem cos_45_degree : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by
  sorry

end cos_45_degree_l87_87751


namespace arithmetic_seq_a7_a8_l87_87254

theorem arithmetic_seq_a7_a8 (a : ℕ → ℤ) (d : ℤ) (h₁ : a 1 + a 2 = 4) (h₂ : d = 2) :
  a 7 + a 8 = 28 := by
  sorry

end arithmetic_seq_a7_a8_l87_87254


namespace dima_walking_speed_l87_87074

def Dima_station_time := 18 * 60 -- in minutes
def Dima_actual_arrival := 17 * 60 + 5 -- in minutes
def car_speed := 60 -- in km/h
def early_arrival := 10 -- in minutes

def walking_speed (arrival_time actual_arrival car_speed early_arrival : ℕ) : ℕ :=
(car_speed * early_arrival / 60) * (60 / (arrival_time - actual_arrival - early_arrival))

theorem dima_walking_speed :
  walking_speed Dima_station_time Dima_actual_arrival car_speed early_arrival = 6 :=
sorry

end dima_walking_speed_l87_87074


namespace triangle_non_existent_l87_87222

theorem triangle_non_existent (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (tangent_condition : (c^2) = 2 * (a^2) + 2 * (b^2)) : False := by
  sorry

end triangle_non_existent_l87_87222


namespace range_of_m_l87_87394

noncomputable def y (m x : ℝ) := m * (1/4)^x - (1/2)^x + 1

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, y m x = 0) → (m ≤ 0 ∨ m = 1 / 4) := sorry

end range_of_m_l87_87394


namespace steve_average_speed_l87_87540

/-
Problem Statement:
Prove that the average speed of Steve's travel for the entire journey is 55 mph given the following conditions:
1. Steve's first part of journey: 5 hours at 40 mph.
2. Steve's second part of journey: 3 hours at 80 mph.
-/

theorem steve_average_speed :
  let time1 := 5 -- hours
  let speed1 := 40 -- mph
  let time2 := 3 -- hours
  let speed2 := 80 -- mph
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 55 := by
  sorry

end steve_average_speed_l87_87540


namespace total_votes_cast_l87_87140

theorem total_votes_cast (V: ℕ) (invalid_votes: ℕ) (diff_votes: ℕ) 
  (H1: invalid_votes = 200) 
  (H2: diff_votes = 700) 
  (H3: (0.01 : ℝ) * V = diff_votes) 
  : (V + invalid_votes = 70200) :=
by
  sorry

end total_votes_cast_l87_87140


namespace fraction_books_left_l87_87699

theorem fraction_books_left (initial_books sold_books remaining_books : ℕ)
  (h1 : initial_books = 9900) (h2 : sold_books = 3300) (h3 : remaining_books = initial_books - sold_books) :
  (remaining_books : ℚ) / initial_books = 2 / 3 :=
by
  sorry

end fraction_books_left_l87_87699


namespace petya_digits_l87_87225

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l87_87225


namespace pen_cost_price_l87_87017

-- Define the variables and assumptions
variable (x : ℝ)

-- Given conditions
def profit_one_pen (x : ℝ) := 10 - x
def profit_three_pens (x : ℝ) := 20 - 3 * x

-- Statement to prove
theorem pen_cost_price : profit_one_pen x = profit_three_pens x → x = 5 :=
by
  sorry

end pen_cost_price_l87_87017


namespace polygon_perimeter_is_35_l87_87620

-- Define the concept of a regular polygon with given side length and exterior angle
def regular_polygon_perimeter (n : ℕ) (side_length : ℕ) : ℕ := 
  n * side_length

theorem polygon_perimeter_is_35 (side_length : ℕ) (exterior_angle : ℕ) (n : ℕ)
  (h1 : side_length = 7) (h2 : exterior_angle = 72) (h3 : 360 / exterior_angle = n) :
  regular_polygon_perimeter n side_length = 35 :=
by
  -- We skip the proof body as only the statement is required
  sorry

end polygon_perimeter_is_35_l87_87620


namespace simplify_fraction_l87_87347

variable (k : ℤ)

theorem simplify_fraction (a b : ℤ)
  (hk : a = 2)
  (hb : b = 4) :
  (6 * k + 12) / 3 = 2 * k + 4 ∧ (a : ℚ) / (b : ℚ) = 1 / 2 := 
by
  sorry

end simplify_fraction_l87_87347


namespace secretary_longest_time_l87_87945

def ratio_times (x : ℕ) : Prop := 
  let t1 := 2 * x
  let t2 := 3 * x
  let t3 := 5 * x
  (t1 + t2 + t3 = 110) ∧ (t3 = 55)

theorem secretary_longest_time :
  ∃ x : ℕ, ratio_times x :=
sorry

end secretary_longest_time_l87_87945


namespace find_m_l87_87426

noncomputable def given_hyperbola (x y : ℝ) (m : ℝ) : Prop :=
    x^2 / m - y^2 / 3 = 1

noncomputable def hyperbola_eccentricity (m : ℝ) (e : ℝ) : Prop :=
    e = Real.sqrt (1 + 3 / m)

theorem find_m (m : ℝ) (h1 : given_hyperbola 1 1 m) (h2 : hyperbola_eccentricity m 2) : m = 1 :=
by
  sorry

end find_m_l87_87426


namespace work_rate_l87_87998

theorem work_rate (x : ℝ) (h : (1 / x + 1 / 15 = 1 / 6)) : x = 10 :=
sorry

end work_rate_l87_87998


namespace remainder_of_n_l87_87648

theorem remainder_of_n (n : ℕ) (h1 : n^2 ≡ 9 [MOD 11]) (h2 : n^3 ≡ 5 [MOD 11]) : n ≡ 3 [MOD 11] :=
sorry

end remainder_of_n_l87_87648


namespace mary_characters_initials_l87_87092

theorem mary_characters_initials :
  ∀ (total_A total_C total_D total_E : ℕ),
  total_A = 60 / 2 →
  total_C = total_A / 2 →
  total_D = 2 * total_E →
  total_A + total_C + total_D + total_E = 60 →
  total_D = 10 :=
by
  intros total_A total_C total_D total_E hA hC hDE hSum
  sorry

end mary_characters_initials_l87_87092


namespace ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l87_87565

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 / 2 = 1

def point_on_x_axis (P : ℝ × ℝ) : Prop :=
  P.snd = 0

def angles_complementary (P A B : ℝ × ℝ) : Prop :=
  let kPA := (A.snd - P.snd) / (A.fst - P.fst)
  let kPB := (B.snd - P.snd) / (B.fst - P.fst)
  kPA + kPB = 0

theorem ellipse_equation_hyperbola_vertices_and_foci :
  (∀ x y : ℝ, hyperbola_eq x y → ellipse_eq x y) :=
sorry

theorem exists_point_P_on_x_axis_angles_complementary (F2 A B : ℝ × ℝ) :
  F2 = (1, 0) → (∃ P : ℝ × ℝ, point_on_x_axis P ∧ angles_complementary P A B) :=
sorry

end ellipse_equation_hyperbola_vertices_and_foci_exists_point_P_on_x_axis_angles_complementary_l87_87565


namespace cheryl_material_left_l87_87358

-- Conditions
def initial_material_type1 (m1 : ℚ) : Prop := m1 = 2/9
def initial_material_type2 (m2 : ℚ) : Prop := m2 = 1/8
def used_material (u : ℚ) : Prop := u = 0.125

-- Define the total material bought
def total_material (m1 m2 : ℚ) : ℚ := m1 + m2

-- Define the material left
def material_left (t u : ℚ) : ℚ := t - u

-- The target theorem
theorem cheryl_material_left (m1 m2 u : ℚ) 
  (h1 : initial_material_type1 m1)
  (h2 : initial_material_type2 m2)
  (h3 : used_material u) : 
  material_left (total_material m1 m2) u = 2/9 :=
by
  sorry

end cheryl_material_left_l87_87358


namespace union_of_A_and_B_l87_87591

def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > 1} := 
by 
  sorry

end union_of_A_and_B_l87_87591


namespace restocked_bags_correct_l87_87192

def initial_stock := 55
def sold_bags := 23
def final_stock := 164

theorem restocked_bags_correct :
  (final_stock - (initial_stock - sold_bags)) = 132 :=
by
  -- The proof would go here, but we use sorry to skip it.
  sorry

end restocked_bags_correct_l87_87192


namespace problem_statement_l87_87984

theorem problem_statement (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
sorry

end problem_statement_l87_87984


namespace arithmetic_geometric_inequality_l87_87349

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) := 
sorry

end arithmetic_geometric_inequality_l87_87349


namespace find_a_l87_87860

noncomputable def A : Set ℝ := {1, 2, 3, 4}
noncomputable def B (a : ℝ) : Set ℝ := { x | x ≤ a }

theorem find_a (a : ℝ) (h_union : A ∪ B a = Set.Iic 5) : a = 5 := by
  sorry

end find_a_l87_87860


namespace sqrt_49_times_sqrt_25_eq_7sqrt5_l87_87991

theorem sqrt_49_times_sqrt_25_eq_7sqrt5 :
  (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  have h1 : Real.sqrt 25 = 5 := by sorry
  have h2 : 49 * 5 = 245 := by sorry
  have h3 : Real.sqrt 245 = 7 * Real.sqrt 5 := by sorry
  sorry

end sqrt_49_times_sqrt_25_eq_7sqrt5_l87_87991


namespace cubic_no_maximum_value_l87_87931

theorem cubic_no_maximum_value (x : ℝ) : ¬ ∃ M, ∀ x : ℝ, 3 * x^2 + 6 * x^3 + 27 * x + 100 ≤ M := 
by
  sorry

end cubic_no_maximum_value_l87_87931


namespace houses_with_white_mailboxes_l87_87881

theorem houses_with_white_mailboxes (total_mail : ℕ) (total_houses : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ)
    (h1 : total_mail = 48) (h2 : total_houses = 8) (h3 : red_mailboxes = 3) (h4 : mail_per_house = 6) :
  total_houses - red_mailboxes = 5 :=
by
  sorry

end houses_with_white_mailboxes_l87_87881


namespace ad_value_l87_87764

variable (a b c d : ℝ)

-- Conditions
def geom_seq := b^2 = a * c ∧ c^2 = b * d
def vertex_of_parabola := (b = 1 ∧ c = 2)

-- Question
theorem ad_value (h_geom : geom_seq a b c d) (h_vertex : vertex_of_parabola b c) : a * d = 2 := by
  sorry

end ad_value_l87_87764


namespace lengths_of_triangle_sides_l87_87526

open Real

noncomputable def triangle_side_lengths (a b c : ℝ) (A B C : ℝ) :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ A + B + C = π ∧ A = 60 * π / 180 ∧
  10 * sqrt 3 = 0.5 * a * b * sin A ∧
  a + b = 13 ∧
  c = sqrt (a^2 + b^2 - 2 * a * b * cos A)

theorem lengths_of_triangle_sides
  (a b c : ℝ) (A B C : ℝ)
  (h : triangle_side_lengths a b c A B C) :
  (a = 5 ∧ b = 8 ∧ c = 7) ∨ (a = 8 ∧ b = 5 ∧ c = 7) :=
sorry

end lengths_of_triangle_sides_l87_87526


namespace shares_proportion_l87_87700

theorem shares_proportion (C D : ℕ) (h1 : D = 1500) (h2 : C = D + 500) : C / Nat.gcd C D = 4 ∧ D / Nat.gcd C D = 3 := by
  sorry

end shares_proportion_l87_87700


namespace total_units_per_day_all_work_together_l87_87753

-- Conditions
def men := 250
def women := 150
def units_per_day_by_men := 15
def units_per_day_by_women := 3

-- Problem statement and proof
theorem total_units_per_day_all_work_together :
  units_per_day_by_men + units_per_day_by_women = 18 :=
sorry

end total_units_per_day_all_work_together_l87_87753


namespace rational_number_property_l87_87740

theorem rational_number_property 
  (x : ℚ) (a : ℤ) (ha : 1 ≤ a) : 
  (x ^ (⌊x⌋)) = a / 2 → (∃ k : ℤ, x = k) ∨ x = 3 / 2 :=
by
  sorry

end rational_number_property_l87_87740


namespace a_cubed_divisible_l87_87446

theorem a_cubed_divisible {a : ℤ} (h1 : 60 ≤ a) (h2 : a^3 ∣ 216000) : a = 60 :=
by {
   sorry
}

end a_cubed_divisible_l87_87446


namespace inequality_proof_l87_87265

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ 1 / 2 * (a + b + c) := 
by
  sorry

end inequality_proof_l87_87265


namespace peter_total_books_is_20_l87_87661

noncomputable def total_books_peter_has (B : ℝ) : Prop :=
  let Peter_Books_Read := 0.40 * B
  let Brother_Books_Read := 0.10 * B
  Peter_Books_Read = Brother_Books_Read + 6

theorem peter_total_books_is_20 :
  ∃ B : ℝ, total_books_peter_has B ∧ B = 20 := 
by
  sorry

end peter_total_books_is_20_l87_87661


namespace AM_GM_inequality_equality_case_of_AM_GM_l87_87600

theorem AM_GM_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (x / y) + (y / x) ≥ 2 :=
by
  sorry

theorem equality_case_of_AM_GM (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : ((x / y) + (y / x) = 2) ↔ (x = y) :=
by
  sorry

end AM_GM_inequality_equality_case_of_AM_GM_l87_87600


namespace find_second_divisor_l87_87773

theorem find_second_divisor :
  ∃ x : ℕ, 377 / 13 / x * (1/4 : ℚ) / 2 = 0.125 ∧ x = 29 :=
by
  use 29
  -- Proof steps would go here
  sorry

end find_second_divisor_l87_87773


namespace sequence_a_2024_l87_87049

theorem sequence_a_2024 (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 1 - 1 / a n) : a 2024 = 1 / 2 :=
by
  sorry

end sequence_a_2024_l87_87049


namespace wire_problem_l87_87392

theorem wire_problem (a b : ℝ) (h_perimeter : a = b) : a / b = 1 := by
  sorry

end wire_problem_l87_87392


namespace initial_candies_l87_87547

-- Define initial variables and conditions
variable (x : ℕ)
variable (remaining_candies_after_first_day : ℕ)
variable (remaining_candies_after_second_day : ℕ)

-- Conditions as per given problem
def condition1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
def condition2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
def final_condition : remaining_candies_after_second_day = 10 := sorry

-- Goal: Prove that initially, Liam had 52 candies
theorem initial_candies : x = 52 := by
  have h1 : remaining_candies_after_first_day = (3 * x / 4) - 3 := sorry
  have h2 : remaining_candies_after_second_day = (3 * remaining_candies_after_first_day / 20) - 5 := sorry
  have h3 : remaining_candies_after_second_day = 10 := sorry
    
  -- Combine conditions to solve for x
  sorry

end initial_candies_l87_87547


namespace num_students_earning_B_l87_87570

open Real

theorem num_students_earning_B (total_students : ℝ) (pA : ℝ) (pB : ℝ) (pC : ℝ) (students_A : ℝ) (students_B : ℝ) (students_C : ℝ) :
  total_students = 31 →
  pA = 0.7 * pB →
  pC = 1.4 * pB →
  students_A = 0.7 * students_B →
  students_C = 1.4 * students_B →
  students_A + students_B + students_C = total_students →
  students_B = 10 :=
by
  intros h_total_students h_pa h_pc h_students_A h_students_C h_total_eq
  sorry

end num_students_earning_B_l87_87570


namespace parabola_intercepts_sum_l87_87627

theorem parabola_intercepts_sum (a b c : ℝ)
  (h₁ : a = 5)
  (h₂ : b = (9 + Real.sqrt 21) / 6)
  (h₃ : c = (9 - Real.sqrt 21) / 6) :
  a + b + c = 8 :=
by
  sorry

end parabola_intercepts_sum_l87_87627


namespace problem_solution_l87_87119

theorem problem_solution (k : ℤ) : k ≤ 0 ∧ -2 < k → k = -1 ∨ k = 0 :=
by
  sorry

end problem_solution_l87_87119


namespace length_CK_angle_BCA_l87_87432

variables {A B C O O₁ O₂ K K₁ K₂ K₃ : Point}
variables {r R : ℝ}
variables {AC CK AK₁ AK₂ : ℝ}

-- Definitions and conditions
def triangle_ABC (A B C : Point) : Prop := True
def incenter (A B C O : Point) : Prop := True
def in_radius_is_equal (O₁ O₂ : Point) (r : ℝ) : Prop := True
def circle_touches_side (circle_center : Point) (side_point : Point) (distance : ℝ) : Prop := True
def circumcenter (A C B O₁ : Point) : Prop := True
def angle (A B C : Point) (θ : ℝ) : Prop := True

-- Conditions from the problem
axiom cond1 : triangle_ABC A B C
axiom cond2 : in_radius_is_equal O₁ O₂ r
axiom cond3 : incenter A B C O
axiom cond4 : circle_touches_side O₁ K₁ 6
axiom cond5 : circle_touches_side O₂ K₂ 8
axiom cond6 : AC = 21
axiom cond7 : circle_touches_side O K 9
axiom cond8 : circumcenter O K₁ K₃ O₁

-- Statements to prove
theorem length_CK : CK = 9 := by
  sorry

theorem angle_BCA : angle B C A 60 := by
  sorry

end length_CK_angle_BCA_l87_87432


namespace solution_set_of_inequality_l87_87235

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) : 
  ((2 * x) / (x - 2) ≤ 1) ↔ (-2 ≤ x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l87_87235


namespace students_speak_both_l87_87607

theorem students_speak_both (total E T N : ℕ) (h1 : total = 150) (h2 : E = 55) (h3 : T = 85) (h4 : N = 30) :
  E + T - (total - N) = 20 := by
  -- Main proof logic
  sorry

end students_speak_both_l87_87607


namespace train_passing_time_l87_87482

theorem train_passing_time :
  ∀ (length : ℕ) (speed_kmph : ℕ),
    length = 120 →
    speed_kmph = 72 →
    ∃ (time : ℕ), time = 6 :=
by
  intro length speed_kmph hlength hspeed_kmph
  sorry

end train_passing_time_l87_87482


namespace f_is_odd_f_is_monotone_l87_87371

noncomputable def f (k x : ℝ) : ℝ := x + k / x

-- Proving f(x) is odd
theorem f_is_odd (k : ℝ) (hk : k ≠ 0) : ∀ x : ℝ, f k (-x) = -f k x :=
by
  intro x
  sorry

-- Proving f(x) is monotonically increasing on [sqrt(k), +∞) for k > 0
theorem f_is_monotone (k : ℝ) (hk : k > 0) : ∀ x1 x2 : ℝ, 
  x1 ∈ Set.Ici (Real.sqrt k) → x2 ∈ Set.Ici (Real.sqrt k) → x1 < x2 → f k x1 < f k x2 :=
by
  intro x1 x2 hx1 hx2 hlt
  sorry

end f_is_odd_f_is_monotone_l87_87371


namespace correct_subtraction_l87_87905

theorem correct_subtraction (x : ℕ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end correct_subtraction_l87_87905


namespace magnitude_2a_minus_b_l87_87029

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (θ : ℝ) (h_angle : θ = 5 * Real.pi / 6)
variables (h_mag_a : ‖a‖ = 4) (h_mag_b : ‖b‖ = Real.sqrt 3)

theorem magnitude_2a_minus_b :
  ‖2 • a - b‖ = Real.sqrt 91 := by
  -- Proof goes here.
  sorry

end magnitude_2a_minus_b_l87_87029


namespace max_divisor_f_l87_87596

-- Given definition
def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

-- Main theorem to be proved
theorem max_divisor_f :
  ∃ m : ℕ, (∀ n : ℕ, 0 < n → m ∣ f n) ∧ m = 36 :=
by
  -- The proof would go here
  sorry

end max_divisor_f_l87_87596


namespace difference_of_squares_65_35_l87_87585

theorem difference_of_squares_65_35 : 65^2 - 35^2 = 3000 := 
  sorry

end difference_of_squares_65_35_l87_87585


namespace sum_of_roots_eq_neg2_l87_87048

-- Define the quadratic equation.
def quadratic_equation (x : ℝ) : ℝ :=
  x^2 + 2 * x - 1

-- Define a predicate to express that x is a root of the quadratic equation.
def is_root (x : ℝ) : Prop :=
  quadratic_equation x = 0

-- Define the statement that the sum of the two roots of the quadratic equation equals -2.
theorem sum_of_roots_eq_neg2 (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) (h3 : x1 ≠ x2) :
  x1 + x2 = -2 :=
  sorry

end sum_of_roots_eq_neg2_l87_87048


namespace special_op_equality_l87_87831

def special_op (x y : ℕ) : ℕ := x * y - x - 2 * y

theorem special_op_equality : (special_op 7 4) - (special_op 4 7) = 3 := by
  sorry

end special_op_equality_l87_87831


namespace evaluate_expression_l87_87155

theorem evaluate_expression : 
  (-2 : ℤ)^2004 + 3 * (-2)^2003 = (-2)^2003 :=
by
  sorry

end evaluate_expression_l87_87155


namespace julia_fourth_day_candies_l87_87843

-- Definitions based on conditions
def first_day (x : ℚ) := (1/5) * x
def second_day (x : ℚ) := (1/2) * (4/5) * x
def third_day (x : ℚ) := (1/2) * (2/5) * x
def fourth_day (x : ℚ) := (2/5) * x - (1/2) * (2/5) * x

-- The Lean statement to prove
theorem julia_fourth_day_candies (x : ℚ) (h : x ≠ 0): 
  fourth_day x / x = 1/5 :=
by
  -- insert proof here
  sorry

end julia_fourth_day_candies_l87_87843


namespace minor_premise_wrong_l87_87026

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 + x

theorem minor_premise_wrong : ¬ is_even_function f ∧ ¬ is_odd_function f := 
by
  sorry

end minor_premise_wrong_l87_87026


namespace total_price_for_pizza_l87_87061

-- Definitions based on conditions
def num_friends : ℕ := 5
def amount_per_person : ℕ := 8

-- The claim to be proven
theorem total_price_for_pizza : num_friends * amount_per_person = 40 := by
  -- Since the proof detail is not required, we use 'sorry' to skip the proof.
  sorry

end total_price_for_pizza_l87_87061


namespace entrance_sum_2_to_3_pm_exit_sum_2_to_3_pm_no_crowd_control_at_4_pm_l87_87086

noncomputable def f : ℕ → ℕ
| n => if 1 ≤ n ∧ n ≤ 8 then 200 * n + 2000
       else if 9 ≤ n ∧ n ≤ 32 then 360 * (3 ^ ((n - 8) / 12)) + 3000
       else if 33 ≤ n ∧ n ≤ 45 then 32400 - 720 * n
       else 0

noncomputable def g : ℕ → ℕ
| n => if 1 ≤ n ∧ n ≤ 18 then 0
       else if 19 ≤ n ∧ n ≤ 32 then 500 * n - 9000
       else if 33 ≤ n ∧ n ≤ 45 then 8800
       else 0

theorem entrance_sum_2_to_3_pm : f 21 + f 22 + f 23 + f 24 = 17460 := by
  sorry

theorem exit_sum_2_to_3_pm : g 21 + g 22 + g 23 + g 24 = 9000 := by
  sorry

theorem no_crowd_control_at_4_pm : f 28 - g 28 < 80000 := by
  sorry

end entrance_sum_2_to_3_pm_exit_sum_2_to_3_pm_no_crowd_control_at_4_pm_l87_87086


namespace region_transformation_area_l87_87064

-- Define the region T with area 15
def region_T : ℝ := 15

-- Define the transformation matrix
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![ 3, 4 ],
  ![ 5, -2 ]
]

-- The determinant of the matrix
def det_matrix_M : ℝ := 3 * (-2) - 4 * 5

-- The proven target statement to show that after the transformation, the area of T' is 390
theorem region_transformation_area :
  ∃ (area_T' : ℝ), area_T' = |det_matrix_M| * region_T ∧ area_T' = 390 :=
by
  sorry

end region_transformation_area_l87_87064


namespace necessary_but_not_sufficient_condition_l87_87817

def represents_ellipse (k : ℝ) (x y : ℝ) :=
    1 < k ∧ k < 5 ∧ k ≠ 3

theorem necessary_but_not_sufficient_condition (k : ℝ) (x y : ℝ):
    (1 < k ∧ k < 5) → (represents_ellipse k x y) :=
by
  sorry

end necessary_but_not_sufficient_condition_l87_87817


namespace sum_of_multiples_of_6_and_9_is_multiple_of_3_l87_87950

theorem sum_of_multiples_of_6_and_9_is_multiple_of_3 
  (x y : ℤ) (hx : ∃ m : ℤ, x = 6 * m) (hy : ∃ n : ℤ, y = 9 * n) : 
  ∃ k : ℤ, x + y = 3 * k := 
by 
  sorry

end sum_of_multiples_of_6_and_9_is_multiple_of_3_l87_87950


namespace cyclists_no_point_b_l87_87906

theorem cyclists_no_point_b (v1 v2 t d : ℝ) (h1 : v1 = 35) (h2 : v2 = 25) (h3 : t = 2) (h4 : d = 30) :
  ∀ (ta tb : ℝ), ta + tb = t ∧ ta * v1 + tb * v2 < d → false :=
by
  sorry

end cyclists_no_point_b_l87_87906


namespace frog_hops_ratio_l87_87919

theorem frog_hops_ratio :
  ∀ (F1 F2 F3 : ℕ),
    F1 = 4 * F2 →
    F1 + F2 + F3 = 99 →
    F2 = 18 →
    (F2 : ℚ) / (F3 : ℚ) = 2 :=
by
  intros F1 F2 F3 h1 h2 h3
  -- algebraic manipulations and proof to be filled here
  sorry

end frog_hops_ratio_l87_87919


namespace range_of_b_plus_c_l87_87378

noncomputable def func (b c x : ℝ) : ℝ := x^2 + b*x + c * 3^x

theorem range_of_b_plus_c {b c : ℝ} (h1 : ∃ x, func b c x = 0)
  (h2 : ∀ x, (func b c x = 0 ↔ func b c (func b c x) = 0)) : 
  0 ≤ b + c ∧ b + c < 4 :=
by
  sorry

end range_of_b_plus_c_l87_87378


namespace horner_operations_count_l87_87076

def polynomial (x : ℝ) : ℝ := 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1

def horner_polynomial (x : ℝ) := (((((5*x + 4)*x + 3)*x + 2)*x + 1)*x + 1)

theorem horner_operations_count (x : ℝ) : 
    (polynomial x = horner_polynomial x) → 
    (x = 2) → 
    (mul_ops : ℕ) = 5 → 
    (add_ops : ℕ) = 5 := 
by 
  sorry

end horner_operations_count_l87_87076


namespace cubic_polynomial_has_three_real_roots_l87_87986

open Polynomial

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry
noncomputable def R : Polynomial ℝ := sorry

axiom P_degree : degree P = 2
axiom Q_degree : degree Q = 3
axiom R_degree : degree R = 3
axiom PQR_relationship : ∀ x : ℝ, P.eval x ^ 2 + Q.eval x ^ 2 = R.eval x ^ 2

theorem cubic_polynomial_has_three_real_roots : 
  (∃ x : ℝ, Q.eval x = 0 ∧ ∃ y : ℝ, Q.eval y = 0 ∧ ∃ z : ℝ, Q.eval z = 0) ∨
  (∃ x : ℝ, R.eval x = 0 ∧ ∃ y : ℝ, R.eval y = 0 ∧ ∃ z : ℝ, R.eval z = 0) :=
sorry

end cubic_polynomial_has_three_real_roots_l87_87986


namespace signup_ways_l87_87136

theorem signup_ways (students groups : ℕ) (h_students : students = 5) (h_groups : groups = 3) :
  (groups ^ students = 243) :=
by
  have calculation : 3 ^ 5 = 243 := by norm_num
  rwa [h_students, h_groups]

end signup_ways_l87_87136


namespace correct_meteor_passing_time_l87_87395

theorem correct_meteor_passing_time :
  let T1 := 7
  let T2 := 13
  let harmonic_mean := (2 * T1 * T2) / (T1 + T2)
  harmonic_mean = 9.1 := 
by
  sorry

end correct_meteor_passing_time_l87_87395


namespace sum_of_factors_eq_l87_87711

theorem sum_of_factors_eq :
  ∃ (d e f : ℤ), (∀ (x : ℤ), x^2 + 21 * x + 110 = (x + d) * (x + e)) ∧
                 (∀ (x : ℤ), x^2 - 19 * x + 88 = (x - e) * (x - f)) ∧
                 (d + e + f = 30) :=
sorry

end sum_of_factors_eq_l87_87711


namespace bottles_left_on_shelf_l87_87847

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_buys : ℕ) (harry_buys : ℕ) (total_buys : ℕ) (remaining_bottles : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_buys = 5)
  (h3 : harry_buys = 6)
  (h4 : total_buys = jason_buys + harry_buys)
  (h5 : remaining_bottles = initial_bottles - total_buys)
  : remaining_bottles = 24 :=
by
  -- Proof goes here
  sorry

end bottles_left_on_shelf_l87_87847


namespace pure_ghee_percentage_l87_87401

theorem pure_ghee_percentage (Q : ℝ) (P : ℝ) (H1 : Q = 10) (H2 : (P / 100) * Q + 10 = 0.80 * (Q + 10)) :
  P = 60 :=
sorry

end pure_ghee_percentage_l87_87401


namespace ultramindmaster_secret_codes_count_l87_87089

/-- 
In the game UltraMindmaster, we need to find the total number of possible secret codes 
formed by placing pegs of any of eight different colors into five slots.
Colors may be repeated, and each slot must be filled.
-/
theorem ultramindmaster_secret_codes_count :
  let colors := 8
  let slots := 5
  colors ^ slots = 32768 := by
    sorry

end ultramindmaster_secret_codes_count_l87_87089


namespace cloth_sold_l87_87939

theorem cloth_sold (total_sell_price : ℤ) (loss_per_meter : ℤ) (cost_price_per_meter : ℤ) (x : ℤ) 
    (h1 : total_sell_price = 18000) 
    (h2 : loss_per_meter = 5) 
    (h3 : cost_price_per_meter = 50) 
    (h4 : (cost_price_per_meter - loss_per_meter) * x = total_sell_price) : 
    x = 400 :=
by
  sorry

end cloth_sold_l87_87939


namespace sum_of_4_corners_is_200_l87_87462

-- Define the conditions: 9x9 grid, numbers start from 10, and filled sequentially from left to right and top to bottom.
def topLeftCorner : ℕ := 10
def topRightCorner : ℕ := 18
def bottomLeftCorner : ℕ := 82
def bottomRightCorner : ℕ := 90

-- The main theorem stating that the sum of the numbers in the four corners is 200.
theorem sum_of_4_corners_is_200 :
  topLeftCorner + topRightCorner + bottomLeftCorner + bottomRightCorner = 200 :=
by
  -- Placeholder for proof
  sorry

end sum_of_4_corners_is_200_l87_87462


namespace range_x1_x2_l87_87124

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_x1_x2 (a b c d x1 x2 : ℝ)
  (h1 : a ≠ 0)
  (h2 : a + 2 * b + 3 * c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hx1 : f a b c x1 = 0)
  (hx2 : f a b c x2 = 0) :
  abs (x1 - x2) ∈ Set.Ico 0 (2 / 3) :=
sorry

end range_x1_x2_l87_87124


namespace value_of_a_plus_b_l87_87050

variable (a b : ℝ)
variable (h1 : |a| = 5)
variable (h2 : |b| = 2)
variable (h3 : a < 0)
variable (h4 : b > 0)

theorem value_of_a_plus_b : a + b = -3 :=
by
  sorry

end value_of_a_plus_b_l87_87050


namespace isosceles_triangle_formed_by_lines_l87_87343

theorem isosceles_triangle_formed_by_lines :
  let P1 := (1/4, 4)
  let P2 := (-3/2, -3)
  let P3 := (2, -3)
  let d12 := ((1/4 + 3/2)^2 + (4 + 3)^2)
  let d13 := ((1/4 - 2)^2 + (4 + 3)^2)
  let d23 := ((-3/2 - 2)^2)
  (d12 = d13) ∧ (d12 ≠ d23) → 
  ∃ (A B C : ℝ × ℝ), 
    A = P1 ∧ B = P2 ∧ C = P3 ∧ 
    ((dist A B = dist A C) ∧ (dist B C ≠ dist A B)) :=
by
  sorry

end isosceles_triangle_formed_by_lines_l87_87343


namespace license_plate_combinations_l87_87272

def number_of_license_plates : ℕ :=
  10^5 * 26^3 * 20

theorem license_plate_combinations :
  number_of_license_plates = 35152000000 := by
  -- Here's where the proof would go
  sorry

end license_plate_combinations_l87_87272


namespace percent_problem_l87_87796

theorem percent_problem (x y z w : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.40 * z) 
  (h3 : z = 0.70 * w) : 
  x = 0.336 * w :=
sorry

end percent_problem_l87_87796


namespace find_b_l87_87344

variable (p q r b : ℤ)

-- Conditions
def condition1 : Prop := p - q = 2
def condition2 : Prop := p - r = 1

-- The main statement to prove
def problem_statement : Prop :=
  b = (r - q) * ((p - q)^2 + (p - q) * (p - r) + (p - r)^2) → b = 7

theorem find_b (h1 : condition1 p q) (h2 : condition2 p r) (h3 : problem_statement p q r b) : b = 7 :=
sorry

end find_b_l87_87344


namespace contrapositive_of_ab_eq_zero_l87_87760

theorem contrapositive_of_ab_eq_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → ab ≠ 0 :=
by
  sorry

end contrapositive_of_ab_eq_zero_l87_87760


namespace perimeter_of_quadrilateral_eq_fifty_l87_87605

theorem perimeter_of_quadrilateral_eq_fifty
  (a b : ℝ)
  (h1 : a = 10)
  (h2 : b = 15)
  (h3 : ∀ (p q r s : ℝ), p + q = r + s) : 
  2 * a + 2 * b = 50 := 
by
  sorry

end perimeter_of_quadrilateral_eq_fifty_l87_87605


namespace contrapositive_equivalence_l87_87681

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 + 3*x - 4 = 0 → x = -4 ∨ x = 1)) ↔ (∀ x : ℝ, (x ≠ -4 ∧ x ≠ 1 → x^2 + 3*x - 4 ≠ 0)) :=
by {
  sorry
}

end contrapositive_equivalence_l87_87681


namespace pizza_toppings_problem_l87_87862

theorem pizza_toppings_problem
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (pepperoni_mushroom_slices : ℕ)
  (pepperoni_olive_slices : ℕ)
  (mushroom_olive_slices : ℕ)
  (pepperoni_mushroom_olive_slices : ℕ) :
  total_slices = 20 →
  pepperoni_slices = 12 →
  mushroom_slices = 14 →
  olive_slices = 12 →
  pepperoni_mushroom_slices = 8 →
  pepperoni_olive_slices = 8 →
  mushroom_olive_slices = 8 →
  total_slices = pepperoni_slices + mushroom_slices + olive_slices
    - pepperoni_mushroom_slices - pepperoni_olive_slices - mushroom_olive_slices
    + pepperoni_mushroom_olive_slices →
  pepperoni_mushroom_olive_slices = 6 :=
by
  intros
  sorry

end pizza_toppings_problem_l87_87862


namespace sasha_work_fraction_l87_87545

theorem sasha_work_fraction :
  let sasha_first := 1 / 3
  let sasha_second := 1 / 5
  let sasha_third := 1 / 15
  let total_sasha_contribution := sasha_first + sasha_second + sasha_third
  let fraction_per_car := total_sasha_contribution / 3
  fraction_per_car = 1 / 5 :=
by
  sorry

end sasha_work_fraction_l87_87545


namespace solve_for_x_l87_87841

theorem solve_for_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) → x = 3 / 2 := by
  sorry

end solve_for_x_l87_87841


namespace value_of_def_ef_l87_87840

theorem value_of_def_ef
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : (a * f) / (c * d) = 1)
  : d * e * f = 250 := 
by 
  sorry

end value_of_def_ef_l87_87840


namespace isosceles_triangle_angle_condition_l87_87253

theorem isosceles_triangle_angle_condition (A B C : ℝ) (h_iso : A = B) (h_angle_eq : A = 2 * C ∨ C = 2 * A) :
    (A = 45 ∨ A = 72) ∧ (B = 45 ∨ B = 72) :=
by
  -- Given isosceles triangle properties.
  sorry

end isosceles_triangle_angle_condition_l87_87253


namespace frosting_sugar_l87_87275

-- Define the conditions as constants
def total_sugar : ℝ := 0.8
def cake_sugar : ℝ := 0.2

-- The theorem stating that the sugar required for the frosting is 0.6 cups
theorem frosting_sugar : total_sugar - cake_sugar = 0.6 := by
  sorry

end frosting_sugar_l87_87275


namespace polynomial_inequality_l87_87082

-- Define the polynomial P and its condition
def P (a b c : ℝ) (x : ℝ) : ℝ := 12 * x^3 + a * x^2 + b * x + c
-- Define the polynomial Q and its condition
def Q (a b c : ℝ) (x : ℝ) : ℝ := (x^2 + x + 2001)^3 + a * (x^2 + x + 2001)^2 + b * (x^2 + x + 2001) + c

-- Assumptions
axiom P_has_distinct_roots (a b c : ℝ) : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0
axiom Q_has_no_real_roots (a b c : ℝ) : ¬ ∃ x : ℝ, Q a b c x = 0

-- The goal to prove
theorem polynomial_inequality (a b c : ℝ) (h1 : ∃ p q r : ℝ, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ P a b c p = 0 ∧ P a b c q = 0 ∧ P a b c r = 0)
  (h2 : ¬ ∃ x : ℝ, Q a b c x = 0) : 2001^3 + a * 2001^2 + b * 2001 + c > 1 / 64 :=
by {
  -- sorry is added to skip the proof part
  sorry
}

end polynomial_inequality_l87_87082


namespace problem_21_divisor_l87_87475

theorem problem_21_divisor 
    (k : ℕ) 
    (h1 : ∃ k, 21^k ∣ 435961) 
    (h2 : 21^k ∣ 435961) 
    : 7^k - k^7 = 1 := 
sorry

end problem_21_divisor_l87_87475


namespace cost_of_one_shirt_l87_87162

theorem cost_of_one_shirt (J S K : ℕ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) (h3 : 3 * J + 2 * S + K = 90) : S = 15 :=
by
  sorry

end cost_of_one_shirt_l87_87162


namespace constant_term_binomial_expansion_n_6_middle_term_coefficient_l87_87468

open Nat

-- Define the binomial expansion term
def binomial_term (n : ℕ) (r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (2 ^ r) * x^(2 * (n-r) - r)

-- (I) Prove the constant term of the binomial expansion when n = 6
theorem constant_term_binomial_expansion_n_6 :
  binomial_term 6 4 (1 : ℝ) = 240 := 
sorry

-- (II) Prove the coefficient of the middle term under given conditions
theorem middle_term_coefficient (n : ℕ) :
  (Nat.choose 8 2 = Nat.choose 8 6) →
  binomial_term 8 4 (1 : ℝ) = 1120 := 
sorry

end constant_term_binomial_expansion_n_6_middle_term_coefficient_l87_87468


namespace sheet_width_l87_87974

theorem sheet_width (L : ℕ) (w : ℕ) (A_typist : ℚ) 
  (L_length : L = 30)
  (A_typist_percentage : A_typist = 0.64) 
  (width_used : ∀ w, w > 0 → (w - 4) * (24 : ℕ) = A_typist * w * 30) : 
  w = 20 :=
by
  intros
  sorry

end sheet_width_l87_87974


namespace stamps_ratio_l87_87587

noncomputable def number_of_stamps_bought := 300
noncomputable def total_stamps_after_purchase := 450
noncomputable def number_of_stamps_before_purchase := total_stamps_after_purchase - number_of_stamps_bought

theorem stamps_ratio : (number_of_stamps_before_purchase : ℚ) / number_of_stamps_bought = 1 / 2 := by
  have h : number_of_stamps_before_purchase = total_stamps_after_purchase - number_of_stamps_bought := rfl
  rw [h]
  norm_num
  sorry

end stamps_ratio_l87_87587


namespace set_A_is_correct_l87_87977

open Complex

def A : Set ℤ := {x | ∃ n : ℕ, n > 0 ∧ x = (I ^ n + (-I) ^ n).re}

theorem set_A_is_correct : A = {-2, 0, 2} :=
sorry

end set_A_is_correct_l87_87977


namespace angle_sum_x_y_l87_87438

theorem angle_sum_x_y 
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) (x : ℝ) (y : ℝ) 
  (hA : angle_A = 34) (hB : angle_B = 80) (hC : angle_C = 30) 
  (hexagon_property : ∀ A B x y : ℝ, A + B + 360 - x + 90 + 120 - y = 720) :
  x + y = 36 :=
by
  sorry

end angle_sum_x_y_l87_87438


namespace marathon_end_time_l87_87466

open Nat

def marathonStart := 15 * 60  -- 3:00 p.m. in minutes (15 hours * 60 minutes)
def marathonDuration := 780    -- Duration in minutes

theorem marathon_end_time : marathonStart + marathonDuration = 28 * 60 := -- 4:00 a.m. in minutes (28 hours * 60 minutes)
  sorry

end marathon_end_time_l87_87466


namespace maximize_Sn_l87_87901

def a_n (n : ℕ) : ℤ := 26 - 2 * n

def S_n (n : ℕ) : ℤ := n * (26 - 2 * (n + 1)) / 2 + 26 * n

theorem maximize_Sn : (n = 12 ∨ n = 13) ↔ (∀ m : ℕ, S_n m ≤ S_n 12 ∨ S_n m ≤ S_n 13) :=
by sorry

end maximize_Sn_l87_87901


namespace selection_and_arrangement_l87_87429

-- Defining the problem conditions
def volunteers : Nat := 5
def roles : Nat := 4
def A_excluded_role : String := "music_composer"
def total_methods : Nat := 96

theorem selection_and_arrangement (h1 : volunteers = 5) (h2 : roles = 4) (h3 : A_excluded_role = "music_composer") :
  total_methods = 96 :=
by
  sorry

end selection_and_arrangement_l87_87429


namespace polygon_diagonals_15_sides_l87_87494

/-- Given a convex polygon with 15 sides, the number of diagonals is 90. -/
theorem polygon_diagonals_15_sides (n : ℕ) (h : n = 15) (convex : Prop) : 
  ∃ d : ℕ, d = 90 :=
by
    sorry

end polygon_diagonals_15_sides_l87_87494


namespace average_physics_chemistry_l87_87838

theorem average_physics_chemistry (P C M : ℕ) 
  (h1 : (P + C + M) / 3 = 80)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 80) :
  (P + C) / 2 = 70 := 
sorry

end average_physics_chemistry_l87_87838


namespace ellipse_problem_l87_87292

-- Definitions of conditions from the problem
def F1 := (0, 0)
def F2 := (6, 0)
def ellipse_equation (x y h k a b : ℝ) := ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

-- The main statement to be proved
theorem ellipse_problem :
  let h := 3
  let k := 0
  let a := 5
  let c := 3
  let b := Real.sqrt (a^2 - c^2)
  h + k + a + b = 12 :=
by
  -- Proof would go here
  sorry

end ellipse_problem_l87_87292


namespace total_money_l87_87932

-- Conditions
def mark_amount : ℚ := 5 / 6
def carolyn_amount : ℚ := 2 / 5

-- Combine both amounts and state the theorem to be proved
theorem total_money : mark_amount + carolyn_amount = 1.233 := by
  -- placeholder for the actual proof
  sorry

end total_money_l87_87932


namespace axis_of_symmetry_of_shifted_function_l87_87246

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry_of_shifted_function :
  (∃ x : ℝ, g x = 1 ∧ x = Real.pi / 12) :=
by
  sorry

end axis_of_symmetry_of_shifted_function_l87_87246


namespace sum_arithmetic_series_l87_87010

theorem sum_arithmetic_series :
  let a := -42
  let d := 2
  let l := 0
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = -462 := by
sorry

end sum_arithmetic_series_l87_87010


namespace train_length_correct_l87_87133

noncomputable def train_length (speed_kmh: ℝ) (time_s: ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_s

theorem train_length_correct :
  train_length 60 15 = 250.05 := 
by
  sorry

end train_length_correct_l87_87133


namespace find_a_l87_87115

noncomputable def binomial_coeff (n k : ℕ) := Nat.choose n k

theorem find_a (a : ℝ) 
  (h : ∃ (a : ℝ), a ^ 3 * binomial_coeff 8 3 = 56) : a = 1 :=
by
  sorry

end find_a_l87_87115


namespace bryan_total_after_discount_l87_87517

theorem bryan_total_after_discount 
  (n : ℕ) (p : ℝ) (d : ℝ) (h_n : n = 8) (h_p : p = 1785) (h_d : d = 0.12) :
  (n * p - (n * p * d) = 12566.4) :=
by
  sorry

end bryan_total_after_discount_l87_87517


namespace max_largest_integer_of_five_l87_87477

theorem max_largest_integer_of_five (a b c d e : ℕ) (h1 : (a + b + c + d + e) = 500)
    (h2 : e > c ∧ c > d ∧ d > b ∧ b > a)
    (h3 : (a + b + d + e) / 4 = 105)
    (h4 : b + e = 150) : d ≤ 269 := 
sorry

end max_largest_integer_of_five_l87_87477


namespace systematic_sampling_number_l87_87790

theorem systematic_sampling_number {n m s a b c d : ℕ} (h_n : n = 60) (h_m : m = 4) 
  (h_s : s = 3) (h_a : a = 33) (h_b : b = 48) 
  (h_gcd_1 : ∃ k, s + k * (n / m) = a) (h_gcd_2 : ∃ k, a + k * (n / m) = b) :
  ∃ k, s + k * (n / m) = d → d = 18 := by
  sorry

end systematic_sampling_number_l87_87790


namespace smallest_multiple_of_6_and_15_l87_87776

theorem smallest_multiple_of_6_and_15 : ∃ a : ℕ, a > 0 ∧ a % 6 = 0 ∧ a % 15 = 0 ∧ ∀ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 → a ≤ b :=
  sorry

end smallest_multiple_of_6_and_15_l87_87776


namespace molecular_weight_of_Aluminium_hydroxide_l87_87175

-- Given conditions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Definition of molecular weight of Aluminium hydroxide
def molecular_weight_Al_OH_3 : ℝ := 
  atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

-- Proof statement
theorem molecular_weight_of_Aluminium_hydroxide : molecular_weight_Al_OH_3 = 78.01 :=
  by sorry

end molecular_weight_of_Aluminium_hydroxide_l87_87175


namespace P1_coordinates_l87_87821

-- Define initial point coordinates
def P : (ℝ × ℝ) := (0, 3)

-- Define the transformation functions
def move_left (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1 - units, p.2)
def move_up (p : ℝ × ℝ) (units : ℝ) : (ℝ × ℝ) := (p.1, p.2 + units)

-- Calculate the coordinates of point P1
def P1 : (ℝ × ℝ) := move_up (move_left P 2) 1

-- Statement to prove
theorem P1_coordinates : P1 = (-2, 4) := by
  sorry

end P1_coordinates_l87_87821


namespace tangent_line_sum_l87_87687

theorem tangent_line_sum (a b : ℝ) :
  (∃ x₀ : ℝ, (e^(x₀ - 1) = 1) ∧ (x₀ + a = e^(x₀-1) * (1 - x₀) - b + 1)) → a + b = 1 :=
by
  sorry

end tangent_line_sum_l87_87687


namespace shorter_piece_length_l87_87008

theorem shorter_piece_length (x : ℕ) (h1 : 177 = x + 2*x) : x = 59 :=
by sorry

end shorter_piece_length_l87_87008


namespace server_processes_21600000_requests_l87_87182

theorem server_processes_21600000_requests :
  (15000 * 1440 = 21600000) :=
by
  -- Calculations and step-by-step proof
  sorry

end server_processes_21600000_requests_l87_87182


namespace evaluate_expression_l87_87756

theorem evaluate_expression (a b c : ℕ) (h1 : a = 12) (h2 : b = 8) (h3 : c = 3) :
  (a - b + c - (a - (b + c)) = 6) := by
  sorry

end evaluate_expression_l87_87756


namespace domain_of_f_l87_87946

noncomputable def f (x : ℝ) : ℝ := 1 / x + Real.sqrt (-x^2 + x + 2)

theorem domain_of_f :
  {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 ∧ x ≠ 0} :=
by
  sorry

end domain_of_f_l87_87946


namespace find_extrema_of_A_l87_87814

theorem find_extrema_of_A (x y : ℝ) (h : x^2 + y^2 = 4) : 2 ≤ x^2 + x * y + y^2 ∧ x^2 + x * y + y^2 ≤ 6 :=
by 
  sorry

end find_extrema_of_A_l87_87814


namespace find_T_l87_87813

variable {n : ℕ}
variable {a b : ℕ → ℕ}
variable {S T : ℕ → ℕ}

-- Conditions
axiom h1 : ∀ n, b n - a n = 2^n + 1
axiom h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2

-- Goal
theorem find_T (n : ℕ) (a b S T : ℕ → ℕ)
  (h1 : ∀ n, b n - a n = 2^n + 1)
  (h2 : ∀ n, S n + T n = 2^(n + 1) + n^2 - 2) :
  T n = 2^(n + 1) + n * (n + 1) / 2 - 5 := sorry

end find_T_l87_87813


namespace monotonic_increase_interval_range_of_a_l87_87041

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.exp x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := f x + 2 * Real.exp x - a * x^2
def h (x : ℝ) : ℝ := x

theorem monotonic_increase_interval :
  ∃ I : Set ℝ, I = Set.Ioi 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x ≤ y → f x ≤ f y := 
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (g x1 a - h x1) * (g x2 a - h x2) > 0) ↔ a ∈ Set.Iic 1 :=
  sorry

end monotonic_increase_interval_range_of_a_l87_87041


namespace hyperbola_eccentricity_l87_87386

theorem hyperbola_eccentricity (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∀ x : ℝ, y = (3 / 4) * x → y = (b / a) * x) : 
  (b = (3 / 4) * a) → (e = 5 / 4) := 
by
  sorry

end hyperbola_eccentricity_l87_87386


namespace no_real_value_x_l87_87451

theorem no_real_value_x (R H : ℝ) (π : ℝ := Real.pi) :
  R = 10 → H = 5 →
  ¬∃ x : ℝ,  π * (R + x)^2 * H = π * R^2 * (H + x) ∧ x ≠ 0 :=
by
  intros hR hH; sorry

end no_real_value_x_l87_87451


namespace area_of_triangle_l87_87496

theorem area_of_triangle (A B C : ℝ) (a c : ℝ) (d B_value: ℝ) (h1 : A + B + C = 180) 
                         (h2 : A = B - d) (h3 : C = B + d) (h4 : a = 4) (h5 : c = 3)
                         (h6 : B = 60) :
  (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = 3 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l87_87496


namespace production_problem_l87_87780

theorem production_problem (x y : ℝ) (h₁ : x > 0) (h₂ : ∀ k : ℝ, x * x * x * k = x) : (x * x * y * (1 / (x^2)) = y) :=
by {
  sorry
}

end production_problem_l87_87780


namespace problem_statement_l87_87366

theorem problem_statement 
  (a b c : ℤ)
  (h1 : (5 * a + 2) ^ (1/3) = 3)
  (h2 : (3 * a + b - 1) ^ (1/2) = 4)
  (h3 : c = Int.floor (Real.sqrt 13))
  : a = 5 ∧ b = 2 ∧ c = 3 ∧ Real.sqrt (3 * a - b + c) = 4 := 
by 
  sorry

end problem_statement_l87_87366


namespace pears_total_correct_l87_87812

noncomputable def pickedPearsTotal (sara_picked tim_picked : Nat) : Nat :=
  sara_picked + tim_picked

theorem pears_total_correct :
    pickedPearsTotal 6 5 = 11 :=
  by
    sorry

end pears_total_correct_l87_87812


namespace probability_two_white_balls_same_color_l87_87851

theorem probability_two_white_balls_same_color :
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  (total_combinations_white + total_combinations_black > 0) →
  (total_combinations_white / total_combinations_same_color) = (3 / 4) :=
by
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  intro h
  sorry

end probability_two_white_balls_same_color_l87_87851


namespace soccer_tournament_probability_l87_87548

noncomputable def prob_teamA_more_points : ℚ :=
  (163 : ℚ) / 256

theorem soccer_tournament_probability :
  m + n = 419 ∧ prob_teamA_more_points = 163 / 256 := sorry

end soccer_tournament_probability_l87_87548


namespace last_five_digits_l87_87684

theorem last_five_digits : (99 * 10101 * 111 * 1001) % 100000 = 88889 :=
by
  sorry

end last_five_digits_l87_87684


namespace path_count_from_E_to_G_passing_through_F_l87_87873

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem path_count_from_E_to_G_passing_through_F :
  let E := (0, 0)
  let F := (5, 2)
  let G := (6, 5)
  ∃ (paths_EF paths_FG total_paths : ℕ),
  paths_EF = binom (5 + 2) 5 ∧
  paths_FG = binom (1 + 3) 1 ∧
  total_paths = paths_EF * paths_FG ∧
  total_paths = 84 := 
by
  sorry

end path_count_from_E_to_G_passing_through_F_l87_87873


namespace percentage_of_white_chips_l87_87361

theorem percentage_of_white_chips (T : ℕ) (h1 : 3 = 10 * T / 100) (h2 : 12 = 12): (15 / T * 100) = 50 := by
  sorry

end percentage_of_white_chips_l87_87361


namespace base7_addition_l87_87406

theorem base7_addition (X Y : ℕ) (h1 : X + 5 = 9) (h2 : Y + 2 = 4) : X + Y = 6 :=
by
  sorry

end base7_addition_l87_87406


namespace prism_sphere_surface_area_l87_87894

theorem prism_sphere_surface_area :
  ∀ (a b c : ℝ), (a * b = 6) → (b * c = 2) → (a * c = 3) → 
  4 * Real.pi * ((Real.sqrt ((a ^ 2) + (b ^ 2) + (c ^ 2))) / 2) ^ 2 = 14 * Real.pi :=
by
  intros a b c hab hbc hac
  sorry

end prism_sphere_surface_area_l87_87894


namespace num_roses_given_l87_87106

theorem num_roses_given (n : ℕ) (m : ℕ) (x : ℕ) :
  n = 28 → 
  (∀ (b g : ℕ), b + g = n → b * g = 45 * x) →
  (num_roses : ℕ) = 4 * x →
  (num_tulips : ℕ) = 10 * num_roses →
  (num_daffodils : ℕ) = x →
  num_roses = 16 :=
by
  sorry

end num_roses_given_l87_87106


namespace trapezium_other_side_length_l87_87345

theorem trapezium_other_side_length (a h Area : ℕ) (a_eq : a = 4) (h_eq : h = 6) (Area_eq : Area = 27) : 
  ∃ (b : ℕ), b = 5 := 
by
  sorry

end trapezium_other_side_length_l87_87345


namespace minimum_phi_l87_87519

noncomputable def initial_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * x + ϕ)

noncomputable def translated_function (x : ℝ) (ϕ : ℝ) : ℝ :=
  2 * Real.sin (4 * (x - (Real.pi / 6)) + ϕ)

theorem minimum_phi (ϕ : ℝ) :
  (∃ k : ℤ, ϕ = k * Real.pi + 7 * Real.pi / 6) →
  (∃ ϕ_min : ℝ, (ϕ_min = ϕ ∧ ϕ_min = Real.pi / 6)) :=
by
  sorry

end minimum_phi_l87_87519


namespace sum_of_products_nonpos_l87_87153

theorem sum_of_products_nonpos (a b c : ℝ) (h : a + b + c = 0) : 
  a * b + a * c + b * c ≤ 0 :=
sorry

end sum_of_products_nonpos_l87_87153


namespace factor_polynomial_l87_87188

theorem factor_polynomial (x y : ℝ) : 
  x^4 + 4 * y^4 = (x^2 - 2 * x * y + 2 * y^2) * (x^2 + 2 * x * y + 2 * y^2) :=
by
  sorry

end factor_polynomial_l87_87188


namespace darla_total_payment_l87_87220

-- Define the cost per watt, total watts used, and late fee
def cost_per_watt : ℝ := 4
def total_watts : ℝ := 300
def late_fee : ℝ := 150

-- Define the total cost of electricity
def electricity_cost : ℝ := cost_per_watt * total_watts

-- Define the total amount Darla needs to pay
def total_amount : ℝ := electricity_cost + late_fee

-- The theorem to prove the total amount equals $1350
theorem darla_total_payment : total_amount = 1350 := by
  sorry

end darla_total_payment_l87_87220


namespace men_employed_l87_87579

theorem men_employed (M : ℕ) (W : ℕ)
  (h1 : W = M * 9)
  (h2 : W = (M + 10) * 6) : M = 20 := by
  sorry

end men_employed_l87_87579


namespace determine_k_l87_87981

theorem determine_k (a b c k : ℤ) (h1 : c = -a - b) 
  (h2 : 60 < 6 * (8 * a + b) ∧ 6 * (8 * a + b) < 70)
  (h3 : 80 < 7 * (9 * a + b) ∧ 7 * (9 * a + b) < 90)
  (h4 : 2000 * k < (50^2 * a + 50 * b + c) ∧ (50^2 * a + 50 * b + c) < 2000 * (k + 1)) :
  k = 1 :=
  sorry

end determine_k_l87_87981


namespace kiwi_count_l87_87611

theorem kiwi_count (s b o k : ℕ)
  (h1 : s + b + o + k = 340)
  (h2 : s = 3 * b)
  (h3 : o = 2 * k)
  (h4 : k = 5 * s) :
  k = 104 :=
sorry

end kiwi_count_l87_87611


namespace part_a_part_b_l87_87372

-- Define a polygon type
structure Polygon :=
  (sides : ℕ)
  (area : ℝ)
  (grid_size : ℕ)

-- Define a function to verify drawable polygon
def DrawablePolygon (p : Polygon) : Prop :=
  ∃ (n : ℕ), p.grid_size = n ∧ p.area = n ^ 2

-- Part (a): 20-sided polygon with an area of 9
theorem part_a : DrawablePolygon {sides := 20, area := 9, grid_size := 3} :=
by
  sorry

-- Part (b): 100-sided polygon with an area of 49
theorem part_b : DrawablePolygon {sides := 100, area := 49, grid_size := 7} :=
by
  sorry

end part_a_part_b_l87_87372


namespace smallest_b_undefined_inverse_l87_87169

theorem smallest_b_undefined_inverse (b : ℕ) (h1 : Nat.gcd b 84 > 1) (h2 : Nat.gcd b 90 > 1) : b = 6 :=
sorry

end smallest_b_undefined_inverse_l87_87169


namespace probability_XiaoCong_project_A_probability_same_project_not_C_l87_87758

-- Definition of projects and conditions
inductive Project
| A | B | C

def XiaoCong : Project := sorry
def XiaoYing : Project := sorry

-- (1) Probability of Xiao Cong assigned to project A
theorem probability_XiaoCong_project_A : 
  (1 / 3 : ℝ) = 1 / 3 := 
by sorry

-- (2) Probability of Xiao Cong and Xiao Ying being assigned to the same project, given Xiao Ying not assigned to C
theorem probability_same_project_not_C : 
  (2 / 6 : ℝ) = 1 / 3 :=
by sorry

end probability_XiaoCong_project_A_probability_same_project_not_C_l87_87758


namespace non_zero_number_is_nine_l87_87576

theorem non_zero_number_is_nine (x : ℝ) (h1 : x ≠ 0) (h2 : (x + x^2) / 2 = 5 * x) : x = 9 :=
by
  sorry

end non_zero_number_is_nine_l87_87576


namespace quadratic_no_real_roots_l87_87855

theorem quadratic_no_real_roots : ∀ (a b c : ℝ), a ≠ 0 → Δ = (b*b - 4*a*c) → x^2 + 3 = 0 → Δ < 0 := by
  sorry

end quadratic_no_real_roots_l87_87855


namespace ratio_xy_l87_87668

theorem ratio_xy (x y : ℝ) (h : 2*y - 5*x = 0) : x / y = 2 / 5 :=
by sorry

end ratio_xy_l87_87668


namespace floor_sqrt_23_squared_eq_16_l87_87872

theorem floor_sqrt_23_squared_eq_16 :
  (Int.floor (Real.sqrt 23))^2 = 16 :=
by
  have h1 : 4 < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < 5 := sorry
  have floor_sqrt_23 : Int.floor (Real.sqrt 23) = 4 := sorry
  rw [floor_sqrt_23]
  norm_num

end floor_sqrt_23_squared_eq_16_l87_87872


namespace substitution_result_l87_87976

-- Conditions
def eq1 (x y : ℝ) : Prop := y = 2 * x - 3
def eq2 (x y : ℝ) : Prop := x - 2 * y = 6

-- The statement to be proven
theorem substitution_result (x y : ℝ) (h1 : eq1 x y) : (x - 4 * x + 6 = 6) :=
by sorry

end substitution_result_l87_87976


namespace total_pics_uploaded_l87_87686

-- Definitions of conditions
def pic_in_first_album : Nat := 14
def albums_with_7_pics : Nat := 3
def pics_per_album : Nat := 7

-- Theorem statement
theorem total_pics_uploaded :
  pic_in_first_album + albums_with_7_pics * pics_per_album = 35 := by
  sorry

end total_pics_uploaded_l87_87686


namespace annual_pension_l87_87373

theorem annual_pension (c d r s x k : ℝ) (hc : c ≠ 0) (hd : d ≠ c)
  (h1 : k * (x + c) ^ (3 / 2) = k * x ^ (3 / 2) + r)
  (h2 : k * (x + d) ^ (3 / 2) = k * x ^ (3 / 2) + s) :
  k * x ^ (3 / 2) = 4 * r^2 / (9 * c^2) :=
by
  sorry

end annual_pension_l87_87373


namespace mail_distribution_l87_87047

def pieces_per_block (total_pieces blocks : ℕ) : ℕ := total_pieces / blocks

theorem mail_distribution : pieces_per_block 192 4 = 48 := 
by { 
    -- Proof skipped
    sorry 
}

end mail_distribution_l87_87047


namespace processing_times_maximum_salary_l87_87380

def monthly_hours : ℕ := 8 * 25
def base_salary : ℕ := 800
def earnings_per_A : ℕ := 16
def earnings_per_B : ℕ := 12

theorem processing_times :
  ∃ (x y : ℕ),
    x + 3 * y = 5 ∧ 2 * x + 5 * y = 9 ∧ x = 2 ∧ y = 1 :=
by
  sorry

theorem maximum_salary :
  ∃ (a b W : ℕ),
    a ≥ 50 ∧ 
    b = monthly_hours - 2 * a ∧ 
    W = base_salary + earnings_per_A * a + earnings_per_B * b ∧ 
    a = 50 ∧ 
    b = 100 ∧ 
    W = 2800 :=
by
  sorry

end processing_times_maximum_salary_l87_87380


namespace ellipse_standard_equation_parabola_standard_equation_l87_87550

theorem ellipse_standard_equation (x y : ℝ) (a b : ℝ) (h₁ : a > b ∧ b > 0)
  (h₂ : 2 * a = Real.sqrt ((3 + 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2) 
      + Real.sqrt ((3 - 2) ^ 2 + (-2 * Real.sqrt 6) ^ 2))
  (h₃ : b^2 = a^2 - 4) 
  : (x^2 / 36 + y^2 / 32 = 1) :=
by sorry

theorem parabola_standard_equation (y : ℝ) (p : ℝ) (h₁ : p > 0)
  (h₂ : -p / 2 = -1 / 2) 
  : (y^2 = 2 * p * 1) :=
by sorry

end ellipse_standard_equation_parabola_standard_equation_l87_87550


namespace grinder_price_l87_87258

variable (G : ℝ) (PurchasedMobile : ℝ) (SoldMobile : ℝ) (overallProfit : ℝ)

theorem grinder_price (h1 : PurchasedMobile = 10000)
                      (h2 : SoldMobile = 11000)
                      (h3 : overallProfit = 400)
                      (h4 : 0.96 * G + SoldMobile = G + PurchasedMobile + overallProfit) :
                      G = 15000 := by
  sorry

end grinder_price_l87_87258


namespace min_value_of_sum_l87_87071

theorem min_value_of_sum (x y : ℝ) (h1 : x + 4 * y = 2 * x * y) (h2 : 0 < x) (h3 : 0 < y) : 
  x + y ≥ 9 / 2 :=
sorry

end min_value_of_sum_l87_87071


namespace stddev_newData_l87_87040

-- Definitions and conditions
def variance (data : List ℝ) : ℝ := sorry  -- Placeholder for variance definition
def stddev (data : List ℝ) : ℝ := sorry    -- Placeholder for standard deviation definition

-- Given data
def data : List ℝ := sorry                -- Placeholder for the data x_1, x_2, ..., x_8
def newData : List ℝ := data.map (λ x => 2 * x + 1)

-- Given condition
axiom variance_data : variance data = 16

-- Proof of the statement
theorem stddev_newData : stddev newData = 8 :=
by {
  sorry
}

end stddev_newData_l87_87040


namespace find_m_b_l87_87174

noncomputable def line_equation (x y : ℝ) :=
  (⟨-1, 4⟩ : ℝ × ℝ) • (⟨x, y⟩ - ⟨3, -5⟩ : ℝ × ℝ) = 0

theorem find_m_b : ∃ m b : ℝ, (∀ (x y : ℝ), line_equation x y → y = m * x + b) ∧ m = 1 / 4 ∧ b = -23 / 4 :=
by
  sorry

end find_m_b_l87_87174


namespace min_circles_l87_87924

noncomputable def segments_intersecting_circles (N : ℕ) : Prop :=
  ∀ seg : (ℝ × ℝ) × ℝ, (seg.fst.fst ≥ 0 ∧ seg.fst.fst + seg.snd ≤ 100 ∧ seg.fst.snd ≥ 0 ∧ seg.fst.snd ≤ 100 ∧ seg.snd = 10) →
    ∃ c : ℝ × ℝ, (dist c seg.fst < 1 ∧ c.fst ≥ 0 ∧ c.fst ≤ 100 ∧ c.snd ≥ 0 ∧ c.snd ≤ 100) 

theorem min_circles (N : ℕ) (h : segments_intersecting_circles N) : N ≥ 400 :=
sorry

end min_circles_l87_87924


namespace explicit_form_correct_l87_87602

-- Define the original function form
def f (a b x : ℝ) := 4*x^3 + a*x^2 + b*x + 5

-- Given tangent line slope condition at x = 1
axiom tangent_slope : ∀ (a b : ℝ), (12 * 1^2 + 2 * a * 1 + b = -12)

-- Given the point (1, f(1)) lies on the tangent line y = -12x
axiom tangent_point : ∀ (a b : ℝ), (4 * 1^3 + a * 1^2 + b * 1 + 5 = -12)

-- Definition for the specific f(x) found in solution
def f_explicit (x : ℝ) := 4*x^3 - 3*x^2 - 18*x + 5

-- Finding maximum and minimum values on interval [-3, 1]
def max_value : ℝ := -76
def min_value : ℝ := 16

theorem explicit_form_correct : 
  ∃ a b : ℝ, 
  (∀ x, f a b x = f_explicit x) ∧ 
  (max_value = 16) ∧ 
  (min_value = -76) := 
by
  sorry

end explicit_form_correct_l87_87602


namespace find_Allyson_age_l87_87391

variable (Hiram Allyson : ℕ)

theorem find_Allyson_age (h : Hiram = 40)
  (condition : Hiram + 12 = 2 * Allyson - 4) : Allyson = 28 := by
  sorry

end find_Allyson_age_l87_87391


namespace abs_inequality_solution_l87_87771

theorem abs_inequality_solution (x : ℝ) : (|x + 3| > x + 3) ↔ (x < -3) :=
by
  sorry

end abs_inequality_solution_l87_87771


namespace evaluate_expression_l87_87688

theorem evaluate_expression : (1 / (5^2)^4) * 5^15 = 5^7 :=
by
  sorry

end evaluate_expression_l87_87688


namespace total_population_l87_87142

theorem total_population (x T : ℝ) (h : 128 = (x / 100) * (50 / 100) * T) : T = 25600 / x :=
by
  sorry

end total_population_l87_87142


namespace Nara_height_is_1_69_l87_87914

-- Definitions of the conditions
def SangheonHeight : ℝ := 1.56
def ChihoHeight : ℝ := SangheonHeight - 0.14
def NaraHeight : ℝ := ChihoHeight + 0.27

-- The statement to prove
theorem Nara_height_is_1_69 : NaraHeight = 1.69 :=
by {
  sorry
}

end Nara_height_is_1_69_l87_87914


namespace vector_subtraction_l87_87073

theorem vector_subtraction (p q: ℝ × ℝ × ℝ) (hp: p = (5, -3, 2)) (hq: q = (-1, 4, -2)) :
  p - 2 • q = (7, -11, 6) :=
by
  sorry

end vector_subtraction_l87_87073


namespace distance_circumcenter_centroid_inequality_l87_87497

variable {R r d : ℝ}

theorem distance_circumcenter_centroid_inequality 
  (h1 : d = distance_circumcenter_to_centroid)
  (h2 : R = circumradius)
  (h3 : r = inradius) : d^2 ≤ R * (R - 2 * r) := 
sorry

end distance_circumcenter_centroid_inequality_l87_87497


namespace reduction_for_1750_yuan_max_daily_profit_not_1900_l87_87616

def average_shirts_per_day : ℕ := 40 
def profit_per_shirt_initial : ℕ := 40 
def price_reduction_increase_shirts (reduction : ℝ) : ℝ := reduction * 2 
def daily_profit (reduction : ℝ) : ℝ := (profit_per_shirt_initial - reduction) * (average_shirts_per_day + price_reduction_increase_shirts reduction)

-- Part 1: Proving the reduction that results in 1750 yuan profit
theorem reduction_for_1750_yuan : ∃ x : ℝ, daily_profit x = 1750 ∧ x = 15 := 
by {
  sorry
}

-- Part 2: Proving that the maximum cannot reach 1900 yuan
theorem max_daily_profit_not_1900 : ∀ x : ℝ, daily_profit x ≤ 1800 ∧ (∀ y : ℝ, y ≥ daily_profit x → y < 1900) :=
by {
  sorry
}

end reduction_for_1750_yuan_max_daily_profit_not_1900_l87_87616


namespace number_of_sets_l87_87214

theorem number_of_sets (M : Set ℕ) : 
  {1, 2} ⊆ M → M ⊆ {1, 2, 3, 4} → ∃ n : ℕ, n = 4 :=
by
  sorry

end number_of_sets_l87_87214


namespace find_line_equation_l87_87613

theorem find_line_equation : 
  ∃ (m : ℝ), (∀ (x y : ℝ), (2 * x + y - 5 = 0) → (m = -2)) → 
  ∀ (x₀ y₀ : ℝ), (x₀ = -2) ∧ (y₀ = 3) → 
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * x₀ + b * y₀ + c = 0) ∧ (a = 1 ∧ b = -2 ∧ c = 8) := 
by
  sorry

end find_line_equation_l87_87613


namespace function_zeros_range_l87_87826

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then (1 / 2)^x + 2 / x else x * Real.log x - a

theorem function_zeros_range (a : ℝ) :
  (∀ x : ℝ, f x a = 0 → x < 0) ∧ (∀ x : ℝ, f x a = 0 → x > 0 → (a > -1 / Real.exp 1 ∧ a < 0)) ↔
  (a > -1 / Real.exp 1 ∧ a < 0) :=
sorry

end function_zeros_range_l87_87826


namespace quadratic_roots_l87_87313

theorem quadratic_roots (r s : ℝ) (A : ℝ) (B : ℝ) (C : ℝ) (p q : ℝ) 
  (h1 : A = 3) (h2 : B = 4) (h3 : C = 5) 
  (h4 : r + s = -B / A) (h5 : rs = C / A) 
  (h6 : 4 * rs = q) :
  p = 56 / 9 :=
by 
  -- We assume the correct answer is given as we skip the proof details here.
  sorry

end quadratic_roots_l87_87313


namespace board_total_length_l87_87917

-- Definitions based on conditions
def S : ℝ := 2
def L : ℝ := 2 * S

-- Define the total length of the board
def T : ℝ := S + L

-- The theorem asserting the total length of the board is 6 ft
theorem board_total_length : T = 6 := 
by
  sorry

end board_total_length_l87_87917


namespace altitude_line_eq_circumcircle_eq_l87_87381

noncomputable def point := ℝ × ℝ

noncomputable def A : point := (5, 1)
noncomputable def B : point := (1, 3)
noncomputable def C : point := (4, 4)

theorem altitude_line_eq : ∃ (k b : ℝ), (k = 2 ∧ b = -4) ∧ (∀ x y : ℝ, y = k * x + b ↔ 2 * x - y - 4 = 0) :=
sorry

theorem circumcircle_eq : ∃ (h k r : ℝ), (h = 3 ∧ k = 2 ∧ r = 5) ∧ (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r ↔ (x - 3)^2 + (y - 2)^2 = 5) :=
sorry

end altitude_line_eq_circumcircle_eq_l87_87381


namespace board_game_cost_l87_87997

theorem board_game_cost
  (v h : ℝ)
  (h1 : 3 * v = h + 490)
  (h2 : 5 * v = 2 * h + 540) :
  h = 830 := by
  sorry

end board_game_cost_l87_87997


namespace symmetric_point_l87_87672

-- Definitions
def P : ℝ × ℝ := (5, -2)
def line (x y : ℝ) : Prop := x - y + 5 = 0

-- Statement 
theorem symmetric_point (a b : ℝ) 
  (symmetric_condition1 : ∀ x y, line x y → (b + 2)/(a - 5) * 1 = -1)
  (symmetric_condition2 : ∀ x y, line x y → (a + 5)/2 - (b - 2)/2 + 5 = 0) :
  (a, b) = (-7, 10) :=
sorry

end symmetric_point_l87_87672


namespace avg_score_is_94_l87_87431

-- Define the math scores of the four children
def june_score : ℕ := 97
def patty_score : ℕ := 85
def josh_score : ℕ := 100
def henry_score : ℕ := 94

-- Define the total number of children
def num_children : ℕ := 4

-- Define the total score
def total_score : ℕ := june_score + patty_score + josh_score + henry_score

-- Define the average score
def avg_score : ℕ := total_score / num_children

-- The theorem we want to prove
theorem avg_score_is_94 : avg_score = 94 := by
  -- skipping the proof
  sorry

end avg_score_is_94_l87_87431


namespace evaluate_expression_l87_87589

theorem evaluate_expression : abs (abs (abs (-2 + 2) - 2) * 2) = 4 := 
by
  sorry

end evaluate_expression_l87_87589


namespace parabola_intersection_ratios_l87_87423

noncomputable def parabola_vertex_x1 (a b c : ℝ) := -b / (2 * a)
noncomputable def parabola_vertex_y1 (a b c : ℝ) := (4 * a * c - b^2) / (4 * a)
noncomputable def parabola_vertex_x2 (a d e : ℝ) := d / (2 * a)
noncomputable def parabola_vertex_y2 (a d e : ℝ) := (4 * a * e + d^2) / (4 * a)

theorem parabola_intersection_ratios
  (a b c d e : ℝ)
  (h1 : 144 * a + 12 * b + c = 21)
  (h2 : 784 * a + 28 * b + c = 3)
  (h3 : -144 * a + 12 * d + e = 21)
  (h4 : -784 * a + 28 * d + e = 3) :
  (parabola_vertex_x1 a b c + parabola_vertex_x2 a d e) / 
  (parabola_vertex_y1 a b c + parabola_vertex_y2 a d e) = 5 / 3 := by
  sorry

end parabola_intersection_ratios_l87_87423


namespace correct_parentheses_l87_87034

theorem correct_parentheses : (1 * 2 * 3 + 4) * 5 = 50 := by
  sorry

end correct_parentheses_l87_87034


namespace missing_jar_size_l87_87562

theorem missing_jar_size (total_ounces jars_16 jars_28 jars_unknown m n p: ℕ) (h1 : m = 3) (h2 : n = 3) (h3 : p = 3)
    (total_jars : m + n + p = 9)
    (total_peanut_butter : 16 * m + 28 * n + jars_unknown * p = 252)
    : jars_unknown = 40 := by
  sorry

end missing_jar_size_l87_87562


namespace new_person_weight_l87_87304

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (initial_person_weight : ℝ) 
  (weight_increase : ℝ) (final_person_weight : ℝ) : 
  avg_increase = 2.5 ∧ num_persons = 8 ∧ initial_person_weight = 65 ∧ 
  weight_increase = num_persons * avg_increase ∧ final_person_weight = initial_person_weight + weight_increase 
  → final_person_weight = 85 :=
by 
  intros h
  sorry

end new_person_weight_l87_87304


namespace elevation_angle_second_ship_l87_87566

-- Assume h is the height of the lighthouse.
def h : ℝ := 100

-- Assume d_total is the distance between the two ships.
def d_total : ℝ := 273.2050807568877

-- Assume θ₁ is the angle of elevation from the first ship.
def θ₁ : ℝ := 30

-- Assume θ₂ is the angle of elevation from the second ship.
def θ₂ : ℝ := 45

-- Prove that angle of elevation from the second ship is 45 degrees.
theorem elevation_angle_second_ship : θ₂ = 45 := by
  sorry

end elevation_angle_second_ship_l87_87566


namespace find_XY_length_l87_87978

variables (a b c : ℝ) -- sides of triangle ABC
variables (s : ℝ) -- semi-perimeter s = (a + b + c) / 2

-- Definition of similar triangles and perimeter condition
noncomputable def XY_length
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ) 
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) : ℝ :=
  s * a / (b + c) -- by the given solution

-- The theorem statement
theorem find_XY_length
  (a b c : ℝ) (s : ℝ) -- given sides and semi-perimeter
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ)
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) :
  XY = s * a / (b + c) :=
sorry -- proof


end find_XY_length_l87_87978


namespace pollutant_decay_l87_87298

noncomputable def p (t : ℝ) (p0 : ℝ) := p0 * 2^(-t / 30)

theorem pollutant_decay : 
  ∃ p0 : ℝ, p0 = 300 ∧ p 60 p0 = 75 * Real.log 2 := 
by
  sorry

end pollutant_decay_l87_87298


namespace arithmetic_to_geometric_progression_l87_87632

theorem arithmetic_to_geometric_progression (d : ℝ) (h : ∀ d, (4 + d) * (4 + d) = 7 * (22 + 2 * d)) :
  ∃ d, 7 + 2 * d = 3.752 :=
sorry

end arithmetic_to_geometric_progression_l87_87632


namespace general_term_of_geometric_sequence_l87_87232

theorem general_term_of_geometric_sequence 
  (positive_terms : ∀ n : ℕ, 0 < a_n) 
  (h1 : a_1 = 1) 
  (h2 : ∃ a : ℕ, a_2 = a + 1 ∧ a_3 = 2 * a + 5) : 
  ∃ q : ℕ, ∀ n : ℕ, a_n = q^(n-1) :=
by
  sorry

end general_term_of_geometric_sequence_l87_87232


namespace train_pass_man_in_16_seconds_l87_87100

noncomputable def speed_km_per_hr := 54
noncomputable def speed_m_per_s := (speed_km_per_hr * 1000) / 3600
noncomputable def time_to_pass_platform := 16
noncomputable def length_platform := 90.0072
noncomputable def length_train := speed_m_per_s * time_to_pass_platform
noncomputable def time_to_pass_man := length_train / speed_m_per_s

theorem train_pass_man_in_16_seconds :
  time_to_pass_man = 16 :=
by sorry

end train_pass_man_in_16_seconds_l87_87100


namespace find_common_chord_l87_87085

variable (x y : ℝ)

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 3*y = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 4*x + 2*y + 1 = 0
def common_chord (x y : ℝ) := 6*x + y - 1 = 0

theorem find_common_chord (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : common_chord x y :=
by
  sorry

end find_common_chord_l87_87085


namespace num_socks_in_machine_l87_87645

-- Definition of the number of people who played the match
def num_players : ℕ := 11

-- Definition of the number of socks per player
def socks_per_player : ℕ := 2

-- The goal is to prove that the total number of socks in the washing machine is 22
theorem num_socks_in_machine : num_players * socks_per_player = 22 :=
by
  sorry

end num_socks_in_machine_l87_87645


namespace gain_percent_l87_87467

theorem gain_percent (cp sp : ℝ) (h_cp : cp = 900) (h_sp : sp = 1080) :
    ((sp - cp) / cp) * 100 = 20 :=
by
    sorry

end gain_percent_l87_87467


namespace triangle_ABC_properties_l87_87023

theorem triangle_ABC_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (area_ABC : Real.sqrt 15 * 3 = 1/2 * b * c * Real.sin A)
  (cos_A : Real.cos A = -1/4)
  (b_minus_c : b - c = 2) :
  (a = 8 ∧ Real.sin C = Real.sqrt 15 / 8) ∧
  (Real.cos (2 * A + Real.pi / 6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16) := by
  sorry

end triangle_ABC_properties_l87_87023


namespace investment_principal_l87_87245

theorem investment_principal (A r : ℝ) (n t : ℕ) (P : ℝ) : 
  r = 0.07 → n = 4 → t = 5 → A = 60000 → 
  A = P * (1 + r / n)^(n * t) →
  P = 42409 :=
by
  sorry

end investment_principal_l87_87245


namespace graphene_scientific_notation_l87_87715

theorem graphene_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ (0.00000000034 : ℝ) = a * 10^n ∧ a = 3.4 ∧ n = -10 :=
sorry

end graphene_scientific_notation_l87_87715


namespace find_speed_of_stream_l87_87880

variable (b s : ℝ)

-- Equation derived from downstream condition
def downstream_equation := b + s = 24

-- Equation derived from upstream condition
def upstream_equation := b - s = 10

theorem find_speed_of_stream
  (b s : ℝ)
  (h1 : downstream_equation b s)
  (h2 : upstream_equation b s) :
  s = 7 := by
  -- placeholder for the proof
  sorry

end find_speed_of_stream_l87_87880


namespace harry_total_hours_l87_87588

variable (x h y : ℕ)

theorem harry_total_hours :
  ((h + 2 * y) = 42) → ∃ t, t = h + y :=
  by
    sorry -- Proof is omitted as per the instructions

end harry_total_hours_l87_87588


namespace union_complement_eq_l87_87132

def U : Set Nat := {0, 1, 2, 4, 6, 8}
def M : Set Nat := {0, 4, 6}
def N : Set Nat := {0, 1, 6}
def complement (u : Set α) (s : Set α) : Set α := {x ∈ u | x ∉ s}

theorem union_complement_eq :
  M ∪ (complement U N) = {0, 2, 4, 6, 8} :=
by sorry

end union_complement_eq_l87_87132


namespace john_annual_patients_l87_87866

-- Definitions for the various conditions
def first_hospital_patients_per_day := 20
def second_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 20 / 100)
def third_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 15 / 100)
def total_patients_per_day := first_hospital_patients_per_day + second_hospital_patients_per_day + third_hospital_patients_per_day
def workdays_per_week := 5
def total_patients_per_week := total_patients_per_day * workdays_per_week
def working_weeks_per_year := 50 - 2 -- considering 2 weeks of vacation
def total_patients_per_year := total_patients_per_week * working_weeks_per_year

-- The statement to prove
theorem john_annual_patients : total_patients_per_year = 16080 := by
  sorry

end john_annual_patients_l87_87866


namespace median_of_consecutive_integers_l87_87597

theorem median_of_consecutive_integers (a n : ℤ) (N : ℕ) (h1 : (a + (n - 1)) + (a + (N - n)) = 110) : 
  (2 * a + N - 1) / 2 = 55 := 
by {
  -- The proof goes here.
  sorry
}

end median_of_consecutive_integers_l87_87597


namespace domain_of_function_l87_87311

theorem domain_of_function :
  ∀ x : ℝ, ⌊x^2 - 8 * x + 18⌋ ≠ 0 :=
sorry

end domain_of_function_l87_87311


namespace print_time_l87_87338

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end print_time_l87_87338


namespace num_of_nickels_is_two_l87_87546

theorem num_of_nickels_is_two (d n : ℕ) 
    (h1 : 10 * d + 5 * n = 70) 
    (h2 : d + n = 8) : 
    n = 2 := 
by 
    sorry

end num_of_nickels_is_two_l87_87546


namespace total_pizza_slices_correct_l87_87742

-- Define the conditions
def num_pizzas : Nat := 3
def slices_per_first_two_pizzas : Nat := 8
def num_first_two_pizzas : Nat := 2
def slices_third_pizza : Nat := 12

-- Define the total slices based on conditions
def total_slices : Nat := slices_per_first_two_pizzas * num_first_two_pizzas + slices_third_pizza

-- The theorem to be proven
theorem total_pizza_slices_correct : total_slices = 28 := by
  sorry

end total_pizza_slices_correct_l87_87742


namespace large_monkey_doll_cost_l87_87388

theorem large_monkey_doll_cost :
  ∃ (L : ℝ), (300 / L - 300 / (L - 2) = 25) ∧ L > 0 := by
  sorry

end large_monkey_doll_cost_l87_87388


namespace solution_set_of_inequality_l87_87120

-- Definition of the inequality and its transformation
def inequality (x : ℝ) : Prop :=
  (x - 2) / (x + 1) ≤ 0

noncomputable def transformed_inequality (x : ℝ) : Prop :=
  (x + 1) * (x - 2) ≤ 0 ∧ x + 1 ≠ 0

-- Statement of the theorem
theorem solution_set_of_inequality :
  {x : ℝ | inequality x} = {x : ℝ | -1 < x ∧ x ≤ 2} := 
sorry

end solution_set_of_inequality_l87_87120


namespace solve_quadratic_equation_l87_87975

theorem solve_quadratic_equation (x : ℝ) :
    2 * x * (x - 5) = 3 * (5 - x) ↔ (x = 5 ∨ x = -3/2) :=
by
  sorry

end solve_quadratic_equation_l87_87975


namespace two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l87_87659

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ 2^n - 1) ↔ ∃ k : ℕ, n = 3 * k :=
by sorry

theorem two_pow_n_plus_one_not_div_by_seven (n : ℕ) : n > 0 → ¬(7 ∣ 2^n + 1) :=
by sorry

end two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l87_87659


namespace nine_b_equals_eighteen_l87_87367

theorem nine_b_equals_eighteen (a b : ℝ) (h1 : 6 * a + 3 * b = 0) (h2 : a = b - 3) : 9 * b = 18 :=
  sorry

end nine_b_equals_eighteen_l87_87367


namespace total_points_each_team_l87_87360

def score_touchdown := 7
def score_field_goal := 3
def score_safety := 2

def team_hawks_first_match_score := 3 * score_touchdown + 2 * score_field_goal + score_safety
def team_eagles_first_match_score := 5 * score_touchdown + 4 * score_field_goal
def team_hawks_second_match_score := 4 * score_touchdown + 3 * score_field_goal
def team_falcons_second_match_score := 6 * score_touchdown + 2 * score_safety

def total_score_hawks := team_hawks_first_match_score + team_hawks_second_match_score
def total_score_eagles := team_eagles_first_match_score
def total_score_falcons := team_falcons_second_match_score

theorem total_points_each_team :
  total_score_hawks = 66 ∧ total_score_eagles = 47 ∧ total_score_falcons = 46 :=
by
  unfold total_score_hawks team_hawks_first_match_score team_hawks_second_match_score
           total_score_eagles team_eagles_first_match_score
           total_score_falcons team_falcons_second_match_score
           score_touchdown score_field_goal score_safety
  sorry

end total_points_each_team_l87_87360


namespace find_value_of_k_l87_87772

noncomputable def value_of_k (m n : ℝ) : ℝ :=
  let p := 0.4
  let point1 := (m, n)
  let point2 := (m + 2, n + p)
  let k := 5
  k

theorem find_value_of_k (m n : ℝ) : value_of_k m n = 5 :=
sorry

end find_value_of_k_l87_87772


namespace total_pages_in_book_l87_87146

theorem total_pages_in_book (x : ℕ) : 
  (x - (x / 6 + 8) - ((5 * x / 6 - 8) / 5 + 10) - ((4 * x / 6 - 18) / 4 + 12) = 72) → 
  x = 195 :=
by
  sorry

end total_pages_in_book_l87_87146


namespace max_product_sum_1988_l87_87413

theorem max_product_sum_1988 :
  ∃ (n : ℕ) (a : ℕ), n + a = 1988 ∧ a = 1 ∧ n = 662 ∧ (3^n * 2^a) = 2 * 3^662 :=
by
  sorry

end max_product_sum_1988_l87_87413


namespace noel_baked_dozens_l87_87829

theorem noel_baked_dozens (total_students : ℕ) (percent_like_donuts : ℝ)
    (donuts_per_student : ℕ) (dozen : ℕ) (h_total_students : total_students = 30)
    (h_percent_like_donuts : percent_like_donuts = 0.80)
    (h_donuts_per_student : donuts_per_student = 2)
    (h_dozen : dozen = 12) :
    total_students * percent_like_donuts * donuts_per_student / dozen = 4 := 
by
  sorry

end noel_baked_dozens_l87_87829


namespace certain_number_is_3500_l87_87948

theorem certain_number_is_3500 :
  ∃ x : ℝ, x - (1000 / 20.50) = 3451.2195121951218 ∧ x = 3500 :=
by
  sorry

end certain_number_is_3500_l87_87948


namespace trapezium_distance_l87_87393

theorem trapezium_distance (h : ℝ) (a b A : ℝ) 
  (h_area : A = 95) (h_a : a = 20) (h_b : b = 18) :
  A = (1/2 * (a + b) * h) → h = 5 :=
by
  sorry

end trapezium_distance_l87_87393


namespace quadratic_trinomial_int_l87_87529

theorem quadratic_trinomial_int (a b c x : ℤ) (h : y = (x - a) * (x - 6) + 1) :
  ∃ (b c : ℤ), (x + b) * (x + c) = (x - 8) * (x - 6) + 1 :=
by
  sorry

end quadratic_trinomial_int_l87_87529


namespace range_of_a_l87_87450

-- Given definition of the function
def f (x a : ℝ) := abs (x - a)

-- Statement of the problem
theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < -1 → x₂ < -1 → f x₁ a ≤ f x₂ a) → a ≥ -1 :=
by
  sorry

end range_of_a_l87_87450


namespace solve_for_q_l87_87368

theorem solve_for_q (q : ℕ) : 16^4 = (8^3 / 2 : ℕ) * 2^(16 * q) → q = 1 / 2 :=
by
  sorry

end solve_for_q_l87_87368


namespace relationship_between_a_and_b_l87_87748

-- Definitions for the conditions
variables {a b : ℝ}

-- Main theorem statement
theorem relationship_between_a_and_b (h1 : |Real.log (1 / 4) / Real.log a| = Real.log (1 / 4) / Real.log a)
  (h2 : |Real.log a / Real.log b| = -Real.log a / Real.log b) :
  0 < a ∧ a < 1 ∧ 1 < b :=
by
  sorry

end relationship_between_a_and_b_l87_87748


namespace visits_365_days_l87_87741

theorem visits_365_days : 
  let alice_visits := 3
  let beatrix_visits := 4
  let claire_visits := 5
  let total_days := 365
  ∃ days_with_exactly_two_visits, days_with_exactly_two_visits = 54 :=
by
  sorry

end visits_365_days_l87_87741


namespace average_marks_of_all_students_l87_87523

theorem average_marks_of_all_students (n₁ n₂ a₁ a₂ : ℕ) (h₁ : n₁ = 30) (h₂ : a₁ = 40) (h₃ : n₂ = 50) (h₄ : a₂ = 80) :
  ((n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 65) :=
by
  sorry

end average_marks_of_all_students_l87_87523


namespace codys_grandmother_age_l87_87500

theorem codys_grandmother_age
  (cody_age : ℕ)
  (grandmother_multiplier : ℕ)
  (h_cody_age : cody_age = 14)
  (h_grandmother_multiplier : grandmother_multiplier = 6) :
  (cody_age * grandmother_multiplier = 84) :=
by
  sorry

end codys_grandmother_age_l87_87500


namespace remainder_500th_in_T_l87_87982

def sequence_T (n : ℕ) : ℕ := sorry -- Assume a definition for the sequence T where n represents the position and the sequence contains numbers having exactly 9 ones in their binary representation.

theorem remainder_500th_in_T :
  (sequence_T 500) % 500 = 191 := 
sorry

end remainder_500th_in_T_l87_87982


namespace length_of_BC_is_eight_l87_87369

theorem length_of_BC_is_eight (a : ℝ) (h_area : (1 / 2) * (2 * a) * a^2 = 64) : 2 * a = 8 := 
by { sorry }

end length_of_BC_is_eight_l87_87369


namespace find_R_when_S_7_l87_87216

-- Define the variables and equations in Lean
variables (R S g : ℕ)

-- The theorem statement based on the given conditions and desired conclusion
theorem find_R_when_S_7 (h1 : R = 2 * g * S + 3) (h2: R = 23) (h3 : S = 5) : (∃ g : ℕ, R = 2 * g * 7 + 3) :=
by {
  -- This part enforces the proof will be handled later
  sorry
}

end find_R_when_S_7_l87_87216


namespace arithmetic_sequence_iff_condition_l87_87854

-- Definitions: A sequence and the condition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_iff_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a ↔ (∀ n : ℕ, 2 * a (n + 1) = a n + a (n + 2)) :=
by
  -- Proof is omitted.
  sorry

end arithmetic_sequence_iff_condition_l87_87854


namespace product_not_divisible_by_201_l87_87721

theorem product_not_divisible_by_201 (a b : ℕ) (h₁ : a + b = 201) : ¬ (201 ∣ a * b) := sorry

end product_not_divisible_by_201_l87_87721


namespace garden_perimeter_is_56_l87_87083

-- Define the conditions
def garden_width : ℕ := 12
def playground_length : ℕ := 16
def playground_width : ℕ := 12
def playground_area : ℕ := playground_length * playground_width
def garden_length : ℕ := playground_area / garden_width
def garden_perimeter : ℕ := 2 * (garden_length + garden_width)

-- Statement to prove
theorem garden_perimeter_is_56 :
  garden_perimeter = 56 := by
sorry

end garden_perimeter_is_56_l87_87083


namespace rectangle_ratio_l87_87553

theorem rectangle_ratio (s : ℝ) (w h : ℝ) (h_cond : h = 3 * s) (w_cond : w = 2 * s) :
  h / w = 3 / 2 :=
by
  sorry

end rectangle_ratio_l87_87553


namespace unique_solution_l87_87571

noncomputable def solve_system (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ) (x1 x2 x3 : ℝ) : Prop :=
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) ∧
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) ∧
  (a31 * x1 + a32 * x2 + a33 * x3 = 0)

theorem unique_solution 
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ)
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0)
  (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  ∀ (x1 x2 x3 : ℝ), solve_system a11 a12 a13 a21 a22 a23 a31 a32 a33 x1 x2 x3 → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) :=
by
  sorry

end unique_solution_l87_87571


namespace problem_solution_l87_87734

noncomputable def M (a b c : ℝ) : ℝ := (1 - 1/a) * (1 - 1/b) * (1 - 1/c)

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 1) :
  M a b c ≤ -8 :=
sorry

end problem_solution_l87_87734


namespace series_sum_is_correct_l87_87251

noncomputable def series_sum : ℝ := ∑' k, 5^((2 : ℕ)^k) / (25^((2 : ℕ)^k) - 1)

theorem series_sum_is_correct : series_sum = 1 / (Real.sqrt 5 - 1) := 
by
  sorry

end series_sum_is_correct_l87_87251


namespace max_workers_l87_87415

-- Each worker produces 10 bricks a day and steals as many bricks per day as there are workers at the factory.
def worker_bricks_produced_per_day : ℕ := 10
def worker_bricks_stolen_per_day (n : ℕ) : ℕ := n

-- The factory must have at least 13 more bricks at the end of the day.
def factory_brick_surplus_requirement : ℕ := 13

-- Prove the maximum number of workers that can be hired so that the factory has at least 13 more bricks than at the beginning:
theorem max_workers
  (n : ℕ) -- Let \( n \) be the number of workers at the brick factory.
  (h : worker_bricks_produced_per_day * n - worker_bricks_stolen_per_day n + 13 ≥ factory_brick_surplus_requirement): 
  n = 8 := 
sorry

end max_workers_l87_87415


namespace average_marks_math_chem_l87_87783

variables (M P C : ℕ)

theorem average_marks_math_chem :
  (M + P = 20) → (C = P + 20) → (M + C) / 2 = 20 := 
by
  sorry

end average_marks_math_chem_l87_87783


namespace sarah_ellie_total_reflections_l87_87129

def sarah_tall_reflections : ℕ := 10
def sarah_wide_reflections : ℕ := 5
def sarah_narrow_reflections : ℕ := 8

def ellie_tall_reflections : ℕ := 6
def ellie_wide_reflections : ℕ := 3
def ellie_narrow_reflections : ℕ := 4

def tall_mirror_passages : ℕ := 3
def wide_mirror_passages : ℕ := 5
def narrow_mirror_passages : ℕ := 4

def total_reflections (sarah_tall sarah_wide sarah_narrow ellie_tall ellie_wide ellie_narrow
    tall_passages wide_passages narrow_passages : ℕ) : ℕ :=
  (sarah_tall * tall_passages + sarah_wide * wide_passages + sarah_narrow * narrow_passages) +
  (ellie_tall * tall_passages + ellie_wide * wide_passages + ellie_narrow * narrow_passages)

theorem sarah_ellie_total_reflections :
  total_reflections sarah_tall_reflections sarah_wide_reflections sarah_narrow_reflections
  ellie_tall_reflections ellie_wide_reflections ellie_narrow_reflections
  tall_mirror_passages wide_mirror_passages narrow_mirror_passages = 136 :=
by
  sorry

end sarah_ellie_total_reflections_l87_87129


namespace graph_symmetric_about_x_2_l87_87238

variables {D : Set ℝ} {f : ℝ → ℝ}

theorem graph_symmetric_about_x_2 (h : ∀ x ∈ D, f (x + 1) = f (-x + 3)) : 
  ∀ x ∈ D, f (x) = f (4 - x) :=
by
  sorry

end graph_symmetric_about_x_2_l87_87238


namespace max_sum_squares_of_sides_l87_87025

theorem max_sum_squares_of_sides
  (a : ℝ) (α : ℝ) 
  (hα1 : 0 < α) (hα2 : α < Real.pi / 2) : 
  ∃ b c : ℝ, b^2 + c^2 = a^2 / (1 - Real.cos α) := 
sorry

end max_sum_squares_of_sides_l87_87025


namespace chord_length_invalid_l87_87121

-- Define the circle radius
def radius : ℝ := 5

-- Define the maximum possible chord length in terms of the diameter
def max_chord_length (r : ℝ) : ℝ := 2 * r

-- The problem statement proving that 11 cannot be a chord length given the radius is 5
theorem chord_length_invalid : ¬ (11 ≤ max_chord_length radius) :=
by {
  sorry
}

end chord_length_invalid_l87_87121


namespace gallons_per_cubic_foot_l87_87152

theorem gallons_per_cubic_foot (mix_per_pound : ℝ) (capacity_cubic_feet : ℕ) (weight_per_gallon : ℝ)
    (price_per_tbs : ℝ) (total_cost : ℝ) (total_gallons : ℝ) :
  mix_per_pound = 1.5 →
  capacity_cubic_feet = 6 →
  weight_per_gallon = 8 →
  price_per_tbs = 0.5 →
  total_cost = 270 →
  total_gallons = total_cost / (price_per_tbs * mix_per_pound * weight_per_gallon) →
  total_gallons / capacity_cubic_feet = 7.5 :=
by
  intro h1 h2 h3 h4 h5 h6
  rw [h2, h6]
  sorry

end gallons_per_cubic_foot_l87_87152


namespace min_value_one_over_a_plus_nine_over_b_l87_87731

theorem min_value_one_over_a_plus_nine_over_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  16 ≤ (1 / a) + (9 / b) :=
sorry

end min_value_one_over_a_plus_nine_over_b_l87_87731


namespace values_of_b_for_real_root_l87_87598

noncomputable def polynomial_has_real_root (b : ℝ) : Prop :=
  ∃ x : ℝ, x^5 + b * x^4 - x^3 + b * x^2 - x + b = 0

theorem values_of_b_for_real_root :
  {b : ℝ | polynomial_has_real_root b} = {b : ℝ | b ≤ -1 ∨ b ≥ 1} :=
sorry

end values_of_b_for_real_root_l87_87598


namespace robotics_club_students_l87_87804

theorem robotics_club_students
  (total_students : ℕ)
  (cs_students : ℕ)
  (electronics_students : ℕ)
  (both_students : ℕ)
  (h1 : total_students = 80)
  (h2 : cs_students = 50)
  (h3 : electronics_students = 35)
  (h4 : both_students = 25) :
  total_students - (cs_students - both_students + electronics_students - both_students + both_students) = 20 :=
by
  sorry

end robotics_club_students_l87_87804


namespace quadratic_trinomial_form_l87_87761

noncomputable def quadratic_form (a b c : ℝ) (h : a ≠ 0) : Prop :=
  ∀ x : ℝ, 
    (a * (3.8 * x - 1)^2 + b * (3.8 * x - 1) + c) = (a * (-3.8 * x)^2 + b * (-3.8 * x) + c)

theorem quadratic_trinomial_form (a b c : ℝ) (h : a ≠ 0) : b = a → quadratic_form a b c h :=
by
  intro hba
  unfold quadratic_form
  intro x
  rw [hba]
  sorry

end quadratic_trinomial_form_l87_87761


namespace K_time_correct_l87_87418

open Real

noncomputable def K_speed : ℝ := sorry
noncomputable def M_speed : ℝ := K_speed - 1 / 2
noncomputable def K_time : ℝ := 45 / K_speed
noncomputable def M_time : ℝ := 45 / M_speed

theorem K_time_correct (K_speed_correct : 45 / K_speed - 45 / M_speed = 1 / 2) : K_time = 45 / K_speed :=
by
  sorry

end K_time_correct_l87_87418


namespace continuous_func_unique_l87_87288

theorem continuous_func_unique (f : ℝ → ℝ) (hf_cont : Continuous f)
  (hf_eqn : ∀ x : ℝ, f x + f (x^2) = 2) :
  ∀ x : ℝ, f x = 1 :=
by
  sorry

end continuous_func_unique_l87_87288


namespace sum_of_primes_lt_20_eq_77_l87_87057

/-- Define a predicate to check if a number is prime. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

/-- All prime numbers less than 20. -/
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

/-- Sum of the prime numbers less than 20. -/
noncomputable def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.sum

/-- Statement of the problem. -/
theorem sum_of_primes_lt_20_eq_77 : sum_primes_less_than_20 = 77 := 
  by
  sorry

end sum_of_primes_lt_20_eq_77_l87_87057


namespace equation_value_l87_87107

-- Define the expressions
def a := 10 + 3
def b := 7 - 5

-- State the theorem
theorem equation_value : a^2 + b^2 = 173 := by
  sorry

end equation_value_l87_87107


namespace circle_sum_condition_l87_87279

theorem circle_sum_condition (n : ℕ) (n_ge_1 : n ≥ 1)
  (x : Fin n → ℝ) (sum_x : (Finset.univ.sum x) = n - 1) :
  ∃ j : Fin n, ∀ k : ℕ, k ≥ 1 → k ≤ n → (Finset.range k).sum (fun i => x ⟨(j + i) % n, sorry⟩) ≥ k - 1 :=
sorry

end circle_sum_condition_l87_87279


namespace gingerbread_price_today_is_5_l87_87125

-- Given conditions
variables {x y a b k m : ℤ}

-- Price constraints
axiom price_constraint_yesterday : 9 * x + 7 * y < 100
axiom price_constraint_today1 : 9 * a + 7 * b > 100
axiom price_constraint_today2 : 2 * a + 11 * b < 100

-- Price change constraints
axiom price_change_gingerbread : a = x + k
axiom price_change_pastries : b = y + m
axiom gingerbread_change_range : |k| ≤ 1
axiom pastries_change_range : |m| ≤ 1

theorem gingerbread_price_today_is_5 : a = 5 :=
by
  sorry

end gingerbread_price_today_is_5_l87_87125


namespace tommy_number_of_nickels_l87_87164

theorem tommy_number_of_nickels
  (d p n q : ℕ)
  (h1 : d = p + 10)
  (h2 : n = 2 * d)
  (h3 : q = 4)
  (h4 : p = 10 * q) : n = 100 :=
sorry

end tommy_number_of_nickels_l87_87164


namespace find_root_equation_l87_87552

theorem find_root_equation : ∃ x : ℤ, x - (5 / (x - 4)) = 2 - (5 / (x - 4)) ∧ x = 2 :=
by
  sorry

end find_root_equation_l87_87552


namespace at_least_one_is_half_l87_87561

theorem at_least_one_is_half (x y z : ℝ) (h : x + y + z - 2 * (x * y + y * z + z * x) + 4 * x * y * z = 1 / 2) :
  x = 1 / 2 ∨ y = 1 / 2 ∨ z = 1 / 2 :=
sorry

end at_least_one_is_half_l87_87561


namespace min_value_expression_l87_87291

theorem min_value_expression (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  (x^2 + 8 * x * y + 25 * y^2 + 16 * y * z + 9 * z^2) ≥ 403 / 9 := by
  sorry

end min_value_expression_l87_87291


namespace passenger_speed_relative_forward_correct_l87_87389

-- Define the conditions
def train_speed : ℝ := 60     -- Train's speed in km/h
def passenger_speed_inside_train : ℝ := 3  -- Passenger's speed inside the train in km/h

-- Define the effective speed of the passenger relative to the railway track when moving forward
def passenger_speed_relative_forward (train_speed passenger_speed_inside_train : ℝ) : ℝ :=
  train_speed + passenger_speed_inside_train

-- Prove that the passenger's speed relative to the railway track is 63 km/h when moving forward
theorem passenger_speed_relative_forward_correct :
  passenger_speed_relative_forward train_speed passenger_speed_inside_train = 63 := by
  sorry

end passenger_speed_relative_forward_correct_l87_87389


namespace count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l87_87582

-- Statement for Question 1
theorem count_numbers_with_digit_7 :
  ∃ n, n = 19 ∧ (∀ k, (k < 100 → (k / 10 = 7 ∨ k % 10 = 7) ↔ k ≠ 77)) :=
sorry

-- Statement for Question 2
theorem count_numbers_divisible_by_3_or_5 :
  ∃ n, n = 47 ∧ (∀ k, (k < 100 → (k % 3 = 0 ∨ k % 5 = 0)) ↔ (k % 15 = 0)) :=
sorry

end count_numbers_with_digit_7_count_numbers_divisible_by_3_or_5_l87_87582


namespace pure_imaginary_z1_over_z2_l87_87542

theorem pure_imaginary_z1_over_z2 (b : Real) : 
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  (Complex.re ((z1 / z2) : Complex)) = 0 → b = -3 / 2 :=
by
  intros
  -- Conditions
  let z1 := (3 : Complex) - (b : Real) * Complex.I
  let z2 := (1 : Complex) - 2 * Complex.I
  -- Assuming that the real part of (z1 / z2) is zero
  have h : Complex.re (z1 / z2) = 0 := ‹_›
  -- Require to prove that b = -3 / 2
  sorry

end pure_imaginary_z1_over_z2_l87_87542


namespace Janet_initial_crayons_l87_87638

variable (Michelle_initial Janet_initial Michelle_final : ℕ)

theorem Janet_initial_crayons (h1 : Michelle_initial = 2) (h2 : Michelle_final = 4) (h3 : Michelle_final = Michelle_initial + Janet_initial) :
  Janet_initial = 2 :=
by
  sorry

end Janet_initial_crayons_l87_87638


namespace number_of_ways_to_arrange_BANANA_l87_87736

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l87_87736


namespace option_C_is_nonnegative_rational_l87_87836

def isNonNegativeRational (x : ℚ) : Prop :=
  x ≥ 0

theorem option_C_is_nonnegative_rational :
  isNonNegativeRational (-( - (4^2 : ℚ))) :=
by
  sorry

end option_C_is_nonnegative_rational_l87_87836


namespace greatest_integer_solution_l87_87088

theorem greatest_integer_solution :
  ∃ x : ℤ, (∀ y : ℤ, (6 * (y : ℝ)^2 + 5 * (y : ℝ) - 8) < (3 * (y : ℝ)^2 - 4 * (y : ℝ) + 1) → y ≤ x) 
  ∧ (6 * (x : ℝ)^2 + 5 * (x : ℝ) - 8) < (3 * (x : ℝ)^2 - 4 * (x : ℝ) + 1) ∧ x = 0 :=
by
  sorry

end greatest_integer_solution_l87_87088


namespace corrected_mean_l87_87013

theorem corrected_mean (n : ℕ) (incorrect_mean : ℝ) (incorrect_observation correct_observation : ℝ)
  (h_n : n = 50)
  (h_incorrect_mean : incorrect_mean = 30)
  (h_incorrect_observation : incorrect_observation = 23)
  (h_correct_observation : correct_observation = 48) :
  (incorrect_mean * n - incorrect_observation + correct_observation) / n = 30.5 :=
by
  sorry

end corrected_mean_l87_87013


namespace selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l87_87575

/-
  Conditions:
-/
def Group3 : ℕ := 18
def Group4 : ℕ := 12
def Group5 : ℕ := 6
def TotalParticipantsToSelect : ℕ := 12
def TotalFromGroups345 : ℕ := Group3 + Group4 + Group5

/-
  Questions:
  1. Prove that the number of people to be selected from each group using stratified sampling:
\ 2. Prove that the probability of selecting at least one of A or B from Group 5 is 3/5.
-/

theorem selection_count_Group3 : 
  (Group3 * TotalParticipantsToSelect / TotalFromGroups345) = 6 := 
  by sorry

theorem selection_count_Group4 : 
  (Group4 * TotalParticipantsToSelect / TotalFromGroups345) = 4 := 
  by sorry

theorem selection_count_Group5 : 
  (Group5 * TotalParticipantsToSelect / TotalFromGroups345) = 2 := 
  by sorry

noncomputable def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_A_or_B : 
  (combination 6 2 - combination 4 2) / combination 6 2 = 3 / 5 := 
  by sorry

end selection_count_Group3_selection_count_Group4_selection_count_Group5_probability_A_or_B_l87_87575


namespace functions_not_exist_l87_87409

theorem functions_not_exist :
  ¬ (∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), x ≠ y → |f x - f y| + |g x - g y| > 1) :=
by
  sorry

end functions_not_exist_l87_87409


namespace find_b_l87_87327

theorem find_b {a b : ℝ} (h₁ : 2 * 2 + b = 1 - 2 * a) (h₂ : -2 * 2 + b = -15 + 2 * a) : 
  b = -7 := sorry

end find_b_l87_87327


namespace simplify_vector_expression_l87_87487

-- Definitions for vectors
variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]

-- Defining the vectors
variables (AB CA BD CD : A)

-- A definition using the head-to-tail addition of vectors.
def vector_add (v1 v2 : A) : A := v1 + v2

-- Statement to prove
theorem simplify_vector_expression :
  vector_add (vector_add AB CA) BD = CD :=
sorry

end simplify_vector_expression_l87_87487


namespace probability_of_rolling_perfect_square_l87_87900

theorem probability_of_rolling_perfect_square :
  (3 / 12 : ℚ) = 1 / 4 :=
by
  sorry

end probability_of_rolling_perfect_square_l87_87900


namespace candy_bar_cost_is_7_l87_87867

-- Define the conditions
def chocolate_cost : Nat := 3
def candy_additional_cost : Nat := 4

-- Define the expression for the cost of the candy bar
def candy_cost : Nat := chocolate_cost + candy_additional_cost

-- State the theorem to prove the cost of the candy bar
theorem candy_bar_cost_is_7 : candy_cost = 7 :=
by
  sorry

end candy_bar_cost_is_7_l87_87867


namespace necessarily_positive_y_plus_xsq_l87_87927

theorem necessarily_positive_y_plus_xsq {x y z : ℝ} 
  (hx : 0 < x ∧ x < 2) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  y + x^2 > 0 :=
sorry

end necessarily_positive_y_plus_xsq_l87_87927


namespace total_cards_in_stack_l87_87243

theorem total_cards_in_stack (n : ℕ) (H1: 252 ≤ 2 * n) (H2 : (2 * n) % 2 = 0)
                             (H3 : ∀ k : ℕ, k ≤ 2 * n → (if k % 2 = 0 then k / 2 else (k + 1) / 2) * 2 = k) :
  2 * n = 504 := sorry

end total_cards_in_stack_l87_87243


namespace inequality_proof_l87_87769

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) : (1 / x) < (1 / y) :=
by
  sorry

end inequality_proof_l87_87769


namespace nesting_doll_height_l87_87698

variable (H₀ : ℝ) (n : ℕ)

theorem nesting_doll_height (H₀ : ℝ) (Hₙ : ℝ) (H₁ : H₀ = 243) (H₂ : ∀ n : ℕ, Hₙ = H₀ * (2 / 3) ^ n) (H₃ : Hₙ = 32) : n = 4 :=
by
  sorry

end nesting_doll_height_l87_87698


namespace find_a_l87_87110

-- Define the conditions and the proof goal
theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h_eq : a + a⁻¹ = 5/2) :
  a = 1/2 :=
by
  sorry

end find_a_l87_87110


namespace percentage_calculation_l87_87144

theorem percentage_calculation (amount : ℝ) (percentage : ℝ) (res : ℝ) :
  amount = 400 → percentage = 0.25 → res = amount * percentage → res = 100 := by
  intro h_amount h_percentage h_res
  rw [h_amount, h_percentage] at h_res
  norm_num at h_res
  exact h_res

end percentage_calculation_l87_87144


namespace completion_days_l87_87479

theorem completion_days (D : ℝ) :
  (1 / D + 1 / 9 = 1 / 3.2142857142857144) → D = 5 := by
  sorry

end completion_days_l87_87479


namespace remainder_and_division_l87_87130

theorem remainder_and_division (x y : ℕ) (h1 : x % y = 8) (h2 : (x / y : ℝ) = 76.4) : y = 20 :=
sorry

end remainder_and_division_l87_87130


namespace sum_of_reciprocals_l87_87782

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 :=
by
  sorry

end sum_of_reciprocals_l87_87782


namespace sum_of_areas_of_tangent_circles_l87_87988

theorem sum_of_areas_of_tangent_circles :
  ∀ (a b c : ℝ), 
    a + b = 5 →
    a + c = 12 →
    b + c = 13 →
    π * (a^2 + b^2 + c^2) = 113 * π :=
by
  intros a b c h₁ h₂ h₃
  sorry

end sum_of_areas_of_tangent_circles_l87_87988


namespace quadratic_non_negative_iff_a_in_range_l87_87678

theorem quadratic_non_negative_iff_a_in_range :
  (∀ x : ℝ, x^2 + (a - 2) * x + 1/4 ≥ 0) ↔ 1 ≤ a ∧ a ≤ 3 :=
sorry

end quadratic_non_negative_iff_a_in_range_l87_87678


namespace union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l87_87102

section
  def A : Set ℝ := {x : ℝ | ∃ q : ℚ, x = q}
  def B : Set ℝ := {x : ℝ | ¬ ∃ q : ℚ, x = q}

  theorem union_rational_irrational_is_real : A ∪ B = Set.univ :=
  by
    sorry

  theorem intersection_rational_irrational_is_empty : A ∩ B = ∅ :=
  by
    sorry
end

end union_rational_irrational_is_real_intersection_rational_irrational_is_empty_l87_87102


namespace find_divisor_l87_87416

-- Define the conditions as hypotheses and the main problem as a theorem
theorem find_divisor (x y : ℕ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 6) / y = 6) : y = 8 := sorry

end find_divisor_l87_87416


namespace fruits_turned_yellow_on_friday_l87_87777

theorem fruits_turned_yellow_on_friday :
  ∃ (F : ℕ), F + 2*F = 6 ∧ 14 - F - 2*F = 8 :=
by
  existsi 2
  sorry

end fruits_turned_yellow_on_friday_l87_87777


namespace infinitely_many_triples_no_triples_l87_87336

theorem infinitely_many_triples :
  ∃ (m n p : ℕ), ∃ (k : ℕ), m > 0 ∧ n > 0 ∧ p > 0 ∧ 4 * m * n - m - n = p ^ 2 - 1 := 
sorry

theorem no_triples :
  ¬∃ (m n p : ℕ), m > 0 ∧ n > 0 ∧ p > 0 ∧ 4 * m * n - m - n = p ^ 2 := 
sorry

end infinitely_many_triples_no_triples_l87_87336


namespace how_many_more_cups_of_sugar_l87_87353

def required_sugar : ℕ := 11
def required_flour : ℕ := 9
def added_flour : ℕ := 12
def added_sugar : ℕ := 10

theorem how_many_more_cups_of_sugar :
  required_sugar - added_sugar = 1 :=
by
  sorry

end how_many_more_cups_of_sugar_l87_87353


namespace students_not_next_each_other_l87_87749

open Nat

theorem students_not_next_each_other (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 5) (h2 : k = 2) (h3 : m = 3)
  (h4 : ∀ (A B : ℕ), A ≠ B) : 
  ∃ (total : ℕ), total = 3! * (choose (5-3+1) 2) := 
by
  sorry

end students_not_next_each_other_l87_87749


namespace erik_ate_more_pie_l87_87493

theorem erik_ate_more_pie :
  let erik_pies := 0.67
  let frank_pies := 0.33
  erik_pies - frank_pies = 0.34 :=
by
  sorry

end erik_ate_more_pie_l87_87493


namespace gain_per_year_is_200_l87_87259

noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem gain_per_year_is_200 :
  let borrowed_principal := 5000
  let borrowing_rate := 4
  let borrowing_time := 2
  let lent_principal := 5000
  let lending_rate := 8
  let lending_time := 2

  let interest_paid := simple_interest borrowed_principal borrowing_rate borrowing_time
  let interest_earned := simple_interest lent_principal lending_rate lending_time

  let total_gain := interest_earned - interest_paid
  let gain_per_year := total_gain / 2

  gain_per_year = 200 := by
  sorry

end gain_per_year_is_200_l87_87259


namespace min_value_is_five_l87_87869

noncomputable def min_value (x y : ℝ) : ℝ :=
  if x + 3 * y = 5 * x * y then 3 * x + 4 * y else 0

theorem min_value_is_five {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : min_value x y = 5 :=
by
  sorry

end min_value_is_five_l87_87869


namespace problem1_problem2_l87_87005

variable (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)

theorem problem1 : 
  (a * b + a + b + 1) * (a * b + a * c + b * c + c ^ 2) ≥ 16 * a * b * c := 
by sorry

theorem problem2 : 
  (b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3 := 
by sorry

end problem1_problem2_l87_87005


namespace q_can_complete_work_in_30_days_l87_87387

theorem q_can_complete_work_in_30_days (W_p W_q W_r : ℝ)
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = 1/10)
  (h3 : W_r = 1/30) :
  1 / W_q = 30 :=
by
  -- Note: You can add proof here, but it's not required in the task.
  sorry

end q_can_complete_work_in_30_days_l87_87387


namespace problem1_problem2_l87_87364

-- Proof problem 1
theorem problem1 : (-3)^2 / 3 + abs (-7) + 3 * (-1/3) = 3 :=
by
  sorry

-- Proof problem 2
theorem problem2 : (-1) ^ 2022 - ( (-1/4) - (-1/3) ) / (-1/12) = 2 :=
by
  sorry

end problem1_problem2_l87_87364


namespace joan_games_l87_87630

theorem joan_games (last_year_games this_year_games total_games : ℕ)
  (h1 : last_year_games = 9)
  (h2 : total_games = 13)
  : this_year_games = total_games - last_year_games → this_year_games = 4 := 
by
  intros h
  rw [h1, h2] at h
  exact h

end joan_games_l87_87630


namespace triangle_inequality_l87_87330

theorem triangle_inequality (a b c p q r : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum_zero : p + q + r = 0) : 
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
sorry

end triangle_inequality_l87_87330


namespace blocks_left_l87_87949

def blocks_initial := 78
def blocks_used := 19

theorem blocks_left : blocks_initial - blocks_used = 59 :=
by
  -- Solution is not required here, so we add a sorry placeholder.
  sorry

end blocks_left_l87_87949


namespace quadratic_roots_real_and_equal_l87_87710

open Real

theorem quadratic_roots_real_and_equal :
  ∀ (x : ℝ), x^2 - 4 * x * sqrt 2 + 8 = 0 → ∃ r : ℝ, x = r :=
by
  intro x
  sorry

end quadratic_roots_real_and_equal_l87_87710


namespace three_digit_number_count_correct_l87_87459

def number_of_three_digit_numbers_with_repetition (digit_count : ℕ) (positions : ℕ) : ℕ :=
  let choices_for_repeated_digit := 5  -- 5 choices for repeated digit
  let ways_to_place_repeated_digit := 3 -- 3 ways to choose positions
  let choices_for_remaining_digit := 4 -- 4 choices for the remaining digit
  choices_for_repeated_digit * ways_to_place_repeated_digit * choices_for_remaining_digit

theorem three_digit_number_count_correct :
  number_of_three_digit_numbers_with_repetition 5 3 = 60 := 
sorry

end three_digit_number_count_correct_l87_87459


namespace test_score_after_preparation_l87_87346

-- Define the conditions in Lean 4
def score (k t : ℝ) : ℝ := k * t^2

theorem test_score_after_preparation (k t : ℝ)
    (h1 : score k 2 = 90) (h2 : k = 22.5) :
    score k 3 = 202.5 :=
by
  sorry

end test_score_after_preparation_l87_87346


namespace candy_received_l87_87280

theorem candy_received (pieces_eaten : ℕ) (piles : ℕ) (pieces_per_pile : ℕ) 
  (h_eaten : pieces_eaten = 12) (h_piles : piles = 4) (h_pieces_per_pile : pieces_per_pile = 5) :
  pieces_eaten + piles * pieces_per_pile = 32 := 
by
  sorry

end candy_received_l87_87280


namespace anna_clara_age_l87_87868

theorem anna_clara_age :
  ∃ x : ℕ, (54 - x) * 3 = 80 - x ∧ x = 41 :=
by
  sorry

end anna_clara_age_l87_87868


namespace jessies_weight_after_first_week_l87_87090

-- Definitions from the conditions
def initial_weight : ℕ := 92
def first_week_weight_loss : ℕ := 56

-- The theorem statement
theorem jessies_weight_after_first_week : initial_weight - first_week_weight_loss = 36 := by
  -- Skip the proof
  sorry

end jessies_weight_after_first_week_l87_87090


namespace dan_initial_amount_l87_87876

theorem dan_initial_amount (left_amount : ℕ) (candy_cost : ℕ) : left_amount = 3 ∧ candy_cost = 2 → left_amount + candy_cost = 5 :=
by
  sorry

end dan_initial_amount_l87_87876


namespace length_of_de_equals_eight_l87_87217

theorem length_of_de_equals_eight
  (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (bc : c - b = 3 * (d - c))
  (ab : b - a = 5)
  (ac : c - a = 11)
  (ae : e - a = 21) :
  e - d = 8 := by
  sorry

end length_of_de_equals_eight_l87_87217


namespace coordinates_with_respect_to_origin_l87_87421

theorem coordinates_with_respect_to_origin (P : ℝ × ℝ) (h : P = (2, -3)) : P = (2, -3) :=
by
  sorry

end coordinates_with_respect_to_origin_l87_87421


namespace prove_inequality_l87_87644

noncomputable def problem_statement (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) : Prop :=
  (1 / (p^n + q^n + 1)) + (1 / (q^n + r^n + 1)) + (1 / (r^n + p^n + 1)) ≤ 1

theorem prove_inequality (p q r : ℝ) (n : ℕ) (h_pqr : p * q * r = 1) (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r) : 
  problem_statement p q r n h_pqr :=
by
  sorry

end prove_inequality_l87_87644


namespace track_length_l87_87832

theorem track_length (L : ℝ)
  (h_brenda_first_meeting : ∃ (brenda_run1: ℝ), brenda_run1 = 100)
  (h_sally_first_meeting : ∃ (sally_run1: ℝ), sally_run1 = L/2 - 100)
  (h_brenda_second_meeting : ∃ (brenda_run2: ℝ), brenda_run2 = L - 100)
  (h_sally_second_meeting : ∃ (sally_run2: ℝ), sally_run2 = sally_run1 + 100)
  (h_meeting_total : brenda_run2 + sally_run2 = L) :
  L = 200 :=
by
  sorry

end track_length_l87_87832


namespace exists_centrally_symmetric_inscribed_convex_hexagon_l87_87189

-- Definition of a convex polygon with vertices
def convex_polygon (W : Type) : Prop := sorry

-- Definition of the unit area condition
def has_unit_area (W : Type) : Prop := sorry

-- Definition of being centrally symmetric
def is_centrally_symmetric (V : Type) : Prop := sorry

-- Definition of being inscribed
def is_inscribed_polygon (V W : Type) : Prop := sorry

-- Definition of a convex hexagon
def convex_hexagon (V : Type) : Prop := sorry

-- Main theorem statement
theorem exists_centrally_symmetric_inscribed_convex_hexagon (W : Type) 
  (hW_convex : convex_polygon W) (hW_area : has_unit_area W) : 
  ∃ V : Type, convex_hexagon V ∧ is_centrally_symmetric V ∧ is_inscribed_polygon V W ∧ sorry :=
  sorry

end exists_centrally_symmetric_inscribed_convex_hexagon_l87_87189


namespace jimmy_income_l87_87824

variable (J : ℝ)

def rebecca_income : ℝ := 15000
def income_increase : ℝ := 3000
def rebecca_income_after_increase : ℝ := rebecca_income + income_increase
def combined_income : ℝ := 2 * rebecca_income_after_increase

theorem jimmy_income (h : rebecca_income_after_increase + J = combined_income) : 
  J = 18000 := by
  sorry

end jimmy_income_l87_87824


namespace find_g3_l87_87626

variable {α : Type*} [Field α]

-- Define the function g
noncomputable def g (x : α) : α := sorry

-- Define the condition as a hypothesis
axiom condition (x : α) (hx : x ≠ 0) : 2 * g (1 / x) + 3 * g x / x = 2 * x ^ 2

-- State what needs to be proven
theorem find_g3 : g 3 = 242 / 15 := by
  sorry

end find_g3_l87_87626


namespace joe_paint_problem_l87_87264

theorem joe_paint_problem (f : ℝ) (h₁ : 360 * f + (1 / 6) * (360 - 360 * f) = 135) : f = 1 / 4 := 
by
  sorry

end joe_paint_problem_l87_87264


namespace general_admission_price_l87_87341

theorem general_admission_price :
  ∃ x : ℝ,
    ∃ G V : ℕ,
      VIP_price = 45 ∧ Total_tickets_sold = 320 ∧ Total_revenue = 7500 ∧ VIP_tickets_less = 276 ∧
      G + V = Total_tickets_sold ∧ V = G - VIP_tickets_less ∧ 45 * V + x * G = Total_revenue ∧ x = 21.85 :=
sorry

end general_admission_price_l87_87341


namespace contrapositive_proposition_l87_87825

theorem contrapositive_proposition (x : ℝ) : 
  (x^2 = 1 → (x = 1 ∨ x = -1)) ↔ ((x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1) :=
by
  sorry

end contrapositive_proposition_l87_87825


namespace distance_between_closest_points_l87_87732

noncomputable def distance_closest_points (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  (Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2) - r1 - r2)

theorem distance_between_closest_points :
  distance_closest_points (4, 4) (20, 12) 4 12 = 4 * Real.sqrt 20 - 16 :=
by
  sorry

end distance_between_closest_points_l87_87732


namespace rectangular_sheet_integer_side_l87_87411

theorem rectangular_sheet_integer_side
  (a b : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_cut_a : ∀ x, x ≤ a → ∃ n : ℕ, x = n ∨ x = n + 1)
  (h_cut_b : ∀ y, y ≤ b → ∃ n : ℕ, y = n ∨ y = n + 1) :
  ∃ n m : ℕ, a = n ∨ b = m := 
sorry

end rectangular_sheet_integer_side_l87_87411


namespace intersection_of_A_and_B_l87_87990

-- Definitions for the sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4}

-- Proof statement
theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l87_87990


namespace beads_per_bracelet_l87_87848

def beads_bella_has : Nat := 36
def beads_bella_needs : Nat := 12
def total_bracelets : Nat := 6

theorem beads_per_bracelet : (beads_bella_has + beads_bella_needs) / total_bracelets = 8 :=
by
  sorry

end beads_per_bracelet_l87_87848


namespace merchant_marked_price_percent_l87_87631

theorem merchant_marked_price_percent (L : ℝ) (hL : L = 100) (purchase_price : ℝ) (h1 : purchase_price = L * 0.70) (x : ℝ)
  (selling_price : ℝ) (h2 : selling_price = x * 0.75) :
  (selling_price - purchase_price) / selling_price = 0.30 → x = 133.33 :=
by
  sorry

end merchant_marked_price_percent_l87_87631


namespace score_order_l87_87604

variable (A B C D : ℕ)

-- Condition 1: B + D = A + C
axiom h1 : B + D = A + C
-- Condition 2: A + B > C + D + 10
axiom h2 : A + B > C + D + 10
-- Condition 3: D > B + C + 20
axiom h3 : D > B + C + 20
-- Condition 4: A + B + C + D = 200
axiom h4 : A + B + C + D = 200

-- Question to prove: Order is Donna > Alice > Brian > Cindy
theorem score_order : D > A ∧ A > B ∧ B > C :=
by
  sorry

end score_order_l87_87604


namespace customerPaidPercentGreater_l87_87226

-- Definitions for the conditions
def costOfManufacture (C : ℝ) : ℝ := C
def designerPrice (C : ℝ) : ℝ := C * 1.40
def retailerTaxedPrice (C : ℝ) : ℝ := (C * 1.40) * 1.05
def customerInitialPrice (C : ℝ) : ℝ := ((C * 1.40) * 1.05) * 1.10
def customerFinalPrice (C : ℝ) : ℝ := (((C * 1.40) * 1.05) * 1.10) * 0.90

-- The theorem statement
theorem customerPaidPercentGreater (C : ℝ) (hC : 0 < C) : 
    (customerFinalPrice C - costOfManufacture C) / costOfManufacture C * 100 = 45.53 := by 
  sorry

end customerPaidPercentGreater_l87_87226


namespace telescope_visual_range_increase_l87_87634

theorem telescope_visual_range_increase (original_range : ℝ) (increase_percent : ℝ) 
(h1 : original_range = 100) (h2 : increase_percent = 0.50) : 
original_range + (increase_percent * original_range) = 150 := 
sorry

end telescope_visual_range_increase_l87_87634


namespace number_of_penguins_l87_87879

-- Define the number of animals and zookeepers
def zebras : ℕ := 22
def tigers : ℕ := 8
def zookeepers : ℕ := 12
def headsLessThanFeetBy : ℕ := 132

-- Define the theorem to prove the number of penguins (P)
theorem number_of_penguins (P : ℕ) (H : P + zebras + tigers + zookeepers + headsLessThanFeetBy = 4 * P + 4 * zebras + 4 * tigers + 2 * zookeepers) : P = 10 :=
by
  sorry

end number_of_penguins_l87_87879


namespace geometric_sequence_sum_six_l87_87079

theorem geometric_sequence_sum_six (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) 
  (h1 : 0 < q)
  (h2 : a 1 = 1)
  (h3 : a 3 * a 5 = 64)
  (h4 : ∀ n, a n = a 1 * q^(n-1))
  (h5 : ∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) :
  S 6 = 63 := 
sorry

end geometric_sequence_sum_six_l87_87079


namespace kids_with_red_hair_l87_87200

theorem kids_with_red_hair (total_kids : ℕ) (ratio_red ratio_blonde ratio_black : ℕ) 
  (h_ratio : ratio_red + ratio_blonde + ratio_black = 16) (h_total : total_kids = 48) :
  (total_kids / (ratio_red + ratio_blonde + ratio_black)) * ratio_red = 9 :=
by
  sorry

end kids_with_red_hair_l87_87200


namespace max_groups_l87_87911

theorem max_groups (cards : ℕ) (sum_group : ℕ) (c5 c2 c1 : ℕ) (cond1 : cards = 600) (cond2 : c5 = 200)
  (cond3 : c2 = 200) (cond4 : c1 = 200) (cond5 : sum_group = 9) :
  ∃ max_g : ℕ, max_g = 100 :=
by
  sorry

end max_groups_l87_87911


namespace angle_of_skew_lines_in_range_l87_87403

noncomputable def angle_between_skew_lines (θ : ℝ) (θ_range : 0 < θ ∧ θ ≤ 90) : Prop :=
  θ ∈ (Set.Ioc 0 90)

-- We assume the existence of such an angle θ formed by two skew lines
theorem angle_of_skew_lines_in_range (θ : ℝ) (h_skew : true) : angle_between_skew_lines θ (⟨sorry, sorry⟩) :=
  sorry

end angle_of_skew_lines_in_range_l87_87403


namespace red_lettuce_cost_l87_87510

-- Define the known conditions
def cost_per_pound : Nat := 2
def total_pounds : Nat := 7
def cost_green_lettuce : Nat := 8

-- Define the total cost calculation
def total_cost : Nat := total_pounds * cost_per_pound
def cost_red_lettuce : Nat := total_cost - cost_green_lettuce

-- Statement to prove: cost_red_lettuce = 6
theorem red_lettuce_cost :
  cost_red_lettuce = 6 :=
by
  sorry

end red_lettuce_cost_l87_87510


namespace arvin_fifth_day_running_distance_l87_87058

theorem arvin_fifth_day_running_distance (total_km : ℕ) (first_day_km : ℕ) (increment : ℕ) (days : ℕ) 
  (h1 : total_km = 20) (h2 : first_day_km = 2) (h3 : increment = 1) (h4 : days = 5) : 
  first_day_km + (increment * (days - 1)) = 6 :=
by
  sorry

end arvin_fifth_day_running_distance_l87_87058


namespace minimal_fraction_difference_l87_87098

theorem minimal_fraction_difference (p q : ℕ) (hp : 0 < p) (hq : 0 < q) 
  (h1 : 3 / 5 < p / q) (h2 : p / q < 2 / 3) (hmin: ∀ r s : ℕ, (3 / 5 < r / s ∧ r / s < 2 / 3 ∧ s < q) → false) :
  q - p = 11 := 
sorry

end minimal_fraction_difference_l87_87098


namespace Gage_skating_minutes_l87_87970

theorem Gage_skating_minutes (d1 d2 d3 : ℕ) (m1 m2 : ℕ) (avg : ℕ) (h1 : d1 = 6) (h2 : d2 = 4) (h3 : d3 = 1) (h4 : m1 = 80) (h5 : m2 = 105) (h6 : avg = 95) : 
  (d1 * m1 + d2 * m2 + d3 * x) / (d1 + d2 + d3) = avg ↔ x = 145 := 
by 
  sorry

end Gage_skating_minutes_l87_87970


namespace not_perfect_square_l87_87297

theorem not_perfect_square (p : ℕ) (hp : Nat.Prime p) : ¬ ∃ t : ℕ, 7 * p + 3^p - 4 = t^2 :=
sorry

end not_perfect_square_l87_87297


namespace max_diff_x_y_l87_87263

theorem max_diff_x_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : 
  x - y ≤ Real.sqrt (4 / 3) := 
by
  sorry

end max_diff_x_y_l87_87263


namespace find_a_range_find_value_x1_x2_l87_87003

noncomputable def quadratic_equation_roots_and_discriminant (a : ℝ) :=
  ∃ x1 x2 : ℝ, 
      (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
      (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧
      (x1 ≠ x2) ∧ 
      (∀ Δ > 0, Δ = 9 - 8 * a - 4)

theorem find_a_range (a : ℝ) : 
  (quadratic_equation_roots_and_discriminant a) → a < 5 / 8 :=
sorry

theorem find_value_x1_x2 (a : ℤ) (h : a = 0) (x1 x2 : ℝ) :
  (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
  (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧ 
  (x1 + x2 = 3) ∧ 
  (x1 * x2 = 1) → 
  (x1^2 * x2 + x1 * x2^2 = 3) :=
sorry

end find_a_range_find_value_x1_x2_l87_87003


namespace work_completion_time_l87_87802

theorem work_completion_time
  (W : ℝ) -- Total work
  (p_rate : ℝ := W / 40) -- p's work rate
  (q_rate : ℝ := W / 24) -- q's work rate
  (work_done_by_p_alone : ℝ := 8 * p_rate) -- Work done by p in first 8 days
  (remaining_work : ℝ := W - work_done_by_p_alone) -- Remaining work after 8 days
  (combined_rate : ℝ := p_rate + q_rate) -- Combined work rate of p and q
  (time_to_complete_remaining_work : ℝ := remaining_work / combined_rate) -- Time to complete remaining work
  : (8 + time_to_complete_remaining_work) = 20 :=
by
  sorry

end work_completion_time_l87_87802


namespace heartsuit_value_l87_87531

def heartsuit (x y : ℝ) := 4 * x + 6 * y

theorem heartsuit_value : heartsuit 3 4 = 36 := by
  sorry

end heartsuit_value_l87_87531


namespace Jon_regular_bottle_size_is_16oz_l87_87000

noncomputable def Jon_bottle_size (x : ℝ) : Prop :=
  let daily_intake := 4 * x + 2 * 1.25 * x
  let weekly_intake := 7 * daily_intake
  weekly_intake = 728

theorem Jon_regular_bottle_size_is_16oz : ∃ x : ℝ, Jon_bottle_size x ∧ x = 16 :=
by
  use 16
  sorry

end Jon_regular_bottle_size_is_16oz_l87_87000


namespace inequality_proof_l87_87229

open Real

-- Given conditions
variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1)

-- Goal to prove
theorem inequality_proof : 
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) :=
sorry

end inequality_proof_l87_87229


namespace purple_balls_correct_l87_87590

-- Define the total number of balls and individual counts
def total_balls : ℕ := 100
def white_balls : ℕ := 20
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def red_balls : ℕ := 37

-- Probability that a ball chosen is neither red nor purple
def prob_neither_red_nor_purple : ℚ := 0.6

-- The number of purple balls to be proven
def purple_balls : ℕ := 3

-- The condition used for the proof
def condition : Prop := prob_neither_red_nor_purple = (white_balls + green_balls + yellow_balls) / total_balls

-- The proof problem statement
theorem purple_balls_correct (h : condition) : 
  ∃ P : ℕ, P = purple_balls ∧ P + red_balls = total_balls - (white_balls + green_balls + yellow_balls) :=
by
  have P := total_balls - (white_balls + green_balls + yellow_balls + red_balls)
  existsi P
  sorry

end purple_balls_correct_l87_87590


namespace max_wooden_pencils_l87_87530

theorem max_wooden_pencils (m w : ℕ) (p : ℕ) (h1 : m + w = 72) (h2 : m = w + p) (hp : Nat.Prime p) : w = 35 :=
by
  sorry

end max_wooden_pencils_l87_87530


namespace inverse_proportion_first_third_quadrant_l87_87004

theorem inverse_proportion_first_third_quadrant (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (2 - k) / x > 0) ∧ (x < 0 → (2 - k) / x < 0))) → k < 2 :=
by
  sorry

end inverse_proportion_first_third_quadrant_l87_87004


namespace percentage_reduction_in_price_l87_87704

-- Definitions based on conditions
def original_price (P : ℝ) (X : ℝ) := P * X
def reduced_price (R : ℝ) (X : ℝ) := R * (X + 5)

-- Theorem statement based on the problem to prove
theorem percentage_reduction_in_price
  (R : ℝ) (H1 : R = 55)
  (H2 : original_price P X = 1100)
  (H3 : reduced_price R X = 1100) :
  ((P - R) / P) * 100 = 25 :=
by
  sorry

end percentage_reduction_in_price_l87_87704


namespace vacation_cost_l87_87676

theorem vacation_cost (C P : ℕ) 
    (h1 : C = 5 * P)
    (h2 : C = 7 * (P - 40))
    (h3 : C = 8 * (P - 60)) : C = 700 := 
by 
    sorry

end vacation_cost_l87_87676


namespace pascal_row_10_sum_l87_87489

-- Define the function that represents the sum of Row n in Pascal's Triangle
def pascal_row_sum (n : ℕ) : ℕ := 2^n

-- State the theorem to be proven
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 :=
by
  -- Proof is omitted
  sorry

end pascal_row_10_sum_l87_87489


namespace polynomial_sum_l87_87440

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum :
  ∀ x : ℝ, f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l87_87440


namespace dagger_simplified_l87_87308

def dagger (m n p q : ℚ) : ℚ := (m^2) * p * (q / n)

theorem dagger_simplified :
  dagger (5:ℚ) (9:ℚ) (4:ℚ) (6:ℚ) = (200:ℚ) / (3:ℚ) :=
by
  sorry

end dagger_simplified_l87_87308


namespace sequence_geometric_sum_bn_l87_87284

theorem sequence_geometric (a : ℕ → ℕ) (h_recurrence : ∀ n ≥ 2, (a n)^2 = (a (n - 1)) * (a (n + 1)))
  (h_init1 : a 2 + 2 * a 1 = 4) (h_init2 : (a 3)^2 = a 5) : 
  (∀ n, a n = 2^n) :=
by sorry

theorem sum_bn (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_recurrence : ∀ n ≥ 2, (a n)^2 = (a (n - 1)) * (a (n + 1)))
  (h_init1 : a 2 + 2 * a 1 = 4) (h_init2 : (a 3)^2 = a 5) 
  (h_gen : ∀ n, a n = 2^n) (h_bn : ∀ n, b n = n * a n) :
  (∀ n, S n = (n-1) * 2^(n+1) + 2) :=
by sorry

end sequence_geometric_sum_bn_l87_87284


namespace total_amount_l87_87191

-- Definitions based on the problem conditions
def jack_amount : ℕ := 26
def ben_amount : ℕ := jack_amount - 9
def eric_amount : ℕ := ben_amount - 10

-- Proof statement
theorem total_amount : jack_amount + ben_amount + eric_amount = 50 :=
by
  -- Sorry serves as a placeholder for the actual proof
  sorry

end total_amount_l87_87191


namespace constant_sequence_is_AP_and_GP_l87_87784

theorem constant_sequence_is_AP_and_GP (seq : ℕ → ℕ) (h : ∀ n, seq n = 7) :
  (∃ d, ∀ n, seq n = seq (n + 1) + d) ∧ (∃ r, ∀ n, seq (n + 1) = seq n * r) :=
by
  sorry

end constant_sequence_is_AP_and_GP_l87_87784


namespace trigonometric_identity_l87_87603

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos (A / 2)) ^ 2 = (Real.cos (B / 2)) ^ 2 + (Real.cos (C / 2)) ^ 2 - 2 * (Real.cos (B / 2)) * (Real.cos (C / 2)) * (Real.sin (A / 2)) :=
sorry

end trigonometric_identity_l87_87603


namespace project_completion_time_l87_87578

def work_rate_A : ℚ := 1 / 20
def work_rate_B : ℚ := 1 / 30
def total_project_days (x : ℚ) : Prop := (work_rate_A * (x - 10) + work_rate_B * x = 1)

theorem project_completion_time (x : ℚ) (h : total_project_days x) : x = 13 := 
sorry

end project_completion_time_l87_87578


namespace flat_fee_shipping_l87_87397

theorem flat_fee_shipping (w : ℝ) (c : ℝ) (C : ℝ) (F : ℝ) 
  (h_w : w = 5) 
  (h_c : c = 0.80) 
  (h_C : C = 9)
  (h_shipping : C = F + (c * w)) :
  F = 5 :=
by
  -- proof skipped
  sorry

end flat_fee_shipping_l87_87397


namespace sphere_in_cube_volume_unreachable_l87_87680

noncomputable def volume_unreachable_space (cube_side : ℝ) (sphere_radius : ℝ) : ℝ :=
  let corner_volume := 64 - (32/3) * Real.pi
  let edge_volume := 288 - 72 * Real.pi
  corner_volume + edge_volume

theorem sphere_in_cube_volume_unreachable : 
  (volume_unreachable_space 6 1 = 352 - (248 * Real.pi / 3)) :=
by
  sorry

end sphere_in_cube_volume_unreachable_l87_87680


namespace cube_faces_sum_l87_87405

open Nat

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) (h6 : 0 < f) 
    (h7 : (a + d) * (b + e) * (c + f) = 1386) : 
    a + b + c + d + e + f = 38 := 
sorry

end cube_faces_sum_l87_87405


namespace min_text_length_l87_87619

theorem min_text_length : ∃ (L : ℕ), (∀ x : ℕ, 0.105 * (L : ℝ) < (x : ℝ) ∧ (x : ℝ) < 0.11 * (L : ℝ)) → L = 19 :=
by
  sorry

end min_text_length_l87_87619


namespace percentage_subtraction_l87_87417

variable (a b x m : ℝ) (p : ℝ)

-- Conditions extracted from the problem.
def ratio_a_to_b : Prop := a / b = 4 / 5
def definition_of_x : Prop := x = 1.75 * a
def definition_of_m : Prop := m = b * (1 - p / 100)
def value_m_div_x : Prop := m / x = 0.14285714285714285

-- The proof problem in the form of a Lean statement.
theorem percentage_subtraction 
  (h1 : ratio_a_to_b a b)
  (h2 : definition_of_x a x)
  (h3 : definition_of_m b m p)
  (h4 : value_m_div_x x m) : p = 80 := 
sorry

end percentage_subtraction_l87_87417


namespace total_cost_is_160_l87_87857

-- Define the costs of each dress
def CostOfPaulineDress := 30
def CostOfJeansDress := CostOfPaulineDress - 10
def CostOfIdasDress := CostOfJeansDress + 30
def CostOfPattysDress := CostOfIdasDress + 10

-- The total cost
def TotalCost := CostOfPaulineDress + CostOfJeansDress + CostOfIdasDress + CostOfPattysDress

-- Prove the total cost is $160
theorem total_cost_is_160 : TotalCost = 160 := by
  -- skipping the proof steps
  sorry

end total_cost_is_160_l87_87857


namespace monotonically_increasing_range_of_a_l87_87503

noncomputable def f (a x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 2 * x + 3

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ -Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end monotonically_increasing_range_of_a_l87_87503


namespace tart_fill_l87_87323

theorem tart_fill (cherries blueberries total : ℚ) (h_cherries : cherries = 0.08) (h_blueberries : blueberries = 0.75) (h_total : total = 0.91) :
  total - (cherries + blueberries) = 0.08 :=
by
  sorry

end tart_fill_l87_87323


namespace sum_abcd_l87_87845

variable (a b c d : ℝ)

theorem sum_abcd :
  (∃ y : ℝ, 2 * a + 3 = y ∧ 2 * b + 4 = y ∧ 2 * c + 5 = y ∧ 2 * d + 6 = y ∧ a + b + c + d + 10 = y) →
  a + b + c + d = -11 :=
by
  sorry

end sum_abcd_l87_87845


namespace num_square_tiles_l87_87478

theorem num_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 :=
  sorry

end num_square_tiles_l87_87478


namespace percentage_food_given_out_l87_87641

theorem percentage_food_given_out 
  (first_week_donations : ℕ)
  (second_week_donations : ℕ)
  (total_amount_donated : ℕ)
  (remaining_food : ℕ)
  (amount_given_out : ℕ)
  (percentage_given_out : ℕ) : 
  (first_week_donations = 40) →
  (second_week_donations = 2 * first_week_donations) →
  (total_amount_donated = first_week_donations + second_week_donations) →
  (remaining_food = 36) →
  (amount_given_out = total_amount_donated - remaining_food) →
  (percentage_given_out = (amount_given_out * 100) / total_amount_donated) →
  percentage_given_out = 70 :=
by sorry

end percentage_food_given_out_l87_87641


namespace express_set_A_l87_87968

def A := {x : ℤ | -1 < abs (x - 1) ∧ abs (x - 1) < 2}

theorem express_set_A : A = {0, 1, 2} := 
by
  sorry

end express_set_A_l87_87968


namespace option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l87_87958

section

variable (π : Real) (x : Real)

-- Definition of a fraction in this context
def is_fraction (num denom : Real) : Prop := denom ≠ 0

-- Proving each given option is a fraction
theorem option_a_is_fraction : is_fraction 1 π := 
sorry

theorem option_b_is_fraction : is_fraction x 3 :=
sorry

theorem option_c_is_fraction : is_fraction 2 5 :=
sorry

theorem option_d_is_fraction : is_fraction 1 (x - 1) :=
sorry

end

end option_a_is_fraction_option_b_is_fraction_option_c_is_fraction_option_d_is_fraction_l87_87958


namespace solve_inequality_l87_87160

theorem solve_inequality (x : ℝ) :
  abs ((3 * x - 2) / (x - 2)) > 3 →
  x ∈ Set.Ioo (4 / 3) 2 ∪ Set.Ioi 2 :=
by
  sorry

end solve_inequality_l87_87160


namespace geometric_sequence_product_l87_87318

theorem geometric_sequence_product 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_pos : ∀ n, a n > 0)
  (h_log_sum : Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6) :
  a 1 * a 15 = 10000 := 
sorry

end geometric_sequence_product_l87_87318


namespace flowers_bloom_l87_87463

theorem flowers_bloom (num_unicorns : ℕ) (flowers_per_step : ℕ) (distance_km : ℕ) (step_length_m : ℕ) 
  (h1 : num_unicorns = 6) (h2 : flowers_per_step = 4) (h3 : distance_km = 9) (h4 : step_length_m = 3) : 
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step = 72000 :=
by
  sorry

end flowers_bloom_l87_87463


namespace tickets_total_l87_87956

theorem tickets_total (x y : ℕ) 
  (h1 : 12 * x + 8 * y = 3320)
  (h2 : y = x + 190) : 
  x + y = 370 :=
by
  sorry

end tickets_total_l87_87956


namespace quadratic_coefficients_l87_87328

theorem quadratic_coefficients (b c : ℝ) :
  (∀ x : ℝ, |x + 4| = 3 ↔ x^2 + bx + c = 0) → (b = 8 ∧ c = 7) :=
by
  sorry

end quadratic_coefficients_l87_87328


namespace initial_mean_of_observations_l87_87729

-- Definitions of the given conditions and proof of the correct initial mean
theorem initial_mean_of_observations 
  (M : ℝ) -- Mean of 50 observations
  (initial_sum := 50 * M) -- Initial sum of observations
  (wrong_observation : ℝ := 23) -- Wrong observation
  (correct_observation : ℝ := 45) -- Correct observation
  (understated_by := correct_observation - wrong_observation) -- Amount of understatement
  (correct_sum := initial_sum + understated_by) -- Corrected sum
  (corrected_mean : ℝ := 36.5) -- Corrected new mean
  (eq1 : correct_sum = 50 * corrected_mean) -- Equation from condition of corrected mean
  (eq2 : initial_sum = 50 * corrected_mean - understated_by) -- Restating in terms of initial sum
  : M = 36.06 := -- The initial mean of observations
  sorry -- Proof omitted

end initial_mean_of_observations_l87_87729


namespace intersection_A_B_l87_87622

def A : Set ℝ := { x | x > -1 }
def B : Set ℝ := { y | (y - 2) * (y + 3) < 0 }

theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 2 :=
by
  sorry

end intersection_A_B_l87_87622


namespace incorrect_statement_S9_lt_S10_l87_87168

variable {a : ℕ → ℝ} -- Sequence
variable {S : ℕ → ℝ} -- Sum of the first n terms
variable {d : ℝ}     -- Common difference

-- Arithmetic sequence definition
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0 + n * (n-1) * d / 2)

-- Given conditions
variable 
  (arith_seq : arithmetic_sequence a d)
  (sum_terms : sum_of_first_n_terms a S)
  (H1 : S 9 < S 8)
  (H2 : S 8 = S 7)

-- Prove the statement
theorem incorrect_statement_S9_lt_S10 : 
  ¬ (S 9 < S 10) := 
sorry

end incorrect_statement_S9_lt_S10_l87_87168


namespace garden_perimeter_l87_87084

noncomputable def perimeter_of_garden (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) : ℝ :=
  2 * l + 2 * w

theorem garden_perimeter (w l : ℝ) (h1 : l = 3 * w + 15) (h2 : w * l = 4050) :
  perimeter_of_garden w l h1 h2 = 304.64 :=
sorry

end garden_perimeter_l87_87084


namespace bacteria_colony_exceeds_500_l87_87448

theorem bacteria_colony_exceeds_500 :
  ∃ (n : ℕ), (∀ m : ℕ, m < n → 4 * 3^m ≤ 500) ∧ 4 * 3^n > 500 :=
sorry

end bacteria_colony_exceeds_500_l87_87448


namespace polynomial_product_linear_term_zero_const_six_l87_87031

theorem polynomial_product_linear_term_zero_const_six (a b : ℝ)
  (h1 : (a + 2 * b = 0)) 
  (h2 : b = 6) : (a + b = -6) :=
by
  sorry

end polynomial_product_linear_term_zero_const_six_l87_87031


namespace book_prices_purchasing_plans_l87_87515

theorem book_prices (x y : ℕ) (h1 : 20 * x + 40 * y = 1600) (h2 : 20 * x = 30 * y + 200) : x = 40 ∧ y = 20 :=
by
  sorry

theorem purchasing_plans (m : ℕ) (h3 : 2 * m + 20 ≥ 70) (h4 : 40 * m + 20 * (m + 20) ≤ 2000) :
  (m = 25 ∧ m + 20 = 45) ∨ (m = 26 ∧ m + 20 = 46) :=
by
  -- proof steps
  sorry

end book_prices_purchasing_plans_l87_87515


namespace line_slope_intercept_l87_87501

theorem line_slope_intercept (a b: ℝ) (h₁: ∀ x y, (x, y) = (2, 3) ∨ (x, y) = (10, 19) → y = a * x + b)
  (h₂: (a * 6 + b) = 11) : a - b = 3 :=
by
  sorry

end line_slope_intercept_l87_87501


namespace marbles_cost_correct_l87_87875

def total_cost : ℝ := 20.52
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52

-- The problem is to prove that the amount spent on marbles is $9.05
def amount_spent_on_marbles : ℝ :=
  total_cost - (cost_football + cost_baseball)

theorem marbles_cost_correct :
  amount_spent_on_marbles = 9.05 :=
by
  -- The proof goes here.
  sorry

end marbles_cost_correct_l87_87875


namespace average_pushups_is_correct_l87_87983

theorem average_pushups_is_correct :
  ∀ (David Zachary Emily : ℕ),
    David = 510 →
    Zachary = David - 210 →
    Emily = David - 132 →
    (David + Zachary + Emily) / 3 = 396 :=
by
  intro David Zachary Emily hDavid hZachary hEmily
  -- All calculations and proofs will go here, but we'll leave them as sorry for now.
  sorry

end average_pushups_is_correct_l87_87983


namespace terminal_side_quadrant_l87_87610

theorem terminal_side_quadrant (k : ℤ) : 
  ∃ quadrant, quadrant = 1 ∨ quadrant = 3 ∧
  ∀ (α : ℝ), α = k * 180 + 45 → 
  (quadrant = 1 ∧ (∃ n : ℕ, k = 2 * n)) ∨ (quadrant = 3 ∧ (∃ n : ℕ, k = 2 * n + 1)) :=
by
  sorry

end terminal_side_quadrant_l87_87610


namespace fixed_monthly_fee_l87_87154

theorem fixed_monthly_fee (x y z : ℝ) 
  (h1 : x + y = 18.50) 
  (h2 : x + y + 3 * z = 23.45) : 
  x = 7.42 := 
by 
  sorry

end fixed_monthly_fee_l87_87154


namespace project_completion_l87_87723

theorem project_completion (a b : ℕ) (h1 : 3 * (1 / b : ℚ) + (1 / a : ℚ) + (1 / b : ℚ) = 1) : 
  a + b = 9 ∨ a + b = 10 :=
sorry

end project_completion_l87_87723


namespace burn_5_sticks_per_hour_l87_87646

-- Define the number of sticks each type of furniture makes
def sticks_per_chair := 6
def sticks_per_table := 9
def sticks_per_stool := 2

-- Define the number of each furniture Mary chopped up
def chairs_chopped := 18
def tables_chopped := 6
def stools_chopped := 4

-- Define the total number of hours Mary can keep warm
def hours_warm := 34

-- Calculate the total number of sticks of wood from each type of furniture
def total_sticks_chairs := chairs_chopped * sticks_per_chair
def total_sticks_tables := tables_chopped * sticks_per_table
def total_sticks_stools := stools_chopped * sticks_per_stool

-- Calculate the total number of sticks of wood
def total_sticks := total_sticks_chairs + total_sticks_tables + total_sticks_stools

-- The number of sticks of wood Mary needs to burn per hour
def sticks_per_hour := total_sticks / hours_warm

-- Prove that Mary needs to burn 5 sticks per hour to stay warm
theorem burn_5_sticks_per_hour : sticks_per_hour = 5 := sorry

end burn_5_sticks_per_hour_l87_87646


namespace find_k_and_prove_geometric_sequence_l87_87228

/-
Given conditions:
1. Sequence sa : ℕ → ℝ with sum sequence S : ℕ → ℝ satisfying the recurrence relation S (n + 1) = (k + 1) * S n + 2
2. Initial terms a_1 = 2 and a_2 = 1
-/

def sequence_sum_relation (S : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, S (n + 1) = (k + 1) * S n + 2

def init_sequence_terms (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ a 2 = 1

/-
Proof goal:
1. Prove k = -1/2 given the conditions.
2. Prove sequence a is a geometric sequence with common ratio 1/2 given the conditions.
-/

theorem find_k_and_prove_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ) :
  sequence_sum_relation S k →
  init_sequence_terms a →
  (k = (-1:ℝ)/2) ∧ (∀ n: ℕ, n ≥ 1 → a (n+1) = (1/2) * a n) :=
by
  sorry

end find_k_and_prove_geometric_sequence_l87_87228


namespace sewer_runoff_capacity_l87_87062

theorem sewer_runoff_capacity (gallons_per_hour : ℕ) (hours_per_day : ℕ) (days_till_overflow : ℕ)
  (h1 : gallons_per_hour = 1000)
  (h2 : hours_per_day = 24)
  (h3 : days_till_overflow = 10) :
  gallons_per_hour * hours_per_day * days_till_overflow = 240000 := 
by
  -- We'll use sorry here as the placeholder for the actual proof steps
  sorry

end sewer_runoff_capacity_l87_87062


namespace total_cost_of_two_books_l87_87961

theorem total_cost_of_two_books (C1 C2 total_cost: ℝ) :
  C1 = 262.5 →
  0.85 * C1 = 1.19 * C2 →
  total_cost = C1 + C2 →
  total_cost = 450 :=
by
  intros h1 h2 h3
  sorry

end total_cost_of_two_books_l87_87961


namespace slowest_bailing_rate_proof_l87_87789

def distance : ℝ := 1.5 -- in miles
def rowing_speed : ℝ := 3 -- in miles per hour
def water_intake_rate : ℝ := 8 -- in gallons per minute
def sink_threshold : ℝ := 50 -- in gallons

noncomputable def solve_bailing_rate_proof : ℝ :=
  let time_to_shore_hours : ℝ := distance / rowing_speed
  let time_to_shore_minutes : ℝ := time_to_shore_hours * 60
  let total_water_intake : ℝ := water_intake_rate * time_to_shore_minutes
  let excess_water : ℝ := total_water_intake - sink_threshold
  let bailing_rate_needed : ℝ := excess_water / time_to_shore_minutes
  bailing_rate_needed

theorem slowest_bailing_rate_proof : solve_bailing_rate_proof ≤ 7 :=
  by
    sorry

end slowest_bailing_rate_proof_l87_87789


namespace train_pass_time_l87_87018

-- Definitions based on the conditions
def train_length : ℕ := 360   -- Length of the train in meters
def platform_length : ℕ := 190 -- Length of the platform in meters
def speed_kmh : ℕ := 45       -- Speed of the train in km/h
def speed_ms : ℚ := speed_kmh * (1000 / 3600) -- Speed of the train in m/s

-- Total distance to be covered
def total_distance : ℕ := train_length + platform_length 

-- Time taken to pass the platform
def time_to_pass_platform : ℚ := total_distance / speed_ms

-- Proof that the time taken is 44 seconds
theorem train_pass_time : time_to_pass_platform = 44 := 
by 
  -- this is where the detailed proof would go
  sorry  

end train_pass_time_l87_87018


namespace abc_sum_zero_l87_87221

variable (a b c : ℝ)

-- Conditions given in the original problem
axiom h1 : a + b / c = 1
axiom h2 : b + c / a = 1
axiom h3 : c + a / b = 1

theorem abc_sum_zero : a * b + b * c + c * a = 0 :=
by
  sorry

end abc_sum_zero_l87_87221


namespace rhombus_area_correct_l87_87067

/-- Define the rhombus area calculation in miles given the lengths of its diagonals -/
def scale := 250
def d1 := 6 * scale -- first diagonal in miles
def d2 := 12 * scale -- second diagonal in miles
def areaOfRhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

theorem rhombus_area_correct :
  areaOfRhombus d1 d2 = 2250000 :=
by
  sorry

end rhombus_area_correct_l87_87067


namespace students_play_neither_l87_87703

-- Define the given conditions
def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_sports_players : ℕ := 17

-- The goal is to prove the number of students playing neither sport is 7
theorem students_play_neither :
  total_students - (football_players + long_tennis_players - both_sports_players) = 7 :=
by
  sorry

end students_play_neither_l87_87703


namespace total_gym_cost_l87_87807

def cheap_monthly_fee : ℕ := 10
def cheap_signup_fee : ℕ := 50
def expensive_monthly_fee : ℕ := 3 * cheap_monthly_fee
def expensive_signup_fee : ℕ := 4 * expensive_monthly_fee

def yearly_cost_cheap : ℕ := 12 * cheap_monthly_fee + cheap_signup_fee
def yearly_cost_expensive : ℕ := 12 * expensive_monthly_fee + expensive_signup_fee

theorem total_gym_cost : yearly_cost_cheap + yearly_cost_expensive = 650 := by
  -- Proof goes here
  sorry

end total_gym_cost_l87_87807


namespace find_f_of_1_over_3_l87_87689

theorem find_f_of_1_over_3
  (g : ℝ → ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, g x = 1 - x^2)
  (h2 : ∀ x, x ≠ 0 → f (g x) = (1 - x^2) / x^2) :
  f (1 / 3) = 1 / 2 := by
  sorry -- Proof goes here

end find_f_of_1_over_3_l87_87689


namespace tangent_line_tangent_value_at_one_l87_87105
noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function f

theorem tangent_line_tangent_value_at_one
  (f : ℝ → ℝ)
  (hf1 : f 1 = 3 - 1 / 2)
  (hf'1 : deriv f 1 = 1 / 2)
  (tangent_eq : ∀ x, f 1 + deriv f 1 * (x - 1) = 1 / 2 * x + 2) :
  f 1 + deriv f 1 = 3 :=
by sorry

end tangent_line_tangent_value_at_one_l87_87105


namespace gamin_difference_calculation_l87_87289

def largest_number : ℕ := 532
def smallest_number : ℕ := 406
def difference : ℕ := 126

theorem gamin_difference_calculation : largest_number - smallest_number = difference :=
by
  -- The solution proves that the difference between the largest and smallest numbers is 126.
  sorry

end gamin_difference_calculation_l87_87289


namespace sales_difference_greatest_in_june_l87_87617

def percentage_difference (D B : ℕ) : ℚ :=
  if B = 0 then 0 else (↑(max D B - min D B) / ↑(min D B)) * 100

def january : ℕ × ℕ := (8, 5)
def february : ℕ × ℕ := (10, 5)
def march : ℕ × ℕ := (8, 8)
def april : ℕ × ℕ := (4, 8)
def may : ℕ × ℕ := (5, 10)
def june : ℕ × ℕ := (3, 9)

noncomputable
def greatest_percentage_difference_month : String :=
  let jan_diff := percentage_difference january.1 january.2
  let feb_diff := percentage_difference february.1 february.2
  let mar_diff := percentage_difference march.1 march.2
  let apr_diff := percentage_difference april.1 april.2
  let may_diff := percentage_difference may.1 may.2
  let jun_diff := percentage_difference june.1 june.2
  if max jan_diff (max feb_diff (max mar_diff (max apr_diff (max may_diff jun_diff)))) == jun_diff
  then "June" else "Not June"
  
theorem sales_difference_greatest_in_june : greatest_percentage_difference_month = "June" :=
  by sorry

end sales_difference_greatest_in_june_l87_87617


namespace cost_to_feed_turtles_l87_87319

theorem cost_to_feed_turtles :
  (∀ (weight : ℚ), weight > 0 → (food_per_half_pound_ounces = 1) →
    (total_weight_pounds = 30) →
    (jar_contents_ounces = 15 ∧ jar_cost = 2) →
    (total_cost_dollars = 8)) :=
by
  sorry

variables
  (food_per_half_pound_ounces : ℚ := 1)
  (total_weight_pounds : ℚ := 30)
  (jar_contents_ounces : ℚ := 15)
  (jar_cost : ℚ := 2)
  (total_cost_dollars : ℚ := 8)

-- Assuming all variables needed to state the theorem exist and are meaningful
/-!
  Given:
  - Each turtle needs 1 ounce of food per 1/2 pound of body weight
  - Total turtles' weight is 30 pounds
  - Each jar of food contains 15 ounces and costs $2

  Prove:
  - The total cost to feed Sylvie's turtles is $8
-/

end cost_to_feed_turtles_l87_87319


namespace number_of_herds_l87_87390

-- Definitions from the conditions
def total_sheep : ℕ := 60
def sheep_per_herd : ℕ := 20

-- The statement to prove
theorem number_of_herds : total_sheep / sheep_per_herd = 3 := by
  sorry

end number_of_herds_l87_87390


namespace min_value_xy_l87_87014

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end min_value_xy_l87_87014


namespace complement_A_eq_interval_l87_87176

-- Define the universal set U as the set of all real numbers.
def U : Set ℝ := Set.univ

-- Define the set A using the condition x^2 - 2x - 3 > 0.
def A : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the complement of A with respect to U.
def A_complement : Set ℝ := { x | -1 <= x ∧ x <= 3 }

theorem complement_A_eq_interval : A_complement = { x | -1 <= x ∧ x <= 3 } :=
by
  sorry

end complement_A_eq_interval_l87_87176


namespace zayne_total_revenue_l87_87321

-- Defining the constants and conditions
def price_per_bracelet := 5
def deal_price := 8
def initial_bracelets := 30
def revenue_from_five_dollar_sales := 60

-- Calculating number of bracelets sold for $5 each
def bracelets_sold_five_dollars := revenue_from_five_dollar_sales / price_per_bracelet

-- Calculating remaining bracelets after selling some for $5 each
def remaining_bracelets := initial_bracelets - bracelets_sold_five_dollars

-- Calculating number of pairs sold at two for $8
def pairs_sold := remaining_bracelets / 2

-- Calculating revenue from selling pairs
def revenue_from_deal_sales := pairs_sold * deal_price

-- Total revenue calculation
def total_revenue := revenue_from_five_dollar_sales + revenue_from_deal_sales

-- Theorem to prove the total revenue is $132
theorem zayne_total_revenue : total_revenue = 132 := by
  sorry

end zayne_total_revenue_l87_87321


namespace max_right_angle_triangles_l87_87762

open Real

theorem max_right_angle_triangles (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ x y : ℝ, x^2 + a^2 * y^2 = a^2) :
  ∃n : ℕ, n = 3 := 
by
  sorry

end max_right_angle_triangles_l87_87762


namespace flowers_per_bouquet_l87_87830

noncomputable def num_flowers_per_bouquet (total_flowers wilted_flowers bouquets : ℕ) : ℕ :=
  (total_flowers - wilted_flowers) / bouquets

theorem flowers_per_bouquet : num_flowers_per_bouquet 53 18 5 = 7 := by
  sorry

end flowers_per_bouquet_l87_87830


namespace action_figures_per_shelf_l87_87178

/-- Mike has 64 action figures he wants to display. If each shelf 
    in his room can hold a certain number of figures and he needs 8 
    shelves, prove that each shelf can hold 8 figures. -/
theorem action_figures_per_shelf :
  (64 / 8) = 8 :=
by
  sorry

end action_figures_per_shelf_l87_87178


namespace subtract_3a_result_l87_87846

theorem subtract_3a_result (a : ℝ) : 
  (9 * a^2 - 3 * a + 8) + 3 * a = 9 * a^2 + 8 := 
sorry

end subtract_3a_result_l87_87846


namespace tourists_speeds_l87_87075

theorem tourists_speeds (x y : ℝ) :
  (20 / x + 2.5 = 20 / y) →
  (20 / (x - 2) = 20 / (1.5 * y)) →
  x = 8 ∧ y = 4 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end tourists_speeds_l87_87075


namespace boy_scouts_percentage_l87_87231

variable (S B G : ℝ)

-- Conditions
-- Given B + G = S
axiom condition1 : B + G = S

-- Given 0.75B + 0.625G = 0.7S
axiom condition2 : 0.75 * B + 0.625 * G = 0.7 * S

-- Goal
theorem boy_scouts_percentage : B / S = 0.6 :=
by sorry

end boy_scouts_percentage_l87_87231


namespace number_chosen_l87_87615

theorem number_chosen (x : ℤ) (h : x / 4 - 175 = 10) : x = 740 := by
  sorry

end number_chosen_l87_87615


namespace faye_has_62_pieces_of_candy_l87_87577

-- Define initial conditions
def initialCandy : Nat := 47
def eatenCandy : Nat := 25
def receivedCandy : Nat := 40

-- Define the resulting number of candies after eating and receiving more candies
def resultingCandy : Nat := initialCandy - eatenCandy + receivedCandy

-- State the theorem and provide the proof
theorem faye_has_62_pieces_of_candy :
  resultingCandy = 62 :=
by
  -- proof goes here
  sorry

end faye_has_62_pieces_of_candy_l87_87577


namespace fifteenth_term_l87_87593

variable (a b : ℤ)

def sum_first_n_terms (n : ℕ) : ℤ := n * (2 * a + (n - 1) * b) / 2

axiom sum_first_10 : sum_first_n_terms 10 = 60
axiom sum_first_20 : sum_first_n_terms 20 = 320

def nth_term (n : ℕ) : ℤ := a + (n - 1) * b

theorem fifteenth_term : nth_term 15 = 25 :=
by
  sorry

end fifteenth_term_l87_87593


namespace math_problem_l87_87743

theorem math_problem (a b : ℕ) (h₁ : a = 6) (h₂ : b = 6) : 
  (a^3 + b^3) / (a^2 - a * b + b^2) = 12 :=
by
  sorry

end math_problem_l87_87743


namespace agatha_initial_money_60_l87_87329

def Agatha_initial_money (spent_frame : ℕ) (spent_front_wheel: ℕ) (left_over: ℕ) : ℕ :=
  spent_frame + spent_front_wheel + left_over

theorem agatha_initial_money_60 :
  Agatha_initial_money 15 25 20 = 60 :=
by
  -- This line assumes $15 on frame, $25 on wheel, $20 left translates to a total of $60.
  sorry

end agatha_initial_money_60_l87_87329


namespace fraction_expression_proof_l87_87114

theorem fraction_expression_proof :
  (1 / 8 * 1 / 9 * 1 / 28 = 1 / 2016) ∨ ((1 / 8 - 1 / 9) * 1 / 28 = 1 / 2016) :=
by
  sorry

end fraction_expression_proof_l87_87114


namespace non_rain_hours_correct_l87_87219

def total_hours : ℕ := 9
def rain_hours : ℕ := 4

theorem non_rain_hours_correct : (total_hours - rain_hours) = 5 := 
by
  sorry

end non_rain_hours_correct_l87_87219


namespace total_boys_in_school_l87_87096

-- Define the total percentage of boys belonging to other communities
def percentage_other_communities := 100 - (44 + 28 + 10)

-- Total number of boys in the school, represented by a variable B
def total_boys (B : ℕ) : Prop :=
0.18 * (B : ℝ) = 117

-- The theorem states that the total number of boys B is 650
theorem total_boys_in_school : ∃ B : ℕ, total_boys B ∧ B = 650 :=
sorry

end total_boys_in_school_l87_87096


namespace smallest_repeating_block_length_l87_87250

-- Define the decimal expansion of 3/11
noncomputable def decimalExpansion : Rational → List Nat :=
  sorry

-- Define the repeating block determination of a given decimal expansion
noncomputable def repeatingBlockLength : List Nat → Nat :=
  sorry

-- Define the fraction 3/11
def frac := (3 : Rat) / 11

-- State the theorem
theorem smallest_repeating_block_length :
  repeatingBlockLength (decimalExpansion frac) = 2 :=
  sorry

end smallest_repeating_block_length_l87_87250


namespace commission_rate_correct_l87_87070

-- Define the given conditions
def base_pay := 190
def goal_earnings := 500
def required_sales := 7750

-- Define the commission rate function
def commission_rate (sales commission : ℕ) : ℚ := (commission : ℚ) / (sales : ℚ) * 100

-- The main statement to prove
theorem commission_rate_correct :
  commission_rate required_sales (goal_earnings - base_pay) = 4 :=
by
  sorry

end commission_rate_correct_l87_87070


namespace problem_statement_l87_87492

-- Define f(x) and g(x)
def f (x : ℝ) : ℝ := x^2 + 2 * x + 5
def g (x : ℝ) : ℝ := 2 * x + 3

-- Statement to prove: f(g(3)) - g(f(3)) = 61
theorem problem_statement : f (g 3) - g (f 3) = 61 := by
  sorry

end problem_statement_l87_87492


namespace jane_sleep_hours_for_second_exam_l87_87904

theorem jane_sleep_hours_for_second_exam :
  ∀ (score1 score2 hours1 hours2 : ℝ),
  score1 * hours1 = 675 →
  (score1 + score2) / 2 = 85 →
  score2 * hours2 = 675 →
  hours2 = 135 / 19 :=
by
  intros score1 score2 hours1 hours2 h1 h2 h3
  sorry

end jane_sleep_hours_for_second_exam_l87_87904


namespace find_b_l87_87639

open Real

theorem find_b (b : ℝ) : 
  (∀ x y : ℝ, 4 * y - 3 * x - 2 = 0 -> 6 * y + b * x + 1 = 0 -> 
   exists m₁ m₂ : ℝ, 
   ((y = m₁ * x + _1 / 2) -> m₁ = 3 / 4) ∧ ((y = m₂ * x - 1 / 6) -> m₂ = -b / 6)) -> 
  b = -4.5 :=
by
  sorry

end find_b_l87_87639


namespace triangle_longest_side_l87_87837

theorem triangle_longest_side 
  (x : ℝ)
  (h1 : 7 + (x + 4) + (2 * x + 1) = 36) :
  2 * x + 1 = 17 := by
  sorry

end triangle_longest_side_l87_87837


namespace circle_equation_and_range_of_a_l87_87030

theorem circle_equation_and_range_of_a :
  (∃ m : ℤ, (x - m)^2 + y^2 = 25 ∧ (abs (4 * m - 29)) = 25) ∧
  (∀ a : ℝ, (a > 0 → (4 * (5 * a - 1)^2 - 4 * (a^2 + 1) > 0 → a > 5 / 12 ∨ a < 0))) :=
by
  sorry

end circle_equation_and_range_of_a_l87_87030


namespace random_events_count_is_five_l87_87785

-- Definitions of the events in the conditions
def event1 := "Classmate A successfully runs for class president"
def event2 := "Stronger team wins in a game between two teams"
def event3 := "A school has a total of 998 students, and at least three students share the same birthday"
def event4 := "If sets A, B, and C satisfy A ⊆ B and B ⊆ C, then A ⊆ C"
def event5 := "In ancient times, a king wanted to execute a painter. Secretly, he wrote 'death' on both slips of paper, then let the painter draw a 'life or death' slip. The painter drew a death slip"
def event6 := "It snows in July"
def event7 := "Choosing any two numbers from 1, 3, 9, and adding them together results in an even number"
def event8 := "Riding through 10 intersections, all lights encountered are red"

-- Tally up the number of random events
def is_random_event (event : String) : Bool :=
  event = event1 ∨
  event = event2 ∨
  event = event3 ∨
  event = event6 ∨
  event = event8

def count_random_events (events : List String) : Nat :=
  (events.map (λ event => if is_random_event event then 1 else 0)).sum

-- List of events
def events := [event1, event2, event3, event4, event5, event6, event7, event8]

-- Theorem statement
theorem random_events_count_is_five : count_random_events events = 5 :=
  by
    sorry

end random_events_count_is_five_l87_87785


namespace sail_time_difference_l87_87541

theorem sail_time_difference (distance : ℕ) (v_big : ℕ) (v_small : ℕ) (t_big t_small : ℕ)
  (h_distance : distance = 200)
  (h_v_big : v_big = 50)
  (h_v_small : v_small = 20)
  (h_t_big : t_big = distance / v_big)
  (h_t_small : t_small = distance / v_small)
  : t_small - t_big = 6 := by
  sorry

end sail_time_difference_l87_87541


namespace part1_part2_l87_87290

-- Definitions from condition part
def f (a x : ℝ) := a * x^2 + (1 + a) * x + a

-- Part (1) Statement
theorem part1 (a : ℝ) : 
  (a ≥ -1/3) → (∀ x : ℝ, f a x ≥ 0) :=
sorry

-- Part (2) Statement
theorem part2 (a : ℝ) : 
  (a > 0) → 
  (∀ x : ℝ, f a x < a - 1) → 
  ((0 < a ∧ a < 1) → (-1/a < x ∧ x < -1) ∨ 
   (a = 1) → False ∨
   (a > 1) → (-1 < x ∧ x < -1/a)) :=
sorry

end part1_part2_l87_87290


namespace film_cost_eq_five_l87_87682

variable (F : ℕ)

theorem film_cost_eq_five (H1 : 9 * F + 4 * 4 + 6 * 3 = 79) : F = 5 :=
by
  -- This is a placeholder for your proof
  sorry

end film_cost_eq_five_l87_87682


namespace product_evaluation_l87_87325

-- Define the conditions and the target expression
def product (a : ℕ) : ℕ := (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

-- Main theorem statement
theorem product_evaluation : product 7 = 5040 :=
by
  -- Lean usually requires some import from the broader Mathlib to support arithmetic simplifications
  sorry

end product_evaluation_l87_87325


namespace floor_of_pi_l87_87621

noncomputable def floor_of_pi_eq_three : Prop :=
  ⌊Real.pi⌋ = 3

theorem floor_of_pi : floor_of_pi_eq_three :=
  sorry

end floor_of_pi_l87_87621


namespace curve_crossing_l87_87735

structure Point where
  x : ℝ
  y : ℝ

def curve (t : ℝ) : Point :=
  { x := 2 * t^2 - 3, y := 2 * t^4 - 9 * t^2 + 6 }

theorem curve_crossing : ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve 1 = { x := -1, y := -1 } := by
  sorry

end curve_crossing_l87_87735


namespace vegetarian_gluten_free_fraction_l87_87730

theorem vegetarian_gluten_free_fraction :
  ∀ (total_dishes meatless_dishes gluten_free_meatless_dishes : ℕ),
  meatless_dishes = 4 →
  meatless_dishes = total_dishes / 5 →
  gluten_free_meatless_dishes = meatless_dishes - 3 →
  gluten_free_meatless_dishes / total_dishes = 1 / 20 :=
by sorry

end vegetarian_gluten_free_fraction_l87_87730


namespace fraction_start_with_9_end_with_0_is_1_over_72_l87_87934

-- Definition of valid 8-digit telephone number
def valid_phone_number (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  2 ≤ d.val ∧ d.val ≤ 9 ∧ n.val ≤ 8

-- Definition of phone numbers that start with 9 and end with 0
def starts_with_9_ends_with_0 (d : Fin 10) (n : Fin 10) (m : Fin (10 ^ 6)) : Prop :=
  d.val = 9 ∧ n.val = 0

-- The total number of valid 8-digit phone numbers
noncomputable def total_valid_numbers : ℕ :=
  8 * (10 ^ 6) * 9

-- The number of valid phone numbers that start with 9 and end with 0
noncomputable def valid_start_with_9_end_with_0 : ℕ :=
  10 ^ 6

-- The target fraction
noncomputable def target_fraction : ℚ :=
  valid_start_with_9_end_with_0 / total_valid_numbers

-- Main theorem
theorem fraction_start_with_9_end_with_0_is_1_over_72 :
  target_fraction = (1 / 72 : ℚ) :=
by
  sorry

end fraction_start_with_9_end_with_0_is_1_over_72_l87_87934


namespace division_identity_l87_87746

theorem division_identity : 45 / 0.05 = 900 :=
by
  sorry

end division_identity_l87_87746


namespace find_y_l87_87027

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end find_y_l87_87027


namespace find_n_divides_2n_plus_2_l87_87439

theorem find_n_divides_2n_plus_2 :
  ∃ n : ℕ, (100 ≤ n ∧ n ≤ 1997 ∧ n ∣ (2 * n + 2)) ∧ n = 946 :=
by {
  sorry
}

end find_n_divides_2n_plus_2_l87_87439


namespace no_such_f_exists_l87_87720

theorem no_such_f_exists (f : ℝ → ℝ) (h1 : ∀ x, 0 < x → 0 < f x) 
  (h2 : ∀ x y, 0 < x → 0 < y → f x ^ 2 ≥ f (x + y) * (f x + y)) : false :=
sorry

end no_such_f_exists_l87_87720


namespace selling_price_per_sweater_correct_l87_87359

-- Definitions based on the problem's conditions
def balls_of_yarn_per_sweater := 4
def cost_per_ball_of_yarn := 6
def number_of_sweaters := 28
def total_gain := 308

-- Defining the required selling price per sweater
def total_cost_of_yarn : Nat := balls_of_yarn_per_sweater * cost_per_ball_of_yarn * number_of_sweaters
def total_revenue : Nat := total_cost_of_yarn + total_gain
def selling_price_per_sweater : ℕ := total_revenue / number_of_sweaters

theorem selling_price_per_sweater_correct :
  selling_price_per_sweater = 35 :=
  by
  sorry

end selling_price_per_sweater_correct_l87_87359


namespace number_of_performances_l87_87695

theorem number_of_performances (hanna_songs : ℕ) (mary_songs : ℕ) (alina_songs : ℕ) (tina_songs : ℕ)
    (hanna_cond : hanna_songs = 4)
    (mary_cond : mary_songs = 7)
    (alina_cond : 4 < alina_songs ∧ alina_songs < 7)
    (tina_cond : 4 < tina_songs ∧ tina_songs < 7) :
    ((hanna_songs + mary_songs + alina_songs + tina_songs) / 3) = 7 :=
by
  -- proof steps would go here
  sorry

end number_of_performances_l87_87695


namespace value_of_3k_squared_minus_1_l87_87642

theorem value_of_3k_squared_minus_1 (x k : ℤ)
  (h1 : 7 * x + 2 = 3 * x - 6)
  (h2 : x + 1 = k)
  : 3 * k^2 - 1 = 2 := 
by
  sorry

end value_of_3k_squared_minus_1_l87_87642


namespace problem1_problem2_l87_87404

-- Problem 1
theorem problem1 (x : ℝ) : 
  (x + 2) * (x - 2) - 2 * (x - 3) = 3 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (x + 3)^2 = (1 - 2 * x)^2 ↔ x = 4 ∨ x = -2 / 3 := 
sorry

end problem1_problem2_l87_87404


namespace negation_of_proposition_l87_87141

-- Definitions based on given conditions
def is_not_divisible_by_2 (n : ℤ) := n % 2 ≠ 0
def is_odd (n : ℤ) := n % 2 = 1

-- The negation proposition to be proved
theorem negation_of_proposition : ∃ n : ℤ, is_not_divisible_by_2 n ∧ ¬ is_odd n := 
sorry

end negation_of_proposition_l87_87141


namespace find_value_of_2_minus_c_l87_87352

theorem find_value_of_2_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 3 + d = 8 + c) : 2 - c = -1 := 
by
  sorry

end find_value_of_2_minus_c_l87_87352


namespace geom_seq_sum_first_four_terms_l87_87935

noncomputable def sum_first_n_terms_geom (a₁ q: ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geom_seq_sum_first_four_terms
  (a₁ : ℕ) (q : ℕ) (h₁ : a₁ = 1) (h₂ : a₁ * q^3 = 27) :
  sum_first_n_terms_geom a₁ q 4 = 40 :=
by
  sorry

end geom_seq_sum_first_four_terms_l87_87935


namespace cost_of_four_dozen_apples_l87_87007

-- Define the given conditions and problem
def half_dozen_cost : ℚ := 4.80 -- cost of half a dozen apples
def full_dozen_cost : ℚ := half_dozen_cost / 0.5
def four_dozen_cost : ℚ := 4 * full_dozen_cost

-- Statement of the theorem to prove
theorem cost_of_four_dozen_apples : four_dozen_cost = 38.40 :=
by
  sorry

end cost_of_four_dozen_apples_l87_87007


namespace election_win_by_votes_l87_87943

/-- Two candidates in an election, the winner received 56% of votes and won the election
by receiving 1344 votes. We aim to prove that the winner won by 288 votes. -/
theorem election_win_by_votes
  (V : ℝ)  -- total number of votes
  (w : ℝ)  -- percentage of votes received by the winner
  (w_votes : ℝ)  -- votes received by the winner
  (l_votes : ℝ)  -- votes received by the loser
  (w_percentage : w = 0.56)
  (w_votes_given : w_votes = 1344)
  (total_votes : V = 1344 / 0.56)
  (l_votes_calc : l_votes = (V * 0.44)) :
  1344 - l_votes = 288 :=
by
  -- Proof goes here
  sorry

end election_win_by_votes_l87_87943


namespace evaluate_expression_l87_87558

theorem evaluate_expression :
  let a := 3 * 4 * 5
  let b := (1 : ℝ) / 3
  let c := (1 : ℝ) / 4
  let d := (1 : ℝ) / 5
  (a : ℝ) * (b + c - d) = 23 := by
  sorry

end evaluate_expression_l87_87558


namespace cricket_team_members_l87_87195

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

end cricket_team_members_l87_87195


namespace tangent_line_at_1_l87_87257

def f (x : ℝ) : ℝ := sorry

theorem tangent_line_at_1 (f' : ℝ → ℝ) (h1 : ∀ x, deriv f x = f' x) (h2 : ∀ y, 2 * 1 + y - 3 = 0) :
  f' 1 + f 1 = -1 :=
by
  sorry

end tangent_line_at_1_l87_87257


namespace exists_integers_abcd_l87_87999

theorem exists_integers_abcd (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), x = a^2 + b^2 ∧ y = c^2 + d^2 ∧ z = a * c + b * d :=
sorry

end exists_integers_abcd_l87_87999


namespace nine_b_value_l87_87858

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : a = b - 3) : 
  9 * b = 216 / 11 :=
by
  sorry

end nine_b_value_l87_87858


namespace solution_set_of_inequality_l87_87193

variable (a b x : ℝ)
variable (h1 : ∀ x, ax + b > 0 ↔ 1 < x)

theorem solution_set_of_inequality : ∀ x, (ax + b) * (x - 2) < 0 ↔ (1 < x ∧ x < 2) :=
by sorry

end solution_set_of_inequality_l87_87193


namespace binomial_10_3_l87_87798

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_10_3 : binomial 10 3 = 120 := 
  by 
    sorry

end binomial_10_3_l87_87798


namespace certain_number_l87_87534

theorem certain_number (x : ℝ) : 
  0.55 * x = (4/5 : ℝ) * 25 + 2 → 
  x = 40 :=
by
  sorry

end certain_number_l87_87534


namespace problem_statement_l87_87080

def f(x : ℝ) : ℝ := 3 * x - 3
def g(x : ℝ) : ℝ := x^2 + 1

theorem problem_statement : f (1 + g 2) = 15 := by
  sorry

end problem_statement_l87_87080


namespace possible_values_of_product_l87_87300

theorem possible_values_of_product 
  (P_A P_B P_C P_D P_E : ℕ)
  (H1 : P_A = P_B + P_C + P_D + P_E)
  (H2 : ∃ n1 n2 n3 n4, 
          ((P_B = n1 * (n1 + 1)) ∨ (P_B = n2 * (n2 + 1) * (n2 + 2)) ∨ 
           (P_B = n3 * (n3 + 1) * (n3 + 2) * (n3 + 3)) ∨ (P_B = n4 * (n4 + 1) * (n4 + 2) * (n4 + 3) * (n4 + 4))) ∧
          ∃ m1 m2 m3 m4, 
          ((P_C = m1 * (m1 + 1)) ∨ (P_C = m2 * (m2 + 1) * (m2 + 2)) ∨ 
           (P_C = m3 * (m3 + 1) * (m3 + 2) * (m3 + 3)) ∨ (P_C = m4 * (m4 + 1) * (m4 + 2) * (m4 + 3) * (m4 + 4))) ∧
          ∃ o1 o2 o3 o4, 
          ((P_D = o1 * (o1 + 1)) ∨ (P_D = o2 * (o2 + 1) * (o2 + 2)) ∨ 
           (P_D = o3 * (o3 + 1) * (o3 + 2) * (o3 + 3)) ∨ (P_D = o4 * (o4 + 1) * (o4 + 2) * (o4 + 3) * (o4 + 4))) ∧
          ∃ p1 p2 p3 p4, 
          ((P_E = p1 * (p1 + 1)) ∨ (P_E = p2 * (p2 + 1) * (p2 + 2)) ∨ 
           (P_E = p3 * (p3 + 1) * (p3 + 2) * (p3 + 3)) ∨ (P_E = p4 * (p4 + 1) * (p4 + 2) * (p4 + 3) * (p4 + 4))) ∧ 
          ∃ q1 q2 q3 q4, 
          ((P_A = q1 * (q1 + 1)) ∨ (P_A = q2 * (q2 + 1) * (q2 + 2)) ∨ 
           (P_A = q3 * (q3 + 1) * (q3 + 2) * (q3 + 3)) ∨ (P_A = q4 * (q4 + 1) * (q4 + 2) * (q4 + 3) * (q4 + 4)))) :
  P_A = 6 ∨ P_A = 24 :=
by sorry

end possible_values_of_product_l87_87300


namespace g_2_eq_8_l87_87960

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (x : ℝ) : ℝ := (3 * x - 4) / x

noncomputable def g (x : ℝ) : ℝ := 1 / f_inv x + 7

theorem g_2_eq_8 : g 2 = 8 := 
by 
  unfold g
  unfold f_inv
  sorry

end g_2_eq_8_l87_87960


namespace no_integer_n_gte_1_where_9_divides_7n_plus_n3_l87_87717

theorem no_integer_n_gte_1_where_9_divides_7n_plus_n3 :
  ∀ n : ℕ, 1 ≤ n → ¬ (7^n + n^3) % 9 = 0 := 
by
  intros n hn
  sorry

end no_integer_n_gte_1_where_9_divides_7n_plus_n3_l87_87717


namespace evaluate_f_l87_87871

def f (x : ℚ) : ℚ := (2 * x - 3) / (3 * x ^ 2 - 1)

theorem evaluate_f :
  f (-2) = -7 / 11 ∧ f (0) = 3 ∧ f (1) = -1 / 2 :=
by
  sorry

end evaluate_f_l87_87871


namespace largest_interior_angle_of_triangle_l87_87055

theorem largest_interior_angle_of_triangle (a b c ext : ℝ)
    (h1 : a + b + c = 180)
    (h2 : a / 4 = b / 5)
    (h3 : a / 4 = c / 6)
    (h4 : c + 120 = a + 180) : c = 72 :=
by
  sorry

end largest_interior_angle_of_triangle_l87_87055


namespace find_first_year_l87_87842

-- Define sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

-- Define the conditions
def after_2020 (n : ℕ) : Prop := n > 2020
def sum_of_digits_eq (n required_sum : ℕ) : Prop := sum_of_digits n = required_sum

noncomputable def first_year_after_2020_with_digit_sum_15 : ℕ :=
  2049

-- The statement to be proved
theorem find_first_year : 
  ∃ y : ℕ, after_2020 y ∧ sum_of_digits_eq y 15 ∧ y = first_year_after_2020_with_digit_sum_15 :=
by
  sorry

end find_first_year_l87_87842


namespace phone_answered_within_two_rings_l87_87299

def probability_of_first_ring : ℝ := 0.5
def probability_of_second_ring : ℝ := 0.3
def probability_of_within_two_rings : ℝ := 0.8

theorem phone_answered_within_two_rings :
  probability_of_first_ring + probability_of_second_ring = probability_of_within_two_rings :=
by
  sorry

end phone_answered_within_two_rings_l87_87299


namespace scout_troop_profit_l87_87134

noncomputable def buy_price_per_bar : ℚ := 3 / 4
noncomputable def sell_price_per_bar : ℚ := 2 / 3
noncomputable def num_candy_bars : ℕ := 800

theorem scout_troop_profit :
  num_candy_bars * (sell_price_per_bar : ℚ) - num_candy_bars * (buy_price_per_bar : ℚ) = -66.64 :=
by
  sorry

end scout_troop_profit_l87_87134


namespace prob_xi_ge_2_eq_one_third_l87_87664

noncomputable def pmf (c k : ℝ) : ℝ := c / (k * (k + 1))

theorem prob_xi_ge_2_eq_one_third 
  (c : ℝ) 
  (h₁ : pmf c 1 + pmf c 2 + pmf c 3 = 1) :
  pmf c 2 + pmf c 3 = 1 / 3 :=
by
  sorry

end prob_xi_ge_2_eq_one_third_l87_87664


namespace gain_percent_is_correct_l87_87244

theorem gain_percent_is_correct :
  let CP : ℝ := 450
  let SP : ℝ := 520
  let gain : ℝ := SP - CP
  let gain_percent : ℝ := (gain / CP) * 100
  gain_percent = 15.56 :=
by
  sorry

end gain_percent_is_correct_l87_87244


namespace least_number_divisible_remainder_l87_87286

theorem least_number_divisible_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 := 
sorry

end least_number_divisible_remainder_l87_87286


namespace sam_and_david_licks_l87_87474

theorem sam_and_david_licks :
  let Dan_licks := 58
  let Michael_licks := 63
  let Lance_licks := 39
  let avg_licks := 60
  let total_people := 5
  let total_licks := avg_licks * total_people
  let total_licks_Dan_Michael_Lance := Dan_licks + Michael_licks + Lance_licks
  total_licks - total_licks_Dan_Michael_Lance = 140 := by
  sorry

end sam_and_david_licks_l87_87474


namespace average_weight_of_a_and_b_l87_87112

-- Define the parameters in the conditions
variables (A B C : ℝ)

-- Conditions given in the problem
theorem average_weight_of_a_and_b (h1 : (A + B + C) / 3 = 45) 
                                 (h2 : (B + C) / 2 = 43) 
                                 (h3 : B = 33) : (A + B) / 2 = 41 := 
sorry

end average_weight_of_a_and_b_l87_87112


namespace garden_breadth_l87_87897

theorem garden_breadth (P L B : ℕ) (h1 : P = 700) (h2 : L = 250) (h3 : P = 2 * (L + B)) : B = 100 :=
by
  sorry

end garden_breadth_l87_87897


namespace statement_A_statement_B_statement_C_statement_D_l87_87737

-- Definitions based on the problem conditions
def curve (m : ℝ) (x y : ℝ) : Prop :=
  x^4 + y^4 + m * x^2 * y^2 = 1

def is_symmetric_about_origin (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y ↔ curve m (-x) (-y)

def enclosed_area_eq_pi (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y → (x^2 + y^2)^2 = 1

def does_not_intersect_y_eq_x (m : ℝ) : Prop :=
  ∀ x y : ℝ, curve m x y ∧ x = y → false

def no_common_points_with_region (m : ℝ) : Prop :=
  ∀ x y : ℝ, |x| + |y| < 1 → ¬ curve m x y

-- Statements to prove based on correct answers
theorem statement_A (m : ℝ) : is_symmetric_about_origin m :=
  sorry

theorem statement_B (m : ℝ) (h : m = 2) : enclosed_area_eq_pi m :=
  sorry

theorem statement_C (m : ℝ) (h : m = -2) : ¬ does_not_intersect_y_eq_x m :=
  sorry

theorem statement_D (m : ℝ) (h : m = -1) : no_common_points_with_region m :=
  sorry

end statement_A_statement_B_statement_C_statement_D_l87_87737


namespace tangential_tetrahedron_triangle_impossibility_l87_87787

theorem tangential_tetrahedron_triangle_impossibility (a b c d : ℝ) 
  (h : ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → x > 0) :
  ¬ (∀ (x y z : ℝ) , (x = a ∨ x = b ∨ x = c ∨ x = d) → 
    (y = a ∨ y = b ∨ y = c ∨ y = d) →
    (z = a ∨ z = b ∨ z = c ∨ z = d) → 
    x ≠ y → y ≠ z → z ≠ x → x + y > z ∧ x + z > y ∧ y + z > x) :=
sorry

end tangential_tetrahedron_triangle_impossibility_l87_87787


namespace students_surveyed_l87_87538

theorem students_surveyed (S : ℕ)
  (h1 : (2/3 : ℝ) * 6 + (1/3 : ℝ) * 4 = 16/3)
  (h2 : S * (16/3 : ℝ) = 320) :
  S = 60 :=
sorry

end students_surveyed_l87_87538


namespace notebooks_difference_l87_87457

theorem notebooks_difference 
  (cost_mika : ℝ) (cost_leo : ℝ) (notebook_price : ℝ)
  (h_cost_mika : cost_mika = 2.40)
  (h_cost_leo : cost_leo = 3.20)
  (h_notebook_price : notebook_price > 0.10)
  (h_mika : ∃ m : ℕ, cost_mika = m * notebook_price)
  (h_leo : ∃ l : ℕ, cost_leo = l * notebook_price)
  : ∃ n : ℕ, (l - m = 4) :=
by
  sorry

end notebooks_difference_l87_87457


namespace matching_shoes_probability_is_one_ninth_l87_87995

def total_shoes : ℕ := 10
def pairs_of_shoes : ℕ := 5
def total_combinations : ℕ := (total_shoes * (total_shoes - 1)) / 2
def matching_combinations : ℕ := pairs_of_shoes

def matching_shoes_probability : ℚ := matching_combinations / total_combinations

theorem matching_shoes_probability_is_one_ninth :
  matching_shoes_probability = 1 / 9 :=
by
  sorry

end matching_shoes_probability_is_one_ninth_l87_87995


namespace angle_C_in_triangle_l87_87123

theorem angle_C_in_triangle {A B C : ℝ} 
  (h1 : A - B = 10) 
  (h2 : B = 0.5 * A) : 
  C = 150 :=
by
  -- Placeholder for proof
  sorry

end angle_C_in_triangle_l87_87123


namespace soccer_boys_percentage_l87_87921

theorem soccer_boys_percentage (total_students boys total_playing_soccer girls_not_playing_soccer : ℕ)
  (h_total_students : total_students = 500)
  (h_boys : boys = 350)
  (h_total_playing_soccer : total_playing_soccer = 250)
  (h_girls_not_playing_soccer : girls_not_playing_soccer = 115) :
  (boys - (total_students - total_playing_soccer) / total_playing_soccer * 100) = 86 :=
by
  sorry

end soccer_boys_percentage_l87_87921


namespace inverseP_l87_87888

-- Mathematical definitions
def isOdd (a : ℕ) : Prop := a % 2 = 1
def isPrime (a : ℕ) : Prop := Nat.Prime a

-- Given proposition P (hypothesis)
def P (a : ℕ) : Prop := isOdd a → isPrime a

-- Inverse proposition: if a is prime, then a is odd
theorem inverseP (a : ℕ) (h : isPrime a) : isOdd a :=
sorry

end inverseP_l87_87888


namespace find_inheritance_amount_l87_87233

noncomputable def totalInheritance (tax_amount : ℕ) : ℕ :=
  let federal_rate := 0.20
  let state_rate := 0.10
  let combined_rate := federal_rate + (state_rate * (1 - federal_rate))
  sorry

theorem find_inheritance_amount : totalInheritance 10500 = 37500 := 
  sorry

end find_inheritance_amount_l87_87233


namespace total_prize_money_l87_87959

theorem total_prize_money (P1 P2 P3 : ℕ) (d : ℕ) (total : ℕ) 
(h1 : P1 = 2000) (h2 : d = 400) (h3 : P2 = P1 - d) (h4 : P3 = P2 - d) 
(h5 : total = P1 + P2 + P3) : total = 4800 :=
sorry

end total_prize_money_l87_87959


namespace find_m_in_function_l87_87419

noncomputable def f (m : ℝ) (x : ℝ) := (1 / 3) * x^3 - x^2 - x + m

theorem find_m_in_function {m : ℝ} (h : ∀ x ∈ Set.Icc (0:ℝ) (1:ℝ), f m x ≥ (1/3)) :
  m = 2 :=
sorry

end find_m_in_function_l87_87419


namespace total_texts_received_l87_87791

open Nat 

-- Definition of conditions
def textsBeforeNoon : Nat := 21
def initialTextsAfterNoon : Nat := 2
def doublingTimeHours : Nat := 12

-- Definition to compute the total texts after noon recursively
def textsAfterNoon (n : Nat) : Nat :=
  if n = 0 then initialTextsAfterNoon
  else 2 * textsAfterNoon (n - 1)

-- Definition to sum the geometric series 
def sumGeometricSeries (a r n : Nat) : Nat :=
  if n = 0 then 0
  else a * (1 - r ^ n) / (1 - r)

-- Total text messages Debby received
def totalTextsReceived : Nat :=
  textsBeforeNoon + sumGeometricSeries initialTextsAfterNoon 2 doublingTimeHours

-- Proof statement
theorem total_texts_received: totalTextsReceived = 8211 := 
by 
  sorry

end total_texts_received_l87_87791


namespace sufficient_but_not_necessary_condition_l87_87564

theorem sufficient_but_not_necessary_condition (a : ℝ) : 
  (a > 0) → (|2 * a + 1| > 1) ∧ ¬((|2 * a + 1| > 1) → (a > 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l87_87564


namespace sufficient_not_necessary_condition_l87_87795

theorem sufficient_not_necessary_condition (x : ℝ) (h1 : 0 < x) (h2 : x < 2) : (0 < x ∧ x < 2) → (x^2 - x - 2 < 0) :=
by
  intros h
  sorry

end sufficient_not_necessary_condition_l87_87795


namespace sin_double_angle_shifted_l87_87181

theorem sin_double_angle_shifted (θ : ℝ) (h : Real.cos (θ + Real.pi) = - 1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = - 7 / 9 :=
by
  sorry

end sin_double_angle_shifted_l87_87181


namespace gcd_12_20_l87_87148

theorem gcd_12_20 : Nat.gcd 12 20 = 4 := by
  sorry

end gcd_12_20_l87_87148


namespace johns_age_is_15_l87_87884

-- Definitions from conditions
variables (J F : ℕ) -- J is John's age, F is his father's age
axiom sum_of_ages : J + F = 77
axiom father_age : F = 2 * J + 32

-- Target statement to prove
theorem johns_age_is_15 : J = 15 :=
by
  sorry

end johns_age_is_15_l87_87884


namespace tomas_first_month_distance_l87_87665

theorem tomas_first_month_distance 
  (distance_n_5 : ℝ := 26.3)
  (double_distance_each_month : ∀ (n : ℕ), n ≥ 1 → (distance_n : ℝ) = distance_n_5 / (2 ^ (5 - n)))
  : distance_n_5 / (2 ^ (5 - 1)) = 1.64375 :=
by
  sorry

end tomas_first_month_distance_l87_87665


namespace common_difference_range_l87_87436

noncomputable def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

theorem common_difference_range :
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  (a10 > 0) ∧ (a9 <= 0) → 8 / 3 < d ∧ d <= 3 :=
by
  let a1 := -24
  let a9 := arithmetic_sequence 9 a1 d
  let a10 := arithmetic_sequence 10 a1 d
  intro h
  sorry

end common_difference_range_l87_87436


namespace hyperbola_focal_length_l87_87738

theorem hyperbola_focal_length :
  let a := 2
  let b := Real.sqrt 3
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  2 * c = 2 * Real.sqrt 7 := 
by
  sorry

end hyperbola_focal_length_l87_87738


namespace part_1_part_2_l87_87414

theorem part_1 (a b A B : ℝ)
  (h : b * (Real.sin A)^2 = Real.sqrt 3 * a * Real.cos A * Real.sin B) 
  (h_sine_law : b / Real.sin B = a / Real.sin A)
  (A_in_range: A ∈ Set.Ioo 0 Real.pi):
  A = Real.pi / 3 := 
sorry

theorem part_2 (x : ℝ)
  (A : ℝ := Real.pi / 3)
  (h_sin_cos : ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
                f x = (Real.sin A * (Real.cos x)^2) - (Real.sin (A / 2))^2 * (Real.sin (2 * x))) :
  Set.image f (Set.Icc 0 (Real.pi / 2)) = Set.Icc ((Real.sqrt 3 - 2)/4) (Real.sqrt 3 / 2) :=
sorry

end part_1_part_2_l87_87414


namespace percent_runs_by_running_eq_18_75_l87_87183

/-
Define required conditions.
-/
def total_runs : ℕ := 224
def boundaries_runs : ℕ := 9 * 4
def sixes_runs : ℕ := 8 * 6
def twos_runs : ℕ := 12 * 2
def threes_runs : ℕ := 4 * 3
def byes_runs : ℕ := 6 * 1
def running_runs : ℕ := twos_runs + threes_runs + byes_runs

/-
Define the proof problem to show that the percentage of the total score made by running between the wickets is 18.75%.
-/
theorem percent_runs_by_running_eq_18_75 : (running_runs : ℚ) / total_runs * 100 = 18.75 := by
  sorry

end percent_runs_by_running_eq_18_75_l87_87183


namespace mark_owes_joanna_l87_87247

def dollars_per_room : ℚ := 12 / 3
def rooms_cleaned : ℚ := 9 / 4
def total_amount_owed : ℚ := 9

theorem mark_owes_joanna :
  dollars_per_room * rooms_cleaned = total_amount_owed :=
by
  sorry

end mark_owes_joanna_l87_87247


namespace conclusion_1_conclusion_3_l87_87902

def tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem conclusion_1 : tensor 2 (-2) = 6 :=
by sorry

theorem conclusion_3 (a b : ℝ) (h : a + b = 0) : tensor a a + tensor b b = 2 * a * b :=
by sorry

end conclusion_1_conclusion_3_l87_87902


namespace complex_square_eq_l87_87779

theorem complex_square_eq (i : ℂ) (hi : i * i = -1) : (1 + i)^2 = 2 * i := 
by {
  -- marking the end of existing code for clarity
  sorry
}

end complex_square_eq_l87_87779


namespace work_completion_l87_87683

theorem work_completion (A B : ℝ → ℝ) (h1 : ∀ t, A t = B t) (h3 : A 4 + B 4 = 1) : B 1 = 1/2 :=
by {
  sorry
}

end work_completion_l87_87683


namespace valid_b_values_count_l87_87242

theorem valid_b_values_count : 
  (∃! b : ℤ, ∃ x1 x2 x3 : ℤ, 
    (∀ x : ℤ, x^2 + b * x + 5 ≤ 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧ 
    (20 ≤ b^2 ∧ b^2 < 29)) :=
sorry

end valid_b_values_count_l87_87242


namespace score_on_fourth_board_l87_87707

theorem score_on_fourth_board 
  (score1 score2 score3 score4 : ℕ)
  (h1 : score1 = 30)
  (h2 : score2 = 38)
  (h3 : score3 = 41)
  (total_score : score1 + score2 = 2 * score4) :
  score4 = 34 := by
  sorry

end score_on_fourth_board_l87_87707


namespace angles_on_x_axis_eq_l87_87650

open Set

def S1 : Set ℝ := { β | ∃ k : ℤ, β = k * 360 }
def S2 : Set ℝ := { β | ∃ k : ℤ, β = 180 + k * 360 }
def S_total : Set ℝ := S1 ∪ S2
def S_target : Set ℝ := { β | ∃ n : ℤ, β = n * 180 }

theorem angles_on_x_axis_eq : S_total = S_target := 
by 
  sorry

end angles_on_x_axis_eq_l87_87650


namespace count_integer_values_of_x_l87_87032

theorem count_integer_values_of_x (x : ℕ) (h : ⌈Real.sqrt x⌉ = 12) : ∃ (n : ℕ), n = 23 :=
by
  sorry

end count_integer_values_of_x_l87_87032


namespace quadrilateral_angle_contradiction_l87_87093

theorem quadrilateral_angle_contradiction (a b c d : ℝ)
  (h : 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 0 < c ∧ c < 180 ∧ 0 < d ∧ d < 180)
  (sum_eq_360 : a + b + c + d = 360) :
  (¬ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) → (90 < a ∧ 90 < b ∧ 90 < c ∧ 90 < d) :=
sorry

end quadrilateral_angle_contradiction_l87_87093


namespace probability_one_hits_l87_87962

theorem probability_one_hits (P_A P_B : ℝ) (h_A : P_A = 0.6) (h_B : P_B = 0.6) :
  (P_A * (1 - P_B) + (1 - P_A) * P_B) = 0.48 :=
by
  sorry

end probability_one_hits_l87_87962


namespace dmitriev_is_older_l87_87744

variables (Alekseev Borisov Vasilyev Grigoryev Dima Dmitriev : ℤ)

def Lesha := Alekseev + 1
def Borya := Borisov + 2
def Vasya := Vasilyev + 3
def Grisha := Grigoryev + 4

theorem dmitriev_is_older :
  Dima + 10 = Dmitriev :=
sorry

end dmitriev_is_older_l87_87744


namespace male_female_ratio_l87_87923

-- Definitions and constants
variable (M F : ℕ) -- Number of male and female members respectively
variable (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) -- Average ticket sales condition

-- Statement of the theorem
theorem male_female_ratio (M F : ℕ) (h_avg_members : 66 * (M + F) = 58 * M + 70 * F) : M / F = 1 / 2 :=
sorry

end male_female_ratio_l87_87923


namespace alien_collected_95_units_l87_87143

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  match n with
  | 235 => 2 * 6^2 + 3 * 6^1 + 5 * 6^0
  | _ => 0

theorem alien_collected_95_units : convert_base_six_to_ten 235 = 95 := by
  sorry

end alien_collected_95_units_l87_87143


namespace income_expenditure_ratio_l87_87524

theorem income_expenditure_ratio (I E S : ℝ) (h1 : I = 20000) (h2 : S = 4000) (h3 : S = I - E) :
    I / E = 5 / 4 :=
sorry

end income_expenditure_ratio_l87_87524


namespace two_pow_gt_n_square_plus_one_l87_87484

theorem two_pow_gt_n_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := 
by {
  sorry
}

end two_pow_gt_n_square_plus_one_l87_87484


namespace problem_l87_87896

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem problem (A B : ℕ) (hA : A = gcf 9 15 27) (hB : B = lcm 9 15 27) : A + B = 138 :=
by
  sorry

end problem_l87_87896


namespace unique_three_digit_multiple_of_66_ending_in_4_l87_87443

theorem unique_three_digit_multiple_of_66_ending_in_4 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 66 = 0 ∧ n % 10 = 4 := sorry

end unique_three_digit_multiple_of_66_ending_in_4_l87_87443


namespace part1_part2_l87_87227

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end part1_part2_l87_87227


namespace b_minus_a_less_zero_l87_87068

-- Given conditions
variables {a b : ℝ}

-- Define the condition
def a_greater_b (a b : ℝ) : Prop := a > b

-- Lean 4 proof problem statement
theorem b_minus_a_less_zero (a b : ℝ) (h : a_greater_b a b) : b - a < 0 := 
sorry

end b_minus_a_less_zero_l87_87068


namespace cost_of_six_hotdogs_and_seven_burgers_l87_87969

theorem cost_of_six_hotdogs_and_seven_burgers :
  ∀ (h b : ℝ), 4 * h + 5 * b = 3.75 → 5 * h + 3 * b = 3.45 → 6 * h + 7 * b = 5.43 :=
by
  intros h b h_eqn b_eqn
  sorry

end cost_of_six_hotdogs_and_seven_burgers_l87_87969


namespace final_value_of_A_l87_87965

theorem final_value_of_A : 
  ∀ (A : Int), 
    (A = 20) → 
    (A = -A + 10) → 
    A = -10 :=
by
  intros A h1 h2
  sorry

end final_value_of_A_l87_87965


namespace chris_money_left_l87_87637

def video_game_cost : ℕ := 60
def candy_cost : ℕ := 5
def babysitting_rate : ℕ := 8
def hours_worked : ℕ := 9
def earnings : ℕ := babysitting_rate * hours_worked
def total_cost : ℕ := video_game_cost + candy_cost
def money_left : ℕ := earnings - total_cost

theorem chris_money_left
  (h1 : video_game_cost = 60)
  (h2 : candy_cost = 5)
  (h3 : babysitting_rate = 8)
  (h4 : hours_worked = 9) :
  money_left = 7 :=
by
  -- The detailed proof is omitted.
  sorry

end chris_money_left_l87_87637


namespace distinct_real_pairs_l87_87348

theorem distinct_real_pairs (x y : ℝ) (h1 : x ≠ y) (h2 : x^100 - y^100 = 2^99 * (x - y)) (h3 : x^200 - y^200 = 2^199 * (x - y)) :
  (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
sorry

end distinct_real_pairs_l87_87348


namespace average_goals_l87_87512

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

end average_goals_l87_87512


namespace number_of_chairs_l87_87569

theorem number_of_chairs (x t c b T C B: ℕ) (r1 r2 r3: ℕ)
  (h1: x = 2250) (h2: t = 18) (h3: c = 12) (h4: b = 30) 
  (h5: r1 = 2) (h6: r2 = 3) (h7: r3 = 1) 
  (h_ratio1: T / C = r1 / r2) (h_ratio2: B / C = r3 / r2) 
  (h_eq: t * T + c * C + b * B = x) : C = 66 :=
by
  sorry

end number_of_chairs_l87_87569


namespace value_of_each_gift_card_l87_87481

theorem value_of_each_gift_card (students total_thank_you_cards with_gift_cards total_value : ℕ) 
  (h1 : students = 50)
  (h2 : total_thank_you_cards = 30 * students / 100)
  (h3 : with_gift_cards = total_thank_you_cards / 3)
  (h4 : total_value = 50) :
  total_value / with_gift_cards = 10 := by
  sorry

end value_of_each_gift_card_l87_87481


namespace num_perfect_squares_l87_87815

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l87_87815


namespace smallest_number_is_32_l87_87422

theorem smallest_number_is_32 (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 25) (h3 : c = 25 + 8) : a = 32 :=
by {
  sorry
}

end smallest_number_is_32_l87_87422


namespace probability_at_least_one_exceeds_one_dollar_l87_87370

noncomputable def prob_A : ℚ := 2 / 3
noncomputable def prob_B : ℚ := 1 / 2
noncomputable def prob_C : ℚ := 1 / 4

theorem probability_at_least_one_exceeds_one_dollar :
  (1 - ((1 - prob_A) * (1 - prob_B) * (1 - prob_C))) = 7 / 8 :=
by
  -- The proof can be conducted here
  sorry

end probability_at_least_one_exceeds_one_dollar_l87_87370


namespace largest_base_b_digits_not_18_l87_87103

-- Definition of the problem:
-- Let n = 12^3 in base 10
def n : ℕ := 12 ^ 3

-- Definition of the conditions:
-- In base 8, 1728 (12^3 in base 10) has its digits sum to 17
def sum_of_digits_base_8 (x : ℕ) : ℕ :=
  let digits := x.digits (8)
  digits.sum

-- Proof statement
theorem largest_base_b_digits_not_18 : ∃ b : ℕ, (max b) = 8 ∧ sum_of_digits_base_8 n ≠ 18 := by
  sorry

end largest_base_b_digits_not_18_l87_87103


namespace correct_combined_monthly_rate_of_profit_l87_87828

structure Book :=
  (cost_price : ℕ)
  (selling_price : ℕ)
  (months_held : ℕ)

def profit (b : Book) : ℕ :=
  b.selling_price - b.cost_price

def monthly_rate_of_profit (b : Book) : ℕ :=
  if b.months_held = 0 then profit b else profit b / b.months_held

def combined_monthly_rate_of_profit (b1 b2 b3 : Book) : ℕ :=
  monthly_rate_of_profit b1 + monthly_rate_of_profit b2 + monthly_rate_of_profit b3

theorem correct_combined_monthly_rate_of_profit :
  combined_monthly_rate_of_profit
    {cost_price := 50, selling_price := 90, months_held := 1}
    {cost_price := 120, selling_price := 150, months_held := 2}
    {cost_price := 75, selling_price := 110, months_held := 0} 
    = 90 := 
by
  sorry

end correct_combined_monthly_rate_of_profit_l87_87828


namespace total_students_l87_87859

-- Definitions
def is_half_reading (S : ℕ) (half_reading : ℕ) := half_reading = S / 2
def is_third_playing (S : ℕ) (third_playing : ℕ) := third_playing = S / 3
def is_total_students (S half_reading third_playing homework : ℕ) := half_reading + third_playing + homework = S

-- Homework is given to be 4
def homework : ℕ := 4

-- Total number of students
theorem total_students (S : ℕ) (half_reading third_playing : ℕ)
    (h₁ : is_half_reading S half_reading) 
    (h₂ : is_third_playing S third_playing) 
    (h₃ : is_total_students S half_reading third_playing homework) :
    S = 24 := 
sorry

end total_students_l87_87859


namespace minimum_value_l87_87747

theorem minimum_value (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ z, z = 9 ∧ (forall x y, x > 0 ∧ y > 0 ∧ x + y = 1 → (1/x + 4/y) ≥ z) := 
sorry

end minimum_value_l87_87747


namespace geometric_sequence_eighth_term_l87_87994

theorem geometric_sequence_eighth_term 
  (a : ℕ) (r : ℕ) (h1 : a = 4) (h2 : r = 16 / 4) :
  a * r^(7) = 65536 :=
by
  sorry

end geometric_sequence_eighth_term_l87_87994


namespace number_of_integers_satisfying_l87_87283

theorem number_of_integers_satisfying (n : ℤ) : 
    (25 < n^2 ∧ n^2 < 144) → Finset.card (Finset.filter (fun n => 25 < n^2 ∧ n^2 < 144) (Finset.range 25)) = 12 := by
  sorry

end number_of_integers_satisfying_l87_87283


namespace total_cows_in_herd_l87_87447

theorem total_cows_in_herd {n : ℚ} (h1 : 1/3 + 1/6 + 1/9 = 11/18) 
                           (h2 : (1 - 11/18) = 7/18) 
                           (h3 : 8 = (7/18) * n) : 
                           n = 144/7 :=
by sorry

end total_cows_in_herd_l87_87447


namespace solve_equations_l87_87556

theorem solve_equations :
  (∀ x : ℝ, x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2) ∧ 
  (∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 ↔ x = 1/2) :=
by sorry

end solve_equations_l87_87556


namespace steve_initial_amount_l87_87052

theorem steve_initial_amount
  (P : ℝ) 
  (h : (1.1^2) * P = 121) : 
  P = 100 := 
by 
  sorry

end steve_initial_amount_l87_87052


namespace hyperbola_eccentricity_l87_87333

theorem hyperbola_eccentricity (m : ℝ) (h : 0 < m) :
  ∃ e, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2 → m > 1 :=
by
  sorry

end hyperbola_eccentricity_l87_87333


namespace peter_fraction_equiv_l87_87128

def fraction_pizza_peter_ate (total_slices : ℕ) (slices_ate_alone : ℕ) (shared_slices_brother : ℚ) (shared_slices_sister : ℚ) : ℚ :=
  (slices_ate_alone / total_slices) + (shared_slices_brother / total_slices) + (shared_slices_sister / total_slices)

theorem peter_fraction_equiv :
  fraction_pizza_peter_ate 16 3 (1/2) (1/2) = 1/4 :=
by
  sorry

end peter_fraction_equiv_l87_87128


namespace ten_integers_disjoint_subsets_same_sum_l87_87522

theorem ten_integers_disjoint_subsets_same_sum (S : Finset ℕ) (h : S.card = 10) (h_range : ∀ x ∈ S, 10 ≤ x ∧ x ≤ 99) :
  ∃ A B : Finset ℕ, A ≠ B ∧ A ⊆ S ∧ B ⊆ S ∧ A ∩ B = ∅ ∧ A.sum id = B.sum id :=
by sorry

end ten_integers_disjoint_subsets_same_sum_l87_87522


namespace additional_tobacco_acres_l87_87584

def original_land : ℕ := 1350
def original_ratio_units : ℕ := 9
def new_ratio_units : ℕ := 9

def acres_per_unit := original_land / original_ratio_units

def tobacco_old := 2 * acres_per_unit
def tobacco_new := 5 * acres_per_unit

theorem additional_tobacco_acres :
  tobacco_new - tobacco_old = 450 := by
  sorry

end additional_tobacco_acres_l87_87584


namespace divides_b_n_minus_n_l87_87623

theorem divides_b_n_minus_n (a b : ℕ) (h_a : a > 0) (h_b : b > 0) :
  ∃ n : ℕ, n > 0 ∧ a ∣ (b^n - n) :=
by
  sorry

end divides_b_n_minus_n_l87_87623


namespace triangle_altitude_sum_l87_87724

-- Problem Conditions
def line_eq (x y : ℝ) : Prop := 10 * x + 8 * y = 80

-- Altitudes Length Sum
theorem triangle_altitude_sum :
  ∀ x y : ℝ, line_eq x y → 
  ∀ (a b c: ℝ), a = 8 → b = 10 → c = 40 / Real.sqrt 41 →
  a + b + c = (18 * Real.sqrt 41 + 40) / Real.sqrt 41 :=
by
  sorry

end triangle_altitude_sum_l87_87724


namespace find_integers_a_b_c_l87_87317

theorem find_integers_a_b_c :
  ∃ a b c : ℤ, ((x - a) * (x - 12) + 1 = (x + b) * (x + c)) ∧ 
  ((b + 12) * (c + 12) = 1 → ((b = -11 ∧ c = -11) → a = 10) ∧ 
  ((b = -13 ∧ c = -13) → a = 14)) :=
by
  sorry

end find_integers_a_b_c_l87_87317


namespace probability_same_heads_l87_87810

noncomputable def probability_heads_after_flips (p : ℚ) (n : ℕ) : ℚ :=
  (1 - p)^(n-1) * p

theorem probability_same_heads (p : ℚ) (n : ℕ) : p = 1/3 → 
  ∑' n : ℕ, (probability_heads_after_flips p n)^4 = 1/65 := 
sorry

end probability_same_heads_l87_87810


namespace storks_more_than_birds_l87_87184

def birds := 4
def initial_storks := 3
def additional_storks := 6

theorem storks_more_than_birds :
  (initial_storks + additional_storks) - birds = 5 := 
by
  sorry

end storks_more_than_birds_l87_87184


namespace eval_log32_4_l87_87180

noncomputable def log_base_change (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem eval_log32_4 : log_base_change 32 4 = 2 / 5 := 
by 
  sorry

end eval_log32_4_l87_87180


namespace function_value_sum_l87_87809

namespace MathProof

variable {f : ℝ → ℝ}

theorem function_value_sum :
  (∀ x, f (-x) = -f x) →
  (∀ x, f (x + 5) = f x) →
  f (1 / 3) = 2022 →
  f (1 / 2) = 17 →
  f (-7) + f 12 + f (16 / 3) + f (9 / 2) = 2005 :=
by
  intros h_odd h_periodic h_f13 h_f12
  sorry

end MathProof

end function_value_sum_l87_87809


namespace find_m_l87_87108

theorem find_m (m : ℝ) (x : ℝ) (h : x = 1) (h_eq : (m / (2 - x)) - (1 / (x - 2)) = 3) : m = 2 :=
sorry

end find_m_l87_87108


namespace university_diploma_percentage_l87_87892

-- Define variables
variables (P U J : ℝ)  -- P: Percentage of total population (i.e., 1 or 100%), U: Having a university diploma, J: having the job of their choice
variables (h1 : 10 / 100 * P = 10 / 100 * P * (1 - U) * J)        -- 10% of the people do not have a university diploma but have the job of their choice
variables (h2 : 30 / 100 * (P * (1 - J)) = 30 / 100 * P * U * (1 - J))  -- 30% of the people who do not have the job of their choice have a university diploma
variables (h3 : 40 / 100 * P = 40 / 100 * P * J)                   -- 40% of the people have the job of their choice

-- Statement to prove
theorem university_diploma_percentage : 
  48 / 100 * P = (30 / 100 * P * J) + (18 / 100 * P * (1 - J)) :=
by sorry

end university_diploma_percentage_l87_87892


namespace gcd_48_72_120_l87_87435

theorem gcd_48_72_120 : Nat.gcd (Nat.gcd 48 72) 120 = 24 :=
by
  sorry

end gcd_48_72_120_l87_87435


namespace smallest_three_digit_pqr_l87_87051

theorem smallest_three_digit_pqr (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  100 ≤ p * q^2 * r ∧ p * q^2 * r < 1000 → p * q^2 * r = 126 := 
sorry

end smallest_three_digit_pqr_l87_87051


namespace find_a10_l87_87454

variable {a : ℕ → ℝ}
variable (h1 : ∀ n m, a (n + 1) = a n + a m)
variable (h2 : a 6 + a 8 = 16)
variable (h3 : a 4 = 1)

theorem find_a10 : a 10 = 15 := by
  sorry

end find_a10_l87_87454


namespace f_one_zero_inequality_solution_l87_87097

noncomputable def f : ℝ → ℝ := sorry

axiom increasing_f : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom functional_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y
axiom f_six : f 6 = 1

-- Part 1: Prove that f(1) = 0
theorem f_one_zero : f 1 = 0 := sorry

-- Part 2: Prove that ∀ x ∈ (0, (-3 + sqrt 153) / 2), f(x + 3) - f(1 / x) < 2
theorem inequality_solution : ∀ x, 0 < x → x < (-3 + Real.sqrt 153) / 2 → f (x + 3) - f (1 / x) < 2 := sorry

end f_one_zero_inequality_solution_l87_87097


namespace pokemon_card_cost_l87_87054

theorem pokemon_card_cost 
  (football_cost : ℝ)
  (num_football_packs : ℕ) 
  (baseball_cost : ℝ) 
  (total_spent : ℝ) 
  (h_football : football_cost = 2.73)
  (h_num_football_packs : num_football_packs = 2)
  (h_baseball : baseball_cost = 8.95)
  (h_total : total_spent = 18.42) :
  (total_spent - (num_football_packs * football_cost + baseball_cost) = 4.01) :=
by
  -- Proof goes here
  sorry

end pokemon_card_cost_l87_87054


namespace age_contradiction_l87_87820

-- Given the age ratios and future age of Sandy
def current_ages (x : ℕ) : ℕ × ℕ × ℕ := (4 * x, 3 * x, 5 * x)
def sandy_age_after_6_years (age_sandy_current : ℕ) : ℕ := age_sandy_current + 6

-- Given conditions
def ratio_condition (x : ℕ) (age_sandy age_molly age_danny : ℕ) : Prop :=
  current_ages x = (age_sandy, age_molly, age_danny)

def sandy_age_condition (age_sandy_current : ℕ) : Prop :=
  sandy_age_after_6_years age_sandy_current = 30

def age_sum_condition (age_molly age_danny : ℕ) : Prop :=
  age_molly + age_danny = (age_molly + 4) + (age_danny + 4)

-- Main theorem
theorem age_contradiction : ∃ x age_sandy age_molly age_danny, 
  ratio_condition x age_sandy age_molly age_danny ∧
  sandy_age_condition age_sandy ∧
  (¬ age_sum_condition age_molly age_danny) := 
by
  -- Omitting the proof; the focus is on setting up the statement only
  sorry

end age_contradiction_l87_87820


namespace probability_of_red_then_blue_is_correct_l87_87870

noncomputable def probability_red_then_blue : ℚ :=
  let total_marbles := 5 + 4 + 12 + 2
  let prob_red := 5 / total_marbles
  let remaining_marbles := total_marbles - 1
  let prob_blue_given_red := 2 / remaining_marbles
  prob_red * prob_blue_given_red

theorem probability_of_red_then_blue_is_correct :
  probability_red_then_blue = 5 / 253 := 
by 
  sorry

end probability_of_red_then_blue_is_correct_l87_87870


namespace sufficient_not_necessary_l87_87514

-- Definitions based on the conditions
def f1 (x y : ℝ) : Prop := x^2 + y^2 = 0
def f2 (x y : ℝ) : Prop := x * y = 0

-- The theorem we need to prove
theorem sufficient_not_necessary (x y : ℝ) : f1 x y → f2 x y ∧ ¬ (f2 x y → f1 x y) := 
by sorry

end sufficient_not_necessary_l87_87514


namespace infinite_solutions_l87_87954

-- Define the system of linear equations
def eq1 (x y : ℝ) : Prop := 3 * x - 4 * y = 1
def eq2 (x y : ℝ) : Prop := 6 * x - 8 * y = 2

-- State that there are an unlimited number of solutions
theorem infinite_solutions : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧
  ∀ y : ℝ, ∃ x : ℝ, eq1 x y :=
by
  sorry

end infinite_solutions_l87_87954


namespace inclination_angle_of_line_l87_87827

-- Definitions and conditions
def line_equation (x y : ℝ) : Prop := x - y + 3 = 0

-- Theorem statement
theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : angle = 45 := by
  sorry

end inclination_angle_of_line_l87_87827


namespace min_value_abc_l87_87629

open Real

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/10368 :=
sorry

end min_value_abc_l87_87629


namespace sum_bases_l87_87270

theorem sum_bases (R1 R2 : ℕ) (F1 F2 : ℚ)
  (h1 : F1 = (4 * R1 + 5) / (R1 ^ 2 - 1))
  (h2 : F2 = (5 * R1 + 4) / (R1 ^ 2 - 1))
  (h3 : F1 = (3 * R2 + 8) / (R2 ^ 2 - 1))
  (h4 : F2 = (6 * R2 + 1) / (R2 ^ 2 - 1)) :
  R1 + R2 = 19 :=
sorry

end sum_bases_l87_87270


namespace average_length_one_third_of_strings_l87_87979

theorem average_length_one_third_of_strings (average_six_strings : ℕ → ℕ → ℕ)
    (average_four_strings : ℕ → ℕ → ℕ)
    (total_length : ℕ → ℕ → ℕ)
    (n m : ℕ) :
    (n = 6) →
    (m = 4) →
    (average_six_strings 80 n = 480) →
    (average_four_strings 85 m = 340) →
    (total_length 2 70 = 140) →
    70 = (480 - 340) / 2 :=
by
  intros h_n h_m avg_six avg_four total_len
  sorry

end average_length_one_third_of_strings_l87_87979


namespace traders_gain_percentage_l87_87172

theorem traders_gain_percentage (C : ℝ) (h : 0 < C) : 
  let cost_of_100_pens := 100 * C
  let gain := 40 * C
  let selling_price := cost_of_100_pens + gain
  let gain_percentage := (gain / cost_of_100_pens) * 100
  gain_percentage = 40 := by
  sorry

end traders_gain_percentage_l87_87172


namespace problem_1_problem_2_l87_87277

-- Definitions for the sets A and B:

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | 2 * m - 1 < x ∧ x < 1 + m }

-- Problem 1: When m = -2, find A ∪ B
theorem problem_1 : set_A ∪ set_B (-2) = { x | -5 < x ∧ x ≤ 4 } :=
sorry

-- Problem 2: If A ∩ B = B, find the range of the real number m
theorem problem_2 : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≥ -1 :=
sorry

end problem_1_problem_2_l87_87277


namespace complement_of_angle_correct_l87_87651

noncomputable def complement_of_angle (α : ℝ) : ℝ := 90 - α

theorem complement_of_angle_correct (α : ℝ) (h : complement_of_angle α = 125 + 12 / 60) :
  complement_of_angle α = 35 + 12 / 60 :=
by
  sorry

end complement_of_angle_correct_l87_87651


namespace boat_upstream_time_l87_87444

theorem boat_upstream_time (v t : ℝ) (d c : ℝ) 
  (h1 : d = 24) (h2 : c = 1) (h3 : 4 * (v + c) = d) 
  (h4 : d / (v - c) = t) : t = 6 :=
by
  sorry

end boat_upstream_time_l87_87444


namespace ratio_a7_b7_l87_87928

variable {α : Type*}
variables {a_n b_n : ℕ → α} [AddGroup α] [Field α]
variables {S_n T_n : ℕ → α}

-- Define the sum of the first n terms for sequences a_n and b_n
def sum_of_first_terms_a (n : ℕ) := S_n n = (n * (a_n n + a_n (n-1))) / 2
def sum_of_first_terms_b (n : ℕ) := T_n n = (n * (b_n n + b_n (n-1))) / 2

-- Given condition about the ratio of sums
axiom ratio_condition (n : ℕ) : S_n n / T_n n = (3 * n - 2) / (2 * n + 1)

-- The statement to be proved
theorem ratio_a7_b7 : (a_n 7 / b_n 7) = (37 / 27) := sorry

end ratio_a7_b7_l87_87928


namespace circle_range_of_m_l87_87673

theorem circle_range_of_m (m : ℝ) :
  (∃ h k r : ℝ, (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2 ↔ x ^ 2 + y ^ 2 - x + y + m = 0)) ↔ (m < 1/2) :=
by
  sorry

end circle_range_of_m_l87_87673


namespace solve_arcsin_cos_eq_x_over_3_l87_87094

noncomputable def arcsin (x : Real) : Real := sorry
noncomputable def cos (x : Real) : Real := sorry

theorem solve_arcsin_cos_eq_x_over_3 :
  ∀ x,
  - (3 * Real.pi / 2) ≤ x ∧ x ≤ 3 * Real.pi / 2 →
  arcsin (cos x) = x / 3 →
  x = 3 * Real.pi / 10 ∨ x = 3 * Real.pi / 8 :=
sorry

end solve_arcsin_cos_eq_x_over_3_l87_87094


namespace second_fisherman_more_fish_l87_87885

-- Defining the conditions
def total_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def days_rate1 : ℕ := 30
def days_rate2 : ℕ := 60
def days_rate3 : ℕ := total_days - (days_rate1 + days_rate2)

-- Calculating the total number of fish caught by both fishermen
def total_fish_first_fisherman : ℕ := first_fisherman_rate * total_days
def total_fish_second_fisherman : ℕ := (second_fisherman_rate1 * days_rate1) + 
                                        (second_fisherman_rate2 * days_rate2) + 
                                        (second_fisherman_rate3 * days_rate3)

-- Theorem stating the difference in the number of fish caught
theorem second_fisherman_more_fish : (total_fish_second_fisherman - total_fish_first_fisherman) = 3 := 
by
  sorry

end second_fisherman_more_fish_l87_87885


namespace exponentiation_problem_l87_87150

theorem exponentiation_problem (a b : ℤ) (h : 3 ^ a * 9 ^ b = (1 / 3 : ℚ)) : a + 2 * b = -1 :=
sorry

end exponentiation_problem_l87_87150


namespace rex_lesson_schedule_l87_87335

-- Define the total lessons and weeks
def total_lessons : ℕ := 40
def weeks_completed : ℕ := 6
def weeks_remaining : ℕ := 4

-- Define the proof statement
theorem rex_lesson_schedule : (weeks_completed + weeks_remaining) * 4 = total_lessons := by
  -- Proof placeholder, to be filled in 
  sorry

end rex_lesson_schedule_l87_87335


namespace min_value_of_M_l87_87122

theorem min_value_of_M (P : ℕ → ℝ) (n : ℕ) (M : ℝ):
  (P 1 = 9 / 11) →
  (∀ n ≥ 2, P n = (3 / 4) * (P (n - 1)) + (2 / 3) * (1 - P (n - 1))) →
  (∀ n ≥ 2, P n ≤ M) →
  (M = 97 / 132) := 
sorry

end min_value_of_M_l87_87122


namespace cards_remaining_l87_87260

theorem cards_remaining (initial_cards : ℕ) (cards_given : ℕ) (remaining_cards : ℕ) :
  initial_cards = 242 → cards_given = 136 → remaining_cards = initial_cards - cards_given → remaining_cards = 106 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end cards_remaining_l87_87260


namespace negation_of_forall_ge_zero_l87_87781

theorem negation_of_forall_ge_zero :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 := by
  sorry

end negation_of_forall_ge_zero_l87_87781


namespace final_statement_l87_87316

variable (x : ℝ)

def seven_elevenths_of_five_thirteenths_eq_48 (x : ℝ) :=
  (7/11 : ℝ) * (5/13 : ℝ) * x = 48

def solve_for_x (x : ℝ) : Prop :=
  seven_elevenths_of_five_thirteenths_eq_48 x → x = 196

def calculate_315_percent_of_x (x : ℝ) : Prop :=
  solve_for_x x → 3.15 * x = 617.4

theorem final_statement : calculate_315_percent_of_x x :=
sorry  -- Proof omitted

end final_statement_l87_87316


namespace part_a_l87_87485

theorem part_a (n : ℕ) (h_n : n ≥ 3) (x : Fin n → ℝ) (hx : ∀ i j : Fin n, i ≠ j → x i ≠ x j) (hx_pos : ∀ i : Fin n, 0 < x i) :
  ∃ (i j : Fin n), i ≠ j ∧ 0 < (x i - x j) / (1 + (x i) * (x j)) ∧ (x i - x j) / (1 + (x i) * (x j)) < Real.tan (π / (2 * (n - 1))) :=
by
  sorry

end part_a_l87_87485


namespace pipe_fill_without_hole_l87_87987

theorem pipe_fill_without_hole :
  ∀ (T : ℝ), 
  (1 / T - 1 / 60 = 1 / 20) → 
  T = 15 := 
by
  intros T h
  sorry

end pipe_fill_without_hole_l87_87987


namespace count_ways_with_3_in_M_count_ways_with_2_in_M_l87_87757

structure ArrangementConfig where
  positions : Fin 9 → ℕ
  unique_positions : ∀ (i j : Fin 9) (hi hj : i ≠ j), positions i ≠ positions j
  no_adjacent_same : ∀ (i : Fin 8), positions i ≠ positions (i + 1)

def count_arrangements (fixed_value : ℕ) (fixed_position : Fin 9) : ℕ :=
  -- Implementation of counting the valid arrangements
  sorry

theorem count_ways_with_3_in_M : count_arrangements 3 0 = 6 := sorry

theorem count_ways_with_2_in_M : count_arrangements 2 0 = 12 := sorry

end count_ways_with_3_in_M_count_ways_with_2_in_M_l87_87757


namespace range_of_a_l87_87690

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) * x^3

theorem range_of_a (a : ℝ) :
  f (Real.logb 2 a) + f (Real.logb 0.5 a) ≤ 2 * f 1 → (1/2 : ℝ) ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l87_87690


namespace gravitational_equal_forces_point_l87_87015

variable (d M m : ℝ) (hM : 0 < M) (hm : 0 < m) (hd : 0 < d)

theorem gravitational_equal_forces_point :
  ∃ x : ℝ, (0 < x ∧ x < d) ∧ x = d / (1 + Real.sqrt (m / M)) :=
by
  sorry

end gravitational_equal_forces_point_l87_87015


namespace largest_divisor_of_n4_minus_n_l87_87967

theorem largest_divisor_of_n4_minus_n (n : ℕ) (h : ¬(Prime n) ∧ n ≠ 1) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_of_n4_minus_n_l87_87967


namespace find_x_squared_plus_y_squared_l87_87609

theorem find_x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (x - y)^2 = 49) (h2 : x * y = -12) : x^2 + y^2 = 25 := 
by 
  sorry

end find_x_squared_plus_y_squared_l87_87609


namespace luke_payments_difference_l87_87165

theorem luke_payments_difference :
  let principal := 12000
  let rate := 0.08
  let years := 10
  let n_quarterly := 4
  let n_annually := 1
  let quarterly_rate := rate / n_quarterly
  let annually_rate := rate / n_annually
  let balance_plan1_5years := principal * (1 + quarterly_rate)^(n_quarterly * 5)
  let payment_plan1_5years := balance_plan1_5years / 3
  let remaining_balance_plan1_5years := balance_plan1_5years - payment_plan1_5years
  let final_balance_plan1_10years := remaining_balance_plan1_5years * (1 + quarterly_rate)^(n_quarterly * 5)
  let total_payment_plan1 := payment_plan1_5years + final_balance_plan1_10years
  let final_balance_plan2_10years := principal * (1 + annually_rate)^years
  (total_payment_plan1 - final_balance_plan2_10years).abs = 1022 :=
by
  sorry

end luke_payments_difference_l87_87165


namespace weight_of_each_bag_is_7_l87_87692

-- Defining the conditions
def morning_bags : ℕ := 29
def afternoon_bags : ℕ := 17
def total_weight : ℕ := 322

-- Defining the question in terms of proving a specific weight per bag
def bags_sold := morning_bags + afternoon_bags
def weight_per_bag (w : ℕ) := total_weight = bags_sold * w

-- Proving the question == answer under the given conditions
theorem weight_of_each_bag_is_7 :
  ∃ w : ℕ, weight_per_bag w ∧ w = 7 :=
by
  sorry

end weight_of_each_bag_is_7_l87_87692


namespace students_with_both_pets_l87_87151

theorem students_with_both_pets
  (D C : Finset ℕ)
  (h_union : (D ∪ C).card = 48)
  (h_D : D.card = 30)
  (h_C : C.card = 34) :
  (D ∩ C).card = 16 :=
by sorry

end students_with_both_pets_l87_87151


namespace equation_of_line_passing_through_ellipse_midpoint_l87_87036

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem equation_of_line_passing_through_ellipse_midpoint
  (x1 y1 x2 y2 : ℝ)
  (P : ℝ × ℝ)
  (hP : P = (1, 1))
  (hA : ellipse x1 y1)
  (hB : ellipse x2 y2)
  (midAB : (x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = 1) :
  ∃ (a b c : ℝ), a = 4 ∧ b = 3 ∧ c = -7 ∧ a * P.2 + b * P.1 + c = 0 :=
sorry

end equation_of_line_passing_through_ellipse_midpoint_l87_87036


namespace adam_bought_26_books_l87_87237

-- Conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def avg_books_per_shelf : ℕ := 20
def leftover_books : ℕ := 2

-- Definitions based on conditions
def capacity_books : ℕ := shelves * avg_books_per_shelf
def total_books_after_trip : ℕ := capacity_books + leftover_books

-- Question: How many books did Adam buy on his shopping trip?
def books_bought : ℕ := total_books_after_trip - initial_books

theorem adam_bought_26_books :
  books_bought = 26 :=
by
  sorry

end adam_bought_26_books_l87_87237


namespace base8_9257_digits_product_sum_l87_87248

theorem base8_9257_digits_product_sum :
  let base10 := 9257
  let base8_digits := [2, 2, 0, 5, 1] -- base 8 representation of 9257
  let product_of_digits := 2 * 2 * 0 * 5 * 1
  let sum_of_digits := 2 + 2 + 0 + 5 + 1
  product_of_digits = 0 ∧ sum_of_digits = 10 := 
by
  sorry

end base8_9257_digits_product_sum_l87_87248


namespace star_five_three_l87_87355

def star (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem star_five_three : star 5 3 = 14 := by
  sorry

end star_five_three_l87_87355


namespace fraction_identity_l87_87385

theorem fraction_identity
  (m : ℝ)
  (h : (m - 1) / m = 3) : (m^2 + 1) / m^2 = 5 :=
by
  sorry

end fraction_identity_l87_87385


namespace maximum_value_squared_l87_87166

theorem maximum_value_squared (a b : ℝ) (h₁ : 0 < b) (h₂ : b ≤ a) :
  (∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a + x)^2 + (b - y)^2) →
  (a / b)^2 ≤ 4 / 3 := 
sorry

end maximum_value_squared_l87_87166


namespace perimeters_equal_l87_87635

noncomputable def side_length_square := 15 -- cm
noncomputable def length_rectangle := 18 -- cm
noncomputable def area_rectangle := 216 -- cm²

theorem perimeters_equal :
  let perimeter_square := 4 * side_length_square
  let width_rectangle := area_rectangle / length_rectangle
  let perimeter_rectangle := 2 * (length_rectangle + width_rectangle)
  perimeter_square = perimeter_rectangle :=
by
  sorry

end perimeters_equal_l87_87635


namespace percentage_of_stock_l87_87532

-- Definitions based on conditions
def income := 500  -- I
def investment := 1500  -- Inv
def price := 90  -- Price

-- Initiate the Lean 4 statement for the proof
theorem percentage_of_stock (P : ℝ) (h : income = (investment * P) / price) : P = 30 :=
by
  sorry

end percentage_of_stock_l87_87532


namespace probability_diagonals_intersect_hexagon_l87_87306

theorem probability_diagonals_intersect_hexagon:
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2 -- Total number of diagonals in a convex polygon
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2 -- Total number of ways to choose 2 diagonals
  let non_principal_intersections := 3 * 6 -- Each of 6 non-principal diagonals intersects 3 others
  let principal_intersections := 4 * 3 -- Each of 3 principal diagonals intersects 4 others
  let total_intersections := (non_principal_intersections + principal_intersections) / 2 -- Correcting for double-counting
  let probability := total_intersections / total_pairs -- Probability of intersection inside the hexagon
  probability = 5 / 12 := by
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2
  let non_principal_intersections := 3 * 6
  let principal_intersections := 4 * 3
  let total_intersections := (non_principal_intersections + principal_intersections) / 2
  let probability := total_intersections / total_pairs
  have h : total_diagonals = 9 := by norm_num
  have h_pairs : total_pairs = 36 := by norm_num
  have h_intersections : total_intersections = 15 := by norm_num
  have h_prob : probability = 5 / 12 := by norm_num
  exact h_prob

end probability_diagonals_intersect_hexagon_l87_87306


namespace Delaney_missed_bus_by_l87_87210

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end Delaney_missed_bus_by_l87_87210


namespace find_t_l87_87719

-- Define sets M and N
def M (t : ℝ) : Set ℝ := {1, t^2}
def N (t : ℝ) : Set ℝ := {-2, t + 2}

-- Goal: prove that t = 2 given M ∩ N ≠ ∅
theorem find_t (t : ℝ) (h : (M t ∩ N t).Nonempty) : t = 2 :=
sorry

end find_t_l87_87719


namespace line_passes_through_fixed_point_l87_87544

theorem line_passes_through_fixed_point 
  (m : ℝ) : ∃ x y : ℝ, y = m * x + (2 * m + 1) ∧ (x, y) = (-2, 1) :=
by
  use (-2), (1)
  sorry

end line_passes_through_fixed_point_l87_87544


namespace yard_length_l87_87763

theorem yard_length (trees : ℕ) (distance_per_gap : ℕ) (gaps : ℕ) :
  trees = 26 → distance_per_gap = 16 → gaps = trees - 1 → length_of_yard = gaps * distance_per_gap → length_of_yard = 400 :=
by 
  intros h_trees h_distance_per_gap h_gaps h_length_of_yard
  sorry

end yard_length_l87_87763


namespace days_playing_video_games_l87_87766

-- Define the conditions
def watchesTVDailyHours : ℕ := 4
def videoGameHoursPerPlay : ℕ := 2
def totalWeeklyHours : ℕ := 34
def weeklyTVDailyHours : ℕ := 7 * watchesTVDailyHours

-- Define the number of days playing video games
def playsVideoGamesDays (d : ℕ) : ℕ := d * videoGameHoursPerPlay

-- Define the number of days Mike plays video games
theorem days_playing_video_games (d : ℕ) :
  weeklyTVDailyHours + playsVideoGamesDays d = totalWeeklyHours → d = 3 :=
by
  -- The proof is omitted
  sorry

end days_playing_video_games_l87_87766


namespace canoe_kayak_ratio_l87_87022

-- Define the number of canoes and kayaks
variables (c k : ℕ)

-- Define the conditions
def rental_cost_eq : Prop := 15 * c + 18 * k = 405
def canoe_more_kayak_eq : Prop := c = k + 5

-- Statement to prove
theorem canoe_kayak_ratio (h1 : rental_cost_eq c k) (h2 : canoe_more_kayak_eq c k) : c / k = 3 / 2 :=
by sorry

end canoe_kayak_ratio_l87_87022


namespace probability_divisor_of_8_is_half_l87_87679

def divisors (n : ℕ) : List ℕ := 
  List.filter (λ x => n % x = 0) (List.range (n + 1))

def num_divisors : ℕ := (divisors 8).length
def total_outcomes : ℕ := 8

theorem probability_divisor_of_8_is_half :
  (num_divisors / total_outcomes : ℚ) = 1 / 2 :=
by
  sorry

end probability_divisor_of_8_is_half_l87_87679


namespace Madison_minimum_score_l87_87895

theorem Madison_minimum_score (q1 q2 q3 q4 q5 : ℕ) (h1 : q1 = 84) (h2 : q2 = 81) (h3 : q3 = 87) (h4 : q4 = 83) (h5 : 85 * 5 ≤ q1 + q2 + q3 + q4 + q5) : 
  90 ≤ q5 := 
by
  sorry

end Madison_minimum_score_l87_87895


namespace sum_in_correct_range_l87_87035

-- Define the mixed numbers
def mixed1 := 1 + 1/4
def mixed2 := 4 + 1/3
def mixed3 := 6 + 1/12

-- Their sum
def sumMixed := mixed1 + mixed2 + mixed3

-- Correct sum in mixed number form
def correctSum := 11 + 2/3

-- Range we need to check
def lowerBound := 11 + 1/2
def upperBound := 12

theorem sum_in_correct_range : sumMixed = correctSum ∧ lowerBound < correctSum ∧ correctSum < upperBound := by
  sorry

end sum_in_correct_range_l87_87035


namespace abs_diff_gt_half_prob_l87_87915

noncomputable def probability_abs_diff_gt_half : ℝ :=
  ((1 / 4) * (1 / 8) + 
   (1 / 8) * (1 / 2) + 
   (1 / 8) * 1) * 2

theorem abs_diff_gt_half_prob : probability_abs_diff_gt_half = 5 / 16 := by 
  sorry

end abs_diff_gt_half_prob_l87_87915


namespace triangle_inequality_l87_87255

theorem triangle_inequality (a b c m_A : ℝ)
  (h1 : 2*m_A ≤ b + c)
  (h2 : a^2 + (2*m_A)^2 = (b^2) + (c^2)) :
  a^2 + 4*m_A^2 ≤ (b + c)^2 :=
by {
  sorry
}

end triangle_inequality_l87_87255


namespace box_width_l87_87852

theorem box_width (W S : ℕ) (h1 : 30 * W * 12 = 80 * S^3) (h2 : S ∣ 30 ∧ S ∣ 12) : W = 48 :=
by
  sorry

end box_width_l87_87852


namespace count_3_digit_multiples_of_13_l87_87173

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end count_3_digit_multiples_of_13_l87_87173


namespace factorization_problem_l87_87653

theorem factorization_problem (p q : ℝ) :
  (∃ a b c : ℝ, 
    x^4 + p * x^2 + q = (x^2 + 2 * x + 5) * (a * x^2 + b * x + c)) ↔
  p = 6 ∧ q = 25 := 
sorry

end factorization_problem_l87_87653


namespace find_c_for_min_value_zero_l87_87663

theorem find_c_for_min_value_zero :
  ∃ c : ℝ, c = 1 ∧ (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 ≥ 0) ∧
  (∀ x y : ℝ, 5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 6 * x - 6 * y + 9 = 0 → c = 1) :=
by
  use 1
  sorry

end find_c_for_min_value_zero_l87_87663


namespace sushi_father_lollipops_l87_87091

-- Define the conditions
def lollipops_eaten : ℕ := 5
def lollipops_left : ℕ := 7

-- Define the total number of lollipops brought
def total_lollipops := lollipops_eaten + lollipops_left

-- Proof statement
theorem sushi_father_lollipops : total_lollipops = 12 := sorry

end sushi_father_lollipops_l87_87091


namespace production_average_l87_87357

theorem production_average (n : ℕ) (P : ℕ) (h1 : P / n = 50) (h2 : (P + 90) / (n + 1) = 54) : n = 9 :=
sorry

end production_average_l87_87357


namespace taylor_family_reunion_l87_87458

theorem taylor_family_reunion :
  let number_of_kids := 45
  let number_of_adults := 123
  let number_of_tables := 14
  (number_of_kids + number_of_adults) / number_of_tables = 12 := by sorry

end taylor_family_reunion_l87_87458


namespace ducks_among_non_falcons_l87_87806

-- Definitions based on conditions
def percentage_birds := 100
def percentage_ducks := 40
def percentage_cranes := 20
def percentage_falcons := 15
def percentage_pigeons := 25

-- Question converted into the statement
theorem ducks_among_non_falcons : 
  (percentage_ducks / (percentage_birds - percentage_falcons) * percentage_birds) = 47 :=
by
  sorry

end ducks_among_non_falcons_l87_87806


namespace base_of_exponential_function_l87_87159

theorem base_of_exponential_function (a : ℝ) (h : ∀ x : ℝ, y = a^x) :
  (a > 1 ∧ (a - 1 / a = 1)) ∨ (0 < a ∧ a < 1 ∧ (1 / a - a = 1)) → 
  a = (1 + Real.sqrt 5) / 2 ∨ a = (Real.sqrt 5 - 1) / 2 :=
by sorry

end base_of_exponential_function_l87_87159


namespace tara_road_trip_cost_l87_87706

theorem tara_road_trip_cost :
  let tank_capacity := 12
  let price1 := 3
  let price2 := 3.50
  let price3 := 4
  let price4 := 4.50
  (price1 * tank_capacity) + (price2 * tank_capacity) + (price3 * tank_capacity) + (price4 * tank_capacity) = 180 :=
by
  sorry

end tara_road_trip_cost_l87_87706


namespace penultimate_digit_odd_of_square_last_digit_six_l87_87078

theorem penultimate_digit_odd_of_square_last_digit_six 
  (n : ℕ) 
  (h : (n * n) % 10 = 6) : 
  ((n * n) / 10) % 2 = 1 :=
sorry

end penultimate_digit_odd_of_square_last_digit_six_l87_87078


namespace determine_x_l87_87712

theorem determine_x (x : ℝ) (h : 9 * x^2 + 2 * x^2 + 3 * x^2 / 2 = 300) : x = 2 * Real.sqrt 6 :=
by sorry

end determine_x_l87_87712


namespace length_of_AB_area_of_ΔABF1_l87_87666

theorem length_of_AB (A B : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3)) →
  |((x1 - x2)^2 + (y1 - y2)^2)^(1/2)| = (8 / 3) * (2)^(1/2) :=
by sorry

theorem area_of_ΔABF1 (A B F1 : ℝ × ℝ) (x1 x2 y1 y2 : ℝ) :
  (F1 = (0, -2)) ∧ ((y1 = x1 - 2) ∧ (y2 = x2 - 2) ∧ ((x1 = 0) ∧ (y1 = -2)) ∧ ((x2 = 8/3) ∧ (y2 = 2/3))) →
  (1/2) * (((x1 - x2)^2 + (y1 - y2)^2)^(1/2)) * (|(-2-2)/((2)^(1/2))|) = 16 / 3 :=
by sorry

end length_of_AB_area_of_ΔABF1_l87_87666


namespace inequality_xy_l87_87354

-- Defining the constants and conditions
variables {x y : ℝ}

-- Main theorem to prove the inequality and find pairs for equality
theorem inequality_xy (h : (x + 1) * (y + 2) = 8) :
  (xy - 10)^2 ≥ 64 ∧ ((xy - 10)^2 = 64 → (x, y) = (1, 2) ∨ (x, y) = (-3, -6)) :=
sorry

end inequality_xy_l87_87354


namespace mars_moon_cost_share_l87_87536

theorem mars_moon_cost_share :
  let total_cost := 40 * 10^9 -- total cost in dollars
  let num_people := 200 * 10^6 -- number of people sharing the cost
  (total_cost / num_people) = 200 := by
  sorry

end mars_moon_cost_share_l87_87536


namespace min_S_in_grid_l87_87452

def valid_grid (grid : Fin 10 × Fin 10 → Fin 100) (S : ℕ) : Prop :=
  ∀ i j, 
    (i < 9 → grid (i, j) + grid (i + 1, j) ≤ S) ∧
    (j < 9 → grid (i, j) + grid (i, j + 1) ≤ S)

theorem min_S_in_grid : ∃ grid : Fin 10 × Fin 10 → Fin 100, ∃ S : ℕ, valid_grid grid S ∧ 
  (∀ (other_S : ℕ), valid_grid grid other_S → S ≤ other_S) ∧ S = 106 :=
sorry

end min_S_in_grid_l87_87452


namespace solve_for_y_l87_87649

theorem solve_for_y {y : ℝ} : 
  (2012 + y)^2 = 2 * y^2 ↔ y = 2012 * (Real.sqrt 2 + 1) ∨ y = -2012 * (Real.sqrt 2 - 1) := by
  sorry

end solve_for_y_l87_87649


namespace find_f2a_eq_zero_l87_87922

variable {α : Type} [LinearOrderedField α]

-- Definitions for the function f and its inverse
variable (f : α → α)
variable (finv : α → α)

-- Given conditions
variable (a : α)
variable (h_nonzero : a ≠ 0)
variable (h_inverse1 : ∀ x : α, finv (x + a) = f (x + a)⁻¹)
variable (h_inverse2 : ∀ x : α, f (x) = finv⁻¹ x)
variable (h_fa : f a = a)

-- Statement to be proved in Lean
theorem find_f2a_eq_zero : f (2 * a) = 0 :=
sorry

end find_f2a_eq_zero_l87_87922


namespace old_edition_pages_l87_87823

-- Define the conditions
variables (new_edition : ℕ) (old_edition : ℕ)

-- The conditions given in the problem
axiom new_edition_pages : new_edition = 450
axiom pages_relationship : new_edition = 2 * old_edition - 230

-- Goal: Prove that the old edition Geometry book had 340 pages
theorem old_edition_pages : old_edition = 340 :=
by sorry

end old_edition_pages_l87_87823


namespace alan_more_wings_per_minute_to_beat_record_l87_87937

-- Define relevant parameters and conditions
def kevin_wings := 64
def time_minutes := 8
def alan_rate := 5

-- Theorem: Alan must eat 3 more wings per minute to beat Kevin's record
theorem alan_more_wings_per_minute_to_beat_record : 
  (kevin_wings > alan_rate * time_minutes) → ((kevin_wings - (alan_rate * time_minutes)) / time_minutes = 3) :=
by
  sorry

end alan_more_wings_per_minute_to_beat_record_l87_87937


namespace point_on_parabola_distance_l87_87709

theorem point_on_parabola_distance (a b : ℝ) (h1 : a^2 = 20 * b) (h2 : |b + 5| = 25) : |a * b| = 400 :=
sorry

end point_on_parabola_distance_l87_87709


namespace hexagon_coloring_count_l87_87239

-- Defining the conditions
def has7Colors : Type := Fin 7

-- Hexagon vertices
inductive Vertex
| A | B | C | D | E | F

-- Adjacent vertices
def adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.B => true
| Vertex.B, Vertex.C => true
| Vertex.C, Vertex.D => true
| Vertex.D, Vertex.E => true
| Vertex.E, Vertex.F => true
| Vertex.F, Vertex.A => true
| _, _ => false

-- Non-adjacent vertices (diagonals)
def non_adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.C => true
| Vertex.A, Vertex.D => true
| Vertex.B, Vertex.D => true
| Vertex.B, Vertex.E => true
| Vertex.C, Vertex.E => true
| Vertex.C, Vertex.F => true
| Vertex.D, Vertex.F => true
| Vertex.E, Vertex.A => true
| Vertex.F, Vertex.A => true
| Vertex.F, Vertex.B => true
| _, _ => false

-- Coloring function
def valid_coloring (coloring : Vertex → has7Colors) : Prop :=
  (∀ v1 v2, adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2, non_adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2 v3, adjacent v1 v2 → adjacent v2 v3 → adjacent v1 v3 → coloring v1 ≠ coloring v3)

noncomputable def count_valid_colorings : Nat :=
  -- This is a placeholder for the count function
  sorry

theorem hexagon_coloring_count : count_valid_colorings = 21000 := 
  sorry

end hexagon_coloring_count_l87_87239


namespace second_number_is_22_l87_87021

noncomputable section

variables (x y : ℕ)

-- Definitions based on the conditions
-- Condition 1: The sum of two numbers is 33
def sum_condition : Prop := x + y = 33

-- Condition 2: The second number is twice the first number
def twice_condition : Prop := y = 2 * x

-- Theorem: Given the conditions, the second number y is 22.
theorem second_number_is_22 (h1 : sum_condition x y) (h2 : twice_condition x y) : y = 22 :=
by
  sorry

end second_number_is_22_l87_87021


namespace parabola_focus_coordinates_l87_87491

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), y = 4 * x^2 → (0, y / 16) = (0, 1 / 16) :=
by
  intros x y h
  sorry

end parabola_focus_coordinates_l87_87491


namespace inequality_correct_l87_87302

theorem inequality_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1 / a) < (1 / b) :=
sorry

end inequality_correct_l87_87302


namespace barrels_oil_total_l87_87342

theorem barrels_oil_total :
  let A := 3 / 4
  let B := A + 1 / 10
  A + B = 8 / 5 := by
  sorry

end barrels_oil_total_l87_87342


namespace minimum_f_l87_87599

def f (x : ℝ) : ℝ := |x - 2| + |5 - x|

theorem minimum_f : ∃ x, f x = 3 :=
by
  use 3
  unfold f
  sorry

end minimum_f_l87_87599


namespace problem_statement_l87_87952

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f a b α β 2007 = 5) :
  f a b α β 2008 = 3 := 
by
  sorry

end problem_statement_l87_87952


namespace find_adult_buffet_price_l87_87518

variable {A : ℝ} -- Let A be the price for the adult buffet
variable (children_cost : ℝ := 45) -- Total cost for the children's buffet
variable (senior_discount : ℝ := 0.9) -- Discount for senior citizens
variable (total_cost : ℝ := 159) -- Total amount spent by Mr. Smith
variable (num_adults : ℕ := 2) -- Number of adults (Mr. Smith and his wife)
variable (num_seniors : ℕ := 2) -- Number of senior citizens

theorem find_adult_buffet_price (h1 : children_cost = 45)
    (h2 : total_cost = 159)
    (h3 : ∀ x, num_adults * x + num_seniors * (senior_discount * x) + children_cost = total_cost)
    : A = 30 :=
by
  sorry

end find_adult_buffet_price_l87_87518


namespace smallest_four_digit_divisible_by_8_with_3_even_1_odd_l87_87334

theorem smallest_four_digit_divisible_by_8_with_3_even_1_odd : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ 
  (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
    (d1 % 2 = 0) ∧ (d2 % 2 = 0 ∨ d2 % 2 ≠ 0) ∧ 
    (d3 % 2 = 0) ∧ (d4 % 2 = 0 ∨ d4 % 2 ≠ 0) ∧ 
    (d2 % 2 ≠ 0 ∨ d4 % 2 ≠ 0) ) ∧ n = 1248 :=
by
  sorry

end smallest_four_digit_divisible_by_8_with_3_even_1_odd_l87_87334


namespace reggies_brother_long_shots_l87_87882

-- Define the number of points per type of shot
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define the number of shots made by Reggie
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := 1

-- Define the total number of points made by Reggie
def reggie_points : ℕ :=
  reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points

-- Define the total points by which Reggie loses
def points_lost_by : ℕ := 2

-- Prove the number of long shots made by Reggie's brother
theorem reggies_brother_long_shots : 
  (reggie_points + points_lost_by) / long_shot_points = 4 := by
  sorry

end reggies_brother_long_shots_l87_87882


namespace pig_duck_ratio_l87_87012

theorem pig_duck_ratio (G C D P : ℕ)
(h₁ : G = 66)
(h₂ : C = 2 * G)
(h₃ : D = (G + C) / 2)
(h₄ : P = G - 33) :
  P / D = 1 / 3 :=
by {
  sorry
}

end pig_duck_ratio_l87_87012


namespace sum_difference_4041_l87_87363

def sum_of_first_n_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_difference_4041 :
  sum_of_first_n_integers 2021 - sum_of_first_n_integers 2019 = 4041 :=
by
  sorry

end sum_difference_4041_l87_87363


namespace odd_n_cube_plus_one_not_square_l87_87535

theorem odd_n_cube_plus_one_not_square (n : ℤ) (h : n % 2 = 1) : ¬ ∃ (x : ℤ), x^2 = n^3 + 1 :=
by
  sorry

end odd_n_cube_plus_one_not_square_l87_87535


namespace percent_cities_less_than_50000_l87_87701

-- Definitions of the conditions
def percent_cities_50000_to_149999 := 40
def percent_cities_less_than_10000 := 35
def percent_cities_10000_to_49999 := 10
def percent_cities_150000_or_more := 15

-- Prove that the total percentage of cities with fewer than 50,000 residents is 45%
theorem percent_cities_less_than_50000 :
  percent_cities_less_than_10000 + percent_cities_10000_to_49999 = 45 :=
by
  sorry

end percent_cities_less_than_50000_l87_87701


namespace find_width_of_bobs_tv_l87_87745

def area (w h : ℕ) : ℕ := w * h

def weight_in_oz (area : ℕ) : ℕ := area * 4

def weight_in_lb (weight_in_oz : ℕ) : ℕ := weight_in_oz / 16

def width_of_bobs_tv (x : ℕ) : Prop :=
  area 48 100 = 4800 ∧
  weight_in_lb (weight_in_oz (area 48 100)) = 1200 ∧
  weight_in_lb (weight_in_oz (area x 60)) = 15 * x ∧
  15 * x = 1350

theorem find_width_of_bobs_tv : ∃ x : ℕ, width_of_bobs_tv x := sorry

end find_width_of_bobs_tv_l87_87745


namespace phone_number_fraction_l87_87356

theorem phone_number_fraction : 
  let total_valid_numbers := 6 * (10^6)
  let valid_numbers_with_conditions := 10^5
  valid_numbers_with_conditions / total_valid_numbers = 1 / 60 :=
by sorry

end phone_number_fraction_l87_87356


namespace find_m_l87_87486

/-- Given vectors \(\overrightarrow{OA} = (1, m)\) and \(\overrightarrow{OB} = (m-1, 2)\), if 
\(\overrightarrow{OA} \perp \overrightarrow{AB}\), then \(m = \frac{1}{3}\). -/
theorem find_m (m : ℝ) (h : (1, m).1 * (m - 1 - 1, 2 - m).1 + (1, m).2 * (m - 1 - 1, 2 - m).2 = 0) :
  m = 1 / 3 :=
sorry

end find_m_l87_87486


namespace simplify_expression_l87_87652

theorem simplify_expression :
  let a := 7
  let b := 2
  (a^5 + b^8) * (b^3 - (-b)^3)^7 = 0 := by
  let a := 7
  let b := 2
  sorry

end simplify_expression_l87_87652


namespace half_radius_of_circle_y_l87_87480

theorem half_radius_of_circle_y (Cx Cy : ℝ) (r_x r_y : ℝ) 
  (h1 : Cx = 10 * π) 
  (h2 : Cx = 2 * π * r_x) 
  (h3 : π * r_x ^ 2 = π * r_y ^ 2) :
  (1 / 2) * r_y = 2.5 := 
by
-- sorry skips the proof
sorry

end half_radius_of_circle_y_l87_87480


namespace probability_first_three_cards_spades_l87_87045

theorem probability_first_three_cards_spades :
  let num_spades : ℕ := 13
  let total_cards : ℕ := 52
  let prob_first_spade : ℚ := num_spades / total_cards
  let prob_second_spade_given_first : ℚ := (num_spades - 1) / (total_cards - 1)
  let prob_third_spade_given_first_two : ℚ := (num_spades - 2) / (total_cards - 2)
  let prob_all_three_spades : ℚ := prob_first_spade * prob_second_spade_given_first * prob_third_spade_given_first_two
  prob_all_three_spades = 33 / 2550 :=
by
  sorry

end probability_first_three_cards_spades_l87_87045


namespace tangent_line_parallel_x_axis_coordinates_l87_87925

theorem tangent_line_parallel_x_axis_coordinates :
  (∃ P : ℝ × ℝ, P = (1, -2) ∨ P = (-1, 2)) ↔
  (∃ x y : ℝ, y = x^3 - 3 * x ∧ ∃ y', y' = 3 * x^2 - 3 ∧ y' = 0) :=
by
  sorry

end tangent_line_parallel_x_axis_coordinates_l87_87925


namespace bruces_son_age_l87_87624

variable (Bruce_age : ℕ) (son_age : ℕ)
variable (h1 : Bruce_age = 36)
variable (h2 : Bruce_age + 6 = 3 * (son_age + 6))

theorem bruces_son_age :
  son_age = 8 :=
by {
  sorry
}

end bruces_son_age_l87_87624


namespace james_remaining_balance_l87_87495

theorem james_remaining_balance 
  (initial_balance : ℕ := 500) 
  (ticket_1_2_cost : ℕ := 150)
  (ticket_3_cost : ℕ := ticket_1_2_cost / 3)
  (total_cost : ℕ := 2 * ticket_1_2_cost + ticket_3_cost)
  (roommate_share : ℕ := total_cost / 2) :
  initial_balance - roommate_share = 325 := 
by 
  -- By not considering the solution steps, we skip to the proof.
  sorry

end james_remaining_balance_l87_87495


namespace females_who_chose_malt_l87_87739

-- Definitions
def total_cheerleaders : ℕ := 26
def total_males : ℕ := 10
def total_females : ℕ := 16
def males_who_chose_malt : ℕ := 6

-- Main statement
theorem females_who_chose_malt (C M F : ℕ) (hM : M = 2 * C) (h_total : C + M = total_cheerleaders) (h_males_malt : males_who_chose_malt = total_males) : F = 10 :=
sorry

end females_who_chose_malt_l87_87739


namespace not_possible_one_lies_other_not_l87_87273

-- Variable definitions: Jean is lying (J), Pierre is lying (P)
variable (J P : Prop)

-- Conditions from the problem
def Jean_statement : Prop := P → J
def Pierre_statement : Prop := P → J

-- Theorem statement
theorem not_possible_one_lies_other_not (h1 : Jean_statement J P) (h2 : Pierre_statement J P) : ¬ ((J ∨ ¬ J) ∧ (P ∨ ¬ P) ∧ ((J ∧ ¬ P) ∨ (¬ J ∧ P))) :=
by
  sorry

end not_possible_one_lies_other_not_l87_87273


namespace sheila_weekly_earnings_is_288_l87_87557

-- Define the conditions as constants.
def sheilaWorksHoursPerDay (d : String) : ℕ :=
  if d = "Monday" ∨ d = "Wednesday" ∨ d = "Friday" then 8
  else if d = "Tuesday" ∨ d = "Thursday" then 6
  else 0

def hourlyWage : ℕ := 8

-- Calculate total weekly earnings based on conditions.
def weeklyEarnings : ℕ :=
  (sheilaWorksHoursPerDay "Monday" + sheilaWorksHoursPerDay "Wednesday" + sheilaWorksHoursPerDay "Friday") * hourlyWage +
  (sheilaWorksHoursPerDay "Tuesday" + sheilaWorksHoursPerDay "Thursday") * hourlyWage

-- The Lean statement for the proof.
theorem sheila_weekly_earnings_is_288 : weeklyEarnings = 288 :=
  by
    sorry

end sheila_weekly_earnings_is_288_l87_87557


namespace triangle_statements_l87_87980

-- Definitions of internal angles and sides of the triangle
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = Real.pi ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Statement A: If ABC is an acute triangle, then sin A > cos B
lemma statement_A (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2) :
  Real.sin A > Real.cos B := 
sorry

-- Statement B: If A > B, then sin A > sin B
lemma statement_B (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_AB : A > B) : 
  Real.sin A > Real.sin B := 
sorry

-- Statement C: If ABC is a non-right triangle, then tan A + tan B + tan C = tan A * tan B * tan C
lemma statement_C (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_non_right : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2) : 
  Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C := 
sorry

-- Statement D: If a cos A = b cos B, then triangle ABC must be isosceles
lemma statement_D (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_cos : a * Real.cos A = b * Real.cos B) : 
  ¬(A = B) ∧ ¬(B = C) := 
sorry

-- Theorem to combine all statements
theorem triangle_statements (A B C a b c : ℝ) 
  (h_triangle : Triangle A B C a b c)
  (h_acute : A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2)
  (h_AB : A > B)
  (h_non_right : A ≠ Real.pi / 2 ∧ B ≠ Real.pi / 2 ∧ C ≠ Real.pi / 2)
  (h_cos : a * Real.cos A = b * Real.cos B) : 
  Real.sin A > Real.cos B ∧ Real.sin A > Real.sin B ∧ 
  (Real.tan A + Real.tan B + Real.tan C = Real.tan A * Real.tan B * Real.tan C) ∧ 
  ¬(A = B) ∧ ¬(B = C) := 
by
  exact ⟨statement_A A B C a b c h_triangle h_acute, statement_B A B C a b c h_triangle h_AB, statement_C A B C a b c h_triangle h_non_right, statement_D A B C a b c h_triangle h_cos⟩

end triangle_statements_l87_87980


namespace radius_of_inscribed_semicircle_in_isosceles_triangle_l87_87171

theorem radius_of_inscribed_semicircle_in_isosceles_triangle
    (BC : ℝ) (h : ℝ) (r : ℝ)
    (H_eq : BC = 24)
    (H_height : h = 18)
    (H_area : 0.5 * BC * h = 0.5 * 24 * 18) :
    r = 18 / π := by
    sorry

end radius_of_inscribed_semicircle_in_isosceles_triangle_l87_87171


namespace probability_is_correct_l87_87382

variables (total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items : ℕ)

-- Setting up the problem according to the given conditions
def conditions := (total_items = 10) ∧ 
                  (truckA_first_class = 2) ∧ (truckA_second_class = 2) ∧ 
                  (truckB_first_class = 4) ∧ (truckB_second_class = 2) ∧ 
                  (brokenA = 1) ∧ (brokenB = 1) ∧
                  (remaining_items = 8)

-- Calculating the probability of selecting a first-class item from the remaining items
def probability_of_first_class : ℚ :=
  1/3 * 1/2 + 1/6 * 5/8 + 1/3 * 5/8 + 1/6 * 3/4

-- The theorem to be proved
theorem probability_is_correct : 
  conditions total_items truckA_first_class truckA_second_class truckB_first_class truckB_second_class brokenA brokenB remaining_items →
  probability_of_first_class = 29/48 :=
sorry

end probability_is_correct_l87_87382


namespace set_union_l87_87509

theorem set_union :
  let M := {x | x^2 + 2 * x - 3 = 0}
  let N := {-1, 2, 3}
  M ∪ N = {-1, 1, 2, -3, 3} :=
by
  sorry

end set_union_l87_87509


namespace probability_of_defective_product_is_0_032_l87_87234

-- Defining the events and their probabilities
def P_H1 : ℝ := 0.30
def P_H2 : ℝ := 0.25
def P_H3 : ℝ := 0.45

-- Defining the probabilities of defects given each production line
def P_A_given_H1 : ℝ := 0.03
def P_A_given_H2 : ℝ := 0.02
def P_A_given_H3 : ℝ := 0.04

-- Summing up the total probabilities
def P_A : ℝ :=
  P_H1 * P_A_given_H1 +
  P_H2 * P_A_given_H2 +
  P_H3 * P_A_given_H3

-- The statement to be proven
theorem probability_of_defective_product_is_0_032 :
  P_A = 0.032 :=
by
  -- Proof would go here
  sorry

end probability_of_defective_product_is_0_032_l87_87234


namespace eighth_L_prime_is_31_l87_87543

def setL := {n : ℕ | n > 0 ∧ n % 3 = 1}

def isLPrime (n : ℕ) : Prop :=
  n ∈ setL ∧ n ≠ 1 ∧ ∀ m ∈ setL, (m ∣ n) → (m = 1 ∨ m = n)

theorem eighth_L_prime_is_31 : 
  ∃ n ∈ setL, isLPrime n ∧ 
  (∀ k, (∃ m ∈ setL, isLPrime m ∧ m < n) → k < 8 → m ≠ n) :=
by sorry

end eighth_L_prime_is_31_l87_87543


namespace logarithmic_product_l87_87774

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem logarithmic_product (a b : ℝ) (h1 : a ≠ b) (h2 : f a = f b) : a * b = 1 := by
  sorry

end logarithmic_product_l87_87774


namespace cost_of_insulation_l87_87196

def rectangular_tank_dimension_l : ℕ := 6
def rectangular_tank_dimension_w : ℕ := 3
def rectangular_tank_dimension_h : ℕ := 2
def total_cost : ℕ := 1440

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def cost_per_square_foot (total_cost surface_area : ℕ) : ℕ := total_cost / surface_area

theorem cost_of_insulation : 
  cost_per_square_foot total_cost (surface_area rectangular_tank_dimension_l rectangular_tank_dimension_w rectangular_tank_dimension_h) = 20 :=
by
  sorry

end cost_of_insulation_l87_87196


namespace only_function_l87_87303

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, divides (f m + f n) (m + n)

theorem only_function (f : ℕ → ℕ) (h : satisfies_condition f) : f = id :=
by
  -- Proof goes here.
  sorry

end only_function_l87_87303


namespace compute_expression_l87_87158

theorem compute_expression : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end compute_expression_l87_87158


namespace travel_time_comparison_l87_87889

theorem travel_time_comparison
  (v : ℝ) -- speed during the first trip
  (t1 : ℝ) (t2 : ℝ)
  (h1 : t1 = 80 / v) -- time for the first trip
  (h2 : t2 = 100 / v) -- time for the second trip
  : t2 = 1.25 * t1 :=
by
  sorry

end travel_time_comparison_l87_87889


namespace difference_length_width_l87_87278

-- Definition of variables and conditions
variables (L W : ℝ)
def hall_width_half_length : Prop := W = (1/2) * L
def hall_area_578 : Prop := L * W = 578

-- Theorem to prove the desired result
theorem difference_length_width (h1 : hall_width_half_length L W) (h2 : hall_area_578 L W) : L - W = 17 :=
sorry

end difference_length_width_l87_87278


namespace systematic_sampling_sequence_l87_87202

theorem systematic_sampling_sequence :
  ∃ (s : Set ℕ), s = {3, 13, 23, 33, 43} ∧
  (∀ n, n ∈ s → n ≤ 50 ∧ ∃ k, k < 5 ∧ n = 3 + k * 10) :=
by
  sorry

end systematic_sampling_sequence_l87_87202


namespace absolute_value_of_neg_five_l87_87933

theorem absolute_value_of_neg_five : |(-5 : ℤ)| = 5 := 
by 
  sorry

end absolute_value_of_neg_five_l87_87933


namespace minimum_tangent_length_4_l87_87853

noncomputable def minimum_tangent_length (a b : ℝ) : ℝ :=
  Real.sqrt ((b + 4)^2 + (b - 2)^2 - 2)

theorem minimum_tangent_length_4 :
  ∀ (a b : ℝ), (x^2 + y^2 + 2 * x - 4 * y + 3 = 0) ∧ (x = a ∧ y = b) ∧ (2*a*x + b*y + 6 = 0) → 
    minimum_tangent_length a b = 4 :=
by
  sorry

end minimum_tangent_length_4_l87_87853


namespace sin_cos_cos_sin_unique_pair_exists_uniq_l87_87186

noncomputable def theta (x : ℝ) : ℝ := Real.sin (Real.cos x) - x

theorem sin_cos_cos_sin_unique_pair_exists_uniq (h : 0 < c ∧ c < (1/2) * Real.pi ∧ 0 < d ∧ d < (1/2) * Real.pi) :
  (∃! (c d : ℝ), Real.sin (Real.cos c) = c ∧ Real.cos (Real.sin d) = d ∧ c < d) :=
sorry

end sin_cos_cos_sin_unique_pair_exists_uniq_l87_87186


namespace calc_c_15_l87_87898

noncomputable def c : ℕ → ℝ
| 0 => 1 -- This case won't be used, setup for pattern match
| 1 => 3
| 2 => 5
| (n+3) => c (n+2) * c (n+1)

theorem calc_c_15 : c 15 = 3 ^ 235 :=
sorry

end calc_c_15_l87_87898


namespace sqrt_mul_sqrt_eq_six_l87_87240

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end sqrt_mul_sqrt_eq_six_l87_87240


namespace main_l87_87537

-- Definition for part (a)
def part_a : Prop :=
  ∀ (a b : ℕ), a = 300 ∧ b = 200 → 3^b > 2^a

-- Definition for part (b)
def part_b : Prop :=
  ∀ (c d : ℕ), c = 40 ∧ d = 28 → 3^d > 2^c

-- Definition for part (c)
def part_c : Prop :=
  ∀ (e f : ℕ), e = 44 ∧ f = 53 → 4^f > 5^e

-- Main conjecture proving all parts
theorem main : part_a ∧ part_b ∧ part_c :=
by
  sorry

end main_l87_87537


namespace senior_tickets_count_l87_87206

-- Define variables and problem conditions
variables (A S : ℕ)

-- Total number of tickets equation
def total_tickets (A S : ℕ) : Prop := A + S = 510

-- Total receipts equation
def total_receipts (A S : ℕ) : Prop := 21 * A + 15 * S = 8748

-- Prove that the number of senior citizen tickets S is 327
theorem senior_tickets_count (A S : ℕ) (h1 : total_tickets A S) (h2 : total_receipts A S) : S = 327 :=
sorry

end senior_tickets_count_l87_87206


namespace storks_minus_birds_l87_87877

/-- Define the initial values --/
def s : ℕ := 6         -- Number of storks
def b1 : ℕ := 2        -- Initial number of birds
def b2 : ℕ := 3        -- Number of additional birds

/-- Calculate the total number of birds --/
def b : ℕ := b1 + b2   -- Total number of birds

/-- Prove the number of storks minus the number of birds --/
theorem storks_minus_birds : s - b = 1 :=
by sorry

end storks_minus_birds_l87_87877


namespace units_digit_4659_pow_157_l87_87470

theorem units_digit_4659_pow_157 : 
  (4659^157) % 10 = 9 := 
by 
  sorry

end units_digit_4659_pow_157_l87_87470


namespace Richard_Orlando_ratio_l87_87281

def Jenny_cards : ℕ := 6
def Orlando_more_cards : ℕ := 2
def Total_cards : ℕ := 38

theorem Richard_Orlando_ratio :
  let Orlando_cards := Jenny_cards + Orlando_more_cards
  let Richard_cards := Total_cards - (Jenny_cards + Orlando_cards)
  let ratio := Richard_cards / Orlando_cards
  ratio = 3 :=
by
  sorry

end Richard_Orlando_ratio_l87_87281


namespace profit_percentage_l87_87131

theorem profit_percentage (C S : ℝ) (h : 30 * C = 24 * S) :
  (S - C) / C * 100 = 25 :=
by sorry

end profit_percentage_l87_87131


namespace polynomial_sum_coeff_l87_87038

-- Definitions for the polynomials given
def poly1 (d : ℤ) : ℤ := 15 * d^3 + 19 * d^2 + 17 * d + 18
def poly2 (d : ℤ) : ℤ := 3 * d^3 + 4 * d + 2

-- The main statement to prove
theorem polynomial_sum_coeff :
  let p := 18
  let q := 19
  let r := 21
  let s := 20
  p + q + r + s = 78 :=
by
  sorry

end polynomial_sum_coeff_l87_87038


namespace student_can_escape_l87_87778

open Real

/-- The student can escape the pool given the following conditions:
 1. R is the radius of the circular pool.
 2. The teacher runs 4 times faster than the student swims.
 3. The teacher's running speed is v_T.
 4. The student's swimming speed is v_S = v_T / 4.
 5. The student swims along a circular path of radius r, where
    (1 - π / 4) * R < r < R / 4 -/
theorem student_can_escape (R v_T v_S r : ℝ) (h1 : v_S = v_T / 4)
  (h2 : (1 - π / 4) * R < r) (h3 : r < R / 4) : 
  True :=
sorry

end student_can_escape_l87_87778


namespace jason_money_in_usd_l87_87377

noncomputable def jasonTotalInUSD : ℝ :=
  let init_quarters_value := 49 * 0.25
  let init_dimes_value    := 32 * 0.10
  let init_nickels_value  := 18 * 0.05
  let init_euros_in_usd   := 22.50 * 1.20
  let total_initial       := init_quarters_value + init_dimes_value + init_nickels_value + init_euros_in_usd

  let dad_quarters_value  := 25 * 0.25
  let dad_dimes_value     := 15 * 0.10
  let dad_nickels_value   := 10 * 0.05
  let dad_euros_in_usd    := 12 * 1.20
  let total_additional    := dad_quarters_value + dad_dimes_value + dad_nickels_value + dad_euros_in_usd

  total_initial + total_additional

theorem jason_money_in_usd :
  jasonTotalInUSD = 66 := 
sorry

end jason_money_in_usd_l87_87377


namespace sufficient_condition_for_gt_l87_87502

theorem sufficient_condition_for_gt (a : ℝ) : (∀ x : ℝ, x > a → x > 1) → (∃ x : ℝ, x > 1 ∧ x ≤ a) → a > 1 :=
by
  sorry

end sufficient_condition_for_gt_l87_87502


namespace bobby_additional_candy_l87_87811

variable (initial_candy additional_candy chocolate total_candy : ℕ)
variable (bobby_initial_candy : initial_candy = 38)
variable (bobby_ate_chocolate : chocolate = 16)
variable (bobby_more_candy : initial_candy + additional_candy = 58 + chocolate)

theorem bobby_additional_candy :
  additional_candy = 36 :=
by {
  sorry
}

end bobby_additional_candy_l87_87811


namespace root_of_quadratic_l87_87453

theorem root_of_quadratic (a : ℝ) (h : ∃ (x : ℝ), x = 0 ∧ x^2 + x + 2 * a - 1 = 0) : a = 1 / 2 := by
  sorry

end root_of_quadratic_l87_87453


namespace inequality_proof_l87_87803

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y / (y + z) + y^2 * z / (z + x) + z^2 * x / (x + y) ≥ 1 / 2 * (x^2 + y^2 + z^2)) :=
by sorry

end inequality_proof_l87_87803


namespace cube_split_l87_87042

theorem cube_split (m : ℕ) (h1 : m > 1)
  (h2 : ∃ (p : ℕ), (p = (m - 1) * (m^2 + m + 1) ∨ p = (m - 1)^2 ∨ p = (m - 1)^2 + 2) ∧ p = 2017) :
  m = 46 :=
by {
    sorry
}

end cube_split_l87_87042


namespace birdhouse_distance_l87_87633

theorem birdhouse_distance (car_distance : ℕ) (lawnchair_distance : ℕ) (birdhouse_distance : ℕ) 
  (h1 : car_distance = 200) 
  (h2 : lawnchair_distance = 2 * car_distance) 
  (h3 : birdhouse_distance = 3 * lawnchair_distance) : 
  birdhouse_distance = 1200 :=
by
  sorry

end birdhouse_distance_l87_87633


namespace set_characteristics_l87_87521

-- Define the characteristics of elements in a set
def characteristic_definiteness := true
def characteristic_distinctness := true
def characteristic_unorderedness := true
def characteristic_reality := false -- We aim to prove this

-- The problem statement in Lean
theorem set_characteristics :
  ¬ characteristic_reality :=
by
  -- Here would be the proof, but we add sorry as indicated.
  sorry

end set_characteristics_l87_87521


namespace complex_number_solution_l87_87912

def i : ℂ := Complex.I

theorem complex_number_solution (z : ℂ) (h : z * (1 - i) = 2 * i) : z = -1 + i :=
by
  sorry

end complex_number_solution_l87_87912


namespace work_hours_to_pay_off_debt_l87_87559

theorem work_hours_to_pay_off_debt (initial_debt paid_amount hourly_rate remaining_debt work_hours : ℕ) 
  (h₁ : initial_debt = 100) 
  (h₂ : paid_amount = 40) 
  (h₃ : hourly_rate = 15) 
  (h₄ : remaining_debt = initial_debt - paid_amount) 
  (h₅ : work_hours = remaining_debt / hourly_rate) : 
  work_hours = 4 :=
by
  sorry

end work_hours_to_pay_off_debt_l87_87559


namespace largest_ball_radius_l87_87793

def torus_inner_radius : ℝ := 2
def torus_outer_radius : ℝ := 4
def circle_center : ℝ × ℝ × ℝ := (3, 0, 1)
def circle_radius : ℝ := 1

theorem largest_ball_radius : ∃ r : ℝ, r = 9 / 4 ∧
  (∃ (sphere_center : ℝ × ℝ × ℝ) (torus_center : ℝ × ℝ × ℝ),
  (sphere_center = (0, 0, r)) ∧
  (torus_center = (3, 0, 1)) ∧
  (dist (0, 0, r) (3, 0, 1) = r + 1)) := sorry

end largest_ball_radius_l87_87793


namespace max_intersection_value_l87_87891

noncomputable def max_intersection_size (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) : ℕ :=
  if ((A.card = 2019) ∧ (B.card = 2019) ∧ (A ∩ B ∩ C).card = 2018)
  then (A ∩ B ∩ C).card 
  else 0

theorem max_intersection_value (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) :
  max_intersection_size A B C h1 h2 = 2018 :=
sorry

end max_intersection_value_l87_87891


namespace integral_root_of_equation_l87_87427

theorem integral_root_of_equation : 
  ∀ x : ℤ, (x - 8 / (x - 4)) = 2 - 8 / (x - 4) ↔ x = 2 := 
sorry

end integral_root_of_equation_l87_87427


namespace gcd_lcm_888_1147_l87_87425

theorem gcd_lcm_888_1147 :
  Nat.gcd 888 1147 = 37 ∧ Nat.lcm 888 1147 = 27528 := by
  sorry

end gcd_lcm_888_1147_l87_87425


namespace disinfectant_usage_l87_87197

theorem disinfectant_usage (x : ℝ) (hx1 : 0 < x) (hx2 : 120 / x / 2 = 120 / (x + 4)) : x = 4 :=
by
  sorry

end disinfectant_usage_l87_87197


namespace problem1_problem2_l87_87759

-- Problem 1
theorem problem1 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = -5 / 3 :=
by sorry

-- Problem 2
theorem problem2 (α : ℝ) (h : (Real.tan α) / (Real.tan α - 1) = -1) (h_quad : π < α ∧ α < 3 * π / 2) :
  Real.cos (-π + α) + Real.cos (π / 2 + α) = 3 * Real.sqrt 5 / 5 :=
by sorry

end problem1_problem2_l87_87759


namespace LCM_of_two_numbers_l87_87705

theorem LCM_of_two_numbers (a b : ℕ) (h_hcf : Nat.gcd a b = 11) (h_product : a * b = 1991) : Nat.lcm a b = 181 :=
by
  sorry

end LCM_of_two_numbers_l87_87705


namespace y_values_relation_l87_87138

theorem y_values_relation :
  ∀ y1 y2 y3 : ℝ,
    (y1 = (-3 + 1) ^ 2 + 1) →
    (y2 = (0 + 1) ^ 2 + 1) →
    (y3 = (2 + 1) ^ 2 + 1) →
    y2 < y1 ∧ y1 < y3 :=
by
  sorry

end y_values_relation_l87_87138


namespace factorize_expression_l87_87667

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l87_87667


namespace batsman_avg_after_17th_inning_l87_87455

def batsman_average : Prop :=
  ∃ (A : ℕ), 
    (A + 3 = (16 * A + 92) / 17) → 
    (A + 3 = 44)

theorem batsman_avg_after_17th_inning : batsman_average :=
by
  sorry

end batsman_avg_after_17th_inning_l87_87455


namespace two_colonies_reach_limit_in_same_time_l87_87424

theorem two_colonies_reach_limit_in_same_time (d : ℕ) (h : 16 = d): 
  d = 16 :=
by
  /- Asserting that if one colony takes 16 days, two starting together will also take 16 days -/
  sorry

end two_colonies_reach_limit_in_same_time_l87_87424


namespace last_two_digits_7_pow_2017_l87_87800

noncomputable def last_two_digits_of_pow :=
  ∀ n : ℕ, ∃ (d : ℕ), d < 100 ∧ 7^n % 100 = d

theorem last_two_digits_7_pow_2017 : ∃ (d : ℕ), d = 7 ∧ 7^2017 % 100 = d :=
by
  sorry

end last_two_digits_7_pow_2017_l87_87800


namespace angles_equal_or_cofunctions_equal_l87_87726

def cofunction (θ : ℝ) : ℝ := sorry -- Define the co-function (e.g., sine and cosine)

theorem angles_equal_or_cofunctions_equal (θ₁ θ₂ : ℝ) :
  θ₁ = θ₂ ∨ cofunction θ₁ = cofunction θ₂ → θ₁ = θ₂ :=
sorry

end angles_equal_or_cofunctions_equal_l87_87726


namespace eggs_per_hen_per_day_l87_87586

theorem eggs_per_hen_per_day
  (hens : ℕ) (days : ℕ) (neighborTaken : ℕ) (dropped : ℕ) (finalEggs : ℕ) (E : ℕ) 
  (h1 : hens = 3) 
  (h2 : days = 7) 
  (h3 : neighborTaken = 12) 
  (h4 : dropped = 5) 
  (h5 : finalEggs = 46) 
  (totalEggs : ℕ := hens * E * days) 
  (afterNeighbor : ℕ := totalEggs - neighborTaken) 
  (beforeDropping : ℕ := finalEggs + dropped) : 
  totalEggs = beforeDropping + neighborTaken → E = 3 := sorry

end eggs_per_hen_per_day_l87_87586


namespace polynomial_identity_l87_87822

theorem polynomial_identity (P : ℝ → ℝ) :
  (∀ x, (x - 1) * P (x + 1) - (x + 2) * P x = 0) ↔ ∃ a : ℝ, ∀ x, P x = a * (x^3 - x) :=
by
  sorry

end polynomial_identity_l87_87822


namespace domain_of_transformed_function_l87_87006

theorem domain_of_transformed_function (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → True) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → True :=
sorry

end domain_of_transformed_function_l87_87006


namespace radius_of_circle_area_of_sector_l87_87520

theorem radius_of_circle (L : ℝ) (θ : ℝ) (hL : L = 50) (hθ : θ = 200) : 
  ∃ r : ℝ, r = 45 / Real.pi := 
by
  sorry

theorem area_of_sector (L : ℝ) (r : ℝ) (hL : L = 50) (hr : r = 45 / Real.pi) : 
  ∃ S : ℝ, S = 1125 / Real.pi := 
by
  sorry

end radius_of_circle_area_of_sector_l87_87520


namespace new_average_doubled_l87_87309

theorem new_average_doubled
  (average : ℕ)
  (num_students : ℕ)
  (h_avg : average = 45)
  (h_num_students : num_students = 30)
  : (2 * average * num_students / num_students) = 90 := by
  sorry

end new_average_doubled_l87_87309


namespace zander_stickers_l87_87504

theorem zander_stickers (total_stickers andrew_ratio bill_ratio : ℕ) (initial_stickers: total_stickers = 100) (andrew_fraction : andrew_ratio = 1 / 5) (bill_fraction : bill_ratio = 3 / 10) :
  let andrew_give_away := total_stickers * andrew_ratio
  let remaining_stickers := total_stickers - andrew_give_away
  let bill_give_away := remaining_stickers * bill_ratio
  let total_given_away := andrew_give_away + bill_give_away
  total_given_away = 44 :=
by
  sorry

end zander_stickers_l87_87504


namespace increasing_function_geq_25_l87_87044

theorem increasing_function_geq_25 {m : ℝ} 
  (h : ∀ x y : ℝ, x ≥ -2 ∧ x ≤ y → (4 * x^2 - m * x + 5) ≤ (4 * y^2 - m * y + 5)) :
  (4 * 1^2 - m * 1 + 5) ≥ 25 :=
by {
  -- Proof is omitted
  sorry
}

end increasing_function_geq_25_l87_87044


namespace number_of_sick_animals_l87_87794

def total_animals := 26 + 40 + 34  -- Total number of animals at Stacy's farm
def sick_fraction := 1 / 2  -- Half of all animals get sick

-- Defining sick animals for each type
def sick_chickens := 26 * sick_fraction
def sick_piglets := 40 * sick_fraction
def sick_goats := 34 * sick_fraction

-- The main theorem to prove
theorem number_of_sick_animals :
  sick_chickens + sick_piglets + sick_goats = 50 :=
by
  -- Skeleton of the proof that is to be completed later
  sorry

end number_of_sick_animals_l87_87794


namespace parallel_lines_b_value_l87_87669

-- Define the first line equation in slope-intercept form.
def line1_slope (b : ℝ) : ℝ :=
  3

-- Define the second line equation in slope-intercept form.
def line2_slope (b : ℝ) : ℝ :=
  b + 10

-- Theorem stating that if the lines are parallel, the value of b is -7.
theorem parallel_lines_b_value :
  ∀ b : ℝ, line1_slope b = line2_slope b → b = -7 :=
by
  intro b
  intro h
  sorry

end parallel_lines_b_value_l87_87669


namespace quadratic_inequality_false_range_l87_87608

theorem quadratic_inequality_false_range (a : ℝ) :
  (¬ ∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (a < 0 ∨ a ≥ 3) :=
by
  sorry

end quadratic_inequality_false_range_l87_87608


namespace incorrect_statement_l87_87583

theorem incorrect_statement :
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  (statementA = "correct") ∧ 
  (statementB = "correct") ∧ 
  (statementC = "correct") ∧ 
  (statementD = "incorrect") :=
by
  let statementA := "The shortest distance between two points is a line segment."
  let statementB := "Vertical angles are congruent."
  let statementC := "Complementary angles of the same measure are congruent."
  let statementD := "There is only one line passing through a point outside a given line that is parallel to the given line."
  have hA : statementA = "correct" := sorry
  have hB : statementB = "correct" := sorry
  have hC : statementC = "correct" := sorry
  have hD : statementD = "incorrect" := sorry
  exact ⟨hA, hB, hC, hD⟩

end incorrect_statement_l87_87583


namespace arithmetic_sequence_sum_cubes_l87_87194

theorem arithmetic_sequence_sum_cubes (x : ℤ) (k : ℕ) (h : ∀ i, 0 <= i ∧ i <= k → (x + 2 * i : ℤ)^3 =
  -1331) (hk : k > 3) : k = 6 :=
sorry

end arithmetic_sequence_sum_cubes_l87_87194


namespace simplify_expression_l87_87675

theorem simplify_expression : 2023^2 - 2022 * 2024 = 1 := by
  sorry

end simplify_expression_l87_87675


namespace f_evaluation_l87_87351

def f (a b c : ℚ) : ℚ := a^2 + 2 * b * c

theorem f_evaluation :
  f 1 23 76 + f 23 76 1 + f 76 1 23 = 10000 := by
  sorry

end f_evaluation_l87_87351


namespace tory_earns_more_than_bert_l87_87694

-- Define the initial prices of the toys
def initial_price_phones : ℝ := 18
def initial_price_guns : ℝ := 20

-- Define the quantities sold by Bert and Tory
def quantity_phones : ℕ := 10
def quantity_guns : ℕ := 15

-- Define the discounts
def discount_phones : ℝ := 0.15
def discounted_phones_quantity : ℕ := 3

def discount_guns : ℝ := 0.10
def discounted_guns_quantity : ℕ := 7

-- Define the tax
def tax_rate : ℝ := 0.05

noncomputable def bert_initial_earnings : ℝ := initial_price_phones * quantity_phones

noncomputable def tory_initial_earnings : ℝ := initial_price_guns * quantity_guns

noncomputable def bert_discount : ℝ := discount_phones * initial_price_phones * discounted_phones_quantity

noncomputable def tory_discount : ℝ := discount_guns * initial_price_guns * discounted_guns_quantity

noncomputable def bert_earnings_after_discount : ℝ := bert_initial_earnings - bert_discount

noncomputable def tory_earnings_after_discount : ℝ := tory_initial_earnings - tory_discount

noncomputable def bert_tax : ℝ := tax_rate * bert_earnings_after_discount

noncomputable def tory_tax : ℝ := tax_rate * tory_earnings_after_discount

noncomputable def bert_final_earnings : ℝ := bert_earnings_after_discount + bert_tax

noncomputable def tory_final_earnings : ℝ := tory_earnings_after_discount + tory_tax

noncomputable def earning_difference : ℝ := tory_final_earnings - bert_final_earnings

theorem tory_earns_more_than_bert : earning_difference = 119.805 := by
  sorry

end tory_earns_more_than_bert_l87_87694


namespace increase_by_fraction_l87_87801

theorem increase_by_fraction (original_value : ℕ) (fraction : ℚ) : original_value = 120 → fraction = 5/6 → original_value + original_value * fraction = 220 :=
by
  intros h1 h2
  sorry

end increase_by_fraction_l87_87801


namespace complex_division_l87_87942

-- Conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Question: Prove the complex division
theorem complex_division (h : i = Complex.I) : (8 - i) / (2 + i) = 3 - 2 * i :=
by sorry

end complex_division_l87_87942


namespace minimum_n_for_3_zeros_l87_87412

theorem minimum_n_for_3_zeros :
  ∃ n : ℕ, (∀ m : ℕ, (m < n → ∀ k < 10, m + k ≠ 5 * m ∧ m + k ≠ 5 * m + 25)) ∧
  (∀ k < 10, n + k = 16 ∨ n + k = 16 + 9) ∧
  n = 16 :=
sorry

end minimum_n_for_3_zeros_l87_87412


namespace jasmine_cookies_l87_87203

theorem jasmine_cookies (J : ℕ) (h1 : 20 + J + (J + 10) = 60) : J = 15 :=
sorry

end jasmine_cookies_l87_87203


namespace find_PF2_l87_87149

-- Statement of the problem

def hyperbola_1 (x y: ℝ) := (x^2 / 16) - (y^2 / 20) = 1

theorem find_PF2 (x y PF1 PF2: ℝ) (a : ℝ)
    (h_hyperbola : hyperbola_1 x y)
    (h_a : a = 4) 
    (h_dist_PF1 : PF1 = 9) :
    abs (PF1 - PF2) = 2 * a → PF2 = 17 :=
by
  intro h1
  sorry

end find_PF2_l87_87149


namespace rectangular_field_perimeter_l87_87350

-- Definitions for conditions
def width : ℕ := 75
def length : ℕ := (7 * width) / 5
def perimeter (L W : ℕ) : ℕ := 2 * (L + W)

-- Statement to prove
theorem rectangular_field_perimeter : perimeter length width = 360 := by
  sorry

end rectangular_field_perimeter_l87_87350


namespace find_a5_l87_87989

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = a 1 + (n - 1) * d

theorem find_a5 (a : ℕ → ℤ) (d : ℤ)
  (h_seq : arithmetic_sequence a d)
  (h1 : a 1 + a 5 = 8)
  (h4 : a 4 = 7) : 
  a 5 = 10 := sorry

end find_a5_l87_87989


namespace total_points_first_four_games_l87_87019

-- Define the scores for the first three games
def score1 : ℕ := 10
def score2 : ℕ := 14
def score3 : ℕ := 6

-- Define the score for the fourth game as the average of the first three games
def score4 : ℕ := (score1 + score2 + score3) / 3

-- Define the total points scored in the first four games
def total_points : ℕ := score1 + score2 + score3 + score4

-- State the theorem to prove
theorem total_points_first_four_games : total_points = 40 :=
  sorry

end total_points_first_four_games_l87_87019


namespace convert_base_five_to_ten_l87_87697

theorem convert_base_five_to_ten : ∃ n : ℕ, n = 38 ∧ (1 * 5^2 + 2 * 5^1 + 3 * 5^0 = n) :=
by
  sorry

end convert_base_five_to_ten_l87_87697


namespace cos_sq_minus_exp_equals_neg_one_fourth_l87_87430

theorem cos_sq_minus_exp_equals_neg_one_fourth :
  (Real.cos (30 * Real.pi / 180))^2 - (2 - Real.pi)^0 = -1 / 4 := by
sorry

end cos_sq_minus_exp_equals_neg_one_fourth_l87_87430


namespace Haley_boxes_needed_l87_87113

theorem Haley_boxes_needed (TotalMagazines : ℕ) (MagazinesPerBox : ℕ) 
  (h1 : TotalMagazines = 63) (h2 : MagazinesPerBox = 9) : 
  TotalMagazines / MagazinesPerBox = 7 := by
sorry

end Haley_boxes_needed_l87_87113


namespace alyssa_games_this_year_l87_87402

theorem alyssa_games_this_year : 
    ∀ (X: ℕ), 
    (13 + X + 15 = 39) → 
    X = 11 := 
by
  intros X h
  have h₁ : 13 + 15 = 28 := by norm_num
  have h₂ : X + 28 = 39 := by linarith
  have h₃ : X = 11 := by linarith
  exact h₃

end alyssa_games_this_year_l87_87402


namespace fraction_grades_C_l87_87322

def fraction_grades_A (students : ℕ) : ℕ := (1 / 5) * students
def fraction_grades_B (students : ℕ) : ℕ := (1 / 4) * students
def num_grades_D : ℕ := 5
def total_students : ℕ := 100

theorem fraction_grades_C :
  (total_students - (fraction_grades_A total_students + fraction_grades_B total_students + num_grades_D)) / total_students = 1 / 2 :=
by
  sorry

end fraction_grades_C_l87_87322


namespace find_m_l87_87560

theorem find_m (x y m : ℝ) (h1 : 2 * x + y = 1) (h2 : x + 2 * y = 2) (h3 : x + y = 2 * m - 1) : m = 1 :=
by
  sorry

end find_m_l87_87560


namespace range_of_m_l87_87046

theorem range_of_m (m : ℝ) 
  (hp : ∀ x : ℝ, 2 * x > m * (x ^ 2 + 1)) 
  (hq : ∃ x0 : ℝ, x0 ^ 2 + 2 * x0 - m - 1 = 0) : 
  -2 ≤ m ∧ m < -1 :=
sorry

end range_of_m_l87_87046


namespace even_digits_count_1998_l87_87567

-- Define the function for counting the total number of digits used in the first n positive even integers
def totalDigitsEvenIntegers (n : ℕ) : ℕ :=
  let totalSingleDigit := 4 -- 2, 4, 6, 8
  let numDoubleDigit := 45 -- 10 to 98
  let digitsDoubleDigit := numDoubleDigit * 2
  let numTripleDigit := 450 -- 100 to 998
  let digitsTripleDigit := numTripleDigit * 3
  let numFourDigit := 1499 -- 1000 to 3996
  let digitsFourDigit := numFourDigit * 4
  totalSingleDigit + digitsDoubleDigit + digitsTripleDigit + digitsFourDigit

-- Theorem: The total number of digits used when the first 1998 positive even integers are written is 7440.
theorem even_digits_count_1998 : totalDigitsEvenIntegers 1998 = 7440 :=
  sorry

end even_digits_count_1998_l87_87567


namespace position_of_99_l87_87618

-- Define a function that describes the position of an odd number in the 5-column table.
def position_in_columns (n : ℕ) : ℕ := sorry  -- position in columns is defined by some rule

-- Now, state the theorem regarding the position of 99.
theorem position_of_99 : position_in_columns 99 = 3 := 
by 
  sorry  -- Proof goes here

end position_of_99_l87_87618


namespace savings_correct_l87_87199

noncomputable def savings (income expenditure : ℕ) : ℕ :=
income - expenditure

theorem savings_correct (I E : ℕ) (h_ratio :  I / E = 10 / 4) (h_income : I = 19000) :
  savings I E = 11400 :=
sorry

end savings_correct_l87_87199


namespace omega_min_value_l87_87612

def min_omega (ω : ℝ) : Prop :=
  ω > 0 ∧ ∃ k : ℤ, (k ≠ 0 ∧ ω = 8)

theorem omega_min_value (ω : ℝ) (h1 : ω > 0) (h2 : ∃ k : ℤ, k ≠ 0 ∧ (k * 2 * π) / ω = π / 4) : 
  ω = 8 :=
by
  sorry

end omega_min_value_l87_87612


namespace find_x_equals_4_l87_87640

noncomputable def repeatingExpr (x : ℝ) : ℝ :=
2 + 4 / (1 + 4 / (2 + 4 / (1 + 4 / x)))

theorem find_x_equals_4 :
  ∃ x : ℝ, x = repeatingExpr x ∧ x = 4 :=
by
  use 4
  sorry

end find_x_equals_4_l87_87640


namespace multiply_polynomials_l87_87379

variables {R : Type*} [CommRing R] -- Define R as a commutative ring
variable (x : R) -- Define variable x in R

theorem multiply_polynomials : (2 * x) * (5 * x^2) = 10 * x^3 := 
sorry -- Placeholder for the proof

end multiply_polynomials_l87_87379


namespace max_possible_player_salary_l87_87996

theorem max_possible_player_salary (n : ℕ) (min_salary total_salary : ℕ) (num_players : ℕ) 
  (h1 : num_players = 24) 
  (h2 : min_salary = 20000) 
  (h3 : total_salary = 960000)
  (h4 : n = 23 * min_salary + 500000) 
  (h5 : 23 * min_salary + 500000 ≤ total_salary) 
  : n = total_salary :=
by {
  -- The proof will replace this sorry.
  sorry
}

end max_possible_player_salary_l87_87996


namespace probability_in_dark_l87_87179

theorem probability_in_dark (rev_per_min : ℕ) (given_prob : ℝ) (h1 : rev_per_min = 3) (h2 : given_prob = 0.25) :
  given_prob = 0.25 :=
by
  sorry

end probability_in_dark_l87_87179


namespace simplify_fraction_l87_87938

theorem simplify_fraction :
  (5 : ℚ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 27) = Real.sqrt 3 / 12 := by
sorry

end simplify_fraction_l87_87938


namespace ratio_of_cars_to_trucks_l87_87887

-- Definitions based on conditions
def total_vehicles : ℕ := 60
def trucks : ℕ := 20
def cars : ℕ := total_vehicles - trucks

-- Theorem to prove
theorem ratio_of_cars_to_trucks : (cars / trucks : ℚ) = 2 := by
  sorry

end ratio_of_cars_to_trucks_l87_87887


namespace calculate_R_cubed_plus_R_squared_plus_R_l87_87808

theorem calculate_R_cubed_plus_R_squared_plus_R (R : ℕ) (hR : R > 0)
  (h1 : ∃ q : ℚ, q = (R / (2 * R + 2)) * ((R - 1) / (2 * R + 1)))
  (h2 : (R / (2 * R + 2)) * ((R + 2) / (2 * R + 1)) + ((R + 2) / (2 * R + 2)) * (R / (2 * R + 1)) = 3 * q) :
  R^3 + R^2 + R = 399 :=
by
  sorry

end calculate_R_cubed_plus_R_squared_plus_R_l87_87808


namespace quadratic_eq_unique_k_l87_87861

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end quadratic_eq_unique_k_l87_87861


namespace december_28_is_saturday_l87_87930

def days_per_week := 7

def thanksgiving_day : Nat := 28

def november_length : Nat := 30

def december_28_day_of_week : Nat :=
  (thanksgiving_day % days_per_week + november_length + 28 - thanksgiving_day) % days_per_week

theorem december_28_is_saturday :
  (december_28_day_of_week = 6) :=
by
  sorry

end december_28_is_saturday_l87_87930


namespace option_D_correct_l87_87398

-- Definitions representing conditions
variables (a b : Line) (α : Plane)

-- Conditions
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_in_plane (b : Line) (α : Plane) : Prop := sorry

-- Theorem stating the correctness of option D
theorem option_D_correct (h1 : line_parallel_plane a α)
                         (h2 : line_parallel_line a b) :
                         (line_in_plane b α) ∨ (line_parallel_plane b α) :=
by
  sorry

end option_D_correct_l87_87398


namespace parabola_line_intersect_l87_87111

theorem parabola_line_intersect (a : ℝ) (b : ℝ) (h1 : a ≠ 0) (h2 : ∀ x : ℝ, (y = a * x^2) ↔ (y = 2 * x - 3) → (x, y) = (1, -1)) :
  a = -1 ∧ b = -1 ∧ ((x, y) = (-3, -9) ∨ (x, y) = (1, -1)) := by
  sorry

end parabola_line_intersect_l87_87111


namespace rectangle_width_l87_87473

theorem rectangle_width (r l w : ℝ) (h_r : r = Real.sqrt 12) (h_l : l = 3 * Real.sqrt 2)
  (h_area_eq: Real.pi * r^2 = l * w) : w = 2 * Real.sqrt 2 * Real.pi :=
by
  sorry

end rectangle_width_l87_87473


namespace find_M_N_sum_l87_87816

theorem find_M_N_sum
  (M N : ℕ)
  (h1 : 3 * 75 = 5 * M)
  (h2 : 3 * N = 5 * 90) :
  M + N = 195 := 
sorry

end find_M_N_sum_l87_87816


namespace find_x_l87_87033

theorem find_x (a b x : ℝ) (h_a : a > 0) (h_b : b > 0) (h_x : x > 0)
  (s : ℝ) (h_s1 : s = (a ^ 2) ^ (4 * b)) (h_s2 : s = a ^ (2 * b) * x ^ (3 * b)) :
  x = a ^ 2 :=
sorry

end find_x_l87_87033


namespace sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l87_87383

theorem sixty_percent_of_fifty_minus_thirty_percent_of_thirty : 
  (60 / 100 : ℝ) * 50 - (30 / 100 : ℝ) * 30 = 21 :=
by
  sorry

end sixty_percent_of_fifty_minus_thirty_percent_of_thirty_l87_87383


namespace different_kinds_of_hamburgers_l87_87662

theorem different_kinds_of_hamburgers 
  (n_condiments : ℕ) 
  (condiment_choices : ℕ)
  (meat_patty_choices : ℕ)
  (h1 : n_condiments = 8)
  (h2 : condiment_choices = 2 ^ n_condiments)
  (h3 : meat_patty_choices = 3)
  : condiment_choices * meat_patty_choices = 768 := 
by
  sorry

end different_kinds_of_hamburgers_l87_87662


namespace radian_to_degree_conversion_l87_87926

theorem radian_to_degree_conversion
: (π : ℝ) = 180 → ((-23 / 12) * π) = -345 :=
by
  sorry

end radian_to_degree_conversion_l87_87926


namespace min_workers_for_profit_l87_87513

theorem min_workers_for_profit
    (maintenance_fees : ℝ)
    (worker_hourly_wage : ℝ)
    (widgets_per_hour : ℝ)
    (widget_price : ℝ)
    (work_hours : ℝ)
    (n : ℕ)
    (h_maintenance : maintenance_fees = 470)
    (h_wage : worker_hourly_wage = 10)
    (h_production : widgets_per_hour = 6)
    (h_price : widget_price = 3.5)
    (h_hours : work_hours = 8) :
  470 + 80 * n < 168 * n → n ≥ 6 := 
by
  sorry

end min_workers_for_profit_l87_87513


namespace degrees_to_radians_l87_87899

theorem degrees_to_radians (deg: ℝ) (h : deg = 120) : deg * (π / 180) = 2 * π / 3 :=
by
  simp [h]
  sorry

end degrees_to_radians_l87_87899


namespace proof_base_5_conversion_and_addition_l87_87797

-- Define the given numbers in decimal (base 10)
def n₁ := 45
def n₂ := 25

-- Base 5 conversion function and proofs of correctness
def to_base_5 (n : ℕ) : ℕ := sorry
def from_base_5 (n : ℕ) : ℕ := sorry

-- Converted values to base 5
def a₅ : ℕ := to_base_5 n₁
def b₅ : ℕ := to_base_5 n₂

-- Sum in base 5
def c₅ : ℕ := a₅ + b₅  -- addition in base 5

-- Convert the final sum back to decimal base 10
def d₁₀ : ℕ := from_base_5 c₅

theorem proof_base_5_conversion_and_addition :
  d₁₀ = 65 ∧ to_base_5 65 = 230 :=
by sorry

end proof_base_5_conversion_and_addition_l87_87797


namespace loss_of_30_yuan_is_minus_30_yuan_l87_87441

def profit (p : ℤ) : Prop := p = 20
def loss (l : ℤ) : Prop := l = -30

theorem loss_of_30_yuan_is_minus_30_yuan (p : ℤ) (l : ℤ) (h : profit p) : loss l :=
by
  sorry

end loss_of_30_yuan_is_minus_30_yuan_l87_87441


namespace quadratic_solution_set_R_l87_87819

theorem quadratic_solution_set_R (a b c : ℝ) (h1 : a ≠ 0) (h2 : a < 0) (h3 : b^2 - 4 * a * c < 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c < 0 :=
by sorry

end quadratic_solution_set_R_l87_87819


namespace hashtag_3_8_l87_87433

-- Define the hashtag operation
def hashtag (a b : ℤ) : ℤ := a * b - b + b ^ 2

-- Prove that 3 # 8 equals 80
theorem hashtag_3_8 : hashtag 3 8 = 80 := by
  sorry

end hashtag_3_8_l87_87433


namespace solve_y_pos_in_arithmetic_seq_l87_87267

-- Define the first term as 4
def first_term : ℕ := 4

-- Define the third term as 36
def third_term : ℕ := 36

-- Basing on the properties of an arithmetic sequence, 
-- we solve for the positive second term (y) such that its square equals to 20
theorem solve_y_pos_in_arithmetic_seq : ∃ y : ℝ, y > 0 ∧ y ^ 2 = 20 := by
  sorry

end solve_y_pos_in_arithmetic_seq_l87_87267


namespace scientific_notation_correct_l87_87508

-- Define the input number
def input_number : ℕ := 858000000

-- Define the expected scientific notation result
def scientific_notation (n : ℕ) : ℝ := 8.58 * 10^8

-- The theorem states that the input number in scientific notation is indeed 8.58 * 10^8
theorem scientific_notation_correct :
  scientific_notation input_number = 8.58 * 10^8 :=
sorry

end scientific_notation_correct_l87_87508


namespace counterexample_exists_l87_87428

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- State the theorem equivalently in Lean
theorem counterexample_exists : (sum_of_digits 33 % 6 = 0) ∧ (33 % 6 ≠ 0) := by
  sorry

end counterexample_exists_l87_87428


namespace negation_of_universal_proposition_l87_87296

theorem negation_of_universal_proposition {f : ℝ → ℝ} :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by
  sorry

end negation_of_universal_proposition_l87_87296


namespace fourth_square_area_l87_87147

theorem fourth_square_area (PQ QR RS QS : ℝ)
  (h1 : PQ^2 = 25)
  (h2 : QR^2 = 49)
  (h3 : RS^2 = 64) :
  QS^2 = 138 :=
by
  sorry

end fourth_square_area_l87_87147


namespace area_of_triangle_DEF_l87_87059

theorem area_of_triangle_DEF :
  let D := (0, 2)
  let E := (6, 0)
  let F := (3, 8)
  let base1 := 6
  let height1 := 2
  let base2 := 3
  let height2 := 8
  let base3 := 3
  let height3 := 6
  let area_triangle_DE := 1 / 2 * (base1 * height1)
  let area_triangle_EF := 1 / 2 * (base2 * height2)
  let area_triangle_FD := 1 / 2 * (base3 * height3)
  let area_rectangle := 6 * 8
  ∃ area_def_triangle, 
  area_def_triangle = area_rectangle - (area_triangle_DE + area_triangle_EF + area_triangle_FD) 
  ∧ area_def_triangle = 21 :=
by 
  sorry

end area_of_triangle_DEF_l87_87059


namespace mike_books_l87_87913

theorem mike_books : 51 - 45 = 6 := 
by 
  rfl

end mike_books_l87_87913


namespace maximize_quadratic_expression_l87_87527

theorem maximize_quadratic_expression :
  ∃ x : ℝ, (∀ y : ℝ, -2 * y^2 - 8 * y + 10 ≤ -2 * x^2 - 8 * x + 10) ∧ x = -2 :=
by
  sorry

end maximize_quadratic_expression_l87_87527


namespace motorcycles_count_l87_87476

/-- In a parking lot, there are cars and motorcycles. 
    Each car has 5 wheels (including one spare) and each motorcycle has 2 wheels. 
    There are 19 cars in the parking lot. 
    Altogether all vehicles have 117 wheels. 
    Prove that there are 11 motorcycles in the parking lot. -/
theorem motorcycles_count 
  (C M : ℕ)
  (hc : C = 19)
  (total_wheels : ℕ)
  (total_wheels_eq : total_wheels = 117)
  (car_wheels : ℕ)
  (car_wheels_eq : car_wheels = 5 * C)
  (bike_wheels : ℕ)
  (bike_wheels_eq : bike_wheels = total_wheels - car_wheels)
  (wheels_per_bike : ℕ)
  (wheels_per_bike_eq : wheels_per_bike = 2):
  M = bike_wheels / wheels_per_bike :=
by
  sorry

end motorcycles_count_l87_87476


namespace sum_of_faces_l87_87657

theorem sum_of_faces (n_side_faces_per_prism : ℕ) (n_non_side_faces_per_prism : ℕ)
  (num_prisms : ℕ) (h1 : n_side_faces_per_prism = 3) (h2 : n_non_side_faces_per_prism = 2) 
  (h3 : num_prisms = 3) : 
  n_side_faces_per_prism * num_prisms + n_non_side_faces_per_prism * num_prisms = 15 :=
by
  sorry

end sum_of_faces_l87_87657


namespace maria_remaining_towels_l87_87241

-- Define the number of green towels Maria bought
def greenTowels : ℕ := 58

-- Define the number of white towels Maria bought
def whiteTowels : ℕ := 43

-- Define the total number of towels Maria bought
def totalTowels : ℕ := greenTowels + whiteTowels

-- Define the number of towels Maria gave to her mother
def towelsGiven : ℕ := 87

-- Define the resulting number of towels Maria has
def remainingTowels : ℕ := totalTowels - towelsGiven

-- Prove that the remaining number of towels is 14
theorem maria_remaining_towels : remainingTowels = 14 :=
by
  sorry

end maria_remaining_towels_l87_87241


namespace tshirt_cost_correct_l87_87465

   -- Definitions of the conditions
   def initial_amount : ℕ := 91
   def cost_of_sweater : ℕ := 24
   def cost_of_shoes : ℕ := 11
   def remaining_amount : ℕ := 50

   -- Define the total cost of the T-shirt purchase
   noncomputable def cost_of_tshirt := 
     initial_amount - remaining_amount - cost_of_sweater - cost_of_shoes

   -- Proof statement that cost_of_tshirt = 6
   theorem tshirt_cost_correct : cost_of_tshirt = 6 := 
   by
     sorry
   
end tshirt_cost_correct_l87_87465


namespace distance_AB_bounds_l87_87581

noncomputable def distance_AC : ℕ := 10
noncomputable def distance_AD : ℕ := 10
noncomputable def distance_BE : ℕ := 10
noncomputable def distance_BF : ℕ := 10
noncomputable def distance_AE : ℕ := 12
noncomputable def distance_AF : ℕ := 12
noncomputable def distance_BC : ℕ := 12
noncomputable def distance_BD : ℕ := 12
noncomputable def distance_CD : ℕ := 11
noncomputable def distance_EF : ℕ := 11
noncomputable def distance_CE : ℕ := 5
noncomputable def distance_DF : ℕ := 5

theorem distance_AB_bounds (AB : ℝ) :
  8.8 < AB ∧ AB < 19.2 :=
sorry

end distance_AB_bounds_l87_87581


namespace find_first_group_men_l87_87511

variable (M : ℕ)

def first_group_men := M
def days_for_first_group := 20
def men_in_second_group := 12
def days_for_second_group := 30

theorem find_first_group_men (h1 : first_group_men * days_for_first_group = men_in_second_group * days_for_second_group) :
  first_group_men = 18 :=
by {
  sorry
}

end find_first_group_men_l87_87511


namespace total_chapters_read_l87_87009

def books_read : ℕ := 12
def chapters_per_book : ℕ := 32

theorem total_chapters_read : books_read * chapters_per_book = 384 :=
by
  sorry

end total_chapters_read_l87_87009


namespace rank_matrix_sum_l87_87658

theorem rank_matrix_sum (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (h : ∀ i j, A i j = ↑i + ↑j) : Matrix.rank A = 2 := by
  sorry

end rank_matrix_sum_l87_87658


namespace joan_spent_on_jacket_l87_87170

def total_spent : ℝ := 42.33
def shorts_spent : ℝ := 15.00
def shirt_spent : ℝ := 12.51
def jacket_spent : ℝ := 14.82

theorem joan_spent_on_jacket :
  total_spent - shorts_spent - shirt_spent = jacket_spent :=
by
  sorry

end joan_spent_on_jacket_l87_87170


namespace thirteen_consecutive_nat_power_l87_87792

def consecutive_sum_power (N : ℕ) : ℕ :=
  (N - 6) + (N - 5) + (N - 4) + (N - 3) + (N - 2) + (N - 1) +
  N + (N + 1) + (N + 2) + (N + 3) + (N + 4) + (N + 5) + (N + 6)

theorem thirteen_consecutive_nat_power (N : ℕ) (n : ℕ) :
  N = 13^2020 →
  n = 2021 →
  consecutive_sum_power N = 13^n := by
  sorry

end thirteen_consecutive_nat_power_l87_87792


namespace bacteria_colony_growth_l87_87834

theorem bacteria_colony_growth : 
  ∃ (n : ℕ), n = 4 ∧ 5 * 3 ^ n > 200 ∧ (∀ (m : ℕ), 5 * 3 ^ m > 200 → m ≥ n) :=
by
  sorry

end bacteria_colony_growth_l87_87834


namespace trigonometric_expression_eval_l87_87294

theorem trigonometric_expression_eval :
  2 * (Real.cos (5 * Real.pi / 16))^6 +
  2 * (Real.sin (11 * Real.pi / 16))^6 +
  (3 * Real.sqrt 2 / 8) = 5 / 4 :=
by
  sorry

end trigonometric_expression_eval_l87_87294


namespace larger_number_is_72_l87_87315

theorem larger_number_is_72 (a b : ℕ) (h1 : 5 * b = 6 * a) (h2 : b - a = 12) : b = 72 :=
by
  sorry

end larger_number_is_72_l87_87315


namespace free_space_on_new_drive_l87_87464

theorem free_space_on_new_drive
  (initial_free : ℝ) (initial_used : ℝ) (delete_size : ℝ) (new_files_size : ℝ) (new_drive_size : ℝ) :
  initial_free = 2.4 → initial_used = 12.6 → delete_size = 4.6 → new_files_size = 2 → new_drive_size = 20 →
  (new_drive_size - ((initial_used - delete_size) + new_files_size)) = 10 :=
by simp; sorry

end free_space_on_new_drive_l87_87464


namespace Jake_has_more_peaches_than_Jill_l87_87551

variables (Jake Steven Jill : ℕ)
variable (h1 : Jake = Steven - 5)
variable (h2 : Steven = Jill + 18)
variable (h3 : Jill = 87)

theorem Jake_has_more_peaches_than_Jill (Jake Steven Jill : ℕ) (h1 : Jake = Steven - 5) (h2 : Steven = Jill + 18) (h3 : Jill = 87) :
  Jake - Jill = 13 :=
by
  sorry

end Jake_has_more_peaches_than_Jill_l87_87551


namespace sasha_train_problem_l87_87818

def wagon_number (W : ℕ) (S : ℕ) : Prop :=
  -- Conditions
  (1 ≤ W ∧ W ≤ 9) ∧          -- Wagon number is a single-digit number
  (S < W) ∧                  -- Seat number is less than the wagon number
  ( (W = 1 ∧ S ≠ 1) ∨ 
    (W = 2 ∧ S = 1)
  ) -- Monday is the 1st or 2nd day of the month and corresponding seat constraints

theorem sasha_train_problem :
  ∃ (W S : ℕ), wagon_number W S ∧ W = 2 ∧ S = 1 :=
by
  sorry

end sasha_train_problem_l87_87818


namespace spending_difference_l87_87940

-- Define the cost of the candy bar
def candy_bar_cost : ℕ := 6

-- Define the cost of the chocolate
def chocolate_cost : ℕ := 3

-- Prove the difference between candy_bar_cost and chocolate_cost
theorem spending_difference : candy_bar_cost - chocolate_cost = 3 :=
by
    sorry

end spending_difference_l87_87940


namespace boss_salary_percentage_increase_l87_87506

theorem boss_salary_percentage_increase (W B : ℝ) (h : W = 0.2 * B) : ((B / W - 1) * 100) = 400 := by
sorry

end boss_salary_percentage_increase_l87_87506


namespace nancy_total_money_l87_87693

theorem nancy_total_money (n : ℕ) (d : ℕ) (h1 : n = 9) (h2 : d = 5) : n * d = 45 := 
by
  sorry

end nancy_total_money_l87_87693


namespace tony_squat_weight_l87_87212

-- Definitions from conditions
def curl_weight := 90
def military_press_weight := 2 * curl_weight
def squat_weight := 5 * military_press_weight

-- Theorem statement
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end tony_squat_weight_l87_87212


namespace area_of_rectangle_R_l87_87929

-- Define the side lengths of the squares and rectangles involved
def larger_square_side := 4
def smaller_square_side := 2
def rectangle_side1 := 1
def rectangle_side2 := 4

-- The areas of these shapes
def area_larger_square := larger_square_side * larger_square_side
def area_smaller_square := smaller_square_side * smaller_square_side
def area_first_rectangle := rectangle_side1 * rectangle_side2

-- Define the sum of all possible values for the area of rectangle R
def area_remaining := area_larger_square - (area_smaller_square + area_first_rectangle)

theorem area_of_rectangle_R : area_remaining = 8 := sorry

end area_of_rectangle_R_l87_87929


namespace sequence_a4_eq_neg3_l87_87295

theorem sequence_a4_eq_neg3 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 4 = -3 :=
by
  sorry

end sequence_a4_eq_neg3_l87_87295


namespace parallelepiped_length_l87_87677

theorem parallelepiped_length :
  ∃ n : ℕ, (n ≥ 7) ∧ (n * (n - 2) * (n - 4) = 3 * ((n - 2) * (n - 4) * (n - 6))) ∧ n = 18 :=
by
  sorry

end parallelepiped_length_l87_87677


namespace trigonometric_identity_l87_87963

theorem trigonometric_identity (α : Real) (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 9 / 5 :=
sorry

end trigonometric_identity_l87_87963


namespace integer_division_condition_l87_87187

theorem integer_division_condition (n : ℕ) (h1 : n > 1): (∃ k : ℕ, 2^n + 1 = k * n^2) → n = 3 :=
by sorry

end integer_division_condition_l87_87187


namespace initial_water_in_hole_l87_87201

theorem initial_water_in_hole (total_needed additional_needed initial : ℕ) (h1 : total_needed = 823) (h2 : additional_needed = 147) :
  initial = total_needed - additional_needed :=
by
  sorry

end initial_water_in_hole_l87_87201


namespace total_games_l87_87024

def joan_games_this_year : ℕ := 4
def joan_games_last_year : ℕ := 9

theorem total_games (this_year_games last_year_games : ℕ) 
    (h1 : this_year_games = joan_games_this_year) 
    (h2 : last_year_games = joan_games_last_year) : 
    this_year_games + last_year_games = 13 := 
by
  rw [h1, h2]
  exact rfl

end total_games_l87_87024


namespace curve_representation_l87_87236

   theorem curve_representation :
     ∀ (x y : ℝ), x^4 - y^4 - 4*x^2 + 4*y^2 = 0 ↔ (x + y = 0 ∨ x - y = 0 ∨ x^2 + y^2 = 4) :=
   by
     sorry
   
end curve_representation_l87_87236


namespace vegetables_sold_mass_correct_l87_87660

-- Definitions based on the problem's conditions
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8
def total_mass_vegetables := mass_carrots + mass_zucchini + mass_broccoli
def mass_of_vegetables_sold := total_mass_vegetables / 2

-- Theorem to be proved
theorem vegetables_sold_mass_correct : mass_of_vegetables_sold = 18 := by 
  sorry

end vegetables_sold_mass_correct_l87_87660


namespace cos_double_angle_l87_87420

theorem cos_double_angle (α : ℝ) (h : Real.sin (π/6 - α) = 1/3) :
  Real.cos (2 * (π/3 + α)) = -7/9 :=
by
  sorry

end cos_double_angle_l87_87420


namespace tetrahedron_ineq_l87_87955

variable (P Q R S : ℝ)

-- Given conditions
axiom ortho_condition : S^2 = P^2 + Q^2 + R^2

theorem tetrahedron_ineq (P Q R S : ℝ) (ortho_condition : S^2 = P^2 + Q^2 + R^2) :
  (P + Q + R) / S ≤ Real.sqrt 3 := by
  sorry

end tetrahedron_ineq_l87_87955


namespace power_modulus_l87_87177

theorem power_modulus (n : ℕ) : (2 : ℕ) ^ 345 % 5 = 2 :=
by sorry

end power_modulus_l87_87177


namespace johns_age_less_than_six_times_brothers_age_l87_87104

theorem johns_age_less_than_six_times_brothers_age 
  (B J : ℕ) 
  (h1 : B = 8) 
  (h2 : J + B = 10) 
  (h3 : J = 6 * B - 46) : 
  6 * B - J = 46 :=
by
  rw [h1, h3]
  exact sorry

end johns_age_less_than_six_times_brothers_age_l87_87104


namespace rice_weight_per_container_in_grams_l87_87516

-- Define the initial problem conditions
def total_weight_pounds : ℚ := 35 / 6
def number_of_containers : ℕ := 5
def pound_to_grams : ℚ := 453.592

-- Define the expected answer
def expected_answer : ℚ := 529.1907

-- The statement to prove
theorem rice_weight_per_container_in_grams :
  (total_weight_pounds / number_of_containers) * pound_to_grams = expected_answer :=
by
  sorry

end rice_weight_per_container_in_grams_l87_87516


namespace max_product_of_two_integers_with_sum_2004_l87_87167

theorem max_product_of_two_integers_with_sum_2004 :
  ∃ x y : ℤ, x + y = 2004 ∧ (∀ a b : ℤ, a + b = 2004 → a * b ≤ x * y) ∧ x * y = 1004004 := 
by
  sorry

end max_product_of_two_integers_with_sum_2004_l87_87167


namespace find_abc_l87_87053

theorem find_abc (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : c < 4)
  (h4 : a + b + c = a * b * c) : (a = 1 ∧ b = 2 ∧ c = 3) ∨ 
                                 (a = -3 ∧ b = -2 ∧ c = -1) ∨ 
                                 (a = -1 ∧ b = 0 ∧ c = 1) ∨ 
                                 (a = -2 ∧ b = 0 ∧ c = 2) ∨ 
                                 (a = -3 ∧ b = 0 ∧ c = 3) :=
sorry

end find_abc_l87_87053


namespace zoe_total_cost_l87_87384

theorem zoe_total_cost 
  (app_cost : ℕ)
  (monthly_cost : ℕ)
  (item_cost : ℕ)
  (feature_cost : ℕ)
  (months_played : ℕ)
  (h1 : app_cost = 5)
  (h2 : monthly_cost = 8)
  (h3 : item_cost = 10)
  (h4 : feature_cost = 12)
  (h5 : months_played = 2) :
  app_cost + (months_played * monthly_cost) + item_cost + feature_cost = 43 := 
by 
  sorry

end zoe_total_cost_l87_87384


namespace equivalent_single_discount_l87_87909

theorem equivalent_single_discount :
  ∀ (x : ℝ), ((1 - 0.15) * (1 - 0.10) * (1 - 0.05) * x) = (1 - 0.273) * x :=
by
  intros x
  --- This proof is left blank intentionally.
  sorry

end equivalent_single_discount_l87_87909


namespace tournament_total_players_l87_87844

/--
In a tournament involving n players:
- Each player scored half of all their points in matches against participants who took the last three places.
- Each game results in 1 point.
- Total points from matches among the last three (bad) players = 3.
- The number of games between good and bad players = 3n - 9.
- Total points good players scored from bad players = 3n - 12.
- Games among good players total to (n-3)(n-4)/2 resulting points.
Prove that the total number of participants in the tournament is 9.
-/
theorem tournament_total_players (n : ℕ) :
  3 * (n - 4) = (n - 3) * (n - 4) / 2 → 
  n = 9 :=
by
  intros h
  sorry

end tournament_total_players_l87_87844


namespace cuboid_first_edge_length_l87_87713

theorem cuboid_first_edge_length (x : ℝ) (hx : 180 = x * 5 * 6) : x = 6 :=
by
  sorry

end cuboid_first_edge_length_l87_87713


namespace problem_one_problem_two_l87_87722

theorem problem_one (α : ℝ) (h : Real.tan α = 2) : (3 * Real.sin α - 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

theorem problem_two (α : ℝ) (h : Real.tan α = 2) (h_quadrant : α > π ∧ α < 3 * π / 2) : Real.cos α = - (Real.sqrt 5 / 5) :=
by
  sorry

end problem_one_problem_two_l87_87722


namespace total_dividend_received_l87_87601

noncomputable def investmentAmount : Nat := 14400
noncomputable def faceValue : Nat := 100
noncomputable def premium : Real := 0.20
noncomputable def declaredDividend : Real := 0.07

theorem total_dividend_received :
  let cost_per_share := faceValue * (1 + premium)
  let number_of_shares := investmentAmount / cost_per_share
  let dividend_per_share := faceValue * declaredDividend
  let total_dividend := number_of_shares * dividend_per_share
  total_dividend = 840 := 
by 
  sorry

end total_dividend_received_l87_87601


namespace sin_double_angle_value_l87_87109

theorem sin_double_angle_value (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : (1/2) * Real.cos (2 * α) = Real.sin (π/4 + α)) :
  Real.sin (2 * α) = -1 :=
by
  sorry

end sin_double_angle_value_l87_87109


namespace equal_number_of_boys_and_girls_l87_87670

theorem equal_number_of_boys_and_girls
    (num_classrooms : ℕ) (girls : ℕ) (total_per_classroom : ℕ)
    (equal_boys_and_girls : ∀ (c : ℕ), c ≤ num_classrooms → (girls + boys) = total_per_classroom):
    num_classrooms = 4 → girls = 44 → total_per_classroom = 25 → boys = 44 :=
by
  sorry

end equal_number_of_boys_and_girls_l87_87670


namespace base_r_representation_26_eq_32_l87_87654

theorem base_r_representation_26_eq_32 (r : ℕ) : 
  26 = 3 * r + 6 → r = 8 :=
by
  sorry

end base_r_representation_26_eq_32_l87_87654


namespace bianca_picture_books_shelves_l87_87287

theorem bianca_picture_books_shelves (total_shelves : ℕ) (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 8 →
  mystery_shelves = 5 →
  total_books = 72 →
  total_shelves = (total_books - (mystery_shelves * books_per_shelf)) / books_per_shelf →
  total_shelves = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end bianca_picture_books_shelves_l87_87287


namespace paul_oil_change_rate_l87_87331

theorem paul_oil_change_rate (P : ℕ) (h₁ : 8 * (P + 3) = 40) : P = 2 :=
by
  sorry

end paul_oil_change_rate_l87_87331


namespace parallel_segment_length_l87_87442

/-- In \( \triangle ABC \), given side lengths AB = 500, BC = 550, and AC = 650,
there exists an interior point P such that each segment drawn parallel to the
sides of the triangle and passing through P splits the sides into segments proportional
to the overall sides of the triangle. Prove that the length \( d \) of each segment
parallel to the sides is 28.25 -/
theorem parallel_segment_length
  (A B C P : Type)
  (d AB BC AC : ℝ)
  (ha : AB = 500)
  (hb : BC = 550)
  (hc : AC = 650)
  (hp : AB * BC = AC * 550) -- This condition ensures proportionality of segments
  : d = 28.25 :=
sorry

end parallel_segment_length_l87_87442


namespace solve_xyz_system_l87_87636

theorem solve_xyz_system :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 
    (x * (6 - y) = 9) ∧ 
    (y * (6 - z) = 9) ∧ 
    (z * (6 - x) = 9) ∧ 
    x = 3 ∧ y = 3 ∧ z = 3 :=
by
  sorry

end solve_xyz_system_l87_87636


namespace brianna_marbles_lost_l87_87223

theorem brianna_marbles_lost
  (total_marbles : ℕ)
  (remaining_marbles : ℕ)
  (L : ℕ)
  (gave_away : ℕ)
  (dog_ate : ℚ)
  (h1 : total_marbles = 24)
  (h2 : remaining_marbles = 10)
  (h3 : gave_away = 2 * L)
  (h4 : dog_ate = L / 2)
  (h5 : total_marbles - remaining_marbles = L + gave_away + dog_ate) : L = 4 := 
by
  sorry

end brianna_marbles_lost_l87_87223


namespace polynomial_expansion_sum_l87_87957

theorem polynomial_expansion_sum (a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a_6 + a_5 + a_4 + a_3 + a_2 + a_1 + a = 64 :=
by
  -- Proof is not needed, placeholder here.
  sorry

end polynomial_expansion_sum_l87_87957


namespace sum_quotient_dividend_divisor_l87_87768

theorem sum_quotient_dividend_divisor (D : ℕ) (d : ℕ) (Q : ℕ) 
  (h1 : D = 54) (h2 : d = 9) (h3 : D = Q * d) : 
  (Q + D + d) = 69 :=
by
  sorry

end sum_quotient_dividend_divisor_l87_87768


namespace length_QR_l87_87691

-- Let's define the given conditions and the theorem to prove

-- Define the lengths of the sides of the triangle
def PQ : ℝ := 4
def PR : ℝ := 7
def PM : ℝ := 3.5

-- Define the median formula
def median_formula (PQ PR QR PM : ℝ) := PM = 0.5 * Real.sqrt (2 * PQ^2 + 2 * PR^2 - QR^2)

-- The theorem to prove: QR = 9
theorem length_QR 
  (hPQ : PQ = 4) 
  (hPR : PR = 7) 
  (hPM : PM = 3.5) 
  (hMedian : median_formula PQ PR QR PM) : 
  QR = 9 :=
sorry  -- proof will be here

end length_QR_l87_87691


namespace joann_lollipop_wednesday_l87_87505

variable (a : ℕ) (d : ℕ) (n : ℕ)

def joann_lollipop_count (a d n : ℕ) : ℕ :=
  a + d * n

theorem joann_lollipop_wednesday :
  let a := 4
  let d := 3
  let total_days := 7
  let target_total := 133
  ∀ (monday tuesday wednesday thursday friday saturday sunday : ℕ),
    monday = a ∧
    tuesday = a + d ∧
    wednesday = a + 2 * d ∧
    thursday = a + 3 * d ∧
    friday = a + 4 * d ∧
    saturday = a + 5 * d ∧
    sunday = a + 6 * d ∧
    (monday + tuesday + wednesday + thursday + friday + saturday + sunday = target_total) →
    wednesday = 10 :=
by
  sorry

end joann_lollipop_wednesday_l87_87505


namespace equation_solution_system_solution_l87_87002

theorem equation_solution (x : ℚ) :
  (3 * x + 1) / 5 = 1 - (4 * x + 3) / 2 ↔ x = -7 / 26 :=
by sorry

theorem system_solution (x y : ℚ) :
  (3 * x - 4 * y = 14) ∧ (5 * x + 4 * y = 2) ↔
  (x = 2) ∧ (y = -2) :=
by sorry

end equation_solution_system_solution_l87_87002


namespace dogs_not_eating_any_foods_l87_87754

theorem dogs_not_eating_any_foods :
  let total_dogs := 80
  let dogs_like_watermelon := 18
  let dogs_like_salmon := 58
  let dogs_like_both_salmon_watermelon := 7
  let dogs_like_chicken := 16
  let dogs_like_both_chicken_salmon := 6
  let dogs_like_both_chicken_watermelon := 4
  let dogs_like_all_three := 3
  let dogs_like_any_food := dogs_like_watermelon + dogs_like_salmon + dogs_like_chicken - 
                            dogs_like_both_salmon_watermelon - dogs_like_both_chicken_salmon - 
                            dogs_like_both_chicken_watermelon + dogs_like_all_three
  total_dogs - dogs_like_any_food = 2 := by
  sorry

end dogs_not_eating_any_foods_l87_87754


namespace total_eggs_l87_87314

-- Define the number of eggs eaten in each meal
def breakfast_eggs : ℕ := 2
def lunch_eggs : ℕ := 3
def dinner_eggs : ℕ := 1

-- Prove the total number of eggs eaten is 6
theorem total_eggs : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  sorry

end total_eggs_l87_87314


namespace number_of_parents_who_volunteered_to_bring_refreshments_l87_87249

theorem number_of_parents_who_volunteered_to_bring_refreshments 
  (total : ℕ) (supervise : ℕ) (supervise_and_refreshments : ℕ) (N : ℕ) (R : ℕ)
  (h_total : total = 84)
  (h_supervise : supervise = 25)
  (h_supervise_and_refreshments : supervise_and_refreshments = 11)
  (h_R_eq_1_5N : R = 3 * N / 2)
  (h_eq : total = (supervise - supervise_and_refreshments) + (R - supervise_and_refreshments) + supervise_and_refreshments + N) :
  R = 42 :=
by
  sorry

end number_of_parents_who_volunteered_to_bring_refreshments_l87_87249


namespace operation_proof_l87_87883

def operation (x y : ℤ) : ℤ := x * y - 3 * x - 4 * y

theorem operation_proof : (operation 7 2) - (operation 2 7) = 5 :=
by
  sorry

end operation_proof_l87_87883


namespace smallest_number_divisible_remainders_l87_87647

theorem smallest_number_divisible_remainders :
  ∃ n : ℕ,
    (n % 10 = 9) ∧
    (n % 9 = 8) ∧
    (n % 8 = 7) ∧
    (n % 7 = 6) ∧
    (n % 6 = 5) ∧
    (n % 5 = 4) ∧
    (n % 4 = 3) ∧
    (n % 3 = 2) ∧
    (n % 2 = 1) ∧
    n = 2519 :=
sorry

end smallest_number_divisible_remainders_l87_87647


namespace donna_fully_loaded_truck_weight_l87_87718

-- Define conditions
def empty_truck_weight : ℕ := 12000
def soda_crate_weight : ℕ := 50
def soda_crate_count : ℕ := 20
def dryer_weight : ℕ := 3000
def dryer_count : ℕ := 3

-- Calculate derived weights
def soda_total_weight : ℕ := soda_crate_weight * soda_crate_count
def fresh_produce_weight : ℕ := 2 * soda_total_weight
def dryer_total_weight : ℕ := dryer_weight * dryer_count

-- Define target weight of fully loaded truck
def fully_loaded_truck_weight : ℕ :=
  empty_truck_weight + soda_total_weight + fresh_produce_weight + dryer_total_weight

-- State and prove the theorem
theorem donna_fully_loaded_truck_weight :
  fully_loaded_truck_weight = 24000 :=
by
  -- Provide necessary calculations and proof steps if needed
  sorry

end donna_fully_loaded_truck_weight_l87_87718


namespace quadratic_eq_one_solution_has_ordered_pair_l87_87469

theorem quadratic_eq_one_solution_has_ordered_pair (a c : ℝ) 
  (h1 : a * c = 25) 
  (h2 : a + c = 17) 
  (h3 : a > c) : 
  (a, c) = (15.375, 1.625) :=
sorry

end quadratic_eq_one_solution_has_ordered_pair_l87_87469


namespace chocolate_pieces_l87_87702

theorem chocolate_pieces (total_pieces : ℕ) (michael_portion : ℕ) (paige_portion : ℕ) (mandy_portion : ℕ) 
  (h_total : total_pieces = 60) 
  (h_michael : michael_portion = total_pieces / 2) 
  (h_paige : paige_portion = (total_pieces - michael_portion) / 2) 
  (h_mandy : mandy_portion = total_pieces - (michael_portion + paige_portion)) : 
  mandy_portion = 15 :=
by
  sorry

end chocolate_pieces_l87_87702


namespace sum_of_octal_numbers_l87_87252

theorem sum_of_octal_numbers :
  (176 : ℕ) + 725 + 63 = 1066 := by
sorry

end sum_of_octal_numbers_l87_87252


namespace cos_alpha_minus_pi_l87_87137

theorem cos_alpha_minus_pi (α : Real) (h : Real.sin (α / 2) = Real.sqrt 3 / 4) : 
  Real.cos (α - Real.pi) = -5 / 8 :=
sorry

end cos_alpha_minus_pi_l87_87137


namespace find_LCM_l87_87065

-- Given conditions
def A := ℕ
def B := ℕ
def h := 22
def productAB := 45276

-- The theorem we want to prove
theorem find_LCM (a b lcm : ℕ) (hcf : ℕ) 
  (H_product : a * b = productAB) (H_hcf : hcf = h) : 
  (lcm = productAB / hcf) → 
  (a * b = hcf * lcm) :=
by
  intros H_lcm
  sorry

end find_LCM_l87_87065


namespace solve_system_l87_87039

theorem solve_system (x y z w : ℝ) :
  x - y + z - w = 2 ∧
  x^2 - y^2 + z^2 - w^2 = 6 ∧
  x^3 - y^3 + z^3 - w^3 = 20 ∧
  x^4 - y^4 + z^4 - w^4 = 60 ↔
  (x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
  (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2) :=
sorry

end solve_system_l87_87039


namespace f_one_value_l87_87835

def f (x a: ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem f_one_value (a : ℝ) (h : ∀ x, f x a ≥ 0) : f 1 a = 4 :=
by
  sorry

end f_one_value_l87_87835


namespace lcm_of_15_18_20_is_180_l87_87271

theorem lcm_of_15_18_20_is_180 : Nat.lcm (Nat.lcm 15 18) 20 = 180 := by
  sorry

end lcm_of_15_18_20_is_180_l87_87271


namespace problem_statement_l87_87920

noncomputable def probability_different_colors : ℚ :=
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red)

theorem problem_statement :
  let p_red := 7 / 11
  let p_green := 4 / 11
  (p_red * p_green) + (p_green * p_red) = 56 / 121 := by
  sorry

end problem_statement_l87_87920


namespace units_digit_G_100_l87_87750

def G (n : ℕ) : ℕ := 3 ^ (2 ^ n) + 1

theorem units_digit_G_100 : (G 100) % 10 = 2 := 
by
  sorry

end units_digit_G_100_l87_87750


namespace train_length_is_correct_l87_87324

noncomputable def length_of_train (t : ℝ) (v_train : ℝ) (v_man : ℝ) : ℝ :=
  let relative_speed : ℝ := (v_train - v_man) * (5/18)
  relative_speed * t

theorem train_length_is_correct :
  length_of_train 23.998 63 3 = 400 :=
by
  -- Placeholder for the proof
  sorry

end train_length_is_correct_l87_87324


namespace max_touched_points_by_line_l87_87072

noncomputable section

open Function

-- Definitions of the conditions
def coplanar_circles (circles : Set (Set ℝ)) : Prop :=
  ∀ c₁ c₂ : Set ℝ, c₁ ∈ circles → c₂ ∈ circles → c₁ ≠ c₂ → ∃ p : ℝ, p ∈ c₁ ∧ p ∈ c₂

def max_touched_points (line_circle : ℝ → ℝ) : ℕ :=
  2

-- The theorem statement that needs to be proven
theorem max_touched_points_by_line {circles : Set (Set ℝ)} (h_coplanar : coplanar_circles circles) :
  ∀ line : ℝ → ℝ, (∃ (c₁ c₂ c₃ : Set ℝ), c₁ ∈ circles ∧ c₂ ∈ circles ∧ c₃ ∈ circles ∧ c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃) →
  ∃ (p : ℕ), p = 6 := 
sorry

end max_touched_points_by_line_l87_87072


namespace number_is_125_l87_87655

/-- Let x be a real number such that the difference between x and 3/5 of x is 50. -/
def problem_statement (x : ℝ) : Prop :=
  x - (3 / 5) * x = 50

/-- Prove that the only number that satisfies the above condition is 125. -/
theorem number_is_125 (x : ℝ) (h : problem_statement x) : x = 125 :=
by
  sorry

end number_is_125_l87_87655


namespace maria_trip_distance_l87_87947

variable (D : ℕ) -- Defining the total distance D as a natural number

-- Defining the conditions given in the problem
def first_stop_distance := D / 2
def second_stop_distance := first_stop_distance - (1 / 3 * first_stop_distance)
def third_stop_distance := second_stop_distance - (2 / 5 * second_stop_distance)
def remaining_distance := 180

-- The statement to prove
theorem maria_trip_distance : third_stop_distance = remaining_distance → D = 900 :=
by
  sorry

end maria_trip_distance_l87_87947


namespace washington_high_teacher_student_ratio_l87_87081

theorem washington_high_teacher_student_ratio (students teachers : ℕ) (h_students : students = 1155) (h_teachers : teachers = 42) : (students / teachers : ℚ) = 27.5 :=
by
  sorry

end washington_high_teacher_student_ratio_l87_87081


namespace solve_for_x_l87_87456

-- Definitions of conditions
def sqrt_81_as_3sq : ℝ := (81 : ℝ)^(1/2)  -- sqrt(81)
def sqrt_81_as_3sq_simplified : ℝ := (3^4 : ℝ)^(1/2)  -- equivalent to (3^2) since 81 = 3^4

-- Theorem and goal statement
theorem solve_for_x :
  sqrt_81_as_3sq = sqrt_81_as_3sq_simplified →
  (3 : ℝ)^(3 * (2/3)) = sqrt_81_as_3sq :=
by
  -- Placeholder for the proof
  sorry

end solve_for_x_l87_87456


namespace seven_lines_divide_into_29_regions_l87_87574

open Function

theorem seven_lines_divide_into_29_regions : 
  ∀ n : ℕ, (∀ l m : ℕ, l ≠ m → l < n ∧ m < n) → 1 + n + (n.choose 2) = 29 :=
by
  sorry

end seven_lines_divide_into_29_regions_l87_87574


namespace cube_cut_edges_l87_87944

theorem cube_cut_edges (original_edges new_edges_per_vertex vertices : ℕ) (h1 : original_edges = 12) (h2 : new_edges_per_vertex = 6) (h3 : vertices = 8) :
  original_edges + new_edges_per_vertex * vertices = 60 :=
by
  sorry

end cube_cut_edges_l87_87944


namespace raj_earns_more_l87_87941

theorem raj_earns_more :
  let cost_per_sqft := 2
  let raj_length := 30
  let raj_width := 50
  let lena_length := 40
  let lena_width := 35
  let raj_area := raj_length * raj_width
  let lena_area := lena_length * lena_width
  let raj_earnings := raj_area * cost_per_sqft
  let lena_earnings := lena_area * cost_per_sqft
  raj_earnings - lena_earnings = 200 :=
by
  sorry

end raj_earns_more_l87_87941


namespace average_probable_weight_l87_87554

theorem average_probable_weight (weight : ℝ) (h1 : 61 < weight) (h2 : weight ≤ 64) : 
  (61 + 64) / 2 = 62.5 := 
by
  sorry

end average_probable_weight_l87_87554


namespace chatterboxes_total_jokes_l87_87434

theorem chatterboxes_total_jokes :
  let num_chatterboxes := 10
  let jokes_increasing := (100 * (100 + 1)) / 2
  let jokes_decreasing := (99 * (99 + 1)) / 2
  (jokes_increasing + jokes_decreasing) / num_chatterboxes = 1000 :=
by
  sorry

end chatterboxes_total_jokes_l87_87434


namespace number_of_partitions_indistinguishable_balls_into_boxes_l87_87396

/-- The number of distinct ways to partition 6 indistinguishable balls into 3 indistinguishable boxes is 7. -/
theorem number_of_partitions_indistinguishable_balls_into_boxes :
  ∃ n : ℕ, n = 7 := sorry

end number_of_partitions_indistinguishable_balls_into_boxes_l87_87396


namespace triangle_type_l87_87525

-- Definitions given in the problem
def is_not_equal (a : ℝ) (b : ℝ) : Prop := a ≠ b
def log_eq (b x : ℝ) : Prop := Real.log x = Real.log 4 / Real.log b + Real.log (4 * x - 4) / Real.log b

-- Main theorem stating the type of triangle ABC
theorem triangle_type (a b c A B C : ℝ) (h_b_ne_1 : is_not_equal b 1) (h_C_over_A_root : log_eq b (C / A)) (h_sin_B_over_sin_A_root : log_eq b (Real.sin B / Real.sin A)) : (B = 90) ∧ (A ≠ C) :=
by
  sorry

end triangle_type_l87_87525


namespace sum_of_star_tips_l87_87310

theorem sum_of_star_tips :
  let n := 9
  let alpha := 80  -- in degrees
  let total := n * alpha
  total = 720 := by sorry

end sum_of_star_tips_l87_87310


namespace diamond_value_l87_87016

def diamond (a b : ℕ) : ℚ := 1 / (a : ℚ) + 2 / (b : ℚ)

theorem diamond_value : ∀ (a b : ℕ), a + b = 10 ∧ a * b = 24 → diamond a b = 2 / 3 := by
  intros a b h
  sorry

end diamond_value_l87_87016


namespace value_of_business_l87_87696

theorem value_of_business (V : ℝ) (h₁ : (3/5) * (1/3) * V = 2000) : V = 10000 :=
by
  sorry

end value_of_business_l87_87696


namespace quadratic_root_range_l87_87775

/-- 
  Define the quadratic function y = ax^2 + bx + c for given values.
  Show that there exists x_1 in the interval (-1, 0) such that y = 0.
-/
theorem quadratic_root_range {a b c : ℝ} (h : a ≠ 0) 
  (h_minus3 : a * (-3)^2 + b * (-3) + c = -11)
  (h_minus2 : a * (-2)^2 + b * (-2) + c = -5)
  (h_minus1 : a * (-1)^2 + b * (-1) + c = -1)
  (h_0 : a * 0^2 + b * 0 + c = 1)
  (h_1 : a * 1^2 + b * 1 + c = 1) : 
  ∃ x1 : ℝ, -1 < x1 ∧ x1 < 0 ∧ a * x1^2 + b * x1 + c = 0 :=
sorry

end quadratic_root_range_l87_87775


namespace rect_length_is_20_l87_87850

-- Define the conditions
def rect_length_four_times_width (l w : ℝ) : Prop := l = 4 * w
def rect_area_100 (l w : ℝ) : Prop := l * w = 100

-- The main theorem to prove
theorem rect_length_is_20 {l w : ℝ} (h1 : rect_length_four_times_width l w) (h2 : rect_area_100 l w) : l = 20 := by
  sorry

end rect_length_is_20_l87_87850


namespace arithmetic_sequence_l87_87135

theorem arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) (h₀ : a 1 = 2) (h₁ : a 2 + a 3 = 13)
    (h₂ : ∀ n, a n = a 1 + (n - 1) * d) : a 5 = 14 :=
by
  sorry

end arithmetic_sequence_l87_87135


namespace min_theta_l87_87878

theorem min_theta (theta : ℝ) (k : ℤ) (h : theta + 2 * k * Real.pi = -11 / 4 * Real.pi) : 
  theta = -3 / 4 * Real.pi :=
  sorry

end min_theta_l87_87878


namespace min_value_abs_plus_one_l87_87674

theorem min_value_abs_plus_one : ∃ x : ℝ, |x| + 1 = 1 :=
by
  use 0
  sorry

end min_value_abs_plus_one_l87_87674


namespace boat_distance_against_stream_l87_87282

/-- 
  Given:
  1. The boat goes 13 km along the stream in one hour.
  2. The speed of the boat in still water is 11 km/hr.

  Prove:
  The distance the boat goes against the stream in one hour is 9 km.
-/
theorem boat_distance_against_stream (v_s : ℝ) (distance_along_stream time : ℝ) (v_still : ℝ) :
  distance_along_stream = 13 ∧ time = 1 ∧ v_still = 11 ∧ (v_still + v_s) = 13 → 
  (v_still - v_s) * time = 9 := by
  sorry

end boat_distance_against_stream_l87_87282


namespace max_ways_to_ascend_and_descend_l87_87230

theorem max_ways_to_ascend_and_descend :
  let east := 2
  let west := 3
  let south := 4
  let north := 1
  let ascend_descend_ways (ascend: ℕ) (n_1 n_2 n_3: ℕ) := ascend * (n_1 + n_2 + n_3)
  (ascend_descend_ways south east west north > ascend_descend_ways east west south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways west east south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways north east west south) := sorry

end max_ways_to_ascend_and_descend_l87_87230


namespace find_number_l87_87066

theorem find_number (n : ℕ) (h : (1 / 2 : ℝ) * n + 5 = 13) : n = 16 := 
by
  sorry

end find_number_l87_87066


namespace partI_solution_set_l87_87563

def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) - abs (x - a^2 - a)

theorem partI_solution_set (x : ℝ) : 
  (f x 1 ≤ 1) ↔ (x ≤ -1) :=
sorry

end partI_solution_set_l87_87563


namespace find_r_in_parallelogram_l87_87190

theorem find_r_in_parallelogram 
  (θ : ℝ) 
  (r : ℝ)
  (CAB DBA DBC ACB AOB : ℝ)
  (h1 : CAB = 3 * DBA)
  (h2 : DBC = 2 * DBA)
  (h3 : ACB = r * (t * AOB))
  (h4 : t = 4 / 3)
  (h5 : AOB = 180 - 2 * DBA)
  (h6 : ACB = 180 - 4 * DBA) :
  r = 1 / 3 :=
by
  sorry

end find_r_in_parallelogram_l87_87190
