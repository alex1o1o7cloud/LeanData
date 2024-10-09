import Mathlib

namespace video_games_expenditure_l1720_172010

theorem video_games_expenditure (allowance : ℝ) (books_expense : ℝ) (snacks_expense : ℝ) (clothes_expense : ℝ) 
    (initial_allowance : allowance = 50)
    (books_fraction : books_expense = 1 / 7 * allowance)
    (snacks_fraction : snacks_expense = 1 / 2 * allowance)
    (clothes_fraction : clothes_expense = 3 / 14 * allowance) :
    50 - (books_expense + snacks_expense + clothes_expense) = 7.15 :=
by
  sorry

end video_games_expenditure_l1720_172010


namespace binom_600_eq_1_l1720_172071

theorem binom_600_eq_1 : Nat.choose 600 600 = 1 :=
by sorry

end binom_600_eq_1_l1720_172071


namespace books_sold_l1720_172009

def initial_books : ℕ := 134
def given_books : ℕ := 39
def books_left : ℕ := 68

theorem books_sold : (initial_books - given_books - books_left = 27) := 
by 
  sorry

end books_sold_l1720_172009


namespace birds_percentage_hawks_l1720_172031

-- Define the conditions and the main proof problem
theorem birds_percentage_hawks (H : ℝ) :
  (0.4 * (1 - H) + 0.25 * 0.4 * (1 - H) + H = 0.65) → (H = 0.3) :=
by
  intro h
  sorry

end birds_percentage_hawks_l1720_172031


namespace find_m_l1720_172036

def hyperbola_focus (x y : ℝ) (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a^2 = 9 ∧ b^2 = -m ∧ (x - 0)^2 / a^2 - (y - 0)^2 / b^2 = 1

theorem find_m (m : ℝ) (H : hyperbola_focus 5 0 m) : m = -16 :=
by
  sorry

end find_m_l1720_172036


namespace least_possible_value_of_squares_l1720_172065

theorem least_possible_value_of_squares (a b x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 15 * a + 16 * b = x^2) (h2 : 16 * a - 15 * b = y^2) : 
  ∃ (x : ℕ) (y : ℕ), min (x^2) (y^2) = 231361 := 
sorry

end least_possible_value_of_squares_l1720_172065


namespace red_balloon_is_one_l1720_172087

open Nat

theorem red_balloon_is_one (R B : Nat) (h1 : R + B = 85) (h2 : R ≥ 1) (h3 : ∀ i j, i < R → j < R → i ≠ j → (i < B ∨ j < B)) : R = 1 :=
by
  sorry

end red_balloon_is_one_l1720_172087


namespace ellipse_condition_l1720_172007

theorem ellipse_condition (m : ℝ) :
  (1 < m ∧ m < 3) → ((m > 1 ∧ m < 3 ∧ m ≠ 2) ∨ (m = 2)) :=
by
  sorry

end ellipse_condition_l1720_172007


namespace geo_arith_sequences_sum_first_2n_terms_l1720_172088

variables (n : ℕ)

-- Given conditions in (a)
def common_ratio : ℕ := 3
def arithmetic_diff : ℕ := 2

-- The sequences provided in the solution (b)
def a_n (n : ℕ) : ℕ := common_ratio ^ n
def b_n (n : ℕ) : ℕ := 2 * n + 1

-- Sum formula for geometric series up to 2n terms
def S_2n (n : ℕ) : ℕ := (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n

theorem geo_arith_sequences :
  a_n n = common_ratio ^ n
  ∨ b_n n = 2 * n + 1 := sorry

theorem sum_first_2n_terms :
  S_2n n = (common_ratio^(2 * n + 1) - common_ratio) / 2 + 2 * n := sorry

end geo_arith_sequences_sum_first_2n_terms_l1720_172088


namespace apples_fallen_l1720_172095

theorem apples_fallen (H1 : ∃ ground_apples : ℕ, ground_apples = 10 + 3)
                      (H2 : ∃ tree_apples : ℕ, tree_apples = 5)
                      (H3 : ∃ total_apples : ℕ, total_apples = ground_apples ∧ total_apples = 10 + 3 + 5)
                      : ∃ fallen_apples : ℕ, fallen_apples = 13 :=
by
  sorry

end apples_fallen_l1720_172095


namespace inequality_always_holds_l1720_172059

theorem inequality_always_holds (a : ℝ) (h : a ≥ -2) : ∀ (x : ℝ), x^2 + a * |x| + 1 ≥ 0 :=
by
  sorry

end inequality_always_holds_l1720_172059


namespace value_of_x_l1720_172075

theorem value_of_x (x y : ℕ) (h1 : y = 864) (h2 : x^3 * 6^3 / 432 = y) : x = 12 :=
sorry

end value_of_x_l1720_172075


namespace evaluate_expression_l1720_172008

theorem evaluate_expression : 2 * (2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3) = 24240542 :=
by
  let a := 2009
  let b := 2010
  sorry

end evaluate_expression_l1720_172008


namespace fraction_zero_condition_l1720_172025

theorem fraction_zero_condition (x : ℝ) (h1 : (3 - |x|) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end fraction_zero_condition_l1720_172025


namespace Jamal_crayon_cost_l1720_172003

/-- Jamal bought 4 half dozen colored crayons at $2 per crayon. 
    He got a 10% discount on the total cost, and an additional 5% discount on the remaining amount. 
    After paying in US Dollars (USD), we want to know how much he spent in Euros (EUR) and British Pounds (GBP) 
    given that 1 USD is equal to 0.85 EUR and 1 USD is equal to 0.75 GBP. 
    This statement proves that the total cost was 34.884 EUR and 30.78 GBP. -/
theorem Jamal_crayon_cost :
  let number_of_crayons := 4 * 6
  let initial_cost := number_of_crayons * 2
  let first_discount := 0.10 * initial_cost
  let cost_after_first_discount := initial_cost - first_discount
  let second_discount := 0.05 * cost_after_first_discount
  let final_cost_usd := cost_after_first_discount - second_discount
  let final_cost_eur := final_cost_usd * 0.85
  let final_cost_gbp := final_cost_usd * 0.75
  final_cost_eur = 34.884 ∧ final_cost_gbp = 30.78 := 
by
  sorry

end Jamal_crayon_cost_l1720_172003


namespace total_prairie_area_l1720_172035

theorem total_prairie_area (A B C : ℕ) (Z1 Z2 Z3 : ℚ) (unaffected : ℕ) (total_area : ℕ) : 
  A = 55000 →
  B = 35000 →
  C = 45000 →
  Z1 = 0.80 →
  Z2 = 0.60 →
  Z3 = 0.95 →
  unaffected = 1500 →
  total_area = Z1 * A + Z2 * B + Z3 * C + unaffected →
  total_area = 109250 := sorry

end total_prairie_area_l1720_172035


namespace loss_is_negative_one_point_twenty_seven_percent_l1720_172082

noncomputable def book_price : ℝ := 600
noncomputable def gov_tax_rate : ℝ := 0.05
noncomputable def shipping_fee : ℝ := 20
noncomputable def seller_discount_rate : ℝ := 0.03
noncomputable def selling_price : ℝ := 624

noncomputable def gov_tax : ℝ := gov_tax_rate * book_price
noncomputable def seller_discount : ℝ := seller_discount_rate * book_price
noncomputable def total_cost : ℝ := book_price + gov_tax + shipping_fee - seller_discount
noncomputable def profit : ℝ := selling_price - total_cost
noncomputable def loss_percentage : ℝ := (profit / total_cost) * 100

theorem loss_is_negative_one_point_twenty_seven_percent :
  loss_percentage = -1.27 :=
by
  sorry

end loss_is_negative_one_point_twenty_seven_percent_l1720_172082


namespace expected_value_of_fair_8_sided_die_l1720_172019

-- Define the outcomes of the fair 8-sided die
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the probability of each outcome for a fair die
def prob (n : ℕ) : ℚ := 1 / 8

-- Calculate the expected value of the outcomes
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ x => prob x * x)).sum

-- State the theorem that the expected value is 4.5
theorem expected_value_of_fair_8_sided_die : expected_value = 4.5 :=
  sorry

end expected_value_of_fair_8_sided_die_l1720_172019


namespace sqrt_square_eq_self_l1720_172029

theorem sqrt_square_eq_self (a : ℝ) (h : a ≥ 1/2) :
  Real.sqrt ((2 * a - 1) ^ 2) = 2 * a - 1 :=
by
  sorry

end sqrt_square_eq_self_l1720_172029


namespace age_problem_l1720_172076

theorem age_problem (M D : ℕ) (h1 : M = 40) (h2 : 2 * D + M = 70) : 2 * M + D = 95 := by
  sorry

end age_problem_l1720_172076


namespace Smith_gave_Randy_l1720_172078

theorem Smith_gave_Randy {original_money Randy_keeps gives_Sally Smith_gives : ℕ}
  (h1: original_money = 3000)
  (h2: Randy_keeps = 2000)
  (h3: gives_Sally = 1200)
  (h4: Randy_keeps + gives_Sally = original_money + Smith_gives) :
  Smith_gives = 200 :=
by
  sorry

end Smith_gave_Randy_l1720_172078


namespace ratio_problem_l1720_172068

theorem ratio_problem (a b c d : ℝ) (h1 : a / b = 5) (h2 : b / c = 1 / 2) (h3 : c / d = 6) : 
  d / a = 1 / 15 :=
by sorry

end ratio_problem_l1720_172068


namespace expected_points_experts_over_100_games_probability_of_envelope_five_selected_l1720_172096

-- Game conditions and probabilities
def game_conditions (experts_points audience_points : ℕ) : Prop :=
  experts_points = 6 ∨ audience_points = 6

noncomputable def equal_teams := (1 : ℝ) / 2

-- Expected score of Experts over 100 games
noncomputable def expected_points_experts (games : ℕ) := 465

-- Probability that envelope number 5 is chosen in the next game
noncomputable def probability_envelope_five := (12 : ℝ) / 13

theorem expected_points_experts_over_100_games : 
  expected_points_experts 100 = 465 := 
sorry

theorem probability_of_envelope_five_selected : 
  probability_envelope_five = 0.715 := 
sorry

end expected_points_experts_over_100_games_probability_of_envelope_five_selected_l1720_172096


namespace value_of_7_prime_prime_l1720_172034

-- Define the function q' (written as q_prime in Lean)
def q_prime (q : ℕ) : ℕ := 3 * q - 3

-- Define the specific value problem
theorem value_of_7_prime_prime : q_prime (q_prime 7) = 51 := by
  sorry

end value_of_7_prime_prime_l1720_172034


namespace sandy_ordered_three_cappuccinos_l1720_172058

-- Definitions and conditions
def cost_cappuccino : ℝ := 2
def cost_iced_tea : ℝ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℝ := 1
def num_iced_teas : ℕ := 2
def num_cafe_lattes : ℕ := 2
def num_espressos : ℕ := 2
def change_received : ℝ := 3
def amount_paid : ℝ := 20

-- Calculation of costs
def total_cost_iced_teas : ℝ := num_iced_teas * cost_iced_tea
def total_cost_cafe_lattes : ℝ := num_cafe_lattes * cost_cafe_latte
def total_cost_espressos : ℝ := num_espressos * cost_espresso
def total_cost_other_drinks : ℝ := total_cost_iced_teas + total_cost_cafe_lattes + total_cost_espressos
def total_spent : ℝ := amount_paid - change_received
def cost_cappuccinos := total_spent - total_cost_other_drinks

-- Proof statement
theorem sandy_ordered_three_cappuccinos (num_cappuccinos : ℕ) : cost_cappuccinos = num_cappuccinos * cost_cappuccino → num_cappuccinos = 3 :=
by sorry

end sandy_ordered_three_cappuccinos_l1720_172058


namespace sum_of_four_digit_integers_up_to_4999_l1720_172093

theorem sum_of_four_digit_integers_up_to_4999 : 
  let a := 1000
  let l := 4999
  let n := l - a + 1
  let S := (n / 2) * (a + l)
  S = 11998000 := 
by
  sorry

end sum_of_four_digit_integers_up_to_4999_l1720_172093


namespace circle_equation_through_ABC_circle_equation_with_center_and_points_l1720_172066

-- Define points
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 0⟩
def B : Point := ⟨4, 0⟩
def C : Point := ⟨6, -2⟩

-- First problem: proof of the circle equation given points A, B, and C
theorem circle_equation_through_ABC :
  ∃ (D E F : ℝ), 
  (∀ (P : Point), (P = A ∨ P = B ∨ P = C) → P.x^2 + P.y^2 + D * P.x + E * P.y + F = 0) 
  ↔ (D = -5 ∧ E = 7 ∧ F = 4) := sorry

-- Second problem: proof of the circle equation given the y-coordinate of the center and points A and B
theorem circle_equation_with_center_and_points :
  ∃ (h k r : ℝ), 
  (h = (A.x + B.x) / 2 ∧ k = 2) ∧
  ∀ (P : Point), (P = A ∨ P = B) → (P.x - h)^2 + (P.y - k)^2 = r^2
  ↔ (h = 5 / 2 ∧ k = 2 ∧ r = 5 / 2) := sorry

end circle_equation_through_ABC_circle_equation_with_center_and_points_l1720_172066


namespace second_time_apart_l1720_172013

theorem second_time_apart 
  (glen_speed : ℕ) 
  (hannah_speed : ℕ)
  (initial_distance : ℕ) 
  (initial_time : ℕ)
  (relative_speed : ℕ)
  (hours_later : ℕ) :
  glen_speed = 37 →
  hannah_speed = 15 →
  initial_distance = 130 →
  initial_time = 6 →
  relative_speed = glen_speed + hannah_speed →
  hours_later = initial_distance / relative_speed →
  initial_time + hours_later = 8 + 30 / 60 :=
by
  intros
  sorry

end second_time_apart_l1720_172013


namespace not_directly_or_inversely_proportional_l1720_172045

theorem not_directly_or_inversely_proportional
  (P : ∀ x y : ℝ, x + y = 0 → (∃ k : ℝ, x = k * y))
  (Q : ∀ x y : ℝ, 3 * x * y = 10 → ∃ k : ℝ, x * y = k)
  (R : ∀ x y : ℝ, x = 5 * y → (∃ k : ℝ, x = k * y))
  (S : ∀ x y : ℝ, 3 * x + y = 10 → ¬ (∃ k : ℝ, x * y = k) ∧ ¬ (∃ k : ℝ, x = k * y))
  (T : ∀ x y : ℝ, x / y = Real.sqrt 3 → (∃ k : ℝ, x = k * y)) :
  ∀ x y : ℝ, 3 * x + y = 10 → ¬ (∃ k : ℝ, x * y = k) ∧ ¬ (∃ k : ℝ, x = k * y) := by
  sorry

end not_directly_or_inversely_proportional_l1720_172045


namespace shaniqua_earnings_correct_l1720_172032

noncomputable def calc_earnings : ℝ :=
  let haircut_tuesday := 5 * 10
  let haircut_normal := 5 * 12
  let styling_vip := (6 * 25) * (1 - 0.2)
  let styling_regular := 4 * 25
  let coloring_friday := (7 * 35) * (1 - 0.15)
  let coloring_normal := 3 * 35
  let treatment_senior := (3 * 50) * (1 - 0.1)
  let treatment_other := 4 * 50
  haircut_tuesday + haircut_normal + styling_vip + styling_regular + coloring_friday + coloring_normal + treatment_senior + treatment_other

theorem shaniqua_earnings_correct : calc_earnings = 978.25 := by
  sorry

end shaniqua_earnings_correct_l1720_172032


namespace same_exponent_for_all_bases_l1720_172004

theorem same_exponent_for_all_bases {a : Type} [LinearOrderedField a] {C : a} (ha : ∀ (a : a), a ≠ 0 → a^0 = C) : C = 1 :=
by
  sorry

end same_exponent_for_all_bases_l1720_172004


namespace totalNumberOfPupils_l1720_172005

-- Definitions of the conditions
def numberOfGirls : Nat := 232
def numberOfBoys : Nat := 253

-- Statement of the problem
theorem totalNumberOfPupils : numberOfGirls + numberOfBoys = 485 := by
  sorry

end totalNumberOfPupils_l1720_172005


namespace fraction_evaluation_l1720_172000

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem fraction_evaluation :
  (sqrt 2 * (sqrt 3 - sqrt 7)) / (2 * sqrt (3 + sqrt 5)) =
  (30 - 10 * sqrt 5 - 6 * sqrt 21 + 2 * sqrt 105) / 8 :=
by
  sorry

end fraction_evaluation_l1720_172000


namespace inequality_proof_l1720_172070

theorem inequality_proof (a b c : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |c * x^2 + b * x + a| ≤ 2 :=
by
  sorry

end inequality_proof_l1720_172070


namespace gate_paid_more_l1720_172079

def pre_booked_economy_cost : Nat := 10 * 140
def pre_booked_business_cost : Nat := 10 * 170
def total_pre_booked_cost : Nat := pre_booked_economy_cost + pre_booked_business_cost

def gate_economy_cost : Nat := 8 * 190
def gate_business_cost : Nat := 12 * 210
def gate_first_class_cost : Nat := 10 * 300
def total_gate_cost : Nat := gate_economy_cost + gate_business_cost + gate_first_class_cost

theorem gate_paid_more {gate_paid_more_cost : Nat} :
  total_gate_cost - total_pre_booked_cost = 3940 :=
by
  sorry

end gate_paid_more_l1720_172079


namespace no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l1720_172043

theorem no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ a b c : ℕ, 2 * n^2 + 1 = a^2 ∧ 3 * n^2 + 1 = b^2 ∧ 6 * n^2 + 1 = c^2 := by
  sorry

end no_positive_integer_n_such_that_2n2_plus_1_3n2_plus_1_6n2_plus_1_are_all_squares_l1720_172043


namespace sector_area_l1720_172084

theorem sector_area (theta : ℝ) (d : ℝ) (r : ℝ := d / 2) (circle_area : ℝ := π * r^2) 
    (sector_area : ℝ := (theta / 360) * circle_area) : 
  theta = 120 → d = 6 → sector_area = 3 * π :=
by
  intro htheta hd
  sorry

end sector_area_l1720_172084


namespace solve_abcd_l1720_172094

theorem solve_abcd : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 - d * x| ≤ 1) ∧ 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → |4 * x^3 + a * x^2 + b * x + c| ≤ 1) →
  d = 3 ∧ b = -3 ∧ a = 0 ∧ c = 0 :=
by
  sorry

end solve_abcd_l1720_172094


namespace inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l1720_172002

theorem inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed:
  (∀ a b : ℝ, a > b → a^3 > b^3) → (∀ a b : ℝ, a^3 > b^3 → a > b) :=
  by
  sorry

end inverse_of_a_gt_b_implies_a_cubed_gt_b_cubed_l1720_172002


namespace mk_div_km_l1720_172026

theorem mk_div_km 
  (m n k : ℕ) 
  (hm : 0 < m) 
  (hn : 0 < n) 
  (hk : 0 < k) 
  (h1 : m^n ∣ n^m) 
  (h2 : n^k ∣ k^n) : 
  m^k ∣ k^m := 
  sorry

end mk_div_km_l1720_172026


namespace tangent_condition_l1720_172048

theorem tangent_condition (a b : ℝ) :
  (4 * a^2 + b^2 = 1) ↔ 
  ∀ x y : ℝ, (y = 2 * x + 1) → ((x^2 / a^2) + (y^2 / b^2) = 1) → (∃! y, y = 2 * x + 1 ∧ (x^2 / a^2) + (y^2 / b^2) = 1) :=
sorry

end tangent_condition_l1720_172048


namespace measure_of_angle_y_l1720_172028

theorem measure_of_angle_y (m n : ℝ) (A B C D F G H : ℝ) :
  (m = n) → (A = 40) → (B = 90) → (B = 40) → (y = 80) :=
by
  -- proof steps to be filled in
  sorry

end measure_of_angle_y_l1720_172028


namespace determine_x0_minus_y0_l1720_172080

theorem determine_x0_minus_y0 
  (x0 y0 : ℝ)
  (data_points : List (ℝ × ℝ) := [(1, 2), (3, 5), (6, 8), (x0, y0)])
  (regression_eq : ∀ x, (x + 2) = (x + 2)) :
  x0 - y0 = -3 :=
by
  sorry

end determine_x0_minus_y0_l1720_172080


namespace regression_line_fits_l1720_172092

variables {x y : ℝ}

def points := [(1, 2), (2, 5), (4, 7), (5, 10)]

def regression_line (x : ℝ) : ℝ := x + 3

theorem regression_line_fits :
  (∀ p ∈ points, regression_line p.1 = p.2) ∧ (regression_line 3 = 6) :=
by
  sorry

end regression_line_fits_l1720_172092


namespace guaranteed_winning_strategy_l1720_172030

variable (a b : ℝ)

theorem guaranteed_winning_strategy (h : a ≠ b) : (a^3 + b^3) > (a^2 * b + a * b^2) :=
by 
  sorry

end guaranteed_winning_strategy_l1720_172030


namespace machine_x_produces_40_percent_l1720_172050

theorem machine_x_produces_40_percent (T X Y : ℝ) 
  (h1 : X + Y = T)
  (h2 : 0.009 * X + 0.004 * Y = 0.006 * T) :
  X = 0.4 * T :=
by
  sorry

end machine_x_produces_40_percent_l1720_172050


namespace determine_t_l1720_172037

theorem determine_t (t : ℝ) : 
  (3 * t - 9) * (4 * t - 3) = (4 * t - 16) * (3 * t - 9) → t = 7.8 :=
by
  intros h
  sorry

end determine_t_l1720_172037


namespace martian_year_length_ratio_l1720_172074

theorem martian_year_length_ratio :
  let EarthDay := 24 -- hours
  let MarsDay := EarthDay + 2 / 3 -- hours (since 40 minutes is 2/3 of an hour)
  let MartianYearDays := 668
  let EarthYearDays := 365.25
  (MartianYearDays * MarsDay) / EarthYearDays = 1.88 := by
{
  sorry
}

end martian_year_length_ratio_l1720_172074


namespace find_cost_price_l1720_172039

/-- 
Given:
- SP = 1290 (selling price)
- LossP = 14.000000000000002 (loss percentage)
Prove that: CP = 1500 (cost price)
--/
theorem find_cost_price (SP : ℝ) (LossP : ℝ) (CP : ℝ) (h1 : SP = 1290) (h2 : LossP = 14.000000000000002) : CP = 1500 :=
sorry

end find_cost_price_l1720_172039


namespace sum_algebra_values_l1720_172033

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 1 => 3
  | 2 => 1
  | 3 => 0
  | 4 => -1
  | 5 => -3
  | 6 => -1
  | 7 => 0
  | _ => 1

theorem sum_algebra_values : 
  alphabet_value 1 + 
  alphabet_value 12 + 
  alphabet_value 7 +
  alphabet_value 5 +
  alphabet_value 2 +
  alphabet_value 18 +
  alphabet_value 1 
  = 5 := by
  sorry

end sum_algebra_values_l1720_172033


namespace arithmetic_geometric_sequence_problem_l1720_172086

variable {a_n : ℕ → ℝ} {S : ℕ → ℝ}

-- Define the conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 0 + n * d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = (n * (a_n 0 + a_n (n-1))) / 2

def forms_geometric_sequence (a1 a3 a4 : ℝ) :=
  a3^2 = a1 * a4

-- The main proof statement
theorem arithmetic_geometric_sequence_problem
        (h_arith : is_arithmetic_sequence a_n)
        (h_sum : sum_of_first_n_terms a_n S)
        (h_geom : forms_geometric_sequence (a_n 0) (a_n 2) (a_n 3)) :
        (S 3 - S 2) / (S 5 - S 3) = 2 ∨ (S 3 - S 2) / (S 5 - S 3) = 1 / 2 :=
  sorry

end arithmetic_geometric_sequence_problem_l1720_172086


namespace domain_of_f_2x_minus_1_l1720_172015

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) (dom : ∀ x, f x ≠ 0 → (0 < x ∧ x < 1)) :
  ∀ x, f (2*x - 1) ≠ 0 → (1/2 < x ∧ x < 1) :=
by
  sorry

end domain_of_f_2x_minus_1_l1720_172015


namespace star_three_four_eq_zero_l1720_172053

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - 2 * a * b

theorem star_three_four_eq_zero : star 3 4 = 0 := sorry

end star_three_four_eq_zero_l1720_172053


namespace number_of_three_digit_multiples_of_7_l1720_172069

theorem number_of_three_digit_multiples_of_7 : 
  let smallest_multiple := 7 * Nat.ceil (100 / 7)
  let largest_multiple := 7 * Nat.floor (999 / 7)
  (largest_multiple - smallest_multiple) / 7 + 1 = 128 :=
by
  sorry

end number_of_three_digit_multiples_of_7_l1720_172069


namespace simplify_fraction_to_9_l1720_172017

-- Define the necessary terms and expressions
def problem_expr := (3^12)^2 - (3^10)^2
def problem_denom := (3^11)^2 - (3^9)^2
def simplified_expr := problem_expr / problem_denom

-- State the theorem we want to prove
theorem simplify_fraction_to_9 : simplified_expr = 9 := 
by sorry

end simplify_fraction_to_9_l1720_172017


namespace diapers_per_pack_l1720_172097

def total_boxes := 30
def packs_per_box := 40
def price_per_diaper := 5
def total_revenue := 960000

def total_packs_per_week := total_boxes * packs_per_box
def total_diapers_sold := total_revenue / price_per_diaper

theorem diapers_per_pack :
  total_diapers_sold / total_packs_per_week = 160 :=
by
  -- Placeholder for the actual proof
  sorry

end diapers_per_pack_l1720_172097


namespace children_ages_l1720_172085

-- Define the ages of the four children
variable (a b c d : ℕ)

-- Define the conditions
axiom h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom h2 : a + b + c + d = 31
axiom h3 : (a - 4) + (b - 4) + (c - 4) + (d - 4) = 16
axiom h4 : (a - 7) + (b - 7) + (c - 7) + (d - 7) = 8
axiom h5 : (a - 11) + (b - 11) + (c - 11) + (d - 11) = 1
noncomputable def ages : ℕ × ℕ × ℕ × ℕ := (12, 10, 6, 3)

-- The theorem to prove
theorem children_ages (h1 : a = 12) (h2 : b = 10) (h3 : c = 6) (h4 : d = 3) : a = 12 ∧ b = 10 ∧ c = 6 ∧ d = 3 :=
by sorry

end children_ages_l1720_172085


namespace roots_of_equation_l1720_172022

theorem roots_of_equation :
  {x : ℝ | -x * (x + 3) = x * (x + 3)} = {0, -3} :=
by
  sorry

end roots_of_equation_l1720_172022


namespace evans_family_children_count_l1720_172012

-- Let the family consist of the mother, the father, two grandparents, and children.
-- This proof aims to show x, the number of children, is 1.

theorem evans_family_children_count
  (m g y : ℕ) -- m = mother's age, g = average age of two grandparents, y = average age of children
  (x : ℕ) -- x = number of children
  (avg_family_age : (m + 50 + 2 * g + x * y) / (4 + x) = 30)
  (father_age : 50 = 50)
  (avg_non_father_age : (m + 2 * g + x * y) / (3 + x) = 25) :
  x = 1 :=
sorry

end evans_family_children_count_l1720_172012


namespace opposite_sides_of_line_l1720_172044

theorem opposite_sides_of_line (a : ℝ) (h1 : 0 < a) (h2 : a < 2) : (-a) * (2 - a) < 0 :=
sorry

end opposite_sides_of_line_l1720_172044


namespace value_of_x_l1720_172038

variable (x y : ℕ)

-- Conditions
axiom cond1 : x / y = 15 / 3
axiom cond2 : y = 27

-- Lean statement for the problem
theorem value_of_x : x = 135 :=
by
  have h1 := cond1
  have h2 := cond2
  sorry

end value_of_x_l1720_172038


namespace nelly_part_payment_is_875_l1720_172055

noncomputable def part_payment (total_cost remaining_amount : ℝ) :=
  0.25 * total_cost

theorem nelly_part_payment_is_875 (total_cost : ℝ) (remaining_amount : ℝ)
  (h1 : remaining_amount = 2625)
  (h2 : remaining_amount = 0.75 * total_cost) :
  part_payment total_cost remaining_amount = 875 :=
by
  sorry

end nelly_part_payment_is_875_l1720_172055


namespace time_to_fill_tank_l1720_172077

-- Definitions for conditions
def pipe_a := 50
def pipe_b := 75
def pipe_c := 100

-- Definition for the combined rate and time to fill the tank
theorem time_to_fill_tank : 
  (1 / pipe_a + 1 / pipe_b + 1 / pipe_c) * (300 / 13) = 1 := 
by
  sorry

end time_to_fill_tank_l1720_172077


namespace subtraction_of_decimals_l1720_172006

theorem subtraction_of_decimals :
  888.8888 - 444.4444 = 444.4444 := 
sorry

end subtraction_of_decimals_l1720_172006


namespace speed_of_stream_l1720_172056

theorem speed_of_stream
  (v_a v_s : ℝ)
  (h1 : v_a - v_s = 4)
  (h2 : v_a + v_s = 6) :
  v_s = 1 :=
by {
  sorry
}

end speed_of_stream_l1720_172056


namespace part1_part2_l1720_172024

-- Definition of the function
def f (a x : ℝ) := |x - a|

-- Proof statement for question 1
theorem part1 (a : ℝ)
  (h : ∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) :
  a = 3 := by
  sorry

-- Auxiliary function for question 2
def g (a x : ℝ) := f a (2 * x) + f a (x + 2)

-- Proof statement for question 2
theorem part2 (m : ℝ)
  (h : ∀ x : ℝ, g 3 x ≥ m) :
  m ≤ 1/2 := by
  sorry

end part1_part2_l1720_172024


namespace noemi_start_amount_l1720_172041

/-
  Conditions:
    lost_roulette = -600
    won_blackjack = 400
    lost_poker = -400
    won_baccarat = 500
    meal_cost = 200
    purse_end = 1800

  Prove: start_amount == 2300
-/

noncomputable def lost_roulette : Int := -600
noncomputable def won_blackjack : Int := 400
noncomputable def lost_poker : Int := -400
noncomputable def won_baccarat : Int := 500
noncomputable def meal_cost : Int := 200
noncomputable def purse_end : Int := 1800

noncomputable def net_gain : Int := lost_roulette + won_blackjack + lost_poker + won_baccarat

noncomputable def start_amount : Int := net_gain + meal_cost + purse_end

theorem noemi_start_amount : start_amount = 2300 :=
by
  sorry

end noemi_start_amount_l1720_172041


namespace undefined_values_l1720_172072

-- Define the expression to check undefined values
noncomputable def is_undefined (x : ℝ) : Prop :=
  x^3 - 9 * x = 0

-- Statement: For which real values of x is the expression undefined?
theorem undefined_values (x : ℝ) : is_undefined x ↔ x = 0 ∨ x = -3 ∨ x = 3 :=
sorry

end undefined_values_l1720_172072


namespace points_on_equation_correct_l1720_172020

theorem points_on_equation_correct (x y : ℝ) :
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) :=
by 
  sorry

end points_on_equation_correct_l1720_172020


namespace total_spent_l1720_172063

def spending (A B C : ℝ) : Prop :=
  (A = (13 / 10) * B) ∧
  (C = (4 / 5) * B) ∧
  (A = C + 15)

theorem total_spent (A B C : ℝ) (h : spending A B C) : A + B + C = 93 :=
by
  sorry

end total_spent_l1720_172063


namespace total_expenditure_l1720_172011

-- Definitions of costs and purchases
def bracelet_cost : ℕ := 4
def keychain_cost : ℕ := 5
def coloring_book_cost : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

-- Hypothesis stating the total expenditure for Paula and Olive
theorem total_expenditure
  (bracelet_cost keychain_cost coloring_book_cost : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ) :
  paula_bracelets * bracelet_cost + paula_keychains * keychain_cost + olive_coloring_books * coloring_book_cost + olive_bracelets * bracelet_cost = 20 := 
  by
  -- Applying the given costs
  let bracelet_cost := 4
  let keychain_cost := 5
  let coloring_book_cost := 3 

  -- Applying the purchases made by Paula and Olive
  let paula_bracelets := 2
  let paula_keychains := 1
  let olive_coloring_books := 1
  let olive_bracelets := 1

  sorry

end total_expenditure_l1720_172011


namespace measure_angle_A_l1720_172064

open Real

def triangle_area (a b c S : ℝ) (A B C : ℝ) : Prop :=
  S = (1 / 2) * b * c * sin A

def sides_and_angles (a b c A B C : ℝ) : Prop :=
  A = 2 * B

theorem measure_angle_A (a b c S A B C : ℝ)
  (h1 : triangle_area a b c S A B C)
  (h2 : sides_and_angles a b c A B C)
  (h3 : S = (a ^ 2) / 4) :
  A = π / 2 ∨ A = π / 4 :=
  sorry

end measure_angle_A_l1720_172064


namespace flagpole_proof_l1720_172054

noncomputable def flagpole_height (AC AD DE : ℝ) (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) : ℝ :=
  let DC := AC - AD
  let h_ratio := DE / DC
  h_ratio * AC

theorem flagpole_proof (AC AD DE : ℝ) (h_AC : AC = 4) (h_AD : AD = 3) (h_DE : DE = 1.8) 
  (h_ABC_DEC : (AC ≠ 0) ∧ (AC - AD ≠ 0) ∧ (DE ≠ 0)) :
  flagpole_height AC AD DE h_ABC_DEC = 7.2 := by
  sorry

end flagpole_proof_l1720_172054


namespace increasing_on_interval_of_m_l1720_172098

def f (m x : ℝ) := 2 * x^3 - 3 * m * x^2 + 6 * x

theorem increasing_on_interval_of_m (m : ℝ) :
  (∀ x : ℝ, 2 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0) → m ≤ 5 / 2 :=
sorry

end increasing_on_interval_of_m_l1720_172098


namespace square_area_l1720_172014

theorem square_area (A : ℝ) (s : ℝ) (prob_not_in_B : ℝ)
  (h1 : s * 4 = 32)
  (h2 : prob_not_in_B = 0.20987654320987653)
  (h3 : A - s^2 = prob_not_in_B * A) :
  A = 81 :=
by
  sorry

end square_area_l1720_172014


namespace hyperbola_eccentricity_correct_l1720_172049

noncomputable def hyperbola_eccentricity : Real :=
  let a := 5
  let b := 4
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  c / a

theorem hyperbola_eccentricity_correct :
  hyperbola_eccentricity = Real.sqrt 41 / 5 :=
by
  sorry

end hyperbola_eccentricity_correct_l1720_172049


namespace perimeter_of_flowerbed_l1720_172018

def width : ℕ := 4
def length : ℕ := 2 * width - 1
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem perimeter_of_flowerbed : perimeter length width = 22 := by
  sorry

end perimeter_of_flowerbed_l1720_172018


namespace remainder_sum_l1720_172089

-- Define the conditions given in the problem.
def remainder_13_mod_5 : ℕ := 3
def remainder_12_mod_5 : ℕ := 2
def remainder_11_mod_5 : ℕ := 1

theorem remainder_sum :
  ((13 ^ 6 + 12 ^ 7 + 11 ^ 8) % 5) = 3 := by
  sorry

end remainder_sum_l1720_172089


namespace solution_set_of_inequality_l1720_172027

theorem solution_set_of_inequality :
  { x : ℝ // (x - 2)^2 ≤ 2 * x + 11 } = { x : ℝ | -1 ≤ x ∧ x ≤ 7 } :=
sorry

end solution_set_of_inequality_l1720_172027


namespace winning_candidate_percentage_l1720_172090

theorem winning_candidate_percentage
  (votes_candidate1 : ℕ) (votes_candidate2 : ℕ) (votes_candidate3 : ℕ)
  (total_votes : ℕ) (winning_votes : ℕ) (percentage : ℚ)
  (h1 : votes_candidate1 = 1000)
  (h2 : votes_candidate2 = 2000)
  (h3 : votes_candidate3 = 4000)
  (h4 : total_votes = votes_candidate1 + votes_candidate2 + votes_candidate3)
  (h5 : winning_votes = votes_candidate3)
  (h6 : percentage = (winning_votes : ℚ) / total_votes * 100) :
  percentage = 57.14 := 
sorry

end winning_candidate_percentage_l1720_172090


namespace santino_fruit_total_l1720_172091

-- Definitions of the conditions
def numPapayaTrees : ℕ := 2
def numMangoTrees : ℕ := 3
def papayasPerTree : ℕ := 10
def mangosPerTree : ℕ := 20
def totalFruits (pTrees : ℕ) (pPerTree : ℕ) (mTrees : ℕ) (mPerTree : ℕ) : ℕ :=
  (pTrees * pPerTree) + (mTrees * mPerTree)

-- Theorem that states the total number of fruits is 80 given the conditions
theorem santino_fruit_total : totalFruits numPapayaTrees papayasPerTree numMangoTrees mangosPerTree = 80 := 
  sorry

end santino_fruit_total_l1720_172091


namespace configuration_of_points_l1720_172081

-- Define a type for points
structure Point :=
(x : ℝ)
(y : ℝ)

-- Assuming general position in the plane
def general_position (points : List Point) : Prop :=
  -- Add definition of general position, skipping exact implementation
  sorry

-- Define the congruence condition
def triangles_congruent (points : List Point) : Prop :=
  -- Add definition of the congruent triangles condition
  sorry

-- Define the vertices of two equilateral triangles inscribed in a circle
def two_equilateral_triangles (points : List Point) : Prop :=
  -- Add definition to check if points form two equilateral triangles in a circle
  sorry

theorem configuration_of_points (points : List Point) (h6 : points.length = 6) :
  general_position points →
  triangles_congruent points →
  two_equilateral_triangles points :=
by
  sorry

end configuration_of_points_l1720_172081


namespace no_such_number_l1720_172046

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def productOfDigitsIsPerfectSquare (n : ℕ) : Prop :=
  ∃ (d1 d2 : ℕ), n = 10 * d1 + d2 ∧ isPerfectSquare (d1 * d2)

theorem no_such_number :
  ¬ ∃ (N : ℕ),
    (N > 9) ∧ (N < 100) ∧ -- N is a two-digit number
    (N % 2 = 0) ∧        -- N is even
    (N % 13 = 0) ∧       -- N is a multiple of 13
    productOfDigitsIsPerfectSquare N := -- The product of digits of N is a perfect square
by
  sorry

end no_such_number_l1720_172046


namespace nathan_tokens_l1720_172001

theorem nathan_tokens
  (hockey_games : Nat := 5)
  (hockey_cost : Nat := 4)
  (basketball_games : Nat := 7)
  (basketball_cost : Nat := 5)
  (skee_ball_games : Nat := 3)
  (skee_ball_cost : Nat := 3)
  : hockey_games * hockey_cost + basketball_games * basketball_cost + skee_ball_games * skee_ball_cost = 64 := 
by
  sorry

end nathan_tokens_l1720_172001


namespace arithmetic_sequence_a7_l1720_172021

theorem arithmetic_sequence_a7 (a : ℕ → ℤ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d) (h3 : a 3 = 3) (h5 : a 5 = -3) : a 7 = -9 := 
sorry

end arithmetic_sequence_a7_l1720_172021


namespace michael_truck_meet_once_l1720_172099

noncomputable def meets_count (michael_speed : ℕ) (pail_distance : ℕ) (truck_speed : ℕ) (truck_stop_duration : ℕ) : ℕ :=
  if michael_speed = 4 ∧ pail_distance = 300 ∧ truck_speed = 8 ∧ truck_stop_duration = 45 then 1 else sorry

theorem michael_truck_meet_once :
  meets_count 4 300 8 45 = 1 :=
by simp [meets_count]

end michael_truck_meet_once_l1720_172099


namespace negation_log2_property_l1720_172040

theorem negation_log2_property :
  ¬(∃ x₀ : ℝ, Real.log x₀ / Real.log 2 ≤ 0) ↔ ∀ x : ℝ, Real.log x / Real.log 2 > 0 :=
by
  sorry

end negation_log2_property_l1720_172040


namespace operation_evaluation_l1720_172051

theorem operation_evaluation : 65 + 5 * 12 / (180 / 3) = 66 :=
by
  -- Parentheses
  have h1 : 180 / 3 = 60 := by sorry
  -- Multiplication and Division
  have h2 : 5 * 12 = 60 := by sorry
  have h3 : 60 / 60 = 1 := by sorry
  -- Addition
  exact sorry

end operation_evaluation_l1720_172051


namespace arc_length_of_sector_l1720_172060

theorem arc_length_of_sector (r θ : ℝ) (A : ℝ) (h₁ : r = 4)
  (h₂ : A = 7) : (1 / 2) * r^2 * θ = A → r * θ = 3.5 :=
by
  sorry

end arc_length_of_sector_l1720_172060


namespace problem_equivalence_l1720_172083

variables (P Q : Prop)

theorem problem_equivalence :
  (P ↔ Q) ↔ ((P → Q) ∧ (Q → P) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q)) :=
by sorry

end problem_equivalence_l1720_172083


namespace intersection_equiv_l1720_172047

def A : Set ℝ := { x : ℝ | x > 1 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }
def C : Set ℝ := { x : ℝ | 1 < x ∧ x < 2 }

theorem intersection_equiv : A ∩ B = C :=
by
  sorry

end intersection_equiv_l1720_172047


namespace evaluate_expression_l1720_172067

def f (x : ℕ) : ℕ :=
  match x with
  | 3 => 10
  | 4 => 17
  | 5 => 26
  | 6 => 37
  | 7 => 50
  | _ => 0  -- for any x not in the table, f(x) is undefined and defaults to 0

def f_inv (y : ℕ) : ℕ :=
  match y with
  | 10 => 3
  | 17 => 4
  | 26 => 5
  | 37 => 6
  | 50 => 7
  | _ => 0  -- for any y not in the table, f_inv(y) is undefined and defaults to 0

theorem evaluate_expression :
  f_inv (f_inv 50 * f_inv 10 + f_inv 26) = 5 :=
by
  sorry

end evaluate_expression_l1720_172067


namespace linear_condition_l1720_172023

theorem linear_condition (m : ℝ) : ¬ (m = 2) ↔ (∃ f : ℝ → ℝ, ∀ x, f x = (m - 2) * x + 2) :=
by
  sorry

end linear_condition_l1720_172023


namespace sample_size_l1720_172073

variable (total_employees : ℕ) (young_employees : ℕ) (middle_aged_employees : ℕ) (elderly_employees : ℕ) (young_in_sample : ℕ)

theorem sample_size (h1 : total_employees = 750) (h2 : young_employees = 350) (h3 : middle_aged_employees = 250) (h4 : elderly_employees = 150) (h5 : young_in_sample = 7) :
  ∃ sample_size, young_in_sample * total_employees / young_employees = sample_size ∧ sample_size = 15 :=
by
  sorry

end sample_size_l1720_172073


namespace intersection_A_B_l1720_172062

noncomputable def A : Set ℝ := {x | 9 * x ^ 2 < 1}

noncomputable def B : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 2 * x + 5 / 4}

theorem intersection_A_B :
  (A ∩ B) = {y | y ∈ Set.Ico (1/4 : ℝ) (1/3 : ℝ)} :=
by
  sorry

end intersection_A_B_l1720_172062


namespace largest_positive_integer_solution_l1720_172042

theorem largest_positive_integer_solution (x : ℕ) (h₁ : 1 ≤ x) (h₂ : x + 3 ≤ 6) : 
  x = 3 := by
  sorry

end largest_positive_integer_solution_l1720_172042


namespace chelsea_sugar_bags_l1720_172057

variable (n : ℕ)

-- Defining the conditions as hypotheses
def initial_sugar : ℕ := 24
def remaining_sugar : ℕ := 21
def sugar_lost : ℕ := initial_sugar - remaining_sugar
def torn_bag_sugar : ℕ := 2 * sugar_lost

-- Define the statement to prove
theorem chelsea_sugar_bags :
  n = initial_sugar / torn_bag_sugar → n = 4 :=
by
  sorry

end chelsea_sugar_bags_l1720_172057


namespace second_car_distance_l1720_172016

variables 
  (distance_apart : ℕ := 105)
  (d1 d2 d3 : ℕ := 25) -- distances 25 km, 15 km, 25 km respectively
  (d_road_back : ℕ := 15)
  (final_distance : ℕ := 20)

theorem second_car_distance 
  (car1_total_distance := d1 + d2 + d3 + d_road_back)
  (car2_distance : ℕ) :
  distance_apart - (car1_total_distance + car2_distance) = final_distance →
  car2_distance = 5 :=
sorry

end second_car_distance_l1720_172016


namespace largest_inscribed_square_length_l1720_172052

noncomputable def inscribed_square_length (s : ℝ) (n : ℕ) : ℝ :=
  let t := s / n
  let h := (Real.sqrt 3 / 2) * t
  s - 2 * h

theorem largest_inscribed_square_length :
  inscribed_square_length 12 3 = 12 - 4 * Real.sqrt 3 :=
by
  sorry

end largest_inscribed_square_length_l1720_172052


namespace range_of_a_in_second_quadrant_l1720_172061

theorem range_of_a_in_second_quadrant :
  (∀ (x y : ℝ), x^2 + y^2 + 6*x - 4*a*y + 3*a^2 + 9 = 0 → x < 0 ∧ y > 0) → (0 < a ∧ a < 3) :=
by
  sorry

end range_of_a_in_second_quadrant_l1720_172061
