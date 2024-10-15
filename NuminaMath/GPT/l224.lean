import Mathlib

namespace NUMINAMATH_GPT_determine_numbers_l224_22433

theorem determine_numbers (A B n : ℤ) (h1 : 0 ≤ n ∧ n ≤ 9) (h2 : A = 10 * B + n) (h3 : A + B = 2022) : 
  A = 1839 ∧ B = 183 :=
by
  -- proof will be filled in here
  sorry

end NUMINAMATH_GPT_determine_numbers_l224_22433


namespace NUMINAMATH_GPT_donation_calculation_l224_22416

/-- Patricia's initial hair length -/
def initial_length : ℕ := 14

/-- Patricia's hair growth -/
def growth_length : ℕ := 21

/-- Desired remaining hair length after donation -/
def remaining_length : ℕ := 12

/-- Calculate the donation length -/
def donation_length (L G R : ℕ) : ℕ := (L + G) - R

-- Theorem stating the donation length required for Patricia to achieve her goal.
theorem donation_calculation : donation_length initial_length growth_length remaining_length = 23 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_donation_calculation_l224_22416


namespace NUMINAMATH_GPT_find_d_l224_22458

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l224_22458


namespace NUMINAMATH_GPT_sum_abs_values_of_factors_l224_22442

theorem sum_abs_values_of_factors (a w c d : ℤ)
  (h1 : 6 * (x : ℤ)^2 + x - 12 = (a * x + w) * (c * x + d)) :
  abs a + abs w + abs c + abs d = 22 :=
sorry

end NUMINAMATH_GPT_sum_abs_values_of_factors_l224_22442


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l224_22483

def a : ℚ := 1 / 3
def b : ℚ := -1
def expr : ℚ := 4 * (3 * a^2 * b - a * b^2) - (2 * a * b^2 + 3 * a^2 * b)

theorem simplify_and_evaluate_expression : expr = -3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l224_22483


namespace NUMINAMATH_GPT_delegate_arrangement_probability_l224_22481

theorem delegate_arrangement_probability :
  let delegates := 10
  let countries := 3
  let independent_delegate := 1
  let total_seats := 10
  let m := 379
  let n := 420
  delegates = 10 ∧ countries = 3 ∧ independent_delegate = 1 ∧ total_seats = 10 →
  Nat.gcd m n = 1 →
  m + n = 799 :=
by
  sorry

end NUMINAMATH_GPT_delegate_arrangement_probability_l224_22481


namespace NUMINAMATH_GPT_value_of_a_l224_22407

theorem value_of_a (a b : ℚ) (h₁ : b = 3 * a) (h₂ : b = 12 - 5 * a) : a = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l224_22407


namespace NUMINAMATH_GPT_option_b_results_in_2x_cubed_l224_22437

variable (x : ℝ)

theorem option_b_results_in_2x_cubed : |x^3| + x^3 = 2 * x^3 := 
sorry

end NUMINAMATH_GPT_option_b_results_in_2x_cubed_l224_22437


namespace NUMINAMATH_GPT_inequality_system_integer_solutions_l224_22418

theorem inequality_system_integer_solutions :
  { x : ℤ | 5 * x + 1 > 3 * (x - 1) ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_inequality_system_integer_solutions_l224_22418


namespace NUMINAMATH_GPT_sahil_selling_price_l224_22422

-- Definitions based on the conditions
def purchase_price : ℕ := 10000
def repair_costs : ℕ := 5000
def transportation_charges : ℕ := 1000
def profit_percentage : ℕ := 50

def total_cost : ℕ := purchase_price + repair_costs + transportation_charges
def profit : ℕ := (profit_percentage * total_cost) / 100
def selling_price : ℕ := total_cost + profit

-- The theorem we need to prove
theorem sahil_selling_price : selling_price = 24000 :=
by
  sorry

end NUMINAMATH_GPT_sahil_selling_price_l224_22422


namespace NUMINAMATH_GPT_circle_tangent_unique_point_l224_22477

theorem circle_tangent_unique_point (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (x+4)^2 + (y-a)^2 = 25 → false) →
  (a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0) :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_unique_point_l224_22477


namespace NUMINAMATH_GPT_sin_double_angle_l224_22467

theorem sin_double_angle (k α : ℝ) (h : Real.cos (π / 4 - α) = k) : Real.sin (2 * α) = 2 * k^2 - 1 := 
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l224_22467


namespace NUMINAMATH_GPT_total_number_of_baseball_cards_l224_22400

def baseball_cards_total : Nat :=
  let carlos := 20
  let matias := carlos - 6
  let jorge := matias
  carlos + matias + jorge
   
theorem total_number_of_baseball_cards :
  baseball_cards_total = 48 :=
by
  rfl

end NUMINAMATH_GPT_total_number_of_baseball_cards_l224_22400


namespace NUMINAMATH_GPT_expand_and_simplify_l224_22490

theorem expand_and_simplify :
  ∀ (x : ℝ), 2 * x * (3 * x ^ 2 - 4 * x + 5) - (x ^ 2 - 3 * x) * (4 * x + 5) = 2 * x ^ 3 - x ^ 2 + 25 * x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l224_22490


namespace NUMINAMATH_GPT_average_increase_l224_22406

def scores : List ℕ := [92, 85, 90, 95]

def initial_average (s : List ℕ) : ℚ := (s.take 3).sum / 3

def new_average (s : List ℕ) : ℚ := s.sum / s.length

theorem average_increase :
  initial_average scores + 1.5 = new_average scores := 
by
  sorry

end NUMINAMATH_GPT_average_increase_l224_22406


namespace NUMINAMATH_GPT_train_speed_is_correct_l224_22419

/-- Define the length of the train and the time taken to cross the telegraph post. --/
def train_length : ℕ := 240
def crossing_time : ℕ := 16

/-- Define speed calculation based on train length and crossing time. --/
def train_speed : ℕ := train_length / crossing_time

/-- Prove that the computed speed of the train is 15 meters per second. --/
theorem train_speed_is_correct : train_speed = 15 := sorry

end NUMINAMATH_GPT_train_speed_is_correct_l224_22419


namespace NUMINAMATH_GPT_given_expression_simplifies_to_l224_22443

-- Given conditions: a ≠ ±1, a ≠ 0, b ≠ -1, b ≠ 0
variable (a b : ℝ)
variable (ha1 : a ≠ 1)
variable (ha2 : a ≠ -1)
variable (ha3 : a ≠ 0)
variable (hb1 : b ≠ 0)
variable (hb2 : b ≠ -1)

theorem given_expression_simplifies_to (h1 : a ≠ 1) (h2 : a ≠ -1) (h3 : a ≠ 0) (h4 : b ≠ 0) (h5 : b ≠ -1) :
    (a * b^(2/3) - b^(2/3) - a + 1) / ((1 - a^(1/3)) * ((a^(1/3) + 1)^2 - a^(1/3)) * (b^(1/3) + 1))
  + (a * b)^(1/3) * (1/a^(1/3) + 1/b^(1/3)) = 1 + a^(1/3) := by
  sorry

end NUMINAMATH_GPT_given_expression_simplifies_to_l224_22443


namespace NUMINAMATH_GPT_cost_of_candy_bar_l224_22461

def initial_amount : ℝ := 3.0
def remaining_amount : ℝ := 2.0

theorem cost_of_candy_bar :
  initial_amount - remaining_amount = 1.0 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_candy_bar_l224_22461


namespace NUMINAMATH_GPT_concentration_after_5_days_l224_22459

noncomputable def ozverin_concentration_after_iterations 
    (initial_volume : ℝ) (initial_concentration : ℝ)
    (drunk_volume : ℝ) (iterations : ℕ) : ℝ :=
initial_concentration * (1 - drunk_volume / initial_volume)^iterations

theorem concentration_after_5_days : 
  ozverin_concentration_after_iterations 0.5 0.4 0.05 5 = 0.236 :=
by
  sorry

end NUMINAMATH_GPT_concentration_after_5_days_l224_22459


namespace NUMINAMATH_GPT_cathy_total_money_l224_22456

variable (i d m : ℕ)
variable (h1 : i = 12)
variable (h2 : d = 25)
variable (h3 : m = 2 * d)

theorem cathy_total_money : i + d + m = 87 :=
by
  rw [h1, h2, h3]
  -- Continue proof steps here if necessary
  sorry

end NUMINAMATH_GPT_cathy_total_money_l224_22456


namespace NUMINAMATH_GPT_mean_home_runs_l224_22497

theorem mean_home_runs :
  let players_with_5 := 3
  let players_with_6 := 4
  let players_with_8 := 2
  let players_with_9 := 1
  let players_with_11 := 1
  let total_home_runs := (5 * players_with_5) + (6 * players_with_6) + (8 * players_with_8) + (9 * players_with_9) + (11 * players_with_11)
  let total_players := players_with_5 + players_with_6 + players_with_8 + players_with_9 + players_with_11
  (total_home_runs / total_players : ℚ) = 75 / 11 :=
by
  sorry

end NUMINAMATH_GPT_mean_home_runs_l224_22497


namespace NUMINAMATH_GPT_evaporation_period_length_l224_22427

def initial_water_amount : ℝ := 10
def daily_evaporation_rate : ℝ := 0.0008
def percentage_evaporated : ℝ := 0.004  -- 0.4% expressed as a decimal

theorem evaporation_period_length :
  (percentage_evaporated * initial_water_amount) / daily_evaporation_rate = 50 := by
  sorry

end NUMINAMATH_GPT_evaporation_period_length_l224_22427


namespace NUMINAMATH_GPT_denomination_of_four_bills_l224_22475

theorem denomination_of_four_bills (X : ℕ) (h1 : 10 * 20 + 8 * 10 + 4 * X = 300) : X = 5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_denomination_of_four_bills_l224_22475


namespace NUMINAMATH_GPT_gcd_of_repeated_three_digit_l224_22423

theorem gcd_of_repeated_three_digit : 
  ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → ∀ m ∈ {k : ℕ | ∃ n, 100 ≤ n ∧ n < 1000 ∧ k = 1001 * n}, Nat.gcd 1001 m = 1001 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_repeated_three_digit_l224_22423


namespace NUMINAMATH_GPT_units_digit_of_sequence_l224_22495

theorem units_digit_of_sequence : 
  (2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 + 2 * 3^4 + 2 * 3^5 + 2 * 3^6 + 2 * 3^7 + 2 * 3^8 + 2 * 3^9) % 10 = 8 := 
by 
  sorry

end NUMINAMATH_GPT_units_digit_of_sequence_l224_22495


namespace NUMINAMATH_GPT_intersection_M_N_l224_22454

open Set Int

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l224_22454


namespace NUMINAMATH_GPT_handshake_problem_l224_22409

theorem handshake_problem (n : ℕ) (hn : n = 11) (H : n * (n - 1) / 2 = 55) : 10 = n - 1 :=
by
  sorry

end NUMINAMATH_GPT_handshake_problem_l224_22409


namespace NUMINAMATH_GPT_berry_ratio_l224_22468

-- Define the conditions
variables (S V R : ℕ) -- Number of berries Stacy, Steve, and Sylar have
axiom h1 : S + V + R = 1100
axiom h2 : S = 800
axiom h3 : V = 2 * R

-- Define the theorem to be proved
theorem berry_ratio (h1 : S + V + R = 1100) (h2 : S = 800) (h3 : V = 2 * R) : S / V = 4 :=
by
  sorry

end NUMINAMATH_GPT_berry_ratio_l224_22468


namespace NUMINAMATH_GPT_exists_1990_gon_with_conditions_l224_22425

/-- A polygon structure with side lengths and properties to check equality of interior angles and side lengths -/
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℕ)
  (angles_equal : Prop)

/-- Given conditions -/
def condition_1 (P : Polygon 1990) : Prop := P.angles_equal
def condition_2 (P : Polygon 1990) : Prop :=
  ∃ (σ : Fin 1990 → Fin 1990), ∀ i, P.sides i = (σ i + 1)^2

/-- The main theorem to be proven -/
theorem exists_1990_gon_with_conditions :
  ∃ P : Polygon 1990, condition_1 P ∧ condition_2 P :=
sorry

end NUMINAMATH_GPT_exists_1990_gon_with_conditions_l224_22425


namespace NUMINAMATH_GPT_find_N_l224_22499

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem find_N (N : ℕ) (hN1 : N < 10000)
  (hN2 : N = 26 * sum_of_digits N) : N = 234 ∨ N = 468 := 
  sorry

end NUMINAMATH_GPT_find_N_l224_22499


namespace NUMINAMATH_GPT_symm_central_origin_l224_22469

noncomputable def f₁ (x : ℝ) : ℝ := 3^x

noncomputable def f₂ (x : ℝ) : ℝ := -3^(-x)

theorem symm_central_origin :
  ∀ x : ℝ, ∃ x' y y' : ℝ, (f₁ x = y) ∧ (f₂ x' = y') ∧ (x' = -x) ∧ (y' = -y) :=
by
  sorry

end NUMINAMATH_GPT_symm_central_origin_l224_22469


namespace NUMINAMATH_GPT_cost_price_of_watch_l224_22455

theorem cost_price_of_watch (CP SP_loss SP_gain : ℝ) (h1 : SP_loss = 0.79 * CP)
  (h2 : SP_gain = 1.04 * CP) (h3 : SP_gain - SP_loss = 140) : CP = 560 := by
  sorry

end NUMINAMATH_GPT_cost_price_of_watch_l224_22455


namespace NUMINAMATH_GPT_caterpillar_length_difference_l224_22440

-- Define the lengths of the caterpillars
def green_caterpillar_length : ℝ := 3
def orange_caterpillar_length : ℝ := 1.17

-- State the theorem we need to prove
theorem caterpillar_length_difference :
  green_caterpillar_length - orange_caterpillar_length = 1.83 :=
by
  sorry

end NUMINAMATH_GPT_caterpillar_length_difference_l224_22440


namespace NUMINAMATH_GPT_sum_of_powers_sequence_l224_22428

theorem sum_of_powers_sequence (a b : ℝ) 
  (h₁ : a + b = 1)
  (h₂ : a^2 + b^2 = 3)
  (h₃ : a^3 + b^3 = 4)
  (h₄ : a^4 + b^4 = 7)
  (h₅ : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end NUMINAMATH_GPT_sum_of_powers_sequence_l224_22428


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l224_22408

theorem problem_part1 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 * y - x * y^2 = 4 * Real.sqrt 2 := 
  sorry

theorem problem_part2 (x y : ℝ) (h1 : x = 1 / (3 - 2 * Real.sqrt 2)) (h2 : y = 1 / (3 + 2 * Real.sqrt 2)) : 
  x^2 - x * y + y^2 = 33 := 
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l224_22408


namespace NUMINAMATH_GPT_power_of_m_l224_22472

theorem power_of_m (m : ℕ) (h₁ : ∀ k : ℕ, m^k % 24 = 0) (h₂ : ∀ d : ℕ, d ∣ m → d ≤ 8) : ∃ k : ℕ, m^k = 24 :=
sorry

end NUMINAMATH_GPT_power_of_m_l224_22472


namespace NUMINAMATH_GPT_digit_A_divisibility_l224_22414

theorem digit_A_divisibility :
  ∃ (A : ℕ), (0 ≤ A ∧ A < 10) ∧ (∃ k_5 : ℕ, 353809 * 10 + A = 5 * k_5) ∧ 
  (∃ k_7 : ℕ, 353809 * 10 + A = 7 * k_7) ∧ (∃ k_11 : ℕ, 353809 * 10 + A = 11 * k_11) 
  ∧ A = 0 :=
by 
  sorry

end NUMINAMATH_GPT_digit_A_divisibility_l224_22414


namespace NUMINAMATH_GPT_divisibility_problem_l224_22448

theorem divisibility_problem :
  (2^62 + 1) % (2^31 + 2^16 + 1) = 0 := 
sorry

end NUMINAMATH_GPT_divisibility_problem_l224_22448


namespace NUMINAMATH_GPT_negation_of_existence_l224_22450

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end NUMINAMATH_GPT_negation_of_existence_l224_22450


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_count_l224_22482

theorem arithmetic_sequence_terms_count (a d l : Int) (h1 : a = 20) (h2 : d = -3) (h3 : l = -5) :
  ∃ n : Int, l = a + (n - 1) * d ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_count_l224_22482


namespace NUMINAMATH_GPT_original_cost_price_l224_22402

theorem original_cost_price (S P C : ℝ) (h1 : S = 260) (h2 : S = 1.20 * C) : C = 216.67 := sorry

end NUMINAMATH_GPT_original_cost_price_l224_22402


namespace NUMINAMATH_GPT_opposite_of_neg_quarter_l224_22421

theorem opposite_of_neg_quarter : -(- (1 / 4)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_quarter_l224_22421


namespace NUMINAMATH_GPT_m_in_A_l224_22489

variable (x : ℝ)
variable (A : Set ℝ := {x | x ≤ 2})
noncomputable def m : ℝ := Real.sqrt 2

theorem m_in_A : m ∈ A :=
sorry

end NUMINAMATH_GPT_m_in_A_l224_22489


namespace NUMINAMATH_GPT_omino_tilings_2_by_10_l224_22464

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n+2) => fib n + fib (n+1)

def omino_tilings (n : ℕ) : ℕ :=
  fib (n + 1)

theorem omino_tilings_2_by_10 : omino_tilings 10 = 3025 := by
  sorry

end NUMINAMATH_GPT_omino_tilings_2_by_10_l224_22464


namespace NUMINAMATH_GPT_show_R_r_eq_l224_22449

variables {a b c R r : ℝ}

-- Conditions
def sides_of_triangle (a b c : ℝ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

def circumradius (R a b c : ℝ) (Δ : ℝ) : Prop :=
R = a * b * c / (4 * Δ)

def inradius (r Δ : ℝ) (s : ℝ) : Prop :=
r = Δ / s

theorem show_R_r_eq (a b c : ℝ) (R r : ℝ) (Δ : ℝ) (s : ℝ) (h_sides : sides_of_triangle a b c)
  (h_circumradius : circumradius R a b c Δ)
  (h_inradius : inradius r Δ s)
  (h_semiperimeter : s = (a + b + c) / 2) :
  R * r = a * b * c / (2 * (a + b + c)) :=
sorry

end NUMINAMATH_GPT_show_R_r_eq_l224_22449


namespace NUMINAMATH_GPT_factor_x4_minus_81_l224_22466

variable (x : ℝ)

theorem factor_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
  by { -- proof steps would go here 
    sorry 
}

end NUMINAMATH_GPT_factor_x4_minus_81_l224_22466


namespace NUMINAMATH_GPT_abc_equivalence_l224_22438

theorem abc_equivalence (n : ℕ) (k : ℤ) (a b c : ℤ)
  (hn : 0 < n) (hk : k % 2 = 1)
  (h : a^n + k * b = b^n + k * c ∧ b^n + k * c = c^n + k * a) :
  a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_abc_equivalence_l224_22438


namespace NUMINAMATH_GPT_find_value_l224_22484

theorem find_value (x y : ℝ) (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 5 * x^2 + 11 * x * y + 5 * y^2 = 89 :=
by
  sorry

end NUMINAMATH_GPT_find_value_l224_22484


namespace NUMINAMATH_GPT_maximum_gel_pens_l224_22471

theorem maximum_gel_pens 
  (x y z : ℕ) 
  (h1 : x + y + z = 20)
  (h2 : 10 * x + 50 * y + 80 * z = 1000)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) 
  : y ≤ 13 :=
sorry

end NUMINAMATH_GPT_maximum_gel_pens_l224_22471


namespace NUMINAMATH_GPT_first_alloy_mass_l224_22465

theorem first_alloy_mass (x : ℝ) : 
  (0.12 * x + 2.8) / (x + 35) = 9.454545454545453 / 100 → 
  x = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_first_alloy_mass_l224_22465


namespace NUMINAMATH_GPT_utility_bill_amount_l224_22462

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end NUMINAMATH_GPT_utility_bill_amount_l224_22462


namespace NUMINAMATH_GPT_tom_paid_correct_amount_l224_22432

-- Define the conditions given in the problem
def kg_apples : ℕ := 8
def rate_apples : ℕ := 70
def kg_mangoes : ℕ := 9
def rate_mangoes : ℕ := 45

-- Define the cost calculations
def cost_apples : ℕ := kg_apples * rate_apples
def cost_mangoes : ℕ := kg_mangoes * rate_mangoes
def total_amount : ℕ := cost_apples + cost_mangoes

-- The proof problem statement
theorem tom_paid_correct_amount : total_amount = 965 :=
by
  -- The proof steps are omitted and replaced with sorry
  sorry

end NUMINAMATH_GPT_tom_paid_correct_amount_l224_22432


namespace NUMINAMATH_GPT_cat_food_more_than_dog_food_l224_22412

theorem cat_food_more_than_dog_food :
  let cat_food_packs := 6
  let cans_per_cat_pack := 9
  let dog_food_packs := 2
  let cans_per_dog_pack := 3
  let total_cat_food_cans := cat_food_packs * cans_per_cat_pack
  let total_dog_food_cans := dog_food_packs * cans_per_dog_pack
  total_cat_food_cans - total_dog_food_cans = 48 :=
by
  sorry

end NUMINAMATH_GPT_cat_food_more_than_dog_food_l224_22412


namespace NUMINAMATH_GPT_miles_per_gallon_l224_22413

theorem miles_per_gallon (miles gallons : ℝ) (h : miles = 100 ∧ gallons = 5) : miles / gallons = 20 := by
  cases h with
  | intro miles_eq gallons_eq =>
    rw [miles_eq, gallons_eq]
    norm_num

end NUMINAMATH_GPT_miles_per_gallon_l224_22413


namespace NUMINAMATH_GPT_trigonometric_identity_l224_22444

-- The main statement to prove
theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (2 * Real.sin α - 2 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α) = -2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l224_22444


namespace NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l224_22457

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem monotonicity_of_f (a : ℝ) : 
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧ 
  (a > 0 → ∀ x y : ℝ, 
    (x < y ∧ y ≤ Real.log a → f x a > f y a) ∧ 
    (x > Real.log a → f x a < f y a)) :=
by
  sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ 0 ≤ a ∧ a ≤ Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l224_22457


namespace NUMINAMATH_GPT_infinite_solutions_no_solutions_l224_22411

-- Define the geometric sequence with first term a1 = 1 and common ratio q
def a1 : ℝ := 1
def a2 (q : ℝ) : ℝ := a1 * q
def a3 (q : ℝ) : ℝ := a1 * q^2
def a4 (q : ℝ) : ℝ := a1 * q^3

-- Define the system of linear equations
def system_of_eqns (x y q : ℝ) : Prop :=
  a1 * x + a3 q * y = 3 ∧ a2 q * x + a4 q * y = -2

-- Conditions for infinitely many solutions
theorem infinite_solutions (q x y : ℝ) :
  q = -2 / 3 → ∃ x y, system_of_eqns x y q :=
by
  sorry

-- Conditions for no solutions
theorem no_solutions (q : ℝ) :
  q ≠ -2 / 3 → ¬∃ x y, system_of_eqns x y q :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_no_solutions_l224_22411


namespace NUMINAMATH_GPT_equation_solution_l224_22470

theorem equation_solution (x y : ℕ) :
  (x^2 + 1)^y - (x^2 - 1)^y = 2 * x^y ↔ 
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ ∃ k : ℕ, y = 2 * k ∧ k > 0) :=
by sorry

end NUMINAMATH_GPT_equation_solution_l224_22470


namespace NUMINAMATH_GPT_other_x_intercept_l224_22415

-- Definition of the two foci
def f1 : ℝ × ℝ := (0, 2)
def f2 : ℝ × ℝ := (3, 0)

-- One x-intercept is given as
def intercept1 : ℝ × ℝ := (0, 0)

-- We need to prove the other x-intercept is (15/4, 0)
theorem other_x_intercept : ∃ x : ℝ, (x, 0) = (15/4, 0) ∧
  (dist (x, 0) f1 + dist (x, 0) f2 = dist intercept1 f1 + dist intercept1 f2) :=
by
  sorry

end NUMINAMATH_GPT_other_x_intercept_l224_22415


namespace NUMINAMATH_GPT_root_polynomial_value_l224_22486

theorem root_polynomial_value (m : ℝ) (h : m^2 + 3 * m - 2022 = 0) : m^3 + 4 * m^2 - 2019 * m - 2023 = -1 :=
  sorry

end NUMINAMATH_GPT_root_polynomial_value_l224_22486


namespace NUMINAMATH_GPT_f_neg_2008_value_l224_22431

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem f_neg_2008_value (h : f a b 2008 = 10) : f a b (-2008) = -12 := by
  sorry

end NUMINAMATH_GPT_f_neg_2008_value_l224_22431


namespace NUMINAMATH_GPT_no_such_function_exists_l224_22485

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x ^ 2 - 1996 :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l224_22485


namespace NUMINAMATH_GPT_min_people_wearing_both_hat_and_glove_l224_22452

theorem min_people_wearing_both_hat_and_glove (n : ℕ) (x : ℕ) 
  (h1 : 2 * n = 5 * (8 : ℕ)) -- 2/5 of n people wear gloves
  (h2 : 3 * n = 4 * (15 : ℕ)) -- 3/4 of n people wear hats
  (h3 : n = 20): -- total number of people is 20
  x = 3 := -- minimum number of people wearing both a hat and a glove is 3
by sorry

end NUMINAMATH_GPT_min_people_wearing_both_hat_and_glove_l224_22452


namespace NUMINAMATH_GPT_worth_of_each_gift_l224_22491

def workers_per_block : Nat := 200
def total_amount_for_gifts : Nat := 6000
def number_of_blocks : Nat := 15

theorem worth_of_each_gift (workers_per_block : Nat) (total_amount_for_gifts : Nat) (number_of_blocks : Nat) : 
  (total_amount_for_gifts / (workers_per_block * number_of_blocks)) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_worth_of_each_gift_l224_22491


namespace NUMINAMATH_GPT_largest_square_plot_size_l224_22430

def field_side_length := 50
def available_fence_length := 4000

theorem largest_square_plot_size :
  ∃ (s : ℝ), (0 < s) ∧ (s ≤ field_side_length) ∧ 
  (100 * (field_side_length - s) = available_fence_length) →
  s = 10 :=
by
  sorry

end NUMINAMATH_GPT_largest_square_plot_size_l224_22430


namespace NUMINAMATH_GPT_fewer_mpg_in_city_l224_22445

def city_miles : ℕ := 336
def highway_miles : ℕ := 462
def city_mpg : ℕ := 24

def tank_size : ℕ := city_miles / city_mpg
def highway_mpg : ℕ := highway_miles / tank_size

theorem fewer_mpg_in_city : highway_mpg - city_mpg = 9 :=
by
  sorry

end NUMINAMATH_GPT_fewer_mpg_in_city_l224_22445


namespace NUMINAMATH_GPT_circular_garden_area_l224_22435

open Real

theorem circular_garden_area (r : ℝ) (h₁ : r = 8)
      (h₂ : 2 * π * r = (1 / 4) * π * r ^ 2) :
  π * r ^ 2 = 64 * π :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_circular_garden_area_l224_22435


namespace NUMINAMATH_GPT_inequality_proof_l224_22424

theorem inequality_proof (a b c : ℝ) (ha1 : 0 ≤ a) (ha2 : a ≤ 1) (hb1 : 0 ≤ b) (hb2 : b ≤ 1) (hc1 : 0 ≤ c) (hc2 : c ≤ 1) :
  (a / (b + c + 1) + b / (a + c + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l224_22424


namespace NUMINAMATH_GPT_solution_set_l224_22434

open BigOperators

noncomputable def f (x : ℝ) := 2016^x + Real.log (Real.sqrt (x^2 + 1) + x) / Real.log 2016 - 2016^(-x)

theorem solution_set (x : ℝ) (h1 : ∀ x, f (-x) = -f (x)) (h2 : ∀ x1 x2, x1 < x2 → f (x1) < f (x2)) :
  x > -1 / 4 ↔ f (3 * x + 1) + f (x) > 0 := 
by
  sorry

end NUMINAMATH_GPT_solution_set_l224_22434


namespace NUMINAMATH_GPT_area_of_triangle_XYZ_l224_22474

noncomputable def centroid (p1 p2 p3 : (ℚ × ℚ)) : (ℚ × ℚ) :=
((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

noncomputable def triangle_area (p1 p2 p3 : (ℚ × ℚ)) : ℚ :=
abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2 - p1.2 * p2.1 - p2.2 * p3.1 - p3.2 * p1.1) / 2)

noncomputable def point_A : (ℚ × ℚ) := (5, 12)
noncomputable def point_B : (ℚ × ℚ) := (0, 0)
noncomputable def point_C : (ℚ × ℚ) := (14, 0)

noncomputable def point_X : (ℚ × ℚ) :=
(109 / 13, 60 / 13)
noncomputable def point_Y : (ℚ × ℚ) :=
centroid point_A point_B point_X
noncomputable def point_Z : (ℚ × ℚ) :=
centroid point_B point_C point_Y

theorem area_of_triangle_XYZ : triangle_area point_X point_Y point_Z = 84 / 13 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_XYZ_l224_22474


namespace NUMINAMATH_GPT_goshawk_eurasian_reserve_hawks_l224_22436

variable (H P : ℝ)

theorem goshawk_eurasian_reserve_hawks :
  P = 100 ∧
  (35 / 100) * P = P - (H + (40 / 100) * (P - H) + (25 / 100) * (40 / 100) * (P - H))
    → H = 25 :=
by sorry

end NUMINAMATH_GPT_goshawk_eurasian_reserve_hawks_l224_22436


namespace NUMINAMATH_GPT_baseball_card_problem_l224_22479

theorem baseball_card_problem:
  let initial_cards := 15
  let maria_takes := (initial_cards + 1) / 2
  let cards_after_maria := initial_cards - maria_takes
  let cards_after_peter := cards_after_maria - 1
  let final_cards := cards_after_peter * 3
  final_cards = 18 :=
by
  sorry

end NUMINAMATH_GPT_baseball_card_problem_l224_22479


namespace NUMINAMATH_GPT_number_of_fence_panels_is_10_l224_22429

def metal_rods_per_sheet := 10
def metal_rods_per_beam := 4
def sheets_per_panel := 3
def beams_per_panel := 2
def total_metal_rods := 380

theorem number_of_fence_panels_is_10 :
  (total_metal_rods = 380) →
  (metal_rods_per_sheet = 10) →
  (metal_rods_per_beam = 4) →
  (sheets_per_panel = 3) →
  (beams_per_panel = 2) →
  380 / (3 * 10 + 2 * 4) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_fence_panels_is_10_l224_22429


namespace NUMINAMATH_GPT_number_of_subsets_l224_22439

theorem number_of_subsets (P : Finset ℤ) (h : P = {-1, 0, 1}) : P.powerset.card = 8 := 
by
  rw [h]
  sorry

end NUMINAMATH_GPT_number_of_subsets_l224_22439


namespace NUMINAMATH_GPT_find_sum_of_variables_l224_22403

variables (a b c d : ℤ)

theorem find_sum_of_variables
    (h1 : a - b + c = 7)
    (h2 : b - c + d = 8)
    (h3 : c - d + a = 4)
    (h4 : d - a + b = 3)
    (h5 : a + b + c - d = 10) :
    a + b + c + d = 16 := 
sorry

end NUMINAMATH_GPT_find_sum_of_variables_l224_22403


namespace NUMINAMATH_GPT_range_of_m_eq_l224_22404

theorem range_of_m_eq (m: ℝ) (x: ℝ) :
  (m+1 = 0 ∧ 4 > 0) ∨ 
  ((m + 1 > 0) ∧ ((m^2 - 2 * m - 3)^2 - 4 * (m + 1) * (-m + 3) < 0)) ↔ 
  (m ∈ Set.Icc (-1 : ℝ) 1 ∪ Set.Ico (1 : ℝ) 3) := 
sorry

end NUMINAMATH_GPT_range_of_m_eq_l224_22404


namespace NUMINAMATH_GPT_miles_driven_on_tuesday_l224_22447

-- Define the conditions given in the problem
theorem miles_driven_on_tuesday (T : ℕ) (h_avg : (12 + T + 21) / 3 = 17) :
  T = 18 :=
by
  -- We state what we want to prove, but we leave the proof with sorry
  sorry

end NUMINAMATH_GPT_miles_driven_on_tuesday_l224_22447


namespace NUMINAMATH_GPT_monotonicity_of_f_l224_22420

noncomputable def f (a x : ℝ) : ℝ := (a * x) / (x + 1)

theorem monotonicity_of_f (a : ℝ) :
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → 0 < a → f a x1 < f a x2) ∧
  (∀ x1 x2 : ℝ, -1 < x1 → -1 < x2 → x1 < x2 → a < 0 → f a x1 > f a x2) :=
by {
  sorry
}

end NUMINAMATH_GPT_monotonicity_of_f_l224_22420


namespace NUMINAMATH_GPT_average_other_marbles_l224_22478

def total_marbles : ℕ := 10 -- Define a hypothetical total number for computation
def clear_marbles : ℕ := total_marbles * 40 / 100
def black_marbles : ℕ := total_marbles * 20 / 100
def other_marbles : ℕ := total_marbles - clear_marbles - black_marbles
def marbles_taken : ℕ := 5

theorem average_other_marbles :
  marbles_taken * other_marbles / total_marbles = 2 := by
  sorry

end NUMINAMATH_GPT_average_other_marbles_l224_22478


namespace NUMINAMATH_GPT_water_for_1200ml_flour_l224_22476

-- Define the condition of how much water is mixed with a specific amount of flour
def water_per_flour (flour water : ℕ) : Prop :=
  water = (flour / 400) * 100

-- Given condition: Maria uses 100 mL of water for every 400 mL of flour
def condition : Prop := water_per_flour 400 100

-- Problem Statement: How many mL of water for 1200 mL of flour?
theorem water_for_1200ml_flour (h : condition) : water_per_flour 1200 300 :=
sorry

end NUMINAMATH_GPT_water_for_1200ml_flour_l224_22476


namespace NUMINAMATH_GPT_system_solution_find_a_l224_22460

theorem system_solution (x y : ℝ) (a : ℝ) :
  (|16 + 6 * x - x ^ 2 - y ^ 2| + |6 * x| = 16 + 12 * x - x ^ 2 - y ^ 2)
  ∧ ((a + 15) * y + 15 * x - a = 0) →
  ( (x - 3) ^ 2 + y ^ 2 ≤ 25 ∧ x ≥ 0 ) :=
sorry

theorem find_a (a : ℝ) :
  ∃ (x y : ℝ), 
  ((a + 15) * y + 15 * x - a = 0 ∧ x ≥ 0 ∧ (x - 3) ^ 2 + y ^ 2 ≤ 25) ↔ 
  (a = -20 ∨ a = -12) :=
sorry

end NUMINAMATH_GPT_system_solution_find_a_l224_22460


namespace NUMINAMATH_GPT_count_points_l224_22451

theorem count_points (a b : ℝ) :
  (abs b = 2) ∧ (abs a = 4) → (∃ (P : ℝ × ℝ), P = (a, b) ∧ (abs b = 2) ∧ (abs a = 4) ∧
    ((a = 4 ∨ a = -4) ∧ (b = 2 ∨ b = -2)) ∧
    (P = (4, 2) ∨ P = (4, -2) ∨ P = (-4, 2) ∨ P = (-4, -2)) ∧
    ∃ n, n = 4) :=
sorry

end NUMINAMATH_GPT_count_points_l224_22451


namespace NUMINAMATH_GPT_distinct_paths_from_C_to_D_l224_22480

-- Definitions based on conditions
def grid_rows : ℕ := 7
def grid_columns : ℕ := 8
def total_steps : ℕ := grid_rows + grid_columns -- 15 in this case
def steps_right : ℕ := grid_columns -- 8 in this case

-- Theorem statement
theorem distinct_paths_from_C_to_D :
  Nat.choose total_steps steps_right = 6435 :=
by
  -- The proof itself
  sorry

end NUMINAMATH_GPT_distinct_paths_from_C_to_D_l224_22480


namespace NUMINAMATH_GPT_seashells_left_l224_22401

-- Definitions based on conditions
def initial_seashells : ℕ := 35
def seashells_given_away : ℕ := 18

-- Theorem stating the proof problem
theorem seashells_left (initial_seashells seashells_given_away : ℕ) : initial_seashells - seashells_given_away = 17 := 
    by
        sorry

end NUMINAMATH_GPT_seashells_left_l224_22401


namespace NUMINAMATH_GPT_average_of_all_results_is_24_l224_22441

-- Definitions translated from conditions
def average_1 := 20
def average_2 := 30
def n1 := 30
def n2 := 20
def total_sum_1 := n1 * average_1
def total_sum_2 := n2 * average_2

-- Lean 4 statement
theorem average_of_all_results_is_24
  (h1 : total_sum_1 = n1 * average_1)
  (h2 : total_sum_2 = n2 * average_2) :
  ((total_sum_1 + total_sum_2) / (n1 + n2) = 24) :=
by
  sorry

end NUMINAMATH_GPT_average_of_all_results_is_24_l224_22441


namespace NUMINAMATH_GPT_flat_rate_first_night_l224_22493

theorem flat_rate_first_night
  (f n : ℚ)
  (h1 : f + 3 * n = 210)
  (h2 : f + 6 * n = 350)
  : f = 70 :=
by
  sorry

end NUMINAMATH_GPT_flat_rate_first_night_l224_22493


namespace NUMINAMATH_GPT_pasta_sauce_cost_l224_22453

theorem pasta_sauce_cost :
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  pasta_sauce_cost = 5 :=
by
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  sorry

end NUMINAMATH_GPT_pasta_sauce_cost_l224_22453


namespace NUMINAMATH_GPT_speed_of_second_person_l224_22426

-- Definitions based on the conditions
def speed_person1 := 70 -- km/hr
def distance_AB := 600 -- km

def time_traveled := 4 -- hours (from 10 am to 2 pm)

-- The goal is to prove that the speed of the second person is 80 km/hr
theorem speed_of_second_person :
  (distance_AB - speed_person1 * time_traveled) / time_traveled = 80 := 
by 
  sorry

end NUMINAMATH_GPT_speed_of_second_person_l224_22426


namespace NUMINAMATH_GPT_no_club_member_is_fraternity_member_l224_22410

variable (Student : Type) (isHonest : Student → Prop) 
                       (isFraternityMember : Student → Prop) 
                       (isClubMember : Student → Prop)

axiom some_students_not_honest : ∃ x : Student, ¬ isHonest x
axiom all_frats_honest : ∀ y : Student, isFraternityMember y → isHonest y
axiom no_clubs_honest : ∀ z : Student, isClubMember z → ¬ isHonest z

theorem no_club_member_is_fraternity_member : ∀ w : Student, isClubMember w → ¬ isFraternityMember w :=
by sorry

end NUMINAMATH_GPT_no_club_member_is_fraternity_member_l224_22410


namespace NUMINAMATH_GPT_derivatives_at_zero_l224_22488

noncomputable def f : ℝ → ℝ := sorry

axiom diff_f : ∀ n : ℕ, f (1 / (n + 1)) = (n + 1)^2 / ((n + 1)^2 + 1)

theorem derivatives_at_zero :
  f 0 = 1 ∧ 
  deriv f 0 = 0 ∧ 
  deriv (deriv f) 0 = -2 ∧ 
  ∀ k : ℕ, k ≥ 3 → deriv^[k] f 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_derivatives_at_zero_l224_22488


namespace NUMINAMATH_GPT_not_odd_iff_exists_ne_l224_22494

open Function

variable {f : ℝ → ℝ}

theorem not_odd_iff_exists_ne : (∃ x : ℝ, f (-x) ≠ -f x) ↔ ¬ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end NUMINAMATH_GPT_not_odd_iff_exists_ne_l224_22494


namespace NUMINAMATH_GPT_distance_traveled_on_foot_l224_22417

theorem distance_traveled_on_foot (x y : ℝ) (h1 : x + y = 80) (h2 : x / 8 + y / 16 = 7) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_on_foot_l224_22417


namespace NUMINAMATH_GPT_solve_equation_l224_22446

theorem solve_equation (x : ℝ) : 2 * x - 1 = 3 * x + 3 → x = -4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_equation_l224_22446


namespace NUMINAMATH_GPT_isosceles_triangle_area_l224_22492

theorem isosceles_triangle_area {a b h : ℝ} (h1 : a = 13) (h2 : b = 13) (h3 : h = 10) :
  ∃ (A : ℝ), A = 60 ∧ A = (1 / 2) * h * 12 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l224_22492


namespace NUMINAMATH_GPT_probability_five_chords_form_convex_pentagon_l224_22463

-- Definitions of problem conditions
variable (n : ℕ) (k : ℕ)

-- Eight points on a circle
def points_on_circle : ℕ := 8

-- Number of chords selected
def selected_chords : ℕ := 5

-- Total number of ways to select 5 chords from 28 possible chords
def total_ways : ℕ := Nat.choose 28 5

-- Number of ways to select 5 points from 8, forming a convex pentagon
def favorable_ways : ℕ := Nat.choose 8 5

-- The probability computation
def probability_pentagon (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_five_chords_form_convex_pentagon :
  probability_pentagon total_ways favorable_ways = 1 / 1755 :=
by
  sorry

end NUMINAMATH_GPT_probability_five_chords_form_convex_pentagon_l224_22463


namespace NUMINAMATH_GPT_pregnant_fish_in_each_tank_l224_22473

/-- Mark has 3 tanks for pregnant fish. Each tank has a certain number of pregnant fish and each fish
gives birth to 20 young. Mark has 240 young fish at the end. Prove that there are 4 pregnant fish in
each tank. -/
theorem pregnant_fish_in_each_tank (x : ℕ) (h1 : 3 * 20 * x = 240) : x = 4 := by
  sorry

end NUMINAMATH_GPT_pregnant_fish_in_each_tank_l224_22473


namespace NUMINAMATH_GPT_shift_upwards_l224_22487

theorem shift_upwards (a : ℝ) :
  (∀ x : ℝ, y = -2 * x + a) -> (a = 1) :=
by
  sorry

end NUMINAMATH_GPT_shift_upwards_l224_22487


namespace NUMINAMATH_GPT_find_real_solution_to_given_equation_l224_22496

noncomputable def sqrt_96_minus_sqrt_84 : ℝ := Real.sqrt 96 - Real.sqrt 84

theorem find_real_solution_to_given_equation (x : ℝ) (hx : x + 4 ≥ 0) :
  x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 60 ↔ x = sqrt_96_minus_sqrt_84 := 
by
  sorry

end NUMINAMATH_GPT_find_real_solution_to_given_equation_l224_22496


namespace NUMINAMATH_GPT_manufacturing_sector_angle_l224_22498

theorem manufacturing_sector_angle (h1 : 50 ≤ 100) (h2 : 360 = 4 * 90) : 0.50 * 360 = 180 := 
by
  sorry

end NUMINAMATH_GPT_manufacturing_sector_angle_l224_22498


namespace NUMINAMATH_GPT_find_a_l224_22405

-- Define the main inequality condition
def inequality_condition (a x : ℝ) : Prop := |x^2 + a * x + 4 * a| ≤ 3

-- Define the condition that there is exactly one solution to the inequality
def has_exactly_one_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (inequality_condition a x) ∧ (∀ y : ℝ, x ≠ y → ¬(inequality_condition a y))

-- The theorem that states the specific values of a
theorem find_a (a : ℝ) : has_exactly_one_solution a ↔ a = 8 + 2 * Real.sqrt 13 ∨ a = 8 - 2 * Real.sqrt 13 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l224_22405
