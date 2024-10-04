import Mathlib

namespace value_of_w_over_y_l16_16072

theorem value_of_w_over_y (w x y : ℝ) (h1 : w / x = 1 / 3) (h2 : (x + y) / y = 3) : w / y = 2 / 3 :=
by
  sorry

end value_of_w_over_y_l16_16072


namespace max_buses_in_city_l16_16639

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l16_16639


namespace pure_imaginary_complex_solution_l16_16380

theorem pure_imaginary_complex_solution (a : Real) :
  (a ^ 2 - 1 = 0) ∧ ((a - 1) ≠ 0) → a = -1 := by
  sorry

end pure_imaginary_complex_solution_l16_16380


namespace sum_of_n_terms_l16_16283

noncomputable def S : ℕ → ℕ :=
sorry -- We define S, but its exact form is not used in the statement directly

noncomputable def a : ℕ → ℕ := 
sorry -- We define a, but its exact form is not used in the statement directly

-- Conditions
axiom S3_eq : S 3 = 1
axiom a_rec : ∀ n : ℕ, 0 < n → a (n + 3) = 2 * (a n)

-- Proof problem
theorem sum_of_n_terms : S 2019 = 2^673 - 1 :=
sorry

end sum_of_n_terms_l16_16283


namespace arithmetic_mean_two_digit_multiples_of_8_l16_16860

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let a := 16
  let l := 96
  let d := 8
  let n := (l - a) / d + 1
  let mean := (a + l) / 2
  mean = 56 :=
by
  sorry

end arithmetic_mean_two_digit_multiples_of_8_l16_16860


namespace cubic_sum_l16_16083

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := 
sorry

end cubic_sum_l16_16083


namespace school_choir_robe_cost_l16_16323

theorem school_choir_robe_cost :
  ∀ (total_robes_needed current_robes cost_per_robe : ℕ), 
  total_robes_needed = 30 → 
  current_robes = 12 → 
  cost_per_robe = 2 → 
  (total_robes_needed - current_robes) * cost_per_robe = 36 :=
by
  intros total_robes_needed current_robes cost_per_robe h1 h2 h3
  sorry

end school_choir_robe_cost_l16_16323


namespace find_speed_l16_16750

variables (x : ℝ) (V : ℝ)

def initial_speed (x : ℝ) (V : ℝ) : Prop := 
  let time_initial := x / V
  let time_second := (2 * x) / 20
  let total_distance := 3 * x
  let average_speed := 26.25
  average_speed = total_distance / (time_initial + time_second)

theorem find_speed (x : ℝ) (h : initial_speed x V) : V = 70 :=
by sorry

end find_speed_l16_16750


namespace insulation_cost_of_rectangular_tank_l16_16322

theorem insulation_cost_of_rectangular_tank
  (l w h cost_per_sq_ft : ℕ)
  (hl : l = 4) (hw : w = 5) (hh : h = 3) (hc : cost_per_sq_ft = 20) :
  2 * l * w + 2 * l * h + 2 * w * h * 20 = 1880 :=
by
  sorry

end insulation_cost_of_rectangular_tank_l16_16322


namespace ages_total_l16_16459

variable (A B C : ℕ)

theorem ages_total (h1 : A = B + 2) (h2 : B = 2 * C) (h3 : B = 10) : A + B + C = 27 :=
by
  sorry

end ages_total_l16_16459


namespace natalies_diaries_l16_16823

theorem natalies_diaries : 
  ∀ (initial_diaries : ℕ) (tripled_diaries : ℕ) (total_diaries : ℕ) (lost_diaries : ℕ) (remaining_diaries : ℕ),
  initial_diaries = 15 →
  tripled_diaries = 3 * initial_diaries →
  total_diaries = initial_diaries + tripled_diaries →
  lost_diaries = 3 * total_diaries / 5 →
  remaining_diaries = total_diaries - lost_diaries →
  remaining_diaries = 24 :=
by
  intros initial_diaries tripled_diaries total_diaries lost_diaries remaining_diaries
  intro h1 h2 h3 h4 h5
  sorry

end natalies_diaries_l16_16823


namespace purely_imaginary_satisfies_condition_l16_16065

theorem purely_imaginary_satisfies_condition (m : ℝ) (h1 : m^2 + 3 * m - 4 = 0) (h2 : m + 4 ≠ 0) : m = 1 :=
by
  sorry

end purely_imaginary_satisfies_condition_l16_16065


namespace alma_score_l16_16841

variables (A M S : ℕ)

-- Given conditions
axiom h1 : M = 60
axiom h2 : M = 3 * A
axiom h3 : A + M = 2 * S

theorem alma_score : S = 40 :=
by
  -- proof goes here
  sorry

end alma_score_l16_16841


namespace kelly_harvested_pounds_l16_16098

def total_carrots (bed1 bed2 bed3 : ℕ) : ℕ :=
  bed1 + bed2 + bed3

def total_weight (total : ℕ) (carrots_per_pound : ℕ) : ℕ :=
  total / carrots_per_pound

theorem kelly_harvested_pounds :
  total_carrots 55 101 78 = 234 ∧ total_weight 234 6 = 39 :=
by {
  split,
  { exact rfl }, -- 234 = 234
  { exact rfl }  -- 234 / 6 = 39
}

end kelly_harvested_pounds_l16_16098


namespace caleb_ice_cream_l16_16043

theorem caleb_ice_cream (x : ℕ) (hx1 : ∃ x, x ≥ 0) (hx2 : 4 * x - 36 = 4) : x = 10 :=
by {
  sorry
}

end caleb_ice_cream_l16_16043


namespace fg_of_3_l16_16206

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem fg_of_3 : f (g 3) = 25 := by
  sorry

end fg_of_3_l16_16206


namespace distinct_solutions_l16_16219

theorem distinct_solutions : 
  ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1 - 7| = 2 * |x1 + 1| + |x1 - 3| ∧ |x2 - 7| = 2 * |x2 + 1| + |x2 - 3|) := 
by
  sorry

end distinct_solutions_l16_16219


namespace sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l16_16497

variable (θ : ℝ)

theorem sin_theta_plus_2pi_div_3 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.sin (θ + 2 * Real.pi / 3) = -1 / 3 :=
  sorry

theorem cos_theta_minus_5pi_div_6 (h : Real.sin (θ - Real.pi / 3) = 1 / 3) :
  Real.cos (θ - 5 * Real.pi / 6) = 1 / 3 :=
  sorry

end sin_theta_plus_2pi_div_3_cos_theta_minus_5pi_div_6_l16_16497


namespace solveAdultsMonday_l16_16529

def numAdultsMonday (A : ℕ) : Prop :=
  let childrenMondayCost := 7 * 3
  let childrenTuesdayCost := 4 * 3
  let adultsTuesdayCost := 2 * 4
  let totalChildrenCost := childrenMondayCost + childrenTuesdayCost
  let totalAdultsCost := A * 4 + adultsTuesdayCost
  let totalRevenue := totalChildrenCost + totalAdultsCost
  totalRevenue = 61

theorem solveAdultsMonday : numAdultsMonday 5 := 
  by 
    -- Proof goes here
    sorry

end solveAdultsMonday_l16_16529


namespace find_f_log_3_54_l16_16178

noncomputable def f : ℝ → ℝ := sorry  -- Since we have to define a function and we do not need the exact implementation.

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_property : ∀ x : ℝ, f (x + 2) = - 1 / f x
axiom interval_property : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 3 ^ x

theorem find_f_log_3_54 : f (Real.log 54 / Real.log 3) = -3 / 2 :=
by
  sorry


end find_f_log_3_54_l16_16178


namespace sum_integer_solutions_in_interval_l16_16691

theorem sum_integer_solutions_in_interval :
  (∑ x in (set.Icc (-25 : ℤ) (25 : ℤ)) \ {x : ℤ | (x^2 + x - 56).sqrt - (x^2 + 25*x + 136).sqrt < 8 * ((x - 7) / (x + 8)).sqrt}, (x : ℤ)).sum = 267 :=
by
  sorry

end sum_integer_solutions_in_interval_l16_16691


namespace price_of_baseball_cards_l16_16394

theorem price_of_baseball_cards 
    (packs_Digimon : ℕ)
    (price_per_pack : ℝ)
    (total_spent : ℝ)
    (total_cost_Digimon : ℝ) 
    (price_baseball_deck : ℝ) 
    (h1 : packs_Digimon = 4) 
    (h2 : price_per_pack = 4.45) 
    (h3 : total_spent = 23.86) 
    (h4 : total_cost_Digimon = packs_Digimon * price_per_pack) 
    (h5 : price_baseball_deck = total_spent - total_cost_Digimon) : 
    price_baseball_deck = 6.06 :=
sorry

end price_of_baseball_cards_l16_16394


namespace isosceles_triangle_perimeter_l16_16806

-- Define the sides of the isosceles triangle
def side1 : ℝ := 4
def side2 : ℝ := 8

-- Hypothesis: The perimeter of an isosceles triangle with the given sides
-- Given condition
def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = side1 ∨ a = side2) (h2 : b = side1 ∨ b = side2) :
  ∃ p : ℝ, is_isosceles_triangle a b side2 ∧ p = a + b + side2 → p = 20 :=
sorry

end isosceles_triangle_perimeter_l16_16806


namespace minimum_n_value_l16_16739

-- Define a multiple condition
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

-- Given conditions
def conditions (n : ℕ) : Prop := 
  (n ≥ 8) ∧ is_multiple 4 n ∧ is_multiple 8 n

-- Lean theorem statement for the problem
theorem minimum_n_value (n : ℕ) (h : conditions n) : n = 8 :=
  sorry

end minimum_n_value_l16_16739


namespace y_squared_in_range_l16_16082

theorem y_squared_in_range (y : ℝ) 
  (h : (Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2)) :
  270 ≤ y^2 ∧ y^2 ≤ 280 :=
sorry

end y_squared_in_range_l16_16082


namespace minimum_value_OC_l16_16618

variables (OA OB OC : ℝ × ℝ × ℝ)
variables (θ : ℝ)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2 + v.3^2)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

axiom condition1 : magnitude OA = 3
axiom condition2 : magnitude OB = 4
axiom condition3 : dot_product OA OB = 0
axiom condition4 : OC = (real.sin θ)^2 • OA + (real.cos θ)^2 • OB

theorem minimum_value_OC : 
  OC = ((16 : ℝ)/25) • OA + ((9 : ℝ)/25) • OB :=
sorry

end minimum_value_OC_l16_16618


namespace central_angle_of_sector_l16_16932

variable (A : ℝ) (r : ℝ) (α : ℝ)

-- Given conditions: A is the area of the sector, and r is the radius.
def is_sector (A : ℝ) (r : ℝ) (α : ℝ) : Prop :=
  A = (1 / 2) * α * r^2

-- Proof that the central angle α given the conditions is 3π/4.
theorem central_angle_of_sector (h1 : is_sector (3 * Real.pi / 8) 1 α) : 
  α = 3 * Real.pi / 4 := 
  sorry

end central_angle_of_sector_l16_16932


namespace smallest_prime_square_mod_six_l16_16636

theorem smallest_prime_square_mod_six (p : ℕ) (h_prime : Nat.Prime p) (h_mod : p^2 % 6 = 1) : p = 5 :=
sorry

end smallest_prime_square_mod_six_l16_16636


namespace tunnel_length_l16_16893

noncomputable def train_speed_mph : ℝ := 75
noncomputable def train_length_miles : ℝ := 1 / 4
noncomputable def passing_time_minutes : ℝ := 3

theorem tunnel_length :
  let speed_mpm := train_speed_mph / 60
  let total_distance_traveled := speed_mpm * passing_time_minutes
  let tunnel_length := total_distance_traveled - train_length_miles
  tunnel_length = 3.5 :=
by
  sorry

end tunnel_length_l16_16893


namespace alicia_total_payment_l16_16333

def daily_rent_cost : ℕ := 30
def miles_cost_per_mile : ℝ := 0.25
def rental_days : ℕ := 5
def driven_miles : ℕ := 500

def total_cost (daily_rent_cost : ℕ) (rental_days : ℕ)
               (miles_cost_per_mile : ℝ) (driven_miles : ℕ) : ℝ :=
  (daily_rent_cost * rental_days) + (miles_cost_per_mile * driven_miles)

theorem alicia_total_payment :
  total_cost daily_rent_cost rental_days miles_cost_per_mile driven_miles = 275 := by
  sorry

end alicia_total_payment_l16_16333


namespace initial_bananas_tree_l16_16020

-- Definitions for the conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten_by_raj : ℕ := 70
def bananas_in_basket_of_raj := 2 * bananas_eaten_by_raj
def bananas_cut_from_tree := bananas_eaten_by_raj + bananas_in_basket_of_raj
def initial_bananas_on_tree := bananas_cut_from_tree + bananas_left_on_tree

-- The theorem to be proven
theorem initial_bananas_tree : initial_bananas_on_tree = 310 :=
by sorry

end initial_bananas_tree_l16_16020


namespace log_expression_simplification_l16_16484

open Real

noncomputable def log_expr (a b c d x y z : ℝ) : ℝ :=
  log (a^2 / b) + log (b^2 / c) + log (c^2 / d) - log (a^2 * y * z / (d^2 * x))

theorem log_expression_simplification (a b c d x y z : ℝ) (h : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : d ≠ 0) (h5 : x ≠ 0) (h6 : y ≠ 0) (h7 : z ≠ 0) :
  log_expr a b c d x y z = log (bdx / yz) :=
by
  -- Proof goes here
  sorry

end log_expression_simplification_l16_16484


namespace phone_numbers_count_l16_16464

theorem phone_numbers_count : (2^5 = 32) :=
by sorry

end phone_numbers_count_l16_16464


namespace slope_of_CD_l16_16416

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10 * x - 2 * y + 40 = 0

-- Theorem statement
theorem slope_of_CD :
  ∃ C D : ℝ × ℝ,
    (circle1 C.1 C.2) ∧ (circle2 C.1 C.2) ∧ (circle1 D.1 D.2) ∧ (circle2 D.1 D.2) ∧
    (∃ m : ℝ, m = -2 / 3) := 
  sorry

end slope_of_CD_l16_16416


namespace machine_loan_repaid_in_5_months_l16_16179

theorem machine_loan_repaid_in_5_months :
  ∀ (loan cost selling_price tax_percentage products_per_month profit_per_product months : ℕ),
    loan = 22000 →
    cost = 5 →
    selling_price = 8 →
    tax_percentage = 10 →
    products_per_month = 2000 →
    profit_per_product = (selling_price - cost - (selling_price * tax_percentage / 100)) →
    (products_per_month * months * profit_per_product) ≥ loan →
    months = 5 :=
by
  intros loan cost selling_price tax_percentage products_per_month profit_per_product months
  sorry

end machine_loan_repaid_in_5_months_l16_16179


namespace range_of_a_plus_abs_b_l16_16616

theorem range_of_a_plus_abs_b (a b : ℝ)
  (h1 : -1 ≤ a) (h2 : a ≤ 3)
  (h3 : -5 < b) (h4 : b < 3) :
  -1 ≤ a + |b| ∧ a + |b| < 8 := by
sorry

end range_of_a_plus_abs_b_l16_16616


namespace interest_earned_l16_16102

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (I : ℝ):
  P = 1500 → r = 0.12 → n = 4 →
  A = compound_interest P r n →
  I = A - P →
  I = 862.2 :=
by
  intros hP hr hn hA hI
  sorry

end interest_earned_l16_16102


namespace cos_30_eq_sqrt3_div_2_l16_16759

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l16_16759


namespace sum_squares_condition_l16_16383

theorem sum_squares_condition
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 75)
  (h2 : ab + bc + ca = 40)
  (h3 : c = 5) :
  a + b + c = 5 * Real.sqrt 62 :=
by sorry

end sum_squares_condition_l16_16383


namespace empty_drainpipe_rate_l16_16826

theorem empty_drainpipe_rate :
  (∀ x : ℝ, (1/5 + 1/4 - 1/x = 1/2.5) → x = 20) :=
by 
    intro x
    intro h
    sorry -- Proof is omitted, only the statement is required

end empty_drainpipe_rate_l16_16826


namespace maximum_buses_l16_16643

-- Definition of the problem constraints as conditions in Lean
variables (bus : Type) (stop : Type) (buses : set bus) (stops : set stop)
variables (stops_served_by_bus : bus → finset stop)
variables (stop_occurrences : stop → finset bus)

-- Conditions
def has_9_stops (stops : set stop) : Prop := stops.card = 9
def bus_serves_3_stops (b : bus) : Prop := (stops_served_by_bus b).card = 3
def at_most_1_common_stop (b1 b2 : bus) (h1 : b1 ∈ buses) (h2 : b2 ∈ buses) : Prop :=
  (stops_served_by_bus b1 ∩ stops_served_by_bus b2).card ≤ 1

-- Goal
theorem maximum_buses (h1 : has_9_stops stops)
                      (h2 : ∀ b ∈ buses, bus_serves_3_stops b)
                      (h3 : ∀ b1 b2 ∈ buses, at_most_1_common_stop b1 b2 h1 h2) : 
  buses.card ≤ 12 :=
sorry

end maximum_buses_l16_16643


namespace sarah_interviewed_students_l16_16688

theorem sarah_interviewed_students :
  let oranges := 70
  let pears := 120
  let apples := 147
  let strawberries := 113
  oranges + pears + apples + strawberries = 450 := by
sorry

end sarah_interviewed_students_l16_16688


namespace ordering_of_four_numbers_l16_16612

variable (m n α β : ℝ)
variable (h1 : m < n)
variable (h2 : α < β)
variable (h3 : 2 * (α - m) * (α - n) - 7 = 0)
variable (h4 : 2 * (β - m) * (β - n) - 7 = 0)

theorem ordering_of_four_numbers : α < m ∧ m < n ∧ n < β :=
by
  sorry

end ordering_of_four_numbers_l16_16612


namespace binomial_identity_sum_binomial_coeff_binomial_expectation_l16_16567

-- Definition of binomial coefficient
def binomial_coeff (n r : ℕ) : ℕ := Nat.choose n r

-- 1) Prove that rC_n^r = nC_{n-1}^{r-1}
theorem binomial_identity {n r : ℕ} (hr : 1 ≤ r) : r * binomial_coeff n r = n * binomial_coeff (n - 1) (r - 1) := sorry

-- 2) Prove the sum C_n^1 + 2C_n^2 + 3C_n^3 + ... + nC_n^n
theorem sum_binomial_coeff (n : ℕ) : (Finset.range n).sum (λ r, (r+1) * binomial_coeff n (r+1)) = n * 2^(n-1) := sorry

-- Definition of binomial distribution for random variable
noncomputable def Binomial (n : ℕ) (p : ℝ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.ofFintype (λ k, Nat.choose n k * p^k * (1 - p)^(n - k))

-- 3) Prove that for a random variable X ∼ B(n,p), E(X) = np
theorem binomial_expectation {n : ℕ} {p : ℝ} (hp : 0 ≤ p ∧ p ≤ 1) : 
  (ProbabilityMassFunction.ofFintype (λ k, (↑k : ℝ) * (Binomial n p).pdf k)).support.sum = n * p := sorry

end binomial_identity_sum_binomial_coeff_binomial_expectation_l16_16567


namespace min_value_expression_l16_16874

theorem min_value_expression : ∃ (x y : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 8*x + 6*y + 25 ≥ 0) ∧ (∀ (x y : ℝ), x = 4 ∧ y = -3 → x^2 + y^2 - 8*x + 6*y + 25 = 0) :=
sorry

end min_value_expression_l16_16874


namespace test_group_type_A_probability_atleast_one_type_A_group_probability_l16_16450

noncomputable def probability_type_A_group : ℝ :=
  let pA := 2 / 3
  let pB := 1 / 2
  let P_A1 := 2 * (1 - pA) * pA
  let P_A2 := pA * pA
  let P_B0 := (1 - pB) * (1 - pB)
  let P_B1 := 2 * (1 - pB) * pB
  P_B0 * P_A1 + P_B0 * P_A2 + P_B1 * P_A2

theorem test_group_type_A_probability :
  probability_type_A_group = 4 / 9 :=
by
  sorry

noncomputable def at_least_one_type_A_in_3_groups : ℝ :=
  let P_type_A_group := 4 / 9
  1 - (1 - P_type_A_group) ^ 3

theorem atleast_one_type_A_group_probability :
  at_least_one_type_A_in_3_groups = 604 / 729 :=
by
  sorry

end test_group_type_A_probability_atleast_one_type_A_group_probability_l16_16450


namespace vector_decomposition_l16_16014

noncomputable def x : ℝ × ℝ × ℝ := (5, 15, 0)
noncomputable def p : ℝ × ℝ × ℝ := (1, 0, 5)
noncomputable def q : ℝ × ℝ × ℝ := (-1, 3, 2)
noncomputable def r : ℝ × ℝ × ℝ := (0, -1, 1)

theorem vector_decomposition : x = (4 : ℝ) • p + (-1 : ℝ) • q + (-18 : ℝ) • r :=
by
  sorry

end vector_decomposition_l16_16014


namespace unknown_cube_edge_length_l16_16984

theorem unknown_cube_edge_length (a b c x : ℕ) (h_a : a = 6) (h_b : b = 10) (h_c : c = 12) : a^3 + b^3 + x^3 = c^3 → x = 8 :=
by
  sorry

end unknown_cube_edge_length_l16_16984


namespace investment_of_c_l16_16734

theorem investment_of_c (P_b P_a P_c C_a C_b C_c : ℝ)
  (h1 : P_b = 2000) 
  (h2 : P_a - P_c = 799.9999999999998)
  (h3 : C_a = 8000)
  (h4 : C_b = 10000)
  (h5 : P_b / C_b = P_a / C_a)
  (h6 : P_c / C_c = P_a / C_a)
  : C_c = 4000 :=
by 
  sorry

end investment_of_c_l16_16734


namespace equilateral_triangle_area_decrease_l16_16471

theorem equilateral_triangle_area_decrease (A : ℝ) (A' : ℝ) (s s' : ℝ) 
  (h1 : A = 121 * Real.sqrt 3) 
  (h2 : A = (s^2 * Real.sqrt 3) / 4) 
  (h3 : s' = s - 8) 
  (h4 : A' = (s'^2 * Real.sqrt 3) / 4) :
  A - A' = 72 * Real.sqrt 3 := 
by sorry

end equilateral_triangle_area_decrease_l16_16471


namespace no_linear_term_l16_16087

theorem no_linear_term (m : ℤ) : (∀ (x : ℤ), (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 - 8*m → (8 + m) = 0) → m = -8 :=
by
  sorry

end no_linear_term_l16_16087


namespace tony_walking_speed_l16_16530

-- Define the conditions as hypotheses
def walking_speed_on_weekend (W : ℝ) : Prop := 
  let store_distance := 4 
  let run_speed := 10
  let day1_time := store_distance / W
  let day2_time := store_distance / run_speed
  let day3_time := store_distance / run_speed
  let avg_time := (day1_time + day2_time + day3_time) / 3
  avg_time = 56 / 60

-- State the theorem
theorem tony_walking_speed : ∃ W : ℝ, walking_speed_on_weekend W ∧ W = 2 := 
sorry

end tony_walking_speed_l16_16530


namespace winning_post_distance_l16_16563

theorem winning_post_distance (v_A v_B D : ℝ) (hvA : v_A = (5 / 3) * v_B) (head_start : 80 ≤ D) :
  (D / v_A = (D - 80) / v_B) → D = 200 :=
by
  sorry

end winning_post_distance_l16_16563


namespace minimize_distance_l16_16212

noncomputable def f (x : ℝ) := 9 * x^3
noncomputable def g (x : ℝ) := Real.log x

theorem minimize_distance :
  ∃ m > 0, (∀ x > 0, |f m - g m| ≤ |f x - g x|) ∧ m = 1/3 :=
sorry

end minimize_distance_l16_16212


namespace parabola_coefficients_l16_16271

theorem parabola_coefficients (a b c : ℝ) 
  (h_vertex : ∀ x, a * (x - 4) * (x - 4) + 3 = a * x * x + b * x + c) 
  (h_pass_point : 1 = a * (2 - 4) * (2 - 4) + 3) :
  (a = -1/2) ∧ (b = 4) ∧ (c = -5) :=
by
  sorry

end parabola_coefficients_l16_16271


namespace least_possible_integer_l16_16457

theorem least_possible_integer :
  ∃ N : ℕ,
    (∀ k, 1 ≤ k ∧ k ≤ 30 → k ≠ 24 → k ≠ 25 → N % k = 0) ∧
    (N % 24 ≠ 0) ∧
    (N % 25 ≠ 0) ∧
    N = 659375723440 :=
by
  sorry

end least_possible_integer_l16_16457


namespace cannot_form_square_with_sticks_l16_16718

theorem cannot_form_square_with_sticks
    (num_1cm_sticks : ℕ)
    (num_2cm_sticks : ℕ)
    (num_3cm_sticks : ℕ)
    (num_4cm_sticks : ℕ)
    (len_1cm_stick : ℕ)
    (len_2cm_stick : ℕ)
    (len_3cm_stick : ℕ)
    (len_4cm_stick : ℕ)
    (sum_lengths : ℕ) :
    num_1cm_sticks = 6 →
    num_2cm_sticks = 3 →
    num_3cm_sticks = 6 →
    num_4cm_sticks = 5 →
    len_1cm_stick = 1 →
    len_2cm_stick = 2 →
    len_3cm_stick = 3 →
    len_4cm_stick = 4 →
    sum_lengths = num_1cm_sticks * len_1cm_stick + 
                  num_2cm_sticks * len_2cm_stick + 
                  num_3cm_sticks * len_3cm_stick + 
                  num_4cm_sticks * len_4cm_stick →
    ∃ (s : ℕ), sum_lengths = 4 * s → False := 
by
  intros num_1cm_sticks_eq num_2cm_sticks_eq num_3cm_sticks_eq num_4cm_sticks_eq
         len_1cm_stick_eq len_2cm_stick_eq len_3cm_stick_eq len_4cm_stick_eq
         sum_lengths_def

  sorry

end cannot_form_square_with_sticks_l16_16718


namespace ceil_floor_diff_l16_16080

theorem ceil_floor_diff (x : ℝ) (h : ⌈x⌉ + ⌊x⌋ = 2 * x) : ⌈x⌉ - ⌊x⌋ = 1 := 
by 
  sorry

end ceil_floor_diff_l16_16080


namespace problem_l16_16633

theorem problem (x y z : ℕ) (h1 : xy + z = 56) (h2 : yz + x = 56) (h3 : zx + y = 56) : x + y + z = 21 :=
sorry

end problem_l16_16633


namespace mika_stickers_l16_16406

theorem mika_stickers 
    (initial_stickers : ℝ := 20.5)
    (bought_stickers : ℝ := 26.25)
    (birthday_stickers : ℝ := 19.75)
    (friend_stickers : ℝ := 7.5)
    (sister_stickers : ℝ := 6.3)
    (greeting_card_stickers : ℝ := 58.5)
    (yard_sale_stickers : ℝ := 3.2) :
    initial_stickers + bought_stickers + birthday_stickers + friend_stickers
    - sister_stickers - greeting_card_stickers - yard_sale_stickers = 6 := 
by
    sorry

end mika_stickers_l16_16406


namespace number_of_nurses_l16_16991

variables (D N : ℕ)

-- Condition: The total number of doctors and nurses is 250
def total_staff := D + N = 250

-- Condition: The ratio of doctors to nurses is 2 to 3
def ratio_doctors_to_nurses := D = (2 * N) / 3

-- Proof: The number of nurses is 150
theorem number_of_nurses (h1 : total_staff D N) (h2 : ratio_doctors_to_nurses D N) : N = 150 :=
sorry

end number_of_nurses_l16_16991


namespace decreasing_function_range_l16_16372

noncomputable def f (a x : ℝ) := a * (x^3) - x + 1

theorem decreasing_function_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ 0 := by
  sorry

end decreasing_function_range_l16_16372


namespace non_negative_dot_product_l16_16363

theorem non_negative_dot_product
  (a b c d e f g h : ℝ) :
  (a * c + b * d ≥ 0) ∨ (a * e + b * f ≥ 0) ∨ (a * g + b * h ≥ 0) ∨
  (c * e + d * f ≥ 0) ∨ (c * g + d * h ≥ 0) ∨ (e * g + f * h ≥ 0) :=
sorry

end non_negative_dot_product_l16_16363


namespace chess_group_players_l16_16993

theorem chess_group_players (n : ℕ) (H : n * (n - 1) / 2 = 435) : n = 30 :=
by
  sorry

end chess_group_players_l16_16993


namespace inscribed_circle_radius_l16_16185

theorem inscribed_circle_radius (r : ℝ) (radius : ℝ) (angle_deg : ℝ): 
  radius = 6 ∧ angle_deg = 120 ∧ (∀ θ : ℝ, θ = 60) → r = 3 := 
by
  sorry

end inscribed_circle_radius_l16_16185


namespace number_of_passed_boys_l16_16272

theorem number_of_passed_boys 
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 36 * 120) :
  P = 105 := 
sorry

end number_of_passed_boys_l16_16272


namespace nat_perfect_square_l16_16680

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l16_16680


namespace second_player_wins_l16_16519

theorem second_player_wins : 
  ∀ (a b c : ℝ), (a ≠ 0) → 
  (∃ (first_choice: ℝ), ∃ (second_choice: ℝ), 
    ∃ (third_choice: ℝ), 
    ((first_choice ≠ 0) → (b^2 + 4 * first_choice^2 > 0)) ∧ 
    ((first_choice = 0) → (b ≠ 0)) ∧ 
    first_choice * (first_choice * b + a) = 0 ↔ ∃ x : ℝ, a * x^2 + (first_choice + second_choice) * x + third_choice = 0) :=
by sorry

end second_player_wins_l16_16519


namespace remainder_of_B_is_4_l16_16555

theorem remainder_of_B_is_4 (A B : ℕ) (h : B = A * 9 + 13) : B % 9 = 4 :=
by {
  sorry
}

end remainder_of_B_is_4_l16_16555


namespace total_number_of_coins_l16_16556

-- Definitions and conditions
def num_coins_25c := 17
def num_coins_10c := 17

-- Statement to prove
theorem total_number_of_coins : num_coins_25c + num_coins_10c = 34 := by
  sorry

end total_number_of_coins_l16_16556


namespace average_annual_growth_rate_l16_16549

theorem average_annual_growth_rate (x : ℝ) (h : (1 + x)^2 = 1.20) : x < 0.1 :=
sorry

end average_annual_growth_rate_l16_16549


namespace otimes_calculation_l16_16904

def otimes (x y : ℝ) : ℝ := x^2 - 2*y

theorem otimes_calculation (k : ℝ) : otimes k (otimes k k) = -k^2 + 4*k :=
by
  sorry

end otimes_calculation_l16_16904


namespace roger_owes_correct_amount_l16_16975

def initial_house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_payment_percentage : ℝ := 0.30

def down_payment : ℝ := down_payment_percentage * initial_house_price
def remaining_after_down_payment : ℝ := initial_house_price - down_payment
def parents_payment : ℝ := parents_payment_percentage * remaining_after_down_payment
def money_owed : ℝ := remaining_after_down_payment - parents_payment

theorem roger_owes_correct_amount :
  money_owed = 56000 := by
  sorry

end roger_owes_correct_amount_l16_16975


namespace solve_system_of_equations_l16_16979

theorem solve_system_of_equations (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : 1 / (x * y) = x / z + 1)
  (h2 : 1 / (y * z) = y / x + 1)
  (h3 : 1 / (z * x) = z / y + 1) :
  x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2 :=
by
  sorry

end solve_system_of_equations_l16_16979


namespace sum_of_squares_of_solutions_l16_16606

theorem sum_of_squares_of_solutions :
  let a := (1 : ℝ) / 2010
  in (∀ x : ℝ, abs(x^2 - x + a) = a) → 
     ((0^2 + 1^2) + ((((1*1) - 2 * a) + 1) / a)) = 2008 / 1005 :=
by
  intros a h
  sorry

end sum_of_squares_of_solutions_l16_16606


namespace fraction_subtraction_l16_16595

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem fraction_subtraction : 
  (18 / 42 - 2 / 9) = (13 / 63) := 
by 
  sorry

end fraction_subtraction_l16_16595


namespace find_B_l16_16709

def is_prime_203B21 (B : ℕ) : Prop :=
  2 ≤ B ∧ B < 10 ∧ Prime (200000 + 3000 + 100 * B + 20 + 1)

theorem find_B : ∃ B, is_prime_203B21 B ∧ ∀ B', is_prime_203B21 B' → B' = 5 := by
  sorry

end find_B_l16_16709


namespace problem1_problem2_l16_16901

theorem problem1 :
  ( (1/2) ^ (-2) - 0.01 ^ (-1) + (-(1 + 1/7)) ^ (0)) = -95 := by
  sorry

theorem problem2 (x : ℝ) :
  (x - 2) * (x + 1) - (x - 1) ^ 2 = x - 3 := by
  sorry

end problem1_problem2_l16_16901


namespace greatest_value_of_n_l16_16449

theorem greatest_value_of_n : ∀ (n : ℤ), 102 * n^2 ≤ 8100 → n ≤ 8 :=
by 
  sorry

end greatest_value_of_n_l16_16449


namespace percent_blue_marbles_l16_16334

theorem percent_blue_marbles (total_items buttons red_marbles : ℝ) 
  (H1 : buttons = 0.30 * total_items)
  (H2 : red_marbles = 0.50 * (total_items - buttons)) :
  (total_items - buttons - red_marbles) / total_items = 0.35 :=
by 
  sorry

end percent_blue_marbles_l16_16334


namespace min_norm_l16_16935

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b : ℝ × ℝ := (1, 2)

def vec_add (λ : ℝ) : ℝ × ℝ := (a.1 + λ * b.1, a.2 + λ * b.2)
def vec_norm (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem min_norm : (λ : ℝ) -> (vec_norm (vec_add λ)) = minimal norm :=
  minimum = 3 * real.sqrt(5) / 5 :=
begin
  -- The proof will go here.
  sorry
end

end min_norm_l16_16935


namespace product_of_square_and_neighbor_is_divisible_by_12_l16_16409

theorem product_of_square_and_neighbor_is_divisible_by_12 (n : ℤ) : 12 ∣ (n^2 * (n - 1) * (n + 1)) :=
sorry

end product_of_square_and_neighbor_is_divisible_by_12_l16_16409


namespace exterior_angle_BAC_l16_16891

theorem exterior_angle_BAC (square_octagon_coplanar : Prop) (common_side_AD : Prop) : 
    angle_BAC = 135 :=
by
  sorry

end exterior_angle_BAC_l16_16891


namespace initial_pile_counts_l16_16430

def pile_transfers (A B C : ℕ) : Prop :=
  (A + B + C = 48) ∧
  ∃ (A' B' C' : ℕ), 
    (A' = A + B) ∧ (B' = B + C) ∧ (C' = C + A) ∧
    (A' = 2 * 16) ∧ (B' = 2 * 12) ∧ (C' = 2 * 14)

theorem initial_pile_counts :
  ∃ A B C : ℕ, pile_transfers A B C ∧ A = 22 ∧ B = 14 ∧ C = 12 :=
by
  sorry

end initial_pile_counts_l16_16430


namespace range_of_m_l16_16804

theorem range_of_m (x y m : ℝ) : (∃ (x y : ℝ), x + y^2 - x + y + m = 0) → m < 1/2 :=
by
  sorry

end range_of_m_l16_16804


namespace tank_empty_time_correct_l16_16181

noncomputable def tank_time_to_empty (leak_empty_time : ℕ) (inlet_rate : ℕ) (tank_capacity : ℕ) : ℕ :=
(tank_capacity / (tank_capacity / leak_empty_time - inlet_rate * 60))

theorem tank_empty_time_correct :
  tank_time_to_empty 6 3 4320 = 8 := by
  sorry

end tank_empty_time_correct_l16_16181


namespace inequality_example_l16_16631

theorem inequality_example (a b : ℝ) (h : a - b > 0) : a + 1 > b + 1 :=
sorry

end inequality_example_l16_16631


namespace total_amount_paid_l16_16245

-- Define the conditions
def chicken_nuggets_ordered : ℕ := 100
def nuggets_per_box : ℕ := 20
def cost_per_box : ℕ := 4

-- Define the hypothesis on the amount of money paid for the chicken nuggets
theorem total_amount_paid :
  (chicken_nuggets_ordered / nuggets_per_box) * cost_per_box = 20 :=
by
  sorry

end total_amount_paid_l16_16245


namespace percentage_entree_cost_l16_16722

-- Conditions
def total_spent : ℝ := 50.0
def num_appetizers : ℝ := 2
def cost_per_appetizer : ℝ := 5.0
def total_appetizer_cost : ℝ := num_appetizers * cost_per_appetizer
def total_entree_cost : ℝ := total_spent - total_appetizer_cost

-- Proof Problem
theorem percentage_entree_cost :
  (total_entree_cost / total_spent) * 100 = 80 :=
sorry

end percentage_entree_cost_l16_16722


namespace max_buses_in_city_l16_16640

theorem max_buses_in_city (n : ℕ) (stops : ℕ) (shared : ℕ) (condition1 : n = 9) (condition2 : stops = 3) (condition3 : shared ≤ 1) : n = 12 :=
sorry

end max_buses_in_city_l16_16640


namespace floor_multiple_of_floor_l16_16395

noncomputable def r : ℝ := sorry

theorem floor_multiple_of_floor (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : ∃ k, n = k * m) (hr : r ≥ 1) 
  (floor_multiple : ∀ (m n : ℕ), (∃ k : ℕ, n = k * m) → ∃ l, ⌊n * r⌋ = l * ⌊m * r⌋) :
  ∃ k : ℤ, r = k := 
sorry

end floor_multiple_of_floor_l16_16395


namespace no_triangle_100_sticks_yes_triangle_99_sticks_l16_16848

-- Definitions for the sums of lengths of sticks
def sum_lengths (n : ℕ) : ℕ := (n * (n + 1)) / 2

-- Conditions and questions for the problem
def is_divisible_by_3 (x : ℕ) : Prop := x % 3 = 0

-- Proof problem for n = 100
theorem no_triangle_100_sticks : ¬ (is_divisible_by_3 (sum_lengths 100)) := by
  sorry

-- Proof problem for n = 99
theorem yes_triangle_99_sticks : is_divisible_by_3 (sum_lengths 99) := by
  sorry

end no_triangle_100_sticks_yes_triangle_99_sticks_l16_16848


namespace number_of_friends_l16_16335

/- Define the conditions -/
def sandwiches_per_friend : Nat := 3
def total_sandwiches : Nat := 12

/- Define the mathematical statement to be proven -/
theorem number_of_friends : (total_sandwiches / sandwiches_per_friend) = 4 :=
by
  sorry

end number_of_friends_l16_16335


namespace percent_less_l16_16637

theorem percent_less (w u y z : ℝ) (P : ℝ) (hP : P = 0.40)
  (h1 : u = 0.60 * y)
  (h2 : z = 0.54 * y)
  (h3 : z = 1.50 * w) :
  w = (1 - P) * u := 
sorry

end percent_less_l16_16637


namespace smallest_number_h_divisible_8_11_24_l16_16010

theorem smallest_number_h_divisible_8_11_24 : 
  ∃ h : ℕ, (h + 5) % 8 = 0 ∧ (h + 5) % 11 = 0 ∧ (h + 5) % 24 = 0 ∧ h = 259 :=
by
  sorry

end smallest_number_h_divisible_8_11_24_l16_16010


namespace find_k_l16_16143

theorem find_k (k : ℝ) (h : ∀ x : ℝ, x^2 + 10 * x + k = 0 → (∃ a : ℝ, a > 0 ∧ (x = -3 * a ∨ x = -a))) :
  k = 18.75 :=
sorry

end find_k_l16_16143


namespace triangle_properties_proof_l16_16951

noncomputable def triangle_proporties : Prop :=
  ∀ (a b c : ℝ) (A B C : ℝ),
    b = sqrt 3 →
    c = 1 →
    B = 60 * real.pi / 180 →
    a = sqrt (b^2 + c^2) ∧
    A = 90 * real.pi / 180 ∧
    C = 30 * real.pi / 180

theorem triangle_properties_proof : triangle_proporties :=
  by {
    intros a b c A B C hb hc hB,
    have ha : a = sqrt (b^2 + c^2), by {
      dsimp only [hb, hc],
      norm_num,
    },
    have hA : A = 90 * real.pi / 180, by {
      obtain rfl : B = real.pi / 3 := by { exact hB, norm_num },
      obtain rfl : b = sqrt 3 := by { exact hb, norm_num },
      obtain rfl : c = 1 := by { exact hc, norm_num },
      calc
        A = π - (B + C)   : by sorry
        ... = π / 2       : by sorry,
    },
    have hC : C = 30 * real.pi / 180, by {
      obtain rfl : B = real.pi / 3 := by { exact hB, norm_num },
      obtain rfl : b = sqrt 3 := by { exact hb, norm_num },
      obtain rfl : c = 1 := by { exact hc, norm_num },
      calc
        C = real.asin (c * real.sin B / b) :
        ... = real.pi / 6 :
        sorry,
    },
    exact ⟨ha, hA, hC⟩,
  }

end triangle_properties_proof_l16_16951


namespace max_area_rectangle_l16_16687

theorem max_area_rectangle (p : ℝ) (a b : ℝ) (h : p = 2 * (a + b)) : 
  ∃ S : ℝ, S = a * b ∧ (∀ (a' b' : ℝ), p = 2 * (a' + b') → S ≥ a' * b') → a = b :=
by
  sorry

end max_area_rectangle_l16_16687


namespace number_of_pieces_from_rod_l16_16079

theorem number_of_pieces_from_rod (rod_length_m : ℕ) (piece_length_cm : ℕ) (meter_to_cm : ℕ) 
  (h1 : rod_length_m = 34) (h2 : piece_length_cm = 85) (h3 : meter_to_cm = 100) : 
  rod_length_m * meter_to_cm / piece_length_cm = 40 := by
  sorry

end number_of_pieces_from_rod_l16_16079


namespace max_PB_distance_l16_16397

noncomputable def ellipse : Set (ℝ × ℝ) :=
  { p | ∃ x y : ℝ, p = ⟨x, y⟩ ∧ x^2 / 5 + y^2 = 1 }

def B : ℝ × ℝ := (0, 1)

def PB_distance (θ : ℝ) : ℝ :=
  let P : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)
  Real.sqrt ((sqrt 5 * cos θ - 0)^2 + (sin θ - 1)^2)

theorem max_PB_distance : ∃ (θ : ℝ), θ ∈ Icc (0 : ℝ) (2 * Real.pi) ∧ PB_distance θ = 5 / 2 :=
by
  sorry

end max_PB_distance_l16_16397


namespace intersection_A_B_l16_16789

def A := {x : ℝ | 2 * x - 1 ≤ 0}
def B := {x : ℝ | 1 / x > 1}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1 / 2} :=
  sorry

end intersection_A_B_l16_16789


namespace candies_share_equally_l16_16968

theorem candies_share_equally (mark_candies : ℕ) (peter_candies : ℕ) (john_candies : ℕ)
  (h_mark : mark_candies = 30) (h_peter : peter_candies = 25) (h_john : john_candies = 35) :
  (mark_candies + peter_candies + john_candies) / 3 = 30 :=
by
  sorry

end candies_share_equally_l16_16968


namespace laborer_savings_l16_16981

theorem laborer_savings
  (monthly_expenditure_first6 : ℕ := 70)
  (monthly_expenditure_next4 : ℕ := 60)
  (monthly_income : ℕ := 69)
  (expenditure_first6 := 6 * monthly_expenditure_first6)
  (income_first6 := 6 * monthly_income)
  (debt : ℕ := expenditure_first6 - income_first6)
  (expenditure_next4 := 4 * monthly_expenditure_next4)
  (income_next4 := 4 * monthly_income)
  (savings : ℕ := income_next4 - (expenditure_next4 + debt)) :
  savings = 30 := 
by
  sorry

end laborer_savings_l16_16981


namespace sin_sum_square_gt_sin_prod_l16_16830

theorem sin_sum_square_gt_sin_prod (α β γ : ℝ) (h1 : α + β + γ = Real.pi) 
  (h2 : 0 < Real.sin α) (h3 : Real.sin α < 1)
  (h4 : 0 < Real.sin β) (h5 : Real.sin β < 1)
  (h6 : 0 < Real.sin γ) (h7 : Real.sin γ < 1) :
  (Real.sin α + Real.sin β + Real.sin γ) ^ 2 > 9 * Real.sin α * Real.sin β * Real.sin γ := 
sorry

end sin_sum_square_gt_sin_prod_l16_16830


namespace percentage_books_returned_l16_16885

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

end percentage_books_returned_l16_16885


namespace radius_base_circle_of_cone_l16_16294

theorem radius_base_circle_of_cone 
  (θ : ℝ) (R : ℝ) (arc_length : ℝ) (r : ℝ)
  (h1 : θ = 120) 
  (h2 : R = 9)
  (h3 : arc_length = (θ / 360) * 2 * Real.pi * R)
  (h4 : 2 * Real.pi * r = arc_length)
  : r = 3 := 
sorry

end radius_base_circle_of_cone_l16_16294


namespace children_difference_l16_16740

-- Define the initial number of children on the bus
def initial_children : ℕ := 5

-- Define the number of children who got off the bus
def children_off : ℕ := 63

-- Define the number of children on the bus after more got on
def final_children : ℕ := 14

-- Define the number of children who got on the bus
def children_on : ℕ := (final_children + children_off) - initial_children

-- Prove the number of children who got on minus the number of children who got off is equal to 9
theorem children_difference :
  (children_on - children_off) = 9 :=
by
  -- Direct translation from the proof steps
  sorry

end children_difference_l16_16740


namespace glass_pieces_same_color_l16_16553

theorem glass_pieces_same_color (r y b : ℕ) (h : r + y + b = 2002) :
  (∃ k : ℕ, ∀ n, n ≥ k → (r + y + b) = n ∧ (r = 0 ∨ y = 0 ∨ b = 0)) ∧
  (∀ (r1 y1 b1 r2 y2 b2 : ℕ),
    r1 + y1 + b1 = 2002 →
    r2 + y2 + b2 = 2002 →
    (∃ k : ℕ, ∀ n, n ≥ k → (r1 = 0 ∨ y1 = 0 ∨ b1 = 0)) →
    (∃ l : ℕ, ∀ m, m ≥ l → (r2 = 0 ∨ y2 = 0 ∨ b2 = 0)) →
    r1 = r2 ∧ y1 = y2 ∧ b1 = b2):=
by
  sorry

end glass_pieces_same_color_l16_16553


namespace num_pairs_satisfying_equation_l16_16937

theorem num_pairs_satisfying_equation :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end num_pairs_satisfying_equation_l16_16937


namespace gcd_of_72_120_168_l16_16136

theorem gcd_of_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  sorry

end gcd_of_72_120_168_l16_16136


namespace students_on_zoo_trip_l16_16120

theorem students_on_zoo_trip (buses : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) 
  (h1 : buses = 7) (h2 : students_per_bus = 56) (h3 : students_in_cars = 4) : 
  buses * students_per_bus + students_in_cars = 396 :=
by
  sorry

end students_on_zoo_trip_l16_16120


namespace employee_selected_from_10th_group_is_47_l16_16177

theorem employee_selected_from_10th_group_is_47
  (total_employees : ℕ)
  (sampled_employees : ℕ)
  (total_groups : ℕ)
  (random_start : ℕ)
  (common_difference : ℕ)
  (selected_from_5th_group : ℕ) :
  total_employees = 200 →
  sampled_employees = 40 →
  total_groups = 40 →
  random_start = 2 →
  common_difference = 5 →
  selected_from_5th_group = 22 →
  (selected_from_5th_group = (4 * common_difference + random_start)) →
  (9 * common_difference + random_start) = 47 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end employee_selected_from_10th_group_is_47_l16_16177


namespace bob_total_earnings_l16_16477

def hourly_rate_regular := 5
def hourly_rate_overtime := 6
def regular_hours_per_week := 40

def hours_worked_week1 := 44
def hours_worked_week2 := 48

def earnings_week1 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week1 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def earnings_week2 : ℕ :=
  let regular_hours := regular_hours_per_week
  let overtime_hours := hours_worked_week2 - regular_hours_per_week
  (regular_hours * hourly_rate_regular) + (overtime_hours * hourly_rate_overtime)

def total_earnings : ℕ := earnings_week1 + earnings_week2

theorem bob_total_earnings : total_earnings = 472 := by
  sorry

end bob_total_earnings_l16_16477


namespace cos_30_degrees_eq_sqrt_3_div_2_l16_16765

noncomputable def cos_30_degrees : ℝ :=
  real.cos (real.pi / 6)

theorem cos_30_degrees_eq_sqrt_3_div_2 :
  cos_30_degrees = sqrt 3 / 2 :=
sorry

end cos_30_degrees_eq_sqrt_3_div_2_l16_16765


namespace Derek_more_than_Zoe_l16_16193

-- Define the variables for the number of books Emily, Derek, and Zoe have
variables (E : ℝ)

-- Condition: Derek has 75% more books than Emily
def Derek_books : ℝ := 1.75 * E

-- Condition: Zoe has 50% more books than Emily
def Zoe_books : ℝ := 1.5 * E

-- Statement asserting that Derek has 16.67% more books than Zoe
theorem Derek_more_than_Zoe (hD: Derek_books E = 1.75 * E) (hZ: Zoe_books E = 1.5 * E) :
  (Derek_books E - Zoe_books E) / Zoe_books E = 0.1667 :=
by
  sorry

end Derek_more_than_Zoe_l16_16193


namespace monotonic_increase_l16_16172

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 2)

theorem monotonic_increase : ∀ x1 x2 : ℝ, 2 < x1 → x1 < x2 → f x1 < f x2 :=
by
  sorry

end monotonic_increase_l16_16172


namespace range_of_m_l16_16496

theorem range_of_m (m : ℝ) (x : ℝ) (h_eq : m / (x - 2) = 3) (h_pos : x > 0) : m > -6 ∧ m ≠ 0 := 
sorry

end range_of_m_l16_16496


namespace xiao_ying_performance_l16_16880

def regular_weight : ℝ := 0.20
def midterm_weight : ℝ := 0.30
def final_weight : ℝ := 0.50

def regular_score : ℝ := 85
def midterm_score : ℝ := 90
def final_score : ℝ := 92

-- Define the function that calculates the weighted average
def semester_performance (rw mw fw rs ms fs : ℝ) : ℝ :=
  rw * rs + mw * ms + fw * fs

-- The theorem that the weighted average of the scores is 90
theorem xiao_ying_performance : semester_performance regular_weight midterm_weight final_weight regular_score midterm_score final_score = 90 := by
  sorry

end xiao_ying_performance_l16_16880


namespace vertices_sum_zero_l16_16657

theorem vertices_sum_zero
  (a b c d e f g h : ℝ)
  (h1 : a = (b + e + d) / 3)
  (h2 : b = (c + f + a) / 3)
  (h3 : c = (d + g + b) / 3)
  (h4 : d = (a + h + e) / 3)
  :
  (a + b + c + d) - (e + f + g + h) = 0 :=
by
  sorry

end vertices_sum_zero_l16_16657


namespace exists_xi_l16_16132

variable (f : ℝ → ℝ)
variable (hf_diff : ∀ x, DifferentiableAt ℝ f x)
variable (hf_twice_diff : ∀ x, DifferentiableAt ℝ (deriv f) x)
variable (hf₀ : f 0 = 2)
variable (hf_prime₀ : deriv f 0 = -2)
variable (hf₁ : f 1 = 1)

theorem exists_xi (h0 : f 0 = 2) (h1 : deriv f 0 = -2) (h2 : f 1 = 1) :
  ∃ ξ ∈ Set.Ioo 0 1, f ξ * deriv f ξ + deriv (deriv f) ξ = 0 :=
sorry

end exists_xi_l16_16132


namespace max_distance_on_ellipse_l16_16401

def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

def P_on_ellipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

def distance (p1 p2: ℝ × ℝ) : ℝ := 
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_on_ellipse : 
  (B : ℝ × ℝ) (hB : B = (0, 1)) (hP : ∃ θ : ℝ, P_on_ellipse θ) 
  (h_cond : ∀ θ, ellipse (sqrt 5 * cos θ) (sin θ)) :
  ∃ θ : ℝ, distance (0, 1) (sqrt 5 * cos θ, sin θ) = 5 / 2 := 
sorry

end max_distance_on_ellipse_l16_16401


namespace chicken_burger_cost_l16_16029

namespace BurgerCost

variables (C B : ℕ)

theorem chicken_burger_cost (h1 : B = C + 300) 
                            (h2 : 3 * B + 3 * C = 21000) : 
                            C = 3350 := 
sorry

end BurgerCost

end chicken_burger_cost_l16_16029


namespace cats_weight_difference_l16_16987

-- Define the weights of Anne's and Meg's cats
variables (A M : ℕ)

-- Given conditions:
-- 1. Ratio of weights Meg's cat to Anne's cat is 13:21
-- 2. Meg's cat's weight is 20 kg plus half the weight of Anne's cat

theorem cats_weight_difference (h1 : M = 20 + (A / 2)) (h2 : 13 * A = 21 * M) : A - M = 64 := 
by {
    sorry
}

end cats_weight_difference_l16_16987


namespace find_number_l16_16384

-- Define the number x and state the condition 55 + x = 88
def x := 33

-- State the theorem to be proven: if 55 + x = 88, then x = 33
theorem find_number (h : 55 + x = 88) : x = 33 :=
by
  sorry

end find_number_l16_16384


namespace train_speed_in_kmh_l16_16892

-- Definitions of conditions
def time_to_cross_platform := 30  -- in seconds
def time_to_cross_man := 17  -- in seconds
def length_of_platform := 260  -- in meters

-- Conversion factor from m/s to km/h
def meters_per_second_to_kilometers_per_hour (v : ℕ) : ℕ :=
  v * 36 / 10

-- The theorem statement
theorem train_speed_in_kmh :
  (∃ (L V : ℕ),
    L = V * time_to_cross_man ∧
    L + length_of_platform = V * time_to_cross_platform ∧
    meters_per_second_to_kilometers_per_hour V = 72) :=
sorry

end train_speed_in_kmh_l16_16892


namespace problem_inequality_solution_set_inequality_proof_l16_16074

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem problem_inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -2 < x ∧ x < 2} :=
sorry

theorem inequality_proof (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) :
  |x + y| < |(x * y) / 2 + 2| :=
sorry

end problem_inequality_solution_set_inequality_proof_l16_16074


namespace tan20_plus_4sin20_l16_16588

noncomputable def problem_statement : Prop :=
  tan (20 * Real.pi / 180) + 4 * sin (20 * Real.pi / 180) = Real.sqrt 3

theorem tan20_plus_4sin20 :
  problem_statement :=
by
  sorry

end tan20_plus_4sin20_l16_16588


namespace find_k_l16_16144

theorem find_k (k : ℝ) (h : ∀ x : ℝ, x^2 + 10 * x + k = 0 → (∃ a : ℝ, a > 0 ∧ (x = -3 * a ∨ x = -a))) :
  k = 18.75 :=
sorry

end find_k_l16_16144


namespace positive_difference_abs_eq_15_l16_16004

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l16_16004


namespace max_value_of_b_minus_a_l16_16614

theorem max_value_of_b_minus_a (a b : ℝ) (h1 : a < 0) (h2 : ∀ x : ℝ, a < x ∧ x < b → (3 * x^2 + a) * (2 * x + b) ≥ 0) : b - a ≤ 1 / 3 :=
by
  sorry

end max_value_of_b_minus_a_l16_16614


namespace product_of_roots_eq_20_l16_16559

open Real

theorem product_of_roots_eq_20 :
  (∀ x : ℝ, (x^2 + 18 * x + 30 = 2 * sqrt (x^2 + 18 * x + 45)) → 
  (x^2 + 18 * x + 20 = 0)) → 
  ∀ α β : ℝ, (α ≠ β ∧ α * β = 20) :=
by
  intros h x hx
  sorry

end product_of_roots_eq_20_l16_16559


namespace force_of_water_on_lock_wall_l16_16194

noncomputable def force_on_the_wall (l h γ g : ℝ) : ℝ :=
  γ * g * l * (h^2 / 2)

theorem force_of_water_on_lock_wall :
  force_on_the_wall 20 5 1000 9.81 = 2.45 * 10^6 := by
  sorry

end force_of_water_on_lock_wall_l16_16194


namespace average_weight_ten_students_l16_16825

theorem average_weight_ten_students (avg_wt_girls avg_wt_boys : ℕ) 
  (count_girls count_boys : ℕ)
  (h1 : count_girls = 5) 
  (h2 : avg_wt_girls = 45) 
  (h3 : count_boys = 5) 
  (h4 : avg_wt_boys = 55) : 
  (count_girls * avg_wt_girls + count_boys * avg_wt_boys) / (count_girls + count_boys) = 50 :=
by sorry

end average_weight_ten_students_l16_16825


namespace problem1_problem2_problem3_problem4_l16_16568

-- Problem 1: 27 - 16 + (-7) - 18 = -14
theorem problem1 : 27 - 16 + (-7) - 18 = -14 := 
by 
  sorry

-- Problem 2: (-6) * (-3/4) / (-3/2) = -3
theorem problem2 : (-6) * (-3/4) / (-3/2) = -3 := 
by
  sorry

-- Problem 3: (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81
theorem problem3 : (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81 := 
by
  sorry

-- Problem 4: -2^4 + 3 * (-1)^4 - (-2)^3 = -5
theorem problem4 : -2^4 + 3 * (-1)^4 - (-2)^3 = -5 := 
by
  sorry

end problem1_problem2_problem3_problem4_l16_16568


namespace train_speed_is_correct_l16_16466

/-- Define the length of the train (in meters) -/
def train_length : ℕ := 120

/-- Define the length of the bridge (in meters) -/
def bridge_length : ℕ := 255

/-- Define the time to cross the bridge (in seconds) -/
def time_to_cross : ℕ := 30

/-- Define the total distance covered by the train while crossing the bridge -/
def total_distance : ℕ := train_length + bridge_length

/-- Define the speed of the train in meters per second -/
def speed_m_per_s : ℚ := total_distance / time_to_cross

/-- Conversion factor from m/s to km/hr -/
def m_per_s_to_km_per_hr : ℚ := 3.6

/-- The expected speed of the train in km/hr -/
def expected_speed_km_per_hr : ℕ := 45

/-- The theorem stating that the speed of the train is 45 km/hr -/
theorem train_speed_is_correct :
  (speed_m_per_s * m_per_s_to_km_per_hr) = expected_speed_km_per_hr := by
  sorry

end train_speed_is_correct_l16_16466


namespace compute_paths_in_grid_l16_16902

def grid : List (List Char) := [
  [' ', ' ', ' ', ' ', ' ', ' ', 'C', ' ', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', ' ', 'C', 'O', 'C', ' ', ' ', ' ', ' '],
  [' ', ' ', ' ', ' ', 'C', 'O', 'M', 'O', 'C', ' ', ' ', ' '],
  [' ', ' ', ' ', 'C', 'O', 'M', 'P', 'M', 'O', 'C', ' ', ' '],
  [' ', ' ', 'C', 'O', 'M', 'P', 'U', 'P', 'M', 'O', 'C', ' '],
  [' ', 'C', 'O', 'M', 'P', 'U', 'T', 'U', 'P', 'M', 'O', 'C'],
  ['C', 'O', 'M', 'P', 'U', 'T', 'E', 'T', 'U', 'P', 'M', 'O', 'C']
]

def is_valid_path (path : List (Nat × Nat)) : Bool :=
  -- This function checks if a given path is valid according to the problem's grid and rules.
  sorry

def count_paths_from_C_to_E (grid: List (List Char)) : Nat :=
  -- This function would count the number of valid paths from a 'C' in the leftmost column to an 'E' in the rightmost column.
  sorry

theorem compute_paths_in_grid : count_paths_from_C_to_E grid = 64 :=
by
  sorry

end compute_paths_in_grid_l16_16902


namespace minNumberOfRectangles_correct_l16_16241

variable (k n : ℤ)

noncomputable def minNumberOfRectangles (k n : ℤ) : ℤ :=
  if 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1 then
    if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1)
  else 0 -- 0 if the conditions are not met

theorem minNumberOfRectangles_correct (k n : ℤ) (h : 2 ≤ k ∧ k ≤ n ∧ n ≤ 2*k - 1) : 
  minNumberOfRectangles k n = 
  if n = k ∨ n = 2*k - 1 then n else 2 * (n - k + 1) := 
by 
  -- Proof will go here
  sorry

end minNumberOfRectangles_correct_l16_16241


namespace small_bonsai_sold_eq_l16_16405

-- Define the conditions
def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- The proof problem: Prove that the number of small bonsai sold is 3
theorem small_bonsai_sold_eq : ∃ x : ℕ, 30 * x + 20 * 5 = 190 ∧ x = 3 :=
by
  sorry

end small_bonsai_sold_eq_l16_16405


namespace sin_double_angle_half_l16_16942

theorem sin_double_angle_half (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end sin_double_angle_half_l16_16942


namespace max_sum_of_arithmetic_sequence_l16_16954

theorem max_sum_of_arithmetic_sequence 
  (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (a1 : a 1 = 29) 
  (S10_eq_S20 : S 10 = S 20) :
  (∃ n, ∀ m, S n ≥ S m) ∧ ∃ n, (S n = S 15) :=
sorry

end max_sum_of_arithmetic_sequence_l16_16954


namespace find_x_l16_16017

theorem find_x (x y : ℤ) (h1 : x + 2 * y = 10) (h2 : y = 1) : x = 8 :=
by sorry

end find_x_l16_16017


namespace net_profit_start_year_better_investment_option_l16_16183

-- Question 1: From which year does the developer start to make a net profit?
def investment_cost : ℕ := 81 -- in 10,000 yuan
def first_year_renovation_cost : ℕ := 1 -- in 10,000 yuan
def renovation_cost_increase : ℕ := 2 -- in 10,000 yuan per year
def annual_rental_income : ℕ := 30 -- in 10,000 yuan per year

theorem net_profit_start_year : ∃ n : ℕ, n ≥ 4 ∧ ∀ m < 4, ¬ (annual_rental_income * m > investment_cost + m^2) :=
by sorry

-- Question 2: Which option is better: maximizing total profit or average annual profit?
def profit_function (n : ℕ) : ℤ := 30 * n - (81 + n^2)
def average_annual_profit (n : ℕ) : ℤ := (30 * n - (81 + n^2)) / n
def max_total_profit_year : ℕ := 15
def max_total_profit : ℤ := 144 -- in 10,000 yuan
def max_average_profit_year : ℕ := 9
def max_average_profit : ℤ := 12 -- in 10,000 yuan

theorem better_investment_option : (average_annual_profit max_average_profit_year) ≥ (profit_function max_total_profit_year) / max_total_profit_year :=
by sorry

end net_profit_start_year_better_investment_option_l16_16183


namespace B_starts_after_A_l16_16895

theorem B_starts_after_A :
  ∀ (A_walk_speed B_cycle_speed dist_from_start t : ℝ), 
    A_walk_speed = 10 →
    B_cycle_speed = 20 →
    dist_from_start = 80 →
    B_cycle_speed * (dist_from_start - A_walk_speed * t) / A_walk_speed = t →
    t = 4 :=
by 
  intros A_walk_speed B_cycle_speed dist_from_start t hA_speed hB_speed hdist heq;
  sorry

end B_starts_after_A_l16_16895


namespace gold_coins_count_l16_16455

theorem gold_coins_count (G : ℕ) 
  (h1 : 50 * G + 125 + 30 = 305) :
  G = 3 := 
by
  sorry

end gold_coins_count_l16_16455


namespace yellow_papers_count_l16_16031

theorem yellow_papers_count (n : ℕ) (total_papers : ℕ) (periphery_papers : ℕ) (inner_papers : ℕ) 
  (h1 : n = 10) 
  (h2 : total_papers = n * n) 
  (h3 : periphery_papers = 4 * n - 4)
  (h4 : inner_papers = total_papers - periphery_papers) :
  inner_papers = 64 :=
by
  sorry

end yellow_papers_count_l16_16031


namespace beckett_younger_than_olaf_l16_16337

-- Define variables for ages
variables (O B S J : ℕ) (x : ℕ)

-- Express conditions as Lean hypotheses
def conditions :=
  B = O - x ∧  -- Beckett's age
  B = 12 ∧    -- Beckett is 12 years old
  S = O - 2 ∧ -- Shannen's age
  J = 2 * S + 5 ∧ -- Jack's age
  O + B + S + J = 71 -- Sum of ages
  
-- The theorem stating that Beckett is 8 years younger than Olaf
theorem beckett_younger_than_olaf (h : conditions O B S J x) : x = 8 :=
by
  -- The proof is omitted (using sorry)
  sorry

end beckett_younger_than_olaf_l16_16337


namespace find_n_22_or_23_l16_16111

theorem find_n_22_or_23 (n : ℕ) : 
  (∃ (sol_count : ℕ), sol_count = 30 ∧ (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 3 * x + 2 * y + 4 * z = n)) → 
  (n = 22 ∨ n = 23) := 
sorry

end find_n_22_or_23_l16_16111


namespace tanya_efficiency_greater_sakshi_l16_16261

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end tanya_efficiency_greater_sakshi_l16_16261


namespace negation_equiv_l16_16138

theorem negation_equiv {x : ℝ} : 
  (¬ (x^2 < 1 → -1 < x ∧ x < 1)) ↔ (x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) :=
by
  sorry

end negation_equiv_l16_16138


namespace number_of_positive_integer_pairs_l16_16938

theorem number_of_positive_integer_pairs (x y : ℕ) : 
  (x^2 - y^2 = 77) → (0 < x) → (0 < y) → (∃ x1 y1 x2 y2, (x1, y1) ≠ (x2, y2) ∧ 
  x1^2 - y1^2 = 77 ∧ x2^2 - y2^2 = 77 ∧ 0 < x1 ∧ 0 < y1 ∧ 0 < x2 ∧ 0 < y2 ∧
  ∀ a b, (a^2 - b^2 = 77 → a = x1 ∧ b = y1) ∨ (a = x2 ∧ b = y2)) :=
sorry

end number_of_positive_integer_pairs_l16_16938


namespace range_of_a_l16_16625

theorem range_of_a (a : ℝ) : (∃ x : ℝ, a * x^2 + x + 1 < 0) ↔ (a < 1/4) := 
sorry

end range_of_a_l16_16625


namespace sqrt_fraction_identity_l16_16304

theorem sqrt_fraction_identity (n : ℕ) (h : n > 0) : 
    Real.sqrt ((1 : ℝ) / n - (1 : ℝ) / (n * n)) = Real.sqrt (n - 1) / n :=
by
  sorry

end sqrt_fraction_identity_l16_16304


namespace arithmetic_sequence_problem_l16_16928

noncomputable def a1 := 3
noncomputable def S (n : ℕ) (a1 d : ℕ) : ℕ := n * (a1 + (n - 1) * d / 2)

theorem arithmetic_sequence_problem (d : ℕ) 
  (h1 : S 1 a1 d = 3) 
  (h2 : S 1 a1 d / 2 + S 4 a1 d / 4 = 18) : 
  S 5 a1 d = 75 :=
sorry

end arithmetic_sequence_problem_l16_16928


namespace obtuse_triangle_side_range_l16_16949

theorem obtuse_triangle_side_range {a : ℝ} (h1 : a > 3) (h2 : (a - 3)^2 < 36) : 3 < a ∧ a < 9 := 
by
  sorry

end obtuse_triangle_side_range_l16_16949


namespace min_value_expression_l16_16661

open Real

theorem min_value_expression (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_abc : a * b * c = 1 / 2) : 
    a^2 + 8 * a * b + 32 * b^2 + 16 * b * c + 8 * c^2 ≥ 18 :=
sorry

end min_value_expression_l16_16661


namespace abs_five_minus_e_l16_16488

noncomputable def e : ℝ := Real.exp 1

theorem abs_five_minus_e : |5 - e| = 5 - e := by
  sorry

end abs_five_minus_e_l16_16488


namespace r_values_if_polynomial_divisible_l16_16062

noncomputable
def find_r_iff_divisible (r : ℝ) : Prop :=
  (10 * (r^2 * (1 - 2*r))) = -6 ∧ 
  (2 * r + (1 - 2*r)) = 1 ∧ 
  (r^2 + 2 * r * (1 - 2*r)) = -5.2

theorem r_values_if_polynomial_divisible (r : ℝ) :
  (find_r_iff_divisible r) ↔ 
  (r = (2 + Real.sqrt 30) / 5 ∨ r = (2 - Real.sqrt 30) / 5) := 
by
  sorry

end r_values_if_polynomial_divisible_l16_16062


namespace checkered_fabric_cost_l16_16298

variable (P : ℝ) (cost_per_yard : ℝ) (total_yards : ℕ)
variable (x : ℝ) (C : ℝ)

theorem checkered_fabric_cost :
  P = 45 ∧ cost_per_yard = 7.50 ∧ total_yards = 16 →
  C = cost_per_yard * (total_yards - x) →
  7.50 * (16 - x) = 45 →
  C = 75 :=
by
  intro h1 h2 h3
  sorry

end checkered_fabric_cost_l16_16298


namespace coordinate_of_point_A_l16_16418

theorem coordinate_of_point_A (a b : ℝ) 
    (h1 : |b| = 3) 
    (h2 : |a| = 4) 
    (h3 : a > b) : 
    (a, b) = (4, 3) ∨ (a, b) = (4, -3) :=
by
    sorry

end coordinate_of_point_A_l16_16418


namespace find_m_such_that_no_linear_term_in_expansion_l16_16088

theorem find_m_such_that_no_linear_term_in_expansion :
  ∃ m : ℝ, ∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9 * x^2 - 8 * m ∧ ((8 + m) = 0) :=
by
  sorry

end find_m_such_that_no_linear_term_in_expansion_l16_16088


namespace proposition_true_l16_16729

theorem proposition_true (a b : ℝ) (h1 : 0 > a) (h2 : a > b) : (1/a) < (1/b) := 
sorry

end proposition_true_l16_16729


namespace max_value_of_expression_l16_16242

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 3) :
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 := 
sorry

end max_value_of_expression_l16_16242


namespace assignment_statement_meaning_l16_16811

-- Define the meaning of the assignment statement
def is_assignment_statement (s: String) : Prop := s = "Variable = Expression"

-- Define the specific assignment statement we are considering
def assignment_statement : String := "i = i + 1"

-- Define the meaning of the specific assignment statement
def assignment_meaning (s: String) : Prop := s = "Add 1 to the original value of i and then assign it back to i, the value of i increases by 1"

-- The proof statement
theorem assignment_statement_meaning :
  is_assignment_statement "Variable = Expression" → assignment_meaning "i = i + 1" :=
by
  intros
  sorry

end assignment_statement_meaning_l16_16811


namespace pentagon_stack_l16_16066

/-- Given a stack of identical regular pentagons with vertices numbered from 1 to 5, rotated and flipped
such that the sums of numbers at each vertex are the same, the number of pentagons in the stacks can be
any natural number except 1 and 3. -/
theorem pentagon_stack (n : ℕ) (h0 : identical_pentagons_with_vertices_1_to_5)
  (h1 : pentagons_can_be_rotated_and_flipped)
  (h2 : stacked_vertex_to_vertex)
  (h3 : sums_at_each_vertex_are_equal) :
  ∃ k : ℕ, k = n ∧ n ≠ 1 ∧ n ≠ 3 :=
sorry

end pentagon_stack_l16_16066


namespace correct_inequality_l16_16965

variable (a b c d : ℝ)
variable (h₁ : a > b)
variable (h₂ : b > 0)
variable (h₃ : 0 > c)
variable (h₄ : c > d)

theorem correct_inequality :
  (c / a) - (d / b) > 0 :=
by sorry

end correct_inequality_l16_16965


namespace problem_l16_16627

def point := ℕ × ℕ

def steps (start finish : point) := 
  ((finish.fst - start.fst) + (finish.snd - start.snd))

def paths (start finish : point) :=
  if (finish.fst >= start.fst) ∧ (finish.snd >= start.snd) then
    Nat.choose ((finish.fst - start.fst) + (finish.snd - start.snd)) (finish.snd - start.snd)
  else 0

def A : point := (0, 0)
def B : point := (4, 2)
def C : point := (7, 4)

theorem problem : steps A C = 11 ∧ paths A B * paths B C = 150 :=
by
  sorry

end problem_l16_16627


namespace plane_equation_correct_l16_16589

def plane_equation (x y z : ℝ) : ℝ := 10 * x - 5 * y + 4 * z - 141

noncomputable def gcd (a b c d : ℤ) : ℤ := Int.gcd (Int.gcd a b) (Int.gcd c d)

theorem plane_equation_correct :
  (∀ x y z, plane_equation x y z = 0 ↔ 10 * x - 5 * y + 4 * z - 141 = 0)
  ∧ gcd 10 (-5) 4 (-141) = 1
  ∧ 10 > 0 := by
  sorry

end plane_equation_correct_l16_16589


namespace hyperbola_s_squared_l16_16576

theorem hyperbola_s_squared 
  (s : ℝ) 
  (a b : ℝ) 
  (h1 : a = 3)
  (h2 : b^2 = 144 / 13) 
  (h3 : (2, s) ∈ {p : ℝ × ℝ | (p.2)^2 / a^2 - (p.1)^2 / b^2 = 1}) :
  s^2 = 441 / 36 :=
by sorry

end hyperbola_s_squared_l16_16576


namespace solve_for_m_l16_16085

theorem solve_for_m {m : ℝ} (h : ∀ x : ℝ, (m - 5) * x = 0) : m = 5 :=
sorry

end solve_for_m_l16_16085


namespace height_of_flagpole_l16_16742

-- Define the given conditions
variables (h : ℝ) -- height of the flagpole
variables (s_f : ℝ) (s_b : ℝ) (h_b : ℝ) -- s_f: shadow length of flagpole, s_b: shadow length of building, h_b: height of building

-- Problem conditions
def flagpole_shadow := (s_f = 45)
def building_shadow := (s_b = 50)
def building_height := (h_b = 20)

-- Mathematically equivalent statement
theorem height_of_flagpole
  (h_f : ℝ) (hsf : flagpole_shadow s_f) (hsb : building_shadow s_b) (hhb : building_height h_b)
  (similar_conditions : h / s_f = h_b / s_b) :
  h_f = 18 :=
by
  sorry

end height_of_flagpole_l16_16742


namespace circle_condition_iff_l16_16989

-- Given a condition a < 2, we need to show it is a necessary and sufficient condition
-- for the equation x^2 + y^2 - 2x + 2y + a = 0 to represent a circle.

theorem circle_condition_iff (a : ℝ) :
  (∃ (x y : ℝ), (x - 1) ^ 2 + (y + 1) ^ 2 = 2 - a) ↔ (a < 2) :=
sorry

end circle_condition_iff_l16_16989


namespace head_start_distance_l16_16448

theorem head_start_distance (v_A v_B L H : ℝ) (h1 : v_A = 15 / 13 * v_B)
    (h2 : t_A = L / v_A) (h3 : t_B = (L - H) / v_B) (h4 : t_B = t_A - 0.25 * L / v_B) :
    H = 23 / 60 * L :=
sorry

end head_start_distance_l16_16448


namespace find_a_b_find_m_l16_16506

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_a_b (a b : ℝ) (h₁ : f 1 a b = 4)
  (h₂ : 3 * a + 2 * b = 9) : a = 1 ∧ b = 3 :=
by
  sorry

theorem find_m (m : ℝ) (h : ∀ x, (m ≤ x ∧ x ≤ m + 1) → (3 * x^2 + 6 * x > 0)) :
  m ≥ 0 ∨ m ≤ -3 :=
by
  sorry

end find_a_b_find_m_l16_16506


namespace pyramid_volume_l16_16480

theorem pyramid_volume
  (FB AC FA FC AB BC : ℝ)
  (hFB : FB = 12)
  (hAC : AC = 4)
  (hFA : FA = 7)
  (hFC : FC = 7)
  (hAB : AB = 7)
  (hBC : BC = 7) :
  (1/3 * AC * (1/2 * FB * 3)) = 24 := by sorry

end pyramid_volume_l16_16480


namespace flowers_bees_butterflies_comparison_l16_16428

def num_flowers : ℕ := 12
def num_bees : ℕ := 7
def num_butterflies : ℕ := 4
def difference_flowers_bees : ℕ := num_flowers - num_bees

theorem flowers_bees_butterflies_comparison :
  difference_flowers_bees - num_butterflies = 1 :=
by
  -- The proof will go here
  sorry

end flowers_bees_butterflies_comparison_l16_16428


namespace rectangle_length_l16_16419

variable (w l : ℝ)

def perimeter (w l : ℝ) : ℝ := 2 * w + 2 * l

theorem rectangle_length (h1 : l = w + 2) (h2 : perimeter w l = 20) : l = 6 :=
by sorry

end rectangle_length_l16_16419


namespace perimeter_of_triangle_l16_16415

-- The given condition about the average length of the triangle sides.
def average_side_length (a b c : ℝ) (h : (a + b + c) / 3 = 12) : Prop :=
  a + b + c = 36

-- The theorem to prove the perimeter of triangle ABC.
theorem perimeter_of_triangle (a b c : ℝ) (h : (a + b + c) / 3 = 12) : a + b + c = 36 :=
  by
    sorry

end perimeter_of_triangle_l16_16415


namespace intersection_A_B_l16_16626

noncomputable def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by 
  sorry

end intersection_A_B_l16_16626


namespace max_distance_on_ellipse_l16_16399

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2/5 + y^2 = 1

def upper_vertex (x y : ℝ) : Prop :=
  x = 0 ∧ y = 1

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem max_distance_on_ellipse :
  ∃ P : ℝ × ℝ, ellipse P.1 P.2 → 
    ∀ B : ℝ × ℝ, upper_vertex B.1 B.2 → 
      distance P.1 P.2 B.1 B.2 ≤ 5/2 :=
sorry

end max_distance_on_ellipse_l16_16399


namespace chicken_rabbit_problem_l16_16310

theorem chicken_rabbit_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x = 23 ∧ y = 12 :=
by
  sorry

end chicken_rabbit_problem_l16_16310


namespace small_cone_altitude_l16_16318

theorem small_cone_altitude (h_f: ℝ) (a_lb: ℝ) (a_ub: ℝ) : 
  h_f = 24 → a_lb = 225 * Real.pi → a_ub = 25 * Real.pi → ∃ h_s, h_s = 12 := 
by
  intros h1 h2 h3
  sorry

end small_cone_altitude_l16_16318


namespace gopi_servant_salary_l16_16936

theorem gopi_servant_salary (S : ℝ) (h1 : 9 / 12 * S + 110 = 150) : S = 200 :=
by
  sorry

end gopi_servant_salary_l16_16936


namespace intersection_ellipse_line_range_b_l16_16794

theorem intersection_ellipse_line_range_b (b : ℝ) : 
  (∀ m : ℝ, ∃ x y : ℝ, x^2 + 2*y^2 = 3 ∧ y = m*x + b) ↔ 
  (- (Real.sqrt 6) / 2) ≤ b ∧ b ≤ (Real.sqrt 6) / 2 :=
by {
  sorry
}

end intersection_ellipse_line_range_b_l16_16794


namespace upstream_distance_l16_16884

theorem upstream_distance
  (man_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (effective_downstream_speed: ℝ)
  (stream_speed : ℝ)
  (upstream_time : ℝ)
  (upstream_distance : ℝ):
  man_speed = 7 ∧ downstream_distance = 45 ∧ downstream_time = 5 ∧ effective_downstream_speed = man_speed + stream_speed 
  ∧ effective_downstream_speed * downstream_time = downstream_distance 
  ∧ upstream_time = 5 ∧ upstream_distance = (man_speed - stream_speed) * upstream_time 
  → upstream_distance = 25 :=
by
  sorry

end upstream_distance_l16_16884


namespace sum_of_solutions_is_267_l16_16692

open Set

noncomputable def inequality (x : ℝ) : Prop :=
  sqrt (x^2 + x - 56) - sqrt (x^2 + 25*x + 136) < 8 * sqrt ((x - 7) / (x + 8))

noncomputable def valid_integers : Set ℝ :=
  {x | x ∈ Icc (-25 : ℝ) 25 ∧ (x ∈ (-20 : ℝ, -18) ∨ x ∈ Ici (7 : ℝ))}

theorem sum_of_solutions_is_267 :
  ∑ i in (Icc (-25 : ℝ) 25).to_finset.filter (λ x, inequality x), x = 267 :=
sorry

end sum_of_solutions_is_267_l16_16692


namespace g_nine_l16_16273

variable (g : ℝ → ℝ)

theorem g_nine : (∀ x y : ℝ, g (x + y) = g x * g y) → g 3 = 4 → g 9 = 64 :=
by intros h1 h2; sorry

end g_nine_l16_16273


namespace symmetry_about_x2_symmetry_about_2_0_l16_16303

-- Define the conditions and their respective conclusions.
theorem symmetry_about_x2 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) : 
  ∀ x, f (x) = f (4 - x) := 
sorry

theorem symmetry_about_2_0 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = -f (3 + x)) : 
  ∀ x, f (x) = -f (4 - x) := 
sorry

end symmetry_about_x2_symmetry_about_2_0_l16_16303


namespace roger_owes_correct_amount_l16_16974

def initial_house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_payment_percentage : ℝ := 0.30

def down_payment : ℝ := down_payment_percentage * initial_house_price
def remaining_after_down_payment : ℝ := initial_house_price - down_payment
def parents_payment : ℝ := parents_payment_percentage * remaining_after_down_payment
def money_owed : ℝ := remaining_after_down_payment - parents_payment

theorem roger_owes_correct_amount :
  money_owed = 56000 := by
  sorry

end roger_owes_correct_amount_l16_16974


namespace find_a_l16_16357

theorem find_a (a : ℝ) : (∀ x : ℝ, (x + 1) * (x - 3) = x^2 + a * x - 3) → a = -2 :=
  by
    sorry

end find_a_l16_16357


namespace green_eyes_students_l16_16117

def total_students := 45
def brown_hair_condition (green_eyes : ℕ) := 3 * green_eyes
def both_attributes := 9
def neither_attributes := 5

theorem green_eyes_students (green_eyes : ℕ) :
  (total_students = (green_eyes - both_attributes) + both_attributes
    + (brown_hair_condition green_eyes - both_attributes) + neither_attributes) →
  green_eyes = 10 :=
by
  sorry

end green_eyes_students_l16_16117


namespace quadratic_root_c_l16_16196

theorem quadratic_root_c (c : ℝ) :
  (∀ x : ℝ, x^2 + 3 * x + c = (x + (3/2))^2 - 7/4) → c = 1/2 :=
by
  sorry

end quadratic_root_c_l16_16196


namespace cartesian_eq_of_curve_C_distance_AB_l16_16656

variables {α x y ρ θ : ℝ}

theorem cartesian_eq_of_curve_C (h1 : x = sin α + cos α) (h2 : y = sin α - cos α) : 
  x^2 + y^2 = 2 :=
by sorry

theorem distance_AB (h3 : √2 * ρ * sin (π / 4 - θ) + 1 / 2 = 0)
  (h4 : x^2 + y^2 = 2)
  (h5 : ∀ (x y : ℝ), x - y + 1 / 2 = 0) :
  |AB| = (√30) / 2 :=
by sorry

end cartesian_eq_of_curve_C_distance_AB_l16_16656


namespace triangle_side_a_value_l16_16658

noncomputable def a_value (A B c : ℝ) : ℝ :=
  30 * Real.sqrt 2 - 10 * Real.sqrt 6

theorem triangle_side_a_value
  (A B : ℝ) (c : ℝ)
  (hA : A = 60)
  (hB : B = 45)
  (hc : c = 20) :
  a_value A B c = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by
  sorry

end triangle_side_a_value_l16_16658


namespace cube_face_sharing_l16_16022

theorem cube_face_sharing (n : ℕ) :
  (∃ W B : ℕ, (W + B = n^3) ∧ (3 * W = 3 * B) ∧ W = B ∧ W = n^3 / 2) ↔ n % 2 = 0 :=
by
  sorry

end cube_face_sharing_l16_16022


namespace twenty_yuan_banknotes_count_l16_16429

theorem twenty_yuan_banknotes_count (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
                                    (total_banknotes : x + y + z = 24)
                                    (total_amount : 10 * x + 20 * y + 50 * z = 1000) :
                                    y = 4 := 
sorry

end twenty_yuan_banknotes_count_l16_16429


namespace knights_on_red_chairs_l16_16552

theorem knights_on_red_chairs (K L K_r L_b : ℕ) (h1: K + L = 20)
  (h2: K - K_r + L_b = 10) (h3: K_r + L - L_b = 10) (h4: K_r = L_b) : K_r = 5 := by
  sorry

end knights_on_red_chairs_l16_16552


namespace soccer_league_teams_l16_16720

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 :=
by
  -- Proof will go here
  sorry

end soccer_league_teams_l16_16720


namespace dot_product_conditioned_l16_16510

variables (a b : ℝ×ℝ)

def condition1 : Prop := 2 • a + b = (1, 6)
def condition2 : Prop := a + 2 • b = (-4, 9)
def dot_product (u v : ℝ×ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem dot_product_conditioned :
  condition1 a b ∧ condition2 a b → dot_product a b = -2 :=
by
  sorry

end dot_product_conditioned_l16_16510


namespace largest_prime_inequality_l16_16108

def largest_prime_divisor (n : Nat) : Nat :=
  sorry  -- Placeholder to avoid distractions in problem statement

theorem largest_prime_inequality (q : Nat) (h_q_prime : Prime q) (hq_odd : q % 2 = 1) :
    ∃ k : Nat, k > 0 ∧ largest_prime_divisor (q^(2^k) - 1) < q ∧ q < largest_prime_divisor (q^(2^k) + 1) :=
sorry

end largest_prime_inequality_l16_16108


namespace units_digit_product_even_composite_l16_16864

/-- The units digit of the product of the first three even composite numbers greater than 10 is 8. -/
theorem units_digit_product_even_composite :
  let a := 12
  let b := 14
  let c := 16
  (a * b * c) % 10 = 8 :=
by
  let a := 12
  let b := 14
  let c := 16
  have h : (a * b * c) % 10 = 8
  { sorry }
  exact h

end units_digit_product_even_composite_l16_16864


namespace range_of_f_l16_16211

-- Define the function f(x) = 4 sin^3(x) + sin^2(x) - 4 sin(x) + 8
noncomputable def f (x : ℝ) : ℝ :=
  4 * (Real.sin x) ^ 3 + (Real.sin x) ^ 2 - 4 * (Real.sin x) + 8

-- Statement to prove the range of f(x)
theorem range_of_f :
  ∀ x : ℝ, 6 + 3 / 4 ≤ f x ∧ f x ≤ 9 + 25 / 27 :=
sorry

end range_of_f_l16_16211


namespace quadratic_inequality_solution_l16_16988

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 3 * x + 2 < 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end quadratic_inequality_solution_l16_16988


namespace relation_between_a_b_l16_16820

variables {x y a b : ℝ}

theorem relation_between_a_b 
  (h1 : a = (x^2 + y^2) * (x - y))
  (h2 : b = (x^2 - y^2) * (x + y))
  (h3 : x < y) 
  (h4 : y < 0) : 
  a > b := 
by sorry

end relation_between_a_b_l16_16820


namespace range_of_m_l16_16516

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 3) (h4 : ∀ x y, x > 0 → y > 0 → x + y = 3 → (4 / (x + 1) + 16 / y > m^2 - 3 * m + 11)) : 1 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l16_16516


namespace false_log_exists_x_l16_16870

theorem false_log_exists_x {x : ℝ} : ¬ ∃ x : ℝ, Real.log x = 0 :=
by sorry

end false_log_exists_x_l16_16870


namespace no_linear_term_l16_16086

theorem no_linear_term (m : ℤ) : (∀ (x : ℤ), (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 - 8*m → (8 + m) = 0) → m = -8 :=
by
  sorry

end no_linear_term_l16_16086


namespace marbles_total_l16_16431

theorem marbles_total (yellow blue red total : ℕ)
  (hy : yellow = 5)
  (h_ratio : blue / red = 3 / 4)
  (h_red : red = yellow + 3)
  (h_total : total = yellow + blue + red) : total = 19 :=
by
  sorry

end marbles_total_l16_16431


namespace proof_A2_less_than_3A1_plus_n_l16_16290

-- Define the conditions in terms of n, A1, and A2.
variables (n : ℕ)

-- A1 and A2 are the numbers of selections to select two students
-- such that their weight difference is ≤ 1 kg and ≤ 2 kg respectively.
variables (A1 A2 : ℕ)

-- The main theorem needs to prove that A2 < 3 * A1 + n.
theorem proof_A2_less_than_3A1_plus_n (h : A2 < 3 * A1 + n) : A2 < 3 * A1 + n :=
by {
  sorry -- proof goes here, but it's not required for the Lean statement.
}

end proof_A2_less_than_3A1_plus_n_l16_16290


namespace determine_a_l16_16768

theorem determine_a (a b c : ℕ) (h_b : b = 5) (h_c : c = 6) (h_order : c > b ∧ b > a ∧ a > 2) :
(a - 2) * (b - 2) * (c - 2) = 4 * (b - 2) + 4 * (c - 2) → a = 4 :=
by 
  sorry

end determine_a_l16_16768


namespace digit_b_divisible_by_7_l16_16700

theorem digit_b_divisible_by_7 (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) 
  (hdiv : (4000 + 110 * B + 3) % 7 = 0) : B = 0 :=
by
  sorry

end digit_b_divisible_by_7_l16_16700


namespace problem_statement_l16_16900

theorem problem_statement : 100 * 29.98 * 2.998 * 1000 = (2998)^2 :=
by
  sorry

end problem_statement_l16_16900


namespace train_length_is_250_l16_16186

noncomputable def train_length (V₁ V₂ V₃ : ℕ) (T₁ T₂ T₃ : ℕ) : ℕ :=
  let S₁ := (V₁ * (5/18) * T₁)
  let S₂ := (V₂ * (5/18)* T₂)
  let S₃ := (V₃ * (5/18) * T₃)
  if S₁ = S₂ ∧ S₂ = S₃ then S₁ else 0

theorem train_length_is_250 :
  train_length 50 60 70 18 20 22 = 250 := by
  -- proof omitted
  sorry

end train_length_is_250_l16_16186


namespace problem_1_problem_2_problem_3_problem_4_l16_16039

-- Problem 1
theorem problem_1 (x y : ℝ) : 
  -4 * x^2 * y * (x * y - 5 * y^2 - 1) = -4 * x^3 * y^2 + 20 * x^2 * y^3 + 4 * x^2 * y :=
by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) :
  (-3 * a)^2 - (2 * a + 1) * (a - 2) = 7 * a^2 + 3 * a + 2 :=
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) :
  (-2 * x - 3 * y) * (3 * y - 2 * x) - (2 * x - 3 * y)^2 = 12 * x * y - 18 * y^2 :=
by
  sorry

-- Problem 4
theorem problem_4 : 2010^2 - 2011 * 2009 = 1 :=
by
  sorry

end problem_1_problem_2_problem_3_problem_4_l16_16039


namespace object_speed_conversion_l16_16041

theorem object_speed_conversion 
  (distance : ℝ)
  (velocity : ℝ) 
  (conversion_factor : ℝ) 
  (distance_in_km : ℝ)
  (time_in_seconds : ℝ) 
  (time_in_minutes : ℝ) 
  (speed_in_kmh : ℝ) :
  distance = 200 ∧ 
  velocity = 1/3 ∧ 
  time_in_seconds = distance / velocity ∧ 
  time_in_minutes = time_in_seconds / 60 ∧ 
  conversion_factor = 3600 * 0.001 ∧ 
  speed_in_kmh = velocity * conversion_factor ↔ 
  speed_in_kmh = 0.4 :=
by sorry

end object_speed_conversion_l16_16041


namespace number_of_players_per_game_l16_16992

def total_players : ℕ := 50
def total_games : ℕ := 1225

-- If each player plays exactly one game with each of the other players,
-- there are C(total_players, 2) = total_games games.
theorem number_of_players_per_game : ∃ k : ℕ, k = 2 ∧ (total_players * (total_players - 1)) / 2 = total_games := 
  sorry

end number_of_players_per_game_l16_16992


namespace desired_interest_percentage_l16_16744

-- Definitions based on conditions
def face_value : ℝ := 20
def dividend_rate : ℝ := 0.09  -- 9% converted to fraction
def market_value : ℝ := 15

-- The main statement
theorem desired_interest_percentage : 
  ((dividend_rate * face_value) / market_value) * 100 = 12 :=
by
  sorry

end desired_interest_percentage_l16_16744


namespace car_cost_l16_16126

theorem car_cost (days_in_week : ℕ) (sue_days : ℕ) (sister_days : ℕ) 
  (sue_payment : ℕ) (car_cost : ℕ) 
  (h1 : days_in_week = 7)
  (h2 : sue_days = days_in_week - sister_days)
  (h3 : sister_days = 4)
  (h4 : sue_payment = 900)
  (h5 : sue_payment * days_in_week = sue_days * car_cost) :
  car_cost = 2100 := 
by {
  sorry
}

end car_cost_l16_16126


namespace fraction_equiv_subtract_l16_16866

theorem fraction_equiv_subtract (n : ℚ) : (4 - n) / (7 - n) = 3 / 5 → n = 0.5 :=
by
  intros h
  sorry

end fraction_equiv_subtract_l16_16866


namespace find_t_find_s_find_a_find_c_l16_16545

-- Proof Problem I4.1
theorem find_t (p q r t : ℝ) (h1 : (p + q + r) / 3 = 12) (h2 : (p + q + r + t + 2 * t) / 5 = 15) : t = 13 :=
sorry

-- Proof Problem I4.2
theorem find_s (k t s : ℝ) (hk : k ≠ 0) (h1 : k^4 + (1 / k^4) = t + 1) (h2 : t = 13) (h_s : s = k^2 + (1 / k^2)) : s = 4 :=
sorry

-- Proof Problem I4.3
theorem find_a (s a b : ℝ) (hxₘ : 1 ≠ 11) (hyₘ : 2 ≠ 7) (h1 : (a, b) = ((1 * 11 + s * 1) / (1 + s), (1 * 7 + s * 2) / (1 + s))) (h_s : s = 4) : a = 3 :=
sorry

-- Proof Problem I4.4
theorem find_c (a c : ℝ) (h1 : ∀ x, a * x^2 + 12 * x + c = 0 → (a*x^2 + 12 * x + c = 0)) (h2 : ∃ x, a * x^2 + 12 * x + c = 0) : c = 36 / a :=
sorry

end find_t_find_s_find_a_find_c_l16_16545


namespace arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l16_16851

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in
  let l := 96 in
  let n := (96 - 16) / 8 + 1 in
  let S := (n * (a + l)) / 2 in
  S / n

theorem arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
by
  let a := 16
  let l := 96
  let n := (l - a) / 8 + 1
  let S := (n * (a + l)) / 2
  have h1 : n = 11 := by sorry
  have h2 : S = 616 := by sorry
  show
    (S / n : ℕ) = 56
    by
      rw [h1, h2]
      exact rfl

end arithmetic_mean_of_all_positive_two_digit_multiples_of_8_is_56_l16_16851


namespace john_toy_store_fraction_l16_16512

theorem john_toy_store_fraction :
  let allowance := 4.80
  let arcade_spent := 3 / 5 * allowance
  let remaining_after_arcade := allowance - arcade_spent
  let candy_store_spent := 1.28
  let toy_store_spent := remaining_after_arcade - candy_store_spent
  (toy_store_spent / remaining_after_arcade) = 1 / 3 := by
    sorry

end john_toy_store_fraction_l16_16512


namespace arithmetic_sequence_product_l16_16238

noncomputable def b (n : ℕ) : ℤ := sorry -- define the arithmetic sequence

theorem arithmetic_sequence_product (d : ℤ) 
  (h_seq : ∀ n, b (n + 1) = b n + d)
  (h_inc : ∀ m n, m < n → b m < b n)
  (h_prod : b 4 * b 5 = 30) :
  b 3 * b 6 = -1652 ∨ b 3 * b 6 = -308 ∨ b 3 * b 6 = -68 ∨ b 3 * b 6 = 28 := 
sorry

end arithmetic_sequence_product_l16_16238


namespace sufficient_condition_l16_16171

theorem sufficient_condition (a : ℝ) (h : a > 0) : a^2 + a ≥ 0 :=
sorry

end sufficient_condition_l16_16171


namespace evaluate_cubic_difference_l16_16931

theorem evaluate_cubic_difference (x y : ℚ) (h1 : x + y = 10) (h2 : 2 * x - y = 16) :
  x^3 - y^3 = 17512 / 27 :=
by sorry

end evaluate_cubic_difference_l16_16931


namespace shaded_area_is_correct_l16_16336

-- Conditions definition
def shaded_numbers : ℕ := 2015
def boundary_properties (segment : ℕ) : Prop := 
  segment = 1 ∨ segment = 2

theorem shaded_area_is_correct : ∀ n : ℕ, n = shaded_numbers → boundary_properties n → 
  (∃ area : ℚ, area = 47.5) :=
by
  sorry

end shaded_area_is_correct_l16_16336


namespace solution_set_ineq_l16_16070

-- Definitions based on the problem conditions
def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x : ℝ, x ∈ (λ y, y ∈ ℝ)
axiom f_property : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 3
axiom f_at_5 : f 5 = 18

-- Statement to prove the solution set of the given inequality
theorem solution_set_ineq (x : ℝ) : f (3 * x - 1) > 9 * x ↔ x > 2 :=
by
  sorry

end solution_set_ineq_l16_16070


namespace square_area_l16_16752

theorem square_area (x : ℝ) (side1 side2 : ℝ) 
  (h_side1 : side1 = 6 * x - 27) 
  (h_side2 : side2 = 30 - 2 * x) 
  (h_equiv : side1 = side2) : 
  (side1 * side1 = 248.0625) := 
by
  sorry

end square_area_l16_16752


namespace triangle_side_length_x_l16_16958

theorem triangle_side_length_x
  (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ)
  (hy : y = 7)
  (hz : z = 3)
  (hcos : cos_Y_minus_Z = 7 / 8) :
  ∃ x : ℝ, x = Real.sqrt 18.625 :=
by
  sorry

end triangle_side_length_x_l16_16958


namespace tan_660_eq_neg_sqrt3_l16_16846

theorem tan_660_eq_neg_sqrt3 : Real.tan (660 * Real.pi / 180) = -Real.sqrt 3 :=
by
  sorry

end tan_660_eq_neg_sqrt3_l16_16846


namespace number_of_cows_consume_in_96_days_l16_16796

-- Given conditions
def grass_growth_rate := 10 / 3
def consumption_by_70_cows_in_24_days := 70 * 24
def consumption_by_30_cows_in_60_days := 30 * 60
def total_grass_in_96_days := consumption_by_30_cows_in_60_days + 120

-- Problem statement
theorem number_of_cows_consume_in_96_days : 
  (x : ℕ) -> 96 * x = total_grass_in_96_days -> x = 20 :=
by
  intros x h
  sorry

end number_of_cows_consume_in_96_days_l16_16796


namespace katie_remaining_juice_l16_16521

-- Define the initial condition: Katie initially has 5 gallons of juice
def initial_gallons : ℚ := 5

-- Define the amount of juice given to Mark
def juice_given : ℚ := 18 / 7

-- Define the expected remaining fraction of juice
def expected_remaining_gallons : ℚ := 17 / 7

-- The theorem statement that Katie should have 17/7 gallons of juice left
theorem katie_remaining_juice : initial_gallons - juice_given = expected_remaining_gallons := 
by
  -- proof would go here
  sorry

end katie_remaining_juice_l16_16521


namespace num_pairs_of_positive_integers_eq_77_l16_16940

theorem num_pairs_of_positive_integers_eq_77 : 
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.finite ∧
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x^2 - y^2 = 77}.to_finset.card = 2 := 
by 
  sorry

end num_pairs_of_positive_integers_eq_77_l16_16940


namespace range_of_a_l16_16358

noncomputable def p (x : ℝ) : Prop := x^2 - 8 * x - 20 < 0

noncomputable def q (x a : ℝ) : Prop := x^2 - 2 * x + 1 - a^2 ≤ 0

def sufficient_but_not_necessary_condition (a : ℝ) : Prop :=
  ∀ x, p x → q x a

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : sufficient_but_not_necessary_condition a) :
  9 ≤ a :=
sorry

end range_of_a_l16_16358


namespace track_length_l16_16036

theorem track_length (x : ℝ) 
  (h1 : ∀ {d1 d2 : ℝ}, (d1 + d2 = x / 2) → (d1 = 120) → d2 = x / 2 - 120)
  (h2 : ∀ {d1 d2 : ℝ}, (d1 = x / 2 - 120 + 170) → (d1 = x / 2 + 50))
  (h3 : ∀ {d3 : ℝ}, (d3 = 3 * x / 2 - 170)) :
  x = 418 :=
by
  sorry

end track_length_l16_16036


namespace sets_equal_l16_16445

theorem sets_equal :
  let M := {x | x^2 - 2 * x + 1 = 0}
  let N := {1}
  M = N :=
by
  sorry

end sets_equal_l16_16445


namespace prob_log3_integer_l16_16578

theorem prob_log3_integer : 
  (∃ (N: ℕ), (100 ≤ N ∧ N ≤ 999) ∧ ∃ (k: ℕ), N = 3^k) → 
  (∃ (prob : ℚ), prob = 1 / 450) :=
sorry

end prob_log3_integer_l16_16578


namespace factorize_ax2_minus_a_l16_16051

theorem factorize_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_ax2_minus_a_l16_16051


namespace contestant_final_score_l16_16653

theorem contestant_final_score 
    (content_score : ℕ)
    (delivery_score : ℕ)
    (weight_content : ℕ)
    (weight_delivery : ℕ)
    (h1 : content_score = 90)
    (h2 : delivery_score = 85)
    (h3 : weight_content = 6)
    (h4 : weight_delivery = 4) : 
    (content_score * weight_content + delivery_score * weight_delivery) / (weight_content + weight_delivery) = 88 := 
sorry

end contestant_final_score_l16_16653


namespace find_m_value_l16_16792

theorem find_m_value
  (y_squared_4x : ∀ x y : ℝ, y^2 = 4 * x)
  (Focus_F : ℝ × ℝ)
  (M N : ℝ × ℝ)
  (E : ℝ)
  (P Q : ℝ × ℝ)
  (k1 k2 : ℝ)
  (MN_slope : k1 = (N.snd - M.snd) / (N.fst - M.fst))
  (PQ_slope : k2 = (Q.snd - P.snd) / (Q.fst - P.fst))
  (slope_condition : k1 = 3 * k2) :
  E = 3 := 
sorry

end find_m_value_l16_16792


namespace bob_total_calories_l16_16338

def total_calories (slices_300 : ℕ) (calories_300 : ℕ) (slices_400 : ℕ) (calories_400 : ℕ) : ℕ :=
  slices_300 * calories_300 + slices_400 * calories_400

theorem bob_total_calories 
  (slices_300 : ℕ := 3)
  (calories_300 : ℕ := 300)
  (slices_400 : ℕ := 4)
  (calories_400 : ℕ := 400) :
  total_calories slices_300 calories_300 slices_400 calories_400 = 2500 := 
by 
  sorry

end bob_total_calories_l16_16338


namespace angle_sum_of_roots_of_complex_eq_32i_l16_16592

noncomputable def root_angle_sum : ℝ :=
  let θ1 := 22.5
  let θ2 := 112.5
  let θ3 := 202.5
  let θ4 := 292.5
  θ1 + θ2 + θ3 + θ4

theorem angle_sum_of_roots_of_complex_eq_32i :
  root_angle_sum = 630 := by
  sorry

end angle_sum_of_roots_of_complex_eq_32i_l16_16592


namespace zoo_rabbits_count_l16_16280

theorem zoo_rabbits_count (parrots rabbits : ℕ) (h_ratio : parrots * 4 = rabbits * 3) (h_parrots_count : parrots = 21) : rabbits = 28 :=
by
  sorry

end zoo_rabbits_count_l16_16280


namespace triangle_side_length_l16_16352

theorem triangle_side_length (a b p : ℝ) (H_perimeter : a + b + 10 = p) (H_a : a = 7) (H_b : b = 15) (H_p : p = 32) : 10 = 10 :=
by
  sorry

end triangle_side_length_l16_16352


namespace solve_system1_solve_system2_l16_16267

theorem solve_system1 (x y : ℝ) (h1 : y = x - 4) (h2 : x + y = 6) : x = 5 ∧ y = 1 :=
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x + y = 1) (h2 : 4 * x - y = 5) : x = 1 ∧ y = -1 :=
by sorry

end solve_system1_solve_system2_l16_16267


namespace quadratic_inequality_ab_l16_16783

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set -1 < x < 1/3,
    prove that ab = 6. -/
theorem quadratic_inequality_ab (a b : ℝ) (h1 : ∀ x, -1 < x ∧ x < 1 / 3 → a * x ^ 2 + b * x + 1 > 0):
  a * b = 6 := 
sorry

end quadratic_inequality_ab_l16_16783


namespace smallest_repeating_block_of_5_over_13_l16_16047

theorem smallest_repeating_block_of_5_over_13 : 
  ∃ n, n = 6 ∧ (∃ m, (5 / 13 : ℚ) = (m/(10^6) : ℚ) ) := 
sorry

end smallest_repeating_block_of_5_over_13_l16_16047


namespace problem1_problem2_l16_16481

theorem problem1 : -24 - (-15) + (-1) + (-15) = -25 := 
by 
  sorry

theorem problem2 : -27 / (3 / 2) * (2 / 3) = -12 := 
by 
  sorry

end problem1_problem2_l16_16481


namespace sum_of_ages_is_l16_16156

-- Define the ages of the triplets and twins
def age_triplet (x : ℕ) := x
def age_twin (x : ℕ) := x - 3

-- Define the total age sum
def total_age_sum (x : ℕ) := 3 * age_triplet x + 2 * age_twin x

-- State the theorem
theorem sum_of_ages_is (x : ℕ) (h : total_age_sum x = 89) : ∃ x : ℕ, total_age_sum x = 89 := 
sorry

end sum_of_ages_is_l16_16156


namespace roger_remaining_debt_is_correct_l16_16976

def house_price : ℝ := 100000
def down_payment_rate : ℝ := 0.20
def parents_payment_rate : ℝ := 0.30

def remaining_debt (house_price down_payment_rate parents_payment_rate : ℝ) : ℝ :=
  let down_payment := house_price * down_payment_rate
  let remaining_balance_after_down_payment := house_price - down_payment
  let parents_payment := remaining_balance_after_down_payment * parents_payment_rate
  remaining_balance_after_down_payment - parents_payment

theorem roger_remaining_debt_is_correct :
  remaining_debt house_price down_payment_rate parents_payment_rate = 56000 :=
by sorry

end roger_remaining_debt_is_correct_l16_16976


namespace roots_of_unity_expression_l16_16534

-- Defining the complex cube roots of unity
def omega := Complex.exp (2 * Real.pi * Complex.I / 3)
def omega2 := Complex.exp (-2 * Real.pi * Complex.I / 3)

-- Main theorem statement to prove
theorem roots_of_unity_expression :
  ((-1 + Complex.i * Real.sqrt 3) / 2) ^ 12 + ((-1 - Complex.i * Real.sqrt 3) / 2) ^ 12 = 2 :=
by
  -- Definitions of the cube roots and their properties
  have h1 : omega ^ 3 = 1 := sorry
  have h2 : omega2 ^ 3 = 1 := sorry
  have h3 : (-1 + Complex.i * Real.sqrt 3) / 2 = omega := sorry
  have h4 : (-1 - Complex.i * Real.sqrt 3) / 2 = omega2 := sorry
  -- Using the properties of the roots and their definitions to prove the statement
  sorry

end roots_of_unity_expression_l16_16534


namespace solve_fractional_equation_l16_16539

-- Define the fractional equation as a function
def fractional_equation (x : ℝ) : Prop :=
  (3 / 2) - (2 * x) / (3 * x - 1) = 7 / (6 * x - 2)

-- State the theorem we need to prove
theorem solve_fractional_equation : fractional_equation 2 :=
by
  -- Placeholder for proof
  sorry

end solve_fractional_equation_l16_16539


namespace difference_of_two_numbers_l16_16845

theorem difference_of_two_numbers (a b : ℕ) (h₀ : a + b = 25800) (h₁ : b = 12 * a) (h₂ : b % 10 = 0) (h₃ : b / 10 = a) : b - a = 21824 :=
by 
  -- sorry to skip the proof
  sorry

end difference_of_two_numbers_l16_16845


namespace circle_radius_formula_correct_l16_16499

noncomputable def touch_circles_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  let numerator := c * Real.sqrt ((s - a) * (s - b) * (s - c))
  let denominator := c * Real.sqrt s + 2 * Real.sqrt ((s - a) * (s - b) * (s - c))
  numerator / denominator

theorem circle_radius_formula_correct (a b c : ℝ) : 
  let s := (a + b + c) / 2
  let Δ := (s - a) * (s - b) * (s - c)
  ∀ (r : ℝ), (r = touch_circles_radius a b c) :=
sorry

end circle_radius_formula_correct_l16_16499


namespace steve_halfway_time_longer_l16_16307

theorem steve_halfway_time_longer :
  ∀ (Td: ℝ) (Ts: ℝ),
  Td = 33 →
  Ts = 2 * Td →
  (Ts / 2) - (Td / 2) = 16.5 :=
by
  intros Td Ts hTd hTs
  rw [hTd, hTs]
  sorry

end steve_halfway_time_longer_l16_16307


namespace tanya_efficiency_greater_sakshi_l16_16260

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end tanya_efficiency_greater_sakshi_l16_16260


namespace peaches_left_at_stand_l16_16249

def initial_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def spoiled_peaches : ℝ := 12.0
def sold_peaches : ℝ := 27.0

theorem peaches_left_at_stand :
  initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 81.0 :=
by
  -- initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 84.0
  sorry

end peaches_left_at_stand_l16_16249


namespace alex_baked_cherry_pies_l16_16897

theorem alex_baked_cherry_pies (total_pies : ℕ) (ratio_apple : ℕ) (ratio_blueberry : ℕ) (ratio_cherry : ℕ)
  (h1 : total_pies = 30)
  (h2 : ratio_apple = 1)
  (h3 : ratio_blueberry = 5)
  (h4 : ratio_cherry = 4) :
  (total_pies * ratio_cherry / (ratio_apple + ratio_blueberry + ratio_cherry) = 12) :=
by {
  sorry
}

end alex_baked_cherry_pies_l16_16897


namespace expected_value_correct_l16_16469

-- Define the probabilities
def prob_8 : ℚ := 3 / 8
def prob_other : ℚ := 5 / 56 -- Derived from the solution steps but using only given conditions explicitly.

-- Define the expected value calculation
def expected_value_die : ℚ :=
  (1 * prob_other) + (2 * prob_other) + (3 * prob_other) + (4 * prob_other) +
  (5 * prob_other) + (6 * prob_other) + (7 * prob_other) + (8 * prob_8)

-- The theorem to prove
theorem expected_value_correct : expected_value_die = 77 / 14 := by
  sorry

end expected_value_correct_l16_16469


namespace rectangular_solid_surface_area_l16_16907

theorem rectangular_solid_surface_area (a b c : ℕ) (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c) 
  (volume_eq : a * b * c = 273) :
  2 * (a * b + b * c + c * a) = 302 := 
sorry

end rectangular_solid_surface_area_l16_16907


namespace average_speed_of_train_l16_16032

-- Condition: Distance traveled is 42 meters
def distance : ℕ := 42

-- Condition: Time taken is 6 seconds
def time : ℕ := 6

-- Average speed computation
theorem average_speed_of_train : distance / time = 7 := by
  -- Left to the prover
  sorry

end average_speed_of_train_l16_16032


namespace ratio_of_ages_l16_16104

variable (J L M : ℕ)

def louis_age := L = 14
def matilda_age := M = 35
def matilda_older := M = J + 7
def jerica_multiple := ∃ k : ℕ, J = k * L

theorem ratio_of_ages
  (hL : louis_age L)
  (hM : matilda_age M)
  (hMO : matilda_older J M)
  : J / L = 2 :=
by
  sorry

end ratio_of_ages_l16_16104


namespace arrow_reading_l16_16417

-- Define the interval and values within it
def in_range (x : ℝ) : Prop := 9.75 ≤ x ∧ x ≤ 10.00
def closer_to_990 (x : ℝ) : Prop := |x - 9.90| < |x - 9.875|

-- The main theorem statement expressing the problem
theorem arrow_reading (x : ℝ) (hx1 : in_range x) (hx2 : closer_to_990 x) : x = 9.90 :=
by sorry

end arrow_reading_l16_16417


namespace original_number_of_boys_l16_16736

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 135 = (n + 3) * 36) : 
  n = 27 := 
by 
  sorry

end original_number_of_boys_l16_16736


namespace latus_rectum_of_parabola_l16_16793

theorem latus_rectum_of_parabola (p : ℝ) (hp : 0 < p) (A : ℝ × ℝ) (hA : A = (1, 1/2)) :
  ∃ a : ℝ, y^2 = 4 * a * x → A.2 ^ 2 = 4 * a * A.1 → x = -1 / (4 * a) → x = -1 / 16 :=
by
  sorry

end latus_rectum_of_parabola_l16_16793


namespace total_cost_l16_16815

def cost_burger := 5
def cost_sandwich := 4
def cost_smoothie := 4
def count_smoothies := 2

theorem total_cost :
  cost_burger + cost_sandwich + count_smoothies * cost_smoothie = 17 :=
by
  sorry

end total_cost_l16_16815


namespace simplify_expression_l16_16479

theorem simplify_expression :
  (sqrt 5 * 5^(1/2) + 18 / 3 * 4 - 8^(3/2) + 10 - 3^2) = 30 - 16 * sqrt 2 := 
by
  sorry

end simplify_expression_l16_16479


namespace gcd_three_numbers_l16_16134

theorem gcd_three_numbers (a b c : ℕ) (h1 : a = 72) (h2 : b = 120) (h3 : c = 168) :
  Nat.gcd (Nat.gcd a b) c = 24 :=
by
  rw [h1, h2, h3]
  exact sorry

end gcd_three_numbers_l16_16134


namespace football_team_starting_lineup_count_l16_16121

theorem football_team_starting_lineup_count :
  let total_members := 12
  let offensive_lineman_choices := 4
  let quarterback_choices := 2
  let remaining_after_ol := total_members - 1 -- after choosing one offensive lineman
  let remaining_after_qb := remaining_after_ol - 1 -- after choosing one quarterback
  let running_back_choices := remaining_after_ol
  let wide_receiver_choices := remaining_after_qb - 1
  let tight_end_choices := remaining_after_qb - 2
  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 5760 := 
by
  sorry

end football_team_starting_lineup_count_l16_16121


namespace number_of_boys_is_50_l16_16016

-- Definitions for conditions:
def total_students : Nat := 100
def boys (x : Nat) : Nat := x
def girls (x : Nat) : Nat := x

-- Theorem statement:
theorem number_of_boys_is_50 (x : Nat) (g : Nat) (h1 : x + g = total_students) (h2 : g = boys x) : boys x = 50 :=
by
  sorry

end number_of_boys_is_50_l16_16016


namespace books_read_by_Megan_l16_16970

theorem books_read_by_Megan 
    (M : ℕ)
    (Kelcie : ℕ := M / 4)
    (Greg : ℕ := 2 * (M / 4) + 9)
    (total : M + Kelcie + Greg = 65) :
  M = 32 :=
by sorry

end books_read_by_Megan_l16_16970


namespace n_n_plus_1_divisible_by_2_l16_16472

theorem n_n_plus_1_divisible_by_2 (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 99) : (n * (n + 1)) % 2 = 0 := 
sorry

end n_n_plus_1_divisible_by_2_l16_16472


namespace comprehensive_survey_l16_16446

def suitable_for_census (s: String) : Prop := 
  s = "Surveying the heights of all classmates in the class"

theorem comprehensive_survey : suitable_for_census "Surveying the heights of all classmates in the class" :=
by
  sorry

end comprehensive_survey_l16_16446


namespace geom_sequence_next_term_l16_16863

def geom_seq (a r : ℕ → ℤ) (i : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * r i

theorem geom_sequence_next_term (y : ℤ) (a : ℕ → ℤ) (r : ℕ → ℤ) (n : ℕ) : 
  geom_seq a r 0 →
  a 0 = 3 →
  a 1 = 9 * y^2 →
  a 2 = 27 * y^4 →
  a 3 = 81 * y^6 →
  r 0 = 3 * y^2 →
  a 4 = 243 * y^8 :=
by
  intro h_seq h1 h2 h3 h4 hr
  sorry

end geom_sequence_next_term_l16_16863


namespace compound_O_atoms_l16_16574

theorem compound_O_atoms (Cu_weight C_weight O_weight compound_weight : ℝ)
  (Cu_atoms : ℕ) (C_atoms : ℕ) (O_atoms : ℕ)
  (hCu : Cu_weight = 63.55)
  (hC : C_weight = 12.01)
  (hO : O_weight = 16.00)
  (h_compound_weight : compound_weight = 124)
  (h_atoms : Cu_atoms = 1 ∧ C_atoms = 1)
  : O_atoms = 3 :=
sorry

end compound_O_atoms_l16_16574


namespace gcd_g_x_l16_16501

noncomputable def g (x : ℕ) : ℕ :=
  (3 * x + 5) * (7 * x + 2) * (13 * x + 7) * (2 * x + 10)

theorem gcd_g_x (x : ℕ) (h : x % 19845 = 0) : Nat.gcd (g x) x = 700 :=
  sorry

end gcd_g_x_l16_16501


namespace factorization_identity_l16_16490

noncomputable def factor_expression (a b c : ℝ) : ℝ :=
  ((a ^ 2 + 1 - (b ^ 2 + 1)) ^ 3 + ((b ^ 2 + 1) - (c ^ 2 + 1)) ^ 3 + ((c ^ 2 + 1) - (a ^ 2 + 1)) ^ 3) /
  ((a - b) ^ 3 + (b - c) ^ 3 + (c - a) ^ 3)

theorem factorization_identity (a b c : ℝ) : 
  factor_expression a b c = (a + b) * (b + c) * (c + a) := 
by 
  sorry

end factorization_identity_l16_16490


namespace expense_of_three_yuan_l16_16378

def isIncome (x : Int) : Prop := x > 0
def isExpense (x : Int) : Prop := x < 0
def incomeOfTwoYuan : Int := 2

theorem expense_of_three_yuan : isExpense (-3) :=
by
  -- Assuming the conditions:
  -- Income is positive: isIncome incomeOfTwoYuan (which is 2)
  -- Expenses are negative
  -- Expenses of 3 yuan should be denoted as -3 yuan
  sorry

end expense_of_three_yuan_l16_16378


namespace a_is_perfect_square_l16_16670

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l16_16670


namespace blanch_slices_eaten_for_dinner_l16_16585

theorem blanch_slices_eaten_for_dinner :
  ∀ (total_slices eaten_breakfast eaten_lunch eaten_snack slices_left eaten_dinner : ℕ),
  total_slices = 15 →
  eaten_breakfast = 4 →
  eaten_lunch = 2 →
  eaten_snack = 2 →
  slices_left = 2 →
  eaten_dinner = total_slices - (eaten_breakfast + eaten_lunch + eaten_snack) - slices_left →
  eaten_dinner = 5 := by
  intros total_slices eaten_breakfast eaten_lunch eaten_snack slices_left eaten_dinner
  intros h_total_slices h_eaten_breakfast h_eaten_lunch h_eaten_snack h_slices_left h_eaten_dinner
  rw [h_total_slices, h_eaten_breakfast, h_eaten_lunch, h_eaten_snack, h_slices_left] at h_eaten_dinner
  exact h_eaten_dinner

end blanch_slices_eaten_for_dinner_l16_16585


namespace min_CD_squared_diff_l16_16667

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3)) + (Real.sqrt (y + 6)) + (Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2)) + (Real.sqrt (y + 2)) + (Real.sqrt (z + 2))
noncomputable def f (x y z : ℝ) : ℝ := (C x y z) ^ 2 - (D x y z) ^ 2

theorem min_CD_squared_diff (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  f x y z ≥ 41.4736 :=
sorry

end min_CD_squared_diff_l16_16667


namespace general_term_formula_l16_16504

variable {a : ℕ → ℝ} -- Define the sequence as a function ℕ → ℝ

-- Conditions
axiom geom_seq (n : ℕ) (h : n ≥ 2): a (n + 1) = a 2 * (2 : ℝ) ^ (n - 1)
axiom a2_eq_2 : a 2 = 2
axiom a3_a4_cond : 2 * a 3 + a 4 = 16

theorem general_term_formula (n : ℕ) : a n = 2 ^ (n - 1) := by
  sorry -- Proof is not required

end general_term_formula_l16_16504


namespace fraction_to_decimal_l16_16023

theorem fraction_to_decimal : (9 : ℚ) / 25 = 0.36 :=
by
  sorry

end fraction_to_decimal_l16_16023


namespace ratio_dark_blue_to_total_l16_16609

-- Definitions based on the conditions
def total_marbles := 63
def red_marbles := 38
def green_marbles := 4
def dark_blue_marbles := total_marbles - red_marbles - green_marbles

-- The statement to be proven
theorem ratio_dark_blue_to_total : (dark_blue_marbles : ℚ) / total_marbles = 1 / 3 := by
  sorry

end ratio_dark_blue_to_total_l16_16609


namespace max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l16_16110

def b_n (n : ℕ) : ℤ := (10 ^ n - 9) / 3
def e_n (n : ℕ) : ℤ := Int.gcd (b_n n) (b_n (n + 1))

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, e_n n ≤ 3 :=
by
  -- Provide the proof here
  sorry

theorem max_possible_value_of_e_n : ∃ n : ℕ, e_n n = 3 :=
by
  -- Provide the proof here
  sorry

end max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l16_16110


namespace smallest_positive_multiple_of_6_and_15_gt_40_l16_16914

-- Define the LCM function to compute the least common multiple
def lcm (m n : ℕ) : ℕ := m * n / Nat.gcd m n

-- Define the statement of the proof problem
theorem smallest_positive_multiple_of_6_and_15_gt_40 : 
  ∃ a : ℕ, (a % 6 = 0) ∧ (a % 15 = 0) ∧ (a > 40) ∧ (∀ b : ℕ, (b % 6 = 0) ∧ (b % 15 = 0) ∧ (b > 40) → a ≤ b) :=
sorry

end smallest_positive_multiple_of_6_and_15_gt_40_l16_16914


namespace sqrt_expr_equals_sum_l16_16909

theorem sqrt_expr_equals_sum :
  ∃ x y z : ℤ,
    (x + y * Int.sqrt z = Real.sqrt (77 + 28 * Real.sqrt 3)) ∧
    (x^2 + y^2 * z = 77) ∧
    (2 * x * y = 28) ∧
    (x + y + z = 16) :=
by
  sorry

end sqrt_expr_equals_sum_l16_16909


namespace egg_production_l16_16833

theorem egg_production (n_chickens1 n_chickens2 n_eggs1 n_eggs2 n_days1 n_days2 : ℕ)
  (h1 : n_chickens1 = 6) (h2 : n_eggs1 = 30) (h3 : n_days1 = 5) (h4 : n_chickens2 = 10) (h5 : n_days2 = 8) :
  n_eggs2 = 80 :=
sorry

end egg_production_l16_16833


namespace arithmetic_mean_positive_two_digit_multiples_of_8_l16_16859

theorem arithmetic_mean_positive_two_digit_multiples_of_8 : 
  let sequence := list.range (11) |>.map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  mean = 56 :=
by
  let sequence := (list.range (11)).map (λ n => 16 + 8 * n)
  let sum_seq := sequence.foldl (+) 0
  let mean := sum_seq / sequence.length
  sorry

end arithmetic_mean_positive_two_digit_multiples_of_8_l16_16859


namespace sum_m_n_zero_l16_16922

theorem sum_m_n_zero (m n p : ℝ) (h1 : mn + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 :=
sorry

end sum_m_n_zero_l16_16922


namespace titmice_all_on_one_tree_l16_16716

-- Define the problem conditions
def titmice (n : ℕ) (m : ℕ) := List (Fin n) 

-- Define the bird movement mechanics
def move_titmouse (config : titmice 2021 120) (i j : Fin 120) : titmice 2021 120 :=
  if config.get? i > config.get? j then config else sorry

-- Define the finite number of moves
def finite_moves : nat := sorry

-- Define the proof problem
theorem titmice_all_on_one_tree :
  ∀ config : titmice 2021 120,
  ∃ moves : nat, 
  ∃ final_config : titmice 2021 120, 
  (∀ i j : Fin 120, final_config.get? i = final_config.get? j) :=
begin
   sorry  
end

end titmice_all_on_one_tree_l16_16716


namespace perfect_square_condition_l16_16676

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l16_16676


namespace find_sides_of_triangle_ABC_find_angle_A_l16_16209

variable (a b c A B C : ℝ)

-- Part (Ⅰ)
theorem find_sides_of_triangle_ABC
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hArea : 1/2 * a * b * Real.sin (Real.pi / 3) = Real.sqrt 3) :
  a = 2 ∧ b = 2 := sorry

-- Part (Ⅱ)
theorem find_angle_A
  (hC : C = Real.pi / 3)
  (hc : c = 2)
  (hTrig : Real.sin C + Real.sin (B - A) = 2 * Real.sin (2 * A)) :
  A = Real.pi / 2 ∨ A = Real.pi / 6 := sorry

end find_sides_of_triangle_ABC_find_angle_A_l16_16209


namespace problem1_problem2_l16_16217

variable (a : ℝ)

def quadratic_roots (a x : ℝ) : Prop := a*x^2 + 2*x + 1 = 0

-- Problem 1: If 1/2 is a root, find the set A
theorem problem1 (h : quadratic_roots a (1/2)) : 
  {x : ℝ | quadratic_roots (a) x } = { -1/4, 1/2 } :=
sorry

-- Problem 2: If A contains exactly one element, find the set B consisting of such a
theorem problem2 (h : ∃! (x : ℝ), quadratic_roots a x ) : 
  {a : ℝ | ∃! (x : ℝ), quadratic_roots a x} = { 0, 1 } :=
sorry

end problem1_problem2_l16_16217


namespace sum_of_interior_angles_of_pentagon_l16_16284

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  -- We skip the proof as per instruction
  sorry

end sum_of_interior_angles_of_pentagon_l16_16284


namespace tom_and_eva_children_count_l16_16232

theorem tom_and_eva_children_count (karen_donald_children : ℕ)
  (total_legs_in_pool : ℕ) (people_not_in_pool : ℕ) 
  (total_legs_each_person : ℕ) (karen_donald : ℕ) (tom_eva : ℕ) 
  (total_people_in_pool : ℕ) (total_people : ℕ) :
  karen_donald_children = 6 ∧ total_legs_in_pool = 16 ∧ people_not_in_pool = 6 ∧ total_legs_each_person = 2 ∧
  karen_donald = 2 ∧ tom_eva = 2 ∧ total_people_in_pool = total_legs_in_pool / total_legs_each_person ∧ 
  total_people = total_people_in_pool + people_not_in_pool ∧ 
  total_people - (karen_donald + karen_donald_children + tom_eva) = 4 :=
by
  intros
  sorry

end tom_and_eva_children_count_l16_16232


namespace subset_relation_l16_16218

variables (M N : Set ℕ) 

theorem subset_relation (hM : M = {1, 2, 3, 4}) (hN : N = {2, 3, 4}) : N ⊆ M :=
sorry

end subset_relation_l16_16218


namespace roger_remaining_debt_is_correct_l16_16977

def house_price : ℝ := 100000
def down_payment_rate : ℝ := 0.20
def parents_payment_rate : ℝ := 0.30

def remaining_debt (house_price down_payment_rate parents_payment_rate : ℝ) : ℝ :=
  let down_payment := house_price * down_payment_rate
  let remaining_balance_after_down_payment := house_price - down_payment
  let parents_payment := remaining_balance_after_down_payment * parents_payment_rate
  remaining_balance_after_down_payment - parents_payment

theorem roger_remaining_debt_is_correct :
  remaining_debt house_price down_payment_rate parents_payment_rate = 56000 :=
by sorry

end roger_remaining_debt_is_correct_l16_16977


namespace smallest_possible_input_l16_16024

def F (n : ℕ) := 9 * n + 120

theorem smallest_possible_input : ∃ n : ℕ, n > 0 ∧ F n = 129 :=
by {
  -- Here we would provide the proof steps, but we use sorry for now.
  sorry
}

end smallest_possible_input_l16_16024


namespace smallest_of_seven_even_numbers_l16_16270

theorem smallest_of_seven_even_numbers (a b c d e f g : ℕ) 
  (h1 : a % 2 = 0) 
  (h2 : b = a + 2) 
  (h3 : c = a + 4) 
  (h4 : d = a + 6) 
  (h5 : e = a + 8) 
  (h6 : f = a + 10) 
  (h7 : g = a + 12) 
  (h_sum : a + b + c + d + e + f + g = 700) : 
  a = 94 :=
by sorry

end smallest_of_seven_even_numbers_l16_16270


namespace carrots_weight_l16_16096

theorem carrots_weight (carrots_bed1: ℕ) (carrots_bed2: ℕ) (carrots_bed3: ℕ) (carrots_per_pound: ℕ)
  (h_bed1: carrots_bed1 = 55)
  (h_bed2: carrots_bed2 = 101)
  (h_bed3: carrots_bed3 = 78)
  (h_c_per_p: carrots_per_pound = 6) :
  (carrots_bed1 + carrots_bed2 + carrots_bed3) / carrots_per_pound = 39 := by
  sorry

end carrots_weight_l16_16096


namespace nat_perfect_square_l16_16681

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l16_16681


namespace max_distance_l16_16396

-- Given the definition of the ellipse
def ellipse (x y : ℝ) := x^2 / 5 + y^2 = 1

-- The upper vertex
def upperVertex : ℝ × ℝ := (0, 1)

-- A point P on the ellipse
def pointOnEllipse (θ : ℝ) : ℝ × ℝ := (sqrt 5 * cos θ, sin θ)

-- The distance function
def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The maximum distance from the point P to the upper vertex B
theorem max_distance (θ : ℝ) :
  let P := pointOnEllipse θ in
  let B := upperVertex in
  P ∈ {p : ℝ × ℝ | ellipse p.1 p.2} →
  ∃ θ, distance P B = 5 / 2 :=
by
  sorry

end max_distance_l16_16396


namespace positive_difference_abs_eq_l16_16007

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l16_16007


namespace probability_at_least_one_8_rolled_l16_16158

theorem probability_at_least_one_8_rolled :
  let total_outcomes := 64
  let no_8_outcomes := 49
  (total_outcomes - no_8_outcomes) / total_outcomes = 15 / 64 :=
by
  let total_outcomes := 8 * 8
  let no_8_outcomes := 7 * 7
  have h1 : total_outcomes = 64 := by norm_num
  have h2 : no_8_outcomes = 49 := by norm_num
  rw [← h1, ← h2]
  norm_num
  sorry

end probability_at_least_one_8_rolled_l16_16158


namespace eddies_sister_pies_per_day_l16_16908

theorem eddies_sister_pies_per_day 
  (Eddie_daily : ℕ := 3) 
  (Mother_daily : ℕ := 8) 
  (total_days : ℕ := 7)
  (total_pies : ℕ := 119) :
  ∃ (S : ℕ), S = 6 ∧ (Eddie_daily * total_days + Mother_daily * total_days + S * total_days = total_pies) :=
by
  sorry

end eddies_sister_pies_per_day_l16_16908


namespace trapezoid_area_possible_l16_16769

def lengths : List ℕ := [1, 4, 4, 5]

theorem trapezoid_area_possible (l₁ l₂ l₃ l₄ : ℕ) (h : List.mem l₁ lengths ∧ List.mem l₂ lengths ∧ List.mem l₃ lengths ∧ List.mem l₄ lengths) :
  (l₁ = 1 ∨ l₁ = 4 ∨ l₁ = 5) ∧ (l₂ = 1 ∨ l₂ = 4 ∨ l₂ = 5) ∧ (l₃ = 1 ∨ l₃ = 4 ∨ l₃ = 5) ∧ (l₄ = 1 ∨ l₄ = 4 ∨ l₄ = 5) →
  (∃ (area : ℕ), area = 6 ∨ area = 10) :=
by
  sorry

end trapezoid_area_possible_l16_16769


namespace norma_found_cards_l16_16252

/-- Assume Norma originally had 88.0 cards. -/
def original_cards : ℝ := 88.0

/-- Assume Norma now has a total of 158 cards. -/
def total_cards : ℝ := 158

/-- Prove that Norma found 70 cards. -/
theorem norma_found_cards : total_cards - original_cards = 70 := 
by
  sorry

end norma_found_cards_l16_16252


namespace range_of_a_l16_16371

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3 / 2 ≤ a ∧ a < 3) :=
by
  sorry

end range_of_a_l16_16371


namespace difference_in_pennies_l16_16332

theorem difference_in_pennies (p : ℤ) : 
  let alice_nickels := 3 * p + 2
  let bob_nickels := 2 * p + 6
  let difference_nickels := alice_nickels - bob_nickels
  let difference_in_pennies := difference_nickels * 5
  difference_in_pennies = 5 * p - 20 :=
by
  sorry

end difference_in_pennies_l16_16332


namespace multiply_expression_l16_16163

variable (y : ℝ)

theorem multiply_expression : 
  (16 * y^3) * (12 * y^5) * (1 / (4 * y)^3) = 3 * y^5 := by
  sorry

end multiply_expression_l16_16163


namespace find_m_for_parallel_lines_l16_16622

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 6 * x + m * y - 1 = 0 ↔ 2 * x - y + 1 = 0) → m = -3 :=
by
  sorry

end find_m_for_parallel_lines_l16_16622


namespace find_n_l16_16377

theorem find_n (n : ℤ) (h : 8 + 6 = n + 8) : n = 6 :=
by
  sorry

end find_n_l16_16377


namespace degree_to_radian_60_eq_pi_div_3_l16_16299

theorem degree_to_radian_60_eq_pi_div_3 (pi : ℝ) (deg : ℝ) 
  (h : 180 * deg = pi) : 60 * deg = pi / 3 := 
by
  sorry

end degree_to_radian_60_eq_pi_div_3_l16_16299


namespace compute_difference_of_reciprocals_l16_16061

theorem compute_difference_of_reciprocals
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = x / y) :
  (1 / x) - (1 / y) = - (1 / y^2) :=
by
  sorry

end compute_difference_of_reciprocals_l16_16061


namespace arithmetic_mean_of_two_digit_multiples_of_8_l16_16855

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let a := 16,
      l := 96,
      d := 8,
      n := 11,
      S_n := 616 in
  (2 * S_n) / n = 56 :=
by
  let a := 16
  let l := 96
  let d := 8
  let n := 11
  let S_n := 616
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l16_16855


namespace cr_inequality_l16_16829

theorem cr_inequality 
  (a b : ℝ) (r : ℝ)
  (cr : ℝ := if r < 1 then 1 else 2^(r - 1)) 
  (h0 : r ≥ 0) : 
  |a + b|^r ≤ cr * (|a|^r + |b|^r) :=
by 
  sorry

end cr_inequality_l16_16829


namespace simplify_sqrt_expr_l16_16125

-- We need to prove that simplifying √(5 - 2√6) is equal to √3 - √2.
theorem simplify_sqrt_expr : 
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 :=
by 
  sorry

end simplify_sqrt_expr_l16_16125


namespace cord_length_before_cut_l16_16021

-- Definitions based on the conditions
def parts_after_cut := 20
def longest_piece := 8
def shortest_piece := 2
def initial_parts := 19

-- Lean statement to prove the length of the cord before it was cut
theorem cord_length_before_cut : 
  (initial_parts * ((longest_piece / 2) + shortest_piece) = 114) :=
by 
  sorry

end cord_length_before_cut_l16_16021


namespace days_y_needs_l16_16737

theorem days_y_needs
  (d : ℝ)
  (h1 : (1:ℝ) / 21 * 14 = 1 - 5 * (1 / d)) :
  d = 10 :=
sorry

end days_y_needs_l16_16737


namespace find_a_l16_16620

theorem find_a (a : ℝ) (h : (a + 3) = 0) : a = -3 :=
by sorry

end find_a_l16_16620


namespace chess_game_probability_l16_16452

theorem chess_game_probability (p_A_wins p_draw : ℝ) (h1 : p_A_wins = 0.3) (h2 : p_draw = 0.2) :
  p_A_wins + p_draw = 0.5 :=
by
  rw [h1, h2]
  norm_num

end chess_game_probability_l16_16452


namespace algebraic_expression_value_l16_16505

theorem algebraic_expression_value (a x : ℝ) (h : 3 * a - x = x + 2) (hx : x = 2) : a^2 - 2 * a + 1 = 1 :=
by {
  sorry
}

end algebraic_expression_value_l16_16505


namespace maximum_buses_l16_16644

-- Definition of the problem constraints as conditions in Lean
variables (bus : Type) (stop : Type) (buses : set bus) (stops : set stop)
variables (stops_served_by_bus : bus → finset stop)
variables (stop_occurrences : stop → finset bus)

-- Conditions
def has_9_stops (stops : set stop) : Prop := stops.card = 9
def bus_serves_3_stops (b : bus) : Prop := (stops_served_by_bus b).card = 3
def at_most_1_common_stop (b1 b2 : bus) (h1 : b1 ∈ buses) (h2 : b2 ∈ buses) : Prop :=
  (stops_served_by_bus b1 ∩ stops_served_by_bus b2).card ≤ 1

-- Goal
theorem maximum_buses (h1 : has_9_stops stops)
                      (h2 : ∀ b ∈ buses, bus_serves_3_stops b)
                      (h3 : ∀ b1 b2 ∈ buses, at_most_1_common_stop b1 b2 h1 h2) : 
  buses.card ≤ 12 :=
sorry

end maximum_buses_l16_16644


namespace percent_calculation_l16_16799

theorem percent_calculation (x : ℝ) (h : 0.30 * 0.40 * x = 24) : 0.20 * 0.60 * x = 24 := 
by
  sorry

end percent_calculation_l16_16799


namespace length_of_woods_l16_16289

theorem length_of_woods (area width : ℝ) (h_area : area = 24) (h_width : width = 8) : (area / width) = 3 :=
by
  sorry

end length_of_woods_l16_16289


namespace exists_solution_iff_l16_16201

theorem exists_solution_iff (m : ℝ) (x y : ℝ) :
  ((y = (3 * m + 2) * x + 1) ∧ (y = (5 * m - 4) * x + 5)) ↔ m ≠ 3 :=
by sorry

end exists_solution_iff_l16_16201


namespace find_M_coordinate_l16_16388

-- Definitions of the given points
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, 0, 2⟩
def B : Point3D := ⟨1, -3, 1⟩
def M (y : ℝ) : Point3D := ⟨0, y, 0⟩

-- Definition for the squared distance between two points
def dist_sq (p1 p2 : Point3D) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2

-- Main theorem statement
theorem find_M_coordinate (y : ℝ) : 
  dist_sq (M y) A = dist_sq (M y) B → y = -1 :=
by
  simp [dist_sq, A, B, M]
  sorry

end find_M_coordinate_l16_16388


namespace a_is_perfect_square_l16_16682

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l16_16682


namespace max_profit_l16_16173

noncomputable def C (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then (1 / 3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def L (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x < 80 then -(1 / 3) * x^2 + 40 * x - 250
  else -(x + 10000 / x) + 1200

theorem max_profit :
  ∃ x : ℝ, (L x) = 1000 ∧ x = 100 :=
by
  sorry

end max_profit_l16_16173


namespace total_colors_needed_l16_16650

def num_planets : ℕ := 8
def num_people : ℕ := 3

theorem total_colors_needed : num_people * num_planets = 24 := by
  sorry

end total_colors_needed_l16_16650


namespace candy_in_one_bowl_l16_16339

theorem candy_in_one_bowl (total_candies : ℕ) (eaten_candies : ℕ) (bowls : ℕ) (taken_per_bowl : ℕ) 
  (h1 : total_candies = 100) (h2 : eaten_candies = 8) (h3 : bowls = 4) (h4 : taken_per_bowl = 3) :
  (total_candies - eaten_candies) / bowls - taken_per_bowl = 20 :=
by
  sorry

end candy_in_one_bowl_l16_16339


namespace trapezoid_area_l16_16770

theorem trapezoid_area :
  ∃ S, (S = 6 ∨ S = 10) ∧ 
  ((∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 4 ∧ d = 5 ∧ 
    (∃ (is_isosceles_trapezoid : Prop), is_isosceles_trapezoid)) ∨
   (∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 4 ∧ 
    (∃ (is_right_angled_trapezoid : Prop), is_right_angled_trapezoid)) ∨ 
   (∃ (a b c d : ℝ), (a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1) →
   (∀ (is_impossible_trapezoid : Prop), ¬ is_impossible_trapezoid))) :=
sorry

end trapezoid_area_l16_16770


namespace diplomats_not_speaking_russian_l16_16407

-- Definitions to formalize the problem
def total_diplomats : ℕ := 150
def speak_french : ℕ := 17
def speak_both_french_and_russian : ℕ := (10 * total_diplomats) / 100
def speak_neither_french_nor_russian : ℕ := (20 * total_diplomats) / 100

-- Theorem to prove the desired quantity
theorem diplomats_not_speaking_russian : 
  speak_neither_french_nor_russian + (speak_french - speak_both_french_and_russian) = 32 := by
  sorry

end diplomats_not_speaking_russian_l16_16407


namespace domain_of_function_l16_16056

theorem domain_of_function :
  { x : ℝ // (6 - x - x^2) > 0 } = { x : ℝ // -3 < x ∧ x < 2 } :=
by
  sorry

end domain_of_function_l16_16056


namespace sum_of_interior_angles_of_pentagon_l16_16285

theorem sum_of_interior_angles_of_pentagon : (5 - 2) * 180 = 540 := 
by
  -- We skip the proof as per instruction
  sorry

end sum_of_interior_angles_of_pentagon_l16_16285


namespace smallest_sum_B_d_l16_16630

theorem smallest_sum_B_d :
  ∃ B d : ℕ, (B < 5) ∧ (d > 6) ∧ (125 * B + 25 * B + B = 4 * d + 4) ∧ (B + d = 77) :=
by
  sorry

end smallest_sum_B_d_l16_16630


namespace find_value_of_expression_l16_16607

theorem find_value_of_expression 
  (x y z w : ℤ)
  (hx : x = 3)
  (hy : y = 2)
  (hz : z = 4)
  (hw : w = -1) :
  x^2 * y - 2 * x * y + 3 * x * z - (x + y) * (y + z) * (z + w) = -48 :=
by
  sorry

end find_value_of_expression_l16_16607


namespace total_amount_paid_l16_16244

-- Define the conditions
def chicken_nuggets_ordered : ℕ := 100
def nuggets_per_box : ℕ := 20
def cost_per_box : ℕ := 4

-- Define the hypothesis on the amount of money paid for the chicken nuggets
theorem total_amount_paid :
  (chicken_nuggets_ordered / nuggets_per_box) * cost_per_box = 20 :=
by
  sorry

end total_amount_paid_l16_16244


namespace perfect_square_condition_l16_16674

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l16_16674


namespace coeff_x_binomial_expansion_l16_16199

theorem coeff_x_binomial_expansion :
  ∑ k in finset.range 21, binomial 20 k * (-1) ^ k * x ^ (k / 2) = 190 :=
sorry

end coeff_x_binomial_expansion_l16_16199


namespace symmetric_line_condition_l16_16803

theorem symmetric_line_condition (x y : ℝ) :
  (∀ x y : ℝ, x - 2 * y - 3 = 0 → -y + 2 * x - 3 = 0) →
  (∀ x y : ℝ, x + y = 0 → ∃ a b c : ℝ, 2 * x - y - 3 = 0) :=
sorry

end symmetric_line_condition_l16_16803


namespace triangle_area_l16_16467

theorem triangle_area (a b c : ℕ) (h1 : a + b + c = 12) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) 
                      (h5 : a + b > c) (h6 : b + c > a) (h7 : c + a > b) :
  ∃ A : ℝ, A = 2 * Real.sqrt 6 ∧
    ∃ (h : 0 ≤ A), A = (Real.sqrt (A * 12 * (12 - a) * (12 - b) * (12 - c))) :=
sorry

end triangle_area_l16_16467


namespace total_fruits_purchased_l16_16879

-- Defining the costs of apples and bananas
def cost_per_apple : ℝ := 0.80
def cost_per_banana : ℝ := 0.70

-- Defining the total cost the customer spent
def total_cost : ℝ := 6.50

-- Defining the total number of fruits purchased as 9
theorem total_fruits_purchased (A B : ℕ) : 
  (cost_per_apple * A + cost_per_banana * B = total_cost) → 
  (A + B = 9) :=
by
  sorry

end total_fruits_purchased_l16_16879


namespace sum_last_two_digits_fibonacci_factorial_series_l16_16340

theorem sum_last_two_digits_fibonacci_factorial_series :
  let fib_factorial n := (factorial n) % 100,
      series := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
      last_two_digits := series.map fib_factorial
  in last_two_digits.sum % 100 = 5 :=
by 
  -- Need to define fib_factorial, map over the series,
  -- calculate the sum of last two digits, and prove
  sorry

end sum_last_two_digits_fibonacci_factorial_series_l16_16340


namespace length_of_AB_l16_16527

theorem length_of_AB 
  (AB BC CD AD : ℕ)
  (h1 : AB = 1 * BC / 2)
  (h2 : BC = 6 * CD / 5)
  (h3 : AB + BC + CD = 56)
  : AB = 12 := sorry

end length_of_AB_l16_16527


namespace range_of_a_l16_16365

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + (1 / 2) * Real.log x

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := (2 * a * x^2 + 1) / (2 * x)

def p (a : ℝ) : Prop := ∀ x, 1 ≤ x → f_prime (a) (x) ≤ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1

theorem range_of_a (a : ℝ) : (p a ∧ q a) → -1 < a ∧ a ≤ -1 / 2 :=
by
  sorry

end range_of_a_l16_16365


namespace percentage_failed_hindi_l16_16655

theorem percentage_failed_hindi 
  (F_E F_B P_BE : ℕ) 
  (h₁ : F_E = 42) 
  (h₂ : F_B = 28) 
  (h₃ : P_BE = 56) :
  ∃ F_H, F_H = 30 := 
by
  sorry

end percentage_failed_hindi_l16_16655


namespace slope_of_line_l16_16771

theorem slope_of_line (x y : ℝ) : (∃ (m b : ℝ), (3 * y + 2 * x = 12) ∧ (m = -2 / 3) ∧ (y = m * x + b)) :=
sorry

end slope_of_line_l16_16771


namespace sum_in_base_b_l16_16115

noncomputable def s_in_base (b : ℕ) := 13 + 15 + 17

theorem sum_in_base_b (b : ℕ) (h : (13 * 15 * 17 : ℕ) = 4652) : s_in_base b = 51 := by
  sorry

end sum_in_base_b_l16_16115


namespace dot_product_vec1_vec2_l16_16044

-- Define the vectors
def vec1 := (⟨-4, -1⟩ : ℤ × ℤ)
def vec2 := (⟨6, 8⟩ : ℤ × ℤ)

-- Define the dot product function
def dot_product (v1 v2 : ℤ × ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of vec1 and vec2 is -32
theorem dot_product_vec1_vec2 : dot_product vec1 vec2 = -32 :=
by
  sorry

end dot_product_vec1_vec2_l16_16044


namespace point_P_in_first_quadrant_l16_16953

def lies_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : lies_in_first_quadrant 2 1 :=
by {
  sorry
}

end point_P_in_first_quadrant_l16_16953


namespace red_balls_in_bag_l16_16375

theorem red_balls_in_bag : ∃ x : ℕ, (3 : ℚ) / (4 + (x : ℕ)) = 1 / 2 ∧ x = 2 := sorry

end red_balls_in_bag_l16_16375


namespace real_part_of_z1_is_zero_l16_16839

-- Define the imaginary unit i with its property
def i := Complex.I

-- Define z1 using the given expression
noncomputable def z1 := (1 - 2 * i) / (2 + i^5)

-- State the theorem about the real part of z1
theorem real_part_of_z1_is_zero : z1.re = 0 :=
by
  sorry

end real_part_of_z1_is_zero_l16_16839


namespace rectangle_coloring_problem_l16_16773

theorem rectangle_coloring_problem :
  let n := 3
  let m := 4
  ∃ n, ∃ m, n = 3 ∧ m = 4 := sorry

end rectangle_coloring_problem_l16_16773


namespace polynomial_square_binomial_l16_16195

-- Define the given polynomial and binomial
def polynomial (x : ℚ) (a : ℚ) : ℚ :=
  25 * x^2 + 40 * x + a

def binomial (x b : ℚ) : ℚ :=
  (5 * x + b)^2

-- Theorem to state the problem
theorem polynomial_square_binomial (a : ℚ) : 
  (∃ b, polynomial x a = binomial x b) ↔ a = 16 :=
by
  sorry

end polynomial_square_binomial_l16_16195


namespace determine_a_l16_16624

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem determine_a : (∃ a: ℝ, (∀ x: ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) ∧ ∀ x: ℝ, f x a ≤ 6 → -2 ≤ x ∧ x ≤ 3) ↔ a = 1 :=
by
  sorry

end determine_a_l16_16624


namespace difference_of_squares_l16_16300

theorem difference_of_squares (a b : ℕ) (h1: a = 630) (h2: b = 570) : a^2 - b^2 = 72000 :=
by
  sorry

end difference_of_squares_l16_16300


namespace part1_solution_part2_solution_l16_16243

-- Define the inequality for part (1)
def ineq_part1 (x : ℝ) : Prop := 1 - (4 / (x + 1)) < 0

-- Define the solution set P for part (1)
def P (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Prove that the solution set for the inequality is P
theorem part1_solution :
  ∀ (x : ℝ), ineq_part1 x ↔ P x :=
by
  -- proof omitted
  sorry

-- Define the inequality for part (2)
def ineq_part2 (x : ℝ) : Prop := abs (x + 2) < 3

-- Define the solution set Q for part (2)
def Q (x : ℝ) : Prop := -5 < x ∧ x < 1

-- Define P as depending on some parameter a
def P_param (a : ℝ) (x : ℝ) : Prop := -1 < x ∧ x < a

-- Prove the range of a given P ∪ Q = Q 
theorem part2_solution :
  ∀ a : ℝ, (∀ x : ℝ, (P_param a x ∨ Q x) ↔ Q x) → 
    (0 < a ∧ a ≤ 1) :=
by
  -- proof omitted
  sorry

end part1_solution_part2_solution_l16_16243


namespace find_noon_temperature_l16_16190

theorem find_noon_temperature (T T₄₀₀ T₈₀₀ : ℝ) 
  (h1 : T₄₀₀ = T + 8)
  (h2 : T₈₀₀ = T₄₀₀ - 11)
  (h3 : T₈₀₀ = T + 1) : 
  T = 4 :=
by
  sorry

end find_noon_temperature_l16_16190


namespace total_fast_food_order_cost_l16_16816

def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def smoothies_quantity : ℕ := 2

theorem total_fast_food_order_cost : burger_cost + sandwich_cost + smoothies_quantity * smoothie_cost = 17 := 
by
  sorry

end total_fast_food_order_cost_l16_16816


namespace smaller_tank_capacity_l16_16155

/-- Problem Statement:
Three-quarters of the oil from a certain tank (that was initially full) was poured into a
20000-liter capacity tanker that already had 3000 liters of oil.
To make the large tanker half-full, 4000 more liters of oil would be needed.
What is the capacity of the smaller tank?
-/

theorem smaller_tank_capacity (C : ℝ) 
  (h1 : 3 / 4 * C + 3000 + 4000 = 10000) : 
  C = 4000 :=
sorry

end smaller_tank_capacity_l16_16155


namespace Dhoni_spending_difference_l16_16487

-- Definitions
def RentPercent := 20
def LeftOverPercent := 61
def TotalSpendPercent := 100 - LeftOverPercent
def DishwasherPercent := TotalSpendPercent - RentPercent

-- Theorem statement
theorem Dhoni_spending_difference :
  DishwasherPercent = RentPercent - 1 := 
by
  sorry

end Dhoni_spending_difference_l16_16487


namespace find_function_g_l16_16601

noncomputable def g (x : ℝ) : ℝ := (5^x - 3^x) / 8

theorem find_function_g (x y : ℝ) (h1 : g 2 = 2) (h2 : ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y) :
  g x = (5^x - 3^x) / 8 :=
by
  sorry

end find_function_g_l16_16601


namespace pumps_time_to_empty_pool_l16_16018

theorem pumps_time_to_empty_pool :
  (1 / (1 / 6 + 1 / 9) * 60) = 216 :=
by
  norm_num
  sorry

end pumps_time_to_empty_pool_l16_16018


namespace milk_production_per_cow_l16_16899

theorem milk_production_per_cow :
  ∀ (total_cows : ℕ) (milk_price_per_gallon butter_price_per_stick total_earnings : ℝ)
    (customers customer_milk_demand gallons_per_butter : ℕ),
  total_cows = 12 →
  milk_price_per_gallon = 3 →
  butter_price_per_stick = 1.5 →
  total_earnings = 144 →
  customers = 6 →
  customer_milk_demand = 6 →
  gallons_per_butter = 2 →
  (∀ (total_milk_sold_to_customers produced_milk used_for_butter : ℕ),
    total_milk_sold_to_customers = customers * customer_milk_demand →
    produced_milk = total_milk_sold_to_customers + used_for_butter →
    used_for_butter = (total_earnings - (total_milk_sold_to_customers * milk_price_per_gallon)) / butter_price_per_stick / gallons_per_butter →
    produced_milk / total_cows = 4)
:= by sorry

end milk_production_per_cow_l16_16899


namespace perfect_square_iff_n_eq_one_l16_16776

theorem perfect_square_iff_n_eq_one (n : ℕ) : ∃ m : ℕ, n^2 + 3 * n = m^2 ↔ n = 1 := by
  sorry

end perfect_square_iff_n_eq_one_l16_16776


namespace bridge_length_at_least_200_l16_16275

theorem bridge_length_at_least_200 :
  ∀ (length_train : ℝ) (speed_kmph : ℝ) (time_secs : ℝ),
  length_train = 200 ∧ speed_kmph = 32 ∧ time_secs = 20 →
  ∃ l : ℝ, l ≥ length_train :=
by
  sorry

end bridge_length_at_least_200_l16_16275


namespace arithmetic_sequence_a5_l16_16955

theorem arithmetic_sequence_a5 (a_n : ℕ → ℝ) 
  (h_arith : ∀ n, a_n (n+1) - a_n n = a_n (n+2) - a_n (n+1))
  (h_condition : a_n 1 + a_n 9 = 10) :
  a_n 5 = 5 :=
sorry

end arithmetic_sequence_a5_l16_16955


namespace number_of_fowls_l16_16106

theorem number_of_fowls (chickens : ℕ) (ducks : ℕ) (h1 : chickens = 28) (h2 : ducks = 18) : chickens + ducks = 46 :=
by
  sorry

end number_of_fowls_l16_16106


namespace range_of_a_l16_16711

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x + 5 > 3 ∧ x > a ∧ x ≤ -2) ↔ a ≤ -2 :=
by
  sorry

end range_of_a_l16_16711


namespace inequality_am_gm_l16_16240

theorem inequality_am_gm 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) >= 6 := 
sorry

end inequality_am_gm_l16_16240


namespace not_always_product_greater_l16_16703

-- Define the premise and the conclusion
theorem not_always_product_greater (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b < 1) : a * b < a :=
sorry

end not_always_product_greater_l16_16703


namespace angle_PQC_correct_l16_16205

variables {A B C D P Q : Type}
variables [AffinePlane A B C D P Q]

theorem angle_PQC_correct
  (h1 : is_isosceles_triangle A B C)
  (h2 : ∠ A = 30)
  (h3 : is_midpoint D B C)
  (h4 : ∃ P, lies_on_segment P A D)
  (h5 : ∃ Q, lies_on_segment Q A B)
  (h6 : segment_length P B = segment_length P Q) :
  ∠ PQC = 15 :=
  sorry

end angle_PQC_correct_l16_16205


namespace area_ratio_of_square_side_multiplied_by_10_l16_16168

theorem area_ratio_of_square_side_multiplied_by_10 (s : ℝ) (A_original A_resultant : ℝ) 
  (h1 : A_original = s^2)
  (h2 : A_resultant = (10 * s)^2) :
  (A_original / A_resultant) = (1 / 100) :=
by
  sorry

end area_ratio_of_square_side_multiplied_by_10_l16_16168


namespace smoke_diagram_total_height_l16_16669

theorem smoke_diagram_total_height : 
  ∀ (h1 h2 h3 h4 h5 : ℕ),
    h1 < h2 ∧ h2 < h3 ∧ h3 < h4 ∧ h4 < h5 ∧ 
    (h2 - h1 = 2) ∧ (h3 - h2 = 2) ∧ (h4 - h3 = 2) ∧ (h5 - h4 = 2) ∧ 
    (h5 = h1 + h2) → 
    h1 + h2 + h3 + h4 + h5 = 50 := 
by 
  sorry

end smoke_diagram_total_height_l16_16669


namespace least_possible_mn_correct_l16_16660

def least_possible_mn (m n : ℕ) : ℕ :=
  m + n

theorem least_possible_mn_correct (m n : ℕ) :
  (Nat.gcd (m + n) 210 = 1) →
  (n^n ∣ m^m) →
  ¬(n ∣ m) →
  least_possible_mn m n = 407 :=
by
  sorry

end least_possible_mn_correct_l16_16660


namespace ellipse_equation_l16_16787

theorem ellipse_equation (e : ℝ) (P : ℝ × ℝ) (d_max : ℝ) (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
    (h3 : e = Real.sqrt 3 / 2) (h4 : P = (0, 3 / 2)) (h5 : ∀ P1 : ℝ × ℝ, (P1.1 ^ 2 / a ^ 2 + P1.2 ^ 2 / b ^ 2 = 1) → 
    ∃ P2 : ℝ × ℝ, dist P P2 = d_max ∧ (P2.1 ^ 2 / a ^ 2 + P2.2 ^ 2 / b ^ 2 = 1)) :
  (a = 2 ∧ b = 1) → (∀ x y : ℝ, (x ^ 2 / 4) + y ^ 2 ≤ 1) := by
  sorry

end ellipse_equation_l16_16787


namespace problem_M_m_evaluation_l16_16782

theorem problem_M_m_evaluation
  (a b c d e : ℝ)
  (h : a < b)
  (h' : b < c)
  (h'' : c < d)
  (h''' : d < e)
  (h'''' : a < e) :
  (max (min a (max b c))
       (max (min a d) (max b e))) = e := 
by
  sorry

end problem_M_m_evaluation_l16_16782


namespace find_c_value_l16_16915

theorem find_c_value (c : ℝ)
  (h : 4 * (3.6 * 0.48 * c / (0.12 * 0.09 * 0.5)) = 3200.0000000000005) :
  c = 2.5 :=
by sorry

end find_c_value_l16_16915


namespace solve_exponential_equation_l16_16690

theorem solve_exponential_equation (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3^x + 4^y = 5^z ↔ x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end solve_exponential_equation_l16_16690


namespace Vlad_height_feet_l16_16437

theorem Vlad_height_feet 
  (sister_height_feet : ℕ)
  (sister_height_inches : ℕ)
  (vlad_height_diff : ℕ)
  (vlad_height_inches : ℕ)
  (vlad_height_feet : ℕ)
  (vlad_height_rem : ℕ)
  (sister_height := (sister_height_feet * 12) + sister_height_inches)
  (vlad_height := sister_height + vlad_height_diff)
  (vlad_height_feet_rem := (vlad_height / 12, vlad_height % 12)) 
  (h_sister_height : sister_height_feet = 2)
  (h_sister_height_inches : sister_height_inches = 10)
  (h_vlad_height_diff : vlad_height_diff = 41)
  (h_vlad_height : vlad_height = 75)
  (h_vlad_height_feet : vlad_height_feet = 6)
  (h_vlad_height_rem : vlad_height_rem = 3) :
  vlad_height_feet = 6 := by
  sorry

end Vlad_height_feet_l16_16437


namespace initially_marked_points_l16_16412

theorem initially_marked_points (k : ℕ) (h : 4 * k - 3 = 101) : k = 26 :=
by
  sorry

end initially_marked_points_l16_16412


namespace math_problem_l16_16067

theorem math_problem
  (a b c : ℝ)
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 5 :=
sorry

end math_problem_l16_16067


namespace min_cost_theater_tickets_l16_16573

open Real

variable (x y : ℝ)

theorem min_cost_theater_tickets :
  (x + y = 140) →
  (y ≥ 2 * x) →
  ∀ x y, 60 * x + 100 * y ≥ 12160 :=
by
  sorry

end min_cost_theater_tickets_l16_16573


namespace larry_channels_l16_16963

-- Initial conditions
def init_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def channels_reduce_request : ℕ := 10
def sports_package : ℕ := 8
def supreme_sports_package : ℕ := 7

-- Calculation representing the overall change step-by-step
theorem larry_channels : 
  init_channels - channels_taken_away + channels_replaced - channels_reduce_request + sports_package + supreme_sports_package = 147 :=
by sorry

end larry_channels_l16_16963


namespace arithmetic_mean_two_digit_multiples_of_8_l16_16854

theorem arithmetic_mean_two_digit_multiples_of_8 : 
  (∑ i in finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99) (id) / (finset.filter (λ x, x % 8 = 0) (finset.Icc 10 99)).card) = 56 := 
by sorry

end arithmetic_mean_two_digit_multiples_of_8_l16_16854


namespace sum_m_n_zero_l16_16921

theorem sum_m_n_zero (m n p : ℝ) (h1 : mn + p^2 + 4 = 0) (h2 : m - n = 4) : m + n = 0 :=
sorry

end sum_m_n_zero_l16_16921


namespace percentage_subtraction_l16_16986

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

end percentage_subtraction_l16_16986


namespace range_of_a_is_eight_thirds_to_four_l16_16925

noncomputable def piecewise_f (a : ℝ) (x : ℝ) : ℝ :=
if h : x > 1 then a ^ x else (2 - a / 2) * x + 2

theorem range_of_a_is_eight_thirds_to_four (a : ℝ) :
  (∀ x y : ℝ, x < y → piecewise_f a x < piecewise_f a y) ↔ (8 / 3 ≤ a ∧ a < 4) :=
sorry

end range_of_a_is_eight_thirds_to_four_l16_16925


namespace gcd_three_numbers_l16_16133

theorem gcd_three_numbers (a b c : ℕ) (h1 : a = 72) (h2 : b = 120) (h3 : c = 168) :
  Nat.gcd (Nat.gcd a b) c = 24 :=
by
  rw [h1, h2, h3]
  exact sorry

end gcd_three_numbers_l16_16133


namespace max_a_value_l16_16220

theorem max_a_value (a : ℝ) :
  (∀ x : ℝ, x < a → x^2 - 2 * x - 3 > 0) →
  (¬ (∀ x : ℝ, x^2 - 2 * x - 3 > 0 → x < a)) →
  a = -1 :=
by
  sorry

end max_a_value_l16_16220


namespace wise_men_correct_guesses_l16_16293

noncomputable def max_correct_guesses (n k : ℕ) : ℕ :=
  if n > k + 1 then n - k - 1 else 0

theorem wise_men_correct_guesses (n k : ℕ) :
  ∃ (m : ℕ), m = max_correct_guesses n k ∧ m ≤ n - k - 1 :=
by {
  sorry
}

end wise_men_correct_guesses_l16_16293


namespace fourth_powers_sum_is_8432_l16_16486

def sum_fourth_powers (x y : ℝ) : ℝ := x^4 + y^4

theorem fourth_powers_sum_is_8432 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x * y = 4) : 
  sum_fourth_powers x y = 8432 :=
by
  sorry

end fourth_powers_sum_is_8432_l16_16486


namespace arithmetic_mean_of_two_digit_multiples_of_8_is_56_l16_16858

noncomputable def arithmetic_mean_of_two_digit_multiples_of_8 : ℕ :=
  let a := 16 in -- first term
  let l := 96 in -- last term
  let d := 8 in  -- common difference
  let n := 11 in -- number of terms
  let sum := (n / 2) * (a + l) in -- sum of terms
  sum / n -- arithmetic mean

theorem arithmetic_mean_of_two_digit_multiples_of_8_is_56 :
  arithmetic_mean_of_two_digit_multiples_of_8 = 56 :=
sorry

end arithmetic_mean_of_two_digit_multiples_of_8_is_56_l16_16858


namespace chord_eq_l16_16635

/-- 
If a chord of the ellipse x^2 / 36 + y^2 / 9 = 1 is bisected by the point (4,2),
then the equation of the line on which this chord lies is x + 2y - 8 = 0.
-/
theorem chord_eq {x y : ℝ} (H : ∃ A B : ℝ × ℝ, 
  (A.1 ^ 2 / 36 + A.2 ^ 2 / 9 = 1) ∧ 
  (B.1 ^ 2 / 36 + B.2 ^ 2 / 9 = 1) ∧ 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (4, 2)) :
  x + 2 * y = 8 :=
sorry

end chord_eq_l16_16635


namespace probability_bypass_kth_intersection_l16_16458

variable (n k : ℕ)

def P (n k : ℕ) : ℚ := (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2

theorem probability_bypass_kth_intersection :
  P n k = (2 * k * n - 2 * k^2 + 2 * k - 1) / n^2 :=
by
  sorry

end probability_bypass_kth_intersection_l16_16458


namespace part1_part2_l16_16933

-- Define the function f(x) = |x - 1| + |x - 2|
def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

-- Prove the statement about f(x) and the inequality
theorem part1 : { x : ℝ | (2 / 3) ≤ x ∧ x ≤ 4 } ⊆ { x : ℝ | f x ≤ x + 1 } :=
sorry

-- State k = 1 as the minimum value of f(x)
def k : ℝ := 1

-- Prove the non-existence of positive a and b satisfying the given conditions
theorem part2 : ¬ ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ 2 * a + b = k ∧ (1 / a + 2 / b = 4) :=
sorry

end part1_part2_l16_16933


namespace original_selling_price_is_1100_l16_16757

-- Let P be the original purchase price.
variable (P : ℝ)

-- Condition 1: Bill made a profit of 10% on the original purchase price.
def original_selling_price := 1.10 * P

-- Condition 2: If he had purchased that product for 10% less 
-- and sold it at a profit of 30%, he would have received $70 more.
def new_purchase_price := 0.90 * P
def new_selling_price := 1.17 * P
def price_difference := new_selling_price - original_selling_price

-- Theorem: The original selling price was $1100.
theorem original_selling_price_is_1100 (h : price_difference P = 70) : 
  original_selling_price P = 1100 :=
sorry

end original_selling_price_is_1100_l16_16757


namespace probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l16_16328

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 1
def total_outcomes_plan1 := 15
def winning_outcomes_plan1 := 6
def probability_plan1 : ℚ := winning_outcomes_plan1 / total_outcomes_plan1

-- Definition of the total number of outcomes and outcomes where a player wins for Plan 2
def total_outcomes_plan2 := 36
def winning_outcomes_plan2 := 11
def probability_plan2 : ℚ := winning_outcomes_plan2 / total_outcomes_plan2

-- Statements to prove
theorem probability_of_winning_plan1_is_2_over_5 : probability_plan1 = 2 / 5 :=
by sorry

theorem probability_of_winning_plan2_is_11_over_36 : probability_plan2 = 11 / 36 :=
by sorry

theorem choose_plan1 : probability_plan1 > probability_plan2 :=
by sorry

end probability_of_winning_plan1_is_2_over_5_probability_of_winning_plan2_is_11_over_36_choose_plan1_l16_16328


namespace students_can_be_helped_on_fourth_day_l16_16889

theorem students_can_be_helped_on_fourth_day : 
  ∀ (total_books first_day_students second_day_students third_day_students books_per_student : ℕ),
  total_books = 120 →
  first_day_students = 4 →
  second_day_students = 5 →
  third_day_students = 6 →
  books_per_student = 5 →
  (total_books - (first_day_students * books_per_student + second_day_students * books_per_student + third_day_students * books_per_student)) / books_per_student = 9 :=
by
  intros total_books first_day_students second_day_students third_day_students books_per_student h_total h_first h_second h_third h_books_per_student
  sorry

end students_can_be_helped_on_fourth_day_l16_16889


namespace correct_operation_l16_16728

-- Defining the options as hypotheses
variable {a b : ℕ}

theorem correct_operation (hA : 4*a + 3*b ≠ 7*a*b)
    (hB : a^4 * a^3 = a^7)
    (hC : (3*a)^3 ≠ 9*a^3)
    (hD : a^6 / a^2 ≠ a^3) :
    a^4 * a^3 = a^7 := by
  sorry

end correct_operation_l16_16728


namespace polygon_sum_of_sides_l16_16128

-- Define the problem conditions and statement
theorem polygon_sum_of_sides :
  ∀ (A B C D E F : ℝ)
    (area_polygon : ℝ)
    (AB BC FA DE horizontal_distance_DF : ℝ),
    area_polygon = 75 →
    AB = 7 →
    BC = 10 →
    FA = 6 →
    DE = AB →
    horizontal_distance_DF = 8 →
    (DE + (2 * area_polygon - AB * BC) / (2 * horizontal_distance_DF) = 8.25) := 
by
  intro A B C D E F area_polygon AB BC FA DE horizontal_distance_DF
  intro h_area_polygon h_AB h_BC h_FA h_DE h_horizontal_distance_DF
  sorry

end polygon_sum_of_sides_l16_16128


namespace certain_event_l16_16868

-- Define the conditions for the problem
def EventA : Prop := ∃ (seat_number : ℕ), seat_number % 2 = 1
def EventB : Prop := ∃ (shooter_hits : Prop), shooter_hits
def EventC : Prop := ∃ (broadcast_news : Prop), broadcast_news
def EventD : Prop := 
  ∀ (red_ball_count white_ball_count : ℕ), (red_ball_count = 2) ∧ (white_ball_count = 1) → 
  ∀ (draw_count : ℕ), (draw_count = 2) → 
  (∃ (red_ball_drawn : Prop), red_ball_drawn)

-- Define the main statement to prove EventD is the certain event
theorem certain_event : EventA → EventB → EventC → EventD
:= 
sorry

end certain_event_l16_16868


namespace num_ways_to_choose_starters_l16_16146

theorem num_ways_to_choose_starters :
  let players := 16
  let triplets := 3
  let twins := 2
  let others := players - triplets
  let spots_after_triplets := 7 - triplets
  let scenario1_pairs_left := spots_after_triplets - twins
  let scenario2_spots := spots_after_triplets
  (Nat.choose (others - twins) scenario1_pairs_left + 
   Nat.choose (others - twins) scenario2_spots = 385) :=
by
  sorry

end num_ways_to_choose_starters_l16_16146


namespace simplify_and_evaluate_l16_16265

theorem simplify_and_evaluate : 
  (1 / (3 - 2) - 1 / (3 + 1)) / (3 / (3^2 - 1)) = 2 :=
by
  sorry

end simplify_and_evaluate_l16_16265


namespace point_on_inverse_proportion_l16_16947

theorem point_on_inverse_proportion (k : ℝ) (hk : k ≠ 0) :
  (2 * 3 = k) → (1 * 6 = k) :=
by
  intro h
  sorry

end point_on_inverse_proportion_l16_16947


namespace Tanya_efficiency_higher_l16_16255

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end Tanya_efficiency_higher_l16_16255


namespace intersection_complement_eq_l16_16508

-- Definitions of the sets M and N
def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Complements with respect to the reals
def complement_R (A : Set ℝ) : Set ℝ := {x | x ∉ A}

-- Target goal to prove
theorem intersection_complement_eq :
  M ∩ (complement_R N) = {1, 2} :=
by
  sorry

end intersection_complement_eq_l16_16508


namespace inverse_proportional_k_value_l16_16381

theorem inverse_proportional_k_value (k : ℝ) :
  (∃ x y : ℝ, y = k / x ∧ x = - (Real.sqrt 2) / 2 ∧ y = Real.sqrt 2) → 
  k = -1 :=
by
  sorry

end inverse_proportional_k_value_l16_16381


namespace volume_PQRS_is_48_39_cm3_l16_16227

noncomputable def area_of_triangle (a h : ℝ) : ℝ := 0.5 * a * h

noncomputable def volume_of_tetrahedron (base_area height : ℝ) : ℝ := (1/3) * base_area * height

noncomputable def height_from_area (area base : ℝ) : ℝ := (2 * area) / base

noncomputable def volume_of_tetrahedron_PQRS : ℝ :=
  let PQ := 5
  let area_PQR := 18
  let area_PQS := 16
  let angle_PQ := 45
  let h_PQR := height_from_area area_PQR PQ
  let h_PQS := height_from_area area_PQS PQ
  let h := h_PQS * (Real.sin (angle_PQ * Real.pi / 180))
  volume_of_tetrahedron area_PQR h

theorem volume_PQRS_is_48_39_cm3 : volume_of_tetrahedron_PQRS = 48.39 := by
  sorry

end volume_PQRS_is_48_39_cm3_l16_16227


namespace negation_proposition_l16_16084

theorem negation_proposition (p : Prop) : 
  (∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ ¬ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) :=
by
  sorry

end negation_proposition_l16_16084


namespace product_zero_when_b_is_3_l16_16910

theorem product_zero_when_b_is_3 (b : ℤ) (h : b = 3) :
  (b - 13) * (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) *
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by {
  sorry
}

end product_zero_when_b_is_3_l16_16910


namespace probability_at_least_one_8_l16_16159

theorem probability_at_least_one_8 (n : ℕ) (hn : n = 8) : 
  (1 - (7/8) * (7/8)) = 15 / 64 :=
by
  rw [← hn]
  sorry

end probability_at_least_one_8_l16_16159


namespace bead_necklaces_sold_l16_16197

def cost_per_necklace : ℕ := 7
def total_earnings : ℕ := 70
def gemstone_necklaces_sold : ℕ := 7

theorem bead_necklaces_sold (B : ℕ) 
  (h1 : total_earnings = cost_per_necklace * (B + gemstone_necklaces_sold))  :
  B = 3 :=
by {
  sorry
}

end bead_necklaces_sold_l16_16197


namespace total_students_standing_committee_ways_different_grade_pairs_ways_l16_16426

-- Given conditions
def freshmen : ℕ := 5
def sophomores : ℕ := 6
def juniors : ℕ := 4

-- Proofs (statements only, no proofs provided)
theorem total_students : freshmen + sophomores + juniors = 15 :=
by sorry

theorem standing_committee_ways : freshmen * sophomores * juniors = 120 :=
by sorry

theorem different_grade_pairs_ways :
  freshmen * sophomores + sophomores * juniors + juniors * freshmen = 74 :=
by sorry

end total_students_standing_committee_ways_different_grade_pairs_ways_l16_16426


namespace find_m_equals_powers_of_2_l16_16664

-- Define the function r_k(n) as the remainder of n divided by k
def r_k (n k : ℕ) : ℕ := n - k * (n / k)

-- Define the function r(n) as the sum of r_k(n) for k going from 1 to n
def r (n : ℕ) : ℕ := (Finset.range n).sum (λ k, r_k (n) (k + 1))

-- Main theorem to prove
theorem find_m_equals_powers_of_2 (m : ℕ) (h1 : 1 < m) (h2 : m ≤ 2014) :
  r m = r (m - 1) ↔ ∃ s : ℕ, m = 2^s ∧ 1 < 2^s ∧ 2^s ≤ 2014 :=
by
  sorry

end find_m_equals_powers_of_2_l16_16664


namespace salmon_at_rest_oxygen_units_l16_16124

noncomputable def salmonSwimSpeed (x : ℝ) : ℝ := (1/2) * Real.log (x / 100 * Real.pi) / Real.log 3

theorem salmon_at_rest_oxygen_units :
  ∃ x : ℝ, salmonSwimSpeed x = 0 ∧ x = 100 / Real.pi :=
by
  sorry

end salmon_at_rest_oxygen_units_l16_16124


namespace sum_of_x_and_y_l16_16542

theorem sum_of_x_and_y 
  (x y : ℝ)
  (h : ((x + 1) + (y-1)) / 2 = 10) : x + y = 20 :=
sorry

end sum_of_x_and_y_l16_16542


namespace find_g_1_l16_16239

noncomputable def g (x : ℝ) : ℝ := sorry -- express g(x) as a 4th degree polynomial with unknown coefficients

-- Conditions given in the problem
axiom cond1 : |g (-1)| = 15
axiom cond2 : |g (0)| = 15
axiom cond3 : |g (2)| = 15
axiom cond4 : |g (3)| = 15
axiom cond5 : |g (4)| = 15

-- The statement we need to prove
theorem find_g_1 : |g 1| = 11 :=
sorry

end find_g_1_l16_16239


namespace correct_figure_is_D_l16_16229

def option_A : Prop := sorry -- placeholder for option A as a diagram representation
def option_B : Prop := sorry -- placeholder for option B as a diagram representation
def option_C : Prop := sorry -- placeholder for option C as a diagram representation
def option_D : Prop := sorry -- placeholder for option D as a diagram representation
def equilateral_triangle (figure : Prop) : Prop := sorry -- placeholder for the condition representing an equilateral triangle in the oblique projection method

theorem correct_figure_is_D : equilateral_triangle option_D := 
sorry

end correct_figure_is_D_l16_16229


namespace min_value_func_l16_16279

noncomputable def func (x : ℝ) : ℝ := Real.sin x + Real.sqrt 3 * Real.cos x

theorem min_value_func : ∃ x : ℝ, func x = -2 :=
by
  existsi (Real.pi / 2 + Real.pi / 3)
  sorry

end min_value_func_l16_16279


namespace interest_rate_of_A_to_B_l16_16883

theorem interest_rate_of_A_to_B :
  ∀ (principal gain interest_B_to_C : ℝ), 
  principal = 3500 →
  gain = 525 →
  interest_B_to_C = 0.15 →
  (principal * interest_B_to_C * 3 - gain) = principal * (10 / 100) * 3 :=
by
  intros principal gain interest_B_to_C h_principal h_gain h_interest_B_to_C
  sorry

end interest_rate_of_A_to_B_l16_16883


namespace number_of_buses_in_month_l16_16756

-- Given conditions
def weekday_buses := 36
def saturday_buses := 24
def sunday_holiday_buses := 12
def num_weekdays := 18
def num_saturdays := 4
def num_sundays_holidays := 6

-- Statement to prove
theorem number_of_buses_in_month : 
  num_weekdays * weekday_buses + num_saturdays * saturday_buses + num_sundays_holidays * sunday_holiday_buses = 816 := 
by 
  sorry

end number_of_buses_in_month_l16_16756


namespace find_k_l16_16140

-- Given conditions and hypothesis stated
axiom quadratic_eq (x k : ℝ) : x^2 + 10 * x + k = 0

def roots_in_ratio_3_1 (α β : ℝ) : Prop :=
  α / β = 3

-- Statement of the theorem to be proved
theorem find_k {α β k : ℝ} (h1 : quadratic_eq α k) (h2 : quadratic_eq β k)
               (h3 : α ≠ 0) (h4 : β ≠ 0) (h5 : roots_in_ratio_3_1 α β) :
  k = 18.75 :=
by
  sorry

end find_k_l16_16140


namespace max_positive_integer_difference_l16_16167

theorem max_positive_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) : ∃ d : ℕ, d = 6 :=
by
  sorry

end max_positive_integer_difference_l16_16167


namespace negation_of_existential_l16_16548

theorem negation_of_existential :
  (∀ x : ℝ, x^2 + x - 1 ≤ 0) ↔ ¬ (∃ x : ℝ, x^2 + x - 1 > 0) :=
by sorry

end negation_of_existential_l16_16548


namespace arithmetic_mean_two_digit_multiples_of_8_l16_16856

theorem arithmetic_mean_two_digit_multiples_of_8 :
  let sequence : List ℕ := List.range' 16 97 (by sorry)
  let multiples_of_8 := filter (λ x, x % 8 = 0) sequence
  let num_terms := length multiples_of_8
  let sum := foldr (λ (x : ℕ) acc, x + acc) 0 multiples_of_8
  let arithmetic_mean := sum / num_terms

  arithmetic_mean = 56 := sorry

end arithmetic_mean_two_digit_multiples_of_8_l16_16856


namespace combined_weight_of_Meg_and_Chris_cats_l16_16424

-- Definitions based on the conditions
def ratio (M A C : ℕ) : Prop := 13 * A = 21 * M ∧ 13 * C = 28 * M 
def half_anne (M A : ℕ) : Prop := M = 20 + A / 2
def total_weight (M A C T : ℕ) : Prop := T = M + A + C

-- Theorem statement
theorem combined_weight_of_Meg_and_Chris_cats (M A C T : ℕ) 
  (h1 : ratio M A C) 
  (h2 : half_anne M A) 
  (h3 : total_weight M A C T) : 
  M + C = 328 := 
sorry

end combined_weight_of_Meg_and_Chris_cats_l16_16424


namespace area_PCD_eq_l16_16346

/-- Define the points P, D, and C as given in the conditions. -/
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 18⟩
def D : Point := ⟨3, 18⟩
def C (q : ℝ) : Point := ⟨0, q⟩

/-- Define the function to compute the area of triangle PCD given q. -/
noncomputable def area_triangle_PCD (q : ℝ) : ℝ :=
  1 / 2 * (D.x - P.x) * (P.y - q)

theorem area_PCD_eq (q : ℝ) : 
  area_triangle_PCD q = 27 - 3 / 2 * q := 
by 
  sorry

end area_PCD_eq_l16_16346


namespace solve_quadratic_l16_16712

theorem solve_quadratic (x : ℝ) : (x^2 + 2*x = 0) ↔ (x = 0 ∨ x = -2) :=
by
  sorry

end solve_quadratic_l16_16712


namespace solution_exists_l16_16301

theorem solution_exists :
  ∃ x : ℝ, x = 2 ∧ (-2 * x + 4 = 0) :=
sorry

end solution_exists_l16_16301


namespace average_weight_increase_l16_16715

noncomputable def average_increase (A : ℝ) : ℝ :=
  let initial_total := 10 * A
  let new_total := initial_total + 25
  let new_average := new_total / 10
  new_average - A

theorem average_weight_increase (A : ℝ) : average_increase A = 2.5 := by
  sorry

end average_weight_increase_l16_16715


namespace inequality_l16_16114

variable (a b c : ℝ)

noncomputable def condition (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 / 8

theorem inequality (h : condition a b c) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + a^2 * c^2 + b^2 * c^2 ≥ 15 / 16 :=
sorry

end inequality_l16_16114


namespace arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l16_16857

open Nat

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_8 (n : ℕ) : Prop := n % 8 = 0

theorem arithmetic_mean_of_two_digit_multiples_of_8_eq_56 :
  let multiples := filter (λ n, is_two_digit n ∧ is_multiple_of_8 n) (range 100)
  arithmetic_mean multiples = 56 :=
by
  -- Definitions for clarity:
  def arithmetic_mean (l : List ℕ) :=
    l.sum / l.length
  
  have multiples_eq : multiples = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96] := by
    -- veriifying multiples
    sorry
  -- Arithmetic mean calculation
  suffices arith_mean_calc : arithmetic_mean multiples = 56 by
    exact this
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_eq_56_l16_16857


namespace square_area_from_diagonal_l16_16276

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) :
  ∃ (A : ℝ), A = 72 :=
by
  sorry

end square_area_from_diagonal_l16_16276


namespace simplify_expression_l16_16535

-- Define the complex numbers and conditions
def ω : ℂ := (-1 + complex.I * real.sqrt 3) / 2
def ω_conj : ℂ := (-1 - complex.I * real.sqrt 3) / 2

-- Conditions
axiom ω_is_root_of_unity : ω^3 = 1
axiom ω_conj_is_root_of_unity : ω_conj^3 = 1

-- Theorem statement
theorem simplify_expression : ω^12 + ω_conj^12 = 2 := by
  sorry

end simplify_expression_l16_16535


namespace find_x_solution_l16_16374

theorem find_x_solution (x : ℝ) 
  (h : ∑' n:ℕ, ((-1)^(n+1)) * (2 * n + 1) * x^n = 16) : 
  x = -15/16 :=
sorry

end find_x_solution_l16_16374


namespace calculate_f_5_l16_16216

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

theorem calculate_f_5 : f 5 = 4485 := 
by {
  -- The proof of the theorem will go here, using the Horner's method as described.
  sorry
}

end calculate_f_5_l16_16216


namespace arithmetic_sequence_sum_l16_16929

variable (S : ℕ → ℝ)
variable (a_n : ℕ → ℝ)

theorem arithmetic_sequence_sum (h₁ : S 5 = 8) (h₂ : S 10 = 20) : S 15 = 36 := 
by
  sorry

end arithmetic_sequence_sum_l16_16929


namespace express_as_terminating_decimal_l16_16598

section terminating_decimal

theorem express_as_terminating_decimal
  (a b : ℚ)
  (h1 : a = 125)
  (h2 : b = 144)
  (h3 : b = 2^4 * 3^2): 
  a / b = 0.78125 := 
by 
  sorry

end terminating_decimal

end express_as_terminating_decimal_l16_16598


namespace quadratic_no_real_roots_l16_16960

theorem quadratic_no_real_roots (a b c d : ℝ)  :
  a^2 - 4 * b < 0 → c^2 - 4 * d < 0 → ( (a + c) / 2 )^2 - 4 * ( (b + d) / 2 ) < 0 :=
by
  sorry

end quadratic_no_real_roots_l16_16960


namespace marble_ratio_l16_16434

theorem marble_ratio (W L M : ℕ) (h1 : W = 16) (h2 : L = W + W / 4) (h3 : W + L + M = 60) :
  M / (W + L) = 2 / 3 := 
sorry

end marble_ratio_l16_16434


namespace less_money_than_Bob_l16_16945

noncomputable def Jennas_money (P: ℝ) : ℝ := 2 * P
noncomputable def Phils_money (B: ℝ) : ℝ := B / 3
noncomputable def Bobs_money : ℝ := 60
noncomputable def Johns_money (P: ℝ) : ℝ := P + 0.35 * P
noncomputable def average (x y: ℝ) : ℝ := (x + y) / 2

theorem less_money_than_Bob :
  ∀ (P Q J B : ℝ),
    P = Phils_money B →
    J = Jennas_money P →
    Q = Johns_money P →
    B = Bobs_money →
    average J Q = B - 0.25 * B →
    B - J = 20
  :=
by
  intros P Q J B hP hJ hQ hB h_avg
  -- Proof goes here
  sorry

end less_money_than_Bob_l16_16945


namespace point_D_is_on_y_axis_l16_16444

def is_on_y_axis (p : ℝ × ℝ) : Prop := p.fst = 0

def point_A : ℝ × ℝ := (3, 0)
def point_B : ℝ × ℝ := (1, 2)
def point_C : ℝ × ℝ := (2, 1)
def point_D : ℝ × ℝ := (0, -3)

theorem point_D_is_on_y_axis : is_on_y_axis point_D :=
by
  sorry

end point_D_is_on_y_axis_l16_16444


namespace number_of_circles_l16_16831

theorem number_of_circles (side : ℝ) (enclosed_area : ℝ) (num_circles : ℕ) (radius : ℝ) :
  side = 14 ∧ enclosed_area = 42.06195997410015 ∧ 2 * radius = side ∧ π * radius^2 = 49 * π → num_circles = 4 :=
by
  intros
  sorry

end number_of_circles_l16_16831


namespace find_m_such_that_no_linear_term_in_expansion_l16_16089

theorem find_m_such_that_no_linear_term_in_expansion :
  ∃ m : ℝ, ∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9 * x^2 - 8 * m ∧ ((8 + m) = 0) :=
by
  sorry

end find_m_such_that_no_linear_term_in_expansion_l16_16089


namespace quadratic_inequality_solution_l16_16075

theorem quadratic_inequality_solution:
  ∃ P q : ℝ,
  (1 / P < 0) ∧
  (-P * q = 6) ∧
  (P^2 = 8) ∧
  (P = -2 * Real.sqrt 2) ∧
  (q = 3 / 2 * Real.sqrt 2) :=
by
  sorry

end quadratic_inequality_solution_l16_16075


namespace dolphins_score_l16_16808

theorem dolphins_score (S D : ℕ) (h1 : S + D = 48) (h2 : S = D + 20) : D = 14 := by
    sorry

end dolphins_score_l16_16808


namespace train_stop_time_per_hour_l16_16597

theorem train_stop_time_per_hour
    (speed_excl_stoppages : ℕ)
    (speed_incl_stoppages : ℕ)
    (h1 : speed_excl_stoppages = 48)
    (h2 : speed_incl_stoppages = 36) :
    ∃ (t : ℕ), t = 15 :=
by
  sorry

end train_stop_time_per_hour_l16_16597


namespace hyperbola_foci_l16_16351

/-- Define a hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop := 4 * y^2 - 25 * x^2 = 100

/-- Definition of the foci of the hyperbola -/
def foci_coords (c : ℝ) : Prop := c = Real.sqrt 29

/-- Proof that the foci of the hyperbola 4y^2 - 25x^2 = 100 are (0, -sqrt(29)) and (0, sqrt(29)) -/
theorem hyperbola_foci (x y : ℝ) (c : ℝ) (hx : hyperbola_eq x y) (hc : foci_coords c) :
  (x = 0 ∧ (y = -c ∨ y = c)) :=
sorry

end hyperbola_foci_l16_16351


namespace train_speed_is_42_point_3_km_per_h_l16_16749

-- Definitions for the conditions.
def train_length : ℝ := 150
def bridge_length : ℝ := 320
def crossing_time : ℝ := 40
def meter_per_sec_to_km_per_hour : ℝ := 3.6
def total_distance : ℝ := train_length + bridge_length

-- The theorem we want to prove
theorem train_speed_is_42_point_3_km_per_h : 
    (total_distance / crossing_time) * meter_per_sec_to_km_per_hour = 42.3 :=
by 
    -- Proof omitted
    sorry

end train_speed_is_42_point_3_km_per_h_l16_16749


namespace more_sparrows_than_pigeons_l16_16475

-- Defining initial conditions
def initial_sparrows := 3
def initial_starlings := 5
def initial_pigeons := 2
def additional_sparrows := 4
def additional_starlings := 2
def additional_pigeons := 3

-- Final counts after additional birds join
def final_sparrows := initial_sparrows + additional_sparrows
def final_pigeons := initial_pigeons + additional_pigeons

-- The statement to be proved
theorem more_sparrows_than_pigeons:
  final_sparrows - final_pigeons = 2 :=
by
  -- proof skipped
  sorry

end more_sparrows_than_pigeons_l16_16475


namespace final_score_is_correct_l16_16652

-- Definitions based on given conditions
def speechContentScore : ℕ := 90
def speechDeliveryScore : ℕ := 85
def weightContent : ℕ := 6
def weightDelivery : ℕ := 4

-- The final score calculation theorem
theorem final_score_is_correct : 
  (speechContentScore * weightContent + speechDeliveryScore * weightDelivery) / (weightContent + weightDelivery) = 88 :=
  by
    sorry

end final_score_is_correct_l16_16652


namespace fifth_inequality_l16_16971

theorem fifth_inequality (h1: 1 / Real.sqrt 2 < 1)
                         (h2: 1 / Real.sqrt 2 + 1 / Real.sqrt 6 < Real.sqrt 2)
                         (h3: 1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 < Real.sqrt 3) :
                         1 / Real.sqrt 2 + 1 / Real.sqrt 6 + 1 / Real.sqrt 12 + 1 / Real.sqrt 20 + 1 / Real.sqrt 30 < Real.sqrt 5 := 
sorry

end fifth_inequality_l16_16971


namespace age_of_15th_student_l16_16698

theorem age_of_15th_student 
  (total_age_15_students : ℕ)
  (total_age_3_students : ℕ)
  (total_age_11_students : ℕ)
  (h1 : total_age_15_students = 225)
  (h2 : total_age_3_students = 42)
  (h3 : total_age_11_students = 176) :
  total_age_15_students - (total_age_3_students + total_age_11_students) = 7 :=
by
  sorry

end age_of_15th_student_l16_16698


namespace dot_product_is_2_l16_16611

variable (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_is_2 (ha : a = (1, 0)) (hb : b = (2, 1)) :
  dot_product a b = 2 := by
  sorry

end dot_product_is_2_l16_16611


namespace area_of_cos_integral_l16_16697

theorem area_of_cos_integral : 
  (∫ x in (0:ℝ)..(3 * Real.pi / 2), |Real.cos x|) = 3 :=
by
  sorry

end area_of_cos_integral_l16_16697


namespace correct_sum_is_826_l16_16306

theorem correct_sum_is_826 (ABC : ℕ)
  (h1 : 100 ≤ ABC ∧ ABC < 1000)  -- Ensuring ABC is a three-digit number
  (h2 : ∃ A B C : ℕ, ABC = 100 * A + 10 * B + C ∧ C = 6) -- Misread ones digit is 6
  (incorrect_sum : ℕ)
  (h3 : incorrect_sum = ABC + 57)  -- Sum obtained by Yoongi was 823
  (h4 : incorrect_sum = 823) : ABC + 57 + 3 = 826 :=  -- Correcting the sum considering the 6 to 9 error
by
  sorry

end correct_sum_is_826_l16_16306


namespace y_coordinate_of_second_point_l16_16389

variable {m n k : ℝ}

theorem y_coordinate_of_second_point (h1 : m = 2 * n + 5) (h2 : k = 0.5) : (n + k) = n + 0.5 := 
by
  sorry

end y_coordinate_of_second_point_l16_16389


namespace number_of_terms_in_arithmetic_sequence_l16_16492

/-- Define the conditions. -/
def a : ℕ := 2
def d : ℕ := 5
def a_n : ℕ := 57

/-- Define the proof problem. -/
theorem number_of_terms_in_arithmetic_sequence :
  ∃ n : ℕ, a_n = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l16_16492


namespace ratio_of_side_lengths_of_frustum_l16_16610

theorem ratio_of_side_lengths_of_frustum (L1 L2 H : ℚ) (V_prism V_frustum : ℚ)
  (h1 : V_prism = L1^2 * H)
  (h2 : V_frustum = (1/3) * (L1^2 * (H * (L1 / (L1 - L2))) - L2^2 * (H * (L2 / (L1 - L2)))))
  (h3 : V_frustum = (2/3) * V_prism) :
  L1 / L2 = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end ratio_of_side_lengths_of_frustum_l16_16610


namespace no_solution_exists_l16_16054

theorem no_solution_exists : ¬ ∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 2 * m^2 := by sorry

end no_solution_exists_l16_16054


namespace collinear_D_G_X_l16_16821

open EuclideanGeometry Triangle Circle

/- Define the problem setup -/

variables {A B C : Point}
variables (ABC : Triangle A B C)
variables (Ω : Circle A B C)  -- circumcircle of triangle ABC
variables (B0 : Point) (C0 : Point)  -- midpoints
variables (D : Point)  -- foot of the altitude from A
variables (G : Point)  -- centroid of triangle ABC
variables (ω : Circle)  -- circle through B0 and C0 tangent to Ω
variables (X : Point)  -- point of tangency

/- Define the conditions explicitly -/

-- B0 is the midpoint of AC
axiom B0_midpoint : midpoint B0 A C

-- C0 is the midpoint of AB
axiom C0_midpoint : midpoint C0 A B

-- D is the foot of the altitude from A
axiom D_foot : altitude_foot D A B C

-- G is the centroid of triangle ABC
axiom G_centroid : centroid G A B C

-- ω is a circle through B0 and C0 that is tangent to Ω at a point X ≠ A
axiom ω_tangent_Ω : tangent_at ω Ω X ∧ X ≠ A ∧ on_circle B0 ω ∧ on_circle C0 ω

/- The theorem to prove collinearity of D, G, and X -/

theorem collinear_D_G_X : collinear D G X :=
by
sor_approval_steps.re_states
_end_approval_stepsora

end collinear_D_G_X_l16_16821


namespace f_2_value_l16_16498

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^3 + b * x - 4

theorem f_2_value :
  (f a b (-2)) = 2 → (f a b 2) = -10 :=
by
  intro h
  -- Provide the solution steps here, starting with simplifying the equation. Sorry for now
  sorry

end f_2_value_l16_16498


namespace election_winner_votes_difference_l16_16721

theorem election_winner_votes_difference (V : ℝ) (h1 : 0.62 * V = 1054) : 0.24 * V = 408 :=
by
  sorry

end election_winner_votes_difference_l16_16721


namespace total_fast_food_order_cost_l16_16817

def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def smoothies_quantity : ℕ := 2

theorem total_fast_food_order_cost : burger_cost + sandwich_cost + smoothies_quantity * smoothie_cost = 17 := 
by
  sorry

end total_fast_food_order_cost_l16_16817


namespace integral_eq_log_div_l16_16343

open Real

noncomputable def integral_result (m n : ℝ) (hm : 0 < m) (hn : 0 < n) : ℝ :=
  ∫ x in 0..1, (x ^ m - x ^ n) / log x

theorem integral_eq_log_div (m n : ℝ) (hm : 0 < m) (hn : 0 < n) :
  integral_result m n hm hn = log (abs ((m + 1) / (n + 1))) :=
sorry

end integral_eq_log_div_l16_16343


namespace positive_difference_abs_eq_15_l16_16005

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l16_16005


namespace find_percentage_of_alcohol_in_second_solution_l16_16454

def alcohol_content_second_solution (V2: ℕ) (p1 p2 p_final: ℕ) (V1 V_final: ℕ) : ℕ :=
  ((V_final * p_final) - (V1 * p1)) * 100 / V2

def percentage_correct : Prop :=
  alcohol_content_second_solution 125 20 12 15 75 200 = 12

theorem find_percentage_of_alcohol_in_second_solution : percentage_correct :=
by
  sorry

end find_percentage_of_alcohol_in_second_solution_l16_16454


namespace contestant_final_score_l16_16654

theorem contestant_final_score 
    (content_score : ℕ)
    (delivery_score : ℕ)
    (weight_content : ℕ)
    (weight_delivery : ℕ)
    (h1 : content_score = 90)
    (h2 : delivery_score = 85)
    (h3 : weight_content = 6)
    (h4 : weight_delivery = 4) : 
    (content_score * weight_content + delivery_score * weight_delivery) / (weight_content + weight_delivery) = 88 := 
sorry

end contestant_final_score_l16_16654


namespace paper_stars_per_bottle_l16_16174

theorem paper_stars_per_bottle (a b total_bottles : ℕ) (h1 : a = 33) (h2 : b = 307) (h3 : total_bottles = 4) :
  (a + b) / total_bottles = 85 :=
by
  sorry

end paper_stars_per_bottle_l16_16174


namespace vector_operation_result_l16_16145

variables {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C O E : V)

theorem vector_operation_result :
  (A - B) - (C - B) + (O - E) - (O - C) = (A - E) :=
by
  sorry

end vector_operation_result_l16_16145


namespace arithmetic_sequence_m_value_l16_16361

variable {a : ℕ → ℝ} {S : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

noncomputable def find_m (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) : Prop :=
  (a (m + 1) + a (m - 1) - a m ^ 2 = 0) → (S (2 * m - 1) = 38) → m = 10

-- Problem Statement
theorem arithmetic_sequence_m_value :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ),
    arithmetic_sequence a → 
    sum_of_first_n_terms S a → 
    find_m a S m :=
by
  intros a S m ha hs h₁ h₂
  sorry

end arithmetic_sequence_m_value_l16_16361


namespace find_arrays_l16_16775

theorem find_arrays (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a ∣ b * c * d - 1 ∧ b ∣ a * c * d - 1 ∧ c ∣ a * b * d - 1 ∧ d ∣ a * b * c - 1 →
  (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨
  (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) := by
  sorry

end find_arrays_l16_16775


namespace correct_option_l16_16302

theorem correct_option : (∃ x, x = -3 ∧ x^3 = -27) :=
by {
  -- Given conditions
  let x := -3
  use x
  constructor
  . rfl
  . norm_num
}

end correct_option_l16_16302


namespace compute_expression_l16_16113

variable {R : Type*} [LinearOrderedField R]

theorem compute_expression (r s t : R)
  (h_eq_root: ∀ x, x^3 - 4 * x^2 + 4 * x - 6 = 0)
  (h1: r + s + t = 4)
  (h2: r * s + r * t + s * t = 4)
  (h3: r * s * t = 6) :
  r * s / t + s * t / r + t * r / s = -16 / 3 :=
sorry

end compute_expression_l16_16113


namespace pool_depth_l16_16064

theorem pool_depth 
  (length width : ℝ) 
  (chlorine_per_120_cubic_feet chlorine_cost : ℝ) 
  (total_spent volume_per_quart_of_chlorine : ℝ) 
  (H_length : length = 10) 
  (H_width : width = 8)
  (H_chlorine_per_120_cubic_feet : chlorine_per_120_cubic_feet = 1 / 120)
  (H_chlorine_cost : chlorine_cost = 3)
  (H_total_spent : total_spent = 12)
  (H_volume_per_quart_of_chlorine : volume_per_quart_of_chlorine = 120) :
  ∃ depth : ℝ, total_spent / chlorine_cost * volume_per_quart_of_chlorine = length * width * depth ∧ depth = 6 :=
by 
  sorry

end pool_depth_l16_16064


namespace triangle_similar_l16_16978

variables {a b c m_a m_b m_c t : ℝ}

-- Define the triangle ABC and its properties
def triangle_ABC (a b c m_a m_b m_c t : ℝ) : Prop :=
  t = (1 / 2) * a * m_a ∧
  t = (1 / 2) * b * m_b ∧
  t = (1 / 2) * c * m_c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧
  t > 0

-- Define the similarity condition for the triangles
def similitude_from_reciprocals (a b c m_a m_b m_c t : ℝ) : Prop :=
  (1 / m_a) / (1 / m_b) = a / b ∧
  (1 / m_b) / (1 / m_c) = b / c ∧
  (1 / m_a) / (1 / m_c) = a / c

theorem triangle_similar (a b c m_a m_b m_c t : ℝ) :
  triangle_ABC a b c m_a m_b m_c t →
  similitude_from_reciprocals a b c m_a m_b m_c t :=
by
  intro h
  sorry

end triangle_similar_l16_16978


namespace quadratic_sum_solutions_l16_16590

theorem quadratic_sum_solutions {a b : ℝ} (h : a ≥ b) (h1: a = 1 + Real.sqrt 17) (h2: b = 1 - Real.sqrt 17) :
  3 * a + 2 * b = 5 + Real.sqrt 17 := by
  sorry

end quadratic_sum_solutions_l16_16590


namespace number_of_female_students_l16_16130

theorem number_of_female_students
  (F : ℕ) -- number of female students
  (T : ℕ) -- total number of students
  (h1 : T = F + 8) -- total students = female students + 8 male students
  (h2 : 90 * T = 85 * 8 + 92 * F) -- equation from the sum of scores
  : F = 20 :=
sorry

end number_of_female_students_l16_16130


namespace vertical_asymptote_singleton_l16_16355

theorem vertical_asymptote_singleton (c : ℝ) :
  (∃ x, (x^2 - 2 * x + c) = 0 ∧ ((x - 1) * (x + 3) = 0) ∧ (x ≠ 1 ∨ x ≠ -3)) 
  ↔ (c = 1 ∨ c = -15) :=
by
  sorry

end vertical_asymptote_singleton_l16_16355


namespace calculate_angle_l16_16586

def degrees_to_seconds (d m s : ℕ) : ℕ :=
  d * 3600 + m * 60 + s

def seconds_to_degrees (s : ℕ) : (ℕ × ℕ × ℕ) :=
  (s / 3600, (s % 3600) / 60, s % 60)

theorem calculate_angle : 
  (let d1 := 50
   let m1 := 24
   let angle1_sec := degrees_to_seconds d1 m1 0
   let angle1_sec_tripled := 3 * angle1_sec
   let (d1', m1', s1') := seconds_to_degrees angle1_sec_tripled

   let d2 := 98
   let m2 := 12
   let s2 := 25
   let angle2_sec := degrees_to_seconds d2 m2 s2
   let angle2_sec_divided := angle2_sec / 5
   let (d2', m2', s2') := seconds_to_degrees angle2_sec_divided

   let total_sec := degrees_to_seconds d1' m1' s1' + degrees_to_seconds d2' m2' s2'
   let (final_d, final_m, final_s) := seconds_to_degrees total_sec
   (final_d, final_m, final_s)) = (170, 50, 29) := by sorry

end calculate_angle_l16_16586


namespace max_odd_integers_l16_16581

theorem max_odd_integers (a b c d e f : ℕ) 
  (hprod : a * b * c * d * e * f % 2 = 0) 
  (hpos_a : 0 < a) (hpos_b : 0 < b) 
  (hpos_c : 0 < c) (hpos_d : 0 < d) 
  (hpos_e : 0 < e) (hpos_f : 0 < f) : 
  ∃ x : ℕ, x ≤ 5 ∧ x = 5 :=
by sorry

end max_odd_integers_l16_16581


namespace Liu_Wei_parts_per_day_l16_16330

theorem Liu_Wei_parts_per_day :
  ∀ (total_parts days_needed parts_per_day_worked initial_days days_remaining : ℕ), 
  total_parts = 190 →
  parts_per_day_worked = 15 →
  initial_days = 2 →
  days_needed = 10 →
  days_remaining = days_needed - initial_days →
  (total_parts - (initial_days * parts_per_day_worked)) / days_remaining = 20 :=
by
  intros total_parts days_needed parts_per_day_worked initial_days days_remaining h1 h2 h3 h4 h5
  sorry

end Liu_Wei_parts_per_day_l16_16330


namespace tanya_efficiency_greater_sakshi_l16_16262

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end tanya_efficiency_greater_sakshi_l16_16262


namespace larry_channels_l16_16962

theorem larry_channels : (initial_channels : ℕ) 
                         (channels_taken : ℕ) 
                         (channels_replaced : ℕ) 
                         (reduce_channels : ℕ) 
                         (sports_package : ℕ) 
                         (supreme_sports_package : ℕ) 
                         (final_channels : ℕ)
                         (h1 : initial_channels = 150)
                         (h2 : channels_taken = 20)
                         (h3 : channels_replaced = 12)
                         (h4 : reduce_channels = 10)
                         (h5 : sports_package = 8)
                         (h6 : supreme_sports_package = 7)
                         (h7 : final_channels = initial_channels - channels_taken + channels_replaced - reduce_channels + sports_package + supreme_sports_package)
                         : final_channels = 147 :=
by sorry

end larry_channels_l16_16962


namespace circle_center_l16_16493

theorem circle_center (x y : ℝ) (h : x^2 + 8*x + y^2 - 4*y = 16) : (x, y) = (-4, 2) :=
by 
  sorry

end circle_center_l16_16493


namespace geometric_progression_l16_16427

theorem geometric_progression (b q : ℝ) :
  (b + b*q + b*q^2 + b*q^3 = -40) ∧ 
  (b^2 + (b*q)^2 + (b*q^2)^2 + (b*q^3)^2 = 3280) →
  (b = 2 ∧ q = -3) ∨ (b = -54 ∧ q = -1/3) :=
by sorry

end geometric_progression_l16_16427


namespace real_solutions_l16_16777

open Real

theorem real_solutions (x : ℝ) : (x - 2) ^ 4 + (2 - x) ^ 4 = 50 ↔ 
  x = 2 + sqrt (-12 + 3 * sqrt 17) ∨ x = 2 - sqrt (-12 + 3 * sqrt 17) :=
by
  sorry

end real_solutions_l16_16777


namespace Hamilton_marching_band_members_l16_16151

theorem Hamilton_marching_band_members (m : ℤ) (k : ℤ) :
  30 * m ≡ 5 [ZMOD 31] ∧ m = 26 + 31 * k ∧ 30 * m < 1500 → 30 * m = 780 :=
by
  intro h
  have hmod : 30 * m ≡ 5 [ZMOD 31] := h.left
  have m_eq : m = 26 + 31 * k := h.right.left
  have hlt : 30 * m < 1500 := h.right.right
  sorry

end Hamilton_marching_band_members_l16_16151


namespace matches_needed_eq_l16_16888

def count_matches (n : ℕ) : ℕ :=
  let total_triangles := n * n
  let internal_matches := 3 * total_triangles
  let external_matches := 4 * n
  internal_matches - external_matches + external_matches

theorem matches_needed_eq (n : ℕ) : count_matches 10 = 320 :=
by
  sorry

end matches_needed_eq_l16_16888


namespace cars_in_garage_l16_16990

theorem cars_in_garage (c : ℕ) 
  (bicycles : ℕ := 20) 
  (motorcycles : ℕ := 5) 
  (total_wheels : ℕ := 90) 
  (bicycle_wheels : ℕ := 2 * bicycles)
  (motorcycle_wheels : ℕ := 2 * motorcycles)
  (car_wheels : ℕ := 4 * c) 
  (eq : bicycle_wheels + car_wheels + motorcycle_wheels = total_wheels) : 
  c = 10 := 
by 
  sorry

end cars_in_garage_l16_16990


namespace find_x_for_prime_power_l16_16350

theorem find_x_for_prime_power (x : ℤ) :
  (∃ p k : ℕ, Nat.Prime p ∧ k > 0 ∧ (2 * x * x + x - 6 = p ^ k)) → (x = -3 ∨ x = 2 ∨ x = 5) := by
  sorry

end find_x_for_prime_power_l16_16350


namespace solve_dividend_and_divisor_l16_16253

-- Definitions for base, digits, and mathematical relationships
def base := 5
def P := 1
def Q := 2
def R := 3
def S := 4
def T := 0
def Dividend := 1 * base^6 + 2 * base^5 + 3 * base^4 + 4 * base^3 + 3 * base^2 + 2 * base^1 + 1 * base^0
def Divisor := 2 * base^2 + 3 * base^1 + 2 * base^0

-- The conditions given in the math problem
axiom condition_1 : Q + R = base
axiom condition_2 : P + 1 = Q
axiom condition_3 : Q + P = R
axiom condition_4 : S = 2 * Q
axiom condition_5 : Q^2 = S
axiom condition_6 : Dividend = 24336
axiom condition_7 : Divisor = 67

-- The goal
theorem solve_dividend_and_divisor : Dividend = 24336 ∧ Divisor = 67 :=
by {
  sorry
}

end solve_dividend_and_divisor_l16_16253


namespace even_subset_capacity_sum_112_l16_16236

open Finset

-- Define S_n
def S (n : ℕ) : Finset ℕ := range (n + 1)

-- Define capacity of a subset
def capacity (X : Finset ℕ) : ℕ := if X = ∅ then 0 else X.prod id

-- Define a predicate for an even subset
def is_even_subset (X : Finset ℕ) : Prop := capacity X % 2 = 0

-- Given n = 4
def S_4 : Finset ℕ := S 4

-- Function to calculate sum of capacities of even subsets of S_4
def sum_even_capacities (S : Finset ℕ) : ℕ :=
  univ.filter (λ X : Finset ℕ, is_even_subset X).sum capacity

theorem even_subset_capacity_sum_112 : sum_even_capacities S_4 = 112 := 
by sorry

end even_subset_capacity_sum_112_l16_16236


namespace percentage_increase_decrease_l16_16797

theorem percentage_increase_decrease (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq100 : q < 100) :
  (M * (1 + p / 100) * (1 - q / 100) = 1.1 * M) ↔ (p = (10 + 100 * q) / (100 - q)) :=
by 
  sorry

end percentage_increase_decrease_l16_16797


namespace a_is_perfect_square_l16_16685

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l16_16685


namespace evaluate_expression_l16_16594

theorem evaluate_expression (a : ℝ) (h : a = 4 / 3) : 
  (4 * a^2 - 12 * a + 9) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end evaluate_expression_l16_16594


namespace total_weight_of_new_people_l16_16982

theorem total_weight_of_new_people (W W_new : ℝ) :
  (∀ (old_weights : List ℝ), old_weights.length = 25 →
    ((old_weights.sum - (65 + 70 + 75)) + W_new = old_weights.sum + (4 * 25)) →
    W_new = 310) := by
  intros old_weights old_weights_length increase_condition
  -- Proof will be here
  sorry

end total_weight_of_new_people_l16_16982


namespace trains_crossing_time_l16_16565

noncomputable def length_first_train : ℝ := 120
noncomputable def length_second_train : ℝ := 160
noncomputable def speed_first_train_kmph : ℝ := 60
noncomputable def speed_second_train_kmph : ℝ := 40
noncomputable def kmph_to_mps (v : ℝ) : ℝ := v * 1000 / 3600

noncomputable def speed_first_train : ℝ := kmph_to_mps speed_first_train_kmph
noncomputable def speed_second_train : ℝ := kmph_to_mps speed_second_train_kmph
noncomputable def relative_speed : ℝ := speed_first_train + speed_second_train
noncomputable def total_distance : ℝ := length_first_train + length_second_train
noncomputable def crossing_time : ℝ := total_distance / relative_speed

theorem trains_crossing_time :
  crossing_time = 10.08 := by
  sorry

end trains_crossing_time_l16_16565


namespace backyard_area_l16_16518

-- Definitions from conditions
def length : ℕ := 1000 / 25
def perimeter : ℕ := 1000 / 10
def width : ℕ := (perimeter - 2 * length) / 2

-- Theorem statement: Given the conditions, the area of the backyard is 400 square meters
theorem backyard_area : length * width = 400 :=
by 
  -- Sorry to skip the proof as instructed
  sorry

end backyard_area_l16_16518


namespace max_buses_in_city_l16_16647

/--
In a city with 9 bus stops where each bus serves exactly 3 stops and any two buses share at most one stop, 
prove that the maximum number of buses that can be in the city is 12.
-/
theorem max_buses_in_city :
  ∃ (B : Set (Finset ℕ)), 
    (∀ b ∈ B, b.card = 3) ∧ 
    (∀ b1 b2 ∈ B, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1) ∧ 
    (∀ b ∈ B, ∀ s ∈ b, s < 9) ∧ 
    B.card = 12 := sorry

end max_buses_in_city_l16_16647


namespace remainder_when_nm_div_61_l16_16382

theorem remainder_when_nm_div_61 (n m : ℕ) (k j : ℤ):
  n = 157 * k + 53 → m = 193 * j + 76 → (n + m) % 61 = 7 := by
  intros h1 h2
  sorry

end remainder_when_nm_div_61_l16_16382


namespace ratio_of_price_l16_16557

-- Definitions from conditions
def original_price : ℝ := 3.00
def tom_pay_price : ℝ := 9.00

-- Theorem stating the ratio
theorem ratio_of_price : tom_pay_price / original_price = 3 := by
  sorry

end ratio_of_price_l16_16557


namespace number_of_pairs_l16_16939

theorem number_of_pairs :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 77}.to_finset.card = 2 :=
sorry

end number_of_pairs_l16_16939


namespace option_C_is_quadratic_l16_16443

theorem option_C_is_quadratic : ∀ (x : ℝ), (x = x^2) ↔ (∃ (a b c : ℝ), a ≠ 0 ∧ a*x^2 + b*x + c = 0) := 
by
  sorry

end option_C_is_quadratic_l16_16443


namespace greatest_of_5_consecutive_integers_l16_16439

theorem greatest_of_5_consecutive_integers (m n : ℤ) (h : 5 * n + 10 = m^3) : (n + 4) = 202 := by
sorry

end greatest_of_5_consecutive_integers_l16_16439


namespace math_problem_l16_16081

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c + 2 * (a + b + c) = 672 :=
by
  sorry

end math_problem_l16_16081


namespace permutations_divisibility_l16_16827

theorem permutations_divisibility (n : ℕ) (a b : Fin n → ℕ) 
  (h_n : 2 < n)
  (h_a_perm : ∀ i, ∃ j, a j = i)
  (h_b_perm : ∀ i, ∃ j, b j = i) :
  ∃ (i j : Fin n), i ≠ j ∧ n ∣ (a i * b i - a j * b j) :=
by sorry

end permutations_divisibility_l16_16827


namespace scientific_notation_correct_l16_16707

/-- Given the weight of the "人" shaped gate of the Three Gorges ship lock -/
def weight_kg : ℝ := 867000

/-- The scientific notation representation of the given weight -/
def scientific_notation_weight_kg : ℝ := 8.67 * 10^5

theorem scientific_notation_correct :
  weight_kg = scientific_notation_weight_kg :=
sorry

end scientific_notation_correct_l16_16707


namespace heartbeats_during_race_l16_16033

-- Define the conditions as constants
def heart_rate := 150 -- beats per minute
def race_distance := 26 -- miles
def pace := 5 -- minutes per mile

-- Formulate the statement
theorem heartbeats_during_race :
  heart_rate * (race_distance * pace) = 19500 :=
by
  sorry

end heartbeats_during_race_l16_16033


namespace probability_even_gx_l16_16948

noncomputable def f (a b x : ℝ) : ℝ := log (x^2 + a * x + 2 * b)
noncomputable def g (a b x : ℝ) : ℝ := (a^x - b^(-x)) / ((a + b) * x)

-- Mathematical conditions and statements
theorem probability_even_gx (a b : ℝ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) :
  (∀ x : ℝ, 0 < x^2 + a * x + 2 * b) →
  (∃ p : ℝ, p = 6 / 26 ∧ p = 3 / 13) :=
by
  sorry

end probability_even_gx_l16_16948


namespace problem_l16_16663

def g (x : ℤ) : ℤ := 3 * x^2 - x + 4

theorem problem : g (g 3) = 2328 := by
  sorry

end problem_l16_16663


namespace tom_apple_fraction_l16_16435

theorem tom_apple_fraction (initial_oranges initial_apples oranges_sold_fraction oranges_remaining total_fruits_remaining apples_initial apples_sold_fraction : ℕ→ℚ) :
  initial_oranges = 40 →
  initial_apples = 70 →
  oranges_sold_fraction = 1 / 4 →
  oranges_remaining = initial_oranges - initial_oranges * oranges_sold_fraction →
  total_fruits_remaining = 65 →
  total_fruits_remaining = oranges_remaining + (initial_apples - initial_apples * apples_sold_fraction) →
  apples_sold_fraction = 1 / 2 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tom_apple_fraction_l16_16435


namespace arithmetic_mean_of_twodigit_multiples_of_8_l16_16853

theorem arithmetic_mean_of_twodigit_multiples_of_8 :
  let sequence := list.range' 16 81 8 in
  let sum := list.sum sequence in
  let n := sequence.length in
  (sum / n) = 56 := 
by 
  sorry

end arithmetic_mean_of_twodigit_multiples_of_8_l16_16853


namespace glue_needed_l16_16234

-- Definitions based on conditions
def num_friends : ℕ := 7
def clippings_per_friend : ℕ := 3
def drops_per_clipping : ℕ := 6

-- Calculation
def total_clippings : ℕ := num_friends * clippings_per_friend
def total_drops_of_glue : ℕ := drops_per_clipping * total_clippings

-- Theorem statement
theorem glue_needed : total_drops_of_glue = 126 := by
  sorry

end glue_needed_l16_16234


namespace percentage_increase_in_efficiency_l16_16259

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end percentage_increase_in_efficiency_l16_16259


namespace min_fence_posts_l16_16886

theorem min_fence_posts (length width wall_length interval : ℕ) (h_dim : length = 80) (w_dim : width = 50) (h_wall : wall_length = 150) (h_interval : interval = 10) : 
  length/interval + 1 + 2 * (width/interval - 1) = 17 :=
by
  sorry

end min_fence_posts_l16_16886


namespace distribute_neg3_l16_16012

theorem distribute_neg3 (x y : ℝ) : -3 * (x - x * y) = -3 * x + 3 * x * y :=
by sorry

end distribute_neg3_l16_16012


namespace acid_solution_replaced_l16_16695

theorem acid_solution_replaced (P : ℝ) :
  (0.5 * 0.50 + 0.5 * P = 0.35) → P = 0.20 :=
by
  intro h
  sorry

end acid_solution_replaced_l16_16695


namespace two_colonies_reach_limit_same_time_l16_16015

theorem two_colonies_reach_limit_same_time
  (doubles_in_size : ∀ (n : ℕ), n = n * 2)
  (reaches_limit_in_25_days : ∃ N : ℕ, ∀ t : ℕ, t = 25 → N = N * 2^t) :
  ∀ t : ℕ, t = 25 := sorry

end two_colonies_reach_limit_same_time_l16_16015


namespace subtraction_example_l16_16037

theorem subtraction_example : 3.57 - 1.45 = 2.12 :=
by 
  sorry

end subtraction_example_l16_16037


namespace prime_remainder_l16_16058

theorem prime_remainder (p : ℕ) (k : ℕ) (h1 : Prime p) (h2 : p > 3) :
  (∃ k, p = 6 * k + 1 ∧ (p^3 + 17) % 24 = 18) ∨
  (∃ k, p = 6 * k - 1 ∧ (p^3 + 17) % 24 = 16) :=
by
  sorry

end prime_remainder_l16_16058


namespace car_travel_distance_l16_16315

noncomputable def distance_in_miles (b t : ℝ) : ℝ :=
  (25 * b) / (1320 * t)

theorem car_travel_distance (b t : ℝ) : 
  let distance_in_feet := (b / 3) * (300 / t)
  let distance_in_miles' := distance_in_feet / 5280
  distance_in_miles' = distance_in_miles b t := 
by
  sorry

end car_travel_distance_l16_16315


namespace min_value_of_f_l16_16934

open Real

def f (x : ℝ) : ℝ := 2 * cos x + sin (2 * x)

theorem min_value_of_f : ∃ x : ℝ, f x = -3 * sqrt 3 / 2 :=
by {
  -- Proof is omitted
  sorry
}

end min_value_of_f_l16_16934


namespace teachers_quit_before_lunch_percentage_l16_16741

variables (n_initial n_after_one_hour n_after_lunch n_quit_before_lunch : ℕ)

def initial_teachers : ℕ := 60
def teachers_after_one_hour (n_initial : ℕ) : ℕ := n_initial / 2
def teachers_after_lunch : ℕ := 21
def quit_before_lunch (n_after_one_hour n_after_lunch : ℕ) : ℕ := n_after_one_hour - n_after_lunch
def percentage_quit (n_quit_before_lunch n_after_one_hour : ℕ) : ℕ := (n_quit_before_lunch * 100) / n_after_one_hour

theorem teachers_quit_before_lunch_percentage :
  ∀ n_initial n_after_one_hour n_after_lunch n_quit_before_lunch,
  n_initial = initial_teachers →
  n_after_one_hour = teachers_after_one_hour n_initial →
  n_after_lunch = teachers_after_lunch →
  n_quit_before_lunch = quit_before_lunch n_after_one_hour n_after_lunch →
  percentage_quit n_quit_before_lunch n_after_one_hour = 30 := by 
    sorry

end teachers_quit_before_lunch_percentage_l16_16741


namespace quadratic_root_sum_eight_l16_16045

theorem quadratic_root_sum_eight (p r : ℝ) (hp : p > 0) (hr : r > 0) 
  (h : ∀ (x₁ x₂ : ℝ), (x₁ + x₂ = p) -> (x₁ * x₂ = r) -> (x₁ + x₂ = 8)) : r = 8 :=
sorry

end quadratic_root_sum_eight_l16_16045


namespace sum_mod_7_eq_5_l16_16596

theorem sum_mod_7_eq_5 : 
  (51730 + 51731 + 51732 + 51733 + 51734 + 51735) % 7 = 5 := 
by 
  sorry

end sum_mod_7_eq_5_l16_16596


namespace average_weight_of_all_girls_l16_16735

theorem average_weight_of_all_girls 
    (avg_weight_group1 : ℝ) (avg_weight_group2 : ℝ) 
    (num_girls_group1 : ℕ) (num_girls_group2 : ℕ) 
    (h1 : avg_weight_group1 = 50.25) 
    (h2 : avg_weight_group2 = 45.15) 
    (h3 : num_girls_group1 = 16) 
    (h4 : num_girls_group2 = 8) : 
    (avg_weight_group1 * num_girls_group1 + avg_weight_group2 * num_girls_group2) / (num_girls_group1 + num_girls_group2) = 48.55 := 
by 
    sorry

end average_weight_of_all_girls_l16_16735


namespace max_buses_in_city_l16_16648

/--
In a city with 9 bus stops where each bus serves exactly 3 stops and any two buses share at most one stop, 
prove that the maximum number of buses that can be in the city is 12.
-/
theorem max_buses_in_city :
  ∃ (B : Set (Finset ℕ)), 
    (∀ b ∈ B, b.card = 3) ∧ 
    (∀ b1 b2 ∈ B, b1 ≠ b2 → (b1 ∩ b2).card ≤ 1) ∧ 
    (∀ b ∈ B, ∀ s ∈ b, s < 9) ∧ 
    B.card = 12 := sorry

end max_buses_in_city_l16_16648


namespace constant_function_n_2_or_4_l16_16235

variables {α : Type*} [metric_space α]

def equilateral_triangle (A B C : α) : Prop := dist A B = dist B C ∧ dist B C = dist C A

def circumcircle (A B C : α) : set α :=
  { M | dist M A = dist M B ∧ dist M B = dist M C }

noncomputable def f (A B C M : α) (n : ℕ) : ℝ :=
  (dist M A) ^ n + (dist M B) ^ n + (dist M C) ^ n

theorem constant_function_n_2_or_4
  (A B C : α) (h : equilateral_triangle A B C) :
  ∀ (M ∈ circumcircle A B C), ∃ n ∈ {2, 4}, f A B C M n = f A B C (A) n :=
sorry

end constant_function_n_2_or_4_l16_16235


namespace number_of_cows_l16_16224

variable {D C : ℕ}

theorem number_of_cows (h : 2 * D + 4 * C = 2 * (D + C) + 24) : C = 12 :=
by sorry

end number_of_cows_l16_16224


namespace origin_inside_ellipse_iff_abs_k_range_l16_16213

theorem origin_inside_ellipse_iff_abs_k_range (k : ℝ) :
  (k^2 * 0^2 + 0^2 - 4 * k * 0 + 2 * k * 0 + k^2 - 1 < 0) ↔ (0 < |k| ∧ |k| < 1) :=
by sorry

end origin_inside_ellipse_iff_abs_k_range_l16_16213


namespace max_buses_in_city_l16_16642

-- Definition of the problem statement and the maximal number of buses
theorem max_buses_in_city (bus_stops : Finset ℕ) (buses : Finset (Finset ℕ)) 
  (h_stops : bus_stops.card = 9) 
  (h_buses_stops : ∀ b ∈ buses, b.card = 3) 
  (h_shared_stops : ∀ b₁ b₂ ∈ buses, b₁ ≠ b₂ → (b₁ ∩ b₂).card ≤ 1) : 
  buses.card ≤ 12 := 
sorry

end max_buses_in_city_l16_16642


namespace sufficient_but_not_necessary_condition_l16_16367

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) (h : x > 0) : (∃ y : ℝ, (y < -3 ∨ y > -1) ∧ y > 0) := by
  sorry

end sufficient_but_not_necessary_condition_l16_16367


namespace solve_system_l16_16694

theorem solve_system (x y z a : ℝ) 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = a^2) 
  (h3 : x^3 + y^3 + z^3 = a^3) : 
  (x = 0 ∧ y = 0 ∧ z = a) ∨ 
  (x = 0 ∧ y = a ∧ z = 0) ∨ 
  (x = a ∧ y = 0 ∧ z = 0) := 
sorry

end solve_system_l16_16694


namespace cyclic_sum_inequality_l16_16920

theorem cyclic_sum_inequality (x y z a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (x / (a * y + b * z)) + (y / (a * z + b * x)) + (z / (a * x + b * y)) ≥ 3 / (a + b) :=
  sorry

end cyclic_sum_inequality_l16_16920


namespace duration_of_loan_l16_16410

namespace SimpleInterest

variables (P SI R : ℝ) (T : ℝ)

-- Defining the conditions
def principal := P = 1500
def simple_interest := SI = 735
def rate := R = 7 / 100

-- The question: Prove the duration (T) of the loan
theorem duration_of_loan (hP : principal P) (hSI : simple_interest SI) (hR : rate R) :
  T = 7 :=
sorry

end SimpleInterest

end duration_of_loan_l16_16410


namespace negation_of_sum_of_squares_l16_16202

variables (a b : ℝ)

theorem negation_of_sum_of_squares:
  ¬(a^2 + b^2 = 0) → (a ≠ 0 ∨ b ≠ 0) := 
by
  sorry

end negation_of_sum_of_squares_l16_16202


namespace seats_per_row_and_total_students_l16_16515

theorem seats_per_row_and_total_students (R S : ℕ) 
  (h1 : S = 5 * R + 6) 
  (h2 : S = 12 * (R - 3)) : 
  R = 6 ∧ S = 36 := 
by 
  sorry

end seats_per_row_and_total_students_l16_16515


namespace salt_mixture_l16_16628

theorem salt_mixture (x y : ℝ) (p c z : ℝ) (hx : x = 50) (hp : p = 0.60) (hc : c = 0.40) (hy_eq : y = 50) :
  (50 * z) + (50 * 0.60) = 0.40 * (50 + 50) → (50 * z) + (50 * p) = c * (x + y) → y = 50 :=
by sorry

end salt_mixture_l16_16628


namespace like_terms_calc_l16_16208

theorem like_terms_calc {m n : ℕ} (h1 : m + 2 = 6) (h2 : n + 1 = 3) : (- (m : ℤ))^3 + (n : ℤ)^2 = -60 :=
  sorry

end like_terms_calc_l16_16208


namespace standard_circle_equation_passing_through_P_l16_16507

-- Define the condition that a point P is a solution to the system of equations derived from the line
def PointPCondition (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1 = 0) ∧ (3 * x - 2 * y + 5 = 0)

-- Define the center and radius of the given circle C
def CenterCircleC : ℝ × ℝ := (2, -3)
def RadiusCircleC : ℝ := 4  -- Since the radius squared is 16

-- Define the condition that a point is on a circle with a given center and radius
def OnCircle (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.fst)^2 + (y + center.snd)^2 = radius^2

-- State the problem
theorem standard_circle_equation_passing_through_P :
  ∃ (x y : ℝ), PointPCondition x y ∧ OnCircle CenterCircleC 5 x y :=
sorry

end standard_circle_equation_passing_through_P_l16_16507


namespace jericho_owes_annika_l16_16995

variable (J A M : ℝ)
variable (h1 : 2 * J = 60)
variable (h2 : M = A / 2)
variable (h3 : 30 - A - M = 9)

theorem jericho_owes_annika :
  A = 14 :=
by
  sorry

end jericho_owes_annika_l16_16995


namespace age_of_youngest_child_l16_16150

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65) : x = 7 :=
sorry

end age_of_youngest_child_l16_16150


namespace primitive_root_set_equality_l16_16523

theorem primitive_root_set_equality 
  {p : ℕ} (hp : Nat.Prime p) (hodd: p % 2 = 1) (g : ℕ) (hg : g ^ (p - 1) % p = 1) :
  (∀ k, 1 ≤ k ∧ k ≤ (p - 1) / 2 → ∃ m, 1 ≤ m ∧ m ≤ (p - 1) / 2 ∧ (k^2 + 1) % p = g ^ m % p) ↔ p = 3 :=
by sorry

end primitive_root_set_equality_l16_16523


namespace point_in_second_quadrant_l16_16423

def P : ℝ × ℝ := (-5, 4)

theorem point_in_second_quadrant (p : ℝ × ℝ) (hx : p.1 = -5) (hy : p.2 = 4) : p.1 < 0 ∧ p.2 > 0 :=
by
  sorry

example : P.1 < 0 ∧ P.2 > 0 :=
  point_in_second_quadrant P rfl rfl

end point_in_second_quadrant_l16_16423


namespace chocolate_bar_min_breaks_l16_16176

theorem chocolate_bar_min_breaks (n : ℕ) (h : n = 40) : ∃ k : ℕ, k = n - 1 := 
by 
  sorry

end chocolate_bar_min_breaks_l16_16176


namespace Z_bijective_H_l16_16107

open Set

noncomputable def H : Set ℚ :=
  { x | x = 1/2 ∨ (∃ y, y ∈ H ∧ (x = 1 / (1 + y) ∨ x = y / (1 + y))) }

theorem Z_bijective_H : ∃ (f : ℤ → ℚ), Function.Bijective f ∧ ∀ i, f i ∈ H := by
  sorry

end Z_bijective_H_l16_16107


namespace basket_A_apples_count_l16_16226

-- Conditions
def total_baskets : ℕ := 5
def avg_fruits_per_basket : ℕ := 25
def fruits_in_B : ℕ := 30
def fruits_in_C : ℕ := 20
def fruits_in_D : ℕ := 25
def fruits_in_E : ℕ := 35

-- Calculation of total number of fruits
def total_fruits : ℕ := total_baskets * avg_fruits_per_basket
def other_baskets_fruits : ℕ := fruits_in_B + fruits_in_C + fruits_in_D + fruits_in_E

-- Question and Proof Goal
theorem basket_A_apples_count : total_fruits - other_baskets_fruits = 15 := by
  sorry

end basket_A_apples_count_l16_16226


namespace find_value_of_x8_plus_x4_plus_1_l16_16613

theorem find_value_of_x8_plus_x4_plus_1 (x : ℂ) (hx : x^2 + x + 1 = 0) : x^8 + x^4 + 1 = 0 :=
sorry

end find_value_of_x8_plus_x4_plus_1_l16_16613


namespace amaya_total_marks_l16_16582

theorem amaya_total_marks 
  (m_a s_a a m m_s : ℕ) 
  (h_music : m_a = 70)
  (h_social_studies : s_a = m_a + 10)
  (h_maths_art_diff : m = a - 20)
  (h_maths_fraction : m = a - 1/10 * a)
  (h_maths_eq_fraction : m = 9/10 * a)
  (h_arts : 9/10 * a = a - 20)
  (h_total : m_a + s_a + a + m = 530) :
  m_a + s_a + a + m = 530 :=
by
  -- Proof to be completed
  sorry

end amaya_total_marks_l16_16582


namespace percentage_increase_in_efficiency_l16_16258

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end percentage_increase_in_efficiency_l16_16258


namespace rectangle_area_unchanged_l16_16835

theorem rectangle_area_unchanged
  (x y : ℝ)
  (h1 : x * y = (x + 3) * (y - 1))
  (h2 : x * y = (x - 3) * (y + 1.5)) :
  x * y = 31.5 :=
sorry

end rectangle_area_unchanged_l16_16835


namespace a_is_perfect_square_l16_16672

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l16_16672


namespace jane_mistake_l16_16463

theorem jane_mistake (x y z : ℤ) (h1 : x - y + z = 15) (h2 : x - y - z = 7) : x - y = 11 :=
by sorry

end jane_mistake_l16_16463


namespace find_angle_of_triangle_l16_16078

variable {a b l : ℝ}

theorem find_angle_of_triangle (h1 : a > 0) (h2 : b > 0) (h3 : l > 0) :
  let α := 2 * Real.arccos (l * (a + b) / (2 * a * b)) in
  α = 2 * Real.arccos (l * (a + b) / (2 * a * b)) :=
sorry

end find_angle_of_triangle_l16_16078


namespace smallest_positive_integer_satisfying_conditions_l16_16495

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (N : ℕ), N = 242 ∧
    ( ∃ (i : Fin 4), (N + i) % 8 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 9 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 25 = 0 ) ∧
    ( ∃ (i : Fin 4), (N + i) % 121 = 0 ) :=
sorry

end smallest_positive_integer_satisfying_conditions_l16_16495


namespace largest_among_four_theorem_l16_16379

noncomputable def largest_among_four (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) : Prop :=
  (a^2 + b^2 > 1) ∧ (a^2 + b^2 > 2 * a * b) ∧ (a^2 + b^2 > a)

theorem largest_among_four_theorem (a b : ℝ) (h1 : 0 < a ∧ a < b) (h2 : a + b = 1) :
  largest_among_four a b h1 h2 :=
sorry

end largest_among_four_theorem_l16_16379


namespace find_f_2011_l16_16460

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2011 :
  (∀ x : ℝ, f (x^2 + x) + 2 * f (x^2 - 3 * x + 2) = 9 * x^2 - 15 * x) →
  f 2011 = 6029 :=
by
  intros hf
  sorry

end find_f_2011_l16_16460


namespace first_discount_percentage_l16_16420

theorem first_discount_percentage (x : ℝ) 
  (h₁ : ∀ (p : ℝ), p = 70) 
  (h₂ : ∀ (d₁ d₂ : ℝ), d₁ = x / 100 ∧ d₂ = 0.01999999999999997 )
  (h₃ : ∀ (final_price : ℝ), final_price = 61.74):
  x = 10 := 
by
  sorry

end first_discount_percentage_l16_16420


namespace Tanya_efficiency_higher_l16_16254

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end Tanya_efficiency_higher_l16_16254


namespace find_third_number_in_second_set_l16_16137

theorem find_third_number_in_second_set (x y: ℕ) 
    (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
    (h2 : (128 + 255 + y + 1023 + x) / 5 = 423) 
: y = 511 := 
sorry

end find_third_number_in_second_set_l16_16137


namespace quadratic_inequality_range_l16_16905

theorem quadratic_inequality_range (a x : ℝ) :
  (∀ x : ℝ, x^2 - x - a^2 + a + 1 > 0) ↔ (-1/2 < a ∧ a < 3/2) :=
by
  sorry

end quadratic_inequality_range_l16_16905


namespace appears_in_31st_equation_l16_16994

theorem appears_in_31st_equation : 
  ∃ n : ℕ, 2016 ∈ {x | 2*x^2 ≤ 2016 ∧ 2016 < 2*(x+1)^2} ∧ n = 31 :=
by
  sorry

end appears_in_31st_equation_l16_16994


namespace find_A_l16_16465

theorem find_A (A B C D E F G H I J : ℕ)
  (h1 : A > B ∧ B > C)
  (h2 : D > E ∧ E > F)
  (h3 : G > H ∧ H > I ∧ I > J)
  (h4 : (D = E + 2) ∧ (E = F + 2))
  (h5 : (G = H + 2) ∧ (H = I + 2) ∧ (I = J + 2))
  (h6 : A + B + C = 10) : A = 6 :=
sorry

end find_A_l16_16465


namespace cos_30_eq_sqrt3_div_2_l16_16758

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := 
by
  sorry

end cos_30_eq_sqrt3_div_2_l16_16758


namespace total_rowing_time_l16_16170

theorem total_rowing_time (s_b : ℕ) (s_s : ℕ) (d : ℕ) : 
  s_b = 9 → s_s = 6 → d = 170 → 
  (d / (s_b + s_s) + d / (s_b - s_s)) = 68 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_rowing_time_l16_16170


namespace actual_average_height_is_correct_l16_16871

-- Definitions based on given conditions
def number_of_students : ℕ := 20
def incorrect_average_height : ℝ := 175.0
def incorrect_height_of_student : ℝ := 151.0
def actual_height_of_student : ℝ := 136.0

-- Prove that the actual average height is 174.25 cm
theorem actual_average_height_is_correct :
  (incorrect_average_height * number_of_students - (incorrect_height_of_student - actual_height_of_student)) / number_of_students = 174.25 :=
sorry

end actual_average_height_is_correct_l16_16871


namespace unique_sequence_l16_16053

theorem unique_sequence (a : ℕ → ℝ) 
  (h1 : a 0 = 1) 
  (h2 : ∀ n : ℕ, a n > 0) 
  (h3 : ∀ n : ℕ, a n - a (n + 1) = a (n + 2)) : 
  ∀ n : ℕ, a n = ( (-1 + Real.sqrt 5) / 2)^n := 
sorry

end unique_sequence_l16_16053


namespace sum_of_interior_angles_of_pentagon_l16_16287

theorem sum_of_interior_angles_of_pentagon :
    (5 - 2) * 180 = 540 := by 
  -- The proof goes here
  sorry

end sum_of_interior_angles_of_pentagon_l16_16287


namespace average_annual_growth_rate_in_2014_and_2015_l16_16571

noncomputable def average_annual_growth_rate (p2013 p2015 : ℝ) (x : ℝ) : Prop :=
  p2013 * (1 + x)^2 = p2015

theorem average_annual_growth_rate_in_2014_and_2015 :
  average_annual_growth_rate 6.4 10 0.25 :=
by
  unfold average_annual_growth_rate
  sorry

end average_annual_growth_rate_in_2014_and_2015_l16_16571


namespace sin_law_ratio_l16_16807

theorem sin_law_ratio {A B C : ℝ} {a b c : ℝ} (hA : a = 1) (hSinA : Real.sin A = 1 / 3) :
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 3 := 
  sorry

end sin_law_ratio_l16_16807


namespace graphs_symmetric_l16_16230

noncomputable def exp2 : ℝ → ℝ := λ x => 2^x
noncomputable def log2 : ℝ → ℝ := λ x => Real.log x / Real.log 2

theorem graphs_symmetric :
  ∀ (x y : ℝ), (y = exp2 x) ↔ (x = log2 y) := sorry

end graphs_symmetric_l16_16230


namespace cos_30_deg_l16_16761

theorem cos_30_deg : 
  ∃ Q : ℝ × ℝ, Q = (cos (π / 6), sin (π / 6)) → 
  cos (π / 6) = (√3) / 2 :=
by
  use (cos (π / 6), sin (π / 6))
  sorry

end cos_30_deg_l16_16761


namespace find_y_values_l16_16474

theorem find_y_values
  (y₁ y₂ y₃ y₄ y₅ : ℝ)
  (h₁ : y₁ + 3 * y₂ + 6 * y₃ + 10 * y₄ + 15 * y₅ = 3)
  (h₂ : 3 * y₁ + 6 * y₂ + 10 * y₃ + 15 * y₄ + 21 * y₅ = 20)
  (h₃ : 6 * y₁ + 10 * y₂ + 15 * y₃ + 21 * y₄ + 28 * y₅ = 86)
  (h₄ : 10 * y₁ + 15 * y₂ + 21 * y₃ + 28 * y₄ + 36 * y₅ = 225) :
  15 * y₁ + 21 * y₂ + 28 * y₃ + 36 * y₄ + 45 * y₅ = 395 :=
by {
  sorry
}

end find_y_values_l16_16474


namespace dima_can_find_heavy_ball_l16_16048

noncomputable def find_heavy_ball
  (balls : Fin 9) -- 9 balls, indexed from 0 to 8 representing the balls 1 to 9
  (heavy : Fin 9) -- One of the balls is heavier
  (weigh : Fin 9 → Fin 9 → Ordering) -- A function that compares two groups of balls and gives an Ordering: .lt, .eq, or .gt
  (predetermined_sets : List (Fin 9 × Fin 9)) -- A list of tuples representing balls on each side for the two weighings
  (valid_sets : predetermined_sets.length ≤ 2) : Prop := -- Not more than two weighings
  ∃ idx : Fin 9, idx = heavy -- Need to prove that we can always find the heavier ball

theorem dima_can_find_heavy_ball :
  ∀ (balls : Fin 9) (heavy : Fin 9)
    (weigh : Fin 9 → Fin 9 → Ordering)
    (predetermined_sets : List (Fin 9 × Fin 9))
    (valid_sets : predetermined_sets.length ≤ 2),
  find_heavy_ball balls heavy weigh predetermined_sets valid_sets :=
sorry -- Proof is omitted

end dima_can_find_heavy_ball_l16_16048


namespace polynomial_calculation_l16_16038

theorem polynomial_calculation :
  (49^5 - 5 * 49^4 + 10 * 49^3 - 10 * 49^2 + 5 * 49 - 1) = 254804368 :=
by
  sorry

end polynomial_calculation_l16_16038


namespace find_k_l16_16139

-- Given conditions and hypothesis stated
axiom quadratic_eq (x k : ℝ) : x^2 + 10 * x + k = 0

def roots_in_ratio_3_1 (α β : ℝ) : Prop :=
  α / β = 3

-- Statement of the theorem to be proved
theorem find_k {α β k : ℝ} (h1 : quadratic_eq α k) (h2 : quadratic_eq β k)
               (h3 : α ≠ 0) (h4 : β ≠ 0) (h5 : roots_in_ratio_3_1 α β) :
  k = 18.75 :=
by
  sorry

end find_k_l16_16139


namespace stratified_sampling_counts_l16_16175

-- Defining the given conditions
def num_elderly : ℕ := 27
def num_middle_aged : ℕ := 54
def num_young : ℕ := 81
def total_sample : ℕ := 42

-- Proving the required stratified sample counts
theorem stratified_sampling_counts :
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  elderly_count = 7 ∧ middle_aged_count = 14 ∧ young_count = 21 :=
by 
  let ratio_elderly := 1
  let ratio_middle_aged := 2
  let ratio_young := 3
  let total_ratio := ratio_elderly + ratio_middle_aged + ratio_young
  let elderly_count := (ratio_elderly * total_sample) / total_ratio
  let middle_aged_count := (ratio_middle_aged * total_sample) / total_ratio
  let young_count := (ratio_young * total_sample) / total_ratio
  have h1 : elderly_count = 7 := by sorry
  have h2 : middle_aged_count = 14 := by sorry
  have h3 : young_count = 21 := by sorry
  exact ⟨h1, h2, h3⟩

end stratified_sampling_counts_l16_16175


namespace initial_birds_in_cage_l16_16554

-- Define a theorem to prove the initial number of birds in the cage
theorem initial_birds_in_cage (B : ℕ) 
  (H1 : 2 / 15 * B = 8) : B = 60 := 
by sorry

end initial_birds_in_cage_l16_16554


namespace original_cost_of_statue_l16_16520

theorem original_cost_of_statue (sale_price : ℝ) (profit_percent : ℝ) (original_cost : ℝ) 
  (h1 : sale_price = 620) 
  (h2 : profit_percent = 0.25) 
  (h3 : sale_price = (1 + profit_percent) * original_cost) : 
  original_cost = 496 :=
by
  sorry

end original_cost_of_statue_l16_16520


namespace lcm_24_36_42_l16_16913

-- Definitions of the numbers involved
def a : ℕ := 24
def b : ℕ := 36
def c : ℕ := 42

-- Statement for the lowest common multiple
theorem lcm_24_36_42 : Nat.lcm (Nat.lcm a b) c = 504 :=
by
  -- The proof will be filled in here
  sorry

end lcm_24_36_42_l16_16913


namespace smallest_square_area_l16_16312

theorem smallest_square_area (a b c d : ℕ) (hsquare : ∃ s : ℕ, s ≥ a + c ∧ s * s = a * b + c * d) :
    (a = 3) → (b = 5) → (c = 4) → (d = 6) → ∃ s : ℕ, s * s = 49 :=
by
  intros h1 h2 h3 h4
  cases hsquare with s hs
  use s
  -- Here we need to ensure s * s = 49
  sorry

end smallest_square_area_l16_16312


namespace parametric_line_eq_l16_16983

theorem parametric_line_eq (t : ℝ) : 
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x = 3 * t + 6 → y = 5 * t - 8 → y = m * x + b)) ∧ m = 5 / 3 ∧ b = -18 :=
sorry

end parametric_line_eq_l16_16983


namespace geometric_common_ratio_of_arithmetic_seq_l16_16386

theorem geometric_common_ratio_of_arithmetic_seq 
  (a : ℕ → ℝ) (d q : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 2)
  (h_nonzero_diff : d ≠ 0)
  (h_geo_seq : a 1 = 2 ∧ a 3 = 2 * q ∧ a 11 = 2 * q^2) : 
  q = 4 := 
by
  sorry

end geometric_common_ratio_of_arithmetic_seq_l16_16386


namespace F5_div_641_Fermat_rel_prime_l16_16491

def Fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

theorem F5_div_641 : Fermat_number 5 % 641 = 0 := 
  sorry

theorem Fermat_rel_prime (k n : ℕ) (hk: k ≠ n) : Nat.gcd (Fermat_number k) (Fermat_number n) = 1 :=
  sorry

end F5_div_641_Fermat_rel_prime_l16_16491


namespace farthest_points_hyperbola_l16_16438

noncomputable def farthest_points_locus (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | (P.1 ^ 2 - P.2 ^ 2 = (dist A B / 2)^2)}

theorem farthest_points_hyperbola (A B : ℝ × ℝ) :
  farthest_points_locus A B = {P | (P.1 ^ 2 - P.2 ^ 2) = (dist A B / 2)^2} :=
  sorry

end farthest_points_hyperbola_l16_16438


namespace systematic_sampling_first_number_l16_16063

theorem systematic_sampling_first_number
    (n : ℕ)  -- total number of products
    (k : ℕ)  -- sample size
    (common_diff : ℕ)  -- common difference in the systematic sample
    (x : ℕ)  -- an element in the sample
    (first_num : ℕ)  -- first product number in the sample
    (h1 : n = 80)  -- total number of products is 80
    (h2 : k = 5)  -- sample size is 5
    (h3 : common_diff = 16)  -- common difference is 16
    (h4 : x = 42)  -- 42 is in the sample
    (h5 : x = common_diff * 2 + first_num)  -- position of 42 in the arithmetic sequence
: first_num = 10 := 
sorry

end systematic_sampling_first_number_l16_16063


namespace triangle_transform_same_l16_16468

def Point := ℝ × ℝ

def reflect_x (p : Point) : Point :=
(p.1, -p.2)

def rotate_180 (p : Point) : Point :=
(-p.1, -p.2)

def reflect_y (p : Point) : Point :=
(-p.1, p.2)

def transform (p : Point) : Point :=
reflect_y (rotate_180 (reflect_x p))

theorem triangle_transform_same (A B C : Point) :
A = (2, 1) → B = (4, 1) → C = (2, 3) →
(transform A = (2, 1) ∧ transform B = (4, 1) ∧ transform C = (2, 3)) :=
by
  intros
  sorry

end triangle_transform_same_l16_16468


namespace find_k_from_roots_ratio_l16_16141

theorem find_k_from_roots_ratio (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = -10)
  (h2 : x1 * x2 = k)
  (h3 : x1/x2 = 3) : 
  k = 18.75 := 
sorry

end find_k_from_roots_ratio_l16_16141


namespace option_A_not_correct_option_B_correct_option_C_correct_option_D_correct_l16_16730

theorem option_A_not_correct 
  (x : ℝ) : ¬ (∀ y, y = (x^2 + 1)/x → y ≥ 2) := 
sorry

theorem option_B_correct 
  (x y : ℝ) (h : x > 1) (hy : y = 2x + (4 / (x - 1)) - 1) : 
  y ≥ 4 * Real.sqrt 2 + 1 :=
sorry

theorem option_C_correct 
  {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 3 * x * y) : 
  2 * x + y ≥ 3 := 
sorry

theorem option_D_correct 
  {x y : ℝ} (h : 9 * x^2 + y^2 + x * y = 1) : 
  3 * x + y ≤ (2 * Real.sqrt 21) / 7 := 
sorry

end option_A_not_correct_option_B_correct_option_C_correct_option_D_correct_l16_16730


namespace a_le_neg2_l16_16710

theorem a_le_neg2 (a : ℝ) : (∀ x : ℝ, (x + 5 > 3) → (x > a)) → a ≤ -2 :=
by
  intro h
  have h_neg : ∀ x : ℝ, (x > -2) → (x > a) := 
    by 
      intro x hx
      exact h x (by linarith)

  specialize h_neg (-1) (by linarith)
  linarith

end a_le_neg2_l16_16710


namespace m_necessary_not_sufficient_cond_l16_16425

theorem m_necessary_not_sufficient_cond (m : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^3 - 3 * x + m = 0) → m ≤ 2 :=
sorry

end m_necessary_not_sufficient_cond_l16_16425


namespace part1_part2_part3_l16_16369

-- Definitions based on conditions
def fractional_eq (x a : ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Part (1): Proof statement for a == -1 if x == 5 is a root
theorem part1 (x : ℝ) (a : ℝ) (h : x = 5) (heq : fractional_eq x a) : a = -1 :=
sorry

-- Part (2): Proof statement for a == 2 if the equation has a double root
theorem part2 (a : ℝ) (h_double_root : ∀ x, fractional_eq x a → x = 0 ∨ x = 2) : a = 2 :=
sorry

-- Part (3): Proof statement for a == -3 or == 2 if the equation has no solution
theorem part3 (a : ℝ) (h_no_solution : ¬∃ x, fractional_eq x a) : a = -3 ∨ a = 2 :=
sorry

end part1_part2_part3_l16_16369


namespace third_term_geometric_series_l16_16780

variable {b1 b3 q : ℝ}
variable (hb1 : b1 * (-1/4) = -1/2)
variable (hs : b1 / (1 - q) = 8/5)
variable (hq : |q| < 1)

theorem third_term_geometric_series (hb1 : b1 * (-1 / 4) = -1 / 2)
  (hs : b1 / (1 - q) = 8 / 5)
  (hq : |q| < 1)
  : b3 = b1 * q^2 := by
    sorry

end third_term_geometric_series_l16_16780


namespace probability_club_then_heart_eq_13_over_204_l16_16558

noncomputable def deck_prob : ℝ :=
  let prob_first_club := (13 : ℝ) / 52
  let prob_second_heart := (13 : ℝ) / 51
  prob_first_club * prob_second_heart

theorem probability_club_then_heart_eq_13_over_204 :
  deck_prob = 13 / 204 :=
by
  sorry

end probability_club_then_heart_eq_13_over_204_l16_16558


namespace tamara_total_earnings_l16_16583

-- Definitions derived from the conditions in the problem statement.
def pans : ℕ := 2
def pieces_per_pan : ℕ := 8
def price_per_piece : ℕ := 2

-- Theorem stating the required proof problem.
theorem tamara_total_earnings : 
  (pans * pieces_per_pan * price_per_piece) = 32 :=
by
  sorry

end tamara_total_earnings_l16_16583


namespace waynes_son_time_to_shovel_l16_16733

-- Definitions based on the conditions
variables (S W : ℝ) (son_rate : S = 1 / 21) (wayne_rate : W = 6 * S) (together_rate : 3 * (S + W) = 1)

theorem waynes_son_time_to_shovel : 
  1 / S = 21 :=
by
  -- Proof will be provided later
  sorry

end waynes_son_time_to_shovel_l16_16733


namespace percentage_increase_in_efficiency_l16_16257

def sEfficiency : ℚ := 1 / 20
def tEfficiency : ℚ := 1 / 16

theorem percentage_increase_in_efficiency :
    ((tEfficiency - sEfficiency) / sEfficiency) * 100 = 25 :=
by
  sorry

end percentage_increase_in_efficiency_l16_16257


namespace at_least_one_less_than_two_l16_16784

theorem at_least_one_less_than_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b > 2) :
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
sorry

end at_least_one_less_than_two_l16_16784


namespace hyperbola_eccentricity_is_correct_l16_16543

/-- Definition of a hyperbola centered at the origin with given conditions -/
def hyperbola_C (x y : ℝ) : Prop :=
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a^2 * y^2 - b^2 * x^2 = a^2 * b^2)

/-- Definition of the circle -/
def circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 1

/-- The asymptotes of the hyperbola C -/
def asymptotes_tangent_to_circle (k : ℝ) : Prop :=
  abs (2 * k) = sqrt (k^2 + 1)

/-- Eccentricity of hyperbola C -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_correct {a b : ℝ} (h_asymptotes : ∃ k, asymptotes_tangent_to_circle k)
  (h_C : hyperbola_C a b) : eccentricity a b = 2 * sqrt 3 / 3 ∨ eccentricity a b = 2 :=
sorry

end hyperbola_eccentricity_is_correct_l16_16543


namespace number_of_organizations_in_foundation_l16_16321

def company_raised : ℕ := 2500
def donation_percentage : ℕ := 80
def each_organization_receives : ℕ := 250
def total_donated : ℕ := (donation_percentage * company_raised) / 100

theorem number_of_organizations_in_foundation : total_donated / each_organization_receives = 8 :=
by
  sorry

end number_of_organizations_in_foundation_l16_16321


namespace sum_123_consecutive_even_numbers_l16_16060

theorem sum_123_consecutive_even_numbers :
  let n := 123
  let a := 2
  let d := 2
  let sum_arithmetic_series (n a l : ℕ) := n * (a + l) / 2
  let last_term := a + (n - 1) * d
  sum_arithmetic_series n a last_term = 15252 :=
by
  sorry

end sum_123_consecutive_even_numbers_l16_16060


namespace rate_of_interest_l16_16148

theorem rate_of_interest (P T SI: ℝ) (h1 : P = 2500) (h2 : T = 5) (h3 : SI = P - 2000) (h4 : SI = (P * R * T) / 100):
  R = 4 :=
by
  sorry

end rate_of_interest_l16_16148


namespace positive_difference_abs_eq_15_l16_16002

theorem positive_difference_abs_eq_15:
  ∃ x1 x2 : ℝ, (| x1 - 3 | = 15 ∧ | x2 - 3 | = 15) ∧ | x1 - x2 | = 30 :=
by
  sorry

end positive_difference_abs_eq_15_l16_16002


namespace lisa_flew_distance_l16_16116

-- Define the given conditions
def speed := 32  -- speed in miles per hour
def time := 8    -- time in hours

-- Define the derived distance
def distance := speed * time  -- using the formula Distance = Speed × Time

-- Prove that the calculated distance is 256 miles
theorem lisa_flew_distance : distance = 256 :=
by
  sorry

end lisa_flew_distance_l16_16116


namespace miles_driven_each_day_l16_16489

theorem miles_driven_each_day
  (total_distance : ℕ)
  (days_in_semester : ℕ)
  (h_total : total_distance = 1600)
  (h_days : days_in_semester = 80):
  total_distance / days_in_semester = 20 := by
  sorry

end miles_driven_each_day_l16_16489


namespace find_a_solve_inequality_l16_16368

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x + a) / (2^x - 1)

theorem find_a (h : ∀ x : ℝ, f x a = -f (-x) a) : a = 1 := sorry

theorem solve_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : f x 1 > 3 := sorry

end find_a_solve_inequality_l16_16368


namespace second_hand_degree_per_minute_l16_16308

theorem second_hand_degree_per_minute :
  (∀ (t : ℝ), t = 60 → 360 / t = 6) :=
by
  intro t
  intro ht
  rw [ht]
  norm_num

end second_hand_degree_per_minute_l16_16308


namespace age_difference_l16_16476

theorem age_difference (b_age : ℕ) (bro_age : ℕ) (h1 : b_age = 5) (h2 : b_age + bro_age = 19) : 
  bro_age - b_age = 9 :=
by
  sorry

end age_difference_l16_16476


namespace cos_30_deg_l16_16760

theorem cos_30_deg : 
  ∃ Q : ℝ × ℝ, Q = (cos (π / 6), sin (π / 6)) → 
  cos (π / 6) = (√3) / 2 :=
by
  use (cos (π / 6), sin (π / 6))
  sorry

end cos_30_deg_l16_16760


namespace range_of_q_l16_16362

variable (a_n : ℕ → ℝ) (q : ℝ) (S_n : ℕ → ℝ)
variable (hg_seq : ∀ n : ℕ, n > 0 → ∃ a_1 : ℝ, S_n n = a_1 * (1 - q ^ n) / (1 - q))
variable (pos_sum : ∀ n : ℕ, n > 0 → S_n n > 0)

theorem range_of_q : q ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (0 : ℝ) := sorry

end range_of_q_l16_16362


namespace distance_A_to_B_is_7km_l16_16297

theorem distance_A_to_B_is_7km
  (v1 v2 : ℝ) 
  (t_meet_before : ℝ)
  (t1_after_meet t2_after_meet : ℝ)
  (d1_before_meet d2_before_meet : ℝ)
  (d_after_meet : ℝ)
  (h1 : d1_before_meet = d2_before_meet + 1)
  (h2 : t_meet_before = d1_before_meet / v1)
  (h3 : t_meet_before = d2_before_meet / v2)
  (h4 : t1_after_meet = 3 / 4)
  (h5 : t2_after_meet = 4 / 3)
  (h6 : d1_before_meet + v1 * t1_after_meet = d_after_meet)
  (h7 : d2_before_meet + v2 * t2_after_meet = d_after_meet)
  : d_after_meet = 7 := 
sorry

end distance_A_to_B_is_7km_l16_16297


namespace greatest_drop_in_price_is_May_l16_16277

def priceChangeJan := -1.25
def priceChangeFeb := 2.75
def priceChangeMar := -0.75
def priceChangeApr := 1.50
def priceChangeMay := -3.00
def priceChangeJun := -1.00

theorem greatest_drop_in_price_is_May :
  priceChangeMay < priceChangeJan ∧
  priceChangeMay < priceChangeMar ∧
  priceChangeMay < priceChangeApr ∧
  priceChangeMay < priceChangeJun ∧
  priceChangeMay < priceChangeFeb :=
by sorry

end greatest_drop_in_price_is_May_l16_16277


namespace area_of_rectangle_l16_16451

-- Define the given conditions
def length : Real := 5.9
def width : Real := 3
def expected_area : Real := 17.7

theorem area_of_rectangle : (length * width) = expected_area := 
by 
  sorry

end area_of_rectangle_l16_16451


namespace tank_fill_rate_l16_16162

theorem tank_fill_rate
  (length width depth : ℝ)
  (time_to_fill : ℝ)
  (h_length : length = 10)
  (h_width : width = 6)
  (h_depth : depth = 5)
  (h_time : time_to_fill = 60) : 
  (length * width * depth) / time_to_fill = 5 :=
by
  -- Proof would go here
  sorry

end tank_fill_rate_l16_16162


namespace Tanya_efficiency_higher_l16_16256

variable (Sakshi_days Tanya_days : ℕ)
variable (Sakshi_efficiency Tanya_efficiency increase_in_efficiency percentage_increase : ℚ)

theorem Tanya_efficiency_higher (h1: Sakshi_days = 20) (h2: Tanya_days = 16) :
  Sakshi_efficiency = 1 / 20 ∧ Tanya_efficiency = 1 / 16 ∧ 
  increase_in_efficiency = Tanya_efficiency - Sakshi_efficiency ∧ 
  percentage_increase = (increase_in_efficiency / Sakshi_efficiency) * 100 ∧
  percentage_increase = 25 := by
  sorry

end Tanya_efficiency_higher_l16_16256


namespace find_digits_l16_16494

theorem find_digits (x y z : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) (h3 : 0 ≤ z ∧ z ≤ 9) :
  (10 * x + 5) * (3 * 100 + y * 10 + z) = 7850 ↔ (x = 2 ∧ y = 1 ∧ z = 4) :=
by
  sorry

end find_digits_l16_16494


namespace cos_30_degrees_l16_16767

-- Defining the problem context
def unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem cos_30_degrees : 
  let Q := unit_circle (30 * Real.pi / 180) in -- 30 degrees in radians
  (Q.1 = (Real.sqrt 3) / 2) :=
by
  sorry

end cos_30_degrees_l16_16767


namespace tan_theta_value_l16_16513

open Real

theorem tan_theta_value (θ : ℝ) (h : sin (θ / 2) - 2 * cos (θ / 2) = 0) : tan θ = -4 / 3 :=
sorry

end tan_theta_value_l16_16513


namespace f_even_f_mono_decreasing_l16_16370

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 1 / (2 * x^2)

theorem f_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  simp [f, pow_two, neg_sq]

theorem f_mono_decreasing : ∀ ⦃x1 x2 : ℝ⦄, 0 < x1 → x1 < x2 → x2 < real.sqrt 2 / 2 → f x1 > f x2 := by
  sorry

end f_even_f_mono_decreasing_l16_16370


namespace three_digit_numbers_div_by_17_l16_16376

theorem three_digit_numbers_div_by_17 : ∃ n : ℕ, n = 53 ∧ 
  let min_k := Nat.ceil (100 / 17)
  let max_k := Nat.floor (999 / 17)
  min_k = 6 ∧ max_k = 58 ∧ (max_k - min_k + 1) = n :=
by
  sorry

end three_digit_numbers_div_by_17_l16_16376


namespace fraction_juniors_study_Japanese_l16_16898

-- Define the size of the junior and senior classes
variable (J S : ℕ)

-- Condition 1: The senior class is twice the size of the junior class
axiom senior_twice_junior : S = 2 * J

-- The fraction of the seniors studying Japanese
noncomputable def fraction_seniors_study_Japanese : ℚ := 3 / 8

-- The total fraction of students in both classes that study Japanese
noncomputable def fraction_total_study_Japanese : ℚ := 1 / 3

-- Define the unknown fraction of juniors studying Japanese
variable (x : ℚ)

-- The proof problem transformed from the questions and the correct answer
theorem fraction_juniors_study_Japanese :
  (fraction_seniors_study_Japanese * ↑S + x * ↑J = fraction_total_study_Japanese * (↑J + ↑S)) → (x = 1 / 4) :=
by
  -- We use the given conditions and solve for x
  sorry

end fraction_juniors_study_Japanese_l16_16898


namespace extremum_range_l16_16215

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * x^2 - a * x + 1

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 4 * x - a

theorem extremum_range 
  (h : ∀ a : ℝ, (∃ (x : ℝ) (hx : -1 < x ∧ x < 1), f_prime a x = 0) → 
                (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime a x ≠ 0)):
  ∀ a : ℝ, -1 < a ∧ a < 7 :=
sorry

end extremum_range_l16_16215


namespace harry_carries_buckets_rounds_l16_16918

noncomputable def george_rate := 2
noncomputable def total_buckets := 110
noncomputable def total_rounds := 22
noncomputable def harry_buckets_each_round := 3

theorem harry_carries_buckets_rounds :
  (george_rate * total_rounds + harry_buckets_each_round * total_rounds = total_buckets) :=
by sorry

end harry_carries_buckets_rounds_l16_16918


namespace sum_of_squares_of_solutions_l16_16603

theorem sum_of_squares_of_solutions :
  (∀ x, (abs (x^2 - x + (1 / 2010)) = (1 / 2010)) → (x = 0 ∨ x = 1 ∨ (∃ a b, 
  a + b = 1 ∧ a * b = 1 / 1005 ∧ x = a ∧ x = b))) →
  ((0^2 + 1^2) + (1 - 2 * (1 / 1005)) = (2008 / 1005)) :=
by
  intro h
  sorry

end sum_of_squares_of_solutions_l16_16603


namespace sequence_period_16_l16_16046

theorem sequence_period_16 (a : ℝ) (h : a > 0) 
  (u : ℕ → ℝ) (h1 : u 1 = a) (h2 : ∀ n, u (n + 1) = -1 / (u n + 1)) : 
  u 16 = a :=
sorry

end sequence_period_16_l16_16046


namespace sphere_hemisphere_radius_relationship_l16_16462

theorem sphere_hemisphere_radius_relationship (r : ℝ) (R : ℝ) (π : ℝ) (h : 0 < π):
  (4 / 3) * π * R^3 = (2 / 3) * π * r^3 →
  r = 3 * (2^(1/3 : ℝ)) →
  R = 3 :=
by
  sorry

end sphere_hemisphere_radius_relationship_l16_16462


namespace average_weight_ten_students_l16_16824

theorem average_weight_ten_students (avg_wt_girls avg_wt_boys : ℕ) 
  (count_girls count_boys : ℕ)
  (h1 : count_girls = 5) 
  (h2 : avg_wt_girls = 45) 
  (h3 : count_boys = 5) 
  (h4 : avg_wt_boys = 55) : 
  (count_girls * avg_wt_girls + count_boys * avg_wt_boys) / (count_girls + count_boys) = 50 :=
by sorry

end average_weight_ten_students_l16_16824


namespace exist_m_squared_plus_9_mod_2_pow_n_minus_1_l16_16599

theorem exist_m_squared_plus_9_mod_2_pow_n_minus_1 (n : ℕ) (hn : n > 0) :
  (∃ m : ℤ, (m^2 + 9) % (2^n - 1) = 0) ↔ ∃ k : ℕ, n = 2^k :=
by
  sorry

end exist_m_squared_plus_9_mod_2_pow_n_minus_1_l16_16599


namespace simplify_and_evaluate_expr_l16_16538

theorem simplify_and_evaluate_expr (a b : ℤ) (h₁ : a = -1) (h₂ : b = 2) :
  (2 * a + b - 2 * (3 * a - 2 * b)) = 14 := by
  rw [h₁, h₂]
  sorry

end simplify_and_evaluate_expr_l16_16538


namespace find_selling_price_functional_relationship_and_max_find_value_of_a_l16_16316

section StoreProduct

variable (x : ℕ) (y : ℕ) (a k b : ℝ)

-- Definitions for the given conditions
def cost_price : ℝ := 50
def selling_price := x 
def sales_quantity := y 
def future_cost_increase := a

-- Given points
def point1 : ℝ × ℕ := (55, 90) 
def point2 : ℝ × ℕ := (65, 70)

-- Linear relationship between selling price and sales quantity
def linearfunc := y = k * x + b

-- Proof of the first statement
theorem find_selling_price (k := -2) (b := 200) : 
    (profit = 800 → (x = 60 ∨ x = 90)) :=
by
  -- People prove the theorem here
  sorry

-- Proof for the functional relationship between W and x
theorem functional_relationship_and_max (x := 75) : 
    W = -2*x^2 + 300*x - 10000 ∧ W_max = 1250 :=
by
  -- People prove the theorem here
  sorry

-- Proof for the value of a when the cost price increases
theorem find_value_of_a (cost_increase := 4) : 
    (W'_max = 960 → a = 4) :=
by
  -- People prove the theorem here
  sorry

end StoreProduct

end find_selling_price_functional_relationship_and_max_find_value_of_a_l16_16316


namespace marbles_in_jar_l16_16153

theorem marbles_in_jar (T : ℕ) (T_half : T / 2 = 12) (red_marbles : ℕ) (orange_marbles : ℕ) (total_non_blue : red_marbles + orange_marbles = 12) (red_count : red_marbles = 6) (orange_count : orange_marbles = 6) : T = 24 :=
by
  sorry

end marbles_in_jar_l16_16153


namespace intersection_in_second_quadrant_l16_16442

theorem intersection_in_second_quadrant (k : ℝ) (x y : ℝ) 
  (hk : 0 < k) (hk2 : k < 1/2) 
  (h1 : k * x - y = k - 1) 
  (h2 : k * y - x = 2 * k) : 
  x < 0 ∧ y > 0 := 
sorry

end intersection_in_second_quadrant_l16_16442


namespace find_m_value_l16_16617

variable (m a0 a1 a2 a3 a4 a5 : ℚ)

-- Defining the conditions given in the problem
def poly_expansion_condition : Prop := (m * 1 - 1)^5 = a5 * 1^5 + a4 * 1^4 + a3 * 1^3 + a2 * 1^2 + a1 * 1 + a0
def a1_a2_a3_a4_a5_condition : Prop := a1 + a2 + a3 + a4 + a5 = 33

-- We are required to prove that given these conditions, m = 3.
theorem find_m_value (h1 : a0 = -1) (h2 : poly_expansion_condition m a0 a1 a2 a3 a4 a5) 
(h3 : a1_a2_a3_a4_a5_condition a1 a2 a3 a4 a5) : m = 3 := by
  sorry

end find_m_value_l16_16617


namespace evaluate_polynomial_at_neg2_l16_16049

theorem evaluate_polynomial_at_neg2 : 2 * (-2)^4 + 3 * (-2)^3 + 5 * (-2)^2 + (-2) + 4 = 30 :=
by 
  sorry

end evaluate_polynomial_at_neg2_l16_16049


namespace negate_proposition_l16_16364

theorem negate_proposition :
  (¬(∀ x : ℝ, x^2 + x + 1 ≠ 0)) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) :=
by
  sorry

end negate_proposition_l16_16364


namespace distance_from_dormitory_to_city_l16_16169

theorem distance_from_dormitory_to_city (D : ℝ) :
  (1 / 4) * D + (1 / 2) * D + 10 = D → D = 40 :=
by
  intro h
  sorry

end distance_from_dormitory_to_city_l16_16169


namespace find_distance_between_sides_of_trapezium_l16_16912

variable (side1 side2 h area : ℝ)
variable (h1 : side1 = 20)
variable (h2 : side2 = 18)
variable (h3 : area = 228)
variable (trapezium_area : area = (1 / 2) * (side1 + side2) * h)

theorem find_distance_between_sides_of_trapezium : h = 12 := by
  sorry

end find_distance_between_sides_of_trapezium_l16_16912


namespace value_of_expression_l16_16950

theorem value_of_expression (x : ℝ) (h : x^2 + x + 1 = 8) : 4 * x^2 + 4 * x + 9 = 37 :=
by
  sorry

end value_of_expression_l16_16950


namespace value_of_a_l16_16727

theorem value_of_a (a : ℝ) : (1 / (Real.log 3 / Real.log a) + 1 / (Real.log 4 / Real.log a) + 1 / (Real.log 5 / Real.log a) = 1) → a = 60 :=
by
  sorry

end value_of_a_l16_16727


namespace xiao_wang_programming_methods_l16_16166

theorem xiao_wang_programming_methods :
  ∃ (n : ℕ), n = 20 :=
by sorry

end xiao_wang_programming_methods_l16_16166


namespace cos_30_degrees_l16_16766

-- Defining the problem context
def unit_circle (θ : ℝ) : ℝ × ℝ :=
  (Real.cos θ, Real.sin θ)

theorem cos_30_degrees : 
  let Q := unit_circle (30 * Real.pi / 180) in -- 30 degrees in radians
  (Q.1 = (Real.sqrt 3) / 2) :=
by
  sorry

end cos_30_degrees_l16_16766


namespace second_gym_signup_fee_covers_4_months_l16_16105

-- Define constants
def cheap_gym_monthly_fee : ℕ := 10
def cheap_gym_signup_fee : ℕ := 50
def total_spent_first_year : ℕ := 650

-- Define the monthly fee of the second gym
def second_gym_monthly_fee : ℕ := 3 * cheap_gym_monthly_fee

-- Calculate the amount spent on the second gym
def spent_on_second_gym : ℕ := total_spent_first_year - (12 * cheap_gym_monthly_fee + cheap_gym_signup_fee)

-- Define the number of months the sign-up fee covers
def months_covered_by_signup_fee : ℕ := spent_on_second_gym / second_gym_monthly_fee

theorem second_gym_signup_fee_covers_4_months :
  months_covered_by_signup_fee = 4 :=
by
  sorry

end second_gym_signup_fee_covers_4_months_l16_16105


namespace repeating_decimal_to_fraction_l16_16774

theorem repeating_decimal_to_fraction (h : (0.0909090909 : ℝ) = 1 / 11) : (0.2727272727 : ℝ) = 3 / 11 :=
sorry

end repeating_decimal_to_fraction_l16_16774


namespace triangle_is_isosceles_l16_16502

variable (A B C a b c : ℝ)
variable (sin : ℝ → ℝ)

theorem triangle_is_isosceles (h1 : a * sin A - b * sin B = 0) :
  a = b :=
by
  sorry

end triangle_is_isosceles_l16_16502


namespace a_is_perfect_square_l16_16683

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l16_16683


namespace probability_at_least_eight_stayed_correct_l16_16686

noncomputable def probability_at_least_eight_stayed (n : ℕ) (c : ℕ) (p : ℚ) : ℚ :=
  let certain_count := c
  let unsure_count := n - c
  let k := 3
  let prob_eight := 
    (Nat.choose unsure_count k : ℚ) * (p^k) * ((1 - p)^(unsure_count - k))
  let prob_nine := p^unsure_count
  prob_eight + prob_nine

theorem probability_at_least_eight_stayed_correct :
  probability_at_least_eight_stayed 9 5 (3/7) = 513 / 2401 :=
by
  sorry

end probability_at_least_eight_stayed_correct_l16_16686


namespace tree_growth_factor_l16_16809

theorem tree_growth_factor 
  (initial_total : ℕ) 
  (initial_maples : ℕ) 
  (initial_lindens : ℕ) 
  (spring_total : ℕ) 
  (autumn_total : ℕ)
  (initial_maple_percentage : initial_maples = 3 * initial_total / 5)
  (spring_maple_percentage : initial_maples = spring_total / 5)
  (autumn_maple_percentage : initial_maples * 2 = autumn_total * 3 / 5) :
  autumn_total = 6 * initial_total :=
sorry

end tree_growth_factor_l16_16809


namespace kelly_carrot_weight_l16_16101

-- Define the number of carrots harvested from each bed
def carrots_bed1 : ℕ := 55
def carrots_bed2 : ℕ := 101
def carrots_bed3 : ℕ := 78
def carrots_per_pound : ℕ := 6

-- Define the total number of carrots
def total_carrots := carrots_bed1 + carrots_bed2 + carrots_bed3

-- Define the total weight in pounds
def total_weight := total_carrots / carrots_per_pound

-- The theorem to prove the total weight is 39 pounds
theorem kelly_carrot_weight : total_weight = 39 := by
  sorry

end kelly_carrot_weight_l16_16101


namespace sum_quotient_remainder_div9_l16_16867

theorem sum_quotient_remainder_div9 (n : ℕ) (h₁ : n = 248 * 5 + 4) :
  let q := n / 9
  let r := n % 9
  q + r = 140 :=
by
  sorry

end sum_quotient_remainder_div9_l16_16867


namespace problem_statement_l16_16237

noncomputable def a_n (n : ℕ) : ℝ :=
  if n ≥ 2 then (Nat.choose n 2) * 3^(n-2) else 0

theorem problem_statement :
  (2016 / 2015) * (∑ n in (Finset.range 2016).filter (λ x, x ≥ 2), 3^n / a_n n) = 18 :=
by 
  sorry

end problem_statement_l16_16237


namespace simplify_expression_l16_16536

theorem simplify_expression :
  (Complex.mk (-1) (Real.sqrt 3) / 2) ^ 12 + (Complex.mk (-1) (-Real.sqrt 3) / 2) ^ 12 = 2 := by
  sorry

end simplify_expression_l16_16536


namespace prop_for_real_l16_16373

theorem prop_for_real (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end prop_for_real_l16_16373


namespace rows_in_initial_patios_l16_16887

theorem rows_in_initial_patios (r c : ℕ) (h1 : r * c = 60) (h2 : (2 * c : ℚ) / r = 3 / 2) (h3 : (r + 5) * (c - 3) = 60) : r = 10 :=
sorry

end rows_in_initial_patios_l16_16887


namespace baseball_league_games_l16_16570

theorem baseball_league_games
  (N M : ℕ)
  (hN_gt_2M : N > 2 * M)
  (hM_gt_4 : M > 4)
  (h_total_games : 4 * N + 5 * M = 94) :
  4 * N = 64 :=
by
  sorry

end baseball_league_games_l16_16570


namespace age_difference_28_l16_16433

variable (li_lin_age_father_sum li_lin_age_future father_age_future : ℕ)

theorem age_difference_28 
    (h1 : li_lin_age_father_sum = 50)
    (h2 : ∀ x, li_lin_age_future = x → father_age_future = 3 * x - 2)
    (h3 : li_lin_age_future + 4 = li_lin_age_father_sum + 8 - (father_age_future + 4))
    : li_lin_age_father_sum - li_lin_age_future = 28 :=
sorry

end age_difference_28_l16_16433


namespace seating_arrangement_l16_16119

/--
Prove that the number of ways for Xiao Zhang's son, daughter, and any one other person
to sit together in a row of 6 people (Xiao Zhang, his son, daughter, father, mother,
and younger brother) is 216.
-/
theorem seating_arrangement :
  let arrangements := 6 * 3 * 12 in
  arrangements = 216 :=
by
  simp [arrangements]
  sorry

end seating_arrangement_l16_16119


namespace inverse_proposition_l16_16274

-- Definition of the proposition
def complementary_angles_on_same_side (l m : Line) : Prop := sorry
def parallel_lines (l m : Line) : Prop := sorry

-- The original proposition
def original_proposition (l m : Line) : Prop := complementary_angles_on_same_side l m → parallel_lines l m

-- The statement of the proof problem
theorem inverse_proposition (l m : Line) :
  (complementary_angles_on_same_side l m → parallel_lines l m) →
  (parallel_lines l m → complementary_angles_on_same_side l m) := sorry

end inverse_proposition_l16_16274


namespace car_average_speed_l16_16877

theorem car_average_speed (distance time : ℕ) (h1 : distance = 715) (h2 : time = 11) : distance / time = 65 := by
  sorry

end car_average_speed_l16_16877


namespace ratio_of_slices_l16_16347

theorem ratio_of_slices
  (initial_slices : ℕ)
  (slices_eaten_for_lunch : ℕ)
  (remaining_slices_after_lunch : ℕ)
  (slices_left_for_tomorrow : ℕ)
  (slices_eaten_for_dinner : ℕ)
  (ratio : ℚ) :
  initial_slices = 12 → 
  slices_eaten_for_lunch = initial_slices / 2 →
  remaining_slices_after_lunch = initial_slices - slices_eaten_for_lunch →
  slices_left_for_tomorrow = 4 →
  slices_eaten_for_dinner = remaining_slices_after_lunch - slices_left_for_tomorrow →
  ratio = (slices_eaten_for_dinner : ℚ) / remaining_slices_after_lunch →
  ratio = 1 / 3 :=
by sorry

end ratio_of_slices_l16_16347


namespace correct_pairings_l16_16961

-- Define the employees
inductive Employee
| Jia
| Yi
| Bing
deriving DecidableEq

-- Define the wives
inductive Wife
| A
| B
| C
deriving DecidableEq

-- Define the friendship and age relationships
def isGoodFriend (x y : Employee) : Prop :=
  -- A's husband is Yi's good friend.
  (x = Employee.Jia ∧ y = Employee.Yi) ∨
  (x = Employee.Yi ∧ y = Employee.Jia)

def isYoungest (x : Employee) : Prop :=
  -- Specify that Jia is the youngest
  x = Employee.Jia

def isOlder (x y : Employee) : Prop :=
  -- Bing is older than C's husband.
  x = Employee.Bing ∧ y ≠ Employee.Bing

-- The pairings of husbands and wives: Jia—A, Yi—C, Bing—B.
def pairings (x : Employee) : Wife :=
  match x with
  | Employee.Jia => Wife.A
  | Employee.Yi => Wife.C
  | Employee.Bing => Wife.B

-- Proving the given pairings fit the conditions.
theorem correct_pairings : 
  ∀ (x : Employee), 
  isGoodFriend (Employee.Jia) (Employee.Yi) ∧ 
  isYoungest Employee.Jia ∧ 
  (isOlder Employee.Bing Employee.Jia ∨ isOlder Employee.Bing Employee.Yi) → 
  pairings x = match x with
               | Employee.Jia => Wife.A
               | Employee.Yi => Wife.C
               | Employee.Bing => Wife.B :=
by
  sorry

end correct_pairings_l16_16961


namespace fraction_simplification_l16_16413

theorem fraction_simplification :
  (20 / 21) * (35 / 54) * (63 / 50) = (7 / 9) :=
by
  sorry

end fraction_simplification_l16_16413


namespace problem_l16_16223

def operation (a b : ℤ) (h : a ≠ 0) : ℤ := (b - a) ^ 2 / a ^ 2

theorem problem : 
  operation (-1) (operation 1 (-1) (by decide)) (by decide) = 25 := 
by
  sorry

end problem_l16_16223


namespace discounted_price_correct_l16_16550

noncomputable def discounted_price (original_price : ℝ) (discount : ℝ) : ℝ :=
  original_price - (discount / 100 * original_price)

theorem discounted_price_correct :
  discounted_price 800 30 = 560 :=
by
  -- Correctness of the discounted price calculation
  sorry

end discounted_price_correct_l16_16550


namespace jake_split_shots_l16_16813

theorem jake_split_shots (shot_volume : ℝ) (purity : ℝ) (alcohol_consumed : ℝ) 
    (h1 : shot_volume = 1.5) (h2 : purity = 0.50) (h3 : alcohol_consumed = 3) : 
    2 * (alcohol_consumed / (purity * shot_volume)) = 8 :=
by
  sorry

end jake_split_shots_l16_16813


namespace total_expenditure_l16_16325

-- Define the conditions.
def singers : ℕ := 30
def current_robes : ℕ := 12
def robe_cost : ℕ := 2

-- Define the statement.
theorem total_expenditure (singers current_robes robe_cost : ℕ) : 
  (singers - current_robes) * robe_cost = 36 := by
  sorry

end total_expenditure_l16_16325


namespace max_buses_l16_16645

-- Define the total number of stops and buses as finite sets
def stops : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a bus as a subset of stops with exactly 3 stops
structure Bus :=
(stops : Finset ℕ)
(h_stops : stops.card = 3)

-- Define the condition that any two buses share at most one stop
def shares_at_most_one (b1 b2 : Bus) : Prop :=
(b1.stops ∩ b2.stops).card ≤ 1

-- Define the predicate for a valid set of buses
def valid_buses (buses : Finset Bus) : Prop :=
∀ b1 b2 ∈ buses, b1 ≠ b2 → shares_at_most_one b1 b2

-- The main theorem statement
theorem max_buses (buses : Finset Bus) (h_valid : valid_buses buses) : buses.card ≤ 12 :=
sorry

end max_buses_l16_16645


namespace sum_of_a_b_vert_asymptotes_l16_16944

theorem sum_of_a_b_vert_asymptotes (a b : ℝ) 
  (h1 : ∀ x : ℝ, x = -1 → x^2 + a * x + b = 0) 
  (h2 : ∀ x : ℝ, x = 3 → x^2 + a * x + b = 0) : 
  a + b = -5 :=
sorry

end sum_of_a_b_vert_asymptotes_l16_16944


namespace second_percentage_increase_l16_16422

theorem second_percentage_increase (P : ℝ) (x : ℝ) :
  1.25 * P * (1 + x / 100) = 1.625 * P ↔ x = 30 :=
by
  sorry

end second_percentage_increase_l16_16422


namespace positive_difference_abs_eq_15_l16_16003

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l16_16003


namespace find_k_from_roots_ratio_l16_16142

theorem find_k_from_roots_ratio (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1 + x2 = -10)
  (h2 : x1 * x2 = k)
  (h3 : x1/x2 = 3) : 
  k = 18.75 := 
sorry

end find_k_from_roots_ratio_l16_16142


namespace discount_percentage_l16_16447

theorem discount_percentage (discount amount_paid : ℝ) (h_discount : discount = 40) (h_paid : amount_paid = 120) : 
  (discount / (discount + amount_paid)) * 100 = 25 := by
  sorry

end discount_percentage_l16_16447


namespace investment_of_q_is_correct_l16_16564

-- Define investments and the profit ratio
def p_investment : ℝ := 30000
def profit_ratio_p : ℝ := 2
def profit_ratio_q : ℝ := 3

-- Define q's investment as x
def q_investment : ℝ := 45000

-- The goal is to prove that q_investment is indeed 45000 given the above conditions
theorem investment_of_q_is_correct :
  (p_investment / q_investment) = (profit_ratio_p / profit_ratio_q) :=
sorry

end investment_of_q_is_correct_l16_16564


namespace mod_pow_sum_7_l16_16483

theorem mod_pow_sum_7 :
  (45 ^ 1234 + 27 ^ 1234) % 7 = 5 := by
  sorry

end mod_pow_sum_7_l16_16483


namespace find_k_from_direction_vector_l16_16837

/-- Given points p1 and p2, the direction vector's k component
    is -3 when the x component is 3. -/
theorem find_k_from_direction_vector
  (p1 : ℤ × ℤ) (p2 : ℤ × ℤ)
  (h1 : p1 = (2, -1))
  (h2 : p2 = (-4, 5))
  (dv_x : ℤ) (dv_k : ℤ)
  (h3 : (dv_x, dv_k) = (3, -3)) :
  True :=
by
  sorry

end find_k_from_direction_vector_l16_16837


namespace divisibility_by_37_l16_16903

theorem divisibility_by_37 (a b c : ℕ) :
  (100 * a + 10 * b + c) % 37 = 0 → 
  (100 * b + 10 * c + a) % 37 = 0 ∧
  (100 * c + 10 * a + b) % 37 = 0 :=
by
  sorry

end divisibility_by_37_l16_16903


namespace exists_triangles_arrangement_l16_16034

noncomputable def triangles_arrangement (T : Fin 10 → Triangle ℝ) : Prop :=
  (∀ i j, i ≠ j → (∃ p, p ∈ (T i).points ∧ p ∈ (T j).points)) ∧
  (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬(∃ p, p ∈ (T i).points ∧ p ∈ (T j).points ∧ p ∈ (T k).points))

theorem exists_triangles_arrangement :
  ∃ T : Fin 10 → Triangle ℝ, triangles_arrangement T :=
sorry

end exists_triangles_arrangement_l16_16034


namespace find_Q_x_l16_16665

noncomputable def Q : ℝ → ℝ := sorry

variables (Q0 Q1 Q2 : ℝ)

axiom Q_def : ∀ x, Q x = Q0 + Q1 * x + Q2 * x^2
axiom Q_minus_2 : Q (-2) = -3

theorem find_Q_x : ∀ x, Q x = (3 / 5) * (1 + x - x^2) :=
by 
  -- Proof to be completed
  sorry

end find_Q_x_l16_16665


namespace problem_statement_l16_16204

noncomputable
def ellipse_equation (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1

def midpoint (x1 y1 x2 y2 mx my : ℝ) : Prop :=
  mx = (x1 + x2)/2 ∧ my = (y1 + y2)/2

theorem problem_statement :
  ∃ (a b : ℝ), a^2 = 24 ∧ b^2 = 8 ∧ (a > b ∧ b > 0) ∧
  (ellipse_equation a b) ∧
  (∃ (x1 y1 x2 y2 : ℝ), midpoint x1 y1 x2 y2 3 1) ∧
  (∃ (k : ℝ), ∀ (x : ℝ), y = k * (x - 3) + 1) ∧
  (∃ (d : ℝ), d = 3 * sqrt 2 / sqrt 3 ∧ |AB| = 2*sqrt 6)
:= sorry

end problem_statement_l16_16204


namespace sum_of_prime_factors_1320_l16_16164

theorem sum_of_prime_factors_1320 : 
  let smallest_prime := 2
  let largest_prime := 11
  smallest_prime + largest_prime = 13 :=
by
  sorry

end sum_of_prime_factors_1320_l16_16164


namespace proportion_correct_l16_16629

theorem proportion_correct {a b : ℝ} (h : 2 * a = 5 * b) : a / 5 = b / 2 :=
by {
  sorry
}

end proportion_correct_l16_16629


namespace solve_for_x_y_l16_16785

theorem solve_for_x_y (x y : ℝ) (h1 : x^2 + x * y + y = 14) (h2 : y^2 + x * y + x = 28) : 
  x + y = -7 ∨ x + y = 6 :=
by 
  -- We'll write sorry here to indicate the proof is to be completed
  sorry

end solve_for_x_y_l16_16785


namespace option_d_correct_l16_16165

theorem option_d_correct (a b c : ℝ) (h : a > b ∧ b > c ∧ c > 0) : a / b < a / c :=
by
  sorry

end option_d_correct_l16_16165


namespace find_starting_number_l16_16719

theorem find_starting_number (k m : ℕ) (hk : 67 = (m - k) / 3 + 1) (hm : m = 300) : k = 102 := by
  sorry

end find_starting_number_l16_16719


namespace chicken_nuggets_cost_l16_16247

theorem chicken_nuggets_cost :
  ∀ (nuggets_ordered boxes_cost : ℕ) (nuggets_per_box : ℕ),
  nuggets_ordered = 100 →
  nuggets_per_box = 20 →
  boxes_cost = 4 →
  (nuggets_ordered / nuggets_per_box) * boxes_cost = 20 :=
by
  intros nuggets_ordered boxes_cost nuggets_per_box h1 h2 h3
  sorry

end chicken_nuggets_cost_l16_16247


namespace percentage_of_boys_l16_16091

theorem percentage_of_boys (total_students boys_per_group girls_per_group : ℕ)
  (ratio_condition : boys_per_group + girls_per_group = 7)
  (total_condition : total_students = 42)
  (ratio_b_condition : boys_per_group = 3)
  (ratio_g_condition : girls_per_group = 4) :
  (boys_per_group : ℚ) / (boys_per_group + girls_per_group : ℚ) * 100 = 42.86 :=
by sorry

end percentage_of_boys_l16_16091


namespace expectation_is_four_thirds_l16_16327

-- Define the probability function
def P_ξ (k : ℕ) : ℚ :=
  if k = 0 then (1/2)^2 * (2/3)
  else if k = 1 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3)
  else if k = 2 then (1/2) * (1/2) * (2/3) + (1/2) * (1/2) * (1/3) + (1/2) * (1/2) * (1/3)
  else if k = 3 then (1/2) * (1/2) * (1/3)
  else 0

-- Define the expected value function
def E_ξ : ℚ :=
  0 * P_ξ 0 + 1 * P_ξ 1 + 2 * P_ξ 2 + 3 * P_ξ 3

-- Formal statement of the problem
theorem expectation_is_four_thirds : E_ξ = 4 / 3 :=
  sorry

end expectation_is_four_thirds_l16_16327


namespace range_of_a_for_local_min_max_l16_16068

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a_for_local_min_max (a e x1 x2 : ℝ) (h_a : 0 < a) (h_a_ne : a ≠ 1) (h_x1_x2 : x1 < x2) 
  (h_min : ∀ x, f a e x > f a e x1) (h_max : ∀ x, f a e x < f a e x2) : 
  (1 / Real.exp 1) < a ∧ a < 1 := 
sorry

end range_of_a_for_local_min_max_l16_16068


namespace carrots_weight_l16_16097

theorem carrots_weight (carrots_bed1: ℕ) (carrots_bed2: ℕ) (carrots_bed3: ℕ) (carrots_per_pound: ℕ)
  (h_bed1: carrots_bed1 = 55)
  (h_bed2: carrots_bed2 = 101)
  (h_bed3: carrots_bed3 = 78)
  (h_c_per_p: carrots_per_pound = 6) :
  (carrots_bed1 + carrots_bed2 + carrots_bed3) / carrots_per_pound = 39 := by
  sorry

end carrots_weight_l16_16097


namespace probability_not_same_level_is_four_fifths_l16_16331

-- Definitions of the conditions
def nobility_levels := 5
def total_outcomes := nobility_levels * nobility_levels
def same_level_outcomes := nobility_levels

-- Definition of the probability
def probability_not_same_level := 1 - (same_level_outcomes / total_outcomes : ℚ)

-- The theorem statement
theorem probability_not_same_level_is_four_fifths :
  probability_not_same_level = 4 / 5 := 
  by sorry

end probability_not_same_level_is_four_fifths_l16_16331


namespace find_other_endpoint_l16_16421

theorem find_other_endpoint (x_m y_m x_1 y_1 x_2 y_2 : ℝ) 
  (h_mid_x : x_m = (x_1 + x_2) / 2)
  (h_mid_y : y_m = (y_1 + y_2) / 2)
  (h_x_m : x_m = 3)
  (h_y_m : y_m = 4)
  (h_x_1 : x_1 = 0)
  (h_y_1 : y_1 = -1) :
  (x_2, y_2) = (6, 9) :=
sorry

end find_other_endpoint_l16_16421


namespace cos_30_degrees_eq_sqrt_3_div_2_l16_16764

noncomputable def cos_30_degrees : ℝ :=
  real.cos (real.pi / 6)

theorem cos_30_degrees_eq_sqrt_3_div_2 :
  cos_30_degrees = sqrt 3 / 2 :=
sorry

end cos_30_degrees_eq_sqrt_3_div_2_l16_16764


namespace gcd_of_72_120_168_l16_16135

theorem gcd_of_72_120_168 : Nat.gcd (Nat.gcd 72 120) 168 = 24 := 
by
  sorry

end gcd_of_72_120_168_l16_16135


namespace total_time_assignment_l16_16964

-- Define the time taken for each part
def time_first_part : ℕ := 25
def time_second_part : ℕ := 2 * time_first_part
def time_third_part : ℕ := 45

-- Define the total time taken for the assignment
def total_time : ℕ := time_first_part + time_second_part + time_third_part

-- The theorem stating that the total time is 120 minutes
theorem total_time_assignment : total_time = 120 := by
  sorry

end total_time_assignment_l16_16964


namespace question1_question2_l16_16788

variable (a : ℤ)
def point_P : (ℤ × ℤ) := (2*a - 2, a + 5)

-- Part 1: If point P lies on the x-axis, its coordinates are (-12, 0).
theorem question1 (h1 : a + 5 = 0) : point_P a = (-12, 0) :=
sorry

-- Part 2: If point P lies in the second quadrant and the distance from point P to the x-axis is equal to the distance from point P to the y-axis,
-- the value of a^2023 + 2023 is 2022.
theorem question2 (h2 : 2*a - 2 < 0) (h3 : -(2*a - 2) = a + 5) : a ^ 2023 + 2023 = 2022 :=
sorry

end question1_question2_l16_16788


namespace problem_angle_magnitude_and_sin_l16_16225

theorem problem_angle_magnitude_and_sin (
  a b c : ℝ) (A B C : ℝ) 
  (h1 : a = Real.sqrt 7) (h2 : b = 3) 
  (h3 : Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3)
  (triangle_is_acute : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) : 
  A = Real.pi / 3 ∧ Real.sin (2 * B + Real.pi / 6) = -1 / 7 :=
by
  sorry

end problem_angle_magnitude_and_sin_l16_16225


namespace product_fractions_l16_16478

open BigOperators

theorem product_fractions : ∏ n in Finset.range 28 \ Finset.singleton 0 ∪ Finset.singleton 1, (n + 2) / (n + 1) = 15 := by
  sorry

end product_fractions_l16_16478


namespace woman_wait_time_for_man_to_catch_up_l16_16025

theorem woman_wait_time_for_man_to_catch_up :
  ∀ (mans_speed womans_speed : ℕ) (time_after_passing : ℕ) (distance_up_slope : ℕ) (incline_percentage : ℕ),
  mans_speed = 5 →
  womans_speed = 25 →
  time_after_passing = 5 →
  distance_up_slope = 1 →
  incline_percentage = 5 →
  max 0 (mans_speed - incline_percentage * 1) = 0 →
  time_after_passing = 0 :=
by
  intros
  -- Insert proof here when needed
  sorry

end woman_wait_time_for_man_to_catch_up_l16_16025


namespace solve_for_a_l16_16781

theorem solve_for_a (a : ℝ) (h : 50 - |a - 2| = |4 - a|) :
  a = -22 ∨ a = 28 :=
sorry

end solve_for_a_l16_16781


namespace coffee_ratio_is_one_to_five_l16_16341

-- Given conditions
def thermos_capacity : ℕ := 20 -- capacity in ounces
def times_filled_per_day : ℕ := 2
def school_days_per_week : ℕ := 5
def new_weekly_coffee_consumption : ℕ := 40 -- in ounces

-- Definitions based on the conditions
def old_daily_coffee_consumption := thermos_capacity * times_filled_per_day
def old_weekly_coffee_consumption := old_daily_coffee_consumption * school_days_per_week

-- Theorem: The ratio of the new weekly coffee consumption to the old weekly coffee consumption is 1:5
theorem coffee_ratio_is_one_to_five : 
  new_weekly_coffee_consumption / old_weekly_coffee_consumption = 1 / 5 := 
by
  -- Proof is omitted
  sorry

end coffee_ratio_is_one_to_five_l16_16341


namespace a_is_perfect_square_l16_16673

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l16_16673


namespace a_eq_3x_or_neg2x_l16_16943

theorem a_eq_3x_or_neg2x (a b x : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 19 * x^3) (h3 : a - b = x) :
    a = 3 * x ∨ a = -2 * x :=
by
  -- The proof will go here
  sorry

end a_eq_3x_or_neg2x_l16_16943


namespace min_value_of_expression_l16_16500

theorem min_value_of_expression {x y z : ℝ} 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : 2 * x * (x + 1 / y + 1 / z) = y * z) : 
  (x + 1 / y) * (x + 1 / z) >= Real.sqrt 2 :=
by
  sorry

end min_value_of_expression_l16_16500


namespace sum_of_interior_angles_of_pentagon_l16_16286

theorem sum_of_interior_angles_of_pentagon :
    (5 - 2) * 180 = 540 := by 
  -- The proof goes here
  sorry

end sum_of_interior_angles_of_pentagon_l16_16286


namespace seeds_in_pots_l16_16349

theorem seeds_in_pots (x : ℕ) (total_seeds : ℕ) (seeds_fourth_pot : ℕ) 
  (h1 : total_seeds = 10) (h2 : seeds_fourth_pot = 1) 
  (h3 : 3 * x + seeds_fourth_pot = total_seeds) : x = 3 :=
by
  sorry

end seeds_in_pots_l16_16349


namespace range_of_b_l16_16790

-- Given function y = x^2 - 2bx + b^2 + b - 5
def quadratic_function (x b : ℝ) : ℝ := x^2 - 2 * b * x + b^2 + b - 5

-- Conditions:
-- Condition 1: The function intersects the x-axis (discriminant >= 0)
def discriminant_condition (b : ℝ) : Prop :=
  let Δ := (-2 * b)^2 - 4 * (b^2 + b - 5) in
  Δ >= 0

-- Condition 2: The function decreases for x < 3.5
def decreasing_condition (b : ℝ) : Prop :=
  b >= 3.5

-- Prove that the range of b is 3.5 ≤ b ≤ 5
theorem range_of_b (b : ℝ) : discriminant_condition b → decreasing_condition b → 3.5 ≤ b ∧ b ≤ 5 :=
by
  sorry

end range_of_b_l16_16790


namespace sum_m_n_zero_l16_16923

theorem sum_m_n_zero
  (m n p : ℝ)
  (h1 : mn + p^2 + 4 = 0)
  (h2 : m - n = 4) :
  m + n = 0 :=
sorry

end sum_m_n_zero_l16_16923


namespace sum_a4_a5_a6_l16_16129

section ArithmeticSequence

variable {a : ℕ → ℝ}

-- Condition 1: The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

-- Condition 2: Given information
axiom a2_a8_eq_6 : a 2 + a 8 = 6

-- Question: Prove that a 4 + a 5 + a 6 = 9
theorem sum_a4_a5_a6 : is_arithmetic_sequence a → a 4 + a 5 + a 6 = 9 :=
by
  intro h_arith
  sorry

end ArithmeticSequence

end sum_a4_a5_a6_l16_16129


namespace problem1_problem2_problem3_problem4_problem5_problem6_l16_16587

-- First problem: \(\frac{1}{3} + \left(-\frac{1}{2}\right) = -\frac{1}{6}\)
theorem problem1 : (1 / 3 : ℚ) + (-1 / 2) = -1 / 6 := by sorry

-- Second problem: \(-2 - \left(-9\right) = 7\)
theorem problem2 : (-2 : ℚ) - (-9) = 7 := by sorry

-- Third problem: \(\frac{15}{16} - \left(-7\frac{1}{16}\right) = 8\)
theorem problem3 : (15 / 16 : ℚ) - (-(7 + 1 / 16)) = 8 := by sorry

-- Fourth problem: \(-\left|-4\frac{2}{7}\right| - \left|+1\frac{5}{7}\right| = -6\)
theorem problem4 : -|(-4 - 2 / 7 : ℚ)| - |(1 + 5 / 7)| = -6 := by sorry

-- Fifth problem: \(6 + \left(-12\right) + 8.3 + \left(-7.5\right) = -5.2\)
theorem problem5 : (6 : ℚ) + (-12) + (83 / 10) + (-75 / 10) = -52 / 10 := by sorry

-- Sixth problem: \(\left(-\frac{1}{8}\right) + 3.25 + 2\frac{3}{5} + \left(-5.875\right) + 1.15 = 1\)
theorem problem6 : (-1 / 8 : ℚ) + 3 + 1 / 4 + 2 + 3 / 5 + (-5 - 875 / 1000) + 1 + 15 / 100 = 1 := by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l16_16587


namespace cos_30_eq_sqrt3_div_2_l16_16763

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l16_16763


namespace ratio_of_dad_to_jayson_l16_16011

-- Define the conditions
def JaysonAge : ℕ := 10
def MomAgeWhenBorn : ℕ := 28
def MomCurrentAge (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ) : ℕ := MomAgeWhenBorn + JaysonAge
def DadCurrentAge (MomCurrentAge : ℕ) : ℕ := MomCurrentAge + 2

-- Define the proof problem
theorem ratio_of_dad_to_jayson (JaysonAge : ℕ) (MomAgeWhenBorn : ℕ)
  (h1 : JaysonAge = 10) (h2 : MomAgeWhenBorn = 28) :
  DadCurrentAge (MomCurrentAge JaysonAge MomAgeWhenBorn) / JaysonAge = 4 :=
by 
  sorry

end ratio_of_dad_to_jayson_l16_16011


namespace alma_score_l16_16842

variables (A M S : ℕ)

-- Given conditions
axiom h1 : M = 60
axiom h2 : M = 3 * A
axiom h3 : A + M = 2 * S

theorem alma_score : S = 40 :=
by
  -- proof goes here
  sorry

end alma_score_l16_16842


namespace female_students_in_sample_l16_16184

-- Definitions of the given conditions
def male_students : ℕ := 28
def female_students : ℕ := 21
def total_students : ℕ := male_students + female_students
def sample_size : ℕ := 14
def stratified_sampling_fraction : ℚ := (sample_size : ℚ) / (total_students : ℚ)
def female_sample_count : ℚ := stratified_sampling_fraction * (female_students : ℚ)

-- The theorem to prove
theorem female_students_in_sample : female_sample_count = 6 :=
by
  sorry

end female_students_in_sample_l16_16184


namespace circumscribed_triangle_area_relation_l16_16881

theorem circumscribed_triangle_area_relation
  (a b c D E F : ℝ)
  (h₁ : a = 18) (h₂ : b = 24) (h₃ : c = 30)
  (triangle_right : a^2 + b^2 = c^2)
  (triangle_area : (1/2) * a * b = 216)
  (circle_area : π * (c / 2)^2 = 225 * π)
  (non_triangle_areas : D + E + 216 = F) :
  D + E + 216 = F :=
by
  sorry

end circumscribed_triangle_area_relation_l16_16881


namespace denominator_of_speed_l16_16572

theorem denominator_of_speed (h : 0.8 = 8 / d * 3600 / 1000) : d = 36 := 
by
  sorry

end denominator_of_speed_l16_16572


namespace probability_at_least_one_eight_l16_16160

theorem probability_at_least_one_eight :
  let total_outcomes := 64 in
  let outcomes_without_8 := 49 in
  let favorable_outcomes := total_outcomes - outcomes_without_8 in
  let probability := (favorable_outcomes : ℚ) / total_outcomes in
  probability = 15 / 64 :=
by
  let total_outcomes := 64
  let outcomes_without_8 := 49
  let favorable_outcomes := total_outcomes - outcomes_without_8
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  show probability = 15 / 64
  sorry

end probability_at_least_one_eight_l16_16160


namespace train_A_length_l16_16997

theorem train_A_length
  (speed_A : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (len_A : ℕ)
  (h1 : speed_A = 54) 
  (h2 : speed_B = 36) 
  (h3 : time_to_cross = 15)
  (h4 : len_A = (speed_A + speed_B) * 1000 / 3600 * time_to_cross) :
  len_A = 375 :=
sorry

end train_A_length_l16_16997


namespace pool_width_40_l16_16541

theorem pool_width_40
  (hose_rate : ℕ)
  (pool_length : ℕ)
  (pool_depth : ℕ)
  (pool_capacity_percent : ℚ)
  (drain_time : ℕ)
  (water_drained : ℕ)
  (total_capacity : ℚ)
  (pool_width : ℚ) :
  hose_rate = 60 ∧
  pool_length = 150 ∧
  pool_depth = 10 ∧
  pool_capacity_percent = 0.8 ∧
  drain_time = 800 ∧
  water_drained = hose_rate * drain_time ∧
  total_capacity = water_drained / pool_capacity_percent ∧
  total_capacity = pool_length * pool_width * pool_depth →
  pool_width = 40 :=
by
  sorry

end pool_width_40_l16_16541


namespace range_of_a_l16_16069

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) (e : ℝ) (f : ℝ → ℝ)
  (h1 : 0 < a ∧ a ≠ 1)
  (h2 : f x = 2 * a ^ x - e * x ^ 2)
  (h_min : ∃ x₁, is_local_min f x₁)
  (h_max : ∃ x₂, is_local_max f x₂)
  (h_inequality : x₁ < x₂) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
sorry

end range_of_a_l16_16069


namespace solution_set_inequality_l16_16551

theorem solution_set_inequality (x : ℝ) : (x ≠ 1) → 
  ((x - 3) * (x + 2) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3) :=
by
  intros h
  sorry

end solution_set_inequality_l16_16551


namespace probability_of_two_points_is_three_sevenths_l16_16696

/-- Define the problem's conditions and statement. -/
def num_choices (n : ℕ) : ℕ :=
  match n with
  | 1 => 4  -- choose 1 option from 4
  | 2 => 6  -- choose 2 options from 4 (binomial coefficient)
  | 3 => 4  -- choose 3 options from 4 (binomial coefficient)
  | _ => 0

def total_ways : ℕ := 14  -- Total combinations of choosing 1 to 3 options from 4

def two_points_ways : ℕ := 6  -- 3 ways for 1 correct, 3 ways for 2 correct (B, C, D combinations)

def probability_two_points : ℚ :=
  (two_points_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_two_points_is_three_sevenths :
  probability_two_points = (3 / 7 : ℚ) :=
sorry

end probability_of_two_points_is_three_sevenths_l16_16696


namespace area_of_region_l16_16228

theorem area_of_region : 
  (∃ (A : ℝ), A = 12 ∧ ∀ (x y : ℝ), |x| + |y| + |x - 2| ≤ 4 → 
    (0 ≤ y ∧ y ≤ 6 - 2*x ∧ x ≥ 2) ∨
    (0 ≤ y ∧ y ≤ 2 ∧ 0 ≤ x ∧ x < 2) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ -1 ≤ x ∧ x < 0) ∨
    (0 ≤ y ∧ y ≤ 2*x + 2 ∧ x < -1)) :=
sorry

end area_of_region_l16_16228


namespace shopkeeper_profit_percentage_l16_16030

theorem shopkeeper_profit_percentage
  (cost_price : ℝ)
  (goods_lost_pct : ℝ)
  (loss_pct : ℝ)
  (remaining_goods : ℝ)
  (selling_price : ℝ)
  (profit_pct : ℝ)
  (h1 : cost_price = 100)
  (h2 : goods_lost_pct = 0.20)
  (h3 : loss_pct = 0.12)
  (h4 : remaining_goods = cost_price * (1 - goods_lost_pct))
  (h5 : selling_price = cost_price * (1 - loss_pct))
  (h6 : profit_pct = ((selling_price - remaining_goods) / remaining_goods) * 100) : 
  profit_pct = 10 := 
sorry

end shopkeeper_profit_percentage_l16_16030


namespace stock_rise_in_morning_l16_16050

theorem stock_rise_in_morning (x : ℕ) (V : ℕ → ℕ) (h0 : V 0 = 100)
  (h100 : V 100 = 200) (h_recurrence : ∀ n, V n = 100 + n * x - n) :
  x = 2 :=
  by
  sorry

end stock_rise_in_morning_l16_16050


namespace polygon_sides_l16_16577

theorem polygon_sides (h : ∀ (n : ℕ), (180 * (n - 2)) / n = 150) : n = 12 :=
by
  sorry

end polygon_sides_l16_16577


namespace inverse_proportional_x_y_l16_16269

theorem inverse_proportional_x_y (x y k : ℝ) (h_inverse : x * y = k) (h_given : 40 * 5 = k) : x = 20 :=
by 
  sorry

end inverse_proportional_x_y_l16_16269


namespace scientific_notation_460_billion_l16_16747

theorem scientific_notation_460_billion : 460000000000 = 4.6 * 10^11 := 
sorry

end scientific_notation_460_billion_l16_16747


namespace area_of_region_l16_16998

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 11 = 0) -> 
  ∃ (A : ℝ), A = 24 * Real.pi :=
by 
  sorry

end area_of_region_l16_16998


namespace five_letter_word_combinations_l16_16659

open Nat

theorem five_letter_word_combinations :
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  total_combinations = 456976 := 
by
  let first_letter_choices := 26
  let other_letter_choices := 26
  let total_combinations := first_letter_choices ^ 1 * other_letter_choices ^ 3
  show total_combinations = 456976
  sorry

end five_letter_word_combinations_l16_16659


namespace potato_bag_weight_l16_16314

theorem potato_bag_weight :
  ∃ w : ℝ, w = 16 / (w / 4) ∧ w = 16 := 
by
  sorry

end potato_bag_weight_l16_16314


namespace positive_difference_abs_eq_l16_16008

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l16_16008


namespace equation_of_curve_C_range_of_m_l16_16786

theorem equation_of_curve_C (x y m : ℝ) (hx : x ≠ 0) (hm : m > 1) (k1 k2 : ℝ) 
  (h_k1 : k1 = (y - 1) / x) (h_k2 : k2 = (y + 1) / (2 * x))
  (h_prod : k1 * k2 = -1 / m^2) :
  (x^2) / (m^2) + (y^2) = 1 := 
sorry

theorem range_of_m (m : ℝ) :
  (1 < m ∧ m ≤ Real.sqrt 3)
  ∨ (m < 1 ∨ m > Real.sqrt 3) :=
sorry

end equation_of_curve_C_range_of_m_l16_16786


namespace quadratic_eq_has_two_distinct_real_roots_l16_16281

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Problem statement: Prove that the quadratic equation x^2 + 3x - 2 = 0 has two distinct real roots
theorem quadratic_eq_has_two_distinct_real_roots :
  discriminant 1 3 (-2) > 0 :=
by
  -- Proof goes here
  sorry

end quadratic_eq_has_two_distinct_real_roots_l16_16281


namespace matt_profit_trade_l16_16969

theorem matt_profit_trade
  (total_cards : ℕ := 8)
  (value_per_card : ℕ := 6)
  (traded_cards_count : ℕ := 2)
  (trade_value_per_card : ℕ := 6)
  (received_cards_count_1 : ℕ := 3)
  (received_value_per_card_1 : ℕ := 2)
  (received_cards_count_2 : ℕ := 1)
  (received_value_per_card_2 : ℕ := 9)
  (profit : ℕ := 3) :
  profit = (received_cards_count_1 * received_value_per_card_1 
           + received_cards_count_2 * received_value_per_card_2) 
           - (traded_cards_count * trade_value_per_card) :=
  by
  sorry

end matt_profit_trade_l16_16969


namespace angle_is_50_l16_16214

-- Define the angle, supplement, and complement
def angle (x : ℝ) := x
def supplement (x : ℝ) := 180 - x
def complement (x : ℝ) := 90 - x
def condition (x : ℝ) := supplement x = 3 * (complement x) + 10

theorem angle_is_50 :
  ∃ x : ℝ, condition x ∧ x = 50 :=
by
  -- Here we show the existence of x that satisfies the condition and is equal to 50
  sorry

end angle_is_50_l16_16214


namespace find_length_of_AB_l16_16957

-- Definitions of the conditions
def areas_ratio (A B C D : Point) (areaABC areaADC : ℝ) :=
  (areaABC / areaADC) = (7 / 3)

def total_length (A B C D : Point) (AB CD : ℝ) :=
  AB + CD = 280

-- Statement of the proof problem
theorem find_length_of_AB
  (A B C D : Point)
  (AB CD : ℝ)
  (areaABC areaADC : ℝ)
  (h_height_not_zero : h ≠ 0) -- Assumption to ensure height is non-zero
  (h_areas_ratio : areas_ratio A B C D areaABC areaADC)
  (h_total_length : total_length A B C D AB CD) :
  AB = 196 := sorry

end find_length_of_AB_l16_16957


namespace number_50_is_sample_size_l16_16295

def number_of_pairs : ℕ := 50
def is_sample_size (n : ℕ) : Prop := n = number_of_pairs

-- We are to show that 50 represents the sample size
theorem number_50_is_sample_size : is_sample_size 50 :=
sorry

end number_50_is_sample_size_l16_16295


namespace max_rectangle_area_l16_16746

theorem max_rectangle_area (P : ℕ) (hP : P = 40) (l w : ℕ) (h : 2 * l + 2 * w = P) : ∃ A, A = l * w ∧ ∀ l' w', 2 * l' + 2 * w' = P → l' * w' ≤ 100 :=
by 
  sorry

end max_rectangle_area_l16_16746


namespace f_is_n_l16_16666

noncomputable def f : ℕ+ → ℤ :=
  sorry

def f_defined_for_all_positive_integers (n : ℕ+) : Prop :=
  ∃ k, f n = k

def f_is_integer (n : ℕ+) : Prop :=
  ∃ k : ℤ, f n = k

def f_two_is_two : Prop :=
  f 2 = 2

def f_multiply_rule (m n : ℕ+) : Prop :=
  f (m * n) = f m * f n

def f_ordered (m n : ℕ+) (h : m > n) : Prop :=
  f m > f n

theorem f_is_n (n : ℕ+) :
  (f_defined_for_all_positive_integers n) →
  (f_is_integer n) →
  (f_two_is_two) →
  (∀ m n, f_multiply_rule m n) →
  (∀ m n (h : m > n), f_ordered m n h) →
  f n = n :=
sorry

end f_is_n_l16_16666


namespace solution_set_f_inequality_l16_16071

variable (f : ℝ → ℝ)

axiom domain_of_f : ∀ x : ℝ, true
axiom avg_rate_of_f : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 3
axiom f_at_5 : f 5 = 18

theorem solution_set_f_inequality : {x : ℝ | f (3 * x - 1) > 9 * x} = {x : ℝ | x > 2} :=
by
  sorry

end solution_set_f_inequality_l16_16071


namespace number_of_donuts_finished_l16_16233

-- Definitions from conditions
def ounces_per_donut : ℕ := 2
def ounces_per_pot : ℕ := 12
def cost_per_pot : ℕ := 3
def total_spent : ℕ := 18

-- Theorem statement
theorem number_of_donuts_finished (H1 : ounces_per_donut = 2)
                                   (H2 : ounces_per_pot = 12)
                                   (H3 : cost_per_pot = 3)
                                   (H4 : total_spent = 18) : 
  ∃ n : ℕ, n = 36 :=
  sorry

end number_of_donuts_finished_l16_16233


namespace votes_cast_l16_16118

theorem votes_cast (A F T : ℕ) (h1 : A = 40 * T / 100) (h2 : F = A + 58) (h3 : T = F + A) : 
  T = 290 := 
by
  sorry

end votes_cast_l16_16118


namespace julie_bought_boxes_l16_16393

-- Definitions for the conditions
def packages_per_box := 5
def sheets_per_package := 250
def sheets_per_newspaper := 25
def newspapers := 100

-- Calculations based on conditions
def total_sheets_needed := newspapers * sheets_per_newspaper
def sheets_per_box := packages_per_box * sheets_per_package

-- The goal: to prove that the number of boxes of paper Julie bought is 2
theorem julie_bought_boxes : total_sheets_needed / sheets_per_box = 2 :=
  by
    sorry

end julie_bought_boxes_l16_16393


namespace number_of_elements_in_set_P_l16_16264

theorem number_of_elements_in_set_P
  (p q : ℕ) -- we are dealing with non-negative integers here
  (h1 : p = 3 * q)
  (h2 : p + q = 4500)
  : p = 3375 :=
by
  sorry -- Proof goes here

end number_of_elements_in_set_P_l16_16264


namespace mike_total_spent_l16_16250

noncomputable def total_spent_by_mike (food_cost wallet_cost shirt_cost shoes_cost belt_cost 
  discounted_shirt_cost discounted_shoes_cost discounted_belt_cost : ℝ) : ℝ :=
  food_cost + wallet_cost + discounted_shirt_cost + discounted_shoes_cost + discounted_belt_cost

theorem mike_total_spent :
  let food_cost := 30
  let wallet_cost := food_cost + 60
  let shirt_cost := wallet_cost / 3
  let shoes_cost := 2 * wallet_cost
  let belt_cost := shoes_cost - 45
  let discounted_shirt_cost := shirt_cost - (0.2 * shirt_cost)
  let discounted_shoes_cost := shoes_cost - (0.15 * shoes_cost)
  let discounted_belt_cost := belt_cost - (0.1 * belt_cost)
  total_spent_by_mike food_cost wallet_cost shirt_cost shoes_cost belt_cost
    discounted_shirt_cost discounted_shoes_cost discounted_belt_cost = 418.50 := by
  sorry

end mike_total_spent_l16_16250


namespace unique_rectangle_l16_16131

theorem unique_rectangle (a b : ℝ) (h : a < b) :
  ∃! (x y : ℝ), (x < y) ∧ (2 * (x + y) = a + b) ∧ (x * y = (a * b) / 4) := 
sorry

end unique_rectangle_l16_16131


namespace positive_difference_abs_eq_l16_16006

theorem positive_difference_abs_eq (x₁ x₂ : ℝ) (h₁ : x₁ - 3 = 15) (h₂ : x₂ - 3 = -15) : x₁ - x₂ = 30 :=
by
  sorry

end positive_difference_abs_eq_l16_16006


namespace problem1_problem2_l16_16042

variable {x : ℝ} (hx : x > 0)

theorem problem1 : (2 / (3 * x)) * Real.sqrt (9 * x^3) + 6 * Real.sqrt (x / 4) - 2 * x * Real.sqrt (1 / x) = 3 * Real.sqrt x := 
by sorry

theorem problem2 : (Real.sqrt 24 + Real.sqrt 6) / Real.sqrt 3 + (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 3 * Real.sqrt 2 + 2 := 
by sorry

end problem1_problem2_l16_16042


namespace parabola_and_hyperbola_equations_l16_16619

-- Definitions: Conditions from a)
def common_focus_on_x_axis : Prop :=
  ∃ p : ℝ, p > 0 ∧ (λ x y : ℝ, y^2 = 2 * p * x) ∧ (λ x y : ℝ, y = 2 * x ∨ y = -2 * x)

def asymptote_intersects_at_point : Prop :=
  ∃ P : ℝ × ℝ, P = (4, 8)

-- Prove that:
theorem parabola_and_hyperbola_equations :
  common_focus_on_x_axis → asymptote_intersects_at_point →
  (∃ p : ℝ, p = 8 ∧ ∀ x y : ℝ, y^2 = 16 * x) ∧ 
  (∃ λ : ℝ, λ = 16 / 5 ∧ ∀ x y : ℝ, (5 * x^2) / 16 - (5 * y^2) / 64 = 1) :=
by
  intros
  sorry

end parabola_and_hyperbola_equations_l16_16619


namespace sum_of_squares_of_solutions_l16_16602

theorem sum_of_squares_of_solutions :
  (∑ α in {x | ∃ ε : ℝ, ε = x² - x + 1/2010 ∧ |ε| = 1/2010}, α^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l16_16602


namespace Alma_test_score_l16_16844

-- Define the constants and conditions
variables (Alma_age Melina_age Alma_score : ℕ)

-- Conditions
axiom Melina_is_60 : Melina_age = 60
axiom Melina_3_times_Alma : Melina_age = 3 * Alma_age
axiom sum_ages_twice_score : Melina_age + Alma_age = 2 * Alma_score

-- Goal
theorem Alma_test_score : Alma_score = 40 :=
by
  sorry

end Alma_test_score_l16_16844


namespace elephant_weight_equivalence_l16_16461

-- Define the conditions as variables
def elephants := 1000000000
def buildings := 25000

-- Define the question and expected answer
def expected_answer := 40000

-- State the theorem
theorem elephant_weight_equivalence:
  (elephants / buildings = expected_answer) :=
by
  sorry

end elephant_weight_equivalence_l16_16461


namespace cows_eat_husk_l16_16092

theorem cows_eat_husk :
  ∀ (cows : ℕ) (days : ℕ) (husk_per_cow : ℕ),
    cows = 45 →
    days = 45 →
    husk_per_cow = 1 →
    (cows * husk_per_cow = 45) :=
by
  intros cows days husk_per_cow h_cows h_days h_husk_per_cow
  sorry

end cows_eat_husk_l16_16092


namespace find_reciprocal_l16_16966

open Real

theorem find_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^3 + y^3 + 1 / 27 = x * y) : 1 / x = 3 := 
sorry

end find_reciprocal_l16_16966


namespace real_solutions_unique_l16_16055

theorem real_solutions_unique (a b c : ℝ) :
  (2 * a - b = a^2 * b ∧ 2 * b - c = b^2 * c ∧ 2 * c - a = c^2 * a) →
  (a, b, c) = (-1, -1, -1) ∨ (a, b, c) = (0, 0, 0) ∨ (a, b, c) = (1, 1, 1) :=
by
  sorry

end real_solutions_unique_l16_16055


namespace katharina_order_is_correct_l16_16095

-- Define the mixed up order around a circle starting with L
def mixedUpOrder : List Char := ['L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']

-- Define the positions and process of Jaxon's list generation
def jaxonList : List Nat := [1, 4, 7, 3, 8, 5, 2, 6]

-- Define the resulting order from Jaxon's process
def resultingOrder (initialList : List Char) (positions : List Nat) : List Char :=
  positions.map (λ i => initialList.get! (i - 1))

-- Define the function to prove Katharina's order
theorem katharina_order_is_correct :
  resultingOrder mixedUpOrder jaxonList = ['L', 'R', 'O', 'M', 'S', 'Q', 'N', 'P'] :=
by
  -- Proof omitted
  sorry

end katharina_order_is_correct_l16_16095


namespace sum_of_inverses_gt_one_l16_16713

variable (a1 a2 a3 S : ℝ)

theorem sum_of_inverses_gt_one
  (h1 : a1 > 1)
  (h2 : a2 > 1)
  (h3 : a3 > 1)
  (h_sum : a1 + a2 + a3 = S)
  (ineq1 : a1^2 / (a1 - 1) > S)
  (ineq2 : a2^2 / (a2 - 1) > S)
  (ineq3 : a3^2 / (a3 - 1) > S) :
  1 / (a1 + a2) + 1 / (a2 + a3) + 1 / (a3 + a1) > 1 := by
  sorry

end sum_of_inverses_gt_one_l16_16713


namespace length_of_BC_l16_16090

-- Definitions of given conditions
def AB : ℝ := 4
def AC : ℝ := 3
def dot_product_AC_BC : ℝ := 1

-- Hypothesis used in the problem
axiom nonneg_AC (AC : ℝ) : AC ≥ 0
axiom nonneg_AB (AB : ℝ) : AB ≥ 0

-- Statement to be proved
theorem length_of_BC (AB AC dot_product_AC_BC : ℝ)
  (h1 : AB = 4) (h2 : AC = 3) (h3 : dot_product_AC_BC = 1) : exists (BC : ℝ), BC = 3 := by
  sorry

end length_of_BC_l16_16090


namespace range_of_b_l16_16791

theorem range_of_b (b : ℝ) :
  (∃ x : ℝ, x^2 - 2 * b * x + b^2 + b - 5 = 0) ∧
  (∀ x < 3.5, ∃ δ > 0, ∀ ε, x < ε → ε^2 - 2 * b * ε + b^2 + b - 5 < x^2 - 2 * b * x + b^2 + b - 5) →
  (3.5 ≤ b ∧ b ≤ 5) :=
by
  sorry

end range_of_b_l16_16791


namespace find_number_l16_16738

theorem find_number (number : ℝ) (h : 0.003 * number = 0.15) : number = 50 :=
by
  sorry

end find_number_l16_16738


namespace waitress_tips_average_l16_16894

theorem waitress_tips_average :
  let tip1 := (2 : ℚ) / 4
  let tip2 := (3 : ℚ) / 8
  let tip3 := (5 : ℚ) / 16
  let tip4 := (1 : ℚ) / 4
  (tip1 + tip2 + tip3 + tip4) / 4 = 23 / 64 :=
by
  sorry

end waitress_tips_average_l16_16894


namespace total_distance_between_first_and_fifth_poles_l16_16353

noncomputable def distance_between_poles (n : ℕ) (d : ℕ) : ℕ :=
  d / n

theorem total_distance_between_first_and_fifth_poles :
  ∀ (n : ℕ) (d : ℕ), (n = 3 ∧ d = 90) → (4 * distance_between_poles n d = 120) :=
by
  sorry

end total_distance_between_first_and_fifth_poles_l16_16353


namespace reporters_not_covering_politics_l16_16532

theorem reporters_not_covering_politics (P_X P_Y P_Z intlPol otherPol econOthers : ℝ)
  (h1 : P_X = 0.15) (h2 : P_Y = 0.10) (h3 : P_Z = 0.08)
  (h4 : otherPol = 0.50) (h5 : intlPol = 0.05) (h6 : econOthers = 0.02) :
  (1 - (P_X + P_Y + P_Z + intlPol + otherPol + econOthers)) = 0.10 := by
  sorry

end reporters_not_covering_politics_l16_16532


namespace perfect_square_condition_l16_16675

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l16_16675


namespace kolya_advantageous_methods_l16_16873

-- Define the context and conditions
variables (n : ℕ) (h₀ : n ≥ 2)
variables (a b : ℕ) (h₁ : a + b = 2*n + 1) (h₂ : a ≥ 2) (h₃ : b ≥ 2)

-- Define outcomes of the methods
def method1_outcome (a b : ℕ) := max a b + min (a - 1) (b - 1)
def method2_outcome (a b : ℕ) := min a b + min (a - 1) (b - 1)
def method3_outcome (a b : ℕ) := max (method1_outcome a b - 1) (method2_outcome a b - 1)

-- Prove which methods are the most and least advantageous
theorem kolya_advantageous_methods :
  method1_outcome a b >= method2_outcome a b ∧ method1_outcome a b >= method3_outcome a b :=
sorry

end kolya_advantageous_methods_l16_16873


namespace distinct_digits_unique_D_l16_16822

theorem distinct_digits_unique_D 
  (A B C D : ℕ)
  (hA : A ≠ B)
  (hB : B ≠ C)
  (hC : C ≠ D)
  (hD : D ≠ A)
  (h1 : D < 10)
  (h2 : B < 10)
  (h3 : C < 10)
  (h4 : A < 10)
  (h_add : A * 1000 + A * 100 + C * 10 + B + B * 1000 + C * 100 + B * 10 + D = B * 1000 + D * 100 + A * 10 + B) :
  D = 0 :=
by sorry

end distinct_digits_unique_D_l16_16822


namespace minhyuk_needs_slices_l16_16305

-- Definitions of Yeongchan and Minhyuk's apple division
def yeongchan_portion : ℚ := 1 / 3
def minhyuk_slices : ℚ := 1 / 12

-- Statement to prove
theorem minhyuk_needs_slices (x : ℕ) : yeongchan_portion = x * minhyuk_slices → x = 4 :=
by
  sorry

end minhyuk_needs_slices_l16_16305


namespace find_f_values_l16_16967

def func_property1 (f : ℕ → ℕ) : Prop := 
  ∀ a b : ℕ, a ≠ b → a * f a + b * f b > a * f b + b * f a

def func_property2 (f : ℕ → ℕ) : Prop := 
  ∀ n : ℕ, f (f n) = 3 * n

theorem find_f_values (f : ℕ → ℕ) (h1 : func_property1 f) (h2 : func_property2 f) : 
  f 1 + f 6 + f 28 = 66 :=
sorry

end find_f_values_l16_16967


namespace min_tip_percentage_l16_16182

noncomputable def meal_cost : ℝ := 37.25
noncomputable def total_paid : ℝ := 40.975
noncomputable def tip_percentage (P : ℝ) : Prop := P > 0 ∧ P < 15 ∧ (meal_cost + (P/100) * meal_cost = total_paid)

theorem min_tip_percentage : ∃ P : ℝ, tip_percentage P ∧ P = 10 := by
  sorry

end min_tip_percentage_l16_16182


namespace gcd_of_polynomial_l16_16366

def multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem gcd_of_polynomial (b : ℕ) (h : multiple_of b 456) :
  Nat.gcd (4 * b^3 + b^2 + 6 * b + 152) b = 152 := sorry

end gcd_of_polynomial_l16_16366


namespace factorization_correct_l16_16869

theorem factorization_correct (x y : ℝ) : x^2 - 4 * y^2 = (x - 2 * y) * (x + 2 * y) :=
by sorry

end factorization_correct_l16_16869


namespace renovate_total_time_eq_79_5_l16_16157

-- Definitions based on the given conditions
def time_per_bedroom : ℝ := 4
def num_bedrooms : ℕ := 3
def time_per_kitchen : ℝ := time_per_bedroom * 1.5
def time_per_garden : ℝ := 3
def time_per_terrace : ℝ := time_per_garden - 2
def time_per_basement : ℝ := time_per_kitchen * 0.75

-- Total time excluding the living room
def total_time_excl_living_room : ℝ :=
  (num_bedrooms * time_per_bedroom) +
  time_per_kitchen +
  time_per_garden +
  time_per_terrace +
  time_per_basement

-- Time for the living room
def time_per_living_room : ℝ := 2 * total_time_excl_living_room

-- Total time for everything
def total_time : ℝ := total_time_excl_living_room + time_per_living_room

-- The theorem we need to prove
theorem renovate_total_time_eq_79_5 : total_time = 79.5 := by
  sorry

end renovate_total_time_eq_79_5_l16_16157


namespace smallest_area_of_square_containing_rectangles_l16_16313

noncomputable def smallest_area_square : ℕ :=
  let side1 := 3
  let side2 := 5
  let side3 := 4
  let side4 := 6
  let smallest_side := side1 + side3
  let square_area := smallest_side * smallest_side
  square_area

theorem smallest_area_of_square_containing_rectangles : smallest_area_square = 49 :=
by
  sorry

end smallest_area_of_square_containing_rectangles_l16_16313


namespace geometric_sequence_n_l16_16956

-- Definition of the conditions

-- a_1 + a_n = 82
def condition1 (a₁ an : ℕ) : Prop := a₁ + an = 82
-- a_3 * a_{n-2} = 81
def condition2 (a₃ aₙm2 : ℕ) : Prop := a₃ * aₙm2 = 81
-- S_n = 121
def condition3 (Sₙ : ℕ) : Prop := Sₙ = 121

-- Prove n = 5 given the above conditions
theorem geometric_sequence_n (a₁ a₃ an aₙm2 Sₙ n : ℕ)
  (h1 : condition1 a₁ an)
  (h2 : condition2 a₃ aₙm2)
  (h3 : condition3 Sₙ) :
  n = 5 :=
sorry

end geometric_sequence_n_l16_16956


namespace nat_perfect_square_l16_16679

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l16_16679


namespace ten_elements_sequence_no_infinite_sequence_l16_16840

def is_valid_seq (a : ℕ → ℕ) : Prop :=
  ∀ n, (a (n + 1))^2 - 4 * (a n) * (a (n + 2)) ≥ 0

theorem ten_elements_sequence : 
  ∃ a : ℕ → ℕ, (a 9 + 1 = 10) ∧ is_valid_seq a :=
sorry

theorem no_infinite_sequence :
  ¬∃ a : ℕ → ℕ, is_valid_seq a ∧ ∀ n, a n ≥ 1 :=
sorry

end ten_elements_sequence_no_infinite_sequence_l16_16840


namespace triangle_max_perimeter_l16_16094

noncomputable def max_perimeter_triangle_ABC (a b c : ℝ) (A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) : ℝ := 
  a + b + c

theorem triangle_max_perimeter (a b c A B C : ℝ) (h1 : B = 60) (h2 : b = 2 * Real.sqrt 3) :
  max_perimeter_triangle_ABC a b c A B C h1 h2 ≤ 6 * Real.sqrt 3 :=
sorry

end triangle_max_perimeter_l16_16094


namespace scientific_notation_of_750000_l16_16320

theorem scientific_notation_of_750000 : 750000 = 7.5 * 10^5 :=
by
  sorry

end scientific_notation_of_750000_l16_16320


namespace problem_part1_problem_part2_l16_16073

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + abs (2*x - a)

-- Proof statements
theorem problem_part1 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) : a = 0 := sorry

theorem problem_part2 (a : ℝ) (h_a_gt_two : a > 2) : 
  ∃ x : ℝ, ∀ y : ℝ, f x a ≤ f y a ∧ f x a = a - 1 := sorry

end problem_part1_problem_part2_l16_16073


namespace fraction_of_innocent_cases_l16_16882

-- Definitions based on the given conditions
def total_cases : ℕ := 17
def dismissed_cases : ℕ := 2
def delayed_cases : ℕ := 1
def guilty_cases : ℕ := 4

-- The remaining cases after dismissals
def remaining_cases : ℕ := total_cases - dismissed_cases

-- The remaining cases that are not innocent
def non_innocent_cases : ℕ := delayed_cases + guilty_cases

-- The innocent cases
def innocent_cases : ℕ := remaining_cases - non_innocent_cases

-- The fraction of the remaining cases that were ruled innocent
def fraction_innocent : Rat := innocent_cases / remaining_cases

-- The theorem we want to prove
theorem fraction_of_innocent_cases :
  fraction_innocent = 2 / 3 := by
  sorry

end fraction_of_innocent_cases_l16_16882


namespace ratio_of_a_over_5_to_b_over_4_l16_16221

theorem ratio_of_a_over_5_to_b_over_4 (a b : ℝ) (h1 : 4 * a = 5 * b) (h2 : a ≠ 0 ∧ b ≠ 0) :
  (a / 5) / (b / 4) = 1 := by
  sorry

end ratio_of_a_over_5_to_b_over_4_l16_16221


namespace solve_equation_l16_16282

theorem solve_equation : ∀ x : ℝ, (2 / 3 * x - 2 = 4) → x = 9 :=
by
  intro x
  intro h
  sorry

end solve_equation_l16_16282


namespace inequality_proof_l16_16973

theorem inequality_proof (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := 
by
  sorry

end inequality_proof_l16_16973


namespace not_a_solution_set4_l16_16187

def set1 : ℝ × ℝ := (1, 2)
def set2 : ℝ × ℝ := (2, 0)
def set3 : ℝ × ℝ := (0.5, 3)
def set4 : ℝ × ℝ := (-2, 4)

noncomputable def is_solution (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 = 4

theorem not_a_solution_set4 : ¬ is_solution set4 := 
by 
  sorry

end not_a_solution_set4_l16_16187


namespace radius_of_congruent_spheres_in_cone_l16_16959

noncomputable def radius_of_congruent_spheres (base_radius height : ℝ) : ℝ := 
  let slant_height := Real.sqrt (height^2 + base_radius^2)
  let r := (4 : ℝ) / (10 + 4) * slant_height
  r

theorem radius_of_congruent_spheres_in_cone :
  radius_of_congruent_spheres 4 10 = 4 * Real.sqrt 29 / 7 := by
  sorry

end radius_of_congruent_spheres_in_cone_l16_16959


namespace smallest_possible_k_l16_16810

def infinite_increasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a n < a (n + 1)

def divisible_by_1005_or_1006 (a : ℕ) : Prop :=
a % 1005 = 0 ∨ a % 1006 = 0

def not_divisible_by_97 (a : ℕ) : Prop :=
a % 97 ≠ 0

def diff_less_than_k (a : ℕ → ℕ) (k : ℕ) : Prop :=
∀ n, (a (n + 1) - a n) ≤ k

theorem smallest_possible_k :
  ∀ (a : ℕ → ℕ), infinite_increasing_seq a →
  (∀ n, divisible_by_1005_or_1006 (a n)) →
  (∀ n, not_divisible_by_97 (a n)) →
  (∃ k, diff_less_than_k a k) →
  (∃ k, k = 2010 ∧ diff_less_than_k a k) :=
by
  sorry

end smallest_possible_k_l16_16810


namespace positive_difference_abs_eq_15_l16_16001

theorem positive_difference_abs_eq_15:
  ∃ x1 x2 : ℝ, (| x1 - 3 | = 15 ∧ | x2 - 3 | = 15) ∧ | x1 - x2 | = 30 :=
by
  sorry

end positive_difference_abs_eq_15_l16_16001


namespace james_fish_tanks_l16_16390

theorem james_fish_tanks (n t1 t2 t3 : ℕ) (h1 : t1 = 20) (h2 : t2 = 2 * t1) (h3 : t3 = 2 * t1) (h4 : t1 + t2 + t3 = 100) : n = 3 :=
sorry

end james_fish_tanks_l16_16390


namespace chicken_nuggets_cost_l16_16246

theorem chicken_nuggets_cost :
  ∀ (nuggets_ordered boxes_cost : ℕ) (nuggets_per_box : ℕ),
  nuggets_ordered = 100 →
  nuggets_per_box = 20 →
  boxes_cost = 4 →
  (nuggets_ordered / nuggets_per_box) * boxes_cost = 20 :=
by
  intros nuggets_ordered boxes_cost nuggets_per_box h1 h2 h3
  sorry

end chicken_nuggets_cost_l16_16246


namespace problem_statement_l16_16354

def diamond (x y : ℝ) : ℝ := (x + y) ^ 2 * (x - y) ^ 2

theorem problem_statement : diamond 2 (diamond 3 4) = 5745329 := by
  sorry

end problem_statement_l16_16354


namespace fraction_division_l16_16198

theorem fraction_division :
  (5 : ℚ) / ((13 : ℚ) / 7) = 35 / 13 :=
by
  sorry

end fraction_division_l16_16198


namespace abs_of_negative_l16_16724

theorem abs_of_negative (a : ℝ) (h : a < 0) : |a| = -a :=
sorry

end abs_of_negative_l16_16724


namespace distribute_teachers_to_schools_l16_16772

theorem distribute_teachers_to_schools : 
  ∃ (n : ℕ), n = 5 ∧ ∃ (m : ℕ), m = 3 ∧ (∑ (i : Finset (Fin m)) in (Finset.powerset_len 1 (Finset.univ)) ∪ (Finset.powerset_len 2 (Finset.univ)), 
  if i.card = 1 then ( finset.card ((finset.univ.image (λ (x : Fin m), choose 5 x.factorial) * 
  finset.card ((finset.univ.image factorial.card) * choose 3).factorial)) else 0 +
  if i.card = 2 then ( finset.card ((finset.univ.image (λ (x : Fin m), choose 5 x.factorial) * 
  finset.card ((finset.univ.image factorial.card) * choose 3).factorial)) else 0
  ) = 150 := by sorry

end distribute_teachers_to_schools_l16_16772


namespace attractions_visit_order_l16_16103

-- Define the conditions
def type_A_count : ℕ := 2
def type_B_count : ℕ := 4

-- Define the requirements that Javier visits type A attractions before type B
theorem attractions_visit_order : ∀ (A B : ℕ), 
  A = type_A_count → 
  B = type_B_count → 
  (fact A) * (fact B) = 48 :=
by
  intros A B hA hB
  rw [hA, hB]
  dsimp
  norm_num
  sorry

end attractions_visit_order_l16_16103


namespace rhombus_side_length_l16_16708

/-
  Define the length of the rhombus diagonal and the area of the rhombus.
-/
def diagonal1 : ℝ := 20
def area : ℝ := 480

/-
  The theorem states that given these conditions, the length of each side of the rhombus is 26 m.
-/
theorem rhombus_side_length (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = diagonal1) (h2 : A = area):
  2 * 26 * 26 * 2 = A * 2 * 2 + (d1 / 2) * (d1 / 2) :=
sorry

end rhombus_side_length_l16_16708


namespace sequence_a_n_perfect_square_l16_16207

theorem sequence_a_n_perfect_square :
  (∃ a : ℕ → ℤ, ∃ b : ℕ → ℤ,
    a 0 = 1 ∧ b 0 = 0 ∧
    (∀ n : ℕ, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n : ℕ, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    (∀ n : ℕ, ∃ k : ℤ, a n = k^2)) :=
sorry

end sequence_a_n_perfect_square_l16_16207


namespace max_buses_l16_16646

-- Define the total number of stops and buses as finite sets
def stops : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define a bus as a subset of stops with exactly 3 stops
structure Bus :=
(stops : Finset ℕ)
(h_stops : stops.card = 3)

-- Define the condition that any two buses share at most one stop
def shares_at_most_one (b1 b2 : Bus) : Prop :=
(b1.stops ∩ b2.stops).card ≤ 1

-- Define the predicate for a valid set of buses
def valid_buses (buses : Finset Bus) : Prop :=
∀ b1 b2 ∈ buses, b1 ≠ b2 → shares_at_most_one b1 b2

-- The main theorem statement
theorem max_buses (buses : Finset Bus) (h_valid : valid_buses buses) : buses.card ≤ 12 :=
sorry

end max_buses_l16_16646


namespace smallest_area_square_l16_16311

theorem smallest_area_square (a b u v : ℕ) (h₁ : a = 3) (h₂ : b = 5) (h₃ : u = 4) (h₄ : v = 6) :
  ∃ s : ℕ, s^2 = 81 ∧ 
    (∀ xa ya xb yb xu yu xv yv : ℕ, 
      (xa + a ≤ s) ∧ (ya + b ≤ s) ∧ (xb + u ≤ s) ∧ (yb + v ≤ s) ∧ 
      ─xa < xb → xb < xa + a → ─ya < yb → yb < ya + b →
      ─xu < xv → xv < xu + u → ─yu < yv → yv < yu + v ∧
      (ya + b ≤ yv ∨ yu + v ≤ yb))
    := sorry

end smallest_area_square_l16_16311


namespace least_possible_value_of_expression_l16_16999

noncomputable def min_expression_value (x : ℝ) : ℝ :=
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023

theorem least_possible_value_of_expression :
  ∃ x : ℝ, min_expression_value x = 2022 :=
by
  sorry

end least_possible_value_of_expression_l16_16999


namespace cos_30_eq_sqrt3_div_2_l16_16762

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l16_16762


namespace relation_xy_l16_16632

theorem relation_xy (a c b d : ℝ) (x y p : ℝ) 
  (h1 : a^x = c^(3 * p))
  (h2 : c^(3 * p) = b^2)
  (h3 : c^y = b^p)
  (h4 : b^p = d^3) :
  y = 3 * p^2 / 2 :=
by
  sorry

end relation_xy_l16_16632


namespace decreasing_function_range_l16_16805

theorem decreasing_function_range {a : ℝ} (h1 : ∀ x y : ℝ, x < y → (1 - 2 * a)^x > (1 - 2 * a)^y) : 
    0 < a ∧ a < 1 / 2 :=
by
  sorry

end decreasing_function_range_l16_16805


namespace fill_up_mini_vans_l16_16385

/--
In a fuel station, the service costs $2.20 per vehicle and every liter of fuel costs $0.70.
Assume that mini-vans have a tank size of 65 liters, and trucks have a tank size of 143 liters.
Given that 2 trucks were filled up and the total cost was $347.7,
prove the number of mini-vans filled up is 3.
-/
theorem fill_up_mini_vans (m : ℝ) (t : ℝ) 
    (service_cost_per_vehicle fuel_cost_per_liter : ℝ)
    (van_tank_size truck_tank_size total_cost : ℝ):
    service_cost_per_vehicle = 2.20 →
    fuel_cost_per_liter = 0.70 →
    van_tank_size = 65 →
    truck_tank_size = 143 →
    t = 2 →
    total_cost = 347.7 →
    (service_cost_per_vehicle * m + service_cost_per_vehicle * t) + (fuel_cost_per_liter * van_tank_size * m) + (fuel_cost_per_liter * truck_tank_size * t) = total_cost →
    m = 3 :=
by
  intros
  sorry

end fill_up_mini_vans_l16_16385


namespace product_of_sequence_l16_16865

theorem product_of_sequence :
  (1 + 1 / 1) * (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) *
  (1 + 1 / 6) * (1 + 1 / 7) * (1 + 1 / 8) = 9 :=
by sorry

end product_of_sequence_l16_16865


namespace poly_roots_equivalence_l16_16561

noncomputable def poly (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem poly_roots_equivalence (a b c d : ℝ) 
    (h1 : poly a b c d 4 = 102) 
    (h2 : poly a b c d 3 = 102) 
    (h3 : poly a b c d (-3) = 102) 
    (h4 : poly a b c d (-4) = 102) : 
    {x : ℝ | poly a b c d x = 246} = {0, 5, -5} := 
by 
    sorry

end poly_roots_equivalence_l16_16561


namespace eesha_late_by_15_minutes_l16_16531

theorem eesha_late_by_15_minutes 
  (T usual_time : ℕ) (delay : ℕ) (slower_factor : ℚ) (T' : ℕ) 
  (usual_time_eq : usual_time = 60)
  (delay_eq : delay = 30)
  (slower_factor_eq : slower_factor = 0.75)
  (new_time_eq : T' = unusual_time * slower_factor) 
  (T'' : ℕ) (total_time_eq: T'' = T' + delay)
  (time_taken : ℕ) (time_diff_eq : time_taken = T'' - usual_time) :
  time_taken = 15 :=
by
  -- Proof construction
  sorry

end eesha_late_by_15_minutes_l16_16531


namespace car_wash_cost_l16_16231

-- Definitions based on the conditions
def washes_per_bottle : ℕ := 4
def bottle_cost : ℕ := 4   -- Assuming cost is recorded in dollars
def total_weeks : ℕ := 20

-- Stating the problem
theorem car_wash_cost : (total_weeks / washes_per_bottle) * bottle_cost = 20 := 
by
  -- Placeholder for the proof
  sorry

end car_wash_cost_l16_16231


namespace sector_area_l16_16503

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) (area : ℝ) 
  (h1 : arc_length = 6) 
  (h2 : central_angle = 2) 
  (h3 : radius = arc_length / central_angle): 
  area = (1 / 2) * arc_length * radius := 
  sorry

end sector_area_l16_16503


namespace lottery_probability_approximation_l16_16009

noncomputable def binom (n k : ℕ) : ℚ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem lottery_probability_approximation :
  (1 - (binom 85 5 / binom 90 5)) ≈ 0.25365 := 
sorry

end lottery_probability_approximation_l16_16009


namespace total_days_in_month_eq_l16_16896

-- Definition of the conditions
def took_capsules_days : ℕ := 27
def forgot_capsules_days : ℕ := 4

-- The statement to be proved
theorem total_days_in_month_eq : took_capsules_days + forgot_capsules_days = 31 := by
  sorry

end total_days_in_month_eq_l16_16896


namespace caroline_lassis_l16_16482

theorem caroline_lassis (c : ℕ → ℕ): c 3 = 13 → c 15 = 65 :=
by
  sorry

end caroline_lassis_l16_16482


namespace banana_count_l16_16847

theorem banana_count : (2 + 7) = 9 := by
  rfl

end banana_count_l16_16847


namespace part1_solution_set_k_3_part2_solution_set_k_lt_0_l16_16608

open Set

-- Definitions
def inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 + (k - 2) * x - 2 < 0

-- Part 1: When k = 3
theorem part1_solution_set_k_3 : ∀ x : ℝ, inequality 3 x ↔ -1 < x ∧ x < (2 / 3) :=
by
  sorry

-- Part 2: When k < 0
theorem part2_solution_set_k_lt_0 :
  ∀ k : ℝ, k < 0 → 
    (k = -2 → ∀ x : ℝ, inequality k x ↔ x ≠ -1) ∧
    (k < -2 → ∀ x : ℝ, inequality k x ↔ x < -1 ∨ x > 2 / k) ∧
    (-2 < k → ∀ x : ℝ, inequality k x ↔ x > -1 ∨ x < 2 / k) :=
by
  sorry

end part1_solution_set_k_3_part2_solution_set_k_lt_0_l16_16608


namespace sum_m_n_zero_l16_16924

theorem sum_m_n_zero
  (m n p : ℝ)
  (h1 : mn + p^2 + 4 = 0)
  (h2 : m - n = 4) :
  m + n = 0 :=
sorry

end sum_m_n_zero_l16_16924


namespace hundred_times_reciprocal_l16_16800

theorem hundred_times_reciprocal (x : ℝ) (h : 5 * x = 2) : 100 * (1 / x) = 250 := 
by 
  sorry

end hundred_times_reciprocal_l16_16800


namespace polynomial_root_sum_l16_16819

theorem polynomial_root_sum :
  ∃ a b c : ℝ,
    (∀ x : ℝ, Polynomial.eval x (Polynomial.X ^ 3 - 10 * Polynomial.X ^ 2 + 16 * Polynomial.X - 2) = 0) →
    a + b + c = 10 → ab + ac + bc = 16 → abc = 2 →
    (a / (bc + 2) + b / (ac + 2) + c / (ab + 2) = 4) := sorry

end polynomial_root_sum_l16_16819


namespace motorcyclists_speeds_l16_16161

theorem motorcyclists_speeds 
  (distance_AB : ℝ) (distance1 : ℝ) (distance2 : ℝ) (time_diff : ℝ) 
  (x y : ℝ) 
  (h1 : distance_AB = 600) 
  (h2 : distance1 = 250) 
  (h3 : distance2 = 200) 
  (h4 : time_diff = 3)
  (h5 : distance1 / x = distance2 / y)
  (h6 : distance_AB / x + time_diff = distance_AB / y) : 
  x = 50 ∧ y = 40 := 
sorry

end motorcyclists_speeds_l16_16161


namespace kelly_carrot_weight_l16_16100

-- Define the number of carrots harvested from each bed
def carrots_bed1 : ℕ := 55
def carrots_bed2 : ℕ := 101
def carrots_bed3 : ℕ := 78
def carrots_per_pound : ℕ := 6

-- Define the total number of carrots
def total_carrots := carrots_bed1 + carrots_bed2 + carrots_bed3

-- Define the total weight in pounds
def total_weight := total_carrots / carrots_per_pound

-- The theorem to prove the total weight is 39 pounds
theorem kelly_carrot_weight : total_weight = 39 := by
  sorry

end kelly_carrot_weight_l16_16100


namespace quadrilateral_area_inequality_l16_16533

theorem quadrilateral_area_inequality (a b c d : ℝ) :
  ∃ (S_ABCD : ℝ), S_ABCD ≤ (1 / 4) * (a + c) ^ 2 + b * d :=
sorry

end quadrilateral_area_inequality_l16_16533


namespace factorize_ax2_minus_a_l16_16052

theorem factorize_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_ax2_minus_a_l16_16052


namespace molecular_weight_one_mole_l16_16862

variable (molecular_weight : ℕ → ℕ)

theorem molecular_weight_one_mole (h : molecular_weight 7 = 2856) :
  molecular_weight 1 = 408 :=
sorry

end molecular_weight_one_mole_l16_16862


namespace john_father_age_difference_l16_16149

theorem john_father_age_difference (J F X : ℕ) (h1 : J + F = 77) (h2 : J = 15) (h3 : F = 2 * J + X) : X = 32 :=
by
  -- Adding the "sory" to skip the proof
  sorry

end john_father_age_difference_l16_16149


namespace smaller_of_x_y_l16_16980

theorem smaller_of_x_y (x y a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : x * y = c) (h6 : x^2 - b * x + a * y = 0) : min x y = c / a :=
by sorry

end smaller_of_x_y_l16_16980


namespace inequality_system_solution_l16_16268

theorem inequality_system_solution:
  ∀ (x : ℝ),
  (1 - (2*x - 1) / 2 > (3*x - 1) / 4) ∧ (2 - 3*x ≤ 4 - x) →
  -1 ≤ x ∧ x < 1 :=
by
  intro x
  intro h
  sorry

end inequality_system_solution_l16_16268


namespace star_polygon_points_l16_16344

theorem star_polygon_points (n : ℕ) (A B : ℕ → ℝ) 
  (h_angles_congruent_A : ∀ i j, A i = A j)
  (h_angles_congruent_B : ∀ i j, B i = B j)
  (h_angle_relation : ∀ i, A i = B i - 15) :
  n = 24 :=
by
  sorry

end star_polygon_points_l16_16344


namespace find_x2_plus_y2_l16_16911

theorem find_x2_plus_y2 (x y : ℕ) (h1 : xy + x + y = 35) (h2 : x^2 * y + x * y^2 = 306) : x^2 + y^2 = 290 :=
sorry

end find_x2_plus_y2_l16_16911


namespace chapatis_order_count_l16_16751

theorem chapatis_order_count (chapati_cost rice_cost veg_cost total_paid chapati_count : ℕ) 
  (rice_plates veg_plates : ℕ)
  (H1 : chapati_cost = 6)
  (H2 : rice_cost = 45)
  (H3 : veg_cost = 70)
  (H4 : total_paid = 1111)
  (H5 : rice_plates = 5)
  (H6 : veg_plates = 7)
  (H7 : chapati_count = (total_paid - (rice_plates * rice_cost + veg_plates * veg_cost)) / chapati_cost) :
  chapati_count = 66 :=
by
  sorry

end chapatis_order_count_l16_16751


namespace slope_intercept_of_line_l16_16926

theorem slope_intercept_of_line :
  ∃ (l : ℝ → ℝ), (∀ x, l x = (4 * x - 9) / 3) ∧ l 3 = 1 ∧ ∃ k, k / (1 + k^2) = 1 / 2 ∧ l x = (k^2 - 1) / (1 + k^2) := sorry

end slope_intercept_of_line_l16_16926


namespace center_determines_position_l16_16569

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define what it means for the Circle's position being determined by its center.
theorem center_determines_position (c : Circle) : c.center = c.center :=
by
  sorry

end center_determines_position_l16_16569


namespace ron_l16_16019

-- Definitions for the given problem conditions
def cost_of_chocolate_bar : ℝ := 1.5
def s'mores_per_chocolate_bar : ℕ := 3
def number_of_scouts : ℕ := 15
def s'mores_per_scout : ℕ := 2

-- Proof that Ron will spend $15.00 on chocolate bars
theorem ron's_chocolate_bar_cost :
  (number_of_scouts * s'mores_per_scout / s'mores_per_chocolate_bar) * cost_of_chocolate_bar = 15 :=
by
  sorry

end ron_l16_16019


namespace first_cat_blue_eyed_kittens_l16_16668

variable (B : ℕ)
variable (C1 : 35 * (B + 17) = 100 * (B + 4))

theorem first_cat_blue_eyed_kittens : B = 3 :=
by
  -- proof
  sorry

end first_cat_blue_eyed_kittens_l16_16668


namespace fraction_of_menu_items_my_friend_can_eat_l16_16528

theorem fraction_of_menu_items_my_friend_can_eat {menu_size vegan_dishes nut_free_vegan_dishes : ℕ}
    (h1 : vegan_dishes = 6)
    (h2 : vegan_dishes = menu_size / 6)
    (h3 : nut_free_vegan_dishes = vegan_dishes - 5) :
    (nut_free_vegan_dishes : ℚ) / menu_size = 1 / 36 :=
by
  sorry

end fraction_of_menu_items_my_friend_can_eat_l16_16528


namespace sum_of_numbers_le_1_1_l16_16440

theorem sum_of_numbers_le_1_1 :
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  filtered.sum = 1.4 :=
by
  let nums := [1.4, 0.9, 1.2, 0.5, 1.3]
  let filtered := nums.filter (fun x => x <= 1.1)
  have : filtered = [0.9, 0.5] := sorry
  have : filtered.sum = 1.4 := sorry
  exact this

end sum_of_numbers_le_1_1_l16_16440


namespace find_extrema_l16_16985

noncomputable def f (x : ℝ) : ℝ := (4 * x) / (x ^ 2 + 1)

theorem find_extrema :
  ∃ (a b : ℝ), (∀ x ∈ Set.Icc (-2:ℝ) 2, f x ≤ a) ∧ 
               (∃ x ∈ Set.Icc (-2:ℝ) 2, f x = a) ∧
               (∀ x ∈ Set.Icc (-2:ℝ) 2, b ≤ f x) ∧
               (∃ x ∈ Set.Icc (-2:ℝ) 2, f x = b) ∧
               a = 2 ∧ b = -(8 / 5) := 
by
  use [2, -(8 / 5)]
  sorry

end find_extrema_l16_16985


namespace age_of_teacher_l16_16836

variables (age_students : ℕ) (age_all : ℕ) (teacher_age : ℕ)

def avg_age_students := 15
def num_students := 10
def num_people := 11
def avg_age_people := 16

theorem age_of_teacher
  (h1 : age_students = num_students * avg_age_students)
  (h2 : age_all = num_people * avg_age_people)
  (h3 : age_all = age_students + teacher_age) : teacher_age = 26 :=
by
  sorry

end age_of_teacher_l16_16836


namespace smallest_whole_number_above_perimeter_triangle_l16_16726

theorem smallest_whole_number_above_perimeter_triangle (s : ℕ) (h1 : 12 < s) (h2 : s < 26) :
  53 = Nat.ceil ((7 + 19 + s : ℕ) / 1) := by
  sorry

end smallest_whole_number_above_perimeter_triangle_l16_16726


namespace sum_of_vars_l16_16077

theorem sum_of_vars 
  (x y z : ℝ) 
  (h1 : x + y = 4) 
  (h2 : y + z = 6) 
  (h3 : z + x = 8) : 
  x + y + z = 9 := 
by 
  sorry

end sum_of_vars_l16_16077


namespace find_a1_l16_16706

theorem find_a1 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 1 / (1 - a n))
  (h2 : a 2 = 2)
  : a 1 = 1 / 2 :=
sorry

end find_a1_l16_16706


namespace division_of_fraction_simplified_l16_16040

theorem division_of_fraction_simplified :
  12 / (2 / (5 - 3)) = 12 := 
by
  sorry

end division_of_fraction_simplified_l16_16040


namespace solve_asterisk_l16_16566

theorem solve_asterisk (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end solve_asterisk_l16_16566


namespace complement_union_l16_16526

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4}

-- Define the set S
def S : Set ℕ := {1, 3}

-- Define the set T
def T : Set ℕ := {4}

-- Define the complement of S in I
def complement_I_S : Set ℕ := I \ S

-- State the theorem to be proved
theorem complement_union : (complement_I_S ∪ T) = {2, 4} := by
  sorry

end complement_union_l16_16526


namespace black_region_area_is_correct_l16_16180

noncomputable def area_of_black_region : ℕ :=
  let area_large_square := 10 * 10
  let area_first_smaller_square := 4 * 4
  let area_second_smaller_square := 2 * 2
  area_large_square - (area_first_smaller_square + area_second_smaller_square)

theorem black_region_area_is_correct :
  area_of_black_region = 80 :=
by
  sorry

end black_region_area_is_correct_l16_16180


namespace desired_interest_percentage_l16_16745

theorem desired_interest_percentage (face_value market_value dividend_percentage : ℝ) 
  (h1 : face_value = 20) (h2 : market_value = 15) (h3 : dividend_percentage = 0.09) : 
  let dividend_received := dividend_percentage * face_value in
  let interest_percentage := (dividend_received / market_value) * 100 in
  interest_percentage = 12 :=
by
  sorry

end desired_interest_percentage_l16_16745


namespace rectangle_area_l16_16579

theorem rectangle_area (x : ℝ) (w : ℝ) (h1 : (3 * w)^2 + w^2 = x^2) : (3 * w) * w = 3 * x^2 / 10 :=
by
  sorry

end rectangle_area_l16_16579


namespace middle_group_frequency_l16_16812

theorem middle_group_frequency (sample_size : ℕ) (num_rectangles : ℕ)
  (A_middle : ℝ) (other_area_sum : ℝ)
  (h1 : sample_size = 300)
  (h2 : num_rectangles = 9)
  (h3 : A_middle = 1 / 5 * other_area_sum)
  (h4 : other_area_sum + A_middle = 1) :
  sample_size * A_middle = 50 :=
by
  sorry

end middle_group_frequency_l16_16812


namespace pete_flag_total_circles_squares_l16_16127

def US_flag_stars : ℕ := 50
def US_flag_stripes : ℕ := 13

def circles (stars : ℕ) : ℕ := (stars / 2) - 3
def squares (stripes : ℕ) : ℕ := (2 * stripes) + 6

theorem pete_flag_total_circles_squares : 
  circles US_flag_stars + squares US_flag_stripes = 54 := 
by
  unfold circles squares US_flag_stars US_flag_stripes
  sorry

end pete_flag_total_circles_squares_l16_16127


namespace remainder_of_product_l16_16524

theorem remainder_of_product (a b c : ℕ) (hc : c ≥ 3) (h1 : a % c = 1) (h2 : b % c = 2) : (a * b) % c = 2 :=
by
  sorry

end remainder_of_product_l16_16524


namespace kelly_harvested_pounds_l16_16099

def total_carrots (bed1 bed2 bed3 : ℕ) : ℕ :=
  bed1 + bed2 + bed3

def total_weight (total : ℕ) (carrots_per_pound : ℕ) : ℕ :=
  total / carrots_per_pound

theorem kelly_harvested_pounds :
  total_carrots 55 101 78 = 234 ∧ total_weight 234 6 = 39 :=
by {
  split,
  { exact rfl }, -- 234 = 234
  { exact rfl }  -- 234 / 6 = 39
}

end kelly_harvested_pounds_l16_16099


namespace repetend_of_frac_4_div_17_is_235294_l16_16059

noncomputable def decimalRepetend_of_4_div_17 : String :=
  let frac := 4 / 17
  let repetend := "235294"
  repetend

theorem repetend_of_frac_4_div_17_is_235294 :
  (∃ n m : ℕ, (4 / 17 : ℚ) = n + (m / 10^6) ∧ m % 10^6 = 235294) :=
sorry

end repetend_of_frac_4_div_17_is_235294_l16_16059


namespace calculate_expression_l16_16200

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 = 1 / y^2) :
  (x^2 - 1 / x^2) * (y^2 + 1 / y^2) = x^4 - y^4 := by
  sorry

end calculate_expression_l16_16200


namespace solve_inequality_l16_16693

theorem solve_inequality (a : ℝ) : 
    (∀ x : ℝ, x^2 + (a + 2)*x + 2*a < 0 ↔ 
        (if a < 2 then -2 < x ∧ x < -a
         else if a = 2 then false
         else -a < x ∧ x < -2)) :=
by
  sorry

end solve_inequality_l16_16693


namespace no_intersect_x_axis_intersection_points_m_minus3_l16_16615

-- Define the quadratic function y = x^2 - 6x + 2m - 1
def quadratic_function (x m : ℝ) : ℝ := x^2 - 6 * x + 2 * m - 1

-- Theorem for Question 1: The function does not intersect the x-axis if and only if m > 5
theorem no_intersect_x_axis (m : ℝ) : (∀ x : ℝ, quadratic_function x m ≠ 0) ↔ m > 5 := sorry

-- Specific case when m = -3
def quadratic_function_m_minus3 (x : ℝ) : ℝ := x^2 - 6 * x - 7

-- Theorem for Question 2: Intersection points with coordinate axes for m = -3
theorem intersection_points_m_minus3 :
  ((∃ x : ℝ, quadratic_function_m_minus3 x = 0 ∧ (x = -1 ∨ x = 7)) ∧
   quadratic_function_m_minus3 0 = -7) := sorry

end no_intersect_x_axis_intersection_points_m_minus3_l16_16615


namespace office_expense_reduction_l16_16952

theorem office_expense_reduction (x : ℝ) (h : 0 ≤ x) (h' : x ≤ 1) : 
  2500 * (1 - x) ^ 2 = 1600 :=
sorry

end office_expense_reduction_l16_16952


namespace remainder_of_2_pow_2018_plus_1_mod_2018_l16_16473

theorem remainder_of_2_pow_2018_plus_1_mod_2018 : (2 ^ 2018 + 1) % 2018 = 2 := by
  sorry

end remainder_of_2_pow_2018_plus_1_mod_2018_l16_16473


namespace circle_through_and_tangent_l16_16013

noncomputable def circle_eq (a b r : ℝ) (x y : ℝ) : ℝ :=
  (x - a) ^ 2 + (y - b) ^ 2 - r ^ 2

theorem circle_through_and_tangent
(h1 : circle_eq 1 2 2 1 0 = 0)
(h2 : ∀ x y, circle_eq 1 2 2 x y = 0 → (x = 1 → y = 2 ∨ y = -2))
: ∀ x y, circle_eq 1 2 2 x y = 0 → (x - 1) ^ 2 + (y - 2) ^ 2 = 4 :=
by
  sorry

end circle_through_and_tangent_l16_16013


namespace tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l16_16404

def tight_sequence (a : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → (1/2 : ℚ) ≤ a (n + 1) / a n ∧ a (n + 1) / a n ≤ 2

def sum_of_first_n_terms (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)

noncomputable def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
∀ n : ℕ, n > 0 → a n = a 1 * q ^ (n - 1)

theorem tight_sequence_from_sum_of_terms (S : ℕ → ℚ) (a : ℕ → ℚ) : 
  (∀ n : ℕ, n > 0 → S n = (1 / 4) * (n^2 + 3 * n)) →
  (∀ n : ℕ, n > 0 → a n = S n - S (n - 1)) →
  tight_sequence a :=
sorry

theorem range_of_q_for_tight_sequences (a : ℕ → ℚ) (S : ℕ → ℚ) (q : ℚ) :
  geometric_sequence a q →
  tight_sequence a →
  tight_sequence S →
  (1 / 2 : ℚ) ≤ q ∧ q < 1 :=
sorry

end tight_sequence_from_sum_of_terms_range_of_q_for_tight_sequences_l16_16404


namespace double_seven_eighth_l16_16725

theorem double_seven_eighth (n : ℕ) (h : n = 48) : 2 * (7 / 8 * n) = 84 := by
  sorry

end double_seven_eighth_l16_16725


namespace deborah_oranges_zero_l16_16717

-- Definitions for given conditions.
def initial_oranges : Float := 55.0
def oranges_added_by_susan : Float := 35.0
def total_oranges_after : Float := 90.0

-- Defining Deborah's oranges in her bag.
def oranges_in_bag : Float := total_oranges_after - (initial_oranges + oranges_added_by_susan)

-- The theorem to be proved.
theorem deborah_oranges_zero : oranges_in_bag = 0 := by
  -- Placeholder for the proof.
  sorry

end deborah_oranges_zero_l16_16717


namespace final_score_is_correct_l16_16651

-- Definitions based on given conditions
def speechContentScore : ℕ := 90
def speechDeliveryScore : ℕ := 85
def weightContent : ℕ := 6
def weightDelivery : ℕ := 4

-- The final score calculation theorem
theorem final_score_is_correct : 
  (speechContentScore * weightContent + speechDeliveryScore * weightDelivery) / (weightContent + weightDelivery) = 88 :=
  by
    sorry

end final_score_is_correct_l16_16651


namespace fraction_of_capacity_l16_16876

theorem fraction_of_capacity
    (bus_capacity : ℕ)
    (x : ℕ)
    (first_pickup : ℕ)
    (second_pickup : ℕ)
    (unable_to_board : ℕ)
    (bus_full : bus_capacity = x + (second_pickup - unable_to_board))
    (carry_fraction : x / bus_capacity = 3 / 5) : 
    true := 
sorry

end fraction_of_capacity_l16_16876


namespace A_false_B_true_C_true_D_true_l16_16732

theorem A_false :
  ¬ ∃ x, ∀ y = (x^2 + 1) / x, y = 2 :=
by
  sorry

theorem B_true (x : ℝ) (h : x > 1) :
  (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4 * real.sqrt 2 + 1) :=
by
  sorry

theorem C_true (x y : ℝ) (h : x + 2 * y = 3 * x * y) (hx : 0 < x) (hy : 0 < y) :
  (2 * x + y ≥ 3) :=
by
  sorry

theorem D_true (x y : ℝ) (h : 9 * x^2 + y^2 + x * y = 1) :
  ∃ c, c = (3 * x + y) ∧ c ≤ (2 * real.sqrt 21 / 7) :=
by
  sorry

end A_false_B_true_C_true_D_true_l16_16732


namespace cost_of_first_10_kgs_of_apples_l16_16754

theorem cost_of_first_10_kgs_of_apples 
  (l q : ℝ) 
  (h1 : 30 * l + 3 * q = 663) 
  (h2 : 30 * l + 6 * q = 726) : 
  10 * l = 200 :=
by
  -- Proof would follow here
  sorry

end cost_of_first_10_kgs_of_apples_l16_16754


namespace average_of_rstu_l16_16634

theorem average_of_rstu (r s t u : ℝ) (h : (5 / 4) * (r + s + t + u) = 15) : (r + s + t + u) / 4 = 3 :=
by
  sorry

end average_of_rstu_l16_16634


namespace find_n_tan_eq_l16_16778

theorem find_n_tan_eq (n : ℤ) (h1 : -90 < n ∧ n < 90) (h2 : ∀ k : ℤ, 225 - 180 * k = 45) : n = 45 := by
  sorry

end find_n_tan_eq_l16_16778


namespace final_price_correct_l16_16408

noncomputable def price_cucumbers : ℝ := 5
noncomputable def price_tomatoes : ℝ := price_cucumbers - 0.20 * price_cucumbers
noncomputable def total_cost_before_discount : ℝ := 2 * price_tomatoes + 3 * price_cucumbers
noncomputable def discount : ℝ := 0.10 * total_cost_before_discount
noncomputable def final_price : ℝ := total_cost_before_discount - discount

theorem final_price_correct : final_price = 20.70 := by
  sorry

end final_price_correct_l16_16408


namespace bus_students_after_fifth_stop_l16_16946

theorem bus_students_after_fifth_stop :
  let initial := 72
  let firstStop := (2 / 3 : ℚ) * initial
  let secondStop := (2 / 3 : ℚ) * firstStop
  let thirdStop := (2 / 3 : ℚ) * secondStop
  let fourthStop := (2 / 3 : ℚ) * thirdStop
  let fifthStop := fourthStop + 12
  fifthStop = 236 / 9 :=
by
  sorry

end bus_students_after_fifth_stop_l16_16946


namespace max_statements_true_l16_16403

noncomputable def max_true_statements (a b : ℝ) : ℕ :=
  (if (a^2 > b^2) then 1 else 0) +
  (if (a < b) then 1 else 0) +
  (if (a < 0) then 1 else 0) +
  (if (b < 0) then 1 else 0) +
  (if (1 / a < 1 / b) then 1 else 0)

theorem max_statements_true : ∀ (a b : ℝ), max_true_statements a b ≤ 4 :=
by
  intro a b
  sorry

end max_statements_true_l16_16403


namespace anthony_total_pencils_l16_16753

theorem anthony_total_pencils :
  let original_pencils := 9
  let given_pencils := 56
  original_pencils + given_pencils = 65 := by
  sorry

end anthony_total_pencils_l16_16753


namespace janice_homework_time_l16_16391

variable (H : ℝ)
variable (cleaning_room walk_dog take_trash : ℝ)

-- Conditions from the problem translated directly
def cleaning_room_time : cleaning_room = H / 2 := sorry
def walk_dog_time : walk_dog = H + 5 := sorry
def take_trash_time : take_trash = H / 6 := sorry
def total_time_before_movie : 35 + (H + cleaning_room + walk_dog + take_trash) = 120 := sorry

-- The main theorem to prove
theorem janice_homework_time (H : ℝ)
        (cleaning_room : ℝ := H / 2)
        (walk_dog : ℝ := H + 5)
        (take_trash : ℝ := H / 6) :
    H + cleaning_room + walk_dog + take_trash + 35 = 120 → H = 30 :=
by
  sorry

end janice_homework_time_l16_16391


namespace geometric_seq_a8_l16_16714

noncomputable def geometric_seq_term (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n-1)

noncomputable def geometric_seq_sum (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r^n) / (1 - r)

theorem geometric_seq_a8
  (a₁ r : ℝ)
  (h1 : geometric_seq_sum a₁ r 3 = 7/4)
  (h2 : geometric_seq_sum a₁ r 6 = 63/4)
  (h3 : r ≠ 1) :
  geometric_seq_term a₁ r 8 = 32 :=
by
  sorry

end geometric_seq_a8_l16_16714


namespace abs_diff_of_two_numbers_l16_16288

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 160) : |x - y| = 2 * Real.sqrt 65 :=
by
  sorry

end abs_diff_of_two_numbers_l16_16288


namespace first_term_arith_seq_l16_16109

noncomputable def is_increasing (a b c : ℕ) (d : ℕ) : Prop := b = a + d ∧ c = a + 2 * d ∧ 0 < d

theorem first_term_arith_seq (a₁ a₂ a₃ : ℕ) (d: ℕ) :
  is_increasing a₁ a₂ a₃ d ∧ a₁ + a₂ + a₃ = 12 ∧ a₁ * a₂ * a₃ = 48 → a₁ = 2 := sorry

end first_term_arith_seq_l16_16109


namespace marbles_ratio_l16_16251

theorem marbles_ratio (miriam_current_marbles miriam_initial_marbles marbles_brother marbles_sister marbles_total_given marbles_savanna : ℕ)
  (h1 : miriam_current_marbles = 30)
  (h2 : marbles_brother = 60)
  (h3 : marbles_sister = 2 * marbles_brother)
  (h4 : miriam_initial_marbles = 300)
  (h5 : marbles_total_given = miriam_initial_marbles - miriam_current_marbles)
  (h6 : marbles_savanna = marbles_total_given - (marbles_brother + marbles_sister)) :
  (marbles_savanna : ℚ) / miriam_current_marbles = 3 :=
by
  sorry

end marbles_ratio_l16_16251


namespace heaviest_tv_l16_16584

theorem heaviest_tv :
  let area (width height : ℝ) := width * height
  let weight (area : ℝ) := area * 4
  let weight_in_pounds (weight : ℝ) := weight / 16
  let bill_area := area 48 100
  let bob_area := area 70 60
  let steve_area := area 84 92
  let bill_weight := weight bill_area
  let bob_weight := weight bob_area
  let steve_weight := weight steve_area
  let bill_weight_pounds := weight_in_pounds (weight bill_area)
  let bob_weight_pounds := weight_in_pounds (weight bob_area)
  let steve_weight_pounds := weight_in_pounds (weight steve_area)
  bob_weight_pounds + bill_weight_pounds < steve_weight_pounds
  ∧ abs ((steve_weight_pounds) - (bill_weight_pounds + bob_weight_pounds)) = 318 :=
by
  sorry

end heaviest_tv_l16_16584


namespace max_distance_on_ellipse_l16_16398

noncomputable def ellipse_parametric : (θ : ℝ) → ℝ × ℝ := λ θ, (√5 * Real.cos θ, Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def upper_vertex : ℝ × ℝ := (0, 1)

theorem max_distance_on_ellipse :
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi ∧ distance (ellipse_parametric θ) upper_vertex = 5 / 2 :=
sorry

end max_distance_on_ellipse_l16_16398


namespace problem_constant_term_binomial_l16_16210

open Real

theorem problem_constant_term_binomial :
  let a := ∫ x in 0..(π / 2), sin x + cos x
  let f := (a * (fun (x : ℝ) => x) - (fun (x : ℝ) => 1 / x))^6
  constant_term f = -160 :=
by
  sorry

end problem_constant_term_binomial_l16_16210


namespace range_of_m_l16_16222

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then (x - m) ^ 2 - 2 else 2 * x ^ 3 - 3 * x ^ 2

theorem range_of_m (m : ℝ) : (∃ x : ℝ, f x m = -1) ↔ m ≥ 1 :=
by
  sorry

end range_of_m_l16_16222


namespace length_of_faster_train_l16_16850

theorem length_of_faster_train
    (speed_faster : ℕ)
    (speed_slower : ℕ)
    (time_cross : ℕ)
    (h_fast : speed_faster = 72)
    (h_slow : speed_slower = 36)
    (h_time : time_cross = 15) :
    (speed_faster - speed_slower) * (1000 / 3600) * time_cross = 150 := 
by
  sorry

end length_of_faster_train_l16_16850


namespace at_least_one_gt_one_l16_16436

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : (x > 1) ∨ (y > 1) :=
sorry

end at_least_one_gt_one_l16_16436


namespace probability_two_face_cards_l16_16996

def cardDeck : ℕ := 52
def totalFaceCards : ℕ := 12

-- Probability of selecting one face card as the first card
def probabilityFirstFaceCard : ℚ := totalFaceCards / cardDeck

-- Probability of selecting another face card as the second card
def probabilitySecondFaceCard (cardsLeft : ℕ) : ℚ := (totalFaceCards - 1) / cardsLeft

-- Combined probability of selecting two face cards
theorem probability_two_face_cards :
  let combined_probability := probabilityFirstFaceCard * probabilitySecondFaceCard (cardDeck - 1)
  combined_probability = 22 / 442 := 
  by
    sorry

end probability_two_face_cards_l16_16996


namespace perfect_square_condition_l16_16677

theorem perfect_square_condition (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) : ∃ k : ℕ, a = k^2 :=
by
  sorry

end perfect_square_condition_l16_16677


namespace page_shoes_count_l16_16122

theorem page_shoes_count (p_i : ℕ) (d : ℝ) (b : ℕ) (h1 : p_i = 120) (h2 : d = 0.45) (h3 : b = 15) : 
  (p_i - (d * p_i)) + b = 81 :=
by
  sorry

end page_shoes_count_l16_16122


namespace least_number_of_equal_cubes_l16_16317

def cuboid_dimensions := (18, 27, 36)
def ratio := (1, 2, 3)

theorem least_number_of_equal_cubes :
  ∃ n, n = 648 ∧
  ∃ a b c : ℕ,
    (a, b, c) = (3, 6, 9) ∧
    (18 % a = 0 ∧ 27 % b = 0 ∧ 36 % c = 0) ∧
    18 * 27 * 36 = n * (a * b * c) :=
sorry

end least_number_of_equal_cubes_l16_16317


namespace unique_positive_x_eq_3_l16_16309

theorem unique_positive_x_eq_3 (x : ℝ) (h_pos : 0 < x) (h_eq : x + 17 = 60 * (1 / x)) : x = 3 :=
by
  sorry

end unique_positive_x_eq_3_l16_16309


namespace blueberries_in_blue_box_l16_16872

theorem blueberries_in_blue_box (B S : ℕ) (h1: S - B = 10) (h2 : 50 = S) : B = 40 := 
by
  sorry

end blueberries_in_blue_box_l16_16872


namespace simplify_complex_expr_l16_16537

noncomputable def z1 : ℂ := (-1 + complex.I * real.sqrt 3) / 2
noncomputable def z2 : ℂ := (-1 - complex.I * real.sqrt 3) / 2

theorem simplify_complex_expr :
  z1^12 + z2^12 = 2 := 
  sorry

end simplify_complex_expr_l16_16537


namespace fred_likes_12_pairs_of_digits_l16_16917

theorem fred_likes_12_pairs_of_digits :
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs ↔ ∃ (n : ℕ), n < 100 ∧ n % 8 = 0 ∧ n = 10 * a + b) ∧
    pairs.card = 12) :=
by
  sorry

end fred_likes_12_pairs_of_digits_l16_16917


namespace marvin_solved_yesterday_l16_16702

variables (M : ℕ)

def Marvin_yesterday := M
def Marvin_today := 3 * M
def Arvin_yesterday := 2 * M
def Arvin_today := 6 * M
def total_problems := Marvin_yesterday + Marvin_today + Arvin_yesterday + Arvin_today

theorem marvin_solved_yesterday :
  total_problems M = 480 → M = 40 :=
sorry

end marvin_solved_yesterday_l16_16702


namespace sum_of_areas_of_six_rectangles_eq_572_l16_16689

theorem sum_of_areas_of_six_rectangles_eq_572 :
  let lengths := [1, 3, 5, 7, 9, 11]
  let areas := lengths.map (λ x => 2 * x^2)
  areas.sum = 572 :=
by 
  sorry

end sum_of_areas_of_six_rectangles_eq_572_l16_16689


namespace perimeter_is_140_l16_16546

-- Definitions for conditions
def width (w : ℝ) := w
def length (w : ℝ) := width w + 10
def perimeter (w : ℝ) := 2 * (length w + width w)

-- Cost condition
def cost_condition (w : ℝ) : Prop := (perimeter w) * 6.5 = 910

-- Proving that if cost_condition holds, the perimeter is 140
theorem perimeter_is_140 (w : ℝ) (h : cost_condition w) : perimeter w = 140 :=
by sorry

end perimeter_is_140_l16_16546


namespace ball_bounces_to_C_l16_16755

/--
On a rectangular table with dimensions 9 cm in length and 7 cm in width, a small ball is shot from point A at a 45-degree angle. Upon reaching point E, it bounces off at a 45-degree angle and continues to roll forward. Throughout its motion, the ball bounces off the table edges at a 45-degree angle each time. Prove that, starting from point A, the ball first reaches point C after exactly 14 bounces.
-/
theorem ball_bounces_to_C (length width : ℝ) (angle : ℝ) (bounce_angle : ℝ) :
  length = 9 ∧ width = 7 ∧ angle = 45 ∧ bounce_angle = 45 → bounces_to_C = 14 :=
by
  intros
  sorry

end ball_bounces_to_C_l16_16755


namespace inequality_sqrt_sum_ge_2_l16_16522
open Real

theorem inequality_sqrt_sum_ge_2 {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  sqrt (a^3 / (1 + b * c)) + sqrt (b^3 / (1 + a * c)) + sqrt (c^3 / (1 + a * b)) ≥ 2 :=
by
  sorry

end inequality_sqrt_sum_ge_2_l16_16522


namespace price_of_food_before_tax_and_tip_l16_16575

noncomputable def actual_price_of_food (total_paid : ℝ) (tip_rate tax_rate : ℝ) : ℝ :=
  total_paid / (1 + tip_rate) / (1 + tax_rate)

theorem price_of_food_before_tax_and_tip :
  actual_price_of_food 211.20 0.20 0.10 = 160 :=
by
  sorry

end price_of_food_before_tax_and_tip_l16_16575


namespace angle_complement_supplement_l16_16600

theorem angle_complement_supplement (x : ℝ) (h : 90 - x = 3 / 4 * (180 - x)) : x = 180 :=
by
  sorry

end angle_complement_supplement_l16_16600


namespace domain_of_function_l16_16699

noncomputable def domain_f (x : ℝ) : Prop :=
  -x^2 + 2 * x + 3 > 0 ∧ 1 - x > 0 ∧ x ≠ 0

theorem domain_of_function :
  {x : ℝ | domain_f x} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end domain_of_function_l16_16699


namespace prob_equals_two_yellow_marbles_l16_16319

noncomputable def probability_two_yellow_marbles : ℚ :=
  let total_marbles : ℕ := 3 + 4 + 8
  let yellow_marbles : ℕ := 4
  let first_draw_prob : ℚ := yellow_marbles / total_marbles
  let second_total_marbles : ℕ := total_marbles - 1
  let second_yellow_marbles : ℕ := yellow_marbles - 1
  let second_draw_prob : ℚ := second_yellow_marbles / second_total_marbles
  first_draw_prob * second_draw_prob

theorem prob_equals_two_yellow_marbles :
  probability_two_yellow_marbles = 2 / 35 :=
by
  sorry

end prob_equals_two_yellow_marbles_l16_16319


namespace rectangular_solid_surface_area_l16_16906

theorem rectangular_solid_surface_area (a b c : ℕ) (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c) 
  (volume_eq : a * b * c = 273) :
  2 * (a * b + b * c + c * a) = 302 := 
sorry

end rectangular_solid_surface_area_l16_16906


namespace find_x2_plus_y2_l16_16941

noncomputable def xy : ℝ := 12
noncomputable def eq2 (x y : ℝ) : Prop := x^2 * y + x * y^2 + x + y = 120

theorem find_x2_plus_y2 (x y : ℝ) (h1 : xy = 12) (h2 : eq2 x y) : 
  x^2 + y^2 = 10344 / 169 :=
sorry

end find_x2_plus_y2_l16_16941


namespace irrational_number_among_choices_l16_16188

theorem irrational_number_among_choices : ∃ x ∈ ({17/6, -27/100, 0, Real.sqrt 2} : Set ℝ), Irrational x ∧ x = Real.sqrt 2 := by
  sorry

end irrational_number_among_choices_l16_16188


namespace girls_at_start_l16_16154

theorem girls_at_start (B G : ℕ) (h1 : B + G = 600) (h2 : 6 * B + 7 * G = 3840) : G = 240 :=
by
  -- actual proof is omitted
  sorry

end girls_at_start_l16_16154


namespace exists_k_for_inequality_l16_16927

noncomputable def C : ℕ := sorry -- C is a positive integer > 0
def a : ℕ → ℝ := sorry -- a sequence of positive real numbers

axiom C_pos : 0 < C
axiom a_pos : ∀ n : ℕ, 0 < a n
axiom recurrence_relation : ∀ n : ℕ, a (n + 1) = n / a n + C

theorem exists_k_for_inequality :
  ∃ k : ℕ, ∀ n : ℕ, n ≥ k → a (n + 2) > a n :=
  sorry

end exists_k_for_inequality_l16_16927


namespace car_average_speed_l16_16878

theorem car_average_speed (distance time : ℕ) (h1 : distance = 715) (h2 : time = 11) : distance / time = 65 := by
  sorry

end car_average_speed_l16_16878


namespace three_x_y_z_l16_16798

variable (x y z : ℝ)

def equation1 : Prop := y + z = 17 - 2 * x
def equation2 : Prop := x + z = -11 - 2 * y
def equation3 : Prop := x + y = 9 - 2 * z

theorem three_x_y_z : equation1 x y z ∧ equation2 x y z ∧ equation3 x y z → 3 * x + 3 * y + 3 * z = 45 / 4 :=
by
  intros h
  sorry

end three_x_y_z_l16_16798


namespace regular_dinosaur_weight_l16_16035

namespace DinosaurWeight

-- Given Conditions
def Barney_weight (x : ℝ) : ℝ := 5 * x + 1500
def combined_weight (x : ℝ) : ℝ := Barney_weight x + 5 * x

-- Target Proof
theorem regular_dinosaur_weight :
  (∃ x : ℝ, combined_weight x = 9500) -> 
  ∃ x : ℝ, x = 800 :=
by {
  sorry
}

end DinosaurWeight

end regular_dinosaur_weight_l16_16035


namespace units_digit_of_8_pow_2022_l16_16441

theorem units_digit_of_8_pow_2022 : (8 ^ 2022) % 10 = 4 := 
by
  -- We here would provide the proof of this theorem
  sorry

end units_digit_of_8_pow_2022_l16_16441


namespace box_base_length_max_l16_16704

noncomputable def V (x : ℝ) := x^2 * ((60 - x) / 2)

theorem box_base_length_max 
  (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 60)
  (h3 : ∀ y : ℝ, 0 < y ∧ y < 60 → V x ≥ V y)
  : x = 40 :=
sorry

end box_base_length_max_l16_16704


namespace complete_square_transformation_l16_16266

theorem complete_square_transformation : 
  ∀ (x : ℝ), (x^2 - 8 * x + 9 = 0) → ((x - 4)^2 = 7) :=
by
  intros x h
  sorry

end complete_square_transformation_l16_16266


namespace positive_difference_abs_eq_15_l16_16000

theorem positive_difference_abs_eq_15:
  ∃ x1 x2 : ℝ, (| x1 - 3 | = 15 ∧ | x2 - 3 | = 15) ∧ | x1 - x2 | = 30 :=
by
  sorry

end positive_difference_abs_eq_15_l16_16000


namespace cost_price_of_cloth_l16_16748

-- Definitions for conditions
def sellingPrice (totalMeters : ℕ) : ℕ := 8500
def profitPerMeter : ℕ := 15
def totalMeters : ℕ := 85

-- Proof statement with conditions and expected proof
theorem cost_price_of_cloth : 
  (sellingPrice totalMeters) = 8500 -> 
  profitPerMeter = 15 -> 
  totalMeters = 85 -> 
  (8500 - (profitPerMeter * totalMeters)) / totalMeters = 85 := 
by 
  sorry

end cost_price_of_cloth_l16_16748


namespace g_function_property_l16_16701

variable {g : ℝ → ℝ}
variable {a b : ℝ}

theorem g_function_property 
  (h1 : ∀ a c : ℝ, c^3 * g a = a^3 * g c)
  (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 :=
  sorry

end g_function_property_l16_16701


namespace find_value_of_f2_plus_g3_l16_16662

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem find_value_of_f2_plus_g3 : f (2 + g 3) = 37 :=
by
  simp [f, g]
  norm_num
  done

end find_value_of_f2_plus_g3_l16_16662


namespace g_at_neg1_l16_16930

-- Defining even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def is_odd (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x

-- Given functions f and g
variables (f g : ℝ → ℝ)

-- Given conditions
axiom f_even : is_even f
axiom g_odd : is_odd g
axiom fg_relation : ∀ x : ℝ, f x - g x = 2^(1 - x)

-- Proof statement
theorem g_at_neg1 : g (-1) = -3 / 2 :=
by
  sorry

end g_at_neg1_l16_16930


namespace find_m_of_quadratic_root_zero_l16_16359

theorem find_m_of_quadratic_root_zero (m : ℝ) (h : ∃ x, (m * x^2 + 5 * x + m^2 - 2 * m = 0) ∧ x = 0) : m = 2 :=
sorry

end find_m_of_quadratic_root_zero_l16_16359


namespace cubic_roots_l16_16345

theorem cubic_roots (a x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ = 1) (h₂ : x₂ = 1) (h₃ : x₃ = a)
  (cond : (2 / x₁) + (2 / x₂) = (3 / x₃)) :
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = a ∧ (a = 2 ∨ a = 3 / 4)) :=
by
  sorry

end cubic_roots_l16_16345


namespace probability_cs_majors_consecutive_l16_16849

def total_ways_to_choose_5_out_of_12 : ℕ :=
  Nat.choose 12 5

def number_of_ways_cs_majors_consecutive : ℕ :=
  12

theorem probability_cs_majors_consecutive :
  (number_of_ways_cs_majors_consecutive : ℚ) / (total_ways_to_choose_5_out_of_12 : ℚ) = 1 / 66 := by
  sorry

end probability_cs_majors_consecutive_l16_16849


namespace conner_ties_sydney_l16_16414

def sydney_initial_collect := 837
def conner_initial_collect := 723

def sydney_collect_day_one := 4
def conner_collect_day_one := 8 * sydney_collect_day_one / 2

def sydney_collect_day_two := (sydney_initial_collect + sydney_collect_day_one) - ((sydney_initial_collect + sydney_collect_day_one) / 10)
def conner_collect_day_two := conner_initial_collect + conner_collect_day_one + 123

def sydney_collect_day_three := sydney_collect_day_two + 2 * conner_collect_day_one
def conner_collect_day_three := (conner_collect_day_two - (123 / 4))

theorem conner_ties_sydney :
  sydney_collect_day_three <= conner_collect_day_three :=
by
  sorry

end conner_ties_sydney_l16_16414


namespace find_whole_number_M_l16_16329

theorem find_whole_number_M (M : ℕ) (h : 8 < M / 4 ∧ M / 4 < 9) : M = 33 :=
sorry

end find_whole_number_M_l16_16329


namespace fraction_comparison_and_differences_l16_16342

theorem fraction_comparison_and_differences :
  (1/3 < 0.5) ∧ (0.5 < 3/5) ∧ 
  (0.5 - 1/3 = 1/6) ∧ 
  (3/5 - 0.5 = 1/10) :=
by
  sorry

end fraction_comparison_and_differences_l16_16342


namespace Alma_test_score_l16_16843

-- Define the constants and conditions
variables (Alma_age Melina_age Alma_score : ℕ)

-- Conditions
axiom Melina_is_60 : Melina_age = 60
axiom Melina_3_times_Alma : Melina_age = 3 * Alma_age
axiom sum_ages_twice_score : Melina_age + Alma_age = 2 * Alma_score

-- Goal
theorem Alma_test_score : Alma_score = 40 :=
by
  sorry

end Alma_test_score_l16_16843


namespace smaller_two_digit_product_is_34_l16_16838

theorem smaller_two_digit_product_is_34 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 5082) : min a b = 34 :=
by
  sorry

end smaller_two_digit_product_is_34_l16_16838


namespace trigonometric_identity_proof_l16_16093

noncomputable def m : ℝ := 2 * Real.sin (Real.pi / 10)
noncomputable def n : ℝ := 4 - m^2

theorem trigonometric_identity_proof :
  (m = 2 * Real.sin (Real.pi / 10)) →
  (m^2 + n = 4) →
  (m * Real.sqrt n) / (2 * Real.cos (3 * Real.pi / 20)^2 - 1) = 2 :=
by
  intros h1 h2
  sorry

end trigonometric_identity_proof_l16_16093


namespace school_choir_robe_cost_l16_16324

theorem school_choir_robe_cost :
  ∀ (total_robes_needed current_robes cost_per_robe : ℕ), 
  total_robes_needed = 30 → 
  current_robes = 12 → 
  cost_per_robe = 2 → 
  (total_robes_needed - current_robes) * cost_per_robe = 36 :=
by
  intros total_robes_needed current_robes cost_per_robe h1 h2 h3
  sorry

end school_choir_robe_cost_l16_16324


namespace max_buses_in_city_l16_16641

-- Definition of the problem statement and the maximal number of buses
theorem max_buses_in_city (bus_stops : Finset ℕ) (buses : Finset (Finset ℕ)) 
  (h_stops : bus_stops.card = 9) 
  (h_buses_stops : ∀ b ∈ buses, b.card = 3) 
  (h_shared_stops : ∀ b₁ b₂ ∈ buses, b₁ ≠ b₂ → (b₁ ∩ b₂).card ≤ 1) : 
  buses.card ≤ 12 := 
sorry

end max_buses_in_city_l16_16641


namespace car_b_speed_l16_16296

theorem car_b_speed
  (v_A v_B : ℝ) (d_A d_B d : ℝ)
  (h1 : v_A = 5 / 3 * v_B)
  (h2 : d_A = v_A * 5)
  (h3 : d_B = v_B * 5)
  (h4 : d = d_A + d_B)
  (h5 : d_A = d / 2 + 25) :
  v_B = 15 := 
sorry

end car_b_speed_l16_16296


namespace number_of_students_l16_16544

theorem number_of_students (avg_age_students : ℕ) (teacher_age : ℕ) (new_avg_age : ℕ) (n : ℕ) (T : ℕ) 
    (h1 : avg_age_students = 10) (h2 : teacher_age = 26) (h3 : new_avg_age = 11)
    (h4 : T = n * avg_age_students) 
    (h5 : (T + teacher_age) / (n + 1) = new_avg_age) : n = 15 :=
by
  -- Proof should go here
  sorry

end number_of_students_l16_16544


namespace eggs_remainder_l16_16485

def daniel_eggs := 53
def eliza_eggs := 68
def fiona_eggs := 26
def george_eggs := 47
def total_eggs := daniel_eggs + eliza_eggs + fiona_eggs + george_eggs

theorem eggs_remainder :
  total_eggs % 15 = 14 :=
by
  sorry

end eggs_remainder_l16_16485


namespace sector_radius_l16_16028

theorem sector_radius (θ : ℝ) (s : ℝ) (R : ℝ) 
  (hθ : θ = 150)
  (hs : s = (5 / 2) * Real.pi)
  : (θ / 360) * (2 * Real.pi * R) = (5 / 2) * Real.pi → 
  R = 3 := 
sorry

end sector_radius_l16_16028


namespace rectangle_area_l16_16027

theorem rectangle_area (w l : ℝ) (hw : w = 2) (hl : l = 3) : w * l = 6 := by
  sorry

end rectangle_area_l16_16027


namespace roots_product_eq_three_l16_16112

theorem roots_product_eq_three
  (p q r : ℝ)
  (h : (3:ℝ) * p ^ 3 - 8 * p ^ 2 + p - 9 = 0 ∧
       (3:ℝ) * q ^ 3 - 8 * q ^ 2 + q - 9 = 0 ∧
       (3:ℝ) * r ^ 3 - 8 * r ^ 2 + r - 9 = 0) :
  p * q * r = 3 :=
sorry

end roots_product_eq_three_l16_16112


namespace sin_A_over_1_minus_cos_A_l16_16360

variable {a b c : ℝ} -- Side lengths of the triangle
variable {A B C : ℝ} -- Angles opposite to the sides

theorem sin_A_over_1_minus_cos_A 
  (h_area : 0.5 * b * c * Real.sin A = a^2 - (b - c)^2) :
  Real.sin A / (1 - Real.cos A) = 3 :=
sorry

end sin_A_over_1_minus_cos_A_l16_16360


namespace range_of_a_l16_16919

variable (x a : ℝ)

-- Definition of α: x > a
def α : Prop := x > a

-- Definition of β: (x - 1) / x > 0
def β : Prop := (x - 1) / x > 0

-- Theorem to prove the range of a
theorem range_of_a (h : α x a → β x) : 1 ≤ a :=
  sorry

end range_of_a_l16_16919


namespace even_function_value_sum_l16_16525

noncomputable def g (x : ℝ) (d e f : ℝ) : ℝ :=
  d * x^8 - e * x^6 + f * x^2 + 5

theorem even_function_value_sum (d e f : ℝ) (h : g 15 d e f = 7) :
  g 15 d e f + g (-15) d e f = 14 := by
  sorry

end even_function_value_sum_l16_16525


namespace youngest_son_trips_l16_16972

theorem youngest_son_trips 
  (p : ℝ) (n_oldest : ℝ) (c : ℝ) (Y : ℝ)
  (h1 : p = 100)
  (h2 : n_oldest = 35)
  (h3 : c = 4)
  (h4 : p / c = Y) :
  Y = 25 := sorry

end youngest_son_trips_l16_16972


namespace parallelogram_perimeter_eq_60_l16_16638

-- Given conditions from the problem
variables (P Q R M N O : Type*)
variables (PQ PR QR PM MN NO PO : ℝ)
variables {PQ_eq_PR : PQ = PR}
variables {PQ_val : PQ = 30}
variables {PR_val : PR = 30}
variables {QR_val : QR = 28}
variables {MN_parallel_PR : true}  -- Parallel condition we can treat as true for simplification
variables {NO_parallel_PQ : true}  -- Another parallel condition treated as true

-- Statement of the problem to be proved
theorem parallelogram_perimeter_eq_60 :
  PM + MN + NO + PO = 60 :=
sorry

end parallelogram_perimeter_eq_60_l16_16638


namespace pole_length_after_cut_l16_16026

theorem pole_length_after_cut (original_length : ℝ) (percentage_retained : ℝ) : 
  original_length = 20 → percentage_retained = 0.7 → 
  original_length * percentage_retained = 14 :=
by
  intros h0 h1
  rw [h0, h1]
  norm_num

end pole_length_after_cut_l16_16026


namespace conditional_prob_correct_l16_16723

/-- Define the events A and B as per the problem -/
def event_A (x y : ℕ) : Prop := (x + y) % 2 = 0

def event_B (x y : ℕ) : Prop := (x % 2 = 0 ∨ y % 2 = 0) ∧ x ≠ y

/-- Define the probability of event A -/
def prob_A : ℚ := 1 / 2

/-- Define the combined probability of both events A and B occurring -/
def prob_A_and_B : ℚ := 1 / 6

/-- Calculate the conditional probability P(B | A) -/
def conditional_prob : ℚ := prob_A_and_B / prob_A

theorem conditional_prob_correct : conditional_prob = 1 / 3 := by
  -- This is where you would provide the proof if required
  sorry

end conditional_prob_correct_l16_16723


namespace base9_problem_l16_16580

def base9_add (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual addition for base 9
def base9_mul (a : ℕ) (b : ℕ) : ℕ := sorry -- Define actual multiplication for base 9

theorem base9_problem : base9_mul (base9_add 35 273) 2 = 620 := sorry

end base9_problem_l16_16580


namespace players_per_group_l16_16705

-- Definitions for given conditions
def num_new_players : Nat := 48
def num_returning_players : Nat := 6
def num_groups : Nat := 9

-- Proof that the number of players in each group is 6
theorem players_per_group :
  let total_players := num_new_players + num_returning_players
  total_players / num_groups = 6 := by
  sorry

end players_per_group_l16_16705


namespace tan_angle_add_l16_16802

theorem tan_angle_add (x : ℝ) (h : Real.tan x = -3) : Real.tan (x + Real.pi / 6) = 2 * Real.sqrt 3 + 1 := 
by
  sorry

end tan_angle_add_l16_16802


namespace probability_even_sum_l16_16801

theorem probability_even_sum (x y : ℕ) (h : x + y ≤ 10) : 
  (∃ (p : ℚ), p = 6 / 11 ∧ (x + y) % 2 = 0) :=
sorry

end probability_even_sum_l16_16801


namespace unique_a_value_l16_16076

theorem unique_a_value (a : ℝ) :
  let M := { x : ℝ | x^2 = 2 }
  let N := { x : ℝ | a * x = 1 }
  N ⊆ M ↔ (a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2) :=
by
  sorry

end unique_a_value_l16_16076


namespace max_distance_B_P_l16_16400

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  (Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem max_distance_B_P : 
  let B : ℝ × ℝ := (0, 1)
  let ellipse (P : ℝ × ℝ) := (P.1^2) / 5 + P.2^2 = 1
  ∀ (P : ℝ × ℝ), ellipse P → distance P.1 P.2 B.1 B.2 ≤ 5 / 2 :=
begin
  sorry
end

end max_distance_B_P_l16_16400


namespace Cl_invalid_electrons_l16_16562

noncomputable def Cl_mass_number : ℕ := 35
noncomputable def Cl_protons : ℕ := 17
noncomputable def Cl_neutrons : ℕ := Cl_mass_number - Cl_protons
noncomputable def Cl_electrons : ℕ := Cl_protons

theorem Cl_invalid_electrons : Cl_electrons ≠ 18 :=
by
  sorry

end Cl_invalid_electrons_l16_16562


namespace democrats_ratio_l16_16291

variable (F M D_F D_M TotalParticipants : ℕ)

-- Assume the following conditions
variables (H1 : F + M = 660)
variables (H2 : D_F = 1 / 2 * F)
variables (H3 : D_F = 110)
variables (H4 : D_M = 1 / 4 * M)
variables (H5 : TotalParticipants = 660)

theorem democrats_ratio 
  (H1 : F + M = 660)
  (H2 : D_F = 1 / 2 * F)
  (H3 : D_F = 110)
  (H4 : D_M = 1 / 4 * M)
  (H5 : TotalParticipants = 660) :
  (D_F + D_M) / TotalParticipants = 1 / 3
:= 
  sorry

end democrats_ratio_l16_16291


namespace ring_roads_count_l16_16189

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem ring_roads_count : 
  binomial 8 4 * binomial 8 4 - (binomial 10 4 * binomial 6 4) = 1750 := by 
sorry

end ring_roads_count_l16_16189


namespace minimum_value_of_f_l16_16779

noncomputable def f (x : ℝ) : ℝ := 2 * x + (3 * x) / (x^2 + 3) + (2 * x * (x + 5)) / (x^2 + 5) + (3 * (x + 3)) / (x * (x^2 + 5))

theorem minimum_value_of_f : ∃ a : ℝ, a > 0 ∧ (∀ x > 0, f x ≥ 7) ∧ (f a = 7) :=
by
  sorry

end minimum_value_of_f_l16_16779


namespace train_crosses_platform_in_26_seconds_l16_16456

def km_per_hr_to_m_per_s (km_per_hr : ℕ) : ℕ :=
  km_per_hr * 5 / 18

def train_crossing_time
  (train_speed_km_per_hr : ℕ)
  (train_length_m : ℕ)
  (platform_length_m : ℕ) : ℕ :=
  let total_distance_m := train_length_m + platform_length_m
  let train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr
  total_distance_m / train_speed_m_per_s

theorem train_crosses_platform_in_26_seconds :
  train_crossing_time 72 300 220 = 26 :=
by
  sorry

end train_crosses_platform_in_26_seconds_l16_16456


namespace nat_perfect_square_l16_16678

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l16_16678


namespace probability_exactly_three_between_ivanov_and_petrov_l16_16743

open Finset

def num_people := 11

def arrangements (n : ℕ) := (n - 1)!

def event_A (pos_Ivanov pos_Petrov : ℕ) : Prop :=
(pos_Petrov = (pos_Ivanov + 4) % num_people) ∨
(pos_Petrov = (pos_Ivanov + 7) % num_people)

def favorable_outcomes (n : ℕ) : ℕ := 
let num_positions := num_people in
let ivanov_fixed := 1 in
let choices_for_petrov := 2 in
choices_for_petrov * arrangements (n - 2)

def total_possibilities (n : ℕ) :=
arrangements n

theorem probability_exactly_three_between_ivanov_and_petrov :
let n := num_people in
let favorable := favorable_outcomes n in
let total := total_possibilities n in
(n > 1) →
(favorable.to_rat / total.to_rat) = 1 / 10 :=
begin
  intros,
  sorry
end

end probability_exactly_three_between_ivanov_and_petrov_l16_16743


namespace fixed_point_exists_line_intersects_circle_shortest_chord_l16_16511

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25
noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem fixed_point_exists : ∃ P : ℝ × ℝ, (∀ m : ℝ, line_l P.1 P.2 m) ∧ P = (3, 1) :=
by
  sorry

theorem line_intersects_circle : ∀ m : ℝ, ∃ x₁ y₁ x₂ y₂ : ℝ, line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by
  sorry

theorem shortest_chord : ∃ m : ℝ, m = -3/4 ∧ (∀ x y, line_l x y m ↔ 2 * x - y - 5 = 0) :=
by
  sorry

end fixed_point_exists_line_intersects_circle_shortest_chord_l16_16511


namespace value_of_nested_custom_div_l16_16192

def custom_div (x y z : ℕ) (hz : z ≠ 0) : ℕ :=
  (x + y) / z

theorem value_of_nested_custom_div : custom_div (custom_div 45 15 60 (by decide)) (custom_div 3 3 6 (by decide)) (custom_div 20 10 30 (by decide)) (by decide) = 2 :=
sorry

end value_of_nested_custom_div_l16_16192


namespace option_b_option_c_option_d_l16_16731

theorem option_b (x : ℝ) (h : x > 1) : (∀ y, y = 2*x + 4 / (x - 1) - 1 → y ≥ 4*Real.sqrt 2 + 1) :=
by
  sorry

theorem option_c (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3 * x * y) : 2*x + y ≥ 3 :=
by
  sorry

theorem option_d (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) : 3*x + y ≤ 2*Real.sqrt 21 / 7 :=
by
  sorry

end option_b_option_c_option_d_l16_16731


namespace radical_center_on_euler_line_l16_16818

open EuclideanGeometry

-- Definitions of relevant points and properties
variables {ABC : Triangle} (G : Point)
variables (A B C D E F P Q R : Point)

-- Assumptions
variables (hG_centroid : Centroid G ABC)
variables (hA_circumcircle : Meets AG (Circumcircle ABC) P)
variables (hB_circumcircle : Meets BG (Circumcircle ABC) Q)
variables (hC_circumcircle : Meets CG (Circumcircle ABC) R)
variables (hAD_altitude : Altitude AD ABC)
variables (hBE_altitude : Altitude BE ABC)
variables (hCF_altitude : Altitude CF ABC)

theorem radical_center_on_euler_line :
  LiesOn (RadicalCenter (Circumscribed (DQR)) (Circumscribed (EPR)) (Circumscribed (FPQ))) (EulerLine ABC) :=
sorry

end radical_center_on_euler_line_l16_16818


namespace martha_saving_l16_16248

-- Definitions for the conditions
def daily_allowance : ℕ := 12
def half_daily_allowance : ℕ := daily_allowance / 2
def quarter_daily_allowance : ℕ := daily_allowance / 4
def days_saving_half : ℕ := 6
def day_saving_quarter : ℕ := 1

-- Statement to be proved
theorem martha_saving :
  (days_saving_half * half_daily_allowance) + (day_saving_quarter * quarter_daily_allowance) = 39 := by
  sorry

end martha_saving_l16_16248


namespace John_scored_24point5_goals_l16_16348

theorem John_scored_24point5_goals (T G : ℝ) (n : ℕ) (A : ℝ)
  (h1 : T = 65)
  (h2 : n = 9)
  (h3 : A = 4.5) :
  G = T - (n * A) :=
by
  sorry

end John_scored_24point5_goals_l16_16348


namespace initial_mean_correctness_l16_16547

variable (M : ℝ)

theorem initial_mean_correctness (h1 : 50 * M + 20 = 50 * 36.5) : M = 36.1 :=
by 
  sorry

end initial_mean_correctness_l16_16547


namespace library_students_l16_16890

theorem library_students (total_books : ℕ) (books_per_student : ℕ) (students_day1 : ℕ) (students_day2 : ℕ) (students_day3 : ℕ) :
  total_books = 120 →
  books_per_student = 5 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day3 = 6 →
  let books_used_day1 := students_day1 * books_per_student in
  let books_used_day2 := students_day2 * books_per_student in
  let books_used_day3 := students_day3 * books_per_student in
  let total_books_used := books_used_day1 + books_used_day2 + books_used_day3 in
  let remaining_books := total_books - total_books_used in
  remaining_books / books_per_student = 9 :=
by
  intros h1 h2 h3 h4 h5
  simp [h1, h2, h3, h4, h5]
  sorry

end library_students_l16_16890


namespace select_two_integers_divisibility_l16_16402

open Polynomial

theorem select_two_integers_divisibility
  (F : Polynomial ℤ)
  (m : ℕ)
  (a : Fin m → ℤ)
  (H : ∀ n : ℤ, ∃ i : Fin m, a i ∣ F.eval n) :
  ∃ i j : Fin m, i ≠ j ∧ ∀ n : ℤ, ∃ k : Fin m, k = i ∨ k = j ∧ a k ∣ F.eval n :=
by
  sorry

end select_two_integers_divisibility_l16_16402


namespace set_inter_complement_l16_16509

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem set_inter_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  sorry

end set_inter_complement_l16_16509


namespace prime_ratio_sum_l16_16834

theorem prime_ratio_sum (p q m : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
(h_roots : ∀ x : ℝ, x^2 - 99 * x + m = 0 → x = p ∨ x = q) :
  (p : ℚ) / q + q / p = 9413 / 194 :=
sorry

end prime_ratio_sum_l16_16834


namespace probability_single_draws_probability_two_different_colors_l16_16292

-- Define probabilities for black, yellow and green as events A, B, and C respectively.
variables (A B C : ℝ)

-- Conditions based on the problem statement
axiom h1 : A + B = 5/9
axiom h2 : B + C = 2/3
axiom h3 : A + B + C = 1

-- Here is the statement to prove the calculated probabilities of single draws
theorem probability_single_draws : 
  A = 1/3 ∧ B = 2/9 ∧ C = 4/9 :=
sorry

-- Define the event of drawing two balls of the same color
variables (black yellow green : ℕ)
axiom balls_count : black + yellow + green = 9
axiom black_component : A = black / 9
axiom yellow_component : B = yellow / 9
axiom green_component : C = green / 9

-- Using the counts to infer the probability of drawing two balls of different colors
axiom h4 : black = 3
axiom h5 : yellow = 2
axiom h6 : green = 4

theorem probability_two_different_colors :
  (1 - (3/36 + 1/36 + 6/36)) = 13/18 :=
sorry

end probability_single_draws_probability_two_different_colors_l16_16292


namespace sum_squares_of_solutions_eq_l16_16604

noncomputable def sum_of_squares_of_solutions : ℚ := sorry

theorem sum_squares_of_solutions_eq :
  (∃ x : ℚ, abs (x^2 - x + (1 : ℚ) / 2010) = (1 : ℚ) / 2010) →
  sum_of_squares_of_solutions = (2008 : ℚ) / 1005 :=
sorry

end sum_squares_of_solutions_eq_l16_16604


namespace phase_shift_correct_l16_16057

-- Given the function y = 3 * sin (x - π / 5)
-- We need to prove that the phase shift is π / 5.

theorem phase_shift_correct :
  ∀ x : ℝ, 3 * Real.sin (x - Real.pi / 5) = 3 * Real.sin (x - C) →
  C = Real.pi / 5 :=
by
  sorry

end phase_shift_correct_l16_16057


namespace stable_scores_l16_16540

theorem stable_scores (S_A S_B S_C S_D : ℝ) (hA : S_A = 2.2) (hB : S_B = 6.6) (hC : S_C = 7.4) (hD : S_D = 10.8) : 
  S_A ≤ S_B ∧ S_A ≤ S_C ∧ S_A ≤ S_D :=
by
  sorry

end stable_scores_l16_16540


namespace proof_problem_l16_16191

-- Define the operation table as a function in Lean 4
def op (a b : ℕ) : ℕ :=
  if a = 1 then
    if b = 1 then 2 else if b = 2 then 1 else if b = 3 then 4 else 3
  else if a = 2 then
    if b = 1 then 1 else if b = 2 then 3 else if b = 3 then 2 else 4
  else if a = 3 then
    if b = 1 then 4 else if b = 2 then 2 else if b = 3 then 1 else 3
  else
    if b = 1 then 3 else if b = 2 then 4 else if b = 3 then 3 else 2

-- State the theorem to prove
theorem proof_problem : op (op 3 1) (op 4 2) = 2 :=
by
  sorry

end proof_problem_l16_16191


namespace candy_eaten_l16_16591

theorem candy_eaten 
  {initial_pieces remaining_pieces eaten_pieces : ℕ} 
  (h₁ : initial_pieces = 12) 
  (h₂ : remaining_pieces = 3) 
  (h₃ : eaten_pieces = initial_pieces - remaining_pieces) 
  : eaten_pieces = 9 := 
by 
  sorry

end candy_eaten_l16_16591


namespace sandy_correct_sums_l16_16263

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 55) : c = 23 :=
by
  sorry

end sandy_correct_sums_l16_16263


namespace grace_wins_probability_l16_16795

def probability_grace_wins : ℚ :=
  let total_possible_outcomes := 36
  let losing_combinations := 6
  let winning_combinations := total_possible_outcomes - losing_combinations
  winning_combinations / total_possible_outcomes

theorem grace_wins_probability :
    probability_grace_wins = 5 / 6 := by
  sorry

end grace_wins_probability_l16_16795


namespace ratio_of_volumes_l16_16560

-- Define the edge lengths
def edge_length_cube1 : ℝ := 9
def edge_length_cube2 : ℝ := 24

-- Theorem stating the ratio of the volumes
theorem ratio_of_volumes :
  (edge_length_cube1 / edge_length_cube2) ^ 3 = 27 / 512 :=
by
  sorry

end ratio_of_volumes_l16_16560


namespace total_cost_l16_16814

def cost_burger := 5
def cost_sandwich := 4
def cost_smoothie := 4
def count_smoothies := 2

theorem total_cost :
  cost_burger + cost_sandwich + count_smoothies * cost_smoothie = 17 :=
by
  sorry

end total_cost_l16_16814


namespace marks_deducted_per_wrong_answer_l16_16517

theorem marks_deducted_per_wrong_answer
  (correct_awarded : ℕ)
  (total_marks : ℕ)
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (final_marks : ℕ) :
  correct_awarded = 3 →
  total_marks = 38 →
  total_questions = 70 →
  correct_answers = 27 →
  incorrect_answers = total_questions - correct_answers →
  final_marks = total_marks →
  final_marks = correct_answers * correct_awarded - incorrect_answers * 1 →
  1 = 1
  := by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end marks_deducted_per_wrong_answer_l16_16517


namespace max_min_x_plus_y_on_circle_l16_16387

-- Define the conditions
def polar_eq (ρ θ : Real) : Prop :=
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) + 6 = 0

-- Define the standard form of the circle
def circle_eq (x y : Real) : Prop :=
  (x - 2)^2 + (y - 2)^2 = 2

-- Define the parametric equations of the circle
def parametric_eq (α : Real) (x y : Real) : Prop :=
  x = 2 + Real.sqrt 2 * Real.cos α ∧ y = 2 + Real.sqrt 2 * Real.sin α

-- Define the problem in Lean
theorem max_min_x_plus_y_on_circle :
  (∀ (ρ θ : Real), polar_eq ρ θ → circle_eq (ρ * Real.cos θ) (ρ * Real.sin θ)) →
  (∀ (α : Real), parametric_eq α (2 + Real.sqrt 2 * Real.cos α) (2 + Real.sqrt 2 * Real.sin α)) →
  (∀ (P : Real × Real), circle_eq P.1 P.2 → 2 ≤ P.1 + P.2 ∧ P.1 + P.2 ≤ 6) :=
by
  intros hpolar hparam P hcircle
  sorry

end max_min_x_plus_y_on_circle_l16_16387


namespace percentage_of_fair_haired_employees_who_are_women_l16_16875

variable (E : ℝ) -- Total number of employees
variable (h1 : 0.1 * E = women_with_fair_hair_E) -- 10% of employees are women with fair hair
variable (h2 : 0.25 * E = fair_haired_employees_E) -- 25% of employees have fair hair

theorem percentage_of_fair_haired_employees_who_are_women :
  (women_with_fair_hair_E / fair_haired_employees_E) * 100 = 40 :=
by
  sorry

end percentage_of_fair_haired_employees_who_are_women_l16_16875


namespace find_fifth_score_l16_16392

-- Define the known scores
def score1 : ℕ := 90
def score2 : ℕ := 93
def score3 : ℕ := 85
def score4 : ℕ := 97

-- Define the average of all scores
def average : ℕ := 92

-- Define the total number of scores
def total_scores : ℕ := 5

-- Define the total sum of all scores using the average
def total_sum : ℕ := total_scores * average

-- Define the sum of the four known scores
def known_sum : ℕ := score1 + score2 + score3 + score4

-- Define the fifth score
def fifth_score : ℕ := 95

-- Theorem statement: The fifth score plus the known sum equals the total sum.
theorem find_fifth_score : fifth_score + known_sum = total_sum := by
  sorry

end find_fifth_score_l16_16392


namespace second_interest_rate_l16_16411

theorem second_interest_rate (P1 P2 : ℝ) (r : ℝ) (total_amount total_income: ℝ) (h1 : total_amount = 2500)
  (h2 : P1 = 1500.0000000000007) (h3 : total_income = 135) :
  P2 = total_amount - P1 →
  P1 * 0.05 = 75 →
  P2 * r = 60 →
  r = 0.06 :=
sorry

end second_interest_rate_l16_16411


namespace locus_of_intersection_l16_16203

theorem locus_of_intersection (m : ℝ) :
  (∃ x y : ℝ, (m * x - y + 1 = 0) ∧ (x - m * y - 1 = 0)) ↔ (∃ x y : ℝ, (x - y = 0) ∨ (x - y + 1 = 0)) :=
by
  sorry

end locus_of_intersection_l16_16203


namespace increasing_function_range_l16_16621

theorem increasing_function_range (a : ℝ) (f : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = x^3 - a * x - 1) :
  (∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) ↔ a ≤ 0 :=
sorry

end increasing_function_range_l16_16621


namespace roots_equal_of_quadratic_eq_zero_l16_16916

theorem roots_equal_of_quadratic_eq_zero (a : ℝ) :
  (∃ x : ℝ, (x^2 - a*x + 1) = 0 ∧ (∀ y : ℝ, (y^2 - a*y + 1) = 0 → y = x)) → (a = 2 ∨ a = -2) :=
by
  sorry

end roots_equal_of_quadratic_eq_zero_l16_16916


namespace find_m_of_ellipse_l16_16623

theorem find_m_of_ellipse (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (10 - m) + y^2 / (m - 2) = 1) ∧ (m - 2 > 10 - m) ∧ ((4)^2 = (m - 2) - (10 - m))) → m = 14 :=
by sorry

end find_m_of_ellipse_l16_16623


namespace new_line_length_l16_16593

/-- Eli drew a line that was 1.5 meters long and then erased 37.5 centimeters of it.
    We need to prove that the length of the line now is 112.5 centimeters. -/
theorem new_line_length (initial_length_m : ℝ) (erased_length_cm : ℝ) 
    (h1 : initial_length_m = 1.5) (h2 : erased_length_cm = 37.5) :
    initial_length_m * 100 - erased_length_cm = 112.5 :=
by
  sorry

end new_line_length_l16_16593


namespace find_cement_used_lexi_l16_16123

def cement_used_total : ℝ := 15.1
def cement_used_tess : ℝ := 5.1
def cement_used_lexi : ℝ := cement_used_total - cement_used_tess

theorem find_cement_used_lexi : cement_used_lexi = 10 := by
  sorry

end find_cement_used_lexi_l16_16123


namespace mixed_fruit_juice_litres_opened_l16_16278

theorem mixed_fruit_juice_litres_opened (cocktail_cost_per_litre : ℝ)
  (mixed_juice_cost_per_litre : ℝ) (acai_cost_per_litre : ℝ)
  (acai_litres_added : ℝ) (total_mixed_juice_opened : ℝ) :
  cocktail_cost_per_litre = 1399.45 ∧
  mixed_juice_cost_per_litre = 262.85 ∧
  acai_cost_per_litre = 3104.35 ∧
  acai_litres_added = 23.333333333333336 ∧
  (mixed_juice_cost_per_litre * total_mixed_juice_opened + 
  acai_cost_per_litre * acai_litres_added = 
  cocktail_cost_per_litre * (total_mixed_juice_opened + acai_litres_added)) →
  total_mixed_juice_opened = 35 :=
sorry

end mixed_fruit_juice_litres_opened_l16_16278


namespace square_area_from_circles_l16_16356

theorem square_area_from_circles :
  (∀ (r : ℝ), r = 7 → ∀ (n : ℕ), n = 4 → (∃ (side_length : ℝ), side_length = 2 * (2 * r))) →
  ∀ (side_length : ℝ), side_length = 28 →
  (∃ (area : ℝ), area = side_length * side_length ∧ area = 784) :=
sorry

end square_area_from_circles_l16_16356


namespace gcd_of_products_l16_16832

theorem gcd_of_products (a b a' b' d d' : ℕ) (h1 : Nat.gcd a b = d) (h2 : Nat.gcd a' b' = d') (ha : 0 < a) (hb : 0 < b) (ha' : 0 < a') (hb' : 0 < b') :
  Nat.gcd (Nat.gcd (aa') (ab')) (Nat.gcd (ba') (bb')) = d * d' := 
sorry

end gcd_of_products_l16_16832


namespace least_possible_area_of_square_l16_16147

theorem least_possible_area_of_square :
  (∃ (side_length : ℝ), 3.5 ≤ side_length ∧ side_length < 4.5 ∧ 
    (∃ (area : ℝ), area = side_length * side_length ∧ 
    (∀ (side : ℝ), 3.5 ≤ side ∧ side < 4.5 → side * side ≥ 12.25))) :=
sorry

end least_possible_area_of_square_l16_16147


namespace sum_of_squares_of_solutions_l16_16605

theorem sum_of_squares_of_solutions :
  (∑ x in {x : ℝ | | x^2 - x + 1/2010 | = 1/2010}, x^2) = 2008 / 1005 :=
by
  sorry

end sum_of_squares_of_solutions_l16_16605


namespace travel_ways_l16_16152

theorem travel_ways (buses : Nat) (trains : Nat) (boats : Nat) 
  (hb : buses = 5) (ht : trains = 6) (hb2 : boats = 2) : 
  buses + trains + boats = 13 := by
  sorry

end travel_ways_l16_16152


namespace arithmetic_mean_of_multiples_l16_16861

-- Define the given conditions
def smallest_multiple : ℕ := 16
def largest_multiple : ℕ := 96
def common_difference : ℕ := 8
def number_of_terms : ℕ := 11

-- Define the statement we want to prove
theorem arithmetic_mean_of_multiples : 
  (smallest_multiple + largest_multiple) / 2 = 56 := 
begin
  sorry
end

end arithmetic_mean_of_multiples_l16_16861


namespace total_expenditure_l16_16326

-- Define the conditions.
def singers : ℕ := 30
def current_robes : ℕ := 12
def robe_cost : ℕ := 2

-- Define the statement.
theorem total_expenditure (singers current_robes robe_cost : ℕ) : 
  (singers - current_robes) * robe_cost = 36 := by
  sorry

end total_expenditure_l16_16326


namespace a_is_perfect_square_l16_16671

theorem a_is_perfect_square (a b : ℕ) (h : ∃ (k : ℕ), a^2 + b^2 + a = k * a * b) : ∃ n : ℕ, a = n^2 := by
  sorry

end a_is_perfect_square_l16_16671


namespace correct_divisor_l16_16649

noncomputable def dividend := 12 * 35

theorem correct_divisor (x : ℕ) : (x * 20 = dividend) → x = 21 :=
sorry

end correct_divisor_l16_16649


namespace lambda_inequality_l16_16828

-- Define the problem hypothesis and conclusion
theorem lambda_inequality (n : ℕ) (hn : n ≥ 4) (lambda_n : ℝ) :
  lambda_n ≥ 2 * Real.sin ((n-2) * Real.pi / (2 * n)) :=
by
  -- Placeholder for the proof
  sorry

end lambda_inequality_l16_16828


namespace arithmetic_mean_of_two_digit_multiples_of_8_l16_16852

theorem arithmetic_mean_of_two_digit_multiples_of_8 :
  let multiples := { x : ℕ | 8 ∣ x ∧ 10 ≤ x ∧ x < 100 } in
  let count := multiples.toFinset.card in
  let sum := multiples.toFinset.sum id in
  (sum / count : ℕ) = 56 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_8_l16_16852


namespace minimal_abs_diff_l16_16514

theorem minimal_abs_diff (a b : ℤ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b - 8 * a + 7 * b = 569) : abs (a - b) = 23 :=
sorry

end minimal_abs_diff_l16_16514


namespace Nancy_picked_l16_16470

def Alyssa_picked : ℕ := 42
def Total_picked : ℕ := 59

theorem Nancy_picked : Total_picked - Alyssa_picked = 17 := by
  sorry

end Nancy_picked_l16_16470


namespace a_is_perfect_square_l16_16684

theorem a_is_perfect_square (a b : ℕ) (h : ab ∣ (a^2 + b^2 + a)) : (∃ k : ℕ, a = k^2) :=
sorry

end a_is_perfect_square_l16_16684


namespace children_left_l16_16432

-- Define the initial problem constants and conditions
def totalGuests := 50
def halfGuests := totalGuests / 2
def numberOfMen := 15
def numberOfWomen := halfGuests
def numberOfChildren := totalGuests - (numberOfWomen + numberOfMen)
def proportionMenLeft := numberOfMen / 5
def totalPeopleStayed := 43
def totalPeopleLeft := totalGuests - totalPeopleStayed

-- Define the proposition to prove
theorem children_left : 
  totalPeopleLeft - proportionMenLeft = 4 := by 
    sorry

end children_left_l16_16432


namespace increased_time_between_maintenance_checks_l16_16453

theorem increased_time_between_maintenance_checks (original_time : ℕ) (percentage_increase : ℕ) : 
  original_time = 20 → percentage_increase = 25 →
  original_time + (original_time * percentage_increase / 100) = 25 :=
by
  intros
  sorry

end increased_time_between_maintenance_checks_l16_16453
