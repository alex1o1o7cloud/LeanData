import Mathlib

namespace decomposition_x_pqr_l2031_203147

-- Definitions of vectors x, p, q, r
def x : ℝ := sorry
def p : ℝ := sorry
def q : ℝ := sorry
def r : ℝ := sorry

-- The linear combination we want to prove
theorem decomposition_x_pqr : 
  (x = -1 • p + 4 • q + 3 • r) :=
sorry

end decomposition_x_pqr_l2031_203147


namespace find_x_l2031_203106

def infinite_sqrt (d : ℝ) : ℝ := sorry -- A placeholder since infinite nesting is non-trivial

def bowtie (c d : ℝ) : ℝ := c - infinite_sqrt d

theorem find_x (x : ℝ) (h : bowtie 7 x = 3) : x = 20 :=
sorry

end find_x_l2031_203106


namespace ratio_second_third_l2031_203176

theorem ratio_second_third (S T : ℕ) (h_sum : 200 + S + T = 500) (h_third : T = 100) : S / T = 2 := by
  sorry

end ratio_second_third_l2031_203176


namespace smallest_vertical_distance_between_graphs_l2031_203157

noncomputable def f (x : ℝ) : ℝ := abs x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem smallest_vertical_distance_between_graphs :
  ∃ (d : ℝ), (∀ (x : ℝ), |f x - g x| ≥ d) ∧ (∀ (ε : ℝ), ε > 0 → ∃ (x : ℝ), |f x - g x| < d + ε) ∧ d = 3 / 4 :=
by
  sorry

end smallest_vertical_distance_between_graphs_l2031_203157


namespace max_value_sin_sin2x_l2031_203169

open Real

/-- Given x is an acute angle, find the maximum value of the function y = sin x * sin (2 * x). -/
theorem max_value_sin_sin2x (x : ℝ) (hx : 0 < x ∧ x < π / 2) :
    ∃ max_y : ℝ, ∀ y : ℝ, y = sin x * sin (2 * x) -> y ≤ max_y ∧ max_y = 4 * sqrt 3 / 9 :=
by
  -- To be completed
  sorry

end max_value_sin_sin2x_l2031_203169


namespace find_ordered_pair_l2031_203154

theorem find_ordered_pair (x y : ℚ) 
  (h1 : 7 * x - 30 * y = 3) 
  (h2 : 3 * y - x = 5) : 
  x = -53 / 3 ∧ y = -38 / 9 :=
sorry

end find_ordered_pair_l2031_203154


namespace trapezoid_shorter_base_length_l2031_203168

theorem trapezoid_shorter_base_length (longer_base : ℕ) (segment_length : ℕ) (shorter_base : ℕ) 
  (h1 : longer_base = 120) (h2 : segment_length = 7)
  (h3 : segment_length = (longer_base - shorter_base) / 2) : 
  shorter_base = 106 := by
  sorry

end trapezoid_shorter_base_length_l2031_203168


namespace problem_l2031_203112

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + Real.sin x) / (x ^ 2 + 1)

noncomputable def f' (x : ℝ) : ℝ := ((2 + Real.cos x) * (x ^ 2 + 1) - (2 * x + Real.sin x) * (2 * x)) / (x ^ 2 + 1) ^ 2

theorem problem : f 2016 + f' 2016 + f (-2016) - f' (-2016) = 2 := by
  sorry

end problem_l2031_203112


namespace some_number_proof_l2031_203150

def g (n : ℕ) : ℕ :=
  if n < 3 then 1 else 
  if n % 2 = 0 then g (n - 1) else 
    g (n - 2) * n

theorem some_number_proof : g 106 - g 103 = 105 :=
by sorry

end some_number_proof_l2031_203150


namespace evaluate_expression_l2031_203152

def numerator : ℤ :=
  (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1)

def denominator : ℤ :=
  (2 - 3) + (4 - 5) + (6 - 7) + (8 - 9) + (10 - 11) + 12

theorem evaluate_expression : numerator / denominator = 6 / 7 := by
  sorry

end evaluate_expression_l2031_203152


namespace calculate_number_of_models_l2031_203171

-- Define the constants and conditions
def time_per_set : ℕ := 2  -- time per set in minutes
def sets_bathing_suits : ℕ := 2  -- number of bathing suit sets each model wears
def sets_evening_wear : ℕ := 3  -- number of evening wear sets each model wears
def total_show_time : ℕ := 60  -- total show time in minutes

-- Calculate the total time each model takes
def model_time : ℕ := 
  (sets_bathing_suits + sets_evening_wear) * time_per_set

-- Proof problem statement
theorem calculate_number_of_models : 
  (total_show_time / model_time) = 6 := by
  sorry

end calculate_number_of_models_l2031_203171


namespace endpoint_sum_l2031_203138

theorem endpoint_sum
  (x y : ℤ)
  (H_midpoint_x : (x + 15) / 2 = 10)
  (H_midpoint_y : (y - 8) / 2 = -3) :
  x + y = 7 :=
sorry

end endpoint_sum_l2031_203138


namespace river_flow_speed_eq_l2031_203182

-- Definitions of the given conditions
def ship_speed : ℝ := 30
def distance_downstream : ℝ := 144
def distance_upstream : ℝ := 96

-- Lean 4 statement to prove the condition
theorem river_flow_speed_eq (v : ℝ) :
  (distance_downstream / (ship_speed + v) = distance_upstream / (ship_speed - v)) :=
by { sorry }

end river_flow_speed_eq_l2031_203182


namespace concentric_circles_false_statement_l2031_203198

theorem concentric_circles_false_statement
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : b < c) :
  ¬ (b + a = c + b) :=
sorry

end concentric_circles_false_statement_l2031_203198


namespace trig_identity_l2031_203116

-- Proving the equality (we state the problem here)
theorem trig_identity :
  Real.sin (40 * Real.pi / 180) * (Real.tan (10 * Real.pi / 180) - Real.sqrt 3) = -8 / 3 :=
by
  sorry

end trig_identity_l2031_203116


namespace anna_age_when_married_l2031_203137

-- Define constants for the conditions
def j_married : ℕ := 22
def m : ℕ := 30
def combined_age_today : ℕ := 5 * j_married
def j_current : ℕ := j_married + m

-- Define Anna's current age based on the combined age today and Josh's current age
def a_current : ℕ := combined_age_today - j_current

-- Define Anna's age when married
def a_married : ℕ := a_current - m

-- Statement of the theorem to be proved
theorem anna_age_when_married : a_married = 28 :=
by
  sorry

end anna_age_when_married_l2031_203137


namespace inscribed_square_side_length_l2031_203195

theorem inscribed_square_side_length (AC BC : ℝ) (h₀ : AC = 6) (h₁ : BC = 8) :
  ∃ x : ℝ, x = 24 / 7 :=
by
  sorry

end inscribed_square_side_length_l2031_203195


namespace quadratic_has_real_roots_iff_l2031_203196

theorem quadratic_has_real_roots_iff (k : ℝ) : (∃ x : ℝ, x^2 + 2*x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end quadratic_has_real_roots_iff_l2031_203196


namespace solve_for_x_l2031_203163

theorem solve_for_x (x : ℝ) (h : Real.exp (Real.log 7) = 9 * x + 2) : x = 5 / 9 :=
by {
    -- Proof needs to be filled here
    sorry
}

end solve_for_x_l2031_203163


namespace problem_l2031_203130

variable (R S : Prop)

theorem problem (h1 : R → S) :
  ((¬S → ¬R) ∧ (¬R ∨ S)) :=
by
  sorry

end problem_l2031_203130


namespace jack_weight_l2031_203102

-- Define weights and conditions
def weight_of_rocks : ℕ := 5 * 4
def weight_of_anna : ℕ := 40
def weight_of_jack : ℕ := weight_of_anna - weight_of_rocks

-- Prove that Jack's weight is 20 pounds
theorem jack_weight : weight_of_jack = 20 := by
  sorry

end jack_weight_l2031_203102


namespace tv_price_change_l2031_203136

theorem tv_price_change (P : ℝ) :
  let decrease := 0.20
  let increase := 0.45
  let new_price := P * (1 - decrease)
  let final_price := new_price * (1 + increase)
  final_price - P = 0.16 * P := 
by
  sorry

end tv_price_change_l2031_203136


namespace tan_product_30_60_l2031_203145

theorem tan_product_30_60 : 
  (1 + Real.tan (30 * Real.pi / 180)) * (1 + Real.tan (60 * Real.pi / 180)) = 2 + (4 * Real.sqrt 3) / 3 := 
  sorry

end tan_product_30_60_l2031_203145


namespace min_value_of_u_l2031_203177

theorem min_value_of_u (x y : ℝ) (hx : -2 < x ∧ x < 2) (hy : -2 < y ∧ y < 2) (hxy : x * y = -1) :
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ (12 / 5)) :=
by
  sorry

end min_value_of_u_l2031_203177


namespace line_through_fixed_point_and_parabola_l2031_203153

theorem line_through_fixed_point_and_parabola :
  (∀ (a : ℝ), ∃ (P : ℝ × ℝ), 
    (a - 1) * P.1 - P.2 + 2 * a + 1 = 0 ∧ 
    (∀ (x y : ℝ), (y^2 = - ((9:ℝ) / 2) * x ∧ x = -2 ∧ y = 3) ∨ (x^2 = (4:ℝ) / 3 * y ∧ x = -2 ∧ y = 3))) :=
by
  sorry

end line_through_fixed_point_and_parabola_l2031_203153


namespace sum_of_bases_l2031_203179

theorem sum_of_bases (S₁ S₂ G₁ G₂ : ℚ)
  (h₁ : G₁ = 4 * S₁ / (S₁^2 - 1) + 8 / (S₁^2 - 1))
  (h₂ : G₂ = 8 * S₁ / (S₁^2 - 1) + 4 / (S₁^2 - 1))
  (h₃ : G₁ = 3 * S₂ / (S₂^2 - 1) + 6 / (S₂^2 - 1))
  (h₄ : G₂ = 6 * S₂ / (S₂^2 - 1) + 3 / (S₂^2 - 1)) :
  S₁ + S₂ = 23 :=
by
  sorry

end sum_of_bases_l2031_203179


namespace inequality_solution_l2031_203155

theorem inequality_solution :
  {x : ℝ | (x^2 - 1) / (x - 3)^2 ≥ 0} = (Set.Iic (-1) ∪ Set.Ici 1) :=
by
  sorry

end inequality_solution_l2031_203155


namespace ratio_of_supply_to_demand_l2031_203146

def supply : ℕ := 1800000
def demand : ℕ := 2400000

theorem ratio_of_supply_to_demand : (supply / demand : ℚ) = 3 / 4 := by
  sorry

end ratio_of_supply_to_demand_l2031_203146


namespace fraction_equals_i_l2031_203156

theorem fraction_equals_i (m n : ℝ) (i : ℂ) (h : i * i = -1) (h_cond : m * (1 + i) = (11 + n * i)) :
  (m + n * i) / (m - n * i) = i :=
sorry

end fraction_equals_i_l2031_203156


namespace tan_seventeen_pi_over_four_l2031_203105

theorem tan_seventeen_pi_over_four : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_seventeen_pi_over_four_l2031_203105


namespace geometric_seq_a4_l2031_203129

variable {a : ℕ → ℝ}

theorem geometric_seq_a4 (h : ∀ n, a (n + 2) / a n = a 2 / a 0)
  (root_condition1 : a 2 * a 6 = 64)
  (root_condition2 : a 2 + a 6 = 34) :
  a 4 = 8 :=
by
  sorry

end geometric_seq_a4_l2031_203129


namespace length_of_larger_cuboid_l2031_203164

theorem length_of_larger_cuboid
  (n : ℕ)
  (l_small : ℝ) (w_small : ℝ) (h_small : ℝ)
  (w_large : ℝ) (h_large : ℝ)
  (V_large : ℝ)
  (n_eq : n = 56)
  (dim_small : l_small = 5 ∧ w_small = 3 ∧ h_small = 2)
  (dim_large : w_large = 14 ∧ h_large = 10)
  (V_large_eq : V_large = n * (l_small * w_small * h_small)) :
  ∃ l_large : ℝ, l_large = V_large / (w_large * h_large) ∧ l_large = 12 := by
  sorry

end length_of_larger_cuboid_l2031_203164


namespace problem1_problem2_l2031_203114

variable (a b : ℝ)

-- Proof problem for Question 1
theorem problem1 : 2 * a * (a^2 - 3 * a - 1) = 2 * a^3 - 6 * a^2 - 2 * a :=
by sorry

-- Proof problem for Question 2
theorem problem2 : (a^2 * b - 2 * a * b^2 + b^3) / b - (a + b)^2 = -4 * a * b :=
by sorry

end problem1_problem2_l2031_203114


namespace spider_legs_total_l2031_203104

-- Definitions based on given conditions
def spiders : ℕ := 4
def legs_per_spider : ℕ := 8

-- Theorem statement
theorem spider_legs_total : (spiders * legs_per_spider) = 32 := by
  sorry

end spider_legs_total_l2031_203104


namespace solve_for_n_l2031_203125

theorem solve_for_n : 
  (∃ n : ℤ, (1 / (n + 2) + 2 / (n + 2) + (n + 1) / (n + 2) = 3)) ↔ n = -1 :=
sorry

end solve_for_n_l2031_203125


namespace increasing_digits_count_l2031_203173

theorem increasing_digits_count : 
  ∃ n, n = 120 ∧ ∀ x : ℕ, x ≤ 1000 → (∀ i j : ℕ, i < j → ((x / 10^i % 10) < (x / 10^j % 10)) → 
  x ≤ 1000 ∧ (x / 10^i % 10) ≠ (x / 10^j % 10)) :=
sorry

end increasing_digits_count_l2031_203173


namespace a_eq_b_pow_n_l2031_203172

variables (a b n : ℕ)
variable (h : ∀ (k : ℕ), k ≠ b → b - k ∣ a - k^n)

theorem a_eq_b_pow_n : a = b^n := 
by
  sorry

end a_eq_b_pow_n_l2031_203172


namespace tan_alpha_not_unique_l2031_203139

theorem tan_alpha_not_unique (α : ℝ) (h1 : α > 0) (h2 : α < Real.pi) (h3 : (Real.sin α)^2 + Real.cos (2 * α) = 1) :
  ¬(∃ t : ℝ, Real.tan α = t) :=
by
  sorry

end tan_alpha_not_unique_l2031_203139


namespace doubled_radius_and_arc_length_invariant_l2031_203144

theorem doubled_radius_and_arc_length_invariant (r l : ℝ) : (l / r) = (2 * l / (2 * r)) :=
by
  sorry

end doubled_radius_and_arc_length_invariant_l2031_203144


namespace negation_proposition_l2031_203190

theorem negation_proposition :
  (¬ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 < 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 ≥ 0) := 
by
  sorry

end negation_proposition_l2031_203190


namespace value_of_expression_l2031_203101

theorem value_of_expression (a b c : ℝ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 30) (h3 : a + b + c = 15) : 40 * a * b / c = 1200 :=
by
  sorry

end value_of_expression_l2031_203101


namespace find_a_l2031_203115

theorem find_a (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) 
  (h_max : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^(2*x) + 2 * a^x - 1 ≤ 7) 
  (h_eq : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^(2*x) + 2 * a^x - 1 = 7) : 
  a = 2 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l2031_203115


namespace milan_rate_per_minute_l2031_203199

-- Definitions based on the conditions
def monthly_fee : ℝ := 2.0
def total_bill : ℝ := 23.36
def total_minutes : ℕ := 178
def expected_rate_per_minute : ℝ := 0.12

-- Theorem statement based on the question
theorem milan_rate_per_minute :
  (total_bill - monthly_fee) / total_minutes = expected_rate_per_minute := 
by 
  sorry

end milan_rate_per_minute_l2031_203199


namespace calculate_expression_l2031_203191

theorem calculate_expression : ((18^18 / 18^17)^3 * 8^3) / 2^9 = 5832 := by
  sorry

end calculate_expression_l2031_203191


namespace bus_stop_time_l2031_203158

theorem bus_stop_time (speed_excl_stops speed_incl_stops : ℝ) (h1 : speed_excl_stops = 50) (h2 : speed_incl_stops = 45) : (60 * ((speed_excl_stops - speed_incl_stops) / speed_excl_stops)) = 6 := 
by
  sorry

end bus_stop_time_l2031_203158


namespace amoebas_after_ten_days_l2031_203100

def amoeba_split_fun (n : Nat) : Nat := 3^n

theorem amoebas_after_ten_days : amoeba_split_fun 10 = 59049 := by
  have h : 3 ^ 10 = 59049 := by norm_num
  exact h

end amoebas_after_ten_days_l2031_203100


namespace usual_time_to_school_l2031_203109

variables (R T : ℝ)

theorem usual_time_to_school (h₁ : T > 0) (h₂ : R > 0) (h₃ : R / T = (5 / 4 * R) / (T - 4)) :
  T = 20 :=
by
  sorry

end usual_time_to_school_l2031_203109


namespace common_ratio_is_two_l2031_203178

-- Define the geometric sequence
def geom_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a_1 * q^n

-- Define the conditions
variables (a_1 q : ℝ)
variables (h_inc : 1 < q) (h_pos : 0 < a_1)
variables (h_seq : ∀ n : ℕ, 2 * (geom_seq a_1 q n + geom_seq a_1 q (n+2)) = 5 * geom_seq a_1 q (n+1))

-- Statement to prove
theorem common_ratio_is_two : q = 2 :=
by
  sorry

end common_ratio_is_two_l2031_203178


namespace tan_theta_of_obtuse_angle_l2031_203110

noncomputable def theta_expression (θ : Real) : Complex :=
  Complex.mk (3 * Real.sin θ) (Real.cos θ)

theorem tan_theta_of_obtuse_angle {θ : Real} (h_modulus : Complex.abs (theta_expression θ) = Real.sqrt 5) 
  (h_obtuse : π / 2 < θ ∧ θ < π) : Real.tan θ = -1 := 
  sorry

end tan_theta_of_obtuse_angle_l2031_203110


namespace remainder_problem_l2031_203121

theorem remainder_problem (x y : ℤ) (k m : ℤ) 
  (hx : x = 126 * k + 11) 
  (hy : y = 126 * m + 25) :
  (x + y + 23) % 63 = 59 := 
by
  sorry

end remainder_problem_l2031_203121


namespace perimeter_of_shaded_shape_l2031_203108

noncomputable def shaded_perimeter (x : ℝ) : ℝ := 
  let l := 18 - 2 * x
  3 * l

theorem perimeter_of_shaded_shape (x : ℝ) (hx : x > 0) (h_sectors : 2 * x + (18 - 2 * x) = 18) : 
  shaded_perimeter x = 54 := 
by
  rw [shaded_perimeter]
  rw [← h_sectors]
  simp
  sorry

end perimeter_of_shaded_shape_l2031_203108


namespace Johnson_Carter_Tie_August_l2031_203119

structure MonthlyHomeRuns where
  March : Nat
  April : Nat
  May : Nat
  June : Nat
  July : Nat
  August : Nat
  September : Nat

def Johnson_runs : MonthlyHomeRuns := { March:= 2, April:= 11, May:= 15, June:= 9, July:= 7, August:= 9, September:= 0 }
def Carter_runs : MonthlyHomeRuns := { March:= 1, April:= 9, May:= 8, June:= 19, July:= 6, August:= 10, September:= 0 }

noncomputable def cumulative_runs (runs: MonthlyHomeRuns) (month: String) : Nat :=
  match month with
  | "March" => runs.March
  | "April" => runs.March + runs.April
  | "May" => runs.March + runs.April + runs.May
  | "June" => runs.March + runs.April + runs.May + runs.June
  | "July" => runs.March + runs.April + runs.May + runs.June + runs.July
  | "August" => runs.March + runs.April + runs.May + runs.June + runs.July + runs.August
  | _ => 0

theorem Johnson_Carter_Tie_August :
  cumulative_runs Johnson_runs "August" = cumulative_runs Carter_runs "August" := 
  by
  sorry

end Johnson_Carter_Tie_August_l2031_203119


namespace function_characterization_l2031_203131
noncomputable def f : ℕ → ℕ := sorry

theorem function_characterization (h : ∀ m n : ℕ, m^2 + f n ∣ m * f m + n) : 
  ∀ n : ℕ, f n = n :=
by
  intro n
  sorry

end function_characterization_l2031_203131


namespace total_spent_in_may_l2031_203162

-- Conditions as definitions
def cost_per_weekday : ℕ := (2 * 15) + (2 * 18)
def cost_per_weekend_day : ℕ := (3 * 12) + (2 * 20)
def weekdays_in_may : ℕ := 22
def weekend_days_in_may : ℕ := 9

-- The statement to prove
theorem total_spent_in_may :
  cost_per_weekday * weekdays_in_may + cost_per_weekend_day * weekend_days_in_may = 2136 :=
by
  sorry

end total_spent_in_may_l2031_203162


namespace factorize_x4_plus_16_l2031_203181

theorem factorize_x4_plus_16: ∀ (x : ℝ), x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  sorry

end factorize_x4_plus_16_l2031_203181


namespace exists_n_prime_factors_m_exp_n_plus_n_exp_m_l2031_203186

theorem exists_n_prime_factors_m_exp_n_plus_n_exp_m (m k : ℕ) (hm : m > 0) (hm_odd : m % 2 = 1) (hk : k > 0) :
  ∃ n : ℕ, n > 0 ∧ (∃ primes : Finset ℕ, primes.card ≥ k ∧ ∀ p ∈ primes, p.Prime ∧ p ∣ m ^ n + n ^ m) := 
sorry

end exists_n_prime_factors_m_exp_n_plus_n_exp_m_l2031_203186


namespace real_solutions_of_equation_l2031_203149

theorem real_solutions_of_equation (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 12) ↔ (x = 13 ∨ x = -5) :=
by
  sorry

end real_solutions_of_equation_l2031_203149


namespace victor_percentage_of_marks_l2031_203113

theorem victor_percentage_of_marks (marks_obtained : ℝ) (maximum_marks : ℝ) (h1 : marks_obtained = 285) (h2 : maximum_marks = 300) : 
  (marks_obtained / maximum_marks) * 100 = 95 :=
by
  sorry

end victor_percentage_of_marks_l2031_203113


namespace prob_both_A_B_prob_exactly_one_l2031_203148

def prob_A : ℝ := 0.8
def prob_not_B : ℝ := 0.1
def prob_B : ℝ := 1 - prob_not_B

lemma prob_independent (a b : Prop) : Prop := -- Placeholder for actual independence definition
sorry

-- Given conditions
variables (P_A : ℝ := prob_A) (P_not_B : ℝ := prob_not_B) (P_B : ℝ := prob_B) (indep : ∀ A B, prob_independent A B)

-- Questions translated to Lean statements
theorem prob_both_A_B : P_A * P_B = 0.72 := sorry

theorem prob_exactly_one : (P_A * P_not_B) + ((1 - P_A) * P_B) = 0.26 := sorry

end prob_both_A_B_prob_exactly_one_l2031_203148


namespace solve_system_of_equations_l2031_203174

theorem solve_system_of_equations (x y : ℝ) 
  (h1 : 6.751 * x + 3.249 * y = 26.751) 
  (h2 : 3.249 * x + 6.751 * y = 23.249) : 
  x = 3 ∧ y = 2 := 
sorry

end solve_system_of_equations_l2031_203174


namespace correct_option_l2031_203189

-- Definitions based on the problem's conditions
def option_A (x : ℝ) : Prop := x^2 * x^4 = x^8
def option_B (x : ℝ) : Prop := (x^2)^3 = x^5
def option_C (x : ℝ) : Prop := x^2 + x^2 = 2 * x^2
def option_D (x : ℝ) : Prop := (3 * x)^2 = 3 * x^2

-- Theorem stating that out of the given options, option C is correct
theorem correct_option (x : ℝ) : option_C x :=
by {
  sorry
}

end correct_option_l2031_203189


namespace simplify_polynomial_l2031_203197

/-- Simplification of the polynomial expression -/
theorem simplify_polynomial (x : ℝ) :
  x * (4 * x^2 - 2) - 5 * (x^2 - 3 * x + 5) = 4 * x^3 - 5 * x^2 + 13 * x - 25 :=
by
  sorry

end simplify_polynomial_l2031_203197


namespace shaded_fraction_eighth_triangle_l2031_203184

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2
def square_number (n : Nat) : Nat := n * n

theorem shaded_fraction_eighth_triangle :
  let shaded_triangles := triangular_number 7
  let total_triangles := square_number 8
  shaded_triangles / total_triangles = 7 / 16 := 
by
  sorry

end shaded_fraction_eighth_triangle_l2031_203184


namespace numberOfTrucks_l2031_203161

-- Conditions
def numberOfTanksPerTruck : ℕ := 3
def capacityPerTank : ℕ := 150
def totalWaterCapacity : ℕ := 1350

-- Question and proof goal
theorem numberOfTrucks : 
  (totalWaterCapacity / (numberOfTanksPerTruck * capacityPerTank) = 3) := 
by 
  sorry

end numberOfTrucks_l2031_203161


namespace tangent_circle_given_r_l2031_203124

theorem tangent_circle_given_r (r : ℝ) (h_pos : 0 < r)
    (h_tangent : ∀ x y : ℝ, (2 * x + y = r) → (x^2 + y^2 = 2 * r))
  : r = 10 :=
sorry

end tangent_circle_given_r_l2031_203124


namespace polynomial_root_sum_l2031_203135

theorem polynomial_root_sum : 
  ∀ (r1 r2 r3 r4 : ℝ), 
  (r1^4 - r1 - 504 = 0) ∧ 
  (r2^4 - r2 - 504 = 0) ∧ 
  (r3^4 - r3 - 504 = 0) ∧ 
  (r4^4 - r4 - 504 = 0) → 
  r1^4 + r2^4 + r3^4 + r4^4 = 2016 := by
sorry

end polynomial_root_sum_l2031_203135


namespace remainder_of_3x_plus_5y_l2031_203187

-- Conditions and parameter definitions
def x (k : ℤ) := 13 * k + 7
def y (m : ℤ) := 17 * m + 11

-- Proof statement
theorem remainder_of_3x_plus_5y (k m : ℤ) : (3 * x k + 5 * y m) % 221 = 76 := by
  sorry

end remainder_of_3x_plus_5y_l2031_203187


namespace question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l2031_203151

theorem question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1
    (a b c d : ℤ)
    (h1 : a + b = 11)
    (h2 : b + c = 9)
    (h3 : c + d = 3)
    : a + d = -1 :=
by
  sorry

end question_a_plus_b_eq_11_b_plus_c_eq_9_c_plus_d_eq_3_a_plus_d_eq_neg1_l2031_203151


namespace pizza_slices_with_both_toppings_l2031_203142

theorem pizza_slices_with_both_toppings (total_slices pepperoni_slices mushroom_slices n : ℕ) 
    (h1 : total_slices = 14) 
    (h2 : pepperoni_slices = 8) 
    (h3 : mushroom_slices = 12) 
    (h4 : ∀ s, s = pepperoni_slices + mushroom_slices - n ∧ s = total_slices := by sorry) :
    n = 6 :=
sorry

end pizza_slices_with_both_toppings_l2031_203142


namespace triangle_right_angled_l2031_203117

-- Define the variables and the condition of the problem
variables {a b c : ℝ}

-- Given condition of the problem
def triangle_condition (a b c : ℝ) : Prop :=
  2 * (a ^ 8 + b ^ 8 + c ^ 8) = (a ^ 4 + b ^ 4 + c ^ 4) ^ 2

-- The theorem to prove the triangle is right-angled
theorem triangle_right_angled (h : triangle_condition a b c) : a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 :=
sorry

end triangle_right_angled_l2031_203117


namespace num_people_second_hour_l2031_203134

theorem num_people_second_hour 
  (n1_in n2_in n1_left n2_left : ℕ) 
  (rem_hour1 rem_hour2 : ℕ)
  (h1 : n1_in = 94)
  (h2 : n1_left = 27)
  (h3 : n2_left = 9)
  (h4 : rem_hour2 = 76)
  (h5 : rem_hour1 = n1_in - n1_left)
  (h6 : rem_hour2 = rem_hour1 + n2_in - n2_left) :
  n2_in = 18 := 
  by 
  sorry

end num_people_second_hour_l2031_203134


namespace possible_third_side_l2031_203193

theorem possible_third_side (x : ℝ) : (3 + 4 > x) ∧ (abs (4 - 3) < x) → (x = 2) :=
by 
  sorry

end possible_third_side_l2031_203193


namespace find_number_l2031_203118

theorem find_number (x : ℝ) :
  (1.5 * 1265) / x = 271.07142857142856 → x = 7 :=
by
  intro h
  sorry

end find_number_l2031_203118


namespace negation_of_exists_l2031_203185

theorem negation_of_exists (p : Prop) : 
  (∃ (x₀ : ℝ), x₀ > 0 ∧ |x₀| ≤ 2018) ↔ 
  ¬(∀ (x : ℝ), x > 0 → |x| > 2018) :=
by sorry

end negation_of_exists_l2031_203185


namespace ratio_3_2_l2031_203180

theorem ratio_3_2 (m n : ℕ) (h1 : m + n = 300) (h2 : m > 100) (h3 : n > 100) : m / n = 3 / 2 := by
  sorry

end ratio_3_2_l2031_203180


namespace arithmetic_sequence_difference_l2031_203165

theorem arithmetic_sequence_difference 
  (a b c : ℝ) 
  (h1: 2 + (7 / 4) = a)
  (h2: 2 + 2 * (7 / 4) = b)
  (h3: 2 + 3 * (7 / 4) = c)
  (h4: 2 + 4 * (7 / 4) = 9):
  c - a = 3.5 :=
by sorry

end arithmetic_sequence_difference_l2031_203165


namespace calculate_expression_l2031_203160

theorem calculate_expression : 64 + 5 * 12 / (180 / 3) = 65 := by
  sorry

end calculate_expression_l2031_203160


namespace garden_width_is_14_l2031_203140

theorem garden_width_is_14 (w : ℝ) (h1 : ∃ (l : ℝ), l = 3 * w ∧ l * w = 588) : w = 14 :=
sorry

end garden_width_is_14_l2031_203140


namespace inequality_problem_l2031_203126

theorem inequality_problem
  (a b c d : ℝ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h_sum : a + b + c + d = 1) :
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1 / 5 :=
by
  sorry

end inequality_problem_l2031_203126


namespace michael_points_scored_l2031_203143

theorem michael_points_scored (team_points : ℕ) (other_players : ℕ) (average_points : ℕ) (michael_points : ℕ) :
  team_points = 72 → other_players = 8 → average_points = 9 → 
  michael_points = team_points - other_players * average_points → michael_points = 36 :=
by
  intro h_team_points h_other_players h_average_points h_calculation
  -- skip the actual proof for now
  sorry

end michael_points_scored_l2031_203143


namespace range_of_sum_l2031_203188

theorem range_of_sum (a b c : ℝ) (h1: a > b) (h2 : b > c) (h3 : a + b + c = 1) (h4 : a^2 + b^2 + c^2 = 3) :
-2/3 < b + c ∧ b + c < 0 := 
by 
  sorry

end range_of_sum_l2031_203188


namespace calculate_expression_l2031_203133

theorem calculate_expression : 8 / 2 - 3 - 12 + 3 * (5^2 - 4) = 52 := 
by
  sorry

end calculate_expression_l2031_203133


namespace least_possible_value_of_smallest_integer_l2031_203123

theorem least_possible_value_of_smallest_integer :
  ∀ (A B C D : ℤ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  (A + B + C + D) / 4 = 76 ∧ D = 90 →
  A = 37 :=
by
  sorry

end least_possible_value_of_smallest_integer_l2031_203123


namespace power_of_sqrt2_minus_1_l2031_203192

noncomputable def a (n : ℕ) : ℝ := (Real.sqrt 2 - 1) ^ n
noncomputable def b (n : ℕ) : ℝ := (Real.sqrt 2 + 1) ^ n
noncomputable def c (n : ℕ) : ℝ := (b n + a n) / 2
noncomputable def d (n : ℕ) : ℝ := (b n - a n) / 2

theorem power_of_sqrt2_minus_1 (n : ℕ) : a n = Real.sqrt (d n ^ 2 + 1) - Real.sqrt (d n ^ 2) :=
by
  sorry

end power_of_sqrt2_minus_1_l2031_203192


namespace train_average_speed_l2031_203111

open Real -- Assuming all required real number operations 

noncomputable def average_speed (distances : List ℝ) (times : List ℝ) : ℝ := 
  let total_distance := distances.sum
  let total_time := times.sum
  total_distance / total_time

theorem train_average_speed :
  average_speed [125, 270] [2.5, 3] = 71.82 := 
by 
  -- Details of the actual proof steps are omitted
  sorry

end train_average_speed_l2031_203111


namespace possible_quadrilateral_areas_l2031_203175

-- Define the problem set up
structure Point where
  x : ℝ
  y : ℝ

structure Square where
  side_length : ℝ
  A : Point
  B : Point
  C : Point
  D : Point

-- Defines the division points on each side of the square
def division_points (A B C D : Point) : List Point :=
  [
    -- Points on AB
    { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
    -- Points on BC
    { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
    -- Points on CD
    { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
    -- Points on DA
    { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
  ]

-- Possible areas calculation using the Shoelace Theorem
def quadrilateral_areas : List ℝ :=
  [6, 7, 7.5, 8, 8.5, 9, 10]

-- Math proof problem in Lean, we need to prove that the quadrilateral areas match the given values
theorem possible_quadrilateral_areas (ABCD : Square) (pts : List Point) :
    (division_points ABCD.A ABCD.B ABCD.C ABCD.D) = [
      { x := 1, y := 4 }, { x := 2, y := 4 }, { x := 3, y := 4 },
      { x := 4, y := 3 }, { x := 4, y := 2 }, { x := 4, y := 1 },
      { x := 3, y := 0 }, { x := 2, y := 0 }, { x := 1, y := 0 },
      { x := 0, y := 3 }, { x := 0, y := 2 }, { x := 0, y := 1 }
    ] → 
    (∃ areas, areas ⊆ quadrilateral_areas) := by
  sorry

end possible_quadrilateral_areas_l2031_203175


namespace common_card_cost_l2031_203120

def totalDeckCost (rareCost uncommonCost commonCost numRares numUncommons numCommons : ℝ) : ℝ :=
  (numRares * rareCost) + (numUncommons * uncommonCost) + (numCommons * commonCost)

theorem common_card_cost (numRares numUncommons numCommons : ℝ) (rareCost uncommonCost totalCost : ℝ) : 
  numRares = 19 → numUncommons = 11 → numCommons = 30 → 
  rareCost = 1 → uncommonCost = 0.5 → totalCost = 32 → 
  commonCost = 0.25 :=
by 
  intros 
  sorry

end common_card_cost_l2031_203120


namespace carrie_is_left_with_50_l2031_203159

-- Definitions for the conditions given in the problem
def amount_given : ℕ := 91
def cost_of_sweater : ℕ := 24
def cost_of_tshirt : ℕ := 6
def cost_of_shoes : ℕ := 11

-- Definition of the total amount spent
def total_spent : ℕ := cost_of_sweater + cost_of_tshirt + cost_of_shoes

-- Definition of the amount left
def amount_left : ℕ := amount_given - total_spent

-- The theorem we want to prove
theorem carrie_is_left_with_50 : amount_left = 50 :=
by
  have h1 : amount_given = 91 := rfl
  have h2 : total_spent = 41 := rfl
  have h3 : amount_left = 50 := rfl
  exact rfl

end carrie_is_left_with_50_l2031_203159


namespace largest_circle_center_is_A_l2031_203166

-- Define the given lengths of the pentagon's sides
def AB : ℝ := 16
def BC : ℝ := 14
def CD : ℝ := 17
def DE : ℝ := 13
def AE : ℝ := 14

-- Define the radii of the circles centered at points A, B, C, D, E
variables (R_A R_B R_C R_D R_E : ℝ)

-- Conditions based on the problem statement
def radius_conditions : Prop :=
  R_A + R_B = AB ∧
  R_B + R_C = BC ∧
  R_C + R_D = CD ∧
  R_D + R_E = DE ∧
  R_E + R_A = AE

-- The main theorem to prove
theorem largest_circle_center_is_A (h : radius_conditions R_A R_B R_C R_D R_E) :
  10 ≥ R_A ∧ R_A ≥ R_B ∧ R_A ≥ R_C ∧ R_A ≥ R_D ∧ R_A ≥ R_E :=
by sorry

end largest_circle_center_is_A_l2031_203166


namespace neither_sufficient_nor_necessary_l2031_203194

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem neither_sufficient_nor_necessary (a : ℝ) :
  (a ∈ M → a ∈ N) = false ∧ (a ∈ N → a ∈ M) = false := by
  sorry

end neither_sufficient_nor_necessary_l2031_203194


namespace sum_of_numbers_l2031_203103

theorem sum_of_numbers (a b : ℕ) (h : a + 4 * b = 30) : a + b = 12 :=
sorry

end sum_of_numbers_l2031_203103


namespace Alexis_mangoes_l2031_203141

-- Define the variables for the number of mangoes each person has.
variable (A D Ash : ℕ)

-- Conditions given in the problem.
axiom h1 : A = 4 * (D + Ash)
axiom h2 : A + D + Ash = 75

-- The proof goal.
theorem Alexis_mangoes : A = 60 :=
sorry

end Alexis_mangoes_l2031_203141


namespace maze_paths_unique_l2031_203183

-- Define the conditions and branching points
def maze_structure (x : ℕ) (b : ℕ) : Prop :=
  x > 0 ∧ b > 0 ∧
  -- This represents the structure and unfolding paths at each point
  ∀ (i : ℕ), i < x → ∃ j < b, True

-- Define a function to count the number of unique paths given the number of branching points
noncomputable def count_paths (x : ℕ) (b : ℕ) : ℕ :=
  x * (2 ^ b)

-- State the main theorem
theorem maze_paths_unique : ∃ x b, maze_structure x b ∧ count_paths x b = 16 :=
by
  -- The proof contents are skipped for now
  sorry

end maze_paths_unique_l2031_203183


namespace surface_area_of_cube_l2031_203128

theorem surface_area_of_cube (edge : ℝ) (h : edge = 5) : 6 * (edge * edge) = 150 := by
  have h_square : edge * edge = 25 := by
    rw [h]
    norm_num
  rw [h_square]
  norm_num

end surface_area_of_cube_l2031_203128


namespace chooseOneFromEachCategory_chooseTwoDifferentTypes_l2031_203107

-- Define the number of different paintings in each category
def traditionalChinesePaintings : ℕ := 5
def oilPaintings : ℕ := 2
def watercolorPaintings : ℕ := 7

-- Part (1): Prove that the number of ways to choose one painting from each category is 70
theorem chooseOneFromEachCategory : traditionalChinesePaintings * oilPaintings * watercolorPaintings = 70 := by
  sorry

-- Part (2): Prove that the number of ways to choose two paintings of different types is 59
theorem chooseTwoDifferentTypes :
  (traditionalChinesePaintings * oilPaintings) + 
  (traditionalChinesePaintings * watercolorPaintings) + 
  (oilPaintings * watercolorPaintings) = 59 := by
  sorry

end chooseOneFromEachCategory_chooseTwoDifferentTypes_l2031_203107


namespace roger_collected_nickels_l2031_203170

theorem roger_collected_nickels 
  (N : ℕ)
  (initial_pennies : ℕ := 42) 
  (initial_dimes : ℕ := 15)
  (donated_coins : ℕ := 66)
  (left_coins : ℕ := 27)
  (h_total_coins_initial : initial_pennies + N + initial_dimes - donated_coins = left_coins) :
  N = 36 := 
sorry

end roger_collected_nickels_l2031_203170


namespace friends_reach_destinations_l2031_203122

noncomputable def travel_times (d : ℕ) := 
  let walking_speed := 6
  let cycling_speed := 18
  let meet_time := d / (walking_speed + cycling_speed)
  let remaining_time := d / cycling_speed
  let total_time_A := meet_time + (d - cycling_speed * meet_time) / walking_speed
  let total_time_B := (cycling_speed * meet_time) / walking_speed + (d - cycling_speed * meet_time) / walking_speed
  let total_time_C := remaining_time + meet_time
  (total_time_A, total_time_B, total_time_C)

theorem friends_reach_destinations (d : ℕ) (d_eq_24 : d = 24) : 
  let (total_time_A, total_time_B, total_time_C) := travel_times d
  total_time_A ≤ 160 / 60 ∧ total_time_B ≤ 160 / 60 ∧ total_time_C ≤ 160 / 60 :=
by 
  sorry

end friends_reach_destinations_l2031_203122


namespace angles_equal_l2031_203127

theorem angles_equal (A B C : ℝ) (h1 : A + B = 180) (h2 : B + C = 180) : A = C := sorry

end angles_equal_l2031_203127


namespace find_factor_l2031_203167

theorem find_factor (x f : ℝ) (h1 : x = 6)
    (h2 : (2 * x + 9) * f = 63) : f = 3 :=
sorry

end find_factor_l2031_203167


namespace product_with_a_equals_3_l2031_203132

theorem product_with_a_equals_3 (a : ℤ) (h : a = 3) : 
  (a - 12) * (a - 11) * (a - 10) * (a - 9) * (a - 8) * (a - 7) * (a - 6) * 
  (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a * 3 = 0 :=
by
  sorry

end product_with_a_equals_3_l2031_203132
