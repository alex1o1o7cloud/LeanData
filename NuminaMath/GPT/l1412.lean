import Mathlib

namespace go_stones_problem_l1412_141231

theorem go_stones_problem
  (x : ℕ) 
  (h1 : x / 7 + 40 = 555 / 5) 
  (black_stones : ℕ) 
  (h2 : black_stones = 55) :
  (x - black_stones = 442) :=
sorry

end go_stones_problem_l1412_141231


namespace second_printer_cost_l1412_141284

theorem second_printer_cost (p1_cost : ℕ) (num_units : ℕ) (total_spent : ℕ) (x : ℕ) 
  (h1 : p1_cost = 375) 
  (h2 : num_units = 7) 
  (h3 : total_spent = p1_cost * num_units) 
  (h4 : total_spent = x * num_units) : 
  x = 375 := 
sorry

end second_printer_cost_l1412_141284


namespace ratio_arithmetic_sequences_l1412_141298

variable (a : ℕ → ℕ) (b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (h : ∀ n : ℕ, S n / T n = (3 * n - 1) / (2 * n + 3))

theorem ratio_arithmetic_sequences :
  a 7 / b 7 = 38 / 29 :=
sorry

end ratio_arithmetic_sequences_l1412_141298


namespace find_first_discount_l1412_141290

theorem find_first_discount (price_initial : ℝ) (price_final : ℝ) (discount_additional : ℝ) (x : ℝ) :
  price_initial = 350 → price_final = 266 → discount_additional = 5 →
  price_initial * (1 - x / 100) * (1 - discount_additional / 100) = price_final →
  x = 20 :=
by
  intros h1 h2 h3 h4
  -- skippable in proofs, just holds the place
  sorry

end find_first_discount_l1412_141290


namespace rectangle_to_square_l1412_141260

theorem rectangle_to_square (length width : ℕ) (h1 : 2 * (length + width) = 40) (h2 : length - 8 = width + 2) :
  width + 2 = 7 :=
by {
  -- Proof goes here
  sorry
}

end rectangle_to_square_l1412_141260


namespace find_polynomial_l1412_141226

def polynomial (a b c : ℚ) : ℚ → ℚ := λ x => a * x^2 + b * x + c

theorem find_polynomial
  (a b c : ℚ)
  (h1 : polynomial a b c (-3) = 0)
  (h2 : polynomial a b c 6 = 0)
  (h3 : polynomial a b c 2 = -24) :
  a = 6/5 ∧ b = -18/5 ∧ c = -108/5 :=
by 
  sorry

end find_polynomial_l1412_141226


namespace triangle_altitude_l1412_141232

variable (Area : ℝ) (base : ℝ) (altitude : ℝ)

theorem triangle_altitude (hArea : Area = 1250) (hbase : base = 50) :
  2 * Area / base = altitude :=
by
  sorry

end triangle_altitude_l1412_141232


namespace tan_add_tan_105_eq_l1412_141289

noncomputable def tan : ℝ → ℝ := sorry -- Use the built-in library later for actual implementation

-- Given conditions
def tan_45_eq : tan 45 = 1 := by sorry
def tan_60_eq : tan 60 = Real.sqrt 3 := by sorry

-- Angle addition formula for tangent
theorem tan_add (a b : ℝ) :
  tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry

-- Main theorem to prove
theorem tan_105_eq :
  tan 105 = -2 - Real.sqrt 3 := by sorry

end tan_add_tan_105_eq_l1412_141289


namespace products_not_all_greater_than_one_quarter_l1412_141200

theorem products_not_all_greater_than_one_quarter
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hb : 0 < b ∧ b < 1)
  (hc : 0 < c ∧ c < 1) :
  ¬ ((1 - a) * b > 1 / 4 ∧ (1 - b) * c > 1 / 4 ∧ (1 - c) * a > 1 / 4) :=
by
  sorry

end products_not_all_greater_than_one_quarter_l1412_141200


namespace min_value_x_plus_inv_x_l1412_141274

open Real

theorem min_value_x_plus_inv_x (x : ℝ) (hx : 0 < x) : x + 1/x ≥ 2 := by
  sorry

end min_value_x_plus_inv_x_l1412_141274


namespace trajectory_sum_of_distances_to_axes_l1412_141263

theorem trajectory_sum_of_distances_to_axes (x y : ℝ) (h : |x| + |y| = 6) :
  |x| + |y| = 6 := 
by 
  sorry

end trajectory_sum_of_distances_to_axes_l1412_141263


namespace beef_cubes_per_slab_l1412_141249

-- Define the conditions as variables
variables (kabob_sticks : ℕ) (cubes_per_stick : ℕ) (cost_per_slab : ℕ) (total_cost : ℕ) (total_kabob_sticks : ℕ)

-- Assume the conditions from step a)
theorem beef_cubes_per_slab 
  (h1 : cubes_per_stick = 4) 
  (h2 : cost_per_slab = 25) 
  (h3 : total_cost = 50) 
  (h4 : total_kabob_sticks = 40)
  : total_cost / cost_per_slab * (total_kabob_sticks * cubes_per_stick) / (total_cost / cost_per_slab) = 80 := 
by {
  -- the proof goes here
  sorry
}

end beef_cubes_per_slab_l1412_141249


namespace train_speed_l1412_141264

theorem train_speed (distance_AB : ℕ) (start_time_A : ℕ) (start_time_B : ℕ) (meet_time : ℕ) (speed_B : ℕ) (time_travel_A : ℕ) (time_travel_B : ℕ)
  (total_distance : ℕ) (distance_B_covered : ℕ) (speed_A : ℕ)
  (h1 : distance_AB = 330)
  (h2 : start_time_A = 8)
  (h3 : start_time_B = 9)
  (h4 : meet_time = 11)
  (h5 : speed_B = 75)
  (h6 : time_travel_A = meet_time - start_time_A)
  (h7 : time_travel_B = meet_time - start_time_B)
  (h8 : distance_B_covered = time_travel_B * speed_B)
  (h9 : total_distance = distance_AB)
  (h10 : total_distance = time_travel_A * speed_A + distance_B_covered):
  speed_A = 60 := 
by
  sorry

end train_speed_l1412_141264


namespace population_net_increase_l1412_141205

-- Define conditions
def birth_rate : ℚ := 5 / 2    -- 5 people every 2 seconds
def death_rate : ℚ := 3 / 2    -- 3 people every 2 seconds
def one_day_in_seconds : ℕ := 86400   -- Number of seconds in one day

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Prove that the net increase in one day is 86400 people given the conditions
theorem population_net_increase :
  net_increase_per_second * one_day_in_seconds = 86400 :=
sorry

end population_net_increase_l1412_141205


namespace xy_sum_one_l1412_141259

theorem xy_sum_one (x y : ℝ) (h : x > 0) (k : y > 0) (hx : x^5 + 5*x^3*y + 5*x^2*y^2 + 5*x*y^3 + y^5 = 1) : x + y = 1 :=
sorry

end xy_sum_one_l1412_141259


namespace cube_root_rational_l1412_141248

theorem cube_root_rational (a b : ℚ) (r : ℚ) (h1 : ∃ x : ℚ, x^3 = a) (h2 : ∃ y : ℚ, y^3 = b) (h3 : ∃ x y : ℚ, x + y = r ∧ x^3 = a ∧ y^3 = b) :
  (∃ x : ℚ, x^3 = a) ∧ (∃ y : ℚ, y^3 = b) :=
sorry

end cube_root_rational_l1412_141248


namespace intersection_A_B_l1412_141216

def set_A : Set ℝ := {x : ℝ | |x| = x}
def set_B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}
def set_intersection : Set ℝ := {x : ℝ | 0 ≤ x}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection :=
by
  sorry

-- You can verify if the Lean code builds successfully using Lean 4 environment.

end intersection_A_B_l1412_141216


namespace monica_total_savings_l1412_141245

theorem monica_total_savings :
  ∀ (weekly_saving : ℤ) (weeks_per_cycle : ℤ) (cycles : ℤ),
    weekly_saving = 15 →
    weeks_per_cycle = 60 →
    cycles = 5 →
    weekly_saving * weeks_per_cycle * cycles = 4500 :=
by
  intros weekly_saving weeks_per_cycle cycles
  sorry

end monica_total_savings_l1412_141245


namespace geometric_progression_general_term_l1412_141266

noncomputable def a_n (n : ℕ) : ℝ := 2^(n-1)

theorem geometric_progression_general_term :
  (∀ n : ℕ, n ≥ 1 → a_n n > 0) ∧
  a_n 1 = 1 ∧
  a_n 2 + a_n 3 = 6 →
  ∀ n, a_n n = 2^(n-1) :=
by
  intros h
  sorry

end geometric_progression_general_term_l1412_141266


namespace NewYearSeasonMarkup_is_25percent_l1412_141204

variable (C N : ℝ)
variable (h1 : N >= 0)
variable (h2 : 0.92 * (1 + N) * 1.20 * C = 1.38 * C)

theorem NewYearSeasonMarkup_is_25percent : N = 0.25 :=
  by
  sorry

end NewYearSeasonMarkup_is_25percent_l1412_141204


namespace exhibition_admission_fees_ratio_l1412_141214

theorem exhibition_admission_fees_ratio
  (a c : ℕ)
  (h1 : 30 * a + 15 * c = 2925)
  (h2 : a % 5 = 0)
  (h3 : c % 5 = 0) :
  (a / 5 = c / 5) :=
by
  sorry

end exhibition_admission_fees_ratio_l1412_141214


namespace number_is_2250_l1412_141207

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l1412_141207


namespace oak_trees_cut_down_l1412_141227

-- Define the conditions
def initial_oak_trees : ℕ := 9
def final_oak_trees : ℕ := 7

-- Prove that the number of oak trees cut down is 2
theorem oak_trees_cut_down : (initial_oak_trees - final_oak_trees) = 2 :=
by
  -- Proof is omitted
  sorry

end oak_trees_cut_down_l1412_141227


namespace find_percentage_l1412_141230

noncomputable def percentage (X : ℝ) : ℝ := (377.8020134228188 * 100 * 5.96) / 1265

theorem find_percentage : percentage 178 = 178 := by
  -- Conditions
  let P : ℝ := 178
  let A : ℝ := 1265
  let divisor : ℝ := 5.96
  let result : ℝ := 377.8020134228188

  -- Define the percentage calculation
  let X := (result * 100 * divisor) / A

  -- Verify the calculation matches
  have h : X = P := by sorry

  trivial

end find_percentage_l1412_141230


namespace pure_gala_trees_l1412_141242

variables (T F G : ℕ)

theorem pure_gala_trees :
  (0.1 * T : ℝ) + F = 238 ∧ F = (3 / 4) * ↑T → G = T - F → G = 70 :=
by
  intro h
  sorry

end pure_gala_trees_l1412_141242


namespace find_xyz_l1412_141267

theorem find_xyz (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 45) (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 15) (h3 : x + y + z = 5) : x * y * z = 10 :=
by
  sorry

end find_xyz_l1412_141267


namespace rugby_team_new_avg_weight_l1412_141213

noncomputable def new_average_weight (original_players : ℕ) (original_avg_weight : ℕ) 
  (new_player_weights : List ℕ) : ℚ :=
  let total_original_weight := original_players * original_avg_weight
  let total_new_weight := new_player_weights.foldl (· + ·) 0
  let new_total_weight := total_original_weight + total_new_weight
  let new_total_players := original_players + new_player_weights.length
  (new_total_weight : ℚ) / (new_total_players : ℚ)

theorem rugby_team_new_avg_weight :
  new_average_weight 20 180 [210, 220, 230] = 185.22 := by
  sorry

end rugby_team_new_avg_weight_l1412_141213


namespace shortest_chord_l1412_141236

noncomputable def line_eq (m : ℝ) (x y : ℝ) : Prop := 2 * m * x - y - 8 * m - 3 = 0
noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 6)^2 = 25

theorem shortest_chord (m : ℝ) :
  (∃ x y, line_eq m x y ∧ circle_eq x y) →
  m = 1 / 6 :=
by sorry

end shortest_chord_l1412_141236


namespace jack_morning_emails_l1412_141283

-- Define the conditions as constants
def totalEmails : ℕ := 10
def emailsAfternoon : ℕ := 3
def emailsEvening : ℕ := 1

-- Problem statement to prove emails in the morning
def emailsMorning : ℕ := totalEmails - (emailsAfternoon + emailsEvening)

-- The theorem to prove
theorem jack_morning_emails : emailsMorning = 6 := by
  sorry

end jack_morning_emails_l1412_141283


namespace min_max_x_l1412_141276

-- Definitions for the initial conditions and surveys
def students : ℕ := 100
def like_math_initial : ℕ := 50
def dislike_math_initial : ℕ := 50
def like_math_final : ℕ := 60
def dislike_math_final : ℕ := 40

-- Variables for the students' responses
variables (a b c d : ℕ)

-- Conditions based on the problem statement
def initial_survey : Prop := a + d = like_math_initial ∧ b + c = dislike_math_initial
def final_survey : Prop := a + c = like_math_final ∧ b + d = dislike_math_final

-- Definition of x as the number of students who changed their answer
def x : ℕ := c + d

-- Prove the minimum and maximum value of x with given conditions
theorem min_max_x (a b c d : ℕ) 
  (initial_cond : initial_survey a b c d)
  (final_cond : final_survey a b c d)
  : 10 ≤ (x c d) ∧ (x c d) ≤ 90 :=
by
  -- This is where the proof would go, but we'll simply state sorry for now.
  sorry

end min_max_x_l1412_141276


namespace min_value_of_f_l1412_141279

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 16) + Real.sqrt ((x + 1)^2 + 9))

theorem min_value_of_f :
  ∃ (x : ℝ), f x = 5 * Real.sqrt 2 := sorry

end min_value_of_f_l1412_141279


namespace germs_left_after_sprays_l1412_141215

-- Define the percentages as real numbers
def S1 : ℝ := 0.50 -- 50%
def S2 : ℝ := 0.35 -- 35%
def S3 : ℝ := 0.20 -- 20%
def S4 : ℝ := 0.10 -- 10%

-- Define the overlaps as real numbers
def overlap12 : ℝ := 0.10 -- between S1 and S2
def overlap23 : ℝ := 0.07 -- between S2 and S3
def overlap34 : ℝ := 0.05 -- between S3 and S4
def overlap13 : ℝ := 0.03 -- between S1 and S3
def overlap14 : ℝ := 0.02 -- between S1 and S4

theorem germs_left_after_sprays :
  let total_killed := S1 + S2 + S3 + S4
  let total_overlap := overlap12 + overlap23 + overlap34 + overlap13 + overlap14
  let adjusted_overlap := overlap12 + overlap23 + overlap34
  let effective_killed := total_killed - adjusted_overlap
  let percentage_left := 1.0 - effective_killed
  percentage_left = 0.07 := by
  -- proof steps to be inserted here
  sorry

end germs_left_after_sprays_l1412_141215


namespace gamma_received_eight_donuts_l1412_141218

noncomputable def total_donuts : ℕ := 40
noncomputable def delta_donuts : ℕ := 8
noncomputable def remaining_donuts : ℕ := total_donuts - delta_donuts
noncomputable def gamma_donuts : ℕ := 8
noncomputable def beta_donuts : ℕ := 3 * gamma_donuts

theorem gamma_received_eight_donuts 
  (h1 : total_donuts = 40)
  (h2 : delta_donuts = 8)
  (h3 : beta_donuts = 3 * gamma_donuts)
  (h4 : remaining_donuts = total_donuts - delta_donuts)
  (h5 : remaining_donuts = gamma_donuts + beta_donuts) :
  gamma_donuts = 8 := 
sorry

end gamma_received_eight_donuts_l1412_141218


namespace infinitely_many_arithmetic_progression_triples_l1412_141238

theorem infinitely_many_arithmetic_progression_triples :
  ∃ (u v: ℤ) (a b c: ℤ), 
  (∀ n: ℤ, (a = 2 * u) ∧ 
    (b = 2 * u + v) ∧
    (c = 2 * u + 2 * v) ∧ 
    (u > 0) ∧
    (v > 0) ∧
    ∃ k m n: ℤ, 
    (a * b + 1 = k * k) ∧ 
    (b * c + 1 = m * m) ∧ 
    (c * a + 1 = n * n)) :=
sorry

end infinitely_many_arithmetic_progression_triples_l1412_141238


namespace hannah_payment_l1412_141235

def costWashingMachine : ℝ := 100
def costDryer : ℝ := costWashingMachine - 30
def totalCostBeforeDiscount : ℝ := costWashingMachine + costDryer
def discount : ℝ := totalCostBeforeDiscount * 0.1
def finalCost : ℝ := totalCostBeforeDiscount - discount

theorem hannah_payment : finalCost = 153 := by
  simp [costWashingMachine, costDryer, totalCostBeforeDiscount, discount, finalCost]
  sorry

end hannah_payment_l1412_141235


namespace david_distance_to_airport_l1412_141203

theorem david_distance_to_airport (t : ℝ) (d : ℝ) :
  (35 * (t + 1) = d) ∧ (d - 35 = 50 * (t - 1.5)) → d = 210 :=
by
  sorry

end david_distance_to_airport_l1412_141203


namespace sum_of_operations_l1412_141211

noncomputable def triangle (a b c : ℕ) : ℕ :=
  a + 2 * b - c

theorem sum_of_operations :
  triangle 3 5 7 + triangle 6 1 8 = 6 :=
by
  sorry

end sum_of_operations_l1412_141211


namespace area_comparison_l1412_141208

-- Define the side lengths of the triangles
def a₁ := 17
def b₁ := 17
def c₁ := 12

def a₂ := 17
def b₂ := 17
def c₂ := 16

-- Define the semiperimeters
def s₁ := (a₁ + b₁ + c₁) / 2
def s₂ := (a₂ + b₂ + c₂) / 2

-- Define the areas using Heron's formula
noncomputable def area₁ := (s₁ * (s₁ - a₁) * (s₁ - b₁) * (s₁ - c₁)).sqrt
noncomputable def area₂ := (s₂ * (s₂ - a₂) * (s₂ - b₂) * (s₂ - c₂)).sqrt

-- The theorem to prove
theorem area_comparison : area₁ < area₂ := sorry

end area_comparison_l1412_141208


namespace solve_for_t_l1412_141209

theorem solve_for_t (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : x = y → t = 1/2 :=
by
  sorry

end solve_for_t_l1412_141209


namespace square_difference_l1412_141253

theorem square_difference (x : ℤ) (h : x^2 = 1444) : (x + 1) * (x - 1) = 1443 := 
by 
  sorry

end square_difference_l1412_141253


namespace third_quadrant_to_first_third_fourth_l1412_141246

theorem third_quadrant_to_first_third_fourth (k : ℤ) (α : ℝ) 
  (h : 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2) : 
  ∃ n : ℤ, (2 * k / 3 % 2) * Real.pi + Real.pi / 3 < α / 3 ∧ α / 3 < (2 * k / 3 % 2) * Real.pi + Real.pi / 2 ∨
            (2 * (3 * n + 1) % 2) * Real.pi + Real.pi < α / 3 ∧ α / 3 < (2 * (3 * n + 1) % 2) * Real.pi + 7 * Real.pi / 6 ∨
            (2 * (3 * n + 2) % 2) * Real.pi + 5 * Real.pi / 3 < α / 3 ∧ α / 3 < (2 * (3 * n + 2) % 2) * Real.pi + 11 * Real.pi / 6 :=
sorry

end third_quadrant_to_first_third_fourth_l1412_141246


namespace dig_site_date_l1412_141262

theorem dig_site_date (S F T Fourth : ℤ) 
  (h₁ : F = S - 352)
  (h₂ : T = F + 3700)
  (h₃ : Fourth = 2 * T)
  (h₄ : Fourth = 8400) : S = 852 := 
by 
  sorry

end dig_site_date_l1412_141262


namespace total_snowballs_l1412_141270

theorem total_snowballs (Lc : ℕ) (Ch : ℕ) (Pt : ℕ)
  (h1 : Ch = Lc + 31)
  (h2 : Lc = 19)
  (h3 : Pt = 47) : 
  Ch + Lc + Pt = 116 := by
  sorry

end total_snowballs_l1412_141270


namespace ratio_of_fallen_cakes_is_one_half_l1412_141247

noncomputable def ratio_fallen_to_total (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ) :=
  fallen_cakes / total_cakes

theorem ratio_of_fallen_cakes_is_one_half :
  ∀ (total_cakes fallen_cakes pick_up destroyed_cakes : ℕ),
    total_cakes = 12 →
    pick_up = fallen_cakes / 2 →
    pick_up = destroyed_cakes →
    destroyed_cakes = 3 →
    ratio_fallen_to_total total_cakes fallen_cakes pick_up destroyed_cakes = 1 / 2 :=
by
  intros total_cakes fallen_cakes pick_up destroyed_cakes h1 h2 h3 h4
  rw [h1, h4, ratio_fallen_to_total]
  -- proof goes here
  sorry

end ratio_of_fallen_cakes_is_one_half_l1412_141247


namespace monthly_incomes_l1412_141271

theorem monthly_incomes (a b c d e : ℕ) : 
  a + b = 8100 ∧ 
  b + c = 10500 ∧ 
  a + c = 8400 ∧
  (a + b + d) / 3 = 4800 ∧
  (c + d + e) / 3 = 6000 ∧
  (b + a + e) / 3 = 4500 → 
  (a = 3000 ∧ b = 5100 ∧ c = 5400 ∧ d = 6300 ∧ e = 5400) :=
by sorry

end monthly_incomes_l1412_141271


namespace a3_value_l1412_141220

-- Define the geometric sequence
def geom_seq (r : ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * r ^ n

-- Given conditions
variables (a : ℕ → ℝ) (r : ℝ)
axiom h_geom : geom_seq r a
axiom h_a1 : a 1 = 1
axiom h_a5 : a 5 = 4

-- Goal to prove
theorem a3_value : a 3 = 2 ∨ a 3 = -2 := by
  sorry

end a3_value_l1412_141220


namespace neil_initial_games_l1412_141250

theorem neil_initial_games (N : ℕ) 
  (H₀ : ℕ) (H₀_eq : H₀ = 58)
  (H₁ : ℕ) (H₁_eq : H₁ = H₀ - 6)
  (H₁_condition : H₁ = 4 * (N + 6)) : N = 7 :=
by {
  -- Substituting the given values and simplifying to show the final equation
  sorry
}

end neil_initial_games_l1412_141250


namespace emberly_total_miles_l1412_141261

noncomputable def totalMilesWalkedInMarch : ℕ :=
  let daysInMarch := 31
  let daysNotWalked := 4
  let milesPerDay := 4
  (daysInMarch - daysNotWalked) * milesPerDay

theorem emberly_total_miles : totalMilesWalkedInMarch = 108 :=
by
  sorry

end emberly_total_miles_l1412_141261


namespace digit_is_two_l1412_141224

theorem digit_is_two (d : ℕ) (h : d < 10) : (∃ k : ℤ, d - 2 = 11 * k) ↔ d = 2 := 
by sorry

end digit_is_two_l1412_141224


namespace least_positive_angle_l1412_141210

theorem least_positive_angle (θ : ℝ) (h : Real.cos (10 * Real.pi / 180) = Real.sin (15 * Real.pi / 180) + Real.sin θ) :
  θ = 32.5 * Real.pi / 180 := 
sorry

end least_positive_angle_l1412_141210


namespace parabola_above_line_l1412_141255

variable {a b c : ℝ}

theorem parabola_above_line
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (H : (b - c) ^ 2 - 4 * a * c < 0) :
  (b + c) ^ 2 - 4 * c * (a + b) < 0 := 
sorry

end parabola_above_line_l1412_141255


namespace expand_product_l1412_141234

theorem expand_product :
  (3 * x + 4) * (x - 2) * (x + 6) = 3 * x^3 + 16 * x^2 - 20 * x - 48 :=
by
  sorry

end expand_product_l1412_141234


namespace max_lessons_l1412_141223

theorem max_lessons (x y z : ℕ) (h1 : y * z = 6) (h2 : x * z = 21) (h3 : x * y = 14) : 3 * x * y * z = 126 :=
sorry

end max_lessons_l1412_141223


namespace units_digit_base7_product_l1412_141225

theorem units_digit_base7_product (a b : ℕ) (ha : a = 354) (hb : b = 78) : (a * b) % 7 = 4 := by
  sorry

end units_digit_base7_product_l1412_141225


namespace sum_geometric_series_l1412_141288

noncomputable def S_n (n : ℕ) : ℝ :=
  3 - 3 * ((2 / 3)^n)

theorem sum_geometric_series (a : ℝ) (r : ℝ) (n : ℕ) (h_a : a = 1) (h_r : r = 2 / 3) :
  S_n n = a * (1 - r^n) / (1 - r) :=
by
  sorry

end sum_geometric_series_l1412_141288


namespace slower_speed_is_35_l1412_141254

-- Define the given conditions
def distance : ℝ := 70 -- distance is 70 km
def speed_on_time : ℝ := 40 -- on-time average speed is 40 km/hr
def delay : ℝ := 0.25 -- delay is 15 minutes or 0.25 hours

-- This is the statement we need to prove
theorem slower_speed_is_35 :
  ∃ slower_speed : ℝ, 
    slower_speed = distance / (distance / speed_on_time + delay) ∧ slower_speed = 35 :=
by
  sorry

end slower_speed_is_35_l1412_141254


namespace factor_expression_l1412_141217

theorem factor_expression (x : ℝ) : 
  5 * x * (x - 2) + 9 * (x - 2) - 4 * (x - 2) = 5 * (x - 2) * (x + 1) :=
by
  -- proof goes here
  sorry

end factor_expression_l1412_141217


namespace at_least_one_zero_l1412_141206

theorem at_least_one_zero (a b : ℝ) : (¬ (a ≠ 0 ∧ b ≠ 0)) → (a = 0 ∨ b = 0) := by
  intro h
  have h' : ¬ ((a ≠ 0) ∧ (b ≠ 0)) := h
  sorry

end at_least_one_zero_l1412_141206


namespace minimum_questions_needed_to_determine_birthday_l1412_141278

def min_questions_to_determine_birthday : Nat := 9

theorem minimum_questions_needed_to_determine_birthday : min_questions_to_determine_birthday = 9 :=
sorry

end minimum_questions_needed_to_determine_birthday_l1412_141278


namespace graph_single_point_l1412_141256

theorem graph_single_point (c : ℝ) : 
  (∃ x y : ℝ, ∀ (x' y' : ℝ), 4 * x'^2 + y'^2 + 16 * x' - 6 * y' + c = 0 → (x' = x ∧ y' = y)) → c = 7 := 
by
  sorry

end graph_single_point_l1412_141256


namespace books_needed_to_buy_clarinet_l1412_141221

def cost_of_clarinet : ℕ := 90
def initial_savings : ℕ := 10
def price_per_book : ℕ := 5
def halfway_loss : ℕ := (cost_of_clarinet - initial_savings) / 2

theorem books_needed_to_buy_clarinet 
    (cost_of_clarinet initial_savings price_per_book halfway_loss : ℕ)
    (initial_savings_lost : halfway_loss = (cost_of_clarinet - initial_savings) / 2) : 
    ((cost_of_clarinet - initial_savings + halfway_loss) / price_per_book) = 24 := 
sorry

end books_needed_to_buy_clarinet_l1412_141221


namespace time_saved_correct_l1412_141241

-- Define the conditions as constants
def section1_problems : Nat := 20
def section2_problems : Nat := 15

def time_with_calc_sec1 : Nat := 3
def time_without_calc_sec1 : Nat := 8

def time_with_calc_sec2 : Nat := 5
def time_without_calc_sec2 : Nat := 10

-- Calculate the total times
def total_time_with_calc : Nat :=
  (section1_problems * time_with_calc_sec1) +
  (section2_problems * time_with_calc_sec2)

def total_time_without_calc : Nat :=
  (section1_problems * time_without_calc_sec1) +
  (section2_problems * time_without_calc_sec2)

-- The time saved using a calculator
def time_saved : Nat :=
  total_time_without_calc - total_time_with_calc

-- State the proof problem
theorem time_saved_correct :
  time_saved = 175 := by
  sorry

end time_saved_correct_l1412_141241


namespace inequality_has_no_solution_l1412_141281

theorem inequality_has_no_solution (x : ℝ) : -x^2 + 2*x - 2 > 0 → false :=
by
  sorry

end inequality_has_no_solution_l1412_141281


namespace consecutive_ints_prod_square_l1412_141291

theorem consecutive_ints_prod_square (n : ℤ) : 
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = k^2 :=
sorry

end consecutive_ints_prod_square_l1412_141291


namespace expand_expression_l1412_141280

theorem expand_expression : 
  (5 * x^2 + 2 * x - 3) * (3 * x^3 - x^2) = 15 * x^5 + x^4 - 11 * x^3 + 3 * x^2 := 
by
  sorry

end expand_expression_l1412_141280


namespace hats_needed_to_pay_51_l1412_141252

def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def jeans_cost : ℕ := 10
def total_amount : ℕ := 51
def num_shirts : ℕ := 3
def num_jeans : ℕ := 2

theorem hats_needed_to_pay_51 :
  ∃ (n : ℕ), total_amount = num_shirts * shirt_cost + num_jeans * jeans_cost + n * hat_cost ∧ n = 4 :=
by
  sorry

end hats_needed_to_pay_51_l1412_141252


namespace other_root_is_seven_thirds_l1412_141268

theorem other_root_is_seven_thirds {m : ℝ} (h : ∃ r : ℝ, 3 * r * r + m * r - 7 = 0 ∧ r = -1) : 
  ∃ r' : ℝ, r' ≠ -1 ∧ 3 * r' * r' + m * r' - 7 = 0 ∧ r' = 7 / 3 :=
by
  sorry

end other_root_is_seven_thirds_l1412_141268


namespace max_min_of_f_on_interval_l1412_141275

noncomputable def f (x : ℝ) : ℝ := 3 * x ^ 4 + 4 * x ^ 3 + 34

theorem max_min_of_f_on_interval :
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f x ≤ 50) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 50) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, 33 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 1, f x = 33) :=
by
  sorry

end max_min_of_f_on_interval_l1412_141275


namespace inequality_not_always_correct_l1412_141201

variables (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x > y) (h₄ : z > 0)

theorem inequality_not_always_correct :
  ¬ ∀ z > 0, (xz^2 / z > yz^2 / z) :=
sorry

end inequality_not_always_correct_l1412_141201


namespace bus_interval_l1412_141219

theorem bus_interval (num_departures : ℕ) (total_duration : ℕ) (interval : ℕ)
  (h1 : num_departures = 11)
  (h2 : total_duration = 60)
  (h3 : interval = total_duration / (num_departures - 1)) :
  interval = 6 :=
by
  sorry

end bus_interval_l1412_141219


namespace find_minimum_value_of_f_l1412_141299

def f (x : ℝ) : ℝ := (x ^ 2 + 4 * x + 5) * (x ^ 2 + 4 * x + 2) + 2 * x ^ 2 + 8 * x + 1

theorem find_minimum_value_of_f : ∃ x : ℝ, f x = -9 :=
by
  sorry

end find_minimum_value_of_f_l1412_141299


namespace adam_has_more_apples_l1412_141285

-- Define the number of apples Jackie has
def JackiesApples : Nat := 9

-- Define the number of apples Adam has
def AdamsApples : Nat := 14

-- Statement of the problem: Prove that Adam has 5 more apples than Jackie
theorem adam_has_more_apples :
  AdamsApples - JackiesApples = 5 :=
by
  sorry

end adam_has_more_apples_l1412_141285


namespace jason_initial_pears_l1412_141273

-- Define the initial number of pears Jason picked.
variable (P : ℕ)

-- Conditions translated to Lean:
-- Jason gave Keith 47 pears and received 12 from Mike, leaving him with 11 pears.
variable (h1 : P - 47 + 12 = 11)

-- The theorem stating the problem:
theorem jason_initial_pears : P = 46 :=
by
  sorry

end jason_initial_pears_l1412_141273


namespace constant_term_binomial_l1412_141295

theorem constant_term_binomial (n : ℕ) (h : n = 5) : ∃ (r : ℕ), r = 6 ∧ (Nat.choose (2 * n) r) = 210 := by
  sorry

end constant_term_binomial_l1412_141295


namespace unique_fraction_difference_l1412_141239

theorem unique_fraction_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  (1 / x) - (1 / y) = (y - x) / (x * y) :=
by sorry

end unique_fraction_difference_l1412_141239


namespace simplify_expression1_simplify_expression2_l1412_141251

theorem simplify_expression1 (x : ℝ) : 
  5*x^2 + x + 3 + 4*x - 8*x^2 - 2 = -3*x^2 + 5*x + 1 :=
by
  sorry

theorem simplify_expression2 (a : ℝ) : 
  (5*a^2 + 2*a - 1) - 4*(3 - 8*a + 2*a^2) = -3*a^2 + 34*a - 13 :=
by
  sorry

end simplify_expression1_simplify_expression2_l1412_141251


namespace inverse_proposition_l1412_141257

theorem inverse_proposition (a b : ℝ) (h1 : a < 1) (h2 : b < 1) : a + b ≠ 2 :=
by sorry

end inverse_proposition_l1412_141257


namespace total_amount_245_l1412_141212

-- Define the conditions and the problem
theorem total_amount_245 (a : ℝ) (x y z : ℝ) (h1 : y = 0.45 * a) (h2 : z = 0.30 * a) (h3 : y = 63) :
  x + y + z = 245 := 
by
  -- Starting the proof (proof steps are unnecessary as per the procedure)
  sorry

end total_amount_245_l1412_141212


namespace soccer_team_points_l1412_141228

theorem soccer_team_points
  (x y : ℕ)
  (h1 : x + y = 8)
  (h2 : 3 * x - y = 12) : 
  (x + y = 8 ∧ 3 * x - y = 12) :=
by
  exact ⟨h1, h2⟩

end soccer_team_points_l1412_141228


namespace product_of_solutions_l1412_141292

theorem product_of_solutions : (∃ x : ℝ, |x| = 3*(|x| - 2)) → (x = 3 ∨ x = -3) → 3 * -3 = -9 :=
by sorry

end product_of_solutions_l1412_141292


namespace abc_divisibility_l1412_141237

theorem abc_divisibility (a b c : ℕ) (h1 : a^2 * b ∣ a^3 + b^3 + c^3) (h2 : b^2 * c ∣ a^3 + b^3 + c^3) (h3 : c^2 * a ∣ a^3 + b^3 + c^3) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end abc_divisibility_l1412_141237


namespace negation_proof_l1412_141240

theorem negation_proof : 
  (¬(∀ x : ℝ, x < 2^x) ↔ ∃ x : ℝ, x ≥ 2^x) :=
by
  sorry

end negation_proof_l1412_141240


namespace gcd_poly_correct_l1412_141258

-- Define the conditions
def is_even_multiple_of (x k : ℕ) : Prop :=
  ∃ (n : ℕ), x = k * 2 * n

variable (b : ℕ)

-- Given condition
axiom even_multiple_7768 : is_even_multiple_of b 7768

-- Define the polynomials
def poly1 (b : ℕ) := 4 * b * b + 37 * b + 72
def poly2 (b : ℕ) := 3 * b + 8

-- Proof statement
theorem gcd_poly_correct : gcd (poly1 b) (poly2 b) = 8 :=
  sorry

end gcd_poly_correct_l1412_141258


namespace totalMilkConsumption_l1412_141296

-- Conditions
def regularMilk (week: ℕ) : ℝ := 0.5
def soyMilk (week: ℕ) : ℝ := 0.1

-- Theorem statement
theorem totalMilkConsumption : regularMilk 1 + soyMilk 1 = 0.6 := 
by 
  sorry

end totalMilkConsumption_l1412_141296


namespace smallest_base_to_express_100_with_three_digits_l1412_141243

theorem smallest_base_to_express_100_with_three_digits : 
  ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ b' : ℕ, (b'^2 ≤ 100 ∧ 100 < b'^3) → b ≤ b' ∧ b = 5 :=
by
  sorry

end smallest_base_to_express_100_with_three_digits_l1412_141243


namespace arithmetic_mean_eqn_l1412_141229

theorem arithmetic_mean_eqn : 
  (3/5 + 6/7) / 2 = 51/70 :=
  by sorry

end arithmetic_mean_eqn_l1412_141229


namespace find_y_l1412_141202

theorem find_y (x y : ℤ)
  (h1 : (100 + 200300 + x) / 3 = 250)
  (h2 : (300 + 150100 + x + y) / 4 = 200) :
  y = -4250 :=
sorry

end find_y_l1412_141202


namespace fran_speed_l1412_141233

theorem fran_speed :
  ∀ (Joann_speed Fran_time : ℝ), Joann_speed = 15 → Joann_time = 4 → Fran_time = 3.5 →
    (Joann_speed * Joann_time = Fran_time * (120 / 7)) :=
by
  intro Joann_speed Joann_time Fran_time
  sorry

end fran_speed_l1412_141233


namespace problem_solved_probability_l1412_141294

theorem problem_solved_probability :
  let PA := 1 / 2
  let PB := 1 / 3
  let PC := 1 / 4
  1 - ((1 - PA) * (1 - PB) * (1 - PC)) = 3 / 4 := 
sorry

end problem_solved_probability_l1412_141294


namespace remainder_when_divided_by_15_l1412_141282

theorem remainder_when_divided_by_15 (N : ℕ) (k : ℤ) (h1 : N = 60 * k + 49) : (N % 15) = 4 :=
sorry

end remainder_when_divided_by_15_l1412_141282


namespace arccos_proof_l1412_141287

noncomputable def arccos_identity : Prop := 
  ∃ x : ℝ, x = 1 / Real.sqrt 2 ∧ Real.arccos x = Real.pi / 4

theorem arccos_proof : arccos_identity :=
by
  sorry

end arccos_proof_l1412_141287


namespace digits_are_different_probability_l1412_141269

noncomputable def prob_diff_digits : ℚ :=
  let total := 999 - 100 + 1
  let same_digits := 9
  1 - (same_digits / total)

theorem digits_are_different_probability :
  prob_diff_digits = 99 / 100 :=
by
  sorry

end digits_are_different_probability_l1412_141269


namespace total_lives_l1412_141244

-- Defining the number of lives for each animal according to the given conditions:
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := dog_lives + 7
def elephant_lives : ℕ := 2 * cat_lives - 5
def fish_lives : ℕ := if (dog_lives + mouse_lives) < (elephant_lives / 2) then (dog_lives + mouse_lives) else elephant_lives / 2

-- The main statement we need to prove:
theorem total_lives :
  cat_lives + dog_lives + mouse_lives + elephant_lives + fish_lives = 47 :=
by
  sorry

end total_lives_l1412_141244


namespace drug_price_reduction_eq_l1412_141222

variable (x : ℝ)
variable (initial_price : ℝ := 144)
variable (final_price : ℝ := 81)

theorem drug_price_reduction_eq :
  initial_price * (1 - x)^2 = final_price :=
by
  sorry

end drug_price_reduction_eq_l1412_141222


namespace cashier_amount_l1412_141297

def amount_to_cashier (discount : ℝ) (shorts_count : ℕ) (shorts_price : ℕ) (shirts_count : ℕ) (shirts_price : ℕ) : ℝ :=
  let total_cost := (shorts_count * shorts_price) + (shirts_count * shirts_price)
  let discount_amount := discount * total_cost
  total_cost - discount_amount

theorem cashier_amount : amount_to_cashier 0.1 3 15 5 17 = 117 := 
by
  sorry

end cashier_amount_l1412_141297


namespace total_blue_marbles_l1412_141293

theorem total_blue_marbles (red_Jenny blue_Jenny red_Mary blue_Mary red_Anie blue_Anie : ℕ)
  (h1: red_Jenny = 30)
  (h2: blue_Jenny = 25)
  (h3: red_Mary = 2 * red_Jenny)
  (h4: blue_Mary = blue_Anie / 2)
  (h5: red_Anie = red_Mary + 20)
  (h6: blue_Anie = 2 * blue_Jenny) :
  blue_Mary + blue_Jenny + blue_Anie = 100 :=
by
  sorry

end total_blue_marbles_l1412_141293


namespace legacy_earnings_l1412_141277

theorem legacy_earnings 
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (hours_per_room : ℕ)
  (earnings_per_hour : ℕ)
  (total_floors : floors = 4)
  (total_rooms_per_floor : rooms_per_floor = 10)
  (time_per_room : hours_per_room = 6)
  (rate_per_hour : earnings_per_hour = 15) :
  floors * rooms_per_floor * hours_per_room * earnings_per_hour = 3600 := 
by
  sorry

end legacy_earnings_l1412_141277


namespace abs_diff_of_roots_eq_one_l1412_141265

theorem abs_diff_of_roots_eq_one {p q : ℝ} (h₁ : p + q = 7) (h₂ : p * q = 12) : |p - q| = 1 := 
by 
  sorry

end abs_diff_of_roots_eq_one_l1412_141265


namespace inequality_proof_l1412_141272

theorem inequality_proof (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 ≥ a2) (h2 : a2 ≥ a3) (h3 : a3 > 0) 
  (h4 : b1 ≥ b2) (h5 : b2 ≥ b3) (h6 : b3 > 0)
  (h7 : a1 * a2 * a3 = b1 * b2 * b3)
  (h8 : a1 - a3 ≤ b1 - b3) : 
  a1 + a2 + a3 ≤ 2 * (b1 + b2 + b3) := sorry

end inequality_proof_l1412_141272


namespace correct_calculation_is_d_l1412_141286

theorem correct_calculation_is_d :
  (-7) + (-7) ≠ 0 ∧
  ((-1 / 10) - (1 / 10)) ≠ 0 ∧
  (0 + (-101)) ≠ 101 ∧
  (1 / 3 + -1 / 2 = -1 / 6) :=
by
  sorry

end correct_calculation_is_d_l1412_141286
