import Mathlib

namespace find_theta_l2323_232387

theorem find_theta (θ : Real) (h : abs θ < π / 2) (h_eq : Real.sin (π + θ) = -Real.sqrt 3 * Real.cos (2 * π - θ)) :
  θ = π / 3 :=
sorry

end find_theta_l2323_232387


namespace problem_l2323_232302

theorem problem (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, f (x + 2) = -f x) :
  f 4 = 0 ∧ (∀ x, f (x + 4) = f x) ∧ (∀ x, f (2 - x) = f (2 + x)) :=
sorry

end problem_l2323_232302


namespace find_cosine_of_dihedral_angle_l2323_232384

def dihedral_cosine (R r : ℝ) (α β : ℝ) : Prop :=
  R = 2 * r ∧ β = Real.pi / 4 → Real.cos α = 8 / 9

theorem find_cosine_of_dihedral_angle : ∃ α, ∀ R r : ℝ, dihedral_cosine R r α (Real.pi / 4) :=
sorry

end find_cosine_of_dihedral_angle_l2323_232384


namespace sequence_general_term_l2323_232357

/-- 
  Define the sequence a_n recursively as:
  a_1 = 2
  a_n = 2 * a_(n-1) - 1

  Prove that the general term of the sequence is:
  a_n = 2^(n-1) + 1
-/
theorem sequence_general_term {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) 
  (n : ℕ) : 
  a n = 2^(n-1) + 1 := by
  sorry

end sequence_general_term_l2323_232357


namespace investment_at_6_percent_l2323_232333

variables (x y : ℝ)

-- Conditions from the problem
def total_investment : Prop := x + y = 15000
def total_interest : Prop := 0.06 * x + 0.075 * y = 1023

-- Conclusion to prove
def invest_6_percent (x : ℝ) : Prop := x = 6800

theorem investment_at_6_percent (h1 : total_investment x y) (h2 : total_interest x y) : invest_6_percent x :=
by
  sorry

end investment_at_6_percent_l2323_232333


namespace lenny_initial_money_l2323_232325

-- Definitions based on the conditions
def spent_on_video_games : ℕ := 24
def spent_at_grocery_store : ℕ := 21
def amount_left : ℕ := 39

-- Statement of the problem
theorem lenny_initial_money : spent_on_video_games + spent_at_grocery_store + amount_left = 84 :=
by
  sorry

end lenny_initial_money_l2323_232325


namespace max_temperature_when_80_l2323_232320

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10 * t + 60

-- State the theorem
theorem max_temperature_when_80 : ∃ t : ℝ, temperature t = 80 ∧ t = 5 + Real.sqrt 5 := 
by {
  -- Theorem proof is skipped with sorry
  sorry
}

end max_temperature_when_80_l2323_232320


namespace percent_increase_in_sales_l2323_232393

theorem percent_increase_in_sales :
  let new := 416
  let old := 320
  (new - old) / old * 100 = 30 := by
  sorry

end percent_increase_in_sales_l2323_232393


namespace total_hours_watching_tv_and_playing_games_l2323_232360

-- Defining the conditions provided in the problem
def hours_watching_tv_saturday : ℕ := 6
def hours_watching_tv_sunday : ℕ := 3
def hours_watching_tv_tuesday : ℕ := 2
def hours_watching_tv_thursday : ℕ := 4

def hours_playing_games_monday : ℕ := 3
def hours_playing_games_wednesday : ℕ := 5
def hours_playing_games_friday : ℕ := 1

-- The proof statement
theorem total_hours_watching_tv_and_playing_games :
  hours_watching_tv_saturday + hours_watching_tv_sunday + hours_watching_tv_tuesday + hours_watching_tv_thursday
  + hours_playing_games_monday + hours_playing_games_wednesday + hours_playing_games_friday = 24 := 
by
  sorry

end total_hours_watching_tv_and_playing_games_l2323_232360


namespace Alex_has_more_than_200_marbles_on_Monday_of_next_week_l2323_232346

theorem Alex_has_more_than_200_marbles_on_Monday_of_next_week :
  ∃ k : ℕ, k > 0 ∧ 3 * 2^k > 200 ∧ k % 7 = 1 := by
  sorry

end Alex_has_more_than_200_marbles_on_Monday_of_next_week_l2323_232346


namespace contradiction_in_stock_price_l2323_232313

noncomputable def stock_price_contradiction : Prop :=
  ∃ (P D : ℝ), (D = 0.20 * P) ∧ (0.10 = (D / P) * 100)

theorem contradiction_in_stock_price : ¬(stock_price_contradiction) := sorry

end contradiction_in_stock_price_l2323_232313


namespace quadratic_distinct_roots_k_range_l2323_232392

theorem quadratic_distinct_roots_k_range (k : ℝ) :
  (k - 1) * x^2 + 2 * x - 2 = 0 ∧ 
  ∀ Δ, Δ = 2^2 - 4*(k-1)*(-2) ∧ Δ > 0 ∧ (k ≠ 1) ↔ k > 1/2 ∧ k ≠ 1 :=
by
  sorry

end quadratic_distinct_roots_k_range_l2323_232392


namespace triangular_weight_is_60_l2323_232365

variable (w_round w_triangular w_rectangular : ℝ)

axiom rectangular_weight : w_rectangular = 90
axiom balance1 : w_round + w_triangular = 3 * w_round
axiom balance2 : 4 * w_round + w_triangular = w_triangular + w_round + w_rectangular

theorem triangular_weight_is_60 :
  w_triangular = 60 :=
by
  sorry

end triangular_weight_is_60_l2323_232365


namespace solution_set_inequality_l2323_232338

theorem solution_set_inequality (x : ℝ) : x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 :=
by sorry

end solution_set_inequality_l2323_232338


namespace exists_n_with_common_divisor_l2323_232348

theorem exists_n_with_common_divisor :
  ∃ (n : ℕ), ∀ (k : ℕ), (k ≤ 20) → Nat.gcd (n + k) 30030 > 1 :=
by
  sorry

end exists_n_with_common_divisor_l2323_232348


namespace angle_relation_l2323_232364

theorem angle_relation (R : ℝ) (hR : R > 0) (d : ℝ) (hd : d > R) 
  (α β : ℝ) : β = 3 * α :=
sorry

end angle_relation_l2323_232364


namespace min_score_guarantees_payoff_l2323_232323

-- Defining the probability of a single roll being a six
def prob_single_six : ℚ := 1 / 6 

-- Defining the event of rolling two sixes independently
def prob_two_sixes : ℚ := prob_single_six * prob_single_six

-- Defining the score of two die rolls summing up to 12
def is_score_twelve (a b : ℕ) : Prop := a + b = 12

-- Proving the probability of Jim scoring 12 in two rolls guarantees some monetary payoff.
theorem min_score_guarantees_payoff :
  (prob_two_sixes = 1/36) :=
by
  sorry

end min_score_guarantees_payoff_l2323_232323


namespace five_digit_palindromes_count_l2323_232350

theorem five_digit_palindromes_count : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) → 
  900 = 9 * 10 * 10 := 
by
  intro h
  sorry

end five_digit_palindromes_count_l2323_232350


namespace part_1_a_part_1_b_part_2_l2323_232390

open Set

variable (a : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | x^2 - 4 > 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def compl_U_A : Set ℝ := compl A

theorem part_1_a :
  A ∩ B 1 = {x : ℝ | x < -2} :=
by
  sorry

theorem part_1_b :
  A ∪ B 1 = {x : ℝ | x > 2 ∨ x ≤ 1} :=
by
  sorry

theorem part_2 :
  compl_U_A ⊆ B a → a ≥ 2 :=
by
  sorry

end part_1_a_part_1_b_part_2_l2323_232390


namespace total_practice_hours_l2323_232304

def schedule : List ℕ := [6, 4, 5, 7, 3]

-- We define the conditions
def total_scheduled_hours : ℕ := schedule.sum

def average_daily_practice_time (total : ℕ) : ℕ := total / schedule.length

def rainy_day_lost_hours : ℕ := average_daily_practice_time total_scheduled_hours

def player_A_missed_hours : ℕ := 2

def player_B_missed_hours : ℕ := 3

def total_missed_hours : ℕ := player_A_missed_hours + player_B_missed_hours

def total_hours_practiced : ℕ := total_scheduled_hours - (rainy_day_lost_hours + total_missed_hours)

-- Now we state the theorem we want to prove
theorem total_practice_hours : total_hours_practiced = 15 := by
  -- omitted proof
  sorry

end total_practice_hours_l2323_232304


namespace ratio_of_perimeters_l2323_232326

theorem ratio_of_perimeters (L : ℝ) (H : ℝ) (hL1 : L = 8) 
  (hH1 : H = 8) (hH2 : H = 2 * (H / 2)) (hH3 : 4 > 0) (hH4 : 0 < 4 / 3)
  (hW1 : ∀ a, a / 3 > 0 → 8 = L )
  (hPsmall : ∀ P, P = 2 * ((4 / 3) + 8) )
  (hPlarge : ∀ P, P = 2 * ((H - 4 / 3) + 8) )
  :
  (2 * ((4 / 3) + 8)) / (2 * ((8 - (4 / 3)) + 8)) = (7 / 11) := by
  sorry

end ratio_of_perimeters_l2323_232326


namespace roots_ratio_sum_l2323_232303

theorem roots_ratio_sum (α β : ℝ) (hαβ : α > β) (h1 : 3*α^2 + α - 1 = 0) (h2 : 3*β^2 + β - 1 = 0) :
  α / β + β / α = -7 / 3 :=
sorry

end roots_ratio_sum_l2323_232303


namespace max_sum_terms_arithmetic_seq_l2323_232386

theorem max_sum_terms_arithmetic_seq (a1 d : ℝ) (h1 : a1 > 0) 
  (h2 : 3 * (2 * a1 + 2 * d) = 11 * (2 * a1 + 10 * d)) :
  ∃ (n : ℕ),  (∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d > 0) ∧  a1 + n * d ≤ 0 ∧ n = 7 :=
by
  sorry

end max_sum_terms_arithmetic_seq_l2323_232386


namespace g_at_3_l2323_232331

def g (x : ℝ) : ℝ := -3 * x^4 + 4 * x^3 - 7 * x^2 + 5 * x - 2

theorem g_at_3 : g 3 = -185 := by
  sorry

end g_at_3_l2323_232331


namespace min_S_value_l2323_232351

noncomputable def S (x y z : ℝ) : ℝ := (1 + z) / (2 * x * y * z)

theorem min_S_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = 1) :
  S x y z ≥ 4 := 
sorry

end min_S_value_l2323_232351


namespace algebra_inequality_l2323_232311

theorem algebra_inequality (a b c : ℝ) 
  (H : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
sorry

end algebra_inequality_l2323_232311


namespace smallest_x_value_min_smallest_x_value_l2323_232355

noncomputable def smallest_x_not_defined : ℝ := ( 47 - (Real.sqrt 2041) ) / 12

theorem smallest_x_value :
  ∀ x : ℝ, (6 * x^2 - 47 * x + 7 = 0) → x = smallest_x_not_defined ∨ (x = (47 + (Real.sqrt 2041)) / 12) :=
sorry

theorem min_smallest_x_value :
  smallest_x_not_defined < (47 + (Real.sqrt 2041)) / 12 :=
sorry

end smallest_x_value_min_smallest_x_value_l2323_232355


namespace partial_fraction_sum_inverse_l2323_232327

theorem partial_fraction_sum_inverse (p q r A B C : ℝ)
  (hroots : (∀ s, s^3 - 20 * s^2 + 96 * s - 91 = (s - p) * (s - q) * (s - r)))
  (hA : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20 * s^2 + 96 * s - 91) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 225 :=
sorry

end partial_fraction_sum_inverse_l2323_232327


namespace remainder_76_pow_77_mod_7_l2323_232339

/-- Statement of the problem:
Prove that the remainder of \(76^{77}\) divided by 7 is 6.
-/
theorem remainder_76_pow_77_mod_7 :
  (76 ^ 77) % 7 = 6 := 
by
  sorry

end remainder_76_pow_77_mod_7_l2323_232339


namespace volume_not_occupied_by_cones_l2323_232340

/-- Two cones with given dimensions are enclosed in a cylinder, and we want to find the volume 
    in the cylinder not occupied by the cones. -/
theorem volume_not_occupied_by_cones : 
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  volume_cylinder - total_volume_cones = (2600 / 3) * π :=
by
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  sorry

end volume_not_occupied_by_cones_l2323_232340


namespace height_of_scale_model_eq_29_l2323_232334

def empireStateBuildingHeight : ℕ := 1454

def scaleRatio : ℕ := 50

def scaleModelHeight (actualHeight : ℕ) (ratio : ℕ) : ℤ :=
  Int.ofNat actualHeight / ratio

theorem height_of_scale_model_eq_29 : scaleModelHeight empireStateBuildingHeight scaleRatio = 29 :=
by
  -- Proof would go here
  sorry

end height_of_scale_model_eq_29_l2323_232334


namespace equivalent_statements_l2323_232376

-- Definitions
variables (P Q : Prop)

-- Original statement
def original_statement := P → Q

-- Statements
def statement_I := P → Q
def statement_II := Q → P
def statement_III := ¬ Q → ¬ P
def statement_IV := ¬ P ∨ Q

-- Proof problem
theorem equivalent_statements : 
  (statement_III P Q ∧ statement_IV P Q) ↔ original_statement P Q :=
sorry

end equivalent_statements_l2323_232376


namespace overtaking_time_l2323_232306

theorem overtaking_time (t_a t_b t_k : ℝ) (t_b_start : t_b = t_a - 5) 
                       (overtake_eq1 : 40 * t_b = 30 * t_a)
                       (overtake_eq2 : 60 * (t_a - 10) = 30 * t_a) :
                       t_b = 15 :=
by
  sorry

end overtaking_time_l2323_232306


namespace car_Z_probability_l2323_232335

theorem car_Z_probability :
  let P_X := 1/6
  let P_Y := 1/10
  let P_XYZ := 0.39166666666666666
  ∃ P_Z : ℝ, P_X + P_Y + P_Z = P_XYZ ∧ P_Z = 0.125 :=
by
  sorry

end car_Z_probability_l2323_232335


namespace min_distance_sum_l2323_232343

theorem min_distance_sum (x : ℝ) : 
  ∃ y, y = |x + 1| + 2 * |x - 5| + |2 * x - 7| + |(x - 11) / 2| ∧ y = 45 / 8 :=
sorry

end min_distance_sum_l2323_232343


namespace balance_force_l2323_232329

structure Vector2D where
  x : ℝ
  y : ℝ

def F1 : Vector2D := ⟨1, 1⟩
def F2 : Vector2D := ⟨2, 3⟩

def vector_add (a b : Vector2D) : Vector2D := ⟨a.x + b.x, a.y + b.y⟩
def vector_neg (a : Vector2D) : Vector2D := ⟨-a.x, -a.y⟩

theorem balance_force : 
  ∃ F3 : Vector2D, vector_add (vector_add F1 F2) F3 = ⟨0, 0⟩ ∧ F3 = ⟨-3, -4⟩ := 
by
  sorry

end balance_force_l2323_232329


namespace solution_is_option_C_l2323_232361

-- Define the equation.
def equation (x y : ℤ) : Prop := x - 2 * y = 3

-- Define the given conditions as terms in Lean.
def option_A := (1, 1)   -- (x = 1, y = 1)
def option_B := (-1, 1)  -- (x = -1, y = 1)
def option_C := (1, -1)  -- (x = 1, y = -1)
def option_D := (-1, -1) -- (x = -1, y = -1)

-- The goal is to prove that option C is a solution to the equation.
theorem solution_is_option_C : equation 1 (-1) :=
by {
  -- Proof will go here
  sorry
}

end solution_is_option_C_l2323_232361


namespace digit_B_for_divisibility_l2323_232397

theorem digit_B_for_divisibility (B : ℕ) (h : (40000 + 1000 * B + 100 * B + 20 + 6) % 7 = 0) : B = 1 :=
sorry

end digit_B_for_divisibility_l2323_232397


namespace kendra_shirts_needed_l2323_232363

def school_shirts_per_week : Nat := 5
def club_shirts_per_week : Nat := 3
def spirit_day_shirt_per_week : Nat := 1
def saturday_shirts_per_week : Nat := 3
def sunday_shirts_per_week : Nat := 3
def family_reunion_shirt_per_month : Nat := 1

def total_shirts_needed_per_week : Nat :=
  school_shirts_per_week + club_shirts_per_week + spirit_day_shirt_per_week +
  saturday_shirts_per_week + sunday_shirts_per_week

def total_shirts_needed_per_four_weeks : Nat :=
  total_shirts_needed_per_week * 4 + family_reunion_shirt_per_month

theorem kendra_shirts_needed : total_shirts_needed_per_four_weeks = 61 := by
  sorry

end kendra_shirts_needed_l2323_232363


namespace ticket_cost_difference_l2323_232316

theorem ticket_cost_difference (num_prebuy : ℕ) (price_prebuy : ℕ) (num_gate : ℕ) (price_gate : ℕ)
  (h_prebuy : num_prebuy = 20) (h_price_prebuy : price_prebuy = 155)
  (h_gate : num_gate = 30) (h_price_gate : price_gate = 200) :
  num_gate * price_gate - num_prebuy * price_prebuy = 2900 :=
by
  sorry

end ticket_cost_difference_l2323_232316


namespace percentage_discount_l2323_232391

theorem percentage_discount (individual_payment_without_discount final_payment discount_per_person : ℝ)
  (h1 : 3 * individual_payment_without_discount = final_payment + 3 * discount_per_person)
  (h2 : discount_per_person = 4)
  (h3 : final_payment = 48) :
  discount_per_person / (individual_payment_without_discount * 3) * 100 = 20 :=
by
  -- Proof to be provided here
  sorry

end percentage_discount_l2323_232391


namespace find_value_of_a_l2323_232374

theorem find_value_of_a (a : ℚ) (h : a + a / 4 - 1 / 2 = 2) : a = 2 :=
by
  sorry

end find_value_of_a_l2323_232374


namespace number_of_beetles_in_sixth_jar_l2323_232307

theorem number_of_beetles_in_sixth_jar :
  ∃ (x : ℕ), 
      (x + (x+1) + (x+2) + (x+3) + (x+4) + (x+5) + (x+6) + (x+7) + (x+8) + (x+9) = 150) ∧
      (2 * x ≥ x + 9) ∧
      (x + 5 = 16) :=
by {
  -- This is just the statement, the proof steps are ommited.
  -- You can fill in the proof here using Lean tactics as necessary.
  sorry
}

end number_of_beetles_in_sixth_jar_l2323_232307


namespace oxygen_atom_diameter_in_scientific_notation_l2323_232336

theorem oxygen_atom_diameter_in_scientific_notation :
  0.000000000148 = 1.48 * 10^(-10) :=
sorry

end oxygen_atom_diameter_in_scientific_notation_l2323_232336


namespace find_x_l2323_232322

variable (x : ℝ)

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 9 * x + 6

theorem find_x : delta (phi x) = 23 → x = -1 / 6 := by
  intro h
  sorry

end find_x_l2323_232322


namespace train_length_l2323_232347

variable (L_train : ℝ)
variable (speed_kmhr : ℝ := 45)
variable (time_seconds : ℝ := 30)
variable (bridge_length_m : ℝ := 275)
variable (train_speed_ms : ℝ := speed_kmhr * (1000 / 3600))
variable (total_distance : ℝ := train_speed_ms * time_seconds)

theorem train_length
  (h_total : total_distance = L_train + bridge_length_m) :
  L_train = 100 :=
by 
  sorry

end train_length_l2323_232347


namespace part_I_part_II_l2323_232372

-- Definitions of the sets A, B, and C
def A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 3 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m }

-- Proof statements
theorem part_I : A ∩ B = { x | 3 ≤ x ∧ x ≤ 6 } :=
by sorry

theorem part_II (m : ℝ) : (B ∪ C m = B) → (m ≤ 3) :=
by sorry

end part_I_part_II_l2323_232372


namespace trig_identity_l2323_232398

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := by
  sorry

end trig_identity_l2323_232398


namespace length_of_diagonal_l2323_232344

theorem length_of_diagonal (area : ℝ) (h1 h2 : ℝ) (d : ℝ) 
  (h_area : area = 75)
  (h_offsets : h1 = 6 ∧ h2 = 4) :
  d = 15 :=
by
  -- Given the conditions and formula, we can conclude
  sorry

end length_of_diagonal_l2323_232344


namespace project_profit_starts_from_4th_year_l2323_232389

def initial_investment : ℝ := 144
def maintenance_cost (n : ℕ) : ℝ := 4 * n^2 + 40 * n
def annual_income : ℝ := 100

def net_profit (n : ℕ) : ℝ := 
  annual_income * n - maintenance_cost n - initial_investment

theorem project_profit_starts_from_4th_year :
  ∀ n : ℕ, 3 < n ∧ n < 12 → net_profit n > 0 :=
by
  intros n hn
  sorry

end project_profit_starts_from_4th_year_l2323_232389


namespace work_done_by_forces_l2323_232349

-- Definitions of given forces and displacement
noncomputable def F1 : ℝ × ℝ := (Real.log 2, Real.log 2)
noncomputable def F2 : ℝ × ℝ := (Real.log 5, Real.log 2)
noncomputable def S : ℝ × ℝ := (2 * Real.log 5, 1)

-- Statement of the theorem
theorem work_done_by_forces :
  let F := (F1.1 + F2.1, F1.2 + F2.2)
  let W := F.1 * S.1 + F.2 * S.2
  W = 2 :=
by
  sorry

end work_done_by_forces_l2323_232349


namespace box_width_l2323_232388

theorem box_width (h : ℝ) (d : ℝ) (l : ℝ) (w : ℝ) 
  (h_eq_8 : h = 8)
  (l_eq_2h : l = 2 * h)
  (d_eq_20 : d = 20) :
  w = 4 * Real.sqrt 5 :=
by
  sorry

end box_width_l2323_232388


namespace expression_simplification_l2323_232399

theorem expression_simplification :
  (4 * 6 / (12 * 8)) * ((5 * 12 * 8) / (4 * 5 * 5)) = 1 / 2 :=
by
  sorry

end expression_simplification_l2323_232399


namespace mary_talking_ratio_l2323_232318

theorem mary_talking_ratio:
  let mac_download_time := 10
  let windows_download_time := 3 * mac_download_time
  let audio_glitch_time := 2 * 4
  let video_glitch_time := 6
  let total_glitch_time := audio_glitch_time + video_glitch_time
  let total_download_time := mac_download_time + windows_download_time
  let total_time := 82
  let talking_time := total_time - total_download_time
  let talking_time_without_glitch := talking_time - total_glitch_time
  talking_time_without_glitch / total_glitch_time = 2 :=
by
  sorry

end mary_talking_ratio_l2323_232318


namespace regular_price_of_Pony_jeans_l2323_232382

theorem regular_price_of_Pony_jeans 
(Fox_price : ℝ) 
(Pony_price : ℝ) 
(savings : ℝ) 
(Fox_discount_rate : ℝ) 
(Pony_discount_rate : ℝ)
(h1 : Fox_price = 15)
(h2 : savings = 8.91)
(h3 : Fox_discount_rate + Pony_discount_rate = 0.22)
(h4 : Pony_discount_rate = 0.10999999999999996) : Pony_price = 18 := 
sorry

end regular_price_of_Pony_jeans_l2323_232382


namespace robin_packages_l2323_232321

theorem robin_packages (p t n : ℕ) (h1 : p = 18) (h2 : t = 486) : t / p = n ↔ n = 27 :=
by
  rw [h1, h2]
  norm_num
  sorry

end robin_packages_l2323_232321


namespace initial_volume_of_mixture_l2323_232369

theorem initial_volume_of_mixture (V : ℝ) :
  let V_new := V + 8
  let initial_water := 0.20 * V
  let new_water := initial_water + 8
  let new_mixture := V_new
  new_water = 0.25 * new_mixture →
  V = 120 :=
by
  intro h
  sorry

end initial_volume_of_mixture_l2323_232369


namespace fraction_c_over_d_l2323_232366

-- Assume that we have a polynomial equation ax^3 + bx^2 + cx + d = 0 with roots 1, 2, 3
def polynomial (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The roots of the polynomial are 1, 2, 3
def roots (a b c d : ℝ) : Prop := polynomial a b c d 1 ∧ polynomial a b c d 2 ∧ polynomial a b c d 3

-- Vieta's formulas give us the relation for c and d in terms of the roots
theorem fraction_c_over_d (a b c d : ℝ) (h : roots a b c d) : c / d = -11 / 6 :=
sorry

end fraction_c_over_d_l2323_232366


namespace simplified_expression_result_l2323_232359

theorem simplified_expression_result :
  ((2 + 3 + 6 + 7) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by {
  sorry
}

end simplified_expression_result_l2323_232359


namespace original_salary_l2323_232394

theorem original_salary (x : ℝ)
  (h1 : x * 1.10 * 0.95 = 3135) : x = 3000 :=
by
  sorry

end original_salary_l2323_232394


namespace add_fractions_add_fractions_as_mixed_l2323_232380

theorem add_fractions : (3 / 4) + (5 / 6) + (4 / 3) = (35 / 12) := sorry

theorem add_fractions_as_mixed : (3 / 4) + (5 / 6) + (4 / 3) = 2 + 11 / 12 := sorry

end add_fractions_add_fractions_as_mixed_l2323_232380


namespace regular_21_gon_symmetries_and_angle_sum_l2323_232356

theorem regular_21_gon_symmetries_and_angle_sum :
  let L' := 21
  let R' := 360 / 21
  L' + R' = 38.142857 := by
    sorry

end regular_21_gon_symmetries_and_angle_sum_l2323_232356


namespace math_books_count_l2323_232358

theorem math_books_count (M H : ℤ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end math_books_count_l2323_232358


namespace distinct_ordered_pairs_l2323_232381

theorem distinct_ordered_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h : 1/m + 1/n = 1/5) : 
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (1 / m + 1 / n = 1 / 5) :=
sorry

end distinct_ordered_pairs_l2323_232381


namespace pedestrian_speeds_unique_l2323_232305

variables 
  (x y : ℝ)
  (d : ℝ := 105)  -- Distance between cities
  (t1 : ℝ := 7.5) -- Time for current speeds
  (t2 : ℝ := 105 / 13) -- Time for adjusted speeds

theorem pedestrian_speeds_unique :
  (x + y = 14) →
  (3 * x + y = 14) →
  x = 6 ∧ y = 8 :=
by
  intros h1 h2
  have : 2 * x = 12 :=
    by ring_nf; sorry
  have hx : x = 6 :=
    by linarith
  have hy : y = 8 :=
    by linarith
  exact ⟨hx, hy⟩

end pedestrian_speeds_unique_l2323_232305


namespace find_a_l2323_232309

-- We define the conditions given in the problem
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- The expression defined as per the problem statement
def expansion_coeff_x2 (a : ℝ) : ℝ :=
  (binom 4 2) * 4 - 2 * (binom 4 1) * (binom 5 1) * a + (binom 5 2) * a^2

-- We now express the proof statement in Lean 4. 
-- We need to prove that given the coefficient of x^2 is -16, then a = 2
theorem find_a (a : ℝ) (h : expansion_coeff_x2 a = -16) : a = 2 :=
  by sorry

end find_a_l2323_232309


namespace berry_saturday_reading_l2323_232370

-- Given data
def sunday_pages := 43
def monday_pages := 65
def tuesday_pages := 28
def wednesday_pages := 0
def thursday_pages := 70
def friday_pages := 56
def average_goal := 50
def days_in_week := 7

-- Calculate total pages to meet the weekly goal
def weekly_goal := days_in_week * average_goal

-- Calculate pages read so far from Sunday to Friday
def pages_read := sunday_pages + monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages

-- Calculate required pages to read on Saturday
def saturday_pages_required := weekly_goal - pages_read

-- The theorem statement: Berry needs to read 88 pages on Saturday.
theorem berry_saturday_reading : saturday_pages_required = 88 := 
by {
  -- The proof is omitted as per the instructions
  sorry
}

end berry_saturday_reading_l2323_232370


namespace value_of_other_bills_l2323_232332

theorem value_of_other_bills (total_payment : ℕ) (num_fifty_dollar_bills : ℕ) (value_fifty_dollar_bill : ℕ) (num_other_bills : ℕ) 
  (total_fifty_dollars : ℕ) (remaining_payment : ℕ) (value_of_each_other_bill : ℕ) :
  total_payment = 170 →
  num_fifty_dollar_bills = 3 →
  value_fifty_dollar_bill = 50 →
  num_other_bills = 2 →
  total_fifty_dollars = num_fifty_dollar_bills * value_fifty_dollar_bill →
  remaining_payment = total_payment - total_fifty_dollars →
  value_of_each_other_bill = remaining_payment / num_other_bills →
  value_of_each_other_bill = 10 :=
by
  intros t_total_payment t_num_fifty_dollar_bills t_value_fifty_dollar_bill t_num_other_bills t_total_fifty_dollars t_remaining_payment t_value_of_each_other_bill
  sorry

end value_of_other_bills_l2323_232332


namespace complement_union_l2323_232341

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (hM : M = {1, 3, 5, 7}) (hN : N = {5, 6, 7}) : U \ (M ∪ N) = {2, 4, 8} :=
by
  sorry

end complement_union_l2323_232341


namespace logan_average_speed_l2323_232367

theorem logan_average_speed 
  (tamika_hours : ℕ)
  (tamika_speed : ℕ)
  (logan_hours : ℕ)
  (tamika_distance : ℕ)
  (logan_distance : ℕ)
  (distance_diff : ℕ)
  (diff_condition : tamika_distance = logan_distance + distance_diff) :
  tamika_hours = 8 →
  tamika_speed = 45 →
  logan_hours = 5 →
  tamika_distance = tamika_speed * tamika_hours →
  distance_diff = 85 →
  logan_distance / logan_hours = 55 :=
by
  sorry

end logan_average_speed_l2323_232367


namespace lizard_eyes_fewer_than_spots_and_wrinkles_l2323_232352

noncomputable def lizard_problem : Nat :=
  let eyes_jan := 3
  let wrinkles_jan := 3 * eyes_jan
  let spots_jan := 7 * (wrinkles_jan ^ 2)
  let eyes_cousin := 3
  let wrinkles_cousin := 2 * eyes_cousin
  let spots_cousin := 5 * (wrinkles_cousin ^ 2)
  let total_eyes := eyes_jan + eyes_cousin
  let total_wrinkles := wrinkles_jan + wrinkles_cousin
  let total_spots := spots_jan + spots_cousin
  (total_spots + total_wrinkles) - total_eyes

theorem lizard_eyes_fewer_than_spots_and_wrinkles :
  lizard_problem = 756 :=
by
  sorry

end lizard_eyes_fewer_than_spots_and_wrinkles_l2323_232352


namespace correct_propositions_l2323_232319

structure Proposition :=
  (statement : String)
  (is_correct : Prop)

def prop1 : Proposition := {
  statement := "All sufficiently small positive numbers form a set.",
  is_correct := False -- From step b
}

def prop2 : Proposition := {
  statement := "The set containing 1, 2, 3, 1, 9 is represented by enumeration as {1, 2, 3, 1, 9}.",
  is_correct := False -- From step b
}

def prop3 : Proposition := {
  statement := "{1, 3, 5, 7} and {7, 5, 3, 1} denote the same set.",
  is_correct := True -- From step b
}

def prop4 : Proposition := {
  statement := "{y = -x} represents the collection of all points on the graph of the function y = -x.",
  is_correct := False -- From step b
}

theorem correct_propositions :
  prop3.is_correct ∧ ¬prop1.is_correct ∧ ¬prop2.is_correct ∧ ¬prop4.is_correct :=
by
  -- Here we put the proof steps, but for the exercise's purpose, we use sorry.
  sorry

end correct_propositions_l2323_232319


namespace goose_survived_first_year_l2323_232330

theorem goose_survived_first_year (total_eggs : ℕ) (eggs_hatched_ratio : ℚ) (first_month_survival_ratio : ℚ) 
  (first_year_no_survival_ratio : ℚ) 
  (eggs_hatched_ratio_eq : eggs_hatched_ratio = 2/3) 
  (first_month_survival_ratio_eq : first_month_survival_ratio = 3/4)
  (first_year_no_survival_ratio_eq : first_year_no_survival_ratio = 3/5)
  (total_eggs_eq : total_eggs = 500) :
  ∃ (survived_first_year : ℕ), survived_first_year = 100 :=
by
  sorry

end goose_survived_first_year_l2323_232330


namespace stations_visited_l2323_232377

-- Define the total number of nails
def total_nails : ℕ := 560

-- Define the number of nails left at each station
def nails_per_station : ℕ := 14

-- Main theorem statement
theorem stations_visited : total_nails / nails_per_station = 40 := by
  sorry

end stations_visited_l2323_232377


namespace arrange_desc_l2323_232383

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (35 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (35 * Real.pi / 180)
noncomputable def d : ℝ := Real.log 5

theorem arrange_desc : d > c ∧ c > b ∧ b > a := by
  sorry

end arrange_desc_l2323_232383


namespace calculate_f_5_l2323_232314

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

theorem calculate_f_5 : f 5 = 4485 := 
by {
  -- The proof of the theorem will go here, using the Horner's method as described.
  sorry
}

end calculate_f_5_l2323_232314


namespace P_subset_Q_l2323_232368

def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x > 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end P_subset_Q_l2323_232368


namespace select_4_officers_from_7_members_l2323_232308

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Statement of the problem
theorem select_4_officers_from_7_members : binom 7 4 = 35 :=
by
  -- Proof not required, so we use sorry to skip it
  sorry

end select_4_officers_from_7_members_l2323_232308


namespace smallest_non_representable_l2323_232373

def isRepresentable (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_representable : ∀ n : ℕ, 0 < n → ¬ isRepresentable 11 ∧ ∀ k : ℕ, 0 < k ∧ k < 11 → isRepresentable k :=
by sorry

end smallest_non_representable_l2323_232373


namespace time_in_3467_hours_l2323_232300

-- Define the current time, the number of hours, and the modulus
def current_time : ℕ := 2
def hours_from_now : ℕ := 3467
def clock_modulus : ℕ := 12

-- Define the function to calculate the future time on a 12-hour clock
def future_time (current_time : ℕ) (hours_from_now : ℕ) (modulus : ℕ) : ℕ := 
  (current_time + hours_from_now) % modulus

-- Theorem statement
theorem time_in_3467_hours :
  future_time current_time hours_from_now clock_modulus = 9 :=
by
  -- Proof would go here
  sorry

end time_in_3467_hours_l2323_232300


namespace total_chocolate_bars_l2323_232395

theorem total_chocolate_bars (n_small_boxes : ℕ) (bars_per_box : ℕ) (total_bars : ℕ) :
  n_small_boxes = 16 → bars_per_box = 25 → total_bars = 16 * 25 → total_bars = 400 :=
by
  intros
  sorry

end total_chocolate_bars_l2323_232395


namespace abc_system_proof_l2323_232396

theorem abc_system_proof (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 + a = b^2) (h5 : b^2 + b = c^2) (h6 : c^2 + c = a^2) :
  (a - b) * (b - c) * (c - a) = 1 :=
by
  sorry

end abc_system_proof_l2323_232396


namespace limit_exists_implies_d_eq_zero_l2323_232324

variable (a₁ d : ℝ) (S : ℕ → ℝ)

noncomputable def limExists := ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (S n - L) < ε

def is_sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) :=
  ∀ n : ℕ, S n = (a₁ * n + d * (n * (n - 1) / 2))

theorem limit_exists_implies_d_eq_zero (h₁ : ∀ n : ℕ, n > 0 → S n = (a₁ * n + d * (n * (n - 1) / 2))) :
  limExists S → d = 0 :=
by sorry

end limit_exists_implies_d_eq_zero_l2323_232324


namespace problem_divisible_by_900_l2323_232371

theorem problem_divisible_by_900 (X : ℕ) (a b c d : ℕ) 
  (h1 : 1000 <= X)
  (h2 : X < 10000)
  (h3 : X = 1000 * a + 100 * b + 10 * c + d)
  (h4 : d ≠ 0)
  (h5 : (X + (1000 * a + 100 * c + 10 * b + d)) % 900 = 0)
  : X % 90 = 45 := 
sorry

end problem_divisible_by_900_l2323_232371


namespace tyler_meals_l2323_232315

def num_meals : ℕ := 
  let num_meats := 3
  let num_vegetable_combinations := Nat.choose 5 3
  let num_desserts := 5
  num_meats * num_vegetable_combinations * num_desserts

theorem tyler_meals :
  num_meals = 150 := by
  sorry

end tyler_meals_l2323_232315


namespace max_ab_l2323_232379

theorem max_ab (a b : ℝ) (h : a + b = 1) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab_l2323_232379


namespace area_of_BCD_l2323_232354

variables (a b c x y : ℝ)

-- Conditions
axiom h1 : x = (1 / 2) * a * b
axiom h2 : y = (1 / 2) * b * c

-- Conclusion to prove
theorem area_of_BCD (a b c x y : ℝ) (h1 : x = (1 / 2) * a * b) (h2 : y = (1 / 2) * b * c) : 
  (1 / 2) * b * c = y :=
sorry

end area_of_BCD_l2323_232354


namespace solve_fraction_equation_l2323_232375

theorem solve_fraction_equation (x : ℚ) (h : x ≠ -1) : 
  (x / (x + 1) = 2 * x / (3 * x + 3) - 1) → x = -3 / 4 :=
by
  sorry

end solve_fraction_equation_l2323_232375


namespace probability_red_or_white_is_11_over_13_l2323_232328

-- Given data
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - blue_marbles - red_marbles

def blue_size : ℕ := 2
def red_size : ℕ := 1
def white_size : ℕ := 1

-- Total size value of all marbles
def total_size_value : ℕ := (blue_size * blue_marbles) + (red_size * red_marbles) + (white_size * white_marbles)

-- Probability of selecting a red or white marble
def probability_red_or_white : ℚ := (red_size * red_marbles + white_size * white_marbles) / total_size_value

-- Theorem to prove
theorem probability_red_or_white_is_11_over_13 : probability_red_or_white = 11 / 13 :=
by sorry

end probability_red_or_white_is_11_over_13_l2323_232328


namespace papers_left_l2323_232378

def total_papers_bought : ℕ := 20
def pictures_drawn_today : ℕ := 6
def pictures_drawn_yesterday_before_work : ℕ := 6
def pictures_drawn_yesterday_after_work : ℕ := 6

theorem papers_left :
  total_papers_bought - (pictures_drawn_today + pictures_drawn_yesterday_before_work + pictures_drawn_yesterday_after_work) = 2 := 
by 
  sorry

end papers_left_l2323_232378


namespace calc_expression_l2323_232312

variable {x : ℝ}

theorem calc_expression :
    (2 + 3 * x) * (-2 + 3 * x) = 9 * x ^ 2 - 4 := sorry

end calc_expression_l2323_232312


namespace men_became_absent_l2323_232362

theorem men_became_absent (num_men absent : ℤ) 
  (num_men_eq : num_men = 180) 
  (days_planned : ℤ) (days_planned_eq : days_planned = 55)
  (days_taken : ℤ) (days_taken_eq : days_taken = 60)
  (work_planned : ℤ) (work_planned_eq : work_planned = num_men * days_planned)
  (work_taken : ℤ) (work_taken_eq : work_taken = (num_men - absent) * days_taken)
  (work_eq : work_planned = work_taken) :
  absent = 15 :=
  by sorry

end men_became_absent_l2323_232362


namespace area_R_l2323_232385

-- Define the given matrix as a 2x2 real matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, -5]

-- Define the original area of region R
def area_R : ℝ := 15

-- Define the area scaling factor as the absolute value of the determinant of A
def scaling_factor : ℝ := |Matrix.det A|

-- Prove that the area of the region R' is 585
theorem area_R' : scaling_factor * area_R = 585 := by
  sorry

end area_R_l2323_232385


namespace january_1_is_monday_l2323_232353

theorem january_1_is_monday
  (days_in_january : ℕ)
  (mondays_in_january : ℕ)
  (thursdays_in_january : ℕ) :
  days_in_january = 31 ∧ mondays_in_january = 5 ∧ thursdays_in_january = 5 → 
  ∃ d : ℕ, d = 1 ∧ (d % 7 = 1) :=
by
  sorry

end january_1_is_monday_l2323_232353


namespace intensity_of_replacement_paint_l2323_232310

theorem intensity_of_replacement_paint (f : ℚ) (I_new : ℚ) (I_orig : ℚ) (I_repl : ℚ) :
  f = 2/3 → I_new = 40 → I_orig = 60 → I_repl = (40 - 1/3 * 60) * (3/2) := by
  sorry

end intensity_of_replacement_paint_l2323_232310


namespace work_completion_l2323_232342

theorem work_completion (x y : ℕ) : 
  (1 / (x + y) = 1 / 12) ∧ (1 / y = 1 / 24) → x = 24 :=
by
  sorry

end work_completion_l2323_232342


namespace order_of_a_b_c_l2323_232317

noncomputable def a := 2 + Real.sqrt 3
noncomputable def b := 1 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 + Real.sqrt 5

theorem order_of_a_b_c : a > c ∧ c > b := 
by {
  sorry
}

end order_of_a_b_c_l2323_232317


namespace missing_side_length_of_pan_l2323_232301

-- Definition of the given problem's conditions
def pan_side_length := 29
def total_fudge_pieces := 522
def fudge_piece_area := 1

-- Proof statement in Lean 4
theorem missing_side_length_of_pan : 
  (total_fudge_pieces * fudge_piece_area) = (pan_side_length * 18) :=
by
  sorry

end missing_side_length_of_pan_l2323_232301


namespace dan_speed_must_exceed_48_l2323_232345

theorem dan_speed_must_exceed_48 (d : ℕ) (s_cara : ℕ) (time_delay : ℕ) : 
  d = 120 → s_cara = 30 → time_delay = 3 / 2 → ∃ v : ℕ, v > 48 :=
by
  intro h1 h2 h3
  use 49
  sorry

end dan_speed_must_exceed_48_l2323_232345


namespace union_intersection_l2323_232337

-- Define the sets A, B, and C
def A : Set ℕ := {1, 2, 6}
def B : Set ℕ := {2, 4}
def C : Set ℕ := {1, 2, 3, 4}

-- The theorem stating that (A ∪ B) ∩ C = {1, 2, 4}
theorem union_intersection : (A ∪ B) ∩ C = {1, 2, 4} := sorry

end union_intersection_l2323_232337
