import Mathlib

namespace sum_of_odd_powers_l1877_187753

variable (x y z a : ℝ) (k : ℕ)

theorem sum_of_odd_powers (h1 : x + y + z = a) (h2 : x^3 + y^3 + z^3 = a^3) (hk : k % 2 = 1) : 
  x^k + y^k + z^k = a^k :=
sorry

end sum_of_odd_powers_l1877_187753


namespace merchant_marked_price_percentage_l1877_187762

variables (L S M C : ℝ)
variable (h1 : C = 0.7 * L)
variable (h2 : C = 0.75 * S)
variable (h3 : S = 0.9 * M)

theorem merchant_marked_price_percentage : M = 1.04 * L :=
by
  sorry

end merchant_marked_price_percentage_l1877_187762


namespace smallest_a_for_polynomial_l1877_187790

theorem smallest_a_for_polynomial (a b x₁ x₂ x₃ : ℕ) 
    (h1 : x₁ * x₂ * x₃ = 2730)
    (h2 : x₁ + x₂ + x₃ = a)
    (h3 : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0)
    (h4 : ∀ y₁ y₂ y₃ : ℕ, y₁ * y₂ * y₃ = 2730 ∧ y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + y₂ + y₃ ≥ a) :
  a = 54 :=
  sorry

end smallest_a_for_polynomial_l1877_187790


namespace cost_of_bananas_and_cantaloupe_l1877_187777

variable (a b c d : ℝ)

theorem cost_of_bananas_and_cantaloupe :
  (a + b + c + d = 30) →
  (d = 3 * a) →
  (c = a - b) →
  (b + c = 6) :=
by
  intros h1 h2 h3
  sorry

end cost_of_bananas_and_cantaloupe_l1877_187777


namespace distinct_infinite_solutions_l1877_187796

theorem distinct_infinite_solutions (n : ℕ) (hn : n > 0) : 
  ∃ p q : ℤ, p + q * Real.sqrt 5 = (9 + 4 * Real.sqrt 5) ^ n ∧ (p * p - 5 * q * q = 1) ∧ 
  ∀ m : ℕ, (m ≠ n → (9 + 4 * Real.sqrt 5) ^ m ≠ (9 + 4 * Real.sqrt 5) ^ n) :=
by
  sorry

end distinct_infinite_solutions_l1877_187796


namespace sum_of_roots_l1877_187780

theorem sum_of_roots (x : ℝ) (h : (x + 3) * (x - 2) = 15) : x = -1 :=
sorry

end sum_of_roots_l1877_187780


namespace max_sum_ac_bc_l1877_187702

noncomputable def triangle_ab_bc_sum_max (AB : ℝ) (C : ℝ) : ℝ :=
  if AB = Real.sqrt 6 - Real.sqrt 2 ∧ C = Real.pi / 6 then 4 else 0

theorem max_sum_ac_bc {A B C : ℝ} (h1 : AB = Real.sqrt 6 - Real.sqrt 2) (h2 : C = Real.pi / 6) :
  triangle_ab_bc_sum_max AB C = 4 :=
by {
  sorry
}

end max_sum_ac_bc_l1877_187702


namespace num_biology_books_is_15_l1877_187731

-- conditions
def num_chemistry_books : ℕ := 8
def total_ways : ℕ := 2940

-- main statement to prove
theorem num_biology_books_is_15 : ∃ B: ℕ, (B * (B - 1)) / 2 * (num_chemistry_books * (num_chemistry_books - 1)) / 2 = total_ways ∧ B = 15 :=
by
  sorry

end num_biology_books_is_15_l1877_187731


namespace geometric_sequence_sum_l1877_187708

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ)
  (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 5 + a 6 = 90 :=
sorry

end geometric_sequence_sum_l1877_187708


namespace find_x_range_l1877_187776

noncomputable def f (x : ℝ) : ℝ := if h : x ≥ 0 then 3^(-x) else 3^(x)

theorem find_x_range (x : ℝ) (h1 : f 2 = -f (2*x - 1) ∧ f 2 < 0) : -1/2 < x ∧ x < 3/2 := by
  -- Proof goes here
  sorry

end find_x_range_l1877_187776


namespace geoff_needed_more_votes_to_win_l1877_187773

-- Definitions based on the conditions
def total_votes : ℕ := 6000
def percent_to_fraction (p : ℕ) : ℚ := p / 100
def geoff_percent : ℚ := percent_to_fraction 1
def win_percent : ℚ := percent_to_fraction 51

-- Specific values derived from the conditions
def geoff_votes : ℚ := geoff_percent * total_votes
def win_votes : ℚ := win_percent * total_votes + 1

-- The theorem we intend to prove
theorem geoff_needed_more_votes_to_win :
  (win_votes - geoff_votes) = 3001 := by
  sorry

end geoff_needed_more_votes_to_win_l1877_187773


namespace brets_dinner_tip_calculation_l1877_187771

/-
  We need to prove that the percentage of the tip Bret included is 20%, given the conditions.
-/

theorem brets_dinner_tip_calculation :
  let num_meals := 4
  let cost_per_meal := 12
  let num_appetizers := 2
  let cost_per_appetizer := 6
  let rush_fee := 5
  let total_cost := 77
  (total_cost - (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer + rush_fee))
  / (num_meals * cost_per_meal + num_appetizers * cost_per_appetizer) * 100 = 20 :=
by
  sorry

end brets_dinner_tip_calculation_l1877_187771


namespace average_speed_l1877_187767

theorem average_speed (speed1 speed2 time1 time2: ℝ) (h1 : speed1 = 60) (h2 : time1 = 3) (h3 : speed2 = 85) (h4 : time2 = 2) : 
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 70 :=
by
  -- Definitions
  have distance1 := speed1 * time1
  have distance2 := speed2 * time2
  have total_distance := distance1 + distance2
  have total_time := time1 + time2
  -- Proof skeleton
  sorry

end average_speed_l1877_187767


namespace value_of_f_37_5_l1877_187709

-- Mathematical definitions and conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f (x)
def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f (x)
def interval_condition (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) = x

-- Main theorem to be proved
theorem value_of_f_37_5 (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_periodic : satisfies_condition f) 
  (h_interval : interval_condition f) : 
  f 37.5 = 0.5 := 
sorry

end value_of_f_37_5_l1877_187709


namespace smallest_rational_in_set_l1877_187714

theorem smallest_rational_in_set : 
  ∀ (a b c d : ℚ), 
    a = -2/3 → b = -1 → c = 0 → d = 1 → 
    (a > b ∧ b < c ∧ c < d) → b = -1 := 
by
  intros a b c d ha hb hc hd h
  sorry

end smallest_rational_in_set_l1877_187714


namespace number_of_students_l1877_187737

theorem number_of_students (n : ℕ)
  (h_avg : 100 * n = total_marks_unknown)
  (h_wrong_marks : total_marks_wrong = total_marks_unknown + 50)
  (h_correct_avg : total_marks_correct / n = 95)
  (h_corrected_marks : total_marks_correct = total_marks_wrong - 50) :
  n = 10 :=
by
  sorry

end number_of_students_l1877_187737


namespace find_k_l1877_187775

variable {a_n : ℕ → ℤ}    -- Define the arithmetic sequence as a function from natural numbers to integers
variable {a1 d : ℤ}        -- a1 is the first term, d is the common difference

-- Conditions
axiom seq_def : ∀ n, a_n n = a1 + (n - 1) * d
axiom sum_condition : 9 * a1 + 36 * d = 4 * a1 + 6 * d
axiom ak_a4_zero (k : ℕ): a_n 4 + a_n k = 0

-- Problem Statement to prove
theorem find_k : ∃ k : ℕ, a_n 4 + a_n k = 0 → k = 10 :=
by
  use 10
  intro h
  -- proof omitted
  sorry

end find_k_l1877_187775


namespace new_sailor_weight_l1877_187736

-- Define the conditions
variables {average_weight : ℝ} (new_weight : ℝ)
variable (old_weight : ℝ := 56)

-- State the property we need to prove
theorem new_sailor_weight
  (h : (new_weight - old_weight) = 8) :
  new_weight = 64 :=
by
  sorry

end new_sailor_weight_l1877_187736


namespace find_xyz_l1877_187711

theorem find_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + x * y * z = 15) :
  x * y * z = 9 / 2 := by
  sorry

end find_xyz_l1877_187711


namespace min_value_is_3_l1877_187729

theorem min_value_is_3 (a b : ℝ) (h1 : a > b / 2) (h2 : 2 * a > b) : (2 * a + b) / a ≥ 3 :=
sorry

end min_value_is_3_l1877_187729


namespace part1_part2_l1877_187717

open Nat

-- Part (I)
theorem part1 (a b : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x + b = 0 → x = 2 ∨ x = 3) :
  a + b = 11 :=
by sorry

-- Part (II)
theorem part2 (c : ℝ) (h2 : ∀ x : ℝ, -x^2 + 6 * x + c ≤ 0) :
  c ≤ -9 :=
by sorry

end part1_part2_l1877_187717


namespace whisker_relationship_l1877_187707

theorem whisker_relationship :
  let P_whiskers := 14
  let C_whiskers := 22
  (C_whiskers - P_whiskers = 8) ∧ (C_whiskers / P_whiskers = 11 / 7) :=
by
  let P_whiskers := 14
  let C_whiskers := 22
  have h1 : C_whiskers - P_whiskers = 8 := by sorry
  have h2 : C_whiskers / P_whiskers = 11 / 7 := by sorry
  exact And.intro h1 h2

end whisker_relationship_l1877_187707


namespace max_value_expression_l1877_187732

theorem max_value_expression (s : ℝ) : 
  ∃ M, M = -3 * s^2 + 36 * s + 7 ∧ (∀ t : ℝ, -3 * t^2 + 36 * t + 7 ≤ M) :=
by
  use 115
  sorry

end max_value_expression_l1877_187732


namespace angle_45_deg_is_75_venerts_l1877_187755

-- There are 600 venerts in a full circle.
def venus_full_circle : ℕ := 600

-- A full circle on Earth is 360 degrees.
def earth_full_circle : ℕ := 360

-- Conversion factor from degrees to venerts.
def degrees_to_venerts (deg : ℕ) : ℕ :=
  deg * (venus_full_circle / earth_full_circle)

-- Angle of 45 degrees in venerts.
def angle_45_deg_in_venerts : ℕ := 45 * (venus_full_circle / earth_full_circle)

theorem angle_45_deg_is_75_venerts :
  angle_45_deg_in_venerts = 75 :=
by
  -- Proof will be inserted here.
  sorry

end angle_45_deg_is_75_venerts_l1877_187755


namespace train_length_correct_l1877_187720

noncomputable def speed_km_per_hour : ℝ := 56
noncomputable def time_seconds : ℝ := 32.142857142857146
noncomputable def bridge_length_m : ℝ := 140
noncomputable def train_length_m : ℝ := 360

noncomputable def speed_m_per_s : ℝ := speed_km_per_hour * (1000 / 3600)
noncomputable def total_distance_m : ℝ := speed_m_per_s * time_seconds

theorem train_length_correct :
  (total_distance_m - bridge_length_m) = train_length_m :=
  by
    sorry

end train_length_correct_l1877_187720


namespace jeremy_school_distance_l1877_187791

def travel_time_rush_hour := 15 / 60 -- hours
def travel_time_clear_day := 10 / 60 -- hours
def speed_increase := 20 -- miles per hour

def distance_to_school (d v : ℝ) : Prop :=
  d = v * travel_time_rush_hour ∧ d = (v + speed_increase) * travel_time_clear_day

theorem jeremy_school_distance (d v : ℝ) (h_speed : v = 40) : d = 10 :=
by
  have travel_time_rush_hour := 1/4
  have travel_time_clear_day := 1/6
  have speed_increase := 20
  
  have h1 : d = v * travel_time_rush_hour := by sorry
  have h2 : d = (v + speed_increase) * travel_time_clear_day := by sorry
  have eqn := distance_to_school d v
  sorry

end jeremy_school_distance_l1877_187791


namespace marbles_count_l1877_187744

variable (r b : ℕ)

theorem marbles_count (hr1 : 8 * (r - 1) = r + b - 2) (hr2 : 4 * r = r + b - 3) : r + b = 9 := 
by sorry

end marbles_count_l1877_187744


namespace smallest_composite_square_side_length_l1877_187760

theorem smallest_composite_square_side_length (n : ℕ) (h : ∃ k, 14 * n = k^2) : 
  ∃ m : ℕ, n = 14 ∧ m = 14 :=
by
  sorry

end smallest_composite_square_side_length_l1877_187760


namespace mod_computation_l1877_187749

theorem mod_computation (n : ℤ) : 
  0 ≤ n ∧ n < 23 ∧ 47582 % 23 = n ↔ n = 3 := 
by 
  -- Proof omitted
  sorry

end mod_computation_l1877_187749


namespace sum_of_geometric_progression_l1877_187799

theorem sum_of_geometric_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (a1 a3 : ℝ) (h1 : a1 + a3 = 5) (h2 : a1 * a3 = 4)
  (h3 : a 1 = a1) (h4 : a 3 = a3)
  (h5 : ∀ k, a (k + 1) > a k)  -- Sequence is increasing
  (h6 : S n = a 1 * ((1 - (2:ℝ) ^ n) / (1 - 2)))
  (h7 : n = 6) :
  S 6 = 63 :=
sorry

end sum_of_geometric_progression_l1877_187799


namespace even_function_periodic_symmetric_about_2_l1877_187722

variables {F : ℝ → ℝ}

theorem even_function_periodic_symmetric_about_2
  (h_even : ∀ x, F x = F (-x))
  (h_symmetric : ∀ x, F (2 - x) = F (2 + x))
  (h_cond : F 2011 + 2 * F 1 = 18) :
  F 2011 = 6 :=
sorry

end even_function_periodic_symmetric_about_2_l1877_187722


namespace diamond_expression_l1877_187747

-- Define the diamond operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Declare the main theorem
theorem diamond_expression :
  (diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4)) = -29 / 132 := 
by
  sorry

end diamond_expression_l1877_187747


namespace water_tank_capacity_l1877_187712

theorem water_tank_capacity (C : ℝ) :
  0.4 * C - 0.1 * C = 36 → C = 120 :=
by sorry

end water_tank_capacity_l1877_187712


namespace compound_interest_rate_l1877_187761

-- Defining the principal amount and total repayment
def P : ℝ := 200
def A : ℝ := 220

-- The annual compound interest rate
noncomputable def annual_compound_interest_rate (P A : ℝ) (n : ℕ) : ℝ :=
  (A / P)^(1 / n) - 1

-- Introducing the conditions
axiom compounded_annually : ∀ (P A : ℝ), annual_compound_interest_rate P A 1 = 0.1

-- Stating the theorem
theorem compound_interest_rate :
  annual_compound_interest_rate P A 1 = 0.1 :=
by {
  exact compounded_annually P A
}

end compound_interest_rate_l1877_187761


namespace solve_df1_l1877_187718

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (df1 : ℝ)

-- The condition given in the problem
axiom func_def : ∀ x, f x = 2 * x * df1 + (Real.log x)

-- Express the relationship from the derivative and solve for f'(1) = -1
theorem solve_df1 : df1 = -1 :=
by
  -- Here we will insert the proof steps in Lean, but they are omitted in this statement.
  sorry

end solve_df1_l1877_187718


namespace sin_2A_value_l1877_187726

variable {A B C : ℝ}
variable {a b c : ℝ}
variable (h₁ : a / (2 * Real.cos A) = b / (3 * Real.cos B))
variable (h₂ : b / (3 * Real.cos B) = c / (6 * Real.cos C))

theorem sin_2A_value (h₃ : a / (2 * Real.cos A) = c / (6 * Real.cos C)) :
  Real.sin (2 * A) = 3 * Real.sqrt 11 / 10 := sorry

end sin_2A_value_l1877_187726


namespace silver_coins_change_l1877_187706

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l1877_187706


namespace roots_of_quadratic_l1877_187766

variable {γ δ : ℝ}

theorem roots_of_quadratic (hγ : γ^2 - 5*γ + 6 = 0) (hδ : δ^2 - 5*δ + 6 = 0) : 
  8*γ^5 + 15*δ^4 = 8425 := 
by
  sorry

end roots_of_quadratic_l1877_187766


namespace rebus_puzzle_verified_l1877_187765

-- Defining the conditions
def A := 1
def B := 1
def C := 0
def D := 1
def F := 1
def L := 1
def M := 0
def N := 1
def P := 0
def Q := 1
def T := 1
def G := 8
def H := 1
def K := 4
def W := 4
def X := 1

noncomputable def verify_rebus_puzzle : Prop :=
  (A * B * 10 = 110) ∧
  (6 * G / (10 * H + 7) = 4) ∧
  (L + N * 10 = 20) ∧
  (12 - K = 8) ∧
  (101 + 10 * W + X = 142)

-- Lean statement to verify the problem
theorem rebus_puzzle_verified : verify_rebus_puzzle :=
by {
  -- Values are already defined and will be concluded by Lean
  sorry
}

end rebus_puzzle_verified_l1877_187765


namespace value_set_l1877_187784

open Real Set

noncomputable def possible_values (a b c : ℝ) : Set ℝ :=
  {x | ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 2 ∧ x = c / a + c / b}

theorem value_set (c : ℝ) (hc : c > 0) : possible_values a b c = Ici (2 * c) := by
  sorry

end value_set_l1877_187784


namespace savings_in_cents_l1877_187723

def price_local : ℝ := 149.99
def price_payment : ℝ := 26.50
def number_payments : ℕ := 5
def fee_delivery : ℝ := 19.99

theorem savings_in_cents :
  (price_local - (number_payments * price_payment + fee_delivery)) * 100 = -250 := by
  sorry

end savings_in_cents_l1877_187723


namespace students_not_in_biology_l1877_187782

theorem students_not_in_biology (S : ℕ) (f : ℚ) (hS : S = 840) (hf : f = 0.35) :
  S - (f * S) = 546 :=
by
  sorry

end students_not_in_biology_l1877_187782


namespace correct_operation_l1877_187795

theorem correct_operation (x : ℝ) : (x^3 * x^2 = x^5) :=
by sorry

end correct_operation_l1877_187795


namespace maggie_kept_bouncy_balls_l1877_187770

def packs_bought_yellow : ℝ := 8.0
def packs_given_away_green : ℝ := 4.0
def packs_bought_green : ℝ := 4.0
def balls_per_pack : ℝ := 10.0

theorem maggie_kept_bouncy_balls :
  packs_bought_yellow * balls_per_pack + (packs_bought_green - packs_given_away_green) * balls_per_pack = 80.0 :=
by sorry

end maggie_kept_bouncy_balls_l1877_187770


namespace sum_geometric_sequence_terms_l1877_187786

theorem sum_geometric_sequence_terms (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) :
  a * (1 - r^4500) / (1 - r) = 813 := 
by
  sorry

end sum_geometric_sequence_terms_l1877_187786


namespace usual_time_is_36_l1877_187764

noncomputable def usual_time_to_school (R : ℝ) (T : ℝ) : Prop :=
  let new_rate := (9/8 : ℝ) * R
  let new_time := T - 4
  R * T = new_rate * new_time

theorem usual_time_is_36 (R : ℝ) (T : ℝ) (h : T = 36) : usual_time_to_school R T :=
by
  sorry

end usual_time_is_36_l1877_187764


namespace anita_total_cartons_l1877_187793

-- Defining the conditions
def cartons_of_strawberries : ℕ := 10
def cartons_of_blueberries : ℕ := 9
def additional_cartons_needed : ℕ := 7

-- Adding the core theorem to be proved
theorem anita_total_cartons :
  cartons_of_strawberries + cartons_of_blueberries + additional_cartons_needed = 26 := 
by
  sorry

end anita_total_cartons_l1877_187793


namespace thought_number_and_appended_digit_l1877_187769

theorem thought_number_and_appended_digit (x y : ℕ) (hx : x > 0) (hy : y ≤ 9):
  (10 * x + y - x^2 = 8 * x) ↔ (x = 2 ∧ y = 0) ∨ (x = 3 ∧ y = 3) ∨ (x = 4 ∧ y = 8) := sorry

end thought_number_and_appended_digit_l1877_187769


namespace length_of_train_l1877_187704

def speed_kmh : ℝ := 162
def time_seconds : ℝ := 2.222044458665529
def speed_ms : ℝ := 45  -- from conversion: 162 * (1000 / 3600)

theorem length_of_train :
  (speed_kmh * (1000 / 3600)) * time_seconds = 100 := by
  -- Proof is left out
  sorry 

end length_of_train_l1877_187704


namespace bus_capacity_percentage_l1877_187724

theorem bus_capacity_percentage (x : ℕ) (h1 : 150 * x / 100 + 150 * 70 / 100 = 195) : x = 60 :=
by
  sorry

end bus_capacity_percentage_l1877_187724


namespace gcd_5800_14025_l1877_187792

theorem gcd_5800_14025 : Int.gcd 5800 14025 = 25 := by
  sorry

end gcd_5800_14025_l1877_187792


namespace desired_overall_percentage_l1877_187734

-- Define the scores in the three subjects
def score1 := 50
def score2 := 70
def score3 := 90

-- Define the expected overall percentage
def expected_overall_percentage := 70

-- The main theorem to prove
theorem desired_overall_percentage :
  (score1 + score2 + score3) / 3 = expected_overall_percentage :=
by
  sorry

end desired_overall_percentage_l1877_187734


namespace problem_l1877_187756

theorem problem (a b : ℝ) (h : a > b) (k : b > 0) : b * (a - b) > 0 := 
by
  sorry

end problem_l1877_187756


namespace computer_operations_correct_l1877_187789

-- Define the rate of operations per second
def operations_per_second : ℝ := 4 * 10^8

-- Define the total number of seconds the computer operates
def total_seconds : ℝ := 6 * 10^5

-- Define the expected total number of operations
def expected_operations : ℝ := 2.4 * 10^14

-- Theorem stating the total number of operations is as expected
theorem computer_operations_correct :
  operations_per_second * total_seconds = expected_operations :=
by
  sorry

end computer_operations_correct_l1877_187789


namespace average_male_grade_l1877_187774

theorem average_male_grade (avg_all avg_fem : ℝ) (N_male N_fem : ℕ) 
    (h1 : avg_all = 90) 
    (h2 : avg_fem = 92) 
    (h3 : N_male = 8) 
    (h4 : N_fem = 12) :
    let total_students := N_male + N_fem
    let total_sum_all := avg_all * total_students
    let total_sum_fem := avg_fem * N_fem
    let total_sum_male := total_sum_all - total_sum_fem
    let avg_male := total_sum_male / N_male
    avg_male = 87 :=
by 
  let total_students := N_male + N_fem
  let total_sum_all := avg_all * total_students
  let total_sum_fem := avg_fem * N_fem
  let total_sum_male := total_sum_all - total_sum_fem
  let avg_male := total_sum_male / N_male
  sorry

end average_male_grade_l1877_187774


namespace factorization_sum_l1877_187719

theorem factorization_sum :
  ∃ a b c : ℤ, (∀ x : ℝ, (x^2 + 20 * x + 96 = (x + a) * (x + b)) ∧
                      (x^2 + 18 * x + 81 = (x - b) * (x + c))) →
              (a + b + c = 30) :=
by
  sorry

end factorization_sum_l1877_187719


namespace value_of_a_l1877_187742

def f (x : ℝ) : ℝ := x^2 + 9
def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 25) : a = 3 :=
by
  sorry

end value_of_a_l1877_187742


namespace no_real_solution_for_eq_l1877_187797

theorem no_real_solution_for_eq (y : ℝ) : ¬ ∃ y : ℝ, ((y - 4 * y + 10)^2 + 4 = -2 * |y|) :=
by
  sorry

end no_real_solution_for_eq_l1877_187797


namespace Sara_quarters_after_borrowing_l1877_187733

theorem Sara_quarters_after_borrowing (initial_quarters borrowed_quarters : ℕ) (h1 : initial_quarters = 783) (h2 : borrowed_quarters = 271) :
  initial_quarters - borrowed_quarters = 512 := by
  sorry

end Sara_quarters_after_borrowing_l1877_187733


namespace joan_spent_on_thursday_l1877_187743

theorem joan_spent_on_thursday : 
  ∀ (n : ℕ), 
  2 * (4 + n) = 18 → 
  n = 14 := 
by 
  sorry

end joan_spent_on_thursday_l1877_187743


namespace a_2011_value_l1877_187716

noncomputable def sequence_a : ℕ → ℝ
| 0 => 6/7
| (n + 1) => if 0 ≤ sequence_a n ∧ sequence_a n < 1/2 then 2 * sequence_a n
              else 2 * sequence_a n - 1

theorem a_2011_value : sequence_a 2011 = 6/7 := sorry

end a_2011_value_l1877_187716


namespace number_of_dots_on_faces_l1877_187738

theorem number_of_dots_on_faces (d A B C D : ℕ) 
  (h1 : d = 6)
  (h2 : A = 3)
  (h3 : B = 5)
  (h4 : C = 6)
  (h5 : D = 5) :
  A = 3 ∧ B = 5 ∧ C = 6 ∧ D = 5 :=
by {
  sorry
}

end number_of_dots_on_faces_l1877_187738


namespace intersection_eq_l1877_187710

def set1 : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def set2 : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem intersection_eq : (set1 ∩ set2) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_eq_l1877_187710


namespace car_return_speed_l1877_187721

theorem car_return_speed (d : ℕ) (speed_CD : ℕ) (avg_speed_round_trip : ℕ) 
  (round_trip_distance : ℕ) (time_CD : ℕ) (time_round_trip : ℕ) (r: ℕ) 
  (h1 : d = 150) (h2 : speed_CD = 75) (h3 : avg_speed_round_trip = 60)
  (h4 : d * 2 = round_trip_distance) 
  (h5 : time_CD = d / speed_CD) 
  (h6 : time_round_trip = time_CD + d / r) 
  (h7 : avg_speed_round_trip = round_trip_distance / time_round_trip) :
  r = 50 :=
by {
  -- proof steps will go here
  sorry
}

end car_return_speed_l1877_187721


namespace expression_always_integer_l1877_187783

theorem expression_always_integer (m : ℕ) : 
  ∃ k : ℤ, (m / 3 + m^2 / 2 + m^3 / 6 : ℚ) = (k : ℚ) := 
sorry

end expression_always_integer_l1877_187783


namespace count_six_digit_palindromes_l1877_187735

def num_six_digit_palindromes : ℕ := 9000

theorem count_six_digit_palindromes :
  (∃ a b c d : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
     num_six_digit_palindromes = 9000) :=
sorry

end count_six_digit_palindromes_l1877_187735


namespace present_value_of_machine_l1877_187757

theorem present_value_of_machine {
  V0 : ℝ
} (h : 36100 = V0 * (0.95)^2) : V0 = 39978.95 :=
sorry

end present_value_of_machine_l1877_187757


namespace range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l1877_187794

variable {x m : ℝ}

-- First statement: Given m = 4 and p ∧ q, prove the range of x is 4 < x < 5
theorem range_of_x_given_p_and_q (m : ℝ) (h : m = 4) :
  (x^2 - 7*x + 10 < 0) ∧ (x^2 - 4*m*x + 3*m^2 < 0) → (4 < x ∧ x < 5) :=
sorry

-- Second statement: Prove the range of m given ¬q is a sufficient but not necessary condition for ¬p
theorem range_of_m_given_neg_q_sufficient_for_neg_p :
  (m ≤ 2) ∧ (3*m ≥ 5) ∧ (m > 0) → (5/3 ≤ m ∧ m ≤ 2) :=
sorry

end range_of_x_given_p_and_q_range_of_m_given_neg_q_sufficient_for_neg_p_l1877_187794


namespace no_x0_leq_zero_implies_m_gt_1_l1877_187798

theorem no_x0_leq_zero_implies_m_gt_1 (m : ℝ) :
  (¬ ∃ x0 : ℝ, x0^2 + 2 * x0 + m ≤ 0) ↔ m > 1 :=
sorry

end no_x0_leq_zero_implies_m_gt_1_l1877_187798


namespace greatest_perimeter_l1877_187728

theorem greatest_perimeter (w l : ℕ) (h1 : w * l = 12) : 
  ∃ (P : ℕ), P = 2 * (w + l) ∧ ∀ (w' l' : ℕ), w' * l' = 12 → 2 * (w' + l') ≤ P := 
sorry

end greatest_perimeter_l1877_187728


namespace random_walk_expected_distance_l1877_187787

noncomputable def expected_distance_after_random_walk (n : ℕ) : ℚ :=
(sorry : ℚ) -- We'll define this in the proof

-- Proof problem statement in Lean 4
theorem random_walk_expected_distance :
  expected_distance_after_random_walk 6 = 15 / 8 :=
by 
  sorry

end random_walk_expected_distance_l1877_187787


namespace find_f_7_l1877_187705

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_7 (h_odd : ∀ x, f (-x) = -f x)
                 (h_periodic : ∀ x, f (x + 4) = f x)
                 (h_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x ^ 2) :
  f 7 = -2 := 
sorry

end find_f_7_l1877_187705


namespace fraction_identity_l1877_187778

theorem fraction_identity (a b : ℝ) (h₀ : a^2 + a = 4) (h₁ : b^2 + b = 4) (h₂ : a ≠ b) :
  (b / a) + (a / b) = - (9 / 4) :=
sorry

end fraction_identity_l1877_187778


namespace fran_speed_l1877_187751

variable (s : ℝ)

theorem fran_speed
  (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ)
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5)
  (h4 : fran_time * s = joann_speed * joann_time)
  : s = 120 / 7 := 
by
  sorry

end fran_speed_l1877_187751


namespace roger_expenses_fraction_l1877_187739

theorem roger_expenses_fraction {B t s n : ℝ} (h1 : t = 0.25 * (B - s))
  (h2 : s = 0.10 * (B - t)) (h3 : n = 5) :
  (t + s + n) / B = 0.41 :=
sorry

end roger_expenses_fraction_l1877_187739


namespace find_p_l1877_187725

theorem find_p (h p : Polynomial ℝ) 
  (H1 : h + p = 3 * X^2 - X + 4)
  (H2 : h = X^4 - 5 * X^2 + X + 6) : 
  p = -X^4 + 8 * X^2 - 2 * X - 2 :=
sorry

end find_p_l1877_187725


namespace units_digit_quotient_l1877_187703

theorem units_digit_quotient (n : ℕ) (h1 : n % 2 = 1): 
  (4^n + 6^n) / 10 % 10 = 1 :=
by 
  -- Given the cyclical behavior of 4^n % 10 and 6^n % 10
  -- 4^n % 10 cycles between 4 and 6, 6^n % 10 is always 6
  -- Since n is odd, 4^n % 10 = 4 and 6^n % 10 = 6
  -- Adding them gives us 4 + 6 = 10, and thus a quotient of 1
  sorry

end units_digit_quotient_l1877_187703


namespace min_value_of_quadratic_l1877_187746

open Real

theorem min_value_of_quadratic 
  (x y z : ℝ) 
  (h : 3 * x + 2 * y + z = 1) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ 3 / 34 := 
sorry

end min_value_of_quadratic_l1877_187746


namespace seventh_graders_more_than_sixth_graders_l1877_187768

-- Definitions based on conditions
variables (S6 S7 : ℕ)
variable (h : 7 * S6 = 6 * S7)

-- Proposition based on the conclusion
theorem seventh_graders_more_than_sixth_graders (h : 7 * S6 = 6 * S7) : S7 > S6 :=
by {
  -- Skipping the proof with sorry
  sorry
}

end seventh_graders_more_than_sixth_graders_l1877_187768


namespace equipment_B_production_l1877_187779

theorem equipment_B_production
  (total_production : ℕ)
  (sample_size : ℕ)
  (A_sample_production : ℕ)
  (B_sample_production : ℕ)
  (A_total_production : ℕ)
  (B_total_production : ℕ)
  (total_condition : total_production = 4800)
  (sample_condition : sample_size = 80)
  (A_sample_condition : A_sample_production = 50)
  (B_sample_condition : B_sample_production = 30)
  (ratio_condition : (A_sample_production / B_sample_production) = (5 / 3))
  (production_condition : A_total_production + B_total_production = total_production) :
  B_total_production = 1800 := 
sorry

end equipment_B_production_l1877_187779


namespace find_initial_milk_amount_l1877_187715

-- Define the initial amount of milk as a variable in liters
variable (T : ℝ)

-- Given conditions
def consumed (T : ℝ) := 0.4 * T
def leftover (T : ℝ) := 0.69

-- The total milk at first was T if T = 0.69 / 0.6
theorem find_initial_milk_amount 
  (h1 : leftover T = 0.69)
  (h2 : consumed T = 0.4 * T) :
  T = 1.15 :=
by
  sorry

end find_initial_milk_amount_l1877_187715


namespace six_times_product_plus_one_equals_seven_pow_sixteen_l1877_187758

theorem six_times_product_plus_one_equals_seven_pow_sixteen :
  6 * (7 + 1) * (7^2 + 1) * (7^4 + 1) * (7^8 + 1) + 1 = 7^16 := 
  sorry

end six_times_product_plus_one_equals_seven_pow_sixteen_l1877_187758


namespace determinant_roots_l1877_187754

theorem determinant_roots (s p q a b c : ℂ) 
  (h : ∀ x : ℂ, x^3 - s*x^2 + p*x + q = (x - a) * (x - b) * (x - c)) :
  (1 + a) * ((1 + b) * (1 + c) - 1) - ((1) * (1 + c) - 1) + ((1) - (1 + b)) = p + 3 * s :=
by {
  -- expanded determinant calculations
  sorry
}

end determinant_roots_l1877_187754


namespace find_second_number_l1877_187740

theorem find_second_number (G N: ℕ) (h1: G = 101) (h2: 4351 % G = 8) (h3: N % G = 10) : N = 4359 :=
by 
  sorry

end find_second_number_l1877_187740


namespace minimum_for_specific_values_proof_minimum_for_arbitrary_values_proof_l1877_187700

noncomputable def minimum_for_specific_values : ℝ :=
  let m := 2 
  let n := 2 
  let p := 2 
  let xyz := 8 
  let x := 2
  let y := 2
  let z := 2
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem minimum_for_specific_values_proof : minimum_for_specific_values = 36 := by
  sorry

noncomputable def minimum_for_arbitrary_values (m n p : ℝ) (h : m * n * p = 8) : ℝ :=
  let x := 2
  let y := 2
  let z := 2
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem minimum_for_arbitrary_values_proof (m n p : ℝ) (h : m * n * p = 8) : minimum_for_arbitrary_values m n p h = 12 + 4 * (m + n + p) := by
  sorry

end minimum_for_specific_values_proof_minimum_for_arbitrary_values_proof_l1877_187700


namespace arithmetic_sequence_common_difference_l1877_187750

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h_a30 : a 30 = 100)
  (h_a100 : a 100 = 30) :
  d = -1 := sorry

end arithmetic_sequence_common_difference_l1877_187750


namespace radius_of_circle_l1877_187763

theorem radius_of_circle (P Q : ℝ) (h : P / Q = 25) : ∃ r : ℝ, 2 * π * r = Q ∧ π * r^2 = P ∧ r = 50 := 
by
  -- Proof starts here
  sorry

end radius_of_circle_l1877_187763


namespace fraction_not_covered_l1877_187727

/--
Given that frame X has a diameter of 16 cm and frame Y has a diameter of 12 cm,
prove that the fraction of the surface of frame X that is not covered by frame Y is 7/16.
-/
theorem fraction_not_covered (dX dY : ℝ) (hX : dX = 16) (hY : dY = 12) : 
  let rX := dX / 2
  let rY := dY / 2
  let AX := Real.pi * rX^2
  let AY := Real.pi * rY^2
  let uncovered_area := AX - AY
  let fraction_not_covered := uncovered_area / AX
  fraction_not_covered = 7 / 16 :=
by
  sorry

end fraction_not_covered_l1877_187727


namespace good_games_count_l1877_187745

-- Define the conditions
def games_from_friend : Nat := 50
def games_from_garage_sale : Nat := 27
def games_that_didnt_work : Nat := 74

-- Define the total games bought
def total_games_bought : Nat := games_from_friend + games_from_garage_sale

-- State the theorem to prove the number of good games
theorem good_games_count : total_games_bought - games_that_didnt_work = 3 :=
by
  sorry

end good_games_count_l1877_187745


namespace N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l1877_187713

-- Define the number with 1986 ones
def N : ℕ := (10^1986 - 1) / 9

-- Definition of having at least n distinct divisors
def has_at_least_n_distinct_divisors (num : ℕ) (n : ℕ) :=
  ∃ (divisors : Finset ℕ), divisors.card ≥ n ∧ ∀ d ∈ divisors, d ∣ num

theorem N_has_at_least_8_distinct_divisors :
  has_at_least_n_distinct_divisors N 8 :=
sorry

theorem N_has_at_least_32_distinct_divisors :
  has_at_least_n_distinct_divisors N 32 :=
sorry


end N_has_at_least_8_distinct_divisors_N_has_at_least_32_distinct_divisors_l1877_187713


namespace vector_addition_result_l1877_187748

-- Define the given vectors
def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (-3, 4)

-- Statement to prove that the sum of the vectors is (-1, 5)
theorem vector_addition_result : vector_a + vector_b = (-1, 5) :=
by
  -- Use the fact that vector addition in ℝ^2 is component-wise
  sorry

end vector_addition_result_l1877_187748


namespace simplify_fraction_l1877_187772

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  ( (x^2 + 1) / (x - 1) - (2*x) / (x - 1) ) = x - 1 :=
by
  -- Your proof steps would go here.
  sorry

end simplify_fraction_l1877_187772


namespace relationship_among_a_b_c_l1877_187759

noncomputable def a : ℝ := Real.log (Real.tan (70 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def b : ℝ := Real.log (Real.sin (25 * Real.pi / 180)) / Real.log (1 / 2)
noncomputable def c : ℝ := (1 / 2) ^ Real.cos (25 * Real.pi / 180)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  -- proofs would go here
  sorry

end relationship_among_a_b_c_l1877_187759


namespace opposite_seven_is_minus_seven_l1877_187788

theorem opposite_seven_is_minus_seven :
  ∃ x : ℤ, 7 + x = 0 ∧ x = -7 := 
sorry

end opposite_seven_is_minus_seven_l1877_187788


namespace neither_5_nice_nor_6_nice_count_l1877_187730

def is_k_nice (N k : ℕ) : Prop :=
  N % k = 1

def count_5_nice (N : ℕ) : ℕ :=
  (N - 1) / 5 + 1

def count_6_nice (N : ℕ) : ℕ :=
  (N - 1) / 6 + 1

def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

def count_30_nice (N : ℕ) : ℕ :=
  (N - 1) / 30 + 1

theorem neither_5_nice_nor_6_nice_count : 
  ∀ N < 200, 
  (N - (count_5_nice 199 + count_6_nice 199 - count_30_nice 199)) = 133 := 
by
  sorry

end neither_5_nice_nor_6_nice_count_l1877_187730


namespace borrowed_amount_correct_l1877_187781

variables (monthly_payment : ℕ) (months : ℕ) (total_payment : ℕ) (borrowed_amount : ℕ)

def total_payment_calculation (monthly_payment : ℕ) (months : ℕ) : ℕ :=
  monthly_payment * months

theorem borrowed_amount_correct :
  monthly_payment = 15 →
  months = 11 →
  total_payment = total_payment_calculation monthly_payment months →
  total_payment = 110 * borrowed_amount / 100 →
  borrowed_amount = 150 :=
by
  intros h1 h2 h3 h4
  sorry

end borrowed_amount_correct_l1877_187781


namespace regular_polygon_sides_l1877_187741

-- Define the conditions of the problem 
def is_regular_polygon (n : ℕ) (exterior_angle : ℝ) : Prop :=
  360 / n = exterior_angle

-- State the theorem
theorem regular_polygon_sides (h : is_regular_polygon n 18) : n = 20 := 
sorry

end regular_polygon_sides_l1877_187741


namespace tan_A_of_triangle_conditions_l1877_187785

open Real

def triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ B = π / 4

def form_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b^2 = a^2 + c^2

theorem tan_A_of_triangle_conditions
  (A B C a b c : ℝ)
  (h_angles : triangle_angles A B C)
  (h_seq : form_arithmetic_sequence a b c) :
  tan A = sqrt 2 - 1 :=
by
  sorry

end tan_A_of_triangle_conditions_l1877_187785


namespace fraction_to_decimal_l1877_187752

theorem fraction_to_decimal : (9 : ℚ) / 25 = 0.36 :=
by
  sorry

end fraction_to_decimal_l1877_187752


namespace triangle_with_angle_ratio_is_right_triangle_l1877_187701

theorem triangle_with_angle_ratio_is_right_triangle (x : ℝ) (h1 : 1 * x + 2 * x + 3 * x = 180) : 
  ∃ A B C : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x ∧ (A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end triangle_with_angle_ratio_is_right_triangle_l1877_187701
