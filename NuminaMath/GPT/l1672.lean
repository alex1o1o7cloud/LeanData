import Mathlib

namespace range_of_m_l1672_167262

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) (1 : ℝ), x^2 - 4 * x - 2 * m + 1 ≤ 0) ↔ m ∈ Set.Ici (3 : ℝ) := 
sorry

end range_of_m_l1672_167262


namespace mountaineering_team_problem_l1672_167229

structure Climber :=
  (total_students : ℕ)
  (advanced_climbers : ℕ)
  (intermediate_climbers : ℕ)
  (beginners : ℕ)

structure Experience :=
  (advanced_points : ℕ)
  (intermediate_points : ℕ)
  (beginner_points : ℕ)

structure TeamComposition :=
  (advanced_needed : ℕ)
  (intermediate_needed : ℕ)
  (beginners_needed : ℕ)
  (max_experience : ℕ)

def team_count (students : Climber) (xp : Experience) (comp : TeamComposition) : ℕ :=
  let total_experience := comp.advanced_needed * xp.advanced_points +
                          comp.intermediate_needed * xp.intermediate_points +
                          comp.beginners_needed * xp.beginner_points
  let max_teams_from_advanced := students.advanced_climbers / comp.advanced_needed
  let max_teams_from_intermediate := students.intermediate_climbers / comp.intermediate_needed
  let max_teams_from_beginners := students.beginners / comp.beginners_needed
  if total_experience ≤ comp.max_experience then
    min (max_teams_from_advanced) $ min (max_teams_from_intermediate) (max_teams_from_beginners)
  else 0

def problem : Prop :=
  team_count
    ⟨172, 45, 70, 57⟩
    ⟨80, 50, 30⟩
    ⟨5, 8, 5, 1000⟩ = 8

-- Let's declare the theorem now:
theorem mountaineering_team_problem : problem := sorry

end mountaineering_team_problem_l1672_167229


namespace solve_quadratic_equation_solve_linear_factor_equation_l1672_167219

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x^2 - 6 * x + 1 = 0 → (x = 3 - 2 * Real.sqrt 2 ∨ x = 3 + 2 * Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

theorem solve_linear_factor_equation :
  ∀ (x : ℝ), x * (2 * x - 1) = 2 * (2 * x - 1) → (x = 1 / 2 ∨ x = 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_equation_solve_linear_factor_equation_l1672_167219


namespace net_pay_rate_is_26_dollars_per_hour_l1672_167230

-- Defining the conditions
noncomputable def total_distance (time_hours : ℝ) (speed_mph : ℝ) : ℝ :=
  time_hours * speed_mph

noncomputable def adjusted_fuel_efficiency (original_efficiency : ℝ) (decrease_percentage : ℝ) : ℝ :=
  original_efficiency * (1 - decrease_percentage)

noncomputable def gasoline_used (distance : ℝ) (efficiency : ℝ) : ℝ :=
  distance / efficiency

noncomputable def earnings (rate_per_mile : ℝ) (distance : ℝ) : ℝ :=
  rate_per_mile * distance

noncomputable def updated_gasoline_price (original_price : ℝ) (increase_percentage : ℝ) : ℝ :=
  original_price * (1 + increase_percentage)

noncomputable def total_cost_gasoline (gasoline_price : ℝ) (gasoline_used : ℝ) : ℝ :=
  gasoline_price * gasoline_used

noncomputable def net_earnings (earnings : ℝ) (cost : ℝ) : ℝ :=
  earnings - cost

noncomputable def net_rate_of_pay (net_earnings : ℝ) (time_hours : ℝ) : ℝ :=
  net_earnings / time_hours

-- Given constants
def time_hours : ℝ := 3
def speed_mph : ℝ := 50
def original_efficiency : ℝ := 30
def decrease_percentage : ℝ := 0.10
def rate_per_mile : ℝ := 0.60
def original_gasoline_price : ℝ := 2.00
def increase_percentage : ℝ := 0.20

-- Proof problem statement
theorem net_pay_rate_is_26_dollars_per_hour :
  net_rate_of_pay 
    (net_earnings
       (earnings rate_per_mile (total_distance time_hours speed_mph))
       (total_cost_gasoline
          (updated_gasoline_price original_gasoline_price increase_percentage)
          (gasoline_used
            (total_distance time_hours speed_mph)
            (adjusted_fuel_efficiency original_efficiency decrease_percentage))))
    time_hours = 26 := 
  sorry

end net_pay_rate_is_26_dollars_per_hour_l1672_167230


namespace range_of_k_l1672_167231

theorem range_of_k 
  (h : ∀ x : ℝ, x = 1 → k^2 * x^2 - 6 * k * x + 8 ≥ 0) :
  k ≥ 4 ∨ k ≤ 2 := by
sorry

end range_of_k_l1672_167231


namespace find_b_l1672_167296

theorem find_b (a b c : ℚ) (h : (3 * x^2 - 4 * x + 2) * (a * x^2 + b * x + c) = 9 * x^4 - 10 * x^3 + 5 * x^2 - 8 * x + 4)
  (ha : a = 3) : b = 2 / 3 :=
by
  sorry

end find_b_l1672_167296


namespace probability_two_tails_two_heads_l1672_167265

theorem probability_two_tails_two_heads :
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  total_probability = 3 / 8 :=
by
  let num_coins := 4
  let num_tails_heads := 2
  let num_sequences := Nat.choose num_coins num_tails_heads
  let single_probability := (1 / 2) ^ num_coins
  let total_probability := num_sequences * single_probability
  sorry

end probability_two_tails_two_heads_l1672_167265


namespace simplify_fraction_l1672_167237

theorem simplify_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b^2 - b^3) / (a * b - a^3) = (a^2 + a * b + b^2) / b :=
by {
  -- Proof skipped
  sorry
}

end simplify_fraction_l1672_167237


namespace rearrange_cards_l1672_167278

theorem rearrange_cards :
  (∀ (arrangement : List ℕ), arrangement = [3, 1, 2, 4, 5, 6] ∨ arrangement = [1, 2, 4, 5, 6, 3] →
  (∀ card, card ∈ arrangement → List.erase arrangement card = [1, 2, 4, 5, 6] ∨
                                        List.erase arrangement card = [3, 1, 2, 4, 5]) →
  List.length arrangement = 6) →
  (∃ n, n = 10) :=
by
  sorry

end rearrange_cards_l1672_167278


namespace repeating_decimal_to_fraction_denominator_l1672_167250

theorem repeating_decimal_to_fraction_denominator :
  ∀ (S : ℚ), (S = 0.27) → (∃ a b : ℤ, b ≠ 0 ∧ S = a / b ∧ Int.gcd a b = 1 ∧ b = 3) :=
by
  sorry

end repeating_decimal_to_fraction_denominator_l1672_167250


namespace frogs_need_new_pond_l1672_167213

theorem frogs_need_new_pond
  (num_frogs : ℕ) 
  (num_tadpoles : ℕ) 
  (num_survivor_tadpoles : ℕ) 
  (pond_capacity : ℕ) 
  (hc1 : num_frogs = 5)
  (hc2 : num_tadpoles = 3 * num_frogs)
  (hc3 : num_survivor_tadpoles = (2 * num_tadpoles) / 3)
  (hc4 : pond_capacity = 8):
  ((num_frogs + num_survivor_tadpoles) - pond_capacity) = 7 :=
by sorry

end frogs_need_new_pond_l1672_167213


namespace school_points_l1672_167251

theorem school_points (a b c : ℕ) (h1 : a + b + c = 285)
  (h2 : ∃ x : ℕ, a - 8 = x ∧ b - 12 = x ∧ c - 7 = x) : a + c = 187 :=
sorry

end school_points_l1672_167251


namespace inequality_proof_l1672_167297

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
sorry

end inequality_proof_l1672_167297


namespace complement_union_eq_l1672_167299

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l1672_167299


namespace nate_matches_left_l1672_167260

def initial_matches : ℕ := 70
def matches_dropped : ℕ := 10
def matches_eaten : ℕ := 2 * matches_dropped
def total_matches_lost : ℕ := matches_dropped + matches_eaten
def remaining_matches : ℕ := initial_matches - total_matches_lost

theorem nate_matches_left : remaining_matches = 40 := by
  sorry

end nate_matches_left_l1672_167260


namespace moles_of_HCl_formed_l1672_167221

-- Conditions: 1 mole of Methane (CH₄) and 2 moles of Chlorine (Cl₂)
def methane := 1 -- 1 mole of methane
def chlorine := 2 -- 2 moles of chlorine

-- Reaction: CH₄ + Cl₂ → CH₃Cl + HCl
-- We state that 1 mole of methane reacts with 1 mole of chlorine to form 1 mole of hydrochloric acid
def reaction (methane chlorine : ℕ) : ℕ := methane

-- Theorem: Prove 1 mole of hydrochloric acid (HCl) is formed
theorem moles_of_HCl_formed : reaction methane chlorine = 1 := by
  sorry

end moles_of_HCl_formed_l1672_167221


namespace percent_of_a_is_4b_l1672_167274

variable (a b : ℝ)
variable (h : a = 1.2 * b)

theorem percent_of_a_is_4b :
  (4 * b) = (10 / 3 * 100 * a) / 100 :=
by sorry

end percent_of_a_is_4b_l1672_167274


namespace journey_speed_second_half_l1672_167210

theorem journey_speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (v : ℝ) : 
  total_time = 10 ∧ first_half_speed = 21 ∧ total_distance = 224 →
  v = 24 :=
by
  intro h
  sorry

end journey_speed_second_half_l1672_167210


namespace size_ratio_l1672_167279

variable {A B C : ℝ} -- Declaring that A, B, and C are real numbers (their sizes)
variable (h1 : A = 3 * B) -- A is three times the size of B
variable (h2 : B = (1 / 2) * C) -- B is half the size of C

theorem size_ratio (h1 : A = 3 * B) (h2 : B = (1 / 2) * C) : A / C = 1.5 :=
by
  sorry -- Proof goes here, to be completed

end size_ratio_l1672_167279


namespace shopkeeper_milk_sold_l1672_167214

theorem shopkeeper_milk_sold :
  let morning_packets := 150
  let morning_250 := 60
  let morning_300 := 40
  let morning_350 := morning_packets - morning_250 - morning_300
  
  let evening_packets := 100
  let evening_400 := evening_packets * 50 / 100
  let evening_500 := evening_packets * 25 / 100
  let evening_450 := evening_packets * 25 / 100

  let morning_milk := morning_250 * 250 + morning_300 * 300 + morning_350 * 350
  let evening_milk := evening_400 * 400 + evening_500 * 500 + evening_450 * 450
  let total_milk := morning_milk + evening_milk

  let remaining_milk := 42000
  let sold_milk := total_milk - remaining_milk

  let ounces_per_mil := 1 / 30
  let sold_milk_ounces := sold_milk * ounces_per_mil

  sold_milk_ounces = 1541.67 := by sorry

end shopkeeper_milk_sold_l1672_167214


namespace mrs_hilt_money_left_l1672_167290

theorem mrs_hilt_money_left (initial_money : ℕ) (cost_of_pencil : ℕ) (money_left : ℕ) (h1 : initial_money = 15) (h2 : cost_of_pencil = 11) : money_left = 4 :=
by
  sorry

end mrs_hilt_money_left_l1672_167290


namespace cost_of_downloading_360_songs_in_2005_is_144_dollars_l1672_167201

theorem cost_of_downloading_360_songs_in_2005_is_144_dollars :
  (∀ (c_2004 c_2005 : ℕ), (∀ c : ℕ, c_2005 = c ∧ c_2004 = c + 32) →
  200 * c_2004 = 360 * c_2005 → 360 * c_2005 / 100 = 144) :=
  by sorry

end cost_of_downloading_360_songs_in_2005_is_144_dollars_l1672_167201


namespace power_mean_inequality_l1672_167247

theorem power_mean_inequality
  (n : ℕ) (hn : 0 < n) (x1 x2 : ℝ) :
  (x1^n + x2^n)^(n+1) / (x1^(n-1) + x2^(n-1))^n ≤ (x1^(n+1) + x2^(n+1))^n / (x1^n + x2^n)^(n-1) :=
by
  sorry

end power_mean_inequality_l1672_167247


namespace problem_l1672_167211

theorem problem {x y n : ℝ} 
  (h1 : 2 * x + y = 4) 
  (h2 : (x + y) / 3 = 1) 
  (h3 : x + 2 * y = n) : n = 5 := 
sorry

end problem_l1672_167211


namespace quadratic_form_rewrite_l1672_167284

theorem quadratic_form_rewrite (x : ℝ) : 2 * x ^ 2 + 7 = 4 * x → 2 * x ^ 2 - 4 * x + 7 = 0 :=
by
    intro h
    linarith

end quadratic_form_rewrite_l1672_167284


namespace initial_students_count_l1672_167261

variable (initial_students : ℕ)
variable (number_of_new_boys : ℕ := 5)
variable (initial_percentage_girls : ℝ := 0.40)
variable (new_percentage_girls : ℝ := 0.32)

theorem initial_students_count (h : initial_percentage_girls * initial_students = new_percentage_girls * (initial_students + number_of_new_boys)) : 
  initial_students = 20 := 
by 
  sorry

end initial_students_count_l1672_167261


namespace initial_pencils_correct_l1672_167287

variable (pencils_taken remaining_pencils initial_pencils : ℕ)

def initial_number_of_pencils (pencils_taken remaining_pencils : ℕ) : ℕ :=
  pencils_taken + remaining_pencils

theorem initial_pencils_correct (h₁ : pencils_taken = 22) (h₂ : remaining_pencils = 12) :
  initial_number_of_pencils pencils_taken remaining_pencils = 34 := by
  rw [h₁, h₂]
  rfl

end initial_pencils_correct_l1672_167287


namespace smallest_possible_z_l1672_167233

theorem smallest_possible_z (w x y z : ℕ) (k : ℕ) (h1 : w = x - 1) (h2 : y = x + 1) (h3 : z = x + 2)
  (h4 : w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z) (h5 : k = 2) (h6 : w^3 + x^3 + y^3 = k * z^3) : z = 6 :=
by
  sorry

end smallest_possible_z_l1672_167233


namespace exist_functions_fg_neq_f1f1_g1g1_l1672_167292

-- Part (a)
theorem exist_functions_fg :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, (f ∘ g) x = (g ∘ f) x) ∧ 
    (∀ x, (f ∘ f) x = (g ∘ g) x) ∧ 
    (∀ x, f x ≠ g x) := 
sorry

-- Part (b)
theorem neq_f1f1_g1g1 
  (f1 g1 : ℝ → ℝ)
  (H_comm : ∀ x, (f1 ∘ g1) x = (g1 ∘ f1) x)
  (H_neq: ∀ x, f1 x ≠ g1 x) :
  ∀ x, (f1 ∘ f1) x ≠ (g1 ∘ g1) x :=
sorry

end exist_functions_fg_neq_f1f1_g1g1_l1672_167292


namespace max_value_a_l1672_167249

noncomputable def setA (a : ℝ) : Set ℝ := { x | (x - 1) * (x - a) ≥ 0 }
noncomputable def setB (a : ℝ) : Set ℝ := { x | x ≥ a - 1 }

theorem max_value_a (a : ℝ) :
  (setA a ∪ setB a = Set.univ) → a ≤ 2 := by
  sorry

end max_value_a_l1672_167249


namespace cups_added_l1672_167257

/--
A bowl was half full of water. Some cups of water were then added to the bowl, filling the bowl to 70% of its capacity. There are now 14 cups of water in the bowl.
Prove that the number of cups of water added to the bowl is 4.
-/
theorem cups_added (C : ℚ) (h1 : C / 2 + 0.2 * C = 14) : 
  14 - C / 2 = 4 :=
by
  sorry

end cups_added_l1672_167257


namespace max_k_condition_l1672_167268

theorem max_k_condition (k : ℕ) (total_goods : ℕ) (num_platforms : ℕ) (platform_capacity : ℕ) :
  total_goods = 1500 ∧ num_platforms = 25 ∧ platform_capacity = 80 → 
  (∀ (c : ℕ), 1 ≤ c ∧ c ≤ k → c ∣ k) → 
  (∀ (total : ℕ), total ≤ num_platforms * platform_capacity → total ≥ total_goods) → 
  k ≤ 26 := 
sorry

end max_k_condition_l1672_167268


namespace sufficient_not_necessary_l1672_167244

-- Define set A and set B
def setA (x : ℝ) := x > 5
def setB (x : ℝ) := x > 3

-- Statement:
theorem sufficient_not_necessary (x : ℝ) : setA x → setB x :=
by
  intro h
  exact sorry

end sufficient_not_necessary_l1672_167244


namespace max_value_expr_l1672_167255

-- Define the expression
def expr (a b c d : ℝ) : ℝ :=
  a + b + c + d - a * b - b * c - c * d - d * a

-- The main theorem
theorem max_value_expr :
  (∀ (a b c d : ℝ), 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 → expr a b c d ≤ 2) ∧
  (∃ (a b c d : ℝ), 0 ≤ a ∧ a = 1 ∧ 0 ≤ b ∧ b = 0 ∧ 0 ≤ c ∧ c = 1 ∧ 0 ≤ d ∧ d = 0 ∧ expr a b c d = 2) :=
  by
  sorry

end max_value_expr_l1672_167255


namespace overlap_length_l1672_167224

-- Variables in the conditions
variables (tape_length overlap total_length : ℕ)

-- Conditions
def two_tapes_overlap := (tape_length + tape_length - overlap = total_length)

-- The proof statement we need to prove
theorem overlap_length (h : two_tapes_overlap 275 overlap 512) : overlap = 38 :=
by
  sorry

end overlap_length_l1672_167224


namespace Marty_combinations_l1672_167223

theorem Marty_combinations:
  let colors := ({blue, green, yellow, black, white} : Finset String)
  let tools := ({brush, roller, sponge, spray_gun} : Finset String)
  colors.card * tools.card = 20 := 
by
  sorry

end Marty_combinations_l1672_167223


namespace select_3_products_select_exactly_1_defective_select_at_least_1_defective_l1672_167200

noncomputable def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

namespace ProductInspection

def total_products : Nat := 100
def qualified_products : Nat := 98
def defective_products : Nat := 2

-- Proof Problem 1
theorem select_3_products (h : combination total_products 3 = 161700) : True := by
  trivial

-- Proof Problem 2
theorem select_exactly_1_defective (h : combination defective_products 1 * combination qualified_products 2 = 9506) : True := by
  trivial

-- Proof Problem 3
theorem select_at_least_1_defective (h : combination total_products 3 - combination qualified_products 3 = 9604) : True := by
  trivial

end ProductInspection

end select_3_products_select_exactly_1_defective_select_at_least_1_defective_l1672_167200


namespace g_of_f_of_3_is_1852_l1672_167243

def f (x : ℤ) : ℤ := x^3 - 2
def g (x : ℤ) : ℤ := 3 * x^2 - x + 2

theorem g_of_f_of_3_is_1852 : g (f 3) = 1852 := by
  sorry

end g_of_f_of_3_is_1852_l1672_167243


namespace solution_to_equation_l1672_167222

noncomputable def equation (x : ℝ) : ℝ := 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (5 - 9 * x) / (x - 2) + 2

theorem solution_to_equation :
  equation 3.294 = 0 ∧ equation (-0.405) = 0 :=
by
  sorry

end solution_to_equation_l1672_167222


namespace union_of_A_B_l1672_167285

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end union_of_A_B_l1672_167285


namespace veronica_max_area_l1672_167280

noncomputable def max_area_garden : ℝ :=
  let l := 105
  let w := 420 - 2 * l
  l * w

theorem veronica_max_area : ∃ (A : ℝ), max_area_garden = 22050 :=
by
  use 22050
  show max_area_garden = 22050
  sorry

end veronica_max_area_l1672_167280


namespace minimum_red_pieces_l1672_167241

theorem minimum_red_pieces (w b r : ℕ) 
  (h1 : b ≤ w / 2) 
  (h2 : r ≥ 3 * b) 
  (h3 : w + b ≥ 55) : r = 57 := 
sorry

end minimum_red_pieces_l1672_167241


namespace final_value_A_eq_B_pow_N_l1672_167258

-- Definitions of conditions
def compute_A (A B : ℕ) (N : ℕ) : ℕ :=
    if N ≤ 0 then 
        1 
    else 
        let rec compute_loop (A' B' N' : ℕ) : ℕ :=
            if N' = 0 then A' 
            else 
                let B'' := B' * B'
                let N'' := N' / 2
                let A'' := if N' % 2 = 1 then A' * B' else A'
                compute_loop A'' B'' N'' 
        compute_loop A B N

-- Theorem statement
theorem final_value_A_eq_B_pow_N (A B N : ℕ) : compute_A A B N = B ^ N :=
    sorry

end final_value_A_eq_B_pow_N_l1672_167258


namespace sum_of_squares_of_roots_l1672_167293

theorem sum_of_squares_of_roots (x₁ x₂ : ℚ) (h : 6 * x₁^2 - 9 * x₁ + 5 = 0 ∧ 6 * x₂^2 - 9 * x₂ + 5 = 0 ∧ x₁ ≠ x₂) : x₁^2 + x₂^2 = 7 / 12 :=
by
  -- Since we are only required to write the statement, we leave the proof as sorry
  sorry

end sum_of_squares_of_roots_l1672_167293


namespace no_fixed_points_l1672_167207

def f (x a : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem no_fixed_points (a : ℝ) :
  (∀ x : ℝ, f x a ≠ x) ↔ (-1/2 < a ∧ a < 3/2) := by
    sorry

end no_fixed_points_l1672_167207


namespace batteries_difference_is_correct_l1672_167236

-- Define the number of batteries used in each item
def flashlights_batteries : ℝ := 3.5
def toys_batteries : ℝ := 15.75
def remote_controllers_batteries : ℝ := 7.25
def wall_clock_batteries : ℝ := 4.8
def wireless_mouse_batteries : ℝ := 3.4

-- Define the combined total of batteries used in the other items
def combined_total : ℝ := flashlights_batteries + remote_controllers_batteries + wall_clock_batteries + wireless_mouse_batteries

-- Define the difference between the total number of batteries used in toys and the combined total of other items
def batteries_difference : ℝ := toys_batteries - combined_total

theorem batteries_difference_is_correct : batteries_difference = -3.2 :=
by
  sorry

end batteries_difference_is_correct_l1672_167236


namespace problem_statement_l1672_167283

variable {a : ℕ+ → ℝ} 

theorem problem_statement (h : ∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) :
  (∀ n : ℕ+, a (n + 1) < a n) ∧ -- Sequence is decreasing (original proposition)
  (∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n)) ∧ -- Inverse
  ((∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) → (∀ n : ℕ+, a (n + 1) < a n)) ∧ -- Converse
  ((∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n))) -- Contrapositive
:= by
  sorry

end problem_statement_l1672_167283


namespace monotonic_when_a_is_neg1_find_extreme_points_l1672_167212

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x ^ 3 - (1/2) * (a^2 + a + 2) * x ^ 2 + a^2 * (a + 2) * x

theorem monotonic_when_a_is_neg1 :
  ∀ x : ℝ, f x (-1) ≤ f x (-1) :=
sorry

theorem find_extreme_points (a : ℝ) :
  if h : a = -1 ∨ a = 2 then
    True  -- The function is monotonically increasing, no extreme points
  else if h : a < -1 ∨ a > 2 then
    ∃ x_max x_min : ℝ, x_max = a + 2 ∧ x_min = a^2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) 
  else
    ∃ x_max x_min : ℝ, x_max = a^2 ∧ x_min = a + 2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) :=
sorry

end monotonic_when_a_is_neg1_find_extreme_points_l1672_167212


namespace exists_integers_not_all_zero_l1672_167228

-- Given conditions
variables (a b c : ℝ)
variables (ab bc ca : ℚ)
variables (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca)
variables (x y z : ℤ)

-- The theorem to prove
theorem exists_integers_not_all_zero (ha : a * b = ab) (hb : b * c = bc) (hc : c * a = ca):
  ∃ (x y z : ℤ), (¬ (x = 0 ∧ y = 0 ∧ z = 0)) ∧ (a * x + b * y + c * z = 0) :=
sorry

end exists_integers_not_all_zero_l1672_167228


namespace lcm_of_fractions_l1672_167202

-- Definitions based on the problem's conditions
def numerators : List ℕ := [7, 8, 3, 5, 13, 15, 22, 27]
def denominators : List ℕ := [10, 9, 8, 12, 14, 100, 45, 35]

-- LCM and GCD functions for lists of natural numbers
def list_lcm (l : List ℕ) : ℕ := l.foldr lcm 1
def list_gcd (l : List ℕ) : ℕ := l.foldr gcd 0

-- Main proposition
theorem lcm_of_fractions : list_lcm numerators / list_gcd denominators = 13860 :=
by {
  -- to be proven
  sorry
}

end lcm_of_fractions_l1672_167202


namespace inverse_undefined_at_one_l1672_167271

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 5)

theorem inverse_undefined_at_one : ∀ (x : ℝ), (x = 1) → ¬∃ y : ℝ, f y = x :=
by
  sorry

end inverse_undefined_at_one_l1672_167271


namespace moles_of_HNO3_l1672_167246

theorem moles_of_HNO3 (HNO3 NaHCO3 NaNO3 : ℝ)
  (h1 : NaHCO3 = 1) (h2 : NaNO3 = 1) :
  HNO3 = 1 :=
by sorry

end moles_of_HNO3_l1672_167246


namespace no_such_function_exists_l1672_167239

theorem no_such_function_exists (f : ℝ → ℝ) (Hf : ∀ x : ℝ, 2 * f (Real.cos x) = f (Real.sin x) + Real.sin x) : False :=
by
  sorry

end no_such_function_exists_l1672_167239


namespace average_letters_per_day_l1672_167277

theorem average_letters_per_day 
  (letters_tuesday : ℕ)
  (letters_wednesday : ℕ)
  (days : ℕ := 2) 
  (letters_total : ℕ := letters_tuesday + letters_wednesday) :
  letters_tuesday = 7 → letters_wednesday = 3 → letters_total / days = 5 :=
by
  -- The proof is omitted
  sorry

end average_letters_per_day_l1672_167277


namespace result_when_j_divided_by_26_l1672_167269

noncomputable def j := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 10 11) (Nat.lcm 12 13)) (Nat.lcm 14 15))

theorem result_when_j_divided_by_26 : j / 26 = 2310 := by 
  sorry

end result_when_j_divided_by_26_l1672_167269


namespace propositions_alpha_and_beta_true_l1672_167235

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = -f (-x)

def strictly_increasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x < f y

def strictly_decreasing_function (f : ℝ → ℝ) : Prop :=
∀ x y, x < y → f x > f y

def alpha (f : ℝ → ℝ) : Prop :=
∀ x, ∃ g h : ℝ → ℝ, even_function g ∧ odd_function h ∧ f x = g x + h x

def beta (f : ℝ → ℝ) : Prop :=
∀ x, strictly_increasing_function f → ∃ p q : ℝ → ℝ, 
  strictly_increasing_function p ∧ strictly_decreasing_function q ∧ f x = p x + q x

theorem propositions_alpha_and_beta_true (f : ℝ → ℝ) :
  alpha f ∧ beta f :=
by
  sorry

end propositions_alpha_and_beta_true_l1672_167235


namespace problem_statement_l1672_167220

namespace LeanProofExample

def not_divisible (n : ℕ) (p : ℕ) : Prop :=
  ¬(p ∣ n)

theorem problem_statement (x y : ℕ) 
  (hx : not_divisible x 59) 
  (hy : not_divisible y 59)
  (h : 3 * x + 28 * y ≡ 0 [MOD 59]) :
  ¬(5 * x + 16 * y ≡ 0 [MOD 59]) :=
  sorry

end LeanProofExample

end problem_statement_l1672_167220


namespace no_x_axis_intersection_iff_l1672_167238

theorem no_x_axis_intersection_iff (m : ℝ) :
    (∀ x : ℝ, x^2 - x + m ≠ 0) ↔ m > 1 / 4 :=
by
  sorry

end no_x_axis_intersection_iff_l1672_167238


namespace find_x_l1672_167295

variable (n : ℝ) (x : ℝ)

theorem find_x (h1 : n = 15.0) (h2 : 3 * n - x = 40) : x = 5.0 :=
by
  sorry

end find_x_l1672_167295


namespace matrix_expression_l1672_167204

variable {F : Type} [Field F] {n : Type} [Fintype n] [DecidableEq n]
variable (B : Matrix n n F)

-- Suppose B is invertible
variable [Invertible B]

-- Condition given in the problem
theorem matrix_expression (h : (B - 3 • (1 : Matrix n n F)) * (B - 5 • (1 : Matrix n n F)) = 0) :
  B + 10 • (B⁻¹) = 10 • (B⁻¹) + (32 / 3 : F) • (1 : Matrix n n F) :=
sorry

end matrix_expression_l1672_167204


namespace leftover_stickers_l1672_167272

-- Definitions for each person's stickers
def ninaStickers : ℕ := 53
def oliverStickers : ℕ := 68
def pattyStickers : ℕ := 29

-- The number of stickers in a package
def packageSize : ℕ := 18

-- The total number of stickers
def totalStickers : ℕ := ninaStickers + oliverStickers + pattyStickers

-- Proof that the number of leftover stickers is 6 when all stickers are divided into packages of 18
theorem leftover_stickers : totalStickers % packageSize = 6 := by
  sorry

end leftover_stickers_l1672_167272


namespace div_power_sub_one_l1672_167294

theorem div_power_sub_one : 11 * 31 * 61 ∣ 20^15 - 1 := 
by
  sorry

end div_power_sub_one_l1672_167294


namespace solve_for_m_l1672_167203

def A := {x : ℝ | x^2 + 3*x - 10 ≤ 0}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem solve_for_m (m : ℝ) (h : B m ⊆ A) : m < 2 :=
by
  sorry

end solve_for_m_l1672_167203


namespace admission_cutoff_score_l1672_167234

theorem admission_cutoff_score (n : ℕ) (x : ℚ) (admitted_average non_admitted_average total_average : ℚ)
    (h1 : admitted_average = x + 15)
    (h2 : non_admitted_average = x - 20)
    (h3 : total_average = 90)
    (h4 : (admitted_average * (2 / 5) + non_admitted_average * (3 / 5)) = total_average) : x = 96 := 
by
  sorry

end admission_cutoff_score_l1672_167234


namespace correct_ordering_of_powers_l1672_167232

theorem correct_ordering_of_powers : 
  7^8 < 3^15 ∧ 3^15 < 4^12 ∧ 4^12 < 8^10 :=
  by
    sorry

end correct_ordering_of_powers_l1672_167232


namespace correct_operation_l1672_167245

noncomputable def check_operations : Prop :=
    ∀ (a : ℝ), ( a^6 / a^3 = a^3 ) ∧ 
               ¬( 3 * a^5 + a^5 = 4 * a^10 ) ∧
               ¬( (2 * a)^3 = 2 * a^3 ) ∧
               ¬( (a^2)^4 = a^6 )

theorem correct_operation : check_operations :=
by
  intro a
  have h1 : a^6 / a^3 = a^3 := by
    sorry
  have h2 : ¬(3 * a^5 + a^5 = 4 * a^10) := by
    sorry
  have h3 : ¬((2 * a)^3 = 2 * a^3) := by
    sorry
  have h4 : ¬((a^2)^4 = a^6) := by
    sorry
  exact ⟨h1, h2, h3, h4⟩

end correct_operation_l1672_167245


namespace more_newborn_elephants_than_baby_hippos_l1672_167275

-- Define the given conditions
def initial_elephants := 20
def initial_hippos := 35
def female_frac := 5 / 7
def births_per_female_hippo := 5
def total_animals_after_birth := 315

-- Calculate the required values
def female_hippos := female_frac * initial_hippos
def baby_hippos := female_hippos * births_per_female_hippo
def total_animals_before_birth := initial_elephants + initial_hippos
def total_newborns := total_animals_after_birth - total_animals_before_birth
def newborn_elephants := total_newborns - baby_hippos

-- Define the proof statement
theorem more_newborn_elephants_than_baby_hippos :
  (newborn_elephants - baby_hippos) = 10 :=
by
  sorry

end more_newborn_elephants_than_baby_hippos_l1672_167275


namespace problem_solution_l1672_167226

-- Definitions and Assumptions
variable (f : ℝ → ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, f x - (deriv^[2]) f x > 0)

-- Statement to Prove
theorem problem_solution : e * f 2015 > f 2016 :=
by
  sorry

end problem_solution_l1672_167226


namespace Tammy_second_day_speed_l1672_167205

variable (v t : ℝ)

/-- This statement represents Tammy's climbing situation -/
theorem Tammy_second_day_speed:
  (t + (t - 2) = 14) ∧
  (v * t + (v + 0.5) * (t - 2) = 52) →
  (v + 0.5 = 4) :=
by
  sorry

end Tammy_second_day_speed_l1672_167205


namespace alice_safe_paths_l1672_167215

/-
Define the coordinate system and conditions.
-/

def total_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

def paths_through_dangerous_area : ℕ :=
  (total_paths 2 2) * (total_paths 2 1)

def safe_paths : ℕ :=
  total_paths 4 3 - paths_through_dangerous_area

theorem alice_safe_paths : safe_paths = 17 := by
  sorry

end alice_safe_paths_l1672_167215


namespace find_excluded_digit_l1672_167217

theorem find_excluded_digit (a b : ℕ) (d : ℕ) (h : a * b = 1024) (ha : a % 10 ≠ d) (hb : b % 10 ≠ d) : 
  ∃ r : ℕ, d = r ∧ r < 10 :=
by 
  sorry

end find_excluded_digit_l1672_167217


namespace find_x_in_interval_l1672_167225

theorem find_x_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) (h_eq : (2 - Real.sin (2 * x)) * Real.sin (x + π / 4) = 1) : x = π / 4 := 
sorry

end find_x_in_interval_l1672_167225


namespace number_of_throwers_l1672_167263

theorem number_of_throwers (total_players throwers right_handed : ℕ) 
  (h1 : total_players = 64)
  (h2 : right_handed = 55) 
  (h3 : ∀ T N, T + N = total_players → 
  T + (2/3 : ℚ) * N = right_handed) : 
  throwers = 37 := 
sorry

end number_of_throwers_l1672_167263


namespace solve_problem_l1672_167227

noncomputable def f : ℝ → ℝ
| x => if x > 0 then Real.logb 2 x else 3^x

theorem solve_problem : f (f (1 / 2)) = 1 / 3 := by
  sorry

end solve_problem_l1672_167227


namespace wind_speed_l1672_167273

theorem wind_speed (w : ℝ) (h : 420 / (253 + w) = 350 / (253 - w)) : w = 23 :=
by
  sorry

end wind_speed_l1672_167273


namespace compute_K_l1672_167298

theorem compute_K (P Q T N K : ℕ) (x y z : ℕ) 
  (hP : P * x + Q * y = z) 
  (hT : T * x + N * y = z)
  (hK : K * x = z)
  (h_unique : P > 0 ∧ Q > 0 ∧ T > 0 ∧ N > 0 ∧ K > 0) :
  K = (P * K - T * Q) / (N - Q) :=
by sorry

end compute_K_l1672_167298


namespace find_m_l1672_167289

theorem find_m (a b c d : ℕ) (m : ℕ) (a_n b_n c_n d_n: ℕ → ℕ)
  (ha : ∀ n, a_n n = a * n + b)
  (hb : ∀ n, b_n n = c * n + d)
  (hc : ∀ n, c_n n = a_n n * b_n n)
  (hd : ∀ n, d_n n = c_n (n + 1) - c_n n)
  (ha1b1 : m = a_n 1 * b_n 1)
  (hca2b2 : a_n 2 * b_n 2 = 4)
  (hca3b3 : a_n 3 * b_n 3 = 8)
  (hca4b4 : a_n 4 * b_n 4 = 16) :
  m = 4 := 
by sorry

end find_m_l1672_167289


namespace general_formula_sum_and_min_value_l1672_167253

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def a1 := (a 1 = -5)
def a_condition := (3 * a 3 + a 5 = 0)

-- Prove the general formula for an arithmetic sequence
theorem general_formula (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0) : 
  ∀ n, a n = 2 * n - 7 := 
by
  sorry

-- Using the general formula to find the sum Sn and its minimum value
theorem sum_and_min_value (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0)
  (h : ∀ n, a n = 2 * n - 7) : 
  ∀ n, S n = n^2 - 6 * n ∧ ∃ n, S n = -9 :=
by
  sorry

end general_formula_sum_and_min_value_l1672_167253


namespace parabola_focus_l1672_167286

theorem parabola_focus (x y : ℝ) : (y = x^2 / 8) → (y = x^2 / 8) ∧ (∃ p, p = (0, 2)) :=
by
  sorry

end parabola_focus_l1672_167286


namespace max_students_can_participate_l1672_167282

theorem max_students_can_participate (max_funds rent cost_per_student : ℕ) (h_max_funds : max_funds = 800) (h_rent : rent = 300) (h_cost_per_student : cost_per_student = 15) :
  ∃ x : ℕ, x ≤ (max_funds - rent) / cost_per_student ∧ x = 33 :=
by
  sorry

end max_students_can_participate_l1672_167282


namespace copy_pages_l1672_167206

theorem copy_pages (total_cents : ℕ) (cost_per_page : ℕ) (h1 : total_cents = 1500) (h2 : cost_per_page = 5) : 
  (total_cents / cost_per_page = 300) :=
sorry

end copy_pages_l1672_167206


namespace arrange_order_l1672_167288

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := Real.log 2 / Real.log 3
noncomputable def c : Real := Real.cos 2

theorem arrange_order : c < b ∧ b < a :=
by
  sorry

end arrange_order_l1672_167288


namespace part_I_part_II_l1672_167267

-- Define the function f
def f (x: ℝ) : ℝ := abs (x - 1) - 2 * abs (x + 1)

-- The conditions and questions transformed into Lean statements
theorem part_I : ∃ m, (∀ x: ℝ, f x ≤ m) ∧ (m = f (-1)) ∧ (m = 2) := by
  sorry

theorem part_II (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : a^2 + 3 * b^2 + 2 * c^2 = 2) : 
  ∃ n, (∀ a b c : ℝ, (0 < a ∧ 0 < b ∧ 0 < c) ∧ (a^2 + 3 * b^2 + 2 * c^2 = 2) → ab + 2 * bc ≤ n) ∧ (n = 1) := by
  sorry

end part_I_part_II_l1672_167267


namespace positive_integer_perfect_square_l1672_167276

theorem positive_integer_perfect_square (n : ℕ) (h1: n > 0) (h2 : ∃ k : ℕ, n^2 - 19 * n - 99 = k^2) : n = 199 :=
sorry

end positive_integer_perfect_square_l1672_167276


namespace longest_side_of_triangle_l1672_167256

theorem longest_side_of_triangle (a b c : ℕ) (h1 : a = 3) (h2 : b = 5) 
    (cond : a^2 + b^2 - 6 * a - 10 * b + 34 = 0) 
    (triangle_ineq1 : a + b > c)
    (triangle_ineq2 : a + c > b)
    (triangle_ineq3 : b + c > a)
    (hScalene: a ≠ b ∧ b ≠ c ∧ a ≠ c) : c = 6 ∨ c = 7 := 
by {
  sorry
}

end longest_side_of_triangle_l1672_167256


namespace people_per_team_l1672_167270

theorem people_per_team 
  (managers : ℕ) (employees : ℕ) (teams : ℕ) 
  (h1 : managers = 23) (h2 : employees = 7) (h3 : teams = 6) :
  (managers + employees) / teams = 5 :=
by
  sorry

end people_per_team_l1672_167270


namespace equation_is_linear_in_one_variable_l1672_167248

theorem equation_is_linear_in_one_variable (n : ℤ) :
  (∀ x : ℝ, (n - 2) * x ^ |n - 1| + 5 = 0 → False) → n = 0 := by
  sorry

end equation_is_linear_in_one_variable_l1672_167248


namespace find_k_l1672_167216

theorem find_k (k : ℝ) : (1 - 1.5 * k = (k - 2.5) / 3) → k = 1 :=
by
  intro h
  sorry

end find_k_l1672_167216


namespace grasshopper_jump_l1672_167242

theorem grasshopper_jump (frog_jump grasshopper_jump : ℕ)
  (h1 : frog_jump = grasshopper_jump + 17)
  (h2 : frog_jump = 53) :
  grasshopper_jump = 36 :=
by
  sorry

end grasshopper_jump_l1672_167242


namespace geometric_sequence_sum_l1672_167252

noncomputable def sum_of_first_n_terms (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_a1 : a_1 = 1) (h_a5 : a_5 = 16) :
  sum_of_first_n_terms 1 q 7 = 127 :=
by
  sorry

end geometric_sequence_sum_l1672_167252


namespace least_positive_integer_n_l1672_167259

theorem least_positive_integer_n (n : ℕ) (h : (n > 0)) :
  (∃ m : ℕ, m > 0 ∧ (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 8) ∧ (∀ k : ℕ, k > 0 ∧ (1 / (k : ℝ) - 1 / (k + 1 : ℝ) < 1 / 8) → m ≤ k)) →
  n = 3 := by
  sorry

end least_positive_integer_n_l1672_167259


namespace negation_of_proposition_l1672_167208

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x > 0) ↔ (∀ x : ℝ, x^2 + 2*x ≤ 0) :=
sorry

end negation_of_proposition_l1672_167208


namespace calculation_proof_l1672_167240

theorem calculation_proof : (96 / 6) * 3 / 2 = 24 := by
  sorry

end calculation_proof_l1672_167240


namespace lcm_of_10_and_21_l1672_167266

theorem lcm_of_10_and_21 : Nat.lcm 10 21 = 210 :=
by
  sorry

end lcm_of_10_and_21_l1672_167266


namespace square_area_from_diagonal_l1672_167218

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : (d / Real.sqrt 2) ^ 2 = 64 :=
by
  sorry

end square_area_from_diagonal_l1672_167218


namespace ageOfX_l1672_167209

def threeYearsAgo (x y : ℕ) := x - 3 = 2 * (y - 3)
def sevenYearsHence (x y : ℕ) := (x + 7) + (y + 7) = 83

theorem ageOfX (x y : ℕ) (h1 : threeYearsAgo x y) (h2 : sevenYearsHence x y) : x = 45 := by
  sorry

end ageOfX_l1672_167209


namespace Ian_hourly_wage_l1672_167281

variable (hours_worked : ℕ)
variable (money_left : ℕ)
variable (hourly_wage : ℕ)

theorem Ian_hourly_wage :
  hours_worked = 8 ∧
  money_left = 72 ∧
  hourly_wage = 18 →
  2 * money_left = hours_worked * hourly_wage :=
by
  intros
  sorry

end Ian_hourly_wage_l1672_167281


namespace count_valid_sequences_returning_rectangle_l1672_167264

/-- The transformations that can be applied to the rectangle -/
inductive Transformation
| rot90   : Transformation
| rot180  : Transformation
| rot270  : Transformation
| reflYeqX  : Transformation
| reflYeqNegX : Transformation

/-- Apply a transformation to a point (x, y) -/
def apply_transformation (t : Transformation) (p : ℝ × ℝ) : ℝ × ℝ :=
match t with
| Transformation.rot90   => (-p.2,  p.1)
| Transformation.rot180  => (-p.1, -p.2)
| Transformation.rot270  => ( p.2, -p.1)
| Transformation.reflYeqX  => ( p.2,  p.1)
| Transformation.reflYeqNegX => (-p.2, -p.1)

/-- Apply a sequence of transformations to a list of points -/
def apply_sequence (seq : List Transformation) (points : List (ℝ × ℝ)) : List (ℝ × ℝ) :=
  seq.foldl (λ acc t => acc.map (apply_transformation t)) points

/-- Prove that there are exactly 12 valid sequences of three transformations that return the rectangle to its original position -/
theorem count_valid_sequences_returning_rectangle :
  let rectangle := [(0,0), (6,0), (6,2), (0,2)];
  let transformations := [Transformation.rot90, Transformation.rot180, Transformation.rot270, Transformation.reflYeqX, Transformation.reflYeqNegX];
  let seq_transformations := List.replicate 3 transformations;
  (seq_transformations.filter (λ seq => apply_sequence seq rectangle = rectangle)).length = 12 :=
sorry

end count_valid_sequences_returning_rectangle_l1672_167264


namespace largest_stamps_per_page_l1672_167254

theorem largest_stamps_per_page (h1 : Nat := 1050) (h2 : Nat := 1260) (h3 : Nat := 1470) :
  Nat.gcd h1 (Nat.gcd h2 h3) = 210 :=
by
  sorry

end largest_stamps_per_page_l1672_167254


namespace volume_decreases_by_sixteen_point_sixty_seven_percent_l1672_167291

variable {P V k : ℝ}

-- Stating the conditions
def inverse_proportionality (P V k : ℝ) : Prop :=
  P * V = k

def increased_pressure (P : ℝ) : ℝ :=
  1.2 * P

-- Theorem statement to prove the volume decrease percentage
theorem volume_decreases_by_sixteen_point_sixty_seven_percent (P V k : ℝ)
  (h1 : inverse_proportionality P V k)
  (h2 : P' = increased_pressure P) :
  V' = V / 1.2 ∧ (100 * (V - V') / V) = 16.67 :=
by
  sorry

end volume_decreases_by_sixteen_point_sixty_seven_percent_l1672_167291
