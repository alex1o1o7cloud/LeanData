import Mathlib

namespace no_prime_pairs_sum_53_l291_291390

open nat

theorem no_prime_pairs_sum_53 : 
  ¬∃ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 :=
by sorry

end no_prime_pairs_sum_53_l291_291390


namespace rank_siblings_l291_291185

variable (Person : Type) (Dan Elena Finn : Person)

variable (height : Person → ℝ)

-- Conditions
axiom different_heights : height Dan ≠ height Elena ∧ height Elena ≠ height Finn ∧ height Finn ≠ height Dan
axiom one_true_statement : (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn)) 
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))

theorem rank_siblings : height Finn > height Elena ∧ height Elena > height Dan := by
  sorry

end rank_siblings_l291_291185


namespace emma_final_balance_correct_l291_291342

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end emma_final_balance_correct_l291_291342


namespace find_d_squared_plus_e_squared_l291_291887

theorem find_d_squared_plus_e_squared {a b c d e : ℕ} 
  (h1 : (a + 1) * (3 * b * c + 1) = d + 3 * e + 1)
  (h2 : (b + 1) * (3 * c * a + 1) = 3 * d + e + 13)
  (h3 : (c + 1) * (3 * a * b + 1) = 4 * (26 - d - e) - 1)
  : d ^ 2 + e ^ 2 = 146 := 
sorry

end find_d_squared_plus_e_squared_l291_291887


namespace find_n_correct_l291_291642

noncomputable def find_n : Prop :=
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * (Real.pi / 180)) = Real.cos (317 * (Real.pi / 180)) → n = 43

theorem find_n_correct : find_n :=
  sorry

end find_n_correct_l291_291642


namespace min_trees_for_three_types_l291_291850

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l291_291850


namespace Annie_total_cookies_l291_291966

theorem Annie_total_cookies :
  let monday_cookies := 5
  let tuesday_cookies := 2 * monday_cookies 
  let wednesday_cookies := 1.4 * tuesday_cookies
  monday_cookies + tuesday_cookies + wednesday_cookies = 29 :=
by
  sorry

end Annie_total_cookies_l291_291966


namespace product_x_z_l291_291398

-- Defining the variables x, y, z as positive integers and the given conditions.
theorem product_x_z (x y z : ℕ) (h1 : x = 4 * y) (h2 : z = 2 * x) (h3 : x + y + z = 3 * y ^ 2) : 
    x * z = 5408 / 9 := 
  sorry

end product_x_z_l291_291398


namespace three_types_in_69_trees_l291_291846

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l291_291846


namespace extra_discount_percentage_l291_291958

theorem extra_discount_percentage 
  (initial_price : ℝ)
  (first_discount : ℝ)
  (new_price : ℝ)
  (final_price : ℝ)
  (extra_discount_amount : ℝ)
  (x : ℝ)
  (discount_formula : x = (extra_discount_amount * 100) / new_price) :
  initial_price = 50 ∧ 
  first_discount = 2.08 ∧ 
  new_price = 47.92 ∧ 
  final_price = 46 ∧ 
  extra_discount_amount = new_price - final_price → 
  x = 4 :=
by
  -- The proof will go here
  sorry

end extra_discount_percentage_l291_291958


namespace distinct_solution_count_l291_291664

theorem distinct_solution_count : ∀ (x : ℝ), (|x - 10| = |x + 4|) → x = 3 :=
by
  sorry

end distinct_solution_count_l291_291664


namespace min_value_x_plus_2y_l291_291691

variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

theorem min_value_x_plus_2y (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 := 
  sorry

end min_value_x_plus_2y_l291_291691


namespace max_ages_acceptable_within_one_std_dev_l291_291456

theorem max_ages_acceptable_within_one_std_dev
  (average_age : ℤ)
  (std_deviation : ℤ)
  (acceptable_range_lower : ℤ)
  (acceptable_range_upper : ℤ)
  (h1 : average_age = 31)
  (h2 : std_deviation = 5)
  (h3 : acceptable_range_lower = average_age - std_deviation)
  (h4 : acceptable_range_upper = average_age + std_deviation) :
  ∃ n : ℕ, n = acceptable_range_upper - acceptable_range_lower + 1 ∧ n = 11 :=
by
  sorry

end max_ages_acceptable_within_one_std_dev_l291_291456


namespace journey_length_l291_291261

theorem journey_length (speed time : ℝ) (portions_covered total_portions : ℕ)
  (h_speed : speed = 40) (h_time : time = 0.7) (h_portions_covered : portions_covered = 4) (h_total_portions : total_portions = 5) :
  (speed * time / portions_covered) * total_portions = 35 :=
by
  sorry

end journey_length_l291_291261


namespace correct_formula_l291_291892

-- Given conditions
def table : List (ℕ × ℕ) := [(2, 0), (3, 2), (4, 6), (5, 12), (6, 20)]

-- Candidate formulas
def formulaA (x : ℕ) : ℕ := 2 * x - 4
def formulaB (x : ℕ) : ℕ := x^2 - 3 * x + 2
def formulaC (x : ℕ) : ℕ := x^3 - 3 * x^2 + 2 * x
def formulaD (x : ℕ) : ℕ := x^2 - 4 * x
def formulaE (x : ℕ) : ℕ := x^2 - 4

-- The statement to be proven
theorem correct_formula : ∀ (x y : ℕ), (x, y) ∈ table → y = formulaB x :=
by
  sorry

end correct_formula_l291_291892


namespace ratio_of_integers_l291_291449

theorem ratio_of_integers (a b : ℤ) (h : 1996 * a + b / 96 = a + b) : a / b = 1 / 2016 ∨ b / a = 2016 :=
by
  sorry

end ratio_of_integers_l291_291449


namespace calculate_expression_l291_291013

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 :=
  by
  sorry

end calculate_expression_l291_291013


namespace min_value_is_neg_one_l291_291648

noncomputable def find_min_value (a b c : ℝ) (h : 4 * a^2 - 2 * a * b + b^2 - c = 0) (h1 : 0 < c) (h2 : |2 * a + b| = sorry) : ℝ :=
  1 / a + 2 / b + 4 / c

theorem min_value_is_neg_one (a b c : ℝ) (h : 4 * a^2 - 2 * a * b + b^2 - c = 0) (h1 : 0 < c) (h2 : |2 * a + b| = sorry) : 
  find_min_value a b c h h1 h2 = -1 :=
sorry

end min_value_is_neg_one_l291_291648


namespace long_furred_and_brown_dogs_l291_291766

-- Define the total number of dogs.
def total_dogs : ℕ := 45

-- Define the number of long-furred dogs.
def long_furred_dogs : ℕ := 26

-- Define the number of brown dogs.
def brown_dogs : ℕ := 22

-- Define the number of dogs that are neither long-furred nor brown.
def neither_long_furred_nor_brown_dogs : ℕ := 8

-- Prove that the number of dogs that are both long-furred and brown is 11.
theorem long_furred_and_brown_dogs : 
  (long_furred_dogs + brown_dogs) - (total_dogs - neither_long_furred_nor_brown_dogs) = 11 :=
by
  sorry

end long_furred_and_brown_dogs_l291_291766


namespace percentage_loss_l291_291781

variable (CP SP : ℝ) (Loss : ℝ := CP - SP) (Percentage_of_Loss : ℝ := (Loss / CP) * 100)

theorem percentage_loss (h1: CP = 1600) (h2: SP = 1440) : Percentage_of_Loss = 10 := by
  sorry

end percentage_loss_l291_291781


namespace socks_thrown_away_l291_291564

theorem socks_thrown_away 
  (initial_socks new_socks current_socks : ℕ) 
  (h1 : initial_socks = 11) 
  (h2 : new_socks = 26) 
  (h3 : current_socks = 33) : 
  initial_socks + new_socks - current_socks = 4 :=
by {
  sorry
}

end socks_thrown_away_l291_291564


namespace pizza_toppings_l291_291960

theorem pizza_toppings :
  ∀ (F V T : ℕ), F = 4 → V = 16 → F * (1 + T) = V → T = 3 :=
by
  intros F V T hF hV h
  sorry

end pizza_toppings_l291_291960


namespace recurrence_relation_solution_l291_291087

theorem recurrence_relation_solution (a : ℕ → ℕ) 
  (h_rec : ∀ n ≥ 2, a n = 4 * a (n - 1) - 3 * a (n - 2))
  (h0 : a 0 = 3)
  (h1 : a 1 = 5) :
  ∀ n, a n = 3^n + 2 :=
by
  sorry

end recurrence_relation_solution_l291_291087


namespace unique_solution_otimes_l291_291630

def otimes (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem unique_solution_otimes : 
  (∃! y : ℝ, otimes 2 y = 20) := 
by
  sorry

end unique_solution_otimes_l291_291630


namespace basketball_team_avg_weight_l291_291220

theorem basketball_team_avg_weight :
  let n_tallest := 5
  let w_tallest := 90
  let n_shortest := 4
  let w_shortest := 75
  let n_remaining := 3
  let w_remaining := 80
  let total_weight := (n_tallest * w_tallest) + (n_shortest * w_shortest) + (n_remaining * w_remaining)
  let total_players := n_tallest + n_shortest + n_remaining
  (total_weight / total_players) = 82.5 :=
by
  sorry

end basketball_team_avg_weight_l291_291220


namespace prob_3_tails_in_8_flips_l291_291498

def unfair_coin_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

def probability_of_3_tails : ℚ :=
  unfair_coin_probability 8 3 (2/3)

theorem prob_3_tails_in_8_flips :
  probability_of_3_tails = 448 / 6561 :=
by
  sorry

end prob_3_tails_in_8_flips_l291_291498


namespace download_time_l291_291973

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end download_time_l291_291973


namespace rate_is_correct_l291_291007

noncomputable def rate_of_interest (P A T : ℝ) : ℝ :=
  let SI := A - P
  (SI * 100) / (P * T)

theorem rate_is_correct :
  rate_of_interest 10000 18500 8 = 10.625 := 
by
  sorry

end rate_is_correct_l291_291007


namespace jovana_added_shells_l291_291699

theorem jovana_added_shells (initial_amount final_amount added_amount : ℕ) 
  (h1 : initial_amount = 5) 
  (h2 : final_amount = 17) 
  (h3 : added_amount = final_amount - initial_amount) : 
  added_amount = 12 := 
by 
  -- Since the proof is not required, we add sorry here to skip the proof.
  sorry 

end jovana_added_shells_l291_291699


namespace grove_tree_selection_l291_291859

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l291_291859


namespace sum_of_first_11_terms_is_minus_66_l291_291526

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d 

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a n + a 1)) / 2

theorem sum_of_first_11_terms_is_minus_66 
  (a : ℕ → ℤ) 
  (h_seq : arithmetic_sequence a)
  (h_roots : ∃ a2 a10, (a2 = a 2 ∧ a10 = a 10) ∧ (a2 + a10 = -12) ∧ (a2 * a10 = -8)) 
  : sum_of_first_n_terms a 11 = -66 :=
by
  sorry

end sum_of_first_11_terms_is_minus_66_l291_291526


namespace bulbs_in_bathroom_and_kitchen_l291_291568

theorem bulbs_in_bathroom_and_kitchen
  (bedroom_bulbs : Nat)
  (basement_bulbs : Nat)
  (garage_bulbs : Nat)
  (bulbs_per_pack : Nat)
  (packs_needed : Nat)
  (total_bulbs : Nat)
  (H1 : bedroom_bulbs = 2)
  (H2 : basement_bulbs = 4)
  (H3 : garage_bulbs = basement_bulbs / 2)
  (H4 : bulbs_per_pack = 2)
  (H5 : packs_needed = 6)
  (H6 : total_bulbs = packs_needed * bulbs_per_pack) :
  (total_bulbs - (bedroom_bulbs + basement_bulbs + garage_bulbs) = 4) :=
by
  sorry

end bulbs_in_bathroom_and_kitchen_l291_291568


namespace money_made_is_40_l291_291322

-- Definitions based on conditions
def BettysStrawberries : ℕ := 16
def MatthewsStrawberries : ℕ := BettysStrawberries + 20
def NataliesStrawberries : ℕ := MatthewsStrawberries / 2
def TotalStrawberries : ℕ := BettysStrawberries + MatthewsStrawberries + NataliesStrawberries
def JarsOfJam : ℕ := TotalStrawberries / 7
def MoneyMade : ℕ := JarsOfJam * 4

-- The theorem to prove
theorem money_made_is_40 : MoneyMade = 40 :=
by
  sorry

end money_made_is_40_l291_291322


namespace determine_b_l291_291487

theorem determine_b (b : ℝ) : (∀ x : ℝ, (-x^2 + b * x + 1 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by sorry

end determine_b_l291_291487


namespace distance_between_foci_hyperbola_l291_291813

theorem distance_between_foci_hyperbola :
  (let a^2 := 50 in
   let b^2 := 8 in
   let c^2 := a^2 + b^2 in
   2 * Real.sqrt c^2 = 2 * Real.sqrt 58) :=
by
  sorry

end distance_between_foci_hyperbola_l291_291813


namespace general_term_formula_sum_first_n_terms_l291_291364

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, a n = 3^(n-2) := 
by
  sorry

theorem sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, S n = (3^(n-2)) / 2 - 1 / 18 := 
by
  sorry

end general_term_formula_sum_first_n_terms_l291_291364


namespace find_f_l291_291507

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (h : ∀ x, x ≠ -1 → f ((1-x) / (1+x)) = (1 - x^2) / (1 + x^2)) 
               (hx : x ≠ -1) :
  f x = 2 * x / (1 + x^2) :=
sorry

end find_f_l291_291507


namespace find_actual_weights_l291_291589

noncomputable def melon_weight : ℝ := 4.5
noncomputable def watermelon_weight : ℝ := 3.5
noncomputable def scale_error : ℝ := 0.5

def weight_bounds (actual_weight measured_weight error_margin : ℝ) :=
  (measured_weight - error_margin ≤ actual_weight) ∧ (actual_weight ≤ measured_weight + error_margin)

theorem find_actual_weights (x y : ℝ) 
  (melon_measured : x = 4)
  (watermelon_measured : y = 3)
  (combined_measured : x + y = 8.5)
  (hx : weight_bounds melon_weight x scale_error)
  (hy : weight_bounds watermelon_weight y scale_error)
  (h_combined : weight_bounds (melon_weight + watermelon_weight) (x + y) (2 * scale_error)) :
  x = melon_weight ∧ y = watermelon_weight := 
sorry

end find_actual_weights_l291_291589


namespace fraction_of_number_l291_291133

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l291_291133


namespace twelfth_equation_l291_291563

theorem twelfth_equation : (14 : ℤ)^2 - (12 : ℤ)^2 = 4 * 13 := by
  sorry

end twelfth_equation_l291_291563


namespace marble_probability_l291_291161

theorem marble_probability
  (total_marbles : ℕ)
  (blue_marbles : ℕ)
  (green_marbles : ℕ)
  (draws : ℕ)
  (prob_first_green : ℚ)
  (prob_second_blue_given_green : ℚ)
  (total_prob : ℚ)
  (h_total : total_marbles = 10)
  (h_blue : blue_marbles = 4)
  (h_green : green_marbles = 6)
  (h_draws : draws = 2)
  (h_prob_first_green : prob_first_green = 3 / 5)
  (h_prob_second_blue_given_green : prob_second_blue_given_green = 4 / 9)
  (h_total_prob : total_prob = 4 / 15) :
  prob_first_green * prob_second_blue_given_green = total_prob := sorry

end marble_probability_l291_291161


namespace sum_of_roots_l291_291102

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l291_291102


namespace james_total_money_l291_291538

theorem james_total_money (bills : ℕ) (value_per_bill : ℕ) (initial_money : ℕ) : 
  bills = 3 → value_per_bill = 20 → initial_money = 75 → initial_money + (bills * value_per_bill) = 135 :=
by
  intros hb hv hi
  rw [hb, hv, hi]
  -- Algebraic simplification
  sorry

end james_total_money_l291_291538


namespace range_of_m_l291_291380

def f (m x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

def f_derivative_nonnegative_on_interval (m : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x → 6 * x^2 - 6 * m * x + 6 ≥ 0

theorem range_of_m (m : ℝ) : f_derivative_nonnegative_on_interval m ↔ m ≤ 2 :=
by
  sorry

end range_of_m_l291_291380


namespace sum_of_two_positive_cubes_lt_1000_l291_291668

open Nat

theorem sum_of_two_positive_cubes_lt_1000 :
  {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000}.card = 35 := by 
  sorry

end sum_of_two_positive_cubes_lt_1000_l291_291668


namespace find_divisor_l291_291240

theorem find_divisor
  (D dividend quotient remainder : ℤ)
  (h_dividend : dividend = 13787)
  (h_quotient : quotient = 89)
  (h_remainder : remainder = 14)
  (h_relation : dividend = (D * quotient) + remainder) :
  D = 155 :=
by
  sorry

end find_divisor_l291_291240


namespace num_cats_l291_291937

-- Definitions based on conditions
variables (C S K Cap : ℕ)
variable (heads : ℕ) (legs : ℕ)

-- Conditions as equations
axiom heads_eq : C + S + K + Cap = 16
axiom legs_eq : 4 * C + 2 * S + 2 * K + 1 * Cap = 41

-- Given values from the problem
axiom K_val : K = 1
axiom Cap_val : Cap = 1

-- The proof goal in terms of satisfying the number of cats
theorem num_cats : C = 5 :=
by
  sorry

end num_cats_l291_291937


namespace determine_n_l291_291649

theorem determine_n (x a : ℝ) (n : ℕ)
  (h1 : (n.choose 3) * x^(n-3) * a^3 = 120)
  (h2 : (n.choose 4) * x^(n-4) * a^4 = 360)
  (h3 : (n.choose 5) * x^(n-5) * a^5 = 720) :
  n = 12 :=
sorry

end determine_n_l291_291649


namespace clock_minutes_to_correct_time_l291_291763

def slow_clock_time_ratio : ℚ := 14 / 15

noncomputable def slow_clock_to_correct_time (slow_clock_time : ℚ) : ℚ :=
  slow_clock_time / slow_clock_time_ratio

theorem clock_minutes_to_correct_time :
  slow_clock_to_correct_time 14 = 15 :=
by
  sorry

end clock_minutes_to_correct_time_l291_291763


namespace minimum_tangent_length_l291_291683

theorem minimum_tangent_length
  (a b : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 3 = 0)
  (h_symmetry : ∀ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 → 2 * a * x + b * y + 6 = 0) :
  ∃ t : ℝ, t = 2 :=
by sorry

end minimum_tangent_length_l291_291683


namespace square_side_length_l291_291173

/-- 
If a square is drawn by joining the midpoints of the sides of a given square and repeating this process continues indefinitely,
and the sum of the areas of all the squares is 32 cm²,
then the length of the side of the first square is 4 cm. 
-/
theorem square_side_length (s : ℝ) (h : ∑' n : ℕ, (s^2) * (1 / 2)^n = 32) : s = 4 := 
by 
  sorry

end square_side_length_l291_291173


namespace line_intersects_circle_probability_l291_291246

def probability_line_intersects_circle : Real :=
  let interval := Icc (-1 : ℝ) (1 : ℝ)
  let intersect_condition (k : ℝ) : Prop :=
    abs (3 * k) / sqrt (k^2 + 1) < 1
  let probability_density := 1 / (interval.snd - interval.fst)
  probability_density * ∫ x in interval.fst..interval.snd, 
    if intersect_condition x then probability_density else 0

theorem line_intersects_circle_probability :
  probability_line_intersects_circle = Real.sqrt 2 / 4 :=
by
  sorry

end line_intersects_circle_probability_l291_291246


namespace min_value_of_quadratic_l291_291749

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l291_291749


namespace airplane_faster_by_90_minutes_l291_291177

def driving_time : ℕ := 3 * 60 + 15  -- in minutes
def drive_to_airport_time : ℕ := 10   -- in minutes
def board_wait_time : ℕ := 20         -- in minutes
def flight_time : ℕ := driving_time / 3  -- in minutes
def get_off_airplane_time : ℕ := 10   -- in minutes

theorem airplane_faster_by_90_minutes :
  driving_time - (drive_to_airport_time + board_wait_time + flight_time + get_off_airplane_time) = 90 :=
by
  calc
    driving_time 
      = 195               : by unfold driving_time
    ...(10 + 20 + 65 + 10 = 105): by unfold drive_to_airport_time board_wait_time flight_time get_off_airplane_time
    195 - 105 = 90         : by norm_num

end airplane_faster_by_90_minutes_l291_291177


namespace fraction_of_number_l291_291136

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l291_291136


namespace matrix_solution_property_l291_291803

theorem matrix_solution_property (N : Matrix (Fin 2) (Fin 2) ℝ) 
    (h : N = Matrix.of ![![2, 4], ![1, 4]]) :
    N ^ 4 - 5 * N ^ 3 + 9 * N ^ 2 - 5 * N = Matrix.of ![![6, 12], ![3, 6]] :=
by 
  sorry

end matrix_solution_property_l291_291803


namespace expand_and_simplify_product_l291_291633

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := (2 * x^2 - 3 * x + 4) * (2 * x^2 + 3 * x + 4)

-- Define the simplified form
def simplified_expr (x : ℝ) : ℝ := 4 * x^4 + 7 * x^2 + 16

theorem expand_and_simplify_product (x : ℝ) : initial_expr x = simplified_expr x := by
  -- We would provide the proof steps here
  sorry

end expand_and_simplify_product_l291_291633


namespace range_eq_domain_l291_291512

def f (x : ℝ) : ℝ := |x - 2| - 2

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem range_eq_domain : (Set.range f) = M :=
by
  sorry

end range_eq_domain_l291_291512


namespace evaluate_expression_l291_291985

theorem evaluate_expression : 2 + (3 / (4 + (5 / 6))) = 76 / 29 := 
by
  sorry

end evaluate_expression_l291_291985


namespace factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l291_291724

-- Problem 1
theorem factorize_polynomial (x y : ℝ) : 
  x^2 - y^2 + 2*x - 2*y = (x - y)*(x + y + 2) := 
sorry

-- Problem 2
theorem triangle_equilateral (a b c : ℝ) (h : a^2 + c^2 - 2*b*(a - b + c) = 0) : 
  a = b ∧ b = c :=
sorry

-- Problem 3
theorem prove_2p_eq_m_plus_n (m n p : ℝ) (h : 1/4*(m - n)^2 = (p - n)*(m - p)) : 
  2*p = m + n :=
sorry

end factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l291_291724


namespace fraction_addition_l291_291360

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4)

theorem fraction_addition (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / b = 3 / 4) : (a + b) / b = 7 / 4 :=
  sorry

end fraction_addition_l291_291360


namespace ratio_of_areas_of_circles_l291_291211

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l291_291211


namespace simplify_exponent_l291_291297

theorem simplify_exponent :
  2000 * 2000^2000 = 2000^2001 :=
by
  sorry

end simplify_exponent_l291_291297


namespace min_value_of_f_l291_291752

noncomputable def f : ℝ → ℝ := λ x, x^2 + 16 * x + 20

theorem min_value_of_f : ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -44 :=
by
  have h : ∀ x, f x = (x + 8)^2 - 44 :=
    by intros x; calc
      f x = x^2 + 16 * x + 20         : rfl
      ...  = (x^2 + 16 * x + 64) - 44 : by ring
      ...  = (x + 8)^2 - 44           : by ring
  use [-8]
  intros x
  constructor
  · calc f x = (x + 8)^2 - 44   : by rw h x
           ...  ≥ 0 - 44        : sub_le_sub_right (sq_nonneg (x + 8)) 44
  · calc f (-8) = ((-8) + 8)^2 - 44 : by rw h (-8)
              ...  = 0 - 44         : by ring
              ...  = -44            : rfl

end min_value_of_f_l291_291752


namespace smallest_x_value_l291_291351

theorem smallest_x_value (x : ℝ) (h : |4 * x + 9| = 37) : x = -11.5 :=
sorry

end smallest_x_value_l291_291351


namespace smallest_root_of_quadratic_l291_291757

theorem smallest_root_of_quadratic :
  ∃ x : ℝ, (12 * x^2 - 50 * x + 48 = 0) ∧ x = 1.333 := 
sorry

end smallest_root_of_quadratic_l291_291757


namespace min_value_of_f_l291_291753

noncomputable def f : ℝ → ℝ := λ x, x^2 + 16 * x + 20

theorem min_value_of_f : ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -44 :=
by
  have h : ∀ x, f x = (x + 8)^2 - 44 :=
    by intros x; calc
      f x = x^2 + 16 * x + 20         : rfl
      ...  = (x^2 + 16 * x + 64) - 44 : by ring
      ...  = (x + 8)^2 - 44           : by ring
  use [-8]
  intros x
  constructor
  · calc f x = (x + 8)^2 - 44   : by rw h x
           ...  ≥ 0 - 44        : sub_le_sub_right (sq_nonneg (x + 8)) 44
  · calc f (-8) = ((-8) + 8)^2 - 44 : by rw h (-8)
              ...  = 0 - 44         : by ring
              ...  = -44            : rfl

end min_value_of_f_l291_291753


namespace probability_of_same_suit_l291_291739

-- Definitions for the conditions
def total_cards : ℕ := 52
def suits : ℕ := 4
def cards_per_suit : ℕ := 13
def total_draws : ℕ := 2

-- Definition of factorial for binomial coefficient calculation
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Binomial coefficient calculation
def binomial_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Calculation of the probability
def prob_same_suit : ℚ :=
  let ways_to_choose_2_cards_from_52 := binomial_coeff total_cards total_draws
  let ways_to_choose_2_cards_per_suit := binomial_coeff cards_per_suit total_draws
  let total_ways_to_choose_2_same_suit := suits * ways_to_choose_2_cards_per_suit
  total_ways_to_choose_2_same_suit / ways_to_choose_2_cards_from_52

theorem probability_of_same_suit :
  prob_same_suit = 4 / 17 :=
by
  sorry

end probability_of_same_suit_l291_291739


namespace red_peaches_each_basket_l291_291588

variable (TotalGreenPeachesInABasket : Nat) (TotalPeachesInABasket : Nat)

theorem red_peaches_each_basket (h1 : TotalPeachesInABasket = 10) (h2 : TotalGreenPeachesInABasket = 3) :
  (TotalPeachesInABasket - TotalGreenPeachesInABasket) = 7 := by
  sorry

end red_peaches_each_basket_l291_291588


namespace problem1_problem2_problem3_general_conjecture_l291_291652

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

-- Prove f(0) + f(1) = sqrt(2) / 2
theorem problem1 : f 0 + f 1 = Real.sqrt 2 / 2 := by
  sorry

-- Prove f(-1) + f(2) = sqrt(2) / 2
theorem problem2 : f (-1) + f 2 = Real.sqrt 2 / 2 := by
  sorry

-- Prove f(-2) + f(3) = sqrt(2) / 2
theorem problem3 : f (-2) + f 3 = Real.sqrt 2 / 2 := by
  sorry

-- Prove ∀ x, f(-x) + f(x+1) = sqrt(2) / 2
theorem general_conjecture (x : ℝ) : f (-x) + f (x + 1) = Real.sqrt 2 / 2 := by
  sorry

end problem1_problem2_problem3_general_conjecture_l291_291652


namespace derivative_of_ln_2x_l291_291509

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x)

theorem derivative_of_ln_2x (x : ℝ) : deriv f x = 1 / x :=
  sorry

end derivative_of_ln_2x_l291_291509


namespace diminished_value_l291_291166

theorem diminished_value (x y : ℝ) (h1 : x = 160)
  (h2 : x / 5 + 4 = x / 4 - y) : y = 4 :=
by
  sorry

end diminished_value_l291_291166


namespace cortney_downloads_all_files_in_2_hours_l291_291980

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end cortney_downloads_all_files_in_2_hours_l291_291980


namespace upper_bound_expression_l291_291818

theorem upper_bound_expression (n : ℤ) (U : ℤ) :
  (∀ n, 4 * n + 7 > 1 ∧ 4 * n + 7 < U → ∃ k : ℤ, k = 50) →
  U = 204 :=
by
  sorry

end upper_bound_expression_l291_291818


namespace strictly_increasing_sequences_exists_l291_291900

theorem strictly_increasing_sequences_exists (n : ℕ) (a : ℕ → ℕ) 
  (h : ∀ k : ℕ, count (fun m => a m = k) (list.range (2 * n)) ≤ n) : 
  ∃ (b c : ℕ → ℕ), (∀ i, 1 ≤ i ∧ i ≤ n → b i < c i) ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ n → a (b i) ≠ a (c i)) ∧ ∀ i : ℕ, i < 2 * n → (∃ j, b j = i ∨ c j = i) := 
sorry

end strictly_increasing_sequences_exists_l291_291900


namespace pirate_total_dollar_amount_l291_291938

def base_5_to_base_10 (digits : List ℕ) (base : ℕ) : ℕ :=
  digits.reverse.enum.map (λ ⟨p, d⟩ => d * base^p) |>.sum

def jewelry_base5 := [3, 1, 2, 4]
def gold_coins_base5 := [3, 1, 2, 2]
def alcohol_base5 := [1, 2, 4]

def jewelry_base10 := base_5_to_base_10 jewelry_base5 5
def gold_coins_base10 := base_5_to_base_10 gold_coins_base5 5
def alcohol_base10 := base_5_to_base_10 alcohol_base5 5

def total_base10 := jewelry_base10 + gold_coins_base10 + alcohol_base10

theorem pirate_total_dollar_amount :
  total_base10 = 865 :=
by
  unfold total_base10 jewelry_base10 gold_coins_base10 alcohol_base10 base_5_to_base_10
  simp
  sorry

end pirate_total_dollar_amount_l291_291938


namespace largest_c_for_sum_squares_ineq_l291_291029

noncomputable theory
open_locale big_operators

-- Define median of a list of real numbers
def median {n : ℕ} (x : fin n → ℝ) : ℝ :=
if n % 2 = 1 then x (fin.of_nat' ((n / 2 : ℕ) + 1)) else (x (fin.of_nat' (n / 2)) + x (fin.of_nat' (n / 2 + 1))) / 2

theorem largest_c_for_sum_squares_ineq :
  ∃ c : ℝ,
    (∀ (x : fin 101 → ℝ),
      (∑ i, x i) = 0 →
      let M := median x in ∑ i, (x i)^2 ≥ c * M^2) ∧
    c = 5151 / 50 :=
by {
  use 5151 / 50,
  intros x sum_zero M,
  sorry
}

end largest_c_for_sum_squares_ineq_l291_291029


namespace smallest_lcm_4_digit_integers_l291_291677

theorem smallest_lcm_4_digit_integers (k l : ℕ) (h1 : 1000 ≤ k ∧ k ≤ 9999) (h2 : 1000 ≤ l ∧ l ≤ 9999) (h3 : Nat.gcd k l = 11) : Nat.lcm k l = 92092 :=
by
  sorry

end smallest_lcm_4_digit_integers_l291_291677


namespace num_integer_pairs_l291_291032

theorem num_integer_pairs (m n : ℤ) :
  0 < m ∧ m < n ∧ n < 53 ∧ 53^2 + m^2 = 52^2 + n^2 →
  ∃ k, k = 3 := 
sorry

end num_integer_pairs_l291_291032


namespace cone_volume_divided_by_pi_l291_291302

theorem cone_volume_divided_by_pi : 
  let r := 15
  let l := 20
  let h := 5 * Real.sqrt 7
  let V := (1/3:ℝ) * Real.pi * r^2 * h
  (V / Real.pi = 1125 * Real.sqrt 7) := sorry

end cone_volume_divided_by_pi_l291_291302


namespace boxes_remaining_to_sell_l291_291229

-- Define the conditions
def first_customer_boxes : ℕ := 5 
def second_customer_boxes : ℕ := 4 * first_customer_boxes
def third_customer_boxes : ℕ := second_customer_boxes / 2
def fourth_customer_boxes : ℕ := 3 * third_customer_boxes
def final_customer_boxes : ℕ := 10
def sales_goal : ℕ := 150

-- Total boxes sold
def total_boxes_sold : ℕ := first_customer_boxes + second_customer_boxes + third_customer_boxes + fourth_customer_boxes + final_customer_boxes

-- Boxes left to sell to hit the sales goal
def boxes_left_to_sell : ℕ := sales_goal - total_boxes_sold

-- Prove the number of boxes left to sell is 75
theorem boxes_remaining_to_sell : boxes_left_to_sell = 75 :=
by
  -- Step to prove goes here
  sorry

end boxes_remaining_to_sell_l291_291229


namespace rachel_left_24_brownies_at_home_l291_291722

-- Defining the conditions
def total_brownies : ℕ := 40
def brownies_brought_to_school : ℕ := 16

-- Formulation of the theorem
theorem rachel_left_24_brownies_at_home : (total_brownies - brownies_brought_to_school = 24) :=
by
  sorry

end rachel_left_24_brownies_at_home_l291_291722


namespace find_k_l291_291197

theorem find_k (k : ℕ) (h1 : k > 0) (h2 : 15 * k^4 < 120) : k = 1 := 
  sorry

end find_k_l291_291197


namespace son_work_time_l291_291923

theorem son_work_time :
  let M := (1 : ℚ) / 7
  let combined_rate := (1 : ℚ) / 3
  let S := combined_rate - M
  1 / S = 5.25 :=  
by
  sorry

end son_work_time_l291_291923


namespace james_total_money_l291_291534

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end james_total_money_l291_291534


namespace min_value_of_quadratic_l291_291750

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l291_291750


namespace trigonometric_identity_l291_291405

open Real

theorem trigonometric_identity (α : ℝ) : 
  sin α * sin α + cos (π / 6 + α) * cos (π / 6 + α) + sin α * cos (π / 6 + α) = 3 / 4 :=
sorry

end trigonometric_identity_l291_291405


namespace find_a_decreasing_l291_291661

-- Define the given function
def f (a x : ℝ) : ℝ := (x - 1) ^ 2 + 2 * a * x + 1

-- State the condition
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y ≤ f x

-- State the proposition
theorem find_a_decreasing :
  ∀ a : ℝ, is_decreasing_on (f a) (Set.Iio 4) → a ≤ -3 :=
by
  intro a
  intro h
  sorry

end find_a_decreasing_l291_291661


namespace principal_invested_years_l291_291993

-- Define the given conditions
def principal : ℕ := 9200
def rate : ℕ := 12
def interest_deficit : ℤ := 5888

-- Define the time to be proved
def time_invested : ℤ := 3

-- Define the simple interest formula
def simple_interest (P R t : ℕ) : ℕ :=
  (P * R * t) / 100

-- Define the problem statement
theorem principal_invested_years :
  ∃ t : ℕ, principal - interest_deficit = simple_interest principal rate t ∧ t = time_invested := 
by
  sorry

end principal_invested_years_l291_291993


namespace divisibility_by_six_l291_291582

theorem divisibility_by_six (n : ℤ) : 6 ∣ (n^3 - n) := 
sorry

end divisibility_by_six_l291_291582


namespace find_x_l291_291606

theorem find_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 :=
sorry

end find_x_l291_291606


namespace find_breadth_of_plot_l291_291150

-- Define the conditions
def length_of_plot (breadth : ℝ) := 3 * breadth
def area_of_plot := 2028

-- Define what we want to prove
theorem find_breadth_of_plot (breadth : ℝ) (h1 : length_of_plot breadth * breadth = area_of_plot) : breadth = 26 :=
sorry

end find_breadth_of_plot_l291_291150


namespace exists_acute_triangle_l291_291596

-- Given five segment lengths and it being possible to form a triangle with any three of them
variables {a1 a2 a3 a4 a5 : ℝ}

-- The condition that any three of these segments can form a triangle
def triangle_inequality (x y z : ℝ) : Prop := (x + y > z) ∧ (x + z > y) ∧ (y + z > x)

axiom segments_can_form_triangle :
  triangle_inequality a1 a2 a3 ∧
  triangle_inequality a1 a2 a4 ∧
  triangle_inequality a1 a2 a5 ∧
  triangle_inequality a1 a3 a4 ∧
  triangle_inequality a1 a3 a5 ∧
  triangle_inequality a1 a4 a5 ∧
  triangle_inequality a2 a3 a4 ∧
  triangle_inequality a2 a3 a5 ∧
  triangle_inequality a2 a4 a5 ∧
  triangle_inequality a3 a4 a5

-- Define what it means to have all angles acute in a triangle
def triangle_all_acute (x y z : ℝ) : Prop :=
  (x^2 + y^2 > z^2) ∧ (x^2 + z^2 > y^2) ∧ (y^2 + z^2 > x^2)

-- The theorem stating that at least one of the triangles has all angles acute
theorem exists_acute_triangle :
  ∃ (i j k : ℝ), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ triangle_inequality i j k ∧ triangle_all_acute i j k :=
by
  sorry

end exists_acute_triangle_l291_291596


namespace product_of_three_numbers_l291_291586

-- Define the problem conditions as variables and assumptions
variables (a b c : ℚ)
axiom h1 : a + b + c = 30
axiom h2 : a = 3 * (b + c)
axiom h3 : b = 6 * c

-- State the theorem to be proven
theorem product_of_three_numbers : a * b * c = 10125 / 14 :=
by
  sorry

end product_of_three_numbers_l291_291586


namespace decimal_to_fraction_l291_291442

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l291_291442


namespace min_value_of_z_l291_291747

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l291_291747


namespace division_by_3_l291_291140

theorem division_by_3 (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := 
sorry

end division_by_3_l291_291140


namespace cortney_downloads_all_files_in_2_hours_l291_291979

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end cortney_downloads_all_files_in_2_hours_l291_291979


namespace resistance_of_one_rod_l291_291590

section RodResistance

variables (R_0 R : ℝ)

-- Given: the resistance of the entire construction is 8 Ω
def entire_construction_resistance : Prop := R = 8

-- Given: formula for the equivalent resistance
def equivalent_resistance_formula : Prop := R = 4 / 10 * R_0

-- To prove: the resistance of one rod is 20 Ω
theorem resistance_of_one_rod 
  (h1 : entire_construction_resistance R)
  (h2 : equivalent_resistance_formula R_0 R) :
  R_0 = 20 :=
sorry

end RodResistance

end resistance_of_one_rod_l291_291590


namespace midpoint_polar_coords_l291_291225

/-- 
Given two points in polar coordinates: (6, π/6) and (2, -π/6),  
the midpoint of the line segment connecting these points in polar coordinates is (√13, π/6).
-/
theorem midpoint_polar_coords :
  let A := (6, Real.pi / 6)
  let B := (2, -Real.pi / 6)
  let A_cart := (6 * Real.cos (Real.pi / 6), 6 * Real.sin (Real.pi / 6))
  let B_cart := (2 * Real.cos (-Real.pi / 6), 2 * Real.sin (-Real.pi / 6))
  let Mx := ((A_cart.fst + B_cart.fst) / 2)
  let My := ((A_cart.snd + B_cart.snd) / 2)
  let r := Real.sqrt (Mx^2 + My^2)
  let theta := Real.arctan (My / Mx)
  0 <= theta ∧ theta < 2 * Real.pi ∧ r > 0 ∧ (r = Real.sqrt 13 ∧ theta = Real.pi / 6) :=
by 
  sorry

end midpoint_polar_coords_l291_291225


namespace min_value_of_z_l291_291748

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l291_291748


namespace sum_of_roots_l291_291104

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l291_291104


namespace find_worst_competitor_l291_291471

structure Competitor :=
  (name : String)
  (gender : String)
  (generation : String)

-- Define the competitors
def man : Competitor := ⟨"man", "male", "generation1"⟩
def wife : Competitor := ⟨"wife", "female", "generation1"⟩
def son : Competitor := ⟨"son", "male", "generation2"⟩
def sister : Competitor := ⟨"sister", "female", "generation1"⟩

-- Conditions
def opposite_genders (c1 c2 : Competitor) : Prop :=
  c1.gender ≠ c2.gender

def different_generations (c1 c2 : Competitor) : Prop :=
  c1.generation ≠ c2.generation

noncomputable def worst_competitor : Competitor :=
  sister

def is_sibling (c1 c2 : Competitor) : Prop :=
  (c1 = man ∧ c2 = sister) ∨ (c1 = sister ∧ c2 = man)

-- Theorem statement
theorem find_worst_competitor (best_competitor : Competitor) :
  (opposite_genders worst_competitor best_competitor) ∧
  (different_generations worst_competitor best_competitor) ∧
  ∃ (sibling : Competitor), (is_sibling worst_competitor sibling) :=
  sorry

end find_worst_competitor_l291_291471


namespace sum_of_areas_of_circles_l291_291422

noncomputable def radius (n : ℕ) : ℝ :=
  3 / 3^n

noncomputable def area (n : ℕ) : ℝ :=
  Real.pi * (radius n)^2

noncomputable def total_area : ℝ :=
  ∑' n, area n

theorem sum_of_areas_of_circles:
  total_area = (9 * Real.pi) / 8 :=
by
  sorry

end sum_of_areas_of_circles_l291_291422


namespace part1_part2_l291_291501

variables {a m n : ℝ}

theorem part1 (h1 : a^m = 2) (h2 : a^n = 3) : a^(4*m + 3*n) = 432 :=
by sorry

theorem part2 (h1 : a^m = 2) (h2 : a^n = 3) : a^(5*m - 2*n) = 32 / 9 :=
by sorry

end part1_part2_l291_291501


namespace count_even_sum_subsets_l291_291049

open Finset

noncomputable def givenSet : Finset ℕ := {31, 47, 58, 62, 89, 132, 164}

def isEven (n : ℕ) : Prop := n % 2 = 0

theorem count_even_sum_subsets :
  (givenSet.powerset.filter (λ s, s.card = 3 ∧ isEven (s.sum id))).card = 16 := sorry

end count_even_sum_subsets_l291_291049


namespace constant_seq_arith_geo_l291_291474

def is_arithmetic_sequence (s : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n + d

def is_geometric_sequence (s : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n * r

theorem constant_seq_arith_geo (s : ℕ → ℝ) (d r : ℝ) :
  is_arithmetic_sequence s d →
  is_geometric_sequence s r →
  (∃ c : ℝ, ∀ n : ℕ, s n = c) ∧ r = 1 :=
by
  sorry

end constant_seq_arith_geo_l291_291474


namespace rectangle_area_l291_291473

/-- Define a rectangle with its length being three times its breadth, and given diagonal length d = 20.
    Prove that the area of the rectangle is 120 square meters. -/
theorem rectangle_area (b : ℝ) (l : ℝ) (d : ℝ) (h1 : l = 3 * b) (h2 : d = 20) (h3 : l^2 + b^2 = d^2) : l * b = 120 :=
by
  sorry

end rectangle_area_l291_291473


namespace find_range_of_m_l291_291195

variable (x m : ℝ)

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + (4 * m - 3) > 0

def proposition_q (m : ℝ) : Prop := (∀ m > 2, m + 1 / (m - 2) ≥ 4) ∧ (∃ m, m + 1 / (m - 2) = 4)

def range_m : Set ℝ := {m | 1 < m ∧ m ≤ 2} ∪ {m | m ≥ 3}

theorem find_range_of_m
  (h_p : proposition_p m ∨ ¬proposition_p m)
  (h_q : proposition_q m ∨ ¬proposition_q m)
  (h_exclusive : (proposition_p m ∧ ¬proposition_q m) ∨ (¬proposition_p m ∧ proposition_q m))
  : m ∈ range_m := sorry

end find_range_of_m_l291_291195


namespace digit_one_not_in_mean_l291_291331

def seq : List ℕ := [5, 55, 555, 5555, 55555, 555555, 5555555, 55555555, 555555555]

noncomputable def arithmetic_mean (l : List ℕ) : ℕ := l.sum / l.length

theorem digit_one_not_in_mean :
  ¬(∃ d, d ∈ (arithmetic_mean seq).digits 10 ∧ d = 1) :=
sorry

end digit_one_not_in_mean_l291_291331


namespace restaurant_discount_l291_291692

theorem restaurant_discount :
  let coffee_price := 6
  let cheesecake_price := 10
  let discount_rate := 0.25
  let total_price := coffee_price + cheesecake_price
  let discount := discount_rate * total_price
  let final_price := total_price - discount
  final_price = 12 := by
  sorry

end restaurant_discount_l291_291692


namespace operation_on_original_number_l291_291785

theorem operation_on_original_number (f : ℕ → ℕ) (x : ℕ) (h : 3 * (f x + 9) = 51) (hx : x = 4) :
  f x = 2 * x :=
by
  sorry

end operation_on_original_number_l291_291785


namespace lines_from_abs_eq_l291_291184

theorem lines_from_abs_eq (x y : ℝ) : 
  (|x| - |y| = 1) ↔ 
  ((x ≥ 1 ∧ y = x - 1) ∨ (x ≥ 1 ∧ y = 1 - x) ∨
  (x ≤ -1 ∧ y = -x - 1) ∨ (x ≤ -1 ∧ y = x + 1)) :=
begin
  sorry
end

end lines_from_abs_eq_l291_291184


namespace largest_of_four_consecutive_odd_numbers_l291_291149

theorem largest_of_four_consecutive_odd_numbers (x : ℤ) : 
  (x % 2 = 1) → 
  ((x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) →
  (x + 6 = 27) :=
by
  sorry

end largest_of_four_consecutive_odd_numbers_l291_291149


namespace min_value_one_div_a_plus_one_div_b_l291_291037

theorem min_value_one_div_a_plus_one_div_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end min_value_one_div_a_plus_one_div_b_l291_291037


namespace fraction_of_number_l291_291129

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l291_291129


namespace min_segments_of_polyline_l291_291619

theorem min_segments_of_polyline (n : ℕ) (h : n ≥ 2) : 
  ∃ s : ℕ, s = 2 * n - 2 := sorry

end min_segments_of_polyline_l291_291619


namespace min_value_of_f_l291_291754

noncomputable def f : ℝ → ℝ := λ x, x^2 + 16 * x + 20

theorem min_value_of_f : ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -44 :=
by
  have h : ∀ x, f x = (x + 8)^2 - 44 :=
    by intros x; calc
      f x = x^2 + 16 * x + 20         : rfl
      ...  = (x^2 + 16 * x + 64) - 44 : by ring
      ...  = (x + 8)^2 - 44           : by ring
  use [-8]
  intros x
  constructor
  · calc f x = (x + 8)^2 - 44   : by rw h x
           ...  ≥ 0 - 44        : sub_le_sub_right (sq_nonneg (x + 8)) 44
  · calc f (-8) = ((-8) + 8)^2 - 44 : by rw h (-8)
              ...  = 0 - 44         : by ring
              ...  = -44            : rfl

end min_value_of_f_l291_291754


namespace sequence_initial_term_l291_291385

theorem sequence_initial_term (a : ℕ) :
  let a_1 := a
  let a_2 := 2
  let a_3 := a_1 + a_2
  let a_4 := a_1 + a_2 + a_3
  let a_5 := a_1 + a_2 + a_3 + a_4
  let a_6 := a_1 + a_2 + a_3 + a_4 + a_5
  a_6 = 56 → a = 5 :=
by
  intros h
  sorry

end sequence_initial_term_l291_291385


namespace unique_solution_l291_291334

theorem unique_solution (a n : ℕ) (h₁ : 0 < a) (h₂ : 0 < n) (h₃ : 3^n = a^2 - 16) : a = 5 ∧ n = 2 :=
by
sorry

end unique_solution_l291_291334


namespace number_of_cubes_with_three_faces_painted_l291_291268

-- Definitions of conditions
def large_cube_side_length : ℕ := 4
def total_smaller_cubes := large_cube_side_length ^ 3

-- Prove the number of smaller cubes with at least 3 faces painted is 8
theorem number_of_cubes_with_three_faces_painted :
  (∃ (n : ℕ), n = 8) :=
by
  -- Conditions recall
  have side_length := large_cube_side_length
  have total_cubes := total_smaller_cubes
  
  -- Recall that the cube is composed by smaller cubes with painted faces.
  have painted_faces_condition : (∀ (cube : ℕ), cube = 8) := sorry
  
  exact ⟨8, painted_faces_condition 8⟩

end number_of_cubes_with_three_faces_painted_l291_291268


namespace calculate_triple_hash_l291_291186

def hash (N : ℝ) : ℝ := 0.5 * N - 2

theorem calculate_triple_hash : hash (hash (hash 100)) = 9 := by
  sorry

end calculate_triple_hash_l291_291186


namespace smallest_trees_in_three_types_l291_291861

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l291_291861


namespace vanya_four_times_faster_l291_291715

-- We let d be the total distance, and define the respective speeds
variables (d : ℝ) (v_m v_v : ℝ)

-- Conditions from the problem
-- 1. Vanya starts after Masha
axiom start_after_masha : ∀ t : ℝ, t > 0

-- 2. Vanya overtakes Masha at one-third of the distance
axiom vanya_overtakes_masha : ∀ t : ℝ, (v_v * t) = d / 3

-- 3. When Vanya reaches the school, Masha still has half of the way to go
axiom masha_halfway : ∀ t : ℝ, (v_m * t) = d / 2

-- Goal to prove
theorem vanya_four_times_faster : v_v = 4 * v_m :=
sorry

end vanya_four_times_faster_l291_291715


namespace mean_of_squares_eq_l291_291325

noncomputable def sum_of_squares (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def arithmetic_mean_of_squares (n : ℕ) : ℚ := sum_of_squares n / n

theorem mean_of_squares_eq (n : ℕ) (h : n ≠ 0) : arithmetic_mean_of_squares n = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end mean_of_squares_eq_l291_291325


namespace quadratic_square_binomial_l291_291840

theorem quadratic_square_binomial (a : ℝ) :
  (∃ d : ℝ, 9 * x ^ 2 - 18 * x + a = (3 * x + d) ^ 2) → a = 9 :=
by
  intro h
  match h with
  | ⟨d, h_eq⟩ => sorry

end quadratic_square_binomial_l291_291840


namespace smallest_k_for_inequality_l291_291030

theorem smallest_k_for_inequality : 
  ∃ k : ℕ,  k > 0 ∧ ( (k-10) ^ 5026 ≥ 2013 ^ 2013 ) ∧ 
  (∀ m : ℕ, m > 0 ∧ ((m-10) ^ 5026) ≥ 2013 ^ 2013 → m ≥ 55) :=
sorry

end smallest_k_for_inequality_l291_291030


namespace closest_multiple_of_12_l291_291759

def is_multiple_of (n m : ℕ) : Prop := ∃ k, n = m * k

-- Define the closest multiple of 4 to 2050 (2048 and 2052)
def closest_multiple_of_4 (n m : ℕ) : ℕ :=
if n % 4 < m % 4 then n - (n % 4)
else m + (4 - (m % 4))

-- Define the conditions for being divisible by both 3 and 4
def is_multiple_of_12 (n : ℕ) : Prop := is_multiple_of n 12

-- Theorem statement
theorem closest_multiple_of_12 (n m : ℕ) (h : n = 2050) (hm : m = 2052) :
  is_multiple_of_12 m :=
sorry

end closest_multiple_of_12_l291_291759


namespace number_of_hockey_players_l291_291222

theorem number_of_hockey_players 
  (cricket_players : ℕ) 
  (football_players : ℕ) 
  (softball_players : ℕ) 
  (total_players : ℕ) 
  (hockey_players : ℕ) 
  (h1 : cricket_players = 10) 
  (h2 : football_players = 16) 
  (h3 : softball_players = 13) 
  (h4 : total_players = 51) 
  (calculation : hockey_players = total_players - (cricket_players + football_players + softball_players)) : 
  hockey_players = 12 :=
by 
  rw [h1, h2, h3, h4] at calculation
  exact calculation

end number_of_hockey_players_l291_291222


namespace tax_rate_equals_65_l291_291775

def tax_rate_percentage := 65
def tax_rate_per_dollars (rate_percentage : ℕ) : ℕ :=
  (rate_percentage / 100) * 100

theorem tax_rate_equals_65 :
  tax_rate_per_dollars tax_rate_percentage = 65 := by
  sorry

end tax_rate_equals_65_l291_291775


namespace polar_to_cartesian_eq_polar_circle_area_l291_291898

theorem polar_to_cartesian_eq (p θ x y : ℝ) (h : p = 2 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 - 2 * x + y^2 = 0 := sorry

theorem polar_circle_area (p θ : ℝ) (h : p = 2 * Real.cos θ) :
  Real.pi = Real.pi := (by ring)


end polar_to_cartesian_eq_polar_circle_area_l291_291898


namespace solve_quadratic_l291_291603

theorem solve_quadratic (x : ℝ) (h : x^2 - 2 * x - 3 = 0) : x = 3 ∨ x = -1 := 
sorry

end solve_quadratic_l291_291603


namespace fraction_of_number_l291_291114

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l291_291114


namespace digit_D_eq_9_l291_291732

-- Define digits and the basic operations on 2-digit numbers
def is_digit (n : ℕ) : Prop := n < 10
def tens (n : ℕ) : ℕ := n / 10
def units (n : ℕ) : ℕ := n % 10
def two_digit (a b : ℕ) : ℕ := 10 * a + b

theorem digit_D_eq_9 (A B C D : ℕ):
  is_digit A → is_digit B → is_digit C → is_digit D →
  (two_digit A B) + (two_digit C B) = two_digit D A →
  (two_digit A B) - (two_digit C B) = A →
  D = 9 :=
by sorry

end digit_D_eq_9_l291_291732


namespace matinee_receipts_l291_291491

theorem matinee_receipts :
  let child_ticket_cost := 4.50
  let adult_ticket_cost := 6.75
  let num_children := 48
  let num_adults := num_children - 20
  total_receipts = num_children * child_ticket_cost + num_adults * adult_ticket_cost :=
by 
  sorry

end matinee_receipts_l291_291491


namespace min_trees_include_three_types_l291_291865

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l291_291865


namespace pratt_certificate_space_bound_l291_291085

-- Define the Pratt certificate space function λ(p)
noncomputable def pratt_space (p : ℕ) : ℝ := sorry

-- Define the log_2 function (if not already available in Mathlib)
noncomputable def log2 (x : ℝ) : ℝ := sorry

-- Assuming that p is a prime number
variable {p : ℕ} (hp : Nat.Prime p)

-- The proof problem
theorem pratt_certificate_space_bound (hp : Nat.Prime p) :
  pratt_space p ≤ 6 * (log2 p) ^ 2 := 
sorry

end pratt_certificate_space_bound_l291_291085


namespace tomato_red_flesh_probability_l291_291428

theorem tomato_red_flesh_probability :
  (P_yellow_skin : ℝ) = 3 / 8 →
  (P_red_flesh_given_yellow_skin : ℝ) = 8 / 15 →
  (P_yellow_skin_given_not_red_flesh : ℝ) = 7 / 30 →
  (P_red_flesh : ℝ) = 1 / 4 := 
by
  intros h1 h2 h3
  sorry

end tomato_red_flesh_probability_l291_291428


namespace horner_eval_at_neg2_l291_291199

noncomputable def f (x : ℝ) : ℝ := x^5 - 3 * x^3 - 6 * x^2 + x - 1

theorem horner_eval_at_neg2 : f (-2) = -35 :=
by
  sorry

end horner_eval_at_neg2_l291_291199


namespace choir_average_age_l291_291728

theorem choir_average_age
  (avg_females_age : ℕ)
  (num_females : ℕ)
  (avg_males_age : ℕ)
  (num_males : ℕ)
  (females_avg_condition : avg_females_age = 28)
  (females_num_condition : num_females = 8)
  (males_avg_condition : avg_males_age = 32)
  (males_num_condition : num_males = 17) :
  ((avg_females_age * num_females + avg_males_age * num_males) / (num_females + num_males) = 768 / 25) :=
by
  sorry

end choir_average_age_l291_291728


namespace number_of_elements_l291_291253

def average_incorrect (N : ℕ) := 21
def correction (incorrect : ℕ) (correct : ℕ) := correct - incorrect
def average_correct (N : ℕ) := 22

theorem number_of_elements (N : ℕ) (incorrect : ℕ) (correct : ℕ) :
  average_incorrect N = 21 ∧ incorrect = 26 ∧ correct = 36 ∧ average_correct N = 22 →
  N = 10 :=
by
  sorry

end number_of_elements_l291_291253


namespace fraction_of_number_l291_291117

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l291_291117


namespace overlap_length_l291_291423

noncomputable def length_of_all_red_segments := 98 -- in cm
noncomputable def total_length := 83 -- in cm
noncomputable def number_of_overlaps := 6 -- count

theorem overlap_length :
  ∃ (x : ℝ), length_of_all_red_segments - total_length = number_of_overlaps * x ∧ x = 2.5 := by
  sorry

end overlap_length_l291_291423


namespace linda_change_l291_291709

-- Defining the conditions
def cost_per_banana : ℝ := 0.30
def number_of_bananas : ℕ := 5
def amount_paid : ℝ := 10.00

-- Proving the statement
theorem linda_change :
  amount_paid - (number_of_bananas * cost_per_banana) = 8.50 :=
by
  sorry

end linda_change_l291_291709


namespace total_money_l291_291539

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end total_money_l291_291539


namespace min_value_5_5_l291_291876

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (6 * z) / (2 * x + y) + (6 * x) / (y + 2 * z) + (4 * y) / (x + z)

theorem min_value_5_5 (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z = 1) :
  given_expression x y z ≥ 5.5 :=
sorry

end min_value_5_5_l291_291876


namespace tickets_per_candy_l291_291919

theorem tickets_per_candy (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) (candies_bought : ℕ)
    (h1 : tickets_whack_a_mole = 26) (h2 : tickets_skee_ball = 19) (h3 : candies_bought = 5) :
    (tickets_whack_a_mole + tickets_skee_ball) / candies_bought = 9 := by
  sorry

end tickets_per_candy_l291_291919


namespace max_adjacent_distinct_pairs_l291_291905

theorem max_adjacent_distinct_pairs (n : ℕ) (h : n = 100) : 
  ∃ (a : ℕ), a = 50 := 
by 
  -- Here we use the provided constraints and game scenario to state the theorem formally.
  sorry

end max_adjacent_distinct_pairs_l291_291905


namespace extremum_value_and_min_on_interval_l291_291044

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem extremum_value_and_min_on_interval
  (a b c : ℝ)
  (h1_eq : 12 * a + b = 0)
  (h2_eq : 4 * a + b = -8)
  (h_max : 16 + c = 28) :
  min (min (f a b c (-3)) (f a b c 3)) (f a b c 2) = -4 :=
by sorry

end extremum_value_and_min_on_interval_l291_291044


namespace probability_red_blue_l291_291108

-- Declare the conditions (probabilities for white, green and yellow marbles).
variables (total_marbles : ℕ) (P_white P_green P_yellow P_red_blue : ℚ)
-- implicitly P_white, P_green, P_yellow, P_red_blue are probabilities, therefore between 0 and 1

-- Assume the conditions given in the problem
axiom total_marbles_condition : total_marbles = 250
axiom P_white_condition : P_white = 2 / 5
axiom P_green_condition : P_green = 1 / 4
axiom P_yellow_condition : P_yellow = 1 / 10

-- Proving the required probability of red or blue marbles
theorem probability_red_blue :
  P_red_blue = 1 - (P_white + P_green + P_yellow) :=
sorry

end probability_red_blue_l291_291108


namespace Elaine_rent_increase_l291_291870

noncomputable def Elaine_rent_percent (E: ℝ) : ℝ :=
  let last_year_rent := 0.20 * E
  let this_year_earnings := 1.25 * E
  let this_year_rent := 0.30 * this_year_earnings
  let ratio := (this_year_rent / last_year_rent) * 100
  ratio

theorem Elaine_rent_increase (E : ℝ) : Elaine_rent_percent E = 187.5 :=
by 
  -- The proof would go here.
  sorry

end Elaine_rent_increase_l291_291870


namespace center_of_circle_sum_l291_291353

open Real

theorem center_of_circle_sum (x y : ℝ) (h k : ℝ) :
  (x - h)^2 + (y - k)^2 = 2 → (h = 3) → (k = 4) → h + k = 7 :=
by
  intro h_eq k_eq
  sorry

end center_of_circle_sum_l291_291353


namespace midpoint_ellipse_trajectory_l291_291035

theorem midpoint_ellipse_trajectory (x y x0 y0 x1 y1 x2 y2 : ℝ) :
  (x0 / 12) + (y0 / 8) = 1 →
  (x1^2 / 24) + (y1^2 / 16) = 1 →
  (x2^2 / 24) + (y2^2 / 16) = 1 →
  x = (x1 + x2) / 2 →
  y = (y1 + y2) / 2 →
  ∃ x y, ((x - 1)^2 / (5 / 2)) + ((y - 1)^2 / (5 / 3)) = 1 :=
by
  sorry

end midpoint_ellipse_trajectory_l291_291035


namespace margin_expression_l291_291689

variable (C S M : ℝ)
variable (n : ℕ)

theorem margin_expression (h : M = (C + S) / n) : M = (2 * S) / (n + 1) :=
sorry

end margin_expression_l291_291689


namespace smallest_m_n_sum_l291_291575

noncomputable def f (m n : ℕ) (x : ℝ) : ℝ := Real.arcsin (Real.log (n * x) / Real.log m)

theorem smallest_m_n_sum 
  (m n : ℕ) 
  (h_m1 : 1 < m) 
  (h_mn_closure : ∀ x, -1 ≤ Real.log (n * x) / Real.log m ∧ Real.log (n * x) / Real.log m ≤ 1) 
  (h_length : (m ^ 2 - 1) / (m * n) = 1 / 2021) : 
  m + n = 86259 := by
sorry

end smallest_m_n_sum_l291_291575


namespace smallest_n_intersection_nonempty_l291_291072

open Finset

theorem smallest_n_intersection_nonempty :
  ∀ (X : Finset ℕ) (subsets : Finset (Finset X)),
    (∀ A ∈ subsets, A.card ≤ 56) →
    subsets.card = 15 →
    (∀ S : Finset (Finset X), S ⊆ subsets → S.card = 7 → (S.sup id).card ≥ 41) →
    ∃ (S1 S2 S3 : Finset X), S1 ∈ subsets ∧ S2 ∈ subsets ∧ S3 ∈ subsets ∧ (S1 ∩ S2 ∩ S3).nonempty := 
sorry

end smallest_n_intersection_nonempty_l291_291072


namespace sandbox_volume_l291_291788

def length : ℕ := 312
def width : ℕ := 146
def depth : ℕ := 75
def volume (l w d : ℕ) : ℕ := l * w * d

theorem sandbox_volume : volume length width depth = 3429000 := by
  sorry

end sandbox_volume_l291_291788


namespace PersonYs_speed_in_still_water_l291_291279

def speed_in_still_water (speed_X : ℕ) (t_1 t_2 : ℕ) (x : ℕ) : Prop :=
  ∀ y : ℤ, 4 * (6 - y + x + y) = 4 * 6 + 4 * x ∧ 16 * (x + y) = 16 * (6 + y) + 4 * (x - 6) →
  x = 10

theorem PersonYs_speed_in_still_water :
  speed_in_still_water 6 4 16 10 :=
by
  sorry

end PersonYs_speed_in_still_water_l291_291279


namespace first_candidate_votes_percentage_l291_291061

theorem first_candidate_votes_percentage 
( total_votes : ℕ ) 
( second_candidate_votes : ℕ ) 
( P : ℕ ) 
( h1 : total_votes = 2400 ) 
( h2 : second_candidate_votes = 480 ) 
( h3 : (P/100 : ℝ) * total_votes + second_candidate_votes = total_votes ) : 
  P = 80 := 
sorry

end first_candidate_votes_percentage_l291_291061


namespace cakes_served_yesterday_l291_291171

theorem cakes_served_yesterday (cakes_today_lunch : ℕ) (cakes_today_dinner : ℕ) (total_cakes : ℕ)
  (h1 : cakes_today_lunch = 5) (h2 : cakes_today_dinner = 6) (h3 : total_cakes = 14) :
  total_cakes - (cakes_today_lunch + cakes_today_dinner) = 3 :=
by
  -- Import necessary libraries
  sorry

end cakes_served_yesterday_l291_291171


namespace minimum_trees_l291_291852

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l291_291852


namespace no_three_digits_all_prime_l291_291403

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function that forms a three-digit number from digits a, b, c
def form_three_digit (a b c : ℕ) : ℕ :=
100 * a + 10 * b + c

-- Define a function to check if all permutations of three digits form prime numbers
def all_permutations_prime (a b c : ℕ) : Prop :=
is_prime (form_three_digit a b c) ∧
is_prime (form_three_digit a c b) ∧
is_prime (form_three_digit b a c) ∧
is_prime (form_three_digit b c a) ∧
is_prime (form_three_digit c a b) ∧
is_prime (form_three_digit c b a)

-- The main theorem stating that there are no three distinct digits making all permutations prime
theorem no_three_digits_all_prime : ¬∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  all_permutations_prime a b c :=
sorry

end no_three_digits_all_prime_l291_291403


namespace matrices_equal_l291_291399

open Matrix

variables {n : ℕ}
variables {x y : Fin n → ℝ}
variables (A B : Matrix (Fin n) (Fin n) ℝ)

-- Definition of matrix A based on the given conditions
def matrixA (i j : Fin n) : ℝ :=
  if x i + y j >= 0 then 1 else 0

-- Definition of matrix B satisfying the given conditions
def matrixB (B : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  (∀ i j, B i j = 0 ∨ B i j = 1) ∧
  (∀ i, (∑ j, B i j) = (∑ j, matrixA x y i j)) ∧
  (∀ j, (∑ i, B i j) = (∑ i, matrixA x y i j))

theorem matrices_equal
  (hB : matrixB B) :
  A = B := sorry

end matrices_equal_l291_291399


namespace ratio_of_spinsters_to_cats_l291_291736

def spinsters := 22
def cats := spinsters + 55

theorem ratio_of_spinsters_to_cats : (spinsters : ℝ) / (cats : ℝ) = 2 / 7 := 
by
  sorry

end ratio_of_spinsters_to_cats_l291_291736


namespace f_value_at_5_l291_291827

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 3 / 2 then 2 * x^2 else sorry

theorem f_value_at_5 (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_periodic : periodic_function f 3)
  (h_definition : ∀ x, 0 ≤ x ∧ x ≤ 3 / 2 → f x = 2 * x^2) :
  f 5 = 2 :=
by
  sorry

end f_value_at_5_l291_291827


namespace total_is_correct_l291_291203

-- Define the given conditions.
def dividend : ℕ := 55
def divisor : ℕ := 11
def quotient := dividend / divisor
def total := dividend + quotient + divisor

-- State the theorem to be proven.
theorem total_is_correct : total = 71 := by sorry

end total_is_correct_l291_291203


namespace solve_for_a_l291_291243

theorem solve_for_a (a : ℤ) :
  (|2 * a + 1| = 3) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end solve_for_a_l291_291243


namespace time_ratio_school_home_l291_291632

open Real

noncomputable def time_ratio (y x : ℝ) : ℝ :=
  let time_school := (y / (3 * x)) + (2 * y / (2 * x)) + (y / (4 * x))
  let time_home := (y / (4 * x)) + (2 * y / (2 * x)) + (y / (3 * x))
  time_school / time_home

theorem time_ratio_school_home (y x : ℝ) (hy : y ≠ 0) (hx : x ≠ 0) : time_ratio y x = 19 / 16 :=
  sorry

end time_ratio_school_home_l291_291632


namespace min_trees_include_three_types_l291_291864

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l291_291864


namespace round_trip_time_l291_291098

def boat_speed := 9 -- speed of the boat in standing water (kmph)
def stream_speed := 6 -- speed of the stream (kmph)
def distance := 210 -- distance to the place (km)

def upstream_speed := boat_speed - stream_speed
def downstream_speed := boat_speed + stream_speed

def time_upstream := distance / upstream_speed
def time_downstream := distance / downstream_speed
def total_time := time_upstream + time_downstream

theorem round_trip_time : total_time = 84 := by
  sorry

end round_trip_time_l291_291098


namespace milk_problem_l291_291451

theorem milk_problem (x : ℕ) (hx : 0 < x)
    (total_cost_wednesday : 10 = x * (10 / x))
    (price_reduced : ∀ x, 0.5 = (10 / x - (10 / x) + 0.5))
    (extra_bags : 2 = (x + 2) - x)
    (extra_cost : 2 + 10 = x * (10 / x) + 2) :
    x^2 + 6 * x - 40 = 0 := by
  sorry

end milk_problem_l291_291451


namespace num_positive_integers_m_l291_291817

theorem num_positive_integers_m (h : ∀ m : ℕ, ∃ d : ℕ, 3087 = d ∧ m^2 = d + 3) :
  ∃! m : ℕ, 0 < m ∧ (3087 % (m^2 - 3) = 0) := by
  sorry

end num_positive_integers_m_l291_291817


namespace max_ab_bc_cd_l291_291075

theorem max_ab_bc_cd {a b c d : ℝ} (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_sum : a + b + c + d = 200) (h_a : a = 2 * d) : 
  ab + bc + cd ≤ 14166.67 :=
sorry

end max_ab_bc_cd_l291_291075


namespace a_n_formula_d_value_l291_291705

-- Given conditions for part 1
def a_seq (a : Nat → ℕ) (d : ℕ) : Prop :=
  ∀ n, a(n+1) = a(n) + d

def cond1 (a : Nat → ℕ) (d : ℕ) : Prop :=
  3 * a(1+1) = 3 * a 1 + a(1 + 2)

def cond2 (a : Nat → ℕ) (d : ℕ) : Prop :=
  (a 1 + a 2 + a 3) + ((2 + 1) / a 1 + (2 * 2 + 2) / (a 1 + d) + (3 * 3 + 3) / (a 1 + 2 * d)) = 21

-- Theorem for part 1
theorem a_n_formula (a : Nat → ℕ) (d : ℕ) (h1 : d > 1) (h2 : a_seq a d) (h3 : cond1 a d) (h4 : cond2 a d) : ∀ n, a n = 3 * n :=
sorry

-- Given conditions for part 2
def b_seq (b : Nat → ℕ) : Prop :=
  ∀ n, b(n+1) = b(n) + b(1)

def cond3 (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) : Prop :=
  (99 * a 1 + 99 * d - ((99 * 100) / 2 + 199 / (99 * d + a 1))) = 99
  
-- Theorem for part 2
theorem d_value (a : Nat → ℕ) (b : Nat → ℕ) (d : ℕ) (h1 : b_seq b) (h2 : cond3 a b d) : d = 51 / 50 :=
sorry

end a_n_formula_d_value_l291_291705


namespace line_equation_l291_291823

theorem line_equation (P : ℝ × ℝ) (hP : P = (1, 5)) (h1 : ∃ a, a ≠ 0 ∧ (P.1 + P.2 = a)) (h2 : x_intercept = y_intercept) : 
  (∃ a, a ≠ 0 ∧ P = (a, 0) ∧ P = (0, a) → x + y - 6 = 0) ∨ (5*P.1 - P.2 = 0) :=
by
  sorry

end line_equation_l291_291823


namespace find_two_digit_number_l291_291943

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l291_291943


namespace vector_addition_correct_l291_291369

def a : ℝ × ℝ := (-1, 6)
def b : ℝ × ℝ := (3, -2)
def c : ℝ × ℝ := (2, 4)

theorem vector_addition_correct : a + b = c := by
  sorry

end vector_addition_correct_l291_291369


namespace probability_of_even_product_is_13_div_18_l291_291497

open_locale classical

-- Define the set of 9 pieces of paper labeled 1 to 9
def paper_set : finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the event that the product of two numbers is even
def even_product_event (x y : ℕ) : Prop := (x * y) % 2 = 0

-- Define the probability of drawing two papers and their product being even
noncomputable def probability_even_product : ℚ :=
probs.pairwise_event_probability paper_set even_product_event

-- The goal is to prove that this probability is 13/18
theorem probability_of_even_product_is_13_div_18 :
  probability_even_product = 13 / 18 :=
sorry

end probability_of_even_product_is_13_div_18_l291_291497


namespace factorial_inequality_l291_291555

theorem factorial_inequality (a b n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < n) (h4 : n.factorial = a.factorial * b.factorial) :
  a + b < n + 2 * Real.log n / Real.log 2 + 4 :=
by
  sorry

end factorial_inequality_l291_291555


namespace remainder_is_cx_plus_d_l291_291071

-- Given a polynomial Q, assume the following conditions
variables {Q : ℕ → ℚ}

-- Conditions
axiom condition1 : Q 15 = 12
axiom condition2 : Q 10 = 4

theorem remainder_is_cx_plus_d : 
  ∃ c d, (c = 8 / 5) ∧ (d = -12) ∧ 
          ∀ x, Q x % ((x - 10) * (x - 15)) = c * x + d :=
by
  sorry

end remainder_is_cx_plus_d_l291_291071


namespace fruit_seller_original_apples_l291_291931

variable (x : ℝ)

theorem fruit_seller_original_apples (h : 0.60 * x = 420) : x = 700 := by
  sorry

end fruit_seller_original_apples_l291_291931


namespace lord_moneybag_l291_291076

theorem lord_moneybag (n : ℕ) (hlow : 300 ≤ n) (hhigh : n ≤ 500)
           (h6 : 6 ∣ n) (h5 : 5 ∣ (n - 1)) (h4 : 4 ∣ (n - 2)) 
           (h3 : 3 ∣ (n - 3)) (h2 : 2 ∣ (n - 4)) (hprime : Nat.Prime (n - 5)) :
  n = 426 := by
  sorry

end lord_moneybag_l291_291076


namespace xiaoyu_reading_days_l291_291920

theorem xiaoyu_reading_days
  (h1 : ∀ (p d : ℕ), p = 15 → d = 24 → p * d = 360)
  (h2 : ∀ (p t : ℕ), t = 360 → p = 18 → t / p = 20) :
  ∀ d : ℕ, d = 20 :=
by
  sorry

end xiaoyu_reading_days_l291_291920


namespace sum_of_two_cubes_lt_1000_l291_291673

theorem sum_of_two_cubes_lt_1000 : 
  let num_sums := finset.card {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} 
  in num_sums = 75 :=
by
  let S := finset.filter (λ n : ℕ, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000) finset.Ico 1 1000
  have : finset.card S = 75 := sorry,
  exact this

end sum_of_two_cubes_lt_1000_l291_291673


namespace parabolas_intersect_on_circle_l291_291330

theorem parabolas_intersect_on_circle :
  let parabola1 (x y : ℝ) := y = (x - 2)^2
  let parabola2 (x y : ℝ) := x + 6 = (y + 1)^2
  ∃ (cx cy r : ℝ), ∀ (x y : ℝ), (parabola1 x y ∧ parabola2 x y) → (x - cx)^2 + (y - cy)^2 = r^2 ∧ r^2 = 33/2 :=
by
  sorry

end parabolas_intersect_on_circle_l291_291330


namespace longer_leg_smallest_triangle_l291_291336

noncomputable def length_of_longer_leg_of_smallest_triangle (n : ℕ) (a : ℝ) : ℝ :=
  if n = 0 then a 
  else if n = 1 then (a / 2) * Real.sqrt 3
  else if n = 2 then ((a / 2) * Real.sqrt 3 / 2) * Real.sqrt 3
  else ((a / 2) * Real.sqrt 3 / 2 * Real.sqrt 3 / 2) * Real.sqrt 3

theorem longer_leg_smallest_triangle : 
  length_of_longer_leg_of_smallest_triangle 3 10 = 45 / 8 := 
sorry

end longer_leg_smallest_triangle_l291_291336


namespace calculate_f_of_f_of_f_30_l291_291187

-- Define the function f (equivalent to $\#N = 0.5N + 2$)
def f (N : ℝ) : ℝ := 0.5 * N + 2

-- The proof statement
theorem calculate_f_of_f_of_f_30 : 
  f (f (f 30)) = 7.25 :=
by
  sorry

end calculate_f_of_f_of_f_30_l291_291187


namespace necessary_but_not_sufficient_l291_291294

theorem necessary_but_not_sufficient (a : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + a < 0) → (a < 11) ∧ ¬((a < 11) → (∃ x : ℝ, x^2 - 2*x + a < 0)) :=
by
  -- Sorry to bypass proof below, which is correct as per the problem statement requirements.
  sorry

end necessary_but_not_sufficient_l291_291294


namespace person_B_age_l291_291081

variables (a b c d e f g : ℕ)

-- Conditions
axiom cond1 : a = b + 2
axiom cond2 : b = 2 * c
axiom cond3 : c = d / 2
axiom cond4 : d = e - 3
axiom cond5 : f = a * d
axiom cond6 : g = b + e
axiom cond7 : a + b + c + d + e + f + g = 292

-- Theorem statement
theorem person_B_age : b = 14 :=
sorry

end person_B_age_l291_291081


namespace joan_half_dollars_spent_on_wednesday_l291_291079

variable (x : ℝ)
variable (h1 : x * 0.5 + 14 * 0.5 = 9)

theorem joan_half_dollars_spent_on_wednesday :
  x = 4 :=
by
  -- The proof is not required, hence using sorry
  sorry

end joan_half_dollars_spent_on_wednesday_l291_291079


namespace eliza_tom_difference_l291_291488

theorem eliza_tom_difference (q : ℕ) : 
  let eliza_quarters := 7 * q + 3
  let tom_quarters := 2 * q + 8
  let quarter_difference := (7 * q + 3) - (2 * q + 8)
  let nickel_value := 5
  let groups_of_5 := quarter_difference / 5
  let difference_in_cents := nickel_value * groups_of_5
  difference_in_cents = 5 * (q - 1) := by
  sorry

end eliza_tom_difference_l291_291488


namespace fraction_ordering_l291_291744

theorem fraction_ordering:
  (6 / 22) < (5 / 17) ∧ (5 / 17) < (8 / 24) :=
by
  sorry

end fraction_ordering_l291_291744


namespace probability_two_most_expensive_l291_291162

open Nat

noncomputable def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem probability_two_most_expensive :
  (combination 8 1) / (combination 10 3) = 1 / 15 :=
by
  sorry

end probability_two_most_expensive_l291_291162


namespace no_integer_solution_l291_291486

theorem no_integer_solution (x : ℤ) : ¬ (x + 12 > 15 ∧ -3 * x > -9) :=
by {
  sorry
}

end no_integer_solution_l291_291486


namespace project_completion_l291_291033

theorem project_completion (a b : ℕ) (h1 : 3 * (1 / b : ℚ) + (1 / a : ℚ) + (1 / b : ℚ) = 1) : 
  a + b = 9 ∨ a + b = 10 :=
sorry

end project_completion_l291_291033


namespace base_b_digit_sum_l291_291914

theorem base_b_digit_sum :
  ∃ (b : ℕ), ((b^2 / 2 + b / 2) % b = 2) ∧ (b = 8) :=
by
  sorry

end base_b_digit_sum_l291_291914


namespace find_piles_l291_291561

theorem find_piles :
  ∃ N : ℕ, 
  (1000 < N ∧ N < 2000) ∧ 
  (N % 2 = 1) ∧ (N % 3 = 1) ∧ (N % 4 = 1) ∧ 
  (N % 5 = 1) ∧ (N % 6 = 1) ∧ (N % 7 = 1) ∧ (N % 8 = 1) ∧ 
  (∃ p : ℕ, p = 41 ∧ p > 1 ∧ p < N ∧ N % p = 0) :=
sorry

end find_piles_l291_291561


namespace least_multiple_of_36_with_product_of_digits_multiple_of_36_l291_291444

def product_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).prod

theorem least_multiple_of_36_with_product_of_digits_multiple_of_36 :
  ∀ n : ℕ, n % 36 = 0 → (∀ k : ℕ, k % 36 = 0 → product_of_digits k % 36 ≠ 0 → k ≥ 1296) ∧ product_of_digits 1296 % 36 = 0 :=
  sorry

end least_multiple_of_36_with_product_of_digits_multiple_of_36_l291_291444


namespace fraction_of_number_l291_291131

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l291_291131


namespace symmetric_line_equation_l291_291576

-- Define the given lines
def original_line (x y : ℝ) : Prop := y = 2 * x + 1
def line_of_symmetry (x y : ℝ) : Prop := y + 2 = 0

-- Define the problem statement as a theorem
theorem symmetric_line_equation :
  ∀ (x y : ℝ), line_of_symmetry x y → (original_line x (2 * (-2 - y) + 1)) ↔ (2 * x + y + 5 = 0) := 
sorry

end symmetric_line_equation_l291_291576


namespace exchange_5_rubles_l291_291328

theorem exchange_5_rubles :
  ¬ ∃ n : ℕ, 1 * n + 2 * n + 3 * n + 5 * n = 500 :=
by 
  sorry

end exchange_5_rubles_l291_291328


namespace least_positive_integer_l291_291917

theorem least_positive_integer (n : ℕ) :
  (∃ n : ℕ, 25^n + 16^n ≡ 1 [MOD 121] ∧ ∀ m : ℕ, (m < n ∧ 25^m + 16^m ≡ 1 [MOD 121]) → false) ↔ n = 32 :=
sorry

end least_positive_integer_l291_291917


namespace sin_beta_value_l291_291656

open Real

theorem sin_beta_value (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h1 : sin α = 5 / 13) 
  (h2 : cos (α + β) = -4 / 5) : 
  sin β = 56 / 65 := 
sorry

end sin_beta_value_l291_291656


namespace number_of_shoes_outside_library_l291_291305

-- Define the conditions
def number_of_people : ℕ := 10
def shoes_per_person : ℕ := 2

-- Define the proof that the number of shoes kept outside the library is 20.
theorem number_of_shoes_outside_library : number_of_people * shoes_per_person = 20 :=
by
  -- Proof left as sorry because the proof steps are not required
  sorry

end number_of_shoes_outside_library_l291_291305


namespace tickets_difference_is_cost_l291_291318

def tickets_won : ℝ := 48.5
def yoyo_cost : ℝ := 11.7
def tickets_left (w : ℝ) (c : ℝ) : ℝ := w - c
def difference (w : ℝ) (l : ℝ) : ℝ := w - l

theorem tickets_difference_is_cost :
  difference tickets_won (tickets_left tickets_won yoyo_cost) = yoyo_cost :=
by
  -- Proof will be written here
  sorry

end tickets_difference_is_cost_l291_291318


namespace calculate_product_l291_291216

theorem calculate_product : (3 * 5 * 7 = 38) → (13 * 15 * 17 = 268) → 1 * 3 * 5 = 15 :=
by
  intros h1 h2
  sorry

end calculate_product_l291_291216


namespace money_made_is_40_l291_291323

-- Definitions based on conditions
def BettysStrawberries : ℕ := 16
def MatthewsStrawberries : ℕ := BettysStrawberries + 20
def NataliesStrawberries : ℕ := MatthewsStrawberries / 2
def TotalStrawberries : ℕ := BettysStrawberries + MatthewsStrawberries + NataliesStrawberries
def JarsOfJam : ℕ := TotalStrawberries / 7
def MoneyMade : ℕ := JarsOfJam * 4

-- The theorem to prove
theorem money_made_is_40 : MoneyMade = 40 :=
by
  sorry

end money_made_is_40_l291_291323


namespace train_length_correct_l291_291470

-- Define the conditions
def bridge_length : ℝ := 180
def train_speed : ℝ := 15
def time_to_cross_bridge : ℝ := 20
def time_to_cross_man : ℝ := 8

-- Define the length of the train
def length_of_train : ℝ := 120

-- Proof statement
theorem train_length_correct :
  (train_speed * time_to_cross_man = length_of_train) ∧
  (train_speed * time_to_cross_bridge = length_of_train + bridge_length) :=
by
  sorry

end train_length_correct_l291_291470


namespace download_time_is_2_hours_l291_291976

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end download_time_is_2_hours_l291_291976


namespace product_grades_probabilities_l291_291167

theorem product_grades_probabilities (P_Q P_S : ℝ) (h1 : P_Q = 0.98) (h2 : P_S = 0.21) :
  P_Q - P_S = 0.77 ∧ 1 - P_Q = 0.02 :=
by
  sorry

end product_grades_probabilities_l291_291167


namespace find_general_formula_and_d_l291_291701

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def seq_sum (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

def S (a : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum a n
def T (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) : ℝ := seq_sum b n

def b (n a : ℕ → ℝ) (n : ℕ) : ℝ := (n^2 + n) / a n

theorem find_general_formula_and_d:
  ∃ (a : ℕ → ℝ) (d : ℝ),
    d > 1 ∧ 
    arithmetic_seq a d ∧ 
    3 * a 2 = 3 * a 1 + a 3 ∧ 
    S a 3 + T a (b a) 3 = 21 ∧ 
    S a 99 - T a (b a) 99 = 99 ∧ 
    (∀ n, b a (n+1) - b a n = b a (n+1) - b a n ) →
    (∀ n, a n = 3 * n) ∧ 
    d = 51 / 50 := 
sorry

end find_general_formula_and_d_l291_291701


namespace frog_climbing_time_l291_291467

-- Defining the conditions as Lean definitions
def well_depth : ℕ := 12
def climb_distance : ℕ := 3
def slip_distance : ℕ := 1
def climb_time : ℚ := 1 -- time in minutes for the frog to climb 3 meters
def slip_time : ℚ := climb_time / 3
def total_time_per_cycle : ℚ := climb_time + slip_time
def total_climbed_at_817 : ℕ := well_depth - 3 -- 3 meters from the top means it climbed 9 meters

-- The equivalent proof statement in Lean:
theorem frog_climbing_time : 
  ∃ (T : ℚ), T = 22 ∧ 
    (well_depth = 9 + 3) ∧
    (∀ (cycles : ℕ), cycles = 4 → 
         total_time_per_cycle * cycles + 2 = T) :=
by 
  sorry

end frog_climbing_time_l291_291467


namespace intersect_point_l291_291259

-- Definitions as per conditions
def f (x : ℝ) (b : ℝ) : ℝ := 4 * x + b
def f_inv (x : ℝ) (a : ℝ) : ℝ := a -- We define inverse as per given (4, a)

-- Variables for the conditions
variables (a b : ℤ)

-- Theorems to prove the conditions match the answers
theorem intersect_point : ∃ a b : ℤ, f 4 b = a ∧ f_inv 4 a = 4 ∧ a = 4 := by
  sorry

end intersect_point_l291_291259


namespace quadratic_has_equal_roots_l291_291493

theorem quadratic_has_equal_roots :
  ∀ m : ℝ, (∀ x : ℝ, x^2 + 6 * x + m = 0 → x = -3) ↔ m = 9 := 
by
  intro m
  constructor
  {
    intro h
    have : (6:ℝ) ^ 2 - 4 * 1 * m = 0,
      from by simp [(pow_two 6), h.eq_c],
    simp [six_pow_two, neg_eq_zero] at this,
    linarith
  }
  {
    intro h
    simp [h],
    exact fun x _ => rfl
  }

end quadratic_has_equal_roots_l291_291493


namespace chosen_number_l291_291289

theorem chosen_number (x : ℤ) (h : x / 12 - 240 = 8) : x = 2976 :=
by sorry

end chosen_number_l291_291289


namespace part1_part2_l291_291807

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1| + |2 * x + 3|

theorem part1 (x : ℝ) : f x ≥ 6 ↔ x ≥ 1 ∨ x ≤ -2 := by
  sorry

theorem part2 (a b : ℝ) (m : ℝ) 
  (a_pos : a > 0) (b_pos : b > 0) 
  (fmin : m = 4) 
  (condition : 2 * a * b + a + 2 * b = m) : 
  a + 2 * b = 2 * Real.sqrt 5 - 2 := by
  sorry

end part1_part2_l291_291807


namespace compute_diameter_of_garden_roller_l291_291729

noncomputable def diameter_of_garden_roller (length : ℝ) (area_per_revolution : ℝ) (pi : ℝ) :=
  let radius := (area_per_revolution / (2 * pi * length))
  2 * radius

theorem compute_diameter_of_garden_roller :
  diameter_of_garden_roller 3 (66 / 5) (22 / 7) = 1.4 := by
  sorry

end compute_diameter_of_garden_roller_l291_291729


namespace cookies_per_person_l291_291483

variable (x y z : ℕ)
variable (h_pos_z : z ≠ 0) -- Ensure z is not zero to avoid division by zero

theorem cookies_per_person (h_cookies : x * y / z = 35) : 35 / 5 = 7 := by
  sorry

end cookies_per_person_l291_291483


namespace length_of_bridge_is_correct_l291_291600

def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * crossing_time_s
  total_distance - train_length

theorem length_of_bridge_is_correct : 
  length_of_bridge 170 45 30 = 205 :=
by
  -- we state the translation and prove here (proof omitted, just the structure is present)
  sorry

end length_of_bridge_is_correct_l291_291600


namespace measure_angle_PSR_is_40_l291_291790

noncomputable def isosceles_triangle (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : Triangle := sorry
noncomputable def square (D R S T : Point) : Square := sorry
noncomputable def angle (A B C : Point) (θ : ℝ) : Prop := sorry

def angle_PQR (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry
def angle_PRQ (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry

theorem measure_angle_PSR_is_40
  (P Q R S T D : Point)
  (PQ PR : ℝ)
  (hPQ_PR : PQ = PR)
  (hQ_eq_D : Q = D)
  (hQPS : angle P Q S 100)
  (hDRST_square : square D R S T) : angle P S R 40 :=
by
  -- Proof omitted for brevity
  sorry

end measure_angle_PSR_is_40_l291_291790


namespace seating_arrangements_l291_291910

def valid_seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  if total_seats = 8 ∧ people = 3 then 12 else 0

theorem seating_arrangements (total_seats people : ℕ) (h1 : total_seats = 8) (h2 : people = 3) :
  valid_seating_arrangements total_seats people = 12 :=
by
  rw [valid_seating_arrangements, h1, h2]
  simp
  done

end seating_arrangements_l291_291910


namespace necessary_but_not_sufficient_condition_l291_291651

-- Define the set A
def A := {x : ℝ | -1 < x ∧ x < 2}

-- Define the necessary but not sufficient condition
def necessary_condition (a : ℝ) : Prop := a ≥ 1

-- Define the proposition that needs to be proved
def proposition (a : ℝ) : Prop := ∀ x ∈ A, x^2 - a < 0

-- The proof statement
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  necessary_condition a → ∃ x ∈ A, proposition a :=
sorry

end necessary_but_not_sufficient_condition_l291_291651


namespace divisors_larger_than_9_factorial_l291_291516

theorem divisors_larger_than_9_factorial (n : ℕ) :
  (∃ k : ℕ, k = 9 ∧ (number_of_divisors_of_10_factorial_greater_than_9_factorial = k)) :=
begin
  sorry
end

def number_of_divisors_of_10_factorial_greater_than_9_factorial : ℕ :=
  (10.fact.divisors.filter (λ d, d > 9.fact)).length

end divisors_larger_than_9_factorial_l291_291516


namespace circle_area_ratio_l291_291215

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l291_291215


namespace fraction_of_number_l291_291134

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l291_291134


namespace change_factor_l291_291093

theorem change_factor (n : ℕ) (avg_original avg_new : ℕ) (F : ℝ)
  (h1 : n = 10) (h2 : avg_original = 80) (h3 : avg_new = 160) 
  (h4 : F * (n * avg_original) = n * avg_new) :
  F = 2 :=
by
  sorry

end change_factor_l291_291093


namespace minimum_value_of_abs_z_l291_291235

noncomputable def min_abs_z (z : ℂ) (h : |z - 2 * complex.I| + |z - 5| = 7) : ℝ :=
  classical.some (exists_minimum (λ z, |z|) (λ z, |z - 2 * complex.I| + |z - 5| = 7) h)

theorem minimum_value_of_abs_z : ∀ z : ℂ, 
  (|z - 2 * complex.I| + |z - 5| = 7) → |z| ≥ 0 
  → min_abs_z z (by sorry) = 10 / real.sqrt 29 :=
by 
  sorry

end minimum_value_of_abs_z_l291_291235


namespace first_player_has_winning_strategy_l291_291496

-- Define the initial heap sizes and rules of the game.
def initial_heaps : List Nat := [38, 45, 61, 70]

-- Define a function that checks using the rules whether the first player has a winning strategy given the initial heap sizes.
def first_player_wins : Bool :=
  -- placeholder for the actual winning strategy check logic
  sorry

-- Theorem statement referring to the equivalency proof problem where player one is established to have the winning strategy.
theorem first_player_has_winning_strategy : first_player_wins = true :=
  sorry

end first_player_has_winning_strategy_l291_291496


namespace relationship_a_b_c_d_l291_291519

theorem relationship_a_b_c_d 
  (a b c d : ℤ)
  (h : (a + b + 1) * (d + a + 2) = (c + d + 1) * (b + c + 2)) : 
  a + b + c + d = -2 := 
sorry

end relationship_a_b_c_d_l291_291519


namespace find_two_digit_number_l291_291951

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l291_291951


namespace independence_events_exactly_one_passing_l291_291459

-- Part 1: Independence of Events

def event_A (die1 : ℕ) : Prop :=
  die1 % 2 = 1

def event_B (die1 die2 : ℕ) : Prop :=
  (die1 + die2) % 3 = 0

def P_event_A : ℚ :=
  1 / 2

def P_event_B : ℚ :=
  1 / 3

def P_event_AB : ℚ :=
  1 / 6

theorem independence_events : P_event_AB = P_event_A * P_event_B :=
by
  sorry

-- Part 2: Probability of Exactly One Passing the Assessment

def probability_of_hitting (p : ℝ) : ℝ :=
  1 - (1 - p)^2

def P_A_hitting : ℝ :=
  0.7

def P_B_hitting : ℝ :=
  0.6

def probability_one_passing : ℝ :=
  (probability_of_hitting P_A_hitting) * (1 - probability_of_hitting P_B_hitting) + (1 - probability_of_hitting P_A_hitting) * (probability_of_hitting P_B_hitting)

theorem exactly_one_passing : probability_one_passing = 0.2212 :=
by
  sorry

end independence_events_exactly_one_passing_l291_291459


namespace matrix_characteristic_eq_l291_291871

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 2], ![2, 1, 2], ![2, 2, 1]]

theorem matrix_characteristic_eq :
  ∃ (a b c : ℚ), a = -6 ∧ b = -12 ∧ c = -18 ∧ 
  (B ^ 3 + a • (B ^ 2) + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0) :=
by
  sorry

end matrix_characteristic_eq_l291_291871


namespace count_cube_sums_less_than_1000_l291_291674

theorem count_cube_sums_less_than_1000 : 
  {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000}.to_finset.card = 43 :=
by
  sorry

end count_cube_sums_less_than_1000_l291_291674


namespace unique_sum_of_two_cubes_lt_1000_l291_291671

theorem unique_sum_of_two_cubes_lt_1000 
  : ∃ (sums : Finset ℕ), 
    (∀ x ∈ sums, ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ x = a^3 + b^3) 
    ∧ sums.card = 40 
    ∧ ∀ x ∈ sums, x < 1000 := 
by sorry

end unique_sum_of_two_cubes_lt_1000_l291_291671


namespace m_plus_n_is_23_l291_291869

noncomputable def find_m_plus_n : ℕ := 
  let A := 12
  let B := 4
  let C := 3
  let D := 3

  -- Declare the radius of E
  let radius_E : ℚ := (21 / 2)
  
  -- Let radius_E be written as m / n where m and n are relatively prime
  let (m : ℕ) := 21
  let (n : ℕ) := 2

  -- Calculate m + n
  m + n

theorem m_plus_n_is_23 : find_m_plus_n = 23 :=
by
  -- Proof is omitted
  sorry

end m_plus_n_is_23_l291_291869


namespace find_rowing_speed_of_person_Y_l291_291276

open Real

def rowing_speed (y : ℝ) : Prop :=
  ∀ (x : ℝ) (current_speed : ℝ),
    x = 6 → 
    (4 * (6 - current_speed) + 4 * (y + current_speed) = 4 * (6 + y)) →
    (16 * (y + current_speed) = 16 * (6 + current_speed) + 4 * (y - 6)) → 
    y = 10

-- We set up the proof problem without solving it.
theorem find_rowing_speed_of_person_Y : ∃ (y : ℝ), rowing_speed y :=
begin
  use 10,
  unfold rowing_speed,
  intros x current_speed h1 h2 h3,

  sorry
end

end find_rowing_speed_of_person_Y_l291_291276


namespace area_product_equal_no_consecutive_integers_l291_291237

open Real

-- Define the areas of the triangles for quadrilateral ABCD
variables {A B C D O : Point} 
variables {S1 S2 S3 S4 : Real}  -- Areas of triangles ABO, BCO, CDO, DAO

-- Given conditions
variables (h_intersection : lies_on_intersection O AC BD)
variables (h_areas : S1 = 1 / 2 * (|AO| * |BM|) ∧ S2 = 1 / 2 * (|CO| * |BM|) ∧ S3 = 1 / 2 * (|CO| * |DN|) ∧ S4 = 1 / 2 * (|AO| * |DN|))

-- Theorem for part (a)
theorem area_product_equal : S1 * S3 = S2 * S4 :=
by sorry

-- Theorem for part (b)
theorem no_consecutive_integers : ¬∃ (n : ℕ), S1 = n ∧ S2 = n + 1 ∧ S3 = n + 2 ∧ S4 = n + 3 :=
by sorry

end area_product_equal_no_consecutive_integers_l291_291237


namespace sum_of_angles_l291_291223

theorem sum_of_angles (A B C D E F : ℝ)
  (h1 : A + B + C = 180) 
  (h2 : D + E + F = 180) : 
  A + B + C + D + E + F = 360 := 
by 
  sorry

end sum_of_angles_l291_291223


namespace area_under_arccos_cos_eq_l291_291989

noncomputable def area_under_arccos_cos : ℝ :=
  ∫ x in (0 : ℝ)..(2 * Real.pi), Real.arccos (Real.cos x)

theorem area_under_arccos_cos_eq :
  area_under_arccos_cos = Real.pi^2 := by
  sorry

end area_under_arccos_cos_eq_l291_291989


namespace smallest_trees_in_three_types_l291_291863

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l291_291863


namespace download_time_l291_291974

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end download_time_l291_291974


namespace fraction_simplification_l291_291411

theorem fraction_simplification :
  (36 / 19) * (57 / 40) * (95 / 171) = (3 / 2) :=
by
  sorry

end fraction_simplification_l291_291411


namespace james_total_money_l291_291536

theorem james_total_money :
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  total_money = 135 := by
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  exact 135

end james_total_money_l291_291536


namespace number_of_sums_of_two_cubes_lt_1000_l291_291675

open Nat

def isPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

def sumOfTwoCubes (n : ℕ) : Prop := ∃ a b : ℕ, (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ n = a^3 + b^3

theorem number_of_sums_of_two_cubes_lt_1000 : 
  (Finset.filter (λ x => sumOfTwoCubes x) (Finset.range 1000)).card = 44 :=
by
  sorry

end number_of_sums_of_two_cubes_lt_1000_l291_291675


namespace arithmetic_seq_property_l291_291867

theorem arithmetic_seq_property (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 9 - a 10 = 24 := 
sorry

end arithmetic_seq_property_l291_291867


namespace percent_greater_than_fraction_l291_291767

theorem percent_greater_than_fraction : 
  (0.80 * 40) - (4/5) * 20 = 16 :=
by
  sorry

end percent_greater_than_fraction_l291_291767


namespace fraction_meaningful_l291_291688

theorem fraction_meaningful (x : ℝ) : (x - 2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end fraction_meaningful_l291_291688


namespace determinant_example_l291_291800

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem determinant_example : det_2x2 7 (-2) (-3) 6 = 36 := by
  sorry

end determinant_example_l291_291800


namespace cos_alpha_value_l291_291505

theorem cos_alpha_value (α : ℝ) (hα1 : 0 < α ∧ α < π / 2) (hα2 : Real.cos (α + π / 4) = 4 / 5) :
  Real.cos α = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end cos_alpha_value_l291_291505


namespace find_rate_per_kg_grapes_l291_291009

-- Define the main conditions
def rate_per_kg_mango := 55
def total_payment := 985
def kg_grapes := 7
def kg_mangoes := 9

-- Define the problem statement
theorem find_rate_per_kg_grapes (G : ℝ) : 
  (kg_grapes * G + kg_mangoes * rate_per_kg_mango = total_payment) → 
  G = 70 :=
by
  sorry

end find_rate_per_kg_grapes_l291_291009


namespace log_expression_value_l291_291284

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_expression_value :
  log_base 10 3 + 3 * log_base 10 2 + 2 * log_base 10 5 + 4 * log_base 10 3 + log_base 10 9 = 5.34 :=
by
  sorry

end log_expression_value_l291_291284


namespace sufficient_condition_l291_291299

theorem sufficient_condition (x y : ℤ) (h : x + y ≠ 2) : x ≠ 1 ∧ y ≠ 1 := 
sorry

end sufficient_condition_l291_291299


namespace remaining_tickets_l291_291623

-- Define initial tickets and used tickets
def initial_tickets := 13
def used_tickets := 6

-- Declare the theorem we want to prove
theorem remaining_tickets (initial_tickets used_tickets : ℕ) (h1 : initial_tickets = 13) (h2 : used_tickets = 6) : initial_tickets - used_tickets = 7 :=
by
  sorry

end remaining_tickets_l291_291623


namespace solid_views_same_shape_and_size_l291_291460

theorem solid_views_same_shape_and_size (solid : Type) (sphere triangular_pyramid cube cylinder : solid)
  (views_same_shape_and_size : solid → Bool) : 
  views_same_shape_and_size cylinder = false :=
sorry

end solid_views_same_shape_and_size_l291_291460


namespace not_perfect_square_T_l291_291356

noncomputable def operation (x y : ℝ) : ℝ := (x * y + 4) / (x + y)

axiom associative {x y z : ℝ} (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) :
  operation x (operation y z) = operation (operation x y) z

noncomputable def T (n : ℕ) : ℝ :=
  if h : n ≥ 4 then
    (List.range (n - 2)).foldr (λ x acc => operation (x + 3) acc) 3
  else 0

theorem not_perfect_square_T (n : ℕ) (h : n ≥ 4) :
  ¬ (∃ k : ℕ, (96 / (T n - 2) : ℝ) = k ^ 2) :=
sorry

end not_perfect_square_T_l291_291356


namespace find_f_10_l291_291508

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : f x = f (1 / x) * Real.log x + 10

theorem find_f_10 : f 10 = 10 :=
by
  sorry

end find_f_10_l291_291508


namespace number_of_pages_to_copy_l291_291971

-- Definitions based on the given conditions
def total_budget : ℕ := 5000
def service_charge : ℕ := 500
def copy_cost : ℕ := 3

-- Derived definition based on the conditions
def remaining_budget : ℕ := total_budget - service_charge

-- The statement we need to prove
theorem number_of_pages_to_copy : (remaining_budget / copy_cost) = 1500 :=
by {
  sorry
}

end number_of_pages_to_copy_l291_291971


namespace prop1_prop2_prop3_prop4_exists_l291_291660

variable {R : Type*} [LinearOrderedField R]
def f (b c x : R) : R := abs x * x + b * x + c

theorem prop1 (b c x : R) (h : b > 0) : 
  ∀ {x y : R}, x ≤ y → f b c x ≤ f b c y := 
sorry

theorem prop2 (b c : R) (h : b < 0) : 
  ¬ ∃ a : R, ∀ x : R, f b c x ≥ f b c a := 
sorry

theorem prop3 (b c x : R) : 
  f b c (-x) = f b c x + 2*c := 
sorry

theorem prop4_exists (c : R) : 
  ∃ b : R, ∃ x y z : R, f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z := 
sorry

end prop1_prop2_prop3_prop4_exists_l291_291660


namespace Rogers_expense_fraction_l291_291725

variables (B m s p : ℝ)

theorem Rogers_expense_fraction (h1 : m = 0.25 * (B - s))
                              (h2 : s = 0.10 * (B - m))
                              (h3 : p = 0.10 * (m + s)) :
  m + s + p = 0.34 * B :=
by
  sorry

end Rogers_expense_fraction_l291_291725


namespace simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l291_291181

-- Part 1: Proving the simplified form of arithmetic operations
theorem simplify_999_times_neg13 : 999 * (-13) = -12987 := by
  sorry

theorem simplify_complex_expr :
  999 * (118 + 4 / 5) + 333 * (-3 / 5) - 999 * (18 + 3 / 5) = 99900 := by
  sorry

-- Part 2: Proving the correct calculation of division
theorem correct_division_calculation : 6 / (-1 / 2 + 1 / 3) = -36 := by
  sorry

end simplify_999_times_neg13_simplify_complex_expr_correct_division_calculation_l291_291181


namespace calculate_expression_l291_291016

variable (x y : ℚ)

theorem calculate_expression (h₁ : x = 4 / 6) (h₂ : y = 5 / 8) : 
  (6 * x + 8 * y) / (48 * x * y) = 9 / 20 :=
by
  -- proof steps here
  sorry

end calculate_expression_l291_291016


namespace solve_number_l291_291945

theorem solve_number :
  ∃ (M : ℕ), 
    (10 ≤ M ∧ M < 100) ∧ -- M is a two-digit number
    M % 2 = 1 ∧ -- M is odd
    M % 9 = 0 ∧ -- M is a multiple of 9
    let d₁ := M / 10, d₂ := M % 10 in -- digits of M
    d₁ * d₂ = (Nat.sqrt (d₁ * d₂))^2 := -- product of digits is a perfect square
begin
  use 99,
  split,
  { -- 10 ≤ 99 < 100
    exact and.intro (le_refl 99) (lt_add_one 99),
  },
  split,
  { -- 99 is odd
    exact nat.odd_iff.2 (nat.dvd_one.trans (nat.dvd_refl 2)),
  },
  split,
  { -- 99 is a multiple of 9
    exact nat.dvd_of_mod_eq_zero (by norm_num),
  },
  { -- product of digits is a perfect square
    let d₁ := 99 / 10,
    let d₂ := 99 % 10,
    have h : d₁ * d₂ = 9 * 9, by norm_num,
    rw h,
    exact (by norm_num : 81 = 9 ^ 2).symm
  }
end

end solve_number_l291_291945


namespace base_number_is_two_l291_291686

theorem base_number_is_two (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^22)
  (h2 : n = 21) : x = 2 :=
sorry

end base_number_is_two_l291_291686


namespace find_a_l291_291097

-- Definition of the curve y = x^3 + ax + 1
def curve (x a : ℝ) : ℝ := x^3 + a * x + 1

-- Definition of the tangent line y = 2x + 1
def tangent_line (x : ℝ) : ℝ := 2 * x + 1

-- The slope of the tangent line is 2
def slope_of_tangent_line (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + a

theorem find_a (a : ℝ) : 
  (∃ x₀, curve x₀ a = tangent_line x₀) ∧ (∃ x₀, slope_of_tangent_line x₀ a = 2) → a = 2 :=
by
  sorry

end find_a_l291_291097


namespace sin_2pi_minus_alpha_l291_291040

theorem sin_2pi_minus_alpha (α : ℝ) (h₁ : Real.cos (α + Real.pi) = Real.sqrt 3 / 2) (h₂ : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
    Real.sin (2 * Real.pi - α) = -1 / 2 := 
sorry

end sin_2pi_minus_alpha_l291_291040


namespace value_of_six_inch_cube_l291_291163

theorem value_of_six_inch_cube :
  let four_inch_cube_value := 400
  let four_inch_side_length := 4
  let six_inch_side_length := 6
  let volume (s : ℕ) : ℕ := s ^ 3
  (volume six_inch_side_length / volume four_inch_side_length) * four_inch_cube_value = 1350 := by
sorry

end value_of_six_inch_cube_l291_291163


namespace find_angle_BDC_l291_291057

theorem find_angle_BDC
  (CAB CAD DBA DBC : ℝ)
  (h1 : CAB = 40)
  (h2 : CAD = 30)
  (h3 : DBA = 75)
  (h4 : DBC = 25) :
  ∃ BDC : ℝ, BDC = 45 :=
by
  sorry

end find_angle_BDC_l291_291057


namespace sum_of_integers_990_l291_291584

theorem sum_of_integers_990 :
  ∃ (n m : ℕ), (n * (n + 1) = 990 ∧ (m - 1) * m * (m + 1) = 990 ∧ (n + n + 1 + m - 1 + m + m + 1 = 90)) :=
sorry

end sum_of_integers_990_l291_291584


namespace james_total_money_l291_291535

theorem james_total_money :
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  total_money = 135 := by
  let bills_found := 3
  let bill_value := 20
  let initial_money := 75
  let total_from_bills := bills_found * bill_value
  let total_money := total_from_bills + initial_money
  exact 135

end james_total_money_l291_291535


namespace diameter_twice_radius_l291_291067

theorem diameter_twice_radius (r d : ℝ) (h : d = 2 * r) : d = 2 * r :=
by
  exact h

end diameter_twice_radius_l291_291067


namespace solve_for_C_l291_291647

theorem solve_for_C : 
  ∃ C : ℝ, 80 - (5 - (6 + 2 * (7 - C - 5))) = 89 ∧ C = -2 :=
by
  sorry

end solve_for_C_l291_291647


namespace Annie_total_cookies_l291_291967

theorem Annie_total_cookies :
  let monday_cookies := 5
  let tuesday_cookies := 2 * monday_cookies 
  let wednesday_cookies := 1.4 * tuesday_cookies
  monday_cookies + tuesday_cookies + wednesday_cookies = 29 :=
by
  sorry

end Annie_total_cookies_l291_291967


namespace earnings_from_jam_l291_291321

def betty_strawberries : ℕ := 16
def matthew_additional_strawberries : ℕ := 20
def jar_strawberries : ℕ := 7
def jar_price : ℕ := 4

theorem earnings_from_jam :
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  total_money = 40 :=
by
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  show total_money = 40
  sorry

end earnings_from_jam_l291_291321


namespace compare_m_n_l291_291194

noncomputable def m (a : ℝ) : ℝ := 6^a / (36^(a + 1) + 1)
noncomputable def n (b : ℝ) : ℝ := (1/3) * b^2 - b + (5/6)

theorem compare_m_n (a b : ℝ) : m a ≤ n b := sorry

end compare_m_n_l291_291194


namespace weight_of_banana_l291_291273

theorem weight_of_banana (A B G : ℝ) (h1 : 3 * A = G) (h2 : 4 * B = 2 * A) (h3 : G = 576) : B = 96 :=
by
  sorry

end weight_of_banana_l291_291273


namespace hyperbola_ellipse_equations_l291_291469

theorem hyperbola_ellipse_equations 
  (F1 F2 P : ℝ × ℝ) 
  (hF1 : F1 = (0, -5))
  (hF2 : F2 = (0, 5))
  (hP : P = (3, 4)) :
  (∃ a b : ℝ, a^2 = 40 ∧ b^2 = 16 ∧ 
    ∀ x y : ℝ, (y^2 / 40 + x^2 / 15 = 1 ↔ y^2 / a^2 + x^2 / (a^2 - 25) = 1) ∧
    (y^2 / 16 - x^2 / 9 = 1 ↔ y^2 / b^2 - x^2 / (25 - b^2) = 1)) :=
sorry

end hyperbola_ellipse_equations_l291_291469


namespace remainder_of_31_pow_31_plus_31_div_32_l291_291151

theorem remainder_of_31_pow_31_plus_31_div_32 :
  (31^31 + 31) % 32 = 30 := 
by 
  trivial -- Replace with actual proof

end remainder_of_31_pow_31_plus_31_div_32_l291_291151


namespace emma_bank_account_balance_l291_291343

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end emma_bank_account_balance_l291_291343


namespace sum_even_sub_sum_odd_l291_291027

def sum_arith_seq (a1 an d : ℕ) (n : ℕ) : ℕ :=
  n * (a1 + an) / 2

theorem sum_even_sub_sum_odd :
  let n_even := 50
  let n_odd := 15
  let s_even := sum_arith_seq 2 100 2 n_even
  let s_odd := sum_arith_seq 1 29 2 n_odd
  s_even - s_odd = 2325 :=
by
  sorry

end sum_even_sub_sum_odd_l291_291027


namespace union_of_sets_l291_291514

open Set

theorem union_of_sets (M N : Set ℝ) (hM : M = {x | -3 < x ∧ x < 1}) (hN : N = {x | x ≤ -3}) :
  M ∪ N = {x | x < 1} := by
  sorry

end union_of_sets_l291_291514


namespace line_intersects_y_axis_at_5_l291_291626

theorem line_intersects_y_axis_at_5 :
  ∃ (b : ℝ), ∀ (x y : ℝ), (x - 2 = 0 ∧ y - 9 = 0) ∨ (x - 4 = 0 ∧ y - 13 = 0) →
  (y = 2 * x + b) ∧ (b = 5) :=
by
  sorry

end line_intersects_y_axis_at_5_l291_291626


namespace no_prime_pair_summing_to_53_l291_291388

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l291_291388


namespace derivative_at_one_l291_291510

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem derivative_at_one : deriv f 1 = 2 :=
by sorry

end derivative_at_one_l291_291510


namespace quadratic_real_roots_l291_291690

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0) ∧ (∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) → k ≥ -1 :=
by
  sorry

end quadratic_real_roots_l291_291690


namespace polynomial_transformation_l291_291548

-- Given the conditions of the polynomial function g and the provided transformation
-- We aim to prove the equivalence in a mathematically formal way using Lean

theorem polynomial_transformation (g : ℝ → ℝ) (h : ∀ x : ℝ, g (x^2 + 2) = x^4 + 5 * x^2 + 1) :
  ∀ x : ℝ, g (x^2 - 2) = x^4 - 3 * x^2 - 3 :=
by
  intro x
  sorry

end polynomial_transformation_l291_291548


namespace eldorado_license_plates_count_l291_291066

theorem eldorado_license_plates_count:
  let letters := 26
  let digits := 10
  let total := (letters ^ 3) * (digits ^ 4)
  total = 175760000 :=
by
  sorry

end eldorado_license_plates_count_l291_291066


namespace quadratic_roots_l291_291249

theorem quadratic_roots (x : ℝ) : 
  (x^2 + 3 * x - 1 = 0) ↔ (x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) := 
by
  sorry

end quadratic_roots_l291_291249


namespace part1_part2_l291_291602

-- Part 1: Proving the inequality
theorem part1 (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := by
  sorry

-- Part 2: Maximizing 2a + b
theorem part2 (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_constraint : a^2 + b^2 = 5) : 
  2 * a + b ≤ 5 := by
  sorry

end part1_part2_l291_291602


namespace distance_between_foci_of_hyperbola_is_correct_l291_291812

noncomputable def distance_between_foci_of_hyperbola : ℝ := 
  let a_sq := 50
  let b_sq := 8
  let c_sq := a_sq + b_sq
  let c := Real.sqrt c_sq
  2 * c

theorem distance_between_foci_of_hyperbola_is_correct :
  distance_between_foci_of_hyperbola = 2 * Real.sqrt 58 :=
by
  sorry

end distance_between_foci_of_hyperbola_is_correct_l291_291812


namespace solve_for_s_l291_291355

theorem solve_for_s (s : ℝ) :
  (s^2 - 6 * s + 8) / (s^2 - 9 * s + 14) = (s^2 - 3 * s - 18) / (s^2 - 2 * s - 24) →
  s = -5 / 4 :=
by {
  sorry
}

end solve_for_s_l291_291355


namespace minimum_trees_with_at_least_three_types_l291_291855

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l291_291855


namespace black_square_area_l291_291008

-- Define the edge length of the cube
def edge_length := 12

-- Define the total amount of yellow paint available
def yellow_paint_area := 432

-- Define the total surface area of the cube
def total_surface_area := 6 * (edge_length * edge_length)

-- Define the area covered by yellow paint per face
def yellow_per_face := yellow_paint_area / 6

-- Define the area of one face of the cube
def face_area := edge_length * edge_length

-- State the theorem: the area of the black square on each face
theorem black_square_area : (face_area - yellow_per_face) = 72 := by
  sorry

end black_square_area_l291_291008


namespace fraction_of_number_l291_291138

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l291_291138


namespace grocer_display_proof_l291_291468

-- Define the arithmetic sequence conditions
def num_cans_in_display (n : ℕ) : Prop :=
  let a := 1
  let d := 2
  (n * n = 225) 

-- Prove the total weight is 1125 kg
def total_weight_supported (weight_per_can : ℕ) (total_cans : ℕ) : Prop :=
  (total_cans * weight_per_can = 1125)

-- State the main theorem combining the two proofs.
theorem grocer_display_proof (n weight_per_can total_cans : ℕ) :
  num_cans_in_display n → total_weight_supported weight_per_can total_cans → 
  n = 15 ∧ total_cans * weight_per_can = 1125 :=
by {
  sorry
}

end grocer_display_proof_l291_291468


namespace emma_final_balance_correct_l291_291340

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end emma_final_balance_correct_l291_291340


namespace correct_points_per_answer_l291_291080

noncomputable def points_per_correct_answer (total_questions : ℕ) 
  (answered_correctly : ℕ) (final_score : ℝ) (penalty_per_incorrect : ℝ)
  (total_incorrect : ℕ := total_questions - answered_correctly) 
  (points_subtracted : ℝ := total_incorrect * penalty_per_incorrect) 
  (earned_points : ℝ := final_score + points_subtracted) : ℝ := 
    earned_points / answered_correctly

theorem correct_points_per_answer :
  points_per_correct_answer 120 104 100 0.25 = 1 := 
by 
  sorry

end correct_points_per_answer_l291_291080


namespace sum_ratio_arithmetic_sequence_l291_291545

theorem sum_ratio_arithmetic_sequence (a₁ d : ℚ) (h : d ≠ 0) 
  (S : ℕ → ℚ)
  (h_sum : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
by
  sorry

end sum_ratio_arithmetic_sequence_l291_291545


namespace carol_sold_cupcakes_l291_291994

variable (initial_cupcakes := 30) (additional_cupcakes := 28) (final_cupcakes := 49)

theorem carol_sold_cupcakes : (initial_cupcakes + additional_cupcakes - final_cupcakes = 9) :=
by sorry

end carol_sold_cupcakes_l291_291994


namespace difference_of_squares_expression_l291_291145

theorem difference_of_squares_expression
  (x y : ℝ) :
  (x + 2 * y) * (x - 2 * y) = x^2 - (2 * y)^2 :=
by sorry

end difference_of_squares_expression_l291_291145


namespace decomposition_sum_of_cubes_l291_291257

theorem decomposition_sum_of_cubes 
  (a b c d e : ℤ) 
  (h : (512 : ℤ) * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 60 := 
sorry

end decomposition_sum_of_cubes_l291_291257


namespace meadow_to_campsite_distance_l291_291397

variable (d1 d2 d_total d_meadow_to_campsite : ℝ)

theorem meadow_to_campsite_distance
  (h1 : d1 = 0.2)
  (h2 : d2 = 0.4)
  (h_total : d_total = 0.7)
  (h_before_meadow : d_before_meadow = d1 + d2)
  (h_distance : d_meadow_to_campsite = d_total - d_before_meadow) :
  d_meadow_to_campsite = 0.1 :=
by 
  sorry

end meadow_to_campsite_distance_l291_291397


namespace minimize_distance_l291_291822

theorem minimize_distance
  (a b c d : ℝ)
  (h1 : (a + 3 * Real.log a) / b = 1)
  (h2 : (d - 3) / (2 * c) = 1) :
  (a - c)^2 + (b - d)^2 = (9 / 5) * (Real.log (Real.exp 1 / 3))^2 :=
by sorry

end minimize_distance_l291_291822


namespace number_of_correct_conclusions_l291_291816

noncomputable def floor (x : ℝ) : ℤ := Int.floor x
noncomputable def f (x : ℝ) : ℝ := x - floor x

theorem number_of_correct_conclusions : 
  ∃ n, n = 3 ∧ 
  (0 ≤ f 0) ∧ 
  (∀ x : ℝ, 0 ≤ f x) ∧ 
  (∀ x : ℝ, f x < 1) ∧ 
  (∀ x : ℝ, f (x + 1) = f x) ∧ 
  ¬ (∀ x : ℝ, f (-x) = f x) := 
sorry

end number_of_correct_conclusions_l291_291816


namespace largest_sum_pairs_l291_291731

theorem largest_sum_pairs (a b c d : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d) (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) (h₆ : a < b) (h₇ : b < c) (h₈ : c < d)
(h₉ : a + b = 9 ∨ a + b = 10) (h₁₀ : b + c = 9 ∨ b + c = 10)
(h₁₁ : b + d = 12) (h₁₂ : c + d = 13) :
d = 8 ∨ d = 7.5 :=
sorry

end largest_sum_pairs_l291_291731


namespace calculate_total_income_l291_291786

/-- Total income calculation proof for a person with given distributions and remaining amount -/
theorem calculate_total_income
  (I : ℝ) -- total income
  (leftover : ℝ := 40000) -- leftover amount after distribution and donation
  (c1_percentage : ℝ := 3 * 0.15) -- percentage given to children
  (c2_percentage : ℝ := 0.30) -- percentage given to wife
  (c3_percentage : ℝ := 0.05) -- percentage donated to orphan house
  (remaining_percentage : ℝ := 1 - (c1_percentage + c2_percentage)) -- remaining percentage after children and wife
  (R : ℝ := remaining_percentage * I) -- remaining amount after children and wife
  (donation : ℝ := c3_percentage * R) -- amount donated to orphan house)
  (left_amount : ℝ := R - donation) -- final remaining amount
  (income : ℝ := (leftover / (1 - remaining_percentage * (1 - c3_percentage)))) -- calculation of the actual income
  : I = income := sorry

end calculate_total_income_l291_291786


namespace colin_average_mile_time_l291_291796

theorem colin_average_mile_time :
  let first_mile_time := 6
  let next_two_miles_total_time := 5 + 5
  let fourth_mile_time := 4
  let total_time := first_mile_time + next_two_miles_total_time + fourth_mile_time
  let number_of_miles := 4
  (total_time / number_of_miles) = 5 := by
    let first_mile_time := 6
    let next_two_miles_total_time := 5 + 5
    let fourth_mile_time := 4
    let total_time := first_mile_time + next_two_miles_total_time + fourth_mile_time
    let number_of_miles := 4
    have h1 : total_time = 20 := by sorry
    have h2 : total_time / number_of_miles = 20 / 4 := by sorry
    have h3 : 20 / 4 = 5 := by sorry
    exact Eq.trans (Eq.trans h2 h3) h1.symm

end colin_average_mile_time_l291_291796


namespace moles_of_ammonium_nitrate_formed_l291_291350

def ammonia := ℝ
def nitric_acid := ℝ
def ammonium_nitrate := ℝ

-- Define the stoichiometric coefficients from the balanced equation.
def stoichiometric_ratio_ammonia : ℝ := 1
def stoichiometric_ratio_nitric_acid : ℝ := 1
def stoichiometric_ratio_ammonium_nitrate : ℝ := 1

-- Define the initial moles of reactants.
def initial_moles_ammonia (moles : ℝ) : Prop := moles = 3
def initial_moles_nitric_acid (moles : ℝ) : Prop := moles = 3

-- The reaction goes to completion as all reactants are used:
theorem moles_of_ammonium_nitrate_formed :
  ∀ (moles_ammonia moles_nitric_acid : ℝ),
    initial_moles_ammonia moles_ammonia →
    initial_moles_nitric_acid moles_nitric_acid →
    (moles_ammonia / stoichiometric_ratio_ammonia) = 
    (moles_nitric_acid / stoichiometric_ratio_nitric_acid) →
    (moles_ammonia / stoichiometric_ratio_ammonia) * stoichiometric_ratio_ammonium_nitrate = 3 :=
by
  intros moles_ammonia moles_nitric_acid h_ammonia h_nitric_acid h_ratio
  rw [h_ammonia, h_nitric_acid] at *
  simp only [stoichiometric_ratio_ammonia, stoichiometric_ratio_nitric_acid, stoichiometric_ratio_ammonium_nitrate] at *
  sorry

end moles_of_ammonium_nitrate_formed_l291_291350


namespace minimum_trees_with_at_least_three_types_l291_291856

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l291_291856


namespace gumballs_result_l291_291541

def gumballs_after_sharing_equally (initial_joanna : ℕ) (initial_jacques : ℕ) (multiplier : ℕ) : ℕ :=
  let joanna_total := initial_joanna + initial_joanna * multiplier
  let jacques_total := initial_jacques + initial_jacques * multiplier
  (joanna_total + jacques_total) / 2

theorem gumballs_result :
  gumballs_after_sharing_equally 40 60 4 = 250 :=
by
  sorry

end gumballs_result_l291_291541


namespace triangle_right_angle_l291_291376

theorem triangle_right_angle (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : γ = α + β) : γ = 90 :=
by
  sorry

end triangle_right_angle_l291_291376


namespace fraction_of_number_l291_291127

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l291_291127


namespace cost_of_450_candies_l291_291774

theorem cost_of_450_candies (box_cost : ℝ) (box_candies : ℕ) (total_candies : ℕ) 
  (h1 : box_cost = 7.50) (h2 : box_candies = 30) (h3 : total_candies = 450) : 
  (total_candies / box_candies) * box_cost = 112.50 :=
by
  sorry

end cost_of_450_candies_l291_291774


namespace no_prime_pairs_sum_53_l291_291395

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l291_291395


namespace arithmetic_sequence_nth_term_l291_291578

noncomputable def nth_arithmetic_term (a : ℤ) (n : ℕ) : ℤ :=
  let a1 := a - 1
  let a2 := a + 1
  let a3 := 2 * a + 3
  if 2 * (a + 1) = (a - 1) + (2 * a + 3) then
    -1 + (n - 1) * 2
  else
    sorry

theorem arithmetic_sequence_nth_term (a : ℤ) (n : ℕ) (h : 2 * (a + 1) = (a - 1) + (2 * a + 3)) :
  nth_arithmetic_term a n = 2 * (n : ℤ) - 3 :=
by
  sorry

end arithmetic_sequence_nth_term_l291_291578


namespace solve_system_l291_291088

theorem solve_system:
  ∃ (x y : ℝ), (26 * x^2 + 42 * x * y + 17 * y^2 = 10 ∧ 10 * x^2 + 18 * x * y + 8 * y^2 = 6) ↔
  (x = -1 ∧ y = 2) ∨ (x = -11 ∧ y = 14) ∨ (x = 11 ∧ y = -14) ∨ (x = 1 ∧ y = -2) :=
by
  sorry

end solve_system_l291_291088


namespace factorization_correct_l291_291256

theorem factorization_correct : ∀ x : ℝ, (x^2 - 2*x - 9 = 0) → ((x-1)^2 = 10) :=
by 
  intros x h
  sorry

end factorization_correct_l291_291256


namespace extended_kobish_word_count_l291_291265

def extended_kobish_alphabet : Finset Char :=
  ('A' : Finset Char).insert 'B'.insert 'C'.insert 'D'.insert 'E'.insert 'F'
  .insert 'G'
  -- insert all letters up to 'U'
  .insert 'H'.insert 'I'.insert 'J'.insert 'K'.insert 'L'.insert 'M'.insert 'N'
  .insert 'O'.insert 'P'.insert 'Q'.insert 'R'.insert 'S'.insert 'T'.insert 'U'

def number_of_valid_words : ℕ :=
  let total_words (n : ℕ) : ℕ := (extended_kobish_alphabet.card) ^ n 
  let total_words_without_B (n : ℕ) : ℕ := (extended_kobish_alphabet.card - 1) ^ n 
  total_words 1 - total_words_without_B 1 + 
  total_words 2 - total_words_without_B 2 + 
  total_words 3 - total_words_without_B 3 + 
  total_words 4 - total_words_without_B 4

theorem extended_kobish_word_count : number_of_valid_words = 35784 :=
by
  sorry

end extended_kobish_word_count_l291_291265


namespace find_w_l291_291726

noncomputable def roots_cubic_eq (x : ℝ) : ℝ := x^3 + 2 * x^2 + 5 * x - 8

def p : ℝ := sorry -- one root of x^3 + 2x^2 + 5x - 8 = 0
def q : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0
def r : ℝ := sorry -- another root of x^3 + 2x^2 + 5x - 8 = 0

theorem find_w 
  (h1 : roots_cubic_eq p = 0)
  (h2 : roots_cubic_eq q = 0)
  (h3 : roots_cubic_eq r = 0)
  (h4 : p + q + r = -2): 
  ∃ w : ℝ, w = 18 := 
sorry

end find_w_l291_291726


namespace cos_squared_formula_15deg_l291_291153

theorem cos_squared_formula_15deg :
  (Real.cos (15 * Real.pi / 180))^2 - (1 / 2) = (Real.sqrt 3) / 4 :=
by
  sorry

end cos_squared_formula_15deg_l291_291153


namespace madeline_biked_more_l291_291401

def madeline_speed : ℕ := 12
def madeline_time : ℕ := 3
def max_speed : ℕ := 15
def max_time : ℕ := 2

theorem madeline_biked_more : (madeline_speed * madeline_time) - (max_speed * max_time) = 6 := 
by 
  sorry

end madeline_biked_more_l291_291401


namespace integer_a_conditions_l291_291986

theorem integer_a_conditions (a : ℤ) :
  (∃ (x y : ℕ), x ≠ y ∧ (a * x * y + 1) ∣ (a * x^2 + 1) ^ 2) → a ≥ -1 :=
sorry

end integer_a_conditions_l291_291986


namespace polynomial_coeffs_l291_291821

theorem polynomial_coeffs (a b c d e f : ℤ) :
  (((2 : ℤ) * x - 1) ^ 5 = a * x ^ 5 + b * x ^ 4 + c * x ^ 3 + d * x ^ 2 + e * x + f) →
  (a + b + c + d + e + f = 1) ∧ 
  (b + c + d + e = -30) ∧
  (a + c + e = 122) :=
by
  intro h
  sorry  -- Proof omitted

end polynomial_coeffs_l291_291821


namespace initial_punch_amount_l291_291713

theorem initial_punch_amount (P : ℝ) (h1 : 16 = (P / 2 + 2 + 12)) : P = 4 :=
by
  sorry

end initial_punch_amount_l291_291713


namespace Lennon_total_reimbursement_l291_291543

def mileage_reimbursement (industrial_weekday: ℕ → ℕ) (commercial_weekday: ℕ → ℕ) (weekend: ℕ → ℕ) : ℕ :=
  let industrial_rate : ℕ := 36
  let commercial_weekday_rate : ℕ := 42
  let weekend_rate : ℕ := 45
  (industrial_weekday 1 * industrial_rate + commercial_weekday 1 * commercial_weekday_rate)    -- Monday
  + (industrial_weekday 2 * industrial_rate + commercial_weekday 2 * commercial_weekday_rate + commercial_weekday 3 * commercial_weekday_rate)  -- Tuesday
  + (industrial_weekday 3 * industrial_rate + commercial_weekday 3 * commercial_weekday_rate)    -- Wednesday
  + (commercial_weekday 4 * commercial_weekday_rate + commercial_weekday 5 * commercial_weekday_rate)  -- Thursday
  + (industrial_weekday 5 * industrial_rate + commercial_weekday 6 * commercial_weekday_rate + industrial_weekday 6 * industrial_rate)    -- Friday
  + (weekend 1 * weekend_rate)                                       -- Saturday

def monday_industrial_miles : ℕ := 10
def monday_commercial_miles : ℕ := 8

def tuesday_industrial_miles : ℕ := 12
def tuesday_commercial_miles_1 : ℕ := 9
def tuesday_commercial_miles_2 : ℕ := 5

def wednesday_industrial_miles : ℕ := 15
def wednesday_commercial_miles : ℕ := 5

def thursday_commercial_miles_1 : ℕ := 10
def thursday_commercial_miles_2 : ℕ := 10

def friday_industrial_miles_1 : ℕ := 5
def friday_commercial_miles : ℕ := 8
def friday_industrial_miles_2 : ℕ := 3

def saturday_commercial_miles : ℕ := 12

def reimbursement_total :=
  mileage_reimbursement
    (fun day => if day = 1 then monday_industrial_miles else if day = 2 then tuesday_industrial_miles else if day = 3 then wednesday_industrial_miles else if day = 5 then friday_industrial_miles_1 + friday_industrial_miles_2 else 0)
    (fun day => if day = 1 then monday_commercial_miles else if day = 2 then tuesday_commercial_miles_1 + tuesday_commercial_miles_2 else if day = 3 then wednesday_commercial_miles else if day = 4 then thursday_commercial_miles_1 + thursday_commercial_miles_2 else if day = 6 then friday_commercial_miles else 0)
    (fun day => if day = 1 then saturday_commercial_miles else 0)

theorem Lennon_total_reimbursement : reimbursement_total = 4470 := 
by sorry

end Lennon_total_reimbursement_l291_291543


namespace cos_equality_l291_291640

noncomputable def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

theorem cos_equality : ∃ n : ℝ, (0 ≤ n ∧ n ≤ 180) ∧ Real.cos (degrees_to_radians n) = Real.cos (degrees_to_radians 317) :=
by
  use 43
  simp [degrees_to_radians, Real.cos]
  sorry

end cos_equality_l291_291640


namespace distinct_solutions_abs_eq_l291_291663

theorem distinct_solutions_abs_eq (x : ℝ) : 
  (|x - 10| = |x + 4|) → ∃! x, |x - 10| = |x + 4| :=
by
  -- We will omit the proof steps and insert sorry to comply with the requirement.
  sorry

end distinct_solutions_abs_eq_l291_291663


namespace find_two_digit_number_l291_291942

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l291_291942


namespace eval_floor_abs_value_l291_291022

theorem eval_floor_abs_value : ⌊|(-45.8 : ℝ)|⌋ = 45 := by
  sorry -- Proof is to be filled in

end eval_floor_abs_value_l291_291022


namespace simplify_exponent_expression_l291_291296

theorem simplify_exponent_expression : 2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end simplify_exponent_expression_l291_291296


namespace family_total_weight_gain_l291_291319

def orlando_gain : ℕ := 5
def jose_gain : ℕ := 2 * orlando_gain + 2
def fernando_gain : ℕ := (jose_gain / 2) - 3
def total_weight_gain : ℕ := orlando_gain + jose_gain + fernando_gain

theorem family_total_weight_gain : total_weight_gain = 20 := by
  -- proof omitted
  sorry

end family_total_weight_gain_l291_291319


namespace solve_for_x_l291_291629

def F (x y z : ℝ) : ℝ := x * y^3 + z^2

theorem solve_for_x :
  F x 3 2 = F x 2 5 → x = 21/19 :=
  by
  sorry

end solve_for_x_l291_291629


namespace min_value_of_quadratic_l291_291751

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l291_291751


namespace num_permutations_with_P_gt_without_P_l291_291875

noncomputable def permutations_with_property_P (n : ℕ) : Finset (Finset (Fin n)) :=
{ perm | ∃ i, (1 ≤ i ∧ i < 2 * n) ∧ (|perm[i] - perm[i+1]| = n) }

noncomputable def permutations_without_property_P (n : ℕ) : Finset (Finset (Fin n)) :=
{ perm | ¬ ∃ i, (1 ≤ i ∧ i < 2 * n) ∧ (|perm[i] - perm[i+1]| = n) }

theorem num_permutations_with_P_gt_without_P (n : ℕ) :
  (permutations_with_property_P n).card > (permutations_without_property_P n).card :=
sorry

end num_permutations_with_P_gt_without_P_l291_291875


namespace find_sum_of_numbers_l291_291902

theorem find_sum_of_numbers 
  (a b : ℕ)
  (h₁ : a.gcd b = 5)
  (h₂ : a * b / a.gcd b = 120)
  (h₃ : (1 : ℚ) / a + 1 / b = 0.09166666666666666) :
  a + b = 55 := 
sorry

end find_sum_of_numbers_l291_291902


namespace f_zero_f_odd_range_of_x_l291_291020

variable {f : ℝ → ℝ}

axiom func_property (x y : ℝ) : f (x + y) = f x + f y
axiom f_third : f (1 / 3) = 1
axiom f_positive (x : ℝ) : x > 0 → f x > 0

-- Part (1)
theorem f_zero : f 0 = 0 :=
sorry

-- Part (2)
theorem f_odd (x : ℝ) : f (-x) = -f x :=
sorry

-- Part (3)
theorem range_of_x (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 :=
sorry

end f_zero_f_odd_range_of_x_l291_291020


namespace fraction_inequality_l291_291839

theorem fraction_inequality (a b c d : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) :
  (a / d) < (b / c) :=
sorry

end fraction_inequality_l291_291839


namespace fraction_multiplication_l291_291122

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l291_291122


namespace routes_M_to_N_l291_291221

-- Define nodes
inductive Node
| M | A | B | C | D | E | N

open Node

-- Define edges in the directed graph
def edge : Node → Node → Prop
| M, A := true
| M, B := true
| A, C := true
| A, D := true
| B, E := true
| B, C := true
| C, N := true
| D, N := true
| E, N := true
| D, C := true
| _, _ := false

-- Define a path exists function
def path_exists (graph: Node → Node → Prop) (start goal : Node) : Prop := -- recursive definition needed
sorry

-- Define a function to count the number of distinct routes
noncomputable def count_routes (graph: Node → Node → Prop) (start goal : Node) : ℕ :=
sorry

-- The theorem to prove
theorem routes_M_to_N : count_routes edge M N = 5 :=
by
  sorry

end routes_M_to_N_l291_291221


namespace min_rows_512_l291_291982

theorem min_rows_512 (n : ℕ) (table : ℕ → ℕ → ℕ) 
  (H : ∀ A (i j : ℕ), i < 10 → j < 10 → i ≠ j → ∃ B, B < n ∧ (table B i ≠ table A i) ∧ (table B j ≠ table A j) ∧ ∀ k, k ≠ i ∧ k ≠ j → table B k = table A k) : 
  n ≥ 512 :=
sorry

end min_rows_512_l291_291982


namespace functional_equation_solution_l291_291635

noncomputable def f (x : ℝ) (c : ℝ) : ℝ :=
  (c * x - c^2) / (1 + c)

def g (x : ℝ) (c : ℝ) : ℝ :=
  c * x - c^2

theorem functional_equation_solution (f g : ℝ → ℝ) (c : ℝ) (h : c ≠ -1) :
  (∀ x y : ℝ, f (x + g y) = x * f y - y * f x + g x) ∧
  (∀ x, f x = (c * x - c^2) / (1 + c)) ∧
  (∀ x, g x = c * x - c^2) :=
sorry

end functional_equation_solution_l291_291635


namespace problem_proof_l291_291873

variable {x y z : ℝ}

theorem problem_proof (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x^2 + y^2 + z^2 + 2 * x * y * z = 1) : 2 * (x + y + z) ≤ 3 := 
sorry

end problem_proof_l291_291873


namespace value_of_business_calculation_l291_291165

noncomputable def value_of_business (total_shares_sold_value : ℝ) (shares_fraction_sold : ℝ) (ownership_fraction : ℝ) : ℝ :=
  (total_shares_sold_value / shares_fraction_sold) * ownership_fraction⁻¹

theorem value_of_business_calculation :
  value_of_business 45000 (3/4) (2/3) = 90000 :=
by
  sorry

end value_of_business_calculation_l291_291165


namespace opposite_of_negative_seven_l291_291263

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_seven : opposite (-7) = 7 := 
by 
  sorry

end opposite_of_negative_seven_l291_291263


namespace gcd_three_numbers_l291_291028

theorem gcd_three_numbers :
  gcd (gcd 324 243) 135 = 27 :=
by
  sorry

end gcd_three_numbers_l291_291028


namespace evaluate_g_of_h_l291_291202

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_g_of_h : g (h (-2)) = 4328 := 
by
  sorry

end evaluate_g_of_h_l291_291202


namespace tiger_distance_proof_l291_291001

-- Declare the problem conditions
def tiger_initial_speed : ℝ := 25
def tiger_initial_time : ℝ := 3
def tiger_slow_speed : ℝ := 10
def tiger_slow_time : ℝ := 4
def tiger_chase_speed : ℝ := 50
def tiger_chase_time : ℝ := 0.5

-- Compute individual distances
def distance1 := tiger_initial_speed * tiger_initial_time
def distance2 := tiger_slow_speed * tiger_slow_time
def distance3 := tiger_chase_speed * tiger_chase_time

-- Compute the total distance
def total_distance := distance1 + distance2 + distance3

-- The final theorem to prove
theorem tiger_distance_proof : total_distance = 140 := by
  sorry

end tiger_distance_proof_l291_291001


namespace correlation_coefficient_l291_291068

theorem correlation_coefficient (variation_explained_by_height : ℝ)
    (variation_explained_by_errors : ℝ)
    (total_variation : variation_explained_by_height + variation_explained_by_errors = 1)
    (percentage_explained_by_height : variation_explained_by_height = 0.71) :
  variation_explained_by_height = 0.71 := 
by
  sorry

end correlation_coefficient_l291_291068


namespace ratio_of_areas_l291_291206

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l291_291206


namespace cashier_total_value_l291_291462

theorem cashier_total_value (total_bills : ℕ) (ten_bills : ℕ) (twenty_bills : ℕ)
  (h1 : total_bills = 30) (h2 : ten_bills = 27) (h3 : twenty_bills = 3) :
  (10 * ten_bills + 20 * twenty_bills) = 330 :=
by
  sorry

end cashier_total_value_l291_291462


namespace expand_subtract_equals_result_l291_291634

-- Definitions of the given expressions
def expand_and_subtract (x : ℝ) : ℝ :=
  (x + 3) * (2 * x - 5) - (2 * x + 1)

-- Expected result
def expected_result (x : ℝ) : ℝ :=
  2 * x ^ 2 - x - 16

-- The theorem stating the equivalence of the expanded and subtracted expression with the expected result
theorem expand_subtract_equals_result (x : ℝ) : expand_and_subtract x = expected_result x :=
  sorry

end expand_subtract_equals_result_l291_291634


namespace fraction_squared_0_0625_implies_value_l291_291745

theorem fraction_squared_0_0625_implies_value (x : ℝ) (hx : x^2 = 0.0625) : x = 0.25 :=
sorry

end fraction_squared_0_0625_implies_value_l291_291745


namespace fraction_of_number_l291_291130

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l291_291130


namespace proportion_x_l291_291147

theorem proportion_x (x : ℝ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end proportion_x_l291_291147


namespace cyclists_no_point_b_l291_291740

theorem cyclists_no_point_b (v1 v2 t d : ℝ) (h1 : v1 = 35) (h2 : v2 = 25) (h3 : t = 2) (h4 : d = 30) :
  ∀ (ta tb : ℝ), ta + tb = t ∧ ta * v1 + tb * v2 < d → false :=
by
  sorry

end cyclists_no_point_b_l291_291740


namespace find_n_correct_l291_291643

noncomputable def find_n : Prop :=
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * (Real.pi / 180)) = Real.cos (317 * (Real.pi / 180)) → n = 43

theorem find_n_correct : find_n :=
  sorry

end find_n_correct_l291_291643


namespace regression_prediction_l291_291043

theorem regression_prediction
  (slope : ℝ) (centroid_x centroid_y : ℝ) (b : ℝ)
  (h_slope : slope = 1.23)
  (h_centroid : centroid_x = 4 ∧ centroid_y = 5)
  (h_intercept : centroid_y = slope * centroid_x + b)
  (x : ℝ) (h_x : x = 10) :
  centroid_y = 5 →
  slope = 1.23 →
  x = 10 →
  b = 5 - 1.23 * 4 →
  (slope * x + b) = 12.38 :=
by
  intros
  sorry

end regression_prediction_l291_291043


namespace two_pt_seven_five_as_fraction_l291_291434

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l291_291434


namespace second_root_of_system_l291_291292

def system_of_equations (x y : ℝ) : Prop :=
  (2 * x^2 + 3 * x * y + y^2 = 70) ∧ (6 * x^2 + x * y - y^2 = 50)

theorem second_root_of_system :
  system_of_equations 3 4 →
  system_of_equations (-3) (-4) :=
by
  sorry

end second_root_of_system_l291_291292


namespace rectangle_area_increase_l291_291894

theorem rectangle_area_increase :
  let l := 33.333333333333336
  let b := l / 2
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 4
  let A_new := l_new * b_new
  A_new - A_original = 30 :=
by
  sorry

end rectangle_area_increase_l291_291894


namespace p_sufficient_but_not_necessary_for_q_l291_291266

def proposition_p (x : ℝ) := x - 1 = 0
def proposition_q (x : ℝ) := (x - 1) * (x + 2) = 0

theorem p_sufficient_but_not_necessary_for_q :
  ( (∀ x, proposition_p x → proposition_q x) ∧ ¬(∀ x, proposition_p x ↔ proposition_q x) ) := 
by
  sorry

end p_sufficient_but_not_necessary_for_q_l291_291266


namespace mushrooms_left_l291_291034

-- Define the initial amount of mushrooms.
def init_mushrooms : ℕ := 15

-- Define the amount of mushrooms eaten.
def eaten_mushrooms : ℕ := 8

-- Define the resulting amount of mushrooms.
def remaining_mushrooms (init : ℕ) (eaten : ℕ) : ℕ := init - eaten

-- The proof statement
theorem mushrooms_left : remaining_mushrooms init_mushrooms eaten_mushrooms = 7 :=
by
    sorry

end mushrooms_left_l291_291034


namespace total_peaches_l291_291934

variable {n m : ℕ}

-- conditions
def equal_subgroups (n : ℕ) := (n % 3 = 0)

def condition_1 (n m : ℕ) := (m - 27) % n = 0 ∧ (m - 27) / n = 5

def condition_2 (n m : ℕ) : Prop := 
  ∃ x : ℕ, 0 < x ∧ x < 7 ∧ (m - x) % n = 0 ∧ ((m - x) / n = 7) 

-- theorem to be proved
theorem total_peaches (n m : ℕ) (h1 : equal_subgroups n) (h2 : condition_1 n m) (h3 : condition_2 n m) : m = 102 := 
sorry

end total_peaches_l291_291934


namespace find_a4_l291_291528

-- Define the sequence
noncomputable def a : ℕ → ℝ := sorry

-- Define the initial term a1 and common difference d
noncomputable def a1 : ℝ := sorry
noncomputable def d : ℝ := sorry

-- The conditions from the problem
def condition1 : Prop := a 2 + a 6 = 10 * Real.sqrt 3
def condition2 : Prop := a 3 + a 7 = 14 * Real.sqrt 3

-- Using the conditions to prove a4
theorem find_a4 (h1 : condition1) (h2 : condition2) : a 4 = 5 * Real.sqrt 3 :=
by
  sorry

end find_a4_l291_291528


namespace math_problem_l291_291708

variable {x y z : ℝ}

def condition1 (x : ℝ) := x = 1.2 * 40
def condition2 (x y : ℝ) := y = x - 0.35 * x
def condition3 (x y z : ℝ) := z = (x + y) / 2

theorem math_problem (x y z : ℝ) (h1 : condition1 x) (h2 : condition2 x y) (h3 : condition3 x y z) :
  z = 39.6 :=
by
  sorry

end math_problem_l291_291708


namespace least_number_to_add_l291_291893

theorem least_number_to_add (x : ℕ) (h1 : (1789 + x) % 6 = 0) (h2 : (1789 + x) % 4 = 0) (h3 : (1789 + x) % 3 = 0) : x = 7 := 
sorry

end least_number_to_add_l291_291893


namespace athlete_last_finish_l291_291110

theorem athlete_last_finish (v1 v2 v3 : ℝ) (h1 : v1 > v2) (h2 : v2 > v3) :
  let T1 := 1 / v1 + 2 / v2 
  let T2 := 1 / v2 + 2 / v3
  let T3 := 1 / v3 + 2 / v1
  T2 > T1 ∧ T2 > T3 :=
by
  sorry

end athlete_last_finish_l291_291110


namespace sufficient_but_not_necessary_condition_l291_291515

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) ↔ (a ≥ 5) :=
by
  sorry

end sufficient_but_not_necessary_condition_l291_291515


namespace relationship_abc_l291_291520

noncomputable def a := (1 / 3 : ℝ) ^ (2 / 3)
noncomputable def b := (2 / 3 : ℝ) ^ (1 / 3)
noncomputable def c := Real.logb (1/2) (1/3)

theorem relationship_abc : c > b ∧ b > a :=
by
  sorry

end relationship_abc_l291_291520


namespace intersecting_circle_radius_l291_291431

-- Definitions representing the conditions
def non_intersecting_circles (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) : Prop :=
  ∀ i j, i ≠ j → dist (O_i i) (O_i j) ≥ r_i i + r_i j

def min_radius_one (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) := 
  ∀ i, r_i i ≥ 1

-- The main theorem stating the proof goal
theorem intersecting_circle_radius 
  (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) (O : ℕ) (r : ℝ)
  (h_non_intersecting : non_intersecting_circles O_i r_i)
  (h_min_radius : min_radius_one O_i r_i)
  (h_intersecting : ∀ i, dist O (O_i i) ≤ r + r_i i) :
  r ≥ 1 := 
sorry

end intersecting_circle_radius_l291_291431


namespace sufficient_not_necessary_condition_l291_291546

variable (a b c : ℝ)

-- Define the condition that the sequence forms a geometric sequence
def geometric_sequence (a1 a2 a3 a4 a5 : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ a1 * q = a2 ∧ a2 * q = a3 ∧ a3 * q = a4 ∧ a4 * q = a5

-- Lean statement proving the problem
theorem sufficient_not_necessary_condition :
  (geometric_sequence 1 a b c 16) → (b = 4) ∧ ¬ (b = 4 → geometric_sequence 1 a b c 16) :=
sorry

end sufficient_not_necessary_condition_l291_291546


namespace correct_average_l291_291573

theorem correct_average :
  let avg_incorrect := 15
  let num_numbers := 20
  let read_incorrect1 := 42
  let read_correct1 := 52
  let read_incorrect2 := 68
  let read_correct2 := 78
  let read_incorrect3 := 85
  let read_correct3 := 95
  let incorrect_sum := avg_incorrect * num_numbers
  let diff1 := read_correct1 - read_incorrect1
  let diff2 := read_correct2 - read_incorrect2
  let diff3 := read_correct3 - read_incorrect3
  let total_diff := diff1 + diff2 + diff3
  let correct_sum := incorrect_sum + total_diff
  let correct_avg := correct_sum / num_numbers
  correct_avg = 16.5 :=
by
  sorry

end correct_average_l291_291573


namespace intersection_with_unit_circle_l291_291524

theorem intersection_with_unit_circle (α : ℝ) : 
    let x := Real.cos (α - Real.pi / 2)
    let y := Real.sin (α - Real.pi / 2)
    (x, y) = (Real.sin α, -Real.cos α) :=
by
  sorry

end intersection_with_unit_circle_l291_291524


namespace exists_fraction_bound_infinite_no_fraction_bound_l291_291607

-- Problem 1: Statement 1
theorem exists_fraction_bound (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 0 < b ∧ (b : ℝ) ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

-- Problem 2: Statement 2
theorem infinite_no_fraction_bound :
  ∃ᶠ n : ℕ in Filter.atTop, ¬ ∃ (a b : ℤ), 0 < b ∧ (b : ℝ) ≤ Real.sqrt n ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

end exists_fraction_bound_infinite_no_fraction_bound_l291_291607


namespace john_dimes_l291_291230

theorem john_dimes :
  ∀ (d : ℕ), 
  (4 * 25 + d * 10 + 5) = 135 → (5: ℕ) + (d: ℕ) * 10 + 4 = 4 + 131 + 3*d → d = 3 :=
by
  sorry

end john_dimes_l291_291230


namespace scalene_polygon_exists_l291_291883

theorem scalene_polygon_exists (n: ℕ) (a: Fin n → ℝ) (h: ∀ i, 1 ≤ a i ∧ a i ≤ 2013) (h_geq: n ≥ 13):
  ∃ (A B C : Fin n), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ a A + a B > a C ∧ a A + a C > a B ∧ a B + a C > a A :=
sorry

end scalene_polygon_exists_l291_291883


namespace probability_of_different_colors_l291_291909

theorem probability_of_different_colors :
  let total_chips := 12
  let prob_blue_then_yellow_red := ((6 / total_chips) * ((4 + 2) / total_chips))
  let prob_yellow_then_blue_red := ((4 / total_chips) * ((6 + 2) / total_chips))
  let prob_red_then_blue_yellow := ((2 / total_chips) * ((6 + 4) / total_chips))
  prob_blue_then_yellow_red + prob_yellow_then_blue_red + prob_red_then_blue_yellow = 11 / 18 := by
    sorry

end probability_of_different_colors_l291_291909


namespace simplify_expression_l291_291410

noncomputable def original_expression (x : ℝ) : ℝ :=
(x - 3 * x / (x + 1)) / ((x - 2) / (x^2 + 2 * x + 1))

theorem simplify_expression:
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 ∧ x ≠ -1 ∧ x ≠ 2 → 
  (original_expression x = x^2 + x) ∧ 
  ((x = 1 → original_expression x = 2) ∧ (x = 0 → original_expression x = 0)) :=
by
  intros
  sorry

end simplify_expression_l291_291410


namespace type_R_completion_time_l291_291000

theorem type_R_completion_time :
  (∃ R : ℝ, (2 / R + 3 / 7 = 1 / 1.2068965517241381) ∧ abs (R - 5) < 0.01) :=
  sorry

end type_R_completion_time_l291_291000


namespace sector_area_l291_291897

theorem sector_area (r : ℝ) : (2 * r + 2 * r = 16) → (1/2 * r^2 * 2 = 16) :=
by
  intro h1
  sorry

end sector_area_l291_291897


namespace log_power_relationship_l291_291370

theorem log_power_relationship (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (hm : m = Real.log c / Real.log a)
  (hn : n = Real.log c / Real.log b)
  (hr : r = a^c) :
  r > m ∧ m > n :=
sorry

end log_power_relationship_l291_291370


namespace value_of_x_plus_y_squared_l291_291554

variable (x y : ℝ)

def condition1 : Prop := x * (x + y) = 40
def condition2 : Prop := y * (x + y) = 90
def condition3 : Prop := x - y = 5

theorem value_of_x_plus_y_squared (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 x y) : (x + y) ^ 2 = 130 :=
by
  sorry

end value_of_x_plus_y_squared_l291_291554


namespace bank_account_balance_l291_291337

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end bank_account_balance_l291_291337


namespace ratio_of_areas_of_circles_l291_291212

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l291_291212


namespace robin_gum_packages_l291_291888

theorem robin_gum_packages (P : ℕ) (h1 : 7 * P + 6 = 41) : P = 5 :=
by
  sorry

end robin_gum_packages_l291_291888


namespace Hay_s_Linens_sales_l291_291834

theorem Hay_s_Linens_sales :
  ∃ (n : ℕ), 500 ≤ 52 * n ∧ 52 * n ≤ 700 ∧
             ∀ m, (500 ≤ 52 * m ∧ 52 * m ≤ 700) → n ≤ m :=
sorry

end Hay_s_Linens_sales_l291_291834


namespace finite_integer_solutions_l291_291566

theorem finite_integer_solutions (n : ℕ) : 
  ∃ (S : Finset (ℤ × ℤ)), ∀ (x y : ℤ), (x^3 + y^3 = n) → (x, y) ∈ S := 
sorry

end finite_integer_solutions_l291_291566


namespace BoatCrafters_total_canoes_l291_291482

def canoe_production (n : ℕ) : ℕ :=
  if n = 0 then 5 else 3 * canoe_production (n-1) - 1

theorem BoatCrafters_total_canoes : 
  (canoe_production 0 - 1) + (canoe_production 1 - 1) + (canoe_production 2 - 1) + (canoe_production 3 - 1) = 196 := 
by
  sorry

end BoatCrafters_total_canoes_l291_291482


namespace no_parallelepiped_exists_l291_291188

theorem no_parallelepiped_exists 
  (xyz_half_volume: ℝ)
  (xy_plus_yz_plus_zx_half_surface_area: ℝ) 
  (sum_of_squares_eq_4: ℝ) : 
  ¬(∃ x y z : ℝ, (x * y * z = xyz_half_volume) ∧ 
                 (x * y + y * z + z * x = xy_plus_yz_plus_zx_half_surface_area) ∧ 
                 (x^2 + y^2 + z^2 = sum_of_squares_eq_4)) := 
by
  let xyz_half_volume := 2 * Real.pi / 3
  let xy_plus_yz_plus_zx_half_surface_area := Real.pi
  let sum_of_squares_eq_4 := 4
  sorry

end no_parallelepiped_exists_l291_291188


namespace find_valid_pairs_l291_291987

-- Defining the conditions and target answer set.
def valid_pairs : List (Nat × Nat) := [(2,2), (3,3), (1,2), (2,1), (2,3), (3,2)]

theorem find_valid_pairs (a b : Nat) :
  (∃ n m : Int, (a^2 + b = n * (b^2 - a)) ∧ (b^2 + a = m * (a^2 - b)))
  ↔ (a, b) ∈ valid_pairs :=
by sorry

end find_valid_pairs_l291_291987


namespace donovan_correct_answers_l291_291804

variable (C : ℝ)
variable (incorrectAnswers : ℝ := 13)
variable (percentageCorrect : ℝ := 0.7292)

theorem donovan_correct_answers :
  (C / (C + incorrectAnswers)) = percentageCorrect → C = 35 := by
  sorry

end donovan_correct_answers_l291_291804


namespace download_time_is_2_hours_l291_291977

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end download_time_is_2_hours_l291_291977


namespace fish_lives_longer_than_dog_l291_291247

-- Definitions based on conditions
def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := 12

-- Theorem stating the desired proof
theorem fish_lives_longer_than_dog :
  fish_lifespan - dog_lifespan = 2 := 
sorry

end fish_lives_longer_than_dog_l291_291247


namespace proof_problem_l291_291196

variable {α : Type*} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ (a1 d : α), ∀ n : ℕ, a n = a1 + n * d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → α}

theorem proof_problem (h_arith_seq : is_arithmetic_sequence a)
    (h_S6_gt_S7 : sum_first_n_terms a 6 > sum_first_n_terms a 7)
    (h_S7_gt_S5 : sum_first_n_terms a 7 > sum_first_n_terms a 5) :
    (∃ d : α, d < 0) ∧ (∃ S11 : α, sum_first_n_terms a 11 > 0) :=
  sorry

end proof_problem_l291_291196


namespace maximize_integral_k_l291_291544

theorem maximize_integral_k (f : ℝ → ℝ) (k : ℝ) 
  (h_cont : Continuous f)
  (h_eq : ∀ x, f x = 1 + k * ∫ t in -π/2 .. π/2, f t * Real.sin (x - t)) :
  k = 2 / π :=
sorry

end maximize_integral_k_l291_291544


namespace decimal_to_fraction_l291_291436

-- Define the decimal number 2.75
def decimal_num : ℝ := 2.75

-- Define the expected fraction in unsimplified form
def unsimplified_fraction := 275 / 100

-- The greatest common divisor of 275 and 100
def gcd_275_100 : ℕ := 25

-- Define the simplified fraction as 11/4
def simplified_fraction := 11 / 4

-- Statement of the theorem to prove
theorem decimal_to_fraction : (decimal_num : ℚ) = simplified_fraction :=
by
  -- Here you can write the proof steps or use sorry to denote the proof is omitted
  sorry

end decimal_to_fraction_l291_291436


namespace line_intersects_y_axis_at_5_l291_291627

theorem line_intersects_y_axis_at_5 :
  ∃ (b : ℝ), ∀ (x y : ℝ), (x - 2 = 0 ∧ y - 9 = 0) ∨ (x - 4 = 0 ∧ y - 13 = 0) →
  (y = 2 * x + b) ∧ (b = 5) :=
by
  sorry

end line_intersects_y_axis_at_5_l291_291627


namespace stream_speed_l291_291597

variable (v : ℝ)

def effective_speed_downstream (v : ℝ) : ℝ := 7.5 + v
def effective_speed_upstream (v : ℝ) : ℝ := 7.5 - v 

theorem stream_speed : (7.5 - v) / (7.5 + v) = 1 / 2 → v = 2.5 :=
by
  intro h
  -- Proof will be resolved here
  sorry

end stream_speed_l291_291597


namespace budget_for_equipment_l291_291464

theorem budget_for_equipment 
    (transportation_p : ℝ := 20)
    (r_d_p : ℝ := 9)
    (utilities_p : ℝ := 5)
    (supplies_p : ℝ := 2)
    (salaries_degrees : ℝ := 216)
    (total_degrees : ℝ := 360)
    (total_budget : ℝ := 100)
    :
    (total_budget - (transportation_p + r_d_p + utilities_p + supplies_p +
    (salaries_degrees / total_degrees * total_budget))) = 4 := 
sorry

end budget_for_equipment_l291_291464


namespace calculation_equivalence_l291_291326

theorem calculation_equivalence : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := 
by
  sorry

end calculation_equivalence_l291_291326


namespace equal_area_division_l291_291525

open_locale classical

variables {A B C D X Y V Z S T P : Type*}
variables [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ D] 
variables [affine_space ℝ X] [affine_space ℝ Y] [affine_space ℝ V] [affine_space ℝ Z]
variables [affine_space ℝ S] [affine_space ℝ T] [affine_space ℝ P]
variables [convex_quadrilateral ABCD: Type*]

/-- A theorem that states: In a convex quadrilateral, the segments connecting the intersection
points of these parallel lines with the midpoints of the sides divide the quadrilateral into equal-area parts. --/
theorem equal_area_division (h_mid_AC : S = midpoint ℝ A C)
                           (h_mid_BD : T = midpoint ℝ B D)
                           (h_mid_AB : X = midpoint ℝ A B)
                           (h_mid_BC : Y = midpoint ℝ B C)
                           (h_mid_CD : V = midpoint ℝ C D)
                           (h_mid_DA : Z = midpoint ℝ D A)
                           (h_parallel_ST_AC : line_parallel S T (A - C))
                           (h_parallel_TD_BD : line_parallel T D (B - C))
                           (h_quad_convex : convex_quadrilateral ABCD) :
                             area (triangle A X P) = (1/4) * (area (quadrilateral A B C D))
                             ∧ area (triangle B Y P) = (1/4) * (area (quadrilateral A B C D))
                             ∧ area (triangle C V P) = (1/4) * (area (quadrilateral A B C D))
                             ∧ area (triangle D Z P) = (1/4) * (area (quadrilateral A B C D)) :=
begin
  sorry -- proof goes here
end

end equal_area_division_l291_291525


namespace circle_area_ratio_l291_291214

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l291_291214


namespace boat_travel_time_downstream_l291_291461

theorem boat_travel_time_downstream
  (v c: ℝ)
  (h1: c = 1)
  (h2: 24 / (v - c) = 6): 
  24 / (v + c) = 4 := 
by
  sorry

end boat_travel_time_downstream_l291_291461


namespace xiao_ming_returns_and_distance_is_correct_l291_291681

theorem xiao_ming_returns_and_distance_is_correct :
  ∀ (walk_distance : ℝ) (turn_angle : ℝ), 
  walk_distance = 5 ∧ turn_angle = 20 → 
  (∃ n : ℕ, (360 % turn_angle = 0) ∧ n = 360 / turn_angle ∧ walk_distance * n = 90) :=
by
  sorry

end xiao_ming_returns_and_distance_is_correct_l291_291681


namespace part1_part2_l291_291703

-- Define the arithmetic sequence
variable (a : ℕ → ℝ) (d : ℝ)
variable h_d : d > 1 -- d is the common difference and greater than 1

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (n^2 + n) / a n

-- Define the sums S_n and T_n
def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))
def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k, b (k + 1))

-- Given conditions
variable h1 : 3 * a 2 = 3 * a 1 + a 3
variable h2 : S 3 + T 3 = 21

-- Given in Part 2
variable h3 : ∀ (n m : ℕ), b n - b m = (n - m) * d -- b_n is arithmetic
variable h4 : S 99 - T 99 = 99

theorem part1 : ∀ n, a n = 3 * n :=
by
  sorry

theorem part2 : d = 51/50 :=
by
  sorry

end part1_part2_l291_291703


namespace graduating_class_total_l291_291614

theorem graduating_class_total (boys girls : ℕ) 
  (h_boys : boys = 138)
  (h_more_girls : girls = boys + 69) :
  boys + girls = 345 :=
sorry

end graduating_class_total_l291_291614


namespace fraction_of_number_l291_291124

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l291_291124


namespace division_quotient_l291_291886

theorem division_quotient (dividend divisor remainder quotient : ℕ)
  (H1 : dividend = 190)
  (H2 : divisor = 21)
  (H3 : remainder = 1)
  (H4 : dividend = divisor * quotient + remainder) : quotient = 9 :=
by {
  sorry
}

end division_quotient_l291_291886


namespace earnings_from_jam_l291_291320

def betty_strawberries : ℕ := 16
def matthew_additional_strawberries : ℕ := 20
def jar_strawberries : ℕ := 7
def jar_price : ℕ := 4

theorem earnings_from_jam :
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  total_money = 40 :=
by
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  show total_money = 40
  sorry

end earnings_from_jam_l291_291320


namespace minimum_value_f_l291_291550

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ t ≥ 0, ∀ (x y : ℝ), 0 < x → 0 < y → f x y ≥ t ∧ t = 4 * Real.sqrt 2 :=
sorry

end minimum_value_f_l291_291550


namespace maximum_expression_value_l291_291269

theorem maximum_expression_value (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (sqrt (sqrt (sqrt 3)) * (a * (b + 2 * c)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (b * (c + 2 * d)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (c * (d + 2 * a)) ^ (1 / 4) +
   sqrt (sqrt (sqrt 3)) * (d * (a + 2 * b)) ^ (1 / 4)) ≤ 4 * sqrt (sqrt 3) :=
sorry

end maximum_expression_value_l291_291269


namespace fraction_of_number_l291_291137

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l291_291137


namespace mean_median_difference_is_minus_4_l291_291059

-- Defining the percentages of students scoring specific points
def perc_60 : ℝ := 0.20
def perc_75 : ℝ := 0.55
def perc_95 : ℝ := 0.10
def perc_110 : ℝ := 1 - (perc_60 + perc_75 + perc_95) -- 0.15

-- Defining the scores
def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_95 : ℝ := 95
def score_110 : ℝ := 110

-- Calculating the mean score
def mean_score : ℝ := (perc_60 * score_60) + (perc_75 * score_75) + (perc_95 * score_95) + (perc_110 * score_110)

-- Given the median score
def median_score : ℝ := score_75

-- Defining the expected difference
def expected_difference : ℝ := mean_score - median_score

theorem mean_median_difference_is_minus_4 :
  expected_difference = -4 := by sorry

end mean_median_difference_is_minus_4_l291_291059


namespace fraction_div_add_result_l291_291371

theorem fraction_div_add_result : 
  (2 / 3) / (4 / 5) + (1 / 2) = (4 / 3) := 
by 
  sorry

end fraction_div_add_result_l291_291371


namespace verify_base_case_l291_291742

theorem verify_base_case : 1 + (1 / 2) + (1 / 3) < 2 :=
sorry

end verify_base_case_l291_291742


namespace yella_computer_usage_difference_l291_291595

-- Define the given conditions
def last_week_usage : ℕ := 91
def this_week_daily_usage : ℕ := 8
def days_in_week : ℕ := 7

-- Compute this week's total usage
def this_week_total_usage := this_week_daily_usage * days_in_week

-- Statement to prove
theorem yella_computer_usage_difference :
  last_week_usage - this_week_total_usage = 35 := 
by
  -- The proof will be filled in here
  sorry

end yella_computer_usage_difference_l291_291595


namespace ratio_of_areas_eq_l291_291209

-- Define the conditions
variables {C D : Type} [circle C] [circle D]
variables (R_C R_D : ℝ)
variable (L : ℝ)

-- Given conditions
axiom arc_length_eq : (60 / 360) * (2 * π * R_C) = L
axiom arc_length_eq' : (40 / 360) * (2 * π * R_D) = L

-- Statement to prove
theorem ratio_of_areas_eq : (π * R_C^2) / (π * R_D^2) = 4 / 9 :=
sorry

end ratio_of_areas_eq_l291_291209


namespace final_expression_simplified_l291_291837

variable (b : ℝ)

theorem final_expression_simplified :
  ((3 * b + 6 - 5 * b) / 3) = (-2 / 3) * b + 2 := by
  sorry

end final_expression_simplified_l291_291837


namespace platform_length_is_150_l291_291157

noncomputable def length_of_platform
  (train_length : ℝ)
  (time_to_cross_platform : ℝ)
  (time_to_cross_pole : ℝ)
  (L : ℝ) : Prop :=
  train_length + L = (train_length / time_to_cross_pole) * time_to_cross_platform

theorem platform_length_is_150 :
  length_of_platform 300 27 18 150 :=
by 
  -- Proof omitted, but the statement is ready for proving
  sorry

end platform_length_is_150_l291_291157


namespace function_above_x_axis_l291_291365

noncomputable def quadratic_function (a x : ℝ) := (a^2 - 3 * a + 2) * x^2 + (a - 1) * x + 2

theorem function_above_x_axis (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x > 0) ↔ (a > 15 / 7 ∨ a ≤ 1) :=
by {
  sorry
}

end function_above_x_axis_l291_291365


namespace find_angle_FYD_l291_291065

noncomputable def angle_FYD (AB CD AXF FYG : ℝ) : ℝ := 180 - AXF

theorem find_angle_FYD (AB CD : ℝ) (AXF : ℝ) (FYG : ℝ) (h1 : AB = CD) (h2 : AXF = 125) (h3 : FYG = 40) :
  angle_FYD AB CD AXF FYG = 55 :=
by
  sorry

end find_angle_FYD_l291_291065


namespace simplify_expression_l291_291591

theorem simplify_expression : 4 * (8 - 2 + 3) - 7 = 29 := 
by {
  sorry
}

end simplify_expression_l291_291591


namespace searchlight_revolutions_l291_291312

theorem searchlight_revolutions (p : ℝ) (r : ℝ) (t : ℝ) 
  (h1 : p = 0.6666666666666667) 
  (h2 : t = 10) 
  (h3 : p = (60 / r - t) / (60 / r)) : 
  r = 2 :=
by sorry

end searchlight_revolutions_l291_291312


namespace max_abc_l291_291193

def A_n (a : ℕ) (n : ℕ) : ℕ := a * (10^(3*n) - 1) / 9
def B_n (b : ℕ) (n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9
def C_n (c : ℕ) (n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

theorem max_abc (a b c n : ℕ) (hpos : n > 0) (h1 : 1 ≤ a ∧ a < 10) (h2 : 1 ≤ b ∧ b < 10) (h3 : 1 ≤ c ∧ c < 10) (h_eq : C_n c n - B_n b n = A_n a n ^ 2) :  a + b + c ≤ 18 :=
by sorry

end max_abc_l291_291193


namespace rectangle_length_eq_15_l291_291168

theorem rectangle_length_eq_15 (w l s p_rect p_square : ℝ)
    (h_w : w = 9)
    (h_s : s = 12)
    (h_p_square : p_square = 4 * s)
    (h_p_rect : p_rect = 2 * w + 2 * l)
    (h_eq_perimeters : p_square = p_rect) : l = 15 := by
  sorry

end rectangle_length_eq_15_l291_291168


namespace number_of_participants_l291_291251

theorem number_of_participants (n : ℕ) (h : n - 1 = 25) : n = 26 := 
by sorry

end number_of_participants_l291_291251


namespace sum_of_roots_of_quadratic_l291_291106

variables {b x₁ x₂ : ℝ}

theorem sum_of_roots_of_quadratic (h : x₁^2 - 2 * x₁ + b = 0) (h' : x₂^2 - 2 * x₂ + b = 0) :
    x₁ + x₂ = 2 :=
sorry

end sum_of_roots_of_quadratic_l291_291106


namespace grove_tree_selection_l291_291860

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l291_291860


namespace minimum_trees_with_at_least_three_types_l291_291857

theorem minimum_trees_with_at_least_three_types 
    (total_trees : ℕ)
    (birches spruces pines aspens : ℕ)
    (h_total : total_trees = 100)
    (h_any_85 : ∀ (S : Finset ℕ), S.card = 85 → 
                  (∃ (b s p a : ℕ), b ∈ S ∧ s ∈ S ∧ p ∈ S ∧ a ∈ S)) :
  ∃ (n : ℕ), n = 69 ∧ ∀ (T : Finset ℕ), T.card = n → 
                  ∃ (b s p : ℕ), b ∈ T ∧ s ∈ T ∧ p ∈ T :=
  sorry

end minimum_trees_with_at_least_three_types_l291_291857


namespace smallest_possible_N_l291_291956

theorem smallest_possible_N (l m n : ℕ) (h_visible : (l - 1) * (m - 1) * (n - 1) = 252) : l * m * n = 392 :=
sorry

end smallest_possible_N_l291_291956


namespace magic_triangle_largest_S_l291_291224

theorem magic_triangle_largest_S :
  ∃ (S : ℕ) (a b c d e f g : ℕ),
    (10 ≤ a) ∧ (a ≤ 16) ∧
    (10 ≤ b) ∧ (b ≤ 16) ∧
    (10 ≤ c) ∧ (c ≤ 16) ∧
    (10 ≤ d) ∧ (d ≤ 16) ∧
    (10 ≤ e) ∧ (e ≤ 16) ∧
    (10 ≤ f) ∧ (f ≤ 16) ∧
    (10 ≤ g) ∧ (g ≤ 16) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧ (a ≠ f) ∧ (a ≠ g) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧ (b ≠ f) ∧ (b ≠ g) ∧
    (c ≠ d) ∧ (c ≠ e) ∧ (c ≠ f) ∧ (c ≠ g) ∧
    (d ≠ e) ∧ (d ≠ f) ∧ (d ≠ g) ∧
    (e ≠ f) ∧ (e ≠ g) ∧
    (f ≠ g) ∧
    (S = a + b + c) ∧
    (S = c + d + e) ∧
    (S = e + f + a) ∧
    (S = g + b + c) ∧
    (S = g + d + e) ∧
    (S = g + f + a) ∧
    ((a + b + c) + (c + d + e) + (e + f + a) = 91 - g) ∧
    (S = 26) := sorry

end magic_triangle_largest_S_l291_291224


namespace desired_on_time_departure_rate_l291_291258

theorem desired_on_time_departure_rate :
  let first_late := 1
  let on_time_flights_next := 3
  let additional_on_time_flights := 2
  let total_on_time_flights := on_time_flights_next + additional_on_time_flights
  let total_flights := first_late + on_time_flights_next + additional_on_time_flights
  let on_time_departure_rate := (total_on_time_flights : ℚ) / (total_flights : ℚ) * 100
  on_time_departure_rate > 83.33 :=
by
  sorry

end desired_on_time_departure_rate_l291_291258


namespace selling_price_correct_l291_291616

noncomputable def selling_price (purchase_price : ℝ) (overhead_expenses : ℝ) (profit_percent : ℝ) : ℝ :=
  let total_cost_price := purchase_price + overhead_expenses
  let profit := (profit_percent / 100) * total_cost_price
  total_cost_price + profit

theorem selling_price_correct :
    selling_price 225 28 18.577075098814234 = 300 := by
  sorry

end selling_price_correct_l291_291616


namespace concurrency_of_lines_l291_291721

open EuclideanGeometry

noncomputable def quadrilateral_cyclic (A B C D : Point) (O : Circle) : Prop := 
  Circle.inscribed A B C D O

noncomputable def circumcenter (A B P : Point) : Point := 
  classical.some (exists_circumcenter A B P)

theorem concurrency_of_lines 
  (A B C D P O : Point)
  (O₁ O₂ O₃ O₄ : Point)
  (h_cyclic : quadrilateral_cyclic A B C D O)
  (h_intersect: Line.inter AC BD P)
  (h_O1 : circumcenter A B P = O₁)
  (h_O2 : circumcenter B C P = O₂)
  (h_O3 : circumcenter C D P = O₃)
  (h_O4 : circumcenter D A P = O₄)
  : concurrency (Line.segment O P) (Line.segment O₁ O₃ | O₂ O₄) :=
sorry

end concurrency_of_lines_l291_291721


namespace ratio_students_l291_291717

theorem ratio_students
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (h_finley : finley_students = 24)
  (h_johnson : johnson_students = 22)
  : (johnson_students : ℚ) / ((finley_students / 2 : ℕ) : ℚ) = 11 / 6 :=
by
  sorry

end ratio_students_l291_291717


namespace fraction_of_number_l291_291135

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l291_291135


namespace subtract_30_divisible_l291_291458

theorem subtract_30_divisible (n : ℕ) (d : ℕ) (r : ℕ) 
  (h1 : n = 13602) (h2 : d = 87) (h3 : r = 30) 
  (h4 : n % d = r) : (n - r) % d = 0 :=
by
  -- Skipping the proof as it's not required
  sorry

end subtract_30_divisible_l291_291458


namespace B_completes_work_in_n_days_l291_291927

-- Define the conditions
def can_complete_work_A_in_d_days (d : ℕ) : Prop := d = 15
def fraction_of_work_left_after_working_together (t : ℕ) (fraction : ℝ) : Prop :=
  t = 5 ∧ fraction = 0.41666666666666663

-- Define the theorem to be proven
theorem B_completes_work_in_n_days (d t : ℕ) (fraction : ℝ) (x : ℕ) 
  (hA : can_complete_work_A_in_d_days d) 
  (hB : fraction_of_work_left_after_working_together t fraction) : x = 20 :=
sorry

end B_completes_work_in_n_days_l291_291927


namespace value_of_b_l291_291232

theorem value_of_b (a b : ℝ) (h1 : 4 * a^2 + 1 = 1) (h2 : b - a = 3) : b = 3 :=
sorry

end value_of_b_l291_291232


namespace vector_condition_l291_291046

open Real

def acute_angle (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.1 + a.2 * b.2) > 0

def not_collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 ≠ 0

theorem vector_condition (x : ℝ) :
  acute_angle (2, x + 1) (x + 2, 6) ∧ not_collinear (2, x + 1) (x + 2, 6) ↔ x > -5/4 ∧ x ≠ 2 :=
by
  sorry

end vector_condition_l291_291046


namespace card_dealing_probability_l291_291111

noncomputable def probability_ace_then_ten_then_jack : ℚ :=
  let prob_ace := 4 / 52
  let prob_ten := 4 / 51
  let prob_jack := 4 / 50
  prob_ace * prob_ten * prob_jack

theorem card_dealing_probability :
  probability_ace_then_ten_then_jack = 16 / 33150 := by
  sorry

end card_dealing_probability_l291_291111


namespace total_dollars_l291_291070

theorem total_dollars (john emma lucas : ℝ) 
  (h_john : john = 4 / 5) 
  (h_emma : emma = 2 / 5) 
  (h_lucas : lucas = 1 / 2) : 
  john + emma + lucas = 1.7 := by
  sorry

end total_dollars_l291_291070


namespace smallest_trees_in_three_types_l291_291862

def grove (birches spruces pines aspens total : Nat): Prop :=
  birches + spruces + pines + aspens = total ∧
  (∀ (subset : Finset Nat), subset.card = 85 → (∃ a b c d, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ d ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a))

theorem smallest_trees_in_three_types (birches spruces pines aspens : Nat) (h : grove birches spruces pines aspens 100) :
  ∃ t, t = 69 ∧ (∀ (subset : Finset Nat), subset.card = t → (∃ a b c, a ∈ subset ∧ b ∈ subset ∧ c ∈ subset ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a)) :=
sorry

end smallest_trees_in_three_types_l291_291862


namespace triangle_XYZ_XY2_XZ2_difference_l291_291396

-- Define the problem parameters and conditions
def YZ : ℝ := 10
def XM : ℝ := 6
def midpoint_YZ (M : ℝ) := 2 * M = YZ

-- The main theorem to be proved
theorem triangle_XYZ_XY2_XZ2_difference :
  ∀ (XY XZ : ℝ), 
  (∀ (M : ℝ), midpoint_YZ M) →
  ((∃ (x : ℝ), (0 ≤ x ∧ x ≤ 10) ∧ XY^2 + XZ^2 = 2 * x^2 - 20 * x + 2 * (11 * x - x^2 - 11) + 100)) →
  (120 - 100 = 20) :=
by
  sorry

end triangle_XYZ_XY2_XZ2_difference_l291_291396


namespace total_cost_six_years_l291_291002

variable {fees : ℕ → ℝ}

-- Conditions
def fee_first_year : fees 1 = 80 := sorry

def fee_increase (n : ℕ) : fees (n + 1) = fees n + (10 + 2 * (n - 1)) := 
sorry

-- Proof problem: Prove that the total cost is 670
theorem total_cost_six_years : (fees 1 + fees 2 + fees 3 + fees 4 + fees 5 + fees 6) = 670 :=
by sorry

end total_cost_six_years_l291_291002


namespace spadesuit_proof_l291_291333

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem spadesuit_proof : 
  spadesuit (spadesuit 5 2) (spadesuit 9 (spadesuit 3 6)) = 3 :=
by
  sorry

end spadesuit_proof_l291_291333


namespace blue_water_bottles_initial_count_l291_291426

theorem blue_water_bottles_initial_count
    (red : ℕ) (black : ℕ) (taken_out : ℕ) (left : ℕ) (initial_blue : ℕ) :
    red = 2 →
    black = 3 →
    taken_out = 5 →
    left = 4 →
    initial_blue + red + black = taken_out + left →
    initial_blue = 4 := by
  intros
  sorry

end blue_water_bottles_initial_count_l291_291426


namespace yunas_math_score_l291_291922

theorem yunas_math_score (K E M : ℕ) 
  (h1 : (K + E) / 2 = 92) 
  (h2 : (K + E + M) / 3 = 94) : 
  M = 98 :=
sorry

end yunas_math_score_l291_291922


namespace fraction_multiplication_l291_291119

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l291_291119


namespace linear_coefficient_l291_291531

theorem linear_coefficient (a b c : ℤ) (h : a = 1 ∧ b = -2 ∧ c = -1) :
    b = -2 := 
by
  -- Use the given hypothesis directly
  exact h.2.1

end linear_coefficient_l291_291531


namespace triangle_side_length_sum_l291_291175

theorem triangle_side_length_sum :
  ∃ (a b c : ℕ), (5: ℝ) ^ 2 + (7: ℝ) ^ 2 - 2 * (5: ℝ) * (7: ℝ) * (Real.cos (Real.pi * 80 / 180)) = (a: ℝ) + Real.sqrt b + Real.sqrt c ∧
  b = 62 ∧ c = 0 :=
sorry

end triangle_side_length_sum_l291_291175


namespace sum_of_coordinates_l291_291720

-- Define the points C and D and the conditions
def point_C : ℝ × ℝ := (0, 0)

def point_D (x : ℝ) : ℝ × ℝ := (x, 5)

def slope_CD (x : ℝ) : Prop :=
  (5 - 0) / (x - 0) = 3 / 4

-- The required theorem to be proved
theorem sum_of_coordinates (D : ℝ × ℝ)
  (hD : D.snd = 5)
  (h_slope : slope_CD D.fst) :
  D.fst + D.snd = 35 / 3 :=
sorry

end sum_of_coordinates_l291_291720


namespace find_T_l291_291241

theorem find_T (T : ℝ) 
  (h : (1/3) * (1/8) * T = (1/4) * (1/6) * 150) : 
  T = 150 :=
sorry

end find_T_l291_291241


namespace probability_two_english_teachers_l291_291734

open Finset

theorem probability_two_english_teachers 
  (english teachers : Finset ℕ) 
  (h_teacher_card : teachers.card = 9) 
  (h_english_card : english.card = 3)
  (h_subset: english ⊆ teachers) :
  (english.choose 2).card.toNat.toRat / (teachers.choose 2).card.toNat.toRat = 1 / 12 := 
  sorry

end probability_two_english_teachers_l291_291734


namespace isosceles_right_triangle_hypotenuse_l291_291004

theorem isosceles_right_triangle_hypotenuse (a : ℝ) (h : ℝ) (hyp : a = 30 ∧ h^2 = a^2 + a^2) : h = 30 * Real.sqrt 2 :=
sorry

end isosceles_right_triangle_hypotenuse_l291_291004


namespace difference_of_squares_expression_l291_291144

theorem difference_of_squares_expression
  (x y : ℝ) :
  (x + 2 * y) * (x - 2 * y) = x^2 - (2 * y)^2 :=
by sorry

end difference_of_squares_expression_l291_291144


namespace problem_statement_l291_291828

noncomputable def ellipse_equation (t : ℝ) (ht : t > 0) : String :=
  if h : t = 2 then "x^2/9 + y^2/2 = 1"
  else "invalid equation"

theorem problem_statement (m : ℝ) (t : ℝ) (ht : t > 0) (ha : t = 2) 
  (A E F B : ℝ × ℝ) (hA : A = (-3, 0)) (hB : B = (1, 0))
  (hl : ∀ x y, x = m * y + 1) (area : ℝ) (har : area = 16/3) :
  ((ellipse_equation t ht) = "x^2/9 + y^2/2 = 1") ∧
  (∃ M N : ℝ × ℝ, 
    (M.1 = 3 ∧ N.1 = 3) ∧
    ((M.1 - B.1) * (N.1 - B.1) + (M.2 - B.2) * (N.2 - B.2) = 0)) := 
sorry

end problem_statement_l291_291828


namespace true_propositions_l291_291316

theorem true_propositions :
  (∀ x y, (x * y = 1 → x * y = (x * y))) ∧
  (¬ (∀ (a b : ℝ), (∀ (A B : ℝ), a = b → A = B) ∧ (A = B → a ≠ b))) ∧
  (∀ m : ℝ, (m ≤ 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0)) ↔
    (true ∧ true ∧ true) :=
by sorry

end true_propositions_l291_291316


namespace cos_equality_l291_291641

noncomputable def degrees_to_radians (d : ℝ) : ℝ := d * (Real.pi / 180)

theorem cos_equality : ∃ n : ℝ, (0 ≤ n ∧ n ≤ 180) ∧ Real.cos (degrees_to_radians n) = Real.cos (degrees_to_radians 317) :=
by
  use 43
  simp [degrees_to_radians, Real.cos]
  sorry

end cos_equality_l291_291641


namespace determinant_of_2x2_matrix_l291_291799

theorem determinant_of_2x2_matrix (a b c d : ℝ) (h_a : a = 7) (h_b : b = -2) (h_c : c = -3) (h_d : d = 6) :
  determinant ![![a, b], ![c, d]] = 36 :=
by 
  rw [h_a, h_b, h_c, h_d]
  dsimp
  norm_num
  sorry

end determinant_of_2x2_matrix_l291_291799


namespace TwentyFifthMultipleOfFour_l291_291587

theorem TwentyFifthMultipleOfFour (n : ℕ) (h : ∀ k, 0 <= k ∧ k <= 24 → n = 16 + 4 * k) : n = 112 :=
by
  sorry

end TwentyFifthMultipleOfFour_l291_291587


namespace equidistant_xaxis_point_l291_291282

theorem equidistant_xaxis_point {x : ℝ} :
  (∃ x : ℝ, ∀ A B : ℝ × ℝ, A = (-3, 0) ∧ B = (2, 5) →
    ∀ P : ℝ × ℝ, P = (x, 0) →
      (dist A P = dist B P) → x = 2) := sorry

end equidistant_xaxis_point_l291_291282


namespace part1_part2_l291_291702

theorem part1 (a : ℕ → ℚ) (d : ℚ) (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : 3 * a 2 = 3 * a 1 + a 3) (h₄ : (a 1 + a 2 + a 3) + (1 / a 1 + 3 / (a 1 + d) + 6 / (a 1 + 2 * d)) = 21) : 
  ∀ n, a n = 3 * n :=
sorry

theorem part2 (a : ℕ → ℚ) (d : ℚ) (b : ℕ → ℚ) 
  (h₁ : ∀ n, a (n + 1) = a n + d) (h₂ : d > 1) 
  (h₃ : b = λ n, (n^2 + n) / a n) 
  (h₄ : ∀ n, a n = 3 * n) 
  (h₅ : ∀ n, S n = (n * (a 1 + a n)) / 2) 
  (h₆ : ∀ n, T n = ∑ i in range n, b i) 
  (h₇ : ∀ n, ∃ x, S 99 - T 99 = 99) : 
  d = 51 / 50 :=
sorry

end part1_part2_l291_291702


namespace equivalence_gcd_prime_power_l291_291992

theorem equivalence_gcd_prime_power (a b n : ℕ) :
  (∀ m, 0 < m ∧ m < n → Nat.gcd n ((n - m) / Nat.gcd n m) = 1) ↔ 
  (∃ p k : ℕ, Nat.Prime p ∧ n = p ^ k) :=
by
  sorry

end equivalence_gcd_prime_power_l291_291992


namespace tickets_spent_on_beanie_l291_291006

-- Define the initial conditions
def initial_tickets : ℕ := 25
def additional_tickets : ℕ := 15
def tickets_left : ℕ := 18

-- Define the total tickets
def total_tickets := initial_tickets + additional_tickets

-- Define what we're proving: the number of tickets spent on the beanie
theorem tickets_spent_on_beanie : initial_tickets + additional_tickets - tickets_left = 22 :=
by 
  -- Provide proof steps here
  sorry

end tickets_spent_on_beanie_l291_291006


namespace no_prime_pairs_sum_53_l291_291392

open nat

theorem no_prime_pairs_sum_53 : 
  ¬∃ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 :=
by sorry

end no_prime_pairs_sum_53_l291_291392


namespace domain_correct_l291_291255

def domain_of_function (x : ℝ) : Prop :=
  (∃ y : ℝ, y = 2 / Real.sqrt (x + 1)) ∧ Real.sqrt (x + 1) ≠ 0

theorem domain_correct (x : ℝ) : domain_of_function x ↔ (x > -1) := by
  sorry

end domain_correct_l291_291255


namespace largest_in_set_average_11_l291_291617

theorem largest_in_set_average_11 :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), (a_1 < a_2) ∧ (a_2 < a_3) ∧ (a_3 < a_4) ∧ (a_4 < a_5) ∧
  (1 ≤ a_1 ∧ 1 ≤ a_2 ∧ 1 ≤ a_3 ∧ 1 ≤ a_4 ∧ 1 ≤ a_5) ∧
  (a_1 + a_2 + a_3 + a_4 + a_5 = 55) ∧
  (a_5 = 45) := 
sorry

end largest_in_set_average_11_l291_291617


namespace james_total_money_l291_291537

theorem james_total_money (bills : ℕ) (value_per_bill : ℕ) (initial_money : ℕ) : 
  bills = 3 → value_per_bill = 20 → initial_money = 75 → initial_money + (bills * value_per_bill) = 135 :=
by
  intros hb hv hi
  rw [hb, hv, hi]
  -- Algebraic simplification
  sorry

end james_total_money_l291_291537


namespace brian_books_chapters_l291_291484

variable (x : ℕ)

theorem brian_books_chapters (h1 : 1 ≤ x) (h2 : 20 + 2 * x + (20 + 2 * x) / 2 = 75) : x = 15 :=
sorry

end brian_books_chapters_l291_291484


namespace negation_of_existence_l291_291899

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
sorry

end negation_of_existence_l291_291899


namespace number_of_men_in_first_group_l291_291611

-- Condition: Let M be the number of men in the first group
variable (M : ℕ)

-- Condition: M men can complete the work in 20 hours
-- Condition: 15 men can complete the same work in 48 hours
-- We want to prove that if M * 20 = 15 * 48, then M = 36
theorem number_of_men_in_first_group (h : M * 20 = 15 * 48) : M = 36 := by
  sorry

end number_of_men_in_first_group_l291_291611


namespace geometric_sequence_b_value_l291_291737

theorem geometric_sequence_b_value (r b : ℝ) (h1 : 120 * r = b) (h2 : b * r = 27 / 16) (hb_pos : b > 0) : b = 15 :=
sorry

end geometric_sequence_b_value_l291_291737


namespace magazines_cover_area_l291_291906

theorem magazines_cover_area (S : ℝ) (n : ℕ) (h_n_15 : n = 15) (h_cover : ∀ m ≤ n, ∃(Sm:ℝ), (Sm ≥ (m : ℝ) / n * S) ) :
  ∃ k : ℕ, k = n - 7 ∧ ∃ (Sk : ℝ), (Sk ≥ 8/15 * S) := 
by
  sorry

end magazines_cover_area_l291_291906


namespace download_time_l291_291972

theorem download_time (speed : ℕ) (file1 file2 file3 : ℕ) (total_time : ℕ) (hours : ℕ) :
  speed = 2 ∧ file1 = 80 ∧ file2 = 90 ∧ file3 = 70 ∧ total_time = file1 / speed + file2 / speed + file3 / speed ∧
  hours = total_time / 60 → hours = 2 := 
by
  sorry

end download_time_l291_291972


namespace seventh_term_of_arithmetic_sequence_l291_291270

theorem seventh_term_of_arithmetic_sequence (a d : ℤ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 6) : 
  a + 6 * d = 7 :=
by
  -- Proof omitted
  sorry

end seventh_term_of_arithmetic_sequence_l291_291270


namespace two_pt_seven_five_as_fraction_l291_291433

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l291_291433


namespace total_money_l291_291540

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end total_money_l291_291540


namespace expected_sufferers_l291_291719

theorem expected_sufferers 
  (fraction_condition : ℚ := 1 / 4)
  (sample_size : ℕ := 400) 
  (expected_number : ℕ := 100) : 
  fraction_condition * sample_size = expected_number := 
by 
  sorry

end expected_sufferers_l291_291719


namespace problem1_correct_problem2_correct_l291_291183

open Real

noncomputable def problem1 : ℝ :=
  sqrt(2 + 1 / 4) - (-9.6)^0 - (3 + 3 / 8)^(-2 / 3) + (1.5)^(-2)

-- Let's avoid using sqrt if we directly determine the value 3/2 etc
theorem problem1_correct : problem1 = 1 / 2 :=
by sorry

noncomputable def log_base (b x : ℝ) := log x / log b

noncomputable def problem2 : ℝ :=
  log_base 3 (427 / 3) + log 25 + log 4 + 7^(log_base 7 2)

theorem problem2_correct : problem2 = 15 / 4 :=
by sorry

end problem1_correct_problem2_correct_l291_291183


namespace coin_prob_not_unique_l291_291005

theorem coin_prob_not_unique (p : ℝ) (w : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : w = 144 / 625) :
  ¬ ∃! p, (∃ w, w = 10 * p^3 * (1 - p)^2 ∧ w = 144 / 625) :=
by
  sorry

end coin_prob_not_unique_l291_291005


namespace parametric_plane_equiv_l291_291472

/-- Define the parametric form of the plane -/
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + s - t, 2 - s, 3 - 2*s + 2*t)

/-- Define the equation of the plane in standard form -/
def plane_equation (x y z : ℝ) : Prop :=
  2 * x + z - 5 = 0

/-- The theorem stating that the parametric form corresponds to the given plane equation -/
theorem parametric_plane_equiv :
  ∃ x y z s t,
    (x, y, z) = parametric_plane s t ∧ plane_equation x y z :=
by
  sorry

end parametric_plane_equiv_l291_291472


namespace shaded_region_area_l291_291084

-- Conditions given in the problem
def diameter (d : ℝ) := d = 4
def length_feet (l : ℝ) := l = 2

-- Proof statement
theorem shaded_region_area (d l : ℝ) (h1 : diameter d) (h2 : length_feet l) : 
  (l * 12 / d * (d / 2)^2 * π = 24 * π) := by
  sorry

end shaded_region_area_l291_291084


namespace problem_statement_l291_291679

variables (x y : ℝ)

theorem problem_statement
  (h1 : abs x = 4)
  (h2 : abs y = 2)
  (h3 : abs (x + y) = x + y) : 
  x - y = 2 ∨ x - y = 6 :=
sorry

end problem_statement_l291_291679


namespace gumballs_each_shared_equally_l291_291542

def initial_gumballs_joanna : ℕ := 40
def initial_gumballs_jacques : ℕ := 60
def multiplier : ℕ := 4

def purchased_gumballs (initial : ℕ) (multiplier : ℕ) : ℕ :=
  initial * multiplier

def total_gumballs (initial : ℕ) (purchased : ℕ) : ℕ :=
  initial + purchased

def total_combined_gumballs (total1 : ℕ) (total2 : ℕ) : ℕ :=
  total1 + total2

def shared_equally (total : ℕ) : ℕ :=
  total / 2

theorem gumballs_each_shared_equally :
  let joanna_initial := initial_gumballs_joanna,
      jacques_initial := initial_gumballs_jacques,
      joanna_purchased := purchased_gumballs joanna_initial multiplier,
      jacques_purchased := purchased_gumballs jacques_initial multiplier,
      joanna_total := total_gumballs joanna_initial joanna_purchased,
      jacques_total := total_gumballs jacques_initial jacques_purchased,
      combined_total := total_combined_gumballs joanna_total jacques_total in
  shared_equally combined_total = 250 :=
by
  sorry

end gumballs_each_shared_equally_l291_291542


namespace decimal_to_fraction_l291_291441

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l291_291441


namespace temp_interpretation_l291_291182

theorem temp_interpretation (below_zero : ℤ) (above_zero : ℤ) (h : below_zero = -2):
  above_zero = 3 → 3 = 0 := by
  intro h2
  have : above_zero = 3 := h2
  sorry

end temp_interpretation_l291_291182


namespace circular_table_permutations_l291_291983

/--
Given 8 chairs evenly spaced around a circular table, and 8 people initially seated in each chair, 
each person gets up and sits down in a different, non-adjacent chair, 
so that again each chair is occupied by one person.
Prove that there are exactly 80 valid permutations of seating.
-/
theorem circular_table_permutations : 
  let chairs := {1, 2, 3, 4, 5, 6, 7, 8}
  let permutations := {σ : Equiv.Perm chairs | ∀ i, σ i ≠ i ∧ σ i ≠ (i + 1) % 8 ∧ σ i ≠ (i - 1) % 8}
  permutations.card = 80 := 
by
  sorry

end circular_table_permutations_l291_291983


namespace max_pencils_to_buy_l291_291479

-- Definition of costs and budget
def pin_cost : ℕ := 3
def pen_cost : ℕ := 4
def pencil_cost : ℕ := 9
def total_budget : ℕ := 72

-- Minimum purchase required: one pin and one pen
def min_purchase : ℕ := pin_cost + pen_cost

-- Remaining budget after minimum purchase
def remaining_budget : ℕ := total_budget - min_purchase

-- Maximum number of pencils can be bought with the remaining budget
def max_pencils := remaining_budget / pencil_cost

-- Theorem stating the maximum number of pencils Alice can purchase
theorem max_pencils_to_buy : max_pencils = 7 :=
by
  -- Proof would go here
  sorry

end max_pencils_to_buy_l291_291479


namespace equal_perimeter_triangle_side_length_l291_291309

theorem equal_perimeter_triangle_side_length (s: ℝ) : 
    ∀ (pentagon_perimeter triangle_perimeter: ℝ), 
    (pentagon_perimeter = 5 * 5) → 
    (triangle_perimeter = 3 * s) → 
    (pentagon_perimeter = triangle_perimeter) → 
    s = 25 / 3 :=
by
  intro pentagon_perimeter triangle_perimeter h1 h2 h3
  sorry

end equal_perimeter_triangle_side_length_l291_291309


namespace minimum_trees_l291_291853

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l291_291853


namespace symmetric_point_coordinates_l291_291425

structure Point : Type where
  x : ℝ
  y : ℝ

def symmetric_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def P : Point := { x := -10, y := -1 }

def P1 : Point := symmetric_y P

def P2 : Point := symmetric_x P1

theorem symmetric_point_coordinates :
  P2 = { x := 10, y := 1 } := by
  sorry

end symmetric_point_coordinates_l291_291425


namespace number_of_permutations_l291_291262

theorem number_of_permutations (readers : Fin 8 → Type) : ∃! (n : ℕ), n = 40320 :=
by
  sorry

end number_of_permutations_l291_291262


namespace wholesale_price_l291_291311

theorem wholesale_price (W R SP : ℝ) (h1 : R = 120) (h2 : SP = R - 0.10 * R) (h3 : SP = W + 0.20 * W) : 
  W = 90 :=
by
  -- Given conditions
  have hR : R = 120 := h1
  have hSP_def : SP = 108 := by
    rw [h1] at h2
    norm_num at h2
  -- Solving for W
  sorry

end wholesale_price_l291_291311


namespace calculate_expression_l291_291015

theorem calculate_expression : 
  (3 * 7.5 * (6 + 4) / 2.5) = 90 := 
by
  sorry

end calculate_expression_l291_291015


namespace scientific_notation_of_21500000_l291_291382

theorem scientific_notation_of_21500000 :
  21500000 = 2.15 * 10^7 :=
by
  sorry

end scientific_notation_of_21500000_l291_291382


namespace find_m_l291_291354

theorem find_m (n : ℝ) : 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 :=
by
  sorry

end find_m_l291_291354


namespace smallest_pieces_left_l291_291362

theorem smallest_pieces_left (m n : ℕ) (h1 : 1 < m) (h2 : 1 < n) : 
    ∃ k, (k = 2 ∧ (m * n) % 3 = 0) ∨ (k = 1 ∧ (m * n) % 3 ≠ 0) :=
by
    sorry

end smallest_pieces_left_l291_291362


namespace Owen_spent_720_dollars_on_burgers_l291_291991

def days_in_June : ℕ := 30
def burgers_per_day : ℕ := 2
def cost_per_burger : ℕ := 12

def total_burgers (days : ℕ) (burgers_per_day : ℕ) : ℕ :=
  days * burgers_per_day

def total_cost (burgers : ℕ) (cost_per_burger : ℕ) : ℕ :=
  burgers * cost_per_burger

theorem Owen_spent_720_dollars_on_burgers :
  total_cost (total_burgers days_in_June burgers_per_day) cost_per_burger = 720 := by
  sorry

end Owen_spent_720_dollars_on_burgers_l291_291991


namespace minimum_selling_price_l291_291306

theorem minimum_selling_price (total_cost : ℝ) (total_fruit : ℝ) (spoilage : ℝ) (min_price : ℝ) :
  total_cost = 760 ∧ total_fruit = 80 ∧ spoilage = 0.05 ∧ min_price = 10 → 
  ∀ price : ℝ, (price * total_fruit * (1 - spoilage) >= total_cost) → price >= min_price :=
by
  intros h price hp
  rcases h with ⟨hc, hf, hs, hm⟩
  sorry

end minimum_selling_price_l291_291306


namespace pseudo_code_output_l291_291658

theorem pseudo_code_output (a b c : Int)
  (h1 : a = 3)
  (h2 : b = -5)
  (h3 : c = 8)
  (ha : a = -5)
  (hb : b = 8)
  (hc : c = -5) : 
  a = -5 ∧ b = 8 ∧ c = -5 :=
by
  sorry

end pseudo_code_output_l291_291658


namespace final_sign_is_minus_l291_291291

theorem final_sign_is_minus 
  (plus_count : ℕ) 
  (minus_count : ℕ) 
  (h_plus : plus_count = 2004) 
  (h_minus : minus_count = 2005) 
  (transform : (ℕ → ℕ → ℕ × ℕ) → Prop) :
  transform (fun plus minus =>
    if plus >= 2 then (plus - 1, minus)
    else if minus >= 2 then (plus, minus - 1)
    else if plus > 0 && minus > 0 then (plus - 1, minus - 1)
    else (0, 0)) →
  (plus_count = 0 ∧ minus_count = 1) := sorry

end final_sign_is_minus_l291_291291


namespace driving_time_per_trip_l291_291570

-- Define the conditions
def filling_time_per_trip : ℕ := 15
def number_of_trips : ℕ := 6
def total_moving_hours : ℕ := 7
def total_moving_time : ℕ := total_moving_hours * 60

-- Define the problem
theorem driving_time_per_trip :
  (total_moving_time - (filling_time_per_trip * number_of_trips)) / number_of_trips = 55 :=
by
  sorry

end driving_time_per_trip_l291_291570


namespace emma_bank_account_balance_l291_291345

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end emma_bank_account_balance_l291_291345


namespace students_taking_German_l291_291056

theorem students_taking_German 
  (total_students : ℕ)
  (students_taking_French : ℕ)
  (students_taking_both : ℕ)
  (students_not_taking_either : ℕ) 
  (students_taking_German : ℕ) 
  (h1 : total_students = 69)
  (h2 : students_taking_French = 41)
  (h3 : students_taking_both = 9)
  (h4 : students_not_taking_either = 15)
  (h5 : students_taking_German = 22) :
  total_students - students_not_taking_either = students_taking_French + students_taking_German - students_taking_both :=
sorry

end students_taking_German_l291_291056


namespace B_lap_time_l291_291172

-- Definitions based on given conditions.
def time_to_complete_lap_A := 40
def meeting_interval := 15

-- The theorem states that given the conditions, B takes 24 seconds to complete the track.
theorem B_lap_time (l : ℝ) (t : ℝ) (h1 : t = 24)
                    (h2 : l / time_to_complete_lap_A + l / t = l / meeting_interval):
  t = 24 := by sorry

end B_lap_time_l291_291172


namespace min_trees_for_three_types_l291_291851

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l291_291851


namespace whale_plankton_feeding_frenzy_l291_291313

theorem whale_plankton_feeding_frenzy
  (x y : ℕ)
  (h1 : x + 5 * y = 54)
  (h2 : 9 * x + 36 * y = 450) :
  y = 4 :=
sorry

end whale_plankton_feeding_frenzy_l291_291313


namespace max_distance_traveled_l291_291415

def distance_traveled (t : ℝ) : ℝ := 15 * t - 6 * t^2

theorem max_distance_traveled : ∃ t : ℝ, distance_traveled t = 75 / 8 :=
by
  sorry

end max_distance_traveled_l291_291415


namespace colin_avg_time_l291_291798

def totalTime (a b c d : ℕ) : ℕ := a + b + c + d

def averageTime (total_time miles : ℕ) : ℕ := total_time / miles

theorem colin_avg_time :
  let first_mile := 6
  let second_mile := 5
  let third_mile := 5
  let fourth_mile := 4
  let total_time := totalTime first_mile second_mile third_mile fourth_mile
  4 > 0 -> averageTime total_time 4 = 5 :=
by
  intros
  -- proof goes here
  sorry

end colin_avg_time_l291_291798


namespace bike_trike_race_l291_291613

theorem bike_trike_race (P : ℕ) (B T : ℕ) (h1 : B = (3 * P) / 5) (h2 : T = (2 * P) / 5) (h3 : 2 * B + 3 * T = 96) :
  P = 40 :=
by
  sorry

end bike_trike_race_l291_291613


namespace root_in_interval_implies_a_in_range_l291_291842

theorem root_in_interval_implies_a_in_range {a : ℝ} (h : ∃ x : ℝ, x ≤ 1 ∧ 2^x - a^2 - a = 0) : 0 < a ∧ a ≤ 1 := sorry

end root_in_interval_implies_a_in_range_l291_291842


namespace tangent_line_through_origin_l291_291830

theorem tangent_line_through_origin (x : ℝ) (h₁ : 0 < x) (h₂ : ∀ x, ∃ y, y = 2 * Real.log x) (h₃ : ∀ x, y = 2 * Real.log x) :
  x = Real.exp 1 :=
sorry

end tangent_line_through_origin_l291_291830


namespace two_point_seven_five_as_fraction_l291_291438

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l291_291438


namespace coastal_village_population_l291_291585

variable (N : ℕ) (k : ℕ) (parts_for_males : ℕ) (total_males : ℕ)

theorem coastal_village_population 
  (h_total_population : N = 540)
  (h_division : k = 4)
  (h_parts_for_males : parts_for_males = 2)
  (h_total_males : total_males = (N / k) * parts_for_males) :
  total_males = 270 := 
by
  sorry

end coastal_village_population_l291_291585


namespace square_side_length_eq_8_over_pi_l291_291896

noncomputable def side_length_square : ℝ := 8 / Real.pi

theorem square_side_length_eq_8_over_pi :
  ∀ (s : ℝ),
  (4 * s = (Real.pi * (s / Real.sqrt 2) ^ 2) / 2) →
  s = side_length_square :=
by
  intro s h
  sorry

end square_side_length_eq_8_over_pi_l291_291896


namespace more_balloons_allan_l291_291179

theorem more_balloons_allan (allan_balloons : ℕ) (jake_initial_balloons : ℕ) (jake_bought_balloons : ℕ) 
  (h1 : allan_balloons = 6) (h2 : jake_initial_balloons = 2) (h3 : jake_bought_balloons = 3) :
  allan_balloons = jake_initial_balloons + jake_bought_balloons + 1 := 
by 
  -- Assuming Jake's total balloons after purchase
  let jake_total_balloons := jake_initial_balloons + jake_bought_balloons
  -- The proof would involve showing that Allan's balloons are one more than Jake's total balloons
  sorry

end more_balloons_allan_l291_291179


namespace person_y_speed_in_still_water_l291_291280

theorem person_y_speed_in_still_water 
    (speed_x_in_still_water : ℝ)
    (time_meeting_towards_each_other : ℝ)
    (time_catching_up_same_direction: ℝ)
    (distance_upstream_meeting: ℝ)
    (distance_downstream_meeting: ℝ)
    (total_distance: ℝ) :
    speed_x_in_still_water = 6 →
    time_meeting_towards_each_other = 4 →
    time_catching_up_same_direction = 16 →
    distance_upstream_meeting = 4 * (6 - distance_upstream_meeting) + 4 * (10 + distance_downstream_meeting) →
    distance_downstream_meeting = 4 * (6 + distance_upstream_meeting) →
    total_distance = 4 * (6 + 10) →
    ∃ (speed_y_in_still_water : ℝ), speed_y_in_still_water = 10 :=
by
  intros h_speed_x h_time_meeting h_time_catching h_distance_upstream h_distance_downstream h_total_distance
  sorry

end person_y_speed_in_still_water_l291_291280


namespace one_third_of_five_times_seven_l291_291637

theorem one_third_of_five_times_seven:
  (1/3 : ℝ) * (5 * 7) = 35 / 3 := 
by
  -- Definitions and calculations go here
  sorry

end one_third_of_five_times_seven_l291_291637


namespace correct_equation_solves_time_l291_291793

noncomputable def solve_time_before_stop (t : ℝ) : Prop :=
  let total_trip_time := 4 -- total trip time in hours including stop
  let stop_time := 0.5 -- stop time in hours
  let total_distance := 180 -- total distance in km
  let speed_before_stop := 60 -- speed before stop in km/h
  let speed_after_stop := 80 -- speed after stop in km/h
  let time_after_stop := total_trip_time - stop_time - t -- time after the stop in hours
  speed_before_stop * t + speed_after_stop * time_after_stop = total_distance -- distance equation

-- The theorem states that the equation is valid for solving t
theorem correct_equation_solves_time :
  solve_time_before_stop t = (60 * t + 80 * (7/2 - t) = 180) :=
sorry -- proof not required

end correct_equation_solves_time_l291_291793


namespace products_arrangement_count_l291_291911

/--
There are five different products: A, B, C, D, and E arranged in a row on a shelf.
- Products A and B must be adjacent.
- Products C and D must not be adjacent.
Prove that there are a total of 24 distinct valid arrangements under these conditions.
-/
theorem products_arrangement_count : 
  ∃ (n : ℕ), 
  (∀ (A B C D E : Type), n = 24 ∧
  ∀ l : List (Type), l = [A, B, C, D, E] ∧
  -- A and B must be adjacent
  (∀ p : List (Type), p = [A, B] ∨ p = [B, A]) ∧
  -- C and D must not be adjacent
  ¬ (∀ q : List (Type), q = [C, D] ∨ q = [D, C])) :=
sorry

end products_arrangement_count_l291_291911


namespace fraction_remain_unchanged_l291_291141

theorem fraction_remain_unchanged (m n a b : ℚ) (h : n ≠ 0 ∧ b ≠ 0) : 
  (a / b = (a + m) / (b + n)) ↔ (a / b = m / n) :=
sorry

end fraction_remain_unchanged_l291_291141


namespace custom_op_evaluation_l291_291522

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : (custom_op 9 6) - (custom_op 6 9) = -12 := by
  sorry

end custom_op_evaluation_l291_291522


namespace simplify_expression_l291_291359

theorem simplify_expression :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 10) / 4) = 13 := by
  sorry

end simplify_expression_l291_291359


namespace decimal_to_fraction_l291_291443

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l291_291443


namespace father_ate_8_brownies_l291_291716

noncomputable def brownies_initial := 24
noncomputable def brownies_mooney_ate := 4
noncomputable def brownies_after_mooney := brownies_initial - brownies_mooney_ate
noncomputable def brownies_mother_made_next_day := 24
noncomputable def brownies_total_expected := brownies_after_mooney + brownies_mother_made_next_day
noncomputable def brownies_actual_on_counter := 36

theorem father_ate_8_brownies :
  brownies_total_expected - brownies_actual_on_counter = 8 :=
by
  sorry

end father_ate_8_brownies_l291_291716


namespace percentage_shaded_l291_291475

def area_rect (width height : ℝ) : ℝ := width * height

def overlap_area (side_length : ℝ) (width_rect : ℝ) (length_rect: ℝ) (length_total: ℝ) : ℝ :=
  (side_length - (length_total - length_rect)) * width_rect

theorem percentage_shaded (sqr_side length_rect width_rect total_length total_width : ℝ) (h1 : sqr_side = 12) (h2 : length_rect = 9) (h3 : width_rect = 12)
  (h4 : total_length = 18) (h5 : total_width = 12) :
  (overlap_area sqr_side width_rect length_rect total_length) / (area_rect total_width total_length) * 100 = 12.5 :=
by
  sorry

end percentage_shaded_l291_291475


namespace intersection_M_N_l291_291400

def M : Set ℝ := {y | ∃ x, x ∈ Set.Icc (-5) 5 ∧ y = 2 * Real.sin x}
def N : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2}

theorem intersection_M_N : {x | 1 < x ∧ x ≤ 2} = {x | x ∈ M ∩ N} :=
by sorry

end intersection_M_N_l291_291400


namespace gain_amount_is_ten_l291_291010

theorem gain_amount_is_ten (S : ℝ) (C : ℝ) (g : ℝ) (G : ℝ) 
  (h1 : S = 110) (h2 : g = 0.10) (h3 : S = C + g * C) : G = 10 :=
by 
  sorry

end gain_amount_is_ten_l291_291010


namespace sum_of_vertices_l291_291031

theorem sum_of_vertices (pentagon_vertices : Nat := 5) (hexagon_vertices : Nat := 6) :
  (2 * pentagon_vertices) + (2 * hexagon_vertices) = 22 :=
by
  sorry

end sum_of_vertices_l291_291031


namespace line_intersects_y_axis_at_l291_291624

-- Define the two points the line passes through
structure Point (α : Type) :=
(x : α)
(y : α)

def p1 : Point ℤ := Point.mk 2 9
def p2 : Point ℤ := Point.mk 4 13

-- Define the function that describes the point where the line intersects the y-axis
def y_intercept : Point ℤ :=
  -- We are proving that the line intersects the y-axis at the point (0, 5)
  Point.mk 0 5

-- State the theorem to be proven
theorem line_intersects_y_axis_at (p1 p2 : Point ℤ) (yi : Point ℤ) :
  p1.x = 2 ∧ p1.y = 9 ∧ p2.x = 4 ∧ p2.y = 13 → yi = Point.mk 0 5 :=
by
  intros
  sorry

end line_intersects_y_axis_at_l291_291624


namespace sum_of_cubes_unique_count_l291_291669

theorem sum_of_cubes_unique_count : 
  {n : ℕ | ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000}.to_finset.card = 35 :=
by
  sorry

end sum_of_cubes_unique_count_l291_291669


namespace factorization_problem_l291_291890

theorem factorization_problem 
    (a m n b : ℝ)
    (h1 : (x + 2) * (x + 4) = x^2 + a * x + m)
    (h2 : (x + 1) * (x + 9) = x^2 + n * x + b) :
    (x + 3) * (x + 3) = x^2 + a * x + b :=
by
  sorry

end factorization_problem_l291_291890


namespace complement_eq_target_l291_291831

namespace ComplementProof

-- Define the universal set U
def U : Set ℕ := {2, 4, 6, 8, 10}

-- Define the set A
def A : Set ℕ := {2, 6, 8}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}

-- Define the target set
def target_set : Set ℕ := {4, 10}

-- Prove that the complement of A with respect to U is equal to {4, 10}
theorem complement_eq_target :
  complement_U_A = target_set := by sorry

end ComplementProof

end complement_eq_target_l291_291831


namespace max_a_l291_291050

variable {a x : ℝ}

theorem max_a (h : x^2 - 2 * x - 3 > 0 → x < a ∧ ¬ (x < a → x^2 - 2 * x - 3 > 0)) : a = 3 :=
sorry

end max_a_l291_291050


namespace no_int_solutions_p_mod_4_neg_1_l291_291073

theorem no_int_solutions_p_mod_4_neg_1 :
  ∀ (p n : ℕ), (p % 4 = 3) → (∀ x y : ℕ, x^2 + y^2 ≠ p^n) :=
by
  intros
  sorry

end no_int_solutions_p_mod_4_neg_1_l291_291073


namespace quadratic_roots_opposite_signs_l291_291025

theorem quadratic_roots_opposite_signs (a : ℝ) :
  (∃ x y : ℝ, (a * x^2 - (a + 3) * x + 2 = 0) ∧ (a * y^2 - (a + 3) * y + 2 = 0) ∧ x * y < 0) ↔ (a < 0) :=
sorry

end quadratic_roots_opposite_signs_l291_291025


namespace onion_to_carrot_ratio_l291_291560

theorem onion_to_carrot_ratio (p c o g : ℕ) (h1 : 6 * p = c) (h2 : c = o) (h3 : g = 1 / 3 * o) (h4 : p = 2) (h5 : g = 8) : o / c = 1 / 1 :=
by
  sorry

end onion_to_carrot_ratio_l291_291560


namespace count_cube_sums_lt_1000_l291_291670

theorem count_cube_sums_lt_1000 :
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000}.to_finset.card = 36 :=
by
  sorry

end count_cube_sums_lt_1000_l291_291670


namespace y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l291_291874

def y : ℕ := 112 + 160 + 272 + 432 + 1040 + 1264 + 4256

theorem y_is_multiple_of_16 : y % 16 = 0 :=
sorry

theorem y_is_multiple_of_8 : y % 8 = 0 :=
sorry

theorem y_is_multiple_of_4 : y % 4 = 0 :=
sorry

theorem y_is_multiple_of_2 : y % 2 = 0 :=
sorry

end y_is_multiple_of_16_y_is_multiple_of_8_y_is_multiple_of_4_y_is_multiple_of_2_l291_291874


namespace sum_even_odd_diff_l291_291915

theorem sum_even_odd_diff (n : ℕ) (h : n = 1500) : 
  let S_odd := n / 2 * (1 + (1 + (n - 1) * 2))
  let S_even := n / 2 * (2 + (2 + (n - 1) * 2))
  (S_even - S_odd) = n :=
by
  sorry

end sum_even_odd_diff_l291_291915


namespace cost_split_difference_l291_291083

-- Definitions of amounts paid
def SarahPaid : ℕ := 150
def DerekPaid : ℕ := 210
def RitaPaid : ℕ := 240

-- Total paid by all three
def TotalPaid : ℕ := SarahPaid + DerekPaid + RitaPaid

-- Each should have paid:
def EachShouldHavePaid : ℕ := TotalPaid / 3

-- Amount Sarah owes Rita
def SarahOwesRita : ℕ := EachShouldHavePaid - SarahPaid

-- Amount Derek should receive back from Rita
def DerekShouldReceiveFromRita : ℕ := DerekPaid - EachShouldHavePaid

-- Difference between the amounts Sarah and Derek owe/should receive from Rita
theorem cost_split_difference : SarahOwesRita - DerekShouldReceiveFromRita = 60 := by
    sorry

end cost_split_difference_l291_291083


namespace product_of_roots_of_quadratic_l291_291290

   -- Definition of the quadratic equation used in the condition
   def quadratic (x : ℝ) : ℝ := x^2 - 2 * x - 8

   -- Problem statement: Prove that the product of the roots of the given quadratic equation is -8.
   theorem product_of_roots_of_quadratic : 
     (∀ x : ℝ, quadratic x = 0 → (x = 4 ∨ x = -2)) → (4 * -2 = -8) :=
   by
     sorry
   
end product_of_roots_of_quadratic_l291_291290


namespace intercepts_of_line_l291_291638

theorem intercepts_of_line (x y : ℝ) (h_eq : 4 * x + 7 * y = 28) :
  (∃ y, (x = 0 ∧ y = 4) ∧ ∃ x, (y = 0 ∧ x = 7)) :=
by
  sorry

end intercepts_of_line_l291_291638


namespace y_difference_positive_l291_291825

theorem y_difference_positive (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * 1^2 + 2 * a * 1 + c)
  (h3 : y2 = a * 2^2 + 2 * a * 2 + c) : y1 - y2 > 0 := 
sorry

end y_difference_positive_l291_291825


namespace Annie_cookies_sum_l291_291969

theorem Annie_cookies_sum :
  let cookies_monday := 5
  let cookies_tuesday := 2 * cookies_monday
  let cookies_wednesday := cookies_tuesday + (40 / 100) * cookies_tuesday
  cookies_monday + cookies_tuesday + cookies_wednesday = 29 :=
by
  sorry

end Annie_cookies_sum_l291_291969


namespace find_intersections_l291_291504

noncomputable def intersection_points (α : ℝ) (t θ : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ t θ : ℝ, p = ((1 + t * Real.cos α), t * Real.sin α) ∧
                   p = (Real.cos θ, Real.sin θ)}

theorem find_intersections :
  let α := Real.pi / 3 in
  intersection_points α t θ = {(1, 0), (1 / 2, - Real.sqrt 3 / 2)} :=
by
  sorry

end find_intersections_l291_291504


namespace lcm_two_primes_is_10_l291_291217

theorem lcm_two_primes_is_10 (x y : ℕ) (h_prime_x : Nat.Prime x) (h_prime_y : Nat.Prime y) (h_lcm : Nat.lcm x y = 10) (h_gt : x > y) : 2 * x + y = 12 :=
sorry

end lcm_two_primes_is_10_l291_291217


namespace complement_union_l291_291558

def universal_set : Set ℝ := { x : ℝ | true }
def M : Set ℝ := { x : ℝ | x ≤ 0 }
def N : Set ℝ := { x : ℝ | x > 2 }

theorem complement_union (x : ℝ) :
  x ∈ compl (M ∪ N) ↔ (0 < x ∧ x ≤ 2) := by
  sorry

end complement_union_l291_291558


namespace interval_solution_l291_291644

theorem interval_solution (x : ℝ) : 
  (1 < 5 * x ∧ 5 * x < 3) ∧ (2 < 8 * x ∧ 8 * x < 4) ↔ (1/4 < x ∧ x < 1/2) := 
by
  sorry

end interval_solution_l291_291644


namespace new_average_weight_l291_291574

def average_weight (A B C D E : ℝ) : Prop :=
  (A + B + C) / 3 = 70 ∧
  (A + B + C + D) / 4 = 70 ∧
  E = D + 3 ∧
  A = 81

theorem new_average_weight (A B C D E : ℝ) (h: average_weight A B C D E) : 
  (B + C + D + E) / 4 = 68 :=
by
  sorry

end new_average_weight_l291_291574


namespace fill_in_the_blank_with_flowchart_l291_291583

def methods_to_describe_algorithm := ["Natural language", "Flowchart", "Pseudocode"]

theorem fill_in_the_blank_with_flowchart : 
  methods_to_describe_algorithm[1] = "Flowchart" :=
sorry

end fill_in_the_blank_with_flowchart_l291_291583


namespace sum_of_roots_l291_291103

theorem sum_of_roots (x₁ x₂ b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + b = 0 → x = x₁ ∨ x = x₂) :
  x₁ + x₂ = 2 :=
sorry

end sum_of_roots_l291_291103


namespace diagonals_in_eight_sided_polygon_l291_291304

-- Definitions based on the conditions
def n := 8  -- Number of sides
def right_angles := 2  -- Number of right angles

-- Calculating the number of diagonals using the formula
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Lean statement for the problem
theorem diagonals_in_eight_sided_polygon : num_diagonals n = 20 :=
by
  -- Substitute n = 8 into the formula and simplify
  sorry

end diagonals_in_eight_sided_polygon_l291_291304


namespace one_half_percent_as_decimal_l291_291835

def percent_to_decimal (x : ℚ) := x / 100

theorem one_half_percent_as_decimal : percent_to_decimal (1 / 2) = 0.005 := 
by
  sorry

end one_half_percent_as_decimal_l291_291835


namespace positive_real_solutions_unique_l291_291636

theorem positive_real_solutions_unique (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
(h : (a^2 - b * d) / (b + 2 * c + d) + (b^2 - c * a) / (c + 2 * d + a) + (c^2 - d * b) / (d + 2 * a + b) + (d^2 - a * c) / (a + 2 * b + c) = 0) : 
a = b ∧ b = c ∧ c = d :=
sorry

end positive_real_solutions_unique_l291_291636


namespace colin_average_time_per_mile_l291_291797

theorem colin_average_time_per_mile :
  (let first_mile := 6
   let second_mile := 5
   let third_mile := 5
   let fourth_mile := 4
   let total_miles := 4
   let total_time := first_mile + second_mile + third_mile + fourth_mile
   let average_time := total_time / total_miles
   average_time = 5) :=
begin
  let first_mile := 6,
  let second_mile := 5,
  let third_mile := 5,
  let fourth_mile := 4,
  let total_miles := 4,
  let total_time := first_mile + second_mile + third_mile + fourth_mile,
  let average_time := total_time / total_miles,
  show average_time = 5,
  sorry,
end

end colin_average_time_per_mile_l291_291797


namespace remainder_sum_l291_291448

theorem remainder_sum (n : ℤ) (h : n % 21 = 13) : (n % 3 + n % 7) = 7 := by
  sorry

end remainder_sum_l291_291448


namespace decimal_to_fraction_l291_291435

-- Define the decimal number 2.75
def decimal_num : ℝ := 2.75

-- Define the expected fraction in unsimplified form
def unsimplified_fraction := 275 / 100

-- The greatest common divisor of 275 and 100
def gcd_275_100 : ℕ := 25

-- Define the simplified fraction as 11/4
def simplified_fraction := 11 / 4

-- Statement of the theorem to prove
theorem decimal_to_fraction : (decimal_num : ℚ) = simplified_fraction :=
by
  -- Here you can write the proof steps or use sorry to denote the proof is omitted
  sorry

end decimal_to_fraction_l291_291435


namespace contestant_wins_probability_l291_291784

section RadioProgramQuiz
  -- Defining the conditions
  def number_of_questions : ℕ := 4
  def number_of_choices_per_question : ℕ := 3
  def probability_of_correct_answer : ℚ := 1 / 3
  
  -- Defining the target probability
  def winning_probability : ℚ := 1 / 9

  -- The theorem
  theorem contestant_wins_probability :
    (let p := probability_of_correct_answer
     let p_correct_all := p^4
     let p_correct_three :=
       4 * (p^3 * (1 - p))
     p_correct_all + p_correct_three = winning_probability) :=
    sorry
end RadioProgramQuiz

end contestant_wins_probability_l291_291784


namespace fraction_of_number_l291_291115

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l291_291115


namespace find_n_for_constant_term_l291_291815

theorem find_n_for_constant_term :
  ∃ n : ℕ, (binom n 4) * 15 = 15 := sorry

end find_n_for_constant_term_l291_291815


namespace no_opposite_meanings_in_C_l291_291180

def opposite_meanings (condition : String) : Prop :=
  match condition with
  | "A" => true
  | "B" => true
  | "C" => false
  | "D" => true
  | _   => false

theorem no_opposite_meanings_in_C :
  opposite_meanings "C" = false :=
by
  -- proof goes here
  sorry

end no_opposite_meanings_in_C_l291_291180


namespace time_to_cross_platform_l291_291771

-- Definitions based on the given conditions
def train_length : ℝ := 300
def time_to_cross_pole : ℝ := 18
def platform_length : ℝ := 350

-- The question reformulated as a theorem in Lean 4
theorem time_to_cross_platform 
  (l_train : ℝ := train_length)
  (t_pole_cross : ℝ := time_to_cross_pole)
  (l_platform : ℝ := platform_length) :
  (l_train / t_pole_cross * (l_train + l_platform) = 39) :=
sorry

end time_to_cross_platform_l291_291771


namespace divides_sum_if_divides_polynomial_l291_291556

theorem divides_sum_if_divides_polynomial (x y : ℕ) : 
  x^2 ∣ x^2 + x * y + x + y → x^2 ∣ x + y :=
by
  sorry

end divides_sum_if_divides_polynomial_l291_291556


namespace opposite_of_negative_seven_l291_291264

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_negative_seven : opposite (-7) = 7 := 
by 
  sorry

end opposite_of_negative_seven_l291_291264


namespace red_packet_grabbing_situations_l291_291055

-- Definitions based on the conditions
def numberOfPeople := 5
def numberOfPackets := 4
def packets := [2, 2, 3, 5]  -- 2-yuan, 2-yuan, 3-yuan, 5-yuan

-- Main theorem statement
theorem red_packet_grabbing_situations : 
  ∃ situations : ℕ, situations = 60 :=
by
  sorry

end red_packet_grabbing_situations_l291_291055


namespace part1_l291_291698

variable {a b : ℝ}
variable {A B C : ℝ}
variable {S : ℝ}

-- Given Conditions
def is_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (b * Real.cos C - c * Real.cos B = 2 * a) ∧ (c = a)

-- To prove
theorem part1 (h : is_triangle A B C a b a) : B = 2 * Real.pi / 3 := sorry

end part1_l291_291698


namespace abs_neg_sub_three_eq_zero_l291_291327

theorem abs_neg_sub_three_eq_zero : |(-3 : ℤ)| - 3 = 0 :=
by sorry

end abs_neg_sub_three_eq_zero_l291_291327


namespace solution_set_of_inequality_l291_291901

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) : 
  ((2 * x) / (x - 2) ≤ 1) ↔ (-2 ≤ x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l291_291901


namespace total_cookies_correct_l291_291965

noncomputable def cookies_monday : ℕ := 5
def cookies_tuesday := 2 * cookies_monday
def cookies_wednesday := cookies_tuesday + (40 * cookies_tuesday / 100)
def total_cookies := cookies_monday + cookies_tuesday + cookies_wednesday

theorem total_cookies_correct : total_cookies = 29 := by
  sorry

end total_cookies_correct_l291_291965


namespace mrs_hilt_read_chapters_l291_291239

-- Define the problem conditions
def books : ℕ := 4
def chapters_per_book : ℕ := 17

-- State the proof problem
theorem mrs_hilt_read_chapters : (books * chapters_per_book) = 68 := 
by
  sorry

end mrs_hilt_read_chapters_l291_291239


namespace find_root_power_117_l291_291523

noncomputable def problem (a b c : ℝ) (x1 x2 : ℝ) :=
  (3 * a - b) / c * x1^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  (3 * a - b) / c * x2^2 + c * (3 * a + b) / (3 * a - b) = 0 ∧
  x1 + x2 = 0

theorem find_root_power_117 (a b c : ℝ) (x1 x2 : ℝ) (h : problem a b c x1 x2) : 
  x1 ^ 117 + x2 ^ 117 = 0 :=
sorry

end find_root_power_117_l291_291523


namespace trig_identity_problem_l291_291655

theorem trig_identity_problem 
  (t m n k : ℕ) 
  (h_rel_prime : Nat.gcd m n = 1) 
  (h_condition1 : (1 + Real.sin t) * (1 + Real.cos t) = 8 / 9) 
  (h_condition2 : (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt k) 
  (h_pos_int_m : 0 < m) 
  (h_pos_int_n : 0 < n) 
  (h_pos_int_k : 0 < k) :
  k + m + n = 15 := 
sorry

end trig_identity_problem_l291_291655


namespace calories_burned_l291_291621

theorem calories_burned {running_minutes walking_minutes total_minutes calories_per_minute_running calories_per_minute_walking calories_total : ℕ}
    (h_run : running_minutes = 35)
    (h_total : total_minutes = 60)
    (h_calories_run : calories_per_minute_running = 10)
    (h_calories_walk : calories_per_minute_walking = 4)
    (h_walk : walking_minutes = total_minutes - running_minutes)
    (h_calories_total : calories_total = running_minutes * calories_per_minute_running + walking_minutes * calories_per_minute_walking) : 
    calories_total = 450 := by
  sorry

end calories_burned_l291_291621


namespace exists_x0_lt_l291_291872

noncomputable def P (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d
noncomputable def Q (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem exists_x0_lt {a b c d p q r s : ℝ} (h1 : r < s) (h2 : s - r > 2)
  (h3 : ∀ x, r < x ∧ x < s → P x a b c d < 0 ∧ Q x p q < 0)
  (h4 : ∀ x, x < r ∨ x > s → P x a b c d >= 0 ∧ Q x p q >= 0) :
  ∃ x0, r < x0 ∧ x0 < s ∧ P x0 a b c d < Q x0 p q :=
sorry

end exists_x0_lt_l291_291872


namespace number_of_unique_sums_of_two_cubes_less_than_1000_l291_291672

def is_perfect_cube_sum (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3

theorem number_of_unique_sums_of_two_cubes_less_than_1000 : 
  (Finset.filter (λ n, n < 1000 ∧ is_perfect_cube_sum n) (Finset.range 1000)).card = 47 :=
sorry

end number_of_unique_sums_of_two_cubes_less_than_1000_l291_291672


namespace shelves_used_l291_291293

def initial_books : ℕ := 86
def books_sold : ℕ := 37
def books_per_shelf : ℕ := 7
def remaining_books : ℕ := initial_books - books_sold
def shelves : ℕ := remaining_books / books_per_shelf

theorem shelves_used : shelves = 7 := by
  -- proof will go here
  sorry

end shelves_used_l291_291293


namespace cupcakes_gluten_nut_nonvegan_l291_291307

-- Definitions based on conditions
def total_cupcakes := 120
def gluten_free_cupcakes := total_cupcakes / 3
def vegan_cupcakes := total_cupcakes / 4
def nut_free_cupcakes := total_cupcakes / 5
def gluten_and_vegan_cupcakes := 15
def vegan_and_nut_free_cupcakes := 10

-- Defining the theorem to prove the main question
theorem cupcakes_gluten_nut_nonvegan : 
  total_cupcakes - ((gluten_free_cupcakes + (vegan_cupcakes - gluten_and_vegan_cupcakes)) - vegan_and_nut_free_cupcakes) = 65 :=
by sorry

end cupcakes_gluten_nut_nonvegan_l291_291307


namespace inequality_of_thirds_of_ordered_triples_l291_291826

variable (a1 a2 a3 b1 b2 b3 : ℝ)

theorem inequality_of_thirds_of_ordered_triples 
  (h1 : a1 ≤ a2) 
  (h2 : a2 ≤ a3) 
  (h3 : b1 ≤ b2)
  (h4 : b2 ≤ b3)
  (h5 : a1 + a2 + a3 = b1 + b2 + b3)
  (h6 : a1 * a2 + a2 * a3 + a1 * a3 = b1 * b2 + b2 * b3 + b1 * b3)
  (h7 : a1 ≤ b1) : 
  a3 ≤ b3 := 
by 
  sorry

end inequality_of_thirds_of_ordered_triples_l291_291826


namespace circle_tangent_line_l291_291696

theorem circle_tangent_line (a : ℝ) : 
  ∃ (a : ℝ), a = 2 ∨ a = -8 := 
by 
  sorry

end circle_tangent_line_l291_291696


namespace factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l291_291346

-- Given condition and question, prove equality for the first expression
theorem factorize_x4_minus_16y4 (x y : ℝ) :
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by sorry

-- Given condition and question, prove equality for the second expression
theorem factorize_minus_2a3_plus_12a2_minus_16a (a : ℝ) :
  -2 * a^3 + 12 * a^2 - 16 * a = -2 * a * (a - 2) * (a - 4) := 
by sorry

end factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l291_291346


namespace distribute_seedlings_l291_291819

noncomputable def box_contents : List ℕ := [28, 51, 135, 67, 123, 29, 56, 38, 79]

def total_seedlings (contents : List ℕ) : ℕ := contents.sum

def obtainable_by_sigmas (contents : List ℕ) (σs : List ℕ) : Prop :=
  ∃ groups : List (List ℕ),
    (groups.length = σs.length) ∧
    (∀ g ∈ groups, contents.contains g.sum) ∧
    (∀ g, g ∈ groups → g.sum ∈ σs)

theorem distribute_seedlings : 
  total_seedlings box_contents = 606 →
  obtainable_by_sigmas box_contents [202, 202, 202] ∧
  ∃ way1 way2 : List (List ℕ),
    (way1 ≠ way2) ∧
    (obtainable_by_sigmas box_contents [202, 202, 202]) :=
by
  sorry

end distribute_seedlings_l291_291819


namespace triangles_combined_area_is_96_l291_291267

noncomputable def combined_area_of_triangles : Prop :=
  let length_rectangle : ℝ := 6
  let width_rectangle : ℝ := 4
  let area_rectangle : ℝ := length_rectangle * width_rectangle
  let ratio_rectangle_to_first_triangle : ℝ := 2 / 5
  let area_first_triangle : ℝ := (5 / 2) * area_rectangle
  let x : ℝ := area_first_triangle / 5
  let base_second_triangle : ℝ := 8
  let height_second_triangle : ℝ := 9  -- calculated height based on the area ratio
  let area_second_triangle : ℝ := (base_second_triangle * height_second_triangle) / 2
  let combined_area : ℝ := area_first_triangle + area_second_triangle
  combined_area = 96

theorem triangles_combined_area_is_96 : combined_area_of_triangles := by
  sorry

end triangles_combined_area_is_96_l291_291267


namespace find_sqrt_abc_sum_l291_291547

theorem find_sqrt_abc_sum (a b c : ℝ) (h1 : b + c = 20) (h2 : c + a = 22) (h3 : a + b = 24) :
    Real.sqrt (a * b * c * (a + b + c)) = 206.1 := by
  sorry

end find_sqrt_abc_sum_l291_291547


namespace triangle_with_angle_ratio_is_right_triangle_l291_291218

theorem triangle_with_angle_ratio_is_right_triangle (x : ℝ) (h1 : 1 * x + 2 * x + 3 * x = 180) : 
  ∃ A B C : ℝ, A = x ∧ B = 2 * x ∧ C = 3 * x ∧ (A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end triangle_with_angle_ratio_is_right_triangle_l291_291218


namespace fifth_eq_l291_291077

theorem fifth_eq :
  (1 = 1) ∧
  (2 + 3 + 4 = 9) ∧
  (3 + 4 + 5 + 6 + 7 = 25) ∧
  (4 + 5 + 6 + 7 + 8 + 9 + 10 = 49) →
  5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 = 81 :=
by
  intros
  sorry

end fifth_eq_l291_291077


namespace find_x_find_a_l291_291053

-- Definitions based on conditions
def inversely_proportional (p q : ℕ) (k : ℕ) := p * q = k

-- Given conditions for (x, y)
def x1 : ℕ := 36
def y1 : ℕ := 4
def k1 : ℕ := x1 * y1 -- or 144
def y2 : ℕ := 9

-- Given conditions for (a, b)
def a1 : ℕ := 50
def b1 : ℕ := 5
def k2 : ℕ := a1 * b1 -- or 250
def b2 : ℕ := 10

-- Proof statements
theorem find_x (x : ℕ) : inversely_proportional x y2 k1 → x = 16 := by
  sorry

theorem find_a (a : ℕ) : inversely_proportional a b2 k2 → a = 25 := by
  sorry

end find_x_find_a_l291_291053


namespace solve_number_l291_291944

theorem solve_number :
  ∃ (M : ℕ), 
    (10 ≤ M ∧ M < 100) ∧ -- M is a two-digit number
    M % 2 = 1 ∧ -- M is odd
    M % 9 = 0 ∧ -- M is a multiple of 9
    let d₁ := M / 10, d₂ := M % 10 in -- digits of M
    d₁ * d₂ = (Nat.sqrt (d₁ * d₂))^2 := -- product of digits is a perfect square
begin
  use 99,
  split,
  { -- 10 ≤ 99 < 100
    exact and.intro (le_refl 99) (lt_add_one 99),
  },
  split,
  { -- 99 is odd
    exact nat.odd_iff.2 (nat.dvd_one.trans (nat.dvd_refl 2)),
  },
  split,
  { -- 99 is a multiple of 9
    exact nat.dvd_of_mod_eq_zero (by norm_num),
  },
  { -- product of digits is a perfect square
    let d₁ := 99 / 10,
    let d₂ := 99 % 10,
    have h : d₁ * d₂ = 9 * 9, by norm_num,
    rw h,
    exact (by norm_num : 81 = 9 ^ 2).symm
  }
end

end solve_number_l291_291944


namespace min_area_is_fifteen_l291_291622

variable (L W : ℕ)

def minimum_possible_area (L W : ℕ) : ℕ :=
  if L = 3 ∧ W = 5 then 3 * 5 else 0

theorem min_area_is_fifteen (hL : 3 ≤ L ∧ L ≤ 5) (hW : 5 ≤ W ∧ W ≤ 7) : 
  minimum_possible_area 3 5 = 15 := 
by
  sorry

end min_area_is_fifteen_l291_291622


namespace find_table_height_l291_291078

theorem find_table_height (b r g h : ℝ) (h1 : h + b - g = 111) (h2 : h + r - b = 80) (h3 : h + g - r = 82) : h = 91 := 
by
  sorry

end find_table_height_l291_291078


namespace fraction_of_number_l291_291125

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l291_291125


namespace series_150_result_l291_291559

noncomputable def series (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1).filter (λ k, k > 0), (-1)^k * ((k^3 + k^2 + k + 1) / (k.factorial : ℚ))

theorem series_150_result :
  ∃ d e f : ℕ, (series 150 = d / e.factorial - f) ∧ d + e + f = 45305 :=
by
  use 45154
  use 150
  use 1
  -- Here we need to prove series 150 = 45154 / 150! - 1
  -- and 45154 + 150 + 1 = 45305
  sorry  -- Proof is omitted as per the instructions.

end series_150_result_l291_291559


namespace trains_cross_time_l291_291300

def speed_in_m_per_s (speed_in_km_per_hr : Float) : Float :=
  (speed_in_km_per_hr * 1000) / 3600

def relative_speed (speed1 : Float) (speed2 : Float) : Float :=
  speed1 + speed2

def total_distance (length1 : Float) (length2 : Float) : Float :=
  length1 + length2

def time_to_cross (total_dist : Float) (relative_spd : Float) : Float :=
  total_dist / relative_spd

theorem trains_cross_time 
  (length_train1 : Float := 270)
  (speed_train1 : Float := 120)
  (length_train2 : Float := 230.04)
  (speed_train2 : Float := 80) :
  time_to_cross (total_distance length_train1 length_train2) 
                (relative_speed (speed_in_m_per_s speed_train1) 
                                (speed_in_m_per_s speed_train2)) = 9 := 
by
  sorry

end trains_cross_time_l291_291300


namespace arithmetic_sequence_sum_l291_291447

theorem arithmetic_sequence_sum :
  let a₁ := -5
  let aₙ := 40
  let n := 10
  (n : ℝ) = 10 →
  (a₁ : ℝ) = -5 →
  (aₙ : ℝ) = 40 →
  ∑ i in finset.range n, (a₁ + i * ((aₙ - a₁) / (n - 1))) = 175 :=
by
  intros
  sorry

end arithmetic_sequence_sum_l291_291447


namespace count_sum_of_cubes_lt_1000_l291_291666

theorem count_sum_of_cubes_lt_1000 : 
  {n : ℕ | n < 1000 ∧ ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3}.to_finset.card = 34 := 
sorry

end count_sum_of_cubes_lt_1000_l291_291666


namespace fraction_multiplication_l291_291121

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l291_291121


namespace inequality_not_less_than_four_by_at_least_one_l291_291260

-- Definitions based on the conditions
def not_less_than_by_at_least (y : ℝ) (a b : ℝ) : Prop := y - a ≥ b

-- Problem statement (theorem) based on the given question and correct answer
theorem inequality_not_less_than_four_by_at_least_one (y : ℝ) :
  not_less_than_by_at_least y 4 1 → y ≥ 5 :=
by
  sorry

end inequality_not_less_than_four_by_at_least_one_l291_291260


namespace binom_10_8_equals_45_l291_291329

theorem binom_10_8_equals_45 : Nat.choose 10 8 = 45 := 
by
  sorry

end binom_10_8_equals_45_l291_291329


namespace final_number_not_perfect_square_l291_291244

theorem final_number_not_perfect_square :
  (∃ final_number : ℕ, 
    ∀ a b : ℕ, a ∈ Finset.range 101 ∧ b ∈ Finset.range 101 ∧ a ≠ b → 
    gcd (a^2 + b^2 + 2) (a^2 * b^2 + 3) = final_number) →
  ∀ final_number : ℕ, ¬ ∃ k : ℕ, final_number = k ^ 2 :=
sorry

end final_number_not_perfect_square_l291_291244


namespace calculate_expression_l291_291011

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end calculate_expression_l291_291011


namespace f_2002_l291_291768

noncomputable def f : ℕ → ℝ := sorry

axiom f_condition (n : ℕ) (h : n > 1) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n ∧ f n = f (n / p) - f p

axiom f_2001 : f 2001 = 1

theorem f_2002 : f 2002 = 2 :=
  sorry

end f_2002_l291_291768


namespace tangent_parallel_points_l291_291381

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∃ (x0 y0 : ℝ), (curve x0 = y0) ∧ 
                 (deriv curve x0 = 4) ∧
                 ((x0 = 1 ∧ y0 = 0) ∨ (x0 = -1 ∧ y0 = -4)) :=
by
  sorry

end tangent_parallel_points_l291_291381


namespace albert_number_solution_l291_291178

theorem albert_number_solution (A B C : ℝ) 
  (h1 : A = 2 * B + 1) 
  (h2 : B = 2 * C + 1) 
  (h3 : C = 2 * A + 2) : 
  A = -11 / 7 := 
by 
  sorry

end albert_number_solution_l291_291178


namespace find_k_l291_291995

noncomputable def polynomial1 : Polynomial Int := sorry

theorem find_k :
  ∃ P : Polynomial Int,
  (P.eval 1 = 2013) ∧
  (P.eval 2013 = 1) ∧
  (∃ k : Int, P.eval k = k) →
  ∃ k : Int, P.eval k = k ∧ k = 1007 :=
by
  sorry

end find_k_l291_291995


namespace problem_solution_l291_291998

/-- Define proposition p: ∀α∈ℝ, sin(π-α) ≠ -sin(α) -/
def p := ∀ α : ℝ, Real.sin (Real.pi - α) ≠ -Real.sin α

/-- Define proposition q: ∃x∈[0,+∞), sin(x) > x -/
def q := ∃ x : ℝ, 0 ≤ x ∧ Real.sin x > x

/-- Prove that ¬p ∨ q is a true proposition -/
theorem problem_solution : ¬p ∨ q :=
by
  sorry

end problem_solution_l291_291998


namespace work_increase_percentage_l291_291062

theorem work_increase_percentage (p : ℕ) (hp : p > 0) : 
  let absent_fraction := 1 / 6
  let work_per_person_original := 1 / p
  let present_people := p - p * absent_fraction
  let work_per_person_new := 1 / present_people
  let work_increase := work_per_person_new - work_per_person_original
  let percentage_increase := (work_increase / work_per_person_original) * 100
  percentage_increase = 20 :=
by
  sorry

end work_increase_percentage_l291_291062


namespace find_rowing_speed_of_person_Y_l291_291277

open Real

def rowing_speed (y : ℝ) : Prop :=
  ∀ (x : ℝ) (current_speed : ℝ),
    x = 6 → 
    (4 * (6 - current_speed) + 4 * (y + current_speed) = 4 * (6 + y)) →
    (16 * (y + current_speed) = 16 * (6 + current_speed) + 4 * (y - 6)) → 
    y = 10

-- We set up the proof problem without solving it.
theorem find_rowing_speed_of_person_Y : ∃ (y : ℝ), rowing_speed y :=
begin
  use 10,
  unfold rowing_speed,
  intros x current_speed h1 h2 h3,

  sorry
end

end find_rowing_speed_of_person_Y_l291_291277


namespace number_of_numerators_repeating_decimal_l291_291233

theorem number_of_numerators_repeating_decimal (S : Set ℚ) (hS : ∀ r ∈ S, ∃ a b c : ℕ, 0 < r ∧ r < 1 ∧ r = (a * 100 + b * 10 + c) / 999 ∧ (r.denom = 999 ∨ (r.denom = 999 / 3 ∨ r.denom = 999 / 37))) : 
  ∃ n : ℕ, n = 660 :=
by
  use 660
  sorry

end number_of_numerators_repeating_decimal_l291_291233


namespace max_difference_intersection_ycoords_l291_291990

theorem max_difference_intersection_ycoords :
  let f₁ (x : ℝ) := 5 - 2 * x^2 + x^3
  let f₂ (x : ℝ) := 1 + x^2 + x^3
  let x1 := (2 : ℝ) / Real.sqrt 3
  let x2 := - (2 : ℝ) / Real.sqrt 3
  let y1 := f₁ x1
  let y2 := f₂ x2
  (f₁ = f₂)
  → abs (y1 - y2) = (16 * Real.sqrt 3 / 9) :=
by
  sorry

end max_difference_intersection_ycoords_l291_291990


namespace slope_problem_l291_291841

theorem slope_problem (m : ℝ) (h₀ : m > 0) (h₁ : (3 - m) = m * (1 - m)) : m = Real.sqrt 3 := by
  sorry

end slope_problem_l291_291841


namespace students_without_glasses_l291_291908

theorem students_without_glasses (total_students: ℕ) (percentage_with_glasses: ℕ) (p: percentage_with_glasses = 40) (t: total_students = 325) : ∃ x : ℕ, x = (total_students * (100 - percentage_with_glasses)) / 100 ∧ x = 195 :=
by
  have total_students := 325
  have percentage_with_glasses := 40
  have percentage_without_glasses := 100 - percentage_with_glasses
  have number_without_glasses := (total_students * percentage_without_glasses) / 100
  exact ⟨number_without_glasses, number_without_glasses, rfl⟩

end students_without_glasses_l291_291908


namespace fraction_simplification_l291_291285

theorem fraction_simplification : (3^2040 + 3^2038) / (3^2040 - 3^2038) = 5 / 4 :=
by
  sorry

end fraction_simplification_l291_291285


namespace highest_numbered_street_l291_291480

theorem highest_numbered_street (L : ℕ) (d : ℕ) (H : L = 15000 ∧ d = 500) : 
    (L / d) - 2 = 28 :=
by
  sorry

end highest_numbered_street_l291_291480


namespace circle_area_ratio_l291_291213

theorem circle_area_ratio (R_C R_D : ℝ)
  (h₁ : (60 / 360 * 2 * Real.pi * R_C) = (40 / 360 * 2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 9 / 4 :=
by 
  sorry

end circle_area_ratio_l291_291213


namespace optionB_is_difference_of_squares_l291_291143

-- Definitions from conditions
def A_expr (x : ℝ) : ℝ := (x - 2) * (x + 1)
def B_expr (x y : ℝ) : ℝ := (x + 2 * y) * (x - 2 * y)
def C_expr (x y : ℝ) : ℝ := (x + y) * (-x - y)
def D_expr (x : ℝ) : ℝ := (-x + 1) * (x - 1)

theorem optionB_is_difference_of_squares (x y : ℝ) : B_expr x y = x^2 - 4 * y^2 :=
by
  -- Proof is intentionally left out as per instructions
  sorry

end optionB_is_difference_of_squares_l291_291143


namespace g_triple_evaluation_l291_291877

def g (x : ℤ) : ℤ := 
if x < 8 then x ^ 2 - 6 
else x - 15

theorem g_triple_evaluation :
  g (g (g 20)) = 4 :=
by sorry

end g_triple_evaluation_l291_291877


namespace ratio_of_areas_l291_291204

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l291_291204


namespace initial_percentage_of_water_l291_291773

theorem initial_percentage_of_water (P : ℕ) : 
  (P / 100) * 120 + 54 = (3 / 4) * 120 → P = 30 :=
by 
  intro h
  sorry

end initial_percentage_of_water_l291_291773


namespace NOQZ_has_same_product_as_MNOQ_l291_291021

/-- Each letter of the alphabet is assigned a value (A=1, B=2, C=3, ..., Z=26). -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5 | 'F' => 6 | 'G' => 7
  | 'H' => 8 | 'I' => 9 | 'J' => 10 | 'K' => 11 | 'L' => 12 | 'M' => 13
  | 'N' => 14 | 'O' => 15 | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19
  | 'T' => 20 | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25 | 'Z' => 26
  | _   => 0  -- We'll assume only uppercase letters are inputs

/-- The product of a four-letter list is the product of the values of its four letters. -/
def list_product (lst : List Char) : ℕ :=
  lst.map letter_value |>.foldl (· * ·) 1

/-- The product of the list MNOQ is calculated. -/
def product_MNOQ : ℕ := list_product ['M', 'N', 'O', 'Q']
/-- The product of the list BEHK is calculated. -/
def product_BEHK : ℕ := list_product ['B', 'E', 'H', 'K']
/-- The product of the list NOQZ is calculated. -/
def product_NOQZ : ℕ := list_product ['N', 'O', 'Q', 'Z']

theorem NOQZ_has_same_product_as_MNOQ :
  product_NOQZ = product_MNOQ := by
  sorry

end NOQZ_has_same_product_as_MNOQ_l291_291021


namespace length_greater_than_width_l291_291735

theorem length_greater_than_width
  (perimeter : ℕ)
  (P : perimeter = 150)
  (l w difference : ℕ)
  (L : l = 60)
  (W : w = 45)
  (D : difference = l - w) :
  difference = 15 :=
by
  sorry

end length_greater_than_width_l291_291735


namespace beats_log_partition_l291_291921

noncomputable def log_base_2_10 : ℝ := real.log 10 / real.log 2
noncomputable def log_base_5_10 : ℝ := real.log 10 / real.log 5

theorem beats_log_partition (n : ℕ) (h : n > 1) :
  ∃ k : ℕ, (⌊(k * log_base_2_10 : ℝ)⌋ + 1 = n) ∨ (⌊(k * log_base_5_10 : ℝ)⌋ + 1 = n) :=
by sorry

end beats_log_partition_l291_291921


namespace simplify_expression_l291_291489

theorem simplify_expression (a : ℝ) (h : a / 2 - 2 / a = 3) : 
  (a^8 - 256) / (16 * a^4) * (2 * a) / (a^2 + 4) = 33 :=
by
  sorry

end simplify_expression_l291_291489


namespace find_certain_number_l291_291680

theorem find_certain_number (n x : ℤ) (h1 : 9 - n / x = 7 + 8 / x) (h2 : x = 6) : n = 8 := by
  sorry

end find_certain_number_l291_291680


namespace person_y_speed_in_still_water_l291_291281

theorem person_y_speed_in_still_water 
    (speed_x_in_still_water : ℝ)
    (time_meeting_towards_each_other : ℝ)
    (time_catching_up_same_direction: ℝ)
    (distance_upstream_meeting: ℝ)
    (distance_downstream_meeting: ℝ)
    (total_distance: ℝ) :
    speed_x_in_still_water = 6 →
    time_meeting_towards_each_other = 4 →
    time_catching_up_same_direction = 16 →
    distance_upstream_meeting = 4 * (6 - distance_upstream_meeting) + 4 * (10 + distance_downstream_meeting) →
    distance_downstream_meeting = 4 * (6 + distance_upstream_meeting) →
    total_distance = 4 * (6 + 10) →
    ∃ (speed_y_in_still_water : ℝ), speed_y_in_still_water = 10 :=
by
  intros h_speed_x h_time_meeting h_time_catching h_distance_upstream h_distance_downstream h_total_distance
  sorry

end person_y_speed_in_still_water_l291_291281


namespace area_of_path_is_675_l291_291765

def rectangular_field_length : ℝ := 75
def rectangular_field_width : ℝ := 55
def path_width : ℝ := 2.5

def area_of_path : ℝ :=
  let new_length := rectangular_field_length + 2 * path_width
  let new_width := rectangular_field_width + 2 * path_width
  let area_with_path := new_length * new_width
  let area_of_grass_field := rectangular_field_length * rectangular_field_width
  area_with_path - area_of_grass_field

theorem area_of_path_is_675 : area_of_path = 675 := by
  sorry

end area_of_path_is_675_l291_291765


namespace range_of_half_alpha_minus_beta_l291_291201

theorem range_of_half_alpha_minus_beta (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -3/2 < (1/2) * α - β ∧ (1/2) * α - β < 11/2 :=
by
  -- sorry to skip the proof
  sorry

end range_of_half_alpha_minus_beta_l291_291201


namespace sufficient_but_not_necessary_l291_291707

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1 / a^2) : a^2 > 1 / a ∧ ¬ ∀ a, a^2 > 1 / a → a > 1 / a^2 :=
by
  sorry

end sufficient_but_not_necessary_l291_291707


namespace smallest_number_of_cookies_proof_l291_291562

def satisfies_conditions (a : ℕ) : Prop :=
  (a % 6 = 5) ∧ (a % 8 = 6) ∧ (a % 10 = 9) ∧ (∃ n : ℕ, a = n * n)

def smallest_number_of_cookies : ℕ :=
  2549

theorem smallest_number_of_cookies_proof :
  satisfies_conditions smallest_number_of_cookies :=
by
  sorry

end smallest_number_of_cookies_proof_l291_291562


namespace no_prime_pair_summing_to_53_l291_291387

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l291_291387


namespace sqrt_20n_integer_exists_l291_291838

theorem sqrt_20n_integer_exists : 
  ∃ n : ℤ, 0 ≤ n ∧ ∃ k : ℤ, k * k = 20 * n :=
sorry

end sqrt_20n_integer_exists_l291_291838


namespace arithmetic_sequence_problem_l291_291064

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_problem 
  (h : is_arithmetic_sequence a)
  (h_cond : a 2 + 2 * a 6 + a 10 = 120) :
  a 3 + a 9 = 60 :=
sorry

end arithmetic_sequence_problem_l291_291064


namespace fraction_of_number_l291_291118

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l291_291118


namespace ratio_of_areas_of_circles_l291_291210

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l291_291210


namespace is_decreasing_on_interval_l291_291645

open Set Real

def f (x : ℝ) : ℝ := x^3 - x^2 - x

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem is_decreasing_on_interval :
  ∀ x ∈ Ioo (-1 / 3 : ℝ) 1, f' x < 0 :=
by
  intro x hx
  sorry

end is_decreasing_on_interval_l291_291645


namespace least_M_bench_sections_l291_291618

/--
A single bench section at a community event can hold either 8 adults, 12 children, or 10 teenagers. 
We are to find the smallest positive integer M such that when M bench sections are connected end to end,
an equal number of adults, children, and teenagers seated together will occupy all the bench space.
-/
theorem least_M_bench_sections
  (M : ℕ)
  (hM_pos : M > 0)
  (adults_capacity : ℕ := 8 * M)
  (children_capacity : ℕ := 12 * M)
  (teenagers_capacity : ℕ := 10 * M)
  (h_equal_capacity : adults_capacity = children_capacity ∧ children_capacity = teenagers_capacity) :
  M = 15 := 
sorry

end least_M_bench_sections_l291_291618


namespace lollipop_distribution_l291_291845

theorem lollipop_distribution :
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  (required_lollipops - initial_lollipops) = 253 :=
by
  let n := 42
  let initial_lollipops := 650
  let required_lollipops := n * (n + 1) / 2
  have h : required_lollipops = 903 := by norm_num
  have h2 : (required_lollipops - initial_lollipops) = 253 := by norm_num
  exact h2

end lollipop_distribution_l291_291845


namespace minimum_value_is_8_l291_291553

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_is_8 :
  ∃ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), minimum_value x y hx hy = 8 :=
by
  sorry

end minimum_value_is_8_l291_291553


namespace wheel_radius_increase_l291_291805

theorem wheel_radius_increase :
  let r := 18
  let distance_AB := 600   -- distance from A to B in miles
  let distance_BA := 582   -- distance from B to A in miles
  let circumference_orig := 2 * Real.pi * r
  let dist_per_rotation_orig := circumference_orig / 63360
  let rotations_orig := distance_AB / dist_per_rotation_orig
  let r' := ((distance_BA * dist_per_rotation_orig * 63360) / (2 * Real.pi * rotations_orig))
  ((r' - r) : ℝ) = 0.34 := by
  sorry

end wheel_radius_increase_l291_291805


namespace perfect_square_trinomial_l291_291052

theorem perfect_square_trinomial {m : ℝ} :
  (∃ (a : ℝ), x^2 + 2 * m * x + 9 = (x + a)^2) → (m = 3 ∨ m = -3) :=
sorry

end perfect_square_trinomial_l291_291052


namespace initial_punch_amount_l291_291712

-- Given conditions
def initial_punch : ℝ
def final_punch : ℝ := 16
def cousin_drink_half (x : ℝ) := x / 2
def mark_add (x : ℝ) := x + 4
def sally_drink (x : ℝ) := x - 2
def mark_final_addition := 12

-- Problem statement in Lean 4
theorem initial_punch_amount (initial_punch : ℝ) : 
  let after_final_addition := final_punch - mark_final_addition
  let before_sally_drink := after_final_addition + 2
  let before_second_refill := before_sally_drink - 4
  let initial_punch := cousin_drink_half (before_second_refill)
  initial_punch = 4 := 
sorry

end initial_punch_amount_l291_291712


namespace megan_homework_problems_l291_291404

theorem megan_homework_problems
  (finished_problems : ℕ)
  (pages_remaining : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ) :
  finished_problems = 26 →
  pages_remaining = 2 →
  problems_per_page = 7 →
  total_problems = finished_problems + (pages_remaining * problems_per_page) →
  total_problems = 40 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end megan_homework_problems_l291_291404


namespace stratified_sampling_11th_grade_representatives_l291_291463

theorem stratified_sampling_11th_grade_representatives 
  (students_10th : ℕ)
  (students_11th : ℕ)
  (students_12th : ℕ)
  (total_rep : ℕ)
  (total_students : students_10th + students_11th + students_12th = 5000)
  (Students_10th : students_10th = 2500)
  (Students_11th : students_11th = 1500)
  (Students_12th : students_12th = 1000)
  (Total_rep : total_rep = 30) : 
  (9 : ℕ) = (3 : ℚ) / (10 : ℚ) * (30 : ℕ) :=
sorry

end stratified_sampling_11th_grade_representatives_l291_291463


namespace fraction_multiplication_l291_291123

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l291_291123


namespace solve_for_m_l291_291579

def f (x : ℝ) (m : ℝ) := x^3 - m * x + 3

def f_prime (x : ℝ) (m : ℝ) := 3 * x^2 - m

theorem solve_for_m (m : ℝ) : f_prime 1 m = 0 → m = 3 :=
by
  sorry

end solve_for_m_l291_291579


namespace water_consumption_eq_l291_291809

-- Define all conditions
variables (x : ℝ) (improvement : ℝ := 0.8) (water : ℝ := 80) (days_difference : ℝ := 5)

-- State the theorem
theorem water_consumption_eq (h : improvement = 0.8) (initial_water := 80) (difference := 5) : 
  initial_water / x - (initial_water * improvement) / x = difference :=
sorry

end water_consumption_eq_l291_291809


namespace solve_fraction_equation_l291_291569

theorem solve_fraction_equation :
  ∀ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) → x = -2 / 11 :=
by
  intro x
  intro h
  sorry

end solve_fraction_equation_l291_291569


namespace find_f_neg4_l291_291038

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^2 - a * x + b

theorem find_f_neg4 (a b : ℝ) (h1 : f 1 a b = -1) (h2 : f 2 a b = 2) : 
  f (-4) a b = 14 :=
by
  sorry

end find_f_neg4_l291_291038


namespace min_employees_to_hire_l291_291778

-- Definitions of the given conditions
def employees_cust_service : ℕ := 95
def employees_tech_support : ℕ := 80
def employees_both : ℕ := 30

-- The theorem stating the minimum number of new employees to hire
theorem min_employees_to_hire (n : ℕ) :
  n = (employees_cust_service - employees_both) 
    + (employees_tech_support - employees_both) 
    + employees_both → 
  n = 145 := sorry

end min_employees_to_hire_l291_291778


namespace product_of_modified_numbers_less_l291_291112

theorem product_of_modified_numbers_less
  {a b c : ℝ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := 
by {
   sorry
}

end product_of_modified_numbers_less_l291_291112


namespace quad_eq_double_root_m_value_l291_291494

theorem quad_eq_double_root_m_value (m : ℝ) : 
  (∀ x : ℝ, x^2 + 6 * x + m = 0) → m = 9 := 
by 
  sorry

end quad_eq_double_root_m_value_l291_291494


namespace nine_chapters_problem_l291_291416

variable (m n : ℕ)

def horses_condition_1 : Prop := m + n = 100
def horses_condition_2 : Prop := 3 * m + n / 3 = 100

theorem nine_chapters_problem (h1 : horses_condition_1 m n) (h2 : horses_condition_2 m n) :
  (m + n = 100 ∧ 3 * m + n / 3 = 100) :=
by
  exact ⟨h1, h2⟩

end nine_chapters_problem_l291_291416


namespace decimal_to_fraction_l291_291437

-- Define the decimal number 2.75
def decimal_num : ℝ := 2.75

-- Define the expected fraction in unsimplified form
def unsimplified_fraction := 275 / 100

-- The greatest common divisor of 275 and 100
def gcd_275_100 : ℕ := 25

-- Define the simplified fraction as 11/4
def simplified_fraction := 11 / 4

-- Statement of the theorem to prove
theorem decimal_to_fraction : (decimal_num : ℚ) = simplified_fraction :=
by
  -- Here you can write the proof steps or use sorry to denote the proof is omitted
  sorry

end decimal_to_fraction_l291_291437


namespace index_card_area_l291_291880

theorem index_card_area
  (L W : ℕ)
  (h1 : L = 4)
  (h2 : W = 6)
  (h3 : (L - 1) * W = 18) :
  (L * (W - 1) = 20) :=
by
  sorry

end index_card_area_l291_291880


namespace symmetry_proof_l291_291527

-- Define the coordinates of point A
def A : ℝ × ℝ := (-1, 8)

-- Define the reflection property across the y-axis
def is_reflection_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

-- Define the point B which we need to prove
def B : ℝ × ℝ := (1, 8)

-- The proof statement
theorem symmetry_proof :
  is_reflection_y_axis A B :=
by
  sorry

end symmetry_proof_l291_291527


namespace hexagon_area_is_20_l291_291755

theorem hexagon_area_is_20 :
  let upper_base1 := 3
  let upper_base2 := 2
  let upper_height := 4
  let lower_base1 := 3
  let lower_base2 := 2
  let lower_height := 4
  let upper_trapezoid_area := (upper_base1 + upper_base2) * upper_height / 2
  let lower_trapezoid_area := (lower_base1 + lower_base2) * lower_height / 2
  let total_area := upper_trapezoid_area + lower_trapezoid_area
  total_area = 20 := 
by {
  sorry
}

end hexagon_area_is_20_l291_291755


namespace three_types_in_69_trees_l291_291848

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l291_291848


namespace find_principal_amount_l291_291756

noncomputable def principal_amount (SI : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  (SI * 100) / (R * T)

theorem find_principal_amount :
  principal_amount 130 4.166666666666667 4 = 780 :=
by
  -- Sorry is used to denote that the proof is yet to be provided
  sorry

end find_principal_amount_l291_291756


namespace intersection_eq_l291_291366

def A : Set ℝ := { x | abs x ≤ 2 }
def B : Set ℝ := { x | 3 * x - 2 ≥ 1 }

theorem intersection_eq :
  A ∩ B = { x | 1 ≤ x ∧ x ≤ 2 } :=
sorry

end intersection_eq_l291_291366


namespace shaded_area_between_circles_l291_291430

theorem shaded_area_between_circles (r1 r2 : ℝ) (h1 : r1 = 4) (h2 : r2 = 5)
  (tangent : True) -- This represents that the circles are externally tangent
  (circumscribed : True) -- This represents the third circle circumscribing the two circles
  : ∃ r3 : ℝ, r3 = 9 ∧ π * r3^2 - (π * r1^2 + π * r2^2) = 40 * π :=
  sorry

end shaded_area_between_circles_l291_291430


namespace smallest_n_for_multiple_of_7_l291_291091

theorem smallest_n_for_multiple_of_7 (x y : ℤ) (h1 : x % 7 = -1 % 7) (h2 : y % 7 = 2 % 7) :
  ∃ n : ℕ, n > 0 ∧ (x^2 + x * y + y^2 + n) % 7 = 0 ∧ n = 4 :=
sorry

end smallest_n_for_multiple_of_7_l291_291091


namespace biking_distance_l291_291314

/-- Mathematical equivalent proof problem for the distance biked -/
theorem biking_distance
  (x t d : ℕ)
  (h1 : d = (x + 1) * (3 * t / 4))
  (h2 : d = (x - 1) * (t + 3)) :
  d = 36 :=
by
  -- The proof goes here
  sorry

end biking_distance_l291_291314


namespace max_pages_l291_291970

/-- Prove that the maximum number of pages the book has is 208 -/
theorem max_pages (pages: ℕ) (h1: pages ≥ 16 * 12 + 1) (h2: pages ≤ 13 * 16) 
(h3: pages ≥ 20 * 10 + 1) (h4: pages ≤ 11 * 20) : 
  pages ≤ 208 :=
by
  -- proof to be filled in
  sorry

end max_pages_l291_291970


namespace cake_fractions_l291_291610

theorem cake_fractions (x y z : ℚ) 
  (h1 : x + y + z = 1)
  (h2 : 2 * z = x)
  (h3 : z = 1 / 2 * (y + 2 / 3 * x)) :
  x = 6 / 11 ∧ y = 2 / 11 ∧ z = 3 / 11 :=
sorry

end cake_fractions_l291_291610


namespace fib_divisibility_l291_291092

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

theorem fib_divisibility (m n : ℕ) (hm : 1 ≤ m) (hn : 1 < n) : 
  (fib (m * n - 1) - fib (n - 1) ^ m) % fib n ^ 2 = 0 :=
sorry

end fib_divisibility_l291_291092


namespace proof1_proof2_l291_291500

variable (a : ℝ) (m n : ℝ)
axiom am_eq_two : a^m = 2
axiom an_eq_three : a^n = 3

theorem proof1 : a^(4 * m + 3 * n) = 432 := by
  sorry

theorem proof2 : a^(5 * m - 2 * n) = 32 / 9 := by
  sorry

end proof1_proof2_l291_291500


namespace find_highest_score_l291_291598

theorem find_highest_score (average innings : ℕ) (avg_excl_two innings_excl_two H L : ℕ)
  (diff_high_low total_runs total_excl_two : ℕ)
  (h1 : diff_high_low = 150)
  (h2 : total_runs = average * innings)
  (h3 : total_excl_two = avg_excl_two * innings_excl_two)
  (h4 : total_runs - total_excl_two = H + L)
  (h5 : H - L = diff_high_low)
  (h6 : average = 62)
  (h7 : innings = 46)
  (h8 : avg_excl_two = 58)
  (h9 : innings_excl_two = 44)
  (h10 : total_runs = 2844)
  (h11 : total_excl_two = 2552) :
  H = 221 :=
by
  sorry

end find_highest_score_l291_291598


namespace find_percentage_l291_291054

variable (P : ℝ)

def percentage_condition (P : ℝ) : Prop :=
  P * 30 = (0.25 * 16) + 2

theorem find_percentage : percentage_condition P → P = 0.2 :=
by
  intro h
  -- Proof steps go here
  sorry

end find_percentage_l291_291054


namespace find_two_digit_number_l291_291940

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l291_291940


namespace convex_parallelogram_faces_1992_l291_291961

theorem convex_parallelogram_faces_1992 (n : ℕ) (h : n > 0) : (n * (n - 1) ≠ 1992) := 
by
  sorry

end convex_parallelogram_faces_1992_l291_291961


namespace shaded_area_ratio_l291_291529

noncomputable def ratio_of_shaded_area_to_circle_area (AB r : ℝ) : ℝ :=
  let AC := r
  let CB := 2 * r
  let radius_semicircle_AB := 3 * r / 2
  let area_semicircle_AB := (1 / 2) * (Real.pi * (radius_semicircle_AB ^ 2))
  let radius_semicircle_AC := r / 2
  let area_semicircle_AC := (1 / 2) * (Real.pi * (radius_semicircle_AC ^ 2))
  let radius_semicircle_CB := r
  let area_semicircle_CB := (1 / 2) * (Real.pi * (radius_semicircle_CB ^ 2))
  let total_area_semicircles := area_semicircle_AB + area_semicircle_AC + area_semicircle_CB
  let non_overlapping_area_semicircle_AB := area_semicircle_AB - (area_semicircle_AC + area_semicircle_CB)
  let shaded_area := non_overlapping_area_semicircle_AB
  let area_circle_CD := Real.pi * (r ^ 2)
  shaded_area / area_circle_CD

theorem shaded_area_ratio (AB r : ℝ) : ratio_of_shaded_area_to_circle_area AB r = 1 / 4 :=
by
  sorry

end shaded_area_ratio_l291_291529


namespace min_value_of_expression_l291_291678

theorem min_value_of_expression (x y : ℝ) (h1 : x > 1) (h2 : y > 1) (h3 : x + y = 3) : 
  ∃ k : ℝ, k = 4 + 2 * Real.sqrt 3 ∧ ∀ z, (z = (1 / (x - 1) + 3 / (y - 1))) → z ≥ k :=
sorry

end min_value_of_expression_l291_291678


namespace exp_mono_increasing_of_gt_l291_291654

variable {a b : ℝ}

theorem exp_mono_increasing_of_gt (h : a > b) : (2 : ℝ) ^ a > (2 : ℝ) ^ b :=
by sorry

end exp_mono_increasing_of_gt_l291_291654


namespace largest_divisor_l291_291521

theorem largest_divisor (x : ℤ) (hx : x % 2 = 1) : 180 ∣ (15 * x + 3) * (15 * x + 9) * (10 * x + 5) := 
by
  sorry

end largest_divisor_l291_291521


namespace two_digit_number_is_9_l291_291954

def dig_product (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n);
  match digits with
  | [a, b] => a * b
  | _ => 0

theorem two_digit_number_is_9 :
  ∃ (M : ℕ), 
    10 ≤ M ∧ M < 100 ∧ -- M is a two-digit number
    Odd M ∧            -- M is odd
    9 ∣ M ∧            -- M is a multiple of 9
    ∃ k, dig_product M = k * k -- product of its digits is a perfect square
    ∧ M = 9 :=       -- the solution is M = 9
by
  sorry

end two_digit_number_is_9_l291_291954


namespace part1_part2_l291_291704

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l291_291704


namespace line_intersects_y_axis_at_l291_291625

-- Define the two points the line passes through
structure Point (α : Type) :=
(x : α)
(y : α)

def p1 : Point ℤ := Point.mk 2 9
def p2 : Point ℤ := Point.mk 4 13

-- Define the function that describes the point where the line intersects the y-axis
def y_intercept : Point ℤ :=
  -- We are proving that the line intersects the y-axis at the point (0, 5)
  Point.mk 0 5

-- State the theorem to be proven
theorem line_intersects_y_axis_at (p1 p2 : Point ℤ) (yi : Point ℤ) :
  p1.x = 2 ∧ p1.y = 9 ∧ p2.x = 4 ∧ p2.y = 13 → yi = Point.mk 0 5 :=
by
  intros
  sorry

end line_intersects_y_axis_at_l291_291625


namespace yella_computer_usage_difference_l291_291594

-- Define the given conditions
def last_week_usage : ℕ := 91
def this_week_daily_usage : ℕ := 8
def days_in_week : ℕ := 7

-- Compute this week's total usage
def this_week_total_usage := this_week_daily_usage * days_in_week

-- Statement to prove
theorem yella_computer_usage_difference :
  last_week_usage - this_week_total_usage = 35 := 
by
  -- The proof will be filled in here
  sorry

end yella_computer_usage_difference_l291_291594


namespace ratio_of_areas_eq_l291_291207

-- Define the conditions
variables {C D : Type} [circle C] [circle D]
variables (R_C R_D : ℝ)
variable (L : ℝ)

-- Given conditions
axiom arc_length_eq : (60 / 360) * (2 * π * R_C) = L
axiom arc_length_eq' : (40 / 360) * (2 * π * R_D) = L

-- Statement to prove
theorem ratio_of_areas_eq : (π * R_C^2) / (π * R_D^2) = 4 / 9 :=
sorry

end ratio_of_areas_eq_l291_291207


namespace fraction_of_number_l291_291132

theorem fraction_of_number (a b : ℝ) (x : ℝ) (hx : x = 48) : (a/b) * x = 42 :=
by
  have ha : a = 7 := rfl
  have hb : b = 8 := rfl
  rw [ha, hb, hx]
  sorry

end fraction_of_number_l291_291132


namespace platform_length_l291_291959

theorem platform_length 
  (train_length : ℝ) (train_speed_kmph : ℝ) (time_s : ℝ) (platform_length : ℝ)
  (H1 : train_length = 360) 
  (H2 : train_speed_kmph = 45) 
  (H3 : time_s = 40)
  (H4 : platform_length = (train_speed_kmph * 1000 / 3600 * time_s) - train_length ) :
  platform_length = 140 :=
by {
 sorry
}

end platform_length_l291_291959


namespace at_least_one_lands_l291_291058

def p : Prop := sorry -- Proposition that Person A lands in the designated area
def q : Prop := sorry -- Proposition that Person B lands in the designated area

theorem at_least_one_lands : p ∨ q := sorry

end at_least_one_lands_l291_291058


namespace total_cookies_correct_l291_291964

noncomputable def cookies_monday : ℕ := 5
def cookies_tuesday := 2 * cookies_monday
def cookies_wednesday := cookies_tuesday + (40 * cookies_tuesday / 100)
def total_cookies := cookies_monday + cookies_tuesday + cookies_wednesday

theorem total_cookies_correct : total_cookies = 29 := by
  sorry

end total_cookies_correct_l291_291964


namespace johns_weekly_allowance_l291_291200

variable (A : ℝ)

theorem johns_weekly_allowance 
  (h1 : ∃ A : ℝ, A > 0) 
  (h2 : (4/15) * A = 0.75) : 
  A = 2.8125 := 
by 
  -- Proof can be filled in here
  sorry

end johns_weekly_allowance_l291_291200


namespace download_time_is_2_hours_l291_291975

theorem download_time_is_2_hours (internet_speed : ℕ) (f1 f2 f3 : ℕ) (total_size : ℕ)
  (total_min : ℕ) (hours : ℕ) :
  internet_speed = 2 ∧ f1 = 80 ∧ f2 = 90 ∧ f3 = 70 ∧ total_size = f1 + f2 + f3
  ∧ total_min = total_size / internet_speed ∧ hours = total_min / 60 → hours = 2 :=
by
  sorry

end download_time_is_2_hours_l291_291975


namespace market_price_article_l291_291219

theorem market_price_article (P : ℝ)
  (initial_tax_rate : ℝ := 0.035)
  (reduced_tax_rate : ℝ := 0.033333333333333)
  (difference_in_tax : ℝ := 11) :
  (initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax) → 
  P = 6600 :=
by
  intro h
  /-
  We assume h: initial_tax_rate * P - reduced_tax_rate * P = difference_in_tax
  And we need to show P = 6600.
  The proof steps show that P = 6600 follows logically given h and the provided conditions.
  -/
  sorry

end market_price_article_l291_291219


namespace ladybird_routes_l291_291615

def num_routes (A X B: Type) (AX_paths: A → X) (XB_paths: X → B) (AX_round_trip_paths: X → X) : ℕ :=
  (3 + 3 * 2) * 3

theorem ladybird_routes : num_routes A X B AX_paths XB_paths AX_round_trip_paths = 27 := by
  show (3 + 6) * 3 = 27
  sorry

end ladybird_routes_l291_291615


namespace two_digit_number_is_9_l291_291955

def dig_product (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n);
  match digits with
  | [a, b] => a * b
  | _ => 0

theorem two_digit_number_is_9 :
  ∃ (M : ℕ), 
    10 ≤ M ∧ M < 100 ∧ -- M is a two-digit number
    Odd M ∧            -- M is odd
    9 ∣ M ∧            -- M is a multiple of 9
    ∃ k, dig_product M = k * k -- product of its digits is a perfect square
    ∧ M = 9 :=       -- the solution is M = 9
by
  sorry

end two_digit_number_is_9_l291_291955


namespace find_number_250_l291_291139

theorem find_number_250 (N : ℤ)
  (h1 : 5 * N = 8 * 156 + 2): N = 250 :=
sorry

end find_number_250_l291_291139


namespace emma_bank_account_balance_l291_291344

def initial_amount : ℝ := 230
def withdrawn_amount : ℝ := 60
def deposit_amount : ℝ := 2 * withdrawn_amount
def final_amount : ℝ := initial_amount - withdrawn_amount + deposit_amount

theorem emma_bank_account_balance : final_amount = 290 := 
by 
  -- Definitions have already been stated; the proof is not required
  sorry

end emma_bank_account_balance_l291_291344


namespace parabola_midpoint_locus_minimum_slope_difference_exists_l291_291824

open Real

def parabola_locus (x y : ℝ) : Prop :=
  x^2 = 4 * y

def slope_difference_condition (x1 x2 k1 k2 : ℝ) : Prop :=
  |k1 - k2| = 1

theorem parabola_midpoint_locus :
  ∀ (x y : ℝ), parabola_locus x y :=
by
  intros x y
  apply sorry

theorem minimum_slope_difference_exists :
  ∀ {x1 y1 x2 y2 k1 k2 : ℝ},
  slope_difference_condition x1 x2 k1 k2 :=
by
  intros x1 y1 x2 y2 k1 k2
  apply sorry

end parabola_midpoint_locus_minimum_slope_difference_exists_l291_291824


namespace proof_f_g_l291_291506

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 1
def g (x : ℝ) : ℝ := 2*x + 3

theorem proof_f_g (x : ℝ) : f (g 2) - g (f 2) = 258 :=
by
  sorry

end proof_f_g_l291_291506


namespace min_distance_l291_291198

noncomputable def curve_one (t : ℝ) : ℝ × ℝ :=
(2 + Real.cos t, Real.sin t - 1)

noncomputable def curve_two (α : ℝ) : ℝ × ℝ :=
(4 * Real.cos α, Real.sin α)

def line_three : ℝ × ℝ → Prop :=
λ p, p.1 = p.2

def P : ℝ × ℝ := (0, 1)

noncomputable def Q (t : ℝ) : ℝ × ℝ :=
(2 + Real.cos t, Real.sin t - 1)

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
((P.1 + Q.1)/2, (P.2 + Q.2)/2)

noncomputable def distance_to_line (M : ℝ × ℝ) : ℝ :=
Float.abs ((1 / Real.sqrt 2) * M.2 - 1) / Real.sqrt 2

theorem min_distance (t : ℝ) (h : Real.sin (t - (Real.pi / 4)) = 1) :
  distance_to_line (midpoint P (Q t)) = (Real.sqrt 2 - 1) / Real.sqrt 2 :=
sorry

end min_distance_l291_291198


namespace marco_paints_8_15_in_32_minutes_l291_291375

-- Define the rates at which Marco and Carla paint
def marco_rate : ℚ := 1 / 60
def combined_rate : ℚ := 1 / 40

-- Define the function to calculate the fraction of the room painted by Marco alone in a given time
def fraction_painted_by_marco (time: ℚ) : ℚ := time * marco_rate

-- State the theorem to prove
theorem marco_paints_8_15_in_32_minutes :
  (marco_rate + (combined_rate - marco_rate) = combined_rate) →
  fraction_painted_by_marco 32 = 8 / 15 := by
  sorry

end marco_paints_8_15_in_32_minutes_l291_291375


namespace express_y_in_terms_of_x_l291_291361

theorem express_y_in_terms_of_x (x y : ℝ) (h : y - 2 * x = 6) : y = 2 * x + 6 :=
by
  sorry

end express_y_in_terms_of_x_l291_291361


namespace student_A_more_stable_performance_l291_291113

theorem student_A_more_stable_performance
    (mean : ℝ)
    (n : ℕ)
    (variance_A variance_B : ℝ)
    (h1 : mean = 1.6)
    (h2 : n = 10)
    (h3 : variance_A = 1.4)
    (h4 : variance_B = 2.5) :
    variance_A < variance_B :=
by {
    -- The proof is omitted as we are only writing the statement here.
    sorry
}

end student_A_more_stable_performance_l291_291113


namespace cos_value_l291_291041

theorem cos_value {α : ℝ} (h : Real.sin (π / 6 + α) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 := 
by sorry

end cos_value_l291_291041


namespace cookies_per_kid_l291_291402

theorem cookies_per_kid (total_calories_per_lunch : ℕ) (burger_calories : ℕ) (carrot_calories_per_stick : ℕ) (num_carrot_sticks : ℕ) (cookie_calories : ℕ) (num_cookies : ℕ) : 
  total_calories_per_lunch = 750 →
  burger_calories = 400 →
  carrot_calories_per_stick = 20 →
  num_carrot_sticks = 5 →
  cookie_calories = 50 →
  num_cookies = (total_calories_per_lunch - (burger_calories + num_carrot_sticks * carrot_calories_per_stick)) / cookie_calories →
  num_cookies = 5 :=
by
  sorry

end cookies_per_kid_l291_291402


namespace boat_speed_greater_than_stream_l291_291254

def boat_stream_speed_difference (S U V : ℝ) := 
  (S / (U - V)) - (S / (U + V)) + (S / (2 * V + 1)) = 1

theorem boat_speed_greater_than_stream 
  (S : ℝ) (U V : ℝ) 
  (h_dist : S = 1) 
  (h_time_diff : boat_stream_speed_difference S U V) :
  U - V = 1 :=
sorry

end boat_speed_greater_than_stream_l291_291254


namespace three_types_in_69_trees_l291_291847

variable (birches spruces pines aspens : ℕ)
variable (total_trees : ℕ := 100)
variable (all_trees : list (string × ℕ))

-- We assert that there are 100 trees in total, and our list of trees represents this
axiom h_total : ∑ t in all_trees, t.2 = total_trees

-- Among any 85 trees, there must be at least one of each type
axiom h_85_trees_all_types : ∀ (s : list (string × ℕ)), s.card = 85 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0 ∧ a > 0 ∧ 
  b + s + p + a = 85 ∧ (("birches", b) ∈ s ∧ ("spruces", s) ∈ s ∧ ("pines", p) ∈ s ∧ ("aspens", a) ∈ s))

-- We need to prove that any subset of 69 or more trees contains at least three different types.
theorem three_types_in_69_trees :
  ∀ (s : list (string × ℕ)), s.card = 69 → ∃ (b s p a : ℕ), (b > 0 ∧ s > 0 ∧ p > 0) ∨ (b > 0 ∧ s > 0 ∧ a > 0) ∨ (b > 0 ∧ p > 0 ∧ a > 0) ∨ (s > 0 ∧ p > 0 ∧ a > 0) :=
begin
  sorry
end

end three_types_in_69_trees_l291_291847


namespace change_combinations_50_cents_l291_291518

-- Define the conditions for creating 50 cents using standard coins
def ways_to_make_change (pennies nickels dimes : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes

theorem change_combinations_50_cents : 
  ∃ num_ways, 
    num_ways = 28 ∧
    ∀ (pennies nickels dimes : ℕ), 
      pennies + 5 * nickels + 10 * dimes = 50 → 
      -- Exclude using only a single half-dollar
      ¬(num_ways = if (pennies = 0 ∧ nickels = 0 ∧ dimes = 0) then 1 else 28) := 
sorry

end change_combinations_50_cents_l291_291518


namespace parallel_vectors_l291_291833

theorem parallel_vectors {m : ℝ} 
  (h : (2 * m + 1) / 2 = 3 / m): m = 3 / 2 ∨ m = -2 :=
by
  sorry

end parallel_vectors_l291_291833


namespace calculate_expression_l291_291014

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 :=
  by
  sorry

end calculate_expression_l291_291014


namespace fruit_seller_apples_l291_291932

theorem fruit_seller_apples : 
  ∃ (x : ℝ), (x * 0.6 = 420) → x = 700 :=
sorry

end fruit_seller_apples_l291_291932


namespace bowls_remaining_l291_291481

def initial_bowls : ℕ := 250

def customers_purchases : List (ℕ × ℕ) :=
  [(5, 7), (10, 15), (15, 22), (5, 36), (7, 46), (8, 0)]

def reward_ranges (bought : ℕ) : ℕ :=
  if bought >= 5 && bought <= 9 then 1
  else if bought >= 10 && bought <= 19 then 3
  else if bought >= 20 && bought <= 29 then 6
  else if bought >= 30 && bought <= 39 then 8
  else if bought >= 40 then 12
  else 0

def total_free_bowls : ℕ :=
  List.foldl (λ acc (n, b) => acc + n * reward_ranges b) 0 customers_purchases

theorem bowls_remaining :
  initial_bowls - total_free_bowls = 1 := by
  sorry

end bowls_remaining_l291_291481


namespace arithmetic_sequence_a6_l291_291227

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : a 6 = 16 :=
sorry

end arithmetic_sequence_a6_l291_291227


namespace average_death_rate_l291_291384

-- Definitions and given conditions
def birth_rate_per_two_seconds := 6
def net_increase_per_day := 172800

-- Calculate number of seconds in a day as a constant
def seconds_per_day : ℕ := 24 * 60 * 60

-- Define the net increase per second
def net_increase_per_second : ℕ := net_increase_per_day / seconds_per_day

-- Define the birth rate per second
def birth_rate_per_second : ℕ := birth_rate_per_two_seconds / 2

-- The final proof statement
theorem average_death_rate : 
  ∃ (death_rate_per_two_seconds : ℕ), 
    death_rate_per_two_seconds = birth_rate_per_two_seconds - 2 * net_increase_per_second := 
by 
  -- We are required to prove this statement
  use (birth_rate_per_second - net_increase_per_second) * 2
  sorry

end average_death_rate_l291_291384


namespace min_distance_l291_291063

noncomputable def point_on_curve (x₁ y₁ : ℝ) : Prop :=
  y₁ = x₁^2 - Real.log x₁

noncomputable def point_on_line (x₂ y₂ : ℝ) : Prop :=
  x₂ - y₂ - 2 = 0

theorem min_distance 
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : point_on_curve x₁ y₁)
  (h₂ : point_on_line x₂ y₂) 
  : (x₂ - x₁)^2 + (y₂ - y₁)^2 = 2 :=
sorry

end min_distance_l291_291063


namespace min_value_of_frac_sum_l291_291503

theorem min_value_of_frac_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 2) :
  (1 / a + 2 / b) = 9 / 2 :=
sorry

end min_value_of_frac_sum_l291_291503


namespace proof1_proof2_l291_291499

variable (a : ℝ) (m n : ℝ)
axiom am_eq_two : a^m = 2
axiom an_eq_three : a^n = 3

theorem proof1 : a^(4 * m + 3 * n) = 432 := by
  sorry

theorem proof2 : a^(5 * m - 2 * n) = 32 / 9 := by
  sorry

end proof1_proof2_l291_291499


namespace solve_for_difference_l291_291039

variable (a b : ℝ)

theorem solve_for_difference (h1 : a^3 - b^3 = 4) (h2 : a^2 + ab + b^2 + a - b = 4) : a - b = 2 :=
sorry

end solve_for_difference_l291_291039


namespace probability_individual_selected_l291_291996

/-- Given a population of 8 individuals, the probability that each 
individual is selected in a simple random sample of size 4 is 1/2. -/
theorem probability_individual_selected :
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  probability = (1 : ℚ) / 2 :=
by
  let population_size := 8
  let sample_size := 4
  let probability := sample_size / population_size
  sorry

end probability_individual_selected_l291_291996


namespace parallelogram_area_l291_291924

theorem parallelogram_area (base height : ℝ) (h_base : base = 25) (h_height : height = 15) :
  base * height = 375 :=
by
  subst h_base
  subst h_height
  sorry

end parallelogram_area_l291_291924


namespace solve_inequality_l291_291155

theorem solve_inequality (x : ℝ) (h : 5 * x - 12 ≤ 2 * (4 * x - 3)) : x ≥ -2 :=
sorry

end solve_inequality_l291_291155


namespace solve_number_l291_291947

theorem solve_number :
  ∃ (M : ℕ), 
    (10 ≤ M ∧ M < 100) ∧ -- M is a two-digit number
    M % 2 = 1 ∧ -- M is odd
    M % 9 = 0 ∧ -- M is a multiple of 9
    let d₁ := M / 10, d₂ := M % 10 in -- digits of M
    d₁ * d₂ = (Nat.sqrt (d₁ * d₂))^2 := -- product of digits is a perfect square
begin
  use 99,
  split,
  { -- 10 ≤ 99 < 100
    exact and.intro (le_refl 99) (lt_add_one 99),
  },
  split,
  { -- 99 is odd
    exact nat.odd_iff.2 (nat.dvd_one.trans (nat.dvd_refl 2)),
  },
  split,
  { -- 99 is a multiple of 9
    exact nat.dvd_of_mod_eq_zero (by norm_num),
  },
  { -- product of digits is a perfect square
    let d₁ := 99 / 10,
    let d₂ := 99 % 10,
    have h : d₁ * d₂ = 9 * 9, by norm_num,
    rw h,
    exact (by norm_num : 81 = 9 ^ 2).symm
  }
end

end solve_number_l291_291947


namespace width_of_wall_l291_291274

theorem width_of_wall (l : ℕ) (w : ℕ) (hl : l = 170) (hw : w = 5 * l + 80) : w = 930 := 
by
  sorry

end width_of_wall_l291_291274


namespace min_trees_include_three_types_l291_291866

noncomputable def minNumTrees (T : Type) (tree_counts : T → ℕ) :=
  ∀ (total_trees : ℕ) (at_least : ℕ → Prop),
    total_trees = 100 →
    (∀ S : Finset T, S.card = 85 → (∃ t ∈ S, at_least (tree_counts t))) →
    (at_least ThreeTypes ↔ ∃ (S : Finset T), S.card = 69)

def ThreeTypes (tree_counts : T → ℕ) := 
  ∀ (birches spruces pines aspens : ℕ),
    birches + spruces + pines + aspens = 100 →
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ pines ≥ 1) ∨ 
    (birches ≥ 1 ∧ spruces ≥ 1 ∧ aspens ≥ 1) ∨ 
    (birches ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1) ∨ 
    (spruces ≥ 1 ∧ pines ≥ 1 ∧ aspens ≥ 1)

theorem min_trees_include_three_types :
  ∃ (T : Type) (tree_counts : T → ℕ), minNumTrees T tree_counts := 
sorry

end min_trees_include_three_types_l291_291866


namespace fraction_of_number_l291_291116

theorem fraction_of_number (x y : ℝ) (h : x = 7/8) (z : ℝ) (h' : z = 48) : 
  x * z = 42 := by
  sorry

end fraction_of_number_l291_291116


namespace part1_part2_part3_l291_291408

-- Part 1
theorem part1 (a b : ℝ) : 3*(a-b)^2 - 6*(a-b)^2 + 2*(a-b)^2 = -(a-b)^2 :=
sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x^2 - 2*y = 4) : 3*x^2 - 6*y - 21 = -9 :=
sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5*b = 3) (h2 : 5*b - 3*c = -5) (h3 : 3*c - d = 10) : 
  (a - 3*c) + (5*b - d) - (5*b - 3*c) = 8 :=
sorry

end part1_part2_part3_l291_291408


namespace mean_greater_than_median_by_l291_291357

-- Define the data: number of students missing specific days
def studentsMissingDays := [3, 1, 4, 1, 1, 5] -- corresponding to 0, 1, 2, 3, 4, 5 days missed

-- Total number of students
def totalStudents := 15

-- Function to calculate the sum of missed days weighted by the number of students
def totalMissedDays := (0 * 3) + (1 * 1) + (2 * 4) + (3 * 1) + (4 * 1) + (5 * 5)

-- Calculate the mean number of missed days
def meanDaysMissed := totalMissedDays / totalStudents

-- Select the median number of missed days (8th student) from the ordered list
def medianDaysMissed := 2

-- Calculate the difference between the mean and median
def difference := meanDaysMissed - medianDaysMissed

-- Define the proof problem statement
theorem mean_greater_than_median_by : 
  difference = 11 / 15 :=
by
  -- This is where the actual proof would be written
  sorry

end mean_greater_than_median_by_l291_291357


namespace verify_first_rope_length_l291_291429

def length_first_rope : ℝ :=
  let rope1_len := 20
  let rope2_len := 2
  let rope3_len := 2
  let rope4_len := 2
  let rope5_len := 7
  let knots := 4
  let knot_loss := 1.2
  let total_len := 35
  rope1_len

theorem verify_first_rope_length : length_first_rope = 20 := by
  sorry

end verify_first_rope_length_l291_291429


namespace calculate_expression_l291_291012

theorem calculate_expression : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end calculate_expression_l291_291012


namespace prob_students_both_days_l291_291912

def num_scenarios (students : ℕ) (choices : ℕ) : ℕ :=
  choices ^ students

def scenarios_sat_sun (total_scenarios : ℕ) (both_days_empty : ℕ) : ℕ :=
  total_scenarios - both_days_empty

theorem prob_students_both_days :
  let students := 3
  let choices := 2
  let total_scenarios := num_scenarios students choices
  let both_days_empty := 2 -- When all choose Saturday or all choose Sunday
  let scenarios_both := scenarios_sat_sun total_scenarios both_days_empty
  let probability := scenarios_both / total_scenarios
  probability = 3 / 4 :=
by
  sorry

end prob_students_both_days_l291_291912


namespace sum_of_two_cubes_count_l291_291667

theorem sum_of_two_cubes_count :
  let cubes := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 9 ∧ n = k^3},
      sums := {m | ∃ a b ∈ cubes, m = a + b ∧ m < 1000} in
  sums.size = 44 :=
by
  -- proof goes here, but it's skipped
  sorry

end sum_of_two_cubes_count_l291_291667


namespace cone_volume_270_degree_sector_l291_291303

noncomputable def coneVolumeDividedByPi (R θ: ℝ) (r h: ℝ) (circumf sector_height: ℝ) : ℝ := 
  if R = 20 
  ∧ θ = 270 / 360 
  ∧ 2 * Mathlib.pi * 20 = 40 * Mathlib.pi 
  ∧ circumf = 30 * Mathlib.pi
  ∧ 2 * Mathlib.pi * r = circumf
  ∧ r = 15
  ∧ sector_height = R
  ∧ r^2 + h^2 = sector_height^2 
  then (1/3) * Mathlib.pi * r^2 * h / Mathlib.pi 
  else 0

theorem cone_volume_270_degree_sector : coneVolumeDividedByPi 20 (270 / 360) 15 (5 * Real.sqrt 7) (30 * Mathlib.pi) 20 = 1125 * Real.sqrt 7 := 
by {
  -- This is where the proof would go
  sorry
}

end cone_volume_270_degree_sector_l291_291303


namespace bank_account_balance_l291_291338

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end bank_account_balance_l291_291338


namespace system_solution_in_first_quadrant_l291_291234

theorem system_solution_in_first_quadrant (c x y : ℝ)
  (h1 : x - y = 5)
  (h2 : c * x + y = 7)
  (hx : x > 3)
  (hy : y > 1) : c < 1 :=
sorry

end system_solution_in_first_quadrant_l291_291234


namespace emily_and_eli_probability_l291_291984

noncomputable def probability_same_number : ℚ :=
  let count_multiples (n k : ℕ) := (k - 1) / n
  let emily_count := count_multiples 20 250
  let eli_count := count_multiples 30 250
  let common_lcm := Nat.lcm 20 30
  let common_count := count_multiples common_lcm 250
  common_count / (emily_count * eli_count : ℚ)

theorem emily_and_eli_probability :
  let probability := probability_same_number
  probability = 1 / 24 :=
by
  sorry

end emily_and_eli_probability_l291_291984


namespace problem1_problem2_l291_291154

-- Problem 1: Simplification and Evaluation
theorem problem1 (x : ℝ) : (x = -3) → 
  ((x^2 - 6*x + 9) / (x^2 - 1)) / ((x^2 - 3*x) / (x + 1))
  = -1 / 2 := sorry

-- Problem 2: Solving the Equation
theorem problem2 (x : ℝ) : 
  (∀ y, (y = x) → 
    (y / (y + 1) = 2*y / (3*y + 3) - 1)) → x = -3 / 4 := sorry

end problem1_problem2_l291_291154


namespace average_of_11_results_l291_291417

theorem average_of_11_results (a b c : ℝ) (avg_first_6 avg_last_6 sixth_result avg_all_11 : ℝ)
  (h1 : avg_first_6 = 58)
  (h2 : avg_last_6 = 63)
  (h3 : sixth_result = 66) :
  avg_all_11 = 60 :=
by
  sorry

end average_of_11_results_l291_291417


namespace problem1_problem2_l291_291653

-- Problem (1)
theorem problem1 (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : ∀ n, a n = a (n + 1) + 3) : a 10 = -23 :=
by {
  sorry
}

-- Problem (2)
theorem problem2 (a : ℕ → ℚ) (h1 : a 6 = (1 / 4)) (h2 : ∃ d : ℚ, ∀ n, 1 / a n = 1 / a 1 + (n - 1) * d) : 
  ∀ n, a n = (4 / (3 * n - 2)) :=
by {
  sorry
}

end problem1_problem2_l291_291653


namespace probability_first_green_then_blue_l291_291159

variable {α : Type} [Fintype α]

noncomputable def prob_first_green_second_blue : ℚ := 
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := (green_marbles : ℚ) / total_marbles
  let prob_second_blue := (blue_marbles : ℚ) / (total_marbles - 1)
  (prob_first_green * prob_second_blue)

theorem probability_first_green_then_blue :
  prob_first_green_second_blue = 4 / 15 := by
  sorry

end probability_first_green_then_blue_l291_291159


namespace min_guests_l291_291420

theorem min_guests (total_food : ℕ) (max_food : ℝ) 
  (H1 : total_food = 337) 
  (H2 : max_food = 2) : 
  ∃ n : ℕ, n = ⌈total_food / max_food⌉ ∧ n = 169 :=
by
  sorry

end min_guests_l291_291420


namespace sheila_hourly_rate_is_6_l291_291148

variable (weekly_earnings : ℕ) (hours_mwf : ℕ) (days_mwf : ℕ) (hours_tt: ℕ) (days_tt : ℕ)
variable [NeZero hours_mwf] [NeZero days_mwf] [NeZero hours_tt] [NeZero days_tt]

-- Define Sheila's working hours and weekly earnings as given conditions
def weekly_hours := (hours_mwf * days_mwf) + (hours_tt * days_tt)
def hourly_rate := weekly_earnings / weekly_hours

-- Specific values from the given problem
def sheila_weekly_earnings : ℕ := 216
def sheila_hours_mwf : ℕ := 8
def sheila_days_mwf : ℕ := 3
def sheila_hours_tt : ℕ := 6
def sheila_days_tt : ℕ := 2

-- The theorem to prove
theorem sheila_hourly_rate_is_6 :
  (sheila_weekly_earnings / ((sheila_hours_mwf * sheila_days_mwf) + (sheila_hours_tt * sheila_days_tt))) = 6 := by
  sorry

end sheila_hourly_rate_is_6_l291_291148


namespace volume_of_ABDH_is_4_3_l291_291019

-- Define the vertices of the cube
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (2, 0, 0)
def D : (ℝ × ℝ × ℝ) := (0, 2, 0)
def H : (ℝ × ℝ × ℝ) := (0, 0, 2)

-- Function to calculate the volume of the pyramid
noncomputable def volume_of_pyramid (A B D H : ℝ × ℝ × ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * 2 * 2 * 2

-- Theorem stating the volume of the pyramid ABDH is 4/3 cubic units
theorem volume_of_ABDH_is_4_3 : volume_of_pyramid A B D H = 4 / 3 := by
  sorry

end volume_of_ABDH_is_4_3_l291_291019


namespace no_solution_inequality_l291_291843

theorem no_solution_inequality (m x : ℝ) (h1 : x - 2 * m < 0) (h2 : x + m > 2) : m ≤ 2 / 3 :=
  sorry

end no_solution_inequality_l291_291843


namespace find_t_l291_291565

theorem find_t (t : ℚ) : 
  ((t + 2) * (3 * t - 2) = (3 * t - 4) * (t + 1) + 5) → t = 5 / 3 :=
by
  intro h
  sorry

end find_t_l291_291565


namespace original_surface_area_l291_291682

theorem original_surface_area (R : ℝ) (h : 2 * π * R^2 = 4 * π) : 4 * π * R^2 = 8 * π :=
by
  sorry

end original_surface_area_l291_291682


namespace count_unique_sums_of_cubes_l291_291665

theorem count_unique_sums_of_cubes : 
  let sums := {n | ∃ a b, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ n = a^3 + b^3 ∧ n < 1000} in
  sums.card = 42 :=
by sorry

end count_unique_sums_of_cubes_l291_291665


namespace number_of_days_woman_weaves_l291_291414

theorem number_of_days_woman_weaves
  (a_1 : ℝ) (a_n : ℝ) (S_n : ℝ) (n : ℝ)
  (h1 : a_1 = 5)
  (h2 : a_n = 1)
  (h3 : S_n = 90)
  (h4 : S_n = n * (a_1 + a_n) / 2) :
  n = 30 :=
by
  rw [h1, h2, h3] at h4
  sorry

end number_of_days_woman_weaves_l291_291414


namespace index_card_area_l291_291881

theorem index_card_area
  (L W : ℕ)
  (h1 : L = 4)
  (h2 : W = 6)
  (h3 : (L - 1) * W = 18) :
  (L * (W - 1) = 20) :=
by
  sorry

end index_card_area_l291_291881


namespace mingyu_change_l291_291882

theorem mingyu_change :
  let eraser_cost := 350
  let pencil_cost := 180
  let erasers_count := 3
  let pencils_count := 2
  let payment := 2000
  let total_eraser_cost := erasers_count * eraser_cost
  let total_pencil_cost := pencils_count * pencil_cost
  let total_cost := total_eraser_cost + total_pencil_cost
  let change := payment - total_cost
  change = 590 := 
by
  -- The proof will go here
  sorry

end mingyu_change_l291_291882


namespace average_weight_of_remaining_students_l291_291844

theorem average_weight_of_remaining_students
  (M F M' F' : ℝ) (A A' : ℝ)
  (h1 : M + F = 60 * A)
  (h2 : M' + F' = 59 * A')
  (h3 : A' = A + 0.2)
  (h4 : M' = M - 45):
  A' = 57 :=
by
  sorry

end average_weight_of_remaining_students_l291_291844


namespace arithmetic_seq_sum_l291_291060

theorem arithmetic_seq_sum {a : ℕ → ℝ} (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha4 : a 4 = 5) : a 3 + a 5 = 10 :=
sorry

end arithmetic_seq_sum_l291_291060


namespace clea_total_time_l291_291017

-- Definitions based on conditions given
def walking_time_on_stationary (x y : ℝ) (h1 : 80 * x = y) : ℝ :=
  80

def walking_time_on_moving (x y : ℝ) (k : ℝ) (h2 : 32 * (x + k) = y) : ℝ :=
  32

def escalator_speed (x k : ℝ) (h3 : k = 1.5 * x) : ℝ :=
  1.5 * x

-- The actual theorem based on the question
theorem clea_total_time 
  (x y k : ℝ)
  (h1 : 80 * x = y)
  (h2 : 32 * (x + k) = y)
  (h3 : k = 1.5 * x) :
  let t1 := y / (2 * x)
  let t2 := y / (3 * x)
  t1 + t2 = 200 / 3 :=
by
  sorry

end clea_total_time_l291_291017


namespace algebraic_expression_evaluation_l291_291363

theorem algebraic_expression_evaluation (a b : ℝ) (h : 1 / a + 1 / (2 * b) = 3) :
  (2 * a - 5 * a * b + 4 * b) / (4 * a * b - 3 * a - 6 * b) = -1 / 2 := 
by
  sorry

end algebraic_expression_evaluation_l291_291363


namespace triangle_angle_sine_ratio_l291_291697

variable {A B C D : Point}
variable (angleBAC : Real)

def angleB := 45 -- 45 degrees in radians
def angleC := 60 -- 60 degrees in radians

def divides := (BCDiv: Real) → BCDiv = 1 / 3 -- D divides BC in ratio 1:2

theorem triangle_angle_sine_ratio
  (h_triangle : Triangle A B C)
  (h_angleB : ∠ B = angleB)
  (h_angleC : ∠ C = angleC)
  (h_divides : divides)
  : sin (∠ BAD) / sin (∠ CAD) = Real.sqrt 6 / 4 := by
  sorry

end triangle_angle_sine_ratio_l291_291697


namespace stereos_production_fraction_l291_291018

/-
Company S produces three kinds of stereos: basic, deluxe, and premium.
Of the stereos produced by Company S last month, 2/5 were basic, 3/10 were deluxe, and the rest were premium.
It takes 1.6 as many hours to produce a deluxe stereo as it does to produce a basic stereo, and 2.5 as many hours to produce a premium stereo as it does to produce a basic stereo.
Prove that the number of hours it took to produce the deluxe and premium stereos last month was 123/163 of the total number of hours it took to produce all the stereos.
-/

def stereos_production (total_stereos : ℕ) (basic_ratio deluxe_ratio : ℚ)
  (deluxe_time_multiplier premium_time_multiplier : ℚ) : ℚ :=
  let basic_stereos := total_stereos * basic_ratio
  let deluxe_stereos := total_stereos * deluxe_ratio
  let premium_stereos := total_stereos - basic_stereos - deluxe_stereos
  let basic_time := basic_stereos
  let deluxe_time := deluxe_stereos * deluxe_time_multiplier
  let premium_time := premium_stereos * premium_time_multiplier
  let total_time := basic_time + deluxe_time + premium_time
  (deluxe_time + premium_time) / total_time

-- Given values
def total_stereos : ℕ := 100
def basic_ratio : ℚ := 2 / 5
def deluxe_ratio : ℚ := 3 / 10
def deluxe_time_multiplier : ℚ := 1.6
def premium_time_multiplier : ℚ := 2.5

theorem stereos_production_fraction : stereos_production total_stereos basic_ratio deluxe_ratio deluxe_time_multiplier premium_time_multiplier = 123 / 163 := by
  sorry

end stereos_production_fraction_l291_291018


namespace prob_C_prob_B_prob_at_least_two_l291_291929

variable {P : Set ℕ → ℝ}

-- Given Conditions
def probability_A := 3 / 4
def probability_AC_incorrect := 1 / 12
def probability_BC_correct := 1 / 4

-- Definitions used in conditions
def probability_C := 2 / 3
def probability_B := 3 / 8

-- Proof Statements
theorem prob_C : P {3} = probability_C := sorry
theorem prob_B : P {2} = probability_B := sorry
theorem prob_at_least_two :
  (P {1, 2, 3} + P {1, 2} (1 - probability_C) + P {1, 3} (1 - probability_B) + P {2, 3} (1 - probability_A)) = 21 / 32 := sorry

end prob_C_prob_B_prob_at_least_two_l291_291929


namespace quadratic_function_passing_origin_l291_291657

theorem quadratic_function_passing_origin (a : ℝ) (h : ∃ x y, y = ax^2 + x + a * (a - 2) ∧ (x, y) = (0, 0)) : a = 2 := by
  sorry

end quadratic_function_passing_origin_l291_291657


namespace greatest_y_value_l291_291090

theorem greatest_y_value (x y : ℤ) (h : x * y + 7 * x + 2 * y = -8) : y ≤ -1 :=
by
  sorry

end greatest_y_value_l291_291090


namespace find_x_l291_291047

-- Definitions of the vectors and conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_a_minus_b (x : ℝ) : ℝ × ℝ := ((1 - x), (4))

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The given condition of perpendicular vectors
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

-- The theorem to prove
theorem find_x : ∃ x : ℝ, is_perpendicular vector_a (vector_a_minus_b x) ∧ x = 9 :=
by {
  -- Sorry statement used to skip proof
  sorry
}

end find_x_l291_291047


namespace polygon_sides_l291_291684

theorem polygon_sides (interior_angle : ℝ) (h : interior_angle = 120) :
  ∃ (n : ℕ), n = 6 :=
begin
  use 6,
  sorry  -- Proof goes here
end

end polygon_sides_l291_291684


namespace geometric_sequence_sixth_term_l291_291419

theorem geometric_sequence_sixth_term (a r : ℕ) (h₁ : a = 8) (h₂ : a * r ^ 3 = 64) : a * r ^ 5 = 256 :=
by
  -- to be filled (proof skipped)
  sorry

end geometric_sequence_sixth_term_l291_291419


namespace firing_sequence_hits_submarine_l291_291413

theorem firing_sequence_hits_submarine (a b : ℕ) (hb : b > 0) : ∃ n : ℕ, (∃ (an bn : ℕ), (an + bn * n) = a + n * b) :=
sorry

end firing_sequence_hits_submarine_l291_291413


namespace simplify_exponent_expression_l291_291295

theorem simplify_exponent_expression : 2000 * (2000 ^ 2000) = 2000 ^ 2001 :=
by sorry

end simplify_exponent_expression_l291_291295


namespace no_solution_l291_291190

theorem no_solution (n : ℕ) (k : ℕ) (hn : Prime n) (hk : 0 < k) :
  ¬ (n ≤ n.factorial - k ^ n ∧ n.factorial - k ^ n ≤ k * n) :=
by
  sorry

end no_solution_l291_291190


namespace larger_number_is_437_l291_291599

-- Definitions from the conditions
def hcf : ℕ := 23
def factor1 : ℕ := 13
def factor2 : ℕ := 19

-- The larger number should be the product of H.C.F and the larger factor.
theorem larger_number_is_437 : hcf * factor2 = 437 := by
  sorry

end larger_number_is_437_l291_291599


namespace non_congruent_squares_on_5x5_grid_l291_291048

def is_lattice_point (x y : ℕ) : Prop := x ≤ 4 ∧ y ≤ 4

def is_square {a b c d : (ℕ × ℕ)} : Prop :=
((a.1 - b.1)^2 + (a.2 - b.2)^2 = (c.1 - d.1)^2 + (c.2 - d.2)^2) ∧ 
((c.1 - b.1)^2 + (c.2 - b.2)^2 = (a.1 - d.1)^2 + (a.2 - d.2)^2)

def number_of_non_congruent_squares : ℕ :=
  4 + -- Standard squares: 1x1, 2x2, 3x3, 4x4
  2 + -- Diagonal squares: with sides √2 and 2√2
  2   -- Diagonal sides of 1x2 and 1x3 rectangles

theorem non_congruent_squares_on_5x5_grid :
  number_of_non_congruent_squares = 8 :=
by
  -- proof goes here
  sorry

end non_congruent_squares_on_5x5_grid_l291_291048


namespace total_cost_correct_l291_291962

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59
def total_cost : ℝ := 12.30

theorem total_cost_correct : football_cost + marbles_cost = total_cost := 
by
  sorry

end total_cost_correct_l291_291962


namespace decryption_correct_l291_291427

theorem decryption_correct (a b : ℤ) (h1 : a - 2 * b = 1) (h2 : 2 * a + b = 7) : a = 3 ∧ b = 1 :=
by
  sorry

end decryption_correct_l291_291427


namespace ratio_of_areas_l291_291205

theorem ratio_of_areas (R_C R_D : ℝ) (h : (60 / 360) * (2 * Real.pi * R_C) = (40 / 360) * (2 * Real.pi * R_D)) :
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l291_291205


namespace find_positive_real_solution_l291_291811

theorem find_positive_real_solution (x : ℝ) (h : 0 < x) :
  (1 / 3) * (4 * x ^ 2 - 3) = (x ^ 2 - 75 * x - 15) * (x ^ 2 + 40 * x + 8) →
  x = (75 + Real.sqrt (75 ^ 2 + 4 * 13)) / 2 ∨ x = (-40 + Real.sqrt (40 ^ 2 - 4 * 7)) / 2 :=
by
  sorry

end find_positive_real_solution_l291_291811


namespace overlap_32_l291_291710

section
variables (t : ℝ)
def position_A : ℝ := 120 - 50 * t
def position_B : ℝ := 220 - 50 * t
def position_N : ℝ := 30 * t - 30
def position_M : ℝ := 30 * t + 10

theorem overlap_32 :
  (∃ t : ℝ, (30 * t + 10 - (120 - 50 * t) = 32) ∨ 
            (-50 * t + 220 - (30 * t - 30) = 32)) ↔
  (t = 71 / 40 ∨ t = 109 / 40) :=
sorry
end

end overlap_32_l291_291710


namespace floor_length_l291_291082

/-- Given the rectangular tiles of size 50 cm by 40 cm, which are laid on a rectangular floor
without overlap and with a maximum of 9 tiles. Prove the floor length is 450 cm. -/
theorem floor_length (tiles_max : ℕ) (tile_length tile_width floor_length floor_width : ℕ)
  (Htile_length : tile_length = 50) (Htile_width : tile_width = 40)
  (Htiles_max : tiles_max = 9)
  (Hconditions : (∀ m n : ℕ, (m * n = tiles_max) → 
                  (floor_length = m * tile_length ∨ floor_length = m * tile_width)))
  : floor_length = 450 :=
by 
  sorry

end floor_length_l291_291082


namespace handshake_count_l291_291412

theorem handshake_count (couples : ℕ) (people : ℕ) (total_handshakes : ℕ) :
  couples = 6 →
  people = 2 * couples →
  total_handshakes = (people * (people - 1)) / 2 - couples →
  total_handshakes = 60 :=
by
  intros h_couples h_people h_handshakes
  sorry

end handshake_count_l291_291412


namespace sum_of_roots_of_quadratic_l291_291105

variables {b x₁ x₂ : ℝ}

theorem sum_of_roots_of_quadratic (h : x₁^2 - 2 * x₁ + b = 0) (h' : x₂^2 - 2 * x₂ + b = 0) :
    x₁ + x₂ = 2 :=
sorry

end sum_of_roots_of_quadratic_l291_291105


namespace min_value_of_z_l291_291746

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end min_value_of_z_l291_291746


namespace contradiction_proof_l291_291287

theorem contradiction_proof (a b c : ℝ) (h1 : 0 < a ∧ a < 2) (h2 : 0 < b ∧ b < 2) (h3 : 0 < c ∧ c < 2) :
  ¬ (a * (2 - b) > 1 ∧ b * (2 - c) > 1 ∧ c * (2 - a) > 1) :=
sorry

end contradiction_proof_l291_291287


namespace cube_volume_multiple_of_6_l291_291957

theorem cube_volume_multiple_of_6 (n : ℕ) (h : ∃ m : ℕ, n^3 = 24 * m) : ∃ k : ℕ, n = 6 * k :=
by
  sorry

end cube_volume_multiple_of_6_l291_291957


namespace ratio_u_v_l291_291776

variables {u v : ℝ}
variables (u_lt_v : u < v)
variables (h_triangle : triangle 15 12 9)
variables (inscribed_circle : is_inscribed_circle 15 12 9 u v)

theorem ratio_u_v : u / v = 1 / 2 :=
sorry

end ratio_u_v_l291_291776


namespace two_point_seven_five_as_fraction_l291_291439

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l291_291439


namespace roots_imply_sum_l291_291604

theorem roots_imply_sum (a b c x1 x2 : ℝ) (hneq : a ≠ 0) (hroots : a * x1 ^ 2 + b * x1 + c = 0 ∧ a * x2 ^ 2 + b * x2 + c = 0) :
  x1 + x2 = -b / a :=
sorry

end roots_imply_sum_l291_291604


namespace BD_distance_16_l291_291718

noncomputable def distanceBD (DA AB : ℝ) (angleBDA : ℝ) : ℝ :=
  (DA^2 + AB^2 - 2 * DA * AB * Real.cos angleBDA).sqrt

theorem BD_distance_16 :
  distanceBD 10 14 (60 * Real.pi / 180) = 16 := by
  sorry

end BD_distance_16_l291_291718


namespace part1_part2_part3_l291_291557

open Set

-- Define the sets A and B and the universal set
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def U : Set ℝ := univ  -- Universal set R

theorem part1 : A ∩ B = { x | 3 ≤ x ∧ x < 7 } :=
by { sorry }

theorem part2 : U \ A = { x | x < 3 ∨ x ≥ 7 } :=
by { sorry }

theorem part3 : U \ (A ∪ B) = { x | x ≤ 2 ∨ x ≥ 10 } :=
by { sorry }

end part1_part2_part3_l291_291557


namespace least_number_to_add_l291_291916

theorem least_number_to_add (a b n : ℕ) (h₁ : a = 1056) (h₂ : b = 29) (h₃ : (a + n) % b = 0) : n = 17 :=
sorry

end least_number_to_add_l291_291916


namespace min_mod_z_l291_291236

open Complex

theorem min_mod_z (z : ℂ) (hz : abs (z - 2 * I) + abs (z - 5) = 7) : abs z = 10 / 7 :=
sorry

end min_mod_z_l291_291236


namespace median_of_divisors_9999_is_100_l291_291156

-- Define the prime factorization of 9999
def factor_9999 : Prop := 9999 = 3^2 * 11 * 101

-- List of all divisors
def divisors_9999 : List ℕ := [1, 3, 9, 11, 33, 99, 101, 303, 909, 1111, 3333, 9999]

-- Median of the positive divisors of 9999
def median_divisors_9999 (divisors : List ℕ) : ℕ :=
  let len := List.length divisors
  if len % 2 = 0 then
    let mid1 := List.nth_le divisors ((len / 2) - 1) (by sorry) -- 6th element for 0-based index
    let mid2 := List.nth_le divisors (len / 2) (by sorry) -- 7th element for 0-based index
    (mid1 + mid2) / 2
  else
    sorry

theorem median_of_divisors_9999_is_100 : median_divisors_9999 divisors_9999 = 100 := by
  sorry

end median_of_divisors_9999_is_100_l291_291156


namespace question_I_question_II_l291_291878

def f (x a : ℝ) : ℝ := |x - a| + 3 * x

theorem question_I (a : ℝ) (h_pos : a > 0) : 
  (f 1 x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) := by sorry

theorem question_II (a : ℝ) (h_pos : a > 0) : 
  (- (a / 2) = -1) ↔ (a = 2) := by sorry

end question_I_question_II_l291_291878


namespace evaluate_expression_l291_291023

theorem evaluate_expression :
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  (sum1 / sum2 - sum2 / sum1) = 11 / 30 :=
by
  let sum1 := 3 + 6 + 9
  let sum2 := 2 + 5 + 8
  sorry

end evaluate_expression_l291_291023


namespace probability_first_green_then_blue_l291_291158

variable {α : Type} [Fintype α]

noncomputable def prob_first_green_second_blue : ℚ := 
  let total_marbles := 10
  let green_marbles := 6
  let blue_marbles := 4
  let prob_first_green := (green_marbles : ℚ) / total_marbles
  let prob_second_blue := (blue_marbles : ℚ) / (total_marbles - 1)
  (prob_first_green * prob_second_blue)

theorem probability_first_green_then_blue :
  prob_first_green_second_blue = 4 / 15 := by
  sorry

end probability_first_green_then_blue_l291_291158


namespace find_x_when_y_30_l291_291250

variable (x y k : ℝ)

noncomputable def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, x * y = k

theorem find_x_when_y_30
  (h_inv_prop : inversely_proportional x y) 
  (h_known_values : x = 5 ∧ y = 15) :
  ∃ x : ℝ, (∃ y : ℝ, y = 30) ∧ x = 5 / 2 := by
  sorry

end find_x_when_y_30_l291_291250


namespace regular_pentagonal_prism_diagonal_count_l291_291770

noncomputable def diagonal_count (n : ℕ) : ℕ := 
  if n = 5 then 10 else 0

theorem regular_pentagonal_prism_diagonal_count :
  diagonal_count 5 = 10 := 
  by
    sorry

end regular_pentagonal_prism_diagonal_count_l291_291770


namespace no_prime_pair_summing_to_53_l291_291389

theorem no_prime_pair_summing_to_53 :
  ∀ (p q : ℕ), Nat.Prime p → Nat.Prime q → p + q = 53 → false :=
by
  sorry

end no_prime_pair_summing_to_53_l291_291389


namespace network_connections_l291_291109

theorem network_connections (n m : ℕ) (hn : n = 30) (hm : m = 5) 
(h_total_conn : (n * 4) / 2 = 60) : 
60 + m = 65 :=
by
  sorry

end network_connections_l291_291109


namespace PersonYs_speed_in_still_water_l291_291278

def speed_in_still_water (speed_X : ℕ) (t_1 t_2 : ℕ) (x : ℕ) : Prop :=
  ∀ y : ℤ, 4 * (6 - y + x + y) = 4 * 6 + 4 * x ∧ 16 * (x + y) = 16 * (6 + y) + 4 * (x - 6) →
  x = 10

theorem PersonYs_speed_in_still_water :
  speed_in_still_water 6 4 16 10 :=
by
  sorry

end PersonYs_speed_in_still_water_l291_291278


namespace inequality_b_does_not_hold_l291_291051

theorem inequality_b_does_not_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬(a + d > b + c) ↔ a + d ≤ b + c :=
by
  -- We only need the statement, so we add sorry at the end
  sorry

end inequality_b_does_not_hold_l291_291051


namespace max_height_piston_l291_291152

theorem max_height_piston (M a P c_v g R: ℝ) (h : ℝ) 
  (h_pos : 0 < h) (M_pos : 0 < M) (a_pos : 0 < a) (P_pos : 0 < P)
  (c_v_pos : 0 < c_v) (g_pos : 0 < g) (R_pos : 0 < R) :
  h = (2 * P ^ 2) / (M ^ 2 * g * a ^ 2 * (1 + c_v / R) ^ 2) := sorry

end max_height_piston_l291_291152


namespace cos_x_minus_pi_over_3_l291_291372

theorem cos_x_minus_pi_over_3 (x : ℝ) (h : Real.sin (x + π / 6) = 4 / 5) :
  Real.cos (x - π / 3) = 4 / 5 :=
sorry

end cos_x_minus_pi_over_3_l291_291372


namespace quadratic_range_l291_291358

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 7

-- Defining the range of the quadratic function for the interval -1 < x < 4
theorem quadratic_range (y : ℝ) : 3 ≤ y ∧ y < 12 ↔ ∃ x : ℝ, -1 < x ∧ x < 4 ∧ y = quadratic_function x :=
by
  sorry

end quadratic_range_l291_291358


namespace intersection_single_point_max_PA_PB_l291_291605

-- Problem (1)
theorem intersection_single_point (a : ℝ) :
  (∀ x : ℝ, 2 * a = |x - a| - 1 → x = a) → a = -1 / 2 :=
sorry

-- Problem (2)
theorem max_PA_PB (m : ℝ) (P : ℝ × ℝ) :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 3)
  P ≠ A ∧ P ≠ B ∧ (P.1 + m * P.2 = 0) ∧ (m * P.1 - P.2 - m + 3 = 0) →
  |dist P A| * |dist P B| ≤ 5 :=
sorry

end intersection_single_point_max_PA_PB_l291_291605


namespace scientific_notation_conversion_l291_291806

theorem scientific_notation_conversion : 
  ∀ (n : ℝ), n = 1.8 * 10^8 → n = 180000000 :=
by
  intros n h
  sorry

end scientific_notation_conversion_l291_291806


namespace sum_of_roots_of_quadratic_l291_291107

variables {b x₁ x₂ : ℝ}

theorem sum_of_roots_of_quadratic (h : x₁^2 - 2 * x₁ + b = 0) (h' : x₂^2 - 2 * x₂ + b = 0) :
    x₁ + x₂ = 2 :=
sorry

end sum_of_roots_of_quadratic_l291_291107


namespace no_prime_pairs_sum_53_l291_291391

open nat

theorem no_prime_pairs_sum_53 : 
  ¬∃ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 :=
by sorry

end no_prime_pairs_sum_53_l291_291391


namespace ratio_of_areas_eq_l291_291208

-- Define the conditions
variables {C D : Type} [circle C] [circle D]
variables (R_C R_D : ℝ)
variable (L : ℝ)

-- Given conditions
axiom arc_length_eq : (60 / 360) * (2 * π * R_C) = L
axiom arc_length_eq' : (40 / 360) * (2 * π * R_D) = L

-- Statement to prove
theorem ratio_of_areas_eq : (π * R_C^2) / (π * R_D^2) = 4 / 9 :=
sorry

end ratio_of_areas_eq_l291_291208


namespace part1_general_formula_part2_find_d_l291_291706

open Nat

-- Define the arithmetic sequence conditions
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define the sequence b_n
def b_n (a : ℕ → ℕ) (n : ℕ) := (n^2 + n) / a n

-- Define the sum of the first n terms
def S_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

def T_n (a : ℕ → ℕ) (n : ℕ) :=
  ∑ i in range (n + 1), b_n a i

-- Define the conditions given in the problem
def condition_1 (a : ℕ → ℕ) : Prop :=
  3 * a 2 = 3 * a 1 + a 3

def condition_2 (a : ℕ → ℕ) : Prop :=
  S_n a 3 + T_n a 3 = 21

def condition_3 (a : ℕ → ℕ) : Prop :=
  b_n a (n + 1) - b_n a n = C ∀ n, ∃ C, ∀ n, b_n a (n + 1) - b_n a n = C

def condition_4 (a : ℕ → ℕ) : Prop :=
  S_n a 99 - T_n a 99 = 99

-- Theorem statement for part 1
theorem part1_general_formula (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_1 a → condition_2 a → (∀ n, a n = 3 * n) :=
sorry

-- Theorem statement for part 2
theorem part2_find_d (a : ℕ → ℕ) (d : ℕ) :
  is_arithmetic_sequence a d → (d > 1) → condition_3 a → condition_4 a → (d = 51 / 50) :=
sorry

end part1_general_formula_part2_find_d_l291_291706


namespace probability_point_between_X_and_Z_l291_291407

theorem probability_point_between_X_and_Z (XW XZ YW : ℝ) (h1 : XW = 4 * XZ) (h2 : XW = 8 * YW) :
  (XZ / XW) = 1 / 4 := by
  sorry

end probability_point_between_X_and_Z_l291_291407


namespace correct_profit_equation_l291_291936

def total_rooms : ℕ := 50
def initial_price : ℕ := 180
def price_increase_step : ℕ := 10
def cost_per_occupied_room : ℕ := 20
def desired_profit : ℕ := 10890

theorem correct_profit_equation (x : ℕ) : 
  (x - cost_per_occupied_room : ℤ) * (total_rooms - (x - initial_price : ℤ) / price_increase_step) = desired_profit :=
by sorry

end correct_profit_equation_l291_291936


namespace june_ride_time_l291_291700

theorem june_ride_time (dist1 time1 dist2 time2 : ℝ) (h : dist1 = 2 ∧ time1 = 8 ∧ dist2 = 5 ∧ time2 = 20) :
  (dist2 / (dist1 / time1) = time2) := by
  -- using the defined conditions
  rcases h with ⟨h1, h2, h3, h4⟩
  rw [h1, h2, h3, h4]
  -- simplifying the expression
  sorry

end june_ride_time_l291_291700


namespace find_angle_ACD_l291_291693

-- Define the vertices of the quadrilateral
variables {A B C D : Type*}

-- Given angles and side equality
variables (angle_DAC : ℝ) (angle_DBC : ℝ) (angle_BCD : ℝ) (eq_BC_AD : Prop)

-- The given conditions in the problem
axiom angle_DAC_is_98 : angle_DAC = 98
axiom angle_DBC_is_82 : angle_DBC = 82
axiom angle_BCD_is_70 : angle_BCD = 70
axiom BC_eq_AD : eq_BC_AD = true

-- Target angle to be proven
def angle_ACD : ℝ := 28

-- The theorem
theorem find_angle_ACD (h1 : angle_DAC = 98)
                       (h2 : angle_DBC = 82)
                       (h3 : angle_BCD = 70)
                       (h4 : eq_BC_AD) : angle_ACD = 28 := 
by
  sorry  -- Proof of the theorem

end find_angle_ACD_l291_291693


namespace cos_double_angle_l291_291676

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 2) = 1 / 2) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_l291_291676


namespace proof_a_l291_291820

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ (y - 3) / (x - 2) = 3}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ a * x + 2 * y + a = 0}

-- Given conditions that M ∩ N = ∅, prove that a = -6 or a = -2
theorem proof_a (h : ∃ a : ℝ, (N a ∩ M = ∅)) : ∃ a : ℝ, a = -6 ∨ a = -2 :=
  sorry

end proof_a_l291_291820


namespace sum_valid_two_digit_integers_l291_291758

theorem sum_valid_two_digit_integers :
  ∃ S : ℕ, S = 36 ∧ (∀ n, 10 ≤ n ∧ n < 100 →
    (∃ a b, n = 10 * a + b ∧ a + b ∣ n ∧ 2 * a * b ∣ n → n = 36)) :=
by
  sorry

end sum_valid_two_digit_integers_l291_291758


namespace vector_projection_unique_l291_291760

theorem vector_projection_unique (a : ℝ) (c d : ℝ) (h : c + 3 * d = 0) :
    ∃ p : ℝ × ℝ, (∀ a : ℝ, ∀ (v : ℝ × ℝ) (w : ℝ × ℝ), 
      v = (a, 3 * a - 2) → 
      w = (c, d) → 
      ∃ p : ℝ × ℝ, p = (3 / 5, -1 / 5)) :=
sorry

end vector_projection_unique_l291_291760


namespace chess_tournament_participants_l291_291925

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 120) : n = 16 :=
sorry

end chess_tournament_participants_l291_291925


namespace express_y_in_terms_of_x_and_p_l291_291571

theorem express_y_in_terms_of_x_and_p (x p : ℚ) (h : x = (1 + p / 100) * (1 / y)) : 
  y = (100 + p) / (100 * x) := 
sorry

end express_y_in_terms_of_x_and_p_l291_291571


namespace license_plates_count_l291_291581

open Finset

def valid_license_plates : Finset (List Char) := 
  let alphabet : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N']
  let alphabetSet := (alphabet.toFinset : Finset Char)
  let firstLetterChoices := (['B', 'C'].toFinset : Finset Char)
  let lastLetterChoices := ('N' : Finset Char)
  let excludedLetters := (['B', 'C', 'M', 'N'].toFinset : Finset Char)
  let usableLetters := alphabetSet \ excludedLetters
  let fValidPlates (freeLetters : Finset Char) : Finset (List Char) → Finset (List Char) :=
    λ acc, acc.bind fun prefix =>
      freeLetters.toFinset.bind (λ l => singleton (prefix ++ [l]))

  (firstLetterChoices.product ((usableLetters.toFinset.powerset 4).bind fValidPlates)).bind
    fun prefix => (<'N'>.product (singleton (prefix.toList))).image
      fun p => p.1 ++ p.2

theorem license_plates_count : valid_license_plates.card = 15840 := by
  sorry

end license_plates_count_l291_291581


namespace GreenValley_Absent_Percentage_l291_291662

theorem GreenValley_Absent_Percentage 
  (total_students boys girls absent_boys_frac absent_girls_frac : ℝ)
  (h1 : total_students = 120)
  (h2 : boys = 70)
  (h3 : girls = 50)
  (h4 : absent_boys_frac = 1 / 7)
  (h5 : absent_girls_frac = 1 / 5) :
  (absent_boys_frac * boys + absent_girls_frac * girls) / total_students * 100 = 16.67 := 
sorry

end GreenValley_Absent_Percentage_l291_291662


namespace find_sum_of_money_invested_l291_291146

theorem find_sum_of_money_invested (P : ℝ) (h1 : SI_15 = P * (15 / 100) * 2)
                                    (h2 : SI_12 = P * (12 / 100) * 2)
                                    (h3 : SI_15 - SI_12 = 720) : 
                                    P = 12000 :=
by
  -- Skipping the proof
  sorry

end find_sum_of_money_invested_l291_291146


namespace N_subset_M_l291_291367

def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x - 2 = 0}

theorem N_subset_M : N ⊆ M := sorry

end N_subset_M_l291_291367


namespace remainder_of_expression_l291_291445

theorem remainder_of_expression :
  let a := 2^206 + 206
  let b := 2^103 + 2^53 + 1
  a % b = 205 := 
sorry

end remainder_of_expression_l291_291445


namespace total_dividend_received_l291_291782

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

end total_dividend_received_l291_291782


namespace conic_section_is_parabola_l291_291189

theorem conic_section_is_parabola (x y : ℝ) : y^4 - 16 * x^2 = 2 * y^2 - 64 → ((y^2 - 1)^2 = 16 * x^2 - 63) ∧ (∃ k : ℝ, y^2 = 4 * k * x + 1) :=
sorry

end conic_section_is_parabola_l291_291189


namespace grunters_win_all_6_games_l291_291252

-- Define the probability of the Grunters winning a single game
def probability_win_single_game : ℚ := 3 / 5

-- Define the number of games
def number_of_games : ℕ := 6

-- Calculate the probability of winning all games (all games are independent)
def probability_win_all_games (p : ℚ) (n : ℕ) : ℚ := p ^ n

-- Prove that the probability of the Grunters winning all 6 games is exactly 729/15625
theorem grunters_win_all_6_games :
  probability_win_all_games probability_win_single_game number_of_games = 729 / 15625 :=
by
  sorry

end grunters_win_all_6_games_l291_291252


namespace rectangle_width_l291_291095

theorem rectangle_width (length : ℕ) (perimeter : ℕ) (h1 : length = 20) (h2 : perimeter = 70) :
  2 * (length + width) = perimeter → width = 15 :=
by
  intro h
  rw [h1, h2] at h
  -- Continue the steps to solve for width (can be simplified if not requesting the whole proof)
  sorry

end rectangle_width_l291_291095


namespace constant_in_denominator_l291_291903

theorem constant_in_denominator (x y z : ℝ) (some_constant : ℝ)
  (h : ((x - y)^3 + (y - z)^3 + (z - x)^3) / (some_constant * (x - y) * (y - z) * (z - x)) = 0.2) :
  some_constant = 15 := 
sorry

end constant_in_denominator_l291_291903


namespace total_marbles_proof_l291_291332

def dan_violet_marbles : Nat := 64
def mary_red_marbles : Nat := 14
def john_blue_marbles (x : Nat) : Nat := x

def total_marble (x : Nat) : Nat := dan_violet_marbles + mary_red_marbles + john_blue_marbles x

theorem total_marbles_proof (x : Nat) : total_marble x = 78 + x := by
  sorry

end total_marbles_proof_l291_291332


namespace remaining_amount_eq_40_l291_291727

-- Definitions and conditions
def initial_amount : ℕ := 100
def food_spending : ℕ := 20
def rides_spending : ℕ := 2 * food_spending
def total_spending : ℕ := food_spending + rides_spending

-- The proposition to be proved
theorem remaining_amount_eq_40 :
  initial_amount - total_spending = 40 :=
by
  sorry

end remaining_amount_eq_40_l291_291727


namespace correct_propositions_l291_291549

-- Definitions of relations between lines and planes
variable {Line : Type}
variable {Plane : Type}

-- Definition of relationships
variable (parallel_lines : Line → Line → Prop)
variable (parallel_plane_with_plane : Plane → Plane → Prop)
variable (parallel_line_with_plane : Line → Plane → Prop)
variable (perpendicular_plane_with_plane : Plane → Plane → Prop)
variable (perpendicular_line_with_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (distinct_lines : Line → Line → Prop)
variable (distinct_planes : Plane → Plane → Prop)

-- The main theorem we are proving with the given conditions
theorem correct_propositions (m n : Line) (α β γ : Plane)
  (hmn : distinct_lines m n) (hαβ : distinct_planes α β) (hαγ : distinct_planes α γ)
  (hβγ : distinct_planes β γ) :
  -- Statement 1
  (parallel_plane_with_plane α β → parallel_plane_with_plane α γ → parallel_plane_with_plane β γ) ∧
  -- Statement 3
  (perpendicular_line_with_plane m α → parallel_line_with_plane m β → perpendicular_plane_with_plane α β) :=
by
  sorry

end correct_propositions_l291_291549


namespace find_abc_sum_l291_291238

noncomputable def x := Real.sqrt ((Real.sqrt 105) / 2 + 7 / 2)

theorem find_abc_sum :
  ∃ (a b c : ℕ), a + b + c = 5824 ∧
  x ^ 100 = 3 * x ^ 98 + 15 * x ^ 96 + 12 * x ^ 94 - x ^ 50 + a * x ^ 46 + b * x ^ 44 + c * x ^ 40 :=
  sorry

end find_abc_sum_l291_291238


namespace minimum_value_is_8_l291_291552

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_is_8 :
  ∃ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), minimum_value x y hx hy = 8 :=
by
  sorry

end minimum_value_is_8_l291_291552


namespace xy_y_sq_eq_y_sq_3y_12_l291_291373

variable (x y : ℝ)

theorem xy_y_sq_eq_y_sq_3y_12 (h : x * (x + y) = x^2 + 3 * y + 12) : 
  x * y + y^2 = y^2 + 3 * y + 12 := 
sorry

end xy_y_sq_eq_y_sq_3y_12_l291_291373


namespace find_asterisk_value_l291_291762

theorem find_asterisk_value : 
  (∃ x : ℕ, (x / 21) * (x / 189) = 1) → x = 63 :=
by
  intro h
  sorry

end find_asterisk_value_l291_291762


namespace convert_to_canonical_form_l291_291485

def quadratic_eqn (x y : ℝ) : ℝ :=
  8 * x^2 + 4 * x * y + 5 * y^2 - 56 * x - 32 * y + 80

def canonical_form (x2 y2 : ℝ) : Prop :=
  (x2^2 / 4) + (y2^2 / 9) = 1

theorem convert_to_canonical_form (x y : ℝ) :
  quadratic_eqn x y = 0 → ∃ (x2 y2 : ℝ), canonical_form x2 y2 :=
sorry

end convert_to_canonical_form_l291_291485


namespace base_h_equation_l291_291814

theorem base_h_equation (h : ℕ) : 
  (5 * h^3 + 7 * h^2 + 3 * h + 4) + (6 * h^3 + 4 * h^2 + 2 * h + 1) = 
  1 * h^4 + 4 * h^3 + 1 * h^2 + 5 * h + 5 → 
  h = 10 := 
sorry

end base_h_equation_l291_291814


namespace function_is_decreasing_on_R_l291_291580

def is_decreasing (a : ℝ) : Prop := a - 1 < 0

theorem function_is_decreasing_on_R (a : ℝ) : (1 < a ∧ a < 2) ↔ is_decreasing a :=
by
  sorry

end function_is_decreasing_on_R_l291_291580


namespace largest_divisor_of_n_squared_sub_n_squared_l291_291981

theorem largest_divisor_of_n_squared_sub_n_squared (n : ℤ) : 6 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_n_squared_sub_n_squared_l291_291981


namespace x_coordinate_equidistant_l291_291283

theorem x_coordinate_equidistant :
  ∃ x : ℝ, (sqrt ((-3 - x)^2 + 0^2) = sqrt ((2 - x)^2 + 5^2)) ∧ x = 2 :=
by
  sorry

end x_coordinate_equidistant_l291_291283


namespace michael_exceeds_suresh_by_36_5_l291_291567

noncomputable def shares_total : ℝ := 730
noncomputable def punith_ratio_to_michael : ℝ := 3 / 4
noncomputable def michael_ratio_to_suresh : ℝ := 3.5 / 3

theorem michael_exceeds_suresh_by_36_5 :
  ∃ P M S : ℝ, P + M + S = shares_total
  ∧ (P / M = punith_ratio_to_michael)
  ∧ (M / S = michael_ratio_to_suresh)
  ∧ (M - S = 36.5) :=
by
  sorry

end michael_exceeds_suresh_by_36_5_l291_291567


namespace fruit_seller_original_apples_l291_291930

variable (x : ℝ)

theorem fruit_seller_original_apples (h : 0.60 * x = 420) : x = 700 := by
  sorry

end fruit_seller_original_apples_l291_291930


namespace y_value_when_x_is_3_l291_291904

theorem y_value_when_x_is_3 :
  (x + y = 30) → (x - y = 12) → (x * y = 189) → (x = 3) → y = 63 :=
by 
  intros h1 h2 h3 h4
  sorry

end y_value_when_x_is_3_l291_291904


namespace intersection_P_Q_l291_291036

section set_intersection

variable (x : ℝ)

def P := { x : ℝ | x ≤ 1 }
def Q := { x : ℝ | -1 ≤ x ∧ x ≤ 2 }

theorem intersection_P_Q : { x | x ∈ P ∧ x ∈ Q } = { x | -1 ≤ x ∧ x ≤ 1 } :=
by
  sorry

end set_intersection

end intersection_P_Q_l291_291036


namespace Adam_current_money_is_8_l291_291478

variable (Adam_initial : ℕ) (spent_on_game : ℕ) (allowance : ℕ)

def money_left_after_spending (initial : ℕ) (spent : ℕ) := initial - spent
def current_money (money_left : ℕ) (allowance : ℕ) := money_left + allowance

theorem Adam_current_money_is_8 
    (h1 : Adam_initial = 5)
    (h2 : spent_on_game = 2)
    (h3 : allowance = 5) :
    current_money (money_left_after_spending Adam_initial spent_on_game) allowance = 8 := 
by sorry

end Adam_current_money_is_8_l291_291478


namespace value_of_k_l291_291650

-- Let k be a real number
variable (k : ℝ)

-- The given condition as a hypothesis
def condition := ∀ x : ℝ, (x + 3) * (x + 2) = k + 3 * x

-- The statement to prove
theorem value_of_k (h : ∀ x : ℝ, (x + 3) * (x + 2) = k + 3 * x) : k = 5 :=
sorry

end value_of_k_l291_291650


namespace no_prime_pairs_sum_53_l291_291393

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l291_291393


namespace solution_set_of_inequality1_solution_set_of_inequality2_l291_291490

-- First inequality problem
theorem solution_set_of_inequality1 :
  {x : ℝ | x^2 + 3*x + 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
sorry

-- Second inequality problem
theorem solution_set_of_inequality2 :
  {x : ℝ | -3*x^2 + 2*x + 2 < 0} =
  {x : ℝ | x ∈ Set.Iio ((1 - Real.sqrt 7) / 3) ∪ Set.Ioi ((1 + Real.sqrt 7) / 3)} :=
sorry

end solution_set_of_inequality1_solution_set_of_inequality2_l291_291490


namespace two_digit_number_is_9_l291_291953

def dig_product (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n);
  match digits with
  | [a, b] => a * b
  | _ => 0

theorem two_digit_number_is_9 :
  ∃ (M : ℕ), 
    10 ≤ M ∧ M < 100 ∧ -- M is a two-digit number
    Odd M ∧            -- M is odd
    9 ∣ M ∧            -- M is a multiple of 9
    ∃ k, dig_product M = k * k -- product of its digits is a perfect square
    ∧ M = 9 :=       -- the solution is M = 9
by
  sorry

end two_digit_number_is_9_l291_291953


namespace find_x_l291_291466

noncomputable def area_of_figure (x : ℝ) : ℝ :=
  let A_rectangle := 3 * x * 2 * x
  let A_square1 := x ^ 2
  let A_square2 := (4 * x) ^ 2
  let A_triangle := (3 * x * 2 * x) / 2
  A_rectangle + A_square1 + A_square2 + A_triangle

theorem find_x (x : ℝ) : area_of_figure x = 1250 → x = 6.93 :=
  sorry

end find_x_l291_291466


namespace sum_of_roots_eq_two_l291_291100

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end sum_of_roots_eq_two_l291_291100


namespace probability_hhh_before_hth_l291_291743

theorem probability_hhh_before_hth : 
  let coin_prob : ℚ := 1 / 2 in 
  let hhh_before_hth_prob : ℚ := 2 / 5 in 
  (probability_of_event_hhh_before_hth coin_prob) = hhh_before_hth_prob :=
sorry

end probability_hhh_before_hth_l291_291743


namespace Kaylee_total_boxes_needed_l291_291231

-- Defining the conditions
def lemon_biscuits := 12
def chocolate_biscuits := 5
def oatmeal_biscuits := 4
def still_needed := 12

-- Defining the total boxes sold so far
def total_sold := lemon_biscuits + chocolate_biscuits + oatmeal_biscuits

-- Defining the total number of boxes that need to be sold in total
def total_needed := total_sold + still_needed

-- Lean statement to prove the required total number of boxes
theorem Kaylee_total_boxes_needed : total_needed = 33 :=
by
  sorry

end Kaylee_total_boxes_needed_l291_291231


namespace basketball_team_win_requirement_l291_291772

theorem basketball_team_win_requirement :
  ∀ (games_won_first_60 : ℕ) (total_games : ℕ) (win_percentage : ℚ) (remaining_games : ℕ),
    games_won_first_60 = 45 →
    total_games = 110 →
    win_percentage = 0.75 →
    remaining_games = 50 →
    ∃ games_won_remaining, games_won_remaining = 38 ∧
    (games_won_first_60 + games_won_remaining) / total_games = win_percentage :=
by
  intros
  sorry

end basketball_team_win_requirement_l291_291772


namespace exist_non_zero_function_iff_sum_zero_l291_291024

theorem exist_non_zero_function_iff_sum_zero (a b c : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x y z : ℝ, a * f (x * y + f z) + b * f (y * z + f x) + c * f (z * x + f y) = 0) ∧ ¬ (∀ x : ℝ, f x = 0)) ↔ (a + b + c = 0) :=
by {
  sorry
}

end exist_non_zero_function_iff_sum_zero_l291_291024


namespace fraction_multiplication_l291_291120

theorem fraction_multiplication :
  (7 / 8) * 48 = 42 := 
sorry

end fraction_multiplication_l291_291120


namespace sin_sum_leq_3_sqrt_3_div_2_l291_291228

theorem sin_sum_leq_3_sqrt_3_div_2 (A B C : ℝ) (h_sum : A + B + C = Real.pi) (h_pos : 0 < A ∧ 0 < B ∧ 0 < C) :
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 :=
sorry

end sin_sum_leq_3_sqrt_3_div_2_l291_291228


namespace james_total_money_l291_291533

section
-- Conditions
def number_of_bills : ℕ := 3
def value_of_each_bill : ℕ := 20
def initial_wallet_amount : ℕ := 75

-- Question:
-- What is the total amount of money James has now?
def total_value_of_bills : ℕ := number_of_bills * value_of_each_bill
def total_money_now : ℕ := initial_wallet_amount + total_value_of_bills

-- Theorem stating that he has $135 now.
theorem james_total_money : total_money_now = 135 := 
  by
    sorry
end

end james_total_money_l291_291533


namespace base_number_is_4_l291_291374

theorem base_number_is_4 (some_number : ℕ) (h : 16^8 = some_number^16) : some_number = 4 :=
sorry

end base_number_is_4_l291_291374


namespace bus_routes_theorem_l291_291069

open Function

def bus_routes_exist : Prop :=
  ∃ (routes : Fin 10 → Set (Fin 10)), 
  (∀ (s : Finset (Fin 10)), (s.card = 8) → ∃ (stop : Fin 10), ∀ i ∈ s, stop ∉ routes i) ∧
  (∀ (s : Finset (Fin 10)), (s.card = 9) → ∀ (stop : Fin 10), ∃ i ∈ s, stop ∈ routes i)

theorem bus_routes_theorem : bus_routes_exist :=
sorry

end bus_routes_theorem_l291_291069


namespace largest_consecutive_odd_number_is_27_l291_291271

theorem largest_consecutive_odd_number_is_27 (a b c : ℤ) 
  (h1: a + b + c = 75)
  (h2: c - a = 6)
  (h3: b = a + 2)
  (h4: c = a + 4) :
  c = 27 := 
sorry

end largest_consecutive_odd_number_is_27_l291_291271


namespace part_a_part_b_case1_part_b_case2_l291_291694

theorem part_a (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x1 / x2 + x2 / x1 = -9 / 4) : 
  p = -1 / 23 :=
sorry

theorem part_b_case1 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -3 / 8 :=
sorry

theorem part_b_case2 (x1 x2 p : ℝ) 
  (h1 : 4 * (x1 ^ 2) + x1 + 4 * p = 0)
  (h2 : 4 * (x2 ^ 2) + x2 + 4 * p = 0) 
  (h3 : x2 = x1^2 - 1) : 
  p = -15 / 8 :=
sorry

end part_a_part_b_case1_part_b_case2_l291_291694


namespace children_count_l291_291935

variable (M W C : ℕ)

theorem children_count (h1 : M = 2 * W) (h2 : W = 3 * C) (h3 : M + W + C = 300) : C = 30 := by
  sorry

end children_count_l291_291935


namespace find_two_digit_number_l291_291949

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l291_291949


namespace B_work_time_l291_291769

theorem B_work_time :
  (∀ A_efficiency : ℝ, A_efficiency = 1 / 12 → ∀ B_efficiency : ℝ, B_efficiency = A_efficiency * 1.2 → (1 / B_efficiency = 10)) :=
by
  intros A_efficiency A_efficiency_eq B_efficiency B_efficiency_eq
  sorry

end B_work_time_l291_291769


namespace probability_below_8_l291_291789

def prob_hit_10 := 0.20
def prob_hit_9 := 0.30
def prob_hit_8 := 0.10

theorem probability_below_8 : (1 - (prob_hit_10 + prob_hit_9 + prob_hit_8) = 0.40) :=
by
  sorry

end probability_below_8_l291_291789


namespace sandy_found_additional_money_l291_291248

-- Define the initial amount of money Sandy had
def initial_amount : ℝ := 13.99

-- Define the cost of the shirt
def shirt_cost : ℝ := 12.14

-- Define the cost of the jacket
def jacket_cost : ℝ := 9.28

-- Define the remaining amount after buying the shirt
def remaining_after_shirt : ℝ := initial_amount - shirt_cost

-- Define the additional money found in Sandy's pocket
def additional_found_money : ℝ := jacket_cost - remaining_after_shirt

-- State the theorem to prove the amount of additional money found
theorem sandy_found_additional_money :
  additional_found_money = 11.13 :=
by sorry

end sandy_found_additional_money_l291_291248


namespace simplify_exponent_l291_291298

theorem simplify_exponent :
  2000 * 2000^2000 = 2000^2001 :=
by
  sorry

end simplify_exponent_l291_291298


namespace cortney_downloads_all_files_in_2_hours_l291_291978

theorem cortney_downloads_all_files_in_2_hours :
  let speed := 2 -- internet speed in megabits per minute
  let file1 := 80 -- file size in megabits
  let file2 := 90 -- file size in megabits
  let file3 := 70 -- file size in megabits
  let time1 := file1 / speed -- time to download first file in minutes
  let time2 := file2 / speed -- time to download second file in minutes
  let time3 := file3 / speed -- time to download third file in minutes
  let total_time_minutes := time1 + time2 + time3
  let total_time_hours := total_time_minutes / 60
  total_time_hours = 2 :=
by
  sorry

end cortney_downloads_all_files_in_2_hours_l291_291978


namespace travel_cost_AB_l291_291406

theorem travel_cost_AB
  (distance_AB : ℕ)
  (booking_fee : ℕ)
  (cost_per_km_flight : ℝ)
  (correct_total_cost : ℝ)
  (h1 : distance_AB = 4000)
  (h2 : booking_fee = 150)
  (h3 : cost_per_km_flight = 0.12) :
  correct_total_cost = 630 :=
by
  sorry

end travel_cost_AB_l291_291406


namespace product_of_numbers_l291_291741

theorem product_of_numbers (x y : ℝ) 
  (h₁ : x + y = 8 * (x - y)) 
  (h₂ : x * y = 40 * (x - y)) : x * y = 4032 := 
by
  sorry

end product_of_numbers_l291_291741


namespace count_integers_with_zero_l291_291836

/-- There are 740 positive integers less than or equal to 3017 that contain the digit 0. -/
theorem count_integers_with_zero (n : ℕ) (h : n ≤ 3017) : 
  (∃ k : ℕ, k ≤ 3017 ∧ ∃ d : ℕ, d < 10 ∧ d ≠ 0 ∧ k / 10 ^ d % 10 = 0) ↔ n = 740 :=
by sorry

end count_integers_with_zero_l291_291836


namespace bank_account_balance_l291_291339

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end bank_account_balance_l291_291339


namespace probability_x_y_less_than_3_l291_291939

theorem probability_x_y_less_than_3 :
  let A := 6 * 2
  let triangle_area := (1 / 2) * 3 * 2
  let P := triangle_area / A
  P = 1 / 4 := by sorry

end probability_x_y_less_than_3_l291_291939


namespace find_B_days_l291_291928

noncomputable def work_rate_A := 1 / 15
noncomputable def work_rate_B (x : ℝ) := 1 / x

theorem find_B_days (x : ℝ) : 
  (5 * (work_rate_A + work_rate_B x) = 0.5833333333333334) →
  (x = 20) := 
by 
  intro h,
  sorry

end find_B_days_l291_291928


namespace fruit_seller_apples_l291_291933

theorem fruit_seller_apples : 
  ∃ (x : ℝ), (x * 0.6 = 420) → x = 700 :=
sorry

end fruit_seller_apples_l291_291933


namespace midpoint_one_sixth_one_ninth_l291_291349

theorem midpoint_one_sixth_one_ninth : (1 / 6 + 1 / 9) / 2 = 5 / 36 := by
  sorry

end midpoint_one_sixth_one_ninth_l291_291349


namespace lcm_of_lap_times_l291_291963

theorem lcm_of_lap_times :
  Nat.lcm (Nat.lcm 5 8) 10 = 40 := by
  sorry

end lcm_of_lap_times_l291_291963


namespace complement_event_l291_291795

def total_students : ℕ := 4
def males : ℕ := 2
def females : ℕ := 2
def choose2 (n : ℕ) := n * (n - 1) / 2

noncomputable def eventA : ℕ := males * females
noncomputable def eventB : ℕ := choose2 males
noncomputable def eventC : ℕ := choose2 females

theorem complement_event {total_students males females : ℕ}
  (h_total : total_students = 4)
  (h_males : males = 2)
  (h_females : females = 2) :
  (total_students.choose 2 - (eventB + eventC)) / total_students.choose 2 = 1 / 3 :=
by
  sorry

end complement_event_l291_291795


namespace initial_distance_proof_l291_291913

noncomputable def initial_distance (V_A V_B T : ℝ) : ℝ :=
  (V_A * T) + (V_B * T)

theorem initial_distance_proof 
  (V_A V_B : ℝ) 
  (T : ℝ) 
  (h1 : V_A / V_B = 5 / 6)
  (h2 : V_B = 90)
  (h3 : T = 8 / 15) :
  initial_distance V_A V_B T = 88 := 
by
  -- proof goes here
  sorry

end initial_distance_proof_l291_291913


namespace matrix_determinant_zero_l291_291808

theorem matrix_determinant_zero (a b c : ℝ) : 
  Matrix.det (Matrix.of ![![1, a + b, b + c], ![1, a + 2 * b, b + 2 * c], ![1, a + 3 * b, b + 3 * c]]) = 0 := 
by
  sorry

end matrix_determinant_zero_l291_291808


namespace boat_speed_in_still_water_l291_291609

open Real

theorem boat_speed_in_still_water (V_s d t : ℝ) (h1 : V_s = 6) (h2 : d = 72) (h3 : t = 3.6) :
  ∃ (V_b : ℝ), V_b = 14 := by
  have V_d := d / t
  have V_b := V_d - V_s
  use V_b
  sorry

end boat_speed_in_still_water_l291_291609


namespace trader_gain_pens_l291_291791

theorem trader_gain_pens (C S : ℝ) (h1 : S = 1.25 * C) 
                         (h2 : 80 * S = 100 * C) : S - C = 0.25 * C :=
by
  have h3 : S = 1.25 * C := h1
  have h4 : 80 * S = 100 * C := h2
  sorry

end trader_gain_pens_l291_291791


namespace circle_center_polar_coords_l291_291530

noncomputable def polar_center (ρ θ : ℝ) : (ℝ × ℝ) :=
  (-1, 0)

theorem circle_center_polar_coords : 
  ∀ ρ θ : ℝ, ρ = -2 * Real.cos θ → polar_center ρ θ = (1, π) :=
by
  intro ρ θ h
  sorry

end circle_center_polar_coords_l291_291530


namespace marble_probability_l291_291160

theorem marble_probability
  (total_marbles : ℕ)
  (blue_marbles : ℕ)
  (green_marbles : ℕ)
  (draws : ℕ)
  (prob_first_green : ℚ)
  (prob_second_blue_given_green : ℚ)
  (total_prob : ℚ)
  (h_total : total_marbles = 10)
  (h_blue : blue_marbles = 4)
  (h_green : green_marbles = 6)
  (h_draws : draws = 2)
  (h_prob_first_green : prob_first_green = 3 / 5)
  (h_prob_second_blue_given_green : prob_second_blue_given_green = 4 / 9)
  (h_total_prob : total_prob = 4 / 15) :
  prob_first_green * prob_second_blue_given_green = total_prob := sorry

end marble_probability_l291_291160


namespace length_real_axis_hyperbola_l291_291191

theorem length_real_axis_hyperbola (a : ℝ) (h : a^2 = 4) : 2 * a = 4 := by
  sorry

end length_real_axis_hyperbola_l291_291191


namespace min_tangent_length_l291_291378

-- Definitions and conditions as given in the problem context
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 3 = 0

def symmetry_line (a b x y : ℝ) : Prop :=
  2 * a * x + b * y + 6 = 0

-- Proving the minimum length of the tangent line
theorem min_tangent_length (a b : ℝ) (h_sym : ∀ x y, circle_equation x y → symmetry_line a b x y) :
  ∃ l, l = 4 :=
sorry

end min_tangent_length_l291_291378


namespace real_solutions_of_polynomial_l291_291335

theorem real_solutions_of_polynomial :
  ∀ x : ℝ, x^4 - 3 * x^3 + x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end real_solutions_of_polynomial_l291_291335


namespace largest_prime_factor_of_85_l291_291450

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) : Prop :=
  is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_85 :
  let a := 65
  let b := 85
  let c := 91
  let d := 143
  let e := 169
  largest_prime_factor b 17 :=
by
  sorry

end largest_prime_factor_of_85_l291_291450


namespace spending_percentage_A_l291_291608

def combined_salary (S_A S_B : ℝ) : Prop := S_A + S_B = 7000
def A_salary (S_A : ℝ) : Prop := S_A = 5250
def B_salary (S_B : ℝ) : Prop := S_B = 1750
def B_spending (P_B : ℝ) : Prop := P_B = 0.85
def same_savings (S_A S_B P_A P_B : ℝ) : Prop := S_A * (1 - P_A) = S_B * (1 - P_B)
def A_spending (P_A : ℝ) : Prop := P_A = 0.95

theorem spending_percentage_A (S_A S_B P_A P_B : ℝ) 
  (h1: combined_salary S_A S_B) 
  (h2: A_salary S_A) 
  (h3: B_salary S_B) 
  (h4: B_spending P_B) 
  (h5: same_savings S_A S_B P_A P_B) : A_spending P_A :=
sorry

end spending_percentage_A_l291_291608


namespace minimum_trees_l291_291854

variable (Trees : Type) [Fintype Trees] [DecidableEq Trees]

def trees_in_grove : Nat := 100

def tree_type := {birches, spruces, pines, aspens} : Set Trees

def condition (s : Finset Trees) : Prop := 
  s.card > 85 → tree_type ⊆ s

theorem minimum_trees (s : Finset Trees) (H : condition s) : 
  ∃ (n : Nat), n ≤ trees_in_grove ∧ n ≥ 69 → 
  ∃ t ⊆ s, t.card = n ∧ (|t ∩ tree_type| >= 3) :=
sorry

end minimum_trees_l291_291854


namespace ratio_of_fractions_proof_l291_291454

noncomputable def ratio_of_fractions (x y : ℝ) : Prop :=
  (5 * x = 6 * y) → (x ≠ 0 ∧ y ≠ 0) → ((1/3) * x / ((1/5) * y) = 2)

theorem ratio_of_fractions_proof (x y : ℝ) (hx: 5 * x = 6 * y) (hnz: x ≠ 0 ∧ y ≠ 0) : ((1/3) * x / ((1/5) * y) = 2) :=
  by 
  sorry

end ratio_of_fractions_proof_l291_291454


namespace find_two_digit_number_l291_291948

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l291_291948


namespace dave_added_apps_l291_291801

-- Define the conditions as a set of given facts
def initial_apps : Nat := 10
def deleted_apps : Nat := 17
def remaining_apps : Nat := 4

-- The statement to prove
theorem dave_added_apps : ∃ x : Nat, initial_apps + x - deleted_apps = remaining_apps ∧ x = 11 :=
by
  use 11
  sorry

end dave_added_apps_l291_291801


namespace smallest_solution_x_squared_abs_x_eq_3x_plus_4_l291_291192

theorem smallest_solution_x_squared_abs_x_eq_3x_plus_4 :
  ∃ x : ℝ, x^2 * |x| = 3 * x + 4 ∧ ∀ y : ℝ, (y^2 * |y| = 3 * y + 4 → y ≥ x) := 
sorry

end smallest_solution_x_squared_abs_x_eq_3x_plus_4_l291_291192


namespace find_two_digit_number_l291_291941

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l291_291941


namespace f_three_equals_322_l291_291891

def f (z : ℝ) : ℝ := (z^2 - 2) * ((z^2 - 2)^2 - 3)

theorem f_three_equals_322 :
  f 3 = 322 :=
by
  -- Proof steps (left out intentionally as per instructions)
  sorry

end f_three_equals_322_l291_291891


namespace sum_of_roots_eq_two_l291_291101

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end sum_of_roots_eq_two_l291_291101


namespace decompose_96_l291_291802

theorem decompose_96 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 96) (h4 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) :=
sorry

end decompose_96_l291_291802


namespace percentage_more_l291_291714

variable (J : ℝ) -- Juan's income
noncomputable def Tim_income := 0.60 * J -- T = 0.60J
noncomputable def Mart_income := 0.84 * J -- M = 0.84J

theorem percentage_more {J : ℝ} (T := Tim_income J) (M := Mart_income J) :
  ((M - T) / T) * 100 = 40 := by
  sorry

end percentage_more_l291_291714


namespace expected_heads_40_after_conditions_l291_291711

-- Defining the problem conditions in Lean
def fairCoin : ProbabilityMassFunction Bool :=
  ProbabilityMassFunction.of_multiset [tt, ff]

-- Defining the probabilities after each coin toss
noncomputable def successive_tosses (n : ℕ) : ProbabilityMassFunction Bool :=
  ProbabilityMassFunction.bind fairCoin (fun b =>
    if b then
      if n ≤ 1 then fairCoin else successive_tosses (n - 1)
    else successive_tosses (n - 1))

-- Expected number of heads from 80 coins
noncomputable def expected_heads_after_tosses (n_coins : ℕ) (max_tosses : ℕ) : ℝ :=
  (n_coins : ℝ) * (successive_tosses max_tosses).to_dist (λ b, if b then 1 else 0)

-- The main theorem to prove: the expected number of heads after 3 tosses
theorem expected_heads_40_after_conditions : expected_heads_after_tosses 80 3 = 40 := by
  sorry

end expected_heads_40_after_conditions_l291_291711


namespace initial_quantity_of_milk_l291_291452

theorem initial_quantity_of_milk (A B C : ℝ) 
    (h1 : B = 0.375 * A)
    (h2 : C = 0.625 * A)
    (h3 : B + 148 = C - 148) : A = 1184 :=
by
  sorry

end initial_quantity_of_milk_l291_291452


namespace cube_faces_opposite_10_is_8_l291_291730

theorem cube_faces_opposite_10_is_8 (nums : Finset ℕ) (h_nums : nums = {6, 7, 8, 9, 10, 11})
  (sum_lateral_first : ℕ) (h_sum_lateral_first : sum_lateral_first = 36)
  (sum_lateral_second : ℕ) (h_sum_lateral_second : sum_lateral_second = 33)
  (faces_opposite_10 : ℕ) (h_faces_opposite_10 : faces_opposite_10 ∈ nums) :
  faces_opposite_10 = 8 :=
by
  sorry

end cube_faces_opposite_10_is_8_l291_291730


namespace two_pt_seven_five_as_fraction_l291_291432

-- Define the decimal value 2.75
def decimal_value : ℚ := 11 / 4

-- Define the question
theorem two_pt_seven_five_as_fraction : 2.75 = decimal_value := by
  sorry

end two_pt_seven_five_as_fraction_l291_291432


namespace max_principals_in_10_years_l291_291317

theorem max_principals_in_10_years (term_length : ℕ) (period_length : ℕ) (max_principals : ℕ)
  (term_length_eq : term_length = 4) (period_length_eq : period_length = 10) :
  max_principals = 4 :=
by
  sorry

end max_principals_in_10_years_l291_291317


namespace students_in_cars_l291_291884

theorem students_in_cars (total_students : ℕ := 396) (buses : ℕ := 7) (students_per_bus : ℕ := 56) :
  total_students - (buses * students_per_bus) = 4 := by
  sorry

end students_in_cars_l291_291884


namespace trapezoid_area_l291_291226

theorem trapezoid_area (l : ℝ) (r : ℝ) (a b : ℝ) (h : ℝ) (A : ℝ) :
  l = 9 →
  r = 4 →
  a + b = l + l →
  h = 2 * r →
  (a + b) / 2 * h = A →
  A = 72 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end trapezoid_area_l291_291226


namespace cone_volume_divided_by_pi_l291_291301

noncomputable def volume_of_cone_divided_by_pi (r : ℝ) (angle : ℝ) : ℝ :=
  if angle = 270 ∧ r = 20 then
    let base_circumference := 30 * Real.pi in
    let base_radius := 15 in
    let slant_height := r in
    let height := Real.sqrt (slant_height ^ 2 - base_radius ^ 2)
    let volume := (1 / 3) * Real.pi * base_radius ^ 2 * height in
    volume / Real.pi
  else 0

theorem cone_volume_divided_by_pi : 
  volume_of_cone_divided_by_pi 20 270 = 375 * Real.sqrt 7 :=
by
  sorry

end cone_volume_divided_by_pi_l291_291301


namespace negation_proposition_l291_291421

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 < 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≥ 0) :=
by 
  sorry

end negation_proposition_l291_291421


namespace find_a5_l291_291997

-- Definitions related to the conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℕ) : Prop :=
  b * b = a * c

-- Main theorem statement
theorem find_a5 (a : ℕ → ℕ) (h_arith : arithmetic_sequence a) (h_a3 : a 3 = 3)
  (h_geo : geometric_sequence (a 1) (a 2) (a 4)) :
  a 5 = 5 ∨ a 5 = 3 :=
  sorry

end find_a5_l291_291997


namespace mean_is_six_greater_than_median_l291_291492

theorem mean_is_six_greater_than_median (x a : ℕ) 
  (h1 : (x + a) + (x + 4) + (x + 7) + (x + 37) + x == 5 * (x + 10)) :
  a = 2 :=
by
  -- proof goes here
  sorry

end mean_is_six_greater_than_median_l291_291492


namespace problem_l291_291455

theorem problem (x : ℝ) (h : x + 1 / x = 5) : x ^ 2 + (1 / x) ^ 2 = 23 := 
sorry

end problem_l291_291455


namespace train_length_is_correct_l291_291620

-- Defining the initial conditions
def train_speed_km_per_hr : Float := 90.0
def time_seconds : Float := 5.0

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed_km_per_hr : Float) : Float :=
  speed_km_per_hr * (1000.0 / 3600.0)

-- Calculate the length of the train in meters
def length_of_train (speed_km_per_hr : Float) (time_s : Float) : Float :=
  km_per_hr_to_m_per_s speed_km_per_hr * time_s

-- Theorem statement
theorem train_length_is_correct : length_of_train train_speed_km_per_hr time_seconds = 125.0 :=
by
  sorry

end train_length_is_correct_l291_291620


namespace tenth_pair_in_twentieth_row_l291_291368

noncomputable def pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if k = 0 ∨ k > n then (0, 0) else (k, n + 1 - k)

theorem tenth_pair_in_twentieth_row : pair_in_row 20 10 = (10, 11) := by
  sorry

end tenth_pair_in_twentieth_row_l291_291368


namespace pear_sales_ratio_l291_291787

theorem pear_sales_ratio : 
  ∀ (total_sold afternoon_sold morning_sold : ℕ), 
  total_sold = 420 ∧ afternoon_sold = 280 ∧ total_sold = afternoon_sold + morning_sold 
  → afternoon_sold / morning_sold = 2 :=
by 
  intros total_sold afternoon_sold morning_sold 
  intro h 
  have h_total : total_sold = 420 := h.1 
  have h_afternoon : afternoon_sold = 280 := h.2.1 
  have h_morning : total_sold = afternoon_sold + morning_sold := h.2.2
  sorry

end pear_sales_ratio_l291_291787


namespace flower_beds_fraction_l291_291169

noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ :=
  (1 / 2) * leg^2

noncomputable def fraction_of_yard_occupied_by_flower_beds : ℝ :=
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard

theorem flower_beds_fraction : 
  let leg_length := (25 - 15) / 2
  let area_of_one_triangle := isosceles_right_triangle_area leg_length
  let total_area_of_flower_beds := 2 * area_of_one_triangle
  let area_of_yard := 25 * 5
  total_area_of_flower_beds / area_of_yard = 1 / 5 :=
by
  sorry

end flower_beds_fraction_l291_291169


namespace jill_spent_50_percent_on_clothing_l291_291885

theorem jill_spent_50_percent_on_clothing (
  T : ℝ) (hT : T ≠ 0)
  (h : 0.05 * T * C + 0.10 * 0.30 * T = 0.055 * T):
  C = 0.5 :=
by
  sorry

end jill_spent_50_percent_on_clothing_l291_291885


namespace upstream_distance_l291_291783

theorem upstream_distance (v : ℝ) 
  (H1 : ∀ d : ℝ, (10 + v) * 2 = 28) 
  (H2 : (10 - v) * 2 = d) : d = 12 := by
  sorry

end upstream_distance_l291_291783


namespace standard_circle_eq_l291_291352

noncomputable def circle_equation : String :=
  "The standard equation of the circle whose center lies on the line y = -4x and is tangent to the line x + y - 1 = 0 at point P(3, -2) is (x - 1)^2 + (y + 4)^2 = 8"

theorem standard_circle_eq
  (center_x : ℝ)
  (center_y : ℝ)
  (tangent_line : ℝ → ℝ → Prop)
  (point : ℝ × ℝ)
  (eqn_line : ∀ x y, tangent_line x y ↔ x + y - 1 = 0)
  (center_on_line : ∀ x y, y = -4 * x → center_y = y)
  (point_on_tangent : point = (3, -2))
  (tangent_at_point : tangent_line (point.1) (point.2)) :
  (center_x = 1 ∧ center_y = -4 ∧ (∃ r : ℝ, r = 2 * Real.sqrt 2)) →
  (∀ x y, (x - 1)^2 + (y + 4)^2 = 8) := by
  sorry

end standard_circle_eq_l291_291352


namespace no_finite_set_of_non_parallel_vectors_l291_291245

theorem no_finite_set_of_non_parallel_vectors (N : ℕ) (hN : N > 3) :
  ¬ ∃ (G : Finset (ℝ × ℝ)), G.card > 2 * N ∧
      (∀ (H : Finset (ℝ × ℝ)), H ⊆ G ∧ H.card = N →
        (∃ (F : Finset (ℝ × ℝ)), F ⊆ G ∧ F ≠ H ∧ F.card = N - 1 ∧ (H ∪ F).sum = 0)) ∧
      (∀ (H : Finset (ℝ × ℝ)), H ⊆ G ∧ H.card = N →
        (∃ (F : Finset (ℝ × ℝ)), F ⊆ G ∧ F ≠ H ∧ F.card = N ∧ (H ∪ F).sum = 0)) :=
  sorry

end no_finite_set_of_non_parallel_vectors_l291_291245


namespace sum_of_roots_eq_two_l291_291099

theorem sum_of_roots_eq_two {b x1 x2 : ℝ} 
  (h : x1 ^ 2 - 2 * x1 + b = 0) 
  (k : x2 ^ 2 - 2 * x2 + b = 0) 
  (neq : x1 ≠ x2) : 
  x1 + x2 = 2 := 
sorry

end sum_of_roots_eq_two_l291_291099


namespace min_operator_result_l291_291695

theorem min_operator_result : 
  min ((-3) + (-6)) (min ((-3) - (-6)) (min ((-3) * (-6)) ((-3) / (-6)))) = -9 := 
by 
  sorry

end min_operator_result_l291_291695


namespace maximum_value_F_l291_291511

noncomputable def f (x : Real) : Real := Real.sin x + Real.cos x

noncomputable def f' (x : Real) : Real := Real.cos x - Real.sin x

noncomputable def F (x : Real) : Real := f x * f' x + (f x) ^ 2

theorem maximum_value_F : ∃ x : Real, F x = 1 + Real.sqrt 2 :=
by
  -- The proof steps are to be added here.
  sorry

end maximum_value_F_l291_291511


namespace negation_of_p_l291_291513

theorem negation_of_p :
  (¬ (∀ x : ℝ, x^3 + 2 < 0)) = ∃ x : ℝ, x^3 + 2 ≥ 0 := 
  by sorry

end negation_of_p_l291_291513


namespace circular_permutation_divisible_41_l291_291779

theorem circular_permutation_divisible_41 (N : ℤ) (a b c d e : ℤ) (h : N = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e)
  (h41 : 41 ∣ N) :
  ∀ (k : ℕ), 41 ∣ (10^((k % 5) * (4 - (k / 5))) * a + 10^((k % 5) * 3 + (k / 5) * 4) * b + 10^((k % 5) * 2 + (k / 5) * 3) * c + 10^((k % 5) + (k / 5) * 2) * d + 10^(k / 5) * e) :=
sorry

end circular_permutation_divisible_41_l291_291779


namespace smallest_positive_integer_l291_291308

-- Definitions of the conditions
def condition1 (k : ℕ) : Prop := k % 10 = 9
def condition2 (k : ℕ) : Prop := k % 9 = 8
def condition3 (k : ℕ) : Prop := k % 8 = 7
def condition4 (k : ℕ) : Prop := k % 7 = 6
def condition5 (k : ℕ) : Prop := k % 6 = 5
def condition6 (k : ℕ) : Prop := k % 5 = 4
def condition7 (k : ℕ) : Prop := k % 4 = 3
def condition8 (k : ℕ) : Prop := k % 3 = 2
def condition9 (k : ℕ) : Prop := k % 2 = 1

-- Statement of the problem
theorem smallest_positive_integer : ∃ k : ℕ, 
  k > 0 ∧
  condition1 k ∧ 
  condition2 k ∧ 
  condition3 k ∧ 
  condition4 k ∧ 
  condition5 k ∧ 
  condition6 k ∧ 
  condition7 k ∧ 
  condition8 k ∧ 
  condition9 k ∧
  k = 2519 := 
sorry

end smallest_positive_integer_l291_291308


namespace emma_final_balance_correct_l291_291341

def emma_initial_savings : ℕ := 230
def withdrawal_amount : ℕ := 60
def deposit_amount : ℕ := 2 * withdrawal_amount
def final_amount_in_account : ℕ := emma_initial_savings - withdrawal_amount + deposit_amount

theorem emma_final_balance_correct : final_amount_in_account = 290 := by
  sorry

end emma_final_balance_correct_l291_291341


namespace baron_munchausen_correct_l291_291628

noncomputable def P (x : ℕ) : ℕ := sorry -- Assume non-constant polynomial with non-negative integer coefficients
noncomputable def Q (x : ℕ) : ℕ := sorry -- Assume non-constant polynomial with non-negative integer coefficients

theorem baron_munchausen_correct (b p0 : ℕ) 
  (hP2 : P 2 = b) 
  (hPp2 : P b = p0) 
  (hQ2 : Q 2 = b) 
  (hQp2 : Q b = p0) : 
  P = Q := sorry

end baron_munchausen_correct_l291_291628


namespace fraction_of_number_l291_291128

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l291_291128


namespace units_digit_3542_pow_876_l291_291792

theorem units_digit_3542_pow_876 : (3542 ^ 876) % 10 = 6 := by 
  sorry

end units_digit_3542_pow_876_l291_291792


namespace sectors_containing_all_numbers_l291_291777

theorem sectors_containing_all_numbers (n : ℕ) (h : 0 < n) :
  ∃ (s : Finset (Fin (2 * n))), (s.card = n) ∧ (∀ i : Fin n, ∃ j : Fin (2 * n), j ∈ s ∧ (j.val % n) + 1 = i.val) :=
  sorry

end sectors_containing_all_numbers_l291_291777


namespace setA_not_determined_l291_291918

-- Define the set of points more than 1 unit away from the origin
def setB : Set ℝ := {x | x > 1 ∨ x < -1}

-- Define the set of prime numbers less than 100
def setC : Set ℕ := {n | Prime n ∧ n < 100}

-- Define the set of solutions for the quadratic equation x^2 + 2x + 7 = 0
def setD : Set ℝ := {x | x^2 + 2 * x + 7 = 0}

-- The main theorem to prove
theorem setA_not_determined (A B C D : Set α) :
  ¬ (∃ (h : α → Prop), h = λ x, "x is a tall student at Chongqing No.1 Middle School") := 
sorry

end setA_not_determined_l291_291918


namespace solve_quadratic_polynomial_l291_291646

noncomputable def q (x : ℝ) : ℝ := -4.5 * x^2 + 4.5 * x + 135

theorem solve_quadratic_polynomial : 
  (q (-5) = 0) ∧ (q 6 = 0) ∧ (q 7 = -54) :=
by
  sorry

end solve_quadratic_polynomial_l291_291646


namespace increased_hypotenuse_length_l291_291868

theorem increased_hypotenuse_length :
  let AB := 24
  let BC := 10
  let AB' := AB + 6
  let BC' := BC + 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let AC' := Real.sqrt (AB'^2 + BC'^2)
  AC' - AC = 8 := by
  sorry

end increased_hypotenuse_length_l291_291868


namespace total_rabbits_and_chickens_l291_291738

theorem total_rabbits_and_chickens (r c : ℕ) (h₁ : r = 64) (h₂ : r = c + 17) : r + c = 111 :=
by {
  sorry
}

end total_rabbits_and_chickens_l291_291738


namespace claire_shirts_proof_l291_291089

theorem claire_shirts_proof : 
  ∀ (brian_shirts andrew_shirts steven_shirts claire_shirts : ℕ),
    brian_shirts = 3 →
    andrew_shirts = 6 * brian_shirts →
    steven_shirts = 4 * andrew_shirts →
    claire_shirts = 5 * steven_shirts →
    claire_shirts = 360 := 
by
  intro brian_shirts andrew_shirts steven_shirts claire_shirts
  intros h_brian h_andrew h_steven h_claire
  sorry

end claire_shirts_proof_l291_291089


namespace two_point_seven_five_as_fraction_l291_291440

theorem two_point_seven_five_as_fraction : 2.75 = 11 / 4 :=
by
  sorry

end two_point_seven_five_as_fraction_l291_291440


namespace divisors_of_10_factorial_larger_than_9_factorial_l291_291517

theorem divisors_of_10_factorial_larger_than_9_factorial :
  ∃ n, n = 9 ∧ (∀ d, d ∣ (Nat.factorial 10) → d > (Nat.factorial 9) → d > (Nat.factorial 1) → n = 9) :=
sorry

end divisors_of_10_factorial_larger_than_9_factorial_l291_291517


namespace percentage_error_edge_percentage_error_edge_l291_291476

open Real

-- Define the main context, E as the actual edge and E' as the calculated edge
variables (E E' : ℝ)

-- Condition: Error in calculating the area is 4.04%
axiom area_error : E' * E' = E * E * 1.0404

-- Statement: To prove that the percentage error in edge calculation is 2%
theorem percentage_error_edge : (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

-- Alternatively, include variable and condition definitions in the actual theorem statement
theorem percentage_error_edge' (E E' : ℝ) (h : E' * E' = E * E * 1.0404) : 
    (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

end percentage_error_edge_percentage_error_edge_l291_291476


namespace probability_second_even_given_first_even_l291_291465

/-- A fair six-sided die is rolled twice in succession. 
    Given that the outcome of the first roll is an even number, 
    what is the probability that the second roll also results in an even number? -/
theorem probability_second_even_given_first_even :
  (P := @classical.Probability (fin 6) (fun i => (i + 1) ∈ {1, 2, 3, 4, 5, 6})) →
  let A := {i : (fin 6) | (i.val % 2) = 0} in
  let B := {j : (fin 6) | (j.val % 2) = 0} in
  ∀ i (hA : i ∈ A), Classical.Probability.toReal (B ∣ A) = 1 / 2 :=
begin
  intros,
  sorry,
end

end probability_second_even_given_first_even_l291_291465


namespace find_PA_values_l291_291409

theorem find_PA_values :
  ∃ P A : ℕ, 10 ≤ P * 10 + A ∧ P * 10 + A < 100 ∧
            (P * 10 + A) ^ 2 / 1000 = P ∧ (P * 10 + A) ^ 2 % 10 = A ∧
            ((P = 9 ∧ A = 5) ∨ (P = 9 ∧ A = 6)) := by
  sorry

end find_PA_values_l291_291409


namespace triangle_inequality_squared_l291_291764

theorem triangle_inequality_squared {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (habc : a + b > c) (hbca : b + c > a) (hcab : c + a > b) :
    a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a) := sorry

end triangle_inequality_squared_l291_291764


namespace quadratic_coefficient_conversion_l291_291003

theorem quadratic_coefficient_conversion :
  ∀ x : ℝ, (3 * x^2 - 1 = 5 * x) → (3 * x^2 - 5 * x - 1 = 0) :=
by
  intros x h
  rw [←sub_eq_zero, ←h]
  ring

end quadratic_coefficient_conversion_l291_291003


namespace min_trees_for_three_types_l291_291849

-- Define types and trees
inductive TreeType
| birch | spruce | pine | aspen
deriving Inhabited, DecidableEq

-- A grove with 100 trees of any of the four types.
structure Tree :=
(type : TreeType)

constant grove : List Tree
axiom grove_size : grove.length = 100

-- Condition: Among any 85 trees, there are trees of all four types.
axiom all_types_in_any_85 : ∀ (s : Finset Tree), s.card = 85 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, ∃ t4 ∈ s, 
      t1.type = TreeType.birch ∧
      t2.type = TreeType.spruce ∧
      t3.type = TreeType.pine ∧
      t4.type = TreeType.aspen

-- We need to show that at least 69 trees are needed to ensure at least 3 types.
theorem min_trees_for_three_types : 
  ∀ (s : Finset Tree), s.card = 69 → 
  ∃ t1 ∈ s, ∃ t2 ∈ s, ∃ t3 ∈ s, 
      t1.type ≠ t2.type ∧ t2.type ≠ t3.type ∧ t1.type ≠ t3.type := 
sorry

end min_trees_for_three_types_l291_291849


namespace faster_by_airplane_l291_291176

theorem faster_by_airplane : 
  let driving_time := 3 * 60 + 15 
  let airport_drive := 10
  let wait_to_board := 20
  let flight_duration := driving_time / 3
  let exit_plane := 10
  driving_time - (airport_drive + wait_to_board + flight_duration + exit_plane) = 90 := 
by
  let driving_time : ℕ := 3 * 60 + 15
  let airport_drive : ℕ := 10
  let wait_to_board : ℕ := 20
  let flight_duration : ℕ := driving_time / 3
  let exit_plane : ℕ := 10
  have h1 : driving_time = 195 := rfl
  have h2 : flight_duration = 65 := by norm_num [h1]
  have h3 : 195 - (10 + 20 + 65 + 10) = 195 - 105 := by norm_num
  have h4 : 195 - 105 = 90 := by norm_num
  exact h4

end faster_by_airplane_l291_291176


namespace corrected_mean_l291_291457

theorem corrected_mean (mean : ℝ) (num_observations : ℕ) 
  (incorrect_observation correct_observation : ℝ)
  (h_mean : mean = 36) (h_num_observations : num_observations = 50)
  (h_incorrect_observation : incorrect_observation = 23) 
  (h_correct_observation : correct_observation = 44)
  : (mean * num_observations + (correct_observation - incorrect_observation)) / num_observations = 36.42 := 
by
  sorry

end corrected_mean_l291_291457


namespace probability_of_odd_product_lt_25_l291_291889

def ball_numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7]

noncomputable def choices : ℕ × ℕ → ℕ :=
λ (x : ℕ × ℕ), (x.1 * x.2)

noncomputable def is_odd : ℕ → Bool := λ n, (n % 2 = 1)

noncomputable def valid_pair (x : ℕ × ℕ) : Prop :=
is_odd x.1 ∧ is_odd x.2 ∧ x.1 * x.2 < 25

noncomputable def probability_valid_product_lt_25 : ℚ :=
12 / 49

theorem probability_of_odd_product_lt_25 :
  (∑ i in ball_numbers.product ball_numbers, if valid_pair i then 1 else 0 : ℚ) / (∑ i in ball_numbers.product ball_numbers, 1 : ℚ) = probability_valid_product_lt_25 :=
by {
  sorry
}

end probability_of_odd_product_lt_25_l291_291889


namespace unique_function_l291_291810

noncomputable def f : ℝ → ℝ := sorry

theorem unique_function 
  (h_f : ∀ x > 0, ∀ y > 0, f x * f y = 2 * f (x + y * f x)) : ∀ x > 0, f x = 2 :=
by
  sorry

end unique_function_l291_291810


namespace min_tangent_length_is_4_l291_291377

-- Define the circle and symmetry conditions
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0
def symmetry_condition (a b : ℝ) : Prop := 2*a*(-1) + b*2 + 6 = 0

-- Define the length of the tangent line from (a, b) to the circle center (-1, 2)
def min_tangent_length (a b : ℝ) : ℝ :=
  let d := Real.sqrt ((a + 1)^2 + (b - 2)^2) in
  d - Real.sqrt 2

-- Prove that the minimum tangent length is 4 given the conditions
theorem min_tangent_length_is_4 (a b : ℝ) :
  symmetry_condition a b →
  ∃ (min_len : ℝ), min_len = min_tangent_length a b ∧ min_len = 4 :=
by
  sorry

end min_tangent_length_is_4_l291_291377


namespace probability_each_guest_gets_one_of_each_kind_l291_291164

noncomputable def calc_probability : ℚ := 
  (3.factorial * 3.factorial * 3.factorial : ℚ) / (9.factorial : ℚ)

def sanitize_fraction (p : ℚ) : ℚ := 
  p.num / p.denom

theorem probability_each_guest_gets_one_of_each_kind :
  let m := 9 in
  let n := 70 in
  sanitize_fraction calc_probability = 9 / 70 →

  m + n = 79 :=
by
  intros
  simp [sanitize_fraction, calc_probability]
  sorry

end probability_each_guest_gets_one_of_each_kind_l291_291164


namespace find_two_digit_number_l291_291950

open Nat

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem find_two_digit_number :
  ∃ N : ℕ, 
    (10 ≤ N ∧ N < 100) ∧ 
    (N % 2 = 1) ∧ 
    (N % 9 = 0) ∧ 
    is_perfect_square ((N / 10) * (N % 10)) ∧ 
    N = 99 :=
by
  sorry

end find_two_digit_number_l291_291950


namespace solve_for_k_l291_291631

theorem solve_for_k (t s k : ℝ) :
  (∀ t s : ℝ, (∃ t s : ℝ, (⟨1, 4⟩ : ℝ × ℝ) + t • ⟨5, -3⟩ = ⟨0, 1⟩ + s • ⟨-2, k⟩) → false) ↔ k = 6 / 5 :=
by
  sorry

end solve_for_k_l291_291631


namespace solution1_solution2_l291_291086

-- Define the first problem
def equation1 (x : ℝ) : Prop :=
  (x + 1) / 3 - 1 = (x - 1) / 2

-- Prove that x = -1 is the solution to the first problem
theorem solution1 : equation1 (-1) := 
by 
  sorry

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  x - y = 1 ∧ 3 * x + y = 7

-- Prove that x = 2 and y = 1 are the solutions to the system of equations
theorem solution2 : system_of_equations 2 1 :=
by 
  sorry

end solution1_solution2_l291_291086


namespace fraction_of_yard_occupied_l291_291170

-- Define the rectangular yard with given length and width
def yard_length : ℝ := 25
def yard_width : ℝ := 5

-- Define the isosceles right triangle and the parallel sides of the trapezoid
def parallel_side1 : ℝ := 15
def parallel_side2 : ℝ := 25
def triangle_leg : ℝ := (parallel_side2 - parallel_side1) / 2

-- Areas
def triangle_area : ℝ := (1 / 2) * triangle_leg ^ 2
def flower_beds_area : ℝ := 2 * triangle_area
def yard_area : ℝ := yard_length * yard_width

-- Fraction calculation
def fraction_occupied : ℝ := flower_beds_area / yard_area

-- The proof statement
theorem fraction_of_yard_occupied:
  fraction_occupied = 1 / 5 :=
by
  sorry

end fraction_of_yard_occupied_l291_291170


namespace ram_leela_money_next_week_l291_291723

theorem ram_leela_money_next_week (x : ℕ)
  (initial_money : ℕ := 100)
  (total_money_after_52_weeks : ℕ := 1478)
  (sum_of_series : ℕ := 1378) :
  let n := 52
  let a1 := x
  let an := x + 51
  let S := (n / 2) * (a1 + an)
  initial_money + S = total_money_after_52_weeks → x = 1 :=
by
  sorry

end ram_leela_money_next_week_l291_291723


namespace two_digit_number_is_9_l291_291952

def dig_product (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n);
  match digits with
  | [a, b] => a * b
  | _ => 0

theorem two_digit_number_is_9 :
  ∃ (M : ℕ), 
    10 ≤ M ∧ M < 100 ∧ -- M is a two-digit number
    Odd M ∧            -- M is odd
    9 ∣ M ∧            -- M is a multiple of 9
    ∃ k, dig_product M = k * k -- product of its digits is a perfect square
    ∧ M = 9 :=       -- the solution is M = 9
by
  sorry

end two_digit_number_is_9_l291_291952


namespace angle_between_vectors_is_45_degrees_l291_291639

-- Define the vectors
def u : ℝ × ℝ := (4, -1)
def v : ℝ × ℝ := (5, 3)

-- Define the theorem to prove the angle between these vectors is 45 degrees
theorem angle_between_vectors_is_45_degrees : 
  let dot_product := (4 * 5) + (-1 * 3)
  let norm_u := Real.sqrt ((4^2) + (-1)^2)
  let norm_v := Real.sqrt ((5^2) + (3^2))
  let cos_theta := dot_product / (norm_u * norm_v)
  let theta := Real.arccos cos_theta
  45 = (theta * 180 / Real.pi) :=
by
  sorry

end angle_between_vectors_is_45_degrees_l291_291639


namespace AM_GM_inequality_l291_291074

theorem AM_GM_inequality (a b : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  (a^n + b^n) / 2 ≥ ((a + b) / 2) ^ n :=
by
  sorry

end AM_GM_inequality_l291_291074


namespace solve_number_l291_291946

theorem solve_number :
  ∃ (M : ℕ), 
    (10 ≤ M ∧ M < 100) ∧ -- M is a two-digit number
    M % 2 = 1 ∧ -- M is odd
    M % 9 = 0 ∧ -- M is a multiple of 9
    let d₁ := M / 10, d₂ := M % 10 in -- digits of M
    d₁ * d₂ = (Nat.sqrt (d₁ * d₂))^2 := -- product of digits is a perfect square
begin
  use 99,
  split,
  { -- 10 ≤ 99 < 100
    exact and.intro (le_refl 99) (lt_add_one 99),
  },
  split,
  { -- 99 is odd
    exact nat.odd_iff.2 (nat.dvd_one.trans (nat.dvd_refl 2)),
  },
  split,
  { -- 99 is a multiple of 9
    exact nat.dvd_of_mod_eq_zero (by norm_num),
  },
  { -- product of digits is a perfect square
    let d₁ := 99 / 10,
    let d₂ := 99 % 10,
    have h : d₁ * d₂ = 9 * 9, by norm_num,
    rw h,
    exact (by norm_num : 81 = 9 ^ 2).symm
  }
end

end solve_number_l291_291946


namespace sum_first_12_terms_l291_291042

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean {α : Type} [Field α] (a b c : α) : Prop :=
b^2 = a * c

def sum_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
n * (a 1 + a n) / 2

theorem sum_first_12_terms 
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : arithmetic_sequence a 1)
  (h2 : geometric_mean (a 3) (a 6) (a 11)) :
  sum_arithmetic_sequence a 12 = 96 :=
sorry

end sum_first_12_terms_l291_291042


namespace abigail_time_to_finish_l291_291477

noncomputable def words_total : ℕ := 1000
noncomputable def words_per_30_min : ℕ := 300
noncomputable def words_already_written : ℕ := 200
noncomputable def time_per_word : ℝ := 30 / words_per_30_min

theorem abigail_time_to_finish :
  (words_total - words_already_written) * time_per_word = 80 :=
by
  sorry

end abigail_time_to_finish_l291_291477


namespace compute_usage_difference_l291_291593

theorem compute_usage_difference
  (usage_last_week : ℕ)
  (usage_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : usage_last_week = 91)
  (h2 : usage_per_day = 8)
  (h3 : days_in_week = 7) :
  (usage_last_week - usage_per_day * days_in_week) = 35 :=
  sorry

end compute_usage_difference_l291_291593


namespace megan_initial_acorns_l291_291879

def initial_acorns (given_away left: ℕ) : ℕ := 
  given_away + left

theorem megan_initial_acorns :
  initial_acorns 7 9 = 16 := 
by 
  unfold initial_acorns
  rfl

end megan_initial_acorns_l291_291879


namespace range_of_m_l291_291045

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem range_of_m {m : ℝ} :
  (∀ x : ℝ, (f x m = 0) → (∃ y z : ℝ, y ≠ z ∧ f y m = 0 ∧ f z m = 0)) ∧
  (∀ x : ℝ, f (1 - x) m ≥ -1)
  → (0 ≤ m ∧ m < 1) := 
sorry

end range_of_m_l291_291045


namespace pizza_pasta_cost_difference_l291_291572

variable (x y z : ℝ)
variable (A1 : 2 * x + 3 * y + 4 * z = 53)
variable (A2 : 5 * x + 6 * y + 7 * z = 107)

theorem pizza_pasta_cost_difference :
  x - z = 1 :=
by
  sorry

end pizza_pasta_cost_difference_l291_291572


namespace sum_of_selected_sections_l291_291601

-- Given volumes of a bamboo, we denote them as a1, a2, ..., a9 forming an arithmetic sequence.
-- Where the sum of the volumes of the top four sections is 3 liters, and the
-- sum of the volumes of the bottom three sections is 4 liters.

-- Definitions based on the conditions
def arith_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

variables {a : ℕ → ℝ} {d : ℝ}
variable (sum_top_four : a 1 + a 2 + a 3 + a 4 = 3)
variable (sum_bottom_three : a 7 + a 8 + a 9 = 4)
variable (seq_condition : arith_seq a d)

theorem sum_of_selected_sections 
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 7 + a 8 + a 9 = 4)
  (h_seq : arith_seq a d) : 
  a 2 + a 3 + a 8 = 17 / 6 := 
sorry -- proof goes here

end sum_of_selected_sections_l291_291601


namespace correct_operation_l291_291288

theorem correct_operation (x : ℝ) (hx : x ≠ 0) :
  (x^3 / x^2 = x) :=
by {
  sorry
}

end correct_operation_l291_291288


namespace year_2024_AD_representation_l291_291794

def year_representation (y: Int) : Int :=
  if y > 0 then y else -y

theorem year_2024_AD_representation : year_representation 2024 = 2024 :=
by sorry

end year_2024_AD_representation_l291_291794


namespace range_of_a_l291_291659

noncomputable def f (a x : ℝ) := (1 / 3) * x^3 - x^2 - 3 * x - a

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ (-9 < a ∧ a < 5 / 3) :=
by apply sorry

end range_of_a_l291_291659


namespace polygon_number_of_sides_l291_291685

-- Define the given conditions
def each_interior_angle (n : ℕ) : ℕ := 120

-- Define the property to calculate the number of sides
def num_sides (each_exterior_angle : ℕ) : ℕ := 360 / each_exterior_angle

-- Statement of the problem
theorem polygon_number_of_sides : num_sides (180 - each_interior_angle 6) = 6 :=
by
  -- Proof is omitted
  sorry

end polygon_number_of_sides_l291_291685


namespace set_intersection_l291_291832

def U : Set ℝ := Set.univ
def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x ≥ 2}
def C_U_B : Set ℝ := {x | x < 2}

theorem set_intersection :
  A ∩ C_U_B = {-1, 0, 1} :=
sorry

end set_intersection_l291_291832


namespace bridge_extension_length_l291_291577

theorem bridge_extension_length (width_of_river length_of_existing_bridge additional_length_needed : ℕ)
  (h1 : width_of_river = 487)
  (h2 : length_of_existing_bridge = 295)
  (h3 : additional_length_needed = width_of_river - length_of_existing_bridge) :
  additional_length_needed = 192 :=
by {
  -- The steps of the proof would go here, but we use sorry for now.
  sorry
}

end bridge_extension_length_l291_291577


namespace calculation_correct_l291_291324

theorem calculation_correct : (3.456 - 1.234) * 0.5 = 1.111 :=
by
  sorry

end calculation_correct_l291_291324


namespace infinitely_many_arithmetic_progression_triples_l291_291026

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

end infinitely_many_arithmetic_progression_triples_l291_291026


namespace right_triangle_with_integer_sides_l291_291988

theorem right_triangle_with_integer_sides (k : ℤ) :
  ∃ (a b c : ℤ), a = 2*k+1 ∧ b = 2*k*(k+1) ∧ c = 2*k^2+2*k+1 ∧ (a^2 + b^2 = c^2) ∧ (c = a + 1) := by
  sorry

end right_triangle_with_integer_sides_l291_291988


namespace find_chosen_number_l291_291174

theorem find_chosen_number (x : ℤ) (h : 2 * x - 138 = 106) : x = 122 :=
by
  sorry

end find_chosen_number_l291_291174


namespace feasible_stations_l291_291275

theorem feasible_stations (n : ℕ) (h: n > 0) 
  (pairings : ∀ (i j : ℕ), i ≠ j → i < n → j < n → ∃ k, (i+k) % n = j ∨ (j+k) % n = i) : n = 4 :=
sorry

end feasible_stations_l291_291275


namespace expression_A_expression_B_expression_C_expression_D_l291_291315

theorem expression_A :
  (Real.sin (7 * Real.pi / 180) * Real.cos (23 * Real.pi / 180) + 
   Real.sin (83 * Real.pi / 180) * Real.cos (67 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_B :
  (2 * Real.cos (75 * Real.pi / 180) * Real.sin (75 * Real.pi / 180)) = 1 / 2 :=
sorry

theorem expression_C :
  (Real.sqrt 3 * Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) / 
   Real.sin (50 * Real.pi / 180) ≠ 1 / 2 :=
sorry

theorem expression_D :
  (1 / ((1 + Real.tan (27 * Real.pi / 180)) * (1 + Real.tan (18 * Real.pi / 180)))) = 1 / 2 :=
sorry

end expression_A_expression_B_expression_C_expression_D_l291_291315


namespace grove_tree_selection_l291_291858

theorem grove_tree_selection (birches spruces pines aspens : ℕ) :
  birches + spruces + pines + aspens = 100 →
  (∀ s : set ℕ, s.card = 85 → (birches ∈ s ∧ spruces ∈ s ∧ pines ∈ s ∧ aspens ∈ s)) →
  ∀ t : set ℕ, t.card = 69 → (birches ∈ t ∧ spruces ∈ t) ∨ (birches ∈ t ∧ pines ∈ t) ∨ (birches ∈ t ∧ aspens ∈ t) ∨ (spruces ∈ t ∧ pines ∈ t) ∨ (spruces ∈ t ∧ aspens ∈ t) ∨ (pines ∈ t ∧ aspens ∈ t) :=
sorry

end grove_tree_selection_l291_291858


namespace proportion_x_l291_291453

theorem proportion_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
sorry

end proportion_x_l291_291453


namespace cricket_run_rate_l291_291383

theorem cricket_run_rate (x : ℝ) (hx : 3.2 * x + 6.25 * 40 = 282) : x = 10 :=
by sorry

end cricket_run_rate_l291_291383


namespace arithmetic_avg_salary_technicians_l291_291094

noncomputable def avg_salary_technicians_problem : Prop :=
  let average_salary_all := 8000
  let total_workers := 21
  let average_salary_rest := 6000
  let technician_count := 7
  let total_salary_all := average_salary_all * total_workers
  let total_salary_rest := average_salary_rest * (total_workers - technician_count)
  let total_salary_technicians := total_salary_all - total_salary_rest
  let average_salary_technicians := total_salary_technicians / technician_count
  average_salary_technicians = 12000

theorem arithmetic_avg_salary_technicians :
  avg_salary_technicians_problem :=
by {
  sorry -- Proof is omitted as per instructions.
}

end arithmetic_avg_salary_technicians_l291_291094


namespace number_subtract_four_l291_291761

theorem number_subtract_four (x : ℤ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end number_subtract_four_l291_291761


namespace no_prime_pairs_sum_53_l291_291394

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_pairs_sum_53 :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_prime_pairs_sum_53_l291_291394


namespace average_weight_of_children_l291_291418

theorem average_weight_of_children (avg_weight_boys avg_weight_girls : ℕ)
                                   (num_boys num_girls : ℕ)
                                   (h1 : avg_weight_boys = 160)
                                   (h2 : avg_weight_girls = 110)
                                   (h3 : num_boys = 8)
                                   (h4 : num_girls = 5) :
                                   (num_boys * avg_weight_boys + num_girls * avg_weight_girls) / (num_boys + num_girls) = 141 :=
by
    sorry

end average_weight_of_children_l291_291418


namespace find_k_l291_291687

theorem find_k (k : ℝ) :
  (∃ x : ℝ, 8 * x - k = 2 * (x + 1) ∧ 2 * (2 * x - 3) = 1 - 3 * x) → k = 4 :=
by
  sorry

end find_k_l291_291687


namespace ratio_first_term_common_diff_l291_291286

theorem ratio_first_term_common_diff {a d : ℤ} 
  (S_20 : ℤ) (S_10 : ℤ)
  (h1 : S_20 = 10 * (2 * a + 19 * d))
  (h2 : S_10 = 5 * (2 * a + 9 * d))
  (h3 : S_20 = 6 * S_10) :
  a / d = 2 :=
by
  sorry

end ratio_first_term_common_diff_l291_291286


namespace fractional_inequality_solution_l291_291424

theorem fractional_inequality_solution :
  {x : ℝ | (2 * x - 1) / (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 1 / 2} := 
by
  sorry

end fractional_inequality_solution_l291_291424


namespace compute_usage_difference_l291_291592

theorem compute_usage_difference
  (usage_last_week : ℕ)
  (usage_per_day : ℕ)
  (days_in_week : ℕ)
  (h1 : usage_last_week = 91)
  (h2 : usage_per_day = 8)
  (h3 : days_in_week = 7) :
  (usage_last_week - usage_per_day * days_in_week) = 35 :=
  sorry

end compute_usage_difference_l291_291592


namespace minimum_value_f_l291_291551

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ t ≥ 0, ∀ (x y : ℝ), 0 < x → 0 < y → f x y ≥ t ∧ t = 4 * Real.sqrt 2 :=
sorry

end minimum_value_f_l291_291551


namespace division_remainder_l291_291096

theorem division_remainder (dividend divisor quotient : ℕ) (h_dividend : dividend = 131) (h_divisor : divisor = 14) (h_quotient : quotient = 9) :
  ∃ remainder : ℕ, dividend = divisor * quotient + remainder ∧ remainder = 5 :=
by
  sorry

end division_remainder_l291_291096


namespace min_value_a_l291_291999

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, (3 * x - 5 * y) ≥ 0 → x > 0 → y > 0 → (1 - a) * x ^ 2 + 2 * x * y - a * y ^ 2 ≤ 0) ↔ a ≥ 55 / 34 := 
by 
  sorry

end min_value_a_l291_291999


namespace arithmetic_progression_correct_l291_291348

noncomputable def nth_term_arithmetic_progression (n : ℕ) : ℝ :=
  4.2 * n + 9.3

def recursive_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  a 1 = 13.5 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n + 4.2

theorem arithmetic_progression_correct (n : ℕ) :
  (nth_term_arithmetic_progression n = 4.2 * n + 9.3) ∧
  ∀ (a : ℕ → ℝ), recursive_arithmetic_progression a → a n = 4.2 * n + 9.3 :=
by
  sorry

end arithmetic_progression_correct_l291_291348


namespace arithmetic_sequence_sum_l291_291446

theorem arithmetic_sequence_sum
  (a l : ℤ) (n d : ℤ)
  (h1 : a = -5) (h2 : l = 40) (h3 : d = 5)
  (h4 : l = a + (n - 1) * d) :
  (n / 2) * (a + l) = 175 :=
by
  sorry

end arithmetic_sequence_sum_l291_291446


namespace youngest_age_is_29_l291_291386

-- Define that the ages form an arithmetic sequence
def arithmetic_sequence (a1 a2 a3 a4 : ℕ) : Prop :=
  ∃ (d : ℕ), a2 = a1 + d ∧ a3 = a1 + 2*d ∧ a4 = a1 + 3*d

-- Define the problem statement
theorem youngest_age_is_29 (a1 a2 a3 a4 : ℕ) (h_seq : arithmetic_sequence a1 a2 a3 a4) (h_oldest : a4 = 50) (h_sum : a1 + a2 + a3 + a4 = 158) :
  a1 = 29 :=
by
  sorry

end youngest_age_is_29_l291_291386


namespace distance_between_points_l291_291242

theorem distance_between_points 
  (v_A v_B : ℝ) 
  (d : ℝ) 
  (h1 : 4 * v_A + 4 * v_B = d)
  (h2 : 3.5 * (v_A + 3) + 3.5 * (v_B + 3) = d) : 
  d = 168 := 
by 
  sorry

end distance_between_points_l291_291242


namespace jake_needs_total_hours_to_pay_off_debts_l291_291532

-- Define the conditions for the debts and payments
variable (debtA debtB debtC : ℝ)
variable (paymentA paymentB paymentC : ℝ)
variable (task1P task2P task3P task4P task5P task6P : ℝ)
variable (task2Payoff task4Payoff task6Payoff : ℝ)

-- Assume provided values
noncomputable def total_hours_needed : ℝ :=
  let remainingA := debtA - paymentA
  let remainingB := debtB - paymentB
  let remainingC := debtC - paymentC
  let hoursTask1 := (remainingA - task2Payoff) / task1P
  let hoursTask2 := task2Payoff / task2P
  let hoursTask3 := (remainingB - task4Payoff) / task3P
  let hoursTask4 := task4Payoff / task4P
  let hoursTask5 := (remainingC - task6Payoff) / task5P
  let hoursTask6 := task6Payoff / task6P
  hoursTask1 + hoursTask2 + hoursTask3 + hoursTask4 + hoursTask5 + hoursTask6

-- Given our specific problem conditions
theorem jake_needs_total_hours_to_pay_off_debts :
  total_hours_needed 150 200 250 60 80 100 15 12 20 10 25 30 30 40 60 = 20.1 :=
by
  sorry

end jake_needs_total_hours_to_pay_off_debts_l291_291532


namespace minimum_value_of_polynomial_l291_291347

-- Define the polynomial expression
def polynomial_expr (x : ℝ) : ℝ := (8 - x) * (6 - x) * (8 + x) * (6 + x)

-- State the theorem with the minimum value
theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial_expr x = -196 := by
  sorry

end minimum_value_of_polynomial_l291_291347


namespace optionB_is_difference_of_squares_l291_291142

-- Definitions from conditions
def A_expr (x : ℝ) : ℝ := (x - 2) * (x + 1)
def B_expr (x y : ℝ) : ℝ := (x + 2 * y) * (x - 2 * y)
def C_expr (x y : ℝ) : ℝ := (x + y) * (-x - y)
def D_expr (x : ℝ) : ℝ := (-x + 1) * (x - 1)

theorem optionB_is_difference_of_squares (x y : ℝ) : B_expr x y = x^2 - 4 * y^2 :=
by
  -- Proof is intentionally left out as per instructions
  sorry

end optionB_is_difference_of_squares_l291_291142


namespace sector_central_angle_l291_291379

theorem sector_central_angle (r α: ℝ) (hC: 4 * r = 2 * r + α * r): α = 2 :=
by
  -- Proof is to be filled in
  sorry

end sector_central_angle_l291_291379


namespace distribution_and_max_score_l291_291612

def XiaoMing_A : ℝ := 0.7
def XiaoMing_B : ℝ := 0.5

theorem distribution_and_max_score :
  let X := {0, 40, 100}
  let p0 := 1 - XiaoMing_A
  let p40 := XiaoMing_A * (1 - XiaoMing_B)
  let p100 := XiaoMing_A * XiaoMing_B
  let distX := { (0, p0), (40, p40), (100, p100) }
  let E_X := 0 * p0 + 40 * p40 + 100 * p100
  let p0_Y := 1 - XiaoMing_B
  let p60 := XiaoMing_B * (1 - XiaoMing_A)
  let p100_Y := XiaoMing_B * XiaoMing_A
  let E_Y := 0 * p0_Y + 60 * p60 + 100 * p100_Y
  E_X = 49 ∧ E_Y = 44 ∧ E_X > E_Y ∧ distX = {(0, 0.3), (40, 0.35), (100, 0.35)} :=
sorry

end distribution_and_max_score_l291_291612


namespace neg_of_all_men_are_honest_l291_291895

variable {α : Type} (man honest : α → Prop)

theorem neg_of_all_men_are_honest :
  ¬ (∀ x, man x → honest x) ↔ ∃ x, man x ∧ ¬ honest x :=
by
  sorry

end neg_of_all_men_are_honest_l291_291895


namespace students_without_glasses_l291_291907

theorem students_without_glasses (total_students : ℕ) (perc_glasses : ℕ) (students_with_glasses students_without_glasses : ℕ) 
  (h1 : total_students = 325) (h2 : perc_glasses = 40) (h3 : students_with_glasses = perc_glasses * total_students / 100)
  (h4 : students_without_glasses = total_students - students_with_glasses) : students_without_glasses = 195 := 
by
  sorry

end students_without_glasses_l291_291907


namespace fraction_of_number_l291_291126

theorem fraction_of_number : (7 / 8) * 48 = 42 := 
by sorry

end fraction_of_number_l291_291126


namespace part1_part2_l291_291502

variables {a m n : ℝ}

theorem part1 (h1 : a^m = 2) (h2 : a^n = 3) : a^(4*m + 3*n) = 432 :=
by sorry

theorem part2 (h1 : a^m = 2) (h2 : a^n = 3) : a^(5*m - 2*n) = 32 / 9 :=
by sorry

end part1_part2_l291_291502


namespace sum_ratio_l291_291829

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ := 
  a1 * (1 - q^n) / (1 - q)

theorem sum_ratio (a1 q : ℝ) 
  (h : 8 * (a1 * q) + (a1 * q^4) = 0) :
  geometric_sequence_sum a1 q 6 / geometric_sequence_sum a1 q 3 = -7 := 
by
  sorry

end sum_ratio_l291_291829


namespace find_working_hours_for_y_l291_291926

theorem find_working_hours_for_y (Wx Wy Wz Ww : ℝ) (h1 : Wx = 1/8)
  (h2 : Wy + Wz = 1/6) (h3 : Wx + Wz = 1/4) (h4 : Wx + Wy + Ww = 1/5)
  (h5 : Wx + Ww + Wz = 1/3) : 1 / Wy = 24 :=
by
  -- Given the conditions
  -- Wx = 1/8
  -- Wy + Wz = 1/6
  -- Wx + Wz = 1/4
  -- Wx + Wy + Ww = 1/5
  -- Wx + Ww + Wz = 1/3
  -- We need to prove that 1 / Wy = 24
  sorry

end find_working_hours_for_y_l291_291926


namespace wholesale_price_l291_291310

theorem wholesale_price (RP SP W : ℝ) (h1 : RP = 120)
  (h2 : SP = 0.9 * RP)
  (h3 : SP = W + 0.2 * W) : W = 90 :=
by
  sorry

end wholesale_price_l291_291310


namespace machine_purchase_price_l291_291733

theorem machine_purchase_price (P : ℝ) (h : 0.80 * P = 6400) : P = 8000 :=
by
  sorry

end machine_purchase_price_l291_291733


namespace board_division_condition_l291_291495

open Nat

theorem board_division_condition (n : ℕ) : 
  (∃ k : ℕ, n = 4 * k) ↔ 
  (∃ v h : ℕ, v = h ∧ (2 * v + 2 * h = n * n ∧ n % 2 = 0)) := 
sorry

end board_division_condition_l291_291495


namespace Annie_cookies_sum_l291_291968

theorem Annie_cookies_sum :
  let cookies_monday := 5
  let cookies_tuesday := 2 * cookies_monday
  let cookies_wednesday := cookies_tuesday + (40 / 100) * cookies_tuesday
  cookies_monday + cookies_tuesday + cookies_wednesday = 29 :=
by
  sorry

end Annie_cookies_sum_l291_291968


namespace original_amount_of_water_l291_291780

variable {W : ℝ} -- Assume W is a real number representing the original amount of water

theorem original_amount_of_water (h1 : 30 * 0.02 = 0.6) (h2 : 0.6 = 0.06 * W) : W = 10 :=
by
  sorry

end original_amount_of_water_l291_291780


namespace symmetric_point_with_respect_to_y_eq_x_l291_291272

variables (P : ℝ × ℝ) (line : ℝ → ℝ)

theorem symmetric_point_with_respect_to_y_eq_x (P : ℝ × ℝ) (hP : P = (1, 3)) (hline : ∀ x, line x = x) :
  (∃ Q : ℝ × ℝ, Q = (3, 1) ∧ Q = (P.snd, P.fst)) :=
by
  sorry

end symmetric_point_with_respect_to_y_eq_x_l291_291272
