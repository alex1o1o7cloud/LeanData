import Mathlib

namespace length_of_AB_l286_286796

theorem length_of_AB :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}
  let focus := (Real.sqrt 3, 0)
  let line := {p : ℝ × ℝ | p.2 = p.1 - Real.sqrt 3}
  ∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line ∧ B ∈ line ∧
  (dist A B = 8 / 5) :=
by
  sorry

end length_of_AB_l286_286796


namespace correct_sampling_methods_l286_286290

-- Define the surveys with their corresponding conditions
structure Survey1 where
  high_income : Nat
  middle_income : Nat
  low_income : Nat
  total_households : Nat

structure Survey2 where
  total_students : Nat
  sample_students : Nat
  differences_small : Bool
  sizes_small : Bool

-- Define the conditions
def survey1_conditions (s : Survey1) : Prop :=
  s.high_income = 125 ∧ s.middle_income = 280 ∧ s.low_income = 95 ∧ s.total_households = 100

def survey2_conditions (s : Survey2) : Prop :=
  s.total_students = 15 ∧ s.sample_students = 3 ∧ s.differences_small = true ∧ s.sizes_small = true

-- Define the answer predicate
def correct_answer (method1 method2 : String) : Prop :=
  method1 = "stratified sampling" ∧ method2 = "simple random sampling"

-- The theorem statement
theorem correct_sampling_methods (s1 : Survey1) (s2 : Survey2) :
  survey1_conditions s1 → survey2_conditions s2 → correct_answer "stratified sampling" "simple random sampling" :=
by
  -- Proof skipped for problem statement purpose
  sorry

end correct_sampling_methods_l286_286290


namespace smallest_consecutive_sum_perfect_square_l286_286860

theorem smallest_consecutive_sum_perfect_square :
  ∃ n : ℕ, (∑ i in (finset.range 20).map (λ i, n + i)) = 250 ∧ (∃ k : ℕ, 10 * (2 * n + 19) = k^2) :=
by
  sorry

end smallest_consecutive_sum_perfect_square_l286_286860


namespace find_g_values_l286_286729

open Function

-- Defining the function g and its properties
axiom g : ℝ → ℝ
axiom g_domain : ∀ x, 0 ≤ x → 0 ≤ g x
axiom g_proper : ∀ x, 0 ≤ x → 0 ≤ g (g x)
axiom g_func : ∀ x, 0 ≤ x → g (g x) = 3 * x / (x + 3)
axiom g_interval : ∀ x, 2 ≤ x ∧ x ≤ 3 → g x = (x + 1) / 2

-- Problem statement translating to Lean
theorem find_g_values :
  g 2021 = 2021.5 ∧ g (1 / 2021) = 6 := by {
  sorry 
}

end find_g_values_l286_286729


namespace correct_statement_l286_286024

variables {α β γ : ℝ → ℝ → ℝ → Prop} -- planes
variables {a b c : ℝ → ℝ → ℝ → Prop} -- lines

def is_parallel (P Q : ℝ → ℝ → ℝ → Prop) : Prop :=
∀ x : ℝ, ∀ y : ℝ, ∀ z : ℝ, (P x y z → Q x y z) ∧ (Q x y z → P x y z)

def is_perpendicular (L : ℝ → ℝ → ℝ → Prop) (P : ℝ → ℝ → ℝ → Prop) : Prop :=
∀ x : ℝ, ∀ y : ℝ, ∀ z : ℝ, L x y z ↔ ¬ P x y z 

theorem correct_statement : 
  (is_perpendicular a α) → 
  (is_parallel b β) → 
  (is_parallel α β) → 
  (is_perpendicular a b) :=
by
  sorry

end correct_statement_l286_286024


namespace calculate_f_ff_f60_l286_286294

def f (N : ℝ) : ℝ := 0.3 * N + 2

theorem calculate_f_ff_f60 : f (f (f 60)) = 4.4 := by
  sorry

end calculate_f_ff_f60_l286_286294


namespace cos_alpha_add_pi_over_4_l286_286494

theorem cos_alpha_add_pi_over_4 (x y r : ℝ) (α : ℝ) (h1 : P = (3, -4)) (h2 : r = Real.sqrt (x^2 + y^2)) (h3 : x / r = Real.cos α) (h4 : y / r = Real.sin α) :
  Real.cos (α + Real.pi / 4) = (7 * Real.sqrt 2) / 10 := by
  sorry

end cos_alpha_add_pi_over_4_l286_286494


namespace isosceles_triangle_perimeter_l286_286817

theorem isosceles_triangle_perimeter {a b : ℝ} (h1 : a = 3) (h2 : b = 1) :
  (a = 3 ∧ b = 1) ∧ (a + b > b ∨ b + b > a) → a + a + b = 7 :=
by
  sorry

end isosceles_triangle_perimeter_l286_286817


namespace find_smaller_number_l286_286858

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x = 5 * y) : x = 10 :=
sorry

end find_smaller_number_l286_286858


namespace find_m_l286_286490

theorem find_m (m x1 x2 : ℝ) (h1 : x1^2 + m * x1 + 5 = 0) (h2 : x2^2 + m * x2 + 5 = 0) (h3 : x1 = 2 * |x2| - 3) : 
  m = -9 / 2 :=
sorry

end find_m_l286_286490


namespace problem1_problem2_problem3_problem4_l286_286465

theorem problem1 : 24 - (-16) + (-25) - 32 = -17 := by
  sorry

theorem problem2 : (-1 / 2) * 2 / 2 * (-1 / 2) = 1 / 4 := by
  sorry

theorem problem3 : -2^2 * 5 - (-2)^3 * (1 / 8) + 1 = -18 := by
  sorry

theorem problem4 : ((-1 / 4) - (5 / 6) + (8 / 9)) / (-1 / 6)^2 + (-2)^2 * (-6)= -31 := by
  sorry

end problem1_problem2_problem3_problem4_l286_286465


namespace intersection_M_N_l286_286060

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l286_286060


namespace intervals_of_monotonicity_l286_286295

noncomputable def y (x : ℝ) : ℝ := 2 ^ (x^2 - 2*x + 4)

theorem intervals_of_monotonicity :
  (∀ x : ℝ, x > 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ < y x₂)) ∧
  (∀ x : ℝ, x < 1 → (∀ y₁ y₂ : ℝ, x₁ < x₂ → y x₁ > y x₂)) :=
by
  sorry

end intervals_of_monotonicity_l286_286295


namespace sum_of_altitudes_of_triangle_l286_286856

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle formed by the line with the coordinate axes
def forms_triangle_with_axes (x y : ℝ) : Prop := 
  line_eq x 0 ∧ line_eq 0 y

-- Prove the sum of the lengths of the altitudes is 511/17
theorem sum_of_altitudes_of_triangle : 
  ∃ x y : ℝ, forms_triangle_with_axes x y → 
  15 + 8 + (120 / 17) = 511 / 17 :=
by
  sorry

end sum_of_altitudes_of_triangle_l286_286856


namespace tan_alpha_sqrt3_l286_286180

theorem tan_alpha_sqrt3 (α : ℝ) (h : Real.sin (α + 20 * Real.pi / 180) = Real.cos (α + 10 * Real.pi / 180) + Real.cos (α - 10 * Real.pi / 180)) :
  Real.tan α = Real.sqrt 3 := 
  sorry

end tan_alpha_sqrt3_l286_286180


namespace max_square_test_plots_l286_286906

theorem max_square_test_plots
    (length : ℕ)
    (width : ℕ)
    (fence : ℕ)
    (fields_measure : length = 30 ∧ width = 45)
    (fence_measure : fence = 2250) :
  ∃ (number_of_plots : ℕ),
    number_of_plots = 150 :=
by
  sorry

end max_square_test_plots_l286_286906


namespace train_length_l286_286105

theorem train_length (L : ℝ) (V1 V2 : ℝ) 
  (h1 : V1 = L / 15) 
  (h2 : V2 = (L + 800) / 45) 
  (h3 : V1 = V2) : 
  L = 400 := 
sorry

end train_length_l286_286105


namespace problem_l286_286146

def otimes (x y : ℝ) : ℝ := x^3 + 5 * x * y - y

theorem problem (a : ℝ) : 
  otimes a (otimes a a) = 5 * a^4 + 24 * a^3 - 10 * a^2 + a :=
by
  sorry

end problem_l286_286146


namespace eugene_boxes_needed_l286_286936

-- Define the number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards not used
def unused_cards : ℕ := 16

-- Define the number of toothpicks per card
def toothpicks_per_card : ℕ := 75

-- Define the number of toothpicks in a box
def toothpicks_per_box : ℕ := 450

-- Calculate the number of cards used
def cards_used : ℕ := total_cards - unused_cards

-- Calculate the number of cards a single box can support
def cards_per_box : ℕ := toothpicks_per_box / toothpicks_per_card

-- Theorem statement
theorem eugene_boxes_needed : cards_used / cards_per_box = 6 := by
  -- The proof steps are not provided as per the instructions. 
  sorry

end eugene_boxes_needed_l286_286936


namespace probability_of_stopping_after_5_draws_l286_286737

def color := {red, yellow, blue}
def draws (n : ℕ) := vector color n

def valid_sequences (first4 : draws 4) (fifth : color) :=
  let first4_colors := first4.to_list in
  ∃ c1 c2 : color, c1 ≠ c2 ∧
    (c1 ∈ first4_colors) ∧ (c2 ∈ first4_colors) ∧ 
    c1 ≠ fifth ∧ c2 ≠ fifth ∧ 
    fifth ≠ c1 ∧ fifth ≠ c2

theorem probability_of_stopping_after_5_draws : 
  (∃ (first4 : draws 4) (fifth : color), valid_sequences first4 fifth) → 
  (3 ^ 4 - 4) / (3 ^ 5) = 4 / 27 := 
by sorry

end probability_of_stopping_after_5_draws_l286_286737


namespace molecular_weight_proof_l286_286252

def atomic_weight_Al : Float := 26.98
def atomic_weight_O : Float := 16.00
def atomic_weight_H : Float := 1.01

def molecular_weight_AlOH3 : Float :=
  (1 * atomic_weight_Al) + (3 * atomic_weight_O) + (3 * atomic_weight_H)

def moles : Float := 7.0

def molecular_weight_7_moles_AlOH3 : Float :=
  moles * molecular_weight_AlOH3

theorem molecular_weight_proof : molecular_weight_7_moles_AlOH3 = 546.07 :=
by
  /- Here we calculate the molecular weight of Al(OH)3 and multiply it by 7.
     molecular_weight_AlOH3 = (1 * 26.98) + (3 * 16.00) + (3 * 1.01) = 78.01
     molecular_weight_7_moles_AlOH3 = 7 * 78.01 = 546.07 -/
  sorry

end molecular_weight_proof_l286_286252


namespace triangle_side_length_l286_286541

-- Definitions based on problem conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

variables (AC BC AD AB CD : ℝ)

-- Conditions from the problem
axiom h1 : BC = 2 * AC
axiom h2 : AD = (1 / 3) * AB

-- Theorem statement to be proved
theorem triangle_side_length (h1 : BC = 2 * AC) (h2 : AD = (1 / 3) * AB) : CD = 2 * AD :=
sorry

end triangle_side_length_l286_286541


namespace avg_age_combined_l286_286727

-- Define the conditions
def avg_age_roomA : ℕ := 45
def avg_age_roomB : ℕ := 20
def num_people_roomA : ℕ := 8
def num_people_roomB : ℕ := 3

-- Definition of the problem statement
theorem avg_age_combined :
  (num_people_roomA * avg_age_roomA + num_people_roomB * avg_age_roomB) / (num_people_roomA + num_people_roomB) = 38 :=
by
  sorry

end avg_age_combined_l286_286727


namespace tom_cost_cheaper_than_jane_l286_286740

def store_A_full_price : ℝ := 125
def store_A_discount_single : ℝ := 0.08
def store_A_discount_bulk : ℝ := 0.12
def store_A_tax_rate : ℝ := 0.07
def store_A_shipping_fee : ℝ := 10
def store_A_club_discount : ℝ := 0.05

def store_B_full_price : ℝ := 130
def store_B_discount_single : ℝ := 0.10
def store_B_discount_bulk : ℝ := 0.15
def store_B_tax_rate : ℝ := 0.05
def store_B_free_shipping_threshold : ℝ := 250
def store_B_club_discount : ℝ := 0.03

def tom_smartphones_qty : ℕ := 2
def jane_smartphones_qty : ℕ := 3

theorem tom_cost_cheaper_than_jane :
  let tom_cost := 
    let total := store_A_full_price * tom_smartphones_qty
    let discount := if tom_smartphones_qty ≥ 2 then store_A_discount_bulk else store_A_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_A_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_A_tax_rate) 
    price_after_tax + store_A_shipping_fee

  let jane_cost := 
    let total := store_B_full_price * jane_smartphones_qty
    let discount := if jane_smartphones_qty ≥ 3 then store_B_discount_bulk else store_B_discount_single
    let price_after_discount := total * (1 - discount)
    let price_after_club_discount := price_after_discount * (1 - store_B_club_discount)
    let price_after_tax := price_after_club_discount * (1 + store_B_tax_rate)
    let shipping_fee := if total > store_B_free_shipping_threshold then 0 else 0
    price_after_tax + shipping_fee
  
  jane_cost - tom_cost = 104.01 := 
by 
  sorry

end tom_cost_cheaper_than_jane_l286_286740


namespace andrea_rhinestones_needed_l286_286456

theorem andrea_rhinestones_needed (total_needed bought_ratio found_ratio : ℝ) 
  (h1 : total_needed = 45) 
  (h2 : bought_ratio = 1 / 3) 
  (h3 : found_ratio = 1 / 5) : 
  total_needed - (bought_ratio * total_needed + found_ratio * total_needed) = 21 := 
by 
  sorry

end andrea_rhinestones_needed_l286_286456


namespace sufficient_but_not_necessary_for_hyperbola_l286_286034

theorem sufficient_but_not_necessary_for_hyperbola (k : ℝ) :
  (k > 3) ↔ (∀ k, k > 3 → (k - 3 > 0 ∧ k > 0)) ∧ ¬((∀ k, (k - 3 > 0 ∧ k > 0) → k > 3)) := 
sorry

end sufficient_but_not_necessary_for_hyperbola_l286_286034


namespace hundredth_odd_positive_integer_l286_286412

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l286_286412


namespace correct_product_l286_286974

namespace SarahsMultiplication

theorem correct_product (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hx' : ∃ (a b : ℕ), x = 10 * a + b ∧ b * 10 + a = x' ∧ 221 = x' * y) : (x * y = 527 ∨ x * y = 923) := by
  sorry

end SarahsMultiplication

end correct_product_l286_286974


namespace num_distinct_exponentiation_values_l286_286279

theorem num_distinct_exponentiation_values : 
  let a := 2
  let b1 := 2
  let b2 := 2
  let b3 := 2
  let standard_value := (a ^ (b1 ^ (b2 ^ b3)))
  let val_1 := (a ^ (a ^ a)) ^ a
  let val_2 := a ^ ((a ^ a) ^ a)
  let val_3 := ((a ^ a) ^ a) ^ a
  let val_4 := (a ^ (a ^ a)) ^ a
  let val_5 := (a ^ a) ^ (a ^ a)
  in 
  (∃ values : Finset ℕ, values.card = 2 ∧
  standard_value ∈ values ∧ 
  (Finset.erase values standard_value).card = 1 ∧ 
  val_1 ∈ values ∧ val_2 ∈ values ∧ val_3 ∈ values ∧ 
  val_4 ∈ values ∧ val_5 ∈ values) :=
by
  let a := 2
  let b1 := 2
  let b2 := 2
  let b3 := 2
  let standard_value := (a ^ (b1 ^ (b2 ^ b3)))
  let val_1 := (a ^ (a ^ a)) ^ a
  let val_2 := a ^ ((a ^ a) ^ a)
  let val_3 := ((a ^ a) ^ a) ^ a
  let val_4 := (a ^ (a ^ a)) ^ a
  let val_5 := (a ^ a) ^ (a ^ a)
  have h : ∃ values : Finset ℕ, values.card = 2 ∧
    standard_value ∈ values ∧ 
    (Finset.erase values standard_value).card = 1 ∧ 
    val_1 ∈ values ∧ val_2 ∈ values ∧ val_3 ∈ values ∧ 
    val_4 ∈ values ∧ val_5 ∈ values := sorry
  exact h

end num_distinct_exponentiation_values_l286_286279


namespace exists_C_a_n1_minus_a_n_l286_286832

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| 2 => 8
| (n+1) => a (n - 1) + (4 / n) * a n

theorem exists_C (C : ℕ) (hC : C = 2) : ∃ C > 0, ∀ n > 0, a n ≤ C * n^2 := by
  use 2
  sorry

theorem a_n1_minus_a_n (n : ℕ) (h : n > 0) : a (n + 1) - a n ≤ 4 * n + 3 := by
  sorry

end exists_C_a_n1_minus_a_n_l286_286832


namespace smallest_lcm_value_theorem_l286_286335

-- Define k and l to be positive 4-digit integers where gcd(k, l) = 5
def is_positive_4_digit (n : ℕ) : Prop := 1000 <= n ∧ n < 10000

noncomputable def smallest_lcm_value : ℕ :=
  201000

theorem smallest_lcm_value_theorem (k l : ℕ) (hk : is_positive_4_digit k) (hl : is_positive_4_digit l) (h : Int.gcd k l = 5) :
  ∃ m, m = Int.lcm k l ∧ m = smallest_lcm_value :=
sorry

end smallest_lcm_value_theorem_l286_286335


namespace calculate_profit_l286_286362

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end calculate_profit_l286_286362


namespace frequency_of_heads_l286_286223

-- Definitions based on given conditions
def coin_tosses := 10
def heads_up := 6
def event_A := "heads up"

-- The Proof Statement
theorem frequency_of_heads :
  (heads_up / coin_tosses : ℚ) = 3 / 5 :=
sorry

end frequency_of_heads_l286_286223


namespace round_table_vip_arrangements_l286_286822

-- Define the conditions
def number_of_people : ℕ := 10
def vip_seats : ℕ := 2

noncomputable def number_of_arrangements : ℕ :=
  let total_arrangements := Nat.factorial number_of_people
  let vip_choices := Nat.choose number_of_people vip_seats
  let remaining_arrangements := Nat.factorial (number_of_people - vip_seats)
  vip_choices * remaining_arrangements

-- Theorem stating the result
theorem round_table_vip_arrangements : number_of_arrangements = 1814400 := by
  sorry

end round_table_vip_arrangements_l286_286822


namespace colored_copies_count_l286_286289

theorem colored_copies_count :
  ∃ C W : ℕ, (C + W = 400) ∧ (10 * C + 5 * W = 2250) ∧ (C = 50) :=
by
  sorry

end colored_copies_count_l286_286289


namespace g_f_g_1_equals_82_l286_286063

def f (x : ℤ) : ℤ := 2 * x + 2
def g (x : ℤ) : ℤ := 5 * x + 2
def x : ℤ := 1

theorem g_f_g_1_equals_82 : g (f (g x)) = 82 := by
  sorry

end g_f_g_1_equals_82_l286_286063


namespace smallest_m_for_divisibility_l286_286716

theorem smallest_m_for_divisibility : 
  ∃ (m : ℕ), 2^1990 ∣ 1989^m - 1 ∧ m = 2^1988 := 
sorry

end smallest_m_for_divisibility_l286_286716


namespace difference_of_numbers_l286_286587

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : abs (x - y) = 7 :=
sorry

end difference_of_numbers_l286_286587


namespace initial_bags_l286_286742

variable (b : ℕ)

theorem initial_bags (h : 5 * (b - 2) = 45) : b = 11 := 
by 
  sorry

end initial_bags_l286_286742


namespace animals_consuming_hay_l286_286752

-- Define the rate of consumption for each animal
def rate_goat : ℚ := 1 / 6 -- goat consumes 1 cartload per 6 weeks
def rate_sheep : ℚ := 1 / 8 -- sheep consumes 1 cartload per 8 weeks
def rate_cow : ℚ := 1 / 3 -- cow consumes 1 cartload per 3 weeks

-- Define the number of animals
def num_goats : ℚ := 5
def num_sheep : ℚ := 3
def num_cows : ℚ := 2

-- Define the total rate of consumption
def total_rate : ℚ := (num_goats * rate_goat) + (num_sheep * rate_sheep) + (num_cows * rate_cow)

-- Define the total amount of hay to be consumed
def total_hay : ℚ := 30

-- Define the time required to consume the total hay at the calculated rate
def time_required : ℚ := total_hay / total_rate

-- Theorem stating the time required to consume 30 cartloads of hay is 16 weeks.
theorem animals_consuming_hay : time_required = 16 := by
  sorry

end animals_consuming_hay_l286_286752


namespace weight_of_balls_l286_286264

theorem weight_of_balls (x y : ℕ) (h1 : 5 * x + 3 * y = 42) (h2 : 5 * y + 3 * x = 38) :
  x = 6 ∧ y = 4 :=
by
  sorry

end weight_of_balls_l286_286264


namespace necessary_but_not_sufficient_l286_286661

theorem necessary_but_not_sufficient (x : Real)
  (p : Prop := x < 1) 
  (q : Prop := x^2 + x - 2 < 0) 
  : p -> (q <-> x > -2 ∧ x < 1) ∧ (q -> p) → ¬ (p -> q) ∧ (x > -2 -> p) :=
by
  sorry

end necessary_but_not_sufficient_l286_286661


namespace even_fn_solution_set_l286_286844

theorem even_fn_solution_set (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x)) (h_f_def : ∀ x ≥ 0, f x = x^3 - 8) :
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by sorry

end even_fn_solution_set_l286_286844


namespace snowman_volume_l286_286995

theorem snowman_volume (r1 r2 r3 : ℝ) (V1 V2 V3 : ℝ) (π : ℝ) 
  (h1 : r1 = 4) (h2 : r2 = 6) (h3 : r3 = 8) 
  (hV1 : V1 = (4/3) * π * (r1^3)) 
  (hV2 : V2 = (4/3) * π * (r2^3)) 
  (hV3 : V3 = (4/3) * π * (r3^3)) :
  V1 + V2 + V3 = (3168/3) * π :=
by 
  sorry

end snowman_volume_l286_286995


namespace range_of_a_l286_286028

theorem range_of_a (a : ℝ) :
  (1 ∉ {x : ℝ | x^2 - 2 * x + a > 0}) → a ≤ 1 :=
by
  sorry

end range_of_a_l286_286028


namespace quadratic_unique_solution_l286_286149

theorem quadratic_unique_solution (k : ℝ) (x : ℝ) :
  (16 ^ 2 - 4 * 2 * k * 4 = 0) → (k = 8 ∧ x = -1 / 2) :=
by
  sorry

end quadratic_unique_solution_l286_286149


namespace range_of_a_l286_286852

theorem range_of_a 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x, f x = -x^2 + 2*(a - 1)*x + 2)
  (increasing_on : ∀ x < 4, deriv f x > 0) : a ≥ 5 :=
sorry

end range_of_a_l286_286852


namespace awards_distribution_l286_286080

-- Definition of our problem in Lean 4.
theorem awards_distribution (awards : Finset ℕ) (students : Finset ℕ) (h_awards_card : awards.card = 6) (h_students_card : students.card = 4) (h_each_student_gets_award : ∀ s : ℕ, s ∈ students → ∃ a : ℕ, a ∈ awards) :
  ∃ (distributions : Finset (ℕ → ℕ)), distributions.card = 1260 :=
by
  sorry

end awards_distribution_l286_286080


namespace no_function_f_l286_286367

noncomputable def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem no_function_f (a b c : ℝ) (h : ∀ x, g a b c (g a b c x) = x) :
  ¬ ∃ f : ℝ → ℝ, ∀ x, f (f x) = g a b c x := 
sorry

end no_function_f_l286_286367


namespace binary_equals_octal_l286_286880

-- Define that 1001101 in binary is a specific integer
def binary_value : ℕ := 0b1001101

-- Define that 115 in octal is a specific integer
def octal_value : ℕ := 0o115

-- State the theorem we need to prove
theorem binary_equals_octal : binary_value = octal_value :=
  by sorry

end binary_equals_octal_l286_286880


namespace smallest_sum_of_20_consecutive_integers_is_perfect_square_l286_286863

theorem smallest_sum_of_20_consecutive_integers_is_perfect_square (n : ℕ) :
  (∃ n : ℕ, 10 * (2 * n + 19) ∧ ∃ k : ℕ, 10 * (2 * n + 19) = k^2) → 10 * (2 * 3 + 19) = 250 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_is_perfect_square_l286_286863


namespace total_cost_sean_bought_l286_286384

theorem total_cost_sean_bought (cost_soda cost_soup cost_sandwich : ℕ) 
  (h_soda : cost_soda = 1)
  (h_soup : cost_soup = 3 * cost_soda)
  (h_sandwich : cost_sandwich = 3 * cost_soup) :
  3 * cost_soda + 2 * cost_soup + cost_sandwich = 18 := 
by
  sorry

end total_cost_sean_bought_l286_286384


namespace silas_payment_ratio_l286_286579

theorem silas_payment_ratio (total_bill : ℕ) (tip_rate : ℝ) (friend_payment : ℕ) (S : ℕ) :
  total_bill = 150 →
  tip_rate = 0.10 →
  friend_payment = 18 →
  (S + 5 * friend_payment = total_bill + total_bill * tip_rate) →
  (S : ℝ) / total_bill = 1 / 2 :=
by
  intros h_total_bill h_tip_rate h_friend_payment h_budget_eq
  sorry

end silas_payment_ratio_l286_286579


namespace percentage_increase_each_job_l286_286049

-- Definitions of original and new amounts for each job as given conditions
def original_first_job : ℝ := 65
def new_first_job : ℝ := 70

def original_second_job : ℝ := 240
def new_second_job : ℝ := 315

def original_third_job : ℝ := 800
def new_third_job : ℝ := 880

-- Proof problem statement
theorem percentage_increase_each_job :
  (new_first_job - original_first_job) / original_first_job * 100 = 7.69 ∧
  (new_second_job - original_second_job) / original_second_job * 100 = 31.25 ∧
  (new_third_job - original_third_job) / original_third_job * 100 = 10 := by
  sorry

end percentage_increase_each_job_l286_286049


namespace parallelogram_area_l286_286072

theorem parallelogram_area :
  ∀ (A B C D : Type) [EuclideanGeometry A B C D],
    ∀ (AB AD : ℝ) (angle_BAD : ℝ) (area : ℝ),
      AB = 12 ∧ AD = 10 ∧ angle_BAD = 150 ∧ parallelogram A B C D →
      area = 60 :=
by sorry

end parallelogram_area_l286_286072


namespace min_quadratic_expr_l286_286876

noncomputable def quadratic_expr (x : ℝ) := x^2 + 10 * x + 3

theorem min_quadratic_expr : ∃ x : ℝ, quadratic_expr x = -22 :=
by
  use -5
  simp [quadratic_expr]
  sorry

end min_quadratic_expr_l286_286876


namespace comm_add_comm_mul_distrib_l286_286787

variable {α : Type*} [AddCommMonoid α] [Mul α] [Distrib α]

theorem comm_add (a b : α) : a + b = b + a :=
by sorry

theorem comm_mul (a b : α) : a * b = b * a :=
by sorry

theorem distrib (a b c : α) : (a + b) * c = a * c + b * c :=
by sorry

end comm_add_comm_mul_distrib_l286_286787


namespace find_a_tangent_line_eq_l286_286500

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x - 1) * Real.exp x

theorem find_a (a : ℝ) : f 1 (-3) = 0 → a = 1 := by
  sorry

theorem tangent_line_eq (x : ℝ) (e : ℝ) : x = 1 ∧ f 1 x = Real.exp 1 → 
    (4 * Real.exp 1 * x - y - 3 * Real.exp 1 = 0) := by
  sorry

end find_a_tangent_line_eq_l286_286500


namespace axis_of_symmetry_l286_286728

-- Define the given parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x + 5

-- Define the statement that we need to prove
theorem axis_of_symmetry : (∃ (a : ℝ), ∀ x, parabola (x) = (x - a) ^ 2 + 4) ∧ 
                           (∃ (b : ℝ), b = 1) :=
by
  sorry

end axis_of_symmetry_l286_286728


namespace integer_sum_of_squares_power_l286_286074

theorem integer_sum_of_squares_power (a p q : ℤ) (k : ℕ) (h : a = p^2 + q^2) : 
  ∃ c d : ℤ, a^k = c^2 + d^2 := 
sorry

end integer_sum_of_squares_power_l286_286074


namespace M_subset_N_l286_286807

def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def N : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem M_subset_N : M ⊆ N :=
by
  sorry

end M_subset_N_l286_286807


namespace jerry_stickers_l286_286981

variable (G F J : ℕ)

theorem jerry_stickers (h1 : F = 18) (h2 : G = F - 6) (h3 : J = 3 * G) : J = 36 :=
by {
  sorry
}

end jerry_stickers_l286_286981


namespace convert_to_base5_l286_286928

theorem convert_to_base5 : ∀ n : ℕ, n = 1729 → Nat.digits 5 n = [2, 3, 4, 0, 4] :=
by
  intros n hn
  rw [hn]
  -- proof steps can be filled in here
  sorry

end convert_to_base5_l286_286928


namespace number_of_girls_l286_286132

theorem number_of_girls
  (B : ℕ) (k : ℕ) (G : ℕ)
  (hB : B = 10) 
  (hk : k = 5)
  (h1 : B / k = 2)
  (h2 : G % k = 0) :
  G = 5 := 
sorry

end number_of_girls_l286_286132


namespace highest_score_of_D_l286_286042

theorem highest_score_of_D
  (a b c d : ℕ)
  (h1 : a + b = c + d)
  (h2 : b + d > a + c)
  (h3 : a > b + c) :
  d > a :=
by
  sorry

end highest_score_of_D_l286_286042


namespace distinct_values_count_l286_286484

open Finset

def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

def distinct_powers (a b : ℕ) : ℝ := 3^((b : ℝ)/(a : ℝ))

theorem distinct_values_count : 
  (S.product S).filter (λ p, p.1 ≠ p.2).image (λ p, distinct_powers p.1 p.2) = 22 := 
begin
  sorry
end

end distinct_values_count_l286_286484


namespace twelfth_term_geometric_sequence_l286_286424

-- Define the first term and common ratio
def a1 : Int := 5
def r : Int := -3

-- Define the formula for the nth term of the geometric sequence
def nth_term (n : Nat) : Int := a1 * r^(n-1)

-- The statement to be proved: that the twelfth term is -885735
theorem twelfth_term_geometric_sequence : nth_term 12 = -885735 := by
  sorry

end twelfth_term_geometric_sequence_l286_286424


namespace part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l286_286377

-- Definitions for part (1)
def P_X_1 : ℚ := 1 / 6
def P_X_2 : ℚ := 5 / 36
def P_X_3 : ℚ := 25 / 216
def P_X_4 : ℚ := 125 / 216
def E_X : ℚ := 671 / 216

theorem part1_prob_dist (X : ℚ) :
  (X = 1 → P_X_1 = 1 / 6) ∧
  (X = 2 → P_X_2 = 5 / 36) ∧
  (X = 3 → P_X_3 = 25 / 216) ∧
  (X = 4 → P_X_4 = 125 / 216) := 
by sorry

theorem part1_expectation :
  E_X = 671 / 216 :=
by sorry

-- Definition for part (2)
def P_A_wins_n_throws (n : ℕ) : ℚ := 1 / 6 * (5 / 6) ^ (2 * n - 2)

theorem part2_prob_A_wins_n_throws (n : ℕ) (hn : n ≥ 1) :
  P_A_wins_n_throws n = 1 / 6 * (5 / 6) ^ (2 * n - 2) :=
by sorry

end part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l286_286377


namespace oprod_eval_l286_286930

def oprod (a b : ℕ) : ℕ :=
  (a * 2 + b) / 2

theorem oprod_eval : oprod (oprod 4 6) 8 = 11 :=
by
  -- Definitions given in conditions
  let r := (4 * 2 + 6) / 2
  have h1 : oprod 4 6 = r := by rfl
  let s := (r * 2 + 8) / 2
  have h2 : oprod r 8 = s := by rfl
  exact (show s = 11 from sorry)

end oprod_eval_l286_286930


namespace p_as_percentage_of_x_l286_286395

-- Given conditions
variables (x y z w t u p : ℝ)
variables (h1 : 0.37 * z = 0.84 * y)
variables (h2 : y = 0.62 * x)
variables (h3 : 0.47 * w = 0.73 * z)
variables (h4 : w = t - u)
variables (h5 : u = 0.25 * t)
variables (h6 : p = z + t + u)

-- Prove that p is 505.675% of x
theorem p_as_percentage_of_x : p = 5.05675 * x := by
  sorry

end p_as_percentage_of_x_l286_286395


namespace projection_orthogonal_l286_286549

variables (a b : ℝ × ℝ)
variables (v : ℝ × ℝ)
variables (h1 : dot_product a b = 0) -- a and b are orthogonal
variables (h2 : proj a (4, -2) = (1, 2)) -- projection of (4, -2) onto a

-- Theorem statement
theorem projection_orthogonal {a b : ℝ × ℝ} {v : ℝ × ℝ}
  (h1 : dot_product a b = 0)
  (h2 : proj a v = (1, 2)) :
  proj b v = (3, -4) :=
sorry

end projection_orthogonal_l286_286549


namespace unique_handshakes_462_l286_286135

theorem unique_handshakes_462 : 
  ∀ (twins triplets : Type) (twin_set : ℕ) (triplet_set : ℕ) (handshakes_among_twins handshakes_among_triplets cross_handshakes_twins cross_handshakes_triplets : ℕ),
  twin_set = 12 ∧
  triplet_set = 4 ∧
  handshakes_among_twins = (24 * 22) / 2 ∧
  handshakes_among_triplets = (12 * 9) / 2 ∧
  cross_handshakes_twins = 24 * (12 / 3) ∧
  cross_handshakes_triplets = 12 * (24 / 3 * 2) →
  (handshakes_among_twins + handshakes_among_triplets + (cross_handshakes_twins + cross_handshakes_triplets) / 2) = 462 := 
by
  sorry

end unique_handshakes_462_l286_286135


namespace road_building_equation_l286_286592

theorem road_building_equation (x : ℝ) (hx : x > 0) :
  (9 / x - 12 / (x + 1) = 1 / 2) :=
sorry

end road_building_equation_l286_286592


namespace penny_money_left_is_5_l286_286710

def penny_initial_money : ℤ := 20
def socks_pairs : ℤ := 4
def price_per_pair_of_socks : ℤ := 2
def price_of_hat : ℤ := 7

def total_cost_of_socks : ℤ := socks_pairs * price_per_pair_of_socks
def total_cost_of_hat_and_socks : ℤ := total_cost_of_socks + price_of_hat
def penny_money_left : ℤ := penny_initial_money - total_cost_of_hat_and_socks

theorem penny_money_left_is_5 : penny_money_left = 5 := by
  sorry

end penny_money_left_is_5_l286_286710


namespace average_of_second_pair_l286_286085

theorem average_of_second_pair (S : ℝ) (S1 : ℝ) (S3 : ℝ) (S2 : ℝ) (avg : ℝ) :
  (S / 6 = 3.95) →
  (S1 / 2 = 3.8) →
  (S3 / 2 = 4.200000000000001) →
  (S = S1 + S2 + S3) →
  (avg = S2 / 2) →
  avg = 3.85 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end average_of_second_pair_l286_286085


namespace range_of_m_l286_286498

theorem range_of_m (x m : ℝ) (h1 : (m - 1) / (x + 1) = 1) (h2 : x < 0) : m < 2 ∧ m ≠ 1 :=
by
  sorry

end range_of_m_l286_286498


namespace plywood_problem_exists_squares_l286_286896

theorem plywood_problem_exists_squares :
  ∃ (a b : ℕ), a^2 + b^2 = 625 ∧ a ≠ 20 ∧ b ≠ 20 ∧ a ≠ 15 ∧ b ≠ 15 := by
  sorry

end plywood_problem_exists_squares_l286_286896


namespace find_dividend_l286_286109

def quotient : ℝ := -427.86
def divisor : ℝ := 52.7
def remainder : ℝ := -14.5
def dividend : ℝ := (quotient * divisor) + remainder

theorem find_dividend : dividend = -22571.122 := by
  sorry

end find_dividend_l286_286109


namespace molecular_weight_NaClO_is_74_44_l286_286476

-- Define the atomic weights
def atomic_weight_Na : Real := 22.99
def atomic_weight_Cl : Real := 35.45
def atomic_weight_O : Real := 16.00

-- Define the calculation of molecular weight
def molecular_weight_NaClO : Real :=
  atomic_weight_Na + atomic_weight_Cl + atomic_weight_O

-- Define the theorem statement
theorem molecular_weight_NaClO_is_74_44 :
  molecular_weight_NaClO = 74.44 :=
by
  -- Placeholder for proof
  sorry

end molecular_weight_NaClO_is_74_44_l286_286476


namespace largest_multiple_of_7_less_than_neg50_l286_286099

theorem largest_multiple_of_7_less_than_neg50 : ∃ x, (∃ k : ℤ, x = 7 * k) ∧ x < -50 ∧ ∀ y, (∃ m : ℤ, y = 7 * m) → y < -50 → y ≤ x :=
sorry

end largest_multiple_of_7_less_than_neg50_l286_286099


namespace li_payment_l286_286883

noncomputable def payment_li (daily_payment_per_unit : ℚ) (days_li_worked : ℕ) : ℚ :=
daily_payment_per_unit * days_li_worked

theorem li_payment (work_per_day : ℚ) (days_li_worked : ℕ) (days_extra_work : ℕ) 
  (difference_payment : ℚ) (daily_payment_per_unit : ℚ) (initial_nanual_workdays : ℕ) :
  work_per_day = 1 →
  days_li_worked = 2 →
  days_extra_work = 3 →
  difference_payment = 2700 →
  daily_payment_per_unit = difference_payment / (initial_nanual_workdays + (3 * 3)) → 
  payment_li daily_payment_per_unit days_li_worked = 450 := 
by 
  intros h_work_per_day h_days_li_worked h_days_extra_work h_diff_payment h_daily_payment 
  sorry

end li_payment_l286_286883


namespace three_students_two_groups_l286_286404

theorem three_students_two_groups : 
  (2 : ℕ) ^ 3 = 8 := 
by
  sorry

end three_students_two_groups_l286_286404


namespace hundredth_odd_integer_l286_286414

theorem hundredth_odd_integer : ∃ (x : ℕ), 2 * x - 1 = 199 ∧ x = 100 :=
by
  use 100
  split
  . exact calc
      2 * 100 - 1 = 200 - 1 : by ring
      _ = 199 : by norm_num
  . refl

end hundredth_odd_integer_l286_286414


namespace sector_area_150_degrees_l286_286396

def sector_area (radius : ℝ) (central_angle : ℝ) : ℝ :=
  0.5 * radius^2 * central_angle

theorem sector_area_150_degrees (r : ℝ) (angle_rad : ℝ) (h1 : r = Real.sqrt 3) (h2 : angle_rad = (5 * Real.pi) / 6) : 
  sector_area r angle_rad = (5 * Real.pi) / 4 :=
by
  simp [sector_area, h1, h2]
  sorry

end sector_area_150_degrees_l286_286396


namespace problem_1_exists_a_problem_2_values_of_a_l286_286504

open Set

-- Definitions for sets A, B, C
def A (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + 4 * a^2 - 3 = 0}
def B : Set ℝ := {x | x^2 - x - 2 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- Lean statements for the two problems
theorem problem_1_exists_a : ∃ a : ℝ, A a ∩ B = A a ∪ B ∧ a = 1/2 := by
  sorry

theorem problem_2_values_of_a (a : ℝ) : 
  (A a ∩ B ≠ ∅ ∧ A a ∩ C = ∅) → 
  (A a = {-1} → a = -1) ∧ (∀ x, A a = {-1, x} → x ≠ 2 → False) := 
  by sorry

end problem_1_exists_a_problem_2_values_of_a_l286_286504


namespace marbles_count_l286_286944

theorem marbles_count (red green blue total : ℕ) (h_red : red = 38)
  (h_green : green = red / 2) (h_total : total = 63) 
  (h_sum : total = red + green + blue) : blue = 6 :=
by
  sorry

end marbles_count_l286_286944


namespace combined_height_l286_286983

theorem combined_height (h_John : ℕ) (h_Lena : ℕ) (h_Rebeca : ℕ)
  (cond1 : h_John = 152)
  (cond2 : h_John = h_Lena + 15)
  (cond3 : h_Rebeca = h_John + 6) :
  h_Lena + h_Rebeca = 295 :=
by
  sorry

end combined_height_l286_286983


namespace max_value_l286_286651

theorem max_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 9 * a^2 + 4 * b^2 + c^2 = 91) :
  a + 2 * b + 3 * c ≤ 30.333 :=
by
  sorry

end max_value_l286_286651


namespace calcium_carbonate_required_l286_286158

theorem calcium_carbonate_required (HCl_moles CaCO3_moles CaCl2_moles CO2_moles H2O_moles : ℕ) 
  (reaction_balanced : CaCO3_moles + 2 * HCl_moles = CaCl2_moles + CO2_moles + H2O_moles) 
  (HCl_moles_value : HCl_moles = 2) : CaCO3_moles = 1 :=
by sorry

end calcium_carbonate_required_l286_286158


namespace distance_between_foci_l286_286011

theorem distance_between_foci (x y : ℝ)
    (h : 2 * x^2 - 12 * x - 8 * y^2 + 16 * y = 100) :
    2 * Real.sqrt 68.75 =
    2 * Real.sqrt (55 + 13.75) :=
by
  sorry

end distance_between_foci_l286_286011


namespace xy_difference_l286_286168

noncomputable def x : ℝ := Real.sqrt 3 + 1
noncomputable def y : ℝ := Real.sqrt 3 - 1

theorem xy_difference : x^2 * y - x * y^2 = 4 := by
  sorry

end xy_difference_l286_286168


namespace length_of_BC_l286_286830

theorem length_of_BC (BD CD : ℝ) (h1 : BD = 3 + 3 * BD) (h2 : CD = 2 + 2 * CD) (h3 : 4 * BD + 3 * CD + 5 = 20) : 2 * CD + 2 = 4 :=
by {
  sorry
}

end length_of_BC_l286_286830


namespace find_remainder_l286_286704

theorem find_remainder : ∃ r : ℝ, r = 14 ∧ 13698 = (153.75280898876406 * 89) + r := 
by
  sorry

end find_remainder_l286_286704


namespace find_a_n_geo_b_find_S_2n_l286_286176
noncomputable def S : ℕ → ℚ
| n => (n^2 + n + 1) / 2

def a (n : ℕ) : ℚ :=
  if n = 1 then 3/2
  else n

theorem find_a_n (n : ℕ) : a n = if n = 1 then 3/2 else n :=
by
  sorry

def b (n : ℕ) : ℚ :=
  a (2 * n - 1) + a (2 * n)

theorem geo_b (n : ℕ) : b (n + 1) = 3 * b n :=
by
  sorry

theorem find_S_2n (n : ℕ) : S (2 * n) = 3/2 * (3^n - 1) :=
by
  sorry

end find_a_n_geo_b_find_S_2n_l286_286176


namespace square_overlap_area_l286_286909

theorem square_overlap_area (β : ℝ) (h1 : 0 < β) (h2 : β < 90) (h3 : Real.cos β = 3 / 5) : 
  area (common_region (square 2) (rotate_square β (square 2))) = 4 / 3 :=
sorry

end square_overlap_area_l286_286909


namespace tigers_count_l286_286286

theorem tigers_count (T C : ℝ) 
  (h1 : 12 + T + C = 39) 
  (h2 : C = 0.5 * (12 + T)) : 
  T = 14 := by
  sorry

end tigers_count_l286_286286


namespace second_half_takes_200_percent_longer_l286_286556

noncomputable def time_take (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

variable (total_distance : ℕ := 640)
variable (first_half_speed : ℕ := 80)
variable (average_speed : ℕ := 40)

theorem second_half_takes_200_percent_longer :
  let first_half_distance := total_distance / 2;
  let first_half_time := time_take first_half_distance first_half_speed;
  let total_time := time_take total_distance average_speed;
  let second_half_time := total_time - first_half_time;
  let time_increase := second_half_time - first_half_time;
  let percentage_increase := (time_increase * 100) / first_half_time;
  percentage_increase = 200 :=
by
  sorry

end second_half_takes_200_percent_longer_l286_286556


namespace value_of_x_squared_plus_reciprocal_squared_l286_286512

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l286_286512


namespace hiker_total_distance_l286_286431

def hiker_distance (day1_hours day1_speed day2_speed : ℕ) : ℕ :=
  let day2_hours := day1_hours - 1
  let day3_hours := day1_hours
  (day1_hours * day1_speed) + (day2_hours * day2_speed) + (day3_hours * day2_speed)

theorem hiker_total_distance :
  hiker_distance 6 3 4 = 62 := 
by 
  sorry

end hiker_total_distance_l286_286431


namespace length_AB_slope_one_OA_dot_OB_const_l286_286324

open Real

def parabola (x y : ℝ) : Prop := y * y = 4 * x
def line_through_focus (x y : ℝ) (k : ℝ) : Prop := x = k * y + 1
def line_slope_one (x y : ℝ) : Prop := y = x - 1

theorem length_AB_slope_one {x1 x2 y1 y2 : ℝ} (hA : parabola x1 y1) (hB : parabola x2 y2) 
  (hL : line_slope_one x1 y1) (hL' : line_slope_one x2 y2) : abs (x1 - x2) + abs (y1 - y2) = 8 := 
by
  sorry

theorem OA_dot_OB_const {x1 x2 y1 y2 : ℝ} {k : ℝ} (hA : parabola x1 y1)
  (hB : parabola x2 y2) (hL : line_through_focus x1 y1 k) (hL' : line_through_focus x2 y2 k) :
  x1 * x2 + y1 * y2 = -3 :=
by
  sorry

end length_AB_slope_one_OA_dot_OB_const_l286_286324


namespace original_fraction_is_two_thirds_l286_286582

theorem original_fraction_is_two_thirds (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ)/(b + 3) = 2 * (a : ℚ)/b → (a : ℚ)/b = 2/3 :=
by
  sorry

end original_fraction_is_two_thirds_l286_286582


namespace average_monthly_balance_is_150_l286_286913

-- Define the balances for each month
def balance_jan : ℕ := 100
def balance_feb : ℕ := 200
def balance_mar : ℕ := 150
def balance_apr : ℕ := 150

-- Define the number of months
def num_months : ℕ := 4

-- Define the total sum of balances
def total_balance : ℕ := balance_jan + balance_feb + balance_mar + balance_apr

-- Define the average balance
def average_balance : ℕ := total_balance / num_months

-- Goal is to prove that the average monthly balance is 150 dollars
theorem average_monthly_balance_is_150 : average_balance = 150 :=
by
  sorry

end average_monthly_balance_is_150_l286_286913


namespace smallest_lcm_value_theorem_l286_286334

-- Define k and l to be positive 4-digit integers where gcd(k, l) = 5
def is_positive_4_digit (n : ℕ) : Prop := 1000 <= n ∧ n < 10000

noncomputable def smallest_lcm_value : ℕ :=
  201000

theorem smallest_lcm_value_theorem (k l : ℕ) (hk : is_positive_4_digit k) (hl : is_positive_4_digit l) (h : Int.gcd k l = 5) :
  ∃ m, m = Int.lcm k l ∧ m = smallest_lcm_value :=
sorry

end smallest_lcm_value_theorem_l286_286334


namespace area_of_shaded_region_l286_286910

noncomputable def shaded_region_area (β : ℝ) (cos_beta : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) : ℝ :=
  let sine_beta := Real.sqrt (1 - (3 / 5)^2)
  let tan_half_beta := sine_beta / (1 + 3 / 5)
  let bp := Real.tan (π / 4 - tan_half_beta)
  2 * (1 / 5) + 2 * (1 / 5)

theorem area_of_shaded_region (β : ℝ) (h : β ≠ 0 ∧ β < π / 2 ∧ Real.cos β = 3 / 5) :
  shaded_region_area β h = 4 / 5 := by
  sorry

end area_of_shaded_region_l286_286910


namespace distance_between_foci_l286_286306

theorem distance_between_foci :
  let x := ℝ
  let y := ℝ
  ∀ (x y : ℝ), 9*x^2 + 36*x + 4*y^2 - 8*y + 1 = 0 →
  ∃ (d : ℝ), d = (Real.sqrt 351) / 3 :=
sorry

end distance_between_foci_l286_286306


namespace exit_condition_l286_286348

-- Define the loop structure in a way that is consistent with how the problem is described
noncomputable def program_loop (k : ℕ) : ℕ :=
  if k < 7 then 35 else sorry -- simulate the steps of the program

-- The proof goal is to show that the condition which stops the loop when s = 35 is k ≥ 7
theorem exit_condition (k : ℕ) (s : ℕ) : 
  (program_loop k = 35) → (k ≥ 7) :=
by {
  sorry
}

end exit_condition_l286_286348


namespace sum_series_eq_two_l286_286921

theorem sum_series_eq_two : (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 2 := 
by
  sorry

end sum_series_eq_two_l286_286921


namespace relationship_among_x_y_z_w_l286_286520

theorem relationship_among_x_y_z_w (x y z w : ℝ) (h : (x + y) / (y + z) = (z + w) / (w + x)) :
  x = z ∨ x + y + w + z = 0 :=
sorry

end relationship_among_x_y_z_w_l286_286520


namespace value_of_a_b_c_l286_286087

theorem value_of_a_b_c 
  (a b c : ℤ) 
  (h1 : x^2 + 12*x + 35 = (x + a)*(x + b)) 
  (h2 : x^2 - 15*x + 56 = (x - b)*(x - c)) : 
  a + b + c = 20 := 
sorry

end value_of_a_b_c_l286_286087


namespace gcd_1343_816_l286_286854

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end gcd_1343_816_l286_286854


namespace solve_eqn_l286_286789

noncomputable def a : ℝ := 5 + 2 * Real.sqrt 6
noncomputable def b : ℝ := 5 - 2 * Real.sqrt 6

theorem solve_eqn (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 10) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_eqn_l286_286789


namespace smallest_sum_l286_286551

noncomputable def problem_statement : Prop :=
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  (∀ A B C D : ℕ, 
    5 * A = 25 * A - 27 * B ∧
    5 * B = 15 * A - 16 * B ∧
    3 * C = 25 * C - 27 * D ∧
    3 * D = 15 * C - 16 * D) ∧
  a = 4 ∧ b = 3 ∧ c = 27 ∧ d = 22 ∧ a + b + c + d = 56

theorem smallest_sum : problem_statement :=
  sorry

end smallest_sum_l286_286551


namespace exponent_property_l286_286675

theorem exponent_property (a : ℝ) (m n : ℝ) (h₁ : a^m = 4) (h₂ : a^n = 8) : a^(m + n) = 32 := 
by 
  sorry

end exponent_property_l286_286675


namespace number_of_other_values_l286_286280

def orig_value : ℕ := 2 ^ (2 ^ (2 ^ 2))

def other_values : Finset ℕ :=
  {2 ^ (2 ^ (2 ^ 2)), 2 ^ ((2 ^ 2) ^ 2), ((2 ^ 2) ^ 2) ^ 2, (2 ^ (2 ^ 2)) ^ 2, (2 ^ 2) ^ (2 ^ 2)}

theorem number_of_other_values :
  other_values.erase orig_value = {256} :=
by
  sorry

end number_of_other_values_l286_286280


namespace real_y_iff_x_ranges_l286_286339

-- Definitions for conditions
variable (x y : ℝ)

-- Condition for the equation
def equation := 9 * y^2 - 6 * x * y + 2 * x + 7 = 0

-- Theorem statement
theorem real_y_iff_x_ranges :
  (∃ y : ℝ, equation x y) ↔ (x ≤ -2 ∨ x ≥ 7) :=
sorry

end real_y_iff_x_ranges_l286_286339


namespace magnitude_of_complex_l286_286022

theorem magnitude_of_complex 
  (z : ℂ)
  (h : (1 + 2*complex.I) * z = -1 + 3*complex.I) :
  complex.abs(z) = real.sqrt 2 :=
by
  sorry

end magnitude_of_complex_l286_286022


namespace arithmetic_sequence_difference_l286_286422

def arithmetic_sequence (a d n : ℕ) : ℤ :=
  a + (n - 1) * d

theorem arithmetic_sequence_difference :
  let a := 3
  let d := 7
  let a₁₀₀₀ := arithmetic_sequence a d 1000
  let a₁₀₀₃ := arithmetic_sequence a d 1003
  abs (a₁₀₀₃ - a₁₀₀₀) = 21 :=
by
  sorry

end arithmetic_sequence_difference_l286_286422


namespace angle_in_third_quadrant_l286_286181

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l286_286181


namespace james_profit_l286_286359

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end james_profit_l286_286359


namespace theta_value_l286_286897

theorem theta_value (theta : ℝ) (h1 : 0 ≤ theta ∧ theta ≤ 90)
    (h2 : Real.cos 60 = Real.cos 45 * Real.cos theta) : theta = 45 :=
  sorry

end theta_value_l286_286897


namespace student_chose_number_l286_286626

theorem student_chose_number : ∃ x : ℤ, 2 * x - 152 = 102 ∧ x = 127 :=
by
  sorry

end student_chose_number_l286_286626


namespace find_b_proof_l286_286215

noncomputable def find_b (a b c : ℝ) (A B C : ℝ) : Prop :=
  let area := sqrt 3 in
  let angle_B := 60 * Real.pi / 180 in
  let condition1 := area = (1 / 2) * a * c * Real.sin angle_B in
  let condition2 := a^2 + c^2 = 3 * a * c in
  let side_b := b = 2 * Real.sqrt 2 in
  condition1 ∧ condition2 → side_b

theorem find_b_proof :
  ∃ (a b c : ℝ) (A B C : ℝ), find_b a b c A B C :=
by
  sorry

end find_b_proof_l286_286215


namespace length_increase_percentage_l286_286088

theorem length_increase_percentage 
  (L B : ℝ)
  (x : ℝ)
  (h1 : B' = B * 0.8)
  (h2 : L' = L * (1 + x / 100))
  (h3 : A = L * B)
  (h4 : A' = L' * B')
  (h5 : A' = A * 1.04) 
  : x = 30 :=
sorry

end length_increase_percentage_l286_286088


namespace steve_speed_l286_286366

theorem steve_speed (v : ℝ) : 
  (John_initial_distance_behind_Steve = 15) ∧ 
  (John_final_distance_ahead_of_Steve = 2) ∧ 
  (John_speed = 4.2) ∧ 
  (final_push_duration = 34) → 
  v * final_push_duration = (John_speed * final_push_duration) - (John_initial_distance_behind_Steve + John_final_distance_ahead_of_Steve) →
  v = 3.7 := 
by
  intros hconds heq
  exact sorry

end steve_speed_l286_286366


namespace range_of_composite_function_l286_286239

noncomputable def range_of_function : Set ℝ :=
  {y | ∃ x : ℝ, y = (1/2) ^ (|x + 1|)}

theorem range_of_composite_function : range_of_function = Set.Ioc 0 1 :=
by
  sorry

end range_of_composite_function_l286_286239


namespace multiplier_eq_l286_286236

-- Definitions of the given conditions
def length (w : ℝ) (m : ℝ) : ℝ := m * w + 2
def perimeter (l : ℝ) (w : ℝ) : ℝ := 2 * l + 2 * w

-- Condition definitions
def l : ℝ := 38
def P : ℝ := 100

-- Proof statement
theorem multiplier_eq (m w : ℝ) (h1 : length w m = l) (h2 : perimeter l w = P) : m = 3 :=
by
  sorry

end multiplier_eq_l286_286236


namespace rational_square_plus_one_pos_l286_286116

theorem rational_square_plus_one_pos (a : ℚ) : 0 < a^2 + 1 := 
begin
  calc 
  0 < a^2 : by nlinarith
  ... < a^2 + 1 : by linarith,
end

end rational_square_plus_one_pos_l286_286116


namespace lock_combination_l286_286226

theorem lock_combination 
  (B A N D S : ℕ) (base d : ℕ)
  (h_base : d = 6)
  (h_B : B = 1) 
  (h_A : A = 2) 
  (h_N : N = 3) 
  (h_D : D = 4) 
  (h_S : S = 5) 
  : (S * base^2 + A * base + N) = 523 := by 
  sorry

end lock_combination_l286_286226


namespace find_a7_l286_286495

-- Define the arithmetic sequence
def a (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions
def S5 : ℤ := 25
def a2 : ℤ := 3

-- Main Goal: Find a_7
theorem find_a7 (a1 d : ℤ) (h1 : sum_first_n_terms a1 d 5 = S5)
                     (h2 : a a1 d 2 = a2) :
  a a1 d 7 = 13 := 
sorry

end find_a7_l286_286495


namespace triangle_side_b_l286_286213

noncomputable def find_side_length_b
  (a b c : ℝ)
  (area : ℝ)
  (B_deg : ℝ)
  (relation : a^2 + c^2 = 3 * a * c)
  (area_condition : area = sqrt 3)
  (angle_condition : B_deg = 60) : Prop :=
  b = 2 * sqrt 2

theorem triangle_side_b
  (a b c : ℝ)
  (area : ℝ := sqrt 3)
  (B_deg : ℝ := 60)
  (h1 : a^2 + c^2 = 3 * a * c)
  (h2 : area = sqrt 3)
  (h3 : B_deg = 60) : find_side_length_b a b c area B_deg h1 h2 h3 :=
  sorry

end triangle_side_b_l286_286213


namespace comb_5_1_eq_5_l286_286000

theorem comb_5_1_eq_5 : Nat.choose 5 1 = 5 :=
by
  sorry

end comb_5_1_eq_5_l286_286000


namespace system_of_equations_solution_exists_l286_286725

theorem system_of_equations_solution_exists :
  ∃ (x y : ℝ), 
    (4 * x^2 + 8 * x * y + 16 * y^2 + 2 * x + 20 * y = -7) ∧
    (2 * x^2 - 16 * x * y + 8 * y^2 - 14 * x + 20 * y = -11) ∧
    (x = 1/2) ∧ (y = -3/4) :=
by
  sorry

end system_of_equations_solution_exists_l286_286725


namespace number_of_oranges_l286_286351

def bananas : ℕ := 7
def apples : ℕ := 2 * bananas
def pears : ℕ := 4
def grapes : ℕ := apples / 2
def total_fruits : ℕ := 40

theorem number_of_oranges : total_fruits - (bananas + apples + pears + grapes) = 8 :=
by sorry

end number_of_oranges_l286_286351


namespace solve_rational_equation_l286_286723

theorem solve_rational_equation (x : ℝ) (h : x ≠ (2/3)) : 
  (6*x + 4) / (3*x^2 + 6*x - 8) = 3*x / (3*x - 2) ↔ x = -4/3 ∨ x = 3 :=
sorry

end solve_rational_equation_l286_286723


namespace randy_money_left_l286_286076

theorem randy_money_left (initial_money lunch ice_cream_cone remaining : ℝ) 
  (h1 : initial_money = 30)
  (h2 : lunch = 10)
  (h3 : remaining = initial_money - lunch)
  (h4 : ice_cream_cone = remaining * (1/4)) :
  (remaining - ice_cream_cone) = 15 := by
  sorry

end randy_money_left_l286_286076


namespace winnie_retains_lollipops_l286_286430

theorem winnie_retains_lollipops :
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  lollipops_total % friends = 10 :=
by
  let lollipops_total := 60 + 105 + 5 + 230
  let friends := 13
  show lollipops_total % friends = 10
  sorry

end winnie_retains_lollipops_l286_286430


namespace donny_money_left_l286_286006

-- Definitions based on Conditions
def initial_amount : ℝ := 78
def cost_kite : ℝ := 8
def cost_frisbee : ℝ := 9

-- Discounted cost of roller skates
def original_cost_roller_skates : ℝ := 15
def discount_rate_roller_skates : ℝ := 0.10
def discounted_cost_roller_skates : ℝ :=
  original_cost_roller_skates * (1 - discount_rate_roller_skates)

-- Cost of LEGO set with coupon
def original_cost_lego_set : ℝ := 25
def coupon_lego_set : ℝ := 5
def discounted_cost_lego_set : ℝ :=
  original_cost_lego_set - coupon_lego_set

-- Cost of puzzle with tax
def original_cost_puzzle : ℝ := 12
def tax_rate_puzzle : ℝ := 0.05
def taxed_cost_puzzle : ℝ :=
  original_cost_puzzle * (1 + tax_rate_puzzle)

-- Total cost calculated from item costs
def total_cost : ℝ :=
  cost_kite + cost_frisbee + discounted_cost_roller_skates + discounted_cost_lego_set + taxed_cost_puzzle

def money_left_after_shopping : ℝ :=
  initial_amount - total_cost

-- Prove the main statement
theorem donny_money_left : money_left_after_shopping = 14.90 := by
  sorry

end donny_money_left_l286_286006


namespace geometric_sequence_fourth_term_l286_286234

theorem geometric_sequence_fourth_term (x : ℝ) (r : ℝ) (h1 : (3 * x + 3)^2 = x * (6 * x + 6)) 
(h2 : r = (3 * x + 3) / x) :
  (6 * x + 6) * r = -24 :=
by {
  -- Definitions of x, r and condition h1, h2 are given.
  -- Conclusion must follow that the fourth term is -24.
  sorry
}

end geometric_sequence_fourth_term_l286_286234


namespace evaluate_expression_at_zero_l286_286877

theorem evaluate_expression_at_zero :
  (0^2 + 5 * 0 - 10) = -10 :=
by
  sorry

end evaluate_expression_at_zero_l286_286877


namespace abs_eq_necessary_but_not_sufficient_l286_286898

theorem abs_eq_necessary_but_not_sufficient (x y : ℝ) :
  (|x| = |y|) → (¬(x = y) → x = -y) :=
by
  sorry

end abs_eq_necessary_but_not_sufficient_l286_286898


namespace dimes_max_diff_l286_286563

-- Definitions and conditions
def num_coins (a b c : ℕ) : Prop := a + b + c = 120
def coin_values (a b c : ℕ) : Prop := 5 * a + 10 * b + 50 * c = 1050
def dimes_difference (a1 a2 b1 b2 c1 c2 : ℕ) : Prop := num_coins a1 b1 c1 ∧ num_coins a2 b2 c2 ∧ coin_values a1 b1 c1 ∧ coin_values a2 b2 c2 ∧ a1 = a2 ∧ c1 = c2

-- Theorem statement
theorem dimes_max_diff : ∃ (a b1 b2 c : ℕ), dimes_difference a a b1 b2 c c ∧ b1 - b2 = 90 :=
by sorry

end dimes_max_diff_l286_286563


namespace triangle_area_l286_286845

theorem triangle_area (A P Q : Point)
  (hA : A = (8, 6))
  (hPerpendicular : ∃ m1 m2 : ℝ, m1 * m2 = -1)
  (hSumYIntercepts : P.2 + Q.2 = -2) :
  ∃ area : ℝ, area = 70 :=
sorry

end triangle_area_l286_286845


namespace min_value_frac_sum_l286_286975

-- Define the main problem
theorem min_value_frac_sum (a b : ℝ) (h1 : 2 * a + 3 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ x : ℝ, (x = 25) ∧ ∀ y, (y = (2 / a + 3 / b)) → y ≥ x :=
sorry

end min_value_frac_sum_l286_286975


namespace gcd_40_56_l286_286249

theorem gcd_40_56 : Int.gcd 40 56 = 8 :=
by
  sorry

end gcd_40_56_l286_286249


namespace locus_of_P_is_ellipse_l286_286023

-- Definitions and conditions
def circle_A (x y : ℝ) : Prop := (x + 3) ^ 2 + y ^ 2 = 100
def fixed_point_B : ℝ × ℝ := (3, 0)
def circle_P_passes_through_B (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 - 3) ^ 2 + center.2 ^ 2 = radius ^ 2
def circle_P_tangent_to_A_internally (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  (center.1 + 3) ^ 2 + center.2 ^ 2 = (10 - radius) ^ 2

-- Statement of the problem to prove in Lean
theorem locus_of_P_is_ellipse :
  ∃ (foci_A B : ℝ × ℝ) (a b : ℝ), (foci_A = (-3, 0)) ∧ (foci_B = (3, 0)) ∧ (a = 5) ∧ (b = 4) ∧ 
  (∀ (x y : ℝ), (∃ (P : ℝ × ℝ) (radius : ℝ), circle_P_passes_through_B P radius ∧ circle_P_tangent_to_A_internally P radius ∧ P = (x, y)) ↔ 
  (x ^ 2) / 25 + (y ^ 2) / 16 = 1)
:=
sorry

end locus_of_P_is_ellipse_l286_286023


namespace Smith_gave_Randy_l286_286381

theorem Smith_gave_Randy {original_money Randy_keeps gives_Sally Smith_gives : ℕ}
  (h1: original_money = 3000)
  (h2: Randy_keeps = 2000)
  (h3: gives_Sally = 1200)
  (h4: Randy_keeps + gives_Sally = original_money + Smith_gives) :
  Smith_gives = 200 :=
by
  sorry

end Smith_gave_Randy_l286_286381


namespace y_intercept_l286_286040

theorem y_intercept (x1 y1 : ℝ) (m : ℝ) (h1 : x1 = -2) (h2 : y1 = 4) (h3 : m = 1 / 2) : 
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ↔ y = 1/2 * x + 5) ∧ b = 5 := 
by
  sorry

end y_intercept_l286_286040


namespace fred_dark_blue_marbles_count_l286_286942

/-- Fred's Marble Problem -/
def freds_marbles (red green dark_blue : ℕ) : Prop :=
  red = 38 ∧ green = red / 2 ∧ red + green + dark_blue = 63

theorem fred_dark_blue_marbles_count (red green dark_blue : ℕ) (h : freds_marbles red green dark_blue) :
  dark_blue = 6 :=
by
  sorry

end fred_dark_blue_marbles_count_l286_286942


namespace shaded_area_common_squares_l286_286908

noncomputable def cos_beta : ℝ := 3 / 5

theorem shaded_area_common_squares :
  ∀ (β : ℝ), (0 < β) → (β < pi / 2) → (cos β = cos_beta) →
  (∃ A, A = 4 / 3) :=
by
  sorry

end shaded_area_common_squares_l286_286908


namespace john_runs_more_than_jane_l286_286196

def street_width : ℝ := 25
def block_side : ℝ := 500
def jane_perimeter (side : ℝ) : ℝ := 4 * side
def john_perimeter (side : ℝ) (width : ℝ) : ℝ := 4 * (side + 2 * width)

theorem john_runs_more_than_jane :
  john_perimeter block_side street_width - jane_perimeter block_side = 200 :=
by
  -- Substituting values to verify the equality:
  -- Calculate: john_perimeter 500 25 = 4 * (500 + 2 * 25) = 4 * 550 = 2200
  -- Calculate: jane_perimeter 500 = 4 * 500 = 2000
  sorry

end john_runs_more_than_jane_l286_286196


namespace calculateTotalProfit_l286_286912

-- Defining the initial investments and changes
def initialInvestmentA : ℕ := 5000
def initialInvestmentB : ℕ := 8000
def initialInvestmentC : ℕ := 9000

def additionalInvestmentA : ℕ := 2000
def withdrawnInvestmentB : ℕ := 1000
def additionalInvestmentC : ℕ := 3000

-- Defining the durations
def months1 : ℕ := 4
def months2 : ℕ := 8
def months3 : ℕ := 6

-- C's share of the profit
def shareOfC : ℕ := 45000

-- Total profit to be proved
def totalProfit : ℕ := 103571

-- Lean 4 theorem statement
theorem calculateTotalProfit :
  let ratioA := (initialInvestmentA * months1) + ((initialInvestmentA + additionalInvestmentA) * months2)
  let ratioB := (initialInvestmentB * months1) + ((initialInvestmentB - withdrawnInvestmentB) * months2)
  let ratioC := (initialInvestmentC * months3) + ((initialInvestmentC + additionalInvestmentC) * months3)
  let totalRatio := ratioA + ratioB + ratioC
  (shareOfC / ratioC : ℚ) = (totalProfit / totalRatio : ℚ) :=
sorry

end calculateTotalProfit_l286_286912


namespace grade_assignment_ways_l286_286905

-- Definitions
def num_students : ℕ := 10
def num_choices_per_student : ℕ := 3

-- Theorem statement
theorem grade_assignment_ways : num_choices_per_student ^ num_students = 59049 := by
  sorry

end grade_assignment_ways_l286_286905


namespace gemstones_count_l286_286635

theorem gemstones_count (F B S W SN : ℕ) 
  (hS : S = 1)
  (hSpaatz : S = F / 2 - 2)
  (hBinkie : B = 4 * F)
  (hWhiskers : W = S + 3)
  (hSnowball : SN = 2 * W) :
  B = 24 :=
by
  sorry

end gemstones_count_l286_286635


namespace smallest_number_condition_l286_286434

theorem smallest_number_condition 
  (x : ℕ) 
  (h1 : ∃ k : ℕ, x - 6 = k * 12)
  (h2 : ∃ k : ℕ, x - 6 = k * 16)
  (h3 : ∃ k : ℕ, x - 6 = k * 18)
  (h4 : ∃ k : ℕ, x - 6 = k * 21)
  (h5 : ∃ k : ℕ, x - 6 = k * 28)
  (h6 : ∃ k : ℕ, x - 6 = k * 35)
  (h7 : ∃ k : ℕ, x - 6 = k * 39) 
  : x = 65526 :=
sorry

end smallest_number_condition_l286_286434


namespace solve_problem_l286_286314

theorem solve_problem (Δ q : ℝ) (h1 : 2 * Δ + q = 134) (h2 : 2 * (Δ + q) + q = 230) : Δ = 43 := by
  sorry

end solve_problem_l286_286314


namespace count_divisors_2022_2022_l286_286674

noncomputable def num_divisors_2022_2022 : ℕ :=
  let fac2022 := 2022
  let factor_triplets := [(2, 3, 337), (3, 337, 2), (2, 337, 3), (337, 2, 3), (337, 3, 2), (3, 2, 337)]
  factor_triplets.length

theorem count_divisors_2022_2022 :
  num_divisors_2022_2022 = 6 :=
  by {
    sorry
  }

end count_divisors_2022_2022_l286_286674


namespace lines_intersect_at_point_l286_286270

/-
Given two lines parameterized as:
Line 1: (x, y) = (2, 0) + s * (3, -4)
Line 2: (x, y) = (6, -10) + v * (5, 3)
Prove that these lines intersect at (242/29, -248/29).
-/

def parametric_line_1 (s : ℚ) : ℚ × ℚ :=
  (2 + 3 * s, -4 * s)

def parametric_line_2 (v : ℚ) : ℚ × ℚ :=
  (6 + 5 * v, -10 + 3 * v)

theorem lines_intersect_at_point :
  ∃ (s v : ℚ), parametric_line_1 s = parametric_line_2 v ∧ parametric_line_1 s = (242 / 29, -248 / 29) :=
sorry

end lines_intersect_at_point_l286_286270


namespace find_number_divided_l286_286372

theorem find_number_divided (n : ℕ) (h : n = 21 * 9 + 1) : n = 190 :=
by
  sorry

end find_number_divided_l286_286372


namespace compare_real_numbers_l286_286282

theorem compare_real_numbers (a b c d : ℝ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = 2) :
  d > a ∧ d > b ∧ d > c :=
by
  sorry

end compare_real_numbers_l286_286282


namespace conditional_probability_l286_286530

noncomputable def P (e : Prop) : ℝ := sorry

variable (A B : Prop)

variables (h1 : P A = 0.6)
variables (h2 : P B = 0.5)
variables (h3 : P (A ∨ B) = 0.7)

theorem conditional_probability :
  (P A ∧ P B) / P B = 0.8 := by
  sorry

end conditional_probability_l286_286530


namespace cars_with_air_bags_l286_286069

/--
On a car lot with 65 cars:
- Some have air-bags.
- 30 have power windows.
- 12 have both air-bag and power windows.
- 2 have neither air-bag nor power windows.

Prove that the number of cars with air-bags is 45.
-/
theorem cars_with_air_bags 
    (total_cars : ℕ)
    (cars_with_power_windows : ℕ)
    (cars_with_both : ℕ)
    (cars_with_neither : ℕ)
    (total_cars_eq : total_cars = 65)
    (cars_with_power_windows_eq : cars_with_power_windows = 30)
    (cars_with_both_eq : cars_with_both = 12)
    (cars_with_neither_eq : cars_with_neither = 2) :
    ∃ (A : ℕ), A = 45 :=
by
  sorry

end cars_with_air_bags_l286_286069


namespace probability_non_adjacent_two_twos_l286_286815

theorem probability_non_adjacent_two_twos : 
  let digits := [2, 0, 2, 3]
  let total_arrangements := 12 - 3
  let favorable_arrangements := 5
  (favorable_arrangements / total_arrangements : ℚ) = 5 / 9 :=
by
  sorry

end probability_non_adjacent_two_twos_l286_286815


namespace percent_increase_first_quarter_l286_286458

theorem percent_increase_first_quarter (S : ℝ) (P : ℝ) :
  (S * 1.75 = (S + (P / 100) * S) * 1.346153846153846) → P = 30 :=
by
  intro h
  sorry

end percent_increase_first_quarter_l286_286458


namespace max_value_of_y_l286_286603

theorem max_value_of_y (x : ℝ) (h₁ : 0 < x) (h₂ : x < 4) : 
  ∃ y : ℝ, (y = x * (8 - 2 * x)) ∧ (∀ z : ℝ, z = x * (8 - 2 * x) → z ≤ 8) :=
sorry

end max_value_of_y_l286_286603


namespace trailing_zeroes_in_1200_factorial_l286_286142

theorem trailing_zeroes_in_1200_factorial :
  ∑ k in Finset.range (Nat.floor (Real.log 1200 / Real.log 5) + 1), Nat.floor (1200 / 5^k) = 298 :=
by
  sorry

end trailing_zeroes_in_1200_factorial_l286_286142


namespace find_k_l286_286035

theorem find_k (k : ℕ) (h : 2 * 3 - k + 1 = 0) : k = 7 :=
sorry

end find_k_l286_286035


namespace fractions_addition_l286_286462

theorem fractions_addition : (1 / 6 - 5 / 12 + 3 / 8) = 1 / 8 :=
by
  sorry

end fractions_addition_l286_286462


namespace basketball_free_throws_l286_286350

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 4 * a) 
  (h2 : x = 2 * a) 
  (h3 : 2 * a + 3 * b + x = 72) : 
  x = 18 := 
sorry

end basketball_free_throws_l286_286350


namespace grains_on_11th_more_than_1_to_9_l286_286443

theorem grains_on_11th_more_than_1_to_9 : 
  let grains_on_square (k : ℕ) := 3 ^ k
  let sum_first_n_squares (n : ℕ) := (3 * (3 ^ n - 1) / (3 - 1))
  grains_on_square 11 - sum_first_n_squares 9 = 147624 :=
by
  sorry

end grains_on_11th_more_than_1_to_9_l286_286443


namespace polygon_sides_l286_286190

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) (h2 : n > 2) : n = 8 := by
  -- Conditions given:
  -- h1: (n - 2) * 180 = 3 * 360
  -- h2: n > 2
  sorry

end polygon_sides_l286_286190


namespace original_denominator_is_21_l286_286911

theorem original_denominator_is_21 (d : ℕ) : (3 + 6) / (d + 6) = 1 / 3 → d = 21 :=
by
  intros h
  sorry

end original_denominator_is_21_l286_286911


namespace isabella_hair_length_after_haircut_cm_l286_286688

theorem isabella_hair_length_after_haircut_cm :
  let initial_length_in : ℝ := 18  -- initial length in inches
  let growth_rate_in_per_week : ℝ := 0.5  -- growth rate in inches per week
  let weeks : ℝ := 4  -- time in weeks
  let hair_trimmed_in : ℝ := 2.25  -- length of hair trimmed in inches
  let cm_per_inch : ℝ := 2.54  -- conversion factor from inches to centimeters
  let final_length_in := initial_length_in + growth_rate_in_per_week * weeks - hair_trimmed_in  -- final length in inches
  let final_length_cm := final_length_in * cm_per_inch  -- final length in centimeters
  final_length_cm = 45.085 := by
  sorry

end isabella_hair_length_after_haircut_cm_l286_286688


namespace bianca_birthday_money_l286_286164

-- Define the conditions
def num_friends : ℕ := 5
def money_per_friend : ℕ := 6

-- State the proof problem
theorem bianca_birthday_money : num_friends * money_per_friend = 30 :=
by
  sorry

end bianca_birthday_money_l286_286164


namespace speed_limit_correct_l286_286036

def speed_limit (distance : ℕ) (time : ℕ) (over_limit : ℕ) : ℕ :=
  let speed := distance / time
  speed - over_limit

theorem speed_limit_correct :
  speed_limit 60 1 10 = 50 :=
by
  sorry

end speed_limit_correct_l286_286036


namespace remaining_amount_is_1520_l286_286341

noncomputable def totalAmountToBePaid (deposit : ℝ) (depositRate : ℝ) (taxRate : ℝ) (processingFee : ℝ) : ℝ :=
  let fullPrice := deposit / depositRate
  let salesTax := taxRate * fullPrice
  let totalAdditionalExpenses := salesTax + processingFee
  (fullPrice - deposit) + totalAdditionalExpenses

theorem remaining_amount_is_1520 :
  totalAmountToBePaid 140 0.10 0.15 50 = 1520 := by
  sorry

end remaining_amount_is_1520_l286_286341


namespace part1_correct_part2_correct_part3_correct_l286_286684

-- Example survival rates data (provided conditions)
def survivalRatesA : List (Option Float) := [some 95.5, some 92, some 96.5, some 91.6, some 96.3, some 94.6, none, none, none, none]
def survivalRatesB : List (Option Float) := [some 95.1, some 91.6, some 93.2, some 97.8, some 95.6, some 92.3, some 96.6, none, none, none]
def survivalRatesC : List (Option Float) := [some 97, some 95.4, some 98.2, some 93.5, some 94.8, some 95.5, some 94.5, some 93.5, some 98, some 92.5]

-- Define high-quality project condition
def isHighQuality (rate : Float) : Bool := rate > 95.0

-- Problem 1: Probability of two high-quality years from farm B
noncomputable def probabilityTwoHighQualityB : Float := (4.0 * 3.0) / (7.0 * 6.0)

-- Problem 2: Distribution of high-quality projects from farms A, B, and C
structure DistributionX := 
(P0 : Float) -- probability of 0 high-quality years
(P1 : Float) -- probability of 1 high-quality year
(P2 : Float) -- probability of 2 high-quality years
(P3 : Float) -- probability of 3 high-quality years

noncomputable def distributionX : DistributionX := 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
}

-- Problem 3: Inference of average survival rate from high-quality project probabilities
structure AverageSurvivalRates := 
(avgB : Float) 
(avgC : Float)
(probHighQualityB : Float)
(probHighQualityC : Float)
(canInfer : Bool)

noncomputable def avgSurvivalRates : AverageSurvivalRates := 
{ avgB := (95.1 + 91.6 + 93.2 + 97.8 + 95.6 + 92.3 + 96.6) / 7.0,
  avgC := (97 + 95.4 + 98.2 + 93.5 + 94.8 + 95.5 + 94.5 + 93.5 + 98 + 92.5) / 10.0,
  probHighQualityB := 4.0 / 7.0,
  probHighQualityC := 5.0 / 10.0,
  canInfer := false
}

-- Definitions for proof statements indicating correctness
theorem part1_correct : probabilityTwoHighQualityB = (2.0 / 7.0) := sorry

theorem part2_correct : distributionX = 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
} := sorry

theorem part3_correct : avgSurvivalRates.canInfer = false := sorry

end part1_correct_part2_correct_part3_correct_l286_286684


namespace num_routes_M_to_N_l286_286284

-- Define the relevant points and connections as predicates
def can_reach_directly (x y : String) : Prop :=
  if (x = "C" ∧ y = "N") ∨ (x = "D" ∧ y = "N") ∨ (x = "B" ∧ y = "N") then true else false

def can_reach_via (x y z : String) : Prop :=
  if (x = "A" ∧ y = "C" ∧ z = "N") ∨ (x = "A" ∧ y = "D" ∧ z = "N") ∨ (x = "B" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "B" ∧ y = "C" ∧ z = "N") ∨ (x = "E" ∧ y = "B" ∧ z = "N") ∨ (x = "F" ∧ y = "A" ∧ z = "N") ∨ 
     (x = "F" ∧ y = "B" ∧ z = "N") then true else false

-- Define a function to compute the number of ways from a starting point to "N"
noncomputable def num_routes_to_N : String → ℕ
| "N" => 1
| "C" => 1
| "D" => 1
| "A" => 2 -- from C to N and D to N
| "B" => 4 -- from B to N directly, from B to N via A (2 ways), from B to N via C
| "E" => 4 -- from E to N via B
| "F" => 6 -- from F to N via A (2 ways), from F to N via B (4 ways)
| "M" => 16 -- from M to N via A, B, E, F
| _ => 0

-- The theorem statement
theorem num_routes_M_to_N : num_routes_to_N "M" = 16 :=
by
  sorry

end num_routes_M_to_N_l286_286284


namespace exists_square_divisible_by_12_between_100_and_200_l286_286940

theorem exists_square_divisible_by_12_between_100_and_200 : 
  ∃ x : ℕ, (∃ y : ℕ, x = y * y) ∧ (12 ∣ x) ∧ (100 ≤ x ∧ x ≤ 200) ∧ x = 144 :=
by
  sorry

end exists_square_divisible_by_12_between_100_and_200_l286_286940


namespace roots_squared_sum_l286_286677

theorem roots_squared_sum (a b : ℝ) (h : a^2 - 8 * a + 8 = 0 ∧ b^2 - 8 * b + 8 = 0) : a^2 + b^2 = 48 := 
sorry

end roots_squared_sum_l286_286677


namespace oranges_per_box_l286_286619

theorem oranges_per_box (total_oranges : ℝ) (total_boxes : ℝ) (h1 : total_oranges = 26500) (h2 : total_boxes = 2650) : 
  total_oranges / total_boxes = 10 :=
by 
  sorry

end oranges_per_box_l286_286619


namespace simplify_and_evaluate_l286_286722

theorem simplify_and_evaluate :
  let x := 2 * Real.sqrt 3
  (x - Real.sqrt 2) * (x + Real.sqrt 2) + x * (x - 1) = 22 - 2 * Real.sqrt 3 := 
by
  let x := 2 * Real.sqrt 3
  sorry

end simplify_and_evaluate_l286_286722


namespace value_of_x_squared_add_reciprocal_squared_l286_286518

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l286_286518


namespace subtract_decimal_l286_286156

theorem subtract_decimal : 3.75 - 1.46 = 2.29 :=
by
  sorry

end subtract_decimal_l286_286156


namespace probability_ratio_l286_286934

noncomputable def numWays3_7_5_5_5 : ℕ :=
  (Nat.choose 5 1) * (Nat.choose 4 1) *
  (Nat.choose 25 3) * (Nat.choose 22 7) * 
  (Nat.choose 15 5) * (Nat.choose 10 5) * 
  (Nat.choose 5 5)

noncomputable def numWays5_5_5_5_5 : ℕ :=
  (Nat.choose 25 5) * (Nat.choose 20 5) *
  (Nat.choose 15 5) * (Nat.choose 10 5) * 
  (Nat.choose 5 5)

noncomputable def p : ℚ :=
  (numWays3_7_5_5_5 : ℚ) / (Nat.choose 25 25)

noncomputable def q : ℚ :=
  (numWays5_5_5_5_5 : ℚ) / (Nat.choose 25 25)

theorem probability_ratio : p / q = 12 := by
  sorry

end probability_ratio_l286_286934


namespace largest_multiple_of_7_smaller_than_neg_50_l286_286101

theorem largest_multiple_of_7_smaller_than_neg_50 : ∃ n, (∃ k : ℤ, n = 7 * k) ∧ n < -50 ∧ ∀ m, (∃ j : ℤ, m = 7 * j) ∧ m < -50 → m ≤ n :=
by
  sorry

end largest_multiple_of_7_smaller_than_neg_50_l286_286101


namespace area_of_triangle_intercepts_l286_286812

theorem area_of_triangle_intercepts :
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  area = 168 :=
by
  let f := fun x => (x - 4)^2 * (x + 3)
  let x_intercepts := [4, -3]
  let y_intercept := f 0
  let vertices := [(4, 0), (-3, 0), (0, y_intercept)]
  let base := 4 - (-3)
  let height := y_intercept
  let area := (1 / 2) * base * height
  show area = 168
  sorry

end area_of_triangle_intercepts_l286_286812


namespace tom_age_ratio_l286_286406

-- Define the conditions
variable (T N : ℕ) (ages_of_children_sum : ℕ)

-- Given conditions as definitions
def condition1 : Prop := T = ages_of_children_sum
def condition2 : Prop := (T - N) = 3 * (T - 4 * N)

-- The theorem statement to be proven
theorem tom_age_ratio : condition1 T ages_of_children_sum ∧ condition2 T N → T / N = 11 / 2 :=
by sorry

end tom_age_ratio_l286_286406


namespace square_plot_area_l286_286254

theorem square_plot_area (cost_per_foot : ℕ) (total_cost : ℕ) (P : ℕ) :
  cost_per_foot = 54 →
  total_cost = 3672 →
  P = 4 * (total_cost / (4 * cost_per_foot)) →
  (total_cost / (4 * cost_per_foot)) ^ 2 = 289 :=
by
  intros h_cost_per_foot h_total_cost h_perimeter
  sorry

end square_plot_area_l286_286254


namespace part1_part2_l286_286466

theorem part1 : 2 * (-1)^3 - (-2)^2 / 4 + 10 = 7 := by
  sorry

theorem part2 : abs (-3) - (-6 + 4) / (-1 / 2)^3 + (-1)^2013 = -14 := by
  sorry

end part1_part2_l286_286466


namespace geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l286_286999

-- Define geometric body type
inductive GeometricBody
  | rectangularPrism
  | cylinder

-- Define the condition where both front and left views are rectangles
def hasRectangularViews (body : GeometricBody) : Prop :=
  body = GeometricBody.rectangularPrism ∨ body = GeometricBody.cylinder

-- The theorem statement
theorem geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder (body : GeometricBody) :
  hasRectangularViews body :=
sorry

end geometric_body_with_rectangular_views_is_rectangular_prism_or_cylinder_l286_286999


namespace andrea_rhinestones_needed_l286_286455

theorem andrea_rhinestones_needed (total_needed bought_ratio found_ratio : ℝ) 
  (h1 : total_needed = 45) 
  (h2 : bought_ratio = 1 / 3) 
  (h3 : found_ratio = 1 / 5) : 
  total_needed - (bought_ratio * total_needed + found_ratio * total_needed) = 21 := 
by 
  sorry

end andrea_rhinestones_needed_l286_286455


namespace JameMade112kProfit_l286_286356

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end JameMade112kProfit_l286_286356


namespace iggy_running_hours_l286_286194

theorem iggy_running_hours :
  ∀ (monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour : ℕ),
  monday = 3 → tuesday = 4 → wednesday = 6 → thursday = 8 → friday = 3 →
  pace_in_minutes = 10 → total_minutes_in_hour = 60 →
  ((monday + tuesday + wednesday + thursday + friday) * pace_in_minutes) / total_minutes_in_hour = 4 :=
by
  intros monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour
  sorry

end iggy_running_hours_l286_286194


namespace brigade_harvest_time_l286_286398

theorem brigade_harvest_time (t : ℕ) :
  (t - 5 = (3 * t / 5) + ((t * (t - 8)) / (5 * (t - 4)))) → t = 20 := sorry

end brigade_harvest_time_l286_286398


namespace probability_of_A_l286_286891

noncomputable def probability_that_a_occurs : Prop :=
  ∀ (A B : Event) (P : ProbMeasure),
    independent A B ∧ P.prob A > 0 ∧ P.prob A = 2 * P.prob B ∧ P.prob (A ∪ B) = 14 * P.prob (A ∩ B) →
    P.prob A = 1 / 5

-- The statement definition
theorem probability_of_A : probability_that_a_occurs := by
  sorry

end probability_of_A_l286_286891


namespace cube_root_equation_l286_286346

theorem cube_root_equation (x : ℝ) (h : (2 * x - 14)^(1/3) = -2) : 2 * x + 3 = 9 := by
  sorry

end cube_root_equation_l286_286346


namespace intersection_correct_l286_286058

-- Define the sets M and N
def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

-- Define the expected intersection result
def expected_intersection : Set ℝ := {5, 7, 9}

-- State the theorem
theorem intersection_correct : ∀ x, x ∈ M ∩ N ↔ x ∈ expected_intersection :=
by
  sorry

end intersection_correct_l286_286058


namespace sum_of_angles_in_figure_l286_286001

theorem sum_of_angles_in_figure : 
  let triangles := 3
  let angles_in_triangle := 180
  let square_angles := 4 * 90
  (triangles * angles_in_triangle + square_angles) = 900 := by
  sorry

end sum_of_angles_in_figure_l286_286001


namespace mike_cards_remaining_l286_286220

-- Define initial condition
def mike_initial_cards : ℕ := 87

-- Define the cards bought by Sam
def sam_bought_cards : ℕ := 13

-- Define the expected remaining cards
def mike_final_cards := mike_initial_cards - sam_bought_cards

-- Theorem to prove the final count of Mike's baseball cards
theorem mike_cards_remaining : mike_final_cards = 74 := by
  sorry

end mike_cards_remaining_l286_286220


namespace right_triangle_medians_l286_286047

theorem right_triangle_medians (m : ℝ) :
  (∃ (a b c d : ℝ), 
    let P := (a, b) 
    ∧ let Q := (a, b + 2 * c)
    ∧ let R := (a + 2 * d, b)
    ∧ let M1 := (a, b + c)
    ∧ let M2 := (a + d, b)
    ∧ (M1.2 - P.2) / (M1.1 - P.1) = 2
    ∧ (M2.2 - R.2) / (M2.1 - R.1) = m 
    ) → m = 2 :=
by
  -- proof goes here
  sorry

end right_triangle_medians_l286_286047


namespace max_sum_of_integer_pairs_l286_286241

theorem max_sum_of_integer_pairs (x y : ℤ) (h : (x-1)^2 + (y+2)^2 = 36) : 
  max (x + y) = 5 :=
sorry

end max_sum_of_integer_pairs_l286_286241


namespace page_cost_in_cents_l286_286689

theorem page_cost_in_cents (notebooks pages_per_notebook total_cost : ℕ)
  (h_notebooks : notebooks = 2)
  (h_pages_per_notebook : pages_per_notebook = 50)
  (h_total_cost : total_cost = 5 * 100) :
  (total_cost / (notebooks * pages_per_notebook)) = 5 :=
by
  sorry

end page_cost_in_cents_l286_286689


namespace eval_expr_l286_286151

theorem eval_expr : (1 / (5^2)^4 * 5^11 * 2) = 250 := by
  sorry

end eval_expr_l286_286151


namespace cost_unit_pen_max_profit_and_quantity_l286_286275

noncomputable def cost_pen_A : ℝ := 5
noncomputable def cost_pen_B : ℝ := 10
noncomputable def profit_pen_A : ℝ := 2
noncomputable def profit_pen_B : ℝ := 3
noncomputable def spent_on_A : ℝ := 400
noncomputable def spent_on_B : ℝ := 800
noncomputable def total_pens : ℝ := 300

theorem cost_unit_pen : (spent_on_A / cost_pen_A) = (spent_on_B / (cost_pen_A + 5)) := by
  sorry

theorem max_profit_and_quantity
    (xa xb : ℝ)
    (h1 : xa ≥ 4 * xb)
    (h2 : xa + xb = total_pens)
    : ∃ (wa : ℝ), wa = 2 * xa + 3 * xb ∧ xa = 240 ∧ xb = 60 ∧ wa = 660 := by
  sorry

end cost_unit_pen_max_profit_and_quantity_l286_286275


namespace smallest_lcm_four_digit_integers_with_gcd_five_l286_286330

open Nat

theorem smallest_lcm_four_digit_integers_with_gcd_five : ∃ k ℓ : ℕ, 1000 ≤ k ∧ k < 10000 ∧ 1000 ≤ ℓ ∧ ℓ < 10000 ∧ gcd k ℓ = 5 ∧ lcm k ℓ = 203010 :=
by
  use 1005
  use 1010
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_lcm_four_digit_integers_with_gcd_five_l286_286330


namespace quadratic_must_have_m_eq_neg2_l286_286804

theorem quadratic_must_have_m_eq_neg2 (m : ℝ) (h : (m - 2) * x^|m| - 3 * x - 4 = 0) :
  (|m| = 2) ∧ (m ≠ 2) → m = -2 :=
by
  sorry

end quadratic_must_have_m_eq_neg2_l286_286804


namespace max_column_sum_l286_286648

-- Definitions of the 5x5 grid and related constraints
def grid : Type := Matrix (Fin 5) (Fin 5) ℕ

def is_valid_grid (G : grid) : Prop :=
  (∀ i j, 1 ≤ G i j ∧ G i j ≤ 5) ∧  -- All numbers are within the range 1 to 5
  (∀ i j k, | G i j - G i k | ≤ 2) ∧  -- Absolute difference in columns is at most 2
  (∀ i, (Finset.univ.map (G i)).card = 5) ∧  -- Each row contains unique numbers 1 to 5
  (∀ j, (Finset.univ.map (G i)).card = 5)    -- Each column contains unique numbers 1 to 5

def column_sum (G : grid) (j : Fin 5) : ℕ := 
  (Finset.univ.sum (λ i => G i j))

-- The maximum possible value of M
def max_M (G : grid) : ℕ :=
  Finset.univ.min' (Finset.image (column_sum G) Finset.univ)

theorem max_column_sum : ∃ G : grid, is_valid_grid G ∧ max_M G = 10 := 
by 
  sorry

end max_column_sum_l286_286648


namespace units_digit_of_sum_is_three_l286_286917

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_of_factorials : ℕ :=
  (List.range 10).map factorial |>.sum

def power_of_ten (n : ℕ) : ℕ :=
  10^n

theorem units_digit_of_sum_is_three : 
  units_digit (sum_of_factorials + power_of_ten 3) = 3 := by
  sorry

end units_digit_of_sum_is_three_l286_286917


namespace pyramid_edges_sum_l286_286623

noncomputable def sum_of_pyramid_edges (s : ℝ) (h : ℝ) : ℝ :=
  let diagonal := s * Real.sqrt 2
  let half_diagonal := diagonal / 2
  let slant_height := Real.sqrt (half_diagonal^2 + h^2)
  4 * s + 4 * slant_height

theorem pyramid_edges_sum
  (s : ℝ) (h : ℝ)
  (hs : s = 15)
  (hh : h = 15) :
  sum_of_pyramid_edges s h = 135 :=
sorry

end pyramid_edges_sum_l286_286623


namespace campers_afternoon_l286_286117

noncomputable def campers_morning : ℕ := 35
noncomputable def campers_total : ℕ := 62

theorem campers_afternoon :
  campers_total - campers_morning = 27 :=
by
  sorry

end campers_afternoon_l286_286117


namespace eval_f_pi_over_8_l286_286182

noncomputable def f (θ : ℝ) : ℝ :=
(2 * (Real.sin (θ / 2)) ^ 2 - 1) / (Real.sin (θ / 2) * Real.cos (θ / 2)) + 2 * Real.tan θ

theorem eval_f_pi_over_8 : f (π / 8) = -4 :=
sorry

end eval_f_pi_over_8_l286_286182


namespace solve_quadratic_1_solve_quadratic_2_l286_286081

open Real

theorem solve_quadratic_1 :
  (∃ x : ℝ, x^2 - 2 * x - 7 = 0) ∧
  (∀ x : ℝ, x^2 - 2 * x - 7 = 0 → x = 1 + 2 * sqrt 2 ∨ x = 1 - 2 * sqrt 2) :=
sorry

theorem solve_quadratic_2 :
  (∃ x : ℝ, 3 * (x - 2)^2 = x * (x - 2)) ∧
  (∀ x : ℝ, 3 * (x - 2)^2 = x * (x - 2) → x = 2 ∨ x = 3) :=
sorry

end solve_quadratic_1_solve_quadratic_2_l286_286081


namespace total_cost_is_18_l286_286383

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18_l286_286383


namespace three_digit_number_prime_factors_l286_286471

theorem three_digit_number_prime_factors (A B C : ℕ) (hA : 1 ≤ A) (hC : 1 ≤ C) (hA_C: A ≠ C): 
  (∃ k : ℕ, 99 * (A - C) = 3 * k) ∧ (∃ m : ℕ, 99 * (A - C) = 11 * m) :=
by
  have h : 99 = 3 * 3 * 11 := by norm_num
  sorry

end three_digit_number_prime_factors_l286_286471


namespace horse_distance_traveled_l286_286754

theorem horse_distance_traveled :
  let r2 := 12
  let n2 := 120
  let D2 := n2 * 2 * Real.pi * r2
  D2 = 2880 * Real.pi :=
by
  sorry

end horse_distance_traveled_l286_286754


namespace radius_correct_l286_286847

noncomputable def radius_of_circle (chord_length tang_secant_segment : ℝ) : ℝ :=
  let r := 6.25
  r

theorem radius_correct
  (chord_length : ℝ)
  (tangent_secant_segment : ℝ)
  (parallel_secant_internal_segment : ℝ)
  : chord_length = 10 ∧ parallel_secant_internal_segment = 12 → radius_of_circle chord_length parallel_secant_internal_segment = 6.25 :=
by
  intros h
  sorry

end radius_correct_l286_286847


namespace polygon_sides_l286_286189

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) (h2 : n > 2) : n = 8 := by
  -- Conditions given:
  -- h1: (n - 2) * 180 = 3 * 360
  -- h2: n > 2
  sorry

end polygon_sides_l286_286189


namespace ticket_queue_correct_l286_286588

-- Define the conditions
noncomputable def ticket_queue_count (m n : ℕ) (h : n ≥ m) : ℕ :=
  (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1))

-- State the theorem
theorem ticket_queue_correct (m n : ℕ) (h : n ≥ m) :
  ticket_queue_count m n h = (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1)) :=
by
  sorry

end ticket_queue_correct_l286_286588


namespace investment_double_l286_286448

theorem investment_double (A : ℝ) (r t : ℝ) (hA : 0 < A) (hr : 0 < r) :
  2 * A ≤ A * (1 + r)^t ↔ t ≥ (Real.log 2) / (Real.log (1 + r)) := 
by
  sorry

end investment_double_l286_286448


namespace scale_fragments_l286_286866

-- definitions for adults and children heights
def height_adults : Fin 100 → ℕ 
def height_children : Fin 100 → ℕ 

-- the main theorem statement
theorem scale_fragments (h₁ : ∀ i, height_children i < height_adults i) :
  ∃ (k : Fin 100 → ℕ), ∀ (i j : Fin 100), 
  k i * height_children i < k i * height_adults i ∧ 
  (i ≠ j → ∀ (a: ℕ), k i * height_children i < a → k i * height_adults i < a) :=
sorry

end scale_fragments_l286_286866


namespace fill_tank_with_leak_l286_286376

theorem fill_tank_with_leak (A L : ℝ) (h1 : A = 1 / 6) (h2 : L = 1 / 18) : (1 / (A - L)) = 9 :=
by
  sorry

end fill_tank_with_leak_l286_286376


namespace resulting_curve_eq_l286_286496

def is_on_circle (x y : ℝ) : Prop := x^2 + y^2 = 9

def transformed_curve (x y: ℝ) : Prop := 
  ∃ (x0 y0 : ℝ), 
    is_on_circle x0 y0 ∧ 
    x = x0 ∧ 
    y = 4 * y0

theorem resulting_curve_eq : ∀ (x y : ℝ), transformed_curve x y → (x^2 / 9 + y^2 / 144 = 1) :=
by
  intros x y h
  sorry

end resulting_curve_eq_l286_286496


namespace binomial_probability_p_l286_286125

noncomputable def binomial_expected_value (n p : ℝ) := n * p
noncomputable def binomial_variance (n p : ℝ) := n * p * (1 - p)

theorem binomial_probability_p (n p : ℝ) (h1: binomial_expected_value n p = 2) (h2: binomial_variance n p = 1) : 
  p = 0.5 :=
by
  sorry

end binomial_probability_p_l286_286125


namespace quadratic_one_positive_root_l286_286307

theorem quadratic_one_positive_root (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (∀ y : ℝ, y ∈ {t | t^2 - a * t + a - 2 = 0} → y = x)) → a ≤ 2 :=
by
  sorry

end quadratic_one_positive_root_l286_286307


namespace tan_to_sin_cos_l286_286659

theorem tan_to_sin_cos (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := 
sorry

end tan_to_sin_cos_l286_286659


namespace find_k_l286_286522

theorem find_k (m : ℝ) (h : ∃ A B : ℝ, (m^3 - 24*m + 16) = (m^2 - 8*m) * (A*m + B) ∧ A - 8 = -k ∧ -8*B = -24) : k = 5 :=
sorry

end find_k_l286_286522


namespace BoatCrafters_boats_total_l286_286288

theorem BoatCrafters_boats_total
  (n_february: ℕ)
  (h_february: n_february = 5)
  (h_march: 3 * n_february = 15)
  (h_april: 3 * 15 = 45) :
  n_february + 15 + 45 = 65 := 
sorry

end BoatCrafters_boats_total_l286_286288


namespace riza_son_age_l286_286717

theorem riza_son_age (R S : ℕ) (h1 : R = S + 25) (h2 : R + S = 105) : S = 40 :=
by
  sorry

end riza_son_age_l286_286717


namespace ked_ben_eggs_ratio_l286_286067

theorem ked_ben_eggs_ratio 
  (saly_needs_ben_weekly_ratio : ℕ)
  (weeks_in_month : ℕ := 4) 
  (total_production_month : ℕ := 124)
  (saly_needs_weekly : ℕ := 10) 
  (ben_needs_weekly : ℕ := 14)
  (ben_needs_monthly : ℕ := ben_needs_weekly * weeks_in_month)
  (saly_needs_monthly : ℕ := saly_needs_weekly * weeks_in_month)
  (total_saly_ben_monthly : ℕ := saly_needs_monthly + ben_needs_monthly)
  (ked_needs_monthly : ℕ := total_production_month - total_saly_ben_monthly)
  (ked_needs_weekly : ℕ := ked_needs_monthly / weeks_in_month) :
  ked_needs_weekly / ben_needs_weekly = 1 / 2 :=
sorry

end ked_ben_eggs_ratio_l286_286067


namespace linear_system_solution_l286_286671

theorem linear_system_solution (k x y : ℝ) (h₁ : x + y = 5 * k) (h₂ : x - y = 9 * k) (h₃ : 2 * x + 3 * y = 6) :
  k = 3 / 4 :=
by
  sorry

end linear_system_solution_l286_286671


namespace no_good_number_exists_l286_286758

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_good (n : ℕ) : Prop :=
  (n % sum_of_digits n = 0) ∧
  ((n + 1) % sum_of_digits (n + 1) = 0) ∧
  ((n + 2) % sum_of_digits (n + 2) = 0) ∧
  ((n + 3) % sum_of_digits (n + 3) = 0)

theorem no_good_number_exists : ¬ ∃ n : ℕ, is_good n :=
by sorry

end no_good_number_exists_l286_286758


namespace turtles_on_Happy_Island_l286_286403

theorem turtles_on_Happy_Island (L H : ℕ) (hL : L = 25) (hH : H = 2 * L + 10) : H = 60 :=
by
  sorry

end turtles_on_Happy_Island_l286_286403


namespace unit_trip_to_expo_l286_286767

theorem unit_trip_to_expo (n : ℕ) (cost : ℕ) (total_cost : ℕ) :
  (n ≤ 30 → cost = 120) ∧ 
  (n > 30 → cost = 120 - 2 * (n - 30) ∧ cost ≥ 90) →
  (total_cost = 4000) →
  (total_cost = n * cost) →
  n = 40 :=
by
  sorry

end unit_trip_to_expo_l286_286767


namespace minimum_cuts_to_divide_cube_l286_286872

open Real

theorem minimum_cuts_to_divide_cube (a b : ℕ) (ha : a = 4) (hb : b = 64) : 
  (∃ n : ℕ, 2^n = b ∧ n = log 64 / log 2) :=
begin
  use 6,
  split,
  { norm_num, },
  { norm_num, },
end

end minimum_cuts_to_divide_cube_l286_286872


namespace pizza_left_percentage_l286_286842

-- Define the fractions eaten by Ravindra and Hongshu
def fraction_eaten_by_ravindra := 2 / 5
def fraction_eaten_by_hongshu := 1 / 2 * fraction_eaten_by_ravindra

-- Define the total fraction eaten and the remaining fraction
def total_fraction_eaten := fraction_eaten_by_ravindra + fraction_eaten_by_hongshu
def fraction_left := 1 - total_fraction_eaten

-- Prove that the remaining fraction as a percentage
theorem pizza_left_percentage : fraction_left * 100 = 40 := by
  -- We skip the proof with sorry
  sorry

end pizza_left_percentage_l286_286842


namespace find_value_of_2_minus_c_l286_286033

theorem find_value_of_2_minus_c (c d : ℤ) (h1 : 5 + c = 6 - d) (h2 : 3 + d = 8 + c) : 2 - c = -1 := 
by
  sorry

end find_value_of_2_minus_c_l286_286033


namespace win_percentage_of_people_with_envelopes_l286_286685

theorem win_percentage_of_people_with_envelopes (total_people : ℕ) (percent_with_envelopes : ℝ) (winners : ℕ) (num_with_envelopes : ℕ) : 
  total_people = 100 ∧ percent_with_envelopes = 0.40 ∧ num_with_envelopes = total_people * percent_with_envelopes ∧ winners = 8 → 
    (winners / num_with_envelopes) * 100 = 20 :=
by
  intros
  sorry

end win_percentage_of_people_with_envelopes_l286_286685


namespace find_point_P_l286_286310

structure Point :=
(x : ℝ)
(y : ℝ)

def A : Point := ⟨2, 3⟩
def B : Point := ⟨4, -3⟩

def vector (P Q : Point) : Point :=
⟨Q.x - P.x, Q.y - P.y⟩

def magnitude_ratio (P A B : Point) (r : ℝ) : Prop :=
  let AP := vector A P
  let PB := vector P B
  (AP.x, AP.y) = (r * PB.x, r * PB.y)

theorem find_point_P (P : Point) : 
  magnitude_ratio P A B (4/3) → (P.x = 10 ∧ P.y = -21) :=
sorry

end find_point_P_l286_286310


namespace find_a_l286_286948

theorem find_a (x a a1 a2 a3 a4 : ℝ) :
  (x + a) ^ 4 = x ^ 4 + a1 * x ^ 3 + a2 * x ^ 2 + a3 * x + a4 → 
  a1 + a2 + a3 = 64 → a = 2 :=
by
  sorry

end find_a_l286_286948


namespace new_credit_card_balance_l286_286838

theorem new_credit_card_balance (i g x r n : ℝ)
    (h_i : i = 126)
    (h_g : g = 60)
    (h_x : x = g / 2)
    (h_r : r = 45)
    (h_n : n = (i + g + x) - r) :
    n = 171 :=
sorry

end new_credit_card_balance_l286_286838


namespace penny_money_left_is_5_l286_286709

def penny_initial_money : ℤ := 20
def socks_pairs : ℤ := 4
def price_per_pair_of_socks : ℤ := 2
def price_of_hat : ℤ := 7

def total_cost_of_socks : ℤ := socks_pairs * price_per_pair_of_socks
def total_cost_of_hat_and_socks : ℤ := total_cost_of_socks + price_of_hat
def penny_money_left : ℤ := penny_initial_money - total_cost_of_hat_and_socks

theorem penny_money_left_is_5 : penny_money_left = 5 := by
  sorry

end penny_money_left_is_5_l286_286709


namespace simplify_and_evaluate_expression_l286_286390

noncomputable def given_expression (a : ℝ) : ℝ :=
  (a - 1 - (2 * a - 1) / (a + 1)) / ((a^2 - 4 * a + 4) / (a + 1))

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 2 + Real.sqrt 3) :
  given_expression a = (2 * Real.sqrt 3 + 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l286_286390


namespace max_principals_in_8_years_l286_286641

theorem max_principals_in_8_years 
  (years_in_term : ℕ)
  (terms_in_given_period : ℕ)
  (term_length : ℕ)
  (term_length_eq : term_length = 4)
  (given_period : ℕ)
  (given_period_eq : given_period = 8) :
  terms_in_given_period = given_period / term_length :=
by
  rw [term_length_eq, given_period_eq]
  sorry

end max_principals_in_8_years_l286_286641


namespace smallest_possible_denominator_l286_286162

theorem smallest_possible_denominator :
  ∃ p q : ℕ, q < 4027 ∧ (1/2014 : ℚ) < p / q ∧ p / q < (1/2013 : ℚ) → ∃ q : ℕ, q = 4027 :=
by
  sorry

end smallest_possible_denominator_l286_286162


namespace integer_pairs_solution_l286_286300

theorem integer_pairs_solution (k : ℕ) (h : k ≠ 1) : 
  ∃ (m n : ℤ), 
    ((m - n) ^ 2 = 4 * m * n / (m + n - 1)) ∧ 
    (m = k^2 + k / 2 ∧ n = k^2 - k / 2) ∨ 
    (m = k^2 - k / 2 ∧ n = k^2 + k / 2) :=
sorry

end integer_pairs_solution_l286_286300


namespace color_points_l286_286219

open Finset

noncomputable def exists_coloring (S : Finset (ℝ × ℝ)) (hS : S.card = 2004) : Prop :=
  ∃ color : (ℝ × ℝ) → bool,
    ∀ p q ∈ S,
      (∃ k, k ∈ {l ∈ S.powerset 2 | l.card = 2} ∧ p ∈ k ∧ q ∈ k ∧ (¬(color p = color q) ↔ ∃ l ∈ (S.powerset 2).erase k, p ∈ l ∧ q ∈ l)). 

theorem color_points (S : Finset (ℝ × ℝ)) (hS : S.card = 2004) (hS_no_three_collinear : ∀ (p q r : (ℝ × ℝ)), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → p ≠ r → ¬Collinear {p, q, r}) :
  exists_coloring S hS :=
sorry

end color_points_l286_286219


namespace tom_marbles_l286_286048

def jason_marbles := 44
def marbles_difference := 20

theorem tom_marbles : (jason_marbles - marbles_difference = 24) :=
by
  sorry

end tom_marbles_l286_286048


namespace transformation_maps_segment_l286_286257

variables (C D : ℝ × ℝ) (C' D' : ℝ × ℝ)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem transformation_maps_segment :
  reflect_x (reflect_y (3, -2)) = (-3, 2) ∧ reflect_x (reflect_y (4, -5)) = (-4, 5) :=
by {
  sorry
}

end transformation_maps_segment_l286_286257


namespace quadratic_sum_roots_twice_difference_l286_286652

theorem quadratic_sum_roots_twice_difference
  (a b c x₁ x₂ : ℝ)
  (h_eq : a * x₁^2 + b * x₁ + c = 0)
  (h_eq2 : a * x₂^2 + b * x₂ + c = 0)
  (h_sum_twice_diff: x₁ + x₂ = 2 * (x₁ - x₂)) :
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_sum_roots_twice_difference_l286_286652


namespace largest_multiple_of_7_smaller_than_neg_50_l286_286102

theorem largest_multiple_of_7_smaller_than_neg_50 : ∃ n, (∃ k : ℤ, n = 7 * k) ∧ n < -50 ∧ ∀ m, (∃ j : ℤ, m = 7 * j) ∧ m < -50 → m ≤ n :=
by
  sorry

end largest_multiple_of_7_smaller_than_neg_50_l286_286102


namespace average_output_l286_286887

theorem average_output (t1 t2 t_total : ℝ) (c1 c2 c_total : ℕ) 
                        (h1 : c1 = 60) (h2 : c2 = 60) 
                        (rate1 : ℝ := 15) (rate2 : ℝ := 60) :
  t1 = c1 / rate1 ∧ t2 = c2 / rate2 ∧ t_total = t1 + t2 ∧ c_total = c1 + c2 → 
  (c_total / t_total = 24) := 
by 
  sorry

end average_output_l286_286887


namespace sum_of_coeffs_eq_negative_21_l286_286298

noncomputable def expand_and_sum_coeff (d : ℤ) : ℤ :=
  let expression := -(4 - d) * (d + 2 * (4 - d))
  let expanded_form := -d^2 + 12*d - 32
  let sum_of_coeffs := -1 + 12 - 32
  sum_of_coeffs

theorem sum_of_coeffs_eq_negative_21 (d : ℤ) : expand_and_sum_coeff d = -21 := by
  sorry

end sum_of_coeffs_eq_negative_21_l286_286298


namespace difference_seven_three_times_l286_286388

theorem difference_seven_three_times (n : ℝ) (h1 : n = 3) 
  (h2 : 7 * n = 3 * n + (21.0 - 9.0)) :
  7 * n - 3 * n = 12.0 := by
  sorry

end difference_seven_three_times_l286_286388


namespace taller_building_height_l286_286591

theorem taller_building_height
  (H : ℕ) -- H is the height of the taller building
  (h_ratio : (H - 36) / H = 5 / 7) -- heights ratio condition
  (h_diff : H > 36) -- height difference must respect physics
  : H = 126 := sorry

end taller_building_height_l286_286591


namespace sum_of_squares_of_roots_l286_286163

theorem sum_of_squares_of_roots : 
  ∀ r1 r2 : ℝ, (r1 + r2 = 10) → (r1 * r2 = 9) → (r1 > 5 ∨ r2 > 5) → (r1^2 + r2^2 = 82) :=
by
  intros r1 r2 h1 h2 h3
  sorry

end sum_of_squares_of_roots_l286_286163


namespace number_of_guests_l286_286097

def cook_per_minute : ℕ := 10
def time_to_cook : ℕ := 80
def guests_ate_per_guest : ℕ := 5
def guests_to_serve : ℕ := 20 -- This is what we'll prove.

theorem number_of_guests 
    (cook_per_8min : cook_per_minute = 10)
    (total_time : time_to_cook = 80)
    (eat_rate : guests_ate_per_guest = 5) :
    (time_to_cook * cook_per_minute) / guests_ate_per_guest = guests_to_serve := 
by 
  sorry

end number_of_guests_l286_286097


namespace average_marks_second_class_l286_286301

variable (average_marks_first_class : ℝ) (students_first_class : ℕ)
variable (students_second_class : ℕ) (combined_average_marks : ℝ)

theorem average_marks_second_class (H1 : average_marks_first_class = 60)
  (H2 : students_first_class = 55) (H3 : students_second_class = 48)
  (H4 : combined_average_marks = 59.067961165048544) :
  48 * 57.92 = 103 * 59.067961165048544 - 3300 := by
  sorry

end average_marks_second_class_l286_286301


namespace gcd_2_pow_2018_2_pow_2029_l286_286247

theorem gcd_2_pow_2018_2_pow_2029 : Nat.gcd (2^2018 - 1) (2^2029 - 1) = 2047 :=
by
  sorry

end gcd_2_pow_2018_2_pow_2029_l286_286247


namespace slices_served_during_dinner_l286_286273

theorem slices_served_during_dinner (slices_lunch slices_total slices_dinner : ℕ)
  (h1 : slices_lunch = 7)
  (h2 : slices_total = 12)
  (h3 : slices_dinner = slices_total - slices_lunch) :
  slices_dinner = 5 := 
by 
  sorry

end slices_served_during_dinner_l286_286273


namespace polygon_sides_l286_286191

theorem polygon_sides (n : ℕ) (h_interior : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_l286_286191


namespace find_principal_l286_286449

theorem find_principal (R P : ℝ) (h₁ : (P * R * 10) / 100 = P * R * 0.1)
  (h₂ : (P * (R + 3) * 10) / 100 = P * (R + 3) * 0.1)
  (h₃ : P * 0.1 * (R + 3) - P * 0.1 * R = 300) : 
  P = 1000 := 
sorry

end find_principal_l286_286449


namespace students_who_saw_l286_286633

variable (B G : ℕ)

theorem students_who_saw (h : B + G = 33) : (2 * G / 3) + (2 * B / 3) = 22 :=
by
  sorry

end students_who_saw_l286_286633


namespace part1_part2_l286_286165

def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, k * k = n

def calculate_P (x y : ℤ) : ℤ := 
  (x - y) / 9

def y_from_x (x : ℤ) : ℤ :=
  let first_three := x / 10
  let last_digit := x % 10
  last_digit * 1000 + first_three

def calculate_s (a b : ℕ) : ℤ :=
  1100 + 20 * a + b

def calculate_t (a b : ℕ) : ℤ :=
  b * 1000 + a * 100 + 23

theorem part1 : calculate_P 5324 (y_from_x 5324) = 88 := by
  sorry

theorem part2 :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 4 ∧ 1 ≤ b ∧ b ≤ 9 ∧
  let s := calculate_s a b
  let t := calculate_t a b
  let P_s := calculate_P s (y_from_x s)
  let P_t := calculate_P t (y_from_x t)
  let difference := P_t - P_s - a - b
  is_perfect_square difference ∧ P_t = -161 := by
  sorry

end part1_part2_l286_286165


namespace no_prize_for_A_l286_286914

variable (A B C D : Prop)

theorem no_prize_for_A 
  (hA : A → B) 
  (hB : B → C) 
  (hC : ¬D → ¬C) 
  (exactly_one_did_not_win : (¬A ∧ B ∧ C ∧ D) ∨ (A ∧ ¬B ∧ C ∧ D) ∨ (A ∧ B ∧ ¬C ∧ D) ∨ (A ∧ B ∧ C ∧ ¬D)) 
: ¬A := 
sorry

end no_prize_for_A_l286_286914


namespace total_cost_of_items_l286_286387

theorem total_cost_of_items (cost_of_soda : ℕ) (cost_of_soup : ℕ) (cost_of_sandwich : ℕ) (total_cost : ℕ) 
  (h1 : cost_of_soda = 1)
  (h2 : cost_of_soup = 3 * cost_of_soda)
  (h3 : cost_of_sandwich = 3 * cost_of_soup) :
  total_cost = 3 * cost_of_soda + 2 * cost_of_soup + cost_of_sandwich :=
by
  unfold total_cost
  show 3 * 1 + 2 * (3 * 1) + (3 * (3 * 1)) = 18
  rfl

end total_cost_of_items_l286_286387


namespace football_field_area_l286_286267

-- Define the conditions
def fertilizer_spread : ℕ := 1200
def area_partial : ℕ := 3600
def fertilizer_partial : ℕ := 400

-- Define the expected result
def area_total : ℕ := 10800

-- Theorem to prove
theorem football_field_area :
  (fertilizer_spread / (fertilizer_partial / area_partial)) = area_total :=
by sorry

end football_field_area_l286_286267


namespace find_v_l286_286083

variables (a b c : ℝ)

def condition1 := (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -6
def condition2 := (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

theorem find_v (h1 : condition1 a b c) (h2 : condition2 a b c) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 17 / 2 :=
by
  sorry

end find_v_l286_286083


namespace wholesale_price_is_90_l286_286764

theorem wholesale_price_is_90 
  (R S W: ℝ)
  (h1 : R = 120)
  (h2 : S = R - 0.1 * R)
  (h3 : S = W + 0.2 * W)
  : W = 90 := 
by
  sorry

end wholesale_price_is_90_l286_286764


namespace matrix_cube_computation_l286_286469

-- Define the original matrix
def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2], ![2, 0]]

-- Define the expected result matrix
def expected_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-8, 0], ![0, -8]]

-- State the theorem to be proved
theorem matrix_cube_computation : matrix1 ^ 3 = expected_matrix :=
  by sorry

end matrix_cube_computation_l286_286469


namespace candy_necklaces_left_l286_286007

theorem candy_necklaces_left (total_packs : ℕ) (candy_per_pack : ℕ) 
  (opened_packs : ℕ) (candy_necklaces : ℕ)
  (h1 : total_packs = 9) 
  (h2 : candy_per_pack = 8) 
  (h3 : opened_packs = 4)
  (h4 : candy_necklaces = total_packs * candy_per_pack) :
  (total_packs - opened_packs) * candy_per_pack = 40 :=
by
  sorry

end candy_necklaces_left_l286_286007


namespace thirteen_percent_greater_than_80_l286_286108

theorem thirteen_percent_greater_than_80 (x : ℝ) (h : x = 1.13 * 80) : x = 90.4 :=
sorry

end thirteen_percent_greater_than_80_l286_286108


namespace alina_sent_fewer_messages_l286_286916

-- Definitions based on conditions
def messages_lucia_day1 : Nat := 120
def messages_lucia_day2 : Nat := 1 / 3 * messages_lucia_day1
def messages_lucia_day3 : Nat := messages_lucia_day1
def messages_total : Nat := 680

-- Def statement for Alina's messages on the first day, which we need to find as 100
def messages_alina_day1 : Nat := 100

-- Condition checks
def condition_alina_day2 : Prop := 2 * messages_alina_day1 = 2 * 100
def condition_alina_day3 : Prop := messages_alina_day1 = 100
def condition_total_messages : Prop := 
  messages_alina_day1 + messages_lucia_day1 +
  2 * messages_alina_day1 + messages_lucia_day2 +
  messages_alina_day1 + messages_lucia_day1 = messages_total

-- Theorem statement
theorem alina_sent_fewer_messages :
  messages_lucia_day1 - messages_alina_day1 = 20 :=
by
  -- Ensure the conditions hold
  have h1 : messages_alina_day1 = 100 := by sorry
  have h2 : condition_alina_day2 := by sorry
  have h3 : condition_alina_day3 := by sorry
  have h4 : condition_total_messages := by sorry
  -- Prove the theorem
  sorry

end alina_sent_fewer_messages_l286_286916


namespace trig_identity_problem_l286_286337

theorem trig_identity_problem
  (x : ℝ) (a b c : ℕ)
  (h1 : 0 < x ∧ x < (Real.pi / 2))
  (h2 : Real.sin x - Real.cos x = Real.pi / 4)
  (h3 : Real.tan x + 1 / Real.tan x = (a : ℝ) / (b - Real.pi^c)) :
  a + b + c = 50 :=
sorry

end trig_identity_problem_l286_286337


namespace minimize_reciprocals_l286_286976

theorem minimize_reciprocals (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 30) :
  (a = 10 ∧ b = 5) → ∀ x y : ℕ, (x > 0) → (y > 0) → (x + 4 * y = 30) → (1 / (x : ℝ) + 1 / (y : ℝ) ≥ 1 / 10 + 1 / 5) := 
by {
  sorry
}

end minimize_reciprocals_l286_286976


namespace sphere_surface_area_l286_286686

theorem sphere_surface_area
  (a : ℝ)
  (expansion : (1 - 2 * 1 : ℝ)^6 = a)
  (a_value : a = 1) :
  4 * Real.pi * ((Real.sqrt (2^2 + 3^2 + a^2) / 2)^2) = 14 * Real.pi :=
by
  sorry

end sphere_surface_area_l286_286686


namespace union_A_B_intersection_complement_A_B_l286_286025

def A := {x : ℝ | 3 ≤ x ∧ x < 7}
def B := {x : ℝ | 4 < x ∧ x < 10}

theorem union_A_B :
  A ∪ B = {x : ℝ | 3 ≤ x ∧ x < 10} :=
sorry

def complement_A := {x : ℝ | x < 3 ∨ x ≥ 7}

theorem intersection_complement_A_B :
  (complement_A ∩ B) = {x : ℝ | 7 ≤ x ∧ x < 10} :=
sorry

end union_A_B_intersection_complement_A_B_l286_286025


namespace chessboard_ratio_sum_l286_286923

theorem chessboard_ratio_sum :
  let m := 19
  let n := 135
  m + n = 154 :=
by
  sorry

end chessboard_ratio_sum_l286_286923


namespace point_distance_is_pm_3_l286_286997

theorem point_distance_is_pm_3 (Q : ℝ) (h : |Q - 0| = 3) : Q = 3 ∨ Q = -3 :=
sorry

end point_distance_is_pm_3_l286_286997


namespace expected_value_of_unfair_die_l286_286002

noncomputable def seven_sided_die_expected_value : ℝ :=
  let p7 := 1 / 3
  let p_other := (2 / 3) / 6
  ((1 + 2 + 3 + 4 + 5 + 6) * p_other + 7 * p7)

theorem expected_value_of_unfair_die :
  seven_sided_die_expected_value = 14 / 3 :=
by
  sorry

end expected_value_of_unfair_die_l286_286002


namespace megan_pictures_l286_286605

theorem megan_pictures (pictures_zoo pictures_museum pictures_deleted : ℕ)
  (hzoo : pictures_zoo = 15)
  (hmuseum : pictures_museum = 18)
  (hdeleted : pictures_deleted = 31) :
  (pictures_zoo + pictures_museum) - pictures_deleted = 2 :=
by
  sorry

end megan_pictures_l286_286605


namespace solve_for_a_l286_286946

def i := Complex.I

theorem solve_for_a (a : ℝ) (h : (2 + i) / (1 + a * i) = i) : a = -2 := 
by 
  sorry

end solve_for_a_l286_286946


namespace hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l286_286561

-- Definitions for the given problem
def one_hundred_million : ℕ := 100000000
def ten_million : ℕ := 10000000
def one_million : ℕ := 1000000
def ten_thousand : ℕ := 10000

-- Proving the statements
theorem hundred_million_is_ten_times_ten_million :
  one_hundred_million = 10 * ten_million :=
by
  sorry

theorem one_million_is_hundred_times_ten_thousand :
  one_million = 100 * ten_thousand :=
by
  sorry

end hundred_million_is_ten_times_ten_million_one_million_is_hundred_times_ten_thousand_l286_286561


namespace system_solution_conditions_l286_286478

theorem system_solution_conditions (α1 α2 α3 α4 : ℝ) :
  (α1 = α4 ∨ α2 = α3) ↔ 
  (∃ x1 x2 x3 x4 : ℝ,
    x1 + x2 = α1 * α2 ∧
    x1 + x3 = α1 * α3 ∧
    x1 + x4 = α1 * α4 ∧
    x2 + x3 = α2 * α3 ∧
    x2 + x4 = α2 * α4 ∧
    x3 + x4 = α3 * α4 ∧
    x1 = x2 ∧
    x2 = x3 ∧
    x1 = α2^2 / 2 ∧
    x3 = α2^2 / 2 ∧
    x4 = α2 * α4 - (α2^2 / 2) ) :=
by sorry

end system_solution_conditions_l286_286478


namespace dot_product_of_vectors_l286_286818

noncomputable def dot_product_eq : Prop :=
  ∀ (AB AC : ℝ) (BAC_deg : ℝ),
  AB = 3 → AC = 4 → BAC_deg = 30 →
  (AB * AC * (Real.cos (BAC_deg * Real.pi / 180))) = 6 * Real.sqrt 3

theorem dot_product_of_vectors :
  dot_product_eq := by
  sorry

end dot_product_of_vectors_l286_286818


namespace value_of_x_squared_add_reciprocal_squared_l286_286516

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l286_286516


namespace range_of_m_l286_286958

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define A as the set of real numbers satisfying 2x^2 - x = 0
def A : Set ℝ := {x | 2 * x^2 - x = 0}

-- Define B based on the parameter m as the set of real numbers satisfying mx^2 - mx - 1 = 0
def B (m : ℝ) : Set ℝ := {x | m * x^2 - m * x - 1 = 0}

-- Define the condition (¬U A) ∩ B = ∅
def condition (m : ℝ) : Prop := (U \ A) ∩ B m = ∅

theorem range_of_m : ∀ m : ℝ, condition m → -4 ≤ m ∧ m ≤ 0 :=
by
  sorry

end range_of_m_l286_286958


namespace units_digit_2009_2008_plus_2013_l286_286601

theorem units_digit_2009_2008_plus_2013 :
  (2009^2008 + 2013) % 10 = 4 :=
by
  sorry

end units_digit_2009_2008_plus_2013_l286_286601


namespace scientific_notation_35100_l286_286539

theorem scientific_notation_35100 : 35100 = 3.51 * 10^4 :=
by
  sorry

end scientific_notation_35100_l286_286539


namespace lioness_age_l286_286572

theorem lioness_age (H L : ℕ) 
  (h1 : L = 2 * H) 
  (h2 : (H / 2 + 5) + (L / 2 + 5) = 19) : 
  L = 12 :=
sorry

end lioness_age_l286_286572


namespace proof_m_range_l286_286806

variable {x m : ℝ}

def A (m : ℝ) : Set ℝ := {x | x^2 + x + m + 2 = 0}
def B : Set ℝ := {x | x > 0}

theorem proof_m_range (h : A m ∩ B = ∅) : m ≤ -2 := 
sorry

end proof_m_range_l286_286806


namespace smallest_term_l286_286780

theorem smallest_term (a1 d : ℕ) (h_a1 : a1 = 7) (h_d : d = 7) :
  ∃ n : ℕ, (a1 + (n - 1) * d) > 150 ∧ (a1 + (n - 1) * d) % 5 = 0 ∧
  (∀ m : ℕ, (a1 + (m - 1) * d) > 150 ∧ (a1 + (m - 1) * d) % 5 = 0 → (a1 + (m - 1) * d) ≥ (a1 + (n - 1) * d)) → a1 + (n - 1) * d = 175 :=
by
  -- We need to prove given the conditions.
  sorry

end smallest_term_l286_286780


namespace selfish_subsets_equals_fibonacci_l286_286130

noncomputable def fibonacci : ℕ → ℕ
| 0           => 0
| 1           => 1
| (n + 2)     => fibonacci (n + 1) + fibonacci n

noncomputable def selfish_subsets_count (n : ℕ) : ℕ := 
sorry -- This will be replaced with the correct recursive function

theorem selfish_subsets_equals_fibonacci (n : ℕ) : 
  selfish_subsets_count n = fibonacci n :=
sorry

end selfish_subsets_equals_fibonacci_l286_286130


namespace man_speed_l286_286757

theorem man_speed (time_in_minutes : ℝ) (distance_in_km : ℝ) (T : time_in_minutes = 24) (D : distance_in_km = 4) : 
  (distance_in_km / (time_in_minutes / 60)) = 10 := by
  sorry

end man_speed_l286_286757


namespace max_intersections_cos_circle_l286_286147

theorem max_intersections_cos_circle :
  let circle := λ x y => (x - 4)^2 + y^2 = 25
  let cos_graph := λ x => (x, Real.cos x)
  ∀ x y, (circle x y ∧ y = Real.cos x) → (∃ (p : ℕ), p ≤ 8) := sorry

end max_intersections_cos_circle_l286_286147


namespace value_of_x_squared_plus_reciprocal_squared_l286_286515

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l286_286515


namespace original_fraction_eq_two_thirds_l286_286584

theorem original_fraction_eq_two_thirds (a b : ℕ) (h : (a^3 : ℚ) / (b + 3) = 2 * (a / b)) : a = 2 ∧ b = 3 :=
by {
  sorry
}

end original_fraction_eq_two_thirds_l286_286584


namespace trigonometric_identity_l286_286746

theorem trigonometric_identity (α : ℝ) :
    1 - 1/4 * (Real.sin (2 * α)) ^ 2 + Real.cos (2 * α) = (Real.cos α) ^ 2 + (Real.cos α) ^ 4 :=
by
  sorry

end trigonometric_identity_l286_286746


namespace area_of_triangle_ABC_l286_286853

theorem area_of_triangle_ABC (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let x1 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let y_vertex := (4 * a * c - b^2) / (4 * a)
  0.5 * (|x2 - x1|) * |y_vertex| = (b^2 - 4 * a * c) * Real.sqrt (b^2 - 4 * a * c) / (8 * a^2) :=
sorry

end area_of_triangle_ABC_l286_286853


namespace sally_balloon_count_l286_286078

theorem sally_balloon_count (n_initial : ℕ) (n_lost : ℕ) (n_final : ℕ) 
  (h_initial : n_initial = 9) 
  (h_lost : n_lost = 2) 
  (h_final : n_final = n_initial - n_lost) : 
  n_final = 7 :=
by
  sorry

end sally_balloon_count_l286_286078


namespace bullet_trains_crossing_time_l286_286111

theorem bullet_trains_crossing_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (speed1 speed2 : ℝ)
  (relative_speed : ℝ)
  (total_distance : ℝ)
  (cross_time : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 10)
  (h_time2 : time2 = 20)
  (h_speed1 : speed1 = length / time1)
  (h_speed2 : speed2 = length / time2)
  (h_relative_speed : relative_speed = speed1 + speed2)
  (h_total_distance : total_distance = length + length)
  (h_cross_time : cross_time = total_distance / relative_speed) :
  cross_time = 240 / 18 := 
by
  sorry

end bullet_trains_crossing_time_l286_286111


namespace arithmetic_sequence_example_l286_286201

variable {α : Type*} [AddGroup α] [Module ℤ α]

noncomputable def a : ℕ → α
| 0 => 2          -- since a_1 = 2
| 1 => a(0) + d   -- a_2 = a_1 + d
| (n + 1) => a n + d

theorem arithmetic_sequence_example (d : α) (h : a(1) + a(2) = 13) : a(4) = 14 := 
by
  -- the proof will go here
  sorry

end arithmetic_sequence_example_l286_286201


namespace option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l286_286427

theorem option_A_incorrect (a : ℝ) : (a^2) * (a^3) ≠ a^6 :=
by sorry

theorem option_B_incorrect (a : ℝ) : (a^2)^3 ≠ a^5 :=
by sorry

theorem option_C_incorrect (a : ℝ) : (a^6) / (a^2) ≠ a^3 :=
by sorry

theorem option_D_correct (a b : ℝ) : (a + 2 * b) * (a - 2 * b) = a^2 - 4 * b^2 :=
by sorry

end option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l286_286427


namespace units_digit_of_expression_l286_286602

def units_digit (n : ℕ) : ℕ := n % 10

noncomputable def expression := (20 * 21 * 22 * 23 * 24 * 25) / 1000

theorem units_digit_of_expression : units_digit (expression) = 2 :=
by
  sorry

end units_digit_of_expression_l286_286602


namespace find_salary_for_january_l286_286846

-- Definitions based on problem conditions
variables (J F M A May : ℝ)
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (hMay : May = 6500)

-- Lean statement
theorem find_salary_for_january : J = 5700 :=
by {
  sorry
}

end find_salary_for_january_l286_286846


namespace temple_run_red_coins_l286_286824

variables (x y z : ℕ)

theorem temple_run_red_coins :
  x + y + z = 2800 →
  x + 3 * y + 5 * z = 7800 →
  z = y + 200 →
  y = 700 := 
by 
  intro h1 h2 h3
  sorry

end temple_run_red_coins_l286_286824


namespace max_quarters_l286_286567

theorem max_quarters (q : ℕ) (h1 : q + q + q / 2 = 20): q ≤ 11 :=
by
  sorry

end max_quarters_l286_286567


namespace wheel_speed_is_12_mph_l286_286370

theorem wheel_speed_is_12_mph
  (r : ℝ) -- speed in miles per hour
  (C : ℝ := 15 / 5280) -- circumference in miles
  (H1 : ∃ t, r * t = C * 3600) -- initial condition that speed times time for one rotation equals 15/5280 miles in seconds
  (H2 : ∃ t, (r + 7) * (t - 1/21600) = C * 3600) -- condition that speed increases by 7 mph when time shortens by 1/6 second
  : r = 12 :=
sorry

end wheel_speed_is_12_mph_l286_286370


namespace cone_curved_surface_area_at_5_seconds_l286_286616

theorem cone_curved_surface_area_at_5_seconds :
  let l := λ t : ℝ => 10 + 2 * t
  let r := λ t : ℝ => 5 + 1 * t
  let CSA := λ t : ℝ => Real.pi * r t * l t
  CSA 5 = 160 * Real.pi :=
by
  -- Definitions and calculations in the problem ensure this statement
  sorry

end cone_curved_surface_area_at_5_seconds_l286_286616


namespace iggy_total_hours_l286_286195

-- Define the conditions
def miles_run_per_day : ℕ → ℕ
| 0 := 3 -- Monday
| 1 := 4 -- Tuesday
| 2 := 6 -- Wednesday
| 3 := 8 -- Thursday
| 4 := 3 -- Friday
| _ := 0 -- Other days

def total_distance : ℕ := List.sum (List.ofFn miles_run_per_day 5)
def miles_per_minute : ℕ := 10
def minutes_per_hour : ℕ := 60
def total_minutes_run : ℕ := total_distance * miles_per_minute
def total_hours_run : ℕ := total_minutes_run / minutes_per_hour

-- The statement to prove
theorem iggy_total_hours :
  total_hours_run = 4 :=
sorry

end iggy_total_hours_l286_286195


namespace probability_of_selecting_red_books_is_3_div_14_l286_286736

-- Define the conditions
def total_books : ℕ := 8
def red_books : ℕ := 4
def blue_books : ℕ := 4
def books_selected : ℕ := 2

-- Define the calculation of the probability
def probability_red_books_selected : ℚ :=
  let total_outcomes := Nat.choose total_books books_selected
  let favorable_outcomes := Nat.choose red_books books_selected
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- State the theorem
theorem probability_of_selecting_red_books_is_3_div_14 :
  probability_red_books_selected = 3 / 14 :=
by
  sorry

end probability_of_selecting_red_books_is_3_div_14_l286_286736


namespace nth_odd_positive_integer_is_199_l286_286416

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l286_286416


namespace triangle_tangent_half_angle_l286_286542

theorem triangle_tangent_half_angle (a b c : ℝ) (A : ℝ) (C : ℝ)
  (h : a + c = 2 * b) :
  Real.tan (A / 2) * Real.tan (C / 2) = 1 / 3 := 
sorry

end triangle_tangent_half_angle_l286_286542


namespace sum_solutions_eq_l286_286599

theorem sum_solutions_eq : 
  let a := 12
  let b := -19
  let c := -21
  (4 * x + 3) * (3 * x - 7) = 0 → (b/a) = 19/12 :=
by
  sorry

end sum_solutions_eq_l286_286599


namespace value_of_x_squared_add_reciprocal_squared_l286_286517

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l286_286517


namespace min_value_expr_l286_286679

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x + 2 * y = 5) :
  (1 / (x - 1) + 1 / (y - 1)) = (3 / 2 + Real.sqrt 2) :=
sorry

end min_value_expr_l286_286679


namespace number_of_puppies_with_4_spots_is_3_l286_286617

noncomputable def total_puppies : Nat := 10
noncomputable def puppies_with_5_spots : Nat := 6
noncomputable def puppies_with_2_spots : Nat := 1
noncomputable def puppies_with_4_spots : Nat := total_puppies - puppies_with_5_spots - puppies_with_2_spots

theorem number_of_puppies_with_4_spots_is_3 :
  puppies_with_4_spots = 3 := 
sorry

end number_of_puppies_with_4_spots_is_3_l286_286617


namespace third_quadrant_angle_bisector_l286_286950

theorem third_quadrant_angle_bisector
  (a b : ℝ)
  (hA : A = (-4,a))
  (hB : B = (-2,b))
  (h_lineA : a = -4)
  (h_lineB : b = -2)
  : a + b + a * b = 2 :=
by
  sorry

end third_quadrant_angle_bisector_l286_286950


namespace count_positive_solutions_of_eq_l286_286507

theorem count_positive_solutions_of_eq : 
  (∃ x : ℝ, x^2 = -6 * x + 9 ∧ x > 0) ∧ (¬ ∃ y : ℝ, y^2 = -6 * y + 9 ∧ y > 0 ∧ y ≠ -3 + 3 * Real.sqrt 2) :=
sorry

end count_positive_solutions_of_eq_l286_286507


namespace parabola_directrix_l286_286952

theorem parabola_directrix (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x ↔ x = -1 → p = 2) :=
by
  sorry

end parabola_directrix_l286_286952


namespace sum_of_solutions_eq_neg_six_l286_286781

theorem sum_of_solutions_eq_neg_six (x r s : ℝ) :
  (81 : ℝ) - 18 * x - 3 * x^2 = 0 →
  (r + s = -6) :=
by
  sorry

end sum_of_solutions_eq_neg_six_l286_286781


namespace fraction_of_students_who_walk_home_l286_286819

theorem fraction_of_students_who_walk_home :
  let busFraction := 1 / 3
  let carpoolFraction := 1 / 5
  let scooterFraction := 1 / 8
  -- Calculate total fraction of students who use the bus, carpool, or scooter
  let totalNonWalkingFraction := busFraction + carpoolFraction + scooterFraction
  -- Find the fraction of students who walk home
  let walkingFraction := 1 - totalNonWalkingFraction
  walkingFraction = 41 / 120 :=
by
  -- Define the fractions
  let busFraction := 1 / 3
  let carpoolFraction := 1 / 5
  let scooterFraction := 1 / 8

  -- Calculate totalNonWalkingFraction
  have h1 : totalNonWalkingFraction = busFraction + carpoolFraction + scooterFraction := rfl
  have h2 : totalNonWalkingFraction = 1 / 3 + 1 / 5 + 1 / 8 := by rw [h1]
  
  -- Use common denominators
  have h3 : 1 / 3 = 40 / 120 := by norm_num
  have h4 : 1 / 5 = 24 / 120 := by norm_num
  have h5 : 1 / 8 = 15 / 120 := by norm_num
  have h6 : totalNonWalkingFraction = 40 / 120 + 24 / 120 + 15 / 120 := by rw [h2, h3, h4, h5]
  
  -- Add the fractions
  have h7 : totalNonWalkingFraction = 79 / 120 := by norm_num
  
  -- Calculate walkingFraction
  have h8 : walkingFraction = 1 - (79 / 120) := by rw h7
  have h9 : walkingFraction = 120 / 120 - 79 / 120 := by rw h8
  
  -- Simplify to find the desired fraction
  have h10 : walkingFraction = 41 / 120 := by norm_num
  
  -- Completed proof
  exact h10

end fraction_of_students_who_walk_home_l286_286819


namespace round_table_arrangement_l286_286536

theorem round_table_arrangement :
  ∀ (n : ℕ), n = 10 → (∃ factorial_value : ℕ, factorial_value = Nat.factorial (n - 1) ∧ factorial_value = 362880) := by
  sorry

end round_table_arrangement_l286_286536


namespace min_value_l286_286172

open Real

noncomputable def y1 (x1 : ℝ) : ℝ := x1 * log x1
noncomputable def y2 (x2 : ℝ) : ℝ := x2 - 3

theorem min_value :
  ∃ (x1 x2 : ℝ), (x1 - x2)^2 + (y1 x1 - y2 x2)^2 = 2 :=
by
  sorry

end min_value_l286_286172


namespace equivalent_function_l286_286745

theorem equivalent_function :
  (∀ x : ℝ, (76 * x ^ 6) ^ 7 = |x|) :=
by
  sorry

end equivalent_function_l286_286745


namespace problem_statement_l286_286628

noncomputable def square : ℝ := sorry -- We define a placeholder
noncomputable def pentagon : ℝ := sorry -- We define a placeholder

axiom eq1 : 2 * square + 4 * pentagon = 25
axiom eq2 : 3 * square + 3 * pentagon = 22

theorem problem_statement : 4 * pentagon = 20.67 := 
by
  sorry

end problem_statement_l286_286628


namespace remainder_of_sum_l286_286012

theorem remainder_of_sum :
  (85 + 86 + 87 + 88 + 89 + 90 + 91 + 92) % 20 = 18 :=
by
  sorry

end remainder_of_sum_l286_286012


namespace bus_capacity_l286_286260

theorem bus_capacity :
  ∀ (left_seats right_seats people_per_seat back_seat : ℕ),
  left_seats = 15 →
  right_seats = left_seats - 3 →
  people_per_seat = 3 →
  back_seat = 11 →
  (left_seats * people_per_seat) + 
  (right_seats * people_per_seat) + 
  back_seat = 92 := by
  intros left_seats right_seats people_per_seat back_seat 
  intros h1 h2 h3 h4 
  sorry

end bus_capacity_l286_286260


namespace possible_k_value_l286_286693

theorem possible_k_value (a n k : ℕ) (h1 : n > 1) (h2 : 10^(n-1) ≤ a ∧ a < 10^n)
    (h3 : b = a * (10^n + 1)) (h4 : k = b / a^2) (h5 : b = a * 10 ^n + a) :
  k = 7 := 
sorry

end possible_k_value_l286_286693


namespace sparrow_swallow_equations_l286_286611

theorem sparrow_swallow_equations (x y : ℝ) : 
  (5 * x + 6 * y = 16) ∧ (4 * x + y = 5 * y + x) :=
  sorry

end sparrow_swallow_equations_l286_286611


namespace vertex_of_parabola_l286_286578

theorem vertex_of_parabola : 
  ∀ (x y : ℝ), (y = -x^2 + 3) → (0, 3) ∈ {(h, k) | ∃ (a : ℝ), y = a * (x - h)^2 + k} :=
by
  sorry

end vertex_of_parabola_l286_286578


namespace mother_age_when_harry_born_l286_286963

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end mother_age_when_harry_born_l286_286963


namespace parallel_lines_slope_condition_l286_286186

theorem parallel_lines_slope_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x + (m + 1) * y + 4 = 0 → mx + 3 * y - 2 = 0) →
  (m = 2 ∨ m = -3) :=
by
  sorry

end parallel_lines_slope_condition_l286_286186


namespace find_m_value_l286_286368

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b : V)
variables (m : ℝ)
variables (A B C D : V)

-- Assuming vectors a and b are non-collinear
axiom non_collinear (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (∃ (k : ℝ), a = k • b)

-- Given vectors
axiom hAB : B - A = 9 • a + m • b
axiom hBC : C - B = -2 • a - 1 • b
axiom hDC : C - D = a - 2 • b

-- Collinearity condition for A, B, and D
axiom collinear (k : ℝ) : B - A = k • (B - D)

theorem find_m_value : m = -3 :=
by sorry

end find_m_value_l286_286368


namespace corvette_trip_time_percentage_increase_l286_286555

theorem corvette_trip_time_percentage_increase
  (total_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (first_half_distance second_half_distance first_half_time second_half_time total_time : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : average_speed = 40)
  (h4 : first_half_distance = total_distance / 2)
  (h5 : second_half_distance = total_distance / 2)
  (h6 : first_half_time = first_half_distance / first_half_speed)
  (h7 : total_time = total_distance / average_speed)
  (h8 : second_half_time = total_time - first_half_time) :
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := sorry

end corvette_trip_time_percentage_increase_l286_286555


namespace income_calculation_l286_286235

theorem income_calculation (savings : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (ratio_condition : income_ratio = 5 ∧ expenditure_ratio = 4) (savings_condition : savings = 3800) :
  income_ratio * savings / (income_ratio - expenditure_ratio) = 19000 :=
by
  sorry

end income_calculation_l286_286235


namespace range_of_a_l286_286038

noncomputable def f (x a : ℝ) : ℝ := x^2 * Real.exp x - a

theorem range_of_a 
 (h : ∃ a, (∀ x₀ x₁ x₂, x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₁ ≠ x₂ ∧ f x₀ a = 0 ∧ f x₁ a = 0 ∧ f x₂ a = 0)) :
  ∃ a, 0 < a ∧ a < 4 / Real.exp 2 :=
by
  sorry

end range_of_a_l286_286038


namespace problem_statement_l286_286508

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l286_286508


namespace complex_pure_imaginary_l286_286345

theorem complex_pure_imaginary (a : ℝ) : 
  ((a^2 - 3*a + 2) = 0) → (a = 2) := 
  by 
  sorry

end complex_pure_imaginary_l286_286345


namespace combined_percent_increase_proof_l286_286772

variable (initial_stock_A_price : ℝ := 25)
variable (initial_stock_B_price : ℝ := 45)
variable (initial_stock_C_price : ℝ := 60)
variable (final_stock_A_price : ℝ := 28)
variable (final_stock_B_price : ℝ := 50)
variable (final_stock_C_price : ℝ := 75)

noncomputable def percent_increase (initial final : ℝ) : ℝ :=
  ((final - initial) / initial) * 100

noncomputable def combined_percent_increase (initial_a initial_b initial_c final_a final_b final_c : ℝ) : ℝ :=
  (percent_increase initial_a final_a + percent_increase initial_b final_b + percent_increase initial_c final_c) / 3

theorem combined_percent_increase_proof :
  combined_percent_increase initial_stock_A_price initial_stock_B_price initial_stock_C_price
                            final_stock_A_price final_stock_B_price final_stock_C_price = 16.04 := by
  sorry

end combined_percent_increase_proof_l286_286772


namespace total_bill_is_95_l286_286769

noncomputable def total_bill := 28 + 8 + 10 + 6 + 14 + 11 + 12 + 6

theorem total_bill_is_95 : total_bill = 95 := by
  sorry

end total_bill_is_95_l286_286769


namespace initial_investment_l286_286770

noncomputable def compound_interest_inv (A r : ℝ) (n t : ℕ) : ℝ :=
  A / ((1 + r / n) ^ (n * t))

theorem initial_investment :
  compound_interest_inv 7372.46 0.065 1 2 ≈ 6510.00 := by 
  sorry

end initial_investment_l286_286770


namespace Mia_and_dad_time_to_organize_toys_l286_286994

theorem Mia_and_dad_time_to_organize_toys :
  let total_toys := 60
  let dad_add_rate := 6
  let mia_remove_rate := 4
  let net_gain_per_cycle := dad_add_rate - mia_remove_rate
  let seconds_per_cycle := 30
  let total_needed_cycles := (total_toys - 2) / net_gain_per_cycle -- 58 toys by the end of repeated cycles, 2 is to ensure dad's last placement
  let last_cycle_time := seconds_per_cycle
  let total_time_seconds := total_needed_cycles * seconds_per_cycle + last_cycle_time
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 15 :=
by
  sorry

end Mia_and_dad_time_to_organize_toys_l286_286994


namespace birth_age_of_mother_l286_286961

def harrys_age : ℕ := 50

def fathers_age (h : ℕ) : ℕ := h + 24

def mothers_age (f h : ℕ) : ℕ := f - h / 25

theorem birth_age_of_mother (h f m : ℕ) (H1 : h = harrys_age)
  (H2 : f = fathers_age h) (H3 : m = mothers_age f h) :
  m - h = 22 := sorry

end birth_age_of_mother_l286_286961


namespace distance_behind_l286_286106

-- Given conditions
variables {A B E : ℝ} -- Speed of Anusha, Banu, and Esha
variables {Da Db De : ℝ} -- distances covered by Anusha, Banu, and Esha

axiom const_speeds : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)

-- The proof to be established
theorem distance_behind (h : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)) :
  100 - De = 19 :=
by sorry

end distance_behind_l286_286106


namespace quadratic_root_properties_l286_286972

theorem quadratic_root_properties (b : ℝ) (t : ℝ) :
  (∀ x : ℝ, x^2 + b*x - 2 = 0 → (x = 2 ∨ x = t)) →
  b = -1 ∧ t = -1 :=
by
  sorry

end quadratic_root_properties_l286_286972


namespace max_rectangles_3x5_in_17x22_l286_286178

theorem max_rectangles_3x5_in_17x22 : ∃ n : ℕ, n = 24 ∧ 
  (∀ (cut_3x5_pieces : ℤ), cut_3x5_pieces ≤ n) :=
by
  sorry

end max_rectangles_3x5_in_17x22_l286_286178


namespace math_problem_l286_286030

open Set

noncomputable def A : Set ℝ := { x | x < 1 }
noncomputable def B : Set ℝ := { x | x * (x - 1) > 6 }
noncomputable def C (m : ℝ) : Set ℝ := { x | -1 + m < x ∧ x < 2 * m }

theorem math_problem (m : ℝ) (m_range : C m ≠ ∅) :
  (A ∪ B = { x | x > 3 ∨ x < 1 }) ∧
  (A ∩ (compl B) = { x | -2 ≤ x ∧ x < 1 }) ∧
  (-1 < m ∧ m ≤ 0.5) :=
by
  sorry

end math_problem_l286_286030


namespace hundredth_odd_integer_l286_286413

theorem hundredth_odd_integer : ∃ (x : ℕ), 2 * x - 1 = 199 ∧ x = 100 :=
by
  use 100
  split
  . exact calc
      2 * 100 - 1 = 200 - 1 : by ring
      _ = 199 : by norm_num
  . refl

end hundredth_odd_integer_l286_286413


namespace smallest_nonfactor_product_of_48_l286_286739

noncomputable def is_factor_of (a b : ℕ) : Prop :=
  b % a = 0

theorem smallest_nonfactor_product_of_48
  (m n : ℕ)
  (h1 : m ≠ n)
  (h2 : is_factor_of m 48)
  (h3 : is_factor_of n 48)
  (h4 : ¬is_factor_of (m * n) 48) :
  m * n = 18 :=
sorry

end smallest_nonfactor_product_of_48_l286_286739


namespace fraction_of_original_price_l286_286104

theorem fraction_of_original_price
  (CP SP : ℝ)
  (h1 : SP = 1.275 * CP)
  (f: ℝ)
  (h2 : f * SP = 0.85 * CP)
  : f = 17 / 25 :=
by
  sorry

end fraction_of_original_price_l286_286104


namespace total_parents_surveyed_l286_286766

-- Define the given conditions
def percent_agree : ℝ := 0.20
def percent_disagree : ℝ := 0.80
def disagreeing_parents : ℕ := 640

-- Define the statement to prove
theorem total_parents_surveyed :
  ∃ (total_parents : ℕ), disagreeing_parents = (percent_disagree * total_parents) ∧ total_parents = 800 :=
by
  sorry

end total_parents_surveyed_l286_286766


namespace lines_parallel_if_perpendicular_to_same_plane_l286_286487

variables (m n : Line) (α : Plane)

-- Define conditions using Lean's logical constructs
def perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry -- This would define the condition
def parallel_lines (l1 l2 : Line) : Prop := sorry -- This would define the condition

-- The statement to prove
theorem lines_parallel_if_perpendicular_to_same_plane 
  (h1 : perpendicular_to_plane m α) 
  (h2 : perpendicular_to_plane n α) : 
  parallel_lines m n :=
sorry

end lines_parallel_if_perpendicular_to_same_plane_l286_286487


namespace initial_water_percentage_l286_286121

noncomputable def initial_percentage_of_water : ℚ :=
  20

theorem initial_water_percentage
  (initial_volume : ℚ := 125)
  (added_water : ℚ := 8.333333333333334)
  (final_volume : ℚ := initial_volume + added_water)
  (desired_percentage : ℚ := 25)
  (desired_amount_of_water : ℚ := desired_percentage / 100 * final_volume)
  (initial_amount_of_water : ℚ := desired_amount_of_water - added_water) :
  (initial_amount_of_water / initial_volume * 100 = initial_percentage_of_water) :=
by
  sorry

end initial_water_percentage_l286_286121


namespace reaction_spontaneous_at_high_temperature_l286_286017

theorem reaction_spontaneous_at_high_temperature
  (ΔH : ℝ) (ΔS : ℝ) (T : ℝ) (ΔG : ℝ)
  (h_ΔH_pos : ΔH > 0)
  (h_ΔS_pos : ΔS > 0)
  (h_ΔG_eq : ΔG = ΔH - T * ΔS) :
  (∃ T_high : ℝ, T_high > 0 ∧ ΔG < 0) := sorry

end reaction_spontaneous_at_high_temperature_l286_286017


namespace speed_of_stream_l286_286607

-- Define the problem conditions
def downstream_distance := 100 -- distance in km
def downstream_time := 8 -- time in hours
def upstream_distance := 75 -- distance in km
def upstream_time := 15 -- time in hours

-- Define the constants
def total_distance (B S : ℝ) := downstream_distance = (B + S) * downstream_time
def total_time (B S : ℝ) := upstream_distance = (B - S) * upstream_time

-- Stating the main theorem to be proved
theorem speed_of_stream (B S : ℝ) (h1 : total_distance B S) (h2 : total_time B S) : S = 3.75 := by
  sorry

end speed_of_stream_l286_286607


namespace largest_multiple_of_7_less_than_neg50_l286_286100

theorem largest_multiple_of_7_less_than_neg50 : ∃ x, (∃ k : ℤ, x = 7 * k) ∧ x < -50 ∧ ∀ y, (∃ m : ℤ, y = 7 * m) → y < -50 → y ≤ x :=
sorry

end largest_multiple_of_7_less_than_neg50_l286_286100


namespace q_sufficient_not_necessary_p_l286_286311

theorem q_sufficient_not_necessary_p (x : ℝ) (p : Prop) (q : Prop) :
  (p ↔ |x| < 2) →
  (q ↔ x^2 - x - 2 < 0) →
  (q → p) ∧ (p ∧ ¬q) :=
by
  sorry

end q_sufficient_not_necessary_p_l286_286311


namespace total_coins_constant_l286_286145

-- Definitions based on the conditions
def stack1 := 12
def stack2 := 17
def stack3 := 23
def stack4 := 8

def totalCoins := stack1 + stack2 + stack3 + stack4 -- 60 coins
def is_divisor (x: ℕ) := x ∣ totalCoins

-- The theorem statement
theorem total_coins_constant {x: ℕ} (h: is_divisor x) : totalCoins = 60 :=
by
  -- skip the proof steps
  sorry

end total_coins_constant_l286_286145


namespace inequality_and_equality_condition_l286_286075

theorem inequality_and_equality_condition (a b : ℝ) :
  a^2 + 4 * b^2 + 4 * b - 4 * a + 5 ≥ 0 ∧ (a^2 + 4 * b^2 + 4 * b - 4 * a + 5 = 0 ↔ (a = 2 ∧ b = -1 / 2)) :=
by
  sorry

end inequality_and_equality_condition_l286_286075


namespace max_correct_answers_l286_286349

variables {a b c : ℕ} -- Define a, b, and c as natural numbers

theorem max_correct_answers : 
  ∀ a b c : ℕ, (a + b + c = 50) → (5 * a - 2 * c = 150) → a ≤ 35 :=
by
  -- Proof steps can be skipped by adding sorry
  sorry

end max_correct_answers_l286_286349


namespace point_M_coordinates_l286_286971

theorem point_M_coordinates (a : ℤ) (h : a + 3 = 0) : (a + 3, 2 * a - 2) = (0, -8) :=
by
  sorry

end point_M_coordinates_l286_286971


namespace rectangle_dimensions_l286_286098

theorem rectangle_dimensions (l w : ℝ) : 
  (∃ x : ℝ, x = l - 3 ∧ x = w - 2 ∧ x^2 = (1 / 2) * l * w) → (l = 9 ∧ w = 8) :=
by
  sorry

end rectangle_dimensions_l286_286098


namespace neighborhood_has_exactly_one_item_l286_286044

noncomputable def neighborhood_conditions : Prop :=
  let total_households := 120
  let households_no_items := 15
  let households_car_and_bike := 28
  let households_car := 52
  let households_bike := 32
  let households_scooter := 18
  let households_skateboard := 8
  let households_at_least_one_item := total_households - households_no_items
  let households_car_only := households_car - households_car_and_bike
  let households_bike_only := households_bike - households_car_and_bike
  let households_exactly_one_item := households_car_only + households_bike_only + households_scooter + households_skateboard
  households_at_least_one_item = 105 ∧ households_exactly_one_item = 54

theorem neighborhood_has_exactly_one_item :
  neighborhood_conditions :=
by
  -- Proof goes here
  sorry

end neighborhood_has_exactly_one_item_l286_286044


namespace probability_at_most_one_A_B_selected_l286_286199

def total_employees : ℕ := 36
def ratio_3_2_1 : (ℕ × ℕ × ℕ) := (3, 2, 1)
def sample_size : ℕ := 12
def youth_group_size : ℕ := 6
def total_combinations_youth : ℕ := Nat.choose 6 2
def event_complementary : ℕ := Nat.choose 2 2

theorem probability_at_most_one_A_B_selected :
  let prob := 1 - event_complementary / total_combinations_youth
  prob = (14 : ℚ) / 15 := sorry

end probability_at_most_one_A_B_selected_l286_286199


namespace expression_equiv_l286_286596

theorem expression_equiv :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end expression_equiv_l286_286596


namespace intersection_proof_l286_286051

noncomputable def M : Set ℕ := {1, 3, 5, 7, 9}

noncomputable def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_proof : M ∩ (N ∩ Set.univ) = {5, 7, 9} :=
by sorry

end intersection_proof_l286_286051


namespace convert_decimal_to_fraction_l286_286882

theorem convert_decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by
  sorry

end convert_decimal_to_fraction_l286_286882


namespace range_of_a_l286_286167

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ a → |x - 1| < 1) → (∃ x : ℝ, |x - 1| < 1 ∧ x < a) → a ≤ 0 := 
sorry

end range_of_a_l286_286167


namespace total_cost_of_items_l286_286386

theorem total_cost_of_items (cost_of_soda : ℕ) (cost_of_soup : ℕ) (cost_of_sandwich : ℕ) (total_cost : ℕ) 
  (h1 : cost_of_soda = 1)
  (h2 : cost_of_soup = 3 * cost_of_soda)
  (h3 : cost_of_sandwich = 3 * cost_of_soup) :
  total_cost = 3 * cost_of_soda + 2 * cost_of_soup + cost_of_sandwich :=
by
  unfold total_cost
  show 3 * 1 + 2 * (3 * 1) + (3 * (3 * 1)) = 18
  rfl

end total_cost_of_items_l286_286386


namespace combined_loss_percentage_l286_286765

theorem combined_loss_percentage
  (cost_price_radio : ℕ := 8000)
  (quantity_radio : ℕ := 5)
  (discount_radio : ℚ := 0.1)
  (tax_radio : ℚ := 0.06)
  (sale_price_radio : ℕ := 7200)
  (cost_price_tv : ℕ := 20000)
  (quantity_tv : ℕ := 3)
  (discount_tv : ℚ := 0.15)
  (tax_tv : ℚ := 0.07)
  (sale_price_tv : ℕ := 18000)
  (cost_price_phone : ℕ := 15000)
  (quantity_phone : ℕ := 4)
  (discount_phone : ℚ := 0.08)
  (tax_phone : ℚ := 0.05)
  (sale_price_phone : ℕ := 14500) :
  let total_cost_price := (quantity_radio * cost_price_radio) + (quantity_tv * cost_price_tv) + (quantity_phone * cost_price_phone)
  let total_sale_price := (quantity_radio * sale_price_radio) + (quantity_tv * sale_price_tv) + (quantity_phone * sale_price_phone)
  let total_loss := total_cost_price - total_sale_price
  let loss_percentage := (total_loss * 100 : ℚ) / total_cost_price
  loss_percentage = 7.5 :=
by
  sorry

end combined_loss_percentage_l286_286765


namespace equation_solution_l286_286296

theorem equation_solution (x : ℝ) (h : (2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2)) : x = 9 :=
by
  sorry

end equation_solution_l286_286296


namespace min_handshakes_l286_286436

theorem min_handshakes (n : ℕ) (h1 : n = 25) 
  (h2 : ∀ (p : ℕ), p < n → ∃ q r : ℕ, q ≠ r ∧ q < n ∧ r < n ∧ q ≠ p ∧ r ≠ p) 
  (h3 : ∃ a b c : ℕ, a < n ∧ b < n ∧ c < n ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ ¬(∃ d : ℕ, (d = a ∨ d = b ∨ d = c) ∧ (¬(a = d ∨ b = d ∨ c = d)) ∧ d < n)) :
  ∃ m : ℕ, m = 28 :=
by
  sorry

end min_handshakes_l286_286436


namespace paint_houses_l286_286970

theorem paint_houses (time_per_house : ℕ) (hour_to_minute : ℕ) (hours_available : ℕ) 
  (h1 : time_per_house = 20) (h2 : hour_to_minute = 60) (h3 : hours_available = 3) :
  (hours_available * hour_to_minute) / time_per_house = 9 :=
by
  sorry

end paint_houses_l286_286970


namespace root_condition_l286_286893

noncomputable def f (x t : ℝ) := x^2 + t * x - t

theorem root_condition {t : ℝ} : (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) := 
  sorry

end root_condition_l286_286893


namespace mk_div_km_l286_286664

theorem mk_div_km 
  (m n k : ℕ) 
  (hm : 0 < m) 
  (hn : 0 < n) 
  (hk : 0 < k) 
  (h1 : m^n ∣ n^m) 
  (h2 : n^k ∣ k^n) : 
  m^k ∣ k^m := 
  sorry

end mk_div_km_l286_286664


namespace smallest_sum_of_consecutive_integers_is_square_l286_286861

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end smallest_sum_of_consecutive_integers_is_square_l286_286861


namespace no_integer_solutions_l286_286302

theorem no_integer_solutions (x y : ℤ) : 19 * x^3 - 84 * y^2 ≠ 1984 :=
by
  sorry

end no_integer_solutions_l286_286302


namespace initial_workers_l286_286570

theorem initial_workers (W : ℕ) (work1 : ℕ) (work2 : ℕ) :
  (work1 = W * 8 * 30) →
  (work2 = (W + 35) * 6 * 40) →
  (work1 / 30 = work2 / 40) →
  W = 105 :=
by
  intros hwork1 hwork2 hprop
  sorry

end initial_workers_l286_286570


namespace area_ratio_l286_286778

theorem area_ratio (l w h : ℝ) (h1 : w * h = 288) (h2 : l * w = 432) (h3 : l * w * h = 5184) :
  (l * h) / (l * w) = 1 / 2 :=
sorry

end area_ratio_l286_286778


namespace monotonicity_decreasing_range_l286_286730

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem monotonicity_decreasing_range (ω : ℝ) :
  (∀ x y : ℝ, (π / 2 < x ∧ x < π ∧ π / 2 < y ∧ y < π ∧ x < y) → f ω x > f ω y) ↔ (1 / 2 ≤ ω ∧ ω ≤ 5 / 4) :=
sorry

end monotonicity_decreasing_range_l286_286730


namespace conditional_probability_P_B_given_A_l286_286483

open Classical

noncomputable def S : Finset ℕ := {1, 2, 3, 4, 5}

def eventA : Set (ℕ × ℕ) := {p | p.1 + p.2 ∈ S ∧ (p.1 + p.2) % 2 = 0 ∧ p.1 < p.2}
def eventB : Set (ℕ × ℕ) := {p | p.1 ∈ {2, 4} ∧ p.2 ∈ {2, 4} ∧ p.1 < p.2}

def P_eventA : ℝ := ↑((3.choose 2 + 2.choose 2) : ℕ) / (5.choose 2)
def P_eventB : ℝ := ↑((2.choose 2) : ℕ) / (5.choose 2)

theorem conditional_probability_P_B_given_A : 
  P (eventB) * P (eventA) =
  (2.choose 2 : ℝ) * (5.choose 2) :=
sorry

end conditional_probability_P_B_given_A_l286_286483


namespace cube_volume_proof_l286_286353

-- Define the conditions
def len_inch : ℕ := 48
def width_inch : ℕ := 72
def total_surface_area_inch : ℕ := len_inch * width_inch
def num_faces : ℕ := 6
def area_one_face_inch : ℕ := total_surface_area_inch / num_faces
def inches_to_feet (length_in_inches : ℕ) : ℕ := length_in_inches / 12

-- Define the key elements of the proof problem
def side_length_inch : ℕ := Int.natAbs (Nat.sqrt area_one_face_inch)
def side_length_ft : ℕ := inches_to_feet side_length_inch
def volume_ft3 : ℕ := side_length_ft ^ 3

-- State the proof problem
theorem cube_volume_proof : volume_ft3 = 8 := by
  -- The proof would be implemented here
  sorry

end cube_volume_proof_l286_286353


namespace Mark_water_balloon_spending_l286_286697

theorem Mark_water_balloon_spending :
  let budget := 24
  let small_bag_cost := 4
  let small_bag_balloons := 50
  let medium_bag_balloons := 75
  let extra_large_bag_cost := 12
  let extra_large_bag_balloons := 200
  let total_balloons := 400
  (2 * extra_large_bag_balloons = total_balloons) → (2 * extra_large_bag_cost = budget) :=
by
  intros
  sorry

end Mark_water_balloon_spending_l286_286697


namespace jerome_contact_list_l286_286980

def classmates := 20
def out_of_school_friends := classmates / 2
def family_members := 3
def total_contacts := classmates + out_of_school_friends + family_members

theorem jerome_contact_list : total_contacts = 33 := by
  sorry

end jerome_contact_list_l286_286980


namespace currency_exchange_rate_l286_286753

theorem currency_exchange_rate (b g x : ℕ) (h1 : 1 * b * g = b * g) (h2 : 1 = 1) :
  (b + g) ^ 2 + 1 = b * g * x → x = 5 :=
sorry

end currency_exchange_rate_l286_286753


namespace mother_to_grandfather_age_ratio_l286_286380

theorem mother_to_grandfather_age_ratio
  (rachel_age : ℕ)
  (grandfather_ratio : ℕ)
  (father_mother_gap : ℕ) 
  (future_rachel_age: ℕ) 
  (future_father_age : ℕ)
  (current_father_age current_mother_age current_grandfather_age : ℕ) 
  (h1 : rachel_age = 12)
  (h2 : grandfather_ratio = 7)
  (h3 : father_mother_gap = 5)
  (h4 : future_rachel_age = 25)
  (h5 : future_father_age = 60)
  (h6 : current_father_age = future_father_age - (future_rachel_age - rachel_age))
  (h7 : current_mother_age = current_father_age - father_mother_gap)
  (h8 : current_grandfather_age = grandfather_ratio * rachel_age) :
  current_mother_age = current_grandfather_age / 2 :=
by
  sorry

end mother_to_grandfather_age_ratio_l286_286380


namespace divide_nuts_equal_l286_286773

-- Define the conditions: sequence of 64 nuts where adjacent differ by 1 gram
def is_valid_sequence (seq : List Int) :=
  seq.length = 64 ∧ (∀ i < 63, (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ + 1) ∨ (seq.get ⟨i, sorry⟩ = seq.get ⟨i+1, sorry⟩ - 1))

-- Main theorem statement: prove that the sequence can be divided into two groups with equal number of nuts and equal weights
theorem divide_nuts_equal (seq : List Int) (h : is_valid_sequence seq) :
  ∃ (s1 s2 : List Int), s1.length = 32 ∧ s2.length = 32 ∧ (s1.sum = s2.sum) :=
sorry

end divide_nuts_equal_l286_286773


namespace solve_inequality_l286_286724

noncomputable def log_b (b x : ℝ) := Real.log x / Real.log b

theorem solve_inequality (x : ℝ) (hx : x ≠ 0 ∧ 0 < x) :
  (64 + (log_b (1/5) (x^2))^3) / (log_b (1/5) (x^6) * log_b 5 (x^2) + 5 * log_b 5 (x^6) + 14 * log_b (1/5) (x^2) + 2) ≤ 0 ↔
  (x ∈ Set.Icc (-25 : ℝ) (- Real.sqrt 5)) ∨
  (x ∈ Set.Icc (- (Real.exp (Real.log 5 / 3))) 0) ∨
  (x ∈ Set.Icc 0 (Real.exp (Real.log 5 / 3))) ∨
  (x ∈ Set.Icc (Real.sqrt 5) 25) :=
by 
  sorry

end solve_inequality_l286_286724


namespace yolkino_palkino_l286_286070

open Nat

/-- On every kilometer of the highway between the villages Yolkino and Palkino, there is a post with a sign.
    On one side of the sign, the distance to Yolkino is written, and on the other side, the distance to Palkino is written.
    The sum of all the digits on each post equals 13.
    Prove that the distance from Yolkino to Palkino is 49 kilometers. -/
theorem yolkino_palkino (n : ℕ) (h : ∀ k : ℕ, k ≤ n → (digits 10 k).sum + (digits 10 (n - k)).sum = 13) : n = 49 :=
by
  sorry

end yolkino_palkino_l286_286070


namespace solve_for_x_l286_286004

-- Define the new operation m ※ n
def operation (m n : ℤ) : ℤ :=
  if m ≥ 0 then m + n else m / n

-- Define the condition given in the problem
def condition (x : ℤ) : Prop :=
  operation (-9) (-x) = x

-- The main theorem to prove
theorem solve_for_x (x : ℤ) : condition x ↔ (x = 3 ∨ x = -3) :=
by
  sorry

end solve_for_x_l286_286004


namespace frame_cover_100x100_l286_286447

theorem frame_cover_100x100 :
  ∃! (cover: (ℕ → ℕ → Prop)), (∀ (n : ℕ) (frame: ℕ → ℕ → Prop),
    (∃ (i j : ℕ), (cover (i + n) j ∧ frame (i + n) j ∧ cover (i - n) j ∧ frame (i - n) j) ∧
                   (∃ (k l : ℕ), (cover k (l + n) ∧ frame k (l + n) ∧ cover k (l - n) ∧ frame k (l - n)))) →
    (∃ (i' j' k' l' : ℕ), cover i' j' ∧ frame i' j' ∧ cover k' l' ∧ frame k' l')) :=
sorry

end frame_cover_100x100_l286_286447


namespace sum_repeating_decimals_l286_286139

theorem sum_repeating_decimals : (0.14 + 0.27) = (41 / 99) := by
  sorry

end sum_repeating_decimals_l286_286139


namespace hundredth_odd_positive_integer_l286_286411

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l286_286411


namespace sum_series_l286_286920

noncomputable def series_sum := (∑' n : ℕ, (4 * (n + 1) - 2) / 3^(n + 1))

theorem sum_series : series_sum = 4 := by
  sorry

end sum_series_l286_286920


namespace minimum_value_l286_286527

theorem minimum_value (x y z : ℝ) (h : x + y + z = 1) : 2 * x^2 + y^2 + 3 * z^2 ≥ 3 / 7 := by
  sorry

end minimum_value_l286_286527


namespace find_range_a_l286_286501

theorem find_range_a (x y a : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) :
  (∀ x y, (1 ≤ x ∧ x ≤ 2) → (2 ≤ y ∧ y ≤ 3) → (xy ≤ a*x^2 + 2*y^2)) ↔ (-1/2 ≤ a) :=
sorry

end find_range_a_l286_286501


namespace total_servings_l286_286202

-- Definitions for the conditions

def servings_per_carrot : ℕ := 4
def plants_per_plot : ℕ := 9
def servings_multiplier_corn : ℕ := 5
def servings_multiplier_green_bean : ℤ := 2

-- Proof statement
theorem total_servings : 
  (plants_per_plot * servings_per_carrot) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn)) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn / servings_multiplier_green_bean)) = 
  306 :=
by
  sorry

end total_servings_l286_286202


namespace product_of_distinct_numbers_l286_286932

theorem product_of_distinct_numbers (x y : ℝ) (h1 : x ≠ y)
  (h2 : 1 / (1 + x^2) + 1 / (1 + y^2) = 2 / (1 + x * y)) :
  x * y = 1 := 
sorry

end product_of_distinct_numbers_l286_286932


namespace pablo_puzzle_l286_286707

open Nat

theorem pablo_puzzle (pieces_per_hour : ℕ) (hours_per_day : ℕ) (days : ℕ) 
    (pieces_per_five_puzzles : ℕ) (num_five_puzzles : ℕ) (total_pieces : ℕ) 
    (num_eight_puzzles : ℕ) :

    pieces_per_hour = 100 →
    hours_per_day = 7 →
    days = 7 →
    pieces_per_five_puzzles = 500 →
    num_five_puzzles = 5 →
    num_eight_puzzles = 8 →
    total_pieces = (pieces_per_hour * hours_per_day * days) →
    num_eight_puzzles * (total_pieces - num_five_puzzles * pieces_per_five_puzzles) / num_eight_puzzles = 300 :=
by
  intros
  sorry

end pablo_puzzle_l286_286707


namespace questionnaires_drawn_from_D_l286_286271

theorem questionnaires_drawn_from_D (a1 a2 a3 a4 total sample_b sample_total sample_d : ℕ)
  (h1 : a2 - a1 = a3 - a2)
  (h2 : a3 - a2 = a4 - a3)
  (h3 : a1 + a2 + a3 + a4 = total)
  (h4 : total = 1000)
  (h5 : sample_b = 30)
  (h6 : a2 = 200)
  (h7 : sample_total = 150)
  (h8 : sample_d * total = sample_total * a4) :
  sample_d = 60 :=
by sorry

end questionnaires_drawn_from_D_l286_286271


namespace find_b_l286_286206

theorem find_b
  (a b c : ℝ)
  (h_area : (1/2) * a * c * Real.sin (ℝ.pi / 3) = Real.sqrt 3)
  (h_ac_eq_4 : a * c = 4)
  (h_cosine : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
begin
  sorry
end

end find_b_l286_286206


namespace sales_tax_paid_l286_286144

theorem sales_tax_paid 
  (total_spent : ℝ) 
  (tax_free_cost : ℝ) 
  (tax_rate : ℝ) 
  (cost_of_taxable_items : ℝ) 
  (sales_tax : ℝ) 
  (h1 : total_spent = 40) 
  (h2 : tax_free_cost = 34.7) 
  (h3 : tax_rate = 0.06) 
  (h4 : cost_of_taxable_items = 5) 
  (h5 : sales_tax = 0.3) 
  (h6 : 1.06 * cost_of_taxable_items + tax_free_cost = total_spent) : 
  sales_tax = tax_rate * cost_of_taxable_items :=
sorry

end sales_tax_paid_l286_286144


namespace quadratic_roots_interlace_l286_286564

variable (p1 p2 q1 q2 : ℝ)

theorem quadratic_roots_interlace
(h : (q1 - q2)^2 + (p1 - p2) * (p1 * q2 - p2 * q1) < 0) :
  (∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1^2 + p1 * r1 + q1 = 0 ∧ r2^2 + p1 * r2 + q1 = 0)) ∧
  (∃ s1 s2 : ℝ, s1 ≠ s2 ∧ (s1^2 + p2 * s1 + q2 = 0 ∧ s2^2 + p2 * s2 + q2 = 0)) ∧
  (∃ a b c d : ℝ, a < b ∧ b < c ∧ c < d ∧ 
  (a^2 + p1*a + q1 = 0 ∧ b^2 + p2*b + q2 = 0 ∧ c^2 + p1*c + q1 = 0 ∧ d^2 + p2*d + q2 = 0)) := 
sorry

end quadratic_roots_interlace_l286_286564


namespace infinite_div_pairs_l286_286546

theorem infinite_div_pairs {a : ℕ → ℕ} (h_seq : ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n ≤ 2001) :
  ∃ (s : ℕ → (ℕ × ℕ)), (∀ n, (s n).2 < (s n).1) ∧ (a ((s n).2) ∣ a ((s n).1)) :=
sorry

end infinite_div_pairs_l286_286546


namespace integral_f_x_4_f_deriv_x_l286_286065

open Real

theorem integral_f_x_4_f_deriv_x (f : ℝ → ℝ) (h_cont : Continuous f) (h_integral_1 : (∫ x in 0..1, f x * f' x) = 0) (h_integral_2 : (∫ x in 0..1, f x ^ 2 * f' x) = 18) : 
  (∫ x in 0..1, f x ^ 4 * f' x) = 486 / 5 := 
begin
  -- the proof will be provided here
  sorry
end

end integral_f_x_4_f_deriv_x_l286_286065


namespace find_y_l286_286256

theorem find_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (hr : x % y = 9) (hxy : (x : ℝ) / y = 96.45) : y = 20 :=
by
  sorry

end find_y_l286_286256


namespace net_profit_calc_l286_286113

theorem net_profit_calc:
  ∃ (x y : ℕ), x + y = 25 ∧ 1700 * x + 1800 * y = 44000 ∧ 2400 * x + 2600 * y = 63000 := by
  sorry

end net_profit_calc_l286_286113


namespace student_avg_always_greater_l286_286625

theorem student_avg_always_greater (x y z : ℝ) (h1 : x < y) (h2 : y < z) : 
  ( ( (x + y) / 2 + z) / 2 ) > ( (x + y + z) / 3 ) :=
by
  sorry

end student_avg_always_greater_l286_286625


namespace count_monomials_l286_286823

def isMonomial (expr : String) : Bool :=
  match expr with
  | "m+n" => false
  | "2x^2y" => true
  | "1/x" => true
  | "-5" => true
  | "a" => true
  | _ => false

theorem count_monomials :
  let expressions := ["m+n", "2x^2y", "1/x", "-5", "a"]
  (expressions.filter isMonomial).length = 3 :=
by { sorry }

end count_monomials_l286_286823


namespace probability_of_drawing_one_black_ball_l286_286820

theorem probability_of_drawing_one_black_ball 
  (white_balls : ℕ)
  (black_balls : ℕ)
  (total_balls : ℕ)
  (drawn_balls : ℕ)
  (h_w : white_balls = 3)
  (h_b : black_balls = 2)
  (h_t : total_balls = white_balls + black_balls)
  (h_d : drawn_balls = 2) :
  (Combination (white_balls + black_balls) drawn_balls) 
  ≠ 0 → 
  (2 * Combination white_balls 1 * Combination black_balls 1 / 
  Combination (white_balls + black_balls) drawn_balls : ℚ) = 3 / 5 := by
    sorry

end probability_of_drawing_one_black_ball_l286_286820


namespace pat_initial_stickers_l286_286705

def initial_stickers (s : ℕ) : ℕ := s  -- Number of stickers Pat had on the first day of the week

def stickers_earned : ℕ := 22  -- Stickers earned during the week

def stickers_end_week (s : ℕ) : ℕ := initial_stickers s + stickers_earned  -- Stickers at the end of the week

theorem pat_initial_stickers (s : ℕ) (h : stickers_end_week s = 61) : s = 39 :=
by
  sorry

end pat_initial_stickers_l286_286705


namespace number_of_months_in_season_l286_286242

def games_per_month : ℝ := 323.0
def total_games : ℝ := 5491.0

theorem number_of_months_in_season : total_games / games_per_month = 17 := 
by
  sorry

end number_of_months_in_season_l286_286242


namespace evaluate_expression_l286_286829

-- Define the greatest power of 2 and 3 that are factors of 360
def a : ℕ := 3 -- 2^3 is the greatest power of 2 that is a factor of 360
def b : ℕ := 2 -- 3^2 is the greatest power of 3 that is a factor of 360

theorem evaluate_expression : (1 / 4)^(b - a) = 4 := 
by 
  have h1 : a = 3 := rfl
  have h2 : b = 2 := rfl
  rw [h1, h2]
  simp
  sorry

end evaluate_expression_l286_286829


namespace contrapositive_equivalence_l286_286263

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 + 3*x - 4 = 0 → x = -4 ∨ x = 1)) ↔ (∀ x : ℝ, (x ≠ -4 ∧ x ≠ 1 → x^2 + 3*x - 4 ≠ 0)) :=
by {
  sorry
}

end contrapositive_equivalence_l286_286263


namespace expected_total_cost_of_removing_blocks_l286_286606

/-- 
  There are six blocks in a row labeled 1 through 6, each with weight 1.
  Two blocks x ≤ y are connected if for all x ≤ z ≤ y, block z has not been removed.
  While there is at least one block remaining, a block is chosen uniformly at random and removed.
  The cost of removing a block is the sum of the weights of the blocks that are connected to it.
  Prove that the expected total cost of removing all blocks is 163 / 10.
-/
theorem expected_total_cost_of_removing_blocks : (6:ℚ) + 5 + 8/3 + 3/2 + 4/5 + 1/3 = 163 / 10 := sorry

end expected_total_cost_of_removing_blocks_l286_286606


namespace find_side_b_l286_286210

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = real.sqrt 3)
  (h_B : B = (60 : ℝ) * real.pi / 180) -- Convert degrees to radians
  (h_cond : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 := 
sorry

end find_side_b_l286_286210


namespace extreme_value_range_of_a_l286_286187

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (2 * x) * (1 - a * x)

theorem extreme_value_range_of_a (a : ℝ) :
  a ∈ Set.Ioo (2 / 3 : ℝ) 2 ↔
    ∃ c ∈ Set.Ioo 0 1, ∀ x : ℝ, f a c = f a x :=
by
  sorry

end extreme_value_range_of_a_l286_286187


namespace total_distance_correct_l286_286868

noncomputable def total_distance_covered (rA rB rC : ℝ) (revA revB revC : ℕ) : ℝ :=
  let pi := Real.pi
  let circumference (r : ℝ) := 2 * pi * r
  let distance (r : ℝ) (rev : ℕ) := circumference r * rev
  distance rA revA + distance rB revB + distance rC revC

theorem total_distance_correct :
  total_distance_covered 22.4 35.7 55.9 600 450 375 = 316015.4 :=
by
  sorry

end total_distance_correct_l286_286868


namespace rectangle_diagonal_l286_286849

theorem rectangle_diagonal (P A: ℝ) (hP : P = 46) (hA : A = 120) : ∃ d : ℝ, d = 17 :=
by
  -- Sorry provides the placeholder for the actual proof.
  sorry

end rectangle_diagonal_l286_286849


namespace initial_payment_mr_dubois_l286_286558

-- Definition of the given conditions
def total_cost_of_car : ℝ := 13380
def monthly_payment : ℝ := 420
def number_of_months : ℝ := 19

-- Calculate the total amount paid in monthly installments
def total_amount_paid_in_installments : ℝ := monthly_payment * number_of_months

-- Statement of the theorem we want to prove
theorem initial_payment_mr_dubois :
  total_cost_of_car - total_amount_paid_in_installments = 5400 :=
by
  sorry

end initial_payment_mr_dubois_l286_286558


namespace convert_1729_to_base5_l286_286927

-- Definition of base conversion from base 10 to base 5.
def convert_to_base5 (n : ℕ) : list ℕ :=
  let rec aux (n : ℕ) (acc : list ℕ) :=
    if h : n = 0 then acc
    else let quotient := n / 5
         let remainder := n % 5
         aux quotient (remainder :: acc)
  aux n []

-- The theorem we seek to prove.
theorem convert_1729_to_base5 : convert_to_base5 1729 = [2, 3, 4, 0, 4] :=
by
  sorry

end convert_1729_to_base5_l286_286927


namespace intersection_complement_l286_286662

-- Defining the sets A and B
def setA : Set ℝ := { x | -3 < x ∧ x < 3 }
def setB : Set ℝ := { x | x < -2 }
def complementB : Set ℝ := { x | x ≥ -2 }

-- The theorem to be proved
theorem intersection_complement :
  setA ∩ complementB = { x | -2 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_complement_l286_286662


namespace initial_files_count_l286_286698

theorem initial_files_count (deleted_files folders files_per_folder total_files initial_files : ℕ)
    (h1 : deleted_files = 21)
    (h2 : folders = 9)
    (h3 : files_per_folder = 8)
    (h4 : total_files = folders * files_per_folder)
    (h5 : initial_files = total_files + deleted_files) :
    initial_files = 93 :=
by
  sorry

end initial_files_count_l286_286698


namespace area_triangle_QXY_l286_286399

-- Definition of the problem
def length_rectangle (PQ PS : ℝ) : Prop :=
  PQ = 8 ∧ PS = 6

def diagonal_division (PR : ℝ) (X Y : ℝ) : Prop :=
  PR = 10 ∧ X = 2.5 ∧ Y = 2.5

-- The statement we need to prove
theorem area_triangle_QXY
  (PQ PS PR X Y : ℝ)
  (h1 : length_rectangle PQ PS)
  (h2 : diagonal_division PR X Y)
  : ∃ (A : ℝ), A = 6 := by
  sorry

end area_triangle_QXY_l286_286399


namespace units_digit_product_l286_286461

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_product (a b c : ℕ) :
  units_digit a = 7 → units_digit b = 3 → units_digit c = 9 →
  units_digit ((a * b) * c) = 9 :=
by
  intros h1 h2 h3
  sorry

end units_digit_product_l286_286461


namespace parabola_focus_standard_equation_l286_286318

theorem parabola_focus_standard_equation :
  ∃ (a b : ℝ), (a = 16 ∧ b = 0) ∨ (a = 0 ∧ b = -8) →
  (∃ (F : ℝ × ℝ), F = (4, 0) ∨ F = (0, -2) ∧ F ∈ {p : ℝ × ℝ | (p.1 - 2 * p.2 - 4 = 0)} →
  (∃ (x y : ℝ), (y^2 = a * x) ∨ (x^2 = b * y))) := sorry

end parabola_focus_standard_equation_l286_286318


namespace initial_water_percentage_l286_286120

noncomputable def initial_percentage_of_water : ℚ :=
  20

theorem initial_water_percentage
  (initial_volume : ℚ := 125)
  (added_water : ℚ := 8.333333333333334)
  (final_volume : ℚ := initial_volume + added_water)
  (desired_percentage : ℚ := 25)
  (desired_amount_of_water : ℚ := desired_percentage / 100 * final_volume)
  (initial_amount_of_water : ℚ := desired_amount_of_water - added_water) :
  (initial_amount_of_water / initial_volume * 100 = initial_percentage_of_water) :=
by
  sorry

end initial_water_percentage_l286_286120


namespace true_statement_count_l286_286991

def n_star (n : ℕ) : ℚ := 1 / n

theorem true_statement_count :
  let s1 := (n_star 4 + n_star 8 = n_star 12)
  let s2 := (n_star 9 - n_star 1 = n_star 8)
  let s3 := (n_star 5 * n_star 3 = n_star 15)
  let s4 := (n_star 16 - n_star 4 = n_star 12)
  (if s1 then 1 else 0) +
  (if s2 then 1 else 0) +
  (if s3 then 1 else 0) +
  (if s4 then 1 else 0) = 1 :=
by
  -- Proof goes here
  sorry

end true_statement_count_l286_286991


namespace volume_of_prism_l286_286243

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 60)
                                     (h2 : y * z = 75)
                                     (h3 : x * z = 100) :
  x * y * z = 671 :=
by
  sorry

end volume_of_prism_l286_286243


namespace product_of_possible_values_l286_286521

noncomputable def math_problem (x : ℚ) : Prop :=
  |(10 / x) - 4| = 3

theorem product_of_possible_values :
  let x1 := 10 / 7
  let x2 := 10
  (x1 * x2) = (100 / 7) :=
by
  sorry

end product_of_possible_values_l286_286521


namespace final_solution_percentage_l286_286569

variable (initial_volume replaced_fraction : ℝ)
variable (initial_concentration replaced_concentration : ℝ)

noncomputable
def final_acid_percentage (initial_volume replaced_fraction initial_concentration replaced_concentration : ℝ) : ℝ :=
  let remaining_volume := initial_volume * (1 - replaced_fraction)
  let replaced_volume := initial_volume * replaced_fraction
  let remaining_acid := remaining_volume * initial_concentration
  let replaced_acid := replaced_volume * replaced_concentration
  let total_acid := remaining_acid + replaced_acid
  let final_volume := initial_volume
  (total_acid / final_volume) * 100

theorem final_solution_percentage :
  final_acid_percentage 100 0.5 0.5 0.3 = 40 :=
by
  sorry

end final_solution_percentage_l286_286569


namespace snowman_volume_l286_286828

theorem snowman_volume
  (r1 r2 r3 : ℝ)
  (volume : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 4)
  (h3 : r3 = 6)
  (h_volume : volume = (4.0 / 3.0) * Real.pi * (r1 ^ 3 + r2 ^ 3 + r3 ^ 3)) :
  volume = (1124.0 / 3.0) * Real.pi :=
by
  sorry

end snowman_volume_l286_286828


namespace subtract_decimal_numbers_l286_286154

theorem subtract_decimal_numbers : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_decimal_numbers_l286_286154


namespace road_construction_equation_l286_286594

theorem road_construction_equation (x : ℝ) (hx : x > 0) :
  (9 / x) - (12 / (x + 1)) = 1 / 2 :=
sorry

end road_construction_equation_l286_286594


namespace simplify_expr_l286_286899

variable (a b : ℝ)

def expr := a * b - (a^2 - a * b + b^2)

theorem simplify_expr : expr a b = - a^2 + 2 * a * b - b^2 :=
by 
  -- No proof is provided as per the instructions
  sorry

end simplify_expr_l286_286899


namespace undefined_expression_iff_l286_286637

theorem undefined_expression_iff (x : ℝ) :
  (x^2 - 24 * x + 144 = 0) ↔ (x = 12) := 
sorry

end undefined_expression_iff_l286_286637


namespace trig_expression_l286_286800

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 := 
by 
  sorry

end trig_expression_l286_286800


namespace karl_total_income_correct_l286_286545

noncomputable def price_of_tshirt : ℝ := 5
noncomputable def price_of_pants : ℝ := 4
noncomputable def price_of_skirt : ℝ := 6
noncomputable def price_of_refurbished_tshirt : ℝ := price_of_tshirt / 2

noncomputable def discount_for_skirts (n : ℕ) : ℝ := (n / 2) * 2 * price_of_skirt * 0.10
noncomputable def discount_for_tshirts (n : ℕ) : ℝ := (n / 5) * 5 * price_of_tshirt * 0.20
noncomputable def discount_for_pants (n : ℕ) : ℝ := 0 -- accounted for in quantity

noncomputable def sales_tax (amount : ℝ) : ℝ := amount * 0.08

noncomputable def total_income : ℝ := 
  let tshirt_income := 8 * price_of_tshirt + 7 * price_of_refurbished_tshirt - discount_for_tshirts 15
  let pants_income := 6 * price_of_pants - discount_for_pants 6
  let skirts_income := 12 * price_of_skirt - discount_for_skirts 12
  let income_before_tax := tshirt_income + pants_income + skirts_income
  income_before_tax + sales_tax income_before_tax

theorem karl_total_income_correct : total_income = 141.80 :=
by
  sorry

end karl_total_income_correct_l286_286545


namespace simplify_to_x5_l286_286879

theorem simplify_to_x5 (x : ℝ) :
  x^2 * x^3 = x^5 :=
by {
  -- proof goes here
  sorry
}

end simplify_to_x5_l286_286879


namespace concert_attendance_difference_l286_286702

theorem concert_attendance_difference :
  let first_concert := 65899
  let second_concert := 66018
  second_concert - first_concert = 119 :=
by
  sorry

end concert_attendance_difference_l286_286702


namespace calculate_expression_l286_286460

theorem calculate_expression :
  (0.125: ℝ) ^ 3 * (-8) ^ 3 = -1 := 
by
  sorry

end calculate_expression_l286_286460


namespace multiple_of_son_age_last_year_l286_286221

theorem multiple_of_son_age_last_year
  (G : ℕ) (S : ℕ) (M : ℕ)
  (h1 : G = 42 - 1)
  (h2 : S = 16 - 1)
  (h3 : G = M * S - 4) :
  M = 3 := by
  sorry

end multiple_of_son_age_last_year_l286_286221


namespace roots_quadratic_identity_l286_286338

theorem roots_quadratic_identity (p q : ℝ) (r s : ℝ) (h1 : r + s = 3 * p) (h2 : r * s = 2 * q) :
  r^2 + s^2 = 9 * p^2 - 4 * q := 
by 
  sorry

end roots_quadratic_identity_l286_286338


namespace remaining_pieces_l286_286352

theorem remaining_pieces (initial_pieces : ℕ) (arianna_lost : ℕ) (samantha_lost : ℕ) (diego_lost : ℕ) (lucas_lost : ℕ) :
  initial_pieces = 128 → arianna_lost = 3 → samantha_lost = 9 → diego_lost = 5 → lucas_lost = 7 →
  initial_pieces - (arianna_lost + samantha_lost + diego_lost + lucas_lost) = 104 := by
  sorry

end remaining_pieces_l286_286352


namespace train_crossing_signal_pole_l286_286265

theorem train_crossing_signal_pole
  (length_train : ℕ)
  (same_length_platform : ℕ)
  (time_crossing_platform : ℕ)
  (h_train_platform : length_train = 420)
  (h_platform : same_length_platform = 420)
  (h_time_platform : time_crossing_platform = 60) : 
  (length_train / (length_train + same_length_platform / time_crossing_platform)) = 30 := 
by 
  sorry

end train_crossing_signal_pole_l286_286265


namespace calculate_crayons_lost_l286_286998

def initial_crayons := 440
def given_crayons := 111
def final_crayons := 223

def crayons_left_after_giving := initial_crayons - given_crayons
def crayons_lost := crayons_left_after_giving - final_crayons

theorem calculate_crayons_lost : crayons_lost = 106 :=
  by
    sorry

end calculate_crayons_lost_l286_286998


namespace impossibility_of_equal_sum_selection_l286_286926

theorem impossibility_of_equal_sum_selection :
  ¬ ∃ (selected non_selected : Fin 10 → ℕ),
    (∀ i, selected i = 1 ∨ selected i = 36 ∨ selected i = 2 ∨ selected i = 35 ∨ 
              selected i = 3 ∨ selected i = 34 ∨ selected i = 4 ∨ selected i = 33 ∨ 
              selected i = 5 ∨ selected i = 32 ∨ selected i = 6 ∨ selected i = 31 ∨ 
              selected i = 7 ∨ selected i = 30 ∨ selected i = 8 ∨ selected i = 29 ∨ 
              selected i = 9 ∨ selected i = 28 ∨ selected i = 10 ∨ selected i = 27) ∧ 
    (∀ i, non_selected i = 1 ∨ non_selected i = 36 ∨ non_selected i = 2 ∨ non_selected i = 35 ∨ 
              non_selected i = 3 ∨ non_selected i = 34 ∨ non_selected i = 4 ∨ non_selected i = 33 ∨ 
              non_selected i = 5 ∨ non_selected i = 32 ∨ non_selected i = 6 ∨ non_selected i = 31 ∨ 
              non_selected i = 7 ∨ non_selected i = 30 ∨ non_selected i = 8 ∨ non_selected i = 29 ∨ 
              non_selected i = 9 ∨ non_selected i = 28 ∨ non_selected i = 10 ∨ non_selected i = 27) ∧ 
    (selected ≠ non_selected) ∧ 
    (Finset.univ.sum selected = Finset.univ.sum non_selected) :=
sorry

end impossibility_of_equal_sum_selection_l286_286926


namespace scientific_notation_of_116_million_l286_286528

theorem scientific_notation_of_116_million : 116000000 = 1.16 * 10^7 :=
sorry

end scientific_notation_of_116_million_l286_286528


namespace no_common_points_eq_l286_286039

theorem no_common_points_eq (a : ℝ) : 
  ((∀ x y : ℝ, y = (a^2 - a) * x + 1 - a → y ≠ 2 * x - 1) ↔ (a = -1)) :=
by
  sorry

end no_common_points_eq_l286_286039


namespace intersection_eq_l286_286053

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l286_286053


namespace find_length_of_b_l286_286208

theorem find_length_of_b
  {A B C : Type*}
  (a b c : ℝ)
  (area : ℝ)
  (angleB : ℝ)
  (h_area : area = sqrt 3)
  (h_angle : angleB = real.pi / 3)  -- 60 degrees in radians
  (h_sides : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  -- 1. Using the given area constraint: area = 1/2 * a * c * sin(B)
  -- 2. Using the given angle: sin(60) = sqrt(3)/2
  -- 3. Using the given sides relation: a^2 + c^2 = 3ac
  -- 4. Using the Law of Cosines: cos(60) = 1/2
  -- fill proof here
  sorry

end find_length_of_b_l286_286208


namespace conditional_probability_l286_286531

noncomputable def P (e : Prop) : ℝ := sorry

variable (A B : Prop)

variables (h1 : P A = 0.6)
variables (h2 : P B = 0.5)
variables (h3 : P (A ∨ B) = 0.7)

theorem conditional_probability :
  (P A ∧ P B) / P B = 0.8 := by
  sorry

end conditional_probability_l286_286531


namespace find_b_correct_l286_286214

axioms (a b c : ℝ) (B : ℝ)
  (area_of_triangle : ℝ)
  (h1 : area_of_triangle = real.sqrt 3)
  (h2 : B = 60 * real.pi / 180) -- converting degrees to radians
  (h3 : a^2 + c^2 = 3 * a * c)

noncomputable def find_b : ℝ := 2 * real.sqrt 2

theorem find_b_correct :
  let b := find_b in 
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_b_correct_l286_286214


namespace mother_age_when_harry_born_l286_286964

variable (harry_age father_age mother_age : ℕ)

-- Conditions
def harry_is_50 (harry_age : ℕ) : Prop := harry_age = 50
def father_is_24_years_older (harry_age father_age : ℕ) : Prop := father_age = harry_age + 24
def mother_younger_by_1_25_of_harry_age (harry_age father_age mother_age : ℕ) : Prop := mother_age = father_age - harry_age / 25

-- Proof Problem
theorem mother_age_when_harry_born (harry_age father_age mother_age : ℕ) 
  (h₁ : harry_is_50 harry_age) 
  (h₂ : father_is_24_years_older harry_age father_age)
  (h₃ : mother_younger_by_1_25_of_harry_age harry_age father_age mother_age) :
  mother_age - harry_age = 22 :=
by
  sorry

end mother_age_when_harry_born_l286_286964


namespace interest_rate_C_l286_286755

theorem interest_rate_C (P A G : ℝ) (R : ℝ) (t : ℝ := 3) (rate_A : ℝ := 0.10) :
  P = 4000 ∧ rate_A = 0.10 ∧ G = 180 →
  (P * rate_A * t + G) = P * (R / 100) * t →
  R = 11.5 :=
by
  intros h_cond h_eq
  -- proof to be filled, use the given conditions and equations
  sorry

end interest_rate_C_l286_286755


namespace angle_EDP_eq_90_l286_286543

-- a) Definitions of triangle and conditions
variables {A B C D E M N P : Type*}
variables (triangle_ABC : IsTriangle A B C)
variables (angle_ABC : angle B A C = 60)
variables (AB BC : Real)
variables (r : Real)
variables (h1 : 5 * AB = 4 * BC)
variables (D_foot : IsFootOfAltitude B D triangle_ABC)
variables (E_foot : IsFootOfAltitude C E triangle_ABC)
variables (M_midpoint : IsMidpoint M B D)
variables (circumcircle_BMC : IsCircumcircleOf B M C)
variables (N_on_AC : IsOnLine N A C)
variables (N_circumcircle : IsOnCircumcircle N circumcircle_BMC)
variables (BN_intersection : IsIntersectionOf BN N A C)
variables (CM_intersection : IsIntersectionOf CM M B C)
variables (P_intersection : BN = CM ∧ BN ∩ CM = P)

-- d) The theorem to be proven
theorem angle_EDP_eq_90
  (angle_ABC : angle B A C = 60)
  (h1 : 5 * AB = 4 * BC)
  (D_foot : IsFootOfAltitude B D triangle_ABC)
  (E_foot : IsFootOfAltitude C E triangle_ABC)
  (M_midpoint : IsMidpoint M B D)
  (circumcircle_BMC : IsCircumcircleOf B M C)
  (N_on_AC : IsOnLine N A C)
  (N_circumcircle : IsOnCircumcircle N circumcircle_BMC)
  (BN_intersection : IsIntersectionOf BN N A C)
  (CM_intersection : IsIntersectionOf CM M B C)
  (P_intersection : BN = CM ∧ BN ∩ CM = P):
  angle E D P = 90 :=
sorry

end angle_EDP_eq_90_l286_286543


namespace calculateBooksRemaining_l286_286095

noncomputable def totalBooksRemaining
    (initialBooks : ℕ)
    (n : ℕ)
    (a₁ : ℕ)
    (d : ℕ)
    (borrowedBooks : ℕ)
    (returnedBooks : ℕ) : ℕ :=
  let sumDonations := n * (2 * a₁ + (n - 1) * d) / 2
  let totalAfterDonations := initialBooks + sumDonations
  totalAfterDonations - borrowedBooks + returnedBooks

theorem calculateBooksRemaining :
  totalBooksRemaining 1000 15 2 2 350 270 = 1160 :=
by
  sorry

end calculateBooksRemaining_l286_286095


namespace length_AC_l286_286681

variable {A B C : Type} [Field A] [Field B] [Field C]

-- Definitions for the problem conditions
noncomputable def length_AB : ℝ := 3
noncomputable def angle_A : ℝ := Real.pi * 120 / 180
noncomputable def area_ABC : ℝ := (15 * Real.sqrt 3) / 4

-- The theorem statement
theorem length_AC (b : ℝ) (h1 : b = length_AB) (h2 : angle_A = Real.pi * 120 / 180) (h3 : area_ABC = (15 * Real.sqrt 3) / 4) : b = 5 :=
sorry

end length_AC_l286_286681


namespace program_output_l286_286597

theorem program_output :
  ∃ a b : ℕ, a = 10 ∧ b = a - 8 ∧ a = a - b ∧ a = 8 :=
by
  let a := 10
  let b := a - 8
  let a := a - b
  use a
  use b
  sorry

end program_output_l286_286597


namespace prob_sunny_l286_286451

variables (A B C : Prop) 
variables (P : Prop → ℝ)

-- Conditions
axiom prob_A : P A = 0.45
axiom prob_B : P B = 0.2
axiom mutually_exclusive : P A + P B + P C = 1

-- Proof problem
theorem prob_sunny : P C = 0.35 :=
by sorry

end prob_sunny_l286_286451


namespace workers_in_first_group_l286_286082

-- Define the first condition: Some workers collect 48 kg of cotton in 4 days
def cotton_collected_by_W_workers_in_4_days (W : ℕ) : ℕ := 48

-- Define the second condition: 9 workers collect 72 kg of cotton in 2 days
def cotton_collected_by_9_workers_in_2_days : ℕ := 72

-- Define the rate of cotton collected per worker per day for both scenarios
def rate_per_worker_first_group (W : ℕ) : ℕ :=
cotton_collected_by_W_workers_in_4_days W / (W * 4)

def rate_per_worker_second_group : ℕ :=
cotton_collected_by_9_workers_in_2_days / (9 * 2)

-- Given the rates are the same for both groups, prove W = 3
theorem workers_in_first_group (W : ℕ) (h : rate_per_worker_first_group W = rate_per_worker_second_group) : W = 3 :=
sorry

end workers_in_first_group_l286_286082


namespace union_sets_l286_286029

open Set

def setM : Set ℝ := {x : ℝ | x^2 < x}
def setN : Set ℝ := {x : ℝ | x^2 + 2*x - 3 < 0}

theorem union_sets : setM ∪ setN = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end union_sets_l286_286029


namespace blowfish_stayed_own_tank_l286_286630

def number_clownfish : ℕ := 50
def number_blowfish : ℕ := 50
def number_clownfish_display_initial : ℕ := 24
def number_clownfish_display_final : ℕ := 16

theorem blowfish_stayed_own_tank : 
    (number_clownfish + number_blowfish = 100) ∧ 
    (number_clownfish = number_blowfish) ∧ 
    (number_clownfish_display_final = 2 / 3 * number_clownfish_display_initial) →
    ∀ (blowfish : ℕ), 
    blowfish = number_blowfish - number_clownfish_display_initial → 
    blowfish = 26 :=
sorry

end blowfish_stayed_own_tank_l286_286630


namespace sequence_difference_l286_286312

theorem sequence_difference : 
  (∃ (a : ℕ → ℤ) (S : ℕ → ℤ), 
    (∀ n : ℕ, S n = n^2 + 2 * n) ∧ 
    (∀ n : ℕ, n > 0 → a n = S n - S (n - 1) ) ∧ 
    (a 4 - a 2 = 4)) :=
by
  sorry

end sequence_difference_l286_286312


namespace parallelogram_area_60_l286_286071

theorem parallelogram_area_60
  (α β : ℝ) (a b : ℝ)
  (h_angle : α = 150) 
  (h_adj_angle : β = 180 - α) 
  (h_len_1 : a = 10)
  (h_len_2 : b = 12) :
  ∃ (area : ℝ), area = 60 := 
by 
  use 60
  sorry

end parallelogram_area_60_l286_286071


namespace temperature_on_Friday_l286_286086

-- Define the temperatures for each day
variables (M T W Th F : ℕ)

-- Declare the given conditions as assumptions
axiom cond1 : (M + T + W + Th) / 4 = 48
axiom cond2 : (T + W + Th + F) / 4 = 46
axiom cond3 : M = 40

-- State the theorem
theorem temperature_on_Friday : F = 32 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end temperature_on_Friday_l286_286086


namespace product_of_d_l286_286148

theorem product_of_d (d1 d2 : ℕ) (h1 : ∃ k1 : ℤ, 49 - 12 * d1 = k1^2)
  (h2 : ∃ k2 : ℤ, 49 - 12 * d2 = k2^2) (h3 : 0 < d1) (h4 : 0 < d2)
  (h5 : d1 ≠ d2) : d1 * d2 = 8 := 
sorry

end product_of_d_l286_286148


namespace non_unique_solution_of_system_irrelevant_m_l286_286166

theorem non_unique_solution_of_system (k m : ℝ) :
  (∀ (x y z : ℝ), 3 * (3 * x^2 + 4 * y^2) = 36 → (k * x^2 + 12 * y^2) = 30 → (m * x^3 - 2 * y^3 + z^2) = 24 → 
  (k = 9)) :=
by {
  sorry
} 

theorem irrelevant_m (m : ℝ) : True :=
by {
  trivial
}

end non_unique_solution_of_system_irrelevant_m_l286_286166


namespace triangle_ABC_right_angled_l286_286978

variable {α : Type*} [LinearOrderedField α]

variables (a b c : α)
variables (A B C : ℝ)

theorem triangle_ABC_right_angled
  (h1 : b^2 = c^2 + a^2 - c * a)
  (h2 : Real.sin A = 2 * Real.sin C)
  (h3 : Real.cos B = 1 / 2) :
  B = (Real.pi / 2) := by
  sorry

end triangle_ABC_right_angled_l286_286978


namespace algebraic_notation_3m_minus_n_squared_l286_286009

theorem algebraic_notation_3m_minus_n_squared (m n : ℝ) : 
  (3 * m - n)^2 = (3 * m - n) ^ 2 :=
by sorry

end algebraic_notation_3m_minus_n_squared_l286_286009


namespace range_of_m_l286_286953

theorem range_of_m (m x : ℝ) (h₁ : (x / (x - 3) - 2 = m / (x - 3))) (h₂ : x ≠ 3) : x > 0 ↔ m < 6 ∧ m ≠ 3 :=
by
  sorry

end range_of_m_l286_286953


namespace systematic_sampling_student_number_l286_286535

theorem systematic_sampling_student_number 
  (total_students : ℕ)
  (sample_size : ℕ)
  (interval_between_numbers : ℕ)
  (student_17_in_sample : ∃ n, 17 = n ∧ n ≤ total_students ∧ n % interval_between_numbers = 5)
  : ∃ m, m = 41 ∧ m ≤ total_students ∧ m % interval_between_numbers = 5 := 
sorry

end systematic_sampling_student_number_l286_286535


namespace no_real_roots_implies_negative_l286_286947

theorem no_real_roots_implies_negative (m : ℝ) : (¬ ∃ x : ℝ, x^2 = m) → m < 0 :=
sorry

end no_real_roots_implies_negative_l286_286947


namespace coefficient_x3_expansion_l286_286477

open Polynomial

noncomputable def expansion : Polynomial ℤ := (X - 1) * Polynomial.C 2 * X + Polynomial.C 1)^ 5

theorem coefficient_x3_expansion : coeff expansion 3 = -40 := by
  -- Proof goes here
  sorry

end coefficient_x3_expansion_l286_286477


namespace correct_total_annual_salary_expression_l286_286615

def initial_workers : ℕ := 8
def initial_salary : ℝ := 1.0 -- in ten thousand yuan
def new_workers : ℕ := 3
def new_worker_initial_salary : ℝ := 0.8 -- in ten thousand yuan
def salary_increase_rate : ℝ := 1.2 -- 20% increase each year

def total_annual_salary (n : ℕ) : ℝ :=
  (3 * n + 5) * salary_increase_rate^n + (new_workers * new_worker_initial_salary)

theorem correct_total_annual_salary_expression (n : ℕ) :
  total_annual_salary n = (3 * n + 5) * 1.2^n + 2.4 := 
by
  sorry

end correct_total_annual_salary_expression_l286_286615


namespace wholesale_price_l286_286762

theorem wholesale_price (R : ℝ) (W : ℝ)
  (hR : R = 120)
  (h_discount : ∀ SP : ℝ, SP = R - (0.10 * R))
  (h_profit : ∀ P : ℝ, P = 0.20 * W)
  (h_SP_eq_W_P : ∀ SP P : ℝ, SP = W + P) :
  W = 90 := by
  sorry

end wholesale_price_l286_286762


namespace cans_per_bag_l286_286720

def total_cans : ℕ := 42
def bags_saturday : ℕ := 4
def bags_sunday : ℕ := 3
def total_bags : ℕ := bags_saturday + bags_sunday

theorem cans_per_bag (h1 : total_cans = 42) (h2 : total_bags = 7) : total_cans / total_bags = 6 :=
by {
    -- proof body to be filled
    sorry
}

end cans_per_bag_l286_286720


namespace power_binary_representation_zero_digit_l286_286715

theorem power_binary_representation_zero_digit
  (a n s : ℕ) (ha : a > 1) (hn : n > 1) (hs : s > 0) :
  a ^ n ≠ 2 ^ s - 1 :=
by
  sorry

end power_binary_representation_zero_digit_l286_286715


namespace ratio_of_perimeters_l286_286423

noncomputable def sqrt2 : ℝ := Real.sqrt 2

theorem ratio_of_perimeters (d1 : ℝ) :
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2 
  (P2 / P1 = 1 + sqrt2) :=
by
  let d2 := (1 + sqrt2) * d1
  let s1 := d1 / sqrt2
  let s2 := d2 / sqrt2
  let P1 := 4 * s1
  let P2 := 4 * s2
  sorry

end ratio_of_perimeters_l286_286423


namespace remainder_of_power_of_five_modulo_500_l286_286459

theorem remainder_of_power_of_five_modulo_500 :
  (5 ^ (5 ^ (5 ^ 2))) % 500 = 25 :=
by
  sorry

end remainder_of_power_of_five_modulo_500_l286_286459


namespace symmetry_center_example_l286_286374

-- Define the function tan(2x - π/4)
noncomputable def func (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

-- Define what it means to be a symmetry center for the function
def is_symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * (p.1) - x) = 2 * p.2 - f x

-- Statement of the proof problem
theorem symmetry_center_example : is_symmetry_center func (-Real.pi / 8, 0) :=
sorry

end symmetry_center_example_l286_286374


namespace sum_of_arithmetic_series_l286_286931

def a₁ : ℕ := 9
def d : ℕ := 4
def n : ℕ := 50

noncomputable def nth_term (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_arithmetic_series (a₁ d n : ℕ) : ℕ := n / 2 * (a₁ + nth_term a₁ d n)

theorem sum_of_arithmetic_series :
  sum_arithmetic_series a₁ d n = 5350 :=
by
  sorry

end sum_of_arithmetic_series_l286_286931


namespace smallest_d_l286_286837

noncomputable def d := 53361

theorem smallest_d :
  ∃ (p q r : ℕ), p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧
    10000 * d = (p * q * r) ^ 2 ∧ d = 53361 :=
  by
    sorry

end smallest_d_l286_286837


namespace necessary_but_not_sufficient_condition_l286_286114

theorem necessary_but_not_sufficient_condition (a c : ℝ) (h : c ≠ 0) : ¬ ((∀ (a : ℝ) (h : c ≠ 0), (ax^2 + y^2 = c) → ((ax^2 + y^2 = c) → ( (c ≠ 0) ))) ∧ ¬ ((∀ (a : ℝ), ¬ (ax^2 + y^2 ≠ c) → ( (ax^2 + y^2 = c) → ((c = 0) ))) )) :=
sorry

end necessary_but_not_sufficient_condition_l286_286114


namespace intersection_M_N_l286_286059

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l286_286059


namespace find_f_x_sq_minus_2_l286_286836

-- Define the polynomial and its given condition
def f (x : ℝ) : ℝ := sorry  -- f is some polynomial, we'll leave it unspecified for now

-- Assume the given condition
axiom f_condition : ∀ x : ℝ, f (x^2 + 2) = x^4 + 6 * x^2 + 4

-- Prove the desired result
theorem find_f_x_sq_minus_2 (x : ℝ) : f (x^2 - 2) = x^4 - 2 * x^2 - 4 :=
sorry

end find_f_x_sq_minus_2_l286_286836


namespace max_value_b_exists_l286_286925

theorem max_value_b_exists :
  ∃ a c : ℝ, ∃ b : ℝ, 
  (∀ x : ℤ, 
  ((x^4 - a * x^3 - b * x^2 - c * x - 2007) = 0) → 
  ∃ r s t : ℤ, r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
  ((x = r) ∨ (x = s) ∨ (x = t))) ∧ 
  (∀ b' : ℝ, b' < b → 
  ¬ ( ∃ a' c' : ℝ, ( ∀ x : ℤ, 
  ((x^4 - a' * x^3 - b' * x^2 - c' * x - 2007) = 0) → 
  ∃ r' s' t' : ℤ, r' ≠ s' ∧ s' ≠ t' ∧ r' ≠ t' ∧ 
  ((x = r') ∨ (x = s') ∨ (x = t') )))) ∧ b = 3343 :=
sorry

end max_value_b_exists_l286_286925


namespace quadratic_roots_satisfy_condition_l286_286492
variable (x1 x2 m : ℝ)

theorem quadratic_roots_satisfy_condition :
  ( ∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1 + x2 = -m) ∧ 
    (x1 * x2 = 5) ∧ (x1 = 2 * |x2| - 3) ) →
  m = -9 / 2 :=
by
  sorry

end quadratic_roots_satisfy_condition_l286_286492


namespace rectangle_area_coefficient_l286_286622

theorem rectangle_area_coefficient (length width d k : ℝ) 
(h1 : length / width = 5 / 2) 
(h2 : d^2 = length^2 + width^2) 
(h3 : k = 10 / 29) :
  (length * width = k * d^2) :=
by
  sorry

end rectangle_area_coefficient_l286_286622


namespace jessica_deposit_fraction_l286_286982

-- Definitions based on conditions
variable (initial_balance : ℝ)
variable (fraction_withdrawn : ℝ) (withdrawn_amount : ℝ)
variable (final_balance remaining_balance fraction_deposit : ℝ)

-- Conditions
def conditions := 
  fraction_withdrawn = 2 / 5 ∧
  withdrawn_amount = 400 ∧
  remaining_balance = initial_balance - withdrawn_amount ∧
  remaining_balance = initial_balance * (1 - fraction_withdrawn) ∧
  final_balance = 750 ∧
  final_balance = remaining_balance + fraction_deposit * remaining_balance

-- The proof problem
theorem jessica_deposit_fraction : 
  conditions initial_balance fraction_withdrawn withdrawn_amount final_balance remaining_balance fraction_deposit →
  fraction_deposit = 1 / 4 :=
by
  intro h
  sorry

end jessica_deposit_fraction_l286_286982


namespace find_y_payment_l286_286096

-- Defining the conditions
def total_payment : ℝ := 700
def x_payment (y_payment : ℝ) : ℝ := 1.2 * y_payment

-- The theorem we want to prove
theorem find_y_payment (y_payment : ℝ) (h1 : y_payment + x_payment y_payment = total_payment) :
  y_payment = 318.18 := 
sorry

end find_y_payment_l286_286096


namespace find_white_balls_l286_286750

noncomputable def white_balls_in_bag (total_balls : ℕ) (green_balls : ℕ) (yellow_balls : ℕ) (red_balls : ℕ) (purple_balls : ℕ) 
  (p_not_red_nor_purple : ℚ) : ℕ :=
total_balls - (red_balls + purple_balls) - (green_balls + yellow_balls)

theorem find_white_balls :
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  white_balls_in_bag total_balls green_balls yellow_balls red_balls purple_balls p_not_red_nor_purple = 21 :=
by
  let total_balls := 60
  let green_balls := 18
  let yellow_balls := 17
  let red_balls := 3
  let purple_balls := 1
  let p_not_red_nor_purple := 0.95
  sorry

end find_white_balls_l286_286750


namespace not_possible_1006_2012_gons_l286_286979

theorem not_possible_1006_2012_gons :
  ∀ (n : ℕ), (∀ (k : ℕ), k ≤ 2011 → 2 * n ≤ k) → n ≠ 1006 :=
by
  intro n h
  -- Here goes the skipped proof part
  sorry

end not_possible_1006_2012_gons_l286_286979


namespace intersection_of_A_and_B_l286_286669

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = expected_intersection :=
by
  sorry

end intersection_of_A_and_B_l286_286669


namespace total_cards_l286_286933

def basketball_boxes : ℕ := 12
def cards_per_basketball_box : ℕ := 20
def football_boxes : ℕ := basketball_boxes - 5
def cards_per_football_box : ℕ := 25

theorem total_cards : basketball_boxes * cards_per_basketball_box + football_boxes * cards_per_football_box = 415 := by
  sorry

end total_cards_l286_286933


namespace sum_of_intercepts_l286_286305

theorem sum_of_intercepts (x y : ℝ) (h : 3 * x - 4 * y - 12 = 0) :
    (y = -3 ∧ x = 4) → x + y = 1 :=
by
  intro h'
  obtain ⟨hy, hx⟩ := h'
  rw [hy, hx]
  norm_num
  done

end sum_of_intercepts_l286_286305


namespace max_dogs_and_fish_l286_286683

theorem max_dogs_and_fish (d c b p f : ℕ) (h_ratio : d / 7 = c / 7 ∧ d / 7 = b / 8 ∧ d / 7 = p / 3 ∧ d / 7 = f / 5)
  (h_dogs_bunnies : d + b = 330)
  (h_twice_fish : f ≥ 2 * c) :
  d = 154 ∧ f = 308 :=
by
  -- This is where the proof would go
  sorry

end max_dogs_and_fish_l286_286683


namespace gcd_40_56_l286_286248

theorem gcd_40_56 : Int.gcd 40 56 = 8 :=
by
  sorry

end gcd_40_56_l286_286248


namespace can_capacity_is_14_l286_286197

noncomputable def capacity_of_can 
    (initial_milk: ℝ) (initial_water: ℝ) 
    (added_milk: ℝ) (ratio_initial: ℝ) (ratio_final: ℝ): ℝ :=
  initial_milk + initial_water + added_milk

theorem can_capacity_is_14
    (M W: ℝ) 
    (ratio_initial : M / W = 1 / 5) 
    (added_milk : ℝ := 2) 
    (ratio_final:  (M + 2) / W = 2.00001 / 5.00001): 
    capacity_of_can M W added_milk (1 / 5) (2.00001 / 5.00001) = 14 := 
  by
    sorry

end can_capacity_is_14_l286_286197


namespace derivative_y_eq_l286_286479

noncomputable def y (x : ℝ) : ℝ := 
  (3 / 2) * Real.log (Real.tanh (x / 2)) + Real.cosh x - (Real.cosh x) / (2 * (Real.sinh x)^2)

theorem derivative_y_eq :
  (deriv y x) = (Real.cosh x)^4 / (Real.sinh x)^3 :=
sorry

end derivative_y_eq_l286_286479


namespace ratio_of_crates_l286_286018

/-
  Gabrielle sells eggs. On Monday she sells 5 crates of eggs. On Tuesday she sells 2 times as many
  crates of eggs as Monday. On Wednesday she sells 2 fewer crates than Tuesday. On Thursday she sells
  some crates of eggs. She sells a total of 28 crates of eggs for the 4 days. Prove the ratio of the 
  number of crates she sells on Thursday to the number she sells on Tuesday is 1/2.
-/

theorem ratio_of_crates 
    (mon_crates : ℕ) 
    (tue_crates : ℕ) 
    (wed_crates : ℕ) 
    (thu_crates : ℕ) 
    (total_crates : ℕ) 
    (h_mon : mon_crates = 5) 
    (h_tue : tue_crates = 2 * mon_crates) 
    (h_wed : wed_crates = tue_crates - 2) 
    (h_total : total_crates = mon_crates + tue_crates + wed_crates + thu_crates) 
    (h_total_val : total_crates = 28): 
  (thu_crates / tue_crates : ℚ) = 1 / 2 := 
by 
  sorry

end ratio_of_crates_l286_286018


namespace max_popsicles_l286_286224

theorem max_popsicles (budget : ℕ) (cost_single : ℕ) (popsicles_single : ℕ) (cost_box3 : ℕ) (popsicles_box3 : ℕ) (cost_box7 : ℕ) (popsicles_box7 : ℕ)
  (h_budget : budget = 10) (h_cost_single : cost_single = 1) (h_popsicles_single : popsicles_single = 1)
  (h_cost_box3 : cost_box3 = 3) (h_popsicles_box3 : popsicles_box3 = 3)
  (h_cost_box7 : cost_box7 = 4) (h_popsicles_box7 : popsicles_box7 = 7) :
  ∃ n, n = 16 :=
by
  sorry

end max_popsicles_l286_286224


namespace inverse_function_correct_l286_286303

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1) ^ 2 + 1

noncomputable def f_inv (y : ℝ) : ℝ :=
  1 - Real.sqrt (y - 1)

theorem inverse_function_correct (x : ℝ) (hx : x ≥ 2) :
  f_inv x = 1 - Real.sqrt (x - 1) ∧ ∀ y : ℝ, (y ≤ 0) → f y = x → y = f_inv x :=
by {
  sorry
}

end inverse_function_correct_l286_286303


namespace time_after_2345_minutes_l286_286253

-- Define the constants
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24
def startTime : Nat := 0 -- midnight on January 1, 2022, treated as 0 minutes.

-- Prove the equivalent time after 2345 minutes
theorem time_after_2345_minutes :
    let totalMinutes := 2345
    let totalHours := totalMinutes / minutesInHour
    let remainingMinutes := totalMinutes % minutesInHour
    let totalDays := totalHours / hoursInDay
    let remainingHours := totalHours % hoursInDay
    startTime + totalDays * hoursInDay * minutesInHour + remainingHours * minutesInHour + remainingMinutes = startTime + 1 * hoursInDay * minutesInHour + 15 * minutesInHour + 5 :=
    by
    sorry

end time_after_2345_minutes_l286_286253


namespace find_m_l286_286489

theorem find_m (m x1 x2 : ℝ) (h1 : x1^2 + m * x1 + 5 = 0) (h2 : x2^2 + m * x2 + 5 = 0) (h3 : x1 = 2 * |x2| - 3) : 
  m = -9 / 2 :=
sorry

end find_m_l286_286489


namespace expression_simplification_l286_286474

theorem expression_simplification : 2 + 1 / (3 + 1 / (2 + 2)) = 30 / 13 := 
by 
  sorry

end expression_simplification_l286_286474


namespace nomogram_relation_l286_286287

noncomputable def root_of_eq (x p q : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem nomogram_relation (x p q : ℝ) (hx : root_of_eq x p q) : 
  q = -x * p - x^2 :=
by 
  sorry

end nomogram_relation_l286_286287


namespace other_root_correct_l286_286562

noncomputable def other_root (p : ℝ) : ℝ :=
  let a := 3
  let c := -2
  let root1 := -1
  (-c / a) / root1

theorem other_root_correct (p : ℝ) (h_eq : 3 * (-1) ^ 2 + p * (-1) = 2) : other_root p = 2 / 3 :=
  by
    unfold other_root
    sorry

end other_root_correct_l286_286562


namespace iodine_solution_problem_l286_286813

theorem iodine_solution_problem (init_concentration : Option ℝ) (init_volume : ℝ)
  (final_concentration : ℝ) (added_volume : ℝ) : 
  init_concentration = none 
  → ∃ x : ℝ, init_volume + added_volume = x :=
by
  sorry

end iodine_solution_problem_l286_286813


namespace factorial_plus_one_div_prime_l286_286714

theorem factorial_plus_one_div_prime (n : ℕ) (h : (n! + 1) % (n + 1) = 0) : Nat.Prime (n + 1) := 
sorry

end factorial_plus_one_div_prime_l286_286714


namespace hundredth_odd_integer_l286_286415

theorem hundredth_odd_integer : ∃ (x : ℕ), 2 * x - 1 = 199 ∧ x = 100 :=
by
  use 100
  split
  . exact calc
      2 * 100 - 1 = 200 - 1 : by ring
      _ = 199 : by norm_num
  . refl

end hundredth_odd_integer_l286_286415


namespace infection_equation_l286_286445

-- Given conditions
def initially_infected : Nat := 1
def total_after_two_rounds : ℕ := 81
def avg_infect_per_round (x : ℕ) : ℕ := x

-- Mathematically equivalent proof problem
theorem infection_equation (x : ℕ) 
  (h1 : initially_infected = 1)
  (h2 : total_after_two_rounds = 81)
  (h3 : ∀ (y : ℕ), initially_infected + avg_infect_per_round y + (avg_infect_per_round y)^2 = total_after_two_rounds):
  (1 + x)^2 = 81 :=
by
  sorry

end infection_equation_l286_286445


namespace power_neg_two_inverse_l286_286749

theorem power_neg_two_inverse : (-2 : ℤ) ^ (-2 : ℤ) = (1 : ℚ) / (4 : ℚ) := by
  -- Condition: a^{-n} = 1 / a^n for any non-zero number a and any integer n
  have h: ∀ (a : ℚ) (n : ℤ), a ≠ 0 → a ^ (-n) = 1 / a ^ n := sorry
  -- Proof goes here
  sorry

end power_neg_two_inverse_l286_286749


namespace problem1_range_of_x_problem2_value_of_a_l286_286321

open Set

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |x + 3| + |x - a|

-- Problem 1
theorem problem1_range_of_x (a : ℝ) (h : a = 4) (h_eq : ∀ x : ℝ, f x a = 7 ↔ x ∈ Icc (-3 : ℝ) 4) :
  ∀ x : ℝ, f x 4 = 7 ↔ x ∈ Icc (-3 : ℝ) 4 := by
  sorry

-- Problem 2
theorem problem2_value_of_a (h₁ : ∀ x : ℝ, x ∈ {x : ℝ | f x 4 ≥ 6} ↔ x ≤ -4 ∨ x ≥ 2) :
  f x a ≥ 6 ↔  x ≤ -4 ∨ x ≥ 2 :=
  by
  sorry

end problem1_range_of_x_problem2_value_of_a_l286_286321


namespace hundredth_odd_integer_l286_286419

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l286_286419


namespace store_revenue_after_sale_l286_286128

/--
A store has 2000 items, each normally selling for $50. 
They offer an 80% discount and manage to sell 90% of the items. 
The store owes $15,000 to creditors. Prove that the store has $3,000 left after the sale.
-/
theorem store_revenue_after_sale :
  let items := 2000
  let retail_price := 50
  let discount := 0.8
  let sale_percentage := 0.9
  let debt := 15000
  let items_sold := items * sale_percentage
  let discount_amount := retail_price * discount
  let sale_price_per_item := retail_price - discount_amount
  let total_revenue := items_sold * sale_price_per_item
  let money_left := total_revenue - debt
  money_left = 3000 :=
by
  sorry

end store_revenue_after_sale_l286_286128


namespace parabola_one_intersection_l286_286188

theorem parabola_one_intersection (k : ℝ) :
  (∀ x : ℝ, x^2 - x + k = 0 → x = 0) → k = 1 / 4 :=
sorry

end parabola_one_intersection_l286_286188


namespace train_cross_platform_time_l286_286437

def train_length : ℝ := 300
def platform_length : ℝ := 550
def signal_pole_time : ℝ := 18

theorem train_cross_platform_time :
  let speed : ℝ := train_length / signal_pole_time
  let total_distance : ℝ := train_length + platform_length
  let crossing_time : ℝ := total_distance / speed
  crossing_time = 51 :=
by
  sorry

end train_cross_platform_time_l286_286437


namespace smallest_integer_value_l286_286005

theorem smallest_integer_value (x : ℤ) (h : 3 * |x| + 8 < 29) : x = -6 :=
sorry

end smallest_integer_value_l286_286005


namespace number_of_functions_with_given_range_l286_286639

theorem number_of_functions_with_given_range : 
  let S := {2, 5, 10}
  let R (x : ℤ) := x^2 + 1
  ∃ f : ℤ → ℤ, (∀ y ∈ S, ∃ x : ℤ, f x = y) ∧ (f '' {x | R x ∈ S} = S) :=
    sorry

end number_of_functions_with_given_range_l286_286639


namespace principal_amount_l286_286277

theorem principal_amount (A2 A3 : ℝ) (interest : ℝ) (principal : ℝ) (h1 : A2 = 3450) 
  (h2 : A3 = 3655) (h_interest : interest = A3 - A2) (h_principal : principal = A2 - interest) : 
  principal = 3245 :=
by
  sorry

end principal_amount_l286_286277


namespace quadratic_congruence_solution_l286_286721

theorem quadratic_congruence_solution (p : ℕ) (hp : Nat.Prime p) : 
  ∃ n : ℕ, 6 * n^2 + 5 * n + 1 ≡ 0 [MOD p] := 
sorry

end quadratic_congruence_solution_l286_286721


namespace minimum_value_f_on_neg_ab_l286_286805

theorem minimum_value_f_on_neg_ab
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : a < b)
  (h2 : b < 0)
  (odd_f : ∀ x : ℝ, f (-x) = -f (x))
  (decreasing_f : ∀ x y : ℝ, 0 < x ∧ x < y → f y < f x)
  (range_ab : ∀ y : ℝ, a ≤ y ∧ y ≤ b → -3 ≤ f y ∧ f y ≤ 4) :
  ∀ x : ℝ, -b ≤ x ∧ x ≤ -a → -4 ≤ f x ∧ f x ≤ 3 := 
sorry

end minimum_value_f_on_neg_ab_l286_286805


namespace senate_seating_l286_286438

-- Definitions for the problem
def num_ways_of_seating (num_democrats : ℕ) (num_republicans : ℕ) : ℕ :=
  if h : num_democrats = 6 ∧ num_republicans = 4 then
    5! * (finset.card (finset.powerset_len 4 (finset.range 6))) * 4!
  else
    0

-- The proof statement
theorem senate_seating : num_ways_of_seating 6 4 = 43200 :=
by {
  -- Placeholder for proof
  sorry
}

end senate_seating_l286_286438


namespace selection_count_l286_286440

theorem selection_count :
  let english_only := 3
  let japanese_only := 2
  let bilingual := 2
  let total_english_ways := (Nat.choose 3 3) + (Nat.choose 3 2 * Nat.choose 2 1)
  let total_japanese_ways := (Nat.choose 2 2) + (Nat.choose 2 1 * Nat.choose 2 1)
  let total_selection_ways := total_english_ways * total_japanese_ways
  total_selection_ways = 27 :=
by
  sorry

end selection_count_l286_286440


namespace range_of_a_l286_286993

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (2 * a + 1) * x + a^2 + a < 0 → 0 < 2 * x - 1 ∧ 2 * x - 1 ≤ 10) →
  (∃ l u : ℝ, (l = 1/2) ∧ (u = 9/2) ∧ (l ≤ a ∧ a ≤ u)) :=
by
  sorry

end range_of_a_l286_286993


namespace domain_of_f_decreasing_on_interval_range_of_f_l286_286666

noncomputable def f (x : ℝ) : ℝ := Real.log (3 + 2 * x - x^2) / Real.log 2

theorem domain_of_f :
  ∀ x : ℝ, (3 + 2 * x - x^2 > 0) ↔ (-1 < x ∧ x < 3) :=
by
  sorry

theorem decreasing_on_interval :
  ∀ (x₁ x₂ : ℝ), (1 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3) →
  f x₂ < f x₁ :=
by
  sorry

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, -1 < x ∧ x < 3 ∧ y = f x) ↔ y ≤ 2 :=
by
  sorry

end domain_of_f_decreasing_on_interval_range_of_f_l286_286666


namespace union_of_sets_l286_286957

noncomputable def set_A : Set ℚ := {7, -1/3}
noncomputable def set_B : Set ℚ := {8/3, -1/3}

theorem union_of_sets :
  (set_A ∪ set_B) = {7, 8/3, -1/3} :=
by
  sorry

end union_of_sets_l286_286957


namespace max_xy_on_line_AB_l286_286313

noncomputable def pointA : ℝ × ℝ := (3, 0)
noncomputable def pointB : ℝ × ℝ := (0, 4)

-- Define the line passing through points A and B
def on_line_AB (P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, P.1 = 3 - 3 * t ∧ P.2 = 4 * t

theorem max_xy_on_line_AB : ∃ (P : ℝ × ℝ), on_line_AB P ∧ P.1 * P.2 = 3 := 
sorry

end max_xy_on_line_AB_l286_286313


namespace percent_sugar_in_resulting_solution_l286_286373

theorem percent_sugar_in_resulting_solution (W : ℝ) (hW : W > 0) :
  let original_sugar_percent := 22 / 100
  let second_solution_sugar_percent := 74 / 100
  let remaining_original_weight := (3 / 4) * W
  let removed_weight := (1 / 4) * W
  let sugar_from_remaining_original := (original_sugar_percent * remaining_original_weight)
  let sugar_from_added_second_solution := (second_solution_sugar_percent * removed_weight)
  let total_sugar := sugar_from_remaining_original + sugar_from_added_second_solution
  let resulting_sugar_percent := total_sugar / W
  resulting_sugar_percent = 35 / 100 :=
by
  sorry

end percent_sugar_in_resulting_solution_l286_286373


namespace triathlon_bike_speed_l286_286973

theorem triathlon_bike_speed :
  ∀ (t_total t_swim t_run t_bike : ℚ) (d_swim d_run d_bike : ℚ)
    (v_swim v_run r_bike : ℚ),
  t_total = 3 →
  d_swim = 1 / 2 →
  v_swim = 1 →
  d_run = 4 →
  v_run = 5 →
  d_bike = 10 →
  t_swim = d_swim / v_swim →
  t_run = d_run / v_run →
  t_bike = t_total - (t_swim + t_run) →
  r_bike = d_bike / t_bike →
  r_bike = 100 / 17 :=
by
  intros t_total t_swim t_run t_bike d_swim d_run d_bike v_swim v_run r_bike
         h_total h_d_swim h_v_swim h_d_run h_v_run h_d_bike h_t_swim h_t_run h_t_bike h_r_bike
  sorry

end triathlon_bike_speed_l286_286973


namespace girls_more_than_boys_l286_286198

theorem girls_more_than_boys (boys girls : ℕ) (ratio_boys ratio_girls : ℕ) 
  (h1 : ratio_boys = 5)
  (h2 : ratio_girls = 13)
  (h3 : boys = 50)
  (h4 : girls = (boys / ratio_boys) * ratio_girls) : 
  girls - boys = 80 :=
by
  sorry

end girls_more_than_boys_l286_286198


namespace min_value_x_y_l286_286801

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 / x + 8 / y = 1) : x + y ≥ 18 := 
sorry

end min_value_x_y_l286_286801


namespace interval_intersection_l286_286636

theorem interval_intersection (x : ℝ) : 
  (1 < 4 * x ∧ 4 * x < 3) ∧ (2 < 6 * x ∧ 6 * x < 4) ↔ (1 / 3 < x ∧ x < 2 / 3) := 
by 
  sorry

end interval_intersection_l286_286636


namespace rectangle_area_l286_286759

-- Definitions
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)
def length (w : ℝ) : ℝ := 2 * w
def area (l w : ℝ) : ℝ := l * w

-- Main Statement
theorem rectangle_area (w l : ℝ) (h_p : perimeter l w = 120) (h_l : l = length w) :
  area l w = 800 :=
by
  sorry

end rectangle_area_l286_286759


namespace grace_hours_pulling_weeds_l286_286327

variable (Charge_mowing : ℕ) (Charge_weeding : ℕ) (Charge_mulching : ℕ)
variable (H_m : ℕ) (H_u : ℕ) (E_s : ℕ)

theorem grace_hours_pulling_weeds 
  (Charge_mowing_eq : Charge_mowing = 6)
  (Charge_weeding_eq : Charge_weeding = 11)
  (Charge_mulching_eq : Charge_mulching = 9)
  (H_m_eq : H_m = 63)
  (H_u_eq : H_u = 10)
  (E_s_eq : E_s = 567) :
  ∃ W : ℕ, 6 * 63 + 11 * W + 9 * 10 = 567 ∧ W = 9 := by
  sorry

end grace_hours_pulling_weeds_l286_286327


namespace randy_money_left_after_expenses_l286_286077

theorem randy_money_left_after_expenses : 
  ∀ (initial_money lunch_cost : ℕ) (ice_cream_fraction : ℚ), 
  initial_money = 30 → 
  lunch_cost = 10 → 
  ice_cream_fraction = 1 / 4 → 
  let post_lunch_money := initial_money - lunch_cost in
  let ice_cream_cost := ice_cream_fraction * post_lunch_money in
  let money_left := post_lunch_money - ice_cream_cost in
  money_left = 15 :=
by
  intros initial_money lunch_cost ice_cream_fraction
  assume h_initial h_lunch h_fraction
  let post_lunch_money := initial_money - lunch_cost
  let ice_cream_cost := ice_cream_fraction * post_lunch_money
  let money_left := post_lunch_money - ice_cream_cost
  sorry

end randy_money_left_after_expenses_l286_286077


namespace find_k_l286_286667

noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem find_k (k : ℝ) (h_pos : 0 < k) (h_exists : ∃ x₀ : ℝ, 1 ≤ x₀ ∧ g x₀ ≤ k * (-x₀^2 + 3 * x₀)) : 
  k > (1 / 2) * (Real.exp 1 + 1 / Real.exp 1) :=
sorry

end find_k_l286_286667


namespace range_of_a_l286_286183

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → a * x^2 - x - 4 > 0) → a > 5 :=
by
  sorry

end range_of_a_l286_286183


namespace store_money_left_after_sale_l286_286129

theorem store_money_left_after_sale :
  ∀ (n p : ℕ) (d s : ℝ) (debt : ℕ), 
  n = 2000 → 
  p = 50 → 
  d = 0.80 → 
  s = 0.90 → 
  debt = 15000 → 
  (nat.floor (s * n) * (p - nat.floor (d * p)) : ℕ) - debt = 3000 :=
by
  intros n p d s debt hn hp hd hs hdebt
  sorry

end store_money_left_after_sale_l286_286129


namespace stamp_solutions_l286_286653

theorem stamp_solutions (n : ℕ) (h1 : ∀ (k : ℕ), k < 115 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) 
  (h2 : ¬ ∃ (a b c : ℕ), 3 * a + n * b + (n + 1) * c = 115) 
  (h3 : ∀ (k : ℕ), 116 ≤ k ∧ k ≤ 120 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) : 
  n = 59 :=
sorry

end stamp_solutions_l286_286653


namespace andrea_still_needs_rhinestones_l286_286454

def total_rhinestones_needed : ℕ := 45
def rhinestones_bought : ℕ := total_rhinestones_needed / 3
def rhinestones_found : ℕ := total_rhinestones_needed / 5
def rhinestones_total_have : ℕ := rhinestones_bought + rhinestones_found
def rhinestones_still_needed : ℕ := total_rhinestones_needed - rhinestones_total_have

theorem andrea_still_needs_rhinestones : rhinestones_still_needed = 21 := by
  rfl

end andrea_still_needs_rhinestones_l286_286454


namespace slope_range_l286_286400

theorem slope_range (k : ℝ) : 
  (∃ (x : ℝ), ∀ (y : ℝ), y = k * (x - 1) + 1) ∧ (0 < 1 - k ∧ 1 - k < 2) → (-1 < k ∧ k < 1) :=
by
  sorry

end slope_range_l286_286400


namespace find_side_b_l286_286207

theorem find_side_b
  (a b c : ℝ)
  (area : ℝ)
  (B : ℝ)
  (h_area : area = sqrt 3)
  (h_B : B = real.pi / 3)
  (h_a2c2 : a^2 + c^2 = 3 * a * c)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) :
  b = 2 * real.sqrt 2 :=
sorry

end find_side_b_l286_286207


namespace gain_percentage_is_8_l286_286624

variable (C S : ℝ) (D : ℝ)
variable (h1 : 20 * C * (1 - D / 100) = 12 * S)
variable (h2 : D ≥ 5 ∧ D ≤ 25)

theorem gain_percentage_is_8 :
  (12 * S * 1.08 - 20 * C * (1 - D / 100)) / (20 * C * (1 - D / 100)) * 100 = 8 :=
by
  sorry

end gain_percentage_is_8_l286_286624


namespace matching_pair_probability_l286_286589

-- Given conditions
def total_gray_socks : ℕ := 12
def total_white_socks : ℕ := 10
def total_socks : ℕ := total_gray_socks + total_white_socks

-- Proof statement
theorem matching_pair_probability (h_grays : total_gray_socks = 12) (h_whites : total_white_socks = 10) :
  (66 + 45) / (total_socks.choose 2) = 111 / 231 :=
by
  sorry

end matching_pair_probability_l286_286589


namespace problem_statement_l286_286509

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l286_286509


namespace day_after_exponential_days_l286_286590

noncomputable def days_since_monday (n : ℕ) : String :=
  let days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  days.get! (n % 7)

theorem day_after_exponential_days :
  days_since_monday (2^20) = "Friday" :=
by
  sorry

end day_after_exponential_days_l286_286590


namespace problem_solution_l286_286506

-- Definitions based on conditions given in the problem statement
def validExpression (n : ℕ) : ℕ := 
  sorry -- Placeholder for function defining valid expressions

def T (n : ℕ) : ℕ := 
  if n = 1 then 1 
  else validExpression n

def R (n : ℕ) : ℕ := T n % 4

def computeSum (k : ℕ) : ℕ := 
  (List.range k).map R |>.sum

-- Lean theorem statement to be proven
theorem problem_solution : 
  computeSum 1000001 = 320 := 
sorry

end problem_solution_l286_286506


namespace safe_paths_count_l286_286050

theorem safe_paths_count :
  let total_paths := Multinomial (4, 4, 4)
  let paths_through_mine := Multinomial (2, 2, 2) * Multinomial (2, 2, 2)
  let paths_near_mine := 6 * Multinomial (3, 2, 1) * Multinomial (3, 2, 1)
  let paths_two_units_away := 6 * Multinomial (3, 2, 0) * Multinomial (1, 2, 4)
  let safe_paths := total_paths - paths_through_mine - paths_near_mine + paths_two_units_away
  in
  total_paths = 10395 ∧ paths_through_mine = 225 ∧ paths_near_mine = 540 ∧ paths_two_units_away = 270 ∧ safe_paths = 9900 :=
begin
  sorry
end

end safe_paths_count_l286_286050


namespace product_of_integer_with_100_l286_286266

theorem product_of_integer_with_100 (x : ℝ) (h : 10 * x = x + 37.89) : 100 * x = 421 :=
by
  -- insert the necessary steps to solve the problem
  sorry

end product_of_integer_with_100_l286_286266


namespace total_wheels_correct_l286_286537

-- Define the initial state of the garage
def initial_bicycles := 20
def initial_cars := 10
def initial_motorcycles := 5
def initial_tricycles := 3
def initial_quads := 2

-- Define the changes in the next hour
def bicycles_leaving := 7
def cars_arriving := 4
def motorcycles_arriving := 3
def motorcycles_leaving := 2

-- Define the damaged vehicles
def damaged_bicycles := 5  -- each missing 1 wheel
def damaged_cars := 2      -- each missing 1 wheel
def damaged_motorcycle := 1 -- missing 2 wheels

-- Define the number of wheels per type of vehicle
def bicycle_wheels := 2
def car_wheels := 4
def motorcycle_wheels := 2
def tricycle_wheels := 3
def quad_wheels := 4

-- Calculate the state of vehicles at the end of the hour
def final_bicycles := initial_bicycles - bicycles_leaving
def final_cars := initial_cars + cars_arriving
def final_motorcycles := initial_motorcycles + motorcycles_arriving - motorcycles_leaving

-- Calculate the total wheels in the garage at the end of the hour
def total_wheels : Nat := 
  (final_bicycles - damaged_bicycles) * bicycle_wheels + damaged_bicycles +
  (final_cars - damaged_cars) * car_wheels + damaged_cars * 3 +
  (final_motorcycles - damaged_motorcycle) * motorcycle_wheels +
  initial_tricycles * tricycle_wheels +
  initial_quads * quad_wheels

-- The goal is to prove that the total number of wheels in the garage is 102 at the end of the hour
theorem total_wheels_correct : total_wheels = 102 := 
  by
    sorry

end total_wheels_correct_l286_286537


namespace giant_spider_weight_ratio_l286_286268

theorem giant_spider_weight_ratio 
    (W_previous : ℝ)
    (A_leg : ℝ)
    (P : ℝ)
    (n : ℕ)
    (W_previous_eq : W_previous = 6.4)
    (A_leg_eq : A_leg = 0.5)
    (P_eq : P = 4)
    (n_eq : n = 8):
    (P * A_leg * n) / W_previous = 2.5 := by
  sorry

end giant_spider_weight_ratio_l286_286268


namespace correct_answer_is_A_l286_286429

-- Definitions derived from problem conditions
def algorithm := Type
def has_sequential_structure (alg : algorithm) : Prop := sorry -- Actual definition should define what a sequential structure is for an algorithm

-- Given: An algorithm must contain a sequential structure.
theorem correct_answer_is_A (alg : algorithm) : has_sequential_structure alg :=
sorry

end correct_answer_is_A_l286_286429


namespace total_cost_sean_bought_l286_286385

theorem total_cost_sean_bought (cost_soda cost_soup cost_sandwich : ℕ) 
  (h_soda : cost_soda = 1)
  (h_soup : cost_soup = 3 * cost_soda)
  (h_sandwich : cost_sandwich = 3 * cost_soup) :
  3 * cost_soda + 2 * cost_soup + cost_sandwich = 18 := 
by
  sorry

end total_cost_sean_bought_l286_286385


namespace minFuseLength_l286_286538

namespace EarthquakeRelief

def fuseLengthRequired (distanceToSafety : ℕ) (speedOperator : ℕ) (burningSpeed : ℕ) (lengthFuse : ℕ) : Prop :=
  (lengthFuse : ℝ) / (burningSpeed : ℝ) > (distanceToSafety : ℝ) / (speedOperator : ℝ)

theorem minFuseLength 
  (distanceToSafety : ℕ := 400) 
  (speedOperator : ℕ := 5) 
  (burningSpeed : ℕ := 12) : 
  ∀ lengthFuse: ℕ, 
  fuseLengthRequired distanceToSafety speedOperator burningSpeed lengthFuse → lengthFuse > 96 := 
by
  sorry

end EarthquakeRelief

end minFuseLength_l286_286538


namespace eve_spending_l286_286644

-- Definitions of the conditions
def cost_mitt : ℝ := 14.00
def cost_apron : ℝ := 16.00
def cost_utensils : ℝ := 10.00
def cost_knife : ℝ := 2 * cost_utensils -- Twice the amount of the utensils
def discount_rate : ℝ := 0.25
def num_nieces : ℝ := 3

-- Total cost before the discount for one kit
def total_cost_one_kit : ℝ :=
  cost_mitt + cost_apron + cost_utensils + cost_knife

-- Discount for one kit
def discount_one_kit : ℝ := 
  total_cost_one_kit * discount_rate

-- Discounted price for one kit
def discounted_cost_one_kit : ℝ :=
  total_cost_one_kit - discount_one_kit

-- Total cost for all kits
def total_cost_all_kits : ℝ :=
  num_nieces * discounted_cost_one_kit

-- The theorem statement
theorem eve_spending : total_cost_all_kits = 135.00 :=
by sorry

end eve_spending_l286_286644


namespace weight_of_b_l286_286110

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 126) (h2 : a + b = 80) (h3 : b + c = 86) : b = 40 :=
sorry

end weight_of_b_l286_286110


namespace a_fraction_of_capital_l286_286768

theorem a_fraction_of_capital (T : ℝ) (B : ℝ) (C : ℝ) (D : ℝ)
  (profit_A : ℝ) (total_profit : ℝ)
  (h1 : B = T * (1 / 4))
  (h2 : C = T * (1 / 5))
  (h3 : D = T - (T * (1 / 4) + T * (1 / 5) + T * x))
  (h4 : profit_A = 805)
  (h5 : total_profit = 2415) :
  x = 161 / 483 :=
by
  sorry

end a_fraction_of_capital_l286_286768


namespace convert_1729_to_base_5_l286_286929

theorem convert_1729_to_base_5 :
  let d := 1729
  let b := 5
  let representation := [2, 3, 4, 0, 4]
  -- Check the representation of 1729 in base 5
  d = (representation.reverse.enum_from 0).sum (fun ⟨i, coef⟩ => coef * b^i) :=
  sorry

end convert_1729_to_base_5_l286_286929


namespace ezekiel_first_day_distance_l286_286475

noncomputable def distance_first_day (total_distance second_day_distance third_day_distance : ℕ) :=
  total_distance - (second_day_distance + third_day_distance)

theorem ezekiel_first_day_distance:
  ∀ (total_distance second_day_distance third_day_distance : ℕ),
  total_distance = 50 →
  second_day_distance = 25 →
  third_day_distance = 15 →
  distance_first_day total_distance second_day_distance third_day_distance = 10 :=
by
  intros total_distance second_day_distance third_day_distance h1 h2 h3
  sorry

end ezekiel_first_day_distance_l286_286475


namespace fraction_equality_l286_286340

variable (a b : ℚ)

theorem fraction_equality (h : (4 * a + 3 * b) / (4 * a - 3 * b) = 4) : a / b = 5 / 4 := by
  sorry

end fraction_equality_l286_286340


namespace polygon_sides_l286_286192

theorem polygon_sides (n : ℕ) (h_interior : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end polygon_sides_l286_286192


namespace power_six_rectangular_form_l286_286141

noncomputable def sin (x : ℂ) : ℂ := (Complex.exp (-Complex.I * x) - Complex.exp (Complex.I * x)) / (2 * Complex.I)
noncomputable def cos (x : ℂ) : ℂ := (Complex.exp (Complex.I * x) + Complex.exp (-Complex.I * x)) / 2

theorem power_six_rectangular_form :
  (2 * cos (20 * Real.pi / 180) + 2 * Complex.I * sin (20 * Real.pi / 180))^6 = -32 + 32 * Complex.I * Real.sqrt 3 := sorry

end power_six_rectangular_form_l286_286141


namespace range_of_a_l286_286347

noncomputable def f (x a : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1
def intersects_at_single_point (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
∃! x, f x a = 3

theorem range_of_a (a : ℝ) :
  intersects_at_single_point f a ↔ -1 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l286_286347


namespace count_valid_numbers_l286_286965

def is_valid_number (N : ℕ) : Prop :=
  let a := N / 1000
  let x := N % 1000
  (N = 1000 * a + x) ∧ (x = N / 8) ∧ (100 ≤ x) ∧ (x < 1000) ∧ (1 ≤ a) ∧ (a ≤ 6)

theorem count_valid_numbers : (finset.filter is_valid_number (finset.range 10000)).card = 6 := 
  sorry

end count_valid_numbers_l286_286965


namespace percentage_increase_l286_286848

theorem percentage_increase (x : ℝ) (h1 : 75 + 0.75 * x * 0.8 = 72) : x = 20 :=
by
  sorry

end percentage_increase_l286_286848


namespace geometric_sequence_general_term_T_n_formula_l286_286401

noncomputable def a (n : ℕ) : ℤ :=
  if n = 0 then 4 else (-2)^(n+1)

def b (n : ℕ) : ℤ :=
  nat.log 2 (abs (a n))

def T (n : ℕ) : ℚ :=
  (range n).sum (λ k, 1 / ((b k) * (b (k + 1))))

theorem geometric_sequence_general_term :
  ∀ n : ℕ, a n = (-2)^(n+1) := 
by
  sorry

theorem T_n_formula :
  ∀ n : ℕ, T n = n / (2 * (n + 2)) := 
by
  sorry

end geometric_sequence_general_term_T_n_formula_l286_286401


namespace penny_remaining_money_l286_286711

theorem penny_remaining_money (initial_money : ℤ) (socks_pairs : ℤ) (socks_cost_per_pair : ℤ) (hat_cost : ℤ) :
  initial_money = 20 → socks_pairs = 4 → socks_cost_per_pair = 2 → hat_cost = 7 → 
  initial_money - (socks_pairs * socks_cost_per_pair + hat_cost) = 5 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end penny_remaining_money_l286_286711


namespace solve_for_a_l286_286967

theorem solve_for_a (a x y : ℝ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 3) : a = 5 :=
by
  sorry

end solve_for_a_l286_286967


namespace smallest_lcm_four_digit_integers_with_gcd_five_l286_286331

open Nat

theorem smallest_lcm_four_digit_integers_with_gcd_five : ∃ k ℓ : ℕ, 1000 ≤ k ∧ k < 10000 ∧ 1000 ≤ ℓ ∧ ℓ < 10000 ∧ gcd k ℓ = 5 ∧ lcm k ℓ = 203010 :=
by
  use 1005
  use 1010
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_lcm_four_digit_integers_with_gcd_five_l286_286331


namespace intersection_M_N_l286_286056

def M := {1, 3, 5, 7, 9}

def N := {x : ℤ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} := by
  sorry

end intersection_M_N_l286_286056


namespace gcd_of_consecutive_digit_sums_l286_286402

theorem gcd_of_consecutive_digit_sums :
  ∀ x y z : ℕ, x + 1 = y → y + 1 = z → gcd (101 * (x + z) + 10 * y) 212 = 212 :=
by
  sorry

end gcd_of_consecutive_digit_sums_l286_286402


namespace wholesale_price_l286_286761

theorem wholesale_price (R : ℝ) (W : ℝ)
  (hR : R = 120)
  (h_discount : ∀ SP : ℝ, SP = R - (0.10 * R))
  (h_profit : ∀ P : ℝ, P = 0.20 * W)
  (h_SP_eq_W_P : ∀ SP P : ℝ, SP = W + P) :
  W = 90 := by
  sorry

end wholesale_price_l286_286761


namespace length_of_wooden_block_l286_286581

theorem length_of_wooden_block (cm_to_m : ℝ := 30 / 100) (base_length : ℝ := 31) :
  base_length + cm_to_m = 31.3 :=
by
  sorry

end length_of_wooden_block_l286_286581


namespace is_isosceles_triangle_l286_286808

theorem is_isosceles_triangle 
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a * Real.cos B + b * Real.cos C + c * Real.cos A = b * Real.cos A + c * Real.cos B + a * Real.cos C) : 
  (A = B ∨ B = C ∨ A = C) :=
sorry

end is_isosceles_triangle_l286_286808


namespace value_of_x_squared_add_reciprocal_squared_l286_286519

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l286_286519


namespace concert_attendance_difference_l286_286701

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := 66018

theorem concert_attendance_difference :
  (second_concert - first_concert) = 119 :=
by
  sorry

end concert_attendance_difference_l286_286701


namespace larger_factor_of_lcm_l286_286855

theorem larger_factor_of_lcm (A B : ℕ) (hcf lcm X Y : ℕ) 
  (h_hcf: hcf = 63)
  (h_A: A = 1071)
  (h_lcm: lcm = hcf * X * Y)
  (h_X: X = 11)
  (h_factors: ∃ k: ℕ, A = hcf * k ∧ lcm = A * (B / k)):
  Y = 17 := 
by sorry

end larger_factor_of_lcm_l286_286855


namespace sum_of_first_100_digits_of_1_div_2222_l286_286744

theorem sum_of_first_100_digits_of_1_div_2222 : 
  (let repeating_block := [0, 0, 0, 4, 5];
  let sum_of_digits (lst : List ℕ) := lst.sum;
  let block_sum := sum_of_digits repeating_block;
  let num_blocks := 100 / 5;
  num_blocks * block_sum = 180) :=
by 
  let repeating_block := [0, 0, 0, 4, 5]
  let sum_of_digits (lst : List ℕ) := lst.sum
  let block_sum := sum_of_digits repeating_block
  let num_blocks := 100 / 5
  have h : num_blocks * block_sum = 180 := sorry
  exact h

end sum_of_first_100_digits_of_1_div_2222_l286_286744


namespace cube_root_neg_27_l286_286463

theorem cube_root_neg_27 : ∃ x : ℝ, x^3 = -27 ∧ x = -3 :=
by
  use -3
  split
  · norm_num
  · rfl

end cube_root_neg_27_l286_286463


namespace calculate_value_l286_286138

theorem calculate_value (x y : ℝ) (h : 2 * x + y = 6) : 
    ((x - y)^2 - (x + y)^2 + y * (2 * x - y)) / (-2 * y) = 3 :=
by 
  sorry

end calculate_value_l286_286138


namespace subtract_decimal_numbers_l286_286155

theorem subtract_decimal_numbers : 3.75 - 1.46 = 2.29 := by
  sorry

end subtract_decimal_numbers_l286_286155


namespace percentage_of_circle_outside_triangle_l286_286231

theorem percentage_of_circle_outside_triangle (A : ℝ)
  (h₁ : 0 < A) -- Total area A is positive
  (A_inter : ℝ) (A_outside_tri : ℝ) (A_total_circle : ℝ)
  (h₂ : A_inter = 0.45 * A)
  (h₃ : A_outside_tri = 0.40 * A)
  (h₄ : A_total_circle = 0.60 * A) :
  100 * (1 - A_inter / A_total_circle) = 25 :=
by
  sorry

end percentage_of_circle_outside_triangle_l286_286231


namespace line_does_not_pass_first_quadrant_l286_286319

open Real

theorem line_does_not_pass_first_quadrant (a b : ℝ) (h₁ : a > 0) (h₂ : b < 0) : 
  ¬∃ x y : ℝ, (x > 0) ∧ (y > 0) ∧ (ax + y - b = 0) :=
sorry

end line_does_not_pass_first_quadrant_l286_286319


namespace division_example_l286_286747

theorem division_example :
  100 / 0.25 = 400 :=
by sorry

end division_example_l286_286747


namespace max_x_minus_y_l286_286552

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^3 + y^3) = x + y) : x - y ≤ (Real.sqrt 2 / 2) :=
by {
  sorry
}

end max_x_minus_y_l286_286552


namespace division_sum_l286_286433

theorem division_sum (quotient divisor remainder : ℕ) (hquot : quotient = 65) (hdiv : divisor = 24) (hrem : remainder = 5) : 
  (divisor * quotient + remainder) = 1565 := by 
  sorry

end division_sum_l286_286433


namespace range_of_a_l286_286670

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l286_286670


namespace fred_dark_blue_marbles_count_l286_286943

/-- Fred's Marble Problem -/
def freds_marbles (red green dark_blue : ℕ) : Prop :=
  red = 38 ∧ green = red / 2 ∧ red + green + dark_blue = 63

theorem fred_dark_blue_marbles_count (red green dark_blue : ℕ) (h : freds_marbles red green dark_blue) :
  dark_blue = 6 :=
by
  sorry

end fred_dark_blue_marbles_count_l286_286943


namespace apples_b_lighter_than_a_l286_286573

-- Definitions based on conditions
def total_weight : ℕ := 72
def weight_basket_a : ℕ := 42
def weight_basket_b : ℕ := total_weight - weight_basket_a

-- Theorem to prove the question equals the answer given the conditions
theorem apples_b_lighter_than_a : (weight_basket_a - weight_basket_b) = 12 := by
  -- Placeholder for proof
  sorry

end apples_b_lighter_than_a_l286_286573


namespace part1_part2_l286_286225

theorem part1 (a x y : ℝ) (h1 : 3 * x - y = 2 * a - 5) (h2 : x + 2 * y = 3 * a + 3)
  (hx : x > 0) (hy : y > 0) : a > 1 :=
sorry

theorem part2 (a b : ℝ) (ha : a > 1) (h3 : a - b = 4) (hb : b < 2) : 
  -2 < a + b ∧ a + b < 8 :=
sorry

end part1_part2_l286_286225


namespace median_inequality_l286_286045

variables {α : ℝ} (A B C M : Point) (a b c : ℝ)

-- Definitions and conditions
def isTriangle (A B C : Point) : Prop := -- definition of triangle
sorry

def isMedian (A B C M : Point) : Prop := -- definition of median
sorry

-- Statement we want to prove
theorem median_inequality (h1 : isTriangle A B C) (h2 : isMedian A B C M) :
  2 * AM ≥ (b + c) * Real.cos (α / 2) :=
sorry

end median_inequality_l286_286045


namespace find_side_b_l286_286205

theorem find_side_b (a b c : ℝ) (A B C : ℝ)
  (area_triangle : Real.sqrt 3) (B_eq_60 : B = 60) (cond_ac: (a^2 + c^2 = 3 * a * c)) :
  ∃ b, b = 2 * Real.sqrt 2 :=
by
  sorry

end find_side_b_l286_286205


namespace jimmy_points_l286_286365

theorem jimmy_points (eng_pts init_eng_pts : ℕ) (math_pts init_math_pts : ℕ) 
  (sci_pts init_sci_pts : ℕ) (hist_pts init_hist_pts : ℕ) 
  (phy_pts init_phy_pts : ℕ) (eng_penalty math_penalty sci_penalty hist_penalty phy_penalty : ℕ)
  (passing_points : ℕ) (total_points_required : ℕ):
  init_eng_pts = 60 →
  init_math_pts = 55 →
  init_sci_pts = 40 →
  init_hist_pts = 70 →
  init_phy_pts = 50 →
  eng_penalty = 5 →
  math_penalty = 3 →
  sci_penalty = 8 →
  hist_penalty = 2 →
  phy_penalty = 6 →
  passing_points = 250 →
  total_points_required = (init_eng_pts - eng_penalty) + (init_math_pts - math_penalty) + 
                         (init_sci_pts - sci_penalty) + (init_hist_pts - hist_penalty) + 
                         (init_phy_pts - phy_penalty) →
  ∀ extra_loss, (total_points_required - extra_loss ≥ passing_points) → extra_loss ≤ 1 :=
by {
  sorry
}

end jimmy_points_l286_286365


namespace num_perpendicular_line_plane_pairs_in_cube_l286_286343

-- Definitions based on the problem conditions

def is_perpendicular_line_plane_pair (l : line) (p : plane) : Prop :=
  -- Assume an implementation that defines when a line is perpendicular to a plane
  sorry

-- Define a cube structure with its vertices, edges, and faces
structure Cube :=
  (vertices : Finset Point)
  (edges : Finset (Point × Point))
  (faces : Finset (Finset Point))

-- Make assumptions about cube properties
variable (cube : Cube)

-- Define the property of counting perpendicular line-plane pairs
def count_perpendicular_line_plane_pairs (c : Cube) : Nat :=
  -- Assume an implementation that counts the number of such pairs in the cube
  sorry

-- The theorem to prove
theorem num_perpendicular_line_plane_pairs_in_cube (c : Cube) :
  count_perpendicular_line_plane_pairs c = 36 :=
  sorry

end num_perpendicular_line_plane_pairs_in_cube_l286_286343


namespace positive_sum_inequality_l286_286020

theorem positive_sum_inequality 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) ≥ (ab + bc + ca)^3 := 
by 
  sorry

end positive_sum_inequality_l286_286020


namespace sum_of_terms_in_geometric_sequence_eq_fourteen_l286_286488

theorem sum_of_terms_in_geometric_sequence_eq_fourteen
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_a1 : a 1 = 1)
  (h_arith : 4 * a 2 = 2 * a 3 ∧ 2 * a 3 - 4 * a 2 = a 4 - 2 * a 3) :
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_of_terms_in_geometric_sequence_eq_fourteen_l286_286488


namespace multiply_by_15_is_225_l286_286613

-- Define the condition
def number : ℕ := 15

-- State the theorem with the conditions and the expected result
theorem multiply_by_15_is_225 : 15 * number = 225 := by
  -- Insert the proof here
  sorry

end multiply_by_15_is_225_l286_286613


namespace sum_of_first_100_terms_AP_l286_286041

theorem sum_of_first_100_terms_AP (a d : ℕ) :
  (15 / 2) * (2 * a + 14 * d) = 45 →
  (85 / 2) * (2 * a + 84 * d) = 255 →
  (100 / 2) * (2 * a + 99 * d) = 300 :=
by
  sorry

end sum_of_first_100_terms_AP_l286_286041


namespace geometric_sum_S30_l286_286733

theorem geometric_sum_S30 (S : ℕ → ℝ) (h1 : S 10 = 10) (h2 : S 20 = 30) : S 30 = 70 := 
by 
  sorry

end geometric_sum_S30_l286_286733


namespace number_of_sets_given_to_sister_l286_286364

-- Defining the total number of cards, sets given to his brother and friend, total cards given away,
-- number of cards per set, and expected answer for sets given to his sister.
def total_cards := 365
def sets_given_to_brother := 8
def sets_given_to_friend := 2
def total_cards_given_away := 195
def cards_per_set := 13
def sets_given_to_sister := 5

theorem number_of_sets_given_to_sister :
  sets_given_to_brother * cards_per_set + 
  sets_given_to_friend * cards_per_set + 
  sets_given_to_sister * cards_per_set = total_cards_given_away :=
by
  -- It skips the proof but ensures the statement is set up correctly.
  sorry

end number_of_sets_given_to_sister_l286_286364


namespace probability_T_H_E_equal_L_A_V_A_l286_286695

noncomputable def probability_condition : ℚ :=
  -- Number of total sample space (3^6)
  (3 ^ 6 : ℚ)

noncomputable def favorable_events_0 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 0 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 0
  26 * 19

noncomputable def favorable_events_1 : ℚ :=
  -- Number of favorable outcomes where 𝑻 ⋅ 𝑯 ⋅ 𝑬 is 1 and 𝑳 ⋅ 𝑨 ⋅ 𝑽 ⋅ 𝑨 is 1
  1

noncomputable def total_favorable_events : ℚ :=
  favorable_events_0 + favorable_events_1

theorem probability_T_H_E_equal_L_A_V_A :
  (total_favorable_events / probability_condition) = 55 / 81 :=
sorry

end probability_T_H_E_equal_L_A_V_A_l286_286695


namespace tablecloth_width_l286_286719

theorem tablecloth_width (length_tablecloth : ℕ) (napkins_count : ℕ) (napkin_length : ℕ) (napkin_width : ℕ) (total_material : ℕ) (width_tablecloth : ℕ) :
  length_tablecloth = 102 →
  napkins_count = 8 →
  napkin_length = 6 →
  napkin_width = 7 →
  total_material = 5844 →
  total_material = length_tablecloth * width_tablecloth + napkins_count * (napkin_length * napkin_width) →
  width_tablecloth = 54 :=
by
  intros h1 h2 h3 h4 h5 h_eq
  sorry

end tablecloth_width_l286_286719


namespace jose_peanuts_l286_286986

def kenya_peanuts : Nat := 133
def difference_peanuts : Nat := 48

theorem jose_peanuts : (kenya_peanuts - difference_peanuts) = 85 := by
  sorry

end jose_peanuts_l286_286986


namespace concert_attendance_difference_l286_286700

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := 66018

theorem concert_attendance_difference :
  (second_concert - first_concert) = 119 :=
by
  sorry

end concert_attendance_difference_l286_286700


namespace divisors_72_l286_286649

theorem divisors_72 : 
  { d | d ∣ 72 ∧ 0 < d } = {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72} := 
sorry

end divisors_72_l286_286649


namespace square_binomial_formula_l286_286428

variable {x y : ℝ}

theorem square_binomial_formula :
  (2 * x + y) * (y - 2 * x) = y^2 - 4 * x^2 := 
  sorry

end square_binomial_formula_l286_286428


namespace num_distinct_sets_of_8_positive_odd_integers_sum_to_20_l286_286810

def numDistinctOddSets (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem num_distinct_sets_of_8_positive_odd_integers_sum_to_20 :
  numDistinctOddSets 6 8 = 1716 :=
by
  sorry

end num_distinct_sets_of_8_positive_odd_integers_sum_to_20_l286_286810


namespace monotonicity_of_f_l286_286955

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + π / 6)

theorem monotonicity_of_f :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < π / 6 → f x₁ < f x₂ :=
by
  sorry

end monotonicity_of_f_l286_286955


namespace find_abc_l286_286523

open Real

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h1 : a * (b + c) = 154)
  (h2 : b * (c + a) = 164) 
  (h3 : c * (a + b) = 172) : 
  (a * b * c = Real.sqrt 538083) := 
by 
  sorry

end find_abc_l286_286523


namespace subtract_decimal_l286_286157

theorem subtract_decimal : 3.75 - 1.46 = 2.29 :=
by
  sorry

end subtract_decimal_l286_286157


namespace daily_wage_of_c_l286_286884

theorem daily_wage_of_c
  (a b c : ℚ)
  (h_ratio : a / 3 = b / 4 ∧ a / 3 = c / 5)
  (h_total_earning : 6 * a + 9 * b + 4 * c = 1850) :
  c = 125 :=
by 
  sorry

end daily_wage_of_c_l286_286884


namespace gcd_40_56_l286_286251

theorem gcd_40_56 : Nat.gcd 40 56 = 8 := 
by 
  sorry

end gcd_40_56_l286_286251


namespace valid_x_y_sum_l286_286336

-- Setup the initial conditions as variables.
variables (x y : ℕ)

-- Declare the conditions as hypotheses.
theorem valid_x_y_sum (h1 : 0 < x) (h2 : x < 25)
  (h3 : 0 < y) (h4 : y < 25) (h5 : x + y + x * y = 119) :
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
sorry

end valid_x_y_sum_l286_286336


namespace original_balance_l286_286779

variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

theorem original_balance (decrease_percentage : ℝ) (current_balance : ℝ) (original_balance : ℝ) :
  decrease_percentage = 0.10 → current_balance = 90000 → 
  current_balance = (1 - decrease_percentage) * original_balance → 
  original_balance = 100000 := by
  sorry

end original_balance_l286_286779


namespace C_increases_as_n_increases_l286_286497

theorem C_increases_as_n_increases (e n R r : ℝ) (he : 0 < e) (hn : 0 < n) (hR : 0 < R) (hr : 0 < r) :
  0 < (2 * e * n * R + e * n^2 * r) / (R + n * r)^2 :=
by
  sorry

end C_increases_as_n_increases_l286_286497


namespace arithmetic_sequence_ninth_term_l286_286233

-- Define the terms in the arithmetic sequence
def sequence_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Given conditions
def a1 : ℚ := 2 / 3
def a17 : ℚ := 5 / 6
def d : ℚ := 1 / 96 -- Calculated common difference

-- Prove the ninth term is 3/4
theorem arithmetic_sequence_ninth_term :
  sequence_term a1 d 9 = 3 / 4 :=
sorry

end arithmetic_sequence_ninth_term_l286_286233


namespace factor_polynomial_l286_286153

theorem factor_polynomial (x : ℝ) : 75 * x^5 - 300 * x^10 = 75 * x^5 * (1 - 4 * x^5) :=
by
  sorry

end factor_polynomial_l286_286153


namespace eval_expr1_l286_286152

theorem eval_expr1 : 
  ( (27 / 8) ^ (-2 / 3) - (49 / 9) ^ 0.5 + (0.008) ^ (-2 / 3) * (2 / 25) ) = 1 / 9 :=
by 
  sorry

end eval_expr1_l286_286152


namespace bug_visits_exactly_16_pavers_l286_286126

-- Defining the dimensions of the garden and the pavers
def garden_width : ℕ := 14
def garden_length : ℕ := 19
def paver_size : ℕ := 2

-- Calculating the number of pavers in width and length
def pavers_width : ℕ := garden_width / paver_size
def pavers_length : ℕ := (garden_length + paver_size - 1) / paver_size  -- Taking ceiling of 19/2

-- Calculating the GCD of the pavers count in width and length
def gcd_pavers : ℕ := Nat.gcd pavers_width pavers_length

-- Calculating the number of pavers the bug crosses
def pavers_crossed : ℕ := pavers_width + pavers_length - gcd_pavers

-- Theorem that states the number of pavers visited
theorem bug_visits_exactly_16_pavers :
  pavers_crossed = 16 := by
  -- Sorry is used to skip the proof steps
  sorry

end bug_visits_exactly_16_pavers_l286_286126


namespace gondor_total_earnings_l286_286177

-- Defining the earnings from repairing a phone and a laptop
def phone_earning : ℕ := 10
def laptop_earning : ℕ := 20

-- Defining the number of repairs
def monday_phone_repairs : ℕ := 3
def tuesday_phone_repairs : ℕ := 5
def wednesday_laptop_repairs : ℕ := 2
def thursday_laptop_repairs : ℕ := 4

-- Calculating total earnings
def monday_earnings : ℕ := monday_phone_repairs * phone_earning
def tuesday_earnings : ℕ := tuesday_phone_repairs * phone_earning
def wednesday_earnings : ℕ := wednesday_laptop_repairs * laptop_earning
def thursday_earnings : ℕ := thursday_laptop_repairs * laptop_earning

def total_earnings : ℕ := monday_earnings + tuesday_earnings + wednesday_earnings + thursday_earnings

-- The theorem to be proven
theorem gondor_total_earnings : total_earnings = 200 := by
  sorry

end gondor_total_earnings_l286_286177


namespace net_effect_on_sale_value_l286_286107

theorem net_effect_on_sale_value
(P Q : ℝ)
(h_new_price : ∃ P', P' = P - 0.22 * P)
(h_new_qty : ∃ Q', Q' = Q + 0.86 * Q) :
  let original_sale_value := P * Q
  let new_sale_value := (0.78 * P) * (1.86 * Q)
  let net_effect := ((new_sale_value / original_sale_value - 1) * 100 : ℝ)
  net_effect = 45.08 :=
by {
  sorry
}

end net_effect_on_sale_value_l286_286107


namespace cos_half_pi_plus_double_alpha_l286_286658

theorem cos_half_pi_plus_double_alpha (α : ℝ) (h : Real.tan α = 1 / 3) : 
  Real.cos (Real.pi / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end cos_half_pi_plus_double_alpha_l286_286658


namespace phone_price_is_correct_l286_286119

-- Definition of the conditions
def monthly_cost := 7
def months := 4
def total_cost := 30

-- Definition to be proven
def phone_price := total_cost - (monthly_cost * months)

theorem phone_price_is_correct : phone_price = 2 :=
by
  sorry

end phone_price_is_correct_l286_286119


namespace average_weight_when_D_joins_is_53_l286_286232

noncomputable def new_average_weight (A B C D E : ℕ) : ℕ :=
  (73 + B + C + D) / 4

theorem average_weight_when_D_joins_is_53 :
  (A + B + C) / 3 = 50 →
  A = 73 →
  (B + C + D + E) / 4 = 51 →
  E = D + 3 →
  73 + B + C + D = 212 →
  new_average_weight A B C D E = 53 :=
by
  sorry

end average_weight_when_D_joins_is_53_l286_286232


namespace intersection_of_sets_l286_286204

-- Definitions from the conditions.
def A := { x : ℝ | x^2 - 2 * x ≤ 0 }
def B := { x : ℝ | x > 1 }

-- The proof problem statement.
theorem intersection_of_sets :
  A ∩ B = { x : ℝ | 1 < x ∧ x ≤ 2 } :=
sorry

end intersection_of_sets_l286_286204


namespace total_stickers_l286_286699

theorem total_stickers :
  (20.0 : ℝ) + (26.0 : ℝ) + (20.0 : ℝ) + (6.0 : ℝ) + (58.0 : ℝ) = 130.0 := by
  sorry

end total_stickers_l286_286699


namespace part1_part2_part3_l286_286015

def climbing_function_1_example (x : ℝ) : Prop :=
  ∃ a : ℝ, a^2 = -8 / a

theorem part1 (x : ℝ) : climbing_function_1_example x ↔ (x = -2) := sorry

def climbing_function_2_example (m : ℝ) : Prop :=
  ∃ a : ℝ, (a^2 = m*a + m) ∧ ∀ d: ℝ, ((d^2 = m*d + m) → d = a)

theorem part2 (m : ℝ) : (m = -4) ∧ climbing_function_2_example m := sorry

def climbing_function_3_example (m n p q : ℝ) (h1 : m ≥ 2) (h2 : p^2 = 3*q) : Prop :=
  ∃ a1 a2 : ℝ, ((a1 + a2 = n/(1-m)) ∧ (a1*a2 = 1/(m-1)) ∧ (|a1 - a2| = p)) ∧ 
  (∀ x : ℝ, (m * x^2 + n * x + 1) ≥ q) 

theorem part3 (m n p q : ℝ) (h1 : m ≥ 2) (h2 : p^2 = 3*q) : climbing_function_3_example m n p q h1 h2 ↔ (0 < q) ∧ (q ≤ 4/11) := sorry

end part1_part2_part3_l286_286015


namespace road_construction_equation_l286_286595

theorem road_construction_equation (x : ℝ) (hx : x > 0) :
  (9 / x) - (12 / (x + 1)) = 1 / 2 :=
sorry

end road_construction_equation_l286_286595


namespace gallons_per_hour_l286_286089

-- Define conditions
def total_runoff : ℕ := 240000
def days : ℕ := 10
def hours_per_day : ℕ := 24

-- Define the goal: proving the sewers handle 1000 gallons of run-off per hour
theorem gallons_per_hour : (total_runoff / (days * hours_per_day)) = 1000 :=
by
  -- Proof can be inserted here
  sorry

end gallons_per_hour_l286_286089


namespace yellow_tickets_needed_l286_286405

def yellow_from_red (r : ℕ) : ℕ := r / 10
def red_from_blue (b : ℕ) : ℕ := b / 10
def blue_needed (current_blue : ℕ) (additional_blue : ℕ) : ℕ := current_blue + additional_blue
def total_blue_from_tickets (y : ℕ) (r : ℕ) (b : ℕ) : ℕ := (y * 10 * 10) + (r * 10) + b

theorem yellow_tickets_needed (y r b additional_blue : ℕ) (h : total_blue_from_tickets y r b + additional_blue = 1000) :
  yellow_from_red (red_from_blue (total_blue_from_tickets y r b + additional_blue)) = 10 := 
by
  sorry

end yellow_tickets_needed_l286_286405


namespace find_m_l286_286240

noncomputable def volume_parallelepiped (v₁ v₂ v₃ : ℝ × ℝ × ℝ) : ℝ :=
  real.abs (matrix.det ![
      ![v₁.1, v₂.1, v₃.1],
      ![v₁.2.1, v₂.2.1, v₃.2.1],
      ![v₁.2.2, v₂.2.2, v₃.2.2]
    ])

theorem find_m (m : ℝ) (h : volume_parallelepiped (3, 2, 5) (2, m, 3) (2, 4, m) = 20) (hm : m > 0) :
  m = 5 :=
sorry

end find_m_l286_286240


namespace certain_number_is_two_l286_286184

theorem certain_number_is_two (n : ℕ) 
  (h1 : 1 = 62) 
  (h2 : 363 = 3634) 
  (h3 : 3634 = n) 
  (h4 : n = 365) 
  (h5 : 36 = 2) : 
  n = 2 := 
by 
  sorry

end certain_number_is_two_l286_286184


namespace intersection_proof_l286_286052

noncomputable def M : Set ℕ := {1, 3, 5, 7, 9}

noncomputable def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_proof : M ∩ (N ∩ Set.univ) = {5, 7, 9} :=
by sorry

end intersection_proof_l286_286052


namespace quadratic_function_origin_l286_286435

theorem quadratic_function_origin {a b c : ℝ} :
  (∀ x, y = ax * x + bx * x + c → y = 0 → 0 = c ∧ b = 0) ∨ (c = 0) :=
sorry

end quadratic_function_origin_l286_286435


namespace wholesale_price_is_90_l286_286763

theorem wholesale_price_is_90 
  (R S W: ℝ)
  (h1 : R = 120)
  (h2 : S = R - 0.1 * R)
  (h3 : S = W + 0.2 * W)
  : W = 90 := 
by
  sorry

end wholesale_price_is_90_l286_286763


namespace pears_total_correct_l286_286566

noncomputable def pickedPearsTotal (sara_picked tim_picked : Nat) : Nat :=
  sara_picked + tim_picked

theorem pears_total_correct :
    pickedPearsTotal 6 5 = 11 :=
  by
    sorry

end pears_total_correct_l286_286566


namespace max_integer_value_fraction_l286_286968

theorem max_integer_value_fraction (x : ℝ) : 
  (∃ t : ℤ, t = 2 ∧ (∀ y : ℝ, y = (4*x^2 + 8*x + 21) / (4*x^2 + 8*x + 9) → y <= t)) :=
sorry

end max_integer_value_fraction_l286_286968


namespace four_numbers_are_perfect_squares_l286_286586

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem four_numbers_are_perfect_squares (a b c d : ℕ) (h1 : is_perfect_square (a * b * c))
                                                      (h2 : is_perfect_square (a * c * d))
                                                      (h3 : is_perfect_square (b * c * d))
                                                      (h4 : is_perfect_square (a * b * d)) : 
                                                      is_perfect_square a ∧
                                                      is_perfect_square b ∧
                                                      is_perfect_square c ∧
                                                      is_perfect_square d :=
by
  sorry

end four_numbers_are_perfect_squares_l286_286586


namespace JameMade112kProfit_l286_286355

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end JameMade112kProfit_l286_286355


namespace average_sales_per_month_after_discount_is_93_l286_286140

theorem average_sales_per_month_after_discount_is_93 :
  let salesJanuary := 120
  let salesFebruary := 80
  let salesMarch := 70
  let salesApril := 150
  let salesMayBeforeDiscount := 50
  let discountRate := 0.10
  let discountedSalesMay := salesMayBeforeDiscount - (discountRate * salesMayBeforeDiscount)
  let totalSales := salesJanuary + salesFebruary + salesMarch + salesApril + discountedSalesMay
  let numberOfMonths := 5
  let averageSales := totalSales / numberOfMonths
  averageSales = 93 :=
by {
  -- The actual proof code would go here, but we will skip the proof steps as instructed.
  sorry
}

end average_sales_per_month_after_discount_is_93_l286_286140


namespace problems_solved_by_trainees_l286_286043

theorem problems_solved_by_trainees (n m : ℕ) (h : ∀ t, t < m → (∃ p, p < n → p ≥ n / 2)) :
  ∃ p < n, (∃ t, t < m → t ≥ m / 2) :=
by
  sorry

end problems_solved_by_trainees_l286_286043


namespace proj_b_l286_286550

noncomputable def proj (u v : ℝ × ℝ) : ℝ × ℝ :=
  let (ux, uy) := u
  let (vx, vy) := v
  let factor := (ux * vx + uy * vy) / (vx * vx + vy * vy)
  (factor * vx, factor * vy)

theorem proj_b (a b v : ℝ × ℝ) (h_ortho : a.1 * b.1 + a.2 * b.2 = 0)
  (h_proj_a : proj v a = (1, 2)) : proj v b = (3, -4) :=
by
  sorry

end proj_b_l286_286550


namespace sharon_distance_to_mothers_house_l286_286568

noncomputable def total_distance (x : ℝ) :=
  x / 240

noncomputable def adjusted_speed (x : ℝ) :=
  x / 240 - 1 / 4

theorem sharon_distance_to_mothers_house (x : ℝ) (h1 : x / 240 = total_distance x) 
(h2 : adjusted_speed x = x / 240 - 1 / 4) 
(h3 : 120 + 120 * x / (x - 60) = 330) : 
x = 140 := 
by 
  sorry

end sharon_distance_to_mothers_house_l286_286568


namespace hundredth_odd_integer_l286_286421

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l286_286421


namespace number_of_regular_pencils_l286_286821

def cost_eraser : ℝ := 0.8
def cost_regular : ℝ := 0.5
def cost_short : ℝ := 0.4
def num_eraser : ℕ := 200
def num_short : ℕ := 35
def total_revenue : ℝ := 194

theorem number_of_regular_pencils (num_regular : ℕ) :
  (num_eraser * cost_eraser) + (num_short * cost_short) + (num_regular * cost_regular) = total_revenue → 
  num_regular = 40 :=
by
  sorry

end number_of_regular_pencils_l286_286821


namespace calculate_initial_money_l286_286112

noncomputable def initial_money (remaining_money: ℝ) (spent_percent: ℝ) : ℝ :=
  remaining_money / (1 - spent_percent)

theorem calculate_initial_money :
  initial_money 3500 0.30 = 5000 := 
by
  rw [initial_money]
  sorry

end calculate_initial_money_l286_286112


namespace eve_total_spend_l286_286647

def hand_mitts_cost : ℝ := 14.00
def apron_cost : ℝ := 16.00
def utensils_cost : ℝ := 10.00
def knife_cost : ℝ := 2 * utensils_cost
def discount_percent : ℝ := 0.25
def nieces_count : ℕ := 3

def total_cost_before_discount : ℝ :=
  (hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * nieces_count

def discount_amount : ℝ :=
  discount_percent * total_cost_before_discount

def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount_amount

theorem eve_total_spend : total_cost_after_discount = 135.00 := by
  sorry

end eve_total_spend_l286_286647


namespace Victoria_money_left_l286_286741

noncomputable def Victoria_initial_money : ℝ := 10000
noncomputable def jacket_price : ℝ := 250
noncomputable def trousers_price : ℝ := 180
noncomputable def purse_price : ℝ := 450
noncomputable def jackets_bought : ℕ := 8
noncomputable def trousers_bought : ℕ := 15
noncomputable def purses_bought : ℕ := 4
noncomputable def discount_rate : ℝ := 0.15
noncomputable def dinner_bill_inclusive : ℝ := 552.50
noncomputable def dinner_service_charge_rate : ℝ := 0.15

theorem Victoria_money_left : 
  Victoria_initial_money - 
  ((jackets_bought * jacket_price + trousers_bought * trousers_price) * (1 - discount_rate) + 
   purses_bought * purse_price + 
   dinner_bill_inclusive / (1 + dinner_service_charge_rate)) = 3725 := 
by 
  sorry

end Victoria_money_left_l286_286741


namespace find_dividend_l286_286678

theorem find_dividend (x D : ℕ) (q r : ℕ) (h_q : q = 4) (h_r : r = 3)
  (h_div : D = x * q + r) (h_sum : D + x + q + r = 100) : D = 75 :=
by
  sorry

end find_dividend_l286_286678


namespace k_value_l286_286016

theorem k_value (k : ℝ) (h : 10 * k * (-1)^3 - (-1) - 9 = 0) : k = -4 / 5 :=
by
  sorry

end k_value_l286_286016


namespace solve_for_a_l286_286966

theorem solve_for_a (a x y : ℝ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 3) : a = 5 :=
by
  sorry

end solve_for_a_l286_286966


namespace probability_perfect_square_l286_286473

theorem probability_perfect_square (choose_numbers : Finset (Fin 49)) (ticket : Finset (Fin 49))
  (h_choose_size : choose_numbers.card = 6) 
  (h_ticket_size : ticket.card = 6)
  (h_choose_square : ∃ (n : ℕ), (choose_numbers.prod id = n * n))
  (h_ticket_square : ∃ (m : ℕ), (ticket.prod id = m * m)) :
  ∃ T, (1 / T = 1 / T) :=
by
  sorry

end probability_perfect_square_l286_286473


namespace monomial_2023_eq_l286_286068

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^(n+1) * (2*n - 1), n)

theorem monomial_2023_eq : monomial 2023 = (4045, 2023) :=
by
  sorry

end monomial_2023_eq_l286_286068


namespace find_g_5_l286_286731

-- Define the function g and the condition it satisfies
variable {g : ℝ → ℝ}
variable (hg : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x)

-- The proof goal
theorem find_g_5 : g 5 = 206 / 35 :=
by
  -- To be proven using the given condition hg
  sorry

end find_g_5_l286_286731


namespace A_plus_B_eq_one_fourth_l286_286066

noncomputable def A : ℚ := 1 / 3
noncomputable def B : ℚ := -1 / 12

theorem A_plus_B_eq_one_fourth :
  A + B = 1 / 4 := by
  sorry

end A_plus_B_eq_one_fourth_l286_286066


namespace mass_percentage_H_in_CaH₂_l286_286791

def atomic_mass_Ca : ℝ := 40.08
def atomic_mass_H : ℝ := 1.008
def molar_mass_CaH₂ : ℝ := atomic_mass_Ca + 2 * atomic_mass_H

theorem mass_percentage_H_in_CaH₂ :
  (2 * atomic_mass_H / molar_mass_CaH₂) * 100 = 4.79 := 
by
  -- Skipping the detailed proof for now
  sorry

end mass_percentage_H_in_CaH₂_l286_286791


namespace series_items_increase_l286_286409

theorem series_items_increase (n : ℕ) (hn : n ≥ 2) :
  (2^n + 1) - 2^(n-1) - 1 = 2^(n-1) :=
by
  sorry

end series_items_increase_l286_286409


namespace math_problem_l286_286493

open Real -- Open the real number namespace

theorem math_problem (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) :=
by
  sorry

end math_problem_l286_286493


namespace expression_value_as_fraction_l286_286037

theorem expression_value_as_fraction (x y : ℕ) (hx : x = 3) (hy : y = 5) : 
  ( ( (1 / (y : ℚ)) / (1 / (x : ℚ)) ) ^ 2 ) = 9 / 25 := 
by
  sorry

end expression_value_as_fraction_l286_286037


namespace nick_paints_wall_in_fraction_l286_286673

theorem nick_paints_wall_in_fraction (nick_paint_time wall_paint_time : ℕ) (h1 : wall_paint_time = 60) (h2 : nick_paint_time = 12) : (nick_paint_time * 1 / wall_paint_time = 1 / 5) :=
by
  sorry

end nick_paints_wall_in_fraction_l286_286673


namespace quadratic_roots_satisfy_condition_l286_286491
variable (x1 x2 m : ℝ)

theorem quadratic_roots_satisfy_condition :
  ( ∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1 + x2 = -m) ∧ 
    (x1 * x2 = 5) ∧ (x1 = 2 * |x2| - 3) ) →
  m = -9 / 2 :=
by
  sorry

end quadratic_roots_satisfy_condition_l286_286491


namespace irrational_number_l286_286258

noncomputable def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number : 
  is_rational (Real.sqrt 4) ∧ 
  is_rational (22 / 7 : ℝ) ∧ 
  is_rational (1.0101 : ℝ) ∧ 
  ¬ is_rational (Real.pi / 3) 
  :=
sorry

end irrational_number_l286_286258


namespace second_half_takes_200_percent_longer_l286_286557

noncomputable def time_take (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

variable (total_distance : ℕ := 640)
variable (first_half_speed : ℕ := 80)
variable (average_speed : ℕ := 40)

theorem second_half_takes_200_percent_longer :
  let first_half_distance := total_distance / 2;
  let first_half_time := time_take first_half_distance first_half_speed;
  let total_time := time_take total_distance average_speed;
  let second_half_time := total_time - first_half_time;
  let time_increase := second_half_time - first_half_time;
  let percentage_increase := (time_increase * 100) / first_half_time;
  percentage_increase = 200 :=
by
  sorry

end second_half_takes_200_percent_longer_l286_286557


namespace smallest_perfect_square_sum_of_20_consecutive_integers_l286_286864

theorem smallest_perfect_square_sum_of_20_consecutive_integers :
  ∃ n : ℕ, ∑ i in finset.range 20, (n + i) = 250 :=
by
  sorry

end smallest_perfect_square_sum_of_20_consecutive_integers_l286_286864


namespace eugene_boxes_needed_l286_286935

-- Define the number of cards in the deck
def total_cards : ℕ := 52

-- Define the number of cards not used
def unused_cards : ℕ := 16

-- Define the number of toothpicks per card
def toothpicks_per_card : ℕ := 75

-- Define the number of toothpicks in a box
def toothpicks_per_box : ℕ := 450

-- Calculate the number of cards used
def cards_used : ℕ := total_cards - unused_cards

-- Calculate the number of cards a single box can support
def cards_per_box : ℕ := toothpicks_per_box / toothpicks_per_card

-- Theorem statement
theorem eugene_boxes_needed : cards_used / cards_per_box = 6 := by
  -- The proof steps are not provided as per the instructions. 
  sorry

end eugene_boxes_needed_l286_286935


namespace find_m_l286_286655

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 + 3 * x2 = 5) : m = 7 / 4 :=
  sorry

end find_m_l286_286655


namespace remainder_when_4_pow_2023_div_17_l286_286480

theorem remainder_when_4_pow_2023_div_17 :
  ∀ (x : ℕ), (x = 4) → x^2 ≡ 16 [MOD 17] → x^2023 ≡ 13 [MOD 17] := by
  intros x hx h
  sorry

end remainder_when_4_pow_2023_div_17_l286_286480


namespace reciprocal_sum_of_roots_l286_286014

theorem reciprocal_sum_of_roots
  (a b c : ℝ)
  (ha : a^3 - 2022 * a + 1011 = 0)
  (hb : b^3 - 2022 * b + 1011 = 0)
  (hc : c^3 - 2022 * c + 1011 = 0)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (1 / a) + (1 / b) + (1 / c) = 2 :=
sorry

end reciprocal_sum_of_roots_l286_286014


namespace ellipse_foci_distance_l286_286790

noncomputable def distance_between_foci : ℝ :=
  let a := 20
  let b := 10
  2 * Real.sqrt (a ^ 2 - b ^ 2)

theorem ellipse_foci_distance : distance_between_foci = 20 * Real.sqrt 3 := by
  sorry

end ellipse_foci_distance_l286_286790


namespace even_function_a_value_l286_286027

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x^2 + (a^2 - 1) * x + (a - 1)) = ((-x)^2 + (a^2 - 1) * (-x) + (a - 1))) → (a = 1 ∨ a = -1) :=
by
  sorry

end even_function_a_value_l286_286027


namespace constant_expression_l286_286389

theorem constant_expression 
  (x y : ℝ) 
  (h₁ : x + y = 1) 
  (h₂ : x ≠ 1) 
  (h₃ : y ≠ 1) : 
  (x / (y^3 - 1) + y / (1 - x^3) + 2 * (x - y) / (x^2 * y^2 + 3)) = 0 :=
by 
  sorry

end constant_expression_l286_286389


namespace walther_janous_inequality_equality_condition_l286_286217

theorem walther_janous_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * y ∧ x / y = 2 ∧ y = z :=
sorry

end walther_janous_inequality_equality_condition_l286_286217


namespace ticket_value_unique_l286_286131

theorem ticket_value_unique (x : ℕ) (h₁ : ∃ n, n > 0 ∧ x * n = 60)
  (h₂ : ∃ m, m > 0 ∧ x * m = 90)
  (h₃ : ∃ p, p > 0 ∧ x * p = 49) : 
  ∃! x, x = 1 :=
by
  sorry

end ticket_value_unique_l286_286131


namespace ratio_A_to_B_l286_286468

theorem ratio_A_to_B (total_weight_X : ℕ) (weight_B : ℕ) (weight_A : ℕ) (h₁ : total_weight_X = 324) (h₂ : weight_B = 270) (h₃ : weight_A = total_weight_X - weight_B):
  weight_A / gcd weight_A weight_B = 1 ∧ weight_B / gcd weight_A weight_B = 5 :=
by
  sorry

end ratio_A_to_B_l286_286468


namespace coin_difference_l286_286708

theorem coin_difference (h : ∃ x y z : ℕ, 5*x + 10*y + 20*z = 40) : (∃ x : ℕ, 5*x = 40) → (∃ y : ℕ, 20*y = 40) → 8 - 2 = 6 :=
by
  intros h1 h2
  exact rfl

end coin_difference_l286_286708


namespace value_of_expression_l286_286743

theorem value_of_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -1) : -a^2 - b^2 + a * b = -21 := by
  sorry

end value_of_expression_l286_286743


namespace product_combination_count_l286_286902

-- Definitions of the problem

-- There are 6 different types of cookies
def num_cookies : Nat := 6

-- There are 4 different types of milk
def num_milks : Nat := 4

-- Charlie will not order more than one of the same type
def charlie_order_limit : Nat := 1

-- Delta will only order cookies, including repeats of types
def delta_only_cookies : Bool := true

-- Prove that there are 2531 ways for Charlie and Delta to leave the store with 4 products collectively
theorem product_combination_count : 
  (number_of_ways : Nat) = 2531 
  := sorry

end product_combination_count_l286_286902


namespace smallest_lcm_l286_286333

/-- If k and l are positive 4-digit integers such that gcd(k, l) = 5, 
the smallest value for lcm(k, l) is 201000. -/
theorem smallest_lcm (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h₅ : Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l286_286333


namespace ball_arrangements_l286_286656

theorem ball_arrangements :
  let balls := 5
  let selected_balls := 4
  let first_box := 1
  let second_box := 2
  let third_box := 1
  let total_ways := Nat.choose balls selected_balls * Nat.choose selected_balls first_box * 
                    Nat.choose (selected_balls - first_box) second_box *  Nat.choose 1 third_box
  in total_ways = 60
:= by
  -- Definitions
  let balls := 5
  let selected_balls := 4
  let first_box := 1
  let second_box := 2
  let third_box := 1
  let total_ways := Nat.choose balls selected_balls * Nat.choose selected_balls first_box * 
                    Nat.choose (selected_balls - first_box) second_box * Nat.choose 1 third_box
  
  -- The proof is omitted.
  sorry

end ball_arrangements_l286_286656


namespace math_problem_l286_286425

theorem math_problem :
  (10^2 + 6^2) / 2 = 68 :=
by
  sorry

end math_problem_l286_286425


namespace asymptote_of_hyperbola_l286_286850

theorem asymptote_of_hyperbola : 
  ∀ x y : ℝ, (y^2 / 4 - x^2 = 1) → (y = 2 * x) ∨ (y = -2 * x) := 
by
  sorry

end asymptote_of_hyperbola_l286_286850


namespace eventually_periodic_sequence_l286_286748

noncomputable def eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ N k : ℕ, k > 0 ∧ ∀ m ≥ N, a m = a (m + k)

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h_pos : ∀ n, a n > 0)
  (h_condition : ∀ n, a n * a (n + 1) = a (n + 2) * a (n + 3)) :
  eventually_periodic a :=
sorry

end eventually_periodic_sequence_l286_286748


namespace calculate_area_of_square_field_l286_286574

def area_of_square_field (t: ℕ) (v: ℕ) (d: ℕ) (s: ℕ) (a: ℕ) : Prop :=
  t = 10 ∧ v = 16 ∧ d = v * t ∧ 4 * s = d ∧ a = s^2

theorem calculate_area_of_square_field (t v d s a : ℕ) 
  (h1: t = 10) (h2: v = 16) (h3: d = v * t) (h4: 4 * s = d) 
  (h5: a = s^2) : a = 1600 := by
  sorry

end calculate_area_of_square_field_l286_286574


namespace two_digit_number_l286_286193

theorem two_digit_number (x : ℕ) (h1 : x ≥ 10 ∧ x < 100)
  (h2 : ∃ k : ℤ, 3 * x - 4 = 10 * k)
  (h3 : 60 < 4 * x - 15 ∧ 4 * x - 15 < 100) :
  x = 28 :=
by
  sorry

end two_digit_number_l286_286193


namespace marbles_count_l286_286945

theorem marbles_count (red green blue total : ℕ) (h_red : red = 38)
  (h_green : green = red / 2) (h_total : total = 63) 
  (h_sum : total = red + green + blue) : blue = 6 :=
by
  sorry

end marbles_count_l286_286945


namespace cans_of_soda_l286_286394

variable (T R E : ℝ)

theorem cans_of_soda (hT: T > 0) (hR: R > 0) (hE: E > 0) : 5 * E * T / R = (5 * E) / R * T :=
by
  sorry

end cans_of_soda_l286_286394


namespace scaling_adults_taller_l286_286865

open Nat

theorem scaling_adults_taller (n : ℕ) (h_c h_a : Fin n → ℚ) (h : ∀ i, h_c i < h_a i) :
  ∃ (A : Fin n → ℕ), ∀ i j, h_c i * (A i) < h_a j * (A j) := by
  sorry

end scaling_adults_taller_l286_286865


namespace find_a_l286_286326

open Set

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem find_a (a : ℝ) :
  ∅ ⊂ (A a ∩ B) ∧ A a ∩ C = ∅ → a = -2 :=
by
  sorry

end find_a_l286_286326


namespace penny_remaining_money_l286_286712

theorem penny_remaining_money (initial_money : ℤ) (socks_pairs : ℤ) (socks_cost_per_pair : ℤ) (hat_cost : ℤ) :
  initial_money = 20 → socks_pairs = 4 → socks_cost_per_pair = 2 → hat_cost = 7 → 
  initial_money - (socks_pairs * socks_cost_per_pair + hat_cost) = 5 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end penny_remaining_money_l286_286712


namespace male_worker_ants_percentage_l286_286228

theorem male_worker_ants_percentage 
  (total_ants : ℕ) 
  (half_ants : ℕ) 
  (female_worker_ants : ℕ) 
  (h1 : total_ants = 110) 
  (h2 : half_ants = total_ants / 2) 
  (h3 : female_worker_ants = 44) :
  (half_ants - female_worker_ants) * 100 / half_ants = 20 := by
  sorry

end male_worker_ants_percentage_l286_286228


namespace ace_then_king_probability_l286_286407

theorem ace_then_king_probability :
  let total_cards := 52
  let aces := 4
  let kings := 4
  let first_ace_prob := (aces : ℚ) / (total_cards : ℚ)
  let second_king_given_ace_prob := (kings : ℚ) / (total_cards - 1 : ℚ)
  (first_ace_prob * second_king_given_ace_prob = (4 : ℚ) / 663) :=
by
  let total_cards := 52
  let aces := 4
  let kings := 4
  let first_ace_prob := (aces : ℚ) / (total_cards : ℚ)
  let second_king_given_ace_prob := (kings : ℚ) / (total_cards - 1 : ℚ)
  exact (first_ace_prob * second_king_given_ace_prob = (4 : ℚ) / 663)
  sorry

end ace_then_king_probability_l286_286407


namespace find_tax_percentage_l286_286124

noncomputable def net_income : ℝ := 12000
noncomputable def total_income : ℝ := 13000
noncomputable def non_taxable_income : ℝ := 3000
noncomputable def taxable_income : ℝ := total_income - non_taxable_income
noncomputable def tax_percentage (T : ℝ) := total_income - (T * taxable_income)

theorem find_tax_percentage : ∃ T : ℝ, tax_percentage T = net_income :=
by
  sorry

end find_tax_percentage_l286_286124


namespace inequality_solution_l286_286323

theorem inequality_solution (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 1/2 > 0) ↔ (0 ≤ m ∧ m < 2) :=
by
  sorry

end inequality_solution_l286_286323


namespace even_perfect_squares_between_50_and_200_l286_286811

theorem even_perfect_squares_between_50_and_200 : ∃ s : Finset ℕ, 
  (∀ n ∈ s, (n^2 ≥ 50) ∧ (n^2 ≤ 200) ∧ n^2 % 2 = 0) ∧ s.card = 4 := by
  sorry

end even_perfect_squares_between_50_and_200_l286_286811


namespace total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l286_286869

-- Define the conditions
def total_people (A B : ℕ) : Prop := A + B = 92
def valid_class_A (A : ℕ) : Prop := 51 < A ∧ A < 55
def total_cost (sets : ℕ) (cost_per_set : ℕ) : ℕ := sets * cost_per_set

-- Prices per set for different ranges of number of sets
def price_per_set (n : ℕ) : ℕ :=
  if n > 90 then 30 else if n > 50 then 40 else 50

-- Question 1
theorem total_amount_for_uniforms (A B : ℕ) (h1 : total_people A B) : total_cost 92 30 = 2760 := sorry

-- Question 2
theorem students_in_classes (A B : ℕ) (h1 : total_people A B) (h2 : valid_class_A A) (h3 : 40 * A + 50 * B = 4080) : A = 52 ∧ B = 40 := sorry

-- Question 3
theorem cost_effective_purchase_plan (A : ℕ) (h1 : 51 < A ∧ A < 55) (B : ℕ) (h2 : 92 - A = B) (h3 : A - 8 + B = 91) :
  ∃ (cost : ℕ), cost = total_cost 91 30 ∧ cost = 2730 := sorry

end total_amount_for_uniforms_students_in_classes_cost_effective_purchase_plan_l286_286869


namespace archipelago_max_value_l286_286230

noncomputable def archipelago_max_islands (N : ℕ) : Prop :=
  N ≥ 7 ∧ 
  (∀ (a b : ℕ), a ≠ b → a ≤ N → b ≤ N → ∃ c : ℕ, c ≤ N ∧ (∃ d, d ≠ c ∧ d ≤ N → d ≠ a ∧ d ≠ b)) ∧ 
  (∀ (a : ℕ), a ≤ N → ∃ b, b ≠ a ∧ b ≤ N ∧ (∃ c, c ≤ N ∧ c ≠ b ∧ c ≠ a))

theorem archipelago_max_value : archipelago_max_islands 36 := sorry

end archipelago_max_value_l286_286230


namespace remainder_3249_div_82_eq_51_l286_286103

theorem remainder_3249_div_82_eq_51 : (3249 % 82) = 51 :=
by
  sorry

end remainder_3249_div_82_eq_51_l286_286103


namespace smallest_fraction_denominator_l286_286160

theorem smallest_fraction_denominator (p q : ℕ) :
  (1:ℚ) / 2014 < p / q ∧ p / q < (1:ℚ) / 2013 → q = 4027 :=
sorry

end smallest_fraction_denominator_l286_286160


namespace remainder_when_2519_divided_by_3_l286_286904

theorem remainder_when_2519_divided_by_3 :
  2519 % 3 = 2 :=
by
  sorry

end remainder_when_2519_divided_by_3_l286_286904


namespace simple_random_sampling_methods_proof_l286_286895

-- Definitions based on conditions
def equal_probability (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
∀ s1 s2 : samples, p s1 = p s2

-- Define that Lottery Drawing Method and Random Number Table Method are part of simple random sampling
def is_lottery_drawing_method (samples : Type) : Prop := sorry
def is_random_number_table_method (samples : Type) : Prop := sorry

def simple_random_sampling_methods (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) : Prop :=
  equal_probability samples p ∧ is_lottery_drawing_method samples ∧ is_random_number_table_method samples

-- Statement to be proven
theorem simple_random_sampling_methods_proof (samples : Type) [sample_space : Fintype samples] (p : samples → ℝ) :
  (∀ s1 s2 : samples, p s1 = p s2) → simple_random_sampling_methods samples p :=
by
  intro h
  unfold simple_random_sampling_methods
  constructor
  exact h
  constructor
  sorry -- Proof for is_lottery_drawing_method
  sorry -- Proof for is_random_number_table_method

end simple_random_sampling_methods_proof_l286_286895


namespace fill_parentheses_l286_286354

variable (a b : ℝ)

theorem fill_parentheses :
  1 - a^2 + 2 * a * b - b^2 = 1 - (a^2 - 2 * a * b + b^2) :=
by
  sorry

end fill_parentheses_l286_286354


namespace binom_10_4_eq_210_l286_286634

theorem binom_10_4_eq_210 : Nat.choose 10 4 = 210 :=
  by sorry

end binom_10_4_eq_210_l286_286634


namespace firm_partners_l286_286608

theorem firm_partners
  (P A : ℕ)
  (h1 : P / A = 2 / 63)
  (h2 : P / (A + 35) = 1 / 34) :
  P = 14 :=
by
  sorry

end firm_partners_l286_286608


namespace a_n_plus_1_is_geometric_general_term_formula_l286_286503

-- Define the sequence a_n.
def a : ℕ → ℤ
| 0       => 0  -- a_0 is not given explicitly, we start the sequence from 1.
| (n + 1) => if n = 0 then 1 else 2 * a n + 1

-- Prove that the sequence {a_n + 1} is a geometric sequence.
theorem a_n_plus_1_is_geometric : ∃ r : ℤ, ∀ n : ℕ, (a (n + 1) + 1) / (a n + 1) = r := by
  sorry

-- Find the general formula for a_n.
theorem general_term_formula : ∃ f : ℕ → ℤ, ∀ n : ℕ, a n = f n := by
  sorry

end a_n_plus_1_is_geometric_general_term_formula_l286_286503


namespace arithmetic_sequence_a5_l286_286173

theorem arithmetic_sequence_a5 {a : ℕ → ℕ} 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 2 + a 8 = 12) : 
  a 5 = 6 :=
by
  sorry

end arithmetic_sequence_a5_l286_286173


namespace andrea_still_needs_rhinestones_l286_286453

def total_rhinestones_needed : ℕ := 45
def rhinestones_bought : ℕ := total_rhinestones_needed / 3
def rhinestones_found : ℕ := total_rhinestones_needed / 5
def rhinestones_total_have : ℕ := rhinestones_bought + rhinestones_found
def rhinestones_still_needed : ℕ := total_rhinestones_needed - rhinestones_total_have

theorem andrea_still_needs_rhinestones : rhinestones_still_needed = 21 := by
  rfl

end andrea_still_needs_rhinestones_l286_286453


namespace find_forty_percent_of_N_l286_286560

-- Define the conditions
def condition (N : ℚ) : Prop :=
  (1/4) * (1/3) * (2/5) * N = 14

-- Define the theorem to prove
theorem find_forty_percent_of_N (N : ℚ) (h : condition N) : 0.4 * N = 168 := by
  sorry

end find_forty_percent_of_N_l286_286560


namespace joels_age_when_dad_twice_l286_286203

theorem joels_age_when_dad_twice
  (joel_age_now : ℕ)
  (dad_age_now : ℕ)
  (years : ℕ)
  (H1 : joel_age_now = 5)
  (H2 : dad_age_now = 32)
  (H3 : years = 22)
  (H4 : dad_age_now + years = 2 * (joel_age_now + years))
  : joel_age_now + years = 27 := 
by sorry

end joels_age_when_dad_twice_l286_286203


namespace jose_peanuts_l286_286985

def kenya_peanuts : Nat := 133
def difference_peanuts : Nat := 48

theorem jose_peanuts : (kenya_peanuts - difference_peanuts) = 85 := by
  sorry

end jose_peanuts_l286_286985


namespace probability_first_three_cards_spades_l286_286274

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

end probability_first_three_cards_spades_l286_286274


namespace find_height_of_box_l286_286692

-- Given the conditions
variables (h l w : ℝ)
variables (V : ℝ)

-- Conditions as definitions in Lean
def length_eq_height (h : ℝ) : ℝ := 3 * h
def length_eq_width (w : ℝ) : ℝ := 4 * w
def volume_eq (h l w : ℝ) : ℝ := l * w * h

-- The proof problem: Prove height of the box is 12 given the conditions
theorem find_height_of_box : 
  (∃ h l w, l = 3 * h ∧ l = 4 * w ∧ l * w * h = 3888) → h = 12 :=
by
  sorry

end find_height_of_box_l286_286692


namespace triangle_side_length_l286_286084

theorem triangle_side_length (P Q R : Type) (cos_Q : ℝ) (PQ QR : ℝ) 
  (sin_Q : ℝ) (h_cos_Q : cos_Q = 0.6) (h_PQ : PQ = 10) (h_sin_Q : sin_Q = 0.8) : 
  QR = 50 / 3 :=
by
  sorry

end triangle_side_length_l286_286084


namespace cylinder_volume_eq_sphere_volume_l286_286169

theorem cylinder_volume_eq_sphere_volume (a h R x : ℝ) (h_pos : h > 0) (a_pos : a > 0) (R_pos : R > 0)
  (h_volume_eq : (a - h) * x^2 - a * h * x + 2 * h * R^2 = 0) :
  ∃ x : ℝ, a > h ∧ x > 0 ∧ x < h ∧ x = 2 * R^2 / a ∨ 
           h < a ∧ 0 < x ∧ x = (a * h / (a - h)) - h ∧ R^2 < h^2 / 2 :=
sorry

end cylinder_volume_eq_sphere_volume_l286_286169


namespace value_of_x_squared_plus_reciprocal_squared_l286_286513

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l286_286513


namespace selection_ways_l286_286629

-- The statement of the problem in Lean 4
theorem selection_ways :
  (Nat.choose 50 4) - (Nat.choose 47 4) = 
  (Nat.choose 3 1) * (Nat.choose 47 3) + 
  (Nat.choose 3 2) * (Nat.choose 47 2) + 
  (Nat.choose 3 3) * (Nat.choose 47 1) := 
sorry

end selection_ways_l286_286629


namespace divisibility_by_seven_l286_286726

theorem divisibility_by_seven (n : ℤ) (b : ℤ) (a : ℤ) (h : n = 10 * a + b) 
  (hb : 0 ≤ b) (hb9 : b ≤ 9) (ha : 0 ≤ a) (d : ℤ) (hd : d = a - 2 * b) :
  (2 * n + d) % 7 = 0 ↔ n % 7 = 0 := 
by
  sorry

end divisibility_by_seven_l286_286726


namespace edward_original_amount_l286_286642

theorem edward_original_amount (spent left total : ℕ) (h1 : spent = 13) (h2 : left = 6) (h3 : total = spent + left) : total = 19 := by 
  sorry

end edward_original_amount_l286_286642


namespace necessary_but_not_sufficient_l286_286174

theorem necessary_but_not_sufficient (a : ℝ) (h : a > 1) : 1 / a < 1 :=
by {
  sorry
}

end necessary_but_not_sufficient_l286_286174


namespace GroundBeefSalesTotalRevenue_l286_286246

theorem GroundBeefSalesTotalRevenue :
  let price_regular := 3.50
  let price_lean := 4.25
  let price_extra_lean := 5.00

  let monday_revenue := 198.5 * price_regular +
                        276.2 * price_lean +
                        150.7 * price_extra_lean

  let tuesday_revenue := 210 * (price_regular * 0.90) +
                         420 * (price_lean * 0.90) +
                         150 * (price_extra_lean * 0.90)
  
  let wednesday_revenue := 230 * price_regular +
                           324.6 * 3.75 +
                           120.4 * price_extra_lean

  monday_revenue + tuesday_revenue + wednesday_revenue = 8189.35 :=
by
  sorry

end GroundBeefSalesTotalRevenue_l286_286246


namespace min_value_pq_l286_286795

theorem min_value_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (h1 : p^2 - 8 * q ≥ 0)
  (h2 : 4 * q^2 - 4 * p ≥ 0) :
  p + q ≥ 6 :=
sorry

end min_value_pq_l286_286795


namespace combined_percent_of_6th_graders_l286_286735

theorem combined_percent_of_6th_graders (num_students_pineview : ℕ) 
                                        (percent_6th_pineview : ℝ) 
                                        (num_students_oakridge : ℕ)
                                        (percent_6th_oakridge : ℝ)
                                        (num_students_maplewood : ℕ)
                                        (percent_6th_maplewood : ℝ) 
                                        (total_students : ℝ) :
    num_students_pineview = 150 →
    percent_6th_pineview = 0.15 →
    num_students_oakridge = 180 →
    percent_6th_oakridge = 0.17 →
    num_students_maplewood = 170 →
    percent_6th_maplewood = 0.15 →
    total_students = 500 →
    ((percent_6th_pineview * num_students_pineview) + 
     (percent_6th_oakridge * num_students_oakridge) + 
     (percent_6th_maplewood * num_students_maplewood)) / 
    total_students * 100 = 15.72 :=
by
  sorry

end combined_percent_of_6th_graders_l286_286735


namespace set_equality_proof_l286_286259

theorem set_equality_proof :
  {x : ℕ | x > 1 ∧ x ≤ 3} = {x : ℕ | x = 2 ∨ x = 3} :=
by
  sorry

end set_equality_proof_l286_286259


namespace calculate_profit_l286_286361

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end calculate_profit_l286_286361


namespace quadratic_function_proof_l286_286798

noncomputable def quadratic_function_condition (a b c : ℝ) :=
  ∀ x : ℝ, ((-3 ≤ x ∧ x ≤ 1) → (a * x^2 + b * x + c) ≤ 0) ∧
           ((x < -3 ∨ 1 < x) → (a * x^2 + b * x + c) > 0) ∧
           (a * 2^2 + b * 2 + c) = 5

theorem quadratic_function_proof (a b c : ℝ) (m : ℝ)
  (h : quadratic_function_condition a b c) :
  (a = 1 ∧ b = 2 ∧ c = -3) ∧ (m ≥ -7/9 ↔ ∃ x : ℝ, a * x^2 + b * x + c = 9 * m + 3) :=
by
  sorry

end quadratic_function_proof_l286_286798


namespace limit_f_div_r2_limit_g_div_rh_l286_286833

noncomputable def f (r : ℝ) : ℕ := sorry

def g (r : ℝ) : ℝ := (f r) - π * r^2

theorem limit_f_div_r2 : 
  tendsto (fun r => (f r : ℝ)/r^2) atTop (𝓝 π) :=
sorry

theorem limit_g_div_rh (h : ℝ) (h_lt : h < 2) : 
  tendsto (fun r => g r / r^h) atTop (𝓝 0) :=
sorry

end limit_f_div_r2_limit_g_div_rh_l286_286833


namespace necessary_but_not_sufficient_condition_l286_286179

-- Prove that x^2 ≥ -x is a necessary but not sufficient condition for |x| = x
theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 ≥ -x → |x| = x ↔ x ≥ 0 := 
sorry

end necessary_but_not_sufficient_condition_l286_286179


namespace triangle_side_b_l286_286212

theorem triangle_side_b 
  (a b c : ℝ)
  (B : ℝ)
  (h1 : 1/2 * a * c * Real.sin B = Real.sqrt 3)
  (h2 : B = π / 3)
  (h3 : a^2 + c^2 = 3 * a * c) : b = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_b_l286_286212


namespace divisible_by_323_if_even_l286_286793

theorem divisible_by_323_if_even (n : ℤ) : 
  (20 ^ n + 16 ^ n - 3 ^ n - 1) % 323 = 0 ↔ n % 2 = 0 := 
by 
  sorry

end divisible_by_323_if_even_l286_286793


namespace total_cost_of_fencing_l286_286654

def costOfFencing (lengths rates : List ℝ) : ℝ :=
  List.sum (List.zipWith (· * ·) lengths rates)

theorem total_cost_of_fencing :
  costOfFencing [14, 20, 35, 40, 15, 30, 25]
                [2.50, 3.00, 3.50, 4.00, 2.75, 3.25, 3.75] = 610.00 :=
by
  sorry

end total_cost_of_fencing_l286_286654


namespace multiple_of_first_number_l286_286859

theorem multiple_of_first_number (F S M : ℕ) (hF : F = 15) (hS : S = 55) (h_relation : S = M * F + 10) : M = 3 :=
by
  -- We are given that F = 15, S = 55 and the relation S = M * F + 10
  -- We need to prove that M = 3
  sorry

end multiple_of_first_number_l286_286859


namespace lateral_surface_area_of_rotated_square_l286_286718

noncomputable def lateralSurfaceAreaOfRotatedSquare (side_length : ℝ) : ℝ :=
  2 * Real.pi * side_length * side_length

theorem lateral_surface_area_of_rotated_square :
  lateralSurfaceAreaOfRotatedSquare 1 = 2 * Real.pi :=
by
  sorry

end lateral_surface_area_of_rotated_square_l286_286718


namespace intersection_M_N_l286_286055

def M := {1, 3, 5, 7, 9}

def N := {x : ℤ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} := by
  sorry

end intersection_M_N_l286_286055


namespace problem_statement_l286_286511

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l286_286511


namespace count_special_digits_base7_l286_286032

theorem count_special_digits_base7 : 
  let n := 2401
  let total_valid_numbers := n - 4^4
  total_valid_numbers = 2145 :=
by
  sorry

end count_special_digits_base7_l286_286032


namespace cost_price_of_watch_l286_286609

theorem cost_price_of_watch (CP SP_loss SP_gain : ℝ) (h1 : SP_loss = 0.79 * CP)
  (h2 : SP_gain = 1.04 * CP) (h3 : SP_gain - SP_loss = 140) : CP = 560 := by
  sorry

end cost_price_of_watch_l286_286609


namespace evaluation_result_l286_286784

theorem evaluation_result : 
  (Int.floor (Real.ceil ((15/8 : Real)^2) + (11/3 : Real)) = 7) := 
sorry

end evaluation_result_l286_286784


namespace point_on_line_and_in_first_quadrant_l286_286378

theorem point_on_line_and_in_first_quadrant (x y : ℝ) (hline : y = -2 * x + 3) (hfirst_quadrant : x > 0 ∧ y > 0) :
    (x, y) = (1, 1) :=
by
  sorry

end point_on_line_and_in_first_quadrant_l286_286378


namespace hundredth_odd_positive_integer_l286_286410

theorem hundredth_odd_positive_integer : 2 * 100 - 1 = 199 := 
by
  sorry

end hundredth_odd_positive_integer_l286_286410


namespace triangle_HD_HA_ratio_l286_286450

noncomputable def triangle_ratio (a b c : ℝ) (H : ℝ × ℝ) (AD : ℝ) : ℝ :=
  if a = 9 ∧ b = 40 ∧ c = 41 then
    let orthocenter := (0, a) in
    let HD := orthocenter.2 in
    let HA := orthocenter.2 in
    HD / HA
  else 0

theorem triangle_HD_HA_ratio :
  triangle_ratio 9 40 41 (0, 9) 40 = 1 := by
  sorry

end triangle_HD_HA_ratio_l286_286450


namespace disproving_iff_l286_286481

theorem disproving_iff (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : (a^2 > b^2) ∧ ¬(a > b) :=
by
  sorry

end disproving_iff_l286_286481


namespace problem_statement_l286_286502

def is_ideal_circle (circle : ℝ × ℝ → ℝ) (l : ℝ × ℝ → ℝ) : Prop :=
  ∃ P Q : ℝ × ℝ, (circle P = 0 ∧ circle Q = 0) ∧ (abs (l P) = 1 ∧ abs (l Q) = 1)

noncomputable def line_l (p : ℝ × ℝ) : ℝ := 3 * p.1 + 4 * p.2 - 12

noncomputable def circle_D (p : ℝ × ℝ) : ℝ := (p.1 - 4) ^ 2 + (p.2 - 4) ^ 2 - 16

theorem problem_statement : is_ideal_circle circle_D line_l :=
sorry  -- The proof would go here

end problem_statement_l286_286502


namespace interval_length_correct_l286_286013

def sin_log_interval_sum : ℝ := sorry

theorem interval_length_correct :
  sin_log_interval_sum = 2^π / (1 + 2^π) :=
by
  -- Definitions
  let is_valid_x (x : ℝ) := x < 1 ∧ x > 0 ∧ (Real.sin (Real.log x / Real.log 2)) < 0
  
  -- Assertion
  sorry

end interval_length_correct_l286_286013


namespace rowing_speed_l286_286756

theorem rowing_speed (V_m V_w V_upstream V_downstream : ℝ)
  (h1 : V_upstream = 25)
  (h2 : V_downstream = 65)
  (h3 : V_w = 5) :
  V_m = 45 :=
by
  -- Lean will verify the theorem given the conditions
  sorry

end rowing_speed_l286_286756


namespace total_cost_is_18_l286_286382

-- Definitions based on the conditions
def cost_soda : ℕ := 1
def cost_3_sodas := 3 * cost_soda
def cost_soup := cost_3_sodas
def cost_2_soups := 2 * cost_soup
def cost_sandwich := 3 * cost_soup
def total_cost := cost_3_sodas + cost_2_soups + cost_sandwich

-- The proof statement
theorem total_cost_is_18 : total_cost = 18 := by
  -- proof will go here
  sorry

end total_cost_is_18_l286_286382


namespace probability_top_card_is_five_l286_286127

-- Definitions based on conditions
def total_cards : ℕ := 52
def number_of_fives : ℕ := 4

-- Statement of the theorem
theorem probability_top_card_is_five : (number_of_fives : ℚ) / total_cards = 1 / 13 := by
  sorry

end probability_top_card_is_five_l286_286127


namespace fraction_of_male_first_class_l286_286559

theorem fraction_of_male_first_class (total_passengers : ℕ) (percent_female : ℚ) (percent_first_class : ℚ)
    (females_in_coach : ℕ) (h1 : total_passengers = 120) (h2 : percent_female = 0.45) (h3 : percent_first_class = 0.10)
    (h4 : females_in_coach = 46) :
    (((percent_first_class * total_passengers - (percent_female * total_passengers - females_in_coach)))
    / (percent_first_class * total_passengers))  = 1 / 3 := 
by
  sorry

end fraction_of_male_first_class_l286_286559


namespace fraction_blue_after_doubling_l286_286682

theorem fraction_blue_after_doubling (x : ℕ) (h1 : ∃ x, (2 : ℚ) / 3 * x + (1 : ℚ) / 3 * x = x) :
  ((2 * (2 / 3 * x)) / ((2 / 3 * x) + (1 / 3 * x))) = (4 / 5) := by
  sorry

end fraction_blue_after_doubling_l286_286682


namespace range_of_x_div_y_l286_286316

theorem range_of_x_div_y {x y : ℝ} (hx : 1 < x ∧ x < 6) (hy : 2 < y ∧ y < 8) : 
  (1/8 < x / y) ∧ (x / y < 3) :=
sorry

end range_of_x_div_y_l286_286316


namespace smallest_sum_of_consecutive_integers_is_square_l286_286862

theorem smallest_sum_of_consecutive_integers_is_square : 
  ∃ (n : ℕ), (∑ i in finset.range 20, (n + i) = 250 ∧ is_square (∑ i in finset.range 20, (n + i))) :=
begin
  sorry
end

end smallest_sum_of_consecutive_integers_is_square_l286_286862


namespace rearrange_pairs_l286_286222

theorem rearrange_pairs {a b : ℕ} (hb: b = (2 / 3 : ℚ) * a) (boys_way_museum boys_way_back : ℕ) :
  boys_way_museum = 3 * a ∧ boys_way_back = 4 * b → 
  ∃ c : ℕ, boys_way_museum = 7 * c ∧ b = c := sorry

end rearrange_pairs_l286_286222


namespace calculation_l286_286540

noncomputable def seq (n : ℕ) : ℕ → ℚ := sorry

axiom cond1 : ∀ (n : ℕ), seq (n + 1) - 2 * seq n = 0
axiom cond2 : ∀ (n : ℕ), seq n ≠ 0

theorem calculation :
  (2 * seq 1 + seq 2) / (seq 3 + seq 5) = 1 / 5 :=
  sorry

end calculation_l286_286540


namespace num_possible_values_of_M_l286_286657

theorem num_possible_values_of_M :
  ∃ n : ℕ, n = 8 ∧
  ∃ (a b : ℕ), (10 <= 10*a + b) ∧ (10*a + b < 100) ∧ (9*(a - b) ∈ {k : ℕ | ∃ m : ℕ, k = m^2}) := sorry

end num_possible_values_of_M_l286_286657


namespace sum_series_eq_two_l286_286922

theorem sum_series_eq_two : (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 2 := 
by
  sorry

end sum_series_eq_two_l286_286922


namespace denominator_exceeds_numerator_by_263_l286_286548

def G : ℚ := 736 / 999

theorem denominator_exceeds_numerator_by_263 : 999 - 736 = 263 := by
  -- Since 736 / 999 is the simplest form already, we simply state the obvious difference
  rfl

end denominator_exceeds_numerator_by_263_l286_286548


namespace purely_imaginary_z_implies_m_zero_l286_286676

theorem purely_imaginary_z_implies_m_zero (m : ℝ) :
  m * (m + 1) = 0 → m ≠ -1 := by sorry

end purely_imaginary_z_implies_m_zero_l286_286676


namespace max_value_at_log2_one_l286_286115

noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * (4 : ℝ) ^ x
def domain (x : ℝ) : Prop := x < 1 ∨ x > 3

theorem max_value_at_log2_one :
  (∃ x, domain x ∧ f x = 0) ∧ (∀ y, domain y → f y ≤ 0) :=
by
  sorry

end max_value_at_log2_one_l286_286115


namespace find_b_l286_286003

theorem find_b (b : ℤ) (h_quad : ∃ m : ℤ, (x + m)^2 + 20 = x^2 + b * x + 56) (h_pos : b > 0) : b = 12 :=
sorry

end find_b_l286_286003


namespace smallest_fraction_denominator_l286_286159

theorem smallest_fraction_denominator (p q : ℕ) :
  (1:ℚ) / 2014 < p / q ∧ p / q < (1:ℚ) / 2013 → q = 4027 :=
sorry

end smallest_fraction_denominator_l286_286159


namespace intersection_M_N_l286_286062

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l286_286062


namespace total_grandchildren_l286_286960

-- Define the conditions 
def daughters := 5
def sons := 4
def children_per_daughter := 8 + 7
def children_per_son := 6 + 3

-- State the proof problem
theorem total_grandchildren : daughters * children_per_daughter + sons * children_per_son = 111 :=
by
  sorry

end total_grandchildren_l286_286960


namespace problem_solution_l286_286924

def p (x : ℝ) : ℝ := x^2 - 4*x + 3
def tilde_p (x : ℝ) : ℝ := p (p x)

-- Proof problem: Prove tilde_p 2 = -4 
theorem problem_solution : tilde_p 2 = -4 := sorry

end problem_solution_l286_286924


namespace magnitude_of_z_l286_286021

open Complex

theorem magnitude_of_z :
  ∃ z : ℂ, (1 + 2 * Complex.I) * z = -1 + 3 * Complex.I ∧ Complex.abs z = Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l286_286021


namespace skating_probability_given_skiing_l286_286532

theorem skating_probability_given_skiing (P_A P_B P_A_or_B : ℝ)
    (h1 : P_A = 0.6) (h2 : P_B = 0.5) (h3 : P_A_or_B = 0.7) : 
    (P_A_or_B = P_A + P_B - P_A * P_B) → 
    ((P_A * P_B) / P_B = 0.8) := 
    by
        intros
        sorry

end skating_probability_given_skiing_l286_286532


namespace test_point_third_l286_286687

def interval := (1000, 2000)
def phi := 0.618
def x1 := 1000 + phi * (2000 - 1000)
def x2 := 1000 + 2000 - x1

-- By definition and given the conditions, x3 is computed in a specific manner
def x3 := x2 + 2000 - x1

theorem test_point_third : x3 = 1764 :=
by
  -- Skipping the proof for now
  sorry

end test_point_third_l286_286687


namespace triangle_inequality_l286_286713

theorem triangle_inequality (a b c : ℝ) (α : ℝ) 
  (h_triangle_sides : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  (2 * b * c * Real.cos α) / (b + c) < (b + c - a) ∧ (b + c - a) < (2 * b * c) / a := 
sorry

end triangle_inequality_l286_286713


namespace pace_ratio_l286_286444

variable (P P' D : ℝ)

-- Usual time to reach the office in minutes
def T_usual := 120

-- Time to reach the office on the late day in minutes
def T_late := 140

-- Distance to the office is the same
def office_distance_usual := P * T_usual
def office_distance_late := P' * T_late

theorem pace_ratio (h : office_distance_usual = office_distance_late) : P' / P = 6 / 7 :=
by
  sorry

end pace_ratio_l286_286444


namespace average_height_students_l286_286857

/-- Given the average heights of female and male students, and the ratio of men to women, the average height -/
theorem average_height_students
  (avg_female_height : ℕ)
  (avg_male_height : ℕ)
  (ratio_men_women : ℕ)
  (h1 : avg_female_height = 170)
  (h2 : avg_male_height = 182)
  (h3 : ratio_men_women = 5) :
  (avg_female_height + 5 * avg_male_height) / (1 + 5) = 180 :=
by
  sorry

end average_height_students_l286_286857


namespace initial_amount_l286_286782

-- Define the given conditions
def amount_spent : ℕ := 16
def amount_left : ℕ := 2

-- Define the statement that we want to prove
theorem initial_amount : amount_spent + amount_left = 18 :=
by
  sorry

end initial_amount_l286_286782


namespace weekly_rental_cost_l286_286621

theorem weekly_rental_cost (W : ℝ) 
  (monthly_cost : ℝ := 40)
  (months_in_year : ℝ := 12)
  (weeks_in_year : ℝ := 52)
  (savings : ℝ := 40)
  (total_year_cost_month : ℝ := months_in_year * monthly_cost)
  (total_year_cost_week : ℝ := total_year_cost_month + savings) :
  (total_year_cost_week / weeks_in_year) = 10 :=
by 
  sorry

end weekly_rental_cost_l286_286621


namespace nth_odd_positive_integer_is_199_l286_286418

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l286_286418


namespace tetrahedron_ratio_l286_286825

open Real

theorem tetrahedron_ratio (a b : ℝ) (h1 : a = PA ∧ PB = a) (h2 : PC = b ∧ AB = b ∧ BC = b ∧ CA = b) (h3 : a < b) :
  (sqrt 6 - sqrt 2) / 2 < a / b ∧ a / b < 1 :=
by
  sorry

end tetrahedron_ratio_l286_286825


namespace pen_rubber_length_difference_l286_286272

theorem pen_rubber_length_difference (P R : ℕ) 
    (h1 : P = R + 3)
    (h2 : P = 12 - 2) 
    (h3 : R + P + 12 = 29) : 
    P - R = 3 :=
  sorry

end pen_rubber_length_difference_l286_286272


namespace shaded_region_area_correct_l286_286907

noncomputable def shaded_region_area (side_length : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
if 0 < beta ∧ beta < Real.pi / 2 ∧ cos_beta = 3 / 5 then
  2 / 5
else
  0

theorem shaded_region_area_correct :
  shaded_region_area 2 β (3 / 5) = 2 / 5 :=
by
  -- conditions
  have beta_cond : 0 < β ∧ β < Real.pi / 2 := sorry
  have cos_beta_cond : cos β = 3 / 5 := sorry
  -- we will finish this proof assuming above have been proved.
  exact if_pos ⟨beta_cond, cos_beta_cond⟩

end shaded_region_area_correct_l286_286907


namespace inequality_holds_for_all_real_l286_286452

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 1 ≥ 2 * |x| := sorry

end inequality_holds_for_all_real_l286_286452


namespace cube_root_of_neg_27_l286_286464

theorem cube_root_of_neg_27 : ∃ y : ℝ, y^3 = -27 ∧ y = -3 := by
  sorry

end cube_root_of_neg_27_l286_286464


namespace triangle_shape_right_angled_l286_286977

theorem triangle_shape_right_angled (a b c : ℝ) (A B C : ℝ) (h1 : b^2 = c^2 + a^2 - c * a) (h2 : Real.sin A = 2 * Real.sin C) :
    ∃ (D : Type) (triangle_shape : TriangeShape D), triangle_shape = TriangeShape.RightAngled :=
by
  sorry

end triangle_shape_right_angled_l286_286977


namespace sequence_value_proof_l286_286696

theorem sequence_value_proof : 
  (∃ (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    (∀ n : ℕ, a (2 * n) = 2 * n * a n) ∧ 
    a (2^50) = 2^1276) :=
sorry

end sequence_value_proof_l286_286696


namespace relative_speed_of_trains_l286_286244

def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

theorem relative_speed_of_trains 
  (speed_train1_kmph : ℕ) 
  (speed_train2_kmph : ℕ) 
  (h1 : speed_train1_kmph = 216) 
  (h2 : speed_train2_kmph = 180) : 
  kmph_to_mps speed_train1_kmph - kmph_to_mps speed_train2_kmph = 10 := 
by 
  sorry

end relative_speed_of_trains_l286_286244


namespace eve_spending_l286_286645

-- Definitions of the conditions
def cost_mitt : ℝ := 14.00
def cost_apron : ℝ := 16.00
def cost_utensils : ℝ := 10.00
def cost_knife : ℝ := 2 * cost_utensils -- Twice the amount of the utensils
def discount_rate : ℝ := 0.25
def num_nieces : ℝ := 3

-- Total cost before the discount for one kit
def total_cost_one_kit : ℝ :=
  cost_mitt + cost_apron + cost_utensils + cost_knife

-- Discount for one kit
def discount_one_kit : ℝ := 
  total_cost_one_kit * discount_rate

-- Discounted price for one kit
def discounted_cost_one_kit : ℝ :=
  total_cost_one_kit - discount_one_kit

-- Total cost for all kits
def total_cost_all_kits : ℝ :=
  num_nieces * discounted_cost_one_kit

-- The theorem statement
theorem eve_spending : total_cost_all_kits = 135.00 :=
by sorry

end eve_spending_l286_286645


namespace eve_total_spend_l286_286646

def hand_mitts_cost : ℝ := 14.00
def apron_cost : ℝ := 16.00
def utensils_cost : ℝ := 10.00
def knife_cost : ℝ := 2 * utensils_cost
def discount_percent : ℝ := 0.25
def nieces_count : ℕ := 3

def total_cost_before_discount : ℝ :=
  (hand_mitts_cost + apron_cost + utensils_cost + knife_cost) * nieces_count

def discount_amount : ℝ :=
  discount_percent * total_cost_before_discount

def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount_amount

theorem eve_total_spend : total_cost_after_discount = 135.00 := by
  sorry

end eve_total_spend_l286_286646


namespace min_value_frac_sum_l286_286486

variable {a b c : ℝ}

theorem min_value_frac_sum (h1 : 0 < a ∧ a < 1) (h2 : 0 < b ∧ b < 1) (h3 : 0 < c ∧ c < 1) (h4 : a * b + b * c + c * a = 1) : 
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = (9 + 3 * Real.sqrt 3) / 2 :=
  sorry

end min_value_frac_sum_l286_286486


namespace total_cats_l286_286094

-- Define the conditions as constants
def asleep_cats : ℕ := 92
def awake_cats : ℕ := 6

-- State the theorem that proves the total number of cats
theorem total_cats : asleep_cats + awake_cats = 98 := 
by
  -- Proof omitted
  sorry

end total_cats_l286_286094


namespace part1_proof_part2_proof_l286_286990

variable {a : ℕ → ℝ} -- sequence a_n
variable (S : ℕ → ℝ) -- sum sequence S_n
variable (q : ℝ) -- common ratio
variable (c : ℝ) -- constant c

-- Definitions due to conditions
def is_geom_seq (a: ℕ → ℝ) (q: ℝ) : Prop := ∀ n, a (n + 1) = a n * q
def is_sum_geom (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (0 to n-1).sum (λ k, a k)
def sum_geom_series (a₁ q: ℝ) (n: ℕ) : ℝ := if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

-- condition premises
def conditions (a : ℕ → ℝ) (q a₁ c: ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ a 0 = a₁ ∧ a₁ > 0 ∧ q > 0

-- (κκκ) Proof statement for Part (1) : ∀ n, (lg (S(n)+lg(S(n+2)) )/2 < lg S(n+1)
theorem part1_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (q a₁ : ℝ) (h : conditions a q a₁ c) :
  ∀ n, (Math.log(S n) + Math.log(S (n + 2))) / 2 < Math.log(S (n + 1)) := 
sorry

-- (κκκ) Proof statement for Part (2): ∀ n, ∀ c, (lg (S(n)-c)+lg(S(n+2)-c) )/2 ≠ lg S(n+1)-c
theorem part2_proof (a : ℕ → ℝ) (S : ℕ → ℝ) (q a₁: ℝ)  (h : conditions a q a₁ c) :
  ∀ n, ∀ c, ¬((Math.log (S n - c) + Math.log (S (n + 2) - c)) / 2 = Math.log (S (n + 1) - c)) := 
sorry

end part1_proof_part2_proof_l286_286990


namespace arithmetic_sequence_solution_l286_286576

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (d : ℝ) 
(h1 : d ≠ 0) 
(h2 : a 1 = 2) 
(h3 : a 1 * a 4 = (a 2) ^ 2) :
∀ n, a n = 2 * n :=
by 
  sorry

end arithmetic_sequence_solution_l286_286576


namespace contrapositive_honor_roll_l286_286996

variable (Student : Type) (scores_hundred : Student → Prop) (honor_roll_qualifies : Student → Prop)

theorem contrapositive_honor_roll (s : Student) :
  (¬ honor_roll_qualifies s) → (¬ scores_hundred s) := 
sorry

end contrapositive_honor_roll_l286_286996


namespace number_of_juniors_l286_286200

theorem number_of_juniors
  (T : ℕ := 28)
  (hT : T = 28)
  (x y : ℕ)
  (hxy : x = y)
  (J S : ℕ)
  (hx : x = J / 4)
  (hy : y = S / 10)
  (hJS : J + S = T) :
  J = 8 :=
by sorry

end number_of_juniors_l286_286200


namespace fraction_calculation_l286_286292

theorem fraction_calculation :
  ( (12^4 + 324) * (26^4 + 324) * (38^4 + 324) * (50^4 + 324) * (62^4 + 324)) /
  ( (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324)) =
  73.481 :=
by
  sorry

end fraction_calculation_l286_286292


namespace unique_solution_for_a_l286_286792

theorem unique_solution_for_a (a : ℝ) :
  (∃! (x y : ℝ), 
    (x * Real.cos a + y * Real.sin a = 5 * Real.cos a + 2 * Real.sin a) ∧
    (-3 ≤ x + 2 * y ∧ x + 2 * y ≤ 7) ∧
    (-9 ≤ 3 * x - 4 * y ∧ 3 * x - 4 * y ≤ 1)) ↔ 
  (∃ k : ℤ, a = Real.arctan 4 + k * Real.pi ∨ a = -Real.arctan 2 + k * Real.pi) :=
sorry

end unique_solution_for_a_l286_286792


namespace point_on_parabola_dist_3_from_focus_l286_286797

def parabola (p : ℝ × ℝ) : Prop := (p.snd)^2 = 4 * p.fst

def focus : ℝ × ℝ := (1, 0)

theorem point_on_parabola_dist_3_from_focus :
  ∃ y: ℝ, ∃ x: ℝ, (parabola (x, y) ∧ (x = 2) ∧ (y = 2 * Real.sqrt 2 ∨ y = -2 * Real.sqrt 2) ∧ (Real.sqrt ((x - focus.fst)^2 + (y - focus.snd)^2) = 3)) :=
by
  sorry

end point_on_parabola_dist_3_from_focus_l286_286797


namespace find_original_class_strength_l286_286890

-- Definitions based on given conditions
def original_average_age : ℝ := 40
def additional_students : ℕ := 12
def new_students_average_age : ℝ := 32
def decrease_in_average : ℝ := 4
def new_average_age : ℝ := original_average_age - decrease_in_average

-- The equation setup
theorem find_original_class_strength (N : ℕ) (T : ℝ) 
  (h1 : T = original_average_age * N) 
  (h2 : T + additional_students * new_students_average_age = new_average_age * (N + additional_students)) : 
  N = 12 := 
sorry

end find_original_class_strength_l286_286890


namespace smallest_possible_denominator_l286_286161

theorem smallest_possible_denominator :
  ∃ p q : ℕ, q < 4027 ∧ (1/2014 : ℚ) < p / q ∧ p / q < (1/2013 : ℚ) → ∃ q : ℕ, q = 4027 :=
by
  sorry

end smallest_possible_denominator_l286_286161


namespace sum_of_solutions_l286_286600

theorem sum_of_solutions :
  let eq := (4 * x + 3) * (3 * x - 7) = 0 in
  (is_solution eq (-3/4) ∧ is_solution eq (7/3)) → 
  (-3 / 4 + 7 / 3 = 19 / 12) :=
by 
  intros eq h
  sorry

end sum_of_solutions_l286_286600


namespace isosceles_vertex_angle_l286_286874

noncomputable def golden_ratio := (1 + Real.sqrt 5) / 2

theorem isosceles_vertex_angle (a b θ : ℝ)
  (h1 : a = golden_ratio * b) :
  ∃ θ, θ = 36 :=
by
  sorry

end isosceles_vertex_angle_l286_286874


namespace clock_hands_form_right_angle_at_180_over_11_l286_286136

-- Define the angular speeds as constants
def ω_hour : ℝ := 0.5  -- Degrees per minute
def ω_minute : ℝ := 6  -- Degrees per minute

-- Function to calculate the angle of the hour hand after t minutes
def angle_hour (t : ℝ) : ℝ := ω_hour * t

-- Function to calculate the angle of the minute hand after t minutes
def angle_minute (t : ℝ) : ℝ := ω_minute * t

-- Theorem: Prove the two hands form a right angle at the given time
theorem clock_hands_form_right_angle_at_180_over_11 : 
  ∃ t : ℝ, (6 * t - 0.5 * t = 90) ∧ t = 180 / 11 :=
by 
  -- This is where the proof would go, but we skip it with sorry
  sorry

end clock_hands_form_right_angle_at_180_over_11_l286_286136


namespace right_angle_sides_of_isosceles_right_triangle_l286_286171

def is_on_line (x y a b c : ℝ) : Prop := a * x + b * y + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

theorem right_angle_sides_of_isosceles_right_triangle
  (C : ℝ × ℝ)
  (hyp_line : ℝ → ℝ → Prop)
  (side_AC side_BC : ℝ → ℝ → Prop)
  (H1 : C = (3, -2))
  (H2 : hyp_line = is_on_line 3 (-1) 2)
  (H3 : side_AC = is_on_line 2 1 (-4))
  (H4 : side_BC = is_on_line 1 (-2) (-7))
  (H5 : ∃ x y, side_BC (3) y ∧ side_AC x (-2)) :
  side_AC = is_on_line 2 1 (-4) ∧ side_BC = is_on_line 1 (-2) (-7) :=
by
  sorry

end right_angle_sides_of_isosceles_right_triangle_l286_286171


namespace tan_neg_3780_eq_zero_l286_286777

theorem tan_neg_3780_eq_zero : Real.tan (-3780 * Real.pi / 180) = 0 := 
by 
  sorry

end tan_neg_3780_eq_zero_l286_286777


namespace intersection_eq_l286_286054

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {x | 2 * x > 7}

theorem intersection_eq : M ∩ N = {5, 7, 9} := sorry

end intersection_eq_l286_286054


namespace james_profit_l286_286360

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end james_profit_l286_286360


namespace range_of_a_l286_286344

def is_in_third_quadrant (A : ℝ × ℝ) : Prop :=
  A.1 < 0 ∧ A.2 < 0

theorem range_of_a (a : ℝ) (h : is_in_third_quadrant (a, a - 1)) : a < 0 :=
by
  sorry

end range_of_a_l286_286344


namespace sum_of_coeffs_expansion_l286_286297

theorem sum_of_coeffs_expansion (d : ℝ) : 
    let expr := -(4 - d) * (d + 2 * (4 - d))
    let poly := -d^2 + 12 * d - 32
    let coeff_sum := -1 + 12 - 32
in coeff_sum = -21 := 
by
    let expr := -(4 - d) * (d + 2 * (4 - d))
    let poly := -d^2 + 12 * d - 32
    let coeff_sum := -1 + 12 - 32
    exact rfl

end sum_of_coeffs_expansion_l286_286297


namespace z_in_second_quadrant_l286_286577

def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (i : ℂ) (hi : i^2 = -1) (h : z * (1 + i^3) = i) : 
  is_second_quadrant z := by
  sorry

end z_in_second_quadrant_l286_286577


namespace max_area_quadrilateral_cdfg_l286_286612

theorem max_area_quadrilateral_cdfg (s : ℝ) (x : ℝ)
  (h1 : s = 1) (h2 : x > 0) (h3 : x < s) (h4 : AE = x) (h5 : AF = x) : 
  ∃ x, x > 0 ∧ x < 1 ∧ (1 - x) * x ≤ 5 / 8 :=
sorry

end max_area_quadrilateral_cdfg_l286_286612


namespace betty_age_l286_286886

theorem betty_age (A M B : ℕ) (h1 : A = 2 * M) (h2 : A = 4 * B) (h3 : M = A - 22) : B = 11 :=
by
  sorry

end betty_age_l286_286886


namespace john_paint_area_l286_286827

noncomputable def area_to_paint (length width height openings : ℝ) : ℝ :=
  let wall_area := 2 * (length * height) + 2 * (width * height)
  let ceiling_area := length * width
  let total_area := wall_area + ceiling_area
  total_area - openings

theorem john_paint_area :
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  2 * (area_to_paint length width height openings) = 1300 :=
by
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  sorry

end john_paint_area_l286_286827


namespace original_fraction_eq_two_thirds_l286_286585

theorem original_fraction_eq_two_thirds (a b : ℕ) (h : (a^3 : ℚ) / (b + 3) = 2 * (a / b)) : a = 2 ∧ b = 3 :=
by {
  sorry
}

end original_fraction_eq_two_thirds_l286_286585


namespace num_tiles_visited_l286_286760

theorem num_tiles_visited (width length : ℕ) (h_w : width = 12) (h_l : length = 18) :
  width + length - Nat.gcd width length = 24 :=
by
  rw [h_w, h_l]
  simp [Nat.gcd]
  sorry

end num_tiles_visited_l286_286760


namespace number_of_sandwiches_l286_286915

-- Defining the conditions
def kinds_of_meat := 12
def kinds_of_cheese := 11
def kinds_of_bread := 5

-- Combinations calculation
def choose_one (n : Nat) := n
def choose_three (n : Nat) := Nat.choose n 3

-- Proof statement to show that the total number of sandwiches is 9900
theorem number_of_sandwiches : (choose_one kinds_of_meat) * (choose_three kinds_of_cheese) * (choose_one kinds_of_bread) = 9900 := by
  sorry

end number_of_sandwiches_l286_286915


namespace subset_A_if_inter_eq_l286_286680

variable {B : Set ℝ}

def A : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem subset_A_if_inter_eq:
  A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = { x | 0 < x ∧ x < 2 } :=
by
  sorry

end subset_A_if_inter_eq_l286_286680


namespace distance_traveled_eq_2400_l286_286397

-- Definitions of the conditions
def circumference_front : ℕ := 30
def circumference_back : ℕ := 32
def revolutions_difference : ℕ := 5

-- Define the number of revolutions made by the back wheel
def revs_back (R : ℕ) := R

-- Define the number of revolutions made by the front wheel
def revs_front (R : ℕ) := R + revolutions_difference

-- Define the distance traveled by the back and front wheels
def distance_back (R : ℕ) : ℕ := revs_back R * circumference_back
def distance_front (R : ℕ) : ℕ := revs_front R * circumference_front

-- State the theorem without a proof (using sorry)
theorem distance_traveled_eq_2400 :
  ∃ R : ℕ, distance_back R = 2400 ∧ distance_back R = distance_front R :=
by {
  sorry
}

end distance_traveled_eq_2400_l286_286397


namespace ben_paperclip_day_l286_286137

theorem ben_paperclip_day :
  ∃ k : ℕ, k = 6 ∧ (∀ n : ℕ, n = k → 5 * 3^n > 500) :=
sorry

end ben_paperclip_day_l286_286137


namespace painting_price_difference_l286_286073

theorem painting_price_difference :
  let previous_painting := 9000
  let recent_painting := 44000
  let five_times_more := 5 * previous_painting + previous_painting
  five_times_more - recent_painting = 10000 :=
by
  intros
  sorry

end painting_price_difference_l286_286073


namespace problem_l286_286320

noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem problem (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) : 
  ((f b - f a) / (b - a) < 1 / (a * (a + 1))) :=
by
  sorry -- Proof steps go here

end problem_l286_286320


namespace remaining_volume_of_cube_with_hole_l286_286133

theorem remaining_volume_of_cube_with_hole : 
  let side_length_cube := 8 
  let side_length_hole := 4 
  let volume_cube := side_length_cube ^ 3 
  let cross_section_hole := side_length_hole ^ 2
  let volume_hole := cross_section_hole * side_length_cube
  let remaining_volume := volume_cube - volume_hole
  remaining_volume = 384 := by {
    sorry
  }

end remaining_volume_of_cube_with_hole_l286_286133


namespace ants_meet_at_q_one_l286_286547

noncomputable def ant_meeting_problem (q : ℚ) : Prop :=
  ∀ (n : ℕ), n > 0 →
  ∃ (ε ε' : Fin n → ℂ),
    ε ∈ {1, -1, Complex.i, -Complex.i} ∧
    ε' ∈ {1, -1, Complex.i, -Complex.i} ∧
    (∑ i in Finset.range n, ε i * q^i : ℂ) =
    (∑ i in Finset.range n, ε' i * q^i : ℂ) ∧
    ε ≠ ε'

theorem ants_meet_at_q_one : ∀ q : ℚ, 0 < q → ant_meeting_problem q → q = 1 := 
begin
  intros q hq h,
  sorry
end

end ants_meet_at_q_one_l286_286547


namespace min_x_plus_y_of_positive_l286_286315

open Real

theorem min_x_plus_y_of_positive (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end min_x_plus_y_of_positive_l286_286315


namespace maxValue_is_6084_over_17_l286_286064

open Real

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem maxValue_is_6084_over_17 (x y : ℝ) (h : x + y = 5) :
  maxValue x y h ≤ 6084 / 17 := 
sorry

end maxValue_is_6084_over_17_l286_286064


namespace least_number_to_divisible_by_11_l286_286261

theorem least_number_to_divisible_by_11 (n : ℕ) (h : n = 11002) : ∃ k : ℕ, (n + k) % 11 = 0 ∧ ∀ m : ℕ, (n + m) % 11 = 0 → m ≥ k :=
by
  sorry

end least_number_to_divisible_by_11_l286_286261


namespace time_to_cross_platform_l286_286614

-- Definition of the given conditions
def length_of_train : ℕ := 1500 -- in meters
def time_to_cross_tree : ℕ := 120 -- in seconds
def length_of_platform : ℕ := 500 -- in meters
def speed : ℚ := length_of_train / time_to_cross_tree -- speed in meters per second

-- Definition of the total distance to cross the platform
def total_distance : ℕ := length_of_train + length_of_platform

-- Theorem to prove the time taken to cross the platform
theorem time_to_cross_platform : (total_distance / speed) = 160 :=
by
  -- Placeholder for the proof
  sorry

end time_to_cross_platform_l286_286614


namespace _l286_286834

noncomputable theorem complex_sum_product_identity :
  let z : ℂ := (1 - complex.I * real.sqrt 3) / 2 in
  (∑ k in finset.range (16 + 1), z ^ (k^2)) * (∑ k in finset.range (16 + 1), (z ^ (k^2))⁻¹) = 9 :=
by
  sorry

end _l286_286834


namespace smallest_b_to_the_a_l286_286229

theorem smallest_b_to_the_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = 2^2023) : b^a = 1 :=
by
  -- Proof steps go here
  sorry

end smallest_b_to_the_a_l286_286229


namespace fraction_to_terminating_decimal_l286_286786

theorem fraction_to_terminating_decimal :
  (47 : ℚ) / (2^2 * 5^4) = 0.0188 :=
sorry

end fraction_to_terminating_decimal_l286_286786


namespace sum_series_l286_286919

noncomputable def series_sum := (∑' n : ℕ, (4 * (n + 1) - 2) / 3^(n + 1))

theorem sum_series : series_sum = 4 := by
  sorry

end sum_series_l286_286919


namespace total_cost_of_books_l286_286627

def book_cost (num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook : ℕ) : ℕ :=
  (num_mathbooks * cost_mathbook) + (num_artbooks * cost_artbook) + (num_sciencebooks * cost_sciencebook)

theorem total_cost_of_books :
  let num_mathbooks := 2
  let num_artbooks := 3
  let num_sciencebooks := 6
  let cost_mathbook := 3
  let cost_artbook := 2
  let cost_sciencebook := 3
  book_cost num_mathbooks num_artbooks num_sciencebooks cost_mathbook cost_artbook cost_sciencebook = 30 :=
by
  sorry

end total_cost_of_books_l286_286627


namespace intersection_correct_l286_286057

-- Define the sets M and N
def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x | 2 * x > 7}

-- Define the expected intersection result
def expected_intersection : Set ℝ := {5, 7, 9}

-- State the theorem
theorem intersection_correct : ∀ x, x ∈ M ∩ N ↔ x ∈ expected_intersection :=
by
  sorry

end intersection_correct_l286_286057


namespace unique_solution_of_fraction_eq_l286_286580

theorem unique_solution_of_fraction_eq (x : ℝ) : (1 / (x - 1) = 2 / (x - 2)) ↔ (x = 0) :=
by
  sorry

end unique_solution_of_fraction_eq_l286_286580


namespace total_practice_hours_l286_286618

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

end total_practice_hours_l286_286618


namespace inequality1_inequality2_l286_286322

-- Problem 1
def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem inequality1 (x : ℝ) : f x > 2 ↔ x < -2/3 ∨ x > 0 := sorry

-- Problem 2
def g (x : ℝ) : ℝ := f x + f (-x)

theorem inequality2 (k : ℝ) (h : ∀ x : ℝ, |k - 1| < g x) : -3 < k ∧ k < 5 := sorry

end inequality1_inequality2_l286_286322


namespace sum_x1_x2_range_l286_286694

variable {x₁ x₂ : ℝ}

-- Definition of x₁ being the real root of the equation x * 2^x = 1
def is_root_1 (x : ℝ) : Prop :=
  x * 2^x = 1

-- Definition of x₂ being the real root of the equation x * log_2 x = 1
def is_root_2 (x : ℝ) : Prop :=
  x * Real.log x / Real.log 2 = 1

theorem sum_x1_x2_range (hx₁ : is_root_1 x₁) (hx₂ : is_root_2 x₂) :
  2 < x₁ + x₂ :=
sorry

end sum_x1_x2_range_l286_286694


namespace dayAfter73DaysFromFridayAnd9WeeksLater_l286_286873

-- Define the days of the week as a data type
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
deriving DecidableEq, Repr

open Weekday

-- Function to calculate the day of the week after a given number of days
def addDays (start_day : Weekday) (days : ℕ) : Weekday :=
  match start_day with
  | Sunday    => match days % 7 with | 0 => Sunday    | 1 => Monday | 2 => Tuesday | 3 => Wednesday | 4 => Thursday | 5 => Friday | 6 => Saturday | _ => Sunday
  | Monday    => match days % 7 with | 0 => Monday    | 1 => Tuesday | 2 => Wednesday | 3 => Thursday | 4 => Friday | 5 => Saturday | 6 => Sunday | _ => Monday
  | Tuesday   => match days % 7 with | 0 => Tuesday   | 1 => Wednesday | 2 => Thursday | 3 => Friday | 4 => Saturday | 5 => Sunday | 6 => Monday | _ => Tuesday
  | Wednesday => match days % 7 with | 0 => Wednesday | 1 => Thursday | 2 => Friday | 3 => Saturday | 4 => Sunday | 5 => Monday | 6 => Tuesday | _ => Wednesday
  | Thursday  => match days % 7 with | 0 => Thursday  | 1 => Friday | 2 => Saturday | 3 => Sunday | 4 => Monday | 5 => Tuesday | 6 => Wednesday | _ => Thursday
  | Friday    => match days % 7 with | 0 => Friday    | 1 => Saturday | 2 => Sunday | 3 => Monday | 4 => Tuesday | 5 => Wednesday | 6 => Thursday | _ => Friday
  | Saturday  => match days % 7 with | 0 => Saturday  | 1 => Sunday | 2 => Monday | 3 => Tuesday | 4 => Wednesday | 5 => Thursday | 6 => Friday | _ => Saturday

-- Theorem that proves the required solution
theorem dayAfter73DaysFromFridayAnd9WeeksLater : addDays Friday 73 = Monday ∧ addDays Monday (9 * 7) = Monday := 
by
  -- Placeholder to acknowledge proof requirements
  sorry

end dayAfter73DaysFromFridayAnd9WeeksLater_l286_286873


namespace find_side_b_l286_286209

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  1 / 2 * a * c * real.sin B

noncomputable def law_of_cosines (a b c : ℝ) (B : ℝ) : ℝ :=
  a^2 + c^2 - 2 * a * c * real.cos B

theorem find_side_b (a b c : ℝ) (B : ℝ) (area : ℝ)
  (h1 : area = √3) 
  (h2 : B = real.pi / 3)
  (h3 : a^2 + c^2 = 3 * a * c) :
  b = 2 * real.sqrt 2 :=
by
  sorry

end find_side_b_l286_286209


namespace no_solns_to_equation_l286_286788

noncomputable def no_solution : Prop :=
  ∀ (n m r : ℕ), (1 ≤ n) → (1 ≤ m) → (1 ≤ r) → n^5 + 49^m ≠ 1221^r

theorem no_solns_to_equation : no_solution :=
sorry

end no_solns_to_equation_l286_286788


namespace jose_peanuts_l286_286988

/-- If Kenya has 133 peanuts and this is 48 more than what Jose has,
    then Jose has 85 peanuts. -/
theorem jose_peanuts (j k : ℕ) (h1 : k = j + 48) (h2 : k = 133) : j = 85 :=
by
  -- Proof goes here
  sorry

end jose_peanuts_l286_286988


namespace circle_area_increase_l286_286238

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  percentage_increase = 125 :=
by
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  sorry

end circle_area_increase_l286_286238


namespace largest_angle_in_triangle_l286_286734

theorem largest_angle_in_triangle (A B C : ℝ) 
  (h_sum : A + B = 126) 
  (h_diff : B = A + 40) 
  (h_triangle : A + B + C = 180) : max A (max B C) = 83 := 
by
  sorry

end largest_angle_in_triangle_l286_286734


namespace driving_time_eqn_l286_286134

open Nat

-- Define the variables and constants
def avg_speed_before := 80 -- km/h
def stop_time := 1 / 3 -- hour
def avg_speed_after := 100 -- km/h
def total_distance := 250 -- km
def total_time := 3 -- hours

variable (t : ℝ) -- the time in hours before the stop

-- State the main theorem
theorem driving_time_eqn :
  avg_speed_before * t + avg_speed_after * (total_time - stop_time - t) = total_distance := by
  sorry

end driving_time_eqn_l286_286134


namespace series_sum_eq_l286_286291

theorem series_sum_eq : 
  (∑' n, (4 * n + 3) / ((4 * n - 2) ^ 2 * (4 * n + 2) ^ 2)) = 1 / 128 := by
sorry

end series_sum_eq_l286_286291


namespace A_odot_B_correct_l286_286989

open Set

def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | x < 0 ∨ x > 2 }
def A_union_B : Set ℝ := A ∪ B
def A_inter_B : Set ℝ := A ∩ B
def A_odot_B : Set ℝ := { x | x ∈ A_union_B ∧ x ∉ A_inter_B }

theorem A_odot_B_correct : A_odot_B = (Iio 0) ∪ Icc 1 2 :=
by
  sorry

end A_odot_B_correct_l286_286989


namespace san_francisco_superbowl_probability_l286_286237

theorem san_francisco_superbowl_probability
  (P_play P_not_play : ℝ)
  (k : ℝ)
  (h1 : P_play = k * P_not_play)
  (h2 : P_play + P_not_play = 1) :
  k > 0 :=
sorry

end san_francisco_superbowl_probability_l286_286237


namespace jess_height_l286_286544

variable (Jana_height Kelly_height Jess_height : ℕ)

-- Conditions
axiom Jana_height_eq : Jana_height = 74
axiom Jana_taller_than_Kelly : Jana_height = Kelly_height + 5
axiom Kelly_shorter_than_Jess : Kelly_height = Jess_height - 3

-- Prove Jess's height
theorem jess_height : Jess_height = 72 := by
  -- Proof goes here
  sorry

end jess_height_l286_286544


namespace find_missing_number_l286_286732

-- Define the known values
def numbers : List ℕ := [1, 22, 24, 25, 26, 27, 2]
def specified_mean : ℕ := 20
def total_counts : ℕ := 8

-- The theorem statement
theorem find_missing_number : (∀ (x : ℕ), (List.sum (x :: numbers) = specified_mean * total_counts) → x = 33) :=
by
  sorry

end find_missing_number_l286_286732


namespace product_of_b_l286_286472

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 4)

noncomputable def g_inv (b : ℝ) (y : ℝ) : ℝ := (y + 4) / 3

theorem product_of_b (b : ℝ) :
  g b 3 = g_inv b (b + 2) → b = 3 := 
by
  sorry

end product_of_b_l286_286472


namespace number_of_valid_n_l286_286143

theorem number_of_valid_n : 
  (∃ (n : ℕ), ∀ (a b c : ℕ), 8 * a + 88 * b + 888 * c = 8000 → n = a + 2 * b + 3 * c) ↔
  (∃ (n : ℕ), n = 1000) := by 
  sorry

end number_of_valid_n_l286_286143


namespace sum_of_proper_divisors_30_is_42_l286_286598

def is_proper_divisor (n d : ℕ) : Prop := d ∣ n ∧ d ≠ n

-- The set of proper divisors of 30.
def proper_divisors_30 : Finset ℕ := {1, 2, 3, 5, 6, 10, 15}

-- The sum of all proper divisors of 30.
def sum_proper_divisors_30 : ℕ := proper_divisors_30.sum id

theorem sum_of_proper_divisors_30_is_42 : sum_proper_divisors_30 = 42 := 
by
  -- Proof can be filled in here
  sorry

end sum_of_proper_divisors_30_is_42_l286_286598


namespace find_term_number_l286_286668

theorem find_term_number :
  ∃ n : ℕ, (2 * (5 : ℝ)^(1/2) = (3 * (n : ℝ) - 1)^(1/2)) ∧ n = 7 :=
sorry

end find_term_number_l286_286668


namespace eugene_used_six_boxes_of_toothpicks_l286_286937

-- Define the given conditions
def toothpicks_per_card : ℕ := 75
def total_cards : ℕ := 52
def unused_cards : ℕ := 16
def toothpicks_per_box : ℕ := 450

-- Compute the required result
theorem eugene_used_six_boxes_of_toothpicks :
  ((total_cards - unused_cards) * toothpicks_per_card) / toothpicks_per_box = 6 :=
by
  sorry

end eugene_used_six_boxes_of_toothpicks_l286_286937


namespace tricycle_wheel_count_l286_286093

theorem tricycle_wheel_count (bicycles wheels_per_bicycle tricycles total_wheels : ℕ)
  (h1 : bicycles = 16)
  (h2 : wheels_per_bicycle = 2)
  (h3 : tricycles = 7)
  (h4 : total_wheels = 53)
  (h5 : total_wheels = (bicycles * wheels_per_bicycle) + (tricycles * (3 : ℕ))) : 
  (3 : ℕ) = 3 := by
  sorry

end tricycle_wheel_count_l286_286093


namespace rate_per_kg_of_grapes_l286_286631

-- Define the conditions 
namespace Problem

-- Given conditions
variables (G : ℝ) (rate_mangoes : ℝ := 55) (cost_paid : ℝ := 1055)
variables (kg_grapes : ℝ := 8) (kg_mangoes : ℝ := 9)

-- Statement to prove
theorem rate_per_kg_of_grapes : 8 * G + 9 * rate_mangoes = cost_paid → G = 70 := 
by
  intro h
  sorry -- proof goes here

end Problem

end rate_per_kg_of_grapes_l286_286631


namespace chris_and_fiona_weight_l286_286467

theorem chris_and_fiona_weight (c d e f : ℕ) (h1 : c + d = 330) (h2 : d + e = 290) (h3 : e + f = 310) : c + f = 350 :=
by
  sorry

end chris_and_fiona_weight_l286_286467


namespace monotonicity_of_f_l286_286954

noncomputable def f (x : ℝ) : ℝ := - (2 * x) / (1 + x^2)

theorem monotonicity_of_f :
  (∀ x y : ℝ, x < y ∧ (y < -1 ∨ x > 1) → f x < f y) ∧
  (∀ x y : ℝ, x < y ∧ -1 < x ∧ y < 1 → f y < f x) := sorry

end monotonicity_of_f_l286_286954


namespace tank_height_l286_286571

theorem tank_height
  (r_A r_B h_A h_B : ℝ)
  (h₁ : 8 = 2 * Real.pi * r_A)
  (h₂ : h_B = 8)
  (h₃ : 10 = 2 * Real.pi * r_B)
  (h₄ : π * r_A ^ 2 * h_A = 0.56 * (π * r_B ^ 2 * h_B)) :
  h_A = 7 :=
sorry

end tank_height_l286_286571


namespace original_fraction_is_two_thirds_l286_286583

theorem original_fraction_is_two_thirds (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) :
  (a^3 : ℚ)/(b + 3) = 2 * (a : ℚ)/b → (a : ℚ)/b = 2/3 :=
by
  sorry

end original_fraction_is_two_thirds_l286_286583


namespace rational_solutions_exist_l286_286867

theorem rational_solutions_exist (x p q : ℚ) (h : p^2 - x * q^2 = 1) :
  ∃ (a b : ℤ), p = (a^2 + x * b^2) / (a^2 - x * b^2) ∧ q = (2 * a * b) / (a^2 - x * b^2) :=
by
  sorry

end rational_solutions_exist_l286_286867


namespace possible_remainder_degrees_l286_286426

open Polynomial

noncomputable def divisor := (C 2) * (X ^ 6) - (C 1) * (X ^ 4) + (C 3) * (X ^ 2) - (C 5)

theorem possible_remainder_degrees (p r : Polynomial ℤ) (h : p = q * divisor + r) :
  degree r < degree divisor :=
sorry

end possible_remainder_degrees_l286_286426


namespace circles_are_disjoint_l286_286505

noncomputable def positional_relationship_of_circles (R₁ R₂ d : ℝ) (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : Prop :=
R₁ + R₂ = d

theorem circles_are_disjoint {R₁ R₂ d : ℝ} (h₁ : R₁ ≠ R₂)
  (h₂ : (2 * R₁)^2 - 4 * (R₂^2 - d * (R₂ - R₁)) = 0) : positional_relationship_of_circles R₁ R₂ d h₁ h₂ :=
by sorry

end circles_are_disjoint_l286_286505


namespace mirror_area_l286_286643

/-- The outer dimensions of the frame are given as 100 cm by 140 cm,
and the frame width is 15 cm. We aim to prove that the area of the mirror
inside the frame is 7700 cm². -/
theorem mirror_area (W H F: ℕ) (hW : W = 100) (hH : H = 140) (hF : F = 15) :
  (W - 2 * F) * (H - 2 * F) = 7700 :=
by
  sorry

end mirror_area_l286_286643


namespace fare_for_90_miles_l286_286457

noncomputable def fare_cost (miles : ℕ) (base_fare cost_per_mile : ℝ) : ℝ :=
  base_fare + cost_per_mile * miles

theorem fare_for_90_miles (base_fare : ℝ) (cost_per_mile : ℝ)
  (h1 : base_fare = 30)
  (h2 : fare_cost 60 base_fare cost_per_mile = 150)
  (h3 : cost_per_mile = (150 - base_fare) / 60) :
  fare_cost 90 base_fare cost_per_mile = 210 :=
  sorry

end fare_for_90_miles_l286_286457


namespace distinct_students_count_l286_286285

theorem distinct_students_count
  (algebra_students : ℕ)
  (calculus_students : ℕ)
  (statistics_students : ℕ)
  (algebra_statistics_overlap : ℕ)
  (no_other_overlaps : algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32) :
  algebra_students = 13 → calculus_students = 10 → statistics_students = 12 → algebra_statistics_overlap = 3 → 
  algebra_students + calculus_students + statistics_students - algebra_statistics_overlap = 32 :=
by
  intros h1 h2 h3 h4
  sorry

end distinct_students_count_l286_286285


namespace parabola_line_intersect_at_one_point_l286_286878

theorem parabola_line_intersect_at_one_point (a : ℚ) :
  (∃ x : ℚ, ax^2 + 5 * x + 4 = 0) → a = 25 / 16 :=
by
  -- Conditions and computation here
  sorry

end parabola_line_intersect_at_one_point_l286_286878


namespace combined_experience_l286_286690

theorem combined_experience : 
  ∀ (James John Mike : ℕ), 
  (James = 20) → 
  (∀ (years_ago : ℕ), (years_ago = 8) → (John = 2 * (James - years_ago) + years_ago)) → 
  (∀ (started : ℕ), (John - started = 16) → (Mike = 16)) → 
  James + John + Mike = 68 :=
begin
  intros James John Mike HJames HJohn HMike,
  rw HJames,
  have HJohn8 : John = 32, {
    rw HJohn,
    intros years_ago Hyears_ago,
    rw Hyears_ago,
    norm_num,
  },
  rw HJohn8 at HMike,
  norm_num at HMike,
  rw HJohn8,
  rw HMike,
  norm_num,
end

end combined_experience_l286_286690


namespace combined_balance_l286_286309

theorem combined_balance (b : ℤ) (g1 g2 : ℤ) (h1 : b = 3456) (h2 : g1 = b / 4) (h3 : g2 = b / 4) : g1 + g2 = 1728 :=
by {
  sorry
}

end combined_balance_l286_286309


namespace reading_hours_l286_286553

theorem reading_hours (h : ℕ) (lizaRate suzieRate : ℕ) (lizaPages suziePages : ℕ) 
  (hliza : lizaRate = 20) (hsuzie : suzieRate = 15) 
  (hlizaPages : lizaPages = lizaRate * h) (hsuziePages : suziePages = suzieRate * h) 
  (h_diff : lizaPages = suziePages + 15) : h = 3 :=
by {
  sorry
}

end reading_hours_l286_286553


namespace exists_permutation_with_large_neighbor_difference_l286_286262

theorem exists_permutation_with_large_neighbor_difference :
  ∃ (σ : Fin 100 → Fin 100), 
    (∀ (i : Fin 99), (|σ i.succ - σ i| ≥ 50)) :=
sorry

end exists_permutation_with_large_neighbor_difference_l286_286262


namespace birth_age_of_mother_l286_286962

def harrys_age : ℕ := 50

def fathers_age (h : ℕ) : ℕ := h + 24

def mothers_age (f h : ℕ) : ℕ := f - h / 25

theorem birth_age_of_mother (h f m : ℕ) (H1 : h = harrys_age)
  (H2 : f = fathers_age h) (H3 : m = mothers_age f h) :
  m - h = 22 := sorry

end birth_age_of_mother_l286_286962


namespace gcd_40_56_l286_286250

theorem gcd_40_56 : Nat.gcd 40 56 = 8 := 
by 
  sorry

end gcd_40_56_l286_286250


namespace Grant_spending_is_200_l286_286809

def Juanita_daily_spending (day: String) : Float :=
  if day = "Sunday" then 2.0 else 0.5

def Juanita_weekly_spending : Float :=
  6 * Juanita_daily_spending "weekday" + Juanita_daily_spending "Sunday"

def Juanita_yearly_spending : Float :=
  52 * Juanita_weekly_spending

def Grant_yearly_spending := Juanita_yearly_spending - 60

theorem Grant_spending_is_200 : Grant_yearly_spending = 200 := by
  sorry

end Grant_spending_is_200_l286_286809


namespace proof_completion_l286_286776

namespace MathProof

def p : ℕ := 10 * 7

def r : ℕ := p - 3

def q : ℚ := (3 / 5) * r

theorem proof_completion : q = 40.2 := by
  sorry

end MathProof

end proof_completion_l286_286776


namespace treaty_signed_on_thursday_l286_286442

def initial_day : ℕ := 0  -- 0 representing Monday, assuming a week cycle from 0 (Monday) to 6 (Sunday)
def days_in_week : ℕ := 7

def treaty_day (n : ℕ) : ℕ :=
(n + initial_day) % days_in_week

theorem treaty_signed_on_thursday :
  treaty_day 1000 = 4 :=  -- 4 representing Thursday
by
  sorry

end treaty_signed_on_thursday_l286_286442


namespace race_distance_l286_286534

def race_distance_problem (V_A V_B T : ℝ) : Prop :=
  V_A * T = 218.75 ∧
  V_B * T = 193.75 ∧
  V_B * (T + 10) = 218.75 ∧
  T = 77.5

theorem race_distance (D : ℝ) (V_A V_B T : ℝ) 
  (h1 : V_A * T = D) 
  (h2 : V_B * T = D - 25) 
  (h3 : V_B * (T + 10) = D) 
  (h4 : V_A * T = 218.75) 
  (h5 : T = 77.5) 
  : D = 218.75 := 
by 
  sorry

end race_distance_l286_286534


namespace total_brownies_correct_l286_286839

def brownies_initial : Nat := 24
def father_ate : Nat := brownies_initial / 3
def remaining_after_father : Nat := brownies_initial - father_ate
def mooney_ate : Nat := remaining_after_father / 4
def remaining_after_mooney : Nat := remaining_after_father - mooney_ate
def benny_ate : Nat := (remaining_after_mooney * 2) / 5
def remaining_after_benny : Nat := remaining_after_mooney - benny_ate
def snoopy_ate : Nat := 3
def remaining_after_snoopy : Nat := remaining_after_benny - snoopy_ate
def new_batch : Nat := 24
def total_brownies : Nat := remaining_after_snoopy + new_batch

theorem total_brownies_correct : total_brownies = 29 :=
by
  sorry

end total_brownies_correct_l286_286839


namespace min_spiders_sufficient_spiders_l286_286751

def grid_size : ℕ := 2019

noncomputable def min_k_catch (k : ℕ) : Prop :=
∀ (fly spider1 spider2 : ℕ × ℕ) (fly_move spider1_move spider2_move: ℕ × ℕ → ℕ × ℕ), 
  (fly_move fly = fly ∨ fly_move fly = (fly.1 + 1, fly.2) ∨ fly_move fly = (fly.1 - 1, fly.2)
  ∨ fly_move fly = (fly.1, fly.2 + 1) ∨ fly_move fly = (fly.1, fly.2 - 1))
  ∧ (spider1_move spider1 = spider1 ∨ spider1_move spider1 = (spider1.1 + 1, spider1.2) ∨ spider1_move spider1 = (spider1.1 - 1, spider1.2)
  ∨ spider1_move spider1 = (spider1.1, spider1.2 + 1) ∨ spider1_move spider1 = (spider1.1, spider1.2 - 1))
  ∧ (spider2_move spider2 = spider2 ∨ spider2_move spider2 = (spider2.1 + 1, spider2.2) ∨ spider2_move spider2 = (spider2.1 - 1, spider2.2)
  ∨ spider2_move spider2 = (spider2.1, spider2.2 + 1) ∨ spider2_move spider2 = (spider2.1, spider2.2 - 1))
  → (spider1 = fly ∨ spider2 = fly)

theorem min_spiders (k : ℕ) : min_k_catch k → k ≥ 2 :=
sorry

theorem sufficient_spiders : min_k_catch 2 :=
sorry

end min_spiders_sufficient_spiders_l286_286751


namespace range_of_a_l286_286026

noncomputable def f (a x : ℝ) : ℝ := a ^ x

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x < 2) : 
  (a ∈ Set.Ioo (Real.sqrt 2 / 2) 1 ∨ a ∈ Set.Ioo 1 (Real.sqrt 2)) :=
by
  sorry

end range_of_a_l286_286026


namespace intersection_M_N_l286_286061

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℝ := {x : ℝ | 2 * x > 7}

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end intersection_M_N_l286_286061


namespace total_distance_covered_is_correct_fuel_cost_excess_is_correct_l286_286918

-- Define the ratios and other conditions for Car A
def carA_ratio_gal_per_mile : ℚ := 4 / 7
def carA_gallons_used : ℚ := 44
def carA_cost_per_gallon : ℚ := 3.50

-- Define the ratios and other conditions for Car B
def carB_ratio_gal_per_mile : ℚ := 3 / 5
def carB_gallons_used : ℚ := 27
def carB_cost_per_gallon : ℚ := 3.25

-- Define the budget
def budget : ℚ := 200

-- Combined total distance covered by both cars
theorem total_distance_covered_is_correct :
  (carA_gallons_used * (7 / 4) + carB_gallons_used * (5 / 3)) = 122 :=
by
  sorry

-- Total fuel cost and whether it stays within budget
theorem fuel_cost_excess_is_correct :
  ((carA_gallons_used * carA_cost_per_gallon) + (carB_gallons_used * carB_cost_per_gallon)) - budget = 41.75 :=
by
  sorry

end total_distance_covered_is_correct_fuel_cost_excess_is_correct_l286_286918


namespace find_q_l286_286329

theorem find_q (q : Nat) (h : 81 ^ 6 = 3 ^ q) : q = 24 :=
by
  sorry

end find_q_l286_286329


namespace divisible_by_6_of_cubed_sum_div_by_18_l286_286218

theorem divisible_by_6_of_cubed_sum_div_by_18 (a b c : ℤ) 
  (h : a^3 + b^3 + c^3 ≡ 0 [ZMOD 18]) : (a * b * c) ≡ 0 [ZMOD 6] :=
sorry

end divisible_by_6_of_cubed_sum_div_by_18_l286_286218


namespace base_329_digits_even_l286_286482

noncomputable def base_of_four_digit_even_final : ℕ := 5

theorem base_329_digits_even (b : ℕ) (h1 : b^3 ≤ 329) (h2 : 329 < b^4)
  (h3 : ∀ d, 329 % b = d → d % 2 = 0) : b = base_of_four_digit_even_final :=
by sorry

end base_329_digits_even_l286_286482


namespace find_tax_rate_l286_286293

variable (total_spent : ℝ) (sales_tax : ℝ) (tax_free_cost : ℝ) (taxable_items_cost : ℝ) 
variable (T : ℝ)

theorem find_tax_rate (h1 : total_spent = 25) 
                      (h2 : sales_tax = 0.30)
                      (h3 : tax_free_cost = 21.7)
                      (h4 : taxable_items_cost = total_spent - tax_free_cost - sales_tax)
                      (h5 : sales_tax = (T / 100) * taxable_items_cost) :
  T = 10 := 
sorry

end find_tax_rate_l286_286293


namespace jasmine_added_is_8_l286_286771

noncomputable def jasmine_problem (J : ℝ) : Prop :=
  let initial_volume := 80
  let initial_jasmine_concentration := 0.10
  let initial_jasmine_amount := initial_volume * initial_jasmine_concentration

  let added_water := 12
  let final_volume := initial_volume + J + added_water
  let final_jasmine_concentration := 0.16
  let final_jasmine_amount := final_volume * final_jasmine_concentration

  initial_jasmine_amount + J = final_jasmine_amount 

theorem jasmine_added_is_8 : jasmine_problem 8 :=
by
  sorry

end jasmine_added_is_8_l286_286771


namespace initial_percentage_of_water_is_20_l286_286122

theorem initial_percentage_of_water_is_20 : 
  ∀ (P : ℝ) (total_initial_volume added_water total_final_volume final_percentage initial_water_percentage : ℝ), 
    total_initial_volume = 125 ∧ 
    added_water = 8.333333333333334 ∧ 
    total_final_volume = total_initial_volume + added_water ∧ 
    final_percentage = 25 ∧ 
    initial_water_percentage = (initial_water_percentage / total_initial_volume) * 100 ∧ 
    (final_percentage / 100) * total_final_volume = added_water + (initial_water_percentage / 100) * total_initial_volume → 
    initial_water_percentage = 20 := 
by 
  sorry

end initial_percentage_of_water_is_20_l286_286122


namespace solve_system_of_inequalities_l286_286392

theorem solve_system_of_inequalities (x : ℝ) : 
  (3 * x > x - 4) ∧ ((4 + x) / 3 > x + 2) → -2 < x ∧ x < -1 :=
by {
  sorry
}

end solve_system_of_inequalities_l286_286392


namespace merchant_marked_price_l286_286903

-- Definitions
def list_price : ℝ := 100
def purchase_price (L : ℝ) : ℝ := 0.8 * L
def selling_price_with_discount (x : ℝ) : ℝ := 0.75 * x
def profit (purchase_price : ℝ) (selling_price : ℝ) : ℝ := selling_price - purchase_price
def desired_profit (selling_price : ℝ) : ℝ := 0.3 * selling_price

-- Statement to prove
theorem merchant_marked_price :
  ∃ (x : ℝ), 
    profit (purchase_price list_price) (selling_price_with_discount x) = desired_profit (selling_price_with_discount x) ∧
    x / list_price = 152.38 / 100 :=
sorry

end merchant_marked_price_l286_286903


namespace calculate_profit_l286_286363

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end calculate_profit_l286_286363


namespace distribution_difference_l286_286610

theorem distribution_difference 
  (total_amnt : ℕ)
  (p_amnt : ℕ) 
  (q_amnt : ℕ) 
  (r_amnt : ℕ)
  (s_amnt : ℕ)
  (h_total : total_amnt = 1000)
  (h_p : p_amnt = 2 * q_amnt)
  (h_s : s_amnt = 4 * r_amnt)
  (h_qr : q_amnt = r_amnt) :
  s_amnt - p_amnt = 250 := 
sorry

end distribution_difference_l286_286610


namespace water_added_l286_286118

theorem water_added (W : ℝ) : 
  (15 + W) * 0.20833333333333336 = 3.75 → W = 3 :=
by
  intro h
  sorry

end water_added_l286_286118


namespace sales_tax_calculation_l286_286441

theorem sales_tax_calculation 
  (total_amount_paid : ℝ)
  (tax_rate : ℝ)
  (cost_tax_free : ℝ) :
  total_amount_paid = 30 → tax_rate = 0.08 → cost_tax_free = 12.72 → 
  (∃ sales_tax : ℝ, sales_tax = 1.28) :=
by
  intros H1 H2 H3
  sorry

end sales_tax_calculation_l286_286441


namespace Eunji_total_wrong_questions_l286_286783

theorem Eunji_total_wrong_questions 
  (solved_A : ℕ) (solved_B : ℕ) (wrong_A : ℕ) (right_diff : ℕ) 
  (h1 : solved_A = 12) 
  (h2 : solved_B = 15) 
  (h3 : wrong_A = 4) 
  (h4 : right_diff = 2) :
  (solved_A - (solved_A - (solved_A - wrong_A) + right_diff) + (solved_A - wrong_A) + right_diff - solved_B - (solved_B - (solved_A - (solved_A - wrong_A) + right_diff))) = 9 :=
by {
  sorry
}

end Eunji_total_wrong_questions_l286_286783


namespace value_of_x_l286_286092

theorem value_of_x (x y z : ℕ) (h1 : x = y / 3) (h2 : y = z / 4) (h3 : z = 48) : x = 4 := 
by
  sorry

end value_of_x_l286_286092


namespace f_analytical_expression_g_value_l286_286956

noncomputable def f (ω x : ℝ) : ℝ := (1/2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x + Real.pi / 2)

noncomputable def g (ω x : ℝ) : ℝ := f ω (x + Real.pi / 4)

theorem f_analytical_expression (x : ℝ) (hω : ω = 2 ∧ ω > 0) : 
  f 2 x = Real.sin (2 * x - Real.pi / 3) :=
sorry

theorem g_value (α : ℝ) (hω : ω = 2 ∧ ω > 0) (h : g 2 (α / 2) = 4/5) : 
  g 2 (-α) = -7/25 :=
sorry

end f_analytical_expression_g_value_l286_286956


namespace hyperbola_eccentricity_l286_286470

theorem hyperbola_eccentricity (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_hyperbola: ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_asymptotes_l1: ∀ x : ℝ, y = (b / a) * x)
  (h_asymptotes_l2: ∀ x : ℝ, y = -(b / a) * x)
  (h_focus: c^2 = a^2 + b^2)
  (h_symmetric: ∀ m : ℝ, m = -c / 2 ∧ (m, (b * c) / (2 * a)) ∈ { p : ℝ × ℝ | p.2 = -(b / a) * p.1 }) :
  (c / a) = 2 := sorry

end hyperbola_eccentricity_l286_286470


namespace least_integer_a_divisible_by_240_l286_286342

theorem least_integer_a_divisible_by_240 (a : ℤ) (h1 : 240 ∣ a^3) : a ≥ 60 := by
  sorry

end least_integer_a_divisible_by_240_l286_286342


namespace monochromatic_triangle_probability_correct_l286_286640

noncomputable def monochromatic_triangle_probability (p : ℝ) : ℝ :=
  1 - (3 * (p^2) * (1 - p) + 3 * ((1 - p)^2) * p)^20

theorem monochromatic_triangle_probability_correct :
  monochromatic_triangle_probability (1/2) = 1 - (3/4)^20 :=
by
  sorry

end monochromatic_triangle_probability_correct_l286_286640


namespace eugene_used_six_boxes_of_toothpicks_l286_286938

-- Define the given conditions
def toothpicks_per_card : ℕ := 75
def total_cards : ℕ := 52
def unused_cards : ℕ := 16
def toothpicks_per_box : ℕ := 450

-- Compute the required result
theorem eugene_used_six_boxes_of_toothpicks :
  ((total_cards - unused_cards) * toothpicks_per_card) / toothpicks_per_box = 6 :=
by
  sorry

end eugene_used_six_boxes_of_toothpicks_l286_286938


namespace greatest_x_is_53_l286_286875

-- Define the polynomial expression
def polynomial (x : ℤ) : ℤ := x^2 + 2 * x + 13

-- Define the condition for the expression to be an integer
def isIntegerWhenDivided (x : ℤ) : Prop := (polynomial x) % (x - 5) = 0

-- Define the theorem to prove the greatest integer value of x
theorem greatest_x_is_53 : ∃ x : ℤ, isIntegerWhenDivided x ∧ (∀ y : ℤ, isIntegerWhenDivided y → y ≤ x) ∧ x = 53 :=
by
  sorry

end greatest_x_is_53_l286_286875


namespace problem_statement_l286_286510

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l286_286510


namespace paige_team_total_players_l286_286375

theorem paige_team_total_players 
    (total_points : ℕ)
    (paige_points : ℕ)
    (other_points_per_player : ℕ)
    (other_players : ℕ) :
    total_points = paige_points + other_points_per_player * other_players →
    (other_players + 1) = 6 :=
by
  intros h
  sorry

end paige_team_total_players_l286_286375


namespace find_positive_n_l286_286941

def consecutive_product (k : ℕ) : ℕ := k * (k + 1) * (k + 2)

theorem find_positive_n (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  n^6 + 5*n^3 + 4*n + 116 = consecutive_product k ↔ n = 3 := 
by 
  sorry

end find_positive_n_l286_286941


namespace percentage_difference_l286_286900

-- Define the numbers
def n : ℕ := 1600
def m : ℕ := 650

-- Define the percentages calculated
def p₁ : ℕ := (20 * n) / 100
def p₂ : ℕ := (20 * m) / 100

-- The theorem to be proved: the difference between the two percentages is 190
theorem percentage_difference : p₁ - p₂ = 190 := by
  sorry

end percentage_difference_l286_286900


namespace lipstick_cost_correct_l286_286774

noncomputable def cost_of_lipsticks (total_cost: ℕ) (cost_slippers: ℚ) (cost_hair_color: ℚ) (paid: ℚ) (number_lipsticks: ℕ) : ℚ :=
  (paid - (6 * cost_slippers + 8 * cost_hair_color)) / number_lipsticks

theorem lipstick_cost_correct :
  cost_of_lipsticks 6 (2.5:ℚ) (3:ℚ) (44:ℚ) 4 = 1.25 := by
  sorry

end lipstick_cost_correct_l286_286774


namespace prime_factors_of_n_l286_286281

def n : ℕ := 400000001

def is_prime (p: ℕ) : Prop := Nat.Prime p

theorem prime_factors_of_n (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : n = p * q) : 
  (p = 19801 ∧ q = 20201) ∨ (p = 20201 ∧ q = 19801) :=
by
  sorry

end prime_factors_of_n_l286_286281


namespace james_profit_l286_286358

-- Definitions and Conditions
def head_of_cattle : ℕ := 100
def purchase_price : ℕ := 40000
def feeding_percentage : ℕ := 20
def weight_per_head : ℕ := 1000
def price_per_pound : ℕ := 2

def feeding_cost : ℕ := (purchase_price * feeding_percentage) / 100
def total_cost : ℕ := purchase_price + feeding_cost
def selling_price_per_head : ℕ := weight_per_head * price_per_pound
def total_selling_price : ℕ := head_of_cattle * selling_price_per_head
def profit : ℕ := total_selling_price - total_cost

-- Theorem to Prove
theorem james_profit : profit = 112000 := by
  sorry

end james_profit_l286_286358


namespace functional_equation_solution_l286_286638

theorem functional_equation_solution :
  ∀ (f : ℤ → ℤ), (∀ (m n : ℤ), f (m + f (f n)) = -f (f (m + 1)) - n) → (∀ (p : ℤ), f p = 1 - p) :=
by
  intro f h
  sorry

end functional_equation_solution_l286_286638


namespace complex_multiplication_l286_286939

-- Define the imaginary unit i
def i := Complex.I

-- Define the theorem we need to prove
theorem complex_multiplication : 
  (3 - 7 * i) * (-6 + 2 * i) = -4 + 48 * i := 
by 
  -- Proof is omitted
  sorry

end complex_multiplication_l286_286939


namespace f_of_13_eq_223_l286_286814

def f (n : ℕ) : ℕ := n^2 + n + 41

theorem f_of_13_eq_223 : f 13 = 223 := 
by sorry

end f_of_13_eq_223_l286_286814


namespace initial_percentage_of_water_is_20_l286_286123

theorem initial_percentage_of_water_is_20 : 
  ∀ (P : ℝ) (total_initial_volume added_water total_final_volume final_percentage initial_water_percentage : ℝ), 
    total_initial_volume = 125 ∧ 
    added_water = 8.333333333333334 ∧ 
    total_final_volume = total_initial_volume + added_water ∧ 
    final_percentage = 25 ∧ 
    initial_water_percentage = (initial_water_percentage / total_initial_volume) * 100 ∧ 
    (final_percentage / 100) * total_final_volume = added_water + (initial_water_percentage / 100) * total_initial_volume → 
    initial_water_percentage = 20 := 
by 
  sorry

end initial_percentage_of_water_is_20_l286_286123


namespace find_other_endpoint_l286_286706

theorem find_other_endpoint (x₁ y₁ x y x_mid y_mid : ℝ) 
  (h1 : x₁ = 5) (h2 : y₁ = 2) (h3 : x_mid = 3) (h4 : y_mid = 10) 
  (hx : (x₁ + x) / 2 = x_mid) (hy : (y₁ + y) / 2 = y_mid) : 
  x = 1 ∧ y = 18 := by
  sorry

end find_other_endpoint_l286_286706


namespace combine_like_terms_substitute_expression_complex_expression_l286_286894

-- Part 1
theorem combine_like_terms (a b : ℝ) : 
  10 * (a - b)^2 - 12 * (a - b)^2 + 9 * (a - b)^2 = 7 * (a - b)^2 :=
by
  sorry

-- Part 2
theorem substitute_expression (x y : ℝ) (h1 : x^2 - 2 * y = -5) : 
  4 * x^2 - 8 * y + 24 = 4 :=
by
  sorry

-- Part 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 2 * b = 1009.5) 
  (h2 : 2 * b - c = -2024.6666)
  (h3 : c - d = 1013.1666) : 
  (a - c) + (2 * b - d) - (2 * b - c) = -2 :=
by
  sorry

end combine_like_terms_substitute_expression_complex_expression_l286_286894


namespace line_through_intersections_l286_286959

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 9

theorem line_through_intersections (x y : ℝ) :
  circle1 x y → circle2 x y → 2 * x - 3 * y = 0 := 
sorry

end line_through_intersections_l286_286959


namespace minimal_perimeter_triangle_l286_286826

noncomputable def cos_P : ℚ := 3 / 5
noncomputable def cos_Q : ℚ := 24 / 25
noncomputable def cos_R : ℚ := -1 / 5

theorem minimal_perimeter_triangle
  (P Q R : ℝ) (a b c : ℕ)
  (h0 : a^2 + b^2 + c^2 - 2 * a * b * cos_P - 2 * b * c * cos_Q - 2 * c * a * cos_R = 0)
  (h1 : cos_P^2 + (1 - cos_P^2) = 1)
  (h2 : cos_Q^2 + (1 - cos_Q^2) = 1)
  (h3 : cos_R^2 + (1 - cos_R^2) = 1) :
  a + b + c = 47 :=
sorry

end minimal_perimeter_triangle_l286_286826


namespace hyperbola_eccentricity_l286_286317

theorem hyperbola_eccentricity (a b c : ℝ) (h_asymptotes : b / a = 3 / 4 ∨ a / b = 3 / 4) :
  (c / a = 5 / 4) ∨ (c / a = 5 / 3) :=
by
  -- Proof omitted
  sorry

end hyperbola_eccentricity_l286_286317


namespace corvette_trip_time_percentage_increase_l286_286554

theorem corvette_trip_time_percentage_increase
  (total_distance : ℝ)
  (first_half_speed : ℝ)
  (average_speed : ℝ)
  (first_half_distance second_half_distance first_half_time second_half_time total_time : ℝ)
  (h1 : total_distance = 640)
  (h2 : first_half_speed = 80)
  (h3 : average_speed = 40)
  (h4 : first_half_distance = total_distance / 2)
  (h5 : second_half_distance = total_distance / 2)
  (h6 : first_half_time = first_half_distance / first_half_speed)
  (h7 : total_time = total_distance / average_speed)
  (h8 : second_half_time = total_time - first_half_time) :
  ((second_half_time - first_half_time) / first_half_time) * 100 = 200 := sorry

end corvette_trip_time_percentage_increase_l286_286554


namespace triangle_sides_inequality_l286_286835

-- Define the sides of a triangle and their sum
variables {a b c : ℝ}

-- Define the condition that they are sides of a triangle.
def triangle_sides (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition that their sum is 1
axiom sum_of_sides (a b c : ℝ) (h : triangle_sides a b c) : a + b + c = 1

-- Define the proof theorem for the inequality
theorem triangle_sides_inequality (h : triangle_sides a b c) (h_sum : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end triangle_sides_inequality_l286_286835


namespace vector_identity_l286_286485

def vec_a : ℝ × ℝ := (2, 2)
def vec_b : ℝ × ℝ := (-1, 3)

theorem vector_identity : 2 • vec_a - vec_b = (5, 1) := by
  sorry

end vector_identity_l286_286485


namespace taxi_fare_total_distance_l286_286526

theorem taxi_fare_total_distance (initial_fare additional_fare : ℝ) (total_fare : ℝ) (initial_distance additional_distance : ℝ) :
  initial_fare = 10 ∧ additional_fare = 1 ∧ initial_distance = 1/5 ∧ (total_fare = 59) →
  (total_distance = initial_distance + additional_distance * ((total_fare - initial_fare) / additional_fare)) →
  total_distance = 10 := 
by 
  sorry

end taxi_fare_total_distance_l286_286526


namespace common_chord_eqn_circle_with_center_on_line_smallest_area_circle_l286_286031

noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_eqn :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - 2*y + 4 = 0) :=
sorry

noncomputable def A : ℝ × ℝ := (-4, 0)
noncomputable def B : ℝ × ℝ := (0, 2)
noncomputable def line_y_eq_neg_x (x y : ℝ) : Prop := y = -x

theorem circle_with_center_on_line :
  ∃ (x y : ℝ), line_y_eq_neg_x x y ∧ ((x + 3)^2 + (y - 3)^2 = 10) :=
sorry

theorem smallest_area_circle :
  ∃ (x y : ℝ), ((x + 2)^2 + (y - 1)^2 = 5) :=
sorry

end common_chord_eqn_circle_with_center_on_line_smallest_area_circle_l286_286031


namespace probability_inequality_l286_286885

theorem probability_inequality :
  let S := {x : ℕ | x > 0 ∧ x < 10}
  let favorable := {x ∈ S | 8 / x > x}
  (favorable.card : ℚ) / (S.card : ℚ) = 1 / 3 :=
by
  let S := {x : ℕ | x > 0 ∧ x < 10}
  let favorable := {x ∈ S | 8 / x > x}
  have S_card : S.card = 9 := by sorry
  have favorable_card : favorable.card = 3 := by sorry
  calc
    (favorable.card : ℚ) / (S.card : ℚ) = (3 : ℚ) / (9 : ℚ) : by rw [favorable_card, S_card]
    ... = 1 / 3 : by norm_num

end probability_inequality_l286_286885


namespace combined_height_l286_286984

theorem combined_height (h_John : ℕ) (h_Lena : ℕ) (h_Rebeca : ℕ)
  (cond1 : h_John = 152)
  (cond2 : h_John = h_Lena + 15)
  (cond3 : h_Rebeca = h_John + 6) :
  h_Lena + h_Rebeca = 295 :=
by
  sorry

end combined_height_l286_286984


namespace bottles_per_case_l286_286328

theorem bottles_per_case (days: ℕ) (daily_intake: ℚ) (total_spent: ℚ) (case_cost: ℚ) (total_cases: ℕ) (total_bottles: ℕ) (B: ℕ) 
    (H1 : days = 240)
    (H2 : daily_intake = 1/2)
    (H3 : total_spent = 60)
    (H4 : case_cost = 12)
    (H5 : total_cases = total_spent / case_cost)
    (H6 : total_bottles = days * daily_intake)
    (H7 : B = total_bottles / total_cases) :
    B = 24 :=
by
    sorry

end bottles_per_case_l286_286328


namespace divisibility_323_l286_286794

theorem divisibility_323 (n : ℕ) : 
  (20^n + 16^n - 3^n - 1) % 323 = 0 ↔ Even n := 
sorry

end divisibility_323_l286_286794


namespace consecutive_integers_solution_l286_286090

theorem consecutive_integers_solution :
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) + 91 = n^2 + (n + 1)^2 ∧ n + 1 = 10 :=
by
  sorry

end consecutive_integers_solution_l286_286090


namespace eval_floor_expr_l286_286785

def frac_part1 : ℚ := (15 / 8)
def frac_part2 : ℚ := (11 / 3)
def square_frac1 : ℚ := frac_part1 ^ 2
def ceil_part : ℤ := ⌈square_frac1⌉
def add_frac2 : ℚ := ceil_part + frac_part2

theorem eval_floor_expr : (⌊add_frac2⌋ : ℤ) = 7 := 
sorry

end eval_floor_expr_l286_286785


namespace pyramid_circumscribed_sphere_volume_l286_286575

theorem pyramid_circumscribed_sphere_volume 
  (PA ABCD : ℝ) 
  (square_base : Prop)
  (perpendicular_PA_base : Prop)
  (AB : ℝ)
  (PA_val : PA = 1)
  (AB_val : AB = 2) 
  : (∃ (volume : ℝ), volume = (4/3) * π * (3/2)^3 ∧ volume = 9 * π / 2) := 
by
  -- Provided the conditions, we need to prove that the volume of the circumscribed sphere is 9π/2
  sorry

end pyramid_circumscribed_sphere_volume_l286_286575


namespace find_side_b_l286_286211

theorem find_side_b (a b c A B C : ℝ) (hB : B = 60) 
  (h_area : (1/2)*a*c*(Real.sin (Real.pi/3)) = Real.sqrt 3) 
  (h_ac : a^2 + c^2 = 3*a*c) : b = 2 * Real.sqrt 2 :=
by
  -- Given conditions
  have hB' : Real.sin (Real.pi/3) = Real.sqrt 3 / 2 := by sorry
  have h_ac_val : a * c = 4 := by sorry
  have h_a2c2 : a^2 + c^2 = 12 := by sorry
  have h_cosB : (a^2 + c^2 - b^2) / (2 * a * c) = 1/2 := by sorry
  have h_b2 : b^2 = 8 := by sorry

  -- Result
  exact Real.sqrt 8

end find_side_b_l286_286211


namespace inequality_mn_l286_286949

theorem inequality_mn (m n : ℤ)
  (h : ∃ x : ℤ, (x + m) * (x + n) = x + m + n) : 
  2 * (m^2 + n^2) < 5 * m * n := 
sorry

end inequality_mn_l286_286949


namespace nth_odd_positive_integer_is_199_l286_286417

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l286_286417


namespace find_number_l286_286888

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 99) : x = 4400 :=
sorry

end find_number_l286_286888


namespace monogram_count_l286_286840

theorem monogram_count :
  ∃ (n : ℕ), n = 156 ∧
    (∃ (beforeM : Fin 13) (afterM : Fin 14),
      ∀ (a : Fin 13) (b : Fin 14),
        a < b → (beforeM = a ∧ afterM = b) → n = 12 * 13
    ) :=
by {
  sorry
}

end monogram_count_l286_286840


namespace inequality_proof_l286_286379

variable (a b c d : ℝ)

theorem inequality_proof (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (1 / (1 / a + 1 / b)) + (1 / (1 / c + 1 / d)) ≤ (1 / (1 / (a + c) + 1 / (b + d))) :=
by
  sorry

end inequality_proof_l286_286379


namespace value_of_x_squared_plus_reciprocal_squared_l286_286514

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 47 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = 7 :=
by sorry

end value_of_x_squared_plus_reciprocal_squared_l286_286514


namespace smallest_integer_inequality_l286_286304

theorem smallest_integer_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ 
           (∀ m : ℤ, m < n → ¬∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) :=
by
  sorry

end smallest_integer_inequality_l286_286304


namespace smaller_bills_denomination_correct_l286_286871

noncomputable def denomination_of_smaller_bills : ℕ :=
  let total_money := 1000
  let part_smaller_bills := 3 / 10
  let smaller_bills_amount := part_smaller_bills * total_money
  let rest_of_money := total_money - smaller_bills_amount
  let bill_100_denomination := 100
  let total_bills := 13
  let num_100_bills := rest_of_money / bill_100_denomination
  let num_smaller_bills := total_bills - num_100_bills
  let denomination := smaller_bills_amount / num_smaller_bills
  denomination

theorem smaller_bills_denomination_correct : denomination_of_smaller_bills = 50 := by
  sorry

end smaller_bills_denomination_correct_l286_286871


namespace investment_doubles_in_9_years_l286_286525

noncomputable def years_to_double (initial_amount : ℕ) (interest_rate : ℕ) : ℕ :=
  72 / interest_rate

theorem investment_doubles_in_9_years :
  ∀ (initial_amount : ℕ) (interest_rate : ℕ) (investment_period_val : ℕ) (expected_value : ℕ),
  initial_amount = 8000 ∧ interest_rate = 8 ∧ investment_period_val = 18 ∧ expected_value = 32000 →
  years_to_double initial_amount interest_rate = 9 :=
by
  intros initial_amount interest_rate investment_period_val expected_value h
  sorry

end investment_doubles_in_9_years_l286_286525


namespace sequence_first_equals_last_four_l286_286216

theorem sequence_first_equals_last_four (n : ℕ) (S : ℕ → ℕ) (h_length : ∀ i < n, S i = 0 ∨ S i = 1)
  (h_condition : ∀ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n - 4 → 
    (S i = S j ∧ S (i + 1) = S (j + 1) ∧ S (i + 2) = S (j + 2) ∧ S (i + 3) = S (j + 3) ∧ S (i + 4) = S (j + 4)) → false) :
  S 1 = S (n - 3) ∧ S 2 = S (n - 2) ∧ S 3 = S (n - 1) ∧ S 4 = S n :=
sorry

end sequence_first_equals_last_four_l286_286216


namespace simplify_and_evaluate_expression_l286_286843

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1 / 2) (h2 : y = 2023) :
  (x + y)^2 + (x + y) * (x - y) - 2 * x^2 = 2023 :=
by
  sorry

end simplify_and_evaluate_expression_l286_286843


namespace smallest_lcm_l286_286332

/-- If k and l are positive 4-digit integers such that gcd(k, l) = 5, 
the smallest value for lcm(k, l) is 201000. -/
theorem smallest_lcm (k l : ℕ) (hk : 1000 ≤ k ∧ k < 10000) (hl : 1000 ≤ l ∧ l < 10000) (h₅ : Nat.gcd k l = 5) :
  Nat.lcm k l = 201000 :=
sorry

end smallest_lcm_l286_286332


namespace gcd_polynomial_l286_286802

theorem gcd_polynomial (b : ℤ) (h : b % 2 = 0 ∧ 1171 ∣ b) : 
  Int.gcd (3 * b^2 + 17 * b + 47) (b + 5) = 1 :=
sorry

end gcd_polynomial_l286_286802


namespace max_value_x_y_squared_l286_286371

theorem max_value_x_y_squared (x y : ℝ) (h : 3 * (x^3 + y^3) = x + y^2) : x + y^2 ≤ 1/3 :=
sorry

end max_value_x_y_squared_l286_286371


namespace common_difference_arithmetic_progression_l286_286046

theorem common_difference_arithmetic_progression {n : ℕ} (x y : ℝ) (a : ℕ → ℝ) 
  (h : ∀ k : ℕ, k ≤ n → a (k+1) = a k + (y - x) / (n + 1)) 
  : (∃ d : ℝ, ∀ i : ℕ, i ≤ n + 1 → a (i+1) = x + i * d) ∧ d = (y - x) / (n + 1) := 
by
  sorry

end common_difference_arithmetic_progression_l286_286046


namespace tan_three_halves_pi_sub_alpha_l286_286660

theorem tan_three_halves_pi_sub_alpha (α : ℝ) (h : Real.cos (π - α) = -3/5) :
    Real.tan (3 * π / 2 - α) = 3/4 ∨ Real.tan (3 * π / 2 - α) = -3/4 := by
  sorry

end tan_three_halves_pi_sub_alpha_l286_286660


namespace direct_proportion_b_zero_l286_286969

theorem direct_proportion_b_zero (b : ℝ) (x y : ℝ) 
  (h : ∀ x, y = x + b → ∃ k, y = k * x) : b = 0 :=
sorry

end direct_proportion_b_zero_l286_286969


namespace hundredth_odd_integer_l286_286420

theorem hundredth_odd_integer : (2 * 100 - 1) = 199 := 
by
  sorry

end hundredth_odd_integer_l286_286420


namespace ellipse_equation_max_area_abcd_l286_286170

open Real

theorem ellipse_equation (x y : ℝ) (a b c : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (x^2 / 2 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry

theorem max_area_abcd (a b c t : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (∀ (t : ℝ), 4 * sqrt 2 * sqrt (1 + t^2) / (t^2 + 2) ≤ 2 * sqrt 2) := by
  sorry

end ellipse_equation_max_area_abcd_l286_286170


namespace road_building_equation_l286_286593

theorem road_building_equation (x : ℝ) (hx : x > 0) :
  (9 / x - 12 / (x + 1) = 1 / 2) :=
sorry

end road_building_equation_l286_286593


namespace new_average_l286_286803

open Nat

-- The Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

-- Sum of the first 35 Fibonacci numbers
def sum_fibonacci_first_35 : ℕ :=
  (List.range 35).map fibonacci |>.sum -- or critical to use: List.foldr (λ x acc, fibonacci x + acc) 0 (List.range 35) 

theorem new_average (n : ℕ) (avg : ℕ) (Fib_Sum : ℕ) 
  (h₁ : n = 35) 
  (h₂ : avg = 25) 
  (h₃ : Fib_Sum = sum_fibonacci_first_35) : 
  (25 * Fib_Sum / 35) = avg * (sum_fibonacci_first_35) / n := 
by 
  sorry

end new_average_l286_286803


namespace no_values_less_than_180_l286_286255

/-- Given that w and n are positive integers less than 180 
    such that w % 13 = 2 and n % 8 = 5, 
    prove that there are no such values for w and n. -/
theorem no_values_less_than_180 (w n : ℕ) (hw : w < 180) (hn : n < 180) 
  (h1 : w % 13 = 2) (h2 : n % 8 = 5) : false :=
by
  sorry

end no_values_less_than_180_l286_286255


namespace maximum_area_of_triangle_ABC_l286_286175

noncomputable def max_area_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem maximum_area_of_triangle_ABC (a b c A B C : ℝ) 
  (h1: a = 4) 
  (h2: (4 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  max_area_triangle_ABC a b c A B C = 4 * Real.sqrt 3 := 
sorry

end maximum_area_of_triangle_ABC_l286_286175


namespace rearrange_2023_prob_l286_286816

theorem rearrange_2023_prob :
  (let total_arrangements := 9 in
   let favorable_arrangements := 5 in
   favorable_arrangements / total_arrangements = 5 / 9) :=
begin
  sorry
end

end rearrange_2023_prob_l286_286816


namespace general_term_of_c_l286_286325

theorem general_term_of_c (a b : ℕ → ℕ) (c : ℕ → ℕ) : 
  (∀ n, a n = 2 ^ n) →
  (∀ n, b n = 3 * n + 2) →
  (∀ n, ∃ m k, a n = b m ∧ n = 2 * k + 1 → c k = a n) →
  ∀ n, c n = 2 ^ (2 * n + 1) :=
by
  intros ha hb hc n
  have h' := hc n
  sorry

end general_term_of_c_l286_286325


namespace students_more_than_pets_l286_286150

theorem students_more_than_pets :
  let students_per_classroom := 15
  let rabbits_per_classroom := 1
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 6
  let total_students := students_per_classroom * number_of_classrooms
  let total_pets := (rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms
  total_students - total_pets = 66 :=
by
  sorry

end students_more_than_pets_l286_286150


namespace min_value_inequality_l286_286369

open Real

theorem min_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) : 3 * a + 2 * b + c ≥ 18 := 
sorry

end min_value_inequality_l286_286369


namespace angles_sum_correct_l286_286632

-- Definitions from the problem conditions
def identicalSquares (n : Nat) := n = 13

variable (α β γ δ ε ζ η θ : ℝ) -- Angles of interest

def anglesSum :=
  (α + β + γ + δ) + (ε + ζ + η + θ)

-- Lean 4 statement
theorem angles_sum_correct
  (h₁ : identicalSquares 13)
  (h₂ : α = 90) (h₃ : β = 90) (h₄ : γ = 90) (h₅ : δ = 90)
  (h₆ : ε = 90) (h₇ : ζ = 90) (h₈ : η = 45) (h₉ : θ = 45) :
  anglesSum α β γ δ ε ζ η θ = 405 :=
by
  simp [anglesSum]
  sorry

end angles_sum_correct_l286_286632


namespace product_of_four_consecutive_integers_is_not_square_l286_286079

theorem product_of_four_consecutive_integers_is_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = (n-1)*n*(n+1)*(n+2) :=
sorry

end product_of_four_consecutive_integers_is_not_square_l286_286079


namespace max_int_difference_l286_286889

theorem max_int_difference (x y : ℤ) (hx : 5 < x ∧ x < 8) (hy : 8 < y ∧ y < 13) : 
  y - x = 5 :=
sorry

end max_int_difference_l286_286889


namespace area_of_region_R_l286_286245

open Real

noncomputable def area_of_strip (width : ℝ) (height : ℝ) : ℝ :=
  width * height

noncomputable def area_of_triangle (leg : ℝ) : ℝ :=
  1 / 2 * leg * leg

theorem area_of_region_R :
  let unit_square_area := 1
  let AE_BE := 1 / sqrt 2
  let area_triangle_ABE := area_of_triangle AE_BE
  let strip_width := 1 / 4
  let strip_height := 1
  let area_strip := area_of_strip strip_width strip_height
  let overlap_area := area_triangle_ABE / 2
  let area_R := area_strip - overlap_area
  area_R = 1 / 8 :=
by
  sorry

end area_of_region_R_l286_286245


namespace train_speed_l286_286870

theorem train_speed (L1 L2 : ℕ) (V2 : ℕ) (t : ℝ) (V1 : ℝ) : 
  L1 = 200 → 
  L2 = 280 → 
  V2 = 30 → 
  t = 23.998 → 
  (0.001 * (L1 + L2)) / (t / 3600) = V1 + V2 → 
  V1 = 42 :=
by 
  intros
  sorry

end train_speed_l286_286870


namespace convert_decimal_to_fraction_l286_286881

theorem convert_decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by
  sorry

end convert_decimal_to_fraction_l286_286881


namespace point_in_second_quadrant_coordinates_l286_286019

theorem point_in_second_quadrant_coordinates (a : ℤ) (h1 : a + 1 < 0) (h2 : 2 * a + 6 > 0) :
  (a + 1, 2 * a + 6) = (-1, 2) :=
sorry

end point_in_second_quadrant_coordinates_l286_286019


namespace glucose_solution_volume_l286_286269

theorem glucose_solution_volume (V : ℕ) (h : 500 / 10 = V / 20) : V = 1000 :=
sorry

end glucose_solution_volume_l286_286269


namespace expected_value_red_balls_l286_286308

open ProbabilityTheory

-- Definitions of the conditions
def bag : set (set ℕ) := {s | s ⊆ {1, 2, 3, 4} ∧ s.card = 2}
def redBalls : set ℕ := {2, 3, 4}

-- Random variable X: number of red balls drawn
def X (s : set ℕ) : ℕ := (s ∩ redBalls).card

-- Probability measure for uniformly drawing 2 balls from 4
noncomputable def uniformMeasure : measure (set ℕ) := measure.count bag

-- Expected value of X under uniformMeasure
noncomputable def expectedValueX : ℝ := ∫ s in uniformMeasure, X s

-- Lean statement to prove the expected value equals 3/2
theorem expected_value_red_balls :
  expectedValueX = 3 / 2 :=
sorry

end expected_value_red_balls_l286_286308


namespace cans_collected_by_first_group_l286_286851

def class_total_students : ℕ := 30
def students_didnt_collect : ℕ := 2
def students_collected_4 : ℕ := 13
def total_cans_collected : ℕ := 232

theorem cans_collected_by_first_group :
  let remaining_students := class_total_students - (students_didnt_collect + students_collected_4)
  let cans_by_13_students := students_collected_4 * 4
  let cans_by_first_group := total_cans_collected - cans_by_13_students
  let cans_per_student := cans_by_first_group / remaining_students
  cans_per_student = 12 := by
  sorry

end cans_collected_by_first_group_l286_286851


namespace value_of_expression_l286_286831

variable {a b : ℝ}
variables (h1 : ∀ x, 3 * x^2 + 9 * x - 18 = 0 → x = a ∨ x = b)

theorem value_of_expression : (3 * a - 2) * (6 * b - 9) = 27 :=
by
  sorry

end value_of_expression_l286_286831


namespace abs_lt_five_implies_interval_l286_286227

theorem abs_lt_five_implies_interval (x : ℝ) : |x| < 5 → -5 < x ∧ x < 5 := by
  sorry

end abs_lt_five_implies_interval_l286_286227


namespace composite_expression_l286_286841

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (a * b = 6 * 2^(2^(4 * n)) + 1) :=
by
  sorry

end composite_expression_l286_286841


namespace pipe_c_empty_time_l286_286892

theorem pipe_c_empty_time :
  (1 / 45 + 1 / 60 - x = 1 / 40) → (1 / x = 72) :=
by
  sorry

end pipe_c_empty_time_l286_286892


namespace solve_fractional_equation_l286_286391

theorem solve_fractional_equation (x : ℝ) (h : x ≠ 1) : 
  (3 * x + 6) / (x^2 + 6 * x - 7) = (3 - x) / (x - 1) ↔ x = -5 ∨ x = 3 :=
sorry

end solve_fractional_equation_l286_286391


namespace derivative_at_one_l286_286499

noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

theorem derivative_at_one : deriv f 1 = -2 :=
by 
  -- Here we would provide the proof that f'(1) = -2
  sorry

end derivative_at_one_l286_286499


namespace skating_probability_given_skiing_l286_286533

theorem skating_probability_given_skiing (P_A P_B P_A_or_B : ℝ)
    (h1 : P_A = 0.6) (h2 : P_B = 0.5) (h3 : P_A_or_B = 0.7) : 
    (P_A_or_B = P_A + P_B - P_A * P_B) → 
    ((P_A * P_B) / P_B = 0.8) := 
    by
        intros
        sorry

end skating_probability_given_skiing_l286_286533


namespace game_c_higher_prob_than_game_d_l286_286439

noncomputable def prob_heads : ℚ := 2 / 3
noncomputable def prob_tails : ℚ := 1 / 3

def game_c_winning_prob : ℚ :=
  let prob_first_three := prob_heads ^ 3 + prob_tails ^ 3
  let prob_last_three := prob_heads ^ 3 + prob_tails ^ 3
  let prob_overlap := prob_heads ^ 5 + prob_tails ^ 5
  prob_first_three + prob_last_three - prob_overlap

def game_d_winning_prob : ℚ :=
  let prob_first_last_two := (prob_heads ^ 2 + prob_tails ^ 2) ^ 2
  let prob_middle_three := prob_heads ^ 3 + prob_tails ^ 3
  let prob_overlap_d := 2 * (prob_heads ^ 4 + prob_tails ^ 4)
  prob_first_last_two + prob_middle_three - prob_overlap_d

theorem game_c_higher_prob_than_game_d :
  game_c_winning_prob - game_d_winning_prob = 29 / 81 := 
sorry

end game_c_higher_prob_than_game_d_l286_286439


namespace ellipse_problem_l286_286799

noncomputable def ellipse_equation (a b c : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (a + c = 2 + real.sqrt 3) ∧ (a - c = 2 - real.sqrt 3) ∧ (b^2 = a^2 - c^2)

noncomputable def line_slopes_geometric (k b x1 x2 : ℝ) : Prop :=
  let y1 := k * x1 + b in
  let y2 := k * x2 + b in
  k^2 = (y1 * y2) / (x1 * x2)

noncomputable def delta_discriminant (b k : ℝ) : Prop :=
  let delta := (8 * k * b)^2 - 4 * (4 * k^2 + 1) * (4 * b^2 - 4) in
  delta > 0

noncomputable def line_properties (k : ℝ) : Prop :=
  4 * k^2 + 1 - b^2 > 0

noncomputable def max_triangle_area (b k : ℝ) : ℝ :=
  if 0 < b ∧ b < real.sqrt 2 ∧ 4 * k^2 = 1 then
    1
  else
    0

theorem ellipse_problem (a b c k b_: ℝ) (x1 x2 : ℝ) :
  ellipse_equation a b c →
  line_slopes_geometric k b_ x1 x2 →
  delta_discriminant b_ k →
  line_properties k →
  max_triangle_area b_ k = 1 := sorry

end ellipse_problem_l286_286799


namespace problem_S_equal_102_l286_286565

-- Define the values in Lean
def S : ℕ := 1 * 3^1 + 2 * 3^2 + 3 * 3^3

-- Theorem to prove that S is equal to 102
theorem problem_S_equal_102 : S = 102 :=
by
  sorry

end problem_S_equal_102_l286_286565


namespace shortest_chord_length_l286_286665

theorem shortest_chord_length
  (x y : ℝ)
  (hx : x^2 + y^2 - 6 * x - 8 * y = 0)
  (point_on_circle : (3, 5) = (x, y)) :
  ∃ (length : ℝ), length = 4 * Real.sqrt 6 := 
by
  sorry

end shortest_chord_length_l286_286665


namespace student_total_marks_l286_286276

variables {M P C : ℕ}

theorem student_total_marks
  (h1 : C = P + 20)
  (h2 : (M + C) / 2 = 35) :
  M + P = 50 :=
sorry

end student_total_marks_l286_286276


namespace ab_leq_1_l286_286951

theorem ab_leq_1 {a b : ℝ} (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : a + b = 2) : ab ≤ 1 :=
sorry

end ab_leq_1_l286_286951


namespace JameMade112kProfit_l286_286357

def JameProfitProblem : Prop :=
  let initial_purchase_cost := 40000
  let feeding_cost_rate := 0.2
  let num_cattle := 100
  let weight_per_cattle := 1000
  let sell_price_per_pound := 2
  let additional_feeding_cost := initial_purchase_cost * feeding_cost_rate
  let total_feeding_cost := initial_purchase_cost + additional_feeding_cost
  let total_purchase_and_feeding_cost := initial_purchase_cost + total_feeding_cost
  let total_revenue := num_cattle * weight_per_cattle * sell_price_per_pound
  let profit := total_revenue - total_purchase_and_feeding_cost
  profit = 112000

theorem JameMade112kProfit :
  JameProfitProblem :=
by
  -- Proof goes here
  sorry

end JameMade112kProfit_l286_286357


namespace nat_triple_solution_l286_286650

theorem nat_triple_solution (x y n : ℕ) :
  (x! + y!) / n! = 3^n ↔ (x = 1 ∧ y = 2 ∧ n = 1) ∨ (x = 2 ∧ y = 1 ∧ n = 1) := 
by
  sorry

end nat_triple_solution_l286_286650


namespace factor_expression_l286_286010

theorem factor_expression (a : ℝ) : 74 * a^2 + 222 * a + 148 = 74 * (a + 2) * (a + 1) :=
by
  sorry

end factor_expression_l286_286010


namespace expand_expression_l286_286299

theorem expand_expression (x y : ℝ) : 24 * (3 * x - 4 * y + 6) = 72 * x - 96 * y + 144 := 
by
  sorry

end expand_expression_l286_286299


namespace number_of_ways_to_assign_shifts_l286_286738

def workers : List String := ["A", "B", "C"]

theorem number_of_ways_to_assign_shifts :
  let shifts := ["day", "night"]
  (workers.length * (workers.length - 1)) = 6 := by
  sorry

end number_of_ways_to_assign_shifts_l286_286738


namespace number_of_red_balls_l286_286901

theorem number_of_red_balls (total_balls : ℕ) (probability : ℚ) (num_red_balls : ℕ) 
  (h1 : total_balls = 12) 
  (h2 : probability = 1 / 22) 
  (h3 : (num_red_balls * (num_red_balls - 1) : ℚ) / (total_balls * (total_balls - 1)) = probability) :
  num_red_balls = 3 := 
by
  sorry

end number_of_red_balls_l286_286901


namespace min_project_time_l286_286283

theorem min_project_time (A B C : ℝ) (D : ℝ := 12) :
  (1 / B + 1 / C) = 1 / 2 →
  (1 / A + 1 / C) = 1 / 3 →
  (1 / A + 1 / B) = 1 / 4 →
  (1 / D) = 1 / 12 →
  ∃ x : ℝ, x = 8 / 5 ∧ 1 / x = 1 / A + 1 / B + 1 / C + 1 / (12:ℝ) :=
by
  intros h1 h2 h3 h4
  -- Combination of given hypotheses to prove the goal
  sorry

end min_project_time_l286_286283


namespace maximize_binom_term_l286_286008

theorem maximize_binom_term :
  ∃ k, k ∈ Finset.range (207) ∧
  (∀ m ∈ Finset.range (207), (Nat.choose 206 k * (Real.sqrt 5)^k) ≥ (Nat.choose 206 m * (Real.sqrt 5)^m)) ∧ k = 143 :=
sorry

end maximize_binom_term_l286_286008


namespace relationship_between_x_y_l286_286524

theorem relationship_between_x_y (x y : ℝ) (h1 : x^2 - y^2 > 2 * x) (h2 : x * y < y) : x < y ∧ y < 0 := 
sorry

end relationship_between_x_y_l286_286524


namespace combined_experience_l286_286691

noncomputable def james_experience : ℕ := 20
noncomputable def john_experience_8_years_ago : ℕ := 2 * (james_experience - 8)
noncomputable def john_current_experience : ℕ := john_experience_8_years_ago + 8
noncomputable def mike_experience : ℕ := john_current_experience - 16

theorem combined_experience :
  james_experience + john_current_experience + mike_experience = 68 :=
by
  sorry

end combined_experience_l286_286691


namespace quadratic_equation_original_eq_l286_286604

theorem quadratic_equation_original_eq :
  ∃ (α β : ℝ), (α + β = 3) ∧ (α * β = -6) ∧ (∀ (x : ℝ), x^2 - 3 * x - 6 = 0 → (x = α ∨ x = β)) :=
sorry

end quadratic_equation_original_eq_l286_286604


namespace expression_evaluation_l286_286775

theorem expression_evaluation : abs (abs (-abs (-2 + 1) - 2) + 2) = 5 := 
by  
  sorry

end expression_evaluation_l286_286775


namespace number_is_100_l286_286185

theorem number_is_100 (x : ℝ) (h : 0.60 * (3 / 5) * x = 36) : x = 100 :=
by sorry

end number_is_100_l286_286185


namespace true_proposition_l286_286992

theorem true_proposition : 
  (∃ x0 : ℝ, x0 > 0 ∧ 3^x0 + x0 = 2016) ∧ 
  ¬(∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, abs x - a * x = abs (-x) - a * (-x)) := by
  sorry

end true_proposition_l286_286992


namespace condo_cats_l286_286529

theorem condo_cats (x y : ℕ) (h1 : 2 * x + y = 29) : 6 * x + 3 * y = 87 := by
  sorry

end condo_cats_l286_286529


namespace game_ends_in_draw_for_all_n_l286_286393

noncomputable def andrey_representation_count (n : ℕ) : ℕ := 
  -- The function to count Andrey's representation should be defined here
  sorry

noncomputable def petya_representation_count (n : ℕ) : ℕ := 
  -- The function to count Petya's representation should be defined here
  sorry

theorem game_ends_in_draw_for_all_n (n : ℕ) (h : 0 < n) : 
  andrey_representation_count n = petya_representation_count n :=
  sorry

end game_ends_in_draw_for_all_n_l286_286393


namespace jose_peanuts_l286_286987

/-- If Kenya has 133 peanuts and this is 48 more than what Jose has,
    then Jose has 85 peanuts. -/
theorem jose_peanuts (j k : ℕ) (h1 : k = j + 48) (h2 : k = 133) : j = 85 :=
by
  -- Proof goes here
  sorry

end jose_peanuts_l286_286987


namespace probability_of_Ace_then_King_l286_286408

def numAces : ℕ := 4
def numKings : ℕ := 4
def totalCards : ℕ := 52

theorem probability_of_Ace_then_King : 
  (numAces / totalCards) * (numKings / (totalCards - 1)) = 4 / 663 :=
by
  sorry

end probability_of_Ace_then_King_l286_286408


namespace tan_75_eq_2_plus_sqrt_3_l286_286091

theorem tan_75_eq_2_plus_sqrt_3 : Real.tan (75 * Real.pi / 180) = 2 + Real.sqrt 3 := 
sorry

end tan_75_eq_2_plus_sqrt_3_l286_286091


namespace concert_attendance_difference_l286_286703

theorem concert_attendance_difference :
  let first_concert := 65899
  let second_concert := 66018
  second_concert - first_concert = 119 :=
by
  sorry

end concert_attendance_difference_l286_286703


namespace funfair_initial_visitors_l286_286620

theorem funfair_initial_visitors {a : ℕ} (ha1 : 50 * a - 40 > 0) (ha2 : 90 - 20 * a > 0) (ha3 : 50 * a - 40 > 90 - 20 * a) :
  (50 * a - 40 = 60) ∨ (50 * a - 40 = 110) ∨ (50 * a - 40 = 160) :=
sorry

end funfair_initial_visitors_l286_286620


namespace construct_pairwise_tangent_circles_l286_286672

-- Define the three points A, B, and C in a 2D plane.
variables (A B C : EuclideanSpace ℝ (Fin 2))

/--
  Given three points A, B, and C in the plane, 
  it is possible to construct three circles that are pairwise tangent at these points.
-/
theorem construct_pairwise_tangent_circles (A B C : EuclideanSpace ℝ (Fin 2)) :
  ∃ (O1 O2 O3 : EuclideanSpace ℝ (Fin 2)) (r1 r2 r3 : ℝ),
    r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧
    dist O1 O2 = r1 + r2 ∧
    dist O2 O3 = r2 + r3 ∧
    dist O3 O1 = r3 + r1 ∧
    dist O1 A = r1 ∧ dist O2 B = r2 ∧ dist O3 C = r3 :=
sorry

end construct_pairwise_tangent_circles_l286_286672


namespace area_after_trimming_l286_286278

-- Define the conditions
def original_side_length : ℝ := 22
def trim_x : ℝ := 6
def trim_y : ℝ := 5

-- Calculate dimensions after trimming
def new_length : ℝ := original_side_length - trim_x
def new_width : ℝ := original_side_length - trim_y

-- Define the goal
theorem area_after_trimming : new_length * new_width = 272 := by
  sorry

end area_after_trimming_l286_286278


namespace math_proof_l286_286432

noncomputable def problem (a b : ℝ) : Prop :=
  a - b = 2 ∧ a^2 + b^2 = 25 → a * b = 10.5

-- We state the problem as a theorem:
theorem math_proof (a b : ℝ) (h1: a - b = 2) (h2: a^2 + b^2 = 25) : a * b = 10.5 :=
by {
  sorry -- Proof goes here
}

end math_proof_l286_286432


namespace ring_arrangements_l286_286663

open Nat

theorem ring_arrangements : 
  let rings := 10
  let selected_rings := 6
  let fingers := 4
  let binom := Nat.choose rings selected_rings
  let perm := Nat.factorial selected_rings
  let dist := fingers ^ selected_rings
  n = binom * perm * dist :=
by
  let rings := 10
  let selected_rings := 6
  let fingers := 4
  let binom := Nat.choose rings selected_rings
  let perm := Nat.factorial selected_rings
  let dist := fingers ^ selected_rings
  have : n = 210 * 720 * 4096 := sorry
  exact this

end ring_arrangements_l286_286663


namespace original_time_40_l286_286446

theorem original_time_40
  (S T : ℝ)
  (h1 : ∀ D : ℝ, D = S * T)
  (h2 : ∀ D : ℝ, D = 0.8 * S * (T + 10)) :
  T = 40 :=
by
  sorry

end original_time_40_l286_286446
