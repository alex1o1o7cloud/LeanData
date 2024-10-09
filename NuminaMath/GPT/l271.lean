import Mathlib

namespace reduction_amount_is_250_l271_27128

-- Definitions from the conditions
def original_price : ℝ := 500
def reduction_rate : ℝ := 0.5

-- The statement to be proved
theorem reduction_amount_is_250 : (reduction_rate * original_price) = 250 := by
  sorry

end reduction_amount_is_250_l271_27128


namespace trapezoid_ratio_l271_27164

theorem trapezoid_ratio (A B C D M N K : Type) 
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup M] [AddCommGroup N] [AddCommGroup K]
  (CM MD CN NA AD BC : ℝ)
  (h1 : CM / MD = 4 / 3)
  (h2 : CN / NA = 4 / 3) 
  : AD / BC = 7 / 12 :=
by
  sorry

end trapezoid_ratio_l271_27164


namespace original_oil_weight_is_75_l271_27184

def initial_oil_weight (original : ℝ) : Prop :=
  let first_remaining := original / 2
  let second_remaining := first_remaining * (4 / 5)
  second_remaining = 30

theorem original_oil_weight_is_75 : ∃ (original : ℝ), initial_oil_weight original ∧ original = 75 :=
by
  use 75
  unfold initial_oil_weight
  sorry

end original_oil_weight_is_75_l271_27184


namespace tagged_fish_ratio_l271_27103

theorem tagged_fish_ratio (tagged_first_catch : ℕ) 
(tagged_second_catch : ℕ) (total_second_catch : ℕ) 
(h1 : tagged_first_catch = 30) (h2 : tagged_second_catch = 2) 
(h3 : total_second_catch = 50) : tagged_second_catch / total_second_catch = 1 / 25 :=
by
  sorry

end tagged_fish_ratio_l271_27103


namespace total_possible_rankings_l271_27194

-- Define the players
inductive Player
| P | Q | R | S

-- Define the tournament results
inductive Result
| win | lose

-- Define Saturday's match outcomes
structure SaturdayOutcome :=
(P_vs_Q: Result)
(R_vs_S: Result)

-- Function to compute the number of possible tournament ranking sequences
noncomputable def countTournamentSequences : Nat :=
  let saturdayOutcomes: List SaturdayOutcome :=
    [ {P_vs_Q := Result.win, R_vs_S := Result.win}
    , {P_vs_Q := Result.win, R_vs_S := Result.lose}
    , {P_vs_Q := Result.lose, R_vs_S := Result.win}
    , {P_vs_Q := Result.lose, R_vs_S := Result.lose}
    ]
  let sundayPermutations (outcome : SaturdayOutcome) : Nat :=
    2 * 2  -- 2 permutations for 1st and 2nd places * 2 permutations for 3rd and 4th places per each outcome
  saturdayOutcomes.foldl (fun acc outcome => acc + sundayPermutations outcome) 0

-- Define the theorem to prove the total number of permutations
theorem total_possible_rankings : countTournamentSequences = 8 :=
by
  -- Proof steps here (proof omitted)
  sorry

end total_possible_rankings_l271_27194


namespace binomial_standard_deviation_l271_27185

noncomputable def standard_deviation_binomial (n : ℕ) (p : ℝ) : ℝ :=
  Real.sqrt (n * p * (1 - p))

theorem binomial_standard_deviation (n : ℕ) (p : ℝ) (hn : 0 ≤ n) (hp : 0 ≤ p) (hp1: p ≤ 1) :
  standard_deviation_binomial n p = Real.sqrt (n * p * (1 - p)) :=
by
  sorry

end binomial_standard_deviation_l271_27185


namespace alpha_eq_beta_l271_27102

variable {α β : ℝ}

theorem alpha_eq_beta
  (h_alpha : 0 < α ∧ α < (π / 2))
  (h_beta : 0 < β ∧ β < (π / 2))
  (h_sin : Real.sin (α + β) + Real.sin (α - β) = Real.sin (2 * β)) :
  α = β :=
by
  sorry

end alpha_eq_beta_l271_27102


namespace find_colored_copies_l271_27168

variable (cost_c cost_w total_copies total_cost : ℝ)
variable (colored_copies white_copies : ℝ)

def colored_copies_condition (cost_c cost_w total_copies total_cost : ℝ) :=
  ∃ (colored_copies white_copies : ℝ),
    colored_copies + white_copies = total_copies ∧
    cost_c * colored_copies + cost_w * white_copies = total_cost

theorem find_colored_copies :
  colored_copies_condition 0.10 0.05 400 22.50 → 
  ∃ (c : ℝ), c = 50 :=
by 
  sorry

end find_colored_copies_l271_27168


namespace initial_soccer_balls_l271_27198

theorem initial_soccer_balls {x : ℕ} (h : x + 18 = 24) : x = 6 := 
sorry

end initial_soccer_balls_l271_27198


namespace minimum_stamps_to_make_47_cents_l271_27116

theorem minimum_stamps_to_make_47_cents (c f : ℕ) (h : 5 * c + 7 * f = 47) : c + f = 7 :=
sorry

end minimum_stamps_to_make_47_cents_l271_27116


namespace number_of_items_in_U_l271_27147

theorem number_of_items_in_U (U A B : Finset ℕ)
  (hB : B.card = 41)
  (not_A_nor_B : U.card - A.card - B.card + (A ∩ B).card = 59)
  (hAB : (A ∩ B).card = 23)
  (hA : A.card = 116) :
  U.card = 193 :=
by sorry

end number_of_items_in_U_l271_27147


namespace intervals_of_monotonicity_range_of_values_for_a_l271_27133

noncomputable def f (a x : ℝ) : ℝ := x - a * Real.log x + (1 + a) / x

theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x ∈ Set.Ioi 0, a ≤ -1 → deriv (f a) x > 0) ∧
  (∀ x ∈ Set.Ioc 0 (1 + a), -1 < a → deriv (f a) x < 0) ∧
  (∀ x ∈ Set.Ioi (1 + a), -1 < a → deriv (f a) x > 0) :=
sorry

theorem range_of_values_for_a (a : ℝ) (e : ℝ) (h : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 1 e, f a x ≤ 0) → (a ≤ -2 ∨ a ≥ (e^2 + 1) / (e - 1)) :=
sorry

end intervals_of_monotonicity_range_of_values_for_a_l271_27133


namespace password_probability_l271_27104

def is_prime_single_digit : Fin 10 → Prop
| 2 | 3 | 5 | 7 => true
| _ => false

def is_vowel : Char → Prop
| 'A' | 'E' | 'I' | 'O' | 'U' => true
| _ => false

def is_positive_even_single_digit : Fin 9 → Prop
| 2 | 4 | 6 | 8 => true
| _ => false

def prime_probability : ℚ := 4 / 10
def vowel_probability : ℚ := 5 / 26
def even_pos_digit_probability : ℚ := 4 / 9

theorem password_probability :
  prime_probability * vowel_probability * even_pos_digit_probability = 8 / 117 := by
  sorry

end password_probability_l271_27104


namespace larger_number_of_hcf_lcm_l271_27181

theorem larger_number_of_hcf_lcm (hcf : ℕ) (a b : ℕ) (f1 f2 : ℕ) 
  (hcf_condition : hcf = 20) 
  (factors_condition : f1 = 21 ∧ f2 = 23) 
  (lcm_condition : Nat.lcm a b = hcf * f1 * f2):
  max a b = 460 := 
  sorry

end larger_number_of_hcf_lcm_l271_27181


namespace cashier_can_satisfy_request_l271_27156

theorem cashier_can_satisfy_request (k : ℕ) (h : k > 8) : ∃ m n : ℕ, k = 3 * m + 5 * n :=
sorry

end cashier_can_satisfy_request_l271_27156


namespace lcm_48_147_l271_27175

theorem lcm_48_147 : Nat.lcm 48 147 = 2352 := sorry

end lcm_48_147_l271_27175


namespace hyperbola_foci_y_axis_l271_27172

theorem hyperbola_foci_y_axis (a b : ℝ) (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1 → (1/a < 0 ∧ 1/b > 0)) : a < 0 ∧ b > 0 :=
by
  sorry

end hyperbola_foci_y_axis_l271_27172


namespace total_time_to_fill_tank_with_leak_l271_27166

theorem total_time_to_fill_tank_with_leak
  (C : ℝ) -- Capacity of the tank
  (rate1 : ℝ := C / 20) -- Rate of pipe 1 filling the tank
  (rate2 : ℝ := C / 30) -- Rate of pipe 2 filling the tank
  (combined_rate : ℝ := rate1 + rate2) -- Combined rate of both pipes
  (effective_rate : ℝ := (2 / 3) * combined_rate) -- Effective rate considering the leak
  : (C / effective_rate = 18) :=
by
  -- The proof would go here but is removed per the instructions.
  sorry

end total_time_to_fill_tank_with_leak_l271_27166


namespace part_a_part_b_l271_27123

namespace ShaltaevBoltaev

variables {s b : ℕ}

-- Condition: 175s > 125b
def condition1 (s b : ℕ) : Prop := 175 * s > 125 * b

-- Condition: 175s < 126b
def condition2 (s b : ℕ) : Prop := 175 * s < 126 * b

-- Prove that 3s + b > 80
theorem part_a (s b : ℕ) (h1 : condition1 s b) (h2 : condition2 s b) : 
  3 * s + b > 80 := sorry

-- Prove that 3s + b > 100
theorem part_b (s b : ℕ) (h1 : condition1 s b) (h2 : condition2 s b) : 
  3 * s + b > 100 := sorry

end ShaltaevBoltaev

end part_a_part_b_l271_27123


namespace number_of_raccoons_l271_27109

/-- Jason pepper-sprays some raccoons and 6 times as many squirrels. 
Given that he pepper-sprays a total of 84 animals, the number of raccoons he pepper-sprays is 12. -/
theorem number_of_raccoons (R : Nat) (h1 : 84 = R + 6 * R) : R = 12 :=
by
  sorry

end number_of_raccoons_l271_27109


namespace price_of_other_frisbees_proof_l271_27113

noncomputable def price_of_other_frisbees (P : ℝ) : Prop :=
  ∃ x : ℝ, x + (60 - x) = 60 ∧ x ≥ 0 ∧ P * x + 4 * (60 - x) = 204 ∧ (60 - x) ≥ 24

theorem price_of_other_frisbees_proof : price_of_other_frisbees 3 :=
by
  sorry

end price_of_other_frisbees_proof_l271_27113


namespace fewest_printers_l271_27197

/-!
# Fewest Printers Purchase Problem
Given two types of computer printers costing $350 and $200 per unit, respectively,
given that the company wants to spend equal amounts on both types of printers.
Prove that the fewest number of printers the company can purchase is 11.
-/

theorem fewest_printers (p1 p2 : ℕ) (h1 : p1 = 350) (h2 : p2 = 200) :
  ∃ n1 n2 : ℕ, p1 * n1 = p2 * n2 ∧ n1 + n2 = 11 := 
sorry

end fewest_printers_l271_27197


namespace negation_equivalence_l271_27110

variable (U : Type) (S R : U → Prop)

-- Original statement: All students of this university are non-residents, i.e., ∀ x, S(x) → ¬ R(x)
def original_statement : Prop := ∀ x, S x → ¬ R x

-- Negation of the original statement: ∃ x, S(x) ∧ R(x)
def negated_statement : Prop := ∃ x, S x ∧ R x

-- Lean statement to prove that the negation of the original statement is equivalent to some students are residents
theorem negation_equivalence : ¬ original_statement U S R = negated_statement U S R :=
sorry

end negation_equivalence_l271_27110


namespace zebra_crossing_distance_l271_27137

theorem zebra_crossing_distance
  (boulevard_width : ℝ)
  (distance_along_stripes : ℝ)
  (stripe_length : ℝ)
  (distance_between_stripes : ℝ) :
  boulevard_width = 60 →
  distance_along_stripes = 22 →
  stripe_length = 65 →
  distance_between_stripes = (60 * 22) / 65 →
  distance_between_stripes = 20.31 :=
by
  intros h1 h2 h3 h4
  sorry

end zebra_crossing_distance_l271_27137


namespace nancy_total_spending_l271_27134

/-- A bead shop sells crystal beads at $9 each and metal beads at $10 each.
    Nancy buys one set of crystal beads and two sets of metal beads. -/
def cost_of_crystal_bead := 9
def cost_of_metal_bead := 10
def sets_of_crystal_beads_bought := 1
def sets_of_metal_beads_bought := 2

/-- Prove the total amount Nancy spends is $29 given the conditions. -/
theorem nancy_total_spending :
  sets_of_crystal_beads_bought * cost_of_crystal_bead +
  sets_of_metal_beads_bought * cost_of_metal_bead = 29 :=
by
  sorry

end nancy_total_spending_l271_27134


namespace tan_add_pi_over_4_l271_27115

theorem tan_add_pi_over_4 (α : ℝ) (h : Real.tan α = 2) : Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end tan_add_pi_over_4_l271_27115


namespace no_solution_inequality_l271_27162

theorem no_solution_inequality (m x : ℝ) (h1 : x - 2 * m < 0) (h2 : x + m > 2) : m ≤ 2 / 3 :=
  sorry

end no_solution_inequality_l271_27162


namespace ratio_is_1_to_3_l271_27192

-- Definitions based on the conditions
def washed_on_wednesday : ℕ := 6
def washed_on_thursday : ℕ := 2 * washed_on_wednesday
def washed_on_friday : ℕ := washed_on_thursday / 2
def total_washed : ℕ := 26
def washed_on_saturday : ℕ := total_washed - washed_on_wednesday - washed_on_thursday - washed_on_friday

-- The ratio calculation
def ratio_saturday_to_wednesday : ℚ := washed_on_saturday / washed_on_wednesday

-- The theorem to prove
theorem ratio_is_1_to_3 : ratio_saturday_to_wednesday = 1 / 3 :=
by
  -- Insert proof here
  sorry

end ratio_is_1_to_3_l271_27192


namespace LCM_quotient_l271_27140

-- Define M as the least common multiple of integers from 12 to 25
def LCM_12_25 : ℕ := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 
                       (Nat.lcm 12 13) 14) 15) 16) 17) (Nat.lcm (Nat.lcm 
                       (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 18 19) 20) 21) 22) 23) 24)

-- Define N as the least common multiple of LCM_12_25, 36, 38, 40, 42, 44, 45
def N : ℕ := Nat.lcm LCM_12_25 (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 36 38) 40) 42) (Nat.lcm 44 45))

-- Prove that N / LCM_12_25 = 1
theorem LCM_quotient : N / LCM_12_25 = 1 := by
    sorry

end LCM_quotient_l271_27140


namespace total_balloons_l271_27136

-- Define the conditions
def joan_balloons : ℕ := 9
def sally_balloons : ℕ := 5
def jessica_balloons : ℕ := 2

-- The statement we want to prove
theorem total_balloons : joan_balloons + sally_balloons + jessica_balloons = 16 :=
by
  sorry

end total_balloons_l271_27136


namespace locus_of_P_is_ellipse_l271_27196

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

end locus_of_P_is_ellipse_l271_27196


namespace no_b_satisfies_l271_27114

theorem no_b_satisfies (b : ℝ) : ¬ (2 * 1 - b * (-2) + 1 ≤ 0 ∧ 2 * (-1) - b * 2 + 1 ≤ 0) :=
by
  sorry

end no_b_satisfies_l271_27114


namespace max_price_reduction_l271_27139

theorem max_price_reduction (CP SP : ℝ) (profit_margin : ℝ) (max_reduction : ℝ) :
  CP = 1000 ∧ SP = 1500 ∧ profit_margin = 0.05 → SP - max_reduction = CP * (1 + profit_margin) → max_reduction = 450 :=
by {
  sorry
}

end max_price_reduction_l271_27139


namespace solve_for_k_in_quadratic_l271_27144

theorem solve_for_k_in_quadratic :
  ∃ k : ℝ, (∀ x1 x2 : ℝ,
    x1 + x2 = 3 ∧
    x1 * x2 + 2 * x1 + 2 * x2 = 1 ∧
    (x1^2 - 3*x1 + k = 0) ∧ (x2^2 - 3*x2 + k = 0)) →
  k = -5 :=
sorry

end solve_for_k_in_quadratic_l271_27144


namespace increasing_function_range_a_l271_27143

theorem increasing_function_range_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = if x > 1 then a^x else (4 - a/2)*x + 2) ∧
  (∀ x y, x < y → f x ≤ f y) →
  4 ≤ a ∧ a < 8 :=
by
  sorry

end increasing_function_range_a_l271_27143


namespace students_taking_only_science_l271_27119

theorem students_taking_only_science (total_students : ℕ) (students_science : ℕ) (students_math : ℕ)
  (h1 : total_students = 120) (h2 : students_science = 80) (h3 : students_math = 75) :
  (students_science - (students_science + students_math - total_students)) = 45 :=
by
  sorry

end students_taking_only_science_l271_27119


namespace nylon_needed_for_one_dog_collor_l271_27127

-- Define the conditions as given in the problem
def nylon_for_dog (x : ℝ) : ℝ := x
def nylon_for_cat : ℝ := 10
def total_nylon_used (x : ℝ) : ℝ := 9 * (nylon_for_dog x) + 3 * (nylon_for_cat)

-- Prove the required statement under the given conditions
theorem nylon_needed_for_one_dog_collor : total_nylon_used 18 = 192 :=
by
  -- adding the proof step using sorry as required
  sorry

end nylon_needed_for_one_dog_collor_l271_27127


namespace ternary_1021_to_decimal_l271_27188

-- Define the function to convert a ternary string to decimal
def ternary_to_decimal (n : String) : Nat :=
  n.foldr (fun c acc => acc * 3 + (c.toNat - '0'.toNat)) 0

-- The statement to prove
theorem ternary_1021_to_decimal : ternary_to_decimal "1021" = 34 := by
  sorry

end ternary_1021_to_decimal_l271_27188


namespace rectangle_ratio_of_semicircles_l271_27191

theorem rectangle_ratio_of_semicircles (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h : a * b = π * b^2) : a / b = π := by
  sorry

end rectangle_ratio_of_semicircles_l271_27191


namespace current_prices_l271_27130

theorem current_prices (initial_ram_price initial_ssd_price : ℝ) 
  (ram_increase_1 ram_decrease_1 ram_decrease_2 : ℝ) 
  (ssd_increase_1 ssd_decrease_1 ssd_decrease_2 : ℝ) 
  (initial_ram : initial_ram_price = 50) 
  (initial_ssd : initial_ssd_price = 100) 
  (ram_increase_factor : ram_increase_1 = 0.30 * initial_ram_price) 
  (ram_decrease_factor_1 : ram_decrease_1 = 0.15 * (initial_ram_price + ram_increase_1)) 
  (ram_decrease_factor_2 : ram_decrease_2 = 0.20 * ((initial_ram_price + ram_increase_1) - ram_decrease_1)) 
  (ssd_increase_factor : ssd_increase_1 = 0.10 * initial_ssd_price) 
  (ssd_decrease_factor_1 : ssd_decrease_1 = 0.05 * (initial_ssd_price + ssd_increase_1)) 
  (ssd_decrease_factor_2 : ssd_decrease_2 = 0.12 * ((initial_ssd_price + ssd_increase_1) - ssd_decrease_1)) 
  : 
  ((initial_ram_price + ram_increase_1 - ram_decrease_1 - ram_decrease_2) = 44.20) ∧ 
  ((initial_ssd_price + ssd_increase_1 - ssd_decrease_1 - ssd_decrease_2) = 91.96) := 
by
  sorry

end current_prices_l271_27130


namespace gunny_bag_can_hold_packets_l271_27182

theorem gunny_bag_can_hold_packets :
  let ton_to_kg := 1000
  let max_capacity_tons := 13
  let pound_to_kg := 0.453592
  let ounce_to_g := 28.3495
  let kilo_to_g := 1000
  let wheat_packet_pounds := 16
  let wheat_packet_ounces := 4
  let max_capacity_kg := max_capacity_tons * ton_to_kg
  let wheat_packet_kg := wheat_packet_pounds * pound_to_kg + (wheat_packet_ounces * ounce_to_g) / kilo_to_g
  max_capacity_kg / wheat_packet_kg >= 1763 := 
by
  sorry

end gunny_bag_can_hold_packets_l271_27182


namespace calculate_expression_l271_27135

theorem calculate_expression : 
  ∀ (x y z : ℤ), x = 2 → y = -3 → z = 7 → (x^2 + y^2 + z^2 - 2 * x * y) = 74 :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end calculate_expression_l271_27135


namespace police_emergency_number_prime_divisor_l271_27131

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 1000 * k + 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n :=
sorry

end police_emergency_number_prime_divisor_l271_27131


namespace min_distance_from_curve_to_focus_l271_27153

noncomputable def minDistanceToFocus (x y θ : ℝ) : ℝ :=
  let a := 3
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  a - c

theorem min_distance_from_curve_to_focus :
  ∀ θ : ℝ, minDistanceToFocus (2 * Real.cos θ) (3 * Real.sin θ) θ = 3 - Real.sqrt 5 :=
by
  sorry

end min_distance_from_curve_to_focus_l271_27153


namespace equation1_no_solution_equation2_solution_l271_27169

/-- Prove that the equation (4-x)/(x-3) + 1/(3-x) = 1 has no solution. -/
theorem equation1_no_solution (x : ℝ) : x ≠ 3 → ¬ (4 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by intro hx; sorry

/-- Prove that the equation (x+1)/(x-1) - 6/(x^2-1) = 1 has solution x = 2. -/
theorem equation2_solution (x : ℝ) : x = 2 ↔ (x + 1) / (x - 1) - 6 / (x^2 - 1) = 1 :=
by sorry

end equation1_no_solution_equation2_solution_l271_27169


namespace min_convex_number_l271_27129

noncomputable def minimum_convex_sets (A B C : ℝ × ℝ) : ℕ :=
  if A ≠ B ∧ B ≠ C ∧ C ≠ A then 3 else 4

theorem min_convex_number (A B C : ℝ × ℝ) (h : A ≠ B ∧ B ≠ C ∧ C ≠ A) :
  minimum_convex_sets A B C = 3 :=
by 
  sorry

end min_convex_number_l271_27129


namespace problem1_problem2_problem3_problem4_l271_27179

-- Define predicate conditions and solutions in Lean 4 for each problem

theorem problem1 (x : ℝ) :
  -2 * x^2 + 3 * x + 9 > 0 ↔ (-3 / 2 < x ∧ x < 3) := by
  sorry

theorem problem2 (x : ℝ) :
  (8 - x) / (5 + x) > 1 ↔ (-5 < x ∧ x ≤ 3 / 2) := by
  sorry

theorem problem3 (x : ℝ) :
  ¬ (-x^2 + 2 * x - 3 > 0) ↔ True := by
  sorry

theorem problem4 (x : ℝ) :
  x^2 - 14 * x + 50 > 0 ↔ True := by
  sorry

end problem1_problem2_problem3_problem4_l271_27179


namespace trig_identity_proof_l271_27160

theorem trig_identity_proof :
  2 * (1 / 2) + (Real.sqrt 3 / 2) * Real.sqrt 3 = 5 / 2 :=
by
  sorry

end trig_identity_proof_l271_27160


namespace monomial_sum_exponents_l271_27145

theorem monomial_sum_exponents (m n : ℕ) (h₁ : m - 1 = 2) (h₂ : n = 2) : m^n = 9 := 
by
  sorry

end monomial_sum_exponents_l271_27145


namespace determine_asymptotes_l271_27161

noncomputable def hyperbola_eccentricity_asymptote_relation (a b : ℝ) (e : ℝ) (k : ℝ) :=
  a > 0 ∧ b > 0 ∧ (e = Real.sqrt 2 * |k|) ∧ (k = b / a)

theorem determine_asymptotes (a b : ℝ) (h : hyperbola_eccentricity_asymptote_relation a b (Real.sqrt (a^2 + b^2) / a) (b / a)) :
  true := sorry

end determine_asymptotes_l271_27161


namespace inequality_d_l271_27150

-- We define the polynomial f with integer coefficients
variable (f : ℤ → ℤ)

-- The function for f^k iteration
def iter (f: ℤ → ℤ) : ℕ → ℤ → ℤ
| 0, x => x
| (n + 1), x => f (iter f n x)

-- Definition of d(a, k) based on the problem statement
def d (a : ℤ) (k : ℕ) : ℝ := |(iter f k a : ℤ) - a|

-- Given condition that d(a, k) is positive
axiom d_pos (a : ℤ) (k : ℕ) : 0 < d f a k

-- The statement to be proved
theorem inequality_d (a : ℤ) (k : ℕ) : d f a k ≥ ↑k / 3 := by
  sorry

end inequality_d_l271_27150


namespace max_points_for_top_teams_l271_27195

-- Definitions based on the problem conditions
def points_for_win : ℕ := 3
def points_for_draw : ℕ := 1
def points_for_loss : ℕ := 0
def number_of_teams : ℕ := 8
def number_of_games_between_each_pair : ℕ := 2
def total_games : ℕ := (number_of_teams * (number_of_teams - 1) / 2) * number_of_games_between_each_pair
def total_points_in_tournament : ℕ := total_games * points_for_win
def top_teams : ℕ := 4

-- Theorem stating the correct answer
theorem max_points_for_top_teams : (total_points_in_tournament / number_of_teams = 33) :=
sorry

end max_points_for_top_teams_l271_27195


namespace bricklayer_wall_l271_27165

/-- 
A bricklayer lays a certain number of meters of wall per day and works for a certain number of days.
Given the daily work rate and the number of days worked, this proof shows that the total meters of 
wall laid equals the product of the daily work rate and the number of days.
-/
theorem bricklayer_wall (daily_rate : ℕ) (days_worked : ℕ) (total_meters : ℕ) 
  (h1 : daily_rate = 8) (h2 : days_worked = 15) : total_meters = 120 :=
by {
  sorry
}

end bricklayer_wall_l271_27165


namespace average_ABC_is_three_l271_27112
-- Import the entirety of the Mathlib library

-- Define the required conditions and the theorem to be proved
theorem average_ABC_is_three (A B C : ℝ) 
    (h1 : 2012 * C - 4024 * A = 8048) 
    (h2 : 2012 * B + 6036 * A = 10010) : 
    (A + B + C) / 3 = 3 := 
by
  sorry

end average_ABC_is_three_l271_27112


namespace expand_expression_l271_27193

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  -- This is where the proof steps would go.
  -- I've removed the proof steps and left "sorry" to indicate the proof is omitted.
  sorry

end expand_expression_l271_27193


namespace million_to_scientific_notation_l271_27167

theorem million_to_scientific_notation (population_henan : ℝ) (h : population_henan = 98.83 * 10^6) :
  population_henan = 9.883 * 10^7 :=
by sorry

end million_to_scientific_notation_l271_27167


namespace two_lines_parallel_same_plane_l271_27171

-- Defining the types for lines and planes
variable (Line : Type) (Plane : Type)

-- Defining the relationships similar to the mathematical conditions
variable (parallel_to_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (intersect : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Defining the non-overlapping relationships between lines (assuming these relations are mutually exclusive)
axiom parallel_or_intersect_or_skew : ∀ (a b: Line), 
  (parallel a b ∨ intersect a b ∨ skew a b)

-- The statement we want to prove
theorem two_lines_parallel_same_plane (a b: Line) (α: Plane) :
  parallel_to_plane a α → parallel_to_plane b α → (parallel a b ∨ intersect a b ∨ skew a b) :=
by
  intro ha hb
  apply parallel_or_intersect_or_skew

end two_lines_parallel_same_plane_l271_27171


namespace solve_for_x_l271_27159

-- We state the problem as a theorem.
theorem solve_for_x (y x : ℚ) : 
  (x - 60) / 3 = (4 - 3 * x) / 6 + y → x = (124 + 6 * y) / 5 :=
by
  -- The actual proof part is skipped with sorry.
  sorry

end solve_for_x_l271_27159


namespace quadruple_nested_function_l271_27108

def a (k : ℕ) : ℕ := (k + 1) ^ 2

theorem quadruple_nested_function (k : ℕ) (h : k = 1) : a (a (a (a (k)))) = 458329 :=
by
  rw [h]
  sorry

end quadruple_nested_function_l271_27108


namespace cell_phone_height_l271_27187

theorem cell_phone_height (width perimeter : ℕ) (h1 : width = 9) (h2 : perimeter = 46) : 
  ∃ length : ℕ, length = 14 ∧ perimeter = 2 * (width + length) :=
by
  sorry

end cell_phone_height_l271_27187


namespace probability_nan_kai_l271_27117

theorem probability_nan_kai :
  let total_outcomes := Nat.choose 6 4
  let successful_outcomes := Nat.choose 4 4
  let probability := (successful_outcomes : ℚ) / total_outcomes
  probability = 1 / 15 :=
by
  sorry

end probability_nan_kai_l271_27117


namespace dan_remaining_money_l271_27105

noncomputable def calculate_remaining_money (initial_amount : ℕ) : ℕ :=
  let candy_bars_qty := 5
  let candy_bar_price := 125
  let candy_bars_discount := 10
  let gum_qty := 3
  let gum_price := 80
  let soda_qty := 4
  let soda_price := 240
  let chips_qty := 2
  let chip_price := 350
  let chips_discount := 15
  let low_tax := 7
  let high_tax := 12

  let total_candy_bars_cost := candy_bars_qty * candy_bar_price
  let discounted_candy_bars_cost := total_candy_bars_cost * (100 - candy_bars_discount) / 100

  let total_gum_cost := gum_qty * gum_price

  let total_soda_cost := soda_qty * soda_price

  let total_chips_cost := chips_qty * chip_price
  let discounted_chips_cost := total_chips_cost * (100 - chips_discount) / 100

  let candy_bars_tax := discounted_candy_bars_cost * low_tax / 100
  let gum_tax := total_gum_cost * low_tax / 100

  let soda_tax := total_soda_cost * high_tax / 100
  let chips_tax := discounted_chips_cost * high_tax / 100

  let total_candy_bars_with_tax := discounted_candy_bars_cost + candy_bars_tax
  let total_gum_with_tax := total_gum_cost + gum_tax
  let total_soda_with_tax := total_soda_cost + soda_tax
  let total_chips_with_tax := discounted_chips_cost + chips_tax

  let total_cost := total_candy_bars_with_tax + total_gum_with_tax + total_soda_with_tax + total_chips_with_tax

  initial_amount - total_cost

theorem dan_remaining_money : 
  calculate_remaining_money 10000 = 7399 :=
sorry

end dan_remaining_money_l271_27105


namespace largest_int_with_remainder_l271_27138

theorem largest_int_with_remainder (k : ℤ) (h₁ : k < 95) (h₂ : k % 7 = 5) : k = 94 := by
sorry

end largest_int_with_remainder_l271_27138


namespace find_t_l271_27183

theorem find_t (t : ℝ) : 
  (∃ (m b : ℝ), (∀ x y, (y = m * x + b) → ((x = 1 ∧ y = 7) ∨ (x = 3 ∧ y = 13) ∨ (x = 5 ∧ y = 19))) ∧ (28 = 28 * m + b) ∧ (t = 28 * m + b)) → 
  t = 88 :=
by
  sorry

end find_t_l271_27183


namespace beau_age_today_l271_27141

theorem beau_age_today (sons_age : ℕ) (triplets : ∀ i j : ℕ, i ≠ j → sons_age = 16) 
                       (beau_age_three_years_ago : ℕ) 
                       (H : 3 * (sons_age - 3) = beau_age_three_years_ago) :
  beau_age_three_years_ago + 3 = 42 :=
by
  -- Normally this is the place to write the proof,
  -- but it's enough to outline the theorem statement as per the instructions.
  sorry

end beau_age_today_l271_27141


namespace ratio_swordfish_to_pufferfish_l271_27132

theorem ratio_swordfish_to_pufferfish (P S : ℕ) (n : ℕ) 
  (hP : P = 15)
  (hTotal : S + P = 90)
  (hRelation : S = n * P) : 
  (S : ℚ) / (P : ℚ) = 5 := 
by 
  sorry

end ratio_swordfish_to_pufferfish_l271_27132


namespace smallest_norm_l271_27174

noncomputable def vectorNorm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_norm (v : ℝ × ℝ)
  (h : vectorNorm (v.1 + 4, v.2 + 2) = 10) :
  vectorNorm v >= 10 - 2 * Real.sqrt 5 :=
by
  sorry

end smallest_norm_l271_27174


namespace Carmen_average_speed_l271_27118

/-- Carmen participates in a two-part cycling race. In the first part, she covers 24 miles in 3 hours.
    In the second part, due to fatigue, her speed decreases, and she takes 4 hours to cover 16 miles.
    Calculate Carmen's average speed for the entire race. -/
theorem Carmen_average_speed :
  let distance1 := 24 -- miles in the first part
  let time1 := 3 -- hours in the first part
  let distance2 := 16 -- miles in the second part
  let time2 := 4 -- hours in the second part
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 40 / 7 :=
by
  sorry

end Carmen_average_speed_l271_27118


namespace real_number_x_equal_2_l271_27101

theorem real_number_x_equal_2 (x : ℝ) (i : ℂ) (h : i * i = -1) :
  (1 - 2 * i) * (x + i) = 4 - 3 * i → x = 2 :=
by
  sorry

end real_number_x_equal_2_l271_27101


namespace perpendicular_line_plane_implies_perpendicular_lines_l271_27176

variables {Line Plane : Type}
variables (m n : Line) (α : Plane)

-- Assume inclusion of lines in planes, parallelism, and perpendicularity properties.
variables (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (subset : Line → Plane → Prop) (perpendicular_lines : Line → Line → Prop)

-- Given definitions based on the conditions
variable (is_perpendicular : perpendicular m α)
variable (is_subset : subset n α)

-- Prove that m is perpendicular to n
theorem perpendicular_line_plane_implies_perpendicular_lines
  (h1 : perpendicular m α)
  (h2 : subset n α) :
  perpendicular_lines m n :=
sorry

end perpendicular_line_plane_implies_perpendicular_lines_l271_27176


namespace last_passenger_seats_probability_l271_27107

theorem last_passenger_seats_probability (n : ℕ) (hn : n > 0) :
  ∀ (P : ℝ), P = 1 / 2 :=
by
  sorry

end last_passenger_seats_probability_l271_27107


namespace solution_mix_percentage_l271_27111

theorem solution_mix_percentage
  (x y z : ℝ)
  (hx1 : x + y + z = 100)
  (hx2 : 0.40 * x + 0.50 * y + 0.30 * z = 46)
  (hx3 : z = 100 - x - y) :
  x = 40 ∧ y = 60 ∧ z = 0 :=
by
  sorry

end solution_mix_percentage_l271_27111


namespace line_segment_length_is_0_7_l271_27186

def isLineSegment (length : ℝ) (finite : Bool) : Prop :=
  finite = true ∧ length = 0.7

theorem line_segment_length_is_0_7 : isLineSegment 0.7 true :=
by
  sorry

end line_segment_length_is_0_7_l271_27186


namespace Eithan_savings_account_l271_27146

variable (initial_amount wife_firstson_share firstson_remaining firstson_secondson_share 
          secondson_remaining secondson_thirdson_share thirdson_remaining 
          charity_donation remaining_after_charity tax_rate final_remaining : ℝ)

theorem Eithan_savings_account:
  initial_amount = 5000 →
  wife_firstson_share = initial_amount * (2/5) →
  firstson_remaining = initial_amount - wife_firstson_share →
  firstson_secondson_share = firstson_remaining * (3/10) →
  secondson_remaining = firstson_remaining - firstson_secondson_share →
  thirdson_remaining = secondson_remaining * (1-0.30) →
  charity_donation = 200 →
  remaining_after_charity = thirdson_remaining - charity_donation →
  tax_rate = 0.05 →
  final_remaining = remaining_after_charity * (1 - tax_rate) →
  final_remaining = 927.2 := 
  by
    intros
    sorry

end Eithan_savings_account_l271_27146


namespace taller_tree_height_l271_27124

/-- The top of one tree is 20 feet higher than the top of another tree.
    The heights of the two trees are in the ratio 2:3.
    The shorter tree is 40 feet tall.
    Show that the height of the taller tree is 60 feet. -/
theorem taller_tree_height 
  (shorter_tree_height : ℕ) 
  (height_difference : ℕ)
  (height_ratio_num : ℕ)
  (height_ratio_denom : ℕ)
  (H1 : shorter_tree_height = 40)
  (H2 : height_difference = 20)
  (H3 : height_ratio_num = 2)
  (H4 : height_ratio_denom = 3)
  : ∃ taller_tree_height : ℕ, taller_tree_height = 60 :=
by
  sorry

end taller_tree_height_l271_27124


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l271_27152

theorem solve_quadratic_1 (x : Real) : x^2 - 2 * x - 4 = 0 ↔ (x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5) :=
by
  sorry

theorem solve_quadratic_2 (x : Real) : (x - 1)^2 = 2 * (x - 1) ↔ (x = 1 ∨ x = 3) :=
by
  sorry

theorem solve_quadratic_3 (x : Real) : (x + 1)^2 = 4 * x^2 ↔ (x = 1 ∨ x = -1 / 3) :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l271_27152


namespace purchase_price_of_article_l271_27190

theorem purchase_price_of_article (P M : ℝ) (h1 : M = 55) (h2 : M = 0.30 * P + 12) : P = 143.33 :=
  sorry

end purchase_price_of_article_l271_27190


namespace squares_difference_l271_27155

theorem squares_difference (a b : ℕ) (h₁ : a = 601) (h₂ : b = 597) : a^2 - b^2 = 4792 := by
  rw [h₁, h₂]
  -- insert actual proof here
  sorry

end squares_difference_l271_27155


namespace similar_triangles_proportionalities_l271_27120

-- Definitions of the conditions as hypotheses
variables (A B C D E F : Type)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
variables (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
variables (AB_DE_ratio : AB / DE = 1 / 2)
variables (BC_length : BC = 2)

-- Defining the hypothesis of similarity
def SimilarTriangles (A B C D E F : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] : Prop :=
  ∀ (AB BC CA DE EF FD : ℝ), (AB / DE = BC / EF) ∧ (BC / EF = CA / FD) ∧ (CA / FD = AB / DE)

-- The proof statement
theorem similar_triangles_proportionalities (A B C D E F : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (triangle_ABC_sim_triangle_DEF : SimilarTriangles A B C D E F)
  (AB_DE_ratio : AB / DE = 1 / 2)
  (BC_length : BC = 2) : 
  EF = 4 := 
by sorry

end similar_triangles_proportionalities_l271_27120


namespace sophia_read_more_pages_l271_27199

variable (total_pages : ℝ) (finished_fraction : ℝ)
variable (pages_read : ℝ) (pages_left : ℝ) (pages_more : ℝ)

theorem sophia_read_more_pages :
  total_pages = 269.99999999999994 ∧
  finished_fraction = 2/3 ∧
  pages_read = finished_fraction * total_pages ∧
  pages_left = total_pages - pages_read →
  pages_more = pages_read - pages_left →
  pages_more = 90 := 
by
  intro h
  sorry

end sophia_read_more_pages_l271_27199


namespace min_expr_l271_27148

theorem min_expr (a b c d : ℕ) (ha : Odd a) (hb : Odd b) (hc : Odd c) (hd : Odd d) (a_pos: 0 < a) (b_pos: 0 < b) (c_pos: 0 < c) (d_pos: 0 < d)
(h_distinct: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  2 * a * b * c * d - (a * b * c + a * b * d + a * c * d + b * c * d) = 34 := 
sorry

end min_expr_l271_27148


namespace regular_seven_gon_l271_27126

theorem regular_seven_gon 
    (A : Fin 7 → ℝ × ℝ)
    (cong_diagonals_1 : ∀ (i : Fin 7), dist (A i) (A ((i + 2) % 7)) = dist (A 0) (A 2))
    (cong_diagonals_2 : ∀ (i : Fin 7), dist (A i) (A ((i + 3) % 7)) = dist (A 0) (A 3))
    : ∀ (i j : Fin 7), dist (A i) (A ((i + 1) % 7)) = dist (A j) (A ((j + 1) % 7)) :=
by sorry

end regular_seven_gon_l271_27126


namespace baba_yaga_powder_problem_l271_27180

theorem baba_yaga_powder_problem (A B d : ℝ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 := 
sorry

end baba_yaga_powder_problem_l271_27180


namespace sum_fifth_to_seventh_terms_arith_seq_l271_27158

theorem sum_fifth_to_seventh_terms_arith_seq (a d : ℤ)
  (h1 : a + 7 * d = 16) (h2 : a + 8 * d = 22) (h3 : a + 9 * d = 28) :
  (a + 4 * d) + (a + 5 * d) + (a + 6 * d) = 12 :=
by
  sorry

end sum_fifth_to_seventh_terms_arith_seq_l271_27158


namespace ratio_of_students_to_dishes_l271_27149

theorem ratio_of_students_to_dishes (m n : ℕ) 
  (h_students : n > 0)
  (h_dishes : ∃ dishes : Finset ℕ, dishes.card = 100)
  (h_each_student_tastes_10 : ∀ student : Finset ℕ, student.card = 10) 
  (h_pairs_taste_by_m_students : ∀ {d1 d2 : ℕ} (hd1 : d1 ∈ Finset.range 100) (hd2 : d2 ∈ Finset.range 100), m = 10) 
  : n / m = 110 := by
  sorry

end ratio_of_students_to_dishes_l271_27149


namespace price_per_working_game_eq_six_l271_27142

-- Define the total number of video games
def total_games : Nat := 10

-- Define the number of non-working video games
def non_working_games : Nat := 8

-- Define the total income from selling working games
def total_earning : Nat := 12

-- Calculate the number of working video games
def working_games : Nat := total_games - non_working_games

-- Define the expected price per working game
def expected_price_per_game : Nat := 6

-- Theorem statement: Prove that the price per working game is $6
theorem price_per_working_game_eq_six :
  total_earning / working_games = expected_price_per_game :=
by sorry

end price_per_working_game_eq_six_l271_27142


namespace square_area_rational_l271_27151

-- Define the condition: the side length of the square is a rational number.
def is_rational (x : ℚ) : Prop := true

-- Define the theorem to be proved: If the side length of a square is rational, then its area is rational.
theorem square_area_rational (s : ℚ) (h : is_rational s) : is_rational (s * s) := 
sorry

end square_area_rational_l271_27151


namespace shaded_area_in_octagon_l271_27163

theorem shaded_area_in_octagon (s r : ℝ) (h_s : s = 4) (h_r : r = s / 2) :
  let area_octagon := 2 * (1 + Real.sqrt 2) * s^2
  let area_semicircles := 8 * (π * r^2 / 2)
  area_octagon - area_semicircles = 32 * (1 + Real.sqrt 2) - 16 * π := by
  sorry

end shaded_area_in_octagon_l271_27163


namespace solution_exists_l271_27177

noncomputable def equation (x : ℝ) := 
  (x^2 - 5 * x + 4) / (x - 1) + (2 * x^2 + 7 * x - 4) / (2 * x - 1)

theorem solution_exists : equation 2 = 4 := by
  sorry

end solution_exists_l271_27177


namespace percentage_of_fair_haired_employees_who_are_women_l271_27122

variable (E : ℝ) -- Total number of employees
variable (h1 : 0.1 * E = women_with_fair_hair_E) -- 10% of employees are women with fair hair
variable (h2 : 0.25 * E = fair_haired_employees_E) -- 25% of employees have fair hair

theorem percentage_of_fair_haired_employees_who_are_women :
  (women_with_fair_hair_E / fair_haired_employees_E) * 100 = 40 :=
by
  sorry

end percentage_of_fair_haired_employees_who_are_women_l271_27122


namespace seat_arrangement_l271_27173

theorem seat_arrangement :
  ∃ (arrangement : Fin 7 → String), 
  (arrangement 6 = "Diane") ∧
  (∃ (i j : Fin 7), i < j ∧ arrangement i = "Carla" ∧ arrangement j = "Adam" ∧ j = (i + 1)) ∧
  (∃ (i j k : Fin 7), i < j ∧ j < k ∧ arrangement i = "Brian" ∧ arrangement j = "Ellie" ∧ (k - i) ≥ 3) ∧
  arrangement 3 = "Carla" := 
sorry

end seat_arrangement_l271_27173


namespace common_root_exists_l271_27157

theorem common_root_exists :
  ∃ x, (3 * x^4 + 13 * x^3 + 20 * x^2 + 17 * x + 7 = 0) ∧ (3 * x^4 + x^3 - 8 * x^2 + 11 * x - 7 = 0) → x = -7 / 3 := 
by
  sorry

end common_root_exists_l271_27157


namespace fraction_subtraction_l271_27106

theorem fraction_subtraction (x : ℝ) : (8000 * x - (0.05 / 100 * 8000) = 796) → x = 0.1 :=
by
  sorry

end fraction_subtraction_l271_27106


namespace prime_p_equals_2_l271_27178

theorem prime_p_equals_2 (p q r s : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hs: Nat.Prime s)
  (h_sum : p + q + r = 2 * s) (h_order : 1 < p ∧ p < q ∧ q < r) : p = 2 :=
sorry

end prime_p_equals_2_l271_27178


namespace exists_i_for_inequality_l271_27154

theorem exists_i_for_inequality (n : ℕ) (x : ℕ → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ x i ∧ x i ≤ 1) :
  ∃ i, 1 ≤ i ∧ i ≤ n - 1 ∧ x i * (1 - x (i + 1)) ≥ (1 / 4) * x 1 * (1 - x n) :=
by
  sorry

end exists_i_for_inequality_l271_27154


namespace sara_caught_five_trout_l271_27170

theorem sara_caught_five_trout (S M : ℕ) (h1 : M = 2 * S) (h2 : M = 10) : S = 5 :=
by
  sorry

end sara_caught_five_trout_l271_27170


namespace perimeter_of_triangle_LMN_l271_27125

variable (K L M N : Type)
variables [MetricSpace K]
variables [MetricSpace L]
variables [MetricSpace M]
variables [MetricSpace N]
variables (KL LN MN : ℝ)
variables (perimeter_LMN : ℝ)

-- Given conditions
axiom KL_eq_24 : KL = 24
axiom LN_eq_24 : LN = 24
axiom MN_eq_9  : MN = 9

-- Prove the perimeter is 57
theorem perimeter_of_triangle_LMN : perimeter_LMN = KL + LN + MN → perimeter_LMN = 57 :=
by sorry

end perimeter_of_triangle_LMN_l271_27125


namespace binomial_12_11_eq_12_l271_27189

-- Define binomial coefficient function
def binomial (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement to prove
theorem binomial_12_11_eq_12 :
  binomial 12 11 = 12 := by
  sorry

end binomial_12_11_eq_12_l271_27189


namespace domain_sqrt_function_l271_27121

noncomputable def quadratic_nonneg_for_all_x (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0

theorem domain_sqrt_function (a : ℝ) :
  quadratic_nonneg_for_all_x a ↔ (0 ≤ a ∧ a ≤ 4) :=
by sorry

end domain_sqrt_function_l271_27121


namespace jenny_coins_value_l271_27100

theorem jenny_coins_value (n d : ℕ) (h1 : d = 30 - n) (h2 : 150 + 5 * n = 300 - 5 * n + 120) :
  (300 - 5 * n : ℚ) / 100 = 1.65 := 
by
  sorry

end jenny_coins_value_l271_27100
