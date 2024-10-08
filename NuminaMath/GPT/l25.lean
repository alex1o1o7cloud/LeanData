import Mathlib

namespace power_function_at_100_l25_25843

-- Given a power function f(x) = x^α that passes through the point (9, 3),
-- show that f(100) = 10.

theorem power_function_at_100 (α : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = x ^ α)
  (h2 : f 9 = 3) : f 100 = 10 :=
sorry

end power_function_at_100_l25_25843


namespace treasures_on_island_l25_25024

-- Define the propositions P and K
def P : Prop := ∃ p : Prop, p
def K : Prop := ∃ k : Prop, k

-- Define the claim by A
def A_claim : Prop := K ↔ P

-- Theorem statement as specified part (b)
theorem treasures_on_island (A_is_knight_or_liar : (A_claim ↔ true) ∨ (A_claim ↔ false)) : ∃ P, P :=
by
  sorry

end treasures_on_island_l25_25024


namespace remainder_four_times_plus_six_l25_25263

theorem remainder_four_times_plus_six (n : ℤ) (h : n % 5 = 3) : (4 * n + 6) % 5 = 3 :=
by
  sorry

end remainder_four_times_plus_six_l25_25263


namespace volume_eq_three_times_other_two_l25_25367

-- declare the given ratio of the radii
def r1 : ℝ := 1
def r2 : ℝ := 2
def r3 : ℝ := 3

-- calculate the volumes based on the given radii
noncomputable def V (r : ℝ) : ℝ := (4/3) * Real.pi * r^3

-- defining the volumes of the three spheres
noncomputable def V1 : ℝ := V r1
noncomputable def V2 : ℝ := V r2
noncomputable def V3 : ℝ := V r3

theorem volume_eq_three_times_other_two : V3 = 3 * (V1 + V2) := 
by
  sorry

end volume_eq_three_times_other_two_l25_25367


namespace volume_of_polyhedron_l25_25587

theorem volume_of_polyhedron (s : ℝ) : 
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  volume = (Real.sqrt 3 / 2) * s^3 :=
by
  let base_area := (3 * Real.sqrt 3 / 2) * s^2
  let height := s
  let volume := (1 / 3) * base_area * height
  show volume = (Real.sqrt 3 / 2) * s^3
  sorry

end volume_of_polyhedron_l25_25587


namespace smaller_angle_measure_l25_25679

theorem smaller_angle_measure (x : ℝ) (h1 : 3 * x + 2 * x = 90) : 2 * x = 36 :=
by {
  sorry
}

end smaller_angle_measure_l25_25679


namespace oprah_winfrey_band_weights_l25_25733

theorem oprah_winfrey_band_weights :
  let weight_trombone := 10
  let weight_tuba := 20
  let weight_drum := 15
  let num_trumpets := 6
  let num_clarinets := 9
  let num_trombones := 8
  let num_tubas := 3
  let num_drummers := 2
  let total_weight := 245

  15 * x = total_weight - (num_trombones * weight_trombone + num_tubas * weight_tuba + num_drummers * weight_drum) 
  → x = 5 := by
  sorry

end oprah_winfrey_band_weights_l25_25733


namespace exists_equal_mod_p_l25_25370

theorem exists_equal_mod_p (p : ℕ) [hp_prime : Fact p.Prime] 
  (m : Fin p → ℕ) 
  (h_consecutive : ∀ i j : Fin p, (i : ℕ) < j → m i + 1 = m j) 
  (sigma : Equiv (Fin p) (Fin p)) :
  ∃ (k l : Fin p), k ≠ l ∧ (m k * m (sigma k) - m l * m (sigma l)) % p = 0 :=
by
  sorry

end exists_equal_mod_p_l25_25370


namespace correct_choice_l25_25886

-- Define the structures and options
inductive Structure
| Sequential
| Conditional
| Loop
| Module

def option_A : List Structure :=
  [Structure.Sequential, Structure.Module, Structure.Conditional]

def option_B : List Structure :=
  [Structure.Sequential, Structure.Loop, Structure.Module]

def option_C : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

def option_D : List Structure :=
  [Structure.Module, Structure.Conditional, Structure.Loop]

-- Define the correct structures
def basic_structures : List Structure :=
  [Structure.Sequential, Structure.Conditional, Structure.Loop]

-- The theorem statement
theorem correct_choice : option_C = basic_structures :=
  by
    sorry  -- Proof would go here

end correct_choice_l25_25886


namespace triangle_area_eq_l25_25774

/-- In a triangle ABC, given that A = arccos(7/8), BC = a, and the altitude from vertex A 
     is equal to the sum of the other two altitudes, show that the area of triangle ABC 
     is (a^2 * sqrt(15)) / 4. -/
theorem triangle_area_eq (a : ℝ) (angle_A : ℝ) (h_angle : angle_A = Real.arccos (7/8))
    (BC : ℝ) (h_BC : BC = a) (H : ∀ (AC AB altitude_A altitude_C altitude_B : ℝ),
    AC = X → AB = Y → 
    altitude_A = (altitude_C + altitude_B) → 
    ∃ (S : ℝ), 
    S = (1/2) * X * Y * Real.sin angle_A ∧ 
    altitude_A = (2 * S / X) + (2 * S / Y) 
    → (X * Y) = 4 * (a^2) 
    → S = ((a^2 * Real.sqrt 15) / 4)) :
S = (a^2 * Real.sqrt 15) / 4 := sorry

end triangle_area_eq_l25_25774


namespace inequality_solution_set_l25_25696

theorem inequality_solution_set (x : ℝ) : 4 * x^2 - 4 * x + 1 ≥ 0 := 
by
  sorry

end inequality_solution_set_l25_25696


namespace infinite_non_prime_seq_l25_25702

-- Let's state the theorem in Lean
theorem infinite_non_prime_seq (k : ℕ) : 
  ∃ᶠ n in at_top, ∀ i : ℕ, (1 ≤ i ∧ i ≤ k) → ¬ Nat.Prime (n + i) := 
sorry

end infinite_non_prime_seq_l25_25702


namespace math_problem_solution_l25_25113

open Real

noncomputable def math_problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a + b + c + d = 4) : Prop :=
  (b / sqrt (a + 2 * c) + c / sqrt (b + 2 * d) + d / sqrt (c + 2 * a) + a / sqrt (d + 2 * b)) ≥ (4 * sqrt 3) / 3

theorem math_problem_solution (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 4) :
  math_problem a b c d ha hb hc hd h := by sorry

end math_problem_solution_l25_25113


namespace find_k_in_geometric_sequence_l25_25087

theorem find_k_in_geometric_sequence (a : ℕ → ℕ) (k : ℕ)
  (h1 : ∀ n, a n = a 2 * 3^(n-2))
  (h2 : a 2 = 3)
  (h3 : a 3 = 9)
  (h4 : a k = 243) :
  k = 6 :=
sorry

end find_k_in_geometric_sequence_l25_25087


namespace solve_x_sq_plus_y_sq_l25_25013

theorem solve_x_sq_plus_y_sq (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = 2) : x^2 + y^2 = 5 :=
by
  sorry

end solve_x_sq_plus_y_sq_l25_25013


namespace find_Q_plus_R_l25_25517

-- P, Q, R must be digits in base 8 (distinct and non-zero)
def is_valid_digit (d : Nat) : Prop :=
  d > 0 ∧ d < 8

def digits_distinct (P Q R : Nat) : Prop :=
  P ≠ Q ∧ Q ≠ R ∧ R ≠ P

-- Define the base 8 number from its digits
def base8_number (P Q R : Nat) : Nat :=
  8^2 * P + 8 * Q + R

-- Define the given condition
def condition (P Q R : Nat) : Prop :=
  is_valid_digit P ∧ is_valid_digit Q ∧ is_valid_digit R ∧ digits_distinct P Q R ∧ 
  (base8_number P Q R + base8_number Q R P + base8_number R P Q = 8^3 * P + 8^2 * P + 8 * P + 8)

-- The result: Q + R in base 8 is 10_8 which is 8 + 2 (in decimal is 10)
theorem find_Q_plus_R (P Q R : Nat) (h : condition P Q R) : Q + R = 8 + 2 :=
sorry

end find_Q_plus_R_l25_25517


namespace cyclist_distance_l25_25798

theorem cyclist_distance
  (v t d : ℝ)
  (h1 : d = v * t)
  (h2 : d = (v + 1) * (t - 0.5))
  (h3 : d = (v - 1) * (t + 1)) :
  d = 6 :=
by
  sorry

end cyclist_distance_l25_25798


namespace inequality_iff_l25_25099

theorem inequality_iff (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : (a > b) ↔ (1/a < 1/b) = false :=
by
  sorry

end inequality_iff_l25_25099


namespace fish_farm_estimated_mass_l25_25669

noncomputable def total_fish_mass_in_pond 
  (initial_fry: ℕ) 
  (survival_rate: ℝ) 
  (haul1_count: ℕ) (haul1_avg_weight: ℝ) 
  (haul2_count: ℕ) (haul2_avg_weight: ℝ) 
  (haul3_count: ℕ) (haul3_avg_weight: ℝ) : ℝ :=
  let surviving_fish := initial_fry * survival_rate
  let total_mass_haul1 := haul1_count * haul1_avg_weight
  let total_mass_haul2 := haul2_count * haul2_avg_weight
  let total_mass_haul3 := haul3_count * haul3_avg_weight
  let average_weight_per_fish := (total_mass_haul1 + total_mass_haul2 + total_mass_haul3) / (haul1_count + haul2_count + haul3_count)
  average_weight_per_fish * surviving_fish

theorem fish_farm_estimated_mass :
  total_fish_mass_in_pond 
    80000           -- initial fry
    0.95            -- survival rate
    40 2.5          -- first haul: 40 fish, 2.5 kg each
    25 2.2          -- second haul: 25 fish, 2.2 kg each
    35 2.8          -- third haul: 35 fish, 2.8 kg each
    = 192280 := by
  sorry

end fish_farm_estimated_mass_l25_25669


namespace bag_contains_twenty_cookies_l25_25796

noncomputable def cookies_in_bag 
  (total_calories : ℕ) 
  (calories_per_cookie : ℕ)
  (bags_in_box : ℕ)
  : ℕ :=
  total_calories / (calories_per_cookie * bags_in_box)

theorem bag_contains_twenty_cookies 
  (H1 : total_calories = 1600) 
  (H2 : calories_per_cookie = 20) 
  (H3 : bags_in_box = 4)
  : cookies_in_bag total_calories calories_per_cookie bags_in_box = 20 := 
by
  have h1 : total_calories = 1600 := H1
  have h2 : calories_per_cookie = 20 := H2
  have h3 : bags_in_box = 4 := H3
  sorry

end bag_contains_twenty_cookies_l25_25796


namespace increase_by_percentage_l25_25397

theorem increase_by_percentage (a b : ℝ) (percentage : ℝ) (final : ℝ) : b = a * percentage → final = a + b → final = 437.5 :=
by
  sorry

end increase_by_percentage_l25_25397


namespace single_equivalent_discount_l25_25022

theorem single_equivalent_discount :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  ∃ (k : ℝ), (1 - k) = (1 - discount1) * (1 - discount2) * (1 - discount3) ∧ k = 0.27325 :=
by
  sorry

end single_equivalent_discount_l25_25022


namespace distinct_values_of_expr_l25_25571

theorem distinct_values_of_expr : 
  let a := 3^(3^(3^3));
  let b := 3^((3^3)^3);
  let c := ((3^3)^3)^3;
  let d := (3^(3^3))^3;
  let e := (3^3)^(3^3);
  (a ≠ b) ∧ (c ≠ b) ∧ (d ≠ b) ∧ (d ≠ a) ∧ (e ≠ a) ∧ (e ≠ b) ∧ (e ≠ d) := sorry

end distinct_values_of_expr_l25_25571


namespace proof_problem_l25_25753

def x : ℝ := 0.80 * 1750
def y : ℝ := 0.35 * 3000
def z : ℝ := 0.60 * 4500
def w : ℝ := 0.40 * 2800
def a : ℝ := z * w
def b : ℝ := x + y

theorem proof_problem : a - b = 3021550 := by
  sorry

end proof_problem_l25_25753


namespace maximum_pencils_l25_25919

-- Define the problem conditions
def red_pencil_cost := 27
def blue_pencil_cost := 23
def max_total_cost := 940
def max_diff := 10

-- Define the main theorem
theorem maximum_pencils (x y : ℕ) 
  (h1 : red_pencil_cost * x + blue_pencil_cost * y ≤ max_total_cost)
  (h2 : y - x ≤ max_diff)
  (hx_min : ∀ z : ℕ, z < x → red_pencil_cost * z + blue_pencil_cost * (z + max_diff) > max_total_cost):
  x = 14 ∧ y = 24 ∧ x + y = 38 := 
  sorry

end maximum_pencils_l25_25919


namespace grapes_purchased_l25_25044

-- Define the given conditions
def price_per_kg_grapes : ℕ := 68
def kg_mangoes : ℕ := 9
def price_per_kg_mangoes : ℕ := 48
def total_paid : ℕ := 908

-- Define the proof problem
theorem grapes_purchased : ∃ (G : ℕ), (price_per_kg_grapes * G + price_per_kg_mangoes * kg_mangoes = total_paid) ∧ (G = 7) :=
by {
  use 7,
  sorry
}

end grapes_purchased_l25_25044


namespace simplify_expression_1_simplify_expression_2_l25_25570

section Problem1
variables (a b c : ℝ) (h1 : c ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem simplify_expression_1 :
  ((a^2 * b / (-c))^3 * (c^2 / (- (a * b)))^2 / (b * c / a)^4)
  = - (a^10 / (b^3 * c^7)) :=
by sorry
end Problem1

section Problem2
variables (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : b ≠ 0)

theorem simplify_expression_2 :
  ((2 / (a^2 - b^2) - 1 / (a^2 - a * b)) / (a / (a + b))) = 1 / a^2 :=
by sorry
end Problem2

end simplify_expression_1_simplify_expression_2_l25_25570


namespace range_of_m_l25_25999

-- Definitions of Propositions p and q
def Proposition_p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m > 0) ∧ (1 > 0)  -- where x₁ + x₂ = -m > 0 and x₁x₂ = 1

def Proposition_q (m : ℝ) : Prop :=
  16 * (m + 2)^2 - 16 < 0  -- discriminant of 4x^2 + 4(m+2)x + 1 = 0 is less than 0

-- Given: "Proposition p or Proposition q" is true
def given (m : ℝ) : Prop :=
  Proposition_p m ∨ Proposition_q m

-- Prove: Range of values for m is (-∞, -1)
theorem range_of_m (m : ℝ) (h : given m) : m < -1 :=
sorry

end range_of_m_l25_25999


namespace volume_of_rectangular_parallelepiped_l25_25948

theorem volume_of_rectangular_parallelepiped (x y z p q r : ℝ) 
  (h1 : p = x * y) 
  (h2 : q = x * z) 
  (h3 : r = y * z) : 
  x * y * z = Real.sqrt (p * q * r) :=
by
  sorry

end volume_of_rectangular_parallelepiped_l25_25948


namespace compare_compound_interest_l25_25380

noncomputable def compound_annually (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ t

noncomputable def compound_monthly (P : ℝ) (r : ℝ) (t : ℕ) := 
  P * (1 + r) ^ (12 * t)

theorem compare_compound_interest :
  let P := 1000
  let r_annual := 0.03
  let r_monthly := 0.0025
  let t := 5
  compound_monthly P r_monthly t > compound_annually P r_annual t :=
by
  sorry

end compare_compound_interest_l25_25380


namespace count_distinct_even_numbers_l25_25575

theorem count_distinct_even_numbers : 
  ∃ c, c = 37 ∧ ∀ d1 d2 d3, d1 ≠ d2 → d2 ≠ d3 → d1 ≠ d3 → (d1 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d2 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (d3 ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ)) → (∃ n : ℕ, n / 10 ^ 2 = d1 ∧ (n / 10) % 10 = d2 ∧ n % 10 = d3 ∧ n % 2 = 0) :=
sorry

end count_distinct_even_numbers_l25_25575


namespace jane_ends_with_crayons_l25_25213

-- Definitions for the conditions in the problem
def initial_crayons : Nat := 87
def crayons_eaten : Nat := 7
def packs_bought : Nat := 5
def crayons_per_pack : Nat := 10
def crayons_break : Nat := 3

-- Statement to prove: Jane ends with 127 crayons
theorem jane_ends_with_crayons :
  initial_crayons - crayons_eaten + (packs_bought * crayons_per_pack) - crayons_break = 127 :=
by
  sorry

end jane_ends_with_crayons_l25_25213


namespace find_k_l25_25365

/--
Given a system of linear equations:
1) x + 2 * y = -a + 1
2) x - 3 * y = 4 * a + 6
If the expression k * x - y remains unchanged regardless of the value of the constant a, 
show that k = -1.
-/
theorem find_k 
  (a x y k : ℝ) 
  (h1 : x + 2 * y = -a + 1) 
  (h2 : x - 3 * y = 4 * a + 6)
  (h3 : ∀ a₁ a₂ x₁ x₂ y₁ y₂, (x₁ + 2 * y₁ = -a₁ + 1) → (x₁ - 3 * y₁ = 4 * a₁ + 6) → 
                               (x₂ + 2 * y₂ = -a₂ + 1) → (x₂ - 3 * y₂ = 4 * a₂ + 6) → 
                               (k * x₁ - y₁ = k * x₂ - y₂)) : 
  k = -1 :=
  sorry

end find_k_l25_25365


namespace factor_theorem_l25_25613

-- Define the polynomial function f(x)
def f (k : ℚ) (x : ℚ) : ℚ := k * x^3 + 27 * x^2 - k * x + 55

-- State the theorem to find the value of k such that x+5 is a factor of f(x)
theorem factor_theorem (k : ℚ) : f k (-5) = 0 ↔ k = 73 / 12 :=
by sorry

end factor_theorem_l25_25613


namespace average_speed_train_l25_25977

theorem average_speed_train (d1 d2 : ℝ) (t1 t2 : ℝ) 
  (h_d1 : d1 = 325) (h_d2 : d2 = 470)
  (h_t1 : t1 = 3.5) (h_t2 : t2 = 4) :
  (d1 + d2) / (t1 + t2) = 106 :=
by
  sorry

end average_speed_train_l25_25977


namespace determine_f_4_l25_25046

theorem determine_f_4 (f g : ℝ → ℝ)
  (h1 : ∀ x y z : ℝ, f (x^2 + y * f z) = x * g x + z * g y)
  (h2 : ∀ x : ℝ, g x = 2 * x) :
  f 4 = 32 :=
sorry

end determine_f_4_l25_25046


namespace complement_intersection_l25_25927

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 2}

-- Define the set B
def B : Set ℕ := {2, 3, 4}

-- Statement to be proven
theorem complement_intersection :
  (U \ A) ∩ B = {3, 4} :=
sorry

end complement_intersection_l25_25927


namespace pencils_in_each_box_l25_25672

theorem pencils_in_each_box (total_pencils : ℕ) (total_boxes : ℕ) (pencils_per_box : ℕ) 
  (h1 : total_pencils = 648) (h2 : total_boxes = 162) : 
  total_pencils / total_boxes = pencils_per_box := 
by
  sorry

end pencils_in_each_box_l25_25672


namespace carol_spending_l25_25073

noncomputable def savings (S : ℝ) : Prop :=
∃ (X : ℝ) (stereo_spending television_spending : ℝ), 
  stereo_spending = (1 / 4) * S ∧
  television_spending = X * S ∧
  stereo_spending + television_spending = 0.25 * S ∧
  (stereo_spending - television_spending) / S = 0.25

theorem carol_spending (S : ℝ) : savings S :=
sorry

end carol_spending_l25_25073


namespace lunch_break_duration_l25_25443

-- Definitions based on the conditions
variables (p h1 h2 L : ℝ)
-- Monday equation
def monday_eq : Prop := (9 - L/60) * (p + h1 + h2) = 0.55
-- Tuesday equation
def tuesday_eq : Prop := (7 - L/60) * (p + h2) = 0.35
-- Wednesday equation
def wednesday_eq : Prop := (5 - L/60) * (p + h1 + h2) = 0.25
-- Thursday equation
def thursday_eq : Prop := (4 - L/60) * p = 0.15

-- Combine all conditions
def all_conditions : Prop :=
  monday_eq p h1 h2 L ∧ tuesday_eq p h2 L ∧ wednesday_eq p h1 h2 L ∧ thursday_eq p L

-- Proof that the lunch break duration is 60 minutes
theorem lunch_break_duration : all_conditions p h1 h2 L → L = 60 :=
by
  sorry

end lunch_break_duration_l25_25443


namespace arithmetic_sequence_a_100_l25_25736

theorem arithmetic_sequence_a_100 :
  ∀ (a : ℕ → ℕ), 
  (a 1 = 100) → 
  (∀ n : ℕ, a (n + 1) = a n + 2) → 
  a 100 = 298 :=
by
  intros a h1 hrec
  sorry

end arithmetic_sequence_a_100_l25_25736


namespace inscribed_square_neq_five_l25_25734

theorem inscribed_square_neq_five (a b : ℝ) 
  (h1 : a - b = 1)
  (h2 : a * b = 1)
  (h3 : a + b = Real.sqrt 5) : a^2 + b^2 ≠ 5 :=
by sorry

end inscribed_square_neq_five_l25_25734


namespace radius_of_circle_B_l25_25750

-- Definitions of circles and their properties
noncomputable def circle_tangent_externally (r1 r2 : ℝ) := ∃ d : ℝ, d = r1 + r2
noncomputable def circle_tangent_internally (r1 r2 : ℝ) := ∃ d : ℝ, d = r2 - r1

-- Problem statement in Lean 4
theorem radius_of_circle_B
  (rA rB rC rD centerA centerB centerC centerD : ℝ)
  (h_rA : rA = 2)
  (h_congruent_B_C : rB = rC)
  (h_circle_A_tangent_to_B : circle_tangent_externally rA rB)
  (h_circle_A_tangent_to_C : circle_tangent_externally rA rC)
  (h_circle_B_C_tangent_e : circle_tangent_externally rB rC)
  (h_circle_B_D_tangent_i : circle_tangent_internally rB rD)
  (h_center_A_passes_D : centerA = centerD)
  (h_rD : rD = 4) : 
  rB = 1 := sorry

end radius_of_circle_B_l25_25750


namespace tree_break_height_l25_25542

-- Define the problem conditions and prove the required height h
theorem tree_break_height (height_tree : ℝ) (distance_shore : ℝ) (height_break : ℝ) : 
  height_tree = 20 → distance_shore = 6 → 
  (distance_shore ^ 2 + height_break ^ 2 = (height_tree - height_break) ^ 2) →
  height_break = 9.1 :=
by
  intros h_tree_eq h_shore_eq hyp_eq
  have h_tree_20 := h_tree_eq
  have h_shore_6 := h_shore_eq
  have hyp := hyp_eq
  sorry -- Proof of the theorem is omitted

end tree_break_height_l25_25542


namespace ratio_of_areas_l25_25372

-- Define the conditions
variable (s : ℝ) (h_pos : s > 0)
-- The total perimeter of four small square pens is reused for one large square pen
def total_fencing_length := 16 * s
def large_square_side_length := 4 * s

-- Define the areas
def small_squares_total_area := 4 * s^2
def large_square_area := (4 * s)^2

-- The statement to prove
theorem ratio_of_areas : small_squares_total_area / large_square_area = 1 / 4 :=
by
  sorry

end ratio_of_areas_l25_25372


namespace problem_statement_l25_25212

def sequence_arithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (a (n+1) / 2^(n+1) - a n / 2^n = 1)

theorem problem_statement : 
  ∃ a : ℕ → ℝ, a 1 = 2 ∧ a 2 = 8 ∧ (∀ n : ℕ, n ≥ 1 → a (n+1) - 2 * a n = 2^(n+1)) → sequence_arithmetic a :=
by
  sorry

end problem_statement_l25_25212


namespace f_2017_equals_neg_one_fourth_l25_25085

noncomputable def f : ℝ → ℝ := sorry -- Original definition will be derived from the conditions

axiom symmetry_about_y_axis : ∀ (x : ℝ), f (-x) = f x
axiom periodicity : ∀ (x : ℝ), f (x + 3) = -f x
axiom specific_interval : ∀ (x : ℝ), (3/2 < x ∧ x < 5/2) → f x = (1/2)^x

theorem f_2017_equals_neg_one_fourth : f 2017 = -1/4 :=
by sorry

end f_2017_equals_neg_one_fourth_l25_25085


namespace percent_increase_salary_l25_25695

theorem percent_increase_salary (new_salary increase : ℝ) (h_new_salary : new_salary = 90000) (h_increase : increase = 25000) :
  (increase / (new_salary - increase)) * 100 = 38.46 := by
  -- Given values
  have h1 : new_salary = 90000 := h_new_salary
  have h2 : increase = 25000 := h_increase
  -- Compute original salary
  let original_salary : ℝ := new_salary - increase
  -- Compute percent increase
  let percent_increase : ℝ := (increase / original_salary) * 100
  -- Show that the percent increase is 38.46
  have h3 : percent_increase = 38.46 := sorry
  exact h3

end percent_increase_salary_l25_25695


namespace count_valid_n_l25_25762

theorem count_valid_n (n : ℕ) (h₁ : (n % 2015) ≠ 0) :
  (n^3 + 3^n) % 5 = 0 :=
by
  sorry

end count_valid_n_l25_25762


namespace select_best_athlete_l25_25998

theorem select_best_athlete
  (avg_A avg_B avg_C avg_D: ℝ)
  (var_A var_B var_C var_D: ℝ)
  (h_avg_A: avg_A = 185)
  (h_avg_B: avg_B = 180)
  (h_avg_C: avg_C = 185)
  (h_avg_D: avg_D = 180)
  (h_var_A: var_A = 3.6)
  (h_var_B: var_B = 3.6)
  (h_var_C: var_C = 7.4)
  (h_var_D: var_D = 8.1) :
  (avg_A > avg_B ∧ avg_A > avg_D ∧ var_A < var_C) →
  (avg_A = 185 ∧ var_A = 3.6) :=
by
  sorry

end select_best_athlete_l25_25998


namespace red_marbles_in_bag_l25_25231

theorem red_marbles_in_bag (T R : ℕ) (hT : T = 84)
    (probability_not_red : ((T - R : ℚ) / T)^2 = 36 / 49) : 
    R = 12 := 
sorry

end red_marbles_in_bag_l25_25231


namespace min_value_xyz_l25_25972

theorem min_value_xyz (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_prod : x * y * z = 8) : 
  x + 3 * y + 6 * z ≥ 18 :=
sorry

end min_value_xyz_l25_25972


namespace find_range_a_l25_25488

def bounded_a (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≤ 2 → a * (4 ^ x) + 2 ^ x + 1 ≥ 0

theorem find_range_a :
  ∃ (a : ℝ), bounded_a a ↔ a ≥ -5 / 16 :=
sorry

end find_range_a_l25_25488


namespace battery_usage_minutes_l25_25889

theorem battery_usage_minutes (initial_battery final_battery : ℝ) (initial_minutes : ℝ) (rate_of_usage : ℝ) :
  initial_battery - final_battery = rate_of_usage * initial_minutes →
  initial_battery = 100 →
  final_battery = 68 →
  initial_minutes = 60 →
  rate_of_usage = 8 / 15 →
  ∃ additional_minutes : ℝ, additional_minutes = 127.5 :=
by
  intros
  sorry

end battery_usage_minutes_l25_25889


namespace luke_points_per_round_l25_25608

-- Define the total number of points scored 
def totalPoints : ℕ := 8142

-- Define the number of rounds played
def rounds : ℕ := 177

-- Define the points gained per round which we need to prove
def pointsPerRound : ℕ := 46

-- Now, we can state: if Luke played 177 rounds and scored a total of 8142 points, then he gained 46 points per round
theorem luke_points_per_round :
  (totalPoints = 8142) → (rounds = 177) → (totalPoints / rounds = pointsPerRound) := by
  sorry

end luke_points_per_round_l25_25608


namespace point_inside_circle_implies_range_l25_25925

theorem point_inside_circle_implies_range (a : ℝ) : 
  (1 - a)^2 + (1 + a)^2 < 4 → -1 < a ∧ a < 1 :=
by
  intro h
  sorry

end point_inside_circle_implies_range_l25_25925


namespace socks_ratio_l25_25256

/-- Alice ordered 6 pairs of green socks and some additional pairs of red socks. The price per pair
of green socks was three times that of the red socks. During the delivery, the quantities of the 
pairs were accidentally swapped. This mistake increased the bill by 40%. Prove that the ratio of the 
number of pairs of green socks to red socks in Alice's original order is 1:2. -/
theorem socks_ratio (r y : ℕ) (h1 : y * r ≠ 0) (h2 : 6 * 3 * y + r * y = (r * 3 * y + 6 * y) * 10 / 7) :
  6 / r = 1 / 2 :=
by
  sorry

end socks_ratio_l25_25256


namespace find_g_minus_6_l25_25787

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom cond1 : g 1 - 1 > 0
axiom cond2 : ∀ x y : ℤ, g x * g y + x + y + x * y = g (x + y) + x * g y + y * g x
axiom cond3 : ∀ x : ℤ, 3 * g (x + 1) = g x + 2 * x + 3

-- The proof we want to make (assertion)
theorem find_g_minus_6 : g (-6) = 723 := 
sorry

end find_g_minus_6_l25_25787


namespace max_m_x_range_l25_25202

variables {a b x : ℝ}

theorem max_m (h1 : a * b > 0) (h2 : a^2 * b = 4) : 
  a + b ≥ 3 :=
sorry

theorem x_range (h : 2 * |x - 1| + |x| ≤ 3) : 
  -1/3 ≤ x ∧ x ≤ 5/3 :=
sorry

end max_m_x_range_l25_25202


namespace find_positive_k_l25_25848

noncomputable def cubic_roots (a b k : ℝ) : Prop :=
  (3 * a * a * a + 9 * a * a - 135 * a + k = 0) ∧
  (a * a * b = -45 / 2)

theorem find_positive_k :
  ∃ (a b : ℝ), ∃ (k : ℝ) (pos : k > 0), (cubic_roots a b k) ∧ (k = 525) :=
by
  sorry

end find_positive_k_l25_25848


namespace isosceles_triangle_l25_25125

theorem isosceles_triangle (a c : ℝ) (A C : ℝ) (h : a * Real.sin A = c * Real.sin C) : a = c → Isosceles :=
sorry

end isosceles_triangle_l25_25125


namespace votes_combined_l25_25162

theorem votes_combined (vote_A vote_B : ℕ) (h_ratio : vote_A = 2 * vote_B) (h_A_votes : vote_A = 14) : vote_A + vote_B = 21 :=
by
  sorry

end votes_combined_l25_25162


namespace find_d_l25_25716

theorem find_d (d : ℝ) : (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  { sorry }

end find_d_l25_25716


namespace brendan_threw_back_l25_25227

-- Brendan's catches in the morning, throwing back x fish and catching more in the afternoon
def brendan_morning (x : ℕ) : ℕ := 8 - x
def brendan_afternoon : ℕ := 5

-- Brendan's and his dad's total catches
def brendan_total (x : ℕ) : ℕ := brendan_morning x + brendan_afternoon
def dad_total : ℕ := 13

-- Combined total fish caught by both
def total_fish (x : ℕ) : ℕ := brendan_total x + dad_total

-- The number of fish thrown back by Brendan
theorem brendan_threw_back : ∃ x : ℕ, total_fish x = 23 ∧ x = 3 :=
by
  sorry

end brendan_threw_back_l25_25227


namespace fraction_comparison_l25_25595

theorem fraction_comparison :
  let d := 0.33333333
  let f := (1 : ℚ) / 3
  f > d ∧ f - d = 1 / (3 * (10^8 : ℚ)) :=
by
  sorry

end fraction_comparison_l25_25595


namespace inequality_of_negatives_l25_25246

theorem inequality_of_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end inequality_of_negatives_l25_25246


namespace box_volume_l25_25378

theorem box_volume (x : ℕ) (h_ratio : (x > 0)) (V : ℕ) (h_volume : V = 20 * x^3) : V = 160 :=
by
  sorry

end box_volume_l25_25378


namespace least_value_of_x_l25_25968

theorem least_value_of_x 
  (x : ℕ) (p : ℕ) 
  (h1 : x > 0) 
  (h2 : Prime p) 
  (h3 : ∃ q, Prime q ∧ q % 2 = 1 ∧ x = 9 * p * q) : 
  x = 90 := 
sorry

end least_value_of_x_l25_25968


namespace division_example_l25_25092

theorem division_example :
  100 / 0.25 = 400 :=
by sorry

end division_example_l25_25092


namespace x_and_y_complete_work_in_12_days_l25_25446

noncomputable def work_rate_x : ℚ := 1 / 24
noncomputable def work_rate_y : ℚ := 1 / 24
noncomputable def combined_work_rate : ℚ := work_rate_x + work_rate_y

theorem x_and_y_complete_work_in_12_days : (1 / combined_work_rate) = 12 :=
by
  sorry

end x_and_y_complete_work_in_12_days_l25_25446


namespace first_term_geometric_sequence_l25_25567

theorem first_term_geometric_sequence (a5 a6 : ℚ) (h1 : a5 = 48) (h2 : a6 = 64) : 
  ∃ a : ℚ, a = 243 / 16 :=
by
  sorry

end first_term_geometric_sequence_l25_25567


namespace min_value_expression_l25_25944

theorem min_value_expression : 
  ∃ x : ℝ, ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) ∧ 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) = -6480.25 :=
by sorry

end min_value_expression_l25_25944


namespace eccentricity_of_ellipse_l25_25437

theorem eccentricity_of_ellipse (p q : ℕ) (hp : Nat.Coprime p q) (z : ℂ) :
  ((z - 2) * (z^2 + 3 * z + 5) * (z^2 + 5 * z + 8) = 0) →
  (∃ p q : ℕ, Nat.Coprime p q ∧ (∃ e : ℝ, e^2 = p / q ∧ p + q = 16)) :=
by
  sorry

end eccentricity_of_ellipse_l25_25437


namespace evaluate_expression_l25_25021

def diamond (a b : ℚ) : ℚ := a - (2 / b)

theorem evaluate_expression :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -(11 / 30) :=
by
  sorry

end evaluate_expression_l25_25021


namespace root_interval_l25_25360

noncomputable def f (a b x : ℝ) : ℝ := 2 * a^x - b^x

theorem root_interval (a b : ℝ) (h₀ : 0 < a) (h₁ : b ≥ 2 * a) :
  ∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a b x = 0 := 
sorry

end root_interval_l25_25360


namespace andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l25_25851

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that André wins the book is 1/4. -/
theorem andre_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let probability := (black_balls : ℚ) / total_balls
  probability = 1 / 4 := 
by 
  sorry

/-- In the first scenario, given the conditions that there are 
3 white balls and 1 black ball, and each person draws a ball 
in alphabetical order without replacement, prove that the 
probability that Dalva wins the book is 1/4. -/
theorem dalva_wins_first_scenario : 
  let total_balls := 4
  let black_balls := 1
  let andre_white := (3 / 4 : ℚ)
  let bianca_white := (2 / 3 : ℚ)
  let carlos_white := (1 / 2 : ℚ)
  let probability := andre_white * bianca_white * carlos_white * (black_balls / (total_balls - 3))
  probability = 1 / 4 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that André wins the book is 5/14. -/
theorem andre_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_first_black := (black_balls : ℚ) / total_balls
  let andre_fifth_black := (((6 / 8 : ℚ) * (5 / 7 : ℚ) * (4 / 6 : ℚ) * (3 / 5 : ℚ)) * black_balls / (total_balls - 4))
  let probability := andre_first_black + andre_fifth_black
  probability = 5 / 14 := 
by 
  sorry

/-- In the second scenario, given the conditions that there are 
6 white balls and 2 black balls, and each person draws a ball 
in alphabetical order until the first black ball is drawn, 
prove that the probability that Dalva wins the book is 1/7. -/
theorem dalva_wins_second_scenario : 
  let total_balls := 8
  let black_balls := 2
  let andre_white := (6 / 8 : ℚ)
  let bianca_white := (5 / 7 : ℚ)
  let carlos_white := (4 / 6 : ℚ)
  let dalva_black := (black_balls / (total_balls - 3))
  let probability := andre_white * bianca_white * carlos_white * dalva_black
  probability = 1 / 7 := 
by 
  sorry

end andre_wins_first_scenario_dalva_wins_first_scenario_andre_wins_second_scenario_dalva_wins_second_scenario_l25_25851


namespace rides_with_remaining_tickets_l25_25751

theorem rides_with_remaining_tickets (T_total : ℕ) (T_spent : ℕ) (C_ride : ℕ)
  (h1 : T_total = 40) (h2 : T_spent = 28) (h3 : C_ride = 4) :
  (T_total - T_spent) / C_ride = 3 := by
  sorry

end rides_with_remaining_tickets_l25_25751


namespace proof_l25_25775

noncomputable def problem_statement : Prop :=
  ( ( (Real.sqrt 1.21 * Real.sqrt 1.44) / (Real.sqrt 0.81 * Real.sqrt 0.64)
    + (Real.sqrt 1.0 * Real.sqrt 3.24) / (Real.sqrt 0.49 * Real.sqrt 2.25) ) ^ 3 
  = 44.6877470366 )

theorem proof : problem_statement := 
  by
  sorry

end proof_l25_25775


namespace geometric_series_first_term_l25_25413

theorem geometric_series_first_term (r : ℝ) (S : ℝ) (a : ℝ) (h_r : r = 1/4) (h_S : S = 80)
  (h_sum : S = a / (1 - r)) : a = 60 :=
by
  -- proof steps
  sorry

end geometric_series_first_term_l25_25413


namespace pencil_price_is_99c_l25_25755

noncomputable def one_pencil_cost (total_spent : ℝ) (notebook_price : ℝ) (notebook_count : ℕ) 
                                  (ruler_pack_price : ℝ) (eraser_price : ℝ) (eraser_count : ℕ) 
                                  (pencil_count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let notebooks_cost := notebook_count * notebook_price
  let discount_amount := discount * notebooks_cost
  let discounted_notebooks_cost := notebooks_cost - discount_amount
  let other_items_cost := ruler_pack_price + (eraser_count * eraser_price)
  let subtotal := discounted_notebooks_cost + other_items_cost
  let pencils_total_after_tax := total_spent - subtotal
  let pencils_total_before_tax := pencils_total_after_tax / (1 + tax)
  let pencil_price := pencils_total_before_tax / pencil_count
  pencil_price

theorem pencil_price_is_99c : one_pencil_cost 7.40 0.85 2 0.60 0.20 5 4 0.15 0.10 = 0.99 := 
sorry

end pencil_price_is_99c_l25_25755


namespace right_triangle_legs_l25_25810

theorem right_triangle_legs (R r : ℝ) : 
  ∃ a b : ℝ, a = Real.sqrt (2 * (R^2 + r^2)) ∧ b = Real.sqrt (2 * (R^2 - r^2)) :=
by
  sorry

end right_triangle_legs_l25_25810


namespace problem1_l25_25504

theorem problem1 (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
sorry

end problem1_l25_25504


namespace passing_marks_l25_25465

theorem passing_marks (T P : ℝ) 
  (h1 : 0.30 * T = P - 60) 
  (h2 : 0.45 * T = P + 30) : 
  P = 240 := 
by
  sorry

end passing_marks_l25_25465


namespace sally_rum_l25_25456

theorem sally_rum (x : ℕ) (h₁ : 3 * x = x + 12 + 8) : x = 10 := by
  sorry

end sally_rum_l25_25456


namespace original_people_complete_work_in_four_days_l25_25508

noncomputable def original_people_work_days (P D : ℕ) :=
  (2 * P) * 2 = (1 / 2) * (P * D)

theorem original_people_complete_work_in_four_days (P D : ℕ) (h : original_people_work_days P D) : D = 4 :=
by
  sorry

end original_people_complete_work_in_four_days_l25_25508


namespace congruence_solution_l25_25719

theorem congruence_solution (x : ℤ) (h : 5 * x + 11 ≡ 3 [ZMOD 19]) : 3 * x + 7 ≡ 6 [ZMOD 19] :=
sorry

end congruence_solution_l25_25719


namespace find_x_l25_25905

theorem find_x (x : ℝ) :
  (1 / 3) * ((2 * x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x^2 - 8 * x + 2 ↔ 
  x = (36 + Real.sqrt 2136) / 30 ∨ x = (36 - Real.sqrt 2136) / 30 := 
sorry

end find_x_l25_25905


namespace simplify_expression_l25_25686

variable (x : ℝ)

theorem simplify_expression : 2 * x - 3 * (2 - x) + 4 * (2 + x) - 5 * (1 - 3 * x) = 24 * x - 3 := 
  sorry

end simplify_expression_l25_25686


namespace problem_condition_l25_25550

noncomputable def f : ℝ → ℝ := sorry

theorem problem_condition (h: ∀ x : ℝ, f x > (deriv f) x) : 3 * f (Real.log 2) > 2 * f (Real.log 3) :=
sorry

end problem_condition_l25_25550


namespace episodes_relationship_l25_25032

variable (x y z : ℕ)

theorem episodes_relationship 
  (h1 : x * z = 50) 
  (h2 : y * z = 75) : 
  y = (3 / 2) * x ∧ z = 50 / x := 
by
  sorry

end episodes_relationship_l25_25032


namespace number_of_intersections_l25_25034

-- Conditions for the problem
def Line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 2
def Line2 (x y : ℝ) : Prop := 5 * x + 3 * y = 6
def Line3 (x y : ℝ) : Prop := x - 4 * y = 8

-- Statement to prove
theorem number_of_intersections : ∃ (p1 p2 p3 : ℝ × ℝ), 
  (Line1 p1.1 p1.2 ∧ Line2 p1.1 p1.2) ∧ 
  (Line1 p2.1 p2.2 ∧ Line3 p2.1 p2.2) ∧ 
  (Line2 p3.1 p3.2 ∧ Line3 p3.1 p3.2) ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end number_of_intersections_l25_25034


namespace find_a_l25_25254

noncomputable def A (a : ℝ) : Set ℝ :=
  {a + 2, (a + 1)^2, a^2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
  sorry

end find_a_l25_25254


namespace first_day_of_month_is_thursday_l25_25978

theorem first_day_of_month_is_thursday :
  (27 - 7 - 7 - 7 + 1) % 7 = 4 :=
by
  sorry

end first_day_of_month_is_thursday_l25_25978


namespace no_divisor_neighbors_l25_25217

def is_divisor (a b : ℕ) : Prop := b % a = 0

def circle_arrangement (arr : Fin 8 → ℕ) : Prop :=
  arr 0 = 7 ∧ arr 1 = 9 ∧ arr 2 = 4 ∧ arr 3 = 5 ∧ arr 4 = 3 ∧ arr 5 = 6 ∧ arr 6 = 8 ∧ arr 7 = 2

def valid_neighbors (arr : Fin 8 → ℕ) : Prop :=
  ¬ is_divisor (arr 0) (arr 1) ∧ ¬ is_divisor (arr 0) (arr 3) ∧
  ¬ is_divisor (arr 1) (arr 2) ∧ ¬ is_divisor (arr 1) (arr 3) ∧ ¬ is_divisor (arr 1) (arr 5) ∧
  ¬ is_divisor (arr 2) (arr 1) ∧ ¬ is_divisor (arr 2) (arr 6) ∧ ¬ is_divisor (arr 2) (arr 3) ∧
  ¬ is_divisor (arr 3) (arr 1) ∧ ¬ is_divisor (arr 3) (arr 4) ∧ ¬ is_divisor (arr 3) (arr 2) ∧ ¬ is_divisor (arr 3) (arr 0) ∧
  ¬ is_divisor (arr 4) (arr 3) ∧ ¬ is_divisor (arr 4) (arr 5) ∧
  ¬ is_divisor (arr 5) (arr 1) ∧ ¬ is_divisor (arr 5) (arr 4) ∧ ¬ is_divisor (arr 5) (arr 6) ∧
  ¬ is_divisor (arr 6) (arr 2) ∧ ¬ is_divisor (arr 6) (arr 5) ∧ ¬ is_divisor (arr 6) (arr 7) ∧
  ¬ is_divisor (arr 7) (arr 6)

theorem no_divisor_neighbors :
  ∀ (arr : Fin 8 → ℕ), circle_arrangement arr → valid_neighbors arr :=
by
  intros arr h
  sorry

end no_divisor_neighbors_l25_25217


namespace dad_use_per_brush_correct_l25_25045

def toothpaste_total : ℕ := 105
def mom_use_per_brush : ℕ := 2
def anne_brother_use_per_brush : ℕ := 1
def brushing_per_day : ℕ := 3
def days_to_finish : ℕ := 5

-- Defining the daily use function for Anne's Dad
def dad_use_per_brush (D : ℕ) : ℕ := D

theorem dad_use_per_brush_correct (D : ℕ) 
  (h : brushing_per_day * (mom_use_per_brush + anne_brother_use_per_brush * 2 + dad_use_per_brush D) * days_to_finish = toothpaste_total) 
  : dad_use_per_brush D = 3 :=
by sorry

end dad_use_per_brush_correct_l25_25045


namespace andy_diana_weight_l25_25070

theorem andy_diana_weight :
  ∀ (a b c d : ℝ),
  a + b = 300 →
  b + c = 280 →
  c + d = 310 →
  a + d = 330 := by
  intros a b c d h₁ h₂ h₃
  -- Proof goes here
  sorry

end andy_diana_weight_l25_25070


namespace reflection_across_x_axis_l25_25247

theorem reflection_across_x_axis (x y : ℝ) : (x, -y) = (-2, 3) ↔ (x, y) = (-2, -3) :=
by sorry

end reflection_across_x_axis_l25_25247


namespace number_of_regular_soda_bottles_l25_25129

-- Define the total number of bottles and the number of diet soda bottles
def total_bottles : ℕ := 30
def diet_soda_bottles : ℕ := 2

-- Define the number of regular soda bottles
def regular_soda_bottles : ℕ := total_bottles - diet_soda_bottles

-- Statement of the main proof problem
theorem number_of_regular_soda_bottles : regular_soda_bottles = 28 := by
  -- Proof goes here
  sorry

end number_of_regular_soda_bottles_l25_25129


namespace number_of_baggies_l25_25701

/-- Conditions -/
def cookies_per_bag : ℕ := 9
def chocolate_chip_cookies : ℕ := 13
def oatmeal_cookies : ℕ := 41

/-- Question: Prove the total number of baggies Olivia can make is 6 --/
theorem number_of_baggies : (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := sorry

end number_of_baggies_l25_25701


namespace part1_part2_l25_25476

-- Define what a double root equation is
def is_double_root_eq (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ * x₁ * a + x₁ * b + c = 0 ∧ x₂ = 2 * x₁ ∧ x₂ * x₂ * a + x₂ * b + c = 0

-- Statement for part 1: proving x^2 - 3x + 2 = 0 is a double root equation
theorem part1 : is_double_root_eq 1 (-3) 2 :=
sorry

-- Statement for part 2: finding correct values of a and b for ax^2 + bx - 6 = 0 to be a double root equation with one root 2
theorem part2 : (∃ a b : ℝ, is_double_root_eq a b (-6) ∧ (a = -3 ∧ b = 9) ∨ (a = -3/4 ∧ b = 9/2)) :=
sorry

end part1_part2_l25_25476


namespace correct_division_result_l25_25506

theorem correct_division_result (x : ℝ) (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by
  sorry

end correct_division_result_l25_25506


namespace analytical_expression_of_f_l25_25179

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b

theorem analytical_expression_of_f (a b : ℝ) (h_a : a > 0)
  (h_max : (∃ x_max : ℝ, f x_max a b = 5 ∧ (∀ x : ℝ, f x_max a b ≥ f x a b)))
  (h_min : (∃ x_min : ℝ, f x_min a b = 1 ∧ (∀ x : ℝ, f x_min a b ≤ f x a b))) :
  f x 3 1 = x^3 + 3 * x^2 + 1 := 
sorry

end analytical_expression_of_f_l25_25179


namespace mean_of_smallest_and_largest_is_12_l25_25777

-- Definition of the condition: the mean of five consecutive even numbers is 12.
def mean_of_five_consecutive_even_numbers_is_12 (n : ℤ) : Prop :=
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 12

-- Theorem stating that the mean of the smallest and largest of these numbers is 12.
theorem mean_of_smallest_and_largest_is_12 (n : ℤ) 
  (h : mean_of_five_consecutive_even_numbers_is_12 n) : 
  (8 + (16 : ℤ)) / (2 : ℤ) = 12 := 
by
  sorry

end mean_of_smallest_and_largest_is_12_l25_25777


namespace balloons_division_correct_l25_25455

def number_of_balloons_per_school (yellow blue more_black num_schools: ℕ) : ℕ :=
  let black := yellow + more_black
  let total := yellow + blue + black
  total / num_schools

theorem balloons_division_correct :
  number_of_balloons_per_school 3414 5238 1762 15 = 921 := 
by
  sorry

end balloons_division_correct_l25_25455


namespace rate_per_kg_mangoes_is_55_l25_25817

def total_amount : ℕ := 1125
def rate_per_kg_grapes : ℕ := 70
def weight_grapes : ℕ := 9
def weight_mangoes : ℕ := 9

def cost_grapes := rate_per_kg_grapes * weight_grapes
def cost_mangoes := total_amount - cost_grapes

theorem rate_per_kg_mangoes_is_55 (rate_per_kg_mangoes : ℕ) (h : rate_per_kg_mangoes = cost_mangoes / weight_mangoes) : rate_per_kg_mangoes = 55 :=
by
  -- proof construction
  sorry

end rate_per_kg_mangoes_is_55_l25_25817


namespace number_of_girl_students_l25_25903

theorem number_of_girl_students (total_third_graders : ℕ) (boy_students : ℕ) (girl_students : ℕ) 
  (h1 : total_third_graders = 123) (h2 : boy_students = 66) (h3 : total_third_graders = boy_students + girl_students) :
  girl_students = 57 :=
by
  sorry

end number_of_girl_students_l25_25903


namespace L_shaped_region_area_l25_25966

noncomputable def area_L_shaped_region (length full_width : ℕ) (sub_length sub_width : ℕ) : ℕ :=
  let area_full_rect := length * full_width
  let small_width := length - sub_length
  let small_height := full_width - sub_width
  let area_small_rect := small_width * small_height
  area_full_rect - area_small_rect

theorem L_shaped_region_area :
  area_L_shaped_region 10 7 3 4 = 49 :=
by sorry

end L_shaped_region_area_l25_25966


namespace quotient_real_iff_quotient_purely_imaginary_iff_l25_25485

variables {a b c d : ℝ} -- Declare real number variables

-- Problem 1: Proving the necessary and sufficient condition for the quotient to be a real number
theorem quotient_real_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ i : ℝ, ∃ r : ℝ, a/c = r ∧ b/d = 0) ↔ (a * d - b * c = 0) := 
by sorry -- Proof to be filled in

-- Problem 2: Proving the necessary and sufficient condition for the quotient to be a purely imaginary number
theorem quotient_purely_imaginary_iff (a b c d : ℝ) : 
  (c ≠ 0 ∨ d ≠ 0) → 
  (∀ r : ℝ, ∃ i : ℝ, a/c = 0 ∧ b/d = i) ↔ (a * c + b * d = 0) := 
by sorry -- Proof to be filled in

end quotient_real_iff_quotient_purely_imaginary_iff_l25_25485


namespace find_value_of_10n_l25_25239

theorem find_value_of_10n (n : ℝ) (h : 2 * n = 14) : 10 * n = 70 :=
sorry

end find_value_of_10n_l25_25239


namespace cos_neg_2theta_l25_25438

theorem cos_neg_2theta (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : Real.cos (-2 * θ) = -7 / 25 := 
by
  sorry

end cos_neg_2theta_l25_25438


namespace number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l25_25543

def number_of_seatings (n : ℕ) : ℕ := Nat.factorial n

theorem number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other :
  let k := 2      -- Kolya and Olya as a unit
  let remaining := 3 -- The remaining people
  let pairs := 4 -- Pairs of seats that Kolya and Olya can take
  let arrangements_kolya_olya := pairs * 2 -- Each pair can have Kolya and Olya in 2 arrangements
  let arrangements_remaining := number_of_seatings remaining 
  arrangements_kolya_olya * arrangements_remaining = 48 := by
{
  -- This would be the location for the proof implementation
  sorry
}

end number_of_ways_to_seat_Kolya_and_Olya_next_to_each_other_l25_25543


namespace tram_length_proof_l25_25996
-- Import the necessary library

-- Define the conditions
def tram_length : ℕ := 32 -- The length of the tram we want to prove

-- The main theorem to be stated
theorem tram_length_proof (L : ℕ) (v : ℕ) 
  (h1 : v = L / 4)  -- The tram passed by Misha in 4 seconds
  (h2 : v = (L + 64) / 12)  -- The tram passed through a tunnel of 64 meters in 12 seconds
  : L = tram_length :=
by
  sorry

end tram_length_proof_l25_25996


namespace find_certain_number_l25_25782

theorem find_certain_number (d q r : ℕ) (HD : d = 37) (HQ : q = 23) (HR : r = 16) :
    ∃ n : ℕ, n = d * q + r ∧ n = 867 := by
  sorry

end find_certain_number_l25_25782


namespace complex_number_purely_imaginary_l25_25869

theorem complex_number_purely_imaginary (m : ℝ) 
  (h1 : m^2 - 5 * m + 6 = 0) 
  (h2 : m^2 - 3 * m ≠ 0) : 
  m = 2 :=
sorry

end complex_number_purely_imaginary_l25_25869


namespace desired_percentage_of_alcohol_l25_25355

def solution_x_alcohol_by_volume : ℝ := 0.10
def solution_y_alcohol_by_volume : ℝ := 0.30
def volume_solution_x : ℝ := 200
def volume_solution_y : ℝ := 600

theorem desired_percentage_of_alcohol :
  ((solution_x_alcohol_by_volume * volume_solution_x + solution_y_alcohol_by_volume * volume_solution_y) / 
  (volume_solution_x + volume_solution_y)) * 100 = 25 := 
sorry

end desired_percentage_of_alcohol_l25_25355


namespace min_k_l_sum_l25_25684

theorem min_k_l_sum (k l : ℕ) (hk : 120 * k = l^3) (hpos_k : k > 0) (hpos_l : l > 0) :
  k + l = 255 :=
sorry

end min_k_l_sum_l25_25684


namespace ab_value_l25_25729

theorem ab_value 
  (a b : ℕ) 
  (a_pos : a > 0)
  (b_pos : b > 0)
  (h1 : a + b = 30)
  (h2 : 3 * a * b + 4 * a = 5 * b + 318) : 
  (a * b = 56) :=
sorry

end ab_value_l25_25729


namespace PaulineDressCost_l25_25348

-- Lets define the variables for each dress cost
variable (P Jean Ida Patty : ℝ)

-- Condition statements
def condition1 : Prop := Patty = Ida + 10
def condition2 : Prop := Ida = Jean + 30
def condition3 : Prop := Jean = P - 10
def condition4 : Prop := P + Jean + Ida + Patty = 160

-- The proof problem statement
theorem PaulineDressCost : 
  condition1 Patty Ida →
  condition2 Ida Jean →
  condition3 Jean P →
  condition4 P Jean Ida Patty →
  P = 30 := by
  sorry

end PaulineDressCost_l25_25348


namespace number_of_students_is_20_l25_25746

-- Define the constants and conditions
def average_age_all_students (N : ℕ) : ℕ := 20
def average_age_9_students : ℕ := 11
def average_age_10_students : ℕ := 24
def age_20th_student : ℕ := 61

theorem number_of_students_is_20 (N : ℕ) 
  (h1 : N * average_age_all_students N = 99 + 240 + 61) 
  (h2 : 99 = 9 * average_age_9_students) 
  (h3 : 240 = 10 * average_age_10_students) 
  (h4 : N = 9 + 10 + 1) : N = 20 :=
sorry

end number_of_students_is_20_l25_25746


namespace find_a_minus_b_l25_25426

-- Definitions based on conditions
def eq1 (a b : Int) : Prop := 2 * b + a = 5
def eq2 (a b : Int) : Prop := a * b = -12

-- Statement of the problem
theorem find_a_minus_b (a b : Int) (h1 : eq1 a b) (h2 : eq2 a b) : a - b = -7 := 
sorry

end find_a_minus_b_l25_25426


namespace patriots_won_games_l25_25006

theorem patriots_won_games (C P M S T E : ℕ) 
  (hC : C > 25)
  (hPC : P > C)
  (hMP : M > P)
  (hSC : S > C)
  (hSP : S < P)
  (hTE : T > E) : 
  P = 35 :=
sorry

end patriots_won_games_l25_25006


namespace L_shaped_figure_area_l25_25067

noncomputable def area_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem L_shaped_figure_area :
  let large_rect_length := 10
  let large_rect_width := 7
  let small_rect_length := 4
  let small_rect_width := 3
  area_rectangle large_rect_length large_rect_width - area_rectangle small_rect_length small_rect_width = 58 :=
by
  sorry

end L_shaped_figure_area_l25_25067


namespace geometric_progression_x_unique_l25_25726

theorem geometric_progression_x_unique (x : ℝ) :
  (70+x)^2 = (30+x)*(150+x) ↔ x = 10 := by
  sorry

end geometric_progression_x_unique_l25_25726


namespace animals_in_field_l25_25117

def dog := 1
def cats := 4
def rabbits_per_cat := 2
def hares_per_rabbit := 3

def rabbits := cats * rabbits_per_cat
def hares := rabbits * hares_per_rabbit

def total_animals := dog + cats + rabbits + hares

theorem animals_in_field : total_animals = 37 := by
  sorry

end animals_in_field_l25_25117


namespace solve_xyz_l25_25031

theorem solve_xyz (a b c : ℝ) (h1 : a = y + z) (h2 : b = x + z) (h3 : c = x + y) 
                   (h4 : 0 < y) (h5 : 0 < z) (h6 : 0 < x)
                   (hab : b + c > a) (hbc : a + c > b) (hca : a + b > c) :
  x = (b - a + c)/2 ∧ y = (a - b + c)/2 ∧ z = (a + b - c)/2 :=
by
  sorry

end solve_xyz_l25_25031


namespace abs_abc_eq_abs_k_l25_25369

variable {a b c k : ℝ}

noncomputable def distinct_nonzero (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem abs_abc_eq_abs_k (h_distinct : distinct_nonzero a b c)
                          (h_nonzero_k : k ≠ 0)
                          (h_eq : a + k / b = b + k / c ∧ b + k / c = c + k / a) :
  |a * b * c| = |k| :=
by
  sorry

end abs_abc_eq_abs_k_l25_25369


namespace total_bowling_balls_is_66_l25_25228

-- Define the given conditions
def red_bowling_balls := 30
def difference_green_red := 6
def green_bowling_balls := red_bowling_balls + difference_green_red

-- The statement to prove
theorem total_bowling_balls_is_66 :
  red_bowling_balls + green_bowling_balls = 66 := by
  sorry

end total_bowling_balls_is_66_l25_25228


namespace ratio_of_length_to_width_l25_25638

variable (P W L : ℕ)
variable (ratio : ℕ × ℕ)

theorem ratio_of_length_to_width (h1 : P = 336) (h2 : W = 70) (h3 : 2 * L + 2 * W = P) : ratio = (7, 5) :=
by
  sorry

end ratio_of_length_to_width_l25_25638


namespace detergent_required_l25_25832

def ounces_of_detergent_per_pound : ℕ := 2
def pounds_of_clothes : ℕ := 9

theorem detergent_required :
  (ounces_of_detergent_per_pound * pounds_of_clothes) = 18 := by
  sorry

end detergent_required_l25_25832


namespace megan_markers_l25_25093

theorem megan_markers (initial_markers : ℕ) (new_markers : ℕ) (total_markers : ℕ) :
  initial_markers = 217 →
  new_markers = 109 →
  total_markers = 326 →
  initial_markers + new_markers = 326 :=
by
  sorry

end megan_markers_l25_25093


namespace remaining_course_distance_l25_25649

def total_distance_km : ℝ := 10.5
def distance_to_break_km : ℝ := 1.5
def additional_distance_m : ℝ := 3730.0

theorem remaining_course_distance :
  let total_distance_m := total_distance_km * 1000
  let distance_to_break_m := distance_to_break_km * 1000
  let total_traveled_m := distance_to_break_m + additional_distance_m
  total_distance_m - total_traveled_m = 5270 := by
  sorry

end remaining_course_distance_l25_25649


namespace alloy_chromium_amount_l25_25573

theorem alloy_chromium_amount
  (x : ℝ) -- The amount of the first alloy used (in kg)
  (h1 : 0.10 * x + 0.08 * 35 = 0.086 * (x + 35)) -- Condition based on percentages of chromium
  : x = 15 := 
by
  sorry

end alloy_chromium_amount_l25_25573


namespace total_distance_covered_is_correct_fuel_cost_excess_is_correct_l25_25745

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

end total_distance_covered_is_correct_fuel_cost_excess_is_correct_l25_25745


namespace complement_intersection_l25_25652

open Set

-- Definitions of sets
def U : Set ℕ := {2, 3, 4, 5, 6}
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- The theorem statement
theorem complement_intersection :
  (U \ B) ∩ A = {2, 6} := by
  sorry

end complement_intersection_l25_25652


namespace sufficient_not_necessary_condition_parallel_lines_l25_25794

theorem sufficient_not_necessary_condition_parallel_lines :
  ∀ (a : ℝ), (a = 1/2 → (∀ x y : ℝ, x + 2*a*y = 1 ↔ (x - x + 1) ≠ 0) 
            ∧ ((∃ a', a' ≠ 1/2 ∧ (∀ x y : ℝ, x + 2*a'*y = 1 ↔ (x - x + 1) ≠ 0)) → (a ≠ 1/2))) :=
by
  intro a
  sorry

end sufficient_not_necessary_condition_parallel_lines_l25_25794


namespace problem_ns_k_divisibility_l25_25520

theorem problem_ns_k_divisibility (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
  (∃ (a b : ℕ), (a = 1 ∨ a = 5) ∧ (b = 1 ∨ b = 5) ∧ a = n ∧ b = k) ↔ 
  n * k ∣ (2^(2^n) + 1) * (2^(2^k) + 1) := 
sorry

end problem_ns_k_divisibility_l25_25520


namespace negation_of_p_l25_25346

def p := ∃ n : ℕ, n^2 > 2 * n - 1

theorem negation_of_p : ¬ p ↔ ∀ n : ℕ, n^2 ≤ 2 * n - 1 :=
by sorry

end negation_of_p_l25_25346


namespace bobbie_letters_to_remove_l25_25447

-- Definitions of the conditions
def samanthaLastNameLength := 7
def bobbieLastNameLength := samanthaLastNameLength + 3
def jamieLastNameLength := 4
def targetBobbieLastNameLength := 2 * jamieLastNameLength

-- Question: How many letters does Bobbie need to take off to have a last name twice the length of Jamie's?
theorem bobbie_letters_to_remove : 
  bobbieLastNameLength - targetBobbieLastNameLength = 2 := by 
  sorry

end bobbie_letters_to_remove_l25_25447


namespace f_relationship_l25_25598

noncomputable def f (x : ℝ) : ℝ := sorry -- definition of f needs to be filled in later

-- Conditions given in the problem
variable (h_diff : Differentiable ℝ f)
variable (h_gt : ∀ x: ℝ, deriv f x > f x)
variable (a : ℝ) (h_pos : a > 0)

theorem f_relationship (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_gt : ∀ x: ℝ, deriv f x > f x) (a : ℝ) (h_pos : a > 0) :
  f a > Real.exp a * f 0 :=
sorry

end f_relationship_l25_25598


namespace unique_triangle_determination_l25_25098

-- Definitions for each type of triangle and their respective conditions
def isosceles_triangle (base_angle : ℝ) (altitude : ℝ) : Type := sorry
def vertex_base_isosceles_triangle (vertex_angle : ℝ) (base : ℝ) : Type := sorry
def circ_radius_side_equilateral_triangle (radius : ℝ) (side : ℝ) : Type := sorry
def leg_radius_right_triangle (leg : ℝ) (radius : ℝ) : Type := sorry
def angles_side_scalene_triangle (angle1 : ℝ) (angle2 : ℝ) (opp_side : ℝ) : Type := sorry

-- Condition: Option A does not uniquely determine a triangle
def option_A_does_not_uniquely_determine : Prop :=
  ∀ (base_angle altitude : ℝ), 
    (∃ t1 t2 : isosceles_triangle base_angle altitude, t1 ≠ t2)

-- Condition: Options B through E uniquely determine the triangle
def options_B_to_E_uniquely_determine : Prop :=
  (∀ (vertex_angle base : ℝ), ∃! t : vertex_base_isosceles_triangle vertex_angle base, true) ∧
  (∀ (radius side : ℝ), ∃! t : circ_radius_side_equilateral_triangle radius side, true) ∧
  (∀ (leg radius : ℝ), ∃! t : leg_radius_right_triangle leg radius, true) ∧
  (∀ (angle1 angle2 opp_side : ℝ), ∃! t : angles_side_scalene_triangle angle1 angle2 opp_side, true)

-- Main theorem combining both conditions
theorem unique_triangle_determination :
  option_A_does_not_uniquely_determine ∧ options_B_to_E_uniquely_determine :=
  sorry

end unique_triangle_determination_l25_25098


namespace gcd_x_y_not_8_l25_25637

theorem gcd_x_y_not_8 (x y : ℕ) (hx : x > 0) (hy : y = x^2 + 8) : ¬ ∃ d, d = 8 ∧ d ∣ x ∧ d ∣ y :=
by
  sorry

end gcd_x_y_not_8_l25_25637


namespace correct_operation_l25_25471

theorem correct_operation (a : ℝ) : 
    (a ^ 2 + a ^ 4 ≠ a ^ 6) ∧ 
    (a ^ 2 * a ^ 3 ≠ a ^ 6) ∧ 
    (a ^ 3 / a ^ 2 = a) ∧ 
    ((a ^ 2) ^ 3 ≠ a ^ 5) :=
by
  sorry

end correct_operation_l25_25471


namespace rate_of_interest_l25_25946

noncomputable def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ (n : ℝ)

theorem rate_of_interest (P : ℝ) (r : ℝ) (A : ℕ → ℝ) :
  A 2 = compound_interest P r 2 →
  A 3 = compound_interest P r 3 →
  A 2 = 2420 →
  A 3 = 2662 →
  r = 10 :=
by
  sorry

end rate_of_interest_l25_25946


namespace quadratic_root_a_value_l25_25018

theorem quadratic_root_a_value (a k : ℝ) (h1 : k = 65) (h2 : a * (5:ℝ)^2 + 3 * (5:ℝ) - k = 0) : a = 2 :=
by
  sorry

end quadratic_root_a_value_l25_25018


namespace find_x_l25_25771

theorem find_x (x : ℝ) (h_pos : 0 < x) (h_eq : x * ⌊x⌋ = 48) : x = 8 :=
sorry

end find_x_l25_25771


namespace find_principal_l25_25160

theorem find_principal
  (P : ℝ)
  (R : ℝ := 4)
  (T : ℝ := 5)
  (SI : ℝ := (P * R * T) / 100) 
  (h : SI = P - 2400) : 
  P = 3000 := 
sorry

end find_principal_l25_25160


namespace mike_pumpkins_l25_25539

def pumpkins : ℕ :=
  let sandy_pumpkins := 51
  let total_pumpkins := 74
  total_pumpkins - sandy_pumpkins

theorem mike_pumpkins : pumpkins = 23 :=
by
  sorry

end mike_pumpkins_l25_25539


namespace part_1_part_2_l25_25135

noncomputable def f (x a : ℝ) : ℝ := x^2 * |x - a|

theorem part_1 (a : ℝ) (h : a = 2) : {x : ℝ | f x a = x} = {0, 1, 1 + Real.sqrt 2} :=
by 
  sorry

theorem part_2 (a : ℝ) : 
  ∃ m : ℝ, m = 
    if a ≤ 1 then 1 - a 
    else if 1 < a ∧ a ≤ 2 then 0 
    else if 2 < a ∧ a ≤ (7 / 3 : ℝ) then 4 * (a - 2) 
    else a - 1 :=
by 
  sorry

end part_1_part_2_l25_25135


namespace intersection_M_N_is_neq_neg1_0_1_l25_25834

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N_is_neq_neg1_0_1 :
  M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_M_N_is_neq_neg1_0_1_l25_25834


namespace P_gt_Q_l25_25779

theorem P_gt_Q (a : ℝ) : 
  let P := a^2 + 2*a
  let Q := 3*a - 1
  P > Q :=
by
  sorry

end P_gt_Q_l25_25779


namespace Jules_height_l25_25551

theorem Jules_height (Ben_initial_height Jules_initial_height Ben_current_height Jules_current_height : ℝ) 
  (h_initial : Ben_initial_height = Jules_initial_height)
  (h_Ben_growth : Ben_current_height = 1.25 * Ben_initial_height)
  (h_Jules_growth : Jules_current_height = Jules_initial_height + (Ben_current_height - Ben_initial_height) / 3)
  (h_Ben_current : Ben_current_height = 75) 
  : Jules_current_height = 65 := 
by
  -- Use the conditions to prove that Jules is now 65 inches tall
  sorry

end Jules_height_l25_25551


namespace intersection_P_Q_l25_25515

def P : Set ℝ := { x | x^2 - 9 < 0 }
def Q : Set ℤ := { x | -1 ≤ x ∧ x ≤ 3 }

theorem intersection_P_Q : (P ∩ (coe '' Q)) = { x : ℝ | x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 } :=
by sorry

end intersection_P_Q_l25_25515


namespace perimeter_of_original_rectangle_l25_25269

-- Define the rectangle's dimensions based on the given condition
def length_of_rectangle := 2 * 8 -- because it forms two squares of side 8 cm each
def width_of_rectangle := 8 -- side of the squares

-- Using the formula for the perimeter of a rectangle: P = 2 * (length + width)
def perimeter_of_rectangle := 2 * (length_of_rectangle + width_of_rectangle)

-- The statement we need to prove
theorem perimeter_of_original_rectangle : perimeter_of_rectangle = 48 := by
  sorry

end perimeter_of_original_rectangle_l25_25269


namespace G_is_even_l25_25102

noncomputable def G (F : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := 
  F x * (1 / (a^x - 1) + 1 / 2)

theorem G_is_even (a : ℝ) (F : ℝ → ℝ) 
  (h₀ : a > 0) 
  (h₁ : a ≠ 1)
  (hF : ∀ x : ℝ, F (-x) = - F x) : 
  ∀ x : ℝ, G F a (-x) = G F a x :=
by 
  sorry

end G_is_even_l25_25102


namespace joes_speed_second_part_l25_25186

theorem joes_speed_second_part
  (d1 d2 t1 t_total: ℝ)
  (s1 s_avg: ℝ)
  (h_d1: d1 = 420)
  (h_d2: d2 = 120)
  (h_s1: s1 = 60)
  (h_s_avg: s_avg = 54) :
  (d1 / s1 + d2 / (d2 / 40) = t_total ∧ t_total = (d1 + d2) / s_avg) →
  d2 / (t_total - d1 / s1) = 40 :=
by
  sorry

end joes_speed_second_part_l25_25186


namespace inequality_proof_l25_25439

theorem inequality_proof {k l m n : ℕ} (h_pos_k : 0 < k) (h_pos_l : 0 < l) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_klmn : k < l ∧ l < m ∧ m < n)
  (h_equation : k * n = l * m) : 
  (n - k) / 2 ^ 2 ≥ k + 2 := 
by sorry

end inequality_proof_l25_25439


namespace squirrel_acorns_l25_25492

theorem squirrel_acorns (S A : ℤ) 
  (h1 : A = 4 * S + 3) 
  (h2 : A = 5 * S - 6) : 
  A = 39 :=
by sorry

end squirrel_acorns_l25_25492


namespace average_mileage_city_l25_25596

variable (total_distance : ℝ) (gallons : ℝ) (highway_mpg : ℝ) (city_mpg : ℝ)

-- The given conditions
def conditions : Prop := (total_distance = 280.6) ∧ (gallons = 23) ∧ (highway_mpg = 12.2)

-- The theorem to prove
theorem average_mileage_city (h : conditions total_distance gallons highway_mpg) :
  total_distance / gallons = 12.2 :=
sorry

end average_mileage_city_l25_25596


namespace bob_first_six_probability_l25_25272

noncomputable def probability_bob_first_six (p : ℚ) : ℚ :=
  (1 - p) * p / (1 - ( (1 - p) * (1 - p)))

theorem bob_first_six_probability :
  probability_bob_first_six (1/6) = 5/11 :=
by
  sorry

end bob_first_six_probability_l25_25272


namespace tall_cupboard_glasses_l25_25857

-- Define the number of glasses held by the tall cupboard (T)
variable (T : ℕ)

-- Condition: Wide cupboard holds twice as many glasses as the tall cupboard
def wide_cupboard_holds_twice_as_many (T : ℕ) : Prop :=
  ∃ W : ℕ, W = 2 * T

-- Condition: Narrow cupboard holds 15 glasses initially, 5 glasses per shelf, one shelf broken
def narrow_cupboard_holds_after_break : Prop :=
  ∃ N : ℕ, N = 10

-- Final statement to prove: Number of glasses in the tall cupboard is 5
theorem tall_cupboard_glasses (T : ℕ) (h1 : wide_cupboard_holds_twice_as_many T) (h2 : narrow_cupboard_holds_after_break) : T = 5 :=
sorry

end tall_cupboard_glasses_l25_25857


namespace order_of_abc_l25_25824

noncomputable def a : ℝ := (1 / 3) ^ (2 / 5)
noncomputable def b : ℝ := 2 ^ (4 / 3)
noncomputable def c : ℝ := Real.log 1 / 3 / Real.log 2

theorem order_of_abc : c < a ∧ a < b :=
by {
  -- The proof would go here
  sorry
}

end order_of_abc_l25_25824


namespace arithmetic_example_l25_25338

theorem arithmetic_example : 3889 + 12.808 - 47.80600000000004 = 3854.002 := 
by
  sorry

end arithmetic_example_l25_25338


namespace inequality_inequality_only_if_k_is_one_half_l25_25014

theorem inequality_inequality_only_if_k_is_one_half :
  (∀ t : ℝ, -1 < t ∧ t < 1 → (1 + t) ^ k * (1 - t) ^ (1 - k) ≤ 1) ↔ k = 1 / 2 :=
by
  sorry

end inequality_inequality_only_if_k_is_one_half_l25_25014


namespace doctors_to_lawyers_ratio_l25_25429

theorem doctors_to_lawyers_ratio
  (d l : ℕ)
  (h1 : (40 * d + 55 * l) / (d + l) = 45)
  (h2 : d + l = 20) :
  d / l = 2 :=
by sorry

end doctors_to_lawyers_ratio_l25_25429


namespace expression_value_l25_25523

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h1 : x + y + z = 0) (h2 : xy + xz + yz ≠ 0) :
  (x^3 + y^3 + z^3) / (xyz * (xy + xz + yz)^2) = 3 / (x^2 + xy + y^2)^2 :=
by
  sorry

end expression_value_l25_25523


namespace ratio_of_areas_of_concentric_circles_l25_25081

theorem ratio_of_areas_of_concentric_circles (C1 C2 : ℝ) (h1 : (60 / 360) * C1 = (45 / 360) * C2) :
  (C1 / C2) ^ 2 = (9 / 16) := by
  sorry

end ratio_of_areas_of_concentric_circles_l25_25081


namespace pradeep_max_marks_l25_25060

theorem pradeep_max_marks (M : ℝ) 
  (pass_condition : 0.35 * M = 210) : M = 600 :=
sorry

end pradeep_max_marks_l25_25060


namespace curve_is_line_l25_25654

theorem curve_is_line (r θ : ℝ) (h : r = 2 / (Real.sin θ + Real.cos θ)) : 
  ∃ m b, ∀ θ, r * Real.cos θ = m * (r * Real.sin θ) + b :=
sorry

end curve_is_line_l25_25654


namespace union_of_M_and_Q_is_correct_l25_25264

-- Given sets M and Q
def M : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {0, 1, 3, 5}

-- Statement to prove
theorem union_of_M_and_Q_is_correct : M ∪ Q = {0, 1, 2, 3, 4, 5, 6} :=
by
  sorry

end union_of_M_and_Q_is_correct_l25_25264


namespace value_of_a_l25_25468

theorem value_of_a (x y z a : ℤ) (k : ℤ) 
  (h1 : x = 4 * k) (h2 : y = 6 * k) (h3 : z = 10 * k) 
  (hy_eq : y^2 = 40 * a - 20) 
  (ha_int : ∃ m : ℤ, a = m) : a = 1 := 
  sorry

end value_of_a_l25_25468


namespace simplify_expr_l25_25100

theorem simplify_expr : 3 * (4 - 2 * Complex.I) - 2 * Complex.I * (3 - 2 * Complex.I) = 8 - 12 * Complex.I :=
by
  sorry

end simplify_expr_l25_25100


namespace sound_speed_temperature_l25_25316

theorem sound_speed_temperature (v : ℝ) (T : ℝ) (h1 : v = 0.4) (h2 : T = 15 * v^2) :
  T = 2.4 :=
by {
  sorry
}

end sound_speed_temperature_l25_25316


namespace larger_volume_of_rotated_rectangle_l25_25524

-- Definitions based on the conditions
def length : ℝ := 4
def width : ℝ := 3

-- Problem statement: Proving the volume of the larger geometric solid
theorem larger_volume_of_rotated_rectangle :
  max (Real.pi * (width ^ 2) * length) (Real.pi * (length ^ 2) * width) = 48 * Real.pi :=
by
  sorry

end larger_volume_of_rotated_rectangle_l25_25524


namespace compute_b_l25_25639

open Real

theorem compute_b
  (a : ℚ) 
  (b : ℚ) 
  (h₀ : (3 + sqrt 5) ^ 3 + a * (3 + sqrt 5) ^ 2 + b * (3 + sqrt 5) + 12 = 0) 
  : b = -14 :=
sorry

end compute_b_l25_25639


namespace sin_double_angle_l25_25339

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
by
  sorry

end sin_double_angle_l25_25339


namespace additional_money_earned_l25_25494

-- Define the conditions as variables
def price_duck : ℕ := 10
def price_chicken : ℕ := 8
def num_chickens_sold : ℕ := 5
def num_ducks_sold : ℕ := 2
def half (x : ℕ) : ℕ := x / 2
def double (x : ℕ) : ℕ := 2 * x

-- Define the calculations based on the conditions
def earnings_chickens : ℕ := num_chickens_sold * price_chicken 
def earnings_ducks : ℕ := num_ducks_sold * price_duck 
def total_earnings : ℕ := earnings_chickens + earnings_ducks 
def cost_wheelbarrow : ℕ := half total_earnings
def selling_price_wheelbarrow : ℕ := double cost_wheelbarrow
def additional_earnings : ℕ := selling_price_wheelbarrow - cost_wheelbarrow

-- The theorem to prove the correct additional earnings
theorem additional_money_earned : additional_earnings = 30 := by
  sorry

end additional_money_earned_l25_25494


namespace slopes_product_l25_25356

variables {a b c x0 y0 alpha beta : ℝ}
variables {P Q : ℝ × ℝ}
variables (M : ℝ × ℝ) (kPQ kOM : ℝ)

-- Conditions: a, b are positive real numbers
axiom a_pos : a > 0
axiom b_pos : b > 0

-- Condition: b^2 = a c
axiom b_squared_eq_a_mul_c : b^2 = a * c

-- Condition: P and Q lie on the hyperbola
axiom P_on_hyperbola : (P.1^2 / a^2) - (P.2^2 / b^2) = 1
axiom Q_on_hyperbola : (Q.1^2 / a^2) - (Q.2^2 / b^2) = 1

-- Condition: M is the midpoint of P and Q
axiom M_is_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Condition: Slopes kPQ and kOM exist
axiom kOM_def : kOM = y0 / x0
axiom kPQ_def : kPQ = beta / alpha

-- Theorem: Value of the product of the slopes
theorem slopes_product : kPQ * kOM = (1 + Real.sqrt 5) / 2 :=
sorry

end slopes_product_l25_25356


namespace percentage_increase_in_cellphone_pay_rate_l25_25727

theorem percentage_increase_in_cellphone_pay_rate
    (regular_rate : ℝ)
    (total_surveys : ℕ)
    (cellphone_surveys : ℕ)
    (total_earnings : ℝ)
    (regular_surveys : ℕ := total_surveys - cellphone_surveys)
    (higher_rate : ℝ := (total_earnings - (regular_surveys * regular_rate)) / cellphone_surveys)
    : regular_rate = 30 ∧ total_surveys = 100 ∧ cellphone_surveys = 50 ∧ total_earnings = 3300
    → ((higher_rate - regular_rate) / regular_rate) * 100 = 20 := by
  sorry

end percentage_increase_in_cellphone_pay_rate_l25_25727


namespace union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l25_25526

open Set

variables (U : Set ℝ) (A B : Set ℝ) (a : ℝ)

def A_def : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }
def B_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 2 }
def comp_U_A : Set ℝ := { x | x < 1 ∨ x > 4 }

theorem union_A_B_at_a_3 (h : a = 3) :
  A_def ∪ B_def 3 = { x | 1 ≤ x ∧ x ≤ 5 } :=
sorry

theorem inter_B_compl_A_at_a_3 (h : a = 3) :
  B_def 3 ∩ comp_U_A = { x | 4 < x ∧ x ≤ 5 } :=
sorry

theorem B_subset_A_imp_a_range (h : B_def a ⊆ A_def) :
  1 ≤ a ∧ a ≤ 2 :=
sorry

end union_A_B_at_a_3_inter_B_compl_A_at_a_3_B_subset_A_imp_a_range_l25_25526


namespace dolls_total_correct_l25_25910

def Jazmin_dolls : Nat := 1209
def Geraldine_dolls : Nat := 2186
def total_dolls : Nat := Jazmin_dolls + Geraldine_dolls

theorem dolls_total_correct : total_dolls = 3395 := by
  sorry

end dolls_total_correct_l25_25910


namespace product_of_intersection_coords_l25_25603

open Real

-- Define the two circles
def circle1 (x y: ℝ) : Prop := x^2 - 2*x + y^2 - 10*y + 21 = 0
def circle2 (x y: ℝ) : Prop := x^2 - 8*x + y^2 - 10*y + 52 = 0

-- Prove that the product of the coordinates of intersection points equals 189
theorem product_of_intersection_coords :
  (∃ (x1 y1 x2 y2 : ℝ), circle1 x1 y1 ∧ circle2 x1 y1 ∧ circle1 x2 y2 ∧ circle2 x2 y2 ∧ x1 * y1 * x2 * y2 = 189) :=
by
  sorry

end product_of_intersection_coords_l25_25603


namespace multiply_expression_l25_25545

variable {x : ℝ}

theorem multiply_expression :
  (x^4 + 10*x^2 + 25) * (x^2 - 25) = x^4 + 10*x^2 :=
by
  sorry

end multiply_expression_l25_25545


namespace tangent_line_at_M_l25_25594

noncomputable def isOnCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def M : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem tangent_line_at_M (hM : isOnCircle (M.1) (M.2)) : (∀ x y, M.1 = x ∨ M.2 = y → x + y = Real.sqrt 2) :=
by
  sorry

end tangent_line_at_M_l25_25594


namespace grid_shaded_area_l25_25678

theorem grid_shaded_area :
  let grid_side := 12
  let grid_area := grid_side^2
  let radius_small := 1.5
  let radius_large := 3
  let area_small := π * radius_small^2
  let area_large := π * radius_large^2
  let total_area_circles := 3 * area_small + area_large
  let visible_area := grid_area - total_area_circles
  let A := 144
  let B := 15.75
  A = 144 ∧ B = 15.75 ∧ (A + B = 159.75) →
  visible_area = 144 - 15.75 * π :=
by
  intros
  sorry

end grid_shaded_area_l25_25678


namespace compute_value_of_expression_l25_25838

theorem compute_value_of_expression :
  ∃ p q : ℝ, (3 * p^2 - 3 * q^2) / (p - q) = 5 ∧ 3 * p^2 - 5 * p - 14 = 0 ∧ 3 * q^2 - 5 * q - 14 = 0 :=
sorry

end compute_value_of_expression_l25_25838


namespace no_solution_to_inequality_l25_25381

theorem no_solution_to_inequality (x : ℝ) (h : x ≥ -1/4) : ¬(-1 - 1 / (3 * x + 4) < 2) :=
by sorry

end no_solution_to_inequality_l25_25381


namespace total_cookies_baked_l25_25327

-- Definitions based on conditions
def pans : ℕ := 5
def cookies_per_pan : ℕ := 8

-- Statement of the theorem to be proven
theorem total_cookies_baked :
  pans * cookies_per_pan = 40 := by
  sorry

end total_cookies_baked_l25_25327


namespace stockholm_to_malmo_distance_l25_25427
-- Import the necessary library

-- Define the parameters for the problem.
def map_distance : ℕ := 120 -- distance in cm
def scale_factor : ℕ := 12 -- km per cm

-- The hypothesis for the map distance and the scale factor
axiom map_distance_hyp : map_distance = 120
axiom scale_factor_hyp : scale_factor = 12

-- Define the real distance function
def real_distance (d : ℕ) (s : ℕ) : ℕ := d * s

-- The problem statement: Prove that the real distance between the two city centers is 1440 km
theorem stockholm_to_malmo_distance : real_distance map_distance scale_factor = 1440 :=
by
  rw [map_distance_hyp, scale_factor_hyp]
  sorry

end stockholm_to_malmo_distance_l25_25427


namespace popsicle_sticks_difference_l25_25640

def popsicle_sticks_boys (boys : ℕ) (sticks_per_boy : ℕ) : ℕ :=
  boys * sticks_per_boy

def popsicle_sticks_girls (girls : ℕ) (sticks_per_girl : ℕ) : ℕ :=
  girls * sticks_per_girl

theorem popsicle_sticks_difference : 
    popsicle_sticks_boys 10 15 - popsicle_sticks_girls 12 12 = 6 := by
  sorry

end popsicle_sticks_difference_l25_25640


namespace waiter_earnings_l25_25176

def num_customers : ℕ := 9
def num_no_tip : ℕ := 5
def tip_per_customer : ℕ := 8
def num_tipping_customers := num_customers - num_no_tip

theorem waiter_earnings : num_tipping_customers * tip_per_customer = 32 := by
  sorry

end waiter_earnings_l25_25176


namespace Gordons_heavier_bag_weight_l25_25145

theorem Gordons_heavier_bag_weight :
  ∀ (G : ℝ), (5 * 2 = 3 + G) → G = 7 :=
by
  intro G h
  sorry

end Gordons_heavier_bag_weight_l25_25145


namespace probability_mass_range_l25_25066

/-- Let ξ be a random variable representing the mass of a badminton product. 
    Suppose P(ξ < 4.8) = 0.3 and P(ξ ≥ 4.85) = 0.32. 
    We want to prove that the probability that the mass is in the range [4.8, 4.85) is 0.38. -/
theorem probability_mass_range (P : ℝ → ℝ) (h1 : P (4.8) = 0.3) (h2 : P (4.85) = 0.32) :
  P (4.8) - P (4.85) = 0.38 :=
by 
  sorry

end probability_mass_range_l25_25066


namespace smallest_solution_x_abs_x_eq_3x_plus_2_l25_25421

theorem smallest_solution_x_abs_x_eq_3x_plus_2 : ∃ x : ℝ, (x * abs x = 3 * x + 2) ∧ (∀ y : ℝ, (y * abs y = 3 * y + 2) → x ≤ y) ∧ x = -2 :=
by
  sorry

end smallest_solution_x_abs_x_eq_3x_plus_2_l25_25421


namespace Reeya_fifth_subject_score_l25_25947

theorem Reeya_fifth_subject_score 
  (a1 a2 a3 a4 : ℕ) (avg : ℕ) (subjects : ℕ) (a1_eq : a1 = 55) (a2_eq : a2 = 67) (a3_eq : a3 = 76) 
  (a4_eq : a4 = 82) (avg_eq : avg = 73) (subjects_eq : subjects = 5) :
  ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5) / subjects = avg ∧ a5 = 85 :=
by
  sorry

end Reeya_fifth_subject_score_l25_25947


namespace area_of_inscribed_triangle_l25_25072

theorem area_of_inscribed_triangle 
  (x : ℝ) 
  (h1 : (2:ℝ) * x ≤ (3:ℝ) * x ∧ (3:ℝ) * x ≤ (4:ℝ) * x) 
  (h2 : (4:ℝ) * x = 2 * 4) :
  ∃ (area : ℝ), area = 12.00 :=
by
  sorry

end area_of_inscribed_triangle_l25_25072


namespace sufficient_condition_for_inequality_l25_25225

theorem sufficient_condition_for_inequality (a x : ℝ) (h1 : -2 < x) (h2 : x < -1) :
  (a + x) * (1 + x) < 0 → a > 2 :=
sorry

end sufficient_condition_for_inequality_l25_25225


namespace age_difference_l25_25411

variable (A J : ℕ)
variable (h1 : A + 5 = 40)
variable (h2 : J = 31)

theorem age_difference (h1 : A + 5 = 40) (h2 : J = 31) : A - J = 4 := by
  sorry

end age_difference_l25_25411


namespace factorization_a_minus_b_l25_25738

theorem factorization_a_minus_b (a b: ℤ) 
  (h : (4 * y + a) * (y + b) = 4 * y * y - 3 * y - 28) : a - b = -11 := by
  sorry

end factorization_a_minus_b_l25_25738


namespace connie_total_markers_l25_25187

/-
Connie has 4 different types of markers: red, blue, green, and yellow.
She has twice as many red markers as green markers.
She has three times as many blue markers as red markers.
She has four times as many yellow markers as green markers.
She has 36 green markers.
Prove that the total number of markers she has is 468.
-/

theorem connie_total_markers
 (g r b y : ℕ) 
 (hg : g = 36) 
 (hr : r = 2 * g)
 (hb : b = 3 * r)
 (hy : y = 4 * g) :
 g + r + b + y = 468 := 
 by
  sorry

end connie_total_markers_l25_25187


namespace cake_remaining_l25_25358

theorem cake_remaining (T J: ℝ) (h1: T = 0.60) (h2: J = 0.25) :
  (1 - ((1 - T) * J + T)) = 0.30 :=
by
  sorry

end cake_remaining_l25_25358


namespace bus_stops_bound_l25_25489

-- Definitions based on conditions
variables (n x : ℕ)

-- Condition 1: Any bus stop is serviced by at most 3 bus lines
def at_most_three_bus_lines (bus_stops : ℕ) : Prop :=
  ∀ (stop : ℕ), stop < bus_stops → stop ≤ 3

-- Condition 2: Any bus line has at least two stops
def at_least_two_stops (bus_lines : ℕ) : Prop :=
  ∀ (line : ℕ), line < bus_lines → line ≥ 2

-- Condition 3: For any two specific bus lines, there is a third line such that passengers can transfer
def transfer_line_exists (bus_lines : ℕ) : Prop :=
  ∀ (line1 line2 : ℕ), line1 < bus_lines ∧ line2 < bus_lines →
  ∃ (line3 : ℕ), line3 < bus_lines

-- Theorem statement: The number of bus stops is at least 5/6 (n-5)
theorem bus_stops_bound (h1 : at_most_three_bus_lines x) (h2 : at_least_two_stops n)
  (h3 : transfer_line_exists n) : x ≥ (5 * (n - 5)) / 6 :=
sorry

end bus_stops_bound_l25_25489


namespace minimize_transfers_l25_25089

-- Define the initial number of pieces in each supermarket
def pieces_in_A := 15
def pieces_in_B := 7
def pieces_in_C := 11
def pieces_in_D := 3
def pieces_in_E := 14

-- Define the target number of pieces in each supermarket after transfers
def target_pieces := 10

-- Define a function to compute the total number of pieces
def total_pieces := pieces_in_A + pieces_in_B + pieces_in_C + pieces_in_D + pieces_in_E

-- Define the minimum number of transfers needed
def min_transfers := 12

-- The main theorem: proving that the minimum number of transfers is 12
theorem minimize_transfers : 
  total_pieces = 5 * target_pieces → 
  ∃ (transfers : ℕ), transfers = min_transfers :=
by
  -- This represents the proof section, we leave it as sorry
  sorry

end minimize_transfers_l25_25089


namespace binary_addition_l25_25352

def bin_to_dec1 := 511  -- 111111111_2 in decimal
def bin_to_dec2 := 127  -- 1111111_2 in decimal

theorem binary_addition : bin_to_dec1 + bin_to_dec2 = 638 := by
  sorry

end binary_addition_l25_25352


namespace roots_value_l25_25374

theorem roots_value (m n : ℝ) (h1 : Polynomial.eval m (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) (h2 : Polynomial.eval n (Polynomial.C 1 + Polynomial.C 3 * Polynomial.X + Polynomial.X ^ 2) = 0) : m^2 + 4 * m + n = -2 := 
sorry

end roots_value_l25_25374


namespace trees_after_planting_l25_25071

variable (x : ℕ)

theorem trees_after_planting (x : ℕ) : 
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  x + additional_trees - trees_removed = x - 6 :=
by
  let additional_trees := 12
  let days := 12 / 2
  let trees_removed := days * 3
  sorry

end trees_after_planting_l25_25071


namespace vertical_line_intersect_parabola_ex1_l25_25029

theorem vertical_line_intersect_parabola_ex1 (m : ℝ) (h : ∀ y : ℝ, (-4 * y^2 + 2*y + 3 = m) → false) :
  m = 13 / 4 :=
sorry

end vertical_line_intersect_parabola_ex1_l25_25029


namespace golden_triangle_expression_l25_25921

noncomputable def t : ℝ := (Real.sqrt 5 - 1) / 2

theorem golden_triangle_expression :
  t = (Real.sqrt 5 - 1) / 2 →
  (1 - 2 * (Real.sin (27 * Real.pi / 180))^2) / (2 * t * Real.sqrt (4 - t^2)) = 1 / 4 :=
by
  intro h_t
  have h1 : t = (Real.sqrt 5 - 1) / 2 := h_t
  sorry

end golden_triangle_expression_l25_25921


namespace g_difference_l25_25295

variable (g : ℝ → ℝ)

-- Condition: g is a linear function
axiom linear_g : ∃ a b : ℝ, ∀ x : ℝ, g x = a * x + b

-- Condition: g(10) - g(4) = 18
axiom g_condition : g 10 - g 4 = 18

theorem g_difference : g 16 - g 4 = 36 :=
by
  sorry

end g_difference_l25_25295


namespace total_cost_is_1_85_times_selling_price_l25_25055

def total_cost (P : ℝ) : ℝ := 140 * 2 * P + 90 * P

def loss (P : ℝ) : ℝ := 70 * 2 * P + 30 * P

def selling_price (P : ℝ) : ℝ := total_cost P - loss P

theorem total_cost_is_1_85_times_selling_price (P : ℝ) :
  total_cost P = 1.85 * selling_price P := by
  sorry

end total_cost_is_1_85_times_selling_price_l25_25055


namespace additional_amount_deductibles_next_year_l25_25561

theorem additional_amount_deductibles_next_year :
  let avg_deductible : ℝ := 3000
  let inflation_rate : ℝ := 0.03
  let plan_a_rate : ℝ := 2 / 3
  let plan_b_rate : ℝ := 1 / 2
  let plan_c_rate : ℝ := 3 / 5
  let plan_a_percent : ℝ := 0.40
  let plan_b_percent : ℝ := 0.30
  let plan_c_percent : ℝ := 0.30
  let additional_a : ℝ := avg_deductible * plan_a_rate
  let additional_b : ℝ := avg_deductible * plan_b_rate
  let additional_c : ℝ := avg_deductible * plan_c_rate
  let weighted_additional : ℝ := (additional_a * plan_a_percent) + (additional_b * plan_b_percent) + (additional_c * plan_c_percent)
  let inflation_increase : ℝ := weighted_additional * inflation_rate
  let total_additional_amount : ℝ := weighted_additional + inflation_increase
  total_additional_amount = 1843.70 :=
sorry

end additional_amount_deductibles_next_year_l25_25561


namespace frosting_problem_l25_25009

-- Define the conditions
def cagney_rate := 1/15  -- Cagney's rate in cupcakes per second
def lacey_rate := 1/45   -- Lacey's rate in cupcakes per second
def total_time := 600  -- Total time in seconds (10 minutes)

-- Function to calculate the combined rate
def combined_rate (r1 r2 : ℝ) : ℝ := r1 + r2

-- Hypothesis combining the conditions
def hypothesis : Prop :=
  combined_rate cagney_rate lacey_rate = 1/11.25

-- Statement to prove: together they can frost 53 cupcakes within 10 minutes 
theorem frosting_problem : ∀ (total_time: ℝ) (hyp : hypothesis),
  total_time / (cagney_rate + lacey_rate) = 53 :=
by
  intro total_time hyp
  sorry

end frosting_problem_l25_25009


namespace speed_of_man_in_still_water_l25_25266

theorem speed_of_man_in_still_water 
  (v_m v_s : ℝ)
  (h1 : 32 = 4 * (v_m + v_s))
  (h2 : 24 = 4 * (v_m - v_s)) :
  v_m = 7 :=
by
  sorry

end speed_of_man_in_still_water_l25_25266


namespace markese_earnings_16_l25_25428

theorem markese_earnings_16 (E M : ℕ) (h1 : M = E - 5) (h2 : E + M = 37) : M = 16 :=
by
  sorry

end markese_earnings_16_l25_25428


namespace amelia_money_left_l25_25531

theorem amelia_money_left :
  let first_course := 15
  let second_course := first_course + 5
  let dessert := 0.25 * second_course
  let total_first_three_courses := first_course + second_course + dessert
  let drink := 0.20 * total_first_three_courses
  let pre_tip_total := total_first_three_courses + drink
  let tip := 0.15 * pre_tip_total
  let total_bill := pre_tip_total + tip
  let initial_money := 60
  let money_left := initial_money - total_bill
  money_left = 4.8 :=
by
  sorry

end amelia_money_left_l25_25531


namespace find_triangles_l25_25627

/-- In a triangle, if the side lengths a, b, c (a ≤ b ≤ c) are integers, form a geometric progression (i.e., b² = ac),
    and at least one of a or c is equal to 100, then the possible values for the triple (a, b, c) are:
    (49, 70, 100), (64, 80, 100), (81, 90, 100), 
    (100, 100, 100), (100, 110, 121), (100, 120, 144),
    (100, 130, 169), (100, 140, 196), (100, 150, 225), (100, 160, 256). 
-/
theorem find_triangles (a b c : ℕ) (h1 : a ≤ b ∧ b ≤ c) 
(h2 : b * b = a * c)
(h3 : a = 100 ∨ c = 100) : 
  (a = 49 ∧ b = 70 ∧ c = 100) ∨ 
  (a = 64 ∧ b = 80 ∧ c = 100) ∨ 
  (a = 81 ∧ b = 90 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 100 ∧ c = 100) ∨ 
  (a = 100 ∧ b = 110 ∧ c = 121) ∨ 
  (a = 100 ∧ b = 120 ∧ c = 144) ∨ 
  (a = 100 ∧ b = 130 ∧ c = 169) ∨ 
  (a = 100 ∧ b = 140 ∧ c = 196) ∨ 
  (a = 100 ∧ b = 150 ∧ c = 225) ∨ 
  (a = 100 ∧ b = 160 ∧ c = 256) := sorry

end find_triangles_l25_25627


namespace alloy_copper_percentage_l25_25292

theorem alloy_copper_percentage 
  (x : ℝ)
  (h1 : 0 ≤ x)
  (h2 : (30 / 100) * x + (70 / 100) * 27 = 24.9) :
  x = 20 :=
sorry

end alloy_copper_percentage_l25_25292


namespace square_divided_into_40_smaller_squares_l25_25251

theorem square_divided_into_40_smaller_squares : ∃ squares : ℕ, squares = 40 :=
by
  sorry

end square_divided_into_40_smaller_squares_l25_25251


namespace problem_statement_l25_25809

variable (f : ℝ → ℝ)

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

theorem problem_statement (h_odd : is_odd f) (h_decr : is_decreasing f) (a b : ℝ) (h_ab : a + b < 0) :
  f (a + b) > 0 ∧ f a + f b > 0 :=
by
  sorry

end problem_statement_l25_25809


namespace range_of_a_l25_25164

noncomputable def setA (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
noncomputable def setB : Set ℝ := {x | x < -1 ∨ x > 3}

theorem range_of_a (a : ℝ) :
  ((setA a ∩ setB) = setA a) ∧ (∃ x, x ∈ (setA a ∩ setB)) →
  (a < -3 ∨ a > 3) ∧ (a < -1 ∨ a > 1) :=
by sorry

end range_of_a_l25_25164


namespace fish_ratio_l25_25308

theorem fish_ratio (k : ℕ) (kendra_fish : ℕ) (home_fish : ℕ)
    (h1 : kendra_fish = 30)
    (h2 : home_fish = 87)
    (h3 : k - 3 + kendra_fish = home_fish) :
  k = 60 ∧ (k / 3, kendra_fish / 3) = (19, 10) :=
by
  sorry

end fish_ratio_l25_25308


namespace first_division_percentage_l25_25795

theorem first_division_percentage (total_students : ℕ) (second_division_percentage just_passed_students : ℕ) 
  (h1 : total_students = 300) (h2 : second_division_percentage = 54) (h3 : just_passed_students = 60) : 
  (100 - second_division_percentage - ((just_passed_students * 100) / total_students)) = 26 :=
by
  sorry

end first_division_percentage_l25_25795


namespace log_expression_simplification_l25_25883

open Real

theorem log_expression_simplification (p q r s t z : ℝ) :
  log (p / q) + log (q / r) + log (r / s) - log (p * t / (s * z)) = log (z / t) :=
  sorry

end log_expression_simplification_l25_25883


namespace part1_part2_l25_25241

def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y
def B (x y : ℝ) : ℝ := x^2 - x * y + x

def difference (x y : ℝ) : ℝ := A x y - 2 * B x y

theorem part1 : difference (-2) 3 = -20 :=
by
  -- Proving that difference (-2) 3 = -20
  sorry

theorem part2 (y : ℝ) : (∀ (x : ℝ), difference x y = 2 * y) → y = 2 / 5 :=
by
  -- Proving that if difference x y is independent of x, then y = 2 / 5
  sorry

end part1_part2_l25_25241


namespace ratio_of_numbers_l25_25233

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : 0 < a) (h3 : b < a) (h4 : a + b = 7 * (a - b)) :
  a / b = 4 / 3 :=
sorry

end ratio_of_numbers_l25_25233


namespace motorcyclists_speeds_l25_25466

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

end motorcyclists_speeds_l25_25466


namespace derivative_f_at_2_l25_25616

noncomputable def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

theorem derivative_f_at_2 : (deriv f 2) = 4 := by
  sorry

end derivative_f_at_2_l25_25616


namespace range_of_m_l25_25274

theorem range_of_m (m : ℝ) (h1 : (m - 3) < 0) (h2 : (m + 1) > 0) : -1 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l25_25274


namespace problem1_problem2_l25_25954

theorem problem1 : (- (2 : ℤ) ^ 3 / 8 - (1 / 4 : ℚ) * ((-2)^2)) = -2 :=
by {
    sorry
}

theorem problem2 : ((- (1 / 12 : ℚ) - 1 / 16 + 3 / 4 - 1 / 6) * -48) = -21 :=
by {
    sorry
}

end problem1_problem2_l25_25954


namespace percent_increase_jordan_alex_l25_25454

theorem percent_increase_jordan_alex :
  let pound_to_dollar := 1.5
  let alex_dollars := 600
  let jordan_pounds := 450
  let jordan_dollars := jordan_pounds * pound_to_dollar
  let percent_increase := ((jordan_dollars - alex_dollars) / alex_dollars) * 100
  percent_increase = 12.5 := 
by
  sorry

end percent_increase_jordan_alex_l25_25454


namespace routes_from_Bristol_to_Carlisle_l25_25687

-- Given conditions as definitions
def routes_Bristol_to_Birmingham : ℕ := 8
def routes_Birmingham_to_Manchester : ℕ := 5
def routes_Manchester_to_Sheffield : ℕ := 4
def routes_Sheffield_to_Newcastle : ℕ := 3
def routes_Newcastle_to_Carlisle : ℕ := 2

-- Define the total number of routes from Bristol to Carlisle
def total_routes_Bristol_to_Carlisle : ℕ := routes_Bristol_to_Birmingham *
                                            routes_Birmingham_to_Manchester *
                                            routes_Manchester_to_Sheffield *
                                            routes_Sheffield_to_Newcastle *
                                            routes_Newcastle_to_Carlisle

-- The theorem to be proved
theorem routes_from_Bristol_to_Carlisle :
  total_routes_Bristol_to_Carlisle = 960 :=
by
  -- Proof will be provided here
  sorry

end routes_from_Bristol_to_Carlisle_l25_25687


namespace fib_ratio_bound_l25_25934

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_ratio_bound {a b n : ℕ} (h1: b > 0) (h2: fib (n-1) > 0)
  (h3: (fib n) * b > (fib (n-1)) * a)
  (h4: (fib (n+1)) * b < (fib n) * a) :
  b ≥ fib (n+1) :=
sorry

end fib_ratio_bound_l25_25934


namespace reflected_point_correct_l25_25950

-- Defining the original point coordinates
def original_point : ℝ × ℝ := (3, -5)

-- Defining the transformation function
def reflect_across_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, point.2)

-- Proving the point after reflection is as expected
theorem reflected_point_correct : reflect_across_y_axis original_point = (-3, -5) :=
by
  sorry

end reflected_point_correct_l25_25950


namespace B_can_finish_work_in_6_days_l25_25913

theorem B_can_finish_work_in_6_days :
  (A_work_alone : ℕ) → (A_work_before_B : ℕ) → (A_B_together : ℕ) → (B_days_alone : ℕ) → 
  (A_work_alone = 12) → (A_work_before_B = 3) → (A_B_together = 3) → B_days_alone = 6 :=
by
  intros A_work_alone A_work_before_B A_B_together B_days_alone
  intros h1 h2 h3
  sorry

end B_can_finish_work_in_6_days_l25_25913


namespace expenditure_representation_l25_25123

def income_represented_pos (income : ℤ) : Prop := income > 0

def expenditure_represented_neg (expenditure : ℤ) : Prop := expenditure < 0

theorem expenditure_representation (income expenditure : ℤ) (h_income: income_represented_pos income) (exp_value: expenditure = 3) : expenditure_represented_neg expenditure := 
sorry

end expenditure_representation_l25_25123


namespace janet_earns_more_as_freelancer_l25_25236

-- Definitions for the problem conditions
def current_job_weekly_hours : ℕ := 40
def current_job_hourly_rate : ℕ := 30

def freelance_client_a_hours_per_week : ℕ := 15
def freelance_client_a_hourly_rate : ℕ := 45

def freelance_client_b_hours_project1_per_week : ℕ := 5
def freelance_client_b_hours_project2_per_week : ℕ := 10
def freelance_client_b_hourly_rate : ℕ := 40

def freelance_client_c_hours_per_week : ℕ := 20
def freelance_client_c_rate_range : ℕ × ℕ := (35, 42)

def weekly_fica_taxes : ℕ := 25
def monthly_healthcare_premiums : ℕ := 400
def monthly_increased_rent : ℕ := 750
def monthly_business_phone_internet : ℕ := 150
def business_expense_percentage : ℕ := 10

def weeks_in_month : ℕ := 4

-- Define the calculations
def current_job_monthly_earnings := current_job_weekly_hours * current_job_hourly_rate * weeks_in_month

def freelance_client_a_weekly_earnings := freelance_client_a_hours_per_week * freelance_client_a_hourly_rate
def freelance_client_b_weekly_earnings := (freelance_client_b_hours_project1_per_week + freelance_client_b_hours_project2_per_week) * freelance_client_b_hourly_rate
def freelance_client_c_weekly_earnings := freelance_client_c_hours_per_week * ((freelance_client_c_rate_range.1 + freelance_client_c_rate_range.2) / 2)

def total_freelance_weekly_earnings := freelance_client_a_weekly_earnings + freelance_client_b_weekly_earnings + freelance_client_c_weekly_earnings
def total_freelance_monthly_earnings := total_freelance_weekly_earnings * weeks_in_month

def total_additional_expenses := (weekly_fica_taxes * weeks_in_month) + monthly_healthcare_premiums + monthly_increased_rent + monthly_business_phone_internet

def business_expense_deduction := (total_freelance_monthly_earnings * business_expense_percentage) / 100
def adjusted_freelance_earnings_after_deduction := total_freelance_monthly_earnings - business_expense_deduction
def adjusted_freelance_earnings_after_expenses := adjusted_freelance_earnings_after_deduction - total_additional_expenses

def earnings_difference := adjusted_freelance_earnings_after_expenses - current_job_monthly_earnings

-- The theorem to be proved
theorem janet_earns_more_as_freelancer :
  earnings_difference = 1162 :=
sorry

end janet_earns_more_as_freelancer_l25_25236


namespace find_m_l25_25147

def point (α : Type) := (α × α)

def collinear {α : Type} [LinearOrderedField α] 
  (p1 p2 p3 : point α) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p2.1) = (p3.2 - p2.2) * (p2.1 - p1.1)

theorem find_m {m : ℚ} 
  (h : collinear (4, 10) (-3, m) (-12, 5)) : 
  m = 125 / 16 :=
by sorry

end find_m_l25_25147


namespace sarah_stamp_collection_value_l25_25096

theorem sarah_stamp_collection_value :
  ∀ (stamps_owned total_value_for_4_stamps : ℝ) (num_stamps_single_series : ℕ), 
  stamps_owned = 20 → 
  total_value_for_4_stamps = 10 → 
  num_stamps_single_series = 4 → 
  (stamps_owned / num_stamps_single_series) * (total_value_for_4_stamps / num_stamps_single_series) = 50 :=
by
  intros stamps_owned total_value_for_4_stamps num_stamps_single_series 
  intro h_stamps_owned
  intro h_total_value_for_4_stamps
  intro h_num_stamps_single_series
  rw [h_stamps_owned, h_total_value_for_4_stamps, h_num_stamps_single_series]
  sorry

end sarah_stamp_collection_value_l25_25096


namespace translated_parabola_correct_l25_25222

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := x^2 + 2

-- Theorem stating that translating the original parabola up by 2 units results in the translated parabola
theorem translated_parabola_correct (x : ℝ) :
  translated_parabola x = original_parabola x + 2 :=
by
  sorry

end translated_parabola_correct_l25_25222


namespace gcd_of_75_and_360_l25_25211

theorem gcd_of_75_and_360 : Nat.gcd 75 360 = 15 := by
  sorry

end gcd_of_75_and_360_l25_25211


namespace circle_radius_through_focus_and_tangent_l25_25026

-- Define the given conditions of the problem
def ellipse_eq (x y : ℝ) : Prop := x^2 + 4 * y^2 = 16

-- State the problem as a theorem
theorem circle_radius_through_focus_and_tangent
  (x y : ℝ) (h : ellipse_eq x y) (r : ℝ) :
  r = 4 - 2 * Real.sqrt 3 :=
sorry

end circle_radius_through_focus_and_tangent_l25_25026


namespace cost_price_l25_25592

variables (SP DS CP : ℝ)
variables (discount_rate profit_rate : ℝ)
variables (H1 : SP = 24000)
variables (H2 : discount_rate = 0.10)
variables (H3 : profit_rate = 0.08)
variables (H4 : DS = SP - (discount_rate * SP))
variables (H5 : DS = CP + (profit_rate * CP))

theorem cost_price (H1 : SP = 24000) (H2 : discount_rate = 0.10) 
  (H3 : profit_rate = 0.08) (H4 : DS = SP - (discount_rate * SP)) 
  (H5 : DS = CP + (profit_rate * CP)) : 
  CP = 20000 := 
sorry

end cost_price_l25_25592


namespace contrapositive_statement_l25_25061

-- Conditions: x and y are real numbers
variables (x y : ℝ)

-- Contrapositive statement: If x ≠ 0 or y ≠ 0, then x^2 + y^2 ≠ 0
theorem contrapositive_statement (hx : x ≠ 0 ∨ y ≠ 0) : x^2 + y^2 ≠ 0 :=
sorry

end contrapositive_statement_l25_25061


namespace reading_minutes_per_disc_l25_25957

-- Define the total reading time
def total_reading_time := 630

-- Define the maximum capacity per disc
def max_capacity_per_disc := 80

-- Define the allowable unused space
def max_unused_space := 4

-- Define the effective capacity of each disc
def effective_capacity_per_disc := max_capacity_per_disc - max_unused_space

-- Define the number of discs needed, rounded up as a ceiling function
def number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)

-- Theorem statement: Each disc will contain 70 minutes of reading if all conditions are met
theorem reading_minutes_per_disc : ∀ (total_reading_time : ℕ) (max_capacity_per_disc : ℕ) (max_unused_space : ℕ)
  (effective_capacity_per_disc := max_capacity_per_disc - max_unused_space) 
  (number_of_discs := Nat.ceil (total_reading_time / effective_capacity_per_disc)), 
  number_of_discs = 9 → total_reading_time / number_of_discs = 70 :=
by
  sorry

end reading_minutes_per_disc_l25_25957


namespace right_triangle_ratio_l25_25700

theorem right_triangle_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∃ (x y : ℝ), 5 * (x * y) = x^2 + y^2 ∧ 5 * (a^2 + b^2) = (x + y)^2 ∧ 
    ((x - y)^2 < x^2 + y^2 ∧ x^2 + y^2 < (x + y)^2)):
  (1/2 < a / b) ∧ (a / b < 2) := by
  sorry

end right_triangle_ratio_l25_25700


namespace difference_between_numbers_l25_25288

noncomputable def L : ℕ := 1614
noncomputable def Q : ℕ := 6
noncomputable def R : ℕ := 15

theorem difference_between_numbers (S : ℕ) (h : L = Q * S + R) : L - S = 1348 :=
by {
  -- proof skipped
  sorry
}

end difference_between_numbers_l25_25288


namespace roses_picked_later_l25_25759

/-- Represents the initial number of roses the florist had. -/
def initial_roses : ℕ := 37

/-- Represents the number of roses the florist sold. -/
def sold_roses : ℕ := 16

/-- Represents the final number of roses the florist ended up with. -/
def final_roses : ℕ := 40

/-- Theorem which states the number of roses picked later is 19 given the conditions. -/
theorem roses_picked_later : (final_roses - (initial_roses - sold_roses)) = 19 :=
by
  -- proof steps are omitted, sorry as a placeholder
  sorry

end roses_picked_later_l25_25759


namespace solve_for_y_l25_25960

theorem solve_for_y (y : ℝ) (h_sum : (1 + 99) * 99 / 2 = 4950)
  (h_avg : (4950 + y) / 100 = 50 * y) : y = 4950 / 4999 :=
by
  sorry

end solve_for_y_l25_25960


namespace percentage_spent_on_household_items_l25_25069

theorem percentage_spent_on_household_items (monthly_income : ℝ) (savings : ℝ) (clothes_percentage : ℝ) (medicines_percentage : ℝ) (household_spent : ℝ) : 
  monthly_income = 40000 ∧ 
  savings = 9000 ∧ 
  clothes_percentage = 0.25 ∧ 
  medicines_percentage = 0.075 ∧ 
  household_spent = monthly_income - (clothes_percentage * monthly_income + medicines_percentage * monthly_income + savings)
  → (household_spent / monthly_income) * 100 = 45 :=
by
  intro h
  cases' h with h1 h_rest
  cases' h_rest with h2 h_rest
  cases' h_rest with h3 h_rest
  cases' h_rest with h4 h5
  have h_clothes := h3
  have h_medicines := h4
  have h_savings := h2
  have h_income := h1
  have h_household := h5
  sorry

end percentage_spent_on_household_items_l25_25069


namespace prove_value_range_for_a_l25_25564

noncomputable def f (x a : ℝ) : ℝ :=
  (x^2 + a*x + 7 + a) / (x + 1)

noncomputable def g (x : ℝ) : ℝ := 
  - ((x + 1) + (8 / (x + 1))) + 6

theorem prove_value_range_for_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 4) ↔ (a ≥ 1 / 3) :=
sorry

end prove_value_range_for_a_l25_25564


namespace arith_seq_a4_a10_l25_25764

variable {a : ℕ → ℕ}
axiom hp1 : a 1 + a 2 + a 3 = 32
axiom hp2 : a 11 + a 12 + a 13 = 118

theorem arith_seq_a4_a10 :
  a 4 + a 10 = 50 :=
by
  have h1 : a 2 = 32 / 3 := sorry
  have h2 : a 12 = 118 / 3 := sorry
  have h3 : a 2 + a 12 = 50 := sorry
  exact sorry

end arith_seq_a4_a10_l25_25764


namespace circle_center_coordinates_l25_25935

theorem circle_center_coordinates :
  ∃ c : ℝ × ℝ, (∀ x y : ℝ, x^2 + y^2 - x + 2*y = 0 ↔ (x-c.1)^2 + (y-c.2)^2 = (5/4)) ∧ c = (1/2, -1) :=
sorry

end circle_center_coordinates_l25_25935


namespace train_A_length_l25_25896

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

end train_A_length_l25_25896


namespace total_cats_l25_25496

theorem total_cats (a b c d : ℝ) (ht : a = 15.5) (hs : b = 11.6) (hg : c = 24.2) (hr : d = 18.3) :
  a + b + c + d = 69.6 :=
by
  sorry

end total_cats_l25_25496


namespace james_veg_consumption_l25_25893

-- Define the given conditions in Lean
def asparagus_per_day : ℝ := 0.25
def broccoli_per_day : ℝ := 0.25
def days_in_week : ℝ := 7
def weeks : ℝ := 2
def kale_per_week : ℝ := 3

-- Define the amount of vegetables (initial, doubled, and added kale)
def initial_veg_per_day := asparagus_per_day + broccoli_per_day
def initial_veg_per_week := initial_veg_per_day * days_in_week
def double_veg_per_week := initial_veg_per_week * weeks
def total_veg_per_week_after_kale := double_veg_per_week + kale_per_week

-- Statement of the proof problem
theorem james_veg_consumption :
  total_veg_per_week_after_kale = 10 := by 
  sorry

end james_veg_consumption_l25_25893


namespace brocard_inequalities_l25_25495

theorem brocard_inequalities (α β γ φ: ℝ) (h1: φ > 0) (h2: φ < π / 6)
  (h3: α > 0) (h4: β > 0) (h5: γ > 0) (h6: α + β + γ = π) : 
  (φ^3 ≤ (α - φ) * (β - φ) * (γ - φ)) ∧ (8 * φ^3 ≤ α * β * γ) := 
by 
  sorry

end brocard_inequalities_l25_25495


namespace decreasing_function_l25_25409

noncomputable def f (a x : ℝ) : ℝ := a^(1 - x)

theorem decreasing_function (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : ∀ x > 1, f a x < 1) :
  ∀ x y : ℝ, x < y → f a x > f a y :=
sorry

end decreasing_function_l25_25409


namespace measure_angle_y_l25_25922

theorem measure_angle_y
  (triangle_angles : ∀ {A B C : ℝ}, (A = 45 ∧ B = 45 ∧ C = 90) ∨ (A = 45 ∧ B = 90 ∧ C = 45) ∨ (A = 90 ∧ B = 45 ∧ C = 45))
  (p q : ℝ) (hpq : p = q) :
  ∃ (y : ℝ), y = 90 :=
by
  sorry

end measure_angle_y_l25_25922


namespace al_original_portion_l25_25080

theorem al_original_portion {a b c d : ℕ} 
  (h1 : a + b + c + d = 2000)
  (h2 : a - 150 + 3 * b + 3 * c + d - 50 = 2500)
  (h3 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  a = 450 :=
sorry

end al_original_portion_l25_25080


namespace min_value_of_x2_add_y2_l25_25554

theorem min_value_of_x2_add_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end min_value_of_x2_add_y2_l25_25554


namespace no_integer_solutions_l25_25340

theorem no_integer_solutions (x y z : ℤ) :
  x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
  -x^2 + 4 * y * z + 3 * z^2 = 36 ∧
  x^2 + 2 * x * y + 9 * z^2 = 121 → false :=
by
  sorry

end no_integer_solutions_l25_25340


namespace sum_of_distinct_prime_factors_of_462_l25_25528

-- Given a number n, define its prime factors.
def prime_factors (n : ℕ) : List ℕ :=
  if h : n = 462 then [2, 3, 7, 11] else []

-- Defines the sum of a list of natural numbers.
def sum_list (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

-- The main theorem statement.
theorem sum_of_distinct_prime_factors_of_462 : sum_list (prime_factors 462) = 23 :=
by
  sorry

end sum_of_distinct_prime_factors_of_462_l25_25528


namespace min_value_expression_l25_25103

open Real

theorem min_value_expression (α β : ℝ) :
  (3 * cos α + 4 * sin β - 7)^2 + (3 * sin α + 4 * cos β - 12)^2 ≥ 36 := by
  sorry

end min_value_expression_l25_25103


namespace find_x_plus_y_l25_25388

theorem find_x_plus_y (x y : ℝ)
  (h1 : (x - 1)^3 + 2015 * (x - 1) = -1)
  (h2 : (y - 1)^3 + 2015 * (y - 1) = 1)
  : x + y = 2 :=
sorry

end find_x_plus_y_l25_25388


namespace binomial_prime_divisor_l25_25712

theorem binomial_prime_divisor (p k : ℕ) (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : p ∣ Nat.choose p k :=
by
  sorry

end binomial_prime_divisor_l25_25712


namespace index_card_area_l25_25299

theorem index_card_area (length width : ℕ) (h_length : length = 5) (h_width : width = 7)
  (h_area_shortened_length : (length - 2) * width = 21) : (length * (width - 2)) = 25 := by
  sorry

end index_card_area_l25_25299


namespace right_triangle_sides_l25_25597

-- Definitions based on the conditions
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2
def perimeter (a b c : ℕ) : ℕ := a + b + c
def inscribed_circle_radius (a b c : ℕ) : ℕ := (a + b - c) / 2

-- The theorem statement
theorem right_triangle_sides (a b c : ℕ) 
  (h_perimeter : perimeter a b c = 40)
  (h_radius : inscribed_circle_radius a b c = 3)
  (h_right : is_right_triangle a b c) :
  (a = 8 ∧ b = 15 ∧ c = 17) ∨ (a = 15 ∧ b = 8 ∧ c = 17) :=
by sorry

end right_triangle_sides_l25_25597


namespace value_of_t_eq_3_over_4_l25_25197

-- Define the values x and y as per the conditions
def x (t : ℝ) : ℝ := 1 - 2 * t
def y (t : ℝ) : ℝ := 2 * t - 2

-- Statement only, proof is omitted using sorry
theorem value_of_t_eq_3_over_4 (t : ℝ) (h : x t = y t) : t = 3 / 4 :=
by
  sorry

end value_of_t_eq_3_over_4_l25_25197


namespace sqrt_of_neg_five_squared_l25_25793

theorem sqrt_of_neg_five_squared : Real.sqrt ((-5 : Real) ^ 2) = 5 := 
by 
  sorry

end sqrt_of_neg_five_squared_l25_25793


namespace number_of_k_solutions_l25_25792

theorem number_of_k_solutions :
  ∃ (n : ℕ), n = 1006 ∧
  (∀ k, (∃ a b : ℕ+, (a ≠ b) ∧ (k * (a + b) = 2013 * Nat.lcm a b)) ↔ k ≤ n ∧ 0 < k) :=
by
  sorry

end number_of_k_solutions_l25_25792


namespace evaluate_expression_l25_25406

theorem evaluate_expression : (5^2 - 4^2)^3 = 729 :=
by
  sorry

end evaluate_expression_l25_25406


namespace dan_has_3_potatoes_left_l25_25025

-- Defining the number of potatoes Dan originally had
def original_potatoes : ℕ := 7

-- Defining the number of potatoes the rabbits ate
def potatoes_eaten : ℕ := 4

-- The theorem we want to prove: Dan has 3 potatoes left.
theorem dan_has_3_potatoes_left : original_potatoes - potatoes_eaten = 3 := by
  sorry

end dan_has_3_potatoes_left_l25_25025


namespace num_of_arithmetic_sequences_l25_25514

-- Define the set of digits {1, 2, ..., 15}
def digits := {n : ℕ | 1 ≤ n ∧ n ≤ 15}

-- Define an arithmetic sequence condition 
def is_arithmetic_sequence (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

-- Define the count of valid sequences with a specific difference
def count_arithmetic_sequences_with_difference (d : ℕ) : ℕ :=
  if d = 1 then 13
  else if d = 5 then 6
  else 0

-- Define the total count of valid sequences
def total_arithmetic_sequences : ℕ :=
  count_arithmetic_sequences_with_difference 1 +
  count_arithmetic_sequences_with_difference 5

-- The final statement to prove
theorem num_of_arithmetic_sequences : total_arithmetic_sequences = 19 := 
  sorry

end num_of_arithmetic_sequences_l25_25514


namespace minimum_value_inequality_l25_25435

noncomputable def min_value (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : ℝ :=
  (4 / a) + (1 / (b - 1))

theorem minimum_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 1 < b) (hab : a + b = 2) : 
  min_value a b ha hb hab ≥ 9 :=
  sorry

end minimum_value_inequality_l25_25435


namespace five_digit_number_with_integer_cube_root_l25_25540

theorem five_digit_number_with_integer_cube_root (n : ℕ) 
  (h1 : n ≥ 10000 ∧ n < 100000) 
  (h2 : n % 10 = 3) 
  (h3 : ∃ k : ℕ, k^3 = n) : 
  n = 19683 ∨ n = 50653 :=
sorry

end five_digit_number_with_integer_cube_root_l25_25540


namespace common_ratio_value_l25_25052

theorem common_ratio_value (x y z : ℝ) (h : (x + y) / z = (x + z) / y ∧ (x + z) / y = (y + z) / x) :
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) → (x + y + z = 0 ∨ x + y + z ≠ 0) → ((x + y) / z = -1 ∨ (x + y) / z = 2) :=
by
  sorry

end common_ratio_value_l25_25052


namespace binom_floor_divisible_l25_25173

theorem binom_floor_divisible {p n : ℕ}
  (hp : Prime p) :
  (Nat.choose n p - n / p) % p = 0 := 
by
  sorry

end binom_floor_divisible_l25_25173


namespace mr_yadav_expenses_l25_25635

theorem mr_yadav_expenses (S : ℝ) 
  (h1 : S > 0) 
  (h2 : 0.6 * S > 0) 
  (h3 : (12 * 0.2 * S) = 48456) : 
  0.2 * S = 4038 :=
by
  sorry

end mr_yadav_expenses_l25_25635


namespace sum_evaluation_l25_25878

theorem sum_evaluation : 5 * 399 + 4 * 399 + 3 * 399 + 398 = 5186 :=
by
  sorry

end sum_evaluation_l25_25878


namespace intersection_P_Q_l25_25218

noncomputable def P : Set ℝ := { x | x < 1 }
noncomputable def Q : Set ℝ := { x | x^2 < 4 }

theorem intersection_P_Q :
  P ∩ Q = { x | -2 < x ∧ x < 1 } :=
by 
  sorry

end intersection_P_Q_l25_25218


namespace min_value_of_a_l25_25163

theorem min_value_of_a (a : ℝ) (x : ℝ) (h1: 0 < a) (h2: a ≠ 1) (h3: 1 ≤ x → a^x ≥ a * x) : a ≥ Real.exp 1 :=
by
  sorry

end min_value_of_a_l25_25163


namespace mod_product_eq_15_l25_25062

theorem mod_product_eq_15 :
  (15 * 24 * 14) % 25 = 15 :=
by
  sorry

end mod_product_eq_15_l25_25062


namespace tunnel_build_equation_l25_25184

theorem tunnel_build_equation (x : ℝ) (h1 : 1280 > 0) (h2 : x > 0) : 
  (1280 - x) / x = (1280 - x) / (1.4 * x) + 2 := 
by
  sorry

end tunnel_build_equation_l25_25184


namespace work_duration_17_333_l25_25899

def work_done (rate: ℚ) (days: ℕ) : ℚ := rate * days

def combined_work_done (rate1: ℚ) (rate2: ℚ) (days: ℕ) : ℚ :=
  (rate1 + rate2) * days

def total_work_done (rate1: ℚ) (rate2: ℚ) (rate3: ℚ) (days: ℚ) : ℚ :=
  (rate1 + rate2 + rate3) * days

noncomputable def total_days_work_last (rate_p rate_q rate_r: ℚ) : ℚ :=
  have work_p := 8 * rate_p
  have work_pq := combined_work_done rate_p rate_q 4
  have remaining_work := 1 - (work_p + work_pq)
  have days_all_together := remaining_work / (rate_p + rate_q + rate_r)
  8 + 4 + days_all_together

theorem work_duration_17_333 (rate_p rate_q rate_r: ℚ) : total_days_work_last rate_p rate_q rate_r = 17.333 :=
  by 
  have hp := 1/40
  have hq := 1/24
  have hr := 1/30
  sorry -- proof omitted

end work_duration_17_333_l25_25899


namespace solve_equation_l25_25424

theorem solve_equation (x : ℝ) (h : ((x^2 + 3*x + 4) / (x + 5)) = x + 6) : x = -13 / 4 :=
by sorry

end solve_equation_l25_25424


namespace count_base_8_digits_5_or_6_l25_25107

-- Define the conditions in Lean
def is_digit_5_or_6 (d : ℕ) : Prop := d = 5 ∨ d = 6

def count_digits_5_or_6 := 
  let total_base_8 := 512
  let total_without_5_6 := 6 * 6 * 6 -- since we exclude 2 out of 8 digits
  total_base_8 - total_without_5_6

-- The statement of the proof problem
theorem count_base_8_digits_5_or_6 : count_digits_5_or_6 = 296 :=
by {
  sorry
}

end count_base_8_digits_5_or_6_l25_25107


namespace find_angle_x_l25_25676

-- Definitions as conditions from the problem statement
def angle_PQR := 120
def angle_PQS (x : ℝ) := 2 * x
def angle_QRS (x : ℝ) := x

-- The theorem to prove
theorem find_angle_x (x : ℝ) (h1 : angle_PQR = 120) (h2 : angle_PQS x + angle_QRS x = angle_PQR) : x = 40 :=
by
  sorry

end find_angle_x_l25_25676


namespace find_value_of_r_l25_25599

theorem find_value_of_r (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a * r / (1 - r^2) = 8) : r = 2 / 3 :=
by
  sorry

end find_value_of_r_l25_25599


namespace distance_between_X_and_Y_l25_25501

theorem distance_between_X_and_Y 
  (b_walked_distance : ℕ) 
  (time_difference : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_rate : ℕ) 
  (time_bob_walked : ℕ) 
  (distance_when_met : ℕ) 
  (bob_walked_8_miles : b_walked_distance = 8) 
  (one_hour_time_difference : time_difference = 1) 
  (yolanda_3_mph : yolanda_rate = 3) 
  (bob_4_mph : bob_rate = 4) 
  (time_bob_2_hours : time_bob_walked = b_walked_distance / bob_rate)
  : 
  distance_when_met = yolanda_rate * (time_bob_walked + time_difference) + bob_rate * time_bob_walked :=
by
  sorry  -- proof steps

end distance_between_X_and_Y_l25_25501


namespace rotated_translated_line_eq_l25_25040

theorem rotated_translated_line_eq :
  ∀ (x y : ℝ), y = 3 * x → y = - (1 / 3) * x + (1 / 3) :=
by
  sorry

end rotated_translated_line_eq_l25_25040


namespace carols_total_peanuts_l25_25612

-- Define the initial number of peanuts Carol has
def initial_peanuts : ℕ := 2

-- Define the number of peanuts given by Carol's father
def peanuts_given : ℕ := 5

-- Define the total number of peanuts Carol has
def total_peanuts : ℕ := initial_peanuts + peanuts_given

-- The statement we need to prove
theorem carols_total_peanuts : total_peanuts = 7 := by
  sorry

end carols_total_peanuts_l25_25612


namespace second_experimental_point_is_correct_l25_25664

-- Define the temperature range
def lower_bound : ℝ := 1400
def upper_bound : ℝ := 1600

-- Define the golden ratio constant
def golden_ratio : ℝ := 0.618

-- Calculate the first experimental point using 0.618 method
def first_point : ℝ := lower_bound + golden_ratio * (upper_bound - lower_bound)

-- Calculate the second experimental point
def second_point : ℝ := upper_bound - (first_point - lower_bound)

-- Theorem stating the calculated second experimental point equals 1476.4
theorem second_experimental_point_is_correct :
  second_point = 1476.4 := by
  sorry

end second_experimental_point_is_correct_l25_25664


namespace correct_number_of_outfits_l25_25605

def num_shirts : ℕ := 7
def num_pants : ℕ := 7
def num_hats : ℕ := 7
def num_colors : ℕ := 7

def total_outfits : ℕ := num_shirts * num_pants * num_hats
def invalid_outfits : ℕ := num_colors
def valid_outfits : ℕ := total_outfits - invalid_outfits

theorem correct_number_of_outfits : valid_outfits = 336 :=
by {
  -- sorry can be removed when providing the proof.
  sorry
}

end correct_number_of_outfits_l25_25605


namespace CA_inter_B_l25_25788

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5, 7}

theorem CA_inter_B :
  (U \ A) ∩ B = {2, 7} := by
  sorry

end CA_inter_B_l25_25788


namespace BANANA_arrangements_l25_25475

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l25_25475


namespace initial_money_l25_25952

-- Define the conditions
def spent_toy_truck : ℕ := 3
def spent_pencil_case : ℕ := 2
def money_left : ℕ := 5

-- Define the total money spent
def total_spent := spent_toy_truck + spent_pencil_case

-- Theorem statement
theorem initial_money (I : ℕ) (h : total_spent + money_left = I) : I = 10 :=
sorry

end initial_money_l25_25952


namespace distribute_balls_into_boxes_l25_25509

theorem distribute_balls_into_boxes : (Nat.choose (5 + 4 - 1) (4 - 1)) = 56 := by
  sorry

end distribute_balls_into_boxes_l25_25509


namespace expression_value_is_one_l25_25262

theorem expression_value_is_one :
  let a1 := 121
  let b1 := 19
  let a2 := 91
  let b2 := 13
  (a1^2 - b1^2) / (a2^2 - b2^2) * ((a2 - b2) * (a2 + b2)) / ((a1 - b1) * (a1 + b1)) = 1 := by
  sorry

end expression_value_is_one_l25_25262


namespace quilt_shaded_fraction_l25_25971

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_squares := 8
  let fully_shaded := 4
  let half_shaded := 4
  let shaded_area := fully_shaded + half_shaded * 1 / 2
  shaded_area / total_squares = 3 / 8 :=
by
  sorry

end quilt_shaded_fraction_l25_25971


namespace sqrt_equality_l25_25619

theorem sqrt_equality (n : ℤ) (h : Real.sqrt (8 + n) = 9) : n = 73 :=
by
  sorry

end sqrt_equality_l25_25619


namespace f_2014_l25_25859

noncomputable def f : ℕ → ℕ := sorry

axiom f_property : ∀ n, f (f n) + f n = 2 * n + 3
axiom f_zero : f 0 = 1

theorem f_2014 : f 2014 = 2015 := 
by sorry

end f_2014_l25_25859


namespace english_homework_correct_time_l25_25929

-- Define the given conditions as constants
def total_time : ℕ := 180 -- 3 hours in minutes
def math_homework_time : ℕ := 45
def science_homework_time : ℕ := 50
def history_homework_time : ℕ := 25
def special_project_time : ℕ := 30

-- Define the function to compute english homework time
def english_homework_time : ℕ :=
  total_time - (math_homework_time + science_homework_time + history_homework_time + special_project_time)

-- The theorem to show the English homework time is 30 minutes
theorem english_homework_correct_time :
  english_homework_time = 30 :=
  by
    sorry

end english_homework_correct_time_l25_25929


namespace trigonometric_identity_l25_25831

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = (1 / (Real.cos (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180))) :=
by
  sorry

end trigonometric_identity_l25_25831


namespace solve_equation1_solve_equation2_pos_solve_equation2_neg_l25_25720

theorem solve_equation1 (x : ℝ) (h : 2 * x^3 = 16) : x = 2 :=
sorry

theorem solve_equation2_pos (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 :=
sorry

theorem solve_equation2_neg (x : ℝ) (h : (x - 1)^2 = 4) : x = -1 :=
sorry

end solve_equation1_solve_equation2_pos_solve_equation2_neg_l25_25720


namespace problem_statement_l25_25778

namespace GeometricRelations

variables {Line Plane : Type} [Nonempty Line] [Nonempty Plane]

-- Define parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Given conditions
variables (m n : Line) (α β : Plane)

-- The theorem to be proven
theorem problem_statement 
  (h1 : perpendicular m β) 
  (h2 : parallel α β) : 
  perpendicular m α :=
sorry

end GeometricRelations

end problem_statement_l25_25778


namespace solution_set_of_quadratic_inequality_l25_25688

theorem solution_set_of_quadratic_inequality (x : ℝ) :
  (x^2 ≤ 4) ↔ (-2 ≤ x ∧ x ≤ 2) :=
by 
  sorry

end solution_set_of_quadratic_inequality_l25_25688


namespace polynomial_solution_l25_25192

open Polynomial
open Real

theorem polynomial_solution (P : Polynomial ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → P.eval (x * sqrt 2) = P.eval (x + sqrt (1 - x^2))) :
  ∃ U : Polynomial ℝ, P = (U.comp (Polynomial.C (1/4) - 2 * X^2 + 5 * X^4 - 4 * X^6 + X^8)) :=
sorry

end polynomial_solution_l25_25192


namespace find_positive_integer_solutions_l25_25307

-- Define the problem conditions
variable {x y z : ℕ}

-- Main theorem statement
theorem find_positive_integer_solutions 
  (h1 : Prime y)
  (h2 : ¬ 3 ∣ z)
  (h3 : ¬ y ∣ z)
  (h4 : x^3 - y^3 = z^2) : 
  x = 8 ∧ y = 7 ∧ z = 13 := 
sorry

end find_positive_integer_solutions_l25_25307


namespace distance_between_adjacent_symmetry_axes_l25_25375

noncomputable def f (x : ℝ) : ℝ := (Real.cos (3 * x))^2 - 1/2

theorem distance_between_adjacent_symmetry_axes :
  (∃ x : ℝ, f x = f (x + π / 3)) → (∃ d : ℝ, d = π / 6) :=
by
  -- Prove the distance is π / 6 based on the properties of f(x).
  sorry

end distance_between_adjacent_symmetry_axes_l25_25375


namespace teacher_engineer_ratio_l25_25907

theorem teacher_engineer_ratio 
  (t e : ℕ) -- t is the number of teachers and e is the number of engineers.
  (h1 : (40 * t + 55 * e) / (t + e) = 45)
  : t = 2 * e :=
by
  sorry

end teacher_engineer_ratio_l25_25907


namespace max_vehicles_div_10_l25_25969

-- Each vehicle is 5 meters long
def vehicle_length : ℕ := 5

-- The speed rule condition
def speed_rule (m : ℕ) : ℕ := 20 * m

-- Maximum number of vehicles in one hour
def max_vehicles_per_hour (m : ℕ) : ℕ := 4000 * m / (m + 1)

-- N is the maximum whole number of vehicles
def N : ℕ := 4000

-- The target statement to prove: quotient when N is divided by 10
theorem max_vehicles_div_10 : N / 10 = 400 :=
by
  -- Definitions and given conditions go here
  sorry

end max_vehicles_div_10_l25_25969


namespace min_surface_area_of_sphere_l25_25472

theorem min_surface_area_of_sphere (a b c : ℝ) (volume : ℝ) (height : ℝ) 
  (h_volume : a * b * c = volume) (h_height : c = height) 
  (volume_val : volume = 12) (height_val : height = 4) : 
  ∃ r : ℝ, 4 * π * r^2 = 22 * π := 
by
  sorry

end min_surface_area_of_sphere_l25_25472


namespace Mark_marbles_correct_l25_25291

def Connie_marbles : ℕ := 323
def Juan_marbles : ℕ := Connie_marbles + 175
def Mark_marbles : ℕ := 3 * Juan_marbles

theorem Mark_marbles_correct : Mark_marbles = 1494 := 
by
  sorry

end Mark_marbles_correct_l25_25291


namespace product_of_two_larger_numbers_is_115_l25_25408

noncomputable def proofProblem : Prop :=
  ∃ (A B C : ℝ), B = 10 ∧ (C - B = B - A) ∧ (A * B = 85) ∧ (B * C = 115)

theorem product_of_two_larger_numbers_is_115 : proofProblem :=
by
  sorry

end product_of_two_larger_numbers_is_115_l25_25408


namespace length_of_bridge_l25_25011

/-- Prove the length of the bridge -/
theorem length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_sec : ℝ) : 
  train_length = 120 →
  train_speed_kmph = 70 →
  crossing_time_sec = 13.884603517432893 →
  (70 * (1000 / 3600) * 13.884603517432893 - 120 = 150) :=
by
  intros h1 h2 h3
  sorry

end length_of_bridge_l25_25011


namespace relationship_among_a_ae_ea_minus_one_l25_25362

theorem relationship_among_a_ae_ea_minus_one (a : ℝ) (h : 0 < a ∧ a < 1) :
  (Real.exp a - 1 > a ∧ a > Real.exp a - 1 ∧ a > a^(Real.exp 1)) :=
by
  sorry

end relationship_among_a_ae_ea_minus_one_l25_25362


namespace maximize_lower_houses_l25_25271

theorem maximize_lower_houses (x y : ℕ) 
    (h1 : x + 2 * y = 30)
    (h2 : 0 < y)
    (h3 : (∃ k, k = 112)) :
  ∃ x y, (x + 2 * y = 30) ∧ ((x * y)) = 112 :=
by
  sorry

end maximize_lower_houses_l25_25271


namespace range_of_k_l25_25473

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x - 3)^2 + (y - 2)^2 = 4 ∧ y = k * x + 3) ∧ 
  (∃ M N : ℝ × ℝ, ((M.1 - N.1)^2 + (M.2 - N.2)^2)^(1/2) ≥ 2) →
  (k ≤ 0) :=
by
  sorry

end range_of_k_l25_25473


namespace coefficient_of_c_l25_25786

theorem coefficient_of_c (f c : ℝ) (h₁ : f = (9/5) * c + 32)
                         (h₂ : f + 25 = (9/5) * (c + 13.88888888888889) + 32) :
  (5/9) = (9/5) := sorry

end coefficient_of_c_l25_25786


namespace sum_of_reciprocal_squares_leq_reciprocal_product_square_l25_25870

theorem sum_of_reciprocal_squares_leq_reciprocal_product_square (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a^2 * b^2 * c^2 * d^2) :=
sorry

end sum_of_reciprocal_squares_leq_reciprocal_product_square_l25_25870


namespace lily_sees_leo_l25_25451

theorem lily_sees_leo : 
  ∀ (d₁ d₂ v₁ v₂ : ℝ), 
  d₁ = 0.75 → 
  d₂ = 0.75 → 
  v₁ = 15 → 
  v₂ = 9 → 
  (d₁ + d₂) / (v₁ - v₂) * 60 = 15 :=
by 
  intros d₁ d₂ v₁ v₂ h₁ h₂ h₃ h₄
  -- skipping the proof with sorry
  sorry

end lily_sees_leo_l25_25451


namespace total_water_intake_l25_25208

def morning_water : ℝ := 1.5
def afternoon_water : ℝ := 3 * morning_water
def evening_water : ℝ := 0.5 * afternoon_water

theorem total_water_intake : 
  (morning_water + afternoon_water + evening_water) = 8.25 :=
by
  sorry

end total_water_intake_l25_25208


namespace sum_a2_a4_a6_l25_25892

-- Define the arithmetic sequence with a positive common difference
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ (d : ℝ), d > 0 ∧ ∀ n, a (n + 1) = a n + d

-- Define that a_1 and a_7 are roots of the quadratic equation x^2 - 10x + 16 = 0
def roots_condition (a : ℕ → ℝ) : Prop :=
(a 1) * (a 7) = 16 ∧ (a 1) + (a 7) = 10

-- The main theorem we want to prove
theorem sum_a2_a4_a6 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : roots_condition a) :
  a 2 + a 4 + a 6 = 15 :=
sorry

end sum_a2_a4_a6_l25_25892


namespace jimmy_hostel_stay_days_l25_25082

-- Definitions based on the conditions
def nightly_hostel_charge : ℕ := 15
def nightly_cabin_charge_per_person : ℕ := 15
def total_lodging_expense : ℕ := 75
def days_in_cabin : ℕ := 2

-- The proof statement
theorem jimmy_hostel_stay_days : 
    ∃ x : ℕ, (nightly_hostel_charge * x + nightly_cabin_charge_per_person * days_in_cabin = total_lodging_expense) ∧ x = 3 := by
    sorry

end jimmy_hostel_stay_days_l25_25082


namespace quadratic_residue_l25_25230

theorem quadratic_residue (a : ℤ) (p : ℕ) (hp : p > 2) (ha_nonzero : a ≠ 0) :
  (∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ 1 [ZMOD p]) ∧
  (¬ ∃ b : ℤ, b^2 ≡ a [ZMOD p] → a^((p - 1) / 2) ≡ -1 [ZMOD p]) :=
sorry

end quadratic_residue_l25_25230


namespace area_of_shaded_region_l25_25050

theorem area_of_shaded_region 
  (r R : ℝ)
  (hR : R = 9)
  (h : 2 * r = R) :
  π * R^2 - 3 * (π * r^2) = 20.25 * π :=
by
  sorry

end area_of_shaded_region_l25_25050


namespace initial_percentage_l25_25324

variable (P : ℝ)

theorem initial_percentage (P : ℝ) 
  (h1 : 0 ≤ P ∧ P ≤ 100)
  (h2 : (7600 * (1 - P / 100) * 0.75) = 5130) :
  P = 10 :=
by
  sorry

end initial_percentage_l25_25324


namespace repeating_decimal_sum_l25_25976

theorem repeating_decimal_sum :
  (0.2 - 0.02) + (0.003 - 0.00003) = (827 / 3333) :=
by
  sorry

end repeating_decimal_sum_l25_25976


namespace simplify_expression_l25_25337

theorem simplify_expression : (2 * 3 * b * 4 * (b ^ 2) * 5 * (b ^ 3) * 6 * (b ^ 4)) = 720 * (b ^ 10) :=
by
  sorry

end simplify_expression_l25_25337


namespace rita_saving_l25_25874

theorem rita_saving
  (num_notebooks : ℕ)
  (price_per_notebook : ℝ)
  (discount_rate : ℝ) :
  num_notebooks = 7 →
  price_per_notebook = 3 →
  discount_rate = 0.15 →
  (num_notebooks * price_per_notebook) - (num_notebooks * (price_per_notebook * (1 - discount_rate))) = 3.15 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end rita_saving_l25_25874


namespace dina_dolls_l25_25840

theorem dina_dolls (Ivy_collectors: ℕ) (h1: Ivy_collectors = 20) (h2: ∀ y : ℕ, 2 * y / 3 = Ivy_collectors) :
  ∃ x : ℕ, 2 * x = 60 :=
  sorry

end dina_dolls_l25_25840


namespace average_speed_l25_25732

-- Define the conditions
def distance1 := 350 -- miles
def time1 := 6 -- hours
def distance2 := 420 -- miles
def time2 := 7 -- hours

-- Define the total distance and total time (excluding break)
def total_distance := distance1 + distance2
def total_time := time1 + time2

-- Define the statement to prove
theorem average_speed : 
  (total_distance / total_time : ℚ) = 770 / 13 := by
  sorry

end average_speed_l25_25732


namespace average_score_l25_25632

theorem average_score (s1 s2 s3 : ℕ) (n : ℕ) (h1 : s1 = 115) (h2 : s2 = 118) (h3 : s3 = 115) (h4 : n = 3) :
    (s1 + s2 + s3) / n = 116 :=
by
    sorry

end average_score_l25_25632


namespace remainder_of_product_mod_10_l25_25151

theorem remainder_of_product_mod_10 :
  (1265 * 4233 * 254 * 1729) % 10 = 0 := by
  sorry

end remainder_of_product_mod_10_l25_25151


namespace possible_perimeters_l25_25088

theorem possible_perimeters (a b c: ℝ) (h1: a = 1) (h2: b = 1) 
  (h3: c = 1) (h: ∀ x y z: ℝ, x = y ∧ y = z):
  ∃ x y: ℝ, (x = 8/3 ∧ y = 5/2) := 
  by
    sorry

end possible_perimeters_l25_25088


namespace no_such_natural_number_exists_l25_25814

theorem no_such_natural_number_exists :
  ¬ ∃ n : ℕ, ∃ m : ℕ, 3^n + 2 * 17^n = m^2 :=
by sorry

end no_such_natural_number_exists_l25_25814


namespace other_root_of_quadratic_l25_25549

variable (p : ℝ)

theorem other_root_of_quadratic (h1: 3 * (-2) * r_2 = -6) : r_2 = 1 :=
by
  sorry

end other_root_of_quadratic_l25_25549


namespace solve_equation_l25_25658

theorem solve_equation (n m : ℤ) : 
  n^4 + 2*n^3 + 2*n^2 + 2*n + 1 = m^2 ↔ (n = 0 ∧ (m = 1 ∨ m = -1)) ∨ (n = -1 ∧ m = 0) :=
by sorry

end solve_equation_l25_25658


namespace oldest_person_Jane_babysat_age_l25_25347

def Jane_current_age : ℕ := 32
def Jane_stop_babysitting_age : ℕ := 22 -- 32 - 10
def max_child_age_when_Jane_babysat : ℕ := Jane_stop_babysitting_age / 2  -- 22 / 2
def years_since_Jane_stopped : ℕ := Jane_current_age - Jane_stop_babysitting_age -- 32 - 22

theorem oldest_person_Jane_babysat_age :
  max_child_age_when_Jane_babysat + years_since_Jane_stopped = 21 :=
by
  sorry

end oldest_person_Jane_babysat_age_l25_25347


namespace inflation_two_years_real_rate_of_return_l25_25645

-- Proof Problem for Question 1
theorem inflation_two_years :
  ((1 + 0.015)^2 - 1) * 100 = 3.0225 :=
by
  sorry

-- Proof Problem for Question 2
theorem real_rate_of_return :
  ((1.07 * 1.07) / (1 + 0.030225) - 1) * 100 = 11.13 :=
by
  sorry

end inflation_two_years_real_rate_of_return_l25_25645


namespace option_C_is_quadratic_l25_25602

-- Define the conditions
def option_A (x : ℝ) : Prop := 2 * x = 3
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (4 * x - 3) * (3 * x + 1) = 0
def option_D (x : ℝ) : Prop := (x + 3) * (x - 2) = (x - 2) * (x + 1)

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x, f x = (a * x^2 + b * x + c = 0)) ∧ a ≠ 0

-- The main theorem statement
theorem option_C_is_quadratic : is_quadratic option_C :=
sorry

end option_C_is_quadratic_l25_25602


namespace students_who_saw_l25_25487

variable (B G : ℕ)

theorem students_who_saw (h : B + G = 33) : (2 * G / 3) + (2 * B / 3) = 22 :=
by
  sorry

end students_who_saw_l25_25487


namespace julie_initial_savings_l25_25615

theorem julie_initial_savings (S r : ℝ) 
  (h1 : (S / 2) * r * 2 = 120) 
  (h2 : (S / 2) * ((1 + r)^2 - 1) = 124) : 
  S = 1800 := 
sorry

end julie_initial_savings_l25_25615


namespace membership_fee_increase_each_year_l25_25027

variable (fee_increase : ℕ)

def yearly_membership_fee_increase (first_year_fee sixth_year_fee yearly_increase : ℕ) : Prop :=
  yearly_increase * 5 = sixth_year_fee - first_year_fee

theorem membership_fee_increase_each_year :
  yearly_membership_fee_increase 80 130 10 :=
by
  unfold yearly_membership_fee_increase
  sorry

end membership_fee_increase_each_year_l25_25027


namespace system_solution_l25_25644
-- importing the Mathlib library

-- define the problem with necessary conditions
theorem system_solution (x y : ℝ → ℝ) (x0 y0 : ℝ) 
    (h1 : ∀ t, deriv x t = y t) 
    (h2 : ∀ t, deriv y t = -x t) 
    (h3 : x 0 = x0)
    (h4 : y 0 = y0):
    (∀ t, x t = x0 * Real.cos t + y0 * Real.sin t) ∧ (∀ t, y t = -x0 * Real.sin t + y0 * Real.cos t) ∧ (∀ t, (x t)^2 + (y t)^2 = x0^2 + y0^2) := 
by 
    sorry

end system_solution_l25_25644


namespace units_digit_33_219_89_plus_89_19_l25_25841

theorem units_digit_33_219_89_plus_89_19 :
  let units_digit x := x % 10
  units_digit (33 * 219 ^ 89 + 89 ^ 19) = 8 :=
by
  sorry

end units_digit_33_219_89_plus_89_19_l25_25841


namespace probability_event_A_l25_25155

def probability_of_defective : Real := 0.3
def probability_of_all_defective : Real := 0.027
def probability_of_event_A : Real := 0.973

theorem probability_event_A :
  1 - probability_of_all_defective = probability_of_event_A :=
by
  sorry

end probability_event_A_l25_25155


namespace relationship_of_y_coordinates_l25_25134

theorem relationship_of_y_coordinates (b y1 y2 y3 : ℝ):
  (y1 = 3 * -2.3 + b) → (y2 = 3 * -1.3 + b) → (y3 = 3 * 2.7 + b) → (y1 < y2 ∧ y2 < y3) := 
by 
  intros h1 h2 h3
  sorry

end relationship_of_y_coordinates_l25_25134


namespace region_area_l25_25220

theorem region_area {x y : ℝ} (h : x^2 + y^2 - 4*x + 2*y = -1) : 
  ∃ (r : ℝ), r = 4*pi := 
sorry

end region_area_l25_25220


namespace number_of_integer_length_chords_through_point_l25_25713

theorem number_of_integer_length_chords_through_point 
  (r : ℝ) (d : ℝ) (P_is_5_units_from_center : d = 5) (circle_has_radius_13 : r = 13) :
  ∃ n : ℕ, n = 3 := by
  sorry

end number_of_integer_length_chords_through_point_l25_25713


namespace work_days_together_l25_25453

theorem work_days_together (A_days B_days : ℕ) (work_left_fraction : ℚ) 
  (hA : A_days = 15) (hB : B_days = 20) (h_fraction : work_left_fraction = 8 / 15) : 
  ∃ d : ℕ, d * (1 / 15 + 1 / 20) = 1 - 8 / 15 ∧ d = 4 :=
by
  sorry

end work_days_together_l25_25453


namespace wage_difference_l25_25482

-- Definitions of the problem
variables (P Q h : ℝ)
axiom total_pay : P * h = 480
axiom wage_relation : P = 1.5 * Q
axiom time_relation : Q * (h + 10) = 480

-- Theorem to prove the hourly wage difference
theorem wage_difference : P - Q = 8 :=
by
  sorry

end wage_difference_l25_25482


namespace max_value_product_focal_distances_l25_25136

theorem max_value_product_focal_distances {a b c : ℝ} 
  (h1 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h2 : ∀ x : ℝ, -a ≤ x ∧ x ≤ a) 
  (e : ℝ) :
  (∀ x : ℝ, (a - e * x) * (a + e * x) ≤ a^2) :=
sorry

end max_value_product_focal_distances_l25_25136


namespace factor_theorem_solution_l25_25953

theorem factor_theorem_solution (t : ℝ) :
  (x - t ∣ 3 * x^2 + 10 * x - 8) ↔ (t = 2 / 3 ∨ t = -4) :=
by
  sorry

end factor_theorem_solution_l25_25953


namespace proportion_solution_l25_25049

theorem proportion_solution (x : ℚ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by sorry

end proportion_solution_l25_25049


namespace transform_quadratic_equation_l25_25133

theorem transform_quadratic_equation :
  ∀ x : ℝ, (x^2 - 8 * x - 1 = 0) → ((x - 4)^2 = 17) :=
by
  intro x
  intro h
  sorry

end transform_quadratic_equation_l25_25133


namespace largest_divisor_of_odd_sequence_for_even_n_l25_25877

theorem largest_divisor_of_odd_sequence_for_even_n (n : ℕ) (h : n % 2 = 0) : 
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13) → 105 ∣ k :=
sorry

end largest_divisor_of_odd_sequence_for_even_n_l25_25877


namespace pqrs_product_l25_25806

theorem pqrs_product :
  let P := (Real.sqrt 2010 + Real.sqrt 2009 + Real.sqrt 2008)
  let Q := (-Real.sqrt 2010 - Real.sqrt 2009 + Real.sqrt 2008)
  let R := (Real.sqrt 2010 - Real.sqrt 2009 - Real.sqrt 2008)
  let S := (-Real.sqrt 2010 + Real.sqrt 2009 - Real.sqrt 2008)
  P * Q * R * S = 1 := by
{
  sorry -- Proof is omitted as per the provided instructions.
}

end pqrs_product_l25_25806


namespace find_tax_percentage_l25_25267

noncomputable def net_income : ℝ := 12000
noncomputable def total_income : ℝ := 13000
noncomputable def non_taxable_income : ℝ := 3000
noncomputable def taxable_income : ℝ := total_income - non_taxable_income
noncomputable def tax_percentage (T : ℝ) := total_income - (T * taxable_income)

theorem find_tax_percentage : ∃ T : ℝ, tax_percentage T = net_income :=
by
  sorry

end find_tax_percentage_l25_25267


namespace compute_fg_difference_l25_25181

def f (x : ℕ) : ℕ := x^2 + 3
def g (x : ℕ) : ℕ := 2 * x + 5

theorem compute_fg_difference : f (g 5) - g (f 5) = 167 := by
  sorry

end compute_fg_difference_l25_25181


namespace part_1_part_2_l25_25963

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) ∧ (x^2 - (a + 2) * x + 2 * a = 0)

-- Proposition q: x₁ and x₂ are two real roots of the equation x^2 - 2mx - 3 = 0
def proposition_q (m x₁ x₂ : ℝ) : Prop :=
  x₁ ^ 2 - 2 * m * x₁ - 3 = 0 ∧ x₂ ^ 2 - 2 * m * x₂ - 3 = 0

-- Inequality condition
def inequality_condition (a m x₁ x₂ : ℝ) : Prop :=
  a ^ 2 - 3 * a ≥ abs (x₁ - x₂)

-- Part 1: If proposition p is true, find the range of the real number a
theorem part_1 (a : ℝ) (h_p : proposition_p a) : -1 < a ∧ a < 1 :=
  sorry

-- Part 2: If exactly one of propositions p or q is true, find the range of the real number a
theorem part_2 (a m x₁ x₂ : ℝ) (h_p_or_q : (proposition_p a ∧ ¬(proposition_q m x₁ x₂)) ∨ (¬(proposition_p a) ∧ (proposition_q m x₁ x₂))) : (a < 1) ∨ (a ≥ 4) :=
  sorry

end part_1_part_2_l25_25963


namespace gcd_division_steps_l25_25656

theorem gcd_division_steps (a b : ℕ) (h₁ : a = 1813) (h₂ : b = 333) : 
  ∃ steps : ℕ, steps = 3 ∧ (Nat.gcd a b = 37) :=
by
  have h₁ : a = 1813 := h₁
  have h₂ : b = 333 := h₂
  sorry

end gcd_division_steps_l25_25656


namespace polynomial_factor_l25_25860

theorem polynomial_factor (a b : ℝ) : 
  (∃ c d : ℝ, (5 * c = a) ∧ (5 * d - 3 * c = b) ∧ (2 * c - 3 * d + 25 = 45) ∧ (2 * d - 15 = -18)) 
  → (a = 151.25 ∧ b = -98.25) :=
by
  sorry

end polynomial_factor_l25_25860


namespace find_b_of_square_binomial_l25_25410

theorem find_b_of_square_binomial (b : ℚ) 
  (h : ∃ c : ℚ, ∀ x : ℚ, (3 * x + c) ^ 2 = 9 * x ^ 2 + 21 * x + b) : 
  b = 49 / 4 := 
sorry

end find_b_of_square_binomial_l25_25410


namespace expression_divisible_by_13_l25_25703

theorem expression_divisible_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a ^ 2007 + b ^ 2007 + c ^ 2007 + 2 * 2007 * a * b * c) % 13 = 0 := 
by 
  sorry

end expression_divisible_by_13_l25_25703


namespace tetrahedron_distance_sum_l25_25941

theorem tetrahedron_distance_sum (S₁ S₂ S₃ S₄ H₁ H₂ H₃ H₄ V k : ℝ) 
  (h1 : S₁ = k) (h2 : S₂ = 2 * k) (h3 : S₃ = 3 * k) (h4 : S₄ = 4 * k)
  (V_eq : (1 / 3) * S₁ * H₁ + (1 / 3) * S₂ * H₂ + (1 / 3) * S₃ * H₃ + (1 / 3) * S₄ * H₄ = V) :
  1 * H₁ + 2 * H₂ + 3 * H₃ + 4 * H₄ = (3 * V) / k :=
by
  sorry

end tetrahedron_distance_sum_l25_25941


namespace expected_value_a_squared_is_correct_l25_25572

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end expected_value_a_squared_is_correct_l25_25572


namespace total_profit_correct_l25_25887

def natasha_money : ℤ := 60
def carla_money : ℤ := natasha_money / 3
def cosima_money : ℤ := carla_money / 2
def sergio_money : ℤ := (3 * cosima_money) / 2

def natasha_items : ℤ := 4
def carla_items : ℤ := 6
def cosima_items : ℤ := 5
def sergio_items : ℤ := 3

def natasha_profit_margin : ℚ := 0.10
def carla_profit_margin : ℚ := 0.15
def cosima_sergio_profit_margin : ℚ := 0.12

def natasha_item_cost : ℚ := (natasha_money : ℚ) / natasha_items
def carla_item_cost : ℚ := (carla_money : ℚ) / carla_items
def cosima_item_cost : ℚ := (cosima_money : ℚ) / cosima_items
def sergio_item_cost : ℚ := (sergio_money : ℚ) / sergio_items

def natasha_profit : ℚ := natasha_items * natasha_item_cost * natasha_profit_margin
def carla_profit : ℚ := carla_items * carla_item_cost * carla_profit_margin
def cosima_profit : ℚ := cosima_items * cosima_item_cost * cosima_sergio_profit_margin
def sergio_profit : ℚ := sergio_items * sergio_item_cost * cosima_sergio_profit_margin

def total_profit : ℚ := natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_correct : total_profit = 11.99 := 
by sorry

end total_profit_correct_l25_25887


namespace number_of_two_legged_birds_l25_25647

theorem number_of_two_legged_birds
  (b m i : ℕ)  -- Number of birds (b), mammals (m), and insects (i)
  (h_heads : b + m + i = 300)  -- Condition on total number of heads
  (h_legs : 2 * b + 4 * m + 6 * i = 980)  -- Condition on total number of legs
  : b = 110 :=
by
  sorry

end number_of_two_legged_birds_l25_25647


namespace ryegrass_percentage_l25_25432

theorem ryegrass_percentage (x_ryegrass_percent : ℝ) (y_ryegrass_percent : ℝ) (mixture_x_percent : ℝ)
  (hx : x_ryegrass_percent = 0.40)
  (hy : y_ryegrass_percent = 0.25)
  (hmx : mixture_x_percent = 0.8667) :
  (x_ryegrass_percent * mixture_x_percent + y_ryegrass_percent * (1 - mixture_x_percent)) * 100 = 38 :=
by
  sorry

end ryegrass_percentage_l25_25432


namespace problem_bound_l25_25119

theorem problem_bound (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * (x * y * z) ∧ 
  y * z + z * x + x * y - 2 * (x * y * z) ≤ 7 / 27 :=
sorry

end problem_bound_l25_25119


namespace find_m_if_a_b_parallel_l25_25862

theorem find_m_if_a_b_parallel :
  ∃ m : ℝ, (∃ a : ℝ × ℝ, a = (-2, 1)) ∧ (∃ b : ℝ × ℝ, b = (1, m)) ∧ (m * -2 = 1) ∧ (m = -1 / 2) :=
by
  sorry

end find_m_if_a_b_parallel_l25_25862


namespace find_percentage_l25_25000

theorem find_percentage (P : ℝ) (N : ℝ) (h1 : N = 140) (h2 : (P / 100) * N = (4 / 5) * N - 21) : P = 65 := by
  sorry

end find_percentage_l25_25000


namespace probability_of_red_second_given_red_first_l25_25803

-- Define the conditions as per the problem.
def total_balls := 5
def red_balls := 3
def yellow_balls := 2
def first_draw_red : ℚ := (red_balls : ℚ) / (total_balls : ℚ)
def both_draws_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Define the probability of drawing a red ball in the second draw given the first was red.
def conditional_probability_red_second_given_first : ℚ :=
  both_draws_red / first_draw_red

-- The main statement to be proved.
theorem probability_of_red_second_given_red_first :
  conditional_probability_red_second_given_first = 1 / 2 :=
by
  sorry

end probability_of_red_second_given_red_first_l25_25803


namespace work_done_at_4_pm_l25_25116

noncomputable def workCompletionTime (aHours : ℝ) (bHours : ℝ) (startTime : ℝ) : ℝ :=
  let aRate := 1 / aHours
  let bRate := 1 / bHours
  let cycleWork := aRate + bRate
  let cyclesNeeded := (1 : ℝ) / cycleWork
  startTime + 2 * cyclesNeeded

theorem work_done_at_4_pm :
  workCompletionTime 8 12 6 = 16 :=  -- 16 in 24-hour format is 4 pm
by 
  sorry

end work_done_at_4_pm_l25_25116


namespace min_value_M_l25_25122

theorem min_value_M (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : ∃ a b, M = 3 * a^2 - a * b^2 - 2 * b - 4 ∧ M = 2 := sorry

end min_value_M_l25_25122


namespace both_selected_prob_l25_25955

def ram_prob : ℚ := 6 / 7
def ravi_prob : ℚ := 1 / 5

theorem both_selected_prob : ram_prob * ravi_prob = 6 / 35 := 
by
  sorry

end both_selected_prob_l25_25955


namespace dividend_percentage_l25_25765

theorem dividend_percentage (investment_amount market_value : ℝ) (interest_rate : ℝ) 
  (h1 : investment_amount = 44) (h2 : interest_rate = 12) (h3 : market_value = 33) : 
  ((interest_rate / 100) * investment_amount / market_value) * 100 = 16 := 
by
  sorry

end dividend_percentage_l25_25765


namespace range_of_f_is_0_2_3_l25_25395

def f (x : ℤ) : ℤ := x + 1
def S : Set ℤ := {-1, 1, 2}

theorem range_of_f_is_0_2_3 : Set.image f S = {0, 2, 3} := by
  sorry

end range_of_f_is_0_2_3_l25_25395


namespace proposition_D_correct_l25_25666

theorem proposition_D_correct :
  ∀ x : ℝ, x^2 + x + 2 > 0 :=
by
  sorry

end proposition_D_correct_l25_25666


namespace g_value_l25_25606

theorem g_value (g : ℝ → ℝ)
  (h0 : g 0 = 0)
  (h_mono : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (h_symm : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (h_prop : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 5) = 1 / 2 :=
sorry

end g_value_l25_25606


namespace parallelogram_area_is_correct_l25_25294

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨0, 2, 3⟩
def B : Point3D := ⟨2, 5, 2⟩
def C : Point3D := ⟨-2, 3, 6⟩

noncomputable def vectorAB (A B : Point3D) : Point3D :=
  { x := B.x - A.x
  , y := B.y - A.y
  , z := B.z - A.z 
  }

noncomputable def vectorAC (A C : Point3D) : Point3D :=
  { x := C.x - A.x
  , y := C.y - A.y
  , z := C.z - A.z 
  }

noncomputable def dotProduct (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

noncomputable def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

noncomputable def sinAngle (u v : Point3D) : ℝ :=
  Real.sqrt (1 - (dotProduct u v / (magnitude u * magnitude v)) ^ 2)

noncomputable def parallelogramArea (u v : Point3D) : ℝ :=
  magnitude u * magnitude v * sinAngle u v

theorem parallelogram_area_is_correct :
  parallelogramArea (vectorAB A B) (vectorAC A C) = 6 * Real.sqrt 5 := by
  sorry

end parallelogram_area_is_correct_l25_25294


namespace prime_half_sum_l25_25433

theorem prime_half_sum
  (a b c : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : Nat.Prime (a.factorial + b + c))
  (h2 : Nat.Prime (b.factorial + c + a))
  (h3 : Nat.Prime (c.factorial + a + b)) :
  Nat.Prime ((a + b + c + 1) / 2) := 
sorry

end prime_half_sum_l25_25433


namespace simplify_polynomial_l25_25962

theorem simplify_polynomial (s : ℝ) :
  (2 * s ^ 2 + 5 * s - 3) - (2 * s ^ 2 + 9 * s - 6) = -4 * s + 3 :=
by 
  sorry

end simplify_polynomial_l25_25962


namespace find_number_l25_25646

theorem find_number (x : ℕ) (h : (537 - x) / (463 + x) = 1 / 9) : x = 437 :=
sorry

end find_number_l25_25646


namespace commute_time_difference_l25_25219

-- Define the conditions as constants
def distance_to_work : ℝ := 1.5
def walking_speed : ℝ := 3
def train_speed : ℝ := 20
def additional_train_time_minutes : ℝ := 10.5

-- The main proof problem
theorem commute_time_difference : 
  (distance_to_work / walking_speed * 60) - 
  ((distance_to_work / train_speed * 60) + additional_train_time_minutes) = 15 :=
by
  sorry

end commute_time_difference_l25_25219


namespace perfect_square_as_sum_of_powers_of_2_l25_25855

theorem perfect_square_as_sum_of_powers_of_2 (n a b : ℕ) (h : n^2 = 2^a + 2^b) (hab : a ≥ b) :
  (∃ k : ℕ, n^2 = 4^(k + 1)) ∨ (∃ k : ℕ, n^2 = 9 * 4^k) :=
by
  sorry

end perfect_square_as_sum_of_powers_of_2_l25_25855


namespace find_x_average_is_60_l25_25190

theorem find_x_average_is_60 : 
  ∃ x : ℕ, (54 + 55 + 57 + 58 + 59 + 62 + 62 + 63 + x) / 9 = 60 ∧ x = 70 :=
by
  existsi 70
  sorry

end find_x_average_is_60_l25_25190


namespace four_consecutive_integers_product_2520_l25_25933

theorem four_consecutive_integers_product_2520 {a b c d : ℕ}
  (h1 : a + 1 = b) 
  (h2 : b + 1 = c) 
  (h3 : c + 1 = d) 
  (h4 : a * b * c * d = 2520) : 
  a = 6 := 
sorry

end four_consecutive_integers_product_2520_l25_25933


namespace johns_age_l25_25749

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 70) : j = 20 := by
  sorry

end johns_age_l25_25749


namespace sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l25_25042

noncomputable def y (x m : ℝ) : ℝ := x^2 + m / x
noncomputable def y_prime (x m : ℝ) : ℝ := 2 * x - m / x^2

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x ≥ 1, y_prime x m ≥ 0) ↔ m ≤ 2 :=
sorry  -- Proof skipped as instructed

-- Now, state that m < 1 is a sufficient but not necessary condition
theorem m_sufficient_but_not_necessary (m : ℝ) :
  m < 1 → (∀ x ≥ 1, y_prime x m ≥ 0) :=
sorry  -- Proof skipped as instructed

end sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l25_25042


namespace negation_of_exists_abs_le_two_l25_25431

theorem negation_of_exists_abs_le_two :
  (¬ ∃ x : ℝ, |x| ≤ 2) ↔ (∀ x : ℝ, |x| > 2) :=
by
  sorry

end negation_of_exists_abs_le_two_l25_25431


namespace fraction_solution_l25_25126

theorem fraction_solution (x : ℝ) (h : 4 - 9 / x + 4 / x^2 = 0) : 3 / x = 12 ∨ 3 / x = 3 / 4 :=
by
  -- Proof to be written here
  sorry

end fraction_solution_l25_25126


namespace remainder_when_expr_divided_by_9_l25_25412

theorem remainder_when_expr_divided_by_9 (n m p : ℤ)
  (h1 : n % 18 = 10)
  (h2 : m % 27 = 16)
  (h3 : p % 6 = 4) :
  (2 * n + 3 * m - p) % 9 = 1 := 
sorry

end remainder_when_expr_divided_by_9_l25_25412


namespace different_types_of_players_l25_25826

theorem different_types_of_players :
  ∀ (cricket hockey football softball : ℕ) (total_players : ℕ),
    cricket = 12 → hockey = 17 → football = 11 → softball = 10 → total_players = 50 →
    cricket + hockey + football + softball = total_players → 
    4 = 4 :=
by
  intros
  rfl

end different_types_of_players_l25_25826


namespace polynomial_terms_equal_l25_25249

theorem polynomial_terms_equal (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (h : p + q = 1) :
  (9 * p^8 * q = 36 * p^7 * q^2) → p = 4 / 5 :=
by
  sorry

end polynomial_terms_equal_l25_25249


namespace arthur_spent_on_second_day_l25_25086

variable (H D : ℝ)
variable (a1 : 3 * H + 4 * D = 10)
variable (a2 : D = 1)

theorem arthur_spent_on_second_day :
  2 * H + 3 * D = 7 :=
by
  sorry

end arthur_spent_on_second_day_l25_25086


namespace squares_difference_l25_25486

theorem squares_difference (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 3) : a^2 - b^2 = 15 :=
by
  sorry

end squares_difference_l25_25486


namespace shopper_saves_more_l25_25849

-- Definitions and conditions
def cover_price : ℝ := 30
def percent_discount : ℝ := 0.25
def dollar_discount : ℝ := 5
def first_discounted_price : ℝ := cover_price * (1 - percent_discount)
def second_discounted_price : ℝ := first_discounted_price - dollar_discount
def first_dollar_discounted_price : ℝ := cover_price - dollar_discount
def second_percent_discounted_price : ℝ := first_dollar_discounted_price * (1 - percent_discount)

def additional_savings : ℝ := second_percent_discounted_price - second_discounted_price

-- Theorem stating the shopper saves 125 cents more with 25% first
theorem shopper_saves_more : additional_savings = 1.25 := by
  sorry

end shopper_saves_more_l25_25849


namespace book_cost_l25_25452

theorem book_cost (n_5 n_3 : ℕ) (N : ℕ) :
  (N = n_5 + n_3) ∧ (N > 10) ∧ (N < 20) ∧ (5 * n_5 = 3 * n_3) →  5 * n_5 = 30 := 
sorry

end book_cost_l25_25452


namespace range_of_target_function_l25_25345

noncomputable def target_function (x : ℝ) : ℝ :=
  1 - 1 / (x^2 - 1)

theorem range_of_target_function :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ target_function x = y ↔ y ∈ (Set.Iio 1 ∪ Set.Ici 2) :=
by
  sorry

end range_of_target_function_l25_25345


namespace range_positive_of_odd_increasing_l25_25690

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Define f as an increasing function on (-∞,0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Given an odd function that is increasing on (-∞,0) and f(-1) = 0, prove the range of x for which f(x) > 0 is (-1, 0) ∪ (1, +∞)
theorem range_positive_of_odd_increasing (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_increasing : is_increasing_on_neg f)
  (h_f_neg_one : f (-1) = 0) :
  {x : ℝ | f x > 0} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end range_positive_of_odd_increasing_l25_25690


namespace minimize_tank_construction_cost_l25_25979

noncomputable def minimum_cost (l w h : ℝ) (P_base P_wall : ℝ) : ℝ :=
  P_base * (l * w) + P_wall * (2 * h * (l + w))

theorem minimize_tank_construction_cost :
  ∃ l w : ℝ, l * w = 9 ∧ l = w ∧
  minimum_cost l w 2 200 150 = 5400 :=
by
  sorry

end minimize_tank_construction_cost_l25_25979


namespace MarkBenchPressAmount_l25_25196

def DaveWeight : ℝ := 175
def DaveBenchPressMultiplier : ℝ := 3
def CraigBenchPressFraction : ℝ := 0.20
def MarkDeficitFromCraig : ℝ := 50

theorem MarkBenchPressAmount : 
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  MarkBenchPress = 55 := by
  let DaveBenchPress := DaveWeight * DaveBenchPressMultiplier
  let CraigBenchPress := DaveBenchPress * CraigBenchPressFraction
  let MarkBenchPress := CraigBenchPress - MarkDeficitFromCraig
  sorry

end MarkBenchPressAmount_l25_25196


namespace principal_amount_l25_25158

theorem principal_amount
  (P : ℝ)
  (r : ℝ := 0.05)
  (t : ℝ := 2)
  (H : P * (1 + r)^t - P - P * r * t = 17) :
  P = 6800 :=
by sorry

end principal_amount_l25_25158


namespace neg_p_iff_a_in_0_1_l25_25131

theorem neg_p_iff_a_in_0_1 (a : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) ∧ (0 < a ∧ a < 1) :=
sorry

end neg_p_iff_a_in_0_1_l25_25131


namespace lcm_36_100_l25_25721

theorem lcm_36_100 : Nat.lcm 36 100 = 900 :=
by
  sorry

end lcm_36_100_l25_25721


namespace complex_number_on_line_l25_25242

theorem complex_number_on_line (a : ℝ) (h : (3 : ℝ) = (a - 1) + 2) : a = 2 :=
by
  sorry

end complex_number_on_line_l25_25242


namespace find_a6_l25_25005

noncomputable def a_n (n : ℕ) : ℝ := sorry
noncomputable def S_n (n : ℕ) : ℝ := sorry
noncomputable def r : ℝ := sorry

axiom h_pos : ∀ n, a_n n > 0
axiom h_s3 : S_n 3 = 14
axiom h_a3 : a_n 3 = 8

theorem find_a6 : a_n 6 = 64 := by sorry

end find_a6_l25_25005


namespace total_paths_A_to_D_l25_25405

-- Given conditions
def paths_from_A_to_B := 2
def paths_from_B_to_C := 2
def paths_from_C_to_D := 2
def direct_path_A_to_C := 1
def direct_path_B_to_D := 1

-- Proof statement
theorem total_paths_A_to_D : 
  paths_from_A_to_B * paths_from_B_to_C * paths_from_C_to_D + 
  direct_path_A_to_C * paths_from_C_to_D + 
  paths_from_A_to_B * direct_path_B_to_D = 12 := 
  by
    sorry

end total_paths_A_to_D_l25_25405


namespace sqrt_x_minus_2_meaningful_l25_25051

theorem sqrt_x_minus_2_meaningful (x : ℝ) (h : 0 ≤ x - 2) : 2 ≤ x :=
by sorry

end sqrt_x_minus_2_meaningful_l25_25051


namespace angle_A_measure_triangle_area_l25_25781

variable {a b c : ℝ} 
variable {A B C : ℝ} 
variable (triangle : a^2 = b^2 + c^2 - 2 * b * c * (Real.cos A))

theorem angle_A_measure (h : (b - c)^2 = a^2 - b * c) : A = Real.pi / 3 :=
sorry

theorem triangle_area 
  (h1 : a = 3) 
  (h2 : Real.sin C = 2 * Real.sin B) 
  (h3 : A = Real.pi / 3) 
  (hb : b = Real.sqrt 3)
  (hc : c = 2 * Real.sqrt 3) : 
  (1/2) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
sorry

end angle_A_measure_triangle_area_l25_25781


namespace coord_of_point_M_in_third_quadrant_l25_25111

noncomputable def point_coordinates (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0 ∧ abs y = 1 ∧ abs x = 2

theorem coord_of_point_M_in_third_quadrant : 
  ∃ (x y : ℝ), point_coordinates x y ∧ (x, y) = (-2, -1) := 
by {
  sorry
}

end coord_of_point_M_in_third_quadrant_l25_25111


namespace system_of_equations_solution_l25_25807

theorem system_of_equations_solution (x y z u v : ℤ) 
  (h1 : x + y + z + u = 5)
  (h2 : y + z + u + v = 1)
  (h3 : z + u + v + x = 2)
  (h4 : u + v + x + y = 0)
  (h5 : v + x + y + z = 4) :
  v = -2 ∧ x = 2 ∧ y = 1 ∧ z = 3 ∧ u = -1 := 
by 
  sorry

end system_of_equations_solution_l25_25807


namespace total_distance_l25_25620

theorem total_distance (D : ℝ) 
  (h₁ : 60 * (D / 2 / 60) = D / 2) 
  (h₂ : 40 * ((D / 2) / 4 / 40) = D / 8) 
  (h₃ : 50 * (105 / 50) = 105)
  (h₄ : D = D / 2 + D / 8 + 105) : 
  D = 280 :=
by sorry

end total_distance_l25_25620


namespace min_sum_factors_l25_25873

theorem min_sum_factors (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_prod : a * b * c = 2310) : a + b + c = 40 :=
sorry

end min_sum_factors_l25_25873


namespace min_value_expression_l25_25621

noncomputable def sinSquare (θ : ℝ) : ℝ :=
  Real.sin (θ) ^ 2

theorem min_value_expression (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : θ₁ > 0) (h₂ : θ₂ > 0) (h₃ : θ₃ > 0) (h₄ : θ₄ > 0)
  (sum_eq_pi : θ₁ + θ₂ + θ₃ + θ₄ = Real.pi) :
  (2 * sinSquare θ₁ + 1 / sinSquare θ₁) *
  (2 * sinSquare θ₂ + 1 / sinSquare θ₂) *
  (2 * sinSquare θ₃ + 1 / sinSquare θ₃) *
  (2 * sinSquare θ₄ + 1 / sinSquare θ₁) ≥ 81 := 
by
  sorry

end min_value_expression_l25_25621


namespace sum_x_y_eq_8_l25_25694

theorem sum_x_y_eq_8 (x y S : ℝ) (h1 : x + y = S) (h2 : y - 3 * x = 7) (h3 : y - x = 7.5) : S = 8 :=
by
  sorry

end sum_x_y_eq_8_l25_25694


namespace petes_average_speed_is_correct_l25_25090

-- Definition of the necessary constants
def map_distance := 5.0 -- inches
def scale := 0.023809523809523808 -- inches per mile
def travel_time := 3.5 -- hours

-- The real distance calculation based on the given map scale
def real_distance := map_distance / scale -- miles

-- Proving the average speed calculation
def average_speed := real_distance / travel_time -- miles per hour

-- Theorem statement: Pete's average speed calculation is correct
theorem petes_average_speed_is_correct : average_speed = 60 :=
by
  -- Proof outline
  -- The real distance is 5 / 0.023809523809523808 ≈ 210
  -- The average speed is 210 / 3.5 ≈ 60
  sorry

end petes_average_speed_is_correct_l25_25090


namespace crocus_bulb_cost_l25_25201

theorem crocus_bulb_cost 
  (space_bulbs : ℕ)
  (crocus_bulbs : ℕ)
  (cost_daffodil_bulb : ℝ)
  (budget : ℝ)
  (purchased_crocus_bulbs : ℕ)
  (total_cost : ℝ)
  (c : ℝ)
  (h_space : space_bulbs = 55)
  (h_cost_daffodil : cost_daffodil_bulb = 0.65)
  (h_budget : budget = 29.15)
  (h_purchased_crocus : purchased_crocus_bulbs = 22)
  (h_total_cost_eq : total_cost = (33:ℕ) * cost_daffodil_bulb)
  (h_eqn : (purchased_crocus_bulbs : ℝ) * c + total_cost = budget) :
  c = 0.35 :=
by 
  sorry

end crocus_bulb_cost_l25_25201


namespace additional_workers_needed_l25_25499

theorem additional_workers_needed :
  let initial_workers := 4
  let initial_parts := 108
  let initial_hours := 3
  let target_parts := 504
  let target_hours := 8
  (target_parts / target_hours) / (initial_parts / (initial_hours * initial_workers)) - initial_workers = 3 := by
  sorry

end additional_workers_needed_l25_25499


namespace correct_option_a_l25_25404

theorem correct_option_a (x y a b : ℝ) : 3 * x - 2 * x = x :=
by sorry

end correct_option_a_l25_25404


namespace cos_pi_div_3_l25_25491

theorem cos_pi_div_3 : Real.cos (π / 3) = 1 / 2 := 
by
  sorry

end cos_pi_div_3_l25_25491


namespace smallest_x_mod_7_one_sq_l25_25766

theorem smallest_x_mod_7_one_sq (x : ℕ) (h : 1 < x) (hx : (x * x) % 7 = 1) : x = 6 :=
  sorry

end smallest_x_mod_7_one_sq_l25_25766


namespace C_share_of_profit_l25_25821

-- Given conditions
def investment_A : ℕ := 8000
def investment_B : ℕ := 4000
def investment_C : ℕ := 2000
def total_profit : ℕ := 252000

-- Objective to prove that C's share of the profit is given by 36000
theorem C_share_of_profit : (total_profit / (investment_A / investment_C + investment_B / investment_C + 1)) = 36000 :=
by
  sorry

end C_share_of_profit_l25_25821


namespace opposite_of_5_is_neg_5_l25_25240

def opposite_number (x y : ℤ) : Prop := x + y = 0

theorem opposite_of_5_is_neg_5 : opposite_number 5 (-5) := by
  sorry

end opposite_of_5_is_neg_5_l25_25240


namespace correct_operations_l25_25312

theorem correct_operations :
  (∀ {a b : ℝ}, -(-a + b) = a + b → False) ∧
  (∀ {a : ℝ}, 3 * a^3 - 3 * a^2 = a → False) ∧
  (∀ {x : ℝ}, (x^6)^2 = x^8 → False) ∧
  (∀ {z : ℝ}, 1 / (2 / 3 : ℝ)⁻¹ = 2 / 3) :=
by
  sorry

end correct_operations_l25_25312


namespace proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l25_25399

variable (a b : ℝ)

-- Condition
def condition : Prop :=
  (1 / a) - (1 / b) = 1 / (a + b)

-- Proof statement
theorem proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab (h : condition a b) :
  (1 / a^2) - (1 / b^2) = 1 / (a * b) :=
sorry

end proof_1_over_a_squared_sub_1_over_b_squared_eq_1_over_ab_l25_25399


namespace instantaneous_velocity_at_t4_l25_25864

-- Definition of the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- The proof problem statement: Proving that the derivative of s at t = 4 is 7
theorem instantaneous_velocity_at_t4 : deriv s 4 = 7 :=
by sorry

end instantaneous_velocity_at_t4_l25_25864


namespace math_competition_probs_l25_25697

-- Definitions related to the problem conditions
def boys : ℕ := 3
def girls : ℕ := 3
def total_students := boys + girls
def total_combinations := (total_students.choose 2)

-- Definition of the probabilities
noncomputable def prob_exactly_one_boy : ℚ := 0.6
noncomputable def prob_at_least_one_boy : ℚ := 0.8
noncomputable def prob_at_most_one_boy : ℚ := 0.8

-- Lean statement for the proof problem
theorem math_competition_probs :
  prob_exactly_one_boy = 0.6 ∧
  prob_at_least_one_boy = 0.8 ∧
  prob_at_most_one_boy = 0.8 :=
by
  sorry

end math_competition_probs_l25_25697


namespace power_vs_square_l25_25293

theorem power_vs_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end power_vs_square_l25_25293


namespace sin_identity_l25_25891

theorem sin_identity (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
  by 
  sorry

end sin_identity_l25_25891


namespace num_girls_on_playground_l25_25865

-- Definitions based on conditions
def total_students : ℕ := 20
def classroom_students := total_students / 4
def playground_students := total_students - classroom_students
def boys_playground := playground_students / 3
def girls_playground := playground_students - boys_playground

-- Theorem statement
theorem num_girls_on_playground : girls_playground = 10 :=
by
  -- Begin preparing proofs
  sorry

end num_girls_on_playground_l25_25865


namespace y1_gt_y2_for_line_through_points_l25_25990

theorem y1_gt_y2_for_line_through_points (x1 y1 x2 y2 k b : ℝ) 
  (h_line_A : y1 = k * x1 + b) 
  (h_line_B : y2 = k * x2 + b) 
  (h_k_neq_0 : k ≠ 0)
  (h_k_pos : k > 0)
  (h_b_nonneg : b ≥ 0)
  (h_x1_gt_x2 : x1 > x2) : 
  y1 > y2 := 
  sorry

end y1_gt_y2_for_line_through_points_l25_25990


namespace two_digit_sum_condition_l25_25902

theorem two_digit_sum_condition (x y : ℕ) (hx : 1 ≤ x) (hx9 : x ≤ 9) (hy : 0 ≤ y) (hy9 : y ≤ 9)
    (h : (x + 1) + (y + 2) - 10 = 2 * (x + y)) :
    (x = 6 ∧ y = 8) ∨ (x = 5 ∧ y = 9) :=
sorry

end two_digit_sum_condition_l25_25902


namespace speed_of_man_in_still_water_l25_25420

variable (v_m v_s : ℝ)

-- Conditions as definitions 
def downstream_distance_eq : Prop :=
  36 = (v_m + v_s) * 3

def upstream_distance_eq : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_of_man_in_still_water (h1 : downstream_distance_eq v_m v_s) (h2 : upstream_distance_eq v_m v_s) : v_m = 9 := 
  by
  sorry

end speed_of_man_in_still_water_l25_25420


namespace LindasOriginalSavings_l25_25189

theorem LindasOriginalSavings : 
  (∃ S : ℝ, (1 / 4) * S = 200) ∧ 
  (3 / 4) * S = 600 ∧ 
  (∀ F : ℝ, 0.80 * F = 600 → F = 750) → 
  S = 800 :=
by
  sorry

end LindasOriginalSavings_l25_25189


namespace students_distribute_l25_25415

theorem students_distribute (x y : ℕ) (h₁ : x + y = 4200)
        (h₂ : x * 108 / 100 + y * 111 / 100 = 4620) :
    x = 1400 ∧ y = 2800 :=
by
  sorry

end students_distribute_l25_25415


namespace percentage_of_import_tax_l25_25643

noncomputable def total_value : ℝ := 2560
noncomputable def taxable_threshold : ℝ := 1000
noncomputable def import_tax : ℝ := 109.20

theorem percentage_of_import_tax :
  let excess_value := total_value - taxable_threshold
  let percentage_tax := (import_tax / excess_value) * 100
  percentage_tax = 7 := 
by
  sorry

end percentage_of_import_tax_l25_25643


namespace amanda_pay_if_not_finished_l25_25780

-- Define Amanda's hourly rate and daily work hours.
def amanda_hourly_rate : ℝ := 50
def amanda_daily_hours : ℝ := 10

-- Define the percentage of pay Jose will withhold.
def withholding_percentage : ℝ := 0.20

-- Define Amanda's total pay if she finishes the sales report.
def amanda_total_pay : ℝ := amanda_hourly_rate * amanda_daily_hours

-- Define the amount withheld if she does not finish the sales report.
def withheld_amount : ℝ := amanda_total_pay * withholding_percentage

-- Define the amount Amanda will receive if she does not finish the sales report.
def amanda_final_pay_not_finished : ℝ := amanda_total_pay - withheld_amount

-- The theorem to prove:
theorem amanda_pay_if_not_finished : amanda_final_pay_not_finished = 400 := by
  sorry

end amanda_pay_if_not_finished_l25_25780


namespace value_of_f_prime_at_2_l25_25146

theorem value_of_f_prime_at_2 :
  ∃ (f' : ℝ → ℝ), 
  (∀ (x : ℝ), f' x = 2 * x + 3 * f' 2 + 1 / x) →
  f' 2 = - (9 / 4) := 
by 
  sorry

end value_of_f_prime_at_2_l25_25146


namespace complex_fraction_l25_25991

theorem complex_fraction (h : (1 : ℂ) - I = 1 - (I : ℂ)) :
  ((1 - I) * (1 - (2 * I))) / (1 + I) = -2 - I := 
by
  sorry

end complex_fraction_l25_25991


namespace express_in_scientific_notation_l25_25624

theorem express_in_scientific_notation :
  (10.58 * 10^9) = 1.058 * 10^10 :=
by
  sorry

end express_in_scientific_notation_l25_25624


namespace solution_set_l25_25387

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (2 - x)

theorem solution_set:
  ∀ x : ℝ, x > -1 ∧ x < 1/3 → f (2*x + 1) < f x := 
by
  sorry

end solution_set_l25_25387


namespace operation_is_double_l25_25175

theorem operation_is_double (x : ℝ) (operation : ℝ → ℝ) (h1: x^2 = 25) (h2: operation x = x / 5 + 9) : operation x = 2 * x :=
by
  sorry

end operation_is_double_l25_25175


namespace smallest_multiple_9_11_13_l25_25931

theorem smallest_multiple_9_11_13 : ∃ n : ℕ, n > 0 ∧ (9 ∣ n) ∧ (11 ∣ n) ∧ (13 ∣ n) ∧ n = 1287 := 
 by {
   sorry
 }

end smallest_multiple_9_11_13_l25_25931


namespace meera_fraction_4kmh_l25_25157

noncomputable def fraction_of_time_at_4kmh (total_time : ℝ) (x : ℝ) : ℝ :=
  x / total_time

theorem meera_fraction_4kmh (total_time x : ℝ) (h1 : x = total_time / 14) :
  fraction_of_time_at_4kmh total_time x = 1 / 14 :=
by
  sorry

end meera_fraction_4kmh_l25_25157


namespace max_profit_at_boundary_l25_25498

noncomputable def profit (x : ℝ) : ℝ :=
  -50 * (x - 55) ^ 2 + 11250

def within_bounds (x : ℝ) : Prop :=
  40 ≤ x ∧ x ≤ 52

theorem max_profit_at_boundary :
  within_bounds 52 ∧ 
  (∀ x : ℝ, within_bounds x → profit x ≤ profit 52) :=
by
  sorry

end max_profit_at_boundary_l25_25498


namespace quadratic_inequality_solution_l25_25394

theorem quadratic_inequality_solution :
  { m : ℝ // ∀ x : ℝ, m * x^2 - 6 * m * x + 5 * m + 1 > 0 } = { m : ℝ // 0 ≤ m ∧ m < 1/4 } :=
sorry

end quadratic_inequality_solution_l25_25394


namespace randy_gave_sally_l25_25326

-- Define the given conditions
def initial_amount_randy : ℕ := 3000
def smith_contribution : ℕ := 200
def amount_kept_by_randy : ℕ := 2000

-- The total amount Randy had after Smith's contribution
def total_amount_randy : ℕ := initial_amount_randy + smith_contribution

-- The amount of money Randy gave to Sally
def amount_given_to_sally : ℕ := total_amount_randy - amount_kept_by_randy

-- The theorem statement: Given the conditions, prove that Randy gave Sally $1,200
theorem randy_gave_sally : amount_given_to_sally = 1200 :=
by
  sorry

end randy_gave_sally_l25_25326


namespace verify_drawn_numbers_when_x_is_24_possible_values_of_x_l25_25470

-- Population size and group division
def population_size := 1000
def number_of_groups := 10
def group_size := population_size / number_of_groups

-- Systematic sampling function
def systematic_sample (x : ℕ) (k : ℕ) : ℕ :=
  (x + 33 * k) % 1000

-- Prove the drawn 10 numbers when x = 24
theorem verify_drawn_numbers_when_x_is_24 :
  (∃ drawn_numbers, drawn_numbers = [24, 157, 290, 323, 456, 589, 622, 755, 888, 921]) :=
  sorry

-- Prove possible values of x given last two digits equal to 87
theorem possible_values_of_x (k : ℕ) (h : k < number_of_groups) :
  (∃ x_values, x_values = [87, 54, 21, 88, 55, 22, 89, 56, 23, 90]) :=
  sorry

end verify_drawn_numbers_when_x_is_24_possible_values_of_x_l25_25470


namespace tank_capacity_l25_25856

theorem tank_capacity (C : ℝ) (h1 : 0.40 * C = 0.90 * C - 36) : C = 72 := 
sorry

end tank_capacity_l25_25856


namespace mass_percentage_Br_HBrO3_l25_25130

theorem mass_percentage_Br_HBrO3 (molar_mass_H : ℝ) (molar_mass_Br : ℝ) (molar_mass_O : ℝ)
  (molar_mass_HBrO3 : ℝ) (mass_percentage_H : ℝ) (mass_percentage_Br : ℝ) :
  molar_mass_H = 1.01 →
  molar_mass_Br = 79.90 →
  molar_mass_O = 16.00 →
  molar_mass_HBrO3 = molar_mass_H + molar_mass_Br + 3 * molar_mass_O →
  mass_percentage_H = 0.78 →
  mass_percentage_Br = (molar_mass_Br / molar_mass_HBrO3) * 100 → 
  mass_percentage_Br = 61.98 :=
sorry

end mass_percentage_Br_HBrO3_l25_25130


namespace minimum_value_expr_l25_25140

noncomputable def expr (x : ℝ) : ℝ := (x^2 + 11) / Real.sqrt (x^2 + 5)

theorem minimum_value_expr : ∃ x : ℝ, expr x = 2 * Real.sqrt 6 :=
by
  sorry

end minimum_value_expr_l25_25140


namespace regular_tetrahedron_height_eq_4r_l25_25441

noncomputable def equilateral_triangle_inscribed_circle_height (r : ℝ) : ℝ :=
3 * r

noncomputable def regular_tetrahedron_inscribed_sphere_height (r : ℝ) : ℝ :=
4 * r

theorem regular_tetrahedron_height_eq_4r (r : ℝ) :
  regular_tetrahedron_inscribed_sphere_height r = 4 * r :=
by
  unfold regular_tetrahedron_inscribed_sphere_height
  sorry

end regular_tetrahedron_height_eq_4r_l25_25441


namespace quadratic_roots_relation_l25_25844

noncomputable def roots_relation (a b c : ℝ) : Prop :=
  ∃ α β : ℝ, (α * β = c / a) ∧ (α + β = -b / a) ∧ β = 3 * α

theorem quadratic_roots_relation (a b c : ℝ) (h : roots_relation a b c) : 3 * b^2 = 16 * a * c :=
by
  sorry

end quadratic_roots_relation_l25_25844


namespace axis_symmetry_shifted_graph_l25_25912

open Real

theorem axis_symmetry_shifted_graph :
  ∀ k : ℤ, ∃ x : ℝ, (y = 2 * sin (2 * x)) ∧
  y = 2 * sin (2 * (x + π / 12)) ↔
  x = k * π / 2 + π / 6 :=
sorry

end axis_symmetry_shifted_graph_l25_25912


namespace abc_inequality_l25_25417

-- Define a mathematical statement to encapsulate the problem
theorem abc_inequality (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by sorry

end abc_inequality_l25_25417


namespace sum_powers_of_i_l25_25519

variable (n : ℕ) (i : ℂ) (h_multiple_of_6 : n % 6 = 0) (h_i : i^2 = -1)

theorem sum_powers_of_i (h_n6 : n = 6) :
    1 + 2*i + 3*i^2 + 4*i^3 + 5*i^4 + 6*i^5 + 7*i^6 = 6*i - 7 := by
  sorry

end sum_powers_of_i_l25_25519


namespace root_equation_l25_25104

variable (m : ℝ)
theorem root_equation (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2020 = 2023 := by
  sorry

end root_equation_l25_25104


namespace fourth_term_of_geometric_sequence_is_320_l25_25522

theorem fourth_term_of_geometric_sequence_is_320
  (a : ℕ) (r : ℕ)
  (h_a : a = 5)
  (h_fifth_term : a * r^4 = 1280) :
  a * r^3 = 320 := 
by
  sorry

end fourth_term_of_geometric_sequence_is_320_l25_25522


namespace john_website_days_l25_25546

theorem john_website_days
  (monthly_visits : ℕ)
  (cents_per_visit : ℝ)
  (dollars_per_day : ℝ)
  (monthly_visits_eq : monthly_visits = 30000)
  (cents_per_visit_eq : cents_per_visit = 0.01)
  (dollars_per_day_eq : dollars_per_day = 10) :
  (monthly_visits / (dollars_per_day / cents_per_visit)) = 30 :=
by
  sorry

end john_website_days_l25_25546


namespace find_k_l25_25661

theorem find_k (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) + (-1) * a * b * c :=
by
  sorry

end find_k_l25_25661


namespace triangles_in_figure_l25_25890

-- Define the conditions of the problem.
def bottom_row_small := 4
def next_row_small := 3
def following_row_small := 2
def topmost_row_small := 1

def small_triangles := bottom_row_small + next_row_small + following_row_small + topmost_row_small

def medium_triangles := 3
def large_triangle := 1

def total_triangles := small_triangles + medium_triangles + large_triangle

-- Lean proof statement that the total number of triangles is 14
theorem triangles_in_figure : total_triangles = 14 :=
by
  unfold total_triangles
  unfold small_triangles
  unfold bottom_row_small next_row_small following_row_small topmost_row_small
  unfold medium_triangles large_triangle
  sorry

end triangles_in_figure_l25_25890


namespace seating_arrangements_l25_25604

def count_arrangements (n k : ℕ) : ℕ :=
  (n.factorial) / (n - k).factorial

theorem seating_arrangements : count_arrangements 6 5 * 3 = 360 :=
  sorry

end seating_arrangements_l25_25604


namespace ratio_first_term_common_difference_l25_25207

theorem ratio_first_term_common_difference
  (a d : ℚ)
  (h : (15 / 2) * (2 * a + 14 * d) = 4 * (8 / 2) * (2 * a + 7 * d)) :
  a / d = -7 / 17 := 
by {
  sorry
}

end ratio_first_term_common_difference_l25_25207


namespace triangle_inequality_l25_25480

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2)
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0)
  (h5 : a < b + c) (h6 : b < a + c) (h7 : c < a + b) :
  a^2 + b^2 + c^2 + 2 * a * b * c < 2 := 
sorry

end triangle_inequality_l25_25480


namespace quadratic_real_roots_range_find_k_l25_25382

theorem quadratic_real_roots_range (k : ℝ) (h : ∃ x1 x2 : ℝ, x^2 - 2 * (k - 1) * x + k^2 = 0):
  k ≤ 1/2 :=
  sorry

theorem find_k (k : ℝ) (x1 x2 : ℝ) (h₁ : x^2 - 2 * (k - 1) * x + k^2 = 0)
  (h₂ : x₁ * x₂ + x₁ + x₂ - 1 = 0) (h_range : k ≤ 1/2) :
    k = -3 :=
  sorry

end quadratic_real_roots_range_find_k_l25_25382


namespace rectangle_area_l25_25286

theorem rectangle_area {A_s A_r : ℕ} (s l w : ℕ) (h1 : A_s = 36) (h2 : A_s = s * s)
  (h3 : w = s) (h4 : l = 3 * w) (h5 : A_r = w * l) : A_r = 108 :=
by
  sorry

end rectangle_area_l25_25286


namespace multiplier_is_3_l25_25839

theorem multiplier_is_3 (x : ℝ) (num : ℝ) (difference : ℝ) (h1 : num = 15.0) (h2 : difference = 40) (h3 : x * num - 5 = difference) : x = 3 := 
by 
  sorry

end multiplier_is_3_l25_25839


namespace avg_age_of_coaches_l25_25939

theorem avg_age_of_coaches (n_girls n_boys n_coaches : ℕ)
  (avg_age_girls avg_age_boys avg_age_members : ℕ)
  (h_girls : n_girls = 30)
  (h_boys : n_boys = 15)
  (h_coaches : n_coaches = 5)
  (h_avg_age_girls : avg_age_girls = 18)
  (h_avg_age_boys : avg_age_boys = 19)
  (h_avg_age_members : avg_age_members = 20) :
  (n_girls * avg_age_girls + n_boys * avg_age_boys + n_coaches * 35) / (n_girls + n_boys + n_coaches) = avg_age_members :=
by sorry

end avg_age_of_coaches_l25_25939


namespace f_pi_over_4_l25_25920

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

theorem f_pi_over_4 (ω φ : ℝ) (h : ω ≠ 0) 
  (symm : ∀ x, f ω φ (π / 4 + x) = f ω φ (π / 4 - x)) : 
  f ω φ (π / 4) = 2 ∨ f ω φ (π / 4) = -2 := 
by 
  sorry

end f_pi_over_4_l25_25920


namespace complement_correct_l25_25805

-- Define the universal set U
def U : Set ℤ := {x | -2 < x ∧ x ≤ 3}

-- Define the set A
def A : Set ℤ := {3}

-- Define the complement of A with respect to U
def complement_U_A : Set ℤ := {x | x ∈ U ∧ x ∉ A}

theorem complement_correct : complement_U_A = { -1, 0, 1, 2 } :=
by
  sorry

end complement_correct_l25_25805


namespace product_of_repeating_decimal_l25_25675

theorem product_of_repeating_decimal (p : ℝ) (h : p = 0.6666666666666667) : p * 6 = 4 :=
sorry

end product_of_repeating_decimal_l25_25675


namespace angle_380_in_first_quadrant_l25_25334

theorem angle_380_in_first_quadrant : ∃ n : ℤ, 380 - 360 * n = 20 ∧ 0 ≤ 20 ∧ 20 ≤ 90 :=
by
  use 1 -- We use 1 because 380 = 20 + 360 * 1
  sorry

end angle_380_in_first_quadrant_l25_25334


namespace angles_equal_sixty_degrees_l25_25650

/-- Given a triangle ABC with sides a, b, c and respective angles α, β, γ, and with circumradius R,
if the following equation holds:
    (a * cos α + b * cos β + c * cos γ) / (a * sin β + b * sin γ + c * sin α) = (a + b + c) / (9 * R),
prove that α = β = γ = 60 degrees. -/
theorem angles_equal_sixty_degrees 
  (a b c R : ℝ) 
  (α β γ : ℝ) 
  (h : (a * Real.cos α + b * Real.cos β + c * Real.cos γ) / (a * Real.sin β + b * Real.sin γ + c * Real.sin α) = (a + b + c) / (9 * R)) :
  α = 60 ∧ β = 60 ∧ γ = 60 := 
sorry

end angles_equal_sixty_degrees_l25_25650


namespace gerald_added_crayons_l25_25065

namespace Proof

variable (original_crayons : ℕ) (total_crayons : ℕ)

theorem gerald_added_crayons (h1 : original_crayons = 7) (h2 : total_crayons = 13) : 
  total_crayons - original_crayons = 6 := by
  sorry

end Proof

end gerald_added_crayons_l25_25065


namespace angle_in_third_quadrant_l25_25430

open Real

/--
Given that 2013° can be represented as 213° + 5 * 360° and that 213° is a third quadrant angle,
we can deduce that 2013° is also a third quadrant angle.
-/
theorem angle_in_third_quadrant (h1 : 2013 = 213 + 5 * 360) (h2 : 180 < 213 ∧ 213 < 270) : 
  (540 < 2013 % 360 ∧ 2013 % 360 < 270) :=
sorry

end angle_in_third_quadrant_l25_25430


namespace direct_proportional_function_point_l25_25138

theorem direct_proportional_function_point 
    (h₁ : ∃ k : ℝ, ∀ x : ℝ, (2, -3).snd = k * (2, -3).fst)
    (h₂ : ∃ k : ℝ, ∀ x : ℝ, (4, -6).snd = k * (4, -6).fst)
    : (∃ k : ℝ, k = -(3 / 2)) :=
by
  sorry

end direct_proportional_function_point_l25_25138


namespace most_significant_price_drop_l25_25665

noncomputable def price_change (month : ℕ) : ℝ :=
  match month with
  | 1 => -1.00
  | 2 => 0.50
  | 3 => -3.00
  | 4 => 2.00
  | 5 => -1.50
  | 6 => -0.75
  | _ => 0.00 -- For any invalid month, we assume no price change

theorem most_significant_price_drop :
  ∀ m : ℕ, (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 5 ∨ m = 6) →
  (∀ n : ℕ, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6) →
  price_change m ≤ price_change n) → m = 3 :=
by
  intros m hm H
  sorry

end most_significant_price_drop_l25_25665


namespace initial_amount_in_cookie_jar_l25_25167

theorem initial_amount_in_cookie_jar (doris_spent : ℕ) (martha_spent : ℕ) (amount_left : ℕ) (spent_eq_martha : martha_spent = doris_spent / 2) (amount_left_eq : amount_left = 12) (doris_spent_eq : doris_spent = 6) : (doris_spent + martha_spent + amount_left = 21) :=
by
  sorry

end initial_amount_in_cookie_jar_l25_25167


namespace numPerfectSquareFactorsOf450_l25_25714

def isPerfectSquare (n : Nat) : Prop :=
  ∃ k : Nat, n = k * k

theorem numPerfectSquareFactorsOf450 : 
  ∃! n : Nat, 
    (∀ d : Nat, d ∣ 450 → isPerfectSquare d) → n = 4 := 
by
  sorry

end numPerfectSquareFactorsOf450_l25_25714


namespace correct_calculation_l25_25048

theorem correct_calculation (a : ℝ) : -3 * a - 2 * a = -5 * a :=
by
  sorry

end correct_calculation_l25_25048


namespace last_number_is_five_l25_25036

theorem last_number_is_five (seq : ℕ → ℕ) (h₀ : seq 0 = 5)
  (h₁ : ∀ n < 32, seq n + seq (n+1) + seq (n+2) + seq (n+3) + seq (n+4) + seq (n+5) = 29) :
  seq 36 = 5 :=
sorry

end last_number_is_five_l25_25036


namespace required_number_l25_25827

-- Define the main variables and conditions
variables {i : ℂ} (z : ℂ)
axiom i_squared : i^2 = -1

-- State the theorem that needs to be proved
theorem required_number (h : z + (4 - 8 * i) = 1 + 10 * i) : z = -3 + 18 * i :=
by {
  -- the exact steps for the proof will follow here
  sorry
}

end required_number_l25_25827


namespace students_per_group_l25_25343

-- Definitions for conditions
def number_of_boys : ℕ := 28
def number_of_girls : ℕ := 4
def number_of_groups : ℕ := 8
def total_students : ℕ := number_of_boys + number_of_girls

-- The Theorem we want to prove
theorem students_per_group : total_students / number_of_groups = 4 := by
  sorry

end students_per_group_l25_25343


namespace inequality_solution_l25_25402

noncomputable def solution_set (x : ℝ) : Prop := 
  (x < -1) ∨ (x > 3)

theorem inequality_solution :
  { x : ℝ | (3 - x) / (x + 1) < 0 } = { x : ℝ | solution_set x } :=
by
  sorry

end inequality_solution_l25_25402


namespace scientific_notation_correct_l25_25724

-- Defining the given number in terms of its scientific notation components.
def million : ℝ := 10^6
def num_million : ℝ := 15.276

-- Expressing the number 15.276 million using its definition.
def fifteen_point_two_seven_six_million : ℝ := num_million * million

-- Scientific notation representation to be proved.
def scientific_notation : ℝ := 1.5276 * 10^7

-- The theorem statement.
theorem scientific_notation_correct :
  fifteen_point_two_seven_six_million = scientific_notation :=
by
  sorry

end scientific_notation_correct_l25_25724


namespace totalInterest_l25_25617

-- Definitions for the amounts and interest rates
def totalInvestment : ℝ := 22000
def investedAt18 : ℝ := 7000
def rate18 : ℝ := 0.18
def rate14 : ℝ := 0.14

-- Calculations as conditions
def interestFrom18 (p r : ℝ) : ℝ := p * r
def investedAt14 (total inv18 : ℝ) : ℝ := total - inv18
def interestFrom14 (p r : ℝ) : ℝ := p * r

-- Proof statement
theorem totalInterest : interestFrom18 investedAt18 rate18 + interestFrom14 (investedAt14 totalInvestment investedAt18) rate14 = 3360 :=
by
  sorry

end totalInterest_l25_25617


namespace zero_points_sum_gt_one_l25_25882

noncomputable def f (x : ℝ) : ℝ := Real.log x + (1 / (2 * x))

noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem zero_points_sum_gt_one (x₁ x₂ m : ℝ) (h₁ : x₁ < x₂) 
  (hx₁ : g x₁ m = 0) (hx₂ : g x₂ m = 0) : 
  x₁ + x₂ > 1 := 
  by
    sorry

end zero_points_sum_gt_one_l25_25882


namespace intersection_of_intervals_l25_25385

theorem intersection_of_intervals :
  let A := {x : ℝ | x < -3}
  let B := {x : ℝ | x > -4}
  A ∩ B = {x : ℝ | -4 < x ∧ x < -3} :=
by
  sorry

end intersection_of_intervals_l25_25385


namespace four_digit_number_l25_25305

theorem four_digit_number (x : ℕ) (hx : 100 ≤ x ∧ x < 1000) (unit_digit : ℕ) (hu : unit_digit = 2) :
    (10 * x + unit_digit) - (2000 + x) = 108 → 10 * x + unit_digit = 2342 :=
by
  intros h
  sorry


end four_digit_number_l25_25305


namespace solve_inequality_l25_25033

theorem solve_inequality (a x : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 1 > 0) : 
  (-2 < a ∧ a < 1 → a < x ∧ x < 2 - a) ∧ 
  (a = 1 → False) ∧ 
  (1 < a ∧ a < 2 → 2 - a < x ∧ x < a) :=
by
  sorry

end solve_inequality_l25_25033


namespace total_length_of_joined_papers_l25_25319

theorem total_length_of_joined_papers :
  let length_each_sheet := 10 -- in cm
  let number_of_sheets := 20
  let overlap_length := 0.5 -- in cm
  let total_overlapping_connections := number_of_sheets - 1
  let total_length_without_overlap := length_each_sheet * number_of_sheets
  let total_overlap_length := overlap_length * total_overlapping_connections
  let total_length := total_length_without_overlap - total_overlap_length
  total_length = 190.5 :=
by {
    sorry
}

end total_length_of_joined_papers_l25_25319


namespace luis_can_make_sum_multiple_of_4_l25_25335

noncomputable def sum_of_dice (dice: List ℕ) : ℕ :=
  dice.sum 

theorem luis_can_make_sum_multiple_of_4 (d1 d2 d3: ℕ) 
  (h1: 1 ≤ d1 ∧ d1 ≤ 6) 
  (h2: 1 ≤ d2 ∧ d2 ≤ 6) 
  (h3: 1 ≤ d3 ∧ d3 ≤ 6) : 
  ∃ (dice: List ℕ), dice.length = 3 ∧ 
  sum_of_dice dice % 4 = 0 := 
by
  sorry

end luis_can_make_sum_multiple_of_4_l25_25335


namespace laura_needs_to_buy_flour_l25_25674

/--
Laura is baking a cake and needs to buy ingredients.
Flour costs $4, sugar costs $2, butter costs $2.5, and eggs cost $0.5.
The cake is cut into 6 slices. Her mother ate 2 slices.
The dog ate the remaining cake, costing $6.
Prove that Laura needs to buy flour worth $4.
-/
theorem laura_needs_to_buy_flour
  (flour_cost sugar_cost butter_cost eggs_cost dog_ate_cost : ℝ)
  (cake_slices mother_ate_slices dog_ate_slices : ℕ)
  (H_flour : flour_cost = 4)
  (H_sugar : sugar_cost = 2)
  (H_butter : butter_cost = 2.5)
  (H_eggs : eggs_cost = 0.5)
  (H_dog_ate : dog_ate_cost = 6)
  (total_slices : cake_slices = 6)
  (mother_slices : mother_ate_slices = 2)
  (dog_slices : dog_ate_slices = 4) :
  flour_cost = 4 :=
by {
  sorry
}

end laura_needs_to_buy_flour_l25_25674


namespace intersection_points_vertex_of_function_value_of_m_shift_l25_25043

noncomputable def quadratic_function (x m : ℝ) : ℝ :=
  (x - m) ^ 2 - 2 * (x - m)

theorem intersection_points (m : ℝ) : 
  ∃ x, quadratic_function x m = 0 ↔ x = m ∨ x = m + 2 := 
by
  sorry

theorem vertex_of_function (m : ℝ) : 
  ∃ x y, y = quadratic_function x m 
  ∧ x = m + 1 ∧ y = -1 := 
by
  sorry

theorem value_of_m_shift (m : ℝ) :
  (m - 2 = 0) → m = 2 :=
by
  sorry

end intersection_points_vertex_of_function_value_of_m_shift_l25_25043


namespace nigella_sold_3_houses_l25_25298

noncomputable def houseA_cost : ℝ := 60000
noncomputable def houseB_cost : ℝ := 3 * houseA_cost
noncomputable def houseC_cost : ℝ := 2 * houseA_cost - 110000
noncomputable def commission_rate : ℝ := 0.02

noncomputable def houseA_commission : ℝ := houseA_cost * commission_rate
noncomputable def houseB_commission : ℝ := houseB_cost * commission_rate
noncomputable def houseC_commission : ℝ := houseC_cost * commission_rate

noncomputable def total_commission : ℝ := houseA_commission + houseB_commission + houseC_commission
noncomputable def base_salary : ℝ := 3000
noncomputable def total_earnings : ℝ := base_salary + total_commission

theorem nigella_sold_3_houses 
  (H1 : total_earnings = 8000) 
  (H2 : houseA_cost = 60000) 
  (H3 : houseB_cost = 3 * houseA_cost) 
  (H4 : houseC_cost = 2 * houseA_cost - 110000) 
  (H5 : commission_rate = 0.02) :
  3 = 3 :=
by 
  -- Proof not required
  sorry

end nigella_sold_3_houses_l25_25298


namespace mason_ate_15_hotdogs_l25_25344

structure EatingContest where
  hotdogWeight : ℕ
  burgerWeight : ℕ
  pieWeight : ℕ
  noahBurgers : ℕ
  jacobPiesLess : ℕ
  masonHotdogsWeight : ℕ

theorem mason_ate_15_hotdogs (data : EatingContest)
    (h1 : data.hotdogWeight = 2)
    (h2 : data.burgerWeight = 5)
    (h3 : data.pieWeight = 10)
    (h4 : data.noahBurgers = 8)
    (h5 : data.jacobPiesLess = 3)
    (h6 : data.masonHotdogsWeight = 30) :
    (data.masonHotdogsWeight / data.hotdogWeight) = 15 :=
by
  sorry

end mason_ate_15_hotdogs_l25_25344


namespace age_ratio_7_9_l25_25442

/-- Definition of Sachin and Rahul's ages -/
def sachin_age : ℝ := 24.5
def rahul_age : ℝ := sachin_age + 7

/-- The ratio of Sachin's age to Rahul's age is 7:9 -/
theorem age_ratio_7_9 : sachin_age / rahul_age = 7 / 9 := by
  sorry

end age_ratio_7_9_l25_25442


namespace regular_polygon_sides_l25_25074

theorem regular_polygon_sides (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (angle_A angle_B angle_C : ℝ)
  (is_circle_inscribed_triangle : angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A ∧ angle_B + angle_C + angle_A = 180)
  (n : ℕ)
  (is_regular_polygon : B = C ∧ angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A) :
  n = 9 := sorry

end regular_polygon_sides_l25_25074


namespace coin_toss_probability_l25_25450

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

theorem coin_toss_probability :
  binomial_probability 3 2 0.5 = 0.375 :=
by
  sorry

end coin_toss_probability_l25_25450


namespace line_curve_intersection_symmetric_l25_25845

theorem line_curve_intersection_symmetric (a b : ℝ) 
    (h1 : ∃ p q : ℝ × ℝ, 
          (p.2 = a * p.1 + 1) ∧ 
          (q.2 = a * q.1 + 1) ∧ 
          (p ≠ q) ∧ 
          (p.1^2 + p.2^2 + b * p.1 - p.2 = 1) ∧ 
          (q.1^2 + q.2^2 + b * q.1 - q.2 = 1) ∧ 
          (p.1 + p.2 = -q.1 - q.2)) : 
  a + b = 2 :=
sorry

end line_curve_intersection_symmetric_l25_25845


namespace pascals_triangle_row_20_fifth_element_l25_25718

-- Define the binomial coefficient function
noncomputable def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.div (Nat.factorial n) ((Nat.factorial k) * (Nat.factorial (n - k)))

-- State the theorem about Row 20, fifth element in Pascal's triangle
theorem pascals_triangle_row_20_fifth_element :
  binomial 20 4 = 4845 := 
by
  sorry

end pascals_triangle_row_20_fifth_element_l25_25718


namespace range_of_alpha_div_three_l25_25823

open Real

theorem range_of_alpha_div_three {k : ℤ} {α : ℝ} 
  (h1 : sin α > 0)
  (h2 : cos α < 0)
  (h3 : sin (α / 3) > cos (α / 3)) :
  (2 * k * π + π / 4 < α / 3 ∧ α / 3 < 2 * k * π + π / 3) 
  ∨ (2 * k * π + 5 * π / 6 < α / 3 ∧ α / 3 < 2 * k * π + π) :=
sorry

end range_of_alpha_div_three_l25_25823


namespace dirk_profit_l25_25648

theorem dirk_profit 
  (days : ℕ) 
  (amulets_per_day : ℕ) 
  (sale_price : ℕ) 
  (cost_price : ℕ) 
  (cut_percentage : ℕ) 
  (profit : ℕ) : 
  days = 2 → amulets_per_day = 25 → sale_price = 40 → cost_price = 30 → cut_percentage = 10 → profit = 300 :=
by
  intros h_days h_amulets_per_day h_sale_price h_cost_price h_cut_percentage
  -- Placeholder for the proof
  sorry

end dirk_profit_l25_25648


namespace find_a_l25_25614

theorem find_a (a x y : ℤ) (h_x : x = 1) (h_y : y = -3) (h_eq : a * x - y = 1) : a = -2 := by
  -- Proof skipped
  sorry

end find_a_l25_25614


namespace Randy_used_blocks_l25_25861

theorem Randy_used_blocks (initial_blocks blocks_left used_blocks : ℕ) 
  (h1 : initial_blocks = 97) 
  (h2 : blocks_left = 72) 
  (h3 : used_blocks = initial_blocks - blocks_left) : 
  used_blocks = 25 :=
by
  sorry

end Randy_used_blocks_l25_25861


namespace john_days_off_l25_25897

def streams_per_week (earnings_per_week : ℕ) (rate_per_hour : ℕ) : ℕ := earnings_per_week / rate_per_hour

def streaming_sessions (hours_per_week : ℕ) (hours_per_session : ℕ) : ℕ := hours_per_week / hours_per_session

def days_off_per_week (total_days : ℕ) (streaming_days : ℕ) : ℕ := total_days - streaming_days

theorem john_days_off (hours_per_session : ℕ) (hourly_rate : ℕ) (weekly_earnings : ℕ) (total_days : ℕ) :
  hours_per_session = 4 → 
  hourly_rate = 10 → 
  weekly_earnings = 160 → 
  total_days = 7 → 
  days_off_per_week total_days (streaming_sessions (streams_per_week weekly_earnings hourly_rate) hours_per_session) = 3 := 
by
  intros
  sorry

end john_days_off_l25_25897


namespace mildred_weight_l25_25083

theorem mildred_weight (carol_weight mildred_is_heavier : ℕ) (h1 : carol_weight = 9) (h2 : mildred_is_heavier = 50) :
  carol_weight + mildred_is_heavier = 59 :=
by
  sorry

end mildred_weight_l25_25083


namespace rear_revolutions_l25_25894

variable (r_r : ℝ)  -- radius of the rear wheel
variable (r_f : ℝ)  -- radius of the front wheel
variable (n_f : ℕ)  -- number of revolutions of the front wheel
variable (n_r : ℕ)  -- number of revolutions of the rear wheel

-- Condition: radius of the front wheel is 2 times the radius of the rear wheel.
axiom front_radius : r_f = 2 * r_r

-- Condition: the front wheel makes 10 revolutions.
axiom front_revolutions : n_f = 10

-- Theorem statement to prove
theorem rear_revolutions : n_r = 20 :=
sorry

end rear_revolutions_l25_25894


namespace A_worked_days_l25_25149

theorem A_worked_days 
  (W : ℝ)                              -- Total work in arbitrary units
  (A_work_days : ℕ)                    -- Days A can complete the work 
  (B_work_days_remaining : ℕ)          -- Days B takes to complete remaining work
  (B_work_days : ℕ)                    -- Days B can complete the work alone
  (hA : A_work_days = 15)              -- A can do the work in 15 days
  (hB : B_work_days_remaining = 12)    -- B completes the remaining work in 12 days
  (hB_alone : B_work_days = 18)        -- B alone can do the work in 18 days
  :
  ∃ (x : ℕ), x = 5                     -- A worked for 5 days before leaving the job
  := 
  sorry                                 -- Proof not provided

end A_worked_days_l25_25149


namespace domain_eq_l25_25987

def domain_of_function :
    Set ℝ := {x | (x - 1 ≥ 0) ∧ (x + 1 > 0)}

theorem domain_eq :
    domain_of_function = {x | x ≥ 1} :=
by
  sorry

end domain_eq_l25_25987


namespace part1_part2a_part2b_part2c_l25_25569

def f (x a : ℝ) := |2 * x - 1| + |x - a|

theorem part1 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) : f x 3 ≤ 4 := sorry

theorem part2a (a x : ℝ) (h0 : a < 1 / 2) (h1 : a ≤ x ∧ x ≤ 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2b (a x : ℝ) (h0 : a = 1 / 2) (h1 : x = 1 / 2) : f x a = |x - 1 + a| := sorry

theorem part2c (a x : ℝ) (h0 : a > 1 / 2) (h1 : 1 / 2 ≤ x ∧ x ≤ a) : f x a = |x - 1 + a| := sorry

end part1_part2a_part2b_part2c_l25_25569


namespace votes_cast_l25_25667

theorem votes_cast (V : ℝ) (h1 : ∃ (x : ℝ), x = 0.35 * V) (h2 : ∃ (y : ℝ), y = x + 2100) : V = 7000 :=
by sorry

end votes_cast_l25_25667


namespace ariana_average_speed_l25_25243

theorem ariana_average_speed
  (sadie_speed : ℝ)
  (sadie_time : ℝ)
  (ariana_time : ℝ)
  (sarah_speed : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (sadie_speed_eq : sadie_speed = 3)
  (sadie_time_eq : sadie_time = 2)
  (ariana_time_eq : ariana_time = 0.5)
  (sarah_speed_eq : sarah_speed = 4)
  (total_time_eq : total_time = 4.5)
  (total_distance_eq : total_distance = 17) :
  ∃ ariana_speed : ℝ, ariana_speed = 6 :=
by {
  sorry
}

end ariana_average_speed_l25_25243


namespace sample_size_l25_25768

variable (num_classes : ℕ) (papers_per_class : ℕ)

theorem sample_size (h_classes : num_classes = 8) (h_papers : papers_per_class = 12) : 
  num_classes * papers_per_class = 96 := 
by 
  sorry

end sample_size_l25_25768


namespace bridge_length_l25_25825

def train_length : ℕ := 170 -- Train length in meters
def train_speed : ℕ := 45 -- Train speed in kilometers per hour
def crossing_time : ℕ := 30 -- Time to cross the bridge in seconds

noncomputable def speed_m_per_s : ℚ := (train_speed * 1000) / 3600

noncomputable def total_distance : ℚ := speed_m_per_s * crossing_time

theorem bridge_length : total_distance - train_length = 205 :=
by
  sorry

end bridge_length_l25_25825


namespace camera_pics_l25_25723

-- Definitions of the given conditions
def phone_pictures := 22
def albums := 4
def pics_per_album := 6

-- The statement to prove the number of pictures uploaded from camera
theorem camera_pics : (albums * pics_per_album) - phone_pictures = 2 :=
by
  sorry

end camera_pics_l25_25723


namespace abc_equality_l25_25019

theorem abc_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
                      (h : a^3 + b^3 + c^3 - 3 * a * b * c = 0) : a = b ∧ b = c :=
by
  sorry

end abc_equality_l25_25019


namespace john_text_messages_l25_25553

/-- John decides to get a new phone number and it ends up being a recycled number. 
    He used to get some text messages a day. 
    Now he is getting 55 text messages a day, 
    and he is getting 245 text messages per week that are not intended for him. 
    How many text messages a day did he used to get?
-/
theorem john_text_messages (m : ℕ) (h1 : 55 = m + 35) (h2 : 245 = 7 * 35) : m = 20 := 
by 
  sorry

end john_text_messages_l25_25553


namespace find_d_l25_25837

theorem find_d (d : ℤ) (h : ∀ x : ℤ, 8 * x^3 + 23 * x^2 + d * x + 45 = 0 → 2 * x + 5 = 0) : 
  d = 163 := 
sorry

end find_d_l25_25837


namespace sequence_a4_value_l25_25698

theorem sequence_a4_value :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = 2 * a n + 1) ∧ a 4 = 15 :=
by
  sorry

end sequence_a4_value_l25_25698


namespace prob_x_lt_y_is_correct_l25_25150

open Set

noncomputable def prob_x_lt_y : ℝ :=
  let rectangle := Icc (0: ℝ) 4 ×ˢ Icc (0: ℝ) 3
  let area_rectangle := 4 * 3
  let triangle := {p : ℝ × ℝ | p.1 ∈ Icc (0: ℝ) 3 ∧ p.2 ∈ Icc (0: ℝ) 3 ∧ p.1 < p.2}
  let area_triangle := 1 / 2 * 3 * 3
  let probability := area_triangle / area_rectangle
  probability

-- To state as a theorem using Lean's notation
theorem prob_x_lt_y_is_correct : prob_x_lt_y = 3 / 8 := sorry

end prob_x_lt_y_is_correct_l25_25150


namespace number_of_even_factors_l25_25967

theorem number_of_even_factors {n : ℕ} (h : n = 2^4 * 3^3 * 7) : 
  ∃ (count : ℕ), count = 32 ∧ ∀ k, (k ∣ n) → k % 2 = 0 → count = 32 :=
by
  sorry

end number_of_even_factors_l25_25967


namespace calc_expression_l25_25871

theorem calc_expression : 5 + 2 * (8 - 3) = 15 :=
by
  -- Proof steps would go here
  sorry

end calc_expression_l25_25871


namespace packet_b_average_height_l25_25923

theorem packet_b_average_height (x y R_A R_B H_A H_B : ℝ)
  (h_RA : R_A = 2 * x + y)
  (h_RB : R_B = 3 * x - y)
  (h_x : x = 10)
  (h_y : y = 6)
  (h_HA : H_A = 192)
  (h_20percent : H_A = H_B + 0.20 * H_B) :
  H_B = 160 := 
sorry

end packet_b_average_height_l25_25923


namespace negate_proposition_l25_25631

theorem negate_proposition :
  (¬ ∃ (x₀ : ℝ), x₀^2 + 2 * x₀ + 3 ≤ 0) ↔ (∀ (x : ℝ), x^2 + 2 * x + 3 > 0) :=
by
  sorry

end negate_proposition_l25_25631


namespace value_of_expression_l25_25238

theorem value_of_expression (n : ℝ) (h : n + 1/n = 6) : n^2 + 1/n^2 + 9 = 43 :=
by
  sorry

end value_of_expression_l25_25238


namespace john_payment_l25_25500

def camera_value : ℝ := 5000
def weekly_rental_percentage : ℝ := 0.10
def rental_period : ℕ := 4
def friend_contribution_percentage : ℝ := 0.40

theorem john_payment :
  let weekly_rental_fee := camera_value * weekly_rental_percentage
  let total_rental_fee := weekly_rental_fee * rental_period
  let friend_contribution := total_rental_fee * friend_contribution_percentage
  let john_payment := total_rental_fee - friend_contribution
  john_payment = 1200 :=
by
  sorry

end john_payment_l25_25500


namespace find_length_of_CE_l25_25330

theorem find_length_of_CE
  (triangle_ABE_right : ∀ A B E : Type, ∃ (angle_AEB : Real), angle_AEB = 45)
  (triangle_BCE_right : ∀ B C E : Type, ∃ (angle_BEC : Real), angle_BEC = 45)
  (triangle_CDE_right : ∀ C D E : Type, ∃ (angle_CED : Real), angle_CED = 45)
  (AE_is_32 : 32 = 32) :
  ∃ (CE : ℝ), CE = 16 * Real.sqrt 2 :=
by
  sorry

end find_length_of_CE_l25_25330


namespace trader_profit_l25_25216

theorem trader_profit (P : ℝ) (hP : 0 < P) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.36 * P
  let profit := selling_price - P
  (profit / P) * 100 = 36 :=
by
  -- The proof will go here
  sorry

end trader_profit_l25_25216


namespace crayons_count_l25_25651

theorem crayons_count
  (crayons_given : Nat := 563)
  (crayons_lost : Nat := 558)
  (crayons_left : Nat := 332) :
  crayons_given + crayons_lost + crayons_left = 1453 := 
sorry

end crayons_count_l25_25651


namespace sum_of_cubes_of_consecutive_numbers_divisible_by_9_l25_25784

theorem sum_of_cubes_of_consecutive_numbers_divisible_by_9 (a : ℕ) (h : a > 1) : 
  9 ∣ ((a - 1)^3 + a^3 + (a + 1)^3) := 
by 
  sorry

end sum_of_cubes_of_consecutive_numbers_divisible_by_9_l25_25784


namespace multiply_same_exponents_l25_25521

theorem multiply_same_exponents (x : ℝ) : (x^3) * (x^3) = x^6 :=
by sorry

end multiply_same_exponents_l25_25521


namespace original_price_per_lesson_l25_25193

theorem original_price_per_lesson (piano_cost lessons_cost : ℤ) (number_of_lessons discount_percent : ℚ) (total_cost : ℤ) (original_price : ℚ) :
  piano_cost = 500 ∧
  number_of_lessons = 20 ∧
  discount_percent = 0.25 ∧
  total_cost = 1100 →
  lessons_cost = total_cost - piano_cost →
  0.75 * (number_of_lessons * original_price) = lessons_cost →
  original_price = 40 :=
by
  intros h h1 h2
  sorry

end original_price_per_lesson_l25_25193


namespace find_other_number_l25_25350

/-- Given HCF(A, B), LCM(A, B), and a known A, proves the value of B. -/
theorem find_other_number (A B : ℕ) 
  (hcf : Nat.gcd A B = 16) 
  (lcm : Nat.lcm A B = 396) 
  (a_val : A = 36) : B = 176 :=
by
  sorry

end find_other_number_l25_25350


namespace deal_saves_customer_two_dollars_l25_25144

-- Define the conditions of the problem
def movie_ticket_price : ℕ := 8
def popcorn_price : ℕ := movie_ticket_price - 3
def drink_price : ℕ := popcorn_price + 1
def candy_price : ℕ := drink_price / 2

def normal_total_price : ℕ := movie_ticket_price + popcorn_price + drink_price + candy_price
def deal_price : ℕ := 20

-- Prove the savings
theorem deal_saves_customer_two_dollars : normal_total_price - deal_price = 2 :=
by
  -- We will fill in the proof here
  sorry

end deal_saves_customer_two_dollars_l25_25144


namespace donation_amount_per_person_l25_25250

theorem donation_amount_per_person (m n : ℕ) 
  (h1 : m + 11 = n + 9) 
  (h2 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (m + 11)) 
  (h3 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (n + 9))
  : ∃ k : ℕ, k = 25 ∨ k = 47 :=
by
  sorry

end donation_amount_per_person_l25_25250


namespace range_of_m_l25_25671

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y - x * y = 0) : 
  ∀ m : ℝ, (xy ≥ m^2 - 6 * m ↔ -2 ≤ m ∧ m ≤ 8) :=
sorry

end range_of_m_l25_25671


namespace initial_salty_cookies_l25_25747

theorem initial_salty_cookies (sweet_init sweet_eaten sweet_left salty_eaten : ℕ) 
  (h1 : sweet_init = 34)
  (h2 : sweet_eaten = 15)
  (h3 : sweet_left = 19)
  (h4 : salty_eaten = 56) :
  sweet_left + sweet_eaten = sweet_init → 
  sweet_init - sweet_eaten = sweet_left →
  ∃ salty_init, salty_init = salty_eaten :=
by
  sorry

end initial_salty_cookies_l25_25747


namespace seven_digit_divisible_by_11_l25_25303

theorem seven_digit_divisible_by_11 (m n : ℕ) (h1: 0 ≤ m ∧ m ≤ 9) (h2: 0 ≤ n ∧ n ≤ 9) (h3 : 10 + n - m ≡ 0 [MOD 11])  : m + n = 1 :=
by
  sorry

end seven_digit_divisible_by_11_l25_25303


namespace probability_of_picking_grain_buds_l25_25341

theorem probability_of_picking_grain_buds :
  let num_stamps := 3
  let num_grain_buds := 1
  let probability := num_grain_buds / num_stamps
  probability = 1 / 3 :=
by
  sorry

end probability_of_picking_grain_buds_l25_25341


namespace triangle_angles_l25_25309

theorem triangle_angles (r_a r_b r_c R : ℝ) (h1 : r_a + r_b = 3 * R) (h2 : r_b + r_c = 2 * R) :
  ∃ (α β γ : ℝ), α = 90 ∧ γ = 60 ∧ β = 30 :=
by
  sorry

end triangle_angles_l25_25309


namespace jake_correct_speed_l25_25916

noncomputable def distance (d t : ℝ) : Prop :=
  d = 50 * (t + 4/60) ∧ d = 70 * (t - 4/60)

noncomputable def correct_speed (d t : ℝ) : ℝ :=
  d / t

theorem jake_correct_speed (d t : ℝ) (h1 : distance d t) : correct_speed d t = 58 :=
by
  sorry

end jake_correct_speed_l25_25916


namespace gcd_10293_29384_l25_25407

theorem gcd_10293_29384 : Nat.gcd 10293 29384 = 1 := by
  sorry

end gcd_10293_29384_l25_25407


namespace jim_miles_driven_l25_25743

theorem jim_miles_driven (total_journey : ℕ) (miles_needed : ℕ) (h : total_journey = 1200 ∧ miles_needed = 985) : total_journey - miles_needed = 215 := 
by sorry

end jim_miles_driven_l25_25743


namespace mean_properties_l25_25037

theorem mean_properties (a b c : ℝ) 
    (h1 : a + b + c = 36) 
    (h2 : a * b * c = 125) 
    (h3 : a * b + b * c + c * a = 93.75) : 
    a^2 + b^2 + c^2 = 1108.5 := 
by 
  sorry

end mean_properties_l25_25037


namespace min_value_l25_25127

-- Given points A, B, and C and their specific coordinates
def A : (ℝ × ℝ) := (1, 3)
def B (a : ℝ) : (ℝ × ℝ) := (a, 1)
def C (b : ℝ) : (ℝ × ℝ) := (-b, 0)

-- Conditions
axiom a_pos (a : ℝ) : a > 0
axiom b_pos (b : ℝ) : b > 0
axiom collinear (a b : ℝ) : 3 * a + 2 * b = 1

-- The theorem to prove
theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hcollinear : 3 * a + 2 * b = 1) : 
  ∃ z, z = 11 + 6 * Real.sqrt 2 ∧ ∀ (x y : ℝ), (x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 1) -> (3 / x + 1 / y) ≥ z :=
by sorry -- Proof to be provided

end min_value_l25_25127


namespace k_value_l25_25418

theorem k_value (k : ℝ) :
    (∀ r s : ℝ, (r + s = -k ∧ r * s = 9) ∧ ((r + 3) + (s + 3) = k)) → k = -3 :=
by
    intro h
    sorry

end k_value_l25_25418


namespace quadratic_inequality_sum_l25_25109

theorem quadratic_inequality_sum (a b : ℝ) (h1 : 1 < 2) 
 (h2 : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) 
 (h3 : 1 + 2 = a)  (h4 : 1 * 2 = b) : 
 a + b = 5 := 
by 
sorry

end quadratic_inequality_sum_l25_25109


namespace algebraic_expression_value_l25_25077

theorem algebraic_expression_value 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = ab + bc + ac)
  (h2 : a = 1) : 
  (a + b - c) ^ 2004 = 1 := 
by
  sorry

end algebraic_expression_value_l25_25077


namespace simplify_expression_l25_25980

theorem simplify_expression (a : ℝ) : 
  ( (a^(16 / 8))^(1 / 4) )^3 * ( (a^(16 / 4))^(1 / 8) )^3 = a^3 := by
  sorry

end simplify_expression_l25_25980


namespace correct_operations_result_l25_25668

-- Define conditions and the problem statement
theorem correct_operations_result (x : ℝ) (h1: x / 8 - 12 = 18) : (x * 8) * 12 = 23040 :=
by
  sorry

end correct_operations_result_l25_25668


namespace pentagon_area_pq_sum_l25_25223

theorem pentagon_area_pq_sum 
  (p q : ℤ) 
  (hp : 0 < q ∧ q < p) 
  (harea : 5 * p * q - q * q = 700) : 
  ∃ sum : ℤ, sum = p + q :=
by
  sorry

end pentagon_area_pq_sum_l25_25223


namespace almond_butter_ratio_l25_25911

theorem almond_butter_ratio
  (peanut_cost almond_cost batch_extra almond_per_batch : ℝ)
  (h1 : almond_cost = 3 * peanut_cost)
  (h2 : peanut_cost = 3)
  (h3 : almond_per_batch = batch_extra)
  (h4 : batch_extra = 3) :
  almond_per_batch / almond_cost = 1 / 3 := sorry

end almond_butter_ratio_l25_25911


namespace problem_statement_l25_25226

theorem problem_statement (a b : ℤ) (h1 : b = 7) (h2: a * b = 2 * (a + b) + 1) :
  b - a = 4 := by
  sorry

end problem_statement_l25_25226


namespace greatest_three_digit_multiple_23_l25_25964

theorem greatest_three_digit_multiple_23 : 
  ∃ n : ℕ, n < 1000 ∧ n % 23 = 0 ∧ (∀ m : ℕ, m < 1000 ∧ m % 23 = 0 → m ≤ n) ∧ n = 989 :=
sorry

end greatest_three_digit_multiple_23_l25_25964


namespace equation1_solution_equation2_solution_l25_25323

theorem equation1_solution (x : ℝ) : 4 * (2 * x - 1) ^ 2 = 36 ↔ x = 2 ∨ x = -1 :=
by sorry

theorem equation2_solution (x : ℝ) : (1 / 4) * (2 * x + 3) ^ 3 - 54 = 0 ↔ x = 3 / 2 :=
by sorry

end equation1_solution_equation2_solution_l25_25323


namespace arithmetic_expression_l25_25970

theorem arithmetic_expression : 5 + 12 / 3 - 3 ^ 2 + 1 = 1 := by
  sorry

end arithmetic_expression_l25_25970


namespace find_integer_in_range_divisible_by_18_l25_25622

theorem find_integer_in_range_divisible_by_18 
  (n : ℕ) (h1 : 900 ≤ n) (h2 : n ≤ 912) (h3 : n % 18 = 0) : n = 900 :=
sorry

end find_integer_in_range_divisible_by_18_l25_25622


namespace problem1_problem2_problem3_l25_25210

variables (a b c : ℝ)

-- First proof problem
theorem problem1 (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : a * b * c ≠ 0 :=
sorry

-- Second proof problem
theorem problem2 (h : a = 0 ∨ b = 0 ∨ c = 0) : a * b * c = 0 :=
sorry

-- Third proof problem
theorem problem3 (h : a * b < 0 ∨ a = 0 ∨ b = 0) : a * b ≤ 0 :=
sorry

end problem1_problem2_problem3_l25_25210


namespace polynomial_evaluation_l25_25710

noncomputable def evaluate_polynomial (x : ℝ) : ℝ :=
  x^3 - 3 * x^2 - 9 * x + 5

theorem polynomial_evaluation (x : ℝ) (h_pos : x > 0) (h_eq : x^2 - 3 * x - 9 = 0) :
  evaluate_polynomial x = 5 :=
by
  sorry

end polynomial_evaluation_l25_25710


namespace meatballs_fraction_each_son_eats_l25_25973

theorem meatballs_fraction_each_son_eats
  (f1 f2 f3 : ℝ)
  (h1 : ∃ f1 f2 f3, f1 + f2 + f3 = 2)
  (meatballs_initial : ∀ n, n = 3) :
  f1 = 2/3 ∧ f2 = 2/3 ∧ f3 = 2/3 := by
  sorry

end meatballs_fraction_each_son_eats_l25_25973


namespace employee_salaries_l25_25928

theorem employee_salaries 
  (x y z : ℝ)
  (h1 : x + y + z = 638)
  (h2 : x = 1.20 * y)
  (h3 : z = 0.80 * y) :
  x = 255.20 ∧ y = 212.67 ∧ z = 170.14 :=
sorry

end employee_salaries_l25_25928


namespace total_hours_worked_l25_25630

theorem total_hours_worked (Amber_hours : ℕ) (h_Amber : Amber_hours = 12) 
  (Armand_hours : ℕ) (h_Armand : Armand_hours = Amber_hours / 3)
  (Ella_hours : ℕ) (h_Ella : Ella_hours = Amber_hours * 2) : 
  Amber_hours + Armand_hours + Ella_hours = 40 :=
by
  rw [h_Amber, h_Armand, h_Ella]
  norm_num
  sorry

end total_hours_worked_l25_25630


namespace cookie_percentage_increase_l25_25789

theorem cookie_percentage_increase (cookies_Monday cookies_Tuesday cookies_Wednesday total_cookies : ℕ) 
  (h1 : cookies_Monday = 5)
  (h2 : cookies_Tuesday = 2 * cookies_Monday)
  (h3 : total_cookies = cookies_Monday + cookies_Tuesday + cookies_Wednesday)
  (h4 : total_cookies = 29) :
  (100 * (cookies_Wednesday - cookies_Tuesday) / cookies_Tuesday = 40) := 
by
  sorry

end cookie_percentage_increase_l25_25789


namespace candy_mixture_problem_l25_25956

theorem candy_mixture_problem:
  ∃ x y : ℝ, x + y = 5 ∧ 3.20 * x + 1.70 * y = 10 ∧ x = 1 :=
by
  sorry

end candy_mixture_problem_l25_25956


namespace rectangle_perimeter_l25_25278

theorem rectangle_perimeter (b : ℕ) (h1 : 3 * b * b = 192) : 2 * ((3 * b) + b) = 64 := 
by
  sorry

end rectangle_perimeter_l25_25278


namespace cos_F_in_triangle_l25_25659

theorem cos_F_in_triangle (D E F : ℝ) (sin_D : ℝ) (cos_E : ℝ) (cos_F : ℝ) 
  (h1 : sin_D = 4 / 5) 
  (h2 : cos_E = 12 / 13) 
  (D_plus_E_plus_F : D + E + F = π) :
  cos_F = -16 / 65 :=
by
  sorry

end cos_F_in_triangle_l25_25659


namespace find_pre_tax_remuneration_l25_25867

def pre_tax_remuneration (x : ℝ) : Prop :=
  let taxable_amount := if x <= 4000 then x - 800 else x * 0.8
  let tax_due := taxable_amount * 0.2
  let final_tax := tax_due * 0.7
  final_tax = 280

theorem find_pre_tax_remuneration : ∃ x : ℝ, pre_tax_remuneration x ∧ x = 2800 := by
  sorry

end find_pre_tax_remuneration_l25_25867


namespace sequence_length_arithmetic_sequence_l25_25143

theorem sequence_length_arithmetic_sequence :
  ∃ n : ℕ, ∀ (a d : ℕ), a = 2 → d = 3 → a + (n - 1) * d = 2014 ∧ n = 671 :=
by {
  sorry
}

end sequence_length_arithmetic_sequence_l25_25143


namespace sales_tax_reduction_difference_l25_25141

def sales_tax_difference (original_rate new_rate market_price : ℝ) : ℝ :=
  (market_price * original_rate) - (market_price * new_rate)

theorem sales_tax_reduction_difference :
  sales_tax_difference 0.035 0.03333 10800 = 18.36 :=
by
  -- This is where the proof would go, but it is not required for this task.
  sorry

end sales_tax_reduction_difference_l25_25141


namespace sum_of_distinct_roots_eq_zero_l25_25180

theorem sum_of_distinct_roots_eq_zero
  (a b m n p : ℝ)
  (h1 : m ≠ n)
  (h2 : m ≠ p)
  (h3 : n ≠ p)
  (h_m : m^3 + a * m + b = 0)
  (h_n : n^3 + a * n + b = 0)
  (h_p : p^3 + a * p + b = 0) : 
  m + n + p = 0 :=
sorry

end sum_of_distinct_roots_eq_zero_l25_25180


namespace bucket_full_weight_l25_25586

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = p) 
  (h2 : x + (3 / 4) * y = q) : 
  x + y = (8 * q - 3 * p) / 5 := 
  by
    sorry

end bucket_full_weight_l25_25586


namespace weight_of_new_person_l25_25354

/-- 
The average weight of 10 persons increases by 6.3 kg when a new person replaces one of them. 
The weight of the replaced person is 65 kg. 
Prove that the weight of the new person is 128 kg. 
-/
theorem weight_of_new_person (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  (avg_increase = 6.3) → 
  (old_weight = 65) → 
  (new_weight = old_weight + 10 * avg_increase) → 
  new_weight = 128 := 
by
  intros
  sorry

end weight_of_new_person_l25_25354


namespace possible_values_of_reciprocal_l25_25003

theorem possible_values_of_reciprocal (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  ∃ S, S = { x : ℝ | x >= 9 } ∧ (∃ x, x = (1/a + 1/b) ∧ x ∈ S) :=
sorry

end possible_values_of_reciprocal_l25_25003


namespace sara_frosting_total_l25_25876

def cakes_baked_each_day : List Nat := [7, 12, 8, 10, 15]
def cakes_eaten_by_Carol : List Nat := [4, 6, 3, 2, 3]
def cans_per_cake_each_day : List Nat := [2, 3, 4, 3, 2]

def total_frosting_cans_needed : Nat :=
  let remaining_cakes := List.zipWith (· - ·) cakes_baked_each_day cakes_eaten_by_Carol
  let required_cans := List.zipWith (· * ·) remaining_cakes cans_per_cake_each_day
  required_cans.foldl (· + ·) 0

theorem sara_frosting_total : total_frosting_cans_needed = 92 := by
  sorry

end sara_frosting_total_l25_25876


namespace exists_radius_for_marked_points_l25_25537

theorem exists_radius_for_marked_points :
  ∃ R : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ < 2 * π) →
    (∃ n : ℕ, (θ ≤ (n * 2 * π * R) % (2 * π * R) + 1 / R ∧ (n * 2 * π * R) % (2 * π * R) < θ + 1))) :=
sorry

end exists_radius_for_marked_points_l25_25537


namespace oil_layer_height_l25_25023

/-- Given a tank with a rectangular bottom measuring 16 cm in length and 12 cm in width, initially containing 6 cm deep water and 6 cm deep oil, and an iron block with dimensions 8 cm in length, 8 cm in width, and 12 cm in height -/

theorem oil_layer_height (volume_water volume_oil volume_iron base_area new_volume_water : ℝ) 
  (base_area_def : base_area = 16 * 12) 
  (volume_water_def : volume_water = base_area * 6) 
  (volume_oil_def : volume_oil = base_area * 6) 
  (volume_iron_def : volume_iron = 8 * 8 * 12) 
  (new_volume_water_def : new_volume_water = volume_water + volume_iron) 
  (new_water_height : new_volume_water / base_area = 10) 
  : (volume_water + volume_oil) / base_area - (new_volume_water / base_area - 6) = 7 :=
by 
  sorry

end oil_layer_height_l25_25023


namespace train_cross_time_l25_25260

def train_length := 100
def bridge_length := 275
def train_speed_kmph := 45

noncomputable def train_speed_mps : ℝ :=
  (train_speed_kmph * 1000.0) / 3600.0

theorem train_cross_time :
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  time = 30 :=
by 
  -- Introduce definitions to make sure they align with the initial conditions
  let total_distance := train_length + bridge_length
  let speed := train_speed_mps
  let time := total_distance / speed
  -- Prove time = 30
  sorry

end train_cross_time_l25_25260


namespace method_1_more_cost_effective_l25_25682

open BigOperators

def racket_price : ℕ := 20
def shuttlecock_price : ℕ := 5
def rackets_bought : ℕ := 4
def shuttlecocks_bought : ℕ := 30
def discount_rate : ℚ := 0.92

def total_price (rackets shuttlecocks : ℕ) := racket_price * rackets + shuttlecock_price * shuttlecocks

def method_1_cost (rackets shuttlecocks : ℕ) := 
  total_price rackets shuttlecocks - shuttlecock_price * rackets

def method_2_cost (total : ℚ) :=
  total * discount_rate

theorem method_1_more_cost_effective :
  method_1_cost rackets_bought shuttlecocks_bought
  <
  method_2_cost (total_price rackets_bought shuttlecocks_bought) :=
by
  sorry

end method_1_more_cost_effective_l25_25682


namespace probability_of_C_l25_25511

def region_prob_A := (1 : ℚ) / 4
def region_prob_B := (1 : ℚ) / 3
def region_prob_D := (1 : ℚ) / 6

theorem probability_of_C :
  (region_prob_A + region_prob_B + region_prob_D + (1 : ℚ) / 4) = 1 :=
by
  sorry

end probability_of_C_l25_25511


namespace solve_equation_l25_25153

noncomputable def smallest_solution : ℝ :=
(15 - Real.sqrt 549) / 6

theorem solve_equation :
  ∃ x : ℝ, 
    (3 * x / (x - 3) + (3 * x^2 - 27) / x = 18) ∧
    x = smallest_solution :=
by
  sorry

end solve_equation_l25_25153


namespace number_of_segments_before_returning_to_start_l25_25820

-- Definitions based on the conditions
def concentric_circles (r R : ℝ) (h_circle : r < R) : Prop := true

def tangent_chord (circle1 circle2 : Prop) (A B : Point) : Prop := 
  circle1 ∧ circle2

def angle_ABC_eq_60 (A B C : Point) (angle_ABC : ℝ) : Prop :=
  angle_ABC = 60

noncomputable def number_of_segments (n : ℕ) (m : ℕ) : Prop := 
  120 * n = 360 * m

theorem number_of_segments_before_returning_to_start (r R : ℝ)
  (h_circle : r < R)
  (circle1 circle2 : Prop := concentric_circles r R h_circle)
  (A B C : Point)
  (h_tangent : tangent_chord circle1 circle2 A B)
  (angle_ABC : ℝ := 0)
  (h_ABC_eq_60 : angle_ABC_eq_60 A B C angle_ABC) :
  ∃ n : ℕ, number_of_segments n 1 ∧ n = 3 := by
  sorry

end number_of_segments_before_returning_to_start_l25_25820


namespace solve_inequalities_l25_25534

theorem solve_inequalities (x : ℝ) :
  (3 * x^2 - x > 4) ∧ (x < 3) ↔ (1 < x ∧ x < 3) := 
by 
  sorry

end solve_inequalities_l25_25534


namespace choir_members_max_l25_25118

theorem choir_members_max (m y n : ℕ) (h_square : m = y^2 + 11) (h_rect : m = n * (n + 5)) : 
  m = 300 := 
sorry

end choir_members_max_l25_25118


namespace greatest_2q_minus_r_l25_25414

theorem greatest_2q_minus_r :
  ∃ (q r : ℕ), 1027 = 21 * q + r ∧ q > 0 ∧ r > 0 ∧ 2 * q - r = 77 :=
by
  sorry

end greatest_2q_minus_r_l25_25414


namespace khalil_paid_correct_amount_l25_25937

-- Defining the charges for dogs and cats
def cost_per_dog : ℕ := 60
def cost_per_cat : ℕ := 40

-- Defining the number of dogs and cats Khalil took to the clinic
def num_dogs : ℕ := 20
def num_cats : ℕ := 60

-- The total amount Khalil paid
def total_amount_paid : ℕ := 3600

-- The theorem to prove the total amount Khalil paid
theorem khalil_paid_correct_amount :
  (cost_per_dog * num_dogs + cost_per_cat * num_cats) = total_amount_paid :=
by
  sorry

end khalil_paid_correct_amount_l25_25937


namespace arithmetic_sequence_identity_l25_25629

theorem arithmetic_sequence_identity (a : ℕ → ℝ) (d : ℝ)
    (h_arith : ∀ n, a (n + 1) = a 1 + n * d)
    (h_sum : a 4 + a 7 + a 10 = 30) :
    a 1 - a 3 - a 6 - a 8 - a 11 + a 13 = -20 :=
sorry

end arithmetic_sequence_identity_l25_25629


namespace meeting_point_l25_25297

def same_start (x : ℝ) (y : ℝ) : Prop := x = y

def walk_time (x : ℝ) (y : ℝ) (t : ℝ) : Prop := 
  x * t + y * t = 24

def hector_speed (s : ℝ) : ℝ := s

def jane_speed (s : ℝ) : ℝ := 3 * s

theorem meeting_point (s t : ℝ) :
  same_start 0 0 ∧ walk_time (hector_speed s) (jane_speed s) t → t = 6 / s ∧ (6 : ℝ) = 6 :=
by
  intros h
  sorry

end meeting_point_l25_25297


namespace correct_observation_value_l25_25536

theorem correct_observation_value (mean : ℕ) (n : ℕ) (incorrect_obs : ℕ) (corrected_mean : ℚ) (original_sum : ℚ) (remaining_sum : ℚ) (corrected_sum : ℚ) :
  mean = 30 →
  n = 50 →
  incorrect_obs = 23 →
  corrected_mean = 30.5 →
  original_sum = (n * mean) →
  remaining_sum = (original_sum - incorrect_obs) →
  corrected_sum = (n * corrected_mean) →
  ∃ x : ℕ, remaining_sum + x = corrected_sum → x = 48 :=
by
  intros h_mean h_n h_incorrect_obs h_corrected_mean h_original_sum h_remaining_sum h_corrected_sum
  have original_mean := h_mean
  have observations := h_n
  have incorrect_observation := h_incorrect_obs
  have new_mean := h_corrected_mean
  have original_sum_calc := h_original_sum
  have remaining_sum_calc := h_remaining_sum
  have corrected_sum_calc := h_corrected_sum
  use 48
  sorry

end correct_observation_value_l25_25536


namespace vector_addition_subtraction_identity_l25_25078

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (BC AB AC : V)

theorem vector_addition_subtraction_identity : BC + AB - AC = 0 := 
by sorry

end vector_addition_subtraction_identity_l25_25078


namespace volume_after_increase_l25_25538

variable (l w h : ℕ)
variable (V S E : ℕ)

noncomputable def original_volume : ℕ := l * w * h
noncomputable def surface_sum : ℕ := (l * w) + (w * h) + (h * l)
noncomputable def edge_sum : ℕ := l + w + h

theorem volume_after_increase (h_volume : original_volume l w h = 5400)
  (h_surface : surface_sum l w h = 1176)
  (h_edge : edge_sum l w h = 60) : 
  (l + 1) * (w + 1) * (h + 1) = 6637 := sorry

end volume_after_increase_l25_25538


namespace remaining_integers_count_l25_25463

def set_of_integers_from_1_to_100 : Finset ℕ := (Finset.range 100).map ⟨Nat.succ, Nat.succ_injective⟩

def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

def T : Finset ℕ := set_of_integers_from_1_to_100
def M2 : Finset ℕ := multiples_of 2 T
def M3 : Finset ℕ := multiples_of 3 T
def M5 : Finset ℕ := multiples_of 5 T

def remaining_set : Finset ℕ := T \ (M2 ∪ M3 ∪ M5)

theorem remaining_integers_count : remaining_set.card = 26 := by
  sorry

end remaining_integers_count_l25_25463


namespace find_polar_equations_and_distance_l25_25115

noncomputable def polar_equation_C1 (rho theta : ℝ) : Prop :=
  rho^2 * Real.cos (2 * theta) = 1

noncomputable def polar_equation_C2 (rho theta : ℝ) : Prop :=
  rho = 2 * Real.cos theta

theorem find_polar_equations_and_distance :
  (∀ rho theta, polar_equation_C1 rho theta ↔ rho^2 * Real.cos (2 * theta) = 1) ∧
  (∀ rho theta, polar_equation_C2 rho theta ↔ rho = 2 * Real.cos theta) ∧
  let theta := Real.pi / 6
  let rho_A := Real.sqrt 2
  let rho_B := Real.sqrt 3
  (|rho_A - rho_B| = |Real.sqrt 3 - Real.sqrt 2|) :=
  by sorry

end find_polar_equations_and_distance_l25_25115


namespace math_proof_problem_l25_25094

def A : Set ℝ := {x | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def complement_R (s : Set ℝ) := {x : ℝ | x ∉ s}

theorem math_proof_problem :
  (complement_R A ∩ B) = {x | 2 < x ∧ x ≤ 3} :=
sorry

end math_proof_problem_l25_25094


namespace total_transport_cost_l25_25329

def cost_per_kg : ℝ := 25000
def mass_sensor_g : ℝ := 350
def mass_communication_g : ℝ := 150

theorem total_transport_cost : 
  (cost_per_kg * (mass_sensor_g / 1000) + cost_per_kg * (mass_communication_g / 1000)) = 12500 :=
by
  sorry

end total_transport_cost_l25_25329


namespace fraction_addition_simplest_form_l25_25741

theorem fraction_addition_simplest_form :
  (7 / 12) + (3 / 8) = 23 / 24 :=
by
  -- Adding a sorry to skip the proof
  sorry

end fraction_addition_simplest_form_l25_25741


namespace find_k_l25_25391

-- Define the arithmetic sequence and the sum of the first n terms
def a (n : ℕ) : ℤ := 2 * n + 2
def S (n : ℕ) : ℤ := n^2 + 3 * n

-- The main assertion
theorem find_k : ∃ (k : ℕ), k > 0 ∧ (S k - a (k + 5) = 44) ∧ k = 7 :=
by
  sorry

end find_k_l25_25391


namespace close_to_one_below_l25_25188

theorem close_to_one_below (k l m n : ℕ) (h1 : k > l) (h2 : l > m) (h3 : m > n) (hk : k = 43) (hl : l = 7) (hm : m = 3) (hn : n = 2) :
  (1 : ℚ) / k + 1 / l + 1 / m + 1 / n < 1 := by
  sorry

end close_to_one_below_l25_25188


namespace largest_number_among_given_l25_25444

theorem largest_number_among_given (
  A B C D E : ℝ
) (hA : A = 0.936)
  (hB : B = 0.9358)
  (hC : C = 0.9361)
  (hD : D = 0.935)
  (hE : E = 0.921):
  C = max A (max B (max C (max D E))) :=
by
  sorry

end largest_number_among_given_l25_25444


namespace ramon_current_age_l25_25056

variable (R : ℕ) (L : ℕ)

theorem ramon_current_age :
  (L = 23) → (R + 20 = 2 * L) → R = 26 :=
by
  intro hL hR
  rw [hL] at hR
  have : R + 20 = 46 := by linarith
  linarith

end ramon_current_age_l25_25056


namespace midpoint_coordinate_sum_l25_25691

theorem midpoint_coordinate_sum
  (x1 y1 x2 y2 : ℝ)
  (h1 : x1 = 10)
  (h2 : y1 = 3)
  (h3 : x2 = 4)
  (h4 : y2 = -3) :
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym =  7 := by
  sorry

end midpoint_coordinate_sum_l25_25691


namespace factorization_correct_l25_25285

-- Define the expression
def expression (x : ℝ) : ℝ := x^2 + 2 * x

-- State the theorem to prove the factorized form is equal to the expression
theorem factorization_correct (x : ℝ) : x^2 + 2 * x = x * (x + 2) :=
by {
  -- Lean will skip the proof because of sorry, ensuring the statement compiles correctly.
  sorry
}

end factorization_correct_l25_25285


namespace pyramid_structure_l25_25924

variables {d e f a b c h i j g : ℝ}

theorem pyramid_structure (h_val : h = 16)
                         (i_val : i = 48)
                         (j_val : j = 72)
                         (g_val : g = 8)
                         (d_def : d = b * a)
                         (e_def1 : e = b * c) 
                         (e_def2 : e = d * a)
                         (f_def : f = c * a)
                         (h_def : h = d * b)
                         (i_def : i = d * a)
                         (j_def : j = e * c)
                         (g_def : g = f * c) : 
   a = 3 ∧ b = 1 ∧ c = 1.5 :=
by sorry

end pyramid_structure_l25_25924


namespace gain_in_meters_l25_25445

noncomputable def cost_price : ℝ := sorry
noncomputable def selling_price : ℝ := 1.5 * cost_price
noncomputable def total_cost_price : ℝ := 30 * cost_price
noncomputable def total_selling_price : ℝ := 30 * selling_price
noncomputable def gain : ℝ := total_selling_price - total_cost_price

theorem gain_in_meters (S C : ℝ) (h_S : S = 1.5 * C) (h_gain : gain = 15 * C) :
  15 * C / S = 10 := by
  sorry

end gain_in_meters_l25_25445


namespace neg_or_implication_l25_25940

theorem neg_or_implication {p q : Prop} : ¬(p ∨ q) → (¬p ∧ ¬q) :=
by
  intros h
  sorry

end neg_or_implication_l25_25940


namespace eq_zero_l25_25366

variable {x y z : ℤ}

theorem eq_zero (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end eq_zero_l25_25366


namespace eval_expression_l25_25705

theorem eval_expression :
  ((-2 : ℤ) ^ 3 : ℝ) ^ (1/3 : ℝ) - (-1 : ℤ) ^ 0 = -3 := by
  sorry

end eval_expression_l25_25705


namespace intersection_sets_l25_25490

def universal_set : Set ℝ := Set.univ
def set_A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def set_B : Set ℝ := {x | -3 < x ∧ x < 4}

theorem intersection_sets (x : ℝ) : 
  (x ∈ set_A ∩ set_B) ↔ (-2 < x ∧ x < 4) :=
by sorry

end intersection_sets_l25_25490


namespace twenty_percent_correct_l25_25039

def certain_number := 400
def forty_percent (x : ℕ) : ℕ := 40 * x / 100
def twenty_percent_of_certain_number (x : ℕ) : ℕ := 20 * x / 100

theorem twenty_percent_correct : 
  (∃ x : ℕ, forty_percent x = 160) → twenty_percent_of_certain_number certain_number = 80 :=
by
  sorry

end twenty_percent_correct_l25_25039


namespace supplementary_angle_difference_l25_25459

theorem supplementary_angle_difference (a b : ℝ) (h1 : a + b = 180) (h2 : 5 * b = 3 * a) : abs (a - b) = 45 :=
  sorry

end supplementary_angle_difference_l25_25459


namespace sum_of_f_l25_25801

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

theorem sum_of_f :
  f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 3 * Real.sqrt 2 :=
by
  sorry

end sum_of_f_l25_25801


namespace melanie_books_bought_l25_25464

def books_before_yard_sale : ℝ := 41.0
def books_after_yard_sale : ℝ := 128
def books_bought : ℝ := books_after_yard_sale - books_before_yard_sale

theorem melanie_books_bought : books_bought = 87 := by
  sorry

end melanie_books_bought_l25_25464


namespace original_people_in_room_l25_25609

theorem original_people_in_room (x : ℝ) (h1 : x / 3 * 2 / 2 = 18) : x = 54 :=
sorry

end original_people_in_room_l25_25609


namespace quadratic_one_real_root_l25_25165

theorem quadratic_one_real_root (m : ℝ) : 
  (∃ x : ℝ, (x^2 - 6*m*x + 2*m = 0) ∧ 
    (∀ y : ℝ, (y^2 - 6*m*y + 2*m = 0) → y = x)) → 
  m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_l25_25165


namespace total_points_l25_25097

theorem total_points (darius_score marius_score matt_score total_points : ℕ) 
    (h1 : darius_score = 10) 
    (h2 : marius_score = darius_score + 3) 
    (h3 : matt_score = darius_score + 5) 
    (h4 : total_points = darius_score + marius_score + matt_score) : 
    total_points = 38 :=
by sorry

end total_points_l25_25097


namespace find_sum_of_abc_l25_25368

variable (a b c : ℝ)

-- Given conditions
axiom h1 : a^2 + a * b + b^2 = 1
axiom h2 : b^2 + b * c + c^2 = 3
axiom h3 : c^2 + c * a + a^2 = 4

-- Positivity constraints
axiom ha : a > 0
axiom hb : b > 0
axiom hc : c > 0

theorem find_sum_of_abc : a + b + c = Real.sqrt 7 := 
by
  sorry

end find_sum_of_abc_l25_25368


namespace coffee_customers_l25_25275

theorem coffee_customers (C : ℕ) :
  let coffee_cost := 5
  let tea_ordered := 8
  let tea_cost := 4
  let total_revenue := 67
  (coffee_cost * C + tea_ordered * tea_cost = total_revenue) → C = 7 := by
  sorry

end coffee_customers_l25_25275


namespace product_of_three_consecutive_natural_numbers_divisible_by_six_l25_25401

theorem product_of_three_consecutive_natural_numbers_divisible_by_six (n : ℕ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
by
  sorry

end product_of_three_consecutive_natural_numbers_divisible_by_six_l25_25401


namespace rectangles_with_trapezoid_area_l25_25981

-- Define the necessary conditions
def small_square_area : ℝ := 1
def total_squares : ℕ := 12
def rows : ℕ := 4
def columns : ℕ := 3
def trapezoid_area : ℝ := 3

-- Statement of the proof problem
theorem rectangles_with_trapezoid_area :
  (∀ rows columns : ℕ, rows * columns = total_squares) →
  (∀ area : ℝ, area = small_square_area) →
  (∀ trapezoid_area : ℝ, trapezoid_area = 3) →
  (rows = 4) →
  (columns = 3) →
  ∃ rectangles : ℕ, rectangles = 10 :=
by
  sorry

end rectangles_with_trapezoid_area_l25_25981


namespace disjunction_of_false_is_false_l25_25168

-- Given conditions
variables (p q : Prop)

-- We are given the assumption that both p and q are false propositions
axiom h1 : ¬ p
axiom h2 : ¬ q

-- We want to prove that the disjunction p ∨ q is false
theorem disjunction_of_false_is_false (p q : Prop) (h1 : ¬ p) (h2 : ¬ q) : ¬ (p ∨ q) := 
by
  sorry

end disjunction_of_false_is_false_l25_25168


namespace evaluate_fraction_l25_25618

theorem evaluate_fraction : (25 * 5 + 5^2) / (5^2 - 15) = 15 := 
by
  sorry

end evaluate_fraction_l25_25618


namespace negation_of_exists_l25_25872

theorem negation_of_exists (x : ℝ) : x^2 + 2 * x + 2 > 0 := sorry

end negation_of_exists_l25_25872


namespace minimum_area_of_square_on_parabola_l25_25885

theorem minimum_area_of_square_on_parabola :
  ∃ (A B C : ℝ × ℝ), 
  (∃ (x₁ x₂ x₃ : ℝ), (A = (x₁, x₁^2)) ∧ (B = (x₂, x₂^2)) ∧ (C = (x₃, x₃^2)) 
  ∧ x₁ < x₂ ∧ x₂ < x₃ 
  ∧ ∀ S : ℝ, (S = (1 + (x₃ + x₂)^2) * ((x₂ - x₃) - (x₃ - x₂))^2) → S ≥ 2) :=
sorry

end minimum_area_of_square_on_parabola_l25_25885


namespace price_per_glass_first_day_l25_25914

variables (O G : ℝ) (P1 : ℝ)

theorem price_per_glass_first_day (H1 : G * P1 = 1.5 * G * 0.40) : 
  P1 = 0.60 :=
by sorry

end price_per_glass_first_day_l25_25914


namespace incentive_given_to_john_l25_25628

-- Conditions (definitions)
def commission_held : ℕ := 25000
def advance_fees : ℕ := 8280
def amount_given_to_john : ℕ := 18500

-- Problem statement
theorem incentive_given_to_john : (amount_given_to_john - (commission_held - advance_fees)) = 1780 := 
by
  sorry

end incentive_given_to_john_l25_25628


namespace total_marks_prove_total_marks_l25_25244

def average_marks : ℝ := 40
def number_of_candidates : ℕ := 50

theorem total_marks (average_marks : ℝ) (number_of_candidates : ℕ) : Real :=
  average_marks * number_of_candidates

theorem prove_total_marks : total_marks average_marks number_of_candidates = 2000 := 
by
  sorry

end total_marks_prove_total_marks_l25_25244


namespace domain_change_l25_25035

theorem domain_change (f : ℝ → ℝ) :
  (∀ x : ℝ, -2 ≤ x + 1 ∧ x + 1 ≤ 3) →
  (∀ x : ℝ, -2 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 3) →
  ∀ x : ℝ, -3 / 2 ≤ x ∧ x ≤ 1 :=
by {
  sorry
}

end domain_change_l25_25035


namespace mul_mod_remainder_l25_25502

theorem mul_mod_remainder (a b m : ℕ)
  (h₁ : a ≡ 8 [MOD 9])
  (h₂ : b ≡ 1 [MOD 9]) :
  (a * b) % 9 = 8 := 
  sorry

def main : IO Unit :=
  IO.println "The theorem statement has been defined."

end mul_mod_remainder_l25_25502


namespace dartboard_odd_sum_probability_l25_25101

theorem dartboard_odd_sum_probability :
  let innerR := 4
  let outerR := 8
  let inner_points := [3, 1, 1]
  let outer_points := [2, 3, 3]
  let total_area := π * outerR^2
  let inner_area := π * innerR^2
  let outer_area := total_area - inner_area
  let each_inner_area := inner_area / 3
  let each_outer_area := outer_area / 3
  let odd_area := 2 * each_inner_area + 2 * each_outer_area
  let even_area := each_inner_area + each_outer_area
  let P_odd := odd_area / total_area
  let P_even := even_area / total_area
  let odd_sum_prob := 2 * (P_odd * P_even)
  odd_sum_prob = 4 / 9 := by
    sorry

end dartboard_odd_sum_probability_l25_25101


namespace contrapositive_l25_25915

theorem contrapositive (x : ℝ) : (x > 1 → x^2 + x > 2) ↔ (x^2 + x ≤ 2 → x ≤ 1) :=
sorry

end contrapositive_l25_25915


namespace simplify_expression_1_simplify_expression_2_l25_25474

-- Define the algebraic simplification problem for the first expression
theorem simplify_expression_1 (x y : ℝ) : 5 * x - 3 * (2 * x - 3 * y) + x = 9 * y :=
by
  sorry

-- Define the algebraic simplification problem for the second expression
theorem simplify_expression_2 (a : ℝ) : 3 * a^2 + 5 - 2 * a^2 - 2 * a + 3 * a - 8 = a^2 + a - 3 :=
by
  sorry

end simplify_expression_1_simplify_expression_2_l25_25474


namespace S7_eq_14_l25_25908

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variables (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith_seq : arithmetic_sequence a)
variables (h_a3 : a 3 = 0) (h_a6_plus_a7 : a 6 + a 7 = 14)

theorem S7_eq_14 : S 7 = 14 := sorry

end S7_eq_14_l25_25908


namespace salaries_of_a_and_b_l25_25314

theorem salaries_of_a_and_b {x y : ℝ}
  (h1 : x + y = 5000)
  (h2 : 0.05 * x = 0.15 * y) :
  x = 3750 :=
by sorry

end salaries_of_a_and_b_l25_25314


namespace tan_960_eq_sqrt_3_l25_25336

theorem tan_960_eq_sqrt_3 : Real.tan (960 * Real.pi / 180) = Real.sqrt 3 := by
  sorry

end tan_960_eq_sqrt_3_l25_25336


namespace problem1_problem2_l25_25178

namespace ArithmeticSequence

-- Part (1)
theorem problem1 (a1 : ℚ) (d : ℚ) (S_n : ℚ) (n : ℕ) (a_n : ℚ) 
  (h1 : a1 = 5 / 6) 
  (h2 : d = -1 / 6) 
  (h3 : S_n = -5) 
  (h4 : S_n = n * (2 * a1 + (n - 1) * d) / 2) 
  (h5 : a_n = a1 + (n - 1) * d) : 
  (n = 15) ∧ (a_n = -3 / 2) :=
sorry

-- Part (2)
theorem problem2 (d : ℚ) (n : ℕ) (a_n : ℚ) (a1 : ℚ) (S_n : ℚ)
  (h1 : d = 2) 
  (h2 : n = 15) 
  (h3 : a_n = -10) 
  (h4 : a_n = a1 + (n - 1) * d) 
  (h5 : S_n = n * (2 * a1 + (n - 1) * d) / 2) : 
  (a1 = -38) ∧ (S_n = -360) :=
sorry

end ArithmeticSequence

end problem1_problem2_l25_25178


namespace total_salary_l25_25730

-- Define the salaries and conditions.
def salaryN : ℝ := 280
def salaryM : ℝ := 1.2 * salaryN

-- State the theorem we want to prove
theorem total_salary : salaryM + salaryN = 616 :=
by
  sorry

end total_salary_l25_25730


namespace tens_digit_of_2023_pow_2024_minus_2025_l25_25653

theorem tens_digit_of_2023_pow_2024_minus_2025 : 
  ∀ (n : ℕ), n = 2023^2024 - 2025 → ((n % 100) / 10) = 0 :=
by
  intros n h
  sorry

end tens_digit_of_2023_pow_2024_minus_2025_l25_25653


namespace prob_B_draws_given_A_draws_black_fairness_l25_25951

noncomputable def event_A1 : Prop := true  -- A draws the red ball
noncomputable def event_A2 : Prop := true  -- B draws the red ball
noncomputable def event_A3 : Prop := true  -- C draws the red ball

noncomputable def prob_A1 : ℝ := 1 / 3
noncomputable def prob_not_A1 : ℝ := 2 / 3
noncomputable def prob_A2_given_not_A1 : ℝ := 1 / 2

theorem prob_B_draws_given_A_draws_black : (prob_not_A1 * prob_A2_given_not_A1) / prob_not_A1 = 1 / 2 := by
  sorry

theorem fairness :
  let prob_A1 := 1 / 3
  let prob_A2 := prob_not_A1 * prob_A2_given_not_A1
  let prob_A3 := prob_not_A1 * prob_A2_given_not_A1 * 1
  prob_A1 = prob_A2 ∧ prob_A2 = prob_A3 := by
  sorry

end prob_B_draws_given_A_draws_black_fairness_l25_25951


namespace atomic_weight_chlorine_l25_25527

-- Define the given conditions and constants
def molecular_weight_compound : ℝ := 53
def atomic_weight_nitrogen : ℝ := 14.01
def atomic_weight_hydrogen : ℝ := 1.01
def number_of_hydrogen_atoms : ℝ := 4
def number_of_nitrogen_atoms : ℝ := 1

-- Define the total weight of nitrogen and hydrogen in the compound
def total_weight_nh : ℝ := (number_of_nitrogen_atoms * atomic_weight_nitrogen) + (number_of_hydrogen_atoms * atomic_weight_hydrogen)

-- Define the statement to be proved: the atomic weight of chlorine
theorem atomic_weight_chlorine : (molecular_weight_compound - total_weight_nh) = 34.95 := by
  sorry

end atomic_weight_chlorine_l25_25527


namespace find_original_price_l25_25377

theorem find_original_price (SP GP : ℝ) (h_SP : SP = 1150) (h_GP : GP = 27.77777777777778) :
  ∃ CP : ℝ, CP = 900 :=
by
  sorry

end find_original_price_l25_25377


namespace solve_equation_l25_25657

theorem solve_equation : ∀ (x : ℝ), 2 * (x - 1) = 2 - (5 * x - 2) → x = 6 / 7 :=
by
  sorry

end solve_equation_l25_25657


namespace symmetric_circle_eq_a_l25_25984

theorem symmetric_circle_eq_a :
  ∀ (a : ℝ), (∀ x y : ℝ, (x^2 + y^2 - a * x + 2 * y + 1 = 0) ↔ (∃ x y : ℝ, (x - y = 1) ∧ ( x^2 + y^2 = 1))) → a = 2 :=
by
  sorry

end symmetric_circle_eq_a_l25_25984


namespace range_of_m_l25_25148

theorem range_of_m (x y m : ℝ) 
  (h1: 3 * x + y = 1 + 3 * m) 
  (h2: x + 3 * y = 1 - m) 
  (h3: x + y > 0) : 
  m > -1 :=
sorry

end range_of_m_l25_25148


namespace perpendicular_planes_condition_l25_25742

variables (α β : Plane) (m : Line) 

-- Assuming the basic definitions:
def perpendicular (α β : Plane) : Prop := sorry
def in_plane (m : Line) (α : Plane) : Prop := sorry
def perpendicular_to_plane (m : Line) (β : Plane) : Prop := sorry

-- Conditions
axiom α_diff_β : α ≠ β
axiom m_in_α : in_plane m α

-- Proving the necessary but not sufficient condition
theorem perpendicular_planes_condition : 
  (perpendicular α β → perpendicular_to_plane m β) ∧ 
  (¬ perpendicular_to_plane m β → ¬ perpendicular α β) ∧ 
  ¬ (perpendicular_to_plane m β → perpendicular α β) :=
sorry

end perpendicular_planes_condition_l25_25742


namespace proof_problem_l25_25715

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

end proof_problem_l25_25715


namespace poly_coeff_sum_l25_25655

theorem poly_coeff_sum (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ) :
  (2 * x + 3)^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + 
                 a_3 * (x + 1)^3 + a_4 * (x + 1)^4 + 
                 a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
                 a_7 * (x + 1)^7 + a_8 * (x + 1)^8 →
  a_0 + a_2 + a_4 + a_6 + a_8 = 3281 :=
by
  sorry

end poly_coeff_sum_l25_25655


namespace simplify_fraction_l25_25641

theorem simplify_fraction (a b : ℤ) (h : a = 2^6 + 2^4) (h1 : b = 2^5 - 2^2) : 
  (a / b : ℚ) = 20 / 7 := by
  sorry

end simplify_fraction_l25_25641


namespace refills_needed_l25_25580

theorem refills_needed 
  (cups_per_day : ℕ)
  (bottle_capacity_oz : ℕ)
  (oz_per_cup : ℕ)
  (total_oz : ℕ)
  (refills : ℕ)
  (h1 : cups_per_day = 12)
  (h2 : bottle_capacity_oz = 16)
  (h3 : oz_per_cup = 8)
  (h4 : total_oz = cups_per_day * oz_per_cup)
  (h5 : refills = total_oz / bottle_capacity_oz) :
  refills = 6 :=
by
  sorry

end refills_needed_l25_25580


namespace slower_pipe_time_l25_25004

/-
One pipe can fill a tank four times as fast as another pipe. 
If together the two pipes can fill the tank in 40 minutes, 
how long will it take for the slower pipe alone to fill the tank?
-/

theorem slower_pipe_time (t : ℕ) (h1 : ∀ t, 1/t + 4/t = 1/40) : t = 200 :=
sorry

end slower_pipe_time_l25_25004


namespace translate_line_up_l25_25483

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := -2 * x

-- Define the transformed line equation as a function
def translated_line (x : ℝ) : ℝ := -2 * x + 1

-- Prove that translating the original line upward by 1 unit gives the translated line
theorem translate_line_up (x : ℝ) :
  original_line x + 1 = translated_line x :=
by
  unfold original_line translated_line
  simp

end translate_line_up_l25_25483


namespace solve_equation_l25_25707

theorem solve_equation (x y : ℝ) : 
    ((16 * x^2 + 1) * (y^2 + 1) = 16 * x * y) ↔ 
        ((x = 1/4 ∧ y = 1) ∨ (x = -1/4 ∧ y = -1)) := 
by
  sorry

end solve_equation_l25_25707


namespace fred_seashells_l25_25739

def seashells_given : ℕ := 25
def seashells_left : ℕ := 22
def seashells_found : ℕ := 47

theorem fred_seashells :
  seashells_found = seashells_given + seashells_left :=
  by sorry

end fred_seashells_l25_25739


namespace value_of_x_l25_25252

theorem value_of_x : (2015^2 + 2015 - 1) / (2015 : ℝ) = 2016 - 1 / 2015 := 
  sorry

end value_of_x_l25_25252


namespace twice_minus_three_algebraic_l25_25975

def twice_minus_three (x : ℝ) : ℝ := 2 * x - 3

theorem twice_minus_three_algebraic (x : ℝ) : 
  twice_minus_three x = 2 * x - 3 :=
by sorry

end twice_minus_three_algebraic_l25_25975


namespace find_y_l25_25808

theorem find_y (y : ℝ) (h : 9 * y^3 = y * 81) : y = 3 * Real.sqrt 3 :=
by
  sorry

end find_y_l25_25808


namespace total_cost_l25_25881

/-- Sam initially has s yellow balloons.
He gives away a of these balloons to Fred.
Mary has m yellow balloons.
Each balloon costs c dollars.
Determine the total cost for the remaining balloons that Sam and Mary jointly have.
Given: s = 6.0, a = 5.0, m = 7.0, c = 9.0 dollars.
Expected result: the total cost is 72.0 dollars.
-/
theorem total_cost (s a m c : ℝ) (h_s : s = 6.0) (h_a : a = 5.0) (h_m : m = 7.0) (h_c : c = 9.0) :
  (s - a + m) * c = 72.0 := 
by
  rw [h_s, h_a, h_m, h_c]
  -- At this stage, the proof would involve showing the expression is 72.0, but since no proof is required:
  sorry

end total_cost_l25_25881


namespace complex_number_in_second_quadrant_l25_25110

theorem complex_number_in_second_quadrant :
  let z := (2 + 4 * Complex.I) / (1 + Complex.I) 
  ∃ (im : ℂ), z = im ∧ im.re < 0 ∧ 0 < im.im := by
  sorry

end complex_number_in_second_quadrant_l25_25110


namespace value_of_a_plus_b_l25_25558

variable {F : Type} [Field F]

theorem value_of_a_plus_b (a b : F) (h1 : ∀ x, x ≠ 0 → a + b / x = 2 ↔ x = -2)
                                      (h2 : ∀ x, x ≠ 0 → a + b / x = 6 ↔ x = -6) :
  a + b = 20 :=
sorry

end value_of_a_plus_b_l25_25558


namespace triangle_at_most_one_obtuse_l25_25600

-- Define the notion of a triangle and obtuse angle
def isTriangle (A B C: ℝ) : Prop := (A + B > C) ∧ (A + C > B) ∧ (B + C > A)
def isObtuseAngle (theta: ℝ) : Prop := 90 < theta ∧ theta < 180

-- A theorem to prove that a triangle cannot have more than one obtuse angle 
theorem triangle_at_most_one_obtuse (A B C: ℝ) (angleA angleB angleC : ℝ) 
    (h1 : isTriangle A B C)
    (h2 : isObtuseAngle angleA)
    (h3 : isObtuseAngle angleB)
    (h4 : angleA + angleB + angleC = 180):
    false :=
by
  sorry

end triangle_at_most_one_obtuse_l25_25600


namespace train_length_l25_25863

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : L = v * 36)
  (h2 : L + 25 = v * 39) :
  L = 300 :=
by
  sorry

end train_length_l25_25863


namespace find_coordinates_l25_25906

def A : Prod ℤ ℤ := (-3, 2)
def move_right (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst + 1, p.snd)
def move_down (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst, p.snd - 2)

theorem find_coordinates :
  move_down (move_right A) = (-2, 0) :=
by
  sorry

end find_coordinates_l25_25906


namespace number_of_pipes_l25_25053

theorem number_of_pipes (d_large d_small: ℝ) (π : ℝ) (h1: d_large = 4) (h2: d_small = 2) : 
  ((π * (d_large / 2)^2) / (π * (d_small / 2)^2) = 4) := 
by
  sorry

end number_of_pipes_l25_25053


namespace num_rectangular_arrays_with_36_chairs_l25_25728

theorem num_rectangular_arrays_with_36_chairs :
  ∃ n : ℕ, (∀ r c : ℕ, r * c = 36 ∧ r ≥ 2 ∧ c ≥ 2 ↔ n = 7) :=
sorry

end num_rectangular_arrays_with_36_chairs_l25_25728


namespace number_of_columns_per_section_l25_25276

variables (S C : ℕ)

-- Define the first condition: S * C + (S - 1) / 2 = 1223
def condition1 := S * C + (S - 1) / 2 = 1223

-- Define the second condition: S = 2 * C + 5
def condition2 := S = 2 * C + 5

-- Formulate the theorem that C = 23 given the two conditions
theorem number_of_columns_per_section
  (h1 : condition1 S C)
  (h2 : condition2 S C) :
  C = 23 :=
sorry

end number_of_columns_per_section_l25_25276


namespace angle_of_inclination_l25_25001

theorem angle_of_inclination 
  (α : ℝ) 
  (h_tan : Real.tan α = -Real.sqrt 3)
  (h_range : 0 ≤ α ∧ α < 180) : α = 120 :=
by
  sorry

end angle_of_inclination_l25_25001


namespace trajectory_equation_of_P_l25_25505

variable {x y : ℝ}
variable (A B P : ℝ × ℝ)

def in_line_through (a b : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  let k := (p.2 - a.2) / (p.1 - a.1)
  (b.2 - a.2) / (b.1 - a.1) = k

theorem trajectory_equation_of_P
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : in_line_through A B P)
  (slope_product : (P.2 / (P.1 + 1)) * (P.2 / (P.1 - 1)) = -1) :
  P.1 ^ 2 + P.2 ^ 2 = 1 ∧ P.1 ≠ 1 ∧ P.1 ≠ -1 := 
sorry

end trajectory_equation_of_P_l25_25505


namespace solve_years_later_twice_age_l25_25767

-- Define the variables and the given conditions
def man_age (S: ℕ) := S + 25
def years_later_twice_age (S M: ℕ) (Y: ℕ) := (M + Y = 2 * (S + Y))

-- Given conditions
def present_age_son := 23
def present_age_man := man_age present_age_son

theorem solve_years_later_twice_age :
  ∃ Y, years_later_twice_age present_age_son present_age_man Y ∧ Y = 2 := by
  sorry

end solve_years_later_twice_age_l25_25767


namespace solution_set_inequality_l25_25677

theorem solution_set_inequality (a x : ℝ) (h : a > 0) :
  (∀ x, (a + 1 ≤ x ∧ x ≤ a + 3) ↔ (|((2 * x - 3 - 2 * a) / (x - a))| ≤ 1)) := 
sorry

end solution_set_inequality_l25_25677


namespace arithmetic_sequence_seventh_term_l25_25469

theorem arithmetic_sequence_seventh_term (a d : ℚ) 
  (h1 : a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = 15)
  (h2 : a + 5 * d = 8) :
  a + 6 * d = 29 / 3 := 
sorry

end arithmetic_sequence_seventh_term_l25_25469


namespace probability_point_inside_circle_l25_25626

theorem probability_point_inside_circle :
  (∃ (m n : ℕ), 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) →
  (∃ (P : ℚ), P = 2/9) :=
by
  sorry

end probability_point_inside_circle_l25_25626


namespace arithmetic_seq_S13_l25_25552

noncomputable def arithmetic_sequence_sum (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_seq_S13 (a_1 d : ℕ) (h : a_1 + 6 * d = 10) :
  arithmetic_sequence_sum a_1 d 13 = 130 :=
by
  sorry

end arithmetic_seq_S13_l25_25552


namespace mike_training_hours_l25_25822

-- Define the individual conditions
def first_weekday_hours : Nat := 2
def first_weekend_hours : Nat := 1
def first_week_days : Nat := 5
def first_weekend_days : Nat := 2

def second_weekday_hours : Nat := 3
def second_weekend_hours : Nat := 2
def second_week_days : Nat := 4  -- since the first day of second week is a rest day
def second_weekend_days : Nat := 2

def first_week_hours : Nat := (first_weekday_hours * first_week_days) + (first_weekend_hours * first_weekend_days)
def second_week_hours : Nat := (second_weekday_hours * second_week_days) + (second_weekend_hours * second_weekend_days)

def total_training_hours : Nat := first_week_hours + second_week_hours

-- The final proof statement
theorem mike_training_hours : total_training_hours = 28 := by
  exact sorry

end mike_training_hours_l25_25822


namespace inequality_arith_geo_mean_l25_25812

theorem inequality_arith_geo_mean (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / Real.sqrt b + b / Real.sqrt a) ≥ (Real.sqrt a + Real.sqrt b) :=
by
  sorry

end inequality_arith_geo_mean_l25_25812


namespace arun_age_in_6_years_l25_25744

theorem arun_age_in_6_years
  (A D n : ℕ)
  (h1 : D = 42)
  (h2 : A = (5 * D) / 7)
  (h3 : A + n = 36) 
  : n = 6 :=
by
  sorry

end arun_age_in_6_years_l25_25744


namespace first_term_geometric_progression_l25_25717

theorem first_term_geometric_progression (S a : ℝ) (r : ℝ) 
  (h1 : S = 10) 
  (h2 : a = 10 * (1 - r)) 
  (h3 : a * (1 + r) = 7) : 
  a = 10 * (1 - Real.sqrt (3 / 10)) ∨ a = 10 * (1 + Real.sqrt (3 / 10)) := 
by 
  sorry

end first_term_geometric_progression_l25_25717


namespace product_ab_zero_l25_25722

variable {a b : ℝ}

theorem product_ab_zero (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
  sorry

end product_ab_zero_l25_25722


namespace piglets_each_ate_6_straws_l25_25282

theorem piglets_each_ate_6_straws (total_straws : ℕ) (fraction_for_adult_pigs : ℚ) (piglets : ℕ) 
  (h1 : total_straws = 300) 
  (h2 : fraction_for_adult_pigs = 3/5) 
  (h3 : piglets = 20) :
  (total_straws * (1 - fraction_for_adult_pigs) / piglets) = 6 :=
by
  sorry

end piglets_each_ate_6_straws_l25_25282


namespace units_digit_5_pow_2023_l25_25008

theorem units_digit_5_pow_2023 : ∀ n : ℕ, (n > 0) → (5^n % 10 = 5) → (5^2023 % 10 = 5) :=
by
  intros n hn hu
  have h_units_digit : ∀ k : ℕ, (k > 0) → 5^k % 10 = 5 := by
    intro k hk
    sorry -- pattern proof not included
  exact h_units_digit 2023 (by norm_num)

end units_digit_5_pow_2023_l25_25008


namespace eccentricity_range_l25_25310

noncomputable def ellipse_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) : Prop :=
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ e = c / a ∧ (2 * ((-a) * (c + a / 2) - (b / 2) * b) + b^2 + c^2 ≥ 0)

theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ e : ℝ, ellipse_eccentricity_range a b h e ∧ (0 < e ∧ e ≤ -1 + Real.sqrt 3) :=
sorry

end eccentricity_range_l25_25310


namespace customers_not_tipping_l25_25007

theorem customers_not_tipping (number_of_customers tip_per_customer total_earned_in_tips : ℕ)
  (h_number : number_of_customers = 7)
  (h_tip : tip_per_customer = 3)
  (h_earned : total_earned_in_tips = 6) :
  number_of_customers - (total_earned_in_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_not_tipping_l25_25007


namespace second_quadrant_necessary_not_sufficient_l25_25047

variable (α : ℝ)

def is_obtuse (α : ℝ) : Prop := 90 < α ∧ α < 180
def is_second_quadrant (α : ℝ) : Prop := 90 < α ∧ α < 180

theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → is_second_quadrant α) ∧ ¬ (∀ α, is_second_quadrant α → is_obtuse α) := by
  sorry

end second_quadrant_necessary_not_sufficient_l25_25047


namespace calculate_expression_l25_25132

variable (x y : ℝ)

theorem calculate_expression :
  (-2 * x^2 * y)^3 = -8 * x^6 * y^3 :=
by 
  sorry

end calculate_expression_l25_25132


namespace triangle_overlap_angle_is_30_l25_25224

noncomputable def triangle_rotation_angle (hypotenuse : ℝ) (overlap_ratio : ℝ) :=
  if hypotenuse = 10 ∧ overlap_ratio = 0.5 then 30 else sorry

theorem triangle_overlap_angle_is_30 :
  triangle_rotation_angle 10 0.5 = 30 :=
sorry

end triangle_overlap_angle_is_30_l25_25224


namespace lottery_probability_l25_25683

theorem lottery_probability :
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  let winnerBallProbability := 1 / winnerBallCombination
  megaBallProbability * winnerBallProbability = 1 / 63562800 :=
by
  let megaBallProbability := 1 / 30
  let winnerBallCombination := Nat.choose 50 5
  have winnerBallCombinationEval: winnerBallCombination = 2118760 := by sorry
  let winnerBallProbability := 1 / winnerBallCombination
  have totalProbability: megaBallProbability * winnerBallProbability = 1 / 63562800 := by sorry
  exact totalProbability

end lottery_probability_l25_25683


namespace hexagon_perimeter_l25_25982

-- Definitions of the conditions
def side_length : ℕ := 5
def number_of_sides : ℕ := 6

-- The perimeter of the hexagon
def perimeter : ℕ := side_length * number_of_sides

-- Proof statement
theorem hexagon_perimeter : perimeter = 30 :=
by
  sorry

end hexagon_perimeter_l25_25982


namespace min_value_expression_l25_25206

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1/a + (a/b^2) + b) ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l25_25206


namespace alice_chicken_weight_l25_25579

theorem alice_chicken_weight (total_cost_needed : ℝ)
  (amount_to_spend_more : ℝ)
  (cost_lettuce : ℝ)
  (cost_tomatoes : ℝ)
  (sweet_potato_quantity : ℝ)
  (cost_per_sweet_potato : ℝ)
  (broccoli_quantity : ℝ)
  (cost_per_broccoli : ℝ)
  (brussel_sprouts_weight : ℝ)
  (cost_per_brussel_sprouts : ℝ)
  (cost_per_pound_chicken : ℝ)
  (total_cost_excluding_chicken : ℝ) :
  total_cost_needed = 35 ∧
  amount_to_spend_more = 11 ∧
  cost_lettuce = 3 ∧
  cost_tomatoes = 2.5 ∧
  sweet_potato_quantity = 4 ∧
  cost_per_sweet_potato = 0.75 ∧
  broccoli_quantity = 2 ∧
  cost_per_broccoli = 2 ∧
  brussel_sprouts_weight = 1 ∧
  cost_per_brussel_sprouts = 2.5 ∧
  total_cost_excluding_chicken = (cost_lettuce + cost_tomatoes + sweet_potato_quantity * cost_per_sweet_potato + broccoli_quantity * cost_per_broccoli + brussel_sprouts_weight * cost_per_brussel_sprouts) →
  (total_cost_needed - amount_to_spend_more - total_cost_excluding_chicken) / cost_per_pound_chicken = 1.5 :=
by
  intros
  sorry

end alice_chicken_weight_l25_25579


namespace forty_percent_of_number_l25_25875

/--
Given that (1/4) * (1/3) * (2/5) * N = 30, prove that 0.40 * N = 360.
-/
theorem forty_percent_of_number {N : ℝ} (h : (1/4 : ℝ) * (1/3) * (2/5) * N = 30) : 0.40 * N = 360 := 
by
  sorry

end forty_percent_of_number_l25_25875


namespace union_complement_with_B_l25_25993

namespace SetTheory

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of A relative to U in Lean
def C_U (A U : Set ℕ) : Set ℕ := U \ A

-- Theorem statement
theorem union_complement_with_B (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {0, 1, 2, 3}) (hB : B = {2, 3, 4}) : 
  (C_U A U) ∪ B = {2, 3, 4} :=
by
  -- Proof goes here
  sorry

end SetTheory

end union_complement_with_B_l25_25993


namespace lateral_surface_area_cut_off_l25_25895

theorem lateral_surface_area_cut_off {a b c d : ℝ} (h₁ : a = 4) (h₂ : b = 25) 
(h₃ : c = (2/5 : ℝ)) (h₄ : d = 2 * (4 / 25) * b) : 
4 + 10 + (1/4 * b) = 20.25 :=
by
  sorry

end lateral_surface_area_cut_off_l25_25895


namespace rides_total_l25_25152

theorem rides_total (rides_day1 rides_day2 : ℕ) (h1 : rides_day1 = 4) (h2 : rides_day2 = 3) : rides_day1 + rides_day2 = 7 := 
by 
  sorry

end rides_total_l25_25152


namespace no_digit_C_makes_2C4_multiple_of_5_l25_25997

theorem no_digit_C_makes_2C4_multiple_of_5 : ∀ (C : ℕ), (2 * 100 + C * 10 + 4 ≠ 0 ∨ 2 * 100 + C * 10 + 4 ≠ 5) := 
by 
  intros C
  have h : 4 ≠ 0 := by norm_num
  have h2 : 4 ≠ 5 := by norm_num
  sorry

end no_digit_C_makes_2C4_multiple_of_5_l25_25997


namespace flu_infection_equation_l25_25590

theorem flu_infection_equation
  (x : ℝ) :
  (1 + x)^2 = 25 :=
sorry

end flu_infection_equation_l25_25590


namespace degree_of_monomial_x_l25_25709

def is_monomial (e : Expr) : Prop := sorry -- Placeholder definition
def degree (e : Expr) : Nat := sorry -- Placeholder definition

theorem degree_of_monomial_x :
  degree x = 1 :=
by
  sorry

end degree_of_monomial_x_l25_25709


namespace maria_average_speed_l25_25038

noncomputable def average_speed (total_distance : ℕ) (total_time : ℕ) : ℚ :=
  total_distance / total_time

theorem maria_average_speed :
  average_speed 200 7 = 28 + 4 / 7 :=
sorry

end maria_average_speed_l25_25038


namespace reflect_across_y_axis_l25_25332

-- Definition of the original point A
def pointA : ℝ × ℝ := (2, 3)

-- Definition of the reflected point across the y-axis
def reflectedPoint (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- The theorem stating the reflection result
theorem reflect_across_y_axis : reflectedPoint pointA = (-2, 3) :=
by
  -- Proof (skipped)
  sorry

end reflect_across_y_axis_l25_25332


namespace largest_real_solution_l25_25591

theorem largest_real_solution (x : ℝ) (h : (⌊x⌋ / x = 7 / 8)) : x ≤ 48 / 7 := by
  sorry

end largest_real_solution_l25_25591


namespace prob_B_hits_once_prob_hits_with_ABC_l25_25866

section
variable (P_A P_B P_C : ℝ)
variable (hA : P_A = 1 / 2)
variable (hB : P_B = 1 / 3)
variable (hC : P_C = 1 / 4)

-- Part (Ⅰ): Probability of hitting the target exactly once when B shoots twice
theorem prob_B_hits_once : 
  (P_B * (1 - P_B) + (1 - P_B) * P_B) = 4 / 9 := 
by
  rw [hB]
  sorry

-- Part (Ⅱ): Probability of hitting the target when A, B, and C each shoot once
theorem prob_hits_with_ABC :
  (1 - ((1 - P_A) * (1 - P_B) * (1 - P_C))) = 3 / 4 := 
by
  rw [hA, hB, hC]
  sorry

end

end prob_B_hits_once_prob_hits_with_ABC_l25_25866


namespace cauliflower_sales_l25_25752

theorem cauliflower_sales :
  let total_earnings := 500
  let b_sales := 57
  let c_sales := 2 * b_sales
  let s_sales := (c_sales / 2) + 16
  let t_sales := b_sales + s_sales
  let ca_sales := total_earnings - (b_sales + c_sales + s_sales + t_sales)
  ca_sales = 126 := by
  sorry

end cauliflower_sales_l25_25752


namespace Elmer_eats_more_than_Penelope_l25_25757

noncomputable def Penelope_food := 20
noncomputable def Greta_food := Penelope_food / 10
noncomputable def Milton_food := Greta_food / 100
noncomputable def Elmer_food := 4000 * Milton_food

theorem Elmer_eats_more_than_Penelope :
  Elmer_food - Penelope_food = 60 := 
by
  sorry

end Elmer_eats_more_than_Penelope_l25_25757


namespace cos_alpha_minus_half_beta_l25_25171

theorem cos_alpha_minus_half_beta
  (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : -π / 2 < β ∧ β < 0)
  (h3 : Real.cos (π / 4 + α) = 1 / 3)
  (h4 : Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α - β / 2) = Real.sqrt 6 / 3 :=
by
  sorry

end cos_alpha_minus_half_beta_l25_25171


namespace range_of_b_l25_25888

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 / 2 then (2 * x + 1) / (x ^ 2) else x + 1

def g (x : ℝ) : ℝ := x ^ 2 - 4 * x - 4

-- The main theorem to prove the range of b
theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : b ∈ Set.Icc (-1) 5 := by
  sorry

end range_of_b_l25_25888


namespace calc_is_a_pow4_l25_25995

theorem calc_is_a_pow4 (a : ℕ) : (a^2)^2 = a^4 := 
by 
  sorry

end calc_is_a_pow4_l25_25995


namespace no_nat_solutions_m_sq_eq_n_sq_plus_2014_l25_25530

theorem no_nat_solutions_m_sq_eq_n_sq_plus_2014 :
  ¬ ∃ (m n : ℕ), m ^ 2 = n ^ 2 + 2014 := 
sorry

end no_nat_solutions_m_sq_eq_n_sq_plus_2014_l25_25530


namespace cleaning_time_l25_25636

def lara_rate := 1 / 4
def chris_rate := 1 / 6
def combined_rate := lara_rate + chris_rate

theorem cleaning_time (t : ℝ) : 
  (combined_rate * (t - 2) = 1) ↔ (t = 22 / 5) :=
by
  sorry

end cleaning_time_l25_25636


namespace number_of_chips_per_day_l25_25748

def total_chips : ℕ := 100
def chips_first_day : ℕ := 10
def total_days : ℕ := 10
def days_remaining : ℕ := total_days - 1
def chips_remaining : ℕ := total_chips - chips_first_day

theorem number_of_chips_per_day : 
  chips_remaining / days_remaining = 10 :=
by 
  unfold chips_remaining days_remaining total_chips chips_first_day total_days
  sorry

end number_of_chips_per_day_l25_25748


namespace reciprocal_pair_c_l25_25106

def is_reciprocal (a b : ℝ) : Prop :=
  a * b = 1

theorem reciprocal_pair_c :
  is_reciprocal (-2) (-1/2) :=
by sorry

end reciprocal_pair_c_l25_25106


namespace complex_in_fourth_quadrant_l25_25900

theorem complex_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8*m + 15 > 0) ∧ (m^2 - 5*m - 14 < 0) →
  (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
sorry

end complex_in_fourth_quadrant_l25_25900


namespace exponent_multiplication_l25_25556

theorem exponent_multiplication (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3^a)^b = 3^3) : 3^a * 3^b = 3^4 :=
by
  sorry

end exponent_multiplication_l25_25556


namespace suitable_altitude_range_l25_25154

theorem suitable_altitude_range :
  ∀ (temperature_at_base : ℝ) (temp_decrease_per_100m : ℝ) (suitable_temp_low : ℝ) (suitable_temp_high : ℝ) (altitude_at_base : ℝ),
  (22 = temperature_at_base) →
  (0.5 = temp_decrease_per_100m) →
  (18 = suitable_temp_low) →
  (20 = suitable_temp_high) →
  (0 = altitude_at_base) →
  400 ≤ ((temperature_at_base - suitable_temp_high) / temp_decrease_per_100m * 100) ∧ ((temperature_at_base - suitable_temp_low) / temp_decrease_per_100m * 100) ≤ 800 :=
by
  intros temperature_at_base temp_decrease_per_100m suitable_temp_low suitable_temp_high altitude_at_base
  intro h1 h2 h3 h4 h5
  sorry

end suitable_altitude_range_l25_25154


namespace mary_income_percent_of_juan_l25_25986

variable (J : ℝ)
variable (T : ℝ)
variable (M : ℝ)

-- Conditions
def tim_income := T = 0.60 * J
def mary_income := M = 1.40 * T

-- Theorem to prove that Mary's income is 84 percent of Juan's income
theorem mary_income_percent_of_juan : tim_income J T → mary_income T M → M = 0.84 * J :=
by
  sorry

end mary_income_percent_of_juan_l25_25986


namespace evaluate_expression_l25_25842

def f (x : ℕ) : ℕ := 3 * x - 4
def g (x : ℕ) : ℕ := x - 1

theorem evaluate_expression : f (1 + g 3) = 5 := by
  sorry

end evaluate_expression_l25_25842


namespace basic_astrophysics_degrees_l25_25791

-- Define the given percentages
def microphotonics_percentage : ℝ := 14
def home_electronics_percentage : ℝ := 24
def food_additives_percentage : ℝ := 10
def gmo_percentage : ℝ := 29
def industrial_lubricants_percentage : ℝ := 8
def total_circle_degrees : ℝ := 360

-- Define a proof problem to show that basic astrophysics research occupies 54 degrees in the circle
theorem basic_astrophysics_degrees :
  total_circle_degrees - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage + gmo_percentage + industrial_lubricants_percentage) = 15 ∧
  0.15 * total_circle_degrees = 54 :=
by
  sorry

end basic_astrophysics_degrees_l25_25791


namespace gcd_and_lcm_of_18_and_24_l25_25961

-- Definitions of gcd and lcm for the problem's context
def my_gcd (a b : ℕ) : ℕ := a.gcd b
def my_lcm (a b : ℕ) : ℕ := a.lcm b

-- Constants given in the problem
def a := 18
def b := 24

-- Proof problem statement
theorem gcd_and_lcm_of_18_and_24 : my_gcd a b = 6 ∧ my_lcm a b = 72 := by
  sorry

end gcd_and_lcm_of_18_and_24_l25_25961


namespace probability_is_1_div_28_l25_25760

noncomputable def probability_valid_combinations : ℚ :=
  let total_combinations := Nat.choose 8 3
  let valid_combinations := 2
  valid_combinations / total_combinations

theorem probability_is_1_div_28 :
  probability_valid_combinations = 1 / 28 := by
  sorry

end probability_is_1_div_28_l25_25760


namespace arithmetic_sequence_sum_l25_25513

theorem arithmetic_sequence_sum {a b : ℤ} (h : ∀ n : ℕ, 3 + n * 6 = if n = 2 then a else if n = 3 then b else 33) : a + b = 48 := by
  sorry

end arithmetic_sequence_sum_l25_25513


namespace max_correct_answers_l25_25214

variable (x y z : ℕ)

theorem max_correct_answers
  (h1 : x + y + z = 100)
  (h2 : x - 3 * y - 2 * z = 50) :
  x ≤ 87 := by
    sorry

end max_correct_answers_l25_25214


namespace always_positive_sum_l25_25943

def f : ℝ → ℝ := sorry  -- assuming f(x) is provided elsewhere

theorem always_positive_sum (f : ℝ → ℝ)
    (h1 : ∀ x, f x = -f (2 - x))
    (h2 : ∀ x, x < 1 → f (x) < f (x + 1))
    (x1 x2 : ℝ)
    (h3 : x1 + x2 > 2)
    (h4 : (x1 - 1) * (x2 - 1) < 0) :
  f x1 + f x2 > 0 :=
by {
  sorry
}

end always_positive_sum_l25_25943


namespace part1_solution_set_part2_solution_l25_25255

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem part1_solution_set :
  {x : ℝ | f x > 2} = {x | x > 1} ∪ {x | x < -5} :=
by
  sorry

theorem part2_solution (t : ℝ) :
  (∀ x, f x ≥ t^2 - (11 / 2) * t) ↔ (1 / 2 ≤ t ∧ t ≤ 5) :=
by
  sorry

end part1_solution_set_part2_solution_l25_25255


namespace inequality_solution_l25_25737

theorem inequality_solution (x : ℝ) : x^3 - 12 * x^2 > -36 * x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end inequality_solution_l25_25737


namespace percentage_mutant_frogs_is_33_l25_25938

def num_extra_legs_frogs := 5
def num_two_heads_frogs := 2
def num_bright_red_frogs := 2
def num_normal_frogs := 18

def total_mutant_frogs := num_extra_legs_frogs + num_two_heads_frogs + num_bright_red_frogs
def total_frogs := total_mutant_frogs + num_normal_frogs

theorem percentage_mutant_frogs_is_33 :
  Float.round (100 * total_mutant_frogs.toFloat / total_frogs.toFloat) = 33 :=
by 
  -- placeholder for the proof
  sorry

end percentage_mutant_frogs_is_33_l25_25938


namespace distance_between_tangency_points_l25_25731

theorem distance_between_tangency_points
  (circle_radius : ℝ) (M_distance : ℝ) (A_distance : ℝ) 
  (h1 : circle_radius = 7)
  (h2 : M_distance = 25)
  (h3 : A_distance = 7) :
  ∃ AB : ℝ, AB = 48 :=
by
  -- Definitions and proofs will go here.
  sorry

end distance_between_tangency_points_l25_25731


namespace hyperbola_eccentricity_l25_25555

-- Define the context/conditions
noncomputable def hyperbola_vertex_to_asymptote_distance (a b e : ℝ) : Prop :=
  (2 = b / e)

noncomputable def hyperbola_focus_to_asymptote_distance (a b e : ℝ) : Prop :=
  (6 = b)

-- Define the main theorem to prove the eccentricity
theorem hyperbola_eccentricity (a b e : ℝ) (h1 : hyperbola_vertex_to_asymptote_distance a b e) (h2 : hyperbola_focus_to_asymptote_distance a b e) : 
  e = 3 := 
sorry 

end hyperbola_eccentricity_l25_25555


namespace expected_sixes_correct_l25_25200

-- Define probabilities for rolling individual numbers on a die
def P (n : ℕ) (k : ℕ) : ℚ := if k = n then 1 / 6 else 0

-- Expected value calculation for two dice
noncomputable def expected_sixes_two_dice_with_resets : ℚ :=
(0 * (13/18)) + (1 * (2/9)) + (2 * (1/36))

-- Main theorem to prove
theorem expected_sixes_correct :
  expected_sixes_two_dice_with_resets = 5 / 18 :=
by
  -- The actual proof steps go here; added sorry to skip the proof.
  sorry

end expected_sixes_correct_l25_25200


namespace power_mod_eq_five_l25_25634

theorem power_mod_eq_five
  (m : ℕ)
  (h₀ : 0 ≤ m)
  (h₁ : m < 8)
  (h₂ : 13^5 % 8 = m) : m = 5 :=
by 
  sorry

end power_mod_eq_five_l25_25634


namespace solution_set_of_inequality_l25_25091

theorem solution_set_of_inequality (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 1) * (a * x - 4) < 0} = {x : ℝ | x > 1 ∨ x < 4 / a} :=
sorry

end solution_set_of_inequality_l25_25091


namespace lines_parallel_if_perpendicular_to_plane_l25_25607

axiom line : Type
axiom plane : Type

-- Definitions of perpendicular and parallel
axiom perp : line → plane → Prop
axiom parallel : line → line → Prop

variables (a b : line) (α : plane)

theorem lines_parallel_if_perpendicular_to_plane (h1 : perp a α) (h2 : perp b α) : parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_plane_l25_25607


namespace max_three_m_plus_four_n_l25_25161

theorem max_three_m_plus_four_n (m n : ℕ) 
  (h : m * (m + 1) + n ^ 2 = 1987) : 3 * m + 4 * n ≤ 221 :=
sorry

end max_three_m_plus_four_n_l25_25161


namespace pyramid_volume_l25_25577

-- Definitions based on the given conditions
def AB : ℝ := 15
def AD : ℝ := 8
def Area_Δ_ABE : ℝ := 120
def Area_Δ_CDE : ℝ := 64
def h : ℝ := 16
def Base_Area : ℝ := AB * AD

-- Statement to prove the volume of the pyramid is 640
theorem pyramid_volume : (1 / 3) * Base_Area * h = 640 :=
sorry

end pyramid_volume_l25_25577


namespace sum_ab_eq_five_l25_25681

theorem sum_ab_eq_five (a b : ℕ) (h : (∃ (ab : ℕ), ab = a * 10 + b ∧ 3 / 13 = ab / 100)) : a + b = 5 :=
sorry

end sum_ab_eq_five_l25_25681


namespace value_of_fraction_of_power_l25_25306

-- Define the values in the problem
def a : ℝ := 6
def b : ℝ := 30

-- The problem asks us to prove
theorem value_of_fraction_of_power : 
  (1 / 3) * (a ^ b) = 2 * (a ^ (b - 1)) :=
by
  -- Initial Setup
  let c := (1 / 3) * (a ^ b)
  let d := 2 * (a ^ (b - 1))
  -- The main claim
  show c = d
  sorry

end value_of_fraction_of_power_l25_25306


namespace decimal_to_binary_45_l25_25836

theorem decimal_to_binary_45 :
  (45 : ℕ) = (0b101101 : ℕ) :=
sorry

end decimal_to_binary_45_l25_25836


namespace find_f_2008_l25_25740

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the problem statement with all given conditions
theorem find_f_2008 (h_odd : is_odd f) (h_f2 : f 2 = 0) (h_rec : ∀ x, f (x + 4) = f x + f 4) : f 2008 = 0 := 
sorry

end find_f_2008_l25_25740


namespace Tanya_bought_9_apples_l25_25461

def original_fruit_count : ℕ := 18
def remaining_fruit_count : ℕ := 9
def pears_count : ℕ := 6
def pineapples_count : ℕ := 2
def plums_basket_count : ℕ := 1

theorem Tanya_bought_9_apples : 
  remaining_fruit_count * 2 = original_fruit_count →
  original_fruit_count - (pears_count + pineapples_count + plums_basket_count) = 9 :=
by
  intros h1
  sorry

end Tanya_bought_9_apples_l25_25461


namespace total_bins_l25_25280

-- Definition of the problem conditions
def road_length : ℕ := 400
def placement_interval : ℕ := 20
def bins_per_side : ℕ := (road_length / placement_interval) - 1

-- Statement of the problem
theorem total_bins : 2 * bins_per_side = 38 := by
  sorry

end total_bins_l25_25280


namespace sum_abc_l25_25477

variable {a b c : ℝ}
variables (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0)
variables (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))

theorem sum_abc (h1 : a * b = 2 * (a + b)) (h2 : b * c = 3 * (b + c)) (h3 : c * a = 4 * (c + a))
   (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) :
   a + b + c = 1128 / 35 := 
sorry

end sum_abc_l25_25477


namespace cream_ratio_l25_25422

noncomputable def joe_coffee_initial := 14
noncomputable def joe_coffee_drank := 3
noncomputable def joe_cream_added := 3

noncomputable def joann_coffee_initial := 14
noncomputable def joann_cream_added := 3
noncomputable def joann_mixture_stirred := 17
noncomputable def joann_amount_drank := 3

theorem cream_ratio (joe_coffee_initial joe_coffee_drank joe_cream_added 
                     joann_coffee_initial joann_cream_added joann_mixture_stirred 
                     joann_amount_drank : ℝ) : 
  (joe_coffee_initial - joe_coffee_drank + joe_cream_added) / 
  (joann_cream_added - (joann_amount_drank * (joann_cream_added / joann_mixture_stirred))) = 17 / 14 :=
by
  -- Prove the theorem statement
  sorry

end cream_ratio_l25_25422


namespace equivalent_lengthEF_l25_25215

namespace GeometryProof

noncomputable def lengthEF 
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : ℝ := 
  50

theorem equivalent_lengthEF
  (AB CD EF : ℝ) 
  (h_AB_parallel_CD : true) 
  (h_lengthAB : AB = 200) 
  (h_lengthCD : CD = 50) 
  (h_angleEF : true) 
  : lengthEF AB CD EF h_AB_parallel_CD h_lengthAB h_lengthCD h_angleEF = 50 :=
by
  sorry

end GeometryProof

end equivalent_lengthEF_l25_25215


namespace molecular_weight_BaO_is_correct_l25_25317

-- Define the atomic weights
def atomic_weight_Ba : ℝ := 137.33
def atomic_weight_O : ℝ := 16.00

-- Define the molecular weight of BaO as the sum of atomic weights of Ba and O
def molecular_weight_BaO := atomic_weight_Ba + atomic_weight_O

-- Theorem stating the molecular weight of BaO
theorem molecular_weight_BaO_is_correct : molecular_weight_BaO = 153.33 := by
  -- Proof can be filled in
  sorry

end molecular_weight_BaO_is_correct_l25_25317


namespace range_of_m_l25_25357

noncomputable def common_points (k : ℝ) (m : ℝ) := 
  ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)

theorem range_of_m (k : ℝ) (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)) ↔ 
  (m ∈ (Set.Ioo 1 5 ∪ Set.Ioi 5)) :=
by
  sorry

end range_of_m_l25_25357


namespace set_intersection_l25_25054

open Set

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def intersection : Set ℕ := {1, 3}

theorem set_intersection : M ∩ N = intersection := by
  sorry

end set_intersection_l25_25054


namespace exponent_equality_l25_25058

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end exponent_equality_l25_25058


namespace fiona_correct_answers_l25_25589

-- 5 marks for each correct answer in Questions 1-15
def marks_questions_1_to_15 (correct1 : ℕ) : ℕ := 5 * correct1

-- 6 marks for each correct answer in Questions 16-25
def marks_questions_16_to_25 (correct2 : ℕ) : ℕ := 6 * correct2

-- 1 mark penalty for incorrect answers in Questions 16-20
def penalty_questions_16_to_20 (incorrect1 : ℕ) : ℕ := incorrect1

-- 2 mark penalty for incorrect answers in Questions 21-25
def penalty_questions_21_to_25 (incorrect2 : ℕ) : ℕ := 2 * incorrect2

-- Total marks given correct and incorrect answers
def total_marks (correct1 correct2 incorrect1 incorrect2 : ℕ) : ℕ :=
  marks_questions_1_to_15 correct1 +
  marks_questions_16_to_25 correct2 -
  penalty_questions_16_to_20 incorrect1 -
  penalty_questions_21_to_25 incorrect2

-- Fiona's total score
def fionas_total_score : ℕ := 80

-- The proof problem: Fiona answered 16 questions correctly
theorem fiona_correct_answers (correct1 correct2 incorrect1 incorrect2 : ℕ) :
  total_marks correct1 correct2 incorrect1 incorrect2 = fionas_total_score → 
  (correct1 + correct2 = 16) := sorry

end fiona_correct_answers_l25_25589


namespace woman_traveled_by_bus_l25_25172

noncomputable def travel_by_bus : ℕ :=
  let total_distance := 1800
  let distance_by_plane := total_distance / 4
  let distance_by_train := total_distance / 6
  let distance_by_taxi := total_distance / 8
  let remaining_distance := total_distance - (distance_by_plane + distance_by_train + distance_by_taxi)
  let distance_by_rental := remaining_distance * 2 / 3
  distance_by_rental / 2

theorem woman_traveled_by_bus :
  travel_by_bus = 275 :=
by 
  sorry

end woman_traveled_by_bus_l25_25172


namespace tangent_line_iff_l25_25818

theorem tangent_line_iff (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 8 * y + 12 = 0 → ax + y + 2 * a = 0) ↔ a = -3 / 4 :=
by
  sorry

end tangent_line_iff_l25_25818


namespace total_time_is_60_l25_25884

def emma_time : ℕ := 20
def fernando_time : ℕ := 2 * emma_time
def total_time : ℕ := emma_time + fernando_time

theorem total_time_is_60 : total_time = 60 := by
  sorry

end total_time_is_60_l25_25884


namespace complement_intersection_l25_25108

open Set -- Open namespace for set operations

-- Define the universal set I
def I : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define set A
def A : Set ℕ := {1, 2, 3, 4}

-- Define set B
def B : Set ℕ := {3, 4, 5, 6}

-- Define the intersection A ∩ B
def A_inter_B : Set ℕ := A ∩ B

-- Define the complement C_I(S) as I \ S, where S is a subset of I
def complement (S : Set ℕ) : Set ℕ := I \ S

-- Prove that the complement of A ∩ B in I is {1, 2, 5, 6}
theorem complement_intersection : complement A_inter_B = {1, 2, 5, 6} :=
by
  sorry -- Proof to be provided

end complement_intersection_l25_25108


namespace largest_value_is_E_l25_25802

theorem largest_value_is_E :
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  E > A ∧ E > B ∧ E > C ∧ E > D := 
by
  let A := 3 + 1 + 2 + 9
  let B := 3 * 1 + 2 + 9
  let C := 3 + 1 * 2 + 9
  let D := 3 + 1 + 2 * 9
  let E := 3 * 1 * 2 * 9
  sorry

end largest_value_is_E_l25_25802


namespace slightly_used_crayons_l25_25725

theorem slightly_used_crayons (total_crayons : ℕ) (percent_new : ℚ) (percent_broken : ℚ) 
  (h1 : total_crayons = 250) (h2 : percent_new = 40/100) (h3 : percent_broken = 1/5) : 
  (total_crayons - percent_new * total_crayons - percent_broken * total_crayons) = 100 :=
by
  -- sorry here to indicate the proof is omitted
  sorry

end slightly_used_crayons_l25_25725


namespace find_a_value_l25_25828

theorem find_a_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq1 : a^b = b^a) (h_eq2 : b = 3 * a) : a = Real.sqrt 3 :=
  sorry

end find_a_value_l25_25828


namespace smallest_sum_of_xy_l25_25512

theorem smallest_sum_of_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y)
  (hcond : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 10) : x + y = 45 :=
sorry

end smallest_sum_of_xy_l25_25512


namespace smallest_x_l25_25565

theorem smallest_x (x : ℕ) : 
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 9 = 8) ↔ x = 314 := 
by
  sorry

end smallest_x_l25_25565


namespace distinct_arrangements_l25_25105

-- Definitions based on the conditions
def boys : ℕ := 4
def girls : ℕ := 4
def total_people : ℕ := boys + girls
def arrangements : ℕ := Nat.factorial boys * Nat.factorial (total_people - 2) * Nat.factorial 6

-- Main statement: Verify the number of distinct arrangements
theorem distinct_arrangements : arrangements = 8640 := by
  -- We will replace this proof with our Lean steps (which is currently omitted)
  sorry

end distinct_arrangements_l25_25105


namespace find_geo_prog_numbers_l25_25328

noncomputable def geo_prog_numbers (a1 a2 a3 : ℝ) : Prop :=
a1 * a2 * a3 = 27 ∧ a1 + a2 + a3 = 13

theorem find_geo_prog_numbers :
  geo_prog_numbers 1 3 9 ∨ geo_prog_numbers 9 3 1 :=
sorry

end find_geo_prog_numbers_l25_25328


namespace ounces_of_wax_for_car_l25_25625

noncomputable def ounces_wax_for_SUV : ℕ := 4
noncomputable def initial_wax_amount : ℕ := 11
noncomputable def wax_spilled : ℕ := 2
noncomputable def wax_left_after_detailing : ℕ := 2
noncomputable def total_wax_used : ℕ := initial_wax_amount - wax_spilled - wax_left_after_detailing

theorem ounces_of_wax_for_car :
  (initial_wax_amount - wax_spilled - wax_left_after_detailing) - ounces_wax_for_SUV = 3 :=
by
  sorry

end ounces_of_wax_for_car_l25_25625


namespace t_shirt_cost_l25_25313

theorem t_shirt_cost
  (marked_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (selling_price : ℝ)
  (cost : ℝ)
  (h1 : marked_price = 240)
  (h2 : discount_rate = 0.20)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = 0.8 * marked_price)
  (h5 : selling_price = cost + profit_rate * cost)
  : cost = 160 := 
sorry

end t_shirt_cost_l25_25313


namespace honey_nectar_relationship_l25_25758

-- Definitions representing the conditions
def nectarA_water_content (x : ℝ) := 0.7 * x
def nectarB_water_content (y : ℝ) := 0.5 * y
def final_honey_water_content := 0.3
def evaporation_loss (initial_content : ℝ) := 0.15 * initial_content

-- The system of equations to prove
theorem honey_nectar_relationship (x y : ℝ) :
  (x + y = 1) ∧ (0.595 * x + 0.425 * y = 0.3) :=
sorry

end honey_nectar_relationship_l25_25758


namespace product_of_local_and_absolute_value_l25_25484

def localValue (n : ℕ) (digit : ℕ) : ℕ :=
  match n with
  | 564823 =>
    match digit with
    | 4 => 4000
    | _ => 0 -- only defining for digit 4 as per problem
  | _ => 0 -- only case for 564823 is considered

def absoluteValue (x : ℤ) : ℤ := if x < 0 then -x else x

theorem product_of_local_and_absolute_value:
  localValue 564823 4 * absoluteValue 4 = 16000 :=
by
  sorry

end product_of_local_and_absolute_value_l25_25484


namespace num_ordered_pairs_l25_25481

theorem num_ordered_pairs : ∃! n : ℕ, n = 4 ∧ 
  ∃ (x y : ℤ), y = (x - 90)^2 - 4907 ∧ 
  (∃ m : ℕ, y = m^2) := 
sorry

end num_ordered_pairs_l25_25481


namespace distance_to_destination_l25_25383

theorem distance_to_destination (x : ℕ) 
    (condition_1 : True)  -- Manex is a tour bus driver. Ignore in the proof.
    (condition_2 : True)  -- Ignores the fact that the return trip is using a different path.
    (condition_3 : x / 30 + (x + 10) / 30 + 2 = 6) : 
    x = 55 :=
sorry

end distance_to_destination_l25_25383


namespace find_S6_l25_25084

def arithmetic_sum (n : ℕ) : ℝ := sorry
def S_3 := 6
def S_9 := 27

theorem find_S6 : ∃ S_6 : ℝ, S_6 = 15 ∧ 
                              S_6 - S_3 = (6 + (S_9 - S_6)) / 2 :=
sorry

end find_S6_l25_25084


namespace Jill_earnings_l25_25673

theorem Jill_earnings :
  let earnings_first_month := 10 * 30
  let earnings_second_month := 20 * 30
  let earnings_third_month := 20 * 15
  earnings_first_month + earnings_second_month + earnings_third_month = 1200 :=
by
  sorry

end Jill_earnings_l25_25673


namespace no_integer_y_such_that_abs_g_y_is_prime_l25_25754

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 → m ≤ n → m ∣ n → m = 1 ∨ m = n

def g (y : ℤ) : ℤ := 8 * y^2 - 55 * y + 21

theorem no_integer_y_such_that_abs_g_y_is_prime : 
  ∀ y : ℤ, ¬ is_prime (|g y|) :=
by sorry

end no_integer_y_such_that_abs_g_y_is_prime_l25_25754


namespace boat_travel_time_downstream_l25_25139

theorem boat_travel_time_downstream
  (v c: ℝ)
  (h1: c = 1)
  (h2: 24 / (v - c) = 6): 
  24 / (v + c) = 4 := 
by
  sorry

end boat_travel_time_downstream_l25_25139


namespace intersection_M_N_l25_25229

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := 
by
  -- Proof to be provided
  sorry

end intersection_M_N_l25_25229


namespace solve_x_l25_25994

noncomputable def diamond (a b : ℝ) : ℝ := a / b

axiom diamond_assoc (a b c : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (c_nonzero : c ≠ 0) : 
  diamond a (diamond b c) = a / (b / c)

axiom diamond_id (a : ℝ) (a_nonzero : a ≠ 0) : diamond a a = 1

theorem solve_x (x : ℝ) (h₁ : 1008 ≠ 0) (h₂ : 12 ≠ 0) (h₃ : x ≠ 0) : diamond 1008 (diamond 12 x) = 50 → x = 25 / 42 :=
by
  sorry

end solve_x_l25_25994


namespace find_x_l25_25704

noncomputable def value_of_x (x : ℝ) := (5 * x) ^ 4 = (15 * x) ^ 3

theorem find_x : ∀ (x : ℝ), (value_of_x x) ∧ (x ≠ 0) → x = 27 / 5 :=
by
  intro x
  intro h
  sorry

end find_x_l25_25704


namespace find_third_card_value_l25_25331

noncomputable def point_values (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 13 ∧
  1 ≤ b ∧ b ≤ 13 ∧
  1 ≤ c ∧ c ≤ 13 ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b = 25 ∧
  b + c = 13

theorem find_third_card_value :
  ∃ a b c : ℕ, point_values a b c ∧ c = 1 :=
by {
  sorry
}

end find_third_card_value_l25_25331


namespace fractional_part_of_water_after_replacements_l25_25425

theorem fractional_part_of_water_after_replacements :
  let total_quarts := 25
  let removed_quarts := 5
  (1 - removed_quarts / (total_quarts : ℚ))^3 = 64 / 125 :=
by
  sorry

end fractional_part_of_water_after_replacements_l25_25425


namespace spherical_to_rectangular_example_l25_25253

noncomputable def spherical_to_rectangular (ρ θ ϕ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin ϕ * Real.cos θ, ρ * Real.sin ϕ * Real.sin θ, ρ * Real.cos ϕ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 4 (Real.pi / 4) (Real.pi / 6) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end spherical_to_rectangular_example_l25_25253


namespace digit_in_tens_place_l25_25198

theorem digit_in_tens_place (n : ℕ) (cycle : List ℕ) (h_cycle : cycle = [16, 96, 76, 56]) (hk : n % 4 = 3) :
  (6 ^ n % 100) / 10 % 10 = 7 := by
  sorry

end digit_in_tens_place_l25_25198


namespace polynomial_remainder_l25_25833

theorem polynomial_remainder (a b : ℤ) :
  (∀ x : ℤ, 3 * x ^ 6 - 2 * x ^ 4 + 5 * x ^ 2 - 9 = (x + 1) * (x + 2) * (q : ℤ) + a * x + b) →
  (a = -174 ∧ b = -177) :=
by sorry

end polynomial_remainder_l25_25833


namespace best_play_wins_probability_best_play_wins_with_certainty_l25_25371

-- Define the conditions

variables (n : ℕ)

-- Part (a): Probability that the best play wins
theorem best_play_wins_probability (hn_pos : 0 < n) : 
  1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) = 1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) :=
  by sorry

-- Part (b): With more than two plays, the best play wins with certainty
theorem best_play_wins_with_certainty (s : ℕ) (hs : 2 < s) : 
  1 = 1 :=
  by sorry

end best_play_wins_probability_best_play_wins_with_certainty_l25_25371


namespace julia_played_with_kids_on_Monday_l25_25284

theorem julia_played_with_kids_on_Monday (k_wednesday : ℕ) (k_monday : ℕ)
  (h1 : k_wednesday = 4) (h2 : k_monday = k_wednesday + 2) : k_monday = 6 := 
by
  sorry

end julia_played_with_kids_on_Monday_l25_25284


namespace neither_coffee_nor_tea_l25_25560

theorem neither_coffee_nor_tea (total_businesspeople coffee_drinkers tea_drinkers both_drinkers : ℕ) 
    (h_total : total_businesspeople = 35)
    (h_coffee : coffee_drinkers = 18)
    (h_tea : tea_drinkers = 15)
    (h_both : both_drinkers = 6) :
    (total_businesspeople - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end neither_coffee_nor_tea_l25_25560


namespace min_value_of_expression_l25_25829

theorem min_value_of_expression :
  ∀ (x y : ℝ), ∃ a b : ℝ, x = 5 ∧ y = -3 ∧ (x^2 + y^2 - 10*x + 6*y + 25) = -9 := 
by
  sorry

end min_value_of_expression_l25_25829


namespace exists_constant_a_l25_25403

theorem exists_constant_a (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : (m : ℝ) / n < Real.sqrt 7) :
  ∃ (a : ℝ), a > 1 ∧ (7 - (m^2 : ℝ) / (n^2 : ℝ) ≥ a / (n^2 : ℝ)) ∧ a = 3 :=
by
  sorry

end exists_constant_a_l25_25403


namespace dana_more_pencils_than_marcus_l25_25185

theorem dana_more_pencils_than_marcus :
  ∀ (Jayden Dana Marcus : ℕ), 
  (Jayden = 20) ∧ 
  (Dana = Jayden + 15) ∧ 
  (Jayden = 2 * Marcus) → 
  (Dana - Marcus = 25) :=
by
  intros Jayden Dana Marcus h
  rcases h with ⟨hJayden, hDana, hMarcus⟩
  sorry

end dana_more_pencils_than_marcus_l25_25185


namespace isosceles_triangle_l25_25593

theorem isosceles_triangle
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β)) :
  α = β ∨ α = γ ∨ β = γ :=
sorry

end isosceles_triangle_l25_25593


namespace least_possible_sum_l25_25296

theorem least_possible_sum (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x + y + z = 26 :=
by sorry

end least_possible_sum_l25_25296


namespace tan_double_angle_l25_25898

theorem tan_double_angle (α : Real) (h1 : Real.sin α - Real.cos α = 4 / 3) (h2 : α ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4)) :
  Real.tan (2 * α) = (7 * Real.sqrt 2) / 8 :=
by
  sorry

end tan_double_angle_l25_25898


namespace christopher_strolling_time_l25_25322

theorem christopher_strolling_time
  (initial_distance : ℝ) (initial_speed : ℝ) (break_time : ℝ)
  (continuation_distance : ℝ) (continuation_speed : ℝ)
  (H1 : initial_distance = 2) (H2 : initial_speed = 4)
  (H3 : break_time = 0.25) (H4 : continuation_distance = 3)
  (H5 : continuation_speed = 6) :
  (initial_distance / initial_speed + break_time + continuation_distance / continuation_speed) = 1.25 := 
  sorry

end christopher_strolling_time_l25_25322


namespace man_rate_in_still_water_l25_25926

theorem man_rate_in_still_water (Vm Vs : ℝ) :
  Vm + Vs = 20 ∧ Vm - Vs = 8 → Vm = 14 :=
by
  sorry

end man_rate_in_still_water_l25_25926


namespace pioneer_ages_l25_25685

def pioneer_data (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ) :=
  Burov ≠ Kolya ∧
  Petya_age = 12 ∧
  Gridnev_age = Petya_age + 1 ∧
  Grisha_age = Petya_age + 1 ∧
  Burov_age = Grisha_age ∧
-- defining the names corresponding to conditions given in problem
  Burov = Grisha ∧ Gridnev = Kolya ∧ Klimenko = Petya 

theorem pioneer_ages (Burov Gridnev Klimenko Kolya Petya Grisha : String) (Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age : ℕ)
  (h : pioneer_data Burov Gridnev Klimenko Kolya Petya Grisha Burov_age Gridnev_age Klimenko_age Petya_age Grisha_age) :
  (Burov, Burov_age) = (Grisha, 13) ∧ 
  (Gridnev, Gridnev_age) = (Kolya, 13) ∧ 
  (Klimenko, Klimenko_age) = (Petya, 12) :=
by
  sorry

end pioneer_ages_l25_25685


namespace triangle_side_possible_values_l25_25373

theorem triangle_side_possible_values (m : ℝ) (h1 : 1 < m) (h2 : m < 7) : 
  m = 5 :=
by
  sorry

end triangle_side_possible_values_l25_25373


namespace krishan_money_l25_25735

-- Define the constants
def Ram : ℕ := 490
def ratio1 : ℕ := 7
def ratio2 : ℕ := 17

-- Defining the relationship
def ratio_RG (Ram Gopal : ℕ) : Prop := Ram / Gopal = ratio1 / ratio2
def ratio_GK (Gopal Krishan : ℕ) : Prop := Gopal / Krishan = ratio1 / ratio2

-- Define the problem
theorem krishan_money (R G K : ℕ) (h1 : R = Ram) (h2 : ratio_RG R G) (h3 : ratio_GK G K) : K = 2890 :=
by
  sorry

end krishan_money_l25_25735


namespace larger_number_is_70380_l25_25159

theorem larger_number_is_70380 (A B : ℕ) 
    (hcf : Nat.gcd A B = 20) 
    (lcm : Nat.lcm A B = 20 * 9 * 17 * 23) :
    max A B = 70380 :=
  sorry

end larger_number_is_70380_l25_25159


namespace ratio_final_to_initial_l25_25535

def initial_amount (P : ℝ) := P
def interest_rate := 4 / 100
def time_period := 25

def simple_interest (P : ℝ) := P * interest_rate * time_period

def final_amount (P : ℝ) := P + simple_interest P

theorem ratio_final_to_initial (P : ℝ) (hP : P > 0) :
  final_amount P / initial_amount P = 2 := by
  sorry

end ratio_final_to_initial_l25_25535


namespace contractor_work_done_l25_25854

def initial_people : ℕ := 10
def remaining_people : ℕ := 8
def total_days : ℕ := 100
def remaining_days : ℕ := 75
def fraction_done : ℚ := 1/4
def total_work : ℚ := 1

theorem contractor_work_done (x : ℕ) 
  (h1 : initial_people * x = fraction_done * total_work) 
  (h2 : remaining_people * remaining_days = (1 - fraction_done) * total_work) :
  x = 60 :=
by
  sorry

end contractor_work_done_l25_25854


namespace probability_of_event_correct_l25_25770

def within_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ Real.pi

def tan_in_range (x : ℝ) : Prop :=
  -1 ≤ Real.tan x ∧ Real.tan x ≤ Real.sqrt 3

def valid_subintervals (x : ℝ) : Prop :=
  within_interval x ∧ tan_in_range x

def interval_length (a b : ℝ) : ℝ :=
  b - a

noncomputable def probability_of_event : ℝ :=
  (interval_length 0 (Real.pi / 3) + interval_length (3 * Real.pi / 4) Real.pi) / Real.pi

theorem probability_of_event_correct :
  probability_of_event = 7 / 12 := sorry

end probability_of_event_correct_l25_25770


namespace ab_equals_six_l25_25423

theorem ab_equals_six (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l25_25423


namespace inequality_proof_l25_25581

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) : 
  a^4 + b^4 + c^4 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end inequality_proof_l25_25581


namespace prove_expression_l25_25010

theorem prove_expression (a : ℝ) (h : a^2 + a - 1 = 0) : 2 * a^2 + 2 * a + 2008 = 2010 := by
  sorry

end prove_expression_l25_25010


namespace probability_of_two_white_balls_correct_l25_25016

noncomputable def probability_of_two_white_balls : ℚ :=
  let total_balls := 15
  let white_balls := 8
  let first_draw_white := (white_balls : ℚ) / total_balls
  let second_draw_white := (white_balls - 1 : ℚ) / (total_balls - 1)
  first_draw_white * second_draw_white

theorem probability_of_two_white_balls_correct :
  probability_of_two_white_balls = 4 / 15 :=
by
  sorry

end probability_of_two_white_balls_correct_l25_25016


namespace binom_11_1_l25_25958

theorem binom_11_1 : Nat.choose 11 1 = 11 :=
by
  sorry

end binom_11_1_l25_25958


namespace kevin_food_expense_l25_25909

theorem kevin_food_expense
    (total_budget : ℕ)
    (samuel_ticket : ℕ)
    (samuel_food_drinks : ℕ)
    (kevin_ticket : ℕ)
    (kevin_drinks : ℕ)
    (kevin_total_exp : ℕ) :
    total_budget = 20 →
    samuel_ticket = 14 →
    samuel_food_drinks = 6 →
    kevin_ticket = 14 →
    kevin_drinks = 2 →
    kevin_total_exp = 20 →
    kevin_food = 4 :=
by
  sorry

end kevin_food_expense_l25_25909


namespace total_marks_l25_25361

-- Variables and conditions
variables (M C P : ℕ)
variable (h1 : C = P + 20)
variable (h2 : (M + C) / 2 = 40)

-- Theorem statement
theorem total_marks (M C P : ℕ) (h1 : C = P + 20) (h2 : (M + C) / 2 = 40) : M + P = 60 :=
sorry

end total_marks_l25_25361


namespace boy_scouts_signed_slips_l25_25467

-- Definitions for the problem conditions have only been used; solution steps are excluded.

theorem boy_scouts_signed_slips (total_scouts : ℕ) (signed_slips : ℕ) (boy_scouts : ℕ) (girl_scouts : ℕ)
  (boy_scouts_signed : ℕ) (girl_scouts_signed : ℕ)
  (h1 : signed_slips = 4 * total_scouts / 5)  -- 80% of the scouts arrived with signed permission slips
  (h2 : boy_scouts = 2 * total_scouts / 5)  -- 40% of the scouts were boy scouts
  (h3 : girl_scouts = total_scouts - boy_scouts)  -- Rest are girl scouts
  (h4 : girl_scouts_signed = 8333 * girl_scouts / 10000)  -- 83.33% of girl scouts with permission slips
  (h5 : signed_slips = boy_scouts_signed + girl_scouts_signed)  -- Total signed slips by both boy and girl scouts
  : (boy_scouts_signed * 100 / boy_scouts = 75) :=    -- 75% of boy scouts with permission slips
by
  -- Proof to be filled in.
  sorry

end boy_scouts_signed_slips_l25_25467


namespace day_crew_fraction_l25_25114

theorem day_crew_fraction (D W : ℝ) (h1 : D > 0) (h2 : W > 0) :
  (D * W / (D * W + (3 / 4 * D * 1 / 2 * W)) = 8 / 11) :=
by
  sorry

end day_crew_fraction_l25_25114


namespace coins_problem_l25_25287

theorem coins_problem : 
  ∃ x : ℕ, 
  (x % 8 = 6) ∧ 
  (x % 7 = 5) ∧ 
  (x % 9 = 1) ∧ 
  (x % 11 = 0) := 
by
  -- Proof to be provided here
  sorry

end coins_problem_l25_25287


namespace square_101_l25_25183

theorem square_101:
  (101 : ℕ)^2 = 10201 :=
by
  sorry

end square_101_l25_25183


namespace solve_for_x_l25_25711

theorem solve_for_x
  (n m x : ℕ)
  (h1 : 7 / 8 = n / 96)
  (h2 : 7 / 8 = (m + n) / 112)
  (h3 : 7 / 8 = (x - m) / 144) :
  x = 140 :=
by
  sorry

end solve_for_x_l25_25711


namespace trihedral_angle_sum_gt_180_l25_25142

theorem trihedral_angle_sum_gt_180
    (a' b' c' α β γ : ℝ)
    (Sabc : Prop)
    (h1 : b' = π - α)
    (h2 : c' = π - β)
    (h3 : a' = π - γ)
    (triangle_inequality : a' + b' + c' < 2 * π) :
    α + β + γ > π :=
by
  sorry

end trihedral_angle_sum_gt_180_l25_25142


namespace rectangle_area_l25_25279

/-- Define a rectangle with its length being three times its breadth, and given diagonal length d = 20.
    Prove that the area of the rectangle is 120 square meters. -/
theorem rectangle_area (b : ℝ) (l : ℝ) (d : ℝ) (h1 : l = 3 * b) (h2 : d = 20) (h3 : l^2 + b^2 = d^2) : l * b = 120 :=
by
  sorry

end rectangle_area_l25_25279


namespace pistachio_shells_percentage_l25_25761

theorem pistachio_shells_percentage (total_pistachios : ℕ) (opened_shelled_pistachios : ℕ) (P : ℝ) :
  total_pistachios = 80 →
  opened_shelled_pistachios = 57 →
  (0.75 : ℝ) * (P / 100) * (total_pistachios : ℝ) = (opened_shelled_pistachios : ℝ) →
  P = 95 :=
by
  intros h_total h_opened h_equation
  sorry

end pistachio_shells_percentage_l25_25761


namespace find_width_l25_25680

namespace RectangleProblem

variables {w l : ℝ}

-- Conditions
def length_is_three_times_width (w l : ℝ) : Prop := l = 3 * w
def sum_of_length_and_width_equals_three_times_area (w l : ℝ) : Prop := l + w = 3 * (l * w)

-- Theorem statement
theorem find_width (w l : ℝ) (h1 : length_is_three_times_width w l) (h2 : sum_of_length_and_width_equals_three_times_area w l) :
  w = 4 / 9 :=
sorry

end RectangleProblem

end find_width_l25_25680


namespace canoe_trip_shorter_l25_25478

def lake_diameter : ℝ := 2
def pi_value : ℝ := 3.14

theorem canoe_trip_shorter : (2 * pi_value * (lake_diameter / 2) - lake_diameter) = 4.28 :=
by
  sorry

end canoe_trip_shorter_l25_25478


namespace lizette_has_813_stamps_l25_25379

def minervas_stamps : ℕ := 688
def additional_stamps : ℕ := 125
def lizettes_stamps : ℕ := minervas_stamps + additional_stamps

theorem lizette_has_813_stamps : lizettes_stamps = 813 := by
  sorry

end lizette_has_813_stamps_l25_25379


namespace solve_cubic_root_eq_l25_25942

theorem solve_cubic_root_eq (x : ℝ) : (∃ x, 3 - x / 3 = -8) -> x = 33 :=
by
  sorry

end solve_cubic_root_eq_l25_25942


namespace average_weight_20_boys_l25_25816

theorem average_weight_20_boys 
  (A : Real)
  (numBoys₁ numBoys₂ : ℕ)
  (weight₂ : Real)
  (avg_weight_class : Real)
  (h_numBoys₁ : numBoys₁ = 20)
  (h_numBoys₂ : numBoys₂ = 8)
  (h_weight₂ : weight₂ = 45.15)
  (h_avg_weight_class : avg_weight_class = 48.792857142857144)
  (h_total_boys : numBoys₁ + numBoys₂ = 28)
  (h_eq_weight : numBoys₁ * A + numBoys₂ * weight₂ = 28 * avg_weight_class) :
  A = 50.25 :=
  sorry

end average_weight_20_boys_l25_25816


namespace intersection_P_Q_l25_25177

def P := {x : ℤ | x^2 - 16 < 0}
def Q := {x : ℤ | ∃ n : ℤ, x = 2 * n}

theorem intersection_P_Q :
  P ∩ Q = {-2, 0, 2} :=
sorry

end intersection_P_Q_l25_25177


namespace david_first_six_l25_25304

def prob_six := (1:ℚ) / 6
def prob_not_six := (5:ℚ) / 6

def prob_david_first_six_cycle : ℚ :=
  prob_not_six * prob_not_six * prob_not_six * prob_six

def prob_no_six_cycle : ℚ :=
  prob_not_six ^ 4

def infinite_series_sum (a r: ℚ) : ℚ := 
  a / (1 - r)

theorem david_first_six :
  infinite_series_sum prob_david_first_six_cycle prob_no_six_cycle = 125 / 671 :=
by
  sorry

end david_first_six_l25_25304


namespace number_of_boys_in_class_l25_25689

theorem number_of_boys_in_class (n : ℕ)
  (avg_height : ℕ) (incorrect_height : ℕ) (actual_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : avg_height = 180)
  (h2 : incorrect_height = 156)
  (h3 : actual_height = 106)
  (h4 : actual_avg_height = 178)
  : n = 25 :=
by 
  -- We have the following conditions:
  -- Incorrect total height = avg_height * n
  -- Difference due to incorrect height = incorrect_height - actual_height
  -- Correct total height = avg_height * n - (incorrect_height - actual_height)
  -- Total height according to actual average = actual_avg_height * n
  -- Equating both, we have:
  -- avg_height * n - (incorrect_height - actual_height) = actual_avg_height * n
  -- We know avg_height, incorrect_height, actual_height, actual_avg_height from h1, h2, h3, h4
  -- Substituting these values and solving:
  -- 180n - (156 - 106) = 178n
  -- 180n - 50 = 178n
  -- 2n = 50
  -- n = 25
  sorry

end number_of_boys_in_class_l25_25689


namespace middle_integer_of_consecutive_odd_l25_25416

theorem middle_integer_of_consecutive_odd (n : ℕ)
  (h1 : n > 2)
  (h2 : n < 8)
  (h3 : (n-2) % 2 = 1)
  (h4 : n % 2 = 1)
  (h5 : (n+2) % 2 = 1)
  (h6 : (n-2) + n + (n+2) = (n-2) * n * (n+2) / 9) :
  n = 5 :=
by sorry

end middle_integer_of_consecutive_odd_l25_25416


namespace intersection_A_B_union_B_Ac_range_a_l25_25457

open Set

-- Conditions
def U : Set ℝ := univ
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def Ac : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def Bc : Set ℝ := {x | x < -2 ∨ x > 5}

-- Questions rewritten as Lean statements

theorem intersection_A_B :
  A ∩ B = {x | 2 < x ∧ x ≤ 5} := sorry

theorem union_B_Ac :
  B ∪ Ac = {x | x ≤ 5 ∨ x ≥ 9} := sorry

theorem range_a (a : ℝ) :
  {x | a ≤ x ∧ x ≤ a + 2} ⊆ Bc → a ∈ Iio (-4) ∪ Ioi 5 := sorry

end intersection_A_B_union_B_Ac_range_a_l25_25457


namespace value_of_x_squared_plus_9y_squared_l25_25988

theorem value_of_x_squared_plus_9y_squared (x y : ℝ) (h1 : x - 3 * y = 3) (h2 : x * y = -9) : x^2 + 9 * y^2 = -45 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l25_25988


namespace compute_expression_l25_25204

theorem compute_expression : 45 * 28 + 72 * 45 = 4500 :=
by
  sorry

end compute_expression_l25_25204


namespace books_left_over_l25_25359

theorem books_left_over 
  (n_boxes : ℕ) (books_per_box : ℕ) (books_per_new_box : ℕ)
  (total_books : ℕ) (full_boxes : ℕ) (books_left : ℕ) : 
  n_boxes = 1421 → 
  books_per_box = 27 → 
  books_per_new_box = 35 →
  total_books = n_boxes * books_per_box →
  full_boxes = total_books / books_per_new_box →
  books_left = total_books % books_per_new_box →
  books_left = 7 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end books_left_over_l25_25359


namespace angle_in_triangle_l25_25930

theorem angle_in_triangle
  (A B C : Type)
  (a b c : ℝ)
  (angle_ABC : ℝ)
  (h1 : a = 15)
  (h2 : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3)
  : angle_ABC = π/3 ∨ angle_ABC = 2 * π / 3 := 
  sorry

end angle_in_triangle_l25_25930


namespace stock_comparison_l25_25440

-- Quantities of the first year depreciation or growth rates
def initial_investment : ℝ := 200.0
def dd_first_year_growth : ℝ := 1.10
def ee_first_year_decline : ℝ := 0.85
def ff_first_year_growth : ℝ := 1.05

-- Quantities of the second year depreciation or growth rates
def dd_second_year_growth : ℝ := 1.05
def ee_second_year_growth : ℝ := 1.15
def ff_second_year_decline : ℝ := 0.90

-- Mathematical expression to determine final values after first year
def dd_after_first_year := initial_investment * dd_first_year_growth
def ee_after_first_year := initial_investment * ee_first_year_decline
def ff_after_first_year := initial_investment * ff_first_year_growth

-- Mathematical expression to determine final values after second year
def dd_final := dd_after_first_year * dd_second_year_growth
def ee_final := ee_after_first_year * ee_second_year_growth
def ff_final := ff_after_first_year * ff_second_year_decline

-- Theorem representing the final comparison
theorem stock_comparison : ff_final < ee_final ∧ ee_final < dd_final :=
by {
  -- Here we would provide the proof, but as per instruction we'll place sorry
  sorry
}

end stock_comparison_l25_25440


namespace age_ratio_in_4_years_l25_25797

variable {p k x : ℕ}

theorem age_ratio_in_4_years (h₁ : p - 8 = 2 * (k - 8)) (h₂ : p - 14 = 3 * (k - 14)) : x = 4 :=
by
  sorry

end age_ratio_in_4_years_l25_25797


namespace min_f_eq_2_m_n_inequality_l25_25400

def f (x : ℝ) := abs (x + 1) + abs (x - 1)

theorem min_f_eq_2 : (∀ x, f x ≥ 2) ∧ (∃ x, f x = 2) :=
by
  sorry

theorem m_n_inequality (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m^3 + n^3 = 2) : m + n ≤ 2 :=
by
  sorry

end min_f_eq_2_m_n_inequality_l25_25400


namespace like_terms_value_l25_25333

theorem like_terms_value (a b : ℤ) (h1 : a + b = 2) (h2 : a - 1 = 1) : a - b = 2 :=
sorry

end like_terms_value_l25_25333


namespace equivalence_of_statements_l25_25562

theorem equivalence_of_statements (S X Y : Prop) : 
  (S → (¬ X ∧ ¬ Y)) ↔ ((X ∨ Y) → ¬ S) :=
by sorry

end equivalence_of_statements_l25_25562


namespace product_of_three_numbers_l25_25258

-- Define the problem conditions as variables and assumptions
variables (a b c : ℚ)
axiom h1 : a + b + c = 30
axiom h2 : a = 3 * (b + c)
axiom h3 : b = 6 * c

-- State the theorem to be proven
theorem product_of_three_numbers : a * b * c = 10125 / 14 :=
by
  sorry

end product_of_three_numbers_l25_25258


namespace marta_total_spent_l25_25557

theorem marta_total_spent :
  let sale_book_cost := 5 * 10
  let online_book_cost := 40
  let bookstore_book_cost := 3 * online_book_cost
  let total_spent := sale_book_cost + online_book_cost + bookstore_book_cost
  total_spent = 210 := sorry

end marta_total_spent_l25_25557


namespace smallest_divisor_l25_25660

theorem smallest_divisor (N D : ℕ) (hN : N = D * 7) (hD : D > 0) (hsq : (N / D) = 7) :
  D = 7 :=
by 
  sorry

end smallest_divisor_l25_25660


namespace intersection_of_sets_l25_25002

theorem intersection_of_sets :
  let M := { x : ℝ | 0 ≤ x ∧ x < 16 }
  let N := { x : ℝ | x ≥ 1/3 }
  M ∩ N = { x : ℝ | 1/3 ≤ x ∧ x < 16 } :=
by
  sorry

end intersection_of_sets_l25_25002


namespace compound_interest_l25_25121

theorem compound_interest (SI : ℝ) (P : ℝ) (R : ℝ) (T : ℝ) (CI : ℝ) :
  SI = 50 →
  R = 5 →
  T = 2 →
  P = (SI * 100) / (R * T) →
  CI = P * (1 + R / 100)^T - P →
  CI = 51.25 :=
by
  intros
  exact sorry -- This placeholder represents the proof that would need to be filled in 

end compound_interest_l25_25121


namespace find_number_l25_25390

theorem find_number : ∃ x : ℝ, 0 < x ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end find_number_l25_25390


namespace sum_f_1_to_2017_l25_25120

noncomputable def f (x : ℝ) : ℝ :=
  if x % 6 < -1 then -(x % 6 + 2) ^ 2 else x % 6

theorem sum_f_1_to_2017 : (List.sum (List.map f (List.range' 1 2017))) = 337 :=
  sorry

end sum_f_1_to_2017_l25_25120


namespace largest_divisor_of_n4_minus_n2_l25_25949

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_n4_minus_n2_l25_25949


namespace relationship_of_ys_l25_25268

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end relationship_of_ys_l25_25268


namespace at_least_one_distinct_root_l25_25959

theorem at_least_one_distinct_root {a b : ℝ} (ha : a > 4) (hb : b > 4) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a * x₁ + b = 0 ∧ x₂^2 + a * x₂ + b = 0) ∨
    (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 + b * y₁ + a = 0 ∧ y₂^2 + b * y₂ + a = 0) :=
sorry

end at_least_one_distinct_root_l25_25959


namespace baker_cakes_l25_25209

theorem baker_cakes (initial_cakes sold_cakes remaining_cakes final_cakes new_cakes : ℕ)
  (h1 : initial_cakes = 110)
  (h2 : sold_cakes = 75)
  (h3 : final_cakes = 111)
  (h4 : new_cakes = final_cakes - (initial_cakes - sold_cakes)) :
  new_cakes = 76 :=
by {
  sorry
}

end baker_cakes_l25_25209


namespace real_roots_of_system_l25_25075

theorem real_roots_of_system :
  { (x, y) : ℝ × ℝ | (x + y)^4 = 6 * x^2 * y^2 - 215 ∧ x * y * (x^2 + y^2) = -78 } =
  { (3, -2), (-2, 3), (-3, 2), (2, -3) } :=
by 
  sorry

end real_roots_of_system_l25_25075


namespace range_of_x_l25_25846

noncomputable def g (x : ℝ) : ℝ := 2^x + 2^(-x) + |x|

theorem range_of_x (x : ℝ) : g (2 * x - 1) < g 3 → -1 < x ∧ x < 2 := by
  sorry

end range_of_x_l25_25846


namespace enclosed_area_of_curve_l25_25769

/-
  The closed curve in the figure is made up of 9 congruent circular arcs each of length \(\frac{\pi}{2}\),
  where each of the centers of the corresponding circles is among the vertices of a regular hexagon of side 3.
  We want to prove that the area enclosed by the curve is \(\frac{27\sqrt{3}}{2} + \frac{9\pi}{8}\).
-/

theorem enclosed_area_of_curve :
  let side_length := 3
  let arc_length := π / 2
  let num_arcs := 9
  let hexagon_area := (3 * Real.sqrt 3 / 2) * side_length^2
  let radius := 1 / 2
  let sector_area := (π * radius^2) / 4
  let total_sector_area := num_arcs * sector_area
  let enclosed_area := hexagon_area + total_sector_area
  enclosed_area = (27 * Real.sqrt 3) / 2 + (9 * π) / 8 :=
by
  sorry

end enclosed_area_of_curve_l25_25769


namespace expand_and_simplify_l25_25156

theorem expand_and_simplify (y : ℚ) (h : y ≠ 0) :
  (3/4 * (8/y - 6*y^2 + 3*y)) = (6/y - 9*y^2/2 + 9*y/4) :=
by
  sorry

end expand_and_simplify_l25_25156


namespace evaluate_difference_of_squares_l25_25847
-- Import necessary libraries

-- Define the specific values for a and b
def a : ℕ := 72
def b : ℕ := 48

-- State the theorem to be proved
theorem evaluate_difference_of_squares : a^2 - b^2 = (a + b) * (a - b) ∧ (a + b) * (a - b) = 2880 := 
by
  -- The proof would go here but should be omitted as per directions
  sorry

end evaluate_difference_of_squares_l25_25847


namespace train_speed_approx_900072_kmph_l25_25510

noncomputable def speed_of_train (train_length platform_length time_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / time_seconds
  speed_m_s * 3.6

theorem train_speed_approx_900072_kmph :
  abs (speed_of_train 225 400.05 25 - 90.0072) < 0.001 :=
by
  sorry

end train_speed_approx_900072_kmph_l25_25510


namespace remainder_98_pow_50_mod_100_l25_25835

theorem remainder_98_pow_50_mod_100 :
  (98 : ℤ) ^ 50 % 100 = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l25_25835


namespace number_of_selected_in_interval_l25_25544

-- Definitions and conditions based on the problem statement
def total_employees : ℕ := 840
def sample_size : ℕ := 42
def systematic_sampling_interval : ℕ := total_employees / sample_size
def interval_start : ℕ := 481
def interval_end : ℕ := 720

-- Main theorem statement that we need to prove
theorem number_of_selected_in_interval :
  let selected_in_interval : ℕ := (interval_end - interval_start + 1) / systematic_sampling_interval
  selected_in_interval = 12 := by
  sorry

end number_of_selected_in_interval_l25_25544


namespace solve_linear_system_l25_25785

theorem solve_linear_system :
  ∃ x y : ℚ, 7 * x = -10 - 3 * y ∧ 4 * x = 5 * y - 32 ∧ 
  x = -219 / 88 ∧ y = 97 / 22 :=
by
  sorry

end solve_linear_system_l25_25785


namespace common_ratio_of_arithmetic_sequence_l25_25460

variable {α : Type} [LinearOrderedField α]

def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_ratio_of_arithmetic_sequence (a : ℕ → α) (q : α)
  (h1 : is_arithmetic_sequence a)
  (h2 : ∀ n : ℕ, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
  (h3 : a 1 > 0)
  (h4 : ∀ n : ℕ, a n < a (n + 1)) :
  q = 2 := 
sorry

end common_ratio_of_arithmetic_sequence_l25_25460


namespace energy_savings_l25_25858

theorem energy_savings (x y : ℝ) 
  (h1 : x = y + 27) 
  (h2 : x + 2.1 * y = 405) :
  x = 149 ∧ y = 122 :=
by
  sorry

end energy_savings_l25_25858


namespace explicit_form_l25_25772

-- Define the functional equation
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x) satisfies
axiom functional_equation (x : ℝ) (h : x ≠ 0) : f x = 2 * f (1 / x) + 3 * x

-- State the theorem that we need to prove
theorem explicit_form (x : ℝ) (h : x ≠ 0) : f x = -x - (2 / x) :=
by
  sorry

end explicit_form_l25_25772


namespace earnings_correct_l25_25917

-- Define the initial number of roses, the number of roses left, and the price per rose.
def initial_roses : ℕ := 13
def roses_left : ℕ := 4
def price_per_rose : ℕ := 4

-- Calculate the number of roses sold.
def roses_sold : ℕ := initial_roses - roses_left

-- Calculate the total earnings.
def earnings : ℕ := roses_sold * price_per_rose

-- Prove that the earnings are 36 dollars.
theorem earnings_correct : earnings = 36 := by
  sorry

end earnings_correct_l25_25917


namespace brady_earns_181_l25_25799

def bradyEarnings (basic_count : ℕ) (gourmet_count : ℕ) (total_cards : ℕ) : ℕ :=
  let basic_earnings := basic_count * 70
  let gourmet_earnings := gourmet_count * 90
  let total_earnings := basic_earnings + gourmet_earnings
  let total_bonus := (total_cards / 100) * 10 + ((total_cards / 100) - 1) * 5
  total_earnings + total_bonus

theorem brady_earns_181 :
  bradyEarnings 120 80 200 = 181 :=
by 
  sorry

end brady_earns_181_l25_25799


namespace locus_points_eq_distance_l25_25137

def locus_is_parabola (x y : ℝ) : Prop :=
  (y - 1) ^ 2 = 16 * (x - 2)

theorem locus_points_eq_distance (x y : ℝ) :
  locus_is_parabola x y ↔ (x, y) = (4, 1) ∨
    dist (x, y) (4, 1) = dist (x, y) (0, y) :=
by
  sorry

end locus_points_eq_distance_l25_25137


namespace initial_number_of_men_l25_25325

def initial_average_age_increased_by_2_years_when_two_women_replace_two_men 
    (M : ℕ) (A men1 men2 women1 women2 : ℕ) : Prop :=
  (men1 = 20) ∧ (men2 = 24) ∧ (women1 = 30) ∧ (women2 = 30) ∧
  ((M * A) + 16 = (M * (A + 2)))

theorem initial_number_of_men (M : ℕ) (A : ℕ) (men1 men2 women1 women2: ℕ):
  initial_average_age_increased_by_2_years_when_two_women_replace_two_men M A men1 men2 women1 women2 → 
  2 * M = 16 → M = 8 :=
by
  sorry

end initial_number_of_men_l25_25325


namespace n_divisible_by_100_l25_25449

theorem n_divisible_by_100 (n : ℤ) (h1 : n > 101) (h2 : 101 ∣ n)
  (h3 : ∀ d : ℤ, 1 < d ∧ d < n → d ∣ n → ∃ k m : ℤ, k ∣ n ∧ m ∣ n ∧ d = k - m) : 100 ∣ n :=
sorry

end n_divisible_by_100_l25_25449


namespace work_days_of_A_and_B_l25_25205

theorem work_days_of_A_and_B (B : ℝ) (A : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 27) :
  1 / (A + B) = 9 :=
by
  sorry

end work_days_of_A_and_B_l25_25205


namespace circle_equation_exists_shortest_chord_line_l25_25265

-- Condition 1: Points A and B
def point_A : (ℝ × ℝ) := (1, -2)
def point_B : (ℝ × ℝ) := (-1, 0)

-- Condition 2: Circle passes through A and B and sum of intercepts is 2
def passes_through (x y : ℝ) (D E F : ℝ) : Prop := 
  (x^2 + y^2 + D * x + E * y + F = 0)

def satisfies_intercepts (D E : ℝ) : Prop := (-D - E = 2)

-- Prove
theorem circle_equation_exists : 
  ∃ D E F, passes_through 1 (-2) D E F ∧ passes_through (-1) 0 D E F ∧ satisfies_intercepts D E :=
sorry

-- Given that P(2, 0.5) is inside the circle from above theorem
def point_P : (ℝ × ℝ) := (2, 0.5)

-- Prove the equation of the shortest chord line l
theorem shortest_chord_line :
  ∃ m b, m = -2 ∧ point_P.2 = m * (point_P.1 - 2) + b ∧ (∀ (x y : ℝ), 4 * x + 2 * y - 9 = 0) :=
sorry

end circle_equation_exists_shortest_chord_line_l25_25265


namespace sum_of_six_terms_l25_25017

variable (a₁ a₂ a₃ a₄ a₅ a₆ q : ℝ)

-- Conditions
def geom_seq := a₂ = q * a₁ ∧ a₃ = q * a₂ ∧ a₄ = q * a₃ ∧ a₅ = q * a₄ ∧ a₆ = q * a₅
def cond₁ : Prop := a₁ + a₃ = 5 / 2
def cond₂ : Prop := a₂ + a₄ = 5 / 4

-- Problem Statement
theorem sum_of_six_terms : geom_seq a₁ a₂ a₃ a₄ a₅ a₆ q → cond₁ a₁ a₃ → cond₂ a₂ a₄ → 
  (a₁ * (1 - q^6) / (1 - q) = 63 / 16) := 
by 
  sorry

end sum_of_six_terms_l25_25017


namespace sum_of_products_mod_7_l25_25235

-- Define the numbers involved
def a := 1789
def b := 1861
def c := 1945
def d := 1533
def e := 1607
def f := 1688

-- Define the sum of products
def sum_of_products := a * b * c + d * e * f

-- The statement to prove:
theorem sum_of_products_mod_7 : sum_of_products % 7 = 3 := 
by sorry

end sum_of_products_mod_7_l25_25235


namespace find_n_for_geom_sum_l25_25783

-- Define the first term and the common ratio
def first_term := 1
def common_ratio := 1 / 2

-- Define the sum function of the first n terms of the geometric sequence
def geom_sum (n : ℕ) : ℚ := first_term * (1 - (common_ratio)^n) / (1 - common_ratio)

-- Define the target sum
def target_sum := 31 / 16

-- State the theorem to prove
theorem find_n_for_geom_sum : ∃ n : ℕ, geom_sum n = target_sum := 
    by
    sorry

end find_n_for_geom_sum_l25_25783


namespace find_length_PQ_l25_25918

noncomputable def length_of_PQ (PQ PR : ℝ) (ST SU : ℝ) (angle_PQPR angle_STSU : ℝ) : ℝ :=
if (angle_PQPR = 120 ∧ angle_STSU = 120 ∧ PR / SU = 8 / 9) then 
  2 
else 
  0

theorem find_length_PQ :
  let PQ := 4 
  let PR := 8
  let ST := 9
  let SU := 18
  let PQ_crop := 2
  let angle_PQPR := 120
  let angle_STSU := 120
  length_of_PQ PQ PR ST SU angle_PQPR angle_STSU = PQ_crop :=
by
  sorry

end find_length_PQ_l25_25918


namespace last_two_digits_of_squared_expression_l25_25301

theorem last_two_digits_of_squared_expression (n : ℕ) :
  (n * 2 * 3 * 4 * 46 * 47 * 48 * 49) ^ 2 % 100 = 76 :=
by
  sorry

end last_two_digits_of_squared_expression_l25_25301


namespace product_variation_l25_25259

theorem product_variation (a b c : ℕ) (h1 : a * b = c) (h2 : b' = 10 * b) (h3 : ∃ d : ℕ, d = a * b') : d = 720 :=
by
  sorry

end product_variation_l25_25259


namespace exponential_function_passes_through_fixed_point_l25_25507

theorem exponential_function_passes_through_fixed_point {a : ℝ} (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : 
  (a^(2 - 2) + 3) = 4 :=
by
  sorry

end exponential_function_passes_through_fixed_point_l25_25507


namespace trajectory_moving_circle_l25_25548

theorem trajectory_moving_circle : 
  (∃ P : ℝ × ℝ, (∃ r : ℝ, (P.1 + 1)^2 = r^2 ∧ (P.1 - 2)^2 + P.2^2 = (r + 1)^2) ∧
  P.2^2 = 8 * P.1) :=
sorry

end trajectory_moving_circle_l25_25548


namespace order_of_fractions_l25_25318

theorem order_of_fractions (a b c d : ℚ)
  (h₁ : a = 21/14)
  (h₂ : b = 25/18)
  (h₃ : c = 23/16)
  (h₄ : d = 27/19)
  (h₅ : a > b)
  (h₆ : a > c)
  (h₇ : a > d)
  (h₈ : b < c)
  (h₉ : b < d)
  (h₁₀ : c > d) :
  b < d ∧ d < c ∧ c < a := 
sorry

end order_of_fractions_l25_25318


namespace domain_of_inverse_l25_25321

noncomputable def f (x : ℝ) : ℝ := 3 ^ x

theorem domain_of_inverse (x : ℝ) : f x > 0 :=
by
  sorry

end domain_of_inverse_l25_25321


namespace f_2015_l25_25819

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f (x + 2) = f (2 - x) + 4 * f 2
axiom symmetric_about_neg1 : ∀ x : ℝ, f (x + 1) = f (-2 - (x + 1))
axiom f_at_1 : f 1 = 3

theorem f_2015 : f 2015 = -3 :=
by
  apply sorry

end f_2015_l25_25819


namespace chips_count_l25_25281

theorem chips_count (B G P R x : ℕ) 
  (hx1 : 5 < x) (hx2 : x < 11) 
  (h : 1^B * 5^G * x^P * 11^R = 28160) : 
  P = 2 :=
by 
  -- Hint: Prime factorize 28160 to apply constraints and identify corresponding exponents.
  have prime_factorization_28160 : 28160 = 2^6 * 5^1 * 7^2 := by sorry
  -- Given 5 < x < 11 and by prime factorization, x can only be 7 (since it factors into the count of 7)
  -- Complete the rest of the proof
  sorry

end chips_count_l25_25281


namespace number_of_pencils_broken_l25_25195

theorem number_of_pencils_broken
  (initial_pencils : ℕ)
  (misplaced_pencils : ℕ)
  (found_pencils : ℕ)
  (bought_pencils : ℕ)
  (final_pencils : ℕ)
  (h_initial : initial_pencils = 20)
  (h_misplaced : misplaced_pencils = 7)
  (h_found : found_pencils = 4)
  (h_bought : bought_pencils = 2)
  (h_final : final_pencils = 16) :
  (initial_pencils - misplaced_pencils + found_pencils + bought_pencils - final_pencils) = 3 := 
by
  sorry

end number_of_pencils_broken_l25_25195


namespace min_colors_correct_l25_25763

def min_colors (n : Nat) : Nat :=
  if n = 1 then 1
  else if n = 2 then 2
  else 3

theorem min_colors_correct (n : Nat) : min_colors n = 
  if n = 1 then 1
  else if n = 2 then 2
  else 3 := by
  sorry

end min_colors_correct_l25_25763


namespace original_cost_of_plants_l25_25020

theorem original_cost_of_plants
  (discount : ℕ)
  (amount_spent : ℕ)
  (original_cost : ℕ)
  (h_discount : discount = 399)
  (h_amount_spent : amount_spent = 68)
  (h_original_cost : original_cost = discount + amount_spent) :
  original_cost = 467 :=
by
  rw [h_discount, h_amount_spent] at h_original_cost
  exact h_original_cost

end original_cost_of_plants_l25_25020


namespace smallest_y_exists_l25_25393

theorem smallest_y_exists (M : ℤ) (y : ℕ) (h : 2520 * y = M ^ 3) : y = 3675 :=
by
  have h_factorization : 2520 = 2^3 * 3^2 * 5 * 7 := sorry
  sorry

end smallest_y_exists_l25_25393


namespace imaginary_part_z_is_correct_l25_25879

open Complex

noncomputable def problem_conditions (z : ℂ) : Prop :=
  (3 - 4 * Complex.I) * z = Complex.abs (4 + 3 * Complex.I)

theorem imaginary_part_z_is_correct (z : ℂ) (hz : problem_conditions z) :
  z.im = 4 / 5 :=
sorry

end imaginary_part_z_is_correct_l25_25879


namespace evaluate_expression_l25_25398

theorem evaluate_expression (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) : 4 * a^4 - 12 * a^3 + 9 * a^2 - 10 = 15 :=
by
  sorry

end evaluate_expression_l25_25398


namespace sum_of_reciprocals_l25_25030

open Real

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x + y = 5 * x * y) (hx2y : x = 2 * y) : 
  (1 / x) + (1 / y) = 5 := 
  sorry

end sum_of_reciprocals_l25_25030


namespace common_difference_is_4_l25_25182

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- Defining the arithmetic sequence {a_n}
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
variable (d : ℤ) (a4_a5_sum : a 4 + a 5 = 24) (S6_val : S 6 = 48)

-- Statement to prove: given the conditions, d = 4
theorem common_difference_is_4 (h_seq : is_arithmetic_sequence a d) :
  d = 4 := sorry

end common_difference_is_4_l25_25182


namespace range_of_a_l25_25199

def p (a : ℝ) : Prop := 0 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 1 / 4

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : a ∈ Set.Ioc 0 (1 / 4) ∨ a ∈ Set.Ioi 1 :=
by
  sorry

end range_of_a_l25_25199


namespace proof_problem_l25_25290

noncomputable def problem_statement (a b c d : ℝ) : Prop :=
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
  (a + b = 2 * c) ∧ (a * b = -5 * d) ∧ (c + d = 2 * a) ∧ (c * d = -5 * b)

theorem proof_problem (a b c d : ℝ) (h : problem_statement a b c d) : a + b + c + d = 30 :=
by
  sorry

end proof_problem_l25_25290


namespace unique_solution_quadratic_l25_25419

theorem unique_solution_quadratic (n : ℕ) : (∀ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intros h
  sorry

end unique_solution_quadratic_l25_25419


namespace gcd_of_17420_23826_36654_l25_25245

theorem gcd_of_17420_23826_36654 : Nat.gcd (Nat.gcd 17420 23826) 36654 = 2 := 
by 
  sorry

end gcd_of_17420_23826_36654_l25_25245


namespace Lisa_initial_pencils_l25_25830

-- Variables
variable (G_L_initial : ℕ) (L_L_initial : ℕ) (G_L_total : ℕ)

-- Conditions
def G_L_initial_def := G_L_initial = 2
def G_L_total_def := G_L_total = 101
def Lisa_gives_pencils : Prop := G_L_total = G_L_initial + L_L_initial

-- Proof statement
theorem Lisa_initial_pencils (G_L_initial : ℕ) (G_L_total : ℕ)
  (h1 : G_L_initial = 2) (h2 : G_L_total = 101) (h3 : G_L_total = G_L_initial + L_L_initial) :
  L_L_initial = 99 := 
by 
  sorry

end Lisa_initial_pencils_l25_25830


namespace apple_cost_l25_25392

theorem apple_cost (A : ℝ) (h_discount : ∃ (n : ℕ), 15 = (5 * (5: ℝ) * A + 3 * 2 + 2 * 3 - n)) : A = 1 :=
by
  sorry

end apple_cost_l25_25392


namespace soda_ratio_l25_25232

theorem soda_ratio (v p : ℝ) (hv : v > 0) (hp : p > 0) : 
  let v_z := 1.3 * v
  let p_z := 0.85 * p
  (p_z / v_z) / (p / v) = 17 / 26 :=
by sorry

end soda_ratio_l25_25232


namespace f_three_equals_322_l25_25041

def f (z : ℝ) : ℝ := (z^2 - 2) * ((z^2 - 2)^2 - 3)

theorem f_three_equals_322 :
  f 3 = 322 :=
by
  -- Proof steps (left out intentionally as per instructions)
  sorry

end f_three_equals_322_l25_25041


namespace odd_and_periodic_40_l25_25396

noncomputable def f : ℝ → ℝ := sorry

theorem odd_and_periodic_40
  (h₁ : ∀ x : ℝ, f (10 + x) = f (10 - x))
  (h₂ : ∀ x : ℝ, f (20 - x) = -f (20 + x)) :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (x + 40) = f (x)) :=
by
  sorry

end odd_and_periodic_40_l25_25396


namespace roger_cookie_price_l25_25610

noncomputable def price_per_roger_cookie (A_cookies: ℕ) (A_price_per_cookie: ℕ) (A_area_per_cookie: ℕ) (R_cookies: ℕ) (R_area_per_cookie: ℕ): ℕ :=
  by
  let A_total_earnings := A_cookies * A_price_per_cookie
  let R_total_area := A_cookies * A_area_per_cookie
  let price_per_R_cookie := A_total_earnings / R_cookies
  exact price_per_R_cookie
  
theorem roger_cookie_price {A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie : ℕ}
  (h1 : A_cookies = 12)
  (h2 : A_price_per_cookie = 60)
  (h3 : A_area_per_cookie = 12)
  (h4 : R_cookies = 18) -- assumed based on area calculation 144 / 8 (we need this input to match solution context)
  (h5 : R_area_per_cookie = 8) :
  price_per_roger_cookie A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie = 40 :=
  by
  sorry

end roger_cookie_price_l25_25610


namespace count_not_divisible_by_2_3_5_l25_25353

theorem count_not_divisible_by_2_3_5 : 
  let count_div_2 := (100 / 2)
  let count_div_3 := (100 / 3)
  let count_div_5 := (100 / 5)
  let count_div_6 := (100 / 6)
  let count_div_10 := (100 / 10)
  let count_div_15 := (100 / 15)
  let count_div_30 := (100 / 30)
  100 - (count_div_2 + count_div_3 + count_div_5) 
      + (count_div_6 + count_div_10 + count_div_15) 
      - count_div_30 = 26 :=
by
  let count_div_2 := 50
  let count_div_3 := 33
  let count_div_5 := 20
  let count_div_6 := 16
  let count_div_10 := 10
  let count_div_15 := 6
  let count_div_30 := 3
  sorry

end count_not_divisible_by_2_3_5_l25_25353


namespace certain_amount_added_l25_25706

theorem certain_amount_added {x y : ℕ} 
    (h₁ : x = 15) 
    (h₂ : 3 * (2 * x + y) = 105) : y = 5 :=
by
  sorry

end certain_amount_added_l25_25706


namespace ratio_kittens_to_breeding_rabbits_l25_25234

def breeding_rabbits : ℕ := 10
def kittens_first_spring (k : ℕ) : ℕ := k * breeding_rabbits
def adopted_kittens_first_spring (k : ℕ) : ℕ := 5 * k
def returned_kittens : ℕ := 5
def remaining_kittens_first_spring (k : ℕ) : ℕ := (k * breeding_rabbits) / 2 + returned_kittens

def kittens_second_spring : ℕ := 60
def adopted_kittens_second_spring : ℕ := 4
def remaining_kittens_second_spring : ℕ := kittens_second_spring - adopted_kittens_second_spring

def total_rabbits (k : ℕ) : ℕ := 
  breeding_rabbits + remaining_kittens_first_spring k + remaining_kittens_second_spring

theorem ratio_kittens_to_breeding_rabbits (k : ℕ) (h : total_rabbits k = 121) :
  k = 10 :=
sorry

end ratio_kittens_to_breeding_rabbits_l25_25234


namespace merchant_discount_l25_25708

-- Definitions used in Lean 4 statement coming directly from conditions
def initial_cost_price : Real := 100
def marked_up_percentage : Real := 0.80
def profit_percentage : Real := 0.35

-- To prove the percentage discount offered
theorem merchant_discount (cp mp sp discount percentage_discount : Real) 
  (H1 : cp = initial_cost_price)
  (H2 : mp = cp + (marked_up_percentage * cp))
  (H3 : sp = cp + (profit_percentage * cp))
  (H4 : discount = mp - sp)
  (H5 : percentage_discount = (discount / mp) * 100) :
  percentage_discount = 25 := 
sorry

end merchant_discount_l25_25708


namespace find_number_l25_25170

theorem find_number (x : ℕ) (h : 3 * (2 * x + 8) = 84) : x = 10 :=
by
  sorry

end find_number_l25_25170


namespace fg_minus_gf_l25_25541

noncomputable def f (x : ℝ) : ℝ := 8 * x - 12
noncomputable def g (x : ℝ) : ℝ := x / 4 - 1

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -16 :=
by
  -- We skip the proof.
  sorry

end fg_minus_gf_l25_25541


namespace relationship_among_abcd_l25_25384

theorem relationship_among_abcd (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : d < c) 
  (h3 : (c - a) * (c - b) < 0) 
  (h4 : (d - a) * (d - b) > 0) : 
  d < a ∧ a < c ∧ c < b := 
by
  sorry

end relationship_among_abcd_l25_25384


namespace quadratic_pos_in_interval_l25_25932

theorem quadratic_pos_in_interval (m n : ℤ)
  (h2014 : (2014:ℤ)^2 + m * 2014 + n > 0)
  (h2015 : (2015:ℤ)^2 + m * 2015 + n > 0) :
  ∀ x : ℝ, 2014 ≤ x ∧ x ≤ 2015 → (x^2 + (m:ℝ) * x + (n:ℝ)) > 0 :=
by
  sorry

end quadratic_pos_in_interval_l25_25932


namespace jellybean_count_l25_25811

theorem jellybean_count (x : ℕ) (h : (0.7 : ℝ) ^ 3 * x = 34) : x = 99 :=
sorry

end jellybean_count_l25_25811


namespace intersecting_lines_triangle_area_l25_25283

theorem intersecting_lines_triangle_area :
  let line1 := { p : ℝ × ℝ | p.2 = p.1 }
  let line2 := { p : ℝ × ℝ | p.1 = -6 }
  let intersection := (-6, -6)
  let base := 6
  let height := 6
  let area := (1 / 2 : ℝ) * base * height
  area = 18 := by
  sorry

end intersecting_lines_triangle_area_l25_25283


namespace largest_integer_n_l25_25516

theorem largest_integer_n (n : ℤ) :
  (n^2 - 11 * n + 24 < 0) → n ≤ 7 :=
by
  sorry

end largest_integer_n_l25_25516


namespace find_n_pos_int_l25_25588

theorem find_n_pos_int (n : ℕ) (h1 : n ^ 3 + 2 * n ^ 2 + 9 * n + 8 = k ^ 3) : n = 7 := 
sorry

end find_n_pos_int_l25_25588


namespace quadratic_zeros_l25_25965

theorem quadratic_zeros : ∀ x : ℝ, (x = 3 ∨ x = -1) ↔ (x^2 - 2*x - 3 = 0) := by
  intro x
  sorry

end quadratic_zeros_l25_25965


namespace pair_C_product_not_36_l25_25257

-- Definitions of the pairs
def pair_A : ℤ × ℤ := (-4, -9)
def pair_B : ℤ × ℤ := (-3, -12)
def pair_C : ℚ × ℚ := (1/2, -72)
def pair_D : ℤ × ℤ := (1, 36)
def pair_E : ℚ × ℚ := (3/2, 24)

-- Mathematical statement for the proof problem
theorem pair_C_product_not_36 :
  pair_C.fst * pair_C.snd ≠ 36 :=
by
  sorry

end pair_C_product_not_36_l25_25257


namespace intersection_is_14_l25_25166

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {y | ∃ x ∈ A, y = 3 * x - 2}

theorem intersection_is_14 : A ∩ B = {1, 4} := 
by sorry

end intersection_is_14_l25_25166


namespace smallest_n_divisible_by_23_l25_25497

theorem smallest_n_divisible_by_23 :
  ∃ n : ℕ, (n^3 + 12 * n^2 + 15 * n + 180) % 23 = 0 ∧
            ∀ m : ℕ, (m^3 + 12 * m^2 + 15 * m + 180) % 23 = 0 → n ≤ m :=
sorry

end smallest_n_divisible_by_23_l25_25497


namespace beth_final_students_l25_25547

-- Define the initial conditions
def initial_students : ℕ := 150
def students_joined : ℕ := 30
def students_left : ℕ := 15

-- Define the number of students after the first additional year
def after_first_year : ℕ := initial_students + students_joined

-- Define the final number of students after students leaving
def final_students : ℕ := after_first_year - students_left

-- Theorem to prove the number of students in the final year
theorem beth_final_students : 
  final_students = 165 :=
by
  sorry

end beth_final_students_l25_25547


namespace star_3_2_l25_25815

-- Definition of the operation
def star (a b : ℤ) : ℤ := a * b^3 - b^2 + 2

-- The proof problem
theorem star_3_2 : star 3 2 = 22 :=
by
  sorry

end star_3_2_l25_25815


namespace solve_eqn_in_integers_l25_25191

theorem solve_eqn_in_integers :
  ∃ (x y : ℤ), xy + 3*x - 5*y = -3 ∧ 
  ((x, y) = (6, 9) ∨ (x, y) = (7, 3) ∨ (x, y) = (8, 1) ∨ 
  (x, y) = (9, 0) ∨ (x, y) = (11, -1) ∨ (x, y) = (17, -2) ∨ 
  (x, y) = (4, -15) ∨ (x, y) = (3, -9) ∨ (x, y) = (2, -7) ∨ 
  (x, y) = (1, -6) ∨ (x, y) = (-1, -5) ∨  (x, y) = (-7, -4)) :=
sorry

end solve_eqn_in_integers_l25_25191


namespace pq_true_l25_25611

-- Proposition p: a^2 + b^2 < 0 is false
def p_false (a b : ℝ) : Prop := ¬ (a^2 + b^2 < 0)

-- Proposition q: (a-2)^2 + |b-3| ≥ 0 is true
def q_true (a b : ℝ) : Prop := (a - 2)^2 + |b - 3| ≥ 0

-- Theorem stating that "p ∨ q" is true
theorem pq_true (a b : ℝ) (h1 : p_false a b) (h2 : q_true a b) : (a^2 + b^2 < 0 ∨ (a - 2)^2 + |b - 3| ≥ 0) :=
by {
  sorry
}

end pq_true_l25_25611


namespace probability_two_draws_l25_25076

def probability_first_red_second_kd (total_cards : ℕ) (red_cards : ℕ) (king_of_diamonds : ℕ) : ℚ :=
  (red_cards / total_cards) * (king_of_diamonds / (total_cards - 1))

theorem probability_two_draws :
  let total_cards := 52
  let red_cards := 26
  let king_of_diamonds := 1
  probability_first_red_second_kd total_cards red_cards king_of_diamonds = 1 / 102 :=
by {
  sorry
}

end probability_two_draws_l25_25076


namespace solve_linear_system_l25_25320

theorem solve_linear_system (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) :
  2 * x₁ + 2 * x₂ - x₃ + x₄ + 4 * x₆ = 0 ∧
  x₁ + 2 * x₂ + 2 * x₃ + 3 * x₅ + x₆ = -2 ∧
  x₁ - 2 * x₂ + x₄ + 2 * x₅ = 0 →
  x₁ = -1 / 4 - 5 / 8 * x₄ - 9 / 8 * x₅ - 9 / 8 * x₆ ∧
  x₂ = -1 / 8 + 3 / 16 * x₄ - 7 / 16 * x₅ + 9 / 16 * x₆ ∧
  x₃ = -3 / 4 + 1 / 8 * x₄ - 11 / 8 * x₅ + 5 / 8 * x₆ :=
by
  sorry

end solve_linear_system_l25_25320


namespace exists_large_natural_with_high_digit_sum_l25_25985

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem exists_large_natural_with_high_digit_sum :
  ∃ b : ℕ, ∀ n : ℕ, n > b → sum_of_digits (factorial n) ≥ 10 ^ 100 :=
by sorry

end exists_large_natural_with_high_digit_sum_l25_25985


namespace find_a_plus_k_l25_25583

variable (a k : ℝ)

noncomputable def f (x : ℝ) : ℝ := (a - 1) * x^k

theorem find_a_plus_k
  (h1 : f a k (Real.sqrt 2) = 2)
  (h2 : (Real.sqrt 2)^2 = 2) : a + k = 4 := 
sorry

end find_a_plus_k_l25_25583


namespace sixth_edge_length_l25_25566

theorem sixth_edge_length (a b c d o : Type) (distance : a -> a -> ℝ) (circumradius : ℝ) 
  (edge_length : ℝ) (h : ∀ (x y : a), x ≠ y → distance x y = edge_length ∨ distance x y = circumradius)
  (eq_edge_length : edge_length = 3) (eq_circumradius : circumradius = 2) : 
  ∃ ad : ℝ, ad = 6 * Real.sqrt (3 / 7) := 
by
  sorry

end sixth_edge_length_l25_25566


namespace multiply_polynomials_l25_25989

def polynomial_multiplication (x : ℝ) : Prop :=
  (x^4 + 24*x^2 + 576) * (x^2 - 24) = x^6 - 13824

theorem multiply_polynomials (x : ℝ) : polynomial_multiplication x :=
by
  sorry

end multiply_polynomials_l25_25989


namespace sufficient_not_necessary_condition_of_sin_l25_25582

open Real

theorem sufficient_not_necessary_condition_of_sin (θ : ℝ) :
  (abs (θ - π / 12) < π / 12) → (sin θ < 1 / 2) :=
sorry

end sufficient_not_necessary_condition_of_sin_l25_25582


namespace solution_set_proof_l25_25434

theorem solution_set_proof {a b : ℝ} :
  (∀ x, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) →
  (∀ x, bx^2 - a * x - 1 > 0) →
  (∀ x, -1 / 2 < x ∧ x < -1 / 3) :=
by
  sorry

end solution_set_proof_l25_25434


namespace solve_expr_l25_25901

theorem solve_expr (x : ℝ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end solve_expr_l25_25901


namespace arun_weight_average_l25_25302

theorem arun_weight_average :
  (∀ w : ℝ, 65 < w ∧ w < 72 → 60 < w ∧ w < 70 → w ≤ 68 → 66 ≤ w ∧ w ≤ 69 → 64 ≤ w ∧ w ≤ 67.5 → 
    (66.75 = (66 + 67.5) / 2)) := by
  sorry

end arun_weight_average_l25_25302


namespace determine_f_1789_l25_25261

theorem determine_f_1789
  (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, 0 < n → f (f n) = 4 * n + 9)
  (h2 : ∀ k : ℕ, f (2^k) = 2^(k+1) + 3) :
  f 1789 = 3581 :=
sorry

end determine_f_1789_l25_25261


namespace consecutive_numbers_l25_25277

theorem consecutive_numbers (x : ℕ) (h : (4 * x + 2) * (4 * x^2 + 6 * x + 6) = 3 * (4 * x^3 + 4 * x^2 + 18 * x + 8)) :
  x = 2 :=
sorry

end consecutive_numbers_l25_25277


namespace calculation_is_correct_l25_25289

-- Define the numbers involved in the calculation
def a : ℝ := 12.05
def b : ℝ := 5.4
def c : ℝ := 0.6

-- Expected result of the calculation
def expected_result : ℝ := 65.67

-- Prove that the calculation is correct
theorem calculation_is_correct : (a * b + c) = expected_result :=
by
  sorry

end calculation_is_correct_l25_25289


namespace simple_interest_rate_l25_25533

theorem simple_interest_rate (P A T : ℕ) (P_val : P = 750) (A_val : A = 900) (T_val : T = 8) : 
  ∃ (R : ℚ), R = 2.5 :=
by {
  sorry
}

end simple_interest_rate_l25_25533


namespace units_digit_of_n_l25_25462

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 11^4) (h2 : m % 10 = 9) : n % 10 = 9 := 
sorry

end units_digit_of_n_l25_25462


namespace total_chickens_l25_25563

theorem total_chickens (hens : ℕ) (roosters : ℕ) (h1 : hens = 52) (h2 : roosters = hens + 16) : hens + roosters = 120 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_chickens_l25_25563


namespace age_of_b_l25_25064

variable (a b c : ℕ)

-- Conditions
def condition1 : Prop := a = b + 2
def condition2 : Prop := b = 2 * c
def condition3 : Prop := a + b + c = 27

theorem age_of_b (h1 : condition1 a b)
                 (h2 : condition2 b c)
                 (h3 : condition3 a b c) : 
                 b = 10 := 
by sorry

end age_of_b_l25_25064


namespace real_complex_number_l25_25493

theorem real_complex_number (x : ℝ) (hx1 : x^2 - 3 * x - 3 > 0) (hx2 : x - 3 = 1) : x = 4 :=
by
  sorry

end real_complex_number_l25_25493


namespace find_x_l25_25852

theorem find_x (x : ℝ) (h : x ≠ 3) : (x^2 - 9) / (x - 3) = 3 * x → x = 3 / 2 := by
  sorry

end find_x_l25_25852


namespace old_fridge_cost_l25_25574

-- Define the daily cost of Kurt's old refrigerator
variable (x : ℝ)

-- Define the conditions given in the problem
def new_fridge_cost_per_day : ℝ := 0.45
def savings_per_month : ℝ := 12
def days_in_month : ℝ := 30

-- State the theorem to prove
theorem old_fridge_cost :
  30 * x - 30 * new_fridge_cost_per_day = savings_per_month → x = 0.85 := 
by
  intro h
  sorry

end old_fridge_cost_l25_25574


namespace spadesuit_evaluation_l25_25800

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_evaluation : spadesuit 3 (spadesuit 4 5) = -72 := by
  sorry

end spadesuit_evaluation_l25_25800


namespace inequality_proof_l25_25079

theorem inequality_proof (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z))) + (y^3 / ((1 + z) * (1 + x))) + (z^3 / ((1 + x) * (1 + y))) ≥ 3 / 4 :=
by {
  sorry
}

end inequality_proof_l25_25079


namespace smallest_N_l25_25169

theorem smallest_N (p q r s t u : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) (ht : 0 < t) (hu : 0 < u)
  (h_sum : p + q + r + s + t + u = 2023) :
  ∃ N : ℕ, N = max (max (max (max (p + q) (q + r)) (r + s)) (s + t)) (t + u) ∧ N = 810 :=
sorry

end smallest_N_l25_25169


namespace abc_product_le_two_l25_25773

theorem abc_product_le_two (a b c : ℝ) (ha : 0 ≤ a) (ha2 : a ≤ 2) (hb : 0 ≤ b) (hb2 : b ≤ 2) (hc : 0 ≤ c) (hc2 : c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 :=
sorry

end abc_product_le_two_l25_25773


namespace find_x_l25_25068

theorem find_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 :=
by
  sorry

end find_x_l25_25068


namespace angle_F_measure_l25_25578

-- Given conditions
def D := 74
def sum_of_angles (x E D : ℝ) := x + E + D = 180
def E_formula (x : ℝ) := 2 * x - 10

-- Proof problem statement in Lean 4
theorem angle_F_measure :
  ∃ x : ℝ, x = (116 / 3) ∧
    sum_of_angles x (E_formula x) D :=
sorry

end angle_F_measure_l25_25578


namespace distinct_exponentiations_are_four_l25_25576

def power (a b : ℕ) : ℕ := a^b

def expr1 := power 3 (power 3 (power 3 3))
def expr2 := power 3 (power (power 3 3) 3)
def expr3 := power (power (power 3 3) 3) 3
def expr4 := power (power 3 (power 3 3)) 3
def expr5 := power (power 3 3) (power 3 3)

theorem distinct_exponentiations_are_four : 
  (expr1 ≠ expr2 ∧ expr1 ≠ expr3 ∧ expr1 ≠ expr4 ∧ expr1 ≠ expr5 ∧
   expr2 ≠ expr3 ∧ expr2 ≠ expr4 ∧ expr2 ≠ expr5 ∧
   expr3 ≠ expr4 ∧ expr3 ≠ expr5 ∧
   expr4 ≠ expr5) :=
sorry

end distinct_exponentiations_are_four_l25_25576


namespace triangle_sides_are_6_8_10_l25_25342

theorem triangle_sides_are_6_8_10 (a b c r r1 r2 r3 : ℕ) (hr_even : Even r) (hr1_even : Even r1) 
(hr2_even : Even r2) (hr3_even : Even r3) (relationship : r * r1 * r2 + r * r2 * r3 + r * r3 * r1 + r1 * r2 * r3 = r * r1 * r2 * r3) :
  (a, b, c) = (6, 8, 10) :=
sorry

end triangle_sides_are_6_8_10_l25_25342


namespace unique_positive_real_solution_l25_25633

theorem unique_positive_real_solution :
  ∃! x : ℝ, 0 < x ∧ (x^8 + 5 * x^7 + 10 * x^6 + 2023 * x^5 - 2021 * x^4 = 0) := sorry

end unique_positive_real_solution_l25_25633


namespace unique_zero_function_l25_25693

theorem unique_zero_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end unique_zero_function_l25_25693


namespace miner_distance_when_explosion_heard_l25_25974

-- Distance function for the miner (in feet)
def miner_distance (t : ℕ) : ℕ := 30 * t

-- Distance function for the sound after the explosion (in feet)
def sound_distance (t : ℕ) : ℕ := 1100 * (t - 45)

theorem miner_distance_when_explosion_heard :
  ∃ t : ℕ, miner_distance t / 3 = 463 ∧ miner_distance t = sound_distance t :=
sorry

end miner_distance_when_explosion_heard_l25_25974


namespace pipe_B_fill_time_l25_25237

theorem pipe_B_fill_time (t : ℝ) :
  (1/10) + (2/t) - (2/15) = 1 ↔ t = 60/31 :=
by
  sorry

end pipe_B_fill_time_l25_25237


namespace inequality_proof_l25_25448

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (ab / (a + b)) + (bc / (b + c)) + (ca / (c + a)) ≤ (3 * (ab + bc + ca)) / (2 * (a + b + c)) :=
by
  sorry

end inequality_proof_l25_25448


namespace letters_written_l25_25458

theorem letters_written (nathan_rate : ℕ) (jacob_rate : ℕ) (combined_rate : ℕ) (hours : ℕ) :
  nathan_rate = 25 →
  jacob_rate = 2 * nathan_rate →
  combined_rate = nathan_rate + jacob_rate →
  hours = 10 →
  combined_rate * hours = 750 :=
by
  intros
  sorry

end letters_written_l25_25458


namespace product_of_modified_numbers_less_l25_25351

theorem product_of_modified_numbers_less
  {a b c : ℝ}
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c := 
by {
   sorry
}

end product_of_modified_numbers_less_l25_25351


namespace light_intensity_at_10_m_l25_25315

theorem light_intensity_at_10_m (k : ℝ) (d1 d2 : ℝ) (I1 I2 : ℝ)
  (h1: I1 = k / d1^2) (h2: I1 = 200) (h3: d1 = 5) (h4: d2 = 10) :
  I2 = k / d2^2 → I2 = 50 :=
sorry

end light_intensity_at_10_m_l25_25315


namespace sum_first_seven_terms_is_28_l25_25662

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence 
def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

-- Given conditions
axiom a2_a4_a6_sum : a 2 + a 4 + a 6 = 12

-- Prove that the sum of the first seven terms is 28
theorem sum_first_seven_terms_is_28 (h : is_arithmetic_seq a d) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
sorry

end sum_first_seven_terms_is_28_l25_25662


namespace total_tour_time_l25_25028

-- Declare constants for distances
def distance1 : ℝ := 55
def distance2 : ℝ := 40
def distance3 : ℝ := 70
def extra_miles : ℝ := 10

-- Declare constants for speeds
def speed1_part1 : ℝ := 60
def speed1_part2 : ℝ := 40
def speed2 : ℝ := 45
def speed3_part1 : ℝ := 45
def speed3_part2 : ℝ := 35
def speed3_part3 : ℝ := 50
def return_speed : ℝ := 55

-- Declare constants for stop times
def stop1 : ℝ := 1
def stop2 : ℝ := 1.5
def stop3 : ℝ := 2

-- Prove the total time required for the tour
theorem total_tour_time :
  (30 / speed1_part1) + (25 / speed1_part2) + stop1 +
  (distance2 / speed2) + stop2 +
  (20 / speed3_part1) + (30 / speed3_part2) + (20 / speed3_part3) + stop3 +
  ((distance1 + distance2 + distance3 + extra_miles) / return_speed) = 11.40 :=
by
  sorry

end total_tour_time_l25_25028


namespace integer_part_not_perfect_square_l25_25012

noncomputable def expr (n : ℕ) : ℝ :=
  2 * Real.sqrt (n + 1) / (Real.sqrt (n + 1) - Real.sqrt n)

theorem integer_part_not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = ⌊expr n⌋ :=
  sorry

end integer_part_not_perfect_square_l25_25012


namespace A_8_coords_l25_25349

-- Define point as a structure
structure Point where
  x : Int
  y : Int

-- Initial point A
def A : Point := {x := 3, y := 2}

-- Symmetric point about the y-axis
def sym_y (p : Point) : Point := {x := -p.x, y := p.y}

-- Symmetric point about the origin
def sym_origin (p : Point) : Point := {x := -p.x, y := -p.y}

-- Symmetric point about the x-axis
def sym_x (p : Point) : Point := {x := p.x, y := -p.y}

-- Function to get the n-th symmetric point in the sequence
def sym_point (n : Nat) : Point :=
  match n % 3 with
  | 0 => A
  | 1 => sym_y A
  | 2 => sym_origin (sym_y A)
  | _ => A  -- Fallback case (should not be reachable for n >= 0)

theorem A_8_coords : sym_point 8 = {x := 3, y := -2} := sorry

end A_8_coords_l25_25349


namespace value_of_a2019_l25_25692

noncomputable def a : ℕ → ℝ
| 0 => 3
| (n + 1) => 1 / (1 - a n)

theorem value_of_a2019 : a 2019 = 2 / 3 :=
sorry

end value_of_a2019_l25_25692


namespace problem_a_problem_b_l25_25601

-- Definition for real roots condition in problem A
def has_real_roots (k : ℝ) : Prop :=
  let a := 1
  let b := -3
  let c := k
  b^2 - 4 * a * c ≥ 0

-- Problem A: Proving the range of k
theorem problem_a (k : ℝ) : has_real_roots k ↔ k ≤ 9 / 4 :=
by
  sorry

-- Definition for a quadratic equation having a given root
def has_root (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Problem B: Proving the value of m given a common root condition
theorem problem_b (m : ℝ) : 
  (has_root 1 (-3) 2 1 ∧ has_root (m-1) 1 (m-3) 1) ↔ m = 3 / 2 :=
by
  sorry

end problem_a_problem_b_l25_25601


namespace sale_in_fifth_month_l25_25221

theorem sale_in_fifth_month (sale1 sale2 sale3 sale4 sale6 : ℕ) (avg : ℕ) (months : ℕ) (total_sales : ℕ)
    (known_sales : sale1 = 6335 ∧ sale2 = 6927 ∧ sale3 = 6855 ∧ sale4 = 7230 ∧ sale6 = 5091)
    (avg_condition : avg = 6500)
    (months_condition : months = 6)
    (total_sales_condition : total_sales = avg * months) :
    total_sales - (sale1 + sale2 + sale3 + sale4 + sale6) = 6562 :=
by
  sorry

end sale_in_fifth_month_l25_25221


namespace moles_of_MgCO3_formed_l25_25376

theorem moles_of_MgCO3_formed 
  (moles_MgO : ℕ) (moles_CO2 : ℕ)
  (h_eq : moles_MgO = 3 ∧ moles_CO2 = 3)
  (balanced_eq : ∀ n : ℕ, n * MgO + n * CO2 = n * MgCO3) : 
  moles_MgCO3 = 3 :=
by
  sorry

end moles_of_MgCO3_formed_l25_25376


namespace quadratic_trinomial_has_two_roots_l25_25311

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  4 * (a^2 - a * b + b^2 - 3 * a * c) > 0 :=
by
  sorry

end quadratic_trinomial_has_two_roots_l25_25311


namespace solve_for_q_l25_25532

theorem solve_for_q (k r q : ℕ) (h1 : 4 / 5 = k / 90) (h2 : 4 / 5 = (k + r) / 105) (h3 : 4 / 5 = (q - r) / 150) : q = 132 := 
  sorry

end solve_for_q_l25_25532


namespace ratio_of_speeds_l25_25063

-- Define the speeds V1 and V2
variable {V1 V2 : ℝ}

-- Given the initial conditions
def bike_ride_time_min := 10 -- in minutes
def subway_ride_time_min := 40 -- in minutes
def total_bike_only_time_min := 210 -- 3.5 hours in minutes

-- Prove the ratio of subway speed to bike speed is 5:1
theorem ratio_of_speeds (h : bike_ride_time_min * V1 + subway_ride_time_min * V2 = total_bike_only_time_min * V1) :
  V2 = 5 * V1 :=
by
  sorry

end ratio_of_speeds_l25_25063


namespace find_a_2013_l25_25389

def sequence_a (n : ℕ) : ℤ :=
  if n = 0 then 2
  else if n = 1 then 5
  else sequence_a (n - 1) - sequence_a (n - 2)

theorem find_a_2013 :
  sequence_a 2013 = 3 :=
sorry

end find_a_2013_l25_25389


namespace coordinates_with_respect_to_origin_l25_25363

def point_coordinates (x y : ℤ) : ℤ × ℤ :=
  (x, y)

def origin : ℤ × ℤ :=
  (0, 0)

theorem coordinates_with_respect_to_origin :
  point_coordinates 2 (-3) = (2, -3) := by
  -- placeholder proof
  sorry

end coordinates_with_respect_to_origin_l25_25363


namespace prove_system_of_inequalities_l25_25128

theorem prove_system_of_inequalities : 
  { x : ℝ | x / (x - 2) ≥ 0 ∧ 2 * x + 1 ≥ 0 } = Set.Icc (-(1:ℝ)/2) 0 ∪ Set.Ioi 2 := 
by
  sorry

end prove_system_of_inequalities_l25_25128


namespace value_of_b_l25_25804

theorem value_of_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 35 * 45 * b) : b = 105 :=
sorry

end value_of_b_l25_25804


namespace difference_two_digit_interchanged_l25_25559

theorem difference_two_digit_interchanged
  (x y : ℕ)
  (h1 : y = 2 * x)
  (h2 : (10 * x + y) - (x + y) = 8) :
  (10 * y + x) - (10 * x + y) = 9 := by
sorry

end difference_two_digit_interchanged_l25_25559


namespace unique_k_exists_l25_25057

theorem unique_k_exists (k : ℕ) (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) :
  (a^2 + b^2 = k * a * b) ↔ k = 2 := sorry

end unique_k_exists_l25_25057


namespace part1_part2_l25_25124

theorem part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) : 
  (1 / a) + (1 / (b + 1)) ≥ 4 / 5 := 
by 
  sorry

theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b + a * b = 8) : 
  a + b ≥ 4 := 
by 
  sorry

end part1_part2_l25_25124


namespace more_regular_than_diet_l25_25194

-- Define the conditions
def num_regular_soda : Nat := 67
def num_diet_soda : Nat := 9

-- State the theorem
theorem more_regular_than_diet :
  num_regular_soda - num_diet_soda = 58 :=
by
  sorry

end more_regular_than_diet_l25_25194


namespace veranda_area_l25_25059

theorem veranda_area (length_room width_room width_veranda : ℕ)
  (h_length : length_room = 20) 
  (h_width : width_room = 12) 
  (h_veranda : width_veranda = 2) : 
  (length_room + 2 * width_veranda) * (width_room + 2 * width_veranda) - (length_room * width_room) = 144 := 
by
  sorry

end veranda_area_l25_25059


namespace tail_wind_distance_l25_25868

-- Definitions based on conditions
def speed_still_air : ℝ := 262.5
def t1 : ℝ := 3
def t2 : ℝ := 4

def effective_speed_tail_wind (w : ℝ) : ℝ := speed_still_air + w
def effective_speed_against_wind (w : ℝ) : ℝ := speed_still_air - w

theorem tail_wind_distance (w : ℝ) (d : ℝ) :
  effective_speed_tail_wind w * t1 = effective_speed_against_wind w * t2 →
  d = t1 * effective_speed_tail_wind w →
  d = 900 :=
by
  sorry

end tail_wind_distance_l25_25868


namespace comb_5_1_eq_5_l25_25992

theorem comb_5_1_eq_5 : Nat.choose 5 1 = 5 :=
by
  sorry

end comb_5_1_eq_5_l25_25992


namespace terminal_side_in_quadrant_l25_25529

theorem terminal_side_in_quadrant (α : ℝ) (h : α = -5) : 
  ∃ (q : ℕ), q = 4 ∧ 270 ≤ (α + 360) % 360 ∧ (α + 360) % 360 < 360 := by 
  sorry

end terminal_side_in_quadrant_l25_25529


namespace votes_to_win_l25_25270

theorem votes_to_win (total_votes : ℕ) (geoff_votes_percent : ℝ) (additional_votes : ℕ) (x : ℝ) 
(h1 : total_votes = 6000)
(h2 : geoff_votes_percent = 0.5)
(h3 : additional_votes = 3000)
(h4 : x = 50.5) :
  ((geoff_votes_percent / 100 * total_votes) + additional_votes) / total_votes * 100 = x :=
by
  sorry

end votes_to_win_l25_25270


namespace sanda_exercise_each_day_l25_25642

def exercise_problem (javier_exercise_daily sanda_exercise_total total_minutes : ℕ) (days_in_week : ℕ) :=
  javier_exercise_daily * days_in_week + sanda_exercise_total = total_minutes

theorem sanda_exercise_each_day 
  (javier_exercise_daily : ℕ := 50)
  (days_in_week : ℕ := 7)
  (total_minutes : ℕ := 620)
  (days_sanda_exercised : ℕ := 3): 
  ∃ (sanda_exercise_each_day : ℕ), exercise_problem javier_exercise_daily (sanda_exercise_each_day * days_sanda_exercised) total_minutes days_in_week → sanda_exercise_each_day = 90 :=
by 
  sorry

end sanda_exercise_each_day_l25_25642


namespace magnitude_of_T_l25_25936

def i : Complex := Complex.I

def T : Complex := (1 + i) ^ 18 - (1 - i) ^ 18

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end magnitude_of_T_l25_25936


namespace gary_initial_money_l25_25525

/-- The initial amount of money Gary had, given that he spent $55 and has $18 left. -/
theorem gary_initial_money (amount_spent : ℤ) (amount_left : ℤ) (initial_amount : ℤ) 
  (h1 : amount_spent = 55) 
  (h2 : amount_left = 18) 
  : initial_amount = amount_spent + amount_left :=
by
  sorry

end gary_initial_money_l25_25525


namespace arithmetic_progression_common_difference_l25_25112

theorem arithmetic_progression_common_difference :
  ∀ (A1 An n d : ℕ), A1 = 3 → An = 103 → n = 21 → An = A1 + (n - 1) * d → d = 5 :=
by
  intros A1 An n d h1 h2 h3 h4
  sorry

end arithmetic_progression_common_difference_l25_25112


namespace Tommy_Ratio_Nickels_to_Dimes_l25_25248

def TommyCoinsProblem :=
  ∃ (P D N Q : ℕ), 
    (D = P + 10) ∧ 
    (Q = 4) ∧ 
    (P = 10 * Q) ∧ 
    (N = 100) ∧ 
    (N / D = 2)

theorem Tommy_Ratio_Nickels_to_Dimes : TommyCoinsProblem := by
  sorry

end Tommy_Ratio_Nickels_to_Dimes_l25_25248


namespace arithmetic_seq_general_formula_l25_25386

-- Definitions based on given conditions
def f (x : ℝ) := x^2 - 2*x + 4
def a (n : ℕ) (d : ℝ) := f (d + n - 1) 

-- The general term formula for the arithmetic sequence
theorem arithmetic_seq_general_formula (d : ℝ) :
  (a 1 d = f (d - 1)) →
  (a 3 d = f (d + 1)) →
  (∀ n : ℕ, a n d = 2*n + 1) :=
by
  intros h1 h3
  sorry

end arithmetic_seq_general_formula_l25_25386


namespace largest_expression_l25_25203

theorem largest_expression :
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  E > A ∧ E > B ∧ E > C ∧ E > D :=
by
  let A := 0.9387
  let B := 0.9381
  let C := 9385 / 10000
  let D := 0.9379
  let E := 0.9389
  sorry

end largest_expression_l25_25203


namespace trigonometric_expression_l25_25518

theorem trigonometric_expression
  (α : ℝ)
  (h1 : Real.tan α = 3) : 
  (Real.sin α + 3 * Real.cos α) / (Real.cos α - 3 * Real.sin α) = -3/4 := 
by
  sorry

end trigonometric_expression_l25_25518


namespace part_a_part_b_part_c_l25_25273

-- Part (a)
theorem part_a : (7 * (2 / 3) + 16 * (5 / 12)) = (34 / 3) :=
by
  sorry

-- Part (b)
theorem part_b : (5 - (2 / (5 / 3))) = (19 / 5) :=
by
  sorry

-- Part (c)
theorem part_c : (1 + (2 / (1 + (3 / (1 + 4))))) = (9 / 4) :=
by
  sorry

end part_a_part_b_part_c_l25_25273


namespace find_y_if_x_l25_25174

theorem find_y_if_x (x : ℝ) (hx : x^2 + 8 * (x / (x - 3))^2 = 53) :
  (∃ y, y = (x - 3)^3 * (x + 4) / (2 * x - 5) ∧ y = 17000 / 21) :=
  sorry

end find_y_if_x_l25_25174


namespace sin_ineq_l25_25790

open Real

theorem sin_ineq (n : ℕ) (h : n > 0) : sin (π / (4 * n)) ≥ (sqrt 2) / (2 * n) :=
sorry

end sin_ineq_l25_25790


namespace trigonometric_identity_l25_25853

theorem trigonometric_identity (α : Real) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10) / Real.sin (α - Real.pi / 5) = 3) :=
sorry

end trigonometric_identity_l25_25853


namespace ratio_of_perimeters_l25_25300

theorem ratio_of_perimeters (s d s' d': ℝ) (h1 : d = s * Real.sqrt 2) (h2 : d' = 2.5 * d) (h3 : d' = s' * Real.sqrt 2) : (4 * s') / (4 * s) = 5 / 2 :=
by
  -- Additional tactical details for completion, proof is omitted as per instructions
  sorry

end ratio_of_perimeters_l25_25300


namespace range_of_m_l25_25983

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - 2 * m * x + m + 3
noncomputable def g (x : ℝ) : ℝ := 2^(x - 2)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ↔ -4 < m ∧ m < 0 :=
by sorry

end range_of_m_l25_25983


namespace rectangle_constant_k_l25_25813

theorem rectangle_constant_k (d : ℝ) (x : ℝ) (h_ratio : 4 * x = (4 / 3) * (3 * x)) (h_diagonal : d^2 = (4 * x)^2 + (3 * x)^2) : 
  ∃ k : ℝ, k = 12 / 25 ∧ (4 * x) * (3 * x) = k * d^2 := 
sorry

end rectangle_constant_k_l25_25813


namespace molecular_physics_statements_l25_25364

theorem molecular_physics_statements :
  (¬A) ∧ B ∧ C ∧ D :=
by sorry

end molecular_physics_statements_l25_25364


namespace youngest_child_age_l25_25699

theorem youngest_child_age (x y z : ℕ) 
  (h1 : 3 * x + 6 = 48) 
  (h2 : 3 * y + 9 = 60) 
  (h3 : 2 * z + 4 = 30) : 
  z = 13 := 
sorry

end youngest_child_age_l25_25699


namespace complex_eq_sub_l25_25670

open Complex

theorem complex_eq_sub {a b : ℝ} (h : (a : ℂ) + 2 * I = I * ((b : ℂ) - I)) : a - b = -3 := by
  sorry

end complex_eq_sub_l25_25670


namespace find_a_b_l25_25850

noncomputable def z : ℂ := 1 + Complex.I
noncomputable def lhs (a b : ℝ) := (z^2 + a*z + b) / (z^2 - z + 1)
noncomputable def rhs : ℂ := 1 - Complex.I

theorem find_a_b (a b : ℝ) (h : lhs a b = rhs) : a = -1 ∧ b = 2 :=
  sorry

end find_a_b_l25_25850


namespace infinite_rational_solutions_x3_y3_9_l25_25095

theorem infinite_rational_solutions_x3_y3_9 :
  ∃ (S : Set (ℚ × ℚ)), S.Infinite ∧ (∀ (x y : ℚ), (x, y) ∈ S → x^3 + y^3 = 9) :=
sorry

end infinite_rational_solutions_x3_y3_9_l25_25095


namespace students_wearing_other_colors_l25_25568

-- Definitions based on conditions
def total_students := 700
def percentage_blue := 45 / 100
def percentage_red := 23 / 100
def percentage_green := 15 / 100

-- The proof problem statement
theorem students_wearing_other_colors :
  (total_students - total_students * (percentage_blue + percentage_red + percentage_green)) = 119 :=
by
  sorry

end students_wearing_other_colors_l25_25568


namespace zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l25_25945

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  x - 1/x - 2 * m * Real.log x

theorem zero_of_f_when_m_is_neg1 : ∃ x > 0, f x (-1) = 0 :=
  by
    use 1
    sorry

theorem monotonicity_of_f_m_gt_neg1 (m : ℝ) (hm : m > -1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x m ≤ f y m) ∨
  (∃ a b : ℝ, 0 < a ∧ a < b ∧
    (∀ x : ℝ, 0 < x ∧ x < a → f x m ≤ f a m) ∧
    (∀ x : ℝ, a < x ∧ x < b → f a m ≥ f x m) ∧
    (∀ x : ℝ, b < x → f b m ≤ f x m)) :=
  by
    cases lt_or_le m 1 with
    | inl hlt =>
        left
        intros x y hx hy hxy
        sorry
    | inr hle =>
        right
        use m - Real.sqrt (m^2 - 1), m + Real.sqrt (m^2 - 1)
        sorry

end zero_of_f_when_m_is_neg1_monotonicity_of_f_m_gt_neg1_l25_25945


namespace least_even_integer_square_l25_25436

theorem least_even_integer_square (E : ℕ) (h_even : E % 2 = 0) (h_square : ∃ (I : ℕ), 300 * E = I^2) : E = 6 ∧ ∃ (I : ℕ), I = 30 ∧ 300 * E = I^2 :=
sorry

end least_even_integer_square_l25_25436


namespace chelsea_sugar_problem_l25_25623

variable (initial_sugar : ℕ)
variable (num_bags : ℕ)
variable (sugar_lost_fraction : ℕ)

def remaining_sugar (initial_sugar : ℕ) (num_bags : ℕ) (sugar_lost_fraction : ℕ) : ℕ :=
  let sugar_per_bag := initial_sugar / num_bags
  let sugar_lost := sugar_per_bag / sugar_lost_fraction
  let remaining_bags_sugar := (num_bags - 1) * sugar_per_bag
  remaining_bags_sugar + (sugar_per_bag - sugar_lost)

theorem chelsea_sugar_problem : 
  remaining_sugar 24 4 2 = 21 :=
by
  sorry

end chelsea_sugar_problem_l25_25623


namespace proposition_induction_l25_25503

theorem proposition_induction {P : ℕ → Prop} (h : ∀ n, P n → P (n + 1)) (hn : ¬ P 7) : ¬ P 6 :=
by
  sorry

end proposition_induction_l25_25503


namespace max_value_of_xy_l25_25015

theorem max_value_of_xy (x y : ℝ) (h₁ : x + y = 40) (h₂ : x > 0) (h₃ : y > 0) : xy ≤ 400 :=
sorry

end max_value_of_xy_l25_25015


namespace simplify_expression_l25_25584

variable (x : ℝ)

theorem simplify_expression :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x) ^ 3) = (25 / 8) * x^2 :=
by
  sorry

end simplify_expression_l25_25584


namespace vines_painted_l25_25479

-- Definitions based on the conditions in the problem statement
def time_per_lily : ℕ := 5
def time_per_rose : ℕ := 7
def time_per_orchid : ℕ := 3
def time_per_vine : ℕ := 2
def total_time_spent : ℕ := 213
def lilies_painted : ℕ := 17
def roses_painted : ℕ := 10
def orchids_painted : ℕ := 6

-- The theorem to prove the number of vines painted
theorem vines_painted (vines_painted : ℕ) : 
  213 = (17 * 5) + (10 * 7) + (6 * 3) + (vines_painted * 2) → 
  vines_painted = 20 :=
by
  intros h
  sorry

end vines_painted_l25_25479


namespace fixed_point_on_line_find_m_values_l25_25880

-- Define the conditions and set up the statements to prove

/-- 
Condition 1: Line equation 
-/
def line_eq (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

/-- 
Condition 2: Circle equation 
-/
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 25

/-- 
Question (1): Fixed point (3,1) is always on the line
-/
theorem fixed_point_on_line (m : ℝ) : line_eq m 3 1 := by
  sorry

/-- 
Question (2): Finding the values of m for the given chord length
-/
theorem find_m_values (m : ℝ) (h_chord : ∀x y : ℝ, circle_eq x y → line_eq m x y → (x - y)^2 = 6) : 
  m = -1/2 ∨ m = 1/2 := by
  sorry

end fixed_point_on_line_find_m_values_l25_25880


namespace find_point_A_l25_25756

-- Definitions of the conditions
def point_A_left_translated_to_B (A B : ℝ × ℝ) : Prop :=
  ∃ l : ℝ, A.1 - l = B.1 ∧ A.2 = B.2

def point_A_upward_translated_to_C (A C : ℝ × ℝ) : Prop :=
  ∃ u : ℝ, A.1 = C.1 ∧ A.2 + u = C.2

-- Given points B and C
def B : ℝ × ℝ := (1, 2)
def C : ℝ × ℝ := (3, 4)

-- The statement to prove the coordinates of point A
theorem find_point_A (A : ℝ × ℝ) : 
  point_A_left_translated_to_B A B ∧ point_A_upward_translated_to_C A C → A = (3, 2) :=
by 
  sorry

end find_point_A_l25_25756


namespace number_of_rows_seating_10_is_zero_l25_25904

theorem number_of_rows_seating_10_is_zero :
  ∀ (y : ℕ) (total_people : ℕ) (total_rows : ℕ),
    (∀ (r : ℕ), r * 9 + (total_rows - r) * 10 = total_people) →
    total_people = 54 →
    total_rows = 6 →
    y = 0 :=
by
  sorry

end number_of_rows_seating_10_is_zero_l25_25904


namespace number_of_divisors_of_n_l25_25776

theorem number_of_divisors_of_n :
  let n : ℕ := (7^3) * (11^2) * (13^4)
  ∃ d : ℕ, d = 60 ∧ ∀ m : ℕ, m ∣ n ↔ ∃ l₁ l₂ l₃ : ℕ, l₁ ≤ 3 ∧ l₂ ≤ 2 ∧ l₃ ≤ 4 ∧ m = 7^l₁ * 11^l₂ * 13^l₃ := 
by
  sorry

end number_of_divisors_of_n_l25_25776


namespace number_of_dials_l25_25585

theorem number_of_dials (k : ℕ) (aligned_sums : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → aligned_sums i % 12 = aligned_sums j % 12) ↔ k = 12 :=
by
  sorry

end number_of_dials_l25_25585


namespace rainy_days_last_week_l25_25663

-- All conditions in Lean definitions
def even_integer (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def cups_of_tea_n (n : ℤ) : ℤ := 3
def total_drinks (R NR : ℤ) (m : ℤ) : Prop := 2 * m * R + 3 * NR = 36
def more_tea_than_hot_chocolate (R NR : ℤ) (m : ℤ) : Prop := 3 * NR - 2 * m * R = 12
def odd_number_of_rainy_days (R : ℤ) : Prop := R % 2 = 1
def total_days_in_week (R NR : ℤ) : Prop := R + NR = 7

-- Main statement
theorem rainy_days_last_week : ∃ R m NR : ℤ, 
  odd_number_of_rainy_days R ∧ 
  total_days_in_week R NR ∧ 
  total_drinks R NR m ∧ 
  more_tea_than_hot_chocolate R NR m ∧
  R = 3 :=
by
  sorry

end rainy_days_last_week_l25_25663
