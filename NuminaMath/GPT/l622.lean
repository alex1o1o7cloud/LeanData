import Mathlib

namespace calculate_total_cost_l622_622359

noncomputable def cost_of_zoo : ℕ := 8850

def condition_1 (goats : ℕ) (cost_per_goat : ℕ) : ℕ :=
  goats * cost_per_goat

def condition_2 (goats : ℕ) (cost_per_llama : ℕ) : ℕ :=
  2 * goats * cost_per_llama

def condition_3 (goats : ℕ) (cost_per_llama : ℕ) (kangaroo_discounted_cost : ℕ) : ℕ := 
  let kangaroos := 3 * goats
  in if kangaroos % 5 = 0
     then kangaroos / 5 * 5 * kangaroo_discounted_cost * 9 div 10
     else kangaroos * kangaroo_discounted_cost

theorem calculate_total_cost (goats : ℕ) (cost_per_goat : ℕ) (cost_per_llama : ℕ) (kangaroo_discounted_cost : ℕ) (total_cost : ℕ) : 
  goats = 3 →
  cost_per_goat = 400 →
  cost_per_llama = 600 →
  kangaroo_discounted_cost = 450 →
  total_cost = (condition_1 goats cost_per_goat) + (condition_2 goats cost_per_llama) + (condition_3 goats cost_per_llama kangaroo_discounted_cost) →
  total_cost = cost_of_zoo := 
by
  intros hgoats hcost_goat hcost_llama hkangaroo_discounted hc_total_cost
  rw [hgoats, hcost_goat, hcost_llama, hkangaroo_discounted, condition_1, condition_2, condition_3] at hc_total_cost
  sorry

#print axioms calculate_total_cost

end calculate_total_cost_l622_622359


namespace wrapping_paper_per_present_l622_622683

theorem wrapping_paper_per_present :
  ∀ (total: ℚ) (presents: ℚ) (frac_used: ℚ),
  total = 3 / 10 → presents = 3 → frac_used = total / presents → frac_used = 1 / 10 :=
by
  intros total presents frac_used htotal hpresents hfrac
  rw [htotal, hpresents, hfrac]
  sorry

end wrapping_paper_per_present_l622_622683


namespace selection_ways_l622_622688

/-- There are a total of 70 ways to select 3 people from 4 teachers and 5 students,
with the condition that there must be at least one teacher and one student among the selected. -/
theorem selection_ways (teachers students : ℕ) (T : 4 = teachers) (S : 5 = students) :
  ∃ (ways : ℕ), ways = 70 := by
  sorry

end selection_ways_l622_622688


namespace Nellie_needs_to_sell_more_rolls_l622_622926

-- Define the conditions
def total_needed : ℕ := 45
def sold_grandmother : ℕ := 1
def sold_uncle : ℕ := 10
def sold_neighbor : ℕ := 6

-- Define the total sold
def total_sold : ℕ := sold_grandmother + sold_uncle + sold_neighbor

-- Define the remaining rolls needed
def remaining_rolls := total_needed - total_sold

-- Statement to prove that remaining_rolls equals 28
theorem Nellie_needs_to_sell_more_rolls : remaining_rolls = 28 := by
  unfold remaining_rolls
  unfold total_sold
  unfold total_needed sold_grandmother sold_uncle sold_neighbor
  calc
  45 - (1 + 10 + 6) = 45 - 17 : by rw [Nat.add_assoc]
  ... = 28 : by norm_num

end Nellie_needs_to_sell_more_rolls_l622_622926


namespace ratio_of_milk_and_water_l622_622761

theorem ratio_of_milk_and_water (x y : ℝ) (hx : 9 * x = 9 * y) : 
  let total_milk := (7 * x + 8 * y)
  let total_water := (2 * x + y)
  (total_milk / total_water) = 5 :=
by
  sorry

end ratio_of_milk_and_water_l622_622761


namespace remainder_mod_7_l622_622085

theorem remainder_mod_7 : (4 * 6^24 + 3^48) % 7 = 5 := by
  sorry

end remainder_mod_7_l622_622085


namespace SteinerTheorem_l622_622006

variable {α β : Type} [EuclideanGeometry α]

open EuclideanGeometry

-- Defining the problem setup
def problem_statement (A B C D E : α) (BD CE : ℝ) (BD_bisects_ABC CE_bisects_ACB : Prop): Prop :=
  ∀ (△ABC: Triangle α),
    BD_bisects_ABC ∧
    CE_bisects_ACB ∧
    BD = CE →
    lengthOf A B = lengthOf A C

-- Statement for the theorem in Lean 4
theorem SteinerTheorem (A B C D E: α) 
  (BD_bisects_ABC: ∀ (p: Point α), onLine B D p ↔ onLine p A C) 
  (CE_bisects_ACB: ∀ (p: Point α), onLine C E p ↔ onLine p A B) 
  (BD_eq_CE : lengthOf B D = lengthOf C E): 
  lengthOf A B = lengthOf A C :=
by
  sorry

end SteinerTheorem_l622_622006


namespace sandy_transaction_gain_percentage_l622_622681

def purchase_prices : List ℕ := [900, 1100, 1200, 800, 1000]
def repair_costs : List ℕ := [300, 400, 500, 350, 450]
def selling_prices : List ℕ := [1320, 1620, 1880, 1150, 1500]

def total_gain_percent (purchase_prices repair_costs selling_prices : List ℕ) : ℚ :=
  let total_cost := (List.zip [purchase_prices, repair_costs]).map (λ (p : List ℕ) => p.reduce (λ a b => a + b)).reduce (λ a b => a + b)
  let total_revenue := selling_prices.reduce (λ a b => a + b)
  let total_gain := total_revenue - total_cost
  (total_gain / total_cost) * 100

theorem sandy_transaction_gain_percentage : total_gain_percent purchase_prices repair_costs selling_prices = 6.71 := 
by
  sorry

end sandy_transaction_gain_percentage_l622_622681


namespace amplitude_and_phase_shift_l622_622100

theorem amplitude_and_phase_shift (x : ℝ) : 
  ∃ (A : ℝ) (φ : ℝ), (λ x, A * Real.sin (4 * x - φ)) = (λ x, 3 * Real.sin (4 * x - (π / 4))) ∧ A = 3 ∧ φ = π / 16 :=
by
  use 3, π / 16
  split
  . intro x
    rw [show 4 * x - π / 4 = 4 * x - π / 4 by rfl]
  . split
    . reflexivity
    . reflexivity

end amplitude_and_phase_shift_l622_622100


namespace vector_magnitude_minimum_l622_622998

variable {α : Type*} [inner_product_space ℝ α]

-- Given conditions
variables (a b c : α)
variable (ha : ‖a‖ = 1)
variable (hab : ⟪a, b⟫ = 1)
variable (hbc : ⟪b, c⟫ = 1)
variable (hac : ⟪a, c⟫ = 2)

-- Goal
theorem vector_magnitude_minimum : 
  ‖a + b + c‖ ≥ 4 := 
sorry

end vector_magnitude_minimum_l622_622998


namespace find_a_for_quadratic_max_l622_622571

theorem find_a_for_quadratic_max :
  ∃ a : ℝ, (∀ x : ℝ, a ≤ x ∧ x ≤ 1/2 → (x^2 + 2 * x - 2 ≤ 1)) ∧
           (∃ x : ℝ, a ≤ x ∧ x ≤ 1/2 ∧ (x^2 + 2 * x - 2 = 1)) ∧ 
           a = -3 :=
sorry

end find_a_for_quadratic_max_l622_622571


namespace peanut_butter_candy_pieces_l622_622743

theorem peanut_butter_candy_pieces :
  ∀ (pb_candy grape_candy banana_candy : ℕ),
  pb_candy = 4 * grape_candy →
  grape_candy = banana_candy + 5 →
  banana_candy = 43 →
  pb_candy = 192 :=
by
  sorry

end peanut_butter_candy_pieces_l622_622743


namespace zero_taker_iff_l622_622828

theorem zero_taker_iff (m : ℕ) (h_m_pos : 0 < m) :
  (∃ k : ℕ, k ∣=k ^ 2 ∧ m ∣ k ∧ (∃ t : ℕ, k = 10 ^ (2021 + t) * 10 ^ 2021 * 2 * 10 + 1) ∧ k % 10 ≠ 0) ↔ ¬(10 ∣ m) :=
by
sor

end zero_taker_iff_l622_622828


namespace range_of_a_l622_622201

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end range_of_a_l622_622201


namespace probability_even_product_l622_622746

-- Define the set of numbers
def num_set : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define what it means for a product to be even
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the event that the product of three numbers is even
def even_product_event (s : Finset ℕ) : Prop :=
  ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ is_even (a * b * c)

-- Statement to prove
theorem probability_even_product : 
  (Finset.card (Finset.filter (λ s, even_product_event s) (Finset.powerset_len 3 num_set))).toReal / 
  (Finset.card (Finset.powerset_len 3 num_set)).toReal = 19 / 20 := 
sorry

end probability_even_product_l622_622746


namespace vector_sum_l622_622870

def u_0 : ℝ × ℝ := (2, 1)
def z_0 : ℝ × ℝ := (1, 5)

-- Define projection of a vector a onto vector b
def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 * b.1 + a.2 * b.2) / (b.1 * b.1 + b.2 * b.2)) • b

-- Define sequences u_n and z_n
def u (n : ℕ) (z : ℕ → ℝ × ℝ) : ℝ × ℝ :=
  projection (3 • z (n-1)) u_0

def z (n : ℕ) (u : ℕ → ℝ × ℝ) : ℝ × ℝ :=
  projection (2 • u n) z_0

-- Sum of the series
noncomputable def sum_u_series : ℝ × ℝ :=
  ∑' n, u (n + 1) z -- Using tsum for infinite sums

theorem vector_sum :
  sum_u_series = (546/40, 273/40) :=
sorry

end vector_sum_l622_622870


namespace range_of_m_l622_622169

noncomputable def f (x m : ℝ) : ℝ := x * (m + Real.exp (-x))

def f_prime (x m : ℝ) : ℝ := m + Real.exp (-x) - x * Real.exp (-x)

def g (x : ℝ) : ℝ := (x - 1) / Real.exp x

def g_prime (x : ℝ) : ℝ := (2 - x) / Real.exp x

theorem range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f_prime x1 m = 0 ∧ f_prime x2 m = 0) ↔ m ∈ (Set.Ioo 0 (Real.exp (-2))) := by
  sorry

end range_of_m_l622_622169


namespace repeating_decimal_sum_as_fraction_l622_622493

theorem repeating_decimal_sum_as_fraction : 0.\overline{7} + 0.\overline{13} = 10 / 11 :=
by
  sorry

end repeating_decimal_sum_as_fraction_l622_622493


namespace recurring_decimal_division_l622_622379

noncomputable def recurring_decimal_fraction : ℚ :=
  let frac_81 := (81 : ℚ) / 99
  let frac_36 := (36 : ℚ) / 99
  frac_81 / frac_36

theorem recurring_decimal_division :
  recurring_decimal_fraction = 9 / 4 :=
by
  sorry

end recurring_decimal_division_l622_622379


namespace minimize_triangle_perimeter_l622_622217

theorem minimize_triangle_perimeter (α : ℝ) (hα : 0 < α ∧ α < π / 2)
    (A : ℝ^3) (M N : set ℝ^3) (sphere : set ℝ^3)
    (h_sphere_tangent_M : sphere ⊆ M) (h_sphere_tangent_N : sphere ⊆ N)
    (B C : ℝ^3) (hB : B ∈ M) (hC : C ∈ N)
    (A' A'': ℝ^3) (hA' : A' = reflect_over_plane M A) (hA'' : A'' = reflect_over_plane N A) 
    (AG : ℝ) (h_projections : AG = dist (proj M A) (proj N A)) :
    2 * AG * sin α = perimeter (triangle A B C) :=
sorry

end minimize_triangle_perimeter_l622_622217


namespace expected_value_7_sided_die_l622_622381

theorem expected_value_7_sided_die : 
  let S := {1, 2, 3, 4, 5, 6, 7}
  in (∑ i in S, i * (1 / 7 : ℚ)) = 4 := 
by
  sorry

end expected_value_7_sided_die_l622_622381


namespace exist_min_distance_l622_622239

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

def on_circle (A : ℝ × ℝ) : Prop :=
  A.1 ^ 2 + A.2 ^ 2 = 16

def on_parabola (B : ℝ × ℝ) : Prop :=
  B.2 = B.1 ^ 2 - 4

theorem exist_min_distance : ∃ (A B : ℝ × ℝ), on_circle A ∧ on_parabola B ∧ 
  ∀ C D : ℝ × ℝ, on_circle C → on_parabola D → distance A B ≤ distance C D :=
sorry

end exist_min_distance_l622_622239


namespace pushkin_family_pension_l622_622388

def is_survivor_pension (pension : String) (main_provider_deceased : Bool) (provision_lifelong : Bool) (assigned_to_family : Bool) : Prop :=
  pension = "survivor's pension" ↔
    main_provider_deceased = true ∧
    provision_lifelong = true ∧
    assigned_to_family = true

theorem pushkin_family_pension :
  ∀ (pension : String),
    let main_provider_deceased := true
    let provision_lifelong := true
    let assigned_to_family := true
    is_survivor_pension pension main_provider_deceased provision_lifelong assigned_to_family →
    pension = "survivor's pension" :=
by
  intros pension
  intro h
  sorry

end pushkin_family_pension_l622_622388


namespace sum_first_2018_terms_of_given_sequence_l622_622534

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_first_2018_terms_of_given_sequence :
  let a := 1
  let d := -1 / 2017
  S_2018 = 1009 :=
by
  sorry

end sum_first_2018_terms_of_given_sequence_l622_622534


namespace count_factorable_polynomials_l622_622119

theorem count_factorable_polynomials : 
  (∃ (a : ℤ), 1 ≤ a * (a + 1) ∧ a * (a + 1) ≤ 2000) ↔ (finset.card (finset.filter (λ n : ℤ, ∃ a : ℤ, n = a * (a + 1) ∧ 1 ≤ n ∧ n ≤ 2000) (finset.range 2000)) = 89) :=
by
  sorry

end count_factorable_polynomials_l622_622119


namespace paperclip_problem_l622_622072

theorem paperclip_problem :
  ∃ k : ℕ, 5 * 4^k > 500 ∧ k + 1 = 5 :=
begin
  sorry
end

end paperclip_problem_l622_622072


namespace product_of_roots_eq_neg_14_l622_622502

theorem product_of_roots_eq_neg_14 :
  ∀ (x : ℝ), 25 * x^2 + 60 * x - 350 = 0 → ((-350) / 25) = -14 :=
by
  intros x h
  sorry

end product_of_roots_eq_neg_14_l622_622502


namespace coloring_square_l622_622467

theorem coloring_square (A B C D : Type) :
  (∃ color : A → bool, ∃ color : B → bool, ∃ color : C → bool, ∃ color : D → bool,
    (color A = color B = color C) ∨ (color B = color C = color D) ∨ (color C = color D = color A) ∨
    (color D = color A = color B)) →
  (((A = B) ∨ (B = C) ∨ (C = D) ∨ (D = A)) →
  ((color A = color B = color C) ∨ (color B = color C = color D) ∨
   (color C = color D = color A) ∨ (color D = color A = color B))) :=
by
  sorry

end coloring_square_l622_622467


namespace smallest_value_of_linear_expression_l622_622586

theorem smallest_value_of_linear_expression :
  (∃ a, 8 * a^2 + 6 * a + 5 = 7 ∧ (∃ b, b = 3 * a + 2 ∧ ∀ c, (8 * c^2 + 6 * c + 5 = 7 → 3 * c + 2 ≥ b))) → -1 = b :=
by
  sorry

end smallest_value_of_linear_expression_l622_622586


namespace lowest_point_graph_l622_622330

theorem lowest_point_graph (x : ℝ) (h : x > -1) : ∃ y, y = (x^2 + 2*x + 2) / (x + 1) ∧ y ≥ 2 ∧ (x = 0 → y = 2) :=
  sorry

end lowest_point_graph_l622_622330


namespace quadratic_real_roots_and_a_value_l622_622986

-- Define the quadratic equation (a-5)x^2 - 4x - 1 = 0
def quadratic_eq (a : ℝ) (x : ℝ) := (a - 5) * x^2 - 4 * x - 1

-- Define the discriminant for the quadratic equation
def discriminant (a : ℝ) := 4 - 4 * (a - 5) * (-1)

-- Main theorem statement
theorem quadratic_real_roots_and_a_value
    (a : ℝ) (x1 x2 : ℝ) 
    (h_roots : (a - 5) ≠ 0)
    (h_eq : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0)
    (h_sum_product : x1 + x2 + x1 * x2 = 3) :
    (a ≥ 1) ∧ (a = 6) :=
  sorry

end quadratic_real_roots_and_a_value_l622_622986


namespace original_number_is_144_l622_622189

theorem original_number_is_144 (x : ℕ) (h : x - x / 3 = x - 48) : x = 144 :=
by
  sorry

end original_number_is_144_l622_622189


namespace trig_expression_value_l622_622122

theorem trig_expression_value :
  (sin (18 * real.pi / 180) * cos (12 * real.pi / 180) + 
   cos (162 * real.pi / 180) * cos (102 * real.pi / 180)) / 
  (sin (22 * real.pi / 180) * cos (8 * real.pi / 180) + 
   cos (158 * real.pi / 180) * cos (98 * real.pi / 180)) = 1 := 
  sorry

end trig_expression_value_l622_622122


namespace ellipse_equation_l622_622073

-- Definitions of the tangents given as conditions
def tangent1 (x y : ℝ) : Prop := 4 * x + 5 * y = 25
def tangent2 (x y : ℝ) : Prop := 9 * x + 20 * y = 75

-- The statement we need to prove
theorem ellipse_equation :
  (∀ (x y : ℝ), tangent1 x y → tangent2 x y → 
  (∃ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0), a = 5 ∧ b = 3 ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1))) :=
sorry

end ellipse_equation_l622_622073


namespace joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l622_622637

noncomputable def distance_joseph : ℝ := 48 * 2.5 + 60 * 1.5
noncomputable def distance_kyle : ℝ := 70 * 2 + 63 * 2.5
noncomputable def distance_emily : ℝ := 65 * 3

theorem joseph_vs_kyle : distance_joseph - distance_kyle = -87.5 := by
  unfold distance_joseph
  unfold distance_kyle
  sorry

theorem emily_vs_joseph : distance_emily - distance_joseph = -15 := by
  unfold distance_emily
  unfold distance_joseph
  sorry

theorem emily_vs_kyle : distance_emily - distance_kyle = -102.5 := by
  unfold distance_emily
  unfold distance_kyle
  sorry

end joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l622_622637


namespace shortest_path_length_l622_622624

open Real

def A := (0, 0)
def D := (15, 20)
def O1 := (7.5, 10)
def O2 := (15, 5)
def r1 := 6
def r2 := 4

def circle1 (x y : ℝ) := (x - O1.1) ^ 2 + (y - O1.2) ^ 2 = r1 ^ 2
def circle2 (x y : ℝ) := (x - O2.1) ^ 2 + (y - O2.2) ^ 2 = r2 ^ 2

theorem shortest_path_length :
  let shortest_path := 30.6 + 5 * π / 3 
  in length_shortest_path_outside_circles A D circle1 circle2 r1 r2 = shortest_path :=
  sorry

end shortest_path_length_l622_622624


namespace C_max_matches_l622_622934

variable (players : Finset ℕ)
variable [DecidableEq ℕ]

-- Defining players as a Finite Set
def A := 0
def B := 1
def C := 2
def D := 3

-- Conditions of the problem:
-- Total number of matches
axiom total_matches : players.card = 4

-- Number of matches each pair of players competes exactly once
axiom round_robin : players.card * (players.card - 1) / 2 = 6

-- 甲 (A) won 2 matches
axiom A_wins : ∑ x in players.filter (λ p, p ≠ A), ite (A > x) 1 0 = 2

-- 乙 (B) won 1 match
axiom B_wins : ∑ x in players.filter (λ p, p ≠ B), ite (B > x) 1 0 = 1

-- Prove that 丙 (C) won at most 3 matches
theorem C_max_matches : (∑ x in players.filter (λ p, p ≠ C), ite (C > x) 1 0) ≤ 3 := sorry

end C_max_matches_l622_622934


namespace score_seventy_five_can_be_achieved_three_ways_l622_622611

-- Defining the problem constraints and goal
def quiz_problem (c u i : ℕ) (S : ℝ) : Prop :=
  c + u + i = 20 ∧ S = 5 * (c : ℝ) + 1.5 * (u : ℝ)

theorem score_seventy_five_can_be_achieved_three_ways :
  ∃ (c1 u1 c2 u2 c3 u3 : ℕ), 0 ≤ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ∧ (5 * (c1 : ℝ) + 1.5 * (u1 : ℝ)) ≤ 100 ∧
  (5 * (c2 : ℝ) + 1.5 * (u2 : ℝ)) = 75 ∧ (5 * (c3 : ℝ) + 1.5 * (u3 : ℝ)) = 75 ∧
  (c1 ≠ c2 ∧ u1 ≠ u2) ∧ (c2 ≠ c3 ∧ u2 ≠ u3) ∧ (c3 ≠ c1 ∧ u3 ≠ u1) ∧ 
  quiz_problem c1 u1 (20 - c1 - u1) 75 ∧
  quiz_problem c2 u2 (20 - c2 - u2) 75 ∧
  quiz_problem c3 u3 (20 - c3 - u3) 75 :=
sorry

end score_seventy_five_can_be_achieved_three_ways_l622_622611


namespace tissue_magnification_l622_622848

theorem tissue_magnification
  (diameter_magnified : ℝ)
  (diameter_actual : ℝ)
  (h1 : diameter_magnified = 5)
  (h2 : diameter_actual = 0.005) :
  diameter_magnified / diameter_actual = 1000 :=
by
  -- proof goes here
  sorry

end tissue_magnification_l622_622848


namespace sum_first_12_terms_l622_622573

def seq (a : ℕ → ℝ) : Prop :=
    a 1 = 1 ∧ ∀ n, a (n + 1) = if a n < 3 then a n + 1 else a n / 3

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

theorem sum_first_12_terms :
  ∀ (a : ℕ → ℝ), seq a → S a 12 = 24 :=
by
  intros a h_seq
  sorry

end sum_first_12_terms_l622_622573


namespace max_stamps_l622_622197

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h_price : price_per_stamp = 37) (h_total : total_money = 4000) : 
  ∃ max_stamps : ℕ, max_stamps = 108 ∧ max_stamps * price_per_stamp ≤ total_money ∧ ∀ n : ℕ, n * price_per_stamp ≤ total_money → n ≤ max_stamps :=
by
  sorry

end max_stamps_l622_622197


namespace magnitude_a_add_b_l622_622521

variable (x : ℝ)

def a : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (2, -4)
def b : ℝ × ℝ := (-1, x)

lemma parallel_b_c : b = (-1, 2) → b ∥ c :=
  by
  sorry

theorem magnitude_a_add_b :
  x = 2 → |(2, 1) + (-1, 2)| = sqrt 10 :=
  by
  intros hx
  have hb : b = (-1, 2) := by simp [hx]
  rw [hb]
  simp_arith
  sorry

end magnitude_a_add_b_l622_622521


namespace simplify_expression_l622_622308

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l622_622308


namespace two_pow_geq_n_cubed_for_n_geq_ten_l622_622514

theorem two_pow_geq_n_cubed_for_n_geq_ten (n : ℕ) (hn : n ≥ 10) : 2^n ≥ n^3 := 
sorry

end two_pow_geq_n_cubed_for_n_geq_ten_l622_622514


namespace equality_of_shaded_areas_l622_622225

open Real

-- Defining the necessary conditions
variable (φ : ℝ)

-- Given conditions
axiom φ_pos (h : 0 < φ)
axiom φ_lt_pi (h : φ < π)

-- Statement to prove
theorem equality_of_shaded_areas (hφ : 0 < φ ∧ φ < π) : tan φ = 2 * φ := by
  -- Proof is not required according to the instructions
  sorry

end equality_of_shaded_areas_l622_622225


namespace aluminum_weight_proportional_l622_622424

noncomputable def area_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length * side_length * Real.sqrt 3) / 4

theorem aluminum_weight_proportional (weight1 weight2 : ℝ) 
  (side_length1 side_length2 : ℝ)
  (h_density_thickness : ∀ s t, area_equilateral_triangle s * weight1 = area_equilateral_triangle t * weight2)
  (h_weight1 : weight1 = 20)
  (h_side_length1 : side_length1 = 2)
  (h_side_length2 : side_length2 = 4) : 
  weight2 = 80 :=
by
  sorry

end aluminum_weight_proportional_l622_622424


namespace binom_sum_l622_622464

def binom (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_sum : binom 7 4 + binom 6 5 = 41 := by
  sorry

end binom_sum_l622_622464


namespace original_amount_l622_622834

theorem original_amount {P : ℕ} {R : ℕ} {T : ℕ} (h1 : P = 1000) (h2 : T = 5) 
  (h3 : ∃ R, (1000 * (R + 5) * 5) / 100 + 1000 = 1750) : 
  1000 + (1000 * R * 5 / 100) = 1500 :=
by
  sorry

end original_amount_l622_622834


namespace mean_combination_l622_622999

variables {m n : ℕ} {x y z : ℕ }

/-- Given a sample with means x and y such that the mean of the combined sample is a*x + (1 - a)*y
    for some 0 < a ≤ 1/2, we need to show that m ≤ n. -/
theorem mean_combination (h1 : x ≠ y) (h2 : 0 < z ∧ z ≤ 1/2)
  (h3 : ∀ {m n}, (m:ℝ) / (m + n) = a  ∧ (n:ℝ) / (m + n) = 1 - a → a ≡ z): m ≤ n :=
sorry

end mean_combination_l622_622999


namespace median_of_partitioned_areas_l622_622255

-- Definitions based on conditions
noncomputable def square_side : ℝ := 5
noncomputable def A := (0 : ℝ, 0 : ℝ)
noncomputable def B := (square_side, 0 : ℝ)
noncomputable def C := (square_side, square_side)
noncomputable def D := (0 : ℝ, square_side)
noncomputable def E := ((square_side / 2), 0 : ℝ)

noncomputable def P := (2, -1)  -- Foot of perpendicular from B to CE (calculated in solution).
noncomputable def Q := (3, 1) -- Symmetrical point for Q
noncomputable def R := (0, 2) -- mid of BC

-- Theorem statement based on the final conclusion.
theorem median_of_partitioned_areas : 
  ∃ areas : ℝ × ℝ × ℝ × ℝ × ℝ, list.median [areas.1, areas.2, areas.3, areas.4, areas.5] = 5 := 
  sorry

end median_of_partitioned_areas_l622_622255


namespace factor_difference_of_squares_l622_622895

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l622_622895


namespace slower_train_passes_driver_of_faster_one_in_18_seconds_l622_622370

theorem slower_train_passes_driver_of_faster_one_in_18_seconds
  (length_train : ℝ)
  (speed_fast_km_hr : ℝ)
  (speed_slow_km_hr : ℝ)
  (length_train = 475 : ℝ)
  (speed_fast_km_hr = 55 : ℝ)
  (speed_slow_km_hr = 40 : ℝ) :
  let speed_fast_m_s := speed_fast_km_hr * (1000 / 3600)
  let speed_slow_m_s := speed_slow_km_hr * (1000 / 3600)
  let relative_speed := speed_slow_m_s + speed_fast_m_s
  let time_seconds := length_train / relative_speed
  time_seconds ≈ 18 := by
  sorry

end slower_train_passes_driver_of_faster_one_in_18_seconds_l622_622370


namespace simplify_fraction_l622_622311

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l622_622311


namespace barney_sold_items_l622_622082

theorem barney_sold_items (restocked_items storeroom_items items_left_in_store : ℕ) 
    (h1 : restocked_items = 4458) 
    (h2 : storeroom_items = 575) 
    (h3 : items_left_in_store = 3472) : 
    restocked_items + storeroom_items - items_left_in_store = 1561 :=
by
    rw [h1, h2, h3]
    norm_num

end barney_sold_items_l622_622082


namespace spherical_to_rectangular_coordinates_l622_622095

-- Define the given conditions
variable (ρ : ℝ) (θ : ℝ) (φ : ℝ)
variable (hρ : ρ = 6) (hθ : θ = 7 * Real.pi / 4) (hφ : φ = Real.pi / 2)

-- Convert spherical coordinates (ρ, θ, φ) to rectangular coordinates (x, y, z) and prove the values
theorem spherical_to_rectangular_coordinates :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 3 * Real.sqrt 2 ∧ y = -3 * Real.sqrt 2 ∧ z = 0 :=
by
  sorry

end spherical_to_rectangular_coordinates_l622_622095


namespace compare_values_of_even_and_monotone_function_l622_622244

variable (f : ℝ → ℝ)

def is_even_function := ∀ x : ℝ, f x = f (-x)
def is_monotone_increasing_on_nonneg := ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

theorem compare_values_of_even_and_monotone_function
  (h_even : is_even_function f)
  (h_monotone : is_monotone_increasing_on_nonneg f) :
  f (-π) > f 3 ∧ f 3 > f (-2) :=
by
  sorry

end compare_values_of_even_and_monotone_function_l622_622244


namespace op_neg2_3_l622_622398

def op (a b : ℤ) : ℤ := a^2 + 2 * a * b

theorem op_neg2_3 : op (-2) 3 = -8 :=
by
  -- proof
  sorry

end op_neg2_3_l622_622398


namespace gcd_consecutive_digits_l622_622909

theorem gcd_consecutive_digits (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) 
  (h₁ : b = a + 1) (h₂ : c = a + 2) (h₃ : d = a + 3) :
  ∃ g, g = gcd (1000 * a + 100 * b + 10 * c + d - (1000 * d + 100 * c + 10 * b + a)) 3096 :=
by {
  sorry
}

end gcd_consecutive_digits_l622_622909


namespace initial_speed_100_l622_622816

/-- Conditions of the problem:
1. The total distance from A to D is 100 km.
2. At point B, the navigator shows that 30 minutes are remaining.
3. At point B, the motorist reduces his speed by 10 km/h.
4. At point C, the navigator shows 20 km remaining, and the motorist again reduces his speed by 10 km/h.
5. The distance from C to D is 20 km.
6. The journey from B to C took 5 minutes longer than from C to D.
-/
theorem initial_speed_100 (x v : ℝ) (h1 : x = 100 - v / 2)
  (h2 : ∀ t, t = x / v)
  (h3 : ∀ t1 t2, t1 = (80 - x) / (v - 10) ∧ t2 = 20 / (v - 20))
  (h4 : (80 - x) / (v - 10) - 20 / (v - 20) = 1/12) :
  v = 100 := 
sorry

end initial_speed_100_l622_622816


namespace how_many_more_rolls_needed_l622_622923

variable (total_needed sold_to_grandmother sold_to_uncle sold_to_neighbor : ℕ)

theorem how_many_more_rolls_needed (h1 : total_needed = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end how_many_more_rolls_needed_l622_622923


namespace smallest_other_integer_l622_622716

variable {x : ℕ} (h_pos : x > 0)

theorem smallest_other_integer (gcd_18_a : Nat.gcd 18 a = x + 3)
                              (lcm_18_a : Nat.lcm 18 a = x * (x + 3)) :
                              a = 6 :=
by
sor

end smallest_other_integer_l622_622716


namespace probability_even_product_l622_622752

open BigOperators

def setS : Finset ℕ := {1, 2, 3, 4, 5, 6}

def choose3 : ℕ := (Finset.card (Finset.powersetLen 3 setS))

def oddS : Finset ℕ := {1, 3, 5}

def chooseOdd3 : ℕ := (Finset.card (Finset.powersetLen 3 oddS))

theorem probability_even_product :
  (1 : ℚ) - (chooseOdd3.to_rat / choose3.to_rat) = 19 / 20 := by
  sorry

end probability_even_product_l622_622752


namespace mean_of_other_two_numbers_l622_622506

def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

theorem mean_of_other_two_numbers :
  let nums := [1234, 1567, 1890, 2023, 2147, 2255, 2401].map (λ x : Int, (x : ℝ))
  mean (nums) = 2020 ->
    mean (nums.eraseFirst (λ x, x == 1234).eraseFirst (λ x, x == 1567).eraseFirst (λ x, x == 1890).eraseFirst (λ x, x == 2023).eraseFirst (λ x, x == 2147)) = 1708.5 :=
by
  sorry

end mean_of_other_two_numbers_l622_622506


namespace equilateral_perimeter_ADEFGC_l622_622469

open Real

-- Definitions of side lengths and midpoints
def triangle_side_length_ABC : ℝ := 6
def midpoint_D_AE (x : ℝ) : Prop := x = triangle_side_length_ABC / 2
def midpoint_F_AE (y : ℝ) : Prop := y = triangle_side_length_ABC / 4

-- Definitions of sides in the equilateral triangle ADE
def side_length_ADE : ℝ := 3

-- Definitions related to point G on DE
def point_G (DG GE : ℝ) : Prop := DG = 2 * GE ∧ DG + GE = side_length_ADE

-- Sum of calculated segments
def perimeter_ADEFGC : ℝ :=
  let AD := 3
  let DF := side_length_ADE / 2
  let FG := 3
  let GC := AD
  let CE := 1
  let EA := AD in
  AD + DF + FG + GC + CE + EA

theorem equilateral_perimeter_ADEFGC :
  perimeter_ADEFGC = 14.5 := by sorry

end equilateral_perimeter_ADEFGC_l622_622469


namespace range_of_a_l622_622992

theorem range_of_a 
  (a : ℝ)
  (h : ∀ x ∈ set.Icc (0:ℝ) 1, |x + a| + |x - 2| ≤ |x - 3|) :
  -1 ≤ a ∧ a ≤ 0 :=
begin
  sorry
end

end range_of_a_l622_622992


namespace collinear_points_l622_622556

-- Define the given points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point := {x := 1, y := 5, z := -2}
def B : Point := {x := 2, y := 4, z := 1}

-- Define C such that C(p, 3, q+2)
def C (p q : ℝ) : Point := {x := p, y := 3, z := q + 2}

-- Define the vectors AB and AC
def vector (P Q : Point) : Point :=
  {x := Q.x - P.x, y := Q.y - P.y, z := Q.z - P.z}

def AB : Point := vector A B
def AC (p q : ℝ) : Point := vector A (C p q)

-- Define the proof problem that p = 3 and q = 2 if A, B, and C are collinear
theorem collinear_points (p q : ℝ) :
  (∃ λ : ℝ, AB.x = λ * (AC p q).x ∧ AB.y = λ * (AC p q).y ∧ AB.z = λ * (AC p q).z) → p = 3 ∧ q = 2 :=
by
  sorry

end collinear_points_l622_622556


namespace five_rinds_possible_l622_622473

-- Definitions of the necessary conditions
def is_spherical (w : Type) : Prop :=
  ∃ r : ℝ, r > 0 ∧ (∀ p : w, dist p center = r) -- Center will be defined later

def has_rind (w : Type) : Prop :=
  outer_layer w -- Assume outer_layer is defined already

def cut_into_pieces (w : Type) (n : ℕ) : Prop :=
  n = 4 → partitioned w -- Assume partitioned extent is defined

def knife_longer_than_diameter (k : Type) (w : Type) : Prop :=
  (∃ d : ℝ, diameter w = d) → blade_length k > d

-- Statements to conclude possibility of 5 rinds
theorem five_rinds_possible (w : Type) (k : Type) [is_spherical w] [has_rind w]
  [cut_into_pieces w 4] [knife_longer_than_diameter k w] : 
  ∃ r : ℕ, r = 5 := 
sorry -- Proof goes here

end five_rinds_possible_l622_622473


namespace distinct_total_prices_count_l622_622742

open Finset

def gift_prices : Finset ℕ := {2, 5, 8, 11, 14}
def box_prices : Finset ℕ := {3, 5, 7, 9, 11}

theorem distinct_total_prices_count : 
  (gift_prices.product box_prices).image (λ p => p.1 + p.2)).card = 19 :=
by
  sorry

end distinct_total_prices_count_l622_622742


namespace rectangle_area_l622_622282

theorem rectangle_area (a b c : ℝ) :
  a = 15 ∧ b = 12 ∧ c = 1 / 3 →
  ∃ (AD AB : ℝ), 
  AD = (180 / 17) ∧ AB = (60 / 17) ∧ 
  (AD * AB = 10800 / 289) :=
by sorry

end rectangle_area_l622_622282


namespace example_l622_622452

theorem example : (144^2 - 121^2) / 23 = 265 := 
by
  have a := 144
  have b := 121
  have fact1 : a^2 - b^2 = (a + b) * (a - b) := by ring
  have fact2 : (144 + 121) = 265 := rfl
  have fact3 : (144 - 121) = 23 := rfl
  have h : (144^2 - 121^2) = 265 * 23 := by 
    rw fact1
    rw [fact2, fact3]
  rw h
  exact div_self (by norm_num [fact3])

end example_l622_622452


namespace area_diff_l622_622411

-- Defining the side lengths of squares
def side_length_small_square : ℕ := 4
def side_length_large_square : ℕ := 10

-- Calculating the areas
def area_small_square : ℕ := side_length_small_square ^ 2
def area_large_square : ℕ := side_length_large_square ^ 2

-- Theorem statement
theorem area_diff (a_small a_large : ℕ) (h1 : a_small = side_length_small_square ^ 2) (h2 : a_large = side_length_large_square ^ 2) : 
  a_large - a_small = 84 :=
by
  sorry

end area_diff_l622_622411


namespace simplify_and_evaluate_l622_622316

theorem simplify_and_evaluate 
  (x y : ℤ) 
  (h1 : |x| = 2) 
  (h2 : y = 1) 
  (h3 : x * y < 0) : 
  3 * x^2 * y - 2 * x^2 - (x * y)^2 - 3 * x^2 * y - 4 * (x * y)^2 = -18 := by
  sorry

end simplify_and_evaluate_l622_622316


namespace sqrt_equation_solution_l622_622323

theorem sqrt_equation_solution (y : ℝ) 
  (h : sqrt (2 + sqrt (3*y - 4)) = 4) :
  y = 200 / 3 := 
  sorry

end sqrt_equation_solution_l622_622323


namespace min_value_of_reciprocals_l622_622256

theorem min_value_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  ∃ c : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 3*y = 1 → c ≤ (1/x) + (1/y)) ∧ c = 8 + 4 * Real.sqrt 3 :=
begin
  use 8 + 4 * Real.sqrt 3,
  split,
  { intros x y hx hy hxy,
    sorry
  },
  refl,
end

end min_value_of_reciprocals_l622_622256


namespace maximum_teams_l622_622602

theorem maximum_teams
  (h1 : ∀ t, t ∈ teams → #t = 3)
  (h2 : ∀ t1 t2, t1 ∈ teams → t2 ∈ teams → t1 ≠ t2 → 
                 ∀ p1 ∈ t1, ∀ p2 ∈ t2, game p1 p2)
  (h3 : #games ≤ 150) :
  ∃ n : ℕ, #teams = n ∧ n = 6 := 
sorry

end maximum_teams_l622_622602


namespace unique_function_l622_622477

-- Define the type of positive natural numbers
def posNat := {n : ℕ // n > 0}

-- Define the function type
def posNatFunc := posNat → posNat

-- The main theorem to prove
theorem unique_function (f : posNatFunc) :
  (∀ m n : posNat, (f m).val^2 + (f n).val ∣ (m.val^2 + n.val)^2) →
  (∀ n : posNat, f n = n) :=
begin
  intros h n,
  sorry
end

end unique_function_l622_622477


namespace ordinate_of_vertex_of_parabola_l622_622406

theorem ordinate_of_vertex_of_parabola (n_1 n_2 : ℤ) (h1 : n_2 - n_1 = 2016)
  (h2 : ∃ x : ℤ, (x - n_1) * (x - n_2) = 2017) :
  let x_v := (n_1 + n_2) / 2 in
  let f (x : ℤ) := (x - n_1) * (x - n_2) in
  f x_v = -1016064 := by
  sorry

end ordinate_of_vertex_of_parabola_l622_622406


namespace right_triangle_legs_solutions_l622_622912

theorem right_triangle_legs_solutions (R r : ℝ) (h_cond : R / r ≥ 1 + Real.sqrt 2) :
  ∃ (a b : ℝ), 
    a = r + R + Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    b = r + R - Real.sqrt (R^2 - 2 * r * R - r^2) ∧ 
    (2 * R)^2 = a^2 + b^2 := by
  sorry

end right_triangle_legs_solutions_l622_622912


namespace percentage_of_three_items_sales_l622_622001

def treadmill_price : ℕ := 100
def chest_of_drawers_price (treadmill_price : ℕ) : ℕ := treadmill_price / 2
def television_price (treadmill_price : ℕ) : ℕ := treadmill_price * 3
def total_sales : ℕ := 600
def three_items_sales (treadmill_price : ℕ) (chest_of_drawers_price : ℕ) (television_price : ℕ) : ℕ := treadmill_price + chest_of_drawers_price + television_price

theorem percentage_of_three_items_sales : 
  let treadmill_price := 100 in
  let chest_of_drawers_price := treadmill_price / 2 in
  let television_price := treadmill_price * 3 in
  let total_sales := 600 in
  let three_items_sales := treadmill_price + chest_of_drawers_price + television_price in
  (three_items_sales * 100 / total_sales) = 75 :=
by
  let treadmill_price := 100
  let chest_of_drawers_price := treadmill_price / 2
  let television_price := treadmill_price * 3
  let total_sales := 600
  let three_items_sales := treadmill_price + chest_of_drawers_price + television_price
  show (three_items_sales * 100 / total_sales) = 75
  sorry

end percentage_of_three_items_sales_l622_622001


namespace fraction_zero_x_value_l622_622390

theorem fraction_zero_x_value (x : ℝ) (h1 : 2 * x = 0) (h2 : x + 3 ≠ 0) : x = 0 :=
by
  sorry

end fraction_zero_x_value_l622_622390


namespace minimum_kills_l622_622698

def distinct_distances (g : Fin 10 → ℝ × ℝ) : Prop :=
  ∀ i j : Fin 10, i ≠ j → (g i.1 - g j.1)^2 + (g i.2 - g j.2)^2 ≠ (g i'.1 - g j'.1)^2 + (g i'.2 - g j'.2)^2

noncomputable def nearest (g : Fin 10 → ℝ × ℝ) (i : Fin 10) : Fin 10 :=
  Fin.find (λ j, ∀ k ≠ j, (g i.1 - g j.1)^2 + (g i.2 - g j.2)^2 < (g i.1 - g k.1)^2 + (g i.2 - g k.2)^2)

def number_of_kills (g : Fin 10 → ℝ × ℝ) : ℕ :=
  (Finset.univ.filter (λ i, (nearest g (nearest g i) = i))).card

theorem minimum_kills (g : Fin 10 → ℝ × ℝ) (h_dist : distinct_distances g) : 
  number_of_kills g ≥ 3 :=
sorry

end minimum_kills_l622_622698


namespace valentines_distribution_l622_622268

theorem valentines_distribution (valentines_initial : ℝ) (valentines_needed : ℝ) (students : ℕ) 
  (h_initial : valentines_initial = 58.0) (h_needed : valentines_needed = 16.0) (h_students : students = 74) : 
  (valentines_initial + valentines_needed) / students = 1 :=
by
  sorry

end valentines_distribution_l622_622268


namespace plane_equation_l622_622056

variables (A B C x₀ y₀ z₀ x y z : ℝ)

-- Define the plane with a normal vector and passing through a specified point
def plane_eq (A B C x₀ y₀ z₀ x y z : ℝ) : Prop :=
  A * (x - x₀) + B * (y - y₀) + C * (z - z₀) = 0

-- Theorem statement
theorem plane_equation
  (A B C x₀ y₀ z₀ x y z : ℝ) 
  (normal_vector : (A, B, C) ≠ (0, 0, 0))
  (point_on_plane : true) : 
  plane_eq A B C x₀ y₀ z₀ x y z :=
begin
  -- Proof goes here
  sorry
end

end plane_equation_l622_622056


namespace a_11_plus_b_11_l622_622664

theorem a_11_plus_b_11 :
  (∀ n : ℕ, a (n + 1) + b (n + 1) = 2 * (n + 1) - 1) → a 11 + b 11 = 21 :=
begin
  intro h,
  specialize h 10,
  exact h,
end

end a_11_plus_b_11_l622_622664


namespace equal_areas_of_sub_quadrilaterals_l622_622031

open Set
open Function

structure Quadrilateral (V : Type) :=
  (A B C D : V)

variable {V : Type} [AddCommGroup V] [Module ℝ V] 

def midpoint (v w : V) : V := (1 / 2 : ℝ) • (v + w)

def area_quad (a b c d : V) : ℝ :=
  (1/2) * abs (det (b - a) (d - a)) + (1/2) * abs (det (c - a) (d - a))

theorem equal_areas_of_sub_quadrilaterals 
  (Q : Quadrilateral V)
  (P : V) (Q₂ : V) (O : V)
  (X : V) (Y : V) (Z : V) (T : V)
  (hP : P = midpoint Q.B Q.D)
  (hQ₂ : Q₂ = midpoint Q.A Q.C)
  (h_intersection : ∃ l : AffineSubspace ℝ V, collinear ℝ ({P, Q₂, O} : Set V))
  (hX : X = midpoint Q.A Q.B)
  (hY : Y = midpoint Q.B Q.C)
  (hZ : Z = midpoint Q.C Q.D)
  (hT : T = midpoint Q.D Q.A):
  area_quad O X Q.B Y = 
  area_quad O Y Q.C Z := 
  area_quad O Z Q.D T :=
  area_quad O T Q.A X :=
sorry

end equal_areas_of_sub_quadrilaterals_l622_622031


namespace soda_price_is_5_l622_622838

def soda_cost (admission_price : ℝ) (kids_half_price : Bool) (discount_rate : ℝ) (group_size : ℕ) (adults : ℕ) (children : ℕ) (total_paid : ℝ) : ℝ :=
  let adults_cost := adults * admission_price
  let children_cost := children * (admission_price / 2)
  let total_admission_cost := adults_cost + children_cost
  let discount := discount_rate * total_admission_cost
  let discounted_admission_cost := total_admission_cost - discount
  total_paid - discounted_admission_cost

theorem soda_price_is_5 : soda_cost 30 true 0.20 10 6 4 197 = 5 :=
  by
  sorry

end soda_price_is_5_l622_622838


namespace construct_triangle_l622_622094

noncomputable def triangle_exists (a b h : ℝ) : Prop :=
∃ (A B C : ℝ × ℝ), 
  let dist := λ (p1 p2 : ℝ × ℝ), (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 in
  let height := λ (A B C : ℝ × ℝ), 
    abs ((B.2 - A.2) * C.1 - (B.1 - A.1) * C.2 + B.1 * A.2 - B.2 * A.1) / sqrt ((B.2 - A.2)^2 + (B.1 - A.1)^2) in
  dist A B = a^2 ∧ dist A C = b^2 ∧ height A B C = h ∧ 
  ∃! (C' : ℝ × ℝ), dist A C' = b^2 ∧ height A B C' = h

theorem construct_triangle (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  triangle_exists a b h :=
sorry

end construct_triangle_l622_622094


namespace circle_area_l622_622873

theorem circle_area (x y : ℝ) : 
  4 * x^2 + 4 * y^2 - 8 * x + 12 * y - 20 = 0 → 
  real.pi * ((real.sqrt 33) / 2)^2 = (33 / 4) * real.pi :=
by
  intros h
  sorry

end circle_area_l622_622873


namespace consecutive_negative_integers_product_sum_l622_622733

theorem consecutive_negative_integers_product_sum (n : ℤ) 
  (h_neg1 : n < 0) 
  (h_neg2 : n + 1 < 0) 
  (h_product : n * (n + 1) = 2720) :
  n + (n + 1) = -105 :=
sorry

end consecutive_negative_integers_product_sum_l622_622733


namespace negation_of_sin_leq_one_l622_622727

theorem negation_of_sin_leq_one :
  ¬(∀ x : ℝ, sin x ≤ 1) → (∃ x : ℝ, sin x > 1) :=
by
  sorry

end negation_of_sin_leq_one_l622_622727


namespace simplify_fraction_l622_622314

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l622_622314


namespace solve_inequality_l622_622905

theorem solve_inequality (x : ℝ) : 
  (\frac{x * (x - 2)}{(x - 5)^2} ≥ 15) ↔ (x ∈ ((-∞ : set ℝ) ∪ (5, +∞ : set ℝ))) ∧ (x ≠ 5) :=
by 
  sorry

end solve_inequality_l622_622905


namespace divisor_of_number_l622_622822

theorem divisor_of_number (n d q p : ℤ) 
  (h₁ : n = d * q + 3)
  (h₂ : n ^ 2 = d * p + 3) : 
  d = 6 := 
sorry

end divisor_of_number_l622_622822


namespace rhombus_area_l622_622725

-- Define basic parameters for the problem
noncomputable def d₁ : ℝ := 30
noncomputable def θ : ℝ := Real.pi / 3 -- 60 degrees in radians
noncomputable def sinθ : ℝ := Real.sin θ -- sin 60 degrees

-- The problem statement in Lean
theorem rhombus_area (h₁ : d₁ = 30) (hθ : θ = Real.pi / 3) (hsinθ : sinθ = Real.sin (Real.pi / 3)) : 
  ∃ area : ℝ, area = 225 * Real.sqrt 3 :=
by
  sorry

end rhombus_area_l622_622725


namespace find_c_and_d_l622_622866

theorem find_c_and_d (c d : ℝ) (h : ℝ → ℝ) (f : ℝ → ℝ) (finv : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 * x - 5)
  (finv_eq : ∀ x, finv x = 6 * x - 3)
  (f_def : ∀ x, f x = c * x + d)
  (inv_prop : ∀ x, f (finv x) = x ∧ finv (f x) = x) :
  4 * c + 6 * d = 11 / 3 :=
by
  sorry

end find_c_and_d_l622_622866


namespace solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l622_622693

theorem solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1 :
  ∀ x : ℝ, 2 * x ^ 2 + 5 * x - 3 ≠ 0 ∧ 2 * x - 1 ≠ 0 → 
  (5 * x + 1) / (2 * x ^ 2 + 5 * x - 3) = (2 * x) / (2 * x - 1) → 
  x = -1 :=
by
  intro x h_cond h_eq
  sorry

end solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l622_622693


namespace factorial_fraction_simplification_l622_622458

theorem factorial_fraction_simplification : 
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 :=
by
  sorry

end factorial_fraction_simplification_l622_622458


namespace angle_bisector_NE_of_KNC_l622_622276

variables {n : ℕ}
variables (A B C K N E : Type*)
variables [has_vertex (regular_polygon (2 * n)) A B C]
variables [has_opposite_vertex (regular_polygon (2 * n)) B E]
variables [is_on_side K A B] [is_on_side N B C]
variables [has_angle_eq (angle K E N) (angle (regular_polygon (2 * n)).central_angle)]

theorem angle_bisector_NE_of_KNC (h : is_on_side K AB) (h2 : is_on_side N BC) (h3 : is_vertex_opposite E B)
  (h4 : ∠KEN = 180 / (2 * n)) :
  is_angle_bisector NE (angle K N C) :=
sorry

end angle_bisector_NE_of_KNC_l622_622276


namespace inequality_proof_l622_622715

noncomputable def f : ℝ → ℝ := sorry -- assume this function is our differentiable and symmetric function
def a := f 0
def b := f (-3)
def c := f 3

theorem inequality_proof :
  (∀ x : ℝ, differentiable ℝ f x) ∧
  (∀ x : ℝ, f (2 - x) = f x) ∧ 
  (∀ x : ℝ, x < 1 → (x - 1) * (deriv f x) < 0) →
  b < c ∧ c < a :=
by 
  sorry

end inequality_proof_l622_622715


namespace profit_calculation_correct_l622_622048

def main_actor_fee : ℕ := 500
def supporting_actor_fee : ℕ := 100
def extra_fee : ℕ := 50
def main_actor_food : ℕ := 10
def supporting_actor_food : ℕ := 5
def remaining_member_food : ℕ := 3
def post_production_cost : ℕ := 850
def revenue : ℕ := 10000

def total_actor_fees : ℕ := 2 * main_actor_fee + 3 * supporting_actor_fee + extra_fee
def total_food_cost : ℕ := 2 * main_actor_food + 4 * supporting_actor_food + 44 * remaining_member_food
def total_equipment_rental : ℕ := 2 * (total_actor_fees + total_food_cost)
def total_cost : ℕ := total_actor_fees + total_food_cost + total_equipment_rental + post_production_cost
def profit : ℕ := revenue - total_cost

theorem profit_calculation_correct : profit = 4584 :=
by
  -- proof omitted
  sorry

end profit_calculation_correct_l622_622048


namespace trigonometric_identity_l622_622676

open Real

-- Lean 4 statement
theorem trigonometric_identity (α β γ x : ℝ) :
  (sin (x - β) * sin (x - γ) / (sin (α - β) * sin (α - γ))) +
  (sin (x - γ) * sin (x - α) / (sin (β - γ) * sin (β - α))) +
  (sin (x - α) * sin (x - β) / (sin (γ - α) * sin (γ - β))) = 1 := 
sorry

end trigonometric_identity_l622_622676


namespace max_area_circle_center_l622_622593

noncomputable def circle_center_max_area (k : ℝ) :=
  if (1 - 3/4 * k^2 > 0) then (0, -1)
  else sorry

theorem max_area_circle_center :
  ∀ (k : ℝ), (x^2 + y^2 + k * x + 2 * y + k^2 = 0) → 
  circle_center_max_area k = (0, -1) :=
begin
  intro k,
  intro h,
  sorry
end

end max_area_circle_center_l622_622593


namespace pushkin_family_pension_l622_622389

def is_survivor_pension (pension : String) (main_provider_deceased : Bool) (provision_lifelong : Bool) (assigned_to_family : Bool) : Prop :=
  pension = "survivor's pension" ↔
    main_provider_deceased = true ∧
    provision_lifelong = true ∧
    assigned_to_family = true

theorem pushkin_family_pension :
  ∀ (pension : String),
    let main_provider_deceased := true
    let provision_lifelong := true
    let assigned_to_family := true
    is_survivor_pension pension main_provider_deceased provision_lifelong assigned_to_family →
    pension = "survivor's pension" :=
by
  intros pension
  intro h
  sorry

end pushkin_family_pension_l622_622389


namespace exists_k_bound_l622_622653

noncomputable theory

-- Definition of the polynomial set M
def M : set (ℝ → ℝ) := { P | ∃ (a b c d : ℝ), 
                             P = λ x, a * x^3 + b * x^2 + c * x + d ∧ 
                             ∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), abs (P x) ≤ 1 }

-- The statement that needs to be proven
theorem exists_k_bound (P : ℝ → ℝ) (hPM : P ∈ M) : ∃ k, ∀ P ∈ M, abs (P x).leading_coeff ≤ k ∧ k = 4 :=
sorry

end exists_k_bound_l622_622653


namespace find_f7_l622_622524

theorem find_f7 (f : ℝ → ℝ) (H1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y)) (H2 : f(6) = 3) : f(7) = 7 / 2 :=
sorry

end find_f7_l622_622524


namespace scientific_notation_correct_l622_622110

noncomputable def x : ℝ := 0.000000023
noncomputable def y : ℝ := 2.3 * 10 ^ (-8)

theorem scientific_notation_correct :
  x = y :=
sorry

end scientific_notation_correct_l622_622110


namespace number_of_possible_medians_l622_622731

-- Define the function to compute median of five sorted list
def median_of_list (l : List ℤ) : ℤ :=
  l.nth_le 2 (by linarith [List.length_eq_of_perm_same_length (List.perm_ext (List.nodup_ext l.length_eq_of_perm_same_length))])

-- The main theorem statement
theorem number_of_possible_medians (x y : ℤ) : 
  let points := [x, 11, 13, y, 12]
  let medians := {median_of_list (points.sorted)}
  medians.card = 3 := sorry

end number_of_possible_medians_l622_622731


namespace distance_between_centers_of_incircle_and_excircle_l622_622646

theorem distance_between_centers_of_incircle_and_excircle (DE DF EF : ℝ) 
  (hDE : DE = 20) (hDF : DF = 21) (hEF : EF = 29) : 
  let s := (DE + DF + EF) / 2,
      K := real.sqrt (s * (s - DE) * (s - DF) * (s - EF)),
      r := K / s in
  real.sqrt(232) / 14 := 
begin
  have s_def : s = 35 := by sorry,
  have K_def : K = 210 := by sorry,
  have r_def : r = 6 := by sorry,
  have DI : real.sqrt(196 + 36) = real.sqrt(232) := by sorry,
  have IE : real.sqrt(232) / 14 = real.sqrt(232) / 14 := by sorry,
  exact IE, 
end

end distance_between_centers_of_incircle_and_excircle_l622_622646


namespace ellipse_eccentricity_l622_622147

theorem ellipse_eccentricity
  {a b n : ℝ}
  (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (P : ℝ × ℝ), P.1 = n ∧ P.2 = 4 ∧ (n^2 / a^2 + 16 / b^2 = 1))
  (F1 F2 : ℝ × ℝ)
  (h4 : F1 = (c, 0))        -- Placeholders for focus coordinates of the ellipse
  (h5 : F2 = (-c, 0))
  (h6 : ∃ c, 4*c = (3 / 2) * (a + c))
  : 3 * c = 5 * a → c / a = 3 / 5 :=
by
  sorry

end ellipse_eccentricity_l622_622147


namespace simplify_expr_l622_622303

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l622_622303


namespace part1_rotate_table_l622_622802

-- Part 1: Prove existence of a rotation where at least two guests sit correctly
theorem part1_rotate_table (seats : Fin 15 → Fin 15) (h : ∀ i : Fin 15, seats i ≠ i) :
  ∃ k : ℕ, 1 ≤ k ∧ k < 15 ∧ ∃ i j : Fin 15, i ≠ j ∧ seats (i + k) % 15 = i ∧ seats (j + k) % 15 = j := by
  sorry

-- Part 2: Provide an example where only one guest sits correctly and no rotation fixes more
example seats_example : ∃ seats : Fin 15 → Fin 15,
    (∃ x, seats x = x)
    ∧ (∀ k : ℕ, 1 ≤ k ∧ k < 15 → ∃! i : Fin 15, seats (i + k) % 15 = i) := by
  sorry

end part1_rotate_table_l622_622802


namespace parabola_translation_l622_622757

theorem parabola_translation (a x y h k: ℝ) : 
  -- Condition 1: Original parabola equation
  (y = a * (x)^2) →
  -- Condition 2: Translated right by 2 units and upwards by 3 units
  (h = 2 ∧ k = 3) →
  -- Condition 3: Translated parabola passes through the point (3, -1)
  (let y' := a * (3 - h)^2 + k in y' = -1) →
  -- Conclusion: Equation of the moved parabola
  (y = -4 * (x - 2)^2 + 3) :=
sorry

end parabola_translation_l622_622757


namespace simplify_expression_l622_622317

theorem simplify_expression (a : ℝ) (h : 0 < a ∧ a < (π / 2)) :
  sqrt (1 + sin a) + sqrt (1 - sin a) - sqrt (2 + 2 * cos a) = 0 :=
by
  sorry

end simplify_expression_l622_622317


namespace proof_equation_C_exists_proof_k1_k2_product_proof_OP_OQ_sum_const_l622_622337

noncomputable def ellipse (a b : ℝ) (x y : ℝ) := x^2 / a^2 + y^2 / b^2 = 1
def equation_C : Prop := ellipse 2*sqrt 3 2 x y = x^2 / 12 + y^2 / 4 = 1
def k1_k2_condition (x0 y0 k1 k2 : ℝ) : Prop :=
    (x0^2 - 3) * k1^2 - 2 * x0 * y0 * k1 + y0^2 - 3 = 0 ∧ 
    (x0^2 - 3) * k2^2 - 2 * x0 * y0 * k2 + y0^2 - 3 = 0
def k1_k2_product (x0 y0 k1 k2 : ℝ) (h: ellipse 2*sqrt 3 2 x0 y0 = x0^2 / 12 + y0^2 / 4 = 1) : Prop :=
    k1 * k2 = -1/3
def OP_OQ_sum (x1 y1 x2 y2 : ℝ) : Prop :=
    x1^2 + y1^2 + x2^2 + y2^2 = 16

theorem proof_equation_C_exists : equation_C := sorry

theorem proof_k1_k2_product (x0 y0 k1 k2 : ℝ) (hxy : ellipse 2*sqrt 3 2 x0 y0 = x0^2 / 12 + y0^2 / 4 = 1) : 
    k1_k2_product x0 y0 k1 k2 hxy := sorry

theorem proof_OP_OQ_sum_const (x1 y1 x2 y2 : ℝ) : OP_OQ_sum x1 y1 x2 y2 := sorry

end proof_equation_C_exists_proof_k1_k2_product_proof_OP_OQ_sum_const_l622_622337


namespace triangle_to_square_difference_l622_622445

noncomputable def number_of_balls_in_triangle (T : ℕ) : ℕ :=
  T * (T + 1) / 2

noncomputable def number_of_balls_in_square (S : ℕ) : ℕ :=
  S * S

theorem triangle_to_square_difference (T S : ℕ) 
  (h1 : number_of_balls_in_triangle T = 1176) 
  (h2 : number_of_balls_in_square S = 1600) :
  T - S = 8 :=
by
  sorry

end triangle_to_square_difference_l622_622445


namespace heartsuit_example_l622_622868

def heartsuit (a b : ℤ) : ℤ := a * b^3 - 2 * b + 3

theorem heartsuit_example : heartsuit 2 3 = 51 :=
by
  sorry

end heartsuit_example_l622_622868


namespace remainder_of_9_pow_333_div_50_l622_622771

theorem remainder_of_9_pow_333_div_50 : (9 ^ 333) % 50 = 29 :=
by
  sorry

end remainder_of_9_pow_333_div_50_l622_622771


namespace C_pays_228_for_cricket_bat_l622_622061

def CostPriceA : ℝ := 152

def ProfitA (price : ℝ) : ℝ := 0.20 * price

def SellingPriceA (price : ℝ) : ℝ := price + ProfitA price

def ProfitB (price : ℝ) : ℝ := 0.25 * price

def SellingPriceB (price : ℝ) : ℝ := price + ProfitB price

theorem C_pays_228_for_cricket_bat :
  SellingPriceB (SellingPriceA CostPriceA) = 228 :=
by
  sorry

end C_pays_228_for_cricket_bat_l622_622061


namespace colten_chickens_l622_622678

/-
Define variables to represent the number of chickens each person has.
-/

variables (C : ℕ)   -- Number of chickens Colten has.
variables (S : ℕ)   -- Number of chickens Skylar has.
variables (Q : ℕ)   -- Number of chickens Quentin has.

/-
Define the given conditions
-/
def condition1 := Q + S + C = 383
def condition2 := Q = 2 * S + 25
def condition3 := S = 3 * C - 4

theorem colten_chickens : C = 37 :=
by
  -- Proof elaboration to be done with sorry for the auto proof
  sorry

end colten_chickens_l622_622678


namespace drum_Y_fill_level_l622_622486

variable (C : ℝ) -- Capacity of Drum X
variable (oil_X : ℝ) -- Amount of oil in Drum X
variable (oil_Y : ℝ) -- Amount of oil in Drum Y
variable (capacity_X : ℝ) -- Full capacity of Drum X
variable (capacity_Y : ℝ) -- Full capacity of Drum Y

-- Given conditions
variables (h1 : oil_X = C / 2) -- Drum X is 1/2 full
variables (h2 : capacity_Y = 2 * capacity_X) -- Drum Y's capacity is twice Drum X's
variables (h3 : oil_Y = capacity_Y / 3) -- Drum Y is 1/3 full

-- Definitions to compute final oil in Drum Y
noncomputable def total_oil_Y : ℝ := oil_Y + oil_X

-- Statement to prove: Drum Y's final fill level
theorem drum_Y_fill_level (h_total : total_oil_Y / capacity_Y = 7 / 12) : 
  oil_X = C / 2 ∧ capacity_Y = 2 * capacity_X ∧ oil_Y = capacity_Y / 3 → 
  total_oil_Y / capacity_Y = 7 / 12 :=
by 
  assume h : oil_X = C / 2 ∧ capacity_Y = 2 * capacity_X ∧ oil_Y = capacity_Y / 3
  show total_oil_Y / capacity_Y = 7 / 12
  sorry

end drum_Y_fill_level_l622_622486


namespace max_squared_norm_sum_val_l622_622242

open RealEuclideanSpace

noncomputable def max_squared_norm_sum (a b c : ℝ^3) (h_a : ∥a∥ = 1) (h_b : ∥b∥ = 2) (h_c : ∥c∥ = 3) : ℝ :=
  ( ∥(a + 2 • b)∥^2 + ∥(b + 2 • c)∥^2 + ∥(c + 2 • a)∥^2 )

theorem max_squared_norm_sum_val (a b c : ℝ^3) (h_a : ∥a∥ = 1) (h_b : ∥b∥ = 2) (h_c : ∥c∥ = 3) :
  max_squared_norm_sum a b c h_a h_b h_c = 94 :=
  sorry

end max_squared_norm_sum_val_l622_622242


namespace clock_face_sum_ge_21_l622_622699

theorem clock_face_sum_ge_21 (arr : Fin 12 → ℕ) :
  (∀ i : Fin 12, arr i ∈ (1:ℕ)..12) → 
  ∃ i : Fin 12, arr i + arr (i + 1) + arr (i + 2) % 12 ≥ 21 :=
sorry

end clock_face_sum_ge_21_l622_622699


namespace log_bounds_l622_622874

theorem log_bounds (x : ℝ) (hx1 : 100000 < x) (hx2 : x < 1000000) 
  (h5 : log 10 100000 = 5) (h6 : log 10 1000000 = 6) : 
  ∃ p q : ℤ, p = 5 ∧ q = 6 ∧ log 10 x > p ∧ log 10 x < q ∧ p + q = 11 :=
by
  -- Prove that log 10 158489 lies between 5 and 6
  use 5
  use 6
  sorry

end log_bounds_l622_622874


namespace y_relation_l622_622955

theorem y_relation (y1 y2 y3 : ℝ) : 
  (-4, y1) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) ∧
  (-1, y2) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) ∧
  (2, y3) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) →
  y2 > y3 ∧ y3 > y1 :=
begin
  sorry
end

end y_relation_l622_622955


namespace maximize_area_of_triangle_l622_622522

theorem maximize_area_of_triangle 
  (R : ℝ) 
  (hR : 0 < R) 
  (circle : ∀ (A : ℝ) (d : ℝ), 0 ≤ d → d ≤ R → real.cos (A / R) = d)
  (A : ℝ)
  (B C : ℝ) 
  (tangent_point : A) 
  (chord_parallel : ∀ (d : ℝ), d = sqrt(2) / 2 * R → B - C = 2 * sqrt(R^2 - d^2)) : 
  (∀ (d : ℝ), 0 ≤ d → d ≤ R → real.sin(d) = (d • tangent_point)):  
  d = (sqrt(2) / 2 * R) → 
  ∀ (area : real), 
  (area = sqrt(R^2 - d^2) * d) → 
  ∀ (d_optimal : ℝ), d_optimal = sqrt(2) / 2 * R → 
  ∀ (area_max : real), 
  area_max = sqrt(R^2 - (sqrt(2) / 2 * R)^2) * (sqrt(2) / 2 * R) := 
sorry

end maximize_area_of_triangle_l622_622522


namespace coefficient_1_over_x_l622_622619

theorem coefficient_1_over_x (n : ℕ) (x : ℝ) (hn : (-2)^n = -32) : 
  (finset.range (n + 1)).sum (λ r, nat.choose n r * (1 / real.sqrt x)^(n - r) * (-3)^r) =
  -270 :=
by
  sorry

end coefficient_1_over_x_l622_622619


namespace cosine_of_cross_section_plane_angle_l622_622335

noncomputable def cosine_of_angle (a : ℝ) (d : ℝ) : ℝ :=
  sqrt(1 - (d^2 / (a^2 / 2)))

theorem cosine_of_cross_section_plane_angle (a d : ℝ) (h_a : a = 2) (h_d : d = 1) :
  cosine_of_angle a d = 3 / 4 :=
by
  rw [h_a, h_d]
  -- the below is simplified algebra that would be done in the proof
  have h : sqrt(1 - (1^2 / (2^2 / 2))) = 3 / 4 := sorry
  exact h

end cosine_of_cross_section_plane_angle_l622_622335


namespace total_sandwiches_l622_622449

theorem total_sandwiches :
  let billy := 49
  let katelyn := billy + 47
  let chloe := katelyn / 4
  billy + katelyn + chloe = 169 :=
by
  sorry

end total_sandwiches_l622_622449


namespace f_mn_eq_f_m_mul_f_n_l622_622252

def is_odd_not_one (x : ℕ) : Prop := x > 1 ∧ x % 2 = 1

def S (m : ℕ) : Finset ℕ := {
  a ∈ Finset.range m ∣ ∃ x y, (x + y = a) ∧ ((x * y - 1) % m = 0)
}

def f (m : ℕ) : ℕ := (S m).card

theorem f_mn_eq_f_m_mul_f_n (m n : ℕ) (hm: is_odd_not_one m) (hn: is_odd_not_one n) (hrel: Nat.coprime m n) :
  f (m * n) = f m * f n :=
sorry

end f_mn_eq_f_m_mul_f_n_l622_622252


namespace max_dn_is_13_l622_622470

open Nat

def a (n : ℕ) : ℕ := 100 + n^2 + 3 * n

def d (n : ℕ) : ℕ := gcd (a n) (a (n + 1))

theorem max_dn_is_13 : ∃ n : ℕ, d n = 13 := sorry

end max_dn_is_13_l622_622470


namespace evaluate_expression_l622_622655

-- Given conditions
def a : ℕ := 3
def b : ℕ := 2

-- Proof problem statement
theorem evaluate_expression : (1 / 3 : ℝ) ^ (b - a) = 3 := sorry

end evaluate_expression_l622_622655


namespace k_of_neg7_l622_622649

noncomputable def h (x : ℝ) : ℝ := 4 * x - 9
noncomputable def k (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 2

theorem k_of_neg7 : k (-7) = 3 / 4 :=
by
  sorry

end k_of_neg7_l622_622649


namespace tom_should_pay_times_original_price_l622_622363

-- Definitions of the given conditions
def original_price : ℕ := 3
def amount_paid : ℕ := 9

-- The theorem to prove
theorem tom_should_pay_times_original_price : ∃ k : ℕ, amount_paid = k * original_price ∧ k = 3 :=
by 
  -- Using sorry to skip the proof for now
  sorry

end tom_should_pay_times_original_price_l622_622363


namespace circle_equation_l622_622628

theorem circle_equation (a r : Real)
  (condition1 : a > -2)
  (condition2 : (3)^2 + (a + 2)^2 = r^2)
  (condition3 : r = abs (2 * a - 4) / 3) :
  (a = -1 ∧ r = 2) :=
by {
  sorry,
}

end circle_equation_l622_622628


namespace previous_tree_height_is_6_l622_622631

-- Define the conditions
def current_rescue_tree_height : ℝ := 20
def current_rescue_rungs : ℕ := 40
def previous_rescue_rungs : ℕ := 12

-- Define the height per rung based on current rescue data
def height_per_rung : ℝ := current_rescue_tree_height / current_rescue_rungs

-- Define the previous tree height using the height per rung
def previous_rescue_tree_height : ℝ := previous_rescue_rungs * height_per_rung

-- Lean 4 statement to prove that the height of the previous tree is 6 feet
theorem previous_tree_height_is_6 : previous_rescue_tree_height = 6 := by
  -- Proof steps go here
  sorry

end previous_tree_height_is_6_l622_622631


namespace find_a_l622_622729

theorem find_a :
  ∃ (a : ℕ), (2018 - a) * (2019 - a) = (2054 - a) * (2011 - a) ∧ a = 2009 :=
by
  use 2009
  sorry

end find_a_l622_622729


namespace minimum_additional_coins_l622_622841

theorem minimum_additional_coins (n c : ℕ) (hn : n = 15) (hc : c = 94) : 
  let total_coins := (n * (n + 1)) / 2 in
  total_coins - c = 26 := 
by
  have hn' : n = 15 := hn
  have hc' : c = 94 := hc
  let total_coins := (n * (n + 1)) / 2
  show total_coins - c = 26
  sorry

end minimum_additional_coins_l622_622841


namespace only_integer_square_less_than_three_times_self_l622_622767

theorem only_integer_square_less_than_three_times_self :
  ∃! (x : ℤ), x^2 < 3 * x :=
by
  use 1
  split
  · -- Show that 1^2 < 3 * 1
    calc 1^2 = 1 : by norm_num
            ... < 3 : by norm_num
            ... = 3 * 1 : by norm_num
  · -- Show that for any x, if x^2 < 3 * x then x = 1
    intro y hy
    cases lt_or_ge y 1 with hy1 hy1
    · -- Case: y < 1
      exfalso
      calc y^2 ≥ 0 : by exact pow_two_nonneg y
              ... ≥ y * 3 - y : by linarith
              ...   = 3 * y - y : by ring
              ...   = 2 * y : by ring
      linarith
    cases lt_or_eq_of_le hy1 with hy1 hy1
    · -- Case: y = 2
      exfalso
      have h' := by linarith
      linarith
    · -- Case: y = 1
      exact hy1
    -- Case: y > 2
    exfalso
    calc y^2 ≥ y * 3 : by nlinarith
            ...   > y * 3 : by linarith
    linarith

end only_integer_square_less_than_three_times_self_l622_622767


namespace coupon1_best_discount_l622_622814

def coupon1_discount (x : ℝ) := 0.15 * x
def coupon2_discount (x : ℝ) := if x ≥ 150 then 30 else 0
def coupon3_discount (x : ℝ) := if x > 150 then 0.20 * (x - 150) else 0

def valid_price (x : ℝ) := 200 < x ∧ x < 600

def listed_prices := [199.95, 229.95, 249.95, 289.95, 319.95]

theorem coupon1_best_discount {x : ℝ} (hx : x ∈ listed_prices) (hx_valid : valid_price x) :
  coupon1_discount x > coupon2_discount x ∧
  coupon1_discount x > coupon3_discount x :=
  sorry

end coupon1_best_discount_l622_622814


namespace sin_cos_tan_positive_sin_cos_tan_negative_l622_622163

open Real

noncomputable def sin_cos_tan_sum (a : ℝ) (ha : a ≠ 0) : ℝ :=
  let x := -4 * a
  let y := 3 * a
  let r := √(x ^ 2 + y ^ 2)
  let sin_alpha := y / r
  let cos_alpha := x / r
  let tan_alpha := sin_alpha / cos_alpha
  sin_alpha + cos_alpha - tan_alpha

theorem sin_cos_tan_positive (a : ℝ) (ha : a ≠ 0) (hpos : a > 0) :
  sin_cos_tan_sum a ha = 11 / 20 :=
  sorry

theorem sin_cos_tan_negative (a : ℝ) (ha : a ≠ 0) (hneg : a < 0) :
  sin_cos_tan_sum a ha = 19 / 20 :=
  sorry

end sin_cos_tan_positive_sin_cos_tan_negative_l622_622163


namespace proposition1_proposition2_final_answer_is_B_l622_622165

open Complex

-- Statement (1)
theorem proposition1 (a b c : ℂ) (h : a^2 + b^2 > c^2) : a^2 + b^2 - c^2 > 0 :=
by sorry

-- Statement (2)
theorem proposition2 (a b c : ℂ) (h : a^2 + b^2 - c^2 > 0) : a^2 + b^2 > c^2 :=
by sorry

-- Counterexample for proposition2
example : ∃ (a b c : ℂ), a^2 + b^2 - c^2 > 0 ∧ ¬ (a^2 + b^2 > c^2) := 
by sorry

-- Verify that the answer is (B)
theorem final_answer_is_B : 
  (proposition1_is_correct : ∀ a b c : ℂ, a^2 + b^2 > c^2 → a^2 + b^2 - c^2 > 0) ∧ 
  (proposition2_is_incorrect : ¬ ∀ a b c : ℂ, a^2 + b^2 - c^2 > 0 → a^2 + b^2 > c^2) := 
by sorry

end proposition1_proposition2_final_answer_is_B_l622_622165


namespace angles_in_trapezoid_l622_622674

theorem angles_in_trapezoid (ABCD : Type) 
  [trapezoid ABCD] 
  {A B C D : point ABCD}
  (h1 : parallel AB CD)
  (h2 : length AB < length CD) : 
  angle A D B + angle A B C > angle C D A + angle D C B := 
sorry

end angles_in_trapezoid_l622_622674


namespace find_B_given_probability_l622_622759

variable {A B : ℕ}

/-- The statement of the problem -/
theorem find_B_given_probability 
  (hU1 : ∀ A : ℕ, A ≥ 2 → let p := (A / (A+2)) * ((A-1) / (A+1)) in p = 1/6)
  (hU2 : ∀ B : ℕ, B ≥ 0 → let p := (2 / (2+B)) * (1 / (1+B)) in 
            ((1/6) * p = 1/60)) :
  B = 3 :=
sorry

end find_B_given_probability_l622_622759


namespace race_head_start_l622_622394

theorem race_head_start
  (v_A v_B L x : ℝ)
  (h1 : v_A = (4 / 3) * v_B)
  (h2 : L / v_A = (L - x * L) / v_B) :
  x = 1 / 4 :=
sorry

end race_head_start_l622_622394


namespace product_of_integers_l622_622723

theorem product_of_integers (x y : ℕ) (h_gcd : Nat.gcd x y = 10) (h_lcm : Nat.lcm x y = 60) : x * y = 600 := by
  sorry

end product_of_integers_l622_622723


namespace blackjack_payment_l622_622806

def casino_payout (b: ℤ) (r: ℤ): ℤ := b + r
def blackjack_payout (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ): ℤ :=
  (ratio_numerator * bet) / ratio_denominator

theorem blackjack_payment (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ) (payout: ℤ):
  ratio_numerator = 3 → 
  ratio_denominator = 2 → 
  bet = 40 →
  payout = blackjack_payout bet ratio_numerator ratio_denominator → 
  casino_payout bet payout = 100 :=
by
  sorry

end blackjack_payment_l622_622806


namespace area_of_ABCD_l622_622277

-- Define points and conditions
variables (A B C D E F : Point)
variables (x y : ℝ)

-- Define the rectangle and point E
def rectangle (A B C D : Point) : Prop :=
  ∃ c d, A = (0, d) ∧ B = (2*c, d) ∧ C = (2*c, 0) ∧ D = (0, 0) ∧ 
          E = (c/2, 0)

-- Define point F as the intersection of BE and AC
def intersection_point (A B C E F : Point) : Prop :=
  line_through A C ∩ line_through B E = F

-- Define the area condition
def area_condition (A F E D : Point) (area_AFED : ℝ) : Prop :=
  calc_area A F E D = area_AFED

-- Given conditions and proof statement
theorem area_of_ABCD (h1 : rectangle A B C D)
                     (h2 : intersection_point A B C E F)
                     (h3 : area_condition A F E D 36) : 
                     calc_area A B C D = 144 :=
by sorry

end area_of_ABCD_l622_622277


namespace find_t_l622_622180

theorem find_t (t : ℝ) (h1 : t ∈ Ioo 0 real.pi) (h2 : sin (2 * t) = - ∫ 0..t, cos x) :
  t = 2 * real.pi / 3 :=
by
  sorry

end find_t_l622_622180


namespace sequence_of_arrows_l622_622407

theorem sequence_of_arrows (n : ℕ) (h : n % 5 = 0) : 
  (n < 570 ∧ n % 5 = 0) → 
  (n + 1 < 573 ∧ (n + 1) % 5 = 1) → 
  (n + 2 < 573 ∧ (n + 2) % 5 = 2) → 
  (n + 3 < 573 ∧ (n + 3) % 5 = 3) →
    true :=
by
  sorry

end sequence_of_arrows_l622_622407


namespace problem1_problem2_l622_622916

theorem problem1 (x : ℝ) (h : 4 * x^2 - 9 = 0) : x = 3/2 ∨ x = -3/2 :=
by
  sorry

theorem problem2 (x : ℝ) (h : 64 * (x-2)^3 - 1 = 0) : x = 2 + 1/4 :=
by
  sorry

end problem1_problem2_l622_622916


namespace incorrect_operation_l622_622016

theorem incorrect_operation (a x y : ℝ) : (x^5 + x^5) ≠ x^{10} :=
by
  sorry

end incorrect_operation_l622_622016


namespace factor_diff_of_squares_l622_622884

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l622_622884


namespace mod_equiv_22_l622_622696

theorem mod_equiv_22 : ∃ m : ℕ, (198 * 864) % 50 = m ∧ 0 ≤ m ∧ m < 50 ∧ m = 22 := by
  sorry

end mod_equiv_22_l622_622696


namespace only_integers_square_less_than_three_times_l622_622769

-- We want to prove that the only integers n that satisfy n^2 < 3n are 1 and 2.
theorem only_integers_square_less_than_three_times (n : ℕ) (h : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
sorry

end only_integers_square_less_than_three_times_l622_622769


namespace geometric_series_sum_l622_622011

open Real

theorem geometric_series_sum :
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  S = -716637955 / 16777216 :=
by
  let a1 := (5 / 4 : ℝ)
  let r := (5 / 4 : ℝ)
  let n := (12 : ℕ)
  let S := a1 * (1 - r^n) / (1 - r)
  have h : S = -716637955 / 16777216 := sorry
  exact h

end geometric_series_sum_l622_622011


namespace geometric_sequence_sum_ratio_l622_622937

theorem geometric_sequence_sum_ratio (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_nonzero_q : q ≠ 0) 
  (a2 : a_n 2 = a_n 1 * q) (a5 : a_n 5 = a_n 1 * q^4) 
  (h_condition : 8 * a_n 2 + a_n 5 = 0)
  (h_sum : ∀ n, S n = a_n 1 * (1 - q^n) / (1 - q)) : 
  S 5 / S 2 = -11 :=
by 
  sorry

end geometric_sequence_sum_ratio_l622_622937


namespace trig_identity_value_l622_622737

theorem trig_identity_value :
  sin (4 / 3 * π) * cos (5 / 6 * π) * tan (-4 / 3 * π) = - (3 * Real.sqrt 3) / 4 :=
by
  sorry

end trig_identity_value_l622_622737


namespace find_ellipse_equation_find_chord_length_l622_622952

def ellipse_with_properties (a b x y : ℝ) := (a > b ∧ b > 0) ∧ (y^2 / a^2 + x^2 / b^2 = 1) ∧ (sqrt (a^2 - b^2) / a = sqrt 3 / 2) ∧ (2 * b = 4)

def line_intersects_ellipse (x y : ℝ) := y = x + 2

theorem find_ellipse_equation :
  ∃ a b : ℝ, ellipse_with_properties a b x y -> (a = 4 ∧ b = 2) :=
sorry

theorem find_chord_length (x1 x2 y1 y2 : ℝ) :
  line_intersects_ellipse x y →
  ellipse_with_properties 4 2 x y →
  (y1 = 2) ∧ (y2 = -6/5) ∧ (x1 = 0) ∧ (x2 = -16/5) →
  dist (x1, y1) (x2, y2) = 16 * sqrt 2 / 5 :=
sorry

end find_ellipse_equation_find_chord_length_l622_622952


namespace locus_of_points_l622_622395

-- Definitions based on conditions from (a)
def projections (M : Point) (ABC : Triangle) : ProjTriangle := 
  sorry -- Assume this function projects M onto the sides of triangle ABC

-- Main statement to prove
theorem locus_of_points (M : Point) (ABC : Triangle) (sigma : ℝ) 
  (PQR : ProjTriangle := projections M ABC) 
  (σ : ℝ := area PQR)
  (R : ℝ := circumradius ABC) 
  (S : ℝ := area ABC) :
  0 < σ / S ∧ σ / S < 1 / 4 → 
  locus M = { C1, C2 : Circle // 
    C1.radius = sqrt(1 - 4 * σ / S) * R ∧ 
    C2.radius = sqrt(1 + 4 * σ / S) * R } ∧
  σ / S = 1 / 4 → 
  locus M = { C : Circle // 
    C.radius = sqrt(2) * R } ∧ 
  σ = 0 → 
  locus M = circumcircle(ABC) := 
sorry

end locus_of_points_l622_622395


namespace number_of_integer_factors_l622_622120

theorem number_of_integer_factors :
  let valid_n (n : ℕ) := ∃ (a b : ℤ), (a + b = -1) ∧ (a * b = - (n : ℤ))
  in (finset.filter valid_n (finset.range 2001)).card = 44 :=
by
  sorry

end number_of_integer_factors_l622_622120


namespace a1_is_minus_8_sequence_is_increasing_largest_n_satisfying_condition_l622_622984

-- Define the sum of the first n terms as given in the problem
def sum_of_first_n_terms (n : ℕ) : ℤ := n^2 - 9 * n

-- Define the sequence {a_n}
def sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  if n = 1 then a 1 else sum_of_first_n_terms n - sum_of_first_n_terms (n - 1)

-- Prove a_1 = -8
theorem a1_is_minus_8 : sequence sum_of_first_n_terms 1 = -8 := by
  sorry

-- Prove the sequence is increasing
theorem sequence_is_increasing : ∀ n : ℕ, n ≥ 1 → sequence sum_of_first_n_terms (n + 1) > sequence sum_of_first_n_terms n := by
  sorry

-- Prove the largest positive integer n = 8 that satisfies S_n < 0
theorem largest_n_satisfying_condition : 
  ∃ n : ℕ, S_n<0 ∧ ∀ m : ℕ, m > n → S_m ≥ 0 :=
by
  sorry

end a1_is_minus_8_sequence_is_increasing_largest_n_satisfying_condition_l622_622984


namespace linear_regression_increase_l622_622139

-- Define the linear regression function
def linear_regression (x : ℝ) : ℝ :=
  1.6 * x + 2

-- Prove that y increases by 1.6 when x increases by 1
theorem linear_regression_increase (x : ℝ) :
  linear_regression (x + 1) - linear_regression x = 1.6 :=
by sorry

end linear_regression_increase_l622_622139


namespace wrapping_paper_fraction_each_present_l622_622686

theorem wrapping_paper_fraction_each_present (total_fraction : ℚ) (num_presents : ℕ) 
  (H : total_fraction = 3/10) (H1 : num_presents = 3) :
  total_fraction / num_presents = 1/10 :=
by sorry

end wrapping_paper_fraction_each_present_l622_622686


namespace sum_of_valid_z_divisible_by_6_l622_622102

theorem sum_of_valid_z_divisible_by_6 : 
  (∑ z in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}.filter (λ z, (17 + z) % 3 = 0), z) = 12 :=
by
  sorry

end sum_of_valid_z_divisible_by_6_l622_622102


namespace integer_solutions_eq_l622_622112

theorem integer_solutions_eq (x y : ℤ) :
  3 * x^2 - y^2 = 3^(x + y) ↔ (x = 1 ∧ y = 0) ∨ 
                       (x = 3 ∧ y = 0) ∨ 
                       (x = -2 ∧ y = 3) ∨ 
                       (x = -6 ∧ y = 9) :=
by
  sorry

end integer_solutions_eq_l622_622112


namespace total_napkins_l622_622270

variable (initial_napkins Olivia_napkins Amelia_multiplier : ℕ)

-- Defining the conditions
def Olivia_gives_napkins : ℕ := 10
def William_initial_napkins : ℕ := 15
def Amelia_gives_napkins : ℕ := 2 * Olivia_gives_napkins

-- Define the total number of napkins William has now
def William_napkins_now : ℕ :=
  initial_napkins + Olivia_napkins + Amelia_gives_napkins

-- Proving the total number of napkins William has now is 45
theorem total_napkins (h1 : Olivia_napkins = 10)
                      (h2: initial_napkins = 15)
                      (h3: Amelia_multiplier = 2)
                      : William_napkins_now initial_napkins Olivia_napkins (Olivia_napkins * Amelia_multiplier) = 45 :=
by
  rw [←h1, ←h2, ←h3]
  sorry

end total_napkins_l622_622270


namespace family_gathering_total_people_l622_622026

theorem family_gathering_total_people (P : ℕ) 
  (h1 : P / 2 = 10) : 
  P = 20 := by
  sorry

end family_gathering_total_people_l622_622026


namespace simplify_expression_l622_622297

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622297


namespace ratio_matt_fem_4_1_l622_622266

-- Define Fem's current age
def FemCurrentAge : ℕ := 11

-- Define the condition about the sum of their ages in two years
def AgeSumInTwoYears (MattCurrentAge : ℕ) : Prop :=
  (FemCurrentAge + 2) + (MattCurrentAge + 2) = 59

-- Define the desired ratio as a property
def DesiredRatio (MattCurrentAge : ℕ) : Prop :=
  MattCurrentAge / FemCurrentAge = 4

-- Create the theorem statement
theorem ratio_matt_fem_4_1 (M : ℕ) (h : AgeSumInTwoYears M) : DesiredRatio M :=
  sorry

end ratio_matt_fem_4_1_l622_622266


namespace cost_of_candy_car_l622_622233

theorem cost_of_candy_car (starting_amount paid_amount change : ℝ) (h1 : starting_amount = 1.80) (h2 : change = 1.35) (h3 : paid_amount = starting_amount - change) : paid_amount = 0.45 := by
  sorry

end cost_of_candy_car_l622_622233


namespace min_moves_to_flip_all_cards_l622_622356

-- Number of cards on the table
def num_cards : ℕ := 7

-- Number of cards flipped in each move
def cards_flipped_per_move : ℕ := 5

-- Minimum number of moves required to flip all cards
def min_moves_required_to_flip_all_cards : ℕ := 3

-- Prove that the minimum number of moves required is 3
-- assuming we start with all cards in the same initial state
theorem min_moves_to_flip_all_cards :
  ∃ (moves : list (fin num_cards → bool)), moves.length = min_moves_required_to_flip_all_cards ∧
  (∀ (card : fin num_cards), (list.count (λ move, move card) moves) % 2 = 1) :=
sorry

end min_moves_to_flip_all_cards_l622_622356


namespace prism_volume_approximation_l622_622811

-- Define the radius of the spheres as 1
def sphere_radius : ℝ := 1

-- Define the conditions of the problem
def base_triangle_side_length : ℝ := 2 * (1 + Real.sqrt 3)

def base_area : ℝ :=
  Real.sqrt 3 / 4 * base_triangle_side_length^2

def prism_height : ℝ :=
  2 * sphere_radius + Real.sqrt (8 / 3)

def prism_volume : ℝ :=
  base_area * prism_height

-- Theorem statement
theorem prism_volume_approximation :
  abs (prism_volume - 47.00) < 0.01 :=
sorry

end prism_volume_approximation_l622_622811


namespace b_should_pay_348_48_l622_622024

/-- Definitions for the given conditions --/

def horses_a : ℕ := 12
def months_a : ℕ := 8

def horses_b : ℕ := 16
def months_b : ℕ := 9

def horses_c : ℕ := 18
def months_c : ℕ := 6

def total_rent : ℕ := 841

/-- Calculate the individual and total contributions in horse-months --/

def contribution_a : ℕ := horses_a * months_a
def contribution_b : ℕ := horses_b * months_b
def contribution_c : ℕ := horses_c * months_c

def total_contributions : ℕ := contribution_a + contribution_b + contribution_c

/-- Calculate cost per horse-month and b's share of the rent --/

def cost_per_horse_month : ℚ := total_rent / total_contributions
def b_share : ℚ := contribution_b * cost_per_horse_month

/-- Lean statement to check b's share --/

theorem b_should_pay_348_48 : b_share = 348.48 := by
  sorry

end b_should_pay_348_48_l622_622024


namespace abhinav_annual_salary_l622_622400

def RamMontlySalary : ℝ := 25600
def ShyamMontlySalary (A : ℝ) := 2 * A
def AbhinavAnnualSalary (A : ℝ) := 12 * A

theorem abhinav_annual_salary (A : ℝ) : 
  0.10 * RamMontlySalary = 0.08 * ShyamMontlySalary A → 
  AbhinavAnnualSalary A = 192000 :=
by
  sorry

end abhinav_annual_salary_l622_622400


namespace maple_taller_than_pine_l622_622206

theorem maple_taller_than_pine :
  let pine_tree := 24 + 1/4
  let maple_tree := 31 + 2/3
  (maple_tree - pine_tree) = 7 + 5/12 :=
by
  sorry

end maple_taller_than_pine_l622_622206


namespace sum_binom_solution_l622_622854

theorem sum_binom_solution:
  let binom := Nat.choose in
  (∑ n in {n | binom 28 16 + binom 28 n = binom 29 17}, n) = 28 :=
by
  sorry

end sum_binom_solution_l622_622854


namespace find_c_l622_622331

-- Defining the given condition
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 + c

theorem find_c : (∃ c : ℝ, ∀ x : ℝ, parabola x c = 2 * x^2 + 1) :=
by 
  sorry

end find_c_l622_622331


namespace polynomial_without_xy_l622_622597

theorem polynomial_without_xy (k : ℝ) (x y : ℝ) :
  ¬(∃ c : ℝ, (x^2 + k * x * y + 4 * x - 2 * x * y + y^2 - 1 = c * x * y)) → k = 2 := by
  sorry

end polynomial_without_xy_l622_622597


namespace incorrect_statements_l622_622513

-- Definitions corresponding to the problem conditions.
variables {α : Type*} [plane α] {m n : line} -- m and n are lines, and α is a plane.

-- Define the properties of being parallel and subset.
def is_parallel (l : line) (p : plane α) : Prop := -- Definition of line parallel to the plane.
  sorry

def is_subset (l : line) (p : plane α) : Prop := -- Definition of line being a subset of the plane.
  sorry

def is_skew (l₁ l₂ : line) : Prop := -- Definition of two skew lines.
  sorry

-- Conditions from the problem.
def statement_A := ∀ (m n : line) (α : plane α), is_subset m α → is_parallel n α → is_parallel m n
def statement_B := ∀ (m n : line) (α : plane α), is_subset m α → ¬ is_parallel n α → is_skew m n → is_parallel n α
def statement_C := ∀ (m n : line) (α : plane α), is_parallel m α → is_parallel n α → is_parallel m n
def statement_D := ∀ (m n : line) (α : plane α), is_parallel m n → is_parallel m α → ¬ is_subset n α → is_parallel n α

-- Proof goals:
theorem incorrect_statements : 
  ¬ statement_A ∧ ¬ statement_B ∧ ¬ statement_C :=
by
  -- Placeholder for proof.
  sorry

end incorrect_statements_l622_622513


namespace missing_range_value_l622_622042

-- Lean code for the translated mathematically equivalent proof problem.

theorem missing_range_value (R : ℕ) (h1 : Set.range ⟨15, R, 30⟩)
  (h2 : Min_range 25) : R = 25 :=
by 
  sorry

end missing_range_value_l622_622042


namespace probability_calculation_l622_622659

def probability_A : Prop :=
  let F := set.Icc (0 : ℝ) 1 ×ˢ set.Icc (0 : ℝ) 1
  let A := {p : ℝ × ℝ | 1 ≤ p.1 + p.2 ∧ p.1 + p.2 < 1.5}
  (∫⁻ p in A, 1 / ((area₀ F).to_nnreal : ℝ)) = 0.375

theorem probability_calculation : probability_A := sorry

end probability_calculation_l622_622659


namespace discounted_price_correct_l622_622074

def discounted_price (P : ℝ) : ℝ :=
  P * 0.80 * 0.90 * 0.95

theorem discounted_price_correct :
  discounted_price 9502.923976608186 = 6498.40 :=
by
  sorry

end discounted_price_correct_l622_622074


namespace natural_number_squares_l622_622416

theorem natural_number_squares (n : ℕ) (h : ∃ k : ℕ, n^2 + 492 = k^2) :
    n = 122 ∨ n = 38 :=
by
  sorry

end natural_number_squares_l622_622416


namespace q_joins_after_2_days_l622_622028

-- Define the conditions
def work_rate_p := 1 / 10
def work_rate_q := 1 / 6
def total_days := 5

-- Define the proof problem
theorem q_joins_after_2_days (a b : ℝ) (t x : ℕ) : 
  a = work_rate_p → b = work_rate_q → t = total_days →
  x * a + (t - x) * (a + b) = 1 → 
  x = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end q_joins_after_2_days_l622_622028


namespace ABCD_is_isosceles_trapezoid_l622_622368

variables {A B C D E : Type} [planar_geometry A B C D E] 

def is_trapezoid (ABCD : planar_geometry A B C D E) : Prop :=
  -- Replace this definition with a formal definition of a trapezoid in Lean if it exists.
  sorry

def is_isosceles_trapezoid (ABCD : planar_geometry A B C D E) : Prop :=
  -- Replace this definition with a formal definition of an isosceles trapezoid in Lean if it exists.
  sorry

theorem ABCD_is_isosceles_trapezoid 
  (ABCD : planar_geometry A B C D E)
  (h1 : is_trapezoid ABCD)
  (h2 : dist A C = dist B C + dist A D)
  (h3 : ∃ θ : ℝ, θ = 60 ∧ angle A C = θ ∨ angle B D = θ) :
  is_isosceles_trapezoid ABCD :=
sorry

end ABCD_is_isosceles_trapezoid_l622_622368


namespace surface_area_of_sphere_l622_622141

noncomputable def right_square_prism := sorry

def height_of_prism : ℝ := 4

def volume_of_prism : ℝ := 16

def is_base_square : Prop := sorry

def are_lateral_edges_perpendicular_to_base : Prop := sorry

def vertices_on_sphere (v : ℝ) : Prop :=
  (∃ (r : ℝ), (diameter_of_prism v = 2 * r))

theorem surface_area_of_sphere :
  vertices_on_sphere 16 → height_of_prism = 4 → volume_of_prism = 16 →
  is_base_square → are_lateral_edges_perpendicular_to_base →
  ∃ (S : ℝ), S = 24 * Real.pi :=
sorry

end surface_area_of_sphere_l622_622141


namespace simplify_expr_l622_622304

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l622_622304


namespace right_triangle_exists_with_area_ab_l622_622570

theorem right_triangle_exists_with_area_ab (a b c d : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
    (h1 : a * b = c * d) (h2 : a + b = c - d) :
    ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ (x * y / 2 = a * b) := sorry

end right_triangle_exists_with_area_ab_l622_622570


namespace number_of_different_tower_heights_l622_622431

theorem number_of_different_tower_heights :
  let heights := {0, 8, 17} in
  let bricks := (3 : ℕ) in
  let max_bricks := 100 in
  let min_height := max_bricks * bricks in
  let max_height := min_height + (max_bricks * 17) in
  (max_height - min_height + 1 = 1701) :=
by
  sorry

end number_of_different_tower_heights_l622_622431


namespace net_profit_100_patches_l622_622287

theorem net_profit_100_patches :
  let cost_per_patch := 1.25
  let num_patches_ordered := 100
  let selling_price_per_patch := 12.00
  let total_cost := cost_per_patch * num_patches_ordered
  let total_revenue := selling_price_per_patch * num_patches_ordered
  let net_profit := total_revenue - total_cost
  net_profit = 1075 :=
by
  sorry

end net_profit_100_patches_l622_622287


namespace not_equiv_in_power_l622_622404

def bin_op (a b : ℝ) : ℝ :=
if a ≥ b then a else b

theorem not_equiv_in_power (a b : ℝ) : (bin_op a b)^2 ≠ bin_op (a^2) (b^2) :=
sorry

end not_equiv_in_power_l622_622404


namespace pears_needed_l622_622635

-- Define the number of bananas and pears relationship
def bananas := ℕ 
def pears := ℕ

-- Define the conditions
def weight_equiv (n m : ℕ) : Prop := n = 3 * m / 2

-- State the theorem
theorem pears_needed (n : bananas) (m : pears) (h₁ : weight_equiv 9 6) (h₂ : n = 36) : m = 24 :=
by 
sorry

end pears_needed_l622_622635


namespace all_statements_imply_implication_l622_622092

variables (p q r : Prop)

theorem all_statements_imply_implication :
  (p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (¬ p ∧ ¬ q ∧ r → ((p → q) → r)) ∧
  (p ∧ ¬ q ∧ ¬ r → ((p → q) → r)) ∧
  (¬ p ∧ q ∧ r → ((p → q) → r)) :=
by { sorry }

end all_statements_imply_implication_l622_622092


namespace probability_sum_of_dice_is_15_l622_622753

def six_sided_dice_values := {2, 3, 4, 5, 6, 7}

def dice_roll_sum_probability_15 : ℚ :=
  let possible_outcomes := ((six_sided_dice_values × six_sided_dice_values) × six_sided_dice_values).to_finset
  let successful_outcomes := possible_outcomes.filter (λ xyz, xyz.1.1 + xyz.1.2 + xyz.2 = 15)
  successful_outcomes.card / (possible_outcomes.card : ℚ)

theorem probability_sum_of_dice_is_15 :
  dice_roll_sum_probability_15 = 5 / 108 := by
  sorry

end probability_sum_of_dice_is_15_l622_622753


namespace area_of_quadrilateral_AEFC_l622_622441

-- Definitions
def equilateral_triangle (a b c : ℝ) := a = b ∧ b = c
def perpendicular (a b c : ℝ) := a * b + b * c = 0

-- Problem statement
theorem area_of_quadrilateral_AEFC :
  let A B C E F B' AC AEFC : ℝ in
  equilateral_triangle A B C ∧
  (B' ∈ AC) ∧
  perpendicular F B' AC ∧
  (A, B, C, E, F, B') ∈ ℝ →
  area AEFC = (1/8) * (17 * real.sqrt 3 - 27) :=
sorry

end area_of_quadrilateral_AEFC_l622_622441


namespace problem_l622_622162

-- Definitions and hypotheses based on the given conditions
variable (a b : ℝ)
def sol_set := {x : ℝ | -1/2 < x ∧ x < 1/3}
def quadratic_inequality (x : ℝ) := a * x^2 + b * x + 2

-- Statement expressing that the inequality holds for the given solution set
theorem problem
  (h : ∀ (x : ℝ), x ∈ sol_set → quadratic_inequality a b x > 0) :
  a - b = -10 :=
sorry

end problem_l622_622162


namespace scout_troop_profit_l622_622831

def cost_per_bar (total_cost : ℝ) (num_bars : ℕ) : ℝ := total_cost / num_bars
def selling_price_per_bar (total_revenue : ℝ) (num_bars : ℕ) : ℝ := total_revenue / num_bars
def profit (revenue cost : ℝ) : ℝ := revenue - cost

theorem scout_troop_profit :
  let num_bars := 1500
  let cost_rate := 3 / 4 -- dollars per bar
  let selling_rate := 2 / 3 -- dollars per bar
  let total_cost := num_bars * cost_rate
  let total_revenue := num_bars * selling_rate
  in profit total_revenue total_cost = -125 :=
by
  let num_bars := 1500
  let cost_rate := 3 / 4 -- dollars per bar
  let selling_rate := 2 / 3 -- dollars per bar
  let total_cost := num_bars * cost_rate
  let total_revenue := num_bars * selling_rate
  have h1 : profit total_revenue total_cost = total_revenue - total_cost := rfl
  have h2 : total_cost = 1500 * (3 / 4) :=
    by
      calc
        1500 * (3 / 4) = 1500 * 0.75 : by simp
                    ... = 1125 : by norm_num
  have h3 : total_revenue = 1500 * (2 / 3) :=
    by
      calc
        1500 * (2 / 3) = 1500 * 0.6667 : by simp
                    ... = 1000 : by norm_num
  have h4 : profit 1000 1125 = 1000 - 1125 := by rfl
  have h5 : 1000 - 1125 = -125 := by norm_num
  rw [h2, h3, h4, h5]
  rfl

end scout_troop_profit_l622_622831


namespace hockey_tournament_max_extra_time_matches_l622_622612

theorem hockey_tournament_max_extra_time_matches :
  let teams := 2016
  ∃ (N : ℕ), (∀ matches : (teams * (teams - 1) / 2) ≥ 0,
        let total_points := 3 * matches in
        forall (extra_time_matches : total_points ≤ 2 * matches + 1 * matches),
        N = 1512) :=
sorry

end hockey_tournament_max_extra_time_matches_l622_622612


namespace series_conjecture_valid_l622_622269

-- Define the conjectured equation as a function statement
def series_eq (n : ℕ) : Prop := (∑ i in Finset.range n, 1 / ((2 * i + 1) * (2 * i + 3))) = n / (2 * n + 1)

-- Prove the conjectured equation for all natural numbers n
theorem series_conjecture_valid : ∀ n : ℕ, series_eq n :=
by
  -- Base case: simple induction proof structure
  induction n with 
  | zero => sorry
  | succ n ih => sorry

end series_conjecture_valid_l622_622269


namespace number_of_elements_set_one_l622_622194

noncomputable def cube_edge_length (A1 A2 A3 A4 B1 B2 B3 B4 : Type*) : ℝ := 1

def perpendicular (v1 v2 : Type*) : Prop :=
  vector.dot v1 v2 = 0

def dot_product_one (v1 v2 : Type*) : Prop :=
  vector.dot v1 v2 = 1

theorem number_of_elements_set_one (A1 A2 A3 A4 B1 B2 B3 B4 : Type*) 
  (h1 : ∀ i j, i ∈ {1,2,3,4} ∧ j ∈ {1,2,3,4} → perpendicular (overrightarrow A1 B1) (overrightarrow A1 B1))
  (h2 : dot_product_one (overrightarrow A1 B1) (overrightarrow A1 B1)):
  ∃ n, n = 1 := 
by 
  sorry

end number_of_elements_set_one_l622_622194


namespace rowing_upstream_speed_l622_622052

-- Definitions based on conditions
def V_m : ℝ := 45 -- speed of the man in still water
def V_downstream : ℝ := 53 -- speed of the man rowing downstream
def V_s : ℝ := V_downstream - V_m -- speed of the stream
def V_upstream : ℝ := V_m - V_s -- speed of the man rowing upstream

-- The goal is to prove that the speed of the man rowing upstream is 37 kmph
theorem rowing_upstream_speed :
  V_upstream = 37 := by
  sorry

end rowing_upstream_speed_l622_622052


namespace probability_within_circle_l622_622058

-- Define the vertices of the square region
def square_region : set (ℝ × ℝ) := {p | -3 ≤ p.1 ∧ p.1 ≤ 3 ∧ -3 ≤ p.2 ∧ p.2 ≤ 3}

-- Define the circle with center at (1, 1) and radius 2
def circle (center : ℝ × ℝ) (radius : ℝ) : set (ℝ × ℝ) :=
  {p | dist p center ≤ radius}

-- The center of the circle
def center_point := (1, 1)

-- The radius of the circle
def radius := 2

-- The area of the square
def area_square := 6 * 6

-- The area of the circle
noncomputable def area_circle := π * radius * radius

-- The probability that point Q lies within two units of (1, 1)
noncomputable def probability := area_circle / area_square

-- The proof that the calculated probability is indeed π/9
theorem probability_within_circle : probability = (π / 9) :=
by
  sorry

end probability_within_circle_l622_622058


namespace two_digit_numbers_units_greater_than_tens_count_l622_622216

def two_digit_units_greater_than_tens (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100 ∧ (n % 10) > (n / 10)

theorem two_digit_numbers_units_greater_than_tens_count : 
  { n : Nat | two_digit_units_greater_than_tens n }.to_finset.card = 36 := 
by
  sorry

end two_digit_numbers_units_greater_than_tens_count_l622_622216


namespace solve_trig_identity_proof_problem_l622_622580

noncomputable def trig_identity_proof_problem (α : ℝ) : Prop :=
  (cos α + sin α = 2/3) →
  (sqrt 2 * sin(2 * α - π / 4) + 1) / (1 + tan α) = -5 / 9 

theorem solve_trig_identity_proof_problem (α : ℝ) :
  trig_identity_proof_problem α :=
by {
  intro h,
  sorry
}

end solve_trig_identity_proof_problem_l622_622580


namespace train_crosses_signal_in_16_seconds_l622_622804

def train_length : ℝ := 300
def platform_length : ℝ := 431.25
def time_to_cross_platform : ℝ := 39
def distance_train_crosses_platform := train_length + platform_length
def speed_train := distance_train_crosses_platform / time_to_cross_platform
def time_to_cross_signal_pole := train_length / speed_train

theorem train_crosses_signal_in_16_seconds :
  time_to_cross_signal_pole = 16 := by
  sorry

end train_crosses_signal_in_16_seconds_l622_622804


namespace range_of_a_l622_622265

def sets_nonempty_intersect (a : ℝ) : Prop :=
  ∃ x, -1 ≤ x ∧ x < 2 ∧ x < a

theorem range_of_a (a : ℝ) (h : sets_nonempty_intersect a) : a > -1 :=
by
  sorry

end range_of_a_l622_622265


namespace beads_removed_l622_622071

def total_beads (blue yellow : Nat) : Nat := blue + yellow

def beads_per_part (total : Nat) (parts : Nat) : Nat := total / parts

def beads_remaining (per_part : Nat) (removed : Nat) : Nat := per_part - removed

def doubled_beads (remaining : Nat) : Nat := 2 * remaining

theorem beads_removed {x : Nat} 
  (blue : Nat) (yellow : Nat) (parts : Nat) (final_per_part : Nat) :
  total_beads blue yellow = 39 →
  parts = 3 →
  beads_per_part 39 parts = 13 →
  doubled_beads (beads_remaining 13 x) = 6 →
  x = 10 := by
  sorry

end beads_removed_l622_622071


namespace Mabel_total_tomatoes_l622_622661

variable (T : Type) [LinearOrderedField T]

-- Conditions given in the problem
def first (T : Type) [LinearOrderedField T] : T := 15
def second (T : Type) [LinearOrderedField T] : T := 2 * first T - 8
def third (T : Type) [LinearOrderedField T] : T := (first T) ^ 2 / 3
def fourth (T : Type) [LinearOrderedField T] : T := (first T + second T) / 2
def fifth (T : Type) [LinearOrderedField T] : T := 3 * Real.sqrt (first T + second T)
def sixth (T : Type) [LinearOrderedField T] : T := fifth T
def seventh (T : Type) [LinearOrderedField T] : T := 1.5 * (first T + second T + third T)
def eighth (T : Type) [LinearOrderedField T] : T := seventh T
def ninth (T : Type) [LinearOrderedField T] : T := 6 + first T + eighth T

-- Summing up all plants
def totalTomatoes (T : Type) [LinearOrderedField T] : T :=
  first T + second T + third T + fourth T + fifth T + sixth T + seventh T + eighth T + ninth T

-- The proof problem
theorem Mabel_total_tomatoes : totalTomatoes ℝ = 692 := 
by
  -- The actual proof would go here, but it is not required
  sorry

end Mabel_total_tomatoes_l622_622661


namespace part1_part2_part3_l622_622629

noncomputable def a (n : ℕ) : ℕ := 3^n - 1

def S (n : ℕ) : ℕ := (finset.range n).sum a

def b (n : ℕ) : ℝ := (Real.logb 3 ↑(a n + 1)) / 3^n

def T (n : ℕ) : ℝ := finset.sum (finset.range n) (λ i, b (i + 1))

def c (n : ℕ) : ℝ :=
  let anp1 := a (n + 1) in
  let an := a n
  in (1 / (1 + 1 / (↑an + 1))) + (1 / (1 - 1 / (↑anp1 + 1)))

def M (n : ℕ) : ℝ := finset.sum (finset.range n) (λ i, c (i + 1))

theorem part1 (h : ∀ n : ℕ, 2 * S n - 3 * a n + 2 * n = 0) : ∀ n : ℕ, a n = 3^n - 1 :=
sorry

theorem part2 (h : ∀ n, a n = 3^n - 1) : ∀ n, T n = 3 / 4 - (1 / 4) * (2 * (n : ℝ) + 3) / 3^n :=
sorry

theorem part3 (h : ∀ n, a n = 3^n - 1) : ∀ n : ℕ, M n > 2 * (n : ℝ) - 1 / 3 :=
sorry

end part1_part2_part3_l622_622629


namespace balloon_difference_l622_622658

theorem balloon_difference (x y : ℝ) (h1 : x = 2 * y - 3) (h2 : y = x / 4 + 1) : x - y = -2.5 :=
by 
  sorry

end balloon_difference_l622_622658


namespace pentagonal_prism_lateral_angle_l622_622845

theorem pentagonal_prism_lateral_angle (φ : ℝ) 
  (h1 : ∃ P : Set ℝ^3, is_pentagonal_prism P)
  (h2 : ∀ F, is_lateral_face F P → is_parallelogram F → ∃ φ, φ = 90): 
  φ = 90 := 
sorry

end pentagonal_prism_lateral_angle_l622_622845


namespace pentagon_area_AFEHG_l622_622081

noncomputable def pentagon_area_problem : ℝ :=
let s := 10 in
let hex_area := (3 * real.sqrt 3 / 2) * s^2 in
let H := (10 / 2, 0) in  -- Midpoint of DE assuming DE on x-axis for simplicity
let G := (2, real.sqrt 3 * s) in  -- Specific point on BC as per solution
let K := (5, real.sqrt 3 * s) in  -- Parallel argument in the solution
  -- Assuming all required points are correctly placed and known
let trapezoid_area := (1 / 2) * (10 + 15) * (5 * real.sqrt 3 / 2) in
let triangle_KGH_area := (1 / 2) * 3 * 15 * (real.sqrt 3 / 2) in
let triangle_BGA_area := (1 / 2) * 2 * 10 * (real.sqrt 3 / 2) in
  hex_area - trapezoid_area - triangle_KGH_area - triangle_BGA_area

theorem pentagon_area_AFEHG :
  let area := pentagon_area_problem in
  area = 205 * real.sqrt 3 / 2 :=
begin
  sorry
end

end pentagon_area_AFEHG_l622_622081


namespace napkins_total_l622_622273

theorem napkins_total (o a w : ℕ) (ho : o = 10) (ha : a = 2 * o) (hw : w = 15) :
  w + o + a = 45 :=
by
  sorry

end napkins_total_l622_622273


namespace min_value_of_reciprocals_l622_622257

theorem min_value_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  ∃ c : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 3*y = 1 → c ≤ (1/x) + (1/y)) ∧ c = 8 + 4 * Real.sqrt 3 :=
begin
  use 8 + 4 * Real.sqrt 3,
  split,
  { intros x y hx hy hxy,
    sorry
  },
  refl,
end

end min_value_of_reciprocals_l622_622257


namespace urn_problem_l622_622849

theorem urn_problem :
  (∃ N : ℕ, 
  let p_green := (6 / 10) * (20 / (20 + N)), 
      p_blue := (4 / 10) * (N / (20 + N)), 
      p_same_color := p_green + p_blue 
  in p_same_color = 0.65) → N = 4 :=
begin
  sorry
end

end urn_problem_l622_622849


namespace profit_calculation_correct_l622_622047

def main_actor_fee : ℕ := 500
def supporting_actor_fee : ℕ := 100
def extra_fee : ℕ := 50
def main_actor_food : ℕ := 10
def supporting_actor_food : ℕ := 5
def remaining_member_food : ℕ := 3
def post_production_cost : ℕ := 850
def revenue : ℕ := 10000

def total_actor_fees : ℕ := 2 * main_actor_fee + 3 * supporting_actor_fee + extra_fee
def total_food_cost : ℕ := 2 * main_actor_food + 4 * supporting_actor_food + 44 * remaining_member_food
def total_equipment_rental : ℕ := 2 * (total_actor_fees + total_food_cost)
def total_cost : ℕ := total_actor_fees + total_food_cost + total_equipment_rental + post_production_cost
def profit : ℕ := revenue - total_cost

theorem profit_calculation_correct : profit = 4584 :=
by
  -- proof omitted
  sorry

end profit_calculation_correct_l622_622047


namespace johns_monthly_earnings_l622_622232

variable (work_days : ℕ) (hours_per_day : ℕ) (former_wage : ℝ) (raise_percentage : ℝ) (days_in_month : ℕ)

def johns_earnings (work_days hours_per_day : ℕ) (former_wage raise_percentage : ℝ) (days_in_month : ℕ) : ℝ :=
  let days_worked := days_in_month / 2
  let total_hours := days_worked * hours_per_day
  let raise := former_wage * raise_percentage
  let new_wage := former_wage + raise
  total_hours * new_wage

theorem johns_monthly_earnings (work_days : ℕ := 15) (hours_per_day : ℕ := 12) (former_wage : ℝ := 20) (raise_percentage : ℝ := 0.3) (days_in_month : ℕ := 30) :
  johns_earnings work_days hours_per_day former_wage raise_percentage days_in_month = 4680 :=
by
  sorry

end johns_monthly_earnings_l622_622232


namespace mutually_exclusive_events_l622_622935

def bag : set string := {"red", "white"}

def draw_three (bag : list string) : list (set string) :=
  { S | S ⊆ bag ∧ set.card S = 3 }

def event1 (draw : set string) : Prop := draw = {"red", "red", "red"}
def event2 (draw : set string) : Prop := "white" ∈ draw

theorem mutually_exclusive_events :
  ∀ (draw : set string), draw ∈ draw_three ["red", "red", "red", "red", "red", "white", "white", "white", "white", "white"] →
  (event1 draw ∧ event2 draw) = false :=
by
  sorry

end mutually_exclusive_events_l622_622935


namespace solve_system_l622_622321

theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x - 1) = y + 6) 
  (h2 : x / 2 + y / 3 = 2) : 
  x = 10 / 3 ∧ y = 1 := 
by 
  sorry

end solve_system_l622_622321


namespace largest_divisor_of_n_l622_622592

theorem largest_divisor_of_n (n : ℤ) (h_pos : n > 0) (h_div : 251 ∣ n ^ 4) : 251 ∣ n :=
sorry

end largest_divisor_of_n_l622_622592


namespace relationship_y1_y2_y3_l622_622961

-- Conditions
def y1 := -((-4)^2) + 5
def y2 := -((-1)^2) + 5
def y3 := -(2^2) + 5

-- Statement to be proved
theorem relationship_y1_y2_y3 : y2 > y3 ∧ y3 > y1 := by
  sorry

end relationship_y1_y2_y3_l622_622961


namespace trapezoid_area_correct_l622_622427

/-
Conditions:
1. Trapezoid is circumscribed around a circle with radius R.
2. A chord connecting the points of tangency of the circle with the lateral sides of the trapezoid is equal to a.
3. The chord is parallel to the base of the trapezoid.
-/

variables (R a : ℝ)
-- It's necessary to assume that R and a are positive real numbers.
variables (hR : R > 0) (ha : a > 0)

-- The area of the trapezoid should be \(\frac{8R^3}{a}\).
def trapezoid_area : ℝ := (8 * R^3) / a

-- The theorem statement that we need to prove.
theorem trapezoid_area_correct : trapezoid_area R a = (8 * R^3) / a :=
by {
  -- proof is omitted
  sorry
}

end trapezoid_area_correct_l622_622427


namespace ratio_of_areas_l622_622221

theorem ratio_of_areas (square_side : ℝ) (x : ℝ) 
  (h_square : square_side = 3 * x)
  (h_AF : 2 * x = AF)
  (h_FE : x = FE)
  (h_CD : 2 * x = CD)
  (h_ED : x = ED) :
  let area_square := square_side ^ 2 in
  let area_triangle_BFD := (5 * x ^ 2) in
  (area_triangle_BFD / area_square) = (5 / 18) :=
sorry

end ratio_of_areas_l622_622221


namespace problem_difference_l622_622682

-- Define the sum of first n natural numbers
def sumFirstN (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Define the rounding rule to the nearest multiple of 5
def roundToNearest5 (x : ℕ) : ℕ :=
  match x % 5 with
  | 0 => x
  | 1 => x - 1
  | 2 => x - 2
  | 3 => x + 2
  | 4 => x + 1
  | _ => x  -- This case is theoretically unreachable

-- Define the sum of the first n natural numbers after rounding to nearest 5
def sumRoundedFirstN (n : ℕ) : ℕ :=
  (List.range (n + 1)).map roundToNearest5 |>.sum

theorem problem_difference : sumFirstN 120 - sumRoundedFirstN 120 = 6900 := by
  sorry

end problem_difference_l622_622682


namespace a1_value_a2_value_a3_value_is_geometric_sequence_a_n_formula_l622_622034

-- Definitions for Z extension and sequences
def initial_sequence : List ℕ := [1, 2, 3]

def Z_extension (seq : List ℕ) : List ℕ :=
seq.head! :: (List.zipWith (+) seq (seq.drop 1)) >>= (λ x, [x])

def a_n : ℕ → ℕ
| 0       := 6  -- This corresponds to the first term after initial sequence sum
| (n + 1) := (Z_extension (initial_sequence)).sum + (a_n n - initial_sequence.sum + initial_sequence.length - 1)

def b_n (n : ℕ) : ℕ := a_n n - 2

-- Theorem and proofs
theorem a1_value : a_n 1 = 14 := 
sorry

theorem a2_value : a_n 2 = 38 :=
sorry

theorem a3_value : a_n 3 = 110 :=
sorry

theorem is_geometric_sequence : ∃ r : ℕ, ∀ n : ℕ, b_n (n + 1) = r * b_n n :=
sorry

theorem a_n_formula : ∀ n, a_n n = 4 * 3^(n + 1) + 2 :=
sorry

end a1_value_a2_value_a3_value_is_geometric_sequence_a_n_formula_l622_622034


namespace probability_five_cards_l622_622918

/-- 
There are a standard deck of 52 cards. 
The sequence of drawing cards is considered:
1. The first card is a King.
2. The second card is a heart.
3. The third card is a Jack.
4. The fourth card is a spade.
5. The fifth card is a Queen.

The goal is to compute the probability of this sequence.
-/
noncomputable def card_probability : ℚ :=
  let prob_king := (4 / 52: ℚ)
  let prob_heart := (12 / 51: ℚ)
  let prob_jack := (4 / 50: ℚ)
  let prob_spade := (12 / 49: ℚ)
  let prob_queen := (4 / 48: ℚ)
  prob_king * prob_heart * prob_jack * prob_spade * prob_queen

theorem probability_five_cards :
  card_probability = (3 / 10125: ℚ) :=
by
  -- Probability calculations
  let prob_king := (4 / 52: ℚ)
  let prob_heart := (12 / 51: ℚ)
  let prob_jack := (4 / 50: ℚ)
  let prob_spade := (12 / 49: ℚ)
  let prob_queen := (4 / 48: ℚ)
  have h : prob_king * prob_heart * prob_jack * prob_spade * prob_queen = (3 / 10125: ℚ)
  sorry

end probability_five_cards_l622_622918


namespace product_of_exponents_lt_four_l622_622689

theorem product_of_exponents_lt_four (n : ℕ) : 
  (∏ k in Finset.range n, (2^k)^(1/(2^k : ℝ))) < 4 := 
sorry

end product_of_exponents_lt_four_l622_622689


namespace variance_all_members_l622_622755

open Real

-- Definitions for the conditions
def mean6 : ℝ := 8
def variance6 : ℝ := 5 / 3
def scores_new : List ℝ := [3, 5]

-- Assumptions based on conditions
axiom mean_senior_members (scores_senior : List ℝ) : scores_senior.length = 6 →
  (scores_senior.sum / 6 = mean6)

axiom variance_senior_members (scores_senior : List ℝ) : scores_senior.length = 6 →
  (scores_senior.map (fun x => (x - mean6)^2).sum / 6 = variance6)

-- The theorem to prove
theorem variance_all_members (scores_senior : List ℝ) :
  scores_senior.length = 6 →
  scores_senior.sum / 6 = mean6 →
  scores_senior.map (fun x => (x - mean6)^2).sum / 6 = variance6 →
  (scores_senior ++ scores_new).map (λ x => (x - (scores_senior.sum + 8) / 8)^2).sum / 8 = 9 / 2 :=
by sorry

end variance_all_members_l622_622755


namespace find_m_l622_622942

theorem find_m (m : ℂ) (h : (m^2 - (1 - complex.I) * m).re = 0) (h2 : m ≠ 0) : m = 1 :=
sorry

end find_m_l622_622942


namespace probability_of_even_product_l622_622747

open BigOperators

def total_combinations : Nat := nat.choose 6 3

def odd_combinations : Nat := nat.choose 3 3

def probability_odd_product : ℚ := odd_combinations / total_combinations

def probability_even_product : ℚ := 1 - probability_odd_product

theorem probability_of_even_product (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6}) :
  probability_even_product = 19 / 20 := by
  sorry

end probability_of_even_product_l622_622747


namespace lines_parallel_l622_622977

def line1 (a : ℝ) : AffinePlane ℝ := { p : ℝ × ℝ | (a + 2) * p.1 + 3 * p.2 = 5 }

def line2 (a : ℝ) : AffinePlane ℝ := { p : ℝ × ℝ | (a - 1) * p.1 + 2 * p.2 = 6 }

theorem lines_parallel (a : ℝ) : (∀ p1 p2 : ℝ × ℝ, p1 ∈ line1 a → p2 ∈ line2 a → 
  ((a + 2)/3) = ((a - 1)/2)) → a = 7 :=
by 
  sorry

end lines_parallel_l622_622977


namespace initial_blueberry_jelly_beans_l622_622462

-- Definitions for initial numbers of jelly beans and modified quantities after eating
variables (b c : ℕ)

-- Conditions stated as Lean hypothesis
axiom initial_relation : b = 2 * c
axiom new_relation : b - 5 = 4 * (c - 5)

-- Theorem statement to prove the initial number of blueberry jelly beans is 30
theorem initial_blueberry_jelly_beans : b = 30 :=
by
  sorry

end initial_blueberry_jelly_beans_l622_622462


namespace smallest_N_for_tournament_l622_622021

theorem smallest_N_for_tournament 
  (N : ℕ) 
  (teams_from_CA : ℕ := 251) 
  (total_teams : ℕ := 5 * N)
  (Alcatraz_wins_against_CA_more_than_any_other_CA : ∀ (team : ℕ), team ∈ CA_teams → Alcatraz_games_against_CA > team_games_against_CA)
  (Alcatraz_unique_loser : ∀ (team : ℕ), team ∈ all_teams → Alcatraz_total_losses > team_total_losses)
  : N = 255 := 
  sorry

end smallest_N_for_tournament_l622_622021


namespace read_journey_to_west_l622_622039

def students := 100
def read_either := 90
def read_red_chamber := 80
def read_both := 60

theorem read_journey_to_west (h1 : students = 100)
                            (h2 : read_either = 90)
                            (h3 : read_red_chamber = 80)
                            (h4 : read_both = 60) :
  ∃ (read_journey_to_west : ℕ), read_journey_to_west = 70 :=
by
  have step1 : read_either - read_both = 30 := by sorry
  have step2 : read_red_chamber - read_both = 20 := by sorry
  have step3 : 30 - 20 = 10 := by sorry
  have step4 : 10 + read_both = 70 := by sorry
  use 70
  exact step4

end read_journey_to_west_l622_622039


namespace proof_inequality_l622_622254

variables {p k : ℕ}
variables (a : Fin k → ℕ)

noncomputable def remainder (b p : ℕ) := b % p

noncomputable def S (p k : ℕ) (a : Fin k → ℕ) : Finset ℕ :=
  (Finset.range (p - 1)).filter (λ n,
    ∀ i j : Fin k, (i : ℕ) < j → remainder (n * a i) p < remainder (n * a j) p)

theorem proof_inequality {p k : ℕ} (hp : p.prime) (hk : k ≥ 3) 
  (ha : ∀ i : Fin k, a i % p ≠ 0) (hd : ∀ i j : Fin k, i ≠ j → (a i - a j) % p ≠ 0) :
  (S p k a).card < 2 * p / (k + 1) :=
sorry

end proof_inequality_l622_622254


namespace equation_of_ellipse_l622_622160

noncomputable theory

def center_origin (c : ℝ × ℝ) : Prop :=
  c = (0, 0)

def foci_positions (f1 f2 : ℝ × ℝ) : Prop :=
  f1 = (-3, 0) ∧ f2 = (3, 0)

def passes_through (p : ℝ × ℝ) (x y : ℝ) (a b : ℝ) : Prop :=
  p = (x, y) ∧ (x/a)^2 + (y/b)^2 = 1

theorem equation_of_ellipse
  (c : ℝ × ℝ) (f1 f2 p : ℝ × ℝ)
  (a b : ℝ)
  (h_c : center_origin c)
  (h_f : foci_positions f1 f2)
  (h_p : passes_through p 3 8 a b)
  (h_rel : 9 = a^2 - b^2) :
  (a = 9 ∧ b = √45) → (∀ x y, (x/a)^2 + (y/b)^2 = 1 ↔ (x/9)^2 + (y/√45)^2 = 1) :=
by {
  intro h_ab,
  cases h_ab with ha hb,
  rw [ha, hb],
  tauto,
}

end equation_of_ellipse_l622_622160


namespace initial_range_calculation_l622_622809

variable (initial_range telescope_range : ℝ)
variable (increased_by : ℝ)
variable (h_telescope : telescope_range = increased_by * initial_range)

theorem initial_range_calculation 
  (h_telescope_range : telescope_range = 150)
  (h_increased_by : increased_by = 3)
  (h_telescope : telescope_range = increased_by * initial_range) :
  initial_range = 50 :=
  sorry

end initial_range_calculation_l622_622809


namespace calculate_expression_value_l622_622853

theorem calculate_expression_value (y : ℕ) (h : y = 3) : y + y * (y ^ y) ^ 2 = 2190 :=
by {
  rw h,
  sorry
}

end calculate_expression_value_l622_622853


namespace closest_integer_to_sum_l622_622501

theorem closest_integer_to_sum : 
  ∃ k : ℤ, k = 102 ∧ 
  abs (500 * ∑ n in finset.Icc 4 20000, 1 / (n ^ 2 - 9) - k) < 0.5 :=
begin
  sorry,
end

end closest_integer_to_sum_l622_622501


namespace probability_even_product_l622_622744

-- Define the set of numbers
def num_set : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define what it means for a product to be even
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the event that the product of three numbers is even
def even_product_event (s : Finset ℕ) : Prop :=
  ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ is_even (a * b * c)

-- Statement to prove
theorem probability_even_product : 
  (Finset.card (Finset.filter (λ s, even_product_event s) (Finset.powerset_len 3 num_set))).toReal / 
  (Finset.card (Finset.powerset_len 3 num_set)).toReal = 19 / 20 := 
sorry

end probability_even_product_l622_622744


namespace expansion_terms_count_l622_622177

theorem expansion_terms_count (a b c d e f g h : Type) :
  (a + b + c + d) * (e + f + g + h) = 16 := 
sorry

end expansion_terms_count_l622_622177


namespace area_of_region_l622_622091

open Real

theorem area_of_region (a : ℝ) (ha : a > 0) :
  let curve1 := λ (x y : ℝ), (x - a * y)^2 = 9 * a^2
  let curve2 := λ (x y : ℝ), (2 * a * x + y)^2 = 4 * a^2
  ∃ (A : ℝ), ∀ (x y : ℝ), (curve1 x y → curve2 x y → A = 24 * a^2 / sqrt (1 + 5 * a^2 + a^4)) :=
by
  sorry

end area_of_region_l622_622091


namespace can_form_polygon_with_area_l622_622739

-- Define the conditions
def num_matches : Nat := 12
def match_length : ℝ := 2
def desired_area : ℝ := 16

-- The total length of all matchsticks
def total_length : ℝ := num_matches * match_length

-- Final statement to formalize the proof problem
theorem can_form_polygon_with_area :
  ∃ (polygon : Type) [isPolygon polygon] (matches_used : polygon → ℝ),
    (∀ (m : polygon), matches_used m = match_length) ∧
    (set.card (set.univ) = num_matches) ∧
    (polygon_area polygon = desired_area) := 
  sorry

end can_form_polygon_with_area_l622_622739


namespace median_duration_l622_622736

noncomputable def song_durations : List ℕ := [45, 50, 55, 70, 72, 75, 78, 80, 125, 130, 135, 145, 150, 180, 185, 190, 195, 200, 250, 255]

theorem median_duration (durations: List ℕ) (h : durations = [45, 50, 55, 70, 72, 75, 78, 80, 125, 130, 135, 145, 150, 180, 185, 190, 195, 200, 250, 255]) :
  durations.nth (durations.length / 2) == 130 :=
by {
  sorry
}

end median_duration_l622_622736


namespace average_marks_l622_622794

theorem average_marks :
  let class1_students := 26
  let class1_avg_marks := 40
  let class2_students := 50
  let class2_avg_marks := 60
  let total_students := class1_students + class2_students
  let total_marks := (class1_students * class1_avg_marks) + (class2_students * class2_avg_marks)
  (total_marks / total_students : ℝ) = 53.16 := by
sorry

end average_marks_l622_622794


namespace find_ellipse_equation_min_area_quadrilateral_l622_622949

-- Define the ellipse equation
noncomputable def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the given conditions
variables (a b : ℝ) (h_ab : a > b > 0)
variables (A : ℝ × ℝ) (h_A : A = (1, Real.sqrt 2 / 2))
variables (F : ℝ × ℝ) (h_F : F = (1, 0))

-- Problem statement: Find the equation of the ellipse 
theorem find_ellipse_equation : ellipse_eq a b 1 (Real.sqrt 2 / 2) → ellipse_eq (Real.sqrt 2) 1 := sorry

-- Define quadrilateral area 
noncomputable def area_quadrilateral (a1 a2 b1 b2 : ℝ × ℝ) : ℝ :=
  let d1 := (a1.1 - b1.1)^2 + (a1.2 - b1.2)^2
  let d2 := (a2.1 - b2.1)^2 + (a2.2 - b2.2)^2
  (d1 * d2) / 2

-- Define perpendicular lines passing through the focus
variables (l1 l2 : ℝ → ℝ × ℝ) (hl1 : ∀ y, l1 y = (y, 1))
variables (hl2 : ∀ x, l2 x = (1, x))

-- Minimum area problem statement
theorem min_area_quadrilateral : ∃ (a1 a2 b1 b2 : ℝ × ℝ), ellipse_eq (Real.sqrt 2) 1 a1.1 a1.2 ∧
                                                       ellipse_eq (Real.sqrt 2) 1 a2.1 a2.2 ∧
                                                       ellipse_eq (Real.sqrt 2) 1 b1.1 b1.2 ∧
                                                       ellipse_eq (Real.sqrt 2) 1 b2.1 b2.2 ∧
                                                       area_quadrilateral a1 a2 b1 b2 = 16 / 9 := sorry

end find_ellipse_equation_min_area_quadrilateral_l622_622949


namespace range_a_l622_622995

noncomputable def g (x : ℝ) : ℝ := x^2 + 2 - 8 * Real.log x

theorem range_a :
  ∀ a : ℝ, a ∈ set.range (λ x, g x) ↔ 6 - 8 * Real.log 2 ≤ a ∧ a ≤ 10 + 1 / Real.exp 2 :=
by
  sorry

end range_a_l622_622995


namespace complex_modulus_eq_sqrt_two_l622_622657

variable (x y : ℝ)

theorem complex_modulus_eq_sqrt_two (h : (1 + Complex.i) * x = 1 + y * Complex.i) : Complex.abs (x + y * Complex.i) = Real.sqrt 2 :=
by 
  sorry

end complex_modulus_eq_sqrt_two_l622_622657


namespace rightmost_three_digits_seven_pow_1983_add_123_l622_622764

theorem rightmost_three_digits_seven_pow_1983_add_123 :
  (7 ^ 1983 + 123) % 1000 = 466 := 
by 
  -- Proof steps are omitted
  sorry 

end rightmost_three_digits_seven_pow_1983_add_123_l622_622764


namespace linear_inverse_l622_622035

-- Given that f(x) and g(x) are linear functions such that f(g(x)) = g(f(x)) = x
-- For all x ∈ ℝ, f(0) = 4 and g(5) = 17
-- We need to prove that f(2006) = 122

noncomputable def f (x : ℝ) : ℝ := (1 / 17) * x + 4
noncomputable def g (x : ℝ) : ℝ := 17 * (x - 4)

theorem linear_inverse (x : ℝ) :
  (∀ x, f(g(x)) = x ∧ g(f(x)) = x) ∧ f(0) = 4 ∧ g(5) = 17 -> f(2006) = 122 := by
  intros
  -- Adding the assumptions
  let f := (λ x, (1/17) * x + 4)
  let g := (λ x, 17 * (x - 4))
  -- sorry is used as placeholder
  sorry

end linear_inverse_l622_622035


namespace find_special_numbers_l622_622438

/-- Definition of special number -/
def is_special (n : ℕ) (h : n ≥ 3) : Prop :=
  let harmonic_sum := (finset.range (n-1)).sum (λ k, 1 / (k + 1) : ℚ)
  ¬ (n ∣ (nat.factorial (n-1) * harmonic_sum.num)).nat_abs

/-- Theorem stating the special numbers in the specified range -/
theorem find_special_numbers :
  {n : ℕ | 10 ≤ n ∧ n ≤ 100 ∧ is_special n (by linarith)} =
  {10, 14, 22, 26, 34, 38, 46, 58, 62, 74, 82, 86, 94} := sorry

end find_special_numbers_l622_622438


namespace point_A_on_x_axis_l622_622278

def point_A : ℝ × ℝ := (-2, 0)

theorem point_A_on_x_axis : point_A.snd = 0 :=
by
  unfold point_A
  sorry

end point_A_on_x_axis_l622_622278


namespace triangle_angle_C_l622_622945

theorem triangle_angle_C (a b c : ℝ) (h_area : (a^2 + b^2 - c^2) / 4 = (1 / 2) * a * b * sin (real.pi / 2)) : 
  ∠C = real.pi / 2 :=
begin
  sorry
end

end triangle_angle_C_l622_622945


namespace max_tan_B_l622_622545

theorem max_tan_B (A B : ℝ) (C : Prop) 
  (sin_pos_A : 0 < Real.sin A) 
  (sin_pos_B : 0 < Real.sin B) 
  (angle_condition : Real.sin B / Real.sin A = Real.cos (A + B)) :
  Real.tan B ≤ Real.sqrt 2 / 4 :=
by
  sorry

end max_tan_B_l622_622545


namespace fraction_tea_in_cup3_l622_622362

-- Definitions based on conditions
def cup1_initial_tea := 6 -- ounces
def cup2_initial_milk := 6 -- ounces
def cup3_initial_honey := 3 -- ounces
def half := (1 / 2 : ℝ)

-- Initial conditions
axiom cup1 : ℝ -- initial ounces of tea in cup 1
axiom cup2 : ℝ -- initial ounces of milk in cup 2
axiom cup3 : ℝ -- initial ounces of honey in cup 3
axiom initial_conditions : cup1 = cup1_initial_tea ∧ cup2 = cup2_initial_milk ∧ cup3 = cup3_initial_honey

-- Process conditions
axiom half_tea_poured1_to_2 : cup2 = cup2 + cup1 * half
axiom stirred_thoroughly : true -- any mixing results in homogeneous mixture in cup 2
axiom half_mixture_poured_back : cup1 = (cup1 - cup1 * half) + (cup2 * half)

-- Final step
axiom pour_from_cup1_to_cup3 : ∀ C3_initial, C3_initial = cup3_initial_honey → 
                                          ∃ C3_final, C3_final = C3_initial + 3 ∧ 
                                          C3_final_fraction_tea = (3 * ((cup1 - cup1_initial_tea * half) + (cup2 * half)) / (3 + C3_initial))

-- The final proof statement
theorem fraction_tea_in_cup3 :
  initial_conditions →
  half_tea_poured1_to_2 →
  stirred_thoroughly →
  half_mixture_poured_back →
  pour_from_cup1_to_cup3 cup3_initial_honey →
  (cup3_initial_honey / (cup3_initial_honey + 3) = (1 / 6 : ℝ)) :=
  sorry

end fraction_tea_in_cup3_l622_622362


namespace michael_sixth_score_is_130_l622_622663

def initial_scores : list ℕ := [120, 130, 140, 150, 160]

def new_median (scores : list ℕ) : ℕ :=
  (scores.nth (scores.length / 2 - 1) + scores.nth (scores.length / 2)) / 2

theorem michael_sixth_score_is_130 (x : ℕ) (h : new_median (initial_scores ++ [x]) = 135) : x = 130 :=
sorry

end michael_sixth_score_is_130_l622_622663


namespace distinct_pos_integers_no_AP_l622_622227

theorem distinct_pos_integers_no_AP (n : ℕ) (k : ℕ) : 
  (n ≤ 10^5) → (k ≤ 1983) → 
  (∃ S : finset ℕ, S.card = k ∧ ∀ a b c ∈ S, a ≠ b → b ≠ c → c ≠ a → (b = (a + c).div 2) → false) :=
by
  intros h1 h2
  sorry

end distinct_pos_integers_no_AP_l622_622227


namespace woman_total_coins_l622_622070

theorem woman_total_coins
  (num_each_coin : ℕ)
  (h : 1 * num_each_coin + 5 * num_each_coin + 10 * num_each_coin + 25 * num_each_coin + 100 * num_each_coin = 351)
  : 5 * num_each_coin = 15 :=
by
  sorry

end woman_total_coins_l622_622070


namespace line_intersects_plane_inside_triangle_l622_622740

theorem line_intersects_plane_inside_triangle (A : Fin 5 → Point ℝ 3) 
  (h_independent: ∀ (p q r s : Fin 5), ¬Collinear ℝ (A '' {p, q, r, s})) :
  ∃ (i j k l m : Fin 5), 
    let P := {a | a ≠ i ∧ a ≠ j};
    ∀ (h : P ⊆ Fin 5), 
      ∃ (L : Line ℝ 3), 
        (A i ∈ L ∧ A j ∈ L) ∧
        (Plane ℝ 3).Contains (A k) (A l) (A m) ∧
        ∃ (T : Triangle ℝ 3), (T.verts = {A k, A l, A m}) ∧ L.IntersectsInside T :=
sorry

end line_intersects_plane_inside_triangle_l622_622740


namespace revive_ivan_tsarevich_l622_622327

/-- The main theorem states that given the following conditions:
- The Wolf and Ivan Tsarevich are 20 versts away from a source of living water.
- The Wolf's speed is 3 versts per hour.
- 1 liter of water is needed to revive Ivan Tsarevich.
- The source flows at a rate of half a liter per hour.
- A Raven at the source with unlimited carrying capacity collects the water.
- The Raven flies towards the Wolf and Ivan Tsarevich at a speed of 6 versts per hour.
- The Raven spills a quarter liter of water every hour.
Then, after 4 hours, it will be possible to revive Ivan Tsarevich. -/
theorem revive_ivan_tsarevich : 
  ∀ (d_wolf : ℝ) (v_wolf : ℝ) (v_water : ℝ) (d_raven : ℝ) (v_raven : ℝ) (spillage : ℝ) (needed_water : ℝ), 
  d_wolf = 20 → v_wolf = 3 → v_water = 1 / 2 → d_raven = 10 → v_raven = 6 → spillage = 1 / 4 → needed_water = 1 →
  (d_wolf / v_wolf + 1 / v_water + d_raven / v_raven) = 4 :=
begin
  intros,
  sorry
end

end revive_ivan_tsarevich_l622_622327


namespace range_of_m_l622_622550

noncomputable def f (x m : ℝ) : ℝ := x^3 + 3*x^2 - m*x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x ∈ set.Icc (-2 : ℝ) 2, deriv (λ x, f x m) x ≥ 0) → m ≤ -3 :=
sorry

end range_of_m_l622_622550


namespace two_digit_number_divisible_by_8_between_70_80_l622_622385

theorem two_digit_number_divisible_by_8_between_70_80 :
  ∃ n : ℕ, n ≥ 70 ∧ n < 80 ∧ n % 8 = 0 ∧ n = 72 := 
by {
  use 72,
  split,
  { -- 72 is greater than or equal to 70
    exact le_refl 72,    
  },
  split,
  { -- 72 is less than 80
    exact dec_trivial,
  },
  split,
  { -- 72 is divisible by 8
    exact dec_trivial,
  },
  { -- n = 72
    exact dec_trivial,
  }
}

end two_digit_number_divisible_by_8_between_70_80_l622_622385


namespace simplify_and_evaluate_l622_622691

noncomputable section

def x := Real.sqrt 3 + 1

theorem simplify_and_evaluate :
  (x / (x^2 - 1) / (1 - (1 / (x + 1)))) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l622_622691


namespace removed_term_sequence_l622_622529

theorem removed_term_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (k : ℕ) :
  (∀ n, S n = 2 * n^2 - n) →
  (∀ n, n ≥ 2 → a n = S n - S (n-1)) →
  (S 21 - a k = 40 * 20) →
  a k = 4 * k - 3 →
  k = 16 :=
by
  intros hs ha h_avg h_ak
  sorry

end removed_term_sequence_l622_622529


namespace apartment_number_l622_622796

variable (n : ℕ)

def apartments (f : ℕ) (per_floor : ℕ) : ℕ := f * per_floor

theorem apartment_number 
  (n : ℕ) 
  (initial_per_floor : ℕ) 
  (actual_per_floor : ℕ) 
  (fourth_floor : ℕ) : 
  (initial_per_floor = 6) ∧ 
  (fourth_floor = 4) ∧ 
  (actual_per_floor = 7) → 
  (19 ≤ n ∧ n ≤ 24) ∧ 
  (22 ≤ n ∧ n ≤ 28) → 
  n ∈ {22, 23, 24} :=
by
  intros h1 h2
  cases h1 with h_initial_per_floor h_floor
  cases h_floor with h_fourth_floor h_actual_per_floor
  cases h2
  split
  {
    sorry
  }


end apartment_number_l622_622796


namespace sum_alternating_series_l622_622384

theorem sum_alternating_series :
  (Finset.sum (Finset.range 1000.succ) (λ n => if n % 2 = 0 then -(n + 1).succ else n)) = -500 := by
  sorry

end sum_alternating_series_l622_622384


namespace noncongruent_triangles_count_l622_622671

def points_are_vertices_of_square : Prop 
  := ∀ {A B C D : Type}, is_square A B C D

def points_are_midpoints_of_sides_of_square : Prop 
  := ∀ {P Q R S A B C D : Type}, is_midpoint_of_sides P Q R S A B C D

theorem noncongruent_triangles_count :
  points_are_vertices_of_square →
  points_are_midpoints_of_sides_of_square →
  ∃ k : ℕ, k = 4 ∧
  (∀ x y z : Type, (x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = P ∨ x = Q ∨ x = R ∨ x = S) ∧ 
  (y = A ∨ y = B ∨ y = C ∨ y = D ∨ y = P ∨ y = Q ∨ y = R ∨ y = S) ∧ 
  (z = A ∨ z = B ∨ z = C ∨ z = D ∨ z = P ∨ z = Q ∨ z = R ∨ z = S) → 
  is_triangle x y z) :=
by {
  -- Proof goes here
  sorry
}

end noncongruent_triangles_count_l622_622671


namespace smallest_last_digit_l622_622714

def is_divisible (d : ℕ) (a b : ℕ) : Prop := (10 * a + b) % d = 0

def valid_two_digit (a b : ℕ) : Prop := is_divisible 17 a b ∨ is_divisible 23 a b

noncomputable def valid_string (s : string) : Prop :=
  s.length = 2023 ∧ s.front = '2' ∧ ∀ (i : ℕ), i < 2022 → valid_two_digit (s.to_nat i) (s.to_nat (i+1))

theorem smallest_last_digit (s : string) (h : valid_string s) : s.back.to_nat = 2 :=
sorry

end smallest_last_digit_l622_622714


namespace solve_fraction_equation_l622_622320

theorem solve_fraction_equation :
  ∀ x : ℝ, (3 / (2 * x - 2) + 1 / (1 - x) = 3) → x = 7 / 6 :=
by
  sorry

end solve_fraction_equation_l622_622320


namespace correct_operations_result_greater_than_1000_l622_622668

theorem correct_operations_result_greater_than_1000
    (finalResultIncorrectOps : ℕ)
    (originalNumber : ℕ)
    (finalResultCorrectOps : ℕ)
    (H1 : finalResultIncorrectOps = 40)
    (H2 : originalNumber = (finalResultIncorrectOps + 12) * 8)
    (H3 : finalResultCorrectOps = (originalNumber * 8) + (2 * originalNumber) + 12) :
  finalResultCorrectOps > 1000 := 
sorry

end correct_operations_result_greater_than_1000_l622_622668


namespace smallest_six_digit_divisible_by_111_l622_622029

theorem smallest_six_digit_divisible_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 :=
by {
  sorry
}

end smallest_six_digit_divisible_by_111_l622_622029


namespace abs_pi_pi_minus_ten_l622_622861

theorem abs_pi_pi_minus_ten : abs (π - abs (π - 10)) = 10 - 2 * π := 
by 
  sorry

end abs_pi_pi_minus_ten_l622_622861


namespace example_expression_equals_19_l622_622856

noncomputable def example_expression : ℝ :=
  (2 * real.sqrt 2) ^ (2 / 3) * (0.1) ^ (-1) - real.log 2 - real.log 5

theorem example_expression_equals_19 : example_expression = 19 := by
  sorry

end example_expression_equals_19_l622_622856


namespace find_t_l622_622981

theorem find_t (t : ℝ) (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :
  (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1)^(n-1)) → -- Geometric sequence condition
  (∀ n, S_n n = 2017 * 2016^n - 2018 * t) →     -- Given sum formula
  t = 2017 / 2018 :=
by
  sorry

end find_t_l622_622981


namespace find_a_b_l622_622564

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (a * log x) / (x + 1) + b / x

theorem find_a_b :
  ∃ a b : ℝ, (∀ x, f x a b = (log x) / (x + 1) + 1 / x) ∧ -- This ensures a = 1 and b = 1
    (f 1 a b = 1) ∧ ((f x a b - (log x) / (x - 1)) > 0) ∧ -- This defines the correct values of f(1)
    (∀ x : ℝ, 0 < x → x ≠ 1 → f x a b > (log x) / (x - 1)) -- This proves the inequality
  :=
by
  -- Lean proof goes here
  sorry

end find_a_b_l622_622564


namespace smallest_of_consecutive_even_numbers_l622_622595

theorem smallest_of_consecutive_even_numbers (n : ℤ) (h : ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ c = 2 * n + 1) :
  ∃ a b c : ℤ, a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ b = a + 2 ∧ c = a + 4 ∧ a = 2 * n - 3 :=
by
  sorry

end smallest_of_consecutive_even_numbers_l622_622595


namespace average_daily_wage_of_b_l622_622022

theorem average_daily_wage_of_b (A_work_days B_work_days total_payment : ℕ) (work_days_together : ℕ) (correct_wage : ℚ) 
  (hA : A_work_days = 12) (hB : B_work_days = 15) (h_work_days_together : work_days_together = 5) 
  (h_total_payment : total_payment = 810) (h_correct_wage : correct_wage = 121.50) : 
  let A_rate := 1 / (A_work_days : ℚ),
      B_rate := 1 / (B_work_days : ℚ),
      combined_rate := A_rate + B_rate,
      work_done_together := combined_rate * work_days_together,
      remaining_work := 1 - work_done_together,
      payment_for_b := (combined_rate * work_days_together) * total_payment / 2,
      average_daily_wage := payment_for_b / work_days_together in
  average_daily_wage = correct_wage :=
by 
  sorry

end average_daily_wage_of_b_l622_622022


namespace min_pairs_condition_l622_622041

noncomputable def min_pairs_of_acquaintances : ℕ :=
  let n := 175 in
  (n * (n - 3))/2

theorem min_pairs_condition
    (n : ℕ)
    (condition : ∀ s : Finset (Fin n), s.card = 6 → ∃ a b c d e f : Fin n, 
      s = {a, b, c, d, e, f} ∧ ((a = b ∧ a = c ∧ b = c) ∨ (d = e ∧ d = f ∧ e = f))
    ) : min_pairs_of_acquaintances = 15050 := 
  sorry

end min_pairs_condition_l622_622041


namespace parabola_eq_find_min_t_line_BD_fixed_point_l622_622525

-- Given Conditions
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1
def right_focus_of_ellipse : ℝ × ℝ := (1, 0)
def parabola_focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Questions rewritten in Lean
theorem parabola_eq (p : ℝ) (h₁ : p = 2) :
    ∃ x y : ℝ, y^2 = 4 * x :=
sorry

theorem find_min_t (a : ℝ) :
    a = 2 → t = -2 :=
sorry

theorem line_BD_fixed_point (x₁ y₁ y₂ x₂ : ℝ) (a : ℝ) (h₁ : a = -1) (h₂ : y₁ + y₂ = 4)
    (h₃ : y₁ * y₂ = 4) :
    ∀ (y : ℝ) (x : ℝ), y - y₂ = (4 / (y₂ - y₁)) * (x - x₂) → (1, 0) :=
sorry

end parabola_eq_find_min_t_line_BD_fixed_point_l622_622525


namespace money_spent_correct_l622_622106

-- Define conditions
def spring_income : ℕ := 2
def summer_income : ℕ := 27
def amount_after_supplies : ℕ := 24

-- Define the resulting money spent on supplies
def money_spent_on_supplies : ℕ :=
  (spring_income + summer_income) - amount_after_supplies

theorem money_spent_correct :
  money_spent_on_supplies = 5 := by
  sorry

end money_spent_correct_l622_622106


namespace no_solution_iff_range_of_a_l622_622554

theorem no_solution_iff_range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬(x^2 + a * x + 4 < 0)) ↔ a ∈ I'm sorry

end no_solution_iff_range_of_a_l622_622554


namespace line_AB_does_not_pass_B_l622_622539

-- Definitions and contexts
theorem line_AB_does_not_pass_B :
  ∀ (a b : ℝ), (a - 1)^2 + b^2 = 4 → 
  ¬ (∃ (x y : ℝ), (x, y) = (1 / 2, 1 / 2) ∧ (2 * a - 2) * x + 2 * b * y - 3 = 0) :=
by
  intros a b h
  by_contradiction
  rcases h_1 with ⟨x, y, hc, he⟩
  sorry

end line_AB_does_not_pass_B_l622_622539


namespace probability_of_even_product_l622_622748

open BigOperators

def total_combinations : Nat := nat.choose 6 3

def odd_combinations : Nat := nat.choose 3 3

def probability_odd_product : ℚ := odd_combinations / total_combinations

def probability_even_product : ℚ := 1 - probability_odd_product

theorem probability_of_even_product (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6}) :
  probability_even_product = 19 / 20 := by
  sorry

end probability_of_even_product_l622_622748


namespace Nellie_needs_to_sell_more_rolls_l622_622927

-- Define the conditions
def total_needed : ℕ := 45
def sold_grandmother : ℕ := 1
def sold_uncle : ℕ := 10
def sold_neighbor : ℕ := 6

-- Define the total sold
def total_sold : ℕ := sold_grandmother + sold_uncle + sold_neighbor

-- Define the remaining rolls needed
def remaining_rolls := total_needed - total_sold

-- Statement to prove that remaining_rolls equals 28
theorem Nellie_needs_to_sell_more_rolls : remaining_rolls = 28 := by
  unfold remaining_rolls
  unfold total_sold
  unfold total_needed sold_grandmother sold_uncle sold_neighbor
  calc
  45 - (1 + 10 + 6) = 45 - 17 : by rw [Nat.add_assoc]
  ... = 28 : by norm_num

end Nellie_needs_to_sell_more_rolls_l622_622927


namespace triangle_area_l622_622567

-- Define the function and its conditions
def f (x : ℝ) : ℝ := √3 * sin (ω * x) - 2 * sin(ω * x / 2) ^ 2

-- Given conditions for ω and period
axiom omega_positive : ω > 0
axiom omega_period : (2 * π) / ω = 3 * π

-- Define the maximum and minimum interval for f
axiom interval_min : f (-3 * π / 4) = -√3 - 1
axiom interval_max : f (π / 2) = 1

-- Define the side lengths opposite angles A, B, C of triangle ABC, and given conditions
variables {a b c : ℝ} {A B C : ℝ}
axiom side_b : b = 2
axiom f_A : f A = √3 - 1
axiom side_relation : √3 * a = 2 * b * sin A

-- Prove that the area of triangle ABC is (3 + √3) / 3
theorem triangle_area :
  let S := (1 / 2) * a * b * sin C
  in S = (3 + √3) / 3 := sorry

end triangle_area_l622_622567


namespace total_cubes_l622_622097

noncomputable def original_cubes : ℕ := 2
noncomputable def additional_cubes : ℕ := 7

theorem total_cubes : original_cubes + additional_cubes = 9 := by
  sorry

end total_cubes_l622_622097


namespace olivia_new_premium_l622_622665

theorem olivia_new_premium :
  let initial_premium := 50
  let accident_increase := 0.10 * initial_premium
  let ticket_increase := 5
  let total_accidents := 1
  let total_tickets := 3
  let new_premium := initial_premium + accident_increase + total_tickets * ticket_increase
  new_premium = 70 :=
by
  let initial_premium := 50
  let accident_increase := 0.10 * initial_premium
  let ticket_increase := 5
  let total_accidents := 1
  let total_tickets := 3
  let new_premium := initial_premium + accident_increase + total_tickets * ticket_increase
  sorry

end olivia_new_premium_l622_622665


namespace domain_of_sqrt_fraction_l622_622482

theorem domain_of_sqrt_fraction {x : ℝ} (h1 : x - 3 ≥ 0) (h2 : 7 - x > 0) :
  3 ≤ x ∧ x < 7 :=
by {
  sorry
}

end domain_of_sqrt_fraction_l622_622482


namespace area_of_triangle_PQZ_l622_622220

theorem area_of_triangle_PQZ (PQ RS QR: ℝ) (PQ_len: PQ = 7) (QR_len: QR = 4) 
    (RX SY: ℝ) (RX_len: RX = 2) (SY_len: SY = 3)
    (PX QY: line) (PX_intersect_QY_at: intersect PX QY = Z) :
    area_of_triangle PQ Z Q = 19.6 :=
by
  sorry

end area_of_triangle_PQZ_l622_622220


namespace eggs_left_over_l622_622636

theorem eggs_left_over (john_eggs : ℕ) (maria_eggs : ℕ) (nikhil_eggs : ℕ) (carton_size : ℕ) :
  john_eggs = 45 → maria_eggs = 38 → nikhil_eggs = 29 → carton_size = 10 → 
  (john_eggs + maria_eggs + nikhil_eggs) % carton_size = 2 :=
by
  intros h_john h_maria h_nikhil h_carton
  rw [h_john, h_maria, h_nikhil, h_carton]
  calc
    (45 + 38 + 29) % 10 = 112 % 10 : by sorry
                      ... = 2 : by sorry

end eggs_left_over_l622_622636


namespace total_dogs_in_kennel_l622_622850

-- Definition of the given conditions
def T := 45       -- Number of dogs that wear tags
def C := 40       -- Number of dogs that wear flea collars
def B := 6        -- Number of dogs that wear both tags and collars
def D_neither := 1 -- Number of dogs that wear neither a collar nor tags

-- Theorem statement
theorem total_dogs_in_kennel : T + C - B + D_neither = 80 := 
by
  -- Proof omitted
  sorry

end total_dogs_in_kennel_l622_622850


namespace rate_per_sq_meter_correct_l622_622343

-- Define the conditions
def length_of_room : ℝ := 5.5
def width_of_room : ℝ := 3.75
def total_cost : ℝ := 28875

-- Define the total area
def area_of_floor : ℝ := length_of_room * width_of_room

-- Define the rate per square meter
def rate_per_sq_meter : ℝ := total_cost / area_of_floor

-- The statement to be proved
theorem rate_per_sq_meter_correct :
  rate_per_sq_meter = 1400 := by
  sorry

end rate_per_sq_meter_correct_l622_622343


namespace base8_to_base10_12345_l622_622391

theorem base8_to_base10_12345 : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 := by
  sorry

end base8_to_base10_12345_l622_622391


namespace simplify_expression_l622_622300

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622300


namespace simplify_expression_l622_622299

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622299


namespace sequence_general_term_l622_622520

open Int

noncomputable def x : ℕ → ℤ
| 1     := 1
| 2     := 6
| (n+3) := 6 * x (n+2) - 9 * x (n+1) + 3^n

theorem sequence_general_term :
  ∀ n, x (n+1) = (3^(n-1) / 2) * (n^2 - n + 2) := 
sorry

end sequence_general_term_l622_622520


namespace steak_weight_correct_l622_622364

-- We define the number of family members and their steak requirement.
def family_members : ℕ := 5
def steak_per_member : ℝ := 1

-- We define the total amount of steak needed.
def total_steak_needed : ℝ := family_members * steak_per_member

-- We define the number of steaks Tommy buys.
def steaks_bought : ℕ := 4

-- We then define the weight of each steak.
def weight_per_steak : ℝ := total_steak_needed / steaks_bought

-- Finally, we state the theorem that asserts the weight of each steak.
theorem steak_weight_correct : weight_per_steak = 1.25 := by
  sorry

end steak_weight_correct_l622_622364


namespace sequence_stabilizes_l622_622240

def P (n : ℕ) : ℕ :=
  n.digits.map (λ d, d.to_nat).prod

def seq (n1 : ℕ) : ℕ → ℕ
| 0     := n1
| (k+1) := seq k + P (seq k)

theorem sequence_stabilizes (n1 : ℕ) : ∃ p : ℕ, ∀ k : ℕ, k ≥ p → seq n1 k = seq n1 (k+1) := 
sorry

end sequence_stabilizes_l622_622240


namespace product_of_divisors_eq_l622_622348

theorem product_of_divisors_eq : ∃ k : ℕ, (∀ d ∈ (finset.Ico 1 (6^16 + 1)).filter (λ d, (6^16) % d = 0), ∃ m n : ℕ, 0 ≤ m ∧ m ≤ 16 ∧ 0 ≤ n ∧ n ≤ 16 ∧ d = (2^m * 3^n)) ∧ 6 ^ k = (finset.Ico 1 (6^16 + 1)).filter (λ d, (6^16) % d = 0).prod := sorry

end product_of_divisors_eq_l622_622348


namespace sufficient_condition_for_negation_l622_622040

theorem sufficient_condition_for_negation {A B : Prop} (h : B → A) : ¬ A → ¬ B :=
by
  intro hA
  intro hB
  apply hA
  exact h hB

end sufficient_condition_for_negation_l622_622040


namespace distance_AC_l622_622105

-- Variable definitions based on given conditions
variables (time_Eddy : ℝ) (time_Freddy : ℝ)
variables (distance_AB : ℝ) (speed_ratio : ℝ)
variables (speed_Eddy : ℝ) (speed_Freddy : ℝ)

-- Given conditions
def eddy_travel_time : Prop := time_Eddy = 3
def freddy_travel_time : Prop := time_Freddy = 4
def distance_AB_value : Prop := distance_AB = 450
def speed_ratio_condition : Prop := speed_ratio = 2
def speed_Eddy_def : Prop := speed_Eddy = distance_AB / time_Eddy
def speed_Freddy_def : Prop := speed_Freddy = speed_Eddy / speed_ratio

-- Prove that the distance between city A and city C is 300 km
theorem distance_AC (h1 : eddy_travel_time) (h2 : freddy_travel_time) 
  (h3 : distance_AB_value) (h4 : speed_ratio_condition) 
  (h5 : speed_Eddy_def) (h6 : speed_Freddy_def) : 
  let distance_AC := speed_Freddy * time_Freddy in distance_AC = 300 :=
by
  sorry

end distance_AC_l622_622105


namespace vertices_of_square_l622_622667

-- Define lattice points as points with integer coordinates
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Define the distance between two lattice points
def distance (P Q : LatticePoint) : ℤ :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y)

-- Define the area of a triangle formed by three lattice points using the determinant method
def area (P Q R : LatticePoint) : ℤ :=
  (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x)

-- Prove that three distinct lattice points form the vertices of a square given the condition
theorem vertices_of_square (P Q R : LatticePoint) (h₀ : P ≠ Q) (h₁ : Q ≠ R) (h₂ : P ≠ R)
    (h₃ : (distance P Q + distance Q R) < 8 * (area P Q R) + 1) :
    ∃ S : LatticePoint, S ≠ P ∧ S ≠ Q ∧ S ≠ R ∧
    (distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P) := 
by sorry

end vertices_of_square_l622_622667


namespace total_sandwiches_l622_622446

theorem total_sandwiches (billy : ℕ) (katelyn_more : ℕ) (katelyn_quarter : ℕ) :
  billy = 49 → katelyn_more = 47 → katelyn_quarter = 4 → 
  billy + (billy + katelyn_more) + ((billy + katelyn_more) / katelyn_quarter) = 169 :=
by
  intros hb hk hq
  rw [hb, hk, hq]
  calc
    49 + (49 + 47) + ((49 + 47) / 4) = 49 + 96 + 24 : by simp
                                ... = 169 : by simp

sorry

end total_sandwiches_l622_622446


namespace max_goods_purchased_l622_622423

theorem max_goods_purchased (initial_spend : ℕ) (reward_rate : ℕ → ℕ → ℕ) (continuous_reward : Prop) :
  initial_spend = 7020 →
  (∀ x y, reward_rate x y = (x / y) * 20) →
  continuous_reward →
  initial_spend + reward_rate initial_spend 100 + reward_rate (reward_rate initial_spend 100) 100 + 
  reward_rate (reward_rate (reward_rate initial_spend 100) 100) 100 = 8760 :=
by
  intros h1 h2 h3
  sorry

end max_goods_purchased_l622_622423


namespace scatter_plot_variables_position_l622_622013

-- Definitions based on conditions
def is_explanatory_variable (v : Type) : Prop := v = "independent variable"
def is_forecast_variable (v : Type) : Prop := v = "dependent variable"

-- Proving the correct positioning on the x and y axis
theorem scatter_plot_variables_position (x y : Type) 
  (h1 : is_explanatory_variable x) 
  (h2 : is_forecast_variable y) : 
  x = "independent variable" ∧ y = "dependent variable" ∧ x = "explanatory variable" ∧ y = "forecast variable" :=
by 
  sorry

end scatter_plot_variables_position_l622_622013


namespace total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l622_622060

theorem total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges :
  ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ a * b * c = 1001 ∧ 2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end total_surface_area_of_rectangular_solid_with_given_volume_and_prime_edges_l622_622060


namespace line_passing_through_P_condition_l622_622051

noncomputable def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem line_passing_through_P_condition (P A B : (ℝ × ℝ)) :
  (P = (3, -1)) → (A = (2, -3)) → (B = (-4, 5)) → 
  (∃ (l : ℝ → ℝ → Prop), 
    l(P.1, P.2) ∧ 
    (l(A.1, A.2) = l(B.1, B.2))) →
  ( (∃ k b, ∀ x y, l x y ↔ y = k * x + b) ∧ 
    (∃ (eq1 eq2 : Prop),
    eq1 = (4 * x + 3 * y - 9 = 0) ∧ 
    eq2 = (x + 2 * y - 1 = 0) ∧ 
    (l 3 (-1) → eq1) ∨ (l 3 (-1) → eq2) )) 
: sorry

end line_passing_through_P_condition_l622_622051


namespace satellite_forecast_probability_l622_622687

theorem satellite_forecast_probability (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.75) :
  1 - ((1 - pA) * (1 - pB)) = 0.95 :=
by
  rw [hA, hB]
  norm_num
  sorry

end satellite_forecast_probability_l622_622687


namespace false_propositions_l622_622435

-- Definitions of the given conditions as propositions
def prop1 : Prop := ∀ (b a x̄ ȳ : ℝ), 
  (∃ (x y : ℝ), (y = b * x + a)) → 
  (ȳ = b * x̄ + a)

def prop2 : Prop := ∀ (x : ℝ), 
  ∃ (y₁ y₂ : ℝ), 
  (y₁ = 3 - 5 * x ∧ y₂ = 3 - 5 * (x + 1)) → 
  (y₂ = y₁ - 5)

def prop3 : Prop := ∀ (R1 R2 : ℝ),
  (R1 = 0.80 ∧ R2 = 0.98) → 
  R2 > R1

def prop4 : Prop := ∀ (x : ℝ), 
  ∃ (y : ℝ), 
  (x = 2 ∧ y = 0.5 * x - 8) → 
  y ≠ -7

-- Statement of the problem given the conditions
theorem false_propositions :
  ∃ (n : ℕ), 
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4) → 
  n = 3 :=
sorry

end false_propositions_l622_622435


namespace difference_of_circumradii_equals_distance_l622_622413

-- Definitions of the geometric entities involved
variables {A B C D : Type} [EuclideanGeometry A B C D]

-- Assume we have a triangle ABC with circumcenter O and incenter I
variables {O I : Point} (ABC : Triangle A B C)

-- Let the line through A intersect BC at D and be perpendicular to IO
variables (h_perpendicular : ∀ l, isPerpendicular (lineThrough A D) (lineThrough I O))

-- Define the circumscribed circles
variables (R_ABD R_ACD : Real)

-- Define the distance between I and O
variables (d_IO : Real)

-- Given conditions
axiom h_RABD : R_ABD = circumradius (Triangle A B D)
axiom h_RACD : R_ACD = circumradius (Triangle A C D)
axiom h_distance_IO : d_IO = distance I O

-- The proof statement
theorem difference_of_circumradii_equals_distance :
  (R_ABD - R_ACD = d_IO) :=
sorry

end difference_of_circumradii_equals_distance_l622_622413


namespace tangent_k_value_k_range_l622_622568

-- Define the functions f and g
def f (x : ℝ) : ℝ := Real.exp(2 * x)
def g (k x : ℝ) : ℝ := k * x + 1

-- Question 1: If the line y = g(x) is tangent to y = f(x), prove that k = 2
theorem tangent_k_value (k : ℝ) :
  (∃ t : ℝ, f t = g k t ∧ deriv f t = k) → k = 2 := sorry

-- Question 2: When k > 0, if there exists a positive real number m such that
-- for any x in (0, m), |f(x) - g(x)| > 2x, prove that k is in the range (4, +∞)
theorem k_range (k : ℝ) (hk : 0 < k) :
  (∃ m : ℝ, 0 < m ∧ ∀ x : ℝ, 0 < x ∧ x < m → |f x - g k x| > 2 * x) → 4 < k := sorry

end tangent_k_value_k_range_l622_622568


namespace pushkin_pension_is_survivors_pension_l622_622386

theorem pushkin_pension_is_survivors_pension
  (died_pushkin : Nat = 1837)
  (lifelong_pension_assigned : ∀ t : Nat, t > died_pushkin → ∃ (recipient : String), recipient = "Pushkin's wife" ∨ recipient = "Pushkin's daughter") :
  ∃ (pension_type : String), pension_type = "survivor's pension" :=
by
  sorry

end pushkin_pension_is_survivors_pension_l622_622386


namespace relationship_y1_y2_y3_l622_622958

theorem relationship_y1_y2_y3 :
  let y1 := -(((-4):ℝ)^2) + 5 in
  let y2 := -(((-1):ℝ)^2) + 5 in
  let y3 := -((2:ℝ)^2) + 5 in
  y2 > y3 ∧ y3 > y1 :=
by
  sorry

end relationship_y1_y2_y3_l622_622958


namespace sausage_length_l622_622437

theorem sausage_length (pieces : ℕ) (length_of_one_piece : ℚ) 
  (hp : pieces = 12) (hl : length_of_one_piece = 2/3) : 
  pieces * length_of_one_piece = 8 := 
by 
  rw [hp, hl]
  norm_num
  sorry

end sausage_length_l622_622437


namespace probability_of_Q_within_two_units_of_origin_l622_622420

noncomputable def probability_Q_within_two_units_of_origin : ℝ :=
  let area_circle := 4 * Real.pi in
  let area_square := 64 in
  area_circle / area_square

theorem probability_of_Q_within_two_units_of_origin :
  probability_Q_within_two_units_of_origin = Real.pi / 16 :=
by
  sorry

end probability_of_Q_within_two_units_of_origin_l622_622420


namespace find_m_n_sum_of_coeffs_l622_622569

-- Step 1: Translate condition definitions
def polynomial (x y : ℝ) (m : ℕ) := -8 * x^3 * y^(m + 1) + x * y^2 - (3 / 4) * x^3 + 6 * y
def monomial (x y : ℝ) (n m : ℕ) := (2 / 5) * Real.pi * x^n * y^(5 - m)

-- Step 2: Translate proof problems
theorem find_m_n (x y : ℝ) (m n : ℕ) :
  (∀ m, polynomial x y m = -8 * x^3 * y^(m + 1) + x * y^2 - (3 / 4) * x^3 + 6 * y) →
  (∀ n, monomial x y n m = (2 / 5) * Real.pi * x^n * y^(5 - m)) →
  (3 + (m + 1) = 6) →
  (n + (5 - m) = 6) →
  m = 2 ∧ n = 3 := sorry

theorem sum_of_coeffs (x y : ℝ) (m : ℕ) :
  (∀ m, polynomial x y m = -8 * x^3 * y^(m + 1) + x * y^2 - (3 / 4) * x^3 + 6 * y) →
  m = 2 →
  (-8 + 1 - (3 / 4) + 6) = -11 / 4 := sorry

end find_m_n_sum_of_coeffs_l622_622569


namespace joe_eat_at_least_two_kinds_of_fruit_l622_622444

theorem joe_eat_at_least_two_kinds_of_fruit :
  -- Define that Joe randomly chooses fruits with equal probability at each meal
  ∀ (p : ℕ), p ∈ {0, 1, 2} →
  -- Define the probability that Joe eats at least two different kinds of fruit in a day
  (1 - 3 * (1/3)^3) = 8/9 := 
by
  sorry

end joe_eat_at_least_two_kinds_of_fruit_l622_622444


namespace company_profit_per_tire_l622_622023

theorem company_profit_per_tire 
  (fixed_cost_per_batch : ℕ)
  (variable_cost_per_tire : ℕ)
  (selling_price_per_tire : ℕ)
  (batch_size : ℕ)
  (total_cost : ℕ)
  (total_revenue : ℕ)
  (profit_from_batch : ℕ)
  (profit_per_tire : ℕ) :
  fixed_cost_per_batch = 22500 → 
  variable_cost_per_tire = 8 → 
  selling_price_per_tire = 20 → 
  batch_size = 15000 → 
  total_cost = fixed_cost_per_batch + (variable_cost_per_tire * batch_size) →
  total_revenue = selling_price_per_tire * batch_size →
  profit_from_batch = total_revenue - total_cost →
  profit_per_tire = profit_from_batch / batch_size →
  profit_per_tire = 1050 / 100 := 
by
  intros,
  sorry

end company_profit_per_tire_l622_622023


namespace game_winner_a_l622_622075

theorem game_winner_a (m n : ℕ) (h_m : m = 2^40) (h_n : n = 3^51) : "Alphonse wins" :=
sorry

end game_winner_a_l622_622075


namespace solution_set_inequality_l622_622553

theorem solution_set_inequality (a b x : ℝ) (h₀ : {x : ℝ | ax - b < 0} = {x : ℝ | 1 < x}) :
  {x : ℝ | (ax + b) * (x - 3) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
by
  sorry

end solution_set_inequality_l622_622553


namespace simplify_fraction_l622_622313

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l622_622313


namespace thief_run_distance_l622_622835

noncomputable def thief_distance_before_overtake
  (initial_distance : ℝ) 
  (thief_speed_kmh : ℝ) 
  (policeman_speed_kmh : ℝ) : ℝ :=
  let thief_speed_ms := thief_speed_kmh * 1000 / 3600 in
  let policeman_speed_ms := policeman_speed_kmh * 1000 / 3600 in
  let relative_speed := policeman_speed_ms - thief_speed_ms in
  let time_to_overtake := initial_distance / relative_speed in
  thief_speed_ms * time_to_overtake

theorem thief_run_distance 
  (initial_distance : ℝ := 300) 
  (thief_speed_kmh : ℝ := 14) 
  (policeman_speed_kmh : ℝ := 18): 
  thief_distance_before_overtake initial_distance thief_speed_kmh policeman_speed_kmh = 1050 :=
by
  sorry

end thief_run_distance_l622_622835


namespace MN_perpendicular_A1B_and_D1B1_l622_622409

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vec (A B : Point3D) : Point3D :=
  { x := B.x - A.x, y := B.y - A.y, z := B.z - A.z }

def dot_product (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

def A := {x := 1, y := 0, z := 0}
def B := {x := 1, y := 1, z := 0}
def C := {x := 0, y := 1, z := 0}
def D := {x := 0, y := 0, z := 0}
def D1 := {x := 0, y := 0, z := 1}
def A1 := {x := 1, y := 0, z := 1}
def B1 := {x := 1, y := 1, z := 1}
def C1 := {x := 0, y := 1, z := 1}

def M : Point3D := 
  { x := 1, y := 1/3, z := 2/3 }

def N : Point3D := 
  { x := 2/3, y := 2/3, z := 1 }

def A1B := vec A1 B
def D1B1 := vec D1 B1
def MN := vec M N

theorem MN_perpendicular_A1B_and_D1B1 : 
  dot_product MN A1B = 0 ∧ dot_product MN D1B1 = 0 := 
by
  sorry

end MN_perpendicular_A1B_and_D1B1_l622_622409


namespace minimum_tanA_9tanB_l622_622983

variable (a b c A B : ℝ)
variable (Aacute : A > 0 ∧ A < π / 2)
variable (h1 : a^2 = b^2 + 2*b*c * Real.sin A)
variable (habc : a = b * Real.sin A)

theorem minimum_tanA_9tanB : 
  ∃ (A B : ℝ), (A > 0 ∧ A < π / 2) ∧ (a^2 = b^2 + 2*b*c * Real.sin A) ∧ (a = b * Real.sin A) ∧ 
  (min ((Real.tan A) - 9*(Real.tan B)) = -2) := 
  sorry

end minimum_tanA_9tanB_l622_622983


namespace infinite_unrepresentable_numbers_l622_622879

theorem infinite_unrepresentable_numbers :
  ∃ (N0 : ℕ), ∀ N ≥ N0, ∃ M < N, ∀ a b c d e : ℕ,
    1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧ 1 ≤ e → a^3 + b^5 + c^7 + d^9 + e^{11} ≠ M :=
by sorry

end infinite_unrepresentable_numbers_l622_622879


namespace largest_birthday_group_l622_622738

theorem largest_birthday_group (people : ℕ) (months : ℕ) (h1 : people = 52) (h2 : months = 12) :
  ∃ n, n = 5 ∧ ∀ d : fin 12 → set (fin 52), ∃ m, (d d.m ∈ (finset.range 52).sum (λ i, %) m.n> n :=
begin
  have n := nat.ceil (people / months),
  refine ⟨n, _, _⟩,
  { sorry },
  { sorry }
end

end largest_birthday_group_l622_622738


namespace greatest_distance_is_9sqrt2_div_2_l622_622224

noncomputable def greatest_distance_between_C_and_D : ℂ :=
  let C : Set ℂ := {z | z^3 - 27 = 0}
  let D : Set ℂ := {z | z^3 - 9*z^2 + 27*z - 27 = 0}
  let distance := λ z1 z2 : ℂ, Complex.abs (z1 - z2)
  sorry

theorem greatest_distance_is_9sqrt2_div_2 :
  greatest_distance_between_C_and_D = (9 * Real.sqrt 2) / 2 :=
by
  sorry

end greatest_distance_is_9sqrt2_div_2_l622_622224


namespace magnitude_of_angle_C_value_of_side_length_c_l622_622204

theorem magnitude_of_angle_C (a c : ℝ) (sinA cosC : ℝ) (h₁: ∀ a c sinA cosC, a / sinA = c / (sqrt 3 * cosC)) :
  ∃ (C : ℝ), C = π / 3 :=
by
  sorry

theorem value_of_side_length_c (a b c : ℝ) (h₁: a + b = 6) (h₂: a * b * (1/2) = 4) :
  c = 2 * sqrt 3 :=
by
  sorry

end magnitude_of_angle_C_value_of_side_length_c_l622_622204


namespace part_I_part_II_l622_622325

-- Define the function f(x)
def f (x a : ℝ) := abs (x - a) + 5 * x

-- Part (I)
theorem part_I (x : ℝ) : 
  (f x (-1) ≤ 5 * x + 3) ↔ (-4 ≤ x ∧ x ≤ 2) := 
by sorry

-- Part (II)
theorem part_II (a : ℝ) (x : ℝ) (h : x ≥ -1) : 
  (∀ x, f x a ≥ 0) ↔ (a ≥ 4 ∨ a ≤ -6) := 
by sorry

end part_I_part_II_l622_622325


namespace arccos_cos_10_l622_622089

theorem arccos_cos_10 : arccos (cos 10) = 3.717 :=
sorry

end arccos_cos_10_l622_622089


namespace N_is_midpoint_CD_l622_622625

noncomputable def midpoint (A B : Point) : Point :=
  -- Definition of midpoint of segment AB
  (1/2)*(A + B)

variable (A B C D N M : Point)
variable (h1 : is_trapezoid A B C D) -- ABCD is a trapezoid with AB || CD
variable (h2 : M = midpoint A B)
variable (h3 : on_segment N C D)
variable (h4 : ∠ADN = 1/2 * ∠MNC)
variable (h5 : ∠BCN = 1/2 * ∠MND)

open_locale real_angle

theorem N_is_midpoint_CD :
  is_midpoint N C D :=
sorry

end N_is_midpoint_CD_l622_622625


namespace simplify_expr_l622_622305

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l622_622305


namespace min_value_inverse_sum_l622_622654

theorem min_value_inverse_sum
  (b : Fin 8 → ℝ)
  (h_pos : ∀ i, 0 < b i)
  (h_sum : (∑ i, b i) = 2) :
  (∑ i, b i⁻¹) ≥ 32 := sorry

end min_value_inverse_sum_l622_622654


namespace evaluate_expression_l622_622489

variables (a b c : ℝ)

theorem evaluate_expression (h1 : c = b - 20) (h2 : b = a + 4) (h3 : a = 2)
  (h4 : a^2 + a ≠ 0) (h5 : b^2 - 6 * b + 8 ≠ 0) (h6 : c^2 + 12 * c + 36 ≠ 0):
  (a^2 + 2 * a) / (a^2 + a) * (b^2 - 4) / (b^2 - 6 * b + 8) * (c^2 + 16 * c + 64) / (c^2 + 12 * c + 36) = 3 / 4 :=
by sorry

end evaluate_expression_l622_622489


namespace parabola_intersects_xaxis_at_least_one_l622_622969

theorem parabola_intersects_xaxis_at_least_one {a b c : ℝ} (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0) ∧ (a * x2^2 + 2 * b * x2 + c = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (b * x1^2 + 2 * c * x1 + a = 0) ∧ (b * x2^2 + 2 * c * x2 + a = 0)) ∨
  (∃ x1 x2, x1 ≠ x2 ∧ (c * x1^2 + 2 * a * x1 + b = 0) ∧ (c * x2^2 + 2 * a * x2 + b = 0)) :=
by
  sorry

end parabola_intersects_xaxis_at_least_one_l622_622969


namespace paint_proof_l622_622630

/-- 
Suppose Jack's room has 27 square meters of wall and ceiling area. He has three choices for paint:
- Using 1 can of paint leaves 1 liter of paint left over,
- Using 5 gallons of paint leaves 1 liter of paint left over,
- Using 4 gallons and 2.8 liters of paint.

1. Prove: The ratio between the volume of a can and the volume of a gallon is 1:5.
2. Prove: The volume of a gallon is 3.8 liters.
3. Prove: The paint's coverage is 1.5 square meters per liter.
-/
theorem paint_proof (A : ℝ) (C G : ℝ) (R : ℝ):
  ∀ (H1: A = 27) (H2: C - 1 = 27) (H3: 5 * G - 1 = 27) (H4: 4 * G + 2.8 = 27), 
  (C / G = 1 / 5) ∧ (G = 3.8) ∧ ((A / (5 * G - 1)) = 1.5) :=
by
  sorry

end paint_proof_l622_622630


namespace side_length_of_ABC_l622_622818

noncomputable def proof : ℝ → Prop :=
  λ a : ℝ,
  let α := Real.arctan (3/4)
  let h := (a * Real.sqrt 3) / 8
  let S_MNK := (a^2 * Real.sqrt 3) / 16
  let S_FPR := (a^2 * Real.sqrt 3) / 64
  let S_MPN := (3 * a * h) / 16 
  let S_FPN := (15 * a^2 * Real.sqrt 3) / 256
  in let total_area := S_MNK + S_FPR + 3 * S_MPN + 3 * S_FPN
  in total_area = 53 * Real.sqrt 3 → a = 16

theorem side_length_of_ABC : proof 16 := sorry

end side_length_of_ABC_l622_622818


namespace range_of_a_l622_622976

-- Definition of logarithm base function in Lean
noncomputable def log_base {a x : ℝ} (hx: x = 2) : ℝ := Real.log x / Real.log a

-- Definition of the function y = log_2(ax - 1) in the interval (1, 2)
def is_monotonically_increasing_in_interval (a : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, 1 < x1 → x1 < 2 → 1 < x2 → x2 < 2 → log_base 2 (a * x1 - 1) ≤ log_base 2 (a * x2 - 1)

-- Theorem stating the range of values for a
theorem range_of_a (a : ℝ) : is_monotonically_increasing_in_interval a ↔ 1 ≤ a := 
by
  sorry

end range_of_a_l622_622976


namespace relationship_y1_y2_y3_l622_622962

-- Conditions
def y1 := -((-4)^2) + 5
def y2 := -((-1)^2) + 5
def y3 := -(2^2) + 5

-- Statement to be proved
theorem relationship_y1_y2_y3 : y2 > y3 ∧ y3 > y1 := by
  sorry

end relationship_y1_y2_y3_l622_622962


namespace relationship_abc_l622_622541

noncomputable def a : ℝ := Real.cos (17 * Real.pi / 180) * Real.cos (23 * Real.pi / 180) - Real.sin (17 * Real.pi / 180) * Real.sin (23 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.cos (25 * Real.pi / 180)^2 - 1
def c : ℝ := Real.sqrt 3 / 2

theorem relationship_abc : b < a ∧ a < c := by
  sorry

end relationship_abc_l622_622541


namespace min_value_expr_l622_622154

-- Given conditions
variables {a b c : ℝ}
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a + b = 1)
variable (hc : c > 1)

-- Statement to prove
theorem min_value_expr (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) (hc : c > 1) :
  ( ((a^2 + 1) / (2 * a * b) - 1) * c + (real.sqrt 2) / (c - 1) ) ≥ 3 * real.sqrt 2 :=
sorry

end min_value_expr_l622_622154


namespace alice_arrives_earlier_l622_622433

/-
Alice and Bob are heading to a park that is 2 miles away from their home. 
They leave home at the same time. 
Alice cycles to the park at a speed of 12 miles per hour, 
while Bob jogs there at a speed of 6 miles per hour. 
Prove that Alice arrives 10 minutes earlier at the park than Bob.
-/

theorem alice_arrives_earlier 
  (d : ℕ) (a_speed : ℕ) (b_speed : ℕ) (arrival_difference_minutes : ℕ) 
  (h1 : d = 2) 
  (h2 : a_speed = 12) 
  (h3 : b_speed = 6) 
  (h4 : arrival_difference_minutes = 10) 
  : (d / a_speed * 60) + arrival_difference_minutes = d / b_speed * 60 :=
by
  sorry

end alice_arrives_earlier_l622_622433


namespace tenth_number_drawn_eq_195_l622_622425

noncomputable def total_students : Nat := 1000
noncomputable def sample_size : Nat := 50
noncomputable def first_selected_number : Nat := 15  -- Note: 0015 is 15 in natural number

theorem tenth_number_drawn_eq_195 
  (h1 : total_students = 1000)
  (h2 : sample_size = 50)
  (h3 : first_selected_number = 15) :
  15 + (20 * 9) = 195 := 
by
  sorry

end tenth_number_drawn_eq_195_l622_622425


namespace gcd_pow_minus_one_l622_622512

theorem gcd_pow_minus_one {m n : ℕ} (hm : 0 < m) (hn : 0 < n) :
  Nat.gcd (2^m - 1) (2^n - 1) = 2^Nat.gcd m n - 1 :=
sorry

end gcd_pow_minus_one_l622_622512


namespace Simplify_division_l622_622690

theorem Simplify_division :
  (5 * 10^9) / (2 * 10^5 * 5) = 5000 := sorry

end Simplify_division_l622_622690


namespace attendance_correction_effect_l622_622880

theorem attendance_correction_effect :
  let monday := 15
  let tuesday := 23
  let wednesday_orig := 18
  let wednesday_corr := 20
  let thursday := 24
  let friday := 18

  let original_mean := (monday + tuesday + wednesday_orig + thursday + friday) / 5
  let new_mean := (monday + tuesday + wednesday_corr + thursday + friday) / 5
  let mean_increase := new_mean - original_mean
  
  let original_median := 
    let data := [monday, tuesday, wednesday_orig, thursday, friday].sort
    data[2]
  
  let new_median := 
    let new_data := [monday, tuesday, wednesday_corr, thursday, friday].sort 
    new_data[2]
  let median_increase := new_median - original_median
  
  mean_increase = 0.4 ∧ median_increase = 2 :=
by
  sorry

end attendance_correction_effect_l622_622880


namespace part_I_part_II_part_III_l622_622566

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem part_I : f (π / 2) = 1 := 
sorry

theorem part_II : ∀ x : ℝ, f (x + 2 * π) = f x :=
sorry

noncomputable def g (x : ℝ) : ℝ := f (x + π / 4) + f (x + 3 * π / 4)

theorem part_III : ∀ x : ℝ, ∃ k : ℤ, x = k * 2 * π + 3 * π / 4 → g x = -2 :=
sorry

end part_I_part_II_part_III_l622_622566


namespace sheets_of_paper_l622_622756

theorem sheets_of_paper (S E : ℕ) (h1 : S - E = 100) (h2 : E = S / 3 - 25) : S = 120 :=
sorry

end sheets_of_paper_l622_622756


namespace sum_cubes_eq_power_l622_622007

/-- Given the conditions, prove that 1^3 + 2^3 + 3^3 + 4^3 = 10^2 -/
theorem sum_cubes_eq_power : 1 + 2 + 3 + 4 = 10 → 1^3 + 2^3 + 3^3 + 4^3 = 10^2 :=
by
  intro h
  sorry

end sum_cubes_eq_power_l622_622007


namespace parabola_slope_intersection_l622_622171

theorem parabola_slope_intersection
  (F : ℝ × ℝ) (M : ℝ × ℝ)
  (k : ℝ)
  (x1 x2 y1 y2 : ℝ)
  (h_parabola : ∀ (x y : ℝ), y ^ 2 = 4 * x ↔ (x, y) ∈ C)
  (h_focus : F = (1, 0))
  (h_M : M = (-1, 2))
  (h_line : ∀ (x y : ℝ), y = k * (x - F.1))
  (h_angle : ∠(A - M) (B - M) = 90)
  (hx_eq1 : x1 + x2 = 4 / k + 2)
  (hx_eq2 : x1 * x2 = 1)
  (hy_eq1 : y1 + y2 = 4 / k)
  (hy_eq2 : y1 * y2 = -4) :
  k = 1 := sorry

end parabola_slope_intersection_l622_622171


namespace is_isosceles_of_cyclic_quadrilateral_l622_622669

open EuclideanGeometry

theorem is_isosceles_of_cyclic_quadrilateral 
  {A B C P X Y : Point}
  (hP : median C A B P) 
  (hX : line A P ∩ line B C = X)
  (hY : line B P ∩ line A C = Y) 
  (hCyclic : cyclic_quad A B X Y) :
  isosceles_triangle A B C :=
sorry

end is_isosceles_of_cyclic_quadrilateral_l622_622669


namespace transform_v3_l622_622532

noncomputable def vector_3d := ℝ × ℝ × ℝ

variables (T : vector_3d → vector_3d)

-- Conditions
axiom linear_property (a b : ℝ) (v w : vector_3d) : T (a • v + b • w) = a • T v + b • T w
axiom cross_product_property (v w : vector_3d) : T (v ×ₗ w) = T v ×ₗ T w
axiom transform_v1 : T (5, 5, 2) = (3, -2, 7)
axiom transform_v2 : T (-5, 2, 5) = (3, 7, -2)

-- Question
theorem transform_v3 : T (2, 7, 9) = (5, 6, 12) := sorry

end transform_v3_l622_622532


namespace speed_of_mans_train_is_80_kmph_l622_622414

-- Define the given constants
def length_goods_train : ℤ := 280 -- length in meters
def time_to_pass : ℤ := 9 -- time in seconds
def speed_goods_train : ℤ := 32 -- speed in km/h

-- Define the conversion factor from km/h to m/s
def kmh_to_ms (v : ℤ) : ℤ := v * 1000 / 3600

-- Define the speed of the goods train in m/s
def speed_goods_train_ms := kmh_to_ms speed_goods_train

-- Define the speed of the man's train in km/h
def speed_mans_train : ℤ := 80

-- Prove that the speed of the man's train is 80 km/h given the conditions
theorem speed_of_mans_train_is_80_kmph :
  ∃ V : ℤ,
    (V + speed_goods_train) * 1000 / 3600 = length_goods_train / time_to_pass → 
    V = speed_mans_train :=
by
  sorry

end speed_of_mans_train_is_80_kmph_l622_622414


namespace point_P_2021_l622_622442

structure Point where
  x : Int
  y : Int

def rotate180 (p : Point) (center : Point) : Point :=
  ⟨2 * center.x - p.x, 2 * center.y - p.y⟩

def iter_n_rotations (n : Int) (P: Point) (vertices : List Point) : Point :=
  (List.range n).foldl (λ p i => rotate180 p (vertices.get! (i % 4))) P

theorem point_P_2021:
  let A := ⟨1, 1⟩
  let B := ⟨2, -1⟩
  let C := ⟨-2, -1⟩
  let D := ⟨-1, 1⟩
  let P := ⟨0, 2⟩
  iter_n_rotations 2021 P [A, B, C, D] = ⟨-2018, 0⟩ :=
by
  sorry

end point_P_2021_l622_622442


namespace inequality_solution_l622_622351

theorem inequality_solution {x : ℝ} (h : 2 * x + 1 > x + 2) : x > 1 :=
by
  sorry

end inequality_solution_l622_622351


namespace ABCD_is_isosceles_trapezoid_l622_622369

variables {A B C D E : Type} [planar_geometry A B C D E] 

def is_trapezoid (ABCD : planar_geometry A B C D E) : Prop :=
  -- Replace this definition with a formal definition of a trapezoid in Lean if it exists.
  sorry

def is_isosceles_trapezoid (ABCD : planar_geometry A B C D E) : Prop :=
  -- Replace this definition with a formal definition of an isosceles trapezoid in Lean if it exists.
  sorry

theorem ABCD_is_isosceles_trapezoid 
  (ABCD : planar_geometry A B C D E)
  (h1 : is_trapezoid ABCD)
  (h2 : dist A C = dist B C + dist A D)
  (h3 : ∃ θ : ℝ, θ = 60 ∧ angle A C = θ ∨ angle B D = θ) :
  is_isosceles_trapezoid ABCD :=
sorry

end ABCD_is_isosceles_trapezoid_l622_622369


namespace correct_sample_counts_l622_622211

def total_students : ℕ := 5600
def associate_students : ℕ := 1300
def undergraduate_students : ℕ := 3000
def postgraduate_students : ℕ := 1300
def sample_size : ℕ := 280
def probability_selection : ℚ := sample_size / total_students

def number_of_associate_sampled : ℕ := (associate_students * probability_selection).toNat
def number_of_undergraduate_sampled : ℕ := (undergraduate_students * probability_selection).toNat
def number_of_postgraduate_sampled : ℕ := (postgraduate_students * probability_selection).toNat

theorem correct_sample_counts :
  number_of_associate_sampled = 65 ∧
  number_of_undergraduate_sampled = 150 ∧
  number_of_postgraduate_sampled = 65 :=
by
  sorry

end correct_sample_counts_l622_622211


namespace four_digit_numbers_descending_l622_622578

theorem four_digit_numbers_descending : 
  (∃ l : List ℕ, l.length = 4 ∧ (∀ i : ℕ, i < l.length - 1 → l[i] > l[i+1])) → 
  ∃ n : ℕ, n = Nat.choose 10 4 := by
  sorry

end four_digit_numbers_descending_l622_622578


namespace find_initial_number_l622_622069

theorem find_initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by {
  sorry
}

end find_initial_number_l622_622069


namespace equilateral_triangle_on_polar_curve_l622_622622

theorem equilateral_triangle_on_polar_curve :
  ∀ (O A B : Point) (a : ℝ),
    (∃ (r : ℝ) (θA θB : ℝ),
      r = 2 ∧ 
      A = PolarCoord r θA ∧ 
      B = PolarCoord r θB ∧
      r * cos θA * cos θA = a ∧
      r * cos θB * cos θB = a ∧
      θB = θA + (2 * Real.pi / 3) ∧
      euclidean_distance O A = euclidean_distance O B) →
      a = 3 / 2 := 
by
  sorry

end equilateral_triangle_on_polar_curve_l622_622622


namespace movie_profit_l622_622049

theorem movie_profit
  (main_actor_fee : ℕ)
  (supporting_actor_fee : ℕ)
  (extra_fee : ℕ)
  (main_actor_food : ℕ)
  (supporting_actor_food : ℕ)
  (extra_food : ℕ)
  (crew_size : ℕ)
  (crew_food : ℕ)
  (post_production_cost : ℕ)
  (revenue : ℕ)
  (main_actors_count : ℕ)
  (supporting_actors_count : ℕ)
  (extras_count : ℕ)
  (food_per_main_actor : ℕ)
  (food_per_supporting_actor : ℕ)
  (food_per_remaining_crew : ℕ)
  (equipment_rental_multiplier : ℕ)
  (total_profit : ℕ) :
  main_actor_fee = 500 → 
  supporting_actor_fee = 100 →
  extra_fee = 50 →
  main_actor_food = 10 →
  supporting_actor_food = 5 →
  extra_food = 5 →
  crew_size = 50 →
  crew_food = 3 →
  post_production_cost = 850 →
  revenue = 10000 →
  main_actors_count = 2 →
  supporting_actors_count = 3 →
  extras_count = 1 →
  equipment_rental_multiplier = 2 →
  total_profit = revenue - ((main_actors_count * main_actor_fee) +
                           (supporting_actors_count * supporting_actor_fee) +
                           (extras_count * extra_fee) +
                           (main_actors_count * main_actor_food) +
                           ((supporting_actors_count + extras_count) * supporting_actor_food) +
                           ((crew_size - main_actors_count - supporting_actors_count - extras_count) * crew_food) +
                           (equipment_rental_multiplier * 
                             ((main_actors_count * main_actor_fee) +
                              (supporting_actors_count * supporting_actor_fee) +
                              (extras_count * extra_fee) +
                              (main_actors_count * main_actor_food) +
                              ((supporting_actors_count + extras_count) * supporting_actor_food) +
                              ((crew_size - main_actors_count - supporting_actors_count - extras_count) * crew_food))) +
                           post_production_cost) →
  total_profit = 4584 :=
begin
  -- proof
  sorry
end

end movie_profit_l622_622049


namespace hyperbola_eq_triangle_area_l622_622974

-- The first proof problem
theorem hyperbola_eq (F1 F2 : ℝ × ℝ) (a b c : ℝ) (h1 : F1 = (-sqrt 5, 0)) (h2 : F2 = (sqrt 5, 0)) (h3 : 2 * a = 4) (h4 : c = sqrt 5) (h5 : c^2 = a^2 + b^2) :
    ∀ x y : ℝ, (x^2 / (2 ^ 2)) - y^2 = 1 :=
by sorry

-- The second proof problem
theorem triangle_area (F1 F2 P : ℝ × ℝ) (a c : ℝ) 
  (h1 : F1 = (-sqrt 5, 0)) (h2 : F2 = (sqrt 5, 0)) 
  (h3 : 2 * a = 4) (h4 : c = sqrt 5) 
  (h5 : ∃ P, P ∈ set_of (λ P, (P - F1).sqr_nnnorm > 0 ∧ (P - F2).sqr_nnnorm > 0) ∧ 
              (P.1 * (F1.1 + F2.1) + P.2 * (F1.2 + F2.2) = 0)) -- P such that PF1 ⊥ PF2
  (h6 : ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = (2*c)^2)
  (h7 : ((P.1 - F1.1 - (P.1 - F2.1)) = ±4 ∨ (P.2 - F1.2 - (P.2 - F2.2)) = ±4)) :
  let |PF1| := sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2),
      |PF2| := sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  in 1/2 * |PF1| * |PF2| = 1 := 
by sorry

end hyperbola_eq_triangle_area_l622_622974


namespace player_current_average_l622_622057

theorem player_current_average (A : ℝ) 
  (h1 : 10 * A + 76 = (A + 4) * 11) : 
  A = 32 :=
sorry

end player_current_average_l622_622057


namespace minimum_purchase_price_mod6_l622_622275

theorem minimum_purchase_price_mod6 
  (coin_values : List ℕ)
  (h1 : (1 : ℕ) ∈ coin_values)
  (h15 : (15 : ℕ) ∈ coin_values)
  (h50 : (50 : ℕ) ∈ coin_values)
  (A C : ℕ)
  (k : ℕ)
  (hA : A ≡ k [MOD 7])
  (hC : C ≡ k + 1 [MOD 7])
  (hP : ∃ P, P = A - C) : 
  ∃ P, P ≡ 6 [MOD 7] ∧ P > 0 :=
by
  sorry

end minimum_purchase_price_mod6_l622_622275


namespace correct_propositions_count_l622_622728

theorem correct_propositions_count :
  (¬ (∀ α : ℝ, (cos α ≠ 0 ↔ ∀ k : ℤ, α ≠ (2 * k * π + π / 2)))) ∧
  (∀ (a b : ℝ), 0 < a → 0 < b → (2 / a + 1 / b = 1) → (ab ≥ 4)) ∧
  (∀ (c : ℝ) (data : list ℝ), let var (xs : list ℝ) := ((sum (map (λ x, (x - ((sum xs) / (length xs))) ^ 2) xs)) / (length xs - 1)) in (var data = var (map (λ x, x + c) data))) ∧
  (∀ p : ℝ, 0 < p → p < 1 → (let ξ : ℝ → (N (0, 1)) in (P (ξ > 1) = p) → (P ((-1 < ξ) ∧ (ξ < 0)) = 0.5 - p))) →
  3 := 
sorry

end correct_propositions_count_l622_622728


namespace sum_of_first_ten_terms_is_110_l622_622540

variable {a : ℕ → ℝ} -- a_n is a sequence of real numbers
variable {S : ℕ → ℝ} -- S_n is the sum of the first n terms of the sequence

-- Define the arithmetic sequence condition with common difference d = -2
axiom arithmetic_seq (d : ℝ) (a : ℕ → ℝ) : d = -2 → (∀ n, a (n + 1) = a n + d)

-- Define the condition that a_7 is the geometric mean of a_3 and a_9
axiom geo_mean_condition (a : ℕ → ℝ) : a 7 = real.sqrt (a 3 * a 9)

-- Sum of first n terms of the sequence
axiom sum_first_n (a : ℕ → ℝ) (S : ℕ → ℝ) : (∀ n, S n = n * a 1 + d * n * (n - 1) / 2)

-- Prove that S₁₀ = 110
theorem sum_of_first_ten_terms_is_110 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (h1 : d = -2)
  (h2 : arithmetic_seq d a)
  (h3 : geo_mean_condition a)
  (h4 : sum_first_n a S) : S 10 = 110 := 
sorry

end sum_of_first_ten_terms_is_110_l622_622540


namespace find_number_l622_622599

theorem find_number (x : ℤ) (N : ℤ) (h1 : x = 3) (h2 : N * 10^x < 21000) : N = 20 := 
by 
  sorry

end find_number_l622_622599


namespace domain_of_f_l622_622380

-- Definitions based on conditions provided in the problem
def domain_f (x : ℝ) : Prop :=
  ∃ n : ℤ, x ∈ Ioo (n * π + Real.asin (1/7)) (n * π + π - Real.asin (1/7))

noncomputable def f (x : ℝ) := Real.log 5 (Real.log 6 (Real.log 7 (Real.sin x)))

-- Theorem to prove the domain of the function f
theorem domain_of_f :
  ∀ x : ℝ, ∃ n : ℤ, x ∈ Ioo (n * π + Real.asin (1/7)) (n * π + π - Real.asin (1/7)) ↔
    ∃ y : ℝ, f y = f x :=
by
  -- Proof is replaced with sorry since it is not required as part of the task
  sorry

end domain_of_f_l622_622380


namespace parallel_lines_sufficient_not_necessary_l622_622246

variable {k1 k2 : ℝ} -- the slopes of lines l1 and l2

theorem parallel_lines_sufficient_not_necessary :
  (l1 ∥ l2 → k1 = k2) ∧ (k1 = k2 → l1 ∥ l2 ∨ l1 = l2) :=
sorry

end parallel_lines_sufficient_not_necessary_l622_622246


namespace sequence_properties_l622_622982

noncomputable def a_n (n : ℕ) : ℝ := n + 2

noncomputable def b_n (n : ℕ) : ℝ := 2 * n + 2

theorem sequence_properties 
  (b_2_eq_geometric : (a_n 1 + 2 = 2^(1 + 1)) ) 
  (b_5_eq_geometric : (a_n 4 * 5^(4 - 1) =(2^(4 - 1))) )
  (b_11_eq_geometric : (a_n 10 + 11 = (2 * 10 + 2)) )
  (b_3_eq_a_6 : b_n 2 = a_n 5) :
  (∀ n, a_n n = n + 2 ∧ b_n n = 2 * n + 2) ∧ 
  (∑ i in finset.range n, 1 / (a_n i * b_n i) = n / (4 * n + 8)) := 
sorry

end sequence_properties_l622_622982


namespace geometric_sequence_a6_l622_622352

variable {a : ℕ → ℝ} -- Define the geometric sequence

-- Conditions
variable (S : ℕ → ℝ)
variable (h₁ : ∀ n, S (2 * n) = 4 * (∑ i in range n, a (2 * i + 1))) 
variable (h₂ : a 1 * a 2 * a 3 = 27)

-- Target
theorem geometric_sequence_a6 : a 6 = 243 :=
by
  sorry

end geometric_sequence_a6_l622_622352


namespace min_players_in_team_l622_622064

theorem min_players_in_team (n : ℕ) : 
  (∀ r : ℕ, r ∈ [8, 9, 10] → n % r = 0) → n = 360 :=
begin
  sorry
end

end min_players_in_team_l622_622064


namespace sequence_sum_l622_622860

theorem sequence_sum : (3 - 4 + 5 - 6 + ... + 101 - 102 + 103) = 53 :=
by sorry

end sequence_sum_l622_622860


namespace correct_operation_l622_622791

variable (a : ℝ)

theorem correct_operation : 
  (3 * a^2 + 2 * a^4 ≠ 5 * a^6) ∧
  (a^2 * a^3 ≠ a^6) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) ∧
  ((-2 * a^3)^2 = 4 * a^6) := by
  sorry

end correct_operation_l622_622791


namespace repeating_decimal_division_l622_622377

theorem repeating_decimal_division : 
  (0.\overline{81} : ℚ) = (81 / 99 : ℚ) →
  (0.\overline{36} : ℚ) = (36 / 99 : ℚ) → 
  (0.\overline{81} / 0.\overline{36} : ℚ) = (9 / 4 : ℚ) :=
by 
  intros h1 h2
  rw [h1, h2]
  change (_ / _) = (_ / _)
  sorry

end repeating_decimal_division_l622_622377


namespace proof_equivalence_l622_622989

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Math.sin (3 * x + φ)

def φ_condition (φ : ℝ) : Prop := (-Real.pi / 2 < φ ∧ φ < Real.pi / 2)

def symmetry_condition (φ : ℝ) : Prop := 3 * (Real.pi / 4) + φ = Int.pi * (Int.ofNat 1) / 2

def shifted_function (x : ℝ) (φ : ℝ) : ℝ := 
  2 * Math.sin (3 * (x - Real.pi / 12) - φ) = -2 * Math.cos (3 * x)

def monotonic_condition (φ : ℝ) : Prop := 
  ¬∀ x ∈ Icc (Real.pi / 12) (Real.pi / 3), deriv (λ x, f x φ) x > 0

def symmetry_point_condition (φ : ℝ) : Prop := 
  2 * Math.sin (3 * (5 * Real.pi / 12) - φ) = 0

def minimum_difference (x1 x2 : ℝ) (φ : ℝ) : Prop := 
  (f x1 φ) * (f x2 φ) = -4 → abs (x1 - x2) = Real.pi / 3

theorem proof_equivalence :
  ∀ φ : ℝ, 
    φ_condition φ → 
    symmetry_condition φ → 
    shifted_function = correct →
    monotonic_condition φ ∧
    symmetry_point_condition φ ∧
    ∀ x1 x2, minimum_difference x1 x2 φ :=
sorry

end proof_equivalence_l622_622989


namespace hardest_work_diff_l622_622703

theorem hardest_work_diff 
  (A B C D : ℕ) 
  (h_ratio : A = 1 * x ∧ B = 2 * x ∧ C = 3 * x ∧ D = 4 * x)
  (h_total : A + B + C + D = 240) :
  (D - A) = 72 :=
by
  sorry

end hardest_work_diff_l622_622703


namespace minimum_value_of_expression_l622_622863

theorem minimum_value_of_expression (x y : ℝ) : 
    ∃ (x y : ℝ), (2 * x * y - 1) ^ 2 + (x - y) ^ 2 = 0 :=
by
  sorry

end minimum_value_of_expression_l622_622863


namespace equation_of_parallel_line_l622_622114

-- Definitions for conditions from the problem
def point_A : ℝ × ℝ := (3, 2)
def line_eq (x y : ℝ) : Prop := 4 * x + y - 2 = 0
def parallel_slope : ℝ := -4

-- Proof problem statement
theorem equation_of_parallel_line (x y : ℝ) :
  (∃ (m b : ℝ), m = parallel_slope ∧ b = 2 + 4 * 3 ∧ y = m * (x - 3) + b) →
  4 * x + y - 14 = 0 :=
sorry

end equation_of_parallel_line_l622_622114


namespace trapezoid_diagonals_l622_622328

variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] 
variables (AD BC CD AC BD : ℝ) (angleD : real.angle)
variables [fact (AD = 10)] [fact (BC = 3)] [fact (CD = 4)] [fact (angleD = real.angle.pi / 3)]
variables (trapezoid : Π {α : Type}[affine_space α], α → α → α → α → Prop)
variables [is_trapezoid : trapezoid A B C D]

noncomputable def AC_value : ℝ :=
2 * real.sqrt 19

noncomputable def BD_value : ℝ :=
real.sqrt 37

theorem trapezoid_diagonals :
trapezoid A B C D ∧
AD = 10 ∧ BC = 3 ∧ CD = 4 ∧ angleD = real.angle.pi / 3 →
dist A C = AC_value ∧ dist B D = BD_value :=
begin
  sorry
end

end trapezoid_diagonals_l622_622328


namespace max_mondays_in_45_days_l622_622720

theorem max_mondays_in_45_days (first_is_monday : Bool) : ∃ m, m ≤ 7 ∧ 
  ∀ n ≤ 45,
  let mondays := {i | i % 7 = 1} in
  mondays.count (λ i, i ≤ n) ≤ m :=
sorry

end max_mondays_in_45_days_l622_622720


namespace particle_speed_l622_622054

/--
A particle's position is given by the coordinates (3t + 4, 6t - 16).
Prove that its speed over the interval from t = 2 to t = 5 is 3 * sqrt(5) units of distance per unit of time.
-/
theorem particle_speed :
  let position_at_time := λ t : ℝ, (3 * t + 4, 6 * t - 16)
  let distance := λ p1 p2 : ℝ × ℝ, Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)
  let t_start : ℝ := 2
  let t_end : ℝ := 5
  let p_start := position_at_time t_start
  let p_end := position_at_time t_end
  distance p_start p_end / (t_end - t_start) = 3 * Real.sqrt 5 := by
  sorry

end particle_speed_l622_622054


namespace no_acute_and_obtuse_opposite_edges_l622_622210

theorem no_acute_and_obtuse_opposite_edges
  (ABCD : Type)
  [convex_four_angled_figure ABCD]
  (h : ∀ angle ∈ plane_angles ABCD, angle = 60) :
  ¬(forall_angles_acute ABCD ∨ forall_angles_obtuse ABCD) :=
begin
  sorry
end

end no_acute_and_obtuse_opposite_edges_l622_622210


namespace max_lattice_points_degree_20_l622_622615

noncomputable def max_lattice_points (P : ℤ → ℤ) (degP : ℕ) :=
  ∀ x : ℤ, 0 ≤ P x ≤ 10 → degree P = 20 → length ({y ∈ ℤ | y = P x ∧ 0 ≤ y ≤ 10}.attach : list ℤ) ≤ 20

theorem max_lattice_points_degree_20 :
  ∃ (P : ℤ → ℤ), (degree P = 20) ∧ (max_lattice_points P 20) :=
sorry

end max_lattice_points_degree_20_l622_622615


namespace area_of_region_bounded_by_f_l622_622262

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 5 then x
  else if 5 < x ∧ x ≤ 8 then 2 * x - 5
  else 0

theorem area_of_region_bounded_by_f :
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 8 ∧ 0 ≤ p.2 ∧ p.2 = f p.1}
  let area := 36.5
  (∃ region_area : ℝ, region_area = area) :=
by
  sorry

end area_of_region_bounded_by_f_l622_622262


namespace marble_arrangement_remainder_l622_622851

theorem marble_arrangement_remainder : 
  (let m := 12 in
   let blue_marbles := 4 in
   let total_marbles := m + blue_marbles in
   let ways := Nat.choose (total_marbles - 1) (blue_marbles - 1) in
   ways % 1000 = 820) :=
begin
  -- This theorem only defines the structure.
  sorry
end

end marble_arrangement_remainder_l622_622851


namespace largest_prime_factor_15_3_plus_10_4_minus_5_5_l622_622910

noncomputable def largest_prime_factor_expression : ℕ :=
  let expr := 15^3 + 10^4 - 5^5 in
  let factors := Nat.factors expr in
  factors.maximum -- Assuming we have a maximum function for the list of factors

theorem largest_prime_factor_15_3_plus_10_4_minus_5_5 : 
  largest_prime_factor_expression = 41 := 
sorry

end largest_prime_factor_15_3_plus_10_4_minus_5_5_l622_622910


namespace max_reshaped_balls_in_cube_l622_622382

theorem max_reshaped_balls_in_cube :
  let radius_ball := 3
  let side_length_cube := 10
  let height_cylinder := 6
  let radius_cylinder := 2
  let V_cube := side_length_cube ^ 3
  let V_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  Nat.floor (V_cube / V_cylinder) = 13 :=
by
  let radius_ball := 3
  let side_length_cube := 10
  let height_cylinder := 6
  let radius_cylinder := 2
  let V_cube := side_length_cube ^ 3
  let V_cylinder := Real.pi * (radius_cylinder ^ 2) * height_cylinder
  have h1 : V_cube = 1000 := by norm_num
  have h2 : V_cylinder = 24 * Real.pi := by norm_num
  have h3 : 1000 / (24 * Real.pi) ≈ 13.27 := by norm_num
  have h4 : Nat.floor 13.27 = 13 := by norm_num
  exact Eq.trans (Nat.floor_eq 13.27) h4

end max_reshaped_balls_in_cube_l622_622382


namespace frog_jump_l622_622247

def coprime (p q : ℕ) : Prop := Nat.gcd p q = 1

theorem frog_jump (p q : ℕ) (h_coprime : coprime p q) :
  ∀ d : ℕ, d < p + q → (∃ m n : ℤ, m ≠ n ∧ (m - n = d ∨ n - m = d)) :=
by
  sorry

end frog_jump_l622_622247


namespace length_YW_l622_622003

open Classical

-- Define the elements of the problem
variables (X Y Z W F : Type) [Incas : Inhabited X] [Inhabited Y]

-- Assume the conditions
variables (XY YZ ZF : ℝ) 
variables (W_is_midpoint_YZ : W = (XY + YZ) / 2)
variables (YZ_eq_XY : YZ = XY)
variables (ZF_eq_17 : ZF = 17)

-- Define the proof problem/statement
theorem length_YW : XY = 17 → W = (YZ / 2) → YZ = XY → (YZ / 2) = 8.5 :=
by
  this := W_is_midpoint_YZ sorry
  sorry

end length_YW_l622_622003


namespace find_J_l622_622010

-- Define the problem conditions
def eq1 : Nat := 32
def eq2 : Nat := 4

-- Define the target equation form
def target_eq (J : Nat) : Prop := (eq1^3) * (eq2^3) = 2^J

theorem find_J : ∃ J : Nat, target_eq J ∧ J = 21 :=
by
  -- Rest of the proof goes here
  sorry

end find_J_l622_622010


namespace smallest_x_l622_622788

theorem smallest_x (x : ℕ) :
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 279 :=
by
  sorry

end smallest_x_l622_622788


namespace basil_leaves_correct_l622_622079

theorem basil_leaves_correct : 
  ∀ (basil_pots rosemary_pots thyme_pots rosemary_leaves_per_pot thyme_leaves_per_pot total_leaves : ℕ) 
    (basil_leaves_per_pot : ℕ),
  basil_pots = 3 → 
  rosemary_pots = 9 → 
  thyme_pots = 6 → 
  rosemary_leaves_per_pot = 18 → 
  thyme_leaves_per_pot = 30 → 
  total_leaves = 354 → 
  (354 - (rosemary_pots * rosemary_leaves_per_pot + thyme_pots * thyme_leaves_per_pot)) / basil_pots = basil_leaves_per_pot → 
  basil_leaves_per_pot = 4 :=
by {
  intros,
  sorry
}

end basil_leaves_correct_l622_622079


namespace total_sounds_produced_l622_622192

-- Defining the total number of nails for one customer and the number of customers
def nails_per_person : ℕ := 20
def number_of_customers : ℕ := 3

-- Proving the total number of nail trimming sounds for 3 customers = 60
theorem total_sounds_produced : nails_per_person * number_of_customers = 60 := by
  sorry

end total_sounds_produced_l622_622192


namespace trivia_game_answer_l622_622019

theorem trivia_game_answer (correct_first_half : Nat)
    (points_per_question : Nat) (final_score : Nat) : 
    correct_first_half = 8 → 
    points_per_question = 8 →
    final_score = 80 →
    (final_score - correct_first_half * points_per_question) / points_per_question = 2 :=
by
    intros h1 h2 h3
    sorry

end trivia_game_answer_l622_622019


namespace find_a1_l622_622350

-- Define the sequence in terms of the conditions
def sequence_a (a : ℕ → ℝ) := ∀ n, b n = a (n + 1) + (-1:ℝ) ^ n * a n
def sum_b_n (a : ℕ → ℝ) (b : ℕ → ℝ) := ∀ n, (∑ i in finset.range n, b i) = n ^ 2
def sum_a_minus_n (a : ℕ → ℝ) := (∑ i in finset.range 2018, a (i + 1) - (i + 1)) = 1

-- Define the main theorem to prove a_1 = 1.5
theorem find_a1 (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : sequence_a a) (h2 : sum_b_n a b) (h3 : sum_a_minus_n a) : a 1 = 1.5 :=
sorry

end find_a1_l622_622350


namespace simplify_fraction_l622_622315

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l622_622315


namespace number_of_pens_l622_622045

theorem number_of_pens (num_pencils : ℕ) (total_cost : ℝ) (avg_price_pencil : ℝ) (avg_price_pen : ℝ) : ℕ :=
  sorry

example : number_of_pens 75 690 2 18 = 30 :=
by 
  sorry

end number_of_pens_l622_622045


namespace impossible_illustration_l622_622490

variables {a b c : ℝ}

-- Defining the quadratic equation and its discriminant
def quadratic (x : ℝ) : ℝ := a * x ^ 2 + b * x + c
def discriminant (a b c : ℝ) : ℝ := b ^ 2 - 4 * a * c

-- Assume the discriminant condition for two distinct real roots
axiom roots_condition (h : discriminant a b c > 0) : 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic x1 = 0 ∧ quadratic x2 = 0

-- Defining the vertex x-coordinate
def vertex_x_coord : ℝ := -b / (2 * a)

-- Assume the depiction from the illustration
axiom illustration (h : discriminant a b c > 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic x1 = 0 ∧ quadratic x2 = 0 ∧ 
  (x1 ≠ -b / (2 * a) + (x1 - -b / (2 * a)) ∧ x2 ≠ -b / (2 * a) - (x2 - -b / (2 * a)))

-- The theorem to prove the impossibility of the given situation
theorem impossible_illustration : discriminant a b c > 0 → 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic x1 = 0 ∧ quadratic x2 = 0 → 
  (x1 ≠ -b / (2 * a) + (x1 - -b / (2 * a)) ∧ x2 ≠ -b / (2 * a) - (x2 - -b / (2 * a))) → False :=
begin
  intros d hxy hi,
  sorry
end

end impossible_illustration_l622_622490


namespace param_line_segment_l622_622345

theorem param_line_segment:
  ∃ (a b c d : ℤ), b = 1 ∧ d = -3 ∧ a + b = -4 ∧ c + d = 9 ∧ a^2 + b^2 + c^2 + d^2 = 179 :=
by
  -- Here, you can use sorry to indicate that proof steps are not required as requested
  sorry

end param_line_segment_l622_622345


namespace how_many_more_rolls_needed_l622_622924

variable (total_needed sold_to_grandmother sold_to_uncle sold_to_neighbor : ℕ)

theorem how_many_more_rolls_needed (h1 : total_needed = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end how_many_more_rolls_needed_l622_622924


namespace count_numbers_with_digit_2_l622_622178

def contains_digit_2 (n : Nat) : Prop :=
  n / 100 = 2 ∨ (n / 10 % 10) = 2 ∨ (n % 10) = 2

theorem count_numbers_with_digit_2 (N : Nat) (H : 200 ≤ N ∧ N ≤ 499) : 
  Nat.card {n // 200 ≤ n ∧ n ≤ 499 ∧ contains_digit_2 n} = 138 :=
by
  sorry

end count_numbers_with_digit_2_l622_622178


namespace find_a2023_l622_622530

theorem find_a2023 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a n + a (n + 1) = n) : a 2023 = 1012 :=
sorry

end find_a2023_l622_622530


namespace trapezoid_angle_F_l622_622626

-- Define the problem with the given conditions and prove the required statement.
theorem trapezoid_angle_F {EF GH : Prop} (h1 : EF ∥ GH) (F E G H : ℝ)
  (h2 : E = 3 * H) (h3 : G = 4 * F) : F = 36 :=
by {
  -- Since EF ∥ GH, we have F + G = 180 (degrees).
  have h_angles_sum : F + G = 180, sorry,
  -- By substituting G = 4 * F into the sum angle equation.
  have h_substitute : F + 4 * F = 180 := by { rw h3 at h_angles_sum, assumption },
  -- Simplify the equation to get 5 * F = 180.
  have h_simplify : 5 * F = 180 := by rw [add_comm] at h_substitute,
  -- Finally, solve for F to get F = 36.
  exact F = 180 / 5, sorry
}

end trapezoid_angle_F_l622_622626


namespace simplify_expression_l622_622306

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l622_622306


namespace simplify_expression_l622_622295

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622295


namespace set_complement_intersection_equals_set_l622_622173

open Set

variable U M N : Set ℕ

theorem set_complement_intersection_equals_set (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {1, 4}) (hN : N = {2, 3}) : 
  ((U \ M) ∩ (U \ N)) = {5, 6} :=
by
  sorry

end set_complement_intersection_equals_set_l622_622173


namespace find_a_l622_622967

noncomputable def f (x : ℝ) : ℝ := x^2 + 9
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 9) : a = Real.sqrt 5 :=
by
  sorry

end find_a_l622_622967


namespace area_of_S_l622_622830

noncomputable def regular_octagon_side_length (d : ℝ) : ℝ :=
d / real.sqrt (2 + real.sqrt 2)

noncomputable def map_region (R : set ℂ) : set ℂ :=
{z | ∃ (w ∈ R), z = w⁻¹}

theorem area_of_S :
  let R := {z : ℂ | is_outside_regular_octagon z 2} in
  let S := map_region R in
  ∃ (area : ℝ), area = 2 * real.pi :=
by {
  -- Proof will be provided here
  sorry
}

end area_of_S_l622_622830


namespace collinear_intersections_l622_622410

-- Define the setup with hexagon inscribed in a circle
variables {S : Type*} [circle S]
variables (A B C D E F : S)

-- Definitions of the lines formed by the points of the hexagon
def line_AB : line := line_through A B
def line_DE : line := line_through D E
def line_BC : line := line_through B C
def line_EF : line := line_through E F
def line_CD : line := line_through C D
def line_FA : line := line_through F A

-- Definitions of the points of intersection
def intersection_AB_DE := point_of_intersection line_AB line_DE
def intersection_BC_EF := point_of_intersection line_BC line_EF
def intersection_CD_FA := point_of_intersection line_CD line_FA

-- The goal to prove
theorem collinear_intersections :
  collinear [intersection_AB_DE, intersection_BC_EF, intersection_CD_FA] :=
sorry

end collinear_intersections_l622_622410


namespace sum_and_product_identity_l622_622373

def x : ℕ → ℕ
| 0     := 2
| (n+1) := x n * x n + x n

def y (n : ℕ) : ℚ := 1 / (1 + x n)

def A (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), y i

def B (n : ℕ) : ℚ := ∏ i in Finset.range (n + 1), y i

theorem sum_and_product_identity (n : ℕ) : 2 * A n + B n = 1 :=
begin
  sorry
end

end sum_and_product_identity_l622_622373


namespace finite_common_terms_l622_622645

theorem finite_common_terms 
  (a b : ℕ → ℕ) 
  (h_a : ∀ n : ℕ, a (n + 1) = n * a n + 1)
  (h_b : ∀ n : ℕ, b (n + 1) = n * b n - 1) : 
  (set.finite {n | ∃ m, a n = b m}) :=
by 
  sorry

end finite_common_terms_l622_622645


namespace factor_difference_of_squares_l622_622893

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l622_622893


namespace find_theta_l622_622973

open Real

-- Definitions for acute angle and the discriminant of the quadratic equation
def is_acute (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

def has_repeated_root (a b c : ℝ) : Prop :=
  b ^ 2 - 4 * a * c = 0

-- Main theorem: Prove θ = π/12 or θ = 5π/12 for the given quadratic equation
theorem find_theta
  (θ : ℝ) (h_acute : is_acute θ) :
  has_repeated_root 1 (4 * cos θ) (cot θ) → θ = π / 12 ∨ θ = 5 * π / 12 :=
by sorry

end find_theta_l622_622973


namespace spadesuit_heart_calculation_l622_622511

def spadesuit (a b : ℝ) : ℝ := a - 1 / (b ^ 2)
def heartsuit (a b : ℝ) : ℝ := a + b ^ 2

theorem spadesuit_heart_calculation :
  spadesuit 3 (heartsuit 3 2) = 146 / 49 :=
by sorry

end spadesuit_heart_calculation_l622_622511


namespace tv_price_change_l622_622196

theorem tv_price_change (P : ℝ) : P > 0 → (let P' := 0.80 * P in
                                           let P'' := 1.45 * P' in 
                                           P'' = 1.16 * P) :=
by
  intros hP
  let P' := 0.80 * P
  let P'' := 1.45 * P'
  have h : P'' = 1.16 * P
  sorry
  exact h

end tv_price_change_l622_622196


namespace sum_of_primitive_roots_is_8_l622_622774

def is_primitive_root_mod (a n : ℕ) : Prop :=
  ∀ b : ℕ, (1 ≤ b) → (b < n) → ∃ k : ℕ, (a ^ k) % n = b

noncomputable def set_of_primitive_roots_mod_11 : Finset ℕ :=
  {x ∈ (Finset.range 11) | is_primitive_root_mod x 11}

noncomputable def sum_of_primitive_roots_mod_11 : ℕ :=
  (set_of_primitive_roots_mod_11 ∩ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}).sum

theorem sum_of_primitive_roots_is_8 : sum_of_primitive_roots_mod_11 = 8 := by
  sorry

end sum_of_primitive_roots_is_8_l622_622774


namespace sqrt_div_sqrt_eq_sqrt_fraction_l622_622491

theorem sqrt_div_sqrt_eq_sqrt_fraction
  (x y : ℝ)
  (h : ((1 / 2) ^ 2 + (1 / 3) ^ 2) / ((1 / 3) ^ 2 + (1 / 6) ^ 2) = 13 * x / (47 * y)) :
  (Real.sqrt x / Real.sqrt y) = (Real.sqrt 47 / Real.sqrt 5) :=
by
  sorry

end sqrt_div_sqrt_eq_sqrt_fraction_l622_622491


namespace wrapping_paper_fraction_each_present_l622_622685

theorem wrapping_paper_fraction_each_present (total_fraction : ℚ) (num_presents : ℕ) 
  (H : total_fraction = 3/10) (H1 : num_presents = 3) :
  total_fraction / num_presents = 1/10 :=
by sorry

end wrapping_paper_fraction_each_present_l622_622685


namespace Mario_savings_percentage_l622_622223

theorem Mario_savings_percentage 
  (P : ℝ) -- Normal price of a single ticket 
  (h_campaign : 5 * P = 3 * P) -- Campaign condition: 5 tickets for the price of 3
  : (2 * P) / (5 * P) * 100 = 40 := 
by
  -- Below this, we would write the actual automated proof, but we leave it as sorry.
  sorry

end Mario_savings_percentage_l622_622223


namespace count_universal_statements_l622_622436

-- Defining the propositions
def P1 : Prop := ∃ x : ℝ, x^2 + 2*x + 1 = 0
def P2 : Prop := ∀ x : ℝ, x^2 + 2*x + 1 = 0
def P3 : Prop := ∀ x : ℝ, ¬(x^2 + 2*x + 1 = 0)
def P4 : Prop := ∃ x : ℝ, x^2 + 2*x + 1 = 0

-- Problem statement
theorem count_universal_statements : 2 = (if P2 then 1 else 0) + (if P3 then 1 else 0) := by
  sorry

end count_universal_statements_l622_622436


namespace find_function_expression_find_alpha_l622_622166

open Real

-- Define the main function
def f (x : ℝ) : ℝ := sin (x + π / 3)

-- Proof structure for part (1)
theorem find_function_expression (A ω : ℝ) (hA : A > 0) (hω : ω > 0) (symm_dist : π = π) 
  (passes_through_point : A * sin (ω * π / 3 + π / 3) = √3 / 2) : f(x) = sin (x + π / 3) :=
sorry

-- Proof structure for part (2)
theorem find_alpha (α : ℝ) (hα : 0 < α ∧ α < π)
  (h : f α + √3 * f (α - π / 2) = 1) : α = π / 6 ∨ α = 5 * π / 6 :=
sorry

end find_function_expression_find_alpha_l622_622166


namespace converse_of_posImpPosSquare_l622_622709

-- Let's define the condition proposition first
def posImpPosSquare (x : ℝ) : Prop := x > 0 → x^2 > 0

-- Now, we state the converse we need to prove
theorem converse_of_posImpPosSquare (x : ℝ) (h : posImpPosSquare x) : x^2 > 0 → x > 0 := sorry

end converse_of_posImpPosSquare_l622_622709


namespace area_of_region_covered_by_two_squares_l622_622695

-- Defining the problem
structure Square (point : Type) :=
  (A B C D E F G H : point)
  (congruent : bool)
  (AB : ℝ)
  (center : G = (A + B) / 2 ∨ G = (C + D) / 2)

theorem area_of_region_covered_by_two_squares : 
  ∀ (P : Type) (S : Square P), 
    (S.congruent = true) → (S.AB = 10) →
    let area := 100 + 100 - 25 in area = 175 :=
by
  intros
  sorry

end area_of_region_covered_by_two_squares_l622_622695


namespace no_conditional_statement_in_D_l622_622789

-- Define what it means for an algorithm to have a conditional statement
def has_conditional_statement (descr : String) : Prop :=
  descr = "Read in three numbers representing the lengths of three sides and calculate the area of the triangle." ∨
  descr = "Given the coordinates of two points, calculate the slope of the line." ∨
  descr = "Given a number x, calculate its common logarithm value."

-- Definitions of the algorithm descriptions
def algorithm_A := "Read in three numbers representing the lengths of three sides and calculate the area of the triangle."
def algorithm_B := "Given the coordinates of two points, calculate the slope of the line."
def algorithm_C := "Given a number x, calculate its common logarithm value."
def algorithm_D := "Given the base area and height of a pyramid, calculate its volume."

-- Proof statement that algorithm D does not contain a conditional statement
theorem no_conditional_statement_in_D : ¬ has_conditional_statement algorithm_D :=
by {
  unfold has_conditional_statement,
  simp,
}

end no_conditional_statement_in_D_l622_622789


namespace angle_PBC_is_10_degrees_l622_622236

-- Define the problem conditions
variables {A B C P : Type*} [equilateral_triangle A B C] (angle_PBC : Real)

-- Translate problem into a Lean Theorem
theorem angle_PBC_is_10_degrees (h1 : 6 * angle_PBC = 3 * PAC) (h2 : 2 * PCA = 3 * PAC) :
  angle_PBC = 10 := 
sorry  -- Proof is omitted

end angle_PBC_is_10_degrees_l622_622236


namespace sum_log_ceiling_floor_l622_622463

theorem sum_log_ceiling_floor : 
  (∑ k in Finset.range (1500 + 1), k * (⌈Real.log k / Real.log (Real.sqrt 3)⌉ - ⌊Real.log k / Real.log (Real.sqrt 3)⌋)) = 1124657 :=
  by
  sorry

end sum_log_ceiling_floor_l622_622463


namespace locus_of_vertices_l622_622943

-- Constants and parameters
variables (O S : Point) -- Given points O and S
variables (h r : Real) -- Height and radius of the given cone
variables (M : Point) -- Midpoint of the segment OS

-- Assumptions
def midpoint (M : Point) (O S : Point) : Prop :=
  dist O M = dist M S

def cone_symmetrical (O S : Point) (h r : Real) : Prop := sorry
-- Define what it means for the cone with vertex at O to be symmetrical to the given cone

-- The main theorem statement
theorem locus_of_vertices (O S : Point) (h r : Real) (M : Point) :
  midpoint M O S →
  cone_symmetrical O S h r :=
sorry

end locus_of_vertices_l622_622943


namespace probability_of_C_l622_622065

theorem probability_of_C (P_A P_B P_C P_D P_E : ℚ)
  (hA : P_A = 2/5)
  (hB : P_B = 1/5)
  (hCD : P_C = P_D)
  (hE : P_E = 2 * P_C)
  (h_total : P_A + P_B + P_C + P_D + P_E = 1) : P_C = 1/10 :=
by
  -- To prove this theorem, you will use the conditions provided in the hypotheses.
  -- Here's how you start the proof:
  sorry

end probability_of_C_l622_622065


namespace exists_balanced_set_l622_622508

def is_balanced (n : ℕ) (s : list ℕ) : Prop :=
  s.countp (λ x, x = 0) = n ∧ s.countp (λ x, x = 1) = n

def are_neighbors (a b : list ℕ) : Prop :=
  ∃ x, x ∈ a ∧ (∃ y ∈ a.erase x, b = (a.erase x).insert_at (a.index_of y) x)

theorem exists_balanced_set (n : ℕ) (h : 0 < n) :
  ∃ S : finset (list ℕ), 
    S.card ≤ (nat.choose (2 * n) n) / (n + 1) ∧ 
    ∀ a, is_balanced n a → (a ∈ S ∨ ∃ b ∈ S, are_neighbors a b) :=
sorry

end exists_balanced_set_l622_622508


namespace alpha_pairing_condition_l622_622146

theorem alpha_pairing_condition (k : ℕ) (α : Fin k → ℝ)
  (h : ∀ n : ℕ, Odd n → (Finset.univ : Finset (Fin k)).sum (λ i, (α i) ^ n) = 0) :
  ∀ i : Fin k, α i ≠ 0 → ∃ j : Fin k, i ≠ j ∧ α j = -α i :=
by
  sorry

end alpha_pairing_condition_l622_622146


namespace hyperbola_eccentricity_l622_622966

-- Conditions
variables {a b : ℝ} (ha : a > 0) (hb : b > 0)

def hyperbola := (x y : ℝ) → x^2 / a^2 - y^2 / b^2 = 1

def Focus_right (F : ℝ × ℝ) : Prop := F = (sqrt (a^2 + b^2), 0)

def Asymptotes (A B : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (A = (k * a, k * b) ∧ B = (k * a, -k * b))

def Orthogonality (A B : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  let ℝ_ → ℝ := A - B
  ℝ_ = 0 -- simplify for Lean statement, actual implementation required

def Perpendicular (A B : ℝ × ℝ) : Prop := A.fst = A.snd ∧ B.fst = B.snd

def Parallel (A B O : ℝ × ℝ) : Prop :=
  (A.snd = O.snd) ∧ (B.snd = O.snd)

-- Prove the eccentricity
theorem hyperbola_eccentricity (e : ℝ) 
  (h1 : hyperbola a b)
  (h2 : Focus_right a b)
  (h3 : Asymptotes a b)
  (h4 : Orthogonality a b)
  (h5 : Perpendicular a b)
  (h6 : Parallel a b) :
  e = (2 * sqrt 3) / 3 := by
    sorry

end hyperbola_eccentricity_l622_622966


namespace bridget_poster_board_side_length_l622_622083

theorem bridget_poster_board_side_length
  (num_cards : ℕ)
  (card_length : ℕ)
  (card_width : ℕ)
  (posterboard_area : ℕ)
  (posterboard_side_length_feet : ℕ)
  (posterboard_side_length_inches : ℕ)
  (cards_area : ℕ) :
  num_cards = 24 ∧
  card_length = 2 ∧
  card_width = 3 ∧
  posterboard_area = posterboard_side_length_inches ^ 2 ∧
  cards_area = num_cards * (card_length * card_width) ∧
  cards_area = posterboard_area ∧
  posterboard_side_length_inches = 12 ∧
  posterboard_side_length_feet = posterboard_side_length_inches / 12 →
  posterboard_side_length_feet = 1 :=
sorry

end bridget_poster_board_side_length_l622_622083


namespace simplify_expression_l622_622307

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l622_622307


namespace increasing_interval_of_f_l622_622726

noncomputable def f (x : ℝ) := 3 ^ |x - 1|

theorem increasing_interval_of_f : ∃ (a : ℝ), ∀ x, 1 < x → f x > f (x - 1) :=
by
  sorry

end increasing_interval_of_f_l622_622726


namespace no_common_points_l622_622972

-- Definitions for the conditions
variables {Point Line Plane : Type}
variable (p : Point)
variable (a b : Line)
variable (alpha : Plane)
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (same_points : Line → Line → Prop)

-- The given conditions
axiom line_a_parallel_to_plane_alpha : parallel a alpha
axiom line_b_contained_in_plane_alpha : contains alpha b

-- The required proof statement
theorem no_common_points (h₁ : parallel a alpha) (h₂ : contains alpha b) : ¬ same_points a b :=
by 
  simp 
  sorry

end no_common_points_l622_622972


namespace pure_acid_total_is_3_8_l622_622609

/-- Volume of Solution A in liters -/
def volume_A : ℝ := 8

/-- Concentration of Solution A (in decimals, i.e., 20% as 0.20) -/
def concentration_A : ℝ := 0.20

/-- Volume of Solution B in liters -/
def volume_B : ℝ := 5

/-- Concentration of Solution B (in decimals, i.e., 35% as 0.35) -/
def concentration_B : ℝ := 0.35

/-- Volume of Solution C in liters -/
def volume_C : ℝ := 3

/-- Concentration of Solution C (in decimals, i.e., 15% as 0.15) -/
def concentration_C : ℝ := 0.15

/-- Total amount of pure acid in the resulting mixture -/
def total_pure_acid : ℝ :=
  (volume_A * concentration_A) +
  (volume_B * concentration_B) +
  (volume_C * concentration_C)

theorem pure_acid_total_is_3_8 : total_pure_acid = 3.8 := by
  sorry

end pure_acid_total_is_3_8_l622_622609


namespace simplify_expression_l622_622294

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622294


namespace restaurant_cooks_l622_622396

variable (C W : ℕ)

theorem restaurant_cooks : 
  (C / W = 3 / 10) ∧ (C / (W + 12) = 3 / 14) → C = 9 :=
by sorry

end restaurant_cooks_l622_622396


namespace jack_driving_years_l622_622229

theorem jack_driving_years (miles_per_four_months : ℕ) (total_miles : ℕ) :
  (miles_per_four_months = 37000) → (total_miles = 999000) → (total_miles / (miles_per_four_months * 3) = 9) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end jack_driving_years_l622_622229


namespace heather_walked_distance_l622_622176

-- Define the given conditions
def total_distance : ℝ := 0.75
def distance_back_to_car : ℝ := 0.08

-- we need to prove that
def distance_from_car_to_entrance : ℝ := 0.335

theorem heather_walked_distance :
  let x := distance_from_car_to_entrance in
  2 * x + distance_back_to_car = total_distance :=
by
  sorry

end heather_walked_distance_l622_622176


namespace sum_of_divisors_57_l622_622784

theorem sum_of_divisors_57 : 
  ∑ d in (finset.filter (λ x, 57 % x = 0) (finset.range (58))), d = 80 := 
by
  sorry

end sum_of_divisors_57_l622_622784


namespace expanding_arith_seq_l622_622546

theorem expanding_arith_seq (x : ℝ) :
  let term_with_largest_binom_coef := binomial_expansion_term 8 4 x,
      terms_with_largest_coef := [binomial_expansion_term 8 2 x, binomial_expansion_term 8 3 x]
  in term_with_largest_binom_coef = (35 / 8) * x^(2 / 3) ∧ 
     terms_with_largest_coef = [7 * x^(7 / 3), 7 * x^(3 / 2)] :=
by
  sorry

def binomial_expansion_term (n r : ℕ) (x : ℝ) : ℝ :=
  nat.choose n r * (1 / 2)^r * x^(3 * n - 5 * r) / 6


end expanding_arith_seq_l622_622546


namespace find_x_l622_622505

-- Conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 8 * x
def area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x / 2

-- Theorem to prove
theorem find_x (x s : ℝ) (h1 : volume_condition x s) (h2 : area_condition x s) : x = 110592 := sorry

end find_x_l622_622505


namespace find_heaviest_coin_min_turns_l622_622763

theorem find_heaviest_coin_min_turns (n : ℕ) (h : n > 2)
  (balances : Fin n → (ℕ → ℕ → Prop)) (faulty : ∃ k, k < n ∧ (∀ (i j : ℕ), (balances k) i j ↔ ¬(balances k) i j)) :
  ∃ (m : ℕ), m = 2n - 1 := 
sorry

end find_heaviest_coin_min_turns_l622_622763


namespace evaluate_expression_l622_622109

theorem evaluate_expression :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^{16} + 5^{16}) * (3^{32} + 5^{32}) * (3^{64} + Real.log2 (5^64)) =
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^{16} + 5^{16}) * (3^{32} + 5^{32}) * (3^{64} + Real.log2 (5^64)) := by
sorry

end evaluate_expression_l622_622109


namespace total_napkins_l622_622271

variable (initial_napkins Olivia_napkins Amelia_multiplier : ℕ)

-- Defining the conditions
def Olivia_gives_napkins : ℕ := 10
def William_initial_napkins : ℕ := 15
def Amelia_gives_napkins : ℕ := 2 * Olivia_gives_napkins

-- Define the total number of napkins William has now
def William_napkins_now : ℕ :=
  initial_napkins + Olivia_napkins + Amelia_gives_napkins

-- Proving the total number of napkins William has now is 45
theorem total_napkins (h1 : Olivia_napkins = 10)
                      (h2: initial_napkins = 15)
                      (h3: Amelia_multiplier = 2)
                      : William_napkins_now initial_napkins Olivia_napkins (Olivia_napkins * Amelia_multiplier) = 45 :=
by
  rw [←h1, ←h2, ←h3]
  sorry

end total_napkins_l622_622271


namespace problem_statement_l622_622869

noncomputable def f (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 1) then 4^x
  else if (-1 < x ∧ x < 0) then -4^(-x)
  else if (-2 < x ∧ x < -1) then -4^(x + 2)
  else if (1 < x ∧ x < 2) then 4^(x - 2)
  else 0

theorem problem_statement :
  (f (-5 / 2) + f 1) = -2 :=
sorry

end problem_statement_l622_622869


namespace closest_percentage_l622_622230

-- Define the lengths of the rectangle
def length : ℝ := 3
def width : ℝ := 4

-- Calculate the distances walked by Jerry and Silvia
def jerry_distance := length + width
def silvia_distance := (length^2 + width^2).sqrt

-- Calculate the reduction percentage
def reduction_percentage := ((jerry_distance - silvia_distance) / jerry_distance) * 100

-- Prove the closest percentage reduction
theorem closest_percentage : abs (reduction_percentage - 30) < 1 :=
by 
  sorry

end closest_percentage_l622_622230


namespace largest_integer_sol_l622_622766

theorem largest_integer_sol (x : ℤ) : (3 * x + 4 < 5 * x - 2) -> x = 3 :=
by
  sorry

end largest_integer_sol_l622_622766


namespace cylinder_surface_area_is_correct_l622_622342

def lateral_surface_unfolds_to_rectangle (r h : ℝ) : Prop :=
  (2 * π * r = 6 * π ∧ h = 4 * π) ∨ (2 * π * r = 4 * π ∧ h = 6 * π)

def surface_area (r h : ℝ) : ℝ :=
  2 * π * r * h + 2 * π * r^2

theorem cylinder_surface_area_is_correct :
  ∃ (r h : ℝ), lateral_surface_unfolds_to_rectangle r h ∧
  (surface_area r h = 24 * π^2 + 18 * π ∨ surface_area r h = 24 * π^2 + 8 * π) :=
begin
  sorry
end

end cylinder_surface_area_is_correct_l622_622342


namespace train_speed_train_speed_sorry_l622_622836

def length_of_train_meters := 300
def time_to_pass_tree_seconds := 12
def distance_kilometers := length_of_train_meters / 1000 -- converting meters to kilometers
def time_hours := time_to_pass_tree_seconds / 3600 -- converting seconds to hours

theorem train_speed :
  (distance_kilometers / time_hours) = 90 := 
by {
  -- Conversions
  have distance_0_3_km : distance_kilometers = 0.3 := by norm_num [distance_kilometers, length_of_train_meters]
  have time_0_0033333_hr : time_hours = 12 / 3600 := by norm_num [time_hours, time_to_pass_tree_seconds]
  
  -- Calculation of speed
  rw [distance_0_3_km, time_0_0033333_hr],
  norm_num,
}

-- skip the actual proof by using sorry
theorem train_speed_sorry :
  (distance_kilometers / time_hours) = 90 := 
sorry

end train_speed_train_speed_sorry_l622_622836


namespace problem_statement_l622_622495

theorem problem_statement :
  ((8^5 / 8^2) * 2^10 - 2^2) = 2^19 - 4 := 
by 
  sorry

end problem_statement_l622_622495


namespace divisor_is_three_l622_622820

theorem divisor_is_three (n d q p : ℕ) (h1 : n = d * q + 3) (h2 : n^2 = d * p + 3) : d = 3 := 
sorry

end divisor_is_three_l622_622820


namespace complex_in_second_quadrant_modulus_of_complex_number_l622_622555

noncomputable def z (k : ℝ) : ℂ := (k^2 - 3*k - 4) + (k - 1)*complex.I

theorem complex_in_second_quadrant (k : ℝ) (hk : 1 < k ∧ k < 4) : 
  (z k).im > 0 ∧ (z k).re < 0 :=
sorry

theorem modulus_of_complex_number (k : ℝ) (hk : (z k * complex.I).re = 0) : 
  complex.abs (z k) = 2 ∨ complex.abs (z k) = 3 :=
sorry

end complex_in_second_quadrant_modulus_of_complex_number_l622_622555


namespace sum_primitive_roots_eq_23_l622_622776

-- Define the set of integers.
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define modulo
def modulo : ℕ := 11

-- Predicate to check if an integer is a primitive root modulo 11.
def isPrimitiveRoot (a : ℕ) : Prop := 
  ∀ k : ℕ, k < modulo → (a ^ k) % modulo ≠ 1

-- Find the sum of primitive roots in the set S modulo 11.
def sumOfPrimitiveRoots : ℕ :=
  (Sum (filter isPrimitiveRoot S))

-- Specification of the proof.
theorem sum_primitive_roots_eq_23 : sumOfPrimitiveRoots = 23 := 
  sorry

end sum_primitive_roots_eq_23_l622_622776


namespace max_teams_participate_l622_622600

theorem max_teams_participate (teams players games : ℕ) 
  (h_players_per_team : ∀ t, t ∈ teams → players = 3)
  (h_game_between_players : ∀ p1 p2, p1 ∈ players → p2 ∈ players → p1 ≠ p2 → games = players * players)
  (h_max_games : ∀ g, g ∈ games → g ≤ 150) :
  teams = 6 :=
by
  sorry

end max_teams_participate_l622_622600


namespace minimum_value_ineq_l622_622258

noncomputable def minimum_value (a b : ℝ) : ℝ := 
  \frac{1}{a} + \frac{1}{b}

theorem minimum_value_ineq (a b : ℝ) (h : a > 0 ∧ b > 0) (h1 : a + 3 * b = 1) :
  minimum_value a b = 4 + 2 * \sqrt{3} :=
sorry

end minimum_value_ineq_l622_622258


namespace range_of_a_l622_622964

open Int

theorem range_of_a (a : ℝ) :
  (a * a - 16 ≥ 0) ∧ (∀ x, (3 ≤ x → 4 * x + a ≥ 0)) ↔ (a ∈ set.Icc (-12 : ℝ) (-4) ∨ a ∈ set.Ici (4)) :=
begin
  sorry
end

end range_of_a_l622_622964


namespace parallel_vectors_implies_k_l622_622174

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (1, 2)

-- Define the given expressions in the problem
def vec_lhs := (5, 4)
def vec_rhs (k : ℝ) := (1 + k, 1/2 + 2 * k)

-- Prove that lhs is parallel to rhs implies k = 1/4
theorem parallel_vectors_implies_k (k : ℝ) :
  (5 * (1/2 + 2 * k) - 4 * (1 + k) = 0) → (k = 1/4) := by
  sorry

end parallel_vectors_implies_k_l622_622174


namespace find_EG_l622_622494

variables (AE EG GF : ℝ)

-- Problem assumptions
axiom AE_more_EG : AE = EG + 2
axiom GF_value : GF = 5

-- Goal to prove
theorem find_EG : ∃ EG : ℝ, AE = EG + 2 ∧ GF = 5 ∧ EG = 4 := 
by
  existsi 4
  split
  exact AE_more_EG
  split
  exact GF_value
  sorry

end find_EG_l622_622494


namespace factorial_fraction_simplification_l622_622455

theorem factorial_fraction_simplification :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 := 
by {
  sorry
}

end factorial_fraction_simplification_l622_622455


namespace part1_simplified_part1_value_at_1230_part2_value_l622_622136

def f (α : ℝ) : ℝ :=
  (Real.cos (π / 2 + α) * Real.cos (2 * π + α) * Real.sin (-α + 3 / 2 * π)) / 
  (Real.sin (α + 7 / 2 * π) * Real.sin (-3 * π - α))

theorem part1_simplified (α : ℝ) :
  f α = -Real.cos α :=
sorry

theorem part1_value_at_1230 :
  f 1230 = Real.sqrt 3 / 2 :=
sorry

theorem part2_value (α : ℝ) (h1 : Real.cos (α - 3 / 2 * π) = 1 / 5) (h2 : ϕ ∈ Set.Icc (π) (3 * π / 2)) :
  f α = 2 * Real.sqrt 6 / 5 :=
sorry

end part1_simplified_part1_value_at_1230_part2_value_l622_622136


namespace star_evaluation_l622_622929

noncomputable def star (a b : ℝ) : ℝ := (a + b) / (a - b)

theorem star_evaluation : (star (star 2 3) 4) = 1 / 9 := 
by sorry

end star_evaluation_l622_622929


namespace only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l622_622903

theorem only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime (n : ℕ) : 
  Prime (2^n + n^2016) ↔ n = 1 := by
  sorry

end only_n_equal_1_is_natural_number_for_which_2_pow_n_plus_n_pow_2016_is_prime_l622_622903


namespace original_cookie_price_l622_622096

theorem original_cookie_price (C : ℝ) (h1 : 1.5 * 16 + (C / 2) * 8 = 32) : C = 2 :=
by
  -- Proof omitted
  sorry

end original_cookie_price_l622_622096


namespace find_common_ratio_l622_622572

theorem find_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h₀ : ∀ n, a (n + 1) = q * a n)
  (h₁ : a 0 = 4)
  (h₂ : q ≠ 1)
  (h₃ : 2 * a 4 = 4 * a 0 - 2 * a 2) :
  q = -1 := 
sorry

end find_common_ratio_l622_622572


namespace largest_intersection_value_l622_622340

noncomputable def polynomial1 (a b : ℝ) (x : ℝ) : ℝ :=
  x ^ 7 - 11 * x ^ 6 + 35 * x ^ 5 - 5 * x ^ 4 + 7 * x ^ 3 + a * x ^ 2 + b * x

noncomputable def polynomial2 (c d e : ℝ) (x : ℝ) : ℝ := 
  c * x ^ 2 + d * x + e

theorem largest_intersection_value (a b c d e : ℝ) (h1 : (∀ x : ℝ, polynomial1 a b x = polynomial2 c d e x ↔ x ∈ {2, -2, 4}) ∧ ( ∀ x : ℝ, polynomial1 a b x - polynomial2 c d e x = 0 → x = 2 ∨ x = -2 ∨ x = 4)) : 
  ∃ x : ℝ, polynomial1 a b x = polynomial2 c d e x ∧ x = 4 :=
sorry

end largest_intersection_value_l622_622340


namespace room_length_l622_622344

theorem room_length
  (width : ℝ) (rate : ℝ) (total_cost : ℝ) (L : ℝ)
  (h₁ : width = 4)
  (h₂ : rate = 700)
  (h₃ : total_cost = 15400)
  (h₄ : total_cost = L * width * rate) :
  L = 5.5 :=
by {
  rw [h₁, h₂] at h₄,
  calc
    L = 15400 / (4 * 700) : by { rw [h₃, h₄], ring, field_simp }
    ... = 5.5 : by norm_num
}

end room_length_l622_622344


namespace sum_of_divisors_57_eq_80_l622_622782

def sum_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).sum

theorem sum_of_divisors_57_eq_80 : sum_of_divisors 57 = 80 := by
  sorry

end sum_of_divisors_57_eq_80_l622_622782


namespace divisible_by_two_pow_l622_622213

noncomputable def f : ℕ → ℕ
| 0       := 1
| 1       := 2
| (n + 1) := 4 * (f n) + (List.sum (List.map (λ j, f j * f (n - j)) (List.range (n + 1))))

theorem divisible_by_two_pow : ∀ (n : ℕ), ∃ k : ℕ, f n = 2^n * k := by sorry

end divisible_by_two_pow_l622_622213


namespace range_of_f_on_interval_l622_622970

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^k

theorem range_of_f_on_interval (k : ℝ) (hk : k < 0) :
  set.range (λ x : {x // 0 < x ∧ x ≤ 1}, f x.val k) = set.Ioi 1 :=
sorry

end range_of_f_on_interval_l622_622970


namespace arithmetic_expression_evaluation_l622_622465

theorem arithmetic_expression_evaluation : 
  -6 * 3 - (-8 * -2) + (-7 * -5) - 10 = -9 := 
by
  sorry

end arithmetic_expression_evaluation_l622_622465


namespace min_sum_areas_of_triangles_l622_622965

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1 / 4, 0)

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = p.1

def O := (0, 0)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def on_opposite_sides_x_axis (p q : ℝ × ℝ) : Prop := p.2 * q.2 < 0

theorem min_sum_areas_of_triangles 
  (A B : ℝ × ℝ)
  (hA : parabola A)
  (hB : parabola B)
  (hAB : on_opposite_sides_x_axis A B)
  (h_dot : dot_product A B = 2) :
  ∃ m : ℝ, m = 3 := by
  sorry

end min_sum_areas_of_triangles_l622_622965


namespace num_decompositions_144_l622_622451

theorem num_decompositions_144 : ∃ D, D = 45 ∧ 
  (∀ (factors : List ℕ), 
    (∀ x, x ∈ factors → x > 1) ∧ factors.prod = 144 → 
    factors.permutations.length = D) :=
sorry

end num_decompositions_144_l622_622451


namespace sqrt_neg_sq_eq_two_l622_622857

theorem sqrt_neg_sq_eq_two : Real.sqrt ((-2 : ℝ)^2) = 2 := by
  -- Proof intentionally omitted.
  sorry

end sqrt_neg_sq_eq_two_l622_622857


namespace max_value_and_set_range_of_a_l622_622168

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos x) ^ 2 - Real.sin (2 * x - 7 * Real.pi / 6)

theorem max_value_and_set (x : ℝ) :
  ∃ (k : ℤ), f(x) = 2 ↔ x = k * Real.pi + Real.pi / 6 := sorry

theorem range_of_a (A a b c : ℝ) (h1 : f(A) = 3 / 2) (h2 : b + c = 2) :
  1 ≤ a ∧ a < 2 := sorry

end max_value_and_set_range_of_a_l622_622168


namespace average_score_makeup_date_l622_622604

theorem average_score_makeup_date :
  ∃ (X : ℝ), 
    let total_students := 100
    let assigned_day_students := total_students * 0.7
    let makeup_date_students := total_students * 0.3
    let average_assigned_day := 0.65
    let average_class := 0.74
  in
    (assigned_day_students * average_assigned_day + makeup_date_students * X) / total_students = average_class ∧ X = 0.95 := sorry

end average_score_makeup_date_l622_622604


namespace simplify_expression_l622_622020

variable (d : ℤ)

theorem simplify_expression :
  (5 + 4 * d) / 9 - 3 + 1 / 3 = (4 * d - 19) / 9 := by
  sorry

end simplify_expression_l622_622020


namespace f_one_eq_zero_f_monotonic_f_inequality_solution_l622_622523

open Function

-- Given conditions: f is defined on (0, +∞), satisfies functional equation, and is negative when x > 1
variables {f : ℝ → ℝ} (h1 : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → f (x1 / x2) = f x1 - f x2)
  (h2 : ∀ x : ℝ, 1 < x → f x < 0)

-- (Ⅰ) Proof for f(1) = 0
theorem f_one_eq_zero : f 1 = 0 :=
begin
  have h : f 1 = f 1 - f 1, { exact h1 1 1 zero_lt_one zero_lt_one },
  rw sub_self (f 1) at h,
  exact h,
end

-- (Ⅱ) Proof for monotonicity of f(x)
theorem f_monotonic : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 > x2 → f x1 < f x2 :=
begin
  intros x1 x2 hx1 hx2 hgt,
  have h : 1 < x1 / x2 := (one_lt_div hx2).mpr hgt,
  specialize h2 (x1 / x2) h,
  rw h1 at h2,
  linarith,
end

-- (Ⅲ) Proof for the solution of f(x^2) > -2 given f(3) = -1
theorem f_inequality_solution (h3 : f 3 = -1) : ∀ x : ℝ, f (x^2) > -2 ↔ x ∈ Set.Ioo (-3) 0 ∪ Set.Ioo 0 3 :=
begin
  intro x,
  split,
  {
    intro h,
    have hx2_neg := lt_of_lt_of_le h (le_of_eq h3.symm),
    have hx2_pos := h2 (x^2) (lt_of_le_of_ne (le_of_eq h3.symm) (ne_of_lt hx2_neg).symm),
    split_ifs;
    try { linarith [hx2_neg, hx2_pos] },
  },
  {
    rintro (hx|hx); 
    specialize h2 (x^2) (lt_trans zero_lt_one (cx_iff_eq.mp h.2 2)),
    intros,
    have := lt_of_lt_of_le (hx_iff_eq.mp h.1 2) h2,
    split_ifs at this; linarith;
  }
end

end f_one_eq_zero_f_monotonic_f_inequality_solution_l622_622523


namespace solve_for_x_l622_622181

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 5) = 2) : x = -3 :=
by
  sorry

end solve_for_x_l622_622181


namespace ratio_second_to_first_day_l622_622474

/-
Problem Statement:
Prove that the ratio of the number of pages Cyrus wrote on the second day to the number of pages he wrote on the first day is 2:1 given the conditions.
-/

noncomputable def cyrus_writing_problem 
    (first_day_pages : ℕ)
    (second_day_pages : ℕ)
    (third_day_pages : ℕ)
    (fourth_day_pages : ℕ)
    (remaining_pages : ℕ)
    (total_pages : ℕ) :=
  first_day_pages = 25 ∧
  third_day_pages = 2 * second_day_pages ∧
  fourth_day_pages = 10 ∧
  remaining_pages = 315 ∧
  total_pages = 500 →
  first_day_pages + second_day_pages + third_day_pages + fourth_day_pages = total_pages - remaining_pages →
  second_day_pages / first_day_pages = 2

theorem ratio_second_to_first_day :
  ∃ (first_day_pages second_day_pages third_day_pages fourth_day_pages remaining_pages total_pages : ℕ),
  cyrus_writing_problem first_day_pages second_day_pages third_day_pages fourth_day_pages remaining_pages total_pages :=
begin
  use [25, 50, 100, 10, 315, 500],
  split,
  { 
    -- Here we'll verify the conditions directly.
    split, {refl},
    split, {refl},
    split, {refl},
    split, {refl},
    refl
  },
  {
    -- Here we will verify the main goal using given parameters.
    calc
    25 + 50 + 100 + 10 = 185 : by norm_num
    ... = 500 - 315 : by norm_num,
  }
end

end ratio_second_to_first_day_l622_622474


namespace inequality_satisfied_for_a_eq_2_l622_622509

theorem inequality_satisfied_for_a_eq_2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x + sqrt(3 * x * y) ≤ 2 * (x + y) :=
sorry

end inequality_satisfied_for_a_eq_2_l622_622509


namespace major_axis_length_l622_622055

theorem major_axis_length (radius : ℝ) (k : ℝ) (minor_axis : ℝ) (major_axis : ℝ)
  (cyl_radius : radius = 2)
  (minor_eq_diameter : minor_axis = 2 * radius)
  (major_longer : major_axis = minor_axis * (1 + k))
  (k_value : k = 0.25) :
  major_axis = 5 :=
by
  -- Proof omitted, using sorry
  sorry

end major_axis_length_l622_622055


namespace find_hyperbola_equation_l622_622336

def hyperbola (x y a b : ℝ) : Prop := 
x^2 / a^2 - y^2 / b^2 = 1

def is_asymptote_to_circle (a b : ℝ) : Prop := 
(y = (sqrt 3) * x) ∨ (y = -(sqrt 3) * x)

def eccentricity (a c : ℝ) : Prop := 
c = 2 * a

def tangent_to_circle (a : ℝ) : Prop := 
|sqrt 3 * a| / (sqrt (a^2 + 1)) = (sqrt 3) / 2

theorem find_hyperbola_equation 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (he : eccentricity a (2 * a)) 
  (ht : tangent_to_circle a) 
  (h3 : is_asymptote_to_circle a b) : 
  hyperbola x y 1 (sqrt 3) := 
sorry

end find_hyperbola_equation_l622_622336


namespace tickets_sold_total_l622_622000

-- Define the conditions
variables (A : ℕ) (S : ℕ) (total_amount : ℝ := 222.50) (adult_ticket_price : ℝ := 4) (student_ticket_price : ℝ := 2.50)
variables (student_tickets_sold : ℕ := 9)

-- Define the total money equation and the question
theorem tickets_sold_total :
  4 * (A : ℝ) + 2.5 * (9 : ℝ) = 222.50 → A + 9 = 59 :=
by sorry

end tickets_sold_total_l622_622000


namespace functional_solution_l622_622936

section EquivalentFunctionalInequality

variable {f : ℝ → ℝ}
variable (c : ℝ) [h0 : c > 1]

noncomputable def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (u v : ℝ), x > 1 → y > 1 → u > 0 → v > 0 → 
  f(x^u * y^u) ≤ f(x)^(1/(4*u)) * f(y)^(1/40)

theorem functional_solution (f : ℝ → ℝ) :
  satisfies_condition f → (∀ x > 1, f x = c^(1 / Real.log x)) :=
by
  sorry

end EquivalentFunctionalInequality

end functional_solution_l622_622936


namespace factor_expression_l622_622900

theorem factor_expression (x : ℝ) : 
  75 * x^11 + 225 * x^22 = 75 * x^11 * (1 + 3 * x^11) :=
by sorry

end factor_expression_l622_622900


namespace fraction_equality_l622_622492

theorem fraction_equality :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 59 / 61 :=
by
  sorry

end fraction_equality_l622_622492


namespace binomial_arithmetic_sequence_l622_622712

theorem binomial_arithmetic_sequence (n : ℕ) (h : 2 * nat.choose n 1 * (1 : ℚ) / 2 = nat.choose n 0 + nat.choose n 2 * (1 : ℚ) / 4) : 
  n = 8 :=
sorry

end binomial_arithmetic_sequence_l622_622712


namespace napkins_total_l622_622272

theorem napkins_total (o a w : ℕ) (ho : o = 10) (ha : a = 2 * o) (hw : w = 15) :
  w + o + a = 45 :=
by
  sorry

end napkins_total_l622_622272


namespace problem1_problem2_l622_622990

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2 * x

theorem problem1 :
  f 2 = 4 ∧
  f (1 / 2) = 1 / 4 ∧
  f (f (-1)) = 1 :=
by
  sorry

theorem problem2 (a : ℝ) (h : f a = 3) :
  a = 1 ∨ a = real.sqrt 3 :=
by
  sorry

end problem1_problem2_l622_622990


namespace certain_value_l622_622175

-- Define the given conditions and prove the resulting value
theorem certain_value (n : ℕ) (v : ℕ) (h1 : (1/2 : ℚ) * n + v = 11) (h2 : n = 10) : v = 6 := by
  have h3 : (5 : ℚ) + v = 11 := by
    rw [h2] at h1
    norm_num at h1
    exact h1
  exact eq_of_add_eq_add_right (by norm_num : 5 + v = 11 - 5) h3

end certain_value_l622_622175


namespace combined_rate_37_point_5_l622_622235

variable (miles_per_gallon_Kelly : ℝ) (miles_per_gallon_Alex : ℝ)
variable (miles_driven_Kelly : ℝ) (miles_driven_Alex : ℝ)

def combined_miles_per_gallon_Kelly_Alex (miles_per_gallon_Kelly miles_per_gallon_Alex miles_driven_Kelly miles_driven_Alex : ℝ) : ℝ :=
  let gasoline_used_Kelly  := miles_driven_Kelly / miles_per_gallon_Kelly
  let gasoline_used_Alex   := miles_driven_Alex / miles_per_gallon_Alex
  let total_gasoline_used := gasoline_used_Kelly + gasoline_used_Alex
  let total_miles_driven   := miles_driven_Kelly + miles_driven_Alex
  total_miles_driven / total_gasoline_used

theorem combined_rate_37_point_5 :
  combined_miles_per_gallon_Kelly_Alex 50 25 100 50 = 37.5 :=
by
  sorry

end combined_rate_37_point_5_l622_622235


namespace find_FC_l622_622518

-- Define all given values and relationships
variables (DC CB AD AB ED FC : ℝ)
variables (h1 : DC = 9) (h2 : CB = 6)
variables (h3 : AB = (1/3) * AD)
variables (h4 : ED = (2/3) * AD)

-- Define the goal
theorem find_FC :
  FC = 9 :=
sorry

end find_FC_l622_622518


namespace problem_statement_l622_622997

noncomputable def parabola (x : ℝ) : ℝ := x^2 / 4

noncomputable def line_through (k : ℝ) (y : ℝ) (x : ℝ) : ℝ := k * x + y

theorem problem_statement :
  ∀ (k : ℝ) (a : ℝ), 
    let M := (0, 2)
    let parabola := λ x : ℝ, x^2 / 4
    let line_through := λ k y x, k * x + y
    let l := λ x : ℝ, a * x - a^2
    let N1 := (2 / a + a, 2)
    let N2 := (-2 / a + a, -2)
    |(0 - ((2 / a) - (-2 / a)) )|^2 = 8
  sorry

end problem_statement_l622_622997


namespace circleO_eq_line_l_eq_l622_622535

variables {P : ℝ × ℝ} {r : ℝ} (l : ℝ → ℝ → Prop)
  (circleO : ℝ → ℝ → Prop) (circleC : ℝ → ℝ → Prop)
  (A B M N : ℝ × ℝ)

def circleO_def := circleO x y ↔ x^2 + y^2 = r^2
def circleC_def := circleC x y ↔ (x + 1)^2 + (y + 1)^2 = 2
def point_P_is_on_circleO := P = (-4, 0) ∧ circleO P.1 P.2
def line_l_intersects_circleO := ∃ A B, l A.1 A.2 ∧ l B.1 B.2 ∧ circleO A.1 A.2 ∧ circleO B.1 B.2
def line_l_intersects_circleC := ∃ M N, l M.1 M.2 ∧ l N.1 N.2 ∧ circleC M.1 M.2 ∧ circleC N.1 N.2
def mid_point_of_segmentAB := M = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2
def PM_eq_PN := ∥P - M∥ = ∥P - N∥

theorem circleO_eq : circleO_def circleO → point_P_is_on_circleO P circleO → line_l_intersects_circleO l circleO A B → line_l_intersects_circleC l circleC M N → mid_point_of_segmentAB M A B → PM_eq_PN P M N → ∀ x y, circleO x y ↔ x^2 + y^2 = 16 :=
by sorry

theorem line_l_eq : circleO_def circleO → circleC_def circleC → point_P_is_on_circleO P circleO → line_l_intersects_circleO l circleO A B → line_l_intersects_circleC l circleC M N → mid_point_of_segmentAB M A B → PM_eq_PN P M N → 
  ∃ m₁ m₂, (∀ x y, l x y ↔ y = 3 * x + m₁) ∨ (∀ x y, l x y ↔ y = 3 * x + m₂) →
  m₁ = 0 ∧ m₂ = 4 :=
by sorry

end circleO_eq_line_l_eq_l622_622535


namespace derivative_at_two_l622_622557

theorem derivative_at_two :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 3 * x^2 + 2 * (deriv f 2)) → (deriv f 2 = 12) :=
begin
  -- To be proved
  sorry
end

end derivative_at_two_l622_622557


namespace fifteenth_prime_l622_622107

open Nat

theorem fifteenth_prime : nth_prime 15 = 47 :=
by
  sorry

end fifteenth_prime_l622_622107


namespace solve_trig_expression_l622_622917

theorem solve_trig_expression :
  (sin (80 * Real.pi / 180) / sin (20 * Real.pi / 180)) - (Real.sqrt 3 / (2 * sin (80 * Real.pi / 180))) = 2 := 
by sorry

end solve_trig_expression_l622_622917


namespace puppies_percentage_l622_622815

theorem puppies_percentage {total_puppies puppies_with_5_spots : ℕ} (h1 : total_puppies = 20)
  (h2 : puppies_with_5_spots = 8) : 
  (puppies_with_5_spots / total_puppies : ℚ) * 100 = 40 :=
by
  rw [h1, h2]
  norm_num
  sorry

end puppies_percentage_l622_622815


namespace remainder_sum_base_s_plus_1_l622_622876

theorem remainder_sum_base_s_plus_1 {s : ℕ} :
  let sum := (Σ i in Finset.range (s + 1), i * ((s + 1)^(i-1) + (s + 1)^(i-2) + ... + 1)) in
  sum % (s - 1) = if s % 2 = 0 then 1 else (s + 1) / 2 := by
  sorry

end remainder_sum_base_s_plus_1_l622_622876


namespace sin_product_l622_622519

theorem sin_product (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.sin (π / 2 - α) = 2 / 5 :=
by
  -- proof shorter placeholder
  sorry

end sin_product_l622_622519


namespace isosceles_trapezoid_l622_622366

-- We define the necessary conditions
variables {A B C D : Point}
variables {BC AD AC BD : length}
variable {angle_A_C_B_D : ℝ}

-- We assume the conditions given in the problem
axiom trapezoid (h₁ : Trapezoid ABCD) : True
axiom base_lengths (h₂ : AC = BC + AD) : True
axiom angle_condition (h₃ : angle_A_C_B_D = 60) : True

-- We state the theorem to be proved
theorem isosceles_trapezoid
  (h₁ : trapezoid)
  (h₂ : base_lengths)
  (h₃ : angle_condition) :
  is_isosceles_trapezoid ABCD :=
sorry

end isosceles_trapezoid_l622_622366


namespace area_outside_semicircle_l622_622062

theorem area_outside_semicircle (a : ℝ) :
  let S_triangle := (a^2 * sqrt 3) / 4
  let S_MOB := S_triangle / 4
  let S_NOC := S_triangle / 4
  let radius := a / 2
  let S_sector_MNO := (1 / 6) * π * radius^2
  S_triangle - (2 * S_MOB + S_sector_MNO) = (a^2 * (3 * sqrt 3 - π)) / 24 :=
by
  let S_triangle := (a^2 * sqrt 3) / 4
  let S_MOB := S_triangle / 4
  let S_NOC := S_triangle / 4
  let radius := a / 2
  let S_sector_MNO := (1 / 6) * π * radius^2
  calc
    S_triangle - (2 * S_MOB + S_sector_MNO)
      = (a^2 * sqrt 3) / 4 - (2 * ((a^2 * sqrt 3) / 16) + ((π * (a / 2)^2) / 6)) : by sorry
    ... = (3 * a^2 * sqrt 3 - a^2 * π) / 24 : by sorry


end area_outside_semicircle_l622_622062


namespace positional_relationship_circles_l622_622549

theorem positional_relationship_circles : 
  let d := 3 in
  let f (x : ℝ) := x^2 - 5 * x + 3 in
  let r1 := (-5 + sqrt (25 - 4 * 3)) / 2 in
  let r2 := (-5 - sqrt (25 - 4 * 3)) / 2 in
  d < abs (r1 - r2) → (d < r1 ∨ d < r2) :=
by
  -- proof steps go here
  sorry

end positional_relationship_circles_l622_622549


namespace solution_set_of_inequality_l622_622548

noncomputable def f (x : ℝ) : ℝ := (1 / x) * (1 / 2 * (Real.log x) ^ 2 + 1 / 2)

theorem solution_set_of_inequality :
  (∀ x : ℝ, x > 0 → x < e → f x - x > f e - e) ↔ (∀ x : ℝ, 0 < x ∧ x < e) :=
by
  sorry

end solution_set_of_inequality_l622_622548


namespace polygon_sides_l622_622590

-- Definition for an interior angle given in the condition
def interior_angle (n : ℕ) : ℝ := 140

-- Definition for the exterior angle of a regular n-sided polygon
def exterior_angle (n : ℕ) : ℝ := 180 - interior_angle n

-- Definition for the number of sides given the exterior angle
def num_sides (ext_angle : ℝ) : ℕ := 360 / ext_angle

-- The theorem to be proved
theorem polygon_sides (n : ℕ) (h : interior_angle n = 140) : num_sides (exterior_angle n) = 9 :=
by sorry

end polygon_sides_l622_622590


namespace train_speed_l622_622043

-- Definition for the given conditions
def distance : ℕ := 240 -- distance in meters
def time_seconds : ℕ := 6 -- time in seconds
def conversion_factor : ℕ := 3600 -- seconds to hour conversion factor
def meters_in_km : ℕ := 1000 -- meters to kilometers conversion factor

-- The proof goal
theorem train_speed (d : ℕ) (t : ℕ) (cf : ℕ) (mk : ℕ) (h1 : d = distance) (h2 : t = time_seconds) (h3 : cf = conversion_factor) (h4 : mk = meters_in_km) :
  (d * cf / t) / mk = 144 :=
by sorry

end train_speed_l622_622043


namespace exists_alpha_l622_622349

noncomputable def seq_a : ℕ → ℝ
| 0     => 1
| (n+1) => real.sqrt (seq_a n ^ 2 + 1 / seq_a n)

theorem exists_alpha (n : ℕ) (h : n ≥ 1) : ∃ α : ℝ, ( α = 1 / 3 ) ∧ (1 / 2 ≤ seq_a n / n^α ∧ seq_a n / n^α ≤ 2) :=
by
  intros
  use (1 / 3)
  sorry

end exists_alpha_l622_622349


namespace zero_not_pronounced_in_reading_l622_622014

theorem zero_not_pronounced_in_reading {n : ℕ} (h : n = 3406000) : ∀ s, (reading_of_number n s) → (string.not_contains s '0') :=
sorry

end zero_not_pronounced_in_reading_l622_622014


namespace y_relation_l622_622954

theorem y_relation (y1 y2 y3 : ℝ) : 
  (-4, y1) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) ∧
  (-1, y2) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) ∧
  (2, y3) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) →
  y2 > y3 ∧ y3 > y1 :=
begin
  sorry
end

end y_relation_l622_622954


namespace simplify_expression_l622_622296

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622296


namespace correct_preparation_l622_622017

-- Define conditions as boolean variables representing each preparatory work
def preparatory_work_correct (A B C D : Prop) : Prop :=
  A ∧ ¬ B ∧ ¬ C ∧ ¬ D

-- Define the specific conditions for this problem
def correct_preparatory_work : Prop :=
  preparatory_work_correct 
    (heat_solidified_medium_to_melt_for_pouring_plates)
    (treat_dry_yeast_with_boiling_water_to_revive_vitality)
    (prepare_onion_epidermal_cells_for_observing_plant_mitosis)
    (prepare_boiling_water_bath_for_biuret_protein_test)

-- Prove the correct preparatory work is A
theorem correct_preparation : correct_preparatory_work :=
  sorry

end correct_preparation_l622_622017


namespace cos_diff_angle_l622_622585

theorem cos_diff_angle
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (h : 3 * Real.sin α = Real.tan α) :
  Real.cos (α - π / 4) = (4 + Real.sqrt 2) / 6 :=
sorry

end cos_diff_angle_l622_622585


namespace rearrange_columns_product_leq_original_l622_622207

variables (A : matrix (fin 3) (fin 24) ℕ)
variables (S : fin 3 → ℕ)
variables (p : ℕ)
variables (A' : matrix (fin 3) (fin 24) ℕ)
variables (S' : fin 3 → ℕ)
variables (p' : ℕ)

-- Define S, p for original matrix A
def row_sums (A : matrix (fin 3) (fin 24) ℕ) : fin 3 → ℕ
| i := ∑ j, A i j

def product_of_sums (S : fin 3 → ℕ) : ℕ :=
S 0 * S 1 * S 2

-- Define S', p' for rearranged matrix A'
def rearranged_column (A : matrix (fin 3) (fin 24) ℕ) (j : fin 24) : (fin 3 → ℕ) :=
let col := λ i, A i j in 
let sorted_col := col.sort (≤) in
λ i, sorted_col i

def rearrange_columns (A : matrix (fin 3) (fin 24) ℕ) : matrix (fin 3) (fin 24) ℕ :=
λ i j, rearranged_column A j i

def new_row_sums (A : matrix (fin 3) (fin 24) ℕ) : fin 3 → ℕ :=
row_sums (rearrange_columns A)

def new_product_of_sums (S' : fin 3 → ℕ) := 
product_of_sums S'

-- Main theorem
theorem rearrange_columns_product_leq_original :
  let S := row_sums A in
  let p := product_of_sums S in
  let A' := rearrange_columns A in
  let S' := new_row_sums A' in
  let p' := new_product_of_sums S' in
  p' ≤ p :=
begin
  sorry
end

end rearrange_columns_product_leq_original_l622_622207


namespace distance_centers_approx_14_3_l622_622218

-- Defining the basic structure of the isosceles triangle and known givens
structure IsoscelesTriangle where
  h : ℝ -- height to the base
  a : ℝ -- base
  b : ℝ -- sides
  P : ℝ -- perimeter
  h_eq : h = 5
  P_eq : P = 50
  P_rel : P = 2 * b + a
  height_rel : h ^ 2 + (a / 2) ^ 2 = b ^ 2

-- Circumradius calculation
def circumradius (t : IsoscelesTriangle) : ℝ :=
  let Δ := (1 / 2) * t.a * t.h
  (t.b * t.b * t.a) / (4 * Δ)

-- Inradius calculation
def inradius (t : IsoscelesTriangle) : ℝ :=
  let Δ := (1 / 2) * t.a * t.h
  Δ / ((t.a + t.b + t.b) / 2)

-- Distance between the centers of the inscribed and circumscribed circles (Euler's formula)
def center_distance (t : IsoscelesTriangle) : ℝ :=
  let R := circumradius t
  let r := inradius t
  real.sqrt (R * (R - 2 * r))

noncomputable def proof_problem : ℝ :=
  let t : IsoscelesTriangle := { 
    h := 5, 
    a := 24, 
    b := 13,
    P := 50, 
    h_eq := rfl, 
    P_eq := rfl, 
    P_rel := by sorry, 
    height_rel := by sorry
  }
  center_distance t

-- Ensuring the result is approximately 14.3
theorem distance_centers_approx_14_3 : abs (proof_problem - 14.3) < 0.1 :=
  by
    sorry

end distance_centers_approx_14_3_l622_622218


namespace green_peaches_per_basket_l622_622741

/-- Define the conditions given in the problem. -/
def n_baskets : ℕ := 7
def n_red_each : ℕ := 10
def n_green_total : ℕ := 14

/-- Prove that there are 2 green peaches in each basket. -/
theorem green_peaches_per_basket : n_green_total / n_baskets = 2 := by
  sorry

end green_peaches_per_basket_l622_622741


namespace area_of_region_l622_622499

noncomputable def area_enclosed_by_function : ℝ :=
  ∫ x in -1..1, (1 - x^2)

theorem area_of_region :
  area_enclosed_by_function = 4 / 3 :=
by
  sorry

end area_of_region_l622_622499


namespace calculate_f_f_f_one_l622_622587

def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem calculate_f_f_f_one : f (f (f 1)) = 9184 :=
by
  sorry

end calculate_f_f_f_one_l622_622587


namespace max_teams_participate_l622_622601

theorem max_teams_participate (teams players games : ℕ) 
  (h_players_per_team : ∀ t, t ∈ teams → players = 3)
  (h_game_between_players : ∀ p1 p2, p1 ∈ players → p2 ∈ players → p1 ≠ p2 → games = players * players)
  (h_max_games : ∀ g, g ∈ games → g ≤ 150) :
  teams = 6 :=
by
  sorry

end max_teams_participate_l622_622601


namespace cube_expression_l622_622581

theorem cube_expression (x : ℝ) (h : sqrt (x + 2) = 3) : (x + 2)^3 = 729 := by
  sorry

end cube_expression_l622_622581


namespace sarah_total_distance_l622_622286

theorem sarah_total_distance :
  let time_to_school := 15 / 60.0
      time_to_home := 25 / 60.0
      average_rate := 10.0 in
  time_to_school + time_to_home = 2.0 / 3.0 ∧
  average_rate = 10.0 →
  average_rate * (time_to_school + time_to_home) = 20.0 / 3.0 :=
by
  sorry

end sarah_total_distance_l622_622286


namespace euler_totient_product_find_primes_l622_622656

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def coprime (a b : ℕ) : Prop := gcd a b = 1

def euler_totient (n : ℕ) : ℕ := (Finset.range n).filter (λ m, coprime m n).card

theorem euler_totient_product (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (hpq : p ≠ q) : 
  euler_totient (p * q) = (p - 1) * (q - 1) := 
sorry

theorem find_primes (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (hpq : p ≠ q)
  (h : euler_totient (p * q) = 3 * p + q) : 
  (p = 3 ∧ q = 11) ∨ (p = 11 ∧ q = 3) := 
sorry

end euler_totient_product_find_primes_l622_622656


namespace facebook_mothers_bonus_condition_l622_622883

theorem facebook_mothers_bonus_condition 
  (annual_earnings : ℕ)
  (bonus_percentage : ℚ)
  (total_employees : ℕ)
  (male_fraction : ℚ)
  (bonus_per_mother : ℚ)
  (female_not_mothers : ℕ) :
  annual_earnings = 5000000 →
  bonus_percentage = 0.25 →
  total_employees = 3300 →
  male_fraction = 1/3 →
  bonus_per_mother = 1250 →
  let number_mothers := (0.25 * annual_earnings) / bonus_per_mother in
  let males := male_fraction * total_employees in
  let females := total_employees - males in
  female_not_mothers = females - number_mothers →
  female_not_mothers = 1200 :=
begin
  intros,
  let number_mothers := (0.25 * annual_earnings) / bonus_per_mother,
  let males := male_fraction * total_employees,
  let females := total_employees - males,
  have : female_not_mothers = females - number_mothers, from ‹female_not_mothers = females - number_mothers›,
  calc female_not_mothers
      = females - number_mothers : by assumption
  ... = (total_employees - (male_fraction * total_employees)) - ((0.25 * annual_earnings) / bonus_per_mother) : by simp
  ... = (3300 - (1/3 * 3300)) - ((0.25 * 5000000) / 1250) : by assumption
  ... = 2200 - 1000 : by norm_num
  ... = 1200 : by norm_num,
end

end facebook_mothers_bonus_condition_l622_622883


namespace correct_conclusions_l622_622980

-- Define the conditions
variables (A B C D P Pa Pc Pd : point) 
variables (E F : point)
variables (is_rectangle : is_rectangle A B C D)
variables (is_perpendicular : is_perpendicular P A B C D)
variables (midpoint_pc : is_midpoint E P C)
variables (midpoint_pd : is_midpoint F P D)

-- Define statements to prove
def statement_1 : Prop := is_perpendicular A B P D
def statement_2 : Prop := is_perpendicular_plane P B C A B C D
def statement_3 : Prop := area P C D > area P A B
def statement_4 : Prop := are_skew_lines A E B F

-- The final Lean 4 statement
theorem correct_conclusions :
  (statement_1 ∧ statement_3) ∧ ¬ statement_2 ∧ ¬ statement_4 := sorry

end correct_conclusions_l622_622980


namespace rectangle_area_in_inscribed_triangle_l622_622428

theorem rectangle_area_in_inscribed_triangle (b h x : ℝ) (hb : 0 < b) (hh : 0 < h) (hx : 0 < x) (hxh : x < h) :
  ∃ (y : ℝ), y = (b * (h - x)) / h ∧ (x * y) = (b * x * (h - x)) / h :=
by
  sorry

end rectangle_area_in_inscribed_triangle_l622_622428


namespace factor_difference_of_squares_l622_622896
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l622_622896


namespace total_sandwiches_l622_622447

theorem total_sandwiches (billy : ℕ) (katelyn_more : ℕ) (katelyn_quarter : ℕ) :
  billy = 49 → katelyn_more = 47 → katelyn_quarter = 4 → 
  billy + (billy + katelyn_more) + ((billy + katelyn_more) / katelyn_quarter) = 169 :=
by
  intros hb hk hq
  rw [hb, hk, hq]
  calc
    49 + (49 + 47) + ((49 + 47) / 4) = 49 + 96 + 24 : by simp
                                ... = 169 : by simp

sorry

end total_sandwiches_l622_622447


namespace find_difference_l622_622639

-- Define the conditions
variables {S : ℕ} {automobile_cost : ℕ}
def total_spending : Prop := S + automobile_cost = 450
def cost_of_automobile : Prop := automobile_cost = 350

-- Statement that needs to be proved
theorem find_difference (h1 : total_spending) (h2 : cost_of_automobile) :
  350 - 3 * S = 50 :=
by
  sorry

end find_difference_l622_622639


namespace ellipse_equation_l622_622975

theorem ellipse_equation (F1 F2 : 𝔸^2) (A B : 𝔸^2)
  (hF1 : F1 = (-1, 0))
  (hF2 : F2 = (1, 0))
  (hf : line_through F2 A B)
  (hAF2 : dist A F2 = 2 * dist F2 B)
  (hAB : dist A B = dist B F1) :
  ( ∀ p : 𝔸^2, p ∈ C ↔ (p.1^2 / 3) + (p.2^2 / 2) = 1 ) :=
sorry

end ellipse_equation_l622_622975


namespace pentagonal_prism_lateral_angle_l622_622846

theorem pentagonal_prism_lateral_angle (φ : ℝ) 
  (h1 : ∃ P : Set ℝ^3, is_pentagonal_prism P)
  (h2 : ∀ F, is_lateral_face F P → is_parallelogram F → ∃ φ, φ = 90): 
  φ = 90 := 
sorry

end pentagonal_prism_lateral_angle_l622_622846


namespace find_principal_sum_l622_622708

theorem find_principal_sum (P R : ℝ) (SI CI : ℝ) 
  (h1 : SI = 10200) 
  (h2 : CI = 11730) 
  (h3 : SI = P * R * 2 / 100)
  (h4 : CI = P * (1 + R / 100)^2 - P) :
  P = 17000 :=
by
  sorry

end find_principal_sum_l622_622708


namespace factor_t_sq_minus_64_l622_622888

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l622_622888


namespace angle_sum_circle_l622_622279

theorem angle_sum_circle (arc_RS arc_BR : ℝ) (h_RS : arc_RS = 62) (h_BR : arc_BR = 58) :
  let X := arc_RS / 2 in
  let Y := arc_BR / 2 in
  X + Y = 60 :=
by
  -- Proof would go here
  sorry

end angle_sum_circle_l622_622279


namespace roots_exist_range_k_l622_622558

theorem roots_exist_range_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, (2 * k * x1^2 + (8 * k + 1) * x1 + 8 * k = 0) ∧ 
                 (2 * k * x2^2 + (8 * k + 1) * x2 + 8 * k = 0)) ↔ 
  (k ≥ -1/16 ∧ k ≠ 0) :=
sorry

end roots_exist_range_k_l622_622558


namespace ratio_AC_AD_l622_622280

-- Define the necessary points and properties of the parallelogram and midpoints.
variables {A B C D K L M : Type} 

-- Conditions
def is_midpoint (X A B : Type) := (dist X A) = (dist X B)
def is_parallelogram (ABCD : Type) := true -- Needs a proper definition of parallelogram

-- Embed conditions within the theorem statement
theorem ratio_AC_AD 
  (ABCD : Type)
  (parallelogram_ABCD : is_parallelogram ABCD)
  (K_mid_AB : is_midpoint K A B)
  (L_mid_BC : is_midpoint L B C)
  (M_mid_CD : is_midpoint M C D)
  (inscribed_KBLM : inscribed K B L M)
  (inscribed_BCDK : inscribed B C D K) :
  measure (AC) / measure (AD) = 2 :=
    sorry

end ratio_AC_AD_l622_622280


namespace find_QR_squared_l622_622219

noncomputable def isosceles_trapezoid (PQ RS : ℝ) (QR PS PR QS : ℝ) : Prop :=
  QR > 0 ∧ 
  PS > 0 ∧ 
  QR ⊥ PQ ∧ 
  QR ⊥ RS ∧ 
  PR ⊥ QS ∧ 
  PQ = RS / 2 ∧ 
  PS = √2058

theorem find_QR_squared 
  (PQ RS QR PS PR QS : ℝ)
  (cond1 : isosceles_trapezoid PQ RS QR PS PR QS) 
  (cond2 : PQ = √14)
  (cond3 : PS = √2058) : QR^2 = 294 :=
  sorry

end find_QR_squared_l622_622219


namespace find_ab_value_l622_622033

theorem find_ab_value (a b : ℤ)
  (h1 : 3 * Real.sqrt (Real.cbrt 5 - Real.cbrt 4) = Real.cbrt a + Real.cbrt b + Real.cbrt 2)
  (ha : a = 20)
  (hb : b = -25) : 
  a * b = -500 := by
  sorry

end find_ab_value_l622_622033


namespace divisor_of_number_l622_622821

theorem divisor_of_number (n d q p : ℤ) 
  (h₁ : n = d * q + 3)
  (h₂ : n ^ 2 = d * p + 3) : 
  d = 6 := 
sorry

end divisor_of_number_l622_622821


namespace range_of_k_l622_622158

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 
  f (-x)
else if x ≤ 2 then 
  abs (x^2 - 1)
else 
  f (x - 1)

def g (x k : ℝ) : ℝ := f x - k * (x - 1)

theorem range_of_k 
  (h: ∀ k, (∃ x₁ x₂ x₃ x₄ : ℝ, g x₁ k = 0 ∧ g x₂ k = 0 ∧ g x₃ k = 0 ∧ g x₄ k = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)) : 
  { k : ℝ | ∃! (n : ℕ), ∃ (x : ℝ), g x k = 0 ∧ 0 ≤ x ∧ x < n } = {(k : ℝ) | (-3/4 ≤ k ∧ k < -3/5) ∨ (3/4 < k ∧ k ≤ 1)} :=
sorry

end range_of_k_l622_622158


namespace chord_through_P_maximizing_angle_AP_trajectory_point_M_l622_622140

/-- 
Given point P(1,1) inside the circle C with equation (x-2)^2+(y-2)^2=8, and a chord AB passing through P,
1. When P(1,1) and |AB|=2√7, the equation of the line containing chord AB is x=1 or y=1
2. When P(1,1), the equation of line AP maximizing the angle ∠PAC is y=-x+2 
3. The trajectory equation for moving point M, where tangent lines at points A and B intersect, is (x-2)*(x'-2)+(y-2)*(y'-2)=8 
-/
structure Point (α : Type) where
  x : α
  y : α

def circle_eq {α : Type} [Field α] (x y : α) : α := (x - 2) ^ 2 + (y - 2) ^ 2

noncomputable def line_eq1 {α : Type} [Field α] (x y : α) : Prop :=
x = 1 ∨ y = 1

noncomputable def line_eq2 {α : Type} [Field α] (x y : α) : Prop :=
y = -x + 2

noncomputable def trajectory_eq {α : Type} [Field α] (x y x' y' : α) : Prop :=
(x - 2) * (x' - 2) + (y - 2) * (y' - 2) = 8

theorem chord_through_P (x y : ℝ) (h : circle_eq x y = 8) (h1 : x = 1 ∧ y = 1) (h2 : 2 * sqrt 7 = 2 * sqrt 7) :
  line_eq1 x y := by
  sorry

theorem maximizing_angle_AP (x y : ℝ) (h : circle_eq x y = 8) (h1 : x = 1 ∧ y = 1) : 
  line_eq2 x y := by
  sorry

theorem trajectory_point_M (x y x' y' : ℝ) (h : circle_eq x y = 8) (h1 : (x - 2) * (x' - 2) + (y - 2) * (y' - 2) = 8) : 
  trajectory_eq x y x' y' := by
  sorry

end chord_through_P_maximizing_angle_AP_trajectory_point_M_l622_622140


namespace abundant_count_under_35_l622_622579

def is_prime (n : ℕ) : Prop := nat.prime n

def proper_divisors (n : ℕ) : finset ℕ := (finset.range n).filter (λ d, d > 0 ∧ n % d = 0)

def is_abundant (n : ℕ) : Prop := proper_divisors n.sum > n

noncomputable def abundant_numbers_under (m : ℕ) : finset ℕ :=
(finset.range m).filter (λ n, ¬ is_prime n ∧ is_abundant n)

theorem abundant_count_under_35 : abundant_numbers_under 35.card = 5 :=
by sorry

end abundant_count_under_35_l622_622579


namespace correct_statement_l622_622651

variables {Line Plane : Type}
variables (m n : Line) (alpha beta : Plane)

-- Define the relations
variables (parallel : Line → Plane → Prop)
variables (subset : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)
variables (parallel_line : Line → Line → Prop)
variables (perpendicular_line : Line → Line → Prop)
variables (intersection : Plane → Plane → Line)

-- Input hypothesis and expected result
theorem correct_statement (hm_perp_alpha : perpendicular m alpha) 
  (hm_parallel_n : parallel_line m n)
  (hn_subset_beta : subset n beta) :
  perpendicular alpha beta :=
sorry

end correct_statement_l622_622651


namespace power_identity_l622_622187

theorem power_identity (x : ℕ) (h : 2^x = 16) : 2^(x + 3) = 128 := 
sorry

end power_identity_l622_622187


namespace greatest_symmetry_lines_circle_l622_622015

theorem greatest_symmetry_lines_circle :
  ∀ (lines_equilateral_triangle lines_semi_circle lines_ellipse lines_rectangle : ℕ),
  (lines_equilateral_triangle = 3) →
  (lines_semi_circle = 1) →
  (lines_ellipse = 2) →
  (lines_rectangle = 2) →
  (∀ n, n < ∞ ) → "infinite" > lines_equilateral_triangle ∧ "infinite" > lines_semi_circle ∧ "infinite" > lines_ellipse ∧ "infinite" > lines_rectangle :=
by
  intros lines_equilateral_triangle lines_semi_circle lines_ellipse lines_rectangle h_eq_tri h_semi_circle h_ellipse h_rectangle h_infinite
  sorry

end greatest_symmetry_lines_circle_l622_622015


namespace trig_identity_cos_sin_15_cos_30_equals_sqrt3_div_2_solution_l622_622847

theorem trig_identity_cos_sin_15 (θ : ℝ) (hθ : θ = 15) : cos (2 * θ) = cos θ ^ 2 - sin θ ^ 2 := 
by sorry

theorem cos_30_equals_sqrt3_div_2 : cos (30 : ℝ) = (sqrt 3) / 2 := 
by sorry

theorem solution : (cos (15 : ℝ))^2 - (sin (15 : ℝ))^2 = (sqrt 3) / 2 := 
by 
  rw <-trig_identity_cos_sin_15 15 rfl
  exact cos_30_equals_sqrt3_div_2

end trig_identity_cos_sin_15_cos_30_equals_sqrt3_div_2_solution_l622_622847


namespace bob_after_alice_l622_622208

def race_distance : ℕ := 15
def alice_speed : ℕ := 7
def bob_speed : ℕ := 9

def alice_time : ℕ := alice_speed * race_distance
def bob_time : ℕ := bob_speed * race_distance

theorem bob_after_alice : bob_time - alice_time = 30 := by
  sorry

end bob_after_alice_l622_622208


namespace expenditure_estimate_l622_622430

theorem expenditure_estimate (a : ℝ) 
  (h_a : a = -1.2)
  (x : ℝ) 
  (h_x : x = 7) 
  (h_eq : ∀ x, 0.8 * x + a = 0.8 * x - 1.2) : 
  0.8 * 7 + a = 4.4 :=
by 
  rw [h_a, h_x]
  norm_num
  sorry

end expenditure_estimate_l622_622430


namespace no_solution_in_Q5_sqrt_2_irrational_l622_622290

theorem no_solution_in_Q5 (x : ℚ_5) : ¬(x^2 = 2) := 
by
  sorry

theorem sqrt_2_irrational : ¬(∃ x : ℚ, x^2 = 2) :=
by
  sorry

end no_solution_in_Q5_sqrt_2_irrational_l622_622290


namespace polynomial_value_at_minus_two_l622_622127

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem polynomial_value_at_minus_two :
  f (-2) = -1 :=
by sorry

end polynomial_value_at_minus_two_l622_622127


namespace factor_difference_of_squares_l622_622898
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l622_622898


namespace exists_removal_one_way_road_with_retained_connectivity_l622_622803

theorem exists_removal_one_way_road_with_retained_connectivity (N : ℕ) (cities : Finset ℕ) (roads : Finset (ℕ × ℕ)) (h_cities : cities.card = N) (h_roads : roads.card = 2 * N - 1) 
  (h_connected : ∀ (a b : ℕ), a ∈ cities → b ∈ cities → (∃ (path : list (ℕ × ℕ)), path ∈ roads.to_list ∧ (list.chain' (λ (p : ℕ × ℕ), p.2 = p.1 → true) path) )) :
  ∃ road ∈ roads, ∀ (a b : ℕ), a ∈ cities → b ∈ cities → (∃ path, (path ∈ (roads.erase road).to_list) ∧ list.chain' (λ (p : ℕ × ℕ), p.snd = p.fst → true) path) :=
sorry

end exists_removal_one_way_road_with_retained_connectivity_l622_622803


namespace minimum_value_of_f_l622_622116

def f (x : ℝ) : ℝ := 3*x^2 - 12*x + 2023

theorem minimum_value_of_f : ∃ x₀, ∀ x, f(x₀) ≤ f(x) ∧ f(x₀) = 2007 :=
by
  sorry

end minimum_value_of_f_l622_622116


namespace smallest_x_l622_622773

theorem smallest_x (x : ℤ) (h : x + 3 < 3 * x - 4) : x = 4 :=
by
  sorry

end smallest_x_l622_622773


namespace minimum_value_of_P_over_Q_l622_622864

def P(x : ℝ) : ℝ := 16 * x^4 + 40 * x^3 + 41 * x^2 + 20 * x + 16
def Q(x : ℝ) : ℝ := 4 * x^2 + 5 * x + 2

theorem minimum_value_of_P_over_Q :
  ∀ a : ℝ, P(a) / Q(a) ≥ 4 * Real.sqrt 3 :=
by
  sorry

end minimum_value_of_P_over_Q_l622_622864


namespace magnitude_of_angle_A_max_area_l622_622203

variables {A B C a b c : ℝ}

-- Given conditions
def conditions (a b c A B C : ℝ) : Prop :=
  (sqrt 3 * a * cos C = (2 * b - sqrt 3 * c) * cos A)

-- Prove the magnitude of angle A
theorem magnitude_of_angle_A (h: conditions a b c A B C) : A = π / 6 := 
by sorry

-- Maximum area when a = 2
noncomputable def area (b c : ℝ) : ℝ :=
  (1 / 2) * b * c * sin (π / 6)

theorem max_area (h: conditions 2 b c (π / 6) B C) (a_eq: a = 2) :
  area b c = 2 + sqrt 3 :=
by sorry

end magnitude_of_angle_A_max_area_l622_622203


namespace range_for_y_l622_622418

def value_in_range (b : Fin 15 → ℕ) : ℝ :=
  ∑ n in Finset.range 15, (b n) * (1 / (2:ℝ)^((2 * n) + 1))

theorem range_for_y (b : Fin 15 → ℕ) (h : ∀ n, b n = 0 ∨ b n = 3) :
  1 ≤ value_in_range b ∧ value_in_range b < 2 :=
sorry

end range_for_y_l622_622418


namespace diagonals_from_vertex_l622_622200

theorem diagonals_from_vertex (n : ℕ) (h : (n-2) * 180 + 360 = 1800) : (n - 3) = 7 :=
sorry

end diagonals_from_vertex_l622_622200


namespace unique_triple_sum_l622_622357

theorem unique_triple_sum :
  ∃ (a b c : ℕ), 
    (10 ≤ a ∧ a < 100) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (10 ≤ c ∧ c < 100) ∧ 
    (a^3 + 3 * b^3 + 9 * c^3 = 9 * a * b * c + 1) ∧ 
    (a + b + c = 9) := 
sorry

end unique_triple_sum_l622_622357


namespace factor_t_sq_minus_64_l622_622889

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l622_622889


namespace sum_of_valid_n_l622_622009

theorem sum_of_valid_n :
  let lcm (a b : ℕ) : ℕ := Nat.lcm a b
  let gcd (a b : ℕ) : ℕ := Nat.gcd a b
  (∑ n in Finset.filter (λ n => lcm n 100 = gcd n 100 + 900) (Finset.range 1001)) = 1800 :=
by
  sorry

end sum_of_valid_n_l622_622009


namespace remainder_of_k_divided_by_7_l622_622795

theorem remainder_of_k_divided_by_7 :
  ∃ k < 42, k % 5 = 2 ∧ k % 6 = 5 ∧ k % 7 = 3 :=
by {
  -- The proof is supplied here
  sorry
}

end remainder_of_k_divided_by_7_l622_622795


namespace factor_t_sq_minus_64_l622_622890

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l622_622890


namespace sum_of_divisors_57_eq_80_l622_622781

def sum_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).sum

theorem sum_of_divisors_57_eq_80 : sum_of_divisors 57 = 80 := by
  sorry

end sum_of_divisors_57_eq_80_l622_622781


namespace arithmetic_sequence_sum_l622_622199

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h : S 7 = 77) 
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2) : 
  a 4 = 11 :=
by
  sorry

end arithmetic_sequence_sum_l622_622199


namespace hyewon_painted_colors_l622_622179

def pentagonal_prism := 
  let num_rectangular_faces := 5 
  let num_pentagonal_faces := 2
  num_rectangular_faces + num_pentagonal_faces

theorem hyewon_painted_colors : pentagonal_prism = 7 := 
by
  sorry

end hyewon_painted_colors_l622_622179


namespace debra_gets_three_heads_l622_622098

noncomputable def debra_probability : ℝ :=
  let p_start_sequence := (1/2)^5 in -- Probability of starting with THTHT
  let Q := 1 / 6 in -- Probability of getting three heads in a row after THTHT
  p_start_sequence * Q -- The final required probability

theorem debra_gets_three_heads :
  debra_probability = 1 / 192 := by
  unfold debra_probability
  unfold p_start_sequence
  unfold Q
  sorry

end debra_gets_three_heads_l622_622098


namespace value_of_z_l622_622182

theorem value_of_z (x y z : ℚ) (h1 : x - y - z = 8) (h2 : x + y + z = 20) (h3 : x - y + 2z = 16) : 
  z = 8 / 3 := 
by 
  sorry

end value_of_z_l622_622182


namespace regular_tetrahedron_l622_622623

-- Define the centers of the spheres and the radii of the respective spheres
variables {A1 A2 A3 A4 O : Type} [metric_space A1] [metric_space A2] [metric_space A3] [metric_space A4] [metric_space O]
variables (r R : ℝ)

-- Define the conditions of the problem as hypotheses
variables (h1 : ∀ i j, i ≠ j → dist (sphere_center i) (sphere_center j) = (radius i) + (radius j))
variables (h2 : ∀ i, dist O (sphere_center i) = r + radius i)
variables (h3 : ∀ {P Q}, P ∈ edge_set (tetrahedron A1 A2 A3 A4) → dist O (midpoint P Q) = R)

-- Statement of the problem
theorem regular_tetrahedron (A1 A2 A3 A4 O : Type) [metric_space A1] [metric_space A2] [metric_space A3] [metric_space A4] [metric_space O]
(r R : ℝ) (h1 : ∀ i j, i ≠ j → dist (sphere_center i) (sphere_center j) = (radius i) + (radius j))
(h2 : ∀ i, dist O (sphere_center i) = r + radius i)
(h3 : ∀ {P Q}, P ∈ edge_set (tetrahedron A1 A2 A3 A4) → dist O (midpoint P Q) = R) :
is_regular_tetrahedron (tetrahedron A1 A2 A3 A4) :=
sorry

end regular_tetrahedron_l622_622623


namespace rolls_remaining_to_sell_l622_622921

-- Conditions
def total_rolls_needed : ℕ := 45
def rolls_sold_to_grandmother : ℕ := 1
def rolls_sold_to_uncle : ℕ := 10
def rolls_sold_to_neighbor : ℕ := 6

-- Theorem statement
theorem rolls_remaining_to_sell : (total_rolls_needed - (rolls_sold_to_grandmother + rolls_sold_to_uncle + rolls_sold_to_neighbor) = 28) :=
by
  sorry

end rolls_remaining_to_sell_l622_622921


namespace a_seq_correct_b_seq_max_m_l622_622142

noncomputable def a_seq (n : ℕ) : ℕ :=
if n = 0 then 3 else (n + 1)^2 + 2

-- Verification that the sequence follows the provided conditions.
theorem a_seq_correct (n : ℕ) : 
  (a_seq 0 = 3) ∧
  (a_seq 1 = 6) ∧
  (a_seq 2 = 11) ∧
  (∀ m : ℕ, m ≥ 1 → a_seq (m + 1) - a_seq m = 2 * m + 1) := sorry

noncomputable def b_seq (n : ℕ) : ℝ := 
(a_seq n : ℝ) / (3 ^ (Real.sqrt (a_seq n - 2)))

theorem b_seq_max_m (m : ℝ) : 
  (∀ n : ℕ, b_seq n ≤ m) ↔ (1 ≤ m) := sorry

end a_seq_correct_b_seq_max_m_l622_622142


namespace parabola_vertex_eq_l622_622333

theorem parabola_vertex_eq :
  (∃ c : ℝ, (∀ x : ℝ, y = 2 * x^2 + c) ∧ y = 1 ∧ x = 0) → c = 1 :=
by
  intro h
  choose c hc using h
  specialize hc 0
  rw [mul_zero, zero_mul, add_zero] at hc
  cases hc
  rw [hc_right]
  exact hc_left

end parabola_vertex_eq_l622_622333


namespace stationary_tank_radius_l622_622817

noncomputable def radius_of_stationary_tank : ℝ :=
let
  height_stationary := 25,
  height_drop := 0.049,
  radius_truck := 7,
  height_truck := 10,
  volume_truck := Real.pi * radius_truck^2 * height_truck,
  volume_pumped := Real.pi * (100:ℝ)^2 * height_drop
in
100

theorem stationary_tank_radius :
  radius_of_stationary_tank = 
  let
    height_stationary := 25,
    height_drop := 0.049,
    radius_truck := 7,
    height_truck := 10,
    volume_truck := Real.pi * radius_truck^2 * height_truck,
    volume_pumped := Real.pi * (100:ℝ)^2 * height_drop
  in
  100 := sorry

end stationary_tank_radius_l622_622817


namespace simplify_expr_l622_622301

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l622_622301


namespace isosceles_triangle_base_length_l622_622202

theorem isosceles_triangle_base_length (a b : ℝ) (h : a = 4 ∧ b = 4) : a + b = 8 :=
by
  sorry

end isosceles_triangle_base_length_l622_622202


namespace find_t_value_l622_622575

variable {a b : Vector} [UnitVector a] [UnitVector b]
variable (t : ℝ)
variable (c : Vector)
variable (angle_ab : angle a b = 60)
variable (c_eq : c = (1 - t) • a + t • b)
variable (dot_prod_cond : b ⬝ c = -1/2)

theorem find_t_value : t = -2 := by
  sorry

end find_t_value_l622_622575


namespace mixed_number_calculation_l622_622450

theorem mixed_number_calculation :
  47 * (4 + 3/7 - (5 + 1/3)) / (3 + 1/2 + (2 + 1/5)) = -7 - 119/171 := by
  sorry

end mixed_number_calculation_l622_622450


namespace remainder_x2023_plus_1_l622_622915

noncomputable def remainder (a b : Polynomial ℂ) : Polynomial ℂ :=
a % b

theorem remainder_x2023_plus_1 :
  remainder (Polynomial.X ^ 2023 + 1) (Polynomial.X ^ 8 - Polynomial.X ^ 6 + Polynomial.X ^ 4 - Polynomial.X ^ 2 + 1) =
  - Polynomial.X ^ 3 + 1 :=
by
  sorry

end remainder_x2023_plus_1_l622_622915


namespace factor_difference_of_squares_l622_622899
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l622_622899


namespace rational_root_of_polynomial_l622_622872

-- Polynomial definition
def P (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 + 4 * x^2 + 6 * x - 8

-- Theorem statement
theorem rational_root_of_polynomial : ∀ x : ℚ, P x = 0 ↔ x = -1 :=
by
  sorry

end rational_root_of_polynomial_l622_622872


namespace sample_size_l622_622046

theorem sample_size (k n : ℕ) (h_ratio : 4 * k + k + 5 * k = n) 
  (h_middle_aged : 10 * (4 + 1 + 5) = n) : n = 100 := 
by
  sorry

end sample_size_l622_622046


namespace modulus_of_complex_number_eq_sqrt10_l622_622346

theorem modulus_of_complex_number_eq_sqrt10 :
  ∀ (a b : ℝ), a = 1 → b = 3 → complex.abs (complex.mk a b) = real.sqrt 10 :=
by
  intros a b ha hb
  rw [ha, hb]
  unfold complex.abs
  simp
  sorry

end modulus_of_complex_number_eq_sqrt10_l622_622346


namespace total_money_taken_in_l622_622701

-- Define the conditions as constants
def total_tickets : ℕ := 800
def advanced_ticket_price : ℝ := 14.5
def door_ticket_price : ℝ := 22.0
def door_tickets_sold : ℕ := 672
def advanced_tickets_sold : ℕ := total_tickets - door_tickets_sold
def total_revenue_advanced : ℝ := advanced_tickets_sold * advanced_ticket_price
def total_revenue_door : ℝ := door_tickets_sold * door_ticket_price
def total_revenue : ℝ := total_revenue_advanced + total_revenue_door

-- State the mathematical proof problem
theorem total_money_taken_in : total_revenue = 16640.00 := by
  sorry

end total_money_taken_in_l622_622701


namespace count_and_largest_special_numbers_l622_622560

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_four_digit_number (n : ℕ) : Prop := 
  1000 ≤ n ∧ n < 10000

theorem count_and_largest_special_numbers :
  ∃ (nums : List ℕ), 
    (∀ n ∈ nums, ∃ x y : ℕ, is_prime x ∧ is_prime y ∧ 
      55 * x * y = n ∧ is_four_digit_number (n * 5))
    ∧ nums.length = 3
    ∧ nums.maximum = some 4785 :=
sorry

end count_and_largest_special_numbers_l622_622560


namespace operation_IV_determined_l622_622215

noncomputable def operation_durations (I_ II_ III_ IV_ : ℕ) : Prop :=
  I + II + III + IV = 152 ∧
  30 + (if I < 30 then I else 30) +
  (if II < 30 then II else 30) + 
  (if III < 30 then III else 30) + 
  (if IV < 30 then IV else 30) = 52 ∧
  10 + 30 + 
  (if I < 40 then I else 40) + 
  (if II < 40 then II else 40) + 
  (if III < 40 then III else 40) + 
  (if IV < 40 then IV else 40) = 82

theorem operation_IV_determined :
  ∃ I II III IV, 
  operation_durations I II III IV ∧
  ∀ IV', operation_durations I II III IV' → IV' = 10 := 
sorry

end operation_IV_determined_l622_622215


namespace upper_limit_arun_weight_l622_622080

theorem upper_limit_arun_weight (x w : ℝ) :
  (65 < w ∧ w < x) ∧
  (60 < w ∧ w < 70) ∧
  (w ≤ 68) ∧
  (w = 67) →
  x = 68 :=
by
  sorry

end upper_limit_arun_weight_l622_622080


namespace probability_sqrt_64x_136_l622_622248
noncomputable section

def real_numbers := Set.Icc 300 400
def selected_number (x : ℝ) := x ∈ real_numbers ∧ ⌊sqrt x⌋ = 17
def sqrt_64x_floor (x : ℝ) := ⌊sqrt (64 * x)⌋

theorem probability_sqrt_64x_136 (hx : (∃ x, selected_number x)) :
  (∀ x, selected_number x → sqrt_64x_floor x = 136) -> 
  ∃ p : ℝ, p = 941 / 7040 :=
by
  sorry

end probability_sqrt_64x_136_l622_622248


namespace nested_g_of_3_l622_622186

def g (x : ℝ) : ℝ := -1 / (x ^ 2)

theorem nested_g_of_3 :
  g (g (g (g (g 3)))) = -1 / (3 ^ 64) :=
by
  sorry

end nested_g_of_3_l622_622186


namespace volume_prism_is_correct_ak_length_is_correct_l622_622721

def radius : ℝ := Real.sqrt (35 / 3)
def height_prism : ℝ := 12
def side_length_base : ℝ := 2 * Real.sqrt 35
def area_of_base : ℝ := (Real.sqrt 3 / 4) * side_length_base^2
def volume_of_prism : ℝ := area_of_base * height_prism
def ak_length_1 : ℝ := 4
def ak_length_2 : ℝ := 8

theorem volume_prism_is_correct : volume_of_prism = 420 * Real.sqrt 3 := 
by
  sorry

theorem ak_length_is_correct (AK : ℝ) : AK = ak_length_1 ∨ AK = ak_length_2 :=
by
  sorry

end volume_prism_is_correct_ak_length_is_correct_l622_622721


namespace max_value_y_l622_622938

theorem max_value_y (x : ℝ) (h : x < 5 / 4) : 
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 :=
sorry

end max_value_y_l622_622938


namespace comparison_l622_622137

def f (x : ℝ) : ℝ := if x >= 0 then -x^2 - 2 * x else -x + 1 / 2

def a : ℝ := f ((1 / 2)^(1 / 3))
def b : ℝ := Real.logBase (1 / 2) (1 / 3)
def c : ℝ := f ((1 / 3)^(1 / 2))

theorem comparison : b < a ∧ a < c :=
by
  sorry

end comparison_l622_622137


namespace nth_equation_pattern_l622_622516

theorem nth_equation_pattern (n : ℕ) : 
  (∑ k in finset.range (2 * n - 1), (n + k)) = (2 * n - 1) ^ 2 :=
sorry

end nth_equation_pattern_l622_622516


namespace sum_of_odd_position_terms_l622_622422

theorem sum_of_odd_position_terms
  (n : ℕ)
  (a d : ℕ)
  (sum_terms : ℕ)
  (h1 : n = 1500)
  (h2 : d = 2)
  (h3 : sum_terms = 7500)
  (sequence : Fin n → ℕ)
  (h_seq : ∀ i : Fin (n - 1), sequence i.succ = sequence i + d)
  (h_sum : Finset.univ.sum sequence = sum_terms) :
  Finset.univ.filter (λ i : Fin n, i.val % 2 = 0).sum sequence = 3000 :=
by
  sorry

end sum_of_odd_position_terms_l622_622422


namespace max_small_boxes_l622_622792

-- Define the dimensions of the larger box in meters
def large_box_length : ℝ := 6
def large_box_width : ℝ := 5
def large_box_height : ℝ := 4

-- Define the dimensions of the smaller box in meters
def small_box_length : ℝ := 0.60
def small_box_width : ℝ := 0.50
def small_box_height : ℝ := 0.40

-- Calculate the volume of the larger box
def large_box_volume : ℝ := large_box_length * large_box_width * large_box_height

-- Calculate the volume of the smaller box
def small_box_volume : ℝ := small_box_length * small_box_width * small_box_height

-- State the theorem to prove the maximum number of smaller boxes that can fit in the larger box
theorem max_small_boxes : large_box_volume / small_box_volume = 1000 :=
by
  sorry

end max_small_boxes_l622_622792


namespace seventh_root_of_c_is_102_l622_622476

-- Define the conditions in Lean
def c : ℕ := 218618940381251
def lhs : ℕ := (101 + 1)^7

-- State the theorem: given these conditions, we prove the question
theorem seventh_root_of_c_is_102 (h : c = lhs) : Nat.root 7 c = 102 :=
by
  sorry

end seventh_root_of_c_is_102_l622_622476


namespace marissa_lunch_calories_l622_622662

theorem marissa_lunch_calories :
  (1 * 400) + (5 * 20) + (5 * 50) = 750 :=
by
  sorry

end marissa_lunch_calories_l622_622662


namespace find_f_3_8_l622_622338

-- Define f with its properties
noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_property_0_1 : ∀ x, 0 ≤ x ∧ x ≤ 1 → (f(x) : ℝ)
axiom f_0 : f 0 = 0
axiom f_non_decreasing : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f(x) ≤ f(y)
axiom f_symmetric : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(1 - x) = 1 - f(x)
axiom f_quarter : ∀ x, 0 ≤ x ∧ x ≤ 1 → f(x / 4) = f(x) / 2
axiom f_two_fifths : ∀ x, 0 ≤ x ∧ x ≤ 1 → f((2 * x) / 5) = f(x) / 3

-- Proof statement
theorem find_f_3_8 : f (3 / 8) = 1 / 6 := 
by 
  sorry

end find_f_3_8_l622_622338


namespace hyperbola_eccentricity_range_l622_622090

variable {a b : ℝ} (ha : 0 < a) (hb : 0 < b)

def is_eccentricity_valid (e : ℝ) : Prop :=
  (sqrt(5) + 1) / 2 < e ∧ e < (sqrt(6) + sqrt(2)) / 2

theorem hyperbola_eccentricity_range (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let e := sqrt(a^2 + b^2) / a in
  is_eccentricity_valid e := by
  sorry

end hyperbola_eccentricity_range_l622_622090


namespace arith_seq_sum_ratio_l622_622133

theorem arith_seq_sum_ratio 
  (S : ℕ → ℝ) 
  (a1 d : ℝ) 
  (h1 : S 1 = 1) 
  (h2 : (S 4) / (S 2) = 4) :
  (S 6) / (S 4) = 9 / 4 :=
sorry

end arith_seq_sum_ratio_l622_622133


namespace max_homework_time_l622_622267

noncomputable def calculate_total_time (T_biology T_history T_chemistry T_mathematics T_english T_geography : ℝ) :=
  T_biology + T_history + T_chemistry + T_mathematics + T_english + T_geography

theorem max_homework_time :
  let T_biology := 36
  let T_history := 1.75 * T_biology
  let T_chemistry := T_biology - (0.25 * T_biology)
  let T_mathematics := Real.sqrt T_history
  let T_english := 2 * (T_history + T_chemistry)
  let T_geography := max (3 * T_history) (2.75 * T_english)
  calculate_total_time T_biology T_history T_chemistry T_mathematics T_english T_geography = 808.94 :=
by
  let T_biology := 36 : ℝ
  let T_history := 1.75 * T_biology
  let T_chemistry := T_biology - 0.25 * T_biology
  let T_mathematics := Real.sqrt T_history
  let T_english := 2 * (T_history + T_chemistry)
  let T_geography := max (3 * T_history) (2.75 * T_english)
  rw [calculate_total_time, T_biology, T_history, T_chemistry, T_mathematics, T_english, T_geography]
  sorry

end max_homework_time_l622_622267


namespace factor_diff_of_squares_l622_622886

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l622_622886


namespace circle_area_l622_622907

theorem circle_area (d : ℝ) (h : d = 12) : 
  let r := d / 2 in
  let area := Real.pi * r^2 in
  area ≈ 113.09724 :=
by
  sorry

end circle_area_l622_622907


namespace odd_factors_count_eq_10_l622_622466

theorem odd_factors_count_eq_10 : 
    (∃ n : ℕ, ∀ k ∈ (finset.Icc 1 100), (nat.factors_count k).odd ↔ k = n^2) → n = 10 := 
by sorry

end odd_factors_count_eq_10_l622_622466


namespace sum_primitive_roots_eq_23_l622_622777

-- Define the set of integers.
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define modulo
def modulo : ℕ := 11

-- Predicate to check if an integer is a primitive root modulo 11.
def isPrimitiveRoot (a : ℕ) : Prop := 
  ∀ k : ℕ, k < modulo → (a ^ k) % modulo ≠ 1

-- Find the sum of primitive roots in the set S modulo 11.
def sumOfPrimitiveRoots : ℕ :=
  (Sum (filter isPrimitiveRoot S))

-- Specification of the proof.
theorem sum_primitive_roots_eq_23 : sumOfPrimitiveRoots = 23 := 
  sorry

end sum_primitive_roots_eq_23_l622_622777


namespace bob_switching_win_prob_l622_622212

-- Definitions and conditions
variable {doors : Finset ℕ} (hd : doors.card = 7) (prizes : doors.Subset) (hp : prizes.card = 2)
variable {initial_choice : ℕ} (hi : initial_choice ∈ doors)
variable {opened_doors : Finset ℕ} (ho : opened_doors.card = 3) (prize_in_opened : opened_doors ∩ prizes ≠ ∅)
variable {final_choice : ℕ} (hf : final_choice ∈ (doors \ (insert initial_choice opened_doors)))

theorem bob_switching_win_prob :
  ∃ p : ℝ, p = 5/14 :=
by
  sorry

end bob_switching_win_prob_l622_622212


namespace minimize_a_plus_b_l622_622536

theorem minimize_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 4 * a + b = 30) :
  a + b = 9 → (a, b) = (7, 2) := sorry

end minimize_a_plus_b_l622_622536


namespace domain_of_sqrt_fraction_l622_622483

theorem domain_of_sqrt_fraction {x : ℝ} (h1 : x - 3 ≥ 0) (h2 : 7 - x > 0) :
  3 ≤ x ∧ x < 7 :=
by {
  sorry
}

end domain_of_sqrt_fraction_l622_622483


namespace problem_I_problem_II_l622_622150

variable {x y x0 y0 : ℝ}

def point_on_circle (x0 y0 : ℝ) : Prop := (x0 - 4)^2 + y0^2 = 36

def B : (ℝ × ℝ) := (-2, 0)

def P_on_segment_A_B (x y x0 y0 : ℝ) : Prop := 
  (x - x0, y - y0) = 2*(-2 - x, -y)

def trajectory_C (x y : ℝ) : Prop := x^2 + y^2 = 4 ∧ x ≠ -2

noncomputable def distance (A B : (ℝ × ℝ)) : ℝ := 
  ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2)

def line_passing_through (x y b: ℝ) : Prop := (distance (-1, 3) (x, y) = 1) ∧
  (2 * (3 * y) - 5 * x = 0)

theorem problem_I {x y : ℝ} : 
  (∃ x0 y0 : ℝ, point_on_circle x0 y0 ∧ P_on_segment_A_B x y x0 y0) -> 
  trajectory_C x y := 
sorry

theorem problem_II {M N : ℝ × ℝ} : 
  (∃ x y : ℝ, trajectory_C x y ∧ line_passing_through x y ≡ 2*sqrt(3)) -> 
  ∃ (k : ℝ), 
  (
    (4 * x + 3 * y = 5) ∧ (distance x y M N = 2 * sqrt(3))
  ) ∨ x = -1 := 
sorry

end problem_I_problem_II_l622_622150


namespace factor_difference_of_squares_l622_622897
noncomputable def factored (t : ℝ) : ℝ := (t - 8) * (t + 8)

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = factored t := by
  sorry

end factor_difference_of_squares_l622_622897


namespace max_value_k_eq_1_range_k_no_zeros_l622_622565

-- Define the function f(x)
noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log (x - 1) - k * (x - 1) + 1

-- Note: 'by' and 'sorry' are placeholders to skip the proof; actual proofs are not required.

-- Proof Problem 1: Prove that when k = 1, the maximum value of f(x) is 0.
theorem max_value_k_eq_1 : ∀ x : ℝ, 1 < x → f x 1 ≤ 0 := 
by
  sorry

-- Proof Problem 2: Prove that k ∈ (1, +∞) is the range such that f(x) has no zeros.
theorem range_k_no_zeros : ∀ k : ℝ, (∀ x : ℝ, 1 < x → f x k ≠ 0) → 1 < k :=
by
  sorry

end max_value_k_eq_1_range_k_no_zeros_l622_622565


namespace expression_positive_l622_622281

theorem expression_positive (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) : 
  5 * x^2 + 5 * y^2 + 5 * z^2 + 6 * x * y - 8 * x * z - 8 * y * z > 0 := 
sorry

end expression_positive_l622_622281


namespace range_of_a_l622_622991

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1 < a ∧ a ≤ 1) := by
  sorry

end range_of_a_l622_622991


namespace math_problem_l622_622562

def f : ℝ → ℝ
| x := if x ≥ 4 then (1/2)^x else f (x + 1)

theorem math_problem : f (Real.log 3 / Real.log 2) = 1/24 :=
sorry

end math_problem_l622_622562


namespace speed_from_top_to_bottom_is_correct_l622_622633

noncomputable def parking_garage_speed (n : ℕ) (t_id : ℕ) (d : ℕ) (t_total : ℕ) : ℚ :=
  let num_gates := n / 3
  let time_id_total := num_gates * t_id
  let time_driving := t_total - time_id_total
  let num_transitions := n - 1
  let total_distance := num_transitions * d
  total_distance / time_driving

theorem speed_from_top_to_bottom_is_correct :
  parking_garage_speed 12 120 800 1440 ≈ 9.17 := by
    sorry

end speed_from_top_to_bottom_is_correct_l622_622633


namespace ellipse_equation_hyperbola_equation_l622_622963

-- Definitions based on given conditions
def A := (2 : ℝ, 0 : ℝ)
def B := (3 : ℝ, 2 * Real.sqrt 6)

-- a) Prove the standard equation of the ellipse
theorem ellipse_equation : ∃ (a b : ℝ), a = 2 ∧ b = 1 ∧ (a ≠ 0 ∧ b ≠ 0 ∧ (Real.eccentricity a b = Real.sqrt 3 / 2)) → ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

-- b) Prove the standard equation of the hyperbola
theorem hyperbola_equation : ∃ (a b : ℝ), a = 1 ∧ b = Real.sqrt 3 ∧ (a ≠ 0 ∧ b ≠ 0) → ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1 :=
sorry

end ellipse_equation_hyperbola_equation_l622_622963


namespace determine_true_propositions_l622_622877

def converse_of_statement_1 (x y : ℝ) : Prop :=
  (x = 0) ∧ (y = 0) → (x^2 + y^2 = 0)

def negation_of_statement_2 (T₁ T₂ : Triangle) (h_sim : Similar T₁ T₂) : Prop :=
  ¬ (T₁.Area = T₂.Area)

def contrapositive_of_statement_3 (A B : Set α) : Prop :=
  ¬ (A ⊆ B) → (A ∩ B ≠ A)

def contrapositive_of_statement_4 (n : ℕ) : Prop :=
  ¬ (3 ∣ n) → (n % 10 = 0)

theorem determine_true_propositions (x y : ℝ) (A B : Set α) (n : ℕ) (T₁ T₂ : Triangle) (h_sim : Similar T₁ T₂) :
  (converse_of_statement_1 x y) ∧ (negation_of_statement_2 T₁ T₂ h_sim) ∧ (contrapositive_of_statement_3 A B) ∧ (contrapositive_of_statement_4 n) ↔ (converse_of_statement_1 x y) ∧ (contrapositive_of_statement_3 A B) :=
sorry

end determine_true_propositions_l622_622877


namespace solution_set_l622_622245

variable {f : ℝ → ℝ}

-- hypothesis: f is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- hypothesis: f(2) = 0
axiom f_at_2 : f 2 = 0

-- hypothesis: for x > 0, it always holds that xf'(x) - f(x) < 0
axiom inequality_hyp : ∀ x : ℝ, 0 < x → x * (deriv f x) - f x < 0

-- question: solve x^2 f(x) > 0
theorem solution_set : {x : ℝ | x^2 * f x > 0} = set.Ioo (-∞) (-2) ∪ set.Ioo 0 2 :=
sorry

end solution_set_l622_622245


namespace ellipse_equation_max_tan_angle_l622_622951

theorem ellipse_equation (a b : ℝ) (h1 : a > b > 0) (h2 : (1:ℝ)^2 / a^2 + (3/2)^2 / b^2 = 1)
  (h3 : a^2 - b^2 = 1) : 
  (a = 2) ∧ (b = sqrt 3) := 
by 
  sorry

theorem max_tan_angle (a b x y : ℝ) (h1 : a = 2) (h2 : b = sqrt 3) 
  (h3 : 1 / a^2 + 9 / (4 * b^2) = 1) 
  (h4 : a^2 - b^2 = 1) 
  (h5 : a^2 * x + b^2 * y = 3)
  (h6 : 0 < x ∧ x ≤ 4) 
  (h7 : ∀ t > 0, a*t - b*t = x - y) :
  ∃ (max_tan : ℝ), max_tan = 8/15 :=
by 
  sorry

end ellipse_equation_max_tan_angle_l622_622951


namespace correct_propositions_l622_622559

-- Definitions of propositions
def proposition1 := ∀ (L : Plane) (P Q : Plane), parallel_planes_to_line L P Q → parallel P Q
def proposition2 := ∀ (P Q R : Plane), parallel_planes_to_plane P Q R → parallel P Q
def proposition3 := ∀ (L : Line) (A B : Line), perpendicular_to_line L A B → parallel A B
def proposition4 := ∀ (H : Plane) (M N : Line), perpendicular_to_plane H M N → parallel M N

-- Conditions and the correct answer assertion
theorem correct_propositions :
  (proposition2 ∧ proposition4) :=
begin
  -- Placeholder to assume propositions 2 and 4 are correct and skipping the actual proof
  sorry
end

end correct_propositions_l622_622559


namespace number_of_integer_factors_l622_622121

theorem number_of_integer_factors :
  let valid_n (n : ℕ) := ∃ (a b : ℤ), (a + b = -1) ∧ (a * b = - (n : ℤ))
  in (finset.filter valid_n (finset.range 2001)).card = 44 :=
by
  sorry

end number_of_integer_factors_l622_622121


namespace generate_all_integers_from_1_to_n_l622_622810

-- Problem definition and conditions
variables {m n : ℕ}

-- Condition: m and n are coprime
axiom coprime_m_n : Nat.coprime m n

-- Condition: m < n
axiom m_less_than_n : m < n

-- Main proof goal
theorem generate_all_integers_from_1_to_n
  (h_coprime : Nat.coprime m n)
  (h_m_lt_n : m < n) :
  ∃ (f : ℕ → ℕ), (∀ k, k ≤ n → ∃ i j, i < j ∧ (i + j) % 2 = 0 ∧ (i + j) / 2 = f k) ∧ (f 1 = 1) :=
by
  sorry

end generate_all_integers_from_1_to_n_l622_622810


namespace average_employees_per_week_l622_622408

-- Define the number of employees hired each week
variables (x : ℕ)
noncomputable def employees_first_week := x + 200
noncomputable def employees_second_week := x
noncomputable def employees_third_week := x + 150
noncomputable def employees_fourth_week := 400

-- Given conditions as hypotheses
axiom h1 : employees_third_week / 2 = employees_fourth_week / 2
axiom h2 : employees_fourth_week = 400

-- Prove the average number of employees hired per week is 225
theorem average_employees_per_week :
  (employees_first_week + employees_second_week + employees_third_week + employees_fourth_week) / 4 = 225 :=
by
  sorry

end average_employees_per_week_l622_622408


namespace coefficient_of_x_in_expansion_l622_622620

theorem coefficient_of_x_in_expansion :
  let f := (x - (1 / (Real.sqrt x)))^10 in
  (f.expand x).coe = 210 :=
by
  -- Proof omitted
  sorry

end coefficient_of_x_in_expansion_l622_622620


namespace minimize_rectangle_area_l622_622526

theorem minimize_rectangle_area (ABCD : Parallelogram) (e : Line) :
  ∃ (A1 B1 C1 D1 : Point), 
    (parallel (line_through A A1) e) ∧ 
    (parallel (line_through B B1) e) ∧ 
    (parallel (line_through C C1) e) ∧ 
    (parallel (line_through D D1) e) ∧ 
    (is_rectangle A1 B1 C1 D1) ∧ 
    (min_area (rectangle A1 B1 C1 D1) (45 : Real)) :=
sorry

end minimize_rectangle_area_l622_622526


namespace exists_triangle_with_incenter_and_circumcircle_l622_622250

theorem exists_triangle_with_incenter_and_circumcircle (k : Circle) (P : Point) (hP : P ∈ interior k) :
  ∃ (A B C : Point), 
    Triangle (A, B, C) ∧
    Circumcircle (A, B, C) = k ∧
    Incenter (A, B, C) = P :=
by
  sorry

end exists_triangle_with_incenter_and_circumcircle_l622_622250


namespace initially_calculated_average_is_16_l622_622705

theorem initially_calculated_average_is_16 (S S' : ℕ) (n : ℕ) (correct_avg incorrect_num correct_num : ℕ) : 
    n = 10 → correct_avg = 17 → incorrect_num = 25 → correct_num = 35 → 
    S = correct_avg * n → S' = S - (correct_num - incorrect_num) → 
    S' / n = 16 := 
by 
  intros h1 h2 h3 h4 h5 h6 
  rw [h1, h2, h3, h4] at h6 h5 
  have hS : S = 170 := by rw [h2] at h5; exact (by norm_num) 
  rw [hS] at h6 
  exact (by norm_num)

end initially_calculated_average_is_16_l622_622705


namespace find_value_of_dot_product_l622_622156

-- Define the given conditions
variables {V : Type*} [InnerProductSpace ℝ V]
variables (m n : V)
variables (a b : V)

-- Given conditions
def condition_1 : ∥m∥ = 1 := sorry
def condition_2 : ∥n∥ = 2 := sorry
def condition_3 : inner (2 • m + n) (m - 3 • n) = 0 := sorry
def condition_4 : a = 4 • m - n := sorry
def condition_5 : b = 7 • m + 2 • n := sorry

-- Prove the value of <a, b>
theorem find_value_of_dot_product : ⟪a, b⟫_ℝ = 0 :=
by
  apply sorry

end find_value_of_dot_product_l622_622156


namespace coconut_oil_needed_l622_622824

def butter_per_cup := 2
def coconut_oil_per_cup := 2
def butter_available := 4
def cups_of_baking_mix := 6

theorem coconut_oil_needed : 
  let cups_covered_by_butter := butter_available / butter_per_cup in
  let cups_requiring_coconut_oil := cups_of_baking_mix - cups_covered_by_butter in
  coconut_oil_per_cup * cups_requiring_coconut_oil = 8 :=
by
  sorry

end coconut_oil_needed_l622_622824


namespace exist_pos_neg_roots_l622_622164

noncomputable def equation1 (x : ℝ) : Prop := 5 * x^2 - 10 = 40
noncomputable def equation2 (x : ℝ) : Prop := (3 * x - 2)^2 = (x + 3)^2
noncomputable def equation3 (x : ℝ) : Prop := (sqrt (2 * x^2 - 18) = sqrt (3 * x - 3))

theorem exist_pos_neg_roots :
  (∃ x : ℝ, equation1 x ∧ x > 0) ∧
  (∃ x : ℝ, equation1 x ∧ x < 0) ∧
  (∃ x : ℝ, equation2 x ∧ x > 0) ∧
  (∃ x : ℝ, equation2 x ∧ x < 0) ∧
  (∃ x : ℝ, equation3 x ∧ x > 0) ∧
  (∃ x : ℝ, equation3 x ∧ x < 0) := by
  sorry

end exist_pos_neg_roots_l622_622164


namespace chef_uses_8_ounces_of_coconut_oil_l622_622827

section PastryChef

variables (baking_mix butter_coconut_oil : ℕ) (butter_remaining : ℕ := 4) (total_baking_mix : ℕ := 6) (ounces_per_cup : ℕ := 2)

-- Definitions based on conditions
def butter_needed (cups : ℕ) : ℕ := cups * ounces_per_cup
def butter_covered_mix (butter : ℕ) : ℕ := butter / ounces_per_cup
def remaining_cups (total_cups : ℕ) (covered_cups : ℕ) : ℕ := total_cups - covered_cups
def coconut_oil_needed (cups : ℕ) : ℕ := cups * ounces_per_cup

theorem chef_uses_8_ounces_of_coconut_oil :
  coconut_oil_needed (remaining_cups total_baking_mix (butter_covered_mix butter_remaining)) = 8 :=
by
  sorry

end PastryChef

end chef_uses_8_ounces_of_coconut_oil_l622_622827


namespace sum_of_divisors_57_eq_80_l622_622783

def sum_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).sum

theorem sum_of_divisors_57_eq_80 : sum_of_divisors 57 = 80 := by
  sorry

end sum_of_divisors_57_eq_80_l622_622783


namespace decode_word_is_fufajka_l622_622787

def decode_digit (n : ℕ) : Option Char :=
  let letters := " абвгдеёжзийклмнопрстуфхцчшщъыьэюя".data  -- Russian alphabet with index starting at 1
  if n >= 1 ∧ n <= 33 then some (letters[n]!) else none

def decode_russian (digits : List ℕ) : Option String :=
  digits.mapM decode_digit

def segment_number : ℕ → List ℕ
| 222122111121 => [22, 21, 22, 11, 21]

theorem decode_word_is_fufajka : decode_russian (segment_number 222122111121) = some "фуфайка" :=
sorry

end decode_word_is_fufajka_l622_622787


namespace modulus_of_complex_division_l622_622453

noncomputable def complexDivisionModulus : ℂ := Complex.normSq (2 * Complex.I / (Complex.I - 1))

theorem modulus_of_complex_division : complexDivisionModulus = Real.sqrt 2 := by
  sorry

end modulus_of_complex_division_l622_622453


namespace parabola_intersection_at_1_2003_l622_622677

theorem parabola_intersection_at_1_2003 (p q : ℝ) (h : p + q = 2002) :
  (1, (1 : ℝ)^2 + p * 1 + q) = (1, 2003) :=
by
  sorry

end parabola_intersection_at_1_2003_l622_622677


namespace sum_S6_l622_622947

variable (a_n : ℕ → ℚ)
variable (d : ℚ)
variable (S : ℕ → ℚ)
variable (a1 : ℚ)

/-- Define arithmetic sequence with common difference -/
def arithmetic_seq (n : ℕ) := a1 + n * d

/-- Define the sum of the first n terms of the sequence -/
def sum_of_arith_seq (n : ℕ) := n * a1 + (n * (n - 1) / 2) * d

/-- The given conditions -/
axiom h1 : d = 5
axiom h2 : (a_n 1 = a1) ∧ (a_n 2 = a1 + d) ∧ (a_n 5 = a1 + 4 * d)
axiom geom_seq : (a1 + d)^2 = a1 * (a1 + 4 * d)

theorem sum_S6 : S 6 = 90 := by
  sorry

end sum_S6_l622_622947


namespace sum_of_divisors_57_l622_622786

theorem sum_of_divisors_57 : 
  ∑ d in (finset.filter (λ x, 57 % x = 0) (finset.range (58))), d = 80 := 
by
  sorry

end sum_of_divisors_57_l622_622786


namespace min_a1_a7_l622_622610

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = a n * r

theorem min_a1_a7 (a : ℕ → ℝ) (h : geom_seq a)
  (h1 : a 3 * a 5 = 64) :
  ∃ m, m = (a 1 + a 7) ∧ m = 16 :=
by
  sorry

end min_a1_a7_l622_622610


namespace smaller_of_two_numbers_in_ratio_l622_622004

theorem smaller_of_two_numbers_in_ratio (x y a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : x / y = a / b) (h3 : x + y = c) : 
  min x y = (a * c) / (a + b) :=
by
  sorry

end smaller_of_two_numbers_in_ratio_l622_622004


namespace min_value_M_l622_622353

theorem min_value_M 
  (S_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (a : ℕ → ℝ)
  (h1 : ∀ n, S_n n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1)))
  (h2 : a 4 - a 2 = 8)
  (h3 : a 3 + a 5 = 26)
  (h4 : ∀ n, T_n n = S_n n / n^2) :
  ∃ M : ℝ, M = 2 ∧ (∀ n > 0, T_n n ≤ M) :=
by sorry

end min_value_M_l622_622353


namespace max_distance_between_S_origin_l622_622732

noncomputable def max_distance_parallelogram (z : ℂ) (h1 : |z| = 1) (h2 : ¬ collinear {z, (2 + complex.i) * z, 3 * complex.conj z}) : ℝ :=
2 * Real.sqrt 5

theorem max_distance_between_S_origin (z : ℂ) (h1 : |z| = 1) (h2 : ¬ collinear {z, (2 + complex.i) * z, 3 * complex.conj z}) :
  ∃ S : ℂ, S ∈ parallelogram {z, (2 + complex.i) * z, 3 * complex.conj z} ∧ complex.abs S = 2 * Real.sqrt 5 :=
sorry

end max_distance_between_S_origin_l622_622732


namespace jawbreakers_in_package_correct_l622_622634

def jawbreakers_ate : Nat := 20
def jawbreakers_left : Nat := 4
def jawbreakers_in_package : Nat := jawbreakers_ate + jawbreakers_left

theorem jawbreakers_in_package_correct : jawbreakers_in_package = 24 := by
  sorry

end jawbreakers_in_package_correct_l622_622634


namespace prime_dates_count_2012_l622_622479

def is_leap_year (year: ℕ): Prop := (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def prime_dates_in_leap_year (year: ℕ): ℕ :=
  if is_leap_year year then
    let prime_months := [2, 3, 5, 7, 11] in
    let prime_days := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] in
    prime_months.foldl (λ acc m,
      match m with
      | 2 => acc + prime_days.filter (λ d, d ≤ 29).length
      | 11 => acc + prime_days.filter (λ d, d ≤ 30).length
      | _ => acc + prime_days.filter (λ d, d ≤ 31).length
      end
    ) 0
  else 0

theorem prime_dates_count_2012 : prime_dates_in_leap_year 2012 = 53 :=
by
  sorry

end prime_dates_count_2012_l622_622479


namespace sum_of_divisors_57_l622_622780

theorem sum_of_divisors_57 : 
  ∀ (n : ℕ), n = 57 → (∑ d in (finset.filter (λ x, 57 % x = 0) (finset.range 58)), d) = 80 :=
by {
  assume n h,
  sorry
}

end sum_of_divisors_57_l622_622780


namespace compare_values_l622_622867

noncomputable def f (x m : ℝ) : ℝ := 2 ^ (| x - m |) - 1

theorem compare_values (m : ℝ) (h : m = 0) :
  let a := f (Real.log 3 / Real.log 0.5) m
  let b := f (Real.log 5 / Real.log 2) m
  let c := f (2 * m) m
  in c < a ∧ a < b := by
  sorry

end compare_values_l622_622867


namespace jogging_distance_apart_l622_622840

theorem jogging_distance_apart
  (alice_speed : ℝ)
  (bob_speed : ℝ)
  (time_in_minutes : ℝ)
  (distance_apart : ℝ)
  (h1 : alice_speed = 1 / 12)
  (h2 : bob_speed = 3 / 40)
  (h3 : time_in_minutes = 120)
  (h4 : distance_apart = alice_speed * time_in_minutes + bob_speed * time_in_minutes) :
  distance_apart = 19 := by
  sorry

end jogging_distance_apart_l622_622840


namespace triangle_SIX_area_two_l622_622485

axiom Dodecagon_Structure 
  (n : ℕ) (side_length : ℝ) (angles : Fin n → ℝ)
  (vertices : Fin n → ℝ × ℝ)
  (distinct_vertices : ∀ i j, i ≠ j → vertices i ≠ vertices j)
  (distance_property : ∀ i, (euclidean_dist (vertices i) (vertices (Fin.rotate 1 i))) = side_length)
  (angle_property : ∀ i, angles i ∈ {90, 270}) 
  (side_length_eq : side_length = 2)
  (n_val : n = 12) : 
  ∃ S I X : Fin n, 
  is_triangle_SIX vertices S I X → 
  (triangle_area (vertices S) (vertices I) (vertices X) = 2)

def euclidean_dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def is_triangle_SIX
  (vertices : Fin 12 → ℝ × ℝ) 
  (S I X : Fin 12) : Prop :=
  S ≠ I ∧ I ≠ X ∧ S ≠ X

def triangle_area 
  (A B C : ℝ × ℝ) : ℝ :=
  real.abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

theorem triangle_SIX_area_two :
  ∀ (vertices : Fin 12 → ℝ × ℝ),
  ∃ (S I X : Fin 12),
  is_triangle_SIX vertices S I X →
  triangle_area (vertices S) (vertices I) (vertices X) = 2 := sorry

end triangle_SIX_area_two_l622_622485


namespace circle_properties_l622_622547

noncomputable def circle_center : ℝ × ℝ :=
  let system (x y : ℝ) := (y = 2 * x - 16 ∧ x + y = 8)
  ⟨classical.some (Exists.unique (λ p, system p.fst p.snd)), classical.some (Exists.unique (λ p, system p.fst p.snd))⟩

theorem circle_properties :
  let C := circle_center in
  let r := real.sqrt 5 in
  let equation_of_circle := (λ x y, (x - C.fst) ^ 2 + y ^ 2) in
  equation_of_circle x y = r ^ 2 ∧
  (∀ a, (∀ {M N : ℝ × ℝ}, (2 * M.fst + a * M.snd + 6 * a = a * M.fst + 14) ∧ (2 * N.fst + a * N.snd + 6 * a = a * N.fst + 14) → (equation_of_circle M.fst M.snd = r ^ 2 ∧ equation_of_circle N.fst N.snd = r ^ 2) → (2 * real.sqrt (r ^ 2 - (real.sqrt ((C.fst - 7) ^ 2 + (C.snd - 1) ^ 2)) ^ 2)) = 2 * real.sqrt 3)) :=
by {
  let C := circle_center,
  let r := real.sqrt 5,
  let equation_of_circle := (λ x y, (x - C.fst) ^ 2 + y ^ 2),
  have h1 : equation_of_circle 8 0 = r ^ 2 := sorry,
  have h2 : ∀ a, (∀ {M N : ℝ × ℝ}, (2 * M.fst + a * M.snd + 6 * a = a * M.fst + 14) ∧ (2 * N.fst + a * N.snd + 6 * a = a * N.fst + 14) → (equation_of_circle M.fst M.snd = r ^ 2 ∧ equation_of_circle N.fst N.snd = r ^ 2) → (2 * real.sqrt (r ^ 2 - (real.sqrt ((C.fst - 7) ^ 2 + (C.snd - 1) ^ 2)) ^ 2)) = 2 * real.sqrt 3) := sorry,
  exact ⟨h1, h2⟩
}

end circle_properties_l622_622547


namespace find_n_l622_622906

theorem find_n (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end find_n_l622_622906


namespace find_a_l622_622131

theorem find_a (a : ℂ) : (↑((1 : ℂ) - (1 : ℂ) * complex.I) ^ 3 / (↑(1 : ℂ) + complex.I)) = a + 3 * complex.I → a = -2 :=
by
  sorry

end find_a_l622_622131


namespace equation_of_tangent_line_l622_622908

noncomputable theory

-- Define the curve.
def curve (x : ℝ) : ℝ := x^3 + 3 * x^2 - 5

-- Define the point of tangency.
def point_of_tangency : ℝ × ℝ := (-1, curve (-1))

-- Define the derivative of the curve.
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 6 * x

-- Define the slope of the tangent line at the point of tangency.
def tangent_slope : ℝ := curve_derivative (-1)

-- State the equation of the tangent line in standard form.
def tangent_line (x y : ℝ) : Prop := 3 * x + y + 6 = 0

-- The proof problem: show that the equation of the tangent line at the given point is correct.
theorem equation_of_tangent_line :
  tangent_line (prod.fst point_of_tangency) (curve (prod.fst point_of_tangency)) :=
by
  -- Proof omitted.
  sorry

end equation_of_tangent_line_l622_622908


namespace simplify_expression_l622_622310

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l622_622310


namespace rolls_remaining_to_sell_l622_622920

-- Conditions
def total_rolls_needed : ℕ := 45
def rolls_sold_to_grandmother : ℕ := 1
def rolls_sold_to_uncle : ℕ := 10
def rolls_sold_to_neighbor : ℕ := 6

-- Theorem statement
theorem rolls_remaining_to_sell : (total_rolls_needed - (rolls_sold_to_grandmother + rolls_sold_to_uncle + rolls_sold_to_neighbor) = 28) :=
by
  sorry

end rolls_remaining_to_sell_l622_622920


namespace conversation_year_l622_622666

theorem conversation_year (a b c d : ℕ) (y1 y2 : ℕ) (conversation_year : ℕ) :
  1900 ≤ y1 ∧ y1 < 2000 ∧ 1900 ≤ y2 ∧ y2 < 2000 ∧
  (y1 ≠ y2) ∧
  (let elder_sum := ((y2 / 1000) + ((y2 / 100) % 10) + ((y2 / 10) % 10) + (y2 % 10)) in
   elder_sum = conversation_year % 100 ∧
   let elder_age := conversation_year - y1 in
   let younger_age := conversation_year - y2 in
   let elder_age_sum := ((elder_age / 10) % 10) + (elder_age % 10) in 
   let younger_age_sum := ((younger_age / 10) % 10) + (younger_age % 10) in
    elder_age = elder_age_sum + 10 ∧ younger_age = younger_age_sum + 10 ∧
    elder_age_sum = younger_age_sum ∧ younger_age_sum = conversation_year % 100) →
  conversation_year = 1941 :=
by
  sorry

end conversation_year_l622_622666


namespace find_angle_2_l622_622134

theorem find_angle_2 (angle1 : ℝ) (angle2 : ℝ) 
  (h1 : angle1 = 60) 
  (h2 : angle1 + angle2 = 180) : 
  angle2 = 120 := 
by
  sorry

end find_angle_2_l622_622134


namespace find_real_solutions_l622_622497

theorem find_real_solutions :
  ∀ (x y z : ℝ), 
  (x + y - z = -1) ∧ 
  (x^2 - y^2 + z^2 = 1) ∧ 
  (-x^3 + y^3 + z^3 = -1) ↔ 
  (x = 1 ∧ y = -1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = -1 ∧ z = -1) :=
by
  intros x y z
  split
  { intro h,
    cases h with hx1 h2,
    cases h2 with hx2 h3,
    sorry }
  { intro h,
    cases h with h1 h2,
    { cases h1 with hx hy,
      cases hy with hz hyz,
      split,
      { exact hx },
      split,
      { exact h1.left },
      { exact h3 }},
    { cases h2 with hx hy,
      cases hy with hz hyz,
      split,
      { exact hx },
      split,
      { exact h1.left },
      { exact h3 }}}

end find_real_solutions_l622_622497


namespace minimal_guests_l622_622805

-- Problem statement: For 120 chairs arranged in a circle,
-- determine the smallest number of guests (N) needed 
-- so that any additional guest must sit next to an already seated guest.

theorem minimal_guests (N : ℕ) : 
  (∀ (chairs : ℕ), chairs = 120 → 
    ∃ (N : ℕ), N = 20 ∧ 
      (∀ (new_guest : ℕ), new_guest + chairs = 120 → 
        new_guest ≤ N + 1 ∧ new_guest ≤ N - 1)) :=
by
  sorry

end minimal_guests_l622_622805


namespace ratio_expression_l622_622940

theorem ratio_expression (a b c : ℝ) (ha : a / b = 20) (hb : b / c = 10) : (a + b) / (b + c) = 210 / 11 := by
  sorry

end ratio_expression_l622_622940


namespace Tim_transactions_l622_622274

theorem Tim_transactions
  (Mabel_Monday : ℕ)
  (Mabel_Tuesday : ℕ := Mabel_Monday + Mabel_Monday / 10)
  (Anthony_Tuesday : ℕ := 2 * Mabel_Tuesday)
  (Cal_Tuesday : ℕ := (2 * Anthony_Tuesday) / 3)
  (Jade_Tuesday : ℕ := Cal_Tuesday + 17)
  (Isla_Wednesday : ℕ := Mabel_Tuesday + Cal_Tuesday - 12)
  (Tim_Thursday : ℕ := (Jade_Tuesday + Isla_Wednesday) * 3 / 2)
  : Tim_Thursday = 614 := by sorry

end Tim_transactions_l622_622274


namespace marble_difference_l622_622760

theorem marble_difference :
  ∀ (a b : ℕ),
  let total_green_marbles := 162 in
  let total_marbles_A := 8 * a in
  let total_marbles_B := 5 * b in
  3 * a + b = total_green_marbles →
  8 * a = 5 * b →
  (4 * b) - (5 * a) = 49 :=
begin
  intros a b total_green_marbles total_marbles_A total_marbles_B H1 H2,
  sorry,
end

end marble_difference_l622_622760


namespace angle_of_lateral_face_of_pentagonal_prism_l622_622844

theorem angle_of_lateral_face_of_pentagonal_prism :
  ∀ (P : ∀ (a b : ℝ), angle_in_lateral_face P a b = 90) →
    (∀ t : ℝ, t = 90) :=
by
  intro P
  sorry

end angle_of_lateral_face_of_pentagonal_prism_l622_622844


namespace min_value_a_l622_622319

theorem min_value_a (a : ℕ) :
  (6 * (a + 1)) / (a^2 + 8 * a + 6) ≤ 1 / 100 ↔ a ≥ 594 := sorry

end min_value_a_l622_622319


namespace only_integers_square_less_than_three_times_l622_622770

-- We want to prove that the only integers n that satisfy n^2 < 3n are 1 and 2.
theorem only_integers_square_less_than_three_times (n : ℕ) (h : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
sorry

end only_integers_square_less_than_three_times_l622_622770


namespace movement_time_l622_622680

-- Define the conditions
def roja_speed : ℝ := 2 -- Roja's speed in km/hr
def pooja_speed : ℝ := 3 -- Pooja's speed in km/hr
def distance_between : ℝ := 20 -- Distance between them in km

-- Define the proof problem
theorem movement_time :
  let relative_speed := roja_speed + pooja_speed in
  let time := distance_between / relative_speed in
  time = 4 :=
by
  -- Temporary placeholder for proof.
  sorry

end movement_time_l622_622680


namespace axis_of_symmetry_shift_l622_622183

-- Define that f is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the problem statement in Lean
theorem axis_of_symmetry_shift (f : ℝ → ℝ) 
  (h_even : is_even_function f) :
  ∃ x, ∀ y, f (x + y) = f ((x - 1) + y) ∧ x = -1 :=
sorry

end axis_of_symmetry_shift_l622_622183


namespace num_special_permutations_l622_622260

open Finset

-- Define a function that checks if a list of first 'n' integers is a permutation
def is_permutation (l : List ℕ) (n : ℕ) : Prop :=
  l.toFinset = range (n + 1)

noncomputable def count_permutations (n : ℕ) : ℕ := sorry

theorem num_special_permutations : count_permutations 5 = 71 := sorry

end num_special_permutations_l622_622260


namespace cost_of_article_l622_622185

-- Define the cost of the article
variable (C : ℝ)

-- Define the gains when selling at different prices
def G1 := 375 - C
def G2 := 420 - C

-- Condition given in the problem
axiom H : G2 = 1.08 * G1

-- The theorem to prove
theorem cost_of_article : C = 187.5 :=
by
  -- Use the provided condition H in the proof
  have : 420 - C = 1.08 * (375 - C) := H
  -- sorry indicates where the proof would go and assumes the theorem is true
  sorry

end cost_of_article_l622_622185


namespace lifeguard_swim_time_l622_622360

theorem lifeguard_swim_time
  (total_distance : ℝ) (time_taken : ℝ)
  (front_crawl_speed : ℝ) (breaststroke_speed : ℝ)
  (front_crawl_time : ℝ) :
  total_distance = 500 →
  time_taken = 12 →
  front_crawl_speed = 45 →
  breaststroke_speed = 35 →
  45 * front_crawl_time + 35 * (12 - front_crawl_time) = 500 →
  front_crawl_time = 8 :=
by
  intros _ _ _ _ h,
  sorry

end lifeguard_swim_time_l622_622360


namespace vertex_of_quadratic_l622_622931

theorem vertex_of_quadratic : 
  ∀ (x : ℝ), 
  (∃ y : ℝ, y = x^2 - 2 * x + 3) → 
  (1, 2) = (1, (1)^2 - 2 * (1) + 3) :=
by 
  intro x h
  use (1, 2)
  sorry

end vertex_of_quadratic_l622_622931


namespace units_digit_of_product_of_odds_between_10_and_50_l622_622855

def product_of_odds_units_digit : ℕ :=
  let odds := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
  let product := odds.foldl (· * ·) 1
  product % 10

theorem units_digit_of_product_of_odds_between_10_and_50 : product_of_odds_units_digit = 5 :=
  sorry

end units_digit_of_product_of_odds_between_10_and_50_l622_622855


namespace distance_between_A_and_B_l622_622481

-- Given Conditions
variable (d v_A v_B : ℝ)
variable (time1 : ℝ := 10 / v_A = (d - 10) / v_B)
variable (time2 : ℝ := (2d - 12) / v_A = (d + 12) / v_B)

-- Proof Statement
theorem distance_between_A_and_B :
  (∃ (d : ℝ), 
    (10 / v_A = (d - 10) / v_B) ∧ 
    ((2d - 12) / v_A = (d + 12) / v_B) ∧ 
    (2d - 12) * 10 = (d + 12) * (d - 10) ∧
    d = 21) :=
sorry

end distance_between_A_and_B_l622_622481


namespace relationship_y1_y2_y3_l622_622959

theorem relationship_y1_y2_y3 :
  let y1 := -(((-4):ℝ)^2) + 5 in
  let y2 := -(((-1):ℝ)^2) + 5 in
  let y3 := -((2:ℝ)^2) + 5 in
  y2 > y3 ∧ y3 > y1 :=
by
  sorry

end relationship_y1_y2_y3_l622_622959


namespace ellipse_equation_l622_622143

noncomputable def ellipse_data := 
  let a := 2 * Real.sqrt 2
  let b := 2
  let c := 2
  let e := c / a
  let Focus := (2, 0) : ℝ × ℝ
  (a, b, c, e, Focus)

theorem ellipse_equation (h₁ : 2 * Real.sqrt 2 > 2)
    (h₂ : 2 > 0)
    (h₃ : 2 / (2 * Real.sqrt 2) = Real.sqrt 2 / 2) :
  (ellipse_data = (2 * Real.sqrt 2, 2, 2, Real.sqrt 2 / 2, (2, 0)) : ℝ × ℝ × ℝ × ℝ × (ℝ × ℝ)) ∧
  ∃ (k m : ℝ), (m = 0) ∧
  ∃ (A B N : ℝ × ℝ), 
    (|A.1 - N.1| = |B.1 - N.1| ∧ |A.2 - N.2| = |B.2 - N.2|) ∧ 
    ∀ (k ≠ 0), 
      let t := k^2 + 1,
      let Area := 8 * Real.sqrt ((t^2) / ((t + 1) * (2 * t - 1))),
      Area = 16 / 3 :=
sorry

end ellipse_equation_l622_622143


namespace count_conditions_imply_negation_l622_622471

-- Define the variables
variables (p q r : Prop)

-- Define each condition
def condition1 := p ∧ q ∧ r
def condition2 := p ∧ q ∧ ¬r
def condition3 := p ∧ ¬q ∧ r
def condition4 := ¬p ∧ q ∧ ¬r
def condition5 := ¬p ∧ ¬q ∧ r

-- Define the negation of (p ∨ q) ∧ r
def negation := ¬(p ∨ q) ∨ ¬r

-- Statement checking how many conditions imply the negation
theorem count_conditions_imply_negation :
  ( (condition1 → negation) → false ) ∧
  ( (condition2 → negation) ) ∧
  ( (condition3 → negation) → false ) ∧
  ( (condition4 → negation) ) ∧
  ( (condition5 → negation) ) → 3 :=
begin
  intro h,
  sorry
end

end count_conditions_imply_negation_l622_622471


namespace avg_bc_l622_622706

variable {a b c : ℝ}

-- Conditions
def avg_abc := (a + b + c) / 3 = 45
def avg_ab := (a + b) / 2 = 40
def weight_b := b = 27

-- The theorem to prove
theorem avg_bc {a b c : ℝ} (h1 : avg_abc) (h2 : avg_ab) (h3 : weight_b) : (b + c) / 2 = 41 :=
by
  -- This is a placeholder for the actual proof
  sorry

end avg_bc_l622_622706


namespace sum_of_divisors_57_l622_622778

theorem sum_of_divisors_57 : 
  ∀ (n : ℕ), n = 57 → (∑ d in (finset.filter (λ x, 57 % x = 0) (finset.range 58)), d) = 80 :=
by {
  assume n h,
  sorry
}

end sum_of_divisors_57_l622_622778


namespace distinct_combinations_l622_622354

def pairs := {red, blue, green} × {striped, dotted, checkered, plain}

def is_not_same_pattern (sock1 sock2 : pairs) : Prop :=
  sock1.snd ≠ sock2.snd

theorem distinct_combinations : 
  (∃ (sock1 : pairs), sock1.fst = red) ∧ 
  (∃ (sock2 : pairs), sock2.fst ≠ red) ∧ 
  is_not_same_pattern sock1 sock2 → 
  12 :=
begin
  sorry
end

end distinct_combinations_l622_622354


namespace solve_system_dalembert_l622_622694

theorem solve_system_dalembert (C1 C2 : ℝ) :
  (∀ (x y : ℝ → ℝ) (t : ℝ),
    (dx/dt = 5 * x t + 4 * y t + exp t)
    ∧ (dy/dt = 4 * x t + 5 * y t + 1)) →
    ∀ t : ℝ,
      (x t + y t + 1 / 36 * exp t + 1 / 37) * exp (-37 * t) = C1
    ∧ (x t - y t - t * exp t - 1) * exp (-t) = C2 :=
sorry

end solve_system_dalembert_l622_622694


namespace relationship_y1_y2_y3_l622_622957

theorem relationship_y1_y2_y3 :
  let y1 := -(((-4):ℝ)^2) + 5 in
  let y2 := -(((-1):ℝ)^2) + 5 in
  let y3 := -((2:ℝ)^2) + 5 in
  y2 > y3 ∧ y3 > y1 :=
by
  sorry

end relationship_y1_y2_y3_l622_622957


namespace count_special_integers_l622_622251

noncomputable def f : ℕ → ℕ
| 1       := 2
| 2       := 1
| (3*n)   := 3 * f n
| (3*n+1) := 3 * f n + 2
| (3*n+2) := 3 * f n + 1

theorem count_special_integers :
  (∃ count, 
    count = (finset.filter (λ n, f n = 2 * n) (finset.range 2015)).card ∧ 
    count = 127) :=
begin
  use (finset.filter (λ n, f n = 2 * n) (finset.range 2015)).card,
  split, 
  { refl, },
  { sorry, }
end

end count_special_integers_l622_622251


namespace prove_range_a_l622_622591

noncomputable def problem_statement : Prop :=
  ∀ (α : ℝ), (α ∈ set.Icc (real.pi / 6) (2 * real.pi / 3)) →
  ∃ (β : ℝ), (β ∈ set.Icc (real.pi / 6) (2 * real.pi / 3)) →
  ∀ (a : ℝ), (real.cos α ≥ real.sin β + a) → (a ≤ -1)

-- The problem statement as a theorem to be proved
theorem prove_range_a : problem_statement :=
  sorry

end prove_range_a_l622_622591


namespace length_AB_given_conditions_l622_622672

variable {A B P Q : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField P] [LinearOrderedField Q]

def length_of_AB (x y : A) : A := x + y

theorem length_AB_given_conditions (x y u v : A) (hx : y = 4 * x) (hv : 5 * u = 2 * v) (hu : u = x + 3) (hv' : v = y - 3) (hPQ : PQ = 3) : length_of_AB x y = 35 :=
by
  sorry

end length_AB_given_conditions_l622_622672


namespace price_of_each_pizza_l622_622517

variable (P : ℝ)

theorem price_of_each_pizza (h1 : 4 * P + 5 = 45) : P = 10 := by
  sorry

end price_of_each_pizza_l622_622517


namespace transformed_quadratic_sum_l622_622172

noncomputable def transform (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => f (x - 2) + 4

def quadratic (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem transformed_quadratic_sum :
  let f := λ x, 3 * (x + 1)^2 - 2
  let g := transform f
  let h := quadratic 3 (-6) 5
  g = h →
  let (a, b, c) := (3, -6, 5)
  a + b + c = 2 :=
by
  sorry

end transformed_quadratic_sum_l622_622172


namespace domain_of_f_l622_622710

noncomputable def f (x : ℝ) : ℝ := (3 * x) / sqrt (1 - x) + Real.log (2 ^ x - 1)

theorem domain_of_f :
  ∃ d, d = {x : ℝ | 0 < x ∧ x < 1} ∧ ∀ x, x ∈ d → (1 - x > 0) ∧ (2 ^ x - 1 > 0) :=
by
  -- Proof omitted
  sorry

end domain_of_f_l622_622710


namespace how_many_more_rolls_needed_l622_622925

variable (total_needed sold_to_grandmother sold_to_uncle sold_to_neighbor : ℕ)

theorem how_many_more_rolls_needed (h1 : total_needed = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end how_many_more_rolls_needed_l622_622925


namespace certain_number_unique_l622_622184

theorem certain_number_unique (x : ℝ) (hx1 : 213 * x = 3408) (hx2 : 21.3 * x = 340.8) : x = 16 :=
by
  sorry

end certain_number_unique_l622_622184


namespace polynomial_degree_calculation_l622_622243

-- Define the degrees of the polynomials
def degree_f : ℕ := 3
def degree_g : ℕ := 6
def degree_h : ℕ := 2

-- Define the degrees after substitution
def degree_fx2 := 2 * degree_f
def degree_gx5 := 5 * degree_g
def degree_hx3 := 3 * degree_h

-- Define the total degree of the product
def total_degree := degree_fx2 + degree_gx5 + degree_hx3

-- Prove that total_degree equals 42
theorem polynomial_degree_calculation :
  total_degree = 42 := by
  -- Calculate the respective degrees after substitution
  have fx2_degree : degree_fx2 = 6 := by
    unfold degree_fx2
    rw [degree_f]
    exact Nat.mul_comm 2 3

  have gx5_degree : degree_gx5 = 30 := by
    unfold degree_gx5
    rw [degree_g]
    exact Nat.mul_comm 5 6

  have hx3_degree : degree_hx3 = 6 := by
    unfold degree_hx3
    rw [degree_h]
    exact Nat.mul_comm 3 2

  -- Sum the degrees and prove the equation
  unfold total_degree
  rw [fx2_degree, gx5_degree, hx3_degree]
  exact Nat.succ_eq_add_one (6 + 30) 6

end polynomial_degree_calculation_l622_622243


namespace parity_invariant_cannot_achieve_all_positives_l622_622987

def initial_table : list (list bool) :=
  [[false, false, false, true],
   [true, false, false, true],
   [false, false, false, true],
   [false, true, true, true]]

def count_negatives (table : list (list bool)) : ℕ :=
  table.foldl (λ acc row, acc + row.count (λ x, x = true)) 0

theorem parity_invariant (table : list (list bool)) :
  let m := count_negatives table in
  m % 2 = 1 → ∀ seq_of_operations, count_negatives (apply_operations table seq_of_operations) % 2 = 1 :=
sorry

theorem cannot_achieve_all_positives :
  ∀ seq_of_operations, ∃ m, count_negatives (apply_operations initial_table seq_of_operations) = m ∧ m % 2 = 1 :=
by {
  show initial_table,
  rw count_negatives,
  exact parity_invariant
}

end parity_invariant_cannot_achieve_all_positives_l622_622987


namespace necessary_sufficient_condition_l622_622968

theorem necessary_sufficient_condition (a b x_0 : ℝ) (h : a > 0) :
  (x_0 = b / a) ↔ (∀ x : ℝ, (1/2) * a * x^2 - b * x ≥ (1/2) * a * x_0^2 - b * x_0) :=
sorry

end necessary_sufficient_condition_l622_622968


namespace find_target_matrix_l622_622913

noncomputable def target_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![[0, -6, 4], [6, 0, -1], [-4, 1, 0]]

theorem find_target_matrix (u : Fin 3 → ℝ) :
  let vec1 : Fin 3 → ℝ := ![-3, 6, 1]
      vec2 : Fin 3 → ℝ := ![4, -2, 5]
      cross_product (v1 v2 : Fin 3 → ℝ) : Fin 3 → ℝ :=
        ![v1 1 * v2 2 - v1 2 * v2 1,
          v1 2 * v2 0 - v1 0 * v2 2,
          v1 0 * v2 1 - v1 1 * v2 0]
  in (target_matrix ⬝ u = cross_product (vec1 + vec2) u) :=
by
  sorry

end find_target_matrix_l622_622913


namespace quadratic_prob_correct_l622_622697

noncomputable def quadratic_real_roots_probability : ℝ :=
let interval : Set ℝ := Set.Icc 0 5 in
let valid_p : Set ℝ := {p | p ∈ interval ∧ p^2 - 4 ≥ 0} in
MeasureTheory.Measure.count (valid_p) / MeasureTheory.Measure.count (interval)

theorem quadratic_prob_correct :
  quadratic_real_roots_probability = 3 / 5 :=
by
  sorry

end quadratic_prob_correct_l622_622697


namespace equal_lengths_l622_622652

open EuclideanGeometry

noncomputable theory
open_locale classical

-- Definitions for the problem setting
variables 
  {k : Type*} [metric_space k] [normed_add_comm_group k] [normed_space ℝ k]
  (AB : segment k) -- AB is the diameter of circle k
  (circle_k : metric.ball k (1/2 * (AB.start + AB.end)) (dist AB.midpoint AB.start))
  (B : k) -- point B on the circle which is on segment AB
  (t : line k) -- t is the tangent to k at point B
  (C D : k) -- points C and D on the tangent line such that B is between C and D
  (AC AD : line k) -- lines through A intersecting circle at E and F, respectively
  (E F : k) -- points of intersection of lines AC and AD with the circle again
  (DE CF : line k) -- lines intersecting circle again at G and H, respectively
  (G H : k) -- points of intersection of lines DE and CF with the circle again

-- Assumptions to relate the points and geometry
axiom tangent_at_B : t.orthogonal_complement_contains B
axiom B_between_CD : B ∈ segment C D
axiom AE_eq_AC_intersect_circle : AC ∩ circle_k = {E}
axiom AF_eq_AD_intersect_circle : AD ∩ circle_k = {F}
axiom DG_eq_DE_intersect_circle : DE ∩ circle_k = {G}
axiom CH_eq_CF_intersect_circle : CF ∩ circle_k = {H}

theorem equal_lengths :
  dist A G = dist A H :=
sorry

end equal_lengths_l622_622652


namespace calculate_expression_l622_622460

theorem calculate_expression : 2 * (-3)^3 - 4 * (-3) + 15 = -27 := 
by
  sorry

end calculate_expression_l622_622460


namespace grasshoppers_never_larger_square_l622_622129

theorem grasshoppers_never_larger_square
  (initial_positions : Fin 4 → ℤ × ℤ)
  (h_initial_unit_square : ∀ i j, i ≠ j → (initial_positions i).1 - (initial_positions j).1 ∈ {0, 1} ∧ (initial_positions i).2 - (initial_positions j).2 ∈ {0, 1})
  (jump_symmetry : ∀ i j, initial_positions i → initial_positions (j ≠ i) → initial_positions j → (2 * (initial_positions i).fst - (initial_positions j).fst, 2 * (initial_positions i).snd - (initial_positions j).snd))
  (positions_derived : Fin 4 → (ℤ × ℤ ⇒ ℤ × ℤ))
  (h_property_grid : ∃ (steps : ℕ), ∀ t < steps, ∀ i, ∃ k, positions_derived t i = k): 
  ¬ ∃ (positions_larger_square : Fin 4 → ℤ × ℤ)
  (h_larger_square : ∀ i, positions_derived i = positions_larger_square i) :=
begin
  sorry
end

end grasshoppers_never_larger_square_l622_622129


namespace problem1_problem2_l622_622461

-- Problem 1 translation
theorem problem1 : (sqrt 3 + sqrt 2) * (sqrt 3 - sqrt 2) = 1 := sorry

-- Problem 2 translation
theorem problem2 : sqrt 8 + (sqrt 12 / sqrt 6) = 3 * sqrt 2 := sorry

end problem1_problem2_l622_622461


namespace sector_area_calculation_l622_622754

theorem sector_area_calculation (r : ℝ) (n : ℝ) (theta : ℝ) (pi : ℝ) : 
  r = 15 ∧ n = 3 ∧ theta = (40 / 360) ∧ pi = real.pi → 
  n * (1 / theta) * pi * r^2 = 75 * pi :=
by 
  intros h,
  rcases h with ⟨hr, hn, htheta, hpi⟩,
  rw [hr, hn, htheta, hpi],
  sorry

end sector_area_calculation_l622_622754


namespace problem_statement_l622_622946

noncomputable theory

variables {A B C D E F : Type} [AddCommGroup A] [Module ℝ A]
variables (BC AD BE CF : A)
variables (BD DC EA CE FB AF : A)

theorem problem_statement
  (h1 : DC = (2:ℝ) • BD)
  (h2 : CE = (2:ℝ) • EA)
  (h3 : AF = (2:ℝ) • FB)
  (hAD : AD = -BD)
  (hBE : BE = -EA)
  (hCF : CF = -FB)
  : (AD + BE + CF = -k • BC) :=
by
  sorry

end problem_statement_l622_622946


namespace proof_of_expression_value_l622_622583

theorem proof_of_expression_value (m n : ℝ) 
  (h1 : m^2 - 2019 * m = 1) 
  (h2 : n^2 - 2019 * n = 1) : 
  (m^2 - 2019 * m + 3) * (n^2 - 2019 * n + 4) = 20 := 
by 
  sorry

end proof_of_expression_value_l622_622583


namespace lambs_total_l622_622104

/-
Each of farmer Cunningham's lambs is either black or white.
There are 193 white lambs, and 5855 black lambs.
Prove that the total number of lambs is 6048.
-/

theorem lambs_total (white_lambs : ℕ) (black_lambs : ℕ) (h1 : white_lambs = 193) (h2 : black_lambs = 5855) :
  white_lambs + black_lambs = 6048 :=
by
  -- proof goes here
  sorry

end lambs_total_l622_622104


namespace stock_percentage_increased_by_30_l622_622234

-- Conditions and question setup
variables {wallet_initial : ℝ} {investment_initial : ℝ} {wallet_final : ℝ}
variables (percentage_increase : ℝ)

-- Given:
-- 1. Josh has $300 in his wallet.
-- 2. Josh has $2000 invested in a business.
-- 3. The business's stock price rises by a certain percentage and then he sells all of his stocks.
-- 4. Josh ends up with $2900 in his wallet.
def given_conditions : Prop :=
  wallet_initial = 300 ∧ investment_initial = 2000 ∧ wallet_final = 2900

-- Question: What is the percentage increase in the stock price?
def percentage_increase_in_stock_price : ℝ :=
  ((wallet_final - wallet_initial) - investment_initial) / investment_initial * 100

-- Theorem to prove the percentage increase
theorem stock_percentage_increased_by_30 (h : given_conditions) : percentage_increase_in_stock_price = 30 :=
by
  unfold given_conditions at h
  unfold percentage_increase_in_stock_price
  cases h with hwallet_initial hinvestment_initial hwallet_final
  exact sorry

end stock_percentage_increased_by_30_l622_622234


namespace simplify_expression_l622_622293

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622293


namespace geometric_sequence_b_eq_neg3_l622_622188

theorem geometric_sequence_b_eq_neg3 (a b c : ℝ) : 
  (∃ r : ℝ, -1 = r * a ∧ a = r * b ∧ b = r * c ∧ c = r * (-9)) → b = -3 :=
by
  intro h
  obtain ⟨r, h1, h2, h3, h4⟩ := h
  -- Proof to be filled in later.
  sorry

end geometric_sequence_b_eq_neg3_l622_622188


namespace problem1_problem2_l622_622038

-- Problem statement 1: Prove (a-2)(a-6) < (a-3)(a-5)
theorem problem1 (a : ℝ) : (a - 2) * (a - 6) < (a - 3) * (a - 5) :=
by
  sorry

-- Problem statement 2: Prove the range of values for 2x - y given -2 < x < 1 and 1 < y < 2 is (-6, 1)
theorem problem2 (x y : ℝ) (hx : -2 < x) (hx1 : x < 1) (hy : 1 < y) (hy1 : y < 2) : -6 < 2 * x - y ∧ 2 * x - y < 1 :=
by
  sorry

end problem1_problem2_l622_622038


namespace average_mpg_proof_l622_622405

-- Given that the distance from town A to town B is twice the distance from town B to town C,
-- the average miles per gallon for the two segments are given.

variables (x : ℝ) (h1 : ∀ x, 0 < x) 
-- Distance from town A to town B is twice the distance from town B to town C
-- From town A to town B, averaged 20 miles per gallon
-- From town B to town C, averaged 25 miles per gallon
variables (d_ab d_bc : ℝ)
variables (mpg_ab : ℝ := 20)
variables (mpg_bc : ℝ := 25)

def distance_ab (x : ℝ) := 2 * x
def fuel_used_ab (d_ab : ℝ) := d_ab / mpg_ab
def fuel_used_bc (d_bc : ℝ) := d_bc / mpg_bc

def total_distance := d_ab + d_bc
def total_fuel_used := (fuel_used_ab d_ab) + (fuel_used_bc d_bc)

def average_mpg := total_distance / total_fuel_used

theorem average_mpg_proof : average_mpg = 21.43 :=
by
  -- We will skip the proof part
  sorry

end average_mpg_proof_l622_622405


namespace factor_in_form_of_2x_l622_622808

theorem factor_in_form_of_2x (w : ℕ) (hw : w = 144) : ∃ x : ℕ, 936 * w = 2^x * P → x = 4 :=
by
  sorry

end factor_in_form_of_2x_l622_622808


namespace num_ways_to_have_5_consecutive_empty_seats_l622_622607

theorem num_ways_to_have_5_consecutive_empty_seats :
  let n := 10
  let k := 4
  let m := 5
  ( ∃ S : set ℕ, S.card = k ∧ 
    0 ≤ min S ∧ max S < n ∧ 
    ∃ I : finset ℕ, I.card = m ∧ 
      disjoint I S ∧ I = (finset.range n).erase' finset.card ) → 
  ∃ N : ℕ, N = 480 := by
  sorry

end num_ways_to_have_5_consecutive_empty_seats_l622_622607


namespace medal_award_ways_l622_622606

theorem medal_award_ways (total_sprinters : ℕ) (american_sprinters : ℕ) :
  total_sprinters = 10 ∧ american_sprinters = 4 ∧ (∑ (cases : ℕ), cases = 78) :=
by
  have num_non_american : ℕ := total_sprinters - american_sprinters
  let ways_no_americans := num_non_american * (num_non_american - 1)
  let ways_one_american := 2 * american_sprinters * num_non_american
  let total_ways := ways_no_americans + ways_one_american
  exact (78 = total_ways)

end medal_award_ways_l622_622606


namespace pasha_does_not_have_winning_strategy_l622_622339

open Function

structure GameBoard (F : Type) :=
(fields : F)
(conn : F → F → Prop)
(travel : ∀ x y : F, conn x y ∨ conn y x)

structure GameState (F : Type) :=
(lilac : F)
(purple : F)
(non_repeating_positions : List (F × F))

noncomputable def pasha_loses_strategy (F : Type) [Inhabited F] (board : GameBoard F) (initial_state : GameState F) : Prop :=
  ∃ strategy : (F × F) → (F × F),
  ∀ state : (F × F), ¬ strategy state = state

theorem pasha_does_not_have_winning_strategy :
  ∀ (F : Type) [Inhabited F] (board : GameBoard F) (initial : GameState F), pasha_loses_strategy F board initial :=
begin
  intros,
  sorry
end

end pasha_does_not_have_winning_strategy_l622_622339


namespace cannot_be_two_l622_622932

theorem cannot_be_two (x y : ℝ) (h: x^2 + y^2 + 2x - 6y = 6) : (x - 1)^2 + (y - 2)^2 ≠ 2 :=
sorry

end cannot_be_two_l622_622932


namespace not_perfect_square_4_2021_l622_622790

-- Define what it means for a number to be a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x * x

-- State the non-perfect square problem for the given choices
theorem not_perfect_square_4_2021 :
  ¬ is_perfect_square (4 ^ 2021) ∧
  is_perfect_square (1 ^ 2018) ∧
  is_perfect_square (6 ^ 2020) ∧
  is_perfect_square (5 ^ 2022) :=
by
  sorry

end not_perfect_square_4_2021_l622_622790


namespace factorial_fraction_simplification_l622_622456

theorem factorial_fraction_simplification :
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 := 
by {
  sorry
}

end factorial_fraction_simplification_l622_622456


namespace average_temp_Addington_l622_622480

theorem average_temp_Addington :
  let temps : List ℚ := [51, 64, 60, 59, 48, 55, 57]
  let sum_temps := temps.foldl (· + ·) 0
  let n := temps.length
  let avg := sum_temps / n
  avg ≈ 56.3 :=
by
  let temps : List ℚ := [51, 64, 60, 59, 48, 55, 57]
  let sum_temps := temps.foldl (· + ·) 0
  let n := temps.length
  let avg := sum_temps / n
  have approx : avg ≈ 56.3 := sorry
  exact approx

end average_temp_Addington_l622_622480


namespace graph_shift_l622_622170

def f (x : ℝ) : ℝ :=
  if x >= -3 ∧ x <= 0 then -2 - x 
  else if x > 0 ∧ x <= 2 then real.sqrt (4 - (x - 2) ^ 2) - 2 
  else if x > 2 ∧ x <= 3 then 2 * (x - 2) 
  else 0

def candidate (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 1 => f x + 1
  | 2 => f (x + 2)
  | 3 => f (2 - x)
  | 4 => f (x - 2)
  | 5 => f x - 1
  | _ => 0

theorem graph_shift (x : ℝ) : candidate 2 x = f (x + 2) :=
sorry

end graph_shift_l622_622170


namespace simplify_expression_l622_622291

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622291


namespace non_constant_monic_polynomial_form_l622_622871

theorem non_constant_monic_polynomial_form (P : ℤ[X])
  (h_monic : P.leadingCoeff = 1)
  (h_nonconstant : P.degree > 0)
  (h_prime : ∀ n : ℕ, ∀ p : ℕ, p > 10^100 → ¬ (p ∣ (P.eval (2^n)))) :
  ∃ m : ℕ, P = Polynomial.monomial m 1 := 
sorry

end non_constant_monic_polynomial_form_l622_622871


namespace guppies_to_angelfish_ratio_l622_622859

noncomputable def goldfish : ℕ := 8
noncomputable def angelfish : ℕ := goldfish + 4
noncomputable def total_fish : ℕ := 44
noncomputable def guppies : ℕ := total_fish - (goldfish + angelfish)

theorem guppies_to_angelfish_ratio :
    guppies / angelfish = 2 := by
    sorry

end guppies_to_angelfish_ratio_l622_622859


namespace calculation_l622_622584

def Δ (x y : ℝ) : ℝ :=
  (x - y) / (1 - x * y)

theorem calculation :
  Δ (Δ 2 3) 4 = -19 := by
  sorry

end calculation_l622_622584


namespace max_value_a_plus_2b_l622_622574

theorem max_value_a_plus_2b (a b : ℝ) 
  (h1 : ∃ (a b : ℝ), (x y : ℝ) -> x^2 + y^2 + 2*a*x + a^2 - 4 = 0 ∧ 
                x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0 ∧ 
                (∃ (t : ℝ) (h: t = 3) , sqrt (a^2 + 4*b^2) = t)) :
  a^2 + 4*b^2 = 9 → a + 2*b ≤ 3*sqrt 2.

end max_value_a_plus_2b_l622_622574


namespace quadratic_general_form_l622_622472

theorem quadratic_general_form {x : ℝ} :
  (x - 8) ^ 2 = 5 → ∃ a b c : ℝ, a = 1 ∧ b = -16 ∧ c = 59 ∧ a * x^2 + b * x + c = 0 :=
by
  intros h,
  use [1, -16, 59],
  split; [refl, split; [refl, split; [refl, skip]]],
  sorry

end quadratic_general_form_l622_622472


namespace truck_travel_distance_l622_622429

noncomputable def truck_distance (gallons: ℕ) : ℕ :=
  let efficiency_10_gallons := 300 / 10 -- miles per gallon
  let efficiency_initial := efficiency_10_gallons
  let efficiency_decreased := efficiency_initial * 9 / 10 -- 10% decrease
  if gallons <= 12 then
    gallons * efficiency_initial
  else
    12 * efficiency_initial + (gallons - 12) * efficiency_decreased

theorem truck_travel_distance (gallons: ℕ) :
  gallons = 15 → truck_distance gallons = 441 :=
by
  intros h
  rw [h]
  -- skipping proof
  sorry

end truck_travel_distance_l622_622429


namespace prove_n_is_prime_l622_622228

open Nat
open Zmod

theorem prove_n_is_prime (n : ℕ) (a : ℕ) (h1: 1 < n) 
  (h2: ∃ a : ℕ, a^(n-1) ≡ 1 [MOD n]) 
  (h3: ∀ p: ℕ, p.Prime → p ∣ (n - 1) → a^((n-1) / p) ≢ 1 [MOD n]) 
  : Nat.Prime n := 
  sorry

end prove_n_is_prime_l622_622228


namespace card_dealing_probability_l622_622515

theorem card_dealing_probability :
  let total_cards := 52
  let first_card_probability := 3 / total_cards
  let second_card_probability := 12 / (total_cards - 1)
  let third_card_probability := 3 / (total_cards - 2)
  let fourth_card_probability := 11 / (total_cards - 3)
  let case1_probability := first_card_probability * second_card_probability * third_card_probability * fourth_card_probability
  let fifth_card_probability := 3 / total_cards
  let sixth_card_probability := 1 / (total_cards - 1)
  let seventh_card_probability := 2 / (total_cards - 2)
  let eighth_card_probability := 11 / (total_cards - 3)
  let case2_probability := fifth_card_probability * sixth_card_probability * seventh_card_probability * eighth_card_probability
  let total_probability := case1_probability + case2_probability
  in total_probability = 627 / 3248700 :=
begin
  sorry -- Proof goes here.
end

end card_dealing_probability_l622_622515


namespace opposite_of_2023_l622_622730

theorem opposite_of_2023 : ∃ x : ℤ, x + 2023 = 0 ∧ x = -2023 := by
    use -2023
    split
    · ring
    · rfl

end opposite_of_2023_l622_622730


namespace complex_modulus_squared_l622_622707

noncomputable def complex_modulus (z : ℂ) : ℝ :=
  real.sqrt (z.re^2 + z.im^2)

theorem complex_modulus_squared (z : ℂ) (hz : z + complex_modulus z = 2 + 8 * complex.I) :
  complex_modulus z ^ 2 = 289 :=
sorry

end complex_modulus_squared_l622_622707


namespace pushkin_pension_is_survivors_pension_l622_622387

theorem pushkin_pension_is_survivors_pension
  (died_pushkin : Nat = 1837)
  (lifelong_pension_assigned : ∀ t : Nat, t > died_pushkin → ∃ (recipient : String), recipient = "Pushkin's wife" ∨ recipient = "Pushkin's daughter") :
  ∃ (pension_type : String), pension_type = "survivor's pension" :=
by
  sorry

end pushkin_pension_is_survivors_pension_l622_622387


namespace proof_equation_properties_l622_622103

theorem proof_equation_properties (m : ℝ) (h_pos : m > 0) : 
  (∀ x, x^2 - 2*x + (1 - m/3/9)= 0 ↔ (x = 1 + sqrt(m/3) ∨ x = 1 - sqrt(m/3))) ∧ 
  (m = 27 → (1 + sqrt(27/3)) * (1 - sqrt(27/3)) = 4 * -2) ∧ 
  ((1 + sqrt(m/3)) + (1 - sqrt(m/3)) = 2) ∧ 
  (discriminant (λ x : ℝ, -3 * (x-1)^2 + m) > 0) :=
by sorry

end proof_equation_properties_l622_622103


namespace sum_arithmetic_sequence_matrix_l622_622503

theorem sum_arithmetic_sequence_matrix (n : ℕ) : 
  ∑ i in finset.range n, ∑ j in finset.range n, (i + j + 1) = n^3 := 
by 
  sorry

end sum_arithmetic_sequence_matrix_l622_622503


namespace largest_two_digit_decimal_smallest_one_digit_decimal_l622_622371

theorem largest_two_digit_decimal (a b c : ℕ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 7) : 
  max_decimal a b c = 7.32 := sorry

theorem smallest_one_digit_decimal (a b c : ℕ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 7) : 
  min_decimal a b c = 2.37 := sorry

end largest_two_digit_decimal_smallest_one_digit_decimal_l622_622371


namespace f_2013_eq_2_l622_622135

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = -f x
axiom h2 : ∀ x : ℝ, f (x + 4) = f x + f 2
axiom h3 : f (-1) = -2

theorem f_2013_eq_2 : f 2013 = 2 := 
by 
  sorry

end f_2013_eq_2_l622_622135


namespace track_circumference_l622_622801

def same_start_point (A B : ℕ) : Prop := A = B

def opposite_direction (a_speed b_speed : ℕ) : Prop := a_speed > 0 ∧ b_speed > 0

def first_meet_after (A B : ℕ) (a_distance b_distance : ℕ) : Prop := a_distance = 150 ∧ b_distance = 150

def second_meet_near_full_lap (B : ℕ) (lap_length short_distance : ℕ) : Prop := short_distance = 90

theorem track_circumference
    (A B : ℕ) (a_speed b_speed lap_length : ℕ)
    (h1 : same_start_point A B)
    (h2 : opposite_direction a_speed b_speed)
    (h3 : first_meet_after A B 150 150)
    (h4 : second_meet_near_full_lap B lap_length 90) :
    lap_length = 300 :=
sorry

end track_circumference_l622_622801


namespace volume_of_revolution_l622_622087

theorem volume_of_revolution :
  (∫ x in 0..π, (3 * sin x)^2 - (sin x)^2) * π = 4 * π^2 := by
sorry

end volume_of_revolution_l622_622087


namespace sum_of_divisors_57_l622_622779

theorem sum_of_divisors_57 : 
  ∀ (n : ℕ), n = 57 → (∑ d in (finset.filter (λ x, 57 % x = 0) (finset.range 58)), d) = 80 :=
by {
  assume n h,
  sorry
}

end sum_of_divisors_57_l622_622779


namespace passengers_remaining_l622_622190

theorem passengers_remaining :
  let initial_passengers := 64
  let reduction_factor := (2 / 3)
  ∀ (n : ℕ), n = 4 → initial_passengers * reduction_factor^n = 1024 / 81 := by
sorry

end passengers_remaining_l622_622190


namespace distance_P_to_intersections_l622_622979

-- Given conditions
def parametric_eqn_line (t : ℝ) : ℝ × ℝ := (sqrt 2 * t / 2 + 1, -sqrt 2 * t / 2)

-- Cartesian equation of the circle derived from the polar equation
noncomputable def cartesian_eqn_circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 2 * y = 0

-- Coordinates of point P
def point_P : ℝ × ℝ := (1, 0)

-- Prove the desired quantity
theorem distance_P_to_intersections (t1 t2 : ℝ) (h1: parametric_eqn_line t1 = (1, 0))
  (h2: parametric_eqn_line t2 = (1, 0)) : 
  |t1 - t2| = sqrt 6 :=
sorry

end distance_P_to_intersections_l622_622979


namespace trig_identity_sin_diff_l622_622347

theorem trig_identity_sin_diff :
  sin (75 * Real.pi / 180) * cos (45 * Real.pi / 180) - cos (75 * Real.pi / 180) * sin (45 * Real.pi / 180) = 1 / 2 :=
by
  -- the theorem asserts the identity, proof is omitted
  sorry

end trig_identity_sin_diff_l622_622347


namespace probability_in_dark_l622_622832

-- Define the given conditions
def revolutions_per_minute : ℝ := 2
def time_in_minutes_to_seconds (minutes : ℝ) : ℝ := minutes * 60
def time_for_one_revolution (rpm : ℝ) : ℝ := time_in_minutes_to_seconds 1 / rpm
def time_in_the_dark : ℝ := 5

-- Define the theorem to be proven
theorem probability_in_dark :
  let T_rev := time_for_one_revolution revolutions_per_minute in
  let prob := time_in_the_dark / T_rev in
  prob = 1 / 6 :=
by
  sorry

end probability_in_dark_l622_622832


namespace third_number_lcm_l622_622115

theorem third_number_lcm (n : ℕ) :
  n ∣ 360 ∧ lcm (lcm 24 36) n = 360 →
  n = 5 :=
by sorry

end third_number_lcm_l622_622115


namespace budget_equality_year_l622_622025

theorem budget_equality_year :
  ∃ n : ℕ, 540000 + 30000 * n = 780000 - 10000 * n ∧ 1990 + n = 1996 :=
by
  sorry

end budget_equality_year_l622_622025


namespace min_value_fraction_l622_622971

theorem min_value_fraction {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (∀ y : ℝ,  y > 0 → (∀ x : ℝ, x > 0 → x + 3 * y = 1 → (1/x + 1/(3*y)) ≥ 4)) :=
sorry

end min_value_fraction_l622_622971


namespace sum_of_solutions_l622_622383

/-- The sum of real numbers satisfying the equation is 19/15 -/
theorem sum_of_solutions (x : ℝ) :
  (∑ x in {x : ℝ | ∃ k : ℤ, k = (15 * x - 7) / 5 ∧ (k ≤ (6 * x + 5) / 8 ∧ (6 * x + 5) / 8 < k + 1)}, x) = 19/15 :=
sorry

end sum_of_solutions_l622_622383


namespace find_common_ratio_l622_622644

noncomputable def eval_q (a₁ a₂ a₃ a₄ q : ℝ) : ℝ :=
  let S₃ := a₁ + a₂ + a₃
  let S₂ := a₁ + a₂
  q

theorem find_common_ratio (a₁ a₂ a₃ a₄ q : ℝ)
  (h1 : 3 * (a₁ + a₂ + a₃) = a₄ - 2)
  (h2 : 3 * (a₁ + a₂) = a₃ - 2) :
  q = 4 :=
by
  let S₃ := a₁ + a₂ + a₃
  let S₂ := a₁ + a₂
  have h := congrArg (λ x, 3 * x) (eq.subst h2 h1)
  rw [add_assoc, add_assoc, ← sub_eq_iff_eq_add'] at h
  sorry

end find_common_ratio_l622_622644


namespace complex_exponential_sum_identity_l622_622108

theorem complex_exponential_sum_identity :
    12 * Complex.exp (Real.pi * Complex.I / 7) + 12 * Complex.exp (19 * Real.pi * Complex.I / 14) =
    24 * Real.cos (5 * Real.pi / 28) * Complex.exp (3 * Real.pi * Complex.I / 4) :=
sorry

end complex_exponential_sum_identity_l622_622108


namespace sum_of_cubes_l622_622598

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = -3) : x^3 + y^3 = 26 :=
sorry

end sum_of_cubes_l622_622598


namespace initial_number_l622_622067

theorem initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_number_l622_622067


namespace maximum_teams_l622_622603

theorem maximum_teams
  (h1 : ∀ t, t ∈ teams → #t = 3)
  (h2 : ∀ t1 t2, t1 ∈ teams → t2 ∈ teams → t1 ≠ t2 → 
                 ∀ p1 ∈ t1, ∀ p2 ∈ t2, game p1 p2)
  (h3 : #games ≤ 150) :
  ∃ n : ℕ, #teams = n ∧ n = 6 := 
sorry

end maximum_teams_l622_622603


namespace quadrilateral_area_formula_l622_622145

variables (a b c d e f ω : ℝ)

def area_of_quadrilateral (a b c d e f ω : ℝ) : ℝ :=
  (1 / 4) * (b^2 + d^2 - a^2 - c^2) * Real.tan ω

theorem quadrilateral_area_formula :
  area_of_quadrilateral a b c d e f ω = (1 / 4) * (b^2 + d^2 - a^2 - c^2) * Real.tan ω :=
by
  sorry

end quadrilateral_area_formula_l622_622145


namespace relay_race_time_l622_622283

theorem relay_race_time (R S D : ℕ) (h1 : S = R + 2) (h2 : D = R - 3) (h3 : R + S + D = 71) : R = 24 :=
by
  sorry

end relay_race_time_l622_622283


namespace interval_contains_zero_of_f_l622_622341
open Real

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * log x + x - 1 / x - 2

theorem interval_contains_zero_of_f :
  ∃ (c : ℝ), (2 < c ∧ c < exp 1) ∧ f c = 0 :=
by
  have h_monotone : ∀ x > 0, f' x > 0 := 
    λ x hx, by {
      let f' := (1 / (2 * x) + 1 + 1 / (x^2)),
      linarith [half_pos, hx]
    }

  have h_cont : continuous_on f (Ioo 2 (exp 1)) := sorry

  have h_sign : f 2 < 0 ∧ 0 < f (exp 1) :=
  by {
    simp [f, real.logr_eq_log, div_eq_mul_one_div, real.continuous_logarithm.loge_eq e],
    split,
    { linarith [log_lt_of_pos _ (by norm_num : 0 < exp 1)], },
    { linarith [one_div_exp_pos, div_pos]}
  }

  exact intermediate_value_theorem h_cont h_sign sorry

end interval_contains_zero_of_f_l622_622341


namespace problem_l622_622237

variable (a b c : ℝ)

theorem problem (h : a^2 * b^2 + 18 * a * b * c > 4 * b^3 + 4 * a^3 * c + 27 * c^2) : a^2 > 3 * b :=
by
  sorry

end problem_l622_622237


namespace parallelogram_side_length_l622_622823

theorem parallelogram_side_length (s : ℝ) (angle : ℝ) (area : ℝ) (h : angle = π / 3) (parallelogram_area : 2 * s * (s * real.sin angle) = area) : s = real.sqrt 6 :=
by
  sorry

end parallelogram_side_length_l622_622823


namespace evaluate_division_l622_622488

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end evaluate_division_l622_622488


namespace problem_triangle_intersections_l622_622440

theorem problem_triangle_intersections
  (A B C M P Q K S T R : Point)
  (circumcircle : Circle)
  (h1 : M ∈ Triangle A B C)
  (h2 : circumcircle = Circumcircle A B C)
  (h3 : LineThrough A M ∩ LineThrough B C = P)
  (h4 : LineThrough A M ∩ circumcircle = S)
  (h5 : LineThrough B M ∩ LineThrough C A = Q)
  (h6 : LineThrough B M ∩ circumcircle = T)
  (h7 : LineThrough C M ∩ LineThrough A B = K)
  (h8 : LineThrough C M ∩ circumcircle = R) :
  (AM/MS) * (SP/PA) + (BM/MT) * (TQ/QB) + (CM/MR) * (RK/KC) = 1 :=
sorry

end problem_triangle_intersections_l622_622440


namespace isosceles_triangle_inequality_l622_622953

theorem isosceles_triangle_inequality
  (a b : ℝ)
  (hb : b > 0)
  (h₁₂ : 12 * (π / 180) = π / 15) 
  (h_sin6 : Real.sin (6 * (π / 180)) > 1 / 10)
  (h_eq : a = 2 * b * Real.sin (6 * (π / 180))) : 
  b < 5 * a := 
by
  sorry

end isosceles_triangle_inequality_l622_622953


namespace evaluate_expression_l622_622126

theorem evaluate_expression (y : ℚ) (h : y = 1 / 3) :
  1 / (3 + 1 / (3 + 1 / (3 - y))) = 27 / 89 :=
by {
  rw h,
  calc
    1 / (3 + 1 / (3 + 1 / (3 - 1 / 3))) = 1 / (3 + 1 / (3 + 1 / (8 / 3))) : by rw [sub_div, sub_self, sub_fraction, sub_mul_div]
    ... = 1 / (3 + 1 / (3 + 3 / 8)) : by rw [div_inv_eq]
    ... = 1 / (3 + 8 / 27) : by rw [add_com, add_fraction]
    ... = 1 / (27 / 89) : by rw [inv_div_eq_div_mul]
    ... = 27 / 89 : by rw.div_div_eq_inv_eq_inv
}

end evaluate_expression_l622_622126


namespace nth_non_cube_250_eq_256_l622_622765

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k^3 = n

noncomputable def nth_non_cube (n : ℕ) : ℕ :=
  (n : ℕ).succ + (range (n : ℕ).succ).count (λ x, is_perfect_cube x)

theorem nth_non_cube_250_eq_256 :
  nth_non_cube 250 = 256 :=
sorry

end nth_non_cube_250_eq_256_l622_622765


namespace min_value_y_in_interval_l622_622484

def y (x : ℝ) : ℝ := tan x + (tan x) / (sin (2 * x - π / 2))

theorem min_value_y_in_interval :
  (∀ x, (π / 4) < x ∧ x < (π / 2) → y x ≥ 3 * sqrt 3) ∧ 
  (∃ x, (π / 4) < x ∧ x < (π / 2) ∧ y x = 3 * sqrt 3) :=
sorry

end min_value_y_in_interval_l622_622484


namespace ellipse_equation_and_eccentricity_l622_622551

theorem ellipse_equation_and_eccentricity
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (line_passes : ∃ x y : ℝ, (x - 2 * y + 2 = 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
                  ∧ ((x = -2 ∧ y = 0) ∨ (x = 0 ∧ y = 1))) :
  ((a = sqrt 5) ∧ (b = 1) ∧
  (a^2 = b^2 + 2^2) ∧
  (c : (ℝ := ℝ)) := 2 ∧
  ((∃ x y : ℝ, (x^2 / 5 + y^2 = 1)) ∧
   (e = 2 / sqrt 5)) :=
by
  sorry

end ellipse_equation_and_eccentricity_l622_622551


namespace rectangle_length_15_l622_622734

theorem rectangle_length_15
  (w l : ℝ)
  (h_ratio : 5 * w = 2 * l + 2 * w)
  (h_area : l * w = 150) :
  l = 15 :=
sorry

end rectangle_length_15_l622_622734


namespace find_y_l622_622589

-- Hypotheses
variable (x y : ℤ)

-- Given conditions
def condition1 : Prop := x = 4
def condition2 : Prop := x + y = 0

-- The goal is to prove y = -4 given the conditions
theorem find_y (h1 : condition1 x) (h2 : condition2 x y) : y = -4 := by
  sorry

end find_y_l622_622589


namespace num_of_valid_arrangements_l622_622613

theorem num_of_valid_arrangements (digits : Multiset ℕ) (condition : 0 ∉ digits.multiset_to_string.head) :
  digits = {3, 0, 5, 7, 0} → 
  Multiset.card digits = 5 → 
  ∑ s in Multiset.permutations digits, if s.head ≠ 0 then 1 else 0 = 48 :=
by
  sorry

end num_of_valid_arrangements_l622_622613


namespace factor_difference_of_squares_l622_622892

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l622_622892


namespace coordinates_on_y_axis_l622_622195

theorem coordinates_on_y_axis (a : ℝ) 
  (h : (a - 3) = 0) : 
  P = (0, -1) :=
by 
  have ha : a = 3 := by sorry
  subst ha
  sorry

end coordinates_on_y_axis_l622_622195


namespace chef_uses_8_ounces_of_coconut_oil_l622_622826

section PastryChef

variables (baking_mix butter_coconut_oil : ℕ) (butter_remaining : ℕ := 4) (total_baking_mix : ℕ := 6) (ounces_per_cup : ℕ := 2)

-- Definitions based on conditions
def butter_needed (cups : ℕ) : ℕ := cups * ounces_per_cup
def butter_covered_mix (butter : ℕ) : ℕ := butter / ounces_per_cup
def remaining_cups (total_cups : ℕ) (covered_cups : ℕ) : ℕ := total_cups - covered_cups
def coconut_oil_needed (cups : ℕ) : ℕ := cups * ounces_per_cup

theorem chef_uses_8_ounces_of_coconut_oil :
  coconut_oil_needed (remaining_cups total_baking_mix (butter_covered_mix butter_remaining)) = 8 :=
by
  sorry

end PastryChef

end chef_uses_8_ounces_of_coconut_oil_l622_622826


namespace largest_x_l622_622911

theorem largest_x (x : ℝ) : 
  (∃ x, (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 6 * x = 7 * x - 2) → 
  (x ≤ 1) := sorry

end largest_x_l622_622911


namespace polyhedron_two_pairs_equal_faces_l622_622713

-- Define the condition that the polyhedron has similar triangular faces and is convex
variables (P : Type) [convex_polyhedron P]
variables (face : P → Set (triangle))
variables [∀ (f : P), similar_triangles (face f)]

-- Defining pairwise equality of faces
noncomputable def equal_faces (F1 F2 F3 F4 : P) : Prop :=
  (F1 = F2) ∧ (F3 = F4) ∧ (F1 ≠ F3) ∧ (F1 ≠ F4) ∧ (F2 ≠ F3) ∧ (F2 ≠ F4)

-- Stating the theorem
theorem polyhedron_two_pairs_equal_faces (P : Type) [convex_polyhedron P]
  (face : P → Set (triangle))
  [∀ (f : P), similar_triangles (face f)] :
  ∃ (F1 F2 F3 F4 : P), equal_faces F1 F2 F3 F4 :=
begin
  sorry
end

end polyhedron_two_pairs_equal_faces_l622_622713


namespace find_k_l622_622241

-- Definitions
def a (n : ℕ) : ℤ := 1 + (n - 1) * 2
def S (n : ℕ) : ℤ := n / 2 * (2 * 1 + (n - 1) * 2)

-- Main theorem statement
theorem find_k (k : ℕ) (h : S (k + 2) - S k = 24) : k = 5 :=
by sorry

end find_k_l622_622241


namespace factorial_fraction_simplification_l622_622457

theorem factorial_fraction_simplification : 
  (10! * 6! * 3!) / (9! * 7!) = 60 / 7 :=
by
  sorry

end factorial_fraction_simplification_l622_622457


namespace max_x_x1_xm1_find_x_l622_622679

-- Definition for max of three numbers
def max3 (a b c : ℝ) : ℝ := max (max a b) c

-- Definition for min of three numbers
def min3 (a b c : ℝ) : ℝ := min (min a b) c

-- Problem 1: Prove max {x, x+1, x-1} = x+1
theorem max_x_x1_xm1 (x : ℝ) : max3 x (x + 1) (x - 1) = x + 1 :=
sorry

-- Problem 2: Find x such that min {5, 5-2x, 2x+5} = max {2, x+1, 2x}
theorem find_x (x : ℝ) : 
  min3 5 (5 - 2 * x) (2 * x + 5) = max3 2 (x + 1) (2 * x) →
  (x = -(3 / 2) ∨ x = (5 / 4)) :=
sorry

end max_x_x1_xm1_find_x_l622_622679


namespace sin_lambda_alpha_odd_function_l622_622561

noncomputable def f (x : ℝ) (λ : ℝ) (α : ℝ) : ℝ :=
  if x >= 0 then
    x^2 + 2015 * x + Real.sin x
  else
    -x^2 + λ * x + Real.cos (x + α)

theorem sin_lambda_alpha_odd_function (λ α : ℝ) (hodd : ∀ x, f x λ α = -f (-x) λ α) :
  Real.sin (λ * α) = 1 :=
  sorry

end sin_lambda_alpha_odd_function_l622_622561


namespace solve_equation_l622_622012

noncomputable def percentage (p : Real) (of : Real) := (p / 100) * of

theorem solve_equation :
  let x : Real := -1473.48 in
  percentage 1240 350 + percentage 990 275 = percentage 860 x + percentage 530 (2250 - x) :=
by
  sorry

end solve_equation_l622_622012


namespace correct_propositions_l622_622076

def proposition1 (P1 P2 L: Type) [Plane P1] [Plane P2] [Line L] (h1 : P1 ∥ L) (h2 : P2 ∥ L) : Prop :=
  P1 ∥ P2

def proposition2 (P1 P2 P3: Type) [Plane P1] [Plane P2] [Plane P3] (h1 : P1 ∥ P3) (h2 : P2 ∥ P3) : Prop :=
  P1 ∥ P2

def proposition3 (L1 L2 L3: Type) [Line L1] [Line L2] [Line L3] (h1 : L1 ⟂ L3) (h2 : L2 ⟂ L3) : Prop :=
  L1 ∥ L2

def proposition4 (L1 L2 P: Type) [Line L1] [Line L2] [Plane P] (h1 : L1 ⟂ P) (h2 : L2 ⟂ P) : Prop :=
  L1 ∥ L2

theorem correct_propositions (P1 P2 P3 : Type) [Plane P1] [Plane P2] [Plane P3]
    (L1 L2 L3 : Type) [Line L1] [Line L2] [Line L3]
    (H1_1 : P1 ∥ L3) (H1_2 : P2 ∥ L3)
    (H2_1 : P1 ∥ P3) (H2_2 : P2 ∥ P3)
    (H3_1 : L1 ⟂ L3) (H3_2 : L2 ⟂ L3)
    (H4_1 : L1 ⟂ P3) (H4_2 : L2 ⟂ P3) :
  (∃ (h1: proposition1 P1 P2 L3 H1_1 H1_2) (h2: proposition3 L1 L2 L3 H3_1 H3_2).false) ∧
  proposition2 P1 P2 P3 H2_1 H2_2 ∧
  proposition4 L1 L2 P3 H4_1 H4_2 :=
by sorry

end correct_propositions_l622_622076


namespace largest_m_satisfying_inequality_l622_622101

theorem largest_m_satisfying_inequality :
  ∃ (m : ℕ), (∀ (a : Fin 2014 → ℝ), (∀ i j : Fin 2014, i ≤ j → a i ≥ a j) → 
  (a 0 + a 1 + ⋯ + a (m - 1) / m.toReal) ≥ 
  (√(∑ i, a i ^ 2 / 2014))) ∧ m = 44 :=
begin
  sorry
end

end largest_m_satisfying_inequality_l622_622101


namespace triangle_perimeter_eq_364_l622_622758

theorem triangle_perimeter_eq_364 
  (P Q R J : Type) 
  (PQ PR QR : ℕ)
  (anglePQR : P → Q → R → ℝ)
  (angle_bisector_Q : P → J → Q)
  (angle_bisector_R : P → J → R)
  (QJ : ℕ)
  (H1 : PQ = PR) 
  (H2 : QJ = 10) 
  (H3 : J = (angle_bisectors_intersection (anglePQR P Q R)))
  (H4 : positive_integer_side_lengths PQ PR QR) :
  (perimeter PQ PR QR = 2 * (PQ + QR/2)) := 
  sorry

end triangle_perimeter_eq_364_l622_622758


namespace cosine_sum_is_zero_l622_622675

theorem cosine_sum_is_zero (α : ℝ) : 
  cos α + cos (72 * (π / 180) + α) + cos (144 * (π / 180) + α) + cos (216 * (π / 180) + α) + cos (288 * (π / 180) + α) = 0 :=
by
  sorry

end cosine_sum_is_zero_l622_622675


namespace num_functions_l622_622643

theorem num_functions (A : Finset ℕ) :
  A = {1, 2, 3, 4, 5, 6, 7, 8} →
  (Σ f : {f : ℕ → ℕ // ∀ x, x ∈ A → f(f x) = c ∨ (∃ a b, a ∈ A ∧ b ∈ A ∧ b ≠ c ∧ f(f a) = b ∧ f(f x) = c) }) = 63488 :=
by
  intro hA
  sorry

end num_functions_l622_622643


namespace perimeter_convex_hull_le_sum_perimeters_l622_622149

noncomputable def perimeter_of_convex_hull (polygons : List (Set (ℝ × ℝ))) : ℝ :=
  sorry -- Placeholder for the actual definition

noncomputable def sum_perimeters (polygons : List (Set (ℝ × ℝ))) : ℝ :=
  sorry -- Placeholder for the actual definition

theorem perimeter_convex_hull_le_sum_perimeters (polygons : List (Set (ℝ × ℝ)))
  (all_convex : ∀ polygon ∈ polygons, convex polygon)
  (no_line_separates : ∀ (L : Set (ℝ × ℝ)) (H : is_line L),
    ¬ (∃ (P : Set (ℝ × ℝ)) (hP : P ∈ polygons), (∀ (p ∈ P), p ∉ L) ∧ (∃ (Q : Set (ℝ × ℝ)) (hQ : Q ∈ polygons), (Q ≠ P) ∧ (∀ (q ∈ Q), q ∈ L)))
  ) : perimeter_of_convex_hull polygons ≤ sum_perimeters polygons :=
  sorry -- Placeholder for the proof

end perimeter_convex_hull_le_sum_perimeters_l622_622149


namespace area_of_square_land_l622_622704

-- Define the problem conditions
variable (A P : ℕ)

-- Define the main theorem statement: proving area A given the conditions
theorem area_of_square_land (h₁ : 5 * A = 10 * P + 45) (h₂ : P = 36) : A = 81 := by
  sorry

end area_of_square_land_l622_622704


namespace solve_for_x_l622_622198

theorem solve_for_x (x : ℝ) : (sqrt (2 * x - 1) = sqrt 3) ↔ (x = 2) :=
by
  sorry

end solve_for_x_l622_622198


namespace paint_cost_for_cube_l622_622397

def cost_per_kg : ℝ := 36.50
def coverage_per_kg : ℝ := 16 -- in square feet
def cube_side_length : ℝ := 8 -- in feet

theorem paint_cost_for_cube :
  let surface_area := 6 * (cube_side_length ^ 2) in
  let paint_required := surface_area / coverage_per_kg in
  let total_cost := paint_required * cost_per_kg in
  total_cost = 876 :=
by
  sorry

end paint_cost_for_cube_l622_622397


namespace gcd_g_x_1155_l622_622443

def g (x : ℕ) := (4 * x + 5) * (5 * x + 3) * (6 * x + 7) * (3 * x + 11)

theorem gcd_g_x_1155 (x : ℕ) (h : x % 18711 = 0) : Nat.gcd (g x) x = 1155 := by
  sorry

end gcd_g_x_1155_l622_622443


namespace tan_alpha_through_point_l622_622985

theorem tan_alpha_through_point (α : ℝ) (x y : ℝ) (h : (x, y) = (3, 4)) : Real.tan α = 4 / 3 :=
sorry

end tan_alpha_through_point_l622_622985


namespace angle_of_lateral_face_of_pentagonal_prism_l622_622843

theorem angle_of_lateral_face_of_pentagonal_prism :
  ∀ (P : ∀ (a b : ℝ), angle_in_lateral_face P a b = 90) →
    (∀ t : ℝ, t = 90) :=
by
  intro P
  sorry

end angle_of_lateral_face_of_pentagonal_prism_l622_622843


namespace tangent_segments_area_l622_622468

theorem tangent_segments_area (r : ℝ) (l : ℝ) (area : ℝ) :
  r = 4 ∧ l = 6 → area = 9 * Real.pi :=
by
  sorry

end tangent_segments_area_l622_622468


namespace find_sum_l622_622621

variable {a : ℕ → ℝ} {r : ℝ}

-- Conditions: a_n > 0 for all n
axiom pos : ∀ n : ℕ, a n > 0

-- Given equation: a_1 * a_5 + 2 * a_3 * a_5 + a_3 * a_7 = 25
axiom given_eq : a 1 * a 5 + 2 * a 3 * a 5 + a 3 * a 7 = 25

theorem find_sum : a 3 + a 5 = 5 :=
by
  sorry

end find_sum_l622_622621


namespace caleb_spent_more_on_ice_cream_l622_622858

theorem caleb_spent_more_on_ice_cream :
  ∀ (number_of_ic_cartons number_of_fy_cartons : ℕ)
    (cost_per_ic_carton cost_per_fy_carton : ℝ)
    (discount_rate sales_tax_rate : ℝ),
    number_of_ic_cartons = 10 →
    number_of_fy_cartons = 4 →
    cost_per_ic_carton = 4 →
    cost_per_fy_carton = 1 →
    discount_rate = 0.15 →
    sales_tax_rate = 0.05 →
    (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
     (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
      number_of_fy_cartons * cost_per_fy_carton) * sales_tax_rate) -
    (number_of_fy_cartons * cost_per_fy_carton) = 30 :=
by
  intros number_of_ic_cartons number_of_fy_cartons cost_per_ic_carton cost_per_fy_carton discount_rate sales_tax_rate
  sorry

end caleb_spent_more_on_ice_cream_l622_622858


namespace pascal_triangle_property_l622_622263

variable (m s : ℕ)

def binomial (n k : ℕ) : ℕ := Nat.binom n k

theorem pascal_triangle_property (h : ∀ k : ℕ, 1 ≤ k ∧ k < s → binomial s k = 0) :
  ∀ n : ℕ, n ≥ 2 → ∀ k : ℕ, 1 ≤ k ∧ k < s^n → binomial (s^n) k = 0 :=
by
  intros n hn k hk
  sorry

end pascal_triangle_property_l622_622263


namespace new_perimeter_after_adding_tiles_l622_622318

-- Define the original condition as per the problem statement
def original_T_shape (n : ℕ) : Prop :=
  n = 6

def original_perimeter (p : ℕ) : Prop :=
  p = 12

-- Define hypothesis required to add three more tiles while sharing a side with existing tiles
def add_three_tiles_with_shared_side (original_tiles : ℕ) (new_tiles_added : ℕ) : Prop :=
  original_tiles + new_tiles_added = 9

-- Prove the new perimeter after adding three tiles to the original T-shaped figure
theorem new_perimeter_after_adding_tiles
  (n : ℕ) (p : ℕ) (new_tiles : ℕ) (new_p : ℕ)
  (h1 : original_T_shape n)
  (h2 : original_perimeter p)
  (h3 : add_three_tiles_with_shared_side n new_tiles)
  : new_p = 16 :=
sorry

end new_perimeter_after_adding_tiles_l622_622318


namespace vector_norm_range_l622_622153

variable {V : Type} [InnerProductSpace ℝ V]

theorem vector_norm_range (a e : V) (h_unit : ∥e∥ = 1)
  (h : 1/2 ≤ inner a e ∧ inner a e ≤ 1) :
  1/2 ≤ ∥a∥ ∧ ∥a∥ ≤ 1 :=
sorry

end vector_norm_range_l622_622153


namespace C1_standard_equation_C2_cartesian_equation_min_distance_PQ_l622_622222

-- Define curve C1 in Cartesian coordinates
def C1_parametric_equations (α : ℝ) : ℝ × ℝ :=
(⟨sqrt 3 * cos α, sin α⟩)

-- Define curve C2 in polar coordinates
noncomputable def C2_polar_equation (ρ θ : ℝ) : Prop :=
ρ * sin (θ + π / 4) = 2 * sqrt 2

-- Define the coordinates transformation
def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
(ρ * cos θ, ρ * sin θ)

-- Prove the standard equation of C1
theorem C1_standard_equation : ∀ α : ℝ,
  let p := C1_parametric_equations α in
  p.1^2 / 3 + p.2^2 = 1 :=
by
  intro α
  let p := C1_parametric_equations α
  have h1 : p.1 = sqrt 3 * cos α := rfl
  have h2 : p.2 = sin α := rfl
  calc
    p.1^2 / 3 + p.2^2
        = (sqrt 3 * cos α)^2 / 3 + (sin α)^2 : by rw [h1, h2]
    ... = 3 * (cos α)^2 / 3 + (sin α)^2 : by rw sqr_sqrt (le_of_lt (cos α).pow_pos) -- non-negative assumption
    ... = (cos α)^2 + (sin α)^2 : by ring
    ... = 1 : by rw cos_sq_add_sin_sq α

-- Prove the Cartesian equation of C2
theorem C2_cartesian_equation : ∀ ρ θ : ℝ,
  C2_polar_equation ρ θ →
  let p := polar_to_cartesian ρ θ in
  p.1 + p.2 - 4 = 0 :=
by
  intros ρ θ h
  let p := polar_to_cartesian ρ θ
  have h1 : p.1 = ρ * cos θ := rfl
  have h2 : p.2 = ρ * sin θ := rfl
  have h3 : cos (θ + π / 4) = (cos θ - sin θ)/ sqrt 2 := sorry
  have h4 : sin (θ + π / 4) = (sin θ + cos θ) / sqrt 2 := sorry
  sorry  -- proof that p.1 + p.2 = 4

-- Prove the minimum distance between P and Q
theorem min_distance_PQ (P Q : ℝ × ℝ) :
  (∃α, P = C1_parametric_equations α) →
  (∃(ρ θ), Q = polar_to_cartesian ρ θ ∧ C2_polar_equation ρ θ) →
  |P.1 - Q.1|^2 + |P.2 - Q.2|^2 = 2
  ∧ P = (⟨3 / 2, 1 / 2⟩) :=
by
  intros hP hQ
  sorry  -- Proof of minimal distance and coordinates of P

end C1_standard_equation_C2_cartesian_equation_min_distance_PQ_l622_622222


namespace percentage_of_male_geese_is_50_l622_622605

noncomputable def percentage_of_male_geese (total_geese: ℝ) (migrating_geese: ℝ) : ℝ :=
  let M_ratio := 0.20 in
  let F_ratio := 0.80 in
  let migration_rate_ratio := 0.25 in
  let M := 100 in
  let F := 100 - M in
  have h1 : F = 100 - M,
  { sorry },
  have Rm : ℝ := (M_ratio * migrating_geese) / (M * total_geese),
  have Rf : ℝ := (F_ratio * migrating_geese) / (F * total_geese),
  have h2 : Rm / Rf = migration_rate_ratio,
  { sorry },
  have ratio_eq : M / F = migration_rate_ratio * (F_ratio / M_ratio),
  { sorry },
  have h3 : M / F = 1,
  { sorry },
  have h4 : M + F = 100,
  { sorry },
by
  exact 50

theorem percentage_of_male_geese_is_50 (total_geese: ℝ) (migrating_geese: ℝ) :
  percentage_of_male_geese total_geese migrating_geese = 50 := 
by
  sorry

end percentage_of_male_geese_is_50_l622_622605


namespace how_many_digits_D_divisible_by_7_l622_622930

theorem how_many_digits_D_divisible_by_7 :
  (∃! D : ℕ, D ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 7 ∣ (400 + 10 * D + 5)) :=
sorry

end how_many_digits_D_divisible_by_7_l622_622930


namespace minimum_value_ineq_l622_622259

noncomputable def minimum_value (a b : ℝ) : ℝ := 
  \frac{1}{a} + \frac{1}{b}

theorem minimum_value_ineq (a b : ℝ) (h : a > 0 ∧ b > 0) (h1 : a + 3 * b = 1) :
  minimum_value a b = 4 + 2 * \sqrt{3} :=
sorry

end minimum_value_ineq_l622_622259


namespace subsets_union_intersection_count_l622_622005

-- Define the set T
def T : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the problem statement
theorem subsets_union_intersection_count :
  let count := 
    let choose_intersection := Nat.choose 6 3 in
    let distribute_remaining := 2^3 in
    (choose_intersection * distribute_remaining) / 2 in
  count = 80 := by
sorry

end subsets_union_intersection_count_l622_622005


namespace find_a_plus_b_l622_622537

theorem find_a_plus_b (a b : ℝ) 
  (h_a : a^3 - 3 * a^2 + 5 * a = 1) 
  (h_b : b^3 - 3 * b^2 + 5 * b = 5) : 
  a + b = 2 := 
sorry

end find_a_plus_b_l622_622537


namespace value_expression_l622_622582

-- Define the conditions
variables (m n p q : ℝ)
hypothesis h1 : m + n = 0
hypothesis h2 : p * q = 1

-- State the theorem to be proved
theorem value_expression : -2023 * m + 3 / (p * q) - 2023 * n = 3 :=
by
  sorry

end value_expression_l622_622582


namespace count_three_marbles_with_yellow_l622_622361

def red_marble : Prop := true
def green_marble : Prop := true
def blue_marble : Prop := true
def orange_marble : Prop := true
def yellow_marble: Prop := true

theorem count_three_marbles_with_yellow :
  ∃ (count : ℕ), count = 11 :=
by
  let red := 1
  let green := 1
  let blue := 1
  let orange := 1
  let yellow := 4
  have H : (choose 4 3 + 4 * choose 4 2 + 6 * choose 4 1 = 11) := sorry
  use 11
  exact H

end count_three_marbles_with_yellow_l622_622361


namespace probability_of_lamps_arrangement_l622_622285

noncomputable def probability_lava_lamps : ℚ :=
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_turn_on := 4
  let left_red_on := 1
  let right_blue_off := 1
  let ways_to_choose_positions := Nat.choose total_lamps red_lamps
  let ways_to_choose_turn_on := Nat.choose total_lamps total_turn_on
  let remaining_positions := total_lamps - left_red_on - right_blue_off
  let remaining_red_lamps := red_lamps - left_red_on
  let remaining_turn_on := total_turn_on - left_red_on
  let arrangements_of_remaining_red := Nat.choose remaining_positions remaining_red_lamps
  let arrangements_of_turn_on :=
    Nat.choose (remaining_positions - right_blue_off) remaining_turn_on
  -- The probability calculation
  (arrangements_of_remaining_red * arrangements_of_turn_on : ℚ) / 
    (ways_to_choose_positions * ways_to_choose_turn_on)

theorem probability_of_lamps_arrangement :
    probability_lava_lamps = 4 / 49 :=
by
  sorry

end probability_of_lamps_arrangement_l622_622285


namespace tangent_line_at_x1_bound_on_f_minus_1_l622_622647

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x
noncomputable def f' (x : ℝ) : ℝ := (Real.exp x * x - (Real.exp x - 1)) / (x^2)

theorem tangent_line_at_x1 :
  let t := x - y + Real.exp 1 - 2 in
  t = 0 → ∀ (x : ℝ), f(1) = Real.exp 1 - 1 ∧ f'(1) = 1 :=
by sorry

theorem bound_on_f_minus_1 (a : ℝ) (h : a > 0) (x : ℝ) :
  0 < |x| ∧ |x| < Real.log (1 + a) → 
  |f x - 1| < a :=
by sorry

end tangent_line_at_x1_bound_on_f_minus_1_l622_622647


namespace common_ratio_of_geometric_sequence_l622_622608

open BigOperators

theorem common_ratio_of_geometric_sequence
  (a1 : ℝ) (q : ℝ)
  (h1 : 2 * (a1 * q^5) = 3 * (a1 * (1 - q^4) / (1 - q)) + 1)
  (h2 : a1 * q^6 = 3 * (a1 * (1 - q^5) / (1 - q)) + 1)
  (h_pos : a1 > 0) :
  q = 3 :=
sorry

end common_ratio_of_geometric_sequence_l622_622608


namespace find_p_plus_q_l622_622002

noncomputable def calculate_p_plus_q (DE EF FD WX : ℕ) (Area : ℕ → ℝ) : ℕ :=
  let s := (DE + EF + FD) / 2
  let triangle_area := (Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))) / 2
  let delta := triangle_area / (225 * WX)
  let gcd := Nat.gcd 41 225
  let p := 41 / gcd
  let q := 225 / gcd
  p + q

theorem find_p_plus_q : calculate_p_plus_q 13 30 19 15 (fun θ => 30 * θ - (41 / 225) * θ^2) = 266 := by
  sorry

end find_p_plus_q_l622_622002


namespace not_possible_area_l622_622191

theorem not_possible_area (P Q R S : Point) 
  (hP : P = (1, 0)) (hQ : Q = (2, 0)) (hR : R = (4, 0)) (hS : S = (8, 0))
  (form_square : ∃ (lines : Finset Line), lines.card = 4 ∧ forms_square lines [P, Q, R, S]) :
  ∀ S, S = (area_of_square [P, Q, R, S]) → S ≠ 26 / 5 :=
by
  sorry

end not_possible_area_l622_622191


namespace repeating_decimal_division_l622_622376

theorem repeating_decimal_division : 
  (0.\overline{81} : ℚ) = (81 / 99 : ℚ) →
  (0.\overline{36} : ℚ) = (36 / 99 : ℚ) → 
  (0.\overline{81} / 0.\overline{36} : ℚ) = (9 / 4 : ℚ) :=
by 
  intros h1 h2
  rw [h1, h2]
  change (_ / _) = (_ / _)
  sorry

end repeating_decimal_division_l622_622376


namespace fair_coin_999th_toss_probability_l622_622365

open ProbabilityTheory

/-- Consider a fair coin where each toss results in heads or tails, with probability 1/2 each. 
    Prove that the probability of getting heads on the 999th toss is 1/2. -/
theorem fair_coin_999th_toss_probability :
  let p : ProbabilityMassFunction (Fin 2) := ProbabilityMassFunction.uniform_of_fin 2 in
  p.mass 0 = 1/2 := 
by
  sorry

end fair_coin_999th_toss_probability_l622_622365


namespace probability_even_product_l622_622745

-- Define the set of numbers
def num_set : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define what it means for a product to be even
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define the event that the product of three numbers is even
def even_product_event (s : Finset ℕ) : Prop :=
  ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ is_even (a * b * c)

-- Statement to prove
theorem probability_even_product : 
  (Finset.card (Finset.filter (λ s, even_product_event s) (Finset.powerset_len 3 num_set))).toReal / 
  (Finset.card (Finset.powerset_len 3 num_set)).toReal = 19 / 20 := 
sorry

end probability_even_product_l622_622745


namespace sin_cos_lt_cos_sin_l622_622289

theorem sin_cos_lt_cos_sin {x : ℝ} (hx : 0 < x ∧ x < π / 2) : sin (cos x) < cos (sin x) :=
sorry

end sin_cos_lt_cos_sin_l622_622289


namespace ratio_of_numbers_l622_622735

theorem ratio_of_numbers (a b : ℕ) (h1 : a.gcd b = 5) (h2 : a.lcm b = 60) (h3 : a = 3 * 5) (h4 : b = 4 * 5) : (a / a.gcd b) / (b / a.gcd b) = 3 / 4 :=
by
  sorry

end ratio_of_numbers_l622_622735


namespace postcards_remainder_l622_622842

theorem postcards_remainder :
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  total % 15 = 3 :=
by
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  show total % 15 = 3
  sorry

end postcards_remainder_l622_622842


namespace sum_of_primitive_roots_is_8_l622_622775

def is_primitive_root_mod (a n : ℕ) : Prop :=
  ∀ b : ℕ, (1 ≤ b) → (b < n) → ∃ k : ℕ, (a ^ k) % n = b

noncomputable def set_of_primitive_roots_mod_11 : Finset ℕ :=
  {x ∈ (Finset.range 11) | is_primitive_root_mod x 11}

noncomputable def sum_of_primitive_roots_mod_11 : ℕ :=
  (set_of_primitive_roots_mod_11 ∩ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}).sum

theorem sum_of_primitive_roots_is_8 : sum_of_primitive_roots_mod_11 = 8 := by
  sorry

end sum_of_primitive_roots_is_8_l622_622775


namespace moles_of_C2H6_formed_l622_622117

-- Definitions of the quantities involved
def moles_H2 : ℕ := 3
def moles_C2H4 : ℕ := 3
def moles_C2H6 : ℕ := 3

-- Stoichiometry condition stated in a way that Lean can understand.
axiom stoichiometry : moles_H2 = moles_C2H4

theorem moles_of_C2H6_formed : moles_C2H6 = 3 :=
by
  -- Assume the constraints and state the final result
  have h : moles_H2 = moles_C2H4 := stoichiometry
  show moles_C2H6 = 3
  sorry

end moles_of_C2H6_formed_l622_622117


namespace max_xy_under_constraint_l622_622148

theorem max_xy_under_constraint (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 1 / 8 
  := sorry

end max_xy_under_constraint_l622_622148


namespace value_of_x_gt_x_sq_l622_622933

theorem value_of_x_gt_x_sq :
  ∃ x ∈ ({-2, -1/2, 0, 1/2, 2} : Set ℚ), x > x^2 :=
by
  use 1/2
  simp
  norm_num
  linarith

end value_of_x_gt_x_sq_l622_622933


namespace solve_digits_l622_622374

variables (h t u : ℕ)

theorem solve_digits :
  (u = h + 6) →
  (u + h = 16) →
  (∀ (x y z : ℕ), 100 * h + 10 * t + u + 100 * u + 10 * t + h = 100 * x + 10 * y + z ∧ y = 9 ∧ z = 6) →
  (h = 5 ∧ t = 4 ∧ u = 11) :=
sorry

end solve_digits_l622_622374


namespace function_has_zero_point_l622_622594

theorem function_has_zero_point {a b : ℝ} (h : a < b) 
  (f : ℝ → ℝ) (hf_cont : ContinuousOn f (set.Ioo a b)) 
  (hf_mono : MonotoneOn f (set.Ioo a b)) 
  (hf : f a * f b < 0) : 
  ∃! c ∈ set.Icc a b, f c = 0 := 
sorry

end function_has_zero_point_l622_622594


namespace max_mondays_in_45_days_l622_622718

theorem max_mondays_in_45_days : 
  ∀ (start_day : ℕ), start_day ∈ fin 7 →
  ∃ (max_mondays : ℕ), max_mondays = 7 ∧ 
  (∀ days, days <= 45 → days / 7 + 1 = max_mondays) :=
by
  sorry

end max_mondays_in_45_days_l622_622718


namespace brian_expenses_l622_622044

def cost_apples_per_bag : ℕ := 14
def cost_kiwis : ℕ := 10
def cost_bananas : ℕ := cost_kiwis / 2
def subway_fare_one_way : ℕ := 350
def maximum_apples : ℕ := 24

theorem brian_expenses : 
  cost_kiwis + cost_bananas + (cost_apples_per_bag * (maximum_apples / 12)) + (subway_fare_one_way * 2) = 50 := by
sorry

end brian_expenses_l622_622044


namespace adrians_speed_l622_622432

def initial_distance : ℝ := 289
def time : ℝ := 13
def remaining_distance : ℝ := 68

def distance_covered (d_initial d_remaining : ℝ) : ℝ :=
  d_initial - d_remaining

def speed (distance time : ℝ) : ℝ :=
  distance / time

theorem adrians_speed :
  speed (distance_covered initial_distance remaining_distance) time = 17 :=
by
  sorry

end adrians_speed_l622_622432


namespace evaluate_expression_l622_622939

theorem evaluate_expression (x y : ℕ) (hx : x = 4) (hy : y = 5) :
  (1 / (y : ℚ) / (1 / (x : ℚ)) + 2) = 14 / 5 :=
by
  rw [hx, hy]
  simp
  sorry

end evaluate_expression_l622_622939


namespace imaginary_part_of_z_l622_622650

noncomputable def z : ℂ := sorry -- define z according to the condition

axiom z_condition : z / (1 + complex.I) = 2 - 3 * complex.I

theorem imaginary_part_of_z : complex.im z = -1 :=
by {
  sorry
}

end imaginary_part_of_z_l622_622650


namespace only_integer_square_less_than_three_times_self_l622_622768

theorem only_integer_square_less_than_three_times_self :
  ∃! (x : ℤ), x^2 < 3 * x :=
by
  use 1
  split
  · -- Show that 1^2 < 3 * 1
    calc 1^2 = 1 : by norm_num
            ... < 3 : by norm_num
            ... = 3 * 1 : by norm_num
  · -- Show that for any x, if x^2 < 3 * x then x = 1
    intro y hy
    cases lt_or_ge y 1 with hy1 hy1
    · -- Case: y < 1
      exfalso
      calc y^2 ≥ 0 : by exact pow_two_nonneg y
              ... ≥ y * 3 - y : by linarith
              ...   = 3 * y - y : by ring
              ...   = 2 * y : by ring
      linarith
    cases lt_or_eq_of_le hy1 with hy1 hy1
    · -- Case: y = 2
      exfalso
      have h' := by linarith
      linarith
    · -- Case: y = 1
      exact hy1
    -- Case: y > 2
    exfalso
    calc y^2 ≥ y * 3 : by nlinarith
            ...   > y * 3 : by linarith
    linarith

end only_integer_square_less_than_three_times_self_l622_622768


namespace quadratic_max_value_l622_622978

theorem quadratic_max_value (a c : ℝ) (h : a ≠ 0) (h1 : c - a = 2) (a_neg : a < 0) : 
  ∃ x, ax^2 - 2a * x + c ≤ 2 :=
begin
  sorry
end

end quadratic_max_value_l622_622978


namespace discarded_flower_days_profit_n_14_profit_70_frequency_l622_622807

-- Define the daily demand and corresponding days
def daily_demand_counts : List (ℕ × ℕ) := [(13, 1), (14, 1), (15, 2), (16, 4), (17, 1), (18, 1)]

-- Define the profit functions
def profit (n : ℕ) : ℝ := if n < 16 then 10 * n - 80 else 80

-- 1. Prove the total number of days of discarded flowers is 4
theorem discarded_flower_days : (daily_demand_counts.filter (λ d => d.fst < 16)).map Prod.snd |> List.sum = 4 :=
by
  sorry

-- 2.1. Prove the profit when n = 14 is 60 yuan
theorem profit_n_14 : profit 14 = 60 :=
by
  sorry

-- 2.2. Prove the frequency of days with profit 70 yuan is 1/5
theorem profit_70_frequency : 
  let count_70 := (daily_demand_counts.filter (λ d => profit d.fst = 70)).map Prod.snd |> List.sum
  frequency_of_70 := (count_70 : ℝ) / 10 
  frequency_of_70 = 1 / 5 :=
by
  sorry

end discarded_flower_days_profit_n_14_profit_70_frequency_l622_622807


namespace Danny_reaches_Steve_house_in_29_minutes_l622_622865

variable {D : ℝ}

axiom (h1 : D / 2 = 14.5)

theorem Danny_reaches_Steve_house_in_29_minutes (h1 : D / 2 = 14.5) : D = 29 := by
  sorry

end Danny_reaches_Steve_house_in_29_minutes_l622_622865


namespace fx_of_f1_eq_one_solution_set_fx_greater_than_2_l622_622642

noncomputable def f : ℝ → ℝ := 
λ x, if x < 2 then 2 * Real.exp(x - 1) 
       else Real.logBase 3 (x^2 - 1)

theorem fx_of_f1_eq_one : f (f 1) = 1 := by
  sorry

theorem solution_set_fx_greater_than_2 :
  {x : ℝ | f x > 2} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | x > Real.sqrt 10} := by
  sorry

end fx_of_f1_eq_one_solution_set_fx_greater_than_2_l622_622642


namespace harmonic_series_increase_l622_622762

theorem harmonic_series_increase (k : ℕ) (hk : 1 < k) :
  (finset.card (finset.Ico (2^k) (2^(k+1))) = 2^k) :=
by
  sorry

end harmonic_series_increase_l622_622762


namespace find_C_l622_622648

noncomputable def h (C D : ℝ) (x : ℝ) : ℝ := 2 * C * x - 3 * D ^ 2
def k (D : ℝ) (x : ℝ) := D * x

theorem find_C (C D : ℝ) (h_eq : h C D (k D 2) = 0) (hD : D ≠ 0) : C = 3 * D / 4 :=
by
  unfold h k at h_eq
  sorry

end find_C_l622_622648


namespace even_function_b_zero_solve_inequality_l622_622161

theorem even_function_b_zero (b : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f x = x^2 + b * x + 1) 
  (h₂ : ∀ x : ℝ, f x = f (-x)) : b = 0 :=
by sorry

theorem solve_inequality (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f x = x^2 + 1) 
  (s : set ℝ) 
  (h₂ : s = {x : ℝ | 1 < x ∧ x < 2}): 
  ∀ x : ℝ, f(x - 1) < |x| ↔ x ∈ s :=
by sorry

end even_function_b_zero_solve_inequality_l622_622161


namespace karen_chairs_min_l622_622638

theorem karen_chairs_min (n : ℕ) (h1 : ∃ (Y : ℕ → Bool) (X : ℕ), Y n = true ∧ X > 1 ∧ X < n ∧ (∀ (d : ℕ), d > 1 ∧ d < n → (n % d = 0 ↔ Y d)) ∧ (∃ (s : Finset ℕ), Y ∈ s ∧ s.card = 9))
  : n = 288 :=
sorry

end karen_chairs_min_l622_622638


namespace sum_of_coefficients_l622_622454

theorem sum_of_coefficients (x y : ℤ) : 
  (∑ k1 k2 k3, (Multinomial 6 [k1, k2, k3] * (x^3) ^ k1 * (-3 * x * y^2) ^ k2 * (y^3) ^ k3) = -64) :=
by
  sorry

end sum_of_coefficients_l622_622454


namespace squirrel_travel_distance_l622_622833

-- Definitions from conditions
def height := 16 -- feet
def circumference := 2 -- feet
def vertical_rise := 4 -- feet

-- Let n be the number of circuits the squirrel makes
def num_circuits := height / vertical_rise

-- Calculate horizontal distance traveled
def horizontal_distance := num_circuits * circumference

-- Calculate the length of the helical path using the Pythagorean theorem
def length_of_helix := Real.sqrt (horizontal_distance^2 + height^2)

-- Prove the length of the helix is equal to approximately 17.89 feet
theorem squirrel_travel_distance : abs (length_of_helix - 17.89) < 0.01 :=
by
  sorry

end squirrel_travel_distance_l622_622833


namespace possible_rectangle_perimeters_l622_622496

def rect_perimeter_values (p_small : ℕ) : set ℕ :=
  if p_small = 10 then {14, 16, 18, 22, 26} else ∅

theorem possible_rectangle_perimeters :
  rect_perimeter_values 10 = {14, 16, 18, 22, 26} :=
sorry

end possible_rectangle_perimeters_l622_622496


namespace relationship_y1_y2_y3_l622_622960

-- Conditions
def y1 := -((-4)^2) + 5
def y2 := -((-1)^2) + 5
def y3 := -(2^2) + 5

-- Statement to be proved
theorem relationship_y1_y2_y3 : y2 > y3 ∧ y3 > y1 := by
  sorry

end relationship_y1_y2_y3_l622_622960


namespace sum_systematic_sample_l622_622288

theorem sum_systematic_sample {N : ℕ} (M B : ℕ) (n smallest_second smallest_first : ℕ) 
  (sample: Finset ℕ)
  (hN : N = 500)
  (hsmalls: smallest_second = 32)
  (hsmallf: smallest_first = 7)
  (hmethod: systematic_sample sample N smallest_first 25)
  (hin_sample : sample = Finset.image (fun (k : ℕ) => smallest_first + 25 * (k - 1)) (Finset.range n))
  (hcard : n = N / 25)
  : sample.sum id = 4890 := by
  sorry

end sum_systematic_sample_l622_622288


namespace curve_is_semicircle_l622_622596

theorem curve_is_semicircle :
  ∀ θ ∈ Icc (-π/2) (π/2), ∃ x y : ℝ, 
  (x = 2 * Real.cos θ ∧ y = 1 + 2 * Real.sin θ) ∧ 
  (x^2 + (y - 1)^2 = 4) ∧ 
  (0 ≤ x ∧ x ≤ 2) ∧ 
  (-1 ≤ y ∧ y ≤ 3) := 
by
  sorry

end curve_is_semicircle_l622_622596


namespace max_value_of_f_in_range_l622_622475

def op (a b : ℝ) : ℝ := if a >= b then a else b^2

def f (x : ℝ) : ℝ := (op 1 x) * x - (op 2 x)

theorem max_value_of_f_in_range :
  ∃ M, M = 6 ∧ ∀ x ∈ Icc (-2) (2), f x ≤ M :=
by
  use 6
  sorry

end max_value_of_f_in_range_l622_622475


namespace quadrilateral_area_is_120_l622_622812

def area_triangle {α : Type} {area : α -> ℝ} (A B C : α) : ℝ :=
  sorry  -- Insert the appropriate function for calculating triangle area

variable {α : Type} [decidable_eq α]

variable (A B C D O : α)

-- Given areas
def S_AOB := 10
def S_AOD := 20
def S_BOC := 30

-- Condition: These areas are less than the fourth triangle's area
def S_COD := 60

-- Sum of areas
def S_ABCD := S_AOB + S_AOD + S_BOC + S_COD

theorem quadrilateral_area_is_120 :
  S_ABCD = 120 :=
by
  -- Proof would go here
  sorry

end quadrilateral_area_is_120_l622_622812


namespace sacks_per_day_l622_622576

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (harvest_rate : ℕ)
  (h1 : total_sacks = 498)
  (h2 : days = 6)
  (h3 : harvest_rate = total_sacks / days) :
  harvest_rate = 83 := by
  sorry

end sacks_per_day_l622_622576


namespace solution_values_a_l622_622498

noncomputable def polynomial (k : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - Polynomial.X ^ 2 - Polynomial.C k

def discriminant_positive (k : ℝ) : Prop :=
  (polynomial k).discriminant > 0

theorem solution_values_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
  x^3 + y^2 + z^2 = a ∧ 
  x^2 + y^3 + z^2 = a ∧ 
  x^2 + y^2 + z^3 = a) ↔ a ∈ Ioo (23 / 27 : ℝ) 1 :=
sorry

end solution_values_a_l622_622498


namespace arith_seq_sum_eq_neg15_l622_622948

theorem arith_seq_sum_eq_neg15
  (a : ℕ → ℤ)
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_a2 : a 2 = -1)
  (h_a4 : a 4 = -5) :
  (∑ i in Finset.range 5, a (i + 1)) = -15 :=
by
  sorry

end arith_seq_sum_eq_neg15_l622_622948


namespace range_of_g_l622_622875

theorem range_of_g (x : ℝ) : 
  let g := λ x : ℝ, (Real.sin x)^4 + (Real.sin x)^2 * (Real.cos x)^2 + (Real.cos x)^4 in
  (Real.sin x)^2 + (Real.cos x)^2 = 1 →
  ∃ (y : ℝ), y ∈ Set.Icc (3/4) 1 ∧ g x = y :=
by
  intros
  let g := λ x : ℝ, (Real.sin x)^4 + (Real.sin x)^2 * (Real.cos x)^2 + (Real.cos x)^4
  sorry

end range_of_g_l622_622875


namespace tangent_line_to_curve_l622_622138

noncomputable def f : ℝ → ℝ := λ x => x^3 + 3 * x^2 - 1

def g (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

theorem tangent_line_to_curve :
  ∃ L : ℝ → ℝ, ∀ x y : ℝ, L x = -3 * x - 2 ∧ tangent_to_curve f L x y :=
begin
  sorry
end

def tangent_to_curve (f : ℝ → ℝ) (L : ℝ → ℝ) (x y : ℝ) : Prop := 
  (f' x = y' x ∧ L x = f x)

end tangent_line_to_curve_l622_622138


namespace C_necessary_but_not_sufficient_for_A_l622_622544

variable {A B C : Prop}

-- Given conditions
def sufficient_not_necessary (h : A → B) (hn : ¬(B → A)) := h
def necessary_sufficient := B ↔ C

-- Prove that C is a necessary but not sufficient condition for A
theorem C_necessary_but_not_sufficient_for_A (h₁ : A → B) (hn : ¬(B → A)) (h₂ : B ↔ C) : (C → A) ∧ ¬(A → C) :=
  by
  sorry

end C_necessary_but_not_sufficient_for_A_l622_622544


namespace area_ratio_l622_622533

variables (A B C I I1 L M N : Type)
variables [acute_angle_triangle ABC] (AB AC BC : ℝ)
variables [midpoint L BC] [incenter I ABC] [a_excenter I1 A ABC]
variables [intersection M (line_through L I) AC]
variables [intersection N (line_through L I1) AB]

theorem area_ratio (h : AB > AC) (a b c : ℝ) (h_abc : BC = a ∧ CA = b ∧ AB = c) :
  S (triangle LMN) / S (triangle ABC) = a * (c - b) / (a + c - b) ^ 2 := 
sorry

end area_ratio_l622_622533


namespace max_value_sine_expression_l622_622914

noncomputable def max_sine_expr : ℝ :=
  let f := λ (x : ℝ), 3 * Real.sin (x + Real.pi / 9) + 5 * Real.sin (x + 4 * Real.pi / 9)
  let max_value := 7 -- this is the maximum value found in the solution
  max_value

theorem max_value_sine_expression : ∃ x : ℝ, max_sine_expr = 7 :=
sorry

end max_value_sine_expression_l622_622914


namespace m_eq_half_l622_622159

theorem m_eq_half (m : ℝ) (h1 : m > 0) (h2 : ∀ x, (0 < x ∧ x < m) → (x * (x - 1) < 0))
  (h3 : ∃ x, (0 < x ∧ x < 1) ∧ ¬(0 < x ∧ x < m)) : m = 1 / 2 :=
sorry

end m_eq_half_l622_622159


namespace arithmetic_mean_is_correct_l622_622538

open Nat

noncomputable def arithmetic_mean_of_numbers (digits : List ℕ) (comma : ℕ) : ℝ :=
  sorry

theorem arithmetic_mean_is_correct :
  arithmetic_mean_of_numbers [1, 2, 4, 5, 8] ',' = 1234.4321 :=
by
  sorry

end arithmetic_mean_is_correct_l622_622538


namespace remaining_volume_after_cylinder_removed_l622_622059

def volume_rectangular_prism (l w h: ℝ) : ℝ :=
  l * w * h

def volume_cylinder (r h: ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem remaining_volume_after_cylinder_removed :
  let l := 5
  let w := 5
  let h := 6
  let r := 2.5
  let cylinder_height := 5
  volume_rectangular_prism l w h - volume_cylinder r cylinder_height = 150 - 31.25 * Real.pi :=
by
  sorry

end remaining_volume_after_cylinder_removed_l622_622059


namespace integral_approx_ln6_l622_622084

noncomputable def trapezoidal_rule_approx_integral (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : ℝ :=
  let Δx := (b - a) / n
  let y : Fin (n+1) → ℝ := λ i => f (a + i * Δx)
  Δx * (0.5 * (y 0 + y n) + (Finset.sum (Finset.finRange n) (λ i => y (i+1))))

theorem integral_approx_ln6 :
  let f := (λ x : ℝ => 1 / x)
  let a := 2
  let b := 12
  let n := 10
  | trapezoidal_rule_approx_integral f a b n - (Real.log 6) | < 0.02 :=
by
  simp [trapezoidal_rule_approx_integral, Real.log, f]
  sorry

end integral_approx_ln6_l622_622084


namespace max_mondays_in_45_days_l622_622719

theorem max_mondays_in_45_days (first_is_monday : Bool) : ∃ m, m ≤ 7 ∧ 
  ∀ n ≤ 45,
  let mondays := {i | i % 7 = 1} in
  mondays.count (λ i, i ≤ n) ≤ m :=
sorry

end max_mondays_in_45_days_l622_622719


namespace count_ab_bc_ca_l622_622375

noncomputable def count_ways : ℕ :=
  (Nat.choose 9 3)

theorem count_ab_bc_ca (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9) :
  (10 * a + b < 10 * b + c ∧ 10 * b + c < 10 * c + a) → count_ways = 84 :=
sorry

end count_ab_bc_ca_l622_622375


namespace second_shipment_is_13_l622_622403

-- Definitions based on the conditions
def first_shipment : ℕ := 7
def third_shipment : ℕ := 45
def total_couscous_used : ℕ := 13 * 5 -- 65
def total_couscous_from_three_shipments (second_shipment : ℕ) : ℕ :=
  first_shipment + second_shipment + third_shipment

-- Statement of the proof problem corresponding to the conditions and question
theorem second_shipment_is_13 (x : ℕ) 
  (h : total_couscous_used = total_couscous_from_three_shipments x) : x = 13 := 
by
  sorry

end second_shipment_is_13_l622_622403


namespace max_non_managers_depA_l622_622205

theorem max_non_managers_depA (mA : ℕ) (nA : ℕ) (sA : ℕ) (gA : ℕ) (totalA : ℕ) :
  mA = 9 ∧ (8 * nA > 37 * mA) ∧ (sA = 2 * gA) ∧ (nA = sA + gA) ∧ (mA + nA ≤ 250) →
  nA = 39 :=
by
  sorry

end max_non_managers_depA_l622_622205


namespace f_even_f_symmetry_l622_622417

def f (x : ℝ) : ℝ := Real.cos (↑(Int.pi) / 2 * x)

theorem f_even (x : ℝ) : f (-x) = f x :=
by
  unfold f
  have h : (↑(Int.pi) / 2) * (-x) = (↑(Int.pi) / 2) * x := by linarith
  rw [h, Real.cos_neg]

theorem f_symmetry (x : ℝ) : f (2 - x) + f x = 0 :=
by
  unfold f
  have h : Real.cos (↑(Int.pi) / 2 * (2 - x)) = -Real.cos (↑(Int.pi) / 2 * x) :=
    by sorry  -- This needs to verify the trigonometric identity.
  rw [h]
  simp

end f_even_f_symmetry_l622_622417


namespace calc_residue_modulo_l622_622086

theorem calc_residue_modulo :
  let a := 320
  let b := 16
  let c := 28
  let d := 5
  let e := 7
  let n := 14
  (a * b - c * d + e) % n = 3 :=
by
  sorry

end calc_residue_modulo_l622_622086


namespace area_of_field_l622_622829

-- Define the given conditions and the problem
theorem area_of_field (L W A : ℝ) (hL : L = 20) (hFencing : 2 * W + L = 88) (hA : A = L * W) : 
  A = 680 :=
by
  sorry

end area_of_field_l622_622829


namespace sum_b_squared_lt_l622_622226

noncomputable def a (n : ℕ) : ℝ := sorry  -- Given sequence {a_n}
noncomputable def S (n : ℕ) : ℝ := sorry  -- Sum of the first n terms of {a_n}
noncomputable def b (n : ℕ) : ℝ := (2 ^ (n - 1) + 1) / ((3 * n - 2) * a n)

axiom a1 (n : ℕ) : 0 < a n                -- Positive term sequence
axiom a1_1 : a 1 = 4					   -- Base case of {a_n}
axiom sum_a : S 1 = a 1                   -- Sum term condition for n = 1
axiom recurrence (n : ℕ) : 2 * S n = S (n + 1) + n

theorem sum_b_squared_lt : ∀ n : ℕ, 
  ∑ i in Finset.range (n + 1), (b (i + 1))^2 < 5 / 12 := 
sorry

end sum_b_squared_lt_l622_622226


namespace pa2_minus_qb2_eq_1_l622_622238

theorem pa2_minus_qb2_eq_1 (p q x y : ℤ) (hp : nat.prime p) (hq : nat.prime q)
  (hp_form : ∃ k, p = 4 * k + 3) (hq_form : ∃ k, q = 4 * k + 3)
  (hxy : x^2 - p * q * y^2 = 1) : 
  ∃ a b : ℤ, 0 < a ∧ 0 < b ∧ abs (p * a^2 - q * b^2) = 1 := 
sorry

end pa2_minus_qb2_eq_1_l622_622238


namespace find_angle_C_l622_622640

variables {A B C H M A' B' : Type} [RealField A] [RealField B] [RealField C] [RealField H] [RealField M] [RealField A'] [RealField B']

-- Assuming the conditions
-- 1. ABC is an acute-angled, nonisosceles triangle.
def isAcuteNonisoscelesTriangle (A B C : Type) [RealField A] [RealField B] [RealField C] : Prop := sorry

-- 2. Altitudes AA' and BB' meet at point H.
def altitudesMeetAtH (A A' B B' H : Type) [RealField A] [RealField A'] [RealField B] [RealField B'] [RealField H] : Prop := sorry

-- 3. The medians of triangle AHB meet at point M.
def mediansMeetAtM (A H B M : Type) [RealField A] [RealField H] [RealField B] [RealField M] : Prop := sorry

-- 4. Line CM bisects segment A'B'.
def lineCMBisects (C M A' B' : Type) [RealField C] [RealField M] [RealField A'] [RealField B'] : Prop := sorry

theorem find_angle_C (A B C H M A' B' : Type) 
  [RealField A] [RealField B] [RealField C] [RealField H] [RealField M] [RealField A'] [RealField B'] :
  isAcuteNonisoscelesTriangle A B C →
  altitudesMeetAtH A A' B B' H →
  mediansMeetAtM A H B M →
  lineCMBisects C M A' B' →
  ∠ C = 45 :=
by
  sorry

end find_angle_C_l622_622640


namespace coconut_oil_needed_l622_622825

def butter_per_cup := 2
def coconut_oil_per_cup := 2
def butter_available := 4
def cups_of_baking_mix := 6

theorem coconut_oil_needed : 
  let cups_covered_by_butter := butter_available / butter_per_cup in
  let cups_requiring_coconut_oil := cups_of_baking_mix - cups_covered_by_butter in
  coconut_oil_per_cup * cups_requiring_coconut_oil = 8 :=
by
  sorry

end coconut_oil_needed_l622_622825


namespace grid_black_probability_l622_622402

theorem grid_black_probability :
  let p_black_each_cell : ℝ := 1 / 3 
  let p_not_black : ℝ := (2 / 3) * (2 / 3)
  let p_one_black : ℝ := 1 - p_not_black
  let total_pairs : ℕ := 8
  (p_one_black ^ total_pairs) = (5 / 9) ^ 8 :=
sorry

end grid_black_probability_l622_622402


namespace perp_condition_l622_622152

noncomputable def λ := (-1 : ℚ) / 7

-- Given conditions
def a : ℝ × ℝ := (-3, 2)
def b : ℝ × ℝ := (-1, 0)

-- Statement to prove
theorem perp_condition : (λ * a + b) ⋅ (a - 2 * b) = 0 := by 
  -- Insert proof automation if required
  sorry

end perp_condition_l622_622152


namespace A_oplus_B_inter_C_l622_622099

def A : Set ℕ := {x | 1 < x ∧ x < 4}
def B : Set ℕ := {x | 1 < x ∧ x < 4}

def A_oplus_B : Set (ℕ × ℚ) := {p | ∃ (x y : ℕ), p = (x, y) ∧ x / 2 ∈ A ∧ 2 / y ∈ B}

def C : Set (ℕ × ℚ) := {p | ∃ (x : ℕ) (y : ℚ), p = (x, y) ∧ y = -1/6 * x + 5/3}

theorem A_oplus_B_inter_C : (A_oplus_B ∩ C) = {(4, 1 : ℚ), (6, 2/3 : ℚ)} := by
  sorry

end A_oplus_B_inter_C_l622_622099


namespace real_axis_length_of_hyperbola_l622_622724

theorem real_axis_length_of_hyperbola : 
  ∀ (x y : ℝ), 2 * x^2 - y^2 = 8 → 2 * (sqrt 4) = 4 :=
by
  intro x y h
  -- proof omitted
  sorry

end real_axis_length_of_hyperbola_l622_622724


namespace product_of_g_of_roots_l622_622249

noncomputable def f (x : ℝ) : ℝ := x^5 - 2*x^3 + x + 1
noncomputable def g (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem product_of_g_of_roots (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h₁ : f x₁ = 0) (h₂ : f x₂ = 0) (h₃ : f x₃ = 0)
  (h₄ : f x₄ = 0) (h₅ : f x₅ = 0) :
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = f (-1 + Real.sqrt 2) * f (-1 - Real.sqrt 2) :=
by
  sorry

end product_of_g_of_roots_l622_622249


namespace power_function_value_at_2_l622_622552

theorem power_function_value_at_2 :
  ∃ (a : ℝ), (∀ (x : ℝ), f x = x ^ a) ∧ f (1/2) = 8 → f 2 = 1/8 :=
by sorry

end power_function_value_at_2_l622_622552


namespace jill_average_number_of_stickers_l622_622231

def average_stickers (packs : List ℕ) : ℚ :=
  (packs.sum : ℚ) / packs.length

theorem jill_average_number_of_stickers :
  average_stickers [5, 7, 9, 9, 11, 15, 15, 17, 19, 21] = 12.8 :=
by
  sorry

end jill_average_number_of_stickers_l622_622231


namespace molecular_weight_l622_622008

-- Definitions of the molar masses of the elements
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_I : ℝ := 126.90
def molar_mass_Ca : ℝ := 40.08
def molar_mass_S : ℝ := 32.07
def molar_mass_O : ℝ := 16.00

-- Definition of the molar masses of the compounds
def molar_mass_NH4I : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_I
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

-- Number of moles
def moles_NH4I : ℝ := 3
def moles_CaSO4 : ℝ := 2

-- Total mass calculation
def total_mass : ℝ :=
  moles_NH4I * molar_mass_NH4I + 
  moles_CaSO4 * molar_mass_CaSO4

-- Problem statement
theorem molecular_weight : total_mass = 707.15 := by
  sorry

end molecular_weight_l622_622008


namespace outfits_count_l622_622392

variable (shirts : Fin 4)
variable (pants : Fin 5)
variable (same_color_shirt_pant : {sh: shirts // ∃ pt : pants, sh = pt})

theorem outfits_count : 19 = (4 * 5 - 1) := by
  sorry

end outfits_count_l622_622392


namespace no_polynomial_solution_l622_622324

theorem no_polynomial_solution {g : ℕ → ℕ} (h : ∀ x, g(x) * g(x) = 4 * x^3 + 4 * x^2 + 4 * x + 1) : false :=
sorry

end no_polynomial_solution_l622_622324


namespace f_neg_eq_f_l622_622588

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero :
  ∃ x, f x ≠ 0

axiom functional_equation :
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_neg_eq_f (x : ℝ) : f (-x) = f x := 
sorry

end f_neg_eq_f_l622_622588


namespace max_non_empty_intersection_subsets_l622_622253

theorem max_non_empty_intersection_subsets (n : ℕ) (h : n > 0) : 
  ∃ k, (∀ S : finset (finset (fin n)), S.card = k → (∀ {X Y : finset (fin n)}, X ∈ S → Y ∈ S → (X ∩ Y).nonempty) → k = 2^(n-1)) :=
begin
  sorry
end

end max_non_empty_intersection_subsets_l622_622253


namespace min_moves_to_heads_l622_622326

def alternating_coins (n : ℕ) : list bool :=
(list.repeat tt (n / 2)) ++ list.repeat ff (n / 2) ++
if n % 2 = 1 then [tt] else []

theorem min_moves_to_heads (coins : list bool) (h : coins = alternating_coins 2023) :
  ∃ m, (∀ i, coins.nth i = some tt) → m ≥ 4044 :=
begin
  sorry
end

end min_moves_to_heads_l622_622326


namespace y_relation_l622_622956

theorem y_relation (y1 y2 y3 : ℝ) : 
  (-4, y1) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) ∧
  (-1, y2) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) ∧
  (2, y3) ∈ set_of (λ p : ℝ × ℝ, p.snd = -p.fst^2 + 5) →
  y2 > y3 ∧ y3 > y1 :=
begin
  sorry
end

end y_relation_l622_622956


namespace conical_frustum_volume_l622_622527

noncomputable def area_of_equilateral_triangle (side : ℝ) : ℝ :=
  (sqrt 3 / 4) * side^2

noncomputable def volume_of_right_prism (base_area : ℝ) (height : ℝ) : ℝ :=
  base_area * height

noncomputable def volume_of_conical_frustum (prism_volume : ℝ) : ℝ :=
  (1 / 3) * prism_volume

theorem conical_frustum_volume (side length height : ℝ)
  (h₁ : side = 1)
  (h₂ : height = 3) :
  volume_of_conical_frustum (volume_of_right_prism (area_of_equilateral_triangle side) height) = (sqrt 3 / 4) :=
by
  sorry

end conical_frustum_volume_l622_622527


namespace construct_right_triangle_l622_622093

theorem construct_right_triangle (c m n : ℝ) (hc : c > 0) (hm : m > 0) (hn : n > 0) : 
  ∃ a b : ℝ, a^2 + b^2 = c^2 ∧ a / b = m / n :=
by
  sorry

end construct_right_triangle_l622_622093


namespace tangent_line_eq_l622_622500

noncomputable def curve (x : ℝ) := x^3 + 1

theorem tangent_line_eq (x y : ℝ) (h : y = curve x) :
  has_tangent_at (-1, 0) curve 3x - y + 3 = 0 := by
  sorry

end tangent_line_eq_l622_622500


namespace simplify_expression_l622_622298

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622298


namespace isosceles_right_triangle_area_l622_622722

theorem isosceles_right_triangle_area (h : ℝ) (A : ℝ) :
  (h = 5 * Real.sqrt 2) →
  (A = 12.5) →
  ∃ (leg : ℝ), (leg = 5) ∧ (A = 1 / 2 * leg^2) := by
  sorry

end isosceles_right_triangle_area_l622_622722


namespace minimum_area_of_circle_C_l622_622053

theorem minimum_area_of_circle_C :
  let y_eq_x_2sqrt2_1 := λ x y, y = x + 2 * Real.sqrt 2 + 1
  ∃ a b : ℝ, (b^2 = 4 * a) ∧ (a >= 1) ∧ (|a + 1| = a + 1) ∧ (∀ (x y : ℝ), y_eq_x_2sqrt2_1 x y → ((a - x)^2 + (b - y)^2) = (a + 1)^2) →
  let r := (a + 1)
  let area := π * r^2
  area = 4 * π :=
begin
  sorry
end

end minimum_area_of_circle_C_l622_622053


namespace probability_sum_prime_l622_622393

theorem probability_sum_prime (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) :
  let outcomes := finset.product (finset.range 1 7) (finset.range 1 7)
  let primes := {2, 3, 5, 7, 11}
  let prime_sums := outcomes.filter (λ (x : ℕ × ℕ), primes (x.1 + x.2))
  (prime_sums.card : ℚ) / (outcomes.card : ℚ) = 5 / 12 := sorry

end probability_sum_prime_l622_622393


namespace min_value_ineq_l622_622261

noncomputable def min_value (x y z : ℝ) := (1/x) + (1/y) + (1/z)

theorem min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 2) :
  min_value x y z ≥ 4.5 :=
sorry

end min_value_ineq_l622_622261


namespace rolls_remaining_to_sell_l622_622922

-- Conditions
def total_rolls_needed : ℕ := 45
def rolls_sold_to_grandmother : ℕ := 1
def rolls_sold_to_uncle : ℕ := 10
def rolls_sold_to_neighbor : ℕ := 6

-- Theorem statement
theorem rolls_remaining_to_sell : (total_rolls_needed - (rolls_sold_to_grandmother + rolls_sold_to_uncle + rolls_sold_to_neighbor) = 28) :=
by
  sorry

end rolls_remaining_to_sell_l622_622922


namespace boat_ride_impossible_l622_622919

theorem boat_ride_impossible : ¬ (∃ (crossings : set (set (fin 5))), 
  (∀ group ∈ crossings, group.nonempty) ∧
  (∀ group, group ⊆ finset.univ(fin 5) → group.nonempty → group ∈ crossings) ∧
  (card crossings = 2^5 - 1)) :=
sorry

end boat_ride_impossible_l622_622919


namespace poly_sequence_correct_l622_622063

-- Sequence of polynomials defined recursively
def f : ℕ → ℕ → ℕ 
| 0, x => 1
| 1, x => 1 + x 
| (k + 1), x => ((x + 1) * f (k) (x) - (x - k) * f (k - 1) (x)) / (k + 1)

-- Prove f(k, k) = 2^k for all k ≥ 0
theorem poly_sequence_correct (k : ℕ) : f k k = 2 ^ k := by
  sorry

end poly_sequence_correct_l622_622063


namespace f_x_notin_setM_f_ax_in_setM_f_sin_kx_in_setM_l622_622531

def setM (f : ℝ → ℝ) : Prop :=
  ∃ T ≠ 0, ∀ x : ℝ, f(x + T) = T * f(x)

theorem f_x_notin_setM : ¬ setM (λ x : ℝ, x) :=
sorry

theorem f_ax_in_setM (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (T : ℝ) (hT : a^T = T) : setM (λ x : ℝ, a^x) :=
sorry

theorem f_sin_kx_in_setM (k : ℝ) (h : setM (λ x : ℝ, sin(k * x))) : 
  ∃ m : ℤ, k = m * Real.pi :=
sorry

end f_x_notin_setM_f_ax_in_setM_f_sin_kx_in_setM_l622_622531


namespace simplify_expr_l622_622302

theorem simplify_expr : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expr_l622_622302


namespace find_f_l622_622111

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f :
  (∀ (x y : ℝ), 0 < x → 0 < y → f (x ^ y) = f x ^ f y) → (∀ (x : ℝ), 0 < x → f x = x) :=
begin
  intros h x hx,
  sorry
end

end find_f_l622_622111


namespace sum_base7_l622_622839

theorem sum_base7 : (26₇ + 64₇ + 135₇ = 261₇) := by
  sorry

end sum_base7_l622_622839


namespace find_x_of_series_eq_100_l622_622901

noncomputable def infinite_series : ℕ → ℚ
| 0       := 2
| (n + 1) := 2 + 4 * (n + 1)

theorem find_x_of_series_eq_100 :
  (∑' n : ℕ, infinite_series n * x^n = 100) → x = 1 / 2 :=
sorry

end find_x_of_series_eq_100_l622_622901


namespace conjugate_times_imaginary_unit_l622_622543

noncomputable def imaginary_unit : ℂ := complex.I
noncomputable def complex_conjugate (z : ℂ) : ℂ := complex.conj z

theorem conjugate_times_imaginary_unit (z : ℂ) 
  (h : (1 - imaginary_unit) * z = 2) :
  complex_conjugate z * imaginary_unit = 1 + imaginary_unit := 
sorry

end conjugate_times_imaginary_unit_l622_622543


namespace fourth_place_points_l622_622209

variables (x : ℕ)

def points_awarded (place : ℕ) : ℕ :=
  if place = 1 then 11
  else if place = 2 then 7
  else if place = 3 then 5
  else if place = 4 then x
  else 0

theorem fourth_place_points:
  (∃ a b c y u : ℕ, a + b + c + y + u = 7 ∧ points_awarded x 1 ^ a * points_awarded x 2 ^ b * points_awarded x 3 ^ c * points_awarded x 4 ^ y * 1 ^ u = 38500) →
  x = 4 :=
sorry

end fourth_place_points_l622_622209


namespace shadow_area_correct_l622_622813

-- Define the edge length of the cube
def edge_length : ℝ := 8

-- Define the diagonal of the cube
def diagonal (a : ℝ) : ℝ := a * sqrt 3

-- Define the area of the shadow (regular hexagon) formula
def shadow_area (a : ℝ) := 6 * (a^2 / 4 * sqrt 3)

-- Given conditions and prove the required value
theorem shadow_area_correct (a b : ℕ) (ha : a = 64) (hb : b = 3) : 
  ∃ (a b : ℕ), shadow_area edge_length = a * (sqrt b) ∧ a + b = 67 :=
by 
  use [64, 3]
  simp [shadow_area, edge_length, sqrt_eq_rpow]
  sorry

end shadow_area_correct_l622_622813


namespace range_of_a_l622_622563

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * x^2 - a * x + log x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := x - a + 1 / x

-- Define the condition for having extreme points
def has_two_extreme_points (f' : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f' x1 = 0 ∧ f' x2 = 0

theorem range_of_a (a : ℝ) : (a > 2) ↔ has_two_extreme_points (f' a) :=
sorry

end range_of_a_l622_622563


namespace smallest_N_divisible_by_170_l622_622125

-- Definitions for the problem conditions
def is_two_digit_number (N : ℕ) : Prop := 10 ≤ N ∧ N < 100

-- Definition for the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := (n.to_digits 10).sum

-- Main theorem statement
theorem smallest_N_divisible_by_170 :
  ∃ N : ℕ, is_two_digit_number N ∧ sum_of_digits (10^N - N) % 170 = 0 ∧ ∀ M : ℕ, is_two_digit_number M → M < N → sum_of_digits (10^M - M) % 170 ≠ 0 :=
sorry

end smallest_N_divisible_by_170_l622_622125


namespace merchant_marking_percentage_l622_622415

theorem merchant_marking_percentage (L : ℝ) (p : ℝ) (d : ℝ) (c : ℝ) (profit : ℝ) 
  (purchase_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (list_price : ℝ) : 
  L = 100 ∧ p = 30 ∧ d = 20 ∧ c = 20 ∧ profit = 20 ∧ 
  purchase_price = L - L * (p / 100) ∧ 
  marked_price = 109.375 ∧ 
  selling_price = marked_price - marked_price * (d / 100) ∧ 
  selling_price - purchase_price = profit * (selling_price / 100) 
  → marked_price = 109.375 := by sorry

end merchant_marking_percentage_l622_622415


namespace not_all_red_after_finite_years_l622_622078

def flower_color := ℤ

def initial_flower_config : list flower_color := [1, 1, 1, 1, 1, -1, -1, -1, -1]

def next_flower (a b : flower_color) : flower_color := a * b

def next_year_flowers (flowers : list flower_color) : list flower_color :=
  (list.zip_with next_flower flowers (flowers.tail ++ [flowers.head]))

lemma product_invariant (flowers : list flower_color) : 
  flowers.prod = (next_year_flowers flowers).prod := 
by sorry

theorem not_all_red_after_finite_years (n : ℕ) : 
  ∀ (flowers : list flower_color), 
    flowers = initial_flower_config →
    flowers.prod = 1 →
    ∀ (years : list (list flower_color)), 
      years = list.iterate next_year_flowers n flowers →
      ∃ (final_flowers : list flower_color),
        final_flowers ∈ years ∧ 
        final_flowers ≠ list.replicate 9 1 :=
by sorry

end not_all_red_after_finite_years_l622_622078


namespace max_mondays_in_45_days_l622_622717

theorem max_mondays_in_45_days : 
  ∀ (start_day : ℕ), start_day ∈ fin 7 →
  ∃ (max_mondays : ℕ), max_mondays = 7 ∧ 
  (∀ days, days <= 45 → days / 7 + 1 = max_mondays) :=
by
  sorry

end max_mondays_in_45_days_l622_622717


namespace isosceles_trapezoid_l622_622367

-- We define the necessary conditions
variables {A B C D : Point}
variables {BC AD AC BD : length}
variable {angle_A_C_B_D : ℝ}

-- We assume the conditions given in the problem
axiom trapezoid (h₁ : Trapezoid ABCD) : True
axiom base_lengths (h₂ : AC = BC + AD) : True
axiom angle_condition (h₃ : angle_A_C_B_D = 60) : True

-- We state the theorem to be proved
theorem isosceles_trapezoid
  (h₁ : trapezoid)
  (h₂ : base_lengths)
  (h₃ : angle_condition) :
  is_isosceles_trapezoid ABCD :=
sorry

end isosceles_trapezoid_l622_622367


namespace wrapping_paper_per_present_l622_622684

theorem wrapping_paper_per_present :
  ∀ (total: ℚ) (presents: ℚ) (frac_used: ℚ),
  total = 3 / 10 → presents = 3 → frac_used = total / presents → frac_used = 1 / 10 :=
by
  intros total presents frac_used htotal hpresents hfrac
  rw [htotal, hpresents, hfrac]
  sorry

end wrapping_paper_per_present_l622_622684


namespace divisor_is_three_l622_622819

theorem divisor_is_three (n d q p : ℕ) (h1 : n = d * q + 3) (h2 : n^2 = d * p + 3) : d = 3 := 
sorry

end divisor_is_three_l622_622819


namespace perfect_square_factors_count_l622_622478

open Nat

theorem perfect_square_factors_count : 
  let n := (2 ^ 12) * (3 ^ 10) * (7 ^ 8)
  count_perfect_square_factors n = 210 := 
by
  sorry

end perfect_square_factors_count_l622_622478


namespace no_borrowing_pairs_l622_622510

/-- 
Given the set of consecutive integers from 950 to 1050, 
proves that there are exactly 10 pairs of consecutive integers 
where no borrowing is required when the integers are subtracted.
-/
theorem no_borrowing_pairs :
  let s := {x : ℕ | 950 ≤ x ∧ x ≤ 1050} in
  ∃! k : ℕ, k = 10 ∧
    ∀ (a b : ℕ), a ∈ s → b = a + 1 → ((a % 10) < (b % 10) → k = 10) :=
by
  sorry

end no_borrowing_pairs_l622_622510


namespace probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l622_622130

theorem probability_exactly_2_boys_1_girl 
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (Nat.choose boys 2 * Nat.choose girls 1 / (Nat.choose total_group select) : ℚ) = 3 / 5 :=
by sorry

theorem probability_at_least_1_girl
  (boys girls : ℕ) 
  (total_group : ℕ) 
  (select : ℕ)
  (h_boys : boys = 4) 
  (h_girls : girls = 2) 
  (h_total_group : total_group = boys + girls) 
  (h_select : select = 3) : 
  (1 - (Nat.choose boys select / Nat.choose total_group select : ℚ)) = 4 / 5 :=
by sorry

end probability_exactly_2_boys_1_girl_probability_at_least_1_girl_l622_622130


namespace minimize_integral_l622_622799

noncomputable def p (a b : ℝ) : ℝ := 2 / (a + b)
noncomputable def q (a b : ℝ) : ℝ := Real.log((a + b) / 2) - 1
noncomputable def integral (a b : ℝ) : ℝ := 
  ∫ x in a..b, (p a b * x + q a b - Real.log x)

theorem minimize_integral (a b : ℝ) (h : 0 < a) (h2 : a < b) :
  (∀ x, a ≤ x ∧ x ≤ b → p a b * x + q a b ≥ Real.log x) ∧ 
  integral a b = (b - a) * Real.log((a + b) / 2) + b - a - b * Real.log b + a * Real.log a :=
sorry

end minimize_integral_l622_622799


namespace number_approximation_l622_622902

def find_number : ℝ :=
  sorry

theorem number_approximation :
  ∃ x : ℝ, 0.6667 * x - 0.25 * x = 10 ∧ x ≈ 23.9936 := 
by
  use 23.9936
  sorry

end number_approximation_l622_622902


namespace initial_percentage_water_l622_622487

theorem initial_percentage_water (W_initial W_final N_initial N_final : ℝ) (h1 : W_initial = 100) 
    (h2 : N_initial = W_initial - W_final) (h3 : W_final = 25) (h4 : W_final / N_final = 0.96) : N_initial / W_initial = 0.99 := 
by
  sorry

end initial_percentage_water_l622_622487


namespace initial_number_of_mice_l622_622372

theorem initial_number_of_mice (x : ℕ) 
  (h1 : x % 2 = 0)
  (h2 : (x / 2) % 3 = 0)
  (h3 : (x / 2 - x / 6) % 4 = 0)
  (h4 : (x / 2 - x / 6 - (x / 2 - x / 6) / 4) % 5 = 0)
  (h5 : (x / 5) = (x / 6) + 2) : 
  x = 60 := 
by sorry

end initial_number_of_mice_l622_622372


namespace rocks_ratio_l622_622852

/-- Define the problem conditions. -/
def bill_throws_sticks := λ ted_throws_sticks : ℕ, ted_throws_sticks + 6
def ted_throws_sticks : ℕ := 10
def ted_throws_rocks : ℕ := 10
def bill_throws_rocks : ℕ := 0

/-- The ratio of the number of rocks Ted tosses to Bill tosses is 10 to 0. -/
theorem rocks_ratio : (ted_throws_rocks, bill_throws_rocks) = (10, 0) :=
by
  apply congr_arg2
  · refl
  · refl
#check rocks_ratio -- This is just to ensure the statement is correctly formulated.

end rocks_ratio_l622_622852


namespace ratio_of_inscribed_square_to_large_square_l622_622862

-- Define the conditions given in the problem statement
def largeSquareSideLength : ℝ := 1
def quarteredSegment (sideLength : ℝ) : ℝ := sideLength / 4
def inscribedSquareSideLength : ℝ := Real.sqrt ((quarteredSegment largeSquareSideLength * 2) / 2)

-- Define the areas of the inscribed and large squares
def area_largeSquare : ℝ := largeSquareSideLength ^ 2
def area_inscribedSquare : ℝ := inscribedSquareSideLength ^ 2

-- Define the ratio of the areas
def ratio_of_areas (area_large : ℝ) (area_inscribed : ℝ) : ℝ := area_inscribed / area_large

-- Prove that the ratio of the area of the inscribed square to the area of the large square is 1/2
theorem ratio_of_inscribed_square_to_large_square 
  (h1 : largeSquareSideLength = 1)
  (h2 : inscribedSquareSideLength = Real.sqrt ((quarteredSegment largeSquareSideLength * 2) / 2)) :
  ratio_of_areas area_largeSquare area_inscribedSquare = 1 / 2 :=
by
  sorry

end ratio_of_inscribed_square_to_large_square_l622_622862


namespace odd_function_f_l622_622542

noncomputable def log_base_half (x : ℝ) : ℝ :=
  log x / log (1 / 2)

theorem odd_function_f (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x)
  (h_neg_domain : ∀ x, x ≤ 0 → f x = log_base_half (-x + 1)) :
  ∀ x, x > 0 → f x = -log_base_half (x + 1) :=
by
  intro x hx
  have h : f (-x) = log_base_half (x + 1) := h_neg_domain (-x) (by linarith)
  rw [h_odd x] at h
  linarith

end odd_function_f_l622_622542


namespace simplify_and_find_min_l622_622692

noncomputable def y (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sin (2 * x)

theorem simplify_and_find_min : 
  (∃ k : ℤ, y (-3 * Real.pi / 8 + k * Real.pi) = -1 * Real.sqrt 2 + 1) ∧ 
  (⊢ ∀ x : ℝ, y x = Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4) + 1) :=
by
  -- Proof is omitted
  sorry

end simplify_and_find_min_l622_622692


namespace factor_difference_of_squares_l622_622894

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) :=
by
  sorry

end factor_difference_of_squares_l622_622894


namespace gcd_expression_l622_622155

noncomputable def odd_multiple_of_7771 (a : ℕ) : Prop := 
  ∃ k : ℕ, k % 2 = 1 ∧ a = 7771 * k

theorem gcd_expression (a : ℕ) (h : odd_multiple_of_7771 a) : 
  Int.gcd (8 * a^2 + 57 * a + 132) (2 * a + 9) = 9 :=
  sorry

end gcd_expression_l622_622155


namespace range_of_a_l622_622993

noncomputable def g (a x : ℝ) : ℝ := a - x^2
noncomputable def h (x : ℝ) : ℝ := 2 * Real.log x

theorem range_of_a (h_cond : ∀ x, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 → g a x = -h x) :
  1 ≤ a ∧ a ≤ Real.exp 2 - 2 :=
begin
  sorry
end

end range_of_a_l622_622993


namespace factor_diff_of_squares_l622_622885

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l622_622885


namespace min_area_l622_622988

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x

def domain (a x : ℝ) := a - 1 ≤ x ∧ x ≤ a + 1

def set_M (a : ℝ) : set (ℝ × ℝ) :=
  { p | ∃ m n, domain a m ∧ domain a n ∧ p = (m, f n) }

def area (S : set (ℝ × ℝ)) : ℝ := sorry  -- assume some way to calculate area

theorem min_area (a : ℝ) : 2 ≤ area (set_M a) :=
sorry

end min_area_l622_622988


namespace find_positive_x_l622_622124

theorem find_positive_x (x : ℝ) (h1 : x > 0) (h2 : x * (Real.floor x) = 50) : x = 50 / 7 :=
by
  sorry

end find_positive_x_l622_622124


namespace razorback_shop_revenue_from_jerseys_zero_l622_622700

theorem razorback_shop_revenue_from_jerseys_zero:
  let num_tshirts := 20
  let num_jerseys := 64
  let revenue_per_tshirt := 215
  let total_revenue_tshirts := 4300
  let total_revenue := total_revenue_tshirts
  let revenue_from_jerseys := total_revenue - total_revenue_tshirts
  revenue_from_jerseys = 0 := by
  sorry

end razorback_shop_revenue_from_jerseys_zero_l622_622700


namespace fixed_circle_intersection_l622_622144

open EuclideanGeometry

variable {circles : Type*} [incirc : Intersection cirlces]
variables (P Q A B C D : Point) (ℓ : Line) (_ : LineThrough ℓ Q)
variables (tangentA : Tangent circles A) (tangentB : Tangent circles B)
variables (tangentPoint : PointOn tangents A B)

theorem fixed_circle_intersection
  (hPC: IncircleIntersection two_circles P Q)
  (hP₁A : IncircleIntersection circle₁ P A)
  (hP₁B : IncircleIntersection circle₁ P B )
  (hP₂A : IncircleIntersection circle₂ Q A)
  (hP₂B : IncircleIntersection circle₂ Q B)
  (hCA : TangentToCircle circles tangentA)
  (hCB : TangentToCircle circles tangentB)
  (hD_intersection : IntersectionPoint α D)
  (hangle_bisector : AngleBisector α C P Q)
  : FixedCircle ∀ ℓ.
  sorry

end fixed_circle_intersection_l622_622144


namespace cost_price_article_l622_622027

variable (SP : ℝ := 21000)
variable (d : ℝ := 0.10)
variable (p : ℝ := 0.08)

theorem cost_price_article : (SP * (1 - d)) / (1 + p) = 17500 := by
  sorry

end cost_price_article_l622_622027


namespace compute_expression_l622_622459

open Real

theorem compute_expression : 
  sqrt (1 / 4) * sqrt 16 - (sqrt (1 / 9))⁻¹ - sqrt 0 + sqrt (45 / 5) = 2 := 
by
  -- The proof details would go here, but they are omitted.
  sorry

end compute_expression_l622_622459


namespace total_sandwiches_l622_622448

theorem total_sandwiches :
  let billy := 49
  let katelyn := billy + 47
  let chloe := katelyn / 4
  billy + katelyn + chloe = 169 :=
by
  sorry

end total_sandwiches_l622_622448


namespace positive_number_property_l622_622421

theorem positive_number_property (x : ℝ) (h1 : 0 < x) (h2 : 0.01 * x * x + 16 = 36) : x = 20 * real.sqrt 5 :=
by
  sorry

end positive_number_property_l622_622421


namespace card_probability_ratio_l622_622358

theorem card_probability_ratio : 
  let all_cards := finset.range 30
  let number_groups := [set.range 5, set.range 5, set.range 5, set.range 5, set.range 5, set.range 5]
  let num_ways_four_cards := nat.choose 30 4
  let p_numerator := 6 * (nat.choose 5 4)
  let q_numerator := (nat.choose 6 2) * (nat.choose 5 2) * (nat.choose 5 2)
  let p := p_numerator / num_ways_four_cards
  let q := q_numerator / num_ways_four_cards
  (q / p = 50)
:= sorry

end card_probability_ratio_l622_622358


namespace length_of_platform_l622_622426

theorem length_of_platform (train_length : ℕ) (time : ℕ) (speed : ℕ) (distance : ℕ) :
  train_length = 50 → time = 10 → speed = 15 → distance = speed * time → 
    (distance - train_length) = 100 :=
by
  intros h_train_length h_time h_speed h_distance
  rw [h_train_length, h_time, h_speed] at h_distance
  norm_num at h_distance
  rw h_distance
  norm_num

end length_of_platform_l622_622426


namespace diameter_of_circle_l622_622037

def given_conditions (A B D C P : Point) (ω η : Circle) (a b : ℝ) : Prop :=
  (diameter ω A B) ∧
  tangent_at A ω D ∧
  tangent_at B ω C ∧
  intersect_at AC BD P ∧
  tangent η ω P ∧
  tangent η AD P ∧
  tangent η BC P ∧
  length AD = a ∧
  length BC = b ∧
  a ≠ b

theorem diameter_of_circle (A B D C P : Point) (ω η : Circle) (a b : ℝ) (h : given_conditions A B D C P ω η a b) :
  diameter ω = sqrt (a * b) :=
sorry

end diameter_of_circle_l622_622037


namespace probability_of_even_product_l622_622749

open BigOperators

def total_combinations : Nat := nat.choose 6 3

def odd_combinations : Nat := nat.choose 3 3

def probability_odd_product : ℚ := odd_combinations / total_combinations

def probability_even_product : ℚ := 1 - probability_odd_product

theorem probability_of_even_product (s : Finset ℕ) (h : s = {1, 2, 3, 4, 5, 6}) :
  probability_even_product = 19 / 20 := by
  sorry

end probability_of_even_product_l622_622749


namespace area_of_isosceles_triangle_l622_622702

-- Problem conditions
def isosceles_triangle_area (b s : ℝ) (height : ℝ) (perimeter : ℝ) : ℝ :=
  let base := 2 * b
  let area := (1 / 2) * base * height
  area

-- Lean proof statement
theorem area_of_isosceles_triangle
  (b s : ℝ)
  (h1 : 10 = height)  -- Altitude of the isosceles triangle
  (h2 : 40 = 2 * s + 2 * b)  -- Perimeter condition
  (h3 : b^2 + 10^2 = s^2)  -- Pythagorean relationship
  : isosceles_triangle_area b s 10 40 = 75 :=
  sorry

end area_of_isosceles_triangle_l622_622702


namespace binom_26_6_l622_622151

theorem binom_26_6 : (42624 : ℕ) = 42504 →
                      (134596 : ℕ) = 134596 →
                      ∃ (c : ℕ), (c = nat.choose 26 6) ∧ (c = 230230) :=
by {
  intros h1 h2,
  use 230230,
  split,
  sorry, -- Proof of equality using the conditions would go here.
  exact h2.symm,
}

end binom_26_6_l622_622151


namespace ratio_r_to_pq_l622_622793

theorem ratio_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 5000) (h₂ : r = 2000) :
  r / (p + q) = 2 / 3 := 
by
  sorry

end ratio_r_to_pq_l622_622793


namespace product_of_slopes_hyperbola_l622_622670

theorem product_of_slopes_hyperbola (a b x0 y0 : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : (x0, y0) ≠ (-a, 0)) (h4 : (x0, y0) ≠ (a, 0)) 
(h5 : x0^2 / a^2 - y0^2 / b^2 = 1) : 
(y0 / (x0 + a) * (y0 / (x0 - a)) = b^2 / a^2) :=
sorry

end product_of_slopes_hyperbola_l622_622670


namespace factor_diff_of_squares_l622_622887

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l622_622887


namespace parabola_vertex_eq_l622_622334

theorem parabola_vertex_eq :
  (∃ c : ℝ, (∀ x : ℝ, y = 2 * x^2 + c) ∧ y = 1 ∧ x = 0) → c = 1 :=
by
  intro h
  choose c hc using h
  specialize hc 0
  rw [mul_zero, zero_mul, add_zero] at hc
  cases hc
  rw [hc_right]
  exact hc_left

end parabola_vertex_eq_l622_622334


namespace find_initial_number_l622_622068

theorem find_initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by {
  sorry
}

end find_initial_number_l622_622068


namespace automobile_distance_2_minutes_l622_622077

theorem automobile_distance_2_minutes (a : ℝ) :
  let acceleration := a / 12
  let time_minutes := 2
  let time_seconds := time_minutes * 60
  let distance_feet := (1 / 2) * acceleration * time_seconds^2
  let distance_yards := distance_feet / 3
  distance_yards = 200 * a := 
by sorry

end automobile_distance_2_minutes_l622_622077


namespace movie_profit_l622_622050

theorem movie_profit
  (main_actor_fee : ℕ)
  (supporting_actor_fee : ℕ)
  (extra_fee : ℕ)
  (main_actor_food : ℕ)
  (supporting_actor_food : ℕ)
  (extra_food : ℕ)
  (crew_size : ℕ)
  (crew_food : ℕ)
  (post_production_cost : ℕ)
  (revenue : ℕ)
  (main_actors_count : ℕ)
  (supporting_actors_count : ℕ)
  (extras_count : ℕ)
  (food_per_main_actor : ℕ)
  (food_per_supporting_actor : ℕ)
  (food_per_remaining_crew : ℕ)
  (equipment_rental_multiplier : ℕ)
  (total_profit : ℕ) :
  main_actor_fee = 500 → 
  supporting_actor_fee = 100 →
  extra_fee = 50 →
  main_actor_food = 10 →
  supporting_actor_food = 5 →
  extra_food = 5 →
  crew_size = 50 →
  crew_food = 3 →
  post_production_cost = 850 →
  revenue = 10000 →
  main_actors_count = 2 →
  supporting_actors_count = 3 →
  extras_count = 1 →
  equipment_rental_multiplier = 2 →
  total_profit = revenue - ((main_actors_count * main_actor_fee) +
                           (supporting_actors_count * supporting_actor_fee) +
                           (extras_count * extra_fee) +
                           (main_actors_count * main_actor_food) +
                           ((supporting_actors_count + extras_count) * supporting_actor_food) +
                           ((crew_size - main_actors_count - supporting_actors_count - extras_count) * crew_food) +
                           (equipment_rental_multiplier * 
                             ((main_actors_count * main_actor_fee) +
                              (supporting_actors_count * supporting_actor_fee) +
                              (extras_count * extra_fee) +
                              (main_actors_count * main_actor_food) +
                              ((supporting_actors_count + extras_count) * supporting_actor_food) +
                              ((crew_size - main_actors_count - supporting_actors_count - extras_count) * crew_food))) +
                           post_production_cost) →
  total_profit = 4584 :=
begin
  -- proof
  sorry
end

end movie_profit_l622_622050


namespace lily_pad_cover_entire_lake_l622_622214

-- Definitions per the conditions
def doublesInSizeEveryDay (P : ℕ → ℝ) : Prop :=
  ∀ n, P (n + 1) = 2 * P n

-- The initial state that it takes 36 days to cover the lake
def coversEntireLakeIn36Days (P : ℕ → ℝ) (L : ℝ) : Prop :=
  P 36 = L

-- The main theorem to prove
theorem lily_pad_cover_entire_lake (P : ℕ → ℝ) (L : ℝ) (h1 : doublesInSizeEveryDay P) (h2 : coversEntireLakeIn36Days P L) :
  ∃ n, n = 36 := 
by
  sorry

end lily_pad_cover_entire_lake_l622_622214


namespace unique_solution_probability_l622_622284

noncomputable def roll_die_twice (a b : ℕ) : Prop :=
  a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6}

noncomputable def system_has_unique_solution (a b : ℕ) : Prop :=
  (a * 2 ∧ b * 2) ≠ (4 * a ∧ 4 * b)

theorem unique_solution_probability : 
  ( ∑ a in {1, 2, 3, 4, 5, 6}, ∑ b in {1, 2, 3, 4, 5, 6}, 
    (if system_has_unique_solution a b then 1 else 0) / 36 = 11 / 12) :=
sorry

end unique_solution_probability_l622_622284


namespace find_c_l622_622332

-- Defining the given condition
def parabola (x : ℝ) (c : ℝ) : ℝ := 2 * x^2 + c

theorem find_c : (∃ c : ℝ, ∀ x : ℝ, parabola x c = 2 * x^2 + 1) :=
by 
  sorry

end find_c_l622_622332


namespace tiling_ways_2x7_l622_622837

def T : ℕ → ℕ
| 0 := 1  -- Here, a T_0 makes sense as we need base cases for the recurrence relation.
| 1 := 2
| 2 := 7
| n := T (n - 1) + S n + S (n - 1) + T (n - 2)

with S : ℕ → ℕ
| 0 := 1  -- Again, an S_0 base case
| 1 := 1  -- Assuming initial condition similar to T
| n := T (n - 1) + S (n - 1)

theorem tiling_ways_2x7 : T 7 = 2356 :=
by
  sorry

end tiling_ways_2x7_l622_622837


namespace circle_equation_tangent_lines_chord_length_l622_622941

-- Definitions from conditions
def circle_center_condition (C : ℝ × ℝ) : Prop :=
  C.1 - C.2 + 1 = 0 ∧ ∀ (A B : ℝ × ℝ), A = (1, -1) → B = (4, 2) → dist C A = dist C B

def point_on_tangent_line (M : ℝ × ℝ) (line : ℝ → ℝ) : Prop :=
  M.2 - line M.1 = 0

def chord_length_condition (center : ℝ × ℝ) (radius : ℝ) (length : ℝ) : Prop :=
  let d := abs center.2
  in 2*radius*radius = d*d + (length/2)*(length/2)

-- Proof statements
theorem circle_equation (C : ℝ × ℝ) (r : ℝ) :
  circle_center_condition C →
  r = 3 →
  C = (1, 2) →
  ∃ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = r^2 :=
by
  intro h_center hr hC
  sorry

theorem tangent_lines (C M : ℝ × ℝ) :
  C = (1, 2) →
  M = (-2, 1) →
  ∃ k : ℝ, k * (M.1 + 2) - (M.2 - 1) = 0 ∧
    ((k = 0 ∧ M.1 = -2) ∨
     (k = -4/3 ∧ ((4 * x + 3 * y + 5 = 0) → k * x - y + 1 + 2*k = 0))) :=
by
  intro hC hM
  sorry

theorem chord_length (C : ℝ × ℝ) (r length : ℝ) :
  C = (1, 2) →
  r = 3 →
  chord_length_condition C r length →
  length = 2 * sqrt 5 :=
by
  intro hC hr h_chord
  sorry

end circle_equation_tangent_lines_chord_length_l622_622941


namespace range_of_s_l622_622507

-- Define the function s(n) as described
noncomputable def s (n : ℕ) : ℕ :=
  if h : n > 1 then
    let factors := n.factorization.to_multiset.to_finset.to_list in
    factors.foldr (λ p acc, acc + (p^2) * (n.factorization p))
                 0
  else 0

-- The theorem to be proved
theorem range_of_s : 
  ∀ m : ℕ, m ≥ 12 →
  ∃ n : ℕ, (n > 1 ∧ ¬n.prime ∧ s n = m) :=
by
  sorry

end range_of_s_l622_622507


namespace trigonometric_inequality_l622_622800

theorem trigonometric_inequality (x : ℝ) (n : ℤ) :
  x ∈ (Ioo (-π/8 + π * n) (π * n) ∪ Ioo (π/8 + π * n) (3 * π/8 + π * n) ∪ Ioo π/2 + π * n (5 * π/8 + π * n)) → 
  9.2894 * sin x * sin (2 * x) * sin (3 * x) > sin (4 * x) :=
begin
  sorry
end

end trigonometric_inequality_l622_622800


namespace x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l622_622036

theorem x_squared_y_squared_iff (x y : ℝ) : x ^ 2 = y ^ 2 ↔ x = y ∨ x = -y := by
  sorry

theorem x_squared_y_squared_not_sufficient (x y : ℝ) : (x ^ 2 = y ^ 2) → (x = y ∨ x = -y) := by
  sorry

theorem x_squared_y_squared_necessary (x y : ℝ) : (x = y) → (x ^ 2 = y ^ 2) := by
  sorry

end x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l622_622036


namespace simplify_expression_l622_622292

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l622_622292


namespace problem_induction_l622_622157

open Classical
open Nat

theorem problem_induction (k : ℕ) (h_even : Even k) (h_ge_2 : k ≥ 2)
  (ind_hyp : 1 - (∑ i in range k, if (i + 1) % 2 = 0 then -(1 / (i + 1 : ℚ)) else (1 / (i + 1 : ℚ))) = 2 * (∑ j in range ((2 * k - 2) // 2), 1 / (k + 2 * (j + 1) : ℚ))) :
  1 - (∑ i in range (k + 2), if (i + 1) % 2 = 0 then -(1 / (i + 1 : ℚ)) else (1 / (i + 1 : ℚ))) = 2 * (∑ j in range ((2 * (k + 2) - 2) // 2), 1 / ((k + 2) + 2 * (j + 1) : ℚ)) :=
by
  sorry

end problem_induction_l622_622157


namespace tables_needed_l622_622401

theorem tables_needed (children : ℕ) (children_per_table : ℕ) (tables : ℕ) : 
  children = 152 → children_per_table = 7 → tables = 22 → (children / children_per_table + if (children % children_per_table = 0) then 0 else 1 = tables) :=
by
  intros
  sorry

end tables_needed_l622_622401


namespace probability_even_product_l622_622751

open BigOperators

def setS : Finset ℕ := {1, 2, 3, 4, 5, 6}

def choose3 : ℕ := (Finset.card (Finset.powersetLen 3 setS))

def oddS : Finset ℕ := {1, 3, 5}

def chooseOdd3 : ℕ := (Finset.card (Finset.powersetLen 3 oddS))

theorem probability_even_product :
  (1 : ℚ) - (chooseOdd3.to_rat / choose3.to_rat) = 19 / 20 := by
  sorry

end probability_even_product_l622_622751


namespace number_of_maple_trees_planted_today_l622_622355

-- Define the initial conditions
def initial_maple_trees : ℕ := 2
def poplar_trees : ℕ := 5
def final_maple_trees : ℕ := 11

-- State the main proposition
theorem number_of_maple_trees_planted_today : 
  (final_maple_trees - initial_maple_trees) = 9 := by
  sorry

end number_of_maple_trees_planted_today_l622_622355


namespace quadrilateral_correct_choice_l622_622018

/-- Define the triangle inequality theorem for four line segments.
    A quadrilateral can be formed if for any:
    - The sum of the lengths of any three segments is greater than the length of the fourth segment.
-/
def is_quadrilateral (a b c d : ℕ) : Prop :=
  (a + b + c > d) ∧ (a + b + d > c) ∧ (a + c + d > b) ∧ (b + c + d > a)

/-- Determine which set of three line segments can form a quadrilateral with a fourth line segment of length 5.
    We prove that the correct choice is the set (3, 3, 3). --/
theorem quadrilateral_correct_choice :
  is_quadrilateral 3 3 3 5 ∧  ¬ is_quadrilateral 1 1 1 5 ∧  ¬ is_quadrilateral 1 1 8 5 ∧  ¬ is_quadrilateral 1 2 2 5 :=
by
  sorry

end quadrilateral_correct_choice_l622_622018


namespace units_digit_sum_factorials_l622_622878

theorem units_digit_sum_factorials : 
  let T := (Finset.range 16).sum (λ n, Nat.factorial n)
  Nat.unitsDigit T = 3 := 
by
  sorry

end units_digit_sum_factorials_l622_622878


namespace intersection_domain_range_l622_622994

-- Define domain and function
def domain : Set ℝ := {-1, 0, 1}
def f (x : ℝ) : ℝ := |x|

-- Prove the theorem
theorem intersection_domain_range :
  let range : Set ℝ := {y | ∃ x ∈ domain, f x = y}
  let A : Set ℝ := domain
  let B : Set ℝ := range 
  A ∩ B = {0, 1} :=
by
  -- The proof is skipped with sorry
  sorry

end intersection_domain_range_l622_622994


namespace smallest_value_sum_product_l622_622881

theorem smallest_value_sum_product (b : Fin 100 → ℤ) (h : ∀ i, b i = 1 ∨ b i = -1) : 
  (∃ T, T = 22 ∧ T = ∑ i in Finset.range (100 - 1), ∑ j in Finset.range (100 - i - 1), b i * b (i + j + 1)) :=
sorry

end smallest_value_sum_product_l622_622881


namespace min_quadrilateral_area_l622_622950

noncomputable def ellipse_eq (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def parabola_eq (p x y : ℝ) : Prop :=
  y^2 = 2 * p * x

theorem min_quadrilateral_area :
  (∃ (a b : ℝ), a > b > 0 ∧ (2 * sqrt 3 = b) ∧ (e : ℝ) / (a) = 1 / 2) →
  (∃ (p : ℝ), parabola_eq p x y ∧ y^2 = 8 * x) →
  (∃ S_min : ℝ, S_min = 96) :=
begin
  intros h1 h2,
  use 96,
  sorry
end

end min_quadrilateral_area_l622_622950


namespace initial_number_l622_622066

theorem initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by
  sorry

end initial_number_l622_622066


namespace remainder_when_divided_by_6_l622_622772

theorem remainder_when_divided_by_6 (n : ℕ) (h₁ : n = 482157)
  (odd_n : n % 2 ≠ 0) (div_by_3 : n % 3 = 0) : n % 6 = 3 :=
by
  -- Proof goes here
  sorry

end remainder_when_divided_by_6_l622_622772


namespace simplify_expression_l622_622309

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l622_622309


namespace lemon_loaf_each_piece_weight_l622_622412

def pan_length := 20  -- cm
def pan_width := 18   -- cm
def pan_height := 5   -- cm
def total_pieces := 25
def density := 2      -- g/cm³

noncomputable def weight_of_each_piece : ℕ := by
  have volume := pan_length * pan_width * pan_height
  have volume_of_each_piece := volume / total_pieces
  have mass_of_each_piece := volume_of_each_piece * density
  exact mass_of_each_piece

theorem lemon_loaf_each_piece_weight :
  weight_of_each_piece = 144 :=
sorry

end lemon_loaf_each_piece_weight_l622_622412


namespace range_of_a_l622_622167

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a ^ x

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) ↔ (3 / 2 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l622_622167


namespace x_pow16_operations_x_pow_n_operations_l622_622797

-- Part (a)
theorem x_pow16_operations (x : ℝ) (hx : x ≠ 0) : 
  ∃ (ops : ℕ), ops ≤ 12 ∧ (x ^ 16 = someCalculatingResult x 12 ops) := 
sorry

-- Part (b)
theorem x_pow_n_operations (x : ℝ) (n : ℕ) (hx : x ≠ 0) : 
  ∃ (ops : ℕ), ops ≤ 1 + (3 / 2) * log 2 (n : ℝ) ∧ (x ^ n = someCalculatingResult x n ops) := 
sorry

end x_pow16_operations_x_pow_n_operations_l622_622797


namespace angles_less_than_20_l622_622673

theorem angles_less_than_20 (p : Point) (lines : list (Line p)) (h₁ : lines.length = 10)
    (h₂ : ∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → ∃! q, q ∈ l₁ ∩ l₂)
    (h₃ : ∃ a, (∀ l ∈ lines, ∃ q ∈ l, q = p) ∧ (∑ θ in (all_angles p lines), θ = 360)) :
  ∃ θ ∈ (all_angles p lines), θ < 20 := 
sorry

end angles_less_than_20_l622_622673


namespace arrangement_count_correct_l622_622399

def number_of_arrangements : ℕ :=
  let students := fin 6 in
  let communities := fin 3 in
  let A1_to_A := {arr | arr student A1 = community A} in
  let B_and_C_not_to_C := {arr | arr student A2 ≠ community C ∧ arr student A3 ≠ community C} in
  (arrangements A1_to_A ∩ B_and_C_not_to_C).card

theorem arrangement_count_correct : number_of_arrangements = 5 := 
sorry

end arrangement_count_correct_l622_622399


namespace nine_cards_orderable_in_12_moves_l622_622088

def f (n : ℕ) : ℕ := sorry

theorem nine_cards_orderable_in_12_moves :
  ∀ (pi : List ℕ), (∀ (card ∈ pi), card ∈ Finset.range 9) → length pi = 9 →
  (∃ (k : ℕ), ∃ (moves : ℕ), moves ≤ 12 ∧ (sort pi = List.range (card_sort)) ∨ (sort pi = List.range_reverse(card_sort))) :=
sorry

end nine_cards_orderable_in_12_moves_l622_622088


namespace find_y_l622_622030

theorem find_y (x y : ℤ) (q : ℤ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x = q * y + 6) (h4 : (x : ℚ) / y = 96.15) : y = 40 :=
sorry

end find_y_l622_622030


namespace stickers_distribution_l622_622577

theorem stickers_distribution : 
  let stickers := 12
  let sheets := 5
  ∃ ways : ℕ,
    ways = Nat.choose ((stickers - sheets) + sheets - 1) (sheets - 1) ∧ ways = 330 :=
by
  let stickers := 12
  let sheets := 5
  have ways := Nat.choose ((stickers - sheets) + sheets - 1) (sheets - 1)
  have h : ways = 330 := sorry
  use ways
  exact ⟨rfl, h⟩

end stickers_distribution_l622_622577


namespace common_roots_of_f1_and_f2_l622_622711

noncomputable def f1 : ℝ → ℝ := λ x, x^4 + 2*x^3 - x^2 - 2*x - 3
noncomputable def f2 : ℝ → ℝ := λ x, x^4 + 3*x^3 + x^2 - 4*x - 6

theorem common_roots_of_f1_and_f2 :
  ∃ x, (f1 x = 0 ∧ f2 x = 0) ∧
        (x = -1 ∨ x = (-1 + Real.sqrt 13) / 2 ∨ x = (-1 - Real.sqrt 13) / 2) :=
sorry

end common_roots_of_f1_and_f2_l622_622711


namespace find_m_l622_622132

open Real

variables (OA OB AB : EuclideanSpace ℝ (Fin 2))

def perp (u v : EuclideanSpace ℝ (Fin 2)) : Prop := inner u v = 0

def OA : EuclideanSpace ℝ (Fin 2) := ![(-1 : ℝ), (2 : ℝ)]
def OB (m : ℝ) : EuclideanSpace ℝ (Fin 2) := ![(3 : ℝ), m]
def AB (m : ℝ) : EuclideanSpace ℝ (Fin 2) := OB m - OA

theorem find_m (m : ℝ) (h : perp OA (AB m)) : m = 4 := by
  sorry

end find_m_l622_622132


namespace yz_plane_equidistant_point_l622_622123

theorem yz_plane_equidistant_point :
  ∃ y z : ℝ, 
    (0, y, z) ∈ yz_plane ∧
    dist (0, y, z) (1, 0, 2) = dist (0, y, z) (0, 2, 1) ∧ 
    dist (0, y, z) (1, 0, 2) = dist (0, y, z) (2, 3, 1) ∧ 
    (y = 7/2 ∧ z = -3) := 
  sorry

end yz_plane_equidistant_point_l622_622123


namespace symmetric_graphs_implies_range_of_a_l622_622996

theorem symmetric_graphs_implies_range_of_a {a : ℝ} :
  (∃ (x : ℝ), x < 0 ∧ f x = g (-x)) → a < real.exp 1 :=
by
  assume h : ∃ (x : ℝ), x < 0 ∧ (λ x, real.exp x + 2) x = (λ x, real.log (x + a) + 2) (-x)
  sorry

end symmetric_graphs_implies_range_of_a_l622_622996


namespace problem_condition_l622_622264

noncomputable def f : ℝ → ℝ := sorry

theorem problem_condition (h: ∀ x : ℝ, f x > (deriv f) x) : 3 * f (Real.log 2) > 2 * f (Real.log 3) :=
sorry

end problem_condition_l622_622264


namespace accurate_bottle_weight_l622_622434

-- Define the options as constants
def OptionA : ℕ := 500 -- milligrams
def OptionB : ℕ := 500 * 1000 -- grams
def OptionC : ℕ := 500 * 1000 * 1000 -- kilograms
def OptionD : ℕ := 500 * 1000 * 1000 * 1000 -- tons

-- Define a threshold range for the weight of a standard bottle of mineral water in grams
def typicalBottleWeightMin : ℕ := 400 -- for example
def typicalBottleWeightMax : ℕ := 600 -- for example

-- Translate the question and conditions into a proof statement
theorem accurate_bottle_weight : OptionB = 500 * 1000 :=
by
  -- Normally, we would add the necessary steps here to prove the statement
  sorry

end accurate_bottle_weight_l622_622434


namespace isosceles_triangle_circles_l622_622798

-- Define the given values for the isosceles triangle
def base (Δ : Type) := 48
def side (Δ : Type) := 30

-- Define the correct answers to be proved
def r := 8
def R := 25
def d := 15

-- The main theorem to prove the radius of the inscribed circle, the circumscribed circle, and the distance between their centers
theorem isosceles_triangle_circles (h : ℝ) (A : ℝ) (s : ℝ)
  (sin_C : ℝ) (inradius : ℝ) (circumradius : ℝ) (center_dist : ℝ) :
  base ℝ = 48 →
  side ℝ = 30 →
  h = 18 →
  A = 432 →
  s = 54 →
  sin_C = 3/5 →
  inradius = r →
  circumradius = R →
  center_dist = d :=
by {
  intros h_base h_side h_height h_area h_semiperim h_sinC h_inr h_circumr h_centerdist,
  sorry
}

end isosceles_triangle_circles_l622_622798


namespace surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l622_622660

-- Given conditions
def income_per_day : List Int := [65, 68, 50, 66, 50, 75, 74]
def expenditure_per_day : List Int := [-60, -64, -63, -58, -60, -64, -65]

-- Part 1: Proving the surplus by the end of the week is 14 yuan
theorem surplus_by_end_of_week_is_14 :
  List.sum income_per_day + List.sum expenditure_per_day = 14 :=
by
  sorry

-- Part 2: Proving the estimated income needed per month to maintain normal expenses is 1860 yuan
theorem estimated_monthly_income_is_1860 :
  (List.sum (List.map Int.natAbs expenditure_per_day) / 7) * 30 = 1860 :=
by
  sorry

end surplus_by_end_of_week_is_14_estimated_monthly_income_is_1860_l622_622660


namespace Nellie_needs_to_sell_more_rolls_l622_622928

-- Define the conditions
def total_needed : ℕ := 45
def sold_grandmother : ℕ := 1
def sold_uncle : ℕ := 10
def sold_neighbor : ℕ := 6

-- Define the total sold
def total_sold : ℕ := sold_grandmother + sold_uncle + sold_neighbor

-- Define the remaining rolls needed
def remaining_rolls := total_needed - total_sold

-- Statement to prove that remaining_rolls equals 28
theorem Nellie_needs_to_sell_more_rolls : remaining_rolls = 28 := by
  unfold remaining_rolls
  unfold total_sold
  unfold total_needed sold_grandmother sold_uncle sold_neighbor
  calc
  45 - (1 + 10 + 6) = 45 - 17 : by rw [Nat.add_assoc]
  ... = 28 : by norm_num

end Nellie_needs_to_sell_more_rolls_l622_622928


namespace centroid_locus_property_l622_622627

theorem centroid_locus_property 
  (ABC : Triangle)
  (area_ABC : ℝ) -- Assume area of triangle ABC is given and equals area_ABC
  (M_is_centroid : is_centroid M ABC)
  : (∀ (N : Point), N ∈ boundary_of ABC →
      ∃ (P : Point), P ∈ inside_or_boundary_of ABC ∧ 
      triangle_area M N P ≥ (1 / 6) * area_ABC) :=
begin
  sorry
end

end centroid_locus_property_l622_622627


namespace sum_of_divisors_57_l622_622785

theorem sum_of_divisors_57 : 
  ∑ d in (finset.filter (λ x, 57 % x = 0) (finset.range (58))), d = 80 := 
by
  sorry

end sum_of_divisors_57_l622_622785


namespace triplet_solution_l622_622504

theorem triplet_solution (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) :
  (a + b + c = (1 / a) + (1 / b) + (1 / c) ∧ a ^ 2 + b ^ 2 + c ^ 2 = (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2))
  ↔ (∃ x, (a = 1 ∨ a = -1 ∨ a = x ∨ a = 1/x) ∧
           (b = 1 ∨ b = -1 ∨ b = x ∨ b = 1/x) ∧
           (c = 1 ∨ c = -1 ∨ c = x ∨ c = 1/x)) := 
sorry

end triplet_solution_l622_622504


namespace altitude_inequality_l622_622128

open_locale classical

-- Definition of m(PQR) as the minimum altitude length function
def m (P Q R : ℝ × ℝ) : ℝ :=
if collinear P Q R
then 0
else min (altitude_length P Q R) (min (altitude_length Q P R) (altitude_length R P Q))

-- Conditions for collinearity and altitude length
-- altitude_length is assumed as a helper function that calculates the altitude from a given point to the opposite side
def collinear (P Q R : ℝ × ℝ) : Prop :=
(P.1 - Q.1) * (R.2 - Q.2) = (R.1 - Q.1) * (P.2 - Q.2)

def altitude_length (P Q R : ℝ × ℝ) : ℝ := 
2 * area P Q R / distance Q R -- This assumes area and distance are predefined functions

theorem altitude_inequality (A B C X : ℝ × ℝ) : 
  m A B C ≤ m A B X + m A X C + m X B C := 
sorry

end altitude_inequality_l622_622128


namespace unique_real_root_t_l622_622618

theorem unique_real_root_t (t : ℝ) :
  (∃ x : ℝ, 3 * x + 7 * t - 2 + (2 * t * x^2 + 7 * t^2 - 9) / (x - t) = 0 ∧ 
  ∀ y : ℝ, 3 * y + 7 * t - 2 + (2 * t * y^2 + 7 * t^2 - 9) / (y - t) = 0 ∧ x ≠ y → false) →
  t = -3 ∨ t = -7 / 2 ∨ t = 1 :=
by
  sorry

end unique_real_root_t_l622_622618


namespace complex_number_in_first_quadrant_l622_622617

theorem complex_number_in_first_quadrant :
  let z := 1 / (((1 : ℂ) + complex.I)^2 + 1) + complex.I in
  complex.re z = 1 / 5 ∧ complex.im z = 3 / 5 ∧ 0 < complex.re z ∧ 0 < complex.im z :=
by
  sorry

end complex_number_in_first_quadrant_l622_622617


namespace not_both_hit_bullseye_probability_l622_622439

-- Definitions based on the conditions
def prob_A_bullseye : ℚ := 1 / 3
def prob_B_bullseye : ℚ := 1 / 2
def prob_both_hit_bullseye : ℚ := prob_A_bullseye * prob_B_bullseye

-- Statement of the proof problem
theorem not_both_hit_bullseye_probability : 1 - prob_both_hit_bullseye = 5 / 6 := by
  sorry

end not_both_hit_bullseye_probability_l622_622439


namespace product_of_elements_with_order_d_l622_622641

theorem product_of_elements_with_order_d (p : ℕ) [Fact (Nat.Prime p)] (d : ℕ) (h_d : d ∣ p - 1) :
  let S := { x : ℤ // x ≠ 0 ∧ ∃ k, k ∈ Finset.range (p - 1) ∧ (x : ℤ)^k ≡ 1 [ZMOD p] ∧ Nat.find (natPrime_order_mod (of_int p) x) = d } in
  if d = 2 then 
    ∏ x in S, x ≡ -1 [ZMOD p]
  else 
    ∏ x in S, x ≡ 1 [ZMOD p] :=
sorry

end product_of_elements_with_order_d_l622_622641


namespace probability_equal_each_attempt_l622_622614

noncomputable def simple_random_sampling (n : ℕ) (individual : ℕ) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → (prob_individual_selected : ℝ) = (1 / n)

theorem probability_equal_each_attempt (n : ℕ) (individual : ℕ) :
  simple_random_sampling n individual :=
by
  sorry

end probability_equal_each_attempt_l622_622614


namespace recurring_decimal_division_l622_622378

noncomputable def recurring_decimal_fraction : ℚ :=
  let frac_81 := (81 : ℚ) / 99
  let frac_36 := (36 : ℚ) / 99
  frac_81 / frac_36

theorem recurring_decimal_division :
  recurring_decimal_fraction = 9 / 4 :=
by
  sorry

end recurring_decimal_division_l622_622378


namespace distance_between_foci_l622_622113

theorem distance_between_foci (x y : ℝ) :
  let a^2 := 9,
      b^2 := 4,
      c^2 := a^2 - b^2
  in 2 * Real.sqrt c^2 = 2 * Real.sqrt 5 :=
by
  sorry

end distance_between_foci_l622_622113


namespace lateral_edge_of_regular_triangular_prism_l622_622329

theorem lateral_edge_of_regular_triangular_prism :
  ∀ (a : ℝ), (a = 1) → ∃ (ℓ : ℝ), (∃ (r : ℝ), r = (a * real.sqrt 3) / 6 ∧ ℓ = 2 * r) ∧ ℓ = real.sqrt 3 / 3 :=
by
  intro a h₁
  use real.sqrt 3 / 3
  use (a * real.sqrt 3) / 6
  split
  { split
    { exact rfl
    }
    { calc
        ℓ = 2 * ((a * real.sqrt 3) / 6) : by rw [← h₁]
         ... = real.sqrt 3 / 3   : by ring
    }
  }
  { exact rfl
  }

end lateral_edge_of_regular_triangular_prism_l622_622329


namespace count_factorable_polynomials_l622_622118

theorem count_factorable_polynomials : 
  (∃ (a : ℤ), 1 ≤ a * (a + 1) ∧ a * (a + 1) ≤ 2000) ↔ (finset.card (finset.filter (λ n : ℤ, ∃ a : ℤ, n = a * (a + 1) ∧ 1 ≤ n ∧ n ≤ 2000) (finset.range 2000)) = 89) :=
by
  sorry

end count_factorable_polynomials_l622_622118


namespace prob_twoDigitDivBy3_l622_622419

-- Define the set and the condition for a number being two-digit and divisible by 3
def isTwoDigit (n : ℕ) : Prop := n >= 60 ∧ n <= 99
def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0
def set := {n : ℕ | n >= 60 ∧ n <= 1000}
def twoDigitDivBy3Set := {n ∈ set | isTwoDigit n ∧ isDivisibleBy3 n}

noncomputable def countTotal : ℕ :=
  (set.toFinset.card)

noncomputable def countTwoDigitDivBy3 : ℕ :=
  (twoDigitDivBy3Set.toFinset.card)

theorem prob_twoDigitDivBy3 : 
  countTwoDigitDivBy3.toNat / countTotal.toNat = 14 / 941 :=
by
  sorry

end prob_twoDigitDivBy3_l622_622419


namespace factor_t_sq_minus_64_l622_622891

def isDifferenceOfSquares (a b : Int) : Prop := a = b^2

theorem factor_t_sq_minus_64 (t : Int) (h : isDifferenceOfSquares 64 8) : (t^2 - 64) = (t - 8) * (t + 8) := by
  sorry

end factor_t_sq_minus_64_l622_622891


namespace jamie_dimes_l622_622632

theorem jamie_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 240) : d = 10 :=
sorry

end jamie_dimes_l622_622632


namespace b_general_formula_H_2017_value_l622_622944

noncomputable def S (n : ℕ) : ℚ := (3^n - 1) / 2

def a (n : ℕ) : ℚ :=
  if n = 1 then 1 else S n - S (n - 1)

def b (n : ℕ) : ℚ := Real.logb 9 (a (n + 1))

noncomputable def T (n : ℕ) : ℚ := (1 / 2) * (n * (n + 1) / 2)

noncomputable def H (n : ℕ) : ℚ := 4 * (1 - (1 / (n + 1)))

theorem b_general_formula (n : ℕ) : b n = n / 2 := 
  sorry

theorem H_2017_value : H 2017 = 4034 / 1009 :=
  sorry

end b_general_formula_H_2017_value_l622_622944


namespace final_values_l622_622616

-- Declare the variables
variables (A B C : ℕ)

-- The assignment conditions are given as hypotheses
def assignment_conditions :=
C = 2 ∧
B = 1 ∧
A = 2

theorem final_values (h : assignment_conditions) : A = 2 ∧ B = 1 ∧ C = 2 :=
by sorry

end final_values_l622_622616


namespace pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l622_622528

-- Definitions based on the problem's conditions
def a (n : Nat) : Nat := n * n

def pos_count (n : Nat) : Nat :=
  List.length (List.filter (λ m : Nat => a m < n) (List.range (n + 1)))

def pos_pos_count (n : Nat) : Nat :=
  pos_count (pos_count n)

-- Theorem statements
theorem pos_count_a5_eq_2 : pos_count 5 = 2 := 
by
  -- Proof would go here
  sorry

theorem pos_pos_count_an_eq_n2 (n : Nat) : pos_pos_count n = n * n :=
by
  -- Proof would go here
  sorry

end pos_count_a5_eq_2_pos_pos_count_an_eq_n2_l622_622528


namespace simplify_fraction_l622_622312

theorem simplify_fraction : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 :=
by sorry

end simplify_fraction_l622_622312


namespace sum_of_digits_of_10_pow_100_minus_158_l622_622882

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  (n.toString.foldl (λ acc c, acc + c.toNat - '0'.toNat) 0)

theorem sum_of_digits_of_10_pow_100_minus_158 : 
  sum_of_digits (10^100 - 158) = 887 := 
sorry

end sum_of_digits_of_10_pow_100_minus_158_l622_622882


namespace cost_of_300_candies_l622_622193

theorem cost_of_300_candies (cost_per_candy_cents : ℕ) (total_candies : ℕ) :
  cost_per_candy_cents = 5 → total_candies = 300 → (total_candies * cost_per_candy_cents) / 100 = 15 :=
by
  intros h_cost h_total
  rw [h_cost, h_total]
  norm_num
  sorry  -- This ensures the theorem compiles successfully, skipping the detailed arithmetic proof steps.

end cost_of_300_candies_l622_622193


namespace probability_even_product_l622_622750

open BigOperators

def setS : Finset ℕ := {1, 2, 3, 4, 5, 6}

def choose3 : ℕ := (Finset.card (Finset.powersetLen 3 setS))

def oddS : Finset ℕ := {1, 3, 5}

def chooseOdd3 : ℕ := (Finset.card (Finset.powersetLen 3 oddS))

theorem probability_even_product :
  (1 : ℚ) - (chooseOdd3.to_rat / choose3.to_rat) = 19 / 20 := by
  sorry

end probability_even_product_l622_622750


namespace exists_zero_remainder_remainder_le_15_exists_n_with_remainder_l622_622032

-- Define the properties and structure of a two-digit number
def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define the sum of the digits of a two-digit number
def sum_of_digits (n : ℕ) : ℕ :=
  let a := n / 10 in
  let b := n % 10 in
  a + b

-- Theorem 1: Prove that there exists a two-digit number such that the remainder is 0
theorem exists_zero_remainder :
  ∃ n, is_two_digit_number n ∧ n % sum_of_digits n = 0 :=
by sorry

-- Theorem 2: Prove that the remainder cannot be greater than 15
theorem remainder_le_15 (n : ℕ) (h : is_two_digit_number n) :
  n % sum_of_digits n ≤ 15 :=
by sorry

-- Theorem 3: Prove that for any remainder <= 12, there exists a two-digit number with that remainder
theorem exists_n_with_remainder (r : ℕ) (h_r : r ≤ 12) :
  ∃ n, is_two_digit_number n ∧ n % sum_of_digits n = r :=
by sorry

end exists_zero_remainder_remainder_le_15_exists_n_with_remainder_l622_622032


namespace find_f_of_neg_2_l622_622322

theorem find_f_of_neg_2
  (f : ℚ → ℚ)
  (h : ∀ (x : ℚ), x ≠ 0 → 3 * f (1/x) + 2 * f x / x = x^2)
  : f (-2) = 13/5 :=
sorry

end find_f_of_neg_2_l622_622322


namespace numbers_not_as_difference_of_squares_l622_622904

theorem numbers_not_as_difference_of_squares :
  {n : ℕ | ¬ ∃ x y : ℕ, x^2 - y^2 = n} = {1, 4} ∪ {4*k + 2 | k : ℕ} :=
by sorry

end numbers_not_as_difference_of_squares_l622_622904
