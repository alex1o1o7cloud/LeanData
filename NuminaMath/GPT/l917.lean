import Mathlib

namespace keira_guarantees_capture_l917_91735

theorem keira_guarantees_capture (k : ℕ) (n : ℕ) (h_k_pos : 0 < k) (h_n_cond : n > k / 2023) :
    k ≥ 1012 :=
sorry

end keira_guarantees_capture_l917_91735


namespace number_division_l917_91777

theorem number_division (n q r d : ℕ) (h1 : d = 18) (h2 : q = 11) (h3 : r = 1) (h4 : n = (d * q) + r) : n = 199 := 
by 
  sorry

end number_division_l917_91777


namespace three_distinct_real_roots_l917_91742

theorem three_distinct_real_roots 
  (c : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (x1*x1 + 6*x1 + c)*(x1*x1 + 6*x1 + c) = 0 ∧ 
    (x2*x2 + 6*x2 + c)*(x2*x2 + 6*x2 + c) = 0 ∧ 
    (x3*x3 + 6*x3 + c)*(x3*x3 + 6*x3 + c) = 0) 
  ↔ c = (11 - Real.sqrt 13) / 2 :=
by sorry

end three_distinct_real_roots_l917_91742


namespace fourth_number_value_l917_91787

variable (A B C D E F : ℝ)

theorem fourth_number_value 
  (h1 : A + B + C + D + E + F = 180)
  (h2 : A + B + C + D = 100)
  (h3 : D + E + F = 105) : 
  D = 25 := 
by 
  sorry

end fourth_number_value_l917_91787


namespace total_amount_paid_l917_91739

-- Definitions of the conditions
def cost_earbuds : ℝ := 200
def tax_rate : ℝ := 0.15

-- Statement to prove
theorem total_amount_paid : (cost_earbuds + (cost_earbuds * tax_rate)) = 230 := sorry

end total_amount_paid_l917_91739


namespace chocolate_ticket_fraction_l917_91759

theorem chocolate_ticket_fraction (box_cost : ℝ) (ticket_count_per_free_box : ℕ) (ticket_count_included : ℕ) :
  ticket_count_per_free_box = 10 →
  ticket_count_included = 1 →
  (1 / 9 : ℝ) * box_cost =
  box_cost / ticket_count_per_free_box + box_cost / (ticket_count_per_free_box - ticket_count_included + 1) :=
by 
  intros h1 h2 
  have h : ticket_count_per_free_box = 10 := h1 
  have h' : ticket_count_included = 1 := h2 
  sorry

end chocolate_ticket_fraction_l917_91759


namespace evaluate_expression_l917_91785

theorem evaluate_expression :
  (2 ^ (-1 : ℤ) + 2 ^ (-2 : ℤ))⁻¹ = (4 / 3 : ℚ) := by
    sorry

end evaluate_expression_l917_91785


namespace ab_bc_ca_lt_quarter_l917_91760

theorem ab_bc_ca_lt_quarter (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 1) :
  (a * b)^(5/4) + (b * c)^(5/4) + (c * a)^(5/4) < 1/4 :=
sorry

end ab_bc_ca_lt_quarter_l917_91760


namespace initial_cats_l917_91771

-- Define the conditions as hypotheses
variables (total_cats now : ℕ) (cats_given : ℕ)

-- State the main theorem
theorem initial_cats:
  total_cats = 31 → cats_given = 14 → (total_cats - cats_given) = 17 :=
by sorry

end initial_cats_l917_91771


namespace find_t_eq_l917_91794

variable (a V V_0 S t : ℝ)

theorem find_t_eq (h1 : V = a * t + V_0) (h2 : S = (1/3) * a * t^3 + V_0 * t) : t = (V - V_0) / a :=
sorry

end find_t_eq_l917_91794


namespace range_of_a_l917_91788

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, (2 * a + 1) * x + a - 2 > (2 * a + 1) * 0 + a - 2)
  (h2 : a - 2 < 0) : -1 / 2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l917_91788


namespace tim_coins_value_l917_91768

variable (d q : ℕ)

-- Given Conditions
def total_coins (d q : ℕ) : Prop := d + q = 18
def quarter_to_dime_relation (d q : ℕ) : Prop := q = d + 2

-- Prove the value of the coins
theorem tim_coins_value (d q : ℕ) (h1 : total_coins d q) (h2 : quarter_to_dime_relation d q) : 10 * d + 25 * q = 330 := by
  sorry

end tim_coins_value_l917_91768


namespace correct_calculation_l917_91796

theorem correct_calculation (a b : ℝ) :
  ((ab)^3 = a^3 * b^3) ∧ 
  ¬(a + 2 * a^2 = 3 * a^3) ∧ 
  ¬(a * (-a)^4 = -a^5) ∧ 
  ¬((a^3)^2 = a^5) :=
  by
  sorry

end correct_calculation_l917_91796


namespace shifted_parabola_correct_l917_91766

-- Define original equation of parabola
def original_parabola (x : ℝ) : ℝ := 2 * x^2 - 1

-- Define shifted equation of parabola
def shifted_parabola (x : ℝ) : ℝ := 2 * (x + 1)^2 - 1

-- Proof statement: the expression of the new parabola after shifting 1 unit to the left
theorem shifted_parabola_correct :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 1) :=
by
  -- Proof is omitted, sorry
  sorry

end shifted_parabola_correct_l917_91766


namespace smallest_number_of_students_l917_91775

theorem smallest_number_of_students
  (tenth_graders eighth_graders ninth_graders : ℕ)
  (ratio1 : 7 * eighth_graders = 4 * tenth_graders)
  (ratio2 : 9 * ninth_graders = 5 * tenth_graders) :
  (∀ n, (∃ a b c, a = 7 * b ∧ b = 4 * n ∧ a = 9 * c ∧ c = 5 * n) → n = 134) :=
by {
  -- We currently just assume the result for Lean to be syntactically correct
  sorry
}

end smallest_number_of_students_l917_91775


namespace work_together_10_days_l917_91762

noncomputable def rate_A (W : ℝ) : ℝ := W / 20
noncomputable def rate_B (W : ℝ) : ℝ := W / 20

theorem work_together_10_days (W : ℝ) (hW : W > 0) :
  let A := rate_A W
  let B := rate_B W
  let combined_rate := A + B
  W / combined_rate = 10 :=
by
  sorry

end work_together_10_days_l917_91762


namespace integer_solution_unique_l917_91717

theorem integer_solution_unique
  (a b c d : ℤ)
  (h : a^2 + 5 * b^2 - 2 * c^2 - 2 * c * d - 3 * d^2 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end integer_solution_unique_l917_91717


namespace probability_fourth_roll_six_l917_91740

noncomputable def fair_die_prob : ℚ := 1 / 6
noncomputable def biased_die_prob : ℚ := 3 / 4
noncomputable def biased_die_other_face_prob : ℚ := 1 / 20
noncomputable def prior_prob : ℚ := 1 / 2

def p := 41
def q := 67

theorem probability_fourth_roll_six (p q : ℕ) (h1 : fair_die_prob = 1 / 6) (h2 : biased_die_prob = 3 / 4) (h3 : prior_prob = 1 / 2) :
  p + q = 108 :=
sorry

end probability_fourth_roll_six_l917_91740


namespace Helen_raisins_l917_91773

/-- Given that Helen baked 19 chocolate chip cookies yesterday, baked some raisin cookies and 237 chocolate chip cookies this morning,
    and baked 25 more chocolate chip cookies than raisin cookies in total,
    prove that the number of raisin cookies (R) she baked is 231. -/
theorem Helen_raisins (R : ℕ) (h1 : 25 + R = 256) : R = 231 :=
by
  sorry

end Helen_raisins_l917_91773


namespace induction_proof_l917_91729

open Nat

noncomputable def S (n : ℕ) : ℚ :=
  match n with
  | 0     => 0
  | (n+1) => S n + 1 / ((n+1) * (n+2))

theorem induction_proof : ∀ n : ℕ, S n = n / (n + 1) := by
  intro n
  induction n with
  | zero => 
    -- Base case: S(1) = 1/2
    sorry
  | succ n ih =>
    -- Induction step: Assume S(n) = n / (n + 1), prove S(n+1) = (n+1) / (n+2)
    sorry

end induction_proof_l917_91729


namespace initial_cupcakes_baked_l917_91793

variable (toddAte := 21)       -- Todd ate 21 cupcakes.
variable (packages := 6)       -- She could make 6 packages.
variable (cupcakesPerPackage := 3) -- Each package contains 3 cupcakes.
variable (cupcakesLeft := packages * cupcakesPerPackage) -- Cupcakes left after Todd ate some.

theorem initial_cupcakes_baked : cupcakesLeft + toddAte = 39 :=
by
  -- Proof placeholder
  sorry

end initial_cupcakes_baked_l917_91793


namespace unique_cube_coloring_l917_91752

-- Definition of vertices at the bottom of the cube with specific colors
inductive Color 
| Red | Green | Blue | Purple

open Color

def bottom_colors : Fin 4 → Color
| 0 => Red
| 1 => Green
| 2 => Blue
| 3 => Purple

-- Definition of the property that ensures each face of the cube has different colored corners
def all_faces_different_colors (top_colors : Fin 4 → Color) : Prop :=
  (top_colors 0 ≠ Red) ∧ (top_colors 0 ≠ Green) ∧ (top_colors 0 ≠ Blue) ∧
  (top_colors 1 ≠ Green) ∧ (top_colors 1 ≠ Blue) ∧ (top_colors 1 ≠ Purple) ∧
  (top_colors 2 ≠ Red) ∧ (top_colors 2 ≠ Blue) ∧ (top_colors 2 ≠ Purple) ∧
  (top_colors 3 ≠ Red) ∧ (top_colors 3 ≠ Green) ∧ (top_colors 3 ≠ Purple)

-- Prove there is exactly one way to achieve this coloring of the top corners
theorem unique_cube_coloring : ∃! (top_colors : Fin 4 → Color), all_faces_different_colors top_colors :=
sorry

end unique_cube_coloring_l917_91752


namespace probability_all_different_digits_l917_91714

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l917_91714


namespace amount_paid_is_200_l917_91756

-- Definitions of the costs and change received
def cost_of_pants := 140
def cost_of_shirt := 43
def cost_of_tie := 15
def change_received := 2

-- Total cost calculation
def total_cost := cost_of_pants + cost_of_shirt + cost_of_tie

-- Lean proof statement
theorem amount_paid_is_200 : total_cost + change_received = 200 := by
  -- Definitions ensure the total cost and change received are used directly from conditions
  sorry

end amount_paid_is_200_l917_91756


namespace simplify_expression_l917_91713

theorem simplify_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) : 
  Real.sqrt (1 - 2 * Real.sin 2 * Real.cos 2) = Real.sin 2 - Real.cos 2 :=
sorry

end simplify_expression_l917_91713


namespace train_length_is_correct_l917_91781

noncomputable def speed_kmph_to_mps (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def distance_crossed (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

noncomputable def train_length (speed_kmph crossing_time bridge_length : ℝ) : ℝ :=
  distance_crossed (speed_kmph_to_mps speed_kmph) crossing_time - bridge_length

theorem train_length_is_correct :
  ∀ (crossing_time bridge_length speed_kmph : ℝ),
    crossing_time = 26.997840172786177 →
    bridge_length = 150 →
    speed_kmph = 36 →
    train_length speed_kmph crossing_time bridge_length = 119.97840172786177 :=
by
  intros crossing_time bridge_length speed_kmph h1 h2 h3
  rw [h1, h2, h3]
  simp only [speed_kmph_to_mps, distance_crossed, train_length]
  sorry

end train_length_is_correct_l917_91781


namespace number_of_trees_l917_91753

theorem number_of_trees (length_of_yard : ℕ) (distance_between_trees : ℕ) 
(h1 : length_of_yard = 273) 
(h2 : distance_between_trees = 21) : 
(length_of_yard / distance_between_trees) + 1 = 14 := by
  sorry

end number_of_trees_l917_91753


namespace wait_time_probability_l917_91792

theorem wait_time_probability
  (P_B1_8_00 : ℚ)
  (P_B1_8_20 : ℚ)
  (P_B1_8_40 : ℚ)
  (P_B2_9_00 : ℚ)
  (P_B2_9_20 : ℚ)
  (P_B2_9_40 : ℚ)
  (h_independent : true)
  (h_employee_arrival : true)
  (h_P_B1 : P_B1_8_00 = 1/4 ∧ P_B1_8_20 = 1/2 ∧ P_B1_8_40 = 1/4)
  (h_P_B2 : P_B2_9_00 = 1/4 ∧ P_B2_9_20 = 1/2 ∧ P_B2_9_40 = 1/4) :
  (P_B1_8_00 * P_B2_9_20 + P_B1_8_00 * P_B2_9_40 = 3/16) :=
sorry

end wait_time_probability_l917_91792


namespace milk_cost_l917_91734

theorem milk_cost (x : ℝ) (h1 : 4 * 2.50 + 2 * x = 17) : x = 3.50 :=
by
  sorry

end milk_cost_l917_91734


namespace monomial_sum_l917_91789

theorem monomial_sum (m n : ℤ) (h1 : n - 1 = 4) (h2 : m - 1 = 2) : m - 2 * n = -7 := by
  sorry

end monomial_sum_l917_91789


namespace solution_set_of_inequalities_l917_91750

-- Definitions
def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

def is_strictly_decreasing (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Main Statement
theorem solution_set_of_inequalities
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_periodic : is_periodic f 2)
  (h_decreasing : is_strictly_decreasing f 0 1)
  (h_f_pi : f π = 1)
  (h_f_2pi : f (2 * π) = 2) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ f x ∧ f x ≤ 2} = {x | π - 2 ≤ x ∧ x ≤ 8 - 2 * π} :=
  sorry

end solution_set_of_inequalities_l917_91750


namespace chord_length_l917_91741

theorem chord_length {r : ℝ} (h : r = 15) : 
  ∃ (CD : ℝ), CD = 26 * Real.sqrt 3 :=
by
  sorry

end chord_length_l917_91741


namespace rostov_survey_min_players_l917_91703

theorem rostov_survey_min_players :
  ∃ m : ℕ, (∀ n : ℕ, n < m → (95 + n * 1) % 100 ≠ 0) ∧ m = 11 :=
sorry

end rostov_survey_min_players_l917_91703


namespace tenly_more_stuffed_animals_than_kenley_l917_91769

def mckenna_stuffed_animals := 34
def kenley_stuffed_animals := 2 * mckenna_stuffed_animals
def total_stuffed_animals_all := 175
def total_stuffed_animals_mckenna_kenley := mckenna_stuffed_animals + kenley_stuffed_animals
def tenly_stuffed_animals := total_stuffed_animals_all - total_stuffed_animals_mckenna_kenley
def stuffed_animals_difference := tenly_stuffed_animals - kenley_stuffed_animals

theorem tenly_more_stuffed_animals_than_kenley :
  stuffed_animals_difference = 5 := by
  sorry

end tenly_more_stuffed_animals_than_kenley_l917_91769


namespace value_of_nested_fraction_l917_91757

theorem value_of_nested_fraction :
  10 + 5 + (1 / 2) * (9 + 5 + (1 / 2) * (8 + 5 + (1 / 2) * (7 + 5 + (1 / 2) * (6 + 5 + (1 / 2) * (5 + 5 + (1 / 2) * (4 + 5 + (1 / 2) * (3 + 5 ))))))) = 28 + (1 / 128) :=
sorry

end value_of_nested_fraction_l917_91757


namespace cos_product_triangle_l917_91749

theorem cos_product_triangle (A B C : ℝ) (h : A + B + C = π) (hA : A > 0) (hB : B > 0) (hC : C > 0) : 
  Real.cos A * Real.cos B * Real.cos C ≤ 1 / 8 := 
sorry

end cos_product_triangle_l917_91749


namespace sum_of_squares_l917_91712

theorem sum_of_squares (x : ℤ) (h : (x + 1) ^ 2 - x ^ 2 = 199) : x ^ 2 + (x + 1) ^ 2 = 19801 :=
sorry

end sum_of_squares_l917_91712


namespace simplify_fraction_l917_91716

variable {F : Type*} [Field F]

theorem simplify_fraction (a b : F) (h: a ≠ -1) :
  b / (a * b + b) = 1 / (a + 1) :=
by
  sorry

end simplify_fraction_l917_91716


namespace min_colors_required_l917_91795

-- Define predicate for the conditions
def conditions (n : ℕ) (m : ℕ) (k : ℕ)(Paint : ℕ → Set ℕ) : Prop := 
  (∀ S : Finset ℕ, S.card = n → (∃ c ∈ ⋃ p ∈ S, Paint p, c ∈ S)) ∧ 
  (∀ c, ¬ (∀ i ∈ (Finset.range m).1, c ∈ Paint i))

-- The main theorem statement
theorem min_colors_required :
  ∀ (Paint : ℕ → Set ℕ), conditions 20 100 21 Paint → 
  ∃ k, conditions 20 100 k Paint ∧ k = 21 :=
sorry

end min_colors_required_l917_91795


namespace angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l917_91772

-- Conditions for (1): In ΔABC, A = 60°, a = 4√3, b = 4√2, prove B = 45°.
theorem angle_B_in_triangle_ABC
  (A : Real)
  (a b : Real)
  (hA : A = 60)
  (ha : a = 4 * Real.sqrt 3)
  (hb : b = 4 * Real.sqrt 2) :
  ∃ B : Real, B = 45 := by
  sorry

-- Conditions for (2): In ΔABC, a = 3√3, c = 2, B = 150°, prove b = 7.
theorem side_b_in_triangle_ABC
  (a c B : Real)
  (ha : a = 3 * Real.sqrt 3)
  (hc : c = 2)
  (hB : B = 150) :
  ∃ b : Real, b = 7 := by
  sorry

end angle_B_in_triangle_ABC_side_b_in_triangle_ABC_l917_91772


namespace valid_fraction_l917_91758

theorem valid_fraction (x: ℝ) : x^2 + 1 ≠ 0 :=
by
  sorry

end valid_fraction_l917_91758


namespace ratio_of_areas_of_concentric_circles_l917_91746

theorem ratio_of_areas_of_concentric_circles
  (C1 C2 : ℝ)
  (h : (30 / 360) * C1 = (24 / 360) * C2) :
  (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 16 / 25 := by
  sorry

end ratio_of_areas_of_concentric_circles_l917_91746


namespace each_boy_makes_14_dollars_l917_91711

noncomputable def victor_shrimp_caught := 26
noncomputable def austin_shrimp_caught := victor_shrimp_caught - 8
noncomputable def brian_shrimp_caught := (victor_shrimp_caught + austin_shrimp_caught) / 2
noncomputable def total_shrimp_caught := victor_shrimp_caught + austin_shrimp_caught + brian_shrimp_caught
noncomputable def money_made := (total_shrimp_caught / 11) * 7
noncomputable def each_boys_earnings := money_made / 3

theorem each_boy_makes_14_dollars : each_boys_earnings = 14 := by
  sorry

end each_boy_makes_14_dollars_l917_91711


namespace ab_plus_2_l917_91776

theorem ab_plus_2 (a b : ℝ) (h : ∀ x : ℝ, (x - 3) * (3 * x + 7) = x^2 - 12 * x + 27 → x = a ∨ x = b) (ha : a ≠ b) :
  (a + 2) * (b + 2) = -30 :=
sorry

end ab_plus_2_l917_91776


namespace bear_problem_l917_91767

-- Definitions of the variables
variables (W B Br : ℕ)

-- Given conditions
def condition1 : B = 2 * W := sorry
def condition2 : B = 60 := sorry
def condition3 : W + B + Br = 190 := sorry

-- The proof statement
theorem bear_problem : Br - B = 40 :=
by
  -- we would use the given conditions to prove this statement
  sorry

end bear_problem_l917_91767


namespace solve_system_l917_91770

theorem solve_system :
  ∀ (x y z : ℝ),
  (x^2 - 23 * y - 25 * z = -681) →
  (y^2 - 21 * x - 21 * z = -419) →
  (z^2 - 19 * x - 21 * y = -313) →
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by
  intros x y z h1 h2 h3
  sorry

end solve_system_l917_91770


namespace quadratic_single_intersection_l917_91791

theorem quadratic_single_intersection (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + m = 0 → x^2 - 2 * x + m = (x-1)^2) :=
sorry

end quadratic_single_intersection_l917_91791


namespace solve_for_x_l917_91705

theorem solve_for_x : ∃ x : ℚ, (1/4 : ℚ) + (1/x) = 7/8 ∧ x = 8/5 :=
by {
  sorry
}

end solve_for_x_l917_91705


namespace planting_flowers_cost_l917_91765

theorem planting_flowers_cost 
  (flower_cost : ℕ) (clay_cost : ℕ) (soil_cost : ℕ)
  (h₁ : flower_cost = 9)
  (h₂ : clay_cost = flower_cost + 20)
  (h₃ : soil_cost = flower_cost - 2) :
  flower_cost + clay_cost + soil_cost = 45 :=
sorry

end planting_flowers_cost_l917_91765


namespace second_race_distance_remaining_l917_91715

theorem second_race_distance_remaining
  (race_distance : ℕ)
  (A_finish_time : ℕ)
  (B_remaining_distance : ℕ)
  (A_start_behind : ℕ)
  (A_speed : ℝ)
  (B_speed : ℝ)
  (A_distance_second_race : ℕ)
  (B_distance_second_race : ℝ)
  (v_ratio : ℝ)
  (B_remaining_second_race : ℝ) :
  race_distance = 10000 →
  A_finish_time = 50 →
  B_remaining_distance = 500 →
  A_start_behind = 500 →
  A_speed = race_distance / A_finish_time →
  B_speed = (race_distance - B_remaining_distance) / A_finish_time →
  v_ratio = A_speed / B_speed →
  v_ratio = 20 / 19 →
  A_distance_second_race = race_distance + A_start_behind →
  B_distance_second_race = B_speed * (A_distance_second_race / A_speed) →
  B_remaining_second_race = race_distance - B_distance_second_race →
  B_remaining_second_race = 25 := 
by
  sorry

end second_race_distance_remaining_l917_91715


namespace lexi_laps_l917_91780

theorem lexi_laps (total_distance lap_distance : ℝ) (h1 : total_distance = 3.25) (h2 : lap_distance = 0.25) :
  total_distance / lap_distance = 13 :=
by
  sorry

end lexi_laps_l917_91780


namespace angle_BAC_l917_91720

theorem angle_BAC
  (elevation_angle_B_from_A : ℝ)
  (depression_angle_C_from_A : ℝ)
  (h₁ : elevation_angle_B_from_A = 60)
  (h₂ : depression_angle_C_from_A = 70) :
  elevation_angle_B_from_A + depression_angle_C_from_A = 130 :=
by
  sorry

end angle_BAC_l917_91720


namespace power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l917_91754

-- Definitions and conditions
def is_power_of_2 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

def is_sum_of_two_or_more_consecutive_naturals (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k ≥ 2 ∧ n = (k * a) + (k * (k - 1)) / 2

-- Proofs to be stated
theorem power_of_2_not_sum_of_consecutive (n : ℕ) (h : is_power_of_2 n) : ¬ is_sum_of_two_or_more_consecutive_naturals n :=
by
    sorry

theorem not_power_of_2_is_sum_of_consecutive (M : ℕ) (h : ¬ is_power_of_2 M) : is_sum_of_two_or_more_consecutive_naturals M :=
by
    sorry

end power_of_2_not_sum_of_consecutive_not_power_of_2_is_sum_of_consecutive_l917_91754


namespace gcd_of_128_144_480_450_l917_91790

theorem gcd_of_128_144_480_450 : Nat.gcd (Nat.gcd 128 144) (Nat.gcd 480 450) = 6 := 
by
  sorry

end gcd_of_128_144_480_450_l917_91790


namespace inequality_solution_set_l917_91755

theorem inequality_solution_set (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end inequality_solution_set_l917_91755


namespace row_even_col_odd_contradiction_row_odd_col_even_contradiction_l917_91774

theorem row_even_col_odd_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∃ i : Fin 15, M r i = 2) ∧ 
      (∀ c : Fin 15, ∀ j : Fin 20, M j c = 5)) := 
sorry

theorem row_odd_col_even_contradiction : 
  ¬ (∃ (M : Matrix (Fin 20) (Fin 15) ℕ), 
      (∀ r : Fin 20, ∀ i : Fin 15, M r i = 5) ∧ 
      (∀ c : Fin 15, ∃ j : Fin 20, M j c = 2)) := 
sorry

end row_even_col_odd_contradiction_row_odd_col_even_contradiction_l917_91774


namespace minimum_a_inequality_l917_91721

variable {x y : ℝ}

/-- The inequality (x + y) * (1/x + a/y) ≥ 9 holds for any positive real numbers x and y 
     if and only if a ≥ 4.  -/
theorem minimum_a_inequality (a : ℝ) (h : ∀ (x y : ℝ), 0 < x → 0 < y → (x + y) * (1 / x + a / y) ≥ 9) :
  a ≥ 4 :=
by
  sorry

end minimum_a_inequality_l917_91721


namespace solve_abs_quadratic_l917_91737

theorem solve_abs_quadratic :
  ∃ x : ℝ, (|x - 3| + x^2 = 10) ∧ 
  (x = (-1 + Real.sqrt 53) / 2 ∨ x = (1 + Real.sqrt 29) / 2 ∨ x = (1 - Real.sqrt 29) / 2) :=
by sorry

end solve_abs_quadratic_l917_91737


namespace interest_rate_of_first_investment_l917_91784

theorem interest_rate_of_first_investment (x y : ℝ) (h1 : x + y = 2000) (h2 : y = 650) (h3 : 0.10 * x - 0.08 * y = 83) : (0.10 * x) / x = 0.10 := by
  sorry

end interest_rate_of_first_investment_l917_91784


namespace solve_quadratic_eq_l917_91718

theorem solve_quadratic_eq (x : ℝ) : x ^ 2 + 2 * x - 5 = 0 → (x = -1 + Real.sqrt 6 ∨ x = -1 - Real.sqrt 6) :=
by 
  intro h
  sorry

end solve_quadratic_eq_l917_91718


namespace average_difference_l917_91733

theorem average_difference :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 28) / 3
  avg1 - avg2 = 4 :=
by
  -- Define the averages
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 28) / 3
  sorry

end average_difference_l917_91733


namespace sqrt_neg4_sq_eq_4_l917_91704

theorem sqrt_neg4_sq_eq_4 : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := by
  sorry

end sqrt_neg4_sq_eq_4_l917_91704


namespace ratio_PeteHand_to_TracyCartwheel_l917_91731

noncomputable def SusanWalkingSpeed (PeteBackwardSpeed : ℕ) : ℕ :=
  PeteBackwardSpeed / 3

noncomputable def TracyCartwheelSpeed (SusanSpeed : ℕ) : ℕ :=
  SusanSpeed * 2

def PeteHandsWalkingSpeed : ℕ := 2

def PeteBackwardWalkingSpeed : ℕ := 12

theorem ratio_PeteHand_to_TracyCartwheel :
  let SusanSpeed := SusanWalkingSpeed PeteBackwardWalkingSpeed
  let TracySpeed := TracyCartwheelSpeed SusanSpeed
  (PeteHandsWalkingSpeed : ℕ) / (TracySpeed : ℕ) = 1 / 4 :=
by
  sorry

end ratio_PeteHand_to_TracyCartwheel_l917_91731


namespace parallel_lines_k_value_l917_91701

theorem parallel_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 5 * x + 3 → y = (3 * k) * x + 1 → true) → k = 5 / 3 :=
by
  intros
  sorry

end parallel_lines_k_value_l917_91701


namespace motorcyclist_average_speed_l917_91710

theorem motorcyclist_average_speed :
  let distance_ab := 120
  let speed_ab := 45
  let distance_bc := 130
  let speed_bc := 60
  let distance_cd := 150
  let speed_cd := 50
  let time_ab := distance_ab / speed_ab
  let time_bc := distance_bc / speed_bc
  let time_cd := distance_cd / speed_cd
  (time_ab = time_bc + 2)
  → (time_cd = time_ab / 2)
  → avg_speed = (distance_ab + distance_bc + distance_cd) / (time_ab + time_bc + time_cd)
  → avg_speed = 2400 / 47 := sorry

end motorcyclist_average_speed_l917_91710


namespace solve_inequality_l917_91726

theorem solve_inequality (x : ℝ) :
  x * Real.log (x^2 + x + 1) / Real.log 10 < 0 ↔ x < -1 :=
sorry

end solve_inequality_l917_91726


namespace min_generic_tees_per_package_l917_91732

def total_golf_tees_needed (n : ℕ) : ℕ := 80
def max_generic_packages_used : ℕ := 2
def tees_per_aero_flight_package : ℕ := 2
def aero_flight_packages_needed : ℕ := 28
def total_tees_from_aero_flight_packages (n : ℕ) : ℕ := aero_flight_packages_needed * tees_per_aero_flight_package

theorem min_generic_tees_per_package (G : ℕ) :
  (total_golf_tees_needed 4) - (total_tees_from_aero_flight_packages aero_flight_packages_needed) ≤ max_generic_packages_used * G → G ≥ 12 :=
by
  sorry

end min_generic_tees_per_package_l917_91732


namespace john_initial_payment_l917_91799

-- Definitions based on the conditions from step a)
def cost_per_soda : ℕ := 2
def num_sodas : ℕ := 3
def change_received : ℕ := 14

-- Problem Statement: Prove that the total amount of money John paid initially is $20
theorem john_initial_payment :
  cost_per_soda * num_sodas + change_received = 20 := 
by
  sorry -- Proof steps are omitted as per instructions

end john_initial_payment_l917_91799


namespace original_volume_of_ice_l917_91700

variable (V : ℝ) 

theorem original_volume_of_ice (h1 : V * (1 / 4) * (1 / 4) = 0.25) : V = 4 :=
  sorry

end original_volume_of_ice_l917_91700


namespace ratio_of_liquid_rise_l917_91723

theorem ratio_of_liquid_rise
  (h1 h2 : ℝ) (r1 r2 rm : ℝ)
  (V1 V2 Vm : ℝ)
  (H1 : r1 = 4)
  (H2 : r2 = 9)
  (H3 : V1 = (1 / 3) * π * r1^2 * h1)
  (H4 : V2 = (1 / 3) * π * r2^2 * h2)
  (H5 : V1 = V2)
  (H6 : rm = 2)
  (H7 : Vm = (4 / 3) * π * rm^3)
  (H8 : h2 = h1 * (81 / 16))
  (h1' h2' : ℝ)
  (H9 : h1' = h1 + Vm / ((1 / 3) * π * r1^2))
  (H10 : h2' = h2 + Vm / ((1 / 3) * π * r2^2)) :
  (h1' - h1) / (h2' - h2) = 81 / 16 :=
sorry

end ratio_of_liquid_rise_l917_91723


namespace ratio_jordana_jennifer_10_years_l917_91745

-- Let's define the necessary terms and conditions:
def Jennifer_future_age := 30
def Jordana_current_age := 80
def years := 10

-- Define the ratio of ages function:
noncomputable def ratio_of_ages (future_age_jen : ℕ) (current_age_jord : ℕ) (yrs : ℕ) : ℚ :=
  (current_age_jord + yrs) / future_age_jen

-- The statement we need to prove:
theorem ratio_jordana_jennifer_10_years :
  ratio_of_ages Jennifer_future_age Jordana_current_age years = 3 := by
  sorry

end ratio_jordana_jennifer_10_years_l917_91745


namespace nails_needed_l917_91764

theorem nails_needed (nails_own nails_found nails_total_needed : ℕ) 
  (h1 : nails_own = 247) 
  (h2 : nails_found = 144) 
  (h3 : nails_total_needed = 500) : 
  nails_total_needed - (nails_own + nails_found) = 109 := 
by
  sorry

end nails_needed_l917_91764


namespace fg_of_2_l917_91730

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 - 4 * x
def g (x : ℝ) : ℝ := x^2 + 2

-- Prove the specific property
theorem fg_of_2 : f (g 2) = -19 := by
  -- Placeholder for the proof
  sorry

end fg_of_2_l917_91730


namespace smallest_five_digit_congruent_to_three_mod_seventeen_l917_91728

theorem smallest_five_digit_congruent_to_three_mod_seventeen :
  ∃ (n : ℤ), 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 3 ∧ n = 10012 :=
by
  sorry

end smallest_five_digit_congruent_to_three_mod_seventeen_l917_91728


namespace expression_result_zero_l917_91708

theorem expression_result_zero (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = y + 1) : 
  (x + 1 / x) * (y - 1 / y) = 0 := 
by sorry

end expression_result_zero_l917_91708


namespace find_sum_of_squares_l917_91744

variable (x y : ℝ)

theorem find_sum_of_squares (h₁ : x * y = 8) (h₂ : x^2 * y + x * y^2 + x + y = 94) : 
  x^2 + y^2 = 7540 / 81 :=
by
  sorry

end find_sum_of_squares_l917_91744


namespace value_of_k_l917_91724

theorem value_of_k (k : ℤ) : 
  (∀ x : ℤ, (x + k) * (x - 4) = x^2 - 4 * x + k * x - 4 * k ∧ 
  (k - 4) * x = 0) → k = 4 := 
by 
  sorry

end value_of_k_l917_91724


namespace find_x_ineq_solution_l917_91761

open Set

theorem find_x_ineq_solution :
  {x : ℝ | (x - 2) / (x - 4) ≥ 3} = Ioc 4 5 := 
sorry

end find_x_ineq_solution_l917_91761


namespace solve_for_x_l917_91722

theorem solve_for_x (x : ℝ) (h₁ : x ≠ -3) :
  (7 * x^2 - 3) / (x + 3) - 3 / (x + 3) = 1 / (x + 3) ↔ x = 1 ∨ x = -1 := 
sorry

end solve_for_x_l917_91722


namespace coloring_probability_l917_91706

-- Definition of the problem and its conditions
def num_cells := 16
def num_diags := 2
def chosen_diags := 7

-- Define the probability
noncomputable def prob_coloring_correct : ℚ :=
  (num_diags ^ chosen_diags : ℚ) / (num_diags ^ num_cells)

-- The Lean theorem statement
theorem coloring_probability : prob_coloring_correct = 1 / 512 := 
by 
  unfold prob_coloring_correct
  -- The proof steps would follow here (omitted)
  sorry

end coloring_probability_l917_91706


namespace find_b_l917_91738

theorem find_b (b : ℝ) : (∃ x y : ℝ, x = 1 ∧ y = 2 ∧ y = 2 * x + b) → b = 0 := by
  sorry

end find_b_l917_91738


namespace sacks_filled_l917_91751

theorem sacks_filled (pieces_per_sack : ℕ) (total_pieces : ℕ) (h1 : pieces_per_sack = 20) (h2 : total_pieces = 80) : (total_pieces / pieces_per_sack) = 4 :=
by {
  sorry
}

end sacks_filled_l917_91751


namespace combined_height_is_9_l917_91763

def barrys_reach : ℝ := 5 -- Barry can reach apples that are 5 feet high

def larrys_full_height : ℝ := 5 -- Larry's full height is 5 feet

def larrys_shoulder_height : ℝ := larrys_full_height * 0.8 -- Larry's shoulder height is 20% less than his full height

def combined_reach (b_reach : ℝ) (l_shoulder : ℝ) : ℝ := b_reach + l_shoulder

theorem combined_height_is_9 : combined_reach barrys_reach larrys_shoulder_height = 9 := by
  sorry

end combined_height_is_9_l917_91763


namespace evaluate_expression_l917_91748

theorem evaluate_expression :
  (3^2016 + 3^2014 + 3^2012) / (3^2016 - 3^2014 + 3^2012) = 91 / 73 := 
  sorry

end evaluate_expression_l917_91748


namespace van_distance_covered_l917_91797

noncomputable def distance_covered (V : ℝ) := 
  let D := V * 6
  D

theorem van_distance_covered : ∃ (D : ℝ), ∀ (V : ℝ), 
  (D = 288) ∧ (D = distance_covered V) ∧ (D = 32 * 9) :=
by
  sorry

end van_distance_covered_l917_91797


namespace lateral_surface_area_of_cylinder_l917_91786

theorem lateral_surface_area_of_cylinder :
  let r := 1
  let h := 2
  2 * Real.pi * r * h = 4 * Real.pi :=
by
  sorry

end lateral_surface_area_of_cylinder_l917_91786


namespace boy_usual_time_l917_91736

theorem boy_usual_time (R T : ℝ) (h : R * T = (7 / 6) * R * (T - 2)) : T = 14 :=
by
  sorry

end boy_usual_time_l917_91736


namespace ticket_cost_correct_l917_91709

def metro_sells (tickets_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  tickets_per_minute * minutes

def total_earnings (tickets_sold : ℕ) (ticket_cost : ℕ) : ℕ :=
  tickets_sold * ticket_cost

theorem ticket_cost_correct (ticket_cost : ℕ) : 
  (metro_sells 5 6 = 30) ∧ (total_earnings 30 ticket_cost = 90) → ticket_cost = 3 :=
by
  intro h
  sorry

end ticket_cost_correct_l917_91709


namespace santino_total_fruits_l917_91702

theorem santino_total_fruits :
  let p := 2
  let m := 3
  let a := 4
  let o := 5
  let fp := 10
  let fm := 20
  let fa := 15
  let fo := 25
  p * fp + m * fm + a * fa + o * fo = 265 := by
  sorry

end santino_total_fruits_l917_91702


namespace students_drawn_from_grade10_l917_91725

-- Define the initial conditions
def total_students_grade12 : ℕ := 750
def total_students_grade11 : ℕ := 850
def total_students_grade10 : ℕ := 900
def sample_size : ℕ := 50

-- Prove the number of students drawn from grade 10 is 18
theorem students_drawn_from_grade10 : 
  total_students_grade12 = 750 ∧
  total_students_grade11 = 850 ∧
  total_students_grade10 = 900 ∧
  sample_size = 50 →
  (sample_size * total_students_grade10 / 
  (total_students_grade12 + total_students_grade11 + total_students_grade10) = 18) :=
by
  sorry

end students_drawn_from_grade10_l917_91725


namespace shale_mix_per_pound_is_5_l917_91778

noncomputable def cost_of_shale_mix_per_pound 
  (cost_limestone : ℝ) (cost_compound : ℝ) (weight_limestone : ℝ) (total_weight : ℝ) : ℝ :=
  let total_cost_limestone := weight_limestone * cost_limestone 
  let weight_shale := total_weight - weight_limestone
  let total_cost := total_weight * cost_compound
  let total_cost_shale := total_cost - total_cost_limestone
  total_cost_shale / weight_shale

theorem shale_mix_per_pound_is_5 :
  cost_of_shale_mix_per_pound 3 4.25 37.5 100 = 5 := 
by 
  sorry

end shale_mix_per_pound_is_5_l917_91778


namespace arithmetic_seq_sum_l917_91798

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
    (h_sum : a 0 + a 1 + a 2 + a 3 = 30) : a 1 + a 2 = 15 :=
by
  sorry

end arithmetic_seq_sum_l917_91798


namespace share_of_e_l917_91707

variable (E F : ℝ)
variable (D : ℝ := (5/3) * E)
variable (D_alt : ℝ := (1/2) * F)
variable (E_alt : ℝ := (3/2) * F)
variable (profit : ℝ := 25000)

theorem share_of_e (h1 : D = (5/3) * E) (h2 : D = (1/2) * F) (h3 : E = (3/2) * F) :
  (E / ((5/2) * F + (3/2) * F + F)) * profit = 7500 :=
by
  sorry

end share_of_e_l917_91707


namespace derivative_f_at_zero_l917_91743

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then 4 * x * (1 - |x|) else 0

theorem derivative_f_at_zero : HasDerivAt f 4 0 :=
by
  -- Proof omitted
  sorry

end derivative_f_at_zero_l917_91743


namespace square_difference_l917_91782

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x - 2) * (x + 2) = 9797 :=
by 
  have diff_squares : (x - 2) * (x + 2) = x^2 - 4 := by ring
  rw [diff_squares, h]
  norm_num

end square_difference_l917_91782


namespace intersection_of_sets_l917_91747

def setA : Set ℝ := {x | (x - 2) / x ≤ 0}
def setB : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def setC : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets : setA ∩ setB = setC :=
by
  sorry

end intersection_of_sets_l917_91747


namespace total_students_eq_seventeen_l917_91719

theorem total_students_eq_seventeen 
    (N : ℕ)
    (initial_students : N - 1 = 16)
    (avg_first_day : 77 * (N - 1) = 77 * 16)
    (avg_second_day : 78 * N = 78 * N)
    : N = 17 :=
sorry

end total_students_eq_seventeen_l917_91719


namespace rate_of_stream_l917_91727

def effectiveSpeedDownstream (v : ℝ) : ℝ := 36 + v
def effectiveSpeedUpstream (v : ℝ) : ℝ := 36 - v

theorem rate_of_stream (v : ℝ) (hf1 : effectiveSpeedUpstream v = 3 * effectiveSpeedDownstream v) : v = 18 := by
  sorry

end rate_of_stream_l917_91727


namespace simplify_expression_l917_91783

theorem simplify_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 :=
sorry

end simplify_expression_l917_91783


namespace alexandra_brianna_meeting_probability_l917_91779

noncomputable def probability_meeting (A B : ℕ × ℕ) : ℚ :=
if A = (0,0) ∧ B = (5,7) then 347 / 768 else 0

theorem alexandra_brianna_meeting_probability :
  probability_meeting (0,0) (5,7) = 347 / 768 := 
by sorry

end alexandra_brianna_meeting_probability_l917_91779
