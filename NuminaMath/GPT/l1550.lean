import Mathlib

namespace NUMINAMATH_GPT_a_plus_b_in_D_l1550_155012

def setA : Set ℤ := {x | ∃ k : ℤ, x = 4 * k}
def setB : Set ℤ := {x | ∃ m : ℤ, x = 4 * m + 1}
def setC : Set ℤ := {x | ∃ n : ℤ, x = 4 * n + 2}
def setD : Set ℤ := {x | ∃ t : ℤ, x = 4 * t + 3}

theorem a_plus_b_in_D (a b : ℤ) (ha : a ∈ setB) (hb : b ∈ setC) : a + b ∈ setD := by
  sorry

end NUMINAMATH_GPT_a_plus_b_in_D_l1550_155012


namespace NUMINAMATH_GPT_jean_spots_on_sides_l1550_155095

variables (total_spots upper_torso_spots back_hindquarters_spots side_spots : ℕ)

def half (x : ℕ) := x / 2
def third (x : ℕ) := x / 3

-- Given conditions
axiom h1 : upper_torso_spots = 30
axiom h2 : upper_torso_spots = half total_spots
axiom h3 : back_hindquarters_spots = third total_spots
axiom h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots

-- Theorem to prove
theorem jean_spots_on_sides (h1 : upper_torso_spots = 30)
    (h2 : upper_torso_spots = half total_spots)
    (h3 : back_hindquarters_spots = third total_spots)
    (h4 : side_spots = total_spots - upper_torso_spots - back_hindquarters_spots) :
    side_spots = 10 := by
  sorry

end NUMINAMATH_GPT_jean_spots_on_sides_l1550_155095


namespace NUMINAMATH_GPT_quadratic_solve_l1550_155072

theorem quadratic_solve (x : ℝ) : (x + 4)^2 = 5 * (x + 4) → x = -4 ∨ x = 1 :=
by sorry

end NUMINAMATH_GPT_quadratic_solve_l1550_155072


namespace NUMINAMATH_GPT_beau_age_today_l1550_155062

-- Definitions based on conditions
def sons_are_triplets : Prop := ∀ (i j : Nat), i ≠ j → i = 0 ∨ i = 1 ∨ i = 2 → j = 0 ∨ j = 1 ∨ j = 2
def sons_age_today : Nat := 16
def sum_of_ages_equals_beau_age_3_years_ago (beau_age_3_years_ago : Nat) : Prop :=
  beau_age_3_years_ago = 3 * (sons_age_today - 3)

-- Proposition to prove
theorem beau_age_today (beau_age_3_years_ago : Nat) (h_triplets : sons_are_triplets) 
  (h_ages_sum : sum_of_ages_equals_beau_age_3_years_ago beau_age_3_years_ago) : 
  beau_age_3_years_ago + 3 = 42 := 
by
  sorry

end NUMINAMATH_GPT_beau_age_today_l1550_155062


namespace NUMINAMATH_GPT_problem1_problem2_l1550_155009

variable (a : ℝ)

def quadratic_roots (a x : ℝ) : Prop := a*x^2 + 2*x + 1 = 0

-- Problem 1: If 1/2 is a root, find the set A
theorem problem1 (h : quadratic_roots a (1/2)) : 
  {x : ℝ | quadratic_roots (a) x } = { -1/4, 1/2 } :=
sorry

-- Problem 2: If A contains exactly one element, find the set B consisting of such a
theorem problem2 (h : ∃! (x : ℝ), quadratic_roots a x ) : 
  {a : ℝ | ∃! (x : ℝ), quadratic_roots a x} = { 0, 1 } :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1550_155009


namespace NUMINAMATH_GPT_hearts_per_card_l1550_155008

-- Definitions of the given conditions
def num_suits := 4
def num_cards_total := 52
def num_cards_per_suit := num_cards_total / num_suits
def cost_per_cow := 200
def total_cost := 83200
def num_cows := total_cost / cost_per_cow

-- The mathematical proof problem translated to Lean 4:
theorem hearts_per_card :
    (2 * (num_cards_total / num_suits) = num_cows) → (num_cows = 416) → (num_cards_total / num_suits = 208) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_hearts_per_card_l1550_155008


namespace NUMINAMATH_GPT_general_term_formula_sum_first_n_terms_l1550_155067

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, a n = 3^(n-2) := 
by
  sorry

theorem sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (hS3 : S 3 = a 2 + 10 * a 1)
    (ha5 : a 5 = 9) : ∀ n, S n = (3^(n-2)) / 2 - 1 / 18 := 
by
  sorry

end NUMINAMATH_GPT_general_term_formula_sum_first_n_terms_l1550_155067


namespace NUMINAMATH_GPT_clea_ride_escalator_time_l1550_155046

theorem clea_ride_escalator_time (x y k : ℝ) (h1 : 80 * x = y) (h2 : 30 * (x + k) = y) : (y / k) + 5 = 53 :=
by {
  sorry
}

end NUMINAMATH_GPT_clea_ride_escalator_time_l1550_155046


namespace NUMINAMATH_GPT_line_canonical_form_l1550_155097

theorem line_canonical_form :
  (∀ x y z : ℝ, 4 * x + y - 3 * z + 2 = 0 → 2 * x - y + z - 8 = 0 ↔
    ∃ t : ℝ, x = 1 + -2 * t ∧ y = -6 + -10 * t ∧ z = -6 * t) :=
by
  sorry

end NUMINAMATH_GPT_line_canonical_form_l1550_155097


namespace NUMINAMATH_GPT_probability_different_colors_l1550_155011

def total_chips := 7 + 5 + 4

def probability_blue_draw : ℚ := 7 / total_chips
def probability_red_draw : ℚ := 5 / total_chips
def probability_yellow_draw : ℚ := 4 / total_chips
def probability_different_color (color1_prob color2_prob : ℚ) : ℚ := color1_prob * (1 - color2_prob)

theorem probability_different_colors :
  (probability_blue_draw * probability_different_color 7 (7 / total_chips)) +
  (probability_red_draw * probability_different_color 5 (5 / total_chips)) +
  (probability_yellow_draw * probability_different_color 4 (4 / total_chips)) 
  = 83 / 128 := 
by 
  sorry

end NUMINAMATH_GPT_probability_different_colors_l1550_155011


namespace NUMINAMATH_GPT_smallest_number_of_rectangles_l1550_155069

theorem smallest_number_of_rectangles (m n a b : ℕ) (h₁ : m = 12) (h₂ : n = 12) (h₃ : a = 3) (h₄ : b = 4) :
  (12 * 12) / (3 * 4) = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_rectangles_l1550_155069


namespace NUMINAMATH_GPT_unique_solution_condition_l1550_155042

theorem unique_solution_condition (a b : ℝ) : (4 * x - 6 + a = (b + 1) * x + 2) → b ≠ 3 :=
by
  intro h
  -- Given the condition equation
  have eq1 : 4 * x - 6 + a = (b + 1) * x + 2 := h
  -- Simplify to the form (3 - b) * x = 8 - a
  sorry

end NUMINAMATH_GPT_unique_solution_condition_l1550_155042


namespace NUMINAMATH_GPT_ages_correct_l1550_155003

-- Let A be Anya's age and P be Petya's age
def anya_age : ℕ := 4
def petya_age : ℕ := 12

-- The conditions
def condition1 (A P : ℕ) : Prop := P = 3 * A
def condition2 (A P : ℕ) : Prop := P - A = 8

-- The statement to be proven
theorem ages_correct : condition1 anya_age petya_age ∧ condition2 anya_age petya_age :=
by
  unfold condition1 condition2 anya_age petya_age -- Reveal the definitions
  have h1 : petya_age = 3 * anya_age := by
    sorry
  have h2 : petya_age - anya_age = 8 := by
    sorry
  exact ⟨h1, h2⟩ -- Combine both conditions into a single conjunction

end NUMINAMATH_GPT_ages_correct_l1550_155003


namespace NUMINAMATH_GPT_ellipse_tangent_line_l1550_155084

theorem ellipse_tangent_line (m : ℝ) : 
  (∀ (x y : ℝ), (x ^ 2 / 4) + (y ^ 2 / m) = 1 → (y = mx + 2)) → m = 1 :=
by sorry

end NUMINAMATH_GPT_ellipse_tangent_line_l1550_155084


namespace NUMINAMATH_GPT_gcd_fact_plus_two_l1550_155092

theorem gcd_fact_plus_two (n m : ℕ) (h1 : n = 6) (h2 : m = 8) :
  Nat.gcd (n.factorial + 2) (m.factorial + 2) = 2 :=
  sorry

end NUMINAMATH_GPT_gcd_fact_plus_two_l1550_155092


namespace NUMINAMATH_GPT_area_ratio_of_squares_l1550_155045

theorem area_ratio_of_squares (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a * a) = 16 * (b * b) :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l1550_155045


namespace NUMINAMATH_GPT_tips_fraction_l1550_155030

theorem tips_fraction (S T : ℝ) (h : T / (S + T) = 0.6363636363636364) : T / S = 1.75 :=
sorry

end NUMINAMATH_GPT_tips_fraction_l1550_155030


namespace NUMINAMATH_GPT_rhombus_obtuse_angle_l1550_155001

theorem rhombus_obtuse_angle (perimeter height : ℝ) (h_perimeter : perimeter = 8) (h_height : height = 1) : 
  ∃ θ : ℝ, θ = 150 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_obtuse_angle_l1550_155001


namespace NUMINAMATH_GPT_avg_divisible_by_4_between_15_and_55_eq_34_l1550_155054

theorem avg_divisible_by_4_between_15_and_55_eq_34 :
  let numbers := (List.filter (λ x => x % 4 = 0) (List.range' 16 37))
  (List.sum numbers) / (numbers.length) = 34 := by
  sorry

end NUMINAMATH_GPT_avg_divisible_by_4_between_15_and_55_eq_34_l1550_155054


namespace NUMINAMATH_GPT_find_polynomial_coefficients_l1550_155059

-- Define the quadratic polynomial q(x) = ax^2 + bx + c
def polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions for polynomial
axiom condition1 (a b c : ℝ) : polynomial a b c (-2) = 9
axiom condition2 (a b c : ℝ) : polynomial a b c 1 = 2
axiom condition3 (a b c : ℝ) : polynomial a b c 3 = 10

-- Conjecture for the polynomial q(x)
theorem find_polynomial_coefficients : 
  ∃ (a b c : ℝ), 
    polynomial a b c (-2) = 9 ∧
    polynomial a b c 1 = 2 ∧
    polynomial a b c 3 = 10 ∧
    a = 19 / 15 ∧
    b = -2 / 15 ∧
    c = 13 / 15 :=
by {
  -- Placeholder proof
  sorry
}

end NUMINAMATH_GPT_find_polynomial_coefficients_l1550_155059


namespace NUMINAMATH_GPT_find_x_value_l1550_155021

theorem find_x_value : (8 = 2^3) ∧ (8 * 8^32 = 8^33) ∧ (8^33 = 2^99) → ∃ x, 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 + 8^32 = 2^x ∧ x = 99 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_x_value_l1550_155021


namespace NUMINAMATH_GPT_max_days_for_C_l1550_155058

-- Define the durations of the processes and the total project duration
def A := 2
def B := 5
def D := 4
def T := 9

-- Define the condition to prove the maximum days required for process C
theorem max_days_for_C (x : ℕ) (h : 2 + x + 4 = 9) : x = 3 := by
  sorry

end NUMINAMATH_GPT_max_days_for_C_l1550_155058


namespace NUMINAMATH_GPT_lars_bakes_for_six_hours_l1550_155039

variable (h : ℕ)

-- Conditions
def bakes_loaves : ℕ := 10 * h
def bakes_baguettes : ℕ := 15 * h
def total_breads : ℕ := bakes_loaves h + bakes_baguettes h

-- Proof goal
theorem lars_bakes_for_six_hours (h : ℕ) (H : total_breads h = 150) : h = 6 :=
sorry

end NUMINAMATH_GPT_lars_bakes_for_six_hours_l1550_155039


namespace NUMINAMATH_GPT_smallest_number_l1550_155089

-- Definitions based on the conditions given in the problem
def satisfies_conditions (b : ℕ) : Prop :=
  b % 5 = 2 ∧ b % 4 = 3 ∧ b % 7 = 1

-- Lean proof statement
theorem smallest_number (b : ℕ) : satisfies_conditions b → b = 87 :=
sorry

end NUMINAMATH_GPT_smallest_number_l1550_155089


namespace NUMINAMATH_GPT_john_labor_cost_l1550_155016

def plank_per_tree : ℕ := 25
def table_cost : ℕ := 300
def profit : ℕ := 12000
def trees_chopped : ℕ := 30
def planks_per_table : ℕ := 15
def total_table_revenue := (trees_chopped * plank_per_tree / planks_per_table) * table_cost
def labor_cost := total_table_revenue - profit

theorem john_labor_cost :
  labor_cost = 3000 :=
by
  sorry

end NUMINAMATH_GPT_john_labor_cost_l1550_155016


namespace NUMINAMATH_GPT_shoe_size_ratio_l1550_155099

theorem shoe_size_ratio (J A : ℕ) (hJ : J = 7) (hAJ : A + J = 21) : A / J = 2 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_shoe_size_ratio_l1550_155099


namespace NUMINAMATH_GPT_complement_A_intersect_B_eq_l1550_155051

def setA : Set ℝ := { x : ℝ | |x - 2| ≤ 2 }

def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }

def A_intersect_B := setA ∩ setB

def complement (A : Set ℝ) : Set ℝ := { x : ℝ | x ∉ A }

theorem complement_A_intersect_B_eq {A : Set ℝ} {B : Set ℝ} 
  (hA : A = { x : ℝ | |x - 2| ≤ 2 })
  (hB : B = { y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2 }) :
  complement (A ∩ B) = { x : ℝ | x ≠ 0 } :=
by
  sorry

end NUMINAMATH_GPT_complement_A_intersect_B_eq_l1550_155051


namespace NUMINAMATH_GPT_complex_fraction_eval_l1550_155061

theorem complex_fraction_eval (i : ℂ) (hi : i^2 = -1) : (3 + i) / (1 + i) = 2 - i := 
by 
  sorry

end NUMINAMATH_GPT_complex_fraction_eval_l1550_155061


namespace NUMINAMATH_GPT_population_Lake_Bright_l1550_155028

-- Definition of total population
def T := 80000

-- Definition of population of Gordonia
def G := (1 / 2) * T

-- Definition of population of Toadon
def Td := (60 / 100) * G

-- Proof that the population of Lake Bright is 16000
theorem population_Lake_Bright : T - (G + Td) = 16000 :=
by {
    -- Leaving the proof as sorry
    sorry
}

end NUMINAMATH_GPT_population_Lake_Bright_l1550_155028


namespace NUMINAMATH_GPT_sequence_a100_l1550_155098

theorem sequence_a100 :
  ∃ a : ℕ → ℕ, (a 1 = 1) ∧ (∀ m n : ℕ, 0 < m → 0 < n → a (n + m) = a n + a m + n * m) ∧ (a 100 = 5050) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a100_l1550_155098


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l1550_155000

theorem average_of_remaining_two_numbers (S a₁ a₂ a₃ a₄ : ℝ)
    (h₁ : S / 6 = 3.95)
    (h₂ : (a₁ + a₂) / 2 = 3.8)
    (h₃ : (a₃ + a₄) / 2 = 3.85) :
    (S - (a₁ + a₂ + a₃ + a₄)) / 2 = 4.2 := 
sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l1550_155000


namespace NUMINAMATH_GPT_third_cyclist_speed_l1550_155065

theorem third_cyclist_speed (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : 
  ∃ V : ℝ, V = (a + 3 * b + Real.sqrt (a^2 - 10 * a * b + 9 * b^2)) / 4 :=
by
  sorry

end NUMINAMATH_GPT_third_cyclist_speed_l1550_155065


namespace NUMINAMATH_GPT_incorrect_statement_proof_l1550_155082

-- Define the conditions as assumptions
def inductive_reasoning_correct : Prop := ∀ (P : Prop), ¬(P → P)
def analogical_reasoning_correct : Prop := ∀ (P Q : Prop), ¬(P → Q)
def reasoning_by_plausibility_correct : Prop := ∀ (P : Prop), ¬(P → P)

-- Define the incorrect statement to be proven
def inductive_reasoning_incorrect_statement : Prop := 
  ¬ (∀ (P Q : Prop), ¬(P ↔ Q))

-- The theorem to be proven
theorem incorrect_statement_proof 
  (h1 : inductive_reasoning_correct)
  (h2 : analogical_reasoning_correct)
  (h3 : reasoning_by_plausibility_correct) : inductive_reasoning_incorrect_statement :=
sorry

end NUMINAMATH_GPT_incorrect_statement_proof_l1550_155082


namespace NUMINAMATH_GPT_tangent_line_ellipse_l1550_155006

variable {a b x x0 y y0 : ℝ}

theorem tangent_line_ellipse (h : a * x0^2 + b * y0^2 = 1) :
  a * x0 * x + b * y0 * y = 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_ellipse_l1550_155006


namespace NUMINAMATH_GPT_find_number_eq_seven_point_five_l1550_155077

theorem find_number_eq_seven_point_five :
  ∃ x : ℝ, x / 3 = x - 5 ∧ x = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_eq_seven_point_five_l1550_155077


namespace NUMINAMATH_GPT_toy_store_shelves_l1550_155037

theorem toy_store_shelves (initial_bears : ℕ) (shipment_bears : ℕ) (bears_per_shelf : ℕ)
                          (h_initial : initial_bears = 5) (h_shipment : shipment_bears = 7) 
                          (h_per_shelf : bears_per_shelf = 6) : 
                          (initial_bears + shipment_bears) / bears_per_shelf = 2 :=
by
  sorry

end NUMINAMATH_GPT_toy_store_shelves_l1550_155037


namespace NUMINAMATH_GPT_ratio_50kg_to_05tons_not_100_to_1_l1550_155004

theorem ratio_50kg_to_05tons_not_100_to_1 (weight1 : ℕ) (weight2 : ℕ) (r : ℕ) 
  (h1 : weight1 = 50) (h2 : weight2 = 500) (h3 : r = 100) : ¬ (weight1 * r = weight2) := 
by
  sorry

end NUMINAMATH_GPT_ratio_50kg_to_05tons_not_100_to_1_l1550_155004


namespace NUMINAMATH_GPT_find_R_l1550_155032

theorem find_R (R : ℝ) (h_diff : ∃ a b : ℝ, a ≠ b ∧ (a - b = 12 ∨ b - a = 12) ∧ a + b = 2 ∧ a * b = -R) : R = 35 :=
by
  obtain ⟨a, b, h_neq, h_diff_12, h_sum, h_prod⟩ := h_diff
  sorry

end NUMINAMATH_GPT_find_R_l1550_155032


namespace NUMINAMATH_GPT_smallest_lcm_of_4digit_integers_with_gcd_5_l1550_155036

theorem smallest_lcm_of_4digit_integers_with_gcd_5 :
  ∃ (a b : ℕ), 1000 ≤ a ∧ a < 10000 ∧ 1000 ≤ b ∧ b < 10000 ∧ gcd a b = 5 ∧ lcm a b = 201000 :=
by
  sorry

end NUMINAMATH_GPT_smallest_lcm_of_4digit_integers_with_gcd_5_l1550_155036


namespace NUMINAMATH_GPT_ronalds_egg_sharing_l1550_155047

theorem ronalds_egg_sharing (total_eggs : ℕ) (eggs_per_friend : ℕ) (num_friends : ℕ) 
  (h1 : total_eggs = 16) (h2 : eggs_per_friend = 2) 
  (h3 : num_friends = total_eggs / eggs_per_friend) : 
  num_friends = 8 := 
by 
  sorry

end NUMINAMATH_GPT_ronalds_egg_sharing_l1550_155047


namespace NUMINAMATH_GPT_fraction_of_sum_after_6_years_l1550_155070

-- Define the principal amount, rate, and time period as given in the conditions
def P : ℝ := 1
def R : ℝ := 0.02777777777777779
def T : ℕ := 6

-- Definition of the Simple Interest calculation
def simple_interest (P R : ℝ) (T : ℕ) : ℝ :=
  P * R * T

-- Definition of the total amount after 6 years
def total_amount (P SI : ℝ) : ℝ :=
  P + SI

-- The main theorem to prove
theorem fraction_of_sum_after_6_years :
  total_amount P (simple_interest P R T) = 1.1666666666666667 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_sum_after_6_years_l1550_155070


namespace NUMINAMATH_GPT_parametric_equations_curveC2_minimum_distance_M_to_curveC_l1550_155079

noncomputable def curveC1_param (α : ℝ) : ℝ × ℝ :=
  (Real.cos α, Real.sin α)

def scaling_transform (x y : ℝ) : ℝ × ℝ :=
  (3 * x, 2 * y)

theorem parametric_equations_curveC2 (θ : ℝ) :
  scaling_transform (Real.cos θ) (Real.sin θ) = (3 * Real.cos θ, 2 * Real.sin θ) :=
sorry

noncomputable def curveC (ρ θ : ℝ) : Prop :=
  2 * ρ * Real.sin θ + ρ * Real.cos θ = 10

noncomputable def distance_to_curveC (θ : ℝ) : ℝ :=
  abs (3 * Real.cos θ + 4 * Real.sin θ - 10) / Real.sqrt 5

theorem minimum_distance_M_to_curveC : 
  ∀ θ, distance_to_curveC θ >= Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_parametric_equations_curveC2_minimum_distance_M_to_curveC_l1550_155079


namespace NUMINAMATH_GPT_find_shaun_age_l1550_155090

def current_ages (K G S : ℕ) :=
  K + 4 = 2 * (G + 4) ∧
  S + 8 = 2 * (K + 8) ∧
  S + 12 = 3 * (G + 12)

theorem find_shaun_age (K G S : ℕ) (h : current_ages K G S) : S = 48 :=
  by
    sorry

end NUMINAMATH_GPT_find_shaun_age_l1550_155090


namespace NUMINAMATH_GPT_carpet_needed_in_sq_yards_l1550_155043

theorem carpet_needed_in_sq_yards :
  let length := 15
  let width := 10
  let area_sq_feet := length * width
  let conversion_factor := 9
  let area_sq_yards := area_sq_feet / conversion_factor
  area_sq_yards = 16.67 := by
  sorry

end NUMINAMATH_GPT_carpet_needed_in_sq_yards_l1550_155043


namespace NUMINAMATH_GPT_tangent_slope_at_one_l1550_155022

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x + Real.sqrt x

theorem tangent_slope_at_one :
  (deriv f 1) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_slope_at_one_l1550_155022


namespace NUMINAMATH_GPT_alex_initial_silk_l1550_155083

theorem alex_initial_silk (m_per_dress : ℕ) (m_per_friend : ℕ) (num_friends : ℕ) (num_dresses : ℕ) (initial_silk : ℕ) :
  m_per_dress = 5 ∧ m_per_friend = 20 ∧ num_friends = 5 ∧ num_dresses = 100 ∧ 
  (initial_silk - (num_friends * m_per_friend)) / m_per_dress * m_per_dress = num_dresses * m_per_dress → 
  initial_silk = 600 :=
by
  intros
  sorry

end NUMINAMATH_GPT_alex_initial_silk_l1550_155083


namespace NUMINAMATH_GPT_true_inverse_of_opposites_true_contrapositive_of_real_roots_l1550_155026

theorem true_inverse_of_opposites (X Y : Int) :
  (X = -Y) → (X + Y = 0) :=
by 
  sorry

theorem true_contrapositive_of_real_roots (q : Real) :
  (¬ ∃ x : Real, x^2 + 2*x + q = 0) → (q > 1) :=
by
  sorry

end NUMINAMATH_GPT_true_inverse_of_opposites_true_contrapositive_of_real_roots_l1550_155026


namespace NUMINAMATH_GPT_ratio_of_investments_l1550_155024

variable (A B C : ℝ) (k : ℝ)

-- Conditions
def investments_ratio := (6 * k + 5 * k + 4 * k = 7250) ∧ (5 * k - 6 * k = 250)

-- Theorem we need to prove
theorem ratio_of_investments (h : investments_ratio k) : (A / B = 6 / 5) ∧ (B / C = 5 / 4) := 
  sorry

end NUMINAMATH_GPT_ratio_of_investments_l1550_155024


namespace NUMINAMATH_GPT_John_Anna_total_eBooks_l1550_155013

variables (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) 

def John_bought (Anna_bought : ℕ) : ℕ := Anna_bought - 15
def John_left (Anna_bought : ℕ) (eBooks_lost_by_John : ℕ) : ℕ := John_bought Anna_bought - eBooks_lost_by_John

theorem John_Anna_total_eBooks (Anna_bought_eq_50 : Anna_bought = 50)
    (John_bought_eq_35 : John_bought Anna_bought = 35) (eBooks_lost_eq_3 : eBooks_lost_by_John = 3) :
    (Anna_bought + John_left Anna_bought eBooks_lost_by_John = 82) :=
by sorry

end NUMINAMATH_GPT_John_Anna_total_eBooks_l1550_155013


namespace NUMINAMATH_GPT_solutions_count_l1550_155033

noncomputable def number_of_solutions (a : ℝ) : ℕ :=
if a < 0 then 1
else if 0 ≤ a ∧ a < Real.exp 1 then 0
else if a = Real.exp 1 then 1
else if a > Real.exp 1 then 2
else 0

theorem solutions_count (a : ℝ) :
  (a < 0 ∧ number_of_solutions a = 1) ∨
  (0 ≤ a ∧ a < Real.exp 1 ∧ number_of_solutions a = 0) ∨
  (a = Real.exp 1 ∧ number_of_solutions a = 1) ∨
  (a > Real.exp 1 ∧ number_of_solutions a = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_solutions_count_l1550_155033


namespace NUMINAMATH_GPT_tank_fill_time_l1550_155040

-- Define the conditions
def capacity := 800
def rate_A := 40
def rate_B := 30
def rate_C := -20

def net_rate_per_cycle := rate_A + rate_B + rate_C
def cycle_duration := 3
def total_cycles := capacity / net_rate_per_cycle
def total_time := total_cycles * cycle_duration

-- The proof that tank will be full after 48 minutes
theorem tank_fill_time : total_time = 48 := by
  sorry

end NUMINAMATH_GPT_tank_fill_time_l1550_155040


namespace NUMINAMATH_GPT_kelly_spends_correct_amount_l1550_155029

noncomputable def total_cost_with_discount : ℝ :=
  let mango_cost_per_pound := (0.60 : ℝ) * 2
  let orange_cost_per_pound := (0.40 : ℝ) * 4
  let mango_total_cost := 5 * mango_cost_per_pound
  let orange_total_cost := 5 * orange_cost_per_pound
  let total_cost_without_discount := mango_total_cost + orange_total_cost
  let discount := 0.10 * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount
  total_cost_with_discount

theorem kelly_spends_correct_amount :
  total_cost_with_discount = 12.60 := by
  sorry

end NUMINAMATH_GPT_kelly_spends_correct_amount_l1550_155029


namespace NUMINAMATH_GPT_solution_range_of_a_l1550_155085

theorem solution_range_of_a (a : ℝ) (x y : ℝ) :
  3 * x + y = 1 + a → x + 3 * y = 3 → x + y < 2 → a < 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_range_of_a_l1550_155085


namespace NUMINAMATH_GPT_bread_cost_l1550_155076

theorem bread_cost
  (B : ℝ)
  (cost_peanut_butter : ℝ := 2)
  (initial_money : ℝ := 14)
  (money_leftover : ℝ := 5.25) :
  3 * B + cost_peanut_butter = (initial_money - money_leftover) → B = 2.25 :=
by
  sorry

end NUMINAMATH_GPT_bread_cost_l1550_155076


namespace NUMINAMATH_GPT_equilibrium_temperature_l1550_155005

-- Initial conditions for heat capacities and masses
variables (c_B c_W m_B m_W : ℝ) (h : c_W * m_W = 3 * c_B * m_B)

-- Initial temperatures
def T_W_initial := 100
def T_B_initial := 20
def T_f_initial := 80

-- Final equilibrium temperature after second block is added
def final_temp := 68

theorem equilibrium_temperature (t : ℝ)
  (h_first_eq : c_W * m_W * (T_W_initial - T_f_initial) = c_B * m_B * (T_f_initial - T_B_initial))
  (h_second_eq : c_W * m_W * (T_f_initial - t) + c_B * m_B * (T_f_initial - t) = c_B * m_B * (t - T_B_initial)) :
  t = final_temp :=
by 
  sorry

end NUMINAMATH_GPT_equilibrium_temperature_l1550_155005


namespace NUMINAMATH_GPT_common_fraction_difference_l1550_155060

def repeating_decimal := 23 / 99
def non_repeating_decimal := 23 / 100
def fraction_difference := 23 / 9900

theorem common_fraction_difference : repeating_decimal - non_repeating_decimal = fraction_difference := 
by
  sorry

end NUMINAMATH_GPT_common_fraction_difference_l1550_155060


namespace NUMINAMATH_GPT_sqrt_xyz_sum_l1550_155041

theorem sqrt_xyz_sum {x y z : ℝ} (h₁ : y + z = 24) (h₂ : z + x = 26) (h₃ : x + y = 28) :
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 83655 := by
  sorry

end NUMINAMATH_GPT_sqrt_xyz_sum_l1550_155041


namespace NUMINAMATH_GPT_equilateral_triangle_iff_l1550_155075

theorem equilateral_triangle_iff (a b c : ℝ) :
  a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_iff_l1550_155075


namespace NUMINAMATH_GPT_find_constant_l1550_155002

-- Define the relationship between Fahrenheit and Celsius
def temp_rel (c f k : ℝ) : Prop :=
  f = (9 / 5) * c + k

-- Temperature increases
def temp_increase (c1 c2 f1 f2 : ℝ) : Prop :=
  (f2 - f1 = 30) ∧ (c2 - c1 = 16.666666666666668)

-- Freezing point condition
def freezing_point (f : ℝ) : Prop :=
  f = 32

-- Main theorem to prove
theorem find_constant (k : ℝ) :
  ∃ (c1 c2 f1 f2: ℝ), temp_rel c1 f1 k ∧ temp_rel c2 f2 k ∧ 
  temp_increase c1 c2 f1 f2 ∧ freezing_point f1 → k = 32 :=
by sorry

end NUMINAMATH_GPT_find_constant_l1550_155002


namespace NUMINAMATH_GPT_find_original_wage_l1550_155053

theorem find_original_wage (W : ℝ) (h : 1.50 * W = 51) : W = 34 :=
sorry

end NUMINAMATH_GPT_find_original_wage_l1550_155053


namespace NUMINAMATH_GPT_max_value_of_expression_l1550_155068

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h_sum : x + y + z = 3) 
  (h_order : x ≥ y ∧ y ≥ z) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 12 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1550_155068


namespace NUMINAMATH_GPT_find_XY_sum_in_base10_l1550_155088

def base8_addition_step1 (X : ℕ) : Prop :=
  X + 5 = 9

def base8_addition_step2 (Y X : ℕ) : Prop :=
  Y + 3 = X

theorem find_XY_sum_in_base10 (X Y : ℕ) (h1 : base8_addition_step1 X) (h2 : base8_addition_step2 Y X) :
  X + Y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_XY_sum_in_base10_l1550_155088


namespace NUMINAMATH_GPT_same_color_probability_l1550_155017

theorem same_color_probability 
  (B R : ℕ)
  (hB : B = 5)
  (hR : R = 5)
  : (B + R = 10) → (1/2 * 4/9 + 1/2 * 4/9 = 4/9) := by
  intros
  sorry

end NUMINAMATH_GPT_same_color_probability_l1550_155017


namespace NUMINAMATH_GPT_solve_for_x_l1550_155074

-- Definitions based on provided conditions
variables (x : ℝ) -- defining x as a real number
def condition : Prop := 0.25 * x = 0.15 * 1600 - 15

-- The theorem stating that x equals 900 given the condition
theorem solve_for_x (h : condition x) : x = 900 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1550_155074


namespace NUMINAMATH_GPT_probability_either_A1_or_B1_not_both_is_half_l1550_155050

-- Definitions of the students
inductive Student
| A : ℕ → Student
| B : ℕ → Student
| C : ℕ → Student

-- Excellent grades students
def math_students := [Student.A 1, Student.A 2, Student.A 3]
def physics_students := [Student.B 1, Student.B 2]
def chemistry_students := [Student.C 1, Student.C 2]

-- Total number of ways to select one student from each category
def total_ways : ℕ := 3 * 2 * 2

-- Number of ways either A_1 or B_1 is selected but not both
def special_ways : ℕ := 1 * 1 * 2 + 2 * 1 * 2

-- Probability calculation
def probability := (special_ways : ℚ) / total_ways

-- Theorem to be proven
theorem probability_either_A1_or_B1_not_both_is_half :
  probability = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_probability_either_A1_or_B1_not_both_is_half_l1550_155050


namespace NUMINAMATH_GPT_rectangle_percentage_increase_l1550_155007

theorem rectangle_percentage_increase (L W : ℝ) (P : ℝ) (h : (1 + P / 100) ^ 2 = 1.44) : P = 20 :=
by {
  -- skipped proof
  sorry
}

end NUMINAMATH_GPT_rectangle_percentage_increase_l1550_155007


namespace NUMINAMATH_GPT_hyogeun_weight_l1550_155035

noncomputable def weights_are_correct : Prop :=
  ∃ H S G : ℝ, 
    H + S + G = 106.6 ∧
    G = S - 7.7 ∧
    S = H - 4.8 ∧
    H = 41.3

theorem hyogeun_weight : weights_are_correct :=
by
  sorry

end NUMINAMATH_GPT_hyogeun_weight_l1550_155035


namespace NUMINAMATH_GPT_value_of_expression_l1550_155020

variable (x y : ℝ)

theorem value_of_expression 
  (h1 : x + Real.sqrt (x * y) + y = 9)
  (h2 : x^2 + x * y + y^2 = 27) :
  x - Real.sqrt (x * y) + y = 3 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1550_155020


namespace NUMINAMATH_GPT_line_equation_l1550_155094

/-
Given points M(2, 3) and N(4, -5), and a line l passes through the 
point P(1, 2). Prove that the line l has equal distances from points 
M and N if and only if its equation is either 4x + y - 6 = 0 or 
3x + 2y - 7 = 0.
-/

theorem line_equation (M N P : ℝ × ℝ)
(hM : M = (2, 3))
(hN : N = (4, -5))
(hP : P = (1, 2))
(l : ℝ → ℝ → Prop)
(h_l : ∀ x y, l x y ↔ (4 * x + y - 6 = 0 ∨ 3 * x + 2 * y - 7 = 0))
: ∀ (dM dN : ℝ), 
(∀ x y , l x y → (x = 1) → (y = 2) ∧ (|M.1 - x| + |M.2 - y| = |N.1 - x| + |N.2 - y|)) :=
sorry

end NUMINAMATH_GPT_line_equation_l1550_155094


namespace NUMINAMATH_GPT_gcd_exponential_identity_l1550_155087

theorem gcd_exponential_identity (a b : ℕ) :
  Nat.gcd (2^a - 1) (2^b - 1) = 2^(Nat.gcd a b) - 1 := sorry

end NUMINAMATH_GPT_gcd_exponential_identity_l1550_155087


namespace NUMINAMATH_GPT_sasha_remainder_l1550_155018

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) 
  (h3 : a + d = 20) (h_b_range : 0 ≤ b ∧ b ≤ 101) (h_d_range : 0 ≤ d ∧ d ≤ 102) : b = 20 := 
sorry

end NUMINAMATH_GPT_sasha_remainder_l1550_155018


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1550_155027

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a <= 10) (h2 : 10 <= c)
  (h3 : (a + 10 + c) / 3 = a + 8)
  (h4 : (a + 10 + c) / 3 = c - 20) :
  a + 10 + c = 66 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1550_155027


namespace NUMINAMATH_GPT_part1_part2_l1550_155081

def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 2| + x
def g (x : ℝ) : ℝ := |x - 2| - |2 * x - 3| + x

theorem part1 (a : ℝ) : (∀ x, f a x ≤ f a 2) ↔ a ≤ -1 :=
by sorry

theorem part2 (x : ℝ) : f 1 x < |2 * x - 3| ↔ x > 0.5 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1550_155081


namespace NUMINAMATH_GPT_solve_x_for_collinear_and_same_direction_l1550_155071

-- Define vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (-1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (-x, 2)

-- Define the conditions for collinearity and same direction
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k • b.1, k • b.2)

def same_direction (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (k • b.1, k • b.2)

theorem solve_x_for_collinear_and_same_direction
  (x : ℝ)
  (h_collinear : collinear (vector_a x) (vector_b x))
  (h_same_direction : same_direction (vector_a x) (vector_b x)) :
  x = Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_solve_x_for_collinear_and_same_direction_l1550_155071


namespace NUMINAMATH_GPT_problem_1_problem_2_l1550_155056

def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

theorem problem_1 (x : ℝ) : f x ≥ 2 ↔ (x ≤ -7 ∨ x ≥ 5 / 3) :=
sorry

theorem problem_2 : ∃ x : ℝ, f x = -9 / 2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1550_155056


namespace NUMINAMATH_GPT_solution_exists_unique_l1550_155031

variable (a b c : ℝ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)

theorem solution_exists_unique (x y z : ℝ)
  (hx : x = (b + c) / 2)
  (hy : y = (c + a) / 2)
  (hz : z = (a + b) / 2)
  (h1 : x + y + z = a + b + c)
  (h2 : 4 * x * y * z - (a^2 * x + b^2 * y + c^2 * z) = a * b * c) :
  x = (b + c) / 2 ∧ y = (c + a) / 2 ∧ z = (a + b) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_unique_l1550_155031


namespace NUMINAMATH_GPT_multiply_polynomials_l1550_155096

variable {x y z : ℝ}

theorem multiply_polynomials :
  (3 * x^4 - 4 * y^3 - 6 * z^2) * (9 * x^8 + 16 * y^6 + 36 * z^4 + 12 * x^4 * y^3 + 18 * x^4 * z^2 + 24 * y^3 * z^2)
  = 27 * x^12 - 64 * y^9 - 216 * z^6 - 216 * x^4 * y^3 * z^2 := by {
  sorry
}

end NUMINAMATH_GPT_multiply_polynomials_l1550_155096


namespace NUMINAMATH_GPT_logs_left_after_3_hours_l1550_155057

theorem logs_left_after_3_hours : 
  ∀ (initial_logs : ℕ) (burn_rate : ℕ) (add_rate : ℕ) (time : ℕ),
  initial_logs = 6 →
  burn_rate = 3 →
  add_rate = 2 →
  time = 3 →
  initial_logs + (add_rate * time) - (burn_rate * time) = 3 := 
by
  intros initial_logs burn_rate add_rate time h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_logs_left_after_3_hours_l1550_155057


namespace NUMINAMATH_GPT_salary_of_A_l1550_155078

-- Given:
-- A + B = 6000
-- A's savings = 0.05A
-- B's savings = 0.15B
-- A's savings = B's savings

theorem salary_of_A (A B : ℝ) (h1 : A + B = 6000) (h2 : 0.05 * A = 0.15 * B) :
  A = 4500 :=
sorry

end NUMINAMATH_GPT_salary_of_A_l1550_155078


namespace NUMINAMATH_GPT_solve_for_x_opposites_l1550_155063

theorem solve_for_x_opposites (x : ℝ) (h : -2 * x = -(3 * x - 1)) : x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_opposites_l1550_155063


namespace NUMINAMATH_GPT_at_least_one_zero_l1550_155052

theorem at_least_one_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) : 
  a = 0 ∨ b = 0 ∨ c = 0 := 
sorry

end NUMINAMATH_GPT_at_least_one_zero_l1550_155052


namespace NUMINAMATH_GPT_radius_inner_circle_l1550_155038

theorem radius_inner_circle (s : ℝ) (n : ℕ) (d : ℝ) (r : ℝ) :
  s = 4 ∧ n = 16 ∧ d = s / 4 ∧ ∀ k, k = d / 2 → r = (Real.sqrt (s^2 / 4 + k^2) - k) / 2 
  → r = Real.sqrt 4.25 / 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_inner_circle_l1550_155038


namespace NUMINAMATH_GPT_factorize_expression_l1550_155015

theorem factorize_expression (a b x y : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1550_155015


namespace NUMINAMATH_GPT_max_consecutive_integers_sum_lt_1000_l1550_155014

theorem max_consecutive_integers_sum_lt_1000
  (n : ℕ)
  (h : (n * (n + 1)) / 2 < 1000) : n ≤ 44 :=
by
  sorry

end NUMINAMATH_GPT_max_consecutive_integers_sum_lt_1000_l1550_155014


namespace NUMINAMATH_GPT_solve_for_x_l1550_155064

theorem solve_for_x (x : ℝ) (hx_pos : x > 0) (h_eq : 3 * x^2 + 13 * x - 10 = 0) : x = 2 / 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1550_155064


namespace NUMINAMATH_GPT_analogical_reasoning_correct_l1550_155049

variable (a b c : Real)

theorem analogical_reasoning_correct (h : c ≠ 0) (h_eq : (a + b) * c = a * c + b * c) : 
  (a + b) / c = a / c + b / c :=
  sorry

end NUMINAMATH_GPT_analogical_reasoning_correct_l1550_155049


namespace NUMINAMATH_GPT_a2020_lt_inv_2020_l1550_155080

theorem a2020_lt_inv_2020 (a : ℕ → ℝ) (ha0 : a 0 > 0) 
    (h_rec : ∀ n, a (n + 1) = a n / Real.sqrt (1 + 2020 * a n ^ 2)) :
    a 2020 < 1 / 2020 :=
sorry

end NUMINAMATH_GPT_a2020_lt_inv_2020_l1550_155080


namespace NUMINAMATH_GPT_proportion_first_number_l1550_155025

theorem proportion_first_number (x : ℝ) (h : x / 5 = 0.96 / 8) : x = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_proportion_first_number_l1550_155025


namespace NUMINAMATH_GPT_find_sum_abc_l1550_155010

-- Define the real numbers a, b, c
variables {a b c : ℝ}

-- Define the conditions that a, b, c are positive reals.
axiom ha_pos : 0 < a
axiom hb_pos : 0 < b
axiom hc_pos : 0 < c

-- Define the condition that a^2 + b^2 + c^2 = 989
axiom habc_sq : a^2 + b^2 + c^2 = 989

-- Define the condition that (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013
axiom habc_sq_sum : (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013

-- The proposition to be proven
theorem find_sum_abc : a + b + c = 32 :=
by
  -- ...(proof goes here)
  sorry

end NUMINAMATH_GPT_find_sum_abc_l1550_155010


namespace NUMINAMATH_GPT_john_money_left_l1550_155055

def cost_of_drink (q : ℝ) : ℝ := q
def cost_of_small_pizza (q : ℝ) : ℝ := cost_of_drink q
def cost_of_large_pizza (q : ℝ) : ℝ := 4 * cost_of_drink q
def total_cost (q : ℝ) : ℝ := 2 * cost_of_drink q + 2 * cost_of_small_pizza q + cost_of_large_pizza q
def initial_money : ℝ := 50
def remaining_money (q : ℝ) : ℝ := initial_money - total_cost q

theorem john_money_left (q : ℝ) : remaining_money q = 50 - 8 * q :=
by
  sorry

end NUMINAMATH_GPT_john_money_left_l1550_155055


namespace NUMINAMATH_GPT_exp_mono_increasing_of_gt_l1550_155034

variable {a b : ℝ}

theorem exp_mono_increasing_of_gt (h : a > b) : (2 : ℝ) ^ a > (2 : ℝ) ^ b :=
by sorry

end NUMINAMATH_GPT_exp_mono_increasing_of_gt_l1550_155034


namespace NUMINAMATH_GPT_largest_quantity_l1550_155023

noncomputable def D := (2007 / 2006) + (2007 / 2008)
noncomputable def E := (2007 / 2008) + (2009 / 2008)
noncomputable def F := (2008 / 2007) + (2008 / 2009)

theorem largest_quantity : D > E ∧ D > F :=
by { sorry }

end NUMINAMATH_GPT_largest_quantity_l1550_155023


namespace NUMINAMATH_GPT_juliet_older_than_maggie_l1550_155048

-- Definitions from the given conditions
def Juliet_age : ℕ := 10
def Ralph_age (J : ℕ) : ℕ := J + 2
def Maggie_age (R : ℕ) : ℕ := 19 - R

-- Theorem statement
theorem juliet_older_than_maggie :
  Juliet_age - Maggie_age (Ralph_age Juliet_age) = 3 :=
by
  sorry

end NUMINAMATH_GPT_juliet_older_than_maggie_l1550_155048


namespace NUMINAMATH_GPT_expansion_terms_count_l1550_155044

-- Define the number of terms in the first polynomial
def first_polynomial_terms : ℕ := 3

-- Define the number of terms in the second polynomial
def second_polynomial_terms : ℕ := 4

-- Prove that the number of terms in the expansion is 12
theorem expansion_terms_count : first_polynomial_terms * second_polynomial_terms = 12 :=
by
  sorry

end NUMINAMATH_GPT_expansion_terms_count_l1550_155044


namespace NUMINAMATH_GPT_geometric_progression_ratio_l1550_155086

theorem geometric_progression_ratio (r : ℕ) (h : 4 + 4 * r + 4 * r^2 + 4 * r^3 = 60) : r = 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_ratio_l1550_155086


namespace NUMINAMATH_GPT_gcd_lcm_product_l1550_155066

theorem gcd_lcm_product (a b : ℕ) (ha : a = 18) (hb : b = 42) :
  Nat.gcd a b * Nat.lcm a b = 756 :=
by
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l1550_155066


namespace NUMINAMATH_GPT_min_tiles_needed_l1550_155073

theorem min_tiles_needed : 
  ∀ (tile_length tile_width region_length region_width: ℕ),
  tile_length = 5 → 
  tile_width = 6 → 
  region_length = 3 * 12 → 
  region_width = 4 * 12 → 
  (region_length * region_width) / (tile_length * tile_width) ≤ 58 :=
by
  intros tile_length tile_width region_length region_width h_tile_length h_tile_width h_region_length h_region_width
  sorry

end NUMINAMATH_GPT_min_tiles_needed_l1550_155073


namespace NUMINAMATH_GPT_feta_price_calculation_l1550_155091

noncomputable def feta_price_per_pound (sandwiches_price : ℝ) (sandwiches_count : ℕ) 
  (salami_price : ℝ) (brie_factor : ℝ) (olive_price_per_pound : ℝ) 
  (olive_weight : ℝ) (bread_price : ℝ) (total_spent : ℝ)
  (feta_weight : ℝ) :=
  (total_spent - (sandwiches_count * sandwiches_price + salami_price + brie_factor * salami_price + olive_price_per_pound * olive_weight + bread_price)) / feta_weight

theorem feta_price_calculation : 
  feta_price_per_pound 7.75 2 4.00 3 10.00 0.25 2.00 40.00 0.5 = 8.00 := 
by
  sorry

end NUMINAMATH_GPT_feta_price_calculation_l1550_155091


namespace NUMINAMATH_GPT_intersection_point_l1550_155093

theorem intersection_point :
  ∃ (x y : ℝ), (2 * x + 3 * y + 8 = 0) ∧ (x - y - 1 = 0) ∧ (x = -1) ∧ (y = -2) := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_l1550_155093


namespace NUMINAMATH_GPT_calculate_sum_l1550_155019

theorem calculate_sum : (-2) + 1 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_calculate_sum_l1550_155019
