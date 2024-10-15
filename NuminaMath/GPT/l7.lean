import Mathlib

namespace NUMINAMATH_GPT_scarves_per_box_l7_751

theorem scarves_per_box (S : ℕ) 
  (boxes : ℕ)
  (mittens_per_box : ℕ)
  (total_clothes : ℕ)
  (h1 : boxes = 4)
  (h2 : mittens_per_box = 6)
  (h3 : total_clothes = 32)
  (total_mittens := boxes * mittens_per_box)
  (total_scarves := total_clothes - total_mittens) :
  total_scarves / boxes = 2 :=
by
  sorry

end NUMINAMATH_GPT_scarves_per_box_l7_751


namespace NUMINAMATH_GPT_No_of_boxes_in_case_l7_772

-- Define the conditions
def George_has_total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def George_has_boxes : ℕ := George_has_total_blocks / blocks_per_box

-- The theorem to prove
theorem No_of_boxes_in_case : George_has_boxes = 2 :=
by
  sorry

end NUMINAMATH_GPT_No_of_boxes_in_case_l7_772


namespace NUMINAMATH_GPT_accessory_factory_growth_l7_717

theorem accessory_factory_growth (x : ℝ) :
  600 + 600 * (1 + x) + 600 * (1 + x) ^ 2 = 2180 :=
sorry

end NUMINAMATH_GPT_accessory_factory_growth_l7_717


namespace NUMINAMATH_GPT_pen_sales_average_l7_762

theorem pen_sales_average (d : ℕ) (h1 : 96 + 44 * d > 0) (h2 : (96 + 44 * d) / (d + 1) = 48) : d = 12 :=
by
  sorry

end NUMINAMATH_GPT_pen_sales_average_l7_762


namespace NUMINAMATH_GPT_smallest_debt_exists_l7_721

theorem smallest_debt_exists :
  ∃ (p g : ℤ), 50 = 200 * p + 150 * g := by
  sorry

end NUMINAMATH_GPT_smallest_debt_exists_l7_721


namespace NUMINAMATH_GPT_first_woman_hours_l7_706

-- Definitions and conditions
variables (W k y t η : ℝ)
variables (work_rate : k * y * 45 = W)
variables (total_work : W = k * (t * ((y-1) * y) / 2 + y * η))
variables (first_vs_last : (y-1) * t + η = 5 * η)

-- The goal to prove
theorem first_woman_hours :
  (y - 1) * t + η = 75 := 
by
  sorry

end NUMINAMATH_GPT_first_woman_hours_l7_706


namespace NUMINAMATH_GPT_find_range_a_l7_755

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1) * x + (a - 2)

theorem find_range_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ x > 1 ∧ y < 1 ) :
  -2 < a ∧ a < 1 := sorry

end NUMINAMATH_GPT_find_range_a_l7_755


namespace NUMINAMATH_GPT_find_possible_values_l7_738
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def satisfies_conditions (a bc de fg : ℕ) : Prop :=
  (a % 2 = 0) ∧ (is_prime bc) ∧ (de % 5 = 0) ∧ (fg % 3 = 0) ∧
  (fg - de = de - bc) ∧ (de - bc = bc - a)

theorem find_possible_values :
  ∃ (debc1 debc2 : ℕ),
    (satisfies_conditions 6 (debc1 % 100) ((debc1 / 100) % 100) ((debc1 / 10000) % 100)) ∧
    (satisfies_conditions 6 (debc2 % 100) ((debc2 / 100) % 100) ((debc2 / 10000) % 100)) ∧
    (debc1 = 2013 ∨ debc1 = 4023) ∧
    (debc2 = 2013 ∨ debc2 = 4023) :=
  sorry

end NUMINAMATH_GPT_find_possible_values_l7_738


namespace NUMINAMATH_GPT_speed_difference_l7_773

theorem speed_difference (distance : ℕ) (time_jordan time_alex : ℕ) (h_distance : distance = 12) (h_time_jordan : time_jordan = 10) (h_time_alex : time_alex = 15) :
  (distance / (time_jordan / 60) - distance / (time_alex / 60) = 24) := by
  -- Lean code to correctly parse and understand the natural numbers, division, and maintain the theorem structure.
  sorry

end NUMINAMATH_GPT_speed_difference_l7_773


namespace NUMINAMATH_GPT_parabola_properties_l7_771

-- Define the parabola function as y = x^2 + px + q
def parabola (p q : ℝ) (x : ℝ) : ℝ := x^2 + p * x + q

-- Prove the properties of parabolas for varying p and q.
theorem parabola_properties (p q p' q' : ℝ) :
  (∀ x : ℝ, parabola p q x = x^2 + p * x + q) ∧
  (∀ x : ℝ, parabola p' q' x = x^2 + p' * x + q') →
  (∀ x : ℝ, ( ∃ k h : ℝ, parabola p q x = (x + h)^2 + k ) ∧ 
               ( ∃ k' h' : ℝ, parabola p' q' x = (x + h')^2 + k' ) ) ∧
  (∀ x : ℝ, h = -p / 2 ∧ k = q - p^2 / 4 ) ∧
  (∀ x : ℝ, h' = -p' / 2 ∧ k' = q' - p'^2 / 4 ) ∧
  (∀ x : ℝ, (h, k) ≠ (h', k') → parabola p q x ≠ parabola p' q' x) ∧
  (∀ x : ℝ, h = h' ∧ k = k' → parabola p q x = parabola p' q' x) :=
by
  sorry

end NUMINAMATH_GPT_parabola_properties_l7_771


namespace NUMINAMATH_GPT_find_f_29_l7_790

theorem find_f_29 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 3) = (x - 3) * (x + 4)) : f 29 = 170 := 
by
  sorry

end NUMINAMATH_GPT_find_f_29_l7_790


namespace NUMINAMATH_GPT_parabola_vertex_l7_703

theorem parabola_vertex :
  (∃ h k : ℝ, ∀ x : ℝ, (y : ℝ) = (x - 2)^2 + 5 ∧ h = 2 ∧ k = 5) :=
sorry

end NUMINAMATH_GPT_parabola_vertex_l7_703


namespace NUMINAMATH_GPT_Maggie_bought_one_fish_book_l7_741

-- Defining the variables and constants
def books_about_plants := 9
def science_magazines := 10
def price_book := 15
def price_magazine := 2
def total_amount_spent := 170
def cost_books_about_plants := books_about_plants * price_book
def cost_science_magazines := science_magazines * price_magazine
def cost_books_about_fish := total_amount_spent - (cost_books_about_plants + cost_science_magazines)
def books_about_fish := cost_books_about_fish / price_book

-- Theorem statement
theorem Maggie_bought_one_fish_book : books_about_fish = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Maggie_bought_one_fish_book_l7_741


namespace NUMINAMATH_GPT_multiplicative_inverse_sum_is_zero_l7_761

theorem multiplicative_inverse_sum_is_zero (a b : ℝ) (h : a * b = 1) :
  a^(2015) * b^(2016) + a^(2016) * b^(2017) + a^(2017) * b^(2016) + a^(2016) * b^(2015) = 0 :=
sorry

end NUMINAMATH_GPT_multiplicative_inverse_sum_is_zero_l7_761


namespace NUMINAMATH_GPT_solve_fraction_eq_l7_756

theorem solve_fraction_eq (x : ℝ) 
  (h₁ : x ≠ -9) 
  (h₂ : x ≠ -7) 
  (h₃ : x ≠ -10) 
  (h₄ : x ≠ -6) 
  (h₅ : 1 / (x + 9) + 1 / (x + 7) = 1 / (x + 10) + 1 / (x + 6)) : 
  x = -8 := 
sorry

end NUMINAMATH_GPT_solve_fraction_eq_l7_756


namespace NUMINAMATH_GPT_riley_mistakes_l7_777

theorem riley_mistakes :
  ∃ R O : ℕ, R + O = 17 ∧ O = 35 - ((35 - R) / 2 + 5) ∧ R = 3 := by
  sorry

end NUMINAMATH_GPT_riley_mistakes_l7_777


namespace NUMINAMATH_GPT_distance_between_trees_l7_724

-- The conditions given
def trees_on_yard := 26
def yard_length := 500
def trees_at_ends := true

-- Theorem stating the proof
theorem distance_between_trees (h1 : trees_on_yard = 26) 
                               (h2 : yard_length = 500) 
                               (h3 : trees_at_ends = true) : 
  500 / (26 - 1) = 20 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l7_724


namespace NUMINAMATH_GPT_range_of_k_l7_747

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k^2 - 1) * x^2 - (k + 1) * x + 1 > 0) ↔ (1 ≤ k ∧ k ≤ 5 / 3) := 
sorry

end NUMINAMATH_GPT_range_of_k_l7_747


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l7_737

theorem arithmetic_sequence_fifth_term :
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  a5 = 19 :=
by
  let a1 := 3
  let d := 4
  let a5 := a1 + (5 - 1) * d
  show a5 = 19
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l7_737


namespace NUMINAMATH_GPT_parallel_lines_condition_l7_735

theorem parallel_lines_condition (a : ℝ) : 
  (∃ l1 l2 : ℝ → ℝ, 
    (∀ x y : ℝ, l1 x + a * y + 6 = 0) ∧ 
    (∀ x y : ℝ, (a - 2) * x + 3 * y + 2 * a = 0) ∧
    l1 = l2 ↔ a = 3) :=
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l7_735


namespace NUMINAMATH_GPT_union_complement_l7_763

open Set

variable (U A B : Set ℕ)
variable (u_spec : U = {1, 2, 3, 4, 5})
variable (a_spec : A = {1, 2, 3})
variable (b_spec : B = {2, 4})

theorem union_complement (U A B : Set ℕ)
  (u_spec : U = {1, 2, 3, 4, 5})
  (a_spec : A = {1, 2, 3})
  (b_spec : B = {2, 4}) :
  A ∪ (U \ B) = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_GPT_union_complement_l7_763


namespace NUMINAMATH_GPT_measureable_weights_count_l7_753

theorem measureable_weights_count (a b c : ℕ) (ha : a = 1) (hb : b = 3) (hc : c = 9) :
  ∃ s : Finset ℕ, s.card = 13 ∧ ∀ x ∈ s, x ≥ 1 ∧ x ≤ 13 := 
sorry

end NUMINAMATH_GPT_measureable_weights_count_l7_753


namespace NUMINAMATH_GPT_parabola_equation_l7_778

theorem parabola_equation (A B : ℝ × ℝ) (x₁ x₂ y₁ y₂ p : ℝ) :
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  x₁ + x₂ = (p + 8) / 2 →
  x₁ * x₂ = 4 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 45 →
  (y₁ = 2 * x₁ - 4) →
  (y₂ = 2 * x₂ - 4) →
  ((y₁^2 = 2 * p * x₁) ∧ (y₂^2 = 2 * p * x₂)) →
  (y₁^2 = 4 * x₁ ∨ y₂^2 = -36 * x₂) := 
by {
  sorry
}

end NUMINAMATH_GPT_parabola_equation_l7_778


namespace NUMINAMATH_GPT_compute_vector_expression_l7_779

theorem compute_vector_expression :
  4 • (⟨3, -5⟩ : ℝ × ℝ) - 3 • (⟨2, -6⟩ : ℝ × ℝ) + 2 • (⟨0, 3⟩ : ℝ × ℝ) = (⟨6, 4⟩ : ℝ × ℝ) := 
sorry

end NUMINAMATH_GPT_compute_vector_expression_l7_779


namespace NUMINAMATH_GPT_real_solutions_eq_l7_702

theorem real_solutions_eq :
  ∀ x : ℝ, (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12) → (x = 10 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_eq_l7_702


namespace NUMINAMATH_GPT_num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l7_759

theorem num_of_tenths_in_1_9 : (1.9 / 0.1) = 19 :=
by sorry

theorem num_of_hundredths_in_0_8 : (0.8 / 0.01) = 80 :=
by sorry

end NUMINAMATH_GPT_num_of_tenths_in_1_9_num_of_hundredths_in_0_8_l7_759


namespace NUMINAMATH_GPT_person_age_l7_716

-- Define the conditions
def current_age : ℕ := 18

-- Define the equation based on the person's statement
def age_equation (A Y : ℕ) : Prop := 3 * (A + 3) - 3 * (A - Y) = A

-- Statement to be proven
theorem person_age (Y : ℕ) : 
  age_equation current_age Y → Y = 3 := 
by 
  sorry

end NUMINAMATH_GPT_person_age_l7_716


namespace NUMINAMATH_GPT_factorization_of_expression_l7_767

noncomputable def factorized_form (x : ℝ) : ℝ :=
  (x + 5 / 2 + Real.sqrt 13 / 2) * (x + 5 / 2 - Real.sqrt 13 / 2)

theorem factorization_of_expression (x : ℝ) :
  x^2 - 5 * x + 3 = factorized_form x :=
by
  sorry

end NUMINAMATH_GPT_factorization_of_expression_l7_767


namespace NUMINAMATH_GPT_sequence_sum_eq_ten_implies_n_eq_120_l7_745

theorem sequence_sum_eq_ten_implies_n_eq_120 :
  (∀ (a : ℕ → ℝ), (∀ n, a n = 1 / (Real.sqrt n + Real.sqrt (n + 1))) →
    (∃ n, (Finset.sum (Finset.range n) a) = 10 → n = 120)) :=
by
  intro a h
  use 120
  intro h_sum
  sorry

end NUMINAMATH_GPT_sequence_sum_eq_ten_implies_n_eq_120_l7_745


namespace NUMINAMATH_GPT_trigonometric_identity_l7_752

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  7 * (Real.sin α)^2 + 3 * (Real.cos α)^2 = 31 / 5 := by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l7_752


namespace NUMINAMATH_GPT_prob_kong_meng_is_one_sixth_l7_704

variable (bag : List String := ["孔", "孟", "之", "乡"])
variable (draws : List String := [])
def total_events : ℕ := 4 * 3
def favorable_events : ℕ := 2
def probability_kong_meng : ℚ := favorable_events / total_events

theorem prob_kong_meng_is_one_sixth :
  (probability_kong_meng = 1 / 6) :=
by
  sorry

end NUMINAMATH_GPT_prob_kong_meng_is_one_sixth_l7_704


namespace NUMINAMATH_GPT_sheila_hourly_wage_l7_748

def weekly_working_hours : Nat :=
  (8 * 3) + (6 * 2)

def weekly_earnings : Nat :=
  468

def hourly_wage : Nat :=
  weekly_earnings / weekly_working_hours

theorem sheila_hourly_wage : hourly_wage = 13 :=
by
  sorry

end NUMINAMATH_GPT_sheila_hourly_wage_l7_748


namespace NUMINAMATH_GPT_negation_of_forall_ge_2_l7_743

theorem negation_of_forall_ge_2 :
  (¬ ∀ x : ℝ, x ≥ 2) = (∃ x₀ : ℝ, x₀ < 2) :=
sorry

end NUMINAMATH_GPT_negation_of_forall_ge_2_l7_743


namespace NUMINAMATH_GPT_power_function_point_l7_799

theorem power_function_point (m n: ℝ) (h: (m - 1) * m^n = 8) : n^(-m) = 1/9 := 
  sorry

end NUMINAMATH_GPT_power_function_point_l7_799


namespace NUMINAMATH_GPT_tan_pi_over_12_eq_l7_791

theorem tan_pi_over_12_eq : Real.tan (Real.pi / 12) = 2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_over_12_eq_l7_791


namespace NUMINAMATH_GPT_bikes_in_parking_lot_l7_776

theorem bikes_in_parking_lot (C : ℕ) (Total_Wheels : ℕ) (Wheels_per_car : ℕ) (Wheels_per_bike : ℕ) (h1 : C = 14) (h2 : Total_Wheels = 76) (h3 : Wheels_per_car = 4) (h4 : Wheels_per_bike = 2) : 
  ∃ B : ℕ, 4 * C + 2 * B = Total_Wheels ∧ B = 10 :=
by
  sorry

end NUMINAMATH_GPT_bikes_in_parking_lot_l7_776


namespace NUMINAMATH_GPT_cost_of_camel_is_6000_l7_780

noncomputable def cost_of_camel : ℕ := 6000

variables (C H O E : ℕ)
variables (cost_of_camel_rs cost_of_horses cost_of_oxen cost_of_elephants : ℕ)

-- Conditions
axiom cond1 : 10 * C = 24 * H
axiom cond2 : 16 * H = 4 * O
axiom cond3 : 6 * O = 4 * E
axiom cond4 : 10 * E = 150000

theorem cost_of_camel_is_6000
    (cond1 : 10 * C = 24 * H)
    (cond2 : 16 * H = 4 * O)
    (cond3 : 6 * O = 4 * E)
    (cond4 : 10 * E = 150000) :
  cost_of_camel = 6000 := 
sorry

end NUMINAMATH_GPT_cost_of_camel_is_6000_l7_780


namespace NUMINAMATH_GPT_positional_relationship_perpendicular_l7_718

theorem positional_relationship_perpendicular 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h : b * Real.sin A - a * Real.sin B = 0) :
  (∀ x y : ℝ, (x * Real.sin A + a * y + c = 0) ↔ (b * x - y * Real.sin B + Real.sin C = 0)) :=
sorry

end NUMINAMATH_GPT_positional_relationship_perpendicular_l7_718


namespace NUMINAMATH_GPT_at_least_two_inequalities_hold_l7_708

variable {a b c : ℝ}

theorem at_least_two_inequalities_hold (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c ≥ a * b * c) :
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨
  (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨
  (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) :=
  sorry

end NUMINAMATH_GPT_at_least_two_inequalities_hold_l7_708


namespace NUMINAMATH_GPT_nth_equation_l7_796

theorem nth_equation (n : ℕ) (h : 0 < n) : (- (n : ℤ)) * (n : ℝ) / (n + 1) = - (n : ℤ) + (n : ℝ) / (n + 1) :=
sorry

end NUMINAMATH_GPT_nth_equation_l7_796


namespace NUMINAMATH_GPT_CE_length_l7_782

theorem CE_length (AF ED AE area : ℝ) (hAF : AF = 30) (hED : ED = 50) (hAE : AE = 120) (h_area : area = 7200) : 
  ∃ CE : ℝ, CE = 138 :=
by
  -- omitted proof steps
  sorry

end NUMINAMATH_GPT_CE_length_l7_782


namespace NUMINAMATH_GPT_simplify_neg_neg_l7_736

theorem simplify_neg_neg (a b : ℝ) : -(-a - b) = a + b :=
sorry

end NUMINAMATH_GPT_simplify_neg_neg_l7_736


namespace NUMINAMATH_GPT_inequality_l7_781

theorem inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 / (a^3 + b^3 + a * b * c) + 1 / (b^3 + c^3 + a * b * c) + 1 / (c^3 + a^3 + a * b * c) ≤ 1 / (a * b * c) :=
sorry

end NUMINAMATH_GPT_inequality_l7_781


namespace NUMINAMATH_GPT_relationship_between_A_and_B_l7_758

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 + 2 * x = 0}

theorem relationship_between_A_and_B : B ⊆ A :=
sorry

end NUMINAMATH_GPT_relationship_between_A_and_B_l7_758


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l7_714

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l7_714


namespace NUMINAMATH_GPT_total_goals_other_members_l7_764

theorem total_goals_other_members (x y : ℕ) (h1 : y = (7 * x) / 15 - 18)
  (h2 : 1 / 3 * x + 1 / 5 * x + 18 + y = x)
  (h3 : ∀ n, 0 ≤ n ∧ n ≤ 3 → ¬(n * 8 > y))
  : y = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_goals_other_members_l7_764


namespace NUMINAMATH_GPT_yoongi_rank_l7_715

def namjoon_rank : ℕ := 2
def yoongi_offset : ℕ := 10

theorem yoongi_rank : namjoon_rank + yoongi_offset = 12 := 
by
  sorry

end NUMINAMATH_GPT_yoongi_rank_l7_715


namespace NUMINAMATH_GPT_minnie_penny_time_difference_l7_784

noncomputable def minnie_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def penny_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def break_time (minutes: ℝ) := minutes / 60

noncomputable def minnie_total_time :=
  minnie_time_uphill 12 6 + minnie_time_downhill 18 25 + minnie_time_flat 25 18

noncomputable def penny_total_time :=
  penny_time_flat 25 25 + penny_time_downhill 12 35 + 
  penny_time_uphill 18 12 + break_time 10

noncomputable def time_difference := (minnie_total_time - penny_total_time) * 60

theorem minnie_penny_time_difference :
  time_difference = 66 := by
  sorry

end NUMINAMATH_GPT_minnie_penny_time_difference_l7_784


namespace NUMINAMATH_GPT_total_math_and_biology_homework_l7_739

-- Definitions
def math_homework_pages : ℕ := 8
def biology_homework_pages : ℕ := 3

-- Theorem stating the problem to prove
theorem total_math_and_biology_homework :
  math_homework_pages + biology_homework_pages = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_math_and_biology_homework_l7_739


namespace NUMINAMATH_GPT_complex_square_l7_730

theorem complex_square (i : ℂ) (hi : i^2 = -1) : (1 + i)^2 = 2 * i :=
by
  sorry

end NUMINAMATH_GPT_complex_square_l7_730


namespace NUMINAMATH_GPT_discount_problem_l7_720

variable (x : ℝ)

theorem discount_problem :
  (400 * (1 - x)^2 = 225) :=
sorry

end NUMINAMATH_GPT_discount_problem_l7_720


namespace NUMINAMATH_GPT_ellipse_equation_l7_728

theorem ellipse_equation (c a b : ℝ)
  (foci1 foci2 : ℝ × ℝ) 
  (h_foci1 : foci1 = (-1, 0)) 
  (h_foci2 : foci2 = (1, 0)) 
  (h_c : c = 1) 
  (h_major_axis : 2 * a = 10) 
  (h_b_sq : b^2 = a^2 - c^2) :
  (∀ x y : ℝ, (x^2 / 25 + y^2 / 24 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l7_728


namespace NUMINAMATH_GPT_sufficient_condition_increasing_l7_757

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 1

theorem sufficient_condition_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 < x → x < y → (f x a ≤ f y a)) → a = -1 := sorry

end NUMINAMATH_GPT_sufficient_condition_increasing_l7_757


namespace NUMINAMATH_GPT_fraction_equality_l7_749

-- Defining the main problem statement
theorem fraction_equality (x y z : ℚ) (k : ℚ) 
  (h1 : x = 3 * k) (h2 : y = 5 * k) (h3 : z = 7 * k) :
  (y + z) / (3 * x - y) = 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l7_749


namespace NUMINAMATH_GPT_graph_shift_cos_function_l7_795

theorem graph_shift_cos_function (f : ℝ → ℝ) (φ : ℝ) :
  (∀ x, f x = 2 * Real.cos (π * x / 3 + φ)) ∧ 
  (∃ x, f x = 0 ∧ x = 2) ∧ 
  (f 1 > f 3) →
  (∀ x, f x = 2 * Real.cos (π * (x - 1/2) / 3)) :=
by
  sorry

end NUMINAMATH_GPT_graph_shift_cos_function_l7_795


namespace NUMINAMATH_GPT_third_group_members_l7_727

-- Define the total number of members in the choir
def total_members : ℕ := 70

-- Define the number of members in the first group
def first_group_members : ℕ := 25

-- Define the number of members in the second group
def second_group_members : ℕ := 30

-- Prove that the number of members in the third group is 15
theorem third_group_members : total_members - first_group_members - second_group_members = 15 := 
by 
  sorry

end NUMINAMATH_GPT_third_group_members_l7_727


namespace NUMINAMATH_GPT_first_position_remainder_one_l7_769

theorem first_position_remainder_one (a : ℕ) (h1 : 1 ≤ a ∧ a ≤ 2023)
(h2 : ∀ b c d : ℕ, b = a ∧ c = a + 2 ∧ d = a + 4 → 
  b % 3 ≠ c % 3 ∧ c % 3 ≠ d % 3 ∧ d % 3 ≠ b % 3):
  a % 3 = 1 :=
sorry

end NUMINAMATH_GPT_first_position_remainder_one_l7_769


namespace NUMINAMATH_GPT_find_k_eq_neg2_l7_732

theorem find_k_eq_neg2 (k : ℝ) (h : (-1)^2 - k * (-1) + 1 = 0) : k = -2 :=
by sorry

end NUMINAMATH_GPT_find_k_eq_neg2_l7_732


namespace NUMINAMATH_GPT_sin_cos_sixth_power_l7_794

theorem sin_cos_sixth_power (θ : ℝ) 
  (h : Real.sin (3 * θ) = 1 / 2) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
  sorry

end NUMINAMATH_GPT_sin_cos_sixth_power_l7_794


namespace NUMINAMATH_GPT_midpoint_of_complex_numbers_l7_746

theorem midpoint_of_complex_numbers :
  let A := (1 - 1*I) / (1 + 1)
  let B := (1 + 1*I) / (1 + 1)
  (A + B) / 2 = 1 / 2 := by
sorry

end NUMINAMATH_GPT_midpoint_of_complex_numbers_l7_746


namespace NUMINAMATH_GPT_smallest_n_l7_770

theorem smallest_n (n : ℕ) (hn : n > 0) (h : 623 * n % 32 = 1319 * n % 32) : n = 4 :=
sorry

end NUMINAMATH_GPT_smallest_n_l7_770


namespace NUMINAMATH_GPT_cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l7_797

theorem cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2 : 
  Real.cos (- (11 / 4) * Real.pi) = - Real.sqrt 2 / 2 := 
sorry

end NUMINAMATH_GPT_cos_neg_11_div_4_pi_eq_neg_sqrt_2_div_2_l7_797


namespace NUMINAMATH_GPT_max_ratio_of_odd_integers_is_nine_l7_705

-- Define odd positive integers x and y whose mean is 55
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_positive (n : ℕ) : Prop := 0 < n
def mean_is_55 (x y : ℕ) : Prop := (x + y) / 2 = 55

-- The problem statement
theorem max_ratio_of_odd_integers_is_nine (x y : ℕ) 
  (hx : is_positive x) (hy : is_positive y)
  (ox : is_odd x) (oy : is_odd y)
  (mean : mean_is_55 x y) : 
  ∀ r, r = (x / y : ℚ) → r ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_max_ratio_of_odd_integers_is_nine_l7_705


namespace NUMINAMATH_GPT_axis_of_symmetry_l7_710

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + (Real.pi / 2))) * (Real.cos (x + (Real.pi / 4)))

theorem axis_of_symmetry : 
  ∃ (a : ℝ), a = 5 * Real.pi / 8 ∧ ∀ x : ℝ, f (2 * a - x) = f x := 
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l7_710


namespace NUMINAMATH_GPT_find_x_solution_l7_774

theorem find_x_solution (x : ℚ) : (∀ y : ℚ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_solution_l7_774


namespace NUMINAMATH_GPT_distribution_of_balls_l7_719

theorem distribution_of_balls (n k : ℕ) (h_n : n = 6) (h_k : k = 3) : k^n = 729 := by
  rw [h_n, h_k]
  exact rfl

end NUMINAMATH_GPT_distribution_of_balls_l7_719


namespace NUMINAMATH_GPT_Jung_age_is_26_l7_786

-- Define the ages of Li, Zhang, and Jung
def Li : ℕ := 12
def Zhang : ℕ := 2 * Li
def Jung : ℕ := Zhang + 2

-- The goal is to prove Jung's age is 26 years
theorem Jung_age_is_26 : Jung = 26 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_Jung_age_is_26_l7_786


namespace NUMINAMATH_GPT_total_amount_paid_l7_711

-- Definitions
def original_aquarium_price : ℝ := 120
def aquarium_discount : ℝ := 0.5
def aquarium_coupon : ℝ := 0.1
def aquarium_sales_tax : ℝ := 0.05

def plants_decorations_price_before_discount : ℝ := 75
def plants_decorations_discount : ℝ := 0.15
def plants_decorations_sales_tax : ℝ := 0.08

def fish_food_price : ℝ := 25
def fish_food_sales_tax : ℝ := 0.06

-- Final result to be proved
theorem total_amount_paid : 
  let discounted_aquarium_price := original_aquarium_price * (1 - aquarium_discount)
  let coupon_aquarium_price := discounted_aquarium_price * (1 - aquarium_coupon)
  let total_aquarium_price := coupon_aquarium_price * (1 + aquarium_sales_tax)
  let discounted_plants_decorations_price := plants_decorations_price_before_discount * (1 - plants_decorations_discount)
  let total_plants_decorations_price := discounted_plants_decorations_price * (1 + plants_decorations_sales_tax)
  let total_fish_food_price := fish_food_price * (1 + fish_food_sales_tax)
  total_aquarium_price + total_plants_decorations_price + total_fish_food_price = 152.05 :=
by 
  sorry

end NUMINAMATH_GPT_total_amount_paid_l7_711


namespace NUMINAMATH_GPT_A_share_of_gain_l7_750

-- Definitions of conditions
variables 
  (x : ℕ) -- Initial investment by A
  (annual_gain : ℕ := 24000) -- Total annual gain
  (A_investment_period : ℕ := 12) -- Months A invested
  (B_investment_period : ℕ := 6) -- Months B invested after 6 months
  (C_investment_period : ℕ := 4) -- Months C invested after 8 months

-- Investment ratios
def A_ratio := x * A_investment_period
def B_ratio := (2 * x) * B_investment_period
def C_ratio := (3 * x) * C_investment_period

-- Proof statement
theorem A_share_of_gain : 
  A_ratio = 12 * x ∧ B_ratio = 12 * x ∧ C_ratio = 12 * x ∧ annual_gain = 24000 →
  annual_gain / 3 = 8000 :=
by
  sorry

end NUMINAMATH_GPT_A_share_of_gain_l7_750


namespace NUMINAMATH_GPT_minimize_water_tank_construction_cost_l7_700

theorem minimize_water_tank_construction_cost 
  (volume : ℝ := 4800)
  (depth : ℝ := 3)
  (cost_bottom_per_m2 : ℝ := 150)
  (cost_walls_per_m2 : ℝ := 120)
  (x : ℝ) :
  (volume = x * x * depth) →
  (∀ y, y = cost_bottom_per_m2 * x * x + cost_walls_per_m2 * 4 * x * depth) →
  (x = 40) ∧ (y = 297600) :=
by
  sorry

end NUMINAMATH_GPT_minimize_water_tank_construction_cost_l7_700


namespace NUMINAMATH_GPT_factor_expression_l7_723

theorem factor_expression (x y z : ℝ) :
  ((x^3 - y^3)^3 + (y^3 - z^3)^3 + (z^3 - x^3)^3) / 
  ((x - y)^3 + (y - z)^3 + (z - x)^3) = 
  ((x^2 + x * y + y^2) * (y^2 + y * z + z^2) * (z^2 + z * x + x^2)) :=
by {
  sorry  -- The proof goes here
}

end NUMINAMATH_GPT_factor_expression_l7_723


namespace NUMINAMATH_GPT_ryan_lamps_probability_l7_787

theorem ryan_lamps_probability :
  let total_lamps := 8
  let red_lamps := 4
  let blue_lamps := 4
  let total_ways_to_arrange := Nat.choose total_lamps red_lamps
  let total_ways_to_turn_on := Nat.choose total_lamps 4
  let remaining_blue := blue_lamps - 1 -- Due to leftmost lamp being blue and off
  let remaining_red := red_lamps - 1 -- Due to rightmost lamp being red and on
  let remaining_red_after_middle := remaining_red - 1 -- Due to middle lamp being red and off
  let remaining_lamps := remaining_blue + remaining_red_after_middle
  let ways_to_assign_remaining_red := Nat.choose remaining_lamps remaining_red_after_middle
  let ways_to_turn_on_remaining_lamps := Nat.choose remaining_lamps 2
  let favorable_ways := ways_to_assign_remaining_red * ways_to_turn_on_remaining_lamps
  let total_possibilities := total_ways_to_arrange * total_ways_to_turn_on
  favorable_ways / total_possibilities = (10 / 490) := by
  sorry

end NUMINAMATH_GPT_ryan_lamps_probability_l7_787


namespace NUMINAMATH_GPT_max_sum_arithmetic_sequence_l7_766

theorem max_sum_arithmetic_sequence (n : ℕ) (M : ℝ) (hM : 0 < M) 
  (a : ℕ → ℝ) (h_arith_seq : ∀ k, a (k + 1) - a k = a 1 - a 0) 
  (h_constraint : a 1 ^ 2 + a (n + 1) ^ 2 ≤ M) :
  ∃ S, S = (n + 1) * (Real.sqrt (10 * M)) / 2 :=
sorry

end NUMINAMATH_GPT_max_sum_arithmetic_sequence_l7_766


namespace NUMINAMATH_GPT_max_value_quadratic_function_l7_742

noncomputable def quadratic_function (x : ℝ) : ℝ :=
  -3 * x^2 + 8

theorem max_value_quadratic_function : ∃(x : ℝ), quadratic_function x = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_value_quadratic_function_l7_742


namespace NUMINAMATH_GPT_find_a_plus_2b_l7_712

variable (a b : ℝ)

theorem find_a_plus_2b (h : (a^2 + 4 * a + 6) * (2 * b^2 - 4 * b + 7) ≤ 10) : 
  a + 2 * b = 0 := 
sorry

end NUMINAMATH_GPT_find_a_plus_2b_l7_712


namespace NUMINAMATH_GPT_min_value_l7_709

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 :=
sorry

end NUMINAMATH_GPT_min_value_l7_709


namespace NUMINAMATH_GPT_twelfth_term_of_geometric_sequence_l7_733

theorem twelfth_term_of_geometric_sequence (a : ℕ) (r : ℕ) (h1 : a * r ^ 4 = 8) (h2 : a * r ^ 8 = 128) : 
  a * r ^ 11 = 1024 :=
sorry

end NUMINAMATH_GPT_twelfth_term_of_geometric_sequence_l7_733


namespace NUMINAMATH_GPT_walls_per_person_l7_713

theorem walls_per_person (people : ℕ) (rooms : ℕ) (r4_walls r5_walls : ℕ) (total_walls : ℕ) (walls_each_person : ℕ)
  (h1 : people = 5)
  (h2 : rooms = 9)
  (h3 : r4_walls = 5 * 4)
  (h4 : r5_walls = 4 * 5)
  (h5 : total_walls = r4_walls + r5_walls)
  (h6 : walls_each_person = total_walls / people) :
  walls_each_person = 8 := by
  sorry

end NUMINAMATH_GPT_walls_per_person_l7_713


namespace NUMINAMATH_GPT_consecutive_sum_ways_l7_722

theorem consecutive_sum_ways (S : ℕ) (hS : S = 385) :
  ∃! n : ℕ, ∃! k : ℕ, n ≥ 2 ∧ S = n * (2 * k + n - 1) / 2 :=
sorry

end NUMINAMATH_GPT_consecutive_sum_ways_l7_722


namespace NUMINAMATH_GPT_perfect_square_trinomial_l7_785

variable (x y : ℝ)

theorem perfect_square_trinomial (a : ℝ) :
  (∃ b c : ℝ, 4 * x^2 - (a - 1) * x * y + 9 * y^2 = (b * x + c * y) ^ 2) ↔ 
  (a = 13 ∨ a = -11) := 
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l7_785


namespace NUMINAMATH_GPT_nth_group_sum_correct_l7_798

-- Define the function that computes the sum of the numbers in the nth group
def nth_group_sum (n : ℕ) : ℕ :=
  n * (n^2 + 1) / 2

-- The theorem statement
theorem nth_group_sum_correct (n : ℕ) : 
  nth_group_sum n = n * (n^2 + 1) / 2 := by
  sorry

end NUMINAMATH_GPT_nth_group_sum_correct_l7_798


namespace NUMINAMATH_GPT_johns_speed_l7_726

theorem johns_speed (J : ℝ)
  (lewis_speed : ℝ := 60)
  (distance_AB : ℝ := 240)
  (meet_distance_A : ℝ := 160)
  (time_lewis_to_B : ℝ := distance_AB / lewis_speed)
  (time_lewis_back_80 : ℝ := 80 / lewis_speed)
  (total_time_meet : ℝ := time_lewis_to_B + time_lewis_back_80)
  (total_distance_john_meet : ℝ := J * total_time_meet) :
  total_distance_john_meet = meet_distance_A → J = 30 := 
by
  sorry

end NUMINAMATH_GPT_johns_speed_l7_726


namespace NUMINAMATH_GPT_hockey_cards_count_l7_775

-- Define integer variables for the number of hockey, football and baseball cards
variables (H F B : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := F = 4 * H
def condition2 : Prop := B = F - 50
def condition3 : Prop := H > 0
def condition4 : Prop := H + F + B = 1750

-- The theorem to prove
theorem hockey_cards_count 
  (h1 : condition1 H F)
  (h2 : condition2 F B)
  (h3 : condition3 H)
  (h4 : condition4 H F B) : 
  H = 200 := by
sorry

end NUMINAMATH_GPT_hockey_cards_count_l7_775


namespace NUMINAMATH_GPT_fraction_div_addition_l7_729

noncomputable def fraction_5_6 : ℚ := 5 / 6
noncomputable def fraction_9_10 : ℚ := 9 / 10
noncomputable def fraction_1_15 : ℚ := 1 / 15
noncomputable def fraction_402_405 : ℚ := 402 / 405

theorem fraction_div_addition :
  (fraction_5_6 / fraction_9_10) + fraction_1_15 = fraction_402_405 :=
by
  sorry

end NUMINAMATH_GPT_fraction_div_addition_l7_729


namespace NUMINAMATH_GPT_fraction_of_students_received_As_l7_793

/-- Assume A is the fraction of students who received A's,
and B is the fraction of students who received B's,
and T is the total fraction of students who received either A's or B's. -/
theorem fraction_of_students_received_As
  (A B T : ℝ)
  (hB : B = 0.2)
  (hT : T = 0.9)
  (h : A + B = T) :
  A = 0.7 := 
by
  -- establishing the proof steps
  sorry

end NUMINAMATH_GPT_fraction_of_students_received_As_l7_793


namespace NUMINAMATH_GPT_hawks_score_l7_707

theorem hawks_score (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 22) : H = 30 :=
by
  sorry

end NUMINAMATH_GPT_hawks_score_l7_707


namespace NUMINAMATH_GPT_label_sum_l7_783

theorem label_sum (n : ℕ) : 
  (∃ S : ℕ → ℕ, S 1 = 2 ∧ (∀ k, k > 1 → (S (k + 1) = 2 * S k)) ∧ S n = 2 * 3 ^ (n - 1)) := 
sorry

end NUMINAMATH_GPT_label_sum_l7_783


namespace NUMINAMATH_GPT_find_point_on_parabola_l7_754

open Real

theorem find_point_on_parabola :
  ∃ (x y : ℝ), 
  (0 ≤ x ∧ 0 ≤ y) ∧
  (x^2 = 8 * y) ∧
  sqrt (x^2 + (y - 2)^2) = 120 ∧
  (x = 2 * sqrt 236 ∧ y = 118) :=
by
  sorry

end NUMINAMATH_GPT_find_point_on_parabola_l7_754


namespace NUMINAMATH_GPT_tan_domain_l7_788

open Real

theorem tan_domain (k : ℤ) (x : ℝ) :
  (∀ k : ℤ, x ≠ (k * π / 2) + (3 * π / 8)) ↔ 
  (∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2) := sorry

end NUMINAMATH_GPT_tan_domain_l7_788


namespace NUMINAMATH_GPT_max_min_condition_monotonic_condition_l7_740

-- (1) Proving necessary and sufficient condition for f(x) to have both a maximum and minimum value
theorem max_min_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ -2*x₁ + a - (1/x₁) = 0 ∧ -2*x₂ + a - (1/x₂) = 0) ↔ a > Real.sqrt 8 :=
sorry

-- (2) Proving the range of values for a such that f(x) is monotonic on [1, 2]
theorem monotonic_condition (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≥ 0) ∨
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (-2 * x + a - (1 / x)) ≤ 0) ↔ a ≤ 3 ∨ a ≥ 4.5 :=
sorry

end NUMINAMATH_GPT_max_min_condition_monotonic_condition_l7_740


namespace NUMINAMATH_GPT_problem_1_problem_2_l7_744

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem problem_1 (x : ℝ) : (f x 2) ≥ (7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) := 
by
  sorry

theorem problem_2 (m n : ℝ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h : (f (1/m) 1) + (f (1/(2*n)) 1) = 1) : m + 4 * n ≥ 2 * Real.sqrt 2 + 3 := 
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l7_744


namespace NUMINAMATH_GPT_derivative_at_one_l7_760

variable (x : ℝ)

def f (x : ℝ) := x^2 - 2*x + 3

theorem derivative_at_one : deriv f 1 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_derivative_at_one_l7_760


namespace NUMINAMATH_GPT_range_of_m_l7_792

noncomputable def f (m x : ℝ) : ℝ := x^3 + m * x^2 + (m + 6) * x + 1

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, ∃ c : ℝ, (f m x) ≤ c ∧ (f m y) ≥ (f m x) ∧ ∀ z : ℝ, f m z ≥ f m x ∧ f m z ≤ c) ↔ (m < -3 ∨ m > 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l7_792


namespace NUMINAMATH_GPT_cone_height_l7_701

theorem cone_height (r_sphere : ℝ) (r_cone : ℝ) (waste_percentage : ℝ) 
  (h : ℝ) (V_sphere : ℝ) (V_cone : ℝ) : 
  r_sphere = 9 → r_cone = 9 → waste_percentage = 0.75 → 
  V_sphere = (4 / 3) * Real.pi * r_sphere^3 → 
  V_cone = (1 / 3) * Real.pi * r_cone^2 * h → 
  V_cone = waste_percentage * V_sphere → 
  h = 27 :=
by
  intros r_sphere_eq r_cone_eq waste_eq V_sphere_eq V_cone_eq V_cone_waste_eq
  sorry

end NUMINAMATH_GPT_cone_height_l7_701


namespace NUMINAMATH_GPT_blue_paper_side_length_l7_768

theorem blue_paper_side_length (side_red : ℝ) (side_blue : ℝ) (same_area : side_red^2 = side_blue * x) (side_red_val : side_red = 5) (side_blue_val : side_blue = 4) : x = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_blue_paper_side_length_l7_768


namespace NUMINAMATH_GPT_semicircle_triangle_l7_731

variable (a b r : ℝ)

-- Conditions: 
-- (1) Semicircle of radius r inside a right-angled triangle
-- (2) Shorter edges of the triangle (tangents to the semicircle) have lengths a and b
-- (3) Diameter of the semicircle lies on the hypotenuse of the triangle

theorem semicircle_triangle (h1 : a > 0) (h2 : b > 0) (h3 : r > 0)
  (tangent_property : true) -- Assumed relevant tangent properties are true
  (angle_property : true) -- Assumed relevant angle properties are true
  (geom_configuration : true) -- Assumed specific geometric configuration is correct
  : 1 / r = 1 / a + 1 / b := 
  sorry

end NUMINAMATH_GPT_semicircle_triangle_l7_731


namespace NUMINAMATH_GPT_infinite_solutions_x2_y2_z2_x3_y3_z3_l7_734

-- Define the parametric forms
def param_x (k : ℤ) := k * (2 * k^2 + 1)
def param_y (k : ℤ) := 2 * k^2 + 1
def param_z (k : ℤ) := -k * (2 * k^2 + 1)

-- Prove the equation
theorem infinite_solutions_x2_y2_z2_x3_y3_z3 :
  ∀ k : ℤ, param_x k ^ 2 + param_y k ^ 2 + param_z k ^ 2 = param_x k ^ 3 + param_y k ^ 3 + param_z k ^ 3 :=
by
  intros k
  -- Calculation needs to be proved here, we place a placeholder for now
  sorry

end NUMINAMATH_GPT_infinite_solutions_x2_y2_z2_x3_y3_z3_l7_734


namespace NUMINAMATH_GPT_chord_length_intercepted_by_line_on_curve_l7_725

-- Define the curve and line from the problem
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 1 = 0
def line (x y : ℝ) : Prop := 2*x + y = 0

-- Prove the length of the chord intercepted by the line on the curve is 4
theorem chord_length_intercepted_by_line_on_curve : 
  ∀ (x y : ℝ), curve x y → line x y → False := sorry

end NUMINAMATH_GPT_chord_length_intercepted_by_line_on_curve_l7_725


namespace NUMINAMATH_GPT_maximum_b_value_l7_765

noncomputable def f (a x : ℝ) := (1 / 2) * x ^ 2 + a * x
noncomputable def g (a b x : ℝ) := 2 * a ^ 2 * Real.log x + b

theorem maximum_b_value (a b : ℝ) (h_a : 0 < a) :
  (∃ x : ℝ, f a x = g a b x ∧ (deriv (f a) x = deriv (g a b) x))
  → b ≤ Real.exp (1 / 2) := 
sorry

end NUMINAMATH_GPT_maximum_b_value_l7_765


namespace NUMINAMATH_GPT_ratio_of_terms_l7_789

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)

theorem ratio_of_terms
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (h₀ : ∀ n : ℕ, S n = geometric_sum (a 1) (a 2) n)
  (h₁ : ∀ n : ℕ, T n = geometric_sum (b 1) (b 2) n)
  (h₂ : ∀ n : ℕ, n > 0 → S n / T n = (3 ^ n + 1) / 4) :
  a 3 / b 4 = 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_terms_l7_789
