import Mathlib

namespace rob_baseball_cards_l1541_154113

theorem rob_baseball_cards
  (r j r_d : ℕ)
  (hj : j = 40)
  (h_double : r_d = j / 5)
  (h_cards : r = 3 * r_d) :
  r = 24 :=
by
  sorry

end rob_baseball_cards_l1541_154113


namespace shaded_regions_area_l1541_154184

/-- Given a grid of 1x1 squares with 2015 shaded regions where boundaries are either:
    - Horizontal line segments
    - Vertical line segments
    - Segments connecting the midpoints of adjacent sides of 1x1 squares
    - Diagonals of 1x1 squares

    Prove that the total area of these 2015 shaded regions is 47.5.
-/
theorem shaded_regions_area (n : ℕ) (h1 : n = 2015) : 
  ∃ (area : ℝ), area = 47.5 :=
by sorry

end shaded_regions_area_l1541_154184


namespace average_test_score_45_percent_l1541_154128

theorem average_test_score_45_percent (x : ℝ) 
  (h1 : 0.45 * x + 0.50 * 78 + 0.05 * 60 = 84.75) : 
  x = 95 :=
by sorry

end average_test_score_45_percent_l1541_154128


namespace first_player_winning_strategy_l1541_154157

theorem first_player_winning_strategy (num_chips : ℕ) : 
  (num_chips = 110) → 
  ∃ (moves : ℕ → ℕ × ℕ), (∀ n, 1 ≤ (moves n).1 ∧ (moves n).1 ≤ 9) ∧ 
  (∀ n, (moves n).1 ≠ (moves (n-1)).1) →
  (∃ move_sequence : ℕ → ℕ, ∀ k, move_sequence k ≤ num_chips ∧ 
  ((move_sequence (k+1) < move_sequence k) ∨ (move_sequence (k+1) = 0 ∧ move_sequence k = 1)) ∧ 
  (move_sequence k > 0) ∧ (move_sequence 0 = num_chips) →
  num_chips ≡ 14 [MOD 32]) :=
by 
  sorry

end first_player_winning_strategy_l1541_154157


namespace sale_in_first_month_l1541_154156

theorem sale_in_first_month 
  (sale_month_2 : ℕ)
  (sale_month_3 : ℕ)
  (sale_month_4 : ℕ)
  (sale_month_5 : ℕ)
  (required_sale_month_6 : ℕ)
  (average_sale_6_months : ℕ)
  (total_sale_6_months : ℕ)
  (total_known_sales : ℕ)
  (sale_first_month : ℕ) : 
    sale_month_2 = 3920 →
    sale_month_3 = 3855 →
    sale_month_4 = 4230 →
    sale_month_5 = 3560 →
    required_sale_month_6 = 2000 →
    average_sale_6_months = 3500 →
    total_sale_6_months = 6 * average_sale_6_months →
    total_known_sales = sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 →
    total_sale_6_months - (total_known_sales + required_sale_month_6) = sale_first_month →
    sale_first_month = 3435 :=
by
  intros h2 h3 h4 h5 h6 h_avg h_total h_known h_calc
  sorry

end sale_in_first_month_l1541_154156


namespace evaluate_expression_l1541_154166

theorem evaluate_expression : 2009 * (2007 / 2008) + (1 / 2008) = 2008 := 
by 
  sorry

end evaluate_expression_l1541_154166


namespace sin_alpha_plus_half_pi_l1541_154144

theorem sin_alpha_plus_half_pi (α : ℝ) 
  (h1 : Real.tan (α - Real.pi) = 3 / 4)
  (h2 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2)) : 
  Real.sin (α + Real.pi / 2) = -4 / 5 :=
by
  -- Placeholder for the proof
  sorry

end sin_alpha_plus_half_pi_l1541_154144


namespace maximal_product_sum_l1541_154196

theorem maximal_product_sum : 
  ∃ (k m : ℕ), 
  k = 671 ∧ 
  m = 2 ∧ 
  2017 = 3 * k + 2 * m ∧ 
  ∀ a b : ℕ, a + b = 2017 ∧ (a < k ∨ b < m) → a * b ≤ 3 * k * 2 * m
:= 
sorry

end maximal_product_sum_l1541_154196


namespace job_time_relation_l1541_154105

theorem job_time_relation (a b c m n x : ℝ) 
  (h1 : m / a = 1 / b + 1 / c)
  (h2 : n / b = 1 / a + 1 / c)
  (h3 : x / c = 1 / a + 1 / b) :
  x = (m + n + 2) / (m * n - 1) := 
sorry

end job_time_relation_l1541_154105


namespace points_five_units_away_from_neg_one_l1541_154146

theorem points_five_units_away_from_neg_one (x : ℝ) :
  |x + 1| = 5 ↔ x = 4 ∨ x = -6 :=
by
  sorry

end points_five_units_away_from_neg_one_l1541_154146


namespace sara_disproves_tom_l1541_154145

-- Define the type and predicate of cards
inductive Card
| K
| M
| card5
| card7
| card8

open Card

-- Define the conditions
def is_consonant : Card → Prop
| K => true
| M => true
| _ => false

def is_odd : Card → Prop
| card5 => true
| card7 => true
| _ => false

def is_even : Card → Prop
| card8 => true
| _ => false

-- Tom's statement
def toms_statement : Prop :=
  ∀ c, is_consonant c → is_odd c

-- The card Sara turns over (card8) to disprove Tom's statement
theorem sara_disproves_tom : is_even card8 ∧ is_consonant card8 → ¬toms_statement :=
by
  sorry

end sara_disproves_tom_l1541_154145


namespace measure_of_angle_Q_l1541_154178

theorem measure_of_angle_Q (Q R : ℝ) 
  (h1 : Q = 2 * R)
  (h2 : 130 + 90 + 110 + 115 + Q + R = 540) :
  Q = 63.33 :=
by
  sorry

end measure_of_angle_Q_l1541_154178


namespace total_capacity_iv_bottle_l1541_154187

-- Definitions of the conditions
def initial_volume : ℝ := 100 -- milliliters
def rate_of_flow : ℝ := 2.5 -- milliliters per minute
def observation_time : ℝ := 12 -- minutes
def empty_space_at_12_min : ℝ := 80 -- milliliters

-- Definition of the problem statement in Lean 4
theorem total_capacity_iv_bottle :
  initial_volume + rate_of_flow * observation_time + empty_space_at_12_min = 150 := 
by
  sorry

end total_capacity_iv_bottle_l1541_154187


namespace problem_statement_l1541_154118

theorem problem_statement (x y z w : ℝ)
  (h1 : x + y + z + w = 0)
  (h7 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := 
sorry

end problem_statement_l1541_154118


namespace james_second_hour_distance_l1541_154101

theorem james_second_hour_distance :
  ∃ x : ℝ, 
    x + 1.20 * x + 1.50 * x = 37 ∧ 
    1.20 * x = 12 :=
by
  sorry

end james_second_hour_distance_l1541_154101


namespace smallest_possible_intersections_l1541_154197

theorem smallest_possible_intersections (n : ℕ) (hn : n = 2000) :
  ∃ N : ℕ, N ≥ 3997 :=
by
  sorry

end smallest_possible_intersections_l1541_154197


namespace calculation_l1541_154194

-- Define the exponents and base values as conditions
def exponent : ℕ := 3 ^ 2
def neg_base : ℤ := -2
def pos_base : ℤ := 2

-- The calculation expressions as conditions
def term1 : ℤ := neg_base^exponent
def term2 : ℤ := pos_base^exponent

-- The proof statement: Show that the sum of the terms equals 0
theorem calculation : term1 + term2 = 0 := sorry

end calculation_l1541_154194


namespace molecular_weight_of_one_mole_l1541_154102

theorem molecular_weight_of_one_mole (molecular_weight_8_moles : ℝ) (h : molecular_weight_8_moles = 992) : 
  molecular_weight_8_moles / 8 = 124 :=
by
  -- proof goes here
  sorry

end molecular_weight_of_one_mole_l1541_154102


namespace monotonic_intervals_range_of_a_for_inequality_l1541_154140

noncomputable def f (a x : ℝ) : ℝ := (x + a) / (a * Real.exp x)

theorem monotonic_intervals (a : ℝ) :
  (if a > 0 then
    ∀ x, (x < (1 - a) → 0 < deriv (f a) x) ∧ ((1 - a) < x → deriv (f a) x < 0)
  else
    ∀ x, (x < (1 - a) → deriv (f a) x < 0) ∧ ((1 - a) < x → 0 < deriv (f a) x)) := 
sorry

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x, 0 < x → (3 + 2 * Real.log x) / Real.exp x ≤ f a x + 2 * x) ↔
  a ∈ Set.Iio (-1/2) ∪ Set.Ioi 0 :=
sorry

end monotonic_intervals_range_of_a_for_inequality_l1541_154140


namespace find_radii_l1541_154173

theorem find_radii (r R : ℝ) (h₁ : R - r = 2) (h₂ : R + r = 16) : r = 7 ∧ R = 9 := by
  sorry

end find_radii_l1541_154173


namespace ellipse_parabola_intersection_l1541_154182

open Real

theorem ellipse_parabola_intersection (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) ↔ (-1 ≤ a ∧ a ≤ 17 / 8) :=
by
  sorry

end ellipse_parabola_intersection_l1541_154182


namespace ariel_age_l1541_154168

theorem ariel_age : ∃ A : ℕ, (A + 15 = 4 * A) ∧ A = 5 :=
by
  -- Here we skip the proof
  sorry

end ariel_age_l1541_154168


namespace fraction_food_l1541_154107

-- Define the salary S and remaining amount H
def S : ℕ := 170000
def H : ℕ := 17000

-- Define fractions of the salary spent on house rent and clothes
def fraction_rent : ℚ := 1 / 10
def fraction_clothes : ℚ := 3 / 5

-- Define the fraction F to be proven
def F : ℚ := 1 / 5

-- Define the remaining fraction of the salary
def remaining_fraction : ℚ := H / S

theorem fraction_food :
  ∀ S H : ℕ,
  S = 170000 →
  H = 17000 →
  F = 1 / 5 →
  F + (fraction_rent + fraction_clothes) + remaining_fraction = 1 :=
by
  intros S H hS hH hF
  sorry

end fraction_food_l1541_154107


namespace relativ_prime_and_divisible_exists_l1541_154108

theorem relativ_prime_and_divisible_exists
  (a b c : ℕ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c) :
  ∃ r s : ℕ, Nat.gcd r s = 1 ∧ 0 < r ∧ 0 < s ∧ c ∣ (a * r + b * s) :=
by
  sorry

end relativ_prime_and_divisible_exists_l1541_154108


namespace vector_subtraction_l1541_154103

variables (a b : ℝ × ℝ)

-- Definitions based on conditions
def vector_a : ℝ × ℝ := (1, -2)
def m : ℝ := 2
def vector_b : ℝ × ℝ := (4, m)

-- Prove given question equals answer
theorem vector_subtraction :
  vector_a = (1, -2) →
  vector_b = (4, m) →
  (1 * 4 + (-2) * m = 0) →
  5 • vector_a - vector_b = (1, -12) := by
  intros h1 h2 h3
  sorry

end vector_subtraction_l1541_154103


namespace total_revenue_is_correct_l1541_154167

-- Define the constants and conditions
def price_of_jeans : ℕ := 11
def price_of_tees : ℕ := 8
def quantity_of_tees_sold : ℕ := 7
def quantity_of_jeans_sold : ℕ := 4

-- Define the total revenue calculation
def total_revenue : ℕ :=
  (price_of_tees * quantity_of_tees_sold) +
  (price_of_jeans * quantity_of_jeans_sold)

-- The theorem to prove
theorem total_revenue_is_correct : total_revenue = 100 := 
by
  -- Proof is omitted for now
  sorry

end total_revenue_is_correct_l1541_154167


namespace altitude_eqn_median_eqn_l1541_154126

def Point := (ℝ × ℝ)

def A : Point := (4, 0)
def B : Point := (6, 7)
def C : Point := (0, 3)

theorem altitude_eqn (B C: Point) : 
  ∃ (k b : ℝ), (b = 6) ∧ (k = - 3 / 2) ∧ (∀ x y : ℝ, y = k * x + b →
  3 * x + 2 * y - 12 = 0)
:=
sorry

theorem median_eqn (A B C : Point) :
  ∃ (k b : ℝ), (b = 20) ∧ (k = -3/5) ∧ (∀ x y : ℝ, y = k * x + b →
  5 * x + y - 20 = 0)
:=
sorry

end altitude_eqn_median_eqn_l1541_154126


namespace plane_figures_l1541_154199

def polyline_two_segments : Prop := -- Definition for a polyline composed of two line segments
  sorry

def polyline_three_segments : Prop := -- Definition for a polyline composed of three line segments
  sorry

def closed_three_segments : Prop := -- Definition for a closed figure composed of three line segments
  sorry

def quadrilateral_equal_opposite_sides : Prop := -- Definition for a quadrilateral with equal opposite sides
  sorry

def trapezoid : Prop := -- Definition for a trapezoid
  sorry

def is_plane_figure (fig : Prop) : Prop :=
  sorry  -- Axiom or definition that determines whether a figure is a plane figure.

-- Translating the proof problem
theorem plane_figures :
  is_plane_figure polyline_two_segments ∧
  ¬ is_plane_figure polyline_three_segments ∧
  is_plane_figure closed_three_segments ∧
  ¬ is_plane_figure quadrilateral_equal_opposite_sides ∧
  is_plane_figure trapezoid :=
by
  sorry

end plane_figures_l1541_154199


namespace ott_fractional_part_l1541_154138

theorem ott_fractional_part (M L N O x : ℝ)
  (hM : M = 6 * x)
  (hL : L = 5 * x)
  (hN : N = 4 * x)
  (hO : O = 0)
  (h_each : O + M + L + N = x + x + x) :
  (3 * x) / (M + L + N) = 1 / 5 :=
by
  sorry

end ott_fractional_part_l1541_154138


namespace multiple_of_P_l1541_154190

theorem multiple_of_P (P Q R : ℝ) (T : ℝ) (x : ℝ) (total_profit Rs900 : ℝ)
  (h1 : P = 6 * Q)
  (h2 : P = 10 * R)
  (h3 : R = T / 5.1)
  (h4 : total_profit = Rs900 + (T - R)) :
  x = 10 :=
by
  sorry

end multiple_of_P_l1541_154190


namespace necessary_and_sufficient_condition_l1541_154165

theorem necessary_and_sufficient_condition (a b : ℝ) : a > b ↔ a^3 > b^3 :=
by {
  sorry
}

end necessary_and_sufficient_condition_l1541_154165


namespace tammy_investment_change_l1541_154121

-- Defining initial investment, losses, and gains
def initial_investment : ℝ := 100
def first_year_loss : ℝ := 0.10
def second_year_gain : ℝ := 0.25

-- Defining the final amount after two years
def final_amount (initial_investment : ℝ) (first_year_loss : ℝ) (second_year_gain : ℝ) : ℝ :=
  let remaining_after_first_year := initial_investment * (1 - first_year_loss)
  remaining_after_first_year * (1 + second_year_gain)

-- Statement to prove
theorem tammy_investment_change :
  let percentage_change := ((final_amount initial_investment first_year_loss second_year_gain - initial_investment) / initial_investment) * 100
  percentage_change = 12.5 :=
by
  sorry

end tammy_investment_change_l1541_154121


namespace percentage_fruits_in_good_condition_l1541_154158

theorem percentage_fruits_in_good_condition (oranges bananas : ℕ) (rotten_oranges_pct rotten_bananas_pct : ℚ)
    (h_oranges : oranges = 600) (h_bananas : bananas = 400)
    (h_rotten_oranges_pct : rotten_oranges_pct = 0.15) (h_rotten_bananas_pct : rotten_bananas_pct = 0.06) :
    let rotten_oranges := (rotten_oranges_pct * oranges : ℚ)
    let rotten_bananas := (rotten_bananas_pct * bananas : ℚ)
    let total_rotten := rotten_oranges + rotten_bananas
    let total_fruits := (oranges + bananas : ℚ)
    let good_fruits := total_fruits - total_rotten
    let percentage_good_fruits := (good_fruits / total_fruits) * 100
    percentage_good_fruits = 88.6 :=
by
    sorry

end percentage_fruits_in_good_condition_l1541_154158


namespace speed_rowing_upstream_l1541_154125

theorem speed_rowing_upstream (V_m V_down : ℝ) (V_s V_up : ℝ)
  (h1 : V_m = 28) (h2 : V_down = 30) (h3 : V_down = V_m + V_s) (h4 : V_up = V_m - V_s) : 
  V_up = 26 :=
by
  sorry

end speed_rowing_upstream_l1541_154125


namespace f_at_63_l1541_154164

-- Define the function f: ℤ → ℤ with given properties
def f : ℤ → ℤ :=
  sorry -- Placeholder, as we are only stating the problem, not the solution

-- Conditions
axiom f_at_1 : f 1 = 6
axiom f_eq : ∀ x : ℤ, f (2 * x + 1) = 3 * f x

-- The goal is to prove f(63) = 1458
theorem f_at_63 : f 63 = 1458 :=
  sorry

end f_at_63_l1541_154164


namespace students_exceed_guinea_pigs_and_teachers_l1541_154151

def num_students_per_classroom : Nat := 25
def num_guinea_pigs_per_classroom : Nat := 3
def num_teachers_per_classroom : Nat := 1
def num_classrooms : Nat := 5

def total_students : Nat := num_students_per_classroom * num_classrooms
def total_guinea_pigs : Nat := num_guinea_pigs_per_classroom * num_classrooms
def total_teachers : Nat := num_teachers_per_classroom * num_classrooms
def total_guinea_pigs_and_teachers : Nat := total_guinea_pigs + total_teachers

theorem students_exceed_guinea_pigs_and_teachers :
  total_students - total_guinea_pigs_and_teachers = 105 :=
by
  sorry

end students_exceed_guinea_pigs_and_teachers_l1541_154151


namespace paving_stone_length_l1541_154143

theorem paving_stone_length
  (length_courtyard : ℝ)
  (width_courtyard : ℝ)
  (num_paving_stones : ℝ)
  (width_paving_stone : ℝ)
  (total_area : ℝ := length_courtyard * width_courtyard)
  (area_per_paving_stone : ℝ := (total_area / num_paving_stones))
  (length_paving_stone : ℝ := (area_per_paving_stone / width_paving_stone)) :
  length_courtyard = 20 ∧
  width_courtyard = 16.5 ∧
  num_paving_stones = 66 ∧
  width_paving_stone = 2 →
  length_paving_stone = 2.5 :=
by {
   sorry
}

end paving_stone_length_l1541_154143


namespace largest_n_satisfying_inequality_l1541_154131

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), (∀ k : ℕ, (8 : ℚ) / 15 < n / (n + k) ∧ n / (n + k) < (7 : ℚ) / 13) ∧ 
  ∀ n' : ℕ, (∀ k : ℕ, (8 : ℚ) / 15 < n' / (n' + k) ∧ n' / (n' + k) < (7 : ℚ) / 13) → n' ≤ n :=
sorry

end largest_n_satisfying_inequality_l1541_154131


namespace problem_statement_l1541_154134

-- Define the expression in Lean
def expr : ℤ := 120 * (120 - 5) - (120 * 120 - 10 + 2)

-- Theorem stating the value of the expression
theorem problem_statement : expr = -592 := by
  sorry

end problem_statement_l1541_154134


namespace least_multiple_of_21_gt_380_l1541_154155

theorem least_multiple_of_21_gt_380 : ∃ n : ℕ, (21 * n > 380) ∧ (21 * n = 399) :=
sorry

end least_multiple_of_21_gt_380_l1541_154155


namespace min_value_of_xy_cond_l1541_154177

noncomputable def minValueOfXY (x y : ℝ) : ℝ :=
  if 2 * Real.cos (x + y - 1) ^ 2 = ((x + 1) ^ 2 + (y - 1) ^ 2 - 2 * x * y) / (x - y + 1) then 
    x * y
  else 
    0

theorem min_value_of_xy_cond (x y : ℝ) 
  (h : 2 * Real.cos (x + y - 1) ^ 2 = ((x + 1) ^ 2 + (y - 1) ^ 2 - 2 * x * y) / (x - y + 1)) : 
  (∃ k : ℤ, x = (k * Real.pi + 1) / 2 ∧ y = (k * Real.pi + 1) / 2) → 
  x * y = 1/4 := 
by
  -- The proof is omitted.
  sorry

end min_value_of_xy_cond_l1541_154177


namespace skitties_remainder_l1541_154152

theorem skitties_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 :=
sorry

end skitties_remainder_l1541_154152


namespace fourth_term_geometric_series_l1541_154159

theorem fourth_term_geometric_series (a₁ a₅ : ℕ) (r : ℕ) :
  a₁ = 6 → a₅ = 1458 → (∀ n, aₙ = a₁ * r^(n-1)) → r = 3 → (∃ a₄, a₄ = a₁ * r^(4-1) ∧ a₄ = 162) :=
by intros h₁ h₅ H r_sol
   sorry

end fourth_term_geometric_series_l1541_154159


namespace inheritance_value_l1541_154185

def inheritance_proof (x : ℝ) (federal_tax_ratio : ℝ) (state_tax_ratio : ℝ) (total_tax : ℝ) : Prop :=
  let federal_taxes := federal_tax_ratio * x
  let remaining_after_federal := x - federal_taxes
  let state_taxes := state_tax_ratio * remaining_after_federal
  let total_taxes := federal_taxes + state_taxes
  total_taxes = total_tax

theorem inheritance_value :
  inheritance_proof 41379 0.25 0.15 15000 :=
by
  sorry

end inheritance_value_l1541_154185


namespace right_triangle_cos_pq_l1541_154188

theorem right_triangle_cos_pq (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : c = 13) (h2 : b / c = 5/13) : a = 12 :=
by
  sorry

end right_triangle_cos_pq_l1541_154188


namespace initial_weight_l1541_154110

noncomputable def initial_average_weight (A : ℝ) : Prop :=
  let total_weight_initial := 20 * A
  let total_weight_new := total_weight_initial + 210
  let new_average_weight := 181.42857142857142
  total_weight_new / 21 = new_average_weight

theorem initial_weight:
  ∃ A : ℝ, initial_average_weight A ∧ A = 180 :=
by
  sorry

end initial_weight_l1541_154110


namespace nested_abs_expression_eval_l1541_154124

theorem nested_abs_expression_eval :
  abs (abs (-abs (-2 + 3) - 2) + 3) = 6 := sorry

end nested_abs_expression_eval_l1541_154124


namespace scientific_notation_example_l1541_154181

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * 10^b

theorem scientific_notation_example : 
  scientific_notation 0.00519 5.19 (-3) :=
by 
  sorry

end scientific_notation_example_l1541_154181


namespace min_m_quad_eq_integral_solutions_l1541_154106

theorem min_m_quad_eq_integral_solutions :
  (∃ m : ℕ, (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42) ∧ m > 0) →
  (∃ m : ℕ, m = 130 ∧ (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42)) :=
by
  sorry

end min_m_quad_eq_integral_solutions_l1541_154106


namespace Fedya_third_l1541_154135

/-- Definitions for order of children's arrival -/
inductive Child
| Roman | Fedya | Liza | Katya | Andrew

open Child

def arrival_order (order : Child → ℕ) : Prop :=
  order Liza > order Roman ∧
  order Katya < order Liza ∧
  order Fedya = order Katya + 1 ∧
  order Katya ≠ 1

/-- Theorem stating that Fedya is third based on the given conditions -/
theorem Fedya_third (order : Child → ℕ) (H : arrival_order order) : order Fedya = 3 :=
sorry

end Fedya_third_l1541_154135


namespace total_pushups_l1541_154191

def Zachary_pushups : ℕ := 44
def David_pushups : ℕ := Zachary_pushups + 58

theorem total_pushups : Zachary_pushups + David_pushups = 146 := by
  sorry

end total_pushups_l1541_154191


namespace minimum_club_members_l1541_154139

theorem minimum_club_members : ∃ (b : ℕ), (b = 7) ∧ ∃ (a : ℕ), (2 : ℚ) / 5 < (a : ℚ) / b ∧ (a : ℚ) / b < 1 / 2 := 
sorry

end minimum_club_members_l1541_154139


namespace box_length_is_24_l1541_154154

theorem box_length_is_24 (L : ℕ) (h1 : ∀ s : ℕ, (L * 40 * 16 = 30 * s^3) → s ∣ 40 ∧ s ∣ 16) (h2 : ∃ s : ℕ, s ∣ 40 ∧ s ∣ 16) : L = 24 :=
by
  sorry

end box_length_is_24_l1541_154154


namespace gcd_8251_6105_l1541_154175

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l1541_154175


namespace maximize_x3y4_correct_l1541_154149

noncomputable def maximize_x3y4 : ℝ × ℝ :=
  let x := 160 / 7
  let y := 120 / 7
  (x, y)

theorem maximize_x3y4_correct :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 40 ∧ (x, y) = maximize_x3y4 ∧ 
  ∀ (x' y' : ℝ), 0 < x' ∧ 0 < y' ∧ x' + y' = 40 → x ^ 3 * y ^ 4 ≥ x' ^ 3 * y' ^ 4 :=
by
  sorry

end maximize_x3y4_correct_l1541_154149


namespace time_at_2010_minutes_after_3pm_is_930pm_l1541_154123

def time_after_2010_minutes (current_time : Nat) (minutes_passed : Nat) : Nat :=
  sorry

theorem time_at_2010_minutes_after_3pm_is_930pm :
  time_after_2010_minutes 900 2010 = 1290 :=
by
  sorry

end time_at_2010_minutes_after_3pm_is_930pm_l1541_154123


namespace radius_relation_l1541_154161

-- Define the conditions under which the spheres exist
variable {R r : ℝ}

-- The problem statement
theorem radius_relation (h : r = R * (2 - Real.sqrt 2)) : r = R * (2 - Real.sqrt 2) :=
sorry

end radius_relation_l1541_154161


namespace find_some_number_l1541_154112

theorem find_some_number (d : ℝ) (x : ℝ) (h1 : d = (0.889 * x) / 9.97) (h2 : d = 4.9) :
  x = 54.9 := by
  sorry

end find_some_number_l1541_154112


namespace product_of_digits_base8_of_12345_is_0_l1541_154130

def base8_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else Nat.digits 8 n 

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_digits_base8_of_12345_is_0 :
  product_of_digits (base8_representation 12345) = 0 := 
sorry

end product_of_digits_base8_of_12345_is_0_l1541_154130


namespace part_I_part_II_l1541_154180

-- Part (I)
theorem part_I (a₁ : ℝ) (d : ℝ) (S : ℕ → ℝ) (k : ℕ) :
  a₁ = 3 / 2 →
  d = 1 →
  (∀ n, S n = (n / 2 : ℝ) * (n + 2)) →
  S (k ^ 2) = S k ^ 2 →
  k = 4 :=
by
  intros ha₁ hd hSn hSeq
  sorry

-- Part (II)
theorem part_II (a : ℝ) (d : ℝ) (S : ℕ → ℝ) :
  (∀ k : ℕ, S (k ^ 2) = (S k) ^ 2) →
  ( (∀ n, a = 0 ∧ d = 0 ∧ a + d * (n - 1) = 0) ∨
    (∀ n, a = 1 ∧ d = 0 ∧ a + d * (n - 1) = 1) ∨
    (∀ n, a = 1 ∧ d = 2 ∧ a + d * (n - 1) = 2 * n - 1) ) :=
by
  intros hSeq
  sorry

end part_I_part_II_l1541_154180


namespace pawns_on_black_squares_even_l1541_154141

theorem pawns_on_black_squares_even (A : Fin 8 → Fin 8) :
  ∃ n : ℕ, ∀ i, (i + A i).val % 2 = 1 → n % 2 = 0 :=
sorry

end pawns_on_black_squares_even_l1541_154141


namespace initial_people_on_train_l1541_154171

theorem initial_people_on_train 
    (P : ℕ)
    (h1 : 116 = P - 4)
    (h2 : P = 120)
    : 
    P = 116 + 4 := by
have h3 : P = 120 := by sorry
exact h3

end initial_people_on_train_l1541_154171


namespace apples_jackie_l1541_154109

theorem apples_jackie (A : ℕ) (J : ℕ) (h1 : A = 8) (h2 : J = A + 2) : J = 10 := by
  -- Adam has 8 apples
  sorry

end apples_jackie_l1541_154109


namespace find_a_b_c_l1541_154100

theorem find_a_b_c (a b c : ℝ) 
  (h_min : ∀ x, -9 * x^2 + 54 * x - 45 ≥ 36) 
  (h1 : 0 = a * (1 - 1) * (1 - 5)) 
  (h2 : 0 = a * (5 - 1) * (5 - 5)) :
  a + b + c = 36 :=
sorry

end find_a_b_c_l1541_154100


namespace train_speed_l1541_154117

theorem train_speed (L : ℝ) (T : ℝ) (hL : L = 200) (hT : T = 20) :
  L / T = 10 := by
  rw [hL, hT]
  norm_num
  done

end train_speed_l1541_154117


namespace range_of_a_l1541_154163

theorem range_of_a (a : ℝ) : (¬ (∃ x0 : ℝ, a * x0^2 + x0 + 1/2 ≤ 0)) → a > 1/2 :=
by
  sorry

end range_of_a_l1541_154163


namespace thabo_HNF_calculation_l1541_154170

variable (THABO_BOOKS : ℕ)

-- Conditions as definitions
def total_books : ℕ := 500
def fiction_books : ℕ := total_books * 40 / 100
def non_fiction_books : ℕ := total_books * 60 / 100
def paperback_non_fiction_books (HNF : ℕ) : ℕ := HNF + 50
def total_non_fiction_books (HNF : ℕ) : ℕ := HNF + paperback_non_fiction_books HNF

-- Lean statement to prove
theorem thabo_HNF_calculation (HNF : ℕ) :
  total_books = 500 →
  fiction_books = 200 →
  non_fiction_books = 300 →
  total_non_fiction_books HNF = 300 →
  2 * HNF + 50 = 300 →
  HNF = 125 :=
by
  intros _
         _
         _
         _
         _
  sorry

end thabo_HNF_calculation_l1541_154170


namespace angle_bisector_coordinates_distance_to_x_axis_l1541_154116

structure Point where
  x : ℝ
  y : ℝ

def M (m : ℝ) : Point :=
  ⟨m - 1, 2 * m + 3⟩

theorem angle_bisector_coordinates (m : ℝ) :
  (M m = ⟨-5, -5⟩) ∨ (M m = ⟨-(5/3), 5/3⟩) := sorry

theorem distance_to_x_axis (m : ℝ) :
  (|2 * m + 3| = 1) → (M m = ⟨-2, 1⟩) ∨ (M m = ⟨-3, -1⟩) := sorry

end angle_bisector_coordinates_distance_to_x_axis_l1541_154116


namespace m_range_l1541_154193

noncomputable def otimes (a b : ℝ) : ℝ := 
if a > b then a else b

theorem m_range (m : ℝ) : (otimes (2 * m - 5) 3 = 3) ↔ (m ≤ 4) := by
  sorry

end m_range_l1541_154193


namespace power_of_128_div_7_eq_16_l1541_154176

theorem power_of_128_div_7_eq_16 : (128 : ℝ) ^ (4 / 7) = 16 := by
  sorry

end power_of_128_div_7_eq_16_l1541_154176


namespace calculateDifferentialSavings_l1541_154127

/-- 
Assumptions for the tax brackets and deductions/credits.
-/
def taxBracketsCurrent (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 15 / 100
  else if income ≤ 45000 then
    15000 * 15 / 100 + (income - 15000) * 42 / 100
  else
    15000 * 15 / 100 + (45000 - 15000) * 42 / 100 + (income - 45000) * 50 / 100

def taxBracketsProposed (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 12 / 100
  else if income ≤ 45000 then
    15000 * 12 / 100 + (income - 15000) * 28 / 100
  else
    15000 * 12 / 100 + (45000 - 15000) * 28 / 100 + (income - 45000) * 50 / 100

def standardDeduction : ℕ := 3000
def childrenCredit (num_children : ℕ) : ℕ := num_children * 1000

def taxableIncome (income : ℕ) : ℕ :=
  income - standardDeduction

def totalTaxLiabilityCurrent (income num_children : ℕ) : ℕ :=
  (taxBracketsCurrent (taxableIncome income)) - (childrenCredit num_children)

def totalTaxLiabilityProposed (income num_children : ℕ) : ℕ :=
  (taxBracketsProposed (taxableIncome income)) - (childrenCredit num_children)

def differentialSavings (income num_children : ℕ) : ℕ :=
  totalTaxLiabilityCurrent income num_children - totalTaxLiabilityProposed income num_children

/-- 
Statement of the Lean 4 proof problem.
-/
theorem calculateDifferentialSavings : differentialSavings 34500 2 = 2760 :=
by
  sorry

end calculateDifferentialSavings_l1541_154127


namespace solve_equation_l1541_154122

theorem solve_equation :
  { x : ℝ | x * (x - 3)^2 * (5 - x) = 0 } = {0, 3, 5} :=
by
  sorry

end solve_equation_l1541_154122


namespace mike_passing_percentage_l1541_154179

theorem mike_passing_percentage (scored shortfall max_marks : ℝ) (total_marks := scored + shortfall) :
    scored = 212 →
    shortfall = 28 →
    max_marks = 800 →
    (total_marks / max_marks) * 100 = 30 :=
by
  intros
  sorry

end mike_passing_percentage_l1541_154179


namespace friends_activity_l1541_154153

-- Defining the problem conditions
def total_friends : ℕ := 5
def organizers : ℕ := 3
def managers : ℕ := total_friends - organizers

-- Stating the proof problem
theorem friends_activity (h1 : organizers = 3) (h2 : managers = 2) :
  Nat.choose total_friends organizers = 10 :=
sorry

end friends_activity_l1541_154153


namespace grunters_win_all_5_games_grunters_win_at_least_one_game_l1541_154119

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win all 5 games is 243/1024. --/
theorem grunters_win_all_5_games :
  (3/4)^5 = 243 / 1024 :=
sorry

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win at least one game is 1023/1024. --/
theorem grunters_win_at_least_one_game :
  1 - (1/4)^5 = 1023 / 1024 :=
sorry

end grunters_win_all_5_games_grunters_win_at_least_one_game_l1541_154119


namespace no_solution_system_l1541_154195

theorem no_solution_system : ¬ ∃ (x y z : ℝ), 
  x^2 - 2*y + 2 = 0 ∧ 
  y^2 - 4*z + 3 = 0 ∧ 
  z^2 + 4*x + 4 = 0 := 
by
  sorry

end no_solution_system_l1541_154195


namespace volume_Q3_l1541_154133

def Q0 : ℚ := 8
def delta : ℚ := (1 / 3) ^ 3
def ratio : ℚ := 6 / 27

def Q (i : ℕ) : ℚ :=
  match i with
  | 0 => Q0
  | 1 => Q0 + 4 * delta
  | n + 1 => Q n + delta * (ratio ^ n)

theorem volume_Q3 : Q 3 = 5972 / 729 := 
by
  sorry

end volume_Q3_l1541_154133


namespace geometric_sequence_is_alternating_l1541_154174

theorem geometric_sequence_is_alternating (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = -3 / 2)
  (h2 : a 4 + a 5 = 12)
  (hg : ∀ n, a (n + 1) = q * a n) :
  ∃ q, q < 0 ∧ ∀ n, a n * a (n + 1) ≤ 0 :=
by sorry

end geometric_sequence_is_alternating_l1541_154174


namespace inequality_solution_l1541_154114

theorem inequality_solution (x : ℝ) :
  27 ^ (Real.log x / Real.log 3) ^ 2 - 8 * x ^ (Real.log x / Real.log 3) ≥ 3 ↔
  x ∈ Set.Icc 0 (1 / 3) ∪ Set.Ici 3 :=
sorry

end inequality_solution_l1541_154114


namespace second_train_speed_l1541_154160

theorem second_train_speed
  (v : ℕ)
  (h1 : 8 * v - 8 * 11 = 160) :
  v = 31 :=
sorry

end second_train_speed_l1541_154160


namespace age_ratio_albert_mary_l1541_154186

variable (A M B : ℕ) 

theorem age_ratio_albert_mary
    (h1 : A = 4 * B)
    (h2 : M = A - 10)
    (h3 : B = 5) :
    A = 2 * M :=
by
    sorry

end age_ratio_albert_mary_l1541_154186


namespace correct_answers_max_l1541_154129

def max_correct_answers (c w b : ℕ) : Prop :=
  c + w + b = 25 ∧ 4 * c - 3 * w = 40

theorem correct_answers_max : ∃ c w b : ℕ, max_correct_answers c w b ∧ ∀ c', max_correct_answers c' w b → c' ≤ 13 :=
by
  sorry

end correct_answers_max_l1541_154129


namespace prince_wish_fulfilled_l1541_154132

theorem prince_wish_fulfilled
  (k : ℕ)
  (k_gt_1 : 1 < k)
  (k_lt_13 : k < 13)
  (city : Fin 13 → Fin k) 
  (initial_goblets : Fin k → Fin 13)
  (is_gold : Fin 13 → Bool) :
  ∃ i j : Fin 13, i ≠ j ∧ city i = city j ∧ is_gold i = true ∧ is_gold j = true := 
sorry

end prince_wish_fulfilled_l1541_154132


namespace math_problem_l1541_154150

-- Definitions based on conditions
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

-- Main theorem statement
theorem math_problem :
  avg4 (avg4 2 2 0 2) (avg2 3 1) 0 3 = 13 / 8 :=
by
  sorry

end math_problem_l1541_154150


namespace books_ratio_l1541_154162

theorem books_ratio (c e : ℕ) (h_ratio : c / e = 2 / 5) (h_sampled : c = 10) : e = 25 :=
by
  sorry

end books_ratio_l1541_154162


namespace gcd_factorial_l1541_154142

theorem gcd_factorial (n m l : ℕ) (h1 : n = 7) (h2 : m = 10) (h3 : l = 4): 
  Nat.gcd (Nat.factorial n) (Nat.factorial m / Nat.factorial l) = 2520 :=
by
  sorry

end gcd_factorial_l1541_154142


namespace zero_points_of_function_l1541_154136

theorem zero_points_of_function : 
  (∃ x y : ℝ, y = x - 4 / x ∧ y = 0) → (∃! x : ℝ, x = -2 ∨ x = 2) :=
by
  sorry

end zero_points_of_function_l1541_154136


namespace customers_who_did_not_tip_l1541_154198

def total_customers := 10
def total_tips := 15
def tip_per_customer := 3

theorem customers_who_did_not_tip : total_customers - (total_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_who_did_not_tip_l1541_154198


namespace Raven_age_l1541_154147

-- Define the conditions
def Phoebe_age_current : Nat := 10
def Phoebe_age_in_5_years : Nat := Phoebe_age_current + 5

-- Define the hypothesis that in 5 years Raven will be 4 times as old as Phoebe
def Raven_in_5_years (R : Nat) : Prop := R + 5 = 4 * Phoebe_age_in_5_years

-- State the theorem to be proved
theorem Raven_age : ∃ R : Nat, Raven_in_5_years R ∧ R = 55 :=
by
  sorry

end Raven_age_l1541_154147


namespace problem_l1541_154137

    theorem problem (a b c : ℝ) : 
        a < b → 
        (∀ x : ℝ, (x ≤ -2 ∨ |x - 30| < 2) ↔ (0 ≤ (x - a) * (x - b) / (x - c))) → 
        a + 2 * b + 3 * c = 86 := by 
    sorry

end problem_l1541_154137


namespace teacher_age_is_94_5_l1541_154120

noncomputable def avg_age_students : ℝ := 18
noncomputable def num_students : ℝ := 50
noncomputable def avg_age_class_with_teacher : ℝ := 19.5
noncomputable def num_total : ℝ := 51

noncomputable def total_age_students : ℝ := num_students * avg_age_students
noncomputable def total_age_class_with_teacher : ℝ := num_total * avg_age_class_with_teacher

theorem teacher_age_is_94_5 : ∃ T : ℝ, total_age_students + T = total_age_class_with_teacher ∧ T = 94.5 := by
  sorry

end teacher_age_is_94_5_l1541_154120


namespace prime_gt_three_times_n_l1541_154111

def nth_prime (n : ℕ) : ℕ :=
  -- Define the nth prime function, can use mathlib functionality
  sorry

theorem prime_gt_three_times_n (n : ℕ) (h : 12 ≤ n) : nth_prime n > 3 * n :=
  sorry

end prime_gt_three_times_n_l1541_154111


namespace find_fifth_score_l1541_154183

-- Define the known scores
def score1 : ℕ := 90
def score2 : ℕ := 93
def score3 : ℕ := 85
def score4 : ℕ := 97

-- Define the average of all scores
def average : ℕ := 92

-- Define the total number of scores
def total_scores : ℕ := 5

-- Define the total sum of all scores using the average
def total_sum : ℕ := total_scores * average

-- Define the sum of the four known scores
def known_sum : ℕ := score1 + score2 + score3 + score4

-- Define the fifth score
def fifth_score : ℕ := 95

-- Theorem statement: The fifth score plus the known sum equals the total sum.
theorem find_fifth_score : fifth_score + known_sum = total_sum := by
  sorry

end find_fifth_score_l1541_154183


namespace files_per_folder_l1541_154169

theorem files_per_folder
    (initial_files : ℕ)
    (deleted_files : ℕ)
    (folders : ℕ)
    (remaining_files : ℕ)
    (files_per_folder : ℕ)
    (initial_files_eq : initial_files = 93)
    (deleted_files_eq : deleted_files = 21)
    (folders_eq : folders = 9)
    (remaining_files_eq : remaining_files = initial_files - deleted_files)
    (files_per_folder_eq : files_per_folder = remaining_files / folders) :
    files_per_folder = 8 :=
by
    -- Here, sorry is used to skip the actual proof steps 
    sorry

end files_per_folder_l1541_154169


namespace company_b_profit_l1541_154192

-- Definitions as per problem conditions
def A_profit : ℝ := 90000
def A_share : ℝ := 0.60
def B_share : ℝ := 0.40

-- Theorem statement to be proved
theorem company_b_profit : B_share * (A_profit / A_share) = 60000 :=
by
  sorry

end company_b_profit_l1541_154192


namespace min_value_x_y_l1541_154189

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 19 / x + 98 / y = 1) : x + y ≥ 117 + 14 * Real.sqrt 38 := 
sorry

end min_value_x_y_l1541_154189


namespace min_hypotenuse_of_right_triangle_l1541_154172

theorem min_hypotenuse_of_right_triangle (a b c k : ℝ) (h₁ : k = a + b + c) (h₂ : a^2 + b^2 = c^2) : 
  c ≥ (Real.sqrt 2 - 1) * k := 
sorry

end min_hypotenuse_of_right_triangle_l1541_154172


namespace radian_measure_of_sector_l1541_154104

-- Lean statement for the proof problem
theorem radian_measure_of_sector (R : ℝ) (hR : 0 < R) (h_area : (1 / 2) * (2 : ℝ) * R^2 = R^2) : 
  (2 : ℝ) = 2 :=
by 
  sorry
 
end radian_measure_of_sector_l1541_154104


namespace julios_grape_soda_l1541_154148

variable (a b c d e f g : ℕ)
variable (ha : a = 4)
variable (hc : c = 1)
variable (hd : d = 3)
variable (he : e = 2)
variable (hf : f = 14)
variable (hg : g = 7)

theorem julios_grape_soda : 
  let julios_soda := a * e + b * e
  let mateos_soda := (c + d) * e
  julios_soda = mateos_soda + f
  → b = g := by
  sorry

end julios_grape_soda_l1541_154148


namespace cost_relationship_l1541_154115

variable {α : Type} [LinearOrderedField α]
variables (bananas_cost apples_cost pears_cost : α)

theorem cost_relationship :
  (5 * bananas_cost = 3 * apples_cost) →
  (10 * apples_cost = 6 * pears_cost) →
  (25 * bananas_cost = 9 * pears_cost) := by
  intros h1 h2
  sorry

end cost_relationship_l1541_154115
