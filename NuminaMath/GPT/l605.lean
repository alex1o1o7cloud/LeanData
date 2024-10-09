import Mathlib

namespace shakes_indeterminable_l605_60599

theorem shakes_indeterminable (B S C x : ℝ) (h1 : 3 * B + 7 * S + C = 120) (h2 : 4 * B + x * S + C = 164.50) : ¬ (∃ B S C, ∀ x, 4 * B + x * S + C = 164.50) → false := 
by 
  sorry

end shakes_indeterminable_l605_60599


namespace range_of_a_l605_60568

theorem range_of_a 
  (a : ℝ)
  (H1 : ∀ x : ℝ, -2 < x ∧ x < 3 → -2 < x ∧ x < a)
  (H2 : ¬(∀ x : ℝ, -2 < x ∧ x < a → -2 < x ∧ x < 3)) :
  3 < a :=
by
  sorry

end range_of_a_l605_60568


namespace arithmetic_geom_sequences_l605_60543

theorem arithmetic_geom_sequences
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ q, ∀ n, b (n + 1) = b n * q)
  (h1 : a 2 + a 3 = 14)
  (h2 : a 4 - a 1 = 6)
  (h3 : b 2 = a 1)
  (h4 : b 3 = a 3) :
  (∀ n, a n = 2 * n + 2) ∧ (∃ m, b 6 = a m ∧ m = 31) := sorry

end arithmetic_geom_sequences_l605_60543


namespace circle_and_parabola_no_intersection_l605_60503

theorem circle_and_parabola_no_intersection (m : ℝ) (h : m ≠ 0) :
  (m > 0 ∨ m < -4) ↔
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x = 0) → (y^2 = 4 * m * x) → x ≠ -m := 
sorry

end circle_and_parabola_no_intersection_l605_60503


namespace area_of_shaded_region_l605_60551

open Real

-- Define points and squares
structure Point (α : Type*) := (x : α) (y : α)

def A := Point.mk 0 12 -- top-left corner of large square
def G := Point.mk 0 0  -- bottom-left corner of large square
def F := Point.mk 4 0  -- bottom-right corner of small square
def E := Point.mk 4 4  -- top-right corner of small square
def C := Point.mk 12 0 -- bottom-right corner of large square
def D := Point.mk 3 0  -- intersection of AF extended with the bottom edge

-- Define the length of sides
def side_small_square : ℝ := 4
def side_large_square : ℝ := 12

-- Areas calculation
def area_square (side : ℝ) : ℝ := side * side

def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height

-- Theorem statement
theorem area_of_shaded_region : area_square side_small_square - area_triangle 3 side_small_square = 10 :=
by
  rw [area_square, area_triangle]
  -- Plug in values: 4^2 - 0.5 * 3 * 4
  norm_num
  sorry

end area_of_shaded_region_l605_60551


namespace option2_is_cheaper_l605_60532

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price_option1 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.10
  apply_discount price_after_second_discount 0.05

def final_price_option2 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.05
  apply_discount price_after_second_discount 0.15

theorem option2_is_cheaper (initial_price : ℝ) (h : initial_price = 12000) :
  final_price_option2 initial_price = 6783 ∧ final_price_option1 initial_price = 7182 → 6783 < 7182 :=
by
  intros
  sorry

end option2_is_cheaper_l605_60532


namespace light_travel_distance_120_years_l605_60521

theorem light_travel_distance_120_years :
  let annual_distance : ℝ := 9.46e12
  let years : ℝ := 120
  (annual_distance * years) = 1.1352e15 := 
by
  sorry

end light_travel_distance_120_years_l605_60521


namespace solve_for_x_l605_60536

def delta (x : ℝ) : ℝ := 5 * x + 6
def phi (x : ℝ) : ℝ := 6 * x + 5

theorem solve_for_x : ∀ x : ℝ, delta (phi x) = -1 → x = - 16 / 15 :=
by
  intro x
  intro h
  -- Proof skipped
  sorry

end solve_for_x_l605_60536


namespace find_d_l605_60509

theorem find_d (d : ℕ) : (1059 % d = 1417 % d) ∧ (1059 % d = 2312 % d) ∧ (1417 % d = 2312 % d) ∧ (d > 1) → d = 179 :=
by
  sorry

end find_d_l605_60509


namespace min_distance_from_circle_to_line_l605_60552

noncomputable def circle_center : (ℝ × ℝ) := (3, -1)
noncomputable def circle_radius : ℝ := 2

def on_circle (P : ℝ × ℝ) : Prop := (P.1 - circle_center.1) ^ 2 + (P.2 + circle_center.2) ^ 2 = circle_radius ^ 2
def on_line (Q : ℝ × ℝ) : Prop := Q.1 = -3

theorem min_distance_from_circle_to_line (P Q : ℝ × ℝ)
  (h1 : on_circle P) (h2 : on_line Q) : dist P Q = 4 := 
sorry

end min_distance_from_circle_to_line_l605_60552


namespace combinedHeightCorrect_l605_60556

def empireStateBuildingHeightToTopFloor : ℕ := 1250
def empireStateBuildingAntennaHeight : ℕ := 204

def willisTowerHeightToTopFloor : ℕ := 1450
def willisTowerAntennaHeight : ℕ := 280

def oneWorldTradeCenterHeightToTopFloor : ℕ := 1368
def oneWorldTradeCenterAntennaHeight : ℕ := 408

def totalHeightEmpireStateBuilding := empireStateBuildingHeightToTopFloor + empireStateBuildingAntennaHeight
def totalHeightWillisTower := willisTowerHeightToTopFloor + willisTowerAntennaHeight
def totalHeightOneWorldTradeCenter := oneWorldTradeCenterHeightToTopFloor + oneWorldTradeCenterAntennaHeight

def combinedHeight := totalHeightEmpireStateBuilding + totalHeightWillisTower + totalHeightOneWorldTradeCenter

theorem combinedHeightCorrect : combinedHeight = 4960 := by
  sorry

end combinedHeightCorrect_l605_60556


namespace inverse_value_l605_60561

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value (x : ℝ) (h : g (-3) = x) : (g ∘ g⁻¹) x = x := by
  sorry

end inverse_value_l605_60561


namespace solve_system_l605_60567

theorem solve_system :
  ∃ (x1 y1 x2 y2 x3 y3 : ℚ), 
    (x1 = 0 ∧ y1 = 0) ∧ 
    (x2 = -14 ∧ y2 = 6) ∧ 
    (x3 = -85/6 ∧ y3 = 35/6) ∧ 
    ((x1 + 2*y1)*(x1 + 3*y1) = x1 + y1 ∧ (2*x1 + y1)*(3*x1 + y1) = -99*(x1 + y1)) ∧ 
    ((x2 + 2*y2)*(x2 + 3*y2) = x2 + y2 ∧ (2*x2 + y2)*(3*x2 + y2) = -99*(x2 + y2)) ∧ 
    ((x3 + 2*y3)*(x3 + 3*y3) = x3 + y3 ∧ (2*x3 + y3)*(3*x3 + y3) = -99*(x3 + y3)) :=
by
  -- skips the actual proof
  sorry

end solve_system_l605_60567


namespace exists_integers_for_expression_l605_60531

theorem exists_integers_for_expression (n : ℤ) : 
  ∃ a b c d : ℤ, n = a^2 + b^2 - c^2 - d^2 := 
sorry

end exists_integers_for_expression_l605_60531


namespace width_of_beam_l605_60534

theorem width_of_beam (L W k : ℝ) (h1 : L = k * W) (h2 : 250 = k * 1.5) : 
  (k = 166.6667) → (583.3333 = 166.6667 * W) → W = 3.5 :=
by 
  intro hk1 
  intro h583
  sorry

end width_of_beam_l605_60534


namespace Atlantic_Call_additional_charge_is_0_20_l605_60591

def United_Telephone_base_rate : ℝ := 7.00
def United_Telephone_rate_per_minute : ℝ := 0.25
def Atlantic_Call_base_rate : ℝ := 12.00
def United_Telephone_total_charge_100_minutes : ℝ := United_Telephone_base_rate + 100 * United_Telephone_rate_per_minute
def Atlantic_Call_total_charge_100_minutes (x : ℝ) : ℝ := Atlantic_Call_base_rate + 100 * x

theorem Atlantic_Call_additional_charge_is_0_20 :
  ∃ x : ℝ, United_Telephone_total_charge_100_minutes = Atlantic_Call_total_charge_100_minutes x ∧ x = 0.20 :=
by {
  -- Since United_Telephone_total_charge_100_minutes = 32.00, we need to prove:
  -- Atlantic_Call_total_charge_100_minutes 0.20 = 32.00
  sorry
}

end Atlantic_Call_additional_charge_is_0_20_l605_60591


namespace probability_two_roads_at_least_5_miles_long_l605_60588

-- Probabilities of roads being at least 5 miles long
def prob_A_B := 3 / 4
def prob_B_C := 2 / 3
def prob_C_D := 1 / 2

-- Theorem: Probability of at least two roads being at least 5 miles long
theorem probability_two_roads_at_least_5_miles_long :
  prob_A_B * prob_B_C * (1 - prob_C_D) +
  prob_A_B * prob_C_D * (1 - prob_B_C) +
  (1 - prob_A_B) * prob_B_C * prob_C_D +
  prob_A_B * prob_B_C * prob_C_D = 11 / 24 := 
by
  sorry -- Proof goes here

end probability_two_roads_at_least_5_miles_long_l605_60588


namespace slope_of_line_l605_60538

-- Defining the parametric equations of the line
def parametric_x (t : ℝ) : ℝ := 3 + 4 * t
def parametric_y (t : ℝ) : ℝ := 4 - 5 * t

-- Stating the problem in Lean: asserting the slope of the line
theorem slope_of_line : 
  (∃ (m : ℝ), ∀ t : ℝ, parametric_y t = m * parametric_x t + (4 - 3 * m)) 
  → (∃ m : ℝ, m = -5 / 4) :=
  by sorry

end slope_of_line_l605_60538


namespace brianna_sandwiches_l605_60516

theorem brianna_sandwiches (meats : ℕ) (cheeses : ℕ) (h_meats : meats = 8) (h_cheeses : cheeses = 7) :
  (Nat.choose meats 2) * (Nat.choose cheeses 1) = 196 := 
by
  rw [h_meats, h_cheeses]
  norm_num
  sorry

end brianna_sandwiches_l605_60516


namespace problem_solution_l605_60525

theorem problem_solution : (3.242 * 14) / 100 = 0.45388 := by
  sorry

end problem_solution_l605_60525


namespace four_n_div_four_remainder_zero_l605_60564

theorem four_n_div_four_remainder_zero (n : ℤ) (h : n % 4 = 3) : (4 * n) % 4 = 0 := 
by
  sorry

end four_n_div_four_remainder_zero_l605_60564


namespace percent_spent_on_other_items_l605_60579

def total_amount_spent (T : ℝ) : ℝ := T
def clothing_percent (p : ℝ) : Prop := p = 0.45
def food_percent (p : ℝ) : Prop := p = 0.45
def clothing_tax (t : ℝ) (T : ℝ) : ℝ := 0.05 * (0.45 * T)
def food_tax (t : ℝ) (T : ℝ) : ℝ := 0.0 * (0.45 * T)
def other_items_tax (p : ℝ) (T : ℝ) : ℝ := 0.10 * (p * T)
def total_tax (T : ℝ) (tax : ℝ) : Prop := tax = 0.0325 * T

theorem percent_spent_on_other_items (T : ℝ) (p_clothing p_food x : ℝ) (tax : ℝ) 
  (h1 : clothing_percent p_clothing) (h2 : food_percent p_food)
  (h3 : clothing_tax tax T = 0.05 * (0.45 * T))
  (h4 : food_tax tax T = 0.0)
  (h5 : other_items_tax x T = 0.10 * (x * T))
  (h6 : total_tax T (clothing_tax tax T + food_tax tax T + other_items_tax x T)) : 
  x = 0.10 :=
by
  sorry

end percent_spent_on_other_items_l605_60579


namespace smallest_integral_value_of_y_l605_60514

theorem smallest_integral_value_of_y :
  ∃ y : ℤ, (1 / 4 : ℝ) < y / 7 ∧ y / 7 < 2 / 3 ∧ ∀ z : ℤ, (1 / 4 : ℝ) < z / 7 ∧ z / 7 < 2 / 3 → y ≤ z :=
by
  -- The statement is defined and the proof is left as "sorry" to illustrate that no solution steps are used directly.
  sorry

end smallest_integral_value_of_y_l605_60514


namespace factorization_of_polynomial_l605_60565

theorem factorization_of_polynomial : 
  (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by sorry

end factorization_of_polynomial_l605_60565


namespace xyz_equality_l605_60560

theorem xyz_equality (x y z : ℝ) (h : x^2 + y^2 + z^2 = x * y + y * z + z * x) : x = y ∧ y = z :=
by
  sorry

end xyz_equality_l605_60560


namespace point_P_in_first_quadrant_l605_60502

def lies_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : lies_in_first_quadrant 2 1 :=
by {
  sorry
}

end point_P_in_first_quadrant_l605_60502


namespace smallest_x_consecutive_cubes_l605_60513

theorem smallest_x_consecutive_cubes :
  ∃ (u v w x : ℕ), u < v ∧ v < w ∧ w < x ∧ u + 1 = v ∧ v + 1 = w ∧ w + 1 = x ∧ (u^3 + v^3 + w^3 = x^3) ∧ (x = 6) :=
by {
  sorry
}

end smallest_x_consecutive_cubes_l605_60513


namespace cost_of_socks_l605_60540

/-- Given initial amount of $100 and cost of shirt is $24,
    find out the cost of socks if the remaining amount is $65. --/
theorem cost_of_socks
  (initial_amount : ℕ)
  (cost_of_shirt : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 100)
  (h2 : cost_of_shirt = 24)
  (h3 : remaining_amount = 65) : 
  (initial_amount - cost_of_shirt - remaining_amount) = 11 :=
by
  sorry

end cost_of_socks_l605_60540


namespace tan_frac_eq_one_l605_60501

open Real

-- Conditions given in the problem
def sin_frac_cond (x y : ℝ) : Prop := (sin x / sin y) + (sin y / sin x) = 4
def cos_frac_cond (x y : ℝ) : Prop := (cos x / cos y) + (cos y / cos x) = 3

-- Statement of the theorem to be proved
theorem tan_frac_eq_one (x y : ℝ) (h1 : sin_frac_cond x y) (h2 : cos_frac_cond x y) : (tan x / tan y) + (tan y / tan x) = 1 :=
by
  sorry

end tan_frac_eq_one_l605_60501


namespace time_to_complete_together_l605_60563

-- Definitions for the given conditions
variables (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Theorem statement for the mathematically equivalent proof problem
theorem time_to_complete_together (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
   (1 : ℝ) / ((1 / x) + (1 / y)) = x * y / (x + y) :=
sorry

end time_to_complete_together_l605_60563


namespace lower_limit_for_a_l605_60589

theorem lower_limit_for_a 
  {k : ℤ} 
  (a b : ℤ) 
  (h1 : k ≤ a) 
  (h2 : a < 17) 
  (h3 : 3 < b) 
  (h4 : b < 29) 
  (h5 : 3.75 = 4 - 0.25) 
  : (7 ≤ a) :=
sorry

end lower_limit_for_a_l605_60589


namespace greatest_cds_in_box_l605_60546

theorem greatest_cds_in_box (r c p n : ℕ) (hr : r = 14) (hc : c = 12) (hp : p = 8) (hn : n = 2) :
  n = Nat.gcd r (Nat.gcd c p) :=
by
  rw [hr, hc, hp]
  sorry

end greatest_cds_in_box_l605_60546


namespace sum_of_decimals_is_fraction_l605_60519

def decimal_to_fraction_sum : ℚ :=
  (1 / 10) + (2 / 100) + (3 / 1000) + (4 / 10000) + (5 / 100000) + (6 / 1000000) + (7 / 10000000)

theorem sum_of_decimals_is_fraction :
  decimal_to_fraction_sum = 1234567 / 10000000 :=
by sorry

end sum_of_decimals_is_fraction_l605_60519


namespace octagon_edge_length_from_pentagon_l605_60508

noncomputable def regular_pentagon_edge_length : ℝ := 16
def num_of_pentagon_edges : ℕ := 5
def num_of_octagon_edges : ℕ := 8

theorem octagon_edge_length_from_pentagon (total_length_thread : ℝ) :
  total_length_thread = num_of_pentagon_edges * regular_pentagon_edge_length →
  (total_length_thread / num_of_octagon_edges) = 10 :=
by
  intro h
  sorry

end octagon_edge_length_from_pentagon_l605_60508


namespace max_remainder_l605_60530

-- Definition of the problem
def max_remainder_condition (x : ℕ) (y : ℕ) : Prop :=
  x % 7 = y

theorem max_remainder (y : ℕ) :
  (max_remainder_condition (7 * 102 + y) y ∧ y < 7) → (y = 6 ∧ 7 * 102 + 6 = 720) :=
by
  sorry

end max_remainder_l605_60530


namespace age_problem_l605_60578

theorem age_problem 
  (K S E F : ℕ)
  (h1 : K = S - 5)
  (h2 : S = 2 * E)
  (h3 : E = F + 9)
  (h4 : K = 33) : 
  F = 10 :=
by 
  sorry

end age_problem_l605_60578


namespace ratio_of_larger_to_smaller_l605_60541

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x > y) (h4 : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end ratio_of_larger_to_smaller_l605_60541


namespace smallest_prime_reversing_to_composite_l605_60511

theorem smallest_prime_reversing_to_composite (p : ℕ) :
  p = 23 ↔ (p < 100 ∧ p ≥ 10 ∧ Nat.Prime p ∧ 
  ∃ c, c < 100 ∧ c ≥ 10 ∧ ¬ Nat.Prime c ∧ c = (p % 10) * 10 + p / 10 ∧ (p / 10 = 2 ∨ p / 10 = 3)) :=
by
  sorry

end smallest_prime_reversing_to_composite_l605_60511


namespace at_least_one_less_than_two_l605_60584

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
by 
  sorry

end at_least_one_less_than_two_l605_60584


namespace simplify_and_evaluate_expression_l605_60548

variables (m n : ℚ)

theorem simplify_and_evaluate_expression (h1 : m = -1) (h2 : n = 1 / 2) :
  ( (2 / (m - n) - 1 / (m + n)) / ((m * n + 3 * n ^ 2) / (m ^ 3 - m * n ^ 2)) ) = -2 :=
by
  sorry

end simplify_and_evaluate_expression_l605_60548


namespace portion_of_money_given_to_Blake_l605_60598

theorem portion_of_money_given_to_Blake
  (initial_amount : ℝ)
  (tripled_amount : ℝ)
  (sale_amount : ℝ)
  (amount_given_to_Blake : ℝ)
  (h1 : initial_amount = 20000)
  (h2 : tripled_amount = 3 * initial_amount)
  (h3 : sale_amount = tripled_amount)
  (h4 : amount_given_to_Blake = 30000) :
  amount_given_to_Blake / sale_amount = 1 / 2 :=
sorry

end portion_of_money_given_to_Blake_l605_60598


namespace walking_west_is_negative_l605_60573

-- Definitions based on conditions
def east (m : Int) : Int := m
def west (m : Int) : Int := -m

-- Proof statement (no proof required, so use "sorry")
theorem walking_west_is_negative (m : Int) (h : east 8 = 8) : west 10 = -10 :=
by
  sorry

end walking_west_is_negative_l605_60573


namespace linear_equation_in_two_variables_l605_60572

def is_linear_equation_two_variables (eq : String → Prop) : Prop :=
  eq "D"

-- Given Conditions
def eqA (x y z : ℝ) : Prop := 2 * x + 3 * y = z
def eqB (x y : ℝ) : Prop := 4 / x + y = 5
def eqC (x y : ℝ) : Prop := 1 / 2 * x^2 + y = 0
def eqD (x y : ℝ) : Prop := y = 1 / 2 * (x + 8)

-- Problem Statement to be Proved
theorem linear_equation_in_two_variables :
  is_linear_equation_two_variables (λ s =>
    ∃ x y z : ℝ, 
      (s = "A" → eqA x y z) ∨ 
      (s = "B" → eqB x y) ∨ 
      (s = "C" → eqC x y) ∨ 
      (s = "D" → eqD x y)
  ) :=
sorry

end linear_equation_in_two_variables_l605_60572


namespace intersection_complement_is_l605_60507

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem intersection_complement_is :
  N ∩ (U \ M) = {3, 5} :=
  sorry

end intersection_complement_is_l605_60507


namespace complex_number_solution_l605_60518

theorem complex_number_solution (a b : ℤ) (z : ℂ) (h1 : z = a + b * Complex.I) (h2 : z^3 = 2 + 11 * Complex.I) : a + b = 3 :=
sorry

end complex_number_solution_l605_60518


namespace imaginary_part_of_fraction_l605_60523

open Complex

theorem imaginary_part_of_fraction :
  let z := (5 * Complex.I) / (1 + 2 * Complex.I)
  z.im = 1 :=
by
  let z := (5 * Complex.I) / (1 + 2 * Complex.I)
  show z.im = 1
  sorry

end imaginary_part_of_fraction_l605_60523


namespace correct_propositions_l605_60517

variable (A : Set ℝ)
variable (oplus : ℝ → ℝ → ℝ)

def condition_a1 : Prop := ∀ a b : ℝ, a ∈ A → b ∈ A → (oplus a b) ∈ A
def condition_a2 : Prop := ∀ a : ℝ, a ∈ A → (oplus a a) = 0
def condition_a3 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus (oplus a b) c) = (oplus a c) + (oplus b c) + c

def proposition_1 : Prop := 0 ∈ A
def proposition_2 : Prop := (1 ∈ A) → (oplus (oplus 1 1) 1) = 0
def proposition_3 : Prop := ∀ a : ℝ, a ∈ A → (oplus a 0) = a → a = 0
def proposition_4 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus a 0) = a → (oplus a b) = (oplus c b) → a = c

theorem correct_propositions 
  (h1 : condition_a1 A oplus) 
  (h2 : condition_a2 A oplus)
  (h3 : condition_a3 A oplus) : 
  (proposition_1 A) ∧ (¬proposition_2 A oplus) ∧ (proposition_3 A oplus) ∧ (proposition_4 A oplus) := by
  sorry

end correct_propositions_l605_60517


namespace fraction_of_hidden_sea_is_five_over_eight_l605_60512

noncomputable def cloud_fraction := 1 / 2
noncomputable def island_uncovered_fraction := 1 / 4 
noncomputable def island_covered_fraction := island_uncovered_fraction / (1 - cloud_fraction)

-- The total island area is the sum of covered and uncovered.
noncomputable def total_island_fraction := island_uncovered_fraction + island_covered_fraction 

-- The sea area covered by the cloud is half minus the fraction of the island covered by the cloud.
noncomputable def sea_covered_by_cloud := cloud_fraction - island_covered_fraction 

-- The sea occupies the remainder of the landscape not taken by the uncoveed island.
noncomputable def total_sea_fraction := 1 - island_uncovered_fraction - cloud_fraction + island_covered_fraction 

-- The sea fraction visible and not covered by clouds
noncomputable def sea_visible_not_covered := total_sea_fraction - sea_covered_by_cloud 

-- The fraction of the sea hidden by the cloud
noncomputable def sea_fraction_hidden_by_cloud := sea_covered_by_cloud / total_sea_fraction 

theorem fraction_of_hidden_sea_is_five_over_eight : sea_fraction_hidden_by_cloud = 5 / 8 := 
by
  sorry

end fraction_of_hidden_sea_is_five_over_eight_l605_60512


namespace sum_of_fourth_powers_l605_60580

-- Define the sum of fourth powers as per the given formula
noncomputable def sum_fourth_powers (n: ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30

-- Define the statement to be proved
theorem sum_of_fourth_powers :
  2 * sum_fourth_powers 100 = 41006666600 :=
by sorry

end sum_of_fourth_powers_l605_60580


namespace double_probability_correct_l605_60555

def is_double (a : ℕ × ℕ) : Prop := a.1 = a.2

def total_dominoes : ℕ := 13 * 13

def double_count : ℕ := 13

def double_probability := (double_count : ℚ) / total_dominoes

theorem double_probability_correct : double_probability = 13 / 169 := by
  sorry

end double_probability_correct_l605_60555


namespace number_of_ways_to_distribute_balls_l605_60515

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end number_of_ways_to_distribute_balls_l605_60515


namespace number_of_valid_pairs_l605_60582

theorem number_of_valid_pairs :
  ∃ (n : Nat), n = 8 ∧ 
  (∃ (a b : Int), 4 < a ∧ a < b ∧ b < 22 ∧ (4 + a + b + 22) / 4 = 13) :=
sorry

end number_of_valid_pairs_l605_60582


namespace required_run_rate_l605_60596

def initial_run_rate : ℝ := 3.2
def overs_completed : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 50

theorem required_run_rate :
  (target_runs - initial_run_rate * overs_completed) / remaining_overs = 5 := 
by
  sorry

end required_run_rate_l605_60596


namespace quadratic_roots_min_value_l605_60553

theorem quadratic_roots_min_value (m α β : ℝ) (h_eq : 4 * α^2 - 4 * m * α + m + 2 = 0) (h_eq2 : 4 * β^2 - 4 * m * β + m + 2 = 0) :
  (∃ m_val : ℝ, m_val = -1 ∧ α^2 + β^2 = 1 / 2) :=
by
  sorry

end quadratic_roots_min_value_l605_60553


namespace percent_decrease_second_year_l605_60524

theorem percent_decrease_second_year
  (V_0 V_1 V_2 : ℝ)
  (p_2 : ℝ)
  (h1 : V_1 = V_0 * 0.7)
  (h2 : V_2 = V_1 * (1 - p_2 / 100))
  (h3 : V_2 = V_0 * 0.63) :
  p_2 = 10 :=
sorry

end percent_decrease_second_year_l605_60524


namespace complement_union_l605_60570

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 4}

theorem complement_union : U \ (A ∪ B) = {3, 5} :=
by
  sorry

end complement_union_l605_60570


namespace max_discount_rate_l605_60571

def cost_price : ℝ := 4
def selling_price : ℝ := 5
def min_profit_margin : ℝ := 0.1 * cost_price

theorem max_discount_rate (x : ℝ) (hx : 0 ≤ x) :
  (selling_price * (1 - x / 100) - cost_price ≥ min_profit_margin) → x ≤ 12 :=
sorry

end max_discount_rate_l605_60571


namespace smallest_fourth_number_l605_60585

-- Define the given conditions
def first_three_numbers_sum : ℕ := 28 + 46 + 59 
def sum_of_digits_of_first_three_numbers : ℕ := 2 + 8 + 4 + 6 + 5 + 9 

-- Define the condition for the fourth number represented as 10a + b and its digits 
def satisfies_condition (a b : ℕ) : Prop := 
  first_three_numbers_sum + 10 * a + b = 4 * (sum_of_digits_of_first_three_numbers + a + b)

-- Statement to prove the smallest fourth number
theorem smallest_fourth_number : ∃ (a b : ℕ), satisfies_condition a b ∧ 10 * a + b = 11 := 
sorry

end smallest_fourth_number_l605_60585


namespace polynomial_sum_l605_60504

theorem polynomial_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 + 1 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 33 :=
by
  sorry

end polynomial_sum_l605_60504


namespace largest_pos_int_divisor_l605_60569

theorem largest_pos_int_divisor:
  ∃ n : ℕ, (n + 10 ∣ n^3 + 2011) ∧ (∀ m : ℕ, (m + 10 ∣ m^3 + 2011) → m ≤ n) :=
sorry

end largest_pos_int_divisor_l605_60569


namespace store_profit_is_20_percent_l605_60528

variable (C : ℝ)
variable (marked_up_price : ℝ := 1.20 * C)          -- First markup price
variable (new_year_price : ℝ := 1.50 * C)           -- Second markup price
variable (discounted_price : ℝ := 1.20 * C)         -- Discounted price in February
variable (profit : ℝ := discounted_price - C)       -- Profit on items sold in February

theorem store_profit_is_20_percent (C : ℝ) : profit = 0.20 * C := 
  sorry

end store_profit_is_20_percent_l605_60528


namespace necessary_not_sufficient_cond_l605_60554

theorem necessary_not_sufficient_cond (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y < 4 → xy < 4) ∧ ¬(xy < 4 → x + y < 4) :=
  by
    sorry

end necessary_not_sufficient_cond_l605_60554


namespace poly_coeff_difference_l605_60506

theorem poly_coeff_difference :
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (2 + x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 →
  a = 16 →
  1 = a - a_1 + a_2 - a_3 + a_4 →
  a_2 - a_1 + a_4 - a_3 = -15 :=
by
  intros a a_1 a_2 a_3 a_4 h_poly h_a h_eq
  sorry

end poly_coeff_difference_l605_60506


namespace union_of_A_and_B_l605_60587

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B : Set ℝ := {x | x < 3}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 3} := by
  sorry

end union_of_A_and_B_l605_60587


namespace evaluate_expression_l605_60537

theorem evaluate_expression : 
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end evaluate_expression_l605_60537


namespace intersection_of_A_and_B_l605_60557

def set_A : Set ℝ := {x | x >= 1 ∨ x <= -2}
def set_B : Set ℝ := {x | -3 < x ∧ x < 2}

def set_C : Set ℝ := {x | (-3 < x ∧ x <= -2) ∨ (1 <= x ∧ x < 2)}

theorem intersection_of_A_and_B (x : ℝ) : x ∈ set_A ∧ x ∈ set_B ↔ x ∈ set_C :=
  by
  sorry

end intersection_of_A_and_B_l605_60557


namespace power_of_p_in_product_l605_60597

theorem power_of_p_in_product (p q : ℕ) (x : ℕ) (hp : Prime p) (hq : Prime q) 
  (h : (x + 1) * 6 = 30) : x = 4 := 
by sorry

end power_of_p_in_product_l605_60597


namespace point_not_on_graph_l605_60535

theorem point_not_on_graph : 
  ∀ (k : ℝ), (k ≠ 0) → (∀ x y : ℝ, y = k * x → (x, y) = (1, 2)) → ¬ (∀ x y : ℝ, y = k * x → (x, y) = (1, -2)) :=
by
  sorry

end point_not_on_graph_l605_60535


namespace sarah_score_l605_60526

theorem sarah_score (j g s : ℕ) 
  (h1 : g = 2 * j) 
  (h2 : s = g + 50) 
  (h3 : (s + g + j) / 3 = 110) : 
  s = 162 := 
by 
  sorry

end sarah_score_l605_60526


namespace exactly_two_pass_probability_l605_60574

theorem exactly_two_pass_probability (PA PB PC : ℚ) (hPA : PA = 2 / 3) (hPB : PB = 3 / 4) (hPC : PC = 2 / 5) :
  ((PA * PB * (1 - PC)) + (PA * (1 - PB) * PC) + ((1 - PA) * PB * PC) = 7 / 15) := by
  sorry

end exactly_two_pass_probability_l605_60574


namespace smallest_two_digit_number_l605_60505

theorem smallest_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100)
  (h2 : ∃ k : ℕ, (N - (N / 10 + (N % 10) * 10)) = k ∧ k > 0 ∧ (∃ m : ℕ, k = m * m))
  : N = 90 := 
sorry

end smallest_two_digit_number_l605_60505


namespace john_sarah_money_total_l605_60533

theorem john_sarah_money_total (j_money s_money : ℚ) (H1 : j_money = 5/8) (H2 : s_money = 7/16) :
  (j_money + s_money : ℚ) = 1.0625 := 
by
  sorry

end john_sarah_money_total_l605_60533


namespace value_of_C_is_2_l605_60586

def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0
def isDivisibleBy7 (n : ℕ) : Prop := n % 7 = 0

def sumOfDigitsFirstNumber (A B : ℕ) : ℕ := 6 + 5 + A + 3 + 1 + B + 4
def sumOfDigitsSecondNumber (A B C : ℕ) : ℕ := 4 + 1 + 7 + A + B + 5 + C

theorem value_of_C_is_2 (A B : ℕ) (hDiv3First : isDivisibleBy3 (sumOfDigitsFirstNumber A B))
  (hDiv7First : isDivisibleBy7 (sumOfDigitsFirstNumber A B))
  (hDiv3Second : isDivisibleBy3 (sumOfDigitsSecondNumber A B 2))
  (hDiv7Second : isDivisibleBy7 (sumOfDigitsSecondNumber A B 2)) : 
  (∃ (C : ℕ), C = 2) :=
sorry

end value_of_C_is_2_l605_60586


namespace floor_sqrt_50_l605_60522

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end floor_sqrt_50_l605_60522


namespace product_units_digit_of_five_consecutive_l605_60558

theorem product_units_digit_of_five_consecutive (n : ℕ) : 
  ((n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10) = 0 := 
sorry

end product_units_digit_of_five_consecutive_l605_60558


namespace angle_B_in_triangle_l605_60583

theorem angle_B_in_triangle
  (a b c : ℝ)
  (h_area : 2 * (a * c * ((a^2 + c^2 - b^2) / (2 * a * c)).sin) = (a^2 + c^2 - b^2) * (Real.sqrt 3 / 6)) :
  ∃ B : ℝ, B = π / 6 :=
by
  sorry

end angle_B_in_triangle_l605_60583


namespace linear_equation_with_two_variables_l605_60500

def equation (a x y : ℝ) : ℝ := (a^2 - 4) * x^2 + (2 - 3 * a) * x + (a + 1) * y + 3 * a

theorem linear_equation_with_two_variables (a : ℝ) :
  (equation a x y = 0) ∧ (a^2 - 4 = 0) ∧ (2 - 3 * a ≠ 0) ∧ (a + 1 ≠ 0) →
  (a = 2 ∨ a = -2) :=
by sorry

end linear_equation_with_two_variables_l605_60500


namespace time_to_cover_same_distance_l605_60593

theorem time_to_cover_same_distance
  (a b c d : ℕ) (k : ℕ) 
  (h_k : k = 3) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_speed_eq : 3 * (a + 2 * b) = 3 * a - b) : 
  (a + 2 * b) * (c + d) / (3 * a - b) = (a + 2 * b) * (c + d) / (3 * a - b) :=
by sorry

end time_to_cover_same_distance_l605_60593


namespace number_of_first_grade_students_l605_60550

noncomputable def sampling_ratio (total_students : ℕ) (sampled_students : ℕ) : ℚ :=
  sampled_students / total_students

noncomputable def num_first_grade_selected (first_grade_students : ℕ) (ratio : ℚ) : ℚ :=
  ratio * first_grade_students

theorem number_of_first_grade_students
  (total_students : ℕ)
  (sampled_students : ℕ)
  (first_grade_students : ℕ)
  (h_total : total_students = 2400)
  (h_sampled : sampled_students = 100)
  (h_first_grade : first_grade_students = 840)
  : num_first_grade_selected first_grade_students (sampling_ratio total_students sampled_students) = 35 := by
  sorry

end number_of_first_grade_students_l605_60550


namespace problem_I_problem_II_l605_60562

theorem problem_I (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) :
  c / a = 2 :=
sorry

theorem problem_II (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) 
  (h3 : b = 4) (h4 : Real.cos C = 1 / 4) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
sorry

end problem_I_problem_II_l605_60562


namespace proof_l605_60544

noncomputable def line_standard_form (t : ℝ) : Prop :=
  let (x, y) := (t + 3, 3 - t)
  x + y = 6

noncomputable def circle_standard_form (θ : ℝ) : Prop :=
  let (x, y) := (2 * Real.cos θ, 2 * Real.sin θ + 2)
  x^2 + (y - 2)^2 = 4

noncomputable def distance_center_to_line (x1 y1 : ℝ) : ℝ :=
  let (a, b, c) := (1, 1, -6)
  let num := abs (a * x1 + b * y1 + c)
  let denom := Real.sqrt (a^2 + b^2)
  num / denom

theorem proof : 
  (∀ t, line_standard_form t) ∧ 
  (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → circle_standard_form θ) ∧ 
  distance_center_to_line 0 2 = 2 * Real.sqrt 2 :=
by
  sorry

end proof_l605_60544


namespace complete_square_solution_l605_60575

theorem complete_square_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : (x - 2)^2 = 2 := 
by sorry

end complete_square_solution_l605_60575


namespace f_increasing_on_Ioo_l605_60529

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_increasing_on_Ioo : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 < f x2 :=
by sorry

end f_increasing_on_Ioo_l605_60529


namespace relationship_between_y1_y2_l605_60577

theorem relationship_between_y1_y2 
  (y1 y2 : ℝ) 
  (hA : y1 = 6 / -3) 
  (hB : y2 = 6 / 2) : y1 < y2 :=
by 
  sorry

end relationship_between_y1_y2_l605_60577


namespace find_pairs_xy_l605_60539

theorem find_pairs_xy (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : 7^x - 3 * 2^y = 1) : 
  (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
sorry

end find_pairs_xy_l605_60539


namespace elena_fraction_left_l605_60595

variable (M : ℝ) -- Total amount of money
variable (B : ℝ) -- Total cost of all the books

-- Condition: Elena spends one-third of her money to buy half of the books
def condition : Prop := (1 / 3) * M = (1 / 2) * B

-- Goal: Fraction of the money left after buying all the books is one-third
theorem elena_fraction_left (h : condition M B) : (M - B) / M = 1 / 3 :=
by
  sorry

end elena_fraction_left_l605_60595


namespace parabola_directrix_l605_60592

theorem parabola_directrix (x y : ℝ) (h : y = - (1/8) * x^2) : y = 2 :=
sorry

end parabola_directrix_l605_60592


namespace longest_path_is_critical_path_l605_60549

noncomputable def longest_path_in_workflow_diagram : String :=
"Critical Path"

theorem longest_path_is_critical_path :
  (longest_path_in_workflow_diagram = "Critical Path") :=
  by
  sorry

end longest_path_is_critical_path_l605_60549


namespace mean_proportional_49_64_l605_60576

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end mean_proportional_49_64_l605_60576


namespace kayak_rental_cost_l605_60542

variable (K : ℕ) -- the cost of a kayak rental per day
variable (x : ℕ) -- the number of kayaks rented

-- Conditions
def canoe_cost_per_day : ℕ := 11
def total_revenue : ℕ := 460
def canoes_more_than_kayaks : ℕ := 5

def ratio_condition : Prop := 4 * x = 3 * (x + 5)
def total_revenue_condition : Prop := canoe_cost_per_day * (x + 5) + K * x = total_revenue

-- Main statement
theorem kayak_rental_cost :
  ratio_condition x →
  total_revenue_condition K x →
  K = 16 := by sorry

end kayak_rental_cost_l605_60542


namespace andrew_total_appeizers_count_l605_60527

theorem andrew_total_appeizers_count :
  let hotdogs := 30
  let cheese_pops := 20
  let chicken_nuggets := 40
  hotdogs + cheese_pops + chicken_nuggets = 90 := 
by 
  sorry

end andrew_total_appeizers_count_l605_60527


namespace boy_lap_time_l605_60520

noncomputable def muddy_speed : ℝ := 5 * 1000 / 3600
noncomputable def sandy_speed : ℝ := 7 * 1000 / 3600
noncomputable def uphill_speed : ℝ := 4 * 1000 / 3600

noncomputable def muddy_distance : ℝ := 10
noncomputable def sandy_distance : ℝ := 15
noncomputable def uphill_distance : ℝ := 10

noncomputable def time_for_muddy : ℝ := muddy_distance / muddy_speed
noncomputable def time_for_sandy : ℝ := sandy_distance / sandy_speed
noncomputable def time_for_uphill : ℝ := uphill_distance / uphill_speed

noncomputable def total_time_for_one_side : ℝ := time_for_muddy + time_for_sandy + time_for_uphill
noncomputable def total_time_for_lap : ℝ := 4 * total_time_for_one_side

theorem boy_lap_time : total_time_for_lap = 95.656 := by
  sorry

end boy_lap_time_l605_60520


namespace equation_1_solution_1_equation_2_solution_l605_60590

theorem equation_1_solution_1 (x : ℝ) (h : 4 * (x - 1) ^ 2 = 25) : x = 7 / 2 ∨ x = -3 / 2 := by
  sorry

theorem equation_2_solution (x : ℝ) (h : (1 / 3) * (x + 2) ^ 3 - 9 = 0) : x = 1 := by
  sorry

end equation_1_solution_1_equation_2_solution_l605_60590


namespace sum_of_coordinates_of_B_is_zero_l605_60594

structure Point where
  x : Int
  y : Int

def translation_to_right (P : Point) (n : Int) : Point :=
  { x := P.x + n, y := P.y }

def translation_down (P : Point) (n : Int) : Point :=
  { x := P.x, y := P.y - n }

def A : Point := { x := -1, y := 2 }

def B : Point := translation_down (translation_to_right A 1) 2

theorem sum_of_coordinates_of_B_is_zero :
  B.x + B.y = 0 := by
  sorry

end sum_of_coordinates_of_B_is_zero_l605_60594


namespace probability_max_roll_correct_l605_60581
open Classical

noncomputable def probability_max_roll_fourth : ℚ :=
  let six_sided_max := 1 / 6
  let eight_sided_max := 3 / 4
  let ten_sided_max := 4 / 5

  let prob_A_given_B1 := (1 / 6) ^ 3
  let prob_A_given_B2 := (3 / 4) ^ 3
  let prob_A_given_B3 := (4 / 5) ^ 3

  let prob_B1 := 1 / 3
  let prob_B2 := 1 / 3
  let prob_B3 := 1 / 3

  let prob_A := prob_A_given_B1 * prob_B1 + prob_A_given_B2 * prob_B2 + prob_A_given_B3 * prob_B3

  -- Calculate probabilities with Bayes' Theorem
  let P_B1_A := (prob_A_given_B1 * prob_B1) / prob_A
  let P_B2_A := (prob_A_given_B2 * prob_B2) / prob_A
  let P_B3_A := (prob_A_given_B3 * prob_B3) / prob_A

  -- Probability of the fourth roll showing the maximum face value
  P_B1_A * six_sided_max + P_B2_A * eight_sided_max + P_B3_A * ten_sided_max

theorem probability_max_roll_correct : 
  ∃ (p q : ℕ), probability_max_roll_fourth = p / q ∧ Nat.gcd p q = 1 ∧ p + q = 4386 :=
by sorry

end probability_max_roll_correct_l605_60581


namespace sin_complementary_angle_l605_60545

theorem sin_complementary_angle (θ : ℝ) (h1 : Real.tan θ = 2) (h2 : Real.cos θ < 0) : 
  Real.sin (Real.pi / 2 - θ) = -Real.sqrt 5 / 5 :=
sorry

end sin_complementary_angle_l605_60545


namespace inequality_x_solution_l605_60566

theorem inequality_x_solution (a b c d x : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  ( (a^3 / (a^3 + 15 * b * c * d))^(1/2) = a^x / (a^x + b^x + c^x + d^x) ) ↔ x = 15 / 8 := 
sorry

end inequality_x_solution_l605_60566


namespace neg_prop_true_l605_60510

theorem neg_prop_true (a : ℝ) :
  ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) → ∃ a : ℝ, a > 2 ∧ a^2 ≥ 4 :=
by
  intros h
  sorry

end neg_prop_true_l605_60510


namespace rainfall_ratio_l605_60559

theorem rainfall_ratio (rain_15_days : ℕ) (total_rain : ℕ) (days_in_month : ℕ) (rain_per_day_first_15 : ℕ) :
  rain_per_day_first_15 * 15 = rain_15_days →
  rain_15_days + (days_in_month - 15) * (rain_per_day_first_15 * 2) = total_rain →
  days_in_month = 30 →
  total_rain = 180 →
  rain_per_day_first_15 = 4 →
  2 = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end rainfall_ratio_l605_60559


namespace quadrilateral_perimeter_correct_l605_60547

noncomputable def quadrilateral_perimeter : ℝ :=
  let AB := 15
  let BC := 20
  let CD := 9
  let AC := Real.sqrt (AB^2 + BC^2)
  let AD := Real.sqrt (AC^2 + CD^2)
  AB + BC + CD + AD

theorem quadrilateral_perimeter_correct :
  quadrilateral_perimeter = 44 + Real.sqrt 706 := by
  sorry

end quadrilateral_perimeter_correct_l605_60547
