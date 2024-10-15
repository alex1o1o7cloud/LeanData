import Mathlib

namespace NUMINAMATH_GPT_smallest_sum_l84_8416

theorem smallest_sum (r s t : ℕ) (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t) 
  (h_prod : r * s * t = 1230) : r + s + t = 52 :=
sorry

end NUMINAMATH_GPT_smallest_sum_l84_8416


namespace NUMINAMATH_GPT_parabolas_intersect_at_single_point_l84_8432

theorem parabolas_intersect_at_single_point (p q : ℝ) (h : -2 * p + q = 2023) :
  ∃ (x0 y0 : ℝ), (∀ p q : ℝ, y0 = x0^2 + p * x0 + q → -2 * p + q = 2023) ∧ x0 = -2 ∧ y0 = 2027 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_parabolas_intersect_at_single_point_l84_8432


namespace NUMINAMATH_GPT_tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l84_8467

variable (α : ℝ)

theorem tan_alpha_sub_2pi_over_3 (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    Real.tan (α - 2 * π / 3) = 2 * Real.sqrt 3 :=
sorry

theorem two_sin_sq_alpha_sub_cos_sq_alpha (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    2 * (Real.sin α) ^ 2 - (Real.cos α) ^ 2 = -43 / 52 :=
sorry

end NUMINAMATH_GPT_tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l84_8467


namespace NUMINAMATH_GPT_cost_price_l84_8429

theorem cost_price (MP : ℝ) (SP : ℝ) (C : ℝ) 
  (h1 : MP = 87.5) 
  (h2 : SP = 0.95 * MP) 
  (h3 : SP = 1.25 * C) : 
  C = 66.5 := 
by
  sorry

end NUMINAMATH_GPT_cost_price_l84_8429


namespace NUMINAMATH_GPT_smallest_positive_period_symmetry_axis_range_of_f_l84_8446
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem symmetry_axis (k : ℤ) :
  ∃ k : ℤ, ∃ x : ℝ, f x = f (x + k * (Real.pi / 2)) ∧ x = (Real.pi / 6) + k * (Real.pi / 2) := sorry

theorem range_of_f : 
  ∀ x, -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1/2 ≤ f x ∧ f x ≤ 1 := sorry

end NUMINAMATH_GPT_smallest_positive_period_symmetry_axis_range_of_f_l84_8446


namespace NUMINAMATH_GPT_statement_A_statement_B_statement_C_statement_D_statement_E_l84_8482

-- Define a statement for each case and prove each one
theorem statement_A (x : ℝ) (h : x ≥ 0) : x^2 ≥ x :=
sorry

theorem statement_B (x : ℝ) (h : x^2 ≥ 0) : abs x ≥ 0 :=
sorry

theorem statement_C (x : ℝ) (h : x^2 ≤ x) : ¬ (x ≤ 1) :=
sorry

theorem statement_D (x : ℝ) (h : x^2 ≥ x) : ¬ (x ≤ 0) :=
sorry

theorem statement_E (x : ℝ) (h : x ≤ -1) : x^2 ≥ abs x :=
sorry

end NUMINAMATH_GPT_statement_A_statement_B_statement_C_statement_D_statement_E_l84_8482


namespace NUMINAMATH_GPT_ticket_cost_proof_l84_8409

def adult_ticket_price : ℕ := 55
def child_ticket_price : ℕ := 28
def senior_ticket_price : ℕ := 42

def num_adult_tickets : ℕ := 4
def num_child_tickets : ℕ := 2
def num_senior_tickets : ℕ := 1

def total_ticket_cost : ℕ :=
  (num_adult_tickets * adult_ticket_price) + (num_child_tickets * child_ticket_price) + (num_senior_tickets * senior_ticket_price)

theorem ticket_cost_proof : total_ticket_cost = 318 := by
  sorry

end NUMINAMATH_GPT_ticket_cost_proof_l84_8409


namespace NUMINAMATH_GPT_polynomial_coefficients_equivalence_l84_8486

theorem polynomial_coefficients_equivalence
    {a0 a1 a2 a3 a4 a5 : ℤ}
    (h_poly : (2*x-1)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5):
    (a0 + a1 + a2 + a3 + a4 + a5 = 1) ∧
    (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| = 243) ∧
    (a1 + a3 + a5 = 122) ∧
    ((a0 + a2 + a4)^2 - (a1 + a3 + a5)^2 = -243) :=
    sorry

end NUMINAMATH_GPT_polynomial_coefficients_equivalence_l84_8486


namespace NUMINAMATH_GPT_glass_volume_230_l84_8460

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end NUMINAMATH_GPT_glass_volume_230_l84_8460


namespace NUMINAMATH_GPT_computation_problem_points_l84_8437

/-- A teacher gives out a test of 30 problems. Each computation problem is worth some points, and
each word problem is worth 5 points. The total points you can receive on the test is 110 points,
and there are 20 computation problems. How many points is each computation problem worth? -/

theorem computation_problem_points (x : ℕ) (total_problems : ℕ := 30) (word_problem_points : ℕ := 5)
    (total_points : ℕ := 110) (computation_problems : ℕ := 20) :
    20 * x + (total_problems - computation_problems) * word_problem_points = total_points → x = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_computation_problem_points_l84_8437


namespace NUMINAMATH_GPT_xy_difference_l84_8442

theorem xy_difference (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 10) (h3 : x = 15) : x - y = 10 :=
by
  sorry

end NUMINAMATH_GPT_xy_difference_l84_8442


namespace NUMINAMATH_GPT_employee_gross_pay_l84_8413

theorem employee_gross_pay
  (pay_rate_regular : ℝ) (pay_rate_overtime : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ)
  (h1 : pay_rate_regular = 11.25)
  (h2 : pay_rate_overtime = 16)
  (h3 : regular_hours = 40)
  (h4 : overtime_hours = 10.75) :
  (pay_rate_regular * regular_hours + pay_rate_overtime * overtime_hours = 622) :=
by
  sorry

end NUMINAMATH_GPT_employee_gross_pay_l84_8413


namespace NUMINAMATH_GPT_tan_angle_add_l84_8418

theorem tan_angle_add (x : ℝ) (h : Real.tan x = -3) : Real.tan (x + Real.pi / 6) = 2 * Real.sqrt 3 + 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_angle_add_l84_8418


namespace NUMINAMATH_GPT_evaluate_product_l84_8483

-- Define the given numerical values
def a : ℝ := 2.5
def b : ℝ := 50.5
def c : ℝ := 0.15

-- State the theorem we want to prove
theorem evaluate_product : a * (b + c) = 126.625 := by
  sorry

end NUMINAMATH_GPT_evaluate_product_l84_8483


namespace NUMINAMATH_GPT_trajectory_eq_l84_8459

-- Define the points O, A, and B
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-1, -2)

-- Define the vector equation for point C given the parameters s and t
def C (s t : ℝ) : ℝ × ℝ := (s * 2 + t * -1, s * 1 + t * -2)

-- Prove the equation of the trajectory of C given s + t = 1
theorem trajectory_eq (s t : ℝ) (h : s + t = 1) : ∃ x y : ℝ, C s t = (x, y) ∧ x - y - 1 = 0 := by
  -- The proof will be added here
  sorry

end NUMINAMATH_GPT_trajectory_eq_l84_8459


namespace NUMINAMATH_GPT_edward_spent_on_books_l84_8470

def money_spent_on_books (initial_amount spent_on_pens amount_left : ℕ) : ℕ :=
  initial_amount - amount_left - spent_on_pens

theorem edward_spent_on_books :
  ∃ (x : ℕ), x = 6 → 
  ∀ {initial_amount spent_on_pens amount_left : ℕ},
    initial_amount = 41 →
    spent_on_pens = 16 →
    amount_left = 19 →
    x = money_spent_on_books initial_amount spent_on_pens amount_left :=
by
  sorry

end NUMINAMATH_GPT_edward_spent_on_books_l84_8470


namespace NUMINAMATH_GPT_sufficient_condition_implies_range_of_p_l84_8411

open Set Real

theorem sufficient_condition_implies_range_of_p (p : ℝ) :
  (∀ x : ℝ, 4 * x + p < 0 → x^2 - x - 2 > 0) →
  (∃ x : ℝ, x^2 - x - 2 > 0 ∧ ¬ (4 * x + p < 0)) →
  p ∈ Set.Ici 4 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_implies_range_of_p_l84_8411


namespace NUMINAMATH_GPT_proposition_D_l84_8495

theorem proposition_D (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by {
    sorry
}

end NUMINAMATH_GPT_proposition_D_l84_8495


namespace NUMINAMATH_GPT_freshmen_more_than_sophomores_l84_8458

theorem freshmen_more_than_sophomores :
  ∀ (total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores : ℕ),
    total_students = 1200 →
    juniors = 264 →
    not_sophomores = 660 →
    not_freshmen = 300 →
    seniors = 240 →
    adv_grade = 20 →
    freshmen = total_students - not_freshmen - seniors - adv_grade →
    sophomores = total_students - not_sophomores - seniors - adv_grade →
    freshmen - sophomores = 360 :=
by
  intros total_students juniors not_sophomores not_freshmen seniors adv_grade freshmen sophomores
  intros h_total h_juniors h_not_sophomores h_not_freshmen h_seniors h_adv_grade h_freshmen h_sophomores
  sorry

end NUMINAMATH_GPT_freshmen_more_than_sophomores_l84_8458


namespace NUMINAMATH_GPT_sum_of_powers_mod_7_l84_8422

theorem sum_of_powers_mod_7 :
  ((1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7) % 7 = 1) := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_mod_7_l84_8422


namespace NUMINAMATH_GPT_ratio_of_rats_l84_8448

theorem ratio_of_rats (x y : ℝ) (h : (0.56 * x) / (0.84 * y) = 1 / 2) : x / y = 3 / 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_rats_l84_8448


namespace NUMINAMATH_GPT_sum_of_roots_l84_8496

theorem sum_of_roots (x : ℝ) :
  (x^2 = 10 * x - 13) → ∃ s, s = 10 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l84_8496


namespace NUMINAMATH_GPT_decrease_percent_revenue_l84_8485

theorem decrease_percent_revenue 
  (T C : ℝ) 
  (hT : T > 0) 
  (hC : C > 0) 
  (new_tax : ℝ := 0.65 * T) 
  (new_consumption : ℝ := 1.15 * C) 
  (original_revenue : ℝ := T * C) 
  (new_revenue : ℝ := new_tax * new_consumption) :
  100 * (original_revenue - new_revenue) / original_revenue = 25.25 :=
sorry

end NUMINAMATH_GPT_decrease_percent_revenue_l84_8485


namespace NUMINAMATH_GPT_phoenix_equal_roots_implies_a_eq_c_l84_8464

-- Define the "phoenix" equation property
def is_phoenix (a b c : ℝ) : Prop := a + b + c = 0

-- Define the property that a quadratic equation has equal real roots
def has_equal_real_roots (a b c : ℝ) : Prop := b^2 - 4 * a * c = 0

theorem phoenix_equal_roots_implies_a_eq_c (a b c : ℝ) (h₀ : a ≠ 0) 
  (h₁ : is_phoenix a b c) (h₂ : has_equal_real_roots a b c) : a = c :=
sorry

end NUMINAMATH_GPT_phoenix_equal_roots_implies_a_eq_c_l84_8464


namespace NUMINAMATH_GPT_min_voters_for_Tall_victory_l84_8445

def total_voters := 105
def districts := 5
def sections_per_district := 7
def voters_per_section := 3
def sections_to_win_district := 4
def districts_to_win := 3
def sections_to_win := sections_to_win_district * districts_to_win
def min_voters_to_win_section := 2

theorem min_voters_for_Tall_victory : 
  (total_voters = 105) ∧ 
  (districts = 5) ∧ 
  (sections_per_district = 7) ∧ 
  (voters_per_section = 3) ∧ 
  (sections_to_win_district = 4) ∧ 
  (districts_to_win = 3) 
  → 
  min_voters_to_win_section * sections_to_win = 24 :=
by
  sorry
  
end NUMINAMATH_GPT_min_voters_for_Tall_victory_l84_8445


namespace NUMINAMATH_GPT_single_elimination_games_needed_l84_8402

theorem single_elimination_games_needed (teams : ℕ) (h : teams = 19) : 
∃ games, games = 18 ∧ (∀ (teams_left : ℕ), teams_left = teams - 1 → games = teams - 1) :=
by
  -- define the necessary parameters and properties here 
  sorry

end NUMINAMATH_GPT_single_elimination_games_needed_l84_8402


namespace NUMINAMATH_GPT_find_solutions_l84_8461

def is_solution (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 1 ∧
  a ∣ (b + c) ∧
  b ∣ (c + d) ∧
  c ∣ (d + a) ∧
  d ∣ (a + b)

theorem find_solutions : ∀ (a b c d : ℕ),
  is_solution a b c d →
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1) ∨
  (a = 5 ∧ b = 3 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 4 ∧ c = 1 ∧ d = 3) ∨
  (a = 7 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 3 ∧ b = 1 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 4 ∧ d = 3) ∨
  (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 1) ∨
  (a = 7 ∧ b = 2 ∧ c = 5 ∧ d = 3) ∨
  (a = 7 ∧ b = 3 ∧ c = 4 ∧ d = 5) :=
by
  intros a b c d h
  sorry

end NUMINAMATH_GPT_find_solutions_l84_8461


namespace NUMINAMATH_GPT_bus_ride_duration_l84_8493

theorem bus_ride_duration (total_hours : ℕ) (train_hours : ℕ) (walk_minutes : ℕ) (wait_factor : ℕ) 
    (h_total : total_hours = 8)
    (h_train : train_hours = 6)
    (h_walk : walk_minutes = 15)
    (h_wait : wait_factor = 2) : 
    let total_minutes := total_hours * 60
    let train_minutes := train_hours * 60
    let wait_minutes := wait_factor * walk_minutes
    let travel_minutes := total_minutes - train_minutes
    let bus_ride_minutes := travel_minutes - walk_minutes - wait_minutes
    bus_ride_minutes = 75 :=
by
  sorry

end NUMINAMATH_GPT_bus_ride_duration_l84_8493


namespace NUMINAMATH_GPT_value_of_expression_at_three_l84_8404

theorem value_of_expression_at_three (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_at_three_l84_8404


namespace NUMINAMATH_GPT_product_of_two_special_numbers_is_perfect_square_l84_8451

-- Define the structure of the required natural numbers
structure SpecialNumber where
  m : ℕ
  n : ℕ
  value : ℕ := 2^m * 3^n

-- The main theorem to be proved
theorem product_of_two_special_numbers_is_perfect_square :
  ∀ (a b c d e : SpecialNumber),
  ∃ x y : SpecialNumber, ∃ k : ℕ, (x.value * y.value) = k * k :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_special_numbers_is_perfect_square_l84_8451


namespace NUMINAMATH_GPT_inverse_proposition_l84_8433

theorem inverse_proposition (x : ℝ) : 
  (¬ (x > 2) → ¬ (x > 1)) ↔ ((x > 1) → (x > 2)) := 
by 
  sorry

end NUMINAMATH_GPT_inverse_proposition_l84_8433


namespace NUMINAMATH_GPT_max_lamps_on_road_l84_8414

theorem max_lamps_on_road (k: ℕ) (lk: ℕ): 
  lk = 1000 → (∀ n: ℕ, n < k → n≥ 1 ∧ ∀ m: ℕ, if m > n then m > 1 else true) → (lk ≤ k) ∧ 
  (∀ i:ℕ,∃ j, (i ≠ j) → (lk < 1000)) → k = 1998 :=
by sorry

end NUMINAMATH_GPT_max_lamps_on_road_l84_8414


namespace NUMINAMATH_GPT_rotated_parabola_equation_l84_8457

def parabola_equation (x y : ℝ) : Prop := y = x^2 - 4 * x + 3

def standard_form (x y : ℝ) : Prop := y = (x - 2)^2 - 1

def after_rotation (x y : ℝ) : Prop := (y + 1)^2 = x - 2

theorem rotated_parabola_equation (x y : ℝ) (h : standard_form x y) : after_rotation x y :=
sorry

end NUMINAMATH_GPT_rotated_parabola_equation_l84_8457


namespace NUMINAMATH_GPT_concentration_after_removal_l84_8497

/-- 
Given:
1. A container has 27 liters of 40% acidic liquid.
2. 9 liters of water is removed from this container.

Prove that the concentration of the acidic liquid in the container after removal is 60%.
-/
theorem concentration_after_removal :
  let initial_volume := 27
  let initial_concentration := 0.4
  let water_removed := 9
  let pure_acid := initial_concentration * initial_volume
  let new_volume := initial_volume - water_removed
  let final_concentration := (pure_acid / new_volume) * 100
  final_concentration = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_concentration_after_removal_l84_8497


namespace NUMINAMATH_GPT_maximum_k_for_ray_below_f_l84_8489

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 3 * x - 2

theorem maximum_k_for_ray_below_f :
  let g (x : ℝ) : ℝ := (x * Real.log x + 3 * x - 2) / (x - 1)
  ∃ k : ℤ, ∀ x > 1, g x > k ∧ k = 5 :=
by sorry

end NUMINAMATH_GPT_maximum_k_for_ray_below_f_l84_8489


namespace NUMINAMATH_GPT_cost_of_500_pencils_is_15_dollars_l84_8410

-- Defining the given conditions
def cost_per_pencil_cents : ℕ := 3
def pencils_count : ℕ := 500
def cents_to_dollars : ℕ := 100

-- The proof problem: statement only, no proof provided
theorem cost_of_500_pencils_is_15_dollars :
  (cost_per_pencil_cents * pencils_count) / cents_to_dollars = 15 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_500_pencils_is_15_dollars_l84_8410


namespace NUMINAMATH_GPT_ratio_of_7th_terms_l84_8443

theorem ratio_of_7th_terms (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : ∀ n, T n = n * (b 1 + b n) / 2)
  (h3 : ∀ n, S n / T n = (5 * n + 10) / (2 * n - 1)) :
  a 7 / b 7 = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_7th_terms_l84_8443


namespace NUMINAMATH_GPT_mod_arith_proof_l84_8440

theorem mod_arith_proof (m : ℕ) (hm1 : 0 ≤ m) (hm2 : m < 50) : 198 * 935 % 50 = 30 := 
by
  sorry

end NUMINAMATH_GPT_mod_arith_proof_l84_8440


namespace NUMINAMATH_GPT_incorrect_observation_value_l84_8453

theorem incorrect_observation_value
  (mean : ℕ → ℝ)
  (n : ℕ)
  (observed_mean : ℝ)
  (incorrect_value : ℝ)
  (correct_value : ℝ)
  (corrected_mean : ℝ)
  (H1 : n = 50)
  (H2 : observed_mean = 36)
  (H3 : correct_value = 43)
  (H4 : corrected_mean = 36.5)
  (H5 : mean n = observed_mean)
  (H6 : mean (n - 1 + 1) = corrected_mean - correct_value + incorrect_value) :
  incorrect_value = 18 := sorry

end NUMINAMATH_GPT_incorrect_observation_value_l84_8453


namespace NUMINAMATH_GPT_circles_are_externally_tangent_l84_8415

noncomputable def circleA : Prop := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 1 = 0
noncomputable def circleB : Prop := ∀ x y : ℝ, x^2 + y^2 - 2 * x - 6 * y + 1 = 0

theorem circles_are_externally_tangent (hA : circleA) (hB : circleB) : 
  ∃ P Q : ℝ, (P = 5) ∧ (Q = 5) := 
by 
  -- start proving with given conditions
  sorry

end NUMINAMATH_GPT_circles_are_externally_tangent_l84_8415


namespace NUMINAMATH_GPT_problem_B_false_l84_8434

def diamondsuit (x y : ℝ) : ℝ := abs (x + y - 1)

theorem problem_B_false : ∀ x y : ℝ, 2 * (diamondsuit x y) ≠ diamondsuit (2 * x) (2 * y) :=
by
  intro x y
  dsimp [diamondsuit]
  sorry

end NUMINAMATH_GPT_problem_B_false_l84_8434


namespace NUMINAMATH_GPT_find_value_of_a_l84_8401

theorem find_value_of_a (b : ℤ) (q : ℚ) (a : ℤ) (h₁ : b = 2120) (h₂ : q = 0.5) (h₃ : (a : ℚ) / b = q) : a = 1060 :=
sorry

end NUMINAMATH_GPT_find_value_of_a_l84_8401


namespace NUMINAMATH_GPT_three_Y_five_l84_8472

-- Define the operation Y
def Y (a b : ℕ) : ℕ := 3 * b + 8 * a - a^2

-- State the theorem to prove the value of 3 Y 5
theorem three_Y_five : Y 3 5 = 30 :=
by
  sorry

end NUMINAMATH_GPT_three_Y_five_l84_8472


namespace NUMINAMATH_GPT_multiples_of_7_units_digit_7_l84_8438

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end NUMINAMATH_GPT_multiples_of_7_units_digit_7_l84_8438


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l84_8487

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 3 → x^2 - 2 * x > 0) ∧ ¬ (x^2 - 2 * x > 0 → x > 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l84_8487


namespace NUMINAMATH_GPT_large_rect_area_is_294_l84_8419

-- Define the dimensions of the smaller rectangles
def shorter_side : ℕ := 7
def longer_side : ℕ := 2 * shorter_side

-- Condition 1: Each smaller rectangle has a shorter side measuring 7 feet
axiom smaller_rect_shorter_side : ∀ (r : ℕ), r = shorter_side → r = 7

-- Condition 4: The longer side of each smaller rectangle is twice the shorter side
axiom smaller_rect_longer_side : ∀ (r : ℕ), r = longer_side → r = 2 * shorter_side

-- Condition 2: Three rectangles are aligned vertically
def vertical_height : ℕ := 3 * shorter_side

-- Condition 3: One rectangle is aligned horizontally adjoining them
def horizontal_length : ℕ := longer_side

-- The dimensions of the larger rectangle EFGH
def large_rect_width : ℕ := vertical_height
def large_rect_length : ℕ := horizontal_length

-- Calculate the area of the larger rectangle EFGH
def large_rect_area : ℕ := large_rect_width * large_rect_length

-- Prove that the area of the large rectangle is 294 square feet
theorem large_rect_area_is_294 : large_rect_area = 294 := by
  sorry

end NUMINAMATH_GPT_large_rect_area_is_294_l84_8419


namespace NUMINAMATH_GPT_mr_li_age_l84_8481

theorem mr_li_age (xiaofang_age : ℕ) (h1 : xiaofang_age = 5)
  (h2 : ∀ t : ℕ, (t = 3) → ∀ mr_li_age_in_3_years : ℕ, (mr_li_age_in_3_years = xiaofang_age + t + 20)) :
  ∃ mr_li_age : ℕ, mr_li_age = 25 :=
by
  sorry

end NUMINAMATH_GPT_mr_li_age_l84_8481


namespace NUMINAMATH_GPT_compare_abc_l84_8454

noncomputable def a : ℝ := ∫ x in (0:ℝ)..1, x ^ (-1/3 : ℝ)
noncomputable def b : ℝ := 1 - ∫ x in (0:ℝ)..1, x ^ (1/2 : ℝ)
noncomputable def c : ℝ := ∫ x in (0:ℝ)..1, x ^ (3 : ℝ)

theorem compare_abc : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_compare_abc_l84_8454


namespace NUMINAMATH_GPT_pet_store_cages_l84_8471

theorem pet_store_cages (total_puppies sold_puppies puppies_per_cage : ℕ) (h1 : total_puppies = 45) (h2 : sold_puppies = 39) (h3 : puppies_per_cage = 2) :
  (total_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_cages_l84_8471


namespace NUMINAMATH_GPT_solve_equation_l84_8444

theorem solve_equation : ∃! x : ℕ, 3^x = x + 2 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l84_8444


namespace NUMINAMATH_GPT_spider_legs_is_multiple_of_human_legs_l84_8499

def human_legs : ℕ := 2
def spider_legs : ℕ := 8

theorem spider_legs_is_multiple_of_human_legs :
  spider_legs = 4 * human_legs :=
by 
  sorry

end NUMINAMATH_GPT_spider_legs_is_multiple_of_human_legs_l84_8499


namespace NUMINAMATH_GPT_probability_of_defective_product_l84_8463

theorem probability_of_defective_product :
  let total_products := 10
  let defective_products := 2
  (defective_products: ℚ) / total_products = 1 / 5 :=
by
  let total_products := 10
  let defective_products := 2
  have h : (defective_products: ℚ) / total_products = 1 / 5
  {
    exact sorry
  }
  exact h

end NUMINAMATH_GPT_probability_of_defective_product_l84_8463


namespace NUMINAMATH_GPT_work_completion_days_l84_8403

theorem work_completion_days (D_a : ℝ) (R_a R_b : ℝ)
  (h1 : R_a = 1 / D_a)
  (h2 : R_b = 1 / (1.5 * D_a))
  (h3 : R_a = 1.5 * R_b)
  (h4 : 1 / 18 = R_a + R_b) : D_a = 30 := 
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l84_8403


namespace NUMINAMATH_GPT_maximum_value_of_f_l84_8435

noncomputable def f (t : ℝ) : ℝ := ((3^t - 4 * t) * t) / (9^t)

theorem maximum_value_of_f : ∃ t : ℝ, f t = 1/16 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_f_l84_8435


namespace NUMINAMATH_GPT_josh_total_money_l84_8412

-- Define the initial conditions
def initial_wallet : ℝ := 300
def initial_investment : ℝ := 2000
def stock_increase_rate : ℝ := 0.30

-- The expected total amount Josh will have after selling his stocks
def expected_total_amount : ℝ := 2900

-- Define the problem: that the total money in Josh's wallet after selling all stocks equals $2900
theorem josh_total_money :
  let increased_value := initial_investment * stock_increase_rate
  let new_investment := initial_investment + increased_value
  let total_money := new_investment + initial_wallet
  total_money = expected_total_amount :=
by
  sorry

end NUMINAMATH_GPT_josh_total_money_l84_8412


namespace NUMINAMATH_GPT_no_passing_quadrant_III_l84_8473

def y (k x : ℝ) : ℝ := k * x - k

theorem no_passing_quadrant_III (k : ℝ) (h : k < 0) :
  ¬(∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x - k) :=
sorry

end NUMINAMATH_GPT_no_passing_quadrant_III_l84_8473


namespace NUMINAMATH_GPT_quadratic_factors_l84_8469

theorem quadratic_factors {a b c : ℝ} (h : a = 1) (h_roots : (1:ℝ) + 2 = b ∧ (-1:ℝ) * 2 = c) :
  (x^2 - b * x + c) = (x - 1) * (x - 2) := by
  sorry

end NUMINAMATH_GPT_quadratic_factors_l84_8469


namespace NUMINAMATH_GPT_polynomial_root_product_l84_8480

theorem polynomial_root_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 1 = 0 → r^6 - b * r - c = 0) → b * c = 40 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_root_product_l84_8480


namespace NUMINAMATH_GPT_f1_odd_f2_even_l84_8478

noncomputable def f1 (x : ℝ) : ℝ := x + x^3 + x^5
noncomputable def f2 (x : ℝ) : ℝ := x^2 + 1

theorem f1_odd : ∀ x : ℝ, f1 (-x) = - f1 x := 
by
  sorry

theorem f2_even : ∀ x : ℝ, f2 (-x) = f2 x := 
by
  sorry

end NUMINAMATH_GPT_f1_odd_f2_even_l84_8478


namespace NUMINAMATH_GPT_square_diff_theorem_l84_8427

theorem square_diff_theorem
  (a b c p x : ℝ)
  (h1 : a + b + c = 2 * p)
  (h2 : x = (b^2 + c^2 - a^2) / (2 * c))
  (h3 : c ≠ 0) :
  b^2 - x^2 = 4 / c^2 * (p * (p - a) * (p - b) * (p - c)) := by
  sorry

end NUMINAMATH_GPT_square_diff_theorem_l84_8427


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l84_8494

theorem necessary_but_not_sufficient_condition (a : ℕ → ℝ) (a1_pos : 0 < a 1) (q : ℝ) (geo_seq : ∀ n, a (n+1) = q * a n) : 
  (∀ n : ℕ, a (2*n + 1) + a (2*n + 2) < 0) → q < 0 :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l84_8494


namespace NUMINAMATH_GPT_caitlins_team_number_l84_8408

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the two-digit prime numbers
def two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- Lean statement
theorem caitlins_team_number (h_date birthday_before today birthday_after : ℕ)
  (p₁ p₂ p₃ : ℕ)
  (h1 : two_digit_prime p₁)
  (h2 : two_digit_prime p₂)
  (h3 : two_digit_prime p₃)
  (h4 : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h5 : p₁ + p₂ = today ∨ p₁ + p₃ = today ∨ p₂ + p₃ = today)
  (h6 : (p₁ + p₂ = birthday_before ∨ p₁ + p₃ = birthday_before ∨ p₂ + p₃ = birthday_before)
       ∧ birthday_before < today)
  (h7 : (p₁ + p₂ = birthday_after ∨ p₁ + p₃ = birthday_after ∨ p₂ + p₃ = birthday_after)
       ∧ birthday_after > today) :
  p₃ = 11 := by
  sorry

end NUMINAMATH_GPT_caitlins_team_number_l84_8408


namespace NUMINAMATH_GPT_inequality_one_system_of_inequalities_l84_8479

theorem inequality_one (x : ℝ) : 
  (2 * x - 2) / 3 ≤ 2 - (2 * x + 2) / 2 → x ≤ 1 :=
sorry

theorem system_of_inequalities (x : ℝ) : 
  (3 * (x - 2) - 1 ≥ -4 - 2 * (x - 2) → x ≥ 7 / 5) ∧
  ((1 - 2 * x) / 3 > (3 * (2 * x - 1)) / 2 → x < 1 / 2) → false :=
sorry

end NUMINAMATH_GPT_inequality_one_system_of_inequalities_l84_8479


namespace NUMINAMATH_GPT_students_decrement_l84_8420

theorem students_decrement:
  ∃ d : ℕ, ∃ A : ℕ, 
  (∃ n1 n2 n3 n4 n5 : ℕ, n1 = A ∧ n2 = A - d ∧ n3 = A - 2 * d ∧ n4 = A - 3 * d ∧ n5 = A - 4 * d) ∧
  (5 = 5) ∧
  (n1 + n2 + n3 + n4 + n5 = 115) ∧
  (A = 27) → d = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_decrement_l84_8420


namespace NUMINAMATH_GPT_seonmi_initial_money_l84_8450

theorem seonmi_initial_money (M : ℝ) (h1 : M/6 = 250) : M = 1500 :=
by
  sorry

end NUMINAMATH_GPT_seonmi_initial_money_l84_8450


namespace NUMINAMATH_GPT_geo_seq_property_l84_8468

theorem geo_seq_property (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n+1) = r * a n)
  (h4_8 : a 4 + a 8 = -3) : a 6 * (a 2 + 2 * a 6 + a 10) = 9 := 
sorry

end NUMINAMATH_GPT_geo_seq_property_l84_8468


namespace NUMINAMATH_GPT_two_d_minus_c_zero_l84_8455

theorem two_d_minus_c_zero :
  ∃ (c d : ℕ), (∀ x : ℕ, x^2 - 18 * x + 72 = (x - c) * (x - d)) ∧ c > d ∧ (2 * d - c = 0) := 
sorry

end NUMINAMATH_GPT_two_d_minus_c_zero_l84_8455


namespace NUMINAMATH_GPT_inequality_of_thirds_of_ordered_triples_l84_8436

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

end NUMINAMATH_GPT_inequality_of_thirds_of_ordered_triples_l84_8436


namespace NUMINAMATH_GPT_subscriptions_to_grandfather_l84_8431

/-- 
Maggie earns $5.00 for every magazine subscription sold. 
She sold 4 subscriptions to her parents, 2 to the next-door neighbor, 
and twice that amount to another neighbor. Maggie earned $55 in total. 
Prove that the number of subscriptions Maggie sold to her grandfather is 1.
-/
theorem subscriptions_to_grandfather (G : ℕ) 
  (h1 : 5 * (4 + G + 2 + 4) = 55) : 
  G = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_subscriptions_to_grandfather_l84_8431


namespace NUMINAMATH_GPT_value_x_plus_2y_plus_3z_l84_8452

variable (x y z : ℝ)

theorem value_x_plus_2y_plus_3z :
  x + y = 5 →
  z^2 = x * y + y - 9 →
  x + 2 * y + 3 * z = 8 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_value_x_plus_2y_plus_3z_l84_8452


namespace NUMINAMATH_GPT_equation_represents_point_l84_8417

theorem equation_represents_point 
  (a b x y : ℝ) 
  (h : (x - a) ^ 2 + (y + b) ^ 2 = 0) : 
  x = a ∧ y = -b := 
by
  sorry

end NUMINAMATH_GPT_equation_represents_point_l84_8417


namespace NUMINAMATH_GPT_diver_descend_rate_l84_8400

theorem diver_descend_rate (depth : ℕ) (time : ℕ) (rate : ℕ) 
  (h1 : depth = 6400) (h2 : time = 200) : rate = 32 :=
by
  sorry

end NUMINAMATH_GPT_diver_descend_rate_l84_8400


namespace NUMINAMATH_GPT_smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l84_8405

noncomputable def f (x m : ℝ) := 2 * (Real.cos x) ^ 2 + Real.sqrt 3 * Real.sin (2 * x) + m

theorem smallest_positive_period_pi (m : ℝ) :
  ∀ x : ℝ, f (x + π) m = f x m := sorry

theorem increasing_intervals_in_0_to_pi (m : ℝ) :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) ∨ (2 * π / 3 ≤ x ∧ x ≤ π) →
  ∀ y : ℝ, ((0 ≤ y ∧ y ≤ π / 6 ∨ (2 * π / 3 ≤ y ∧ y ≤ π)) ∧ x < y) → f x m < f y m := sorry

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 6) → -4 < f x m ∧ f x m < 4) ↔ (-6 < m ∧ m < 1) := sorry

end NUMINAMATH_GPT_smallest_positive_period_pi_increasing_intervals_in_0_to_pi_range_of_m_l84_8405


namespace NUMINAMATH_GPT_g_at_10_l84_8498

noncomputable def g : ℕ → ℝ :=
sorry

axiom g_1 : g 1 = 2
axiom g_prop (m n : ℕ) (hmn : m ≥ n) : g (m + n) + g (m - n) = 2 * (g m + g n)

theorem g_at_10 : g 10 = 200 := 
sorry

end NUMINAMATH_GPT_g_at_10_l84_8498


namespace NUMINAMATH_GPT_fraction_dad_roasted_l84_8421

theorem fraction_dad_roasted :
  ∀ (dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast : ℕ),
    dad_marshmallows = 21 →
    joe_marshmallows = 4 * dad_marshmallows →
    joe_roast = joe_marshmallows / 2 →
    total_roast = 49 →
    dad_roast = total_roast - joe_roast →
    (dad_roast : ℚ) / (dad_marshmallows : ℚ) = 1 / 3 :=
by
  intros dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast
  intro h_dad_marshmallows
  intro h_joe_marshmallows
  intro h_joe_roast
  intro h_total_roast
  intro h_dad_roast
  sorry

end NUMINAMATH_GPT_fraction_dad_roasted_l84_8421


namespace NUMINAMATH_GPT_sixth_element_row_20_l84_8447

theorem sixth_element_row_20 : (Nat.choose 20 5) = 15504 := by
  sorry

end NUMINAMATH_GPT_sixth_element_row_20_l84_8447


namespace NUMINAMATH_GPT_determine_swimming_day_l84_8449

def practices_sport_each_day (sports : ℕ → ℕ → Prop) : Prop :=
  ∀ (d : ℕ), ∃ s, sports d s

def runs_four_days_no_consecutive (sports : ℕ → ℕ → Prop) : Prop :=
  ∃ (days : ℕ → ℕ), (∀ i, sports (days i) 0) ∧ 
    (∀ i j, i ≠ j → days i ≠ days j) ∧ 
    (∀ i j, (days i + 1 = days j) → false)

def plays_basketball_tuesday (sports : ℕ → ℕ → Prop) : Prop :=
  sports 2 1

def plays_golf_friday_after_tuesday (sports : ℕ → ℕ → Prop) : Prop :=
  sports 5 2

def swims_and_plays_tennis_condition (sports : ℕ → ℕ → Prop) : Prop :=
  ∃ (swim_day tennis_day : ℕ), swim_day ≠ tennis_day ∧ 
    sports swim_day 3 ∧ 
    sports tennis_day 4 ∧ 
    ∀ (d : ℕ), (sports d 3 → sports (d + 1) 4 → false) ∧ 
    (∀ (d : ℕ), sports d 3 → ∀ (r : ℕ), sports (d + 2) 0 → false)

theorem determine_swimming_day (sports : ℕ → ℕ → Prop) : 
  practices_sport_each_day sports → 
  runs_four_days_no_consecutive sports → 
  plays_basketball_tuesday sports → 
  plays_golf_friday_after_tuesday sports → 
  swims_and_plays_tennis_condition sports → 
  ∃ (d : ℕ), d = 7 := 
sorry

end NUMINAMATH_GPT_determine_swimming_day_l84_8449


namespace NUMINAMATH_GPT_difference_in_peaches_l84_8491

-- Define the number of peaches Audrey has
def audrey_peaches : ℕ := 26

-- Define the number of peaches Paul has
def paul_peaches : ℕ := 48

-- Define the expected difference
def expected_difference : ℕ := 22

-- The theorem stating the problem
theorem difference_in_peaches : (paul_peaches - audrey_peaches = expected_difference) :=
by
  sorry

end NUMINAMATH_GPT_difference_in_peaches_l84_8491


namespace NUMINAMATH_GPT_team_total_points_l84_8475

theorem team_total_points :
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let emily_initial := 3 * (connor_initial + amy_initial + jason_initial)
  let team_before_bonus := connor_initial + amy_initial + jason_initial
  let connor_total := connor_initial + 3
  let amy_total := amy_initial + 5
  let jason_total := jason_initial + 1
  let emily_total := emily_initial
  let team_total := connor_total + amy_total + jason_total + emily_total
  team_total = 89 :=
by
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let emily_initial := 3 * (connor_initial + amy_initial + jason_initial)
  let team_before_bonus := connor_initial + amy_initial + jason_initial
  let connor_total := connor_initial + 3
  let amy_total := amy_initial + 5
  let jason_total := jason_initial + 1
  let emily_total := emily_initial
  let team_total := connor_total + amy_total + jason_total + emily_total
  sorry

end NUMINAMATH_GPT_team_total_points_l84_8475


namespace NUMINAMATH_GPT_solve_for_a_minus_c_l84_8484

theorem solve_for_a_minus_c 
  (a b c d : ℝ) 
  (h1 : a - b = c + d + 9) 
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_a_minus_c_l84_8484


namespace NUMINAMATH_GPT_triangle_is_isosceles_or_right_l84_8466

theorem triangle_is_isosceles_or_right (A B C a b : ℝ) (h : a * Real.cos (π - A) + b * Real.sin (π / 2 + B) = 0)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) : 
  (A = B ∨ A + B = π / 2) := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_or_right_l84_8466


namespace NUMINAMATH_GPT_calc_value_exponents_l84_8425

theorem calc_value_exponents :
  (3^3) * (5^3) * (3^5) * (5^5) = 15^8 :=
by sorry

end NUMINAMATH_GPT_calc_value_exponents_l84_8425


namespace NUMINAMATH_GPT_lcm_inequality_l84_8490

open Nat

theorem lcm_inequality (n : ℕ) (h : n ≥ 5 ∧ Odd n) :
  Nat.lcm n (n + 1) * (n + 2) > Nat.lcm (n + 1) (n + 2) * (n + 3) := by
  sorry

end NUMINAMATH_GPT_lcm_inequality_l84_8490


namespace NUMINAMATH_GPT_binom_identity1_binom_identity2_l84_8441

variable (n k : ℕ)

theorem binom_identity1 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) + (Nat.choose n (k + 1)) = (Nat.choose (n + 1) (k + 1)) :=
sorry

theorem binom_identity2 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) = (n * Nat.choose (n - 1) (k - 1)) / k :=
sorry

end NUMINAMATH_GPT_binom_identity1_binom_identity2_l84_8441


namespace NUMINAMATH_GPT_add_gold_coins_l84_8476

open Nat

theorem add_gold_coins (G S X : ℕ) 
  (h₁ : G = S / 3) 
  (h₂ : (G + X) / S = 1 / 2) 
  (h₃ : G + X + S = 135) : 
  X = 15 := 
sorry

end NUMINAMATH_GPT_add_gold_coins_l84_8476


namespace NUMINAMATH_GPT_division_remainder_l84_8407

theorem division_remainder :
  (1225 * 1227 * 1229) % 12 = 3 :=
by sorry

end NUMINAMATH_GPT_division_remainder_l84_8407


namespace NUMINAMATH_GPT_arithmetic_sequence_a9_l84_8439

theorem arithmetic_sequence_a9 (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * a 0 + (n - 1))) →
  S 6 = 3 * S 3 →
  a 9 = 10 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a9_l84_8439


namespace NUMINAMATH_GPT_part_a_l84_8474

theorem part_a (a b : ℤ) (h : a^2 - (b^2 - 4 * b + 1) * a - (b^4 - 2 * b^3) = 0) : 
  ∃ k : ℤ, b^2 + a = k^2 :=
sorry

end NUMINAMATH_GPT_part_a_l84_8474


namespace NUMINAMATH_GPT_siblings_of_John_l84_8488

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (height : String)

def John : Child := {name := "John", eyeColor := "Brown", hairColor := "Blonde", height := "Tall"}
def Emma : Child := {name := "Emma", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Oliver : Child := {name := "Oliver", eyeColor := "Brown", hairColor := "Black", height := "Short"}
def Mia : Child := {name := "Mia", eyeColor := "Blue", hairColor := "Blonde", height := "Short"}
def Lucas : Child := {name := "Lucas", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Sophia : Child := {name := "Sophia", eyeColor := "Blue", hairColor := "Blonde", height := "Tall"}

theorem siblings_of_John : 
  (John.hairColor = Mia.hairColor ∧ John.hairColor = Sophia.hairColor) ∧
  ((John.eyeColor = Mia.eyeColor ∨ John.eyeColor = Sophia.eyeColor) ∨
   (John.height = Mia.height ∨ John.height = Sophia.height)) ∧
  (Mia.eyeColor = Sophia.eyeColor ∨ Mia.hairColor = Sophia.hairColor ∨ Mia.height = Sophia.height) ∧
  (John.hairColor = "Blonde") ∧
  (John.height = "Tall") ∧
  (Mia.hairColor = "Blonde") ∧
  (Sophia.hairColor = "Blonde") ∧
  (Sophia.height = "Tall") 
  → True := sorry

end NUMINAMATH_GPT_siblings_of_John_l84_8488


namespace NUMINAMATH_GPT_highest_point_difference_l84_8423

theorem highest_point_difference :
  let A := -112
  let B := -80
  let C := -25
  max A (max B C) - min A (min B C) = 87 :=
by
  sorry

end NUMINAMATH_GPT_highest_point_difference_l84_8423


namespace NUMINAMATH_GPT_tiling_possible_with_one_type_l84_8462

theorem tiling_possible_with_one_type
  {a b m n : ℕ} (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n)
  (H : (∃ (k : ℕ), a = k * n) ∨ (∃ (l : ℕ), b = l * m)) :
  (∃ (i : ℕ), a = i * n) ∨ (∃ (j : ℕ), b = j * m) :=
  sorry

end NUMINAMATH_GPT_tiling_possible_with_one_type_l84_8462


namespace NUMINAMATH_GPT_least_value_b_l84_8428

-- Defining the conditions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

variables (a b c : ℕ)

-- Conditions
axiom angle_sum : a + b + c = 180
axiom primes : is_prime a ∧ is_prime b ∧ is_prime c
axiom order : a > b ∧ b > c

-- The statement to be proved
theorem least_value_b (h : a + b + c = 180) (hp : is_prime a ∧ is_prime b ∧ is_prime c) (ho : a > b ∧ b > c) : b = 5 :=
sorry

end NUMINAMATH_GPT_least_value_b_l84_8428


namespace NUMINAMATH_GPT_solve_x_given_y_l84_8456

theorem solve_x_given_y (x : ℝ) (h : 2 = 2 / (5 * x + 3)) : x = -2 / 5 :=
sorry

end NUMINAMATH_GPT_solve_x_given_y_l84_8456


namespace NUMINAMATH_GPT_slope_of_line_l84_8465

-- Definition of the line equation in slope-intercept form
def line_eq (x : ℝ) : ℝ := -5 * x + 9

-- Statement: The slope of the line y = -5x + 9 is -5
theorem slope_of_line : (∀ x : ℝ, ∃ m b : ℝ, line_eq x = m * x + b ∧ m = -5) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_slope_of_line_l84_8465


namespace NUMINAMATH_GPT_initial_rope_length_l84_8426

variable (R₀ R₁ R₂ R₃ : ℕ)
variable (h_cut1 : 2 * R₀ = R₁) -- Josh cuts the original rope in half
variable (h_cut2 : 2 * R₁ = R₂) -- He cuts one of the halves in half again
variable (h_cut3 : 5 * R₂ = R₃) -- He cuts one of the resulting pieces into fifths
variable (h_held_piece : R₃ = 5) -- The piece Josh is holding is 5 feet long

theorem initial_rope_length:
  R₀ = 100 :=
by
  sorry

end NUMINAMATH_GPT_initial_rope_length_l84_8426


namespace NUMINAMATH_GPT_no_solution_exists_l84_8492

open Int

theorem no_solution_exists (x y z : ℕ) (hx : x > 0) (hy : y > 0)
  (hz : z = Nat.gcd x y) : x + y^2 + z^3 ≠ x * y * z := 
sorry

end NUMINAMATH_GPT_no_solution_exists_l84_8492


namespace NUMINAMATH_GPT_solution_set_of_inequality_l84_8424

theorem solution_set_of_inequality : {x : ℝ | x^2 < 2 * x} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l84_8424


namespace NUMINAMATH_GPT_passed_boys_count_l84_8477

theorem passed_boys_count (total_boys avg_passed avg_failed overall_avg : ℕ) 
  (total_boys_eq : total_boys = 120) 
  (avg_passed_eq : avg_passed = 39) 
  (avg_failed_eq : avg_failed = 15) 
  (overall_avg_eq : overall_avg = 38) :
  let marks_by_passed := total_boys * overall_avg 
                         - (total_boys - passed) * avg_failed;
  let passed := marks_by_passed / avg_passed;
  passed = 115 := 
by
  sorry

end NUMINAMATH_GPT_passed_boys_count_l84_8477


namespace NUMINAMATH_GPT_pirate_loot_l84_8430

theorem pirate_loot (a b c d e : ℕ) (h1 : a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1 ∨ e = 1)
  (h2 : a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 ∨ e = 2)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h4 : a + b = 2 * (c + d) ∨ b + c = 2 * (a + e)) :
  (a, b, c, d, e) = (1, 1, 1, 1, 2) ∨ 
  (a, b, c, d, e) = (1, 1, 2, 2, 2) ∨
  (a, b, c, d, e) = (1, 2, 3, 3, 3) ∨
  (a, b, c, d, e) = (1, 2, 2, 2, 3) :=
sorry

end NUMINAMATH_GPT_pirate_loot_l84_8430


namespace NUMINAMATH_GPT_silver_nitrate_mass_fraction_l84_8406

variable (n : ℝ) (M : ℝ) (m_total : ℝ)
variable (m_agno3 : ℝ) (omega_agno3 : ℝ)

theorem silver_nitrate_mass_fraction 
  (h1 : n = 0.12) 
  (h2 : M = 170) 
  (h3 : m_total = 255)
  (h4 : m_agno3 = n * M) 
  (h5 : omega_agno3 = (m_agno3 * 100) / m_total) : 
  m_agno3 = 20.4 ∧ omega_agno3 = 8 :=
by
  -- insert proof here eventually 
  sorry

end NUMINAMATH_GPT_silver_nitrate_mass_fraction_l84_8406
