import Mathlib

namespace NUMINAMATH_GPT_opposite_sqrt3_l1587_158729

def opposite (x : ℝ) : ℝ := -x

theorem opposite_sqrt3 :
  opposite (Real.sqrt 3) = -Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_opposite_sqrt3_l1587_158729


namespace NUMINAMATH_GPT_max_area_quad_l1587_158772

noncomputable def MaxAreaABCD : ℝ :=
  let x : ℝ := 3
  let θ : ℝ := Real.pi / 2
  let φ : ℝ := Real.pi
  let area_ABC := (1/2) * x * 3 * Real.sin θ
  let area_BCD := (1/2) * 3 * 5 * Real.sin (φ - θ)
  area_ABC + area_BCD

theorem max_area_quad (x : ℝ) (h : x > 0)
  (BC_eq_3 : True)
  (CD_eq_5 : True)
  (centroids_form_isosceles : True) :
  MaxAreaABCD = 12 := by
  sorry

end NUMINAMATH_GPT_max_area_quad_l1587_158772


namespace NUMINAMATH_GPT_additional_profit_is_80000_l1587_158777

-- Define the construction cost of a regular house
def construction_cost_regular (C : ℝ) : ℝ := C

-- Define the construction cost of the special house
def construction_cost_special (C : ℝ) : ℝ := C + 200000

-- Define the selling price of a regular house
def selling_price_regular : ℝ := 350000

-- Define the selling price of the special house
def selling_price_special : ℝ := 1.8 * 350000

-- Define the profit from selling a regular house
def profit_regular (C : ℝ) : ℝ := selling_price_regular - (construction_cost_regular C)

-- Define the profit from selling the special house
def profit_special (C : ℝ) : ℝ := selling_price_special - (construction_cost_special C)

-- Define the additional profit made by building and selling the special house compared to a regular house
def additional_profit (C : ℝ) : ℝ := (profit_special C) - (profit_regular C)

-- Theorem to prove the additional profit is $80,000
theorem additional_profit_is_80000 (C : ℝ) : additional_profit C = 80000 :=
sorry

end NUMINAMATH_GPT_additional_profit_is_80000_l1587_158777


namespace NUMINAMATH_GPT_weight_of_b_l1587_158737

/--
Given:
1. The sum of weights (a, b, c) is 129 kg.
2. The sum of weights (a, b) is 80 kg.
3. The sum of weights (b, c) is 86 kg.

Prove that the weight of b is 37 kg.
-/
theorem weight_of_b (a b c : ℝ) 
  (h1 : a + b + c = 129) 
  (h2 : a + b = 80) 
  (h3 : b + c = 86) : 
  b = 37 :=
sorry

end NUMINAMATH_GPT_weight_of_b_l1587_158737


namespace NUMINAMATH_GPT_back_seat_tickets_sold_l1587_158717

def total_tickets : ℕ := 20000
def main_seat_price : ℕ := 55
def back_seat_price : ℕ := 45
def total_revenue : ℕ := 955000

theorem back_seat_tickets_sold :
  ∃ (M B : ℕ), 
    M + B = total_tickets ∧ 
    main_seat_price * M + back_seat_price * B = total_revenue ∧ 
    B = 14500 :=
by
  sorry

end NUMINAMATH_GPT_back_seat_tickets_sold_l1587_158717


namespace NUMINAMATH_GPT_ratio_of_volumes_l1587_158793

theorem ratio_of_volumes (C D : ℚ) (h1: C = (3/4) * C) (h2: D = (5/8) * D) : C / D = 5 / 6 :=
sorry

end NUMINAMATH_GPT_ratio_of_volumes_l1587_158793


namespace NUMINAMATH_GPT_smallest_integer_k_condition_l1587_158722

theorem smallest_integer_k_condition :
  ∃ k : ℤ, k > 1 ∧ k % 12 = 1 ∧ k % 5 = 1 ∧ k % 3 = 1 ∧ k = 61 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_k_condition_l1587_158722


namespace NUMINAMATH_GPT_parabola_equation_l1587_158752

theorem parabola_equation (h1: ∃ k, ∀ x y : ℝ, (x, y) = (4, -2) → y^2 = k * x) 
                          (h2: ∃ m, ∀ x y : ℝ, (x, y) = (4, -2) → x^2 = -2 * m * y) :
                          (y : ℝ)^2 = x ∨ (x : ℝ)^2 = -8 * y :=
by 
  sorry

end NUMINAMATH_GPT_parabola_equation_l1587_158752


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1587_158789

theorem sum_of_reciprocals (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + y = 5 * x * y) : (1 / x) + (1 / y) = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1587_158789


namespace NUMINAMATH_GPT_jessica_total_monthly_payment_l1587_158788

-- Definitions for the conditions
def basicCableCost : ℕ := 15
def movieChannelsCost : ℕ := 12
def sportsChannelsCost : ℕ := movieChannelsCost - 3

-- The statement to be proven
theorem jessica_total_monthly_payment :
  basicCableCost + movieChannelsCost + sportsChannelsCost = 36 := 
by
  sorry

end NUMINAMATH_GPT_jessica_total_monthly_payment_l1587_158788


namespace NUMINAMATH_GPT_probability_within_two_units_of_origin_correct_l1587_158775

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let square_area := 36
  let circle_area := 4 * Real.pi
  circle_area / square_area

theorem probability_within_two_units_of_origin_correct :
  probability_within_two_units_of_origin = Real.pi / 9 := by
  sorry

end NUMINAMATH_GPT_probability_within_two_units_of_origin_correct_l1587_158775


namespace NUMINAMATH_GPT_calc_expression_l1587_158790

def r (θ : ℚ) : ℚ := 1 / (1 + θ)
def s (θ : ℚ) : ℚ := θ + 1

theorem calc_expression : s (r (s (r (s (r 2))))) = 24 / 17 :=
by 
  sorry

end NUMINAMATH_GPT_calc_expression_l1587_158790


namespace NUMINAMATH_GPT_bagel_spending_l1587_158718

theorem bagel_spending (B D : ℝ) (h1 : D = 0.5 * B) (h2 : B = D + 15) : B + D = 45 := by
  sorry

end NUMINAMATH_GPT_bagel_spending_l1587_158718


namespace NUMINAMATH_GPT_find_L_l1587_158797

-- Conditions definitions
def initial_marbles := 57
def marbles_won_second_game := 25
def final_marbles := 64

-- Definition of L
def L := initial_marbles - 18

theorem find_L (L : ℕ) (H1 : initial_marbles = 57) (H2 : marbles_won_second_game = 25) (H3 : final_marbles = 64) : 
(initial_marbles - L) + marbles_won_second_game = final_marbles -> 
L = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_L_l1587_158797


namespace NUMINAMATH_GPT_unique_numbers_l1587_158732

theorem unique_numbers (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (S : x + y = 17) 
  (Q : x^2 + y^2 = 145) 
  : x = 8 ∧ y = 9 ∨ x = 9 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_GPT_unique_numbers_l1587_158732


namespace NUMINAMATH_GPT_correct_negation_of_exactly_one_even_l1587_158794

-- Define a predicate to check if a natural number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define a predicate to check if a natural number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Problem statement in Lean
theorem correct_negation_of_exactly_one_even (a b c : ℕ) :
  ¬ ( (is_even a ∧ is_odd b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_even b ∧ is_odd c) ∨ 
      (is_odd a ∧ is_odd b ∧ is_even c) ) ↔ 
  ( (is_odd a ∧ is_odd b ∧ is_odd c) ∨ 
    (is_even a ∧ is_even b ∧ is_even c) ) :=
by 
  sorry

end NUMINAMATH_GPT_correct_negation_of_exactly_one_even_l1587_158794


namespace NUMINAMATH_GPT_no_positive_integer_pairs_l1587_158727

theorem no_positive_integer_pairs (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) : ¬ (x^2 + y^2 = x^3 + 2 * y) :=
by sorry

end NUMINAMATH_GPT_no_positive_integer_pairs_l1587_158727


namespace NUMINAMATH_GPT_blue_faces_cube_l1587_158751

theorem blue_faces_cube (n : ℕ) (h1 : n > 0) (h2 : (6 * n^2) = 1 / 3 * 6 * n^3) : n = 3 :=
by
  -- we only need the statement for now; the proof is omitted.
  sorry

end NUMINAMATH_GPT_blue_faces_cube_l1587_158751


namespace NUMINAMATH_GPT_exist_divisible_n_and_n1_l1587_158784

theorem exist_divisible_n_and_n1 (d : ℕ) (hd : 0 < d) :
  ∃ (n n1 : ℕ), n % d = 0 ∧ n1 % d = 0 ∧ n ≠ n1 ∧
  (∃ (k a b c : ℕ), b ≠ 0 ∧ n = 10^k * (10 * a + b) + c ∧ n1 = 10^k * a + c) :=
by
  sorry

end NUMINAMATH_GPT_exist_divisible_n_and_n1_l1587_158784


namespace NUMINAMATH_GPT_roots_expression_value_l1587_158744

theorem roots_expression_value {m n : ℝ} (h₁ : m^2 - 3 * m - 2 = 0) (h₂ : n^2 - 3 * n - 2 = 0) : 
  (7 * m^2 - 21 * m - 3) * (3 * n^2 - 9 * n + 5) = 121 := 
by 
  sorry

end NUMINAMATH_GPT_roots_expression_value_l1587_158744


namespace NUMINAMATH_GPT_provider_assignment_ways_l1587_158711

theorem provider_assignment_ways (total_providers : ℕ) (children : ℕ) (h1 : total_providers = 15) (h2 : children = 4) : 
  (Finset.range total_providers).card.factorial / (Finset.range (total_providers - children)).card.factorial = 32760 :=
by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_provider_assignment_ways_l1587_158711


namespace NUMINAMATH_GPT_man_year_of_birth_l1587_158721

theorem man_year_of_birth (x : ℕ) (hx1 : (x^2 + x >= 1850)) (hx2 : (x^2 + x < 1900)) : (1850 + (x^2 + x - x)) = 1892 :=
by {
  sorry
}

end NUMINAMATH_GPT_man_year_of_birth_l1587_158721


namespace NUMINAMATH_GPT_yoongi_correct_calculation_l1587_158706

theorem yoongi_correct_calculation (x : ℕ) (h : x + 9 = 30) : x - 7 = 14 :=
sorry

end NUMINAMATH_GPT_yoongi_correct_calculation_l1587_158706


namespace NUMINAMATH_GPT_matching_red_pair_probability_l1587_158759

def total_socks := 8
def red_socks := 4
def blue_socks := 2
def green_socks := 2

noncomputable def total_pairs := Nat.choose total_socks 2
noncomputable def red_pairs := Nat.choose red_socks 2
noncomputable def blue_pairs := Nat.choose blue_socks 2
noncomputable def green_pairs := Nat.choose green_socks 2
noncomputable def total_matching_pairs := red_pairs + blue_pairs + green_pairs
noncomputable def probability_red := (red_pairs : ℚ) / total_matching_pairs

theorem matching_red_pair_probability : probability_red = 3 / 4 :=
  by sorry

end NUMINAMATH_GPT_matching_red_pair_probability_l1587_158759


namespace NUMINAMATH_GPT_role_assignment_l1587_158757

theorem role_assignment (m w : ℕ) (m_roles w_roles e_roles : ℕ) 
  (hm : m = 5) (hw : w = 6) (hm_roles : m_roles = 2) (hw_roles : w_roles = 2) (he_roles : e_roles = 2) :
  ∃ (total_assignments : ℕ), total_assignments = 25200 :=
by
  sorry

end NUMINAMATH_GPT_role_assignment_l1587_158757


namespace NUMINAMATH_GPT_problem_solution_l1587_158783

/-- 
Assume we have points A, B, C, D, and E as defined in the problem with the following properties:
- Triangle ABC has a right angle at C
- AC = 4
- BC = 3
- Triangle ABD has a right angle at A
- AD = 15
- Points C and D are on opposite sides of line AB
- The line through D parallel to AC meets CB extended at E.

Prove that the ratio DE/DB simplifies to 57/80 where p = 57 and q = 80, making p + q = 137.
-/
theorem problem_solution :
  ∃ (p q : ℕ), gcd p q = 1 ∧ (∃ D E : ℝ, DE/DB = p/q ∧ p + q = 137) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1587_158783


namespace NUMINAMATH_GPT_train_speeds_l1587_158739

-- Definitions used in conditions
def initial_distance : ℝ := 300
def time_elapsed : ℝ := 2
def remaining_distance : ℝ := 40
def speed_difference : ℝ := 10

-- Stating the problem in Lean
theorem train_speeds :
  ∃ (v_fast v_slow : ℝ),
    v_slow + speed_difference = v_fast ∧
    (2 * (v_slow + v_fast)) = (initial_distance - remaining_distance) ∧
    v_slow = 60 ∧
    v_fast = 70 :=
by
  sorry

end NUMINAMATH_GPT_train_speeds_l1587_158739


namespace NUMINAMATH_GPT_fish_count_total_l1587_158704

def Jerk_Tuna_fish : ℕ := 144
def Tall_Tuna_fish : ℕ := 2 * Jerk_Tuna_fish
def Total_fish_together : ℕ := Jerk_Tuna_fish + Tall_Tuna_fish

theorem fish_count_total :
  Total_fish_together = 432 :=
by
  sorry

end NUMINAMATH_GPT_fish_count_total_l1587_158704


namespace NUMINAMATH_GPT_ratio_x_y_l1587_158766

theorem ratio_x_y (x y : ℝ) (h : (3 * x^2 - y) / (x + y) = 1 / 2) : 
  x / y = 3 / (6 * x - 1) := 
sorry

end NUMINAMATH_GPT_ratio_x_y_l1587_158766


namespace NUMINAMATH_GPT_part1_part2_l1587_158743

-- Definitions for problem conditions and questions

/-- 
Let p and q be two distinct prime numbers greater than 5. 
Show that if p divides 5^q - 2^q then q divides p - 1.
-/
theorem part1 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div : p ∣ 5^q - 2^q) : q ∣ p - 1 :=
by sorry

/-- 
Let p and q be two distinct prime numbers greater than 5.
Deduce that pq does not divide (5^p - 2^p)(5^q - 2^q).
-/
theorem part2 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : 5 < p) (hq_gt_5 : 5 < q) (h_distinct : p ≠ q) 
  (h_div_q_p1 : q ∣ p - 1)
  (h_div_p_q1 : p ∣ q - 1) : ¬(pq : ℕ) ∣ (5^p - 2^p) * (5^q - 2^q) :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1587_158743


namespace NUMINAMATH_GPT_symmetrical_point_with_respect_to_x_axis_l1587_158756

-- Define the point P with coordinates (-2, -1)
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the given point
def P : Point := { x := -2, y := -1 }

-- Define the symmetry with respect to the x-axis
def symmetry_x_axis (p : Point) : Point :=
{ x := p.x, y := -p.y }

-- Verify the symmetrical point
theorem symmetrical_point_with_respect_to_x_axis :
  symmetry_x_axis P = { x := -2, y := 1 } :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_symmetrical_point_with_respect_to_x_axis_l1587_158756


namespace NUMINAMATH_GPT_percent_of_x_is_z_l1587_158745

variable {x y z : ℝ}

theorem percent_of_x_is_z 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z / x = 1.2 := 
sorry

end NUMINAMATH_GPT_percent_of_x_is_z_l1587_158745


namespace NUMINAMATH_GPT_hyperbola_eccentricity_is_5_over_3_l1587_158733

noncomputable def hyperbola_asymptote_condition (a b : ℝ) : Prop :=
  a / b = 3 / 4

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_5_over_3 (a b : ℝ) (h : hyperbola_asymptote_condition a b) :
  hyperbola_eccentricity a b = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_is_5_over_3_l1587_158733


namespace NUMINAMATH_GPT_more_white_birds_than_grey_l1587_158792

def num_grey_birds_in_cage : ℕ := 40
def num_remaining_birds : ℕ := 66

def num_grey_birds_freed : ℕ := num_grey_birds_in_cage / 2
def num_grey_birds_left_in_cage : ℕ := num_grey_birds_in_cage - num_grey_birds_freed
def num_white_birds : ℕ := num_remaining_birds - num_grey_birds_left_in_cage

theorem more_white_birds_than_grey : num_white_birds - num_grey_birds_in_cage = 6 := by
  sorry

end NUMINAMATH_GPT_more_white_birds_than_grey_l1587_158792


namespace NUMINAMATH_GPT_price_of_mixture_l1587_158785

theorem price_of_mixture (P1 P2 P3 : ℝ) (h1 : P1 = 126) (h2 : P2 = 135) (h3 : P3 = 175.5) : 
  (P1 + P2 + 2 * P3) / 4 = 153 :=
by 
  -- Main goal is to show (126 + 135 + 2 * 175.5) / 4 = 153
  sorry

end NUMINAMATH_GPT_price_of_mixture_l1587_158785


namespace NUMINAMATH_GPT_dentist_cleaning_cost_l1587_158795

theorem dentist_cleaning_cost
  (F: ℕ)
  (C: ℕ)
  (B: ℕ)
  (tooth_extraction_cost: ℕ)
  (HC1: F = 120)
  (HC2: B = 5 * F)
  (HC3: tooth_extraction_cost = 290)
  (HC4: B = C + 2 * F + tooth_extraction_cost) :
  C = 70 :=
by
  sorry

end NUMINAMATH_GPT_dentist_cleaning_cost_l1587_158795


namespace NUMINAMATH_GPT_alpha_beta_sum_l1587_158771

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 102 * x + 2021) / (x^2 + 89 * x - 3960)) : α + β = 176 := by
  sorry

end NUMINAMATH_GPT_alpha_beta_sum_l1587_158771


namespace NUMINAMATH_GPT_max_discarded_grapes_l1587_158750

theorem max_discarded_grapes (n : ℕ) : ∃ r, r < 8 ∧ n % 8 = r ∧ r = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_discarded_grapes_l1587_158750


namespace NUMINAMATH_GPT_john_personal_payment_l1587_158734

-- Definitions of the conditions
def cost_of_one_hearing_aid : ℕ := 2500
def number_of_hearing_aids : ℕ := 2
def insurance_coverage_percent : ℕ := 80

-- Derived definitions based on conditions
def total_cost : ℕ := cost_of_one_hearing_aid * number_of_hearing_aids
def insurance_coverage_amount : ℕ := total_cost * insurance_coverage_percent / 100
def johns_share : ℕ := total_cost - insurance_coverage_amount

-- Theorem statement (proof not included)
theorem john_personal_payment : johns_share = 1000 :=
sorry

end NUMINAMATH_GPT_john_personal_payment_l1587_158734


namespace NUMINAMATH_GPT_price_of_coffee_table_l1587_158715

-- Define the given values
def price_sofa : ℕ := 1250
def price_armchair : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Define the target value (price of the coffee table)
def price_coffee_table : ℕ := 330

-- The theorem to prove
theorem price_of_coffee_table :
  total_invoice = price_sofa + num_armchairs * price_armchair + price_coffee_table :=
by sorry

end NUMINAMATH_GPT_price_of_coffee_table_l1587_158715


namespace NUMINAMATH_GPT_monotonic_increasing_implies_range_a_l1587_158712

-- Definition of the function f(x) = ax^3 - x^2 + x - 5
def f (a x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

-- Derivative of f(x) with respect to x
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

-- The statement that proves the monotonicity condition implies the range for a
theorem monotonic_increasing_implies_range_a (a : ℝ) : 
  ( ∀ x, f_prime a x ≥ 0 ) → a ≥ (1:ℝ) / 3 := by
  sorry

end NUMINAMATH_GPT_monotonic_increasing_implies_range_a_l1587_158712


namespace NUMINAMATH_GPT_set_complement_intersection_l1587_158735

open Set

variable (U A B : Set ℕ)

theorem set_complement_intersection :
  U = {2, 3, 5, 7, 8} →
  A = {2, 8} →
  B = {3, 5, 8} →
  (U \ A) ∩ B = {3, 5} :=
by
  intros
  sorry

end NUMINAMATH_GPT_set_complement_intersection_l1587_158735


namespace NUMINAMATH_GPT_compare_points_l1587_158702

def parabola (x : ℝ) : ℝ := -x^2 - 4 * x + 1

theorem compare_points (y₁ y₂ : ℝ) :
  parabola (-3) = y₁ →
  parabola (-2) = y₂ →
  y₁ < y₂ :=
by
  intros hy₁ hy₂
  sorry

end NUMINAMATH_GPT_compare_points_l1587_158702


namespace NUMINAMATH_GPT_prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l1587_158701

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (4 * x + a) / (x^2 + 1)

-- 1. Prove that a = 0 given that f(x) is an odd function
theorem prove_a_eq_0 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = - f x a) : a = 0 := sorry

-- 2. Prove that f(x) = 4x / (x^2 + 1) is monotonically decreasing on [1, +∞) for x > 0
theorem prove_monotonic_decreasing (x : ℝ) (hx : x > 0) :
  ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (f x1 0) > (f x2 0) := sorry

-- 3. Prove that |f(x1) - f(x2)| ≤ m for all x1, x2 ∈ R implies m ≥ 4
theorem prove_m_ge_4 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, |f x1 0 - f x2 0| ≤ m) : m ≥ 4 := sorry

end NUMINAMATH_GPT_prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l1587_158701


namespace NUMINAMATH_GPT_money_left_after_spending_l1587_158742

def initial_money : ℕ := 24
def doris_spent : ℕ := 6
def martha_spent : ℕ := doris_spent / 2
def total_spent : ℕ := doris_spent + martha_spent
def money_left := initial_money - total_spent

theorem money_left_after_spending : money_left = 15 := by
  sorry

end NUMINAMATH_GPT_money_left_after_spending_l1587_158742


namespace NUMINAMATH_GPT_polynomial_coefficient_sum_l1587_158773

theorem polynomial_coefficient_sum :
  let p := (x + 3) * (4 * x^3 - 2 * x^2 + 7 * x - 6)
  let q := 4 * x^4 + 10 * x^3 + x^2 + 15 * x - 18
  p = q →
  (4 + 10 + 1 + 15 - 18 = 12) :=
by
  intro p_eq_q
  sorry

end NUMINAMATH_GPT_polynomial_coefficient_sum_l1587_158773


namespace NUMINAMATH_GPT_chess_tournament_games_l1587_158749

theorem chess_tournament_games (n : ℕ) (h : n = 25) : 2 * n * (n - 1) = 1200 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_games_l1587_158749


namespace NUMINAMATH_GPT_percentage_shaded_is_14_29_l1587_158786

noncomputable def side_length : ℝ := 20
noncomputable def rect_length : ℝ := 35
noncomputable def rect_width : ℝ := side_length
noncomputable def rect_area : ℝ := rect_length * rect_width
noncomputable def overlap_length : ℝ := 2 * side_length - rect_length
noncomputable def overlap_area : ℝ := overlap_length * side_length
noncomputable def shaded_percentage : ℝ := (overlap_area / rect_area) * 100

theorem percentage_shaded_is_14_29 :
  shaded_percentage = 14.29 :=
sorry

end NUMINAMATH_GPT_percentage_shaded_is_14_29_l1587_158786


namespace NUMINAMATH_GPT_problem_statement_l1587_158741

theorem problem_statement
  (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2) + (y^2 / b^2) + (z^2 / c^2) = 16 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1587_158741


namespace NUMINAMATH_GPT_pupils_correct_l1587_158716

def totalPeople : ℕ := 676
def numberOfParents : ℕ := 22
def numberOfPupils : ℕ := totalPeople - numberOfParents

theorem pupils_correct :
  numberOfPupils = 654 := 
by
  sorry

end NUMINAMATH_GPT_pupils_correct_l1587_158716


namespace NUMINAMATH_GPT_polar_equation_graph_l1587_158725

theorem polar_equation_graph :
  ∀ (ρ θ : ℝ), (ρ > 0) → ((ρ - 1) * (θ - π) = 0) ↔ (ρ = 1 ∨ θ = π) :=
by
  sorry

end NUMINAMATH_GPT_polar_equation_graph_l1587_158725


namespace NUMINAMATH_GPT_smallest_integer_n_l1587_158769

theorem smallest_integer_n (n : ℤ) (h : n^2 - 9 * n + 20 > 0) : n ≥ 6 := 
sorry

end NUMINAMATH_GPT_smallest_integer_n_l1587_158769


namespace NUMINAMATH_GPT_complex_real_number_l1587_158764

-- Definition of the complex number z
def z (a : ℝ) : ℂ := (a^2 + 2011) + (a - 1) * Complex.I

-- The proof problem statement
theorem complex_real_number (a : ℝ) (h : z a = (a^2 + 2011 : ℂ)) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_real_number_l1587_158764


namespace NUMINAMATH_GPT_correct_operation_A_l1587_158748

-- Definitions for the problem
def division_rule (a : ℝ) (m n : ℕ) : Prop := a^m / a^n = a^(m - n)
def multiplication_rule (a : ℝ) (m n : ℕ) : Prop := a^m * a^n = a^(m + n)
def power_rule (a : ℝ) (m n : ℕ) : Prop := (a^m)^n = a^(m * n)
def addition_like_terms_rule (a : ℝ) (m : ℕ) : Prop := a^m + a^m = 2 * a^m

-- The theorem to prove
theorem correct_operation_A (a : ℝ) : division_rule a 4 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_operation_A_l1587_158748


namespace NUMINAMATH_GPT_inequality_proof_l1587_158723

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1587_158723


namespace NUMINAMATH_GPT_solutions_of_equation_l1587_158754

theorem solutions_of_equation :
  ∀ x : ℝ, x * (x - 3) = x - 3 ↔ x = 1 ∨ x = 3 :=
by sorry

end NUMINAMATH_GPT_solutions_of_equation_l1587_158754


namespace NUMINAMATH_GPT_intersection_A_compB_l1587_158703

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of B relative to ℝ
def comp_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- State the main theorem to prove
theorem intersection_A_compB : A ∩ comp_B = {x | -3 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_compB_l1587_158703


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1587_158731

theorem sum_of_squares_of_roots :
  (∃ r1 r2 : ℝ, (r1 + r2 = 10 ∧ r1 * r2 = 16) ∧ (r1^2 + r2^2 = 68)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1587_158731


namespace NUMINAMATH_GPT_proof_S5_l1587_158746

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q a1, ∀ n, a (n + 1) = a1 * q ^ (n + 1)

theorem proof_S5 (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ) : 
  (geometric_sequence a) → 
  (a 2 * a 5 = 2 * a 3) → 
  ((a 4 + 2 * a 7) / 2 = 5 / 4) → 
  (S 5 = a1 * (1 - (1 / 2) ^ 5) / (1 - 1 / 2)) → 
  S 5 = 31 := 
by sorry

end NUMINAMATH_GPT_proof_S5_l1587_158746


namespace NUMINAMATH_GPT_stickers_total_l1587_158765

theorem stickers_total (ryan_has : Ryan_stickers = 30)
  (steven_has : Steven_stickers = 3 * Ryan_stickers)
  (terry_has : Terry_stickers = Steven_stickers + 20) :
  Ryan_stickers + Steven_stickers + Terry_stickers = 230 :=
sorry

end NUMINAMATH_GPT_stickers_total_l1587_158765


namespace NUMINAMATH_GPT_smallest_n_l1587_158768

theorem smallest_n (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, n = 3 * k) (h3 : ∃ m : ℕ, 3 * n = 5 * m) : n = 15 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1587_158768


namespace NUMINAMATH_GPT_find_x_l1587_158736

def side_of_square_eq_twice_radius_of_larger_circle (s: ℝ) (r_l: ℝ) : Prop :=
  s = 2 * r_l

def radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle (r_l: ℝ) (x: ℝ) (r_s: ℝ) : Prop :=
  r_l = x - (1 / 3) * r_s

def circumference_of_smaller_circle_eq (r_s: ℝ) (circumference: ℝ) : Prop :=
  2 * Real.pi * r_s = circumference

def side_squared_eq_area (s: ℝ) (area: ℝ) : Prop :=
  s^2 = area

noncomputable def value_of_x (r_s r_l: ℝ) : ℝ :=
  14 + 4 / (3 * Real.pi)

theorem find_x 
  (s r_l r_s x: ℝ)
  (h1: side_squared_eq_area s 784)
  (h2: side_of_square_eq_twice_radius_of_larger_circle s r_l)
  (h3: radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle r_l x r_s)
  (h4: circumference_of_smaller_circle_eq r_s 8) :
  x = value_of_x r_s r_l :=
sorry

end NUMINAMATH_GPT_find_x_l1587_158736


namespace NUMINAMATH_GPT_second_batch_students_l1587_158774

theorem second_batch_students :
  ∃ x : ℕ,
    (40 * 45 + x * 55 + 60 * 65 : ℝ) / (40 + x + 60) = 56.333333333333336 ∧
    x = 50 :=
by
  use 50
  sorry

end NUMINAMATH_GPT_second_batch_students_l1587_158774


namespace NUMINAMATH_GPT_num_of_cows_is_7_l1587_158796

variables (C H : ℕ)

-- Define the conditions
def cow_legs : ℕ := 4 * C
def chicken_legs : ℕ := 2 * H
def cow_heads : ℕ := C
def chicken_heads : ℕ := H

def total_legs : ℕ := cow_legs C + chicken_legs H
def total_heads : ℕ := cow_heads C + chicken_heads H
def legs_condition : Prop := total_legs C H = 2 * total_heads C H + 14

-- The theorem to be proved
theorem num_of_cows_is_7 (h : legs_condition C H) : C = 7 :=
by sorry

end NUMINAMATH_GPT_num_of_cows_is_7_l1587_158796


namespace NUMINAMATH_GPT_unique_cubic_coefficients_l1587_158738

noncomputable def cubic_function (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem unique_cubic_coefficients
  (a b c : ℝ)
  (h1 : ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) :
  (a = 0 ∧ b = -3 ∧ c = 0) :=
by
  sorry

end NUMINAMATH_GPT_unique_cubic_coefficients_l1587_158738


namespace NUMINAMATH_GPT_product_of_de_l1587_158720

theorem product_of_de (d e : ℤ) (h1: ∀ (r : ℝ), r^2 - r - 1 = 0 → r^6 - (d : ℝ) * r - (e : ℝ) = 0) : 
  d * e = 40 :=
by
  sorry

end NUMINAMATH_GPT_product_of_de_l1587_158720


namespace NUMINAMATH_GPT_find_radius_of_large_circle_l1587_158762

noncomputable def radius_of_large_circle (r : ℝ) : Prop :=
  let r_A := 3
  let r_B := 2
  let d := 6
  (r - r_A)^2 + (r - r_B)^2 + 2 * (r - r_A) * (r - r_B) = d^2 ∧
  r = (5 + Real.sqrt 33) / 2

theorem find_radius_of_large_circle : ∃ (r : ℝ), radius_of_large_circle r :=
by {
  sorry
}

end NUMINAMATH_GPT_find_radius_of_large_circle_l1587_158762


namespace NUMINAMATH_GPT_gcd_39_91_l1587_158719
-- Import the Mathlib library to ensure all necessary functions and theorems are available

-- Lean statement for proving the GCD of 39 and 91 is 13.
theorem gcd_39_91 : Nat.gcd 39 91 = 13 := by
  sorry

end NUMINAMATH_GPT_gcd_39_91_l1587_158719


namespace NUMINAMATH_GPT_circle_intersection_exists_l1587_158705

theorem circle_intersection_exists (a b : ℝ) :
  ∃ (m n : ℤ), (m - a)^2 + (n - b)^2 ≤ (1 / 14)^2 →
  ∀ x y, (x - a)^2 + (y - b)^2 = 100^2 :=
sorry

end NUMINAMATH_GPT_circle_intersection_exists_l1587_158705


namespace NUMINAMATH_GPT_inequality_solution_set_l1587_158730

theorem inequality_solution_set : 
  {x : ℝ | (x - 2) * (x + 1) ≤ 0} = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1587_158730


namespace NUMINAMATH_GPT_probability_genuine_coins_given_weight_condition_l1587_158787

/--
Given the following conditions:
- Ten counterfeit coins of equal weight are mixed with 20 genuine coins.
- The weight of a counterfeit coin is different from the weight of a genuine coin.
- Two pairs of coins are selected randomly without replacement from the 30 coins. 

Prove that the probability that all 4 selected coins are genuine, given that the combined weight
of the first pair is equal to the combined weight of the second pair, is 5440/5481.
-/
theorem probability_genuine_coins_given_weight_condition :
  let num_coins := 30
  let num_genuine := 20
  let num_counterfeit := 10
  let pairs_selected := 2
  let pairs_remaining := num_coins - pairs_selected * 2
  let P := (num_genuine / num_coins) * ((num_genuine - 1) / (num_coins - 1)) * ((num_genuine - 2) / pairs_remaining) * ((num_genuine - 3) / (pairs_remaining - 1))
  let event_A_given_B := P / (7 / 16)
  event_A_given_B = 5440 / 5481 := 
sorry

end NUMINAMATH_GPT_probability_genuine_coins_given_weight_condition_l1587_158787


namespace NUMINAMATH_GPT_hexagonal_tile_difference_l1587_158700

theorem hexagonal_tile_difference :
  let initial_blue_tiles := 15
  let initial_green_tiles := 9
  let new_green_border_tiles := 18
  let new_blue_border_tiles := 18
  let total_green_tiles := initial_green_tiles + new_green_border_tiles
  let total_blue_tiles := initial_blue_tiles + new_blue_border_tiles
  total_blue_tiles - total_green_tiles = 6 := by {
    sorry
  }

end NUMINAMATH_GPT_hexagonal_tile_difference_l1587_158700


namespace NUMINAMATH_GPT_david_is_30_l1587_158724

-- Definitions representing the conditions
def uncleBobAge : ℕ := 60
def emilyAge : ℕ := (2 * uncleBobAge) / 3
def davidAge : ℕ := emilyAge - 10

-- Statement that represents the equivalence to be proven
theorem david_is_30 : davidAge = 30 :=
by
  sorry

end NUMINAMATH_GPT_david_is_30_l1587_158724


namespace NUMINAMATH_GPT_minimum_jumps_to_cover_circle_l1587_158726

/--
Given 2016 points arranged in a circle and the ability to jump either 2 or 3 points clockwise,
prove that the minimum number of jumps required to visit every point at least once and return to the starting 
point is 2017.
-/
theorem minimum_jumps_to_cover_circle (n : Nat) (h : n = 2016) : 
  ∃ (a b : Nat), 2 * a + 3 * b = n ∧ (a + b) = 2017 := 
sorry

end NUMINAMATH_GPT_minimum_jumps_to_cover_circle_l1587_158726


namespace NUMINAMATH_GPT_abc_plus_ab_plus_a_div_4_l1587_158710

noncomputable def prob_abc_div_4 (a b c : ℕ) (isPositive_a : 0 < a) (isPositive_b : 0 < b) (isPositive_c : 0 < c) (a_in_range : a ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (b_in_range : b ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (c_in_range : c ∈ {k | 1 ≤ k ∧ k ≤ 2009}) : ℚ :=
  let total_elements : ℚ := 2009
  let multiples_of_4 := 502
  let non_multiples_of_4 := total_elements - multiples_of_4
  let prob_a_div_4 : ℚ := multiples_of_4 / total_elements
  let prob_a_not_div_4 : ℚ := non_multiples_of_4 / total_elements
  sorry

theorem abc_plus_ab_plus_a_div_4 : ∃ P : ℚ, prob_abc_div_4 a b c isPositive_a isPositive_b isPositive_c a_in_range b_in_range c_in_range = P :=
by sorry

end NUMINAMATH_GPT_abc_plus_ab_plus_a_div_4_l1587_158710


namespace NUMINAMATH_GPT_diff_roots_eq_sqrt_2p2_add_2p_sub_2_l1587_158740

theorem diff_roots_eq_sqrt_2p2_add_2p_sub_2 (p : ℝ) :
  let a := 1
  let b := -2 * p
  let c := p^2 - p + 1
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let r1 := (-b + sqrt_discriminant) / (2 * a)
  let r2 := (-b - sqrt_discriminant) / (2 * a)
  r1 - r2 = Real.sqrt (2*p^2 + 2*p - 2) :=
by
  sorry

end NUMINAMATH_GPT_diff_roots_eq_sqrt_2p2_add_2p_sub_2_l1587_158740


namespace NUMINAMATH_GPT_evaluate_expression_l1587_158780

theorem evaluate_expression : 8 * ((1 : ℚ) / 3)^3 - 1 = -19 / 27 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1587_158780


namespace NUMINAMATH_GPT_range_of_m_l1587_158791

variable (x y m : ℝ)

def system_of_eq1 := 2 * x + y = -4 * m + 5
def system_of_eq2 := x + 2 * y = m + 4
def inequality1 := x - y > -6
def inequality2 := x + y < 8

theorem range_of_m:
  system_of_eq1 x y m → 
  system_of_eq2 x y m → 
  inequality1 x y → 
  inequality2 x y → 
  -5 < m ∧ m < 7/5 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_range_of_m_l1587_158791


namespace NUMINAMATH_GPT_best_marksman_score_l1587_158707

theorem best_marksman_score (n : ℕ) (hypothetical_score : ℕ) (average_if_hypothetical : ℕ) (actual_total_score : ℕ) (H1 : n = 8) (H2 : hypothetical_score = 92) (H3 : average_if_hypothetical = 84) (H4 : actual_total_score = 665) :
    ∃ (actual_best_score : ℕ), actual_best_score = 77 :=
by
    have hypothetical_total_score : ℕ := 7 * average_if_hypothetical + hypothetical_score
    have difference : ℕ := hypothetical_total_score - actual_total_score
    use hypothetical_score - difference
    sorry

end NUMINAMATH_GPT_best_marksman_score_l1587_158707


namespace NUMINAMATH_GPT_tim_scored_sum_first_8_even_numbers_l1587_158767

-- Define the first 8 even numbers.
def first_8_even_numbers : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16]

-- Define the sum of those numbers.
def sum_first_8_even_numbers : ℕ := List.sum first_8_even_numbers

-- The theorem stating the problem.
theorem tim_scored_sum_first_8_even_numbers : sum_first_8_even_numbers = 72 := by
  sorry

end NUMINAMATH_GPT_tim_scored_sum_first_8_even_numbers_l1587_158767


namespace NUMINAMATH_GPT_find_roots_range_l1587_158713

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem find_roots_range 
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hx : -1 < -1/2 ∧ -1/2 < 0 ∧ 0 < 1/2 ∧ 1/2 < 1 ∧ 1 < 3/2 ∧ 3/2 < 2 ∧ 2 < 5/2 ∧ 5/2 < 3)
  (hy : ∀ {x : ℝ}, x = -1 → quadratic_function a b c x = -2 ∧
                   x = -1/2 → quadratic_function a b c x = -1/4 ∧
                   x = 0 → quadratic_function a b c x = 1 ∧
                   x = 1/2 → quadratic_function a b c x = 7/4 ∧
                   x = 1 → quadratic_function a b c x = 2 ∧
                   x = 3/2 → quadratic_function a b c x = 7/4 ∧
                   x = 2 → quadratic_function a b c x = 1 ∧
                   x = 5/2 → quadratic_function a b c x = -1/4 ∧
                   x = 3 → quadratic_function a b c x = -2) :
  ∃ x1 x2 : ℝ, -1/2 < x1 ∧ x1 < 0 ∧ 2 < x2 ∧ x2 < 5/2 ∧ quadratic_function a b c x1 = 0 ∧ quadratic_function a b c x2 = 0 :=
by sorry

end NUMINAMATH_GPT_find_roots_range_l1587_158713


namespace NUMINAMATH_GPT_range_of_m_l1587_158728

theorem range_of_m (m : ℝ) : (2 + m > 0) ∧ (1 - m > 0) ∧ (2 + m > 1 - m) → -1/2 < m ∧ m < 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_m_l1587_158728


namespace NUMINAMATH_GPT_work_completion_time_l1587_158709

theorem work_completion_time (B_rate A_rate Combined_rate : ℝ) (B_time : ℝ) :
  (B_rate = 1 / 60) →
  (A_rate = 4 * B_rate) →
  (Combined_rate = A_rate + B_rate) →
  (B_time = 1 / Combined_rate) →
  B_time = 12 :=
by sorry

end NUMINAMATH_GPT_work_completion_time_l1587_158709


namespace NUMINAMATH_GPT_expression_eq_16x_l1587_158708

variable (x y z w : ℝ)

theorem expression_eq_16x
  (h1 : y = 2 * x)
  (h2 : z = 3 * y)
  (h3 : w = z + x) :
  x + y + z + w = 16 * x :=
sorry

end NUMINAMATH_GPT_expression_eq_16x_l1587_158708


namespace NUMINAMATH_GPT_b_investment_l1587_158763

theorem b_investment (a_investment : ℝ) (c_investment : ℝ) (total_profit : ℝ) (a_share_profit : ℝ) (b_investment : ℝ) : a_investment = 6300 → c_investment = 10500 → total_profit = 14200 → a_share_profit = 4260 → b_investment = 4220 :=
by
  intro h_a h_c h_total h_a_share
  have h1 : 6300 / (6300 + 4220 + 10500) = 4260 / 14200 := sorry
  have h2 : 6300 * 14200 = 4260 * (6300 + 4220 + 10500) := sorry
  have h3 : b_investment = 4220 := sorry
  exact h3

end NUMINAMATH_GPT_b_investment_l1587_158763


namespace NUMINAMATH_GPT_union_M_N_intersection_complementM_N_l1587_158753

open Set  -- Open the Set namespace for convenient notation.

noncomputable def funcDomain : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def setN : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}
noncomputable def complementFuncDomain : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 2}

theorem union_M_N :
  (funcDomain ∪ setN) = {x : ℝ | -1 ≤ x ∧ x < 3} :=
by
  sorry

theorem intersection_complementM_N :
  (complementFuncDomain ∩ setN) = {x : ℝ | 2 ≤ x ∧ x < 3} :=
by
  sorry

end NUMINAMATH_GPT_union_M_N_intersection_complementM_N_l1587_158753


namespace NUMINAMATH_GPT_number_of_paths_l1587_158747

theorem number_of_paths (r u : ℕ) (h_r : r = 5) (h_u : u = 4) : 
  (Nat.choose (r + u) u) = 126 :=
by
  -- The proof is omitted, as requested.
  sorry

end NUMINAMATH_GPT_number_of_paths_l1587_158747


namespace NUMINAMATH_GPT_units_digit_G_1000_l1587_158799

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_G_1000 : (G 1000) % 10 = 4 :=
  sorry

end NUMINAMATH_GPT_units_digit_G_1000_l1587_158799


namespace NUMINAMATH_GPT_magic_8_ball_probability_l1587_158782

theorem magic_8_ball_probability :
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  (Nat.choose 7 3) * (p^3) * (q^4) = 590625 / 2097152 :=
by
  let p := 3 / 8
  let q := 5 / 8
  let n := 7
  let k := 3
  sorry

end NUMINAMATH_GPT_magic_8_ball_probability_l1587_158782


namespace NUMINAMATH_GPT_div_fraction_l1587_158761

/-- The result of dividing 3/7 by 2 1/2 equals 6/35 -/
theorem div_fraction : (3/7) / (2 + 1/2) = 6/35 :=
by 
  sorry

end NUMINAMATH_GPT_div_fraction_l1587_158761


namespace NUMINAMATH_GPT_systematic_sampling_l1587_158714

theorem systematic_sampling (total_employees groups group_size draw_5th draw_10th : ℕ)
  (h1 : total_employees = 200)
  (h2 : groups = 40)
  (h3 : group_size = total_employees / groups)
  (h4 : draw_5th = 22)
  (h5 : ∃ x : ℕ, draw_5th = (5-1) * group_size + x)
  (h6 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ groups → draw_10th = (k-1) * group_size + x) :
  draw_10th = 47 := 
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_l1587_158714


namespace NUMINAMATH_GPT_number_of_thrown_out_carrots_l1587_158760

-- Definitions from the conditions
def initial_carrots : ℕ := 48
def picked_next_day : ℕ := 42
def total_carrots : ℕ := 45

-- Proposition stating the problem
theorem number_of_thrown_out_carrots (x : ℕ) : initial_carrots - x + picked_next_day = total_carrots → x = 45 :=
by
  sorry

end NUMINAMATH_GPT_number_of_thrown_out_carrots_l1587_158760


namespace NUMINAMATH_GPT_extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l1587_158778

theorem extremum_implies_derivative_zero {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_extremum : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :
  deriv f x₀ = 0 :=
sorry

theorem derivative_zero_not_implies_extremum {f : ℝ → ℝ} {x₀ : ℝ}
    (h_deriv : DifferentiableAt ℝ f x₀) (h_deriv_zero : deriv f x₀ = 0) :
  ¬ (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → f x ≤ f x₀ ∨ f x ≥ f x₀) :=
sorry

end NUMINAMATH_GPT_extremum_implies_derivative_zero_derivative_zero_not_implies_extremum_l1587_158778


namespace NUMINAMATH_GPT_proof_problem_l1587_158770

variable {a b x y : ℝ}

def dollar (a b : ℝ) : ℝ := (a - b) ^ 2

theorem proof_problem : dollar ((x + y) ^ 2) (y ^ 2 + x ^ 2) = 4 * x ^ 2 * y ^ 2 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1587_158770


namespace NUMINAMATH_GPT_john_heroes_on_large_sheets_front_l1587_158776

noncomputable def num_pictures_on_large_sheets_front : ℕ :=
  let total_pictures := 20
  let minutes_spent := 75 - 5
  let average_time_per_picture := 5
  let front_pictures := total_pictures / 2
  let x := front_pictures / 3
  2 * x

theorem john_heroes_on_large_sheets_front : num_pictures_on_large_sheets_front = 6 :=
by
  sorry

end NUMINAMATH_GPT_john_heroes_on_large_sheets_front_l1587_158776


namespace NUMINAMATH_GPT_three_fifths_difference_products_l1587_158779

theorem three_fifths_difference_products :
  (3 / 5) * ((7 * 9) - (4 * 3)) = 153 / 5 :=
by
  sorry

end NUMINAMATH_GPT_three_fifths_difference_products_l1587_158779


namespace NUMINAMATH_GPT_min_overlap_l1587_158758

noncomputable def drinks_coffee := 0.60
noncomputable def drinks_tea := 0.50
noncomputable def drinks_neither := 0.10
noncomputable def drinks_either := 1 - drinks_neither
noncomputable def total_overlap := drinks_coffee + drinks_tea - drinks_either

theorem min_overlap (hcoffee : drinks_coffee = 0.60) (htea : drinks_tea = 0.50) (hneither : drinks_neither = 0.10) :
  total_overlap = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_min_overlap_l1587_158758


namespace NUMINAMATH_GPT_sequence_geometric_and_general_formula_find_minimum_n_l1587_158798

theorem sequence_geometric_and_general_formula 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n) :
  (∀ n : ℕ, n ≥ 1 → a (n + 1) + 1 = 2 * (a n + 1)) ∧ (∀ n : ℕ, n ≥ 1 → a n = 2^n - 1) :=
sorry

theorem find_minimum_n 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (b T : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → S n + n = 2 * a n)
  (h2 : ∀ n : ℕ, b n = (2 * n + 1) * a n + (2 * n + 1))
  (h3 : T 0 = 0)
  (h4 : ∀ n : ℕ, T (n + 1) = T n + b (n + 1)) :
  ∃ n : ℕ, n ≥ 1 ∧ (T n - 2) / (2 * n - 1) > 2010 :=
sorry

end NUMINAMATH_GPT_sequence_geometric_and_general_formula_find_minimum_n_l1587_158798


namespace NUMINAMATH_GPT_Ariel_current_age_l1587_158781

-- Define the conditions
def Ariel_birth_year : Nat := 1992
def Ariel_start_fencing_year : Nat := 2006
def Ariel_fencing_years : Nat := 16

-- Define the problem as a theorem
theorem Ariel_current_age :
  (Ariel_start_fencing_year - Ariel_birth_year) + Ariel_fencing_years = 30 := by
sorry

end NUMINAMATH_GPT_Ariel_current_age_l1587_158781


namespace NUMINAMATH_GPT_root_analysis_l1587_158755

noncomputable def root1 (a : ℝ) : ℝ :=
2 * a + 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def root2 (a : ℝ) : ℝ :=
2 * a - 2 * Real.sqrt (a^2 - 3 * a + 2)

noncomputable def derivedRoot (a : ℝ) : ℝ :=
(3 * a - 2) / a

theorem root_analysis (a : ℝ) (ha : a > 0) :
( (2/3 ≤ a ∧ a < 1) ∨ (2 < a) → (root1 a ≥ 0 ∧ root2 a ≥ 0)) ∧
( 0 < a ∧ a < 2/3 → (derivedRoot a < 0 ∧ root1 a ≥ 0)) :=
sorry

end NUMINAMATH_GPT_root_analysis_l1587_158755
