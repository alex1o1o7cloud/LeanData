import Mathlib

namespace NUMINAMATH_GPT_first_common_digit_three_digit_powers_l1223_122312

theorem first_common_digit_three_digit_powers (m n: ℕ) (hm: 100 ≤ 2^m ∧ 2^m < 1000) (hn: 100 ≤ 3^n ∧ 3^n < 1000) :
  (∃ d, (2^m).div 100 = d ∧ (3^n).div 100 = d ∧ d = 2) :=
sorry

end NUMINAMATH_GPT_first_common_digit_three_digit_powers_l1223_122312


namespace NUMINAMATH_GPT_complete_the_square_l1223_122394

theorem complete_the_square (x : ℝ) : 
  x^2 - 2 * x - 5 = 0 ↔ (x - 1)^2 = 6 := 
by {
  -- This is where you would provide the proof
  sorry
}

end NUMINAMATH_GPT_complete_the_square_l1223_122394


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l1223_122317

theorem repeating_decimal_as_fraction :
  (∃ y : ℚ, y = 737910 ∧ 0.73 + 864 / 999900 = y / 999900) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l1223_122317


namespace NUMINAMATH_GPT_carter_total_drum_sticks_l1223_122353

def sets_per_show_used := 5
def sets_per_show_tossed := 6
def nights := 30

theorem carter_total_drum_sticks : 
  (sets_per_show_used + sets_per_show_tossed) * nights = 330 := by
  sorry

end NUMINAMATH_GPT_carter_total_drum_sticks_l1223_122353


namespace NUMINAMATH_GPT_simplify_polynomial_l1223_122310

theorem simplify_polynomial (s : ℝ) :
  (2*s^2 + 5*s - 3) - (2*s^2 + 9*s - 7) = -4*s + 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l1223_122310


namespace NUMINAMATH_GPT_sufficient_necessary_condition_l1223_122377

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * a * x^3 + (1 / 2) * a * x^2 - 2 * a * x + 2 * a + 1

theorem sufficient_necessary_condition (a : ℝ) :
  (-6 / 5 < a ∧ a < -3 / 16) ↔
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧
   (∃ c₁ c₂ : ℝ, deriv (f a) c₁ = 0 ∧ deriv (f a) c₂ = 0 ∧
   deriv (deriv (f a)) c₁ < 0 ∧ deriv (deriv (f a)) c₂ > 0 ∧
   f a c₁ > 0 ∧ f a c₂ < 0)) := sorry

end NUMINAMATH_GPT_sufficient_necessary_condition_l1223_122377


namespace NUMINAMATH_GPT_sum_of_smallest_and_second_smallest_l1223_122351

-- Define the set of numbers
def numbers : Set ℕ := {10, 11, 12, 13}

-- Define the smallest and second smallest numbers
def smallest_number : ℕ := 10
def second_smallest_number : ℕ := 11

-- Prove the sum of the smallest and the second smallest numbers
theorem sum_of_smallest_and_second_smallest : smallest_number + second_smallest_number = 21 := by
  sorry

end NUMINAMATH_GPT_sum_of_smallest_and_second_smallest_l1223_122351


namespace NUMINAMATH_GPT_find_x_l1223_122374

theorem find_x
  (x : ℝ)
  (h : 5^29 * x^15 = 2 * 10^29) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1223_122374


namespace NUMINAMATH_GPT_has_two_distinct_roots_and_ordered_l1223_122371

-- Define the context and the conditions of the problem.
variables (a b c : ℝ) (h : a < b) (h2 : b < c)

-- Define the quadratic function derived from the problem.
def quadratic (x : ℝ) : ℝ :=
  (x - a) * (x - b) + (x - a) * (x - c) + (x - b) * (x - c)

-- State the main theorem.
theorem has_two_distinct_roots_and_ordered:
  ∃ x1 x2 : ℝ, quadratic a b c x1 = 0 ∧ quadratic a b c x2 = 0 ∧ a < x1 ∧ x1 < b ∧ b < x2 ∧ x2 < c :=
sorry

end NUMINAMATH_GPT_has_two_distinct_roots_and_ordered_l1223_122371


namespace NUMINAMATH_GPT_product_of_roots_l1223_122383

variable {x1 x2 : ℝ}

theorem product_of_roots (hx1 : x1 * Real.log x1 = 2006) (hx2 : x2 * Real.exp x2 = 2006) : x1 * x2 = 2006 :=
sorry

end NUMINAMATH_GPT_product_of_roots_l1223_122383


namespace NUMINAMATH_GPT_body_diagonal_length_l1223_122384

theorem body_diagonal_length (a b c : ℝ) (h1 : a * b = 6) (h2 : a * c = 8) (h3 : b * c = 12) :
  (a^2 + b^2 + c^2 = 29) :=
by
  sorry

end NUMINAMATH_GPT_body_diagonal_length_l1223_122384


namespace NUMINAMATH_GPT_min_value_inv_sum_l1223_122392

open Real

theorem min_value_inv_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  3 ≤ (1 / x) + (1 / y) + (1 / z) :=
sorry

end NUMINAMATH_GPT_min_value_inv_sum_l1223_122392


namespace NUMINAMATH_GPT_find_larger_number_l1223_122323

theorem find_larger_number (x y : ℝ) (h1 : x - y = 1860) (h2 : 0.075 * x = 0.125 * y) :
  x = 4650 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l1223_122323


namespace NUMINAMATH_GPT_find_real_solutions_l1223_122358

theorem find_real_solutions (x : ℝ) : 
  x^4 + (3 - x)^4 = 130 ↔ x = 1.5 + Real.sqrt 1.5 ∨ x = 1.5 - Real.sqrt 1.5 :=
sorry

end NUMINAMATH_GPT_find_real_solutions_l1223_122358


namespace NUMINAMATH_GPT_cube_inequality_l1223_122368

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 :=
by
  sorry

end NUMINAMATH_GPT_cube_inequality_l1223_122368


namespace NUMINAMATH_GPT_roots_sum_and_product_l1223_122390

theorem roots_sum_and_product (p q : ℝ) (h_sum : p / 3 = 9) (h_prod : q / 3 = 24) : p + q = 99 :=
by
  -- We are given h_sum: p / 3 = 9
  -- We are given h_prod: q / 3 = 24
  -- We need to prove p + q = 99
  sorry

end NUMINAMATH_GPT_roots_sum_and_product_l1223_122390


namespace NUMINAMATH_GPT_incenter_divides_segment_l1223_122333

variables (A B C I M : Type) (R r : ℝ)

-- Definitions based on conditions
def is_incenter (I : Type) (A B C : Type) : Prop := sorry
def is_circumcircle (C : Type) : Prop := sorry
def angle_bisector_intersects_at (A B C M : Type) : Prop := sorry
def divides_segment (I M : Type) (a b : ℝ) : Prop := sorry

-- Proof problem statement
theorem incenter_divides_segment (h1 : is_circumcircle C)
                                   (h2 : is_incenter I A B C)
                                   (h3 : angle_bisector_intersects_at A B C M)
                                   (h4 : divides_segment I M a b) :
  a * b = 2 * R * r :=
sorry

end NUMINAMATH_GPT_incenter_divides_segment_l1223_122333


namespace NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l1223_122339

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log x - a / x

theorem monotonicity_of_f (a : ℝ) (h : 0 < a) :
  ∀ x y : ℝ, (0 < x) → (0 < y) → (x < y) → (f x a < f y a) :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a < x ^ 2) ↔ (-1 ≤ a) :=
by
  sorry

end NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l1223_122339


namespace NUMINAMATH_GPT_avg_height_is_28_l1223_122322

-- Define the height relationship between trees
def height_relation (a b : ℕ) := a = 2 * b ∨ a = b / 2

-- Given tree heights (partial information)
def height_tree_2 := 14
def height_tree_5 := 20

-- Define the tree heights variables
variables (height_tree_1 height_tree_3 height_tree_4 height_tree_6 : ℕ)

-- Conditions based on the given data and height relations
axiom h1 : height_relation height_tree_1 height_tree_2
axiom h2 : height_relation height_tree_2 height_tree_3
axiom h3 : height_relation height_tree_3 height_tree_4
axiom h4 : height_relation height_tree_4 height_tree_5
axiom h5 : height_relation height_tree_5 height_tree_6

-- Compute total and average height
def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4 + height_tree_5 + height_tree_6
def average_height := total_height / 6

-- Prove the average height is 28 meters
theorem avg_height_is_28 : average_height = 28 := by
  sorry

end NUMINAMATH_GPT_avg_height_is_28_l1223_122322


namespace NUMINAMATH_GPT_bush_height_at_2_years_l1223_122342

theorem bush_height_at_2_years (H: ℕ → ℕ) 
  (quadruple_height: ∀ (n: ℕ), H (n+1) = 4 * H n)
  (H_4: H 4 = 64) : H 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_bush_height_at_2_years_l1223_122342


namespace NUMINAMATH_GPT_find_c_l1223_122309

variable {r s b c : ℚ}

-- Conditions based on roots of the original quadratic equation
def roots_of_original_quadratic (r s : ℚ) := 
  (5 * r ^ 2 - 8 * r + 2 = 0) ∧ (5 * s ^ 2 - 8 * s + 2 = 0)

-- New quadratic equation with roots shifted by 3
def new_quadratic_roots (r s b c : ℚ) :=
  (r - 3) + (s - 3) = -b ∧ (r - 3) * (s - 3) = c 

theorem find_c (r s : ℚ) (hb : b = 22/5) : 
  (roots_of_original_quadratic r s) → 
  (new_quadratic_roots r s b c) → 
  c = 23/5 := 
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_c_l1223_122309


namespace NUMINAMATH_GPT_total_carrots_grown_l1223_122315

theorem total_carrots_grown :
  let Sandy := 6.5
  let Sam := 3.25
  let Sophie := 2.75 * Sam
  let Sara := (Sandy + Sam + Sophie) - 7.5
  Sandy + Sam + Sophie + Sara = 29.875 :=
by
  sorry

end NUMINAMATH_GPT_total_carrots_grown_l1223_122315


namespace NUMINAMATH_GPT_at_least_half_team_B_can_serve_on_submarine_l1223_122356

theorem at_least_half_team_B_can_serve_on_submarine
    (max_height : ℕ)
    (team_A_avg_height : ℕ)
    (team_B_median_height : ℕ)
    (team_C_tallest_height : ℕ)
    (team_D_mode_height : ℕ)
    (h1 : max_height = 168)
    (h2 : team_A_avg_height = 166)
    (h3 : team_B_median_height = 167)
    (h4 : team_C_tallest_height = 169)
    (h5 : team_D_mode_height = 167) :
  ∀ (height : ℕ), height ≤ max_height → ∃ (b_sailors : ℕ → Prop) (H : ∃ n, b_sailors n),
  (∃ (n_half : ℕ), (∀ h ≤ team_B_median_height, b_sailors h) ∧ (2 * n_half ≤ n)) :=
sorry

end NUMINAMATH_GPT_at_least_half_team_B_can_serve_on_submarine_l1223_122356


namespace NUMINAMATH_GPT_find_m_given_root_of_quadratic_l1223_122329

theorem find_m_given_root_of_quadratic (m : ℝ) : (∃ x : ℝ, x = 3 ∧ x^2 - m * x - 6 = 0) → m = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_m_given_root_of_quadratic_l1223_122329


namespace NUMINAMATH_GPT_range_of_t_l1223_122379

theorem range_of_t (a b c t: ℝ) 
  (h1 : 6 * a = 2 * b - 6)
  (h2 : 6 * a = 3 * c)
  (h3 : b ≥ 0)
  (h4 : c ≤ 2)
  (h5 : t = 2 * a + b - c) : 
  0 ≤ t ∧ t ≤ 6 :=
sorry

end NUMINAMATH_GPT_range_of_t_l1223_122379


namespace NUMINAMATH_GPT_minimize_wood_frame_l1223_122386

noncomputable def min_wood_frame (x y : ℝ) : Prop :=
  let area_eq : Prop := x * y + x^2 / 4 = 8
  let length := 2 * (x + y) + Real.sqrt 2 * x
  let y_expr := 8 / x - x / 4
  let length_expr := (3 / 2 + Real.sqrt 2) * x + 16 / x
  let min_x := Real.sqrt (16 / (3 / 2 + Real.sqrt 2))
  area_eq ∧ y = y_expr ∧ length = length_expr ∧ x = 2.343 ∧ y = 2.828

theorem minimize_wood_frame : ∃ x y : ℝ, min_wood_frame x y :=
by
  use 2.343
  use 2.828
  unfold min_wood_frame
  -- we leave the proof of the properties as sorry
  sorry

end NUMINAMATH_GPT_minimize_wood_frame_l1223_122386


namespace NUMINAMATH_GPT_least_integer_condition_l1223_122341

theorem least_integer_condition : ∃ x : ℤ, (x^2 = 2 * x + 72) ∧ (x = -6) :=
sorry

end NUMINAMATH_GPT_least_integer_condition_l1223_122341


namespace NUMINAMATH_GPT_rowing_speed_in_still_water_l1223_122334

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.5) 
(h2 : ∀ t : ℝ, (v + c) * t = (v - c) * 2 * t) : 
  v = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_rowing_speed_in_still_water_l1223_122334


namespace NUMINAMATH_GPT_average_goals_increase_l1223_122343

theorem average_goals_increase (A : ℚ) (h1 : 4 * A + 2 = 4) : (4 / 5 - A) = 0.3 := by
  sorry

end NUMINAMATH_GPT_average_goals_increase_l1223_122343


namespace NUMINAMATH_GPT_inequality_geq_l1223_122370

theorem inequality_geq (t : ℝ) (n : ℕ) (ht : t ≥ 1/2) : 
  t^(2*n) ≥ (t-1)^(2*n) + (2*t-1)^n := 
sorry

end NUMINAMATH_GPT_inequality_geq_l1223_122370


namespace NUMINAMATH_GPT_am_gm_inequality_l1223_122389

theorem am_gm_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * (a * b * c) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l1223_122389


namespace NUMINAMATH_GPT_negation_even_l1223_122311

open Nat

theorem negation_even (x : ℕ) (h : 0 < x) :
  (∀ x : ℕ, 0 < x → Even x) ↔ ¬ (∃ x : ℕ, 0 < x ∧ Odd x) :=
by
  sorry

end NUMINAMATH_GPT_negation_even_l1223_122311


namespace NUMINAMATH_GPT_jackson_email_problem_l1223_122388

variables (E_0 E_1 E_2 E_3 X : ℕ)

/-- Jackson's email deletion and receipt problem -/
theorem jackson_email_problem
  (h1 : E_1 = E_0 - 50 + 15)
  (h2 : E_2 = E_1 - X + 5)
  (h3 : E_3 = E_2 + 10)
  (h4 : E_3 = 30) :
  X = 50 :=
sorry

end NUMINAMATH_GPT_jackson_email_problem_l1223_122388


namespace NUMINAMATH_GPT_point_A_in_first_quadrant_l1223_122301

def point_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_A_in_first_quadrant : point_in_first_quadrant 1 2 := by
  sorry

end NUMINAMATH_GPT_point_A_in_first_quadrant_l1223_122301


namespace NUMINAMATH_GPT_expression_eval_l1223_122338

theorem expression_eval : 2 * 3 + 2 * 3 = 12 := by
  sorry

end NUMINAMATH_GPT_expression_eval_l1223_122338


namespace NUMINAMATH_GPT_count_4_digit_divisible_by_45_l1223_122369

theorem count_4_digit_divisible_by_45 : 
  ∃ n, n = 11 ∧ (∀ a b : ℕ, a + b = 2 ∨ a + b = 11 → (20 + b * 10 + 5) % 45 = 0) :=
sorry

end NUMINAMATH_GPT_count_4_digit_divisible_by_45_l1223_122369


namespace NUMINAMATH_GPT_cos_value_l1223_122372

theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by sorry

end NUMINAMATH_GPT_cos_value_l1223_122372


namespace NUMINAMATH_GPT_inequality_lemma_l1223_122307

theorem inequality_lemma (x y z : ℝ) (h1 : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z)
    (h2 : (1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1)) :
    (1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1) := 
by
  sorry

end NUMINAMATH_GPT_inequality_lemma_l1223_122307


namespace NUMINAMATH_GPT_lillian_candies_total_l1223_122375

variable (initial_candies : ℕ)
variable (candies_given_by_father : ℕ)

theorem lillian_candies_total (initial_candies : ℕ) (candies_given_by_father : ℕ) :
  initial_candies = 88 →
  candies_given_by_father = 5 →
  initial_candies + candies_given_by_father = 93 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lillian_candies_total_l1223_122375


namespace NUMINAMATH_GPT_least_number_of_groups_l1223_122350

def num_students : ℕ := 24
def max_students_per_group : ℕ := 10

theorem least_number_of_groups : ∃ x, ∀ y, y ≤ max_students_per_group ∧ num_students = x * y → x = 3 := by
  sorry

end NUMINAMATH_GPT_least_number_of_groups_l1223_122350


namespace NUMINAMATH_GPT_average_of_first_5_multiples_of_5_l1223_122324

theorem average_of_first_5_multiples_of_5 : 
  (5 + 10 + 15 + 20 + 25) / 5 = 15 :=
by
  sorry

end NUMINAMATH_GPT_average_of_first_5_multiples_of_5_l1223_122324


namespace NUMINAMATH_GPT_find_positive_root_l1223_122352

open Real

theorem find_positive_root 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (x : ℝ) :
  sqrt (a * b * x * (a + b + x)) + sqrt (b * c * x * (b + c + x)) + sqrt (c * a * x * (c + a + x)) = sqrt (a * b * c * (a + b + c)) →
  x = (a * b * c) / (a * b + b * c + c * a + 2 * sqrt (a * b * c * (a + b + c))) := 
sorry

end NUMINAMATH_GPT_find_positive_root_l1223_122352


namespace NUMINAMATH_GPT_percent_republicans_voting_for_A_l1223_122357

theorem percent_republicans_voting_for_A (V : ℝ) (percent_Democrats : ℝ) 
  (percent_Republicans : ℝ) (percent_D_voting_for_A : ℝ) 
  (percent_total_voting_for_A : ℝ) (R : ℝ) 
  (h1 : percent_Democrats = 0.60)
  (h2 : percent_Republicans = 0.40)
  (h3 : percent_D_voting_for_A = 0.85)
  (h4 : percent_total_voting_for_A = 0.59) :
  R = 0.2 :=
by 
  sorry

end NUMINAMATH_GPT_percent_republicans_voting_for_A_l1223_122357


namespace NUMINAMATH_GPT_smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2_l1223_122381

theorem smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2 :
  ∀ (x y z w : ℝ),
    x = Real.sqrt 3 →
    y = -1 / 3 →
    z = -2 →
    w = 0 →
    (x > 1) ∧ (y < 0) ∧ (z < 0) ∧ (|y| = 1 / 3) ∧ (|z| = 2) ∧ (w = 0) →
    min (min (min x y) z) w = z :=
by
  intros x y z w hx hy hz hw hcond
  sorry

end NUMINAMATH_GPT_smallest_real_number_among_sqrt3_neg13_neg2_and_0_is_neg2_l1223_122381


namespace NUMINAMATH_GPT_female_muscovy_ducks_l1223_122399

theorem female_muscovy_ducks :
  let total_ducks := 40
  let muscovy_percentage := 0.5
  let female_muscovy_percentage := 0.3
  let muscovy_ducks := total_ducks * muscovy_percentage
  let female_muscovy_ducks := muscovy_ducks * female_muscovy_percentage
  female_muscovy_ducks = 6 :=
by
  sorry

end NUMINAMATH_GPT_female_muscovy_ducks_l1223_122399


namespace NUMINAMATH_GPT_min_value_expression_l1223_122328

noncomputable def expression (x : ℝ) : ℝ :=
  Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((x - 2)^2 + (x + 2)^2)

theorem min_value_expression : ∃ x : ℝ, expression x = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1223_122328


namespace NUMINAMATH_GPT_winning_candidate_percentage_votes_l1223_122355

theorem winning_candidate_percentage_votes
  (total_votes : ℕ) (majority_votes : ℕ) (P : ℕ) 
  (h1 : total_votes = 6500) 
  (h2 : majority_votes = 1300) 
  (h3 : (P * total_votes) / 100 - ((100 - P) * total_votes) / 100 = majority_votes) : 
  P = 60 :=
sorry

end NUMINAMATH_GPT_winning_candidate_percentage_votes_l1223_122355


namespace NUMINAMATH_GPT_plant_arrangement_count_l1223_122387

-- Define the count of identical plants
def basil_count := 3
def aloe_count := 2

-- Define the count of identical lamps in each color
def white_lamp_count := 3
def red_lamp_count := 3

-- Define the total ways to arrange the plants under the lamps.
def arrangement_ways := 128

-- Formalize the problem statement proving the arrangements count
theorem plant_arrangement_count :
  (∃ f : Fin (basil_count + aloe_count) → Fin (white_lamp_count + red_lamp_count), True) ↔
  arrangement_ways = 128 :=
sorry

end NUMINAMATH_GPT_plant_arrangement_count_l1223_122387


namespace NUMINAMATH_GPT_incorrect_statements_l1223_122395

-- Definitions based on conditions from the problem.

def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop := a * x^2 + b * x + c < 0
def solution_set (a b c : ℝ) : Set ℝ := {x | quadratic_inequality a b c x}
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Lean statements of the conditions and the final proof problem.
theorem incorrect_statements (a b c : ℝ) (M : Set ℝ) :
  (M = ∅ → (a < 0 ∧ discriminant a b c < 0) → false) ∧
  (M = {x | x ≠ x0} → a < b → (a + 4 * c) / (b - a) = 2 + 2 * Real.sqrt 2 → false) := sorry

end NUMINAMATH_GPT_incorrect_statements_l1223_122395


namespace NUMINAMATH_GPT_find_s_range_l1223_122330

variables {a b c s t y1 y2 : ℝ}

-- Conditions
def is_vertex (a b c s t : ℝ) : Prop := ∀ x : ℝ, (a * x^2 + b * x + c = a * (x - s)^2 + t)

def passes_points (a b c y1 y2 : ℝ) : Prop := 
  (a * (-2)^2 + b * (-2) + c = y1) ∧ (a * 4^2 + b * 4 + c = y2)

def valid_constants (a y1 y2 t : ℝ) : Prop := 
  (a ≠ 0) ∧ (y1 > y2) ∧ (y2 > t)

-- Theorem
theorem find_s_range {a b c s t y1 y2 : ℝ}
  (hv : is_vertex a b c s t)
  (hp : passes_points a b c y1 y2)
  (vc : valid_constants a y1 y2 t) : 
  s > 1 ∧ s ≠ 4 :=
sorry -- Proof skipped

end NUMINAMATH_GPT_find_s_range_l1223_122330


namespace NUMINAMATH_GPT_find_k_l1223_122300

def otimes (a b : ℝ) := a * b + a + b^2

theorem find_k (k : ℝ) (h1 : otimes 1 k = 2) (h2 : 0 < k) :
  k = 1 :=
sorry

end NUMINAMATH_GPT_find_k_l1223_122300


namespace NUMINAMATH_GPT_students_in_class_C_l1223_122316

theorem students_in_class_C 
    (total_students : ℕ := 80) 
    (percent_class_A : ℕ := 40) 
    (class_B_difference : ℕ := 21) 
    (h_percent : percent_class_A = 40) 
    (h_class_B_diff : class_B_difference = 21) 
    (h_total_students : total_students = 80) : 
    total_students - ((percent_class_A * total_students) / 100 - class_B_difference + (percent_class_A * total_students) / 100) = 37 := by
    sorry

end NUMINAMATH_GPT_students_in_class_C_l1223_122316


namespace NUMINAMATH_GPT_gamma_distribution_moments_l1223_122346

noncomputable def gamma_density (α β x : ℝ) : ℝ :=
  (1 / (β ^ (α + 1) * Real.Gamma (α + 1))) * x ^ α * Real.exp (-x / β)

open Real

theorem gamma_distribution_moments (α β : ℝ) (x_bar D_B : ℝ) (hα : α > -1) (hβ : β > 0) :
  α = x_bar ^ 2 / D_B - 1 ∧ β = D_B / x_bar :=
by
  sorry

end NUMINAMATH_GPT_gamma_distribution_moments_l1223_122346


namespace NUMINAMATH_GPT_number_of_legs_twice_heads_diff_eq_22_l1223_122314

theorem number_of_legs_twice_heads_diff_eq_22 (P H : ℕ) (L : ℤ) (Heads : ℕ) (X : ℤ) (h1 : P = 11)
  (h2 : L = 4 * P + 2 * H) (h3 : Heads = P + H) (h4 : L = 2 * Heads + X) : X = 22 :=
by
  sorry

end NUMINAMATH_GPT_number_of_legs_twice_heads_diff_eq_22_l1223_122314


namespace NUMINAMATH_GPT_sin_6phi_l1223_122365

theorem sin_6phi (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * (Real.sqrt 8)) / 5) : 
  Real.sin (6 * φ) = -198 * Real.sqrt 2 / 15625 :=
by
  sorry

end NUMINAMATH_GPT_sin_6phi_l1223_122365


namespace NUMINAMATH_GPT_subset_condition_l1223_122302

theorem subset_condition (m : ℝ) (A : Set ℝ) (B : Set ℝ) :
  A = {1, 3} ∧ B = {1, 2, m} ∧ A ⊆ B → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_subset_condition_l1223_122302


namespace NUMINAMATH_GPT_functions_increasing_in_interval_l1223_122326

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x

theorem functions_increasing_in_interval :
  ∀ x, -Real.pi / 4 < x → x < Real.pi / 4 →
  (f x < f (x + 1e-6)) ∧ (g x < g (x + 1e-6)) :=
sorry

end NUMINAMATH_GPT_functions_increasing_in_interval_l1223_122326


namespace NUMINAMATH_GPT_tetrahedron_area_theorem_l1223_122376

noncomputable def tetrahedron_faces_areas_and_angles
  (a b c d : ℝ) (α β γ : ℝ) : Prop :=
  d^2 = a^2 + b^2 + c^2 - 2 * a * b * Real.cos γ - 2 * b * c * Real.cos α - 2 * c * a * Real.cos β

theorem tetrahedron_area_theorem
  (a b c d : ℝ) (α β γ : ℝ) :
  tetrahedron_faces_areas_and_angles a b c d α β γ :=
sorry

end NUMINAMATH_GPT_tetrahedron_area_theorem_l1223_122376


namespace NUMINAMATH_GPT_A_inter_B_eq_l1223_122308

-- Define set A based on the condition for different integer k.
def A (k : ℤ) : Set ℝ := {x | 2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi}

-- Define set B based on its condition.
def B : Set ℝ := {x | -5 ≤ x ∧ x < 4}

-- The final proof problem to show A ∩ B equals to the given set.
theorem A_inter_B_eq : 
  (⋃ k : ℤ, A k) ∩ B = {x | (-Real.pi < x ∧ x < 0) ∨ (Real.pi < x ∧ x < 4)} :=
by
  sorry

end NUMINAMATH_GPT_A_inter_B_eq_l1223_122308


namespace NUMINAMATH_GPT_no_roots_less_than_x0_l1223_122336

theorem no_roots_less_than_x0
  (x₀ a b c d : ℝ)
  (h₁ : ∀ x ≥ x₀, x^2 + a * x + b > 0)
  (h₂ : ∀ x ≥ x₀, x^2 + c * x + d > 0) :
  ∀ x ≥ x₀, x^2 + ((a + c) / 2) * x + ((b + d) / 2) > 0 := 
by
  sorry

end NUMINAMATH_GPT_no_roots_less_than_x0_l1223_122336


namespace NUMINAMATH_GPT_positive_abc_l1223_122397

theorem positive_abc (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 := 
by
  sorry

end NUMINAMATH_GPT_positive_abc_l1223_122397


namespace NUMINAMATH_GPT_intersection_M_N_l1223_122319

open Set Real

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - abs x)

def M : Set ℝ := {x | x^2 - x ≤ 0}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1223_122319


namespace NUMINAMATH_GPT_maximum_value_expression_maximum_value_expression_achieved_l1223_122364

theorem maximum_value_expression (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  (1 / (x^2 - 4 * x + 9) + 1 / (y^2 - 4 * y + 9) + 1 / (z^2 - 4 * z + 9)) ≤ 7 / 18 :=
sorry

theorem maximum_value_expression_achieved :
  (1 / (0^2 - 4 * 0 + 9) + 1 / (0^2 - 4 * 0 + 9) + 1 / (1^2 - 4 * 1 + 9)) = 7 / 18 :=
sorry

end NUMINAMATH_GPT_maximum_value_expression_maximum_value_expression_achieved_l1223_122364


namespace NUMINAMATH_GPT_range_of_m_l1223_122361

theorem range_of_m (m : ℝ) (x0 : ℝ)
  (h : (4^(-x0) - m * 2^(-x0 + 1)) = -(4^x0 - m * 2^(x0 + 1))) :
  m ≥ 1/2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1223_122361


namespace NUMINAMATH_GPT_total_ducks_and_ducklings_l1223_122313

theorem total_ducks_and_ducklings :
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  (ducks1 + ducks2 + ducks3) + (ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3) = 99 :=
by
  sorry

end NUMINAMATH_GPT_total_ducks_and_ducklings_l1223_122313


namespace NUMINAMATH_GPT_pow_1999_mod_26_l1223_122349

theorem pow_1999_mod_26 (n : ℕ) (h1 : 17^1 % 26 = 17)
  (h2 : 17^2 % 26 = 17) (h3 : 17^3 % 26 = 17) : 17^1999 % 26 = 17 := by
  sorry

end NUMINAMATH_GPT_pow_1999_mod_26_l1223_122349


namespace NUMINAMATH_GPT_normal_mean_is_zero_if_symmetric_l1223_122304

-- Definition: A normal distribution with mean μ and standard deviation σ.
structure NormalDist where
  μ : ℝ
  σ : ℝ

-- Condition: The normal curve is symmetric about the y-axis.
def symmetric_about_y_axis (nd : NormalDist) : Prop :=
  nd.μ = 0

-- Theorem: If the normal curve is symmetric about the y-axis, then the mean μ of the corresponding normal distribution is 0.
theorem normal_mean_is_zero_if_symmetric (nd : NormalDist) (h : symmetric_about_y_axis nd) : nd.μ = 0 := 
by sorry

end NUMINAMATH_GPT_normal_mean_is_zero_if_symmetric_l1223_122304


namespace NUMINAMATH_GPT_part_a_part_b_l1223_122363

-- Part a: Prove for specific numbers 2015 and 2017
theorem part_a : ∃ (x y : ℕ), (2015^2 + 2017^2) / 2 = x^2 + y^2 := sorry

-- Part b: Prove for any two different odd natural numbers
theorem part_b (a b : ℕ) (h1 : a ≠ b) (h2 : a % 2 = 1) (h3 : b % 2 = 1) :
  ∃ (x y : ℕ), (a^2 + b^2) / 2 = x^2 + y^2 := sorry

end NUMINAMATH_GPT_part_a_part_b_l1223_122363


namespace NUMINAMATH_GPT_perimeter_paper_count_l1223_122382

theorem perimeter_paper_count (n : Nat) (h : n = 10) : 
  let top_side := n
  let right_side := n - 1
  let bottom_side := n - 1
  let left_side := n - 2
  top_side + right_side + bottom_side + left_side = 36 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_paper_count_l1223_122382


namespace NUMINAMATH_GPT_find_point_symmetric_about_y_axis_l1223_122398

def point := ℤ × ℤ

def symmetric_about_y_axis (A B : point) : Prop :=
  B.1 = -A.1 ∧ B.2 = A.2

theorem find_point_symmetric_about_y_axis (A B : point) 
  (hA : A = (-5, 2)) 
  (hSym : symmetric_about_y_axis A B) : 
  B = (5, 2) := 
by
  -- We declare the proof but omit the steps for this exercise.
  sorry

end NUMINAMATH_GPT_find_point_symmetric_about_y_axis_l1223_122398


namespace NUMINAMATH_GPT_upward_shift_of_parabola_l1223_122360

variable (k : ℝ) -- Define k as a real number representing the vertical shift

def original_function (x : ℝ) : ℝ := -x^2 -- Define the original function

def shifted_function (x : ℝ) : ℝ := original_function x + 2 -- Define the shifted function by 2 units upwards

theorem upward_shift_of_parabola (x : ℝ) : shifted_function x = -x^2 + k :=
by
  sorry

end NUMINAMATH_GPT_upward_shift_of_parabola_l1223_122360


namespace NUMINAMATH_GPT_find_b_l1223_122378

theorem find_b (b p : ℝ) (h_factor : ∃ k : ℝ, 3 * (x^3 : ℝ) + b * x + 9 = (x^2 + p * x + 3) * (k * x + k)) :
  b = -6 :=
by
  obtain ⟨k, h_eq⟩ := h_factor
  sorry

end NUMINAMATH_GPT_find_b_l1223_122378


namespace NUMINAMATH_GPT_gdp_scientific_notation_l1223_122321

noncomputable def gdp_nanning_2007 : ℝ := 1060 * 10^8

theorem gdp_scientific_notation :
  gdp_nanning_2007 = 1.06 * 10^11 :=
by sorry

end NUMINAMATH_GPT_gdp_scientific_notation_l1223_122321


namespace NUMINAMATH_GPT_line_circle_relationship_l1223_122348

theorem line_circle_relationship (m : ℝ) :
  (∃ x y : ℝ, (mx + y - m - 1 = 0) ∧ (x^2 + y^2 = 2)) ∨ 
  (∃ x : ℝ, (x - 1)^2 + (m*(x - 1) + (1 - 1))^2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_line_circle_relationship_l1223_122348


namespace NUMINAMATH_GPT_unique_primes_solution_l1223_122393

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_primes_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) :
  p^3 - q^5 = (p + q)^2 ↔ (p = 7 ∧ q = 3) :=
by
  sorry

end NUMINAMATH_GPT_unique_primes_solution_l1223_122393


namespace NUMINAMATH_GPT_dino_dolls_count_l1223_122385

theorem dino_dolls_count (T : ℝ) (H : 0.7 * T = 140) : T = 200 :=
sorry

end NUMINAMATH_GPT_dino_dolls_count_l1223_122385


namespace NUMINAMATH_GPT_watch_correction_needed_l1223_122373

def watch_loses_rate : ℚ := 15 / 4  -- rate of loss per day in minutes
def initial_set_time : ℕ := 15  -- March 15th at 10 A.M.
def report_time : ℕ := 24  -- March 24th at 4 P.M.
def correction (loss_rate per_day min_hrs : ℚ) (days_hrs : ℚ) : ℚ :=
  (days_hrs * (loss_rate / (per_day * min_hrs)))

theorem watch_correction_needed :
  correction watch_loses_rate 24 60 (222) = 34.6875 := 
sorry

end NUMINAMATH_GPT_watch_correction_needed_l1223_122373


namespace NUMINAMATH_GPT_probability_of_two_non_defective_pens_l1223_122367

-- Definitions for conditions from the problem
def total_pens : ℕ := 16
def defective_pens : ℕ := 3
def selected_pens : ℕ := 2
def non_defective_pens : ℕ := total_pens - defective_pens

-- Function to calculate probability of drawing non-defective pens
noncomputable def probability_no_defective (total : ℕ) (defective : ℕ) (selected : ℕ) : ℚ :=
  (non_defective_pens / total_pens) * ((non_defective_pens - 1) / (total_pens - 1))

-- Theorem stating the correct answer
theorem probability_of_two_non_defective_pens : 
  probability_no_defective total_pens defective_pens selected_pens = 13 / 20 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_two_non_defective_pens_l1223_122367


namespace NUMINAMATH_GPT_gain_percent_is_87_point_5_l1223_122320

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  ((S - C) / C) * 100

theorem gain_percent_is_87_point_5 {C S : ℝ} (h : 75 * C = 40 * S) :
  gain_percent C S = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_is_87_point_5_l1223_122320


namespace NUMINAMATH_GPT_intersection_point_k_value_l1223_122347

theorem intersection_point_k_value :
  (∃ (k : ℝ), (∀ (x y : ℝ),
    ((y = 2 * x + 3 ∧ y = k * x + 2) → (x = 1 ∧ y = 5))) → k = 3) :=
sorry

end NUMINAMATH_GPT_intersection_point_k_value_l1223_122347


namespace NUMINAMATH_GPT_average_headcount_is_correct_l1223_122359

/-- The student headcount data for the specified semesters -/
def student_headcount : List ℕ := [11700, 10900, 11500, 10500, 11600, 10700, 11300]

noncomputable def average_headcount : ℕ :=
  (student_headcount.sum) / student_headcount.length

theorem average_headcount_is_correct : average_headcount = 11029 := by
  sorry

end NUMINAMATH_GPT_average_headcount_is_correct_l1223_122359


namespace NUMINAMATH_GPT_length_of_GH_l1223_122362

def EF := 180
def IJ := 120

theorem length_of_GH (EF_parallel_GH : true) (GH_parallel_IJ : true) : GH = 72 := 
sorry

end NUMINAMATH_GPT_length_of_GH_l1223_122362


namespace NUMINAMATH_GPT_jake_balloons_bought_l1223_122380

theorem jake_balloons_bought (B : ℕ) (h : 6 = (2 + B) + 1) : B = 3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_jake_balloons_bought_l1223_122380


namespace NUMINAMATH_GPT_sum_of_roots_l1223_122391

theorem sum_of_roots (x : ℝ) : (x - 4)^2 = 16 → x = 8 ∨ x = 0 := by
  intro h
  have h1 : x - 4 = 4 ∨ x - 4 = -4 := by
    sorry
  cases h1
  case inl h2 =>
    rw [h2] at h
    exact Or.inl (by linarith)
  case inr h2 =>
    rw [h2] at h
    exact Or.inr (by linarith)

end NUMINAMATH_GPT_sum_of_roots_l1223_122391


namespace NUMINAMATH_GPT_inequalities_proof_l1223_122344

theorem inequalities_proof (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  (a < (c / 2)) ∧ (b < a + c / 2) ∧ ¬(b < c / 2) :=
by
  constructor
  { sorry }
  { constructor
    { sorry }
    { sorry } }

end NUMINAMATH_GPT_inequalities_proof_l1223_122344


namespace NUMINAMATH_GPT_product_of_repeating_decimal_and_22_l1223_122354

noncomputable def repeating_decimal_to_fraction : ℚ :=
  0.45 + 0.0045 * (10 ^ (-2 : ℤ))

theorem product_of_repeating_decimal_and_22 : (repeating_decimal_to_fraction * 22 = 10) :=
by
  sorry

end NUMINAMATH_GPT_product_of_repeating_decimal_and_22_l1223_122354


namespace NUMINAMATH_GPT_product_of_two_numbers_l1223_122335

theorem product_of_two_numbers (a b : ℕ) (h1 : Nat.gcd a b = 8) (h2 : Nat.lcm a b = 48) : a * b = 384 :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1223_122335


namespace NUMINAMATH_GPT_total_sums_attempted_l1223_122332

-- Define the necessary conditions
def num_sums_right : ℕ := 8
def num_sums_wrong : ℕ := 2 * num_sums_right

-- Define the theorem to prove
theorem total_sums_attempted : num_sums_right + num_sums_wrong = 24 := by
  sorry

end NUMINAMATH_GPT_total_sums_attempted_l1223_122332


namespace NUMINAMATH_GPT_equation1_equation2_equation3_equation4_l1223_122327

theorem equation1 (x : ℝ) : (x - 1) ^ 2 - 5 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
  sorry

theorem equation2 (x : ℝ) : x * (x + 4) = -3 * (x + 4) ↔ x = -4 ∨ x = -3 := by
  sorry

theorem equation3 (y : ℝ) : 2 * y ^ 2 - 5 * y + 2 = 0 ↔ y = 1 / 2 ∨ y = 2 := by
  sorry

theorem equation4 (m : ℝ) : 2 * m ^ 2 - 7 * m - 3 = 0 ↔ m = (7 + Real.sqrt 73) / 4 ∨ m = (7 - Real.sqrt 73) / 4 := by
  sorry

end NUMINAMATH_GPT_equation1_equation2_equation3_equation4_l1223_122327


namespace NUMINAMATH_GPT_andrew_vacation_days_l1223_122340

theorem andrew_vacation_days (days_worked last_year vacation_per_10 worked_days in_march in_september : ℕ)
  (h1 : vacation_per_10 = 10)
  (h2 : days_worked_last_year = 300)
  (h3 : worked_days = days_worked_last_year / vacation_per_10)
  (h4 : in_march = 5)
  (h5 : in_september = 2 * in_march)
  (h6 : days_taken = in_march + in_september)
  (h7 : vacation_days_remaining = worked_days - days_taken) :
  vacation_days_remaining = 15 :=
by
  sorry

end NUMINAMATH_GPT_andrew_vacation_days_l1223_122340


namespace NUMINAMATH_GPT_scenario_1_scenario_2_scenario_3_scenario_4_l1223_122303

-- Definitions based on conditions
def prob_A_hit : ℚ := 2 / 3
def prob_B_hit : ℚ := 3 / 4

-- Scenario 1: Prove that the probability of A shooting 3 times and missing at least once is 19/27
theorem scenario_1 : 
  (1 - (prob_A_hit ^ 3)) = 19 / 27 :=
by sorry

-- Scenario 2: Prove that the probability of A hitting the target exactly 2 times and B hitting the target exactly 1 time after each shooting twice is 1/6
theorem scenario_2 : 
  (2 * ((prob_A_hit ^ 2) * (1 - prob_A_hit)) * (2 * (prob_B_hit * (1 - prob_B_hit)))) = 1 / 6 :=
by sorry

-- Scenario 3: Prove that the probability of A missing the target and B hitting the target 2 times after each shooting twice is 1/16
theorem scenario_3 :
  ((1 - prob_A_hit) ^ 2) * (prob_B_hit ^ 2) = 1 / 16 :=
by sorry

-- Scenario 4: Prove that the probability that both A and B hit the target once after each shooting twice is 1/6
theorem scenario_4 : 
  (2 * (prob_A_hit * (1 - prob_A_hit)) * 2 * (prob_B_hit * (1 - prob_B_hit))) = 1 / 6 :=
by sorry

end NUMINAMATH_GPT_scenario_1_scenario_2_scenario_3_scenario_4_l1223_122303


namespace NUMINAMATH_GPT_final_value_of_A_l1223_122306

-- Define the initial value of A
def initial_value (A : ℤ) : Prop := A = 15

-- Define the reassignment condition
def reassignment_cond (A : ℤ) : Prop := A = -A + 5

-- The theorem stating that given the initial value and reassignment condition, the final value of A is -10
theorem final_value_of_A (A : ℤ) (h1 : initial_value A) (h2 : reassignment_cond A) : A = -10 := by
  sorry

end NUMINAMATH_GPT_final_value_of_A_l1223_122306


namespace NUMINAMATH_GPT_marble_count_l1223_122331

variable (r b g : ℝ)

-- Conditions
def condition1 : b = r / 1.3 := sorry
def condition2 : g = 1.5 * r := sorry

-- Theorem statement
theorem marble_count (h1 : b = r / 1.3) (h2 : g = 1.5 * r) :
  r + b + g = 3.27 * r :=
by sorry

end NUMINAMATH_GPT_marble_count_l1223_122331


namespace NUMINAMATH_GPT_max_min_values_l1223_122366

theorem max_min_values (x y : ℝ) 
  (h : (x - 3)^2 + 4 * (y - 1)^2 = 4) :
  ∃ (t u : ℝ), (∀ (z : ℝ), (x-3)^2 + 4*(y-1)^2 = 4 → t ≤ (x+y-3)/(x-y+1) ∧ (x+y-3)/(x-y+1) ≤ u) ∧ t = -1 ∧ u = 1 := 
by
  sorry

end NUMINAMATH_GPT_max_min_values_l1223_122366


namespace NUMINAMATH_GPT_option_d_true_l1223_122305

theorem option_d_true (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpr_qr : p * r < q * r) : 1 > q / p :=
sorry

end NUMINAMATH_GPT_option_d_true_l1223_122305


namespace NUMINAMATH_GPT_hiker_walking_speed_l1223_122325

theorem hiker_walking_speed (v : ℝ) :
  (∃ (hiker_shares_cyclist_distance : 20 / 60 * v = 25 * (5 / 60)), v = 6.25) :=
by
  sorry

end NUMINAMATH_GPT_hiker_walking_speed_l1223_122325


namespace NUMINAMATH_GPT_parallel_vectors_k_eq_neg1_l1223_122318

theorem parallel_vectors_k_eq_neg1
  (k : ℤ)
  (a : ℤ × ℤ := (2 * k + 2, 4))
  (b : ℤ × ℤ := (k + 1, 8))
  (h : a.1 * b.2 = a.2 * b.1) :
  k = -1 :=
by
sorry

end NUMINAMATH_GPT_parallel_vectors_k_eq_neg1_l1223_122318


namespace NUMINAMATH_GPT_complete_the_square_l1223_122396

theorem complete_the_square (x : ℝ) : x^2 + 6 * x + 3 = 0 ↔ (x + 3)^2 = 6 := 
by
  sorry

end NUMINAMATH_GPT_complete_the_square_l1223_122396


namespace NUMINAMATH_GPT_common_chord_of_circles_l1223_122337

theorem common_chord_of_circles
  (x y : ℝ)
  (h1 : x^2 + y^2 + 2 * x = 0)
  (h2 : x^2 + y^2 - 4 * y = 0)
  : x + 2 * y = 0 := 
by
  -- Lean will check the logical consistency of the statement.
  sorry

end NUMINAMATH_GPT_common_chord_of_circles_l1223_122337


namespace NUMINAMATH_GPT_unique_positive_solution_l1223_122345

theorem unique_positive_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) : x = 15 := by
  sorry

end NUMINAMATH_GPT_unique_positive_solution_l1223_122345
