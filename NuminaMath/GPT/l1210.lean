import Mathlib

namespace NUMINAMATH_GPT_polynomial_necessary_but_not_sufficient_l1210_121040

-- Definitions
def polynomial_condition (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

def specific_value : ℝ := 1

-- Theorem statement
theorem polynomial_necessary_but_not_sufficient :
  (polynomial_condition specific_value ∧ ¬ ∀ x, polynomial_condition x -> x = specific_value) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_necessary_but_not_sufficient_l1210_121040


namespace NUMINAMATH_GPT_smaller_number_is_five_l1210_121051

theorem smaller_number_is_five (x y : ℕ) (h1 : x + y = 18) (h2 : x - y = 8) : y = 5 :=
sorry

end NUMINAMATH_GPT_smaller_number_is_five_l1210_121051


namespace NUMINAMATH_GPT_stratified_sampling_l1210_121022

theorem stratified_sampling (n : ℕ) (h_ratio : 2 + 3 + 5 = 10) (h_sample : (5 : ℚ) / 10 = 150 / n) : n = 300 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l1210_121022


namespace NUMINAMATH_GPT_variance_transformed_list_l1210_121036

noncomputable def stddev (xs : List ℝ) : ℝ := sorry
noncomputable def variance (xs : List ℝ) : ℝ := sorry

theorem variance_transformed_list :
  ∀ (a_1 a_2 a_3 a_4 a_5 : ℝ),
  stddev [a_1, a_2, a_3, a_4, a_5] = 2 →
  variance [3 * a_1 - 2, 3 * a_2 - 2, 3 * a_3 - 2, 3 * a_4 - 2, 3 * a_5 - 2] = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_variance_transformed_list_l1210_121036


namespace NUMINAMATH_GPT_fraction_pow_zero_l1210_121003

theorem fraction_pow_zero :
  let a := 7632148
  let b := -172836429
  (a / b ≠ 0) → (a / b)^0 = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_pow_zero_l1210_121003


namespace NUMINAMATH_GPT_game_winner_l1210_121066

theorem game_winner (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  (mn % 2 = 1 → first_player_wins) ∧ (mn % 2 = 0 → second_player_wins) :=
sorry

end NUMINAMATH_GPT_game_winner_l1210_121066


namespace NUMINAMATH_GPT_rotate_90deg_l1210_121028

def Shape := Type

structure Figure :=
(triangle : Shape)
(circle : Shape)
(square : Shape)
(pentagon : Shape)

def rotated_position (fig : Figure) : Figure :=
{ triangle := fig.circle,
  circle := fig.square,
  square := fig.pentagon,
  pentagon := fig.triangle }

theorem rotate_90deg (fig : Figure) :
  rotated_position fig = { triangle := fig.circle,
                           circle := fig.square,
                           square := fig.pentagon,
                           pentagon := fig.triangle } :=
by {
  sorry
}

end NUMINAMATH_GPT_rotate_90deg_l1210_121028


namespace NUMINAMATH_GPT_center_of_circle_l1210_121011

theorem center_of_circle (A B : ℝ × ℝ) (hA : A = (2, -3)) (hB : B = (10, 5)) :
    (A.1 + B.1) / 2 = 6 ∧ (A.2 + B.2) / 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_center_of_circle_l1210_121011


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l1210_121074

open Real

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 20) :
  (1 / a + 1 / b) ≥ 1 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l1210_121074


namespace NUMINAMATH_GPT_correct_choice_C_l1210_121039

theorem correct_choice_C (x : ℝ) : x^2 ≥ x - 1 := 
sorry

end NUMINAMATH_GPT_correct_choice_C_l1210_121039


namespace NUMINAMATH_GPT_train_speed_l1210_121047

theorem train_speed (lt_train : ℝ) (lt_bridge : ℝ) (time_cross : ℝ) (total_speed_kmph : ℝ) :
  lt_train = 150 ∧ lt_bridge = 225 ∧ time_cross = 30 ∧ total_speed_kmph = (375 / 30) * 3.6 → 
  total_speed_kmph = 45 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_l1210_121047


namespace NUMINAMATH_GPT_cost_of_one_dozen_pens_l1210_121049

theorem cost_of_one_dozen_pens
  (x : ℝ)
  (hx : 20 * x = 150) :
  12 * 5 * (150 / 20) = 450 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_dozen_pens_l1210_121049


namespace NUMINAMATH_GPT_total_project_hours_l1210_121085

def research_hours : ℕ := 10
def proposal_hours : ℕ := 2
def report_hours_left : ℕ := 8

theorem total_project_hours :
  research_hours + proposal_hours + report_hours_left = 20 := 
  sorry

end NUMINAMATH_GPT_total_project_hours_l1210_121085


namespace NUMINAMATH_GPT_find_number_l1210_121001

theorem find_number (x : ℕ) (h : 3 * x = 33) : x = 11 :=
sorry

end NUMINAMATH_GPT_find_number_l1210_121001


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l1210_121094

open Nat

noncomputable def S (n : ℕ) : ℝ := n^2
noncomputable def T (n : ℕ) : ℝ := n * (2 * n + 3)

theorem arithmetic_sequence_ratio 
  (h : ∀ n : ℕ, (2 * n + 3) * S n = n * T n) : 
  (S 5 - S 4) / (T 6 - T 5) = 9 / 25 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l1210_121094


namespace NUMINAMATH_GPT_total_voters_l1210_121077

-- Definitions
def number_of_voters_first_hour (x : ℕ) := x
def percentage_october_22 (x : ℕ) := 35 * x / 100
def percentage_october_29 (x : ℕ) := 65 * x / 100
def additional_voters_october_22 := 80
def final_percentage_october_29 (total_votes : ℕ) := 45 * total_votes / 100

-- Statement
theorem total_voters (x : ℕ) (h1 : percentage_october_22 x + additional_voters_october_22 = 35 * (x + additional_voters_october_22) / 100)
                      (h2 : percentage_october_29 x = 65 * x / 100)
                      (h3 : final_percentage_october_29 (x + additional_voters_october_22) = 45 * (x + additional_voters_october_22) / 100):
  x + additional_voters_october_22 = 260 := 
sorry

end NUMINAMATH_GPT_total_voters_l1210_121077


namespace NUMINAMATH_GPT_jake_more_peaches_than_jill_l1210_121002

theorem jake_more_peaches_than_jill :
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  jake_peaches - jill_peaches = 3 :=
by
  let steven_peaches := 14
  let jake_peaches := steven_peaches - 6
  let jill_peaches := 5
  sorry

end NUMINAMATH_GPT_jake_more_peaches_than_jill_l1210_121002


namespace NUMINAMATH_GPT_arithmetic_sequence_diff_l1210_121016

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition for the arithmetic sequence
def condition (a : ℕ → ℝ) : Prop := 
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

-- Definition of the common difference
def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The proof problem statement in Lean 4
theorem arithmetic_sequence_diff (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a → condition a → common_difference a d → a 7 - a 8 = -d :=
by
  intros _ _ _
  -- Proof will be conducted here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_diff_l1210_121016


namespace NUMINAMATH_GPT_max_perimeter_of_rectangle_with_area_36_l1210_121041

theorem max_perimeter_of_rectangle_with_area_36 :
  ∃ l w : ℕ, l * w = 36 ∧ (∀ l' w' : ℕ, l' * w' = 36 → 2 * (l + w) ≥ 2 * (l' + w')) ∧ 2 * (l + w) = 74 := 
sorry

end NUMINAMATH_GPT_max_perimeter_of_rectangle_with_area_36_l1210_121041


namespace NUMINAMATH_GPT_vertex_set_is_parabola_l1210_121065

variables (a c k : ℝ) (ha : a > 0) (hc : c > 0) (hk : k ≠ 0)

theorem vertex_set_is_parabola :
  ∃ (f : ℝ → ℝ), (∀ t : ℝ, f t = (-k^2 / (4 * a)) * t^2 + c) :=
sorry

end NUMINAMATH_GPT_vertex_set_is_parabola_l1210_121065


namespace NUMINAMATH_GPT_ratio_of_area_l1210_121007

-- Define the sides of the triangles
def sides_GHI := (6, 8, 10)
def sides_JKL := (9, 12, 15)

-- Define the function to calculate the area of a triangle given its sides as a Pythagorean triple
def area_of_right_triangle (a b : Nat) : Nat :=
  (a * b) / 2

-- Calculate the areas
def area_GHI := area_of_right_triangle 6 8
def area_JKL := area_of_right_triangle 9 12

-- Calculate the ratio of the areas
def ratio_of_areas := area_GHI * 2 / area_JKL * 3

-- Statement to prove
theorem ratio_of_area (sides_GHI sides_JKL : (Nat × Nat × Nat)) :
  (ratio_of_areas = (4 : ℚ) / 9) :=
sorry

end NUMINAMATH_GPT_ratio_of_area_l1210_121007


namespace NUMINAMATH_GPT_slope_of_tangent_line_at_1_1_l1210_121026

theorem slope_of_tangent_line_at_1_1 : 
  ∃ f' : ℝ → ℝ, (∀ x, f' x = 3 * x^2) ∧ (f' 1 = 3) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_line_at_1_1_l1210_121026


namespace NUMINAMATH_GPT_remainder_div_13_l1210_121024

theorem remainder_div_13 {k : ℤ} (N : ℤ) (h : N = 39 * k + 20) : N % 13 = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_13_l1210_121024


namespace NUMINAMATH_GPT_problem_solution_l1210_121029

theorem problem_solution :
  (∀ (p q : ℚ), 
    (∀ (x : ℚ), (x + 3 * p) * (x^2 - x + (1 / 3) * q) = x^3 + (3 * p - 1) * x^2 + ((1 / 3) * q - 3 * p) * x + p * q) →
    (3 * p - 1 = 0) →
    ((1 / 3) * q - 3 * p = 0) →
    p = 1 / 3 ∧ q = 3)
  ∧ ((1 / 3) ^ 2020 * 3 ^ 2021 = 3) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1210_121029


namespace NUMINAMATH_GPT_max_length_cos_theta_l1210_121034

def domain (x y : ℝ) : Prop := (x^2 + (y - 1)^2 ≤ 1 ∧ x ≥ (Real.sqrt 2 / 3))

theorem max_length_cos_theta :
  (∃ x y : ℝ, domain x y ∧ ∀ θ : ℝ, (0 < θ ∧ θ < (Real.pi / 2)) → θ = Real.arctan (Real.sqrt 2) → 
  (Real.cos θ = Real.sqrt 3 / 3)) := sorry

end NUMINAMATH_GPT_max_length_cos_theta_l1210_121034


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1210_121017

theorem solution_set_of_inequality (x : ℝ) : x^2 > x ↔ x < 0 ∨ 1 < x := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1210_121017


namespace NUMINAMATH_GPT_inequality_solution_l1210_121054

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 4 / x + 19 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1210_121054


namespace NUMINAMATH_GPT_inequality_proof_l1210_121071

theorem inequality_proof (a b : ℝ) (h1 : a ≠ b) (h2 : a < 0) : a < 2 * b - b^2 / a := 
by
  -- mathematical proof goes here
  sorry

end NUMINAMATH_GPT_inequality_proof_l1210_121071


namespace NUMINAMATH_GPT_opposite_of_pi_eq_neg_pi_l1210_121008

theorem opposite_of_pi_eq_neg_pi (π : Real) (h : π = Real.pi) : -π = -Real.pi :=
by sorry

end NUMINAMATH_GPT_opposite_of_pi_eq_neg_pi_l1210_121008


namespace NUMINAMATH_GPT_determine_treasures_possible_l1210_121089

structure Subject :=
  (is_knight : Prop)
  (is_liar : Prop)
  (is_normal : Prop)

def island_has_treasures : Prop := sorry

def can_determine_treasures (A B C : Subject) (at_most_one_normal : Bool) : Prop :=
  if at_most_one_normal then
    ∃ (question : (Subject → Prop)),
      (∀ response1, ∃ (question2 : (Subject → Prop)),
        (∀ response2, island_has_treasures ↔ (response1 ∧ response2)))
  else
    false

theorem determine_treasures_possible (A B C : Subject) (at_most_one_normal : Bool) :
  at_most_one_normal = true → can_determine_treasures A B C at_most_one_normal :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_treasures_possible_l1210_121089


namespace NUMINAMATH_GPT_exists_arithmetic_progression_with_sum_zero_l1210_121033

theorem exists_arithmetic_progression_with_sum_zero : 
  ∃ (a d : Int) (n : Int), n > 0 ∧ (n * (2 * a + (n - 1) * d)) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_exists_arithmetic_progression_with_sum_zero_l1210_121033


namespace NUMINAMATH_GPT_minimize_time_theta_l1210_121075

theorem minimize_time_theta (α θ : ℝ) (h1 : 0 < α) (h2 : α < 90) (h3 : θ = α / 2) : 
  θ = α / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_time_theta_l1210_121075


namespace NUMINAMATH_GPT_mustard_found_at_second_table_l1210_121079

variables (total_mustard first_table third_table second_table : ℝ)

def mustard_found (total_mustard first_table third_table : ℝ) := total_mustard - (first_table + third_table)

theorem mustard_found_at_second_table
    (h_total : total_mustard = 0.88)
    (h_first : first_table = 0.25)
    (h_third : third_table = 0.38) :
    mustard_found total_mustard first_table third_table = 0.25 :=
by
    rw [mustard_found, h_total, h_first, h_third]
    simp
    sorry

end NUMINAMATH_GPT_mustard_found_at_second_table_l1210_121079


namespace NUMINAMATH_GPT_education_expenses_l1210_121097

theorem education_expenses (rent milk groceries petrol miscellaneous savings total_salary education : ℝ) 
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_petrol : petrol = 2000)
  (h_miscellaneous : miscellaneous = 6100)
  (h_savings : savings = 2400)
  (h_saving_percentage : savings = 0.10 * total_salary)
  (h_total_salary : total_salary = savings / 0.10)
  (h_total_expenses : total_salary - savings = rent + milk + groceries + petrol + miscellaneous + education) :
  education = 2500 :=
by
  sorry

end NUMINAMATH_GPT_education_expenses_l1210_121097


namespace NUMINAMATH_GPT_jogging_problem_l1210_121030

theorem jogging_problem (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : ¬ ∃ p : ℕ, Prime p ∧ p^2 ∣ z) : 
  (x - y * Real.sqrt z) = 60 - 30 * Real.sqrt 2 → x + y + z = 92 :=
by
  intro h5
  have h6 : (60 - (60 - 30 * Real.sqrt 2))^2 = 1800 :=
    by sorry
  sorry

end NUMINAMATH_GPT_jogging_problem_l1210_121030


namespace NUMINAMATH_GPT_min_value_3x_plus_4y_l1210_121086

variable (x y : ℝ)

theorem min_value_3x_plus_4y (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_value_3x_plus_4y_l1210_121086


namespace NUMINAMATH_GPT_partial_fraction_decomposition_l1210_121020

theorem partial_fraction_decomposition (x : ℝ) :
  (5 * x - 3) / (x^2 - 5 * x - 14) = (32 / 9) / (x - 7) + (13 / 9) / (x + 2) := by
  sorry

end NUMINAMATH_GPT_partial_fraction_decomposition_l1210_121020


namespace NUMINAMATH_GPT_intersection_points_are_integers_l1210_121025

theorem intersection_points_are_integers :
  ∀ (a b : Fin 2021 → ℕ), Function.Injective a → Function.Injective b →
  ∀ i j, i ≠ j → 
  ∃ x : ℤ, (∃ y : ℚ, y = (a i : ℚ) / (x + (b i : ℚ))) ∧ 
           (∃ y : ℚ, y = (a j : ℚ) / (x + (b j : ℚ))) := 
sorry

end NUMINAMATH_GPT_intersection_points_are_integers_l1210_121025


namespace NUMINAMATH_GPT_equation_has_exactly_one_solution_l1210_121091

theorem equation_has_exactly_one_solution (m : ℝ) : 
  (m ∈ { -1 } ∪ Set.Ioo (-1/2 : ℝ) (1/0) ) ↔ ∃ (x : ℝ), 2 * Real.sqrt (1 - m * (x + 2)) = x + 4 :=
sorry

end NUMINAMATH_GPT_equation_has_exactly_one_solution_l1210_121091


namespace NUMINAMATH_GPT_share_y_is_18_l1210_121035

-- Definitions from conditions
def total_amount := 70
def ratio_x := 100
def ratio_y := 45
def ratio_z := 30
def total_ratio := ratio_x + ratio_y + ratio_z
def part_value := total_amount / total_ratio
def share_y := ratio_y * part_value

-- Statement to be proved
theorem share_y_is_18 : share_y = 18 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_share_y_is_18_l1210_121035


namespace NUMINAMATH_GPT_marcy_fewer_tickets_l1210_121099

theorem marcy_fewer_tickets (A M : ℕ) (h1 : A = 26) (h2 : M = 5 * A) (h3 : A + M = 150) : M - A = 104 :=
by
  sorry

end NUMINAMATH_GPT_marcy_fewer_tickets_l1210_121099


namespace NUMINAMATH_GPT_simplify_expression_l1210_121057

theorem simplify_expression :
  (4 + 5) * (4 ^ 2 + 5 ^ 2) * (4 ^ 4 + 5 ^ 4) * (4 ^ 8 + 5 ^ 8) * (4 ^ 16 + 5 ^ 16) * (4 ^ 32 + 5 ^ 32) * (4 ^ 64 + 5 ^ 64) = 5 ^ 128 - 4 ^ 128 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1210_121057


namespace NUMINAMATH_GPT_work_efficiency_ratio_l1210_121052
noncomputable section

variable (A_eff B_eff : ℚ)

-- Conditions
def efficient_together (A_eff B_eff : ℚ) : Prop := A_eff + B_eff = 1 / 12
def efficient_alone (A_eff : ℚ) : Prop := A_eff = 1 / 16

-- Theorem to prove
theorem work_efficiency_ratio (A_eff B_eff : ℚ) (h1 : efficient_together A_eff B_eff) (h2 : efficient_alone A_eff) : A_eff / B_eff = 3 := by
  sorry

end NUMINAMATH_GPT_work_efficiency_ratio_l1210_121052


namespace NUMINAMATH_GPT_problem_statement_l1210_121098

variable (a b c : ℝ)

theorem problem_statement (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) ≥ 6 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1210_121098


namespace NUMINAMATH_GPT_amoeba_count_after_one_week_l1210_121069

/-- An amoeba is placed in a puddle and splits into three amoebas on the same day. Each subsequent
    day, every amoeba in the puddle splits into three new amoebas. -/
theorem amoeba_count_after_one_week : 
  let initial_amoebas := 1
  let daily_split := 3
  let days := 7
  (initial_amoebas * (daily_split ^ days)) = 2187 :=
by
  sorry

end NUMINAMATH_GPT_amoeba_count_after_one_week_l1210_121069


namespace NUMINAMATH_GPT_man_receives_total_amount_l1210_121042
noncomputable def total_amount_received : ℝ := 
  let itemA_price := 1300
  let itemB_price := 750
  let itemC_price := 1800
  
  let itemA_loss := 0.20 * itemA_price
  let itemB_loss := 0.15 * itemB_price
  let itemC_loss := 0.10 * itemC_price

  let itemA_selling_price := itemA_price - itemA_loss
  let itemB_selling_price := itemB_price - itemB_loss
  let itemC_selling_price := itemC_price - itemC_loss

  let vat_rate := 0.12
  let itemA_vat := vat_rate * itemA_selling_price
  let itemB_vat := vat_rate * itemB_selling_price
  let itemC_vat := vat_rate * itemC_selling_price

  let final_itemA := itemA_selling_price + itemA_vat
  let final_itemB := itemB_selling_price + itemB_vat
  let final_itemC := itemC_selling_price + itemC_vat

  final_itemA + final_itemB + final_itemC

theorem man_receives_total_amount :
  total_amount_received = 3693.2 := by
  sorry

end NUMINAMATH_GPT_man_receives_total_amount_l1210_121042


namespace NUMINAMATH_GPT_number_of_small_triangles_l1210_121053

noncomputable def area_of_large_triangle (hypotenuse_large : ℝ) : ℝ :=
  let leg := hypotenuse_large / Real.sqrt 2
  (1 / 2) * (leg * leg)

noncomputable def area_of_small_triangle (hypotenuse_small : ℝ) : ℝ :=
  let leg := hypotenuse_small / Real.sqrt 2
  (1 / 2) * (leg * leg)

theorem number_of_small_triangles (hypotenuse_large : ℝ) (hypotenuse_small : ℝ) :
  hypotenuse_large = 14 → hypotenuse_small = 2 →
  let number_of_triangles := (area_of_large_triangle hypotenuse_large) / (area_of_small_triangle hypotenuse_small)
  number_of_triangles = 49 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_number_of_small_triangles_l1210_121053


namespace NUMINAMATH_GPT_parabola_origin_l1210_121060

theorem parabola_origin (x y c : ℝ) (h : y = x^2 - 2 * x + c - 4) (h0 : (0, 0) = (x, y)) : c = 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_origin_l1210_121060


namespace NUMINAMATH_GPT_value_of_A_l1210_121072

theorem value_of_A 
  (H M A T E: ℤ)
  (H_value: H = 10)
  (MATH_value: M + A + T + H = 35)
  (TEAM_value: T + E + A + M = 42)
  (MEET_value: M + 2*E + T = 38) : 
  A = 21 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_A_l1210_121072


namespace NUMINAMATH_GPT_johns_profit_l1210_121027

noncomputable def earnings : ℕ := 30000
noncomputable def purchase_price : ℕ := 18000
noncomputable def trade_in_value : ℕ := 6000

noncomputable def depreciation : ℕ := purchase_price - trade_in_value
noncomputable def profit : ℕ := earnings - depreciation

theorem johns_profit : profit = 18000 := by
  sorry

end NUMINAMATH_GPT_johns_profit_l1210_121027


namespace NUMINAMATH_GPT_correct_sampling_methods_l1210_121005

-- Define conditions for the sampling problems
structure SamplingProblem where
  scenario: String
  samplingMethod: String

-- Define the three scenarios
def firstScenario : SamplingProblem :=
  { scenario := "Draw 5 bottles from 15 bottles of drinks for food hygiene inspection", samplingMethod := "Simple random sampling" }

def secondScenario : SamplingProblem :=
  { scenario := "Sample 20 staff members from 240 staff members in a middle school", samplingMethod := "Stratified sampling" }

def thirdScenario : SamplingProblem :=
  { scenario := "Select 25 audience members from a full science and technology report hall", samplingMethod := "Systematic sampling" }

-- Main theorem combining all conditions and proving the correct answer
theorem correct_sampling_methods :
  (firstScenario.samplingMethod = "Simple random sampling") ∧
  (secondScenario.samplingMethod = "Stratified sampling") ∧
  (thirdScenario.samplingMethod = "Systematic sampling") :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_correct_sampling_methods_l1210_121005


namespace NUMINAMATH_GPT_cost_price_per_meter_l1210_121004

namespace ClothCost

theorem cost_price_per_meter (selling_price_total : ℝ) (meters_sold : ℕ) (loss_per_meter : ℝ) : 
  selling_price_total = 18000 → 
  meters_sold = 300 → 
  loss_per_meter = 5 →
  (selling_price_total / meters_sold) + loss_per_meter = 65 := 
by
  intros hsp hms hloss
  sorry

end ClothCost

end NUMINAMATH_GPT_cost_price_per_meter_l1210_121004


namespace NUMINAMATH_GPT_min_value_of_sum_l1210_121093

theorem min_value_of_sum (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) : 
  ∃ x : ℝ, x = (1 / (a - 1) + 1 / b) ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_sum_l1210_121093


namespace NUMINAMATH_GPT_washer_and_dryer_proof_l1210_121088

noncomputable def washer_and_dryer_problem : Prop :=
  ∃ (price_of_washer price_of_dryer : ℕ),
    price_of_washer + price_of_dryer = 600 ∧
    (∃ (k : ℕ), price_of_washer = k * price_of_dryer) ∧
    price_of_dryer = 150 ∧
    price_of_washer / price_of_dryer = 3

theorem washer_and_dryer_proof : washer_and_dryer_problem :=
sorry

end NUMINAMATH_GPT_washer_and_dryer_proof_l1210_121088


namespace NUMINAMATH_GPT_find_sin_2a_l1210_121096

noncomputable def problem_statement (a : ℝ) : Prop :=
a ∈ Set.Ioo (Real.pi / 2) Real.pi ∧
3 * Real.cos (2 * a) = Real.sqrt 2 * Real.sin ((Real.pi / 4) - a)

theorem find_sin_2a (a : ℝ) (h : problem_statement a) : Real.sin (2 * a) = -8 / 9 :=
sorry

end NUMINAMATH_GPT_find_sin_2a_l1210_121096


namespace NUMINAMATH_GPT_Suma_can_complete_in_6_days_l1210_121038

-- Define the rates for Renu and their combined rate
def Renu_rate := (1 : ℚ) / 6
def Combined_rate := (1 : ℚ) / 3

-- Define Suma's time to complete the work alone
def Suma_days := 6

-- defining the work rate Suma is required to achieve given the known rates and combined rate
def Suma_rate := Combined_rate - Renu_rate

-- Require to prove 
theorem Suma_can_complete_in_6_days : (1 / Suma_rate) = Suma_days :=
by
  -- Using the definitions provided and some basic algebra to prove the theorem 
  sorry

end NUMINAMATH_GPT_Suma_can_complete_in_6_days_l1210_121038


namespace NUMINAMATH_GPT_probability_intersection_l1210_121095

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end NUMINAMATH_GPT_probability_intersection_l1210_121095


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1210_121082

variable (AB AC BC s K r : ℝ)
variable (AB_eq AC_eq BC_eq : AB = AC ∧ AC = 8 ∧ BC = 7)
variable (s_eq : s = (AB + AC + BC) / 2)
variable (K_eq : K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)))
variable (r_eq : r * s = K)

/-- Prove that the radius of the inscribed circle is 23.75 / 11.5 given the conditions of the triangle --/
theorem inscribed_circle_radius :
  AB = 8 → AC = 8 → BC = 7 → 
  s = (AB + AC + BC) / 2 → 
  K = Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) →
  r * s = K →
  r = (23.75 / 11.5) :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1210_121082


namespace NUMINAMATH_GPT_john_total_distance_l1210_121014

theorem john_total_distance :
  let s₁ : ℝ := 45       -- Speed for the first part (mph)
  let t₁ : ℝ := 2        -- Time for the first part (hours)
  let s₂ : ℝ := 50       -- Speed for the second part (mph)
  let t₂ : ℝ := 3        -- Time for the second part (hours)
  let d₁ : ℝ := s₁ * t₁ -- Distance for the first part
  let d₂ : ℝ := s₂ * t₂ -- Distance for the second part
  d₁ + d₂ = 240          -- Total distance
:= by
  sorry

end NUMINAMATH_GPT_john_total_distance_l1210_121014


namespace NUMINAMATH_GPT_green_marble_prob_l1210_121009

-- Problem constants
def total_marbles : ℕ := 84
def prob_white : ℚ := 1 / 4
def prob_red_or_blue : ℚ := 0.4642857142857143

-- Defining the individual variables for the counts
variable (W R B G : ℕ)

-- Conditions
axiom total_marbles_eq : W + R + B + G = total_marbles
axiom prob_white_eq : (W : ℚ) / total_marbles = prob_white
axiom prob_red_or_blue_eq : (R + B : ℚ) / total_marbles = prob_red_or_blue

-- Proving the probability of drawing a green marble
theorem green_marble_prob :
  (G : ℚ) / total_marbles = 2 / 7 :=
by
  sorry  -- Proof is not required and thus omitted

end NUMINAMATH_GPT_green_marble_prob_l1210_121009


namespace NUMINAMATH_GPT_simplify_expression_l1210_121056

variable (x y : ℝ)

theorem simplify_expression : (x^2 + x * y) / (x * y) * (y^2 / (x + y)) = y := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1210_121056


namespace NUMINAMATH_GPT_ending_number_divisible_by_9_l1210_121012

theorem ending_number_divisible_by_9 (E : ℕ) 
  (h1 : ∀ n, 10 ≤ n → n ≤ E → n % 9 = 0 → ∃ m ≥ 1, n = 18 + 9 * (m - 1)) 
  (h2 : (E - 18) / 9 + 1 = 111110) : 
  E = 999999 :=
by
  sorry

end NUMINAMATH_GPT_ending_number_divisible_by_9_l1210_121012


namespace NUMINAMATH_GPT_Jaron_prize_points_l1210_121081

def points_bunnies (bunnies: Nat) (points_per_bunny: Nat) : Nat :=
  bunnies * points_per_bunny

def points_snickers (snickers: Nat) (points_per_snicker: Nat) : Nat :=
  snickers * points_per_snicker

def total_points (bunny_points: Nat) (snicker_points: Nat) : Nat :=
  bunny_points + snicker_points

theorem Jaron_prize_points :
  let bunnies := 8
  let points_per_bunny := 100
  let snickers := 48
  let points_per_snicker := 25
  let bunny_points := points_bunnies bunnies points_per_bunny
  let snicker_points := points_snickers snickers points_per_snicker
  total_points bunny_points snicker_points = 2000 := 
by
  sorry

end NUMINAMATH_GPT_Jaron_prize_points_l1210_121081


namespace NUMINAMATH_GPT_track_length_l1210_121078

theorem track_length (x : ℝ) : 
  (∃ B S : ℝ, B + S = x ∧ S = (x / 2 - 75) ∧ B = 75 ∧ S + 100 = x / 2 + 25 ∧ B = x / 2 - 50 ∧ B / S = (x / 2 - 50) / 100) → 
  x = 220 :=
by
  sorry

end NUMINAMATH_GPT_track_length_l1210_121078


namespace NUMINAMATH_GPT_combined_weight_l1210_121063

-- Define the main proof problem
theorem combined_weight (student_weight : ℝ) (sister_weight : ℝ) :
  (student_weight - 5 = 2 * sister_weight) ∧ (student_weight = 79) → (student_weight + sister_weight = 116) :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_l1210_121063


namespace NUMINAMATH_GPT_cards_left_l1210_121010

variable (initialCards : ℕ) (givenCards : ℕ) (remainingCards : ℕ)

def JasonInitialCards := 13
def CardsGivenAway := 9

theorem cards_left : initialCards = JasonInitialCards → givenCards = CardsGivenAway → remainingCards = initialCards - givenCards → remainingCards = 4 :=
by
  intros
  subst_vars
  sorry

end NUMINAMATH_GPT_cards_left_l1210_121010


namespace NUMINAMATH_GPT_double_neg_eq_pos_l1210_121061

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end NUMINAMATH_GPT_double_neg_eq_pos_l1210_121061


namespace NUMINAMATH_GPT_train_length_l1210_121059

theorem train_length (L : ℝ) : (L + 200) / 15 = (L + 300) / 20 → L = 100 :=
by
  intro h
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_train_length_l1210_121059


namespace NUMINAMATH_GPT_first_bakery_sacks_per_week_l1210_121006

theorem first_bakery_sacks_per_week (x : ℕ) 
    (H1 : 4 * x + 4 * 4 + 4 * 12 = 72) : x = 2 :=
by 
  -- we will provide the proof here if needed
  sorry

end NUMINAMATH_GPT_first_bakery_sacks_per_week_l1210_121006


namespace NUMINAMATH_GPT_packet_weight_l1210_121062

theorem packet_weight :
  ∀ (num_packets : ℕ) (total_weight_kg : ℕ), 
  num_packets = 20 → total_weight_kg = 2 →
  (total_weight_kg * 1000) / num_packets = 100 := by
  intro num_packets total_weight_kg h1 h2
  sorry

end NUMINAMATH_GPT_packet_weight_l1210_121062


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_a9_l1210_121055

theorem arithmetic_sequence_a1_a9 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum_456 : a 4 + a 5 + a 6 = 36) : 
  a 1 + a 9 = 24 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_a9_l1210_121055


namespace NUMINAMATH_GPT_possible_combinations_of_scores_l1210_121080

theorem possible_combinations_of_scores 
    (scores : Set ℕ := {0, 3, 5})
    (total_scores : ℕ := 32)
    (teams : ℕ := 3)
    : (∃ (number_of_combinations : ℕ), number_of_combinations = 255) := by
  sorry

end NUMINAMATH_GPT_possible_combinations_of_scores_l1210_121080


namespace NUMINAMATH_GPT_estimate_sqrt_expr_l1210_121070

theorem estimate_sqrt_expr :
  2 < (Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3)) ∧ 
  (Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3)) < 3 := 
sorry

end NUMINAMATH_GPT_estimate_sqrt_expr_l1210_121070


namespace NUMINAMATH_GPT_find_least_N_exists_l1210_121048

theorem find_least_N_exists (N : ℕ) :
  (∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), 
    N = (a₁ + 2) * (b₁ + 2) * (c₁ + 2) - 8 ∧ 
    N + 1 = (a₂ + 2) * (b₂ + 2) * (c₂ + 2) - 8) ∧
  N = 55 := 
sorry

end NUMINAMATH_GPT_find_least_N_exists_l1210_121048


namespace NUMINAMATH_GPT_sum_possible_x_l1210_121090

noncomputable def sum_of_x (x : ℝ) : ℝ :=
  let lst : List ℝ := [1, 2, 5, 2, 3, 2, x]
  let mean := (1 + 2 + 5 + 2 + 3 + 2 + x) / 7
  let median := 2
  let mode := 2
  if lst = List.reverse lst ∧ mean ≠ mode then
    mean
  else 
    0

theorem sum_possible_x : sum_of_x 1 + sum_of_x 5 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_sum_possible_x_l1210_121090


namespace NUMINAMATH_GPT_segment_length_l1210_121018
noncomputable def cube_root27 : ℝ := 3

theorem segment_length : ∀ (x : ℝ), (|x - cube_root27| = 4) → ∃ (a b : ℝ), (a = cube_root27 + 4) ∧ (b = cube_root27 - 4) ∧ |a - b| = 8 :=
by
  sorry

end NUMINAMATH_GPT_segment_length_l1210_121018


namespace NUMINAMATH_GPT_multiplication_of_decimals_l1210_121068

theorem multiplication_of_decimals : (0.4 * 0.75 = 0.30) := by
  sorry

end NUMINAMATH_GPT_multiplication_of_decimals_l1210_121068


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_a4_value_l1210_121015

theorem arithmetic_geometric_sequence_a4_value 
  (a : ℕ → ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4) : 
  a 4 = 1 := 
sorry

end NUMINAMATH_GPT_arithmetic_geometric_sequence_a4_value_l1210_121015


namespace NUMINAMATH_GPT_total_cost_of_barbed_wire_l1210_121058

noncomputable def cost_of_barbed_wire : ℝ :=
  let area : ℝ := 3136
  let side_length : ℝ := Real.sqrt area
  let perimeter_without_gates : ℝ := 4 * side_length - 2 * 1
  let rate_per_meter : ℝ := 1.10
  perimeter_without_gates * rate_per_meter

theorem total_cost_of_barbed_wire :
  cost_of_barbed_wire = 244.20 :=
sorry

end NUMINAMATH_GPT_total_cost_of_barbed_wire_l1210_121058


namespace NUMINAMATH_GPT_no_integer_solutions_l1210_121032

theorem no_integer_solutions (x y : ℤ) : x^3 + 4 * x^2 + x ≠ 18 * y^3 + 18 * y^2 + 6 * y + 3 := 
by 
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1210_121032


namespace NUMINAMATH_GPT_find_3x2y2_l1210_121046

theorem find_3x2y2 (x y : ℤ) 
  (h1 : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 := by
  sorry

end NUMINAMATH_GPT_find_3x2y2_l1210_121046


namespace NUMINAMATH_GPT_units_digit_two_pow_2010_l1210_121023

-- Conditions from part a)
def two_power_units_digit (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => 0 -- This case will not occur due to modulo operation

-- Question translated to a proof problem
theorem units_digit_two_pow_2010 : (two_power_units_digit 2010) = 4 :=
by 
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_units_digit_two_pow_2010_l1210_121023


namespace NUMINAMATH_GPT_sin_2A_cos_C_l1210_121043

theorem sin_2A (A B : ℝ) (h1 : Real.sin A = 3 / 5) (h2 : Real.cos B = -5 / 13) : 
  Real.sin (2 * A) = 24 / 25 :=
sorry

theorem cos_C (A B C : ℝ) (h1 : Real.sin A = 3 / 5) (h2 : Real.cos B = -5 / 13) 
  (h3 : ∀ x y z : ℝ, x + y + z = π) :
  Real.cos C = 56 / 65 :=
sorry

end NUMINAMATH_GPT_sin_2A_cos_C_l1210_121043


namespace NUMINAMATH_GPT_non_degenerate_ellipse_condition_l1210_121050

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ (x y : ℝ), x^2 + 9 * y^2 - 6 * x + 18 * y = k) → k > -9 :=
by
  sorry

end NUMINAMATH_GPT_non_degenerate_ellipse_condition_l1210_121050


namespace NUMINAMATH_GPT_exists_ellipse_l1210_121084

theorem exists_ellipse (a : ℝ) : ∃ a : ℝ, ∀ x y : ℝ, (x^2 + y^2 / a = 1) → a > 0 ∧ a ≠ 1 := 
by 
  sorry

end NUMINAMATH_GPT_exists_ellipse_l1210_121084


namespace NUMINAMATH_GPT_rational_eq_reciprocal_l1210_121037

theorem rational_eq_reciprocal (x : ℚ) (h : x = 1 / x) : x = 1 ∨ x = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_rational_eq_reciprocal_l1210_121037


namespace NUMINAMATH_GPT_flour_for_each_cupcake_l1210_121045

noncomputable def flour_per_cupcake (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ) : ℝ :=
  remaining_flour / num_cupcakes

theorem flour_for_each_cupcake :
  ∀ (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ),
    total_flour = 6 →
    remaining_flour = 2 →
    cake_flour_per_cake = 0.5 →
    cake_price = 2.5 →
    cupcake_price = 1 →
    total_revenue = 30 →
    num_cakes = 4 / 0.5 →
    num_cupcakes = 10 →
    flour_per_cupcake total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes = 0.2 :=
by intros; sorry

end NUMINAMATH_GPT_flour_for_each_cupcake_l1210_121045


namespace NUMINAMATH_GPT_find_k_l1210_121087

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x ^ 2 + (k - 1) * x + 3

theorem find_k (k : ℝ) (h : ∀ x, f k x = f k (-x)) : k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1210_121087


namespace NUMINAMATH_GPT_function_periodic_l1210_121076

open Real

def periodic (f : ℝ → ℝ) := ∃ T > 0, ∀ x, f (x + T) = f x

theorem function_periodic (a : ℚ) (b d c : ℝ) (f : ℝ → ℝ) 
    (hf : ∀ x : ℝ, f (x + ↑a + b) - f (x + b) = c * (x + 2 * ↑a + ⌊x⌋ - 2 * ⌊x + ↑a⌋ - ⌊b⌋) + d) : 
    periodic f :=
sorry

end NUMINAMATH_GPT_function_periodic_l1210_121076


namespace NUMINAMATH_GPT_total_eyes_in_extended_family_l1210_121092

def mom_eyes := 1
def dad_eyes := 3
def kids_eyes := 3 * 4
def moms_previous_child_eyes := 5
def dads_previous_children_eyes := 6 + 2
def dads_ex_wife_eyes := 1
def dads_ex_wifes_new_partner_eyes := 7
def child_of_ex_wife_and_partner_eyes := 8

theorem total_eyes_in_extended_family :
  mom_eyes + dad_eyes + kids_eyes + moms_previous_child_eyes + dads_previous_children_eyes +
  dads_ex_wife_eyes + dads_ex_wifes_new_partner_eyes + child_of_ex_wife_and_partner_eyes = 45 :=
by
  -- add proof here
  sorry

end NUMINAMATH_GPT_total_eyes_in_extended_family_l1210_121092


namespace NUMINAMATH_GPT_pentagon_perimeter_l1210_121021

-- Problem statement: Given an irregular pentagon with specified side lengths,
-- prove that its perimeter is equal to 52.9 cm.

theorem pentagon_perimeter 
  (a b c d e : ℝ)
  (h1 : a = 5.2)
  (h2 : b = 10.3)
  (h3 : c = 15.8)
  (h4 : d = 8.7)
  (h5 : e = 12.9) 
  : a + b + c + d + e = 52.9 := 
by
  sorry

end NUMINAMATH_GPT_pentagon_perimeter_l1210_121021


namespace NUMINAMATH_GPT_quadratic_points_order_l1210_121044

theorem quadratic_points_order (y1 y2 y3 : ℝ) :
  (y1 = -2 * (1:ℝ) ^ 2 + 4) →
  (y2 = -2 * (2:ℝ) ^ 2 + 4) →
  (y3 = -2 * (-3:ℝ) ^ 2 + 4) →
  y1 > y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_quadratic_points_order_l1210_121044


namespace NUMINAMATH_GPT_angle_C_is_100_l1210_121067

-- Define the initial measures in the equilateral triangle
def initial_angle (A B C : ℕ) (h_equilateral : A = B ∧ B = C ∧ C = 60) : ℕ := C

-- Definition to capture the increase in angle C
def increased_angle (C : ℕ) : ℕ := C + 40

-- Now, we need to state the theorem assuming the given conditions
theorem angle_C_is_100
  (A B C : ℕ)
  (h_equilateral : A = 60 ∧ B = 60 ∧ C = 60)
  (h_increase : C = 60 + 40)
  : C = 100 := 
sorry

end NUMINAMATH_GPT_angle_C_is_100_l1210_121067


namespace NUMINAMATH_GPT_farm_area_l1210_121013

theorem farm_area (length width area : ℝ) 
  (h1 : length = 0.6) 
  (h2 : width = 3 * length) 
  (h3 : area = length * width) : 
  area = 1.08 := 
by 
  sorry

end NUMINAMATH_GPT_farm_area_l1210_121013


namespace NUMINAMATH_GPT_distance_from_Asheville_to_Darlington_l1210_121000

theorem distance_from_Asheville_to_Darlington (BC AC BD AD : ℝ) 
(h0 : BC = 12) 
(h1 : BC = (1/3) * AC) 
(h2 : BC = (1/4) * BD) :
AD = 72 :=
sorry

end NUMINAMATH_GPT_distance_from_Asheville_to_Darlington_l1210_121000


namespace NUMINAMATH_GPT_sum_of_solutions_eq_zero_l1210_121073

theorem sum_of_solutions_eq_zero :
  ∀ x : ℝ, (-π ≤ x ∧ x ≤ 3 * π ∧ (1 / Real.sin x + 1 / Real.cos x = 4))
  → x = 0 := sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_zero_l1210_121073


namespace NUMINAMATH_GPT_function_above_x_axis_l1210_121031

noncomputable def quadratic_function (a x : ℝ) := (a^2 - 3 * a + 2) * x^2 + (a - 1) * x + 2

theorem function_above_x_axis (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x > 0) ↔ (a > 15 / 7 ∨ a ≤ 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_function_above_x_axis_l1210_121031


namespace NUMINAMATH_GPT_coefficient_of_quadratic_polynomial_l1210_121019

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem coefficient_of_quadratic_polynomial (a b c : ℝ) (h : a > 0) :
  |f a b c 1| = 2 ∧ |f a b c 2| = 2 ∧ |f a b c 3| = 2 →
  (a = 4 ∧ b = -16 ∧ c = 14) ∨ (a = 2 ∧ b = -6 ∧ c = 2) ∨ (a = 2 ∧ b = -10 ∧ c = 10) :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_quadratic_polynomial_l1210_121019


namespace NUMINAMATH_GPT_study_tour_buses_l1210_121083

variable (x : ℕ) (num_people : ℕ)

def seats_A := 45
def seats_B := 60
def extra_people := 30
def fewer_B := 6

theorem study_tour_buses (h : seats_A * x + extra_people = seats_B * (x - fewer_B)) : 
  x = 26 ∧ (seats_A * 26 + extra_people = 1200) := 
  sorry

end NUMINAMATH_GPT_study_tour_buses_l1210_121083


namespace NUMINAMATH_GPT_max_m_value_inequality_abc_for_sum_l1210_121064

-- Define the mathematical conditions and the proof problem.

theorem max_m_value (x m : ℝ) (h1 : |x - 2| - |x + 3| ≥ |m + 1|) :
  m ≤ 4 :=
sorry

theorem inequality_abc_for_sum (a b c : ℝ) (habc_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_eq_M : a + 2 * b + c = 4) :
  (1 / (a + b)) + (1 / (b + c)) ≥ 1 :=
sorry

end NUMINAMATH_GPT_max_m_value_inequality_abc_for_sum_l1210_121064
