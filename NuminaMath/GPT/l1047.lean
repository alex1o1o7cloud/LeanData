import Mathlib

namespace NUMINAMATH_GPT_lenny_initial_money_l1047_104720

-- Definitions based on the conditions
def spent_on_video_games : ℕ := 24
def spent_at_grocery_store : ℕ := 21
def amount_left : ℕ := 39

-- Statement of the problem
theorem lenny_initial_money : spent_on_video_games + spent_at_grocery_store + amount_left = 84 :=
by
  sorry

end NUMINAMATH_GPT_lenny_initial_money_l1047_104720


namespace NUMINAMATH_GPT_number_of_intersection_points_l1047_104763

noncomputable section

-- Define a type for Points in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining the five points
variables (A B C D E : Point)

-- Define the conditions that no three points are collinear
def no_three_collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

-- Define the theorem statement
theorem number_of_intersection_points (h1 : no_three_collinear A B C)
  (h2 : no_three_collinear A B D)
  (h3 : no_three_collinear A B E)
  (h4 : no_three_collinear A C D)
  (h5 : no_three_collinear A C E)
  (h6 : no_three_collinear A D E)
  (h7 : no_three_collinear B C D)
  (h8 : no_three_collinear B C E)
  (h9 : no_three_collinear B D E)
  (h10 : no_three_collinear C D E) :
  ∃ (N : ℕ), N = 40 :=
  sorry

end NUMINAMATH_GPT_number_of_intersection_points_l1047_104763


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1047_104795

theorem quadratic_inequality_solution:
  ∀ x : ℝ, (x^2 + 2 * x < 3) ↔ (-3 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1047_104795


namespace NUMINAMATH_GPT_quadratic_eq_solutions_l1047_104757

open Real

theorem quadratic_eq_solutions (x : ℝ) :
  (2 * x + 1) ^ 2 = (2 * x + 1) * (x - 1) ↔ x = -1 / 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_GPT_quadratic_eq_solutions_l1047_104757


namespace NUMINAMATH_GPT_complement_union_l1047_104707

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (hM : M = {1, 3, 5, 7}) (hN : N = {5, 6, 7}) : U \ (M ∪ N) = {2, 4, 8} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l1047_104707


namespace NUMINAMATH_GPT_find_special_integers_l1047_104799

theorem find_special_integers 
  : ∃ n : ℕ, 100 ≤ n ∧ n ≤ 1997 ∧ (2^n + 2) % n = 0 ∧ (n = 66 ∨ n = 198 ∨ n = 398 ∨ n = 798) :=
by
  sorry

end NUMINAMATH_GPT_find_special_integers_l1047_104799


namespace NUMINAMATH_GPT_correct_propositions_l1047_104730

structure Proposition :=
  (statement : String)
  (is_correct : Prop)

def prop1 : Proposition := {
  statement := "All sufficiently small positive numbers form a set.",
  is_correct := False -- From step b
}

def prop2 : Proposition := {
  statement := "The set containing 1, 2, 3, 1, 9 is represented by enumeration as {1, 2, 3, 1, 9}.",
  is_correct := False -- From step b
}

def prop3 : Proposition := {
  statement := "{1, 3, 5, 7} and {7, 5, 3, 1} denote the same set.",
  is_correct := True -- From step b
}

def prop4 : Proposition := {
  statement := "{y = -x} represents the collection of all points on the graph of the function y = -x.",
  is_correct := False -- From step b
}

theorem correct_propositions :
  prop3.is_correct ∧ ¬prop1.is_correct ∧ ¬prop2.is_correct ∧ ¬prop4.is_correct :=
by
  -- Here we put the proof steps, but for the exercise's purpose, we use sorry.
  sorry

end NUMINAMATH_GPT_correct_propositions_l1047_104730


namespace NUMINAMATH_GPT_wall_width_l1047_104742

theorem wall_width (brick_length brick_height brick_depth : ℝ)
    (wall_length wall_height : ℝ)
    (num_bricks : ℝ)
    (total_bricks_volume : ℝ)
    (total_wall_volume : ℝ) :
    brick_length = 25 →
    brick_height = 11.25 →
    brick_depth = 6 →
    wall_length = 800 →
    wall_height = 600 →
    num_bricks = 6400 →
    total_bricks_volume = num_bricks * (brick_length * brick_height * brick_depth) →
    total_wall_volume = wall_length * wall_height * (total_bricks_volume / (brick_length * brick_height * brick_depth)) →
    (total_bricks_volume / (wall_length * wall_height) = 22.5) :=
by
  intros
  sorry -- proof not required

end NUMINAMATH_GPT_wall_width_l1047_104742


namespace NUMINAMATH_GPT_order_of_a_b_c_l1047_104728

noncomputable def a := 2 + Real.sqrt 3
noncomputable def b := 1 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 + Real.sqrt 5

theorem order_of_a_b_c : a > c ∧ c > b := 
by {
  sorry
}

end NUMINAMATH_GPT_order_of_a_b_c_l1047_104728


namespace NUMINAMATH_GPT_temperature_on_Monday_l1047_104770

theorem temperature_on_Monday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 31) : 
  M = 39 :=
by
  sorry

end NUMINAMATH_GPT_temperature_on_Monday_l1047_104770


namespace NUMINAMATH_GPT_parkingGarageCharges_l1047_104769

variable (W : ℕ)

/-- 
  Conditions:
  1. Weekly rental cost is \( W \) dollars.
  2. Monthly rental cost is $24 per month.
  3. A person saves $232 in a year by renting by the month rather than by the week.
  4. There are 52 weeks in a year.
  5. There are 12 months in a year.
-/
def garageChargesPerWeek : Prop :=
  52 * W = 12 * 24 + 232

theorem parkingGarageCharges
  (h : garageChargesPerWeek W) : W = 10 :=
by
  sorry

end NUMINAMATH_GPT_parkingGarageCharges_l1047_104769


namespace NUMINAMATH_GPT_problem1_l1047_104778

theorem problem1 :
  0.064^(-1 / 3) - (-1 / 8)^0 + 16^(3 / 4) + 0.25^(1 / 2) = 10 :=
by
  sorry

end NUMINAMATH_GPT_problem1_l1047_104778


namespace NUMINAMATH_GPT_gcd_612_468_l1047_104732

theorem gcd_612_468 : gcd 612 468 = 36 :=
by
  sorry

end NUMINAMATH_GPT_gcd_612_468_l1047_104732


namespace NUMINAMATH_GPT_mark_trees_total_l1047_104736

def mark_trees (current_trees new_trees : Nat) : Nat :=
  current_trees + new_trees

theorem mark_trees_total (x y : Nat) (h1 : x = 13) (h2 : y = 12) :
  mark_trees x y = 25 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_mark_trees_total_l1047_104736


namespace NUMINAMATH_GPT_total_unbroken_seashells_l1047_104748

/-
Given:
On the first day, Tom found 7 seashells but 4 were broken.
On the second day, he found 12 seashells but 5 were broken.
On the third day, he found 15 seashells but 8 were broken.

We need to prove that Tom found 17 unbroken seashells in total over the three days.
-/

def first_day_total := 7
def first_day_broken := 4
def first_day_unbroken := first_day_total - first_day_broken

def second_day_total := 12
def second_day_broken := 5
def second_day_unbroken := second_day_total - second_day_broken

def third_day_total := 15
def third_day_broken := 8
def third_day_unbroken := third_day_total - third_day_broken

def total_unbroken := first_day_unbroken + second_day_unbroken + third_day_unbroken

theorem total_unbroken_seashells : total_unbroken = 17 := by
  sorry

end NUMINAMATH_GPT_total_unbroken_seashells_l1047_104748


namespace NUMINAMATH_GPT_g_at_3_l1047_104734

def g (x : ℝ) : ℝ := -3 * x^4 + 4 * x^3 - 7 * x^2 + 5 * x - 2

theorem g_at_3 : g 3 = -185 := by
  sorry

end NUMINAMATH_GPT_g_at_3_l1047_104734


namespace NUMINAMATH_GPT_jonas_pairs_of_pants_l1047_104756

theorem jonas_pairs_of_pants (socks pairs_of_shoes t_shirts new_socks : Nat) (P : Nat) :
  socks = 20 → pairs_of_shoes = 5 → t_shirts = 10 → new_socks = 35 →
  2 * (2 * socks + 2 * pairs_of_shoes + t_shirts + P) = 2 * (2 * socks + 2 * pairs_of_shoes + t_shirts) + 70 →
  P = 5 :=
by
  intros hs hps ht hr htotal
  sorry

end NUMINAMATH_GPT_jonas_pairs_of_pants_l1047_104756


namespace NUMINAMATH_GPT_volume_not_occupied_by_cones_l1047_104706

/-- Two cones with given dimensions are enclosed in a cylinder, and we want to find the volume 
    in the cylinder not occupied by the cones. -/
theorem volume_not_occupied_by_cones : 
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  volume_cylinder - total_volume_cones = (2600 / 3) * π :=
by
  let radius := 10
  let height_cylinder := 26
  let height_cone1 := 10
  let height_cone2 := 16
  let volume_cylinder := π * (radius ^ 2) * height_cylinder
  let volume_cone1 := (1 / 3) * π * (radius ^ 2) * height_cone1
  let volume_cone2 := (1 / 3) * π * (radius ^ 2) * height_cone2
  let total_volume_cones := volume_cone1 + volume_cone2
  sorry

end NUMINAMATH_GPT_volume_not_occupied_by_cones_l1047_104706


namespace NUMINAMATH_GPT_lamp_height_difference_l1047_104796

def old_lamp_height : ℝ := 1
def new_lamp_height : ℝ := 2.3333333333333335
def height_difference : ℝ := new_lamp_height - old_lamp_height

theorem lamp_height_difference :
  height_difference = 1.3333333333333335 := by
  sorry

end NUMINAMATH_GPT_lamp_height_difference_l1047_104796


namespace NUMINAMATH_GPT_balance_force_l1047_104724

structure Vector2D where
  x : ℝ
  y : ℝ

def F1 : Vector2D := ⟨1, 1⟩
def F2 : Vector2D := ⟨2, 3⟩

def vector_add (a b : Vector2D) : Vector2D := ⟨a.x + b.x, a.y + b.y⟩
def vector_neg (a : Vector2D) : Vector2D := ⟨-a.x, -a.y⟩

theorem balance_force : 
  ∃ F3 : Vector2D, vector_add (vector_add F1 F2) F3 = ⟨0, 0⟩ ∧ F3 = ⟨-3, -4⟩ := 
by
  sorry

end NUMINAMATH_GPT_balance_force_l1047_104724


namespace NUMINAMATH_GPT_goose_survived_first_year_l1047_104725

theorem goose_survived_first_year (total_eggs : ℕ) (eggs_hatched_ratio : ℚ) (first_month_survival_ratio : ℚ) 
  (first_year_no_survival_ratio : ℚ) 
  (eggs_hatched_ratio_eq : eggs_hatched_ratio = 2/3) 
  (first_month_survival_ratio_eq : first_month_survival_ratio = 3/4)
  (first_year_no_survival_ratio_eq : first_year_no_survival_ratio = 3/5)
  (total_eggs_eq : total_eggs = 500) :
  ∃ (survived_first_year : ℕ), survived_first_year = 100 :=
by
  sorry

end NUMINAMATH_GPT_goose_survived_first_year_l1047_104725


namespace NUMINAMATH_GPT_length_of_diagonal_l1047_104710

theorem length_of_diagonal (area : ℝ) (h1 h2 : ℝ) (d : ℝ) 
  (h_area : area = 75)
  (h_offsets : h1 = 6 ∧ h2 = 4) :
  d = 15 :=
by
  -- Given the conditions and formula, we can conclude
  sorry

end NUMINAMATH_GPT_length_of_diagonal_l1047_104710


namespace NUMINAMATH_GPT_second_train_speed_l1047_104703

theorem second_train_speed (len1 len2 dist t : ℕ) (h1 : len1 = 100) (h2 : len2 = 150) (h3 : dist = 50) (h4 : t = 60) : 
  (len1 + len2 + dist) / t = 5 := 
  by
  -- Definitions from conditions
  have h_len1 : len1 = 100 := h1
  have h_len2 : len2 = 150 := h2
  have h_dist : dist = 50 := h3
  have h_time : t = 60 := h4
  
  -- Proof deferred
  sorry

end NUMINAMATH_GPT_second_train_speed_l1047_104703


namespace NUMINAMATH_GPT_area_of_rectangular_field_l1047_104788

theorem area_of_rectangular_field (W L : ℕ) (hL : L = 10) (hFencing : 2 * W + L = 146) : W * L = 680 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangular_field_l1047_104788


namespace NUMINAMATH_GPT_fraction_decomposition_l1047_104780

theorem fraction_decomposition (A B : ℚ) :
  (∀ x : ℚ, x ≠ -1 ∧ x ≠ 2  →
    7 * x - 18 = A * (3 * x + 1) + B * (x - 2))
  ↔ (A = -4 / 7 ∧ B = 61 / 7) :=
by
  sorry

end NUMINAMATH_GPT_fraction_decomposition_l1047_104780


namespace NUMINAMATH_GPT_min_score_guarantees_payoff_l1047_104708

-- Defining the probability of a single roll being a six
def prob_single_six : ℚ := 1 / 6 

-- Defining the event of rolling two sixes independently
def prob_two_sixes : ℚ := prob_single_six * prob_single_six

-- Defining the score of two die rolls summing up to 12
def is_score_twelve (a b : ℕ) : Prop := a + b = 12

-- Proving the probability of Jim scoring 12 in two rolls guarantees some monetary payoff.
theorem min_score_guarantees_payoff :
  (prob_two_sixes = 1/36) :=
by
  sorry

end NUMINAMATH_GPT_min_score_guarantees_payoff_l1047_104708


namespace NUMINAMATH_GPT_quadratic_function_value_at_neg_one_l1047_104743

theorem quadratic_function_value_at_neg_one (b c : ℝ) 
  (h1 : (1:ℝ) ^ 2 + b * 1 + c = 0) 
  (h2 : (3:ℝ) ^ 2 + b * 3 + c = 0) : 
  ((-1:ℝ) ^ 2 + b * (-1) + c = 8) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_value_at_neg_one_l1047_104743


namespace NUMINAMATH_GPT_rectangular_prism_volume_l1047_104777

variables (a b c : ℝ)

theorem rectangular_prism_volume
  (h1 : a * b = 24)
  (h2 : b * c = 8)
  (h3 : c * a = 3) :
  a * b * c = 24 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l1047_104777


namespace NUMINAMATH_GPT_eval_expr_equals_1_l1047_104767

noncomputable def eval_expr (a b : ℕ) : ℚ :=
  (a + b) / (a * b) / ((a / b) - (b / a))

theorem eval_expr_equals_1 (a b : ℕ) (h₁ : a = 3) (h₂ : b = 2) : eval_expr a b = 1 :=
by
  sorry

end NUMINAMATH_GPT_eval_expr_equals_1_l1047_104767


namespace NUMINAMATH_GPT_composite_sum_of_powers_l1047_104798

theorem composite_sum_of_powers (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ a^2016 + b^2016 + c^2016 + d^2016 = x * y :=
by sorry

end NUMINAMATH_GPT_composite_sum_of_powers_l1047_104798


namespace NUMINAMATH_GPT_cylinder_height_l1047_104750

theorem cylinder_height {D r : ℝ} (hD : D = 10) (hr : r = 3) : 
  ∃ h : ℝ, h = 8 :=
by
  -- hD -> Diameter of hemisphere = 10
  -- hr -> Radius of cylinder's base = 3
  sorry

end NUMINAMATH_GPT_cylinder_height_l1047_104750


namespace NUMINAMATH_GPT_correct_values_correct_result_l1047_104764

theorem correct_values (a b : ℝ) :
  ((2 * x - a) * (3 * x + b) = 6 * x^2 + 11 * x - 10) ∧
  ((2 * x + a) * (x + b) = 2 * x^2 - 9 * x + 10) →
  (a = -5) ∧ (b = -2) :=
sorry

theorem correct_result :
  (2 * x - 5) * (3 * x - 2) = 6 * x^2 - 19 * x + 10 :=
sorry

end NUMINAMATH_GPT_correct_values_correct_result_l1047_104764


namespace NUMINAMATH_GPT_dan_speed_must_exceed_48_l1047_104711

theorem dan_speed_must_exceed_48 (d : ℕ) (s_cara : ℕ) (time_delay : ℕ) : 
  d = 120 → s_cara = 30 → time_delay = 3 / 2 → ∃ v : ℕ, v > 48 :=
by
  intro h1 h2 h3
  use 49
  sorry

end NUMINAMATH_GPT_dan_speed_must_exceed_48_l1047_104711


namespace NUMINAMATH_GPT_min_S_value_l1047_104714

noncomputable def S (x y z : ℝ) : ℝ := (1 + z) / (2 * x * y * z)

theorem min_S_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = 1) :
  S x y z ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_S_value_l1047_104714


namespace NUMINAMATH_GPT_lizard_eyes_fewer_than_spots_and_wrinkles_l1047_104709

noncomputable def lizard_problem : Nat :=
  let eyes_jan := 3
  let wrinkles_jan := 3 * eyes_jan
  let spots_jan := 7 * (wrinkles_jan ^ 2)
  let eyes_cousin := 3
  let wrinkles_cousin := 2 * eyes_cousin
  let spots_cousin := 5 * (wrinkles_cousin ^ 2)
  let total_eyes := eyes_jan + eyes_cousin
  let total_wrinkles := wrinkles_jan + wrinkles_cousin
  let total_spots := spots_jan + spots_cousin
  (total_spots + total_wrinkles) - total_eyes

theorem lizard_eyes_fewer_than_spots_and_wrinkles :
  lizard_problem = 756 :=
by
  sorry

end NUMINAMATH_GPT_lizard_eyes_fewer_than_spots_and_wrinkles_l1047_104709


namespace NUMINAMATH_GPT_student_ticket_price_is_2_50_l1047_104794

-- Defining the given conditions
def adult_ticket_price : ℝ := 4
def total_tickets_sold : ℕ := 59
def total_revenue : ℝ := 222.50
def student_tickets_sold : ℕ := 9

-- The number of adult tickets sold
def adult_tickets_sold : ℕ := total_tickets_sold - student_tickets_sold

-- The total revenue from adult tickets
def revenue_from_adult_tickets : ℝ := adult_tickets_sold * adult_ticket_price

-- The remaining revenue must come from student tickets and defining the price of student ticket
noncomputable def student_ticket_price : ℝ :=
  (total_revenue - revenue_from_adult_tickets) / student_tickets_sold

-- The theorem to be proved
theorem student_ticket_price_is_2_50 : student_ticket_price = 2.50 :=
by
  sorry

end NUMINAMATH_GPT_student_ticket_price_is_2_50_l1047_104794


namespace NUMINAMATH_GPT_initial_cloves_l1047_104758

theorem initial_cloves (used_cloves left_cloves initial_cloves : ℕ) (h1 : used_cloves = 86) (h2 : left_cloves = 7) : initial_cloves = 93 :=
by
  sorry

end NUMINAMATH_GPT_initial_cloves_l1047_104758


namespace NUMINAMATH_GPT_nested_roots_identity_l1047_104761

theorem nested_roots_identity (x : ℝ) (hx : x ≥ 0) : Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = x ^ (15 / 16) :=
sorry

end NUMINAMATH_GPT_nested_roots_identity_l1047_104761


namespace NUMINAMATH_GPT_limit_exists_implies_d_eq_zero_l1047_104719

variable (a₁ d : ℝ) (S : ℕ → ℝ)

noncomputable def limExists := ∃ L : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (S n - L) < ε

def is_sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) :=
  ∀ n : ℕ, S n = (a₁ * n + d * (n * (n - 1) / 2))

theorem limit_exists_implies_d_eq_zero (h₁ : ∀ n : ℕ, n > 0 → S n = (a₁ * n + d * (n * (n - 1) / 2))) :
  limExists S → d = 0 :=
by sorry

end NUMINAMATH_GPT_limit_exists_implies_d_eq_zero_l1047_104719


namespace NUMINAMATH_GPT_sally_balloon_count_l1047_104749

theorem sally_balloon_count (n_initial : ℕ) (n_lost : ℕ) (n_final : ℕ) 
  (h_initial : n_initial = 9) 
  (h_lost : n_lost = 2) 
  (h_final : n_final = n_initial - n_lost) : 
  n_final = 7 :=
by
  sorry

end NUMINAMATH_GPT_sally_balloon_count_l1047_104749


namespace NUMINAMATH_GPT_min_value_of_fraction_sum_l1047_104739

theorem min_value_of_fraction_sum (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_sum : x^2 + y^2 + z^2 = 1) :
  (2 * (1/(1-x^2) + 1/(1-y^2) + 1/(1-z^2))) = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_min_value_of_fraction_sum_l1047_104739


namespace NUMINAMATH_GPT_area_of_BCD_l1047_104701

variables (a b c x y : ℝ)

-- Conditions
axiom h1 : x = (1 / 2) * a * b
axiom h2 : y = (1 / 2) * b * c

-- Conclusion to prove
theorem area_of_BCD (a b c x y : ℝ) (h1 : x = (1 / 2) * a * b) (h2 : y = (1 / 2) * b * c) : 
  (1 / 2) * b * c = y :=
sorry

end NUMINAMATH_GPT_area_of_BCD_l1047_104701


namespace NUMINAMATH_GPT_joeys_age_next_multiple_l1047_104793

-- Definitions of the conditions and problem setup
def joey_age (chloe_age : ℕ) : ℕ := chloe_age + 2
def max_age : ℕ := 2
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Main Lean statement
theorem joeys_age_next_multiple (chloe_age : ℕ) (H1 : is_prime chloe_age)
  (H2 : ∀ n : ℕ, (joey_age chloe_age + n) % (max_age + n) = 0)
  (H3 : ∀ i : ℕ, i < 11 → is_prime (chloe_age + i))
  : sum_of_digits (joey_age chloe_age + 1) = 5 :=
  sorry

end NUMINAMATH_GPT_joeys_age_next_multiple_l1047_104793


namespace NUMINAMATH_GPT_tyler_meals_l1047_104717

def num_meals : ℕ := 
  let num_meats := 3
  let num_vegetable_combinations := Nat.choose 5 3
  let num_desserts := 5
  num_meats * num_vegetable_combinations * num_desserts

theorem tyler_meals :
  num_meals = 150 := by
  sorry

end NUMINAMATH_GPT_tyler_meals_l1047_104717


namespace NUMINAMATH_GPT_find_f_105_5_l1047_104766

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom product_condition : ∀ x : ℝ, f x * f (x + 2) = -1
axiom specific_interval : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → f x = x

theorem find_f_105_5 : f 105.5 = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_find_f_105_5_l1047_104766


namespace NUMINAMATH_GPT_max_reflections_l1047_104715

theorem max_reflections (P Q R M : Type) (angle : ℝ) :
  0 < angle ∧ angle ≤ 30 ∧ (∃ n : ℕ, 10 * n = angle) →
  ∃ n : ℕ, n ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_max_reflections_l1047_104715


namespace NUMINAMATH_GPT_sum_of_palindromes_l1047_104790

/-- Definition of a three-digit palindrome -/
def is_palindrome (n : ℕ) : Prop :=
  n / 100 = n % 10

theorem sum_of_palindromes (a b : ℕ) (h1 : is_palindrome a)
  (h2 : is_palindrome b) (h3 : a * b = 334491) (h4 : 100 ≤ a)
  (h5 : a < 1000) (h6 : 100 ≤ b) (h7 : b < 1000) : a + b = 1324 :=
sorry

end NUMINAMATH_GPT_sum_of_palindromes_l1047_104790


namespace NUMINAMATH_GPT_car_Z_probability_l1047_104722

theorem car_Z_probability :
  let P_X := 1/6
  let P_Y := 1/10
  let P_XYZ := 0.39166666666666666
  ∃ P_Z : ℝ, P_X + P_Y + P_Z = P_XYZ ∧ P_Z = 0.125 :=
by
  sorry

end NUMINAMATH_GPT_car_Z_probability_l1047_104722


namespace NUMINAMATH_GPT_g_at_6_is_zero_l1047_104755

def g (x : ℝ) : ℝ := 3*x^4 - 18*x^3 + 31*x^2 - 29*x - 72

theorem g_at_6_is_zero : g 6 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_g_at_6_is_zero_l1047_104755


namespace NUMINAMATH_GPT_larger_tablet_diagonal_length_l1047_104779

theorem larger_tablet_diagonal_length :
  ∀ (d : ℝ), (d^2 / 2 = 25 / 2 + 5.5) → d = 6 :=
by
  intro d
  sorry

end NUMINAMATH_GPT_larger_tablet_diagonal_length_l1047_104779


namespace NUMINAMATH_GPT_part1_part2_l1047_104718

namespace ClothingFactory

variables {x y m : ℝ} -- defining variables

-- The conditions
def condition1 : Prop := x + 2 * y = 5
def condition2 : Prop := 3 * x + y = 7
def condition3 : Prop := 1.8 * (100 - m) + 1.6 * m ≤ 168

-- Theorems to Prove
theorem part1 (h1 : x + 2 * y = 5) (h2 : 3 * x + y = 7) : 
  x = 1.8 ∧ y = 1.6 := 
sorry

theorem part2 (h1 : x = 1.8) (h2 : y = 1.6) (h3 : 1.8 * (100 - m) + 1.6 * m ≤ 168) : 
  m ≥ 60 := 
sorry

end ClothingFactory

end NUMINAMATH_GPT_part1_part2_l1047_104718


namespace NUMINAMATH_GPT_original_cost_of_each_magazine_l1047_104753

-- Definitions and conditions
def magazine_cost (C : ℝ) : Prop :=
  let total_magazines := 10
  let sell_price := 3.50
  let gain := 5
  let total_revenue := total_magazines * sell_price
  let total_cost := total_revenue - gain
  C = total_cost / total_magazines

-- Goal to prove
theorem original_cost_of_each_magazine : ∃ C : ℝ, magazine_cost C ∧ C = 3 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_of_each_magazine_l1047_104753


namespace NUMINAMATH_GPT_crank_slider_motion_l1047_104737

def omega : ℝ := 10
def OA : ℝ := 90
def AB : ℝ := 90
def AM : ℝ := 60
def t : ℝ := sorry -- t is a variable, no specific value required

theorem crank_slider_motion :
  (∀ t : ℝ, ((90 * Real.cos (10 * t)), (90 * Real.sin (10 * t) + 60)) = (x, y)) ∧
  (∀ t : ℝ, ((-900 * Real.sin (10 * t)), (900 * Real.cos (10 * t))) = (vx, vy)) :=
sorry

end NUMINAMATH_GPT_crank_slider_motion_l1047_104737


namespace NUMINAMATH_GPT_calculate_f_5_l1047_104721

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

theorem calculate_f_5 : f 5 = 4485 := 
by {
  -- The proof of the theorem will go here, using the Horner's method as described.
  sorry
}

end NUMINAMATH_GPT_calculate_f_5_l1047_104721


namespace NUMINAMATH_GPT_mary_talking_ratio_l1047_104729

theorem mary_talking_ratio:
  let mac_download_time := 10
  let windows_download_time := 3 * mac_download_time
  let audio_glitch_time := 2 * 4
  let video_glitch_time := 6
  let total_glitch_time := audio_glitch_time + video_glitch_time
  let total_download_time := mac_download_time + windows_download_time
  let total_time := 82
  let talking_time := total_time - total_download_time
  let talking_time_without_glitch := talking_time - total_glitch_time
  talking_time_without_glitch / total_glitch_time = 2 :=
by
  sorry

end NUMINAMATH_GPT_mary_talking_ratio_l1047_104729


namespace NUMINAMATH_GPT_overtaking_time_l1047_104733

theorem overtaking_time (t_a t_b t_k : ℝ) (t_b_start : t_b = t_a - 5) 
                       (overtake_eq1 : 40 * t_b = 30 * t_a)
                       (overtake_eq2 : 60 * (t_a - 10) = 30 * t_a) :
                       t_b = 15 :=
by
  sorry

end NUMINAMATH_GPT_overtaking_time_l1047_104733


namespace NUMINAMATH_GPT_math_problem_l1047_104771

noncomputable def is_solution (x : ℝ) : Prop :=
  1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 12

theorem math_problem :
  (is_solution ((7 + Real.sqrt 153) / 2)) ∧ (is_solution ((7 - Real.sqrt 153) / 2)) := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1047_104771


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1047_104745

theorem arithmetic_sequence_sum {a_n : ℕ → ℤ} (d : ℤ) (S : ℕ → ℤ) 
  (h_seq : ∀ n, a_n (n + 1) = a_n n + d)
  (h_sum : ∀ n, S n = (n * (2 * a_n 1 + (n - 1) * d)) / 2)
  (h_condition : a_n 1 = 2 * a_n 3 - 3) : 
  S 9 = 27 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1047_104745


namespace NUMINAMATH_GPT_pollywog_maturation_rate_l1047_104774

theorem pollywog_maturation_rate :
  ∀ (initial_pollywogs : ℕ) (melvin_rate : ℕ) (total_days : ℕ) (melvin_days : ℕ) (remaining_pollywogs : ℕ),
  initial_pollywogs = 2400 →
  melvin_rate = 10 →
  total_days = 44 →
  melvin_days = 20 →
  remaining_pollywogs = initial_pollywogs - (melvin_rate * melvin_days) →
  (total_days * (remaining_pollywogs / (total_days - melvin_days))) = remaining_pollywogs →
  (remaining_pollywogs / (total_days - melvin_days)) = 50 := 
by
  intros initial_pollywogs melvin_rate total_days melvin_days remaining_pollywogs
  intros h_initial h_melvin h_total h_melvin_days h_remaining h_eq
  sorry

end NUMINAMATH_GPT_pollywog_maturation_rate_l1047_104774


namespace NUMINAMATH_GPT_book_arrangements_l1047_104792

theorem book_arrangements (total_books : ℕ) (at_least_in_library : ℕ) (at_least_checked_out : ℕ) 
  (h_total : total_books = 10) (h_at_least_in : at_least_in_library = 2) 
  (h_at_least_out : at_least_checked_out = 3) : 
  ∃ arrangements : ℕ, arrangements = 6 :=
by
  sorry

end NUMINAMATH_GPT_book_arrangements_l1047_104792


namespace NUMINAMATH_GPT_convert_to_rectangular_and_find_line_l1047_104740

noncomputable def circle_eq1 (x y : ℝ) : Prop := x^2 + y^2 = 4 * x
noncomputable def circle_eq2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * y = 0
noncomputable def line_eq (x y : ℝ) : Prop := y = -x

theorem convert_to_rectangular_and_find_line :
  (∀ x y : ℝ, circle_eq1 x y → x^2 + y^2 = 4 * x) →
  (∀ x y : ℝ, circle_eq2 x y → x^2 + y^2 + 4 * y = 0) →
  (∀ x y : ℝ, circle_eq1 x y ∧ circle_eq2 x y → line_eq x y)
:=
sorry

end NUMINAMATH_GPT_convert_to_rectangular_and_find_line_l1047_104740


namespace NUMINAMATH_GPT_ticket_cost_difference_l1047_104727

theorem ticket_cost_difference (num_prebuy : ℕ) (price_prebuy : ℕ) (num_gate : ℕ) (price_gate : ℕ)
  (h_prebuy : num_prebuy = 20) (h_price_prebuy : price_prebuy = 155)
  (h_gate : num_gate = 30) (h_price_gate : price_gate = 200) :
  num_gate * price_gate - num_prebuy * price_prebuy = 2900 :=
by
  sorry

end NUMINAMATH_GPT_ticket_cost_difference_l1047_104727


namespace NUMINAMATH_GPT_smallest_x_value_min_smallest_x_value_l1047_104712

noncomputable def smallest_x_not_defined : ℝ := ( 47 - (Real.sqrt 2041) ) / 12

theorem smallest_x_value :
  ∀ x : ℝ, (6 * x^2 - 47 * x + 7 = 0) → x = smallest_x_not_defined ∨ (x = (47 + (Real.sqrt 2041)) / 12) :=
sorry

theorem min_smallest_x_value :
  smallest_x_not_defined < (47 + (Real.sqrt 2041)) / 12 :=
sorry

end NUMINAMATH_GPT_smallest_x_value_min_smallest_x_value_l1047_104712


namespace NUMINAMATH_GPT_perimeter_of_rectangle_l1047_104716

theorem perimeter_of_rectangle (s : ℝ) (h1 : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_rectangle_l1047_104716


namespace NUMINAMATH_GPT_infinite_series_sum_l1047_104785

theorem infinite_series_sum :
  ∑' n : ℕ, (n + 1) * (1 / 1950)^n = 3802500 / 3802601 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l1047_104785


namespace NUMINAMATH_GPT_sum_of_squares_l1047_104747

theorem sum_of_squares (a b : ℝ) (h1 : (a + b)^2 = 11) (h2 : (a - b)^2 = 5) : a^2 + b^2 = 8 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_l1047_104747


namespace NUMINAMATH_GPT_xy_gt_xz_l1047_104738

variable {R : Type*} [LinearOrderedField R]
variables (x y z : R)

theorem xy_gt_xz (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z :=
by
  sorry

end NUMINAMATH_GPT_xy_gt_xz_l1047_104738


namespace NUMINAMATH_GPT_partition_nat_set_l1047_104752

theorem partition_nat_set :
  ∃ (P : ℕ → ℕ), (∀ (n : ℕ), P n < 100) ∧ (∀ (a b c : ℕ), a + 99 * b = c → (P a = P b ∨ P b = P c ∨ P c = P a)) :=
sorry

end NUMINAMATH_GPT_partition_nat_set_l1047_104752


namespace NUMINAMATH_GPT_water_tank_full_capacity_l1047_104754

-- Define the conditions
variable {C x : ℝ}
variable (h1 : x / C = 1 / 3)
variable (h2 : (x + 6) / C = 1 / 2)

-- Prove that C = 36
theorem water_tank_full_capacity : C = 36 :=
by
  sorry

end NUMINAMATH_GPT_water_tank_full_capacity_l1047_104754


namespace NUMINAMATH_GPT_sum_first_19_terms_l1047_104783

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)
variable (a₀ a₃ a₁₇ a₁₀ : ℝ)

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ a₀ d, ∀ n, a n = a₀ + n * d

noncomputable def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_first_19_terms (h1 : is_arithmetic_sequence a)
                          (h2 : a 3 + a 17 = 10)
                          (h3 : sum_first_n_terms S a) :
  S 19 = 95 :=
sorry

end NUMINAMATH_GPT_sum_first_19_terms_l1047_104783


namespace NUMINAMATH_GPT_number_of_outfits_l1047_104787

def shirts : ℕ := 5
def hats : ℕ := 3

theorem number_of_outfits : shirts * hats = 15 :=
by 
  -- This part intentionally left blank since no proof required.
  sorry

end NUMINAMATH_GPT_number_of_outfits_l1047_104787


namespace NUMINAMATH_GPT_parabola_focus_distance_l1047_104759

theorem parabola_focus_distance (p : ℝ) : 
  (∀ (y : ℝ), y^2 = 2 * p * 4 → abs (4 + p / 2) = 5) → 
  p = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1047_104759


namespace NUMINAMATH_GPT_max_volume_small_cube_l1047_104768

theorem max_volume_small_cube (a : ℝ) (h : a = 2) : (a^3 = 8) := by
  sorry

end NUMINAMATH_GPT_max_volume_small_cube_l1047_104768


namespace NUMINAMATH_GPT_january_1_is_monday_l1047_104700

theorem january_1_is_monday
  (days_in_january : ℕ)
  (mondays_in_january : ℕ)
  (thursdays_in_january : ℕ) :
  days_in_january = 31 ∧ mondays_in_january = 5 ∧ thursdays_in_january = 5 → 
  ∃ d : ℕ, d = 1 ∧ (d % 7 = 1) :=
by
  sorry

end NUMINAMATH_GPT_january_1_is_monday_l1047_104700


namespace NUMINAMATH_GPT_current_age_l1047_104746

theorem current_age (A B S Y : ℕ) 
  (h1: Y = 4) 
  (h2: S = 2 * Y) 
  (h3: B = S + 3) 
  (h4: A + 10 = 2 * (B + 10))
  (h5: A + 10 = 3 * (S + 10))
  (h6: A + 10 = 4 * (Y + 10)) 
  (h7: (A + 10) + (B + 10) + (S + 10) + (Y + 10) = 88) : 
  A = 46 :=
sorry

end NUMINAMATH_GPT_current_age_l1047_104746


namespace NUMINAMATH_GPT_cos_periodicity_even_function_property_l1047_104744

theorem cos_periodicity_even_function_property (n : ℤ) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) (h_range : -180 ≤ n ∧ n ≤ 180) : n = 43 :=
by
  sorry

end NUMINAMATH_GPT_cos_periodicity_even_function_property_l1047_104744


namespace NUMINAMATH_GPT_time_in_3467_hours_l1047_104704

-- Define the current time, the number of hours, and the modulus
def current_time : ℕ := 2
def hours_from_now : ℕ := 3467
def clock_modulus : ℕ := 12

-- Define the function to calculate the future time on a 12-hour clock
def future_time (current_time : ℕ) (hours_from_now : ℕ) (modulus : ℕ) : ℕ := 
  (current_time + hours_from_now) % modulus

-- Theorem statement
theorem time_in_3467_hours :
  future_time current_time hours_from_now clock_modulus = 9 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_time_in_3467_hours_l1047_104704


namespace NUMINAMATH_GPT_missing_side_length_of_pan_l1047_104705

-- Definition of the given problem's conditions
def pan_side_length := 29
def total_fudge_pieces := 522
def fudge_piece_area := 1

-- Proof statement in Lean 4
theorem missing_side_length_of_pan : 
  (total_fudge_pieces * fudge_piece_area) = (pan_side_length * 18) :=
by
  sorry

end NUMINAMATH_GPT_missing_side_length_of_pan_l1047_104705


namespace NUMINAMATH_GPT_middle_digit_is_zero_l1047_104782

noncomputable def N_in_base8 (a b c : ℕ) : ℕ := 512 * a + 64 * b + 8 * c
noncomputable def N_in_base10 (a b c : ℕ) : ℕ := 100 * b + 10 * c + a

theorem middle_digit_is_zero (a b c : ℕ) (h : N_in_base8 a b c = N_in_base10 a b c) :
  b = 0 :=
by 
  sorry

end NUMINAMATH_GPT_middle_digit_is_zero_l1047_104782


namespace NUMINAMATH_GPT_intersection_on_y_axis_l1047_104775

theorem intersection_on_y_axis (k : ℝ) (x y : ℝ) :
  (2 * x + 3 * y - k = 0) →
  (x - k * y + 12 = 0) →
  (x = 0) →
  k = 6 ∨ k = -6 :=
by
  sorry

end NUMINAMATH_GPT_intersection_on_y_axis_l1047_104775


namespace NUMINAMATH_GPT_part1_part2_l1047_104789

-- Part (1)
theorem part1 (x y : ℚ) 
  (h1 : 2022 * x + 2020 * y = 2021)
  (h2 : 2023 * x + 2021 * y = 2022) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

-- Part (2)
theorem part2 (x y a b : ℚ)
  (ha : a ≠ b) 
  (h1 : (a + 1) * x + (a - 1) * y = a)
  (h2 : (b + 1) * x + (b - 1) * y = b) :
  x = 1/2 ∧ y = 1/2 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_part1_part2_l1047_104789


namespace NUMINAMATH_GPT_postal_code_permutations_l1047_104773

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def multiplicity_permutations (n : Nat) (repetitions : List Nat) : Nat :=
  factorial n / List.foldl (λ acc k => acc * factorial k) 1 repetitions

theorem postal_code_permutations : multiplicity_permutations 4 [2, 1, 1] = 12 :=
by
  unfold multiplicity_permutations
  unfold factorial
  sorry

end NUMINAMATH_GPT_postal_code_permutations_l1047_104773


namespace NUMINAMATH_GPT_abs_inequality_solution_l1047_104751

theorem abs_inequality_solution (x : ℝ) : 2 * |x - 1| - 1 < 0 ↔ (1 / 2 < x ∧ x < 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1047_104751


namespace NUMINAMATH_GPT_blueberries_in_blue_box_l1047_104762

theorem blueberries_in_blue_box (B S : ℕ) (h1 : S - B = 12) (h2 : S + B = 76) : B = 32 :=
sorry

end NUMINAMATH_GPT_blueberries_in_blue_box_l1047_104762


namespace NUMINAMATH_GPT_solve_inequalities_l1047_104776

theorem solve_inequalities (x : ℝ) :
  (1 / x < 1 ∧ |4 * x - 1| > 2) →
  (x < -1/4 ∨ x > 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l1047_104776


namespace NUMINAMATH_GPT_height_of_scale_model_eq_29_l1047_104723

def empireStateBuildingHeight : ℕ := 1454

def scaleRatio : ℕ := 50

def scaleModelHeight (actualHeight : ℕ) (ratio : ℕ) : ℤ :=
  Int.ofNat actualHeight / ratio

theorem height_of_scale_model_eq_29 : scaleModelHeight empireStateBuildingHeight scaleRatio = 29 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_height_of_scale_model_eq_29_l1047_104723


namespace NUMINAMATH_GPT_union_intersection_l1047_104726

-- Define the sets A, B, and C
def A : Set ℕ := {1, 2, 6}
def B : Set ℕ := {2, 4}
def C : Set ℕ := {1, 2, 3, 4}

-- The theorem stating that (A ∪ B) ∩ C = {1, 2, 4}
theorem union_intersection : (A ∪ B) ∩ C = {1, 2, 4} := sorry

end NUMINAMATH_GPT_union_intersection_l1047_104726


namespace NUMINAMATH_GPT_value_of_other_bills_l1047_104735

theorem value_of_other_bills (total_payment : ℕ) (num_fifty_dollar_bills : ℕ) (value_fifty_dollar_bill : ℕ) (num_other_bills : ℕ) 
  (total_fifty_dollars : ℕ) (remaining_payment : ℕ) (value_of_each_other_bill : ℕ) :
  total_payment = 170 →
  num_fifty_dollar_bills = 3 →
  value_fifty_dollar_bill = 50 →
  num_other_bills = 2 →
  total_fifty_dollars = num_fifty_dollar_bills * value_fifty_dollar_bill →
  remaining_payment = total_payment - total_fifty_dollars →
  value_of_each_other_bill = remaining_payment / num_other_bills →
  value_of_each_other_bill = 10 :=
by
  intros t_total_payment t_num_fifty_dollar_bills t_value_fifty_dollar_bill t_num_other_bills t_total_fifty_dollars t_remaining_payment t_value_of_each_other_bill
  sorry

end NUMINAMATH_GPT_value_of_other_bills_l1047_104735


namespace NUMINAMATH_GPT_goose_eggs_count_l1047_104791

theorem goose_eggs_count (E : ℕ)
    (hatch_fraction : ℚ := 1/3)
    (first_month_survival : ℚ := 4/5)
    (first_year_survival : ℚ := 2/5)
    (no_migration : ℚ := 3/4)
    (predator_survival : ℚ := 2/3)
    (final_survivors : ℕ := 140) :
    (predator_survival * no_migration * first_year_survival * first_month_survival * hatch_fraction * E : ℚ) = final_survivors → E = 1050 := by
  sorry

end NUMINAMATH_GPT_goose_eggs_count_l1047_104791


namespace NUMINAMATH_GPT_no_real_solution_l1047_104781

theorem no_real_solution (x : ℝ) : 
  (¬ (x^4 + 3*x^3)/(x^2 + 3*x + 1) + x = -7) :=
sorry

end NUMINAMATH_GPT_no_real_solution_l1047_104781


namespace NUMINAMATH_GPT_cosine_of_arcsine_l1047_104797

theorem cosine_of_arcsine (h : -1 ≤ (8 : ℝ) / 17 ∧ (8 : ℝ) / 17 ≤ 1) : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
sorry

end NUMINAMATH_GPT_cosine_of_arcsine_l1047_104797


namespace NUMINAMATH_GPT_distance_between_vertices_of_hyperbola_l1047_104772

theorem distance_between_vertices_of_hyperbola :
  ∀ (x y : ℝ), 16 * x^2 - 32 * x - y^2 + 10 * y + 19 = 0 → 
  2 * Real.sqrt (7 / 4) = Real.sqrt 7 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_distance_between_vertices_of_hyperbola_l1047_104772


namespace NUMINAMATH_GPT_max_temperature_when_80_l1047_104731

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10 * t + 60

-- State the theorem
theorem max_temperature_when_80 : ∃ t : ℝ, temperature t = 80 ∧ t = 5 + Real.sqrt 5 := 
by {
  -- Theorem proof is skipped with sorry
  sorry
}

end NUMINAMATH_GPT_max_temperature_when_80_l1047_104731


namespace NUMINAMATH_GPT_inequalities_always_true_l1047_104784

variables {x y a b : Real}

/-- All given conditions -/
def conditions (x y a b : Real) :=
  x < a ∧ y < b ∧ x < 0 ∧ y < 0 ∧ a > 0 ∧ b > 0

theorem inequalities_always_true {x y a b : Real} (h : conditions x y a b) :
  (x + y < a + b) ∧ 
  (x - y < a - b) ∧ 
  (x * y < a * b) ∧ 
  ((x + y) / (x - y) < (a + b) / (a - b)) :=
sorry

end NUMINAMATH_GPT_inequalities_always_true_l1047_104784


namespace NUMINAMATH_GPT_sum_of_interior_angles_of_polygon_l1047_104760

theorem sum_of_interior_angles_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 36) :
  ∃ interior_sum : ℝ, interior_sum = 1440 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_of_polygon_l1047_104760


namespace NUMINAMATH_GPT_range_of_a_l1047_104786

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 + 2*x else -(x^2 + 2*(-x))

theorem range_of_a (a : ℝ) (h : f (3 - a^2) > f (2 * a)) : -3 < a ∧ a < 1 := sorry

end NUMINAMATH_GPT_range_of_a_l1047_104786


namespace NUMINAMATH_GPT_Steven_has_16_apples_l1047_104765

variable (Jake_Peaches Steven_Peaches Jake_Apples Steven_Apples : ℕ)

theorem Steven_has_16_apples
  (h1 : Jake_Peaches = Steven_Peaches - 6)
  (h2 : Steven_Peaches = 17)
  (h3 : Steven_Peaches = Steven_Apples + 1)
  (h4 : Jake_Apples = Steven_Apples + 8) :
  Steven_Apples = 16 := by
  sorry

end NUMINAMATH_GPT_Steven_has_16_apples_l1047_104765


namespace NUMINAMATH_GPT_compute_cubic_sum_l1047_104741

theorem compute_cubic_sum (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : x * y + x ^ 2 + y ^ 2 = 17) : x ^ 3 + y ^ 3 = 52 :=
sorry

end NUMINAMATH_GPT_compute_cubic_sum_l1047_104741


namespace NUMINAMATH_GPT_partial_fraction_sum_inverse_l1047_104702

theorem partial_fraction_sum_inverse (p q r A B C : ℝ)
  (hroots : (∀ s, s^3 - 20 * s^2 + 96 * s - 91 = (s - p) * (s - q) * (s - r)))
  (hA : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20 * s^2 + 96 * s - 91) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 225 :=
sorry

end NUMINAMATH_GPT_partial_fraction_sum_inverse_l1047_104702


namespace NUMINAMATH_GPT_five_digit_palindromes_count_l1047_104713

theorem five_digit_palindromes_count : 
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) → 
  900 = 9 * 10 * 10 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_five_digit_palindromes_count_l1047_104713
