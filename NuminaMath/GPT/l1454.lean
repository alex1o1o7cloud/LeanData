import Mathlib

namespace NUMINAMATH_GPT_infinite_integer_triples_solution_l1454_145474

theorem infinite_integer_triples_solution (a b c : ℤ) : 
  ∃ (a b c : ℤ), ∀ n : ℤ, a^2 + b^2 = c^2 + 3 :=
sorry

end NUMINAMATH_GPT_infinite_integer_triples_solution_l1454_145474


namespace NUMINAMATH_GPT_proposition_1_proposition_2_proposition_3_proposition_4_l1454_145415

theorem proposition_1 : ∀ x : ℝ, 2 * x^2 - 3 * x + 4 > 0 := sorry

theorem proposition_2 : ¬ (∀ x ∈ ({-1, 0, 1} : Set ℤ), 2 * x + 1 > 0) := sorry

theorem proposition_3 : ∃ x : ℕ, x^2 ≤ x := sorry

theorem proposition_4 : ∃ x : ℕ, x ∣ 29 := sorry

end NUMINAMATH_GPT_proposition_1_proposition_2_proposition_3_proposition_4_l1454_145415


namespace NUMINAMATH_GPT_function_inverse_overlap_form_l1454_145476

theorem function_inverse_overlap_form (a b c d : ℝ) (h : ¬(a = 0 ∧ c = 0)) : 
  (∀ x, (c * x + d) * (dx - b) = (a * x + b) * (-c * x + a)) → 
  (∃ f : ℝ → ℝ, (∀ x, f x = x ∨ f x = (a * x + b) / (c * x - a))) :=
by 
  sorry

end NUMINAMATH_GPT_function_inverse_overlap_form_l1454_145476


namespace NUMINAMATH_GPT_smallest_nonneg_integer_divisible_by_4_l1454_145472

theorem smallest_nonneg_integer_divisible_by_4 :
  ∃ n : ℕ, (7 * (n - 3)^5 - n^2 + 16 * n - 30) % 4 = 0 ∧ ∀ m : ℕ, m < n -> (7 * (m - 3)^5 - m^2 + 16 * m - 30) % 4 ≠ 0 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_smallest_nonneg_integer_divisible_by_4_l1454_145472


namespace NUMINAMATH_GPT_ratio_of_P_to_Q_l1454_145403

theorem ratio_of_P_to_Q (p q r s : ℕ) (h1 : p + q + r + s = 1000)
    (h2 : s = 4 * r) (h3 : q = r) (h4 : s - p = 250) : 
    p = 2 * q :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_ratio_of_P_to_Q_l1454_145403


namespace NUMINAMATH_GPT_total_time_for_journey_l1454_145451

theorem total_time_for_journey (x : ℝ) : 
  let time_first_part := x / 50
  let time_second_part := 3 * x / 80
  time_first_part + time_second_part = 23 * x / 400 :=
by 
  sorry

end NUMINAMATH_GPT_total_time_for_journey_l1454_145451


namespace NUMINAMATH_GPT_dice_probability_five_or_six_l1454_145425

theorem dice_probability_five_or_six :
  let outcomes := 36
  let favorable := 18
  let probability := favorable / outcomes
  probability = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_five_or_six_l1454_145425


namespace NUMINAMATH_GPT_notebooks_multiple_of_3_l1454_145452

theorem notebooks_multiple_of_3 (N : ℕ) (h1 : ∃ k : ℕ, N = 3 * k) :
  ∃ k : ℕ, N = 3 * k :=
by
  sorry

end NUMINAMATH_GPT_notebooks_multiple_of_3_l1454_145452


namespace NUMINAMATH_GPT_line_through_origin_and_intersection_eq_x_y_l1454_145450

theorem line_through_origin_and_intersection_eq_x_y :
  ∀ (x y : ℝ), (x - 2 * y + 2 = 0) ∧ (2 * x - y - 2 = 0) →
  ∃ m b : ℝ, m = 1 ∧ b = 0 ∧ (y = m * x + b) :=
by
  sorry

end NUMINAMATH_GPT_line_through_origin_and_intersection_eq_x_y_l1454_145450


namespace NUMINAMATH_GPT_B_subscription_difference_l1454_145404

noncomputable def subscription_difference (A B C P : ℕ) (delta : ℕ) (comb_sub: A + B + C = 50000) (c_profit: 8400 = 35000 * C / 50000) :=
  B - C

theorem B_subscription_difference (A B C : ℕ) (z: ℕ) 
  (h1 : A + B + C = 50000) 
  (h2 : A = B + 4000) 
  (h3 : (B - C) = z)
  (h4 :  8400 = 35000 * C / 50000):
  B - C = 10000 :=
by {
  sorry
}

end NUMINAMATH_GPT_B_subscription_difference_l1454_145404


namespace NUMINAMATH_GPT_sneaker_final_price_l1454_145473

-- Definitions of the conditions
def original_price : ℝ := 120
def coupon_value : ℝ := 10
def discount_percent : ℝ := 0.1

-- The price after the coupon is applied
def price_after_coupon := original_price - coupon_value

-- The membership discount amount
def membership_discount := price_after_coupon * discount_percent

-- The final price the man will pay
def final_price := price_after_coupon - membership_discount

theorem sneaker_final_price : final_price = 99 := by
  sorry

end NUMINAMATH_GPT_sneaker_final_price_l1454_145473


namespace NUMINAMATH_GPT_range_of_a_l1454_145484

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 - a * x - 3)

def monotonic_increasing (a : ℝ) : Prop :=
  ∀ x > 1, 2 * x - a > 0

def positive_argument (a : ℝ) : Prop :=
  ∀ x > 1, x^2 - a * x - 3 > 0

theorem range_of_a :
  {a : ℝ | monotonic_increasing a ∧ positive_argument a} = {a : ℝ | a ≤ -2} :=
sorry

end NUMINAMATH_GPT_range_of_a_l1454_145484


namespace NUMINAMATH_GPT_tank_saltwater_solution_l1454_145457

theorem tank_saltwater_solution (x : ℝ) :
  let water1 := 0.75 * x
  let water1_evaporated := (1/3) * water1
  let water2 := water1 - water1_evaporated
  let salt2 := 0.25 * x
  let water3 := water2 + 12
  let salt3 := salt2 + 24
  let step2_eq := (salt3 / (water3 + 24)) = 0.4
  let water4 := water3 - (1/4) * water3
  let salt4 := salt3
  let water5 := water4 + 15
  let salt5 := salt4 + 30
  let step4_eq := (salt5 / (water5 + 30)) = 0.5
  step2_eq ∧ step4_eq → x = 192 :=
by
  sorry

end NUMINAMATH_GPT_tank_saltwater_solution_l1454_145457


namespace NUMINAMATH_GPT_correlation_implies_slope_positive_l1454_145407

-- Definition of the regression line
def regression_line (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Given conditions
variables (x y : ℝ)
variables (b a r : ℝ)

-- The statement of the proof problem
theorem correlation_implies_slope_positive (h1 : r > 0) (h2 : regression_line x y b a) : b > 0 :=
sorry

end NUMINAMATH_GPT_correlation_implies_slope_positive_l1454_145407


namespace NUMINAMATH_GPT_henry_has_more_games_l1454_145479

-- Define the conditions and initial states
def initial_games_henry : ℕ := 33
def given_games_neil : ℕ := 5
def initial_games_neil : ℕ := 2

-- Define the number of games Henry and Neil have now
def games_henry_now : ℕ := initial_games_henry - given_games_neil
def games_neil_now : ℕ := initial_games_neil + given_games_neil

-- State the theorem to be proven
theorem henry_has_more_games : games_henry_now / games_neil_now = 4 :=
by
  sorry

end NUMINAMATH_GPT_henry_has_more_games_l1454_145479


namespace NUMINAMATH_GPT_tickets_spent_on_hat_l1454_145402

def tickets_won_whack_a_mole := 32
def tickets_won_skee_ball := 25
def tickets_left := 50
def total_tickets := tickets_won_whack_a_mole + tickets_won_skee_ball

theorem tickets_spent_on_hat : 
  total_tickets - tickets_left = 7 :=
by
  sorry

end NUMINAMATH_GPT_tickets_spent_on_hat_l1454_145402


namespace NUMINAMATH_GPT_g_g_g_g_2_eq_1406_l1454_145431

def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 5 * x + 1

theorem g_g_g_g_2_eq_1406 : g (g (g (g 2))) = 1406 := by
  sorry

end NUMINAMATH_GPT_g_g_g_g_2_eq_1406_l1454_145431


namespace NUMINAMATH_GPT_find_numbers_l1454_145424

theorem find_numbers (x y : ℕ) :
  x + y = 1244 →
  10 * x + 3 = (y - 2) / 10 →
  x = 12 ∧ y = 1232 :=
by
  intro h_sum h_trans
  -- We'll use sorry here to state that the proof is omitted.
  sorry

end NUMINAMATH_GPT_find_numbers_l1454_145424


namespace NUMINAMATH_GPT_number_of_zeros_of_f_l1454_145432

noncomputable def f : ℝ → ℝ
| x => if x >= 0 then x^3 - 3*x + 1 else x^2 - 2*x - 4

theorem number_of_zeros_of_f : ∃ z, z = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_l1454_145432


namespace NUMINAMATH_GPT_inequality_holds_for_interval_l1454_145492

theorem inequality_holds_for_interval (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 5 → x^2 - 2 * (a - 2) * x + a < 0) → a ≥ 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_inequality_holds_for_interval_l1454_145492


namespace NUMINAMATH_GPT_ihsan_children_l1454_145448

theorem ihsan_children :
  ∃ n : ℕ, (n + n^2 + n^3 + n^4 = 2800) ∧ (n = 7) :=
sorry

end NUMINAMATH_GPT_ihsan_children_l1454_145448


namespace NUMINAMATH_GPT_x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one_l1454_145489

theorem x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one {x : ℝ} :
  (x > 1 → |x| > 1) ∧ (¬(|x| > 1 → x > 1)) :=
by
  sorry

end NUMINAMATH_GPT_x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one_l1454_145489


namespace NUMINAMATH_GPT_first_variety_cost_l1454_145436

noncomputable def cost_of_second_variety : ℝ := 8.75
noncomputable def ratio_of_first_variety : ℚ := 5 / 6
noncomputable def ratio_of_second_variety : ℚ := 1 - ratio_of_first_variety
noncomputable def cost_of_mixture : ℝ := 7.50

theorem first_variety_cost :
  ∃ x : ℝ, x * (ratio_of_first_variety : ℝ) + cost_of_second_variety * (ratio_of_second_variety : ℝ) = cost_of_mixture * (ratio_of_first_variety + ratio_of_second_variety : ℝ) 
    ∧ x = 7.25 :=
sorry

end NUMINAMATH_GPT_first_variety_cost_l1454_145436


namespace NUMINAMATH_GPT_simplify_fraction_l1454_145456

theorem simplify_fraction : (270 / 18) * (7 / 140) * (9 / 4) = 27 / 16 :=
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1454_145456


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1454_145433

noncomputable def f (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem necessary_and_sufficient_condition
  {a b c : ℝ}
  (ha_pos : a > 0) :
  ( (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, f a b c x = y } → ∃! x : ℝ, f a b c x = y) ∧ 
    (∀ y : ℝ, y ∈ { y : ℝ | ∃ x : ℝ, y = f a b c x } → ∃! x : ℝ, f a b c x = y)
  ) ↔
  f a b c (f a b c (-b / (2 * a))) < 0 :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1454_145433


namespace NUMINAMATH_GPT_ratio_equivalence_to_minutes_l1454_145421

-- Define conditions and equivalence
theorem ratio_equivalence_to_minutes :
  ∀ (x : ℝ), (8 / 4 = 8 / x) → x = 4 / 60 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_ratio_equivalence_to_minutes_l1454_145421


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1454_145447

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Statement of the problem
theorem arithmetic_sequence_common_difference
  (h1 : a 2 + a 6 = 8)
  (h2 : a 3 + a 4 = 3)
  (h_arith : ∀ n, a (n+1) = a n + d) :
  d = 5 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1454_145447


namespace NUMINAMATH_GPT_boys_at_reunion_l1454_145468

theorem boys_at_reunion (n : ℕ) (H : n * (n - 1) / 2 = 45) : n = 10 :=
by sorry

end NUMINAMATH_GPT_boys_at_reunion_l1454_145468


namespace NUMINAMATH_GPT_gear_q_revolutions_per_minute_l1454_145429

noncomputable def gear_p_revolutions_per_minute : ℕ := 10

noncomputable def additional_revolutions : ℕ := 15

noncomputable def calculate_q_revolutions_per_minute
  (p_rev_per_min : ℕ) (additional_rev : ℕ) : ℕ :=
  2 * (p_rev_per_min / 2 + additional_rev)

theorem gear_q_revolutions_per_minute :
  calculate_q_revolutions_per_minute gear_p_revolutions_per_minute additional_revolutions = 40 :=
by
  sorry

end NUMINAMATH_GPT_gear_q_revolutions_per_minute_l1454_145429


namespace NUMINAMATH_GPT_depth_of_river_bank_l1454_145480

theorem depth_of_river_bank (top_width bottom_width area depth : ℝ) 
  (h₁ : top_width = 12)
  (h₂ : bottom_width = 8)
  (h₃ : area = 500)
  (h₄ : area = (1 / 2) * (top_width + bottom_width) * depth) :
  depth = 50 :=
sorry

end NUMINAMATH_GPT_depth_of_river_bank_l1454_145480


namespace NUMINAMATH_GPT_setA_times_setB_equals_desired_l1454_145416

def setA : Set ℝ := { x | abs (x - 1/2) < 1 }
def setB : Set ℝ := { x | 1/x ≥ 1 }
def setAB : Set ℝ := { x | (x ∈ setA ∪ setB) ∧ (x ∉ setA ∩ setB) }

theorem setA_times_setB_equals_desired :
  setAB = { x | (-1/2 < x ∧ x ≤ 0) ∨ (1 < x ∧ x < 3/2) } :=
by
  sorry

end NUMINAMATH_GPT_setA_times_setB_equals_desired_l1454_145416


namespace NUMINAMATH_GPT_dress_designs_count_l1454_145460

theorem dress_designs_count :
  let colors := 5
  let patterns := 4
  let sizes := 3
  colors * patterns * sizes = 60 :=
by
  let colors := 5
  let patterns := 4
  let sizes := 3
  have h : colors * patterns * sizes = 60 := by norm_num
  exact h

end NUMINAMATH_GPT_dress_designs_count_l1454_145460


namespace NUMINAMATH_GPT_transformed_ellipse_equation_l1454_145434

namespace EllipseTransformation

open Real

def original_ellipse (x y : ℝ) : Prop :=
  x^2 / 6 + y^2 = 1

def transformation (x' y' x y : ℝ) : Prop :=
  x' = 1 / 2 * x ∧ y' = 2 * y

theorem transformed_ellipse_equation (x y x' y' : ℝ) 
  (h : original_ellipse x y) (tr : transformation x' y' x y) :
  2 * x'^2 / 3 + y'^2 / 4 = 1 :=
by 
  sorry

end EllipseTransformation

end NUMINAMATH_GPT_transformed_ellipse_equation_l1454_145434


namespace NUMINAMATH_GPT_passes_through_1_1_l1454_145469

theorem passes_through_1_1 (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^ (x - 1))} :=
by
  -- proof not required
  sorry

end NUMINAMATH_GPT_passes_through_1_1_l1454_145469


namespace NUMINAMATH_GPT_minimum_surface_area_of_circumscribed_sphere_of_prism_l1454_145400

theorem minimum_surface_area_of_circumscribed_sphere_of_prism :
  ∃ S : ℝ, 
    (∀ h r, r^2 * h = 4 → r^2 + (h^2 / 4) = R → 4 * π * R^2 = S) ∧ 
    (∀ S', S' ≤ S) ∧ 
    S = 12 * π :=
sorry

end NUMINAMATH_GPT_minimum_surface_area_of_circumscribed_sphere_of_prism_l1454_145400


namespace NUMINAMATH_GPT_intersection_point_l1454_145405

theorem intersection_point :
  ∃ (x y : ℝ), (y = 2 * x) ∧ (x + y = 3) ∧ (x = 1) ∧ (y = 2) := 
by
  sorry

end NUMINAMATH_GPT_intersection_point_l1454_145405


namespace NUMINAMATH_GPT_chicken_nugget_ratio_l1454_145498

theorem chicken_nugget_ratio (k d a t : ℕ) (h1 : a = 20) (h2 : t = 100) (h3 : k + d + a = t) : (k + d) / a = 4 :=
by
  sorry

end NUMINAMATH_GPT_chicken_nugget_ratio_l1454_145498


namespace NUMINAMATH_GPT_sin_double_angle_given_cos_identity_l1454_145494

theorem sin_double_angle_given_cos_identity (α : ℝ) 
  (h : Real.cos (α + π / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_given_cos_identity_l1454_145494


namespace NUMINAMATH_GPT_tallest_is_first_l1454_145441

variable (P : Type) -- representing people
variable (line : Fin 9 → P) -- original line order (0 = shortest, 8 = tallest)
variable (Hoseok : P) -- Hoseok

-- Conditions
axiom tallest_person : line 8 = Hoseok

-- Theorem
theorem tallest_is_first :
  ∃ line' : Fin 9 → P, (∀ i : Fin 9, line' i = line (8 - i)) → line' 0 = Hoseok :=
  by
  sorry

end NUMINAMATH_GPT_tallest_is_first_l1454_145441


namespace NUMINAMATH_GPT_hexagon_area_l1454_145427

-- Definitions of the conditions
def DEF_perimeter := 42
def circumcircle_radius := 10
def area_of_hexagon_DE'F'D'E'F := 210

-- The theorem statement
theorem hexagon_area (DEF_perimeter : ℕ) (circumcircle_radius : ℕ) : Prop :=
  DEF_perimeter = 42 → circumcircle_radius = 10 → 
  area_of_hexagon_DE'F'D'E'F = 210

-- Example invocation of the theorem, proof omitted.
example : hexagon_area DEF_perimeter circumcircle_radius :=
by {
  sorry
}

end NUMINAMATH_GPT_hexagon_area_l1454_145427


namespace NUMINAMATH_GPT_train_length_is_correct_l1454_145426

noncomputable def train_speed_kmh : ℝ := 40
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
noncomputable def cross_time : ℝ := 25.2
noncomputable def train_length : ℝ := train_speed_ms * cross_time

theorem train_length_is_correct : train_length = 280.392 := by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l1454_145426


namespace NUMINAMATH_GPT_contractor_earnings_l1454_145481

def total_days : ℕ := 30
def work_rate : ℝ := 25
def fine_rate : ℝ := 7.5
def absent_days : ℕ := 8
def worked_days : ℕ := total_days - absent_days
def total_earned : ℝ := worked_days * work_rate
def total_fine : ℝ := absent_days * fine_rate
def total_received : ℝ := total_earned - total_fine

theorem contractor_earnings : total_received = 490 :=
by
  sorry

end NUMINAMATH_GPT_contractor_earnings_l1454_145481


namespace NUMINAMATH_GPT_forecast_interpretation_l1454_145428

-- Define the conditions
def condition (precipitation_probability : ℕ) : Prop :=
  precipitation_probability = 78

-- Define the interpretation question as a proof
theorem forecast_interpretation (precipitation_probability: ℕ) (cond : condition precipitation_probability) :
  precipitation_probability = 78 :=
by
  sorry

end NUMINAMATH_GPT_forecast_interpretation_l1454_145428


namespace NUMINAMATH_GPT_plant_supplier_money_left_correct_l1454_145446

noncomputable def plant_supplier_total_earnings : ℕ :=
  35 * 52 + 30 * 32 + 20 * 77 + 25 * 22 + 40 * 15

noncomputable def plant_supplier_total_expenses : ℕ :=
  3 * 65 + 2 * 45 + 280 + 150 + 100 + 125 + 225 + 550

noncomputable def plant_supplier_money_left : ℕ :=
  plant_supplier_total_earnings - plant_supplier_total_expenses

theorem plant_supplier_money_left_correct :
  plant_supplier_money_left = 3755 :=
by
  sorry

end NUMINAMATH_GPT_plant_supplier_money_left_correct_l1454_145446


namespace NUMINAMATH_GPT_vector_subtraction_l1454_145445

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-3, -4)

-- Statement we want to prove: 2a - b = (-1, 6)
theorem vector_subtraction : 2 • a - b = (-1, 6) := by
  sorry

end NUMINAMATH_GPT_vector_subtraction_l1454_145445


namespace NUMINAMATH_GPT_relationship_between_A_and_p_l1454_145475

variable {x y p : ℝ}

theorem relationship_between_A_and_p (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : x ≠ y * 2) (h4 : x ≠ p * y)
  (A : ℝ) (hA : A = (x^2 - 3 * y^2) / (3 * x^2 + y^2))
  (hEq : (p * x * y) / (x^2 - (2 + p) * x * y + 2 * p * y^2) - y / (x - 2 * y) = 1 / 2) :
  A = (9 * p^2 - 3) / (27 * p^2 + 1) := 
sorry

end NUMINAMATH_GPT_relationship_between_A_and_p_l1454_145475


namespace NUMINAMATH_GPT_john_back_squat_increase_l1454_145417

-- Definitions based on conditions
def back_squat_initial : ℝ := 200
def k : ℝ := 0.8
def j : ℝ := 0.9
def total_weight_moved : ℝ := 540

-- The variable representing the increase in back squat
variable (x : ℝ)

-- The Lean statement to prove
theorem john_back_squat_increase :
  3 * (j * k * (back_squat_initial + x)) = total_weight_moved → x = 50 := by
  sorry

end NUMINAMATH_GPT_john_back_squat_increase_l1454_145417


namespace NUMINAMATH_GPT_joan_missed_games_l1454_145423

-- Define the number of total games and games attended as constants
def total_games : ℕ := 864
def games_attended : ℕ := 395

-- The theorem statement: the number of missed games is equal to 469
theorem joan_missed_games : total_games - games_attended = 469 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_joan_missed_games_l1454_145423


namespace NUMINAMATH_GPT_maria_correct_answers_l1454_145454

theorem maria_correct_answers (x : ℕ) (n c d s : ℕ) (h1 : n = 30) (h2 : c = 20) (h3 : d = 5) (h4 : s = 325)
  (h5 : n = x + (n - x)) : 20 * x - 5 * (30 - x) = 325 → x = 19 :=
by 
  intros h_eq
  sorry

end NUMINAMATH_GPT_maria_correct_answers_l1454_145454


namespace NUMINAMATH_GPT_team_arrangements_l1454_145471

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

theorem team_arrangements :
  let num_players := 10
  let team_blocks := 4
  let cubs_players := 3
  let red_sox_players := 3
  let yankees_players := 2
  let dodgers_players := 2
  (factorial team_blocks) * (factorial cubs_players) * (factorial red_sox_players) * (factorial yankees_players) * (factorial dodgers_players) = 3456 := 
by
  -- Proof steps will be inserted here
  sorry

end NUMINAMATH_GPT_team_arrangements_l1454_145471


namespace NUMINAMATH_GPT_simplify_expression_l1454_145497

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b - 4) - 2 * b^2 = 9 * b^3 + 4 * b^2 - 12 * b :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1454_145497


namespace NUMINAMATH_GPT_final_balance_is_60_million_l1454_145493

-- Define the initial conditions
def initial_balance : ℕ := 100
def earnings_from_selling_players : ℕ := 2 * 10
def cost_of_buying_players : ℕ := 4 * 15

-- Define the final balance calculation and state the theorem
theorem final_balance_is_60_million : initial_balance + earnings_from_selling_players - cost_of_buying_players = 60 := by
  sorry

end NUMINAMATH_GPT_final_balance_is_60_million_l1454_145493


namespace NUMINAMATH_GPT_factors_of_2520_l1454_145462

theorem factors_of_2520 : (∃ (factors : Finset ℕ), factors.card = 48 ∧ ∀ d, d ∈ factors ↔ d > 0 ∧ 2520 % d = 0) :=
sorry

end NUMINAMATH_GPT_factors_of_2520_l1454_145462


namespace NUMINAMATH_GPT_possible_combinations_l1454_145485

noncomputable def dark_chocolate_price : ℝ := 5
noncomputable def milk_chocolate_price : ℝ := 4.50
noncomputable def white_chocolate_price : ℝ := 6
noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def leonardo_money : ℝ := 4 + 0.59

noncomputable def total_money := leonardo_money

noncomputable def dark_chocolate_with_tax := dark_chocolate_price * (1 + sales_tax_rate)
noncomputable def milk_chocolate_with_tax := milk_chocolate_price * (1 + sales_tax_rate)
noncomputable def white_chocolate_with_tax := white_chocolate_price * (1 + sales_tax_rate)

theorem possible_combinations :
  total_money = 4.59 ∧ (total_money >= 0 ∧ total_money < dark_chocolate_with_tax ∧ total_money < white_chocolate_with_tax ∧
  total_money ≥ milk_chocolate_with_tax ∧ milk_chocolate_with_tax = 4.82) :=
by
  sorry

end NUMINAMATH_GPT_possible_combinations_l1454_145485


namespace NUMINAMATH_GPT_david_still_has_l1454_145430

variable (P L S R : ℝ)

def initial_amount : ℝ := 1800
def post_spending_condition (S : ℝ) : ℝ := S - 800
def remaining_money (P S : ℝ) : ℝ := P - S

theorem david_still_has :
  ∀ (S : ℝ),
    initial_amount = P →
    post_spending_condition S = L →
    remaining_money P S = R →
    R = L →
    R = 500 :=
by
  intros S hP hL hR hCl
  sorry

end NUMINAMATH_GPT_david_still_has_l1454_145430


namespace NUMINAMATH_GPT_max_volume_of_sphere_in_cube_l1454_145419

theorem max_volume_of_sphere_in_cube (a : ℝ) (h : a = 1) : 
  ∃ V, V = π / 6 ∧ 
        ∀ (r : ℝ), r = a / 2 →
        V = (4 / 3) * π * r^3 :=
by
  sorry

end NUMINAMATH_GPT_max_volume_of_sphere_in_cube_l1454_145419


namespace NUMINAMATH_GPT_curve_symmetric_about_y_eq_x_l1454_145453

def curve_eq (x y : ℝ) : Prop := x * y * (x + y) = 1

theorem curve_symmetric_about_y_eq_x :
  ∀ (x y : ℝ), curve_eq x y ↔ curve_eq y x :=
by sorry

end NUMINAMATH_GPT_curve_symmetric_about_y_eq_x_l1454_145453


namespace NUMINAMATH_GPT_avg_age_when_youngest_born_l1454_145411

theorem avg_age_when_youngest_born
  (num_people : ℕ) (avg_age_now : ℝ) (youngest_age_now : ℝ) (sum_ages_others_then : ℝ) 
  (h1 : num_people = 7) 
  (h2 : avg_age_now = 30) 
  (h3 : youngest_age_now = 6) 
  (h4 : sum_ages_others_then = 150) :
  (sum_ages_others_then / num_people) = 21.43 :=
by
  sorry

end NUMINAMATH_GPT_avg_age_when_youngest_born_l1454_145411


namespace NUMINAMATH_GPT_time_for_2km_l1454_145499

def distance_over_time (t : ℕ) : ℝ := 
  sorry -- Function representing the distance walked over time

theorem time_for_2km : ∃ t : ℕ, distance_over_time t = 2 ∧ t = 105 :=
by
  sorry

end NUMINAMATH_GPT_time_for_2km_l1454_145499


namespace NUMINAMATH_GPT_find_b_l1454_145418

-- Definitions based on the given conditions
def good_point (a b : ℝ) (φ : ℝ) : Prop :=
  a + (b - a) * φ = 2.382 ∨ b - (b - a) * φ = 2.382

theorem find_b (b : ℝ) (φ : ℝ := 0.618) :
  good_point 2 b φ → b = 2.618 ∨ b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1454_145418


namespace NUMINAMATH_GPT_value_of_a4_l1454_145410

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := n^2 - 3 * n - 4

-- State the main proof problem.
theorem value_of_a4 : a_n 4 = 0 := by
  sorry

end NUMINAMATH_GPT_value_of_a4_l1454_145410


namespace NUMINAMATH_GPT_find_fifth_month_sale_l1454_145491

theorem find_fifth_month_sale (s1 s2 s3 s4 s6 A : ℝ) (h1 : s1 = 800) (h2 : s2 = 900) (h3 : s3 = 1000) (h4 : s4 = 700) (h5 : s6 = 900) (h6 : A = 850) :
  ∃ s5 : ℝ, (s1 + s2 + s3 + s4 + s5 + s6) / 6 = A ∧ s5 = 800 :=
by
  sorry

end NUMINAMATH_GPT_find_fifth_month_sale_l1454_145491


namespace NUMINAMATH_GPT_smallest_base_l1454_145458

theorem smallest_base (b : ℕ) (n : ℕ) : (n = 512) → (b^3 ≤ n ∧ n < b^4) → ((n / b^3) % b + 1) % 2 = 0 → b = 6 := sorry

end NUMINAMATH_GPT_smallest_base_l1454_145458


namespace NUMINAMATH_GPT_find_x_from_percents_l1454_145420

theorem find_x_from_percents (x : ℝ) (h : 0.65 * x = 0.20 * 487.50) : x = 150 :=
by
  -- Distilled condition from problem
  have h1 : 0.65 * x = 0.20 * 487.50 := h
  -- Start actual logic here
  sorry

end NUMINAMATH_GPT_find_x_from_percents_l1454_145420


namespace NUMINAMATH_GPT_intersection_eq_l1454_145435

open Set

def A : Set ℕ := {0, 2, 4, 6}
def B : Set ℕ := {x | 3 < x ∧ x < 7}

theorem intersection_eq : A ∩ B = {4, 6} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_eq_l1454_145435


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1454_145459

-- Problem 1 Statement
theorem problem1 : (π - 3.14)^0 + (1 / 2)^(-1) + (-1)^(2023) = 2 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 2 Statement
theorem problem2 (b : ℝ) : (-b)^2 * b + 6 * b^4 / (2 * b) + (-2 * b)^3 = -4 * b^3 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 3 Statement
theorem problem3 (x : ℝ) : (x - 1)^2 - x * (x + 2) = -4 * x + 1 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

end NUMINAMATH_GPT_problem1_problem2_problem3_l1454_145459


namespace NUMINAMATH_GPT_Alex_runs_faster_l1454_145470

def Rick_speed : ℚ := 5
def Jen_speed : ℚ := (3 / 4) * Rick_speed
def Mark_speed : ℚ := (4 / 3) * Jen_speed
def Alex_speed : ℚ := (5 / 6) * Mark_speed

theorem Alex_runs_faster : Alex_speed = 25 / 6 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_Alex_runs_faster_l1454_145470


namespace NUMINAMATH_GPT_glee_club_female_members_l1454_145482

theorem glee_club_female_members (m f : ℕ) 
  (h1 : f = 2 * m) 
  (h2 : m + f = 18) : 
  f = 12 :=
by
  sorry

end NUMINAMATH_GPT_glee_club_female_members_l1454_145482


namespace NUMINAMATH_GPT_sum_of_digits_of_77_is_14_l1454_145466

-- Define the conditions given in the problem
def triangular_array_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Define what it means to be the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- The actual Lean theorem statement
theorem sum_of_digits_of_77_is_14 (N : ℕ) (h : triangular_array_sum N = 3003) : sum_of_digits N = 14 :=
by
  sorry  -- Proof to be completed here

end NUMINAMATH_GPT_sum_of_digits_of_77_is_14_l1454_145466


namespace NUMINAMATH_GPT_smallest_solution_l1454_145487

theorem smallest_solution (x : ℝ) (h : x^4 - 16 * x^2 + 63 = 0) :
  x = -3 :=
sorry

end NUMINAMATH_GPT_smallest_solution_l1454_145487


namespace NUMINAMATH_GPT_main_theorem_l1454_145406

-- defining the conditions
def cost_ratio_pen_pencil (x : ℕ) : Prop :=
  ∀ (pen pencil : ℕ), pen = 5 * pencil ∧ x = pencil

def cost_3_pens_pencils (pen pencil total_cost : ℕ) : Prop :=
  total_cost = 3 * pen + 7 * pencil  -- assuming "some pencils" translates to 7 pencils for this demonstration

def total_cost_dozen_pens (pen total_cost : ℕ) : Prop :=
  total_cost = 12 * pen

-- proving the main statement from conditions
theorem main_theorem (pen pencil total_cost : ℕ) (x : ℕ) 
  (h1 : cost_ratio_pen_pencil x)
  (h2 : cost_3_pens_pencils (5 * x) x 100)
  (h3 : total_cost_dozen_pens (5 * x) 300) :
  total_cost = 300 :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l1454_145406


namespace NUMINAMATH_GPT_find_a1_l1454_145437

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (s : ℕ → ℝ) :=
∀ n : ℕ, s n = (n * (a 1 + a n)) / 2

theorem find_a1 
  (a : ℕ → ℝ) (s : ℕ → ℝ)
  (d : ℝ)
  (h_seq : arithmetic_sequence a d)
  (h_sum : sum_first_n_terms a s)
  (h_S10_eq_S11 : s 10 = s 11) : 
  a 1 = 20 := 
sorry

end NUMINAMATH_GPT_find_a1_l1454_145437


namespace NUMINAMATH_GPT_find_avg_speed_l1454_145478

variables (v t : ℝ)

noncomputable def avg_speed_cond := 
  (v + Real.sqrt 15) * (t - Real.pi / 4) = v * t

theorem find_avg_speed (h : avg_speed_cond v t) : v = Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_find_avg_speed_l1454_145478


namespace NUMINAMATH_GPT_intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l1454_145438

variable (a b : ℝ)

-- Define the equations given in the problem
def line1 := ∀ y : ℝ, 3 = (1/3) * y + a
def line2 := ∀ x : ℝ, 3 = (1/3) * x + b

-- The Lean statement for the proof
theorem intersecting_lines_at_3_3_implies_a_plus_b_eq_4 :
  (line1 3) ∧ (line2 3) → a + b = 4 :=
by 
  sorry

end NUMINAMATH_GPT_intersecting_lines_at_3_3_implies_a_plus_b_eq_4_l1454_145438


namespace NUMINAMATH_GPT_find_interior_angles_l1454_145413

theorem find_interior_angles (A B C : ℝ) (h1 : B = A + 10) (h2 : C = B + 10) (h3 : A + B + C = 180) : 
  A = 50 ∧ B = 60 ∧ C = 70 := by
  sorry

end NUMINAMATH_GPT_find_interior_angles_l1454_145413


namespace NUMINAMATH_GPT_max_value_of_expression_l1454_145465

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1454_145465


namespace NUMINAMATH_GPT_Julio_fish_catch_rate_l1454_145461

theorem Julio_fish_catch_rate (F : ℕ) : 
  (9 * F) - 15 = 48 → F = 7 :=
by
  intro h1
  --- proof
  sorry

end NUMINAMATH_GPT_Julio_fish_catch_rate_l1454_145461


namespace NUMINAMATH_GPT_ellipse_equation_with_m_l1454_145486

theorem ellipse_equation_with_m (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m ∈ Set.Ioi 5 := 
sorry

end NUMINAMATH_GPT_ellipse_equation_with_m_l1454_145486


namespace NUMINAMATH_GPT_find_f_neg2016_l1454_145414

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f_neg2016 (a b k : ℝ) (h : f a b 2016 = k) (h_ab : a * b ≠ 0) : f a b (-2016) = 2 - k :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg2016_l1454_145414


namespace NUMINAMATH_GPT_slope_angle_y_eq_neg1_l1454_145449

theorem slope_angle_y_eq_neg1 : (∃ line : ℝ → ℝ, ∀ y: ℝ, line y = -1 → ∃ θ : ℝ, θ = 0) :=
by
  -- Sorry is used to skip the proof.
  sorry

end NUMINAMATH_GPT_slope_angle_y_eq_neg1_l1454_145449


namespace NUMINAMATH_GPT_smallest_whole_number_l1454_145443

theorem smallest_whole_number :
  ∃ x : ℕ, x % 3 = 2 ∧ x % 5 = 3 ∧ x % 7 = 4 ∧ x = 23 :=
sorry

end NUMINAMATH_GPT_smallest_whole_number_l1454_145443


namespace NUMINAMATH_GPT_f_of_pi_over_6_l1454_145477

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

theorem f_of_pi_over_6 (ω ϕ : ℝ) (h₀ : ω > 0) (h₁ : -Real.pi / 2 ≤ ϕ) (h₂ : ϕ < Real.pi / 2) 
  (transformed : ∀ x, f ω ϕ (x/2 - Real.pi/6) = Real.sin x) :
  f ω ϕ (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_f_of_pi_over_6_l1454_145477


namespace NUMINAMATH_GPT_quadratic_solution_interval_l1454_145412

noncomputable def quadratic_inequality (z : ℝ) : Prop :=
  z^2 - 56*z + 360 ≤ 0

theorem quadratic_solution_interval :
  {z : ℝ // quadratic_inequality z} = {z : ℝ // 8 ≤ z ∧ z ≤ 45} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_interval_l1454_145412


namespace NUMINAMATH_GPT_least_value_expression_l1454_145444

open Real

theorem least_value_expression (x : ℝ) : 
  let expr := (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * cos (2 * x)
  ∃ a : ℝ, expr = a ∧ ∀ b : ℝ, b < a → False :=
sorry

end NUMINAMATH_GPT_least_value_expression_l1454_145444


namespace NUMINAMATH_GPT_current_women_count_l1454_145409

variable (x : ℕ) -- Let x be the common multiplier.
variable (initial_men : ℕ := 4 * x)
variable (initial_women : ℕ := 5 * x)

-- Conditions
variable (men_after_entry : ℕ := initial_men + 2)
variable (women_after_leave : ℕ := initial_women - 3)
variable (current_women : ℕ := 2 * women_after_leave)
variable (current_men : ℕ := 14)

-- Theorem statement
theorem current_women_count (h : men_after_entry = current_men) : current_women = 24 := by
  sorry

end NUMINAMATH_GPT_current_women_count_l1454_145409


namespace NUMINAMATH_GPT_percent_singles_l1454_145495

theorem percent_singles (total_hits home_runs triples doubles : ℕ) 
  (h_total: total_hits = 50) 
  (h_hr: home_runs = 3) 
  (h_tr: triples = 2) 
  (h_double: doubles = 8) : 
  100 * (total_hits - (home_runs + triples + doubles)) / total_hits = 74 := 
by
  -- proofs
  sorry

end NUMINAMATH_GPT_percent_singles_l1454_145495


namespace NUMINAMATH_GPT_ninth_grade_class_notification_l1454_145490

theorem ninth_grade_class_notification (n : ℕ) (h1 : 1 + n + n * n = 43) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_ninth_grade_class_notification_l1454_145490


namespace NUMINAMATH_GPT_mixed_numbers_sum_l1454_145401

-- Declare the mixed numbers as fraction equivalents
def mixed1 : ℚ := 2 + 1/10
def mixed2 : ℚ := 3 + 11/100
def mixed3 : ℚ := 4 + 111/1000

-- Assert that the sum of mixed1, mixed2, and mixed3 is equal to 9.321
theorem mixed_numbers_sum : mixed1 + mixed2 + mixed3 = 9321 / 1000 := by
  sorry

end NUMINAMATH_GPT_mixed_numbers_sum_l1454_145401


namespace NUMINAMATH_GPT_probability_correct_l1454_145496

/-
  Problem statement:
  Consider a modified city map where a student walks from intersection A to intersection B, passing through C and D.
  The student always walks east or south and at each intersection, decides the direction to go with a probability of 1/2.
  The map requires 4 eastward and 3 southward moves to reach B from A. C is 2 east, 1 south move from A. D is 3 east, 2 south moves from A.
  Prove that the probability the student goes through both C and D is 12/35.
-/

noncomputable def probability_passing_C_and_D : ℚ :=
  let total_paths_A_to_B := Nat.choose 7 4
  let paths_A_to_C := Nat.choose 3 2
  let paths_C_to_D := Nat.choose 2 1
  let paths_D_to_B := Nat.choose 2 1
  (paths_A_to_C * paths_C_to_D * paths_D_to_B) / total_paths_A_to_B

theorem probability_correct :
  probability_passing_C_and_D = 12 / 35 :=
by
  sorry

end NUMINAMATH_GPT_probability_correct_l1454_145496


namespace NUMINAMATH_GPT_evaluate_expression_l1454_145439

theorem evaluate_expression (x y z : ℝ) : 
  (x + (y + z)) - ((-x + y) + z) = 2 * x := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1454_145439


namespace NUMINAMATH_GPT_caltech_equilateral_triangles_l1454_145442

theorem caltech_equilateral_triangles (n : ℕ) (h : n = 900) :
  let total_triangles := (n * (n - 1) / 2) * 2
  let overcounted_triangles := n / 3
  total_triangles - overcounted_triangles = 808800 :=
by
  sorry

end NUMINAMATH_GPT_caltech_equilateral_triangles_l1454_145442


namespace NUMINAMATH_GPT_largest_angle_of_trapezoid_arithmetic_sequence_l1454_145408

variables (a d : ℝ)

-- Given Conditions
def smallest_angle : Prop := a = 45
def trapezoid_property : Prop := a + 3 * d = 135

theorem largest_angle_of_trapezoid_arithmetic_sequence 
  (ha : smallest_angle a) (ht : a + (a + 3 * d) = 180) : 
  a + 3 * d = 135 :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_of_trapezoid_arithmetic_sequence_l1454_145408


namespace NUMINAMATH_GPT_sum_reciprocal_inequality_l1454_145488

theorem sum_reciprocal_inequality (p q a b c d e : ℝ) (hp : 0 < p) (ha : p ≤ a) (hb : p ≤ b) (hc : p ≤ c) (hd : p ≤ d) (he : p ≤ e) (haq : a ≤ q) (hbq : b ≤ q) (hcq : c ≤ q) (hdq : d ≤ q) (heq : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) ≤ 25 + 6 * ((Real.sqrt (q / p) - Real.sqrt (p / q)) ^ 2) :=
by sorry

end NUMINAMATH_GPT_sum_reciprocal_inequality_l1454_145488


namespace NUMINAMATH_GPT_at_least_one_not_less_than_two_l1454_145455

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x, (x = a + 1/b ∨ x = b + 1/c ∨ x = c + 1/a) ∧ 2 ≤ x :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_two_l1454_145455


namespace NUMINAMATH_GPT_find_vertex_l1454_145464

noncomputable def parabola_vertex (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y - 3 * x + 6 = 0

theorem find_vertex :
  ∃ (x y : ℝ), parabola_vertex x y ∧ x = -14/3 ∧ y = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_vertex_l1454_145464


namespace NUMINAMATH_GPT_angle_equiv_330_neg390_l1454_145463

theorem angle_equiv_330_neg390 : ∃ k : ℤ, 330 = -390 + 360 * k :=
by
  sorry

end NUMINAMATH_GPT_angle_equiv_330_neg390_l1454_145463


namespace NUMINAMATH_GPT_percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l1454_145467

theorem percentage_increase_first_job :
  let old_salary := 65
  let new_salary := 70
  (new_salary - old_salary) / old_salary * 100 = 7.69 := by
  sorry

theorem percentage_increase_second_job :
  let old_salary := 120
  let new_salary := 138
  (new_salary - old_salary) / old_salary * 100 = 15 := by
  sorry

theorem percentage_increase_third_job :
  let old_salary := 200
  let new_salary := 220
  (new_salary - old_salary) / old_salary * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l1454_145467


namespace NUMINAMATH_GPT_container_volume_ratio_l1454_145483

theorem container_volume_ratio (C D : ℝ) (hC: C > 0) (hD: D > 0)
  (h: (3/4) * C = (5/8) * D) : (C / D) = (5 / 6) :=
by
  sorry

end NUMINAMATH_GPT_container_volume_ratio_l1454_145483


namespace NUMINAMATH_GPT_common_difference_l1454_145422

variable (a : ℕ → ℝ)

def arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∃ a1, ∀ n, a n = a1 + (n - 1) * d

def geometric_sequence (a1 a2 a5 : ℝ) : Prop :=
  a1 * (a1 + 4 * (a2 - a1)) = (a2 - a1)^2

theorem common_difference {d : ℝ} (hd : d ≠ 0)
  (h_arith : arithmetic a d)
  (h_sum : a 1 + a 2 + a 5 = 13)
  (h_geom : geometric_sequence (a 1) (a 2) (a 5)) :
  d = 2 :=
sorry

end NUMINAMATH_GPT_common_difference_l1454_145422


namespace NUMINAMATH_GPT_length_of_first_leg_of_triangle_l1454_145440

theorem length_of_first_leg_of_triangle 
  (a b c : ℝ) 
  (h1 : b = 8) 
  (h2 : c = 10) 
  (h3 : c^2 = a^2 + b^2) : 
  a = 6 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_leg_of_triangle_l1454_145440
