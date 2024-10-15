import Mathlib

namespace NUMINAMATH_GPT_value_of_expression_l565_56559

theorem value_of_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) : 
  (x^4 + 3 * y^3 + 10) / 7 = 283 / 7 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l565_56559


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_min_focal_distance_l565_56515

theorem asymptotes_of_hyperbola_min_focal_distance :
  ∀ (x y m : ℝ),
  (m = 1 → 
   (∀ x y : ℝ, (x^2 / (m^2 + 8) - y^2 / (6 - 2 * m) = 1) → 
   (y = 2/3 * x ∨ y = -2/3 * x))) := 
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_min_focal_distance_l565_56515


namespace NUMINAMATH_GPT_triangle_ratio_l565_56538

-- Define the conditions and the main theorem statement
theorem triangle_ratio (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h_eq : b * Real.cos C + c * Real.cos B = 2 * b) 
  (h_law_sines_a : a = 2 * b * Real.sin B / Real.sin A) 
  (h_angles : A + B + C = Real.pi) :
  b / a = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_ratio_l565_56538


namespace NUMINAMATH_GPT_wire_length_from_sphere_volume_l565_56573

theorem wire_length_from_sphere_volume
  (r_sphere : ℝ) (r_cylinder : ℝ) (h : ℝ)
  (h_sphere : r_sphere = 12)
  (h_cylinder : r_cylinder = 4)
  (volume_conservation : (4/3 * Real.pi * r_sphere^3) = (Real.pi * r_cylinder^2 * h)) :
  h = 144 :=
by {
  sorry
}

end NUMINAMATH_GPT_wire_length_from_sphere_volume_l565_56573


namespace NUMINAMATH_GPT_circumference_to_diameter_ratio_l565_56502

theorem circumference_to_diameter_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) :
  C / D = 3.14 :=
by
  rw [hC, hD]
  norm_num

end NUMINAMATH_GPT_circumference_to_diameter_ratio_l565_56502


namespace NUMINAMATH_GPT_find_c_d_l565_56501

def star (c d : ℕ) : ℕ := c^d + c*d

theorem find_c_d (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d) (h_star : star c d = 28) : c + d = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_c_d_l565_56501


namespace NUMINAMATH_GPT_am_gm_inequality_l565_56512

theorem am_gm_inequality {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ a + b + c :=
by
  sorry

end NUMINAMATH_GPT_am_gm_inequality_l565_56512


namespace NUMINAMATH_GPT_lowest_possible_price_l565_56531

theorem lowest_possible_price 
  (MSRP : ℝ)
  (regular_discount_percentage additional_discount_percentage : ℝ)
  (h1 : MSRP = 40)
  (h2 : regular_discount_percentage = 0.30)
  (h3 : additional_discount_percentage = 0.20) : 
  (MSRP * (1 - regular_discount_percentage) * (1 - additional_discount_percentage) = 22.40) := 
by
  sorry

end NUMINAMATH_GPT_lowest_possible_price_l565_56531


namespace NUMINAMATH_GPT_divisible_by_117_l565_56513

theorem divisible_by_117 (n : ℕ) (hn : 0 < n) :
  117 ∣ (3^(2*(n+1)) * 5^(2*n) - 3^(3*n+2) * 2^(2*n)) :=
sorry

end NUMINAMATH_GPT_divisible_by_117_l565_56513


namespace NUMINAMATH_GPT_unreachable_y_l565_56597

noncomputable def y_function (x : ℝ) : ℝ := (2 - 3 * x) / (5 * x - 1)

theorem unreachable_y : ¬ ∃ x : ℝ, y_function x = -3 / 5 ∧ x ≠ 1 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_unreachable_y_l565_56597


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l565_56539

theorem quadratic_has_two_distinct_real_roots (p : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - 3) * (x1 - 2) - p^2 = 0 ∧ (x2 - 3) * (x2 - 2) - p^2 = 0 :=
by
  -- This part will be replaced with the actual proof
  sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l565_56539


namespace NUMINAMATH_GPT_sin_of_7pi_over_6_l565_56528

theorem sin_of_7pi_over_6 : Real.sin (7 * Real.pi / 6) = -1 / 2 :=
by
  -- Conditions from the statement in a)
  -- Given conditions: \(\sin (180^\circ + \theta) = -\sin \theta\)
  -- \(\sin 30^\circ = \frac{1}{2}\)
  sorry

end NUMINAMATH_GPT_sin_of_7pi_over_6_l565_56528


namespace NUMINAMATH_GPT_total_area_three_plots_l565_56591

variable (x y z A : ℝ)

theorem total_area_three_plots :
  (x = (2 / 5) * A) →
  (z = x - 16) →
  (y = (9 / 8) * z) →
  (A = x + y + z) →
  A = 96 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_area_three_plots_l565_56591


namespace NUMINAMATH_GPT_proof_problem_l565_56540

-- Define the function f(x) = -x - x^3
def f (x : ℝ) : ℝ := -x - x^3

-- Define the main theorem according to the conditions and the required proofs.
theorem proof_problem (x1 x2 : ℝ) (h : x1 + x2 ≤ 0) :
  (f x1) * (f (-x1)) ≤ 0 ∧ (f x1 + f x2) ≥ (f (-x1) + f (-x2)) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l565_56540


namespace NUMINAMATH_GPT_find_x_l565_56543

noncomputable def e_squared := Real.exp 2

theorem find_x (x : ℝ) (h : Real.log (x^2 - 5*x + 10) = 2) :
  x = 4.4 ∨ x = 0.6 :=
sorry

end NUMINAMATH_GPT_find_x_l565_56543


namespace NUMINAMATH_GPT_zero_in_A_l565_56517

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : 0 ∈ A := by
  sorry

end NUMINAMATH_GPT_zero_in_A_l565_56517


namespace NUMINAMATH_GPT_suzhou_visitors_accuracy_l565_56511

/--
In Suzhou, during the National Day holiday in 2023, the city received 17.815 million visitors.
Given that number, prove that it is accurate to the thousands place.
-/
theorem suzhou_visitors_accuracy :
  (17.815 : ℝ) * 10^6 = 17815000 ∧ true := 
by
sorry

end NUMINAMATH_GPT_suzhou_visitors_accuracy_l565_56511


namespace NUMINAMATH_GPT_more_students_suggested_bacon_than_mashed_potatoes_l565_56523

-- Define the number of students suggesting each type of food
def students_suggesting_mashed_potatoes := 479
def students_suggesting_bacon := 489

-- State the theorem that needs to be proven
theorem more_students_suggested_bacon_than_mashed_potatoes :
  students_suggesting_bacon - students_suggesting_mashed_potatoes = 10 := 
  by
  sorry

end NUMINAMATH_GPT_more_students_suggested_bacon_than_mashed_potatoes_l565_56523


namespace NUMINAMATH_GPT_trigonometric_identity_l565_56598

theorem trigonometric_identity (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (Real.cos (α / 2) ^ 2) = 4 * Real.sin α :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l565_56598


namespace NUMINAMATH_GPT_smallest_real_number_among_minus3_minus2_0_2_is_minus3_l565_56594

theorem smallest_real_number_among_minus3_minus2_0_2_is_minus3 :
  min (min (-3:ℝ) (-2)) (min 0 2) = -3 :=
by {
    sorry
}

end NUMINAMATH_GPT_smallest_real_number_among_minus3_minus2_0_2_is_minus3_l565_56594


namespace NUMINAMATH_GPT_ananthu_can_complete_work_in_45_days_l565_56561

def amit_work_rate : ℚ := 1 / 15

def time_amit_worked : ℚ := 3

def total_work : ℚ := 1

def total_days : ℚ := 39

noncomputable def ananthu_days (x : ℚ) : Prop :=
  let amit_work_done := time_amit_worked * amit_work_rate
  let remaining_work := total_work - amit_work_done
  let ananthu_work_rate := remaining_work / (total_days - time_amit_worked)
  1 /x = ananthu_work_rate

theorem ananthu_can_complete_work_in_45_days :
  ananthu_days 45 :=
by
  sorry

end NUMINAMATH_GPT_ananthu_can_complete_work_in_45_days_l565_56561


namespace NUMINAMATH_GPT_value_of_expression_l565_56545

theorem value_of_expression (a b : ℤ) (h : a - 2 * b - 3 = 0) : 9 - 2 * a + 4 * b = 3 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_expression_l565_56545


namespace NUMINAMATH_GPT_max_value_expression_l565_56571

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ (M : ℝ), M = (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) / ((x + y)^3 * (y + z)^3) ∧ M = 1/24 := 
sorry

end NUMINAMATH_GPT_max_value_expression_l565_56571


namespace NUMINAMATH_GPT_peters_brother_read_percentage_l565_56533

-- Definitions based on given conditions
def total_books : ℕ := 20
def peter_read_percentage : ℕ := 40
def difference_between_peter_and_brother : ℕ := 6

-- Statement to prove
theorem peters_brother_read_percentage :
  peter_read_percentage / 100 * total_books - difference_between_peter_and_brother = 2 → 
  2 / total_books * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_peters_brother_read_percentage_l565_56533


namespace NUMINAMATH_GPT_n_fraction_of_sum_l565_56504

theorem n_fraction_of_sum (l : List ℝ) (h1 : l.length = 21) (n : ℝ) (h2 : n ∈ l)
  (h3 : ∃ m, l.erase n = m ∧ m.length = 20 ∧ n = 4 * (m.sum / 20)) :
  n = (l.sum) / 6 :=
by
  sorry

end NUMINAMATH_GPT_n_fraction_of_sum_l565_56504


namespace NUMINAMATH_GPT_range_of_a_l565_56537

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := 
by sorry

end NUMINAMATH_GPT_range_of_a_l565_56537


namespace NUMINAMATH_GPT_newer_model_distance_l565_56579

-- Given conditions
def older_model_distance : ℕ := 160
def newer_model_factor : ℝ := 1.25

-- The statement to be proved
theorem newer_model_distance :
  newer_model_factor * (older_model_distance : ℝ) = 200 := by
  sorry

end NUMINAMATH_GPT_newer_model_distance_l565_56579


namespace NUMINAMATH_GPT_angle_size_proof_l565_56574

-- Define the problem conditions
def fifteen_points_on_circle (θ : ℕ) : Prop :=
  θ = 360 / 15 

-- Define the central angles
def central_angle_between_adjacent_points (θ : ℕ) : ℕ :=
  360 / 15  

-- Define the two required central angles
def central_angle_A1O_A3 (θ : ℕ) : ℕ :=
  2 * θ

def central_angle_A3O_A7 (θ : ℕ) : ℕ :=
  4 * θ

-- Define the problem using the given conditions and the proven answer
noncomputable def angle_A1_A3_A7 : ℕ :=
  108

-- Lean 4 statement of the math problem to prove
theorem angle_size_proof (θ : ℕ) (h1 : fifteen_points_on_circle θ) :
  central_angle_A1O_A3 θ = 48 ∧ central_angle_A3O_A7 θ = 96 → 
  angle_A1_A3_A7 = 108 :=
by sorry

#check angle_size_proof

end NUMINAMATH_GPT_angle_size_proof_l565_56574


namespace NUMINAMATH_GPT_compound_interest_rate_l565_56596

theorem compound_interest_rate
  (A P : ℝ) (t n : ℝ)
  (HA : A = 1348.32)
  (HP : P = 1200)
  (Ht : t = 2)
  (Hn : n = 1) :
  ∃ r : ℝ, 0 ≤ r ∧ ((A / P) ^ (1 / (n * t)) - 1) = r ∧ r = 0.06 := 
sorry

end NUMINAMATH_GPT_compound_interest_rate_l565_56596


namespace NUMINAMATH_GPT_mark_profit_l565_56527

variable (initial_cost tripling_factor new_value profit : ℕ)

-- Conditions
def initial_card_cost := 100
def card_tripling_factor := 3

-- Calculations based on conditions
def card_new_value := initial_card_cost * card_tripling_factor
def card_profit := card_new_value - initial_card_cost

-- Proof Statement
theorem mark_profit (initial_card_cost tripling_factor card_new_value card_profit : ℕ) 
  (h1: initial_card_cost = 100)
  (h2: tripling_factor = 3)
  (h3: card_new_value = initial_card_cost * tripling_factor)
  (h4: card_profit = card_new_value - initial_card_cost) :
  card_profit = 200 :=
  by sorry

end NUMINAMATH_GPT_mark_profit_l565_56527


namespace NUMINAMATH_GPT_mean_inequality_l565_56595

variable (a b : ℝ)

-- Conditions: a and b are distinct and non-zero
axiom h₀ : a ≠ b
axiom h₁ : a ≠ 0
axiom h₂ : b ≠ 0

theorem mean_inequality (h₀ : a ≠ b) (h₁ : a ≠ 0) (h₂ : b ≠ 0) : 
  (a^2 + b^2) / 2 > (a + b) / 2 ∧ (a + b) / 2 > Real.sqrt (a * b) :=
sorry -- Proof is not provided, only statement.

end NUMINAMATH_GPT_mean_inequality_l565_56595


namespace NUMINAMATH_GPT_simplify_polynomial_l565_56518

theorem simplify_polynomial (r : ℝ) :
  (2 * r ^ 3 + 5 * r ^ 2 - 4 * r + 8) - (r ^ 3 + 9 * r ^ 2 - 2 * r - 3)
  = r ^ 3 - 4 * r ^ 2 - 2 * r + 11 :=
by sorry

end NUMINAMATH_GPT_simplify_polynomial_l565_56518


namespace NUMINAMATH_GPT_ellipse_major_axis_value_l565_56530

theorem ellipse_major_axis_value (m : ℝ) (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : ∀ {x y : ℝ}, (x, y) = P → (x^2 / m) + (y^2 / 16) = 1)
  (h2 : dist P F1 = 3)
  (h3 : dist P F2 = 7)
  : m = 25 :=
sorry

end NUMINAMATH_GPT_ellipse_major_axis_value_l565_56530


namespace NUMINAMATH_GPT_tips_fraction_of_salary_l565_56580

theorem tips_fraction_of_salary (S T x : ℝ) (h1 : T = x * S) 
  (h2 : T / (S + T) = 1 / 3) : x = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_tips_fraction_of_salary_l565_56580


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l565_56582

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 12) (h2 : x * y = 9) : x^2 + y^2 = 162 :=
by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l565_56582


namespace NUMINAMATH_GPT_abs_diff_60th_terms_arithmetic_sequences_l565_56555

theorem abs_diff_60th_terms_arithmetic_sequences :
  let C : (ℕ → ℤ) := λ n => 25 + 15 * (n - 1)
  let D : (ℕ → ℤ) := λ n => 40 - 15 * (n - 1)
  |C 60 - D 60| = 1755 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_60th_terms_arithmetic_sequences_l565_56555


namespace NUMINAMATH_GPT_min_value_of_function_l565_56508

theorem min_value_of_function (x : ℝ) (hx : x > 4) : 
  ∃ y : ℝ, y = x + 1 / (x - 4) ∧ (∀ z : ℝ, z = x + 1 / (x - 4) → z ≥ 6) :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l565_56508


namespace NUMINAMATH_GPT_negation_equivalence_l565_56519

variable (x : ℝ)

def original_proposition := ∃ x : ℝ, x^2 - 3*x + 3 < 0

def negation_proposition := ∀ x : ℝ, x^2 - 3*x + 3 ≥ 0

theorem negation_equivalence : ¬ original_proposition ↔ negation_proposition :=
by 
  -- Lean doesn’t require the actual proof here
  sorry

end NUMINAMATH_GPT_negation_equivalence_l565_56519


namespace NUMINAMATH_GPT_ratio_of_sweater_vests_to_shirts_l565_56584

theorem ratio_of_sweater_vests_to_shirts (S V O : ℕ) (h1 : S = 3) (h2 : O = 18) (h3 : O = V * S) : (V : ℚ) / (S : ℚ) = 2 := 
  by
  sorry

end NUMINAMATH_GPT_ratio_of_sweater_vests_to_shirts_l565_56584


namespace NUMINAMATH_GPT_circle_proof_problem_l565_56593

variables {P Q R : Type}
variables {p q r dPQ dPR dQR : ℝ}

-- Given Conditions
variables (hpq : p > q) (hqr : q > r)
variables (hdPQ : ℝ) (hdPR : ℝ) (hdQR : ℝ)

-- Statement of the problem: prove that all conditions can be true
theorem circle_proof_problem :
  (∃ hpq' : dPQ = p + q, true) ∧
  (∃ hqr' : dQR = q + r, true) ∧
  (∃ hpr' : dPR > p + r, true) ∧
  (∃ hpq_diff : dPQ > p - q, true) →
  false := 
sorry

end NUMINAMATH_GPT_circle_proof_problem_l565_56593


namespace NUMINAMATH_GPT_number_of_fish_given_to_dog_l565_56563

-- Define the conditions
def condition1 (D C : ℕ) : Prop := C = D / 2
def condition2 (D C : ℕ) : Prop := D + C = 60

-- Theorem to prove the number of fish given to the dog
theorem number_of_fish_given_to_dog (D : ℕ) (C : ℕ) (h1 : condition1 D C) (h2 : condition2 D C) : D = 40 :=
by
  sorry

end NUMINAMATH_GPT_number_of_fish_given_to_dog_l565_56563


namespace NUMINAMATH_GPT_gcd_of_numbers_l565_56554

theorem gcd_of_numbers :
  let a := 125^2 + 235^2 + 349^2
  let b := 124^2 + 234^2 + 350^2
  gcd a b = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_of_numbers_l565_56554


namespace NUMINAMATH_GPT_maximize_profit_l565_56550

def cost_A : ℝ := 3
def price_A : ℝ := 3.3
def cost_B : ℝ := 2.4
def price_B : ℝ := 2.8
def total_devices : ℕ := 50

def profit (x : ℕ) : ℝ := (price_A - cost_A) * x + (price_B - cost_B) * (total_devices - x)

def functional_relationship (x : ℕ) : ℝ := -0.1 * x + 20

def purchase_condition (x : ℕ) : Prop := 4 * x ≥ total_devices - x

theorem maximize_profit :
    functional_relationship (10) = 19 ∧ 
    (∀ x : ℕ, purchase_condition x → functional_relationship x ≤ 19) :=
by {
    -- Proof omitted
    sorry
}

end NUMINAMATH_GPT_maximize_profit_l565_56550


namespace NUMINAMATH_GPT_mitya_age_l565_56578

theorem mitya_age {M S: ℕ} (h1 : M = S + 11) (h2 : S = 2 * (S - (M - S))) : M = 33 :=
by
  -- proof steps skipped
  sorry

end NUMINAMATH_GPT_mitya_age_l565_56578


namespace NUMINAMATH_GPT_genuine_coin_remains_l565_56587

theorem genuine_coin_remains (n : ℕ) (g f : ℕ) (h : n = 2022) (h_g : g > n/2) (h_f : f = n - g) : 
  (after_moves : ℕ) -> after_moves = n - 1 -> ∃ remaining_g : ℕ, remaining_g > 0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_genuine_coin_remains_l565_56587


namespace NUMINAMATH_GPT_units_digit_char_of_p_l565_56526

theorem units_digit_char_of_p (p : ℕ) (h_pos : 0 < p) (h_even : p % 2 = 0)
    (h_units_zero : (p^3 % 10) - (p^2 % 10) = 0) (h_units_eleven : (p + 5) % 10 = 1) :
    p % 10 = 6 :=
sorry

end NUMINAMATH_GPT_units_digit_char_of_p_l565_56526


namespace NUMINAMATH_GPT_Carrie_hourly_wage_l565_56583

theorem Carrie_hourly_wage (hours_per_week : ℕ) (weeks_per_month : ℕ) (cost_bike : ℕ) (remaining_money : ℕ)
  (total_hours : ℕ) (total_savings : ℕ) (x : ℕ) :
  hours_per_week = 35 → 
  weeks_per_month = 4 → 
  cost_bike = 400 → 
  remaining_money = 720 → 
  total_hours = hours_per_week * weeks_per_month → 
  total_savings = cost_bike + remaining_money → 
  total_savings = total_hours * x → 
  x = 8 :=
by 
  intros h_hw h_wm h_cb h_rm h_th h_ts h_tx
  sorry

end NUMINAMATH_GPT_Carrie_hourly_wage_l565_56583


namespace NUMINAMATH_GPT_lindsey_owns_more_cars_than_cathy_l565_56535

theorem lindsey_owns_more_cars_than_cathy :
  ∀ (cathy carol susan lindsey : ℕ),
    cathy = 5 →
    carol = 2 * cathy →
    susan = carol - 2 →
    cathy + carol + susan + lindsey = 32 →
    lindsey = cathy + 4 :=
by
  intros cathy carol susan lindsey h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_lindsey_owns_more_cars_than_cathy_l565_56535


namespace NUMINAMATH_GPT_five_goats_choir_l565_56589

theorem five_goats_choir 
  (total_members : ℕ)
  (num_rows : ℕ)
  (total_members_eq : total_members = 51)
  (num_rows_eq : num_rows = 4) :
  ∃ row_people : ℕ, row_people ≥ 13 :=
by 
  sorry

end NUMINAMATH_GPT_five_goats_choir_l565_56589


namespace NUMINAMATH_GPT_recommended_sleep_hours_l565_56532

theorem recommended_sleep_hours
  (R : ℝ)   -- The recommended number of hours of sleep per day
  (h1 : 2 * 3 + 5 * (0.60 * R) = 30) : R = 8 :=
sorry

end NUMINAMATH_GPT_recommended_sleep_hours_l565_56532


namespace NUMINAMATH_GPT_domain_of_function_l565_56564

theorem domain_of_function :
  (∀ x : ℝ, (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 0)) :=
sorry

end NUMINAMATH_GPT_domain_of_function_l565_56564


namespace NUMINAMATH_GPT_interior_angles_sum_l565_56552

theorem interior_angles_sum (n : ℕ) (h : ∀ (k : ℕ), k = n → 60 * n = 360) : 
  180 * (n - 2) = 720 :=
by
  sorry

end NUMINAMATH_GPT_interior_angles_sum_l565_56552


namespace NUMINAMATH_GPT_permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l565_56542

-- Part (a)
theorem permutations_with_exactly_one_descent (n : ℕ) : 
  ∃ (count : ℕ), count = 2^n - n - 1 := sorry

-- Part (b)
theorem permutations_with_exactly_two_descents (n : ℕ) : 
  ∃ (count : ℕ), count = 3^n - 2^n * (n + 1) + (n * (n + 1)) / 2 := sorry

end NUMINAMATH_GPT_permutations_with_exactly_one_descent_permutations_with_exactly_two_descents_l565_56542


namespace NUMINAMATH_GPT_possible_values_for_D_l565_56510

noncomputable def distinct_digit_values (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  B < 10 ∧ A < 10 ∧ D < 10 ∧ C < 10 ∧ C = 9 ∧ (B + A = 9 + D)

theorem possible_values_for_D :
  ∃ (Ds : Finset Nat), (∀ D ∈ Ds, ∃ A B C, distinct_digit_values A B C D) ∧
  Ds.card = 5 :=
sorry

end NUMINAMATH_GPT_possible_values_for_D_l565_56510


namespace NUMINAMATH_GPT_left_vertex_of_ellipse_l565_56534

theorem left_vertex_of_ellipse :
  ∃ (a b c : ℝ), 
    (a > b) ∧ (b > 0) ∧ (b = 4) ∧ (c = 3) ∧ 
    (c^2 = a^2 - b^2) ∧ 
    (3^2 = a^2 - 4^2) ∧ 
    (a = 5) ∧ 
    (∀ x y : ℝ, (x, y) = (-5, 0)) := 
sorry

end NUMINAMATH_GPT_left_vertex_of_ellipse_l565_56534


namespace NUMINAMATH_GPT_solve_quadratic_eqn_l565_56557

theorem solve_quadratic_eqn (x : ℝ) : 3 * x ^ 2 = 27 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eqn_l565_56557


namespace NUMINAMATH_GPT_sum_of_interior_angles_at_vertex_A_l565_56565

-- Definitions of the interior angles for a square and a regular octagon.
def square_interior_angle : ℝ := 90
def octagon_interior_angle : ℝ := 135

-- Theorem that states the sum of the interior angles at vertex A formed by the square and octagon.
theorem sum_of_interior_angles_at_vertex_A : square_interior_angle + octagon_interior_angle = 225 := by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_at_vertex_A_l565_56565


namespace NUMINAMATH_GPT_work_increase_percentage_l565_56586

theorem work_increase_percentage (p w : ℕ) (hp : p > 0) : 
  (((4 / 3 : ℚ) * w) - w) / w * 100 = 33.33 := 
sorry

end NUMINAMATH_GPT_work_increase_percentage_l565_56586


namespace NUMINAMATH_GPT_probability_spade_heart_diamond_l565_56592

-- Condition: Definition of probability functions and a standard deck
def probability_of_first_spade (deck : Finset ℕ) : ℚ := 13 / 52
def probability_of_second_heart (deck : Finset ℕ) (first_card_spade : Prop) : ℚ := 13 / 51
def probability_of_third_diamond (deck : Finset ℕ) (first_card_spade : Prop) (second_card_heart : Prop) : ℚ := 13 / 50

-- Combined probability calculation
def probability_sequence_spade_heart_diamond (deck : Finset ℕ) : ℚ := 
  probability_of_first_spade deck * 
  probability_of_second_heart deck (true) * 
  probability_of_third_diamond deck (true) (true)

-- Lean statement proving the problem
theorem probability_spade_heart_diamond :
  probability_sequence_spade_heart_diamond (Finset.range 52) = 2197 / 132600 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_probability_spade_heart_diamond_l565_56592


namespace NUMINAMATH_GPT_problem_statement_l565_56549

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 4

theorem problem_statement (a x₁ x₂: ℝ) (ha : a > 0) (hx : x₁ < x₂) (hxsum : x₁ + x₂ = 0) :
  f a x₁ < f a x₂ := by
  sorry

end NUMINAMATH_GPT_problem_statement_l565_56549


namespace NUMINAMATH_GPT_correct_expression_l565_56553

theorem correct_expression :
  ¬ (|4| = -4) ∧
  ¬ (|4| = -4) ∧
  (-(4^2) ≠ 16)  ∧
  ((-4)^2 = 16) := by
  sorry

end NUMINAMATH_GPT_correct_expression_l565_56553


namespace NUMINAMATH_GPT_total_pencils_is_5_l565_56570

-- Define the initial number of pencils and the number of pencils Tim added
def initial_pencils : Nat := 2
def pencils_added_by_tim : Nat := 3

-- Prove the total number of pencils is equal to 5
theorem total_pencils_is_5 : initial_pencils + pencils_added_by_tim = 5 := by
  sorry

end NUMINAMATH_GPT_total_pencils_is_5_l565_56570


namespace NUMINAMATH_GPT_turkey_2003_problem_l565_56562

theorem turkey_2003_problem (x m n : ℕ) (hx : 0 < x) (hm : 0 < m) (hn : 0 < n) (h : x^m = 2^(2 * n + 1) + 2^n + 1) :
  x = 2^(2 * n + 1) + 2^n + 1 ∧ m = 1 ∨ x = 23 ∧ m = 2 ∧ n = 4 :=
sorry

end NUMINAMATH_GPT_turkey_2003_problem_l565_56562


namespace NUMINAMATH_GPT_cylinder_cone_surface_area_l565_56514

theorem cylinder_cone_surface_area (r h : ℝ) (π : ℝ) (l : ℝ)
    (h_relation : h = Real.sqrt 3 * r)
    (l_relation : l = 2 * r)
    (cone_lateral_surface_area : π * r * l = 2 * π * r ^ 2) :
    (2 * π * r * h) / (π * r ^ 2) = 2 * Real.sqrt 3 :=
by
    sorry

end NUMINAMATH_GPT_cylinder_cone_surface_area_l565_56514


namespace NUMINAMATH_GPT_binom_10_4_eq_210_l565_56581

theorem binom_10_4_eq_210 : Nat.choose 10 4 = 210 :=
  by sorry

end NUMINAMATH_GPT_binom_10_4_eq_210_l565_56581


namespace NUMINAMATH_GPT_distinct_real_roots_range_l565_56509

theorem distinct_real_roots_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 - a = 0) ∧ (x2^2 - 4*x2 - a = 0)) ↔ a > -4 :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_roots_range_l565_56509


namespace NUMINAMATH_GPT_contrapositive_even_addition_l565_56572

theorem contrapositive_even_addition (a b : ℕ) :
  (¬((a % 2 = 0) ∧ (b % 2 = 0)) → (a + b) % 2 ≠ 0) :=
sorry

end NUMINAMATH_GPT_contrapositive_even_addition_l565_56572


namespace NUMINAMATH_GPT_solve_system_and_find_6a_plus_b_l565_56585

theorem solve_system_and_find_6a_plus_b (x y a b : ℝ)
  (h1 : 3 * x - 2 * y + 20 = 0)
  (h2 : 2 * x + 15 * y - 3 = 0)
  (h3 : a * x - b * y = 3) :
  6 * a + b = -3 := by
  sorry

end NUMINAMATH_GPT_solve_system_and_find_6a_plus_b_l565_56585


namespace NUMINAMATH_GPT_union_eq_C_l565_56529

def A: Set ℝ := { x | x > 2 }
def B: Set ℝ := { x | x < 0 }
def C: Set ℝ := { x | x * (x - 2) > 0 }

theorem union_eq_C : (A ∪ B) = C :=
by
  sorry

end NUMINAMATH_GPT_union_eq_C_l565_56529


namespace NUMINAMATH_GPT_max_area_equilateral_in_rectangle_l565_56548

-- Define the dimensions of the rectangle
def length_efgh : ℕ := 15
def width_efgh : ℕ := 8

-- The maximum possible area of an equilateral triangle inscribed in the rectangle
theorem max_area_equilateral_in_rectangle : 
  ∃ (s : ℝ), 
  s = ((16 * Real.sqrt 3) / 3) ∧ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ length_efgh → 
    (∃ (area : ℝ), area = (Real.sqrt 3 / 4 * s^2) ∧
      area = 64 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_GPT_max_area_equilateral_in_rectangle_l565_56548


namespace NUMINAMATH_GPT_scientific_notation_of_12400_l565_56599

theorem scientific_notation_of_12400 :
  12400 = 1.24 * 10^4 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_12400_l565_56599


namespace NUMINAMATH_GPT_train_pass_platform_in_correct_time_l565_56566

def length_of_train : ℝ := 2500
def time_to_cross_tree : ℝ := 90
def length_of_platform : ℝ := 1500

noncomputable def speed_of_train : ℝ := length_of_train / time_to_cross_tree
noncomputable def total_distance_to_cover : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance_to_cover / speed_of_train

theorem train_pass_platform_in_correct_time :
  abs (time_to_pass_platform - 143.88) < 0.01 :=
sorry

end NUMINAMATH_GPT_train_pass_platform_in_correct_time_l565_56566


namespace NUMINAMATH_GPT_smallest_real_number_l565_56521

theorem smallest_real_number (A B C D : ℝ) 
  (hA : A = |(-2 : ℝ)|) 
  (hB : B = -1) 
  (hC : C = 0) 
  (hD : D = -1 / 2) : 
  min A (min B (min C D)) = B := 
by
  sorry

end NUMINAMATH_GPT_smallest_real_number_l565_56521


namespace NUMINAMATH_GPT_rowed_upstream_distance_l565_56516

def distance_downstream := 120
def time_downstream := 2
def distance_upstream := 2
def speed_stream := 15

def speed_boat (V_b : ℝ) := V_b

theorem rowed_upstream_distance (V_b : ℝ) (D_u : ℝ) :
  (distance_downstream = (V_b + speed_stream) * time_downstream) ∧
  (D_u = (V_b - speed_stream) * time_upstream) →
  D_u = 60 :=
by 
  sorry

end NUMINAMATH_GPT_rowed_upstream_distance_l565_56516


namespace NUMINAMATH_GPT_circle_center_count_l565_56551

noncomputable def num_circle_centers (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) : ℕ :=
  if (c = d) then 4 else 8

-- Here is the theorem statement
theorem circle_center_count (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) :
  num_circle_centers b c d h₁ h₂ = if (c = d) then 4 else 8 :=
sorry

end NUMINAMATH_GPT_circle_center_count_l565_56551


namespace NUMINAMATH_GPT_king_william_probability_l565_56558

theorem king_william_probability :
  let m := 2
  let n := 15
  m + n = 17 :=
by
  sorry

end NUMINAMATH_GPT_king_william_probability_l565_56558


namespace NUMINAMATH_GPT_inequality_solution_set_l565_56524

theorem inequality_solution_set (x : ℝ) :
  abs (1 + x + x^2 / 2) < 1 ↔ -2 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l565_56524


namespace NUMINAMATH_GPT_base_value_l565_56541

theorem base_value (b : ℕ) : (b - 1)^2 * (b - 2) = 256 → b = 17 :=
by
  sorry

end NUMINAMATH_GPT_base_value_l565_56541


namespace NUMINAMATH_GPT_find_remainder_l565_56505

theorem find_remainder (y : ℕ) (hy : 7 * y % 31 = 1) : (17 + 2 * y) % 31 = 4 :=
sorry

end NUMINAMATH_GPT_find_remainder_l565_56505


namespace NUMINAMATH_GPT_f_neg2_range_l565_56567

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x

theorem f_neg2_range (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2) (h2 : 2 ≤ f (1) ∧ f (1) ≤ 4) :
  ∀ k, f (-2) = k → 5 ≤ k ∧ k ≤ 10 :=
  sorry

end NUMINAMATH_GPT_f_neg2_range_l565_56567


namespace NUMINAMATH_GPT_part1_l565_56520

def is_Xn_function (n : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f x1 = f x2 ∧ x1 + x2 = 2 * n

theorem part1 : is_Xn_function 0 (fun x => abs x) ∧ is_Xn_function (1/2) (fun x => x^2 - x) :=
by
  sorry

end NUMINAMATH_GPT_part1_l565_56520


namespace NUMINAMATH_GPT_largest_angle_isosceles_triangle_l565_56500

theorem largest_angle_isosceles_triangle (A B C : ℕ) 
  (h_isosceles : A = B) 
  (h_base_angle : A = 50) : 
  max A (max B C) = 80 := 
by 
  -- proof is omitted  
  sorry

end NUMINAMATH_GPT_largest_angle_isosceles_triangle_l565_56500


namespace NUMINAMATH_GPT_oranges_in_buckets_l565_56556

theorem oranges_in_buckets :
  ∀ (x : ℕ),
  (22 + x + (x - 11) = 89) →
  (x - 22 = 17) :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_oranges_in_buckets_l565_56556


namespace NUMINAMATH_GPT_trigonometric_identity_l565_56577

theorem trigonometric_identity (θ : ℝ) (h : Real.tan (θ + Real.pi / 4) = 2) : 
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = -2 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l565_56577


namespace NUMINAMATH_GPT_four_brothers_money_l565_56536

theorem four_brothers_money 
  (a_1 a_2 a_3 a_4 : ℝ) 
  (x : ℝ)
  (h1 : a_1 + a_2 + a_3 + a_4 = 48)
  (h2 : a_1 + 3 = x)
  (h3 : a_2 - 3 = x)
  (h4 : 3 * a_3 = x)
  (h5 : a_4 / 3 = x) :
  a_1 = 6 ∧ a_2 = 12 ∧ a_3 = 3 ∧ a_4 = 27 :=
by
  sorry

end NUMINAMATH_GPT_four_brothers_money_l565_56536


namespace NUMINAMATH_GPT_length_of_goods_train_l565_56576

theorem length_of_goods_train 
  (speed_km_per_hr : ℕ) (platform_length_m : ℕ) (time_sec : ℕ) 
  (h1 : speed_km_per_hr = 72) (h2 : platform_length_m = 300) (h3 : time_sec = 26) : 
  ∃ length_of_train : ℕ, length_of_train = 220 :=
by
  sorry

end NUMINAMATH_GPT_length_of_goods_train_l565_56576


namespace NUMINAMATH_GPT_sufficient_not_necessary_range_l565_56547

theorem sufficient_not_necessary_range (x a : ℝ) : (∀ x, x < 1 → x < a) ∧ (∃ x, x < a ∧ ¬ (x < 1)) ↔ 1 < a := by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_range_l565_56547


namespace NUMINAMATH_GPT_M_inter_N_eq_M_l565_56575

def M : Set ℝ := {x | x > 1}
def N : Set ℝ := {y | y ≥ 1}

theorem M_inter_N_eq_M : M ∩ N = M := by
  sorry

end NUMINAMATH_GPT_M_inter_N_eq_M_l565_56575


namespace NUMINAMATH_GPT_rabbit_total_distance_l565_56560

theorem rabbit_total_distance 
  (r₁ r₂ : ℝ) 
  (h1 : r₁ = 7) 
  (h2 : r₂ = 15) 
  (q : ∀ (x : ℕ), x = 4) 
  : (3.5 * π + 8 + 7.5 * π + 8 + 3.5 * π + 8) = 14.5 * π + 24 := 
by
  sorry

end NUMINAMATH_GPT_rabbit_total_distance_l565_56560


namespace NUMINAMATH_GPT_a6_b6_gt_a4b2_ab4_l565_56506

theorem a6_b6_gt_a4b2_ab4 {a b : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  a^6 + b^6 > a^4 * b^2 + a^2 * b^4 :=
sorry

end NUMINAMATH_GPT_a6_b6_gt_a4b2_ab4_l565_56506


namespace NUMINAMATH_GPT_problem_quadratic_roots_l565_56568

theorem problem_quadratic_roots (m : ℝ) :
  (∀ x : ℝ, (m + 3) * x^2 - 4 * m * x + 2 * m - 1 = 0 →
    (∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ |x₁| > x₂)) ↔ -3 < m ∧ m < 0 :=
sorry

end NUMINAMATH_GPT_problem_quadratic_roots_l565_56568


namespace NUMINAMATH_GPT_race_completion_times_l565_56569

theorem race_completion_times :
  ∃ (Patrick Manu Amy Olivia Sophie Jack : ℕ),
  Patrick = 60 ∧
  Manu = Patrick + 12 ∧
  Amy = Manu / 2 ∧
  Olivia = (2 * Amy) / 3 ∧
  Sophie = Olivia - 10 ∧
  Jack = Sophie + 8 ∧
  Manu = 72 ∧
  Amy = 36 ∧
  Olivia = 24 ∧
  Sophie = 14 ∧
  Jack = 22 := 
by
  -- proof here
  sorry

end NUMINAMATH_GPT_race_completion_times_l565_56569


namespace NUMINAMATH_GPT_largest_valid_integer_l565_56525

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def satisfies_conditions (n : ℕ) : Prop :=
  (100 ≤ n ∧ n < 1000) ∧
  ∀ d ∈ n.digits 10, d ≠ 0 ∧ n % d = 0 ∧
  sum_of_digits n % 6 = 0

theorem largest_valid_integer : ∃ n : ℕ, satisfies_conditions n ∧ (∀ m : ℕ, satisfies_conditions m → m ≤ n) ∧ n = 936 :=
by
  sorry

end NUMINAMATH_GPT_largest_valid_integer_l565_56525


namespace NUMINAMATH_GPT_fraction_of_august_tips_l565_56503

variable {A : ℝ} -- A denotes the average monthly tips for the other months.
variable {total_tips_6_months : ℝ} (h1 : total_tips_6_months = 6 * A)
variable {august_tips : ℝ} (h2 : august_tips = 6 * A)
variable {total_tips : ℝ} (h3 : total_tips = total_tips_6_months + august_tips)

theorem fraction_of_august_tips (h1 : total_tips_6_months = 6 * A)
                                (h2 : august_tips = 6 * A)
                                (h3 : total_tips = total_tips_6_months + august_tips) :
    (august_tips / total_tips) = 1 / 2 :=
by
    sorry

end NUMINAMATH_GPT_fraction_of_august_tips_l565_56503


namespace NUMINAMATH_GPT_age_problem_l565_56590

open Classical

noncomputable def sum_cubes_ages (r j m : ℕ) : ℕ :=
  r^3 + j^3 + m^3

theorem age_problem (r j m : ℕ) (h1 : 5 * r + 2 * j = 3 * m)
    (h2 : 3 * m^2 + 2 * j^2 = 5 * r^2) (h3 : Nat.gcd r (Nat.gcd j m) = 1) :
    sum_cubes_ages r j m = 3 := by
  sorry

end NUMINAMATH_GPT_age_problem_l565_56590


namespace NUMINAMATH_GPT_ant_crawling_routes_ratio_l565_56507

theorem ant_crawling_routes_ratio 
  (m n : ℕ) 
  (h1 : m = 2) 
  (h2 : n = 6) : 
  n / m = 3 :=
by
  -- Proof is omitted (we only need the statement as per the instruction)
  sorry

end NUMINAMATH_GPT_ant_crawling_routes_ratio_l565_56507


namespace NUMINAMATH_GPT_find_a_l565_56522

variable {x a : ℝ}

def A (x : ℝ) : Prop := x ≤ -1 ∨ x > 2
def B (x a : ℝ) : Prop := x < a ∨ x > a + 1

theorem find_a (hA : ∀ x, (x + 1) / (x - 2) ≥ 0 ↔ A x)
                (hB : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a > 0 ↔ B x a)
                (hSub : ∀ x, A x → B x a) :
  -1 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_find_a_l565_56522


namespace NUMINAMATH_GPT_age_of_youngest_child_l565_56546

theorem age_of_youngest_child (mother_fee : ℝ) (child_fee_per_year : ℝ) 
  (total_fee : ℝ) (t : ℝ) (y : ℝ) (child_fee : ℝ)
  (h_mother_fee : mother_fee = 2.50)
  (h_child_fee_per_year : child_fee_per_year = 0.25)
  (h_total_fee : total_fee = 4.00)
  (h_child_fee : child_fee = total_fee - mother_fee)
  (h_y : y = 6 - 2 * t)
  (h_fee_eq : child_fee = y * child_fee_per_year) : y = 2 := 
by
  sorry

end NUMINAMATH_GPT_age_of_youngest_child_l565_56546


namespace NUMINAMATH_GPT_sum_of_squares_is_149_l565_56588

-- Define the integers and their sum and product
def integers_sum (b : ℤ) : ℤ := (b - 1) + b + (b + 1)
def integers_product (b : ℤ) : ℤ := (b - 1) * b * (b + 1)

-- Define the condition given in the problem
def condition (b : ℤ) : Prop :=
  integers_product b = 12 * integers_sum b + b^2

-- Define the sum of squares of three consecutive integers
def sum_of_squares (b : ℤ) : ℤ :=
  (b - 1)^2 + b^2 + (b + 1)^2

-- The main statement to be proved
theorem sum_of_squares_is_149 (b : ℤ) (h : condition b) : sum_of_squares b = 149 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_is_149_l565_56588


namespace NUMINAMATH_GPT_mark_and_alice_probability_l565_56544

def probability_sunny_days : ℚ := 51 / 250

theorem mark_and_alice_probability :
  (∀ (day : ℕ), day < 5 → (∃ rain_prob sun_prob : ℚ, rain_prob = 0.8 ∧ sun_prob = 0.2 ∧ rain_prob + sun_prob = 1))
  → probability_sunny_days = 51 / 250 :=
by sorry

end NUMINAMATH_GPT_mark_and_alice_probability_l565_56544
