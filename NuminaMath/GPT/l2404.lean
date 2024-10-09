import Mathlib

namespace factorization_of_cubic_polynomial_l2404_240406

-- Define the elements and the problem
variable (a : ℝ)

theorem factorization_of_cubic_polynomial :
  a^3 - 3 * a = a * (a + Real.sqrt 3) * (a - Real.sqrt 3) := by
  sorry

end factorization_of_cubic_polynomial_l2404_240406


namespace sufficient_not_necessary_l2404_240465

theorem sufficient_not_necessary (a : ℝ) : (a > 1 → 1 / a < 1) ∧ (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
by
  sorry

end sufficient_not_necessary_l2404_240465


namespace number_of_foals_l2404_240499

theorem number_of_foals (t f : ℕ) (h1 : t + f = 11) (h2 : 2 * t + 4 * f = 30) : f = 4 :=
by
  sorry

end number_of_foals_l2404_240499


namespace min_colors_rect_condition_l2404_240442

theorem min_colors_rect_condition (n : ℕ) (hn : n ≥ 2) :
  ∃ k : ℕ, (∀ (coloring : Fin n → Fin n → Fin k), 
           (∀ i j, coloring i j < k) → 
           (∀ c, ∃ i j, coloring i j = c) →
           (∃ i1 i2 j1 j2, i1 ≠ i2 ∧ j1 ≠ j2 ∧ 
                            coloring i1 j1 ≠ coloring i1 j2 ∧ 
                            coloring i1 j1 ≠ coloring i2 j1 ∧ 
                            coloring i1 j2 ≠ coloring i2 j2 ∧ 
                            coloring i2 j1 ≠ coloring i2 j2)) → 
           k = 2 * n :=
sorry

end min_colors_rect_condition_l2404_240442


namespace convert_deg_to_rad_l2404_240412

theorem convert_deg_to_rad (deg_to_rad : ℝ → ℝ) (conversion_factor : deg_to_rad 1 = π / 180) :
  deg_to_rad (-300) = - (5 * π) / 3 :=
by
  sorry

end convert_deg_to_rad_l2404_240412


namespace complex_number_z_l2404_240483

theorem complex_number_z (i : ℂ) (z : ℂ) (hi : i * i = -1) (h : 2 * i / z = 1 - i) : z = -1 + i :=
by
  sorry

end complex_number_z_l2404_240483


namespace range_of_a_l2404_240405

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0 → -1 < a ∧ a < 3 :=
by
  intro h
  sorry

end range_of_a_l2404_240405


namespace cos_555_value_l2404_240492

noncomputable def cos_555_equals_neg_sqrt6_add_sqrt2_div4 : Prop :=
  (Real.cos 555 = -((Real.sqrt 6 + Real.sqrt 2) / 4))

theorem cos_555_value : cos_555_equals_neg_sqrt6_add_sqrt2_div4 :=
  by sorry

end cos_555_value_l2404_240492


namespace original_population_l2404_240432

theorem original_population (P : ℕ) (h1 : 0.1 * (P : ℝ) + 0.2 * (0.9 * P) = 4500) : P = 6250 :=
sorry

end original_population_l2404_240432


namespace solution_set_correct_l2404_240458

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then 2^(-x) - 4 else 2^(x) - 4

theorem solution_set_correct : 
  (∀ x, f x = f |x|) → 
  (∀ x, f x = 2^(-x) - 4 ∨ f x = 2^(x) - 4) → 
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  intro h1 h2
  sorry

end solution_set_correct_l2404_240458


namespace average_marks_of_first_class_l2404_240490

theorem average_marks_of_first_class (n1 n2 : ℕ) (avg2 avg_all : ℝ)
  (h_n1 : n1 = 25) (h_n2 : n2 = 40) (h_avg2 : avg2 = 65) (h_avg_all : avg_all = 59.23076923076923) :
  ∃ (A : ℝ), A = 50 :=
by 
  sorry

end average_marks_of_first_class_l2404_240490


namespace difference_between_largest_and_smallest_l2404_240454

def largest_number := 9765310
def smallest_number := 1035679
def expected_difference := 8729631
def digits := [3, 9, 6, 0, 5, 1, 7]

theorem difference_between_largest_and_smallest :
  (largest_number - smallest_number) = expected_difference :=
sorry

end difference_between_largest_and_smallest_l2404_240454


namespace expression_inside_absolute_value_l2404_240419

theorem expression_inside_absolute_value (E : ℤ) (x : ℤ) (h1 : x = 10) (h2 : 30 - |E| = 26) :
  E = 4 ∨ E = -4 := 
by
  sorry

end expression_inside_absolute_value_l2404_240419


namespace one_div_a_plus_one_div_b_l2404_240430

theorem one_div_a_plus_one_div_b (a b : ℝ) (h₀ : a ≠ b) (ha : a^2 - 3 * a + 2 = 0) (hb : b^2 - 3 * b + 2 = 0) :
  1 / a + 1 / b = 3 / 2 :=
by
  -- Proof goes here
  sorry

end one_div_a_plus_one_div_b_l2404_240430


namespace brochures_per_box_l2404_240404

theorem brochures_per_box (total_brochures : ℕ) (boxes : ℕ) 
  (htotal : total_brochures = 5000) (hboxes : boxes = 5) : 
  (1000 / 5000 : ℚ) = 1 / 5 := 
by sorry

end brochures_per_box_l2404_240404


namespace number_of_lines_dist_l2404_240425

theorem number_of_lines_dist {A B : ℝ × ℝ} (hA : A = (3, 0)) (hB : B = (0, 4)) : 
  ∃ n : ℕ, n = 3 ∧
  ∀ l : ℝ → ℝ → Prop, 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ A → dist A p = 2) ∧ 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ B → dist B p = 3) → n = 3 := 
by sorry

end number_of_lines_dist_l2404_240425


namespace find_n_interval_l2404_240459

theorem find_n_interval :
  ∃ n : ℕ, n < 1000 ∧
  (∃ ghijkl : ℕ, (ghijkl < 999999) ∧ (ghijkl * n = 999999 * ghijkl)) ∧
  (∃ mnop : ℕ, (mnop < 9999) ∧ (mnop * (n + 5) = 9999 * mnop)) ∧
  151 ≤ n ∧ n ≤ 300 :=
sorry

end find_n_interval_l2404_240459


namespace square_side_increase_l2404_240443

variable (s : ℝ)  -- original side length of the square.
variable (p : ℝ)  -- percentage increase of the side length.

theorem square_side_increase (h1 : (s * (1 + p / 100))^2 = 1.21 * s^2) : p = 10 := 
by
  sorry

end square_side_increase_l2404_240443


namespace average_apples_sold_per_day_l2404_240417

theorem average_apples_sold_per_day (boxes_sold : ℕ) (days : ℕ) (apples_per_box : ℕ) (H1 : boxes_sold = 12) (H2 : days = 4) (H3 : apples_per_box = 25) : (boxes_sold * apples_per_box) / days = 75 :=
by {
  -- Based on given conditions, the total apples sold is 12 * 25 = 300.
  -- Dividing by the number of days, 300 / 4 gives us 75 apples/day.
  -- The proof is omitted as instructed.
  sorry
}

end average_apples_sold_per_day_l2404_240417


namespace value_of_x_is_4_l2404_240453

variable {A B C D E F G H P : ℕ}

theorem value_of_x_is_4 (h1 : 5 + A + B = 19)
                        (h2 : A + B + C = 19)
                        (h3 : C + D + E = 19)
                        (h4 : D + E + F = 19)
                        (h5 : F + x + G = 19)
                        (h6 : x + G + H = 19)
                        (h7 : H + P + 10 = 19) :
                        x = 4 :=
by
  sorry

end value_of_x_is_4_l2404_240453


namespace intersection_height_correct_l2404_240456

noncomputable def intersection_height 
  (height_pole_1 height_pole_2 distance : ℝ) : ℝ := 
  let slope_1 := -(height_pole_1 / distance)
  let slope_2 := height_pole_2 / distance
  let y_intercept_1 := height_pole_1
  let y_intercept_2 := 0
  let x_intersection := height_pole_1 / (slope_2 - slope_1)
  let y_intersection := slope_2 * x_intersection + y_intercept_2
  y_intersection

theorem intersection_height_correct 
  : intersection_height 30 90 150 = 22.5 := 
by sorry

end intersection_height_correct_l2404_240456


namespace permutation_equals_power_l2404_240476

-- Definition of permutation with repetition
def permutation_with_repetition (n k : ℕ) : ℕ := n ^ k

-- Theorem to prove
theorem permutation_equals_power (n k : ℕ) : permutation_with_repetition n k = n ^ k :=
by
  sorry

end permutation_equals_power_l2404_240476


namespace no_real_roots_of_quadratic_l2404_240438

theorem no_real_roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = 1 ∧ c = 1) :
  (b^2 - 4 * a * c < 0) → ¬∃ x : ℝ, a * x^2 + b * x + c = 0 := by
  sorry

end no_real_roots_of_quadratic_l2404_240438


namespace solution_set_inequality_system_l2404_240409

theorem solution_set_inequality_system (x : ℝ) :
  (x - 3 < 2 ∧ 3 * x + 1 ≥ 2 * x) ↔ (-1 ≤ x ∧ x < 5) := by
  sorry

end solution_set_inequality_system_l2404_240409


namespace fraction_pow_zero_l2404_240428

theorem fraction_pow_zero :
  (4310000 / -21550000 : ℝ) ≠ 0 →
  (4310000 / -21550000 : ℝ) ^ 0 = 1 :=
by
  intro h
  sorry

end fraction_pow_zero_l2404_240428


namespace mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l2404_240475

noncomputable def ratio_of_A_students (total_students_A : ℕ) (A_students_A : ℕ) : ℚ :=
  A_students_A / total_students_A

theorem mrs_berkeley_A_students_first_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 18 →
    (A_students_A / total_students_A) * total_students_B = 12 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

theorem mrs_berkeley_A_students_extended_class :
  ∀ (total_students_A : ℕ) (A_students_A : ℕ) (total_students_B : ℕ),
    total_students_A = 30 →
    A_students_A = 20 →
    total_students_B = 27 →
    (A_students_A / total_students_A) * total_students_B = 18 :=
by
  intros total_students_A A_students_A total_students_B hA1 hA2 hB
  sorry

end mrs_berkeley_A_students_first_class_mrs_berkeley_A_students_extended_class_l2404_240475


namespace dividend_is_correct_l2404_240418

def divisor : ℕ := 17
def quotient : ℕ := 9
def remainder : ℕ := 6

def calculate_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem dividend_is_correct : calculate_dividend divisor quotient remainder = 159 :=
  by sorry

end dividend_is_correct_l2404_240418


namespace units_digit_24_pow_4_plus_42_pow_4_l2404_240433

theorem units_digit_24_pow_4_plus_42_pow_4 : 
    (24^4 + 42^4) % 10 = 2 :=
by
  sorry

end units_digit_24_pow_4_plus_42_pow_4_l2404_240433


namespace line_intersects_circle_l2404_240435

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, y = k * (x - 1) ∧ x^2 + y^2 = 1 :=
by
  sorry

end line_intersects_circle_l2404_240435


namespace probability_of_event_a_l2404_240461

-- Given conditions and question
variables (a b : Prop)
variables (p : Prop → ℝ)

-- Given conditions
axiom p_a : p a = 4 / 5
axiom p_b : p b = 2 / 5
axiom p_a_and_b_given : p (a ∧ b) = 0.32
axiom independent_a_b : p (a ∧ b) = p a * p b

-- The proof statement we need to prove: p a = 0.8
theorem probability_of_event_a :
  p a = 0.8 :=
sorry

end probability_of_event_a_l2404_240461


namespace correct_statements_about_f_l2404_240452

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem correct_statements_about_f : 
  (∀ x, (f x) ≤ (f e)) ∧ (f e = 1 / e) ∧ 
  (∀ x, (f x = 0) → x = 1) ∧ 
  (f 2 < f π ∧ f π < f 3) :=
by
  sorry

end correct_statements_about_f_l2404_240452


namespace roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l2404_240498

theorem roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells 
  (k n : ℕ) (h_k : k = 4) (h_n : n = 100)
  (shift_rule : ∀ (m : ℕ), m ≤ n → 
    ∃ (chips_moved : ℕ), chips_moved = 1 ∧ chips_moved ≤ m) 
  : ∃ m, m ≤ n ∧ m = 50 := 
by
  sorry

end roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l2404_240498


namespace cost_of_second_type_of_rice_is_22_l2404_240450

noncomputable def cost_second_type_of_rice (c1 : ℝ) (w1 : ℝ) (w2 : ℝ) (avg : ℝ) (total_weight : ℝ) : ℝ :=
  ((total_weight * avg) - (w1 * c1)) / w2

theorem cost_of_second_type_of_rice_is_22 :
  cost_second_type_of_rice 16 8 4 18 12 = 22 :=
by
  sorry

end cost_of_second_type_of_rice_is_22_l2404_240450


namespace aquarium_final_volume_l2404_240469

theorem aquarium_final_volume :
  let length := 4
  let width := 6
  let height := 3
  let total_volume := length * width * height
  let initial_volume := total_volume / 2
  let spilled_volume := initial_volume / 2
  let remaining_volume := initial_volume - spilled_volume
  let final_volume := remaining_volume * 3
  final_volume = 54 :=
by sorry

end aquarium_final_volume_l2404_240469


namespace board_arithmetic_impossibility_l2404_240416

theorem board_arithmetic_impossibility :
  ¬ (∃ (a b : ℕ), a ≡ 0 [MOD 7] ∧ b ≡ 1 [MOD 7] ∧ (a * b + a^3 + b^3) = 2013201420152016) := 
    sorry

end board_arithmetic_impossibility_l2404_240416


namespace remaining_speed_l2404_240480
open Real

theorem remaining_speed
  (D T : ℝ) (h1 : 40 * (T / 3) = (2 / 3) * D)
  (h2 : (T / 3) * 3 = T) :
  (D / 3) / ((2 * ((2 / 3) * D) / (40) / (3)) * 2 / 3) = 10 :=
by
  sorry

end remaining_speed_l2404_240480


namespace soccer_ball_cost_l2404_240484

theorem soccer_ball_cost :
  ∃ x y : ℝ, x + y = 100 ∧ 2 * x + 3 * y = 262 ∧ x = 38 :=
by
  sorry

end soccer_ball_cost_l2404_240484


namespace expansion_identity_l2404_240470

theorem expansion_identity : 121 + 2 * 11 * 9 + 81 = 400 := by
  sorry

end expansion_identity_l2404_240470


namespace chelsea_total_time_l2404_240403

def num_batches := 4
def bake_time_per_batch := 20  -- minutes
def ice_time_per_batch := 30   -- minutes
def cupcakes_per_batch := 6
def additional_time_first_batch := 10 -- per cupcake
def additional_time_second_batch := 15 -- per cupcake
def additional_time_third_batch := 12 -- per cupcake
def additional_time_fourth_batch := 20 -- per cupcake

def total_bake_ice_time := bake_time_per_batch + ice_time_per_batch
def total_bake_ice_time_all_batches := total_bake_ice_time * num_batches

def total_additional_time_first_batch := additional_time_first_batch * cupcakes_per_batch
def total_additional_time_second_batch := additional_time_second_batch * cupcakes_per_batch
def total_additional_time_third_batch := additional_time_third_batch * cupcakes_per_batch
def total_additional_time_fourth_batch := additional_time_fourth_batch * cupcakes_per_batch

def total_additional_time := 
  total_additional_time_first_batch +
  total_additional_time_second_batch +
  total_additional_time_third_batch +
  total_additional_time_fourth_batch

def total_time := total_bake_ice_time_all_batches + total_additional_time

theorem chelsea_total_time : total_time = 542 := by
  sorry

end chelsea_total_time_l2404_240403


namespace lines_not_form_triangle_l2404_240437

theorem lines_not_form_triangle {m : ℝ} :
  (∀ x y : ℝ, 2 * x - 3 * y + 1 ≠ 0 → 4 * x + 3 * y + 5 ≠ 0 → mx - y - 1 ≠ 0) →
  (m = -4 / 3 ∨ m = 2 / 3 ∨ m = 4 / 3) :=
sorry

end lines_not_form_triangle_l2404_240437


namespace overall_labor_costs_l2404_240486

noncomputable def construction_worker_daily_wage : ℝ := 100
noncomputable def electrician_daily_wage : ℝ := 2 * construction_worker_daily_wage
noncomputable def plumber_daily_wage : ℝ := 2.5 * construction_worker_daily_wage

noncomputable def total_construction_work : ℝ := 2 * construction_worker_daily_wage
noncomputable def total_electrician_work : ℝ := electrician_daily_wage
noncomputable def total_plumber_work : ℝ := plumber_daily_wage

theorem overall_labor_costs :
  total_construction_work + total_electrician_work + total_plumber_work = 650 :=
by
  sorry

end overall_labor_costs_l2404_240486


namespace circle_center_coordinates_l2404_240482

theorem circle_center_coordinates :
  ∀ x y, (x^2 + y^2 - 4 * x - 2 * y - 5 = 0) → (x, y) = (2, 1) :=
by
  sorry

end circle_center_coordinates_l2404_240482


namespace age_difference_l2404_240463

/-- The age difference between each child d -/
theorem age_difference (d : ℝ) 
  (h1 : ∃ a b c e : ℝ, d = a ∧ 2*d = b ∧ 3*d = c ∧ 4*d = e)
  (h2 : 12 + (12 - d) + (12 - 2*d) + (12 - 3*d) + (12 - 4*d) = 40) : 
  d = 2 := 
sorry

end age_difference_l2404_240463


namespace no_quaint_two_digit_integers_l2404_240434

theorem no_quaint_two_digit_integers :
  ∀ x : ℕ, 10 ≤ x ∧ x < 100 ∧ (∃ a b : ℕ, x = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9) →  ¬(10 * x.div 10 + x % 10 = (x.div 10) + (x % 10)^3) :=
by
  sorry

end no_quaint_two_digit_integers_l2404_240434


namespace fraction_zero_l2404_240467

theorem fraction_zero (x : ℝ) (h : (x^2 - 1) / (x + 1) = 0) : x = 1 := 
sorry

end fraction_zero_l2404_240467


namespace least_positive_integer_congruences_l2404_240464

theorem least_positive_integer_congruences :
  ∃ n : ℕ, 
    n > 0 ∧ 
    (n % 4 = 1) ∧ 
    (n % 5 = 2) ∧ 
    (n % 6 = 3) ∧ 
    (n = 57) :=
by
  sorry

end least_positive_integer_congruences_l2404_240464


namespace work_completion_l2404_240407

theorem work_completion (a b : ℕ) (hab : a = 2 * b) (hwork_together : (1/a + 1/b) = 1/8) : b = 24 := by
  sorry

end work_completion_l2404_240407


namespace clothing_price_decrease_l2404_240457

theorem clothing_price_decrease (P : ℝ) (h₁ : P > 0) :
  let price_first_sale := (4 / 5) * P
  let price_second_sale := (1 / 2) * P
  let price_difference := price_first_sale - price_second_sale
  let percent_decrease := (price_difference / price_first_sale) * 100
  percent_decrease = 37.5 :=
by
  sorry

end clothing_price_decrease_l2404_240457


namespace find_value_of_A_l2404_240448

theorem find_value_of_A (x y A : ℝ)
  (h1 : 2^x = A)
  (h2 : 7^(2*y) = A)
  (h3 : 1 / x + 2 / y = 2) : 
  A = 7 * Real.sqrt 2 := 
sorry

end find_value_of_A_l2404_240448


namespace jamie_dimes_l2404_240451

theorem jamie_dimes (y : ℕ) (h : 5 * y + 10 * y + 25 * y = 1440) : y = 36 :=
by 
  sorry

end jamie_dimes_l2404_240451


namespace tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l2404_240449

-- Definitions
variables {α β : ℝ}

-- Condition: 0 < α < π / 2
def valid_alpha (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Condition: sin α = 4 / 5
def sin_alpha (α : ℝ) : Prop := Real.sin α = 4 / 5

-- Condition: 0 < β < π / 2
def valid_beta (β : ℝ) : Prop := 0 < β ∧ β < Real.pi / 2

-- Condition: cos (α + β) = -1 / 2
def cos_alpha_add_beta (α β : ℝ) : Prop := Real.cos (α + β) = - 1 / 2

/-- Proofs begin -/
-- Proof for tan α = 4 / 3 given 0 < α < π / 2 and sin α = 4 / 5
theorem tan_alpha_eq (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : Real.tan α = 4 / 3 := 
  sorry

-- Proof for cos (2α + π / 4) = -31√2 / 50 given 0 < α < π / 2 and sin α = 4 / 5
theorem cos_two_alpha_plus_quarter_pi (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : 
  Real.cos (2 * α + Real.pi / 4) = -31 * Real.sqrt 2 / 50 := 
  sorry

-- Proof for sin β = 4 + 3√3 / 10 given 0 < α < π / 2, sin α = 4 / 5, 0 < β < π / 2 and cos (α + β) = -1 / 2
theorem sin_beta_eq (α β : ℝ) (h_validα : valid_alpha α) (h_sinα : sin_alpha α) 
  (h_validβ : valid_beta β) (h_cosαβ : cos_alpha_add_beta α β) : Real.sin β = 4 + 3 * Real.sqrt 3 / 10 := 
  sorry

end tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l2404_240449


namespace sara_spent_on_movies_l2404_240429

def cost_of_movie_tickets : ℝ := 2 * 10.62
def cost_of_rented_movie : ℝ := 1.59
def cost_of_purchased_movie : ℝ := 13.95

theorem sara_spent_on_movies :
  cost_of_movie_tickets + cost_of_rented_movie + cost_of_purchased_movie = 36.78 := by
  sorry

end sara_spent_on_movies_l2404_240429


namespace total_transportation_cost_l2404_240424

def weights_in_grams : List ℕ := [300, 450, 600]
def cost_per_kg : ℕ := 15000

def convert_to_kg (w : ℕ) : ℚ :=
  w / 1000

def calculate_cost (weight_in_kg : ℚ) (cost_per_kg : ℕ) : ℚ :=
  weight_in_kg * cost_per_kg

def total_cost (weights_in_grams : List ℕ) (cost_per_kg : ℕ) : ℚ :=
  weights_in_grams.map (λ w => calculate_cost (convert_to_kg w) cost_per_kg) |>.sum

theorem total_transportation_cost :
  total_cost weights_in_grams cost_per_kg = 20250 := by
  sorry

end total_transportation_cost_l2404_240424


namespace find_value_of_expression_l2404_240472

variable {x : ℝ}

theorem find_value_of_expression (h : x^2 - 2 * x = 3) : 3 * x^2 - 6 * x - 4 = 5 :=
sorry

end find_value_of_expression_l2404_240472


namespace store_A_more_advantageous_l2404_240479

theorem store_A_more_advantageous (x : ℕ) (h : x > 5) : 
  6000 + 4500 * (x - 1) < 4800 * x := 
by 
  sorry

end store_A_more_advantageous_l2404_240479


namespace fractional_equation_no_solution_l2404_240497

theorem fractional_equation_no_solution (x : ℝ) (h1 : x ≠ 3) : (2 - x) / (x - 3) ≠ 1 + 1 / (3 - x) :=
by
  sorry

end fractional_equation_no_solution_l2404_240497


namespace robert_finite_moves_l2404_240414

noncomputable def onlyFiniteMoves (numbers : List ℕ) : Prop :=
  ∀ (a b : ℕ), a > b → ∃ (moves : ℕ), moves < numbers.length

theorem robert_finite_moves (numbers : List ℕ) :
  onlyFiniteMoves numbers := sorry

end robert_finite_moves_l2404_240414


namespace find_m_value_l2404_240413

def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem find_m_value :
  ∃ m : ℝ, (∀ x ∈ (Set.Icc 0 3), f x m ≤ 1) ∧ (∃ x ∈ (Set.Icc 0 3), f x m = 1) ↔ m = -2 :=
by
  sorry

end find_m_value_l2404_240413


namespace kitchen_upgrade_cost_l2404_240478

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_cost_l2404_240478


namespace blue_balls_unchanged_l2404_240473

def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5
def added_yellow_balls : ℕ := 4

theorem blue_balls_unchanged :
  initial_blue_balls = 2 := by
  sorry

end blue_balls_unchanged_l2404_240473


namespace pieces_of_wood_for_chair_is_correct_l2404_240427

-- Define the initial setup and constants
def total_pieces_of_wood := 672
def pieces_of_wood_per_table := 12
def number_of_tables := 24
def number_of_chairs := 48

-- Calculation in the conditions
def pieces_of_wood_used_for_tables := number_of_tables * pieces_of_wood_per_table
def pieces_of_wood_left_for_chairs := total_pieces_of_wood - pieces_of_wood_used_for_tables

-- Question and answer verification
def pieces_of_wood_per_chair := pieces_of_wood_left_for_chairs / number_of_chairs

theorem pieces_of_wood_for_chair_is_correct :
  pieces_of_wood_per_chair = 8 := 
by
  -- Proof omitted
  sorry

end pieces_of_wood_for_chair_is_correct_l2404_240427


namespace solution_set_of_quadratic_inequality_l2404_240422

theorem solution_set_of_quadratic_inequality 
  (a b c x₁ x₂ : ℝ)
  (h1 : a > 0) 
  (h2 : a * x₁^2 + b * x₁ + c = 0)
  (h3 : a * x₂^2 + b * x₂ + c = 0)
  : {x : ℝ | a * x^2 + b * x + c > 0} = ({x : ℝ | x > x₁} ∩ {x : ℝ | x > x₂}) ∪ ({x : ℝ | x < x₁} ∩ {x : ℝ | x < x₂}) :=
sorry

end solution_set_of_quadratic_inequality_l2404_240422


namespace calculate_total_weight_AlBr3_l2404_240445

-- Definitions for the atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90

-- Definition for the molecular weight of AlBr3
def molecular_weight_AlBr3 : ℝ := atomic_weight_Al + 3 * atomic_weight_Br

-- Number of moles
def number_of_moles : ℝ := 5

-- Total weight of 5 moles of AlBr3
def total_weight_5_moles_AlBr3 : ℝ := molecular_weight_AlBr3 * number_of_moles

-- Desired result
def expected_total_weight : ℝ := 1333.40

-- Statement to prove that total_weight_5_moles_AlBr3 equals the expected total weight
theorem calculate_total_weight_AlBr3 :
  total_weight_5_moles_AlBr3 = expected_total_weight :=
sorry

end calculate_total_weight_AlBr3_l2404_240445


namespace polynomial_condition_degree_n_l2404_240466

open Polynomial

theorem polynomial_condition_degree_n 
  (P_n : ℤ[X]) (n : ℕ) (hn_pos : 0 < n) (hn_deg : P_n.degree = n) 
  (hx0 : P_n.eval 0 = 0)
  (hx_conditions : ∃ (a : ℤ) (b : Fin n → ℤ), ∀ i, P_n.eval (b i) = n) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 := 
sorry

end polynomial_condition_degree_n_l2404_240466


namespace largest_A_divisible_by_8_l2404_240423

theorem largest_A_divisible_by_8 (A B C : ℕ) (h1 : A = 8 * B + C) (h2 : B = C) (h3 : C < 8) : A ≤ 9 * 7 :=
by sorry

end largest_A_divisible_by_8_l2404_240423


namespace work_problem_l2404_240496

theorem work_problem (W : ℝ) (A_rate : ℝ) (AB_rate : ℝ) : A_rate = W / 14 ∧ AB_rate = W / 10 → 1 / (AB_rate - A_rate) = 35 :=
by
  sorry

end work_problem_l2404_240496


namespace man_l2404_240421

-- Define the man's rowing speed in still water, the speed of the current, the downstream speed and headwind reduction.
def v : Real := 17.5
def speed_current : Real := 4.5
def speed_downstream : Real := 22
def headwind_reduction : Real := 1.5

-- Define the man's speed against the current and headwind.
def speed_against_current_headwind := v - speed_current - headwind_reduction

-- The statement to prove. 
theorem man's_speed_against_current_and_headwind :
  speed_against_current_headwind = 11.5 := by
  -- Using the conditions (which are already defined in lean expressions above), we can end the proof here.
  sorry

end man_l2404_240421


namespace games_in_tournament_l2404_240462

def single_elimination_games (n : Nat) : Nat :=
  n - 1

theorem games_in_tournament : single_elimination_games 24 = 23 := by
  sorry

end games_in_tournament_l2404_240462


namespace find_starting_number_l2404_240436

theorem find_starting_number (n : ℕ) (h : ((28 + n) / 2) = 18) : n = 8 :=
sorry

end find_starting_number_l2404_240436


namespace Larry_wins_probability_l2404_240447

noncomputable def probability_Larry_wins (p_larry: ℚ) (p_paul: ℚ): ℚ :=
  let q_larry := 1 - p_larry
  let q_paul := 1 - p_paul
  p_larry / (1 - q_larry * q_paul)

theorem Larry_wins_probability:
  probability_Larry_wins (1/3 : ℚ) (1/2 : ℚ) = (2/5 : ℚ) :=
by {
  sorry
}

end Larry_wins_probability_l2404_240447


namespace min_w_value_l2404_240471

def w (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 + 12 * x - 10 * y + 45

theorem min_w_value : ∀ x y : ℝ, (w x y) ≥ 28 ∧ (∃ x y : ℝ, (w x y) = 28) :=
by
  sorry

end min_w_value_l2404_240471


namespace original_deck_total_l2404_240426

theorem original_deck_total (b y : ℕ) 
    (h1 : (b : ℚ) / (b + y) = 2 / 5)
    (h2 : (b : ℚ) / (b + y + 6) = 5 / 14) :
    b + y = 50 := by
  sorry

end original_deck_total_l2404_240426


namespace stationery_sales_l2404_240431

theorem stationery_sales :
  let pen_percentage : ℕ := 42
  let pencil_percentage : ℕ := 27
  let total_sales_percentage : ℕ := 100
  total_sales_percentage - (pen_percentage + pencil_percentage) = 31 :=
by
  sorry

end stationery_sales_l2404_240431


namespace two_digit_number_representation_l2404_240420

theorem two_digit_number_representation (a b : ℕ) (ha : a < 10) (hb : b < 10) : 10 * b + a = d :=
  sorry

end two_digit_number_representation_l2404_240420


namespace least_possible_value_of_b_plus_c_l2404_240493

theorem least_possible_value_of_b_plus_c :
  ∃ (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (∃ (r1 r2 : ℝ), r1 - r2 = 30 ∧ 2 * r1 ^ 2 + b * r1 + c = 0 ∧ 2 * r2 ^ 2 + b * r2 + c = 0) ∧ b + c = 126 := 
by
  sorry 

end least_possible_value_of_b_plus_c_l2404_240493


namespace ratio_equivalence_l2404_240491

theorem ratio_equivalence (x : ℝ) (h : 3 / x = 3 / 16) : x = 16 := 
by
  sorry

end ratio_equivalence_l2404_240491


namespace find_m_range_l2404_240400

def p (m : ℝ) : Prop := (4 - 4 * m) ≤ 0
def q (m : ℝ) : Prop := (5 - 2 * m) > 1

theorem find_m_range (m : ℝ) (hp_false : ¬ p m) (hq_true : q m) : 1 ≤ m ∧ m < 2 :=
by {
 sorry
}

end find_m_range_l2404_240400


namespace frequency_of_group_of_samples_l2404_240440

def sample_capacity : ℝ := 32
def frequency_rate : ℝ := 0.125

theorem frequency_of_group_of_samples : frequency_rate * sample_capacity = 4 :=
by 
  sorry

end frequency_of_group_of_samples_l2404_240440


namespace daily_wage_c_l2404_240446

-- Definitions according to the conditions
def days_worked_a : ℕ := 6
def days_worked_b : ℕ := 9
def days_worked_c : ℕ := 4

def ratio_wages : ℕ × ℕ × ℕ := (3, 4, 5)
def total_earning : ℕ := 1628

-- Goal: Prove that the daily wage of c is Rs. 110
theorem daily_wage_c : (5 * (total_earning / (18 + 36 + 20))) = 110 :=
by
  sorry

end daily_wage_c_l2404_240446


namespace sum_ratio_l2404_240485

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ := 
  a1 * (1 - q^n) / (1 - q)

theorem sum_ratio (a1 q : ℝ) 
  (h : 8 * (a1 * q) + (a1 * q^4) = 0) :
  geometric_sequence_sum a1 q 6 / geometric_sequence_sum a1 q 3 = -7 := 
by
  sorry

end sum_ratio_l2404_240485


namespace tangent_line_eqn_l2404_240474

noncomputable def f (x : ℝ) : ℝ := 5 * x + Real.log x

theorem tangent_line_eqn : ∀ x y : ℝ, (x, y) = (1, f 1) → 6 * x - y - 1 = 0 := 
by
  intro x y h
  sorry

end tangent_line_eqn_l2404_240474


namespace product_of_t_values_l2404_240402

theorem product_of_t_values (t : ℝ) (h : t^2 = 49) : (7 * (-7) = -49) := sorry

end product_of_t_values_l2404_240402


namespace Sam_wins_probability_l2404_240488

-- Define the basic probabilities
def prob_hit : ℚ := 2 / 5
def prob_miss : ℚ := 3 / 5

-- Define the desired probability that Sam wins
noncomputable def p : ℚ := 5 / 8

-- The mathematical problem statement in Lean
theorem Sam_wins_probability :
  p = prob_hit + (prob_miss * prob_miss * p) := 
sorry

end Sam_wins_probability_l2404_240488


namespace range_of_x0_l2404_240487

noncomputable def point_on_circle_and_line (x0 : ℝ) (y0 : ℝ) : Prop :=
(x0^2 + y0^2 = 1) ∧ (3 * x0 + 2 * y0 = 4)

theorem range_of_x0 
  (x0 : ℝ) (y0 : ℝ) 
  (h1 : 3 * x0 + 2 * y0 = 4)
  (h2 : ∃ A B : ℝ × ℝ, (A.1^2 + A.2^2 = 1) ∧ (B.1^2 + B.2^2 = 1) ∧ (A ≠ B) ∧ (A + B = (x0, y0))) :
  0 < x0 ∧ x0 < 24 / 13 :=
sorry

end range_of_x0_l2404_240487


namespace fraction_scaled_l2404_240477

theorem fraction_scaled (x y : ℝ) :
  ∃ (k : ℝ), (k = 3 * y) ∧ ((5 * x + 3 * y) / (x + 3 * y) = 5 * ((x + (3 * y)) / (x + (3 * y)))) := 
  sorry

end fraction_scaled_l2404_240477


namespace geom_seq_general_term_sum_geometric_arithmetic_l2404_240494

noncomputable def a_n (n : ℕ) : ℕ := 2^n
def b_n (n : ℕ) : ℕ := 2*n - 1

theorem geom_seq_general_term (a : ℕ → ℕ) (a1 : a 1 = 2)
  (a2 : a 3 = (a 2) + 4) : ∀ n, a n = a_n n :=
by
  sorry

theorem sum_geometric_arithmetic (a b : ℕ → ℕ) 
  (a_def : ∀ n, a n = 2 ^ n) (b_def : ∀ n, b n = 2 * n - 1) : 
  ∀ n, (Finset.range n).sum (λ i => (a (i + 1) + b (i + 1))) = 2^(n+1) + n^2 - 2 :=
by
  sorry

end geom_seq_general_term_sum_geometric_arithmetic_l2404_240494


namespace cost_per_square_meter_l2404_240411

noncomputable def costPerSquareMeter 
  (length : ℝ) (breadth : ℝ) (width : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / ((length * width) + (breadth * width) - (width * width))

theorem cost_per_square_meter (H1 : length = 110)
                              (H2 : breadth = 60)
                              (H3 : width = 10)
                              (H4 : total_cost = 4800) : 
  costPerSquareMeter length breadth width total_cost = 3 := 
by
  sorry

end cost_per_square_meter_l2404_240411


namespace angle_measure_l2404_240441

theorem angle_measure (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : P = 206 :=
by
  sorry

end angle_measure_l2404_240441


namespace volume_ratio_of_cones_l2404_240489

theorem volume_ratio_of_cones (R : ℝ) (hR : 0 < R) :
  let circumference := 2 * Real.pi * R
  let sector1_circumference := (2 / 3) * circumference
  let sector2_circumference := (1 / 3) * circumference
  let r1 := sector1_circumference / (2 * Real.pi)
  let r2 := sector2_circumference / (2 * Real.pi)
  let s := R
  let h1 := Real.sqrt (R^2 - r1^2)
  let h2 := Real.sqrt (R^2 - r2^2)
  let V1 := (Real.pi * r1^2 * h1) / 3
  let V2 := (Real.pi * r2^2 * h2) / 3
  V1 / V2 = Real.sqrt 10 := 
by
  sorry

end volume_ratio_of_cones_l2404_240489


namespace boys_girls_relationship_l2404_240455

theorem boys_girls_relationship (b g : ℕ) (h1 : b > 0) (h2 : g > 2) (h3 : ∀ n : ℕ, n < b → (n + 1) + 2 ≤ g) (h4 : b + 2 = g) : b = g - 2 := 
by
  sorry

end boys_girls_relationship_l2404_240455


namespace range_of_a_l2404_240401

-- Definitions and theorems
theorem range_of_a (a : ℝ) : 
  (∀ (x y z : ℝ), x + y + z = 1 → abs (a - 2) ≤ x^2 + 2*y^2 + 3*z^2) → (16 / 11 ≤ a ∧ a ≤ 28 / 11) := 
by
  sorry

end range_of_a_l2404_240401


namespace find_omega_l2404_240495

theorem find_omega (ω : Real) (h : ∀ x : Real, (1 / 2) * Real.cos (ω * x - (Real.pi / 6)) = (1 / 2) * Real.cos (ω * (x + Real.pi) - (Real.pi / 6))) : ω = 2 ∨ ω = -2 :=
by
  sorry

end find_omega_l2404_240495


namespace sum_of_six_terms_l2404_240481

theorem sum_of_six_terms (a1 : ℝ) (S4 : ℝ) (d : ℝ) (a1_eq : a1 = 1 / 2) (S4_eq : S4 = 20) :
  S4 = (4 * a1 + (4 * (4 - 1) / 2) * d) → (S4 = 20) →
  (6 * a1 + (6 * (6 - 1) / 2) * d = 48) :=
by
  intros
  sorry

end sum_of_six_terms_l2404_240481


namespace discount_percentage_l2404_240415

theorem discount_percentage
  (number_of_fandoms : ℕ)
  (tshirts_per_fandom : ℕ)
  (price_per_shirt : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (total_expected_price_with_discount_without_tax : ℝ)
  (total_expected_price_without_discount : ℝ)
  (discount_amount : ℝ)
  (discount_percentage : ℝ) :

  number_of_fandoms = 4 ∧
  tshirts_per_fandom = 5 ∧
  price_per_shirt = 15 ∧
  tax_rate = 10 / 100 ∧
  total_paid = 264 ∧
  total_expected_price_with_discount_without_tax = total_paid / (1 + tax_rate) ∧
  total_expected_price_without_discount = number_of_fandoms * tshirts_per_fandom * price_per_shirt ∧
  discount_amount = total_expected_price_without_discount - total_expected_price_with_discount_without_tax ∧
  discount_percentage = (discount_amount / total_expected_price_without_discount) * 100 ->

  discount_percentage = 20 :=
sorry

end discount_percentage_l2404_240415


namespace new_oranges_added_l2404_240410

def initial_oranges : Nat := 31
def thrown_away_oranges : Nat := 9
def final_oranges : Nat := 60
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges (initial_oranges thrown_away_oranges final_oranges : Nat) : Nat := 
  final_oranges - (initial_oranges - thrown_away_oranges)

theorem new_oranges_added :
  new_oranges initial_oranges thrown_away_oranges final_oranges = 38 := by
  sorry

end new_oranges_added_l2404_240410


namespace find_salary_january_l2404_240439

noncomputable section
open Real

def average_salary_jan_to_apr (J F M A : ℝ) : Prop := 
  (J + F + M + A) / 4 = 8000

def average_salary_feb_to_may (F M A May : ℝ) : Prop := 
  (F + M + A + May) / 4 = 9500

def may_salary_value (May : ℝ) : Prop := 
  May = 6500

theorem find_salary_january : 
  ∀ J F M A May, 
    average_salary_jan_to_apr J F M A → 
    average_salary_feb_to_may F M A May → 
    may_salary_value May → 
    J = 500 :=
by
  intros J F M A May h1 h2 h3
  sorry

end find_salary_january_l2404_240439


namespace largest_good_number_is_576_smallest_bad_number_is_443_l2404_240408

def is_good_number (M : ℕ) : Prop :=
  ∃ (a b c d : ℤ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

def largest_good_number : ℕ := 576

def smallest_bad_number : ℕ := 443

theorem largest_good_number_is_576 : ∀ M : ℕ, is_good_number M → M ≤ 576 := 
by
  sorry

theorem smallest_bad_number_is_443 : ∀ M : ℕ, ¬ is_good_number M → 443 ≤ M :=
by
  sorry

end largest_good_number_is_576_smallest_bad_number_is_443_l2404_240408


namespace time_after_1750_minutes_is_1_10_pm_l2404_240444

def add_minutes_to_time (hours : Nat) (minutes : Nat) : Nat × Nat :=
  let total_minutes := hours * 60 + minutes
  (total_minutes / 60, total_minutes % 60)

def time_after_1750_minutes (current_hour : Nat) (current_minute : Nat) : Nat × Nat :=
  let (new_hour, new_minute) := add_minutes_to_time current_hour current_minute
  let final_hour := (new_hour + 1750 / 60) % 24
  let final_minute := (new_minute + 1750 % 60) % 60
  (final_hour, final_minute)

theorem time_after_1750_minutes_is_1_10_pm : 
  time_after_1750_minutes 8 0 = (13, 10) :=
by {
  sorry
}

end time_after_1750_minutes_is_1_10_pm_l2404_240444


namespace find_digits_sum_l2404_240468

theorem find_digits_sum (a b c : Nat) (ha : 0 <= a ∧ a <= 9) (hb : 0 <= b ∧ b <= 9) 
  (hc : 0 <= c ∧ c <= 9) 
  (h1 : 2 * a = c) 
  (h2 : b = b) : 
  a + b + c = 11 :=
  sorry

end find_digits_sum_l2404_240468


namespace every_real_has_cube_root_l2404_240460

theorem every_real_has_cube_root : ∀ y : ℝ, ∃ x : ℝ, x^3 = y := 
by
  sorry

end every_real_has_cube_root_l2404_240460
