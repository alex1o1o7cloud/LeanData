import Mathlib

namespace NUMINAMATH_GPT_hash_op_example_l575_57591

def hash_op (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem hash_op_example : hash_op 2 5 3 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_hash_op_example_l575_57591


namespace NUMINAMATH_GPT_margin_expression_l575_57562

variable (C S M : ℝ)
variable (n : ℕ)

theorem margin_expression (h : M = (C + S) / n) : M = (2 * S) / (n + 1) :=
sorry

end NUMINAMATH_GPT_margin_expression_l575_57562


namespace NUMINAMATH_GPT_negation_of_p_l575_57552

noncomputable def p : Prop := ∀ x : ℝ, x > 0 → 2 * x^2 + 1 > 0

theorem negation_of_p : (∃ x : ℝ, x > 0 ∧ 2 * x^2 + 1 ≤ 0) ↔ ¬p :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l575_57552


namespace NUMINAMATH_GPT_range_of_m_l575_57527

theorem range_of_m (m : ℝ) :
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ x - (m^2 - 2 * m + 4) * y + 6 > 0) →
  -1 < m ∧ m < 3 :=
by
  intros h
  rcases h with ⟨x, y, hx, hy, hineq⟩
  rw [hx, hy] at hineq
  sorry

end NUMINAMATH_GPT_range_of_m_l575_57527


namespace NUMINAMATH_GPT_spending_difference_l575_57535

-- Define the given conditions
def ice_cream_cartons := 19
def yoghurt_cartons := 4
def ice_cream_cost_per_carton := 7
def yoghurt_cost_per_carton := 1

-- Calculate the total cost based on the given conditions
def total_ice_cream_cost := ice_cream_cartons * ice_cream_cost_per_carton
def total_yoghurt_cost := yoghurt_cartons * yoghurt_cost_per_carton

-- The statement to prove
theorem spending_difference :
  total_ice_cream_cost - total_yoghurt_cost = 129 :=
by
  sorry

end NUMINAMATH_GPT_spending_difference_l575_57535


namespace NUMINAMATH_GPT_johns_new_total_lift_l575_57544

theorem johns_new_total_lift :
  let initial_squat := 700
  let initial_bench := 400
  let initial_deadlift := 800
  let squat_loss_percentage := 30 / 100.0
  let squat_loss := squat_loss_percentage * initial_squat
  let new_squat := initial_squat - squat_loss
  let new_bench := initial_bench
  let new_deadlift := initial_deadlift - 200
  new_squat + new_bench + new_deadlift = 1490 := 
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_johns_new_total_lift_l575_57544


namespace NUMINAMATH_GPT_batsman_new_average_l575_57512

-- Let A be the average score before the 16th inning
def avg_before (A : ℝ) : Prop :=
  ∃ total_runs: ℝ, total_runs = 15 * A

-- Condition 1: The batsman makes 64 runs in the 16th inning
def score_in_16th_inning := 64

-- Condition 2: This increases his average by 3 runs
def avg_increase (A : ℝ) : Prop :=
  A + 3 = (15 * A + score_in_16th_inning) / 16

theorem batsman_new_average (A : ℝ) (h1 : avg_before A) (h2 : avg_increase A) :
  (A + 3) = 19 :=
sorry

end NUMINAMATH_GPT_batsman_new_average_l575_57512


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l575_57590

variables (x y : ℝ)

theorem necessary_and_sufficient_condition (h1 : x > y) (h2 : 1/x > 1/y) : x * y < 0 :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l575_57590


namespace NUMINAMATH_GPT_max_tetrahedron_volume_l575_57541

theorem max_tetrahedron_volume 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (right_triangle : ∃ A B C : Type, 
    ∃ (angle_C : ℝ) (h_angle_C : angle_C = π / 2), 
    ∃ (BC CA : ℝ), BC = a ∧ CA = b) : 
  ∃ V : ℝ, V = (a^2 * b^2) / (6 * (a^(2/3) + b^(2/3))^(3/2)) := 
sorry

end NUMINAMATH_GPT_max_tetrahedron_volume_l575_57541


namespace NUMINAMATH_GPT_cody_games_remaining_l575_57546

-- Definitions based on the conditions
def initial_games : ℕ := 9
def games_given_away : ℕ := 4

-- Theorem statement
theorem cody_games_remaining : initial_games - games_given_away = 5 :=
by sorry

end NUMINAMATH_GPT_cody_games_remaining_l575_57546


namespace NUMINAMATH_GPT_flower_cost_l575_57587

theorem flower_cost (F : ℕ) (h1 : F + (F + 20) + (F - 2) = 45) : F = 9 :=
by
  sorry

end NUMINAMATH_GPT_flower_cost_l575_57587


namespace NUMINAMATH_GPT_calculate_expression_l575_57586

theorem calculate_expression : 2 * (-2) + (-3) = -7 := 
  sorry

end NUMINAMATH_GPT_calculate_expression_l575_57586


namespace NUMINAMATH_GPT_triangular_pyramid_nonexistence_l575_57531

theorem triangular_pyramid_nonexistence
    (h : ℕ)
    (hb : ℕ)
    (P : ℕ)
    (h_eq : h = 60)
    (hb_eq : hb = 61)
    (P_eq : P = 62) :
    ¬ ∃ (a b c : ℝ), a + b + c = P ∧ 60^2 = 61^2 - (a^2 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_triangular_pyramid_nonexistence_l575_57531


namespace NUMINAMATH_GPT_time_addition_sum_l575_57524

/-- Given the start time of 3:15:20 PM and adding a duration of 
    305 hours, 45 minutes, and 56 seconds, the resultant hour, 
    minute, and second values sum to 26. -/
theorem time_addition_sum : 
  let current_hour := 15
  let current_minute := 15
  let current_second := 20
  let added_hours := 305
  let added_minutes := 45
  let added_seconds := 56
  let final_hour := ((current_hour + (added_hours % 12) + ((current_minute + added_minutes) / 60) + ((current_second + added_seconds) / 3600)) % 12)
  let final_minute := ((current_minute + added_minutes + ((current_second + added_seconds) / 60)) % 60)
  let final_second := ((current_second + added_seconds) % 60)
  final_hour + final_minute + final_second = 26 := 
  sorry

end NUMINAMATH_GPT_time_addition_sum_l575_57524


namespace NUMINAMATH_GPT_solve_for_m_l575_57560

def z1 := Complex.mk 3 2
def z2 (m : ℝ) := Complex.mk 1 m

theorem solve_for_m (m : ℝ) (h : (z1 * z2 m).re = 0) : m = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l575_57560


namespace NUMINAMATH_GPT_div_by_eleven_l575_57568

theorem div_by_eleven (a b : ℤ) (h : (a^2 + 9 * a * b + b^2) % 11 = 0) : 
  (a^2 - b^2) % 11 = 0 :=
sorry

end NUMINAMATH_GPT_div_by_eleven_l575_57568


namespace NUMINAMATH_GPT_carter_has_255_cards_l575_57585

-- Definition of the number of baseball cards Marcus has.
def marcus_cards : ℕ := 350

-- Definition of the number of more cards Marcus has than Carter.
def difference : ℕ := 95

-- Definition of the number of baseball cards Carter has.
def carter_cards : ℕ := marcus_cards - difference

-- Theorem stating that Carter has 255 baseball cards.
theorem carter_has_255_cards : carter_cards = 255 :=
sorry

end NUMINAMATH_GPT_carter_has_255_cards_l575_57585


namespace NUMINAMATH_GPT_smallest_number_of_coins_to_pay_up_to_2_dollars_l575_57571

def smallest_number_of_coins_to_pay_up_to (max_amount : Nat) : Nat :=
  sorry  -- This function logic needs to be defined separately

theorem smallest_number_of_coins_to_pay_up_to_2_dollars :
  smallest_number_of_coins_to_pay_up_to 199 = 11 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_coins_to_pay_up_to_2_dollars_l575_57571


namespace NUMINAMATH_GPT_intersection_M_N_l575_57582

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := { x | 2 * x > 7 }

theorem intersection_M_N :
  M ∩ N = {5, 7, 9} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l575_57582


namespace NUMINAMATH_GPT_ratio_of_still_lifes_to_portraits_l575_57523

noncomputable def total_paintings : ℕ := 80
noncomputable def portraits : ℕ := 16
noncomputable def still_lifes : ℕ := total_paintings - portraits
axiom still_lifes_is_multiple_of_portraits : ∃ k : ℕ, still_lifes = k * portraits

theorem ratio_of_still_lifes_to_portraits : still_lifes / portraits = 4 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_ratio_of_still_lifes_to_portraits_l575_57523


namespace NUMINAMATH_GPT_range_of_a_l575_57505

-- Definitions for the conditions
def p (x : ℝ) := x ≤ 2
def q (x : ℝ) (a : ℝ) := x < a + 2

-- Theorem statement
theorem range_of_a (a : ℝ) : (∀ x : ℝ, q x a → p x) → a ≤ 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l575_57505


namespace NUMINAMATH_GPT_fraction_sum_eq_five_fourths_l575_57506

theorem fraction_sum_eq_five_fourths (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b) / c = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_eq_five_fourths_l575_57506


namespace NUMINAMATH_GPT_ferris_wheel_cost_l575_57537

theorem ferris_wheel_cost (roller_coaster_cost log_ride_cost zach_initial_tickets zach_additional_tickets total_tickets ferris_wheel_cost : ℕ) 
  (h1 : roller_coaster_cost = 7)
  (h2 : log_ride_cost = 1)
  (h3 : zach_initial_tickets = 1)
  (h4 : zach_additional_tickets = 9)
  (h5 : total_tickets = zach_initial_tickets + zach_additional_tickets)
  (h6 : total_tickets - (roller_coaster_cost + log_ride_cost) = ferris_wheel_cost) :
  ferris_wheel_cost = 2 := 
by
  sorry

end NUMINAMATH_GPT_ferris_wheel_cost_l575_57537


namespace NUMINAMATH_GPT_solve_fractional_equation_l575_57503

theorem solve_fractional_equation : 
  ∃ x : ℝ, (x - 1) / 2 = 1 - (3 * x + 2) / 5 ↔ x = 1 := 
sorry

end NUMINAMATH_GPT_solve_fractional_equation_l575_57503


namespace NUMINAMATH_GPT_sequence_count_zeros_ones_15_l575_57511

-- Definition of the problem
def count_sequences (n : Nat) : Nat := sorry -- Function calculating the number of valid sequences

-- The theorem stating that for sequence length 15, the number of such sequences is 266
theorem sequence_count_zeros_ones_15 : count_sequences 15 = 266 := 
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_sequence_count_zeros_ones_15_l575_57511


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l575_57565

theorem quadratic_has_two_distinct_real_roots (m : ℝ) :
  ∃ a b : ℝ, a ≠ b ∧ (a ^ 2 + m * a - 8 = 0) ∧ (b ^ 2 + m * b - 8 = 0) :=
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l575_57565


namespace NUMINAMATH_GPT_find_X_l575_57548

theorem find_X (X : ℝ) (h : 0.80 * X - 0.35 * 300 = 31) : X = 170 :=
by
  sorry

end NUMINAMATH_GPT_find_X_l575_57548


namespace NUMINAMATH_GPT_dice_circle_probability_l575_57507

theorem dice_circle_probability :
  ∀ (d : ℕ), (2 ≤ d ∧ d ≤ 432) ∧
  ((∃ (x y : ℕ), (1 ≤ x ∧ x ≤ 6) ∧ (1 ≤ y ∧ y <= 6) ∧ d = x^3 + y^3)) →
  ((d * (d - 4) < 0) ↔ (d = 2)) →
  (∃ (P : ℚ), P = 1 / 36) :=
by
  sorry

end NUMINAMATH_GPT_dice_circle_probability_l575_57507


namespace NUMINAMATH_GPT_geometric_sum_ratio_l575_57502

theorem geometric_sum_ratio (a₁ q : ℝ) (h₁ : q ≠ 1) (h₂ : (1 - q^4) / (1 - q^2) = 5) :
  (1 - q^8) / (1 - q^4) = 17 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sum_ratio_l575_57502


namespace NUMINAMATH_GPT_distance_interval_l575_57526

theorem distance_interval (d : ℝ) (h1 : ¬(d ≥ 8)) (h2 : ¬(d ≤ 7)) (h3 : ¬(d ≤ 6 → north)):
  7 < d ∧ d < 8 :=
by
  have h_d8 : d < 8 := by linarith
  have h_d7 : d > 7 := by linarith
  exact ⟨h_d7, h_d8⟩

end NUMINAMATH_GPT_distance_interval_l575_57526


namespace NUMINAMATH_GPT_find_positive_real_solution_l575_57578

theorem find_positive_real_solution (x : ℝ) (h₁ : 0 < x) (h₂ : 1/2 * (4 * x ^ 2 - 4) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4)) :
  x = 20 + Real.sqrt 410 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_real_solution_l575_57578


namespace NUMINAMATH_GPT_average_grade_of_female_students_l575_57530

theorem average_grade_of_female_students
  (avg_all_students : ℝ)
  (avg_male_students : ℝ)
  (num_males : ℕ)
  (num_females : ℕ)
  (total_students := num_males + num_females)
  (total_score_all_students := avg_all_students * total_students)
  (total_score_male_students := avg_male_students * num_males) :
  avg_all_students = 90 →
  avg_male_students = 87 →
  num_males = 8 →
  num_females = 12 →
  ((total_score_all_students - total_score_male_students) / num_females) = 92 := by
  intros h_avg_all h_avg_male h_num_males h_num_females
  sorry

end NUMINAMATH_GPT_average_grade_of_female_students_l575_57530


namespace NUMINAMATH_GPT_complete_square_transform_l575_57501

theorem complete_square_transform :
  ∀ x : ℝ, x^2 - 4 * x - 6 = 0 → (x - 2)^2 = 10 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_complete_square_transform_l575_57501


namespace NUMINAMATH_GPT_range_of_a_l575_57540

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ |x| = a * x - a) ∧ (¬ ∃ x : ℝ, x < 0 ∧ |x| = a * x - a) ↔ (a > 1 ∨ a ≤ -1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l575_57540


namespace NUMINAMATH_GPT_f_zero_f_odd_range_of_x_l575_57515

variable {f : ℝ → ℝ}

axiom func_property (x y : ℝ) : f (x + y) = f x + f y
axiom f_third : f (1 / 3) = 1
axiom f_positive (x : ℝ) : x > 0 → f x > 0

-- Part (1)
theorem f_zero : f 0 = 0 :=
sorry

-- Part (2)
theorem f_odd (x : ℝ) : f (-x) = -f x :=
sorry

-- Part (3)
theorem range_of_x (x : ℝ) : f x + f (2 + x) < 2 → x < -2 / 3 :=
sorry

end NUMINAMATH_GPT_f_zero_f_odd_range_of_x_l575_57515


namespace NUMINAMATH_GPT_ice_cream_stack_order_l575_57572

theorem ice_cream_stack_order (scoops : Finset ℕ) (h_scoops : scoops.card = 5) :
  (scoops.prod id) = 120 :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_stack_order_l575_57572


namespace NUMINAMATH_GPT_min_value_of_expression_l575_57574

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b + a * b = 3) :
  2 * a + b ≥ 4 * Real.sqrt 2 - 3 := 
sorry

end NUMINAMATH_GPT_min_value_of_expression_l575_57574


namespace NUMINAMATH_GPT_minimum_ab_l575_57533

theorem minimum_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : ab + 2 = 2 * (a + b)) : ab ≥ 6 + 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_ab_l575_57533


namespace NUMINAMATH_GPT_graph_of_equation_l575_57577

theorem graph_of_equation {x y : ℝ} (h : (x - 2 * y)^2 = x^2 - 4 * y^2) :
  (y = 0) ∨ (x = 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_l575_57577


namespace NUMINAMATH_GPT_max_weight_of_chocolates_l575_57575

def max_total_weight (chocolates : List ℕ) (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) : ℕ :=
300

theorem max_weight_of_chocolates (chocolates : List ℕ)
  (H_wt : ∀ c ∈ chocolates, c ≤ 100)
  (H_div : ∀ (S L : List ℕ), (S ⊆ chocolates) → (L ⊆ chocolates) 
                        → (S ≠ L) 
                        → ((S.sum ≤ 100 ∨ L.sum ≤ 100))) :
  max_total_weight chocolates H_wt H_div = 300 :=
sorry

end NUMINAMATH_GPT_max_weight_of_chocolates_l575_57575


namespace NUMINAMATH_GPT_cost_price_for_fabrics_l575_57576

noncomputable def total_cost_price (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  selling_price - (meters_sold * profit_per_meter)

noncomputable def cost_price_per_meter (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  total_cost_price meters_sold selling_price profit_per_meter / meters_sold

theorem cost_price_for_fabrics :
  cost_price_per_meter 45 6000 12 = 121.33 ∧
  cost_price_per_meter 60 10800 15 = 165 ∧
  cost_price_per_meter 30 3900 10 = 120 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_for_fabrics_l575_57576


namespace NUMINAMATH_GPT_scientific_notation_of_61345_05_billion_l575_57539

theorem scientific_notation_of_61345_05_billion :
  ∃ x : ℝ, (61345.05 * 10^9) = x ∧ x = 6.134505 * 10^12 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_61345_05_billion_l575_57539


namespace NUMINAMATH_GPT_cost_price_to_marked_price_l575_57518

theorem cost_price_to_marked_price (MP CP SP : ℝ)
  (h1 : SP = MP * 0.87)
  (h2 : SP = CP * 1.359375) :
  (CP / MP) * 100 = 64 := by
  sorry

end NUMINAMATH_GPT_cost_price_to_marked_price_l575_57518


namespace NUMINAMATH_GPT_smallest_integer_20p_larger_and_19p_smaller_l575_57596

theorem smallest_integer_20p_larger_and_19p_smaller :
  ∃ (N x y : ℕ), N = 162 ∧ N = 12 / 10 * x ∧ N = 81 / 100 * y :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_20p_larger_and_19p_smaller_l575_57596


namespace NUMINAMATH_GPT_health_risk_probability_l575_57594

theorem health_risk_probability :
  let a := 0.08 * 500
  let b := 0.08 * 500
  let c := 0.08 * 500
  let d := 0.18 * 500
  let e := 0.18 * 500
  let f := 0.18 * 500
  let g := 0.05 * 500
  let h := 500 - (3 * 40 + 3 * 90 + 25)
  let q := 500 - (a + d + e + g)
  let p := 1
  let q := 3
  p + q = 4 := sorry

end NUMINAMATH_GPT_health_risk_probability_l575_57594


namespace NUMINAMATH_GPT_total_items_sold_at_garage_sale_l575_57528

-- Define the conditions for the problem
def items_more_expensive_than_radio : Nat := 16
def items_less_expensive_than_radio : Nat := 23

-- Declare the total number of items using the given conditions
theorem total_items_sold_at_garage_sale 
  (h1 : items_more_expensive_than_radio = 16)
  (h2 : items_less_expensive_than_radio = 23) :
  items_more_expensive_than_radio + 1 + items_less_expensive_than_radio = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_items_sold_at_garage_sale_l575_57528


namespace NUMINAMATH_GPT_sue_received_votes_l575_57516

theorem sue_received_votes (total_votes : ℕ) (sue_percentage : ℚ) (h1 : total_votes = 1000) (h2 : sue_percentage = 35 / 100) :
  (sue_percentage * total_votes) = 350 := by
  sorry

end NUMINAMATH_GPT_sue_received_votes_l575_57516


namespace NUMINAMATH_GPT_vacation_days_l575_57534

def num_families : ℕ := 3
def people_per_family : ℕ := 4
def towels_per_day_per_person : ℕ := 1
def washer_capacity : ℕ := 14
def num_loads : ℕ := 6

def total_people : ℕ := num_families * people_per_family
def towels_per_day : ℕ := total_people * towels_per_day_per_person
def total_towels : ℕ := num_loads * washer_capacity

def days_at_vacation_rental := total_towels / towels_per_day

theorem vacation_days : days_at_vacation_rental = 7 := by
  sorry

end NUMINAMATH_GPT_vacation_days_l575_57534


namespace NUMINAMATH_GPT_arrangement_problem_l575_57558
   
   def numberOfArrangements (n : Nat) : Nat :=
     n.factorial

   def exclusiveArrangements (total people : Nat) (positions : Nat) : Nat :=
     (positions.choose 2) * (total - 2).factorial

   theorem arrangement_problem : 
     (numberOfArrangements 5) - (exclusiveArrangements 5 3) = 84 := 
   by
     sorry
   
end NUMINAMATH_GPT_arrangement_problem_l575_57558


namespace NUMINAMATH_GPT_final_payment_order_450_l575_57573

noncomputable def finalPayment (orderAmount : ℝ) : ℝ :=
  let serviceCharge := if orderAmount < 500 then 0.04 * orderAmount
                      else if orderAmount < 1000 then 0.05 * orderAmount
                      else 0.06 * orderAmount
  let salesTax := if orderAmount < 500 then 0.05 * orderAmount
                  else if orderAmount < 1000 then 0.06 * orderAmount
                  else 0.07 * orderAmount
  let totalBeforeDiscount := orderAmount + serviceCharge + salesTax
  let discount := if totalBeforeDiscount < 600 then 0.05 * totalBeforeDiscount
                  else if totalBeforeDiscount < 800 then 0.10 * totalBeforeDiscount
                  else 0.15 * totalBeforeDiscount
  totalBeforeDiscount - discount

theorem final_payment_order_450 :
  finalPayment 450 = 465.98 := by
  sorry

end NUMINAMATH_GPT_final_payment_order_450_l575_57573


namespace NUMINAMATH_GPT_sum_of_squares_l575_57563

theorem sum_of_squares (w x y z a b c : ℝ) 
  (hwx : w * x = a^2) 
  (hwy : w * y = b^2) 
  (hwz : w * z = c^2) 
  (hw : w ≠ 0) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = (a^4 + b^4 + c^4) / w^2 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l575_57563


namespace NUMINAMATH_GPT_solve_equation_l575_57554

theorem solve_equation (x : ℝ) (hx : x ≠ 0) : 
  x^2 + 36 / x^2 = 13 ↔ (x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l575_57554


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l575_57559

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₁ : a = 12) (h₂ : b = 12) (h₃ : c = 17) : a + b + c = 41 :=
by
  rw [h₁, h₂, h₃]
  norm_num

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l575_57559


namespace NUMINAMATH_GPT_sum_of_reciprocals_eq_one_l575_57519

theorem sum_of_reciprocals_eq_one {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : x + y = (x * y) ^ 2) : (1/x) + (1/y) = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_eq_one_l575_57519


namespace NUMINAMATH_GPT_dot_product_value_l575_57597

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (3, 1)

theorem dot_product_value :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_dot_product_value_l575_57597


namespace NUMINAMATH_GPT_part1_part2_l575_57532

noncomputable def f : ℝ → ℝ 
| x => if 0 ≤ x then 2^x - 1 else -2^(-x) + 1

theorem part1 (x : ℝ) (h : x < 0) : f x = -2^(-x) + 1 := sorry

theorem part2 (a : ℝ) : f a ≤ 3 ↔ a ≤ 2 := sorry

end NUMINAMATH_GPT_part1_part2_l575_57532


namespace NUMINAMATH_GPT_no_solution_fractional_eq_l575_57564

theorem no_solution_fractional_eq (y : ℝ) (h : y ≠ 3) : 
  ¬ ( (y-2)/(y-3) = 2 - 1/(3-y) ) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_fractional_eq_l575_57564


namespace NUMINAMATH_GPT_evaluate_expression_l575_57536

theorem evaluate_expression : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l575_57536


namespace NUMINAMATH_GPT_find_P_l575_57589

-- We start by defining the cubic polynomial
def cubic_eq (P : ℝ) (x : ℝ) := 5 * x^3 - 5 * (P + 1) * x^2 + (71 * P - 1) * x + 1

-- Define the condition that all roots are natural numbers
def has_three_natural_roots (P : ℝ) : Prop :=
  ∃ a b c : ℕ, 
    cubic_eq P a = 66 * P ∧ cubic_eq P b = 66 * P ∧ cubic_eq P c = 66 * P

-- Prove the value of P that satisfies the condition
theorem find_P : ∀ P : ℝ, has_three_natural_roots P → P = 76 := 
by
  -- We start the proof here
  sorry

end NUMINAMATH_GPT_find_P_l575_57589


namespace NUMINAMATH_GPT_new_apples_grew_l575_57583

-- The number of apples originally on the tree.
def original_apples : ℕ := 11

-- The number of apples picked by Rachel.
def picked_apples : ℕ := 7

-- The number of apples currently on the tree.
def current_apples : ℕ := 6

-- The number of apples left on the tree after picking.
def remaining_apples : ℕ := original_apples - picked_apples

-- The number of new apples that grew on the tree.
def new_apples : ℕ := current_apples - remaining_apples

-- The theorem we need to prove.
theorem new_apples_grew :
  new_apples = 2 := by
    sorry

end NUMINAMATH_GPT_new_apples_grew_l575_57583


namespace NUMINAMATH_GPT_tiffany_cans_l575_57542

variable {M : ℕ}

theorem tiffany_cans : (M + 12 = 2 * M) → (M = 12) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tiffany_cans_l575_57542


namespace NUMINAMATH_GPT_verify_sum_of_new_rates_proof_l575_57514

-- Given conditions and initial setup
variable (k : ℕ)
variable (h_initial : ℕ := 5 * k) -- Hanhan's initial hourly rate
variable (x_initial : ℕ := 4 * k) -- Xixi's initial hourly rate
variable (increment : ℕ := 20)    -- Increment in hourly rates

-- New rates after increment
variable (h_new : ℕ := h_initial + increment) -- Hanhan's new hourly rate
variable (x_new : ℕ := x_initial + increment) -- Xixi's new hourly rate

-- Given ratios
variable (initial_ratio : h_initial / x_initial = 5 / 4) 
variable (new_ratio : h_new / x_new = 6 / 5)

-- Target sum of the new hourly rates
def sum_of_new_rates_proof : Prop :=
  h_new + x_new = 220

theorem verify_sum_of_new_rates_proof : sum_of_new_rates_proof k :=
by
  sorry

end NUMINAMATH_GPT_verify_sum_of_new_rates_proof_l575_57514


namespace NUMINAMATH_GPT_inverse_proportion_inequality_l575_57543

theorem inverse_proportion_inequality 
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : 0 < x2)
  (h3 : y1 = 6 / x1)
  (h4 : y2 = 6 / x2) : 
  y1 < y2 :=
sorry

end NUMINAMATH_GPT_inverse_proportion_inequality_l575_57543


namespace NUMINAMATH_GPT_smallest_consecutive_even_sum_560_l575_57513

theorem smallest_consecutive_even_sum_560 (n : ℕ) (h : 7 * n + 42 = 560) : n = 74 :=
  by
    sorry

end NUMINAMATH_GPT_smallest_consecutive_even_sum_560_l575_57513


namespace NUMINAMATH_GPT_tetrahedron_edges_midpoint_distances_sum_l575_57567

theorem tetrahedron_edges_midpoint_distances_sum (a b c d e f m1 m2 m3 m4 m5 m6 : ℝ) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 4 * (m1^2 + m2^2 + m3^2 + m4^2 + m5^2 + m6^2) :=
sorry

end NUMINAMATH_GPT_tetrahedron_edges_midpoint_distances_sum_l575_57567


namespace NUMINAMATH_GPT_sum_of_squares_l575_57566

theorem sum_of_squares (x y z a b c k : ℝ)
  (h₁ : x * y = k * a)
  (h₂ : x * z = b)
  (h₃ : y * z = c)
  (hk : k ≠ 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0) :
  x^2 + y^2 + z^2 = (k * (a * b + a * c + b * c)) / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l575_57566


namespace NUMINAMATH_GPT_range_of_a_when_min_f_ge_neg_a_l575_57595

noncomputable def f (a x : ℝ) := a * Real.log x + 2 * x

theorem range_of_a_when_min_f_ge_neg_a (a : ℝ) (h₀ : a ≠ 0)
  (h₁ : ∀ x > 0, f a x ≥ -a) :
  -2 ≤ a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_when_min_f_ge_neg_a_l575_57595


namespace NUMINAMATH_GPT_prime_divisors_difference_l575_57549

def prime_factors (n : ℕ) : ℕ := sorry -- definition placeholder

theorem prime_divisors_difference (n : ℕ) (hn : 0 < n) : 
  ∃ k m : ℕ, 0 < k ∧ 0 < m ∧ k - m = n ∧ prime_factors k - prime_factors m = 1 := 
sorry

end NUMINAMATH_GPT_prime_divisors_difference_l575_57549


namespace NUMINAMATH_GPT_problem_false_statements_l575_57556

noncomputable def statement_I : Prop :=
  ∀ x : ℝ, ⌊x + Real.pi⌋ = ⌊x⌋ + 3

noncomputable def statement_II : Prop :=
  ∀ x : ℝ, ⌊x + Real.sqrt 2⌋ = ⌊x⌋ + ⌊Real.sqrt 2⌋

noncomputable def statement_III : Prop :=
  ∀ x : ℝ, ⌊x * Real.pi⌋ = ⌊x⌋ * ⌊Real.pi⌋

theorem problem_false_statements : ¬(statement_I ∨ statement_II ∨ statement_III) := 
by
  sorry

end NUMINAMATH_GPT_problem_false_statements_l575_57556


namespace NUMINAMATH_GPT_distance_to_school_l575_57584

variables (d : ℝ)
def jog_rate := 5
def bus_rate := 30
def total_time := 1 

theorem distance_to_school :
  (d / jog_rate) + (d / bus_rate) = total_time ↔ d = 30 / 7 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_school_l575_57584


namespace NUMINAMATH_GPT_ratio_of_cakes_l575_57580

/-- Define the usual number of cheesecakes, muffins, and red velvet cakes baked in a week -/
def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet_cakes : ℕ := 8

/-- Define the total number of cakes usually baked in a week -/
def usual_cakes : ℕ := usual_cheesecakes + usual_muffins + usual_red_velvet_cakes

/-- Assume Carter baked this week a multiple of usual cakes, denoted as x -/
def multiple (x : ℕ) : Prop := usual_cakes * x = usual_cakes + 38

/-- Assume he baked usual_cakes + 38 equals 57 cakes -/
def total_cakes_this_week : ℕ := 57

/-- The theorem stating the problem: proving the ratio is 3:1 -/
theorem ratio_of_cakes (x : ℕ) (hx : multiple x) : 
  (total_cakes_this_week : ℚ) / (usual_cakes : ℚ) = (3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cakes_l575_57580


namespace NUMINAMATH_GPT_min_value_frac_2_over_a_plus_3_over_b_l575_57529

theorem min_value_frac_2_over_a_plus_3_over_b 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_value_frac_2_over_a_plus_3_over_b_l575_57529


namespace NUMINAMATH_GPT_problem_l575_57557

def x : ℕ := 660
def percentage_25_of_x : ℝ := 0.25 * x
def percentage_12_of_1500 : ℝ := 0.12 * 1500
def difference_of_percentages : ℝ := percentage_12_of_1500 - percentage_25_of_x

theorem problem : difference_of_percentages = 15 := by
  -- begin proof (content replaced by sorry)
  sorry

end NUMINAMATH_GPT_problem_l575_57557


namespace NUMINAMATH_GPT_age_difference_l575_57553

variable (A B C : ℕ)

-- Conditions
def ages_total_condition (a b c : ℕ) : Prop :=
  a + b = b + c + 11

-- Proof problem statement
theorem age_difference (a b c : ℕ) (h : ages_total_condition a b c) : a - c = 11 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l575_57553


namespace NUMINAMATH_GPT_max_xy_l575_57500

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 1) : xy <= 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_max_xy_l575_57500


namespace NUMINAMATH_GPT_inverse_function_f_l575_57504

noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

noncomputable def f_inv (x : ℝ) : ℝ := Real.log x / Real.log 2 - 1

theorem inverse_function_f : ∀ x > 0, f_inv (f x) = x :=
by
  intro x hx
  dsimp [f, f_inv]
  sorry

end NUMINAMATH_GPT_inverse_function_f_l575_57504


namespace NUMINAMATH_GPT_fair_dice_can_be_six_l575_57598

def fair_dice_outcomes : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem fair_dice_can_be_six : 6 ∈ fair_dice_outcomes :=
by {
  -- This formally states that 6 is a possible outcome when throwing a fair dice
  sorry
}

end NUMINAMATH_GPT_fair_dice_can_be_six_l575_57598


namespace NUMINAMATH_GPT_find_b_value_l575_57570

-- Define the conditions: line equation and given range for b
def line_eq (x : ℝ) (b : ℝ) : ℝ := b - x

-- Define the points P, Q, S
def P (b : ℝ) : ℝ × ℝ := ⟨0, b⟩
def Q (b : ℝ) : ℝ × ℝ := ⟨b, 0⟩
def S (b : ℝ) : ℝ × ℝ := ⟨6, b - 6⟩

-- Define the area ratio condition
def area_ratio_condition (b : ℝ) : Prop :=
  (0 < b ∧ b < 6) ∧ ((6 - b) / b) ^ 2 = 4 / 25

-- Define the main theorem to prove
theorem find_b_value (b : ℝ) : area_ratio_condition b → b = 4.3 := by
  sorry

end NUMINAMATH_GPT_find_b_value_l575_57570


namespace NUMINAMATH_GPT_ribeye_steak_cost_l575_57522

/-- Define the conditions in Lean -/
def appetizer_cost : ℕ := 8
def wine_cost : ℕ := 3
def wine_glasses : ℕ := 2
def dessert_cost : ℕ := 6
def total_spent : ℕ := 38
def tip_percentage : ℚ := 0.20

/-- Proving the cost of the ribeye steak before the discount -/
theorem ribeye_steak_cost (S : ℚ) (h : 20 + (S / 2) + (tip_percentage * (20 + S)) = total_spent) : S = 20 :=
by
  sorry

end NUMINAMATH_GPT_ribeye_steak_cost_l575_57522


namespace NUMINAMATH_GPT_solve_for_x_l575_57561

theorem solve_for_x (x : ℝ) (hx₁ : x ≠ 3) (hx₂ : x ≠ -2) 
  (h : (x + 5) / (x - 3) = (x - 2) / (x + 2)) : x = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l575_57561


namespace NUMINAMATH_GPT_main_theorem_l575_57547

variable (x : ℤ)

def H : ℤ := 12 - (3 + 7) + x
def T : ℤ := 12 - 3 + 7 + x

theorem main_theorem : H - T + x = -14 + x :=
by
  sorry

end NUMINAMATH_GPT_main_theorem_l575_57547


namespace NUMINAMATH_GPT_percent_round_trip_tickets_is_100_l575_57509

noncomputable def percent_round_trip_tickets (P : ℕ) (x : ℚ) : ℚ :=
  let R := x / 0.20
  R

theorem percent_round_trip_tickets_is_100
  (P : ℕ)
  (x : ℚ)
  (h : 20 * x = P) :
  percent_round_trip_tickets P (x / P) = 100 :=
by
  sorry

end NUMINAMATH_GPT_percent_round_trip_tickets_is_100_l575_57509


namespace NUMINAMATH_GPT_probability_of_vowel_initials_l575_57517

/-- In a class with 26 students, each student has unique initials that are double letters
    (i.e., AA, BB, ..., ZZ). If the vowels are A, E, I, O, U, and W, then the probability of
    randomly picking a student whose initials are vowels is 3/13. -/
theorem probability_of_vowel_initials :
  let total_students := 26
  let vowels := ['A', 'E', 'I', 'O', 'U', 'W']
  let num_vowels := 6
  let probability := num_vowels / total_students
  probability = 3 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_vowel_initials_l575_57517


namespace NUMINAMATH_GPT_winning_lottery_ticket_is_random_l575_57551

-- Definitions of the events
inductive Event
| certain : Event
| impossible : Event
| random : Event

open Event

-- Conditions
def boiling_water_event : Event := certain
def lottery_ticket_event : Event := random
def athlete_running_30mps_event : Event := impossible
def draw_red_ball_event : Event := impossible

-- Problem Statement
theorem winning_lottery_ticket_is_random : 
    lottery_ticket_event = random :=
sorry

end NUMINAMATH_GPT_winning_lottery_ticket_is_random_l575_57551


namespace NUMINAMATH_GPT_minimum_value_property_l575_57592

noncomputable def min_value_expression (x : ℝ) (h : x > 10) : ℝ :=
  (x^2 + 36) / (x - 10)

noncomputable def min_value : ℝ := 4 * Real.sqrt 34 + 20

theorem minimum_value_property (x : ℝ) (h : x > 10) :
  min_value_expression x h >= min_value := by
  sorry

end NUMINAMATH_GPT_minimum_value_property_l575_57592


namespace NUMINAMATH_GPT_polygon_is_octahedron_l575_57555

theorem polygon_is_octahedron (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_polygon_is_octahedron_l575_57555


namespace NUMINAMATH_GPT_matching_function_l575_57588

open Real

def table_data : List (ℝ × ℝ) := [(1, 4), (2, 2), (4, 1)]

theorem matching_function :
  ∃ a b c : ℝ, a > 0 ∧ 
               (∀ x y, (x, y) ∈ table_data → y = a * x^2 + b * x + c) := 
sorry

end NUMINAMATH_GPT_matching_function_l575_57588


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l575_57569

theorem quadratic_inequality_solution (x : ℝ) : 16 ≤ x ∧ x ≤ 20 → x^2 - 36 * x + 323 ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l575_57569


namespace NUMINAMATH_GPT_map_representation_l575_57508

-- Defining the conditions
noncomputable def map_scale : ℝ := 28 -- 1 inch represents 28 miles

-- Defining the specific instance provided in the problem
def inches_represented : ℝ := 13.7
def miles_represented : ℝ := 383.6

-- Statement of the problem
theorem map_representation (D : ℝ) : (D / map_scale) = (D : ℝ) / 28 := 
by
  -- Prove the statement
  sorry

end NUMINAMATH_GPT_map_representation_l575_57508


namespace NUMINAMATH_GPT_camryn_flute_practice_interval_l575_57510

theorem camryn_flute_practice_interval (x : ℕ) 
  (h1 : ∃ n : ℕ, n * 11 = 33) 
  (h2 : x ∣ 33) 
  (h3 : x < 11) 
  (h4 : x > 1) 
  : x = 3 := 
sorry

end NUMINAMATH_GPT_camryn_flute_practice_interval_l575_57510


namespace NUMINAMATH_GPT_multiply_fractions_l575_57593

theorem multiply_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = (1 / 7) := by
  sorry

end NUMINAMATH_GPT_multiply_fractions_l575_57593


namespace NUMINAMATH_GPT_compute_div_mul_l575_57525

noncomputable def a : ℚ := 0.24
noncomputable def b : ℚ := 0.006

theorem compute_div_mul : ((a / b) * 2) = 80 := by
  sorry

end NUMINAMATH_GPT_compute_div_mul_l575_57525


namespace NUMINAMATH_GPT_total_doughnuts_made_l575_57520

def num_doughnuts_per_box : ℕ := 10
def num_boxes_sold : ℕ := 27
def doughnuts_given_away : ℕ := 30

theorem total_doughnuts_made :
  num_boxes_sold * num_doughnuts_per_box + doughnuts_given_away = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_doughnuts_made_l575_57520


namespace NUMINAMATH_GPT_correct_operation_l575_57579

variable (a : ℝ)

theorem correct_operation : 
  (3 * a^2 + 2 * a^4 ≠ 5 * a^6) ∧
  (a^2 * a^3 ≠ a^6) ∧
  ((2 * a^2)^3 ≠ 6 * a^6) ∧
  ((-2 * a^3)^2 = 4 * a^6) := by
  sorry

end NUMINAMATH_GPT_correct_operation_l575_57579


namespace NUMINAMATH_GPT_triangle_angle_sum_l575_57545

theorem triangle_angle_sum (A : ℕ) (h1 : A = 55) (h2 : ∀ (B : ℕ), B = 2 * A) : (A + 2 * A = 165) :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l575_57545


namespace NUMINAMATH_GPT_point_symmetric_to_line_l575_57538

-- Define the problem statement
theorem point_symmetric_to_line (M : ℝ × ℝ) (l : ℝ × ℝ) (N : ℝ × ℝ) :
  M = (1, 4) →
  l = (1, -1) →
  (∃ a b, N = (a, b) ∧ a + b = 5 ∧ a - b = 1) →
  N = (3, 2) :=
by
  sorry

end NUMINAMATH_GPT_point_symmetric_to_line_l575_57538


namespace NUMINAMATH_GPT_reciprocals_sum_eq_neg_one_over_three_l575_57581

-- Let the reciprocals of the roots of the polynomial 7x^2 + 2x + 6 be alpha and beta.
-- Given that a and b are roots of the polynomial, and alpha = 1/a and beta = 1/b,
-- Prove that alpha + beta = -1/3.

theorem reciprocals_sum_eq_neg_one_over_three
  (a b : ℝ)
  (ha : 7 * a ^ 2 + 2 * a + 6 = 0)
  (hb : 7 * b ^ 2 + 2 * b + 6 = 0)
  (h_sum : a + b = -2 / 7)
  (h_prod : a * b = 6 / 7) :
  (1 / a) + (1 / b) = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_reciprocals_sum_eq_neg_one_over_three_l575_57581


namespace NUMINAMATH_GPT_finish_work_in_time_l575_57521

noncomputable def work_in_days_A (DA : ℕ) := DA
noncomputable def work_in_days_B (DA : ℕ) := DA / 2
noncomputable def combined_work_rate (DA : ℕ) : ℚ := 1 / work_in_days_A DA + 2 / work_in_days_A DA

theorem finish_work_in_time (DA : ℕ) (h_combined_rate : combined_work_rate DA = 0.25) : DA = 12 :=
sorry

end NUMINAMATH_GPT_finish_work_in_time_l575_57521


namespace NUMINAMATH_GPT_unknown_diagonal_length_l575_57550

noncomputable def rhombus_diagonal_length
  (area : ℝ) (d2 : ℝ) : ℝ :=
  (2 * area) / d2

theorem unknown_diagonal_length
  (area : ℝ) (d2 : ℝ) (h_area : area = 150)
  (h_d2 : d2 = 30) :
  rhombus_diagonal_length area d2 = 10 :=
  by
  rw [h_area, h_d2]
  -- Here, the essential proof would go
  -- Since solving would require computation,
  -- which we are omitting, we use:
  sorry

end NUMINAMATH_GPT_unknown_diagonal_length_l575_57550


namespace NUMINAMATH_GPT_solution_count_l575_57599

noncomputable def equation_has_one_solution : Prop :=
∀ x : ℝ, (x - (8 / (x - 2))) = (4 - (8 / (x - 2))) → x = 4

theorem solution_count : equation_has_one_solution :=
by
  sorry

end NUMINAMATH_GPT_solution_count_l575_57599
