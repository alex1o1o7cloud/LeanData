import Mathlib

namespace NUMINAMATH_GPT_y_squared_range_l566_56638

theorem y_squared_range (y : ℝ) 
  (h : Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2) : 
  9200 ≤ y^2 ∧ y^2 ≤ 9400 := 
sorry

end NUMINAMATH_GPT_y_squared_range_l566_56638


namespace NUMINAMATH_GPT_inequality_solution_set_l566_56652

theorem inequality_solution_set (x : ℝ) :
  ((1 / 2 - x) * (x - 1 / 3) > 0) ↔ (1 / 3 < x ∧ x < 1 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l566_56652


namespace NUMINAMATH_GPT_go_to_yolka_together_l566_56654

noncomputable def anya_will_not_wait : Prop := true
noncomputable def boris_wait_time : ℕ := 10 -- in minutes
noncomputable def vasya_wait_time : ℕ := 15 -- in minutes
noncomputable def meeting_time_window : ℕ := 60 -- total available time in minutes

noncomputable def probability_all_go_together : ℝ :=
  (1 / 3) * (3500 / 3600)

theorem go_to_yolka_together :
  anya_will_not_wait ∧
  boris_wait_time = 10 ∧
  vasya_wait_time = 15 ∧
  meeting_time_window = 60 →
  probability_all_go_together = 0.324 :=
by
  intros
  sorry

end NUMINAMATH_GPT_go_to_yolka_together_l566_56654


namespace NUMINAMATH_GPT_find_a_l566_56682

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.sqrt 2

theorem find_a (a : ℝ) (h : f a (f a (Real.sqrt 2)) = -Real.sqrt 2) : 
  a = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l566_56682


namespace NUMINAMATH_GPT_max_k_value_l566_56624

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_k_value :
  (∀ x : ℝ, 0 < x → (∃ k : ℝ, k * x = Real.log x ∧ k ≤ f x)) ∧
  (∀ x : ℝ, 0 < x → f x ≤ 1 / Real.exp 1) ∧
  (∀ x : ℝ, 0 < x → (k = f x → k ≤ 1 / Real.exp 1)) := 
sorry

end NUMINAMATH_GPT_max_k_value_l566_56624


namespace NUMINAMATH_GPT_cone_section_area_half_base_ratio_l566_56623

theorem cone_section_area_half_base_ratio (h_base h_upper h_lower : ℝ) (A_base A_upper : ℝ) 
  (h_total : h_upper + h_lower = h_base)
  (A_upper : A_upper = A_base / 2) :
  h_upper = h_lower :=
by
  sorry

end NUMINAMATH_GPT_cone_section_area_half_base_ratio_l566_56623


namespace NUMINAMATH_GPT_largest_number_is_A_l566_56626

noncomputable def numA : ℝ := 4.25678
noncomputable def numB : ℝ := 4.2567777 -- repeating 7
noncomputable def numC : ℝ := 4.25676767 -- repeating 67
noncomputable def numD : ℝ := 4.25675675 -- repeating 567
noncomputable def numE : ℝ := 4.25672567 -- repeating 2567

theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE := by
  sorry

end NUMINAMATH_GPT_largest_number_is_A_l566_56626


namespace NUMINAMATH_GPT_eagles_points_l566_56610

theorem eagles_points (s e : ℕ) (h1 : s + e = 52) (h2 : s - e = 6) : e = 23 :=
by
  sorry

end NUMINAMATH_GPT_eagles_points_l566_56610


namespace NUMINAMATH_GPT_relay_team_member_distance_l566_56630

theorem relay_team_member_distance (n_people : ℕ) (total_distance : ℕ)
  (h1 : n_people = 5) (h2 : total_distance = 150) : total_distance / n_people = 30 :=
by 
  sorry

end NUMINAMATH_GPT_relay_team_member_distance_l566_56630


namespace NUMINAMATH_GPT_william_shared_marble_count_l566_56672

theorem william_shared_marble_count : ∀ (initial_marbles shared_marbles remaining_marbles : ℕ),
  initial_marbles = 10 → remaining_marbles = 7 → 
  shared_marbles = initial_marbles - remaining_marbles → 
  shared_marbles = 3 := by 
    intros initial_marbles shared_marbles remaining_marbles h_initial h_remaining h_shared
    rw [h_initial, h_remaining] at h_shared
    exact h_shared

end NUMINAMATH_GPT_william_shared_marble_count_l566_56672


namespace NUMINAMATH_GPT_find_initial_books_each_l566_56677

variable (x : ℝ)
variable (sandy_books : ℝ := x)
variable (tim_books : ℝ := 2 * x + 33)
variable (benny_books : ℝ := 3 * x - 24)
variable (total_books : ℝ := 100)

theorem find_initial_books_each :
  sandy_books + tim_books + benny_books = total_books → x = 91 / 6 := by
  sorry

end NUMINAMATH_GPT_find_initial_books_each_l566_56677


namespace NUMINAMATH_GPT_cone_volume_l566_56668

theorem cone_volume (V_cyl : ℝ) (r h : ℝ) (h_cyl : V_cyl = 150 * Real.pi) :
  (1 / 3) * V_cyl = 50 * Real.pi :=
by
  rw [h_cyl]
  ring


end NUMINAMATH_GPT_cone_volume_l566_56668


namespace NUMINAMATH_GPT_minimum_value_expression_l566_56667

theorem minimum_value_expression (x : ℝ) (h : x > 4) : 
  ∃ (m : ℝ), m = 6 ∧ ∀ y : ℝ, y = (x + 5) / (Real.sqrt (x - 4)) → y ≥ m :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l566_56667


namespace NUMINAMATH_GPT_correct_statement_is_d_l566_56676

/-- A definition for all the conditions given in the problem --/
def very_small_real_form_set : Prop := false
def smallest_natural_number_is_one : Prop := false
def sets_equal : Prop := false
def empty_set_subset_of_any_set : Prop := true

/-- The main statement to be proven --/
theorem correct_statement_is_d : (very_small_real_form_set = false) ∧ 
                                 (smallest_natural_number_is_one = false) ∧ 
                                 (sets_equal = false) ∧ 
                                 (empty_set_subset_of_any_set = true) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_is_d_l566_56676


namespace NUMINAMATH_GPT_cos_add_pi_over_4_l566_56637

theorem cos_add_pi_over_4 (α : ℝ) (h : Real.sin (α - π/4) = 1/3) : Real.cos (π/4 + α) = -1/3 := 
  sorry

end NUMINAMATH_GPT_cos_add_pi_over_4_l566_56637


namespace NUMINAMATH_GPT_find_n_l566_56683

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 52) (h2 : Nat.gcd n 16 = 8) : n = 26 := by
  sorry

end NUMINAMATH_GPT_find_n_l566_56683


namespace NUMINAMATH_GPT_ratio_proof_l566_56648

variable (a b c d : ℚ)

-- Given conditions
axiom h1 : b / a = 3
axiom h2 : c / b = 4
axiom h3 : d = 5 * b

-- Theorem to be proved
theorem ratio_proof : (a + b + d) / (b + c + d) = 19 / 30 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_proof_l566_56648


namespace NUMINAMATH_GPT_simplify_expression_l566_56608

theorem simplify_expression (x : ℝ) (h : x^2 + x - 6 = 0) : 
  (x - 1) / ((2 / (x - 1)) - 1) = 8 / 3 :=
sorry

end NUMINAMATH_GPT_simplify_expression_l566_56608


namespace NUMINAMATH_GPT_remaining_shirt_cost_l566_56671

theorem remaining_shirt_cost (total_shirts : ℕ) (cost_3_shirts : ℕ) (total_cost : ℕ) 
  (h1 : total_shirts = 5) 
  (h2 : cost_3_shirts = 3 * 15) 
  (h3 : total_cost = 85) :
  (total_cost - cost_3_shirts) / (total_shirts - 3) = 20 :=
by
  sorry

end NUMINAMATH_GPT_remaining_shirt_cost_l566_56671


namespace NUMINAMATH_GPT_find_z_in_sequence_l566_56616

theorem find_z_in_sequence (x y z a b : ℤ) 
  (h1 : b = 1)
  (h2 : a + b = 0)
  (h3 : y + a = 1)
  (h4 : z + y = 3)
  (h5 : x + z = 2) :
  z = 1 :=
sorry

end NUMINAMATH_GPT_find_z_in_sequence_l566_56616


namespace NUMINAMATH_GPT_tip_percentage_l566_56658

def julie_food_cost : ℝ := 10
def letitia_food_cost : ℝ := 20
def anton_food_cost : ℝ := 30
def julie_tip : ℝ := 4
def letitia_tip : ℝ := 4
def anton_tip : ℝ := 4

theorem tip_percentage : 
  (julie_tip + letitia_tip + anton_tip) / (julie_food_cost + letitia_food_cost + anton_food_cost) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_tip_percentage_l566_56658


namespace NUMINAMATH_GPT_inequality_solution_set_l566_56604

theorem inequality_solution_set (x : ℝ) : 
  (∃ x, (2 < x ∧ x < 3)) ↔ 
  ((x - 2) * (x - 3) / (x^2 + 1) < 0) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l566_56604


namespace NUMINAMATH_GPT_total_interest_rate_is_correct_l566_56678

theorem total_interest_rate_is_correct :
  let total_investment := 100000
  let interest_rate_first := 0.09
  let interest_rate_second := 0.11
  let invested_in_second := 29999.999999999993
  let invested_in_first := total_investment - invested_in_second
  let interest_first := invested_in_first * interest_rate_first
  let interest_second := invested_in_second * interest_rate_second
  let total_interest := interest_first + interest_second
  let total_interest_rate := (total_interest / total_investment) * 100
  total_interest_rate = 9.6 :=
by
  sorry

end NUMINAMATH_GPT_total_interest_rate_is_correct_l566_56678


namespace NUMINAMATH_GPT_connie_total_markers_l566_56621

def red_markers : ℕ := 5420
def blue_markers : ℕ := 3875
def green_markers : ℕ := 2910
def yellow_markers : ℕ := 6740

def total_markers : ℕ := red_markers + blue_markers + green_markers + yellow_markers

theorem connie_total_markers : total_markers = 18945 := by
  sorry

end NUMINAMATH_GPT_connie_total_markers_l566_56621


namespace NUMINAMATH_GPT_distance_downstream_in_12min_l566_56636

-- Define the given constants
def boat_speed_still_water : ℝ := 15  -- km/hr
def current_speed : ℝ := 3  -- km/hr
def time_minutes : ℝ := 12  -- minutes

-- Prove the distance traveled downstream in 12 minutes
theorem distance_downstream_in_12min
  (b_velocity_still : ℝ)
  (c_velocity : ℝ)
  (time_m : ℝ)
  (h1 : b_velocity_still = boat_speed_still_water)
  (h2 : c_velocity = current_speed)
  (h3 : time_m = time_minutes) :
  let effective_speed := b_velocity_still + c_velocity
  let effective_speed_km_per_min := effective_speed / 60
  let distance := effective_speed_km_per_min * time_m
  distance = 3.6 :=
by
  sorry

end NUMINAMATH_GPT_distance_downstream_in_12min_l566_56636


namespace NUMINAMATH_GPT_polar_coordinates_of_point_l566_56662

theorem polar_coordinates_of_point {x y : ℝ} (hx : x = -3) (hy : y = 1) :
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.pi - Real.arctan (y / abs x)
  r = Real.sqrt 10 ∧ θ = Real.pi - Real.arctan (1 / 3) := 
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_polar_coordinates_of_point_l566_56662


namespace NUMINAMATH_GPT_Lee_surpasses_Hernandez_in_May_l566_56656

def monthly_totals_Hernandez : List ℕ :=
  [4, 8, 9, 5, 7, 6]

def monthly_totals_Lee : List ℕ :=
  [3, 9, 10, 6, 8, 8]

def cumulative_sum (lst : List ℕ) : List ℕ :=
  List.scanl (· + ·) 0 lst

noncomputable def cumulative_Hernandez := cumulative_sum monthly_totals_Hernandez
noncomputable def cumulative_Lee := cumulative_sum monthly_totals_Lee

-- Lean 4 statement asserting when Lee surpasses Hernandez in cumulative home runs
theorem Lee_surpasses_Hernandez_in_May :
  cumulative_Hernandez[3] < cumulative_Lee[3] :=
sorry

end NUMINAMATH_GPT_Lee_surpasses_Hernandez_in_May_l566_56656


namespace NUMINAMATH_GPT_log_equation_solution_l566_56635

theorem log_equation_solution (x : ℝ) (hx : 0 < x) :
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 ↔ (x = 4 ∨ x = 8) :=
by
  sorry

end NUMINAMATH_GPT_log_equation_solution_l566_56635


namespace NUMINAMATH_GPT_problem_statement_l566_56657

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : ∀ x, 0 < x → f a b x ≥ f a b 1) : 
  Real.log a < -2 * b :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l566_56657


namespace NUMINAMATH_GPT_fraction_order_l566_56612

theorem fraction_order :
  (19 / 15 < 17 / 13) ∧ (17 / 13 < 15 / 11) :=
by
  sorry

end NUMINAMATH_GPT_fraction_order_l566_56612


namespace NUMINAMATH_GPT_solve_absolute_inequality_l566_56685

theorem solve_absolute_inequality (x : ℝ) : |x - 1| - |x - 2| > 1 / 2 ↔ x > 7 / 4 :=
by sorry

end NUMINAMATH_GPT_solve_absolute_inequality_l566_56685


namespace NUMINAMATH_GPT_JuliaPlayedTuesday_l566_56609

variable (Monday : ℕ) (Wednesday : ℕ) (Total : ℕ)
variable (KidsOnTuesday : ℕ)

theorem JuliaPlayedTuesday :
  Monday = 17 →
  Wednesday = 2 →
  Total = 34 →
  KidsOnTuesday = Total - (Monday + Wednesday) →
  KidsOnTuesday = 15 :=
by
  intros hMon hWed hTot hTue
  rw [hTot, hMon, hWed] at hTue
  exact hTue

end NUMINAMATH_GPT_JuliaPlayedTuesday_l566_56609


namespace NUMINAMATH_GPT_solve_eq_l566_56684

theorem solve_eq (x a b : ℝ) (h₁ : x^2 + 10 * x = 34) (h₂ : a = 59) (h₃ : b = 5) :
  a + b = 64 :=
by {
  -- insert proof here, eventually leading to a + b = 64
  sorry
}

end NUMINAMATH_GPT_solve_eq_l566_56684


namespace NUMINAMATH_GPT_find_two_digit_number_l566_56634

theorem find_two_digit_number (x y a b : ℕ) :
  10 * x + y + 46 = 10 * a + b →
  a * b = 6 →
  a + b = 14 →
  (x = 7 ∧ y = 7) ∨ (x = 8 ∧ y = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_two_digit_number_l566_56634


namespace NUMINAMATH_GPT_range_of_a_l566_56665

def quadratic_function (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l566_56665


namespace NUMINAMATH_GPT_sum_ap_series_l566_56693

-- Definition of the arithmetic progression sum for given parameters
def ap_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Specific sum calculation for given p
def S_p (p : ℕ) : ℕ :=
  ap_sum p (2 * p - 1) 40

-- Total sum from p = 1 to p = 10
def total_sum : ℕ :=
  (Finset.range 10).sum (λ i => S_p (i + 1))

-- The theorem stating the desired proof
theorem sum_ap_series : total_sum = 80200 := by
  sorry

end NUMINAMATH_GPT_sum_ap_series_l566_56693


namespace NUMINAMATH_GPT_student_chose_number_l566_56659

theorem student_chose_number :
  ∃ x : ℕ, 7 * x - 150 = 130 ∧ x = 40 := sorry

end NUMINAMATH_GPT_student_chose_number_l566_56659


namespace NUMINAMATH_GPT_smallest_solution_equation_l566_56641

noncomputable def equation (x : ℝ) : ℝ :=
  (3*x / (x-3)) + ((3*x^2 - 45) / x) + 3

theorem smallest_solution_equation : 
  ∃ x : ℝ, equation x = 14 ∧ x = (1 - Real.sqrt 649) / 12 :=
sorry

end NUMINAMATH_GPT_smallest_solution_equation_l566_56641


namespace NUMINAMATH_GPT_ivan_income_tax_l566_56669

theorem ivan_income_tax :
  let salary_probation := 20000
  let probation_months := 2
  let salary_after_probation := 25000
  let after_probation_months := 8
  let bonus := 10000
  let tax_rate := 0.13
  let total_income := salary_probation * probation_months +
                      salary_after_probation * after_probation_months + bonus
  total_income * tax_rate = 32500 := sorry

end NUMINAMATH_GPT_ivan_income_tax_l566_56669


namespace NUMINAMATH_GPT_parabola_equation_l566_56696

theorem parabola_equation (p x0 : ℝ) (h_p : p > 0) (h_dist_focus : x0 + p / 2 = 10) (h_parabola : 2 * p * x0 = 36) :
  (2 * p = 4) ∨ (2 * p = 36) :=
by sorry

end NUMINAMATH_GPT_parabola_equation_l566_56696


namespace NUMINAMATH_GPT_problem_solution_l566_56688

theorem problem_solution
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d :=
sorry

end NUMINAMATH_GPT_problem_solution_l566_56688


namespace NUMINAMATH_GPT_andrew_paid_correct_amount_l566_56664

-- Definitions of the conditions
def cost_of_grapes : ℝ := 7 * 68
def cost_of_mangoes : ℝ := 9 * 48
def cost_of_apples : ℝ := 5 * 55
def cost_of_oranges : ℝ := 4 * 38

def total_cost_grapes_and_mangoes_before_discount : ℝ := cost_of_grapes + cost_of_mangoes
def discount_on_grapes_and_mangoes : ℝ := 0.10 * total_cost_grapes_and_mangoes_before_discount
def total_cost_grapes_and_mangoes_after_discount : ℝ := total_cost_grapes_and_mangoes_before_discount - discount_on_grapes_and_mangoes

def total_cost_all_fruits_before_tax : ℝ := total_cost_grapes_and_mangoes_after_discount + cost_of_apples + cost_of_oranges
def sales_tax : ℝ := 0.05 * total_cost_all_fruits_before_tax
def total_amount_to_pay : ℝ := total_cost_all_fruits_before_tax + sales_tax

-- Statement to be proved
theorem andrew_paid_correct_amount :
  total_amount_to_pay = 1306.41 :=
by
  sorry

end NUMINAMATH_GPT_andrew_paid_correct_amount_l566_56664


namespace NUMINAMATH_GPT_find_numbers_l566_56617

theorem find_numbers (A B C : ℝ) 
  (h1 : A - B = 1860) 
  (h2 : 0.075 * A = 0.125 * B) 
  (h3 : 0.15 * B = 0.05 * C) : 
  A = 4650 ∧ B = 2790 ∧ C = 8370 := 
by
  sorry

end NUMINAMATH_GPT_find_numbers_l566_56617


namespace NUMINAMATH_GPT_tan_neg_480_eq_sqrt_3_l566_56690

theorem tan_neg_480_eq_sqrt_3 : Real.tan (-8 * Real.pi / 3) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_neg_480_eq_sqrt_3_l566_56690


namespace NUMINAMATH_GPT_find_cookbooks_stashed_in_kitchen_l566_56698

-- Definitions of the conditions
def total_books := 99
def books_in_boxes := 3 * 15
def books_in_room := 21
def books_on_table := 4
def books_picked_up := 12
def current_books := 23

-- Main statement
theorem find_cookbooks_stashed_in_kitchen :
  let books_donated := books_in_boxes + books_in_room + books_on_table
  let books_left_initial := total_books - books_donated
  let books_left_before_pickup := current_books - books_picked_up
  books_left_initial - books_left_before_pickup = 18 := by
  sorry

end NUMINAMATH_GPT_find_cookbooks_stashed_in_kitchen_l566_56698


namespace NUMINAMATH_GPT_additional_pencils_l566_56687

theorem additional_pencils (original_pencils new_pencils per_container distributed_pencils : ℕ)
  (h1 : original_pencils = 150)
  (h2 : per_container = 5)
  (h3 : distributed_pencils = 36)
  (h4 : new_pencils = distributed_pencils * per_container) :
  (new_pencils - original_pencils) = 30 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_additional_pencils_l566_56687


namespace NUMINAMATH_GPT_train_b_speed_l566_56633

variable (v : ℝ) -- the speed of Train B

theorem train_b_speed 
  (speedA : ℝ := 30) -- speed of Train A
  (head_start_hours : ℝ := 2) -- head start time in hours
  (overtake_distance : ℝ := 285) -- distance at which Train B overtakes Train A
  (train_a_travel_distance : ℝ := speedA * head_start_hours) -- distance Train A travels in the head start time
  (total_distance : ℝ := 345) -- total distance Train B travels to overtake Train A
  (train_a_travel_time : ℝ := overtake_distance / speedA) -- time taken by Train A to travel the overtake distance
  : v * train_a_travel_time = total_distance → v = 36.32 :=
by
  sorry

end NUMINAMATH_GPT_train_b_speed_l566_56633


namespace NUMINAMATH_GPT_abs_h_eq_one_l566_56674

theorem abs_h_eq_one (h : ℝ) (roots_square_sum_eq : ∀ x : ℝ, x^2 + 6 * h * x + 8 = 0 → x^2 + (x + 6 * h)^2 = 20) : |h| = 1 :=
by
  sorry

end NUMINAMATH_GPT_abs_h_eq_one_l566_56674


namespace NUMINAMATH_GPT_sum_of_8th_and_10th_terms_arithmetic_sequence_l566_56679

theorem sum_of_8th_and_10th_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 25) (h2 : a + 5 * d = 61) :
  (a + 7 * d) + (a + 9 * d) = 230 := 
sorry

end NUMINAMATH_GPT_sum_of_8th_and_10th_terms_arithmetic_sequence_l566_56679


namespace NUMINAMATH_GPT_cos2_alpha_add_sin2_alpha_eq_eight_over_five_l566_56631

theorem cos2_alpha_add_sin2_alpha_eq_eight_over_five (x y : ℝ) (r : ℝ) (α : ℝ) 
(hx : x = 2) 
(hy : y = 1)
(hr : r = Real.sqrt (x^2 + y^2))
(hcos : Real.cos α = x / r)
(hsin : Real.sin α = y / r) :
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
sorry

end NUMINAMATH_GPT_cos2_alpha_add_sin2_alpha_eq_eight_over_five_l566_56631


namespace NUMINAMATH_GPT_parallel_line_through_point_l566_56622

theorem parallel_line_through_point (x y : ℝ) (m b : ℝ) (h₁ : y = -3 * x + b) (h₂ : x = 2) (h₃ : y = 1) :
  b = 7 :=
by
  -- x, y are components of the point P (2,1)
  -- equation of line parallel to y = -3x + 2 has slope -3 but different y-intercept
  -- y = -3x + b is the general form, and must pass through (2,1) => 1 = -3*2 + b
  -- Therefore, b must be 7
  sorry

end NUMINAMATH_GPT_parallel_line_through_point_l566_56622


namespace NUMINAMATH_GPT_return_time_is_2_hours_l566_56645

noncomputable def distance_home_city_hall := 6
noncomputable def speed_to_city_hall := 3 -- km/h
noncomputable def additional_distance_return := 2 -- km
noncomputable def speed_return := 4 -- km/h
noncomputable def total_trip_time := 4 -- hours

theorem return_time_is_2_hours :
  (distance_home_city_hall + additional_distance_return) / speed_return = 2 :=
by
  sorry

end NUMINAMATH_GPT_return_time_is_2_hours_l566_56645


namespace NUMINAMATH_GPT_parabola_vertex_on_x_axis_l566_56611

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2 * x + c = 0) → c = 1 := by
  sorry

end NUMINAMATH_GPT_parabola_vertex_on_x_axis_l566_56611


namespace NUMINAMATH_GPT_cubic_sum_l566_56680

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_l566_56680


namespace NUMINAMATH_GPT_largest_n_divisible_l566_56646

theorem largest_n_divisible : ∃ n : ℕ, (∀ k : ℕ, (k^3 + 150) % (k + 5) = 0 → k ≤ n) ∧ n = 20 := 
by
  sorry

end NUMINAMATH_GPT_largest_n_divisible_l566_56646


namespace NUMINAMATH_GPT_rationalize_denominator_l566_56628

theorem rationalize_denominator (h : ∀ x: ℝ, x = 1 / (Real.sqrt 3 - 2)) : 
    1 / (Real.sqrt 3 - 2) = - Real.sqrt 3 - 2 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l566_56628


namespace NUMINAMATH_GPT_raisins_in_other_boxes_l566_56602

theorem raisins_in_other_boxes (total_raisins : ℕ) (raisins_box1 : ℕ) (raisins_box2 : ℕ) (other_boxes : ℕ) (num_other_boxes : ℕ) :
  total_raisins = 437 →
  raisins_box1 = 72 →
  raisins_box2 = 74 →
  num_other_boxes = 3 →
  other_boxes = (total_raisins - raisins_box1 - raisins_box2) / num_other_boxes →
  other_boxes = 97 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_raisins_in_other_boxes_l566_56602


namespace NUMINAMATH_GPT_a5_value_l566_56675

variable {a : ℕ → ℝ} (q : ℝ) (a2 a3 : ℝ)

-- Assume the conditions: geometric sequence, a_2 = 2, a_3 = -4
def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∃ q, ∀ n, a (n + 1) = a n * q

-- Given conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 2 = 2
axiom h3 : a 3 = -4

-- Theorem to prove
theorem a5_value : a 5 = -16 :=
by
  -- Here you would provide the proof based on the conditions
  sorry

end NUMINAMATH_GPT_a5_value_l566_56675


namespace NUMINAMATH_GPT_readers_scifi_l566_56615

variable (S L B T : ℕ)

-- Define conditions given in the problem
def totalReaders := 650
def literaryReaders := 550
def bothReaders := 150

-- Define the main problem to prove
theorem readers_scifi (S L B T : ℕ) (hT : T = totalReaders) (hL : L = literaryReaders) (hB : B = bothReaders) (hleq : T = S + L - B) : S = 250 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_readers_scifi_l566_56615


namespace NUMINAMATH_GPT_magnitude_of_z_l566_56627

open Complex -- open the complex number namespace

theorem magnitude_of_z (z : ℂ) (h : z + I = 3) : Complex.abs z = Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_z_l566_56627


namespace NUMINAMATH_GPT_fraction_of_remaining_supplies_used_l566_56600

theorem fraction_of_remaining_supplies_used 
  (initial_food : ℕ)
  (food_used_first_day_fraction : ℚ)
  (food_remaining_after_three_days : ℕ) 
  (food_used_second_period_fraction : ℚ) :
  initial_food = 400 →
  food_used_first_day_fraction = 2 / 5 →
  food_remaining_after_three_days = 96 →
  (initial_food - initial_food * food_used_first_day_fraction) * (1 - food_used_second_period_fraction) = food_remaining_after_three_days →
  food_used_second_period_fraction = 3 / 5 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fraction_of_remaining_supplies_used_l566_56600


namespace NUMINAMATH_GPT_members_with_both_non_athletic_parents_l566_56661

-- Let's define the conditions
variable (total_members athletic_dads athletic_moms both_athletic none_have_dads : ℕ)
variable (H1 : total_members = 50)
variable (H2 : athletic_dads = 25)
variable (H3 : athletic_moms = 30)
variable (H4 : both_athletic = 10)
variable (H5 : none_have_dads = 5)

-- Define the conclusion we want to prove
theorem members_with_both_non_athletic_parents : 
  (total_members - (athletic_dads + athletic_moms - both_athletic) + none_have_dads - total_members) = 10 :=
sorry

end NUMINAMATH_GPT_members_with_both_non_athletic_parents_l566_56661


namespace NUMINAMATH_GPT_willam_land_percentage_l566_56689

-- Definitions from conditions
def farm_tax_rate : ℝ := 0.6
def total_tax_collected : ℝ := 3840
def mr_willam_tax_paid : ℝ := 500

-- Goal to prove: percentage of Mr. Willam's land over total taxable land of the village
noncomputable def percentage_mr_willam_land : ℝ :=
  (mr_willam_tax_paid / total_tax_collected) * 100

theorem willam_land_percentage :
  percentage_mr_willam_land = 13.02 := 
  by 
  sorry

end NUMINAMATH_GPT_willam_land_percentage_l566_56689


namespace NUMINAMATH_GPT_razorback_tshirt_shop_sales_l566_56647

theorem razorback_tshirt_shop_sales :
  let price_per_tshirt := 16 
  let tshirts_sold := 45 
  price_per_tshirt * tshirts_sold = 720 :=
by
  sorry

end NUMINAMATH_GPT_razorback_tshirt_shop_sales_l566_56647


namespace NUMINAMATH_GPT_find_m_for_parallel_lines_l566_56601

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, (3 + m) * x + 4 * y = 5 - 3 * m) →
  (∀ x y : ℝ, 2 * x + (5 + m) * y = 8) →
  m = -7 :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_parallel_lines_l566_56601


namespace NUMINAMATH_GPT_order_of_x_given_conditions_l566_56691

variables (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)

def system_equations :=
  x₁ + x₂ + x₃ = a₁ ∧
  x₂ + x₃ + x₄ = a₂ ∧
  x₃ + x₄ + x₅ = a₃ ∧
  x₄ + x₅ + x₁ = a₄ ∧
  x₅ + x₁ + x₂ = a₅

def a_descending_order :=
  a₁ > a₂ ∧
  a₂ > a₃ ∧
  a₃ > a₄ ∧
  a₄ > a₅

theorem order_of_x_given_conditions (h₁ : system_equations x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅) :
  a_descending_order a₁ a₂ a₃ a₄ a₅ →
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ := sorry

end NUMINAMATH_GPT_order_of_x_given_conditions_l566_56691


namespace NUMINAMATH_GPT_distance_between_cities_l566_56695

theorem distance_between_cities
  (S : ℤ)
  (h1 : ∀ x : ℤ, 0 ≤ x ∧ x ≤ S → (gcd x (S - x) = 1 ∨ gcd x (S - x) = 3 ∨ gcd x (S - x) = 13)) :
  S = 39 := 
sorry

end NUMINAMATH_GPT_distance_between_cities_l566_56695


namespace NUMINAMATH_GPT_calculate_value_l566_56613

theorem calculate_value (a b c : ℤ) (h₁ : a = 5) (h₂ : b = -3) (h₃ : c = 4) : 2 * c / (a + b) = 4 :=
by
  rw [h₁, h₂, h₃]
  sorry

end NUMINAMATH_GPT_calculate_value_l566_56613


namespace NUMINAMATH_GPT_average_donation_is_integer_l566_56660

variable (num_classes : ℕ) (students_per_class : ℕ) (num_teachers : ℕ) (total_donation : ℕ)

def valid_students (n : ℕ) : Prop := 30 < n ∧ n ≤ 45

theorem average_donation_is_integer (h_classes : num_classes = 14)
                                    (h_teachers : num_teachers = 35)
                                    (h_donation : total_donation = 1995)
                                    (h_students_per_class : valid_students students_per_class)
                                    (h_total_people : ∃ n, 
                                      n = num_teachers + num_classes * students_per_class ∧ 30 < students_per_class ∧ students_per_class ≤ 45) :
  total_donation % (num_teachers + num_classes * students_per_class) = 0 ∧ 
  total_donation / (num_teachers + num_classes * students_per_class) = 3 := 
sorry

end NUMINAMATH_GPT_average_donation_is_integer_l566_56660


namespace NUMINAMATH_GPT_hannahs_weekly_pay_l566_56640

-- Define conditions
def hourly_wage : ℕ := 30
def total_hours : ℕ := 18
def dock_per_late : ℕ := 5
def late_times : ℕ := 3

-- The amount paid after deductions for being late
def pay_after_deductions : ℕ :=
  let wage_before_deductions := hourly_wage * total_hours
  let total_dock := dock_per_late * late_times
  wage_before_deductions - total_dock

-- The proof statement
theorem hannahs_weekly_pay : pay_after_deductions = 525 := 
  by
  -- No proof necessary; statement and conditions must be correctly written to run
  sorry

end NUMINAMATH_GPT_hannahs_weekly_pay_l566_56640


namespace NUMINAMATH_GPT_combined_stickers_leftover_l566_56642

theorem combined_stickers_leftover (r p g : ℕ) (h_r : r % 5 = 1) (h_p : p % 5 = 4) (h_g : g % 5 = 3) :
  (r + p + g) % 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_combined_stickers_leftover_l566_56642


namespace NUMINAMATH_GPT_multiple_of_2_and_3_is_divisible_by_6_l566_56632

theorem multiple_of_2_and_3_is_divisible_by_6 (n : ℤ) (h1 : n % 2 = 0) (h2 : n % 3 = 0) : n % 6 = 0 :=
sorry

end NUMINAMATH_GPT_multiple_of_2_and_3_is_divisible_by_6_l566_56632


namespace NUMINAMATH_GPT_total_tiles_l566_56651

theorem total_tiles (s : ℕ) (H1 : 2 * s - 1 = 57) : s^2 = 841 := by
  sorry

end NUMINAMATH_GPT_total_tiles_l566_56651


namespace NUMINAMATH_GPT_find_integer_modulo_l566_56686

theorem find_integer_modulo : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 123456 [MOD 11] := by
  use 3
  sorry

end NUMINAMATH_GPT_find_integer_modulo_l566_56686


namespace NUMINAMATH_GPT_wyatt_total_envelopes_l566_56673

theorem wyatt_total_envelopes :
  let b := 10
  let y := b - 4
  let t := b + y
  t = 16 :=
by
  let b := 10
  let y := b - 4
  let t := b + y
  sorry

end NUMINAMATH_GPT_wyatt_total_envelopes_l566_56673


namespace NUMINAMATH_GPT_donation_percentage_l566_56663

noncomputable def income : ℝ := 266666.67
noncomputable def remaining_income : ℝ := 0.25 * income
noncomputable def final_amount : ℝ := 40000

theorem donation_percentage :
  ∃ D : ℝ, D = 40 /\ (1 - D / 100) * remaining_income = final_amount :=
by
  sorry

end NUMINAMATH_GPT_donation_percentage_l566_56663


namespace NUMINAMATH_GPT_slope_of_line_through_origin_and_center_l566_56605

def Point := (ℝ × ℝ)

def is_center (p : Point) : Prop :=
  p = (3, 1)

def is_dividing_line (l : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, l x = y → y / x = 1 / 3

theorem slope_of_line_through_origin_and_center :
  ∃ l : ℝ → ℝ, (∀ p1 p2 : Point,
  p1 = (0, 0) →
  p2 = (3, 1) →
  is_center p2 →
  is_dividing_line l) :=
sorry

end NUMINAMATH_GPT_slope_of_line_through_origin_and_center_l566_56605


namespace NUMINAMATH_GPT_quadratic_root_value_l566_56681

theorem quadratic_root_value
  (a : ℝ) 
  (h : a^2 + 3 * a - 1010 = 0) :
  2 * a^2 + 6 * a + 4 = 2024 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_value_l566_56681


namespace NUMINAMATH_GPT_find_a_value_l566_56639

-- Define the conditions
def inverse_variation (a b : ℝ) : Prop := ∃ k : ℝ, a * b^3 = k

-- Define the proof problem
theorem find_a_value
  (a b : ℝ)
  (h1 : inverse_variation a b)
  (h2 : a = 4)
  (h3 : b = 1) :
  ∃ a', a' = 1 / 2 ∧ inverse_variation a' 2 := 
sorry

end NUMINAMATH_GPT_find_a_value_l566_56639


namespace NUMINAMATH_GPT_find_natural_n_l566_56699

theorem find_natural_n (n : ℕ) :
  (992768 ≤ n ∧ n ≤ 993791) ↔ 
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) = 1991) := sorry

end NUMINAMATH_GPT_find_natural_n_l566_56699


namespace NUMINAMATH_GPT_butterfat_milk_mixture_l566_56603

theorem butterfat_milk_mixture :
  ∃ (x : ℝ), 0.10 * x + 0.45 * 8 = 0.20 * (x + 8) ∧ x = 20 := by
  sorry

end NUMINAMATH_GPT_butterfat_milk_mixture_l566_56603


namespace NUMINAMATH_GPT_combined_salaries_l566_56650

theorem combined_salaries (A B C D E : ℝ) 
  (hA : A = 9000) 
  (h_avg : (A + B + C + D + E) / 5 = 8200) :
  (B + C + D + E) = 32000 :=
by
  sorry

end NUMINAMATH_GPT_combined_salaries_l566_56650


namespace NUMINAMATH_GPT_f_10_half_l566_56643

noncomputable def f (x : ℝ) : ℝ := x^2 / (2 * x + 1)
noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | n + 1 => f (fn n x)

theorem f_10_half :
  fn 10 (1 / 2) = 1 / (3 ^ 1024 - 1) :=
sorry

end NUMINAMATH_GPT_f_10_half_l566_56643


namespace NUMINAMATH_GPT_second_train_speed_l566_56666

variable (t v : ℝ)

-- Defining the first condition: 20t = vt + 55
def condition1 : Prop := 20 * t = v * t + 55

-- Defining the second condition: 20t + vt = 495
def condition2 : Prop := 20 * t + v * t = 495

-- Prove that the speed of the second train is 16 km/hr under given conditions
theorem second_train_speed : ∃ t : ℝ, condition1 t 16 ∧ condition2 t 16 := sorry

end NUMINAMATH_GPT_second_train_speed_l566_56666


namespace NUMINAMATH_GPT_probability_XOXOXOX_is_1_div_35_l566_56629

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end NUMINAMATH_GPT_probability_XOXOXOX_is_1_div_35_l566_56629


namespace NUMINAMATH_GPT_distribution_plans_l566_56653

theorem distribution_plans (teachers schools : ℕ) (h_teachers : teachers = 3) (h_schools : schools = 6) : 
  ∃ plans : ℕ, plans = 210 :=
by
  sorry

end NUMINAMATH_GPT_distribution_plans_l566_56653


namespace NUMINAMATH_GPT_find_m_value_l566_56606

theorem find_m_value :
  ∃ (m : ℝ), (∃ (midpoint: ℝ × ℝ), midpoint = ((5 + m) / 2, 1) ∧ midpoint.1 - 2 * midpoint.2 = 0) -> m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l566_56606


namespace NUMINAMATH_GPT_alster_caught_two_frogs_l566_56670

-- Definitions and conditions
variables (alster quinn bret : ℕ)

-- Condition 1: Quinn catches twice the amount of frogs as Alster
def quinn_catches_twice_as_alster : Prop := quinn = 2 * alster

-- Condition 2: Bret catches three times the amount of frogs as Quinn
def bret_catches_three_times_as_quinn : Prop := bret = 3 * quinn

-- Condition 3: Bret caught 12 frogs
def bret_caught_twelve : Prop := bret = 12

-- Theorem: How many frogs did Alster catch? Alster caught 2 frogs
theorem alster_caught_two_frogs (h1 : quinn_catches_twice_as_alster alster quinn)
                                (h2 : bret_catches_three_times_as_quinn quinn bret)
                                (h3 : bret_caught_twelve bret) :
                                alster = 2 :=
by sorry

end NUMINAMATH_GPT_alster_caught_two_frogs_l566_56670


namespace NUMINAMATH_GPT_digging_foundation_l566_56618

-- Define given conditions
variable (m1 d1 m2 d2 k : ℝ)
variable (md_proportionality : m1 * d1 = k)
variable (k_value : k = 20 * 6)

-- Prove that for 30 men, it takes 4 days to dig the foundation
theorem digging_foundation : m1 = 20 ∧ d1 = 6 ∧ m2 = 30 → d2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_digging_foundation_l566_56618


namespace NUMINAMATH_GPT_round_trip_by_car_time_l566_56655

variable (time_walk time_car : ℕ)
variable (h1 : time_walk + time_car = 20)
variable (h2 : 2 * time_walk = 32)

theorem round_trip_by_car_time : 2 * time_car = 8 :=
by
  sorry

end NUMINAMATH_GPT_round_trip_by_car_time_l566_56655


namespace NUMINAMATH_GPT_abc_value_l566_56614

variables (a b c : ℝ)

theorem abc_value (h1 : a * (b + c) = 156) (h2 : b * (c + a) = 168) (h3 : c * (a + b) = 180) :
  a * b * c = 288 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_abc_value_l566_56614


namespace NUMINAMATH_GPT_cube_remainder_l566_56607

theorem cube_remainder (n : ℤ) (h : n % 13 = 5) : (n^3) % 17 = 6 :=
by
  sorry

end NUMINAMATH_GPT_cube_remainder_l566_56607


namespace NUMINAMATH_GPT_combined_weight_is_correct_l566_56697

def EvanDogWeight := 63
def IvanDogWeight := EvanDogWeight / 7
def CombinedWeight := EvanDogWeight + IvanDogWeight

theorem combined_weight_is_correct 
: CombinedWeight = 72 :=
by 
  sorry

end NUMINAMATH_GPT_combined_weight_is_correct_l566_56697


namespace NUMINAMATH_GPT_closest_distance_l566_56649

theorem closest_distance (x y z : ℕ)
  (h1 : x + y = 10)
  (h2 : y + z = 13)
  (h3 : z + x = 11) :
  min x (min y z) = 4 :=
by
  -- Here you would provide the proof steps in Lean, but for the statement itself, we leave it as sorry.
  sorry

end NUMINAMATH_GPT_closest_distance_l566_56649


namespace NUMINAMATH_GPT_cost_price_l566_56625

theorem cost_price (MP SP C : ℝ) (h1 : MP = 112.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) : 
  C = 85.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_l566_56625


namespace NUMINAMATH_GPT_max_value_f_value_of_f_at_alpha_l566_56620

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * Real.sin x

theorem max_value_f :
  (∀ x, f x ≤ 3)
  ∧ (∃ x, f x = 3)
  ∧ {x : ℝ | ∃ k : ℤ, x = (π / 3) + 2 * k * π} = {x : ℝ | ∃ k : ℤ, x = (π / 3) + 2 * k * π} :=
sorry

theorem value_of_f_at_alpha {α : ℝ} (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end NUMINAMATH_GPT_max_value_f_value_of_f_at_alpha_l566_56620


namespace NUMINAMATH_GPT_find_fourth_mark_l566_56694

-- Definitions of conditions
def average_of_four (a b c d : ℕ) : Prop :=
  (a + b + c + d) / 4 = 60

def known_marks (a b c : ℕ) : Prop :=
  a = 30 ∧ b = 55 ∧ c = 65

-- Theorem statement
theorem find_fourth_mark {d : ℕ} (h_avg : average_of_four 30 55 65 d) (h_known : known_marks 30 55 65) : d = 90 := 
by 
  sorry

end NUMINAMATH_GPT_find_fourth_mark_l566_56694


namespace NUMINAMATH_GPT_sum_remainder_l566_56692

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 12) (h2 : b % 30 = 9) (h3 : c % 30 = 15) :
  (a + b + c) % 30 = 6 := 
sorry

end NUMINAMATH_GPT_sum_remainder_l566_56692


namespace NUMINAMATH_GPT_find_y_l566_56644

variable (α : ℝ) (y : ℝ)
axiom sin_alpha_neg_half : Real.sin α = -1 / 2
axiom point_on_terminal_side : 2^2 + y^2 = (Real.sin α)^2 + (Real.cos α)^2

theorem find_y : y = -2 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_l566_56644


namespace NUMINAMATH_GPT_f_diff_eq_l566_56619

def f (n : ℕ) : ℚ := 1 / 4 * (n * (n + 1) * (n + 3))

theorem f_diff_eq (r : ℕ) : 
  f (r + 1) - f r = 1 / 4 * (3 * r^2 + 11 * r + 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_f_diff_eq_l566_56619
