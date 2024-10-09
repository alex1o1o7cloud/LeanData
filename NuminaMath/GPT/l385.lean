import Mathlib

namespace arnold_total_protein_l385_38513

-- Conditions
def protein_in_collagen_powder (scoops: ℕ) : ℕ := 9 * scoops
def protein_in_protein_powder (scoops: ℕ) : ℕ := 21 * scoops
def protein_in_steak : ℕ := 56
def protein_in_greek_yogurt : ℕ := 15
def protein_in_almonds (cups: ℕ) : ℕ := 6 * (cups * 4) / 4
def half_cup_almonds_protein : ℕ := 12

-- Statement
theorem arnold_total_protein : 
  protein_in_collagen_powder 1 + protein_in_protein_powder 2 + protein_in_steak + protein_in_greek_yogurt + half_cup_almonds_protein = 134 :=
  by
    sorry

end arnold_total_protein_l385_38513


namespace sqrt_sum_4_pow_4_eq_32_l385_38571

theorem sqrt_sum_4_pow_4_eq_32 : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 :=
by
  sorry

end sqrt_sum_4_pow_4_eq_32_l385_38571


namespace problem1_problem2_l385_38589

-- Problem 1
theorem problem1 (x : ℝ) : (1 : ℝ) * (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := 
sorry

-- Problem 2
theorem problem2 (a b : ℝ) : (a - b)^2 * (b - a)^4 + (b - a)^3 * (a - b)^3 = 0 := 
sorry

end problem1_problem2_l385_38589


namespace ravi_overall_profit_l385_38502

-- Define the purchase prices
def refrigerator_purchase_price := 15000
def mobile_phone_purchase_price := 8000

-- Define the percentages
def refrigerator_loss_percent := 2
def mobile_phone_profit_percent := 10

-- Define the calculations for selling prices
def refrigerator_loss_amount := (refrigerator_loss_percent / 100) * refrigerator_purchase_price
def refrigerator_selling_price := refrigerator_purchase_price - refrigerator_loss_amount

def mobile_phone_profit_amount := (mobile_phone_profit_percent / 100) * mobile_phone_purchase_price
def mobile_phone_selling_price := mobile_phone_purchase_price + mobile_phone_profit_amount

-- Define the total purchase and selling prices
def total_purchase_price := refrigerator_purchase_price + mobile_phone_purchase_price
def total_selling_price := refrigerator_selling_price + mobile_phone_selling_price

-- Define the overall profit calculation
def overall_profit := total_selling_price - total_purchase_price

-- Statement of the theorem
theorem ravi_overall_profit :
  overall_profit = 500 := by
  sorry

end ravi_overall_profit_l385_38502


namespace gcd_111_148_l385_38557

theorem gcd_111_148 : Nat.gcd 111 148 = 37 :=
by
  sorry

end gcd_111_148_l385_38557


namespace misha_second_attempt_points_l385_38534

/--
Misha made a homemade dartboard at his summer cottage. The round board is 
divided into several sectors by circles, and you can throw darts at it. 
Points are awarded based on the sector hit.

Misha threw 8 darts three times. In his second attempt, he scored twice 
as many points as in his first attempt, and in his third attempt, he scored 
1.5 times more points than in his second attempt. How many points did he 
score in his second attempt?
-/
theorem misha_second_attempt_points:
  ∀ (x : ℕ), 
  (x ≥ 24) →
  (2 * x ≥ 48) →
  (3 * x = 72) →
  (2 * x = 48) :=
by
  intros x h1 h2 h3
  sorry

end misha_second_attempt_points_l385_38534


namespace symmetry_about_origin_l385_38580

def f (x : ℝ) : ℝ := x^3 - x

theorem symmetry_about_origin : 
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end symmetry_about_origin_l385_38580


namespace min_moves_seven_chests_l385_38538

/-
Problem:
Seven chests are placed in a circle, each containing a certain number of coins: [20, 15, 5, 6, 10, 17, 18].
Prove that the minimum number of moves required to equalize the number of coins in all chests is 22.
-/

def min_moves_to_equalize_coins (coins: List ℕ) : ℕ :=
  -- Function that would calculate the minimum number of moves
  sorry

theorem min_moves_seven_chests :
  min_moves_to_equalize_coins [20, 15, 5, 6, 10, 17, 18] = 22 :=
sorry

end min_moves_seven_chests_l385_38538


namespace highway_length_is_105_l385_38511

-- Define the speeds of the two cars
def speed_car1 : ℝ := 15
def speed_car2 : ℝ := 20

-- Define the time they travel for
def time_travelled : ℝ := 3

-- Define the distances covered by the cars
def distance_car1 : ℝ := speed_car1 * time_travelled
def distance_car2 : ℝ := speed_car2 * time_travelled

-- Define the total length of the highway
def length_highway : ℝ := distance_car1 + distance_car2

-- The theorem statement
theorem highway_length_is_105 : length_highway = 105 :=
by
  -- Skipping the proof for now
  sorry

end highway_length_is_105_l385_38511


namespace police_emergency_number_has_prime_divisor_gt_7_l385_38517

theorem police_emergency_number_has_prime_divisor_gt_7 (n : ℕ) (h : n % 1000 = 133) : ∃ p : ℕ, Nat.Prime p ∧ p > 7 ∧ p ∣ n := sorry

end police_emergency_number_has_prime_divisor_gt_7_l385_38517


namespace range_of_m_in_inverse_proportion_function_l385_38512

theorem range_of_m_in_inverse_proportion_function (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (1 - m) / x > 0) ∧ (x < 0 → (1 - m) / x < 0))) ↔ m < 1 :=
by
  sorry

end range_of_m_in_inverse_proportion_function_l385_38512


namespace sufficient_but_not_necessary_l385_38528

-- Define what it means for a line to be perpendicular to a plane
def line_perpendicular_to_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- Define what it means for a line to be perpendicular to countless lines in a plane
def line_perpendicular_to_countless_lines_in_plane (l : Type) (alpha : Type) : Prop := 
  sorry -- Definition not provided

-- The formal statement
theorem sufficient_but_not_necessary (l : Type) (alpha : Type) :
  (line_perpendicular_to_plane l alpha) → (line_perpendicular_to_countless_lines_in_plane l alpha) ∧ 
  ¬ ((line_perpendicular_to_countless_lines_in_plane l alpha) → (line_perpendicular_to_plane l alpha)) :=
by sorry

end sufficient_but_not_necessary_l385_38528


namespace airport_distance_l385_38561

theorem airport_distance (d t : ℝ) (h1 : d = 45 * (t + 0.75))
                         (h2 : d - 45 = 65 * (t - 1.25)) :
  d = 241.875 :=
by
  sorry

end airport_distance_l385_38561


namespace problem1_l385_38566

theorem problem1 :
  (-1 : ℤ)^2024 - (-1 : ℤ)^2023 = 2 := by
  sorry

end problem1_l385_38566


namespace expand_and_simplify_expression_l385_38595

theorem expand_and_simplify_expression : 
  ∀ (x : ℝ), (3 * x - 4) * (2 * x + 6) = 6 * x^2 + 10 * x - 24 := 
by 
  intro x
  sorry

end expand_and_simplify_expression_l385_38595


namespace solution_to_problem_l385_38565

theorem solution_to_problem (a x y n m : ℕ) (h1 : a * (x^n - x^m) = (a * x^m - 4) * y^2)
  (h2 : m % 2 = n % 2) (h3 : (a * x) % 2 = 1) : 
  x = 1 :=
sorry

end solution_to_problem_l385_38565


namespace distribute_paper_clips_l385_38559

theorem distribute_paper_clips (total_clips : ℕ) (boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : total_clips = 81) (h2 : boxes = 9) :
  total_clips / boxes = clips_per_box ↔ clips_per_box = 9 :=
by
  sorry

end distribute_paper_clips_l385_38559


namespace min_jumps_required_to_visit_all_points_and_return_l385_38529

theorem min_jumps_required_to_visit_all_points_and_return :
  ∀ (n : ℕ), n = 2016 →
  ∀ jumps : ℕ → ℕ, (∀ i, jumps i = 2 ∨ jumps i = 3) →
  (∀ i, (jumps (i + 1) + jumps (i + 2)) % n = 0) →
  ∃ (min_jumps : ℕ), min_jumps = 2017 :=
by
  sorry

end min_jumps_required_to_visit_all_points_and_return_l385_38529


namespace quadratic_bounds_l385_38519

variable (a b c: ℝ)

-- Conditions
def quadratic_function (x: ℝ) : ℝ := a * x^2 + b * x + c

def within_range_neg_1_to_1 (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) : Prop :=
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7

-- Main statement
theorem quadratic_bounds
  (h : ∀ x: ℝ, -1 ≤ x ∧ x ≤ 1 → -1 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 1) :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -7 ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 7 := sorry

end quadratic_bounds_l385_38519


namespace find_page_number_l385_38504

theorem find_page_number (n p : ℕ) (h1 : (n * (n + 1)) / 2 + 2 * p = 2046) : p = 15 :=
sorry

end find_page_number_l385_38504


namespace amy_hours_per_week_l385_38537

theorem amy_hours_per_week (hours_summer_per_week : ℕ) (weeks_summer : ℕ) (earnings_summer : ℕ)
  (weeks_school_year : ℕ) (earnings_school_year_goal : ℕ) :
  (hours_summer_per_week = 40) →
  (weeks_summer = 12) →
  (earnings_summer = 4800) →
  (weeks_school_year = 36) →
  (earnings_school_year_goal = 7200) →
  (∃ hours_school_year_per_week : ℕ, hours_school_year_per_week = 20) :=
by
  sorry

end amy_hours_per_week_l385_38537


namespace investment_rate_l385_38577

theorem investment_rate (total_investment income1_rate income2_rate income_total remaining_investment expected_income : ℝ)
  (h1 : total_investment = 12000)
  (h2 : income1_rate = 0.03)
  (h3 : income2_rate = 0.045)
  (h4 : expected_income = 600)
  (h5 : (5000 * income1_rate + 4000 * income2_rate) = 330)
  (h6 : remaining_investment = total_investment - 5000 - 4000) :
  (remaining_investment * 0.09 = expected_income - (5000 * income1_rate + 4000 * income2_rate)) :=
by
  sorry

end investment_rate_l385_38577


namespace num_children_with_identical_cards_l385_38531

theorem num_children_with_identical_cards (children_mama children_nyanya children_manya total_children mixed_cards : ℕ) 
  (h_mama: children_mama = 20) 
  (h_nyanya: children_nyanya = 30) 
  (h_manya: children_manya = 40) 
  (h_total: total_children = children_mama + children_nyanya) 
  (h_mixed: mixed_cards = children_manya) 
  : total_children - children_manya = 10 :=
by
  -- Sorry to indicate the proof is skipped
  sorry

end num_children_with_identical_cards_l385_38531


namespace combined_money_half_l385_38554

theorem combined_money_half
  (J S : ℚ)
  (h1 : J = S)
  (h2 : J - (3/7 * J + 2/5 * J + 1/4 * J) = 24)
  (h3 : S - (1/2 * S + 1/3 * S) = 36) :
  1.5 * J = 458.18 := 
by
  sorry

end combined_money_half_l385_38554


namespace area_of_triangle_DEF_eq_480_l385_38514

theorem area_of_triangle_DEF_eq_480 (DE EF DF : ℝ) (h1 : DE = 20) (h2 : EF = 48) (h3 : DF = 52) :
  let s := (DE + EF + DF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - EF) * (s - DF))
  area = 480 :=
by
  sorry

end area_of_triangle_DEF_eq_480_l385_38514


namespace filter_replacement_month_l385_38567

theorem filter_replacement_month (n : ℕ) (h : n = 25) : (7 * (n - 1)) % 12 = 0 → "January" = "January" :=
by
  intros
  sorry

end filter_replacement_month_l385_38567


namespace volume_of_rectangular_prism_l385_38515

variables (a b c : ℝ)

theorem volume_of_rectangular_prism 
  (h1 : a * b = 12) 
  (h2 : b * c = 18) 
  (h3 : c * a = 9) 
  (h4 : (1 / a) * (1 / b) * (1 / c) = (1 / 216)) :
  a * b * c = 216 :=
sorry

end volume_of_rectangular_prism_l385_38515


namespace find_w_value_l385_38564

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_w_value
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : sqrt x / sqrt y - sqrt y / sqrt x = 7 / 12)
  (h2 : x - y = 7) :
  x + y = 25 := 
by
  sorry

end find_w_value_l385_38564


namespace derivative_of_f_l385_38569

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((2 * x + 1) / Real.sqrt 2) + (2 * x + 1) / (4 * x^2 + 4 * x + 3)

theorem derivative_of_f (x : ℝ) : deriv f x = 8 / (4 * x^2 + 4 * x + 3)^2 :=
by
  -- Proof will be provided here
  sorry

end derivative_of_f_l385_38569


namespace time_spent_washing_car_l385_38506

theorem time_spent_washing_car (x : ℝ) 
  (h1 : x + (1/4) * x = 100) : x = 80 := 
sorry  

end time_spent_washing_car_l385_38506


namespace cos_pi_over_2_minus_2alpha_l385_38583

theorem cos_pi_over_2_minus_2alpha (α : ℝ) (h : Real.tan α = 2) : Real.cos (Real.pi / 2 - 2 * α) = 4 / 5 := 
by 
  sorry

end cos_pi_over_2_minus_2alpha_l385_38583


namespace non_congruent_rectangles_with_even_dimensions_l385_38568

/-- Given a rectangle with perimeter 120 inches and even integer dimensions,
    prove that there are 15 non-congruent rectangles that meet these criteria. -/
theorem non_congruent_rectangles_with_even_dimensions (h w : ℕ) (h_even : h % 2 = 0) (w_even : w % 2 = 0) (perimeter_condition : 2 * (h + w) = 120) :
  ∃ n : ℕ, n = 15 := sorry

end non_congruent_rectangles_with_even_dimensions_l385_38568


namespace relationship_between_u_and_v_l385_38527

variables {r u v p : ℝ}
variables (AB G : ℝ)

theorem relationship_between_u_and_v (hAB : AB = 2 * r) (hAG_GF : u = (p^2 / (2 * r)) - p) :
    v^2 = u^3 / (2 * r - u) :=
sorry

end relationship_between_u_and_v_l385_38527


namespace neg_p_equiv_l385_38536

def p : Prop := ∃ x₀ : ℝ, x₀^2 + 1 > 3 * x₀

theorem neg_p_equiv :
  ¬ p ↔ ∀ x : ℝ, x^2 + 1 ≤ 3 * x := by
  sorry

end neg_p_equiv_l385_38536


namespace inequality_proved_l385_38572

theorem inequality_proved (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end inequality_proved_l385_38572


namespace product_of_prs_eq_60_l385_38539

theorem product_of_prs_eq_60 (p r s : ℕ) (h1 : 3 ^ p + 3 ^ 5 = 270) (h2 : 2 ^ r + 46 = 94) (h3 : 6 ^ s + 5 ^ 4 = 1560) :
  p * r * s = 60 :=
  sorry

end product_of_prs_eq_60_l385_38539


namespace initial_breads_count_l385_38541

theorem initial_breads_count :
  ∃ (X : ℕ), ((((X / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2 = 3 ∧ X = 127 :=
by sorry

end initial_breads_count_l385_38541


namespace line_circle_chord_shortest_l385_38522

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_circle_chord_shortest (m : ℝ) :
  (∀ x y : ℝ, circle_C x y → line_l x y m → m = -3 / 4) :=
sorry

end line_circle_chord_shortest_l385_38522


namespace matrix_pow_A4_l385_38570

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -1], ![1, 1]]

-- State the theorem
theorem matrix_pow_A4 :
  A^4 = ![![0, -9], ![9, -9]] :=
by
  sorry -- Proof is omitted

end matrix_pow_A4_l385_38570


namespace solve_identity_l385_38510

theorem solve_identity (x : ℝ) (a b p q : ℝ)
  (h : (6 * x + 1) / (6 * x ^ 2 + 19 * x + 15) = a / (x - p) + b / (x - q)) :
  a = -1 ∧ b = 2 ∧ p = -3/4 ∧ q = -5/3 :=
by
  sorry

end solve_identity_l385_38510


namespace sum_of_two_consecutive_squares_l385_38562

variable {k m A : ℕ}

theorem sum_of_two_consecutive_squares :
  (∃ k : ℕ, A^2 = (k+1)^3 - k^3) → (∃ m : ℕ, A = m^2 + (m+1)^2) :=
by sorry

end sum_of_two_consecutive_squares_l385_38562


namespace common_root_equation_l385_38588

theorem common_root_equation {m : ℝ} (x : ℝ) (h1 : m * x - 1000 = 1001) (h2 : 1001 * x = m - 1000 * x) : m = 2001 ∨ m = -2001 :=
by
  -- Skipping the proof details
  sorry

end common_root_equation_l385_38588


namespace no_valid_pairs_l385_38500

theorem no_valid_pairs : ∀ (x y : ℕ), x > 0 → y > 0 → x^2 + y^2 + 1 = x^3 → false := 
by
  intros x y hx hy h
  sorry

end no_valid_pairs_l385_38500


namespace range_of_p_add_q_l385_38578

theorem range_of_p_add_q (p q : ℝ) :
  (∀ x : ℝ, ¬(x^2 + 2 * p * x - (q^2 - 2) = 0)) → 
  (p + q) ∈ Set.Ioo (-2 : ℝ) (2 : ℝ) :=
by
  intro h
  sorry

end range_of_p_add_q_l385_38578


namespace hyperbola_focus_exists_l385_38518

-- Define the basic premises of the problem
def is_hyperbola (x y : ℝ) : Prop :=
  -2 * x^2 + 3 * y^2 - 8 * x - 24 * y + 4 = 0

-- Define a condition for the focusing property of the hyperbola.
def is_focus (x y : ℝ) : Prop :=
  (x = -2) ∧ (y = 4 + (10 * Real.sqrt 3 / 3))

-- The theorem to be proved
theorem hyperbola_focus_exists : ∃ x y : ℝ, is_hyperbola x y ∧ is_focus x y :=
by
  -- Proof to be filled in
  sorry

end hyperbola_focus_exists_l385_38518


namespace count_whole_numbers_in_interval_l385_38579

theorem count_whole_numbers_in_interval : 
  let a := 7 / 4
  let b := 3 * Real.pi
  ∃ n : ℕ, n = 8 ∧ ∀ k : ℕ, (2 ≤ k ∧ k ≤ 9) ↔ (a < k ∧ k < b) :=
by
  sorry

end count_whole_numbers_in_interval_l385_38579


namespace total_savings_l385_38526

-- Definition to specify the denomination of each bill
def bill_value : ℕ := 100

-- Condition: Number of $100 bills Michelle has
def num_bills : ℕ := 8

-- The theorem to prove the total savings amount
theorem total_savings : num_bills * bill_value = 800 :=
by
  sorry

end total_savings_l385_38526


namespace number_of_pies_l385_38597

-- Definitions based on the conditions
def box_weight : ℕ := 120
def weight_for_applesauce : ℕ := box_weight / 2
def weight_per_pie : ℕ := 4
def remaining_weight : ℕ := box_weight - weight_for_applesauce

-- The proof problem statement
theorem number_of_pies : (remaining_weight / weight_per_pie) = 15 :=
by
  sorry

end number_of_pies_l385_38597


namespace space_shuttle_speed_l385_38516

theorem space_shuttle_speed :
  ∀ (speed_kph : ℕ) (minutes_per_hour seconds_per_minute : ℕ),
    speed_kph = 32400 →
    minutes_per_hour = 60 →
    seconds_per_minute = 60 →
    (speed_kph / (minutes_per_hour * seconds_per_minute)) = 9 :=
by
  intros speed_kph minutes_per_hour seconds_per_minute
  intro h_speed
  intro h_minutes
  intro h_seconds
  sorry

end space_shuttle_speed_l385_38516


namespace farmer_payment_per_acre_l385_38585

-- Define the conditions
def monthly_payment : ℝ := 300
def length_ft : ℝ := 360
def width_ft : ℝ := 1210
def sqft_per_acre : ℝ := 43560

-- Define the question and its correct answer
def payment_per_acre_per_month : ℝ := 30

-- Prove that the farmer pays $30 per acre per month
theorem farmer_payment_per_acre :
  (monthly_payment / ((length_ft * width_ft) / sqft_per_acre)) = payment_per_acre_per_month :=
by
  sorry

end farmer_payment_per_acre_l385_38585


namespace math_marks_is_95_l385_38546

-- Define the conditions as Lean assumptions
variables (english_marks math_marks physics_marks chemistry_marks biology_marks : ℝ)
variable (average_marks : ℝ)
variable (num_subjects : ℝ)

-- State the conditions
axiom h1 : english_marks = 96
axiom h2 : physics_marks = 82
axiom h3 : chemistry_marks = 97
axiom h4 : biology_marks = 95
axiom h5 : average_marks = 93
axiom h6 : num_subjects = 5

-- Formalize the problem: Prove that math_marks = 95
theorem math_marks_is_95 : math_marks = 95 :=
by
  sorry

end math_marks_is_95_l385_38546


namespace evaluation_expression_l385_38548

theorem evaluation_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y + 8 * y^x - 2 * x * y = 893 :=
by
  rw [h1, h2]
  -- Here we would perform the arithmetic steps to show the equality
  sorry

end evaluation_expression_l385_38548


namespace height_min_surface_area_l385_38542

def height_of_box (x : ℝ) : ℝ := x + 4

def surface_area (x : ℝ) : ℝ := 2 * x^2 + 4 * x * (x + 4)

theorem height_min_surface_area :
  ∀ x : ℝ, surface_area x ≥ 150 → x ≥ 5 → height_of_box x = 9 :=
by
  intros x h1 h2
  sorry

end height_min_surface_area_l385_38542


namespace velocity_equal_distance_l385_38552

theorem velocity_equal_distance (v t : ℝ) (h : v * t = t) (ht : t ≠ 0) : v = 1 :=
by sorry

end velocity_equal_distance_l385_38552


namespace solution_set_of_x_squared_lt_one_l385_38551

theorem solution_set_of_x_squared_lt_one : {x : ℝ | x^2 < 1} = { x | -1 < x ∧ x < 1 } :=
by
  sorry

end solution_set_of_x_squared_lt_one_l385_38551


namespace wolf_does_not_catch_hare_l385_38582

-- Define the distance the hare needs to cover
def distanceHare := 250 -- meters

-- Define the initial separation between the wolf and the hare
def separation := 30 -- meters

-- Define the speed of the hare
def speedHare := 550 -- meters per minute

-- Define the speed of the wolf
def speedWolf := 600 -- meters per minute

-- Define the time it takes for the hare to reach the refuge
def tHare := (distanceHare : ℚ) / speedHare

-- Define the total distance the wolf needs to cover
def totalDistanceWolf := distanceHare + separation

-- Define the time it takes for the wolf to cover the total distance
def tWolf := (totalDistanceWolf : ℚ) / speedWolf

-- Final proposition to be proven
theorem wolf_does_not_catch_hare : tHare < tWolf :=
by
  sorry

end wolf_does_not_catch_hare_l385_38582


namespace max_loaves_given_l385_38521

variables {a1 d : ℕ}

-- Mathematical statement: The conditions given in the problem
def arith_sequence_correct (a1 d : ℕ) : Prop :=
  (5 * a1 + 10 * d = 60) ∧ (2 * a1 + 7 * d = 3 * a1 + 3 * d)

-- Lean theorem statement
theorem max_loaves_given (a1 d : ℕ) (h : arith_sequence_correct a1 d) : a1 + 4 * d = 16 :=
sorry

end max_loaves_given_l385_38521


namespace find_C_l385_38584

noncomputable def A_annual_income : ℝ := 403200.0000000001
noncomputable def A_monthly_income : ℝ := A_annual_income / 12 -- 33600.00000000001

noncomputable def x : ℝ := A_monthly_income / 5 -- 6720.000000000002

noncomputable def C : ℝ := (2 * x) / 1.12 -- should be 12000.000000000004

theorem find_C : C = 12000.000000000004 := 
by sorry

end find_C_l385_38584


namespace percentage_in_first_subject_l385_38533

theorem percentage_in_first_subject (P : ℝ) (H1 : 80 = 80) (H2 : 75 = 75) (H3 : (P + 80 + 75) / 3 = 75) : P = 70 :=
by
  sorry

end percentage_in_first_subject_l385_38533


namespace spherical_to_cartesian_l385_38547

theorem spherical_to_cartesian 
  (ρ θ φ : ℝ)
  (hρ : ρ = 3) 
  (hθ : θ = 7 * Real.pi / 12) 
  (hφ : φ = Real.pi / 4) :
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) = 
  (3 * Real.sqrt 2 / 2 * Real.cos (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2 * Real.sin (7 * Real.pi / 12), 
   3 * Real.sqrt 2 / 2) :=
by
  sorry

end spherical_to_cartesian_l385_38547


namespace no_base_for_final_digit_one_l385_38532

theorem no_base_for_final_digit_one (b : ℕ) (h : 3 ≤ b ∧ b ≤ 10) : ¬ (842 % b = 1) :=
by
  cases h with 
  | intro hb1 hb2 => sorry

end no_base_for_final_digit_one_l385_38532


namespace Q_joined_after_4_months_l385_38573

namespace Business

-- Definitions
def P_cap := 4000
def Q_cap := 9000
def P_time := 12
def profit_ratio := (2 : ℚ) / 3

-- Statement to prove
theorem Q_joined_after_4_months (x : ℕ) (h : P_cap * P_time / (Q_cap * (12 - x)) = profit_ratio) :
  x = 4 := 
sorry

end Business

end Q_joined_after_4_months_l385_38573


namespace vector_MN_l385_38556

def M : ℝ × ℝ := (-3, 3)
def N : ℝ × ℝ := (-5, -1)
def vector_sub (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2)

theorem vector_MN :
  vector_sub N M = (-2, -4) :=
by
  sorry

end vector_MN_l385_38556


namespace balloons_problem_l385_38549

variable (b_J b_S b_J_f b_g : ℕ)

theorem balloons_problem
  (h1 : b_J = 9)
  (h2 : b_S = 5)
  (h3 : b_J_f = 12)
  (h4 : b_g = (b_J + b_S) - b_J_f)
  : b_g = 2 :=
by {
  sorry
}

end balloons_problem_l385_38549


namespace cubical_box_edge_length_l385_38553

noncomputable def edge_length_of_box_in_meters : ℝ :=
  let number_of_cubes := 999.9999999999998
  let edge_length_cube_cm := 10
  let volume_cube_cm := edge_length_cube_cm^3
  let total_volume_box_cm := volume_cube_cm * number_of_cubes
  let total_volume_box_meters := total_volume_box_cm / (100^3)
  (total_volume_box_meters)^(1/3)

theorem cubical_box_edge_length :
  edge_length_of_box_in_meters = 1 := 
sorry

end cubical_box_edge_length_l385_38553


namespace minimum_handshakes_l385_38525

def binom (n k : ℕ) : ℕ := n.choose k

theorem minimum_handshakes (n_A n_B k_A k_B : ℕ) (h1 : binom (n_A + n_B) 2 + n_A + n_B = 465)
  (h2 : n_A < n_B) (h3 : k_A = n_A) (h4 : k_B = n_B) : k_A = 15 :=
by sorry

end minimum_handshakes_l385_38525


namespace fans_per_bleacher_l385_38596

theorem fans_per_bleacher 
  (total_fans : ℕ) 
  (sets_of_bleachers : ℕ) 
  (h_total : total_fans = 2436) 
  (h_sets : sets_of_bleachers = 3) : 
  total_fans / sets_of_bleachers = 812 := 
by 
  sorry

end fans_per_bleacher_l385_38596


namespace surface_area_of_cube_given_sphere_surface_area_l385_38545

noncomputable def edge_length_of_cube (sphere_surface_area : ℝ) : ℝ :=
  let a_square := 2
  Real.sqrt a_square

def surface_area_of_cube (a : ℝ) : ℝ :=
  6 * a^2

theorem surface_area_of_cube_given_sphere_surface_area (sphere_surface_area : ℝ) :
  sphere_surface_area = 6 * Real.pi → 
  surface_area_of_cube (edge_length_of_cube sphere_surface_area) = 12 :=
by
  sorry

end surface_area_of_cube_given_sphere_surface_area_l385_38545


namespace average_income_of_all_customers_l385_38535

theorem average_income_of_all_customers
  (n m : ℕ) 
  (a b : ℝ) 
  (customers_responded : n = 50) 
  (wealthiest_count : m = 10) 
  (other_customers_count : n - m = 40) 
  (wealthiest_avg_income : a = 55000) 
  (other_avg_income : b = 42500) : 
  (m * a + (n - m) * b) / n = 45000 := 
by
  -- transforming given conditions into useful expressions
  have h1 : m = 10 := by assumption
  have h2 : n = 50 := by assumption
  have h3 : n - m = 40 := by assumption
  have h4 : a = 55000 := by assumption
  have h5 : b = 42500 := by assumption
  sorry

end average_income_of_all_customers_l385_38535


namespace meeting_time_l385_38503

noncomputable def combined_speed : ℕ := 10 -- km/h
noncomputable def distance_to_cover : ℕ := 50 -- km
noncomputable def start_time : ℕ := 6 -- pm (in hours)
noncomputable def speed_a : ℕ := 6 -- km/h
noncomputable def speed_b : ℕ := 4 -- km/h

theorem meeting_time : start_time + (distance_to_cover / combined_speed) = 11 :=
by
  sorry

end meeting_time_l385_38503


namespace chameleons_all_red_l385_38599

theorem chameleons_all_red (Y G R : ℕ) (total : ℕ) (P : Y = 7) (Q : G = 10) (R_cond : R = 17) (total_cond : Y + G + R = total) (total_value : total = 34) :
  ∃ x, x = R ∧ x = total ∧ ∀ z : ℕ, z ≠ 0 → total % 3 = z % 3 → ((R : ℕ) % 3 = z) :=
by
  sorry

end chameleons_all_red_l385_38599


namespace y_value_for_equations_l385_38509

theorem y_value_for_equations (x y : ℝ) (h1 : x^2 + y^2 = 25) (h2 : x^2 + y = 10) :
  y = (1 - Real.sqrt 61) / 2 := by
  sorry

end y_value_for_equations_l385_38509


namespace expand_expression_l385_38508

theorem expand_expression (x : ℝ) : (7 * x + 5) * (3 * x^2) = 21 * x^3 + 15 * x^2 :=
by
  sorry

end expand_expression_l385_38508


namespace sum_of_three_eq_six_l385_38586

theorem sum_of_three_eq_six
  (a b c : ℕ) (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 150) :
  a + b + c = 6 :=
sorry

end sum_of_three_eq_six_l385_38586


namespace determinant_of_non_right_triangle_l385_38505

theorem determinant_of_non_right_triangle (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
(h_sum_ABC : A + B + C = π) :
  Matrix.det ![
    ![2 * Real.sin A, 1, 1],
    ![1, 2 * Real.sin B, 1],
    ![1, 1, 2 * Real.sin C]
  ] = 2 := by
  sorry

end determinant_of_non_right_triangle_l385_38505


namespace Luke_mowing_lawns_l385_38598

theorem Luke_mowing_lawns (L : ℕ) (h1 : 18 + L = 27) : L = 9 :=
by
  sorry

end Luke_mowing_lawns_l385_38598


namespace perpendicular_slope_l385_38594

theorem perpendicular_slope :
  ∀ (x y : ℝ), 5 * x - 2 * y = 10 → y = ((5 : ℝ) / 2) * x - 5 → ∃ (m : ℝ), m = - (2 / 5) := by
  sorry

end perpendicular_slope_l385_38594


namespace min_distinct_sums_max_distinct_sums_l385_38530

theorem min_distinct_sums (n : ℕ) (h : 0 < n) : ∃ a b, (a + (n - 1) * b) = (n * (n + 1)) / 2 := sorry

theorem max_distinct_sums (n : ℕ) (h : 0 < n) : 
  ∃ m, m = 2^n - 1 := sorry

end min_distinct_sums_max_distinct_sums_l385_38530


namespace units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l385_38558

/-- Find the units digit of the largest power of 2 that divides into (2^5)! -/
theorem units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial : ∃ d : ℕ, d = 8 := by
  sorry

end units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l385_38558


namespace solve_log_equation_l385_38524

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solve_log_equation (x : ℝ) (hx : 2 * log_base 5 x - 3 * log_base 5 4 = 1) :
  x = 4 * Real.sqrt 5 ∨ x = -4 * Real.sqrt 5 :=
sorry

end solve_log_equation_l385_38524


namespace intersecting_chords_length_l385_38581

theorem intersecting_chords_length
  (h1 : ∃ c1 c2 : ℝ, c1 = 8 ∧ c2 = x + 4 * x ∧ x = 2)
  (h2 : ∀ (a b c d : ℝ), a * b = c * d → a = 4 ∧ b = 4 ∧ c = x ∧ d = 4 * x ∧ x = 2):
  (10 : ℝ) = (x + 4 * x) := by
  sorry

end intersecting_chords_length_l385_38581


namespace second_derivative_at_x₀_l385_38520

noncomputable def f (x : ℝ) : ℝ := sorry
variables (x₀ a b : ℝ)

-- Condition: f(x₀ + Δx) - f(x₀) = a * Δx + b * (Δx)^2
axiom condition : ∀ Δx, f (x₀ + Δx) - f x₀ = a * Δx + b * (Δx)^2

theorem second_derivative_at_x₀ : deriv (deriv f) x₀ = 2 * b :=
sorry

end second_derivative_at_x₀_l385_38520


namespace rational_solves_abs_eq_l385_38501

theorem rational_solves_abs_eq (x : ℚ) : |6 + x| = |6| + |x| → 0 ≤ x := 
sorry

end rational_solves_abs_eq_l385_38501


namespace hollow_iron_ball_diameter_l385_38540

theorem hollow_iron_ball_diameter (R r : ℝ) (s : ℝ) (thickness : ℝ) 
  (h1 : thickness = 1) (h2 : s = 7.5) 
  (h3 : R - r = thickness) 
  (h4 : 4 / 3 * π * R^3 = 4 / 3 * π * s * (R^3 - r^3)) : 
  2 * R = 44.44 := 
sorry

end hollow_iron_ball_diameter_l385_38540


namespace arc_length_correct_l385_38587

noncomputable def arcLengthOfCurve : ℝ :=
  ∫ φ in (0 : ℝ)..(5 * Real.pi / 12), (2 : ℝ) * (Real.sqrt (φ ^ 2 + 1))

theorem arc_length_correct :
  arcLengthOfCurve = (65 / 144) + Real.log (3 / 2) := by
  sorry

end arc_length_correct_l385_38587


namespace average_class_weight_l385_38555

theorem average_class_weight
  (n_boys n_girls n_total : ℕ)
  (avg_weight_boys avg_weight_girls total_students : ℕ)
  (h1 : n_boys = 15)
  (h2 : n_girls = 10)
  (h3 : n_total = 25)
  (h4 : avg_weight_boys = 48)
  (h5 : avg_weight_girls = 405 / 10) 
  (h6 : total_students = 25) :
  (48 * 15 + 40.5 * 10) / 25 = 45 := 
sorry

end average_class_weight_l385_38555


namespace parents_gave_money_l385_38563

def money_before_birthday : ℕ := 159
def money_from_grandmother : ℕ := 25
def money_from_aunt_uncle : ℕ := 20
def total_money_after_birthday : ℕ := 279

theorem parents_gave_money :
  total_money_after_birthday = money_before_birthday + money_from_grandmother + money_from_aunt_uncle + 75 :=
by
  sorry

end parents_gave_money_l385_38563


namespace value_of_q_l385_38591

theorem value_of_q (m p q a b : ℝ) 
  (h₁ : a * b = 6) 
  (h₂ : (a + 1 / b) * (b + 1 / a) = q): 
  q = 49 / 6 := 
sorry

end value_of_q_l385_38591


namespace find_nine_day_segment_l385_38544

/-- 
  Definitions:
  - ws_day: The Winter Solstice day, December 21, 2012.
  - j1_day: New Year's Day, January 1, 2013.
  - Calculate the total days difference between ws_day and j1_day.
  - Check that the distribution of days into 9-day segments leads to January 1, 2013, being the third day of the second segment.
-/
def ws_day : ℕ := 21
def j1_day : ℕ := 1
def days_in_december : ℕ := 31
def days_ws_to_end_dec : ℕ := days_in_december - ws_day + 1
def total_days : ℕ := days_ws_to_end_dec + j1_day

theorem find_nine_day_segment : (total_days % 9) = 3 ∧ (total_days / 9) = 1 := by
  sorry  -- Proof skipped

end find_nine_day_segment_l385_38544


namespace tangent_line_eqn_l385_38576

theorem tangent_line_eqn 
  (x y : ℝ)
  (H_curve : y = x^3 + 3 * x^2 - 5)
  (H_point : (x, y) = (-1, -3)) :
  (3 * x + y + 6 = 0) := 
sorry

end tangent_line_eqn_l385_38576


namespace total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l385_38592

-- Given conditions
def kids_A := 7
def kids_B := 9
def kids_C := 5

def pencils_per_child_A := 4
def erasers_per_child_A := 2
def skittles_per_child_A := 13

def pencils_per_child_B := 6
def rulers_per_child_B := 1
def skittles_per_child_B := 8

def pencils_per_child_C := 3
def sharpeners_per_child_C := 1
def skittles_per_child_C := 15

-- Calculated totals
def total_pencils := kids_A * pencils_per_child_A + kids_B * pencils_per_child_B + kids_C * pencils_per_child_C
def total_erasers := kids_A * erasers_per_child_A
def total_rulers := kids_B * rulers_per_child_B
def total_sharpeners := kids_C * sharpeners_per_child_C
def total_skittles := kids_A * skittles_per_child_A + kids_B * skittles_per_child_B + kids_C * skittles_per_child_C

-- Proof obligations
theorem total_pencils_correct : total_pencils = 97 := by
  sorry

theorem total_erasers_correct : total_erasers = 14 := by
  sorry

theorem total_rulers_correct : total_rulers = 9 := by
  sorry

theorem total_sharpeners_correct : total_sharpeners = 5 := by
  sorry

theorem total_skittles_correct : total_skittles = 238 := by
  sorry

end total_pencils_correct_total_erasers_correct_total_rulers_correct_total_sharpeners_correct_total_skittles_correct_l385_38592


namespace find_m_plus_n_l385_38590

def num_fir_trees : ℕ := 4
def num_pine_trees : ℕ := 5
def num_acacia_trees : ℕ := 6

def num_non_acacia_trees : ℕ := num_fir_trees + num_pine_trees
def total_trees : ℕ := num_fir_trees + num_pine_trees + num_acacia_trees

def prob_no_two_acacia_adj : ℚ :=
  (Nat.choose (num_non_acacia_trees + 1) num_acacia_trees * Nat.choose num_non_acacia_trees num_fir_trees : ℚ) /
  Nat.choose total_trees num_acacia_trees

theorem find_m_plus_n : (prob_no_two_acacia_adj = 84/159) -> (84 + 159 = 243) :=
by {
  admit
}

end find_m_plus_n_l385_38590


namespace goldbach_conjecture_2024_l385_38550

-- Definitions for the problem
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Lean 4 statement for the proof problem
theorem goldbach_conjecture_2024 :
  is_even 2024 ∧ 2024 > 2 → ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ 2024 = p1 + p2 :=
by
  sorry

end goldbach_conjecture_2024_l385_38550


namespace peanuts_added_correct_l385_38593

-- Define the initial and final number of peanuts
def initial_peanuts : ℕ := 4
def final_peanuts : ℕ := 12

-- Define the number of peanuts Mary added
def peanuts_added : ℕ := final_peanuts - initial_peanuts

-- State the theorem that proves the number of peanuts Mary added
theorem peanuts_added_correct : peanuts_added = 8 :=
by
  -- Add the proof here
  sorry

end peanuts_added_correct_l385_38593


namespace total_tickets_l385_38523

theorem total_tickets (n_friends : ℕ) (tickets_per_friend : ℕ) (h1 : n_friends = 6) (h2 : tickets_per_friend = 39) : n_friends * tickets_per_friend = 234 :=
by
  -- Place for proof, to be constructed
  sorry

end total_tickets_l385_38523


namespace sequence_expression_l385_38575

-- Given conditions
variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)
variable (h1 : ∀ n, S n = (1/4) * (a n + 1)^2)

-- Theorem statement
theorem sequence_expression (n : ℕ) : a n = 2 * n - 1 :=
sorry

end sequence_expression_l385_38575


namespace num_of_chords_l385_38574

theorem num_of_chords (n : ℕ) (h : n = 8) : (n.choose 2) = 28 :=
by
  -- Proof of this theorem will be here
  sorry

end num_of_chords_l385_38574


namespace problem1_problem2_problem3_problem4_l385_38507

section

variables (x y : Real)

-- Given conditions
def x_def : x = 3 + 2 * Real.sqrt 2 := sorry
def y_def : y = 3 - 2 * Real.sqrt 2 := sorry

-- Problem 1: Prove x + y = 6
theorem problem1 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x + y = 6 := 
by sorry

-- Problem 2: Prove x - y = 4 * sqrt 2
theorem problem2 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x - y = 4 * Real.sqrt 2 :=
by sorry

-- Problem 3: Prove xy = 1
theorem problem3 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x * y = 1 := 
by sorry

-- Problem 4: Prove x^2 - 3xy + y^2 - x - y = 25
theorem problem4 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x^2 - 3 * x * y + y^2 - x - y = 25 :=
by sorry

end

end problem1_problem2_problem3_problem4_l385_38507


namespace closest_fraction_to_team_aus_medals_l385_38560

theorem closest_fraction_to_team_aus_medals 
  (won_medals : ℕ) (total_medals : ℕ) 
  (choices : List ℚ)
  (fraction_won : ℚ)
  (c1 : won_medals = 28)
  (c2 : total_medals = 150)
  (c3 : choices = [1/4, 1/5, 1/6, 1/7, 1/8])
  (c4 : fraction_won = 28 / 150) :
  abs (fraction_won - 1/5) < abs (fraction_won - 1/4) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/6) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/7) ∧
  abs (fraction_won - 1/5) < abs (fraction_won - 1/8) := 
sorry

end closest_fraction_to_team_aus_medals_l385_38560


namespace M_plus_2N_equals_330_l385_38543

theorem M_plus_2N_equals_330 (M N : ℕ) :
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + 2 * N = 330 := by
  sorry

end M_plus_2N_equals_330_l385_38543
