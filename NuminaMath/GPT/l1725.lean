import Mathlib

namespace sum_mod_12_l1725_172513

def remainder_sum_mod :=
  let nums := [10331, 10333, 10335, 10337, 10339, 10341, 10343]
  let sum_nums := nums.sum
  sum_nums % 12 = 7

theorem sum_mod_12 : remainder_sum_mod :=
by
  sorry

end sum_mod_12_l1725_172513


namespace time_spent_per_piece_l1725_172598

-- Conditions
def number_of_chairs : ℕ := 7
def number_of_tables : ℕ := 3
def total_furniture : ℕ := number_of_chairs + number_of_tables
def total_time_spent : ℕ := 40

-- Proof statement
theorem time_spent_per_piece : total_time_spent / total_furniture = 4 :=
by
  -- Proof goes here
  sorry

end time_spent_per_piece_l1725_172598


namespace gcd_polynomial_eval_l1725_172518

theorem gcd_polynomial_eval (b : ℤ) (h : ∃ (k : ℤ), b = 570 * k) :
  Int.gcd (4 * b ^ 3 + b ^ 2 + 5 * b + 95) b = 95 := by
  sorry

end gcd_polynomial_eval_l1725_172518


namespace solve_for_x_l1725_172586

theorem solve_for_x (x : ℝ) (h : (3 * x - 17) / 4 = (x + 12) / 5) : x = 12.09 :=
by
  sorry

end solve_for_x_l1725_172586


namespace total_distance_total_distance_alt_l1725_172555

variable (D : ℝ) -- declare the variable for the total distance

-- defining the conditions
def speed_walking : ℝ := 4 -- speed in km/hr when walking
def speed_running : ℝ := 8 -- speed in km/hr when running
def total_time : ℝ := 3.75 -- total time in hours

-- proving that D = 10 given the conditions
theorem total_distance 
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time) : 
    D = 10 := 
sorry

-- Alternative theorem version declaring variables directly
theorem total_distance_alt
    (speed_walking speed_running total_time : ℝ) -- declaring variables
    (D : ℝ) -- the total distance
    (h1 : D / (2 * speed_walking) + D / (2 * speed_running) = total_time)
    (hw : speed_walking = 4)
    (hr : speed_running = 8)
    (ht : total_time = 3.75) : 
    D = 10 := 
sorry

end total_distance_total_distance_alt_l1725_172555


namespace staircase_steps_l1725_172567

theorem staircase_steps (x : ℕ) (h1 : x + 2 * x + (2 * x - 10) = 2 * 45) : x = 20 :=
by 
  -- The proof is skipped
  sorry

end staircase_steps_l1725_172567


namespace find_x_coord_of_N_l1725_172540

theorem find_x_coord_of_N
  (M N : ℝ × ℝ)
  (hM : M = (3, -5))
  (hN : N = (x, 2))
  (parallel : M.1 = N.1) :
  x = 3 :=
sorry

end find_x_coord_of_N_l1725_172540


namespace age_difference_l1725_172572

theorem age_difference (d : ℕ) (h1 : 18 + (18 - d) + (18 - 2 * d) + (18 - 3 * d) = 48) : d = 4 :=
sorry

end age_difference_l1725_172572


namespace math_proof_problem_l1725_172547

theorem math_proof_problem : 
  (325 - Real.sqrt 125) / 425 = 65 - 5 := 
by sorry

end math_proof_problem_l1725_172547


namespace clea_ride_down_time_l1725_172533

theorem clea_ride_down_time (c s d : ℝ) (h1 : d = 70 * c) (h2 : d = 28 * (c + s)) :
  (d / s) = 47 := by
  sorry

end clea_ride_down_time_l1725_172533


namespace slope_reciprocal_and_a_bounds_l1725_172528

theorem slope_reciprocal_and_a_bounds (x : ℝ) (f g : ℝ → ℝ) 
    (h1 : ∀ x, f x = Real.log x - a * (x - 1)) 
    (h2 : ∀ x, g x = Real.exp x) :
    ((∀ k₁ k₂, (∃ x₁, k₁ = deriv f x₁) ∧ (∃ x₂, k₂ = deriv g x₂) ∧ k₁ * k₂ = 1) 
    ↔ (Real.exp 1 - 1) / Real.exp 1 < a ∧ a < (Real.exp 2 - 1) / Real.exp 1 ∨ a = 0) :=
by
  sorry

end slope_reciprocal_and_a_bounds_l1725_172528


namespace least_positive_integer_l1725_172559

theorem least_positive_integer (n : ℕ) : 
  (530 + n) % 4 = 0 → n = 2 :=
by {
  sorry
}

end least_positive_integer_l1725_172559


namespace tile_covering_possible_l1725_172594

theorem tile_covering_possible (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  ((m % 6 = 0) ∨ (n % 6 = 0)) := 
sorry

end tile_covering_possible_l1725_172594


namespace roots_of_quadratic_l1725_172551

theorem roots_of_quadratic (m n : ℝ) (h₁ : m + n = -2) (h₂ : m * n = -2022) (h₃ : ∀ x, x^2 + 2 * x - 2022 = 0 → x = m ∨ x = n) :
  m^2 + 3 * m + n = 2020 :=
sorry

end roots_of_quadratic_l1725_172551


namespace number_of_hens_l1725_172575

-- Conditions as Lean definitions
def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 136

-- Mathematically equivalent proof problem
theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 28 :=
by
  sorry

end number_of_hens_l1725_172575


namespace browser_usage_information_is_false_l1725_172577

def num_people_using_A : ℕ := 316
def num_people_using_B : ℕ := 478
def num_people_using_both_A_and_B : ℕ := 104
def num_people_only_using_one_browser : ℕ := 567

theorem browser_usage_information_is_false :
  num_people_only_using_one_browser ≠ (num_people_using_A - num_people_using_both_A_and_B) + (num_people_using_B - num_people_using_both_A_and_B) :=
by
  sorry

end browser_usage_information_is_false_l1725_172577


namespace find_angle_x_l1725_172561

theorem find_angle_x (x : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ)
  (h₁ : α = 45)
  (h₂ : β = 3 * x)
  (h₃ : γ = x)
  (h₄ : α + β + γ = 180) :
  x = 33.75 :=
sorry

end find_angle_x_l1725_172561


namespace simplify_and_evaluate_expression_l1725_172536

theorem simplify_and_evaluate_expression :
  (1 - 2 / (Real.tan (Real.pi / 3) - 1 + 1)) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - 2 * (Real.tan (Real.pi / 3) - 1) + 1) / 
  ((Real.tan (Real.pi / 3) - 1) ^ 2 - (Real.tan (Real.pi / 3) - 1)) = 
  (3 - Real.sqrt 3) / 3 :=
sorry

end simplify_and_evaluate_expression_l1725_172536


namespace intervals_of_monotonicity_l1725_172516

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos (x + Real.pi / 3)

theorem intervals_of_monotonicity :
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (Real.pi / 12 + k * Real.pi) (7 * Real.pi / 12 + k * Real.pi) → (f x ≤ f (7 * Real.pi / 12 + k * Real.pi)))) ∧
  (∀ k : ℤ, (∀ x, x ∈ Set.Icc (-5 * Real.pi / 12 + k * Real.pi) (Real.pi / 12 + k * Real.pi) → (f x ≥ f (Real.pi / 12 + k * Real.pi)))) ∧
  (f (Real.pi / 2) = -Real.sqrt 3) ∧
  (f (Real.pi / 12) = 1 - Real.sqrt 3 / 2) := sorry

end intervals_of_monotonicity_l1725_172516


namespace validate_operation_l1725_172511

theorem validate_operation (x y m a b : ℕ) :
  (2 * x - x ≠ 2) →
  (2 * m + 3 * m ≠ 5 * m^2) →
  (5 * xy - 4 * xy = xy) →
  (2 * a + 3 * b ≠ 5 * a * b) →
  (5 * xy - 4 * xy = xy) :=
by
  intros hA hB hC hD
  exact hC

end validate_operation_l1725_172511


namespace find_algebraic_expression_l1725_172530

-- Definitions as per the conditions
variable (a b : ℝ)

-- Given condition
def given_condition (σ : ℝ) : Prop := σ * (2 * a * b) = 4 * a^2 * b

-- The statement to prove
theorem find_algebraic_expression (σ : ℝ) (h : given_condition a b σ) : σ = 2 * a := 
sorry

end find_algebraic_expression_l1725_172530


namespace sum_first_n_terms_l1725_172529

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_n_terms
  (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h_a2a4 : a 2 + a 4 = 8)
  (h_common_diff : ∀ n : ℕ, a (n + 1) = a n + 2) :
  ∃ S_n : ℕ → ℝ, ∀ n : ℕ, S_n n = n^2 - n :=
by 
  sorry

end sum_first_n_terms_l1725_172529


namespace max_parts_divided_by_three_planes_l1725_172505

theorem max_parts_divided_by_three_planes (parts_0_plane parts_1_plane parts_2_planes parts_3_planes: ℕ)
  (h0 : parts_0_plane = 1)
  (h1 : parts_1_plane = 2)
  (h2 : parts_2_planes = 4)
  (h3 : parts_3_planes = 8) :
  parts_3_planes = 8 :=
by
  sorry

end max_parts_divided_by_three_planes_l1725_172505


namespace probability_P_plus_S_mod_7_correct_l1725_172558

noncomputable def probability_P_plus_S_mod_7 : ℚ :=
  let n := 60
  let total_ways := (n * (n - 1)) / 2
  let num_special_pairs := total_ways - ((52 * 51) / 2)
  num_special_pairs / total_ways

theorem probability_P_plus_S_mod_7_correct :
  probability_P_plus_S_mod_7 = 148 / 590 :=
by
  rw [probability_P_plus_S_mod_7]
  sorry

end probability_P_plus_S_mod_7_correct_l1725_172558


namespace sally_weekly_bread_l1725_172576

-- Define the conditions
def monday_bread : Nat := 3
def tuesday_bread : Nat := 2
def wednesday_bread : Nat := 4
def thursday_bread : Nat := 2
def friday_bread : Nat := 1
def saturday_bread : Nat := 2 * 2  -- 2 sandwiches, 2 pieces each
def sunday_bread : Nat := 2

-- Define the total bread count
def total_bread : Nat := 
  monday_bread + 
  tuesday_bread + 
  wednesday_bread + 
  thursday_bread + 
  friday_bread + 
  saturday_bread + 
  sunday_bread

-- The proof statement
theorem sally_weekly_bread : total_bread = 18 := by
  sorry

end sally_weekly_bread_l1725_172576


namespace price_of_scooter_l1725_172541

-- Assume upfront_payment and percentage_upfront are given
def upfront_payment : ℝ := 240
def percentage_upfront : ℝ := 0.20

noncomputable
def total_price (upfront_payment : ℝ) (percentage_upfront : ℝ) : ℝ :=
  (upfront_payment / percentage_upfront)

theorem price_of_scooter : total_price upfront_payment percentage_upfront = 1200 :=
  by
    sorry

end price_of_scooter_l1725_172541


namespace sets_are_equal_l1725_172504

def setA : Set ℤ := { n | ∃ x y : ℤ, n = x^2 + 2 * y^2 }
def setB : Set ℤ := { n | ∃ x y : ℤ, n = x^2 - 6 * x * y + 11 * y^2 }

theorem sets_are_equal : setA = setB := 
by
  sorry

end sets_are_equal_l1725_172504


namespace garden_perimeter_is_48_l1725_172525

def square_garden_perimeter (pond_area garden_remaining_area : ℕ) : ℕ :=
  let garden_area := pond_area + garden_remaining_area
  let side_length := Int.natAbs (Int.sqrt garden_area)
  4 * side_length

theorem garden_perimeter_is_48 :
  square_garden_perimeter 20 124 = 48 :=
  by
  sorry

end garden_perimeter_is_48_l1725_172525


namespace small_triangle_area_ratio_l1725_172563

theorem small_triangle_area_ratio (a b n : ℝ) 
  (h₀ : a > 0) 
  (h₁ : b > 0) 
  (h₂ : n > 0) 
  (h₃ : ∃ (r s : ℝ), r > 0 ∧ s > 0 ∧ (1/2) * a * r = n * a * b ∧ s = (b^2) / (2 * n * b)) :
  (b^2 / (4 * n)) / (a * b) = 1 / (4 * n) :=
by sorry

end small_triangle_area_ratio_l1725_172563


namespace Katy_jellybeans_l1725_172557

variable (Matt Matilda Steve Katy : ℕ)

def jellybean_relationship (Matt Matilda Steve Katy : ℕ) : Prop :=
  (Matt = 10 * Steve) ∧
  (Matilda = Matt / 2) ∧
  (Steve = 84) ∧
  (Katy = 3 * Matilda) ∧
  (Katy = Matt / 2)

theorem Katy_jellybeans : ∃ Katy, jellybean_relationship Matt Matilda Steve Katy ∧ Katy = 1260 := by
  sorry

end Katy_jellybeans_l1725_172557


namespace total_copies_produced_l1725_172502

theorem total_copies_produced
  (rate_A : ℕ)
  (rate_B : ℕ)
  (rate_C : ℕ)
  (time_A : ℕ)
  (time_B : ℕ)
  (time_C : ℕ)
  (total_time : ℕ)
  (ha : rate_A = 10)
  (hb : rate_B = 10)
  (hc : rate_C = 10)
  (hA_time : time_A = 15)
  (hB_time : time_B = 20)
  (hC_time : time_C = 25)
  (h_total_time : total_time = 30) :
  rate_A * time_A + rate_B * time_B + rate_C * time_C = 600 :=
by 
  -- Machine A: 10 copies per minute * 15 minutes = 150 copies
  -- Machine B: 10 copies per minute * 20 minutes = 200 copies
  -- Machine C: 10 copies per minute * 25 minutes = 250 copies
  -- Hence, the total number of copies = 150 + 200 + 250 = 600
  sorry

end total_copies_produced_l1725_172502


namespace investment_A_l1725_172548

-- Define constants B and C's investment values, C's share, and total profit.
def B_investment : ℕ := 8000
def C_investment : ℕ := 9000
def C_share : ℕ := 36000
def total_profit : ℕ := 88000

-- Problem statement to prove
theorem investment_A (A_investment : ℕ) : 
  (A_investment + B_investment + C_investment = 17000) → 
  (C_investment * total_profit = C_share * (A_investment + B_investment + C_investment)) →
  A_investment = 5000 :=
by 
  intros h1 h2
  sorry

end investment_A_l1725_172548


namespace garden_area_increase_l1725_172531

theorem garden_area_increase : 
  let length_old := 60
  let width_old := 20
  let perimeter := 2 * (length_old + width_old)
  let side_new := perimeter / 4
  let area_old := length_old * width_old
  let area_new := side_new * side_new
  area_new - area_old = 400 :=
by
  sorry

end garden_area_increase_l1725_172531


namespace smaller_square_area_percentage_l1725_172543

noncomputable def area_percentage_of_smaller_square :=
  let side_length_large_square : ℝ := 4
  let area_large_square := side_length_large_square ^ 2
  let side_length_smaller_square := side_length_large_square / 5
  let area_smaller_square := side_length_smaller_square ^ 2
  (area_smaller_square / area_large_square) * 100
theorem smaller_square_area_percentage :
  area_percentage_of_smaller_square = 4 := 
sorry

end smaller_square_area_percentage_l1725_172543


namespace attendance_rate_comparison_l1725_172584

theorem attendance_rate_comparison (attendees_A total_A attendees_B total_B : ℕ) 
  (hA : (attendees_A / total_A: ℚ) > (attendees_B / total_B: ℚ)) : 
  (attendees_A > attendees_B) → false :=
by
  sorry

end attendance_rate_comparison_l1725_172584


namespace reading_time_per_disc_l1725_172509

theorem reading_time_per_disc (total_minutes : ℕ) (disc_capacity : ℕ) (d : ℕ) (reading_per_disc : ℕ) :
  total_minutes = 528 ∧ disc_capacity = 45 ∧ d = 12 ∧ total_minutes = d * reading_per_disc → reading_per_disc = 44 :=
by
  sorry

end reading_time_per_disc_l1725_172509


namespace olivia_quarters_left_l1725_172521

-- Define the initial condition and action condition as parameters
def initial_quarters : ℕ := 11
def quarters_spent : ℕ := 4
def quarters_left : ℕ := initial_quarters - quarters_spent

-- The theorem to state the result
theorem olivia_quarters_left : quarters_left = 7 := by
  sorry

end olivia_quarters_left_l1725_172521


namespace max_value_T_n_l1725_172510

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) := a₁ * q^n

noncomputable def sum_of_first_n_terms (a₁ q : ℝ) (n : ℕ) :=
  a₁ * (1 - q^(n + 1)) / (1 - q)

noncomputable def T_n (a₁ q : ℝ) (n : ℕ) :=
  (9 * sum_of_first_n_terms a₁ q n - sum_of_first_n_terms a₁ q (2 * n)) /
  geometric_sequence a₁ q (n + 1)

theorem max_value_T_n
  (a₁ : ℝ) (n : ℕ) (h : n > 0) (q : ℝ) (hq : q = 2) :
  ∃ n₀ : ℕ, T_n a₁ q n₀ = 3 := sorry

end max_value_T_n_l1725_172510


namespace work_completion_time_l1725_172570

theorem work_completion_time (work_per_day_A : ℚ) (work_per_day_B : ℚ) (work_per_day_C : ℚ) 
(days_A_worked: ℚ) (days_C_worked: ℚ) :
work_per_day_A = 1 / 20 ∧ work_per_day_B = 1 / 30 ∧ work_per_day_C = 1 / 10 ∧
days_A_worked = 2 ∧ days_C_worked = 4  → 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked) +
(1 - 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked)))
/ work_per_day_B + days_C_worked) 
= 15 := by
sorry

end work_completion_time_l1725_172570


namespace card_area_after_shortening_l1725_172512

theorem card_area_after_shortening 
  (length : ℕ) (width : ℕ) (area_after_shortening : ℕ) 
  (h_initial : length = 8) (h_initial_width : width = 3)
  (h_area_shortened_by_2 : area_after_shortening = 15) :
  (length - 2) * width = 8 :=
by
  -- Original dimensions
  let original_length := 8
  let original_width := 3
  -- Area after shortening one side by 2 inches
  let area_after_shortening_width := (original_length) * (original_width - 2)
  let area_after_shortening_length := (original_length - 2) * (original_width)
  sorry

end card_area_after_shortening_l1725_172512


namespace recipe_calls_for_nine_cups_of_flour_l1725_172514

def cups_of_flour (x : ℕ) := 
  ∃ cups_added_sugar : ℕ, 
    cups_added_sugar = (6 - 4) ∧ 
    x = cups_added_sugar + 7

theorem recipe_calls_for_nine_cups_of_flour : cups_of_flour 9 :=
by
  sorry

end recipe_calls_for_nine_cups_of_flour_l1725_172514


namespace age_difference_ratio_l1725_172597

theorem age_difference_ratio (h : ℕ) (f : ℕ) (m : ℕ) 
  (harry_age : h = 50) 
  (father_age : f = h + 24) 
  (mother_age : m = 22 + h) :
  (f - m) / h = 1 / 25 := 
by 
  sorry

end age_difference_ratio_l1725_172597


namespace triangle_angle_extension_l1725_172522

theorem triangle_angle_extension :
  ∀ (BAC ABC BCA CDB DBC : ℝ),
  180 = BAC + ABC + BCA →
  CDB = BAC + ABC →
  DBC = BAC + BCA →
  (CDB + DBC) / (BAC + ABC) = 2 :=
by
  intros BAC ABC BCA CDB DBC h1 h2 h3
  sorry

end triangle_angle_extension_l1725_172522


namespace domain_correct_l1725_172552

noncomputable def domain_function (x : ℝ) : Prop :=
  (4 * x - 3 > 0) ∧ (Real.log (4 * x - 3) / Real.log 0.5 > 0)

theorem domain_correct : {x : ℝ | domain_function x} = {x : ℝ | (3 / 4 : ℝ) < x ∧ x < 1} :=
by
  sorry

end domain_correct_l1725_172552


namespace first_reduction_percentage_l1725_172544

theorem first_reduction_percentage (P : ℝ) (x : ℝ) (h : 0.30 * (1 - x / 100) * P = 0.225 * P) : x = 25 :=
by
  sorry

end first_reduction_percentage_l1725_172544


namespace total_votes_l1725_172554

theorem total_votes (V : ℕ) (h1 : ∃ c : ℕ, c = 84) (h2 : ∃ m : ℕ, m = 476) (h3 : ∃ d : ℕ, d = ((84 * V - 16 * V) / 100)) : 
  V = 700 := 
by 
  sorry 

end total_votes_l1725_172554


namespace elective_course_schemes_l1725_172537

theorem elective_course_schemes : Nat.choose 4 2 = 6 := by
  sorry

end elective_course_schemes_l1725_172537


namespace symmetric_point_l1725_172565

theorem symmetric_point (a b : ℝ) (h1 : a = 2) (h2 : 3 = -b) : (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_point_l1725_172565


namespace cos_225_eq_l1725_172550

noncomputable def angle_in_radians (deg : ℝ) : ℝ := deg * (Real.pi / 180)

theorem cos_225_eq :
  Real.cos (angle_in_radians 225) = - (Real.sqrt 2) / 2 := 
  sorry

end cos_225_eq_l1725_172550


namespace value_to_subtract_l1725_172507

theorem value_to_subtract (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 34) / 10 = 2) : x = 5 :=
by 
  sorry

end value_to_subtract_l1725_172507


namespace james_fish_tanks_l1725_172503

theorem james_fish_tanks (n t1 t2 t3 : ℕ) (h1 : t1 = 20) (h2 : t2 = 2 * t1) (h3 : t3 = 2 * t1) (h4 : t1 + t2 + t3 = 100) : n = 3 :=
sorry

end james_fish_tanks_l1725_172503


namespace projected_increase_l1725_172560

theorem projected_increase (R : ℝ) (P : ℝ) 
  (h1 : ∃ P, ∀ (R : ℝ), 0.9 * R = 0.75 * (R + (P / 100) * R)) 
  (h2 : ∀ (R : ℝ), R > 0) :
  P = 20 :=
by
  sorry

end projected_increase_l1725_172560


namespace radioactive_decay_minimum_years_l1725_172515

noncomputable def min_years (a : ℝ) (n : ℕ) : Prop :=
  (a * (1 - 3 / 4) ^ n ≤ a * 1 / 100)

theorem radioactive_decay_minimum_years (a : ℝ) (h : 0 < a) : ∃ n : ℕ, min_years a n ∧ n = 4 :=
by {
  sorry
}

end radioactive_decay_minimum_years_l1725_172515


namespace votes_for_winning_candidate_l1725_172556

-- Define the variables and conditions
variable (V : ℝ) -- Total number of votes
variable (W : ℝ) -- Votes for the winner

-- Condition 1: The winner received 75% of the votes
axiom winner_votes: W = 0.75 * V

-- Condition 2: The winner won by 500 votes
axiom win_by_500: W - 0.25 * V = 500

-- The statement we want to prove
theorem votes_for_winning_candidate : W = 750 :=
by sorry

end votes_for_winning_candidate_l1725_172556


namespace count_even_numbers_l1725_172527

theorem count_even_numbers (a b : ℕ) (h1 : a > 300) (h2 : b ≤ 600) (h3 : ∀ n, 300 < n ∧ n ≤ 600 → n % 2 = 0) : 
  ∃ c : ℕ, c = 150 :=
by
  sorry

end count_even_numbers_l1725_172527


namespace clownfish_ratio_l1725_172582

theorem clownfish_ratio (C B : ℕ) (h₁ : C = B) (h₂ : C + B = 100) (h₃ : C = B) : 
  (let B := 50; 
  let initially_clownfish := B - 26; -- Number of clownfish that initially joined display tank
  let swam_back := (B - 26) - 16; -- Number of clownfish that swam back
  initially_clownfish > 0 → 
  swam_back > 0 → 
  (swam_back : ℚ) / (initially_clownfish : ℚ) = 1 / 3) :=
by 
  sorry

end clownfish_ratio_l1725_172582


namespace estimated_percentage_negative_attitude_l1725_172501

-- Define the conditions
def total_parents := 2500
def sample_size := 400
def negative_attitude := 360

-- Prove the estimated percentage of parents with a negative attitude is 90%
theorem estimated_percentage_negative_attitude : 
  (negative_attitude: ℝ) / (sample_size: ℝ) * 100 = 90 := by
  sorry

end estimated_percentage_negative_attitude_l1725_172501


namespace find_interest_rate_l1725_172566

theorem find_interest_rate
  (P : ℝ) (A : ℝ) (n t : ℕ) (hP : P = 3000) (hA : A = 3307.5) (hn : n = 2) (ht : t = 1) :
  ∃ r : ℝ, r = 10 :=
by
  sorry

end find_interest_rate_l1725_172566


namespace L_shaped_region_area_l1725_172592

-- Define the conditions
def square_area (side_length : ℕ) : ℕ := side_length * side_length

def WXYZ_side_length : ℕ := 6
def XUVW_side_length : ℕ := 2
def TYXZ_side_length : ℕ := 3

-- Define the areas of the squares
def WXYZ_area : ℕ := square_area WXYZ_side_length
def XUVW_area : ℕ := square_area XUVW_side_length
def TYXZ_area : ℕ := square_area TYXZ_side_length

-- Lean statement to prove the area of the L-shaped region
theorem L_shaped_region_area : WXYZ_area - XUVW_area - TYXZ_area = 23 := by
  sorry

end L_shaped_region_area_l1725_172592


namespace find_functional_f_l1725_172506

-- Define the problem domain and functions
variable (f : ℕ → ℕ)
variable (ℕ_star : Set ℕ) -- ℕ_star is {1,2,3,...}

-- Conditions
axiom f_increasing (h1 : ℕ) (h2 : ℕ) (h1_lt_h2 : h1 < h2) : f h1 < f h2
axiom f_functional (x : ℕ) (y : ℕ) : f (y * f x) = x^2 * f (x * y)

-- The proof problem
theorem find_functional_f : (∀ x ∈ ℕ_star, f x = x^2) :=
sorry

end find_functional_f_l1725_172506


namespace average_cookies_per_package_l1725_172564

def cookies_per_package : List ℕ := [9, 11, 14, 12, 0, 18, 15, 16, 19, 21]

theorem average_cookies_per_package :
  (cookies_per_package.sum : ℚ) / cookies_per_package.length = 13.5 := by
  sorry

end average_cookies_per_package_l1725_172564


namespace simplify_expression_l1725_172595

theorem simplify_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 9^(3/2)) = -7 :=
by sorry

end simplify_expression_l1725_172595


namespace matt_assignment_problems_l1725_172520

theorem matt_assignment_problems (P : ℕ) (h : 5 * P - 2 * P = 60) : P = 20 :=
by
  sorry

end matt_assignment_problems_l1725_172520


namespace round_table_legs_l1725_172553

theorem round_table_legs:
  ∀ (chairs tables disposed chairs_legs tables_legs : ℕ) (total_legs : ℕ),
  chairs = 80 →
  chairs_legs = 5 →
  tables = 20 →
  disposed = 40 * chairs / 100 →
  total_legs = 300 →
  total_legs - (chairs - disposed) * chairs_legs = tables * tables_legs →
  tables_legs = 3 :=
by 
  intros chairs tables disposed chairs_legs tables_legs total_legs
  sorry

end round_table_legs_l1725_172553


namespace xiaopangs_score_is_16_l1725_172546

-- Define the father's score
def fathers_score : ℕ := 48

-- Define Xiaopang's score in terms of father's score
def xiaopangs_score (fathers_score : ℕ) : ℕ := fathers_score / 2 - 8

-- The theorem to prove that Xiaopang's score is 16
theorem xiaopangs_score_is_16 : xiaopangs_score fathers_score = 16 := 
by
  sorry

end xiaopangs_score_is_16_l1725_172546


namespace linda_coats_l1725_172571

variable (wall_area : ℝ) (cover_per_gallon : ℝ) (gallons_bought : ℝ)

theorem linda_coats (h1 : wall_area = 600)
                    (h2 : cover_per_gallon = 400)
                    (h3 : gallons_bought = 3) :
  (gallons_bought / (wall_area / cover_per_gallon)) = 2 :=
by
  sorry

end linda_coats_l1725_172571


namespace max_workers_l1725_172562

theorem max_workers (S a n : ℕ) (h1 : n > 0) (h2 : S > 0) (h3 : a > 0)
  (h4 : (S:ℚ) / (a * n) > (3 * S:ℚ) / (a * (n + 5))) :
  2 * n + 5 = 9 := 
by
  sorry

end max_workers_l1725_172562


namespace eunice_pots_l1725_172590

theorem eunice_pots (total_seeds pots_with_3_seeds last_pot_seeds : ℕ)
  (h1 : total_seeds = 10)
  (h2 : pots_with_3_seeds * 3 + last_pot_seeds = total_seeds)
  (h3 : last_pot_seeds = 1) : pots_with_3_seeds + 1 = 4 :=
by
  -- Proof omitted
  sorry

end eunice_pots_l1725_172590


namespace remainder_div_82_l1725_172549

theorem remainder_div_82 (x : ℤ) (h : ∃ k : ℤ, x + 17 = 41 * k + 22) : (x % 82 = 5) :=
by
  sorry

end remainder_div_82_l1725_172549


namespace find_sum_abc_l1725_172581

noncomputable def f (x a b c : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + c

theorem find_sum_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (habc_distinct : a ≠ b) (hfa : f a a b c = a^3) (hfb : f b a b c = b^3) : 
  a + b + c = 18 := 
sorry

end find_sum_abc_l1725_172581


namespace trigonometric_identity_l1725_172534

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = -3 / 2 :=
by
  sorry

end trigonometric_identity_l1725_172534


namespace find_a_l1725_172523

-- Define the variables and conditions
variable (a x y : ℤ)

-- Given conditions
def x_value := (x = 2)
def y_value := (y = 1)
def equation := (a * x - y = 3)

-- The theorem to prove
theorem find_a : x_value x → y_value y → equation a x y → a = 2 :=
by
  intros
  sorry

end find_a_l1725_172523


namespace probability_at_least_one_six_l1725_172578

theorem probability_at_least_one_six (h: ℚ) : h = 91 / 216 :=
by 
  sorry

end probability_at_least_one_six_l1725_172578


namespace sector_area_is_4_l1725_172591

/-- Given a sector of a circle with perimeter 8 and central angle 2 radians,
    the area of the sector is 4. -/
theorem sector_area_is_4 (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l / r = 2) : 
    (1 / 2) * l * r = 4 :=
sorry

end sector_area_is_4_l1725_172591


namespace spherical_to_rectangular_l1725_172526

theorem spherical_to_rectangular
  (ρ θ φ : ℝ)
  (ρ_eq : ρ = 10)
  (θ_eq : θ = 5 * Real.pi / 4)
  (φ_eq : φ = Real.pi / 4) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_l1725_172526


namespace gcd_1021_2729_l1725_172545

theorem gcd_1021_2729 : Int.gcd 1021 2729 = 1 :=
by
  sorry

end gcd_1021_2729_l1725_172545


namespace calculate_probability_two_cards_sum_to_15_l1725_172568

-- Define the probability calculation as per the problem statement
noncomputable def probability_two_cards_sum_to_15 : ℚ :=
  let total_cards := 52
  let number_cards := 36 -- 9 values (2 through 10) each with 4 cards
  let card_combinations := (number_cards * (number_cards - 1)) / 2 -- Total pairs to choose from
  let favourable_combinations := 144 -- Manually calculated from cases in the solution
  favourable_combinations / card_combinations

theorem calculate_probability_two_cards_sum_to_15 :
  probability_two_cards_sum_to_15 = 8 / 221 :=
by
  -- Here we ignore the proof steps and directly state it assuming the provided assumption
  admit

end calculate_probability_two_cards_sum_to_15_l1725_172568


namespace beta_cannot_be_determined_l1725_172508

variables (α β : ℝ)
def consecutive_interior_angles (α β : ℝ) : Prop := -- define what it means for angles to be consecutive interior angles
  α + β = 180  -- this is true for interior angles, for illustrative purposes.

theorem beta_cannot_be_determined
  (h1 : consecutive_interior_angles α β)
  (h2 : α = 55) :
  ¬(∃ β, β = α) :=
by
  sorry

end beta_cannot_be_determined_l1725_172508


namespace max_min_value_sum_l1725_172539

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4*x) * Real.sin (x - 2) + x + 1

theorem max_min_value_sum (M m : ℝ) 
  (hM : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≤ M)
  (hm : ∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≥ m)
  (hM_max : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = M)
  (hm_min : ∃ x ∈ Set.Icc (-1 : ℝ) 5, f x = m)
  : M + m = 6 :=
sorry

end max_min_value_sum_l1725_172539


namespace smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l1725_172542

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * sqrt 3 * cos x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, 2 * cos x)

noncomputable def f (x : ℝ) : ℝ := 
  let a_dot_b := (a x).1 * (b x).1 + (a x).2 * (b x).2
  let b_norm_sq := (b x).1 ^ 2 + (b x).2 ^ 2
  a_dot_b + b_norm_sq + 3 / 2

theorem smallest_positive_period_of_f :
  ∀ x, f (x + π) = f x := sorry

theorem symmetry_center_of_f :
  ∃ k : ℤ, ∀ x, f x = 5 ↔ x = (-π / 12 + k * (π / 2) : ℝ) := sorry

theorem range_of_f_in_interval :
  ∀ x, (π / 6 ≤ x ∧ x ≤ π / 2) → (5 / 2 ≤ f x ∧ f x ≤ 10) := sorry

end smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l1725_172542


namespace polynomial_remainder_l1725_172588

noncomputable def divisionRemainder (f g : Polynomial ℝ) : Polynomial ℝ := Polynomial.modByMonic f g

theorem polynomial_remainder :
  divisionRemainder (Polynomial.X ^ 5 + 2) (Polynomial.X ^ 2 - 4 * Polynomial.X + 7) = -29 * Polynomial.X - 54 :=
by
  sorry

end polynomial_remainder_l1725_172588


namespace competition_arrangements_l1725_172500

noncomputable def count_arrangements (students : Fin 4) (events : Fin 3) : Nat :=
  -- The actual counting function is not implemented
  sorry

theorem competition_arrangements (students : Fin 4) (events : Fin 3) :
  let count := count_arrangements students events
  (∃ (A B C D : Fin 4), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
    B ≠ C ∧ B ≠ D ∧ 
    C ≠ D ∧ 
    (A ≠ 0) ∧ 
    count = 24) := sorry

end competition_arrangements_l1725_172500


namespace probability_of_three_heads_in_eight_tosses_l1725_172593

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l1725_172593


namespace length_ab_square_l1725_172535

theorem length_ab_square (s a : ℝ) (h_square : s = 2 * a) (h_area : 3000 = 1/2 * (s + (s - 2 * a)) * s) : 
  s = 20 * Real.sqrt 15 :=
by
  sorry

end length_ab_square_l1725_172535


namespace second_piece_cost_l1725_172587

theorem second_piece_cost
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (single_piece1 : ℕ)
  (single_piece2 : ℕ)
  (remaining_piece_count : ℕ)
  (remaining_piece_cost : ℕ)
  (total_cost : total_spent = 610)
  (number_of_items : num_pieces = 7)
  (first_item_cost : single_piece1 = 49)
  (remaining_piece_item_cost : remaining_piece_cost = 96)
  (first_item_total_cost : remaining_piece_count = 5)
  (sum_equation : single_piece1 + single_piece2 + (remaining_piece_count * remaining_piece_cost) = total_spent) :
  single_piece2 = 81 := 
  sorry

end second_piece_cost_l1725_172587


namespace equation_of_line_AB_l1725_172532

noncomputable def circle_center : ℝ × ℝ := (1, 0)  -- center of the circle (x-1)^2 + y^2 = 1
noncomputable def circle_radius : ℝ := 1          -- radius of the circle (x-1)^2 + y^2 = 1
noncomputable def point_P : ℝ × ℝ := (3, 1)       -- point P(3,1)

theorem equation_of_line_AB :
  ∃ (AB : ℝ → ℝ → Prop),
    (∀ x y, AB x y ↔ (2 * x + y - 3 = 0)) := sorry

end equation_of_line_AB_l1725_172532


namespace pattern_equation_l1725_172583

theorem pattern_equation (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2))) :=
by
  sorry

end pattern_equation_l1725_172583


namespace tangent_line_parabola_l1725_172524

theorem tangent_line_parabola (k : ℝ) (tangent : ∀ y : ℝ, ∃ x : ℝ, 4 * x + 3 * y + k = 0 ∧ y^2 = 12 * x) : 
  k = 27 / 4 :=
sorry

end tangent_line_parabola_l1725_172524


namespace rectangle_area_l1725_172589

theorem rectangle_area (P : ℕ) (w : ℕ) (h : ℕ) (A : ℕ) 
  (hP : P = 28) 
  (hw : w = 6)
  (hW : P = 2 * (h + w)) 
  (hA : A = h * w) : 
  A = 48 :=
by
  sorry

end rectangle_area_l1725_172589


namespace more_oranges_than_apples_l1725_172596

def apples : ℕ := 14
def oranges : ℕ := 2 * 12

theorem more_oranges_than_apples : oranges - apples = 10 :=
by
  sorry

end more_oranges_than_apples_l1725_172596


namespace which_is_right_triangle_l1725_172569

-- Definitions for each group of numbers
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (4, 5, 6)
def sides_D := (7, 8, 9)

-- Definition of a condition for right triangle using the converse of the Pythagorean theorem
def is_right_triangle (a b c: ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem which_is_right_triangle :
    ¬is_right_triangle 1 2 3 ∧
    ¬is_right_triangle 4 5 6 ∧
    ¬is_right_triangle 7 8 9 ∧
    is_right_triangle 3 4 5 :=
by
  sorry

end which_is_right_triangle_l1725_172569


namespace geo_seq_condition_l1725_172585

-- Definitions based on conditions
variable (a b c : ℝ)

-- Condition of forming a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, -1 * r = a ∧ a * r = b ∧ b * r = c ∧ c * r = -9

-- Proof problem statement
theorem geo_seq_condition (h : geometric_sequence a b c) : b = -3 ∧ a * c = 9 :=
sorry

end geo_seq_condition_l1725_172585


namespace find_positive_integers_l1725_172517

theorem find_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) : x^2 - Nat.factorial y = 2019 ↔ x = 45 ∧ y = 3 :=
by
  sorry

end find_positive_integers_l1725_172517


namespace probability_of_both_l1725_172580

variable (A B : Prop)

-- Assumptions
def p_A : ℝ := 0.55
def p_B : ℝ := 0.60

-- Probability of both A and B telling the truth at the same time
theorem probability_of_both : p_A * p_B = 0.33 := by
  sorry

end probability_of_both_l1725_172580


namespace Milly_spends_135_minutes_studying_l1725_172573

-- Definitions of homework times
def mathHomeworkTime := 60
def geographyHomeworkTime := mathHomeworkTime / 2
def scienceHomeworkTime := (mathHomeworkTime + geographyHomeworkTime) / 2

-- Definition of Milly's total study time
def totalStudyTime := mathHomeworkTime + geographyHomeworkTime + scienceHomeworkTime

-- Theorem stating that Milly spends 135 minutes studying
theorem Milly_spends_135_minutes_studying : totalStudyTime = 135 :=
by
  -- Proof omitted
  sorry

end Milly_spends_135_minutes_studying_l1725_172573


namespace carrie_strawberry_harvest_l1725_172579

/-- Carrie has a rectangular garden that measures 10 feet by 7 feet.
    She plants the entire garden with strawberry plants. Carrie is able to
    plant 5 strawberry plants per square foot, and she harvests an average of
    12 strawberries per plant. How many strawberries can she expect to harvest?
-/
theorem carrie_strawberry_harvest :
  let width := 10
  let length := 7
  let plants_per_sqft := 5
  let strawberries_per_plant := 12
  let area := width * length
  let total_plants := plants_per_sqft * area
  let total_strawberries := strawberries_per_plant * total_plants
  total_strawberries = 4200 :=
by
  sorry

end carrie_strawberry_harvest_l1725_172579


namespace balance_balls_l1725_172599

variable (G B Y W P : ℝ)

-- Given conditions
def cond1 : 4 * G = 9 * B := sorry
def cond2 : 3 * Y = 8 * B := sorry
def cond3 : 7 * B = 5 * W := sorry
def cond4 : 4 * P = 10 * B := sorry

-- Theorem we need to prove
theorem balance_balls : 5 * G + 3 * Y + 3 * W + P = 26 * B :=
by
  -- skipping the proof
  sorry

end balance_balls_l1725_172599


namespace base6_base5_subtraction_in_base10_l1725_172574

def base6_to_nat (n : ℕ) : ℕ :=
  3 * 6^2 + 2 * 6^1 + 5 * 6^0

def base5_to_nat (n : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

theorem base6_base5_subtraction_in_base10 : base6_to_nat 325 - base5_to_nat 231 = 59 := by
  sorry

end base6_base5_subtraction_in_base10_l1725_172574


namespace function_has_local_minimum_at_zero_l1725_172538

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2 * x + 2) / (2 * (x - 1))

def is_local_minimum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y, abs (y - x) < ε → f x ≤ f y

theorem function_has_local_minimum_at_zero :
  -4 < 0 ∧ 0 < 1 ∧ is_local_minimum f 0 := 
sorry

end function_has_local_minimum_at_zero_l1725_172538


namespace equal_saturdays_and_sundays_l1725_172519

theorem equal_saturdays_and_sundays (start_day : ℕ) (h : start_day < 7) :
  ∃! d, (d < 7 ∧ ((d + 2) % 7 = 0 → (d = 5))) :=
by
  sorry

end equal_saturdays_and_sundays_l1725_172519
