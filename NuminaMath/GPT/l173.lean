import Mathlib

namespace find_smallest_x_l173_173156

noncomputable def smallest_pos_real_x : ℝ :=
  55 / 7

theorem find_smallest_x (x : ℝ) (h : x > 0) (hx : ⌊x^2⌋ - x * ⌊x⌋ = 6) : x = smallest_pos_real_x :=
  sorry

end find_smallest_x_l173_173156


namespace LemonadeCalories_l173_173455

noncomputable def total_calories (lemon_juice sugar water honey : ℕ) (cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey : ℕ) : ℝ :=
  (lemon_juice / 100) * cal_per_100g_lemon_juice +
  (sugar / 100) * cal_per_100g_sugar +
  (honey / 100) * cal_per_100g_honey

noncomputable def calories_in_250g (total_calories : ℝ) (total_weight : ℕ) : ℝ :=
  (total_calories / total_weight) * 250

theorem LemonadeCalories :
  let lemon_juice := 150
  let sugar := 200
  let water := 300
  let honey := 50
  let cal_per_100g_lemon_juice := 25
  let cal_per_100g_sugar := 386
  let cal_per_100g_honey := 64
  let total_weight := lemon_juice + sugar + water + honey
  let total_cal := total_calories lemon_juice sugar water honey cal_per_100g_lemon_juice cal_per_100g_sugar cal_per_100g_honey
  calories_in_250g total_cal total_weight = 301 :=
by
  sorry

end LemonadeCalories_l173_173455


namespace moles_of_C2H6_formed_l173_173988

-- Define the initial conditions
def initial_moles_H2 : ℕ := 3
def initial_moles_C2H4 : ℕ := 3
def reaction_ratio_C2H4_H2_C2H6 (C2H4 H2 C2H6 : ℕ) : Prop :=
  C2H4 = H2 ∧ C2H4 = C2H6

-- State the theorem to prove
theorem moles_of_C2H6_formed : reaction_ratio_C2H4_H2_C2H6 initial_moles_C2H4 initial_moles_H2 3 :=
by {
  sorry
}

end moles_of_C2H6_formed_l173_173988


namespace find_x_l173_173589

theorem find_x (x : ℝ) (h : (x^2 - x - 6) / (x + 1) = (x^2 - 2*x - 3) * (0 : ℂ).im) : x = 3 :=
sorry

end find_x_l173_173589


namespace integer_solutions_l173_173254

theorem integer_solutions (m : ℤ) :
  (∃ x : ℤ, (m * x - 1) / (x - 1) = 2 + 1 / (1 - x)) → 
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + 1 / 2 = 0) →
  m = 3 :=
by
  sorry

end integer_solutions_l173_173254


namespace sum_of_roots_l173_173977

theorem sum_of_roots (x : ℝ) (h : x^2 = 10 * x + 16) : x = 10 :=
by 
  -- Rearrange the equation to standard form: x^2 - 10x - 16 = 0
  have eqn : x^2 - 10 * x - 16 = 0 := by sorry
  -- Use the formula for the sum of the roots of a quadratic equation
  -- Prove the sum of the roots is 10
  sorry

end sum_of_roots_l173_173977


namespace road_repair_completion_time_l173_173621

theorem road_repair_completion_time :
  (∀ (r : ℝ), 1 = 45 * r * 3) → (∀ (t : ℝ), (30 * (1 / (3 * 45))) * t = 1) → t = 4.5 :=
by
  intros rate_eq time_eq
  sorry

end road_repair_completion_time_l173_173621


namespace max_figures_in_grid_l173_173437

-- Definition of the grid size
def grid_size : ℕ := 9

-- Definition of the figure coverage
def figure_coverage : ℕ := 4

-- The total number of unit squares in the grid is 9 * 9 = 81
def total_unit_squares : ℕ := grid_size * grid_size

-- Each figure covers exactly 4 unit squares
def units_per_figure : ℕ := figure_coverage

-- The number of such 2x2 blocks that can be formed in 9x9 grid.
def maximal_figures_possible : ℕ := (grid_size / 2) * (grid_size / 2)

-- The main theorem to be proved
theorem max_figures_in_grid : 
  maximal_figures_possible = total_unit_squares / units_per_figure := by
  sorry

end max_figures_in_grid_l173_173437


namespace monotonic_intervals_range_of_a_for_inequality_l173_173860

noncomputable def f (a x : ℝ) : ℝ := (x + a) / (a * Real.exp x)

theorem monotonic_intervals (a : ℝ) :
  (if a > 0 then
    ∀ x, (x < (1 - a) → 0 < deriv (f a) x) ∧ ((1 - a) < x → deriv (f a) x < 0)
  else
    ∀ x, (x < (1 - a) → deriv (f a) x < 0) ∧ ((1 - a) < x → 0 < deriv (f a) x)) := 
sorry

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x, 0 < x → (3 + 2 * Real.log x) / Real.exp x ≤ f a x + 2 * x) ↔
  a ∈ Set.Iio (-1/2) ∪ Set.Ioi 0 :=
sorry

end monotonic_intervals_range_of_a_for_inequality_l173_173860


namespace price_per_box_l173_173040

theorem price_per_box (total_apples : ℕ) (apples_per_box : ℕ) (total_revenue : ℕ) : 
  total_apples = 10000 → apples_per_box = 50 → total_revenue = 7000 → 
  total_revenue / (total_apples / apples_per_box) = 35 :=
by
  intros h1 h2 h3
  -- we can skip the actual proof with sorry. This indicates that the proof is not provided,
  -- but the statement is what needs to be proven.
  sorry

end price_per_box_l173_173040


namespace bobby_shoes_cost_l173_173768

theorem bobby_shoes_cost :
  let mold_cost := 250
  let hourly_rate := 75
  let hours_worked := 8
  let discount_rate := 0.20
  let labor_cost := hourly_rate * hours_worked
  let discounted_labor_cost := labor_cost * (1 - discount_rate)
  let total_cost := mold_cost + discounted_labor_cost
  mold_cost = 250 ∧ hourly_rate = 75 ∧ hours_worked = 8 ∧ discount_rate = 0.20 →
  total_cost = 730 := 
by
  sorry

end bobby_shoes_cost_l173_173768


namespace fourth_student_seat_number_l173_173610

theorem fourth_student_seat_number (n : ℕ) (pop_size sample_size : ℕ)
  (s1 s2 s3 : ℕ)
  (h_pop_size : pop_size = 52)
  (h_sample_size : sample_size = 4)
  (h_6_in_sample : s1 = 6)
  (h_32_in_sample : s2 = 32)
  (h_45_in_sample : s3 = 45)
  : ∃ s4 : ℕ, s4 = 19 :=
by
  sorry

end fourth_student_seat_number_l173_173610


namespace thabo_HNF_calculation_l173_173855

variable (THABO_BOOKS : ℕ)

-- Conditions as definitions
def total_books : ℕ := 500
def fiction_books : ℕ := total_books * 40 / 100
def non_fiction_books : ℕ := total_books * 60 / 100
def paperback_non_fiction_books (HNF : ℕ) : ℕ := HNF + 50
def total_non_fiction_books (HNF : ℕ) : ℕ := HNF + paperback_non_fiction_books HNF

-- Lean statement to prove
theorem thabo_HNF_calculation (HNF : ℕ) :
  total_books = 500 →
  fiction_books = 200 →
  non_fiction_books = 300 →
  total_non_fiction_books HNF = 300 →
  2 * HNF + 50 = 300 →
  HNF = 125 :=
by
  intros _
         _
         _
         _
         _
  sorry

end thabo_HNF_calculation_l173_173855


namespace abc_plus_ab_plus_a_div_4_l173_173448

noncomputable def prob_abc_div_4 (a b c : ℕ) (isPositive_a : 0 < a) (isPositive_b : 0 < b) (isPositive_c : 0 < c) (a_in_range : a ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (b_in_range : b ∈ {k | 1 ≤ k ∧ k ≤ 2009}) (c_in_range : c ∈ {k | 1 ≤ k ∧ k ≤ 2009}) : ℚ :=
  let total_elements : ℚ := 2009
  let multiples_of_4 := 502
  let non_multiples_of_4 := total_elements - multiples_of_4
  let prob_a_div_4 : ℚ := multiples_of_4 / total_elements
  let prob_a_not_div_4 : ℚ := non_multiples_of_4 / total_elements
  sorry

theorem abc_plus_ab_plus_a_div_4 : ∃ P : ℚ, prob_abc_div_4 a b c isPositive_a isPositive_b isPositive_c a_in_range b_in_range c_in_range = P :=
by sorry

end abc_plus_ab_plus_a_div_4_l173_173448


namespace selection_methods_count_l173_173289

noncomputable def num_selection_methods (total_students chosen_students : ℕ) (A B : ℕ) : ℕ :=
  let with_A_and_B := Nat.choose (total_students - 2) (chosen_students - 2)
  let with_one_A_or_B := Nat.choose (total_students - 2) (chosen_students - 1) * Nat.choose 2 1
  with_A_and_B + with_one_A_or_B

theorem selection_methods_count :
  num_selection_methods 10 4 1 2 = 140 :=
by
  -- We can add detailed proof here, for now we provide a placeholder
  sorry

end selection_methods_count_l173_173289


namespace calculate_expression_l173_173563

theorem calculate_expression :
  (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end calculate_expression_l173_173563


namespace Dawn_commissioned_paintings_l173_173758

theorem Dawn_commissioned_paintings (time_per_painting : ℕ) (total_earnings : ℕ) (earnings_per_hour : ℕ) 
  (h1 : time_per_painting = 2) 
  (h2 : total_earnings = 3600) 
  (h3 : earnings_per_hour = 150) : 
  (total_earnings / (time_per_painting * earnings_per_hour) = 12) :=
by 
  sorry

end Dawn_commissioned_paintings_l173_173758


namespace cos_inequality_l173_173050

open Real

-- Given angles of a triangle A, B, C

theorem cos_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hTriangle : A + B + C = π) :
  1 / (1 + cos B ^ 2 + cos C ^ 2) + 1 / (1 + cos C ^ 2 + cos A ^ 2) + 1 / (1 + cos A ^ 2 + cos B ^ 2) ≤ 2 :=
by
  sorry

end cos_inequality_l173_173050


namespace pq_implies_q_l173_173541

theorem pq_implies_q (p q : Prop) (h₁ : p ∨ q) (h₂ : ¬p) : q :=
by
  sorry

end pq_implies_q_l173_173541


namespace student_survey_l173_173073

-- Define the conditions given in the problem
theorem student_survey (S F : ℝ) (h1 : F = 25 + 65) (h2 : F = 0.45 * S) : S = 200 :=
by
  sorry

end student_survey_l173_173073


namespace english_vocab_related_to_reading_level_l173_173184

theorem english_vocab_related_to_reading_level (N : ℕ) (K_squared : ℝ) (critical_value : ℝ) (p_value : ℝ)
  (hN : N = 100)
  (hK_squared : K_squared = 7)
  (h_critical_value : critical_value = 6.635)
  (h_p_value : p_value = 0.010) :
  p_value <= 0.01 → K_squared > critical_value → true :=
by
  intro h_p_value_le h_K_squared_gt
  sorry

end english_vocab_related_to_reading_level_l173_173184


namespace draw_probability_l173_173217

variable (P_lose_a win_a : ℝ)
variable (not_lose_a : ℝ := 0.8)
variable (win_prob_a : ℝ := 0.6)

-- Given conditions
def A_not_losing : Prop := not_lose_a = win_prob_a + win_a

-- Main theorem to prove
theorem draw_probability : P_lose_a = 0.2 :=
by
  sorry

end draw_probability_l173_173217


namespace quadratic_distinct_roots_l173_173105

theorem quadratic_distinct_roots (p q₁ q₂ : ℝ) 
  (h_eq : p = q₁ + q₂ + 1) :
  q₁ ≥ 1/4 → 
  (∃ x, x^2 + x + q₁ = 0 ∧ ∃ x', x' ≠ x ∧ x'^2 + x' + q₁ = 0) 
  ∨ 
  (∃ y, y^2 + p*y + q₂ = 0 ∧ ∃ y', y' ≠ y ∧ y'^2 + p*y' + q₂ = 0) :=
by 
  sorry

end quadratic_distinct_roots_l173_173105


namespace sum_of_coordinates_of_B_l173_173335

theorem sum_of_coordinates_of_B (x y : ℕ) (hM : (2 * 6 = x + 10) ∧ (2 * 8 = y + 8)) :
    x + y = 10 :=
sorry

end sum_of_coordinates_of_B_l173_173335


namespace boundary_shadow_function_l173_173279

theorem boundary_shadow_function 
    (r : ℝ) (O P : ℝ × ℝ × ℝ) (f : ℝ → ℝ)
    (h_radius : r = 1)
    (h_center : O = (1, 0, 1))
    (h_light_source : P = (1, -1, 2)) :
  (∀ x, f x = (x - 1) ^ 2 / 4 - 1) := 
by 
  sorry

end boundary_shadow_function_l173_173279


namespace david_is_30_l173_173490

-- Definitions representing the conditions
def uncleBobAge : ℕ := 60
def emilyAge : ℕ := (2 * uncleBobAge) / 3
def davidAge : ℕ := emilyAge - 10

-- Statement that represents the equivalence to be proven
theorem david_is_30 : davidAge = 30 :=
by
  sorry

end david_is_30_l173_173490


namespace find_pool_length_l173_173638

noncomputable def pool_length : ℝ :=
  let drain_rate := 60 -- cubic feet per minute
  let width := 40 -- feet
  let depth := 10 -- feet
  let capacity_percent := 0.80
  let drain_time := 800 -- minutes
  let drained_volume := drain_rate * drain_time -- cubic feet
  let full_capacity := drained_volume / capacity_percent -- cubic feet
  let length := full_capacity / (width * depth) -- feet
  length

theorem find_pool_length : pool_length = 150 := by
  sorry

end find_pool_length_l173_173638


namespace percentage_dogs_and_video_games_l173_173029

theorem percentage_dogs_and_video_games (total_students : ℕ)
  (students_dogs_movies : ℕ)
  (students_prefer_dogs : ℕ) :
  total_students = 30 →
  students_dogs_movies = 3 →
  students_prefer_dogs = 18 →
  (students_prefer_dogs - students_dogs_movies) * 100 / total_students = 50 :=
by
  intros h1 h2 h3
  sorry

end percentage_dogs_and_video_games_l173_173029


namespace range_of_m_l173_173428

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 4 / y = 1) (H : x + y > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l173_173428


namespace compute_moles_of_NaHCO3_l173_173266

def equilibrium_constant : Real := 7.85 * 10^5

def balanced_equation (NaHCO3 HCl H2O CO2 NaCl : ℝ) : Prop :=
  NaHCO3 = HCl ∧ NaHCO3 = H2O ∧ NaHCO3 = CO2 ∧ NaHCO3 = NaCl

theorem compute_moles_of_NaHCO3
  (K : Real)
  (hK : K = 7.85 * 10^5)
  (HCl_required : ℝ)
  (hHCl : HCl_required = 2)
  (Water_formed : ℝ)
  (hWater : Water_formed = 2)
  (CO2_formed : ℝ)
  (hCO2 : CO2_formed = 2)
  (NaCl_formed : ℝ)
  (hNaCl : NaCl_formed = 2) :
  ∃ NaHCO3 : ℝ, NaHCO3 = 2 :=
by
  -- Conditions: equilibrium constant, balanced equation
  have equilibrium_condition := equilibrium_constant
  -- Here you would normally work through the steps of the proof using the given conditions,
  -- but we are setting it up as a theorem without a proof for now.
  existsi 2
  -- Placeholder for the formal proof.
  sorry

end compute_moles_of_NaHCO3_l173_173266


namespace q_is_false_given_conditions_l173_173971

theorem q_is_false_given_conditions
  (h₁: ¬(p ∧ q) = true) 
  (h₂: ¬¬p = true) 
  : q = false := 
sorry

end q_is_false_given_conditions_l173_173971


namespace sum_of_digits_correct_l173_173154

theorem sum_of_digits_correct :
  ∃ a b c : ℕ,
    (1 + 7 + 3 + a) % 9 = 0 ∧
    (1 + 3 - (7 + b)) % 11 = 0 ∧
    (c % 2 = 0) ∧
    ((1 + 7 + 3 + c) % 3 = 0) ∧
    (a + b + c = 19) :=
sorry

end sum_of_digits_correct_l173_173154


namespace reflection_line_slope_l173_173078

/-- Given two points (1, -2) and (7, 4), and the reflection line y = mx + b. 
    The image of (1, -2) under the reflection is (7, 4). Prove m + b = 4. -/
theorem reflection_line_slope (m b : ℝ)
    (h1: (∀ (x1 y1 x2 y2: ℝ), 
        (x1, y1) = (1, -2) → 
        (x2, y2) = (7, 4) → 
        (y2 - y1) / (x2 - x1) = 1)) 
    (h2: ∀ (x1 y1 x2 y2: ℝ),
        (x1, y1) = (1, -2) → 
        (x2, y2) = (7, 4) →
        (x1 + x2) / 2 = 4 ∧ (y1 + y2) / 2 = 1) 
    (h3: y = mx + b → m = -1 → (4, 1).1 = 4 ∧ (4, 1).2 = 1 → b = 5) : 
    m + b = 4 := by 
  -- No Proof Required
  sorry

end reflection_line_slope_l173_173078


namespace other_root_l173_173062

theorem other_root (k : ℝ) : 
  5 * (2:ℝ)^2 + k * (2:ℝ) - 8 = 0 → 
  ∃ q : ℝ, 5 * q^2 + k * q - 8 = 0 ∧ q ≠ 2 ∧ q = -4/5 :=
by {
  sorry
}

end other_root_l173_173062


namespace percent_kindergarten_combined_l173_173176

-- Define the constants provided in the problem
def studentsPinegrove : ℕ := 150
def studentsMaplewood : ℕ := 250

def percentKindergartenPinegrove : ℝ := 18.0
def percentKindergartenMaplewood : ℝ := 14.0

-- The proof statement
theorem percent_kindergarten_combined :
  (27.0 + 35.0) / (150.0 + 250.0) * 100.0 = 15.5 :=
by 
  sorry

end percent_kindergarten_combined_l173_173176


namespace inequality_abc_l173_173054

theorem inequality_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  abs (b / a - b / c) + abs (c / a - c / b) + abs (b * c + 1) > 1 :=
by
  sorry

end inequality_abc_l173_173054


namespace problem_solution_l173_173307

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_solution (a m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) →
  a = 1 ∧ (∃ n : ℝ, f n 1 ≤ m - f (-n) 1) → 4 ≤ m := 
by
  sorry

end problem_solution_l173_173307


namespace discount_percentage_correct_l173_173965

-- Definitions corresponding to the conditions
def number_of_toys : ℕ := 5
def cost_per_toy : ℕ := 3
def total_price_paid : ℕ := 12
def original_price : ℕ := number_of_toys * cost_per_toy
def discount_amount : ℕ := original_price - total_price_paid
def discount_percentage : ℕ := (discount_amount * 100) / original_price

-- Statement of the problem
theorem discount_percentage_correct :
  discount_percentage = 20 := 
  sorry

end discount_percentage_correct_l173_173965


namespace math_proof_l173_173269

noncomputable def side_length_of_smaller_square (d e f : ℕ) : ℝ :=
  (d - Real.sqrt e) / f

def are_positive_integers (d e f : ℕ) : Prop := d > 0 ∧ e > 0 ∧ f > 0
def is_not_divisible_by_square_of_any_prime (e : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p * p ∣ e)

def proof_problem : Prop :=
  ∃ (d e f : ℕ),
    are_positive_integers d e f ∧
    is_not_divisible_by_square_of_any_prime e ∧
    side_length_of_smaller_square d e f = (4 - Real.sqrt 10) / 3 ∧
    d + e + f = 17

theorem math_proof : proof_problem := sorry

end math_proof_l173_173269


namespace average_speed_correct_l173_173552

variable (t1 t2 : ℝ) -- time components in hours
variable (v1 v2 : ℝ) -- speed components in km/h

-- conditions
def time1 := 20 / 60 -- 20 minutes converted to hours
def time2 := 40 / 60 -- 40 minutes converted to hours
def speed1 := 60 -- speed in km/h for the first segment
def speed2 := 90 -- speed in km/h for the second segment

-- total distance traveled
def distance1 := speed1 * time1
def distance2 := speed2 * time2
def total_distance := distance1 + distance2

-- total time taken
def total_time := time1 + time2

-- average speed
def average_speed := total_distance / total_time

-- proof statement
theorem average_speed_correct : average_speed = 80 := by
  sorry

end average_speed_correct_l173_173552


namespace hexagonal_tile_difference_l173_173484

theorem hexagonal_tile_difference :
  let initial_blue_tiles := 15
  let initial_green_tiles := 9
  let new_green_border_tiles := 18
  let new_blue_border_tiles := 18
  let total_green_tiles := initial_green_tiles + new_green_border_tiles
  let total_blue_tiles := initial_blue_tiles + new_blue_border_tiles
  total_blue_tiles - total_green_tiles = 6 := by {
    sorry
  }

end hexagonal_tile_difference_l173_173484


namespace mean_equality_l173_173636

theorem mean_equality (z : ℚ) :
  ((8 + 7 + 28) / 3 : ℚ) = (14 + z) / 2 → z = 44 / 3 :=
by
  sorry

end mean_equality_l173_173636


namespace four_digit_number_count_l173_173088

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l173_173088


namespace gasoline_reduction_l173_173331

theorem gasoline_reduction (P Q : ℝ) :
  let new_price := 1.25 * P
  let new_budget := 1.10 * (P * Q)
  let new_quantity := new_budget / new_price
  let percent_reduction := 1 - (new_quantity / Q)
  percent_reduction = 0.12 :=
by
  sorry

end gasoline_reduction_l173_173331


namespace circle_center_l173_173309

theorem circle_center (x y : ℝ) :
  x^2 + y^2 - 4 * x - 2 * y - 5 = 0 → (x - 2)^2 + (y - 1)^2 = 10 :=
by sorry

end circle_center_l173_173309


namespace square_side_length_l173_173632

noncomputable def side_length_square_inscribed_in_hexagon : ℝ :=
  50 * Real.sqrt 3

theorem square_side_length (a b: ℝ) (h1 : a = 50) (h2 : b = 50 * (2 - Real.sqrt 3)) 
(s1 s2 s3 s4 s5 s6: ℝ) (ha : s1 = s2) (hb : s2 = s3) (hc : s3 = s4) 
(hd : s4 = s5) (he : s5 = s6) (hf : s6 = s1) : side_length_square_inscribed_in_hexagon = 50 * Real.sqrt 3 :=
by
  sorry

end square_side_length_l173_173632


namespace percent_of_x_eq_to_y_l173_173107

variable {x y : ℝ}

theorem percent_of_x_eq_to_y (h: 0.5 * (x - y) = 0.3 * (x + y)) : y = 0.25 * x :=
by
  sorry

end percent_of_x_eq_to_y_l173_173107


namespace mt_product_l173_173432

def g : ℝ → ℝ := sorry

axiom func_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

axiom g3_value : g 3 = 6

def m : ℕ := 1

def t : ℝ := 6

theorem mt_product : m * t = 6 :=
by 
  sorry

end mt_product_l173_173432


namespace abs_diff_inequality_l173_173774

theorem abs_diff_inequality (a b c h : ℝ) (hab : |a - c| < h) (hbc : |b - c| < h) : |a - b| < 2 * h := 
by
  sorry

end abs_diff_inequality_l173_173774


namespace perpendicular_lines_condition_l173_173657

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, x + (m + 1) * y = 2 - m → m * x + 2 * y = -8) ↔ m = -2 / 3 :=
by sorry

end perpendicular_lines_condition_l173_173657


namespace fraction_of_2d_nails_l173_173180

theorem fraction_of_2d_nails (x : ℝ) (h1 : x + 0.5 = 0.75) : x = 0.25 :=
by
  sorry

end fraction_of_2d_nails_l173_173180


namespace sealed_envelope_problem_l173_173251

theorem sealed_envelope_problem :
  ∃ (n : ℕ), (10 ≤ n ∧ n < 100) →
  ((n = 12 ∧ (n % 10 ≠ 2) ∧ n ≠ 35 ∧ (n % 10 ≠ 5)) ∨
   (n ≠ 12 ∧ (n % 10 ≠ 2) ∧ n = 35 ∧ (n % 10 = 5))) →
  ¬(n % 10 ≠ 5) :=
by
  sorry

end sealed_envelope_problem_l173_173251


namespace find_c_l173_173986

theorem find_c (c : ℝ) :
  (∃ (infinitely_many_y : ℝ → Prop), (∀ y, infinitely_many_y y ↔ 3 * (5 + 2 * c * y) = 18 * y + 15))
  → c = 3 :=
by
  sorry

end find_c_l173_173986


namespace digits_solution_l173_173790

noncomputable def validate_reverse_multiplication
  (A B C D E : ℕ) : Prop :=
  (A * 10000 + B * 1000 + C * 100 + D * 10 + E) * 4 =
  (E * 10000 + D * 1000 + C * 100 + B * 10 + A)

theorem digits_solution :
  validate_reverse_multiplication 2 1 9 7 8 :=
by
  sorry

end digits_solution_l173_173790


namespace molecular_weight_of_compound_l173_173781

noncomputable def molecularWeight (Ca_wt : ℝ) (O_wt : ℝ) (H_wt : ℝ) (nCa : ℕ) (nO : ℕ) (nH : ℕ) : ℝ :=
  (nCa * Ca_wt) + (nO * O_wt) + (nH * H_wt)

theorem molecular_weight_of_compound :
  molecularWeight 40.08 15.999 1.008 1 2 2 = 74.094 :=
by
  sorry

end molecular_weight_of_compound_l173_173781


namespace necessary_and_sufficient_condition_l173_173801

theorem necessary_and_sufficient_condition (a b : ℝ) : a > b ↔ a^3 > b^3 :=
by {
  sorry
}

end necessary_and_sufficient_condition_l173_173801


namespace mark_lloyd_ratio_l173_173212

theorem mark_lloyd_ratio (M L C : ℕ) (h1 : M = L) (h2 : M = C - 10) (h3 : C = 100) (h4 : M + L + C + 80 = 300) : M = L :=
by {
  sorry -- proof steps go here
}

end mark_lloyd_ratio_l173_173212


namespace abc_over_sum_leq_four_thirds_l173_173283

theorem abc_over_sum_leq_four_thirds (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) 
  (h_a_leq_2 : a ≤ 2) (h_b_leq_2 : b ≤ 2) (h_c_leq_2 : c ≤ 2) :
  (abc / (a + b + c) ≤ 4/3) :=
by
  sorry

end abc_over_sum_leq_four_thirds_l173_173283


namespace product_lcm_gcd_eq_128_l173_173747

theorem product_lcm_gcd_eq_128 : (Int.gcd 8 16) * (Int.lcm 8 16) = 128 :=
by
  sorry

end product_lcm_gcd_eq_128_l173_173747


namespace altitude_eqn_median_eqn_l173_173824

def Point := (ℝ × ℝ)

def A : Point := (4, 0)
def B : Point := (6, 7)
def C : Point := (0, 3)

theorem altitude_eqn (B C: Point) : 
  ∃ (k b : ℝ), (b = 6) ∧ (k = - 3 / 2) ∧ (∀ x y : ℝ, y = k * x + b →
  3 * x + 2 * y - 12 = 0)
:=
sorry

theorem median_eqn (A B C : Point) :
  ∃ (k b : ℝ), (b = 20) ∧ (k = -3/5) ∧ (∀ x y : ℝ, y = k * x + b →
  5 * x + y - 20 = 0)
:=
sorry

end altitude_eqn_median_eqn_l173_173824


namespace album_cost_l173_173300

-- Definitions for given conditions
def M (X : ℕ) : ℕ := X - 2
def K (X : ℕ) : ℕ := X - 34
def F (X : ℕ) : ℕ := X - 35

-- We need to prove that X = 35
theorem album_cost : ∃ X : ℕ, (M X) + (K X) + (F X) < X ∧ X = 35 :=
by
  sorry -- Proof not required.

end album_cost_l173_173300


namespace area_triangle_PZQ_l173_173007

/-- 
In rectangle PQRS, side PQ measures 8 units and side QR measures 4 units.
Points X and Y are on side RS such that segment RX measures 2 units and
segment SY measures 3 units. Lines PX and QY intersect at point Z.
Prove the area of triangle PZQ is 128/3 square units.
-/

theorem area_triangle_PZQ {PQ QR RX SY : ℝ} (h1 : PQ = 8) (h2 : QR = 4) (h3 : RX = 2) (h4 : SY = 3) :
  let area_PZQ : ℝ := 8 * 4 / 2 * 8 / (3 * 2)
  area_PZQ = 128 / 3 :=
by
  sorry

end area_triangle_PZQ_l173_173007


namespace max_value_inequality_l173_173568

theorem max_value_inequality (x y k : ℝ) (hx : 0 < x) (hy : 0 < y) (hk : 0 < k) :
  (kx + y)^2 / (x^2 + y^2) ≤ 2 :=
sorry

end max_value_inequality_l173_173568


namespace cost_of_10_apples_l173_173685

-- Define the price for 10 apples as a variable
noncomputable def price_10_apples (P : ℝ) : ℝ := P

-- Theorem stating that the cost for 10 apples is the provided price
theorem cost_of_10_apples (P : ℝ) : price_10_apples P = P :=
  by
    sorry

end cost_of_10_apples_l173_173685


namespace train_length_l173_173282

theorem train_length :
  (∃ L : ℕ, (L / 15) = (L + 800) / 45) → L = 400 :=
by
  sorry

end train_length_l173_173282


namespace find_f_5_l173_173389

def f (x : ℝ) : ℝ := sorry -- we need to create a function under our condition

theorem find_f_5 : f 5 = 0 :=
sorry

end find_f_5_l173_173389


namespace find_number_l173_173983

-- Define the given conditions and statement as Lean types
theorem find_number (x : ℝ) :
  (0.3 * x > 0.6 * 50 + 30) -> x = 200 :=
by
  -- Proof here
  sorry

end find_number_l173_173983


namespace units_digit_of_expression_l173_173923

noncomputable def units_digit (n : ℕ) : ℕ :=
  n % 10

def expr : ℕ := 2 * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9)

theorem units_digit_of_expression : units_digit expr = 6 :=
by
  sorry

end units_digit_of_expression_l173_173923


namespace value_of_power_l173_173516

theorem value_of_power (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2014 = 1 :=
by
  sorry

end value_of_power_l173_173516


namespace deers_distribution_l173_173033

theorem deers_distribution (a_1 d a_2 a_5 : ℚ) 
  (h1 : a_2 = a_1 + d)
  (h2 : 5 * a_1 + 10 * d = 5)
  (h3 : a_2 = 2 / 3) :
  a_5 = 1 / 3 :=
sorry

end deers_distribution_l173_173033


namespace expression_eq_16x_l173_173453

variable (x y z w : ℝ)

theorem expression_eq_16x
  (h1 : y = 2 * x)
  (h2 : z = 3 * y)
  (h3 : w = z + x) :
  x + y + z + w = 16 * x :=
sorry

end expression_eq_16x_l173_173453


namespace trapezoid_upper_side_length_l173_173262

theorem trapezoid_upper_side_length (area base1 height : ℝ) (h1 : area = 222) (h2 : base1 = 23) (h3 : height = 12) : 
  ∃ base2, base2 = 14 :=
by
  -- The proof will be provided here.
  sorry

end trapezoid_upper_side_length_l173_173262


namespace unique_real_solution_l173_173123

theorem unique_real_solution :
  ∀ x : ℝ, (x > 0 → (x ^ 16 + 1) * (x ^ 12 + x ^ 8 + x ^ 4 + 1) = 18 * x ^ 8 → x = 1) :=
by
  introv
  sorry

end unique_real_solution_l173_173123


namespace geom_seq_min_val_l173_173294

-- Definition of geometric sequence with common ratio q
def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Main theorem
theorem geom_seq_min_val (a : ℕ → ℝ) (q : ℝ) 
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_geom : geom_seq a q)
  (h_cond : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) :
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end geom_seq_min_val_l173_173294


namespace number_of_orders_l173_173198

open Nat

theorem number_of_orders (total_targets : ℕ) (targets_A : ℕ) (targets_B : ℕ) (targets_C : ℕ)
  (h1 : total_targets = 10)
  (h2 : targets_A = 4)
  (h3 : targets_B = 3)
  (h4 : targets_C = 3)
  : total_orders = 80 :=
sorry

end number_of_orders_l173_173198


namespace fraction_identity_l173_173327

theorem fraction_identity : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 :=
by
  sorry

end fraction_identity_l173_173327


namespace range_of_a_plus_b_l173_173189

variable {a b : ℝ}

-- Assumptions
def are_positive_and_unequal (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a ≠ b
def equation_holds (a b : ℝ) : Prop := a^2 - a + b^2 - b + a * b = 0

-- Problem Statement
theorem range_of_a_plus_b (h₁ : are_positive_and_unequal a b) (h₂ : equation_holds a b) : 1 < a + b ∧ a + b < 4 / 3 :=
sorry

end range_of_a_plus_b_l173_173189


namespace range_of_a_for_three_distinct_real_roots_l173_173439

theorem range_of_a_for_three_distinct_real_roots (a : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x, f x = x^3 - 3*x^2 - a ∧ ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end range_of_a_for_three_distinct_real_roots_l173_173439


namespace intersection_x_value_l173_173700

theorem intersection_x_value :
  (∃ x y, y = 3 * x - 7 ∧ y = 48 - 5 * x) → x = 55 / 8 :=
by
  sorry

end intersection_x_value_l173_173700


namespace minimum_club_members_l173_173859

theorem minimum_club_members : ∃ (b : ℕ), (b = 7) ∧ ∃ (a : ℕ), (2 : ℚ) / 5 < (a : ℚ) / b ∧ (a : ℚ) / b < 1 / 2 := 
sorry

end minimum_club_members_l173_173859


namespace cards_net_cost_equivalence_l173_173559

-- Define the purchase amount
def purchase_amount : ℝ := 10000

-- Define cashback percentages
def debit_card_cashback : ℝ := 0.01
def credit_card_cashback : ℝ := 0.005

-- Define interest rate for keeping money in the debit account
def interest_rate : ℝ := 0.005

-- A function to calculate the net cost after 1 month using the debit card
def net_cost_debit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage

-- A function to calculate the net cost after 1 month using the credit card
def net_cost_credit_card (purchase_amount : ℝ) (cashback_percentage : ℝ) (interest_rate : ℝ) : ℝ :=
  purchase_amount - purchase_amount * cashback_percentage - purchase_amount * interest_rate

-- Final theorem stating that the net cost using both cards is the same
theorem cards_net_cost_equivalence : 
  net_cost_debit_card purchase_amount debit_card_cashback = 
  net_cost_credit_card purchase_amount credit_card_cashback interest_rate :=
by
  sorry

end cards_net_cost_equivalence_l173_173559


namespace friends_activity_l173_173826

-- Defining the problem conditions
def total_friends : ℕ := 5
def organizers : ℕ := 3
def managers : ℕ := total_friends - organizers

-- Stating the proof problem
theorem friends_activity (h1 : organizers = 3) (h2 : managers = 2) :
  Nat.choose total_friends organizers = 10 :=
sorry

end friends_activity_l173_173826


namespace y_intercept_exists_l173_173014

def line_eq (x y : ℝ) : Prop := x + 2 * y + 2 = 0

theorem y_intercept_exists : ∃ y : ℝ, line_eq 0 y ∧ y = -1 :=
by
  sorry

end y_intercept_exists_l173_173014


namespace negative_exp_eq_l173_173378

theorem negative_exp_eq :
  (-2 : ℤ)^3 = (-2 : ℤ)^3 := by
  sorry

end negative_exp_eq_l173_173378


namespace base_rate_first_company_proof_l173_173537

noncomputable def base_rate_first_company : ℝ := 8.00
def charge_per_minute_first_company : ℝ := 0.25
def base_rate_second_company : ℝ := 12.00
def charge_per_minute_second_company : ℝ := 0.20
def minutes : ℕ := 80

theorem base_rate_first_company_proof :
  base_rate_first_company = 8.00 :=
sorry

end base_rate_first_company_proof_l173_173537


namespace hill_height_l173_173321

theorem hill_height (h : ℝ) (time_up : ℝ := h / 9) (time_down : ℝ := h / 12) (total_time : ℝ := time_up + time_down) (time_cond : total_time = 175) : h = 900 :=
by 
  sorry

end hill_height_l173_173321


namespace Sharik_cannot_eat_all_meatballs_within_one_million_flies_l173_173261

theorem Sharik_cannot_eat_all_meatballs_within_one_million_flies:
  (∀ n: ℕ, ∃ i: ℕ, i > n ∧ ((∀ j < i, ∀ k: ℕ, ∃ m: ℕ, (m ≠ k) → (∃ f, f < 10^6) )) → f > 10^6 ) :=
sorry

end Sharik_cannot_eat_all_meatballs_within_one_million_flies_l173_173261


namespace fraction_multiplication_l173_173267

noncomputable def a : ℚ := 5 / 8
noncomputable def b : ℚ := 7 / 12
noncomputable def c : ℚ := 3 / 7
noncomputable def n : ℚ := 1350

theorem fraction_multiplication : a * b * c * n = 210.9375 := by
  sorry

end fraction_multiplication_l173_173267


namespace pawns_on_black_squares_even_l173_173825

theorem pawns_on_black_squares_even (A : Fin 8 → Fin 8) :
  ∃ n : ℕ, ∀ i, (i + A i).val % 2 = 1 → n % 2 = 0 :=
sorry

end pawns_on_black_squares_even_l173_173825


namespace find_g7_l173_173233

-- Given the required functional equation and specific value g(6) = 7
theorem find_g7 (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x + g y) (H2 : g 6 = 7) : g 7 = 49 / 6 := by
  sorry

end find_g7_l173_173233


namespace cube_volume_proof_l173_173529

-- Define the conditions
def len_inch : ℕ := 48
def width_inch : ℕ := 72
def total_surface_area_inch : ℕ := len_inch * width_inch
def num_faces : ℕ := 6
def area_one_face_inch : ℕ := total_surface_area_inch / num_faces
def inches_to_feet (length_in_inches : ℕ) : ℕ := length_in_inches / 12

-- Define the key elements of the proof problem
def side_length_inch : ℕ := Int.natAbs (Nat.sqrt area_one_face_inch)
def side_length_ft : ℕ := inches_to_feet side_length_inch
def volume_ft3 : ℕ := side_length_ft ^ 3

-- State the proof problem
theorem cube_volume_proof : volume_ft3 = 8 := by
  -- The proof would be implemented here
  sorry

end cube_volume_proof_l173_173529


namespace minimum_jumps_to_cover_circle_l173_173488

/--
Given 2016 points arranged in a circle and the ability to jump either 2 or 3 points clockwise,
prove that the minimum number of jumps required to visit every point at least once and return to the starting 
point is 2017.
-/
theorem minimum_jumps_to_cover_circle (n : Nat) (h : n = 2016) : 
  ∃ (a b : Nat), 2 * a + 3 * b = n ∧ (a + b) = 2017 := 
sorry

end minimum_jumps_to_cover_circle_l173_173488


namespace flight_relation_not_preserved_l173_173956

noncomputable def swap_city_flights (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) : Prop := sorry

theorem flight_relation_not_preserved (cities : Finset ℕ) (flights : ℕ → ℕ → Bool) (M N : ℕ) (hM : M ∈ cities) (hN : N ∈ cities) : 
  ¬ swap_city_flights cities flights :=
sorry

end flight_relation_not_preserved_l173_173956


namespace least_multiple_of_21_gt_380_l173_173844

theorem least_multiple_of_21_gt_380 : ∃ n : ℕ, (21 * n > 380) ∧ (21 * n = 399) :=
sorry

end least_multiple_of_21_gt_380_l173_173844


namespace m_range_l173_173897

noncomputable def otimes (a b : ℝ) : ℝ := 
if a > b then a else b

theorem m_range (m : ℝ) : (otimes (2 * m - 5) 3 = 3) ↔ (m ≤ 4) := by
  sorry

end m_range_l173_173897


namespace smallest_value_other_integer_l173_173083

noncomputable def smallest_possible_value_b : ℕ :=
  by sorry

theorem smallest_value_other_integer (x : ℕ) (h_pos : x > 0) (b : ℕ) 
  (h_gcd : Nat.gcd 36 b = x + 3) (h_lcm : Nat.lcm 36 b = x * (x + 3)) :
  b = 108 :=
  by sorry

end smallest_value_other_integer_l173_173083


namespace inequality_proof_l173_173681

theorem inequality_proof (p : ℝ) (x y z v : ℝ) (hp : p ≥ 2) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v ≥ 0) :
  (x + y) ^ p + (z + v) ^ p + (x + z) ^ p + (y + v) ^ p ≤ x ^ p + y ^ p + z ^ p + v ^ p + (x + y + z + v) ^ p := 
by sorry

end inequality_proof_l173_173681


namespace hyperbola_eccentricity_is_5_over_3_l173_173452

noncomputable def hyperbola_asymptote_condition (a b : ℝ) : Prop :=
  a / b = 3 / 4

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_is_5_over_3 (a b : ℝ) (h : hyperbola_asymptote_condition a b) :
  hyperbola_eccentricity a b = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_is_5_over_3_l173_173452


namespace real_numbers_satisfy_relation_l173_173762

theorem real_numbers_satisfy_relation (a b : ℝ) :
  2 * (a^2 + 1) * (b^2 + 1) = (a + 1) * (b + 1) * (a * b + 1) → 
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end real_numbers_satisfy_relation_l173_173762


namespace equation_for_number_l173_173651

variable (a : ℤ)

theorem equation_for_number : 3 * a + 5 = 9 :=
sorry

end equation_for_number_l173_173651


namespace lottery_probability_l173_173769

theorem lottery_probability (x_1 x_2 x_3 x_4 : ℝ) (p : ℝ) (h0 : 0 < p ∧ p < 1) : 
  x_1 = p * x_3 → 
  x_2 = p * x_4 + (1 - p) * x_1 → 
  x_3 = p + (1 - p) * x_2 → 
  x_4 = p + (1 - p) * x_3 → 
  x_2 = 0.19 :=
by
  sorry

end lottery_probability_l173_173769


namespace files_per_folder_l173_173854

theorem files_per_folder
    (initial_files : ℕ)
    (deleted_files : ℕ)
    (folders : ℕ)
    (remaining_files : ℕ)
    (files_per_folder : ℕ)
    (initial_files_eq : initial_files = 93)
    (deleted_files_eq : deleted_files = 21)
    (folders_eq : folders = 9)
    (remaining_files_eq : remaining_files = initial_files - deleted_files)
    (files_per_folder_eq : files_per_folder = remaining_files / folders) :
    files_per_folder = 8 :=
by
    -- Here, sorry is used to skip the actual proof steps 
    sorry

end files_per_folder_l173_173854


namespace spadesuit_proof_l173_173136

def spadesuit (a b : ℤ) : ℤ := abs (a - b)

theorem spadesuit_proof : 
  spadesuit (spadesuit 5 2) (spadesuit 9 (spadesuit 3 6)) = 3 :=
by
  sorry

end spadesuit_proof_l173_173136


namespace opposite_sqrt3_l173_173458

def opposite (x : ℝ) : ℝ := -x

theorem opposite_sqrt3 :
  opposite (Real.sqrt 3) = -Real.sqrt 3 :=
by
  sorry

end opposite_sqrt3_l173_173458


namespace rational_coefficient_exists_in_binomial_expansion_l173_173588

theorem rational_coefficient_exists_in_binomial_expansion :
  ∃! (n : ℕ), n > 0 ∧ (∀ r, (r % 3 = 0 → (n - r) % 2 = 0 → n = 7)) :=
by
  sorry

end rational_coefficient_exists_in_binomial_expansion_l173_173588


namespace no_integer_solutions_l173_173323

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), 21 * x - 35 * y = 59 :=
by
  sorry

end no_integer_solutions_l173_173323


namespace min_distance_l173_173998

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance :
  ∃ m : ℝ, (∀ x > 0, x ≠ m → (f m - g m) ≤ (f x - g x)) ∧ m = Real.sqrt 2 / 2 :=
by
  sorry

end min_distance_l173_173998


namespace olivia_remaining_usd_l173_173609

def initial_usd : ℝ := 78
def initial_eur : ℝ := 50
def exchange_rate : ℝ := 1.20
def spent_usd_supermarket : ℝ := 15
def book_eur : ℝ := 10
def spent_usd_lunch : ℝ := 12

theorem olivia_remaining_usd :
  let total_usd := initial_usd + (initial_eur * exchange_rate)
  let remaining_after_supermarket := total_usd - spent_usd_supermarket
  let remaining_after_book := remaining_after_supermarket - (book_eur * exchange_rate)
  let final_remaining := remaining_after_book - spent_usd_lunch
  final_remaining = 99 :=
by
  sorry

end olivia_remaining_usd_l173_173609


namespace projection_of_a_onto_b_l173_173451

open Real

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-2, 4)

theorem projection_of_a_onto_b :
  let dot_product := vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2
  let magnitude_b_squared := vector_b.1 ^ 2 + vector_b.2 ^ 2
  let scalar_projection := dot_product / magnitude_b_squared
  let proj_vector := (scalar_projection * vector_b.1, scalar_projection * vector_b.2)
  proj_vector = (-4/5, 8/5) :=
by
  sorry

end projection_of_a_onto_b_l173_173451


namespace sum_of_first_ten_terms_l173_173508

theorem sum_of_first_ten_terms (S : ℕ → ℕ) (h : ∀ n, S n = n^2 - 4 * n + 1) : S 10 = 61 :=
by
  sorry

end sum_of_first_ten_terms_l173_173508


namespace Vlad_score_l173_173511

-- Defining the initial conditions of the problem
def total_rounds : ℕ := 30
def points_per_win : ℕ := 5
def total_points : ℕ := total_rounds * points_per_win

-- Taro's score as described in the problem
def Taros_score := (3 * total_points / 5) - 4

-- Prove that Vlad's score is 64 points
theorem Vlad_score : total_points - Taros_score = 64 := by
  sorry

end Vlad_score_l173_173511


namespace quadratic_no_real_roots_l173_173060

theorem quadratic_no_real_roots :
  ∀ x : ℝ, ¬(x^2 - 2 * x + 3 = 0) :=
by
  sorry

end quadratic_no_real_roots_l173_173060


namespace inverse_variation_l173_173074

theorem inverse_variation (x y : ℝ) (h1 : 7 * y = 1400 / x^3) (h2 : x = 4) : y = 25 / 8 :=
  by
  sorry

end inverse_variation_l173_173074


namespace calculate_total_parts_l173_173947

theorem calculate_total_parts (sample_size : ℕ) (draw_probability : ℚ) (N : ℕ) 
  (h_sample_size : sample_size = 30) 
  (h_draw_probability : draw_probability = 0.25) 
  (h_relation : sample_size = N * draw_probability) : 
  N = 120 :=
by
  rw [h_sample_size, h_draw_probability] at h_relation
  sorry

end calculate_total_parts_l173_173947


namespace problem_l173_173085

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 2}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def compN : Set ℝ := {x | x < -1 ∨ 1 < x}
def intersection : Set ℝ := {x | x < -1 ∨ (1 < x ∧ x ≤ 2)}

theorem problem (x : ℝ) : x ∈ (M ∩ compN) ↔ x ∈ intersection := by
  sorry

end problem_l173_173085


namespace quadratic_factorization_b_value_l173_173146

theorem quadratic_factorization_b_value (b : ℤ) (c d e f : ℤ) (h1 : 24 * c + 24 * d = 240) :
  (24 * (c * e) + b + 24) = 0 →
  (c * e = 24) →
  (c * f + d * e = b) →
  (d * f = 24) →
  (c + d = 10) →
  b = 52 :=
by
  intros
  sorry

end quadratic_factorization_b_value_l173_173146


namespace end_of_month_books_count_l173_173435

theorem end_of_month_books_count:
  ∀ (initial_books : ℝ) (loaned_out_books : ℝ) (return_rate : ℝ)
    (rounded_loaned_out_books : ℝ) (returned_books : ℝ)
    (not_returned_books : ℝ) (end_of_month_books : ℝ),
    initial_books = 75 →
    loaned_out_books = 60.00000000000001 →
    return_rate = 0.65 →
    rounded_loaned_out_books = 60 →
    returned_books = return_rate * rounded_loaned_out_books →
    not_returned_books = rounded_loaned_out_books - returned_books →
    end_of_month_books = initial_books - not_returned_books →
    end_of_month_books = 54 :=
by
  intros initial_books loaned_out_books return_rate
         rounded_loaned_out_books returned_books
         not_returned_books end_of_month_books
  intros h_initial_books h_loaned_out_books h_return_rate
         h_rounded_loaned_out_books h_returned_books
         h_not_returned_books h_end_of_month_books
  sorry

end end_of_month_books_count_l173_173435


namespace tan_sum_pi_over_4_x_l173_173527

theorem tan_sum_pi_over_4_x (x : ℝ) (h1 : x > -π/2 ∧ x < 0) (h2 : Real.cos x = 4/5) :
  Real.tan (π/4 + x) = 1/7 :=
by
  sorry

end tan_sum_pi_over_4_x_l173_173527


namespace john_pre_lunch_drive_l173_173936

def drive_before_lunch (h : ℕ) : Prop :=
  45 * h + 45 * 3 = 225

theorem john_pre_lunch_drive : ∃ h : ℕ, drive_before_lunch h ∧ h = 2 :=
by
  sorry

end john_pre_lunch_drive_l173_173936


namespace total_wet_surface_area_l173_173084

-- Necessary definitions based on conditions
def length : ℝ := 6
def width : ℝ := 4
def water_level : ℝ := 1.25

-- Defining the areas
def bottom_area : ℝ := length * width
def side_area (height : ℝ) (side_length : ℝ) : ℝ := height * side_length

-- Proof statement
theorem total_wet_surface_area :
  bottom_area + 2 * side_area water_level length + 2 * side_area water_level width = 49 := 
sorry

end total_wet_surface_area_l173_173084


namespace Jessie_weight_l173_173339

theorem Jessie_weight (c l w : ℝ) (hc : c = 27) (hl : l = 101) : c + l = w ↔ w = 128 := by
  sorry

end Jessie_weight_l173_173339


namespace decrease_percent_in_revenue_l173_173140

theorem decrease_percent_in_revenue 
  (T C : ℝ) 
  (original_revenue : ℝ := T * C)
  (new_tax : ℝ := 0.80 * T)
  (new_consumption : ℝ := 1.15 * C)
  (new_revenue : ℝ := new_tax * new_consumption) :
  ((original_revenue - new_revenue) / original_revenue) * 100 = 8 := 
sorry

end decrease_percent_in_revenue_l173_173140


namespace fish_count_total_l173_173446

def Jerk_Tuna_fish : ℕ := 144
def Tall_Tuna_fish : ℕ := 2 * Jerk_Tuna_fish
def Total_fish_together : ℕ := Jerk_Tuna_fish + Tall_Tuna_fish

theorem fish_count_total :
  Total_fish_together = 432 :=
by
  sorry

end fish_count_total_l173_173446


namespace N_subset_M_l173_173793

def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x - 2 = 0}

theorem N_subset_M : N ⊆ M := sorry

end N_subset_M_l173_173793


namespace complex_fraction_value_l173_173733

-- Define the imaginary unit
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_value : (3 : ℂ) / ((1 - i) ^ 2) = (3 / 2) * i := by
  sorry

end complex_fraction_value_l173_173733


namespace smallest_k_l173_173974

-- Define the non-decreasing property of digits in a five-digit number
def non_decreasing (n : Fin 5 → ℕ) : Prop :=
  n 0 ≤ n 1 ∧ n 1 ≤ n 2 ∧ n 2 ≤ n 3 ∧ n 3 ≤ n 4

-- Define the overlap property in at least one digit
def overlap (n1 n2 : Fin 5 → ℕ) : Prop :=
  ∃ i : Fin 5, n1 i = n2 i

-- The main theorem stating the problem
theorem smallest_k {N1 Nk : Fin 5 → ℕ} :
  (∀ n : Fin 5 → ℕ, non_decreasing n → overlap N1 n ∨ overlap Nk n) → 
  ∃ (k : Nat), k = 2 :=
sorry

end smallest_k_l173_173974


namespace rectangle_side_ratio_l173_173287

theorem rectangle_side_ratio
  (s : ℝ)  -- the side length of the inner square
  (y x : ℝ) -- the side lengths of the rectangles (y: shorter, x: longer)
  (h1 : 9 * s^2 = (3 * s)^2)  -- the area of the outer square is 9 times that of the inner square
  (h2 : s + 2*y = 3*s)  -- the total side length relation due to geometry
  (h3 : x + y = 3*s)  -- another side length relation
: x / y = 2 :=
by
  sorry

end rectangle_side_ratio_l173_173287


namespace f_odd_f_increasing_on_2_infty_solve_inequality_f_l173_173173

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem f_odd (x : ℝ) (hx : x ≠ 0) : f (-x) = -f x := by
  sorry

theorem f_increasing_on_2_infty (x₁ x₂ : ℝ) (hx₁ : 2 < x₁) (hx₂ : 2 < x₂) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

theorem solve_inequality_f (x : ℝ) (hx : -5 < x ∧ x < -1) : f (2*x^2 + 5*x + 8) + f (x - 3 - x^2) < 0 := by
  sorry

end f_odd_f_increasing_on_2_infty_solve_inequality_f_l173_173173


namespace exists_m_square_between_l173_173792

theorem exists_m_square_between (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : a * d = b * c) : 
  ∃ m : ℤ, a < m^2 ∧ m^2 < d := 
sorry

end exists_m_square_between_l173_173792


namespace group_B_same_order_l173_173746

-- Definitions for the expressions in each group
def expr_A1 := 2 * 9 / 3
def expr_A2 := 2 + 9 * 3

def expr_B1 := 36 - 9 + 5
def expr_B2 := 36 / 6 * 5

def expr_C1 := 56 / 7 * 5
def expr_C2 := 56 + 7 * 5

-- Theorem stating that Group B expressions have the same order of operations
theorem group_B_same_order : (expr_B1 = expr_B2) := 
  sorry

end group_B_same_order_l173_173746


namespace inequality_holds_for_all_real_l173_173545

theorem inequality_holds_for_all_real (k : ℝ) :
  (∀ x : ℝ, k * x ^ 2 - 6 * k * x + k + 8 ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end inequality_holds_for_all_real_l173_173545


namespace average_score_l173_173394

variable (score : Fin 5 → ℤ)
variable (actual_score : ℤ)
variable (rank : Fin 5)
variable (average : ℤ)

def students_scores_conditions := 
  score 0 = 10 ∧ score 1 = -5 ∧ score 2 = 0 ∧ score 3 = 8 ∧ score 4 = -3 ∧
  actual_score = 90 ∧ rank.val = 2

theorem average_score (h : students_scores_conditions score actual_score rank) :
  average = 92 :=
sorry

end average_score_l173_173394


namespace point_on_x_axis_l173_173065

theorem point_on_x_axis (m : ℝ) (h : 3 * m + 1 = 0) : m = -1 / 3 :=
by 
  sorry

end point_on_x_axis_l173_173065


namespace sample_size_drawn_l173_173924

theorem sample_size_drawn (sample_size : ℕ) (probability : ℚ) (N : ℚ) 
  (h1 : sample_size = 30) 
  (h2 : probability = 0.25) 
  (h3 : probability = sample_size / N) : 
  N = 120 := by
  sorry

end sample_size_drawn_l173_173924


namespace frank_is_15_years_younger_than_john_l173_173137

variables (F J : ℕ)

theorem frank_is_15_years_younger_than_john
  (h1 : J + 3 = 2 * (F + 3))
  (h2 : F + 4 = 16) : J - F = 15 := by
  sorry

end frank_is_15_years_younger_than_john_l173_173137


namespace solve_x_l173_173948

noncomputable def op (a b : ℝ) : ℝ := (1 / b) - (1 / a)

theorem solve_x (x : ℝ) (h : op (x - 1) 2 = 1) : x = -1 := 
by {
  -- proof outline here...
  sorry
}

end solve_x_l173_173948


namespace abs_diff_two_numbers_l173_173115

theorem abs_diff_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 := by
  sorry

end abs_diff_two_numbers_l173_173115


namespace trajectory_of_center_of_moving_circle_l173_173726

noncomputable def circle_tangency_condition_1 (x y : ℝ) : Prop := (x + 1) ^ 2 + y ^ 2 = 1
noncomputable def circle_tangency_condition_2 (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 9

def ellipse_equation (x y : ℝ) : Prop := x ^ 2 / 4 + y ^ 2 / 3 = 1

theorem trajectory_of_center_of_moving_circle (x y : ℝ) :
  circle_tangency_condition_1 x y ∧ circle_tangency_condition_2 x y →
  ellipse_equation x y := sorry

end trajectory_of_center_of_moving_circle_l173_173726


namespace seq_fifth_term_l173_173296

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 3) ∧ (a 2 = 6) ∧ (∀ n : ℕ, a (n + 2) = a (n + 1) - a n)

theorem seq_fifth_term (a : ℕ → ℤ) (h : seq a) : a 5 = -6 :=
by
  sorry

end seq_fifth_term_l173_173296


namespace inheritance_value_l173_173869

def inheritance_proof (x : ℝ) (federal_tax_ratio : ℝ) (state_tax_ratio : ℝ) (total_tax : ℝ) : Prop :=
  let federal_taxes := federal_tax_ratio * x
  let remaining_after_federal := x - federal_taxes
  let state_taxes := state_tax_ratio * remaining_after_federal
  let total_taxes := federal_taxes + state_taxes
  total_taxes = total_tax

theorem inheritance_value :
  inheritance_proof 41379 0.25 0.15 15000 :=
by
  sorry

end inheritance_value_l173_173869


namespace production_days_l173_173174

theorem production_days (n P : ℕ) (h1 : P = 60 * n) (h2 : (P + 90) / (n + 1) = 65) : n = 5 := sorry

end production_days_l173_173174


namespace simplify_expression_l173_173218

variable {x y z : ℝ}

theorem simplify_expression (h : x^2 - y^2 ≠ 0) (hx : x ≠ 0) (hz : z ≠ 0) :
  (x^2 - y^2)⁻¹ * (x⁻¹ - z⁻¹) = (z - x) * x⁻¹ * z⁻¹ * (x^2 - y^2)⁻¹ := by
  sorry

end simplify_expression_l173_173218


namespace weight_of_b_l173_173507

/--
Given:
1. The sum of weights (a, b, c) is 129 kg.
2. The sum of weights (a, b) is 80 kg.
3. The sum of weights (b, c) is 86 kg.

Prove that the weight of b is 37 kg.
-/
theorem weight_of_b (a b c : ℝ) 
  (h1 : a + b + c = 129) 
  (h2 : a + b = 80) 
  (h3 : b + c = 86) : 
  b = 37 :=
sorry

end weight_of_b_l173_173507


namespace kelly_baking_powder_difference_l173_173980

theorem kelly_baking_powder_difference :
  let amount_yesterday := 0.4
  let amount_now := 0.3
  amount_yesterday - amount_now = 0.1 :=
by
  -- Definitions for amounts 
  let amount_yesterday := 0.4
  let amount_now := 0.3
  
  -- Applying definitions in the computation
  show amount_yesterday - amount_now = 0.1
  sorry

end kelly_baking_powder_difference_l173_173980


namespace rectangle_length_width_l173_173985

theorem rectangle_length_width (x y : ℝ) (h1 : 2 * (x + y) = 26) (h2 : x * y = 42) : 
  (x = 7 ∧ y = 6) ∨ (x = 6 ∧ y = 7) :=
by
  sorry

end rectangle_length_width_l173_173985


namespace y_power_x_equals_49_l173_173433

theorem y_power_x_equals_49 (x y : ℝ) (h : |x - 2| = -(y + 7)^2) : y ^ x = 49 := by
  sorry

end y_power_x_equals_49_l173_173433


namespace boys_and_girls_l173_173116

theorem boys_and_girls (x y : ℕ) (h1 : x + y = 21) (h2 : 5 * x + 2 * y = 69) : x = 9 ∧ y = 12 :=
by 
  sorry

end boys_and_girls_l173_173116


namespace distinct_integers_problem_l173_173578

variable (a b c d e : ℤ)

theorem distinct_integers_problem
  (h1 : a ≠ b) 
  (h2 : a ≠ c) 
  (h3 : a ≠ d) 
  (h4 : a ≠ e) 
  (h5 : b ≠ c) 
  (h6 : b ≠ d) 
  (h7 : b ≠ e) 
  (h8 : c ≠ d) 
  (h9 : c ≠ e) 
  (h10 : d ≠ e) 
  (h_prod : (4 - a) * (4 - b) * (4 - c) * (4 - d) * (4 - e) = 12) : 
  a + b + c + d + e = 17 := 
sorry

end distinct_integers_problem_l173_173578


namespace number_of_children_tickets_l173_173020

theorem number_of_children_tickets 
    (x y : ℤ) 
    (h1 : x + y = 225) 
    (h2 : 6 * x + 9 * y = 1875) : 
    x = 50 := 
  sorry

end number_of_children_tickets_l173_173020


namespace monotonic_increasing_implies_range_a_l173_173483

-- Definition of the function f(x) = ax^3 - x^2 + x - 5
def f (a x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

-- Derivative of f(x) with respect to x
def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

-- The statement that proves the monotonicity condition implies the range for a
theorem monotonic_increasing_implies_range_a (a : ℝ) : 
  ( ∀ x, f_prime a x ≥ 0 ) → a ≥ (1:ℝ) / 3 := by
  sorry

end monotonic_increasing_implies_range_a_l173_173483


namespace pupils_sent_up_exam_l173_173075

theorem pupils_sent_up_exam (average_marks : ℕ) (specific_scores : List ℕ) (new_average : ℕ) : 
  (average_marks = 39) → 
  (specific_scores = [25, 12, 15, 19]) → 
  (new_average = 44) → 
  ∃ n : ℕ, (n > 4) ∧ (average_marks * n) = 39 * n ∧ ((39 * n - specific_scores.sum) / (n - specific_scores.length)) = new_average →
  n = 21 :=
by
  intros h_avg h_scores h_new_avg
  sorry

end pupils_sent_up_exam_l173_173075


namespace second_polygon_sides_l173_173981

theorem second_polygon_sides (s : ℝ) (P : ℝ) (n : ℕ) : 
  (50 * 3 * s = P) ∧ (n * s = P) → n = 150 := 
by {
  sorry
}

end second_polygon_sides_l173_173981


namespace temperature_difference_l173_173579

def Shanghai_temp : ℤ := 3
def Beijing_temp : ℤ := -5

theorem temperature_difference :
  Shanghai_temp - Beijing_temp = 8 := by
  sorry

end temperature_difference_l173_173579


namespace profit_equation_example_l173_173225

noncomputable def profit_equation (a b : ℝ) (x : ℝ) : Prop :=
  a * (1 + x) ^ 2 = b

theorem profit_equation_example :
  profit_equation 250 360 x :=
by
  have : 25 * (1 + x) ^ 2 = 36 := sorry
  sorry

end profit_equation_example_l173_173225


namespace simplify_expression_l173_173740

theorem simplify_expression (a : ℤ) (h_range : -3 < a ∧ a ≤ 0) (h_notzero : a ≠ 0) (h_notone : a ≠ 1 ∧ a ≠ -1) :
  (a - (2 * a - 1) / a) / (1 / a - a) = -3 :=
by
  have h_eq : (a - (2 * a - 1) / a) / (1 / a - a) = (1 - a) / (1 + a) :=
    sorry
  have h_a_neg_two : a = -2 :=
    sorry
  rw [h_eq, h_a_neg_two]
  sorry


end simplify_expression_l173_173740


namespace sum_of_coefficients_of_poly_is_neg_1_l173_173608

noncomputable def evaluate_poly_sum (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) : ℂ :=
  α^2005 + β^2005

theorem sum_of_coefficients_of_poly_is_neg_1 (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  evaluate_poly_sum α β h1 h2 = -1 := by
  sorry

end sum_of_coefficients_of_poly_is_neg_1_l173_173608


namespace inequality1_inequality2_l173_173517

variables (Γ B P : ℕ)

def convex_polyhedron : Prop :=
  Γ - B + P = 2

theorem inequality1 (h : convex_polyhedron Γ B P) : 
  3 * Γ ≥ 6 + P :=
sorry

theorem inequality2 (h : convex_polyhedron Γ B P) : 
  3 * B ≥ 6 + P :=
sorry

end inequality1_inequality2_l173_173517


namespace sum_of_values_l173_173880

noncomputable def f (x : ℝ) : ℝ :=
if x < 3 then 5 * x + 20 else 3 * x - 21

theorem sum_of_values (h₁ : ∃ x, x < 3 ∧ f x = 4) (h₂ : ∃ x, x ≥ 3 ∧ f x = 4) :
  ∃a b : ℝ, a = -16 / 5 ∧ b = 25 / 3 ∧ (a + b = 77 / 15) :=
by {
  sorry
}

end sum_of_values_l173_173880


namespace find_x_l173_173168

theorem find_x (x : ℕ) (h_odd : x % 2 = 1) (h_pos : 0 < x) :
  (∃ l : List ℕ, l.length = 8 ∧ (∀ n ∈ l, n < 80 ∧ n % 2 = 1) ∧ l.Nodup = true ∧
  (∀ k m, k > 0 → m % 2 = 1 → k * x * m ∈ l)) → x = 5 := by
  sorry

end find_x_l173_173168


namespace zero_positive_integers_prime_polynomial_l173_173162

noncomputable def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem zero_positive_integers_prime_polynomial :
  ∀ (n : ℕ), ¬ is_prime (n^3 - 7 * n^2 + 16 * n - 12) :=
by
  sorry

end zero_positive_integers_prime_polynomial_l173_173162


namespace max_value_y_l173_173328

/-- Given x < 0, the maximum value of y = (1 + x^2) / x is -2 -/
theorem max_value_y {x : ℝ} (h : x < 0) : ∃ y, y = 1 + x^2 / x ∧ y ≤ -2 :=
sorry

end max_value_y_l173_173328


namespace rationalize_sqrt_three_sub_one_l173_173390

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l173_173390


namespace no_positive_integer_pairs_l173_173481

theorem no_positive_integer_pairs (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) : ¬ (x^2 + y^2 = x^3 + 2 * y) :=
by sorry

end no_positive_integer_pairs_l173_173481


namespace mark_bread_time_l173_173937

def rise_time1 : Nat := 120
def rise_time2 : Nat := 120
def kneading_time : Nat := 10
def baking_time : Nat := 30

def total_time : Nat := rise_time1 + rise_time2 + kneading_time + baking_time

theorem mark_bread_time : total_time = 280 := by
  sorry

end mark_bread_time_l173_173937


namespace six_digit_permutation_reverse_div_by_11_l173_173403

theorem six_digit_permutation_reverse_div_by_11 
  (a b c : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 9)
  (h_b : 0 ≤ b ∧ b ≤ 9)
  (h_c : 0 ≤ c ∧ c ≤ 9)
  (X : ℕ)
  (h_X : X = 100001 * a + 10010 * b + 1100 * c) :
  11 ∣ X :=
by 
  sorry

end six_digit_permutation_reverse_div_by_11_l173_173403


namespace ben_has_56_marbles_l173_173872

-- We define the conditions first
variables (B : ℕ) (L : ℕ)

-- Leo has 20 more marbles than Ben
def condition1 : Prop := L = B + 20

-- Total number of marbles is 132
def condition2 : Prop := B + L = 132

-- The goal: proving the number of marbles Ben has is 56
theorem ben_has_56_marbles (h1 : condition1 B L) (h2 : condition2 B L) : B = 56 :=
by sorry

end ben_has_56_marbles_l173_173872


namespace x_equals_eleven_l173_173627

theorem x_equals_eleven (x : ℕ) 
  (h : (1 / 8) * 2^36 = 8^x) : x = 11 :=
sorry

end x_equals_eleven_l173_173627


namespace find_x_l173_173520

theorem find_x (x : ℚ) (h : (3 * x - 7) / 4 = 15) : x = 67 / 3 :=
sorry

end find_x_l173_173520


namespace inequality_any_k_l173_173791

theorem inequality_any_k (x y z : ℝ) (k : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x * y * z = 1) (h5 : 1/x + 1/y + 1/z ≥ x + y + z) : 
  x ^ (-k : ℤ) + y ^ (-k : ℤ) + z ^ (-k : ℤ) ≥ x ^ k + y ^ k + z ^ k :=
sorry

end inequality_any_k_l173_173791


namespace arithmetic_geo_sum_l173_173188

theorem arithmetic_geo_sum (a : ℕ → ℤ) (d : ℤ) :
  (∀ n, a (n + 1) = a n + d) →
  (d = 2) →
  (a 3) ^ 2 = (a 1) * (a 4) →
  (a 2 + a 3 = -10) := 
by
  intros h_arith h_d h_geo
  sorry

end arithmetic_geo_sum_l173_173188


namespace total_seeds_in_garden_l173_173138

-- Definitions based on the conditions
def top_bed_rows : ℕ := 4
def top_bed_seeds_per_row : ℕ := 25
def num_top_beds : ℕ := 2

def medium_bed_rows : ℕ := 3
def medium_bed_seeds_per_row : ℕ := 20
def num_medium_beds : ℕ := 2

-- Calculation of total seeds in top beds
def seeds_per_top_bed : ℕ := top_bed_rows * top_bed_seeds_per_row
def total_seeds_top_beds : ℕ := num_top_beds * seeds_per_top_bed

-- Calculation of total seeds in medium beds
def seeds_per_medium_bed : ℕ := medium_bed_rows * medium_bed_seeds_per_row
def total_seeds_medium_beds : ℕ := num_medium_beds * seeds_per_medium_bed

-- Proof goal
theorem total_seeds_in_garden : total_seeds_top_beds + total_seeds_medium_beds = 320 :=
by
  sorry

end total_seeds_in_garden_l173_173138


namespace michael_scored_times_more_goals_l173_173694

theorem michael_scored_times_more_goals (x : ℕ) (hb : Bruce_goals = 4) (hm : Michael_goals = 4 * x) (ht : Bruce_goals + Michael_goals = 16) : x = 3 := by
  sorry

end michael_scored_times_more_goals_l173_173694


namespace regular_octagon_interior_angle_l173_173131

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : 
  let sum_interior_angles := 180 * (n - 2)
  let num_sides := 8
  let one_interior_angle := sum_interior_angles / num_sides
  one_interior_angle = 135 :=
by
  sorry

end regular_octagon_interior_angle_l173_173131


namespace decomposition_of_x_l173_173366

-- Definitions derived from the conditions
def x : ℝ × ℝ × ℝ := (11, 5, -3)
def p : ℝ × ℝ × ℝ := (1, 0, 2)
def q : ℝ × ℝ × ℝ := (-1, 0, 1)
def r : ℝ × ℝ × ℝ := (2, 5, -3)

-- Theorem statement proving the decomposition
theorem decomposition_of_x : x = (3 : ℝ) • p + (-6 : ℝ) • q + (1 : ℝ) • r := by
  sorry

end decomposition_of_x_l173_173366


namespace car_mpg_in_city_l173_173913

theorem car_mpg_in_city:
  ∃ (h c T : ℝ), 
    (420 = h * T) ∧ 
    (336 = c * T) ∧ 
    (c = h - 6) ∧ 
    (c = 24) :=
by
  sorry

end car_mpg_in_city_l173_173913


namespace round_robin_matches_l173_173386

-- Define the number of players in the tournament
def numPlayers : ℕ := 10

-- Define a function to calculate the number of matches in a round-robin tournament
def calculateMatches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

-- Theorem statement to prove that the number of matches in a 10-person round-robin chess tournament is 45
theorem round_robin_matches : calculateMatches numPlayers = 45 := by
  sorry

end round_robin_matches_l173_173386


namespace smallest_integer_in_ratio_l173_173068

theorem smallest_integer_in_ratio {a b c : ℕ} (h1 : a = 2 * b / 3) (h2 : c = 5 * b / 3) (h3 : a + b + c = 60) : b = 12 := 
  sorry

end smallest_integer_in_ratio_l173_173068


namespace Luke_spent_per_week_l173_173599

-- Definitions based on the conditions
def money_from_mowing := 9
def money_from_weeding := 18
def total_money := money_from_mowing + money_from_weeding
def weeks := 9
def amount_spent_per_week := total_money / weeks

-- The proof statement
theorem Luke_spent_per_week :
  amount_spent_per_week = 3 := 
  sorry

end Luke_spent_per_week_l173_173599


namespace unpainted_cube_count_is_correct_l173_173096

def unit_cube_count : ℕ := 6 * 6 * 6
def opposite_faces_painted_squares : ℕ := 16 * 2
def remaining_faces_painted_squares : ℕ := 9 * 4
def total_painted_squares (overlap_count : ℕ) : ℕ :=
  opposite_faces_painted_squares + remaining_faces_painted_squares - overlap_count
def overlap_count : ℕ := 4 * 2
def painted_cubes : ℕ := total_painted_squares overlap_count
def unpainted_cubes : ℕ := unit_cube_count - painted_cubes

theorem unpainted_cube_count_is_correct : unpainted_cubes = 156 := by
  sorry

end unpainted_cube_count_is_correct_l173_173096


namespace exists_n_for_m_l173_173097

def π (x : ℕ) : ℕ := sorry -- Placeholder for the prime counting function

theorem exists_n_for_m (m : ℕ) (hm : m > 1) : ∃ n : ℕ, n > 1 ∧ n / π n = m :=
by sorry

end exists_n_for_m_l173_173097


namespace bicycle_count_l173_173100

theorem bicycle_count (T : ℕ) (B : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 :=
by {
  sorry
}

end bicycle_count_l173_173100


namespace area_constant_k_l173_173227

theorem area_constant_k (l w d : ℝ) (h_ratio : l / w = 5 / 2) (h_diagonal : d = Real.sqrt (l^2 + w^2)) :
  ∃ k : ℝ, (k = 10 / 29) ∧ (l * w = k * d^2) :=
by
  sorry

end area_constant_k_l173_173227


namespace value_taken_away_l173_173538

theorem value_taken_away (n x : ℕ) (h1 : n = 4) (h2 : 2 * n + 20 = 8 * n - x) : x = 4 :=
by
  sorry

end value_taken_away_l173_173538


namespace binary_digit_sum_property_l173_173777

def binary_digit_sum (n : Nat) : Nat :=
  n.digits 2 |>.foldr (· + ·) 0

theorem binary_digit_sum_property (k : Nat) (h_pos : 0 < k) :
  (Finset.range (2^k)).sum (λ n => binary_digit_sum (n + 1)) = 2^(k - 1) * k + 1 := 
sorry

end binary_digit_sum_property_l173_173777


namespace perimeter_of_smaller_rectangle_l173_173625

theorem perimeter_of_smaller_rectangle (s t u : ℝ) (h1 : 4 * s = 160) (h2 : t = s / 2) (h3 : u = t / 3) : 
    2 * (t + u) = 400 / 3 := by
  sorry

end perimeter_of_smaller_rectangle_l173_173625


namespace chemist_target_temperature_fahrenheit_l173_173996

noncomputable def kelvinToCelsius (K : ℝ) : ℝ := K - 273.15
noncomputable def celsiusToFahrenheit (C : ℝ) : ℝ := (C * 9 / 5) + 32

theorem chemist_target_temperature_fahrenheit :
  celsiusToFahrenheit (kelvinToCelsius (373.15 - 40)) = 140 :=
by
  sorry

end chemist_target_temperature_fahrenheit_l173_173996


namespace total_students_in_class_l173_173780

theorem total_students_in_class (R S : ℕ)
  (h1 : 2 + 12 * 1 + 12 * 2 + 3 * R = S * 2)
  (h2 : S = 2 + 12 + 12 + R) :
  S = 42 :=
by
  sorry

end total_students_in_class_l173_173780


namespace cube_difference_divisibility_l173_173326

-- Given conditions
variables {m n : ℤ} (h1 : m % 2 = 1) (h2 : n % 2 = 1) (k : ℕ)

-- The equivalent statement to be proven
theorem cube_difference_divisibility (h1 : m % 2 = 1) (h2 : n % 2 = 1) : 
  (2^k ∣ m^3 - n^3) ↔ (2^k ∣ m - n) :=
sorry

end cube_difference_divisibility_l173_173326


namespace final_match_l173_173373

-- Definitions of players and conditions
inductive Player
| Antony | Bart | Carl | Damian | Ed | Fred | Glen | Harry

open Player

-- Condition definitions
def beat (p1 p2 : Player) : Prop := sorry

-- Given conditions
axiom Bart_beats_Antony : beat Bart Antony
axiom Carl_beats_Damian : beat Carl Damian
axiom Glen_beats_Harry : beat Glen Harry
axiom Glen_beats_Carl : beat Glen Carl
axiom Carl_beats_Bart : beat Carl Bart
axiom Ed_beats_Fred : beat Ed Fred
axiom Glen_beats_Ed : beat Glen Ed

-- The proof statement
theorem final_match : beat Glen Carl :=
by
  sorry

end final_match_l173_173373


namespace find_positive_real_number_solution_l173_173714

theorem find_positive_real_number_solution (x : ℝ) (h : (x - 5) / 10 = 5 / (x - 10)) (hx : x > 0) : x = 15 :=
sorry

end find_positive_real_number_solution_l173_173714


namespace parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l173_173370

-- Define the first parabola proof problem
theorem parabola_vertex_at_origin_axis_x_passing_point :
  (∃ (m : ℝ), ∀ (x y : ℝ), y^2 = m * x ↔ (y, x) = (0, 0) ∨ (x = 6 ∧ y = -3)) → 
  ∃ m : ℝ, m = 1.5 ∧ (y^2 = m * x) :=
sorry

-- Define the second parabola proof problem
theorem parabola_vertex_at_origin_axis_y_distance_focus :
  (∃ (p : ℝ), ∀ (x y : ℝ), x^2 = 4 * p * y ↔ (y, x) = (0, 0) ∨ (p = 3)) → 
  ∃ q : ℝ, q = 12 ∧ (x^2 = q * y ∨ x^2 = -q * y) :=
sorry

end parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l173_173370


namespace men_absent_l173_173108

theorem men_absent (x : ℕ) (H1 : 10 * 6 = 60) (H2 : (10 - x) * 10 = 60) : x = 4 :=
by
  sorry

end men_absent_l173_173108


namespace minimum_value_of_K_l173_173892

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 1) / Real.exp x

noncomputable def f_K (K x : ℝ) : ℝ :=
  if f x ≤ K then f x else K

theorem minimum_value_of_K :
  (∀ x > 0, f_K (1 / Real.exp 1) x = f x) → (∃ K : ℝ, K = 1 / Real.exp 1) :=
by
  sorry

end minimum_value_of_K_l173_173892


namespace salary_proof_l173_173479

-- Defining the monthly salaries of the officials
def D_Dupon : ℕ := 6000
def D_Duran : ℕ := 8000
def D_Marten : ℕ := 5000

-- Defining the statements made by each official
def Dupon_statement1 : Prop := D_Dupon = 6000
def Dupon_statement2 : Prop := D_Duran = D_Dupon + 2000
def Dupon_statement3 : Prop := D_Marten = D_Dupon - 1000

def Duran_statement1 : Prop := D_Duran > D_Marten
def Duran_statement2 : Prop := D_Duran - D_Marten = 3000
def Duran_statement3 : Prop := D_Marten = 9000

def Marten_statement1 : Prop := D_Marten < D_Dupon
def Marten_statement2 : Prop := D_Dupon = 7000
def Marten_statement3 : Prop := D_Duran = D_Dupon + 3000

-- Defining the constraints about the number of truth and lies
def Told_the_truth_twice_and_lied_once : Prop :=
  (Dupon_statement1 ∧ Dupon_statement2 ∧ ¬Dupon_statement3) ∨
  (Dupon_statement1 ∧ ¬Dupon_statement2 ∧ Dupon_statement3) ∨
  (¬Dupon_statement1 ∧ Dupon_statement2 ∧ Dupon_statement3) ∨
  (Duran_statement1 ∧ Duran_statement2 ∧ ¬Duran_statement3) ∨
  (Duran_statement1 ∧ ¬Duran_statement2 ∧ Duran_statement3) ∨
  (¬Duran_statement1 ∧ Duran_statement2 ∧ Duran_statement3) ∨
  (Marten_statement1 ∧ Marten_statement2 ∧ ¬Marten_statement3) ∨
  (Marten_statement1 ∧ ¬Marten_statement2 ∧ Marten_statement3) ∨
  (¬Marten_statement1 ∧ Marten_statement2 ∧ Marten_statement3)

-- The final proof goal
theorem salary_proof : Told_the_truth_twice_and_lied_once →
  D_Dupon = 6000 ∧ D_Duran = 8000 ∧ D_Marten = 5000 := by 
  sorry

end salary_proof_l173_173479


namespace min_value_of_xy_cond_l173_173883

noncomputable def minValueOfXY (x y : ℝ) : ℝ :=
  if 2 * Real.cos (x + y - 1) ^ 2 = ((x + 1) ^ 2 + (y - 1) ^ 2 - 2 * x * y) / (x - y + 1) then 
    x * y
  else 
    0

theorem min_value_of_xy_cond (x y : ℝ) 
  (h : 2 * Real.cos (x + y - 1) ^ 2 = ((x + 1) ^ 2 + (y - 1) ^ 2 - 2 * x * y) / (x - y + 1)) : 
  (∃ k : ℤ, x = (k * Real.pi + 1) / 2 ∧ y = (k * Real.pi + 1) / 2) → 
  x * y = 1/4 := 
by
  -- The proof is omitted.
  sorry

end min_value_of_xy_cond_l173_173883


namespace shaded_region_area_l173_173763

noncomputable def side_length := 1 -- Length of each side of the squares, in cm.

-- Conditions
def top_square_center_above_edge : Prop := 
  ∀ square1 square2 square3 : ℝ, square3 = (square1 + square2) / 2

-- Question: Area of the shaded region
def area_of_shaded_region := 1 -- area in cm^2

-- Lean 4 Statement
theorem shaded_region_area :
  top_square_center_above_edge → area_of_shaded_region = 1 := 
by
  sorry

end shaded_region_area_l173_173763


namespace true_proposition_l173_173330

-- Definitions of propositions
def p := ∃ (x : ℝ), x - x + 1 ≥ 0
def q := ∀ (a b : ℝ), a^2 < b^2 → a < b

-- Theorem statement
theorem true_proposition : p ∧ ¬q :=
by
  sorry

end true_proposition_l173_173330


namespace players_taking_all_three_subjects_l173_173715

-- Define the variables for the number of players in each category
def num_players : ℕ := 18
def num_physics : ℕ := 10
def num_biology : ℕ := 7
def num_chemistry : ℕ := 5
def num_physics_biology : ℕ := 3
def num_biology_chemistry : ℕ := 2
def num_physics_chemistry : ℕ := 1

-- Define the proposition we want to prove
theorem players_taking_all_three_subjects :
  ∃ x : ℕ, x = 2 ∧
  num_players = num_physics + num_biology + num_chemistry
                - num_physics_chemistry
                - num_physics_biology
                - num_biology_chemistry
                + x :=
by {
  sorry -- Placeholder for the proof
}

end players_taking_all_three_subjects_l173_173715


namespace chef_dressing_total_volume_l173_173238

theorem chef_dressing_total_volume :
  ∀ (V1 V2 : ℕ) (P1 P2 : ℕ) (total_amount : ℕ),
    V1 = 128 →
    V2 = 128 →
    P1 = 8 →
    P2 = 13 →
    total_amount = V1 + V2 →
    total_amount = 256 :=
by
  intros V1 V2 P1 P2 total_amount hV1 hV2 hP1 hP2 h_total
  rw [hV1, hV2, add_comm, add_comm] at h_total
  exact h_total

end chef_dressing_total_volume_l173_173238


namespace sum_of_first_n_odd_integers_eq_169_l173_173332

theorem sum_of_first_n_odd_integers_eq_169 (n : ℕ) 
  (h : n^2 = 169) : n = 13 :=
by sorry

end sum_of_first_n_odd_integers_eq_169_l173_173332


namespace pencil_and_pen_choice_count_l173_173170

-- Definitions based on the given conditions
def numPencilTypes : Nat := 4
def numPenTypes : Nat := 6

-- Statement we want to prove
theorem pencil_and_pen_choice_count : (numPencilTypes * numPenTypes) = 24 :=
by
  sorry

end pencil_and_pen_choice_count_l173_173170


namespace base3_sum_l173_173207

theorem base3_sum : 
  (1 * 3^0 - 2 * 3^1 - 2 * 3^0 + 2 * 3^2 + 1 * 3^1 - 1 * 3^0 - 1 * 3^3) = (2 * 3^2 + 1 * 3^1 + 0 * 3^0) := 
by 
  sorry

end base3_sum_l173_173207


namespace paving_stone_length_l173_173833

theorem paving_stone_length
  (length_courtyard : ℝ)
  (width_courtyard : ℝ)
  (num_paving_stones : ℝ)
  (width_paving_stone : ℝ)
  (total_area : ℝ := length_courtyard * width_courtyard)
  (area_per_paving_stone : ℝ := (total_area / num_paving_stones))
  (length_paving_stone : ℝ := (area_per_paving_stone / width_paving_stone)) :
  length_courtyard = 20 ∧
  width_courtyard = 16.5 ∧
  num_paving_stones = 66 ∧
  width_paving_stone = 2 →
  length_paving_stone = 2.5 :=
by {
   sorry
}

end paving_stone_length_l173_173833


namespace power_of_128_div_7_eq_16_l173_173857

theorem power_of_128_div_7_eq_16 : (128 : ℝ) ^ (4 / 7) = 16 := by
  sorry

end power_of_128_div_7_eq_16_l173_173857


namespace coordinate_identification_l173_173693

noncomputable def x1 := (4 * Real.pi) / 5
noncomputable def y1 := -(Real.pi) / 5

noncomputable def x2 := (12 * Real.pi) / 5
noncomputable def y2 := -(3 * Real.pi) / 5

noncomputable def x3 := (4 * Real.pi) / 3
noncomputable def y3 := -(Real.pi) / 3

theorem coordinate_identification :
  (x1, y1) = (4 * Real.pi / 5, -(Real.pi) / 5) ∧
  (x2, y2) = (12 * Real.pi / 5, -(3 * Real.pi) / 5) ∧
  (x3, y3) = (4 * Real.pi / 3, -(Real.pi) / 3) :=
by
  -- proof goes here
  sorry

end coordinate_identification_l173_173693


namespace file_size_correct_l173_173909

theorem file_size_correct:
  (∀ t1 t2 : ℕ, (60 / 5 = t1) ∧ (15 - t1 = t2) ∧ (t2 * 10 = 30) → (60 + 30 = 90)) := 
by
  sorry

end file_size_correct_l173_173909


namespace cricket_run_rate_l173_173748

theorem cricket_run_rate (r : ℝ) (o₁ T o₂ : ℕ) (r₁ : ℝ) (Rₜ : ℝ) : 
  r = 4.8 ∧ o₁ = 10 ∧ T = 282 ∧ o₂ = 40 ∧ r₁ = (T - r * o₁) / o₂ → Rₜ = 5.85 := 
by 
  intros h
  sorry

end cricket_run_rate_l173_173748


namespace calculate_exponent_l173_173046

theorem calculate_exponent (m : ℝ) : (243 : ℝ)^(1 / 3) = 3^m → m = 5 / 3 :=
by
  sorry

end calculate_exponent_l173_173046


namespace value_of_a3_l173_173142

def a_n (n : ℕ) : ℤ := (-1)^n * (n^2 + 1)

theorem value_of_a3 : a_n 3 = -10 :=
by
  -- The proof would go here.
  sorry

end value_of_a3_l173_173142


namespace rate_of_interest_l173_173354

-- Given conditions
def P : ℝ := 1500
def SI : ℝ := 735
def r : ℝ := 7
def t := r  -- The time period in years is equal to the rate of interest

-- The formula for simple interest and the goal
theorem rate_of_interest : SI = P * r * t / 100 ↔ r = 7 := 
by
  -- We will use the given conditions and check if they support r = 7
  sorry

end rate_of_interest_l173_173354


namespace value_of_F_l173_173396

theorem value_of_F (D E F : ℕ) (hD : D < 10) (hE : E < 10) (hF : F < 10)
    (h1 : (8 + 5 + D + 7 + 3 + E + 2) % 3 = 0)
    (h2 : (4 + 1 + 7 + D + E + 6 + F) % 3 = 0) : 
    F = 6 :=
by
  sorry

end value_of_F_l173_173396


namespace smallest_c1_in_arithmetic_sequence_l173_173003

theorem smallest_c1_in_arithmetic_sequence (S3 S7 : ℕ) (S3_natural : S3 > 0) (S7_natural : S7 > 0)
    (c1_geq_one_third : ∀ d : ℚ, (c1 : ℚ) = (7*S3 - S7) / 14 → c1 ≥ 1/3) : 
    ∃ c1 : ℚ, c1 = 5/14 ∧ c1 ≥ 1/3 := 
by 
  sorry

end smallest_c1_in_arithmetic_sequence_l173_173003


namespace find_fifth_score_l173_173919

-- Define the known scores
def score1 : ℕ := 90
def score2 : ℕ := 93
def score3 : ℕ := 85
def score4 : ℕ := 97

-- Define the average of all scores
def average : ℕ := 92

-- Define the total number of scores
def total_scores : ℕ := 5

-- Define the total sum of all scores using the average
def total_sum : ℕ := total_scores * average

-- Define the sum of the four known scores
def known_sum : ℕ := score1 + score2 + score3 + score4

-- Define the fifth score
def fifth_score : ℕ := 95

-- Theorem statement: The fifth score plus the known sum equals the total sum.
theorem find_fifth_score : fifth_score + known_sum = total_sum := by
  sorry

end find_fifth_score_l173_173919


namespace reflection_twice_is_identity_l173_173469

-- Define the reflection matrix R over the vector (1, 2)
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  -- Note: The specific definition of the reflection matrix over (1, 2) is skipped as we only need the final proof statement.
  sorry

-- Assign the reflection matrix R to variable R
def R := reflection_matrix

-- Prove that R^2 = I
theorem reflection_twice_is_identity : R * R = 1 := by
  sorry

end reflection_twice_is_identity_l173_173469


namespace televisions_selection_ways_l173_173344

noncomputable def combination (n k : ℕ) : ℕ :=
if h : k ≤ n then Nat.choose n k else 0

theorem televisions_selection_ways :
  let TypeA := 4
  let TypeB := 5
  let choosen := 3
  (∃ (n m : ℕ), n + m = choosen ∧ 1 ≤ n ∧ n ≤ TypeA ∧ 1 ≤ m ∧ m ≤ TypeB ∧
    combination TypeA n * combination TypeB m = 70) :=
by
  sorry

end televisions_selection_ways_l173_173344


namespace molecular_weight_8_moles_Al2O3_l173_173723

noncomputable def molecular_weight_Al2O3 (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) : ℝ :=
  2 * atomic_weight_Al + 3 * atomic_weight_O

theorem molecular_weight_8_moles_Al2O3
  (atomic_weight_Al : ℝ := 26.98)
  (atomic_weight_O : ℝ := 16.00)
  : molecular_weight_Al2O3 atomic_weight_Al atomic_weight_O * 8 = 815.68 := by
  sorry

end molecular_weight_8_moles_Al2O3_l173_173723


namespace maximum_p_l173_173061

noncomputable def p (a b c : ℝ) : ℝ :=
  (2 / (a ^ 2 + 1)) - (2 / (b ^ 2 + 1)) + (3 / (c ^ 2 + 1))

theorem maximum_p (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : abc + a + c = b) : 
  p a b c ≤ 10 / 3 ∧ ∃ a b c, abc + a + c = b ∧ p a b c = 10 / 3 :=
sorry

end maximum_p_l173_173061


namespace quotient_is_four_l173_173917

theorem quotient_is_four (dividend : ℕ) (k : ℕ) (h1 : dividend = 16) (h2 : k = 4) : dividend / k = 4 :=
by
  sorry

end quotient_is_four_l173_173917


namespace evaluate_expression_l173_173773

theorem evaluate_expression :
  (42 / (9 - 3 * 2)) * 4 = 56 :=
by
  sorry

end evaluate_expression_l173_173773


namespace trigonometric_identity_l173_173930

theorem trigonometric_identity (α : Real) (h : Real.tan (α / 2) = 4) :
    (6 * Real.sin α - 7 * Real.cos α + 1) / (8 * Real.sin α + 9 * Real.cos α - 1) = -85 / 44 := by
  sorry

end trigonometric_identity_l173_173930


namespace kopecks_to_rubles_l173_173720

noncomputable def exchangeable_using_coins (total : ℕ) (num_coins : ℕ) : Prop :=
  ∃ (x y z t u v w : ℕ), 
    total = x * 1 + y * 2 + z * 5 + t * 10 + u * 20 + v * 50 + w * 100 ∧ 
    num_coins = x + y + z + t + u + v + w

theorem kopecks_to_rubles (A B : ℕ)
  (h : exchangeable_using_coins A B) : exchangeable_using_coins (100 * B) A :=
sorry

end kopecks_to_rubles_l173_173720


namespace cakesServedDuringDinner_today_is_6_l173_173021

def cakesServedDuringDinner (x : ℕ) : Prop :=
  5 + x + 3 = 14

theorem cakesServedDuringDinner_today_is_6 : cakesServedDuringDinner 6 :=
by
  unfold cakesServedDuringDinner
  -- The proof is omitted
  sorry

end cakesServedDuringDinner_today_is_6_l173_173021


namespace triangle_side_length_l173_173288

theorem triangle_side_length 
  (X Z : ℝ) (x z y : ℝ)
  (h1 : x = 36)
  (h2 : z = 72)
  (h3 : Z = 4 * X) :
  y = 72 := by
  sorry

end triangle_side_length_l173_173288


namespace selling_price_of_book_l173_173400

theorem selling_price_of_book (cost_price : ℕ) (profit_rate : ℕ) (profit : ℕ) (selling_price : ℕ) :
  cost_price = 50 → profit_rate = 80 → profit = (profit_rate * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 90 :=
by
  intros h_cost_price h_profit_rate h_profit h_selling_price
  rw [h_cost_price, h_profit_rate] at h_profit
  simp at h_profit
  rw [h_cost_price, h_profit] at h_selling_price
  exact h_selling_price

end selling_price_of_book_l173_173400


namespace smallest_possible_intersections_l173_173914

theorem smallest_possible_intersections (n : ℕ) (hn : n = 2000) :
  ∃ N : ℕ, N ≥ 3997 :=
by
  sorry

end smallest_possible_intersections_l173_173914


namespace unique_10_tuple_solution_l173_173946

noncomputable def condition (x : Fin 10 → ℝ) : Prop :=
  (1 - x 0)^2 +
  (x 0 - x 1)^2 + 
  (x 1 - x 2)^2 + 
  (x 2 - x 3)^2 + 
  (x 3 - x 4)^2 + 
  (x 4 - x 5)^2 + 
  (x 5 - x 6)^2 + 
  (x 6 - x 7)^2 + 
  (x 7 - x 8)^2 + 
  (x 8 - x 9)^2 + 
  x 9^2 + 
  (1/2) * (x 9 - x 0)^2 = 1/10

theorem unique_10_tuple_solution : 
  ∃! (x : Fin 10 → ℝ), condition x := 
sorry

end unique_10_tuple_solution_l173_173946


namespace weight_loss_total_l173_173596

theorem weight_loss_total :
  ∀ (weight1 weight2 weight3 weight4 : ℕ),
    weight1 = 27 →
    weight2 = weight1 - 7 →
    weight3 = 28 →
    weight4 = 28 →
    weight1 + weight2 + weight3 + weight4 = 103 :=
by
  intros weight1 weight2 weight3 weight4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end weight_loss_total_l173_173596


namespace tokens_per_pitch_l173_173461

theorem tokens_per_pitch 
  (tokens_macy : ℕ) (tokens_piper : ℕ)
  (hits_macy : ℕ) (hits_piper : ℕ)
  (misses_total : ℕ) (p : ℕ)
  (h1 : tokens_macy = 11)
  (h2 : tokens_piper = 17)
  (h3 : hits_macy = 50)
  (h4 : hits_piper = 55)
  (h5 : misses_total = 315)
  (h6 : 28 * p = hits_macy + hits_piper + misses_total) :
  p = 15 := 
by 
  sorry

end tokens_per_pitch_l173_173461


namespace boys_bound_l173_173544

open Nat

noncomputable def num_students := 1650
noncomputable def num_rows := 22
noncomputable def num_cols := 75
noncomputable def max_pairs_same_sex := 11

-- Assume we have a function that gives the number of boys.
axiom number_of_boys : ℕ
axiom col_pairs_property : ∀ (c1 c2 : ℕ), ∀ (r : ℕ), c1 ≠ c2 → r ≤ num_rows → 
  (number_of_boys ≤ max_pairs_same_sex)

theorem boys_bound : number_of_boys ≤ 920 :=
sorry

end boys_bound_l173_173544


namespace final_price_is_99_l173_173576

-- Conditions:
def original_price : ℝ := 120
def coupon_discount : ℝ := 10
def membership_discount_rate : ℝ := 0.10

-- Define final price calculation
def final_price (original_price coupon_discount membership_discount_rate : ℝ) : ℝ :=
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  price_after_coupon - membership_discount

-- Question: Is the final price equal to $99?
theorem final_price_is_99 :
  final_price original_price coupon_discount membership_discount_rate = 99 :=
by
  sorry

end final_price_is_99_l173_173576


namespace green_face_probability_l173_173994

def probability_of_green_face (total_faces green_faces : Nat) : ℚ :=
  green_faces / total_faces

theorem green_face_probability :
  let total_faces := 10
  let green_faces := 3
  let blue_faces := 5
  let red_faces := 2
  probability_of_green_face total_faces green_faces = 3/10 :=
by
  sorry

end green_face_probability_l173_173994


namespace compute_expression_l173_173514

-- Define the conditions
variables (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0)

-- State the theorem to be proved
theorem compute_expression (ω : ℂ) (hω_nonreal : ω^3 = 1) (hω_eq : ω^2 + ω + 1 = 0) :
  (1 - ω + ω^2)^6 + (1 + ω - ω^2)^6 = 128 := 
sorry

end compute_expression_l173_173514


namespace carl_additional_marbles_l173_173281

def initial_marbles := 12
def lost_marbles := initial_marbles / 2
def additional_marbles_from_mom := 25
def marbles_in_jar_after_game := 41

theorem carl_additional_marbles :
  (marbles_in_jar_after_game - additional_marbles_from_mom) + lost_marbles - initial_marbles = 10 :=
by
  sorry

end carl_additional_marbles_l173_173281


namespace total_red_marbles_l173_173472

-- Definitions derived from the problem conditions
def Jessica_red_marbles : ℕ := 3 * 12
def Sandy_red_marbles : ℕ := 4 * Jessica_red_marbles
def Alex_red_marbles : ℕ := Jessica_red_marbles + 2 * 12

-- Statement we need to prove that total number of marbles is 240
theorem total_red_marbles : 
  Jessica_red_marbles + Sandy_red_marbles + Alex_red_marbles = 240 := by
  -- We provide the proof later
  sorry

end total_red_marbles_l173_173472


namespace Fedya_third_l173_173798

/-- Definitions for order of children's arrival -/
inductive Child
| Roman | Fedya | Liza | Katya | Andrew

open Child

def arrival_order (order : Child → ℕ) : Prop :=
  order Liza > order Roman ∧
  order Katya < order Liza ∧
  order Fedya = order Katya + 1 ∧
  order Katya ≠ 1

/-- Theorem stating that Fedya is third based on the given conditions -/
theorem Fedya_third (order : Child → ℕ) (H : arrival_order order) : order Fedya = 3 :=
sorry

end Fedya_third_l173_173798


namespace growth_rate_inequality_l173_173921

theorem growth_rate_inequality (a b x : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_x_pos : x > 0) :
  x ≤ (a + b) / 2 :=
sorry

end growth_rate_inequality_l173_173921


namespace find_minimum_value_M_l173_173696

theorem find_minimum_value_M : (∃ (M : ℝ), (∀ (x : ℝ), -x^2 + 2 * x ≤ M) ∧ M = 1) := 
sorry

end find_minimum_value_M_l173_173696


namespace product_of_digits_base8_of_12345_is_0_l173_173805

def base8_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else Nat.digits 8 n 

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_digits_base8_of_12345_is_0 :
  product_of_digits (base8_representation 12345) = 0 := 
sorry

end product_of_digits_base8_of_12345_is_0_l173_173805


namespace new_area_of_rectangle_l173_173244

theorem new_area_of_rectangle (L W : ℝ) (h : L * W = 600) :
  let new_length := 0.8 * L
  let new_width := 1.05 * W
  new_length * new_width = 504 :=
by 
  sorry

end new_area_of_rectangle_l173_173244


namespace equation_one_solution_equation_two_solution_l173_173555

theorem equation_one_solution (x : ℝ) (h : 2 * (2 - x) - 5 * (2 - x) = 9) : x = 5 :=
sorry

theorem equation_two_solution (x : ℝ) (h : x / 3 - (3 * x - 1) / 6 = 1) : x = -5 :=
sorry

end equation_one_solution_equation_two_solution_l173_173555


namespace jennifer_fruits_left_l173_173969

open Nat

theorem jennifer_fruits_left :
  (p o a g : ℕ) → p = 10 → o = 20 → a = 2 * p → g = 2 → (p - g) + (o - g) + (a - g) = 44 :=
by
  intros p o a g h_p h_o h_a h_g
  rw [h_p, h_o, h_a, h_g]
  sorry

end jennifer_fruits_left_l173_173969


namespace complex_fraction_evaluation_l173_173868

theorem complex_fraction_evaluation :
  ( 
    ((3 + 1/3) / 10 + 0.175 / 0.35) / 
    (1.75 - (1 + 11/17) * (51/56)) - 
    ((11/18 - 1/15) / 1.4) / 
    ((0.5 - 1/9) * 3)
  ) = 1/2 := 
sorry

end complex_fraction_evaluation_l173_173868


namespace julios_grape_soda_l173_173794

variable (a b c d e f g : ℕ)
variable (ha : a = 4)
variable (hc : c = 1)
variable (hd : d = 3)
variable (he : e = 2)
variable (hf : f = 14)
variable (hg : g = 7)

theorem julios_grape_soda : 
  let julios_soda := a * e + b * e
  let mateos_soda := (c + d) * e
  julios_soda = mateos_soda + f
  → b = g := by
  sorry

end julios_grape_soda_l173_173794


namespace find_slope_l173_173583

noncomputable def parabola_equation (x y : ℝ) := y^2 = 8 * x

def point_M : ℝ × ℝ := (-2, 2)

def line_through_focus (k x : ℝ) : ℝ := k * (x - 2)

def focus : ℝ × ℝ := (2, 0)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem find_slope (k : ℝ) : 
  (∀ x y A B, 
    parabola_equation x y → 
    (x = A ∨ x = B) → 
    line_through_focus k x = y → 
    parabola_equation A (k * (A - 2)) → 
    parabola_equation B (k * (B - 2)) → 
    dot_product (A + 2, (k * (A -2)) - 2) (B + 2, (k * (B - 2)) - 2) = 0) →
  k = 2 :=
sorry

end find_slope_l173_173583


namespace ratio_of_areas_l173_173558

theorem ratio_of_areas (AB BC O : ℝ) (h_diameter : AB = 4) (h_BC : BC = 3)
  (ABD DBE ABDeqDBE : Prop) (x y : ℝ) 
  (h_area_ABCD : x = 7 * y) :
  (x / y) = 7 :=
by
  sorry

end ratio_of_areas_l173_173558


namespace lines_of_first_character_l173_173376

-- Definitions for the number of lines each character has
def L3 : Nat := 2

def L2 : Nat := 3 * L3 + 6

def L1 : Nat := L2 + 8

-- The theorem we are proving
theorem lines_of_first_character : L1 = 20 :=
by
  -- The proof would go here
  sorry

end lines_of_first_character_l173_173376


namespace problem_l173_173133

theorem problem (p q r : ℂ)
  (h1 : p + q + r = 0)
  (h2 : p * q + q * r + r * p = -2)
  (h3 : p * q * r = 2)
  (hp : p ^ 3 = 2 * p + 2)
  (hq : q ^ 3 = 2 * q + 2)
  (hr : r ^ 3 = 2 * r + 2) :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = -18 := by
  sorry

end problem_l173_173133


namespace solve_inequality_find_m_range_l173_173431

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem solve_inequality (a : ℝ) : 
  ∀ x : ℝ, f x + a - 1 > 0 ↔ 
    (a = 1 ∧ x ≠ 2) ∨ 
    (a > 1) ∨ 
    (a < 1 ∧ (x > 3 - a ∨ x < a + 1)) :=
sorry

theorem find_m_range (m : ℝ) : 
  (∀ x : ℝ, f x > g x m) ↔ m < 5 :=
sorry

end solve_inequality_find_m_range_l173_173431


namespace apples_jackie_l173_173822

theorem apples_jackie (A : ℕ) (J : ℕ) (h1 : A = 8) (h2 : J = A + 2) : J = 10 := by
  -- Adam has 8 apples
  sorry

end apples_jackie_l173_173822


namespace rotate_D_90_clockwise_l173_173759

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℤ) : Point ℤ :=
  ⟨p.y, -p.x⟩

def D : Point ℤ := ⟨-3, 2⟩
def E : Point ℤ := ⟨0, 5⟩
def F : Point ℤ := ⟨0, 2⟩

theorem rotate_D_90_clockwise :
  rotate_90_clockwise D = Point.mk 2 (-3) :=
by
  sorry

end rotate_D_90_clockwise_l173_173759


namespace selling_price_correct_l173_173725

noncomputable def cost_price : ℝ := 90.91

noncomputable def profit_rate : ℝ := 0.10

noncomputable def profit : ℝ := profit_rate * cost_price

noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 100.00 := by
  sorry

end selling_price_correct_l173_173725


namespace extreme_point_of_f_l173_173053

open Real

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - log x

theorem extreme_point_of_f : ∃ x₀ > 0, f x₀ = f (sqrt 3 / 3) ∧ 
  (∀ x < sqrt 3 / 3, f x > f (sqrt 3 / 3)) ∧
  (∀ x > sqrt 3 / 3, f x > f (sqrt 3 / 3)) :=
sorry

end extreme_point_of_f_l173_173053


namespace eighteenth_entry_l173_173429

def r_8 (n : ℕ) : ℕ := n % 8

theorem eighteenth_entry (n : ℕ) (h : r_8 (3 * n) ≤ 3) : n = 17 :=
sorry

end eighteenth_entry_l173_173429


namespace trigonometric_identity_l173_173649

noncomputable def point_on_terminal_side (x y : ℝ) : Prop :=
    ∃ α : ℝ, x = Real.cos α ∧ y = Real.sin α

theorem trigonometric_identity (x y : ℝ) (h : point_on_terminal_side 1 3) :
    (Real.sin (π - α) - Real.sin (π / 2 + α)) / (2 * Real.cos (α - 2 * π)) = 1 :=
by
  sorry

end trigonometric_identity_l173_173649


namespace B_and_C_together_l173_173932

-- Defining the variables and conditions
variable (A B C : ℕ)
variable (h1 : A + B + C = 500)
variable (h2 : A + C = 200)
variable (h3 : C = 50)

-- The theorem to prove that B + C = 350
theorem B_and_C_together : B + C = 350 :=
by
  -- Replacing with the actual proof steps
  sorry

end B_and_C_together_l173_173932


namespace find_b_c_d_l173_173735

def f (x : ℝ) := x^3 + 2 * x^2 + 3 * x + 4
def h (x : ℝ) := x^3 + 6 * x^2 - 8 * x + 16

theorem find_b_c_d :
  (∀ r : ℝ, f r = 0 → h (r^3) = 0) ∧ h (x : ℝ) = x^3 + 6 * x^2 - 8 * x + 16 :=
by 
  -- proof not required
  sorry

end find_b_c_d_l173_173735


namespace hyperbola_m_value_l173_173094

theorem hyperbola_m_value
  (m : ℝ)
  (h1 : 3 * m * x^2 - m * y^2 = 3)
  (focus : ∃ c, (0, c) = (0, 2)) :
  m = -1 :=
sorry

end hyperbola_m_value_l173_173094


namespace polynomial_coeff_sum_neg_33_l173_173776

theorem polynomial_coeff_sum_neg_33
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (2 - 3 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -33 :=
by sorry

end polynomial_coeff_sum_neg_33_l173_173776


namespace sandy_hourly_wage_l173_173035

theorem sandy_hourly_wage (x : ℝ)
    (h1 : 10 * x + 6 * x + 14 * x = 450) : x = 15 :=
by
    sorry

end sandy_hourly_wage_l173_173035


namespace crushing_load_example_l173_173519

noncomputable def crushing_load (T H : ℝ) : ℝ :=
  (30 * T^5) / H^3

theorem crushing_load_example : crushing_load 5 10 = 93.75 := by
  sorry

end crushing_load_example_l173_173519


namespace box_length_is_24_l173_173803

theorem box_length_is_24 (L : ℕ) (h1 : ∀ s : ℕ, (L * 40 * 16 = 30 * s^3) → s ∣ 40 ∧ s ∣ 16) (h2 : ∃ s : ℕ, s ∣ 40 ∧ s ∣ 16) : L = 24 :=
by
  sorry

end box_length_is_24_l173_173803


namespace total_digits_in_numbering_pages_l173_173255

theorem total_digits_in_numbering_pages (n : ℕ) (h : n = 100000) : 
  let digits1 := 9 * 1
  let digits2 := (99 - 10 + 1) * 2
  let digits3 := (999 - 100 + 1) * 3
  let digits4 := (9999 - 1000 + 1) * 4
  let digits5 := (99999 - 10000 + 1) * 5
  let digits6 := 6
  (digits1 + digits2 + digits3 + digits4 + digits5 + digits6) = 488895 :=
by
  sorry

end total_digits_in_numbering_pages_l173_173255


namespace winning_vote_majority_l173_173775

theorem winning_vote_majority (h1 : 0.70 * 900 = 630)
                             (h2 : 0.30 * 900 = 270) :
  630 - 270 = 360 :=
by
  sorry

end winning_vote_majority_l173_173775


namespace sasha_study_more_l173_173086

theorem sasha_study_more (d_wkdy : List ℤ) (d_wknd : List ℤ) (h_wkdy : d_wkdy = [5, -5, 15, 25, -15]) (h_wknd : d_wknd = [30, 30]) :
  (d_wkdy.sum + d_wknd.sum) / 7 = 12 := by
  sorry

end sasha_study_more_l173_173086


namespace company_b_profit_l173_173870

-- Definitions as per problem conditions
def A_profit : ℝ := 90000
def A_share : ℝ := 0.60
def B_share : ℝ := 0.40

-- Theorem statement to be proved
theorem company_b_profit : B_share * (A_profit / A_share) = 60000 :=
by
  sorry

end company_b_profit_l173_173870


namespace number_division_l173_173051

theorem number_division (x : ℚ) (h : x / 2 = 100 + x / 5) : x = 1000 / 3 := 
by
  sorry

end number_division_l173_173051


namespace ticket_representation_l173_173602

-- Define a structure for representing a movie ticket
structure Ticket where
  rows : Nat
  seats : Nat

-- Define the specific instance of representing 7 rows and 5 seats
def ticket_7_5 : Ticket := ⟨7, 5⟩

-- The theorem stating our problem: the representation of 7 rows and 5 seats is (7,5)
theorem ticket_representation : ticket_7_5 = ⟨7, 5⟩ :=
  by
    -- Proof goes here (omitted as per instructions)
    sorry

end ticket_representation_l173_173602


namespace problem_solution_l173_173973

variables {m n : ℝ}

theorem problem_solution (h1 : m^2 - n^2 = m * n) (h2 : m ≠ 0) (h3 : n ≠ 0) :
  (n / m) - (m / n) = -1 :=
sorry

end problem_solution_l173_173973


namespace cost_per_kg_mixture_l173_173509

variables (C1 C2 R Cm : ℝ)

-- Statement of the proof problem
theorem cost_per_kg_mixture :
  C1 = 6 → C2 = 8.75 → R = 5 / 6 → Cm = C1 * R + C2 * (1 - R) → Cm = 6.458333333333333 :=
by intros hC1 hC2 hR hCm; sorry

end cost_per_kg_mixture_l173_173509


namespace bob_height_in_inches_l173_173440

theorem bob_height_in_inches (tree_height shadow_tree bob_shadow : ℝ)
  (h1 : tree_height = 50)
  (h2 : shadow_tree = 25)
  (h3 : bob_shadow = 6) :
  (12 * (tree_height / shadow_tree) * bob_shadow) = 144 :=
by sorry

end bob_height_in_inches_l173_173440


namespace average_test_score_45_percent_l173_173813

theorem average_test_score_45_percent (x : ℝ) 
  (h1 : 0.45 * x + 0.50 * 78 + 0.05 * 60 = 84.75) : 
  x = 95 :=
by sorry

end average_test_score_45_percent_l173_173813


namespace function_properties_and_k_range_l173_173413

theorem function_properties_and_k_range :
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 ^ x) ∧ (∀ y, y > 0)) ∧
  (∀ k : ℝ, (∃ t : ℝ, t > 0 ∧ (t^2 - 2*t + k = 0)) ↔ (0 < k ∧ k < 1)) :=
by sorry

end function_properties_and_k_range_l173_173413


namespace total_weight_of_10_moles_CaH2_is_420_96_l173_173581

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008
def molecular_weight_CaH2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_H
def moles_CaH2 : ℝ := 10
def total_weight_CaH2 : ℝ := molecular_weight_CaH2 * moles_CaH2

theorem total_weight_of_10_moles_CaH2_is_420_96 :
  total_weight_CaH2 = 420.96 :=
by
  sorry

end total_weight_of_10_moles_CaH2_is_420_96_l173_173581


namespace increase_in_sets_when_profit_38_price_reduction_for_1200_profit_l173_173106

-- Definitions for conditions
def original_profit_per_set := 40
def original_sets_sold_per_day := 20
def additional_sets_per_dollar_drop := 2

-- The proof problems

-- Part 1: Prove the increase in sets when profit reduces to $38
theorem increase_in_sets_when_profit_38 :
  let decrease_in_profit := (original_profit_per_set - 38)
  additional_sets_per_dollar_drop * decrease_in_profit = 4 :=
by
  sorry

-- Part 2: Prove the price reduction needed for $1200 daily profit
theorem price_reduction_for_1200_profit :
  ∃ x, (original_profit_per_set - x) * (original_sets_sold_per_day + 2 * x) = 1200 ∧ x = 20 :=
by
  sorry

end increase_in_sets_when_profit_38_price_reduction_for_1200_profit_l173_173106


namespace inequality_solution_set_l173_173325

theorem inequality_solution_set (x : ℝ) :
  (3 * x - 1) / (2 - x) ≥ 1 ↔ (3 / 4 ≤ x ∧ x < 2) :=
by sorry

end inequality_solution_set_l173_173325


namespace complex_numbers_xyz_l173_173104

theorem complex_numbers_xyz (x y z : ℂ) (h1 : x * y + 5 * y = -20) (h2 : y * z + 5 * z = -20) (h3 : z * x + 5 * x = -20) :
  x * y * z = 100 :=
sorry

end complex_numbers_xyz_l173_173104


namespace x_intercept_of_perpendicular_line_l173_173353

theorem x_intercept_of_perpendicular_line 
  (a : ℝ)
  (l1 : ℝ → ℝ → Prop)
  (l1_eq : ∀ x y, l1 x y ↔ (a+3)*x + y - 4 = 0)
  (l2 : ℝ → ℝ → Prop)
  (l2_eq : ∀ x y, l2 x y ↔ x + (a-1)*y + 4 = 0)
  (perpendicular : ∀ x y, l1 x y → l2 x y → (a+3)*(a-1) = -1) :
  (∃ x : ℝ, l1 x 0 ∧ x = 2) :=
sorry

end x_intercept_of_perpendicular_line_l173_173353


namespace smallest_positive_z_l173_173019

open Real

theorem smallest_positive_z (x y z : ℝ) (m k n : ℤ) 
  (h1 : cos x = 0) 
  (h2 : sin y = 1) 
  (h3 : cos (x + z) = -1 / 2) :
  z = 5 * π / 6 :=
by
  sorry

end smallest_positive_z_l173_173019


namespace prob_at_least_one_solves_l173_173940

theorem prob_at_least_one_solves (p1 p2 : ℝ) (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (1 : ℝ) - (1 - p1) * (1 - p2) = 1 - ((1 - p1) * (1 - p2)) :=
by sorry

end prob_at_least_one_solves_l173_173940


namespace jerry_claim_percentage_l173_173239

theorem jerry_claim_percentage
  (salary_years : ℕ)
  (annual_salary : ℕ)
  (medical_bills : ℕ)
  (punitive_multiplier : ℕ)
  (received_amount : ℕ)
  (total_claim : ℕ)
  (percentage_claim : ℕ) :
  salary_years = 30 →
  annual_salary = 50000 →
  medical_bills = 200000 →
  punitive_multiplier = 3 →
  received_amount = 5440000 →
  total_claim = (annual_salary * salary_years) + medical_bills + (punitive_multiplier * ((annual_salary * salary_years) + medical_bills)) →
  percentage_claim = (received_amount * 100) / total_claim →
  percentage_claim = 80 :=
by
  sorry

end jerry_claim_percentage_l173_173239


namespace joan_gave_melanie_apples_l173_173422

theorem joan_gave_melanie_apples (original_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : original_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  sorry

end joan_gave_melanie_apples_l173_173422


namespace problem1_problem2_l173_173153

theorem problem1 : |Real.sqrt 2 - Real.sqrt 3| + 2 * Real.sqrt 2 = Real.sqrt 3 + Real.sqrt 2 := 
by {
  sorry
}

theorem problem2 : Real.sqrt 5 * (Real.sqrt 5 - 1 / Real.sqrt 5) = 4 := 
by {
  sorry
}

end problem1_problem2_l173_173153


namespace system_of_equations_solution_system_of_inequalities_solution_l173_173751

theorem system_of_equations_solution (x y : ℝ) :
  (3 * x - 4 * y = 1) → (5 * x + 2 * y = 6) → 
  x = 1 ∧ y = 0.5 := by
  sorry

theorem system_of_inequalities_solution (x : ℝ) :
  (3 * x + 6 > 0) → (x - 2 < -x) → 
  -2 < x ∧ x < 1 := by
  sorry

end system_of_equations_solution_system_of_inequalities_solution_l173_173751


namespace multiple_of_weight_lifted_l173_173703

variable (F : ℝ) (M : ℝ)

theorem multiple_of_weight_lifted 
  (H1: ∀ (B : ℝ), B = 2 * F) 
  (H2: ∀ (B : ℝ), ∀ (W : ℝ), W = 3 * B) 
  (H3: ∃ (B : ℝ), (3 * B = 600)) 
  (H4: M * F = 150) : 
  M = 1.5 :=
by
  sorry

end multiple_of_weight_lifted_l173_173703


namespace big_eighteen_basketball_games_count_l173_173639

def num_teams_in_division := 6
def num_teams := 18
def games_within_division := 3
def games_between_divisions := 1
def divisions := 3

theorem big_eighteen_basketball_games_count :
  (num_teams * ((num_teams_in_division - 1) * games_within_division + (num_teams - num_teams_in_division) * games_between_divisions)) / 2 = 243 :=
by
  have teams_in_other_divisions : num_teams - num_teams_in_division = 12 := rfl
  have games_per_team_within_division : (num_teams_in_division - 1) * games_within_division = 15 := rfl
  have games_per_team_between_division : 12 * games_between_divisions = 12 := rfl
  sorry

end big_eighteen_basketball_games_count_l173_173639


namespace mean_home_runs_l173_173885

theorem mean_home_runs :
  let n_5 := 3
  let n_8 := 5
  let n_9 := 3
  let n_11 := 1
  let total_home_runs := 5 * n_5 + 8 * n_8 + 9 * n_9 + 11 * n_11
  let total_players := n_5 + n_8 + n_9 + n_11
  let mean := total_home_runs / total_players
  mean = 7.75 :=
by
  sorry

end mean_home_runs_l173_173885


namespace factor_polynomial_l173_173742

theorem factor_polynomial :
  9 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 5 * x ^ 2 =
  (3 * x ^ 2 + 59 * x + 231) * (3 * x ^ 2 + 53 * x + 231) := by
  sorry

end factor_polynomial_l173_173742


namespace range_of_m_l173_173659

-- Definitions based on the conditions
def p (m : ℝ) : Prop := 4 - 4 * m > 0
def q (m : ℝ) : Prop := m + 2 > 0

-- Problem statement in Lean 4
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ≤ -2 ∨ m ≥ 1 := by
  sorry

end range_of_m_l173_173659


namespace area_of_fourth_rectangle_l173_173111

theorem area_of_fourth_rectangle
    (x y z w : ℝ)
    (h1 : x * y = 24)
    (h2 : z * y = 15)
    (h3 : z * w = 9) :
    y * w = 15 := 
sorry

end area_of_fourth_rectangle_l173_173111


namespace max_length_PQ_l173_173595

-- Define the curve in polar coordinates
def curve (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

-- Definition of points P and Q lying on the curve
def point_on_curve (ρ θ : ℝ) (P : ℝ × ℝ) : Prop :=
  curve ρ θ ∧ P = (ρ * Real.cos θ, ρ * Real.sin θ)

def points_on_curve (P Q : ℝ × ℝ) : Prop :=
  ∃ θ₁ θ₂ ρ₁ ρ₂, point_on_curve ρ₁ θ₁ P ∧ point_on_curve ρ₂ θ₂ Q

-- The theorem stating the maximum length of PQ
theorem max_length_PQ {P Q : ℝ × ℝ} (h : points_on_curve P Q) : dist P Q ≤ 4 :=
sorry

end max_length_PQ_l173_173595


namespace feet_to_inches_conversion_l173_173736

-- Define the constant equivalence between feet and inches
def foot_to_inches := 12

-- Prove the conversion factor between feet and inches
theorem feet_to_inches_conversion:
  foot_to_inches = 12 :=
by
  sorry

end feet_to_inches_conversion_l173_173736


namespace recipe_sugar_amount_l173_173001

-- Definitions from A)
def cups_of_salt : ℕ := 9
def additional_cups_of_sugar (sugar salt : ℕ) : Prop := sugar = salt + 2

-- Statement to prove
theorem recipe_sugar_amount (salt : ℕ) (h : salt = cups_of_salt) : ∃ sugar : ℕ, additional_cups_of_sugar sugar salt ∧ sugar = 11 :=
by
  sorry

end recipe_sugar_amount_l173_173001


namespace average_age_students_l173_173313

theorem average_age_students 
  (total_students : ℕ)
  (group1 : ℕ)
  (group1_avg_age : ℕ)
  (group2 : ℕ)
  (group2_avg_age : ℕ)
  (student15_age : ℕ)
  (avg_age : ℕ) 
  (h1 : total_students = 15)
  (h2 : group1_avg_age = 14)
  (h3 : group2 = 8)
  (h4 : group2_avg_age = 16)
  (h5 : student15_age = 13)
  (h6 : avg_age = (84 + 128 + 13) / 15)
  (h7 : avg_age = 15) :
  group1 = 6 :=
by sorry

end average_age_students_l173_173313


namespace gov_addresses_l173_173149

theorem gov_addresses (S H K : ℕ) 
  (H1 : S = 2 * H) 
  (H2 : K = S + 10) 
  (H3 : S + H + K = 40) : 
  S = 12 := 
sorry 

end gov_addresses_l173_173149


namespace B_current_age_l173_173960

theorem B_current_age (A B : ℕ) (h1 : A = B + 15) (h2 : A - 5 = 2 * (B - 5)) : B = 20 :=
by sorry

end B_current_age_l173_173960


namespace quadratic_symmetry_l173_173779

def quadratic (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b * x + c

theorem quadratic_symmetry (b c : ℝ) :
  let f := quadratic b c
  (f 2) < (f 1) ∧ (f 1) < (f 4) :=
by
  sorry

end quadratic_symmetry_l173_173779


namespace number_of_whole_numbers_between_roots_l173_173952

theorem number_of_whole_numbers_between_roots :
  let sqrt_18 := Real.sqrt 18
  let sqrt_98 := Real.sqrt 98
  Nat.card { x : ℕ | sqrt_18 < x ∧ x < sqrt_98 } = 5 := 
by
  sorry

end number_of_whole_numbers_between_roots_l173_173952


namespace period_of_f_l173_173401

noncomputable def f (x : ℝ) : ℝ := sorry

theorem period_of_f (a : ℝ) (h : a ≠ 0) (H : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : 
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = 4 * |a| :=
by
  sorry

end period_of_f_l173_173401


namespace additional_men_joined_l173_173546

theorem additional_men_joined (men_initial : ℕ) (days_initial : ℕ)
  (days_new : ℕ) (additional_men : ℕ) :
  men_initial = 600 →
  days_initial = 20 →
  days_new = 15 →
  (men_initial * days_initial) = ((men_initial + additional_men) * days_new) →
  additional_men = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end additional_men_joined_l173_173546


namespace intersection_hyperbola_l173_173708

theorem intersection_hyperbola (t : ℝ) :
  ∃ A B : ℝ, ∀ (x y : ℝ),
  (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 5 = 0) →
  (x^2 / A - y^2 / B = 1) :=
sorry

end intersection_hyperbola_l173_173708


namespace mike_passing_percentage_l173_173865

theorem mike_passing_percentage (scored shortfall max_marks : ℝ) (total_marks := scored + shortfall) :
    scored = 212 →
    shortfall = 28 →
    max_marks = 800 →
    (total_marks / max_marks) * 100 = 30 :=
by
  intros
  sorry

end mike_passing_percentage_l173_173865


namespace garden_width_min_5_l173_173967

theorem garden_width_min_5 (width length : ℝ) (h_length : length = width + 20) (h_area : width * length ≥ 150) :
  width ≥ 5 :=
sorry

end garden_width_min_5_l173_173967


namespace spherical_to_rectangular_coordinates_l173_173614

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ), ρ = 6 → θ = 7 * Real.pi / 4 → φ = Real.pi / 4 →
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (3, -3, 3 * Real.sqrt 2) := by
  sorry

end spherical_to_rectangular_coordinates_l173_173614


namespace spencer_sessions_per_day_l173_173048

theorem spencer_sessions_per_day :
  let jumps_per_minute := 4
  let minutes_per_session := 10
  let jumps_per_session := jumps_per_minute * minutes_per_session
  let total_jumps := 400
  let days := 5
  let jumps_per_day := total_jumps / days
  let sessions_per_day := jumps_per_day / jumps_per_session
  sessions_per_day = 2 :=
by
  sorry

end spencer_sessions_per_day_l173_173048


namespace intersecting_diagonals_of_parallelogram_l173_173165

theorem intersecting_diagonals_of_parallelogram (A C : ℝ × ℝ) (hA : A = (2, -3)) (hC : C = (14, 9)) :
    ∃ M : ℝ × ℝ, M = (8, 3) ∧ M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) :=
by {
  sorry
}

end intersecting_diagonals_of_parallelogram_l173_173165


namespace helen_baked_more_raisin_cookies_l173_173315

-- Definitions based on conditions
def raisin_cookies_yesterday : ℕ := 300
def raisin_cookies_day_before : ℕ := 280

-- Theorem to prove the answer
theorem helen_baked_more_raisin_cookies : raisin_cookies_yesterday - raisin_cookies_day_before = 20 :=
by
  sorry

end helen_baked_more_raisin_cookies_l173_173315


namespace problem_statement_l173_173849

theorem problem_statement (x y z w : ℝ)
  (h1 : x + y + z + w = 0)
  (h7 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := 
sorry

end problem_statement_l173_173849


namespace evaluate_f_at_2_l173_173462

def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem evaluate_f_at_2 :
  f 2 = -2 :=
by
  sorry

end evaluate_f_at_2_l173_173462


namespace root_of_quadratic_l173_173426

theorem root_of_quadratic (a b c : ℝ) :
  (4 * a + 2 * b + c = 0) ↔ (a * 2^2 + b * 2 + c = 0) :=
by
  sorry

end root_of_quadratic_l173_173426


namespace rectangle_area_l173_173231

theorem rectangle_area (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x * y = 5 :=
by
  -- Conditions given to us:
  -- 1. (h1) The sum of the sides is 5.
  -- 2. (h2) The sum of the squares of the sides is 15.
  -- We need to prove that the product of the sides is 5.
  sorry

end rectangle_area_l173_173231


namespace solve_for_m_l173_173765

theorem solve_for_m (m : ℝ) (x1 x2 : ℝ)
    (h1 : x1^2 - (2 * m - 1) * x1 + m^2 = 0)
    (h2 : x2^2 - (2 * m - 1) * x2 + m^2 = 0)
    (h3 : (x1 + 1) * (x2 + 1) = 3)
    (h_reality : (2 * m - 1)^2 - 4 * m^2 ≥ 0) :
    m = -3 := by
  sorry

end solve_for_m_l173_173765


namespace initial_discount_percentage_l173_173618

variable (d : ℝ) (x : ℝ)
variable (h1 : 0 < d) (h2 : 0 ≤ x) (h3 : x ≤ 100)
variable (h4 : (1 - x / 100) * 0.6 * d = 0.33 * d)

theorem initial_discount_percentage : x = 45 :=
by
  sorry

end initial_discount_percentage_l173_173618


namespace sum_of_constants_l173_173711

theorem sum_of_constants :
  ∃ (a b c d e : ℤ), 1000 * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e) ∧ a + b + c + d + e = 92 :=
by
  sorry

end sum_of_constants_l173_173711


namespace gcd_8251_6105_l173_173856

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l173_173856


namespace range_of_x_inequality_l173_173669

theorem range_of_x_inequality (a : ℝ) (x : ℝ)
  (h : -1 ≤ a ∧ a ≤ 1) : 
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end range_of_x_inequality_l173_173669


namespace prove_sum_l173_173962

theorem prove_sum (a b : ℝ) (h1 : a * (a - 4) = 12) (h2 : b * (b - 4) = 12) (h3 : a ≠ b) : a + b = 4 := by
  sorry

end prove_sum_l173_173962


namespace sqrt_four_eq_pm_two_l173_173141

theorem sqrt_four_eq_pm_two : ∀ (x : ℝ), x * x = 4 ↔ x = 2 ∨ x = -2 := 
by
  sorry

end sqrt_four_eq_pm_two_l173_173141


namespace egyptians_panamanians_l173_173110

-- Given: n + m = 12 and (n(n-1))/2 + (m(m-1))/2 = 31 and n > m
-- Prove: n = 7 and m = 5

theorem egyptians_panamanians (n m : ℕ) (h1 : n + m = 12) (h2 : n > m) 
(h3 : n * (n - 1) / 2 + m * (m - 1) / 2 = 31) :
  n = 7 ∧ m = 5 := 
by
  sorry

end egyptians_panamanians_l173_173110


namespace polygon_number_of_sides_and_interior_sum_l173_173052

-- Given conditions
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)
def exterior_angle_sum : ℝ := 360

-- Proof problem statement
theorem polygon_number_of_sides_and_interior_sum (n : ℕ)
  (h : interior_angle_sum n = 3 * exterior_angle_sum) :
  n = 8 ∧ interior_angle_sum n = 1080 :=
by
  sorry

end polygon_number_of_sides_and_interior_sum_l173_173052


namespace base_of_isosceles_triangle_l173_173192

namespace TriangleProblem

def equilateral_triangle_perimeter (s : ℕ) : ℕ := 3 * s
def isosceles_triangle_perimeter (s b : ℕ) : ℕ := 2 * s + b

theorem base_of_isosceles_triangle (s b : ℕ) (h1 : equilateral_triangle_perimeter s = 45) 
    (h2 : isosceles_triangle_perimeter s b = 40) : b = 10 :=
by
  sorry

end TriangleProblem

end base_of_isosceles_triangle_l173_173192


namespace sequence_formula_l173_173471

theorem sequence_formula (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = n^2 + 1) :
  (a 1 = 2) ∧ (∀ n ≥ 2, a n = 2 * n - 1) := 
by
  sorry

end sequence_formula_l173_173471


namespace compare_points_l173_173443

def parabola (x : ℝ) : ℝ := -x^2 - 4 * x + 1

theorem compare_points (y₁ y₂ : ℝ) :
  parabola (-3) = y₁ →
  parabola (-2) = y₂ →
  y₁ < y₂ :=
by
  intros hy₁ hy₂
  sorry

end compare_points_l173_173443


namespace total_rats_l173_173027

variable (Kenia Hunter Elodie : ℕ) -- Number of rats each person has

-- Conditions
-- Elodie has 30 rats
axiom h1 : Elodie = 30
-- Elodie has 10 rats more than Hunter
axiom h2 : Elodie = Hunter + 10
-- Kenia has three times as many rats as Hunter and Elodie have together
axiom h3 : Kenia = 3 * (Hunter + Elodie)

-- Prove that the total number of pets the three have together is 200
theorem total_rats : Kenia + Hunter + Elodie = 200 := 
by 
  sorry

end total_rats_l173_173027


namespace large_square_min_side_and_R_max_area_l173_173299

-- Define the conditions
variable (s : ℝ) -- the side length of the larger square
variable (rect_1_side1 rect_1_side2 : ℝ) -- sides of the first rectangle
variable (square_side : ℝ) -- side of the inscribed square
variable (R_area : ℝ) -- area of the rectangle R

-- The known dimensions
axiom h1 : rect_1_side1 = 2
axiom h2 : rect_1_side2 = 4
axiom h3 : square_side = 2
axiom h4 : ∀ x y : ℝ, x > 0 → y > 0 → R_area = x * y -- non-overlapping condition

-- Define the result to be proved
theorem large_square_min_side_and_R_max_area 
  (h_r_fit_1 : rect_1_side1 + square_side ≤ s)
  (h_r_fit_2 : rect_1_side2 + square_side ≤ s)
  (h_R_max_area : R_area = 4)
  : s = 4 ∧ R_area = 4 := 
by 
  sorry

end large_square_min_side_and_R_max_area_l173_173299


namespace value_of_expression_l173_173247

theorem value_of_expression (x y : ℕ) (h₁ : x = 12) (h₂ : y = 7) : (x - y) * (x + y) = 95 := by
  -- Here we assume all necessary conditions as given:
  -- x = 12 and y = 7
  -- and we prove that (x - y)(x + y) = 95
  sorry

end value_of_expression_l173_173247


namespace probability_of_square_product_is_17_over_96_l173_173002

def num_tiles : Nat := 12
def num_die_faces : Nat := 8

def is_perfect_square (n : Nat) : Prop :=
  ∃ k : Nat, k * k = n

def favorable_outcomes_count : Nat :=
  -- Valid pairs where tile's number and die's number product is a perfect square
  List.length [ (1, 1), (1, 4), (2, 2), (4, 1),
                (1, 9), (3, 3), (9, 1), (4, 4),
                (2, 8), (8, 2), (5, 5), (6, 6),
                (4, 9), (9, 4), (7, 7), (8, 8),
                (9, 9) ] -- Equals 17 pairs

def total_outcomes_count : Nat :=
  num_tiles * num_die_faces

def probability_square_product : ℚ :=
  favorable_outcomes_count / total_outcomes_count

theorem probability_of_square_product_is_17_over_96 :
  probability_square_product = (17 : ℚ) / 96 := 
  by sorry

end probability_of_square_product_is_17_over_96_l173_173002


namespace teacher_age_is_94_5_l173_173829

noncomputable def avg_age_students : ℝ := 18
noncomputable def num_students : ℝ := 50
noncomputable def avg_age_class_with_teacher : ℝ := 19.5
noncomputable def num_total : ℝ := 51

noncomputable def total_age_students : ℝ := num_students * avg_age_students
noncomputable def total_age_class_with_teacher : ℝ := num_total * avg_age_class_with_teacher

theorem teacher_age_is_94_5 : ∃ T : ℝ, total_age_students + T = total_age_class_with_teacher ∧ T = 94.5 := by
  sorry

end teacher_age_is_94_5_l173_173829


namespace all_positive_integers_in_A_l173_173743

variable (A : Set ℕ)

-- Conditions
def has_at_least_three_elements : Prop :=
  ∃ a b c : ℕ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c

def all_divisors_in_set : Prop :=
  ∀ m : ℕ, m ∈ A → (∀ d : ℕ, d ∣ m → d ∈ A)

def  bc_plus_one_in_set : Prop :=
  ∀ b c : ℕ, 1 < b → b < c → b ∈ A → c ∈ A → 1 + b * c ∈ A

-- Theorem statement
theorem all_positive_integers_in_A
  (h1 : has_at_least_three_elements A)
  (h2 : all_divisors_in_set A)
  (h3 : bc_plus_one_in_set A) : ∀ n : ℕ, n > 0 → n ∈ A := 
by
  -- proof steps would go here
  sorry

end all_positive_integers_in_A_l173_173743


namespace man_year_of_birth_l173_173464

theorem man_year_of_birth (x : ℕ) (hx1 : (x^2 + x >= 1850)) (hx2 : (x^2 + x < 1900)) : (1850 + (x^2 + x - x)) = 1892 :=
by {
  sorry
}

end man_year_of_birth_l173_173464


namespace solve_sqrt_equation_l173_173660

theorem solve_sqrt_equation :
  ∀ (x : ℝ), (3 * Real.sqrt x + 3 * x⁻¹/2 = 7) →
  (x = (49 + 14 * Real.sqrt 13 + 13) / 36 ∨ x = (49 - 14 * Real.sqrt 13 + 13) / 36) :=
by
  intro x hx
  sorry

end solve_sqrt_equation_l173_173660


namespace area_six_layers_l173_173134

theorem area_six_layers
  (A : ℕ → ℕ)
  (h1 : A 1 + A 2 + A 3 = 280)
  (h2 : A 2 = 54)
  (h3 : A 3 = 28)
  (h4 : A 4 = 14)
  (h5 : A 1 + 2 * A 2 + 3 * A 3 + 4 * A 4 + 6 * A 6 = 500)
  : A 6 = 9 := 
sorry

end area_six_layers_l173_173134


namespace pastor_prayer_ratio_l173_173626

theorem pastor_prayer_ratio 
  (R : ℚ) 
  (paul_prays_per_day : ℚ := 20)
  (paul_sunday_times : ℚ := 2 * paul_prays_per_day)
  (paul_total : ℚ := 6 * paul_prays_per_day + paul_sunday_times)
  (bruce_ratio : ℚ := R)
  (bruce_prays_per_day : ℚ := bruce_ratio * paul_prays_per_day)
  (bruce_sunday_times : ℚ := 2 * paul_sunday_times)
  (bruce_total : ℚ := 6 * bruce_prays_per_day + bruce_sunday_times)
  (condition : paul_total = bruce_total + 20) :
  R = 1/2 :=
sorry

end pastor_prayer_ratio_l173_173626


namespace average_first_21_multiples_of_17_l173_173415

theorem average_first_21_multiples_of_17:
  let n := 21
  let a1 := 17
  let a21 := 17 * n
  let sum := n / 2 * (a1 + a21)
  (sum / n = 187) :=
by
  sorry

end average_first_21_multiples_of_17_l173_173415


namespace film_radius_l173_173375

theorem film_radius 
  (thickness : ℝ)
  (container_volume : ℝ)
  (r : ℝ)
  (H1 : thickness = 0.25)
  (H2 : container_volume = 128) :
  r = Real.sqrt (512 / Real.pi) :=
by
  -- Placeholder for proof
  sorry

end film_radius_l173_173375


namespace train_B_time_to_reach_destination_l173_173196

theorem train_B_time_to_reach_destination
    (T t : ℝ)
    (train_A_speed : ℝ) (train_B_speed : ℝ)
    (train_A_extra_hours : ℝ)
    (h1 : train_A_speed = 110)
    (h2 : train_B_speed = 165)
    (h3 : train_A_extra_hours = 9)
    (h_eq : 110 * (T + train_A_extra_hours) = 110 * T + 165 * t) :
    t = 6 := 
by
  sorry

end train_B_time_to_reach_destination_l173_173196


namespace xiao_ming_total_score_l173_173314

theorem xiao_ming_total_score :
  ∃ (a_1 a_2 a_3 a_4 a_5 : ℕ), 
  a_1 < a_2 ∧ 
  a_2 < a_3 ∧ 
  a_3 < a_4 ∧ 
  a_4 < a_5 ∧ 
  a_1 + a_2 = 10 ∧ 
  a_4 + a_5 = 18 ∧ 
  a_1 + a_2 + a_3 + a_4 + a_5 = 35 :=
by
  sorry

end xiao_ming_total_score_l173_173314


namespace percentage_fruits_in_good_condition_l173_173817

theorem percentage_fruits_in_good_condition (oranges bananas : ℕ) (rotten_oranges_pct rotten_bananas_pct : ℚ)
    (h_oranges : oranges = 600) (h_bananas : bananas = 400)
    (h_rotten_oranges_pct : rotten_oranges_pct = 0.15) (h_rotten_bananas_pct : rotten_bananas_pct = 0.06) :
    let rotten_oranges := (rotten_oranges_pct * oranges : ℚ)
    let rotten_bananas := (rotten_bananas_pct * bananas : ℚ)
    let total_rotten := rotten_oranges + rotten_bananas
    let total_fruits := (oranges + bananas : ℚ)
    let good_fruits := total_fruits - total_rotten
    let percentage_good_fruits := (good_fruits / total_fruits) * 100
    percentage_good_fruits = 88.6 :=
by
    sorry

end percentage_fruits_in_good_condition_l173_173817


namespace solve_equation_l173_173797

theorem solve_equation :
  { x : ℝ | x * (x - 3)^2 * (5 - x) = 0 } = {0, 3, 5} :=
by
  sorry

end solve_equation_l173_173797


namespace fred_games_this_year_l173_173175

variable (last_year_games : ℕ)
variable (difference : ℕ)

theorem fred_games_this_year (h1 : last_year_games = 36) (h2 : difference = 11) : 
  last_year_games - difference = 25 := 
by
  sorry

end fred_games_this_year_l173_173175


namespace probability_within_sphere_correct_l173_173459

noncomputable def probability_within_sphere : ℝ :=
  let cube_volume := (2 : ℝ) * (2 : ℝ) * (2 : ℝ)
  let sphere_volume := (4 * Real.pi / 3) * (0.5) ^ 3
  sphere_volume / cube_volume

theorem probability_within_sphere_correct (x y z : ℝ) 
  (hx1 : -1 ≤ x) (hx2 : x ≤ 1) 
  (hy1 : -1 ≤ y) (hy2 : y ≤ 1) 
  (hz1 : -1 ≤ z) (hz2 : z ≤ 1) 
  (hx_sq : x^2 ≤ 0.5) 
  (hxyz : x^2 + y^2 + z^2 ≤ 0.25) : 
  probability_within_sphere = Real.pi / 48 :=
by
  sorry

end probability_within_sphere_correct_l173_173459


namespace count_solutions_l173_173553

noncomputable def num_solutions : ℕ :=
  let eq1 (x y : ℝ) := 2 * x + 5 * y = 10
  let eq2 (x y : ℝ) := abs (abs (x + 1) - abs (y - 1)) = 1
  sorry

theorem count_solutions : num_solutions = 2 := by
  sorry

end count_solutions_l173_173553


namespace train_crosses_platform_in_39_seconds_l173_173090

theorem train_crosses_platform_in_39_seconds :
  ∀ (length_train length_platform : ℝ) (time_cross_signal : ℝ),
  length_train = 300 →
  length_platform = 25 →
  time_cross_signal = 36 →
  ((length_train + length_platform) / (length_train / time_cross_signal)) = 39 := by
  intros length_train length_platform time_cross_signal
  intros h_length_train h_length_platform h_time_cross_signal
  rw [h_length_train, h_length_platform, h_time_cross_signal]
  sorry

end train_crosses_platform_in_39_seconds_l173_173090


namespace Raven_age_l173_173818

-- Define the conditions
def Phoebe_age_current : Nat := 10
def Phoebe_age_in_5_years : Nat := Phoebe_age_current + 5

-- Define the hypothesis that in 5 years Raven will be 4 times as old as Phoebe
def Raven_in_5_years (R : Nat) : Prop := R + 5 = 4 * Phoebe_age_in_5_years

-- State the theorem to be proved
theorem Raven_age : ∃ R : Nat, Raven_in_5_years R ∧ R = 55 :=
by
  sorry

end Raven_age_l173_173818


namespace ariel_age_l173_173827

theorem ariel_age : ∃ A : ℕ, (A + 15 = 4 * A) ∧ A = 5 :=
by
  -- Here we skip the proof
  sorry

end ariel_age_l173_173827


namespace systematic_sampling_l173_173470

theorem systematic_sampling (total_employees groups group_size draw_5th draw_10th : ℕ)
  (h1 : total_employees = 200)
  (h2 : groups = 40)
  (h3 : group_size = total_employees / groups)
  (h4 : draw_5th = 22)
  (h5 : ∃ x : ℕ, draw_5th = (5-1) * group_size + x)
  (h6 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ groups → draw_10th = (k-1) * group_size + x) :
  draw_10th = 47 := 
by
  sorry

end systematic_sampling_l173_173470


namespace average_primes_4_to_15_l173_173454

theorem average_primes_4_to_15 :
  (5 + 7 + 11 + 13) / 4 = 9 :=
by sorry

end average_primes_4_to_15_l173_173454


namespace triangle_inequality_l173_173652

theorem triangle_inequality (a : ℝ) (h1 : a + 3 > 5) (h2 : a + 5 > 3) (h3 : 3 + 5 > a) :
  2 < a ∧ a < 8 :=
by {
  sorry
}

end triangle_inequality_l173_173652


namespace min_value_abs_function_l173_173678

theorem min_value_abs_function : ∀ (x : ℝ), (|x + 1| + |2 - x|) ≥ 3 :=
by
  sorry

end min_value_abs_function_l173_173678


namespace max_n_l173_173674

noncomputable def prod := 160 * 170 * 180 * 190

theorem max_n : ∃ n : ℕ, n = 30499 ∧ n^2 ≤ prod := by
  sorry

end max_n_l173_173674


namespace four_digit_numbers_with_three_identical_digits_l173_173709

theorem four_digit_numbers_with_three_identical_digits :
  ∃ n : ℕ, (n = 18) ∧ (∀ x, 1000 ≤ x ∧ x < 10000 → 
  (x / 1000 = 1) ∧ (
    (x % 1000 / 100 = x % 100 / 10) ∧ (x % 1000 / 100 = x % 10))) :=
by
  sorry

end four_digit_numbers_with_three_identical_digits_l173_173709


namespace maximum_value_of_A_l173_173554

theorem maximum_value_of_A (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
    (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end maximum_value_of_A_l173_173554


namespace sum_of_infinite_series_l173_173689

noncomputable def infinite_series : ℝ :=
  ∑' k : ℕ, (k^3 : ℝ) / (3^k : ℝ)

theorem sum_of_infinite_series :
  infinite_series = (39/16 : ℝ) :=
sorry

end sum_of_infinite_series_l173_173689


namespace find_y_l173_173631

theorem find_y (x y : ℝ) (h₁ : 1.5 * x = 0.3 * y) (h₂ : x = 20) : y = 100 :=
sorry

end find_y_l173_173631


namespace chrysler_floors_difference_l173_173688

theorem chrysler_floors_difference (C L : ℕ) (h1 : C = 23) (h2 : C + L = 35) : C - L = 11 := by
  sorry

end chrysler_floors_difference_l173_173688


namespace angela_age_in_5_years_l173_173975

-- Define the variables representing Angela and Beth's ages.
variable (A B : ℕ)

-- State the conditions as hypotheses.
def condition_1 : Prop := A = 4 * B
def condition_2 : Prop := (A - 5) + (B - 5) = 45

-- State the final proposition that Angela will be 49 years old in five years.
theorem angela_age_in_5_years (h1 : condition_1 A B) (h2 : condition_2 A B) : A + 5 = 49 := by
  sorry

end angela_age_in_5_years_l173_173975


namespace no_common_points_l173_173571

def curve1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def curve2 (x y : ℝ) : Prop := x^2 + 2*y^2 = 2

theorem no_common_points :
  ¬ ∃ (x y : ℝ), curve1 x y ∧ curve2 x y :=
by sorry

end no_common_points_l173_173571


namespace find_x_values_l173_173248

noncomputable def tan_inv := Real.arctan (Real.sqrt 3 / 2)

theorem find_x_values (x : ℝ) :
  (-Real.pi < x ∧ x ≤ Real.pi) ∧ (2 * Real.tan x - Real.sqrt 3 = 0) ↔
  (x = tan_inv ∨ x = tan_inv - Real.pi) :=
by
  sorry

end find_x_values_l173_173248


namespace eq_solution_set_l173_173423

theorem eq_solution_set :
  {x : ℝ | (2 / (x + 2)) + (4 / (x + 8)) ≥ 3 / 4} = {x : ℝ | -2 < x ∧ x ≤ 2} :=
by {
  sorry
}

end eq_solution_set_l173_173423


namespace find_g_at_4_l173_173367

theorem find_g_at_4 (g : ℝ → ℝ) (h : ∀ x, 2 * g x + 3 * g (1 - x) = 4 * x^3 - x) : g 4 = 193.2 :=
sorry

end find_g_at_4_l173_173367


namespace min_value_x_y_l173_173875

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 19 / x + 98 / y = 1) : x + y ≥ 117 + 14 * Real.sqrt 38 := 
sorry

end min_value_x_y_l173_173875


namespace circle_intersection_exists_l173_173491

theorem circle_intersection_exists (a b : ℝ) :
  ∃ (m n : ℤ), (m - a)^2 + (n - b)^2 ≤ (1 / 14)^2 →
  ∀ x y, (x - a)^2 + (y - b)^2 = 100^2 :=
sorry

end circle_intersection_exists_l173_173491


namespace infinite_quadruples_inequality_quadruple_l173_173363

theorem infinite_quadruples 
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  ∃ (a p q r : ℕ), 
    1 < p ∧ 1 < q ∧ 1 < r ∧
    p ∣ (a * q * r + 1) ∧
    q ∣ (a * p * r + 1) ∧
    r ∣ (a * p * q + 1) :=
sorry

theorem inequality_quadruple
  (a p q r : ℤ) 
  (hp : 1 < p) (hq : 1 < q) (hr : 1 < r)
  (hp_div : p ∣ (a * q * r + 1))
  (hq_div : q ∣ (a * p * r + 1))
  (hr_div : r ∣ (a * p * q + 1)) :
  a ≥ (p * q * r - 1) / (p * q + q * r + r * p) :=
sorry

end infinite_quadruples_inequality_quadruple_l173_173363


namespace julian_younger_than_frederick_by_20_l173_173993

noncomputable def Kyle: ℕ := 25
noncomputable def Tyson: ℕ := 20
noncomputable def Julian : ℕ := Kyle - 5
noncomputable def Frederick : ℕ := 2 * Tyson

theorem julian_younger_than_frederick_by_20 : Frederick - Julian = 20 :=
by
  sorry

end julian_younger_than_frederick_by_20_l173_173993


namespace initial_weight_l173_173841

noncomputable def initial_average_weight (A : ℝ) : Prop :=
  let total_weight_initial := 20 * A
  let total_weight_new := total_weight_initial + 210
  let new_average_weight := 181.42857142857142
  total_weight_new / 21 = new_average_weight

theorem initial_weight:
  ∃ A : ℝ, initial_average_weight A ∧ A = 180 :=
by
  sorry

end initial_weight_l173_173841


namespace solve_for_x_l173_173889

theorem solve_for_x (x : ℝ) (h₀ : x^2 - 2 * x = 0) (h₁ : x ≠ 0) : x = 2 :=
sorry

end solve_for_x_l173_173889


namespace ab_times_65_eq_48ab_l173_173038

theorem ab_times_65_eq_48ab (a b : ℕ) (h_ab : 0 ≤ a ∧ a < 10) (h_b : 0 ≤ b ∧ b < 10) :
  (10 * a + b) * 65 = 4800 + 10 * a + b ↔ 10 * a + b = 75 := by
sorry

end ab_times_65_eq_48ab_l173_173038


namespace sum_positive_implies_at_least_one_positive_l173_173082

theorem sum_positive_implies_at_least_one_positive (a b : ℝ) (h : a + b > 0) : a > 0 ∨ b > 0 :=
sorry

end sum_positive_implies_at_least_one_positive_l173_173082


namespace distance_between_stripes_l173_173978

/-- Define the parallel curbs and stripes -/
structure Crosswalk where
  distance_between_curbs : ℝ
  curb_distance_between_stripes : ℝ
  stripe_length : ℝ
  stripe_cross_distance : ℝ
  
open Crosswalk

/-- Conditions given in the problem -/
def crosswalk : Crosswalk where
  distance_between_curbs := 60 -- feet
  curb_distance_between_stripes := 20 -- feet
  stripe_length := 50 -- feet
  stripe_cross_distance := 50 -- feet

/-- Theorem to prove the distance between stripes -/
theorem distance_between_stripes (cw : Crosswalk) :
  2 * (cw.curb_distance_between_stripes * cw.distance_between_curbs) / cw.stripe_length = 24 := sorry

end distance_between_stripes_l173_173978


namespace enrique_shredder_Y_feeds_l173_173391

theorem enrique_shredder_Y_feeds :
  let typeB_contracts := 350
  let pages_per_TypeB := 10
  let shredderY_capacity := 8
  let total_pages_TypeB := typeB_contracts * pages_per_TypeB
  let feeds_ShredderY := (total_pages_TypeB + shredderY_capacity - 1) / shredderY_capacity
  feeds_ShredderY = 438 := sorry

end enrique_shredder_Y_feeds_l173_173391


namespace sum_of_squares_of_roots_l173_173285

theorem sum_of_squares_of_roots : 
  (∃ r1 r2 : ℝ, r1 + r2 = 11 ∧ r1 * r2 = 12 ∧ (r1 ^ 2 + r2 ^ 2) = 97) := 
sorry

end sum_of_squares_of_roots_l173_173285


namespace value_of_x_l173_173999

theorem value_of_x (x c m n : ℝ) (hne: m≠n) (hneq : c ≠ 0) 
  (h1: c = 3) (h2: m = 2) (h3: n = 5)
  (h4: (x + c * m)^2 - (x + c * n)^2 = (m - n)^2) : 
  x = -11 := by
  sorry

end value_of_x_l173_173999


namespace person_B_winning_strategy_l173_173954

-- Definitions for the problem conditions
def winning_strategy_condition (L a b : ℕ) : Prop := 
  b = 2 * a ∧ ∃ k : ℕ, L = k * a

-- Lean theorem statement for the given problem
theorem person_B_winning_strategy (L a b : ℕ) (hL_pos : 0 < L) (ha_lt_hb : a < b) 
(hpos_a : 0 < a) (hpos_b : 0 < b) : 
  (∃ B_strat : Type, winning_strategy_condition L a b) :=
sorry

end person_B_winning_strategy_l173_173954


namespace sequence_term_l173_173530

noncomputable def S (n : ℕ) : ℤ := n^2 - 3 * n

theorem sequence_term (n : ℕ) (h : n ≥ 1) : 
  ∃ a : ℕ → ℤ, a n = 2 * n - 4 := 
  sorry

end sequence_term_l173_173530


namespace bullet_speed_difference_l173_173317

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end bullet_speed_difference_l173_173317


namespace books_ratio_l173_173808

theorem books_ratio (c e : ℕ) (h_ratio : c / e = 2 / 5) (h_sampled : c = 10) : e = 25 :=
by
  sorry

end books_ratio_l173_173808


namespace stream_speed_l173_173220

variables (v_s t_d t_u : ℝ)
variables (D : ℝ) -- Distance is not provided in the problem but assumed for formulation.

theorem stream_speed (h1 : t_u = 2 * t_d) (h2 : v_s = 54 + t_d / t_u) :
  v_s = 18 := 
by
  sorry

end stream_speed_l173_173220


namespace max_marks_l173_173641

theorem max_marks (M : ℝ) (h : 0.80 * M = 240) : M = 300 :=
sorry

end max_marks_l173_173641


namespace evaluate_polynomial_l173_173098

variable {x y : ℚ}

theorem evaluate_polynomial (h : x - 2 * y - 3 = -5) : 2 * y - x = 2 :=
by
  sorry

end evaluate_polynomial_l173_173098


namespace probability_letter_in_MATHEMATICS_l173_173305

theorem probability_letter_in_MATHEMATICS :
  let alphabet := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  let mathematics := ['M', 'A', 'T', 'H', 'E', 'I', 'C', 'S']
  (mathematics.length : ℚ) / (alphabet.length : ℚ) = 4 / 13 :=
by
  sorry

end probability_letter_in_MATHEMATICS_l173_173305


namespace sum_of_numbers_l173_173263

noncomputable def mean (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem sum_of_numbers (a b c : ℕ) (h1 : mean a b c = a + 8)
  (h2 : mean a b c = c - 20) (h3 : b = 7) (h_le1 : a ≤ b) (h_le2 : b ≤ c) :
  a + b + c = 57 :=
by {
  sorry
}

end sum_of_numbers_l173_173263


namespace move_left_is_negative_l173_173221

theorem move_left_is_negative (movement_right : ℝ) (h : movement_right = 3) : -movement_right = -3 := 
by 
  sorry

end move_left_is_negative_l173_173221


namespace line_l_passes_through_fixed_point_intersecting_lines_find_k_l173_173081

-- Define the lines
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0
def line_l1 (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0
def line_l2 (x y : ℝ) : Prop := x - y - 1 = 0

-- 1. Prove line l passes through the point (-2, 1)
theorem line_l_passes_through_fixed_point (k : ℝ) :
  line_l k (-2) 1 :=
by sorry

-- 2. Given lines l, l1, and l2 intersect at a single point, find k
theorem intersecting_lines_find_k (k : ℝ) :
  (∃ x y : ℝ, line_l k x y ∧ line_l1 x y ∧ line_l2 x y) ↔ k = -3 :=
by sorry

end line_l_passes_through_fixed_point_intersecting_lines_find_k_l173_173081


namespace number_of_cows_l173_173144

def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def chicken_cost : ℕ := 100 * 5
def installation_cost : ℕ := 6 * 100
def equipment_cost : ℕ := 6000
def total_cost : ℕ := 147700

theorem number_of_cows : 
  (total_cost - (land_cost + house_cost + chicken_cost + installation_cost + equipment_cost)) / 1000 = 20 := by
  sorry

end number_of_cows_l173_173144


namespace find_m_l173_173557

variables {m : ℝ}
def vec_a : ℝ × ℝ := (-2, 3)
def vec_b (m : ℝ) : ℝ × ℝ := (3, m)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (m : ℝ) (h : perpendicular vec_a (vec_b m)) : m = 2 :=
by
  sorry

end find_m_l173_173557


namespace unique_numbers_l173_173465

theorem unique_numbers (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (S : x + y = 17) 
  (Q : x^2 + y^2 = 145) 
  : x = 8 ∧ y = 9 ∨ x = 9 ∧ y = 8 :=
by
  sorry

end unique_numbers_l173_173465


namespace solve_for_y_l173_173132

-- Given conditions expressed as a Lean definition
def given_condition (y : ℝ) : Prop :=
  (y / 5) / 3 = 15 / (y / 3)

-- Prove the equivalent statement
theorem solve_for_y (y : ℝ) (h : given_condition y) : y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 :=
sorry

end solve_for_y_l173_173132


namespace min_value_2a_plus_b_l173_173772

theorem min_value_2a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (1/a) + (2/b) = 1): 2 * a + b = 8 :=
sorry

end min_value_2a_plus_b_l173_173772


namespace car_travel_distance_20_minutes_l173_173650

noncomputable def train_speed_in_mph : ℝ := 80
noncomputable def car_speed_ratio : ℝ := 3/4
noncomputable def car_speed_in_mph : ℝ := car_speed_ratio * train_speed_in_mph
noncomputable def travel_time_in_hours : ℝ := 20 / 60
noncomputable def distance_travelled_by_car : ℝ := car_speed_in_mph * travel_time_in_hours

theorem car_travel_distance_20_minutes : distance_travelled_by_car = 20 := 
by 
  sorry

end car_travel_distance_20_minutes_l173_173650


namespace zero_points_of_function_l173_173848

theorem zero_points_of_function : 
  (∃ x y : ℝ, y = x - 4 / x ∧ y = 0) → (∃! x : ℝ, x = -2 ∨ x = 2) :=
by
  sorry

end zero_points_of_function_l173_173848


namespace midpoint_coordinates_l173_173959

theorem midpoint_coordinates (A B M : ℝ × ℝ) (hx : A = (2, -4)) (hy : B = (-6, 2)) (hm : M = (-2, -1)) :
  let (x1, y1) := A
  let (x2, y2) := B
  M = ((x1 + x2) / 2, (y1 + y2) / 2) :=
  sorry

end midpoint_coordinates_l173_173959


namespace smallest_positive_integer_b_l173_173167
-- Import the necessary library

-- Define the conditions and problem statement
def smallest_b_factors (r s : ℤ) := r + s

theorem smallest_positive_integer_b :
  ∃ r s : ℤ, r * s = 1800 ∧ ∀ r' s' : ℤ, r' * s' = 1800 → smallest_b_factors r s ≤ smallest_b_factors r' s' :=
by
  -- Declare that the smallest positive integer b satisfying the conditions is 85
  use 45, 40
  -- Check the core condition
  have rs_eq_1800 := (45 * 40 = 1800)
  sorry

end smallest_positive_integer_b_l173_173167


namespace arithmetic_expression_l173_173099

theorem arithmetic_expression :
  (5^6) / (5^4) + 3^3 - 6^2 = 16 := by
  sorry

end arithmetic_expression_l173_173099


namespace correct_proposition_l173_173006

-- Defining the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Defining proposition p
def p : Prop := ∃ x : ℝ, 0 < x ∧ x < 2 ∧ f x < 0

-- Defining proposition q
def q : Prop := ∀ x y : ℝ, x + y > 4 → x > 2 ∧ y > 2

-- Theorem statement to prove the correct answer
theorem correct_proposition : (¬ p) ∧ (¬ q) :=
by
  sorry

end correct_proposition_l173_173006


namespace larger_number_is_588_l173_173934

theorem larger_number_is_588
  (A B hcf : ℕ)
  (lcm_factors : ℕ × ℕ)
  (hcf_condition : hcf = 42)
  (lcm_factors_condition : lcm_factors = (12, 14))
  (hcf_prop : Nat.gcd A B = hcf)
  (lcm_prop : Nat.lcm A B = hcf * lcm_factors.1 * lcm_factors.2) :
  max (A) (B) = 588 :=
by
  sorry

end larger_number_is_588_l173_173934


namespace final_result_is_110_l173_173152

def chosen_number : ℕ := 63
def multiplier : ℕ := 4
def subtracted_value : ℕ := 142

def final_result : ℕ := (chosen_number * multiplier) - subtracted_value

theorem final_result_is_110 : final_result = 110 := by
  sorry

end final_result_is_110_l173_173152


namespace sally_has_more_cards_l173_173972

def SallyInitial : ℕ := 27
def DanTotal : ℕ := 41
def SallyBought : ℕ := 20
def SallyTotal := SallyInitial + SallyBought

theorem sally_has_more_cards : SallyTotal - DanTotal = 6 := by
  sorry

end sally_has_more_cards_l173_173972


namespace power_inequality_l173_173729

theorem power_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ (3 / 4) + b ^ (3 / 4) + c ^ (3 / 4) > (a + b + c) ^ (3 / 4) :=
sorry

end power_inequality_l173_173729


namespace minimum_sum_am_gm_l173_173539

theorem minimum_sum_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b)) + (b / (6 * c)) + (c / (9 * a)) ≥ (1 / 2) :=
sorry

end minimum_sum_am_gm_l173_173539


namespace find_counterfeit_coins_l173_173662

structure Coins :=
  (a a₁ b b₁ c c₁ : ℝ)
  (genuine_weight : ℝ)
  (counterfeit_weight : ℝ)
  (a_is_genuine_or_counterfeit : a = genuine_weight ∨ a = counterfeit_weight)
  (a₁_is_genuine_or_counterfeit : a₁ = genuine_weight ∨ a₁ = counterfeit_weight)
  (b_is_genuine_or_counterfeit : b = genuine_weight ∨ b = counterfeit_weight)
  (b₁_is_genuine_or_counterfeit : b₁ = genuine_weight ∨ b₁ = counterfeit_weight)
  (c_is_genuine_or_counterfeit : c = genuine_weight ∨ c = counterfeit_weight)
  (c₁_is_genuine_or_counterfeit : c₁ = genuine_weight ∨ c₁ = counterfeit_weight)
  (counterfeit_pair_ends_unit_segment : (a = counterfeit_weight ∧ a₁ = counterfeit_weight) 
                                        ∨ (b = counterfeit_weight ∧ b₁ = counterfeit_weight)
                                        ∨ (c = counterfeit_weight ∧ c₁ = counterfeit_weight))

theorem find_counterfeit_coins (coins : Coins) : 
  (coins.a = coins.genuine_weight ∧ coins.b = coins.genuine_weight → coins.a₁ = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.a < coins.b → coins.a = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.b < coins.a → coins.b = coins.counterfeit_weight ∧ coins.a₁ = coins.counterfeit_weight) := 
by
  sorry

end find_counterfeit_coins_l173_173662


namespace smallest_pieces_to_remove_l173_173672

theorem smallest_pieces_to_remove 
  (total_fruit : ℕ)
  (friends : ℕ)
  (h_fruit : total_fruit = 30)
  (h_friends : friends = 4) 
  : ∃ k : ℕ, k = 2 ∧ ((total_fruit - k) % friends = 0) :=
sorry

end smallest_pieces_to_remove_l173_173672


namespace distance_from_B_l173_173320

theorem distance_from_B (s y : ℝ) 
  (h1 : s^2 = 12)
  (h2 : ∀y, (1 / 2) * y^2 = 12 - y^2)
  (h3 : y = 2 * Real.sqrt 2)
: Real.sqrt ((2 * Real.sqrt 2)^2 + (2 * Real.sqrt 2)^2) = 4 := by
  sorry

end distance_from_B_l173_173320


namespace problem_condition_l173_173893

theorem problem_condition (a : ℝ) (x : ℝ) (h_a : -1 ≤ a ∧ a ≤ 1) :
  (x^2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
sorry

end problem_condition_l173_173893


namespace prime_gt_three_times_n_l173_173835

def nth_prime (n : ℕ) : ℕ :=
  -- Define the nth prime function, can use mathlib functionality
  sorry

theorem prime_gt_three_times_n (n : ℕ) (h : 12 ≤ n) : nth_prime n > 3 * n :=
  sorry

end prime_gt_three_times_n_l173_173835


namespace min_m_quad_eq_integral_solutions_l173_173811

theorem min_m_quad_eq_integral_solutions :
  (∃ m : ℕ, (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42) ∧ m > 0) →
  (∃ m : ℕ, m = 130 ∧ (∀ x : ℤ, 10 * x ^ 2 - m * x + 420 = 0 → ∃ p q : ℤ, p + q = m / 10 ∧ p * q = 42)) :=
by
  sorry

end min_m_quad_eq_integral_solutions_l173_173811


namespace inequality_additive_l173_173372

variable {a b c d : ℝ}

theorem inequality_additive (h1 : a > b) (h2 : c > d) : a + c > b + d :=
by
  sorry

end inequality_additive_l173_173372


namespace intersection_complement_P_CUQ_l173_173181

universe U

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}
def CUQ : Set ℕ := U \ Q

theorem intersection_complement_P_CUQ : 
  (P ∩ CUQ) = {1, 2} :=
by 
  sorry

end intersection_complement_P_CUQ_l173_173181


namespace largest_n_satisfying_inequality_l173_173806

theorem largest_n_satisfying_inequality : 
  ∃ (n : ℕ), (∀ k : ℕ, (8 : ℚ) / 15 < n / (n + k) ∧ n / (n + k) < (7 : ℚ) / 13) ∧ 
  ∀ n' : ℕ, (∀ k : ℕ, (8 : ℚ) / 15 < n' / (n' + k) ∧ n' / (n' + k) < (7 : ℚ) / 13) → n' ≤ n :=
sorry

end largest_n_satisfying_inequality_l173_173806


namespace product_of_ab_l173_173260

theorem product_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 7) : a * b = -10 :=
by
  sorry

end product_of_ab_l173_173260


namespace candies_remaining_l173_173338

theorem candies_remaining 
    (red_candies : ℕ)
    (yellow_candies : ℕ)
    (blue_candies : ℕ)
    (yellow_condition : yellow_candies = 3 * red_candies - 20)
    (blue_condition : blue_candies = yellow_candies / 2)
    (initial_red_candies : red_candies = 40) :
    (red_candies + yellow_candies + blue_candies - yellow_candies) = 90 := 
by
  sorry

end candies_remaining_l173_173338


namespace petya_wins_prize_probability_atleast_one_wins_probability_l173_173785

/-- Petya and 9 other people each roll a fair six-sided die. 
    A player wins a prize if they roll a number that nobody else rolls more than once.-/
theorem petya_wins_prize_probability : (5 / 6) ^ 9 = 0.194 :=
sorry

/-- The probability that at least one player gets a prize in the game where Petya and
    9 others roll a fair six-sided die is 0.919. -/
theorem atleast_one_wins_probability : 1 - (1 / 6) ^ 9 = 0.919 :=
sorry

end petya_wins_prize_probability_atleast_one_wins_probability_l173_173785


namespace g_9_pow_4_l173_173698

theorem g_9_pow_4 (f g : ℝ → ℝ) (h1 : ∀ x ≥ 1, f (g x) = x^2) (h2 : ∀ x ≥ 1, g (f x) = x^4) (h3 : g 81 = 81) : (g 9)^4 = 81 :=
sorry

end g_9_pow_4_l173_173698


namespace naomi_stickers_l173_173547

theorem naomi_stickers :
  ∃ S : ℕ, S > 1 ∧
    (S % 5 = 2) ∧
    (S % 9 = 2) ∧
    (S % 11 = 2) ∧
    S = 497 :=
by
  sorry

end naomi_stickers_l173_173547


namespace arithmetic_sequence_common_difference_l173_173603

theorem arithmetic_sequence_common_difference
    (a : ℕ → ℝ)
    (h1 : a 2 + a 3 = 9)
    (h2 : a 4 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n + d) : d = 3 :=
        sorry

end arithmetic_sequence_common_difference_l173_173603


namespace average_marks_of_all_candidates_l173_173615

def n : ℕ := 120
def p : ℕ := 100
def f : ℕ := n - p
def A_p : ℕ := 39
def A_f : ℕ := 15
def total_marks : ℕ := p * A_p + f * A_f
def average_marks : ℚ := total_marks / n

theorem average_marks_of_all_candidates :
  average_marks = 35 := 
sorry

end average_marks_of_all_candidates_l173_173615


namespace maximize_x3y4_correct_l173_173863

noncomputable def maximize_x3y4 : ℝ × ℝ :=
  let x := 160 / 7
  let y := 120 / 7
  (x, y)

theorem maximize_x3y4_correct :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 40 ∧ (x, y) = maximize_x3y4 ∧ 
  ∀ (x' y' : ℝ), 0 < x' ∧ 0 < y' ∧ x' + y' = 40 → x ^ 3 * y ^ 4 ≥ x' ^ 3 * y' ^ 4 :=
by
  sorry

end maximize_x3y4_correct_l173_173863


namespace customers_who_did_not_tip_l173_173920

def total_customers := 10
def total_tips := 15
def tip_per_customer := 3

theorem customers_who_did_not_tip : total_customers - (total_tips / tip_per_customer) = 5 :=
by
  sorry

end customers_who_did_not_tip_l173_173920


namespace race_distance_l173_173903

theorem race_distance (x : ℝ) (D : ℝ) (vA vB : ℝ) (head_start win_margin : ℝ):
  vA = 5 * x →
  vB = 4 * x →
  head_start = 100 →
  win_margin = 200 →
  (D - win_margin) / vB = (D - head_start) / vA →
  D = 600 :=
by 
  sorry

end race_distance_l173_173903


namespace problem_equiv_proof_l173_173089

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Define the set A based on the given condition
def A : Set ℝ := { x | x^2 + x - 2 ≤ 0 }

-- Define the set B based on the given condition
def B : Set ℝ := { y | ∃ x : ℝ, x ∈ A ∧ y = Real.log (x + 3) / Real.log 2 }

-- Define the complement of B in the universal set U
def complement_B : Set ℝ := { y | y < 0 ∨ y ≥ 2 }

-- Define the set C that is the intersection of A and complement of B
def C : Set ℝ := A ∩ complement_B

-- State the theorem we need to prove
theorem problem_equiv_proof : C = { x | -2 ≤ x ∧ x < 0 } :=
sorry

end problem_equiv_proof_l173_173089


namespace find_y_l173_173397

theorem find_y (x y : ℕ) (h1 : x % y = 7) (h2 : (x : ℚ) / y = 86.1) (h3 : Nat.Prime (x + y)) : y = 70 :=
sorry

end find_y_l173_173397


namespace cross_section_area_l173_173623

open Real

theorem cross_section_area (b α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) :
  ∃ (area : ℝ), area = - (b^2 * cos α * tan β) / (2 * cos (3 * α)) :=
by
  sorry

end cross_section_area_l173_173623


namespace area_of_each_triangle_is_half_l173_173933

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def area (t : Triangle) : ℝ :=
  0.5 * |t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y)|

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 0 }
def C : Point := { x := 1, y := 1 }
def D : Point := { x := 0, y := 1 }
def K : Point := { x := 0.5, y := 1 }
def L : Point := { x := 0, y := 0.5 }
def M : Point := { x := 0.5, y := 0 }
def N : Point := { x := 1, y := 0.5 }

def AKB : Triangle := { p1 := A, p2 := K, p3 := B }
def BLC : Triangle := { p1 := B, p2 := L, p3 := C }
def CMD : Triangle := { p1 := C, p2 := M, p3 := D }
def DNA : Triangle := { p1 := D, p2 := N, p3 := A }

theorem area_of_each_triangle_is_half :
  area AKB = 0.5 ∧ area BLC = 0.5 ∧ area CMD = 0.5 ∧ area DNA = 0.5 := by sorry

end area_of_each_triangle_is_half_l173_173933


namespace find_f4_l173_173237

-- Let f be a function from ℝ to ℝ with the following properties:
variable (f : ℝ → ℝ)

-- 1. f(x + 1) is an odd function
axiom f_odd : ∀ x, f (-(x + 1)) = -f (x + 1)

-- 2. f(x - 1) is an even function
axiom f_even : ∀ x, f (-(x - 1)) = f (x - 1)

-- 3. f(0) = 2
axiom f_zero : f 0 = 2

-- Prove that f(4) = -2
theorem find_f4 : f 4 = -2 :=
by
  sorry

end find_f4_l173_173237


namespace map_scale_l173_173550

theorem map_scale (cm12_km90 : 12 * (1 / 90) = 1) : 20 * (90 / 12) = 150 :=
by
  sorry

end map_scale_l173_173550


namespace trig_expression_equality_l173_173573

theorem trig_expression_equality :
  (Real.tan (60 * Real.pi / 180) + 2 * Real.sin (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)) 
  = Real.sqrt 2 :=
by
  have h1 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := by sorry
  have h2 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  sorry

end trig_expression_equality_l173_173573


namespace pirate_treasure_probability_l173_173379

theorem pirate_treasure_probability :
  let p_treasure_no_traps := 1 / 3
  let p_traps_no_treasure := 1 / 6
  let p_neither := 1 / 2
  let choose_4_out_of_8 := 70
  let p_4_treasure_no_traps := (1 / 3) ^ 4
  let p_4_neither := (1 / 2) ^ 4
  choose_4_out_of_8 * p_4_treasure_no_traps * p_4_neither = 35 / 648 :=
by
  sorry

end pirate_treasure_probability_l173_173379


namespace total_peaches_l173_173755

theorem total_peaches (num_baskets num_red num_green : ℕ)
    (h1 : num_baskets = 11)
    (h2 : num_red = 10)
    (h3 : num_green = 18) : (num_red + num_green) * num_baskets = 308 := by
  sorry

end total_peaches_l173_173755


namespace question_inequality_l173_173164

theorem question_inequality (m : ℝ) :
  (∀ x : ℝ, ¬ (m * x ^ 2 - m * x - 1 ≥ 0)) ↔ (-4 < m ∧ m ≤ 0) :=
sorry

end question_inequality_l173_173164


namespace matrix_solution_l173_173341

variable {x : ℝ}

theorem matrix_solution (x: ℝ) :
  let M := (3*x) * (2*x + 1) - (1) * (2*x)
  M = 5 → (x = 5/6) ∨ (x = -1) :=
by
  sorry

end matrix_solution_l173_173341


namespace compute_expression_l173_173887

theorem compute_expression : 2 * (Real.sqrt 144)^2 = 288 := by
  sorry

end compute_expression_l173_173887


namespace tan_double_angle_l173_173337

theorem tan_double_angle (α : Real) (h1 : α > π ∧ α < 3 * π / 2) (h2 : Real.sin (π - α) = -3/5) :
  Real.tan (2 * α) = 24/7 := 
by
  sorry

end tan_double_angle_l173_173337


namespace function_properties_l173_173130

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - x^2)

theorem function_properties : 
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x < y) → f x > f y) :=
by
  sorry

end function_properties_l173_173130


namespace total_pies_sold_l173_173572

-- Defining the conditions
def pies_per_day : ℕ := 8
def days_in_week : ℕ := 7

-- Proving the question
theorem total_pies_sold : pies_per_day * days_in_week = 56 :=
by
  sorry

end total_pies_sold_l173_173572


namespace inequality_solution_l173_173524

theorem inequality_solution 
  (a x : ℝ) : 
  (a = 2 ∨ a = -2 → x > 1 / 4) ∧ 
  (a > 2 → x > 1 / (a + 2) ∨ x < 1 / (2 - a)) ∧ 
  (a < -2 → x < 1 / (a + 2) ∨ x > 1 / (2 - a)) ∧ 
  (-2 < a ∧ a < 2 → 1 / (a + 2) < x ∧ x < 1 / (2 - a)) 
  :=
by
  sorry

end inequality_solution_l173_173524


namespace remainder_is_v_l173_173613

theorem remainder_is_v (x y u v : ℤ) (hx : x > 0) (hy : y > 0)
  (hdiv : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + (2 * u + 1) * y) % y = v :=
by
  sorry

end remainder_is_v_l173_173613


namespace area_of_base_of_cone_l173_173912

theorem area_of_base_of_cone (semicircle_area : ℝ) (h1 : semicircle_area = 2 * Real.pi) : 
  ∃ (base_area : ℝ), base_area = Real.pi :=
by
  sorry

end area_of_base_of_cone_l173_173912


namespace general_integral_of_ODE_l173_173079

noncomputable def general_solution (x y : ℝ) (m C : ℝ) : Prop :=
  (x^2 * y - x - m) / (x^2 * y - x + m) = C * Real.exp (2 * m / x)

theorem general_integral_of_ODE (m : ℝ) (y : ℝ → ℝ) (C : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∀ (y' : ℝ → ℝ) (x : ℝ), deriv y x = m^2 / x^4 - (y x)^2) ∧ 
  (y 1 = 1 / x + m / x^2) ∧ 
  (y 2 = 1 / x - m / x^2) →
  general_solution x (y x) m C :=
by 
  sorry

end general_integral_of_ODE_l173_173079


namespace total_walnut_trees_in_park_l173_173646

theorem total_walnut_trees_in_park 
  (initial_trees planted_by_first planted_by_second planted_by_third removed_trees : ℕ)
  (h_initial : initial_trees = 22)
  (h_first : planted_by_first = 12)
  (h_second : planted_by_second = 15)
  (h_third : planted_by_third = 10)
  (h_removed : removed_trees = 4) :
  initial_trees + (planted_by_first + planted_by_second + planted_by_third - removed_trees) = 55 :=
by
  sorry

end total_walnut_trees_in_park_l173_173646


namespace prove_river_improvement_l173_173201

def river_improvement_equation (x : ℝ) : Prop :=
  4800 / x - 4800 / (x + 200) = 4

theorem prove_river_improvement (x : ℝ) (h : x > 0) : river_improvement_equation x := by
  sorry

end prove_river_improvement_l173_173201


namespace sara_disproves_tom_l173_173861

-- Define the type and predicate of cards
inductive Card
| K
| M
| card5
| card7
| card8

open Card

-- Define the conditions
def is_consonant : Card → Prop
| K => true
| M => true
| _ => false

def is_odd : Card → Prop
| card5 => true
| card7 => true
| _ => false

def is_even : Card → Prop
| card8 => true
| _ => false

-- Tom's statement
def toms_statement : Prop :=
  ∀ c, is_consonant c → is_odd c

-- The card Sara turns over (card8) to disprove Tom's statement
theorem sara_disproves_tom : is_even card8 ∧ is_consonant card8 → ¬toms_statement :=
by
  sorry

end sara_disproves_tom_l173_173861


namespace man_speed_against_current_l173_173656

-- Definitions for the problem conditions
def man_speed_with_current : ℝ := 21
def current_speed : ℝ := 4.3

-- Main proof statement
theorem man_speed_against_current : man_speed_with_current - 2 * current_speed = 12.4 :=
by
  sorry

end man_speed_against_current_l173_173656


namespace evaluate_expression_l173_173846

theorem evaluate_expression : 2009 * (2007 / 2008) + (1 / 2008) = 2008 := 
by 
  sorry

end evaluate_expression_l173_173846


namespace complement_intersection_l173_173744

noncomputable def U : Set Real := Set.univ
noncomputable def M : Set Real := { x : Real | Real.log x < 0 }
noncomputable def N : Set Real := { x : Real | (1 / 2) ^ x ≥ Real.sqrt (1 / 2) }

theorem complement_intersection (U M N : Set Real) : 
  (Set.compl M ∩ N) = Set.Iic 0 :=
by
  sorry

end complement_intersection_l173_173744


namespace not_divisible_by_5_for_4_and_7_l173_173215

-- Define a predicate that checks if a given number is not divisible by another number
def notDivisibleBy (n k : ℕ) : Prop := ¬ (n % k = 0)

-- Define the expression we are interested in
def expression (b : ℕ) : ℕ := 3 * b^3 - b^2 + b - 1

-- The theorem we want to prove
theorem not_divisible_by_5_for_4_and_7 :
  notDivisibleBy (expression 4) 5 ∧ notDivisibleBy (expression 7) 5 :=
by
  sorry

end not_divisible_by_5_for_4_and_7_l173_173215


namespace geometric_sequence_is_alternating_l173_173799

theorem geometric_sequence_is_alternating (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 + a 2 = -3 / 2)
  (h2 : a 4 + a 5 = 12)
  (hg : ∀ n, a (n + 1) = q * a n) :
  ∃ q, q < 0 ∧ ∀ n, a n * a (n + 1) ≤ 0 :=
by sorry

end geometric_sequence_is_alternating_l173_173799


namespace polygon_sides_l173_173633

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l173_173633


namespace calculate_X_l173_173324

theorem calculate_X
  (top_seg1 : ℕ) (top_seg2 : ℕ) (X : ℕ)
  (vert_seg : ℕ)
  (bottom_seg1 : ℕ) (bottom_seg2 : ℕ) (bottom_seg3 : ℕ)
  (h1 : top_seg1 = 3) (h2 : top_seg2 = 2)
  (h3 : vert_seg = 4)
  (h4 : bottom_seg1 = 4) (h5 : bottom_seg2 = 2) (h6 : bottom_seg3 = 5)
  (h_eq : 5 + X = 11) :
  X = 6 :=
by
  -- Proof is omitted as per instructions.
  sorry

end calculate_X_l173_173324


namespace find_values_l173_173597

theorem find_values (a b c : ℤ)
  (h1 : ∀ x, x^2 + 9 * x + 14 = (x + a) * (x + b))
  (h2 : ∀ x, x^2 + 4 * x - 21 = (x + b) * (x - c)) :
  a + b + c = 12 :=
sorry

end find_values_l173_173597


namespace bacteria_growth_time_l173_173881

-- Define the conditions and the final proof statement
theorem bacteria_growth_time (n0 n1 : ℕ) (t : ℕ) :
  (∀ (k : ℕ), k > 0 → n1 = n0 * 3 ^ k) →
  (∀ (h : ℕ), t = 5 * h) →
  n0 = 200 →
  n1 = 145800 →
  t = 30 :=
by
  sorry

end bacteria_growth_time_l173_173881


namespace volume_of_box_with_ratio_125_l173_173070

def volumes : Finset ℕ := {60, 80, 100, 120, 200}

theorem volume_of_box_with_ratio_125 : 80 ∈ volumes ∧ ∃ (x : ℕ), 10 * x^3 = 80 :=
by {
  -- Skipping the proof, as only the statement is required.
  sorry
}

end volume_of_box_with_ratio_125_l173_173070


namespace ball_radius_l173_173928

noncomputable def radius_of_ball (d h : ℝ) : ℝ :=
  let r := d / 2
  (325 / 20 : ℝ)

theorem ball_radius (d h : ℝ) (hd : d = 30) (hh : h = 10) :
  radius_of_ball d h = 16.25 := by
  sorry

end ball_radius_l173_173928


namespace new_sphere_radius_l173_173343

noncomputable def calculateVolume (R r : ℝ) : ℝ :=
  let originalSphereVolume := (4 / 3) * Real.pi * R^3
  let cylinderHeight := 2 * Real.sqrt (R^2 - r^2)
  let cylinderVolume := Real.pi * r^2 * cylinderHeight
  let capHeight := R - Real.sqrt (R^2 - r^2)
  let capVolume := (Real.pi * capHeight^2 * (3 * R - capHeight)) / 3
  let totalCapVolume := 2 * capVolume
  originalSphereVolume - cylinderVolume - totalCapVolume

theorem new_sphere_radius
  (R : ℝ) (r : ℝ) (h : ℝ) (new_sphere_radius : ℝ)
  (h_eq: h = 2 * Real.sqrt (R^2 - r^2))
  (new_sphere_volume_eq: calculateVolume R r = (4 / 3) * Real.pi * new_sphere_radius^3)
  : new_sphere_radius = 16 :=
sorry

end new_sphere_radius_l173_173343


namespace least_sum_possible_l173_173311

theorem least_sum_possible (x y z w k : ℕ) (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) 
  (hx : 4 * x = k) (hy : 5 * y = k) (hz : 6 * z = k) (hw : 7 * w = k) :
  x + y + z + w = 319 := 
  sorry

end least_sum_possible_l173_173311


namespace pairs_divisible_by_three_l173_173564

theorem pairs_divisible_by_three (P T : ℕ) (h : 5 * P = 3 * T) : ∃ k : ℕ, P = 3 * k := 
sorry

end pairs_divisible_by_three_l173_173564


namespace total_capacity_iv_bottle_l173_173899

-- Definitions of the conditions
def initial_volume : ℝ := 100 -- milliliters
def rate_of_flow : ℝ := 2.5 -- milliliters per minute
def observation_time : ℝ := 12 -- minutes
def empty_space_at_12_min : ℝ := 80 -- milliliters

-- Definition of the problem statement in Lean 4
theorem total_capacity_iv_bottle :
  initial_volume + rate_of_flow * observation_time + empty_space_at_12_min = 150 := 
by
  sorry

end total_capacity_iv_bottle_l173_173899


namespace infinite_integer_solutions_l173_173867

theorem infinite_integer_solutions 
  (a b c k D x0 y0 : ℤ) 
  (hD_pos : D = b^2 - 4 * a * c) 
  (hD_non_square : (∀ n : ℤ, D ≠ n^2)) 
  (hk_nonzero : k ≠ 0) 
  (h_initial_sol : a * x0^2 + b * x0 * y0 + c * y0^2 = k) :
  ∃ (X Y : ℤ), a * X^2 + b * X * Y + c * Y^2 = k ∧
  (∀ (m : ℕ), ∃ (Xm Ym : ℤ), a * Xm^2 + b * Xm * Ym + c * Ym^2 = k ∧
  (Xm, Ym) ≠ (x0, y0)) :=
sorry

end infinite_integer_solutions_l173_173867


namespace simplify_expression_l173_173037

variable (x : Int)

theorem simplify_expression : 3 * x + 5 * x + 7 * x = 15 * x :=
  by
  sorry

end simplify_expression_l173_173037


namespace area_of_square_with_diagonal_two_l173_173359

theorem area_of_square_with_diagonal_two {a d : ℝ} (h : d = 2) (h' : d = a * Real.sqrt 2) : a^2 = 2 := 
by
  sorry

end area_of_square_with_diagonal_two_l173_173359


namespace find_f_neg_two_l173_173392

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x^2 - 1 else sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)

axiom f_odd : is_odd_function f
axiom f_pos : ∀ x, x > 0 → f x = x^2 - 1

theorem find_f_neg_two : f (-2) = -3 :=
by
  sorry

end find_f_neg_two_l173_173392


namespace max_blue_cubes_visible_l173_173186

def max_visible_blue_cubes (board : ℕ × ℕ × ℕ → ℕ) : ℕ :=
  board (0, 0, 0)

theorem max_blue_cubes_visible (board : ℕ × ℕ × ℕ → ℕ) :
  max_visible_blue_cubes board = 12 :=
sorry

end max_blue_cubes_visible_l173_173186


namespace can_be_divided_into_two_triangles_l173_173523

-- Definitions and properties of geometrical shapes
def is_triangle (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 3 ∧ vertices = 3

def is_pentagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 5 ∧ vertices = 5

def is_hexagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 6 ∧ vertices = 6

def is_heptagon (sides : ℕ) (vertices : ℕ) : Prop :=
  sides = 7 ∧ vertices = 7

-- The theorem we need to prove
theorem can_be_divided_into_two_triangles :
  ∀ sides vertices,
  (is_pentagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_hexagon sides vertices → is_triangle sides vertices ∧ is_triangle sides vertices) ∧
  (is_heptagon sides vertices → ¬ (is_triangle sides vertices ∧ is_triangle sides vertices)) :=
by sorry

end can_be_divided_into_two_triangles_l173_173523


namespace polynomial_satisfies_conditions_l173_173945

noncomputable def f (x y z : ℝ) : ℝ := (x^2 - y^3) * (y^3 - z^6) * (z^6 - x^2)

theorem polynomial_satisfies_conditions :
  (∀ x y z : ℝ, f x (z^2) y + f x (y^2) z = 0) ∧ 
  (∀ x y z : ℝ, f (z^3) y x + f (x^3) y z = 0) :=
by
  sorry

end polynomial_satisfies_conditions_l173_173945


namespace price_of_coffee_table_l173_173473

-- Define the given values
def price_sofa : ℕ := 1250
def price_armchair : ℕ := 425
def num_armchairs : ℕ := 2
def total_invoice : ℕ := 2430

-- Define the target value (price of the coffee table)
def price_coffee_table : ℕ := 330

-- The theorem to prove
theorem price_of_coffee_table :
  total_invoice = price_sofa + num_armchairs * price_armchair + price_coffee_table :=
by sorry

end price_of_coffee_table_l173_173473


namespace largest_of_three_l173_173724

structure RealTriple (x y z : ℝ) where
  h1 : x + y + z = 3
  h2 : x * y + y * z + z * x = -8
  h3 : x * y * z = -18

theorem largest_of_three {x y z : ℝ} (h : RealTriple x y z) : max x (max y z) = Real.sqrt 5 :=
  sorry

end largest_of_three_l173_173724


namespace speed_of_other_person_l173_173756

-- Definitions related to the problem conditions
def pooja_speed : ℝ := 3  -- Pooja's speed in km/hr
def time : ℝ := 4  -- Time in hours
def distance : ℝ := 20  -- Distance between them after 4 hours in km

-- Define the unknown speed S as a parameter to be solved
variable (S : ℝ)

-- Define the relative speed when moving in opposite directions
def relative_speed (S : ℝ) : ℝ := S + pooja_speed

-- Create a theorem to encapsulate the problem and to be proved
theorem speed_of_other_person 
  (h : distance = relative_speed S * time) : S = 2 := 
  sorry

end speed_of_other_person_l173_173756


namespace cubed_ge_sqrt_ab_squared_l173_173754

theorem cubed_ge_sqrt_ab_squared (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  a^3 + b^3 ≥ (ab)^(1/2) * (a^2 + b^2) :=
sorry

end cubed_ge_sqrt_ab_squared_l173_173754


namespace monotone_on_interval_and_extreme_values_l173_173513

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem monotone_on_interval_and_extreme_values :
  (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2) → f x1 > f x2) ∧ (f 1 = 5 ∧ f 2 = 4) := 
by
  sorry

end monotone_on_interval_and_extreme_values_l173_173513


namespace y_increase_by_30_when_x_increases_by_12_l173_173676

theorem y_increase_by_30_when_x_increases_by_12
  (h : ∀ x y : ℝ, x = 4 → y = 10)
  (x_increase : ℝ := 12) :
  ∃ y_increase : ℝ, y_increase = 30 :=
by
  -- Here we assume the condition h and x_increase
  let ratio := 10 / 4  -- Establish the ratio of increase
  let expected_y_increase := x_increase * ratio
  exact ⟨expected_y_increase, sorry⟩  -- Prove it is 30

end y_increase_by_30_when_x_increases_by_12_l173_173676


namespace find_x_l173_173690

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 152) : x = 16 := 
by 
  sorry

end find_x_l173_173690


namespace angle_bisector_coordinates_distance_to_x_axis_l173_173828

structure Point where
  x : ℝ
  y : ℝ

def M (m : ℝ) : Point :=
  ⟨m - 1, 2 * m + 3⟩

theorem angle_bisector_coordinates (m : ℝ) :
  (M m = ⟨-5, -5⟩) ∨ (M m = ⟨-(5/3), 5/3⟩) := sorry

theorem distance_to_x_axis (m : ℝ) :
  (|2 * m + 3| = 1) → (M m = ⟨-2, 1⟩) ∨ (M m = ⟨-3, -1⟩) := sorry

end angle_bisector_coordinates_distance_to_x_axis_l173_173828


namespace temperature_difference_l173_173705

def highest_temperature : ℝ := 8
def lowest_temperature : ℝ := -1

theorem temperature_difference : highest_temperature - lowest_temperature = 9 := by
  sorry

end temperature_difference_l173_173705


namespace find_side_length_a_l173_173525

noncomputable def length_of_a (A B : ℝ) (b : ℝ) : ℝ :=
  b * Real.sin A / Real.sin B

theorem find_side_length_a :
  ∀ (a b c : ℝ) (A B C : ℝ),
  A = Real.pi / 3 → B = Real.pi / 4 → b = Real.sqrt 6 →
  a = length_of_a A B b →
  a = 3 :=
by
  intros a b c A B C hA hB hb ha
  rw [hA, hB, hb] at ha
  sorry

end find_side_length_a_l173_173525


namespace ott_fractional_part_l173_173823

theorem ott_fractional_part (M L N O x : ℝ)
  (hM : M = 6 * x)
  (hL : L = 5 * x)
  (hN : N = 4 * x)
  (hO : O = 0)
  (h_each : O + M + L + N = x + x + x) :
  (3 * x) / (M + L + N) = 1 / 5 :=
by
  sorry

end ott_fractional_part_l173_173823


namespace furniture_store_revenue_increase_l173_173939

noncomputable def percentage_increase_in_gross (P R : ℕ) : ℚ :=
  ((0.80 * P) * (1.70 * R) - (P * R)) / (P * R) * 100

theorem furniture_store_revenue_increase (P R : ℕ) :
  percentage_increase_in_gross P R = 36 := 
by
  -- We include the conditions directly in the proof.
  -- Follow theorem from the given solution.
  sorry

end furniture_store_revenue_increase_l173_173939


namespace minimum_value_of_function_l173_173628

theorem minimum_value_of_function (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  ∃ y : ℝ, (∀ z : ℝ, z = (1 / x) + (4 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end minimum_value_of_function_l173_173628


namespace age_difference_l173_173402

-- Defining the age variables as fractions
variables (x y : ℚ)

-- Given conditions
axiom ratio1 : 2 * x / y = 2 / y
axiom ratio2 : (5 * x + 20) / (y + 20) = 8 / 3

-- The main theorem to prove the difference between Mahesh's and Suresh's ages.
theorem age_difference : 5 * x - y = (125 / 8) := sorry

end age_difference_l173_173402


namespace total_distance_l173_173183

-- Definitions for the given problem conditions
def Beka_distance : ℕ := 873
def Jackson_distance : ℕ := 563
def Maria_distance : ℕ := 786

-- Theorem that needs to be proved
theorem total_distance : Beka_distance + Jackson_distance + Maria_distance = 2222 := by
  sorry

end total_distance_l173_173183


namespace border_area_l173_173124

theorem border_area (photo_height photo_width border_width : ℕ) (h1 : photo_height = 12) (h2 : photo_width = 16) (h3 : border_width = 3) : 
  let framed_height := photo_height + 2 * border_width 
  let framed_width := photo_width + 2 * border_width 
  let area_of_photo := photo_height * photo_width
  let area_of_framed := framed_height * framed_width 
  let area_of_border := area_of_framed - area_of_photo 
  area_of_border = 204 := 
by
  sorry

end border_area_l173_173124


namespace afternoon_sales_l173_173753

variable (x y : ℕ)

theorem afternoon_sales (hx : y = 2 * x) (hy : x + y = 390) : y = 260 := by
  sorry

end afternoon_sales_l173_173753


namespace problem1_problem2_l173_173591

-- Problem 1: Prove that the solutions of x^2 + 6x - 7 = 0 are x = -7 and x = 1
theorem problem1 (x : ℝ) : x^2 + 6*x - 7 = 0 ↔ (x = -7 ∨ x = 1) := by
  -- Proof omitted
  sorry

-- Problem 2: Prove that the solutions of 4x(2x+1) = 3(2x+1) are x = -1/2 and x = 3/4
theorem problem2 (x : ℝ) : 4*x*(2*x + 1) = 3*(2*x + 1) ↔ (x = -1/2 ∨ x = 3/4) := by
  -- Proof omitted
  sorry

end problem1_problem2_l173_173591


namespace proposition_not_true_at_9_l173_173503

variable {P : ℕ → Prop}

theorem proposition_not_true_at_9 (h : ∀ k : ℕ, k ≥ 1 → P k → P (k + 1)) (h10 : ¬P 10) : ¬P 9 :=
by
  sorry

end proposition_not_true_at_9_l173_173503


namespace infinite_div_pairs_l173_173347

theorem infinite_div_pairs {a : ℕ → ℕ} (h_seq : ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n ≤ 2001) :
  ∃ (s : ℕ → (ℕ × ℕ)), (∀ n, (s n).2 < (s n).1) ∧ (a ((s n).2) ∣ a ((s n).1)) :=
sorry

end infinite_div_pairs_l173_173347


namespace annual_interest_rate_l173_173515

theorem annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) 
  (hP : P = 5000) 
  (hA : A = 5202) 
  (hn : n = 4) 
  (ht : t = 1 / 2)
  (compound_interest : A = P * (1 + r / n)^ (n * t)) : 
  r = 0.080392 :=
by
  sorry

end annual_interest_rate_l173_173515


namespace non_neg_integer_solutions_l173_173691

theorem non_neg_integer_solutions (a b c : ℕ) :
  (∀ x : ℕ, x^2 - 2 * a * x + b = 0 → x ≥ 0) ∧ 
  (∀ y : ℕ, y^2 - 2 * b * y + c = 0 → y ≥ 0) ∧ 
  (∀ z : ℕ, z^2 - 2 * c * z + a = 0 → z ≥ 0) → 
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 0) :=
sorry

end non_neg_integer_solutions_l173_173691


namespace common_ratio_geometric_sequence_l173_173935

variable (a_n : ℕ → ℝ) (a1 : ℝ) (d : ℝ)

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a_n n = a1 + n * d

noncomputable def forms_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
(a_n 4) / (a_n 0) = (a_n 16) / (a_n 4)

theorem common_ratio_geometric_sequence :
  d ≠ 0 → 
  forms_geometric_sequence (a_n : ℕ → ℝ) →
  is_arithmetic_sequence a_n a1 d →
  ((a_n 4) / (a1) = 9) :=
by
  sorry

end common_ratio_geometric_sequence_l173_173935


namespace shaded_regions_area_l173_173878

/-- Given a grid of 1x1 squares with 2015 shaded regions where boundaries are either:
    - Horizontal line segments
    - Vertical line segments
    - Segments connecting the midpoints of adjacent sides of 1x1 squares
    - Diagonals of 1x1 squares

    Prove that the total area of these 2015 shaded regions is 47.5.
-/
theorem shaded_regions_area (n : ℕ) (h1 : n = 2015) : 
  ∃ (area : ℝ), area = 47.5 :=
by sorry

end shaded_regions_area_l173_173878


namespace measure_of_angle_Q_l173_173891

theorem measure_of_angle_Q (Q R : ℝ) 
  (h1 : Q = 2 * R)
  (h2 : 130 + 90 + 110 + 115 + Q + R = 540) :
  Q = 63.33 :=
by
  sorry

end measure_of_angle_Q_l173_173891


namespace inequality_proof_l173_173549

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) :
  (a / (b^2 * (c + 1))) + (b / (c^2 * (a + 1))) + (c / (a^2 * (b + 1))) ≥ 3 / 2 :=
sorry

end inequality_proof_l173_173549


namespace speed_rowing_upstream_l173_173819

theorem speed_rowing_upstream (V_m V_down : ℝ) (V_s V_up : ℝ)
  (h1 : V_m = 28) (h2 : V_down = 30) (h3 : V_down = V_m + V_s) (h4 : V_up = V_m - V_s) : 
  V_up = 26 :=
by
  sorry

end speed_rowing_upstream_l173_173819


namespace find_g_plus_h_l173_173600

theorem find_g_plus_h (g h : ℚ) (d : ℚ) 
  (h_prod : (7 * d^2 - 4 * d + g) * (3 * d^2 + h * d - 9) = 21 * d^4 - 49 * d^3 - 44 * d^2 + 17 * d - 24) :
  g + h = -107 / 24 :=
sorry

end find_g_plus_h_l173_173600


namespace smallest_integer_k_condition_l173_173504

theorem smallest_integer_k_condition :
  ∃ k : ℤ, k > 1 ∧ k % 12 = 1 ∧ k % 5 = 1 ∧ k % 3 = 1 ∧ k = 61 :=
by
  sorry

end smallest_integer_k_condition_l173_173504


namespace cube_sum_l173_173640

theorem cube_sum (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 41) : a^3 + b^3 = 598 :=
by
  sorry

end cube_sum_l173_173640


namespace tom_jerry_age_ratio_l173_173697

-- Definitions representing the conditions in the problem
variable (t j x : ℕ)

-- Condition 1: Three years ago, Tom was three times as old as Jerry
def condition1 : Prop := t - 3 = 3 * (j - 3)

-- Condition 2: Four years before that, Tom was five times as old as Jerry
def condition2 : Prop := t - 7 = 5 * (j - 7)

-- Question: In how many years will the ratio of their ages be 3:2,
-- asserting that the answer is 21
def ageRatioInYears : Prop := (t + x) / (j + x) = 3 / 2 → x = 21

-- The proposition we need to prove
theorem tom_jerry_age_ratio (h1 : condition1 t j) (h2 : condition2 t j) : ageRatioInYears t j x := 
  sorry
  
end tom_jerry_age_ratio_l173_173697


namespace find_a_l173_173764

theorem find_a (a b c d : ℕ) (h1 : 2 * a + 2 = b) (h2 : 2 * b + 2 = c) (h3 : 2 * c + 2 = d) (h4 : 2 * d + 2 = 62) : a = 2 :=
by
  sorry

end find_a_l173_173764


namespace dhoni_remaining_earnings_l173_173171

theorem dhoni_remaining_earnings (rent_percent dishwasher_percent : ℝ) 
  (h1 : rent_percent = 20) (h2 : dishwasher_percent = 15) : 
  100 - (rent_percent + dishwasher_percent) = 65 := 
by 
  sorry

end dhoni_remaining_earnings_l173_173171


namespace train_speed_l173_173820

theorem train_speed (L : ℝ) (T : ℝ) (hL : L = 200) (hT : T = 20) :
  L / T = 10 := by
  rw [hL, hT]
  norm_num
  done

end train_speed_l173_173820


namespace number_of_planks_needed_l173_173927

-- Definitions based on conditions
def bed_height : ℕ := 2
def bed_width : ℕ := 2
def bed_length : ℕ := 8
def plank_width : ℕ := 1
def lumber_length : ℕ := 8
def num_beds : ℕ := 10

-- The theorem statement
theorem number_of_planks_needed : (2 * (bed_length / lumber_length) * bed_height) + (2 * ((bed_width * bed_height) / lumber_length) * lumber_length / 4) * num_beds = 60 :=
  by sorry

end number_of_planks_needed_l173_173927


namespace product_of_three_numbers_l173_173204

theorem product_of_three_numbers 
  (a b c : ℕ) 
  (h1 : a + b + c = 300) 
  (h2 : 9 * a = b - 11) 
  (h3 : 9 * a = c + 15) : 
  a * b * c = 319760 := 
  sorry

end product_of_three_numbers_l173_173204


namespace minimum_groups_l173_173950

theorem minimum_groups (students : ℕ) (max_group_size : ℕ) (h_students : students = 30) (h_max_group_size : max_group_size = 12) : 
  ∃ least_groups : ℕ, least_groups = 3 :=
by
  sorry

end minimum_groups_l173_173950


namespace exists_non_regular_triangle_with_similar_medians_as_sides_l173_173295

theorem exists_non_regular_triangle_with_similar_medians_as_sides 
  (a b c : ℝ) 
  (s_a s_b s_c : ℝ)
  (h1 : 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h2 : 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h3 : 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2)
  (similarity_cond : (2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (∃ (s_a s_b s_c : ℝ), 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2 ∧ 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2 ∧ 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2) ∧
  ((2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :=
sorry

end exists_non_regular_triangle_with_similar_medians_as_sides_l173_173295


namespace p_plus_q_l173_173017

-- Define the circles w1 and w2
def circle1 (x y : ℝ) := x^2 + y^2 + 10*x - 20*y - 77 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 10*x - 20*y + 193 = 0

-- Define the line condition
def line (a x y : ℝ) := y = a * x

-- Prove that p + q = 85, where m^2 = p / q and m is the smallest positive a
theorem p_plus_q : ∃ p q : ℕ, (p.gcd q = 1) ∧ (m^2 = (p : ℝ)/(q : ℝ)) ∧ (p + q = 85) :=
  sorry

end p_plus_q_l173_173017


namespace geometric_seq_arithmetic_example_l173_173888

noncomputable def a_n (n : ℕ) (q : ℝ) : ℝ :=
if n = 0 then 1 else q ^ n

theorem geometric_seq_arithmetic_example {q : ℝ} (h₀ : q ≠ 0)
    (h₁ : ∀ n : ℕ, a_n 0 q = 1)
    (h₂ : 2 * (2 * (q ^ 2)) = 3 * q) :
    (q + q^2 + (q^3)) = 14 :=
by sorry

end geometric_seq_arithmetic_example_l173_173888


namespace initial_sheep_count_l173_173219

theorem initial_sheep_count 
    (S : ℕ)
    (initial_horses : ℕ := 100)
    (initial_chickens : ℕ := 9)
    (gifted_goats : ℕ := 37)
    (male_animals : ℕ := 53)
    (total_animals_half : ℕ := 106) :
    ((initial_horses + S + initial_chickens) / 2 + gifted_goats = total_animals_half) → 
    S = 29 :=
by
  intro h
  sorry

end initial_sheep_count_l173_173219


namespace water_left_in_bucket_l173_173229

theorem water_left_in_bucket :
  ∀ (original_poured water_left : ℝ),
    original_poured = 0.8 →
    water_left = 0.6 →
    ∃ (poured : ℝ), poured = 0.2 ∧ original_poured - poured = water_left :=
by
  intros original_poured water_left ho hw
  apply Exists.intro 0.2
  simp [ho, hw]
  sorry

end water_left_in_bucket_l173_173229


namespace gcd_f_l173_173500

def f (x: ℤ) : ℤ := x^2 - x + 2023

theorem gcd_f (x y : ℤ) (hx : x = 105) (hy : y = 106) : Int.gcd (f x) (f y) = 7 := by
  sorry

end gcd_f_l173_173500


namespace sum_base9_to_base9_eq_l173_173228

-- Definition of base 9 numbers
def base9_to_base10 (n : ℕ) : ℕ :=
  let digit1 := n % 10
  let digit2 := (n / 10) % 10
  let digit3 := (n / 100) % 10
  digit1 + 9 * digit2 + 81 * digit3

-- Definition of base 10 to base 9 conversion
def base10_to_base9 (n : ℕ) : ℕ :=
  let digit1 := n % 9
  let digit2 := (n / 9) % 9
  let digit3 := (n / 81) % 9
  digit1 + 10 * digit2 + 100 * digit3

-- The theorem to prove
theorem sum_base9_to_base9_eq :
  let x := base9_to_base10 236
  let y := base9_to_base10 327
  let z := base9_to_base10 284
  base10_to_base9 (x + y + z) = 858 :=
by {
  sorry
}

end sum_base9_to_base9_eq_l173_173228


namespace handshake_count_l173_173080

theorem handshake_count (n_total n_group1 n_group2 : ℕ) 
  (h_total : n_total = 40) (h_group1 : n_group1 = 25) (h_group2 : n_group2 = 15) 
  (h_sum : n_group1 + n_group2 = n_total) : 
  (15 * 39) / 2 = 292 := 
by sorry

end handshake_count_l173_173080


namespace vegetables_sold_mass_l173_173312

/-- Define the masses of the vegetables --/
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8

/-- Define the total mass of installed vegetables --/
def total_mass : ℕ := mass_carrots + mass_zucchini + mass_broccoli

/-- Define the mass of vegetables sold (half of the total mass) --/
def mass_sold : ℕ := total_mass / 2

/-- Prove that the mass of vegetables sold is 18 kg --/
theorem vegetables_sold_mass : mass_sold = 18 := by
  sorry

end vegetables_sold_mass_l173_173312


namespace find_a_odd_function_l173_173365

theorem find_a_odd_function (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, 0 < x → f x = 1 + a^x)
  (h3 : 0 < a)
  (h4 : a ≠ 1)
  (h5 : f (-1) = -3 / 2) :
  a = 1 / 2 :=
by
  sorry

end find_a_odd_function_l173_173365


namespace evaluate_complex_expression_l173_173789

noncomputable def expression := 
  Complex.mk (-1) (Real.sqrt 3) / 2

noncomputable def conjugate_expression := 
  Complex.mk (-1) (-Real.sqrt 3) / 2

theorem evaluate_complex_expression :
  (expression ^ 12 + conjugate_expression ^ 12) = 2 := by
  sorry

end evaluate_complex_expression_l173_173789


namespace wall_number_of_bricks_l173_173699

theorem wall_number_of_bricks (x : ℝ) :
  (∃ x, 6 * ((x / 7) + (x / 11) - 12) = x) →  x = 179 :=
by
  sorry

end wall_number_of_bricks_l173_173699


namespace quiz_scores_dropped_students_l173_173418

theorem quiz_scores_dropped_students (T S : ℝ) :
  T = 30 * 60.25 →
  T - S = 26 * 63.75 →
  S = 150 :=
by
  intros hT h_rem
  -- Additional steps would be implemented here.
  sorry

end quiz_scores_dropped_students_l173_173418


namespace shakes_sold_l173_173252

variable (s : ℕ) -- the number of shakes sold

-- conditions
def shakes_ounces := 4 * s
def cone_ounces := 6
def total_ounces := 14

-- the theorem to prove
theorem shakes_sold : shakes_ounces + cone_ounces = total_ounces → s = 2 := by
  intros h
  -- proof can be filled in here
  sorry

end shakes_sold_l173_173252


namespace chiming_time_is_5_l173_173957

-- Define the conditions for the clocks
def queen_strikes (h : ℕ) : Prop := (2 * h) % 3 = 0
def king_strikes (h : ℕ) : Prop := (3 * h) % 2 = 0

-- Define the chiming synchronization at the same time condition
def chiming_synchronization (h: ℕ) : Prop :=
  3 * h = 2 * ((2 * h) + 2)

-- The proof statement
theorem chiming_time_is_5 : ∃ h: ℕ, queen_strikes h ∧ king_strikes h ∧ chiming_synchronization h ∧ h = 5 :=
by
  sorry

end chiming_time_is_5_l173_173957


namespace cos_alpha_plus_two_pi_over_three_l173_173565

theorem cos_alpha_plus_two_pi_over_three (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.cos (α + 2 * π / 3) = -1 / 3 :=
by
  sorry

end cos_alpha_plus_two_pi_over_three_l173_173565


namespace angle_C_max_perimeter_l173_173222

def triangle_ABC (A B C a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def circumradius_2 (r : ℝ) : Prop :=
  r = 2

def satisfies_condition (a b c A B C : ℝ) : Prop :=
  (a - c)*(Real.sin A + Real.sin C) = b*(Real.sin A - Real.sin B)

theorem angle_C (A B C a b c : ℝ) (h₁ : triangle_ABC A B C a b c) 
                 (h₂ : satisfies_condition a b c A B C)
                 (h₃ : circumradius_2 (2 : ℝ)) : 
  C = Real.pi / 3 :=
sorry

theorem max_perimeter (A B C a b c r : ℝ) (h₁ : triangle_ABC A B C a b c)
                      (h₂ : satisfies_condition a b c A B C)
                      (h₃ : circumradius_2 r) : 
  4 * Real.sqrt 3 + 2 * Real.sqrt 3 = 6 * Real.sqrt 3 :=
sorry

end angle_C_max_perimeter_l173_173222


namespace parallel_line_distance_equation_l173_173658

theorem parallel_line_distance_equation :
  ∃ m : ℝ, (m = -20 ∨ m = 32) ∧
  ∀ x y : ℝ, (5 * x - 12 * y + 6 = 0) → 
            (5 * x - 12 * y + m = 0) :=
by
  sorry

end parallel_line_distance_equation_l173_173658


namespace tip_calculation_correct_l173_173223

noncomputable def calculate_tip (total_with_tax : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let bill_before_tax := total_with_tax / (1 + tax_rate)
  bill_before_tax * tip_rate

theorem tip_calculation_correct :
  calculate_tip 226 0.13 0.15 = 30 := 
by
  sorry

end tip_calculation_correct_l173_173223


namespace John_distance_proof_l173_173241

def initial_running_time : ℝ := 8
def increase_percentage : ℝ := 0.75
def initial_speed : ℝ := 8
def speed_increase : ℝ := 4

theorem John_distance_proof : 
  (initial_running_time + initial_running_time * increase_percentage) * (initial_speed + speed_increase) = 168 := 
by
  -- Proof can be completed here
  sorry

end John_distance_proof_l173_173241


namespace conformal_2z_conformal_z_minus_2_squared_l173_173575

-- For the function w = 2z
theorem conformal_2z :
  ∀ z : ℂ, true :=
by
  intro z
  sorry

-- For the function w = (z-2)^2
theorem conformal_z_minus_2_squared :
  ∀ z : ℂ, z ≠ 2 → true :=
by
  intro z h
  sorry

end conformal_2z_conformal_z_minus_2_squared_l173_173575


namespace coins_remainder_l173_173208

theorem coins_remainder (n : ℕ) (h1 : n % 8 = 6) (h2 : n % 7 = 5) : 
  (∃ m : ℕ, (n = m * 9)) :=
sorry

end coins_remainder_l173_173208


namespace soccer_ball_selling_price_l173_173594

theorem soccer_ball_selling_price
  (cost_price_per_ball : ℕ)
  (num_balls : ℕ)
  (total_profit : ℕ)
  (h_cost_price : cost_price_per_ball = 60)
  (h_num_balls : num_balls = 50)
  (h_total_profit : total_profit = 1950) :
  (cost_price_per_ball + (total_profit / num_balls) = 99) :=
by 
  -- Note: Proof can be filled here
  sorry

end soccer_ball_selling_price_l173_173594


namespace intersection_point_l173_173043

structure Point3D : Type where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨8, -9, 5⟩
def B : Point3D := ⟨18, -19, 15⟩
def C : Point3D := ⟨2, 5, -8⟩
def D : Point3D := ⟨4, -3, 12⟩

/-- Prove that the intersection point of lines AB and CD is (16, -19, 13) -/
theorem intersection_point :
  ∃ (P : Point3D), 
  (∃ t : ℝ, P = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩) ∧
  (∃ s : ℝ, P = ⟨C.x + s * (D.x - C.x), C.y + s * (D.y - C.y), C.z + s * (D.z - C.z)⟩) ∧
  P = ⟨16, -19, 13⟩ :=
by
  sorry

end intersection_point_l173_173043


namespace smallest_k_divides_l173_173787

-- Given Problem: z^{12} + z^{11} + z^8 + z^7 + z^6 + z^3 + 1 divides z^k - 1
theorem smallest_k_divides (
  k : ℕ
) : (∀ z : ℂ, (z ^ 12 + z ^ 11 + z ^ 8 + z ^ 7 + z ^ 6 + z ^ 3 + 1) ∣ (z ^ k - 1) ↔ k = 182) :=
sorry

end smallest_k_divides_l173_173787


namespace ratio_rate_down_to_up_l173_173210

noncomputable def rate_up (r_up t_up: ℕ) : ℕ := r_up * t_up
noncomputable def rate_down (d_down t_down: ℕ) : ℕ := d_down / t_down
noncomputable def ratio (r_down r_up: ℕ) : ℚ := r_down / r_up

theorem ratio_rate_down_to_up :
  let r_up := 6
  let t_up := 2
  let d_down := 18
  let t_down := 2
  rate_up 6 2 = 12 ∧ rate_down 18 2 = 9 ∧ ratio 9 6 = 3 / 2 :=
by
  sorry

end ratio_rate_down_to_up_l173_173210


namespace celsius_to_fahrenheit_l173_173760

theorem celsius_to_fahrenheit (C F : ℤ) (h1 : C = 50) (h2 : C = 5 / 9 * (F - 32)) : F = 122 :=
by
  sorry

end celsius_to_fahrenheit_l173_173760


namespace wario_missed_field_goals_wide_right_l173_173707

theorem wario_missed_field_goals_wide_right :
  ∀ (attempts missed_fraction wide_right_fraction : ℕ), 
  attempts = 60 →
  missed_fraction = 1 / 4 →
  wide_right_fraction = 20 / 100 →
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  wide_right = 3 :=
by
  intros attempts missed_fraction wide_right_fraction h1 h2 h3
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  sorry

end wario_missed_field_goals_wide_right_l173_173707


namespace moderate_intensity_pushups_l173_173194

theorem moderate_intensity_pushups :
  let normal_heart_rate := 80
  let k := 7
  let y (x : ℕ) := 80 * (Real.log (Real.sqrt (x / 12)) + 1)
  let t (x : ℕ) := y x / normal_heart_rate
  let f (t : ℝ) := k * Real.exp t
  28 ≤ f (Real.log (Real.sqrt 3)) + 1 ∧ f (Real.log (Real.sqrt 3)) + 1 ≤ 34 :=
sorry

end moderate_intensity_pushups_l173_173194


namespace inequality_y_lt_x_div_4_l173_173298

open Real

/-- Problem statement:
Given x ∈ (0, π / 6) and y ∈ (0, π / 6), and x * tan y = 2 * (1 - cos x),
prove that y < x / 4.
-/
theorem inequality_y_lt_x_div_4
  (x y : ℝ)
  (hx : 0 < x ∧ x < π / 6)
  (hy : 0 < y ∧ y < π / 6)
  (h : x * tan y = 2 * (1 - cos x)) :
  y < x / 4 := sorry

end inequality_y_lt_x_div_4_l173_173298


namespace simplify_expr_l173_173666

variable (a b c : ℤ)

theorem simplify_expr :
  (15 * a + 45 * b + 20 * c) + (25 * a - 35 * b - 10 * c) - (10 * a + 55 * b + 30 * c) = 30 * a - 45 * b - 20 * c := 
by
  sorry

end simplify_expr_l173_173666


namespace smallest_b_for_no_real_root_l173_173721

theorem smallest_b_for_no_real_root :
  ∃ b : ℤ, (b < 8 ∧ b > -8) ∧ (∀ x : ℝ, x^2 + (b : ℝ) * x + 10 ≠ -6) ∧ (b = -7) :=
by
  sorry

end smallest_b_for_no_real_root_l173_173721


namespace sum_of_remainders_is_six_l173_173384

theorem sum_of_remainders_is_six (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
  (a + b + c) % 15 = 6 :=
by
  sorry

end sum_of_remainders_is_six_l173_173384


namespace roots_nonpositive_if_ac_le_zero_l173_173634

theorem roots_nonpositive_if_ac_le_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a * c ≤ 0) :
  ¬ (∀ x : ℝ, x^2 - (b/a)*x + (c/a) = 0 → x > 0) :=
sorry

end roots_nonpositive_if_ac_le_zero_l173_173634


namespace no_extreme_value_at_5_20_l173_173407

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 4 * x ^ 2 - k * x - 8

theorem no_extreme_value_at_5_20 (k : ℝ) :
  ¬ (∃ (c : ℝ), (forall (x : ℝ), f k x = f k c + (4 * (x - c) ^ 2 - 8 - 20)) ∧ c = 5) ↔ (k ≤ 40 ∨ k ≥ 160) := sorry

end no_extreme_value_at_5_20_l173_173407


namespace solve_expression_l173_173292

theorem solve_expression (a x : ℝ) (h1 : a ≠ 0) (h2 : x ≠ a) : 
  (a / (2 * a + x) - x / (a - x)) / (x / (2 * a + x) + a / (a - x)) = -1 → 
  x = a / 2 :=
by
  sorry

end solve_expression_l173_173292


namespace steak_entree_cost_l173_173256

theorem steak_entree_cost
  (total_guests : ℕ)
  (steak_factor : ℕ)
  (chicken_entree_cost : ℕ)
  (total_budget : ℕ)
  (H1 : total_guests = 80)
  (H2 : steak_factor = 3)
  (H3 : chicken_entree_cost = 18)
  (H4 : total_budget = 1860) :
  ∃ S : ℕ, S = 25 := by
  -- Proof steps omitted
  sorry

end steak_entree_cost_l173_173256


namespace solve_ellipse_correct_m_l173_173533

noncomputable def ellipse_is_correct_m : Prop :=
  ∃ (m : ℝ), 
    (m > 6) ∧
    ((m - 2) - (10 - m) = 4) ∧
    (m = 8)

theorem solve_ellipse_correct_m : ellipse_is_correct_m :=
sorry

end solve_ellipse_correct_m_l173_173533


namespace amount_received_from_mom_l173_173234

-- Defining the problem conditions
def receives_from_dad : ℕ := 5
def spends : ℕ := 4
def has_more_from_mom_after_spending (M : ℕ) : Prop := 
  (receives_from_dad + M - spends = receives_from_dad + 2)

-- Lean theorem statement
theorem amount_received_from_mom (M : ℕ) (h : has_more_from_mom_after_spending M) : M = 6 := 
by
  sorry

end amount_received_from_mom_l173_173234


namespace third_number_l173_173042

theorem third_number (x : ℝ) 
    (h : 217 + 2.017 + 2.0017 + x = 221.2357) : 
    x = 0.217 :=
sorry

end third_number_l173_173042


namespace rectangle_diagonal_length_l173_173955

theorem rectangle_diagonal_length (l : ℝ) (L W d : ℝ) 
  (h_ratio : L = 5 * l ∧ W = 2 * l)
  (h_perimeter : 2 * (L + W) = 100) :
  d = (5 * Real.sqrt 290) / 7 :=
by
  sorry

end rectangle_diagonal_length_l173_173955


namespace expression_simplifies_to_49_l173_173351

theorem expression_simplifies_to_49 (x : ℝ) : 
  (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end expression_simplifies_to_49_l173_173351


namespace ratio_pen_to_pencil_l173_173712

-- Define the costs
def cost_of_pencil (P : ℝ) : ℝ := P
def cost_of_pen (P : ℝ) : ℝ := 4 * P
def total_cost (P : ℝ) : ℝ := cost_of_pencil P + cost_of_pen P

-- The proof that the total cost of the pen and pencil is $6 given the provided ratio
theorem ratio_pen_to_pencil (P : ℝ) (h_total_cost : total_cost P = 6) (h_pen_cost : cost_of_pen P = 4) :
  cost_of_pen P / cost_of_pencil P = 4 :=
by
  -- Proof skipped
  sorry

end ratio_pen_to_pencil_l173_173712


namespace find_y_l173_173334

theorem find_y (x y: ℝ) (h1: x = 680) (h2: 0.25 * x = 0.20 * y - 30) : y = 1000 :=
by 
  sorry

end find_y_l173_173334


namespace remainder_of_large_number_l173_173119

theorem remainder_of_large_number :
  ∀ (N : ℕ), N = 123456789012 →
    (N % 4 = 0) →
    (N % 9 = 3) →
    (N % 7 = 1) →
    N % 252 = 156 :=
by
  intros N hN h4 h9 h7
  sorry

end remainder_of_large_number_l173_173119


namespace ellipse_parabola_intersection_l173_173916

open Real

theorem ellipse_parabola_intersection (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 2*y) ↔ (-1 ≤ a ∧ a ≤ 17 / 8) :=
by
  sorry

end ellipse_parabola_intersection_l173_173916


namespace cos_C_value_l173_173598

-- Definitions for the perimeter and sine ratios
variable (A B C : ℝ)
variable (a b c : ℝ)
variable (perimeter : ℝ) (sin_ratio_A sin_ratio_B sin_ratio_C : ℚ)

-- Given conditions
axiom perimeter_condition : perimeter = a + b + c
axiom sine_ratio_condition : (sin_ratio_A / sin_ratio_B / sin_ratio_C) = (3 / 2 / 4)
axiom side_lengths : a = 3 ∧ b = 2 ∧ c = 4

-- To prove

theorem cos_C_value (h1 : sine_ratio_A = 3) (h2 : sine_ratio_B = 2) (h3 : sin_ratio_C = 4) :
  (3^2 + 2^2 - 4^2) / (2 * 3 * 2) = -1 / 4 :=
sorry

end cos_C_value_l173_173598


namespace bowling_average_decrease_l173_173000

theorem bowling_average_decrease 
  (original_average : ℚ) 
  (wickets_last_match : ℚ) 
  (runs_last_match : ℚ) 
  (original_wickets : ℚ) 
  (original_total_runs : ℚ := original_wickets * original_average) 
  (new_total_wickets : ℚ := original_wickets + wickets_last_match) 
  (new_total_runs : ℚ := original_total_runs + runs_last_match)
  (new_average : ℚ := new_total_runs / new_total_wickets) :
  original_wickets = 85 → original_average = 12.4 → wickets_last_match = 5 → runs_last_match = 26 → new_average = 12 →
  original_average - new_average = 0.4 := 
by 
  intros 
  sorry

end bowling_average_decrease_l173_173000


namespace appropriate_sampling_method_l173_173405

-- Definitions and conditions
def total_products : ℕ := 40
def first_class_products : ℕ := 10
def second_class_products : ℕ := 25
def defective_products : ℕ := 5
def samples_needed : ℕ := 8

-- Theorem statement
theorem appropriate_sampling_method : 
  (first_class_products + second_class_products + defective_products = total_products) ∧ 
  (2 ≤ first_class_products ∧ 2 ≤ second_class_products ∧ 1 ≤ defective_products) → 
  "Stratified Sampling" = "The appropriate sampling method for quality analysis" :=
  sorry

end appropriate_sampling_method_l173_173405


namespace attendees_count_l173_173015

def n_students_seated : ℕ := 300
def n_students_standing : ℕ := 25
def n_teachers_seated : ℕ := 30

def total_attendees : ℕ :=
  n_students_seated + n_students_standing + n_teachers_seated

theorem attendees_count :
  total_attendees = 355 := by
  sorry

end attendees_count_l173_173015


namespace solve_for_x_l173_173884

theorem solve_for_x (x : ℝ) : (2 / 7) * (1 / 4) * x - 3 = 5 → x = 112 := by
  sorry

end solve_for_x_l173_173884


namespace mb_range_l173_173213
-- Define the slope m and y-intercept b
def m : ℚ := 2 / 3
def b : ℚ := -1 / 2

-- Define the product mb
def mb : ℚ := m * b

-- Prove the range of mb
theorem mb_range : -1 < mb ∧ mb < 0 := by
  unfold mb
  sorry

end mb_range_l173_173213


namespace planes_parallel_l173_173521

variables (α β : Type)
variables (n : ℝ → ℝ → ℝ → Prop) (u v : ℝ × ℝ × ℝ)

-- Conditions: 
def normal_vector_plane_alpha (u : ℝ × ℝ × ℝ) := u = (1, 2, -1)
def normal_vector_plane_beta (v : ℝ × ℝ × ℝ) := v = (-3, -6, 3)

-- Proof Problem: Prove that alpha is parallel to beta
theorem planes_parallel (h1 : normal_vector_plane_alpha u)
                        (h2 : normal_vector_plane_beta v) :
  v = -3 • u :=
by sorry

end planes_parallel_l173_173521


namespace find_cost_per_pound_of_mixture_l173_173206

-- Problem Definitions and Conditions
variable (x : ℝ) -- the variable x represents the pounds of Spanish peanuts used
variable (y : ℝ) -- the cost per pound of the mixture we're trying to find
def cost_virginia_pound : ℝ := 3.50
def cost_spanish_pound : ℝ := 3.00
def weight_virginia : ℝ := 10.0

-- Formula for the cost per pound of the mixture
noncomputable def cost_per_pound_of_mixture : ℝ := (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x)

-- Proof Problem Statement
theorem find_cost_per_pound_of_mixture (h : cost_per_pound_of_mixture x = y) : 
  y = (weight_virginia * cost_virginia_pound + x * cost_spanish_pound) / (weight_virginia + x) := sorry

end find_cost_per_pound_of_mixture_l173_173206


namespace array_sum_remainder_mod_9_l173_173200

theorem array_sum_remainder_mod_9 :
  let sum_terms := ∑' r : ℕ, ∑' c : ℕ, (1 / (4 ^ r)) * (1 / (9 ^ c))
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ sum_terms = m / n ∧ (m + n) % 9 = 5 :=
by
  sorry

end array_sum_remainder_mod_9_l173_173200


namespace total_cost_correct_l173_173278

-- Definitions for the costs of items.
def sandwich_cost : ℝ := 3.49
def soda_cost : ℝ := 0.87

-- Definitions for the quantities.
def num_sandwiches : ℝ := 2
def num_sodas : ℝ := 4

-- The calculation for the total cost.
def total_cost : ℝ := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The claim that needs to be proved.
theorem total_cost_correct : total_cost = 10.46 := by
  sorry

end total_cost_correct_l173_173278


namespace actual_diameter_layer_3_is_20_micrometers_l173_173387

noncomputable def magnified_diameter_to_actual (magnified_diameter_cm : ℕ) (magnification_factor : ℕ) : ℕ :=
  (magnified_diameter_cm * 10000) / magnification_factor

def layer_3_magnified_diameter_cm : ℕ := 3
def layer_3_magnification_factor : ℕ := 1500

theorem actual_diameter_layer_3_is_20_micrometers :
  magnified_diameter_to_actual layer_3_magnified_diameter_cm layer_3_magnification_factor = 20 :=
by
  sorry

end actual_diameter_layer_3_is_20_micrometers_l173_173387


namespace value_of_x_l173_173728

theorem value_of_x (x : ℝ) :
  (x^2 - 1 + (x - 1) * I = 0 ∨ x^2 - 1 = 0 ∧ x - 1 ≠ 0) → x = -1 :=
by
  sorry

end value_of_x_l173_173728


namespace molecular_weight_of_3_moles_CaOH2_is_correct_l173_173731

-- Define the atomic weights as given by the conditions
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Define the molecular formula contributions for Ca(OH)2
def molecular_weight_CaOH2 : ℝ :=
  atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H

-- Define the weight of 3 moles of Ca(OH)2 based on the molecular weight
def weight_of_3_moles_CaOH2 : ℝ :=
  3 * molecular_weight_CaOH2

-- Theorem to prove the final result
theorem molecular_weight_of_3_moles_CaOH2_is_correct :
  weight_of_3_moles_CaOH2 = 222.30 := by
  sorry

end molecular_weight_of_3_moles_CaOH2_is_correct_l173_173731


namespace rational_inequality_solution_l173_173361

theorem rational_inequality_solution {x : ℝ} : (4 / (x + 1) ≤ 1) → (x ∈ Set.Iic (-1) ∪ Set.Ici 3) :=
by 
  sorry

end rational_inequality_solution_l173_173361


namespace carrie_jellybeans_l173_173350

def volume (a : ℕ) : ℕ := a * a * a

def bert_box_volume : ℕ := 216

def carrie_factor : ℕ := 3

def count_error_factor : ℝ := 1.10

noncomputable def jellybeans_carrie (bert_box_volume carrie_factor count_error_factor : ℝ) : ℝ :=
  count_error_factor * (carrie_factor ^ 3 * bert_box_volume)

theorem carrie_jellybeans (bert_box_volume := 216) (carrie_factor := 3) (count_error_factor := 1.10) :
  jellybeans_carrie bert_box_volume carrie_factor count_error_factor = 6415 :=
sorry

end carrie_jellybeans_l173_173350


namespace train_speed_l173_173163

theorem train_speed (l t: ℝ) (h1: l = 441) (h2: t = 21) : l / t = 21 := by
  sorry

end train_speed_l173_173163


namespace man_walk_time_l173_173717

theorem man_walk_time (speed_kmh : ℕ) (distance_km : ℕ) (time_min : ℕ) 
  (h1 : speed_kmh = 10) (h2 : distance_km = 7) : time_min = 42 :=
by
  sorry

end man_walk_time_l173_173717


namespace volume_of_new_pyramid_is_108_l173_173786

noncomputable def volume_of_cut_pyramid : ℝ :=
  let base_edge_length := 12 * Real.sqrt 2
  let slant_edge_length := 15
  let cut_height := 4.5
  -- Calculate the height of the original pyramid using Pythagorean theorem
  let original_height := Real.sqrt (slant_edge_length^2 - (base_edge_length/2 * Real.sqrt 2)^2)
  -- Calculate the remaining height of the smaller pyramid
  let remaining_height := original_height - cut_height
  -- Calculate the scale factor
  let scale_factor := remaining_height / original_height
  -- New base edge length
  let new_base_edge_length := base_edge_length * scale_factor
  -- New base area
  let new_base_area := (new_base_edge_length)^2
  -- Volume of the new pyramid
  (1 / 3) * new_base_area * remaining_height

-- Define the statement to prove
theorem volume_of_new_pyramid_is_108 :
  volume_of_cut_pyramid = 108 :=
by
  sorry

end volume_of_new_pyramid_is_108_l173_173786


namespace dog_catches_sheep_in_20_seconds_l173_173734

variable (v_sheep v_dog : ℕ) (d : ℕ)

def relative_speed (v_dog v_sheep : ℕ) := v_dog - v_sheep

def time_to_catch (d v_sheep v_dog : ℕ) : ℕ := d / (relative_speed v_dog v_sheep)

theorem dog_catches_sheep_in_20_seconds
  (h1 : v_sheep = 16)
  (h2 : v_dog = 28)
  (h3 : d = 240) :
  time_to_catch d v_sheep v_dog = 20 := by {
  sorry
}

end dog_catches_sheep_in_20_seconds_l173_173734


namespace part_I_part_II_l173_173890

-- Part (I)
theorem part_I (a₁ : ℝ) (d : ℝ) (S : ℕ → ℝ) (k : ℕ) :
  a₁ = 3 / 2 →
  d = 1 →
  (∀ n, S n = (n / 2 : ℝ) * (n + 2)) →
  S (k ^ 2) = S k ^ 2 →
  k = 4 :=
by
  intros ha₁ hd hSn hSeq
  sorry

-- Part (II)
theorem part_II (a : ℝ) (d : ℝ) (S : ℕ → ℝ) :
  (∀ k : ℕ, S (k ^ 2) = (S k) ^ 2) →
  ( (∀ n, a = 0 ∧ d = 0 ∧ a + d * (n - 1) = 0) ∨
    (∀ n, a = 1 ∧ d = 0 ∧ a + d * (n - 1) = 1) ∨
    (∀ n, a = 1 ∧ d = 2 ∧ a + d * (n - 1) = 2 * n - 1) ) :=
by
  intros hSeq
  sorry

end part_I_part_II_l173_173890


namespace skitties_remainder_l173_173853

theorem skitties_remainder (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 :=
sorry

end skitties_remainder_l173_173853


namespace escher_prints_probability_l173_173499

theorem escher_prints_probability :
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  probability = 1 / 1320 :=
by
  let total_pieces := 12
  let escher_pieces := 4
  let total_permutations := Nat.factorial total_pieces
  let block_permutations := 9 * Nat.factorial (total_pieces - escher_pieces)
  let probability := block_permutations / (total_pieces * Nat.factorial (total_pieces - 1))
  sorry

end escher_prints_probability_l173_173499


namespace tangent_line_parallel_range_a_l173_173297

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log x + 1/2 * x^2 + a * x

theorem tangent_line_parallel_range_a (a : ℝ) :
  (∃ x > 0, deriv (f a) x = 3) ↔ a ≤ 1 :=
by
  sorry

end tangent_line_parallel_range_a_l173_173297


namespace derivative_at_neg_one_l173_173566

variable (a b : ℝ)

-- Define the function f(x)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 6

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- Given condition f'(1) = 2
axiom h : f' a b 1 = 2

-- Statement to prove f'(-1) = -2
theorem derivative_at_neg_one : f' a b (-1) = -2 :=
by 
  sorry

end derivative_at_neg_one_l173_173566


namespace discount_rate_pony_jeans_l173_173122

theorem discount_rate_pony_jeans
  (fox_price pony_price : ℕ)
  (fox_pairs pony_pairs : ℕ)
  (total_savings total_discount_rate : ℕ)
  (F P : ℕ)
  (h1 : fox_price = 15)
  (h2 : pony_price = 20)
  (h3 : fox_pairs = 3)
  (h4 : pony_pairs = 2)
  (h5 : total_savings = 9)
  (h6 : total_discount_rate = 22)
  (h7 : F + P = total_discount_rate)
  (h8 : fox_pairs * fox_price * F / 100 + pony_pairs * pony_price * P / 100 = total_savings) : 
  P = 18 :=
sorry

end discount_rate_pony_jeans_l173_173122


namespace rational_function_nonnegative_l173_173057

noncomputable def rational_function (x : ℝ) : ℝ :=
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3)

theorem rational_function_nonnegative :
  ∀ x, 0 ≤ x ∧ x < 3 → 0 ≤ rational_function x :=
sorry

end rational_function_nonnegative_l173_173057


namespace liza_butter_amount_l173_173692

theorem liza_butter_amount (B : ℕ) (h1 : B / 2 + B / 5 + (1 / 3) * ((B - B / 2 - B / 5) / 1) = B - 2) : B = 10 :=
sorry

end liza_butter_amount_l173_173692


namespace minimum_moves_to_determine_polynomial_l173_173182

-- Define quadratic polynomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define conditions as per the given problem
variables {f g : ℝ → ℝ}
def is_quadratic (p : ℝ → ℝ) := ∃ a b c : ℝ, ∀ x : ℝ, p x = quadratic a b c x

axiom f_is_quadratic : is_quadratic f
axiom g_is_quadratic : is_quadratic g

-- Define the main problem statement
theorem minimum_moves_to_determine_polynomial (n : ℕ) :
  (∀ (t : ℕ → ℝ), (∀ m ≤ n, (f (t m) = g (t m)) ∨ (f (t m) ≠ g (t m))) →
  (∃ a b c: ℝ, ∀ x: ℝ, f x = quadratic a b c x ∨ g x = quadratic a b c x)) ↔ n = 8 :=
sorry -- Proof is omitted

end minimum_moves_to_determine_polynomial_l173_173182


namespace sin_beta_l173_173127

open Real

theorem sin_beta {α β : ℝ} (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_cosα : cos α = 2 * sqrt 5 / 5)
  (h_sinαβ : sin (α - β) = -3 / 5) :
  sin β = 2 * sqrt 5 / 5 := 
sorry

end sin_beta_l173_173127


namespace exists_two_points_same_color_l173_173436

theorem exists_two_points_same_color :
  ∀ (x : ℝ), ∀ (color : ℝ × ℝ → Prop),
  (∀ (p : ℝ × ℝ), color p = red ∨ color p = blue) →
  (∃ (p1 p2 : ℝ × ℝ), dist p1 p2 = x ∧ color p1 = color p2) :=
by
  intro x color color_prop
  sorry

end exists_two_points_same_color_l173_173436


namespace circle_radius_l173_173155

theorem circle_radius (x y : ℝ) : x^2 - 10*x + y^2 + 4*y + 13 = 0 → ∃ r : ℝ, r = 4 :=
by
  -- sorry here to indicate that the proof is skipped
  sorry

end circle_radius_l173_173155


namespace cos_pi_over_3_plus_2theta_l173_173246

theorem cos_pi_over_3_plus_2theta 
  (theta : ℝ)
  (h : Real.sin (Real.pi / 3 - theta) = 3 / 4) : 
  Real.cos (Real.pi / 3 + 2 * theta) = 1 / 8 :=
by 
  sorry

end cos_pi_over_3_plus_2theta_l173_173246


namespace fill_box_with_cubes_l173_173302

-- Define the dimensions of the box
def boxLength : ℕ := 35
def boxWidth : ℕ := 20
def boxDepth : ℕ := 10

-- Define the greatest common divisor of the box dimensions
def gcdBoxDims : ℕ := Nat.gcd (Nat.gcd boxLength boxWidth) boxDepth

-- Define the smallest number of identical cubes that can fill the box
def smallestNumberOfCubes : ℕ := (boxLength / gcdBoxDims) * (boxWidth / gcdBoxDims) * (boxDepth / gcdBoxDims)

theorem fill_box_with_cubes :
  smallestNumberOfCubes = 56 :=
by
  -- Proof goes here
  sorry

end fill_box_with_cubes_l173_173302


namespace nested_abs_expression_eval_l173_173837

theorem nested_abs_expression_eval :
  abs (abs (-abs (-2 + 3) - 2) + 3) = 6 := sorry

end nested_abs_expression_eval_l173_173837


namespace Penelope_Candies_l173_173908

variable (M : ℕ) (S : ℕ)
variable (h1 : 5 * S = 3 * M)
variable (h2 : M = 25)

theorem Penelope_Candies : S = 15 := by
  sorry

end Penelope_Candies_l173_173908


namespace parabola_hyperbola_focus_l173_173257

theorem parabola_hyperbola_focus (p : ℝ) :
  let parabolaFocus := (p / 2, 0)
  let hyperbolaRightFocus := (2, 0)
  (parabolaFocus = hyperbolaRightFocus) → p = 4 := 
by
  intro h
  sorry

end parabola_hyperbola_focus_l173_173257


namespace product_of_de_l173_173480

theorem product_of_de (d e : ℤ) (h1: ∀ (r : ℝ), r^2 - r - 1 = 0 → r^6 - (d : ℝ) * r - (e : ℝ) = 0) : 
  d * e = 40 :=
by
  sorry

end product_of_de_l173_173480


namespace first_programmer_loses_l173_173770

noncomputable def programSequence : List ℕ :=
  List.range 1999 |>.map (fun i => 2^i)

def validMove (sequence : List ℕ) (move : List ℕ) : Prop :=
  move.length = 5 ∧ move.all (λ i => i < sequence.length ∧ sequence.get! i > 0)

def applyMove (sequence : List ℕ) (move : List ℕ) : List ℕ :=
  move.foldl
    (λ seq i => seq.set i (seq.get! i - 1))
    sequence

def totalWeight (sequence : List ℕ) : ℕ :=
  sequence.foldl (· + ·) 0

theorem first_programmer_loses : ∀ seq moves,
  seq = programSequence →
  (∀ move, validMove seq move → False) →
  applyMove seq moves = seq →
  totalWeight seq = 2^1999 - 1 :=
by
  intro seq moves h_seq h_valid_move h_apply_move
  sorry

end first_programmer_loses_l173_173770


namespace least_possible_perimeter_l173_173382

/-- Proof that the least possible perimeter of a triangle with two sides of length 24 and 51 units,
    and the third side being an integer, is 103 units. -/
theorem least_possible_perimeter (a b : ℕ) (c : ℕ) (h1 : a = 24) (h2 : b = 51) (h3 : c > 27) (h4 : c < 75) :
    a + b + c = 103 :=
by
  sorry

end least_possible_perimeter_l173_173382


namespace students_taking_art_l173_173195

def total_students := 500
def students_taking_music := 40
def students_taking_both := 10
def students_taking_neither := 450

theorem students_taking_art : ∃ A, total_students = students_taking_music - students_taking_both + (A - students_taking_both) + students_taking_both + students_taking_neither ∧ A = 20 :=
by
  sorry

end students_taking_art_l173_173195


namespace fraction_difference_l173_173369

theorem fraction_difference : (18 / 42) - (3 / 8) = 3 / 56 := 
by
  sorry

end fraction_difference_l173_173369


namespace time_at_2010_minutes_after_3pm_is_930pm_l173_173836

def time_after_2010_minutes (current_time : Nat) (minutes_passed : Nat) : Nat :=
  sorry

theorem time_at_2010_minutes_after_3pm_is_930pm :
  time_after_2010_minutes 900 2010 = 1290 :=
by
  sorry

end time_at_2010_minutes_after_3pm_is_930pm_l173_173836


namespace provider_assignment_ways_l173_173482

theorem provider_assignment_ways (total_providers : ℕ) (children : ℕ) (h1 : total_providers = 15) (h2 : children = 4) : 
  (Finset.range total_providers).card.factorial / (Finset.range (total_providers - children)).card.factorial = 32760 :=
by
  rw [h1, h2]
  norm_num
  sorry

end provider_assignment_ways_l173_173482


namespace enthalpy_of_formation_C6H6_l173_173593

theorem enthalpy_of_formation_C6H6 :
  ∀ (enthalpy_C2H2 : ℝ) (enthalpy_C6H6 : ℝ)
  (enthalpy_C6H6_C6H6 : ℝ) (Hess_law : Prop),
  (enthalpy_C2H2 = 226.7) →
  (enthalpy_C6H6 = 631.1) →
  (enthalpy_C6H6_C6H6 = -33.9) →
  Hess_law →
  -- Using the given conditions to accumulate the enthalpy change for the formation of C6H6.
  ∃ Q_formation : ℝ, Q_formation = -82.9 := by
  sorry

end enthalpy_of_formation_C6H6_l173_173593


namespace algebraic_comparison_l173_173560

theorem algebraic_comparison (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^2 / b + b^2 / a ≥ a + b) :=
by
  sorry

end algebraic_comparison_l173_173560


namespace junk_mail_each_house_l173_173601

def blocks : ℕ := 16
def houses_per_block : ℕ := 17
def total_junk_mail : ℕ := 1088
def total_houses : ℕ := blocks * houses_per_block
def junk_mail_per_house : ℕ := total_junk_mail / total_houses

theorem junk_mail_each_house :
  junk_mail_per_house = 4 :=
by
  sorry

end junk_mail_each_house_l173_173601


namespace problem_l173_173862

    theorem problem (a b c : ℝ) : 
        a < b → 
        (∀ x : ℝ, (x ≤ -2 ∨ |x - 30| < 2) ↔ (0 ≤ (x - a) * (x - b) / (x - c))) → 
        a + 2 * b + 3 * c = 86 := by 
    sorry

end problem_l173_173862


namespace relativ_prime_and_divisible_exists_l173_173821

theorem relativ_prime_and_divisible_exists
  (a b c : ℕ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c) :
  ∃ r s : ℕ, Nat.gcd r s = 1 ∧ 0 < r ∧ 0 < s ∧ c ∣ (a * r + b * s) :=
by
  sorry

end relativ_prime_and_divisible_exists_l173_173821


namespace time_difference_alice_bob_l173_173474

theorem time_difference_alice_bob
  (alice_speed : ℕ) (bob_speed : ℕ) (distance : ℕ)
  (h_alice_speed : alice_speed = 7)
  (h_bob_speed : bob_speed = 9)
  (h_distance : distance = 12) :
  (bob_speed * distance - alice_speed * distance) = 24 :=
by
  sorry

end time_difference_alice_bob_l173_173474


namespace initial_apples_l173_173684

theorem initial_apples (picked: ℕ) (newly_grown: ℕ) (still_on_tree: ℕ) (initial: ℕ):
  (picked = 7) →
  (newly_grown = 2) →
  (still_on_tree = 6) →
  (still_on_tree + picked - newly_grown = initial) →
  initial = 11 :=
by
  intros hpicked hnewly_grown hstill_on_tree hcalculation
  sorry

end initial_apples_l173_173684


namespace BANANA_permutations_l173_173687

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l173_173687


namespace neg_ln_gt_zero_l173_173030

theorem neg_ln_gt_zero {x : ℝ} : (¬ ∀ x : ℝ, Real.log (x^2 + 1) > 0) ↔ ∃ x : ℝ, Real.log (x^2 + 1) ≤ 0 := by
  sorry

end neg_ln_gt_zero_l173_173030


namespace john_personal_payment_l173_173456

-- Definitions of the conditions
def cost_of_one_hearing_aid : ℕ := 2500
def number_of_hearing_aids : ℕ := 2
def insurance_coverage_percent : ℕ := 80

-- Derived definitions based on conditions
def total_cost : ℕ := cost_of_one_hearing_aid * number_of_hearing_aids
def insurance_coverage_amount : ℕ := total_cost * insurance_coverage_percent / 100
def johns_share : ℕ := total_cost - insurance_coverage_amount

-- Theorem statement (proof not included)
theorem john_personal_payment : johns_share = 1000 :=
sorry

end john_personal_payment_l173_173456


namespace two_lines_perpendicular_to_same_plane_are_parallel_l173_173319

/- 
Problem: Let a, b be two lines, and α be a plane. Prove that if a ⊥ α and b ⊥ α, then a ∥ b.
-/

variables {Line Plane : Type} 

def is_parallel (l1 l2 : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry
def is_contained_in (l : Line) (p : Plane) : Prop := sorry

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane)
  (ha_perpendicular : is_perpendicular a α)
  (hb_perpendicular : is_perpendicular b α) :
  is_parallel a b :=
by
  sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l173_173319


namespace trig_identity_sin_eq_l173_173605

theorem trig_identity_sin_eq (α : ℝ) (h : Real.cos (π / 6 - α) = 1 / 3) : 
  Real.sin (5 * π / 6 - 2 * α) = -7 / 9 := 
by 
  sorry

end trig_identity_sin_eq_l173_173605


namespace probability_fourth_ball_black_l173_173047

theorem probability_fourth_ball_black :
  let total_balls := 6
  let red_balls := 3
  let black_balls := 3
  let prob_black_first_draw := black_balls / total_balls
  (prob_black_first_draw = 1 / 2) ->
  (prob_black_first_draw = (black_balls / total_balls)) ->
  (black_balls / total_balls = 1 / 2) ->
  1 / 2 = 1 / 2 :=
by
  intros
  sorry

end probability_fourth_ball_black_l173_173047


namespace smallest_expression_l173_173420

theorem smallest_expression (a b : ℝ) (h : b < 0) : a + b < a ∧ a < a - b :=
by
  sorry

end smallest_expression_l173_173420


namespace number_of_players_l173_173406

theorem number_of_players (x y z : ℕ) 
  (h1 : x + y + z = 10)
  (h2 : x * y + y * z + z * x = 31) : 
  (x = 2 ∧ y = 3 ∧ z = 5) ∨ (x = 2 ∧ y = 5 ∧ z = 3) ∨ (x = 3 ∧ y = 2 ∧ z = 5) ∨ 
  (x = 3 ∧ y = 5 ∧ z = 2) ∨ (x = 5 ∧ y = 2 ∧ z = 3) ∨ (x = 5 ∧ y = 3 ∧ z = 2) :=
sorry

end number_of_players_l173_173406


namespace compare_probabilities_l173_173336

-- Definitions
variables (M N: ℕ) (m n: ℕ)
  
-- Conditions
def condition_m_millionaire : Prop := m > 10^6
def condition_n_nonmillionaire : Prop := n ≤ 10^6

-- Probabilities
noncomputable def P_A : ℚ := (M:ℚ) / (M + (n:ℚ)/(m:ℚ) * N)
noncomputable def P_B : ℚ := (M:ℚ) / (M + N)

-- Theorem statement
theorem compare_probabilities
  (hM : M > 0) (hN : N > 0)
  (h_m_millionaire : condition_m_millionaire m)
  (h_n_nonmillionaire : condition_n_nonmillionaire n) :
  P_A M N m n > P_B M N := sorry

end compare_probabilities_l173_173336


namespace find_f_zero_function_decreasing_find_range_x_l173_173018

noncomputable def f : ℝ → ℝ := sorry

-- Define the main conditions as hypotheses
axiom additivity : ∀ x1 x2 : ℝ, f (x1 + x2) = f x1 + f x2
axiom negativity : ∀ x : ℝ, x > 0 → f x < 0

-- First theorem: proving f(0) = 0
theorem find_f_zero : f 0 = 0 := sorry

-- Second theorem: proving the function is decreasing over (-∞, ∞)
theorem function_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := sorry

-- Third theorem: finding the range of x such that f(x) + f(2-3x) < 0
theorem find_range_x (x : ℝ) : f x + f (2 - 3 * x) < 0 → x < 1 := sorry

end find_f_zero_function_decreasing_find_range_x_l173_173018


namespace minimum_value_l173_173637

variable (a b : ℝ)

-- Assume a and b are positive real numbers
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)

-- Given the condition a + b = 2
variable (h₂ : a + b = 2)

theorem minimum_value : (1 / a) + (2 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end minimum_value_l173_173637


namespace set_complement_intersection_l173_173457

open Set

variable (U A B : Set ℕ)

theorem set_complement_intersection :
  U = {2, 3, 5, 7, 8} →
  A = {2, 8} →
  B = {3, 5, 8} →
  (U \ A) ∩ B = {3, 5} :=
by
  intros
  sorry

end set_complement_intersection_l173_173457


namespace slices_in_loaf_initial_l173_173197

-- Define the total slices used from Monday to Friday
def slices_used_weekdays : Nat := 5 * 2

-- Define the total slices used on Saturday
def slices_used_saturday : Nat := 2 * 2

-- Define the total slices used in the week
def total_slices_used : Nat := slices_used_weekdays + slices_used_saturday

-- Define the slices left
def slices_left : Nat := 6

-- Prove the total slices Tony started with
theorem slices_in_loaf_initial :
  let slices := total_slices_used + slices_left
  slices = 20 :=
by
  sorry

end slices_in_loaf_initial_l173_173197


namespace assembly_line_average_output_l173_173611

theorem assembly_line_average_output :
  (60 / 90) + (60 / 60) = (5 / 3) →
  60 + 60 = 120 →
  120 / (5 / 3) = 72 :=
by
  intros h1 h2
  -- Proof follows, but we will end with 'sorry' to indicate further proof steps need to be done.
  sorry

end assembly_line_average_output_l173_173611


namespace area_of_EFGH_l173_173355

def short_side_length : ℕ := 4
def long_side_length : ℕ := short_side_length * 2
def number_of_rectangles : ℕ := 4
def larger_rectangle_length : ℕ := short_side_length
def larger_rectangle_width : ℕ := number_of_rectangles * long_side_length

theorem area_of_EFGH :
  (larger_rectangle_length * larger_rectangle_width) = 128 := 
  by
    sorry

end area_of_EFGH_l173_173355


namespace original_price_per_pound_l173_173645

theorem original_price_per_pound (P x : ℝ)
  (h1 : 0.2 * x * P = 0.2 * x)
  (h2 : x * P = x * P)
  (h3 : 1.08 * (0.8 * x) * 1.08 = 1.08 * x * P) :
  P = 1.08 :=
sorry

end original_price_per_pound_l173_173645


namespace three_digit_numbers_l173_173713

theorem three_digit_numbers (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999) → 
  (n * n % 1000 = n % 1000) ↔ 
  (n = 625 ∨ n = 376) :=
by 
  sorry

end three_digit_numbers_l173_173713


namespace find_length_of_side_c_l173_173303

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

/-- Given that in triangle ABC, sin C = 1 / 2, a = 2 * sqrt 3, b = 2,
we want to prove the length of side c is either 2 or 2 * sqrt 7. -/
theorem find_length_of_side_c (C : Real) (a b c : Real) (h1 : Real.sin C = 1 / 2)
  (h2 : a = 2 * Real.sqrt 3) (h3 : b = 2) :
  c = 2 ∨ c = 2 * Real.sqrt 7 :=
by
  sorry

end find_length_of_side_c_l173_173303


namespace max_g6_l173_173989

noncomputable def g (x : ℝ) : ℝ :=
sorry

theorem max_g6 :
  (∀ x, (g x = a * x^2 + b * x + c) ∧ (a ≥ 0) ∧ (b ≥ 0) ∧ (c ≥ 0)) →
  (g 3 = 3) →
  (g 9 = 243) →
  (g 6 ≤ 6) :=
sorry

end max_g6_l173_173989


namespace exists_square_no_visible_points_l173_173522

-- Define visibility from the origin
def visible_from_origin (x y : ℤ) : Prop :=
  Int.gcd x y = 1

-- Main theorem statement
theorem exists_square_no_visible_points (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 
    (∀ (x y : ℤ), a ≤ x ∧ x ≤ a + n ∧ b ≤ y ∧ y ≤ b + n ∧ (x ≠ 0 ∨ y ≠ 0) → ¬visible_from_origin x y) :=
sorry

end exists_square_no_visible_points_l173_173522


namespace value_of_expression_l173_173984

theorem value_of_expression (x : ℝ) (h : 7 * x^2 - 2 * x - 4 = 4 * x + 11) : 
  (5 * x - 7)^2 = 11.63265306 := 
by 
  sorry

end value_of_expression_l173_173984


namespace intersection_A_compB_l173_173444

-- Define the sets A and B
def A : Set ℝ := {x | -3 < x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of B relative to ℝ
def comp_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- State the main theorem to prove
theorem intersection_A_compB : A ∩ comp_B = {x | -3 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_A_compB_l173_173444


namespace monotonically_increasing_range_of_a_l173_173478

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4 * x - 5)

theorem monotonically_increasing_range_of_a :
  ∀ (a : ℝ), (∀ x : ℝ, x > a → f x > f a) ↔ a ≥ 5 :=
by
  intro a
  unfold f
  sorry

end monotonically_increasing_range_of_a_l173_173478


namespace min_hypotenuse_of_right_triangle_l173_173815

theorem min_hypotenuse_of_right_triangle (a b c k : ℝ) (h₁ : k = a + b + c) (h₂ : a^2 + b^2 = c^2) : 
  c ≥ (Real.sqrt 2 - 1) * k := 
sorry

end min_hypotenuse_of_right_triangle_l173_173815


namespace Zack_traveled_18_countries_l173_173584

variables (countries_Alex countries_George countries_Joseph countries_Patrick countries_Zack : ℕ)
variables (h1 : countries_Alex = 24)
variables (h2 : countries_George = countries_Alex / 4)
variables (h3 : countries_Joseph = countries_George / 2)
variables (h4 : countries_Patrick = 3 * countries_Joseph)
variables (h5 : countries_Zack = 2 * countries_Patrick)

theorem Zack_traveled_18_countries :
  countries_Zack = 18 :=
by sorry

end Zack_traveled_18_countries_l173_173584


namespace _l173_173243

noncomputable def t_value_theorem (a b x d t y : ℕ) (h1 : a + b = x) (h2 : x + d = t) (h3 : t + a = y) (h4 : b + d + y = 16) : t = 8 :=
by sorry

end _l173_173243


namespace range_of_a_l173_173809

theorem range_of_a (a : ℝ) : (¬ (∃ x0 : ℝ, a * x0^2 + x0 + 1/2 ≤ 0)) → a > 1/2 :=
by
  sorry

end range_of_a_l173_173809


namespace find_original_price_l173_173896

-- Definitions for the conditions mentioned in the problem
variables {P : ℝ} -- Original price per gallon in dollars

-- Proof statement assuming the given conditions
theorem find_original_price 
  (h1 : ∃ P : ℝ, P > 0) -- There exists a positive price per gallon in dollars
  (h2 : (250 / (0.9 * P)) = (250 / P + 5)) -- After a 10% price reduction, 5 gallons more can be bought for $250
  : P = 25 / 4.5 := -- The solution states the original price per gallon is approximately $5.56
by
  sorry -- Proof omitted

end find_original_price_l173_173896


namespace remainder_of_13_plus_x_mod_29_l173_173961

theorem remainder_of_13_plus_x_mod_29
  (x : ℕ)
  (hx : 8 * x ≡ 1 [MOD 29])
  (hp : 0 < x) : 
  (13 + x) % 29 = 18 :=
sorry

end remainder_of_13_plus_x_mod_29_l173_173961


namespace nearest_integer_to_a_plus_b_l173_173543

theorem nearest_integer_to_a_plus_b
  (a b : ℝ)
  (h1 : |a| + b = 5)
  (h2 : |a| * b + a^3 = -8) :
  abs (a + b - 3) ≤ 0.5 :=
sorry

end nearest_integer_to_a_plus_b_l173_173543


namespace abc_equal_l173_173607

theorem abc_equal (a b c : ℝ)
  (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ b * x^2 + c * x + a)
  (h2 : ∀ x : ℝ, b * x^2 + c * x + a ≥ c * x^2 + a * x + b) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l173_173607


namespace similarity_transformation_l173_173185

theorem similarity_transformation (C C' : ℝ × ℝ) (r : ℝ) (h1 : r = 3) (h2 : C = (4, 1))
  (h3 : C' = (r * 4, r * 1)) : (C' = (12, 3) ∨ C' = (-12, -3)) := by
  sorry

end similarity_transformation_l173_173185


namespace jason_money_l173_173352

theorem jason_money (fred_money_before : ℕ) (jason_money_before : ℕ)
  (fred_money_after : ℕ) (total_earned : ℕ) :
  fred_money_before = 111 →
  jason_money_before = 40 →
  fred_money_after = 115 →
  total_earned = 4 →
  jason_money_before = 40 := by
  intros h1 h2 h3 h4
  sorry

end jason_money_l173_173352


namespace second_train_speed_l173_173842

theorem second_train_speed
  (v : ℕ)
  (h1 : 8 * v - 8 * 11 = 160) :
  v = 31 :=
sorry

end second_train_speed_l173_173842


namespace triangle_area_is_96_l173_173782

/-- Given a square with side length 8 and an overlapping area that is both three-quarters
    of the area of the square and half of the area of a triangle, prove the triangle's area is 96. -/
theorem triangle_area_is_96 (a : ℕ) (area_of_square : ℕ) (overlapping_area : ℕ) (area_of_triangle : ℕ) 
  (h1 : a = 8) 
  (h2 : area_of_square = a * a) 
  (h3 : overlapping_area = (3 * area_of_square) / 4) 
  (h4 : overlapping_area = area_of_triangle / 2) : 
  area_of_triangle = 96 := 
by 
  sorry

end triangle_area_is_96_l173_173782


namespace consecutive_sum_36_unique_l173_173293

def is_consecutive_sum (a b n : ℕ) :=
  (0 < n) ∧ ((n ≥ 2) ∧ (b = a + n - 1) ∧ (2 * a + n - 1) * n = 72)

theorem consecutive_sum_36_unique :
  ∃! n, ∃ a b, is_consecutive_sum a b n :=
by
  sorry

end consecutive_sum_36_unique_l173_173293


namespace radius_of_scrap_cookie_l173_173635

theorem radius_of_scrap_cookie :
  ∀ (r : ℝ),
    (∃ (r_dough r_cookie : ℝ),
      r_dough = 6 ∧  -- Radius of the large dough
      r_cookie = 2 ∧  -- Radius of each cookie
      8 * (π * r_cookie^2) ≤ π * r_dough^2 ∧  -- Total area of cookies is less than or equal to area of large dough
      (π * r_dough^2) - (8 * (π * r_cookie^2)) = π * r^2  -- Area of scrap dough forms a circle of radius r
    ) → r = 2 := by
  sorry

end radius_of_scrap_cookie_l173_173635


namespace initial_percentage_decrease_l173_173987

theorem initial_percentage_decrease (P x : ℝ) (h1 : 0 < P) (h2 : 0 ≤ x) (h3 : x ≤ 100) :
  ((P - (x / 100) * P) * 1.50 = P * 1.20) → x = 20 :=
by
  sorry

end initial_percentage_decrease_l173_173987


namespace cost_relationship_l173_173830

variable {α : Type} [LinearOrderedField α]
variables (bananas_cost apples_cost pears_cost : α)

theorem cost_relationship :
  (5 * bananas_cost = 3 * apples_cost) →
  (10 * apples_cost = 6 * pears_cost) →
  (25 * bananas_cost = 9 * pears_cost) := by
  intros h1 h2
  sorry

end cost_relationship_l173_173830


namespace apples_to_grapes_equivalent_l173_173938

-- Definitions based on the problem conditions
def apples := ℝ
def grapes := ℝ

-- Given conditions
def given_condition : Prop := (3 / 4) * 12 = 9

-- Question to prove
def question : Prop := (1 / 2) * 6 = 3

-- The theorem statement combining given conditions to prove the question
theorem apples_to_grapes_equivalent : given_condition → question := 
by
    intros
    sorry

end apples_to_grapes_equivalent_l173_173938


namespace sampling_methods_used_l173_173683

-- Definitions based on problem conditions
def TotalHouseholds : Nat := 2000
def FarmerHouseholds : Nat := 1800
def WorkerHouseholds : Nat := 100
def IntellectualHouseholds : Nat := TotalHouseholds - FarmerHouseholds - WorkerHouseholds
def SampleSize : Nat := 40

-- The statement of the proof problem
theorem sampling_methods_used
  (N : Nat := TotalHouseholds)
  (F : Nat := FarmerHouseholds)
  (W : Nat := WorkerHouseholds)
  (I : Nat := IntellectualHouseholds)
  (S : Nat := SampleSize)
:
  (1 ∈ [1, 2, 3]) ∧ (2 ∈ [1, 2, 3]) ∧ (3 ∈ [1, 2, 3]) :=
by
  -- Add the proof here
  sorry

end sampling_methods_used_l173_173683


namespace calculate_expr_l173_173177

theorem calculate_expr : 1 - Real.sqrt 9 = -2 := by
  sorry

end calculate_expr_l173_173177


namespace product_of_possible_values_of_x_l173_173970

theorem product_of_possible_values_of_x : 
  (∀ x, |x - 7| - 5 = 4 → x = 16 ∨ x = -2) -> (16 * -2 = -32) :=
by
  intro h
  have := h 16
  have := h (-2)
  sorry

end product_of_possible_values_of_x_l173_173970


namespace value_of_a_l173_173388
noncomputable def find_a (a b c : ℝ) : ℝ :=
if 2 * b = a + c ∧ (a * c) * (b * c) = ((a * b) ^ 2) ∧ a + b + c = 6 then a else 0

theorem value_of_a (a b c : ℝ) :
  (2 * b = a + c) ∧ ((a * c) * (b * c) = (a * b) ^ 2) ∧ (a + b + c = 6) ∧ (a ≠ c) ∧ (a ≠ b) ∧ (b ≠ c) → a = 4 :=
by sorry

end value_of_a_l173_173388


namespace smallest_a_for_nonprime_l173_173271

theorem smallest_a_for_nonprime (a : ℕ) : (∀ x : ℤ, ∃ d : ℤ, d ∣ (x^4 + a^4) ∧ d ≠ 1 ∧ d ≠ (x^4 + a^4)) ↔ a = 3 := by
  sorry

end smallest_a_for_nonprime_l173_173271


namespace volume_of_prism_l173_173179

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 54) (h2 : b * c = 56) (h3 : a * c = 60) :
    a * b * c = 426 :=
sorry

end volume_of_prism_l173_173179


namespace two_layers_area_zero_l173_173329

theorem two_layers_area_zero (A X Y Z : ℕ)
  (h1 : A = 212)
  (h2 : X + Y + Z = 140)
  (h3 : Y + Z = 24)
  (h4 : Z = 24) : Y = 0 :=
by
  sorry

end two_layers_area_zero_l173_173329


namespace parabola_focus_distance_l173_173253

theorem parabola_focus_distance (p : ℝ) (h_pos : p > 0) (A : ℝ × ℝ)
  (h_A_on_parabola : A.2 = 5 ∧ A.1^2 = 2 * p * A.2)
  (h_AF : abs (A.2 - (p / 2)) = 8) : p = 6 :=
by
  sorry

end parabola_focus_distance_l173_173253


namespace fraction_increase_by_3_l173_173280

theorem fraction_increase_by_3 (x y : ℝ) (h₁ : x' = 3 * x) (h₂ : y' = 3 * y) : 
  (x' * y') / (x' - y') = 3 * (x * y) / (x - y) :=
by
  sorry

end fraction_increase_by_3_l173_173280


namespace trigonometric_values_l173_173894

-- Define cos and sin terms
def cos (x : ℝ) : ℝ := sorry
def sin (x : ℝ) : ℝ := sorry

-- Define the condition given in the problem statement
def condition (x : ℝ) : Prop := cos x - 4 * sin x = 1

-- Define the result we need to prove
def result (x : ℝ) : Prop := sin x + 4 * cos x = 4 ∨ sin x + 4 * cos x = -4

-- The main statement in Lean 4 to be proved
theorem trigonometric_values (x : ℝ) : condition x → result x := by
  sorry

end trigonometric_values_l173_173894


namespace back_seat_tickets_sold_l173_173505

def total_tickets : ℕ := 20000
def main_seat_price : ℕ := 55
def back_seat_price : ℕ := 45
def total_revenue : ℕ := 955000

theorem back_seat_tickets_sold :
  ∃ (M B : ℕ), 
    M + B = total_tickets ∧ 
    main_seat_price * M + back_seat_price * B = total_revenue ∧ 
    B = 14500 :=
by
  sorry

end back_seat_tickets_sold_l173_173505


namespace bagel_spending_l173_173506

theorem bagel_spending (B D : ℝ) (h1 : D = 0.5 * B) (h2 : B = D + 15) : B + D = 45 := by
  sorry

end bagel_spending_l173_173506


namespace triangle_area_l173_173139

open Real

-- Define the conditions
variables (a : ℝ) (B : ℝ) (cosA : ℝ)
variable (S : ℝ)

-- Given conditions of the problem
def triangle_conditions : Prop :=
  a = 5 ∧ B = π / 3 ∧ cosA = 11 / 14

-- State the theorem to be proved
theorem triangle_area (h : triangle_conditions a B cosA) : S = 10 * sqrt 3 :=
sorry

end triangle_area_l173_173139


namespace Carl_typing_words_l173_173232

variable (typingSpeed : ℕ) (hoursPerDay : ℕ) (days : ℕ)

theorem Carl_typing_words (h1 : typingSpeed = 50) (h2 : hoursPerDay = 4) (h3 : days = 7) :
  (typingSpeed * 60 * hoursPerDay * days) = 84000 := by
  sorry

end Carl_typing_words_l173_173232


namespace trig_identity_l173_173101

theorem trig_identity (x : ℝ) (h : Real.cos (x - π / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * π / 3) + Real.sin (π / 3 - x)^2 = 5 / 3 :=
by
  sorry

end trig_identity_l173_173101


namespace ratio_of_population_is_correct_l173_173671

noncomputable def ratio_of_population (M W C : ℝ) : ℝ :=
  (M / (W + C)) * 100

theorem ratio_of_population_is_correct
  (M W C : ℝ) 
  (hW: W = 0.9 * M)
  (hC: C = 0.6 * (M + W)) :
  ratio_of_population M W C = 49.02 := 
by
  sorry

end ratio_of_population_is_correct_l173_173671


namespace smallest_positive_b_l173_173412

theorem smallest_positive_b (b : ℤ) :
  b % 5 = 1 ∧ b % 4 = 2 ∧ b % 7 = 3 → b = 86 :=
by
  sorry

end smallest_positive_b_l173_173412


namespace area_of_gray_region_is_27pi_l173_173399

-- Define the conditions
def concentric_circles (inner_radius outer_radius : ℝ) :=
  2 * inner_radius = outer_radius

def width_of_gray_region (inner_radius outer_radius width : ℝ) :=
  width = outer_radius - inner_radius

-- Define the proof problem
theorem area_of_gray_region_is_27pi
(inner_radius outer_radius : ℝ) 
(h1 : concentric_circles inner_radius outer_radius)
(h2 : width_of_gray_region inner_radius outer_radius 3) :
π * outer_radius^2 - π * inner_radius^2 = 27 * π :=
by
  -- Proof goes here, but it is not required as per instructions
  sorry

end area_of_gray_region_is_27pi_l173_173399


namespace min_value_expression_l173_173009

theorem min_value_expression (α β : ℝ) :
  ∃ x y, x = 3 * Real.cos α + 6 * Real.sin β ∧
         y = 3 * Real.sin α + 6 * Real.cos β ∧
         (x - 10)^2 + (y - 18)^2 = 121 :=
by
  sorry

end min_value_expression_l173_173009


namespace total_pieces_of_gum_and_candy_l173_173606

theorem total_pieces_of_gum_and_candy 
  (packages_A : ℕ) (pieces_A : ℕ) (packages_B : ℕ) (pieces_B : ℕ) 
  (packages_C : ℕ) (pieces_C : ℕ) (packages_X : ℕ) (pieces_X : ℕ)
  (packages_Y : ℕ) (pieces_Y : ℕ) 
  (hA : packages_A = 10) (hA_pieces : pieces_A = 4)
  (hB : packages_B = 5) (hB_pieces : pieces_B = 8)
  (hC : packages_C = 13) (hC_pieces : pieces_C = 12)
  (hX : packages_X = 8) (hX_pieces : pieces_X = 6)
  (hY : packages_Y = 6) (hY_pieces : pieces_Y = 10) : 
  packages_A * pieces_A + packages_B * pieces_B + packages_C * pieces_C + 
  packages_X * pieces_X + packages_Y * pieces_Y = 344 := 
by
  sorry

end total_pieces_of_gum_and_candy_l173_173606


namespace train_speed_l173_173663

theorem train_speed (train_length : ℕ) (cross_time : ℕ) (speed : ℕ) 
  (h_train_length : train_length = 300)
  (h_cross_time : cross_time = 10)
  (h_speed_eq : speed = train_length / cross_time) : 
  speed = 30 :=
by
  sorry

end train_speed_l173_173663


namespace second_newly_inserted_number_eq_l173_173056

theorem second_newly_inserted_number_eq : 
  ∃ q : ℝ, (q ^ 12 = 2) ∧ (1 * (q ^ 2) = 2 ^ (1 / 6)) := 
by
  sorry

end second_newly_inserted_number_eq_l173_173056


namespace total_amount_spent_l173_173145

-- Define the variables B and D representing the amounts Ben and David spent.
variables (B D : ℝ)

-- Define the conditions based on the given problem.
def conditions : Prop :=
  (D = 0.60 * B) ∧ (B = D + 14)

-- The main theorem stating the total amount spent by Ben and David is 56.
theorem total_amount_spent (h : conditions B D) : B + D = 56 :=
sorry  -- Proof omitted.

end total_amount_spent_l173_173145


namespace total_number_of_coins_l173_173004

-- Define conditions
def pennies : Nat := 38
def nickels : Nat := 27
def dimes : Nat := 19
def quarters : Nat := 24
def half_dollars : Nat := 13
def one_dollar_coins : Nat := 17
def two_dollar_coins : Nat := 5
def australian_fifty_cent_coins : Nat := 4
def mexican_one_peso_coins : Nat := 12

-- Define the problem as a theorem
theorem total_number_of_coins : 
  pennies + nickels + dimes + quarters + half_dollars + one_dollar_coins + two_dollar_coins + australian_fifty_cent_coins + mexican_one_peso_coins = 159 := by
  sorry

end total_number_of_coins_l173_173004


namespace tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l173_173531

theorem tanx_eq_2_sin2cos2 (x : ℝ) (h : Real.tan x = 2) : 
  (2 / 3) * (Real.sin x) ^ 2 + (1 / 4) * (Real.cos x) ^ 2 = 7 / 12 := 
by 
  sorry

theorem tanx_eq_2_cos_sin_ratio (x : ℝ) (h : Real.tan x = 2) : 
  (Real.cos x + Real.sin x) / (Real.cos x - Real.sin x) = -3 := 
by 
  sorry

end tanx_eq_2_sin2cos2_tanx_eq_2_cos_sin_ratio_l173_173531


namespace fractional_eq_solve_simplify_and_evaluate_l173_173250

-- Question 1: Solve the fractional equation
theorem fractional_eq_solve (x : ℝ) (h1 : (x / (x + 1) = (2 * x) / (3 * x + 3) + 1)) : 
  x = -1.5 := 
sorry

-- Question 2: Simplify and evaluate the expression for x = -1
theorem simplify_and_evaluate (x : ℝ)
  (h2 : x ≠ 0) (h3 : x ≠ 2) (h4 : x ≠ -2) :
  (x + 2) / (x^2 - 2*x) - (x - 1) / (x^2 - 4*x + 4) / ((x+2) / (x^3 - 4*x)) = 
  (x - 4) / (x - 2) ∧ 
  (x = -1) → ((x - 4) / (x - 2) = (5 / 3)) := 
sorry

end fractional_eq_solve_simplify_and_evaluate_l173_173250


namespace increase_in_average_commission_l173_173076

theorem increase_in_average_commission :
  ∀ (new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 : ℕ),
    new_avg = 400 → 
    n1 = 6 → 
    n2 = n1 - 1 → 
    big_sale = 1300 →
    total_earnings = new_avg * n1 →
    commission = total_earnings - big_sale →
    old_avg = commission / n2 →
    new_avg - old_avg = 180 :=
by 
  intros new_avg old_avg total_earnings big_sale commission s1 s2 n1 n2 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end increase_in_average_commission_l173_173076


namespace months_after_withdrawal_and_advance_eq_eight_l173_173342

-- Define initial conditions
def initial_investment_A : ℝ := 3000
def initial_investment_B : ℝ := 4000
def withdrawal_A : ℝ := 1000
def advancement_B : ℝ := 1000
def total_profit : ℝ := 630
def share_A : ℝ := 240
def share_B : ℝ := total_profit - share_A

-- Define the main proof problem
theorem months_after_withdrawal_and_advance_eq_eight
  (initial_investment_A : ℝ) (initial_investment_B : ℝ)
  (withdrawal_A : ℝ) (advancement_B : ℝ)
  (total_profit : ℝ) (share_A : ℝ) (share_B : ℝ) : 
  ∃ x : ℝ, 
  (3000 * x + 2000 * (12 - x)) / (4000 * x + 5000 * (12 - x)) = 240 / 390 ∧
  x = 8 :=
sorry

end months_after_withdrawal_and_advance_eq_eight_l173_173342


namespace unique_cubic_coefficients_l173_173476

noncomputable def cubic_function (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem unique_cubic_coefficients
  (a b c : ℝ)
  (h1 : ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ cubic_function a b c x ∧ cubic_function a b c x ≤ 1) :
  (a = 0 ∧ b = -3 ∧ c = 0) :=
by
  sorry

end unique_cubic_coefficients_l173_173476


namespace original_sheets_count_is_115_l173_173943

def find_sheets_count (S P : ℕ) : Prop :=
  -- Ann's condition: all papers are used leaving 100 flyers
  S - P = 100 ∧
  -- Bob's condition: all bindings used leaving 35 sheets of paper
  5 * P = S - 35

theorem original_sheets_count_is_115 (S P : ℕ) (h : find_sheets_count S P) : S = 115 :=
by
  sorry

end original_sheets_count_is_115_l173_173943


namespace units_digit_2019_pow_2019_l173_173023

theorem units_digit_2019_pow_2019 : (2019^2019) % 10 = 9 := 
by {
  -- The statement of the problem is proved below
  sorry  -- Solution to be filled in
}

end units_digit_2019_pow_2019_l173_173023


namespace number_of_marked_points_l173_173680

theorem number_of_marked_points (S S' : ℤ) (n : ℤ) 
  (h1 : S = 25) 
  (h2 : S' = S - 5 * n) 
  (h3 : S' = -35) : 
  n = 12 := 
  sorry

end number_of_marked_points_l173_173680


namespace find_real_solutions_l173_173951

theorem find_real_solutions (x : ℝ) :
  x^4 + (3 - x)^4 = 146 ↔ x = 1.5 + Real.sqrt 3.4175 ∨ x = 1.5 - Real.sqrt 3.4175 :=
by
  sorry

end find_real_solutions_l173_173951


namespace total_cost_of_lollipops_l173_173510

/-- Given Sarah bought 12 lollipops and shared one-quarter of them, 
    and Julie reimbursed Sarah 75 cents for the shared lollipops,
    Prove that the total cost of the lollipops in dollars is $3. --/
theorem total_cost_of_lollipops 
(Sarah_lollipops : ℕ) 
(shared_fraction : ℚ) 
(Julie_paid : ℚ) 
(total_lollipops_cost : ℚ)
(h1 : Sarah_lollipops = 12) 
(h2 : shared_fraction = 1/4) 
(h3 : Julie_paid = 75 / 100) 
(h4 : total_lollipops_cost = 
        ((Julie_paid / (Sarah_lollipops * shared_fraction)) * Sarah_lollipops / 100)) :
total_lollipops_cost = 3 := 
sorry

end total_cost_of_lollipops_l173_173510


namespace famous_sentences_correct_l173_173710

def blank_1 : String := "correct_answer_1"
def blank_2 : String := "correct_answer_2"
def blank_3 : String := "correct_answer_3"
def blank_4 : String := "correct_answer_4"
def blank_5 : String := "correct_answer_5"
def blank_6 : String := "correct_answer_6"
def blank_7 : String := "correct_answer_7"
def blank_8 : String := "correct_answer_8"

theorem famous_sentences_correct :
  blank_1 = "correct_answer_1" ∧
  blank_2 = "correct_answer_2" ∧
  blank_3 = "correct_answer_3" ∧
  blank_4 = "correct_answer_4" ∧
  blank_5 = "correct_answer_5" ∧
  blank_6 = "correct_answer_6" ∧
  blank_7 = "correct_answer_7" ∧
  blank_8 = "correct_answer_8" :=
by
  -- The proof details correspond to the part "refer to the correct solution for each blank"
  sorry

end famous_sentences_correct_l173_173710


namespace prince_wish_fulfilled_l173_173807

theorem prince_wish_fulfilled
  (k : ℕ)
  (k_gt_1 : 1 < k)
  (k_lt_13 : k < 13)
  (city : Fin 13 → Fin k) 
  (initial_goblets : Fin k → Fin 13)
  (is_gold : Fin 13 → Bool) :
  ∃ i j : Fin 13, i ≠ j ∧ city i = city j ∧ is_gold i = true ∧ is_gold j = true := 
sorry

end prince_wish_fulfilled_l173_173807


namespace total_meat_supply_l173_173187

-- Definitions of the given conditions
def lion_consumption_per_day : ℕ := 25
def tiger_consumption_per_day : ℕ := 20
def duration_days : ℕ := 2

-- Statement of the proof problem
theorem total_meat_supply :
  (lion_consumption_per_day + tiger_consumption_per_day) * duration_days = 90 :=
by
  sorry

end total_meat_supply_l173_173187


namespace find_quadruples_l173_173381

open Nat

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

theorem find_quadruples (a b p n : ℕ) (hp : is_prime p) (h_ab : a + b ≠ 0) :
  a^3 + b^3 = p^n ↔ (a = 1 ∧ b = 1 ∧ p = 2 ∧ n = 1) ∨
               (a = 1 ∧ b = 2 ∧ p = 3 ∧ n = 2) ∨ 
               (a = 2 ∧ b = 1 ∧ p = 3 ∧ n = 2) ∨
               ∃ (k : ℕ), (a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3*k + 1) ∨ 
                          (a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3*k + 2) ∨
                          (a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3*k + 2) := sorry

end find_quadruples_l173_173381


namespace fraction_division_l173_173966

theorem fraction_division :
  (5 : ℚ) / ((13 : ℚ) / 7) = 35 / 13 :=
by
  sorry

end fraction_division_l173_173966


namespace rob_baseball_cards_l173_173840

theorem rob_baseball_cards
  (r j r_d : ℕ)
  (hj : j = 40)
  (h_double : r_d = j / 5)
  (h_cards : r = 3 * r_d) :
  r = 24 :=
by
  sorry

end rob_baseball_cards_l173_173840


namespace total_worth_of_stock_l173_173109

theorem total_worth_of_stock (total_worth profit_fraction profit_rate loss_fraction loss_rate overall_loss : ℝ) :
  profit_fraction = 0.20 ->
  profit_rate = 0.20 -> 
  loss_fraction = 0.80 -> 
  loss_rate = 0.10 -> 
  overall_loss = 500 ->
  total_worth - (profit_fraction * total_worth * profit_rate) - (loss_fraction * total_worth * loss_rate) = overall_loss ->
  total_worth = 12500 :=
by
  sorry

end total_worth_of_stock_l173_173109


namespace zoe_pictures_l173_173929

theorem zoe_pictures (pictures_taken : ℕ) (dolphin_show_pictures : ℕ)
  (h1 : pictures_taken = 28) (h2 : dolphin_show_pictures = 16) :
  pictures_taken + dolphin_show_pictures = 44 :=
sorry

end zoe_pictures_l173_173929


namespace dreams_ratio_l173_173071

variable (N : ℕ) (D_total : ℕ) (D_per_day : ℕ)

-- Conditions
def days_per_year : Prop := N = 365
def dreams_per_day : Prop := D_per_day = 4
def total_dreams : Prop := D_total = 4380

-- Derived definitions
def dreams_this_year := D_per_day * N
def dreams_last_year := D_total - dreams_this_year

-- Theorem to prove
theorem dreams_ratio 
  (h1 : days_per_year N)
  (h2 : dreams_per_day D_per_day)
  (h3 : total_dreams D_total)
  : dreams_last_year N D_total D_per_day / dreams_this_year N D_per_day = 2 :=
by
  sorry

end dreams_ratio_l173_173071


namespace total_pushups_l173_173877

def Zachary_pushups : ℕ := 44
def David_pushups : ℕ := Zachary_pushups + 58

theorem total_pushups : Zachary_pushups + David_pushups = 146 := by
  sorry

end total_pushups_l173_173877


namespace max_abs_sum_value_l173_173766

noncomputable def max_abs_sum (x y : ℝ) : ℝ := |x| + |y|

theorem max_abs_sum_value (x y : ℝ) (h : x^2 + y^2 = 4) : max_abs_sum x y ≤ 2 * Real.sqrt 2 :=
by {
  sorry
}

end max_abs_sum_value_l173_173766


namespace circle_center_radius_l173_173135

theorem circle_center_radius (x y : ℝ) :
  (x - 1)^2 + (y - 3)^2 = 4 → (1, 3) = (1, 3) ∧ 2 = 2 :=
by
  intro h
  exact ⟨rfl, rfl⟩

end circle_center_radius_l173_173135


namespace f_at_63_l173_173834

-- Define the function f: ℤ → ℤ with given properties
def f : ℤ → ℤ :=
  sorry -- Placeholder, as we are only stating the problem, not the solution

-- Conditions
axiom f_at_1 : f 1 = 6
axiom f_eq : ∀ x : ℤ, f (2 * x + 1) = 3 * f x

-- The goal is to prove f(63) = 1458
theorem f_at_63 : f 63 = 1458 :=
  sorry

end f_at_63_l173_173834


namespace exists_saddle_point_probability_l173_173274

noncomputable def saddle_point_probability := (3 : ℝ) / 10

theorem exists_saddle_point_probability {A : ℕ → ℕ → ℝ}
  (h : ∀ i j, 0 ≤ A i j ∧ A i j ≤ 1 ∧ (∀ k l, (i ≠ k ∨ j ≠ l) → A i j ≠ A k l)) :
  (∃ (p : ℝ), p = saddle_point_probability) :=
by 
  sorry

end exists_saddle_point_probability_l173_173274


namespace polynomial_quotient_l173_173982

open Polynomial

noncomputable def dividend : ℤ[X] := 5 * X^4 - 9 * X^3 + 3 * X^2 + 7 * X - 6
noncomputable def divisor : ℤ[X] := X - 1

theorem polynomial_quotient :
  dividend /ₘ divisor = 5 * X^3 - 4 * X^2 + 7 * X + 7 :=
by
  sorry

end polynomial_quotient_l173_173982


namespace distance_covered_at_40_kmph_l173_173871

theorem distance_covered_at_40_kmph (x : ℝ) 
  (h₁ : x / 40 + (250 - x) / 60 = 5.5) :
  x = 160 :=
sorry

end distance_covered_at_40_kmph_l173_173871


namespace length_of_new_section_l173_173551

-- Definitions from the conditions
def area : ℕ := 35
def width : ℕ := 7

-- The problem statement
theorem length_of_new_section (h : area = 35 ∧ width = 7) : 35 / 7 = 5 :=
by
  -- We'll provide the proof later
  sorry

end length_of_new_section_l173_173551


namespace probability_of_four_card_success_l173_173498

example (cards : Fin 4) (pins : Fin 4) {attempts : ℕ}
  (h1 : ∀ (c : Fin 4) (p : Fin 4), attempts ≤ 3)
  (h2 : ∀ (c : Fin 4), ∃ (p : Fin 4), p ≠ c ∧ attempts ≤ 3) :
  ∃ (three_cards : Fin 3), attempts ≤ 3 :=
sorry

noncomputable def probability_success :
  ℚ := 23 / 24

theorem probability_of_four_card_success :
  probability_success = 23 / 24 :=
sorry

end probability_of_four_card_success_l173_173498


namespace inequality_solution_l173_173795

theorem inequality_solution (x : ℝ) :
  27 ^ (Real.log x / Real.log 3) ^ 2 - 8 * x ^ (Real.log x / Real.log 3) ≥ 3 ↔
  x ∈ Set.Icc 0 (1 / 3) ∪ Set.Ici 3 :=
sorry

end inequality_solution_l173_173795


namespace grunters_win_all_5_games_grunters_win_at_least_one_game_l173_173804

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win all 5 games is 243/1024. --/
theorem grunters_win_all_5_games :
  (3/4)^5 = 243 / 1024 :=
sorry

/-- Given that the Grunters have an independent probability of 3/4 of winning any single game, 
and they play 5 games, the probability that the Grunters will win at least one game is 1023/1024. --/
theorem grunters_win_at_least_one_game :
  1 - (1/4)^5 = 1023 / 1024 :=
sorry

end grunters_win_all_5_games_grunters_win_at_least_one_game_l173_173804


namespace Lance_daily_earnings_l173_173316

theorem Lance_daily_earnings :
  ∀ (hours_per_week : ℕ) (workdays_per_week : ℕ) (hourly_rate : ℕ) (total_earnings : ℕ) (daily_earnings : ℕ),
  hours_per_week = 35 →
  workdays_per_week = 5 →
  hourly_rate = 9 →
  total_earnings = hours_per_week * hourly_rate →
  daily_earnings = total_earnings / workdays_per_week →
  daily_earnings = 63 := 
by
  intros hours_per_week workdays_per_week hourly_rate total_earnings daily_earnings
  intros H1 H2 H3 H4 H5
  sorry

end Lance_daily_earnings_l173_173316


namespace compute_expression_l173_173032

theorem compute_expression (x : ℝ) (h : x = 8) : 
  (x^6 - 64 * x^3 + 1024) / (x^3 - 16) = 480 :=
by
  rw [h]
  sorry

end compute_expression_l173_173032


namespace inequality_holds_l173_173968

theorem inequality_holds (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 :=
by sorry

end inequality_holds_l173_173968


namespace discount_percentage_l173_173502

theorem discount_percentage (number_of_tshirts : ℕ) (cost_per_tshirt amount_paid : ℝ)
  (h1 : number_of_tshirts = 6)
  (h2 : cost_per_tshirt = 20)
  (h3 : amount_paid = 60) : 
  ((number_of_tshirts * cost_per_tshirt - amount_paid) / (number_of_tshirts * cost_per_tshirt) * 100) = 50 := by
  -- The proof will go here
  sorry

end discount_percentage_l173_173502


namespace molecular_weights_correct_l173_173941

-- Define atomic weights
def atomic_weight_Al : Float := 26.98
def atomic_weight_Cl : Float := 35.45
def atomic_weight_K : Float := 39.10

-- Define molecular weight calculations
def molecular_weight_AlCl3 : Float :=
  atomic_weight_Al + 3 * atomic_weight_Cl

def molecular_weight_KCl : Float :=
  atomic_weight_K + atomic_weight_Cl

-- Theorem statement to prove
theorem molecular_weights_correct :
  molecular_weight_AlCl3 = 133.33 ∧ molecular_weight_KCl = 74.55 :=
by
  -- This is where we would normally prove the equivalence
  sorry

end molecular_weights_correct_l173_173941


namespace books_before_purchase_l173_173301

theorem books_before_purchase (x : ℕ) (h : x + 140 = (27 / 25 : ℚ) * x) : x = 1750 :=
sorry

end books_before_purchase_l173_173301


namespace find_n_l173_173879

theorem find_n (n : ℕ) (h : 12^(4 * n) = (1/12)^(n - 30)) : n = 6 := 
by {
  sorry 
}

end find_n_l173_173879


namespace base_729_base8_l173_173997

theorem base_729_base8 (b : ℕ) (X Y : ℕ) (h_distinct : X ≠ Y)
  (h_range : b^3 ≤ 729 ∧ 729 < b^4)
  (h_form : 729 = X * b^3 + Y * b^2 + X * b + Y) : b = 8 :=
sorry

end base_729_base8_l173_173997


namespace kiana_siblings_ages_l173_173128

/-- Kiana has two twin brothers, one is twice as old as the other, 
and their ages along with Kiana's age multiply to 72. Prove that 
the sum of their ages is 13. -/
theorem kiana_siblings_ages
  (y : ℕ) (K : ℕ) (h1 : 2 * y * K = 72) :
  y + 2 * y + K = 13 := 
sorry

end kiana_siblings_ages_l173_173128


namespace intersection_proof_l173_173738

def M : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def N : Set ℕ := { x | Real.sqrt (2^x - 1) < 5 }
def expected_intersection : Set ℕ := {1, 2, 3, 4}

theorem intersection_proof : M ∩ N = expected_intersection := by
  sorry

end intersection_proof_l173_173738


namespace cost_per_pizza_l173_173911

theorem cost_per_pizza (total_amount : ℝ) (num_pizzas : ℕ) (H : total_amount = 24) (H1 : num_pizzas = 3) : 
  (total_amount / num_pizzas) = 8 := 
by 
  sorry

end cost_per_pizza_l173_173911


namespace best_marksman_score_l173_173466

theorem best_marksman_score (n : ℕ) (hypothetical_score : ℕ) (average_if_hypothetical : ℕ) (actual_total_score : ℕ) (H1 : n = 8) (H2 : hypothetical_score = 92) (H3 : average_if_hypothetical = 84) (H4 : actual_total_score = 665) :
    ∃ (actual_best_score : ℕ), actual_best_score = 77 :=
by
    have hypothetical_total_score : ℕ := 7 * average_if_hypothetical + hypothetical_score
    have difference : ℕ := hypothetical_total_score - actual_total_score
    use hypothetical_score - difference
    sorry

end best_marksman_score_l173_173466


namespace inequality_solution_set_l173_173925

theorem inequality_solution_set (x : ℝ) :
  x^2 * (x^2 + 2*x + 1) > 2*x * (x^2 + 2*x + 1) ↔
  ((x < -1) ∨ (-1 < x ∧ x < 0) ∨ (2 < x)) :=
sorry

end inequality_solution_set_l173_173925


namespace area_difference_of_tablets_l173_173236

theorem area_difference_of_tablets 
  (d1 d2 : ℝ) (s1 s2 : ℝ)
  (h1 : d1 = 6) (h2 : d2 = 5) 
  (hs1 : d1^2 = 2 * s1^2) (hs2 : d2^2 = 2 * s2^2) 
  (A1 : ℝ) (A2 : ℝ) (hA1 : A1 = s1^2) (hA2 : A2 = s2^2)
  : A1 - A2 = 5.5 := 
sorry

end area_difference_of_tablets_l173_173236


namespace peter_has_read_more_books_l173_173031

theorem peter_has_read_more_books
  (total_books : ℕ)
  (peter_percentage : ℚ)
  (brother_percentage : ℚ)
  (sarah_percentage : ℚ)
  (peter_books : ℚ := (peter_percentage / 100) * total_books)
  (brother_books : ℚ := (brother_percentage / 100) * total_books)
  (sarah_books : ℚ := (sarah_percentage / 100) * total_books)
  (combined_books : ℚ := brother_books + sarah_books)
  (difference : ℚ := peter_books - combined_books) :
  total_books = 50 → peter_percentage = 60 → brother_percentage = 25 → sarah_percentage = 15 → difference = 10 :=
by
  sorry

end peter_has_read_more_books_l173_173031


namespace theoretical_yield_H2SO4_l173_173944

-- Define the theoretical yield calculation problem in terms of moles of reactions and products
theorem theoretical_yield_H2SO4 
  (moles_SO3 : ℝ) (moles_H2O : ℝ) 
  (reaction : moles_SO3 + moles_H2O = 2.0 + 1.5) 
  (limiting_reactant_H2O : moles_H2O = 1.5) : 
  1.5 = moles_H2O * 1 :=
  sorry

end theoretical_yield_H2SO4_l173_173944


namespace geometric_sequence_term_l173_173349

theorem geometric_sequence_term
  (r a : ℝ)
  (h1 : 180 * r = a)
  (h2 : a * r = 81 / 32)
  (h3 : a > 0) :
  a = 135 / 19 :=
by sorry

end geometric_sequence_term_l173_173349


namespace imaginary_unit_root_l173_173718

theorem imaginary_unit_root (a b : ℝ) (h : (Complex.I : ℂ) ^ 2 + a * Complex.I + b = 0) : a + b = 1 := by
  -- Since this is just the statement, we add a sorry to focus on the structure
  sorry

end imaginary_unit_root_l173_173718


namespace fourth_number_unit_digit_l173_173953

def unit_digit (n : ℕ) : ℕ := n % 10

theorem fourth_number_unit_digit (a b c d : ℕ) (h₁ : a = 7858) (h₂: b = 1086) (h₃ : c = 4582) (h₄ : unit_digit (a * b * c * d) = 8) :
  unit_digit d = 4 :=
sorry

end fourth_number_unit_digit_l173_173953


namespace find_length_of_AB_l173_173704

variable (A B C : ℝ)
variable (cos_C_div2 BC AC AB : ℝ)
variable (C_gt_0 : 0 < C / 2) (C_lt_pi : C / 2 < Real.pi)

axiom h1 : cos_C_div2 = Real.sqrt 5 / 5
axiom h2 : BC = 1
axiom h3 : AC = 5
axiom h4 : AB = Real.sqrt (BC ^ 2 + AC ^ 2 - 2 * BC * AC * (2 * cos_C_div2 ^ 2 - 1))

theorem find_length_of_AB : AB = 4 * Real.sqrt 2 :=
by
  sorry

end find_length_of_AB_l173_173704


namespace find_sums_of_integers_l173_173268

theorem find_sums_of_integers (x y : ℤ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_prod_sum : x * y + x + y = 125) (h_rel_prime : Int.gcd x y = 1) (h_lt_x : x < 30) (h_lt_y : y < 30) : 
  (x + y = 25) ∨ (x + y = 23) ∨ (x + y = 21) := 
by 
  sorry

end find_sums_of_integers_l173_173268


namespace complex_number_quadrant_l173_173604

theorem complex_number_quadrant (a : ℝ) : 
  (a^2 - 2 = 3 * a - 4) ∧ (a^2 - 2 < 0 ∧ 3 * a - 4 < 0) → a = 1 :=
by
  sorry

end complex_number_quadrant_l173_173604


namespace max_chips_can_be_removed_l173_173778

theorem max_chips_can_be_removed (initial_chips : (Fin 10) × (Fin 10) → ℕ) 
  (condition : ∀ i j, initial_chips (i, j) = 1) : 
    ∃ removed_chips : ℕ, removed_chips = 90 :=
by
  sorry

end max_chips_can_be_removed_l173_173778


namespace S_8_arithmetic_sequence_l173_173264

theorem S_8_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h2 : a 4 = 18 - a 5):
  S 8 = 72 :=
by
  sorry

end S_8_arithmetic_sequence_l173_173264


namespace fraction_food_l173_173864

-- Define the salary S and remaining amount H
def S : ℕ := 170000
def H : ℕ := 17000

-- Define fractions of the salary spent on house rent and clothes
def fraction_rent : ℚ := 1 / 10
def fraction_clothes : ℚ := 3 / 5

-- Define the fraction F to be proven
def F : ℚ := 1 / 5

-- Define the remaining fraction of the salary
def remaining_fraction : ℚ := H / S

theorem fraction_food :
  ∀ S H : ℕ,
  S = 170000 →
  H = 17000 →
  F = 1 / 5 →
  F + (fraction_rent + fraction_clothes) + remaining_fraction = 1 :=
by
  intros S H hS hH hF
  sorry

end fraction_food_l173_173864


namespace sum_of_geometric_numbers_l173_173460

def is_geometric (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ∃ r : ℕ, r > 0 ∧ 
  (d2 = d1 * r) ∧ 
  (d3 = d2 * r) ∧ 
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d1 ≠ d3

theorem sum_of_geometric_numbers : 
  (∃ smallest largest : ℕ,
    (smallest = 124) ∧ 
    (largest = 972) ∧ 
    is_geometric (smallest) ∧ 
    is_geometric (largest)
  ) →
  124 + 972 = 1096 :=
by
  sorry

end sum_of_geometric_numbers_l173_173460


namespace greatest_prime_factor_341_l173_173716

theorem greatest_prime_factor_341 : ∃ p, Nat.Prime p ∧ p ≥ 17 ∧ (∀ q, Nat.Prime q ∧ q ∣ 341 → q ≤ p) ∧ p = 19 := by
  sorry

end greatest_prime_factor_341_l173_173716


namespace find_two_numbers_l173_173719

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l173_173719


namespace prime_if_and_only_if_digit_is_nine_l173_173664

theorem prime_if_and_only_if_digit_is_nine (B : ℕ) (h : 0 ≤ B ∧ B < 10) :
  Prime (303200 + B) ↔ B = 9 := 
by
  sorry

end prime_if_and_only_if_digit_is_nine_l173_173664


namespace no_common_points_iff_parallel_l173_173318

-- Definitions based on conditions:
def line (a : Type) : Prop := sorry
def plane (M : Type) : Prop := sorry
def no_common_points (a : Type) (M : Type) : Prop := sorry
def parallel (a : Type) (M : Type) : Prop := sorry

-- Theorem stating the relationship is necessary and sufficient
theorem no_common_points_iff_parallel (a M : Type) :
  no_common_points a M ↔ parallel a M := sorry

end no_common_points_iff_parallel_l173_173318


namespace continuous_iff_integral_condition_l173_173414

open Real 

noncomputable section

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def integral_condition (f : ℝ → ℝ) (a : ℝ) (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (∫ x in a..(a + a_seq n), f x) + (∫ x in (a - a_seq n)..a, f x) ≤ (a_seq n) / n

theorem continuous_iff_integral_condition (a : ℝ) (f : ℝ → ℝ)
  (h_nondec : is_non_decreasing f) :
  ContinuousAt f a ↔ ∃ (a_seq : ℕ → ℝ), (∀ n, 0 < a_seq n) ∧ integral_condition f a a_seq := sorry

end continuous_iff_integral_condition_l173_173414


namespace square_reciprocal_sum_integer_l173_173745

theorem square_reciprocal_sum_integer (a : ℝ) (h : ∃ k : ℤ, a + 1/a = k) : ∃ m : ℤ, a^2 + 1/a^2 = m := by
  sorry

end square_reciprocal_sum_integer_l173_173745


namespace solve_apples_problem_l173_173485

def apples_problem (marin_apples donald_apples total_apples : ℕ) : Prop :=
  marin_apples = 9 ∧ total_apples = 11 → donald_apples = 2

theorem solve_apples_problem : apples_problem 9 2 11 := by
  sorry

end solve_apples_problem_l173_173485


namespace min_cost_per_ounce_l173_173291

theorem min_cost_per_ounce 
  (cost_40 : ℝ := 200) (cost_90 : ℝ := 400)
  (percentage_40 : ℝ := 0.4) (percentage_90 : ℝ := 0.9)
  (desired_percentage : ℝ := 0.5) :
  (∀ (x y : ℝ), 0.4 * x + 0.9 * y = 0.5 * (x + y) → 200 * x + 400 * y / (x + y) = 240) :=
sorry

end min_cost_per_ounce_l173_173291


namespace count_integer_values_not_satisfying_inequality_l173_173157

theorem count_integer_values_not_satisfying_inequality : 
  ∃ n : ℕ, 
  (n = 3) ∧ (∀ x : ℤ, (4 * x^2 + 22 * x + 21 ≤ 25) → (-2 ≤ x ∧ x ≤ 0)) :=
by
  sorry

end count_integer_values_not_satisfying_inequality_l173_173157


namespace P_plus_Q_is_expected_l173_173072

-- defining the set P
def P : Set ℝ := { x | x ^ 2 - 3 * x - 4 ≤ 0 }

-- defining the set Q
def Q : Set ℝ := { x | x ^ 2 - 2 * x - 15 > 0 }

-- defining the set P + Q
def P_plus_Q : Set ℝ := { x | (x ∈ P ∨ x ∈ Q) ∧ ¬(x ∈ P ∧ x ∈ Q) }

-- the expected result
def expected_P_plus_Q : Set ℝ := { x | x < -3 } ∪ { x | -1 ≤ x ∧ x ≤ 4 } ∪ { x | x > 5 }

-- theorem stating that P + Q equals the expected result
theorem P_plus_Q_is_expected : P_plus_Q = expected_P_plus_Q := by
  sorry

end P_plus_Q_is_expected_l173_173072


namespace share_of_a_l173_173629

def shares_sum (a b c : ℝ) := a + b + c = 366
def share_a (a b c : ℝ) := a = 1/2 * (b + c)
def share_b (a b c : ℝ) := b = 2/3 * (a + c)

theorem share_of_a (a b c : ℝ) 
  (h1 : shares_sum a b c) 
  (h2 : share_a a b c) 
  (h3 : share_b a b c) : 
  a = 122 := 
by 
  -- Proof goes here
  sorry

end share_of_a_l173_173629


namespace p_true_of_and_not_p_false_l173_173949

variable {p q : Prop}

theorem p_true_of_and_not_p_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : p :=
sorry

end p_true_of_and_not_p_false_l173_173949


namespace problem1_problem2_l173_173992

variables (x y : ℝ)

-- Given Conditions
def given_conditions :=
  (x = 2 + Real.sqrt 3) ∧ (y = 2 - Real.sqrt 3)

-- Problem 1
theorem problem1 (h : given_conditions x y) : x^2 + y^2 = 14 :=
sorry

-- Problem 2
theorem problem2 (h : given_conditions x y) : (x / y) - (y / x) = 8 * Real.sqrt 3 :=
sorry

end problem1_problem2_l173_173992


namespace sum_of_numbers_l173_173147

theorem sum_of_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := 
sorry

end sum_of_numbers_l173_173147


namespace intersection_roots_l173_173585

theorem intersection_roots :
  x^2 - 4*x - 5 = 0 → (x = 5 ∨ x = -1) := by
  sorry

end intersection_roots_l173_173585


namespace find_4_oplus_2_l173_173308

def operation (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem find_4_oplus_2 : operation 4 2 = 26 :=
by
  sorry

end find_4_oplus_2_l173_173308


namespace math_problem_l173_173831

-- Definitions based on conditions
def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

-- Main theorem statement
theorem math_problem :
  avg4 (avg4 2 2 0 2) (avg2 3 1) 0 3 = 13 / 8 :=
by
  sorry

end math_problem_l173_173831


namespace find_some_number_l173_173839

theorem find_some_number (d : ℝ) (x : ℝ) (h1 : d = (0.889 * x) / 9.97) (h2 : d = 4.9) :
  x = 54.9 := by
  sorry

end find_some_number_l173_173839


namespace max_band_members_l173_173582

theorem max_band_members (n : ℤ) (h1 : 22 * n % 24 = 2) (h2 : 22 * n < 1000) : 22 * n = 770 :=
  sorry

end max_band_members_l173_173582


namespace pupils_correct_l173_173475

def totalPeople : ℕ := 676
def numberOfParents : ℕ := 22
def numberOfPupils : ℕ := totalPeople - numberOfParents

theorem pupils_correct :
  numberOfPupils = 654 := 
by
  sorry

end pupils_correct_l173_173475


namespace angles_of_tangency_triangle_l173_173569

theorem angles_of_tangency_triangle 
  (A B C : ℝ) 
  (ha : A = 40)
  (hb : B = 80)
  (hc : C = 180 - A - B)
  (a1 b1 c1 : ℝ)
  (ha1 : a1 = (1/2) * (180 - A))
  (hb1 : b1 = (1/2) * (180 - B))
  (hc1 : c1 = 180 - a1 - b1) :
  (a1 = 70 ∧ b1 = 50 ∧ c1 = 60) :=
by sorry

end angles_of_tangency_triangle_l173_173569


namespace inconsistent_mixture_volume_l173_173249

theorem inconsistent_mixture_volume :
  ∀ (diesel petrol water total_volume : ℚ),
    diesel = 4 →
    petrol = 4 →
    total_volume = 2.666666666666667 →
    diesel + petrol + water = total_volume →
    false :=
by
  intros diesel petrol water total_volume diesel_eq petrol_eq total_volume_eq volume_eq
  rw [diesel_eq, petrol_eq] at volume_eq
  sorry

end inconsistent_mixture_volume_l173_173249


namespace problem_statement_l173_173642

theorem problem_statement (x : ℝ) (h : 0 < x) : x + 2016^2016 / x^2016 ≥ 2017 := 
by
  sorry

end problem_statement_l173_173642


namespace find_fx_for_negative_x_l173_173191

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem find_fx_for_negative_x (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_given : given_function f) :
  ∀ x, (x < 0) → f x = x + x^2 :=
by
  sorry

end find_fx_for_negative_x_l173_173191


namespace polynomial_multiplication_l173_173258

theorem polynomial_multiplication :
  (5 * X^2 + 3 * X - 4) * (2 * X^3 + X^2 - X + 1) = 
  (10 * X^5 + 11 * X^4 - 10 * X^3 - 2 * X^2 + 7 * X - 4) := 
by {
  sorry
}

end polynomial_multiplication_l173_173258


namespace age_ratio_albert_mary_l173_173898

variable (A M B : ℕ) 

theorem age_ratio_albert_mary
    (h1 : A = 4 * B)
    (h2 : M = A - 10)
    (h3 : B = 5) :
    A = 2 * M :=
by
    sorry

end age_ratio_albert_mary_l173_173898


namespace prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l173_173495

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (4 * x + a) / (x^2 + 1)

-- 1. Prove that a = 0 given that f(x) is an odd function
theorem prove_a_eq_0 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = - f x a) : a = 0 := sorry

-- 2. Prove that f(x) = 4x / (x^2 + 1) is monotonically decreasing on [1, +∞) for x > 0
theorem prove_monotonic_decreasing (x : ℝ) (hx : x > 0) :
  ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (f x1 0) > (f x2 0) := sorry

-- 3. Prove that |f(x1) - f(x2)| ≤ m for all x1, x2 ∈ R implies m ≥ 4
theorem prove_m_ge_4 (m : ℝ) 
  (h : ∀ x1 x2 : ℝ, |f x1 0 - f x2 0| ≤ m) : m ≥ 4 := sorry

end prove_a_eq_0_prove_monotonic_decreasing_prove_m_ge_4_l173_173495


namespace melissa_points_per_game_l173_173730

theorem melissa_points_per_game (total_points : ℕ) (games : ℕ) (h1 : total_points = 81) 
(h2 : games = 3) : total_points / games = 27 :=
by
  sorry

end melissa_points_per_game_l173_173730


namespace total_amount_l173_173706

theorem total_amount (x y z total : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : y = 27) : total = 117 :=
by
  -- Proof here
  sorry

end total_amount_l173_173706


namespace dog_weight_ratio_l173_173976

theorem dog_weight_ratio
  (w7 : ℕ) (r : ℕ) (w13 : ℕ) (w21 : ℕ) (w52 : ℕ):
  (w7 = 6) →
  (w13 = 12 * r) →
  (w21 = 2 * w13) →
  (w52 = w21 + 30) →
  (w52 = 78) →
  r = 2 :=
by 
  sorry

end dog_weight_ratio_l173_173976


namespace work_completion_time_l173_173467

theorem work_completion_time (B_rate A_rate Combined_rate : ℝ) (B_time : ℝ) :
  (B_rate = 1 / 60) →
  (A_rate = 4 * B_rate) →
  (Combined_rate = A_rate + B_rate) →
  (B_time = 1 / Combined_rate) →
  B_time = 12 :=
by sorry

end work_completion_time_l173_173467


namespace rate_of_rainfall_is_one_l173_173245

variable (R : ℝ)
variable (h1 : 2 + 4 * R + 4 * 3 = 18)

theorem rate_of_rainfall_is_one : R = 1 :=
by
  sorry

end rate_of_rainfall_is_one_l173_173245


namespace find_radii_l173_173800

theorem find_radii (r R : ℝ) (h₁ : R - r = 2) (h₂ : R + r = 16) : r = 7 ∧ R = 9 := by
  sorry

end find_radii_l173_173800


namespace min_odd_integers_l173_173161

theorem min_odd_integers (a b c d e f g h i : ℤ)
  (h1 : a + b + c = 30)
  (h2 : a + b + c + d + e + f = 48)
  (h3 : a + b + c + d + e + f + g + h + i = 69) :
  ∃ k : ℕ, k = 1 ∧
  (∃ (aa bb cc dd ee ff gg hh ii : ℤ), (fun (x : ℤ) => x % 2 = 1 → k = 1) (aa + bb + cc + dd + ee + ff + gg + hh + ii)) :=
by
  intros
  sorry

end min_odd_integers_l173_173161


namespace surface_area_of_circumscribed_sphere_of_triangular_pyramid_l173_173665

theorem surface_area_of_circumscribed_sphere_of_triangular_pyramid
  (a : ℝ)
  (h₁ : a > 0) : 
  ∃ S, S = (27 * π / 32 * a^2) := 
by
  sorry

end surface_area_of_circumscribed_sphere_of_triangular_pyramid_l173_173665


namespace scientific_notation_of_0_0000003_l173_173653

theorem scientific_notation_of_0_0000003 : 0.0000003 = 3 * 10^(-7) := by
  sorry

end scientific_notation_of_0_0000003_l173_173653


namespace sin_alpha_plus_half_pi_l173_173802

theorem sin_alpha_plus_half_pi (α : ℝ) 
  (h1 : Real.tan (α - Real.pi) = 3 / 4)
  (h2 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2)) : 
  Real.sin (α + Real.pi / 2) = -4 / 5 :=
by
  -- Placeholder for the proof
  sorry

end sin_alpha_plus_half_pi_l173_173802


namespace geometric_sequence_seventh_term_l173_173741

theorem geometric_sequence_seventh_term (a₁ : ℤ) (a₂ : ℚ) (r : ℚ) (k : ℕ) (a₇ : ℚ)
  (h₁ : a₁ = 3) 
  (h₂ : a₂ = -1 / 2)
  (h₃ : r = a₂ / a₁)
  (h₄ : k = 7)
  (h₅ : a₇ = a₁ * r^(k-1)) : 
  a₇ = 1 / 15552 := 
by
  sorry

end geometric_sequence_seventh_term_l173_173741


namespace find_value_of_fraction_of_x_six_l173_173702

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := (Real.log x) / (Real.log b)

theorem find_value_of_fraction_of_x_six (x : ℝ) (h : log_base (10 * x) 10 + log_base (100 * x ^ 2) 10 = -1) : 
    1 / x ^ 6 = 31622.7766 :=
by
  sorry

end find_value_of_fraction_of_x_six_l173_173702


namespace diana_statues_painted_l173_173535

theorem diana_statues_painted :
  let paint_remaining := (1 : ℚ) / 2
  let paint_per_statue := (1 : ℚ) / 4
  (paint_remaining / paint_per_statue) = 2 :=
by
  sorry

end diana_statues_painted_l173_173535


namespace folded_triangle_square_length_l173_173102

theorem folded_triangle_square_length (side_length folded_distance length_squared : ℚ) 
(h1: side_length = 15) 
(h2: folded_distance = 11) 
(h3: length_squared = 1043281/31109) :
∃ (PQ : ℚ), PQ^2 = length_squared := 
by 
  sorry

end folded_triangle_square_length_l173_173102


namespace sue_shoes_probability_l173_173034

def sueShoes : List (String × ℕ) := [("black", 7), ("brown", 3), ("gray", 2)]

def total_shoes := 24

def prob_same_color (color : String) (pairs : List (String × ℕ)) : ℚ :=
  let total_pairs := pairs.foldr (λ p acc => acc + p.snd) 0
  let matching_pair := pairs.filter (λ p => p.fst = color)
  if matching_pair.length = 1 then
   let n := matching_pair.head!.snd * 2
   (n / total_shoes) * ((n / 2) / (total_shoes - 1))
  else 0

def prob_total (pairs : List (String × ℕ)) : ℚ :=
  (prob_same_color "black" pairs) + (prob_same_color "brown" pairs) + (prob_same_color "gray" pairs)

theorem sue_shoes_probability :
  prob_total sueShoes = 31 / 138 := by
  sorry

end sue_shoes_probability_l173_173034


namespace base7_digits_l173_173534

theorem base7_digits (D E F : ℕ) (h1 : D ≠ 0) (h2 : E ≠ 0) (h3 : F ≠ 0) (h4 : D < 7) (h5 : E < 7) (h6 : F < 7)
  (h_diff1 : D ≠ E) (h_diff2 : D ≠ F) (h_diff3 : E ≠ F)
  (h_eq : (49 * D + 7 * E + F) + (49 * E + 7 * F + D) + (49 * F + 7 * D + E) = 400 * D) :
  E + F = 6 :=
by
  sorry

end base7_digits_l173_173534


namespace present_age_of_son_l173_173092

theorem present_age_of_son (S F : ℕ)
  (h1 : F = S + 24)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 22 :=
by {
  -- The proof is omitted, as per instructions.
  sorry
}

end present_age_of_son_l173_173092


namespace william_ends_with_18_tickets_l173_173542

-- Define the initial number of tickets
def initialTickets : ℕ := 15

-- Define the tickets bought
def ticketsBought : ℕ := 3

-- Prove the total number of tickets William ends with
theorem william_ends_with_18_tickets : initialTickets + ticketsBought = 18 := by
  sorry

end william_ends_with_18_tickets_l173_173542


namespace simplify_and_evaluate_l173_173866

variable (a : ℝ)
variable (ha : a = Real.sqrt 3 - 1)

theorem simplify_and_evaluate : 
  (1 + 3 / (a - 2)) / ((a^2 + 2 * a + 1) / (a - 2)) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_l173_173866


namespace inequality_correct_l173_173732

noncomputable def a : ℝ := Real.exp (-0.5)
def b : ℝ := 0.5
noncomputable def c : ℝ := Real.log 1.5

theorem inequality_correct : a > b ∧ b > c :=
by
  sorry

end inequality_correct_l173_173732


namespace evaluation_at_x_4_l173_173421

noncomputable def simplified_expression (x : ℝ) :=
  (x - 1 - (3 / (x + 1))) / ((x^2 + 2 * x) / (x + 1))

theorem evaluation_at_x_4 : simplified_expression 4 = 1 / 2 :=
by
  sorry

end evaluation_at_x_4_l173_173421


namespace division_result_l173_173087

theorem division_result (x : ℝ) (h : (x - 2) / 13 = 4) : (x - 5) / 7 = 7 := by
  sorry

end division_result_l173_173087


namespace group_contains_2007_l173_173273

theorem group_contains_2007 : 
  ∃ k, 2007 ∈ {a | (k * (k + 1)) / 2 < a ∧ a ≤ ((k + 1) * (k + 2)) / 2} ∧ k = 45 :=
by sorry

end group_contains_2007_l173_173273


namespace gcd_of_two_powers_l173_173058

-- Define the expressions
def two_pow_1015_minus_1 : ℤ := 2^1015 - 1
def two_pow_1024_minus_1 : ℤ := 2^1024 - 1

-- Define the gcd function and the target value
noncomputable def gcd_expr : ℤ := Int.gcd (2^1015 - 1) (2^1024 - 1)
def target : ℤ := 511

-- The statement we want to prove
theorem gcd_of_two_powers : gcd_expr = target := by 
  sorry

end gcd_of_two_powers_l173_173058


namespace distance_between_starting_points_l173_173468

theorem distance_between_starting_points :
  let speed1 := 70
  let speed2 := 80
  let start_time := 10 -- in hours (10 am)
  let meet_time := 14 -- in hours (2 pm)
  let travel_time := meet_time - start_time
  let distance1 := speed1 * travel_time
  let distance2 := speed2 * travel_time
  distance1 + distance2 = 600 :=
by
  sorry

end distance_between_starting_points_l173_173468


namespace find_first_term_l173_173673

noncomputable def firstTermOfGeometricSeries (S : ℝ) (r : ℝ) : ℝ :=
  S * (1 - r) / (1 - r)

theorem find_first_term
  (S : ℝ)
  (r : ℝ)
  (hS : S = 20)
  (hr : r = -3/7) :
  firstTermOfGeometricSeries S r = 200 / 7 :=
  by
    rw [hS, hr]
    sorry

end find_first_term_l173_173673


namespace unique_k_linear_equation_l173_173512

theorem unique_k_linear_equation :
  (∀ x y k : ℝ, (2 : ℝ) * x^|k| + (k - 1) * y = 3 → (|k| = 1 ∧ k ≠ 1) → k = -1) :=
by
  sorry

end unique_k_linear_equation_l173_173512


namespace guppies_eaten_by_moray_eel_l173_173438

-- Definitions based on conditions
def moray_eel_guppies_per_day : ℕ := sorry -- Number of guppies the moray eel eats per day

def number_of_betta_fish : ℕ := 5

def guppies_per_betta : ℕ := 7

def total_guppies_needed_per_day : ℕ := 55

-- Theorem based on the question
theorem guppies_eaten_by_moray_eel :
  moray_eel_guppies_per_day = total_guppies_needed_per_day - (number_of_betta_fish * guppies_per_betta) :=
sorry

end guppies_eaten_by_moray_eel_l173_173438


namespace radio_loss_percentage_l173_173077

theorem radio_loss_percentage (CP SP : ℝ) (h_CP : CP = 2400) (h_SP : SP = 2100) :
  ((CP - SP) / CP) * 100 = 12.5 :=
by
  -- Given cost price
  have h_CP : CP = 2400 := h_CP
  -- Given selling price
  have h_SP : SP = 2100 := h_SP
  sorry

end radio_loss_percentage_l173_173077


namespace jerry_total_cost_correct_l173_173686

theorem jerry_total_cost_correct :
  let bw_cost := 27
  let bw_discount := 0.1 * bw_cost
  let bw_discounted_price := bw_cost - bw_discount
  let color_cost := 32
  let color_discount := 0.05 * color_cost
  let color_discounted_price := color_cost - color_discount
  let total_color_discounted_price := 3 * color_discounted_price
  let total_discounted_price_before_tax := bw_discounted_price + total_color_discounted_price
  let tax_rate := 0.07
  let tax := total_discounted_price_before_tax * tax_rate
  let total_cost := total_discounted_price_before_tax + tax
  (Float.round (total_cost * 100) / 100) = 123.59 :=
sorry

end jerry_total_cost_correct_l173_173686


namespace smallest_multiple_of_7_greater_than_500_l173_173410

theorem smallest_multiple_of_7_greater_than_500 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n > 500 ∧ n = 504 := 
by
  sorry

end smallest_multiple_of_7_greater_than_500_l173_173410


namespace number_of_cuboids_painted_l173_173574

-- Define the problem conditions
def painted_faces (total_faces : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
  total_faces / faces_per_cuboid

-- Define the theorem to prove
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) :
  total_faces = 48 → faces_per_cuboid = 6 → painted_faces total_faces faces_per_cuboid = 8 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end number_of_cuboids_painted_l173_173574


namespace sale_in_first_month_l173_173845

theorem sale_in_first_month 
  (sale_month_2 : ℕ)
  (sale_month_3 : ℕ)
  (sale_month_4 : ℕ)
  (sale_month_5 : ℕ)
  (required_sale_month_6 : ℕ)
  (average_sale_6_months : ℕ)
  (total_sale_6_months : ℕ)
  (total_known_sales : ℕ)
  (sale_first_month : ℕ) : 
    sale_month_2 = 3920 →
    sale_month_3 = 3855 →
    sale_month_4 = 4230 →
    sale_month_5 = 3560 →
    required_sale_month_6 = 2000 →
    average_sale_6_months = 3500 →
    total_sale_6_months = 6 * average_sale_6_months →
    total_known_sales = sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 →
    total_sale_6_months - (total_known_sales + required_sale_month_6) = sale_first_month →
    sale_first_month = 3435 :=
by
  intros h2 h3 h4 h5 h6 h_avg h_total h_known h_calc
  sorry

end sale_in_first_month_l173_173845


namespace eight_digit_numbers_count_l173_173918

theorem eight_digit_numbers_count :
  let first_digit_choices := 9
  let remaining_digits_choices := 10 ^ 7
  9 * 10^7 = 90000000 :=
by
  sorry

end eight_digit_numbers_count_l173_173918


namespace problem1_eval_problem2_eval_l173_173425

-- Problem 1
theorem problem1_eval :
  (1 : ℚ) * (-4.5) - (-5.6667) - (2.5) - 7.6667 = -9 := 
by
  sorry

-- Problem 2
theorem problem2_eval :
  (-(4^2) / (-2)^3) - ((4 / 9) * ((-3 / 2)^2)) = 1 := 
by
  sorry

end problem1_eval_problem2_eval_l173_173425


namespace squares_with_center_25_60_l173_173434

theorem squares_with_center_25_60 :
  let center_x := 25
  let center_y := 60
  let non_neg_int_coords (x : ℤ) (y : ℤ) := x ≥ 0 ∧ y ≥ 0
  let is_center (x : ℤ) (y : ℤ) := x = center_x ∧ y = center_y
  let num_squares := 650
  ∃ n : ℤ, (n = num_squares) ∧ ∀ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ), 
    non_neg_int_coords x₁ y₁ ∧ non_neg_int_coords x₂ y₂ ∧ 
    non_neg_int_coords x₃ y₃ ∧ non_neg_int_coords x₄ y₄ ∧ 
    is_center ((x₁ + x₂ + x₃ + x₄) / 4) ((y₁ + y₂ + y₃ + y₄) / 4) → 
    ∃ (k : ℤ), n = 650 :=
sorry

end squares_with_center_25_60_l173_173434


namespace sum_of_cubes_l173_173259

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^3 + b^3 = 9 :=
by
  sorry

end sum_of_cubes_l173_173259


namespace equation_of_latus_rectum_l173_173005

theorem equation_of_latus_rectum (p : ℝ) (h1 : p = 6) :
  (∀ x y : ℝ, y ^ 2 = -12 * x → x = 3) :=
sorry

end equation_of_latus_rectum_l173_173005


namespace valid_x_for_expression_l173_173447

theorem valid_x_for_expression :
  (∃ x : ℝ, x = 8 ∧ (10 - x ≥ 0) ∧ (x - 4 ≠ 0)) ↔ (∃ x : ℝ, x = 8) :=
by
  sorry

end valid_x_for_expression_l173_173447


namespace percentage_increase_each_year_is_50_l173_173166

-- Definitions based on conditions
def students_passed_three_years_ago : ℕ := 200
def students_passed_this_year : ℕ := 675

-- The prove statement
theorem percentage_increase_each_year_is_50
    (N3 N0 : ℕ)
    (P : ℚ)
    (h1 : N3 = students_passed_three_years_ago)
    (h2 : N0 = students_passed_this_year)
    (h3 : N0 = N3 * (1 + P)^3) :
  P = 0.5 :=
by
  sorry

end percentage_increase_each_year_is_50_l173_173166


namespace overall_percent_supporters_l173_173016

theorem overall_percent_supporters
  (percent_A : ℝ) (percent_B : ℝ)
  (members_A : ℕ) (members_B : ℕ)
  (supporters_A : ℕ)
  (supporters_B : ℕ)
  (total_supporters : ℕ)
  (total_members : ℕ)
  (overall_percent : ℝ) 
  (h1 : percent_A = 0.70) 
  (h2 : percent_B = 0.75)
  (h3 : members_A = 200) 
  (h4 : members_B = 800) 
  (h5 : supporters_A = percent_A * members_A) 
  (h6 : supporters_B = percent_B * members_B) 
  (h7 : total_supporters = supporters_A + supporters_B) 
  (h8 : total_members = members_A + members_B) 
  (h9 : overall_percent = (total_supporters : ℝ) / total_members * 100) :
  overall_percent = 74 := by
  sorry

end overall_percent_supporters_l173_173016


namespace problem_1_simplification_l173_173620

theorem problem_1_simplification (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 2) : 
  (x - 2) / (x ^ 2) / (1 - 2 / x) = 1 / x := 
  sorry

end problem_1_simplification_l173_173620


namespace arith_seq_S13_value_l173_173304

variable {α : Type*} [LinearOrderedField α]

-- Definitions related to an arithmetic sequence
structure ArithSeq (α : Type*) :=
  (a : ℕ → α) -- the sequence itself
  (sum_first_n_terms : ℕ → α) -- sum of the first n terms

def is_arith_seq (seq : ArithSeq α) :=
  ∀ (n : ℕ), seq.a (n + 1) - seq.a n = seq.a 2 - seq.a 1

-- Our conditions
noncomputable def a5 (seq : ArithSeq α) := seq.a 5
noncomputable def a7 (seq : ArithSeq α) := seq.a 7
noncomputable def a9 (seq : ArithSeq α) := seq.a 9
noncomputable def S13 (seq : ArithSeq α) := seq.sum_first_n_terms 13

-- Problem statement
theorem arith_seq_S13_value (seq : ArithSeq α) 
  (h_arith_seq : is_arith_seq seq)
  (h_condition : 2 * (a5 seq) + 3 * (a7 seq) + 2 * (a9 seq) = 14) : 
  S13 seq = 26 := 
  sorry

end arith_seq_S13_value_l173_173304


namespace yoongi_correct_calculation_l173_173492

theorem yoongi_correct_calculation (x : ℕ) (h : x + 9 = 30) : x - 7 = 14 :=
sorry

end yoongi_correct_calculation_l173_173492


namespace boys_and_girls_in_class_l173_173417

theorem boys_and_girls_in_class (m d : ℕ)
  (A : (m - 1 = 10 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)) ∨ 
       (m - 1 = 14 - 4 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)))
  (B : (m - 1 = 13 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)) ∨ 
       (m - 1 = 11 - 4 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)))
  (C : (m - 1 = 13 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4)) ∨ 
       (m - 1 = 19 - 4 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4))) : 
  m = 14 ∧ d = 15 := 
sorry

end boys_and_girls_in_class_l173_173417


namespace range_of_a_l173_173906

theorem range_of_a (a : ℝ) (h : a - 2 * 1 + 4 > 0) : a > -2 :=
by
  -- proof is not required
  sorry

end range_of_a_l173_173906


namespace truck_travel_due_east_distance_l173_173364

theorem truck_travel_due_east_distance :
  ∀ (x : ℕ),
  (20 + 20)^2 + x^2 = 50^2 → x = 30 :=
by
  intro x
  sorry -- proof will be here

end truck_travel_due_east_distance_l173_173364


namespace maximum_weekly_hours_l173_173404

-- Conditions
def regular_rate : ℝ := 8 -- $8 per hour for the first 20 hours
def overtime_rate : ℝ := regular_rate * 1.25 -- 25% higher than the regular rate
def max_weekly_earnings : ℝ := 460 -- Maximum of $460 in a week
def regular_hours : ℕ := 20 -- First 20 hours are regular hours
def regular_earnings : ℝ := regular_hours * regular_rate -- Earnings for regular hours
def max_overtime_earnings : ℝ := max_weekly_earnings - regular_earnings -- Maximum overtime earnings

-- Proof problem statement
theorem maximum_weekly_hours : regular_hours + (max_overtime_earnings / overtime_rate) = 50 := by
  sorry

end maximum_weekly_hours_l173_173404


namespace first_day_of_month_l173_173442

theorem first_day_of_month (d : ℕ) (h : d = 30) (dow_30 : d % 7 = 3) : (1 % 7 = 2) :=
by sorry

end first_day_of_month_l173_173442


namespace area_of_region_bounded_by_lines_and_y_axis_l173_173964

noncomputable def area_of_triangle_bounded_by_lines : ℝ :=
  let y1 (x : ℝ) := 3 * x - 6
  let y2 (x : ℝ) := -2 * x + 18
  let intersection_x := 24 / 5
  let intersection_y := y1 intersection_x
  let base := 18 + 6
  let height := intersection_x
  1 / 2 * base * height

theorem area_of_region_bounded_by_lines_and_y_axis :
  area_of_triangle_bounded_by_lines = 57.6 :=
by
  sorry

end area_of_region_bounded_by_lines_and_y_axis_l173_173964


namespace train_length_correct_l173_173022

def train_length (speed_kph : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_mps := speed_kph * 1000 / 3600
  speed_mps * time_sec

theorem train_length_correct :
  train_length 90 10 = 250 := by
  sorry

end train_length_correct_l173_173022


namespace carl_lawn_area_l173_173008

theorem carl_lawn_area :
  ∃ (width height : ℤ), 
    (width + 1) + (height + 1) - 4 = 24 ∧
    3 * width = height ∧
    3 * ((width + 1) * 3) * ((height + 1) * 3) = 243 :=
by
  sorry

end carl_lawn_area_l173_173008


namespace smallest_pos_int_mod_congruence_l173_173567

theorem smallest_pos_int_mod_congruence : ∃ n : ℕ, 0 < n ∧ n ≡ 2 [MOD 31] ∧ 5 * n ≡ 409 [MOD 31] :=
by
  sorry

end smallest_pos_int_mod_congruence_l173_173567


namespace inversely_proportional_decrease_l173_173066

theorem inversely_proportional_decrease :
  ∀ {x y q c : ℝ}, 
  0 < x ∧ 0 < y ∧ 0 < c ∧ 0 < q →
  (x * y = c) →
  (((1 + q / 100) * x) * ((100 / (100 + q)) * y) = c) →
  ((y - (100 / (100 + q)) * y) / y) * 100 = 100 * q / (100 + q) :=
by
  intros x y q c hb hxy hxy'
  sorry

end inversely_proportional_decrease_l173_173066


namespace correct_answers_max_l173_173852

def max_correct_answers (c w b : ℕ) : Prop :=
  c + w + b = 25 ∧ 4 * c - 3 * w = 40

theorem correct_answers_max : ∃ c w b : ℕ, max_correct_answers c w b ∧ ∀ c', max_correct_answers c' w b → c' ≤ 13 :=
by
  sorry

end correct_answers_max_l173_173852


namespace tangents_form_rectangle_l173_173477

-- Define the first ellipse
def ellipse1 (a b x y : ℝ) : Prop := x^2 / a^4 + y^2 / b^4 = 1

-- Define the second ellipse
def ellipse2 (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define conjugate diameters through lines
def conjugate_diameters (a b m : ℝ) : Prop := True -- (You might want to further define what conjugate diameters imply here)

-- Prove the main statement
theorem tangents_form_rectangle
  (a b m : ℝ)
  (x1 y1 x2 y2 k1 k2 : ℝ)
  (h1 : ellipse1 a b x1 y1)
  (h2 : ellipse1 a b x2 y2)
  (h3 : ellipse2 a b x1 y1)
  (h4 : ellipse2 a b x2 y2)
  (conj1 : conjugate_diameters a b m)
  (tangent_slope1 : k1 = -b^2 / a^2 * (1 / m))
  (conj2 : conjugate_diameters a b (-b^4/a^4 * 1/m))
  (tangent_slope2 : k2 = -b^4 / a^4 * (1 / (-b^4/a^4 * (1/m))))
: k1 * k2 = -1 :=
sorry

end tangents_form_rectangle_l173_173477


namespace sum_of_numbers_l173_173526

theorem sum_of_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end sum_of_numbers_l173_173526


namespace average_words_per_hour_l173_173395

/-- Prove that given a total of 50,000 words written in 100 hours with the 
writing output increasing by 10% each subsequent hour, the average number 
of words written per hour is 500. -/
theorem average_words_per_hour 
(words_total : ℕ) 
(hours_total : ℕ) 
(increase : ℝ) :
  words_total = 50000 ∧ hours_total = 100 ∧ increase = 0.1 →
  (words_total / hours_total : ℝ) = 500 :=
by 
  intros h
  sorry

end average_words_per_hour_l173_173395


namespace canal_depth_l173_173757

-- Define the problem parameters
def top_width : ℝ := 6
def bottom_width : ℝ := 4
def cross_section_area : ℝ := 10290

-- Define the theorem to prove the depth of the canal
theorem canal_depth :
  (1 / 2) * (top_width + bottom_width) * h = cross_section_area → h = 2058 :=
by sorry

end canal_depth_l173_173757


namespace time_to_reach_rest_area_l173_173069

variable (rate_per_minute : ℕ) (remaining_distance_yards : ℕ)

theorem time_to_reach_rest_area (h_rate : rate_per_minute = 2) (h_distance : remaining_distance_yards = 50) :
  (remaining_distance_yards * 3) / rate_per_minute = 75 := by
  sorry

end time_to_reach_rest_area_l173_173069


namespace product_of_possible_values_l173_173548

theorem product_of_possible_values (x : ℝ) (h : (x + 3) * (x - 4) = 18) : ∃ a b, x = a ∨ x = b ∧ a * b = -30 :=
by 
  sorry

end product_of_possible_values_l173_173548


namespace area_of_triangle_ABF_l173_173199

theorem area_of_triangle_ABF (A B F : ℝ × ℝ) (hF : F = (1, 0)) (hA_parabola : A.2^2 = 4 * A.1) (hB_parabola : B.2^2 = 4 * B.1) (h_midpoint_AB : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 2) : 
  ∃ area : ℝ, area = 2 :=
sorry

end area_of_triangle_ABF_l173_173199


namespace find_f_80_l173_173224

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_relation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  f (x * y) = f x / y^2

axiom f_40 : f 40 = 50

-- Proof that f 80 = 12.5
theorem find_f_80 : f 80 = 12.5 := 
by
  sorry

end find_f_80_l173_173224


namespace intersection_is_correct_l173_173275

-- Conditions definitions
def setA : Set ℝ := {x | 2 < x ∧ x < 8}
def setB : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Intersection definition
def intersection : Set ℝ := {x | 2 < x ∧ x ≤ 6}

-- Theorem statement
theorem intersection_is_correct : setA ∩ setB = intersection := 
by
  sorry

end intersection_is_correct_l173_173275


namespace translate_quadratic_function_l173_173118

theorem translate_quadratic_function :
  ∀ x : ℝ, (y = (1 / 3) * x^2) →
          (y₂ = (1 / 3) * (x - 1)^2) →
          (y₃ = y₂ + 3) →
          y₃ = (1 / 3) * (x - 1)^2 + 3 := 
by 
  intros x h₁ h₂ h₃ 
  sorry

end translate_quadratic_function_l173_173118


namespace smallest_positive_integer_a_l173_173931

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = x

theorem smallest_positive_integer_a :
  ∃ (a : ℕ), 0 < a ∧ (isPerfectSquare (10 + a)) ∧ (isPerfectSquare (10 * a)) ∧ 
  ∀ b : ℕ, 0 < b ∧ (isPerfectSquare (10 + b)) ∧ (isPerfectSquare (10 * b)) → a ≤ b :=
sorry

end smallest_positive_integer_a_l173_173931


namespace complex_square_eq_l173_173682

theorem complex_square_eq (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I) : 
  a + b * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end complex_square_eq_l173_173682


namespace max_ab_l173_173055

theorem max_ab (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 3 ≤ a + b ∧ a + b ≤ 4) : ab ≤ 15 / 4 :=
sorry

end max_ab_l173_173055


namespace train_speeds_l173_173449

-- Definitions used in conditions
def initial_distance : ℝ := 300
def time_elapsed : ℝ := 2
def remaining_distance : ℝ := 40
def speed_difference : ℝ := 10

-- Stating the problem in Lean
theorem train_speeds :
  ∃ (v_fast v_slow : ℝ),
    v_slow + speed_difference = v_fast ∧
    (2 * (v_slow + v_fast)) = (initial_distance - remaining_distance) ∧
    v_slow = 60 ∧
    v_fast = 70 :=
by
  sorry

end train_speeds_l173_173449


namespace correct_equation_l173_173025

theorem correct_equation (x : ℝ) :
  232 + x = 3 * (146 - x) :=
sorry

end correct_equation_l173_173025


namespace decimal_sum_sqrt_l173_173644

theorem decimal_sum_sqrt (a b : ℝ) (h₁ : a = Real.sqrt 5 - 2) (h₂ : b = Real.sqrt 13 - 3) : 
  a + b - Real.sqrt 5 = Real.sqrt 13 - 5 := by
  sorry

end decimal_sum_sqrt_l173_173644


namespace kanul_initial_amount_l173_173915

theorem kanul_initial_amount (X Y : ℝ) (loan : ℝ) (R : ℝ) 
  (h1 : loan = 2000)
  (h2 : R = 0.20)
  (h3 : Y = 0.15 * X + loan)
  (h4 : loan = R * Y) : 
  X = 53333.33 :=
by 
  -- The proof would come here, but is not necessary for this example
sorry

end kanul_initial_amount_l173_173915


namespace three_digit_condition_l173_173190

-- Define the three-digit number and its rotated variants
def num (a b c : ℕ) := 100 * a + 10 * b + c
def num_bca (a b c : ℕ) := 100 * b + 10 * c + a
def num_cab (a b c : ℕ) := 100 * c + 10 * a + b

-- The main statement to prove
theorem three_digit_condition (a b c: ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) (h_c : 0 ≤ c ∧ c ≤ 9) :
  2 * num a b c = num_bca a b c + num_cab a b c ↔ 
  (num a b c = 111 ∨ num a b c = 222 ∨ 
  num a b c = 333 ∨ num a b c = 370 ∨ 
  num a b c = 407 ∨ num a b c = 444 ∨ 
  num a b c = 481 ∨ num a b c = 518 ∨ 
  num a b c = 555 ∨ num a b c = 592 ∨ 
  num a b c = 629 ∨ num a b c = 666 ∨ 
  num a b c = 777 ∨ num a b c = 888 ∨ 
  num a b c = 999) := by
  sorry

end three_digit_condition_l173_173190


namespace cyclist_rejoins_group_time_l173_173616

noncomputable def travel_time (group_speed cyclist_speed distance : ℝ) : ℝ :=
  distance / (cyclist_speed - group_speed)

theorem cyclist_rejoins_group_time
  (group_speed : ℝ := 35)
  (cyclist_speed : ℝ := 45)
  (distance : ℝ := 10)
  : travel_time group_speed cyclist_speed distance * 2 = 1 / 4 :=
by
  sorry

end cyclist_rejoins_group_time_l173_173616


namespace ab_neither_sufficient_nor_necessary_l173_173114

theorem ab_neither_sufficient_nor_necessary (a b : ℝ) (h : a * b ≠ 0) :
  (¬ ((a * b > 1) → (a > 1 / b))) ∧ (¬ ((a > 1 / b) → (a * b > 1))) :=
by
  sorry

end ab_neither_sufficient_nor_necessary_l173_173114


namespace f_irreducible_l173_173561

noncomputable def f (n : ℕ) (x : ℤ) : ℤ := x^n + 5 * x^(n-1) + 3

theorem f_irreducible (n : ℕ) (hn : n > 1) : Irreducible (f n) :=
sorry

end f_irreducible_l173_173561


namespace find_a1_l173_173416

open Nat

theorem find_a1 (a : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n)
  (h2 : a 3 = 12) : a 1 = 3 :=
sorry

end find_a1_l173_173416


namespace fourth_term_geometric_series_l173_173858

theorem fourth_term_geometric_series (a₁ a₅ : ℕ) (r : ℕ) :
  a₁ = 6 → a₅ = 1458 → (∀ n, aₙ = a₁ * r^(n-1)) → r = 3 → (∃ a₄, a₄ = a₁ * r^(4-1) ∧ a₄ = 162) :=
by intros h₁ h₅ H r_sol
   sorry

end fourth_term_geometric_series_l173_173858


namespace corn_bag_price_l173_173586

theorem corn_bag_price
  (cost_seeds: ℕ)
  (cost_fertilizers_pesticides: ℕ)
  (cost_labor: ℕ)
  (total_bags: ℕ)
  (desired_profit_percentage: ℕ)
  (total_cost: ℕ := cost_seeds + cost_fertilizers_pesticides + cost_labor)
  (total_revenue: ℕ := total_cost + (total_cost * desired_profit_percentage / 100))
  (price_per_bag: ℕ := total_revenue / total_bags) :
  cost_seeds = 50 →
  cost_fertilizers_pesticides = 35 →
  cost_labor = 15 →
  total_bags = 10 →
  desired_profit_percentage = 10 →
  price_per_bag = 11 :=
by sorry

end corn_bag_price_l173_173586


namespace find_divisor_l173_173648

-- Define the conditions
def dividend : ℕ := 22
def quotient : ℕ := 7
def remainder : ℕ := 1

-- The divisor is what we need to find
def divisor : ℕ := 3

-- The proof problem: proving that the given conditions imply the divisor is 3
theorem find_divisor :
  ∃ d : ℕ, dividend = d * quotient + remainder ∧ d = divisor :=
by
  use 3
  -- Replace actual proof with sorry for now
  sorry

end find_divisor_l173_173648


namespace systematic_sampling_third_group_draw_l173_173178

theorem systematic_sampling_third_group_draw
  (first_draw : ℕ) (second_draw : ℕ) (first_draw_eq : first_draw = 2)
  (second_draw_eq : second_draw = 12) :
  ∃ (third_draw : ℕ), third_draw = 22 :=
by
  sorry

end systematic_sampling_third_group_draw_l173_173178


namespace total_revenue_is_correct_l173_173847

-- Define the constants and conditions
def price_of_jeans : ℕ := 11
def price_of_tees : ℕ := 8
def quantity_of_tees_sold : ℕ := 7
def quantity_of_jeans_sold : ℕ := 4

-- Define the total revenue calculation
def total_revenue : ℕ :=
  (price_of_tees * quantity_of_tees_sold) +
  (price_of_jeans * quantity_of_jeans_sold)

-- The theorem to prove
theorem total_revenue_is_correct : total_revenue = 100 := 
by
  -- Proof is omitted for now
  sorry

end total_revenue_is_correct_l173_173847


namespace invalid_root_l173_173172

theorem invalid_root (a_1 a_0 : ℤ) : ¬(19 * (1/7 : ℚ)^3 + 98 * (1/7 : ℚ)^2 + a_1 * (1/7 : ℚ) + a_0 = 0) :=
by 
  sorry

end invalid_root_l173_173172


namespace peter_vacation_saving_l173_173360

theorem peter_vacation_saving :
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  months_needed = 3 :=
by
  -- definitions
  let goal := 5000
  let current_savings := 2900
  let monthly_savings := 700
  let total_needed := goal - current_savings
  let months_needed := total_needed / monthly_savings
  -- proof
  sorry

end peter_vacation_saving_l173_173360


namespace root_properties_of_polynomial_l173_173346

variables {r s t : ℝ}

def polynomial (x : ℝ) : ℝ := 6 * x^3 + 4 * x^2 + 1500 * x + 3000

theorem root_properties_of_polynomial :
  (∀ x : ℝ, polynomial x = 0 → (x = r ∨ x = s ∨ x = t)) →
  (r + s + t = -2 / 3) →
  (r * s + r * t + s * t = 250) →
  (r * s * t = -500) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = -5992 / 27 :=
by
  sorry

end root_properties_of_polynomial_l173_173346


namespace express_in_scientific_notation_l173_173049

theorem express_in_scientific_notation 
  (A : 149000000 = 149 * 10^6)
  (B : 149000000 = 1.49 * 10^8)
  (C : 149000000 = 14.9 * 10^7)
  (D : 149000000 = 1.5 * 10^8) :
  149000000 = 1.49 * 10^8 := 
by
  sorry

end express_in_scientific_notation_l173_173049


namespace inequality_proof_l173_173493

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l173_173493


namespace right_triangle_cos_pq_l173_173900

theorem right_triangle_cos_pq (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : c = 13) (h2 : b / c = 5/13) : a = 12 :=
by
  sorry

end right_triangle_cos_pq_l173_173900


namespace suraj_average_after_13th_innings_l173_173290

theorem suraj_average_after_13th_innings
  (A : ℝ)
  (h : (12 * A + 96) / 13 = A + 5) :
  (12 * A + 96) / 13 = 36 :=
by
  sorry

end suraj_average_after_13th_innings_l173_173290


namespace find_x_l173_173450

def side_of_square_eq_twice_radius_of_larger_circle (s: ℝ) (r_l: ℝ) : Prop :=
  s = 2 * r_l

def radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle (r_l: ℝ) (x: ℝ) (r_s: ℝ) : Prop :=
  r_l = x - (1 / 3) * r_s

def circumference_of_smaller_circle_eq (r_s: ℝ) (circumference: ℝ) : Prop :=
  2 * Real.pi * r_s = circumference

def side_squared_eq_area (s: ℝ) (area: ℝ) : Prop :=
  s^2 = area

noncomputable def value_of_x (r_s r_l: ℝ) : ℝ :=
  14 + 4 / (3 * Real.pi)

theorem find_x 
  (s r_l r_s x: ℝ)
  (h1: side_squared_eq_area s 784)
  (h2: side_of_square_eq_twice_radius_of_larger_circle s r_l)
  (h3: radius_of_larger_circle_eq_x_minus_third_radius_of_smaller_circle r_l x r_s)
  (h4: circumference_of_smaller_circle_eq r_s 8) :
  x = value_of_x r_s r_l :=
sorry

end find_x_l173_173450


namespace geom_seq_product_arith_seq_l173_173026

theorem geom_seq_product_arith_seq (a b c r : ℝ) (h1 : c = b * r)
  (h2 : b = a * r)
  (h3 : a * b * c = 512)
  (h4 : b = 8)
  (h5 : 2 * b = (a - 2) + (c - 2)) :
  (a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = 16 ∧ b = 8 ∧ c = 4) :=
by
  sorry

end geom_seq_product_arith_seq_l173_173026


namespace simplify_expression_l173_173739

theorem simplify_expression (a : Int) : 2 * a - a = a :=
by
  sorry

end simplify_expression_l173_173739


namespace population_of_missing_village_l173_173265

theorem population_of_missing_village 
  (p1 p2 p3 p4 p5 p6 : ℕ) 
  (h1 : p1 = 803) 
  (h2 : p2 = 900) 
  (h3 : p3 = 1100) 
  (h4 : p4 = 1023) 
  (h5 : p5 = 945) 
  (h6 : p6 = 1249) 
  (avg_population : ℕ) 
  (h_avg : avg_population = 1000) :
  ∃ p7 : ℕ, p7 = 980 ∧ avg_population * 7 = p1 + p2 + p3 + p4 + p5 + p6 + p7 :=
by
  sorry

end population_of_missing_village_l173_173265


namespace candy_crush_ratio_l173_173886

theorem candy_crush_ratio :
  ∃ m : ℕ, (400 + (400 - 70) + (400 - 70) * m = 1390) ∧ (m = 2) :=
by
  sorry

end candy_crush_ratio_l173_173886


namespace periodicity_of_m_arith_fibonacci_l173_173874

def m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) : Prop :=
∀ n : ℕ, v (n + 2) = (v n + v (n + 1)) % m

theorem periodicity_of_m_arith_fibonacci (m : ℕ) (v : ℕ → ℕ) 
  (hv : m_arith_fibonacci m v) : 
  ∃ r : ℕ, r ≤ m^2 ∧ ∀ n : ℕ, v (n + r) = v n := 
by
  sorry

end periodicity_of_m_arith_fibonacci_l173_173874


namespace half_angle_in_first_quadrant_l173_173202

theorem half_angle_in_first_quadrant {α : ℝ} (h : 0 < α ∧ α < π / 2) : 
  0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l173_173202


namespace incircle_tangent_distance_l173_173013

theorem incircle_tangent_distance (a b c : ℝ) (M : ℝ) (BM : ℝ) (x1 y1 z1 x2 y2 z2 : ℝ) 
  (h1 : BM = y1 + z1)
  (h2 : BM = y2 + z2)
  (h3 : x1 + y1 = x2 + y2)
  (h4 : x1 + z1 = c)
  (h5 : x2 + z2 = a) :
  |y1 - y2| = |(a - c) / 2| := by 
  sorry

end incircle_tangent_distance_l173_173013


namespace olympiad_permutations_l173_173340

theorem olympiad_permutations : 
  let total_permutations := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2) 
  let invalid_permutations := 5 * (Nat.factorial 4 / Nat.factorial 2)
  total_permutations - invalid_permutations = 90660 :=
by
  let total_permutations : ℕ := Nat.factorial 9 / (Nat.factorial 2 * Nat.factorial 2)
  let invalid_permutations : ℕ := 5 * (Nat.factorial 4 / Nat.factorial 2)
  show total_permutations - invalid_permutations = 90660
  sorry

end olympiad_permutations_l173_173340


namespace gcd_39_91_l173_173489
-- Import the Mathlib library to ensure all necessary functions and theorems are available

-- Lean statement for proving the GCD of 39 and 91 is 13.
theorem gcd_39_91 : Nat.gcd 39 91 = 13 := by
  sorry

end gcd_39_91_l173_173489


namespace logarithmic_expression_l173_173209

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem logarithmic_expression :
  let log2 := lg 2
  let log5 := lg 5
  log2 + log5 = 1 →
  (log2^3 + 3 * log2 * log5 + log5^3 = 1) :=
by
  intros log2 log5 h
  sorry

end logarithmic_expression_l173_173209


namespace max_theater_members_l173_173901

theorem max_theater_members (N : ℕ) :
  (∃ (k : ℕ), (N = k^2 + 3)) ∧ (∃ (n : ℕ), (N = n * (n + 9))) → N ≤ 360 :=
by
  sorry

end max_theater_members_l173_173901


namespace johns_total_payment_l173_173240

theorem johns_total_payment :
  let silverware_cost := 20
  let dinner_plate_cost := 0.5 * silverware_cost
  let total_cost := dinner_plate_cost + silverware_cost
  total_cost = 30 := sorry

end johns_total_payment_l173_173240


namespace perpendicular_line_sufficient_condition_l173_173979

theorem perpendicular_line_sufficient_condition (a : ℝ) :
  (-a) * ((a + 2) / 3) = -1 ↔ (a = -3 ∨ a = 1) :=
by {
  sorry
}

#print perpendicular_line_sufficient_condition

end perpendicular_line_sufficient_condition_l173_173979


namespace factorize_x4_plus_16_l173_173667

theorem factorize_x4_plus_16 :
  ∀ x : ℝ, (x^4 + 16) = (x^2 - 2 * x + 2) * (x^2 + 2 * x + 2) :=
by
  intro x
  sorry

end factorize_x4_plus_16_l173_173667


namespace func_passes_through_fixed_point_l173_173277

theorem func_passes_through_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  a^(2 * (1 / 2) - 1) = 1 :=
by
  sorry

end func_passes_through_fixed_point_l173_173277


namespace product_of_consecutive_integers_even_l173_173427

theorem product_of_consecutive_integers_even (n : ℤ) : Even (n * (n + 1)) :=
sorry

end product_of_consecutive_integers_even_l173_173427


namespace circular_patch_radius_l173_173727

theorem circular_patch_radius : 
  let r_cylinder := 3  -- radius of the container in cm
  let h_cylinder := 6  -- height of the container in cm
  let t_patch := 0.2   -- thickness of each patch in cm
  let V := π * r_cylinder^2 * h_cylinder -- Volume of the liquid

  let V_patch := V / 2                  -- Volume of each patch
  let r := 3 * Real.sqrt 15              -- the radius we want to prove

  r^2 * π * t_patch = V_patch           -- the volume equation for one patch
  →

  r = 3 * Real.sqrt 15 := 
by
  sorry

end circular_patch_radius_l173_173727


namespace find_value_l173_173619

theorem find_value : (1 / 4 * (5 * 9 * 4) - 7) = 38 := 
by
  sorry

end find_value_l173_173619


namespace calculation_l173_173904

-- Define the exponents and base values as conditions
def exponent : ℕ := 3 ^ 2
def neg_base : ℤ := -2
def pos_base : ℤ := 2

-- The calculation expressions as conditions
def term1 : ℤ := neg_base^exponent
def term2 : ℤ := pos_base^exponent

-- The proof statement: Show that the sum of the terms equals 0
theorem calculation : term1 + term2 = 0 := sorry

end calculation_l173_173904


namespace common_difference_of_arithmetic_sequence_l173_173012

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (n : ℕ) (an : ℕ → α) : α :=
  (n : α) * an 1 + (n * (n - 1) / 2 * (an 2 - an 1))

theorem common_difference_of_arithmetic_sequence (S : ℕ → ℕ) (d : ℕ) (a1 a2 : ℕ)
  (h1 : ∀ n, S n = 4 * n ^ 2 - n)
  (h2 : a1 = S 1)
  (h3 : a2 = S 2 - S 1) :
  d = a2 - a1 → d = 8 := by
  sorry

end common_difference_of_arithmetic_sequence_l173_173012


namespace exists_distinct_numbers_divisible_by_3_l173_173587

-- Define the problem in Lean with the given conditions and goal.
theorem exists_distinct_numbers_divisible_by_3 : 
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0 ∧ d % 3 = 0 ∧
  (a + b + c) % d = 0 ∧ (a + b + d) % c = 0 ∧ (a + c + d) % b = 0 ∧ (b + c + d) % a = 0 :=
by
  sorry

end exists_distinct_numbers_divisible_by_3_l173_173587


namespace complement_intersection_l173_173158

noncomputable def real_universal_set : Set ℝ := Set.univ

noncomputable def set_A (x : ℝ) : Prop := x + 1 < 0
def A : Set ℝ := {x | set_A x}

noncomputable def set_B (x : ℝ) : Prop := x - 3 < 0
def B : Set ℝ := {x | set_B x}

noncomputable def complement_A : Set ℝ := {x | ¬set_A x}

noncomputable def intersection (S₁ S₂ : Set ℝ) : Set ℝ := {x | x ∈ S₁ ∧ x ∈ S₂}

theorem complement_intersection :
  intersection complement_A B = {x | -1 ≤ x ∧ x < 3} :=
sorry

end complement_intersection_l173_173158


namespace number_of_puppies_sold_l173_173121

variables (P : ℕ) (p_0 : ℕ) (k_0 : ℕ) (r : ℕ) (k_s : ℕ)

theorem number_of_puppies_sold 
  (h1 : p_0 = 7) 
  (h2 : k_0 = 6) 
  (h3 : r = 8) 
  (h4 : k_s = 3) : 
  P = p_0 - (r - (k_0 - k_s)) :=
by sorry

end number_of_puppies_sold_l173_173121


namespace cauchy_inequality_minimum_value_inequality_l173_173286

-- Part 1: Prove Cauchy Inequality
theorem cauchy_inequality (a b x y : ℝ) : 
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

-- Part 2: Find the minimum value under the given conditions
theorem minimum_value_inequality (x y : ℝ) (h₁ : x^2 + y^2 = 2) (h₂ : x ≠ y ∨ x ≠ -y) : 
  ∃ m, m = (1 / (9 * x^2) + 9 / y^2) ∧ m = 50 / 9 :=
by
  sorry

end cauchy_inequality_minimum_value_inequality_l173_173286


namespace proof_problem_l173_173095

theorem proof_problem 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) : 
  |a / b + b / a| ≥ 2 := 
sorry

end proof_problem_l173_173095


namespace adam_initial_money_l173_173284

theorem adam_initial_money :
  let cost_of_airplane := 4.28
  let change_received := 0.72
  cost_of_airplane + change_received = 5.00 :=
by
  sorry

end adam_initial_money_l173_173284


namespace initial_people_on_train_l173_173814

theorem initial_people_on_train 
    (P : ℕ)
    (h1 : 116 = P - 4)
    (h2 : P = 120)
    : 
    P = 116 + 4 := by
have h3 : P = 120 := by sorry
exact h3

end initial_people_on_train_l173_173814


namespace matrix_problem_l173_173143

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)
variable (I : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = !![2, 1; 4, 3]) :
  B * A = !![2, 1; 4, 3] :=
sorry

end matrix_problem_l173_173143


namespace gallons_needed_to_grandmas_house_l173_173371

def car_fuel_efficiency : ℝ := 20
def distance_to_grandmas_house : ℝ := 100

theorem gallons_needed_to_grandmas_house : (distance_to_grandmas_house / car_fuel_efficiency) = 5 :=
by
  sorry

end gallons_needed_to_grandmas_house_l173_173371


namespace symmetric_pattern_count_l173_173536

noncomputable def number_of_symmetric_patterns (n : ℕ) : ℕ :=
  let regions := 12
  let total_patterns := 2^regions
  total_patterns - 2

theorem symmetric_pattern_count : number_of_symmetric_patterns 8 = 4094 :=
by
  sorry

end symmetric_pattern_count_l173_173536


namespace min_band_members_exists_l173_173528

theorem min_band_members_exists (n : ℕ) :
  (∃ n, (∃ k : ℕ, n = 9 * k) ∧ (∃ m : ℕ, n = 10 * m) ∧ (∃ p : ℕ, n = 11 * p)) → n = 990 :=
by
  sorry

end min_band_members_exists_l173_173528


namespace max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l173_173211

namespace Geometry

variables {x y : ℝ}

-- Given condition
def satisfies_circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * y + 1 = 0

-- Proof problems
theorem max_x_plus_y (h : satisfies_circle x y) : 
  x + y ≤ 2 + Real.sqrt 6 :=
sorry

theorem range_y_plus_1_over_x (h : satisfies_circle x y) : 
  -Real.sqrt 2 ≤ (y + 1) / x ∧ (y + 1) / x ≤ Real.sqrt 2 :=
sorry

theorem extrema_x2_minus_2x_plus_y2_plus_1 (h : satisfies_circle x y) : 
  8 - 2 * Real.sqrt 15 ≤ x^2 - 2 * x + y^2 + 1 ∧ x^2 - 2 * x + y^2 + 1 ≤ 8 + 2 * Real.sqrt 15 :=
sorry

end Geometry

end max_x_plus_y_range_y_plus_1_over_x_extrema_x2_minus_2x_plus_y2_plus_1_l173_173211


namespace digit_d_multiple_of_9_l173_173617

theorem digit_d_multiple_of_9 (d : ℕ) (hd : d = 1) : ∃ k : ℕ, (56780 + d) = 9 * k := by
  have : 56780 + d = 56780 + 1 := by rw [hd]
  rw [this]
  use 6313
  sorry

end digit_d_multiple_of_9_l173_173617


namespace minimum_sum_of_box_dimensions_l173_173393

theorem minimum_sum_of_box_dimensions :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end minimum_sum_of_box_dimensions_l173_173393


namespace value_of_a_l173_173356

theorem value_of_a (a b : ℝ) (h1 : b = 2120) (h2 : a / b = 0.5) : a = 1060 := 
by
  sorry

end value_of_a_l173_173356


namespace Mitch_saved_amount_l173_173362

theorem Mitch_saved_amount :
  let boat_cost_per_foot := 1500
  let license_and_registration := 500
  let docking_fees := 3 * 500
  let longest_boat_length := 12
  let total_license_and_fees := license_and_registration + docking_fees
  let total_boat_cost := boat_cost_per_foot * longest_boat_length
  let total_saved := total_boat_cost + total_license_and_fees
  total_saved = 20000 :=
by
  sorry

end Mitch_saved_amount_l173_173362


namespace volume_Q3_l173_173850

def Q0 : ℚ := 8
def delta : ℚ := (1 / 3) ^ 3
def ratio : ℚ := 6 / 27

def Q (i : ℕ) : ℚ :=
  match i with
  | 0 => Q0
  | 1 => Q0 + 4 * delta
  | n + 1 => Q n + delta * (ratio ^ n)

theorem volume_Q3 : Q 3 = 5972 / 729 := 
by
  sorry

end volume_Q3_l173_173850


namespace oranges_left_uneaten_l173_173577

variable (total_oranges : ℕ)
variable (half_oranges ripe_oranges unripe_oranges eaten_ripe_oranges eaten_unripe_oranges uneaten_ripe_oranges uneaten_unripe_oranges total_uneaten_oranges : ℕ)

axiom h1 : total_oranges = 96
axiom h2 : half_oranges = total_oranges / 2
axiom h3 : ripe_oranges = half_oranges
axiom h4 : unripe_oranges = half_oranges
axiom h5 : eaten_ripe_oranges = ripe_oranges / 4
axiom h6 : eaten_unripe_oranges = unripe_oranges / 8
axiom h7 : uneaten_ripe_oranges = ripe_oranges - eaten_ripe_oranges
axiom h8 : uneaten_unripe_oranges = unripe_oranges - eaten_unripe_oranges
axiom h9 : total_uneaten_oranges = uneaten_ripe_oranges + uneaten_unripe_oranges

theorem oranges_left_uneaten : total_uneaten_oranges = 78 := by
  sorry

end oranges_left_uneaten_l173_173577


namespace sum_of_factors_of_30_is_72_l173_173358

-- Condition: given the number 30
def number := 30

-- Define the positive factors of 30
def factors : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- Statement to prove the sum of the positive factors
theorem sum_of_factors_of_30_is_72 : (factors.sum) = 72 := 
by
  sorry

end sum_of_factors_of_30_is_72_l173_173358


namespace roots_of_transformed_quadratic_l173_173103

variable {a b c : ℝ}

theorem roots_of_transformed_quadratic
    (h₁: a ≠ 0)
    (h₂: ∀ x, a * (x - 1)^2 - 1 = ax^2 + bx + c - 1)
    (h₃: ax^2 + bx + c = -1) :
    (x = 1) ∧ (x = 1) := 
  sorry

end roots_of_transformed_quadratic_l173_173103


namespace unique_very_set_on_line_l173_173113

def very_set (S : Finset (ℝ × ℝ)) : Prop :=
  ∀ X ∈ S, ∃ (r : ℝ), 
  ∀ Y ∈ S, Y ≠ X → ∃ Z ∈ S, Z ≠ X ∧ r * r = dist X Y * dist X Z

theorem unique_very_set_on_line (n : ℕ) (A B : ℝ × ℝ) (S1 S2 : Finset (ℝ × ℝ))
  (h : 2 ≤ n) (hA1 : A ∈ S1) (hB1 : B ∈ S1) (hA2 : A ∈ S2) (hB2 : B ∈ S2)
  (hS1 : S1.card = n) (hS2 : S2.card = n) (hV1 : very_set S1) (hV2 : very_set S2) :
  S1 = S2 := 
sorry

end unique_very_set_on_line_l173_173113


namespace students_exceed_guinea_pigs_and_teachers_l173_173832

def num_students_per_classroom : Nat := 25
def num_guinea_pigs_per_classroom : Nat := 3
def num_teachers_per_classroom : Nat := 1
def num_classrooms : Nat := 5

def total_students : Nat := num_students_per_classroom * num_classrooms
def total_guinea_pigs : Nat := num_guinea_pigs_per_classroom * num_classrooms
def total_teachers : Nat := num_teachers_per_classroom * num_classrooms
def total_guinea_pigs_and_teachers : Nat := total_guinea_pigs + total_teachers

theorem students_exceed_guinea_pigs_and_teachers :
  total_students - total_guinea_pigs_and_teachers = 105 :=
by
  sorry

end students_exceed_guinea_pigs_and_teachers_l173_173832


namespace evaluate_expression_l173_173722

theorem evaluate_expression (x : ℤ) (h1 : 0 ≤ x ∧ x ≤ 2) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x = 0) :
    ( ((4 - x) / (x - 1) - x) / ((x - 2) / (x - 1)) ) = -2 :=
by
    sorry

end evaluate_expression_l173_173722


namespace union_of_M_and_N_l173_173749

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def compl_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_of_M_and_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} :=
sorry

end union_of_M_and_N_l173_173749


namespace tap_C_fills_in_6_l173_173348

-- Definitions for the rates at which taps fill the tank
def rate_A := 1/10
def rate_B := 1/15
def rate_combined := 1/3

-- Proof problem: Given the conditions, prove that the third tap fills the tank in 6 hours
theorem tap_C_fills_in_6 (rate_A rate_B rate_combined : ℚ) (h : rate_A + rate_B + 1/x = rate_combined) : x = 6 :=
sorry

end tap_C_fills_in_6_l173_173348


namespace range_of_m_l173_173494

theorem range_of_m (m : ℝ) : (2 + m > 0) ∧ (1 - m > 0) ∧ (2 + m > 1 - m) → -1/2 < m ∧ m < 1 :=
by
  intros h
  sorry

end range_of_m_l173_173494


namespace standard_robot_weight_l173_173377

variable (S : ℕ) -- Define the variable for the standard robot's weight
variable (MaxWeight : ℕ := 210) -- Define the variable for the maximum weight of a robot, which is 210 pounds
variable (MinWeight : ℕ) -- Define the variable for the minimum weight of the robot

theorem standard_robot_weight (h1 : 2 * MinWeight ≥ MaxWeight) 
                             (h2 : MinWeight = S + 5) 
                             (h3 : MaxWeight = 210) :
  100 ≤ S ∧ S ≤ 105 := 
by
  sorry

end standard_robot_weight_l173_173377


namespace complex_number_solution_l173_173411

open Complex

theorem complex_number_solution (z : ℂ) (h : z^2 = -99 - 40 * I) : z = 2 - 10 * I ∨ z = -2 + 10 * I :=
sorry

end complex_number_solution_l173_173411


namespace largest_n_satisfying_conditions_l173_173205

theorem largest_n_satisfying_conditions : 
  ∃ n : ℤ, 200 < n ∧ n < 250 ∧ (∃ k : ℤ, 12 * n = k^2) ∧ n = 243 :=
by
  sorry

end largest_n_satisfying_conditions_l173_173205


namespace farmer_feed_total_cost_l173_173622

/-- 
A farmer spent $35 on feed for chickens and goats. He spent 40% of the money on chicken feed, which he bought at a 50% discount off the full price, and spent the rest on goat feed, which he bought at full price. Prove that if the farmer had paid full price for both the chicken feed and the goat feed, he would have spent $49.
-/
theorem farmer_feed_total_cost
  (total_spent : ℝ := 35)
  (chicken_feed_fraction : ℝ := 0.40)
  (goat_feed_fraction : ℝ := 0.60)
  (discount : ℝ := 0.50)
  (chicken_feed_discounted : ℝ := chicken_feed_fraction * total_spent)
  (chicken_feed_full_price : ℝ := chicken_feed_discounted / (1 - discount))
  (goat_feed_full_price : ℝ := goat_feed_fraction * total_spent):
  chicken_feed_full_price + goat_feed_full_price = 49 := 
sorry

end farmer_feed_total_cost_l173_173622


namespace three_x_plus_y_eq_zero_l173_173592

theorem three_x_plus_y_eq_zero (x y : ℝ) (h : (2 * x + y) ^ 3 + x ^ 3 + 3 * x + y = 0) : 3 * x + y = 0 :=
sorry

end three_x_plus_y_eq_zero_l173_173592


namespace coat_shirt_ratio_l173_173398

variable (P S C k : ℕ)

axiom h1 : P + S = 100
axiom h2 : P + C = 244
axiom h3 : C = k * S
axiom h4 : C = 180

theorem coat_shirt_ratio (P S C k : ℕ) (h1 : P + S = 100) (h2 : P + C = 244) (h3 : C = k * S) (h4 : C = 180) :
  C / S = 5 :=
sorry

end coat_shirt_ratio_l173_173398


namespace difference_of_numbers_l173_173041

theorem difference_of_numbers : 
  ∃ (L S : ℕ), L = 1631 ∧ L = 6 * S + 35 ∧ L - S = 1365 := 
by
  sorry

end difference_of_numbers_l173_173041


namespace arithmetic_mean_of_fractions_l173_173151

def mean (a b : ℚ) : ℚ := (a + b) / 2

theorem arithmetic_mean_of_fractions (a b c : ℚ) (h₁ : a = 8/11)
                                      (h₂ : b = 5/6) (h₃ : c = 19/22) :
  mean a c = b :=
by
  sorry

end arithmetic_mean_of_fractions_l173_173151


namespace difference_between_x_and_y_is_36_l173_173064

theorem difference_between_x_and_y_is_36 (x y : ℤ) (h1 : x + y = 20) (h2 : x = 28) : x - y = 36 := 
by 
  sorry

end difference_between_x_and_y_is_36_l173_173064


namespace minimum_value_proof_l173_173752

variables {A B C : ℝ}
variable (triangle_ABC : 
  ∀ {A B C : ℝ}, 
  (A > 0 ∧ A < π / 2) ∧ 
  (B > 0 ∧ B < π / 2) ∧ 
  (C > 0 ∧ C < π / 2))

noncomputable def minimum_value (A B C : ℝ) :=
  3 * (Real.tan B) * (Real.tan C) + 
  2 * (Real.tan A) * (Real.tan C) + 
  1 * (Real.tan A) * (Real.tan B)

theorem minimum_value_proof (h : 
  ∀ (A B C : ℝ), 
  (1 / (Real.tan A * Real.tan B)) + 
  (1 / (Real.tan B * Real.tan C)) + 
  (1 / (Real.tan C * Real.tan A)) = 1) 
  : minimum_value A B C = 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_proof_l173_173752


namespace a_friend_gcd_l173_173120

theorem a_friend_gcd (a b : ℕ) (d : ℕ) (hab : a * b = d * d) (hd : d = Nat.gcd a b) : ∃ k : ℕ, a * d = k * k := by
  sorry

end a_friend_gcd_l173_173120


namespace greatest_integer_b_l173_173385

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem greatest_integer_b (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 12 ≠ 0) ↔ b = 6 := 
by
  sorry

end greatest_integer_b_l173_173385


namespace proof_problem_l173_173430

-- Definitions of the propositions
def p : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → y = 5 - 3 * x
def q : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → 2 * x + 6 * y - 4 = 0

-- Translate the mathematical proof problem into a Lean theorem
theorem proof_problem : 
  (p ∧ ¬q) ∧ ¬((¬p) ∧ q) :=
by
  -- You can fill in the exact proof steps here
  sorry

end proof_problem_l173_173430


namespace line_equation_l173_173230

theorem line_equation {x y : ℝ} (h : (x = 1) ∧ (y = -3)) :
  ∃ c : ℝ, x - 2 * y + c = 0 ∧ c = 7 :=
by
  sorry

end line_equation_l173_173230


namespace incorrect_calculation_l173_173112

theorem incorrect_calculation :
    (5 / 8 + (-7 / 12) ≠ -1 / 24) :=
by
  sorry

end incorrect_calculation_l173_173112


namespace cheating_percentage_l173_173024

theorem cheating_percentage (x : ℝ) :
  (∀ cost_price : ℝ, cost_price = 100 →
   let received_when_buying : ℝ := cost_price * (1 + x / 100)
   let given_when_selling : ℝ := cost_price * (1 - x / 100)
   let profit : ℝ := received_when_buying - given_when_selling
   let profit_percentage : ℝ := profit / cost_price
   profit_percentage = 2 / 9) →
  x = 22.22222222222222 := 
by
  sorry

end cheating_percentage_l173_173024


namespace rational_sum_of_squares_is_square_l173_173409

theorem rational_sum_of_squares_is_square (a b c : ℚ) :
  ∃ r : ℚ, r ^ 2 = (1 / (b - c) ^ 2 + 1 / (c - a) ^ 2 + 1 / (a - b) ^ 2) :=
by
  sorry

end rational_sum_of_squares_is_square_l173_173409


namespace find_m_l173_173661

-- Definitions based on conditions
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def are_roots_of_quadratic (b c m : ℝ) : Prop :=
  b * c = 6 - m ∧ b + c = -(m + 2)

-- The theorem statement
theorem find_m {a b c m : ℝ} (h₁ : a = 5) (h₂ : is_isosceles_triangle a b c) (h₃ : are_roots_of_quadratic b c m) : m = -10 :=
sorry

end find_m_l173_173661


namespace final_quantity_of_milk_l173_173675

-- Define initial conditions
def initial_volume : ℝ := 60
def removed_volume : ℝ := 9

-- Given the initial conditions, calculate the quantity of milk left after two dilutions
theorem final_quantity_of_milk :
  let first_removal_ratio := initial_volume - removed_volume / initial_volume
  let first_milk_volume := initial_volume * (first_removal_ratio)
  let second_removal_ratio := first_milk_volume / initial_volume
  let second_milk_volume := first_milk_volume * (second_removal_ratio)
  second_milk_volume = 43.35 :=
by
  sorry

end final_quantity_of_milk_l173_173675


namespace set_condition_implies_union_l173_173357

open Set

variable {α : Type*} {M P : Set α}

theorem set_condition_implies_union 
  (h : M ∩ P = P) : M ∪ P = M := 
sorry

end set_condition_implies_union_l173_173357


namespace r_earns_per_day_l173_173036

variables (P Q R S : ℝ)

theorem r_earns_per_day
  (h1 : P + Q + R + S = 240)
  (h2 : P + R + S = 160)
  (h3 : Q + R = 150)
  (h4 : Q + R + S = 650 / 3) :
  R = 70 :=
by
  sorry

end r_earns_per_day_l173_173036


namespace man_l173_173091

theorem man's_rate_in_still_water (speed_with_stream speed_against_stream : ℝ) (h1 : speed_with_stream = 26) (h2 : speed_against_stream = 12) : 
  (speed_with_stream + speed_against_stream) / 2 = 19 := 
by
  rw [h1, h2]
  norm_num

end man_l173_173091


namespace Xiaofang_English_score_l173_173345

/-- Given the conditions about the average scores of Xiaofang's subjects:
  1. The average score for 4 subjects is 88.
  2. The average score for the first 2 subjects is 93.
  3. The average score for the last 3 subjects is 87.
Prove that Xiaofang's English test score is 95. -/
theorem Xiaofang_English_score
    (L M E S : ℝ)
    (h1 : (L + M + E + S) / 4 = 88)
    (h2 : (L + M) / 2 = 93)
    (h3 : (M + E + S) / 3 = 87) :
    E = 95 :=
by
  sorry

end Xiaofang_English_score_l173_173345


namespace felicity_gasoline_usage_l173_173214

def gallons_of_gasoline (G D: ℝ) :=
  G = 2 * D

def combined_volume (M D: ℝ) :=
  M = D - 5

def ethanol_consumption (E M: ℝ) :=
  E = 0.35 * M

def biodiesel_consumption (B M: ℝ) :=
  B = 0.65 * M

def distance_relationship_F_A (F A: ℕ) :=
  A = F + 150

def distance_relationship_F_Bn (F Bn: ℕ) :=
  F = Bn + 50

def total_distance (F A Bn: ℕ) :=
  F + A + Bn = 1750

def gasoline_mileage : ℕ := 35

def diesel_mileage : ℕ := 25

def ethanol_mileage : ℕ := 30

def biodiesel_mileage : ℕ := 20

theorem felicity_gasoline_usage : 
  ∀ (F A Bn: ℕ) (G D M E B: ℝ),
  gallons_of_gasoline G D →
  combined_volume M D →
  ethanol_consumption E M →
  biodiesel_consumption B M →
  distance_relationship_F_A F A →
  distance_relationship_F_Bn F Bn →
  total_distance F A Bn →
  G = 56
  := by
    intros
    sorry

end felicity_gasoline_usage_l173_173214


namespace number_of_levels_l173_173310

-- Definitions of the conditions
def blocks_per_step : ℕ := 3
def steps_per_level : ℕ := 8
def total_blocks_climbed : ℕ := 96

-- The theorem to prove
theorem number_of_levels : (total_blocks_climbed / blocks_per_step) / steps_per_level = 4 := by
  sorry

end number_of_levels_l173_173310


namespace largest_integer_n_neg_l173_173771

theorem largest_integer_n_neg (n : ℤ) : (n < 8 ∧ 3 < n) ∧ (n^2 - 11 * n + 24 < 0) → n ≤ 7 := by
  sorry

end largest_integer_n_neg_l173_173771


namespace library_visitor_ratio_l173_173788

theorem library_visitor_ratio (T : ℕ) (h1 : 50 + T + 20 * 4 = 250) : T / 50 = 2 :=
by
  sorry

end library_visitor_ratio_l173_173788


namespace symmetric_points_tangent_line_l173_173922

theorem symmetric_points_tangent_line (k : ℝ) (hk : 0 < k) :
  (∃ P Q : ℝ × ℝ, P.2 = Real.exp P.1 ∧ ∃ x₀ : ℝ, 
    Q.2 = k * Q.1 ∧ Q = (P.2, P.1) ∧ 
    Q.1 = x₀ ∧ k = 1 / x₀ ∧ x₀ = Real.exp 1) → k = 1 / Real.exp 1 := 
by 
  sorry

end symmetric_points_tangent_line_l173_173922


namespace problem_statement_l173_173851

-- Define the expression in Lean
def expr : ℤ := 120 * (120 - 5) - (120 * 120 - 10 + 2)

-- Theorem stating the value of the expression
theorem problem_statement : expr = -592 := by
  sorry

end problem_statement_l173_173851


namespace geometric_sequence_properties_l173_173242

theorem geometric_sequence_properties (a b c : ℝ) (r : ℝ) (h : r ≠ 0)
  (h1 : a = r * (-1))
  (h2 : b = r * a)
  (h3 : c = r * b)
  (h4 : -9 = r * c) :
  b = -3 ∧ a * c = 9 :=
by sorry

end geometric_sequence_properties_l173_173242


namespace plane_figures_l173_173902

def polyline_two_segments : Prop := -- Definition for a polyline composed of two line segments
  sorry

def polyline_three_segments : Prop := -- Definition for a polyline composed of three line segments
  sorry

def closed_three_segments : Prop := -- Definition for a closed figure composed of three line segments
  sorry

def quadrilateral_equal_opposite_sides : Prop := -- Definition for a quadrilateral with equal opposite sides
  sorry

def trapezoid : Prop := -- Definition for a trapezoid
  sorry

def is_plane_figure (fig : Prop) : Prop :=
  sorry  -- Axiom or definition that determines whether a figure is a plane figure.

-- Translating the proof problem
theorem plane_figures :
  is_plane_figure polyline_two_segments ∧
  ¬ is_plane_figure polyline_three_segments ∧
  is_plane_figure closed_three_segments ∧
  ¬ is_plane_figure quadrilateral_equal_opposite_sides ∧
  is_plane_figure trapezoid :=
by
  sorry

end plane_figures_l173_173902


namespace team_selection_ways_l173_173306

theorem team_selection_ways :
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose boys team_size_boys * choose girls team_size_girls = 103950 :=
by
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end team_selection_ways_l173_173306


namespace find_number_l173_173028

theorem find_number : ∃ (x : ℤ), 45 + 3 * x = 72 ∧ x = 9 := by
  sorry

end find_number_l173_173028


namespace combine_like_terms_1_simplify_expression_2_l173_173624

-- Problem 1
theorem combine_like_terms_1 (m n : ℝ) :
  2 * m^2 * n - 3 * m * n + 8 - 3 * m^2 * n + 5 * m * n - 3 = -m^2 * n + 2 * m * n + 5 :=
by 
  -- Proof goes here 
  sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by 
  -- Proof goes here 
  sorry

end combine_like_terms_1_simplify_expression_2_l173_173624


namespace number_of_ways_to_select_book_l173_173044

-- Definitions directly from the problem's conditions
def numMathBooks : Nat := 3
def numChineseBooks : Nat := 5
def numEnglishBooks : Nat := 8

-- The proof problem statement in Lean 4
theorem number_of_ways_to_select_book : numMathBooks + numChineseBooks + numEnglishBooks = 16 := 
by
  show 3 + 5 + 8 = 16
  sorry

end number_of_ways_to_select_book_l173_173044


namespace rectangle_circle_area_ratio_l173_173990

noncomputable def area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) : ℝ :=
  (2 * w^2) / (Real.pi * r^2)

theorem rectangle_circle_area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) :
  area_ratio w r h = 18 / (Real.pi * Real.pi) :=
by
  sorry

end rectangle_circle_area_ratio_l173_173990


namespace points_five_units_away_from_neg_one_l173_173838

theorem points_five_units_away_from_neg_one (x : ℝ) :
  |x + 1| = 5 ↔ x = 4 ∨ x = -6 :=
by
  sorry

end points_five_units_away_from_neg_one_l173_173838


namespace paco_countertop_total_weight_l173_173570

theorem paco_countertop_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 :=
sorry

end paco_countertop_total_weight_l173_173570


namespace find_roots_range_l173_173445

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x ^ 2 + b * x + c

theorem find_roots_range 
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hx : -1 < -1/2 ∧ -1/2 < 0 ∧ 0 < 1/2 ∧ 1/2 < 1 ∧ 1 < 3/2 ∧ 3/2 < 2 ∧ 2 < 5/2 ∧ 5/2 < 3)
  (hy : ∀ {x : ℝ}, x = -1 → quadratic_function a b c x = -2 ∧
                   x = -1/2 → quadratic_function a b c x = -1/4 ∧
                   x = 0 → quadratic_function a b c x = 1 ∧
                   x = 1/2 → quadratic_function a b c x = 7/4 ∧
                   x = 1 → quadratic_function a b c x = 2 ∧
                   x = 3/2 → quadratic_function a b c x = 7/4 ∧
                   x = 2 → quadratic_function a b c x = 1 ∧
                   x = 5/2 → quadratic_function a b c x = -1/4 ∧
                   x = 3 → quadratic_function a b c x = -2) :
  ∃ x1 x2 : ℝ, -1/2 < x1 ∧ x1 < 0 ∧ 2 < x2 ∧ x2 < 5/2 ∧ quadratic_function a b c x1 = 0 ∧ quadratic_function a b c x2 = 0 :=
by sorry

end find_roots_range_l173_173445


namespace remaining_money_correct_l173_173668

open Nat

def initial_money : ℕ := 158
def cost_shoes : ℕ := 45
def cost_bag : ℕ := cost_shoes - 17
def cost_lunch : ℕ := cost_bag / 4
def total_spent : ℕ := cost_shoes + cost_bag + cost_lunch
def remaining_money : ℕ := initial_money - total_spent

theorem remaining_money_correct : remaining_money = 78 := by
  -- Proof goes here
  sorry

end remaining_money_correct_l173_173668


namespace seq_inequality_l173_173643

variable (a : ℕ → ℝ)
variable (n m : ℕ)

-- Conditions
axiom pos_seq (k : ℕ) : a k ≥ 0
axiom add_condition (i j : ℕ) : a (i + j) ≤ a i + a j

-- Statement to prove
theorem seq_inequality (n m : ℕ) (h : m > 0) (h' : n ≥ m) : 
  a n ≤ m * a 1 + ((n : ℝ) / m - 1) * a m := sorry

end seq_inequality_l173_173643


namespace number_of_boys_exceeds_girls_by_l173_173148

theorem number_of_boys_exceeds_girls_by (girls boys: ℕ) (h1: girls = 34) (h2: boys = 841) : boys - girls = 807 := by
  sorry

end number_of_boys_exceeds_girls_by_l173_173148


namespace larger_angle_is_99_l173_173235

theorem larger_angle_is_99 (x : ℝ) (h1 : 2 * x + 18 = 180) : x + 18 = 99 :=
by
  sorry

end larger_angle_is_99_l173_173235


namespace gcd_factorial_l173_173810

theorem gcd_factorial (n m l : ℕ) (h1 : n = 7) (h2 : m = 10) (h3 : l = 4): 
  Nat.gcd (Nat.factorial n) (Nat.factorial m / Nat.factorial l) = 2520 :=
by
  sorry

end gcd_factorial_l173_173810


namespace max_value_x_plus_y_l173_173907

theorem max_value_x_plus_y : ∀ (x y : ℝ), 
  (5 * x + 3 * y ≤ 9) → 
  (3 * x + 5 * y ≤ 11) → 
  x + y ≤ 32 / 17 :=
by
  intros x y h1 h2
  -- proof steps go here
  sorry

end max_value_x_plus_y_l173_173907


namespace simplify_fraction_l173_173169

variable (x y : ℕ)

theorem simplify_fraction (hx : x = 3) (hy : y = 2) :
  (12 * x^2 * y^3) / (9 * x * y^2) = 8 :=
by
  sorry

end simplify_fraction_l173_173169


namespace additional_people_needed_to_mow_lawn_l173_173677

theorem additional_people_needed_to_mow_lawn :
  (∀ (k : ℕ), (∀ (n t : ℕ), n * t = k) → (4 * 6 = k) → (∃ (n : ℕ), n * 3 = k) → (8 - 4 = 4)) :=
by sorry

end additional_people_needed_to_mow_lawn_l173_173677


namespace expand_product_equivalence_l173_173590

variable (x : ℝ)  -- Assuming x is a real number

theorem expand_product_equivalence : (x + 5) * (x + 7) = x^2 + 12 * x + 35 :=
by
  sorry

end expand_product_equivalence_l173_173590


namespace find_p_plus_s_l173_173368

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem find_p_plus_s (p q r s : ℝ) (h : p * q * r * s ≠ 0) 
  (hg : ∀ x : ℝ, g p q r s (g p q r s x) = x) : p + s = 0 := 
by 
  sorry

end find_p_plus_s_l173_173368


namespace second_smallest_N_prevent_Bananastasia_win_l173_173963

-- Definition of the set S, as positive integers not divisible by any p^4.
def S : Set ℕ := {n | ∀ p : ℕ, Prime p → ¬ (p ^ 4 ∣ n)}

-- Definition of the game rules and the condition for Anastasia to prevent Bananastasia from winning.
-- N is a value such that for all a in S, it is not possible for Bananastasia to directly win.

theorem second_smallest_N_prevent_Bananastasia_win :
  ∃ N : ℕ, N = 625 ∧ (∀ a ∈ S, N - a ≠ 0 ∧ N - a ≠ 1) :=
by
  sorry

end second_smallest_N_prevent_Bananastasia_win_l173_173963


namespace problem_inequality_minimum_value_l173_173383

noncomputable def f (x y z : ℝ) : ℝ := 
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem problem_inequality (x y z : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h_sum : x + y + z = 1) :
  f x y z ≥ 0 :=
sorry

theorem minimum_value (x y z : ℝ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h_sum : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end problem_inequality_minimum_value_l173_173383


namespace exists_k_seq_zero_to_one_l173_173737

noncomputable def seq (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) := a

theorem exists_k_seq_zero_to_one (a : ℕ → ℝ) (h : ∀ n, a (n + 2) = |a (n + 1) - a n|) :
  ∃ k : ℕ, 0 ≤ a k ∧ a k < 1 :=
sorry

end exists_k_seq_zero_to_one_l173_173737


namespace transform_binomial_expansion_l173_173701

variable (a b : ℝ)

theorem transform_binomial_expansion (h : (a + b)^4 = a^4 + 4 * a^3 * b + 6 * a^2 * b^2 + 4 * a * b^3 + b^4) :
  (a - b)^4 = a^4 - 4 * a^3 * b + 6 * a^2 * b^2 - 4 * a * b^3 + b^4 :=
by
  sorry

end transform_binomial_expansion_l173_173701


namespace no_solution_system_l173_173905

theorem no_solution_system : ¬ ∃ (x y z : ℝ), 
  x^2 - 2*y + 2 = 0 ∧ 
  y^2 - 4*z + 3 = 0 ∧ 
  z^2 + 4*x + 4 = 0 := 
by
  sorry

end no_solution_system_l173_173905


namespace area_proof_l173_173010

def square_side_length : ℕ := 2
def triangle_leg_length : ℕ := 2

-- Definition of the initial square area
def square_area (side_length : ℕ) : ℕ := side_length * side_length

-- Definition of the area for one isosceles right triangle
def triangle_area (leg_length : ℕ) : ℕ := (leg_length * leg_length) / 2

-- Area of the initial square
def R_square_area : ℕ := square_area square_side_length

-- Area of the 12 isosceles right triangles
def total_triangle_area : ℕ := 12 * triangle_area triangle_leg_length

-- Total area of region R
def R_area : ℕ := R_square_area + total_triangle_area

-- Smallest convex polygon S is a larger square with side length 8
def S_area : ℕ := square_area (4 * square_side_length)

-- Area inside S but outside R
def area_inside_S_outside_R : ℕ := S_area - R_area

theorem area_proof : area_inside_S_outside_R = 36 :=
by
  sorry

end area_proof_l173_173010


namespace radius_relation_l173_173843

-- Define the conditions under which the spheres exist
variable {R r : ℝ}

-- The problem statement
theorem radius_relation (h : r = R * (2 - Real.sqrt 2)) : r = R * (2 - Real.sqrt 2) :=
sorry

end radius_relation_l173_173843


namespace multiple_of_P_l173_173876

theorem multiple_of_P (P Q R : ℝ) (T : ℝ) (x : ℝ) (total_profit Rs900 : ℝ)
  (h1 : P = 6 * Q)
  (h2 : P = 10 * R)
  (h3 : R = T / 5.1)
  (h4 : total_profit = Rs900 + (T - R)) :
  x = 10 :=
by
  sorry

end multiple_of_P_l173_173876


namespace number_of_ways_to_adjust_items_l173_173942

theorem number_of_ways_to_adjust_items :
  let items_on_upper_shelf := 4
  let items_on_lower_shelf := 8
  let move_items := 2
  let total_ways := Nat.choose items_on_lower_shelf move_items
  total_ways = 840 :=
by
  sorry

end number_of_ways_to_adjust_items_l173_173942


namespace range_of_x_l173_173045

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) (x : ℝ) : 
  (x ^ 2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by 
  sorry

end range_of_x_l173_173045


namespace find_c_value_l173_173670

theorem find_c_value 
  (b : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + b * x + 3 ≥ 0) 
  (h2 : ∀ m c : ℝ, (∀ x : ℝ, x^2 + b * x + 3 < c ↔ m - 8 < x ∧ x < m)) 
  : c = 16 :=
sorry

end find_c_value_l173_173670


namespace find_x_l173_173767

theorem find_x : ∃ x : ℤ, x + 3 * 10 = 33 → x = 3 := by
  sorry

end find_x_l173_173767


namespace find_a_b_sum_l173_173518

theorem find_a_b_sum
  (a b : ℝ)
  (h1 : 2 * a = -6)
  (h2 : a ^ 2 - b = 1) :
  a + b = 5 :=
by
  sorry

end find_a_b_sum_l173_173518


namespace annual_feeding_cost_is_correct_l173_173783

-- Definitions based on conditions
def number_of_geckos : Nat := 3
def number_of_iguanas : Nat := 2
def number_of_snakes : Nat := 4
def cost_per_gecko_per_month : Nat := 15
def cost_per_iguana_per_month : Nat := 5
def cost_per_snake_per_month : Nat := 10

-- Statement of the theorem
theorem annual_feeding_cost_is_correct : 
    (number_of_geckos * cost_per_gecko_per_month
    + number_of_iguanas * cost_per_iguana_per_month 
    + number_of_snakes * cost_per_snake_per_month) * 12 = 1140 := by
  sorry

end annual_feeding_cost_is_correct_l173_173783


namespace sarah_jamie_julien_ratio_l173_173374

theorem sarah_jamie_julien_ratio (S J : ℕ) (R : ℝ) :
  -- Conditions
  (J = S + 20) ∧
  (S = R * 50) ∧
  (7 * (J + S + 50) = 1890) ∧
  -- Prove the ratio
  R = 2 := by
  sorry

end sarah_jamie_julien_ratio_l173_173374


namespace fraction_multiplication_l173_173039

theorem fraction_multiplication :
  (3 / 4 : ℚ) * (1 / 2) * (2 / 5) * 5000 = 750 :=
by
  norm_num
  done

end fraction_multiplication_l173_173039


namespace find_b_l173_173441

theorem find_b (b p : ℚ) :
  (∀ x : ℚ, (2 * x^3 + b * x + 7 = (x^2 + p * x + 1) * (2 * x + 7))) →
  b = -45 / 2 :=
sorry

end find_b_l173_173441


namespace weeks_to_work_l173_173895

def iPhone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80

theorem weeks_to_work (iPhone_cost trade_in_value weekly_earnings : ℕ) :
  (iPhone_cost - trade_in_value) / weekly_earnings = 7 :=
by
  sorry

end weeks_to_work_l173_173895


namespace not_function_age_height_l173_173216

theorem not_function_age_height (f : ℕ → ℝ) :
  ¬(∀ (a b : ℕ), a = b → f a = f b) := sorry

end not_function_age_height_l173_173216


namespace line_tangent_72_l173_173424

theorem line_tangent_72 (k : ℝ) : 4 * x + 6 * y + k = 0 → y^2 = 32 * x → (48^2 - 4 * (8 * k) = 0 ↔ k = 72) :=
by
  sorry

end line_tangent_72_l173_173424


namespace div_by_self_condition_l173_173193

theorem div_by_self_condition (n : ℤ) (h : n^2 + 1 ∣ n) : n = 0 :=
by sorry

end div_by_self_condition_l173_173193


namespace cats_not_eating_either_l173_173270

theorem cats_not_eating_either (total_cats : ℕ) (cats_like_apples : ℕ) (cats_like_chicken : ℕ) (cats_like_both : ℕ) 
  (h1 : total_cats = 80)
  (h2 : cats_like_apples = 15)
  (h3 : cats_like_chicken = 60)
  (h4 : cats_like_both = 10) : 
  total_cats - (cats_like_apples + cats_like_chicken - cats_like_both) = 15 :=
by sorry

end cats_not_eating_either_l173_173270


namespace polar_equation_graph_l173_173487

theorem polar_equation_graph :
  ∀ (ρ θ : ℝ), (ρ > 0) → ((ρ - 1) * (θ - π) = 0) ↔ (ρ = 1 ∨ θ = π) :=
by
  sorry

end polar_equation_graph_l173_173487


namespace sum_of_squares_of_roots_l173_173497

theorem sum_of_squares_of_roots :
  (∃ r1 r2 : ℝ, (r1 + r2 = 10 ∧ r1 * r2 = 16) ∧ (r1^2 + r2^2 = 68)) :=
by
  sorry

end sum_of_squares_of_roots_l173_173497


namespace first_player_winning_strategy_l173_173816

theorem first_player_winning_strategy (num_chips : ℕ) : 
  (num_chips = 110) → 
  ∃ (moves : ℕ → ℕ × ℕ), (∀ n, 1 ≤ (moves n).1 ∧ (moves n).1 ≤ 9) ∧ 
  (∀ n, (moves n).1 ≠ (moves (n-1)).1) →
  (∃ move_sequence : ℕ → ℕ, ∀ k, move_sequence k ≤ num_chips ∧ 
  ((move_sequence (k+1) < move_sequence k) ∨ (move_sequence (k+1) = 0 ∧ move_sequence k = 1)) ∧ 
  (move_sequence k > 0) ∧ (move_sequence 0 = num_chips) →
  num_chips ≡ 14 [MOD 32]) :=
by 
  sorry

end first_player_winning_strategy_l173_173816


namespace guest_bedroom_ratio_l173_173655

theorem guest_bedroom_ratio 
  (lr_dr_kitchen : ℝ) (total_house : ℝ) (master_bedroom : ℝ) (guest_bedroom : ℝ) 
  (h1 : lr_dr_kitchen = 1000) 
  (h2 : total_house = 2300)
  (h3 : master_bedroom = 1040)
  (h4 : guest_bedroom = total_house - (lr_dr_kitchen + master_bedroom)) :
  guest_bedroom / master_bedroom = 1 / 4 := 
by
  sorry

end guest_bedroom_ratio_l173_173655


namespace vehicle_speed_l173_173679

theorem vehicle_speed (distance : ℝ) (time : ℝ) (h_dist : distance = 150) (h_time : time = 0.75) : distance / time = 200 :=
  by
    sorry

end vehicle_speed_l173_173679


namespace harvest_season_duration_l173_173150

theorem harvest_season_duration (weekly_rent : ℕ) (total_rent_paid : ℕ) : 
    (weekly_rent = 388) →
    (total_rent_paid = 527292) →
    (total_rent_paid / weekly_rent = 1360) :=
by
  intros h1 h2
  sorry

end harvest_season_duration_l173_173150


namespace scientific_notation_of_11090000_l173_173532

theorem scientific_notation_of_11090000 :
  ∃ (x : ℝ) (n : ℤ), 11090000 = x * 10^n ∧ x = 1.109 ∧ n = 7 :=
by
  -- skip the proof
  sorry

end scientific_notation_of_11090000_l173_173532


namespace min_b_over_a_l173_173562

theorem min_b_over_a (a b : ℝ) (h : ∀ x : ℝ, (Real.log a + b) * Real.exp x - a^2 * Real.exp x ≥ 0) : b / a ≥ 1 := by
  sorry

end min_b_over_a_l173_173562


namespace sandra_tickets_relation_l173_173501

def volleyball_game : Prop :=
  ∃ (tickets_total tickets_left tickets_jude tickets_andrea tickets_sandra : ℕ),
    tickets_total = 100 ∧
    tickets_left = 40 ∧
    tickets_jude = 16 ∧
    tickets_andrea = 2 * tickets_jude ∧
    tickets_total - tickets_left = tickets_jude + tickets_andrea + tickets_sandra ∧
    tickets_sandra = tickets_jude - 4

theorem sandra_tickets_relation : volleyball_game :=
  sorry

end sandra_tickets_relation_l173_173501


namespace tammy_investment_change_l173_173796

-- Defining initial investment, losses, and gains
def initial_investment : ℝ := 100
def first_year_loss : ℝ := 0.10
def second_year_gain : ℝ := 0.25

-- Defining the final amount after two years
def final_amount (initial_investment : ℝ) (first_year_loss : ℝ) (second_year_gain : ℝ) : ℝ :=
  let remaining_after_first_year := initial_investment * (1 - first_year_loss)
  remaining_after_first_year * (1 + second_year_gain)

-- Statement to prove
theorem tammy_investment_change :
  let percentage_change := ((final_amount initial_investment first_year_loss second_year_gain - initial_investment) / initial_investment) * 100
  percentage_change = 12.5 :=
by
  sorry

end tammy_investment_change_l173_173796


namespace calculateDifferentialSavings_l173_173812

/-- 
Assumptions for the tax brackets and deductions/credits.
-/
def taxBracketsCurrent (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 15 / 100
  else if income ≤ 45000 then
    15000 * 15 / 100 + (income - 15000) * 42 / 100
  else
    15000 * 15 / 100 + (45000 - 15000) * 42 / 100 + (income - 45000) * 50 / 100

def taxBracketsProposed (income : ℕ) : ℕ :=
  if income ≤ 15000 then
    income * 12 / 100
  else if income ≤ 45000 then
    15000 * 12 / 100 + (income - 15000) * 28 / 100
  else
    15000 * 12 / 100 + (45000 - 15000) * 28 / 100 + (income - 45000) * 50 / 100

def standardDeduction : ℕ := 3000
def childrenCredit (num_children : ℕ) : ℕ := num_children * 1000

def taxableIncome (income : ℕ) : ℕ :=
  income - standardDeduction

def totalTaxLiabilityCurrent (income num_children : ℕ) : ℕ :=
  (taxBracketsCurrent (taxableIncome income)) - (childrenCredit num_children)

def totalTaxLiabilityProposed (income num_children : ℕ) : ℕ :=
  (taxBracketsProposed (taxableIncome income)) - (childrenCredit num_children)

def differentialSavings (income num_children : ℕ) : ℕ :=
  totalTaxLiabilityCurrent income num_children - totalTaxLiabilityProposed income num_children

/-- 
Statement of the Lean 4 proof problem.
-/
theorem calculateDifferentialSavings : differentialSavings 34500 2 = 2760 :=
by
  sorry

end calculateDifferentialSavings_l173_173812


namespace addition_result_l173_173647

theorem addition_result (x : ℝ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end addition_result_l173_173647


namespace inequality_solution_set_l173_173496

theorem inequality_solution_set : 
  {x : ℝ | (x - 2) * (x + 1) ≤ 0} = {x : ℝ | -1 < x ∧ x ≤ 2} :=
by
  sorry

end inequality_solution_set_l173_173496


namespace find_value_of_expression_l173_173991

theorem find_value_of_expression (x y z : ℝ)
  (h1 : 12 * x - 9 * y^2 = 7)
  (h2 : 6 * y - 9 * z^2 = -2)
  (h3 : 12 * z - 9 * x^2 = 4) : 
  6 * x^2 + 9 * y^2 + 12 * z^2 = 9 :=
  sorry

end find_value_of_expression_l173_173991


namespace scientific_notation_example_l173_173882

def scientific_notation (n : ℝ) (a : ℝ) (b : ℤ) : Prop :=
  n = a * 10^b

theorem scientific_notation_example : 
  scientific_notation 0.00519 5.19 (-3) :=
by 
  sorry

end scientific_notation_example_l173_173882


namespace intersection_eq_l173_173322

def M : Set Real := {x | x^2 < 3 * x}
def N : Set Real := {x | Real.log x < 0}

theorem intersection_eq : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l173_173322


namespace at_least_two_consecutive_heads_probability_l173_173063

theorem at_least_two_consecutive_heads_probability :
  let outcomes := ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]
  let favorable_outcomes := ["HHH", "HHT", "THH"]
  let total_outcomes := outcomes.length
  let num_favorable := favorable_outcomes.length
  (num_favorable / total_outcomes : ℚ) = 1 / 2 :=
by sorry

end at_least_two_consecutive_heads_probability_l173_173063


namespace find_g_of_conditions_l173_173226

theorem find_g_of_conditions (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, x * g y = 2 * y * g x)
  (g_10 : g 10 = 15) : g 2 = 6 :=
sorry

end find_g_of_conditions_l173_173226


namespace measure_angle_E_l173_173463

-- Definitions based on conditions
variables {p q : Type} {A B E : ℝ}

noncomputable def measure_A (A B : ℝ) : ℝ := A
noncomputable def measure_B (A B : ℝ) : ℝ := 9 * A
noncomputable def parallel_lines (p q : Type) : Prop := true

-- Condition: measure of angle A is 1/9 of the measure of angle B
axiom angle_condition : A = (1 / 9) * B

-- Condition: p is parallel to q
axiom parallel_condition : parallel_lines p q

-- Prove that the measure of angle E is 18 degrees
theorem measure_angle_E (y : ℝ) (h1 : A = y) (h2 : B = 9 * y) : E = 18 :=
by
  sorry

end measure_angle_E_l173_173463


namespace maximal_product_sum_l173_173910

theorem maximal_product_sum : 
  ∃ (k m : ℕ), 
  k = 671 ∧ 
  m = 2 ∧ 
  2017 = 3 * k + 2 * m ∧ 
  ∀ a b : ℕ, a + b = 2017 ∧ (a < k ∨ b < m) → a * b ≤ 3 * k * 2 * m
:= 
sorry

end maximal_product_sum_l173_173910


namespace foot_slide_distance_l173_173695

def ladder_foot_slide (l h_initial h_new x_initial d y: ℝ) : Prop :=
  l = 30 ∧ x_initial = 6 ∧ d = 6 ∧
  h_initial = Real.sqrt (l^2 - x_initial^2) ∧
  h_new = h_initial - d ∧
  (l^2 = h_new^2 + (x_initial + y) ^ 2) → y = 18

theorem foot_slide_distance :
  ladder_foot_slide 30 (Real.sqrt (30^2 - 6^2)) ((Real.sqrt (30^2 - 6^2)) - 6) 6 6 18 :=
by
  sorry

end foot_slide_distance_l173_173695


namespace numbering_tube_contacts_l173_173995

theorem numbering_tube_contacts {n : ℕ} (hn : n = 7) :
  ∃ (f g : ℕ → ℕ), (∀ k : ℕ, f k = k % n) ∧ (∀ k : ℕ, g k = (n - k) % n) ∧ 
  (∀ m : ℕ, ∃ k : ℕ, f (k + m) % n = g k % n) :=
by
  sorry

end numbering_tube_contacts_l173_173995


namespace number_of_pickup_trucks_l173_173630

theorem number_of_pickup_trucks 
  (cars : ℕ) (bicycles : ℕ) (tricycles : ℕ) (total_tires : ℕ)
  (tires_per_car : ℕ) (tires_per_bicycle : ℕ) (tires_per_tricycle : ℕ) (tires_per_pickup : ℕ) :
  cars = 15 →
  bicycles = 3 →
  tricycles = 1 →
  total_tires = 101 →
  tires_per_car = 4 →
  tires_per_bicycle = 2 →
  tires_per_tricycle = 3 →
  tires_per_pickup = 4 →
  ((total_tires - (cars * tires_per_car + bicycles * tires_per_bicycle + tricycles * tires_per_tricycle)) / tires_per_pickup) = 8 :=
by
  sorry

end number_of_pickup_trucks_l173_173630


namespace seafood_noodles_l173_173093

theorem seafood_noodles (total_plates lobster_rolls spicy_hot_noodles : ℕ)
  (h_total : total_plates = 55)
  (h_lobster : lobster_rolls = 25)
  (h_spicy : spicy_hot_noodles = 14) :
  total_plates - (lobster_rolls + spicy_hot_noodles) = 16 :=
by
  sorry

end seafood_noodles_l173_173093


namespace units_digit_sum_of_factorials_50_l173_173612

def units_digit (n : Nat) : Nat :=
  n % 10

def sum_of_factorials (n : Nat) : Nat :=
  (List.range' 1 n).map Nat.factorial |>.sum

theorem units_digit_sum_of_factorials_50 :
  units_digit (sum_of_factorials 51) = 3 := 
sorry

end units_digit_sum_of_factorials_50_l173_173612


namespace cube_value_proportional_l173_173276

theorem cube_value_proportional (side_length1 side_length2 : ℝ) (volume1 volume2 : ℝ) (value1 value2 : ℝ) :
  side_length1 = 4 → volume1 = side_length1 ^ 3 → value1 = 500 →
  side_length2 = 6 → volume2 = side_length2 ^ 3 → value2 = value1 * (volume2 / volume1) →
  value2 = 1688 :=
by
  sorry

end cube_value_proportional_l173_173276


namespace determinant_sum_is_34_l173_173784

-- Define matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![5, -2],
  ![3, 4]
]

def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![1, 3],
  ![-1, 2]
]

-- Prove the determinant of the sum of A and B is 34
theorem determinant_sum_is_34 : Matrix.det (A + B) = 34 := by
  sorry

end determinant_sum_is_34_l173_173784


namespace point_above_line_l173_173160

-- Define the point P with coordinates (-2, t)
variable (t : ℝ)

-- Define the line equation
def line_eq (x y : ℝ) : ℝ := 2 * x - 3 * y + 6

-- Proving that t must be greater than 2/3 for the point P to be above the line
theorem point_above_line : (line_eq (-2) t < 0) -> t > 2 / 3 :=
by
  sorry

end point_above_line_l173_173160


namespace exponent_multiplication_l173_173203

-- Define the core condition: the base 625
def base := 625

-- Define the exponents
def exp1 := 0.08
def exp2 := 0.17
def combined_exp := exp1 + exp2

-- The mathematical goal to prove
theorem exponent_multiplication (b : ℝ) (e1 e2 : ℝ) (h1 : b = 625) (h2 : e1 = 0.08) (h3 : e2 = 0.17) :
  (b ^ e1 * b ^ e2) = 5 :=
by {
  -- Sorry is added to skip the actual proof steps.
  sorry
}

end exponent_multiplication_l173_173203


namespace smallest_consecutive_integer_sum_l173_173958

-- Definitions based on conditions
def consecutive_integer_sum (n : ℕ) := 20 * n + 190

-- Theorem statement
theorem smallest_consecutive_integer_sum : 
  ∃ (n k : ℕ), (consecutive_integer_sum n = k^3) ∧ (∀ m l : ℕ, (consecutive_integer_sum m = l^3) → k^3 ≤ l^3) :=
sorry

end smallest_consecutive_integer_sum_l173_173958


namespace jim_gas_gallons_l173_173654

theorem jim_gas_gallons (G : ℕ) (C_NC C_VA : ℕ → ℕ) 
  (h₁ : ∀ G, C_NC G = 2 * G)
  (h₂ : ∀ G, C_VA G = 3 * G)
  (h₃ : C_NC G + C_VA G = 50) :
  G = 10 := 
sorry

end jim_gas_gallons_l173_173654


namespace bowl_capacity_l173_173129

theorem bowl_capacity (C : ℝ) (h1 : (2/3) * C * 5 + (1/3) * C * 4 = 700) : C = 150 := 
by
  sorry

end bowl_capacity_l173_173129


namespace students_in_each_class_l173_173067

-- Define the conditions
def sheets_per_student : ℕ := 5
def total_sheets : ℕ := 400
def number_of_classes : ℕ := 4

-- Define the main proof theorem
theorem students_in_each_class : (total_sheets / sheets_per_student) / number_of_classes = 20 := by
  sorry -- Proof goes here

end students_in_each_class_l173_173067


namespace sum_of_roots_eq_6_l173_173125

theorem sum_of_roots_eq_6 : ∀ (x1 x2 : ℝ), (x1 * x1 = x1 ∧ x1 * x2 = x2) → (x1 + x2 = 6) :=
by
   intro x1 x2 hx
   have H : x1 + x2 = 6 := sorry
   exact H

end sum_of_roots_eq_6_l173_173125


namespace readers_all_three_l173_173408

def total_readers : ℕ := 500
def readers_science_fiction : ℕ := 320
def readers_literary_works : ℕ := 200
def readers_non_fiction : ℕ := 150
def readers_sf_and_lw : ℕ := 120
def readers_sf_and_nf : ℕ := 80
def readers_lw_and_nf : ℕ := 60

theorem readers_all_three :
  total_readers = readers_science_fiction + readers_literary_works + readers_non_fiction - (readers_sf_and_lw + readers_sf_and_nf + readers_lw_and_nf) + 90 :=
by
  sorry

end readers_all_three_l173_173408


namespace quadratic_inequality_a_value_l173_173126

theorem quadratic_inequality_a_value (a t : ℝ)
  (h_a1 : ∀ x : ℝ, t * x ^ 2 - 6 * x + t ^ 2 = 0 → (x = a ∨ x = 1))
  (h_t : t < 0) :
  a = -3 :=
by
  sorry

end quadratic_inequality_a_value_l173_173126


namespace charlie_cortland_apples_l173_173486

/-- Given that Charlie picked 0.17 bags of Golden Delicious apples, 0.17 bags of Macintosh apples, 
   and a total of 0.67 bags of fruit, prove that the number of bags of Cortland apples picked by Charlie is 0.33. -/
theorem charlie_cortland_apples :
  let golden_delicious := 0.17
  let macintosh := 0.17
  let total_fruit := 0.67
  total_fruit - (golden_delicious + macintosh) = 0.33 :=
by
  sorry

end charlie_cortland_apples_l173_173486


namespace problem1_l173_173556

theorem problem1 (x y : ℝ) (h : |x + 1| + (2 * x - y)^2 = 0) : x^2 - y = 3 :=
sorry

end problem1_l173_173556


namespace ones_digit_of_prime_in_arithmetic_sequence_l173_173873

theorem ones_digit_of_prime_in_arithmetic_sequence (p q r : ℕ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) 
  (arithmetic_sequence : q = p + 4 ∧ r = q + 4)
  (h : p > 5) : 
    (p % 10 = 3 ∨ p % 10 = 9) :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_l173_173873


namespace arithmetic_sequence_general_geometric_sequence_sum_l173_173117

theorem arithmetic_sequence_general (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d) 
  (h_a3 : a 3 = -6) 
  (h_a6 : a 6 = 0) :
  ∀ n, a n = 2 * n - 12 := 
sorry

theorem geometric_sequence_sum (a b : ℕ → ℤ) 
  (r : ℤ) 
  (S : ℕ → ℤ)
  (h_geom : ∀ n : ℕ, b (n + 1) = b n * r) 
  (h_b1 : b 1 = -8) 
  (h_b2 : b 2 = a 0 + a 1 + a 2) 
  (h_a1 : a 0 = -10) 
  (h_a2 : a 1 = -8) 
  (h_a3 : a 2 = -6) :
  ∀ n, S n = 4 * (1 - 3 ^ n) := 
sorry

end arithmetic_sequence_general_geometric_sequence_sum_l173_173117


namespace remainder_17_pow_1499_mod_23_l173_173580

theorem remainder_17_pow_1499_mod_23 : (17 ^ 1499) % 23 = 11 :=
by
  sorry

end remainder_17_pow_1499_mod_23_l173_173580


namespace total_amount_returned_l173_173419

noncomputable def continuous_compounding_interest : ℝ :=
  let P : ℝ := 325 / (Real.exp 0.12 - 1)
  let A1 : ℝ := P * Real.exp 0.04
  let A2 : ℝ := A1 * Real.exp 0.05
  let A3 : ℝ := A2 * Real.exp 0.03
  let total_interest : ℝ := 325
  let total_amount : ℝ := P + total_interest
  total_amount

theorem total_amount_returned :
  continuous_compounding_interest = 2874.02 :=
by
  sorry

end total_amount_returned_l173_173419


namespace positive_diff_of_supplementary_angles_l173_173380

theorem positive_diff_of_supplementary_angles (x : ℝ) (h : 5 * x + 3 * x = 180) : 
  abs ((5 * x - 3 * x)) = 45 := by
  sorry

end positive_diff_of_supplementary_angles_l173_173380


namespace add_pure_alcohol_to_achieve_percentage_l173_173011

-- Define the initial conditions
def initial_solution_volume : ℝ := 6
def initial_alcohol_percentage : ℝ := 0.30
def initial_pure_alcohol : ℝ := initial_solution_volume * initial_alcohol_percentage

-- Define the final conditions
def final_alcohol_percentage : ℝ := 0.50

-- Define the unknown to prove
def amount_of_alcohol_to_add : ℝ := 2.4

-- The target statement to prove
theorem add_pure_alcohol_to_achieve_percentage :
  (initial_pure_alcohol + amount_of_alcohol_to_add) / (initial_solution_volume + amount_of_alcohol_to_add) = final_alcohol_percentage :=
by
  sorry

end add_pure_alcohol_to_achieve_percentage_l173_173011


namespace total_tickets_sold_correct_l173_173333

theorem total_tickets_sold_correct :
  ∀ (A : ℕ), (21 * A + 15 * 327 = 8748) → (A + 327 = 509) :=
by
  intros A h
  sorry

end total_tickets_sold_correct_l173_173333


namespace marlon_keeps_4_lollipops_l173_173272

def initial_lollipops : ℕ := 42
def fraction_given_to_emily : ℚ := 2 / 3
def lollipops_given_to_lou : ℕ := 10

theorem marlon_keeps_4_lollipops :
  let lollipops_given_to_emily := fraction_given_to_emily * initial_lollipops
  let lollipops_after_emily := initial_lollipops - lollipops_given_to_emily
  let marlon_keeps := lollipops_after_emily - lollipops_given_to_lou
  marlon_keeps = 4 :=
by
  sorry

end marlon_keeps_4_lollipops_l173_173272


namespace same_curve_option_B_l173_173750

theorem same_curve_option_B : 
  (∀ x y : ℝ, |y| = |x| ↔ y = x ∨ y = -x) ∧ (∀ x y : ℝ, y^2 = x^2 ↔ y = x ∨ y = -x) :=
by
  sorry

end same_curve_option_B_l173_173750


namespace milk_problem_l173_173540

theorem milk_problem (x : ℕ) (hx : 0 < x)
    (total_cost_wednesday : 10 = x * (10 / x))
    (price_reduced : ∀ x, 0.5 = (10 / x - (10 / x) + 0.5))
    (extra_bags : 2 = (x + 2) - x)
    (extra_cost : 2 + 10 = x * (10 / x) + 2) :
    x^2 + 6 * x - 40 = 0 := by
  sorry

end milk_problem_l173_173540


namespace acute_triangle_inequality_l173_173059

variable (f : ℝ → ℝ)
variable {A B : ℝ}
variable (h₁ : ∀ x : ℝ, x * (f'' x) - 2 * (f x) > 0)
variable (h₂ : A + B < Real.pi / 2 ∧ 0 < A ∧ 0 < B)

theorem acute_triangle_inequality :
  f (Real.cos A) * (Real.sin B) ^ 2 < f (Real.sin B) * (Real.cos A) ^ 2 := 
  sorry

end acute_triangle_inequality_l173_173059


namespace area_of_triangle_l173_173761

theorem area_of_triangle (h : ℝ) (a : ℝ) (b : ℝ) (hypotenuse : h = 13) (side_a : a = 5) (right_triangle : a^2 + b^2 = h^2) : 
  ∃ (area : ℝ), area = 30 := 
by
  sorry

end area_of_triangle_l173_173761


namespace avg_growth_rate_proof_l173_173926

noncomputable def avg_growth_rate_correct_eqn (x : ℝ) : Prop :=
  40 * (1 + x)^2 = 48.4

theorem avg_growth_rate_proof (x : ℝ) 
  (h1 : 40 = avg_working_hours_first_week)
  (h2 : 48.4 = avg_working_hours_third_week) :
  avg_growth_rate_correct_eqn x :=
by 
  sorry

/- Defining the known conditions -/
def avg_working_hours_first_week : ℝ := 40
def avg_working_hours_third_week : ℝ := 48.4

end avg_growth_rate_proof_l173_173926


namespace find_vector_n_l173_173159

variable (a b : ℝ)

def is_orthogonal (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def is_same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_vector_n (m n : ℝ × ℝ) (h1 : is_orthogonal m n) (h2 : is_same_magnitude m n) :
  n = (b, -a) :=
  sorry

end find_vector_n_l173_173159
