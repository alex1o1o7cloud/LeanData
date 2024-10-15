import Mathlib

namespace NUMINAMATH_GPT_activity_popularity_order_l1464_146404

-- Definitions for the fractions representing activity popularity
def dodgeball_popularity : Rat := 9 / 24
def magic_show_popularity : Rat := 4 / 12
def singing_contest_popularity : Rat := 1 / 3

-- Theorem stating the order of activities based on popularity
theorem activity_popularity_order :
  dodgeball_popularity > magic_show_popularity ∧ magic_show_popularity = singing_contest_popularity :=
by 
  sorry

end NUMINAMATH_GPT_activity_popularity_order_l1464_146404


namespace NUMINAMATH_GPT_find_original_cost_price_l1464_146448

variable (C S : ℝ)

-- Conditions
def original_profit (C S : ℝ) : Prop := S = 1.25 * C
def new_profit_condition (C S : ℝ) : Prop := 1.04 * C = S - 12.60

-- Main Theorem
theorem find_original_cost_price (h1 : original_profit C S) (h2 : new_profit_condition C S) : C = 60 := 
sorry

end NUMINAMATH_GPT_find_original_cost_price_l1464_146448


namespace NUMINAMATH_GPT_share_of_A_eq_70_l1464_146481

theorem share_of_A_eq_70 (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 595) : A = 70 :=
sorry

end NUMINAMATH_GPT_share_of_A_eq_70_l1464_146481


namespace NUMINAMATH_GPT_ratio_alcohol_to_water_l1464_146408

theorem ratio_alcohol_to_water (vol_alcohol vol_water : ℚ) 
  (h_alcohol : vol_alcohol = 2/7) 
  (h_water : vol_water = 3/7) : 
  vol_alcohol / vol_water = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_alcohol_to_water_l1464_146408


namespace NUMINAMATH_GPT_system_solution_and_range_l1464_146413

theorem system_solution_and_range (a x y : ℝ) (h1 : 2 * x + y = 5 * a) (h2 : x - 3 * y = -a + 7) :
  (x = 2 * a + 1 ∧ y = a - 2) ∧ (-1/2 ≤ a ∧ a < 2 → 2 * a + 1 ≥ 0 ∧ a - 2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_system_solution_and_range_l1464_146413


namespace NUMINAMATH_GPT_g_expression_f_expression_l1464_146420

-- Given functions f and g that satisfy the conditions
variable {f g : ℝ → ℝ}

-- Conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom sum_eq : ∀ x, f x + g x = 2^x + 2 * x

-- Theorem statements to prove
theorem g_expression : g = fun x => 2^x := by sorry
theorem f_expression : f = fun x => 2 * x := by sorry

end NUMINAMATH_GPT_g_expression_f_expression_l1464_146420


namespace NUMINAMATH_GPT_star_evaluation_l1464_146402

def star (X Y : ℚ) := (X + Y) / 4

theorem star_evaluation : star (star 3 8) 6 = 35 / 16 := by
  sorry

end NUMINAMATH_GPT_star_evaluation_l1464_146402


namespace NUMINAMATH_GPT_initial_mixture_amount_l1464_146401

theorem initial_mixture_amount (x : ℝ) (h1 : 20 / 100 * x / (x + 3) = 6 / 35) : x = 18 :=
sorry

end NUMINAMATH_GPT_initial_mixture_amount_l1464_146401


namespace NUMINAMATH_GPT_find_number_l1464_146417

variable (x : ℝ)

theorem find_number (h : 0.46 * x = 165.6) : x = 360 :=
sorry

end NUMINAMATH_GPT_find_number_l1464_146417


namespace NUMINAMATH_GPT_sand_cake_probability_is_12_percent_l1464_146489

def total_days : ℕ := 5
def ham_days : ℕ := 3
def cake_days : ℕ := 1

-- Probability of packing a ham sandwich on any given day
def prob_ham_sandwich : ℚ := ham_days / total_days

-- Probability of packing a piece of cake on any given day
def prob_cake : ℚ := cake_days / total_days

-- Calculate the combined probability that Karen packs a ham sandwich and cake on the same day
def combined_probability : ℚ := prob_ham_sandwich * prob_cake

-- Convert the combined probability to a percentage
def combined_probability_as_percentage : ℚ := combined_probability * 100

-- The proof problem to show that the probability that Karen packs a ham sandwich and cake on the same day is 12%
theorem sand_cake_probability_is_12_percent : combined_probability_as_percentage = 12 := 
  by sorry

end NUMINAMATH_GPT_sand_cake_probability_is_12_percent_l1464_146489


namespace NUMINAMATH_GPT_solve_fraction_eq_zero_l1464_146477

theorem solve_fraction_eq_zero (a : ℝ) (h : a ≠ -1) : (a^2 - 1) / (a + 1) = 0 ↔ a = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_fraction_eq_zero_l1464_146477


namespace NUMINAMATH_GPT_paul_spent_374_43_l1464_146488

noncomputable def paul_total_cost_after_discounts : ℝ :=
  let dress_shirts := 4 * 15.00
  let discount_dress_shirts := dress_shirts * 0.20
  let cost_dress_shirts := dress_shirts - discount_dress_shirts
  
  let pants := 2 * 40.00
  let discount_pants := pants * 0.30
  let cost_pants := pants - discount_pants
  
  let suit := 150.00
  
  let sweaters := 2 * 30.00
  
  let ties := 3 * 20.00
  let discount_tie := 20.00 * 0.50
  let cost_ties := 20.00 + (20.00 - discount_tie) + 20.00

  let shoes := 80.00
  let discount_shoes := shoes * 0.25
  let cost_shoes := shoes - discount_shoes

  let total_after_discounts := cost_dress_shirts + cost_pants + suit + sweaters + cost_ties + cost_shoes
  
  let total_after_coupon := total_after_discounts * 0.90
  
  let total_after_rewards := total_after_coupon - (500 * 0.05)
  
  let total_after_tax := total_after_rewards * 1.05
  
  total_after_tax

theorem paul_spent_374_43 :
  paul_total_cost_after_discounts = 374.43 :=
by
  sorry

end NUMINAMATH_GPT_paul_spent_374_43_l1464_146488


namespace NUMINAMATH_GPT_remaining_surface_area_l1464_146461

def edge_length_original : ℝ := 9
def edge_length_small : ℝ := 2
def surface_area (a : ℝ) : ℝ := 6 * a^2

theorem remaining_surface_area :
  surface_area edge_length_original - 3 * (edge_length_small ^ 2) + 3 * (edge_length_small ^ 2) = 486 :=
by
  sorry

end NUMINAMATH_GPT_remaining_surface_area_l1464_146461


namespace NUMINAMATH_GPT_expression_multiple_of_five_l1464_146485

theorem expression_multiple_of_five (n : ℕ) (h : n ≥ 10) : 
  (∃ k : ℕ, (n + 2) * (n + 1) = 5 * k) :=
sorry

end NUMINAMATH_GPT_expression_multiple_of_five_l1464_146485


namespace NUMINAMATH_GPT_quadratic_b_value_l1464_146498
open Real

theorem quadratic_b_value (b n : ℝ) 
  (h1: b < 0) 
  (h2: ∀ x, x^2 + b * x + (1 / 4) = (x + n)^2 + (1 / 16)) :
  b = - (sqrt 3 / 2) :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_quadratic_b_value_l1464_146498


namespace NUMINAMATH_GPT_faye_homework_problems_l1464_146467

----- Definitions based on the conditions given -----

def total_math_problems : ℕ := 46
def total_science_problems : ℕ := 9
def problems_finished_at_school : ℕ := 40

----- Theorem statement -----

theorem faye_homework_problems : total_math_problems + total_science_problems - problems_finished_at_school = 15 := by
  -- Sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_faye_homework_problems_l1464_146467


namespace NUMINAMATH_GPT_angle_diff_complement_supplement_l1464_146437

theorem angle_diff_complement_supplement (α : ℝ) : (180 - α) - (90 - α) = 90 := by
  sorry

end NUMINAMATH_GPT_angle_diff_complement_supplement_l1464_146437


namespace NUMINAMATH_GPT_find_b_l1464_146407

theorem find_b (b : ℤ) (h : ∃ x : ℝ, x^2 + b * x - 35 = 0 ∧ x = 5) : b = 2 :=
sorry

end NUMINAMATH_GPT_find_b_l1464_146407


namespace NUMINAMATH_GPT_smallest_perfect_square_greater_l1464_146441

theorem smallest_perfect_square_greater (a : ℕ) (h : ∃ n : ℕ, a = n^2) : 
  ∃ m : ℕ, m^2 > a ∧ ∀ k : ℕ, k^2 > a → m^2 ≤ k^2 :=
  sorry

end NUMINAMATH_GPT_smallest_perfect_square_greater_l1464_146441


namespace NUMINAMATH_GPT_evaluate_expression_l1464_146482

variable (b : ℝ)

theorem evaluate_expression : ( ( (b^(16/8))^(1/4) )^3 * ( (b^(16/4))^(1/8) )^3 ) = b^3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1464_146482


namespace NUMINAMATH_GPT_work_duration_B_l1464_146439

theorem work_duration_B (x : ℕ) (h : x = 10) : 
  (x * (1 / 15 : ℚ)) + (2 * (1 / 6 : ℚ)) = 1 := 
by 
  rw [h]
  sorry

end NUMINAMATH_GPT_work_duration_B_l1464_146439


namespace NUMINAMATH_GPT_smallest_b_l1464_146499

theorem smallest_b (b : ℕ) : 
  (b % 3 = 2) ∧ (b % 4 = 3) ∧ (b % 5 = 4) ∧ (b % 7 = 6) ↔ b = 419 :=
by sorry

end NUMINAMATH_GPT_smallest_b_l1464_146499


namespace NUMINAMATH_GPT_vector_field_lines_l1464_146428

noncomputable def vector_lines : Prop :=
  ∃ (C_1 C_2 : ℝ), ∀ (x y z : ℝ), (9 * z^2 + 4 * y^2 = C_1) ∧ (x = C_2)

-- We state the proof goal as follows:
theorem vector_field_lines :
  ∀ (a : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ), 
    (∀ (x y z : ℝ), a (x, y, z) = (0, 9 * z, -4 * y)) →
    vector_lines :=
by
  intro a ha
  sorry

end NUMINAMATH_GPT_vector_field_lines_l1464_146428


namespace NUMINAMATH_GPT_part1_part2_l1464_146478

-- Part (1) statement
theorem part1 {x : ℝ} : (|x - 1| + |x + 2| >= 5) ↔ (x <= -3 ∨ x >= 2) := 
sorry

-- Part (2) statement
theorem part2 (a : ℝ) : (∀ x : ℝ, (|a * x - 2| < 3 ↔ -5/3 < x ∧ x < 1/3)) → a = -3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1464_146478


namespace NUMINAMATH_GPT_train_scheduled_speed_l1464_146490

theorem train_scheduled_speed (a v : ℝ) (hv : 0 < v)
  (h1 : a / v - a / (v + 5) = 1 / 3)
  (h2 : a / (v - 5) - a / v = 5 / 12) : v = 45 :=
by
  sorry

end NUMINAMATH_GPT_train_scheduled_speed_l1464_146490


namespace NUMINAMATH_GPT_peanut_butter_last_days_l1464_146454

-- Definitions for the problem conditions
def daily_consumption : ℕ := 2
def servings_per_jar : ℕ := 15
def num_jars : ℕ := 4

-- The statement to prove
theorem peanut_butter_last_days : 
  (num_jars * servings_per_jar) / daily_consumption = 30 :=
by
  sorry

end NUMINAMATH_GPT_peanut_butter_last_days_l1464_146454


namespace NUMINAMATH_GPT_point_P_in_first_quadrant_l1464_146474

def point_P := (3, 2)
def first_quadrant (p : ℕ × ℕ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem point_P_in_first_quadrant : first_quadrant point_P :=
by
  sorry

end NUMINAMATH_GPT_point_P_in_first_quadrant_l1464_146474


namespace NUMINAMATH_GPT_customer_bought_29_eggs_l1464_146483

-- Defining the conditions
def baskets : List ℕ := [4, 6, 12, 13, 22, 29]
def total_eggs : ℕ := 86
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Stating the problem
theorem customer_bought_29_eggs :
  ∃ eggs_in_basket,
    eggs_in_basket ∈ baskets ∧
    is_multiple_of_three (total_eggs - eggs_in_basket) ∧
    eggs_in_basket = 29 :=
by sorry

end NUMINAMATH_GPT_customer_bought_29_eggs_l1464_146483


namespace NUMINAMATH_GPT_min_value_expr_l1464_146451

theorem min_value_expr (a b : ℝ) (h1 : 2 * a + b = a * b) (h2 : a > 0) (h3 : b > 0) : 
  ∃ a b, (a > 0 ∧ b > 0 ∧ 2 * a + b = a * b) ∧ (∀ x y, (x > 0 ∧ y > 0 ∧ 2 * x + y = x * y) → (1 / (x - 1) + 2 / (y - 2)) ≥ 2) ∧ ((1 / (a - 1) + 2 / (b - 2)) = 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l1464_146451


namespace NUMINAMATH_GPT_coplanar_lines_condition_l1464_146436

theorem coplanar_lines_condition (h : ℝ) : 
  (∃ c : ℝ, 
    (2 : ℝ) = 3 * c ∧ 
    (-1 : ℝ) = c ∧ 
    (h : ℝ) = -2 * c) ↔ 
  (h = 2) :=
by
  sorry

end NUMINAMATH_GPT_coplanar_lines_condition_l1464_146436


namespace NUMINAMATH_GPT_face_opposite_one_is_three_l1464_146447

def faces : List ℕ := [1, 2, 3, 4, 5, 6]

theorem face_opposite_one_is_three (x : ℕ) (h1 : x ∈ faces) (h2 : x ≠ 1) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_face_opposite_one_is_three_l1464_146447


namespace NUMINAMATH_GPT_find_k_l1464_146426

-- Define the function f as described in the problem statement
def f (n : ℕ) : ℕ := 
  if n % 2 = 1 then 
    n + 3 
  else 
    n / 2

theorem find_k (k : ℕ) (h_odd : k % 2 = 1) : f (f (f k)) = k → k = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_k_l1464_146426


namespace NUMINAMATH_GPT_sqrt_sum_odds_l1464_146464

theorem sqrt_sum_odds : 
  (Real.sqrt 1 + Real.sqrt (1+3) + Real.sqrt (1+3+5) + Real.sqrt (1+3+5+7) + Real.sqrt (1+3+5+7+9) + Real.sqrt (1+3+5+7+9+11)) = 21 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_sum_odds_l1464_146464


namespace NUMINAMATH_GPT_polynomial_solution_l1464_146446

theorem polynomial_solution (P : ℝ → ℝ) (hP : ∀ x : ℝ, (x + 1) * P (x - 1) + (x - 1) * P (x + 1) = 2 * x * P x) :
  ∃ (a d : ℝ), ∀ x : ℝ, P x = a * x^3 - a * x + d := 
sorry

end NUMINAMATH_GPT_polynomial_solution_l1464_146446


namespace NUMINAMATH_GPT_sum_of_k_values_l1464_146412

theorem sum_of_k_values (k : ℤ) :
  (∃ (r s : ℤ), (r ≠ s) ∧ (3 * r * s = 9) ∧ (r + s = k / 3)) → k = 0 :=
by sorry

end NUMINAMATH_GPT_sum_of_k_values_l1464_146412


namespace NUMINAMATH_GPT_gcd_15_70_l1464_146473

theorem gcd_15_70 : Int.gcd 15 70 = 5 := by
  sorry

end NUMINAMATH_GPT_gcd_15_70_l1464_146473


namespace NUMINAMATH_GPT_solution_set_inequality_l1464_146450

theorem solution_set_inequality (x : ℝ) : 
  (x + 5) * (3 - 2 * x) ≤ 6 ↔ (x ≤ -9/2 ∨ x ≥ 1) :=
by
  sorry  -- proof skipped as instructed

end NUMINAMATH_GPT_solution_set_inequality_l1464_146450


namespace NUMINAMATH_GPT_eric_has_9306_erasers_l1464_146443

-- Define the conditions as constants
def number_of_friends := 99
def erasers_per_friend := 94

-- Define the total number of erasers based on the conditions
def total_erasers := number_of_friends * erasers_per_friend

-- Theorem stating the total number of erasers Eric has
theorem eric_has_9306_erasers : total_erasers = 9306 := by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_eric_has_9306_erasers_l1464_146443


namespace NUMINAMATH_GPT_perimeter_of_unshaded_rectangle_l1464_146484

theorem perimeter_of_unshaded_rectangle (length width height base area shaded_area perimeter : ℝ)
  (h1 : length = 12)
  (h2 : width = 9)
  (h3 : height = 3)
  (h4 : base = (2 * shaded_area) / height)
  (h5 : shaded_area = 18)
  (h6 : perimeter = 2 * ((length - base) + width))
  : perimeter = 24 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_unshaded_rectangle_l1464_146484


namespace NUMINAMATH_GPT_union_set_l1464_146438

def M : Set ℝ := {x | -2 < x ∧ x < 1}
def P : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem union_set : M ∪ P = {x : ℝ | -2 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_union_set_l1464_146438


namespace NUMINAMATH_GPT_union_complement_eq_universal_l1464_146435

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 5}

-- The proof problem
theorem union_complement_eq_universal :
  U = A ∪ (U \ B) :=
by
  sorry

end NUMINAMATH_GPT_union_complement_eq_universal_l1464_146435


namespace NUMINAMATH_GPT_concentric_circles_radius_difference_l1464_146405

theorem concentric_circles_radius_difference (r R : ℝ)
  (h : R^2 = 4 * r^2) :
  R - r = r :=
by
  sorry

end NUMINAMATH_GPT_concentric_circles_radius_difference_l1464_146405


namespace NUMINAMATH_GPT_holidays_per_month_l1464_146469

theorem holidays_per_month (total_holidays : ℕ) (months_in_year : ℕ) (holidays_in_month : ℕ) 
    (h1 : total_holidays = 48) (h2 : months_in_year = 12) : holidays_in_month = 4 := 
by
  sorry

end NUMINAMATH_GPT_holidays_per_month_l1464_146469


namespace NUMINAMATH_GPT_how_much_does_c_have_l1464_146466

theorem how_much_does_c_have (A B C : ℝ) (h1 : A + B + C = 400) (h2 : A + C = 300) (h3 : B + C = 150) : C = 50 :=
by
  sorry

end NUMINAMATH_GPT_how_much_does_c_have_l1464_146466


namespace NUMINAMATH_GPT_find_pairs_l1464_146452

theorem find_pairs :
  ∀ (x y : ℕ), 0 < x → 0 < y → 7 ^ x - 3 * 2 ^ y = 1 → (x, y) = (1, 1) ∨ (x, y) = (2, 4) :=
by
  intros x y hx hy h
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_find_pairs_l1464_146452


namespace NUMINAMATH_GPT_abigail_total_savings_l1464_146460

def monthly_savings : ℕ := 4000
def months_in_year : ℕ := 12

theorem abigail_total_savings : monthly_savings * months_in_year = 48000 := by
  sorry

end NUMINAMATH_GPT_abigail_total_savings_l1464_146460


namespace NUMINAMATH_GPT_range_of_m_l1464_146463

noncomputable def quadratic_expr_never_equal (m : ℝ) : Prop :=
  ∀ (x : ℝ), 2 * x^2 + 4 * x + m ≠ 3 * x^2 - 2 * x + 6

theorem range_of_m (m : ℝ) : quadratic_expr_never_equal m ↔ m < -3 := 
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1464_146463


namespace NUMINAMATH_GPT_x_intercept_is_one_l1464_146495

theorem x_intercept_is_one (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -1)) (h2 : (x2, y2) = (-2, 3)) :
    ∃ x : ℝ, (0 = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1) ∧ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_is_one_l1464_146495


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1464_146418

theorem sufficient_but_not_necessary_condition (x : ℝ) : 
  (x > 2 → (x-1)^2 > 1) ∧ (∃ (y : ℝ), y ≤ 2 ∧ (y-1)^2 > 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1464_146418


namespace NUMINAMATH_GPT_photos_last_weekend_45_l1464_146471

theorem photos_last_weekend_45 (photos_animals photos_flowers photos_scenery total_photos_this_weekend photos_last_weekend : ℕ)
  (h1 : photos_animals = 10)
  (h2 : photos_flowers = 3 * photos_animals)
  (h3 : photos_scenery = photos_flowers - 10)
  (h4 : total_photos_this_weekend = photos_animals + photos_flowers + photos_scenery)
  (h5 : photos_last_weekend = total_photos_this_weekend - 15) :
  photos_last_weekend = 45 :=
sorry

end NUMINAMATH_GPT_photos_last_weekend_45_l1464_146471


namespace NUMINAMATH_GPT_xyz_eq_7cubed_l1464_146410

theorem xyz_eq_7cubed (x y z : ℤ) (h1 : x^2 * y * z^3 = 7^4) (h2 : x * y^2 = 7^5) : x * y * z = 7^3 := 
by 
  sorry

end NUMINAMATH_GPT_xyz_eq_7cubed_l1464_146410


namespace NUMINAMATH_GPT_elise_spent_on_puzzle_l1464_146431

-- Definitions based on the problem conditions:
def initial_money : ℕ := 8
def saved_money : ℕ := 13
def spent_on_comic : ℕ := 2
def remaining_money : ℕ := 1

-- Prove that the amount spent on the puzzle is $18.
theorem elise_spent_on_puzzle : initial_money + saved_money - spent_on_comic - remaining_money = 18 := by
  sorry

end NUMINAMATH_GPT_elise_spent_on_puzzle_l1464_146431


namespace NUMINAMATH_GPT_car_speed_l1464_146414

theorem car_speed (v : ℝ) (hv : 2 + (1 / v) * 3600 = (1 / 90) * 3600) :
  v = 600 / 7 :=
sorry

end NUMINAMATH_GPT_car_speed_l1464_146414


namespace NUMINAMATH_GPT_circumference_given_area_l1464_146453

noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2

noncomputable def circumference_of_circle (r : ℝ) : ℝ := 2 * Real.pi * r

theorem circumference_given_area :
  (∃ r : ℝ, area_of_circle r = 616) →
  circumference_of_circle 14 = 2 * Real.pi * 14 :=
by
  sorry

end NUMINAMATH_GPT_circumference_given_area_l1464_146453


namespace NUMINAMATH_GPT_merchant_should_choose_option2_l1464_146462

-- Definitions for the initial price and discounts
def P : ℝ := 20000
def d1_1 : ℝ := 0.25
def d1_2 : ℝ := 0.15
def d1_3 : ℝ := 0.05
def d2_1 : ℝ := 0.35
def d2_2 : ℝ := 0.10
def d2_3 : ℝ := 0.05

-- Define the final prices after applying discount options
def finalPrice1 (P : ℝ) (d1_1 d1_2 d1_3 : ℝ) : ℝ :=
  P * (1 - d1_1) * (1 - d1_2) * (1 - d1_3)

def finalPrice2 (P : ℝ) (d2_1 d2_2 d2_3 : ℝ) : ℝ :=
  P * (1 - d2_1) * (1 - d2_2) * (1 - d2_3)

-- Theorem to state the merchant should choose Option 2
theorem merchant_should_choose_option2 : 
  finalPrice1 P d1_1 d1_2 d1_3 = 12112.50 ∧ 
  finalPrice2 P d2_1 d2_2 d2_3 = 11115 ∧ 
  finalPrice1 P d1_1 d1_2 d1_3 - finalPrice2 P d2_1 d2_2 d2_3 = 997.50 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_merchant_should_choose_option2_l1464_146462


namespace NUMINAMATH_GPT_binomial_12_10_l1464_146425

def binomial (n k : ℕ) : ℕ := n.choose k

theorem binomial_12_10 : binomial 12 10 = 66 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_binomial_12_10_l1464_146425


namespace NUMINAMATH_GPT_minimize_f_l1464_146494

def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

theorem minimize_f : ∀ x : ℝ, f x ≥ f 3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_f_l1464_146494


namespace NUMINAMATH_GPT_Manu_takes_12_more_seconds_l1464_146429

theorem Manu_takes_12_more_seconds (P M A : ℕ) 
  (hP : P = 60) 
  (hA1 : A = 36) 
  (hA2 : A = M / 2) : 
  M - P = 12 :=
by
  sorry

end NUMINAMATH_GPT_Manu_takes_12_more_seconds_l1464_146429


namespace NUMINAMATH_GPT_stephanie_quarters_fraction_l1464_146465

/-- Stephanie has a collection containing exactly one of the first 25 U.S. state quarters. 
    The quarters are in the order the states joined the union.
    Suppose 8 states joined the union between 1800 and 1809. -/
theorem stephanie_quarters_fraction :
  (8 / 25 : ℚ) = (8 / 25) :=
by
  sorry

end NUMINAMATH_GPT_stephanie_quarters_fraction_l1464_146465


namespace NUMINAMATH_GPT_degrees_for_combined_research_l1464_146442

-- Define the conditions as constants.
def microphotonics_percentage : ℝ := 0.10
def home_electronics_percentage : ℝ := 0.24
def food_additives_percentage : ℝ := 0.15
def gmo_percentage : ℝ := 0.29
def industrial_lubricants_percentage : ℝ := 0.08
def nanotechnology_percentage : ℝ := 0.07

noncomputable def remaining_percentage : ℝ :=
  1 - (microphotonics_percentage + home_electronics_percentage + food_additives_percentage +
    gmo_percentage + industrial_lubricants_percentage + nanotechnology_percentage)

noncomputable def total_percentage : ℝ :=
  remaining_percentage + nanotechnology_percentage

noncomputable def degrees_in_circle : ℝ := 360

noncomputable def degrees_representing_combined_research : ℝ :=
  total_percentage * degrees_in_circle

-- State the theorem to prove the correct answer
theorem degrees_for_combined_research : degrees_representing_combined_research = 50.4 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_degrees_for_combined_research_l1464_146442


namespace NUMINAMATH_GPT_basketball_team_avg_weight_l1464_146422

theorem basketball_team_avg_weight :
  let n_tallest := 5
  let w_tallest := 90
  let n_shortest := 4
  let w_shortest := 75
  let n_remaining := 3
  let w_remaining := 80
  let total_weight := (n_tallest * w_tallest) + (n_shortest * w_shortest) + (n_remaining * w_remaining)
  let total_players := n_tallest + n_shortest + n_remaining
  (total_weight / total_players) = 82.5 :=
by
  sorry

end NUMINAMATH_GPT_basketball_team_avg_weight_l1464_146422


namespace NUMINAMATH_GPT_solve_for_x_l1464_146411

theorem solve_for_x (x : ℝ) : (1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1464_146411


namespace NUMINAMATH_GPT_simplify_expression_l1464_146480

open Real

theorem simplify_expression (x : ℝ) (hx : 0 < x) : Real.sqrt (Real.sqrt (x^3 * sqrt (x^5))) = x^(11/8) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1464_146480


namespace NUMINAMATH_GPT_rows_before_change_l1464_146403

-- Definitions and conditions
variables {r c : ℕ}

-- The total number of tiles before and after the change
def total_tiles_before (r c : ℕ) := r * c = 30
def total_tiles_after (r c : ℕ) := (r + 4) * (c - 2) = 30

-- Prove that the number of rows before the change is 3
theorem rows_before_change (h1 : total_tiles_before r c) (h2 : total_tiles_after r c) : r = 3 := 
sorry

end NUMINAMATH_GPT_rows_before_change_l1464_146403


namespace NUMINAMATH_GPT_find_number_l1464_146459

-- Define the hypothesis/condition
def condition (x : ℤ) : Prop := 2 * x + 20 = 8 * x - 4

-- Define the statement to prove
theorem find_number (x : ℤ) (h : condition x) : x = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1464_146459


namespace NUMINAMATH_GPT_how_many_peaches_l1464_146479

-- Define the main problem statement and conditions.
theorem how_many_peaches (A P J_A J_P : ℕ) (h_person_apples: A = 16) (h_person_peaches: P = A + 1) (h_jake_apples: J_A = A + 8) (h_jake_peaches: J_P = P - 6) : P = 17 :=
by
  -- Since the proof is not required, we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_how_many_peaches_l1464_146479


namespace NUMINAMATH_GPT_probability_of_consonant_initials_is_10_over_13_l1464_146457

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' ∨ c = 'Y'

def is_consonant (c : Char) : Prop :=
  ¬(is_vowel c) ∧ c ≠ 'W' 

noncomputable def probability_of_consonant_initials : ℚ :=
  let total_letters := 26
  let number_of_vowels := 6
  let number_of_consonants := total_letters - number_of_vowels
  number_of_consonants / total_letters

theorem probability_of_consonant_initials_is_10_over_13 :
  probability_of_consonant_initials = 10 / 13 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_consonant_initials_is_10_over_13_l1464_146457


namespace NUMINAMATH_GPT_Thursday_total_rainfall_correct_l1464_146470

def Monday_rainfall : ℝ := 0.9
def Tuesday_rainfall : ℝ := Monday_rainfall - 0.7
def Wednesday_rainfall : ℝ := Tuesday_rainfall + 0.5 * Tuesday_rainfall
def additional_rain : ℝ := 0.3
def decrease_factor : ℝ := 0.2
def Thursday_rainfall_before_addition : ℝ := Wednesday_rainfall - decrease_factor * Wednesday_rainfall
def Thursday_total_rainfall : ℝ := Thursday_rainfall_before_addition + additional_rain

theorem Thursday_total_rainfall_correct :
  Thursday_total_rainfall = 0.54 :=
by
  sorry

end NUMINAMATH_GPT_Thursday_total_rainfall_correct_l1464_146470


namespace NUMINAMATH_GPT_aquarium_height_l1464_146492

theorem aquarium_height (h : ℝ) (V : ℝ) (final_volume : ℝ) :
  let length := 4
  let width := 6
  let halfway_volume := (length * width * h) / 2
  let spilled_volume := halfway_volume / 2
  let tripled_volume := 3 * spilled_volume
  tripled_volume = final_volume →
  final_volume = 54 →
  h = 3 := by
  intros
  sorry

end NUMINAMATH_GPT_aquarium_height_l1464_146492


namespace NUMINAMATH_GPT_cos_diff_alpha_beta_l1464_146449

theorem cos_diff_alpha_beta (α β : ℝ) (h1 : Real.sin α = 2 / 3) (h2 : Real.cos β = -3 / 4)
    (h3 : α ∈ Set.Ioo (π / 2) π) (h4 : β ∈ Set.Ioo π (3 * π / 2)) :
    Real.cos (α - β) = (3 * Real.sqrt 5 - 2 * Real.sqrt 7) / 12 := 
sorry

end NUMINAMATH_GPT_cos_diff_alpha_beta_l1464_146449


namespace NUMINAMATH_GPT_good_numbers_10_70_l1464_146409

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def no_repeating_digits (n : ℕ) : Prop :=
  (n / 10 ≠ n % 10)

def is_good_number (n : ℕ) : Prop :=
  no_repeating_digits n ∧ (n % sum_of_digits n = 0)

theorem good_numbers_10_70 :
  is_good_number 10 ∧ is_good_number (10 + 11) ∧
  is_good_number 70 ∧ is_good_number (70 + 11) :=
by {
  -- Check that 10 is a good number
  -- Check that 21 is a good number
  -- Check that 70 is a good number
  -- Check that 81 is a good number
  sorry
}

end NUMINAMATH_GPT_good_numbers_10_70_l1464_146409


namespace NUMINAMATH_GPT_functional_equation_odd_l1464_146496

   variable {R : Type*} [AddCommGroup R] [Module ℝ R]

   def isOdd (f : ℝ → ℝ) : Prop :=
     ∀ x : ℝ, f (-x) = -f x

   theorem functional_equation_odd (f : ℝ → ℝ)
       (h_fun : ∀ x y : ℝ, f (x + y) = f x + f y) : isOdd f :=
   by
     sorry
   
end NUMINAMATH_GPT_functional_equation_odd_l1464_146496


namespace NUMINAMATH_GPT_min_value_a_plus_3b_l1464_146445

theorem min_value_a_plus_3b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a * b - 3 = a + 3 * b) :
  ∃ x : ℝ, x = 6 ∧ ∀ y : ℝ, y = a + 3 * b → y ≥ 6 :=
sorry

end NUMINAMATH_GPT_min_value_a_plus_3b_l1464_146445


namespace NUMINAMATH_GPT_fraction_equal_l1464_146456

theorem fraction_equal {a b x : ℝ} (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + b) / (a - b) = (x + 1) / (x - 1) := 
by
  sorry

end NUMINAMATH_GPT_fraction_equal_l1464_146456


namespace NUMINAMATH_GPT_find_quadruples_l1464_146419

def is_prime (n : ℕ) := ∀ m, m ∣ n → m = 1 ∨ m = n

 theorem find_quadruples (p q a b : ℕ) (hp : is_prime p) (hq : is_prime q) (ha : 1 < a)
  : (p^a = 1 + 5 * q^b ↔ ((p = 2 ∧ q = 3 ∧ a = 4 ∧ b = 1) ∨ (p = 3 ∧ q = 2 ∧ a = 4 ∧ b = 4))) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_quadruples_l1464_146419


namespace NUMINAMATH_GPT_find_c_minus_2d_l1464_146476

theorem find_c_minus_2d :
  ∃ (c d : ℕ), (c > d) ∧ (c - 2 * d = 0) ∧ (∀ x : ℕ, (x^2 - 18 * x + 72 = (x - c) * (x - d))) :=
by
  sorry

end NUMINAMATH_GPT_find_c_minus_2d_l1464_146476


namespace NUMINAMATH_GPT_find_a3_l1464_146491

noncomputable def S (n : ℕ) : ℤ := 2 * n^2 - 1
noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

theorem find_a3 : a 3 = 10 := by
  sorry

end NUMINAMATH_GPT_find_a3_l1464_146491


namespace NUMINAMATH_GPT_round_trip_in_first_trip_l1464_146424

def percentage_rt_trip_first_trip := 0.3 -- 30%
def percentage_2t_trip_second_trip := 0.6 -- 60%
def percentage_ow_trip_third_trip := 0.45 -- 45%

theorem round_trip_in_first_trip (P1 P2 P3: ℝ) (C1 C2 C3: ℝ) 
  (h1 : P1 = 0.3) 
  (h2 : 0 < P1 ∧ P1 < 1) 
  (h3 : P2 = 0.6) 
  (h4 : 0 < P2 ∧ P2 < 1) 
  (h5 : P3 = 0.45) 
  (h6 : 0 < P3 ∧ P3 < 1) 
  (h7 : C1 + C2 + C3 = 1) 
  (h8 : (C1 = (1 - P1) * 0.15)) 
  (h9 : C2 = 0.2 * P2) 
  (h10 : C3 = 0.1 * P3) :
  P1 = 0.3 := by
  sorry

end NUMINAMATH_GPT_round_trip_in_first_trip_l1464_146424


namespace NUMINAMATH_GPT_painted_cubes_l1464_146430

/-- 
  Given a cube of side 9 painted red and cut into smaller cubes of side 3,
  prove the number of smaller cubes with paint on exactly 2 sides is 12.
-/
theorem painted_cubes (l : ℕ) (s : ℕ) (n : ℕ) (edges : ℕ) (faces : ℕ)
  (hcube_dimension : l = 9) (hsmaller_cubes_dimension : s = 3) 
  (hedges : edges = 12) (hfaces : faces * edges = 12) 
  (htotal_cubes : n = (l^3) / (s^3)) : 
  n * faces = 12 :=
sorry

end NUMINAMATH_GPT_painted_cubes_l1464_146430


namespace NUMINAMATH_GPT_reciprocal_sum_of_roots_l1464_146458

theorem reciprocal_sum_of_roots :
  (∃ m n : ℝ, (m^2 + 2 * m - 3 = 0) ∧ (n^2 + 2 * n - 3 = 0) ∧ m ≠ n) →
  (∃ m n : ℝ, (1/m + 1/n = 2/3)) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_sum_of_roots_l1464_146458


namespace NUMINAMATH_GPT_value_of_expression_l1464_146486

theorem value_of_expression (a b c d : ℕ) (h1 : a = 4 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) :
  (a * c) / (b * d) = 20 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1464_146486


namespace NUMINAMATH_GPT_eight_lines_no_parallel_no_concurrent_l1464_146421

-- Define the number of regions into which n lines divide the plane
def regions (n : ℕ) : ℕ :=
if n = 0 then 1
else if n = 1 then 2
else n * (n - 1) / 2 + n + 1

theorem eight_lines_no_parallel_no_concurrent :
  regions 8 = 37 :=
by
  sorry

end NUMINAMATH_GPT_eight_lines_no_parallel_no_concurrent_l1464_146421


namespace NUMINAMATH_GPT_jean_to_shirt_ratio_l1464_146432

theorem jean_to_shirt_ratio (shirts_sold jeans_sold shirt_cost total_revenue: ℕ) (h1: shirts_sold = 20) (h2: jeans_sold = 10) (h3: shirt_cost = 10) (h4: total_revenue = 400) : 
(shirt_cost * shirts_sold + jeans_sold * ((total_revenue - (shirt_cost * shirts_sold)) / jeans_sold)) / (total_revenue - (shirt_cost * shirts_sold)) / jeans_sold = 2 := 
sorry

end NUMINAMATH_GPT_jean_to_shirt_ratio_l1464_146432


namespace NUMINAMATH_GPT_closest_point_is_correct_l1464_146444

def line_eq (x : ℝ) : ℝ := -3 * x + 5

def closest_point_on_line_to_given_point : Prop :=
  ∃ (x y : ℝ), y = line_eq x ∧ (x, y) = (17 / 10, -1 / 10) ∧
  (∀ (x' y' : ℝ), y' = line_eq x' → (x' - -4)^2 + (y' - -2)^2 ≥ (x - -4)^2 + (y - -2)^2)
  
theorem closest_point_is_correct : closest_point_on_line_to_given_point :=
sorry

end NUMINAMATH_GPT_closest_point_is_correct_l1464_146444


namespace NUMINAMATH_GPT_plate_and_roller_acceleration_l1464_146455

noncomputable def m : ℝ := 150
noncomputable def g : ℝ := 10
noncomputable def R : ℝ := 1
noncomputable def r : ℝ := 0.4
noncomputable def alpha : ℝ := Real.arccos 0.68

theorem plate_and_roller_acceleration :
  let sin_alpha_half := Real.sin (alpha / 2)
  sin_alpha_half = 0.4 →
  plate_acceleration == 4 ∧ direction == Real.arcsin 0.4 ∧ rollers_acceleration == 4 :=
by
  sorry

end NUMINAMATH_GPT_plate_and_roller_acceleration_l1464_146455


namespace NUMINAMATH_GPT_find_m_l1464_146400

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 9

theorem find_m (m : ℝ) : f 5 - g 5 m = 20 → m = -16.8 :=
by
  -- Given f(x) and g(x, m) definitions, we want to prove m = -16.8 given f 5 - g 5 m = 20.
  sorry

end NUMINAMATH_GPT_find_m_l1464_146400


namespace NUMINAMATH_GPT_exists_large_cube_construction_l1464_146493

theorem exists_large_cube_construction (n : ℕ) :
  ∃ N : ℕ, ∀ n > N, ∃ k : ℕ, k^3 = n :=
sorry

end NUMINAMATH_GPT_exists_large_cube_construction_l1464_146493


namespace NUMINAMATH_GPT_quadratic_eq_integer_roots_iff_l1464_146472

theorem quadratic_eq_integer_roots_iff (n : ℕ) (hn : n > 0) :
  (∃ x y : ℤ, x * y = n ∧ x + y = 4) ↔ (n = 3 ∨ n = 4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_eq_integer_roots_iff_l1464_146472


namespace NUMINAMATH_GPT_circle_land_represents_30105_l1464_146487

-- Definitions based on the problem's conditions
def circleLandNumber (digits : List (ℕ × ℕ)) : ℕ :=
  digits.foldl (λ acc (d_circle : ℕ × ℕ) => acc + d_circle.fst * 10^d_circle.snd) 0

-- Example 207
def number_207 : List (ℕ × ℕ) := [(2, 2), (0, 0), (7, 0)]

-- Example 4520
def number_4520 : List (ℕ × ℕ) := [(4, 3), (5, 1), (2, 0), (0, 0)]

-- The diagram to analyze
def given_diagram : List (ℕ × ℕ) := [(3, 4), (1, 2), (5, 0)]

-- The statement proving the given diagram represents 30105 in Circle Land
theorem circle_land_represents_30105 : circleLandNumber given_diagram = 30105 :=
  sorry

end NUMINAMATH_GPT_circle_land_represents_30105_l1464_146487


namespace NUMINAMATH_GPT_percent_increase_decrease_condition_l1464_146416

theorem percent_increase_decrease_condition (p q M : ℝ) (hp : 0 < p) (hq : 0 < q) (hM : 0 < M) (hq50 : q < 50) :
  (M * (1 + p / 100) * (1 - q / 100) < M) ↔ (p < 100 * q / (100 - q)) := 
sorry

end NUMINAMATH_GPT_percent_increase_decrease_condition_l1464_146416


namespace NUMINAMATH_GPT_midline_equation_l1464_146475

theorem midline_equation (a b : ℝ) (K1 K2 : ℝ)
  (h1 : K1^2 = (a^2) / 4 + b^2)
  (h2 : K2^2 = a^2 + (b^2) / 4) :
  16 * K2^2 - 4 * K1^2 = 15 * a^2 :=
by
  sorry

end NUMINAMATH_GPT_midline_equation_l1464_146475


namespace NUMINAMATH_GPT_percentage_of_rotten_oranges_l1464_146440

-- Define the conditions
def total_oranges : ℕ := 600
def total_bananas : ℕ := 400
def rotten_bananas_percentage : ℝ := 0.08
def good_fruits_percentage : ℝ := 0.878

-- Define the proof problem
theorem percentage_of_rotten_oranges :
  let total_fruits := total_oranges + total_bananas
  let number_of_rotten_bananas := rotten_bananas_percentage * total_bananas
  let number_of_good_fruits := good_fruits_percentage * total_fruits
  let number_of_rotten_fruits := total_fruits - number_of_good_fruits
  let number_of_rotten_oranges := number_of_rotten_fruits - number_of_rotten_bananas
  let percentage_of_rotten_oranges := (number_of_rotten_oranges / total_oranges) * 100
  percentage_of_rotten_oranges = 15 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_rotten_oranges_l1464_146440


namespace NUMINAMATH_GPT_min_digits_fraction_l1464_146434

def minDigitsToRightOfDecimal (n : ℕ) : ℕ :=
  -- This represents the minimum number of digits needed to express n / (2^15 * 5^7)
  -- as a decimal.
  -- The actual function body is hypothetical and not implemented here.
  15

theorem min_digits_fraction :
  minDigitsToRightOfDecimal 987654321 = 15 :=
by
  sorry

end NUMINAMATH_GPT_min_digits_fraction_l1464_146434


namespace NUMINAMATH_GPT_sample_size_correct_l1464_146468

variable (total_employees young_employees middle_aged_employees elderly_employees young_in_sample sample_size : ℕ)

-- Conditions
def total_number_of_employees := 75
def number_of_young_employees := 35
def number_of_middle_aged_employees := 25
def number_of_elderly_employees := 15
def number_of_young_in_sample := 7
def stratified_sampling := true

-- The proof problem statement
theorem sample_size_correct :
  total_employees = total_number_of_employees ∧ 
  young_employees = number_of_young_employees ∧ 
  middle_aged_employees = number_of_middle_aged_employees ∧ 
  elderly_employees = number_of_elderly_employees ∧ 
  young_in_sample = number_of_young_in_sample ∧ 
  stratified_sampling → 
  sample_size = 15 := by sorry

end NUMINAMATH_GPT_sample_size_correct_l1464_146468


namespace NUMINAMATH_GPT_find_min_positive_n_l1464_146406

-- Assume the sequence {a_n} is given
variables {a : ℕ → ℤ}

-- Given conditions
-- a4 < 0 and a5 > |a4|
def condition1 (a : ℕ → ℤ) : Prop := a 4 < 0
def condition2 (a : ℕ → ℤ) : Prop := a 5 > abs (a 4)

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) (a : ℕ → ℤ) : ℤ := n * (a 1 + a n) / 2

-- The main theorem we need to prove
theorem find_min_positive_n (a : ℕ → ℤ) (h1 : condition1 a) (h2 : condition2 a) : ∃ n : ℕ, n = 8 ∧ S n a > 0 :=
by
  sorry

end NUMINAMATH_GPT_find_min_positive_n_l1464_146406


namespace NUMINAMATH_GPT_midpoint_of_diagonal_l1464_146427

-- Definition of the points
def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (14, 9)

-- Statement about the midpoint of a diagonal in a rectangle
theorem midpoint_of_diagonal : 
  ∀ (x1 y1 x2 y2 : ℝ), (x1, y1) = point1 → (x2, y2) = point2 →
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  (midpoint_x, midpoint_y) = (8, 3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_midpoint_of_diagonal_l1464_146427


namespace NUMINAMATH_GPT_chickens_in_farm_l1464_146423

theorem chickens_in_farm (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 := by sorry

end NUMINAMATH_GPT_chickens_in_farm_l1464_146423


namespace NUMINAMATH_GPT_quadratic_roots_ratio_l1464_146497

theorem quadratic_roots_ratio (a b c : ℝ) (h1 : ∀ (s1 s2 : ℝ), s1 * s2 = a → s1 + s2 = -c → 3 * s1 + 3 * s2 = -a → 9 * s1 * s2 = b) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) :
  b / c = 27 := sorry

end NUMINAMATH_GPT_quadratic_roots_ratio_l1464_146497


namespace NUMINAMATH_GPT_max_n_for_positive_sum_l1464_146433

-- Define the arithmetic sequence \(a_n\)
def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + n * d

-- Define the sum of the first n terms of the arithmetic sequence
def S_n (a d : ℤ) (n : ℕ) := n * (2 * a + (n-1) * d) / 2

theorem max_n_for_positive_sum 
  (a : ℤ) 
  (d : ℤ) 
  (h_max_sum : ∃ m : ℕ, S_n a d m = S_n a d (m+1))
  (h_ratio : (arithmetic_sequence a d 15) / (arithmetic_sequence a d 14) < -1) :
  27 = 27 :=
sorry

end NUMINAMATH_GPT_max_n_for_positive_sum_l1464_146433


namespace NUMINAMATH_GPT_multiply_scaled_values_l1464_146415

theorem multiply_scaled_values (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by 
  sorry

end NUMINAMATH_GPT_multiply_scaled_values_l1464_146415
