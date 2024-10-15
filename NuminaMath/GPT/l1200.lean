import Mathlib

namespace NUMINAMATH_GPT_min_value_expression_l1200_120092

theorem min_value_expression (a : ℝ) (h : a > 2) : a + 4 / (a - 2) ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1200_120092


namespace NUMINAMATH_GPT_system_solution_l1200_120027

theorem system_solution (x y : ℝ) (h1 : 4 * x - y = 3) (h2 : x + 6 * y = 17) : x + y = 4 :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l1200_120027


namespace NUMINAMATH_GPT_log_sum_l1200_120076

theorem log_sum : 2 * Real.log 2 + Real.log 25 = 2 := 
by 
  sorry

end NUMINAMATH_GPT_log_sum_l1200_120076


namespace NUMINAMATH_GPT_unique_solution_l1200_120082

noncomputable def pair_satisfying_equation (m n : ℕ) : Prop :=
  2^m - 1 = 3^n

theorem unique_solution : ∀ (m n : ℕ), m > 0 → n > 0 → pair_satisfying_equation m n → (m, n) = (2, 1) :=
by
  intros m n m_pos n_pos h
  sorry

end NUMINAMATH_GPT_unique_solution_l1200_120082


namespace NUMINAMATH_GPT_koala_fiber_consumption_l1200_120032

theorem koala_fiber_consumption (x : ℝ) (H : 12 = 0.30 * x) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_koala_fiber_consumption_l1200_120032


namespace NUMINAMATH_GPT_markup_is_correct_l1200_120025

-- The mathematical interpretation of the given conditions
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.05
def net_profit : ℝ := 12

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the total cost calculation
def total_cost : ℝ := purchase_price + overhead_cost

-- Define the selling price calculation
def selling_price : ℝ := total_cost + net_profit

-- Define the markup calculation
def markup : ℝ := selling_price - purchase_price

-- The statement we want to prove
theorem markup_is_correct : markup = 14.40 :=
by
  -- We will eventually prove this, but for now we use sorry as a placeholder
  sorry

end NUMINAMATH_GPT_markup_is_correct_l1200_120025


namespace NUMINAMATH_GPT_mr_brown_final_price_is_correct_l1200_120073

noncomputable def mr_brown_final_purchase_price :
  Float :=
  let initial_price : Float := 100000
  let mr_brown_price  := initial_price * 1.12
  let improvement := mr_brown_price * 0.05
  let mr_brown_total_investment := mr_brown_price + improvement
  let mr_green_purchase_price := mr_brown_total_investment * 1.04
  let market_decline := mr_green_purchase_price * 0.03
  let value_after_decline := mr_green_purchase_price - market_decline
  let loss := value_after_decline * 0.10
  let ms_white_purchase_price := value_after_decline - loss
  let market_increase := ms_white_purchase_price * 0.08
  let value_after_increase := ms_white_purchase_price + market_increase
  let profit := value_after_increase * 0.05
  let final_price := value_after_increase + profit
  final_price

theorem mr_brown_final_price_is_correct :
  mr_brown_final_purchase_price = 121078.76 := by
  sorry

end NUMINAMATH_GPT_mr_brown_final_price_is_correct_l1200_120073


namespace NUMINAMATH_GPT_Julie_and_Matt_ate_cookies_l1200_120039

def initial_cookies : ℕ := 32
def remaining_cookies : ℕ := 23

theorem Julie_and_Matt_ate_cookies : initial_cookies - remaining_cookies = 9 :=
by
  sorry

end NUMINAMATH_GPT_Julie_and_Matt_ate_cookies_l1200_120039


namespace NUMINAMATH_GPT_students_minus_rabbits_l1200_120018

-- Define the number of students per classroom
def students_per_classroom : ℕ := 24

-- Define the number of rabbits per classroom
def rabbits_per_classroom : ℕ := 3

-- Define the number of classrooms
def number_of_classrooms : ℕ := 5

-- Define the total number of students and rabbits
def total_students : ℕ := students_per_classroom * number_of_classrooms
def total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms

-- The main statement to prove
theorem students_minus_rabbits :
  total_students - total_rabbits = 105 :=
by
  sorry

end NUMINAMATH_GPT_students_minus_rabbits_l1200_120018


namespace NUMINAMATH_GPT_rectangle_dimensions_l1200_120091

theorem rectangle_dimensions (x y : ℝ) (h1 : x = 2 * y) (h2 : 2 * (x + y) = 2 * x * y) : 
  (x = 3 ∧ y = 1.5) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1200_120091


namespace NUMINAMATH_GPT_pure_imaginary_number_implies_x_eq_1_l1200_120036

theorem pure_imaginary_number_implies_x_eq_1 (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x + 1 ≠ 0) : x = 1 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_number_implies_x_eq_1_l1200_120036


namespace NUMINAMATH_GPT_technicians_count_l1200_120072

/-- Given a workshop with 49 workers, where the average salary of all workers 
    is Rs. 8000, the average salary of the technicians is Rs. 20000, and the
    average salary of the rest is Rs. 6000, prove that the number of 
    technicians is 7. -/
theorem technicians_count (T R : ℕ) (h1 : T + R = 49) (h2 : 10 * T + 3 * R = 196) : T = 7 := 
by
  sorry

end NUMINAMATH_GPT_technicians_count_l1200_120072


namespace NUMINAMATH_GPT_fruit_basket_combinations_l1200_120052

theorem fruit_basket_combinations :
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples+1) * (oranges+1) * (bananas+1)
  let empty_basket := 1
  total_combinations - empty_basket = 159 :=
by
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples + 1) * (oranges + 1) * (bananas + 1)
  let empty_basket := 1
  have h_total_combinations : total_combinations = 4 * 8 * 5 := by sorry
  have h_empty_basket : empty_basket = 1 := by sorry
  have h_subtract : 4 * 8 * 5 - 1 = 159 := by sorry
  exact h_subtract

end NUMINAMATH_GPT_fruit_basket_combinations_l1200_120052


namespace NUMINAMATH_GPT_value_of_expression_l1200_120037

theorem value_of_expression (x : ℝ) (h : x^2 - 3 * x = 4) : 3 * x^2 - 9 * x + 8 = 20 := 
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1200_120037


namespace NUMINAMATH_GPT_water_wasted_in_one_hour_l1200_120054

theorem water_wasted_in_one_hour:
  let drips_per_minute : ℕ := 10
  let drop_volume : ℝ := 0.05 -- volume in mL
  let minutes_in_hour : ℕ := 60
  drips_per_minute * drop_volume * minutes_in_hour = 30 := by
  sorry

end NUMINAMATH_GPT_water_wasted_in_one_hour_l1200_120054


namespace NUMINAMATH_GPT_joe_paint_usage_l1200_120094

theorem joe_paint_usage :
  let initial_paint := 360
  let first_week_usage := (1 / 3: ℝ) * initial_paint
  let remaining_after_first_week := initial_paint - first_week_usage
  let second_week_usage := (1 / 5: ℝ) * remaining_after_first_week
  let total_usage := first_week_usage + second_week_usage
  total_usage = 168 :=
by
  sorry

end NUMINAMATH_GPT_joe_paint_usage_l1200_120094


namespace NUMINAMATH_GPT_andrea_still_needs_rhinestones_l1200_120075

def total_rhinestones_needed : ℕ := 45
def rhinestones_bought : ℕ := total_rhinestones_needed / 3
def rhinestones_found : ℕ := total_rhinestones_needed / 5
def rhinestones_total_have : ℕ := rhinestones_bought + rhinestones_found
def rhinestones_still_needed : ℕ := total_rhinestones_needed - rhinestones_total_have

theorem andrea_still_needs_rhinestones : rhinestones_still_needed = 21 := by
  rfl

end NUMINAMATH_GPT_andrea_still_needs_rhinestones_l1200_120075


namespace NUMINAMATH_GPT_time_with_cat_total_l1200_120069

def time_spent_with_cat (petting combing brushing playing feeding cleaning : ℕ) : ℕ :=
  petting + combing + brushing + playing + feeding + cleaning

theorem time_with_cat_total :
  let petting := 12
  let combing := 1/3 * petting
  let brushing := 1/4 * combing
  let playing := 1/2 * petting
  let feeding := 5
  let cleaning := 2/5 * feeding
  time_spent_with_cat petting combing brushing playing feeding cleaning = 30 := by
  sorry

end NUMINAMATH_GPT_time_with_cat_total_l1200_120069


namespace NUMINAMATH_GPT_least_third_side_length_l1200_120058

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end NUMINAMATH_GPT_least_third_side_length_l1200_120058


namespace NUMINAMATH_GPT_handrail_length_is_17_point_3_l1200_120053

noncomputable def length_of_handrail (turn : ℝ) (rise : ℝ) (radius : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let arc_length := (turn / 360) * circumference
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_is_17_point_3 : length_of_handrail 270 10 3 = 17.3 :=
by 
  sorry

end NUMINAMATH_GPT_handrail_length_is_17_point_3_l1200_120053


namespace NUMINAMATH_GPT_number_of_sports_books_l1200_120033

def total_books : ℕ := 58
def school_books : ℕ := 19
def sports_books (total_books school_books : ℕ) : ℕ := total_books - school_books

theorem number_of_sports_books : sports_books total_books school_books = 39 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_sports_books_l1200_120033


namespace NUMINAMATH_GPT_yuna_has_biggest_number_l1200_120078

theorem yuna_has_biggest_number (yoongi : ℕ) (jungkook : ℕ) (yuna : ℕ) (hy : yoongi = 7) (hj : jungkook = 6) (hn : yuna = 9) :
  yuna = 9 ∧ yuna > yoongi ∧ yuna > jungkook :=
by 
  sorry

end NUMINAMATH_GPT_yuna_has_biggest_number_l1200_120078


namespace NUMINAMATH_GPT_unique_measures_of_A_l1200_120008

theorem unique_measures_of_A : 
  ∃ n : ℕ, n = 17 ∧ 
    (∀ A B : ℕ, 
      (A > 0) ∧ (B > 0) ∧ (A + B = 180) ∧ (∃ k : ℕ, A = k * B) → 
      ∃! A : ℕ, A > 0 ∧ (A + B = 180)) :=
sorry

end NUMINAMATH_GPT_unique_measures_of_A_l1200_120008


namespace NUMINAMATH_GPT_difference_between_perfect_and_cracked_l1200_120051

def total_eggs : ℕ := 24
def broken_eggs : ℕ := 3
def cracked_eggs : ℕ := 2 * broken_eggs

def perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs
def difference : ℕ := perfect_eggs - cracked_eggs

theorem difference_between_perfect_and_cracked :
  difference = 9 := by
  sorry

end NUMINAMATH_GPT_difference_between_perfect_and_cracked_l1200_120051


namespace NUMINAMATH_GPT_initial_number_of_peanuts_l1200_120015

theorem initial_number_of_peanuts (x : ℕ) (h : x + 2 = 6) : x = 4 :=
sorry

end NUMINAMATH_GPT_initial_number_of_peanuts_l1200_120015


namespace NUMINAMATH_GPT_y1_mul_y2_eq_one_l1200_120065

theorem y1_mul_y2_eq_one (x1 x2 y1 y2 : ℝ) (h1 : y1^2 = x1) (h2 : y2^2 = x2) 
  (h3 : y1 / (y1^2 - 1) = - (y2 / (y2^2 - 1))) (h4 : y1 + y2 ≠ 0) : y1 * y2 = 1 :=
sorry

end NUMINAMATH_GPT_y1_mul_y2_eq_one_l1200_120065


namespace NUMINAMATH_GPT_probability_sum_greater_than_six_l1200_120064

variable (A : Finset ℕ) (B : Finset ℕ)
variable (balls_in_A : A = {1, 2}) (balls_in_B : B = {3, 4, 5, 6})

theorem probability_sum_greater_than_six : 
  (∃ selected_pair ∈ (A.product B), selected_pair.1 + selected_pair.2 > 6) →
  (Finset.filter (λ pair => pair.1 + pair.2 > 6) (A.product B)).card / 
  (A.product B).card = 3 / 8 := sorry

end NUMINAMATH_GPT_probability_sum_greater_than_six_l1200_120064


namespace NUMINAMATH_GPT_playground_length_l1200_120086

theorem playground_length
  (P : ℕ)
  (B : ℕ)
  (h1 : P = 1200)
  (h2 : B = 500)
  (h3 : P = 2 * (100 + B)) :
  100 = 100 :=
 by sorry

end NUMINAMATH_GPT_playground_length_l1200_120086


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1200_120035

theorem sum_of_squares_of_roots :
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂) →
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = 79 / 25) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1200_120035


namespace NUMINAMATH_GPT_find_s_l1200_120012

section
variables {a b c p q s : ℕ}

-- Conditions given in the problem
variables (h1 : a + b = p)
variables (h2 : p + c = s)
variables (h3 : s + a = q)
variables (h4 : b + c + q = 18)
variables (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
variables (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0)

-- Statement of the problem
theorem find_s (h1 : a + b = p) (h2 : p + c = s) (h3 : s + a = q) (h4 : b + c + q = 18)
  (h5 : a ≠ b ∧ a ≠ c ∧ a ≠ p ∧ a ≠ q ∧ a ≠ s ∧ b ≠ c ∧ b ≠ p ∧ b ≠ q ∧ b ≠ s ∧ c ≠ p ∧ c ≠ q ∧ c ≠ s ∧ p ≠ q ∧ p ≠ s ∧ q ≠ s)
  (h6 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ p ≠ 0 ∧ q ≠ 0 ∧ s ≠ 0) :
  s = 9 :=
sorry
end

end NUMINAMATH_GPT_find_s_l1200_120012


namespace NUMINAMATH_GPT_dice_game_probability_l1200_120088

def is_valid_roll (d1 d2 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6

def score (d1 d2 : ℕ) : ℕ :=
  max d1 d2

def favorable_outcomes : List (ℕ × ℕ) :=
  [ (1, 1), (1, 2), (2, 1), (2, 2), 
    (1, 3), (2, 3), (3, 1), (3, 2), (3, 3) ]

def total_outcomes : ℕ := 36

def favorable_count : ℕ := favorable_outcomes.length

theorem dice_game_probability : 
  (favorable_count : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_dice_game_probability_l1200_120088


namespace NUMINAMATH_GPT_xy_sum_possible_values_l1200_120014

theorem xy_sum_possible_values (x y : ℕ) (h1 : x < 20) (h2 : y < 20) (h3 : 0 < x) (h4 : 0 < y) (h5 : x + y + x * y = 95) :
  x + y = 18 ∨ x + y = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_xy_sum_possible_values_l1200_120014


namespace NUMINAMATH_GPT_samia_walking_distance_l1200_120060

noncomputable def total_distance (x : ℝ) : ℝ := 4 * x
noncomputable def biking_distance (x : ℝ) : ℝ := 3 * x
noncomputable def walking_distance (x : ℝ) : ℝ := x
noncomputable def biking_time (x : ℝ) : ℝ := biking_distance x / 12
noncomputable def walking_time (x : ℝ) : ℝ := walking_distance x / 4
noncomputable def total_time (x : ℝ) : ℝ := biking_time x + walking_time x

theorem samia_walking_distance : ∀ (x : ℝ), total_time x = 1 → walking_distance x = 2 :=
by
  sorry

end NUMINAMATH_GPT_samia_walking_distance_l1200_120060


namespace NUMINAMATH_GPT_no_natural_numbers_satisfying_conditions_l1200_120021

theorem no_natural_numbers_satisfying_conditions :
  ¬ ∃ (a b : ℕ), a < b ∧ ∃ k : ℕ, b^2 + 4*a = k^2 := by
  sorry

end NUMINAMATH_GPT_no_natural_numbers_satisfying_conditions_l1200_120021


namespace NUMINAMATH_GPT_class_student_count_l1200_120093

-- Statement: Prove that under the given conditions, the number of students in the class is 19.
theorem class_student_count (n : ℕ) (avg_students_age : ℕ) (teacher_age : ℕ) (avg_with_teacher : ℕ):
  avg_students_age = 20 → 
  teacher_age = 40 → 
  avg_with_teacher = 21 → 
  21 * (n + 1) = 20 * n + 40 → 
  n = 19 := 
by 
  intros h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_class_student_count_l1200_120093


namespace NUMINAMATH_GPT_bread_slices_leftover_l1200_120026

-- Definitions based on conditions provided in the problem
def total_bread_slices : ℕ := 2 * 20
def total_ham_slices : ℕ := 2 * 8
def sandwiches_made : ℕ := total_ham_slices
def bread_slices_needed : ℕ := sandwiches_made * 2

-- Theorem we want to prove
theorem bread_slices_leftover : total_bread_slices - bread_slices_needed = 8 :=
by 
    -- Insert steps of proof here
    sorry

end NUMINAMATH_GPT_bread_slices_leftover_l1200_120026


namespace NUMINAMATH_GPT_trouser_sale_price_l1200_120059

theorem trouser_sale_price 
  (original_price : ℝ) 
  (percent_decrease : ℝ) 
  (sale_price : ℝ) 
  (h : original_price = 100) 
  (p : percent_decrease = 0.25) 
  (s : sale_price = original_price * (1 - percent_decrease)) : 
  sale_price = 75 :=
by 
  sorry

end NUMINAMATH_GPT_trouser_sale_price_l1200_120059


namespace NUMINAMATH_GPT_combined_work_time_l1200_120007

noncomputable def work_time_first_worker : ℤ := 5
noncomputable def work_time_second_worker : ℤ := 4

theorem combined_work_time :
  (1 / (1 / work_time_first_worker + 1 / work_time_second_worker)) = 20 / 9 :=
by
  unfold work_time_first_worker work_time_second_worker
  -- The detailed reasoning and computation would go here
  sorry

end NUMINAMATH_GPT_combined_work_time_l1200_120007


namespace NUMINAMATH_GPT_num_men_in_second_group_l1200_120071

-- Define the conditions
def numMen1 := 4
def hoursPerDay1 := 10
def daysPerWeek := 7
def earningsPerWeek1 := 1200

def hoursPerDay2 := 6
def earningsPerWeek2 := 1620

-- Define the earning per man-hour
def earningPerManHour := earningsPerWeek1 / (numMen1 * hoursPerDay1 * daysPerWeek)

-- Define the total man-hours required for the second amount of earnings
def totalManHours2 := earningsPerWeek2 / earningPerManHour

-- Define the number of men in the second group
def numMen2 := totalManHours2 / (hoursPerDay2 * daysPerWeek)

-- Theorem stating the number of men in the second group 
theorem num_men_in_second_group : numMen2 = 9 := by
  sorry

end NUMINAMATH_GPT_num_men_in_second_group_l1200_120071


namespace NUMINAMATH_GPT_zongzi_cost_prices_l1200_120066

theorem zongzi_cost_prices (a : ℕ) (n : ℕ)
  (h1 : n * a = 8000)
  (h2 : n * (a - 10) = 6000)
  : a = 40 ∧ a - 10 = 30 :=
by
  sorry

end NUMINAMATH_GPT_zongzi_cost_prices_l1200_120066


namespace NUMINAMATH_GPT_sequence_propositions_l1200_120085

theorem sequence_propositions (a : ℕ → ℝ) (h_seq : a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 > a 4 ∧ a 4 ≥ 0) 
  (h_sub : ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 4 → ∃ k, a i - a j = a k) :
  (∀ k, ∃ d, a k = a 1 - d * (k - 1)) ∧
  (∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ i * a i = j * a j) ∧
  (∃ i, a i = 0) :=
by
  sorry

end NUMINAMATH_GPT_sequence_propositions_l1200_120085


namespace NUMINAMATH_GPT_pq_difference_l1200_120009

theorem pq_difference (p q : ℝ) (h1 : 3 / p = 6) (h2 : 3 / q = 15) : p - q = 3 / 10 := by
  sorry

end NUMINAMATH_GPT_pq_difference_l1200_120009


namespace NUMINAMATH_GPT_winnie_keeps_balloons_l1200_120056

theorem winnie_keeps_balloons (red white green chartreuse friends total remainder : ℕ) (hRed : red = 17) (hWhite : white = 33) (hGreen : green = 65) (hChartreuse : chartreuse = 83) (hFriends : friends = 10) (hTotal : total = red + white + green + chartreuse) (hDiv : total % friends = remainder) : remainder = 8 :=
by
  have hTotal_eq : total = 198 := by
    sorry -- This would be the computation of 17 + 33 + 65 + 83
  have hRemainder_eq : 198 % 10 = remainder := by
    sorry -- This would involve the computation of the remainder
  exact sorry -- This would be the final proof that remainder = 8, tying all parts together

end NUMINAMATH_GPT_winnie_keeps_balloons_l1200_120056


namespace NUMINAMATH_GPT_charity_amount_l1200_120041

theorem charity_amount (total : ℝ) (charities : ℕ) (amount_per_charity : ℝ) 
  (h1 : total = 3109) (h2 : charities = 25) : 
  amount_per_charity = 124.36 :=
by
  sorry

end NUMINAMATH_GPT_charity_amount_l1200_120041


namespace NUMINAMATH_GPT_maximum_value_l1200_120034

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x : ℝ) : ℝ := -Real.log x / x

theorem maximum_value (x1 x2 t : ℝ) (h1 : 0 < t) (h2 : f x1 = t) (h3 : g x2 = t) : 
  ∃ x1 x2, (t > 0) ∧ (f x1 = t) ∧ (g x2 = t) ∧ ((x1 / (x2 * Real.exp t)) = 1 / Real.exp 1) := 
sorry

end NUMINAMATH_GPT_maximum_value_l1200_120034


namespace NUMINAMATH_GPT_initial_money_in_wallet_l1200_120089

theorem initial_money_in_wallet (x : ℝ) 
  (h1 : x = 78 + 16) : 
  x = 94 :=
by
  sorry

end NUMINAMATH_GPT_initial_money_in_wallet_l1200_120089


namespace NUMINAMATH_GPT_inequality_three_integer_solutions_l1200_120080

theorem inequality_three_integer_solutions (c : ℤ) :
  (∃ s1 s2 s3 : ℤ, s1 < s2 ∧ s2 < s3 ∧ 
    (∀ x : ℤ, x^2 + c * x + 1 ≤ 0 ↔ x = s1 ∨ x = s2 ∨ x = s3)) ↔ (c = -4 ∨ c = 4) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_three_integer_solutions_l1200_120080


namespace NUMINAMATH_GPT_average_cost_parking_l1200_120016

theorem average_cost_parking :
  let cost_first_2_hours := 12.00
  let cost_per_additional_hour := 1.75
  let total_hours := 9
  let total_cost := cost_first_2_hours + cost_per_additional_hour * (total_hours - 2)
  let average_cost_per_hour := total_cost / total_hours
  average_cost_per_hour = 2.69 :=
by
  sorry

end NUMINAMATH_GPT_average_cost_parking_l1200_120016


namespace NUMINAMATH_GPT_sum_numbers_eq_432_l1200_120010

theorem sum_numbers_eq_432 (n : ℕ) (h : (n * (n + 1)) / 2 = 432) : n = 28 :=
sorry

end NUMINAMATH_GPT_sum_numbers_eq_432_l1200_120010


namespace NUMINAMATH_GPT_no_linear_term_implies_equal_l1200_120083

theorem no_linear_term_implies_equal (m n : ℝ) (h : (x : ℝ) → (x + m) * (x - n) - x^2 - (- mn) = 0) : m = n :=
by
  sorry

end NUMINAMATH_GPT_no_linear_term_implies_equal_l1200_120083


namespace NUMINAMATH_GPT_transform_equation_l1200_120040

open Real

theorem transform_equation (m : ℝ) (x : ℝ) (h1 : x^2 + 4 * x = m) (h2 : (x + 2)^2 = 5) : m = 1 := by
  sorry

end NUMINAMATH_GPT_transform_equation_l1200_120040


namespace NUMINAMATH_GPT_solution_n_value_l1200_120043

open BigOperators

noncomputable def problem_statement (a b n : ℝ) : Prop :=
  ∃ (A B : ℝ), A = Real.log a ∧ B = Real.log b ∧
    (7 * A + 15 * B) - (4 * A + 9 * B) = (11 * A + 20 * B) - (7 * A + 15 * B) ∧
    (4 + 135) * B = Real.log (b^n)

theorem solution_n_value (a b : ℝ) (h_pos : a > 0) (h_pos_b : b > 0) :
  problem_statement a b 139 :=
by
  sorry

end NUMINAMATH_GPT_solution_n_value_l1200_120043


namespace NUMINAMATH_GPT_probability_plane_contains_points_inside_octahedron_l1200_120024

noncomputable def enhanced_octahedron_probability : ℚ :=
  let total_vertices := 18
  let total_ways := Nat.choose total_vertices 3
  let faces := 8
  let triangles_per_face := 4
  let unfavorable_ways := faces * triangles_per_face
  total_ways - unfavorable_ways

theorem probability_plane_contains_points_inside_octahedron :
  enhanced_octahedron_probability / (816 : ℚ) = 49 / 51 :=
sorry

end NUMINAMATH_GPT_probability_plane_contains_points_inside_octahedron_l1200_120024


namespace NUMINAMATH_GPT_intersection_A_B_l1200_120002

def A := {x : ℝ | (x - 1) * (x - 4) < 0}
def B := {x : ℝ | x <= 2}

theorem intersection_A_B :
  A ∩ B = {x : ℝ | 1 < x ∧ x <= 2} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1200_120002


namespace NUMINAMATH_GPT_triangle_acute_angle_contradiction_l1200_120000

theorem triangle_acute_angle_contradiction
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_tri : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_at_most_one_acute : (α < 90 ∧ β ≥ 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β < 90 ∧ γ ≥ 90) 
                         ∨ (α ≥ 90 ∧ β ≥ 90 ∧ γ < 90)) :
  false :=
by
  sorry

end NUMINAMATH_GPT_triangle_acute_angle_contradiction_l1200_120000


namespace NUMINAMATH_GPT_proof_stage_constancy_l1200_120006

-- Definitions of stages
def Stage1 := "Fertilization and seed germination"
def Stage2 := "Flowering and pollination"
def Stage3 := "Meiosis and fertilization"
def Stage4 := "Formation of sperm and egg cells"

-- Question: Which stages maintain chromosome constancy and promote genetic recombination in plant life?
def Q := "Which stages maintain chromosome constancy and promote genetic recombination in plant life?"

-- Correct answer
def Answer := Stage3

-- Conditions
def s1 := Stage1
def s2 := Stage2
def s3 := Stage3
def s4 := Stage4

-- Theorem statement
theorem proof_stage_constancy : Q = Answer := by
  sorry

end NUMINAMATH_GPT_proof_stage_constancy_l1200_120006


namespace NUMINAMATH_GPT_total_fruits_in_baskets_l1200_120020

structure Baskets where
  mangoes : ℕ
  pears : ℕ
  pawpaws : ℕ
  kiwis : ℕ
  lemons : ℕ

def taniaBaskets : Baskets := {
  mangoes := 18,
  pears := 10,
  pawpaws := 12,
  kiwis := 9,
  lemons := 9
}

theorem total_fruits_in_baskets : taniaBaskets.mangoes + taniaBaskets.pears + taniaBaskets.pawpaws + taniaBaskets.kiwis + taniaBaskets.lemons = 58 :=
by
  sorry

end NUMINAMATH_GPT_total_fruits_in_baskets_l1200_120020


namespace NUMINAMATH_GPT_marbles_difference_l1200_120049

-- Conditions
def L : ℕ := 23
def F : ℕ := 9

-- Proof statement
theorem marbles_difference : L - F = 14 := by
  sorry

end NUMINAMATH_GPT_marbles_difference_l1200_120049


namespace NUMINAMATH_GPT_find_num_students_B_l1200_120084

-- Given conditions as definitions
def num_students_A : ℕ := 24
def avg_weight_A : ℚ := 40
def avg_weight_B : ℚ := 35
def avg_weight_class : ℚ := 38

-- The total weight for sections A and B
def total_weight_A : ℚ := num_students_A * avg_weight_A
def total_weight_B (x: ℕ) : ℚ := x * avg_weight_B

-- The number of students in section B
noncomputable def num_students_B : ℕ := 16

-- The proof problem: Prove that number of students in section B is 16
theorem find_num_students_B (x: ℕ) (h: (total_weight_A + total_weight_B x) / (num_students_A + x) = avg_weight_class) : 
  x = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_num_students_B_l1200_120084


namespace NUMINAMATH_GPT_tangent_intersection_product_l1200_120055

theorem tangent_intersection_product (R r : ℝ) (A B C : ℝ) :
  (AC * CB = R * r) :=
sorry

end NUMINAMATH_GPT_tangent_intersection_product_l1200_120055


namespace NUMINAMATH_GPT_inequality_proof_l1200_120077

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a + b + c + d + 8 / (a*b + b*c + c*d + d*a) ≥ 6 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1200_120077


namespace NUMINAMATH_GPT_find_other_root_l1200_120023

-- Definitions based on conditions
def quadratic_equation (k : ℝ) (x : ℝ) : Prop := x^2 + 2 * k * x + k - 1 = 0

def is_root (k : ℝ) (x : ℝ) : Prop := quadratic_equation k x = true

-- The theorem to prove
theorem find_other_root (k x t: ℝ) (h₁ : is_root k 0) : t = -2 :=
sorry

end NUMINAMATH_GPT_find_other_root_l1200_120023


namespace NUMINAMATH_GPT_sasha_remainder_l1200_120003

theorem sasha_remainder (n a b c d : ℕ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d)
  (h3 : d = 20 - a) (h4 : 0 ≤ b ∧ b ≤ 101) : b = 20 :=
by
  sorry

end NUMINAMATH_GPT_sasha_remainder_l1200_120003


namespace NUMINAMATH_GPT_royal_family_children_l1200_120013

theorem royal_family_children :
  ∃ (d : ℕ), (d + 3 ≤ 20) ∧ (d ≥ 1) ∧ (∃ (n : ℕ), 70 + 2 * n = 35 + (d + 3) * n) ∧ (d + 3 = 7 ∨ d + 3 = 9) :=
by
  sorry

end NUMINAMATH_GPT_royal_family_children_l1200_120013


namespace NUMINAMATH_GPT_find_y_intercept_l1200_120061

theorem find_y_intercept (m x y b : ℤ) (h_slope : m = 2) (h_point : (x, y) = (259, 520)) :
  y = m * x + b → b = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_y_intercept_l1200_120061


namespace NUMINAMATH_GPT_integer_solutions_for_xyz_eq_4_l1200_120087

theorem integer_solutions_for_xyz_eq_4 :
  {n : ℕ // n = 48} :=
sorry

end NUMINAMATH_GPT_integer_solutions_for_xyz_eq_4_l1200_120087


namespace NUMINAMATH_GPT_ellipse_equation_correct_coordinates_c_correct_l1200_120038

-- Definition of the ellipse Γ with given properties
def ellipse_properties (a b : ℝ) (ecc : ℝ) (c_len : ℝ) :=
  a > b ∧ b > 0 ∧ ecc = (Real.sqrt 2) / 2 ∧ c_len = Real.sqrt 2

-- Correct answer for the equation of the ellipse
def correct_ellipse_equation := ∀ x y : ℝ, (x^2) / 2 + y^2 = 1

-- Proving that given the properties of the ellipse, the equation is as stated
theorem ellipse_equation_correct (a b : ℝ) (h : ellipse_properties a b (Real.sqrt 2 / 2) (Real.sqrt 2)) :
  (x^2) / 2 + y^2 = 1 := 
  sorry

-- Definition of the conditions for points A, B, and C
def triangle_conditions (a b : ℝ) (area : ℝ) :=
  ∀ A B : ℝ × ℝ,
    A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
    B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
    area = 3 * Real.sqrt 6 / 4

-- Correct coordinates of point C given the conditions
def correct_coordinates_c (C : ℝ × ℝ) :=
  (C = (1, Real.sqrt 2 / 2) ∨ C = (2, 1))

-- Proving that given the conditions, the coordinates of point C are correct
theorem coordinates_c_correct (a b : ℝ) (h : triangle_conditions a b (3 * Real.sqrt 6 / 4)) (C : ℝ × ℝ) :
  correct_coordinates_c C :=
  sorry

end NUMINAMATH_GPT_ellipse_equation_correct_coordinates_c_correct_l1200_120038


namespace NUMINAMATH_GPT_sum_factors_of_30_l1200_120030

theorem sum_factors_of_30 : (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 :=
by
  sorry

end NUMINAMATH_GPT_sum_factors_of_30_l1200_120030


namespace NUMINAMATH_GPT_real_numbers_int_approximation_l1200_120096

theorem real_numbers_int_approximation:
  ∀ (x y : ℝ), ∃ (m n : ℤ),
  (x - m) ^ 2 + (y - n) * (x - m) + (y - n) ^ 2 ≤ (1 / 3) :=
by
  intros x y
  sorry

end NUMINAMATH_GPT_real_numbers_int_approximation_l1200_120096


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l1200_120057

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + m - 1 = 0) ∧ (x2^2 - 4*x2 + m - 1 = 0)) → m < 5 := sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l1200_120057


namespace NUMINAMATH_GPT_harold_savings_l1200_120031

theorem harold_savings :
  let income_primary := 2500
  let income_freelance := 500
  let rent := 700
  let car_payment := 300
  let car_insurance := 125
  let electricity := 0.25 * car_payment
  let water := 0.15 * rent
  let internet := 75
  let groceries := 200
  let miscellaneous := 150
  let total_income := income_primary + income_freelance
  let total_expenses := rent + car_payment + car_insurance + electricity + water + internet + groceries + miscellaneous
  let amount_before_savings := total_income - total_expenses
  let retirement := (1/3) * amount_before_savings
  let emergency := (1/3) * amount_before_savings
  let amount_after_savings := amount_before_savings - retirement - emergency
  amount_after_savings = 423.34 := 
sorry

end NUMINAMATH_GPT_harold_savings_l1200_120031


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1200_120081

theorem problem1 : 999 * 999 + 1999 = 1000000 := by
  sorry

theorem problem2 : 9 * 72 * 125 = 81000 := by
  sorry

theorem problem3 : 416 - 327 + 184 - 273 = 0 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1200_120081


namespace NUMINAMATH_GPT_division_of_negatives_example_div_l1200_120063

theorem division_of_negatives (a b : Int) (ha : a < 0) (hb : b < 0) (hb_neq : b ≠ 0) : 
  (-a) / (-b) = a / b :=
by sorry

theorem example_div : (-300) / (-50) = 6 :=
by
  apply division_of_negatives
  repeat { sorry }

end NUMINAMATH_GPT_division_of_negatives_example_div_l1200_120063


namespace NUMINAMATH_GPT_third_square_is_G_l1200_120011

-- Conditions
-- Define eight 2x2 squares, where the last placed square is E
def squares : List String := ["F", "H", "G", "D", "A", "B", "C", "E"]

-- Let the third square be G
def third_square := "G"

-- Proof statement
theorem third_square_is_G : squares.get! 2 = third_square :=
by
  sorry

end NUMINAMATH_GPT_third_square_is_G_l1200_120011


namespace NUMINAMATH_GPT_least_possible_sum_l1200_120029

theorem least_possible_sum {c d : ℕ} (hc : c ≥ 2) (hd : d ≥ 2) (h : 3 * c + 6 = 6 * d + 3) : c + d = 5 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_sum_l1200_120029


namespace NUMINAMATH_GPT_problem_1_l1200_120004

theorem problem_1 (m : ℝ) : (¬ ∃ x : ℝ, x^2 + 2 * x + m ≤ 0) ↔ m > 1 := sorry

end NUMINAMATH_GPT_problem_1_l1200_120004


namespace NUMINAMATH_GPT_units_digit_of_power_17_l1200_120067

theorem units_digit_of_power_17 (n : ℕ) (k : ℕ) (h_n4 : n % 4 = 3) : (17^n) % 10 = 3 :=
  by
  -- Since units digits of powers repeat every 4
  sorry

-- Specific problem instance
example : (17^1995) % 10 = 3 := units_digit_of_power_17 1995 17 (by norm_num)

end NUMINAMATH_GPT_units_digit_of_power_17_l1200_120067


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1200_120045

-- Define the parameters a^2 and b^2 according to the problem
def a_sq : ℝ := 25
def b_sq : ℝ := 16

-- State the problem
theorem distance_between_foci_of_ellipse : 
  (2 * Real.sqrt (a_sq - b_sq)) = 6 := by
  -- Proof content is skipped 
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l1200_120045


namespace NUMINAMATH_GPT_vessel_base_length_l1200_120079

noncomputable def volume_of_cube (side: ℝ) : ℝ :=
  side ^ 3

noncomputable def volume_displaced (length breadth height: ℝ) : ℝ :=
  length * breadth * height

theorem vessel_base_length
  (breadth : ℝ) 
  (cube_edge : ℝ)
  (water_rise : ℝ)
  (displaced_volume : ℝ) 
  (h1 : breadth = 30) 
  (h2 : cube_edge = 30) 
  (h3 : water_rise = 15) 
  (h4 : volume_of_cube cube_edge = displaced_volume) :
  volume_displaced (displaced_volume / (breadth * water_rise)) breadth water_rise = displaced_volume :=
  by
  sorry

end NUMINAMATH_GPT_vessel_base_length_l1200_120079


namespace NUMINAMATH_GPT_price_per_package_l1200_120005

theorem price_per_package (P : ℝ) (hp1 : 10 * P + 50 * (4 / 5 * P) = 1096) :
  P = 21.92 :=
by 
  sorry

end NUMINAMATH_GPT_price_per_package_l1200_120005


namespace NUMINAMATH_GPT_rahul_work_days_l1200_120047

theorem rahul_work_days
  (R : ℕ)
  (Rajesh_days : ℕ := 2)
  (total_payment : ℕ := 170)
  (rahul_share : ℕ := 68)
  (combined_work_rate : ℚ := 1) :
  (∃ R : ℕ, (1 / (R : ℚ) + 1 / (Rajesh_days : ℚ) = combined_work_rate) ∧ (68 / (total_payment - rahul_share) = 2 / R) ∧ R = 3) :=
sorry

end NUMINAMATH_GPT_rahul_work_days_l1200_120047


namespace NUMINAMATH_GPT_fescue_in_Y_l1200_120099

-- Define the weight proportions of the mixtures
def weight_X : ℝ := 0.6667
def weight_Y : ℝ := 0.3333

-- Define the proportion of ryegrass in each mixture
def ryegrass_X : ℝ := 0.40
def ryegrass_Y : ℝ := 0.25

-- Define the proportion of ryegrass in the final mixture
def ryegrass_final : ℝ := 0.35

-- Define the proportion of ryegrass contributed by X and Y to the final mixture
def contrib_X : ℝ := weight_X * ryegrass_X
def contrib_Y : ℝ := weight_Y * ryegrass_Y

-- Define the total proportion of ryegrass in the final mixture
def total_ryegrass : ℝ := contrib_X + contrib_Y

-- The lean theorem stating that the percentage of fescue in Y equals 75%
theorem fescue_in_Y :
  total_ryegrass = ryegrass_final →
  (100 - (ryegrass_Y * 100)) = 75 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_fescue_in_Y_l1200_120099


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1200_120001

theorem quadratic_inequality_solution :
  {x : ℝ | (x^2 - 50 * x + 576) ≤ 16} = {x : ℝ | 20 ≤ x ∧ x ≤ 28} :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1200_120001


namespace NUMINAMATH_GPT_exists_q_r_polynomials_l1200_120097

theorem exists_q_r_polynomials (n : ℕ) (p : Polynomial ℝ) 
  (h_deg : p.degree = n) 
  (h_monic : p.leadingCoeff = 1) :
  ∃ q r : Polynomial ℝ, 
    q.degree = n ∧ r.degree = n ∧ 
    (∀ x : ℝ, q.eval x = 0 → r.eval x = 0) ∧
    (∀ y : ℝ, r.eval y = 0 → q.eval y = 0) ∧
    q.leadingCoeff = 1 ∧ r.leadingCoeff = 1 ∧ 
    p = (q + r) / 2 := 
sorry

end NUMINAMATH_GPT_exists_q_r_polynomials_l1200_120097


namespace NUMINAMATH_GPT_arithmetic_geometric_relation_l1200_120017

variable (a₁ a₂ b₁ b₂ b₃ : ℝ)

-- Conditions
def is_arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ (d : ℝ), -2 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -8

def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ (r : ℝ), -2 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -8

-- The problem statement
theorem arithmetic_geometric_relation (h₁ : is_arithmetic_sequence a₁ a₂) (h₂ : is_geometric_sequence b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1 / 2 := by
    sorry

end NUMINAMATH_GPT_arithmetic_geometric_relation_l1200_120017


namespace NUMINAMATH_GPT_intersecting_lines_l1200_120048

theorem intersecting_lines (c d : ℝ) 
  (h1 : 3 = (1/3 : ℝ) * 0 + c)
  (h2 : 0 = (1/3 : ℝ) * 3 + d) :
  c + d = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_intersecting_lines_l1200_120048


namespace NUMINAMATH_GPT_total_area_of_house_is_2300_l1200_120019

-- Definitions based on the conditions in the problem
def area_living_room_dining_room_kitchen : ℕ := 1000
def area_master_bedroom_suite : ℕ := 1040
def area_guest_bedroom : ℕ := area_master_bedroom_suite / 4

-- Theorem to state the total area of the house
theorem total_area_of_house_is_2300 :
  area_living_room_dining_room_kitchen + area_master_bedroom_suite + area_guest_bedroom = 2300 :=
by
  sorry

end NUMINAMATH_GPT_total_area_of_house_is_2300_l1200_120019


namespace NUMINAMATH_GPT_emily_small_gardens_count_l1200_120046

-- Definitions based on conditions
def initial_seeds : ℕ := 41
def seeds_planted_in_big_garden : ℕ := 29
def seeds_per_small_garden : ℕ := 4

-- Theorem statement
theorem emily_small_gardens_count (initial_seeds seeds_planted_in_big_garden seeds_per_small_garden : ℕ) :
  initial_seeds = 41 →
  seeds_planted_in_big_garden = 29 →
  seeds_per_small_garden = 4 →
  (initial_seeds - seeds_planted_in_big_garden) / seeds_per_small_garden = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_emily_small_gardens_count_l1200_120046


namespace NUMINAMATH_GPT_tangent_line_to_circle_l1200_120050

theorem tangent_line_to_circle {c : ℝ} (h : c > 0) :
  (∀ x y : ℝ, x^2 + y^2 = 8 → x + y = c) ↔ c = 4 := sorry

end NUMINAMATH_GPT_tangent_line_to_circle_l1200_120050


namespace NUMINAMATH_GPT_quadratic_is_perfect_square_l1200_120068

theorem quadratic_is_perfect_square (c : ℝ) :
  (∃ b : ℝ, (3 * (x : ℝ) + b)^2 = 9 * x^2 - 24 * x + c) ↔ c = 16 :=
by sorry

end NUMINAMATH_GPT_quadratic_is_perfect_square_l1200_120068


namespace NUMINAMATH_GPT_cos_double_angle_l1200_120095

-- Define the hypothesis
def cos_alpha (α : ℝ) : Prop := Real.cos α = 1 / 2

-- State the theorem
theorem cos_double_angle (α : ℝ) (h : cos_alpha α) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1200_120095


namespace NUMINAMATH_GPT_replaced_person_weight_l1200_120070

theorem replaced_person_weight :
  ∀ (old_avg_weight new_person_weight incr_weight : ℕ),
    old_avg_weight * 8 + incr_weight = new_person_weight →
    incr_weight = 16 →
    new_person_weight = 81 →
    (old_avg_weight - (new_person_weight - incr_weight) / 8) = 65 :=
by
  intros old_avg_weight new_person_weight incr_weight h1 h2 h3
  -- TODO: Proof goes here
  sorry

end NUMINAMATH_GPT_replaced_person_weight_l1200_120070


namespace NUMINAMATH_GPT_percentage_of_volume_occupied_l1200_120044

-- Define the dimensions of the block
def block_length : ℕ := 9
def block_width : ℕ := 7
def block_height : ℕ := 12

-- Define the dimension of the cube
def cube_side : ℕ := 4

-- Define the volumes
def block_volume : ℕ := block_length * block_width * block_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the count of cubes along each dimension
def cubes_along_length : ℕ := block_length / cube_side
def cubes_along_width : ℕ := block_width / cube_side
def cubes_along_height : ℕ := block_height / cube_side

-- Define the total number of cubes that fit into the block
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height

-- Define the total volume occupied by the cubes
def occupied_volume : ℕ := total_cubes * cube_volume

-- Define the percentage of the block's volume occupied by the cubes (as a float for precision)
def volume_percentage : Float := (Float.ofNat occupied_volume / Float.ofNat block_volume) * 100

-- Statement to prove
theorem percentage_of_volume_occupied :
  volume_percentage = 50.79 := by
  sorry

end NUMINAMATH_GPT_percentage_of_volume_occupied_l1200_120044


namespace NUMINAMATH_GPT_neg_four_fifth_less_neg_two_third_l1200_120074

theorem neg_four_fifth_less_neg_two_third : (-4 : ℚ) / 5 < (-2 : ℚ) / 3 :=
  sorry

end NUMINAMATH_GPT_neg_four_fifth_less_neg_two_third_l1200_120074


namespace NUMINAMATH_GPT_space_shuttle_new_orbital_speed_l1200_120028

noncomputable def new_orbital_speed (v_1 : ℝ) (delta_v : ℝ) : ℝ :=
  let v_new := v_1 + delta_v
  v_new * 3600

theorem space_shuttle_new_orbital_speed : 
  new_orbital_speed 2 (500 / 1000) = 9000 :=
by 
  sorry

end NUMINAMATH_GPT_space_shuttle_new_orbital_speed_l1200_120028


namespace NUMINAMATH_GPT_total_call_charges_l1200_120022

-- Definitions based on conditions
def base_fee : ℝ := 39
def included_minutes : ℕ := 300
def excess_charge_per_minute : ℝ := 0.19

-- Given variables
variable (x : ℕ) -- excess minutes
variable (y : ℝ) -- total call charges

-- Theorem stating the relationship between y and x
theorem total_call_charges (h : x > 0) : y = 0.19 * x + 39 := 
by sorry

end NUMINAMATH_GPT_total_call_charges_l1200_120022


namespace NUMINAMATH_GPT_squares_on_grid_l1200_120090

-- Defining the problem conditions
def grid_size : ℕ := 5
def total_points : ℕ := grid_size * grid_size
def used_points : ℕ := 20

-- Stating the theorem to prove the total number of squares formed
theorem squares_on_grid : 
  (total_points = 25) ∧ (used_points = 20) →
  (∃ all_squares : ℕ, all_squares = 21) :=
by
  intros
  sorry

end NUMINAMATH_GPT_squares_on_grid_l1200_120090


namespace NUMINAMATH_GPT_gcd_lcm_product_l1200_120062

theorem gcd_lcm_product (a b: ℕ) (h1 : a = 36) (h2 : b = 210) :
  Nat.gcd a b * Nat.lcm a b = 7560 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l1200_120062


namespace NUMINAMATH_GPT_rubert_james_ratio_l1200_120098

-- Definitions and conditions from a)
def adam_candies : ℕ := 6
def james_candies : ℕ := 3 * adam_candies
def rubert_candies (total_candies : ℕ) : ℕ := total_candies - (adam_candies + james_candies)
def total_candies : ℕ := 96

-- Statement to prove the ratio
theorem rubert_james_ratio : 
  (rubert_candies total_candies) / james_candies = 4 :=
by
  -- Proof is not required, so we leave it as sorry.
  sorry

end NUMINAMATH_GPT_rubert_james_ratio_l1200_120098


namespace NUMINAMATH_GPT_minimum_weighings_for_counterfeit_coin_l1200_120042

/-- Given 9 coins, where 8 have equal weight and 1 is heavier (the counterfeit coin), prove that the 
minimum number of weighings required on a balance scale without weights to find the counterfeit coin is 2. -/
theorem minimum_weighings_for_counterfeit_coin (n : ℕ) (coins : Fin n → ℝ) 
  (h_n : n = 9) 
  (h_real : ∃ w : ℝ, ∀ i : Fin n, i.val < 8 → coins i = w) 
  (h_counterfeit : ∃ i : Fin n, ∀ j : Fin n, j ≠ i → coins i > coins j) : 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_weighings_for_counterfeit_coin_l1200_120042
