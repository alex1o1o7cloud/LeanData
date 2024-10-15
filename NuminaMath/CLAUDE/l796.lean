import Mathlib

namespace NUMINAMATH_CALUDE_log_product_equals_five_thirds_l796_79664

theorem log_product_equals_five_thirds :
  Real.log 9 / Real.log 8 * (Real.log 32 / Real.log 9) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_five_thirds_l796_79664


namespace NUMINAMATH_CALUDE_f_min_value_l796_79672

/-- The function f(x, y) as defined in the problem -/
def f (x y : ℝ) : ℝ := x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

/-- Theorem stating that the minimum value of f(x, y) is 3 -/
theorem f_min_value :
  ∀ x y : ℝ, f x y ≥ 3 := by sorry

end NUMINAMATH_CALUDE_f_min_value_l796_79672


namespace NUMINAMATH_CALUDE_wayne_shrimp_guests_l796_79639

/-- Given Wayne's shrimp appetizer scenario, prove the number of guests he can serve. -/
theorem wayne_shrimp_guests :
  ∀ (shrimp_per_guest : ℕ) 
    (cost_per_pound : ℚ) 
    (shrimp_per_pound : ℕ) 
    (total_spent : ℚ),
  shrimp_per_guest = 5 →
  cost_per_pound = 17 →
  shrimp_per_pound = 20 →
  total_spent = 170 →
  (total_spent / cost_per_pound * shrimp_per_pound) / shrimp_per_guest = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_wayne_shrimp_guests_l796_79639


namespace NUMINAMATH_CALUDE_fox_weasel_hunting_average_l796_79636

/-- Proves that given the initial conditions and the number of animals remaining after 3 weeks,
    the average number of weasels caught by each fox per week is 4. -/
theorem fox_weasel_hunting_average :
  let initial_weasels : ℕ := 100
  let initial_rabbits : ℕ := 50
  let num_foxes : ℕ := 3
  let rabbits_per_fox_per_week : ℕ := 2
  let weeks : ℕ := 3
  let animals_left : ℕ := 96
  let total_animals_caught := initial_weasels + initial_rabbits - animals_left
  let total_rabbits_caught := num_foxes * rabbits_per_fox_per_week * weeks
  let total_weasels_caught := total_animals_caught - total_rabbits_caught
  let weasels_per_fox := total_weasels_caught / num_foxes
  let avg_weasels_per_fox_per_week := weasels_per_fox / weeks
  avg_weasels_per_fox_per_week = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_fox_weasel_hunting_average_l796_79636


namespace NUMINAMATH_CALUDE_printing_completion_time_l796_79670

def start_time : ℕ := 9 * 60  -- 9:00 AM in minutes since midnight
def quarter_completion_time : ℕ := 12 * 60 + 30  -- 12:30 PM in minutes since midnight

-- Time taken to complete one-fourth of the job in minutes
def quarter_job_duration : ℕ := quarter_completion_time - start_time

-- Total time required to complete the entire job in minutes
def total_job_duration : ℕ := 4 * quarter_job_duration

-- Completion time in minutes since midnight
def completion_time : ℕ := start_time + total_job_duration

theorem printing_completion_time :
  completion_time = 23 * 60 := by sorry

end NUMINAMATH_CALUDE_printing_completion_time_l796_79670


namespace NUMINAMATH_CALUDE_stating_fifteenth_term_is_43_l796_79630

/-- 
Given an arithmetic sequence where:
- a₁ is the first term
- d is the common difference
- n is the term number
This function calculates the nth term of the sequence.
-/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- 
Theorem stating that the 15th term of the arithmetic sequence
with first term 1 and common difference 3 is 43.
-/
theorem fifteenth_term_is_43 : 
  arithmeticSequenceTerm 1 3 15 = 43 := by
sorry

end NUMINAMATH_CALUDE_stating_fifteenth_term_is_43_l796_79630


namespace NUMINAMATH_CALUDE_count_ordered_pairs_eq_18_l796_79699

/-- Given that 1372 = 2^2 * 7^2 * 11, this function returns the number of ordered pairs of positive integers (x, y) satisfying x * y = 1372. -/
def count_ordered_pairs : ℕ :=
  let prime_factorization : List (ℕ × ℕ) := [(2, 2), (7, 2), (11, 1)]
  (prime_factorization.map (λ (p, e) => e + 1)).prod

/-- The number of ordered pairs of positive integers (x, y) satisfying x * y = 1372 is 18. -/
theorem count_ordered_pairs_eq_18 : count_ordered_pairs = 18 := by
  sorry

end NUMINAMATH_CALUDE_count_ordered_pairs_eq_18_l796_79699


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l796_79673

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l796_79673


namespace NUMINAMATH_CALUDE_inequality_infimum_l796_79626

theorem inequality_infimum (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), x ∈ Set.Icc 0 1 → x^3 - m ≤ a*x + b ∧ a*x + b ≤ x^3 + m) →
  m ≥ Real.sqrt 3 / 9 :=
sorry

end NUMINAMATH_CALUDE_inequality_infimum_l796_79626


namespace NUMINAMATH_CALUDE_rabbit_population_growth_l796_79613

theorem rabbit_population_growth (initial_rabbits new_rabbits : ℕ) 
  (h1 : initial_rabbits = 8) 
  (h2 : new_rabbits = 5) : 
  initial_rabbits + new_rabbits = 13 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_population_growth_l796_79613


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_side_lengths_l796_79679

theorem rectangular_parallelepiped_side_lengths 
  (x y z : ℝ) 
  (sum_eq : x + y + z = 17)
  (area_eq : 2*x*y + 2*y*z + 2*z*x = 180)
  (sq_sum_eq : x^2 + y^2 = 100) :
  x = 8 ∧ y = 6 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_side_lengths_l796_79679


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l796_79600

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 789 [ZMOD 11]) → n = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l796_79600


namespace NUMINAMATH_CALUDE_greatest_divisor_l796_79623

theorem greatest_divisor (G : ℕ) : G = 127 ↔ 
  G > 0 ∧ 
  (∀ n : ℕ, n > G → ¬(1657 % n = 6 ∧ 2037 % n = 5)) ∧
  1657 % G = 6 ∧ 
  2037 % G = 5 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_l796_79623


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_range_l796_79660

/-- Given three numbers forming a geometric sequence with sum m, prove the range of the middle term -/
theorem geometric_sequence_middle_term_range (a b c m : ℝ) : 
  (∃ r : ℝ, a * r = b ∧ b * r = c) →  -- a, b, c form a geometric sequence
  (a + b + c = m) →                   -- sum of terms is m
  (m > 0) →                           -- m is positive
  (b ∈ Set.Icc (-m) 0 ∪ Set.Ioc 0 (m/3)) :=  -- range of b
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_range_l796_79660


namespace NUMINAMATH_CALUDE_jonathan_daily_burn_l796_79697

-- Define Jonathan's daily calorie intake
def daily_intake : ℕ := 2500

-- Define Jonathan's extra calorie intake on Saturday
def saturday_extra : ℕ := 1000

-- Define Jonathan's weekly caloric deficit
def weekly_deficit : ℕ := 2500

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Theorem to prove Jonathan's daily calorie burn
theorem jonathan_daily_burn :
  (6 * daily_intake + (daily_intake + saturday_extra) + weekly_deficit) / days_in_week = 3000 :=
by sorry

end NUMINAMATH_CALUDE_jonathan_daily_burn_l796_79697


namespace NUMINAMATH_CALUDE_jim_marathon_training_l796_79662

theorem jim_marathon_training (days_phase1 days_phase2 days_phase3 : ℕ)
  (miles_per_day_phase1 miles_per_day_phase2 miles_per_day_phase3 : ℕ)
  (h1 : days_phase1 = 30)
  (h2 : days_phase2 = 30)
  (h3 : days_phase3 = 30)
  (h4 : miles_per_day_phase1 = 5)
  (h5 : miles_per_day_phase2 = 10)
  (h6 : miles_per_day_phase3 = 20) :
  days_phase1 * miles_per_day_phase1 +
  days_phase2 * miles_per_day_phase2 +
  days_phase3 * miles_per_day_phase3 = 1050 := by
  sorry

end NUMINAMATH_CALUDE_jim_marathon_training_l796_79662


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l796_79687

/-- The discriminant of a quadratic equation ax² + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  let a : ℝ := 5
  let b : ℝ := 2
  let c : ℝ := -8
  discriminant a b c = 164 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l796_79687


namespace NUMINAMATH_CALUDE_one_four_digit_perfect_square_palindrome_l796_79611

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem one_four_digit_perfect_square_palindrome :
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_one_four_digit_perfect_square_palindrome_l796_79611


namespace NUMINAMATH_CALUDE_inverse_square_problem_l796_79696

-- Define the relationship between x and y
def inverse_square_relation (x y : ℝ) : Prop :=
  ∃ k : ℝ, x = k / (y * y) ∧ k ≠ 0

-- State the theorem
theorem inverse_square_problem (x₁ x₂ y₁ y₂ : ℝ) :
  inverse_square_relation x₁ y₁ →
  inverse_square_relation x₂ y₂ →
  x₁ = 1 →
  y₁ = 3 →
  y₂ = 6 →
  x₂ = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_problem_l796_79696


namespace NUMINAMATH_CALUDE_sum_problem_l796_79650

/-- The sum of 38 and twice a number is a certain value. The number is 43. -/
theorem sum_problem (x : ℕ) (h : x = 43) : 38 + 2 * x = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_problem_l796_79650


namespace NUMINAMATH_CALUDE_absolute_value_negative_2023_l796_79627

theorem absolute_value_negative_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_negative_2023_l796_79627


namespace NUMINAMATH_CALUDE_shortest_translation_distance_line_to_circle_l796_79685

/-- The shortest distance to translate a line to become tangent to a circle -/
theorem shortest_translation_distance_line_to_circle 
  (line : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) 
  (h_line : ∀ x y, line x y ↔ x - y + 1 = 0)
  (h_circle : ∀ x y, circle x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :
  ∃ d : ℝ, d = Real.sqrt 2 - 1 ∧ 
    (∀ d' : ℝ, d' ≥ 0 → 
      (∃ c : ℝ, ∀ x y, (x - y + c = 0 → circle x y) → d' ≥ d)) :=
sorry

end NUMINAMATH_CALUDE_shortest_translation_distance_line_to_circle_l796_79685


namespace NUMINAMATH_CALUDE_walter_seal_time_l796_79631

/-- The time Walter spends at the zoo -/
def total_time : ℕ := 260

/-- Walter's initial time spent looking at seals -/
def initial_seal_time : ℕ := 20

/-- Time spent looking at penguins -/
def penguin_time (s : ℕ) : ℕ := 8 * s

/-- Time spent looking at elephants -/
def elephant_time : ℕ := 13

/-- Time spent on second visit to seals -/
def second_seal_time (s : ℕ) : ℕ := s / 2

/-- Time spent at giraffe exhibit -/
def giraffe_time (s : ℕ) : ℕ := 3 * s

/-- Total time spent looking at seals -/
def total_seal_time (s : ℕ) : ℕ := s + (s / 2)

theorem walter_seal_time :
  total_seal_time initial_seal_time = 30 ∧
  initial_seal_time + penguin_time initial_seal_time + elephant_time +
  second_seal_time initial_seal_time + giraffe_time initial_seal_time = total_time :=
sorry

end NUMINAMATH_CALUDE_walter_seal_time_l796_79631


namespace NUMINAMATH_CALUDE_water_bottles_needed_l796_79633

/-- Calculates the total number of water bottles needed for a family road trip. -/
theorem water_bottles_needed
  (family_size : ℕ)
  (travel_time : ℕ)
  (water_consumption : ℚ)
  (h1 : family_size = 4)
  (h2 : travel_time = 16)
  (h3 : water_consumption = 1/2) :
  ↑family_size * ↑travel_time * water_consumption = 32 :=
by sorry

end NUMINAMATH_CALUDE_water_bottles_needed_l796_79633


namespace NUMINAMATH_CALUDE_gum_distribution_l796_79668

theorem gum_distribution (cousins : ℕ) (total_gum : ℕ) (gum_per_cousin : ℕ) :
  cousins = 4 →
  total_gum = 20 →
  total_gum = cousins * gum_per_cousin →
  gum_per_cousin = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_gum_distribution_l796_79668


namespace NUMINAMATH_CALUDE_remaining_cube_height_l796_79683

/-- The height of the remaining portion of a cube after cutting off a corner -/
theorem remaining_cube_height (cube_side : Real) (cut_distance : Real) : 
  cube_side = 2 → 
  cut_distance = 1 → 
  (cube_side - (Real.sqrt 3) / 3) = (5 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_remaining_cube_height_l796_79683


namespace NUMINAMATH_CALUDE_rug_area_is_48_l796_79634

/-- Calculates the area of a rug with specific dimensions -/
def rugArea (rect_length rect_width para_base para_height : ℝ) : ℝ :=
  let rect_area := rect_length * rect_width
  let para_area := para_base * para_height
  rect_area + 2 * para_area

/-- Theorem stating that a rug with given dimensions has an area of 48 square meters -/
theorem rug_area_is_48 :
  rugArea 6 4 3 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_rug_area_is_48_l796_79634


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l796_79669

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) 
  (bicycle_speed : ℝ) (foot_distance : ℝ) 
  (h1 : total_distance = 61) 
  (h2 : total_time = 9)
  (h3 : bicycle_speed = 9)
  (h4 : foot_distance = 16) :
  ∃ (foot_speed : ℝ), 
    foot_speed = 4 ∧ 
    foot_distance / foot_speed + (total_distance - foot_distance) / bicycle_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l796_79669


namespace NUMINAMATH_CALUDE_d_investment_is_250_l796_79680

/-- Represents the business investment scenario -/
structure BusinessInvestment where
  c_investment : ℕ
  total_profit : ℕ
  d_profit_share : ℕ

/-- Calculates D's investment based on the given conditions -/
def calculate_d_investment (b : BusinessInvestment) : ℕ :=
  b.c_investment * b.d_profit_share / (b.total_profit - b.d_profit_share)

/-- Theorem stating that D's investment is 250 given the conditions -/
theorem d_investment_is_250 (b : BusinessInvestment) 
  (h1 : b.c_investment = 1000)
  (h2 : b.total_profit = 500)
  (h3 : b.d_profit_share = 100) : 
  calculate_d_investment b = 250 := by
  sorry

#eval calculate_d_investment { c_investment := 1000, total_profit := 500, d_profit_share := 100 }

end NUMINAMATH_CALUDE_d_investment_is_250_l796_79680


namespace NUMINAMATH_CALUDE_sector_area_l796_79681

theorem sector_area (arc_length : Real) (central_angle : Real) (area : Real) :
  arc_length = 4 * Real.pi →
  central_angle = Real.pi / 3 →
  area = 24 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sector_area_l796_79681


namespace NUMINAMATH_CALUDE_fifteen_multiple_and_divisor_of_itself_l796_79646

theorem fifteen_multiple_and_divisor_of_itself : 
  ∃ n : ℕ, n % 15 = 0 ∧ 15 % n = 0 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_multiple_and_divisor_of_itself_l796_79646


namespace NUMINAMATH_CALUDE_intersection_points_theorem_pair_c_not_solution_pair_a_solution_pair_b_solution_pair_d_solution_pair_e_solution_l796_79688

theorem intersection_points_theorem : 
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x = 2 ∨ x = 4) :=
by sorry

theorem pair_c_not_solution :
  ¬∃ (x : ℝ), (x - 2 = x - 4) ∧ (x^2 - 6*x + 8 = 0) :=
by sorry

theorem pair_a_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x + 8 = 0 ∧ 0 = 0) :=
by sorry

theorem pair_b_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x = 8) :=
by sorry

theorem pair_d_solution :
  ∀ (x : ℝ), (x^2 - 6*x + 8 = 0) ↔ (x^2 - 6*x + 9 = 1) :=
by sorry

theorem pair_e_solution :
  ∃ (x : ℝ), (x^2 - 5 = 6*x - 8) ∧ (x^2 - 6*x + 8 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_pair_c_not_solution_pair_a_solution_pair_b_solution_pair_d_solution_pair_e_solution_l796_79688


namespace NUMINAMATH_CALUDE_triangle_acute_iff_sum_squares_gt_8R_squared_l796_79682

theorem triangle_acute_iff_sum_squares_gt_8R_squared 
  (a b c R : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_circumradius : R > 0)
  (h_R_def : 4 * R * (R - a) * (R - b) * (R - c) = a * b * c) :
  (∀ (A B C : ℝ), A + B + C = π → 
    0 < A ∧ A < π/2 ∧ 
    0 < B ∧ B < π/2 ∧ 
    0 < C ∧ C < π/2) ↔ 
  a^2 + b^2 + c^2 > 8 * R^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_acute_iff_sum_squares_gt_8R_squared_l796_79682


namespace NUMINAMATH_CALUDE_bank_deposit_exceeds_500_first_day_exceeding_500_l796_79612

def bank_deposit (n : ℕ) : ℚ :=
  3 * (3^n - 1) / 2

theorem bank_deposit_exceeds_500 :
  ∃ n : ℕ, bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500 :=
by
  sorry

theorem first_day_exceeding_500 :
  (∃ n : ℕ, bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500) →
  (∃ n : ℕ, n = 6 ∧ bank_deposit n > 500 ∧ ∀ m : ℕ, m < n → bank_deposit m ≤ 500) :=
by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_exceeds_500_first_day_exceeding_500_l796_79612


namespace NUMINAMATH_CALUDE_quadrilateral_side_difference_l796_79658

theorem quadrilateral_side_difference (a b c d : ℝ) (h1 : a + b + c + d = 120) 
  (h2 : a + c = 50) (h3 : a^2 + c^2 = 40^2) : |b - d| = 2 * Real.sqrt 775 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_side_difference_l796_79658


namespace NUMINAMATH_CALUDE_steve_pie_ratio_l796_79637

/-- Steve's weekly pie baking schedule -/
structure PieSchedule where
  monday_apple : ℕ
  monday_blueberry : ℕ
  tuesday_cherry : ℕ
  tuesday_blueberry : ℕ
  wednesday_apple : ℕ
  wednesday_blueberry : ℕ
  thursday_cherry : ℕ
  thursday_blueberry : ℕ
  friday_apple : ℕ
  friday_blueberry : ℕ
  saturday_apple : ℕ
  saturday_cherry : ℕ
  saturday_blueberry : ℕ
  sunday_apple : ℕ
  sunday_cherry : ℕ
  sunday_blueberry : ℕ

/-- Calculate the total number of each type of pie baked in a week -/
def total_pies (schedule : PieSchedule) : ℕ × ℕ × ℕ :=
  let apple := schedule.monday_apple + schedule.wednesday_apple + schedule.friday_apple + 
                schedule.saturday_apple + schedule.sunday_apple
  let cherry := schedule.tuesday_cherry + schedule.thursday_cherry + 
                 schedule.saturday_cherry + schedule.sunday_cherry
  let blueberry := schedule.monday_blueberry + schedule.tuesday_blueberry + 
                   schedule.wednesday_blueberry + schedule.thursday_blueberry + 
                   schedule.friday_blueberry + schedule.saturday_blueberry + 
                   schedule.sunday_blueberry
  (apple, cherry, blueberry)

/-- Steve's actual weekly pie baking schedule -/
def steve_schedule : PieSchedule := {
  monday_apple := 16, monday_blueberry := 10,
  tuesday_cherry := 14, tuesday_blueberry := 8,
  wednesday_apple := 20, wednesday_blueberry := 12,
  thursday_cherry := 18, thursday_blueberry := 10,
  friday_apple := 16, friday_blueberry := 10,
  saturday_apple := 10, saturday_cherry := 8, saturday_blueberry := 6,
  sunday_apple := 6, sunday_cherry := 12, sunday_blueberry := 4
}

theorem steve_pie_ratio : 
  ∃ (k : ℕ), k > 0 ∧ total_pies steve_schedule = (17 * k, 13 * k, 15 * k) := by
  sorry

end NUMINAMATH_CALUDE_steve_pie_ratio_l796_79637


namespace NUMINAMATH_CALUDE_or_and_not_implication_l796_79657

theorem or_and_not_implication (p q : Prop) :
  (p ∨ q) → ¬p → (¬p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_or_and_not_implication_l796_79657


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l796_79684

theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (2 * k : ℝ) = k * 2 + (-1) + 1 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l796_79684


namespace NUMINAMATH_CALUDE_x_value_l796_79641

theorem x_value (x : ℝ) (h1 : x > 0) (h2 : Real.sqrt ((8 * x) / 3) = x) : x = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l796_79641


namespace NUMINAMATH_CALUDE_world_cup_2018_21st_edition_l796_79698

/-- Represents the year of the nth World Cup -/
def worldCupYear (n : ℕ) : ℕ := 1950 + 4 * (n - 4)

/-- Theorem stating that the 2018 World Cup was the 21st edition -/
theorem world_cup_2018_21st_edition :
  ∃ n : ℕ, n = 21 ∧ worldCupYear n = 2018 :=
sorry

end NUMINAMATH_CALUDE_world_cup_2018_21st_edition_l796_79698


namespace NUMINAMATH_CALUDE_range_of_a_l796_79609

/-- The range of values for real number a given specific conditions -/
theorem range_of_a (a : ℝ) : 
  (∃ x, x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0) →
  (∃ x, x^2 + 2*x - 8 > 0) →
  (∀ x, (x^2 - 4*a*x + 3*a^2 ≥ 0 ∨ a ≥ 0) → (x^2 + 2*x - 8 ≤ 0)) →
  (∃ x, (x^2 + 2*x - 8 ≤ 0) ∧ (x^2 - 4*a*x + 3*a^2 ≥ 0 ∨ a ≥ 0)) →
  a ≤ -4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l796_79609


namespace NUMINAMATH_CALUDE_decimal_fraction_equality_l796_79632

theorem decimal_fraction_equality (b : ℕ+) : 
  (5 * b + 17 : ℚ) / (7 * b + 12) = 85 / 100 ↔ b = 7 := by sorry

end NUMINAMATH_CALUDE_decimal_fraction_equality_l796_79632


namespace NUMINAMATH_CALUDE_amy_school_year_work_hours_l796_79645

/-- Calculates the number of hours Amy needs to work per week during the school year -/
def school_year_hours_per_week (summer_weeks : ℕ) (summer_hours_per_week : ℕ) (summer_earnings : ℚ)
  (school_year_weeks : ℕ) (school_year_earnings : ℚ) : ℚ :=
  let hourly_rate := summer_earnings / (summer_weeks * summer_hours_per_week : ℚ)
  let total_school_year_hours := school_year_earnings / hourly_rate
  total_school_year_hours / school_year_weeks

/-- Proves that Amy needs to work approximately 18 hours per week during the school year -/
theorem amy_school_year_work_hours :
  let result := school_year_hours_per_week 10 36 3000 30 4500
  18 < result ∧ result < 19 := by
  sorry

end NUMINAMATH_CALUDE_amy_school_year_work_hours_l796_79645


namespace NUMINAMATH_CALUDE_special_checkerboard_black_squares_l796_79644

/-- A checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  isRed : Fin size → Fin size → Bool

/-- The properties of our specific checkerboard -/
def specialCheckerboard : Checkerboard where
  size := 32
  isRed := fun i j => 
    (i.val + j.val) % 2 = 0 ∨ i.val = 0 ∨ i.val = 31 ∨ j.val = 0 ∨ j.val = 31

/-- Count of black squares on the checkerboard -/
def blackSquareCount (c : Checkerboard) : ℕ :=
  (c.size * c.size) - (Finset.sum (Finset.univ : Finset (Fin c.size × Fin c.size)) 
    fun (i, j) => if c.isRed i j then 1 else 0)

/-- Theorem stating the number of black squares on our special checkerboard -/
theorem special_checkerboard_black_squares :
  blackSquareCount specialCheckerboard = 511 := by sorry

end NUMINAMATH_CALUDE_special_checkerboard_black_squares_l796_79644


namespace NUMINAMATH_CALUDE_h_of_two_eq_two_l796_79663

/-- The function h satisfying the given equation for all x -/
noncomputable def h : ℝ → ℝ :=
  fun x => ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) - 1) / (x^(2^4 - 1) - 1)

/-- Theorem stating that h(2) = 2 -/
theorem h_of_two_eq_two : h 2 = 2 := by sorry

end NUMINAMATH_CALUDE_h_of_two_eq_two_l796_79663


namespace NUMINAMATH_CALUDE_cubical_box_edge_length_cubical_box_edge_length_proof_l796_79620

/-- The edge length of a cubical box that can hold 64 cubes with edge length 25 cm is 1 meter. -/
theorem cubical_box_edge_length : Real → Prop := fun edge_length =>
  let small_cube_volume := (25 / 100) ^ 3
  let box_volume := 64 * small_cube_volume
  edge_length ^ 3 = box_volume → edge_length = 1

/-- Proof of the cubical box edge length theorem -/
theorem cubical_box_edge_length_proof : cubical_box_edge_length 1 := by
  sorry


end NUMINAMATH_CALUDE_cubical_box_edge_length_cubical_box_edge_length_proof_l796_79620


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l796_79604

-- Define the triangle's angles
def angle1 : ℝ := 40
def angle2 : ℝ := 70
def angle3 : ℝ := 180 - angle1 - angle2

-- Theorem statement
theorem largest_angle_in_triangle : 
  max angle1 (max angle2 angle3) = 70 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l796_79604


namespace NUMINAMATH_CALUDE_sum_reciprocal_complements_l796_79602

theorem sum_reciprocal_complements (a b c d : ℝ) 
  (h1 : a + b + c + d = 2) 
  (h2 : 1/a + 1/b + 1/c + 1/d = 2) : 
  1/(1-a) + 1/(1-b) + 1/(1-c) + 1/(1-d) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_complements_l796_79602


namespace NUMINAMATH_CALUDE_smallest_four_digit_different_digits_l796_79666

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_different_digits :
  ∀ n : ℕ, is_four_digit n → has_different_digits n → 1023 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_different_digits_l796_79666


namespace NUMINAMATH_CALUDE_train_crossing_time_l796_79635

theorem train_crossing_time (train_length : ℝ) (platform1_length : ℝ) (platform2_length : ℝ) (time1 : ℝ) :
  train_length = 100 →
  platform1_length = 350 →
  platform2_length = 500 →
  time1 = 15 →
  let total_distance1 := train_length + platform1_length
  let speed := total_distance1 / time1
  let total_distance2 := train_length + platform2_length
  let time2 := total_distance2 / speed
  time2 = 20 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l796_79635


namespace NUMINAMATH_CALUDE_number_problem_l796_79656

theorem number_problem (x : ℝ) : (x - 6) / 8 = 6 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l796_79656


namespace NUMINAMATH_CALUDE_shaniqua_style_price_l796_79622

/-- Proves that Shaniqua makes $25 for every style given the conditions -/
theorem shaniqua_style_price 
  (haircut_price : ℕ) 
  (total_earned : ℕ) 
  (num_haircuts : ℕ) 
  (num_styles : ℕ) 
  (h1 : haircut_price = 12)
  (h2 : total_earned = 221)
  (h3 : num_haircuts = 8)
  (h4 : num_styles = 5)
  (h5 : total_earned = num_haircuts * haircut_price + num_styles * (total_earned - num_haircuts * haircut_price) / num_styles) : 
  (total_earned - num_haircuts * haircut_price) / num_styles = 25 := by
  sorry

end NUMINAMATH_CALUDE_shaniqua_style_price_l796_79622


namespace NUMINAMATH_CALUDE_digit_sum_problem_l796_79638

/-- Given that P, Q, and R are single digits and PQR + QR = 1012, prove that P + Q + R = 20 -/
theorem digit_sum_problem (P Q R : ℕ) : 
  P < 10 → Q < 10 → R < 10 → 
  100 * P + 10 * Q + R + 10 * Q + R = 1012 →
  P + Q + R = 20 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l796_79638


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l796_79659

/-- An arithmetic sequence with a specific condition -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  condition : a 3 + a 6 + 3 * a 7 = 20

/-- The theorem stating that for any arithmetic sequence satisfying the given condition, 2a₇ - a₈ = 4 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) : 2 * seq.a 7 - seq.a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l796_79659


namespace NUMINAMATH_CALUDE_floor_of_5_7_l796_79694

theorem floor_of_5_7 : ⌊(5.7 : ℝ)⌋ = 5 := by sorry

end NUMINAMATH_CALUDE_floor_of_5_7_l796_79694


namespace NUMINAMATH_CALUDE_greatest_k_value_l796_79621

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 8 = 0 ∧ 
    x₂^2 + k*x₂ + 8 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 89) →
  k ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_value_l796_79621


namespace NUMINAMATH_CALUDE_prime_factors_of_1998_l796_79651

theorem prime_factors_of_1998 (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a < b ∧ b < c ∧
  a * b * c = 1998 →
  (b + c)^a = 1600 := by
sorry

end NUMINAMATH_CALUDE_prime_factors_of_1998_l796_79651


namespace NUMINAMATH_CALUDE_total_pizza_combinations_l796_79616

def num_toppings : ℕ := 8

def num_one_topping (n : ℕ) : ℕ := n

def num_two_toppings (n : ℕ) : ℕ := n.choose 2

def num_three_toppings (n : ℕ) : ℕ := n.choose 3

theorem total_pizza_combinations :
  num_one_topping num_toppings + num_two_toppings num_toppings + num_three_toppings num_toppings = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_pizza_combinations_l796_79616


namespace NUMINAMATH_CALUDE_min_value_of_expression_l796_79678

theorem min_value_of_expression (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  x*y/z + y*z/x + z*x/y ≥ Real.sqrt 3 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 + b^2 + c^2 = 1 ∧ 
    a*b/c + b*c/a + c*a/b = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l796_79678


namespace NUMINAMATH_CALUDE_parabola_focus_l796_79690

-- Define the parabola
def parabola (x : ℝ) : ℝ := -4 * x^2 - 8 * x + 1

-- Define the focus of a parabola
def focus (a b c : ℝ) : ℝ × ℝ := sorry

-- Theorem statement
theorem parabola_focus :
  focus (-4) (-8) 1 = (-1, 79/16) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l796_79690


namespace NUMINAMATH_CALUDE_triangle_properties_l796_79649

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- The given equation
  (Real.sqrt 3 * Real.sin A * Real.cos A - Real.sin A ^ 2 = 0) ∧
  -- Collinearity condition
  (∃ (k : Real), k ≠ 0 ∧ k * 1 = 2 ∧ k * Real.sin C = Real.sin B) ∧
  -- Given side length
  (a = 3) →
  -- Prove angle A and perimeter
  (A = π / 3) ∧
  (a + b + c = 3 * (1 + Real.sqrt 3)) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l796_79649


namespace NUMINAMATH_CALUDE_complex_modulus_of_fraction_l796_79648

theorem complex_modulus_of_fraction (z : ℂ) : 
  z = (4 - 2*I) / (1 - I) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_of_fraction_l796_79648


namespace NUMINAMATH_CALUDE_lily_pad_half_coverage_l796_79667

theorem lily_pad_half_coverage (total_days : ℕ) (half_coverage_days : ℕ) : 
  (total_days = 34) → 
  (half_coverage_days = total_days - 1) →
  (half_coverage_days = 33) :=
by sorry

end NUMINAMATH_CALUDE_lily_pad_half_coverage_l796_79667


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_B_intersection_complements_A_B_l796_79661

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set A
def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 8}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 5} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 8} := by sorry

-- Theorem for the intersection of complements of A and B
theorem intersection_complements_A_B : (Aᶜ : Set ℝ) ∩ (Bᶜ : Set ℝ) = {x : ℝ | x < 1 ∨ x ≥ 8} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_B_intersection_complements_A_B_l796_79661


namespace NUMINAMATH_CALUDE_unique_solution_condition_l796_79689

/-- The diamond-shaped region defined by |x| + |y - 1| = 1 -/
def DiamondRegion (x y : ℝ) : Prop :=
  (abs x) + (abs (y - 1)) = 1

/-- The line defined by y = a * x + 2012 -/
def Line (a x y : ℝ) : Prop :=
  y = a * x + 2012

/-- The system of equations has a unique solution -/
def UniqueSystemSolution (a : ℝ) : Prop :=
  ∃! (x y : ℝ), DiamondRegion x y ∧ Line a x y

/-- The theorem stating the condition for a unique solution -/
theorem unique_solution_condition (a : ℝ) :
  UniqueSystemSolution a ↔ (a = 2011 ∨ a = -2011) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l796_79689


namespace NUMINAMATH_CALUDE_factory_sampling_probability_l796_79629

/-- Represents a district with a number of factories -/
structure District where
  name : String
  factories : ℕ

/-- Represents the sampling result -/
structure SamplingResult where
  districtA : ℕ
  districtB : ℕ
  districtC : ℕ

/-- The stratified sampling function -/
def stratifiedSampling (districts : List District) (totalSample : ℕ) : SamplingResult :=
  sorry

/-- The probability calculation function -/
def probabilityAtLeastOneFromC (sample : SamplingResult) : ℚ :=
  sorry

theorem factory_sampling_probability :
  let districts := [
    { name := "A", factories := 9 },
    { name := "B", factories := 18 },
    { name := "C", factories := 18 }
  ]
  let sample := stratifiedSampling districts 5
  sample.districtA = 1 ∧
  sample.districtB = 2 ∧
  sample.districtC = 2 ∧
  probabilityAtLeastOneFromC sample = 7/10 :=
by sorry

end NUMINAMATH_CALUDE_factory_sampling_probability_l796_79629


namespace NUMINAMATH_CALUDE_winner_equal_victories_defeats_iff_odd_l796_79643

/-- A tournament of n nations where each nation plays against every other nation exactly once. -/
structure Tournament (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- The number of victories for a team in a tournament. -/
def victories (t : Tournament n) (team : Fin n) : ℕ := sorry

/-- The number of defeats for a team in a tournament. -/
def defeats (t : Tournament n) (team : Fin n) : ℕ := sorry

/-- A team is a winner if it has the maximum number of victories. -/
def is_winner (t : Tournament n) (team : Fin n) : Prop :=
  ∀ other : Fin n, victories t team ≥ victories t other

theorem winner_equal_victories_defeats_iff_odd (n : ℕ) :
  (∃ (t : Tournament n) (w : Fin n), is_winner t w ∧ victories t w = defeats t w) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_winner_equal_victories_defeats_iff_odd_l796_79643


namespace NUMINAMATH_CALUDE_intersection_condition_l796_79607

-- Define the curves
def C₁ (x : ℝ) : Prop := ∃ y : ℝ, y = x^2 ∧ -2 ≤ x ∧ x ≤ 2

def C₂ (m : ℝ) (x y : ℝ) : Prop := x - y + m = 0

-- Theorem statement
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, C₁ x ∧ C₂ m x y) ↔ -1/4 ≤ m ∧ m ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l796_79607


namespace NUMINAMATH_CALUDE_tangent_through_origin_l796_79601

theorem tangent_through_origin (x : ℝ) :
  (∃ y : ℝ, y = Real.exp x ∧ 
   (Real.exp x) * (0 - x) = 0 - y) →
  x = 1 ∧ Real.exp x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_through_origin_l796_79601


namespace NUMINAMATH_CALUDE_price_of_pants_l796_79614

theorem price_of_pants (total_cost shirt_price pants_price shoes_price : ℝ) : 
  total_cost = 340 →
  shirt_price = (3/4) * pants_price →
  shoes_price = pants_price + 10 →
  total_cost = shirt_price + pants_price + shoes_price →
  pants_price = 120 := by
sorry

end NUMINAMATH_CALUDE_price_of_pants_l796_79614


namespace NUMINAMATH_CALUDE_tank_width_is_twelve_l796_79675

/-- Represents the dimensions and plastering cost of a tank. -/
structure Tank where
  length : ℝ
  depth : ℝ
  width : ℝ
  plasteringRate : ℝ
  totalCost : ℝ

/-- Calculates the total surface area of the tank. -/
def surfaceArea (t : Tank) : ℝ :=
  2 * (t.length * t.depth) + 2 * (t.width * t.depth) + (t.length * t.width)

/-- Theorem stating that for a tank with given dimensions and plastering cost,
    the width is 12 meters. -/
theorem tank_width_is_twelve (t : Tank)
  (h1 : t.length = 25)
  (h2 : t.depth = 6)
  (h3 : t.plasteringRate = 0.25)
  (h4 : t.totalCost = 186)
  (h5 : t.totalCost = t.plasteringRate * surfaceArea t) :
  t.width = 12 := by
  sorry

#check tank_width_is_twelve

end NUMINAMATH_CALUDE_tank_width_is_twelve_l796_79675


namespace NUMINAMATH_CALUDE_consecutive_integers_permutation_divisibility_l796_79676

theorem consecutive_integers_permutation_divisibility
  (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1)
  (m : ℕ → ℕ) (hm : ∀ i ∈ Finset.range p, m (i + 1) = m i + 1)
  (σ : Fin p → Fin p) (hσ : Function.Bijective σ) :
  ∃ k l : Fin p, k ≠ l ∧ p ∣ (m k * m (σ k) - m l * m (σ l)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_integers_permutation_divisibility_l796_79676


namespace NUMINAMATH_CALUDE_equal_age_regression_five_year_difference_regression_ten_percent_older_regression_l796_79640

-- Define the type for a couple's ages
structure CoupleAges where
  bride_age : ℝ
  groom_age : ℝ

-- Define the dataset
def dataset : List CoupleAges := sorry

-- Define the regression line
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

-- Function to calculate regression line
def calculate_regression_line (data : List CoupleAges) : RegressionLine := sorry

-- Theorems to prove

theorem equal_age_regression :
  (∀ couple ∈ dataset, couple.bride_age = couple.groom_age) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1 ∧ reg_line.intercept = 0 := by sorry

theorem five_year_difference_regression :
  (∀ couple ∈ dataset, couple.groom_age = couple.bride_age + 5) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1 ∧ reg_line.intercept = 5 := by sorry

theorem ten_percent_older_regression :
  (∀ couple ∈ dataset, couple.groom_age = 1.1 * couple.bride_age) →
  let reg_line := calculate_regression_line dataset
  reg_line.slope = 1.1 ∧ reg_line.intercept = 0 := by sorry

end NUMINAMATH_CALUDE_equal_age_regression_five_year_difference_regression_ten_percent_older_regression_l796_79640


namespace NUMINAMATH_CALUDE_square_sum_xy_l796_79692

theorem square_sum_xy (x y : ℝ) 
  (h1 : x * (x + y) = 35)
  (h2 : y * (x + y) = 77) : 
  (x + y)^2 = 112 := by
sorry

end NUMINAMATH_CALUDE_square_sum_xy_l796_79692


namespace NUMINAMATH_CALUDE_latia_work_hours_l796_79625

/-- Proves that Latia works 30 hours per week given the problem conditions -/
theorem latia_work_hours :
  ∀ (tv_price : ℕ) (hourly_rate : ℕ) (additional_hours : ℕ) (weeks_per_month : ℕ),
  tv_price = 1700 →
  hourly_rate = 10 →
  additional_hours = 50 →
  weeks_per_month = 4 →
  ∃ (hours_per_week : ℕ),
    hours_per_week * weeks_per_month * hourly_rate + additional_hours * hourly_rate = tv_price ∧
    hours_per_week = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_latia_work_hours_l796_79625


namespace NUMINAMATH_CALUDE_consecutive_even_product_divisibility_l796_79693

theorem consecutive_even_product_divisibility (n : ℤ) (h : Even n) :
  ∃ k : ℤ, n * (n + 2) * (n + 4) * (n + 6) = 48 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_product_divisibility_l796_79693


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_l796_79695

-- Proposition 1
def p : Prop := ∃ x : ℝ, Real.tan x = 2
def q : Prop := ∀ x : ℝ, x^2 - x + 1/2 > 0

theorem proposition_1 : ¬(p ∧ ¬q) := by sorry

-- Proposition 2
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y - 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := x + b * y + 1 = 0

theorem proposition_2 (a b : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ b x y → (a * 1 + 3 * b = 0)) ≠ 
  (∀ x y : ℝ, l₁ a x y → l₂ b x y → (a / b = -3)) := by sorry

-- Proposition 3
def original_statement (a b : ℝ) : Prop := 
  a * b ≥ 2 → a^2 + b^2 > 4

def negation_statement (a b : ℝ) : Prop := 
  a * b < 2 → a^2 + b^2 ≤ 4

theorem proposition_3 : 
  (∀ a b : ℝ, ¬(original_statement a b)) ↔ (∀ a b : ℝ, negation_statement a b) := by sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_proposition_3_l796_79695


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l796_79608

def base9ToDecimal (n : Nat) : Nat :=
  (n / 100) * 9^2 + ((n / 10) % 10) * 9 + (n % 10)

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : Nat), 
    n < 1000 ∧ 
    base9ToDecimal n % 7 = 0 ∧
    (∀ m : Nat, m < 1000 → base9ToDecimal m % 7 = 0 → m ≤ n) ∧
    n = 888 := by
  sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l796_79608


namespace NUMINAMATH_CALUDE_first_alloy_copper_percentage_l796_79603

/-- The percentage of copper in the final alloy -/
def final_alloy_percentage : ℝ := 15

/-- The total amount of the final alloy in ounces -/
def total_alloy : ℝ := 121

/-- The amount of the first alloy used in ounces -/
def first_alloy_amount : ℝ := 66

/-- The percentage of copper in the second alloy -/
def second_alloy_percentage : ℝ := 21

/-- The percentage of copper in the first alloy -/
def first_alloy_percentage : ℝ := 10

theorem first_alloy_copper_percentage :
  first_alloy_amount * (first_alloy_percentage / 100) +
  (total_alloy - first_alloy_amount) * (second_alloy_percentage / 100) =
  total_alloy * (final_alloy_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_first_alloy_copper_percentage_l796_79603


namespace NUMINAMATH_CALUDE_parentheses_removal_correctness_l796_79654

theorem parentheses_removal_correctness :
  let A := (x y : ℝ) → 5*x - (x - 2*y) = 5*x - x + 2*y
  let B := (a b : ℝ) → 2*a^2 + (3*a - b) = 2*a^2 + 3*a - b
  let C := (x y : ℝ) → (x - 2*y) - (x^2 - y^2) = x - 2*y - x^2 + y^2
  let D := (x : ℝ) → 3*x^2 - 3*(x + 6) = 3*x^2 - 3*x - 6
  A ∧ B ∧ C ∧ ¬D := by sorry

end NUMINAMATH_CALUDE_parentheses_removal_correctness_l796_79654


namespace NUMINAMATH_CALUDE_team_capacity_ratio_l796_79617

/-- The working capacity ratio of two teams -/
def working_capacity_ratio (p_engineers q_engineers : ℕ) (p_days q_days : ℕ) : ℚ × ℚ :=
  let p_capacity := p_engineers * p_days / p_engineers
  let q_capacity := q_engineers * q_days / q_engineers
  (p_capacity, q_capacity)

/-- Theorem: The ratio of working capacity for the given teams is 16:15 -/
theorem team_capacity_ratio :
  let (p_cap, q_cap) := working_capacity_ratio 20 16 32 30
  p_cap / q_cap = 16 / 15 := by
  sorry

end NUMINAMATH_CALUDE_team_capacity_ratio_l796_79617


namespace NUMINAMATH_CALUDE_range_of_m_l796_79619

/-- The probability that two lines l₁: ax + by = 2 and l₂: x + 2y = 2 are parallel,
    where a and b are results of two dice throws. -/
def P₁ : ℚ := 1/18

/-- The probability that two lines l₁: ax + by = 2 and l₂: x + 2y = 2 intersect,
    where a and b are results of two dice throws. -/
def P₂ : ℚ := 11/12

/-- The theorem stating the range of m for which the point (P₁, P₂) is inside
    the circle (x-m)² + y² = 137/144. -/
theorem range_of_m : ∀ m : ℚ, 
  (P₁ - m)^2 + P₂^2 < 137/144 ↔ -5/18 < m ∧ m < 7/18 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l796_79619


namespace NUMINAMATH_CALUDE_selling_price_calculation_l796_79642

theorem selling_price_calculation (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 110 →
  gain_percent = 13.636363636363626 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 125 := by
sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l796_79642


namespace NUMINAMATH_CALUDE_P_intersect_Q_l796_79647

def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | x ≤ 2}

theorem P_intersect_Q : P ∩ Q = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_P_intersect_Q_l796_79647


namespace NUMINAMATH_CALUDE_bart_burning_period_l796_79610

/-- The number of pieces of firewood Bart gets from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of logs Bart burns per day -/
def logs_per_day : ℕ := 5

/-- The number of trees Bart cuts down for the period -/
def trees_cut : ℕ := 8

/-- The period (in days) that Bart burns the logs -/
def burning_period : ℕ := (pieces_per_tree * trees_cut) / logs_per_day

theorem bart_burning_period :
  burning_period = 120 := by sorry

end NUMINAMATH_CALUDE_bart_burning_period_l796_79610


namespace NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_of_factors_achieved_l796_79652

theorem max_sum_of_factors (a b : ℕ) : a * b = 48 → a ≠ b → a + b ≤ 49 := by sorry

theorem max_sum_of_factors_achieved : ∃ a b : ℕ, a * b = 48 ∧ a ≠ b ∧ a + b = 49 := by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_max_sum_of_factors_achieved_l796_79652


namespace NUMINAMATH_CALUDE_max_volume_container_l796_79671

/-- Represents a rectangular container made from a sheet with cut corners. -/
structure Container where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the container. -/
def volume (c : Container) : ℝ := (c.length - 2 * c.height) * (c.width - 2 * c.height) * c.height

/-- The original sheet dimensions. -/
def sheet_length : ℝ := 8
def sheet_width : ℝ := 5

/-- Theorem stating the maximum volume and the height at which it occurs. -/
theorem max_volume_container :
  ∃ (h : ℝ),
    h > 0 ∧
    h < min sheet_length sheet_width / 2 ∧
    (∀ (c : Container),
      c.length = sheet_length ∧
      c.width = sheet_width ∧
      c.height > 0 ∧
      c.height < min sheet_length sheet_width / 2 →
      volume c ≤ volume { length := sheet_length, width := sheet_width, height := h }) ∧
    volume { length := sheet_length, width := sheet_width, height := h } = 18 :=
  sorry

end NUMINAMATH_CALUDE_max_volume_container_l796_79671


namespace NUMINAMATH_CALUDE_bee_path_distance_l796_79628

open Complex

-- Define ω as e^(πi/4)
noncomputable def ω : ℂ := exp (I * Real.pi / 4)

-- Define the path of the bee
noncomputable def z : ℂ := 1 + 2 * ω + 3 * ω^2 + 4 * ω^3 + 5 * ω^4 + 6 * ω^5 + 7 * ω^6

-- Theorem stating the distance from P₀ to P₇
theorem bee_path_distance : abs z = Real.sqrt (25 - 7 * Real.sqrt 2 / 2) := by sorry

end NUMINAMATH_CALUDE_bee_path_distance_l796_79628


namespace NUMINAMATH_CALUDE_computer_price_ratio_l796_79605

theorem computer_price_ratio (d : ℝ) : 
  d + 0.3 * d = 377 → (d + 377) / d = 2.3 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_ratio_l796_79605


namespace NUMINAMATH_CALUDE_event_ticket_revenue_l796_79686

theorem event_ticket_revenue :
  ∀ (full_price half_price : ℕ),
  full_price + half_price = 180 →
  full_price * 20 + half_price * 10 = 2750 →
  full_price * 20 = 1900 :=
by
  sorry

end NUMINAMATH_CALUDE_event_ticket_revenue_l796_79686


namespace NUMINAMATH_CALUDE_no_such_function_exists_l796_79674

theorem no_such_function_exists : ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, (f^[f x]) x = x + 1 := by
  sorry

#check no_such_function_exists

end NUMINAMATH_CALUDE_no_such_function_exists_l796_79674


namespace NUMINAMATH_CALUDE_factor_problems_l796_79655

theorem factor_problems : 
  (∃ n : ℤ, 25 = 5 * n) ∧ (∃ m : ℤ, 200 = 10 * m) := by
  sorry

end NUMINAMATH_CALUDE_factor_problems_l796_79655


namespace NUMINAMATH_CALUDE_sum_is_five_digits_l796_79624

/-- A nonzero digit is a natural number between 1 and 9, inclusive. -/
def NonzeroDigit : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- Convert a nonzero digit to a natural number. -/
def to_nat (d : NonzeroDigit) : ℕ := d.val

/-- The first number in the sum. -/
def num1 : ℕ := 59876

/-- The second number in the sum, parameterized by a nonzero digit A. -/
def num2 (A : NonzeroDigit) : ℕ := 1000 + 100 * (to_nat A) + 32

/-- The third number in the sum, parameterized by a nonzero digit B. -/
def num3 (B : NonzeroDigit) : ℕ := 10 * (to_nat B) + 1

/-- The sum of the three numbers. -/
def total_sum (A B : NonzeroDigit) : ℕ := num1 + num2 A + num3 B

/-- A number is a 5-digit number if it's between 10000 and 99999, inclusive. -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem sum_is_five_digits (A B : NonzeroDigit) : is_five_digit (total_sum A B) := by
  sorry

end NUMINAMATH_CALUDE_sum_is_five_digits_l796_79624


namespace NUMINAMATH_CALUDE_absolute_value_nonnegative_l796_79665

theorem absolute_value_nonnegative (a : ℝ) : |a| ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_nonnegative_l796_79665


namespace NUMINAMATH_CALUDE_candy_sold_tuesday_l796_79653

/-- Theorem: Candy sold on Tuesday --/
theorem candy_sold_tuesday (initial_candy : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) :
  initial_candy = 80 →
  sold_monday = 15 →
  remaining_wednesday = 7 →
  initial_candy - sold_monday - remaining_wednesday = 58 := by
  sorry

end NUMINAMATH_CALUDE_candy_sold_tuesday_l796_79653


namespace NUMINAMATH_CALUDE_total_balls_l796_79618

theorem total_balls (jungkook_balls yoongi_balls : ℕ) : 
  jungkook_balls = 3 → yoongi_balls = 2 → jungkook_balls + yoongi_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_l796_79618


namespace NUMINAMATH_CALUDE_remaining_average_l796_79691

theorem remaining_average (total : ℕ) (subset : ℕ) (total_avg : ℚ) (subset_avg : ℚ) :
  total = 5 ∧ subset = 3 ∧ total_avg = 11 ∧ subset_avg = 4 →
  ((total_avg * total) - (subset_avg * subset)) / (total - subset) = 21.5 :=
by sorry

end NUMINAMATH_CALUDE_remaining_average_l796_79691


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l796_79677

theorem rectangular_box_volume (a b c : ℝ) 
  (h1 : a * b = 15) 
  (h2 : b * c = 10) 
  (h3 : c * a = 6) : 
  a * b * c = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l796_79677


namespace NUMINAMATH_CALUDE_intersection_area_is_sqrt_80_l796_79606

/-- Represents a square pyramid -/
structure SquarePyramid where
  base_side : ℝ
  edge_length : ℝ

/-- Represents a plane intersecting the pyramid -/
structure IntersectingPlane where
  pyramid : SquarePyramid
  -- The plane passes through midpoints of one lateral edge and two base edges

/-- The area of intersection between the plane and the pyramid -/
noncomputable def intersection_area (plane : IntersectingPlane) : ℝ := sorry

theorem intersection_area_is_sqrt_80 (plane : IntersectingPlane) :
  plane.pyramid.base_side = 4 →
  plane.pyramid.edge_length = 4 →
  intersection_area plane = Real.sqrt 80 := by
  sorry

end NUMINAMATH_CALUDE_intersection_area_is_sqrt_80_l796_79606


namespace NUMINAMATH_CALUDE_range_of_a_l796_79615

/-- The solution set of the inequality |x+a|+|2x-1| ≤ |2x+1| with respect to x -/
def A (a : ℝ) : Set ℝ :=
  {x : ℝ | |x + a| + |2*x - 1| ≤ |2*x + 1|}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x ∈ A a) → a ∈ Set.Icc (-3) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l796_79615
