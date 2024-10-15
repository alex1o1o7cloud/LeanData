import Mathlib

namespace NUMINAMATH_GPT_percentage_of_men_l2387_238797

variable (M W : ℝ)
variable (h1 : M + W = 100)
variable (h2 : 0.20 * W + 0.70 * M = 40)

theorem percentage_of_men : M = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_l2387_238797


namespace NUMINAMATH_GPT_maximize_container_volume_l2387_238716

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ ∀ y : ℝ, 0 < y ∧ y < 24 → 
  ( (48 - 2 * x)^2 * x ≥ (48 - 2 * y)^2 * y ) ∧ x = 8 :=
sorry

end NUMINAMATH_GPT_maximize_container_volume_l2387_238716


namespace NUMINAMATH_GPT_intersection_complement_l2387_238722

def M : Set ℝ := { x | x^2 - x - 6 ≥ 0 }
def N : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }
def neg_R (A : Set ℝ) : Set ℝ := { x | x ∉ A }

theorem intersection_complement (N : Set ℝ) (M : Set ℝ) :
  N ∩ (neg_R M) = { x | -2 < x ∧ x ≤ 1 } := 
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_intersection_complement_l2387_238722


namespace NUMINAMATH_GPT_expression_evaluation_l2387_238745

theorem expression_evaluation : 
  (3.14 - Real.pi)^0 + abs (Real.sqrt 2 - 1) + (1 / 2)^(-1:ℤ) - Real.sqrt 8 = 2 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_expression_evaluation_l2387_238745


namespace NUMINAMATH_GPT_correct_calculation_l2387_238736

theorem correct_calculation :
    (1 + Real.sqrt 2)^2 = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_correct_calculation_l2387_238736


namespace NUMINAMATH_GPT_present_age_of_son_l2387_238780

theorem present_age_of_son (M S : ℕ) (h1 : M = S + 29) (h2 : M + 2 = 2 * (S + 2)) : S = 27 :=
sorry

end NUMINAMATH_GPT_present_age_of_son_l2387_238780


namespace NUMINAMATH_GPT_sam_total_pennies_l2387_238765

theorem sam_total_pennies : 
  ∀ (initial_pennies found_pennies total_pennies : ℕ),
  initial_pennies = 98 → 
  found_pennies = 93 → 
  total_pennies = initial_pennies + found_pennies → 
  total_pennies = 191 := by
  intros
  sorry

end NUMINAMATH_GPT_sam_total_pennies_l2387_238765


namespace NUMINAMATH_GPT_find_a_l2387_238702

theorem find_a (a n : ℕ) (h1 : (2 : ℕ) ^ n = 32) (h2 : (a + 1) ^ n = 243) : a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l2387_238702


namespace NUMINAMATH_GPT_exactly_one_negative_x_or_y_l2387_238734

theorem exactly_one_negative_x_or_y
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (x1_ne_zero : x1 ≠ 0) (x2_ne_zero : x2 ≠ 0) (x3_ne_zero : x3 ≠ 0)
  (y1_ne_zero : y1 ≠ 0) (y2_ne_zero : y2 ≠ 0) (y3_ne_zero : y3 ≠ 0)
  (h1 : x1 * x2 * x3 = - y1 * y2 * y3)
  (h2 : x1^2 + x2^2 + x3^2 = y1^2 + y2^2 + y3^2)
  (h3 : x1 + y1 + x2 + y2 ≥ x3 + y3 ∧ x2 + y2 + x3 + y3 ≥ x1 + y1 ∧ x3 + y3 + x1 + y1 ≥ x2 + y2)
  (h4 : (x1 + y1)^2 + (x2 + y2)^2 ≥ (x3 + y3)^2 ∧ (x2 + y2)^2 + (x3 + y3)^2 ≥ (x1 + y1)^2 ∧ (x3 + y3)^2 + (x1 + y1)^2 ≥ (x2 + y2)^2) :
  ∃! (a : ℝ), (a = x1 ∨ a = x2 ∨ a = x3 ∨ a = y1 ∨ a = y2 ∨ a = y3) ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_exactly_one_negative_x_or_y_l2387_238734


namespace NUMINAMATH_GPT_exponential_graph_passes_through_point_l2387_238777

variable (a : ℝ) (hx1 : a > 0) (hx2 : a ≠ 1)

theorem exponential_graph_passes_through_point :
  ∃ y : ℝ, (y = a^0 + 1) ∧ (y = 2) :=
sorry

end NUMINAMATH_GPT_exponential_graph_passes_through_point_l2387_238777


namespace NUMINAMATH_GPT_difference_of_squares_401_399_l2387_238730

theorem difference_of_squares_401_399 : 401^2 - 399^2 = 1600 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_401_399_l2387_238730


namespace NUMINAMATH_GPT_john_hourly_wage_with_bonus_l2387_238715

structure JohnJob where
  daily_wage : ℕ
  work_hours : ℕ
  bonus_amount : ℕ
  extra_hours : ℕ

def total_daily_wage (job : JohnJob) : ℕ :=
  job.daily_wage + job.bonus_amount

def total_work_hours (job : JohnJob) : ℕ :=
  job.work_hours + job.extra_hours

def hourly_wage (job : JohnJob) : ℕ :=
  total_daily_wage job / total_work_hours job

noncomputable def johns_job : JohnJob :=
  { daily_wage := 80, work_hours := 8, bonus_amount := 20, extra_hours := 2 }

theorem john_hourly_wage_with_bonus :
  hourly_wage johns_job = 10 :=
by
  sorry

end NUMINAMATH_GPT_john_hourly_wage_with_bonus_l2387_238715


namespace NUMINAMATH_GPT_Maya_takes_longer_l2387_238743

-- Define the constants according to the conditions
def Xavier_reading_speed : ℕ := 120
def Maya_reading_speed : ℕ := 60
def novel_pages : ℕ := 360
def minutes_per_hour : ℕ := 60

-- Define the times it takes for Xavier and Maya to read the novel
def Xavier_time : ℕ := novel_pages / Xavier_reading_speed
def Maya_time : ℕ := novel_pages / Maya_reading_speed

-- Define the time difference in hours and then in minutes
def time_difference_hours : ℕ := Maya_time - Xavier_time
def time_difference_minutes : ℕ := time_difference_hours * minutes_per_hour

-- The statement to prove
theorem Maya_takes_longer :
  time_difference_minutes = 180 :=
by
  sorry

end NUMINAMATH_GPT_Maya_takes_longer_l2387_238743


namespace NUMINAMATH_GPT_find_a7_l2387_238785

variable (a : ℕ → ℝ)
variable (r : ℝ)
variable (n : ℕ)

-- Condition 1: The sequence {a_n} is geometric with all positive terms.
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

-- Condition 2: a₄ * a₁₀ = 16
axiom geo_seq_condition : is_geometric_sequence a r ∧ a 4 * a 10 = 16

-- The goal to prove
theorem find_a7 : (is_geometric_sequence a r ∧ a 4 * a 10 = 16) → a 7 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a7_l2387_238785


namespace NUMINAMATH_GPT_units_digit_of_expression_l2387_238704

theorem units_digit_of_expression :
  (9 * 19 * 1989 - 9 ^ 3) % 10 = 0 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_expression_l2387_238704


namespace NUMINAMATH_GPT_probability_20_correct_l2387_238767

noncomputable def probability_sum_20_dodecahedral : ℚ :=
  let num_faces := 12
  let total_outcomes := num_faces * num_faces
  let favorable_outcomes := 5
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_20_correct : probability_sum_20_dodecahedral = 5 / 144 := 
by 
  sorry

end NUMINAMATH_GPT_probability_20_correct_l2387_238767


namespace NUMINAMATH_GPT_avg_tickets_sold_by_males_100_l2387_238720

theorem avg_tickets_sold_by_males_100 
  (female_avg : ℕ := 70) 
  (nonbinary_avg : ℕ := 50) 
  (overall_avg : ℕ := 66) 
  (male_ratio : ℕ := 2) 
  (female_ratio : ℕ := 3) 
  (nonbinary_ratio : ℕ := 5) : 
  ∃ (male_avg : ℕ), male_avg = 100 := 
by 
  sorry

end NUMINAMATH_GPT_avg_tickets_sold_by_males_100_l2387_238720


namespace NUMINAMATH_GPT_desired_average_sale_is_5600_l2387_238763

-- Define the sales for five consecutive months
def sale1 : ℕ := 5266
def sale2 : ℕ := 5768
def sale3 : ℕ := 5922
def sale4 : ℕ := 5678
def sale5 : ℕ := 6029

-- Define the required sale for the sixth month
def sale6 : ℕ := 4937

-- Calculate total sales for the first five months
def total_five_months := sale1 + sale2 + sale3 + sale4 + sale5

-- Calculate total sales for six months
def total_six_months := total_five_months + sale6

-- Calculate the desired average sale for six months
def desired_average := total_six_months / 6

-- The theorem statement: desired average sale for the six months
theorem desired_average_sale_is_5600 : desired_average = 5600 :=
by
  sorry

end NUMINAMATH_GPT_desired_average_sale_is_5600_l2387_238763


namespace NUMINAMATH_GPT_problem_statement_l2387_238749

-- Definitions corresponding to the given condition
noncomputable def sum_to_n (n : ℕ) : ℤ := (n * (n + 1)) / 2
noncomputable def alternating_sum_to_n (n : ℕ) : ℤ := if n % 2 = 0 then -(n / 2) else (n / 2 + 1)

-- Lean statement for the problem
theorem problem_statement :
  (alternating_sum_to_n 2022) * (sum_to_n 2023 - 1) - (alternating_sum_to_n 2023) * (sum_to_n 2022 - 1) = 2023 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2387_238749


namespace NUMINAMATH_GPT_mul_97_103_l2387_238728

theorem mul_97_103 : (97:ℤ) = 100 - 3 → (103:ℤ) = 100 + 3 → 97 * 103 = 9991 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_mul_97_103_l2387_238728


namespace NUMINAMATH_GPT_no_consecutive_positive_integers_with_no_real_solutions_l2387_238757

theorem no_consecutive_positive_integers_with_no_real_solutions :
  ∀ b c : ℕ, (c = b + 1) → (b^2 - 4 * c < 0) → (c^2 - 4 * b < 0) → false :=
by
  intro b c
  sorry

end NUMINAMATH_GPT_no_consecutive_positive_integers_with_no_real_solutions_l2387_238757


namespace NUMINAMATH_GPT_arrange_magnitudes_l2387_238762

theorem arrange_magnitudes (x : ℝ) (h1 : 0.85 < x) (h2 : x < 1.1)
  (y : ℝ := x + Real.sin x) (z : ℝ := x ^ (x ^ x)) : x < y ∧ y < z := 
sorry

end NUMINAMATH_GPT_arrange_magnitudes_l2387_238762


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l2387_238752

theorem sum_of_consecutive_integers (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + 1 = b) (h4 : b + 1 = c) (h5 : a * b * c = 336) : a + b + c = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l2387_238752


namespace NUMINAMATH_GPT_tod_driving_time_l2387_238798
noncomputable def total_driving_time (distance_north distance_west speed : ℕ) : ℕ :=
  (distance_north + distance_west) / speed

theorem tod_driving_time :
  total_driving_time 55 95 25 = 6 :=
by
  sorry

end NUMINAMATH_GPT_tod_driving_time_l2387_238798


namespace NUMINAMATH_GPT_kvass_affordability_l2387_238703

theorem kvass_affordability (x y : ℚ) (hx : x + y = 1) (hxy : 1.2 * (0.5 * x + y) = 1) : 1.44 * y ≤ 1 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_kvass_affordability_l2387_238703


namespace NUMINAMATH_GPT_john_uses_six_pounds_of_vegetables_l2387_238726

-- Define the given conditions:
def pounds_of_beef_bought : ℕ := 4
def pounds_beef_used_in_soup := pounds_of_beef_bought - 1
def pounds_of_vegetables_used := 2 * pounds_beef_used_in_soup

-- Statement to prove:
theorem john_uses_six_pounds_of_vegetables : pounds_of_vegetables_used = 6 :=
by
  sorry

end NUMINAMATH_GPT_john_uses_six_pounds_of_vegetables_l2387_238726


namespace NUMINAMATH_GPT_highest_value_of_a_l2387_238794

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def highest_a : Nat :=
  7

theorem highest_value_of_a (a : Nat) 
  (last_three_digits := a * 100 + 53)
  (number := 4 * 10^8 + 3 * 10^7 + 7 * 10^6 + 5 * 10^5 + 2 * 10^4 + a * 10^3 + 5 * 10^2 + 3 * 10^1 + 9) :
  (∃ a, last_three_digits % 8 = 0 ∧ sum_of_digits number % 9 = 0 ∧ number % 12 = 0 ∧ a <= 9) → a = highest_a :=
by
  intros
  sorry

end NUMINAMATH_GPT_highest_value_of_a_l2387_238794


namespace NUMINAMATH_GPT_percentage_increase_chef_vs_dishwasher_l2387_238714

variables 
  (manager_wage chef_wage dishwasher_wage : ℝ)
  (h_manager_wage : manager_wage = 8.50)
  (h_chef_wage : chef_wage = manager_wage - 3.315)
  (h_dishwasher_wage : dishwasher_wage = manager_wage / 2)

theorem percentage_increase_chef_vs_dishwasher :
  ((chef_wage - dishwasher_wage) / dishwasher_wage) * 100 = 22 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_chef_vs_dishwasher_l2387_238714


namespace NUMINAMATH_GPT_proof_problem_l2387_238781

-- Definitions coming from the conditions
def num_large_divisions := 12
def num_small_divisions_per_large := 5
def seconds_per_small_division := 1
def seconds_per_large_division := num_small_divisions_per_large * seconds_per_small_division
def start_position := 5
def end_position := 9
def divisions_moved := end_position - start_position
def total_seconds_actual := divisions_moved * seconds_per_large_division
def total_seconds_claimed := 4

-- The theorem stating the false claim
theorem proof_problem : total_seconds_actual ≠ total_seconds_claimed :=
by {
  -- We skip the actual proof as instructed
  sorry
}

end NUMINAMATH_GPT_proof_problem_l2387_238781


namespace NUMINAMATH_GPT_number_of_pizza_varieties_l2387_238755

-- Definitions for the problem conditions
def number_of_flavors : Nat := 8
def toppings : List String := ["C", "M", "O", "J", "L"]

-- Function to count valid combinations of toppings
def valid_combinations (n : Nat) : Nat :=
  match n with
  | 1 => 5
  | 2 => 10 - 1 -- Subtracting the invalid combination (O, J)
  | 3 => 10 - 3 -- Subtracting the 3 invalid combinations containing (O, J)
  | _ => 0

def total_topping_combinations : Nat :=
  valid_combinations 1 + valid_combinations 2 + valid_combinations 3

-- The final proof stating the number of pizza varieties
theorem number_of_pizza_varieties : total_topping_combinations * number_of_flavors = 168 := by
  -- Calculation steps can be inserted here, we use sorry for now
  sorry

end NUMINAMATH_GPT_number_of_pizza_varieties_l2387_238755


namespace NUMINAMATH_GPT_puppies_start_count_l2387_238775

theorem puppies_start_count (x : ℕ) (given_away : ℕ) (left : ℕ) (h1 : given_away = 7) (h2 : left = 5) (h3 : x = given_away + left) : x = 12 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_puppies_start_count_l2387_238775


namespace NUMINAMATH_GPT_rectangle_A_plus_P_ne_162_l2387_238731

theorem rectangle_A_plus_P_ne_162 (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (A : ℕ) (P : ℕ) 
  (hA : A = a * b) (hP : P = 2 * a + 2 * b) : A + P ≠ 162 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_A_plus_P_ne_162_l2387_238731


namespace NUMINAMATH_GPT_sum_first_13_terms_l2387_238723

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}
variable (ha : a 2 + a 5 + a 9 + a 12 = 60)

theorem sum_first_13_terms :
  S 13 = 195 := sorry

end NUMINAMATH_GPT_sum_first_13_terms_l2387_238723


namespace NUMINAMATH_GPT_remaining_distance_proof_l2387_238738

/-
In a bicycle course with a total length of 10.5 kilometers (km), if Yoongi goes 1.5 kilometers (km) and then goes another 3730 meters (m), prove that the remaining distance of the course is 5270 meters.
-/

def km_to_m (km : ℝ) : ℝ := km * 1000

def total_course_length_km : ℝ := 10.5
def total_course_length_m : ℝ := km_to_m total_course_length_km

def yoongi_initial_distance_km : ℝ := 1.5
def yoongi_initial_distance_m : ℝ := km_to_m yoongi_initial_distance_km

def yoongi_additional_distance_m : ℝ := 3730

def yoongi_total_distance_m : ℝ := yoongi_initial_distance_m + yoongi_additional_distance_m

def remaining_distance_m (total_course_length_m yoongi_total_distance_m : ℝ) : ℝ :=
  total_course_length_m - yoongi_total_distance_m

theorem remaining_distance_proof : remaining_distance_m total_course_length_m yoongi_total_distance_m = 5270 := 
  sorry

end NUMINAMATH_GPT_remaining_distance_proof_l2387_238738


namespace NUMINAMATH_GPT_xiaohua_final_score_l2387_238783

-- Definitions for conditions
def education_score : ℝ := 9
def experience_score : ℝ := 7
def work_attitude_score : ℝ := 8
def weight_education : ℝ := 1
def weight_experience : ℝ := 2
def weight_attitude : ℝ := 2

-- Computation of the final score
noncomputable def final_score : ℝ :=
  education_score * (weight_education / (weight_education + weight_experience + weight_attitude)) +
  experience_score * (weight_experience / (weight_education + weight_experience + weight_attitude)) +
  work_attitude_score * (weight_attitude / (weight_education + weight_experience + weight_attitude))

-- The statement we want to prove
theorem xiaohua_final_score :
  final_score = 7.8 :=
sorry

end NUMINAMATH_GPT_xiaohua_final_score_l2387_238783


namespace NUMINAMATH_GPT_simplify_fraction_l2387_238760

theorem simplify_fraction (x : ℝ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2387_238760


namespace NUMINAMATH_GPT_lowest_positive_integer_divisible_by_primes_between_10_and_50_l2387_238796

def primes_10_to_50 : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def lcm_list (lst : List ℕ) : ℕ :=
lst.foldr Nat.lcm 1

theorem lowest_positive_integer_divisible_by_primes_between_10_and_50 :
  lcm_list primes_10_to_50 = 614889782588491410 :=
by
  sorry

end NUMINAMATH_GPT_lowest_positive_integer_divisible_by_primes_between_10_and_50_l2387_238796


namespace NUMINAMATH_GPT_algebraic_expression_value_l2387_238778

variable (x : ℝ)

theorem algebraic_expression_value (h : x^2 + 3 * x + 5 = 7) : 3 * x^2 + 9 * x - 2 = 4 :=
by
  -- This is where the detailed proof would go, but we are skipping it with sorry.
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2387_238778


namespace NUMINAMATH_GPT_find_symmetric_point_l2387_238795

structure Point := (x : Int) (y : Int)

def translate_right (p : Point) (n : Int) : Point :=
  { x := p.x + n, y := p.y }

def symmetric_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem find_symmetric_point : 
  ∀ (A B C : Point),
  A = ⟨-1, 2⟩ →
  B = translate_right A 2 →
  C = symmetric_x_axis B →
  C = ⟨1, -2⟩ :=
by
  intros A B C hA hB hC
  sorry

end NUMINAMATH_GPT_find_symmetric_point_l2387_238795


namespace NUMINAMATH_GPT_find_divisor_l2387_238764

theorem find_divisor 
    (x : ℕ) 
    (h : 83 = 9 * x + 2) : 
    x = 9 := 
  sorry

end NUMINAMATH_GPT_find_divisor_l2387_238764


namespace NUMINAMATH_GPT_pyarelal_loss_l2387_238717

theorem pyarelal_loss (total_loss : ℝ) (P : ℝ) (Ashok_capital : ℝ) (ratio_Ashok_Pyarelal : ℝ) :
  total_loss = 670 →
  Ashok_capital = P / 9 →
  ratio_Ashok_Pyarelal = 1 / 9 →
  Pyarelal_loss = 603 :=
by
  intro total_loss_eq Ashok_capital_eq ratio_eq
  sorry

end NUMINAMATH_GPT_pyarelal_loss_l2387_238717


namespace NUMINAMATH_GPT_min_value_of_function_l2387_238739

-- Define the function f
def f (x : ℝ) := 3 * x^2 - 6 * x + 9

-- State the theorem about the minimum value of the function.
theorem min_value_of_function : ∀ x : ℝ, f x ≥ 6 := by
  sorry

end NUMINAMATH_GPT_min_value_of_function_l2387_238739


namespace NUMINAMATH_GPT_product_of_x_and_y_l2387_238707

theorem product_of_x_and_y (x y a b : ℝ)
  (h1 : x = b^(3/2))
  (h2 : y = a)
  (h3 : a + a = b^2)
  (h4 : y = b)
  (h5 : a + a = b^(3/2))
  (h6 : b = 3) :
  x * y = 9 * Real.sqrt 3 := 
  sorry

end NUMINAMATH_GPT_product_of_x_and_y_l2387_238707


namespace NUMINAMATH_GPT_expression_is_integer_iff_divisible_l2387_238753

theorem expression_is_integer_iff_divisible (k n : ℤ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ m : ℤ, n = m * (k + 2) ↔ (∃ C : ℤ, (3 * n - 4 * k + 2) / (k + 2) * C = (3 * n - 4 * k + 2) / (k + 2)) :=
sorry

end NUMINAMATH_GPT_expression_is_integer_iff_divisible_l2387_238753


namespace NUMINAMATH_GPT_volume_of_parallelepiped_l2387_238727

theorem volume_of_parallelepiped 
  (m n Q : ℝ) 
  (ratio_positive : 0 < m ∧ 0 < n)
  (Q_positive : 0 < Q)
  (h_square_area : ∃ a b : ℝ, a / b = m / n ∧ (a^2 + b^2) = Q) :
  ∃ (V : ℝ), V = (m * n * Q * Real.sqrt Q) / (m^2 + n^2) :=
sorry

end NUMINAMATH_GPT_volume_of_parallelepiped_l2387_238727


namespace NUMINAMATH_GPT_bob_can_order_199_sandwiches_l2387_238751

-- Define the types of bread, meat, and cheese
def number_of_bread : ℕ := 5
def number_of_meat : ℕ := 7
def number_of_cheese : ℕ := 6

-- Define the forbidden combinations
def forbidden_turkey_swiss : ℕ := number_of_bread -- 5
def forbidden_rye_roastbeef : ℕ := number_of_cheese -- 6

-- Calculate the total sandwiches and subtract forbidden combinations
def total_sandwiches : ℕ := number_of_bread * number_of_meat * number_of_cheese
def forbidden_sandwiches : ℕ := forbidden_turkey_swiss + forbidden_rye_roastbeef

def sandwiches_bob_can_order : ℕ := total_sandwiches - forbidden_sandwiches

theorem bob_can_order_199_sandwiches :
  sandwiches_bob_can_order = 199 :=
by
  -- The calculation steps are encapsulated in definitions and are considered done
  sorry

end NUMINAMATH_GPT_bob_can_order_199_sandwiches_l2387_238751


namespace NUMINAMATH_GPT_inverse_matrix_l2387_238740

theorem inverse_matrix
  (A : Matrix (Fin 2) (Fin 2) ℚ)
  (B : Matrix (Fin 2) (Fin 2) ℚ)
  (H : A * B = ![![1, 2], ![0, 6]]) :
  A⁻¹ = ![![-1, 0], ![0, 2]] :=
sorry

end NUMINAMATH_GPT_inverse_matrix_l2387_238740


namespace NUMINAMATH_GPT_remainder_of_power_mod_l2387_238746

theorem remainder_of_power_mod (a b n : ℕ) (h_prime : Nat.Prime n) (h_a_not_div : ¬ (n ∣ a)) :
  a ^ b % n = 82 :=
by
  have : n = 379 := sorry
  have : a = 6 := sorry
  have : b = 97 := sorry
  sorry

end NUMINAMATH_GPT_remainder_of_power_mod_l2387_238746


namespace NUMINAMATH_GPT_relation_between_x_and_y_l2387_238756

open Real

noncomputable def x (t : ℝ) : ℝ := t^(1 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t^(t / (t - 1))

theorem relation_between_x_and_y (t : ℝ) (h1 : t > 0) (h2 : t ≠ 1) : (y t)^(x t) = (x t)^(y t) :=
by sorry

end NUMINAMATH_GPT_relation_between_x_and_y_l2387_238756


namespace NUMINAMATH_GPT_wire_length_after_cuts_l2387_238732

-- Given conditions as parameters
def initial_length_cm : ℝ := 23.3
def first_cut_mm : ℝ := 105
def second_cut_cm : ℝ := 4.6

-- Final statement to be proved
theorem wire_length_after_cuts (ell : ℝ) (c1 : ℝ) (c2 : ℝ) : (ell = 23.3) → (c1 = 105) → (c2 = 4.6) → 
  (ell * 10 - c1 - c2 * 10 = 82) := sorry

end NUMINAMATH_GPT_wire_length_after_cuts_l2387_238732


namespace NUMINAMATH_GPT_negation_proof_l2387_238733

theorem negation_proof :
  ¬ (∀ x : ℝ, 2 * x^2 + 1 > 0) ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_negation_proof_l2387_238733


namespace NUMINAMATH_GPT_Powerjet_pumps_250_gallons_in_30_minutes_l2387_238713

theorem Powerjet_pumps_250_gallons_in_30_minutes :
  let r := 500 -- Pump rate in gallons per hour
  let t := 1 / 2 -- Time in hours (30 minutes)
  r * t = 250 := by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_Powerjet_pumps_250_gallons_in_30_minutes_l2387_238713


namespace NUMINAMATH_GPT_washing_machines_total_pounds_l2387_238799

theorem washing_machines_total_pounds (pounds_per_machine_per_day : ℕ) (number_of_machines : ℕ)
  (h1 : pounds_per_machine_per_day = 28) (h2 : number_of_machines = 8) :
  number_of_machines * pounds_per_machine_per_day = 224 :=
by
  sorry

end NUMINAMATH_GPT_washing_machines_total_pounds_l2387_238799


namespace NUMINAMATH_GPT_cube_volume_from_surface_area_l2387_238742

theorem cube_volume_from_surface_area (A : ℝ) (h : A = 54) :
  ∃ V : ℝ, V = 27 := by
  sorry

end NUMINAMATH_GPT_cube_volume_from_surface_area_l2387_238742


namespace NUMINAMATH_GPT_sum_of_sequences_l2387_238719

theorem sum_of_sequences :
  (1 + 11 + 21 + 31 + 41) + (9 + 19 + 29 + 39 + 49) = 250 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_sequences_l2387_238719


namespace NUMINAMATH_GPT_equations_have_same_solution_l2387_238776

theorem equations_have_same_solution (x c : ℝ) 
  (h1 : 3 * x + 9 = 0) (h2 : c * x + 15 = 3) : c = 4 :=
by
  sorry

end NUMINAMATH_GPT_equations_have_same_solution_l2387_238776


namespace NUMINAMATH_GPT_unique_digit_sum_is_21_l2387_238790

theorem unique_digit_sum_is_21
  (Y E M T : ℕ)
  (YE ME : ℕ)
  (HT0 : YE = 10 * Y + E)
  (HT1 : ME = 10 * M + E)
  (H1 : YE * ME = 999)
  (H2 : Y ≠ E)
  (H3 : Y ≠ M)
  (H4 : Y ≠ T)
  (H5 : E ≠ M)
  (H6 : E ≠ T)
  (H7 : M ≠ T)
  (H8 : Y < 10)
  (H9 : E < 10)
  (H10 : M < 10)
  (H11 : T < 10) :
  Y + E + M + T = 21 :=
sorry

end NUMINAMATH_GPT_unique_digit_sum_is_21_l2387_238790


namespace NUMINAMATH_GPT_base_area_of_cone_with_slant_height_10_and_semi_lateral_surface_l2387_238758

theorem base_area_of_cone_with_slant_height_10_and_semi_lateral_surface :
  (l = 10) → (l = 2 * r) → (A = 25 * π) :=
  by
  intros l_eq_ten l_eq_two_r
  have r_is_five : r = 5 := by sorry
  have A_is_25pi : A = 25 * π := by sorry
  exact A_is_25pi

end NUMINAMATH_GPT_base_area_of_cone_with_slant_height_10_and_semi_lateral_surface_l2387_238758


namespace NUMINAMATH_GPT_max_mineral_value_l2387_238774

/-- Jane discovers three types of minerals with given weights and values:
6-pound mineral chunks worth $16 each,
3-pound mineral chunks worth $9 each,
and 2-pound mineral chunks worth $3 each. 
There are at least 30 of each type available.
She can haul a maximum of 21 pounds in her cart.
Prove that the maximum value, in dollars, that Jane can transport is $63. -/
theorem max_mineral_value : 
  ∃ (value : ℕ), (∀ (x y z : ℕ), 6 * x + 3 * y + 2 * z ≤ 21 → 
    (x ≤ 30 ∧ y ≤ 30 ∧ z ≤ 30) → value ≥ 16 * x + 9 * y + 3 * z) ∧ value = 63 :=
by sorry

end NUMINAMATH_GPT_max_mineral_value_l2387_238774


namespace NUMINAMATH_GPT_dogs_for_sale_l2387_238772

variable (D : ℕ)
def number_of_cats := D / 2
def number_of_birds := 2 * D
def number_of_fish := 3 * D
def total_animals := D + number_of_cats D + number_of_birds D + number_of_fish D

theorem dogs_for_sale (h : total_animals D = 39) : D = 6 :=
by
  sorry

end NUMINAMATH_GPT_dogs_for_sale_l2387_238772


namespace NUMINAMATH_GPT_abs_neg_three_halves_l2387_238769

theorem abs_neg_three_halves : abs (-3 / 2 : ℚ) = 3 / 2 := 
by 
  -- Here we would have the steps that show the computation
  -- Applying the definition of absolute value to remove the negative sign
  -- This simplifies to 3 / 2
  sorry

end NUMINAMATH_GPT_abs_neg_three_halves_l2387_238769


namespace NUMINAMATH_GPT_real_solution_exists_l2387_238793

theorem real_solution_exists (x : ℝ) (h1: x ≠ 5) (h2: x ≠ 6) : 
  (x = 1 ∨ x = 2 ∨ x = 3) ↔ 
  ((x - 1) * (x - 5) * (x - 3) * (x - 6) * (x - 3) * (x - 5) * (x - 1) /
  ((x - 5) * (x - 6) * (x - 5)) = 1) := 
by 
  sorry

end NUMINAMATH_GPT_real_solution_exists_l2387_238793


namespace NUMINAMATH_GPT_no_such_triples_l2387_238761

theorem no_such_triples 
  (a b c : ℕ) (h₁ : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h₂ : ¬ ∃ k, k ∣ a + c ∧ k ∣ b + c ∧ k ∣ a + b) 
  (h₃ : c^2 ∣ a + b) 
  (h₄ : b^2 ∣ a + c) 
  (h₅ : a^2 ∣ b + c) : 
  false :=
sorry

end NUMINAMATH_GPT_no_such_triples_l2387_238761


namespace NUMINAMATH_GPT_BD_length_l2387_238721

theorem BD_length
  (A B C D : Type)
  (dist_AC : ℝ := 10)
  (dist_BC : ℝ := 10)
  (dist_AD : ℝ := 12)
  (dist_CD : ℝ := 5) : (BD : ℝ) = 95 / 12 :=
by
  sorry

end NUMINAMATH_GPT_BD_length_l2387_238721


namespace NUMINAMATH_GPT_total_oranges_in_box_l2387_238712

def initial_oranges_in_box : ℝ := 55.0
def oranges_added_by_susan : ℝ := 35.0

theorem total_oranges_in_box :
  initial_oranges_in_box + oranges_added_by_susan = 90.0 := by
  sorry

end NUMINAMATH_GPT_total_oranges_in_box_l2387_238712


namespace NUMINAMATH_GPT_slope_of_line_l2387_238725

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line l intersecting the parabola C at points A and B
def line (k x : ℝ) : ℝ := k * (x - 1)

-- Condition based on the intersection and the given relationship 2 * (BF) = FA
def intersection_condition (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ x1 x2 y1 y2,
    A = (x1, y1) ∧ B = (x2, y2) ∧
    parabola x1 y1 ∧ parabola x2 y2 ∧
    (y1 = line k x1) ∧ (y2 = line k x2) ∧
    2 * (dist (x2, y2) focus) = dist focus (x1, y1)

-- The main theorem to be proven
theorem slope_of_line (k : ℝ) (A B : ℝ × ℝ) :
  intersection_condition k A B → k = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_slope_of_line_l2387_238725


namespace NUMINAMATH_GPT_smallest_n_divisible_11_remainder_1_l2387_238710

theorem smallest_n_divisible_11_remainder_1 :
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 1) ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 7 = 1) ∧ (n % 11 = 0) ∧ 
    (∀ m : ℕ, (m % 2 = 1) ∧ (m % 3 = 1) ∧ (m % 4 = 1) ∧ (m % 5 = 1) ∧ (m % 7 = 1) ∧ (m % 11 = 0) → 2521 ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_11_remainder_1_l2387_238710


namespace NUMINAMATH_GPT_circles_exceeding_n_squared_l2387_238741

noncomputable def num_circles (n : ℕ) : ℕ :=
  if n >= 8 then 
    5 * n + 4 * (n - 1)
  else 
    n * n

theorem circles_exceeding_n_squared (n : ℕ) (hn : n ≥ 8) : num_circles n > n^2 := 
by {
  sorry
}

end NUMINAMATH_GPT_circles_exceeding_n_squared_l2387_238741


namespace NUMINAMATH_GPT_natalia_crates_l2387_238784

noncomputable def total_items (novels comics documentaries albums : ℕ) : ℕ :=
  novels + comics + documentaries + albums

noncomputable def crates_needed (total_items items_per_crate : ℕ) : ℕ :=
  (total_items + items_per_crate - 1) / items_per_crate

theorem natalia_crates : crates_needed (total_items 145 271 419 209) 9 = 117 := by
  sorry

end NUMINAMATH_GPT_natalia_crates_l2387_238784


namespace NUMINAMATH_GPT_number_of_teams_l2387_238744

-- Define the necessary conditions and variables
variable (n : ℕ)
variable (num_games : ℕ)

-- Define the condition that each team plays each other team exactly once 
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

-- The main theorem to prove
theorem number_of_teams (h : total_games n = 91) : n = 14 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l2387_238744


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l2387_238706

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l2387_238706


namespace NUMINAMATH_GPT_kenya_peanuts_count_l2387_238766

def peanuts_jose : ℕ := 85
def diff_kenya_jose : ℕ := 48
def peanuts_kenya : ℕ := peanuts_jose + diff_kenya_jose

theorem kenya_peanuts_count : peanuts_kenya = 133 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_kenya_peanuts_count_l2387_238766


namespace NUMINAMATH_GPT_weight_of_2019_is_correct_l2387_238770

-- Declare the conditions as definitions to be used in Lean 4
def stick_weight : Real := 0.5
def digit_to_sticks (n : Nat) : Nat :=
  match n with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- other digits aren't considered in this problem

-- Calculate the total weight of the number 2019
def weight_of_2019 : Real :=
  (digit_to_sticks 2 + digit_to_sticks 0 + digit_to_sticks 1 + digit_to_sticks 9) * stick_weight

-- Statement to prove the weight of the number 2019
theorem weight_of_2019_is_correct : weight_of_2019 = 9.5 := by
  sorry

end NUMINAMATH_GPT_weight_of_2019_is_correct_l2387_238770


namespace NUMINAMATH_GPT_batsman_average_increases_l2387_238701

theorem batsman_average_increases
  (score_17th: ℕ)
  (avg_increase: ℕ)
  (initial_avg: ℕ)
  (final_avg: ℕ)
  (initial_innings: ℕ):
  score_17th = 74 →
  avg_increase = 3 →
  initial_innings = 16 →
  initial_avg = 23 →
  final_avg = initial_avg + avg_increase →
  (final_avg * (initial_innings + 1) = score_17th + (initial_avg * initial_innings)) →
  final_avg = 26 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_increases_l2387_238701


namespace NUMINAMATH_GPT_find_x_from_conditions_l2387_238705

theorem find_x_from_conditions 
  (x y : ℕ) 
  (h1 : 1 ≤ x)
  (h2 : x ≤ 100)
  (h3 : 1 ≤ y)
  (h4 : y ≤ 100)
  (h5 : y > x)
  (h6 : (21 + 45 + 77 + 2 * x + y) / 6 = 2 * x) 
  : x = 16 := 
sorry

end NUMINAMATH_GPT_find_x_from_conditions_l2387_238705


namespace NUMINAMATH_GPT_x_minus_y_solution_l2387_238768

theorem x_minus_y_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end NUMINAMATH_GPT_x_minus_y_solution_l2387_238768


namespace NUMINAMATH_GPT_percentage_increase_l2387_238791

theorem percentage_increase (P Q : ℝ)
  (price_decreased : ∀ P', P' = 0.80 * P)
  (revenue_increased : ∀ R R', R = P * Q ∧ R' = 1.28000000000000025 * R)
  : ∃ Q', Q' = 1.6000000000000003125 * Q :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l2387_238791


namespace NUMINAMATH_GPT_quadratic_function_example_l2387_238788

theorem quadratic_function_example : ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x^2 + b * x + c = 0) ↔ (x = 1 ∨ x = 5)) ∧ 
  (a * 3^2 + b * 3 + c = 8) ∧ 
  (a = -2 ∧ b = 12 ∧ c = -10) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_example_l2387_238788


namespace NUMINAMATH_GPT_value_of_y_l2387_238773

theorem value_of_y : 
  ∀ y : ℚ, y = (2010^2 - 2010 + 1 : ℚ) / 2010 → y = (2009 + 1 / 2010 : ℚ) := by
  sorry

end NUMINAMATH_GPT_value_of_y_l2387_238773


namespace NUMINAMATH_GPT_point_on_line_l2387_238735

theorem point_on_line : 
  ∃ t : ℚ, (3 * t + 1 = 0) ∧ ((2 - 4) / (t - 1) = (7 - 4) / (3 - 1)) :=
by
  sorry

end NUMINAMATH_GPT_point_on_line_l2387_238735


namespace NUMINAMATH_GPT_sum_remainders_mod_15_l2387_238729

theorem sum_remainders_mod_15 (a b c : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) :
    (a + b + c) % 15 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_remainders_mod_15_l2387_238729


namespace NUMINAMATH_GPT_intersection_M_N_l2387_238718

noncomputable def M : Set ℝ := { x | x^2 - x ≤ 0 }
noncomputable def N : Set ℝ := { x | 1 - abs x > 0 }
noncomputable def intersection : Set ℝ := { x | x ≥ 0 ∧ x < 1 }

theorem intersection_M_N : M ∩ N = intersection :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2387_238718


namespace NUMINAMATH_GPT_number_of_shoes_outside_library_l2387_238747

-- Define the conditions
def number_of_people : ℕ := 10
def shoes_per_person : ℕ := 2

-- Define the proof that the number of shoes kept outside the library is 20.
theorem number_of_shoes_outside_library : number_of_people * shoes_per_person = 20 :=
by
  -- Proof left as sorry because the proof steps are not required
  sorry

end NUMINAMATH_GPT_number_of_shoes_outside_library_l2387_238747


namespace NUMINAMATH_GPT_plates_are_multiple_of_eleven_l2387_238771

theorem plates_are_multiple_of_eleven
    (P : ℕ)    -- Number of plates
    (S : ℕ := 33)    -- Number of spoons
    (g : ℕ := 11)    -- Greatest number of groups
    (hS : S % g = 0)    -- Condition: All spoons can be divided into these groups evenly
    (hP : ∀ (k : ℕ), P = k * g) : ∃ x : ℕ, P = 11 * x :=
by
  sorry

end NUMINAMATH_GPT_plates_are_multiple_of_eleven_l2387_238771


namespace NUMINAMATH_GPT_james_fraction_of_pizza_slices_l2387_238789

theorem james_fraction_of_pizza_slices :
  (2 * 6 = 12) ∧ (8 / 12 = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_james_fraction_of_pizza_slices_l2387_238789


namespace NUMINAMATH_GPT_tan_alpha_calc_l2387_238779

theorem tan_alpha_calc (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α) / Real.cos α ^ 2) = 6 :=
by sorry

end NUMINAMATH_GPT_tan_alpha_calc_l2387_238779


namespace NUMINAMATH_GPT_trajectory_sufficient_not_necessary_l2387_238750

-- Define for any point P if its trajectory is y = |x|
def trajectory (P : ℝ × ℝ) : Prop :=
  P.2 = abs P.1

-- Define for any point P if its distances to the coordinate axes are equal
def equal_distances (P : ℝ × ℝ) : Prop :=
  abs P.1 = abs P.2

-- The main statement: prove that the trajectory is a sufficient but not necessary condition for equal_distances
theorem trajectory_sufficient_not_necessary (P : ℝ × ℝ) :
  trajectory P → equal_distances P ∧ ¬(equal_distances P → trajectory P) := 
sorry

end NUMINAMATH_GPT_trajectory_sufficient_not_necessary_l2387_238750


namespace NUMINAMATH_GPT_find_quotient_l2387_238700

-- Define the problem variables and conditions
def larger_number : ℕ := 1620
def smaller_number : ℕ := larger_number - 1365
def remainder : ℕ := 15

-- Define the proof problem
theorem find_quotient :
  larger_number = smaller_number * 6 + remainder :=
sorry

end NUMINAMATH_GPT_find_quotient_l2387_238700


namespace NUMINAMATH_GPT_exists_fifth_degree_polynomial_l2387_238754

noncomputable def p (x : ℝ) : ℝ :=
  12.4 * (x^5 - 1.38 * x^3 + 0.38 * x)

theorem exists_fifth_degree_polynomial :
  (∃ x1 x2 : ℝ, -1 < x1 ∧ x1 < 1 ∧ -1 < x2 ∧ x2 < 1 ∧ x1 ≠ x2 ∧ 
    p x1 = 1 ∧ p x2 = -1 ∧ p (-1) = 0 ∧ p 1 = 0) :=
  sorry

end NUMINAMATH_GPT_exists_fifth_degree_polynomial_l2387_238754


namespace NUMINAMATH_GPT_problem_statement_l2387_238792

theorem problem_statement (a b : ℝ) (h1 : 1/a < 1/b) (h2 : 1/b < 0) :
  (a + b < a * b) ∧ ¬(a^2 > b^2) ∧ ¬(a < b) ∧ (b/a + a/b > 2) := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2387_238792


namespace NUMINAMATH_GPT_price_reduction_correct_l2387_238737

theorem price_reduction_correct (P : ℝ) : 
  let first_reduction := 0.92 * P
  let second_reduction := first_reduction * 0.90
  second_reduction = 0.828 * P := 
by 
  sorry

end NUMINAMATH_GPT_price_reduction_correct_l2387_238737


namespace NUMINAMATH_GPT_put_balls_in_boxes_l2387_238724

def balls : ℕ := 5
def boxes : ℕ := 3
def number_of_ways : ℕ := boxes ^ balls

theorem put_balls_in_boxes : number_of_ways = 243 := by
  sorry

end NUMINAMATH_GPT_put_balls_in_boxes_l2387_238724


namespace NUMINAMATH_GPT_oak_grove_libraries_total_books_l2387_238711

theorem oak_grove_libraries_total_books :
  let publicLibraryBooks := 1986
  let schoolLibrariesBooks := 5106
  let communityCollegeLibraryBooks := 3294.5
  let medicalLibraryBooks := 1342.25
  let lawLibraryBooks := 2785.75
  publicLibraryBooks + schoolLibrariesBooks + communityCollegeLibraryBooks + medicalLibraryBooks + lawLibraryBooks = 15514.5 :=
by
  sorry

end NUMINAMATH_GPT_oak_grove_libraries_total_books_l2387_238711


namespace NUMINAMATH_GPT_smallest_m_n_sum_l2387_238708

noncomputable def smallestPossibleSum (m n : ℕ) : ℕ :=
  m + n

theorem smallest_m_n_sum :
  ∃ (m n : ℕ), (m > 1) ∧ (m * n * (2021 * (m^2 - 1)) = 2021 * m * m * n) ∧ smallestPossibleSum m n = 4323 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_n_sum_l2387_238708


namespace NUMINAMATH_GPT_inclination_angle_of_line_l2387_238709

def line_equation (x y : ℝ) : Prop := x * (Real.tan (Real.pi / 3)) + y + 2 = 0

theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : 
  ∃ α : ℝ, α = 2 * Real.pi / 3 ∧ 0 ≤ α ∧ α < Real.pi := by
  sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l2387_238709


namespace NUMINAMATH_GPT_salary_january_l2387_238786

theorem salary_january
  (J F M A May : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8300)
  (h3 : May = 6500) :
  J = 5300 :=
by
  sorry

end NUMINAMATH_GPT_salary_january_l2387_238786


namespace NUMINAMATH_GPT_M_inter_N_eq_l2387_238759

def set_M (x : ℝ) : Prop := x^2 - 3 * x < 0
def set_N (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 4

def M := { x : ℝ | set_M x }
def N := { x : ℝ | set_N x }

theorem M_inter_N_eq : M ∩ N = { x | 1 ≤ x ∧ x < 3 } :=
by sorry

end NUMINAMATH_GPT_M_inter_N_eq_l2387_238759


namespace NUMINAMATH_GPT_more_calories_per_dollar_l2387_238787

-- The conditions given in the problem as definitions
def price_burritos : ℕ := 6
def price_burgers : ℕ := 8
def calories_per_burrito : ℕ := 120
def calories_per_burger : ℕ := 400
def num_burritos : ℕ := 10
def num_burgers : ℕ := 5

-- The theorem stating the mathematically equivalent proof problem
theorem more_calories_per_dollar : 
  (num_burgers * calories_per_burger / price_burgers) - (num_burritos * calories_per_burrito / price_burritos) = 50 :=
by
  sorry

end NUMINAMATH_GPT_more_calories_per_dollar_l2387_238787


namespace NUMINAMATH_GPT_dan_balloons_correct_l2387_238748

-- Define the initial conditions
def sam_initial_balloons : Float := 46.0
def sam_given_fred_balloons : Float := 10.0
def total_balloons : Float := 52.0

-- Calculate Sam's remaining balloons
def sam_current_balloons : Float := sam_initial_balloons - sam_given_fred_balloons

-- Define the target: Dan's balloons
def dan_balloons := total_balloons - sam_current_balloons

-- Statement to prove
theorem dan_balloons_correct : dan_balloons = 16.0 := sorry

end NUMINAMATH_GPT_dan_balloons_correct_l2387_238748


namespace NUMINAMATH_GPT_both_games_players_l2387_238782

theorem both_games_players (kabadi_players kho_kho_only total_players both_games : ℕ)
  (h_kabadi : kabadi_players = 10)
  (h_kho_kho_only : kho_kho_only = 15)
  (h_total : total_players = 25)
  (h_equation : kabadi_players + kho_kho_only + both_games = total_players) :
  both_games = 0 :=
by
  -- question == answer given conditions
  sorry

end NUMINAMATH_GPT_both_games_players_l2387_238782
