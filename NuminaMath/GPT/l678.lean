import Mathlib

namespace NUMINAMATH_GPT_cat_food_more_than_dog_food_l678_67801

-- Define the number of packages and cans per package for cat food
def cat_food_packages : ℕ := 9
def cat_food_cans_per_package : ℕ := 10

-- Define the number of packages and cans per package for dog food
def dog_food_packages : ℕ := 7
def dog_food_cans_per_package : ℕ := 5

-- Total number of cans of cat food
def total_cat_food_cans : ℕ := cat_food_packages * cat_food_cans_per_package

-- Total number of cans of dog food
def total_dog_food_cans : ℕ := dog_food_packages * dog_food_cans_per_package

-- Prove the difference between the total cans of cat food and total cans of dog food
theorem cat_food_more_than_dog_food : total_cat_food_cans - total_dog_food_cans = 55 := by
  -- Provide the calculation results directly
  have h_cat : total_cat_food_cans = 90 := by rfl
  have h_dog : total_dog_food_cans = 35 := by rfl
  calc
    total_cat_food_cans - total_dog_food_cans = 90 - 35 := by rw [h_cat, h_dog]
    _ = 55 := rfl

end NUMINAMATH_GPT_cat_food_more_than_dog_food_l678_67801


namespace NUMINAMATH_GPT_eliza_total_clothes_l678_67815

def time_per_blouse : ℕ := 15
def time_per_dress : ℕ := 20
def blouse_time : ℕ := 2 * 60   -- 2 hours in minutes
def dress_time : ℕ := 3 * 60    -- 3 hours in minutes

theorem eliza_total_clothes :
  (blouse_time / time_per_blouse) + (dress_time / time_per_dress) = 17 :=
by
  sorry

end NUMINAMATH_GPT_eliza_total_clothes_l678_67815


namespace NUMINAMATH_GPT_don_can_have_more_rum_l678_67881

-- Definitions based on conditions:
def given_rum : ℕ := 10
def max_consumption_rate : ℕ := 3
def already_had : ℕ := 12

-- Maximum allowed consumption calculation:
def max_allowed_rum : ℕ := max_consumption_rate * given_rum

-- Remaining rum calculation:
def remaining_rum : ℕ := max_allowed_rum - already_had

-- Proof statement of the problem:
theorem don_can_have_more_rum : remaining_rum = 18 := by
  -- Let's compute directly:
  have h1 : max_allowed_rum = 30 := by
    simp [max_allowed_rum, max_consumption_rate, given_rum]

  have h2 : remaining_rum = 18 := by
    simp [remaining_rum, h1, already_had]

  exact h2

end NUMINAMATH_GPT_don_can_have_more_rum_l678_67881


namespace NUMINAMATH_GPT_catalyst_second_addition_is_882_l678_67811

-- Constants for the problem
def lower_bound : ℝ := 500
def upper_bound : ℝ := 1500
def golden_ratio_method : ℝ := 0.618

-- Calculated values
def first_addition : ℝ := lower_bound + golden_ratio_method * (upper_bound - lower_bound)
def second_bound : ℝ := first_addition - lower_bound
def second_addition : ℝ := lower_bound + golden_ratio_method * second_bound

theorem catalyst_second_addition_is_882 :
  lower_bound = 500 → upper_bound = 1500 → golden_ratio_method = 0.618 → second_addition = 882 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_catalyst_second_addition_is_882_l678_67811


namespace NUMINAMATH_GPT_range_of_k_l678_67836

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 3| + |x - 1| > k) ↔ k < 4 :=
by sorry

end NUMINAMATH_GPT_range_of_k_l678_67836


namespace NUMINAMATH_GPT_largest_even_two_digit_largest_odd_two_digit_l678_67855

-- Define conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Theorem statements
theorem largest_even_two_digit : ∃ n, is_two_digit n ∧ is_even n ∧ ∀ m, is_two_digit m ∧ is_even m → m ≤ n := 
sorry

theorem largest_odd_two_digit : ∃ n, is_two_digit n ∧ is_odd n ∧ ∀ m, is_two_digit m ∧ is_odd m → m ≤ n := 
sorry

end NUMINAMATH_GPT_largest_even_two_digit_largest_odd_two_digit_l678_67855


namespace NUMINAMATH_GPT_union_A_B_l678_67867

open Set

-- Define the sets A and B
def setA : Set ℝ := { x | abs x < 3 }
def setB : Set ℝ := { x | x - 1 ≤ 0 }

-- State the theorem we want to prove
theorem union_A_B : setA ∪ setB = { x : ℝ | x < 3 } :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_union_A_B_l678_67867


namespace NUMINAMATH_GPT_batsman_average_increase_l678_67846

theorem batsman_average_increase 
    (A : ℝ) 
    (h1 : 11 * A + 80 = 12 * 47) : 
    47 - A = 3 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_batsman_average_increase_l678_67846


namespace NUMINAMATH_GPT_samuel_faster_than_sarah_l678_67873

theorem samuel_faster_than_sarah
  (efficiency_samuel : ℝ := 0.90)
  (efficiency_sarah : ℝ := 0.75)
  (efficiency_tim : ℝ := 0.80)
  (time_tim : ℝ := 45)
  : (time_tim * efficiency_tim / efficiency_sarah) - (time_tim * efficiency_tim / efficiency_samuel) = 8 :=
by
  sorry

end NUMINAMATH_GPT_samuel_faster_than_sarah_l678_67873


namespace NUMINAMATH_GPT_checker_moves_10_cells_l678_67831

theorem checker_moves_10_cells :
  ∃ (a : ℕ → ℕ), a 1 = 1 ∧ a 2 = 2 ∧ (∀ n, n ≥ 3 → a n = a (n-1) + a (n-2)) ∧ a 10 = 89 :=
by
  -- mathematical proof goes here
  sorry

end NUMINAMATH_GPT_checker_moves_10_cells_l678_67831


namespace NUMINAMATH_GPT_expected_value_of_winnings_after_one_flip_l678_67806

-- Definitions based on conditions from part a)
def prob_heads : ℚ := 1 / 3
def prob_tails : ℚ := 2 / 3
def win_heads : ℚ := 3
def lose_tails : ℚ := -2

-- The statement to prove:
theorem expected_value_of_winnings_after_one_flip :
  prob_heads * win_heads + prob_tails * lose_tails = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_winnings_after_one_flip_l678_67806


namespace NUMINAMATH_GPT_mark_and_carolyn_total_l678_67829

theorem mark_and_carolyn_total (m c : ℝ) (hm : m = 3 / 4) (hc : c = 3 / 10) :
    m + c = 1.05 :=
by
  sorry

end NUMINAMATH_GPT_mark_and_carolyn_total_l678_67829


namespace NUMINAMATH_GPT_not_diff_of_squares_count_l678_67868

theorem not_diff_of_squares_count :
  ∃ n : ℕ, n ≤ 1000 ∧
  (∀ k : ℕ, k > 0 ∧ k ≤ 1000 → (∃ a b : ℤ, k = a^2 - b^2 ↔ k % 4 ≠ 2)) ∧
  n = 250 :=
sorry

end NUMINAMATH_GPT_not_diff_of_squares_count_l678_67868


namespace NUMINAMATH_GPT_proposition_1_proposition_4_l678_67848

-- Definitions
variable {a b c : Type} (Line : Type) (Plane : Type)
variable (a b c : Line) (γ : Plane)

-- Given conditions
variable (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)

-- Parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Propositions to prove
theorem proposition_1 (H1 : parallel a b) (H2 : parallel b c) : parallel a c := sorry

theorem proposition_4 (H3 : perpendicular a γ) (H4 : perpendicular b γ) : parallel a b := sorry

end NUMINAMATH_GPT_proposition_1_proposition_4_l678_67848


namespace NUMINAMATH_GPT_balloon_arrangements_l678_67893

theorem balloon_arrangements : 
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / (Nat.factorial k1 * Nat.factorial k2) = 1260 := 
by
  let n := 7
  let k1 := 2
  let k2 := 2
  sorry

end NUMINAMATH_GPT_balloon_arrangements_l678_67893


namespace NUMINAMATH_GPT_common_chord_eq_l678_67887

theorem common_chord_eq (x y : ℝ) :
  x^2 + y^2 + 2*x = 0 →
  x^2 + y^2 - 4*y = 0 →
  x + 2*y = 0 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_common_chord_eq_l678_67887


namespace NUMINAMATH_GPT_max_tickets_jane_can_buy_l678_67889

def ticket_price : ℝ := 15.75
def processing_fee : ℝ := 1.25
def jane_money : ℝ := 150.00

theorem max_tickets_jane_can_buy : ⌊jane_money / (ticket_price + processing_fee)⌋ = 8 := 
by
  sorry

end NUMINAMATH_GPT_max_tickets_jane_can_buy_l678_67889


namespace NUMINAMATH_GPT_trig_identity_proof_l678_67802

noncomputable def trig_identity (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) : ℝ :=
  (Real.sin (2 * α)) / (Real.cos (2 * β))

theorem trig_identity_proof (α β : ℝ) (h1 : Real.tan (α + β) = 1) (h2 : Real.tan (α - β) = 2) :
  trig_identity α β h1 h2 = 1 :=
sorry

end NUMINAMATH_GPT_trig_identity_proof_l678_67802


namespace NUMINAMATH_GPT_no_integer_pair_satisfies_conditions_l678_67874

theorem no_integer_pair_satisfies_conditions :
  ¬ ∃ (x y : ℤ), x = x^2 + y^2 + 1 ∧ y = 3 * x * y := 
by
  sorry

end NUMINAMATH_GPT_no_integer_pair_satisfies_conditions_l678_67874


namespace NUMINAMATH_GPT_simplify_expression_l678_67872

theorem simplify_expression (a : ℝ) (h : a ≠ 0) : (a^9 * a^15) / a^3 = a^21 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l678_67872


namespace NUMINAMATH_GPT_problem_statement_l678_67824

theorem problem_statement (x : ℝ) (h : (2024 - x)^2 + (2022 - x)^2 = 4038) : 
  (2024 - x) * (2022 - x) = 2017 :=
sorry

end NUMINAMATH_GPT_problem_statement_l678_67824


namespace NUMINAMATH_GPT_discount_double_time_l678_67888

theorem discount_double_time (TD FV : ℝ) (h1 : TD = 10) (h2 : FV = 110) : 
  2 * TD = 20 :=
by
  sorry

end NUMINAMATH_GPT_discount_double_time_l678_67888


namespace NUMINAMATH_GPT_find_b_l678_67869

theorem find_b
  (a b c : ℤ)
  (h1 : a + 5 = b)
  (h2 : 5 + b = c)
  (h3 : b + c = a) : b = -10 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l678_67869


namespace NUMINAMATH_GPT_total_number_of_eggs_l678_67891

theorem total_number_of_eggs 
  (cartons : ℕ) 
  (eggs_per_carton_length : ℕ) 
  (eggs_per_carton_width : ℕ)
  (egg_position_from_front : ℕ)
  (egg_position_from_back : ℕ)
  (egg_position_from_left : ℕ)
  (egg_position_from_right : ℕ) :
  cartons = 28 →
  egg_position_from_front = 14 →
  egg_position_from_back = 20 →
  egg_position_from_left = 3 →
  egg_position_from_right = 2 →
  eggs_per_carton_length = egg_position_from_front + egg_position_from_back - 1 →
  eggs_per_carton_width = egg_position_from_left + egg_position_from_right - 1 →
  cartons * (eggs_per_carton_length * eggs_per_carton_width) = 3696 := 
  by 
  intros
  sorry

end NUMINAMATH_GPT_total_number_of_eggs_l678_67891


namespace NUMINAMATH_GPT_man_speed_l678_67800

theorem man_speed (train_length : ℝ) (time_to_cross : ℝ) (train_speed_kmph : ℝ) 
  (h1 : train_length = 100) (h2 : time_to_cross = 6) (h3 : train_speed_kmph = 54.99520038396929) : 
  ∃ man_speed : ℝ, man_speed = 16.66666666666667 - 15.27644455165814 :=
by sorry

end NUMINAMATH_GPT_man_speed_l678_67800


namespace NUMINAMATH_GPT_total_miles_run_correct_l678_67843

-- Define the number of people on the sprint team and the miles each person runs.
def number_of_people : Float := 150.0
def miles_per_person : Float := 5.0

-- Define the total miles run by the sprint team.
def total_miles_run : Float := number_of_people * miles_per_person

-- State the theorem to prove that the total miles run is equal to 750.0 miles.
theorem total_miles_run_correct : total_miles_run = 750.0 := sorry

end NUMINAMATH_GPT_total_miles_run_correct_l678_67843


namespace NUMINAMATH_GPT_y_intercept_of_line_l678_67838

theorem y_intercept_of_line (m x1 y1 : ℝ) (x_intercept : x1 = 4) (y_intercept_at_x1_zero : y1 = 0) (m_value : m = -3) :
  ∃ b : ℝ, (∀ x y : ℝ, y = m * x + b ∧ x = 0 → y = b) ∧ b = 12 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l678_67838


namespace NUMINAMATH_GPT_race_time_A_l678_67837

theorem race_time_A (v_A v_B : ℝ) (t_A t_B : ℝ) (hA_time_eq : v_A = 1000 / t_A)
  (hB_time_eq : v_B = 960 / t_B) (hA_beats_B_40m : 1000 / v_A = 960 / v_B)
  (hA_beats_B_8s : t_B = t_A + 8) : t_A = 200 := 
  sorry

end NUMINAMATH_GPT_race_time_A_l678_67837


namespace NUMINAMATH_GPT_ratio_of_areas_l678_67882

variable (s' : ℝ) -- Let s' be the side length of square S'

def area_square : ℝ := s' ^ 2
def length_longer_side_rectangle : ℝ := 1.15 * s'
def length_shorter_side_rectangle : ℝ := 0.95 * s'
def area_rectangle : ℝ := length_longer_side_rectangle s' * length_shorter_side_rectangle s'

theorem ratio_of_areas :
  (area_rectangle s') / (area_square s') = (10925 / 10000) :=
by
  -- skip the proof for now
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l678_67882


namespace NUMINAMATH_GPT_remainder_of_division_l678_67839

noncomputable def dividend : Polynomial ℤ := Polynomial.C 1 * Polynomial.X^4 +
                                             Polynomial.C 3 * Polynomial.X^2 + 
                                             Polynomial.C (-4)

noncomputable def divisor : Polynomial ℤ := Polynomial.C 1 * Polynomial.X^3 +
                                            Polynomial.C (-3)

theorem remainder_of_division :
  Polynomial.modByMonic dividend divisor = Polynomial.C 3 * Polynomial.X^2 +
                                            Polynomial.C 3 * Polynomial.X +
                                            Polynomial.C (-4) :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l678_67839


namespace NUMINAMATH_GPT_train_crossing_time_l678_67823

-- Conditions
def length_train1 : ℕ := 200 -- Train 1 length in meters
def length_train2 : ℕ := 160 -- Train 2 length in meters
def speed_train1 : ℕ := 68 -- Train 1 speed in kmph
def speed_train2 : ℕ := 40 -- Train 2 speed in kmph

-- Conversion factors and formulas
def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600
def total_distance (l1 l2 : ℕ) := l1 + l2
def relative_speed (s1 s2 : ℕ) := kmph_to_mps (s1 + s2)
def crossing_time (dist speed : ℕ) := dist / speed

-- Proof statement
theorem train_crossing_time : 
  crossing_time (total_distance length_train1 length_train2) (relative_speed speed_train1 speed_train2) = 12 := by sorry

end NUMINAMATH_GPT_train_crossing_time_l678_67823


namespace NUMINAMATH_GPT_real_number_a_l678_67856

theorem real_number_a (a : ℝ) (ha : ∃ b : ℝ, z = 0 + bi) : a = 1 :=
sorry

end NUMINAMATH_GPT_real_number_a_l678_67856


namespace NUMINAMATH_GPT_ratio_of_x_intercepts_l678_67885

theorem ratio_of_x_intercepts (b s t : ℝ) (hb : b ≠ 0) (h1 : s = -b / 8) (h2 : t = -b / 4) : s / t = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_x_intercepts_l678_67885


namespace NUMINAMATH_GPT_decagon_area_bisection_ratio_l678_67895

theorem decagon_area_bisection_ratio
  (decagon_area : ℝ := 12)
  (below_PQ_area : ℝ := 6)
  (trapezoid_area : ℝ := 4)
  (b1 : ℝ := 3)
  (b2 : ℝ := 6)
  (h : ℝ := 8/9)
  (XQ : ℝ := 4)
  (QY : ℝ := 2) :
  (XQ / QY = 2) :=
by
  sorry

end NUMINAMATH_GPT_decagon_area_bisection_ratio_l678_67895


namespace NUMINAMATH_GPT_first_scenario_machines_l678_67813

theorem first_scenario_machines (M : ℕ) (h1 : 20 = 10 * 2 * M) (h2 : 140 = 20 * 17.5 * 2) : M = 5 :=
by sorry

end NUMINAMATH_GPT_first_scenario_machines_l678_67813


namespace NUMINAMATH_GPT_no_real_solution_exists_l678_67871

theorem no_real_solution_exists:
  ¬ ∃ (x y z : ℝ), (x ^ 2 + 4 * y * z + 2 * z = 0) ∧
                   (x + 2 * x * y + 2 * z ^ 2 = 0) ∧
                   (2 * x * z + y ^ 2 + y + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_exists_l678_67871


namespace NUMINAMATH_GPT_difference_of_numbers_l678_67898

-- Definitions for the digits and the numbers formed
def digits : List ℕ := [5, 3, 1, 4]

def largestNumber : ℕ := 5431
def leastNumber : ℕ := 1345

-- The problem statement
theorem difference_of_numbers (digits : List ℕ) (n_largest n_least : ℕ) :
  n_largest = 5431 ∧ n_least = 1345 → (n_largest - n_least) = 4086 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l678_67898


namespace NUMINAMATH_GPT_minimum_value_of_quad_func_l678_67861

def quad_func (x : ℝ) : ℝ :=
  2 * x^2 - 8 * x + 15

theorem minimum_value_of_quad_func :
  (∀ x : ℝ, quad_func 2 ≤ quad_func x) ∧ (quad_func 2 = 7) :=
by
  -- sorry to skip proof
  sorry

end NUMINAMATH_GPT_minimum_value_of_quad_func_l678_67861


namespace NUMINAMATH_GPT_smallest_n_reducible_fraction_l678_67849

theorem smallest_n_reducible_fraction : ∀ (n : ℕ), (∃ (k : ℕ), gcd (n - 13) (5 * n + 6) = k ∧ k > 1) ↔ n = 84 := by
  sorry

end NUMINAMATH_GPT_smallest_n_reducible_fraction_l678_67849


namespace NUMINAMATH_GPT_homework_points_l678_67864

variable (H Q T : ℕ)

theorem homework_points (h1 : T = 4 * Q)
                        (h2 : Q = H + 5)
                        (h3 : H + Q + T = 265) : 
  H = 40 :=
sorry

end NUMINAMATH_GPT_homework_points_l678_67864


namespace NUMINAMATH_GPT_log_sum_eval_l678_67878

theorem log_sum_eval :
  (Real.logb 5 625 + Real.logb 5 5 - Real.logb 5 (1 / 25)) = 7 :=
by
  have h1 : Real.logb 5 625 = 4 := by sorry
  have h2 : Real.logb 5 5 = 1 := by sorry
  have h3 : Real.logb 5 (1 / 25) = -2 := by sorry
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_log_sum_eval_l678_67878


namespace NUMINAMATH_GPT_my_cousin_reading_time_l678_67841

-- Define the conditions
def reading_time_me_hours : ℕ := 3
def reading_speed_ratio : ℕ := 5
def reading_time_me_min : ℕ := reading_time_me_hours * 60

-- Define the statement to be proved
theorem my_cousin_reading_time : (reading_time_me_min / reading_speed_ratio) = 36 := by
  sorry

end NUMINAMATH_GPT_my_cousin_reading_time_l678_67841


namespace NUMINAMATH_GPT_total_number_of_people_l678_67807

theorem total_number_of_people
  (cannoneers : ℕ) 
  (women : ℕ) 
  (men : ℕ) 
  (hc : cannoneers = 63)
  (hw : women = 2 * cannoneers)
  (hm : men = 2 * women) :
  cannoneers + women + men = 378 := 
sorry

end NUMINAMATH_GPT_total_number_of_people_l678_67807


namespace NUMINAMATH_GPT_layla_goldfish_count_l678_67833

def goldfish_count (total_food : ℕ) (swordtails_count : ℕ) (swordtails_food : ℕ) (guppies_count : ℕ) (guppies_food : ℕ) (goldfish_food : ℕ) : ℕ :=
  total_food - (swordtails_count * swordtails_food + guppies_count * guppies_food) / goldfish_food

theorem layla_goldfish_count : goldfish_count 12 3 2 8 1 1 = 2 := by
  sorry

end NUMINAMATH_GPT_layla_goldfish_count_l678_67833


namespace NUMINAMATH_GPT_greatest_four_digit_divisible_by_3_5_6_l678_67865

theorem greatest_four_digit_divisible_by_3_5_6 : 
  ∃ n, n ≤ 9999 ∧ n ≥ 1000 ∧ (∀ m, m ≤ 9999 ∧ m ≥ 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0 → m ≤ n) ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n = 9990 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_four_digit_divisible_by_3_5_6_l678_67865


namespace NUMINAMATH_GPT_simplify_fraction_144_12672_l678_67819

theorem simplify_fraction_144_12672 : (144 / 12672 : ℚ) = 1 / 88 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_144_12672_l678_67819


namespace NUMINAMATH_GPT_find_x_value_l678_67851

theorem find_x_value : ∃ x : ℝ, 35 - (23 - (15 - x)) = 12 * 2 / 1 / 2 → x = -21 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l678_67851


namespace NUMINAMATH_GPT_find_f_of_9_l678_67897

variable (f : ℝ → ℝ)

-- Conditions
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_of_3 : f 3 = 4

-- Theorem statement to prove
theorem find_f_of_9 : f 9 = 64 := by
  sorry

end NUMINAMATH_GPT_find_f_of_9_l678_67897


namespace NUMINAMATH_GPT_net_rate_of_pay_is_25_l678_67844

-- Define the conditions 
variables (hours : ℕ) (speed : ℕ) (efficiency : ℕ)
variables (pay_per_mile : ℝ) (cost_per_gallon : ℝ)
variables (total_distance : ℕ) (gas_used : ℕ)
variables (total_earnings : ℝ) (total_cost : ℝ) (net_earnings : ℝ) (net_rate_of_pay : ℝ)

-- Assume the given conditions are as stated in the problem
axiom hrs : hours = 3
axiom spd : speed = 50
axiom eff : efficiency = 25
axiom ppm : pay_per_mile = 0.60
axiom cpg : cost_per_gallon = 2.50

-- Assuming intermediate computations
axiom distance_calc : total_distance = speed * hours
axiom gas_calc : gas_used = total_distance / efficiency
axiom earnings_calc : total_earnings = pay_per_mile * total_distance
axiom cost_calc : total_cost = cost_per_gallon * gas_used
axiom net_earnings_calc : net_earnings = total_earnings - total_cost
axiom pay_rate_calc : net_rate_of_pay = net_earnings / hours

-- Proving the final result
theorem net_rate_of_pay_is_25 :
  net_rate_of_pay = 25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_net_rate_of_pay_is_25_l678_67844


namespace NUMINAMATH_GPT_area_of_right_triangle_integers_l678_67812

theorem area_of_right_triangle_integers (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ (A : ℤ), A = (a * b) / 2 := 
sorry

end NUMINAMATH_GPT_area_of_right_triangle_integers_l678_67812


namespace NUMINAMATH_GPT_distance_CD_l678_67814

-- Conditions
variable (width_small : ℝ) 
variable (length_small : ℝ := 2 * width_small) 
variable (perimeter_small : ℝ := 2 * (width_small + length_small))
variable (width_large : ℝ := 3 * width_small)
variable (length_large : ℝ := 2 * length_small)
variable (area_large : ℝ := width_large * length_large)

-- Condition assertions
axiom smaller_rectangle_perimeter : perimeter_small = 6
axiom larger_rectangle_area : area_large = 12

-- Calculating distance hypothesis
theorem distance_CD (CD_x CD_y : ℝ) (width_small length_small width_large length_large : ℝ) 
  (smaller_rectangle_perimeter : 2 * (width_small + length_small) = 6)
  (larger_rectangle_area : (3 * width_small) * (2 * length_small) = 12)
  (CD_x_def : CD_x = 2 * length_small)
  (CD_y_def : CD_y = 2 * width_large - width_small)
  : Real.sqrt ((CD_x) ^ 2 + (CD_y) ^ 2) = Real.sqrt 45 := 
sorry

end NUMINAMATH_GPT_distance_CD_l678_67814


namespace NUMINAMATH_GPT_intersection_of_solution_sets_solution_set_of_modified_inequality_l678_67803

open Set Real

theorem intersection_of_solution_sets :
  let A := {x | x ^ 2 - 2 * x - 3 < 0}
  let B := {x | x ^ 2 + x - 6 < 0}
  A ∩ B = {x | -1 < x ∧ x < 2} := by {
  sorry
}

theorem solution_set_of_modified_inequality :
  let A := {x | x ^ 2 + (-1) * x + (-2) < 0}
  A = {x | true} := by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_solution_sets_solution_set_of_modified_inequality_l678_67803


namespace NUMINAMATH_GPT_gcd_8885_4514_5246_l678_67894

theorem gcd_8885_4514_5246 : Nat.gcd (Nat.gcd 8885 4514) 5246 = 1 :=
sorry

end NUMINAMATH_GPT_gcd_8885_4514_5246_l678_67894


namespace NUMINAMATH_GPT_geometric_series_sum_l678_67842

theorem geometric_series_sum : 
  let a : ℕ := 2
  let r : ℕ := 3
  let n : ℕ := 6
  let S_n := (a * (r^n - 1)) / (r - 1)
  S_n = 728 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l678_67842


namespace NUMINAMATH_GPT_roots_of_quadratic_sum_cube_l678_67879

noncomputable def quadratic_roots (a b c : ℤ) (p q : ℤ) : Prop :=
  p^2 - b * p + c = 0 ∧ q^2 - b * q + c = 0

theorem roots_of_quadratic_sum_cube (p q : ℤ) :
  quadratic_roots 1 (-5) 6 p q →
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 503 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_sum_cube_l678_67879


namespace NUMINAMATH_GPT_ripe_mangoes_remaining_l678_67850

theorem ripe_mangoes_remaining
  (initial_mangoes : ℕ)
  (ripe_fraction : ℚ)
  (consume_fraction : ℚ)
  (initial_total : initial_mangoes = 400)
  (ripe_ratio : ripe_fraction = 3 / 5)
  (consume_ratio : consume_fraction = 60 / 100) :
  (initial_mangoes * ripe_fraction - initial_mangoes * ripe_fraction * consume_fraction) = 96 :=
by
  sorry

end NUMINAMATH_GPT_ripe_mangoes_remaining_l678_67850


namespace NUMINAMATH_GPT_petya_coloring_l678_67840

theorem petya_coloring (k : ℕ) : k = 1 :=
  sorry

end NUMINAMATH_GPT_petya_coloring_l678_67840


namespace NUMINAMATH_GPT_abigail_savings_l678_67808

-- Define the parameters for monthly savings and number of months in a year.
def monthlySavings : ℕ := 4000
def numberOfMonthsInYear : ℕ := 12

-- Define the total savings calculation.
def totalSavings (monthlySavings : ℕ) (numberOfMonths : ℕ) : ℕ :=
  monthlySavings * numberOfMonths

-- State the theorem that we need to prove.
theorem abigail_savings : totalSavings monthlySavings numberOfMonthsInYear = 48000 := by
  sorry

end NUMINAMATH_GPT_abigail_savings_l678_67808


namespace NUMINAMATH_GPT_molecular_weight_N2O3_correct_l678_67863

/-- Conditions -/
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

/-- Proof statement -/
theorem molecular_weight_N2O3_correct :
  (2 * atomic_weight_N + 3 * atomic_weight_O) = 76.02 ∧
  name_of_N2O3 = "dinitrogen trioxide" := sorry

/-- Definition of the compound name based on formula -/
def name_of_N2O3 : String := "dinitrogen trioxide"

end NUMINAMATH_GPT_molecular_weight_N2O3_correct_l678_67863


namespace NUMINAMATH_GPT_smallest_angle_in_convex_polygon_l678_67890

theorem smallest_angle_in_convex_polygon :
  ∀ (n : ℕ) (angles : ℕ → ℕ) (d : ℕ), n = 25 → (∀ i, 1 ≤ i ∧ i ≤ n → angles i = 166 - 1 * (13 - i)) 
  → 1 ≤ d ∧ d ≤ 1 → (angles 1 = 154) := 
by
  sorry

end NUMINAMATH_GPT_smallest_angle_in_convex_polygon_l678_67890


namespace NUMINAMATH_GPT_dogwood_tree_cut_count_l678_67817

theorem dogwood_tree_cut_count
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_left : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0)
  (h3 : trees_left = 2.0) :
  trees_part1 + trees_part2 - trees_left = 7.0 :=
by
  sorry

end NUMINAMATH_GPT_dogwood_tree_cut_count_l678_67817


namespace NUMINAMATH_GPT_max_n_consecutive_sum_2014_l678_67899

theorem max_n_consecutive_sum_2014 : 
  ∃ (k n : ℕ), (2 * k + n - 1) * n = 4028 ∧ n = 53 ∧ k > 0 := sorry

end NUMINAMATH_GPT_max_n_consecutive_sum_2014_l678_67899


namespace NUMINAMATH_GPT_volume_box_constraint_l678_67818

theorem volume_box_constraint : ∀ x : ℕ, ((2 * x + 6) * (x^3 - 8) * (x^2 + 4) < 1200) → x = 2 :=
by
  intros x h
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_volume_box_constraint_l678_67818


namespace NUMINAMATH_GPT_sin_gamma_delta_l678_67845

theorem sin_gamma_delta (γ δ : ℝ)
  (hγ : Complex.exp (Complex.I * γ) = Complex.ofReal 4 / 5 + Complex.I * (3 / 5))
  (hδ : Complex.exp (Complex.I * δ) = Complex.ofReal (-5 / 13) + Complex.I * (12 / 13)) :
  Real.sin (γ + δ) = 21 / 65 :=
by
  sorry

end NUMINAMATH_GPT_sin_gamma_delta_l678_67845


namespace NUMINAMATH_GPT_total_colors_needed_l678_67853

def num_planets : ℕ := 8
def num_people : ℕ := 3

theorem total_colors_needed : num_people * num_planets = 24 := by
  sorry

end NUMINAMATH_GPT_total_colors_needed_l678_67853


namespace NUMINAMATH_GPT_evaluate_polynomial_at_2_l678_67830

def polynomial (x : ℕ) : ℕ := 3 * x^4 + x^3 + 2 * x^2 + x + 4

def horner_method (x : ℕ) : ℕ :=
  let v_0 := x
  let v_1 := 3 * v_0 + 1
  let v_2 := v_1 * v_0 + 2
  v_2

theorem evaluate_polynomial_at_2 :
  horner_method 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_2_l678_67830


namespace NUMINAMATH_GPT_salon_visitors_l678_67857

noncomputable def total_customers (x : ℕ) : ℕ :=
  let revenue_customers_with_one_visit := 10 * x
  let revenue_customers_with_two_visits := 30 * 18
  let revenue_customers_with_three_visits := 10 * 26
  let total_revenue := revenue_customers_with_one_visit + revenue_customers_with_two_visits + revenue_customers_with_three_visits
  if total_revenue = 1240 then
    x + 30 + 10
  else
    0

theorem salon_visitors : 
  ∃ x, total_customers x = 84 :=
by
  use 44
  sorry

end NUMINAMATH_GPT_salon_visitors_l678_67857


namespace NUMINAMATH_GPT_F_of_3153_max_value_of_N_l678_67886

-- Define friendly number predicate
def is_friendly (M : ℕ) : Prop :=
  let a := M / 1000
  let b := (M / 100) % 10
  let c := (M / 10) % 10
  let d := M % 10
  a - b = c - d

-- Define F(M)
def F (M : ℕ) : ℕ :=
  let a := M / 1000
  let b := (M / 100) % 10
  let s := M / 10
  let t := M % 1000
  s - t - 10 * b

-- Prove F(3153) = 152
theorem F_of_3153 : F 3153 = 152 :=
by sorry

-- Define the given predicate for N
def is_k_special (N : ℕ) : Prop :=
  let x := N / 1000
  let y := (N / 100) % 10
  let m := (N / 30) % 10
  let n := N % 10
  (N % 5 = 1) ∧ (1000 * x + 100 * y + 30 * m + n + 1001 = N) ∧
  (0 ≤ y ∧ y < x ∧ x ≤ 8) ∧ (0 ≤ m ∧ m ≤ 3) ∧ (0 ≤ n ∧ n ≤ 8) ∧ 
  is_friendly N

-- Prove the maximum value satisfying the given constraints
theorem max_value_of_N : ∀ N, is_k_special N → N ≤ 9696 :=
by sorry

end NUMINAMATH_GPT_F_of_3153_max_value_of_N_l678_67886


namespace NUMINAMATH_GPT_find_m_of_quadratic_root_zero_l678_67832

theorem find_m_of_quadratic_root_zero (m : ℝ) (h : ∃ x, (m * x^2 + 5 * x + m^2 - 2 * m = 0) ∧ x = 0) : m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_of_quadratic_root_zero_l678_67832


namespace NUMINAMATH_GPT_number_of_mappings_A_to_B_number_of_mappings_B_to_A_l678_67892

theorem number_of_mappings_A_to_B (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (B.card ^ A.card) = 4^5 :=
by sorry

theorem number_of_mappings_B_to_A (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (A.card ^ B.card) = 5^4 :=
by sorry

end NUMINAMATH_GPT_number_of_mappings_A_to_B_number_of_mappings_B_to_A_l678_67892


namespace NUMINAMATH_GPT_circle_area_eq_25pi_l678_67852

theorem circle_area_eq_25pi :
  (∃ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y - 12 = 0) →
  (∃ (area : ℝ), area = 25 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_circle_area_eq_25pi_l678_67852


namespace NUMINAMATH_GPT_long_letter_time_ratio_l678_67854

-- Definitions based on conditions
def letters_per_month := (30 / 3 : Nat)
def regular_letter_pages := (20 / 10 : Nat)
def total_regular_pages := letters_per_month * regular_letter_pages
def long_letter_pages := 24 - total_regular_pages

-- Define the times and calculate the ratios
def time_spent_per_page_regular := (20 / regular_letter_pages : Nat)
def time_spent_per_page_long := (80 / long_letter_pages : Nat)
def time_ratio := time_spent_per_page_long / time_spent_per_page_regular

-- Theorem to prove the ratio
theorem long_letter_time_ratio : time_ratio = 2 := by
  sorry

end NUMINAMATH_GPT_long_letter_time_ratio_l678_67854


namespace NUMINAMATH_GPT_num_solutions_eq_40_l678_67896

theorem num_solutions_eq_40 : 
  ∀ (n : ℕ), 
  (∃ seq : ℕ → ℕ, seq 1 = 4 ∧ (∀ k : ℕ, 1 ≤ k → seq (k + 1) = seq k + 4) ∧ seq 10 = 40) :=
by
  sorry

end NUMINAMATH_GPT_num_solutions_eq_40_l678_67896


namespace NUMINAMATH_GPT_problem_1_problem_2_l678_67859

-- Definition of the operation ⊕
def my_oplus (a b : ℚ) : ℚ := (a + 3 * b) / 2

-- Prove that 4(2 ⊕ 5) = 34
theorem problem_1 : 4 * my_oplus 2 5 = 34 := 
by sorry

-- Definitions of A and B
def A (x y : ℚ) : ℚ := x^2 + 2 * x * y + y^2
def B (x y : ℚ) : ℚ := -2 * x * y + y^2

-- Prove that (A ⊕ B) + (B ⊕ A) = 2x^2 + 4y^2
theorem problem_2 (x y : ℚ) : 
  my_oplus (A x y) (B x y) + my_oplus (B x y) (A x y) = 2 * x^2 + 4 * y^2 := 
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l678_67859


namespace NUMINAMATH_GPT_even_n_equals_identical_numbers_l678_67805

theorem even_n_equals_identical_numbers (n : ℕ) (h1 : n ≥ 2) : 
  (∃ f : ℕ → ℕ, (∀ a b, f a = f b + f b) ∧ n % 2 = 0) :=
sorry


end NUMINAMATH_GPT_even_n_equals_identical_numbers_l678_67805


namespace NUMINAMATH_GPT_stocks_closed_higher_l678_67866

-- Definition of the conditions:
def stocks : Nat := 1980
def increased (H L : Nat) : Prop := H = (1.20 : ℝ) * L
def total_stocks (H L : Nat) : Prop := H + L = stocks

-- Claim to prove
theorem stocks_closed_higher (H L : Nat) (h1 : increased H L) (h2 : total_stocks H L) : H = 1080 :=
by
  sorry

end NUMINAMATH_GPT_stocks_closed_higher_l678_67866


namespace NUMINAMATH_GPT_five_digit_number_divisibility_l678_67809

theorem five_digit_number_divisibility (a : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) : 11 ∣ 100001 * a :=
by
  sorry

end NUMINAMATH_GPT_five_digit_number_divisibility_l678_67809


namespace NUMINAMATH_GPT_smallest_y_not_defined_l678_67822

theorem smallest_y_not_defined : 
  ∃ y : ℝ, (6 * y^2 - 37 * y + 6 = 0) ∧ (∀ z : ℝ, (6 * z^2 - 37 * z + 6 = 0) → y ≤ z) ∧ y = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_y_not_defined_l678_67822


namespace NUMINAMATH_GPT_point_on_line_l678_67820

theorem point_on_line (A B C x₀ y₀ : ℝ) :
  (A * x₀ + B * y₀ + C = 0) ↔ (A * (x₀ - x₀) + B * (y₀ - y₀) = 0) :=
by 
  sorry

end NUMINAMATH_GPT_point_on_line_l678_67820


namespace NUMINAMATH_GPT_monkey_distance_l678_67821

-- Define the initial speeds and percentage adjustments
def swing_speed : ℝ := 10
def run_speed : ℝ := 15
def wind_resistance_percentage : ℝ := 0.10
def branch_assistance_percentage : ℝ := 0.05

-- Conditions
def adjusted_swing_speed : ℝ := swing_speed * (1 - wind_resistance_percentage)
def adjusted_run_speed : ℝ := run_speed * (1 + branch_assistance_percentage)
def run_time : ℝ := 5
def swing_time : ℝ := 10

-- Define the distance formulas based on the conditions
def run_distance : ℝ := adjusted_run_speed * run_time
def swing_distance : ℝ := adjusted_swing_speed * swing_time

-- Total distance calculation
def total_distance : ℝ := run_distance + swing_distance

-- Statement for the proof
theorem monkey_distance : total_distance = 168.75 := by
  sorry

end NUMINAMATH_GPT_monkey_distance_l678_67821


namespace NUMINAMATH_GPT_scientific_notation_of_probe_unit_area_l678_67883

def probe_unit_area : ℝ := 0.0000064

theorem scientific_notation_of_probe_unit_area :
  ∃ (mantissa : ℝ) (exponent : ℤ), probe_unit_area = mantissa * 10^exponent ∧ mantissa = 6.4 ∧ exponent = -6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_probe_unit_area_l678_67883


namespace NUMINAMATH_GPT_percentage_increase_l678_67810

theorem percentage_increase (x : ℝ) (h1 : x = 99.9) : 
  ((x - 90) / 90) * 100 = 11 :=
by 
  -- Add the required proof steps here
  sorry

end NUMINAMATH_GPT_percentage_increase_l678_67810


namespace NUMINAMATH_GPT_trigonometric_identities_l678_67860

theorem trigonometric_identities (α : ℝ) (h0 : 0 < α ∧ α < π / 2) (h1 : Real.sin α = 4 / 5) :
    (Real.tan α = 4 / 3) ∧ 
    ((Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 2) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identities_l678_67860


namespace NUMINAMATH_GPT_evaluate_nav_expression_l678_67847
noncomputable def nav (k m : ℕ) := k * (k - m)

theorem evaluate_nav_expression : (nav 5 1) + (nav 4 1) = 32 :=
by
  -- Skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_evaluate_nav_expression_l678_67847


namespace NUMINAMATH_GPT_alice_oranges_l678_67875

theorem alice_oranges (E A : ℕ) 
  (h1 : A = 2 * E) 
  (h2 : E + A = 180) : 
  A = 120 :=
by
  sorry

end NUMINAMATH_GPT_alice_oranges_l678_67875


namespace NUMINAMATH_GPT_division_4073_by_38_l678_67870

theorem division_4073_by_38 :
  ∃ q r, 4073 = 38 * q + r ∧ 0 ≤ r ∧ r < 38 ∧ q = 107 ∧ r = 7 := by
  sorry

end NUMINAMATH_GPT_division_4073_by_38_l678_67870


namespace NUMINAMATH_GPT_compare_a_x_l678_67876

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem compare_a_x (x a b : ℝ) (h1 : a = log_base 5 (3^x + 4^x))
                    (h2 : b = log_base 4 (5^x - 3^x)) (h3 : a ≥ b) : x ≤ a :=
by
  sorry

end NUMINAMATH_GPT_compare_a_x_l678_67876


namespace NUMINAMATH_GPT_find_abs_x_l678_67834

-- Given conditions
def A (x : ℝ) : ℝ := 3 + x
def B (x : ℝ) : ℝ := 3 - x
def distance (a b : ℝ) : ℝ := abs (a - b)

-- Problem statement: Prove |x| = 4 given the conditions
theorem find_abs_x (x : ℝ) (h : distance (A x) (B x) = 8) : abs x = 4 := 
  sorry

end NUMINAMATH_GPT_find_abs_x_l678_67834


namespace NUMINAMATH_GPT_cab_income_third_day_l678_67877

noncomputable def cab_driver_income (day1 day2 day3 day4 day5 : ℕ) : ℕ := 
day1 + day2 + day3 + day4 + day5

theorem cab_income_third_day 
  (day1 day2 day4 day5 avg_income total_income day3 : ℕ)
  (h1 : day1 = 45)
  (h2 : day2 = 50)
  (h3 : day4 = 65)
  (h4 : day5 = 70)
  (h_avg : avg_income = 58)
  (h_total : total_income = 5 * avg_income)
  (h_day_sum : day1 + day2 + day4 + day5 = 230) :
  total_income - 230 = 60 :=
sorry

end NUMINAMATH_GPT_cab_income_third_day_l678_67877


namespace NUMINAMATH_GPT_count_FourDigitNumsWithThousandsDigitFive_is_1000_l678_67825

def count_FourDigitNumsWithThousandsDigitFive : Nat :=
  let minNum := 5000
  let maxNum := 5999
  maxNum - minNum + 1

theorem count_FourDigitNumsWithThousandsDigitFive_is_1000 :
  count_FourDigitNumsWithThousandsDigitFive = 1000 :=
by
  sorry

end NUMINAMATH_GPT_count_FourDigitNumsWithThousandsDigitFive_is_1000_l678_67825


namespace NUMINAMATH_GPT_tankard_one_quarter_full_l678_67804

theorem tankard_one_quarter_full
  (C : ℝ) 
  (h : (3 / 4) * C = 480) : 
  (1 / 4) * C = 160 := 
by
  sorry

end NUMINAMATH_GPT_tankard_one_quarter_full_l678_67804


namespace NUMINAMATH_GPT_probability_at_least_two_meters_l678_67862

def rope_length : ℝ := 6
def num_nodes : ℕ := 5
def equal_parts : ℕ := 6
def min_length : ℝ := 2

theorem probability_at_least_two_meters (h_rope_division : rope_length / equal_parts = 1) :
  let favorable_cuts := 3
  let total_cuts := num_nodes
  (favorable_cuts : ℝ) / total_cuts = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_two_meters_l678_67862


namespace NUMINAMATH_GPT_negate_proposition_l678_67816

-- Definitions of sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- The original proposition p
def p : Prop := ∀ x, is_odd x → is_even (2 * x)

-- The negation of the proposition p
def neg_p : Prop := ∃ x, is_odd x ∧ ¬ is_even (2 * x)

-- Proof problem statement: Prove that the negation of proposition p is as defined in neg_p
theorem negate_proposition :
  (∀ x, is_odd x → is_even (2 * x)) ↔ (∃ x, is_odd x ∧ ¬ is_even (2 * x)) :=
sorry

end NUMINAMATH_GPT_negate_proposition_l678_67816


namespace NUMINAMATH_GPT_solve_for_x_l678_67827

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 ↔ x = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l678_67827


namespace NUMINAMATH_GPT_num_times_teams_face_each_other_l678_67858

-- Conditions
variable (teams games total_games : ℕ)
variable (k : ℕ)
variable (h1 : teams = 17)
variable (h2 : games = teams * (teams - 1) * k / 2)
variable (h3 : total_games = 1360)

-- Proof problem
theorem num_times_teams_face_each_other : k = 5 := 
by 
  sorry

end NUMINAMATH_GPT_num_times_teams_face_each_other_l678_67858


namespace NUMINAMATH_GPT_four_is_square_root_of_sixteen_l678_67828

theorem four_is_square_root_of_sixteen : (4 : ℝ) * (4 : ℝ) = 16 :=
by
  sorry

end NUMINAMATH_GPT_four_is_square_root_of_sixteen_l678_67828


namespace NUMINAMATH_GPT_slower_train_pass_time_l678_67826

noncomputable def time_to_pass (length_train : ℕ) (speed_faster_kmh : ℕ) (speed_slower_kmh : ℕ) : ℕ :=
  let speed_faster_mps := speed_faster_kmh * 5 / 18
  let speed_slower_mps := speed_slower_kmh * 5 / 18
  let relative_speed := speed_faster_mps + speed_slower_mps
  let distance := length_train
  distance * 18 / (relative_speed * 5)

theorem slower_train_pass_time :
  time_to_pass 500 45 15 = 300 :=
by
  sorry

end NUMINAMATH_GPT_slower_train_pass_time_l678_67826


namespace NUMINAMATH_GPT_number_of_boys_took_exam_l678_67880

theorem number_of_boys_took_exam (T F : ℕ) (h_avg_all : 35 * T = 39 * 100 + 15 * F)
                                (h_total_boys : T = 100 + F) : T = 120 :=
sorry

end NUMINAMATH_GPT_number_of_boys_took_exam_l678_67880


namespace NUMINAMATH_GPT_amount_needed_for_free_delivery_l678_67835

theorem amount_needed_for_free_delivery :
  let chicken_cost := 1.5 * 6.00
  let lettuce_cost := 3.00
  let tomatoes_cost := 2.50
  let sweet_potatoes_cost := 4 * 0.75
  let broccoli_cost := 2 * 2.00
  let brussel_sprouts_cost := 2.50
  let total_cost := chicken_cost + lettuce_cost + tomatoes_cost + sweet_potatoes_cost + broccoli_cost + brussel_sprouts_cost
  let min_spend_for_free_delivery := 35.00
  min_spend_for_free_delivery - total_cost = 11.00 := sorry

end NUMINAMATH_GPT_amount_needed_for_free_delivery_l678_67835


namespace NUMINAMATH_GPT_complex_omega_sum_l678_67884

open Complex

theorem complex_omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 + ω^48 + ω^51 + ω^54 + ω^57 + ω^60 + ω^63 = 1 := 
by
  sorry

end NUMINAMATH_GPT_complex_omega_sum_l678_67884
