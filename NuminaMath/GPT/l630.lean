import Mathlib

namespace NUMINAMATH_GPT_rational_numbers_product_power_l630_63024

theorem rational_numbers_product_power (a b : ℚ) (h : |a - 2| + (2 * b + 1)^2 = 0) :
  (a * b)^2013 = -1 :=
sorry

end NUMINAMATH_GPT_rational_numbers_product_power_l630_63024


namespace NUMINAMATH_GPT_probability_of_bug9_is_zero_l630_63029

-- Definitions based on conditions provided
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def non_vowels : List Char := ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
def digits_or_vowels : List Char := ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'E', 'I', 'O', 'U']

-- Defining the number of choices for each position
def first_symbol_choices : Nat := 5
def second_symbol_choices : Nat := 21
def third_symbol_choices : Nat := 20
def fourth_symbol_choices : Nat := 15

-- Total number of possible license plates
def total_plates : Nat := first_symbol_choices * second_symbol_choices * third_symbol_choices * fourth_symbol_choices

-- Probability calculation for the specific license plate "BUG9"
def probability_bug9 : Nat := 0

theorem probability_of_bug9_is_zero : probability_bug9 = 0 := by sorry

end NUMINAMATH_GPT_probability_of_bug9_is_zero_l630_63029


namespace NUMINAMATH_GPT_ratio_of_average_speeds_l630_63001

-- Conditions
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4
def distance_ab : ℕ := 600
def distance_ac : ℕ := 360

-- Theorem to prove the ratio of their average speeds
theorem ratio_of_average_speeds : (distance_ab / time_eddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 20 ∧
                                  (distance_ac / time_freddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 9 :=
by
  -- Solution steps go here if performing an actual proof
  sorry

end NUMINAMATH_GPT_ratio_of_average_speeds_l630_63001


namespace NUMINAMATH_GPT_jane_sleep_hours_for_second_exam_l630_63052

theorem jane_sleep_hours_for_second_exam :
  ∀ (score1 score2 hours1 hours2 : ℝ),
  score1 * hours1 = 675 →
  (score1 + score2) / 2 = 85 →
  score2 * hours2 = 675 →
  hours2 = 135 / 19 :=
by
  intros score1 score2 hours1 hours2 h1 h2 h3
  sorry

end NUMINAMATH_GPT_jane_sleep_hours_for_second_exam_l630_63052


namespace NUMINAMATH_GPT_degrees_to_radians_l630_63066

theorem degrees_to_radians (deg: ℝ) (h : deg = 120) : deg * (π / 180) = 2 * π / 3 :=
by
  simp [h]
  sorry

end NUMINAMATH_GPT_degrees_to_radians_l630_63066


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l630_63016

noncomputable def p (m : ℝ) : Prop :=
  -6 ≤ m ∧ m ≤ 6

noncomputable def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 9 ≠ 0

theorem sufficient_but_not_necessary (m : ℝ) :
  (p m → q m) ∧ (q m → ¬ p m) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l630_63016


namespace NUMINAMATH_GPT_total_cost_is_160_l630_63027

-- Define the costs of each dress
def CostOfPaulineDress := 30
def CostOfJeansDress := CostOfPaulineDress - 10
def CostOfIdasDress := CostOfJeansDress + 30
def CostOfPattysDress := CostOfIdasDress + 10

-- The total cost
def TotalCost := CostOfPaulineDress + CostOfJeansDress + CostOfIdasDress + CostOfPattysDress

-- Prove the total cost is $160
theorem total_cost_is_160 : TotalCost = 160 := by
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_total_cost_is_160_l630_63027


namespace NUMINAMATH_GPT_nine_b_value_l630_63033

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : a = b - 3) : 
  9 * b = 216 / 11 :=
by
  sorry

end NUMINAMATH_GPT_nine_b_value_l630_63033


namespace NUMINAMATH_GPT_quadratic_eq_unique_k_l630_63041

theorem quadratic_eq_unique_k (k : ℝ) (x1 x2 : ℝ) 
  (h_quad : x1^2 - 3*x1 + k = 0 ∧ x2^2 - 3*x2 + k = 0)
  (h_cond : x1 * x2 + 2 * x1 + 2 * x2 = 1) : k = -5 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_eq_unique_k_l630_63041


namespace NUMINAMATH_GPT_number_of_penguins_l630_63081

-- Define the number of animals and zookeepers
def zebras : ℕ := 22
def tigers : ℕ := 8
def zookeepers : ℕ := 12
def headsLessThanFeetBy : ℕ := 132

-- Define the theorem to prove the number of penguins (P)
theorem number_of_penguins (P : ℕ) (H : P + zebras + tigers + zookeepers + headsLessThanFeetBy = 4 * P + 4 * zebras + 4 * tigers + 2 * zookeepers) : P = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_penguins_l630_63081


namespace NUMINAMATH_GPT_second_fisherman_more_fish_l630_63036

-- Defining the conditions
def total_days : ℕ := 213
def first_fisherman_rate : ℕ := 3
def second_fisherman_rate1 : ℕ := 1
def second_fisherman_rate2 : ℕ := 2
def second_fisherman_rate3 : ℕ := 4
def days_rate1 : ℕ := 30
def days_rate2 : ℕ := 60
def days_rate3 : ℕ := total_days - (days_rate1 + days_rate2)

-- Calculating the total number of fish caught by both fishermen
def total_fish_first_fisherman : ℕ := first_fisherman_rate * total_days
def total_fish_second_fisherman : ℕ := (second_fisherman_rate1 * days_rate1) + 
                                        (second_fisherman_rate2 * days_rate2) + 
                                        (second_fisherman_rate3 * days_rate3)

-- Theorem stating the difference in the number of fish caught
theorem second_fisherman_more_fish : (total_fish_second_fisherman - total_fish_first_fisherman) = 3 := 
by
  sorry

end NUMINAMATH_GPT_second_fisherman_more_fish_l630_63036


namespace NUMINAMATH_GPT_conclusion_1_conclusion_3_l630_63050

def tensor (a b : ℝ) : ℝ := a * (1 - b)

theorem conclusion_1 : tensor 2 (-2) = 6 :=
by sorry

theorem conclusion_3 (a b : ℝ) (h : a + b = 0) : tensor a a + tensor b b = 2 * a * b :=
by sorry

end NUMINAMATH_GPT_conclusion_1_conclusion_3_l630_63050


namespace NUMINAMATH_GPT_houses_with_white_mailboxes_l630_63038

theorem houses_with_white_mailboxes (total_mail : ℕ) (total_houses : ℕ) (red_mailboxes : ℕ) (mail_per_house : ℕ)
    (h1 : total_mail = 48) (h2 : total_houses = 8) (h3 : red_mailboxes = 3) (h4 : mail_per_house = 6) :
  total_houses - red_mailboxes = 5 :=
by
  sorry

end NUMINAMATH_GPT_houses_with_white_mailboxes_l630_63038


namespace NUMINAMATH_GPT_trig_inequalities_l630_63056

theorem trig_inequalities :
  let sin_168 := Real.sin (168 * Real.pi / 180)
  let cos_10 := Real.cos (10 * Real.pi / 180)
  let tan_58 := Real.tan (58 * Real.pi / 180)
  let tan_45 := Real.tan (45 * Real.pi / 180)
  sin_168 < cos_10 ∧ cos_10 < tan_58 :=
  sorry

end NUMINAMATH_GPT_trig_inequalities_l630_63056


namespace NUMINAMATH_GPT_maria_trip_distance_l630_63098

variable (D : ℕ) -- Defining the total distance D as a natural number

-- Defining the conditions given in the problem
def first_stop_distance := D / 2
def second_stop_distance := first_stop_distance - (1 / 3 * first_stop_distance)
def third_stop_distance := second_stop_distance - (2 / 5 * second_stop_distance)
def remaining_distance := 180

-- The statement to prove
theorem maria_trip_distance : third_stop_distance = remaining_distance → D = 900 :=
by
  sorry

end NUMINAMATH_GPT_maria_trip_distance_l630_63098


namespace NUMINAMATH_GPT_find_function_l630_63010

theorem find_function (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m^2 + n^2) = f m ^ 2 + f n ^ 2)
  (h2 : f 1 > 0) : ∀ n : ℕ, f n = n := 
sorry

end NUMINAMATH_GPT_find_function_l630_63010


namespace NUMINAMATH_GPT_blanket_thickness_after_foldings_l630_63011

theorem blanket_thickness_after_foldings (initial_thickness : ℕ) (folds : ℕ) (h1 : initial_thickness = 3) (h2 : folds = 4) :
  (initial_thickness * 2^folds) = 48 :=
by
  -- start with definitions as per the conditions
  rw [h1, h2]
  -- proof would follow
  sorry

end NUMINAMATH_GPT_blanket_thickness_after_foldings_l630_63011


namespace NUMINAMATH_GPT_path_count_from_E_to_G_passing_through_F_l630_63060

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem path_count_from_E_to_G_passing_through_F :
  let E := (0, 0)
  let F := (5, 2)
  let G := (6, 5)
  ∃ (paths_EF paths_FG total_paths : ℕ),
  paths_EF = binom (5 + 2) 5 ∧
  paths_FG = binom (1 + 3) 1 ∧
  total_paths = paths_EF * paths_FG ∧
  total_paths = 84 := 
by
  sorry

end NUMINAMATH_GPT_path_count_from_E_to_G_passing_through_F_l630_63060


namespace NUMINAMATH_GPT_travel_time_comparison_l630_63061

theorem travel_time_comparison
  (v : ℝ) -- speed during the first trip
  (t1 : ℝ) (t2 : ℝ)
  (h1 : t1 = 80 / v) -- time for the first trip
  (h2 : t2 = 100 / v) -- time for the second trip
  : t2 = 1.25 * t1 :=
by
  sorry

end NUMINAMATH_GPT_travel_time_comparison_l630_63061


namespace NUMINAMATH_GPT_university_diploma_percentage_l630_63088

-- Define variables
variables (P U J : ℝ)  -- P: Percentage of total population (i.e., 1 or 100%), U: Having a university diploma, J: having the job of their choice
variables (h1 : 10 / 100 * P = 10 / 100 * P * (1 - U) * J)        -- 10% of the people do not have a university diploma but have the job of their choice
variables (h2 : 30 / 100 * (P * (1 - J)) = 30 / 100 * P * U * (1 - J))  -- 30% of the people who do not have the job of their choice have a university diploma
variables (h3 : 40 / 100 * P = 40 / 100 * P * J)                   -- 40% of the people have the job of their choice

-- Statement to prove
theorem university_diploma_percentage : 
  48 / 100 * P = (30 / 100 * P * J) + (18 / 100 * P * (1 - J)) :=
by sorry

end NUMINAMATH_GPT_university_diploma_percentage_l630_63088


namespace NUMINAMATH_GPT_dan_initial_amount_l630_63032

theorem dan_initial_amount (left_amount : ℕ) (candy_cost : ℕ) : left_amount = 3 ∧ candy_cost = 2 → left_amount + candy_cost = 5 :=
by
  sorry

end NUMINAMATH_GPT_dan_initial_amount_l630_63032


namespace NUMINAMATH_GPT_max_intersection_value_l630_63087

noncomputable def max_intersection_size (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) : ℕ :=
  if ((A.card = 2019) ∧ (B.card = 2019) ∧ (A ∩ B ∩ C).card = 2018)
  then (A ∩ B ∩ C).card 
  else 0

theorem max_intersection_value (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) :
  max_intersection_size A B C h1 h2 = 2018 :=
sorry

end NUMINAMATH_GPT_max_intersection_value_l630_63087


namespace NUMINAMATH_GPT_floor_sqrt_23_squared_eq_16_l630_63064

theorem floor_sqrt_23_squared_eq_16 :
  (Int.floor (Real.sqrt 23))^2 = 16 :=
by
  have h1 : 4 < Real.sqrt 23 := sorry
  have h2 : Real.sqrt 23 < 5 := sorry
  have floor_sqrt_23 : Int.floor (Real.sqrt 23) = 4 := sorry
  rw [floor_sqrt_23]
  norm_num

end NUMINAMATH_GPT_floor_sqrt_23_squared_eq_16_l630_63064


namespace NUMINAMATH_GPT_sum_ratio_15_l630_63002

variable (a b : ℕ → ℕ)
variable (S T : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequences
def sum_a (n : ℕ) := S n
def sum_b (n : ℕ) := T n

-- The ratio condition
def ratio_condition := ∀ n, a n * (n + 1) = b n * (3 * n + 21)

theorem sum_ratio_15
  (ha : sum_a 15 = 15 * a 8)
  (hb : sum_b 15 = 15 * b 8)
  (h_ratio : ratio_condition a b) :
  sum_a 15 / sum_b 15 = 5 :=
sorry

end NUMINAMATH_GPT_sum_ratio_15_l630_63002


namespace NUMINAMATH_GPT_rotational_homothety_commute_iff_centers_coincide_l630_63007

-- Define rotational homothety and its properties
structure RotationalHomothety (P : Type*) :=
(center : P)
(apply : P → P)
(is_homothety : ∀ p, apply (apply p) = apply p)

variables {P : Type*} [TopologicalSpace P] (H1 H2 : RotationalHomothety P)

-- Prove the equivalence statement
theorem rotational_homothety_commute_iff_centers_coincide :
  (H1.center = H2.center) ↔ (H1.apply ∘ H2.apply = H2.apply ∘ H1.apply) :=
sorry

end NUMINAMATH_GPT_rotational_homothety_commute_iff_centers_coincide_l630_63007


namespace NUMINAMATH_GPT_complex_number_solution_l630_63070

def i : ℂ := Complex.I

theorem complex_number_solution (z : ℂ) (h : z * (1 - i) = 2 * i) : z = -1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_solution_l630_63070


namespace NUMINAMATH_GPT_min_value_is_five_l630_63054

noncomputable def min_value (x y : ℝ) : ℝ :=
  if x + 3 * y = 5 * x * y then 3 * x + 4 * y else 0

theorem min_value_is_five {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : min_value x y = 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_is_five_l630_63054


namespace NUMINAMATH_GPT_vasya_100_using_fewer_sevens_l630_63018

-- Definitions and conditions
def seven := 7

-- Theorem to prove
theorem vasya_100_using_fewer_sevens :
  (777 / seven - 77 / seven = 100) ∨
  (seven * seven + seven * seven + seven / seven + seven / seven = 100) :=
by
  sorry

end NUMINAMATH_GPT_vasya_100_using_fewer_sevens_l630_63018


namespace NUMINAMATH_GPT_range_of_a_l630_63035

theorem range_of_a (a : ℝ) : (1 ∉ {x : ℝ | (x - a) / (x + a) < 0}) → ( -1 ≤ a ∧ a ≤ 1 ) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l630_63035


namespace NUMINAMATH_GPT_probability_sqrt_two_digit_less_than_seven_l630_63059

noncomputable def prob_sqrt_less_than_seven : ℚ := 
  let favorable := 39
  let total := 90
  favorable / total

theorem probability_sqrt_two_digit_less_than_seven : 
  prob_sqrt_less_than_seven = 13 / 30 := by
  sorry

end NUMINAMATH_GPT_probability_sqrt_two_digit_less_than_seven_l630_63059


namespace NUMINAMATH_GPT_import_tax_excess_amount_l630_63006

theorem import_tax_excess_amount 
    (tax_rate : ℝ) 
    (tax_paid : ℝ) 
    (total_value : ℝ)
    (X : ℝ) 
    (h1 : tax_rate = 0.07)
    (h2 : tax_paid = 109.2)
    (h3 : total_value = 2560) 
    (eq1 : tax_rate * (total_value - X) = tax_paid) :
    X = 1000 := sorry

end NUMINAMATH_GPT_import_tax_excess_amount_l630_63006


namespace NUMINAMATH_GPT_add_base6_numbers_l630_63019

def base6_to_base10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

def base10_to_base6 (n : ℕ) : (ℕ × ℕ × ℕ) := 
  (n / 6^2, (n % 6^2) / 6^1, (n % 6^2) % 6^1)

theorem add_base6_numbers : 
  let n1 := 3 * 6^1 + 5 * 6^0
  let n2 := 2 * 6^1 + 5 * 6^0
  let sum := n1 + n2
  base10_to_base6 sum = (1, 0, 4) :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_add_base6_numbers_l630_63019


namespace NUMINAMATH_GPT_math_problem_l630_63095

theorem math_problem :
  (-1:ℤ) ^ 2023 - |(-3:ℤ)| + ((-1/3:ℚ) ^ (-2:ℤ)) + ((Real.pi - 3.14)^0) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l630_63095


namespace NUMINAMATH_GPT_certain_number_is_3500_l630_63099

theorem certain_number_is_3500 :
  ∃ x : ℝ, x - (1000 / 20.50) = 3451.2195121951218 ∧ x = 3500 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_3500_l630_63099


namespace NUMINAMATH_GPT_calc_c_15_l630_63039

noncomputable def c : ℕ → ℝ
| 0 => 1 -- This case won't be used, setup for pattern match
| 1 => 3
| 2 => 5
| (n+3) => c (n+2) * c (n+1)

theorem calc_c_15 : c 15 = 3 ^ 235 :=
sorry

end NUMINAMATH_GPT_calc_c_15_l630_63039


namespace NUMINAMATH_GPT_distance_centers_triangle_l630_63094

noncomputable def distance_between_centers (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  let circumradius := (a * b * c) / (4 * K)
  let hypotenuse := by
    by_cases hc : a * a + b * b = c * c
    exact c
    by_cases hb : a * a + c * c = b * b
    exact b
    by_cases ha : b * b + c * c = a * a
    exact a
    exact 0
  let oc := hypotenuse / 2
  Real.sqrt (oc * oc + r * r)

theorem distance_centers_triangle :
  distance_between_centers 7 24 25 = Real.sqrt 165.25 := sorry

end NUMINAMATH_GPT_distance_centers_triangle_l630_63094


namespace NUMINAMATH_GPT_find_sin_expression_l630_63067

noncomputable def trigonometric_identity (γ : ℝ) : Prop :=
  3 * (Real.tan γ)^2 + 3 * (1 / (Real.tan γ))^2 + 2 / (Real.sin γ)^2 + 2 / (Real.cos γ)^2 = 19

theorem find_sin_expression (γ : ℝ) (h : trigonometric_identity γ) : 
  (Real.sin γ)^4 - (Real.sin γ)^2 = -1 / 5 :=
sorry

end NUMINAMATH_GPT_find_sin_expression_l630_63067


namespace NUMINAMATH_GPT_raj_earns_more_l630_63093

theorem raj_earns_more :
  let cost_per_sqft := 2
  let raj_length := 30
  let raj_width := 50
  let lena_length := 40
  let lena_width := 35
  let raj_area := raj_length * raj_width
  let lena_area := lena_length * lena_width
  let raj_earnings := raj_area * cost_per_sqft
  let lena_earnings := lena_area * cost_per_sqft
  raj_earnings - lena_earnings = 200 :=
by
  sorry

end NUMINAMATH_GPT_raj_earns_more_l630_63093


namespace NUMINAMATH_GPT_jamie_workday_percent_l630_63026

theorem jamie_workday_percent
  (total_work_hours : ℕ)
  (first_meeting_minutes : ℕ)
  (second_meeting_multiplier : ℕ)
  (break_minutes : ℕ)
  (total_minutes_per_hour : ℕ)
  (total_work_minutes : ℕ)
  (first_meeting_duration : ℕ)
  (second_meeting_duration : ℕ)
  (total_meeting_time : ℕ)
  (percentage_spent : ℚ) :
  total_work_hours = 10 →
  first_meeting_minutes = 60 →
  second_meeting_multiplier = 2 →
  break_minutes = 30 →
  total_minutes_per_hour = 60 →
  total_work_minutes = total_work_hours * total_minutes_per_hour →
  first_meeting_duration = first_meeting_minutes →
  second_meeting_duration = second_meeting_multiplier * first_meeting_duration →
  total_meeting_time = first_meeting_duration + second_meeting_duration + break_minutes →
  percentage_spent = (total_meeting_time : ℚ) / (total_work_minutes : ℚ) * 100 →
  percentage_spent = 35 :=
sorry

end NUMINAMATH_GPT_jamie_workday_percent_l630_63026


namespace NUMINAMATH_GPT_frog_hops_ratio_l630_63028

theorem frog_hops_ratio :
  ∀ (F1 F2 F3 : ℕ),
    F1 = 4 * F2 →
    F1 + F2 + F3 = 99 →
    F2 = 18 →
    (F2 : ℚ) / (F3 : ℚ) = 2 :=
by
  intros F1 F2 F3 h1 h2 h3
  -- algebraic manipulations and proof to be filled here
  sorry

end NUMINAMATH_GPT_frog_hops_ratio_l630_63028


namespace NUMINAMATH_GPT_correct_subtraction_l630_63084

theorem correct_subtraction (x : ℕ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end NUMINAMATH_GPT_correct_subtraction_l630_63084


namespace NUMINAMATH_GPT_least_number_added_1789_l630_63005

def least_number_added_to_divisible (n d : ℕ) : ℕ := d - (n % d)

theorem least_number_added_1789 :
  least_number_added_to_divisible 1789 (Nat.lcm (Nat.lcm 5 6) (Nat.lcm 4 3)) = 11 :=
by
  -- Step definitions
  have lcm_5_6 := Nat.lcm 5 6
  have lcm_4_3 := Nat.lcm 4 3
  have lcm_total := Nat.lcm lcm_5_6 lcm_4_3
  -- Computation of the final result
  have remainder := 1789 % lcm_total
  have required_add := lcm_total - remainder
  -- Conclusion based on the computed values
  sorry

end NUMINAMATH_GPT_least_number_added_1789_l630_63005


namespace NUMINAMATH_GPT_absolute_value_of_neg_five_l630_63097

theorem absolute_value_of_neg_five : |(-5 : ℤ)| = 5 := 
by 
  sorry

end NUMINAMATH_GPT_absolute_value_of_neg_five_l630_63097


namespace NUMINAMATH_GPT_intersection_A_B_l630_63009

-- Conditions
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-2, 2}

-- Proof of the intersection of A and B
theorem intersection_A_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l630_63009


namespace NUMINAMATH_GPT_prism_sphere_surface_area_l630_63031

theorem prism_sphere_surface_area :
  ∀ (a b c : ℝ), (a * b = 6) → (b * c = 2) → (a * c = 3) → 
  4 * Real.pi * ((Real.sqrt ((a ^ 2) + (b ^ 2) + (c ^ 2))) / 2) ^ 2 = 14 * Real.pi :=
by
  intros a b c hab hbc hac
  sorry

end NUMINAMATH_GPT_prism_sphere_surface_area_l630_63031


namespace NUMINAMATH_GPT_find_pairs_satisfying_conditions_l630_63014

theorem find_pairs_satisfying_conditions (x y : ℝ) :
    abs (x + y) = 3 ∧ x * y = -10 →
    (x = 5 ∧ y = -2) ∨ (x = -2 ∧ y = 5) ∨ (x = 2 ∧ y = -5) ∨ (x = -5 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_satisfying_conditions_l630_63014


namespace NUMINAMATH_GPT_equivalent_single_discount_l630_63074

theorem equivalent_single_discount :
  ∀ (x : ℝ), ((1 - 0.15) * (1 - 0.10) * (1 - 0.05) * x) = (1 - 0.273) * x :=
by
  intros x
  --- This proof is left blank intentionally.
  sorry

end NUMINAMATH_GPT_equivalent_single_discount_l630_63074


namespace NUMINAMATH_GPT_value_of_f_at_5_l630_63075

theorem value_of_f_at_5 (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = - f x) 
  (h_period : ∀ x, f (x + 4) = f x)
  (h_func : ∀ x, -2 ≤ x ∧ x < 0 → f x = 3 * x + 1) : 
  f 5 = 2 :=
  sorry

end NUMINAMATH_GPT_value_of_f_at_5_l630_63075


namespace NUMINAMATH_GPT_coffee_bean_price_l630_63008

theorem coffee_bean_price 
  (x : ℝ)
  (price_second : ℝ) (weight_first weight_second : ℝ)
  (total_weight : ℝ) (price_mixture : ℝ) 
  (value_mixture : ℝ) 
  (h1 : price_second = 12)
  (h2 : weight_first = 25)
  (h3 : weight_second = 25)
  (h4 : total_weight = 100)
  (h5 : price_mixture = 11.25)
  (h6 : value_mixture = total_weight * price_mixture)
  (h7 : weight_first + weight_second = total_weight) :
  25 * x + 25 * 12 = 100 * 11.25 → x = 33 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_coffee_bean_price_l630_63008


namespace NUMINAMATH_GPT_smallest_possible_value_l630_63022

theorem smallest_possible_value (a b c d : ℤ) 
  (h1 : a + b + c + d < 25) 
  (h2 : a > 8) 
  (h3 : b < 5) 
  (h4 : c % 2 = 1) 
  (h5 : d % 2 = 0) : 
  ∃ a' b' c' d' : ℤ, a' > 8 ∧ b' < 5 ∧ c' % 2 = 1 ∧ d' % 2 = 0 ∧ a' + b' + c' + d' < 25 ∧ (a' - b' + c' - d' = -4) := 
by 
  use 9, 4, 1, 10
  sorry

end NUMINAMATH_GPT_smallest_possible_value_l630_63022


namespace NUMINAMATH_GPT_average_mark_of_all_three_boys_is_432_l630_63047

noncomputable def max_score : ℝ := 900
noncomputable def get_score (percent : ℝ) : ℝ := (percent / 100) * max_score

noncomputable def amar_score : ℝ := get_score 64
noncomputable def bhavan_score : ℝ := get_score 36
noncomputable def chetan_score : ℝ := get_score 44

noncomputable def total_score : ℝ := amar_score + bhavan_score + chetan_score
noncomputable def average_score : ℝ := total_score / 3

theorem average_mark_of_all_three_boys_is_432 : average_score = 432 := 
by
  sorry

end NUMINAMATH_GPT_average_mark_of_all_three_boys_is_432_l630_63047


namespace NUMINAMATH_GPT_tetrahedron_ineq_l630_63090

variable (P Q R S : ℝ)

-- Given conditions
axiom ortho_condition : S^2 = P^2 + Q^2 + R^2

theorem tetrahedron_ineq (P Q R S : ℝ) (ortho_condition : S^2 = P^2 + Q^2 + R^2) :
  (P + Q + R) / S ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tetrahedron_ineq_l630_63090


namespace NUMINAMATH_GPT_ratio_of_cars_to_trucks_l630_63040

-- Definitions based on conditions
def total_vehicles : ℕ := 60
def trucks : ℕ := 20
def cars : ℕ := total_vehicles - trucks

-- Theorem to prove
theorem ratio_of_cars_to_trucks : (cars / trucks : ℚ) = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_cars_to_trucks_l630_63040


namespace NUMINAMATH_GPT_exists_rank_with_profit_2016_l630_63003

theorem exists_rank_with_profit_2016 : ∃ n : ℕ, n * (n + 1) / 2 = 2016 :=
by 
  sorry

end NUMINAMATH_GPT_exists_rank_with_profit_2016_l630_63003


namespace NUMINAMATH_GPT_Nara_height_is_1_69_l630_63077

-- Definitions of the conditions
def SangheonHeight : ℝ := 1.56
def ChihoHeight : ℝ := SangheonHeight - 0.14
def NaraHeight : ℝ := ChihoHeight + 0.27

-- The statement to prove
theorem Nara_height_is_1_69 : NaraHeight = 1.69 :=
by {
  sorry
}

end NUMINAMATH_GPT_Nara_height_is_1_69_l630_63077


namespace NUMINAMATH_GPT_garden_breadth_l630_63083

theorem garden_breadth (P L B : ℕ) (h1 : P = 700) (h2 : L = 250) (h3 : P = 2 * (L + B)) : B = 100 :=
by
  sorry

end NUMINAMATH_GPT_garden_breadth_l630_63083


namespace NUMINAMATH_GPT_shifted_parabola_sum_constants_l630_63051

theorem shifted_parabola_sum_constants :
  let a := 2
  let b := -17
  let c := 43
  a + b + c = 28 := sorry

end NUMINAMATH_GPT_shifted_parabola_sum_constants_l630_63051


namespace NUMINAMATH_GPT_inverse_shifted_point_l630_63012

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def inverse_function (f g : ℝ → ℝ) : Prop := ∀ y, f (g y) = y ∧ ∀ x, g (f x) = x

theorem inverse_shifted_point
  (f : ℝ → ℝ)
  (hf_odd : odd_function f)
  (hf_point : f (-1) = 3)
  (g : ℝ → ℝ)
  (hg_inverse : inverse_function f g) :
  g (2 - 5) = 1 :=
by
  sorry

end NUMINAMATH_GPT_inverse_shifted_point_l630_63012


namespace NUMINAMATH_GPT_min_theta_l630_63046

theorem min_theta (theta : ℝ) (k : ℤ) (h : theta + 2 * k * Real.pi = -11 / 4 * Real.pi) : 
  theta = -3 / 4 * Real.pi :=
  sorry

end NUMINAMATH_GPT_min_theta_l630_63046


namespace NUMINAMATH_GPT_abs_inequality_solution_bounded_a_b_inequality_l630_63004

theorem abs_inequality_solution (x : ℝ) : (-4 < x ∧ x < 0) ↔ (|x + 1| + |x + 3| < 4) := sorry

theorem bounded_a_b_inequality (a b : ℝ) (h1 : -4 < a) (h2 : a < 0) (h3 : -4 < b) (h4 : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| := sorry

end NUMINAMATH_GPT_abs_inequality_solution_bounded_a_b_inequality_l630_63004


namespace NUMINAMATH_GPT_probability_of_rolling_perfect_square_l630_63030

theorem probability_of_rolling_perfect_square :
  (3 / 12 : ℚ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_rolling_perfect_square_l630_63030


namespace NUMINAMATH_GPT_reggies_brother_long_shots_l630_63086

-- Define the number of points per type of shot
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define the number of shots made by Reggie
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := 1

-- Define the total number of points made by Reggie
def reggie_points : ℕ :=
  reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points

-- Define the total points by which Reggie loses
def points_lost_by : ℕ := 2

-- Prove the number of long shots made by Reggie's brother
theorem reggies_brother_long_shots : 
  (reggie_points + points_lost_by) / long_shot_points = 4 := by
  sorry

end NUMINAMATH_GPT_reggies_brother_long_shots_l630_63086


namespace NUMINAMATH_GPT_pizza_toppings_problem_l630_63058

theorem pizza_toppings_problem
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (pepperoni_mushroom_slices : ℕ)
  (pepperoni_olive_slices : ℕ)
  (mushroom_olive_slices : ℕ)
  (pepperoni_mushroom_olive_slices : ℕ) :
  total_slices = 20 →
  pepperoni_slices = 12 →
  mushroom_slices = 14 →
  olive_slices = 12 →
  pepperoni_mushroom_slices = 8 →
  pepperoni_olive_slices = 8 →
  mushroom_olive_slices = 8 →
  total_slices = pepperoni_slices + mushroom_slices + olive_slices
    - pepperoni_mushroom_slices - pepperoni_olive_slices - mushroom_olive_slices
    + pepperoni_mushroom_olive_slices →
  pepperoni_mushroom_olive_slices = 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pizza_toppings_problem_l630_63058


namespace NUMINAMATH_GPT_storks_minus_birds_l630_63045

/-- Define the initial values --/
def s : ℕ := 6         -- Number of storks
def b1 : ℕ := 2        -- Initial number of birds
def b2 : ℕ := 3        -- Number of additional birds

/-- Calculate the total number of birds --/
def b : ℕ := b1 + b2   -- Total number of birds

/-- Prove the number of storks minus the number of birds --/
theorem storks_minus_birds : s - b = 1 :=
by sorry

end NUMINAMATH_GPT_storks_minus_birds_l630_63045


namespace NUMINAMATH_GPT_complex_division_l630_63091

-- Conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Question: Prove the complex division
theorem complex_division (h : i = Complex.I) : (8 - i) / (2 + i) = 3 - 2 * i :=
by sorry

end NUMINAMATH_GPT_complex_division_l630_63091


namespace NUMINAMATH_GPT_marbles_cost_correct_l630_63068

def total_cost : ℝ := 20.52
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52

-- The problem is to prove that the amount spent on marbles is $9.05
def amount_spent_on_marbles : ℝ :=
  total_cost - (cost_football + cost_baseball)

theorem marbles_cost_correct :
  amount_spent_on_marbles = 9.05 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_marbles_cost_correct_l630_63068


namespace NUMINAMATH_GPT_inverseP_l630_63076

-- Mathematical definitions
def isOdd (a : ℕ) : Prop := a % 2 = 1
def isPrime (a : ℕ) : Prop := Nat.Prime a

-- Given proposition P (hypothesis)
def P (a : ℕ) : Prop := isOdd a → isPrime a

-- Inverse proposition: if a is prime, then a is odd
theorem inverseP (a : ℕ) (h : isPrime a) : isOdd a :=
sorry

end NUMINAMATH_GPT_inverseP_l630_63076


namespace NUMINAMATH_GPT_find_a_l630_63043

noncomputable def A : Set ℝ := {1, 2, 3, 4}
noncomputable def B (a : ℝ) : Set ℝ := { x | x ≤ a }

theorem find_a (a : ℝ) (h_union : A ∪ B a = Set.Iic 5) : a = 5 := by
  sorry

end NUMINAMATH_GPT_find_a_l630_63043


namespace NUMINAMATH_GPT_ratio_books_donated_l630_63079

theorem ratio_books_donated (initial_books: ℕ) (books_given_nephew: ℕ) (books_after_nephew: ℕ) 
  (books_final: ℕ) (books_purchased: ℕ) (books_donated_library: ℕ) (ratio: ℕ):
    initial_books = 40 → 
    books_given_nephew = initial_books / 4 → 
    books_after_nephew = initial_books - books_given_nephew →
    books_final = 23 →
    books_purchased = 3 →
    books_donated_library = books_after_nephew - (books_final - books_purchased) →
    ratio = books_donated_library / books_after_nephew →
    ratio = 1 / 3 := sorry

end NUMINAMATH_GPT_ratio_books_donated_l630_63079


namespace NUMINAMATH_GPT_Madison_minimum_score_l630_63065

theorem Madison_minimum_score (q1 q2 q3 q4 q5 : ℕ) (h1 : q1 = 84) (h2 : q2 = 81) (h3 : q3 = 87) (h4 : q4 = 83) (h5 : 85 * 5 ≤ q1 + q2 + q3 + q4 + q5) : 
  90 ≤ q5 := 
by
  sorry

end NUMINAMATH_GPT_Madison_minimum_score_l630_63065


namespace NUMINAMATH_GPT_operation_proof_l630_63072

def operation (x y : ℤ) : ℤ := x * y - 3 * x - 4 * y

theorem operation_proof : (operation 7 2) - (operation 2 7) = 5 :=
by
  sorry

end NUMINAMATH_GPT_operation_proof_l630_63072


namespace NUMINAMATH_GPT_candy_bar_cost_is_7_l630_63057

-- Define the conditions
def chocolate_cost : Nat := 3
def candy_additional_cost : Nat := 4

-- Define the expression for the cost of the candy bar
def candy_cost : Nat := chocolate_cost + candy_additional_cost

-- State the theorem to prove the cost of the candy bar
theorem candy_bar_cost_is_7 : candy_cost = 7 :=
by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_is_7_l630_63057


namespace NUMINAMATH_GPT_determine_k_l630_63049

theorem determine_k (k r s : ℝ) (h1 : r + s = -k) (h2 : (r + 3) + (s + 3) = k) : k = 3 :=
by
  sorry

end NUMINAMATH_GPT_determine_k_l630_63049


namespace NUMINAMATH_GPT_rebecca_charge_for_dye_job_l630_63089

def charges_for_services (haircuts per perms per dye_jobs hair_dye_per_dye_job tips : ℕ) : ℕ := 
  4 * 30 + 1 * 40 + 2 * (dye_jobs - hair_dye_per_dye_job) + tips

theorem rebecca_charge_for_dye_job 
  (haircuts: ℕ) (perms: ℕ) (hair_dye_per_dye_job: ℕ) (tips: ℕ) (end_of_day_amount: ℕ) : 
  haircuts = 4 → perms = 1 → hair_dye_per_dye_job = 10 → tips = 50 → 
  end_of_day_amount = 310 → 
  ∃ D: ℕ, D = 60 := 
by
  sorry

end NUMINAMATH_GPT_rebecca_charge_for_dye_job_l630_63089


namespace NUMINAMATH_GPT_part1_part2_l630_63023

open Real

noncomputable def f (x a : ℝ) : ℝ := 45 * abs (x - a) + 45 * abs (x - 5)

theorem part1 (a : ℝ) :
    (∀ (x : ℝ), f x a ≥ 3) ↔ (a ≤ 2 ∨ a ≥ 8) :=
sorry

theorem part2 (a : ℝ) (ha : a = 2) :
    ∀ (x : ℝ), (f x 2 ≥ x^2 - 8*x + 15) ↔ (2 ≤ x ∧ x ≤ 5 + Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_part1_part2_l630_63023


namespace NUMINAMATH_GPT_abs_diff_gt_half_prob_l630_63071

noncomputable def probability_abs_diff_gt_half : ℝ :=
  ((1 / 4) * (1 / 8) + 
   (1 / 8) * (1 / 2) + 
   (1 / 8) * 1) * 2

theorem abs_diff_gt_half_prob : probability_abs_diff_gt_half = 5 / 16 := by 
  sorry

end NUMINAMATH_GPT_abs_diff_gt_half_prob_l630_63071


namespace NUMINAMATH_GPT_find_speed_of_stream_l630_63062

variable (b s : ℝ)

-- Equation derived from downstream condition
def downstream_equation := b + s = 24

-- Equation derived from upstream condition
def upstream_equation := b - s = 10

theorem find_speed_of_stream
  (b s : ℝ)
  (h1 : downstream_equation b s)
  (h2 : upstream_equation b s) :
  s = 7 := by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_speed_of_stream_l630_63062


namespace NUMINAMATH_GPT_johns_age_is_15_l630_63073

-- Definitions from conditions
variables (J F : ℕ) -- J is John's age, F is his father's age
axiom sum_of_ages : J + F = 77
axiom father_age : F = 2 * J + 32

-- Target statement to prove
theorem johns_age_is_15 : J = 15 :=
by
  sorry

end NUMINAMATH_GPT_johns_age_is_15_l630_63073


namespace NUMINAMATH_GPT_coins_in_box_l630_63069

theorem coins_in_box (n : ℕ) 
    (h1 : n % 8 = 7) 
    (h2 : n % 7 = 5) : 
    n = 47 ∧ (47 % 9 = 2) :=
sorry

end NUMINAMATH_GPT_coins_in_box_l630_63069


namespace NUMINAMATH_GPT_board_total_length_l630_63085

-- Definitions based on conditions
def S : ℝ := 2
def L : ℝ := 2 * S

-- Define the total length of the board
def T : ℝ := S + L

-- The theorem asserting the total length of the board is 6 ft
theorem board_total_length : T = 6 := 
by
  sorry

end NUMINAMATH_GPT_board_total_length_l630_63085


namespace NUMINAMATH_GPT_trajectory_is_one_branch_of_hyperbola_l630_63015

open Real

-- Condition 1: Given points F1 and F2
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Condition 2: Moving point P such that |PF1| - |PF2| = 4
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  abs (dist P F1) - abs (dist P F2) = 4

-- Prove the trajectory of point P is one branch of a hyperbola
theorem trajectory_is_one_branch_of_hyperbola (P : ℝ × ℝ) (h : satisfies_condition P) : 
  (∃ a b : ℝ, ∀ x y: ℝ, satisfies_condition (x, y) → (((x^2 / a^2) - (y^2 / b^2) = 1) ∨ ((x^2 / a^2) - (y^2 / b^2) = -1))) :=
sorry

end NUMINAMATH_GPT_trajectory_is_one_branch_of_hyperbola_l630_63015


namespace NUMINAMATH_GPT_total_students_l630_63034

-- Definitions
def is_half_reading (S : ℕ) (half_reading : ℕ) := half_reading = S / 2
def is_third_playing (S : ℕ) (third_playing : ℕ) := third_playing = S / 3
def is_total_students (S half_reading third_playing homework : ℕ) := half_reading + third_playing + homework = S

-- Homework is given to be 4
def homework : ℕ := 4

-- Total number of students
theorem total_students (S : ℕ) (half_reading third_playing : ℕ)
    (h₁ : is_half_reading S half_reading) 
    (h₂ : is_third_playing S third_playing) 
    (h₃ : is_total_students S half_reading third_playing homework) :
    S = 24 := 
sorry

end NUMINAMATH_GPT_total_students_l630_63034


namespace NUMINAMATH_GPT_number_of_ways_split_2000_cents_l630_63042

theorem number_of_ways_split_2000_cents : 
  ∃ n : ℕ, n = 357 ∧ (∃ (nick d q : ℕ), 
    nick > 0 ∧ d > 0 ∧ q > 0 ∧ 5 * nick + 10 * d + 25 * q = 2000) :=
sorry

end NUMINAMATH_GPT_number_of_ways_split_2000_cents_l630_63042


namespace NUMINAMATH_GPT_stratified_sampling_l630_63013

/-- Given a batch of 98 water heaters with 56 from Factory A and 42 from Factory B,
    and a stratified sample of 14 units is to be drawn, prove that the number 
    of water heaters sampled from Factory A is 8 and from Factory B is 6. --/

theorem stratified_sampling (batch_size A B sample_size : ℕ) 
  (h_batch : batch_size = 98) 
  (h_fact_a : A = 56) 
  (h_fact_b : B = 42) 
  (h_sample : sample_size = 14) : 
  (A * sample_size / batch_size = 8) ∧ (B * sample_size / batch_size = 6) := 
  by
    sorry

end NUMINAMATH_GPT_stratified_sampling_l630_63013


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l630_63000

variables {x : ℝ} {f : ℝ → ℝ}

def is_quadratic_and_opens_downwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_symmetric_at_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f (2 + x)

theorem quadratic_inequality_solution
  (h_quadratic : is_quadratic_and_opens_downwards f)
  (h_symmetric : is_symmetric_at_two f) :
  (1 - (Real.sqrt 14) / 4) < x ∧ x < (1 + (Real.sqrt 14) / 4) ↔
  f (Real.log ((1 / (1 / 4)) * (x^2 + x + 1 / 2))) <
  f (Real.log ((1 / (1 / 2)) * (2 * x^2 - x + 5 / 8))) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l630_63000


namespace NUMINAMATH_GPT_triangle_side_length_l630_63025

theorem triangle_side_length (a b c x : ℕ) (A C : ℝ) (h1 : b = x) (h2 : a = x - 2) (h3 : c = x + 2)
  (h4 : C = 2 * A) (h5 : x + 2 = 10) : a = 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l630_63025


namespace NUMINAMATH_GPT_Gage_skating_minutes_l630_63096

theorem Gage_skating_minutes (d1 d2 d3 : ℕ) (m1 m2 : ℕ) (avg : ℕ) (h1 : d1 = 6) (h2 : d2 = 4) (h3 : d3 = 1) (h4 : m1 = 80) (h5 : m2 = 105) (h6 : avg = 95) : 
  (d1 * m1 + d2 * m2 + d3 * x) / (d1 + d2 + d3) = avg ↔ x = 145 := 
by 
  sorry

end NUMINAMATH_GPT_Gage_skating_minutes_l630_63096


namespace NUMINAMATH_GPT_monthly_income_of_P_l630_63021

variable (P Q R : ℝ)

theorem monthly_income_of_P (h1 : (P + Q) / 2 = 5050) 
                           (h2 : (Q + R) / 2 = 6250) 
                           (h3 : (P + R) / 2 = 5200) : 
    P = 4000 := 
sorry

end NUMINAMATH_GPT_monthly_income_of_P_l630_63021


namespace NUMINAMATH_GPT_john_annual_patients_l630_63048

-- Definitions for the various conditions
def first_hospital_patients_per_day := 20
def second_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 20 / 100)
def third_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 15 / 100)
def total_patients_per_day := first_hospital_patients_per_day + second_hospital_patients_per_day + third_hospital_patients_per_day
def workdays_per_week := 5
def total_patients_per_week := total_patients_per_day * workdays_per_week
def working_weeks_per_year := 50 - 2 -- considering 2 weeks of vacation
def total_patients_per_year := total_patients_per_week * working_weeks_per_year

-- The statement to prove
theorem john_annual_patients : total_patients_per_year = 16080 := by
  sorry

end NUMINAMATH_GPT_john_annual_patients_l630_63048


namespace NUMINAMATH_GPT_election_win_by_votes_l630_63092

/-- Two candidates in an election, the winner received 56% of votes and won the election
by receiving 1344 votes. We aim to prove that the winner won by 288 votes. -/
theorem election_win_by_votes
  (V : ℝ)  -- total number of votes
  (w : ℝ)  -- percentage of votes received by the winner
  (w_votes : ℝ)  -- votes received by the winner
  (l_votes : ℝ)  -- votes received by the loser
  (w_percentage : w = 0.56)
  (w_votes_given : w_votes = 1344)
  (total_votes : V = 1344 / 0.56)
  (l_votes_calc : l_votes = (V * 0.44)) :
  1344 - l_votes = 288 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_election_win_by_votes_l630_63092


namespace NUMINAMATH_GPT_butterfly_eq_roots_l630_63017

theorem butterfly_eq_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0)
    (h3 : (a + c)^2 - 4 * a * c = 0) : a = c :=
by
  sorry

end NUMINAMATH_GPT_butterfly_eq_roots_l630_63017


namespace NUMINAMATH_GPT_maximize_Sn_l630_63044

def a_n (n : ℕ) : ℤ := 26 - 2 * n

def S_n (n : ℕ) : ℤ := n * (26 - 2 * (n + 1)) / 2 + 26 * n

theorem maximize_Sn : (n = 12 ∨ n = 13) ↔ (∀ m : ℕ, S_n m ≤ S_n 12 ∨ S_n m ≤ S_n 13) :=
by sorry

end NUMINAMATH_GPT_maximize_Sn_l630_63044


namespace NUMINAMATH_GPT_evaluate_f_l630_63063

def f (x : ℚ) : ℚ := (2 * x - 3) / (3 * x ^ 2 - 1)

theorem evaluate_f :
  f (-2) = -7 / 11 ∧ f (0) = 3 ∧ f (1) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_f_l630_63063


namespace NUMINAMATH_GPT_cyclists_no_point_b_l630_63080

theorem cyclists_no_point_b (v1 v2 t d : ℝ) (h1 : v1 = 35) (h2 : v2 = 25) (h3 : t = 2) (h4 : d = 30) :
  ∀ (ta tb : ℝ), ta + tb = t ∧ ta * v1 + tb * v2 < d → false :=
by
  sorry

end NUMINAMATH_GPT_cyclists_no_point_b_l630_63080


namespace NUMINAMATH_GPT_mike_books_l630_63037

theorem mike_books : 51 - 45 = 6 := 
by 
  rfl

end NUMINAMATH_GPT_mike_books_l630_63037


namespace NUMINAMATH_GPT_max_groups_l630_63078

theorem max_groups (cards : ℕ) (sum_group : ℕ) (c5 c2 c1 : ℕ) (cond1 : cards = 600) (cond2 : c5 = 200)
  (cond3 : c2 = 200) (cond4 : c1 = 200) (cond5 : sum_group = 9) :
  ∃ max_g : ℕ, max_g = 100 :=
by
  sorry

end NUMINAMATH_GPT_max_groups_l630_63078


namespace NUMINAMATH_GPT_problem_l630_63082

def gcf (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c
def lcm (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem problem (A B : ℕ) (hA : A = gcf 9 15 27) (hB : B = lcm 9 15 27) : A + B = 138 :=
by
  sorry

end NUMINAMATH_GPT_problem_l630_63082


namespace NUMINAMATH_GPT_probability_of_red_then_blue_is_correct_l630_63055

noncomputable def probability_red_then_blue : ℚ :=
  let total_marbles := 5 + 4 + 12 + 2
  let prob_red := 5 / total_marbles
  let remaining_marbles := total_marbles - 1
  let prob_blue_given_red := 2 / remaining_marbles
  prob_red * prob_blue_given_red

theorem probability_of_red_then_blue_is_correct :
  probability_red_then_blue = 5 / 253 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_red_then_blue_is_correct_l630_63055


namespace NUMINAMATH_GPT_anna_clara_age_l630_63053

theorem anna_clara_age :
  ∃ x : ℕ, (54 - x) * 3 = 80 - x ∧ x = 41 :=
by
  sorry

end NUMINAMATH_GPT_anna_clara_age_l630_63053


namespace NUMINAMATH_GPT_sue_initially_borrowed_six_movies_l630_63020

variable (M : ℕ)
variable (initial_books : ℕ := 15)
variable (returned_books : ℕ := 8)
variable (returned_movies_fraction : ℚ := 1/3)
variable (additional_books : ℕ := 9)
variable (total_items : ℕ := 20)

theorem sue_initially_borrowed_six_movies (hM : total_items = initial_books - returned_books + additional_books + (M - returned_movies_fraction * M)) : 
  M = 6 := by
  sorry

end NUMINAMATH_GPT_sue_initially_borrowed_six_movies_l630_63020
