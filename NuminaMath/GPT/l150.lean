import Mathlib

namespace NUMINAMATH_GPT_differentiable_function_zero_l150_15081

theorem differentiable_function_zero
    (f : ℝ → ℝ)
    (h_diff : Differentiable ℝ f)
    (h_zero : f 0 = 0)
    (h_ineq : ∀ x : ℝ, 0 < |f x| ∧ |f x| < 1/2 → |deriv f x| ≤ |f x * Real.log (|f x|)|) :
    ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_differentiable_function_zero_l150_15081


namespace NUMINAMATH_GPT_tiffany_lives_l150_15036

theorem tiffany_lives (initial_lives lives_lost lives_after_next_level lives_gained : ℕ)
  (h1 : initial_lives = 43)
  (h2 : lives_lost = 14)
  (h3 : lives_after_next_level = 56)
  (h4 : lives_gained = lives_after_next_level - (initial_lives - lives_lost)) :
  lives_gained = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_tiffany_lives_l150_15036


namespace NUMINAMATH_GPT_woman_lawyer_probability_l150_15069

noncomputable def probability_of_woman_lawyer : ℚ :=
  let total_members : ℚ := 100
  let women_percentage : ℚ := 0.80
  let lawyer_percentage_women : ℚ := 0.40
  let women_members := women_percentage * total_members
  let women_lawyers := lawyer_percentage_women * women_members
  let probability := women_lawyers / total_members
  probability

theorem woman_lawyer_probability :
  probability_of_woman_lawyer = 0.32 := by
  sorry

end NUMINAMATH_GPT_woman_lawyer_probability_l150_15069


namespace NUMINAMATH_GPT_proof_multiple_l150_15097

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

theorem proof_multiple (a b : ℕ) 
  (h₁ : is_multiple a 5) 
  (h₂ : is_multiple b 10) : 
  is_multiple b 5 ∧ 
  is_multiple (a + b) 5 ∧ 
  is_multiple (a + b) 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_multiple_l150_15097


namespace NUMINAMATH_GPT_girls_percentage_l150_15065

theorem girls_percentage (total_students girls boys : ℕ) 
    (total_eq : total_students = 42)
    (ratio : 3 * girls = 4 * boys)
    (total_students_eq : total_students = girls + boys) : 
    (girls * 100 / total_students : ℚ) = 57.14 := 
by 
  sorry

end NUMINAMATH_GPT_girls_percentage_l150_15065


namespace NUMINAMATH_GPT_sum_of_A_and_B_l150_15020

theorem sum_of_A_and_B:
  ∃ A B : ℕ, (A = 2 + 4) ∧ (B - 3 = 1) ∧ (A < 10) ∧ (B < 10) ∧ (A + B = 10) :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_A_and_B_l150_15020


namespace NUMINAMATH_GPT_smallest_non_consecutive_product_not_factor_of_48_l150_15007

def is_factor (a b : ℕ) : Prop := b % a = 0

def non_consecutive_pairs (x y : ℕ) : Prop := (x ≠ y) ∧ (x + 1 ≠ y) ∧ (y + 1 ≠ x)

theorem smallest_non_consecutive_product_not_factor_of_48 :
  ∃ x y, x ∣ 48 ∧ y ∣ 48 ∧ non_consecutive_pairs x y ∧ ¬ (x * y ∣ 48) ∧ (∀ x' y', x' ∣ 48 ∧ y' ∣ 48 ∧ non_consecutive_pairs x' y' ∧ ¬ (x' * y' ∣ 48) → x' * y' ≥ 18) :=
by
  sorry

end NUMINAMATH_GPT_smallest_non_consecutive_product_not_factor_of_48_l150_15007


namespace NUMINAMATH_GPT_elena_hike_total_miles_l150_15062

theorem elena_hike_total_miles (x1 x2 x3 x4 x5 : ℕ)
  (h1 : x1 + x2 = 36)
  (h2 : x2 + x3 = 40)
  (h3 : x3 + x4 + x5 = 45)
  (h4 : x1 + x4 = 38) : 
  x1 + x2 + x3 + x4 + x5 = 81 := 
sorry

end NUMINAMATH_GPT_elena_hike_total_miles_l150_15062


namespace NUMINAMATH_GPT_athlete_speed_200m_in_24s_is_30kmh_l150_15096

noncomputable def speed_in_kmh (distance_meters : ℝ) (time_seconds : ℝ) : ℝ :=
  (distance_meters / 1000) / (time_seconds / 3600)

theorem athlete_speed_200m_in_24s_is_30kmh :
  speed_in_kmh 200 24 = 30 := by
  sorry

end NUMINAMATH_GPT_athlete_speed_200m_in_24s_is_30kmh_l150_15096


namespace NUMINAMATH_GPT_committee_meeting_l150_15014

theorem committee_meeting : 
  ∃ (A B : ℕ), 2 * A + B = 7 ∧ A + 2 * B = 11 ∧ A + B = 6 :=
by 
  sorry

end NUMINAMATH_GPT_committee_meeting_l150_15014


namespace NUMINAMATH_GPT_proof_subset_l150_15044

def M : Set ℝ := {x | x ≥ 0}
def N : Set ℝ := {0, 1, 2}

theorem proof_subset : N ⊆ M := sorry

end NUMINAMATH_GPT_proof_subset_l150_15044


namespace NUMINAMATH_GPT_minimum_value_of_expression_l150_15053

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  1 / x + 4 / y + 9 / z

theorem minimum_value_of_expression (x y z : ℝ) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 1) :
  min_value_expression x y z ≥ 36 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l150_15053


namespace NUMINAMATH_GPT_combination_15_3_l150_15032

theorem combination_15_3 :
  (Nat.choose 15 3 = 455) :=
by
  sorry

end NUMINAMATH_GPT_combination_15_3_l150_15032


namespace NUMINAMATH_GPT_train_passing_time_l150_15042

theorem train_passing_time 
  (length_train : ℕ) 
  (speed_train_kmph : ℕ) 
  (time_to_pass : ℕ)
  (h1 : length_train = 60)
  (h2 : speed_train_kmph = 54)
  (h3 : time_to_pass = 4) :
  time_to_pass = length_train * 18 / (speed_train_kmph * 5) := by
  sorry

end NUMINAMATH_GPT_train_passing_time_l150_15042


namespace NUMINAMATH_GPT_simplify_and_evaluate_l150_15064

def expr (x : ℤ) : ℤ := (x + 2) * (x - 2) - (x - 1) ^ 2

theorem simplify_and_evaluate : expr (-1) = -7 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l150_15064


namespace NUMINAMATH_GPT_trigonometric_identity_l150_15068

theorem trigonometric_identity (α : ℝ) (h : Real.cos α + Real.sin α = 2 / 3) :
  (Real.sqrt 2 * Real.sin (2 * α - Real.pi / 4) + 1) / (1 + Real.tan α) = - 5 / 9 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l150_15068


namespace NUMINAMATH_GPT_polygon_sides_l150_15000

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360) : 
  n = 8 := 
sorry

end NUMINAMATH_GPT_polygon_sides_l150_15000


namespace NUMINAMATH_GPT_solve_for_x_l150_15090

theorem solve_for_x (x : ℝ) (h : (4 + x) / (6 + x) = (1 + x) / (2 + x)) : x = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l150_15090


namespace NUMINAMATH_GPT_difference_between_extremes_l150_15028

/-- Define the structure of a 3-digit integer and its digits. -/
structure ThreeDigitInteger where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  val : ℕ := 100 * hundreds + 10 * tens + units

/-- Define the problem conditions. -/
def satisfiesConditions (x : ThreeDigitInteger) : Prop :=
  x.hundreds > 0 ∧
  4 * x.hundreds = 2 * x.tens ∧
  2 * x.tens = x.units

/-- Given conditions prove the difference between the two greatest possible values of x is 124. -/
theorem difference_between_extremes :
  ∃ (x₁ x₂ : ThreeDigitInteger), 
    satisfiesConditions x₁ ∧ satisfiesConditions x₂ ∧
    (x₁.val = 248 ∧ x₂.val = 124 ∧ (x₁.val - x₂.val = 124)) :=
sorry

end NUMINAMATH_GPT_difference_between_extremes_l150_15028


namespace NUMINAMATH_GPT_eating_time_175_seconds_l150_15076

variable (Ponchik_time Neznaika_time : ℝ)
variable (Ponchik_rate Neznaika_rate : ℝ)

theorem eating_time_175_seconds
    (hP_rate : Ponchik_rate = 1 / Ponchik_time)
    (hP_time : Ponchik_time = 5)
    (hN_rate : Neznaika_rate = 1 / Neznaika_time)
    (hN_time : Neznaika_time = 7)
    (combined_rate := Ponchik_rate + Neznaika_rate)
    (total_minutes := 1 / combined_rate)
    (total_seconds := total_minutes * 60):
    total_seconds = 175 := by
  sorry

end NUMINAMATH_GPT_eating_time_175_seconds_l150_15076


namespace NUMINAMATH_GPT_min_value_fraction_l150_15095

theorem min_value_fraction {a : ℕ → ℕ} (h1 : a 1 = 10)
    (h2 : ∀ n : ℕ, a (n + 1) - a n = 2 * n) :
    ∃ n : ℕ, (n > 0) ∧ (n - 1 + 10 / n = 16 / 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_fraction_l150_15095


namespace NUMINAMATH_GPT_prob_defective_l150_15024

/-- Assume there are two boxes of components. 
    The first box contains 10 pieces, including 2 defective ones; 
    the second box contains 20 pieces, including 3 defective ones. --/
def box1_total : ℕ := 10
def box1_defective : ℕ := 2
def box2_total : ℕ := 20
def box2_defective : ℕ := 3

/-- Randomly select one box from the two boxes, 
    and then randomly pick 1 component from that box. --/
def prob_select_box : ℚ := 1 / 2

/-- Probability of selecting a defective component given that box 1 was selected. --/
def prob_defective_given_box1 : ℚ := box1_defective / box1_total

/-- Probability of selecting a defective component given that box 2 was selected. --/
def prob_defective_given_box2 : ℚ := box2_defective / box2_total

/-- The probability of selecting a defective component is 7/40. --/
theorem prob_defective :
  prob_select_box * prob_defective_given_box1 + prob_select_box * prob_defective_given_box2 = 7 / 40 :=
sorry

end NUMINAMATH_GPT_prob_defective_l150_15024


namespace NUMINAMATH_GPT_percent_of_x_is_y_l150_15054

variable (x y : ℝ)

theorem percent_of_x_is_y (h : 0.20 * (x - y) = 0.15 * (x + y)) : (y / x) * 100 = 100 / 7 :=
by
  sorry

end NUMINAMATH_GPT_percent_of_x_is_y_l150_15054


namespace NUMINAMATH_GPT_max_value_of_sequence_l150_15050

theorem max_value_of_sequence : 
  ∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → (∃ (a : ℝ), a = (m / (m^2 + 6 : ℝ)) ∧ a ≤ (n / (n^2 + 6 : ℝ))) :=
sorry

end NUMINAMATH_GPT_max_value_of_sequence_l150_15050


namespace NUMINAMATH_GPT_final_price_difference_l150_15006

noncomputable def OP : ℝ := 78.2 / 0.85
noncomputable def IP : ℝ := 78.2 + 0.25 * 78.2
noncomputable def DP : ℝ := 97.75 - 0.10 * 97.75
noncomputable def FP : ℝ := 87.975 + 0.0725 * 87.975

theorem final_price_difference : OP - FP = -2.3531875 := 
by sorry

end NUMINAMATH_GPT_final_price_difference_l150_15006


namespace NUMINAMATH_GPT_gcd_7_nplus2_8_2nplus1_l150_15017

theorem gcd_7_nplus2_8_2nplus1 : 
  ∃ d : ℕ, (∀ n : ℕ, d ∣ (7^(n+2) + 8^(2*n+1))) ∧ (∀ n : ℕ, d = 57) :=
sorry

end NUMINAMATH_GPT_gcd_7_nplus2_8_2nplus1_l150_15017


namespace NUMINAMATH_GPT_jina_teddies_l150_15075

variable (T : ℕ)

def initial_teddies (bunnies : ℕ) (koala : ℕ) (add_teddies : ℕ) (total : ℕ) :=
  T + bunnies + add_teddies + koala

theorem jina_teddies (bunnies : ℕ) (koala : ℕ) (add_teddies : ℕ) (total : ℕ) :
  bunnies = 3 * T ∧ koala = 1 ∧ add_teddies = 2 * bunnies ∧ total = 51 → T = 5 :=
by
  sorry

end NUMINAMATH_GPT_jina_teddies_l150_15075


namespace NUMINAMATH_GPT_total_time_correct_l150_15009

-- Definitions based on problem conditions
def first_time : ℕ := 15
def time_increment : ℕ := 7
def number_of_flights : ℕ := 7

-- Time taken for a specific flight
def time_for_nth_flight (n : ℕ) : ℕ := first_time + (n - 1) * time_increment

-- Sum of the times for the first seven flights
def total_time : ℕ := (number_of_flights * (first_time + time_for_nth_flight number_of_flights)) / 2

-- Statement to be proven
theorem total_time_correct : total_time = 252 := 
by
  sorry

end NUMINAMATH_GPT_total_time_correct_l150_15009


namespace NUMINAMATH_GPT_platform_length_is_500_l150_15008

-- Define the length of the train, the time to cross a tree, and the time to cross a platform as given conditions
def train_length := 1500 -- in meters
def time_to_cross_tree := 120 -- in seconds
def time_to_cross_platform := 160 -- in seconds

-- Define the speed based on the train crossing the tree
def train_speed := train_length / time_to_cross_tree -- in meters/second

-- Define the total distance covered when crossing the platform
def total_distance_crossing_platform (platform_length : ℝ) := train_length + platform_length

-- State the main theorem to prove the platform length is 500 meters
theorem platform_length_is_500 (platform_length : ℝ) :
  (train_speed * time_to_cross_platform = total_distance_crossing_platform platform_length) → platform_length = 500 :=
by
  sorry

end NUMINAMATH_GPT_platform_length_is_500_l150_15008


namespace NUMINAMATH_GPT_number_of_integer_pairs_satisfying_conditions_l150_15034

noncomputable def count_integer_pairs (n m : ℕ) : ℕ := Nat.choose (n-1) (m-1)

theorem number_of_integer_pairs_satisfying_conditions :
  ∃ (a b c x y : ℕ), a + b + c = 55 ∧ a + b + c + x + y = 71 ∧ x + y > a + b + c → count_integer_pairs 55 3 * count_integer_pairs 16 2 = 21465 := sorry

end NUMINAMATH_GPT_number_of_integer_pairs_satisfying_conditions_l150_15034


namespace NUMINAMATH_GPT_flowers_count_l150_15091

theorem flowers_count (save_per_day : ℕ) (days : ℕ) (flower_cost : ℕ) (total_savings : ℕ) (flowers : ℕ) 
    (h1 : save_per_day = 2) 
    (h2 : days = 22) 
    (h3 : flower_cost = 4) 
    (h4 : total_savings = save_per_day * days) 
    (h5 : flowers = total_savings / flower_cost) : 
    flowers = 11 := 
sorry

end NUMINAMATH_GPT_flowers_count_l150_15091


namespace NUMINAMATH_GPT_div_by_seven_iff_multiple_of_three_l150_15059

theorem div_by_seven_iff_multiple_of_three (n : ℕ) (hn : 0 < n) : 
  (7 ∣ (2^n - 1)) ↔ (3 ∣ n) := 
sorry

end NUMINAMATH_GPT_div_by_seven_iff_multiple_of_three_l150_15059


namespace NUMINAMATH_GPT_ben_eggs_left_l150_15089

def initial_eggs : ℕ := 50
def day1_morning : ℕ := 5
def day1_afternoon : ℕ := 4
def day2_morning : ℕ := 8
def day2_evening : ℕ := 3
def day3_afternoon : ℕ := 6
def day3_night : ℕ := 2

theorem ben_eggs_left : initial_eggs - (day1_morning + day1_afternoon + day2_morning + day2_evening + day3_afternoon + day3_night) = 22 := 
by
  sorry

end NUMINAMATH_GPT_ben_eggs_left_l150_15089


namespace NUMINAMATH_GPT_paint_problem_l150_15015

-- Definitions based on conditions
def roomsInitiallyPaintable := 50
def roomsAfterLoss := 40
def cansLost := 5

-- The number of rooms each can could paint
def roomsPerCan := (roomsInitiallyPaintable - roomsAfterLoss) / cansLost

-- The total number of cans originally owned
def originalCans := roomsInitiallyPaintable / roomsPerCan

-- Theorem to prove the number of original cans equals 25
theorem paint_problem : originalCans = 25 := by
  sorry

end NUMINAMATH_GPT_paint_problem_l150_15015


namespace NUMINAMATH_GPT_total_ages_l150_15043

theorem total_ages (bride_age groom_age : ℕ) (h1 : bride_age = 102) (h2 : groom_age = bride_age - 19) : bride_age + groom_age = 185 :=
by
  sorry

end NUMINAMATH_GPT_total_ages_l150_15043


namespace NUMINAMATH_GPT_weight_of_one_apple_l150_15058

-- Conditions
def total_weight_of_bag_with_apples : ℝ := 1.82
def weight_of_empty_bag : ℝ := 0.5
def number_of_apples : ℕ := 6

-- The proposition to prove: the weight of one apple
theorem weight_of_one_apple : (total_weight_of_bag_with_apples - weight_of_empty_bag) / number_of_apples = 0.22 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_one_apple_l150_15058


namespace NUMINAMATH_GPT_min_races_needed_l150_15063

noncomputable def minimum_races (total_horses : ℕ) (max_race_horses : ℕ) : ℕ :=
  if total_horses ≤ max_race_horses then 1 else
  if total_horses % max_race_horses = 0 then total_horses / max_race_horses else total_horses / max_race_horses + 1

/-- We need to show that the minimum number of races required to find the top 3 fastest horses
    among 35 horses, where a maximum of 4 horses can race together at a time, is 10. -/
theorem min_races_needed : minimum_races 35 4 = 10 :=
  sorry

end NUMINAMATH_GPT_min_races_needed_l150_15063


namespace NUMINAMATH_GPT_percent_democrats_l150_15029

/-- The percentage of registered voters in the city who are democrats and republicans -/
def D : ℝ := sorry -- Percent of democrats
def R : ℝ := sorry -- Percent of republicans

-- Given conditions
axiom H1 : D + R = 100
axiom H2 : 0.65 * D + 0.20 * R = 47

-- Statement to prove
theorem percent_democrats : D = 60 :=
by
  sorry

end NUMINAMATH_GPT_percent_democrats_l150_15029


namespace NUMINAMATH_GPT_least_possible_value_of_D_l150_15022

-- Defining the conditions as theorems
theorem least_possible_value_of_D :
  ∃ (A B C D : ℕ), 
  (A + B + C + D) / 4 = 18 ∧
  A = 3 * B ∧
  B = C - 2 ∧
  C = 3 / 2 * D ∧
  (∀ x : ℕ, x ≥ 10 → D = x) := 
sorry

end NUMINAMATH_GPT_least_possible_value_of_D_l150_15022


namespace NUMINAMATH_GPT_odd_times_even_is_even_l150_15077

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_times_even_is_even (a b : ℤ) (h₁ : is_odd a) (h₂ : is_even b) : is_even (a * b) :=
by sorry

end NUMINAMATH_GPT_odd_times_even_is_even_l150_15077


namespace NUMINAMATH_GPT_central_angle_eq_one_l150_15080

noncomputable def radian_measure_of_sector (α r : ℝ) : Prop :=
  α * r = 2 ∧ (1 / 2) * α * r^2 = 2

-- Theorem stating the radian measure of the central angle is 1
theorem central_angle_eq_one (α r : ℝ) (h : radian_measure_of_sector α r) : α = 1 :=
by
  -- provide proof steps here
  sorry

end NUMINAMATH_GPT_central_angle_eq_one_l150_15080


namespace NUMINAMATH_GPT_task_completion_time_l150_15066

noncomputable def work_time (A B C : ℝ) : ℝ := 1 / (A + B + C)

theorem task_completion_time (x y z : ℝ) (h1 : 8 * (x + y) = 1) (h2 : 6 * (x + z) = 1) (h3 : 4.8 * (y + z) = 1) :
    work_time x y z = 4 :=
by
  sorry

end NUMINAMATH_GPT_task_completion_time_l150_15066


namespace NUMINAMATH_GPT_find_x_l150_15016

theorem find_x (x : ℝ) (h : 6 * x + 3 * x + 4 * x + 2 * x = 360) : x = 24 :=
sorry

end NUMINAMATH_GPT_find_x_l150_15016


namespace NUMINAMATH_GPT_students_in_each_grade_l150_15094

theorem students_in_each_grade (total_students : ℕ) (total_grades : ℕ) (students_per_grade : ℕ) :
  total_students = 22800 → total_grades = 304 → students_per_grade = total_students / total_grades → students_per_grade = 75 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_students_in_each_grade_l150_15094


namespace NUMINAMATH_GPT_stationery_store_profit_l150_15087

variable (a : ℝ)

def store_cost : ℝ := 100 * a
def markup_price : ℝ := a * 1.2
def discount_price : ℝ := markup_price a * 0.8

def revenue_first_half : ℝ := 50 * markup_price a
def revenue_second_half : ℝ := 50 * discount_price a
def total_revenue : ℝ := revenue_first_half a + revenue_second_half a

def profit : ℝ := total_revenue a - store_cost a

theorem stationery_store_profit : profit a = 8 * a := 
by sorry

end NUMINAMATH_GPT_stationery_store_profit_l150_15087


namespace NUMINAMATH_GPT_phase_shift_3cos_4x_minus_pi_over_4_l150_15046

theorem phase_shift_3cos_4x_minus_pi_over_4 :
    ∃ (φ : ℝ), y = 3 * Real.cos (4 * x - φ) ∧ φ = π / 16 :=
sorry

end NUMINAMATH_GPT_phase_shift_3cos_4x_minus_pi_over_4_l150_15046


namespace NUMINAMATH_GPT_makenna_garden_larger_by_160_l150_15037

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def karl_length : ℕ := 22
def karl_width : ℕ := 50
def makenna_length : ℕ := 28
def makenna_width : ℕ := 45

def karl_area : ℕ := area karl_length karl_width
def makenna_area : ℕ := area makenna_length makenna_width

theorem makenna_garden_larger_by_160 :
  makenna_area = karl_area + 160 := by
  sorry

end NUMINAMATH_GPT_makenna_garden_larger_by_160_l150_15037


namespace NUMINAMATH_GPT_math_problem_l150_15070

theorem math_problem (x y : ℤ) (h1 : x = 12) (h2 : y = 18) : (x - y) * ((x + y) ^ 2) = -5400 := by
  sorry

end NUMINAMATH_GPT_math_problem_l150_15070


namespace NUMINAMATH_GPT_find_smaller_number_l150_15057

theorem find_smaller_number (n m : ℕ) (h1 : n - m = 58)
  (h2 : n^2 % 100 = m^2 % 100) : m = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l150_15057


namespace NUMINAMATH_GPT_hyejin_math_score_l150_15071

theorem hyejin_math_score :
  let ethics := 82
  let korean_language := 90
  let science := 88
  let social_studies := 84
  let avg_score := 88
  let total_subjects := 5
  ∃ (M : ℕ), (ethics + korean_language + science + social_studies + M) / total_subjects = avg_score := by
    sorry

end NUMINAMATH_GPT_hyejin_math_score_l150_15071


namespace NUMINAMATH_GPT_volunteer_arrangements_l150_15019

theorem volunteer_arrangements (students : Fin 5 → String) (events : Fin 3 → String)
  (A : String) (high_jump : String)
  (h : ∀ (arrange : Fin 3 → Fin 5), ¬(students (arrange 0) = A ∧ events 0 = high_jump)) :
  ∃! valid_arrangements, valid_arrangements = 48 :=
by
  sorry

end NUMINAMATH_GPT_volunteer_arrangements_l150_15019


namespace NUMINAMATH_GPT_number_ordering_l150_15010

theorem number_ordering : (10^5 < 2^20) ∧ (2^20 < 5^10) :=
by {
  -- We place the proof steps here
  sorry
}

end NUMINAMATH_GPT_number_ordering_l150_15010


namespace NUMINAMATH_GPT_barbara_candies_left_l150_15048

def initial_candies: ℝ := 18.5
def candies_used_to_make_dessert: ℝ := 4.2
def candies_received_from_friend: ℝ := 6.8
def candies_eaten: ℝ := 2.7

theorem barbara_candies_left : 
  initial_candies - candies_used_to_make_dessert + candies_received_from_friend - candies_eaten = 18.4 := 
by
  sorry

end NUMINAMATH_GPT_barbara_candies_left_l150_15048


namespace NUMINAMATH_GPT_roots_equal_condition_l150_15098

theorem roots_equal_condition (a c : ℝ) (h : a ≠ 0) :
    (∀ x1 x2, (a * x1 * x1 + 4 * a * x1 + c = 0) ∧ (a * x2 * x2 + 4 * a * x2 + c = 0) → x1 = x2) ↔ c = 4 * a := 
by
  sorry

end NUMINAMATH_GPT_roots_equal_condition_l150_15098


namespace NUMINAMATH_GPT_shaded_region_area_l150_15049

noncomputable def radius_large : ℝ := 10
noncomputable def radius_small : ℝ := 4

theorem shaded_region_area :
  let area_large := Real.pi * radius_large^2 
  let area_small := Real.pi * radius_small^2 
  (area_large - 2 * area_small) = 68 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l150_15049


namespace NUMINAMATH_GPT_favorite_movies_total_hours_l150_15039

theorem favorite_movies_total_hours (michael_hrs joyce_hrs nikki_hrs ryn_hrs sam_hrs alex_hrs : ℕ)
  (H1 : nikki_hrs = 30)
  (H2 : michael_hrs = nikki_hrs / 3)
  (H3 : joyce_hrs = michael_hrs + 2)
  (H4 : ryn_hrs = (4 * nikki_hrs) / 5)
  (H5 : sam_hrs = (3 * joyce_hrs) / 2)
  (H6 : alex_hrs = 2 * michael_hrs) :
  michael_hrs + joyce_hrs + nikki_hrs + ryn_hrs + sam_hrs + alex_hrs = 114 := 
sorry

end NUMINAMATH_GPT_favorite_movies_total_hours_l150_15039


namespace NUMINAMATH_GPT_ball_hits_ground_l150_15025

noncomputable def ball_height (t : ℝ) : ℝ := -9 * t^2 + 15 * t + 72

theorem ball_hits_ground :
  (∃ t : ℝ, t = (5 + Real.sqrt 313) / 6 ∧ ball_height t = 0) :=
sorry

end NUMINAMATH_GPT_ball_hits_ground_l150_15025


namespace NUMINAMATH_GPT_pentagon_rectangle_ratio_l150_15041

theorem pentagon_rectangle_ratio (p w l : ℝ) (h₁ : 5 * p = 20) (h₂ : l = 2 * w) (h₃ : 2 * l + 2 * w = 20) : p / w = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_rectangle_ratio_l150_15041


namespace NUMINAMATH_GPT_num_machines_first_scenario_l150_15082

theorem num_machines_first_scenario (r : ℝ) (n : ℕ) :
  (∀ r, (2 : ℝ) * r * 24 = 1) →
  (∀ r, (n : ℝ) * r * 6 = 1) →
  n = 8 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_num_machines_first_scenario_l150_15082


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l150_15072

theorem arithmetic_sequence_general_term (x : ℕ)
  (t1 t2 t3 : ℤ)
  (h1 : t1 = x - 1)
  (h2 : t2 = x + 1)
  (h3 : t3 = 2 * x + 3) :
  (∃ a : ℕ → ℤ, a 1 = t1 ∧ a 2 = t2 ∧ a 3 = t3 ∧ ∀ n, a n = 2 * n - 3) := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l150_15072


namespace NUMINAMATH_GPT_problem_statement_l150_15003

theorem problem_statement (d : ℕ) (h1 : d > 0) (h2 : d ∣ (5 + 2022^2022)) :
  (∃ x y : ℤ, d = 2 * x^2 + 2 * x * y + 3 * y^2) ↔ (d % 20 = 3 ∨ d % 20 = 7) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l150_15003


namespace NUMINAMATH_GPT_total_bathing_suits_l150_15013

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969

theorem total_bathing_suits : men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end NUMINAMATH_GPT_total_bathing_suits_l150_15013


namespace NUMINAMATH_GPT_SetC_not_right_angled_triangle_l150_15035

theorem SetC_not_right_angled_triangle :
  ¬ (7^2 + 24^2 = 26^2) :=
by 
  have h : 7^2 + 24^2 ≠ 26^2 := by decide
  exact h

end NUMINAMATH_GPT_SetC_not_right_angled_triangle_l150_15035


namespace NUMINAMATH_GPT_brownies_pieces_count_l150_15011

theorem brownies_pieces_count
  (pan_length pan_width piece_length piece_width : ℕ)
  (h1 : pan_length = 24)
  (h2 : pan_width = 15)
  (h3 : piece_length = 3)
  (h4 : piece_width = 2) :
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
by
  sorry

end NUMINAMATH_GPT_brownies_pieces_count_l150_15011


namespace NUMINAMATH_GPT_slope_of_line_through_origin_and_A_l150_15005

theorem slope_of_line_through_origin_and_A :
  ∀ (x1 y1 x2 y2 : ℝ), (x1 = 0) → (y1 = 0) → (x2 = -2) → (y2 = -2) →
  (y2 - y1) / (x2 - x1) = 1 :=
by intros; sorry

end NUMINAMATH_GPT_slope_of_line_through_origin_and_A_l150_15005


namespace NUMINAMATH_GPT_angle_bisector_inequality_l150_15061

noncomputable def triangle_ABC (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C] (AB BC CA AK CM AM MK KC : ℝ) 
  (Hbisector_CM : BM / MA = BC / CA)
  (Hbisector_AK : BK / KC = AB / AC)
  (Hcondition : AB > BC) : Prop :=
  AM > MK ∧ MK > KC

theorem angle_bisector_inequality (A B C K M : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (AB BC CA AK CM AM MK KC : ℝ)
  (Hbisector_CM : BM / MA = BC / CA)
  (Hbisector_AK : BK / KC = AB / AC)
  (Hcondition : AB > BC) : AM > MK ∧ MK > KC :=
by
  sorry

end NUMINAMATH_GPT_angle_bisector_inequality_l150_15061


namespace NUMINAMATH_GPT_part_a_constant_part_b_inequality_l150_15085

open Real

noncomputable def cubic_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem part_a_constant (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1 * x2 / x3^2) + cubic_root (x2 * x3 / x1^2) + cubic_root (x3 * x1 / x2^2)) = 
  const_value := sorry

theorem part_b_inequality (x1 x2 x3 : ℝ) (h : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  (cubic_root (x1^2 / (x2 * x3)) + cubic_root (x2^2 / (x3 * x1)) + cubic_root (x3^2 / (x1 * x2))) < (-15 / 4) := sorry

end NUMINAMATH_GPT_part_a_constant_part_b_inequality_l150_15085


namespace NUMINAMATH_GPT_num_real_solutions_abs_eq_l150_15078

theorem num_real_solutions_abs_eq :
  (∃ x y : ℝ, x ≠ y ∧ |x-1| = |x-2| + |x-3| + |x-4| 
    ∧ |y-1| = |y-2| + |y-3| + |y-4| 
    ∧ ∀ z : ℝ, |z-1| = |z-2| + |z-3| + |z-4| → (z = x ∨ z = y)) := sorry

end NUMINAMATH_GPT_num_real_solutions_abs_eq_l150_15078


namespace NUMINAMATH_GPT_jade_transactions_correct_l150_15060

-- Definitions for the conditions
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := mabel_transactions + (mabel_transactions * 10 / 100)
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := cal_transactions + 16

-- The theorem stating what we want to prove
theorem jade_transactions_correct : jade_transactions = 82 := by
  sorry

end NUMINAMATH_GPT_jade_transactions_correct_l150_15060


namespace NUMINAMATH_GPT_correct_conclusion_l150_15051

noncomputable def proof_problem (a x : ℝ) (x1 x2 : ℝ) :=
  (a * (x - 1) * (x - 3) + 2 > 0 ∧ x1 < x2 ∧ 
   (∀ x, a * (x - 1) * (x - 3) + 2 > 0 ↔ x < x1 ∨ x > x2)) →
  (x1 + x2 = 4 ∧ 3 < x1 * x2 ∧ x1 * x2 < 4 ∧ 
   (∀ x, ((3 * a + 2) * x^2 - 4 * a * x + a < 0) ↔ (1 / x2 < x ∧ x < 1 / x1)))

theorem correct_conclusion (a x x1 x2 : ℝ) : 
proof_problem a x x1 x2 :=
by 
  unfold proof_problem 
  sorry

end NUMINAMATH_GPT_correct_conclusion_l150_15051


namespace NUMINAMATH_GPT_initial_number_of_earning_members_l150_15018

theorem initial_number_of_earning_members (n : ℕ) 
  (h1 : 840 * n - 650 * (n - 1) = 1410) : n = 4 :=
by {
  -- Proof omitted
  sorry
}

end NUMINAMATH_GPT_initial_number_of_earning_members_l150_15018


namespace NUMINAMATH_GPT_rod_length_l150_15079

theorem rod_length (pieces : ℕ) (length_per_piece_cm : ℕ) (total_length_m : ℝ) :
  pieces = 35 → length_per_piece_cm = 85 → total_length_m = 29.75 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_rod_length_l150_15079


namespace NUMINAMATH_GPT_first_tribe_term_is_longer_l150_15086

def years_to_days_first_tribe (years : ℕ) : ℕ := 
  years * 12 * 30

def months_to_days_first_tribe (months : ℕ) : ℕ :=
  months * 30

def total_days_first_tribe (years months days : ℕ) : ℕ :=
  (years_to_days_first_tribe years) + (months_to_days_first_tribe months) + days

def years_to_days_second_tribe (years : ℕ) : ℕ := 
  years * 13 * 4 * 7

def moons_to_days_second_tribe (moons : ℕ) : ℕ :=
  moons * 4 * 7

def weeks_to_days_second_tribe (weeks : ℕ) : ℕ :=
  weeks * 7

def total_days_second_tribe (years moons weeks days : ℕ) : ℕ :=
  (years_to_days_second_tribe years) + (moons_to_days_second_tribe moons) + (weeks_to_days_second_tribe weeks) + days

theorem first_tribe_term_is_longer :
  total_days_first_tribe 7 1 18 > total_days_second_tribe 6 12 1 3 :=
by
  sorry

end NUMINAMATH_GPT_first_tribe_term_is_longer_l150_15086


namespace NUMINAMATH_GPT_decimal_to_fraction_l150_15074

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end NUMINAMATH_GPT_decimal_to_fraction_l150_15074


namespace NUMINAMATH_GPT_option_C_is_quadratic_l150_15088

-- Define what it means for an equation to be quadratic
def is_quadratic (p : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ (x : ℝ), p x ↔ a*x^2 + b*x + c = 0

-- Define the equation in option C
def option_C (x : ℝ) : Prop := (x - 1) * (x - 2) = 0

-- The theorem we need to prove
theorem option_C_is_quadratic : is_quadratic option_C :=
  sorry

end NUMINAMATH_GPT_option_C_is_quadratic_l150_15088


namespace NUMINAMATH_GPT_two_cubic_meters_to_cubic_feet_l150_15052

theorem two_cubic_meters_to_cubic_feet :
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  2 * cubic_meter_to_cubic_feet = 70.6294 :=
by
  let meter_to_feet := 3.28084
  let cubic_meter_to_cubic_feet := meter_to_feet ^ 3
  have h : 2 * cubic_meter_to_cubic_feet = 70.6294 := sorry
  exact h

end NUMINAMATH_GPT_two_cubic_meters_to_cubic_feet_l150_15052


namespace NUMINAMATH_GPT_count_two_digit_perfect_squares_divisible_by_4_l150_15038

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end NUMINAMATH_GPT_count_two_digit_perfect_squares_divisible_by_4_l150_15038


namespace NUMINAMATH_GPT_train_speed_l150_15040

noncomputable def jogger_speed : ℝ := 9 -- speed in km/hr
noncomputable def jogger_distance : ℝ := 150 / 1000 -- distance in km
noncomputable def train_length : ℝ := 100 / 1000 -- length in km
noncomputable def time_to_pass : ℝ := 25 -- time in seconds

theorem train_speed 
  (v_j : ℝ := jogger_speed)
  (d_j : ℝ := jogger_distance)
  (L : ℝ := train_length)
  (t : ℝ := time_to_pass) :
  (train_speed_in_kmh : ℝ) = 36 :=
by 
  sorry

end NUMINAMATH_GPT_train_speed_l150_15040


namespace NUMINAMATH_GPT_problem_1_problem_2_l150_15055

theorem problem_1 (h : Real.tan (α / 2) = 2) : Real.tan (α + Real.arctan 1) = -1/7 :=
by
  sorry

theorem problem_2 (h : Real.tan (α / 2) = 2) : (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l150_15055


namespace NUMINAMATH_GPT_probability_black_balls_l150_15084

variable {m1 m2 k1 k2 : ℕ}

/-- Given conditions:
  1. The total number of balls in both urns is 25.
  2. The probability of drawing one white ball from each urn is 0.54.
To prove: The probability of both drawn balls being black is 0.04.
-/
theorem probability_black_balls : 
  m1 + m2 = 25 → 
  (k1 * k2) * 50 = 27 * m1 * m2 → 
  ((m1 - k1) * (m2 - k2) : ℚ) / (m1 * m2) = 0.04 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_probability_black_balls_l150_15084


namespace NUMINAMATH_GPT_walkway_area_296_l150_15012

theorem walkway_area_296 :
  let bed_length := 4
  let bed_width := 3
  let num_rows := 4
  let num_columns := 3
  let walkway_width := 2
  let total_bed_area := num_rows * num_columns * bed_length * bed_width
  let total_garden_width := num_columns * bed_length + (num_columns + 1) * walkway_width
  let total_garden_height := num_rows * bed_width + (num_rows + 1) * walkway_width
  let total_garden_area := total_garden_width * total_garden_height
  let total_walkway_area := total_garden_area - total_bed_area
  total_walkway_area = 296 :=
by 
  sorry

end NUMINAMATH_GPT_walkway_area_296_l150_15012


namespace NUMINAMATH_GPT_ceil_floor_difference_l150_15099

open Int

theorem ceil_floor_difference : 
  (Int.ceil (15 / 8 * (-34 / 4)) - Int.floor (15 / 8 * Int.floor (-34 / 4))) = 2 := 
by
  sorry

end NUMINAMATH_GPT_ceil_floor_difference_l150_15099


namespace NUMINAMATH_GPT_only_one_tuple_exists_l150_15073

theorem only_one_tuple_exists :
  ∃! (x : Fin 15 → ℝ),
    (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2
    + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7 - x 8)^2 + (x 8 - x 9)^2
    + (x 9 - x 10)^2 + (x 10 - x 11)^2 + (x 11 - x 12)^2 + (x 12 - x 13)^2
    + (x 13 - x 14)^2 + (x 14)^2 = 1 / 16 := by
  sorry

end NUMINAMATH_GPT_only_one_tuple_exists_l150_15073


namespace NUMINAMATH_GPT_power_mod_remainder_l150_15056

theorem power_mod_remainder :
  (7 ^ 2023) % 17 = 16 :=
sorry

end NUMINAMATH_GPT_power_mod_remainder_l150_15056


namespace NUMINAMATH_GPT_solve_r_l150_15047

theorem solve_r (r : ℚ) :
  (r^2 - 5*r + 4) / (r^2 - 8*r + 7) = (r^2 - 2*r - 15) / (r^2 - r - 20) →
  r = -5/4 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_solve_r_l150_15047


namespace NUMINAMATH_GPT_rectangular_cube_length_l150_15092

theorem rectangular_cube_length (L : ℝ) (h1 : 2 * (L * 2) + 2 * (L * 0.5) + 2 * (2 * 0.5) = 24) : L = 4.6 := 
by {
  sorry
}

end NUMINAMATH_GPT_rectangular_cube_length_l150_15092


namespace NUMINAMATH_GPT_binom_9_5_l150_15030

open Nat

-- Definition of binomial coefficient
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Theorem to prove binom(9, 5) = 126
theorem binom_9_5 : binom 9 5 = 126 := by
  sorry

end NUMINAMATH_GPT_binom_9_5_l150_15030


namespace NUMINAMATH_GPT_min_S6_minus_S4_l150_15067

variable {a₁ a₂ q : ℝ} (h1 : q > 1) (h2 : (q^2 - 1) * (a₁ + a₂) = 3)

theorem min_S6_minus_S4 : 
  ∃ (a₁ a₂ q : ℝ), q > 1 ∧ (q^2 - 1) * (a₁ + a₂) = 3 ∧ (q^4 * (a₁ + a₂) - (a₁ + a₂ + a₂ * q + a₂ * q^2) = 12) := sorry

end NUMINAMATH_GPT_min_S6_minus_S4_l150_15067


namespace NUMINAMATH_GPT_opposite_of_neg_five_l150_15033

/-- Definition of the opposite of a number -/
def opposite (a : Int) : Int := -a

theorem opposite_of_neg_five : opposite (-5) = 5 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_five_l150_15033


namespace NUMINAMATH_GPT_complement_union_l150_15002

open Set

variable (U : Set ℝ) (A B : Set ℝ)

def C_U (s : Set ℝ) : Set ℝ := U \ s

theorem complement_union (U : Set ℝ) (A B : Set ℝ) (hU : U = univ) (hA : A = { x | x < 0 }) (hB : B = { x | x ≥ 2 }) :
  C_U U (A ∪ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l150_15002


namespace NUMINAMATH_GPT_sum_of_squares_of_sum_and_difference_l150_15001

theorem sum_of_squares_of_sum_and_difference (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 8) : 
  (x + y)^2 + (x - y)^2 = 640 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_sum_and_difference_l150_15001


namespace NUMINAMATH_GPT_which_set_forms_triangle_l150_15027

def satisfies_triangle_inequality (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem which_set_forms_triangle : 
  satisfies_triangle_inequality 4 3 6 ∧ 
  ¬ satisfies_triangle_inequality 1 2 3 ∧ 
  ¬ satisfies_triangle_inequality 7 8 16 ∧ 
  ¬ satisfies_triangle_inequality 9 10 20 :=
by
  sorry

end NUMINAMATH_GPT_which_set_forms_triangle_l150_15027


namespace NUMINAMATH_GPT_least_four_digit_perfect_square_and_cube_l150_15021

theorem least_four_digit_perfect_square_and_cube :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ m1 : ℕ, n = m1^2) ∧ (∃ m2 : ℕ, n = m2^3) ∧ n = 4096 := sorry

end NUMINAMATH_GPT_least_four_digit_perfect_square_and_cube_l150_15021


namespace NUMINAMATH_GPT_ellipse_focal_length_l150_15004

theorem ellipse_focal_length {m : ℝ} : 
  (m > 2 ∧ 4 ≤ 10 - m ∧ 4 ≤ m - 2) → 
  (10 - m - (m - 2) = 4) ∨ (m - 2 - (10 - m) = 4) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focal_length_l150_15004


namespace NUMINAMATH_GPT_train_cross_first_platform_l150_15093

noncomputable def time_to_cross_first_platform (L_t L_p1 L_p2 t2 : ℕ) : ℕ :=
  (L_t + L_p1) / ((L_t + L_p2) / t2)

theorem train_cross_first_platform :
  time_to_cross_first_platform 100 200 300 20 = 15 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_first_platform_l150_15093


namespace NUMINAMATH_GPT_order_theorems_l150_15023

theorem order_theorems : 
  ∃ a b c d e f g : String,
    (a = "H") ∧ (b = "M") ∧ (c = "P") ∧ (d = "C") ∧ 
    (e = "V") ∧ (f = "S") ∧ (g = "E") ∧
    (a = "Heron's Theorem") ∧
    (b = "Menelaus' Theorem") ∧
    (c = "Pascal's Theorem") ∧
    (d = "Ceva's Theorem") ∧
    (e = "Varignon's Theorem") ∧
    (f = "Stewart's Theorem") ∧
    (g = "Euler's Theorem") := 
  sorry

end NUMINAMATH_GPT_order_theorems_l150_15023


namespace NUMINAMATH_GPT_percentage_repeated_digits_five_digit_numbers_l150_15045

theorem percentage_repeated_digits_five_digit_numbers : 
  let total_five_digit_numbers := 90000
  let non_repeated_digits_number := 9 * 9 * 8 * 7 * 6
  let repeated_digits_number := total_five_digit_numbers - non_repeated_digits_number
  let y := (repeated_digits_number.toFloat / total_five_digit_numbers.toFloat) * 100 
  y = 69.8 :=
by
  sorry

end NUMINAMATH_GPT_percentage_repeated_digits_five_digit_numbers_l150_15045


namespace NUMINAMATH_GPT_Adam_total_candy_l150_15031

theorem Adam_total_candy :
  (2 + 5) * 4 = 28 := 
by 
  sorry

end NUMINAMATH_GPT_Adam_total_candy_l150_15031


namespace NUMINAMATH_GPT_pencil_length_after_sharpening_l150_15083

-- Definition of the initial length of the pencil
def initial_length : ℕ := 22

-- Definition of the amount sharpened each day
def sharpened_each_day : ℕ := 2

-- Final length of the pencil after sharpening on Monday and Tuesday
def final_length (initial_length : ℕ) (sharpened_each_day : ℕ) : ℕ :=
  initial_length - sharpened_each_day * 2

-- Theorem stating that the final length is 18 inches
theorem pencil_length_after_sharpening : final_length initial_length sharpened_each_day = 18 := by
  sorry

end NUMINAMATH_GPT_pencil_length_after_sharpening_l150_15083


namespace NUMINAMATH_GPT_AM_GM_inequality_AM_GM_equality_l150_15026

theorem AM_GM_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 ≤ Real.sqrt ((a^2 + b^2 + c^2) / 3) :=
by
  sorry

theorem AM_GM_equality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) / 3 = Real.sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_inequality_AM_GM_equality_l150_15026
