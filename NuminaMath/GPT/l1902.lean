import Mathlib

namespace NUMINAMATH_GPT_base_four_30121_eq_793_l1902_190247

-- Definition to convert a base-four (radix 4) number 30121_4 to its base-ten equivalent
def base_four_to_base_ten (d4 d3 d2 d1 d0 : ℕ) : ℕ :=
  d4 * 4^4 + d3 * 4^3 + d2 * 4^2 + d1 * 4^1 + d0 * 4^0

theorem base_four_30121_eq_793 : base_four_to_base_ten 3 0 1 2 1 = 793 := 
by
  sorry

end NUMINAMATH_GPT_base_four_30121_eq_793_l1902_190247


namespace NUMINAMATH_GPT_situation1_correct_situation2_correct_situation3_correct_l1902_190264

noncomputable def situation1 : Nat :=
  let choices_for_A := 4
  let remaining_perm := Nat.factorial 6
  choices_for_A * remaining_perm

theorem situation1_correct : situation1 = 2880 := by
  sorry

noncomputable def situation2 : Nat :=
  let permutations_A_B := Nat.factorial 2
  let remaining_perm := Nat.factorial 5
  permutations_A_B * remaining_perm

theorem situation2_correct : situation2 = 240 := by
  sorry

noncomputable def situation3 : Nat :=
  let perm_boys := Nat.factorial 3
  let perm_girls := Nat.factorial 4
  perm_boys * perm_girls

theorem situation3_correct : situation3 = 144 := by
  sorry

end NUMINAMATH_GPT_situation1_correct_situation2_correct_situation3_correct_l1902_190264


namespace NUMINAMATH_GPT_percentage_lower_grades_have_cars_l1902_190239

-- Definitions for the conditions
def n_seniors : ℕ := 300
def p_car : ℚ := 0.50
def n_lower : ℕ := 900
def p_total : ℚ := 0.20

-- Definition for the number of students who have cars in the lower grades
def n_cars_lower : ℚ := 
  let total_students := n_seniors + n_lower
  let total_cars := p_total * total_students
  total_cars - (p_car * n_seniors)

-- Prove the percentage of freshmen, sophomores, and juniors who have cars
theorem percentage_lower_grades_have_cars : 
  (n_cars_lower / n_lower) * 100 = 10 := 
by sorry

end NUMINAMATH_GPT_percentage_lower_grades_have_cars_l1902_190239


namespace NUMINAMATH_GPT_cevian_sum_equals_two_l1902_190224

-- Definitions based on conditions
variables {A B C D E F O : Type*}
variables (AD BE CF : ℝ) (R : ℝ)
variables (circumcenter_O : O = circumcenter A B C)
variables (intersect_AD_O : AD = abs ((line A D).proj O))
variables (intersect_BE_O : BE = abs ((line B E).proj O))
variables (intersect_CF_O : CF = abs ((line C F).proj O))

-- Prove the main statement
theorem cevian_sum_equals_two (h : circumcenter_O ∧ intersect_AD_O ∧ intersect_BE_O ∧ intersect_CF_O) :
  1 / AD + 1 / BE + 1 / CF = 2 / R :=
sorry

end NUMINAMATH_GPT_cevian_sum_equals_two_l1902_190224


namespace NUMINAMATH_GPT_smaller_angle_clock_8_10_l1902_190210

/-- The measure of the smaller angle formed by the hour and minute hands of a clock at 8:10 p.m. is 175 degrees. -/
theorem smaller_angle_clock_8_10 : 
  let full_circle := 360
  let hour_increment := 30
  let hour_angle_8 := 8 * hour_increment
  let minute_angle_increment := 6
  let hour_hand_adjustment := 10 * (hour_increment / 60)
  let hour_hand_position := hour_angle_8 + hour_hand_adjustment
  let minute_hand_position := 10 * minute_angle_increment
  let angle_difference := if hour_hand_position > minute_hand_position 
                          then hour_hand_position - minute_hand_position 
                          else minute_hand_position - hour_hand_position  
  let smaller_angle := if 2 * angle_difference > full_circle 
                       then full_circle - angle_difference 
                       else angle_difference
  smaller_angle = 175 :=
by 
  sorry

end NUMINAMATH_GPT_smaller_angle_clock_8_10_l1902_190210


namespace NUMINAMATH_GPT_equality_holds_iff_l1902_190231

theorem equality_holds_iff (k t x y z : ℤ) (h_arith_prog : x + z = 2 * y) :
  (k * y^3 = x^3 + z^3) ↔ (k = 2 * (3 * t^2 + 1)) := by
  sorry

end NUMINAMATH_GPT_equality_holds_iff_l1902_190231


namespace NUMINAMATH_GPT_combination_count_l1902_190244

-- Definitions from conditions
def packagingPapers : Nat := 10
def ribbons : Nat := 4
def stickers : Nat := 5

-- Proof problem statement
theorem combination_count : packagingPapers * ribbons * stickers = 200 := 
by
  sorry

end NUMINAMATH_GPT_combination_count_l1902_190244


namespace NUMINAMATH_GPT_find_highway_speed_l1902_190278

def car_local_distance := 40
def car_local_speed := 20
def car_highway_distance := 180
def average_speed := 44
def speed_of_car_on_highway := 60

theorem find_highway_speed :
  car_local_distance / car_local_speed + car_highway_distance / speed_of_car_on_highway = (car_local_distance + car_highway_distance) / average_speed :=
by
  sorry

end NUMINAMATH_GPT_find_highway_speed_l1902_190278


namespace NUMINAMATH_GPT_find_numbers_l1902_190200

theorem find_numbers (u v : ℝ) (h1 : u^2 + v^2 = 20) (h2 : u * v = 8) :
  (u = 2 ∧ v = 4) ∨ (u = 4 ∧ v = 2) ∨ (u = -2 ∧ v = -4) ∨ (u = -4 ∧ v = -2) := by
sorry

end NUMINAMATH_GPT_find_numbers_l1902_190200


namespace NUMINAMATH_GPT_triangle_relations_l1902_190253

theorem triangle_relations (A B C_1 C_2 C_3 : ℝ)
  (h1 : B > A)
  (h2 : C_2 > C_1 ∧ C_2 > C_3)
  (h3 : A + C_1 = 90) 
  (h4 : C_2 = 90)
  (h5 : B + C_3 = 90) :
  C_1 - C_3 = B - A :=
sorry

end NUMINAMATH_GPT_triangle_relations_l1902_190253


namespace NUMINAMATH_GPT_hyperbola_a_solution_l1902_190295

noncomputable def hyperbola_a_value (a : ℝ) : Prop :=
  (a > 0) ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / 2) = 1) ∧ (∃ e : ℝ, e = 2)

theorem hyperbola_a_solution : ∃ a : ℝ, hyperbola_a_value a ∧ a = (Real.sqrt 6) / 3 :=
  by
    sorry

end NUMINAMATH_GPT_hyperbola_a_solution_l1902_190295


namespace NUMINAMATH_GPT_distance_between_first_and_last_student_l1902_190243

theorem distance_between_first_and_last_student 
  (n : ℕ) (d : ℕ)
  (students : n = 30) 
  (distance_between_students : d = 3) : 
  n - 1 * d = 87 := 
by
  sorry

end NUMINAMATH_GPT_distance_between_first_and_last_student_l1902_190243


namespace NUMINAMATH_GPT_algebraic_expression_is_200_l1902_190241

-- Define the condition
def satisfies_ratio (x : ℕ) : Prop :=
  x / 10 = 20

-- The proof problem statement
theorem algebraic_expression_is_200 : ∃ x : ℕ, satisfies_ratio x ∧ x = 200 :=
by
  -- Providing the necessary proof infrastructure
  use 200
  -- Assuming the proof is correct
  sorry


end NUMINAMATH_GPT_algebraic_expression_is_200_l1902_190241


namespace NUMINAMATH_GPT_complete_the_square_l1902_190259

theorem complete_the_square (x : ℝ) : x^2 - 2 * x - 1 = 0 -> (x - 1)^2 = 2 := by
  sorry

end NUMINAMATH_GPT_complete_the_square_l1902_190259


namespace NUMINAMATH_GPT_dogs_food_consumption_l1902_190201

def cups_per_meal_per_dog : ℝ := 1.5
def number_of_dogs : ℝ := 2
def meals_per_day : ℝ := 3
def cups_per_pound : ℝ := 2.25

theorem dogs_food_consumption : 
  ((cups_per_meal_per_dog * number_of_dogs) * meals_per_day) / cups_per_pound = 4 := 
by
  sorry

end NUMINAMATH_GPT_dogs_food_consumption_l1902_190201


namespace NUMINAMATH_GPT_scout_weekend_earnings_l1902_190215

-- Definitions for conditions
def base_pay_per_hour : ℝ := 10.00
def tip_saturday : ℝ := 5.00
def tip_sunday_low : ℝ := 3.00
def tip_sunday_high : ℝ := 7.00
def transportation_cost_per_delivery : ℝ := 1.00
def hours_worked_saturday : ℝ := 6
def deliveries_saturday : ℝ := 5
def hours_worked_sunday : ℝ := 8
def deliveries_sunday : ℝ := 10
def deliveries_sunday_low_tip : ℝ := 5
def deliveries_sunday_high_tip : ℝ := 5
def holiday_multiplier : ℝ := 2

-- Calculation of total earnings for the weekend after transportation costs
theorem scout_weekend_earnings : 
  let base_pay_saturday := hours_worked_saturday * base_pay_per_hour
  let tips_saturday := deliveries_saturday * tip_saturday
  let transportation_costs_saturday := deliveries_saturday * transportation_cost_per_delivery
  let total_earnings_saturday := base_pay_saturday + tips_saturday - transportation_costs_saturday

  let base_pay_sunday := hours_worked_sunday * base_pay_per_hour * holiday_multiplier
  let tips_sunday := deliveries_sunday_low_tip * tip_sunday_low + deliveries_sunday_high_tip * tip_sunday_high
  let transportation_costs_sunday := deliveries_sunday * transportation_cost_per_delivery
  let total_earnings_sunday := base_pay_sunday + tips_sunday - transportation_costs_sunday

  let total_earnings_weekend := total_earnings_saturday + total_earnings_sunday

  total_earnings_weekend = 280.00 :=
by
  -- Add detailed proof here
  sorry

end NUMINAMATH_GPT_scout_weekend_earnings_l1902_190215


namespace NUMINAMATH_GPT_obtuse_triangle_iff_distinct_real_roots_l1902_190227

theorem obtuse_triangle_iff_distinct_real_roots
  (A B C : ℝ)
  (h_triangle : 2 * A + B = Real.pi)
  (h_isosceles : A = C) :
  (B > Real.pi / 2) ↔ (B^2 - 4 * A * C > 0) :=
sorry

end NUMINAMATH_GPT_obtuse_triangle_iff_distinct_real_roots_l1902_190227


namespace NUMINAMATH_GPT_part_1_solution_set_part_2_a_range_l1902_190228

-- Define the function f
def f (x a : ℝ) := |x - a^2| + |x - 2 * a + 1|

-- Part (1)
theorem part_1_solution_set (x : ℝ) : {x | x ≤ 3 / 2 ∨ x ≥ 11 / 2} = 
  {x | f x 2 ≥ 4} :=
sorry

-- Part (2)
theorem part_2_a_range (a : ℝ) : 
  {a | (a - 1)^2 ≥ 4} = {a | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_GPT_part_1_solution_set_part_2_a_range_l1902_190228


namespace NUMINAMATH_GPT_eight_digit_number_div_by_9_l1902_190266

theorem eight_digit_number_div_by_9 (n : ℕ) (hn : 0 ≤ n ∧ n ≤ 9)
  (h : (8 + 5 + 4 + n + 5 + 2 + 6 + 8) % 9 = 0) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_eight_digit_number_div_by_9_l1902_190266


namespace NUMINAMATH_GPT_prove_expression_value_l1902_190275

theorem prove_expression_value (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_prove_expression_value_l1902_190275


namespace NUMINAMATH_GPT_total_letters_in_names_is_33_l1902_190203

def letters_in_names (jonathan_first_name_letters : Nat) 
                     (jonathan_surname_letters : Nat)
                     (sister_first_name_letters : Nat) 
                     (sister_second_name_letters : Nat) : Nat :=
  jonathan_first_name_letters + jonathan_surname_letters +
  sister_first_name_letters + sister_second_name_letters

theorem total_letters_in_names_is_33 :
  letters_in_names 8 10 5 10 = 33 :=
by 
  sorry

end NUMINAMATH_GPT_total_letters_in_names_is_33_l1902_190203


namespace NUMINAMATH_GPT_sample_size_is_correct_l1902_190260

-- Define the school and selection conditions
def total_classes := 40
def students_per_class := 50

-- Given condition
def selected_students := 150

-- Theorem statement
theorem sample_size_is_correct : selected_students = 150 := 
by 
  sorry

end NUMINAMATH_GPT_sample_size_is_correct_l1902_190260


namespace NUMINAMATH_GPT_Hilt_payment_l1902_190222

def total_cost : ℝ := 2.05
def nickel_value : ℝ := 0.05
def dime_value : ℝ := 0.10

theorem Hilt_payment (n : ℕ) (h : n_n = n ∧ n_d = n) 
  (h_nickel : ℝ := n * nickel_value)
  (h_dime : ℝ := n * dime_value): 
  (n * nickel_value + n * dime_value = total_cost) 
  →  n = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_Hilt_payment_l1902_190222


namespace NUMINAMATH_GPT_outfit_choices_l1902_190272

theorem outfit_choices:
  let shirts := 8
  let pants := 8
  let hats := 8
  -- Each has 8 different colors
  -- No repetition of color within type of clothing
  -- Refuse to wear same color shirt and pants
  (shirts * pants * hats) - (shirts * hats) = 448 := 
sorry

end NUMINAMATH_GPT_outfit_choices_l1902_190272


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1902_190290

theorem simplify_and_evaluate (m : ℤ) (h : m = -2) :
  let expr := (m / (m^2 - 9)) / (1 + (3 / (m - 3)))
  expr = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1902_190290


namespace NUMINAMATH_GPT_stratified_sampling_medium_stores_l1902_190291

noncomputable def total_stores := 300
noncomputable def large_stores := 30
noncomputable def medium_stores := 75
noncomputable def small_stores := 195
noncomputable def sample_size := 20

theorem stratified_sampling_medium_stores : 
  (medium_stores : ℕ) * (sample_size : ℕ) / (total_stores : ℕ) = 5 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_medium_stores_l1902_190291


namespace NUMINAMATH_GPT_judy_pencil_cost_l1902_190271

theorem judy_pencil_cost 
  (pencils_per_week : ℕ)
  (days_per_week : ℕ)
  (pack_cost : ℕ)
  (pack_size : ℕ)
  (total_days : ℕ)
  (pencil_usage : pencils_per_week = 10)
  (school_days : days_per_week = 5)
  (cost_per_pack : pack_cost = 4)
  (pencils_per_pack : pack_size = 30)
  (duration : total_days = 45) : 
  ∃ (total_cost : ℕ), total_cost = 12 :=
sorry

end NUMINAMATH_GPT_judy_pencil_cost_l1902_190271


namespace NUMINAMATH_GPT_num_first_and_second_year_students_total_l1902_190267

-- Definitions based on conditions
def num_sampled_students : ℕ := 55
def num_first_year_students_sampled : ℕ := 10
def num_second_year_students_sampled : ℕ := 25
def num_third_year_students_total : ℕ := 400

-- Given that 20 students from the third year are sampled
def num_third_year_students_sampled := num_sampled_students - num_first_year_students_sampled - num_second_year_students_sampled

-- Proportion equality condition
theorem num_first_and_second_year_students_total (x : ℕ) :
  20 / 55 = 400 / (x + num_third_year_students_total) →
  x = 700 :=
by
  sorry

end NUMINAMATH_GPT_num_first_and_second_year_students_total_l1902_190267


namespace NUMINAMATH_GPT_total_puppies_adopted_l1902_190269

-- Define the number of puppies adopted each week
def first_week_puppies : ℕ := 20
def second_week_puppies : ℕ := (2 / 5) * first_week_puppies
def third_week_puppies : ℕ := 2 * second_week_puppies
def fourth_week_puppies : ℕ := 10 + first_week_puppies

-- Prove that the total number of puppies adopted over the month is 74
theorem total_puppies_adopted : 
  first_week_puppies + second_week_puppies + third_week_puppies + fourth_week_puppies = 74 := by
  sorry

end NUMINAMATH_GPT_total_puppies_adopted_l1902_190269


namespace NUMINAMATH_GPT_largest_possible_c_l1902_190281

theorem largest_possible_c (c : ℝ) (hc : (3 * c + 4) * (c - 2) = 9 * c) : c ≤ 4 :=
sorry

end NUMINAMATH_GPT_largest_possible_c_l1902_190281


namespace NUMINAMATH_GPT_rhombic_dodecahedron_surface_area_rhombic_dodecahedron_volume_l1902_190214

noncomputable def surface_area_rhombic_dodecahedron (a : ℝ) : ℝ :=
  6 * (a ^ 2) * Real.sqrt 2

noncomputable def volume_rhombic_dodecahedron (a : ℝ) : ℝ :=
  2 * (a ^ 3)

theorem rhombic_dodecahedron_surface_area (a : ℝ) :
  surface_area_rhombic_dodecahedron a = 6 * (a ^ 2) * Real.sqrt 2 :=
by
  sorry

theorem rhombic_dodecahedron_volume (a : ℝ) :
  volume_rhombic_dodecahedron a = 2 * (a ^ 3) :=
by
  sorry

end NUMINAMATH_GPT_rhombic_dodecahedron_surface_area_rhombic_dodecahedron_volume_l1902_190214


namespace NUMINAMATH_GPT_odd_number_representation_l1902_190256

theorem odd_number_representation (n : ℤ) : 
  (∃ m : ℤ, 2 * m + 1 = 2 * n + 3) ∧ (¬ ∃ m : ℤ, 2 * m + 1 = 4 * n - 1) :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_odd_number_representation_l1902_190256


namespace NUMINAMATH_GPT_orthogonal_trajectory_eqn_l1902_190299

theorem orthogonal_trajectory_eqn (a C : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 2 * a * x) → 
  (∃ C : ℝ, ∀ x y : ℝ, x^2 + y^2 = C * y) :=
sorry

end NUMINAMATH_GPT_orthogonal_trajectory_eqn_l1902_190299


namespace NUMINAMATH_GPT_trigonometric_identity_l1902_190217

variable (α β : Real) 

theorem trigonometric_identity (h₁ : Real.tan (α + β) = 1) 
                              (h₂ : Real.tan (α - β) = 2) 
                              : (Real.sin (2 * α)) / (Real.cos (2 * β)) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1902_190217


namespace NUMINAMATH_GPT_remaining_three_digit_numbers_l1902_190261

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_invalid_number (n : ℕ) : Prop :=
  ∃ (A B : ℕ), A ≠ B ∧ B ≠ 0 ∧ n = 100 * A + 10 * B + A

def count_valid_three_digit_numbers : ℕ :=
  let total_numbers := 900
  let invalid_numbers := 10 * 9
  total_numbers - invalid_numbers

theorem remaining_three_digit_numbers : count_valid_three_digit_numbers = 810 := by
  sorry

end NUMINAMATH_GPT_remaining_three_digit_numbers_l1902_190261


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1902_190209

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (h_arith : is_arithmetic_sequence a)
  (h_condition : a 2 + a 6 = 37) : 
  a 1 + a 3 + a 5 + a 7 = 74 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1902_190209


namespace NUMINAMATH_GPT_total_distance_l1902_190229

/--
A man completes a journey in 30 hours. He travels the first half of the journey at the rate of 20 km/hr and 
the second half at the rate of 10 km/hr. Prove that the total journey is 400 km.
-/
theorem total_distance (D : ℝ) (h : D / 40 + D / 20 = 30) :
  D = 400 :=
sorry

end NUMINAMATH_GPT_total_distance_l1902_190229


namespace NUMINAMATH_GPT_min_rain_fourth_day_l1902_190263

def rain_overflow_problem : Prop :=
    let holding_capacity := 6 * 12 -- in inches
    let drainage_per_day := 3 -- in inches
    let rainfall_day1 := 10 -- in inches
    let rainfall_day2 := 2 * rainfall_day1 -- 20 inches
    let rainfall_day3 := 1.5 * rainfall_day2 -- 30 inches
    let total_rain_three_days := rainfall_day1 + rainfall_day2 + rainfall_day3 -- 60 inches
    let total_drainage_three_days := 3 * drainage_per_day -- 9 inches
    let remaining_capacity := holding_capacity - (total_rain_three_days - total_drainage_three_days) -- 21 inches
    (remaining_capacity = 21)

theorem min_rain_fourth_day : rain_overflow_problem := sorry

end NUMINAMATH_GPT_min_rain_fourth_day_l1902_190263


namespace NUMINAMATH_GPT_jack_age_difference_l1902_190250

def beckett_age : ℕ := 12
def olaf_age : ℕ := beckett_age + 3
def shannen_age : ℕ := olaf_age - 2
def total_age : ℕ := 71
def jack_age : ℕ := total_age - (beckett_age + olaf_age + shannen_age)
def difference := jack_age - 2 * shannen_age

theorem jack_age_difference :
  difference = 5 :=
by
  -- Math proof goes here
  sorry

end NUMINAMATH_GPT_jack_age_difference_l1902_190250


namespace NUMINAMATH_GPT_ordered_pairs_1806_l1902_190297

theorem ordered_pairs_1806 :
  (∃ (xy_list : List (ℕ × ℕ)), xy_list.length = 12 ∧ ∀ (xy : ℕ × ℕ), xy ∈ xy_list → xy.1 * xy.2 = 1806) :=
sorry

end NUMINAMATH_GPT_ordered_pairs_1806_l1902_190297


namespace NUMINAMATH_GPT_ordered_pairs_count_l1902_190279

theorem ordered_pairs_count :
  (∃ (A B : ℕ), 0 < A ∧ 0 < B ∧ A % 2 = 0 ∧ B % 2 = 0 ∧ (A / 8) = (8 / B))
  → (∃ (n : ℕ), n = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_ordered_pairs_count_l1902_190279


namespace NUMINAMATH_GPT_tan_addition_example_l1902_190216

theorem tan_addition_example (x : ℝ) (h : Real.tan x = 1/3) : 
  Real.tan (x + π/3) = 2 + 5 * Real.sqrt 3 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_addition_example_l1902_190216


namespace NUMINAMATH_GPT_domain_g_l1902_190285

noncomputable def g (x : ℝ) := Real.tan (Real.arccos (x ^ 3))

theorem domain_g :
  {x : ℝ | ∃ y, g x = y} = {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)} :=
by
  sorry

end NUMINAMATH_GPT_domain_g_l1902_190285


namespace NUMINAMATH_GPT_similar_triangle_perimeter_l1902_190257

/-
  Given an isosceles triangle with two equal sides of 18 inches and a base of 12 inches, 
  and a similar triangle with the shortest side of 30 inches, 
  prove that the perimeter of the similar triangle is 120 inches.
-/

def is_isosceles (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem similar_triangle_perimeter
  (a b c : ℕ) (a' b' c' : ℕ) (h1 : is_isosceles a b c)
  (h2 : a = 12) (h3 : b = 18) (h4 : c = 18)
  (h5 : a' = 30) (h6 : a' * 18 = a * b')
  (h7 : a' * 18 = a * c') :
  a' + b' + c' = 120 :=
by {
  sorry
}

end NUMINAMATH_GPT_similar_triangle_perimeter_l1902_190257


namespace NUMINAMATH_GPT_sin_pi_over_six_eq_half_l1902_190202

theorem sin_pi_over_six_eq_half : Real.sin (π / 6) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_pi_over_six_eq_half_l1902_190202


namespace NUMINAMATH_GPT_car_travel_distance_l1902_190282

-- Define the original gas mileage as x
variable (x : ℝ) (D : ℝ)

-- Define the conditions
def initial_condition : Prop := D = 12 * x
def revised_condition : Prop := D = 10 * (x + 2)

-- The proof goal
theorem car_travel_distance
  (h1 : initial_condition x D)
  (h2 : revised_condition x D) :
  D = 120 := by
  sorry

end NUMINAMATH_GPT_car_travel_distance_l1902_190282


namespace NUMINAMATH_GPT_g_42_value_l1902_190235

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n : ℕ) (hn : n > 0) : g (n + 1) > g n
axiom g_multiplicative (m n : ℕ) (hm : m > 0) (hn : n > 0) : g (m * n) = g m * g n
axiom g_property_iii (m n : ℕ) (hm : m > 0) (hn : n > 0) : (m ≠ n ∧ m^n = n^m) → (g m = n ∨ g n = m)

theorem g_42_value : g 42 = 4410 :=
by
  sorry

end NUMINAMATH_GPT_g_42_value_l1902_190235


namespace NUMINAMATH_GPT_arithmetic_sequence_a8_l1902_190296

def sum_arithmetic_sequence_first_n_terms (a d : ℕ) (n : ℕ): ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_a8 
  (a d : ℕ) 
  (h : sum_arithmetic_sequence_first_n_terms a d 15 = 45) : 
  a + 7 * d = 3 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a8_l1902_190296


namespace NUMINAMATH_GPT_dogwood_trees_after_work_l1902_190240

theorem dogwood_trees_after_work 
  (trees_part1 : ℝ) (trees_part2 : ℝ) (trees_part3 : ℝ)
  (trees_cut : ℝ) (trees_planted : ℝ)
  (h1 : trees_part1 = 5.0) (h2 : trees_part2 = 4.0) (h3 : trees_part3 = 6.0)
  (h_cut : trees_cut = 7.0) (h_planted : trees_planted = 3.0) :
  trees_part1 + trees_part2 + trees_part3 - trees_cut + trees_planted = 11.0 :=
by
  sorry

end NUMINAMATH_GPT_dogwood_trees_after_work_l1902_190240


namespace NUMINAMATH_GPT_solve_for_w_l1902_190255

theorem solve_for_w (w : ℝ) : (2 : ℝ)^(2 * w) = (8 : ℝ)^(w - 4) → w = 12 := by
  sorry

end NUMINAMATH_GPT_solve_for_w_l1902_190255


namespace NUMINAMATH_GPT_servings_in_one_week_l1902_190230

theorem servings_in_one_week (daily_servings : ℕ) (days_in_week : ℕ) (total_servings : ℕ)
  (h1 : daily_servings = 3)
  (h2 : days_in_week = 7)
  (h3 : total_servings = daily_servings * days_in_week) :
  total_servings = 21 := by
  sorry

end NUMINAMATH_GPT_servings_in_one_week_l1902_190230


namespace NUMINAMATH_GPT_probability_check_l1902_190284

def total_students : ℕ := 12

def total_clubs : ℕ := 3

def equiprobable_clubs := ∀ s : Fin total_students, ∃ c : Fin total_clubs, true

noncomputable def probability_diff_students : ℝ := 1 - (34650 / (total_clubs ^ total_students))

theorem probability_check :
  equiprobable_clubs →
  probability_diff_students = 0.935 := 
by
  intros
  sorry

end NUMINAMATH_GPT_probability_check_l1902_190284


namespace NUMINAMATH_GPT_average_retail_price_l1902_190283

theorem average_retail_price 
  (products : Fin 20 → ℝ)
  (h1 : ∀ i, 400 ≤ products i) 
  (h2 : ∃ s : Finset (Fin 20), s.card = 10 ∧ ∀ i ∈ s, products i < 1000)
  (h3 : ∃ i, products i = 11000): 
  (Finset.univ.sum products) / 20 = 1200 := 
by
  sorry

end NUMINAMATH_GPT_average_retail_price_l1902_190283


namespace NUMINAMATH_GPT_total_amount_invested_l1902_190245

theorem total_amount_invested (x y total : ℝ) (h1 : 0.10 * x - 0.08 * y = 83) (h2 : y = 650) : total = 2000 :=
sorry

end NUMINAMATH_GPT_total_amount_invested_l1902_190245


namespace NUMINAMATH_GPT_evaluate_abs_expression_l1902_190206

noncomputable def approx_pi : ℝ := 3.14159 -- Defining the approximate value of pi

theorem evaluate_abs_expression : |5 * approx_pi - 16| = 0.29205 :=
by
  sorry -- Proof is skipped, as per instructions

end NUMINAMATH_GPT_evaluate_abs_expression_l1902_190206


namespace NUMINAMATH_GPT_exists_non_deg_triangle_in_sets_l1902_190223

-- Definitions used directly from conditions in a)
def non_deg_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem statement
theorem exists_non_deg_triangle_in_sets (S : Fin 100 → Set ℕ) (h_disjoint : ∀ i j : Fin 100, i ≠ j → Disjoint (S i) (S j))
  (h_union : (⋃ i, S i) = {x | 1 ≤ x ∧ x ≤ 400}) :
  ∃ i : Fin 100, ∃ a b c : ℕ, a ∈ S i ∧ b ∈ S i ∧ c ∈ S i ∧ non_deg_triangle a b c := sorry

end NUMINAMATH_GPT_exists_non_deg_triangle_in_sets_l1902_190223


namespace NUMINAMATH_GPT_no_real_roots_of_ffx_eq_ninex_l1902_190213

variable (a : ℝ)
noncomputable def f (x : ℝ) : ℝ :=
  x^2 * Real.log (4*(a+1)/a) / Real.log 2 +
  2 * x * Real.log (2 * a / (a + 1)) / Real.log 2 +
  Real.log ((a + 1)^2 / (4 * a^2)) / Real.log 2

theorem no_real_roots_of_ffx_eq_ninex (a : ℝ) (h_pos : ∀ x, 1 ≤ x → f a x > 0) :
  ¬ ∃ x, 1 ≤ x ∧ f a (f a x) = 9 * x :=
  sorry

end NUMINAMATH_GPT_no_real_roots_of_ffx_eq_ninex_l1902_190213


namespace NUMINAMATH_GPT_kimberly_store_visits_l1902_190251

def peanuts_per_visit : ℕ := 7
def total_peanuts : ℕ := 21

def visits : ℕ := total_peanuts / peanuts_per_visit

theorem kimberly_store_visits : visits = 3 :=
by
  sorry

end NUMINAMATH_GPT_kimberly_store_visits_l1902_190251


namespace NUMINAMATH_GPT_transform_cos_function_l1902_190273

theorem transform_cos_function :
  ∀ x : ℝ, 2 * Real.cos (x + π / 3) =
           2 * Real.cos (2 * (x - π / 12) + π / 6) := 
sorry

end NUMINAMATH_GPT_transform_cos_function_l1902_190273


namespace NUMINAMATH_GPT_sum_of_altitudes_l1902_190252

theorem sum_of_altitudes (x y : ℝ) (h : 12 * x + 5 * y = 60) :
  let a := (if y = 0 then x else 0)
  let b := (if x = 0 then y else 0)
  let c := (60 / (Real.sqrt (12^2 + 5^2)))
  a + b + c = 281 / 13 :=
sorry

end NUMINAMATH_GPT_sum_of_altitudes_l1902_190252


namespace NUMINAMATH_GPT_simplify_expression_l1902_190276

theorem simplify_expression (x : ℝ) : 
  (3 * x - 4) * (x + 8) - (x + 6) * (3 * x - 2) = 4 * x - 20 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1902_190276


namespace NUMINAMATH_GPT_neg_one_third_squared_l1902_190287

theorem neg_one_third_squared :
  (-(1/3))^2 = 1/9 :=
sorry

end NUMINAMATH_GPT_neg_one_third_squared_l1902_190287


namespace NUMINAMATH_GPT_transformed_sum_of_coordinates_l1902_190254

theorem transformed_sum_of_coordinates (g : ℝ → ℝ) (h : g 8 = 5) :
  let x := 8 / 3
  let y := 14 / 9
  3 * y = g (3 * x) / 3 + 3 ∧ (x + y = 38 / 9) :=
by
  sorry

end NUMINAMATH_GPT_transformed_sum_of_coordinates_l1902_190254


namespace NUMINAMATH_GPT_age_difference_is_40_l1902_190211

-- Define the ages of the daughter and the mother
variables (D M : ℕ)

-- Conditions
-- 1. The mother's age is the digits of the daughter's age reversed
def mother_age_is_reversed_daughter_age : Prop :=
  M = 10 * D + D

-- 2. In thirteen years, the mother will be twice as old as the daughter
def mother_twice_as_old_in_thirteen_years : Prop :=
  M + 13 = 2 * (D + 13)

-- The theorem: The difference in their current ages is 40
theorem age_difference_is_40
  (h1 : mother_age_is_reversed_daughter_age D M)
  (h2 : mother_twice_as_old_in_thirteen_years D M) :
  M - D = 40 :=
sorry

end NUMINAMATH_GPT_age_difference_is_40_l1902_190211


namespace NUMINAMATH_GPT_find_m_l1902_190220

theorem find_m (a0 a1 a2 a3 a4 a5 a6 : ℝ) (m : ℝ)
  (h1 : (1 + m) * x ^ 6 = a0 + a1 * x + a2 * x ^ 2 + a3 * x ^ 3 + a4 * x ^ 4 + a5 * x ^ 5 + a6 * x ^ 6)
  (h2 : a1 - a2 + a3 - a4 + a5 - a6 = -63)
  (h3 : a0 = 1) :
  m = 3 ∨ m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1902_190220


namespace NUMINAMATH_GPT_find_x_minus_y_l1902_190233

theorem find_x_minus_y (x y z : ℤ) (h₁ : x - y - z = 7) (h₂ : x - y + z = 15) : x - y = 11 := by
  sorry

end NUMINAMATH_GPT_find_x_minus_y_l1902_190233


namespace NUMINAMATH_GPT_determine_digit_z_l1902_190268

noncomputable def ends_with_k_digits (n : ℕ) (d :ℕ) (k : ℕ) : Prop :=
  ∃ m, m ≥ 1 ∧ (10^k * m + d = n % 10^(k + 1))

noncomputable def decimal_ends_with_digits (z k n : ℕ) : Prop :=
  ends_with_k_digits (n^9) z k

theorem determine_digit_z :
  (z = 9) ↔ ∀ k ≥ 1, ∃ n ≥ 1, decimal_ends_with_digits z k n :=
by
  sorry

end NUMINAMATH_GPT_determine_digit_z_l1902_190268


namespace NUMINAMATH_GPT_part1_part2_l1902_190258

-- Problem statement (1)
theorem part1 (a : ℝ) (h : a = -3) :
  (∀ x : ℝ, (x^2 + a * x + 2) ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2) →
  { x : ℝ // (x^2 + a * x + 2) ≥ 1 - x^2 } = { x : ℝ // x ≤ 1 / 2 ∨ x ≥ 1 } :=
sorry

-- Problem statement (2)
theorem part2 (a : ℝ) :
  (∀ x : ℝ, (x^2 + a * x + 2) + x^2 + 1 = 2 * x^2 + a * x + 3) →
  (∃ x : ℝ, 1 < x ∧ x < 2 ∧ (2 * x^2 + a * x + 3) = 0) →
  -5 < a ∧ a < -2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1902_190258


namespace NUMINAMATH_GPT_value_of_x_minus_y_l1902_190226

theorem value_of_x_minus_y (x y a : ℝ) (h₁ : x + y > 0) (h₂ : a < 0) (h₃ : a * y > 0) : x - y > 0 :=
sorry

end NUMINAMATH_GPT_value_of_x_minus_y_l1902_190226


namespace NUMINAMATH_GPT_percentage_defective_l1902_190237

theorem percentage_defective (examined rejected : ℚ) (h1 : examined = 66.67) (h2 : rejected = 10) :
  (rejected / examined) * 100 = 15 := by
  sorry

end NUMINAMATH_GPT_percentage_defective_l1902_190237


namespace NUMINAMATH_GPT_number_of_club_members_l1902_190204

theorem number_of_club_members
  (num_committee : ℕ)
  (pair_of_committees_has_unique_member : ∀ (c1 c2 : Fin num_committee), c1 ≠ c2 → ∃! m : ℕ, c1 ≠ c2 ∧ c2 ≠ c1 ∧ m = m)
  (members_belong_to_two_committees : ∀ m : ℕ, ∃ (c1 c2 : Fin num_committee), c1 ≠ c2 ∧ m = m)
  : num_committee = 5 → ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_club_members_l1902_190204


namespace NUMINAMATH_GPT_sum_roots_x_squared_minus_5x_plus_6_eq_5_l1902_190234

noncomputable def sum_of_roots (a b c : Real) : Real :=
  -b / a

theorem sum_roots_x_squared_minus_5x_plus_6_eq_5 :
  sum_of_roots 1 (-5) 6 = 5 := by
  sorry

end NUMINAMATH_GPT_sum_roots_x_squared_minus_5x_plus_6_eq_5_l1902_190234


namespace NUMINAMATH_GPT_interior_angle_sum_of_regular_polygon_l1902_190265

theorem interior_angle_sum_of_regular_polygon (h: ∀ θ, θ = 45) :
  ∃ s, s = 1080 := by
  sorry

end NUMINAMATH_GPT_interior_angle_sum_of_regular_polygon_l1902_190265


namespace NUMINAMATH_GPT_certain_amount_l1902_190238

theorem certain_amount (x : ℝ) (A : ℝ) (h1: x = 900) (h2: 0.25 * x = 0.15 * 1600 - A) : A = 15 :=
by
  sorry

end NUMINAMATH_GPT_certain_amount_l1902_190238


namespace NUMINAMATH_GPT_geometric_sequence_a4_value_l1902_190270

variable {α : Type} [LinearOrderedField α]

noncomputable def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m : ℕ, n < m → ∃ r : α, 0 < r ∧ a m = a n * r^(m - n)

theorem geometric_sequence_a4_value (a : ℕ → α)
  (pos : ∀ n, 0 < a n)
  (geo_seq : is_geometric_sequence a)
  (h : a 1 * a 7 = 36) :
  a 4 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_value_l1902_190270


namespace NUMINAMATH_GPT_distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l1902_190274

-- Definitions based on conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def odd_digits : Finset ℕ := {1, 3, 5}

-- Problem 1: Number of distinct three-digit numbers
theorem distinct_three_digit_numbers : (digits.erase 0).card * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 100 := by
  sorry

-- Problem 2: Number of distinct three-digit odd numbers
theorem distinct_three_digit_odd_numbers : (odd_digits.card) * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 48 := by
  sorry

end NUMINAMATH_GPT_distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l1902_190274


namespace NUMINAMATH_GPT_circle_center_sum_l1902_190225

theorem circle_center_sum (x y : ℝ) (hx : (x, y) = (3, -4)) :
  (x + y) = -1 :=
by {
  -- We are given that the center of the circle is (3, -4)
  sorry -- Proof is omitted
}

end NUMINAMATH_GPT_circle_center_sum_l1902_190225


namespace NUMINAMATH_GPT_train_crosses_pole_in_15_seconds_l1902_190293

theorem train_crosses_pole_in_15_seconds
    (train_speed : ℝ) (train_length_meters : ℝ) (time_seconds : ℝ) : 
    train_speed = 300 →
    train_length_meters = 1250 →
    time_seconds = 15 :=
by
  sorry

end NUMINAMATH_GPT_train_crosses_pole_in_15_seconds_l1902_190293


namespace NUMINAMATH_GPT_line_segment_endpoint_l1902_190221

theorem line_segment_endpoint (x : ℝ) (h1 : (x - 3)^2 + 36 = 289) (h2 : x < 0) : x = 3 - Real.sqrt 253 :=
sorry

end NUMINAMATH_GPT_line_segment_endpoint_l1902_190221


namespace NUMINAMATH_GPT_value_of_expression_eq_34_l1902_190232

theorem value_of_expression_eq_34 : (2 - 6 + 10 - 14 + 18 - 22 + 26 - 30 + 34 - 38 + 42 - 46 + 50 - 54 + 58 - 62 + 66 - 70 + 70) = 34 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_eq_34_l1902_190232


namespace NUMINAMATH_GPT_find_number_l1902_190286

theorem find_number (x : ℤ) (h : 2 * x + 5 = 17) : x = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1902_190286


namespace NUMINAMATH_GPT_ratio_of_boat_to_stream_l1902_190298

theorem ratio_of_boat_to_stream (B S : ℝ) (h : ∀ D : ℝ, D / (B - S) = 2 * (D / (B + S))) :
  B / S = 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_boat_to_stream_l1902_190298


namespace NUMINAMATH_GPT_reconstruct_points_l1902_190289

noncomputable def symmetric (x y : ℝ) := 2 * y - x

theorem reconstruct_points (A' B' C' D' B C D : ℝ) :
  (∃ (A B C D : ℝ),
     B = (A + A') / 2 ∧  -- B is the midpoint of line segment AA'
     C = (B + B') / 2 ∧  -- C is the midpoint of line segment BB'
     D = (C + C') / 2 ∧  -- D is the midpoint of line segment CC'
     A = (D + D') / 2)   -- A is the midpoint of line segment DD'
  ↔ (∃ (A : ℝ), A = symmetric D D') → True := sorry

end NUMINAMATH_GPT_reconstruct_points_l1902_190289


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l1902_190248

-- Define the conditions for each problem explicitly
def cond1 : Prop := ∃ (A B C : Type), -- "A" can only be in the middle or on the sides (positions are constrainted)
  True -- (specific arrangements are abstracted here)

def cond2 : Prop := ∃ (A B C : Type), -- male students must be grouped together
  True

def cond3 : Prop := ∃ (A B C : Type), -- male students cannot be grouped together
  True

def cond4 : Prop := ∃ (A B C : Type), -- the order of "A", "B", "C" from left to right remains unchanged
  True

def cond5 : Prop := ∃ (A B C : Type), -- "A" is not on the far left and "B" is not on the far right
  True

def cond6 : Prop := ∃ (A B C D : Type), -- One more female student, males and females are not next to each other
  True

def cond7 : Prop := ∃ (A B C : Type), -- arranged in two rows, with 3 people in the front row and 2 in the back row
  True

def cond8 : Prop := ∃ (A B C : Type), -- there must be 1 person between "A" and "B"
  True

-- Prove each condition results in the specified number of arrangements

theorem problem1 : cond1 → True := by
  -- Problem (1) is to show 72 arrangements given conditions
  sorry

theorem problem2 : cond2 → True := by
  -- Problem (2) is to show 36 arrangements given conditions
  sorry

theorem problem3 : cond3 → True := by
  -- Problem (3) is to show 12 arrangements given conditions
  sorry

theorem problem4 : cond4 → True := by
  -- Problem (4) is to show 20 arrangements given conditions
  sorry

theorem problem5 : cond5 → True := by
  -- Problem (5) is to show 78 arrangements given conditions
  sorry

theorem problem6 : cond6 → True := by
  -- Problem (6) is to show 144 arrangements given conditions
  sorry

theorem problem7 : cond7 → True := by
  -- Problem (7) is to show 120 arrangements given conditions
  sorry

theorem problem8 : cond8 → True := by
  -- Problem (8) is to show 36 arrangements given conditions
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l1902_190248


namespace NUMINAMATH_GPT_range_of_m_l1902_190292

variable {x m : ℝ}

def absolute_value_inequality (x m : ℝ) : Prop := |x + 1| - |x - 2| > m

theorem range_of_m : (∀ x : ℝ, absolute_value_inequality x m) ↔ m < -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1902_190292


namespace NUMINAMATH_GPT_boat_speed_in_still_water_eq_16_l1902_190212

theorem boat_speed_in_still_water_eq_16 (stream_rate : ℝ) (time_downstream : ℝ) (distance_downstream : ℝ) (V_b : ℝ) 
(h1 : stream_rate = 5) (h2 : time_downstream = 6) (h3 : distance_downstream = 126) : 
  V_b = 16 :=
by sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_eq_16_l1902_190212


namespace NUMINAMATH_GPT_not_difference_of_squares_10_l1902_190205

theorem not_difference_of_squares_10 (a b : ℤ) : a^2 - b^2 ≠ 10 :=
sorry

end NUMINAMATH_GPT_not_difference_of_squares_10_l1902_190205


namespace NUMINAMATH_GPT_cannot_contain_2003_0_l1902_190207

noncomputable def point_not_on_line (m b : ℝ) (h : m * b < 0) : Prop :=
  ∀ y : ℝ, ¬(0 = 2003 * m + b)

-- Prove that if m and b are real numbers and mb < 0, the line y = mx + b
-- cannot contain the point (2003, 0).
theorem cannot_contain_2003_0 (m b : ℝ) (h : m * b < 0) : point_not_on_line m b h :=
by
  sorry

end NUMINAMATH_GPT_cannot_contain_2003_0_l1902_190207


namespace NUMINAMATH_GPT_prism_ratio_l1902_190277

theorem prism_ratio (a b c d : ℝ) (h_d : d = 60) (h_c : c = 104) (h_b : b = 78 * Real.pi) (h_a : a = (4 * Real.pi) / 3) :
  b * c / (a * d) = 8112 / 240 := 
by 
  sorry

end NUMINAMATH_GPT_prism_ratio_l1902_190277


namespace NUMINAMATH_GPT_correct_value_calculation_l1902_190249

theorem correct_value_calculation (x : ℤ) (h : 2 * (x + 6) = 28) : 6 * x = 48 :=
by
  -- Proof steps would be here
  sorry

end NUMINAMATH_GPT_correct_value_calculation_l1902_190249


namespace NUMINAMATH_GPT_vertex_difference_l1902_190242

theorem vertex_difference (n m : ℝ) : 
  ∀ x : ℝ, (∀ x, -x^2 + 2*x + n = -((x - m)^2) + 1) → m - n = 1 := 
by 
  sorry

end NUMINAMATH_GPT_vertex_difference_l1902_190242


namespace NUMINAMATH_GPT_trigonometric_identity_l1902_190288

open Real

theorem trigonometric_identity (α β : ℝ) (h : 2 * cos (2 * α + β) - 3 * cos β = 0) :
  tan α * tan (α + β) = -1 / 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_trigonometric_identity_l1902_190288


namespace NUMINAMATH_GPT_smallest_part_proportional_l1902_190294

/-- If we divide 124 into three parts proportional to 2, 1/2, and 1/4,
    prove that the smallest part is 124 / 11. -/
theorem smallest_part_proportional (x : ℝ) 
  (h : 2 * x + (1 / 2) * x + (1 / 4) * x = 124) : 
  (1 / 4) * x = 124 / 11 :=
sorry

end NUMINAMATH_GPT_smallest_part_proportional_l1902_190294


namespace NUMINAMATH_GPT_factor_quadratic_l1902_190236

theorem factor_quadratic (y : ℝ) : 9 * y ^ 2 - 30 * y + 25 = (3 * y - 5) ^ 2 := by
  sorry

end NUMINAMATH_GPT_factor_quadratic_l1902_190236


namespace NUMINAMATH_GPT_garden_bed_length_l1902_190262

theorem garden_bed_length (total_area : ℕ) (garden_area : ℕ) (width : ℕ) (n : ℕ)
  (total_area_eq : total_area = 42)
  (garden_area_eq : garden_area = 9)
  (num_gardens_eq : n = 2)
  (width_eq : width = 3)
  (lhs_eq : lhs = total_area - n * garden_area)
  (area_to_length_eq : length = lhs / width) :
  length = 8 := by
  sorry

end NUMINAMATH_GPT_garden_bed_length_l1902_190262


namespace NUMINAMATH_GPT_probability_of_4_vertices_in_plane_l1902_190218

-- Definition of the problem conditions
def vertices_of_cube : Nat := 8
def selecting_vertices : Nat := 4

-- Combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Total ways to select 4 vertices from the 8 vertices of a cube
def total_ways : Nat := combination vertices_of_cube selecting_vertices

-- Number of favorable ways that these 4 vertices lie in the same plane
def favorable_ways : Nat := 12

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

-- The ultimate proof problem
theorem probability_of_4_vertices_in_plane :
  probability = 6 / 35 :=
by
  -- Here, the proof steps would go to verify that our setup correctly leads to the given probability.
  sorry

end NUMINAMATH_GPT_probability_of_4_vertices_in_plane_l1902_190218


namespace NUMINAMATH_GPT_lock_rings_l1902_190280

theorem lock_rings (n : ℕ) (h : 6 ^ n - 1 ≤ 215) : n = 3 :=
sorry

end NUMINAMATH_GPT_lock_rings_l1902_190280


namespace NUMINAMATH_GPT_sugar_solution_l1902_190219

theorem sugar_solution (V x : ℝ) (h1 : V > 0) (h2 : 0.1 * (V - x) + 0.5 * x = 0.2 * V) : x / V = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_sugar_solution_l1902_190219


namespace NUMINAMATH_GPT_year_2013_is_not_lucky_l1902_190246

-- Definitions based on conditions
def last_two_digits (year : ℕ) : ℕ := year % 100

def is_valid_date (month : ℕ) (day : ℕ) (year : ℕ) : Prop :=
  month * day = last_two_digits year

def is_lucky_year (year : ℕ) : Prop :=
  ∃ (month : ℕ) (day : ℕ), month <= 12 ∧ day <= 12 ∧ is_valid_date month day year

-- The main statement to prove
theorem year_2013_is_not_lucky : ¬ is_lucky_year 2013 :=
by {
  sorry
}

end NUMINAMATH_GPT_year_2013_is_not_lucky_l1902_190246


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1902_190208

theorem sufficient_but_not_necessary_condition (x y : ℝ) :
  (x = 1 ∧ y = 1 → x + y = 2) ∧ (¬(x + y = 2 → x = 1 ∧ y = 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1902_190208
