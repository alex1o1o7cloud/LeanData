import Mathlib

namespace NUMINAMATH_GPT_systematic_sampling_first_two_numbers_l1340_134072

theorem systematic_sampling_first_two_numbers
  (sample_size : ℕ) (population_size : ℕ) (last_sample_number : ℕ)
  (h1 : sample_size = 50) (h2 : population_size = 8000) (h3 : last_sample_number = 7900) :
  ∃ first second : ℕ, first = 60 ∧ second = 220 :=
by
  -- Proof to be provided.
  sorry

end NUMINAMATH_GPT_systematic_sampling_first_two_numbers_l1340_134072


namespace NUMINAMATH_GPT_positive_integer_x_l1340_134081

theorem positive_integer_x (x : ℕ) (hx : 15 * x = x^2 + 56) : x = 8 := by
  sorry

end NUMINAMATH_GPT_positive_integer_x_l1340_134081


namespace NUMINAMATH_GPT_area_of_transformed_region_l1340_134045

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 0], ![5, 3]]
def area_T : ℝ := 9

-- Theorem statement
theorem area_of_transformed_region : 
  let det_matrix := matrix.det
  (det_matrix = 9) → (area_T = 9) → (area_T * det_matrix = 81) :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_area_of_transformed_region_l1340_134045


namespace NUMINAMATH_GPT_alex_friends_invite_l1340_134009

theorem alex_friends_invite (burger_buns_per_pack : ℕ)
                            (packs_of_buns : ℕ)
                            (buns_needed_by_each_guest : ℕ)
                            (total_buns : ℕ)
                            (friends_who_dont_eat_buns : ℕ)
                            (friends_who_dont_eat_meat : ℕ)
                            (total_friends_invited : ℕ) 
                            (h1 : burger_buns_per_pack = 8)
                            (h2 : packs_of_buns = 3)
                            (h3 : buns_needed_by_each_guest = 3)
                            (h4 : total_buns = packs_of_buns * burger_buns_per_pack)
                            (h5 : friends_who_dont_eat_buns = 1)
                            (h6 : friends_who_dont_eat_meat = 1)
                            (h7 : total_friends_invited = (total_buns / buns_needed_by_each_guest) + friends_who_dont_eat_buns) :
  total_friends_invited = 9 :=
by sorry

end NUMINAMATH_GPT_alex_friends_invite_l1340_134009


namespace NUMINAMATH_GPT_complement_of_intersection_l1340_134027

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end NUMINAMATH_GPT_complement_of_intersection_l1340_134027


namespace NUMINAMATH_GPT_find_w_l1340_134054

theorem find_w (p q r u v w : ℝ)
  (h₁ : (x : ℝ) → x^3 - 6 * x^2 + 11 * x + 10 = (x - p) * (x - q) * (x - r))
  (h₂ : (x : ℝ) → x^3 + u * x^2 + v * x + w = (x - (p + q)) * (x - (q + r)) * (x - (r + p)))
  (h₃ : p + q + r = 6) :
  w = 80 := sorry

end NUMINAMATH_GPT_find_w_l1340_134054


namespace NUMINAMATH_GPT_koala_fiber_intake_l1340_134015

theorem koala_fiber_intake (x : ℝ) (h : 0.40 * x = 16) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_koala_fiber_intake_l1340_134015


namespace NUMINAMATH_GPT_smallest_number_of_students_l1340_134003

theorem smallest_number_of_students
  (n : ℕ)
  (h1 : 3 * 90 + (n - 3) * 65 ≤ n * 80)
  (h2 : ∀ k, k ≤ n - 3 → 65 ≤ k)
  (h3 : (3 * 90) + ((n - 3) * 65) / n = 80) : n = 5 :=
sorry

end NUMINAMATH_GPT_smallest_number_of_students_l1340_134003


namespace NUMINAMATH_GPT_real_solutions_x_inequality_l1340_134032

theorem real_solutions_x_inequality (x : ℝ) :
  (∃ y : ℝ, y^2 + 6 * x * y + x + 8 = 0) ↔ (x ≤ -8 / 9 ∨ x ≥ 1) := 
sorry

end NUMINAMATH_GPT_real_solutions_x_inequality_l1340_134032


namespace NUMINAMATH_GPT_powers_of_2_not_powers_of_4_below_1000000_equals_10_l1340_134031

def num_powers_of_2_not_4 (n : ℕ) : ℕ :=
  let powers_of_2 := (List.range n).filter (fun k => (2^k < 1000000));
  let powers_of_4 := (List.range n).filter (fun k => (4^k < 1000000));
  powers_of_2.length - powers_of_4.length

theorem powers_of_2_not_powers_of_4_below_1000000_equals_10 : 
  num_powers_of_2_not_4 20 = 10 :=
by
  sorry

end NUMINAMATH_GPT_powers_of_2_not_powers_of_4_below_1000000_equals_10_l1340_134031


namespace NUMINAMATH_GPT_simplify_expression_correct_l1340_134048

def simplify_expression (x : ℝ) : Prop :=
  (5 - 2 * x) - (7 + 3 * x) = -2 - 5 * x

theorem simplify_expression_correct (x : ℝ) : simplify_expression x :=
  by
    sorry

end NUMINAMATH_GPT_simplify_expression_correct_l1340_134048


namespace NUMINAMATH_GPT_abs_ineq_l1340_134053

open Real

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

theorem abs_ineq (a b c : ℝ) (h1 : a + b ≥ 0) (h2 : b + c ≥ 0) (h3 : c + a ≥ 0) :
  a + b + c ≥ (absolute_value a + absolute_value b + absolute_value c) / 3 := by
  sorry

end NUMINAMATH_GPT_abs_ineq_l1340_134053


namespace NUMINAMATH_GPT_age_difference_l1340_134008

-- Define the hypothesis and statement
theorem age_difference (A B C : ℕ) 
  (h1 : A + B = B + C + 15)
  (h2 : C = A - 15) : 
  (A + B) - (B + C) = 15 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1340_134008


namespace NUMINAMATH_GPT_engineer_thought_of_l1340_134062

def isProperDivisor (n k : ℕ) : Prop :=
  k ≠ 1 ∧ k ≠ n ∧ k ∣ n

def transformDivisors (n m : ℕ) : Prop :=
  ∀ k, isProperDivisor n k → isProperDivisor m (k + 1)

theorem engineer_thought_of (n : ℕ) :
  (∀ m : ℕ, n = 2^2 ∨ n = 2^3 → transformDivisors n m → (m % 2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_engineer_thought_of_l1340_134062


namespace NUMINAMATH_GPT_parallelogram_area_l1340_134035

theorem parallelogram_area (base height : ℕ) (h_base : base = 36) (h_height : height = 24) : base * height = 864 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1340_134035


namespace NUMINAMATH_GPT_lcm_8_13_14_is_728_l1340_134014

-- Define the numbers and their factorizations
def num1 := 8
def fact1 := 2 ^ 3

def num2 := 13  -- 13 is prime

def num3 := 14
def fact3 := 2 * 7

-- Define the function to calculate the LCM of three integers
def lcm (a b c : ℕ) : ℕ :=
  Nat.lcm a (Nat.lcm b c)

-- State the theorem to prove that the LCM of 8, 13, and 14 is 728
theorem lcm_8_13_14_is_728 : lcm num1 num2 num3 = 728 :=
by
  -- Prove the equality, skipping proof details with sorry
  sorry

end NUMINAMATH_GPT_lcm_8_13_14_is_728_l1340_134014


namespace NUMINAMATH_GPT_sally_picked_11_pears_l1340_134050

theorem sally_picked_11_pears (total_pears : ℕ) (pears_picked_by_Sara : ℕ) (pears_picked_by_Sally : ℕ) 
    (h1 : total_pears = 56) (h2 : pears_picked_by_Sara = 45) :
    pears_picked_by_Sally = total_pears - pears_picked_by_Sara := by
  sorry

end NUMINAMATH_GPT_sally_picked_11_pears_l1340_134050


namespace NUMINAMATH_GPT_students_with_all_three_pets_l1340_134097

variable (x y z : ℕ)
variable (total_students : ℕ := 40)
variable (dog_students : ℕ := total_students * 5 / 8)
variable (cat_students : ℕ := total_students * 1 / 4)
variable (other_students : ℕ := 8)
variable (no_pet_students : ℕ := 6)
variable (only_dog_students : ℕ := 12)
variable (only_other_students : ℕ := 3)
variable (cat_other_no_dog_students : ℕ := 10)

theorem students_with_all_three_pets :
  (x + y + z + 10 + 3 + 12 = total_students - no_pet_students) →
  (x + z + 10 = dog_students) →
  (10 + z = cat_students) →
  (y + z + 10 = other_students) →
  z = 0 :=
by
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_students_with_all_three_pets_l1340_134097


namespace NUMINAMATH_GPT_min_ab_value_l1340_134025

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = a + 9 * b + 7) : a * b ≥ 49 :=
sorry

end NUMINAMATH_GPT_min_ab_value_l1340_134025


namespace NUMINAMATH_GPT_minimum_toothpicks_for_5_squares_l1340_134080

theorem minimum_toothpicks_for_5_squares :
  let single_square_toothpicks := 4
  let additional_shared_side_toothpicks := 3
  ∃ n, n = single_square_toothpicks + 4 * additional_shared_side_toothpicks ∧ n = 15 :=
by
  sorry

end NUMINAMATH_GPT_minimum_toothpicks_for_5_squares_l1340_134080


namespace NUMINAMATH_GPT_parabola_translation_l1340_134018

theorem parabola_translation :
  ∀ (x y : ℝ), y = 3 * x^2 →
  ∃ (new_x new_y : ℝ), new_y = 3 * (new_x + 3)^2 - 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_translation_l1340_134018


namespace NUMINAMATH_GPT_find_f_7_l1340_134078

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom function_period : ∀ x : ℝ, f (x + 2) = -f x
axiom function_value_range : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem find_f_7 : f 7 = -1 := by
  sorry

end NUMINAMATH_GPT_find_f_7_l1340_134078


namespace NUMINAMATH_GPT_total_earnings_is_correct_l1340_134033

def lloyd_normal_hours : ℝ := 7.5
def lloyd_rate : ℝ := 4.5
def lloyd_overtime_rate : ℝ := 2.0
def lloyd_hours_worked : ℝ := 10.5

def casey_normal_hours : ℝ := 8
def casey_rate : ℝ := 5
def casey_overtime_rate : ℝ := 1.5
def casey_hours_worked : ℝ := 9.5

def lloyd_earnings : ℝ := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours_worked - lloyd_normal_hours) * lloyd_rate * lloyd_overtime_rate)

def casey_earnings : ℝ := (casey_normal_hours * casey_rate) + ((casey_hours_worked - casey_normal_hours) * casey_rate * casey_overtime_rate)

def total_earnings : ℝ := lloyd_earnings + casey_earnings

theorem total_earnings_is_correct : total_earnings = 112 := by
  sorry

end NUMINAMATH_GPT_total_earnings_is_correct_l1340_134033


namespace NUMINAMATH_GPT_central_angle_of_section_l1340_134042

theorem central_angle_of_section (A : ℝ) (x: ℝ) (H : (1 / 8 : ℝ) = (x / 360)) : x = 45 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_section_l1340_134042


namespace NUMINAMATH_GPT_total_cans_collected_l1340_134012

def bags_on_saturday : ℕ := 6
def bags_on_sunday : ℕ := 3
def cans_per_bag : ℕ := 8
def total_cans : ℕ := 72

theorem total_cans_collected :
  (bags_on_saturday + bags_on_sunday) * cans_per_bag = total_cans :=
by
  sorry

end NUMINAMATH_GPT_total_cans_collected_l1340_134012


namespace NUMINAMATH_GPT_sin_double_angle_l1340_134038

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 4 / 5) : Real.sin (2 * x) = -7 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1340_134038


namespace NUMINAMATH_GPT_probability_of_unique_color_and_number_l1340_134090

-- Defining the sets of colors and numbers
inductive Color
| red
| yellow
| blue

inductive Number
| one
| two
| three

-- Defining a ball as a combination of a Color and a Number
structure Ball :=
(color : Color)
(number : Number)

-- Setting up the list of 9 balls
def allBalls : List Ball :=
  [⟨Color.red, Number.one⟩, ⟨Color.red, Number.two⟩, ⟨Color.red, Number.three⟩,
   ⟨Color.yellow, Number.one⟩, ⟨Color.yellow, Number.two⟩, ⟨Color.yellow, Number.three⟩,
   ⟨Color.blue, Number.one⟩, ⟨Color.blue, Number.two⟩, ⟨Color.blue, Number.three⟩]

-- Proving the probability calculation as a theorem
noncomputable def probability_neither_same_color_nor_number : ℕ → ℕ → ℚ :=
  λ favorable total => favorable / total

theorem probability_of_unique_color_and_number :
  probability_neither_same_color_nor_number
    (6) -- favorable outcomes
    (84) -- total outcomes
  = 1 / 14 := by
  sorry

end NUMINAMATH_GPT_probability_of_unique_color_and_number_l1340_134090


namespace NUMINAMATH_GPT_total_marks_l1340_134041

-- Define the conditions
def average_marks : ℝ := 35
def number_of_candidates : ℕ := 120

-- Define the total marks as a goal to prove
theorem total_marks : number_of_candidates * average_marks = 4200 :=
by
  sorry

end NUMINAMATH_GPT_total_marks_l1340_134041


namespace NUMINAMATH_GPT_dhoni_dishwasher_spending_l1340_134092

noncomputable def percentage_difference : ℝ := 0.25 - 0.225
noncomputable def percentage_less_than : ℝ := (percentage_difference / 0.25) * 100

theorem dhoni_dishwasher_spending :
  (percentage_difference / 0.25) * 100 = 10 :=
by sorry

end NUMINAMATH_GPT_dhoni_dishwasher_spending_l1340_134092


namespace NUMINAMATH_GPT_sum_of_special_integers_l1340_134063

theorem sum_of_special_integers :
  let a := 0
  let b := 1
  let c := -1
  a + b + c = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_special_integers_l1340_134063


namespace NUMINAMATH_GPT_positive_integers_sum_of_squares_l1340_134068

theorem positive_integers_sum_of_squares
  (a b c d : ℤ)
  (h1 : a^2 + b^2 + c^2 + d^2 = 90)
  (h2 : a + b + c + d = 16) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d := 
by
  sorry

end NUMINAMATH_GPT_positive_integers_sum_of_squares_l1340_134068


namespace NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l1340_134016

theorem parallel_lines (t s k : ℝ) :
  (∀ t, ∃ s, (1 - 2 * t = s) ∧ (2 + k * t = 1 - 2 * s)) →
  k = 4 :=
by
  sorry

theorem perpendicular_lines (t s k : ℝ) :
  (∀ t, ∃ s, (1 - 2 * t = s) ∧ (2 + k * t = 1 - 2 * s)) →
  k = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l1340_134016


namespace NUMINAMATH_GPT_find_positive_integers_l1340_134001

theorem find_positive_integers 
    (a b : ℕ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (h1 : ∃ k1 : ℤ, (a^3 * b - 1) = k1 * (a + 1))
    (h2 : ∃ k2 : ℤ, (b^3 * a + 1) = k2 * (b - 1)) : 
    (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
sorry

end NUMINAMATH_GPT_find_positive_integers_l1340_134001


namespace NUMINAMATH_GPT_area_of_L_shape_is_58_l1340_134082

-- Define the dimensions of the large rectangle
def large_rectangle_length : ℕ := 10
def large_rectangle_width : ℕ := 7

-- Define the dimensions of the smaller rectangle to be removed
def small_rectangle_length : ℕ := 4
def small_rectangle_width : ℕ := 3

-- Define the area of the large rectangle
def area_large_rectangle : ℕ := large_rectangle_length * large_rectangle_width

-- Define the area of the small rectangle
def area_small_rectangle : ℕ := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shaped region
def area_L_shape : ℕ := area_large_rectangle - area_small_rectangle

-- Prove that the area of the "L" shaped region is 58 square units
theorem area_of_L_shape_is_58 : area_L_shape = 58 := by
  sorry

end NUMINAMATH_GPT_area_of_L_shape_is_58_l1340_134082


namespace NUMINAMATH_GPT_fraction_multiplication_validity_l1340_134007

theorem fraction_multiplication_validity (a b m x : ℝ) (hb : b ≠ 0) : 
  (x ≠ m) ↔ (b * (x - m) ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_validity_l1340_134007


namespace NUMINAMATH_GPT_domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l1340_134098
-- Import the necessary library.

-- Define the domains for the given functions.
def domain_func_1 (x : ℝ) : Prop := true

def domain_func_2 (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 2

def domain_func_3 (x : ℝ) : Prop := x ≥ -3 ∧ x ≠ 1

def domain_func_4 (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3

-- Prove the domains of each function.
theorem domain_of_func_1 : ∀ x : ℝ, domain_func_1 x :=
by sorry

theorem domain_of_func_2 : ∀ x : ℝ, domain_func_2 x ↔ (1 ≤ x ∧ x ≤ 2) :=
by sorry

theorem domain_of_func_3 : ∀ x : ℝ, domain_func_3 x ↔ (x ≥ -3 ∧ x ≠ 1) :=
by sorry

theorem domain_of_func_4 : ∀ x : ℝ, domain_func_4 x ↔ (2 ≤ x ∧ x ≤ 5 ∧ x ≠ 3) :=
by sorry

end NUMINAMATH_GPT_domain_of_func_1_domain_of_func_2_domain_of_func_3_domain_of_func_4_l1340_134098


namespace NUMINAMATH_GPT_man_speed_in_still_water_l1340_134091

theorem man_speed_in_still_water 
  (V_u : ℕ) (V_d : ℕ) 
  (hu : V_u = 34) 
  (hd : V_d = 48) : 
  V_s = (V_u + V_d) / 2 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l1340_134091


namespace NUMINAMATH_GPT_robin_packages_gum_l1340_134000

/-
Conditions:
1. Robin has 14 packages of candy.
2. There are 6 pieces in each candy package.
3. Robin has 7 additional pieces.
4. Each package of gum contains 6 pieces.

Proof Problem:
Prove that the number of packages of gum Robin has is 15.
-/
theorem robin_packages_gum (candies_packages : ℕ) (pieces_per_candy_package : ℕ)
                          (additional_pieces : ℕ) (pieces_per_gum_package : ℕ) :
  candies_packages = 14 →
  pieces_per_candy_package = 6 →
  additional_pieces = 7 →
  pieces_per_gum_package = 6 →
  (candies_packages * pieces_per_candy_package + additional_pieces) / pieces_per_gum_package = 15 :=
by intros h1 h2 h3 h4; sorry

end NUMINAMATH_GPT_robin_packages_gum_l1340_134000


namespace NUMINAMATH_GPT_identify_quadratic_equation_l1340_134083

/-- Proving which equation is a quadratic equation from given options -/
def is_quadratic_equation (eq : String) : Prop :=
  eq = "sqrt(x^2)=2" ∨ eq = "x^2 - x - 2" ∨ eq = "1/x^2 - 2=0" ∨ eq = "x^2=0"

theorem identify_quadratic_equation :
  ∀ (eq : String), is_quadratic_equation eq → eq = "x^2=0" :=
by
  intro eq h
  -- add proof steps here
  sorry

end NUMINAMATH_GPT_identify_quadratic_equation_l1340_134083


namespace NUMINAMATH_GPT_tan_ratio_of_triangle_l1340_134028

theorem tan_ratio_of_triangle (A B C : ℝ) (a b c : ℝ) (h1 : a * Real.cos B - b * Real.cos A = (3 / 5) * c) : 
  Real.tan A / Real.tan B = 4 :=
sorry

end NUMINAMATH_GPT_tan_ratio_of_triangle_l1340_134028


namespace NUMINAMATH_GPT_total_cars_produced_l1340_134020

def cars_produced_north_america : ℕ := 3884
def cars_produced_europe : ℕ := 2871
def cars_produced_asia : ℕ := 5273
def cars_produced_south_america : ℕ := 1945

theorem total_cars_produced : cars_produced_north_america + cars_produced_europe + cars_produced_asia + cars_produced_south_america = 13973 := by
  sorry

end NUMINAMATH_GPT_total_cars_produced_l1340_134020


namespace NUMINAMATH_GPT_net_displacement_east_of_A_total_fuel_consumed_l1340_134089

def distances : List Int := [22, -3, 4, -2, -8, -17, -2, 12, 7, -5]
def fuel_consumption_per_km : ℝ := 0.07

theorem net_displacement_east_of_A :
  List.sum distances = 8 := by
  sorry

theorem total_fuel_consumed :
  List.sum (distances.map Int.natAbs) * fuel_consumption_per_km = 5.74 := by
  sorry

end NUMINAMATH_GPT_net_displacement_east_of_A_total_fuel_consumed_l1340_134089


namespace NUMINAMATH_GPT_class_mean_correct_l1340_134013

noncomputable def new_class_mean (number_students_midterm : ℕ) (avg_score_midterm : ℚ)
                                 (number_students_next_day : ℕ) (avg_score_next_day : ℚ)
                                 (number_students_final_day : ℕ) (avg_score_final_day : ℚ)
                                 (total_students : ℕ) : ℚ :=
  let total_score_midterm := number_students_midterm * avg_score_midterm
  let total_score_next_day := number_students_next_day * avg_score_next_day
  let total_score_final_day := number_students_final_day * avg_score_final_day
  let total_score := total_score_midterm + total_score_next_day + total_score_final_day
  total_score / total_students

theorem class_mean_correct :
  new_class_mean 50 65 8 85 2 55 60 = 67 :=
by
  sorry

end NUMINAMATH_GPT_class_mean_correct_l1340_134013


namespace NUMINAMATH_GPT_divisor_of_12401_76_13_l1340_134058

theorem divisor_of_12401_76_13 (D : ℕ) (h1: 12401 = (D * 76) + 13) : D = 163 :=
sorry

end NUMINAMATH_GPT_divisor_of_12401_76_13_l1340_134058


namespace NUMINAMATH_GPT_candy_in_each_bag_l1340_134087

theorem candy_in_each_bag (total_candy : ℕ) (bags : ℕ) (h1 : total_candy = 16) (h2 : bags = 2) : total_candy / bags = 8 :=
by {
    sorry
}

end NUMINAMATH_GPT_candy_in_each_bag_l1340_134087


namespace NUMINAMATH_GPT_functional_equation_solution_l1340_134005

theorem functional_equation_solution (f : ℤ → ℝ) (hf : ∀ x y : ℤ, f (↑((x + y) / 3)) = (f x + f y) / 2) :
    ∃ c : ℝ, ∀ x : ℤ, x ≠ 0 → f x = c :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l1340_134005


namespace NUMINAMATH_GPT_total_games_l1340_134022

-- Defining the conditions.
def games_this_month : ℕ := 9
def games_last_month : ℕ := 8
def games_next_month : ℕ := 7

-- Theorem statement to prove the total number of games.
theorem total_games : games_this_month + games_last_month + games_next_month = 24 := by
  sorry

end NUMINAMATH_GPT_total_games_l1340_134022


namespace NUMINAMATH_GPT_probability_prime_factor_of_120_l1340_134076

open Nat

theorem probability_prime_factor_of_120 : 
  let s := Finset.range 61
  let primes := {2, 3, 5}
  let prime_factors_of_5_fact := primes ∩ s
  (prime_factors_of_5_fact.card : ℚ) / s.card = 1 / 20 :=
by
  sorry

end NUMINAMATH_GPT_probability_prime_factor_of_120_l1340_134076


namespace NUMINAMATH_GPT_max_value_expression_l1340_134044

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end NUMINAMATH_GPT_max_value_expression_l1340_134044


namespace NUMINAMATH_GPT_binders_required_l1340_134095

variables (b1 b2 B1 B2 d1 d2 b3 : ℕ)

def binding_rate_per_binder_per_day : ℚ := B1 / (↑b1 * d1)

def books_per_binder_in_d2_days : ℚ := binding_rate_per_binder_per_day b1 B1 d1 * ↑d2

def binding_rate_for_b2_binders : ℚ := B2 / ↑b2

theorem binders_required (b1 b2 B1 B2 d1 d2 b3 : ℕ)
  (h1 : binding_rate_per_binder_per_day b1 B1 d1 = binding_rate_for_b2_binders b2 B2)
  (h2 : books_per_binder_in_d2_days b1 B1 d1 d2 = binding_rate_for_b2_binders b2 B2) :
  b3 = b2 :=
sorry

end NUMINAMATH_GPT_binders_required_l1340_134095


namespace NUMINAMATH_GPT_orange_juice_fraction_l1340_134051

theorem orange_juice_fraction :
  let capacity1 := 500
  let capacity2 := 600
  let fraction1 := (1/4 : ℚ)
  let fraction2 := (1/3 : ℚ)
  let juice1 := capacity1 * fraction1
  let juice2 := capacity2 * fraction2
  let total_juice := juice1 + juice2
  let total_volume := capacity1 + capacity2
  (total_juice / total_volume = (13/44 : ℚ)) := sorry

end NUMINAMATH_GPT_orange_juice_fraction_l1340_134051


namespace NUMINAMATH_GPT_total_pay_is_880_l1340_134071

theorem total_pay_is_880 (X_pay Y_pay : ℝ) 
  (hY : Y_pay = 400)
  (hX : X_pay = 1.2 * Y_pay):
  X_pay + Y_pay = 880 :=
by
  sorry

end NUMINAMATH_GPT_total_pay_is_880_l1340_134071


namespace NUMINAMATH_GPT_largest_common_value_l1340_134084

theorem largest_common_value (a : ℕ) (h1 : a % 4 = 3) (h2 : a % 9 = 5) (h3 : a < 600) :
  a = 599 :=
sorry

end NUMINAMATH_GPT_largest_common_value_l1340_134084


namespace NUMINAMATH_GPT_ratio_of_areas_l1340_134060

theorem ratio_of_areas (n : ℕ) (r s : ℕ) (square_area : ℕ) (triangle_adf_area : ℕ)
  (h_square_area : square_area = 4)
  (h_triangle_adf_area : triangle_adf_area = n * square_area)
  (h_triangle_sim : s = 8 / r)
  (h_r_eq_n : r = n):
  (s / square_area) = 2 / n :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1340_134060


namespace NUMINAMATH_GPT_sum_of_roots_l1340_134079

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Prove that the sum of the roots of the given quadratic equation is 6
theorem sum_of_roots :
  (quadratic_eq 1 (-6) 9) x → (quadratic_eq 1 (-6) 9) y → x ≠ y → x + y = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l1340_134079


namespace NUMINAMATH_GPT_sum_of_all_x_l1340_134070

theorem sum_of_all_x (x1 x2 : ℝ) (h1 : (x1 + 5)^2 = 81) (h2 : (x2 + 5)^2 = 81) : x1 + x2 = -10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_all_x_l1340_134070


namespace NUMINAMATH_GPT_quadratic_solution_l1340_134085

theorem quadratic_solution (x : ℝ) (h : 2 * x ^ 2 - 2 = 0) : x = 1 ∨ x = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_l1340_134085


namespace NUMINAMATH_GPT_expression_bounds_l1340_134017

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  2 + Real.sqrt 2 ≤ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ∧ 
  (Real.sqrt (a^2 + (1 - b)^2 + 1) + 
   Real.sqrt (b^2 + (1 - c)^2 + 1) + 
   Real.sqrt (c^2 + (1 - d)^2 + 1) + 
   Real.sqrt (d^2 + (1 - a)^2 + 1)) ≤ 4 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_expression_bounds_l1340_134017


namespace NUMINAMATH_GPT_gcd_f_100_f_101_l1340_134067

def f (x : ℕ) : ℕ := x^2 - 2*x + 2023

theorem gcd_f_100_f_101 : Nat.gcd (f 100) (f 101) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_f_100_f_101_l1340_134067


namespace NUMINAMATH_GPT_whose_number_is_larger_l1340_134026

theorem whose_number_is_larger
    (vasya_prod : ℕ := 4^12)
    (petya_prod : ℕ := 2^25) :
    petya_prod > vasya_prod :=
    by
    sorry

end NUMINAMATH_GPT_whose_number_is_larger_l1340_134026


namespace NUMINAMATH_GPT_initial_back_squat_weight_l1340_134037

-- Define a structure to encapsulate the conditions
structure squat_conditions where
  initial_back_squat : ℝ
  front_squat_ratio : ℝ := 0.8
  back_squat_increase : ℝ := 50
  front_squat_triple_ratio : ℝ := 0.9
  total_weight_moved : ℝ := 540

-- Using the conditions provided to prove John's initial back squat weight
theorem initial_back_squat_weight (c : squat_conditions) :
  (3 * 3 * (c.front_squat_triple_ratio * (c.front_squat_ratio * c.initial_back_squat)) = c.total_weight_moved) →
  c.initial_back_squat = 540 / 6.48 := sorry

end NUMINAMATH_GPT_initial_back_squat_weight_l1340_134037


namespace NUMINAMATH_GPT_prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l1340_134055

section ParkingProblem

variable (P_A_more_1_no_more_2 : ℚ) (P_A_more_than_14 : ℚ)

theorem prob_A_fee_exactly_6_yuan :
  (P_A_more_1_no_more_2 = 1/3) →
  (P_A_more_than_14 = 5/12) →
  (1 - (P_A_more_1_no_more_2 + P_A_more_than_14)) = 1/4 :=
by
  -- Skipping the proof
  intros _ _
  sorry

theorem prob_sum_fees_A_B_36_yuan :
  (1/4 : ℚ) = 1/4 :=
by
  -- Skipping the proof
  exact rfl

end ParkingProblem

end NUMINAMATH_GPT_prob_A_fee_exactly_6_yuan_prob_sum_fees_A_B_36_yuan_l1340_134055


namespace NUMINAMATH_GPT_ab_value_l1340_134002

theorem ab_value (a b : ℝ) (h : 6 * a = 20 ∧ 7 * b = 20) : 84 * (a * b) = 800 :=
by sorry

end NUMINAMATH_GPT_ab_value_l1340_134002


namespace NUMINAMATH_GPT_HCl_moles_formed_l1340_134052

-- Define the conditions for the problem:
def moles_H2SO4 := 1 -- moles of H2SO4
def moles_NaCl := 1 -- moles of NaCl
def reaction : List (Int × String) :=
  [(1, "H2SO4"), (2, "NaCl"), (2, "HCl"), (1, "Na2SO4")]  -- the reaction coefficients in (coefficient, chemical) pairs

-- Define the function that calculates the product moles based on limiting reactant
def calculate_HCl (moles_H2SO4 : Int) (moles_NaCl : Int) : Int :=
  if moles_NaCl < 2 then moles_NaCl else 2 * (moles_H2SO4 / 1)

-- Specify the theorem to be proven with the given conditions
theorem HCl_moles_formed :
  calculate_HCl moles_H2SO4 moles_NaCl = 1 :=
by
  sorry -- Proof can be filled in later

end NUMINAMATH_GPT_HCl_moles_formed_l1340_134052


namespace NUMINAMATH_GPT_pool_capacity_l1340_134099

theorem pool_capacity (C : ℝ) (h1 : C * 0.70 = C * 0.40 + 300)
  (h2 : 300 = C * 0.30) : C = 1000 :=
sorry

end NUMINAMATH_GPT_pool_capacity_l1340_134099


namespace NUMINAMATH_GPT_min_width_l1340_134019

theorem min_width (w : ℝ) (h : w * (w + 20) ≥ 150) : w ≥ 10 := by
  sorry

end NUMINAMATH_GPT_min_width_l1340_134019


namespace NUMINAMATH_GPT_problem_inequality_solution_set_problem_minimum_value_l1340_134011

noncomputable def f (x : ℝ) := x^2 / (x - 1)

theorem problem_inequality_solution_set : 
  ∀ x : ℝ, 1 < x ∧ x < (1 + Real.sqrt 5) / 2 → f x > 2 * x + 1 :=
sorry

theorem problem_minimum_value : ∀ x : ℝ, x > 1 → (f x ≥ 4) ∧ (f 2 = 4) :=
sorry

end NUMINAMATH_GPT_problem_inequality_solution_set_problem_minimum_value_l1340_134011


namespace NUMINAMATH_GPT_num_zeros_in_binary_l1340_134066

namespace BinaryZeros

def expression : ℕ := ((18 * 8192 + 8 * 128 - 12 * 16) / 6) + (4 * 64) + (3 ^ 5) - (25 * 2)

def binary_zeros (n : ℕ) : ℕ :=
  (Nat.digits 2 n).count 0

theorem num_zeros_in_binary :
  binary_zeros expression = 6 :=
by
  sorry

end BinaryZeros

end NUMINAMATH_GPT_num_zeros_in_binary_l1340_134066


namespace NUMINAMATH_GPT_EFGH_perimeter_l1340_134074

noncomputable def perimeter_rectangle_EFGH (WE EX WY XZ : ℕ) : Rat :=
  let WX := Real.sqrt (WE ^ 2 + EX ^ 2)
  let p := 15232
  let q := 100
  p / q

theorem EFGH_perimeter :
  let WE := 12
  let EX := 16
  let WY := 24
  let XZ := 32
  perimeter_rectangle_EFGH WE EX WY XZ = 15232 / 100 :=
by
  sorry

end NUMINAMATH_GPT_EFGH_perimeter_l1340_134074


namespace NUMINAMATH_GPT_sum_of_angles_is_290_l1340_134057

-- Given conditions
def angle_A : ℝ := 40
def angle_C : ℝ := 70
def angle_D : ℝ := 50
def angle_F : ℝ := 60

-- Calculate angle B (which is same as angle E)
def angle_B : ℝ := 180 - angle_A - angle_C
def angle_E := angle_B  -- by the condition that B and E are identical

-- Total sum of angles
def total_angle_sum : ℝ := angle_A + angle_B + angle_C + angle_D + angle_F

-- Theorem statement
theorem sum_of_angles_is_290 : total_angle_sum = 290 := by
  sorry

end NUMINAMATH_GPT_sum_of_angles_is_290_l1340_134057


namespace NUMINAMATH_GPT_roots_polynomial_l1340_134040

theorem roots_polynomial (a b c : ℝ) (h1 : a + b + c = 18) (h2 : a * b + b * c + c * a = 19) (h3 : a * b * c = 8) : 
  (1 + a) * (1 + b) * (1 + c) = 46 :=
by
  sorry

end NUMINAMATH_GPT_roots_polynomial_l1340_134040


namespace NUMINAMATH_GPT_part_I_part_II_l1340_134039

noncomputable def f (x a : ℝ) := |x - 4| + |x - a|

theorem part_I (x : ℝ) : (f x 2 > 10) ↔ (x > 8 ∨ x < -2) :=
by sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≥ 1) ↔ (a ≥ 5 ∨ a ≤ 3) :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l1340_134039


namespace NUMINAMATH_GPT_expression_value_l1340_134034

theorem expression_value (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_expression_value_l1340_134034


namespace NUMINAMATH_GPT_percentage_change_area_l1340_134030

theorem percentage_change_area
    (L B : ℝ)
    (Area_original : ℝ) (Area_new : ℝ)
    (Length_new : ℝ) (Breadth_new : ℝ) :
    Area_original = L * B →
    Length_new = L / 2 →
    Breadth_new = 3 * B →
    Area_new = Length_new * Breadth_new →
    (Area_new - Area_original) / Area_original * 100 = 50 :=
  by
  intro h_orig_area hl_new hb_new ha_new
  sorry

end NUMINAMATH_GPT_percentage_change_area_l1340_134030


namespace NUMINAMATH_GPT_q_evaluation_l1340_134036

def q (x y : ℤ) : ℤ :=
if x ≥ 0 ∧ y ≤ 0 then x - y
else if x < 0 ∧ y > 0 then x + 3 * y
else 4 * x - 2 * y

theorem q_evaluation : q (q 2 (-3)) (q (-4) 1) = 6 :=
by
  sorry

end NUMINAMATH_GPT_q_evaluation_l1340_134036


namespace NUMINAMATH_GPT_parallel_line_equation_l1340_134023

theorem parallel_line_equation :
  ∃ (c : ℝ), 
    (∀ x : ℝ, y = (3 / 4) * x + 6 → (y = (3 / 4) * x + c → abs (c - 6) = 4 * (5 / 4))) → c = 1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_equation_l1340_134023


namespace NUMINAMATH_GPT_path_area_and_cost_l1340_134059

-- Define the initial conditions
def field_length : ℝ := 65
def field_width : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_m : ℝ := 2

-- Define the extended dimensions including the path
def extended_length := field_length + 2 * path_width
def extended_width := field_width + 2 * path_width

-- Define the areas
def area_with_path := extended_length * extended_width
def area_of_field := field_length * field_width
def area_of_path := area_with_path - area_of_field

-- Define the cost
def cost_of_constructing_path := area_of_path * cost_per_sq_m

theorem path_area_and_cost :
  area_of_path = 625 ∧ cost_of_constructing_path = 1250 :=
by
  sorry

end NUMINAMATH_GPT_path_area_and_cost_l1340_134059


namespace NUMINAMATH_GPT_rectangle_circles_l1340_134064

theorem rectangle_circles (p q : Prop) (hp : p) (hq : ¬ q) : p ∨ q :=
by sorry

end NUMINAMATH_GPT_rectangle_circles_l1340_134064


namespace NUMINAMATH_GPT_total_parallelograms_in_grid_l1340_134043

theorem total_parallelograms_in_grid (n : ℕ) : 
  ∃ p : ℕ, p = 3 * Nat.choose (n + 2) 4 :=
sorry

end NUMINAMATH_GPT_total_parallelograms_in_grid_l1340_134043


namespace NUMINAMATH_GPT_totalGoals_l1340_134073

-- Define the conditions
def louieLastMatchGoals : Nat := 4
def louiePreviousGoals : Nat := 40
def gamesPerSeason : Nat := 50
def seasons : Nat := 3
def brotherGoalsPerGame := 2 * louieLastMatchGoals

-- Define the properties derived from the conditions
def totalBrotherGoals : Nat := brotherGoalsPerGame * gamesPerSeason * seasons
def totalLouieGoals : Nat := louiePreviousGoals + louieLastMatchGoals

-- State what needs to be proved
theorem totalGoals : louiePreviousGoals + louieLastMatchGoals + brotherGoalsPerGame * gamesPerSeason * seasons = 1244 := by
  sorry

end NUMINAMATH_GPT_totalGoals_l1340_134073


namespace NUMINAMATH_GPT_triangle_angle_C_l1340_134004

theorem triangle_angle_C (A B C : ℝ) (sin cos : ℝ → ℝ) 
  (h1 : 3 * sin A + 4 * cos B = 6)
  (h2 : 4 * sin B + 3 * cos A = 1)
  (triangle_sum : A + B + C = 180) :
  C = 30 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_C_l1340_134004


namespace NUMINAMATH_GPT_sin_theta_value_l1340_134077

open Real

theorem sin_theta_value
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo (3 * π / 4) (5 * π / 4))
  (h2 : sin (θ - π / 4) = 5 / 13) :
  sin θ = - (7 * sqrt 2) / 26 :=
  sorry

end NUMINAMATH_GPT_sin_theta_value_l1340_134077


namespace NUMINAMATH_GPT_probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l1340_134056

theorem probability_winning_on_first_draw : 
  let red := 1 
  let yellow := 3 
  red / (red + yellow) = 1 / 4 :=
by 
  sorry

theorem optimal_ball_to_add_for_fine_gift :
  let red := 1 
  let yellow := 3
  -- After adding a red ball: 2 red, 3 yellow
  let p1 := (2 * 1 + 3 * 2) / (2 + 3) / (1 + 3) = (2/5)
  -- After adding a yellow ball: 1 red, 4 yellow
  let p2 := (1 * 0 + 4 * 3) / (1 + 4) / (1 + 3) = (3/5)
  p1 < p2 :=
by 
  sorry

end NUMINAMATH_GPT_probability_winning_on_first_draw_optimal_ball_to_add_for_fine_gift_l1340_134056


namespace NUMINAMATH_GPT_integers_in_range_eq_l1340_134021

theorem integers_in_range_eq :
  {i : ℤ | i > -2 ∧ i ≤ 3} = {-1, 0, 1, 2, 3} :=
by
  sorry

end NUMINAMATH_GPT_integers_in_range_eq_l1340_134021


namespace NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l1340_134086

/-- 
  Given that (9, -15) is the midpoint of the segment with one endpoint (7, 4),
  find the sum of the coordinates of the other endpoint.
-/
theorem sum_of_other_endpoint_coordinates : 
  ∃ x y : ℤ, ((7 + x) / 2 = 9 ∧ (4 + y) / 2 = -15) ∧ (x + y = -23) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l1340_134086


namespace NUMINAMATH_GPT_non_binary_listeners_l1340_134010

theorem non_binary_listeners (listen_total males_listen females_dont_listen non_binary_dont_listen dont_listen_total : ℕ) 
  (h_listen_total : listen_total = 250) 
  (h_males_listen : males_listen = 85) 
  (h_females_dont_listen : females_dont_listen = 95) 
  (h_non_binary_dont_listen : non_binary_dont_listen = 45) 
  (h_dont_listen_total : dont_listen_total = 230) : 
  (listen_total - males_listen - (dont_listen_total - females_dont_listen - non_binary_dont_listen)) = 70 :=
by 
  -- Let nbl be the number of non-binary listeners
  let nbl := listen_total - males_listen - (dont_listen_total - females_dont_listen - non_binary_dont_listen)
  -- We need to show nbl = 70
  show nbl = 70
  sorry

end NUMINAMATH_GPT_non_binary_listeners_l1340_134010


namespace NUMINAMATH_GPT_acrobat_eq_two_lambs_l1340_134065

variables (ACROBAT DOG BARREL SPOOL LAMB : ℝ)

axiom acrobat_dog_eq_two_barrels : ACROBAT + DOG = 2 * BARREL
axiom dog_eq_two_spools : DOG = 2 * SPOOL
axiom lamb_spool_eq_barrel : LAMB + SPOOL = BARREL

theorem acrobat_eq_two_lambs : ACROBAT = 2 * LAMB :=
by
  sorry

end NUMINAMATH_GPT_acrobat_eq_two_lambs_l1340_134065


namespace NUMINAMATH_GPT_division_result_l1340_134029

def expr := 180 / (12 + 13 * 2)

theorem division_result : expr = 90 / 19 := by
  sorry

end NUMINAMATH_GPT_division_result_l1340_134029


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1340_134049

-- Part (1) Statement
theorem part1_solution (x : ℝ) (m : ℝ) (h_m : m = -1) :
  (3 * x - m) / 2 - (x + m) / 3 = 5 / 6 → x = 0 :=
by
  intros h_eq
  rw [h_m] at h_eq
  sorry  -- Proof to be filled in

-- Part (2) Statement
theorem part2_solution (x m : ℝ) (h_x : x = 5)
  (h_eq : (3 * x - m) / 2 - (x + m) / 3 = 5 / 6) :
  (1 / 2) * m^2 + 2 * m = 30 :=
by
  rw [h_x] at h_eq
  sorry  -- Proof to be filled in

end NUMINAMATH_GPT_part1_solution_part2_solution_l1340_134049


namespace NUMINAMATH_GPT_percentage_exceeds_l1340_134046

-- Defining the constants and conditions
variables {y z x : ℝ}

-- Conditions
def condition1 (y x : ℝ) : Prop := x = 0.6 * y
def condition2 (x z : ℝ) : Prop := z = 1.25 * x

-- Proposition to prove
theorem percentage_exceeds (hyx : condition1 y x) (hxz : condition2 x z) : y = 4/3 * z :=
by 
  -- We skip the proof as requested
  sorry

end NUMINAMATH_GPT_percentage_exceeds_l1340_134046


namespace NUMINAMATH_GPT_least_number_to_subtract_997_l1340_134094

theorem least_number_to_subtract_997 (x : ℕ) (h : x = 997) 
  : ∃ y : ℕ, ∀ m (h₁ : m = (997 - y)), 
    m % 5 = 3 ∧ m % 9 = 3 ∧ m % 11 = 3 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_997_l1340_134094


namespace NUMINAMATH_GPT_problem1_problem2_l1340_134047

-- Problem 1: y(x + y) + (x + y)(x - y) = x^2
theorem problem1 (x y : ℝ) : y * (x + y) + (x + y) * (x - y) = x^2 := 
by sorry

-- Problem 2: ( (2m + 1) / (m + 1) + m - 1 ) ÷ ( (m + 2) / (m^2 + 2m + 1) ) = m^2 + m
theorem problem2 (m : ℝ) (h1 : m ≠ -1) : 
  ( (2 * m + 1) / (m + 1) + m - 1 ) / ( (m + 2) / ((m + 1)^2) ) = m^2 + m := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1340_134047


namespace NUMINAMATH_GPT_john_weight_loss_percentage_l1340_134069

def john_initial_weight := 220
def john_final_weight_after_gain := 200
def weight_gain := 2

theorem john_weight_loss_percentage : 
  ∃ P : ℝ, (john_initial_weight - (P / 100) * john_initial_weight + weight_gain = john_final_weight_after_gain) ∧ P = 10 :=
sorry

end NUMINAMATH_GPT_john_weight_loss_percentage_l1340_134069


namespace NUMINAMATH_GPT_find_s_l1340_134088

theorem find_s (s t : ℝ) (h1 : 8 * s + 4 * t = 160) (h2 : t = 2 * s - 3) : s = 10.75 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l1340_134088


namespace NUMINAMATH_GPT_range_of_m_l1340_134024

-- Definitions and the main problem statement
def f (m x : ℝ) : ℝ := m * x^2 - m * x - 1

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f m x < 0) ↔ (-4 < m ∧ m ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1340_134024


namespace NUMINAMATH_GPT_prism_faces_l1340_134093

theorem prism_faces (E V F : ℕ) (n : ℕ) 
  (h1 : E + V = 40) 
  (h2 : E = 3 * F - 6) 
  (h3 : V - E + F = 2)
  (h4 : V = 2 * n)
  : F = 10 := 
by
  sorry

end NUMINAMATH_GPT_prism_faces_l1340_134093


namespace NUMINAMATH_GPT_initial_mixture_volume_l1340_134096

variable (p q : ℕ) (x : ℕ)

theorem initial_mixture_volume :
  (3 * x) + (2 * x) = 5 * x →
  (3 * x) / (2 * x + 12) = 3 / 4 →
  5 * x = 30 :=
by
  sorry

end NUMINAMATH_GPT_initial_mixture_volume_l1340_134096


namespace NUMINAMATH_GPT_prove_q_l1340_134061

theorem prove_q 
  (p q : ℝ)
  (h : (∀ x, (x + 3) * (x + p) = x^2 + q * x + 12)) : 
  q = 7 :=
sorry

end NUMINAMATH_GPT_prove_q_l1340_134061


namespace NUMINAMATH_GPT_number_of_dice_l1340_134075

theorem number_of_dice (n : ℕ) (h : (1 / 6 : ℝ) ^ (n - 1) = 0.0007716049382716049) : n = 5 :=
sorry

end NUMINAMATH_GPT_number_of_dice_l1340_134075


namespace NUMINAMATH_GPT_prove_ratio_chickens_pigs_horses_sheep_l1340_134006

noncomputable def ratio_chickens_pigs_horses_sheep (c p h s : ℕ) : Prop :=
  (∃ k : ℕ, c = 26*k ∧ p = 5*k) ∧
  (∃ l : ℕ, s = 25*l ∧ h = 9*l) ∧
  (∃ m : ℕ, p = 10*m ∧ h = 3*m) ∧
  c = 156 ∧ p = 30 ∧ h = 9 ∧ s = 25

theorem prove_ratio_chickens_pigs_horses_sheep (c p h s : ℕ) :
  ratio_chickens_pigs_horses_sheep c p h s :=
sorry

end NUMINAMATH_GPT_prove_ratio_chickens_pigs_horses_sheep_l1340_134006
