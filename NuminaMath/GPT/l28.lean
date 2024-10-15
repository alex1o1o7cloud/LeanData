import Mathlib

namespace NUMINAMATH_GPT_mason_grandmother_age_l28_2844

-- Defining the ages of Mason, Sydney, Mason's father, and Mason's grandmother
def mason_age : ℕ := 20

def sydney_age (S : ℕ) : Prop :=
  mason_age = S / 3

def father_age (S F : ℕ) : Prop :=
  F = S + 6

def grandmother_age (F G : ℕ) : Prop :=
  G = 2 * F

theorem mason_grandmother_age (S F G : ℕ) (h1 : sydney_age S) (h2 : father_age S F) (h3 : grandmother_age F G) : G = 132 :=
by
  -- leaving the proof as a sorry
  sorry

end NUMINAMATH_GPT_mason_grandmother_age_l28_2844


namespace NUMINAMATH_GPT_find_coordinates_B_l28_2860

variable (B : ℝ × ℝ)

def A : ℝ × ℝ := (2, 3)
def C : ℝ × ℝ := (0, 1)
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

theorem find_coordinates_B (h : vec A B = (-2) • vec B C) : B = (-2, 5/3) :=
by
  -- Here you would provide proof steps
  sorry

end NUMINAMATH_GPT_find_coordinates_B_l28_2860


namespace NUMINAMATH_GPT_consecutive_integers_sqrt19_sum_l28_2876

theorem consecutive_integers_sqrt19_sum :
  ∃ a b : ℤ, (a < ⌊Real.sqrt 19⌋ ∧ ⌊Real.sqrt 19⌋ < b ∧ a + 1 = b) ∧ a + b = 9 := 
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_sqrt19_sum_l28_2876


namespace NUMINAMATH_GPT_tomatoes_first_shipment_l28_2869

theorem tomatoes_first_shipment :
  ∃ X : ℕ, 
    (∀Y : ℕ, 
      (Y = 300) → -- Saturday sale
      (X - Y = X - 300) ∧
      (∀Z : ℕ, 
        (Z = 200) → -- Sunday rotting
        (X - 300 - Z = X - 500) ∧
        (∀W : ℕ, 
          (W = 2 * X) → -- Monday new shipment
          (X - 500 + W = 2500) →
          (X = 1000)
        )
      )
    ) :=
by
  sorry

end NUMINAMATH_GPT_tomatoes_first_shipment_l28_2869


namespace NUMINAMATH_GPT_total_students_l28_2811

theorem total_students (groups students_per_group : ℕ) (h : groups = 6) (k : students_per_group = 5) :
  groups * students_per_group = 30 := 
by
  sorry

end NUMINAMATH_GPT_total_students_l28_2811


namespace NUMINAMATH_GPT_a_lt_c_lt_b_l28_2870

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := 2 * Real.sqrt 2 * Real.sin (30.5 * Real.pi / 180) * Real.cos (30.5 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem a_lt_c_lt_b : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_a_lt_c_lt_b_l28_2870


namespace NUMINAMATH_GPT_find_number_l28_2885

theorem find_number (x : ℚ) (h : x / 5 = 3 * (x / 6) - 40) : x = 400 / 3 :=
sorry

end NUMINAMATH_GPT_find_number_l28_2885


namespace NUMINAMATH_GPT_binomial_square_b_value_l28_2849

theorem binomial_square_b_value (b : ℝ) (h : ∃ c : ℝ, (9 * x^2 + 24 * x + b) = (3 * x + c) ^ 2) : b = 16 :=
sorry

end NUMINAMATH_GPT_binomial_square_b_value_l28_2849


namespace NUMINAMATH_GPT_computation_result_l28_2836

-- Define the vectors and scalar multiplications
def v1 : ℤ × ℤ := (3, -9)
def v2 : ℤ × ℤ := (2, -7)
def v3 : ℤ × ℤ := (-1, 4)

noncomputable def result : ℤ × ℤ := 
  let scalar_mult (m : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (m * v.1, m * v.2)
  scalar_mult 5 v1 - scalar_mult 3 v2 + scalar_mult 2 v3

-- The main theorem
theorem computation_result : result = (7, -16) :=
  by 
    -- Skip the proof as required
    sorry

end NUMINAMATH_GPT_computation_result_l28_2836


namespace NUMINAMATH_GPT_ounces_per_cup_l28_2872

theorem ounces_per_cup (total_ounces : ℕ) (total_cups : ℕ) 
  (h : total_ounces = 264 ∧ total_cups = 33) : total_ounces / total_cups = 8 :=
by
  sorry

end NUMINAMATH_GPT_ounces_per_cup_l28_2872


namespace NUMINAMATH_GPT_problem_statement_l28_2818

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x : ℝ, f x = x^2 + x + 1) 
  (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, |x - 1| < b → |f x - 3| < a) ↔ b ≤ a / 3 :=
sorry

end NUMINAMATH_GPT_problem_statement_l28_2818


namespace NUMINAMATH_GPT_positive_value_of_m_l28_2865

theorem positive_value_of_m (m : ℝ) (h : (64 * m^2 - 60 * m) = 0) : m = 15 / 16 :=
sorry

end NUMINAMATH_GPT_positive_value_of_m_l28_2865


namespace NUMINAMATH_GPT_vending_machine_problem_l28_2898

variable (x n : ℕ)

theorem vending_machine_problem (h : 25 * x + 10 * 15 + 5 * 30 = 25 * 25 + 10 * 5 + 5 * n) (hx : x = 25) :
  n = 50 := by
sorry

end NUMINAMATH_GPT_vending_machine_problem_l28_2898


namespace NUMINAMATH_GPT_range_of_a_l28_2822

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (a x : ℝ) : ℝ := f a x - g x
noncomputable def k (x : ℝ) : ℝ := (Real.log x + x) / x^2

theorem range_of_a (a : ℝ) (h_zero : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h a x₁ = 0 ∧ h a x₂ = 0) :
  0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l28_2822


namespace NUMINAMATH_GPT_cos_largest_angle_value_l28_2813

noncomputable def cos_largest_angle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : ℝ :=
  (a * a + b * b - c * c) / (2 * a * b)

theorem cos_largest_angle_value : cos_largest_angle 2 3 4 (by rfl) (by rfl) (by rfl) = -1 / 4 := 
sorry

end NUMINAMATH_GPT_cos_largest_angle_value_l28_2813


namespace NUMINAMATH_GPT_socks_difference_l28_2878

-- Definitions of the conditions
def week1 : ℕ := 12
def week2 (S : ℕ) : ℕ := S
def week3 (S : ℕ) : ℕ := (12 + S) / 2
def week4 (S : ℕ) : ℕ := (12 + S) / 2 - 3
def total (S : ℕ) : ℕ := week1 + week2 S + week3 S + week4 S

-- Statement of the theorem
theorem socks_difference (S : ℕ) (h : total S = 57) : S - week1 = 1 :=
by 
  -- Proof is not required
  sorry

end NUMINAMATH_GPT_socks_difference_l28_2878


namespace NUMINAMATH_GPT_range_of_a_l28_2893

open Set

theorem range_of_a (a : ℝ) :
  (M : Set ℝ) = { x | -1 ≤ x ∧ x ≤ 2 } →
  (N : Set ℝ) = { x | 1 - 3 * a < x ∧ x ≤ 2 * a } →
  M ∩ N = M →
  1 ≤ a :=
by
  intro hM hN h_inter
  sorry

end NUMINAMATH_GPT_range_of_a_l28_2893


namespace NUMINAMATH_GPT_oliver_dishes_count_l28_2808

def total_dishes : ℕ := 42
def mango_salsa_dishes : ℕ := 5
def fresh_mango_dishes : ℕ := total_dishes / 6
def mango_jelly_dishes : ℕ := 2
def strawberry_dishes : ℕ := 3
def pineapple_dishes : ℕ := 5
def kiwi_dishes : ℕ := 4
def mango_dishes_oliver_picks_out : ℕ := 3

def total_mango_dishes : ℕ := mango_salsa_dishes + fresh_mango_dishes + mango_jelly_dishes
def mango_dishes_oliver_wont_eat : ℕ := total_mango_dishes - mango_dishes_oliver_picks_out
def max_strawberry_pineapple_dishes : ℕ := strawberry_dishes

def dishes_left_for_oliver : ℕ := total_dishes - mango_dishes_oliver_wont_eat - max_strawberry_pineapple_dishes

theorem oliver_dishes_count : dishes_left_for_oliver = 28 := 
by 
  sorry

end NUMINAMATH_GPT_oliver_dishes_count_l28_2808


namespace NUMINAMATH_GPT_number_of_cars_l28_2881

theorem number_of_cars (people_per_car : ℝ) (total_people : ℝ) (h1 : people_per_car = 63.0) (h2 : total_people = 189) : total_people / people_per_car = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_cars_l28_2881


namespace NUMINAMATH_GPT_value_of_expression_l28_2850

theorem value_of_expression (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by
sorry

end NUMINAMATH_GPT_value_of_expression_l28_2850


namespace NUMINAMATH_GPT_sam_compound_interest_l28_2884

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

theorem sam_compound_interest : 
  compound_interest 3000 0.10 2 1 = 3307.50 :=
by
  sorry

end NUMINAMATH_GPT_sam_compound_interest_l28_2884


namespace NUMINAMATH_GPT_find_z_plus_one_over_y_l28_2841

variable {x y z : ℝ}

theorem find_z_plus_one_over_y (h1 : x * y * z = 1) 
                                (h2 : x + 1 / z = 7) 
                                (h3 : y + 1 / x = 31) 
                                (h4 : 0 < x ∧ 0 < y ∧ 0 < z) : 
                              z + 1 / y = 5 / 27 := 
by
  sorry

end NUMINAMATH_GPT_find_z_plus_one_over_y_l28_2841


namespace NUMINAMATH_GPT_geom_seq_common_ratio_l28_2897

theorem geom_seq_common_ratio (a₁ a₂ a₃ a₄ q : ℝ) 
  (h1 : a₁ + a₄ = 18)
  (h2 : a₂ * a₃ = 32)
  (h3 : a₂ = a₁ * q)
  (h4 : a₃ = a₁ * q^2)
  (h5 : a₄ = a₁ * q^3) : 
  q = 2 ∨ q = (1 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_geom_seq_common_ratio_l28_2897


namespace NUMINAMATH_GPT_remainder_of_2_pow_2005_mod_7_l28_2855

theorem remainder_of_2_pow_2005_mod_7 :
  2 ^ 2005 % 7 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_of_2_pow_2005_mod_7_l28_2855


namespace NUMINAMATH_GPT_sum_of_eight_numbers_l28_2837

theorem sum_of_eight_numbers (nums : List ℝ) (h_len : nums.length = 8) (h_avg : (nums.sum / 8) = 5.5) : nums.sum = 44 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_eight_numbers_l28_2837


namespace NUMINAMATH_GPT_greatest_integer_less_than_M_over_100_l28_2879

theorem greatest_integer_less_than_M_over_100 :
  (1 / (Nat.factorial 3 * Nat.factorial 16) +
   1 / (Nat.factorial 4 * Nat.factorial 15) +
   1 / (Nat.factorial 5 * Nat.factorial 14) +
   1 / (Nat.factorial 6 * Nat.factorial 13) +
   1 / (Nat.factorial 7 * Nat.factorial 12) +
   1 / (Nat.factorial 8 * Nat.factorial 11) +
   1 / (Nat.factorial 9 * Nat.factorial 10) = M / (Nat.factorial 2 * Nat.factorial 17)) →
  (⌊(M : ℚ) / 100⌋ = 27) := 
sorry

end NUMINAMATH_GPT_greatest_integer_less_than_M_over_100_l28_2879


namespace NUMINAMATH_GPT_ratio_owners_on_horse_l28_2891

-- Definitions based on the given conditions.
def number_of_horses : Nat := 12
def number_of_owners : Nat := 12
def total_legs_walking_on_ground : Nat := 60
def owner_leg_count : Nat := 2
def horse_leg_count : Nat := 4
def total_owners_leg_horse_count : Nat := owner_leg_count + horse_leg_count

-- Prove the ratio of the number of owners on their horses' back to the total number of owners is 1:6
theorem ratio_owners_on_horse (R W : Nat) 
  (h1 : R + W = number_of_owners)
  (h2 : total_owners_leg_horse_count * W = total_legs_walking_on_ground) :
  R = 2 → W = 10 → (R : Nat)/(number_of_owners : Nat) = (1 : Nat)/(6 : Nat) := 
sorry

end NUMINAMATH_GPT_ratio_owners_on_horse_l28_2891


namespace NUMINAMATH_GPT_base_not_divisible_by_5_l28_2890

def is_not_divisible_by_5 (c : ℤ) : Prop :=
  ¬(∃ k : ℤ, c = 5 * k)

def check_not_divisible_by_5 (b : ℤ) : Prop :=
  is_not_divisible_by_5 (3 * b^3 - 3 * b^2 - b)

theorem base_not_divisible_by_5 :
  check_not_divisible_by_5 6 ∧ check_not_divisible_by_5 8 :=
by 
  sorry

end NUMINAMATH_GPT_base_not_divisible_by_5_l28_2890


namespace NUMINAMATH_GPT_tangent_point_x_coordinate_l28_2834

-- Define the function representing the curve.
def curve (x : ℝ) : ℝ := x^2 + 1

-- Define the derivative of the curve.
def derivative (x : ℝ) : ℝ := 2 * x

-- The statement to be proved.
theorem tangent_point_x_coordinate (x : ℝ) (h : derivative x = 4) : x = 2 :=
sorry

end NUMINAMATH_GPT_tangent_point_x_coordinate_l28_2834


namespace NUMINAMATH_GPT_fraction_a_over_b_l28_2846

theorem fraction_a_over_b (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 3 * y - 6 * x = b) (hb : b ≠ 0) : a / b = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_a_over_b_l28_2846


namespace NUMINAMATH_GPT_cakes_served_during_lunch_l28_2826

theorem cakes_served_during_lunch (T D L : ℕ) (h1 : T = 15) (h2 : D = 9) : L = T - D → L = 6 :=
by
  intros h
  rw [h1, h2] at h
  exact h

end NUMINAMATH_GPT_cakes_served_during_lunch_l28_2826


namespace NUMINAMATH_GPT_tangent_line_equation_at_1_2_l28_2838

noncomputable def f (x : ℝ) : ℝ := 2 / x

theorem tangent_line_equation_at_1_2 :
  let x₀ := 1
  let y₀ := 2
  let slope := -2
  ∀ (x y : ℝ),
    y - y₀ = slope * (x - x₀) →
    2 * x + y - 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_at_1_2_l28_2838


namespace NUMINAMATH_GPT_time_to_cover_escalator_l28_2862

noncomputable def average_speed (initial_speed final_speed : ℝ) : ℝ :=
  (initial_speed + final_speed) / 2

noncomputable def combined_speed (escalator_speed person_average_speed : ℝ) : ℝ :=
  escalator_speed + person_average_speed

noncomputable def coverage_time (length combined_speed : ℝ) : ℝ :=
  length / combined_speed

theorem time_to_cover_escalator
  (escalator_speed : ℝ := 20)
  (length : ℝ := 300)
  (initial_person_speed : ℝ := 3)
  (final_person_speed : ℝ := 5) :
  coverage_time length (combined_speed escalator_speed (average_speed initial_person_speed final_person_speed)) = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cover_escalator_l28_2862


namespace NUMINAMATH_GPT_smallest_positive_integer_l28_2814

theorem smallest_positive_integer (n : ℕ) (h₁ : n > 1) (h₂ : n % 2 = 1) (h₃ : n % 3 = 1) (h₄ : n % 4 = 1) (h₅ : n % 5 = 1) : n = 61 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l28_2814


namespace NUMINAMATH_GPT_compound_interest_correct_l28_2882

noncomputable def compoundInterest (P: ℝ) (r: ℝ) (n: ℝ) (t: ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_correct :
  compoundInterest 5000 0.04 1 3 - 5000 = 624.32 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_correct_l28_2882


namespace NUMINAMATH_GPT_find_speed_of_first_train_l28_2810

noncomputable def relative_speed (length1 length2 : ℕ) (time_seconds : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hours := time_seconds / 3600
  total_length_km / time_hours

theorem find_speed_of_first_train
  (length1 : ℕ)   -- Length of the first train in meters
  (length2 : ℕ)   -- Length of the second train in meters
  (speed2 : ℝ)    -- Speed of the second train in km/h
  (time_seconds : ℝ)  -- Time in seconds to be clear from each other
  (correct_speed1 : ℝ)  -- Correct speed of the first train in km/h
  (h_length1 : length1 = 160)
  (h_length2 : length2 = 280)
  (h_speed2 : speed2 = 30)
  (h_time_seconds : time_seconds = 21.998240140788738)
  (h_correct_speed1 : correct_speed1 = 41.98) :
  relative_speed length1 length2 time_seconds = speed2 + correct_speed1 :=
by
  sorry

end NUMINAMATH_GPT_find_speed_of_first_train_l28_2810


namespace NUMINAMATH_GPT_males_band_not_orchestra_l28_2801

/-- Define conditions as constants -/
def total_females_band := 150
def total_males_band := 130
def total_females_orchestra := 140
def total_males_orchestra := 160
def females_both := 90
def males_both := 80
def total_students_either := 310

/-- The number of males in the band who are NOT in the orchestra -/
theorem males_band_not_orchestra : total_males_band - males_both = 50 := by
  sorry

end NUMINAMATH_GPT_males_band_not_orchestra_l28_2801


namespace NUMINAMATH_GPT_find_num_20_paise_coins_l28_2842

def num_20_paise_coins (x y : ℕ) : Prop :=
  x + y = 334 ∧ 20 * x + 25 * y = 7100

theorem find_num_20_paise_coins (x y : ℕ) (h : num_20_paise_coins x y) : x = 250 :=
by
  sorry

end NUMINAMATH_GPT_find_num_20_paise_coins_l28_2842


namespace NUMINAMATH_GPT_tangent_line_at_one_l28_2863

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem tangent_line_at_one : ∀ x y, (x = 1 ∧ y = 0) → (x - y - 1 = 0) :=
by 
  intro x y h
  sorry

end NUMINAMATH_GPT_tangent_line_at_one_l28_2863


namespace NUMINAMATH_GPT_hyperbola_with_foci_condition_l28_2821

theorem hyperbola_with_foci_condition (k : ℝ) :
  ( ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 → ∀ x y : ℝ, (x^2 / (k + 3)) + (y^2 / (k + 2)) = 1 ∧ (k + 3 > 0 ∧ k + 2 < 0) ) ↔ (-3 < k ∧ k < -2) :=
sorry

end NUMINAMATH_GPT_hyperbola_with_foci_condition_l28_2821


namespace NUMINAMATH_GPT_rhombus_longer_diagonal_length_l28_2873

theorem rhombus_longer_diagonal_length (side_length : ℕ) (shorter_diagonal : ℕ) 
  (h_side : side_length = 65) (h_short_diag : shorter_diagonal = 56) : 
  ∃ longer_diagonal : ℕ, longer_diagonal = 118 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_longer_diagonal_length_l28_2873


namespace NUMINAMATH_GPT_average_score_for_entire_class_l28_2835

theorem average_score_for_entire_class (n x y : ℕ) (a b : ℝ) (hn : n = 100) (hx : x = 70) (hy : y = 30) (ha : a = 0.65) (hb : b = 0.95) :
    ((x * a + y * b) / n) = 0.74 := by
  sorry

end NUMINAMATH_GPT_average_score_for_entire_class_l28_2835


namespace NUMINAMATH_GPT_ashley_champagne_bottles_l28_2847

theorem ashley_champagne_bottles (guests : ℕ) (glasses_per_guest : ℕ) (servings_per_bottle : ℕ) 
  (h1 : guests = 120) (h2 : glasses_per_guest = 2) (h3 : servings_per_bottle = 6) : 
  (guests * glasses_per_guest) / servings_per_bottle = 40 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_ashley_champagne_bottles_l28_2847


namespace NUMINAMATH_GPT_evaluate_expression_l28_2871

theorem evaluate_expression : (20 + 22) / 2 = 21 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l28_2871


namespace NUMINAMATH_GPT_sqrt_sqrt_16_l28_2845

theorem sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := sorry

end NUMINAMATH_GPT_sqrt_sqrt_16_l28_2845


namespace NUMINAMATH_GPT_remaining_fish_l28_2866

theorem remaining_fish (initial_fish : ℝ) (moved_fish : ℝ) (remaining_fish : ℝ) : initial_fish = 212.0 → moved_fish = 68.0 → remaining_fish = 144.0 → initial_fish - moved_fish = remaining_fish := by sorry

end NUMINAMATH_GPT_remaining_fish_l28_2866


namespace NUMINAMATH_GPT_problem_p_3_l28_2896

theorem problem_p_3 (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) (hn : n = (2^(2*p) - 1) / 3) : n ∣ 2^n - 2 := by
  sorry

end NUMINAMATH_GPT_problem_p_3_l28_2896


namespace NUMINAMATH_GPT_triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l28_2843

noncomputable def sum_sine_3A_3B_3C (A B C : ℝ) : ℝ :=
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C)

theorem triangle_inequality_sine_three_times {A B C : ℝ} (h : A + B + C = Real.pi) (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) : 
  (-2 : ℝ) ≤ sum_sine_3A_3B_3C A B C ∧ sum_sine_3A_3B_3C A B C ≤ (3 * Real.sqrt 3 / 2) :=
by
  sorry

theorem equality_sine_three_times_lower_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = 0) (h2: B = Real.pi / 2) (h3: C = Real.pi / 2) :
  sum_sine_3A_3B_3C A B C = -2 :=
by
  sorry

theorem equality_sine_three_times_upper_bound {A B C : ℝ} (h : A + B + C = Real.pi) (h1: A = Real.pi / 3) (h2: B = Real.pi / 3) (h3: C = Real.pi / 3) :
  sum_sine_3A_3B_3C A B C = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_sine_three_times_equality_sine_three_times_lower_bound_equality_sine_three_times_upper_bound_l28_2843


namespace NUMINAMATH_GPT_number_of_associates_l28_2864

theorem number_of_associates
  (num_managers : ℕ) 
  (avg_salary_managers : ℝ) 
  (avg_salary_associates : ℝ) 
  (avg_salary_company : ℝ)
  (total_employees : ℕ := num_managers + A) -- Adding a placeholder A for the associates
  (total_salary_company : ℝ := (num_managers * avg_salary_managers) + (A * avg_salary_associates)) 
  (average_calculation : avg_salary_company = total_salary_company / total_employees) :
  ∃ A : ℕ, A = 75 :=
by
  let A : ℕ := 75
  sorry

end NUMINAMATH_GPT_number_of_associates_l28_2864


namespace NUMINAMATH_GPT_least_xy_value_l28_2800

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end NUMINAMATH_GPT_least_xy_value_l28_2800


namespace NUMINAMATH_GPT_contrapositive_necessary_condition_l28_2883

theorem contrapositive_necessary_condition {p q : Prop} (h : p → q) : ¬p → ¬q :=
  by sorry

end NUMINAMATH_GPT_contrapositive_necessary_condition_l28_2883


namespace NUMINAMATH_GPT_simple_interest_time_l28_2832

-- Definitions based on given conditions
def SI : ℝ := 640           -- Simple interest
def P : ℝ := 4000           -- Principal
def R : ℝ := 8              -- Rate
def T : ℝ := 2              -- Time in years (correct answer to be proved)

-- Theorem statement
theorem simple_interest_time :
  SI = (P * R * T) / 100 := 
by 
  sorry

end NUMINAMATH_GPT_simple_interest_time_l28_2832


namespace NUMINAMATH_GPT_quadratic_roots_square_diff_l28_2880

theorem quadratic_roots_square_diff (α β : ℝ) (h : α ≠ β)
    (hα : α^2 - 3 * α + 2 = 0) (hβ : β^2 - 3 * β + 2 = 0) :
    (α - β)^2 = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_square_diff_l28_2880


namespace NUMINAMATH_GPT_bcm_hens_count_l28_2867

-- Propositions representing the given conditions
def total_chickens : ℕ := 100
def bcm_ratio : ℝ := 0.20
def bcm_hens_ratio : ℝ := 0.80

-- Theorem statement: proving the number of BCM hens
theorem bcm_hens_count : (total_chickens * bcm_ratio * bcm_hens_ratio = 16) := by
  sorry

end NUMINAMATH_GPT_bcm_hens_count_l28_2867


namespace NUMINAMATH_GPT_value_of_f_m_minus_1_pos_l28_2815

variable (a m : ℝ)
variable (f : ℝ → ℝ)
variable (a_pos : a > 0)
variable (fm_neg : f m < 0)
variable (f_def : ∀ x, f x = x^2 - x + a)

theorem value_of_f_m_minus_1_pos : f (m - 1) > 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_m_minus_1_pos_l28_2815


namespace NUMINAMATH_GPT_find_monotonic_bijections_l28_2887

variable {f : ℝ → ℝ}

-- Define the properties of the function f
def bijective (f : ℝ → ℝ) : Prop :=
  Function.Bijective f

def condition (f : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, f t + f (f t) = 2 * t

theorem find_monotonic_bijections (f : ℝ → ℝ) (hf_bij : bijective f) (hf_cond : condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
sorry

end NUMINAMATH_GPT_find_monotonic_bijections_l28_2887


namespace NUMINAMATH_GPT_smallest_three_digit_n_l28_2807

theorem smallest_three_digit_n (n : ℕ) (h_pos : 100 ≤ n) (h_below : n ≤ 999) 
  (cond1 : n % 9 = 2) (cond2 : n % 6 = 4) : n = 118 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_three_digit_n_l28_2807


namespace NUMINAMATH_GPT_accurate_to_hundreds_place_l28_2805

def rounded_number : ℝ := 8.80 * 10^4

theorem accurate_to_hundreds_place
  (n : ℝ) (h : n = rounded_number) : 
  exists (d : ℤ), n = d * 100 ∧ |round n - n| < 50 :=
sorry

end NUMINAMATH_GPT_accurate_to_hundreds_place_l28_2805


namespace NUMINAMATH_GPT_ratio_cher_to_gab_l28_2895

-- Definitions based on conditions
def sammy_score : ℕ := 20
def gab_score : ℕ := 2 * sammy_score
def opponent_score : ℕ := 85
def total_points : ℕ := opponent_score + 55
def cher_score : ℕ := total_points - (sammy_score + gab_score)

-- Theorem to prove the ratio of Cher's score to Gab's score
theorem ratio_cher_to_gab : cher_score / gab_score = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_cher_to_gab_l28_2895


namespace NUMINAMATH_GPT_problem_statement_l28_2827

variable (a b : ℝ) (f : ℝ → ℝ)
variable (h1 : ∀ x > 0, f x = Real.log x / Real.log 3)
variable (h2 : b = 9 * a)

theorem problem_statement : f a - f b = -2 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l28_2827


namespace NUMINAMATH_GPT_total_beads_in_necklace_l28_2828

noncomputable def amethyst_beads : ℕ := 7
noncomputable def amber_beads : ℕ := 2 * amethyst_beads
noncomputable def turquoise_beads : ℕ := 19
noncomputable def total_beads : ℕ := amethyst_beads + amber_beads + turquoise_beads

theorem total_beads_in_necklace : total_beads = 40 := by
  sorry

end NUMINAMATH_GPT_total_beads_in_necklace_l28_2828


namespace NUMINAMATH_GPT_youtube_likes_l28_2802

theorem youtube_likes (L D : ℕ) 
  (h1 : D = (1 / 2 : ℝ) * L + 100)
  (h2 : D + 1000 = 2600) : 
  L = 3000 := 
by
  sorry

end NUMINAMATH_GPT_youtube_likes_l28_2802


namespace NUMINAMATH_GPT_total_food_for_guinea_pigs_l28_2804

-- Definitions of the food consumption for each guinea pig
def first_guinea_pig_food : ℕ := 2
def second_guinea_pig_food : ℕ := 2 * first_guinea_pig_food
def third_guinea_pig_food : ℕ := second_guinea_pig_food + 3

-- Statement to prove the total food required
theorem total_food_for_guinea_pigs : 
  first_guinea_pig_food + second_guinea_pig_food + third_guinea_pig_food = 13 := by
  sorry

end NUMINAMATH_GPT_total_food_for_guinea_pigs_l28_2804


namespace NUMINAMATH_GPT_P_is_sufficient_but_not_necessary_for_Q_l28_2831

def P (x : ℝ) : Prop := (2 * x - 3)^2 < 1
def Q (x : ℝ) : Prop := x * (x - 3) < 0

theorem P_is_sufficient_but_not_necessary_for_Q : 
  (∀ x, P x → Q x) ∧ (∃ x, Q x ∧ ¬ P x) :=
by
  sorry

end NUMINAMATH_GPT_P_is_sufficient_but_not_necessary_for_Q_l28_2831


namespace NUMINAMATH_GPT_tangent_parabola_line_l28_2875

theorem tangent_parabola_line (a x₀ y₀ : ℝ) 
  (h_line : x₀ - y₀ - 1 = 0)
  (h_parabola : y₀ = a * x₀^2)
  (h_tangent_slope : 2 * a * x₀ = 1) : 
  a = 1 / 4 :=
sorry

end NUMINAMATH_GPT_tangent_parabola_line_l28_2875


namespace NUMINAMATH_GPT_cycle_selling_price_l28_2857

theorem cycle_selling_price (initial_price : ℝ)
  (first_discount_percent : ℝ) (second_discount_percent : ℝ) (third_discount_percent : ℝ)
  (first_discounted_price : ℝ) (second_discounted_price : ℝ) :
  initial_price = 3600 →
  first_discount_percent = 15 →
  second_discount_percent = 10 →
  third_discount_percent = 5 →
  first_discounted_price = initial_price * (1 - first_discount_percent / 100) →
  second_discounted_price = first_discounted_price * (1 - second_discount_percent / 100) →
  final_price = second_discounted_price * (1 - third_discount_percent / 100) →
  final_price = 2616.30 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cycle_selling_price_l28_2857


namespace NUMINAMATH_GPT_optionC_is_correct_l28_2888

def KalobsWindowLength : ℕ := 50
def KalobsWindowWidth : ℕ := 80
def KalobsWindowArea : ℕ := KalobsWindowLength * KalobsWindowWidth

def DoubleKalobsWindowArea : ℕ := 2 * KalobsWindowArea

def optionC_Length : ℕ := 50
def optionC_Width : ℕ := 160
def optionC_Area : ℕ := optionC_Length * optionC_Width

theorem optionC_is_correct : optionC_Area = DoubleKalobsWindowArea := by
  sorry

end NUMINAMATH_GPT_optionC_is_correct_l28_2888


namespace NUMINAMATH_GPT_problem_statement_l28_2886

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ ⦃x y⦄, x > 4 → y > x → f y < f x)
                          (h2 : ∀ x, f (4 + x) = f (4 - x)) : f 3 > f 6 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l28_2886


namespace NUMINAMATH_GPT_find_particular_number_l28_2833

theorem find_particular_number (x : ℝ) (h : 4 * x * 25 = 812) : x = 8.12 :=
by sorry

end NUMINAMATH_GPT_find_particular_number_l28_2833


namespace NUMINAMATH_GPT_purely_imaginary_x_value_l28_2858

theorem purely_imaginary_x_value (x : ℝ) (h1 : x^2 - 1 = 0) (h2 : x + 1 ≠ 0) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_x_value_l28_2858


namespace NUMINAMATH_GPT_monotonic_intervals_inequality_condition_l28_2840

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x

theorem monotonic_intervals (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x < y → f x m < f y m) ∧
  (m > 0 → (∀ x > 0, x < 1/m → ∀ y > x, y < 1/m → f x m < f y m) ∧ (∀ x ≥ 1/m, ∀ y > x, f x m > f y m)) :=
sorry

theorem inequality_condition (m : ℝ) (h : ∀ x ≥ 1, f x m ≤ (m - 1) / x - 2 * m + 1) :
  m ≥ 1/2 :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_inequality_condition_l28_2840


namespace NUMINAMATH_GPT_quartic_polynomial_eval_l28_2817

noncomputable def f (x : ℝ) : ℝ := sorry  -- f is a monic quartic polynomial

theorem quartic_polynomial_eval (h_monic: true)
    (h1 : f (-1) = -1)
    (h2 : f 2 = -4)
    (h3 : f (-3) = -9)
    (h4 : f 4 = -16) : f 1 = 23 :=
sorry

end NUMINAMATH_GPT_quartic_polynomial_eval_l28_2817


namespace NUMINAMATH_GPT_deepak_age_is_21_l28_2856

noncomputable def DeepakCurrentAge (x : ℕ) : Prop :=
  let Rahul := 4 * x
  let Deepak := 3 * x
  let Karan := 5 * x
  Rahul + 6 = 34 ∧
  (Rahul + 6) / 7 = (Deepak + 6) / 5 ∧ (Rahul + 6) / 7 = (Karan + 6) / 9 → 
  Deepak = 21

theorem deepak_age_is_21 : ∃ x : ℕ, DeepakCurrentAge x :=
by
  use 7
  sorry

end NUMINAMATH_GPT_deepak_age_is_21_l28_2856


namespace NUMINAMATH_GPT_probability_one_of_two_sheep_selected_l28_2803

theorem probability_one_of_two_sheep_selected :
  let sheep := ["Happy", "Pretty", "Lazy", "Warm", "Boiling"]
  let total_ways := Nat.choose 5 2
  let favorable_ways := (Nat.choose 2 1) * (Nat.choose 3 1)
  let probability := favorable_ways / total_ways
  probability = 3 / 5 :=
by
  let sheep := ["Happy", "Pretty", "Lazy", "Warm", "Boiling"]
  let total_ways := Nat.choose 5 2
  let favorable_ways := (Nat.choose 2 1) * (Nat.choose 3 1)
  let probability := favorable_ways / total_ways
  sorry

end NUMINAMATH_GPT_probability_one_of_two_sheep_selected_l28_2803


namespace NUMINAMATH_GPT_back_wheel_revolutions_calculation_l28_2816

def front_wheel_radius : ℝ := 3
def back_wheel_radius : ℝ := 0.5
def gear_ratio : ℝ := 2
def front_wheel_revolutions : ℕ := 50

noncomputable def back_wheel_revolutions (front_wheel_radius back_wheel_radius gear_ratio : ℝ) (front_wheel_revolutions : ℕ) : ℝ :=
  let front_circumference := 2 * Real.pi * front_wheel_radius
  let distance_traveled := front_circumference * front_wheel_revolutions
  let back_circumference := 2 * Real.pi * back_wheel_radius
  distance_traveled / back_circumference * gear_ratio

theorem back_wheel_revolutions_calculation :
  back_wheel_revolutions front_wheel_radius back_wheel_radius gear_ratio front_wheel_revolutions = 600 :=
sorry

end NUMINAMATH_GPT_back_wheel_revolutions_calculation_l28_2816


namespace NUMINAMATH_GPT_seating_arrangements_l28_2877

theorem seating_arrangements :
  let total_arrangements := Nat.factorial 8
  let jwp_together := (Nat.factorial 6) * (Nat.factorial 3)
  total_arrangements - jwp_together = 36000 := by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l28_2877


namespace NUMINAMATH_GPT_solve_for_x_l28_2820

theorem solve_for_x (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) : x = 37 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l28_2820


namespace NUMINAMATH_GPT_f_2016_plus_f_2015_l28_2829

theorem f_2016_plus_f_2015 (f : ℝ → ℝ) 
  (H1 : ∀ x, f (-x) = -f x) -- Odd function property
  (H2 : ∀ x, f (x + 1) = f (-x + 1)) -- Even function property for f(x+1)
  (H3 : f 1 = 1) : 
  f 2016 + f 2015 = -1 :=
sorry

end NUMINAMATH_GPT_f_2016_plus_f_2015_l28_2829


namespace NUMINAMATH_GPT_students_count_l28_2819

theorem students_count :
  ∃ S : ℕ, (S + 4) % 9 = 0 ∧ S = 23 :=
by
  sorry

end NUMINAMATH_GPT_students_count_l28_2819


namespace NUMINAMATH_GPT_terminal_side_alpha_minus_beta_nonneg_x_axis_l28_2892

theorem terminal_side_alpha_minus_beta_nonneg_x_axis
  (α β : ℝ) (k : ℤ) (h : α = k * 360 + β) : 
  (∃ m : ℤ, α - β = m * 360) := 
sorry

end NUMINAMATH_GPT_terminal_side_alpha_minus_beta_nonneg_x_axis_l28_2892


namespace NUMINAMATH_GPT_tangent_line_through_M_to_circle_l28_2894

noncomputable def M : ℝ × ℝ := (2, -1)
noncomputable def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

theorem tangent_line_through_M_to_circle :
  ∀ {x y : ℝ}, circle_eq x y → M = (2, -1) → 2*x - y - 5 = 0 :=
sorry

end NUMINAMATH_GPT_tangent_line_through_M_to_circle_l28_2894


namespace NUMINAMATH_GPT_find_missing_digit_l28_2899

theorem find_missing_digit 
  (x : Nat) 
  (h : 16 + x ≡ 0 [MOD 9]) : 
  x = 2 :=
sorry

end NUMINAMATH_GPT_find_missing_digit_l28_2899


namespace NUMINAMATH_GPT_remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l28_2823

theorem remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12 :
  (7 * 11 ^ 24 + 2 ^ 24) % 12 = 11 := by
sorry

end NUMINAMATH_GPT_remainder_7_mul_11_pow_24_plus_2_pow_24_mod_12_l28_2823


namespace NUMINAMATH_GPT_price_of_shoes_on_Monday_l28_2874

noncomputable def priceOnThursday : ℝ := 50

noncomputable def increasedPriceOnFriday : ℝ := priceOnThursday * 1.2

noncomputable def discountedPriceOnMonday : ℝ := increasedPriceOnFriday * 0.85

noncomputable def finalPriceOnMonday : ℝ := discountedPriceOnMonday * 1.05

theorem price_of_shoes_on_Monday :
  finalPriceOnMonday = 53.55 :=
by
  sorry

end NUMINAMATH_GPT_price_of_shoes_on_Monday_l28_2874


namespace NUMINAMATH_GPT_least_positive_integer_added_to_575_multiple_4_l28_2824

theorem least_positive_integer_added_to_575_multiple_4 :
  ∃ n : ℕ, n > 0 ∧ (575 + n) % 4 = 0 ∧ 
           ∀ m : ℕ, (m > 0 ∧ (575 + m) % 4 = 0) → n ≤ m := by
  sorry

end NUMINAMATH_GPT_least_positive_integer_added_to_575_multiple_4_l28_2824


namespace NUMINAMATH_GPT_find_a_l28_2854

theorem find_a (a r : ℝ) (h1 : a * r = 24) (h2 : a * r^4 = 3) : a = 48 :=
sorry

end NUMINAMATH_GPT_find_a_l28_2854


namespace NUMINAMATH_GPT_find_g2_l28_2889

variable (g : ℝ → ℝ)

theorem find_g2 (h : ∀ x : ℝ, g (3 * x - 7) = 5 * x + 11) : g 2 = 26 := by
  sorry

end NUMINAMATH_GPT_find_g2_l28_2889


namespace NUMINAMATH_GPT_annalise_total_cost_correct_l28_2868

-- Define the constants from the problem
def boxes : ℕ := 25
def packs_per_box : ℕ := 18
def tissues_per_pack : ℕ := 150
def tissue_price : ℝ := 0.06
def discount_per_box : ℝ := 0.10
def volume_discount : ℝ := 0.08
def tax_rate : ℝ := 0.05

-- Calculate the total number of tissues
def total_tissues : ℕ := boxes * packs_per_box * tissues_per_pack

-- Calculate the total cost without any discounts
def initial_cost : ℝ := total_tissues * tissue_price

-- Apply the 10% discount on the price of the total packs in each box purchased
def cost_after_box_discount : ℝ := initial_cost * (1 - discount_per_box)

-- Apply the 8% volume discount for buying 10 or more boxes
def cost_after_volume_discount : ℝ := cost_after_box_discount * (1 - volume_discount)

-- Apply the 5% tax on the final price after all discounts
def final_cost : ℝ := cost_after_volume_discount * (1 + tax_rate)

-- Define the expected final cost
def expected_final_cost : ℝ := 3521.07

-- Proof statement
theorem annalise_total_cost_correct : final_cost = expected_final_cost := by
  -- Sorry is used as placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_annalise_total_cost_correct_l28_2868


namespace NUMINAMATH_GPT_least_number_to_add_l28_2809

theorem least_number_to_add (x : ℕ) : (1056 + x) % 28 = 0 ↔ x = 4 :=
by sorry

end NUMINAMATH_GPT_least_number_to_add_l28_2809


namespace NUMINAMATH_GPT_Z_is_1_5_decades_younger_l28_2825

theorem Z_is_1_5_decades_younger (X Y Z : ℝ) (h : X + Y = Y + Z + 15) : (X - Z) / 10 = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_Z_is_1_5_decades_younger_l28_2825


namespace NUMINAMATH_GPT_length_of_shop_proof_l28_2851

-- Given conditions
def monthly_rent : ℝ := 1440
def width : ℝ := 20
def annual_rent_per_sqft : ℝ := 48

-- Correct answer to be proved
def length_of_shop : ℝ := 18

-- The following statement is the proof problem in Lean 4
theorem length_of_shop_proof (h1 : monthly_rent = 1440) 
                            (h2 : width = 20) 
                            (h3 : annual_rent_per_sqft = 48) : 
  length_of_shop = 18 := 
  sorry

end NUMINAMATH_GPT_length_of_shop_proof_l28_2851


namespace NUMINAMATH_GPT_fergus_entry_exit_l28_2861

theorem fergus_entry_exit (n : ℕ) (hn : n = 8) : 
  n * (n - 1) = 56 := 
by
  sorry

end NUMINAMATH_GPT_fergus_entry_exit_l28_2861


namespace NUMINAMATH_GPT_additional_money_needed_l28_2830

theorem additional_money_needed :
  let total_budget := 500
  let budget_dresses := 300
  let budget_shoes := 150
  let budget_accessories := 50
  let extra_fraction := 2 / 5
  let discount_rate := 0.15
  let total_without_discount := 
    budget_dresses * (1 + extra_fraction) +
    budget_shoes * (1 + extra_fraction) +
    budget_accessories * (1 + extra_fraction)
  let discounted_total := total_without_discount * (1 - discount_rate)
  discounted_total > total_budget :=
sorry

end NUMINAMATH_GPT_additional_money_needed_l28_2830


namespace NUMINAMATH_GPT_cricket_team_matches_in_august_l28_2852

noncomputable def cricket_matches_played_in_august (M W W_new: ℕ) : Prop :=
  W = 26 * M / 100 ∧
  W_new = 52 * (M + 65) / 100 ∧ 
  W_new = W + 65

theorem cricket_team_matches_in_august (M W W_new: ℕ) : cricket_matches_played_in_august M W W_new → M = 120 := 
by
  sorry

end NUMINAMATH_GPT_cricket_team_matches_in_august_l28_2852


namespace NUMINAMATH_GPT_total_students_l28_2859

theorem total_students (girls boys : ℕ) (h1 : girls = 300) (h2 : boys = 8 * (girls / 5)) : girls + boys = 780 := by
  sorry

end NUMINAMATH_GPT_total_students_l28_2859


namespace NUMINAMATH_GPT_sufficiency_of_p_for_q_not_necessity_of_p_for_q_l28_2853

noncomputable def p (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m
noncomputable def q (m : ℝ) := ∀ x : ℝ, (- (5 - 2 * m)) ^ x < 0

theorem sufficiency_of_p_for_q : ∀ m : ℝ, (m < 1 → m < 2) :=
by sorry

theorem not_necessity_of_p_for_q : ∀ m : ℝ, ¬ (m < 2 → m < 1) :=
by sorry

end NUMINAMATH_GPT_sufficiency_of_p_for_q_not_necessity_of_p_for_q_l28_2853


namespace NUMINAMATH_GPT_fraction_expression_l28_2839

theorem fraction_expression :
  (1 / 4 - 1 / 6) / (1 / 3 + 1 / 2) = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_expression_l28_2839


namespace NUMINAMATH_GPT_larger_page_sum_137_l28_2848

theorem larger_page_sum_137 (x y : ℕ) (h1 : x + y = 137) (h2 : y = x + 1) : y = 69 :=
sorry

end NUMINAMATH_GPT_larger_page_sum_137_l28_2848


namespace NUMINAMATH_GPT_problem1_range_of_x_problem2_value_of_a_l28_2812

open Set

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |x + 3| + |x - a|

-- Problem 1
theorem problem1_range_of_x (a : ℝ) (h : a = 4) (h_eq : ∀ x : ℝ, f x a = 7 ↔ x ∈ Icc (-3 : ℝ) 4) :
  ∀ x : ℝ, f x 4 = 7 ↔ x ∈ Icc (-3 : ℝ) 4 := by
  sorry

-- Problem 2
theorem problem2_value_of_a (h₁ : ∀ x : ℝ, x ∈ {x : ℝ | f x 4 ≥ 6} ↔ x ≤ -4 ∨ x ≥ 2) :
  f x a ≥ 6 ↔  x ≤ -4 ∨ x ≥ 2 :=
  by
  sorry

end NUMINAMATH_GPT_problem1_range_of_x_problem2_value_of_a_l28_2812


namespace NUMINAMATH_GPT_appropriate_investigation_method_l28_2806

theorem appropriate_investigation_method
  (volume_of_investigation_large : Prop)
  (no_need_for_comprehensive_investigation : Prop) :
  (∃ (method : String), method = "sampling investigation") :=
by
  sorry

end NUMINAMATH_GPT_appropriate_investigation_method_l28_2806
