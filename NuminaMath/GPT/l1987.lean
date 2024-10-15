import Mathlib

namespace NUMINAMATH_GPT_minimum_positive_period_minimum_value_l1987_198758

noncomputable def f (x : Real) : Real :=
  Real.sin (x / 5) - Real.cos (x / 5)

theorem minimum_positive_period (T : Real) : (∀ x, f (x + T) = f x) ∧ T > 0 → T = 10 * Real.pi :=
  sorry

theorem minimum_value : ∃ x, f x = -Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_minimum_positive_period_minimum_value_l1987_198758


namespace NUMINAMATH_GPT_count_valid_outfits_l1987_198741

/-
Problem:
I have 5 shirts, 3 pairs of pants, and 5 hats. The pants come in red, green, and blue. 
The shirts and hats come in those colors, plus orange and purple. 
I refuse to wear an outfit where the shirt and the hat are the same color. 
How many choices for outfits, consisting of one shirt, one hat, and one pair of pants, do I have?
-/

def num_shirts := 5
def num_pants := 3
def num_hats := 5
def valid_outfits := 66

-- The set of colors available for shirts and hats
inductive color
| red | green | blue | orange | purple

-- Conditions and properties translated into Lean
def pants_colors : List color := [color.red, color.green, color.blue]
def shirt_hat_colors : List color := [color.red, color.green, color.blue, color.orange, color.purple]

theorem count_valid_outfits (h1 : num_shirts = 5) 
                            (h2 : num_pants = 3) 
                            (h3 : num_hats = 5) 
                            (h4 : ∀ (s : color), s ∈ shirt_hat_colors) 
                            (h5 : ∀ (p : color), p ∈ pants_colors) 
                            (h6 : ∀ (s h : color), s ≠ h) :
  valid_outfits = 66 :=
by
  sorry

end NUMINAMATH_GPT_count_valid_outfits_l1987_198741


namespace NUMINAMATH_GPT_rectangle_relationships_l1987_198732

theorem rectangle_relationships (x y S : ℝ) (h1 : 2 * x + 2 * y = 10) (h2 : S = x * y) :
  y = 5 - x ∧ S = 5 * x - x ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_relationships_l1987_198732


namespace NUMINAMATH_GPT_kernels_needed_for_movie_night_l1987_198716

structure PopcornPreferences where
  caramel_popcorn: ℝ
  butter_popcorn: ℝ
  cheese_popcorn: ℝ
  kettle_corn_popcorn: ℝ

noncomputable def total_kernels_needed (preferences: PopcornPreferences) : ℝ :=
  (preferences.caramel_popcorn / 6) * 3 +
  (preferences.butter_popcorn / 4) * 2 +
  (preferences.cheese_popcorn / 8) * 4 +
  (preferences.kettle_corn_popcorn / 3) * 1

theorem kernels_needed_for_movie_night :
  let preferences := PopcornPreferences.mk 3 4 6 3
  total_kernels_needed preferences = 7.5 :=
sorry

end NUMINAMATH_GPT_kernels_needed_for_movie_night_l1987_198716


namespace NUMINAMATH_GPT_remainder_when_x_plus_4uy_div_y_l1987_198727

theorem remainder_when_x_plus_4uy_div_y (x y u v : ℕ) (h₀: x = u * y + v) (h₁: 0 ≤ v) (h₂: v < y) : 
  ((x + 4 * u * y) % y) = v := 
by 
  sorry

end NUMINAMATH_GPT_remainder_when_x_plus_4uy_div_y_l1987_198727


namespace NUMINAMATH_GPT_leftover_value_is_5_30_l1987_198738

variable (q_per_roll d_per_roll : ℕ)
variable (j_quarters j_dimes l_quarters l_dimes : ℕ)
variable (value_per_quarter value_per_dime : ℝ)

def total_leftover_value (q_per_roll d_per_roll : ℕ) 
  (j_quarters l_quarters j_dimes l_dimes : ℕ)
  (value_per_quarter value_per_dime : ℝ) : ℝ :=
  let total_quarters := j_quarters + l_quarters
  let total_dimes := j_dimes + l_dimes
  let leftover_quarters := total_quarters % q_per_roll
  let leftover_dimes := total_dimes % d_per_roll
  (leftover_quarters * value_per_quarter) + (leftover_dimes * value_per_dime)

theorem leftover_value_is_5_30 :
  total_leftover_value 45 55 95 140 173 285 0.25 0.10 = 5.3 := 
by
  sorry

end NUMINAMATH_GPT_leftover_value_is_5_30_l1987_198738


namespace NUMINAMATH_GPT_range_of_m_l1987_198787

variable {x m : ℝ}

theorem range_of_m (h1 : x + 2 < 2 * m) (h2 : x - m < 0) (h3 : x < 2 * m - 2) : m ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1987_198787


namespace NUMINAMATH_GPT_stating_martha_painting_time_l1987_198756

/-- 
  Theorem stating the time it takes for Martha to paint the kitchen is 42 hours.
-/
theorem martha_painting_time :
  let width1 := 12
  let width2 := 16
  let height := 10
  let area_pair1 := 2 * width1 * height
  let area_pair2 := 2 * width2 * height
  let total_area := area_pair1 + area_pair2
  let coats := 3
  let total_paint_area := total_area * coats
  let painting_speed := 40
  let time_required := total_paint_area / painting_speed
  time_required = 42 := by
    -- Since we are asked not to provide the proof steps, we use sorry to skip the proof.
    sorry

end NUMINAMATH_GPT_stating_martha_painting_time_l1987_198756


namespace NUMINAMATH_GPT_tax_refund_l1987_198777

-- Definitions based on the problem conditions
def monthly_salary : ℕ := 9000
def treatment_cost : ℕ := 100000
def medication_cost : ℕ := 20000
def tax_rate : ℚ := 0.13

-- Annual salary calculation
def annual_salary := monthly_salary * 12

-- Total spending on treatment and medications
def total_spending := treatment_cost + medication_cost

-- Possible tax refund based on total spending
def possible_tax_refund := total_spending * tax_rate

-- Income tax paid on the annual salary
def income_tax_paid := annual_salary * tax_rate

-- Prove statement that the actual tax refund is equal to income tax paid
theorem tax_refund : income_tax_paid = 14040 := by
  sorry

end NUMINAMATH_GPT_tax_refund_l1987_198777


namespace NUMINAMATH_GPT_x_zero_necessary_but_not_sufficient_l1987_198784

-- Definitions based on conditions
def x_eq_zero (x : ℝ) := x = 0
def xsq_plus_ysq_eq_zero (x y : ℝ) := x^2 + y^2 = 0

-- Statement that x = 0 is a necessary but not sufficient condition for x^2 + y^2 = 0
theorem x_zero_necessary_but_not_sufficient (x y : ℝ) : (x = 0 ↔ x^2 + y^2 = 0) → False :=
by sorry

end NUMINAMATH_GPT_x_zero_necessary_but_not_sufficient_l1987_198784


namespace NUMINAMATH_GPT_eval_expression_l1987_198774

theorem eval_expression : 4^3 - 2 * 4^2 + 2 * 4 - 1 = 39 :=
by 
  -- Here we would write the proof, but according to the instructions we skip it with sorry.
  sorry

end NUMINAMATH_GPT_eval_expression_l1987_198774


namespace NUMINAMATH_GPT_find_C_l1987_198709

theorem find_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 340) : C = 40 :=
by sorry

end NUMINAMATH_GPT_find_C_l1987_198709


namespace NUMINAMATH_GPT_debby_pictures_l1987_198739

theorem debby_pictures : 
  let zoo_pics := 24
  let museum_pics := 12
  let pics_deleted := 14
  zoo_pics + museum_pics - pics_deleted = 22 := 
by
  sorry

end NUMINAMATH_GPT_debby_pictures_l1987_198739


namespace NUMINAMATH_GPT_total_spectators_l1987_198781

-- Definitions of conditions
def num_men : Nat := 7000
def num_children : Nat := 2500
def num_women := num_children / 5

-- Theorem stating the total number of spectators
theorem total_spectators : (num_men + num_children + num_women) = 10000 := by
  sorry

end NUMINAMATH_GPT_total_spectators_l1987_198781


namespace NUMINAMATH_GPT_distinct_real_numbers_a_l1987_198720

theorem distinct_real_numbers_a (a x y z : ℝ) (h_distinct: x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  (a = x + 1 / y ∧ a = y + 1 / z ∧ a = z + 1 / x) ↔ (a = 1 ∨ a = -1) :=
by sorry

end NUMINAMATH_GPT_distinct_real_numbers_a_l1987_198720


namespace NUMINAMATH_GPT_nine_a_eq_frac_minus_eighty_one_over_eleven_l1987_198788

theorem nine_a_eq_frac_minus_eighty_one_over_eleven (a b : ℚ) 
  (h1 : 8 * a + 3 * b = 0) 
  (h2 : a = b - 3) : 
  9 * a = -81 / 11 := 
sorry

end NUMINAMATH_GPT_nine_a_eq_frac_minus_eighty_one_over_eleven_l1987_198788


namespace NUMINAMATH_GPT_impossibility_of_4_level_ideal_interval_tan_l1987_198737

def has_ideal_interval (f : ℝ → ℝ) (D : Set ℝ) (k : ℝ) :=
  ∃ (a b : ℝ), a ≤ b ∧ Set.Icc a b ⊆ D ∧ (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y ∨ f y ≤ f x) ∧
  (Set.image f (Set.Icc a b) = Set.Icc (k * a) (k * b))

def option_D_incorrect : Prop :=
  ¬ has_ideal_interval (fun x => Real.tan x) (Set.Ioc (-(Real.pi / 2)) (Real.pi / 2)) 4

theorem impossibility_of_4_level_ideal_interval_tan :
  option_D_incorrect :=
sorry

end NUMINAMATH_GPT_impossibility_of_4_level_ideal_interval_tan_l1987_198737


namespace NUMINAMATH_GPT_A_inter_B_is_correct_l1987_198749

def set_A : Set ℤ := { x : ℤ | x^2 - x - 2 ≤ 0 }
def set_B : Set ℤ := { x : ℤ | True }

theorem A_inter_B_is_correct : set_A ∩ set_B = { -1, 0, 1, 2 } := by
  sorry

end NUMINAMATH_GPT_A_inter_B_is_correct_l1987_198749


namespace NUMINAMATH_GPT_alex_minus_sam_eq_negative_2_50_l1987_198735

def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.15
def packaging_fee : ℝ := 2.50

def alex_total (original_price tax_rate discount_rate : ℝ) : ℝ :=
  let price_with_tax := original_price * (1 + tax_rate)
  let final_price := price_with_tax * (1 - discount_rate)
  final_price

def sam_total (original_price tax_rate discount_rate packaging_fee : ℝ) : ℝ :=
  let price_with_discount := original_price * (1 - discount_rate)
  let price_with_tax := price_with_discount * (1 + tax_rate)
  let final_price := price_with_tax + packaging_fee
  final_price

theorem alex_minus_sam_eq_negative_2_50 :
  alex_total original_price tax_rate discount_rate - sam_total original_price tax_rate discount_rate packaging_fee = -2.50 := by
  sorry

end NUMINAMATH_GPT_alex_minus_sam_eq_negative_2_50_l1987_198735


namespace NUMINAMATH_GPT_range_of_k_for_quadratic_inequality_l1987_198710

theorem range_of_k_for_quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 + 2 * k * x - 1 < 0) ↔ (-1 < k ∧ k ≤ 0) :=
  sorry

end NUMINAMATH_GPT_range_of_k_for_quadratic_inequality_l1987_198710


namespace NUMINAMATH_GPT_quadratic_condition_l1987_198782

theorem quadratic_condition (m : ℤ) (x : ℝ) :
  (m + 1) * x^(m^2 + 1) - 2 * x - 5 = 0 ∧ m^2 + 1 = 2 ∧ m + 1 ≠ 0 ↔ m = 1 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_condition_l1987_198782


namespace NUMINAMATH_GPT_total_cost_l1987_198770

theorem total_cost (a b : ℕ) : 30 * a + 20 * b = 30 * a + 20 * b :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l1987_198770


namespace NUMINAMATH_GPT_total_hours_driven_l1987_198791

def total_distance : ℝ := 55.0
def distance_in_one_hour : ℝ := 1.527777778

theorem total_hours_driven : (total_distance / distance_in_one_hour) = 36.00 :=
by
  sorry

end NUMINAMATH_GPT_total_hours_driven_l1987_198791


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1987_198725

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.tan (ω * x + φ)
def P (f : ℝ → ℝ) : Prop := f 0 = 0
def Q (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem sufficient_but_not_necessary_condition (ω : ℝ) (φ : ℝ) (hω : ω > 0) :
  (P (f ω φ) → Q (f ω φ)) ∧ ¬(Q (f ω φ) → P (f ω φ)) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1987_198725


namespace NUMINAMATH_GPT_rationalize_fraction_l1987_198740

open BigOperators

theorem rationalize_fraction :
  (3 : ℝ) / (Real.sqrt 50 + 2) = (15 * Real.sqrt 2 - 6) / 46 :=
by
  -- Our proof intention will be inserted here.
  sorry

end NUMINAMATH_GPT_rationalize_fraction_l1987_198740


namespace NUMINAMATH_GPT_small_supermarkets_sample_count_l1987_198708

def large := 300
def medium := 600
def small := 2100
def sample_size := 100
def total := large + medium + small

theorem small_supermarkets_sample_count :
  small * (sample_size / total) = 70 := by
  sorry

end NUMINAMATH_GPT_small_supermarkets_sample_count_l1987_198708


namespace NUMINAMATH_GPT_rectangular_garden_width_l1987_198795

theorem rectangular_garden_width (w : ℕ) (h1 : ∃ l : ℕ, l = 3 * w) (h2 : w * (3 * w) = 507) : w = 13 := 
by 
  sorry

end NUMINAMATH_GPT_rectangular_garden_width_l1987_198795


namespace NUMINAMATH_GPT_five_x_ge_seven_y_iff_exists_abcd_l1987_198771

theorem five_x_ge_seven_y_iff_exists_abcd (x y : ℕ) :
  (5 * x ≥ 7 * y) ↔ ∃ (a b c d : ℕ), x = a + 2 * b + 3 * c + 7 * d ∧ y = b + 2 * c + 5 * d :=
by sorry

end NUMINAMATH_GPT_five_x_ge_seven_y_iff_exists_abcd_l1987_198771


namespace NUMINAMATH_GPT_angle_A_is_70_l1987_198717

-- Definitions of angles given as conditions in the problem
variables (BAD BAC ACB : ℝ)

def angle_BAD := 150
def angle_BAC := 80

-- The Lean 4 statement to prove the measure of angle ACB
theorem angle_A_is_70 (h1 : BAD = 150) (h2 : BAC = 80) : ACB = 70 :=
by {
  sorry
}

end NUMINAMATH_GPT_angle_A_is_70_l1987_198717


namespace NUMINAMATH_GPT_gcd_square_of_difference_l1987_198761

theorem gcd_square_of_difference (x y z : ℕ) (h : 1/x - 1/y = 1/z) :
  ∃ k : ℕ, (Nat.gcd (Nat.gcd x y) z) * (y - x) = k^2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_square_of_difference_l1987_198761


namespace NUMINAMATH_GPT_determinant_transformation_l1987_198713

theorem determinant_transformation 
  (p q r s : ℝ)
  (h : Matrix.det ![![p, q], ![r, s]] = 6) :
  Matrix.det ![![p, 9 * p + 4 * q], ![r, 9 * r + 4 * s]] = 24 := 
sorry

end NUMINAMATH_GPT_determinant_transformation_l1987_198713


namespace NUMINAMATH_GPT_simple_interest_correct_l1987_198772

-- Define the given conditions
def Principal : ℝ := 9005
def Rate : ℝ := 0.09
def Time : ℝ := 5

-- Define the simple interest function
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- State the theorem to prove the total interest earned
theorem simple_interest_correct : simple_interest Principal Rate Time = 4052.25 := sorry

end NUMINAMATH_GPT_simple_interest_correct_l1987_198772


namespace NUMINAMATH_GPT_upload_time_l1987_198711

theorem upload_time (file_size upload_speed : ℕ) (h_file_size : file_size = 160) (h_upload_speed : upload_speed = 8) : file_size / upload_speed = 20 :=
by
  sorry

end NUMINAMATH_GPT_upload_time_l1987_198711


namespace NUMINAMATH_GPT_problem_statement_l1987_198728

theorem problem_statement (x y : ℝ) (h : x * y < 0) : abs (x + y) < abs (x - y) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1987_198728


namespace NUMINAMATH_GPT_inequality_reciprocal_l1987_198759

theorem inequality_reciprocal (a b : ℝ) (hab : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_reciprocal_l1987_198759


namespace NUMINAMATH_GPT_grasshopper_jump_distance_l1987_198751

theorem grasshopper_jump_distance (frog_jump grasshopper_jump : ℝ) (h_frog : frog_jump = 40) (h_difference : frog_jump = grasshopper_jump + 15) : grasshopper_jump = 25 :=
by sorry

end NUMINAMATH_GPT_grasshopper_jump_distance_l1987_198751


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1987_198790

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 3 = 9)
  (h3 : a 5 = 5) :
  S 9 / S 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1987_198790


namespace NUMINAMATH_GPT_tissues_used_l1987_198797

-- Define the conditions
def box_tissues : ℕ := 160
def boxes_bought : ℕ := 3
def tissues_left : ℕ := 270

-- Define the theorem that needs to be proven
theorem tissues_used (total_tissues := boxes_bought * box_tissues) : total_tissues - tissues_left = 210 := by
  sorry

end NUMINAMATH_GPT_tissues_used_l1987_198797


namespace NUMINAMATH_GPT_minimum_cars_with_racing_stripes_l1987_198775

-- Definitions and conditions
variable (numberOfCars : ℕ) (withoutAC : ℕ) (maxWithACWithoutStripes : ℕ)

axiom total_number_of_cars : numberOfCars = 100
axiom cars_without_ac : withoutAC = 49
axiom max_ac_without_stripes : maxWithACWithoutStripes = 49    

-- Proposition
theorem minimum_cars_with_racing_stripes 
  (total_number_of_cars : numberOfCars = 100) 
  (cars_without_ac : withoutAC = 49)
  (max_ac_without_stripes : maxWithACWithoutStripes = 49) :
  ∃ (R : ℕ), R = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_cars_with_racing_stripes_l1987_198775


namespace NUMINAMATH_GPT_paint_walls_l1987_198754

theorem paint_walls (d h e : ℕ) : 
  ∃ (x : ℕ), (d * d * e = 2 * h * h * x) ↔ x = (d^2 * e) / (2 * h^2) := by
  sorry

end NUMINAMATH_GPT_paint_walls_l1987_198754


namespace NUMINAMATH_GPT_largest_sum_is_5_over_6_l1987_198752

def sum_1 := (1/3) + (1/7)
def sum_2 := (1/3) + (1/8)
def sum_3 := (1/3) + (1/2)
def sum_4 := (1/3) + (1/9)
def sum_5 := (1/3) + (1/4)

theorem largest_sum_is_5_over_6 : (sum_3 = 5/6) ∧ ((sum_3 > sum_1) ∧ (sum_3 > sum_2) ∧ (sum_3 > sum_4) ∧ (sum_3 > sum_5)) :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_is_5_over_6_l1987_198752


namespace NUMINAMATH_GPT_shelves_used_l1987_198779

theorem shelves_used (initial_books : ℕ) (sold_books : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (total_shelves : ℕ) :
  initial_books = 120 → sold_books = 39 → books_per_shelf = 9 → remaining_books = initial_books - sold_books → total_shelves = remaining_books / books_per_shelf → total_shelves = 9 :=
by
  intros h_initial_books h_sold_books h_books_per_shelf h_remaining_books h_total_shelves
  rw [h_initial_books, h_sold_books] at h_remaining_books
  rw [h_books_per_shelf, h_remaining_books] at h_total_shelves
  exact h_total_shelves

end NUMINAMATH_GPT_shelves_used_l1987_198779


namespace NUMINAMATH_GPT_ab_difference_l1987_198742

theorem ab_difference (a b : ℤ) (h1 : |a| = 5) (h2 : |b| = 3) (h3 : a + b > 0) : a - b = 2 ∨ a - b = 8 :=
sorry

end NUMINAMATH_GPT_ab_difference_l1987_198742


namespace NUMINAMATH_GPT_percentage_error_in_calculated_area_l1987_198721

theorem percentage_error_in_calculated_area :
  let initial_length_error := 0.03 -- 3%
  let initial_width_error := -0.02 -- 2% deficit
  let temperature_change := 15 -- °C
  let humidity_increase := 20 -- %
  let length_error_temp_increase := (temperature_change / 5) * 0.01
  let width_error_humidity_increase := (humidity_increase / 10) * 0.005
  let total_length_error := initial_length_error + length_error_temp_increase
  let total_width_error := initial_width_error + width_error_humidity_increase
  let total_percentage_error := total_length_error + total_width_error
  total_percentage_error * 100 = 3 -- 3%
:= by
  sorry

end NUMINAMATH_GPT_percentage_error_in_calculated_area_l1987_198721


namespace NUMINAMATH_GPT_find_original_number_l1987_198783

theorem find_original_number (a b c : ℕ) (h : 100 * a + 10 * b + c = 390) 
  (N : ℕ) (hN : N = 4326) : a = 3 ∧ b = 9 ∧ c = 0 :=
by 
  sorry

end NUMINAMATH_GPT_find_original_number_l1987_198783


namespace NUMINAMATH_GPT_find_treasure_island_l1987_198719

-- Define the types for the three islands
inductive Island : Type
| A | B | C

-- Define the possible inhabitants of island A
inductive Inhabitant : Type
| Knight  -- always tells the truth
| Liar    -- always lies
| Normal  -- might tell the truth or lie

-- Define the conditions
def no_treasure_on_A : Prop := ¬ ∃ (x : Island), x = Island.A ∧ (x = Island.A)
def normal_people_on_A_two_treasures : Prop := ∀ (h : Inhabitant), h = Inhabitant.Normal → (∃ (x y : Island), x ≠ y ∧ (x ≠ Island.A ∧ y ≠ Island.A))

-- The question to ask
def question_to_ask (h : Inhabitant) : Prop :=
  (h = Inhabitant.Knight) ↔ (∃ (x : Island), (x = Island.B) ∧ (¬ ∃ (y : Island), (y = Island.A) ∧ (y = Island.A)))

-- The theorem statement
theorem find_treasure_island (inh : Inhabitant) :
  no_treasure_on_A ∧ normal_people_on_A_two_treasures →
  (question_to_ask inh → (∃ (x : Island), x = Island.B)) ∧ (¬ question_to_ask inh → (∃ (x : Island), x = Island.C)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_treasure_island_l1987_198719


namespace NUMINAMATH_GPT_ryan_hours_on_english_l1987_198731

-- Given the conditions
def hours_on_chinese := 2
def hours_on_spanish := 4
def extra_hours_between_english_and_spanish := 3

-- We want to find out the hours on learning English
def hours_on_english := hours_on_spanish + extra_hours_between_english_and_spanish

-- Proof statement
theorem ryan_hours_on_english : hours_on_english = 7 := by
  -- This is where the proof would normally go.
  sorry

end NUMINAMATH_GPT_ryan_hours_on_english_l1987_198731


namespace NUMINAMATH_GPT_intersection_of_sets_l1987_198773

/-- Given the definitions of sets A and B, prove that A ∩ B equals {1, 2}. -/
theorem intersection_of_sets :
  let A := {x : ℝ | 0 < x}
  let B := {-2, -1, 1, 2}
  A ∩ B = {1, 2} :=
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1987_198773


namespace NUMINAMATH_GPT_min_value_of_expression_l1987_198789

open Real

theorem min_value_of_expression (x y z : ℝ) (h₁ : x + y + z = 1) (h₂ : x > 0) (h₃ : y > 0) (h₄ : z > 0) :
  (∃ a, (∀ x y z, a ≤ (1 / (x + y) + (x + y) / z)) ∧ a = 3) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1987_198789


namespace NUMINAMATH_GPT_chess_team_boys_l1987_198703

variable {B G : ℕ}

theorem chess_team_boys
    (h1 : B + G = 30)
    (h2 : 1/3 * G + B = 18) :
    B = 12 :=
by
  sorry

end NUMINAMATH_GPT_chess_team_boys_l1987_198703


namespace NUMINAMATH_GPT_positive_integer_solutions_of_inequality_system_l1987_198745

theorem positive_integer_solutions_of_inequality_system :
  {x : ℤ | 2 * (x - 1) < x + 1 ∧ 1 - (2 * x + 5) / 3 ≤ x ∧ x > 0} = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_of_inequality_system_l1987_198745


namespace NUMINAMATH_GPT_solution_set_for_rational_inequality_l1987_198755

theorem solution_set_for_rational_inequality (x : ℝ) :
  (x - 2) / (x - 1) > 0 ↔ x < 1 ∨ x > 2 := 
sorry

end NUMINAMATH_GPT_solution_set_for_rational_inequality_l1987_198755


namespace NUMINAMATH_GPT_odd_function_increasing_on_negative_interval_l1987_198786

theorem odd_function_increasing_on_negative_interval {f : ℝ → ℝ}
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 3 ≤ x → x ≤ 7 → 3 ≤ y → y ≤ 7 → x < y → f x < f y)
  (h_min_value : f 3 = 1) :
  (∀ x y, -7 ≤ x → x ≤ -3 → -7 ≤ y → y ≤ -3 → x < y → f x < f y) ∧ f (-3) = -1 := 
sorry

end NUMINAMATH_GPT_odd_function_increasing_on_negative_interval_l1987_198786


namespace NUMINAMATH_GPT_selection_ways_l1987_198757

-- The statement of the problem in Lean 4
theorem selection_ways :
  (Nat.choose 50 4) - (Nat.choose 47 4) = 
  (Nat.choose 3 1) * (Nat.choose 47 3) + 
  (Nat.choose 3 2) * (Nat.choose 47 2) + 
  (Nat.choose 3 3) * (Nat.choose 47 1) := 
sorry

end NUMINAMATH_GPT_selection_ways_l1987_198757


namespace NUMINAMATH_GPT_domestic_probability_short_haul_probability_long_haul_probability_l1987_198746

variable (P_internet_domestic P_snacks_domestic P_entertainment_domestic P_legroom_domestic : ℝ)
variable (P_internet_short_haul P_snacks_short_haul P_entertainment_short_haul P_legroom_short_haul : ℝ)
variable (P_internet_long_haul P_snacks_long_haul P_entertainment_long_haul P_legroom_long_haul : ℝ)

noncomputable def P_domestic :=
  P_internet_domestic * P_snacks_domestic * P_entertainment_domestic * P_legroom_domestic

theorem domestic_probability :
  P_domestic 0.40 0.60 0.70 0.50 = 0.084 := by
  sorry

noncomputable def P_short_haul :=
  P_internet_short_haul * P_snacks_short_haul * P_entertainment_short_haul * P_legroom_short_haul

theorem short_haul_probability :
  P_short_haul 0.50 0.75 0.55 0.60 = 0.12375 := by
  sorry

noncomputable def P_long_haul :=
  P_internet_long_haul * P_snacks_long_haul * P_entertainment_long_haul * P_legroom_long_haul

theorem long_haul_probability :
  P_long_haul 0.65 0.80 0.75 0.70 = 0.273 := by
  sorry

end NUMINAMATH_GPT_domestic_probability_short_haul_probability_long_haul_probability_l1987_198746


namespace NUMINAMATH_GPT_remainder_n_squared_plus_3n_plus_4_l1987_198764

theorem remainder_n_squared_plus_3n_plus_4 (n : ℤ) (h : n % 100 = 99) : (n^2 + 3*n + 4) % 100 = 2 := 
by sorry

end NUMINAMATH_GPT_remainder_n_squared_plus_3n_plus_4_l1987_198764


namespace NUMINAMATH_GPT_problem_solution_l1987_198793

theorem problem_solution : (324^2 - 300^2) / 24 = 624 :=
by 
  -- The proof will be inserted here.
  sorry

end NUMINAMATH_GPT_problem_solution_l1987_198793


namespace NUMINAMATH_GPT_probability_first_head_second_tail_l1987_198730

-- Conditions
def fair_coin := true
def prob_heads := 1 / 2
def prob_tails := 1 / 2
def independent_events (A B : Prop) := true

-- Statement
theorem probability_first_head_second_tail :
  fair_coin →
  independent_events (prob_heads = 1/2) (prob_tails = 1/2) →
  (prob_heads * prob_tails) = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_probability_first_head_second_tail_l1987_198730


namespace NUMINAMATH_GPT_range_k_l1987_198799

theorem range_k (k : ℝ) :
  (∀ x : ℝ, (3/8 - k*x - 2*k*x^2) ≥ 0) ↔ (-3 ≤ k ∧ k ≤ 0) :=
sorry

end NUMINAMATH_GPT_range_k_l1987_198799


namespace NUMINAMATH_GPT_percent_increase_in_pizza_area_l1987_198785

theorem percent_increase_in_pizza_area (r : ℝ) (h : 0 < r) :
  let r_large := 1.10 * r
  let A_medium := π * r^2
  let A_large := π * r_large^2
  let percent_increase := ((A_large - A_medium) / A_medium) * 100 
  percent_increase = 21 := 
by sorry

end NUMINAMATH_GPT_percent_increase_in_pizza_area_l1987_198785


namespace NUMINAMATH_GPT_carson_circles_theorem_l1987_198722

-- Define the dimensions of the warehouse
def warehouse_length : ℕ := 600
def warehouse_width : ℕ := 400

-- Define the perimeter calculation
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- Define the distance Carson walked
def distance_walked : ℕ := 16000

-- Define the number of circles Carson skipped
def circles_skipped : ℕ := 2

-- Define the expected number of circles Carson was supposed to circle
def expected_circles :=
  let actual_circles := distance_walked / (perimeter warehouse_length warehouse_width)
  actual_circles + circles_skipped

-- The theorem we want to prove
theorem carson_circles_theorem : expected_circles = 10 := by
  sorry

end NUMINAMATH_GPT_carson_circles_theorem_l1987_198722


namespace NUMINAMATH_GPT_divisor_between_l1987_198748

theorem divisor_between (n a b : ℕ) (h_n_gt_8 : n > 8) (h_a_dvd_n : a ∣ n) (h_b_dvd_n : b ∣ n) 
    (h_a_lt_b : a < b) (h_n_eq_asq_plus_b : n = a^2 + b) (h_a_ne_b : a ≠ b) :
  ∃ d : ℕ, d ∣ n ∧ a < d ∧ d < b :=
sorry

end NUMINAMATH_GPT_divisor_between_l1987_198748


namespace NUMINAMATH_GPT_boys_in_art_class_l1987_198729

noncomputable def number_of_boys (ratio_girls_to_boys : ℕ × ℕ) (total_students : ℕ) : ℕ :=
  let (g, b) := ratio_girls_to_boys
  let k := total_students / (g + b)
  b * k

theorem boys_in_art_class (h : number_of_boys (4, 3) 35 = 15) : true := 
  sorry

end NUMINAMATH_GPT_boys_in_art_class_l1987_198729


namespace NUMINAMATH_GPT_Q_2_plus_Q_neg_2_l1987_198734

noncomputable def cubic_polynomial (a b c k : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + k

theorem Q_2_plus_Q_neg_2 (a b c k : ℝ) 
  (h0 : cubic_polynomial a b c k 0 = k)
  (h1 : cubic_polynomial a b c k 1 = 3 * k)
  (hneg1 : cubic_polynomial a b c k (-1) = 4 * k) :
  cubic_polynomial a b c k 2 + cubic_polynomial a b c k (-2) = 22 * k :=
sorry

end NUMINAMATH_GPT_Q_2_plus_Q_neg_2_l1987_198734


namespace NUMINAMATH_GPT_hyperbola_asymptote_equation_l1987_198753

variable (a b : ℝ)
variable (x y : ℝ)

def arithmetic_mean := (a + b) / 2 = 5
def geometric_mean := (a * b) ^ (1 / 2) = 4
def a_greater_b := a > b
def hyperbola_asymptote := (y = (1 / 2) * x) ∨ (y = -(1 / 2) * x)

theorem hyperbola_asymptote_equation :
  arithmetic_mean a b ∧ geometric_mean a b ∧ a_greater_b a b → hyperbola_asymptote x y :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_equation_l1987_198753


namespace NUMINAMATH_GPT_solve_inequality_l1987_198769

theorem solve_inequality (a : ℝ) : 
  (if a = 0 ∨ a = 1 then { x : ℝ | false }
   else if a < 0 ∨ a > 1 then { x : ℝ | a < x ∧ x < a^2 }
   else if 0 < a ∧ a < 1 then { x : ℝ | a^2 < x ∧ x < a }
   else ∅) = 
  { x : ℝ | (x - a) / (x - a^2) < 0 } :=
by sorry

end NUMINAMATH_GPT_solve_inequality_l1987_198769


namespace NUMINAMATH_GPT_cos_diff_proof_l1987_198718

noncomputable def cos_diff (α β : ℝ) : ℝ := Real.cos (α - β)

theorem cos_diff_proof (α β : ℝ) 
  (h1 : Real.cos α - Real.cos β = 1 / 2)
  (h2 : Real.sin α - Real.sin β = 1 / 3) :
  cos_diff α β = 59 / 72 := by
  sorry

end NUMINAMATH_GPT_cos_diff_proof_l1987_198718


namespace NUMINAMATH_GPT_polynomial_factorization_l1987_198733

noncomputable def polynomial_expr (a b c : ℝ) :=
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2)

noncomputable def factored_form (a b c : ℝ) :=
  (a - b) * (b - c) * (c - a) * (b^2 + c^2 + a^2)

theorem polynomial_factorization (a b c : ℝ) :
  polynomial_expr a b c = factored_form a b c :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_factorization_l1987_198733


namespace NUMINAMATH_GPT_min_box_coeff_l1987_198766

theorem min_box_coeff (a b c d : ℤ) (h_ac : a * c = 40) (h_bd : b * d = 40) : 
  ∃ (min_val : ℤ), min_val = 89 ∧ (a * d + b * c) ≥ min_val :=
sorry

end NUMINAMATH_GPT_min_box_coeff_l1987_198766


namespace NUMINAMATH_GPT_gecko_bug_eating_l1987_198792

theorem gecko_bug_eating (G L F T : ℝ) (hL : L = G / 2)
                                      (hF : F = 3 * L)
                                      (hT : T = 1.5 * F)
                                      (hTotal : G + L + F + T = 63) :
  G = 15 :=
by
  sorry

end NUMINAMATH_GPT_gecko_bug_eating_l1987_198792


namespace NUMINAMATH_GPT_apple_allocation_proof_l1987_198767

theorem apple_allocation_proof : 
    ∃ (ann mary jane kate ned tom bill jack : ℕ), 
    ann = 1 ∧
    mary = 2 ∧
    jane = 3 ∧
    kate = 4 ∧
    ned = jane ∧
    tom = 2 * kate ∧
    bill = 3 * ann ∧
    jack = 4 * mary ∧
    ann + mary + jane + ned + kate + tom + bill + jack = 32 :=
by {
    sorry
}

end NUMINAMATH_GPT_apple_allocation_proof_l1987_198767


namespace NUMINAMATH_GPT_impossible_to_use_up_components_l1987_198747

theorem impossible_to_use_up_components 
  (p q r x y z : ℕ) 
  (condition1 : 2 * x + 2 * z = 2 * p + 2 * r + 2)
  (condition2 : 2 * x + y = 2 * p + q + 1)
  (condition3 : y + z = q + r) : 
  False :=
by sorry

end NUMINAMATH_GPT_impossible_to_use_up_components_l1987_198747


namespace NUMINAMATH_GPT_Xiaoming_speed_l1987_198794

theorem Xiaoming_speed (x xiaohong_speed_xiaoming_diff : ℝ) :
  (50 * (2 * x + 2) = 600) →
  (xiaohong_speed_xiaoming_diff = 2) →
  x + xiaohong_speed_xiaoming_diff = 7 :=
by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_Xiaoming_speed_l1987_198794


namespace NUMINAMATH_GPT_rectangle_width_l1987_198723

theorem rectangle_width (length : ℕ) (perimeter : ℕ) (h1 : length = 20) (h2 : perimeter = 70) :
  2 * (length + width) = perimeter → width = 15 :=
by
  intro h
  rw [h1, h2] at h
  -- Continue the steps to solve for width (can be simplified if not requesting the whole proof)
  sorry

end NUMINAMATH_GPT_rectangle_width_l1987_198723


namespace NUMINAMATH_GPT_find_angle_sum_l1987_198743

theorem find_angle_sum
  {α β : ℝ}
  (hα_acute : 0 < α ∧ α < π / 2)
  (hβ_acute : 0 < β ∧ β < π / 2)
  (h_tan_α : Real.tan α = 1 / 3)
  (h_cos_β : Real.cos β = 3 / 5) :
  α + 2 * β = π - Real.arctan (13 / 9) :=
sorry

end NUMINAMATH_GPT_find_angle_sum_l1987_198743


namespace NUMINAMATH_GPT_smallest_n_l1987_198798

theorem smallest_n (j c g : ℕ) (n : ℕ) (total_cost : ℕ) 
  (h_condition : total_cost = 10 * j ∧ total_cost = 16 * c ∧ total_cost = 18 * g ∧ total_cost = 24 * n) 
  (h_lcm : Nat.lcm (Nat.lcm 10 16) 18 = 720) : n = 30 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1987_198798


namespace NUMINAMATH_GPT_crayons_more_than_erasers_l1987_198780

-- Definitions of the conditions
def initial_crayons := 531
def initial_erasers := 38
def final_crayons := 391
def final_erasers := initial_erasers -- no erasers lost

-- Theorem statement
theorem crayons_more_than_erasers :
  final_crayons - final_erasers = 102 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_crayons_more_than_erasers_l1987_198780


namespace NUMINAMATH_GPT_sqrt_meaningful_value_x_l1987_198750

theorem sqrt_meaningful_value_x (x : ℝ) (h : x-1 ≥ 0) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_value_x_l1987_198750


namespace NUMINAMATH_GPT_systematic_sampling_l1987_198707

-- Definitions for the class of 50 students numbered from 1 to 50, sampling interval, and starting number.
def students : Set ℕ := {n | n ∈ Finset.range 50 ∧ n ≥ 1}
def sampling_interval : ℕ := 10
def start : ℕ := 6

-- The main theorem stating that the selected students' numbers are as given.
theorem systematic_sampling : ∃ (selected : List ℕ), selected = [6, 16, 26, 36, 46] ∧ 
  ∀ x ∈ selected, x ∈ students := 
  sorry

end NUMINAMATH_GPT_systematic_sampling_l1987_198707


namespace NUMINAMATH_GPT_archie_needs_sod_l1987_198704

-- Define the dimensions of the backyard
def backyard_length : ℕ := 20
def backyard_width : ℕ := 13

-- Define the dimensions of the shed
def shed_length : ℕ := 3
def shed_width : ℕ := 5

-- Statement: Prove that the area of the backyard minus the area of the shed equals 245 square yards
theorem archie_needs_sod : 
  backyard_length * backyard_width - shed_length * shed_width = 245 := 
by sorry

end NUMINAMATH_GPT_archie_needs_sod_l1987_198704


namespace NUMINAMATH_GPT_complementary_event_probability_l1987_198765

-- Define A and B as events such that B is the complement of A.
section
variables (A B : Prop) -- A and B are propositions representing events.
variable (P : Prop → ℝ) -- P is a function that gives the probability of an event.

-- Define the conditions for the problem.
variable (h_complementary : ∀ A B, A ∧ B = false ∧ A ∨ B = true) 
variable (h_PA : P A = 1 / 5)

-- The statement to be proved.
theorem complementary_event_probability : P B = 4 / 5 :=
by
  -- Here we would provide the proof, but for now, we use 'sorry' to bypass it.
  sorry
end

end NUMINAMATH_GPT_complementary_event_probability_l1987_198765


namespace NUMINAMATH_GPT_nancy_first_album_pictures_l1987_198768

theorem nancy_first_album_pictures (total_pics : ℕ) (total_albums : ℕ) (pics_per_album : ℕ)
    (h1 : total_pics = 51) (h2 : total_albums = 8) (h3 : pics_per_album = 5) :
    (total_pics - total_albums * pics_per_album = 11) :=
by
    sorry

end NUMINAMATH_GPT_nancy_first_album_pictures_l1987_198768


namespace NUMINAMATH_GPT_new_train_distance_l1987_198701

-- Given conditions
def distance_older_train : ℝ := 200
def percent_more : ℝ := 0.20

-- Conclusion to prove
theorem new_train_distance : (distance_older_train * (1 + percent_more)) = 240 := by
  -- Placeholder to indicate that we are skipping the actual proof steps
  sorry

end NUMINAMATH_GPT_new_train_distance_l1987_198701


namespace NUMINAMATH_GPT_purse_multiple_of_wallet_l1987_198763

theorem purse_multiple_of_wallet (W P : ℤ) (hW : W = 22) (hc : W + P = 107) : ∃ n : ℤ, n * W > P ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_purse_multiple_of_wallet_l1987_198763


namespace NUMINAMATH_GPT_gymnast_scores_difference_l1987_198760

theorem gymnast_scores_difference
  (s1 s2 s3 s4 s5 : ℝ)
  (h1 : (s2 + s3 + s4 + s5) / 4 = 9.46)
  (h2 : (s1 + s2 + s3 + s4) / 4 = 9.66)
  (h3 : (s2 + s3 + s4) / 3 = 9.58)
  : |s5 - s1| = 8.3 :=
sorry

end NUMINAMATH_GPT_gymnast_scores_difference_l1987_198760


namespace NUMINAMATH_GPT_solve_mod_equation_l1987_198715

def is_two_digit_positive_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem solve_mod_equation (u : ℕ) (h1 : is_two_digit_positive_integer u) (h2 : 13 * u % 100 = 52) : u = 4 :=
sorry

end NUMINAMATH_GPT_solve_mod_equation_l1987_198715


namespace NUMINAMATH_GPT_problem_statement_l1987_198712

theorem problem_statement (x y z : ℝ) (h1 : x = 2) (h2 : y = -1) (h3 : z = 3) :
  x^2 + y^2 + z^2 + 2*x*z = 26 :=
by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_problem_statement_l1987_198712


namespace NUMINAMATH_GPT_common_divisor_of_differences_l1987_198726

theorem common_divisor_of_differences 
  (a1 a2 b1 b2 c1 c2 d : ℤ) 
  (h1: d ∣ (a1 - a2)) 
  (h2: d ∣ (b1 - b2)) 
  (h3: d ∣ (c1 - c2)) : 
  d ∣ (a1 * b1 * c1 - a2 * b2 * c2) := 
by sorry

end NUMINAMATH_GPT_common_divisor_of_differences_l1987_198726


namespace NUMINAMATH_GPT_speed_of_second_part_of_trip_l1987_198702

-- Given conditions
def total_distance : Real := 50
def first_part_distance : Real := 25
def first_part_speed : Real := 66
def average_speed : Real := 44.00000000000001

-- The statement we want to prove
theorem speed_of_second_part_of_trip :
  ∃ second_part_speed : Real, second_part_speed = 33 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_second_part_of_trip_l1987_198702


namespace NUMINAMATH_GPT_quadratic_real_roots_iff_l1987_198796

theorem quadratic_real_roots_iff (k : ℝ) : 
  (∃ x : ℝ, (k-1) * x^2 + 3 * x - 1 = 0) ↔ k ≥ -5 / 4 ∧ k ≠ 1 := sorry

end NUMINAMATH_GPT_quadratic_real_roots_iff_l1987_198796


namespace NUMINAMATH_GPT_angle_between_north_and_south_southeast_l1987_198705

-- Given a circular floor pattern with 12 equally spaced rays
def num_rays : ℕ := 12
def total_degrees : ℕ := 360

-- Proving each central angle measure
def central_angle_measure : ℕ := total_degrees / num_rays

-- Define rays of interest
def segments_between_rays : ℕ := 5

-- Prove the angle between the rays pointing due North and South-Southeast
theorem angle_between_north_and_south_southeast :
  (segments_between_rays * central_angle_measure) = 150 := by
  sorry

end NUMINAMATH_GPT_angle_between_north_and_south_southeast_l1987_198705


namespace NUMINAMATH_GPT_find_x_l1987_198778

variables (z y x : Int)

def condition1 : Prop := z + 1 = 0
def condition2 : Prop := y - 1 = 1
def condition3 : Prop := x + 2 = -1

theorem find_x (h1 : condition1 z) (h2 : condition2 y) (h3 : condition3 x) : x = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1987_198778


namespace NUMINAMATH_GPT_coeff_fourth_term_expansion_l1987_198762

theorem coeff_fourth_term_expansion :
  (3 : ℚ) ^ 2 * (-1 : ℚ) / 8 * (Nat.choose 8 3) = -63 :=
by
  sorry

end NUMINAMATH_GPT_coeff_fourth_term_expansion_l1987_198762


namespace NUMINAMATH_GPT_projectile_first_reaches_70_feet_l1987_198724

theorem projectile_first_reaches_70_feet :
  ∃ t : ℝ, t = 7/4 ∧ 0 < t ∧ ∀ s : ℝ, s < t → -16 * s^2 + 80 * s < 70 :=
by 
  sorry

end NUMINAMATH_GPT_projectile_first_reaches_70_feet_l1987_198724


namespace NUMINAMATH_GPT_find_two_digit_number_l1987_198776

theorem find_two_digit_number :
  ∃ x y : ℕ, 10 * x + y = 78 ∧ 10 * x + y < 100 ∧ y ≠ 0 ∧ (10 * x + y) / y = 9 ∧ (10 * x + y) % y = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l1987_198776


namespace NUMINAMATH_GPT_right_triangle_exists_l1987_198706

theorem right_triangle_exists :
  (3^2 + 4^2 = 5^2) ∧ ¬(2^2 + 3^2 = 4^2) ∧ ¬(4^2 + 6^2 = 7^2) ∧ ¬(5^2 + 11^2 = 12^2) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_exists_l1987_198706


namespace NUMINAMATH_GPT_candy_left_l1987_198736

-- Define the number of candies each sibling has
def debbyCandy : ℕ := 32
def sisterCandy : ℕ := 42
def brotherCandy : ℕ := 48

-- Define the total candies collected
def totalCandy : ℕ := debbyCandy + sisterCandy + brotherCandy

-- Define the number of candies eaten
def eatenCandy : ℕ := 56

-- Define the remaining candies after eating some
def remainingCandy : ℕ := totalCandy - eatenCandy

-- The hypothesis stating the initial condition
theorem candy_left (h1 : debbyCandy = 32) (h2 : sisterCandy = 42) (h3 : brotherCandy = 48) (h4 : eatenCandy = 56) : remainingCandy = 66 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_candy_left_l1987_198736


namespace NUMINAMATH_GPT_mark_parking_tickets_eq_l1987_198744

def total_tickets : ℕ := 24
def sarah_speeding_tickets : ℕ := 6
def mark_speeding_tickets : ℕ := 6
def sarah_parking_tickets (S : ℕ) := S
def mark_parking_tickets (S : ℕ) := 2 * S
def total_traffic_tickets (S : ℕ) := S + 2 * S + sarah_speeding_tickets + mark_speeding_tickets

theorem mark_parking_tickets_eq (S : ℕ) (h1 : total_traffic_tickets S = total_tickets)
  (h2 : sarah_speeding_tickets = 6) (h3 : mark_speeding_tickets = 6) :
  mark_parking_tickets S = 8 :=
sorry

end NUMINAMATH_GPT_mark_parking_tickets_eq_l1987_198744


namespace NUMINAMATH_GPT_John_height_l1987_198714

open Real

variable (John Mary Tom Angela Helen Amy Becky Carl : ℝ)

axiom h1 : John = 1.5 * Mary
axiom h2 : Mary = 2 * Tom
axiom h3 : Tom = Angela - 70
axiom h4 : Angela = Helen + 4
axiom h5 : Helen = Amy + 3
axiom h6 : Amy = 1.2 * Becky
axiom h7 : Becky = 2 * Carl
axiom h8 : Carl = 120

theorem John_height : John = 675 := by
  sorry

end NUMINAMATH_GPT_John_height_l1987_198714


namespace NUMINAMATH_GPT_gcd_372_684_l1987_198700

theorem gcd_372_684 : Nat.gcd 372 684 = 12 :=
by
  sorry

end NUMINAMATH_GPT_gcd_372_684_l1987_198700
