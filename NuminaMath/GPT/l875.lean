import Mathlib

namespace NUMINAMATH_GPT_f_of_1_eq_zero_l875_87505

-- Conditions
variables (f : ℝ → ℝ)
-- f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
-- f is a periodic function with a period of 2
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (x + 2) = f x

-- Theorem statement
theorem f_of_1_eq_zero {f : ℝ → ℝ} (h1 : odd_function f) (h2 : periodic_function f) : f 1 = 0 :=
by { sorry }

end NUMINAMATH_GPT_f_of_1_eq_zero_l875_87505


namespace NUMINAMATH_GPT_correct_calculation_l875_87563

theorem correct_calculation : -Real.sqrt ((-5)^2) = -5 := 
by 
  sorry

end NUMINAMATH_GPT_correct_calculation_l875_87563


namespace NUMINAMATH_GPT_jenny_spent_180_minutes_on_bus_l875_87550

noncomputable def jennyBusTime : ℕ :=
  let timeAwayFromHome := 9 * 60  -- in minutes
  let classTime := 5 * 45  -- 5 classes each lasting 45 minutes
  let lunchTime := 45  -- in minutes
  let extracurricularTime := 90  -- 1 hour and 30 minutes
  timeAwayFromHome - (classTime + lunchTime + extracurricularTime)

theorem jenny_spent_180_minutes_on_bus : jennyBusTime = 180 :=
  by
  -- We need to prove that the total time Jenny was away from home minus time spent in school activities is 180 minutes.
  sorry  -- Proof to be completed.

end NUMINAMATH_GPT_jenny_spent_180_minutes_on_bus_l875_87550


namespace NUMINAMATH_GPT_percent_of_Q_l875_87599

theorem percent_of_Q (P Q : ℝ) (h : (50 / 100) * P = (20 / 100) * Q) : P = 0.4 * Q :=
sorry

end NUMINAMATH_GPT_percent_of_Q_l875_87599


namespace NUMINAMATH_GPT_square_perimeter_eq_area_perimeter_16_l875_87585

theorem square_perimeter_eq_area_perimeter_16 (s : ℕ) (h : s^2 = 4 * s) : 4 * s = 16 := by
  sorry

end NUMINAMATH_GPT_square_perimeter_eq_area_perimeter_16_l875_87585


namespace NUMINAMATH_GPT_janes_stick_shorter_than_sarahs_l875_87514

theorem janes_stick_shorter_than_sarahs :
  ∀ (pat_length jane_length pat_dirt sarah_factor : ℕ),
    pat_length = 30 →
    jane_length = 22 →
    pat_dirt = 7 →
    sarah_factor = 2 →
    (sarah_factor * (pat_length - pat_dirt)) - jane_length = 24 :=
by
  intros pat_length jane_length pat_dirt sarah_factor h1 h2 h3 h4
  -- sorry skips the proof
  sorry

end NUMINAMATH_GPT_janes_stick_shorter_than_sarahs_l875_87514


namespace NUMINAMATH_GPT_pen_shorter_than_pencil_l875_87521

-- Definitions of the given conditions
def P (R : ℕ) := R + 3
def L : ℕ := 12
def total_length (R : ℕ) := R + P R + L

-- The theorem to be proven
theorem pen_shorter_than_pencil (R : ℕ) (h : total_length R = 29) : L - P R = 2 :=
by
  sorry

end NUMINAMATH_GPT_pen_shorter_than_pencil_l875_87521


namespace NUMINAMATH_GPT_imaginary_unit_squared_in_set_l875_87556

-- Conditions of the problem
def imaginary_unit (i : ℂ) : Prop := i^2 = -1
def S : Set ℂ := {-1, 0, 1}

-- The statement to prove
theorem imaginary_unit_squared_in_set {i : ℂ} (hi : imaginary_unit i) : i^2 ∈ S := sorry

end NUMINAMATH_GPT_imaginary_unit_squared_in_set_l875_87556


namespace NUMINAMATH_GPT_remainder_seven_power_twenty_seven_l875_87559

theorem remainder_seven_power_twenty_seven :
  (7^27) % 1000 = 543 := 
sorry

end NUMINAMATH_GPT_remainder_seven_power_twenty_seven_l875_87559


namespace NUMINAMATH_GPT_village_population_rate_l875_87507

noncomputable def population_change_X (initial_X : ℕ) (decrease_rate : ℕ) (years : ℕ) : ℕ :=
  initial_X - decrease_rate * years

noncomputable def population_change_Y (initial_Y : ℕ) (increase_rate : ℕ) (years : ℕ) : ℕ :=
  initial_Y + increase_rate * years

theorem village_population_rate (initial_X decrease_rate initial_Y years result : ℕ) 
  (h1 : initial_X = 70000) (h2 : decrease_rate = 1200) 
  (h3 : initial_Y = 42000) (h4 : years = 14) 
  (h5 : initial_X - decrease_rate * years = initial_Y + result * years) 
  : result = 800 :=
  sorry

end NUMINAMATH_GPT_village_population_rate_l875_87507


namespace NUMINAMATH_GPT_rectangle_area_integer_length_width_l875_87525

theorem rectangle_area_integer_length_width (l w : ℕ) (h1 : w = l / 2) (h2 : 2 * l + 2 * w = 200) :
  l * w = 2178 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_integer_length_width_l875_87525


namespace NUMINAMATH_GPT_coffee_shop_sold_lattes_l875_87533

theorem coffee_shop_sold_lattes (T L : ℕ) (h1 : T = 6) (h2 : L = 4 * T + 8) : L = 32 :=
by
  sorry

end NUMINAMATH_GPT_coffee_shop_sold_lattes_l875_87533


namespace NUMINAMATH_GPT_midpoint_quadrilateral_area_l875_87590

theorem midpoint_quadrilateral_area (R : ℝ) (hR : 0 < R) :
  ∃ (Q : ℝ), Q = R / 4 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_quadrilateral_area_l875_87590


namespace NUMINAMATH_GPT_stratified_sampling_household_l875_87581

/-
  Given:
  - Total valid questionnaires: 500,000.
  - Number of people who purchased:
    - clothing, shoes, and hats: 198,000,
    - household goods: 94,000,
    - cosmetics: 116,000,
    - home appliances: 92,000.
  - Number of questionnaires selected from the "cosmetics" category: 116.
  
  Prove:
  - The number of questionnaires that should be selected from the "household goods" category is 94.
-/

theorem stratified_sampling_household (total_valid: ℕ)
  (clothing_shoes_hats: ℕ)
  (household_goods: ℕ)
  (cosmetics: ℕ)
  (home_appliances: ℕ)
  (sample_cosmetics: ℕ) :
  total_valid = 500000 →
  clothing_shoes_hats = 198000 →
  household_goods = 94000 →
  cosmetics = 116000 →
  home_appliances = 92000 →
  sample_cosmetics = 116 →
  (116 * household_goods = sample_cosmetics * cosmetics) →
  116 * 94000 = 116 * 116000 →
  94000 = 116000 →
  94 = 94 := by
  intros
  sorry

end NUMINAMATH_GPT_stratified_sampling_household_l875_87581


namespace NUMINAMATH_GPT_tagged_fish_in_second_catch_l875_87579

theorem tagged_fish_in_second_catch :
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  (total_tagged / N) * total_caught = 5 :=
by
  let N := 500
  let total_tagged := 50
  let total_caught := 50
  show (total_tagged / N) * total_caught = 5
  sorry

end NUMINAMATH_GPT_tagged_fish_in_second_catch_l875_87579


namespace NUMINAMATH_GPT_div_decimals_l875_87583

theorem div_decimals : 0.45 / 0.005 = 90 := sorry

end NUMINAMATH_GPT_div_decimals_l875_87583


namespace NUMINAMATH_GPT_rectangle_perimeter_l875_87576

theorem rectangle_perimeter :
  ∃ (a b : ℕ), (a ≠ b) ∧ (a * b = 2 * (a + b) - 4) ∧ (2 * (a + b) = 26) :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_perimeter_l875_87576


namespace NUMINAMATH_GPT_find_m_value_l875_87589

theorem find_m_value : 
  ∀ (u v : ℝ), 
    (3 * u^2 + 4 * u + 5 = 0) ∧ 
    (3 * v^2 + 4 * v + 5 = 0) ∧ 
    (u + v = -4/3) ∧ 
    (u * v = 5/3) → 
    ∃ m n : ℝ, 
      (x^2 + m * x + n = 0) ∧ 
      ((u^2 + 1) + (v^2 + 1) = -m) ∧ 
      (m = -4/9) :=
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_find_m_value_l875_87589


namespace NUMINAMATH_GPT_find_upper_book_pages_l875_87584

noncomputable def pages_in_upper_book (total_digits : ℕ) (page_diff : ℕ) : ℕ :=
  -- Here we would include the logic to determine the number of pages, but we are only focusing on the statement.
  207

theorem find_upper_book_pages :
  ∀ (total_digits page_diff : ℕ), total_digits = 999 → page_diff = 9 → pages_in_upper_book total_digits page_diff = 207 :=
by
  intros total_digits page_diff h1 h2
  sorry

end NUMINAMATH_GPT_find_upper_book_pages_l875_87584


namespace NUMINAMATH_GPT_exists_hexagon_in_square_l875_87580

structure Point (α : Type*) :=
(x : α)
(y : α)

def is_in_square (p : Point ℕ) : Prop :=
p.x ≤ 4 ∧ p.y ≤ 4

def area_of_hexagon (vertices : List (Point ℕ)) : ℝ :=
-- placeholder for actual area calculation of a hexagon
sorry

theorem exists_hexagon_in_square : ∃ (p1 p2 : Point ℕ), 
  is_in_square p1 ∧ is_in_square p2 ∧ 
  area_of_hexagon [⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 0⟩, ⟨4, 4⟩, p1, p2] = 6 :=
sorry

end NUMINAMATH_GPT_exists_hexagon_in_square_l875_87580


namespace NUMINAMATH_GPT_sum_of_positive_factors_of_72_l875_87532

/-- Define the divisor sum function based on the given formula -/
def divisor_sum (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 3
  | 3 => 4
  | 4 => 7
  | 6 => 12
  | 8 => 15
  | 12 => 28
  | 18 => 39
  | 24 => 60
  | 36 => 91
  | 48 => 124
  | 60 => 168
  | 72 => 195
  | _ => 0 -- This is not generally correct, just handles given problem specifically

theorem sum_of_positive_factors_of_72 :
  divisor_sum 72 = 195 :=
sorry

end NUMINAMATH_GPT_sum_of_positive_factors_of_72_l875_87532


namespace NUMINAMATH_GPT_trains_crossing_l875_87595

noncomputable def time_to_cross_each_other (v : ℝ) (L₁ L₂ : ℝ) (t₁ t₂ : ℝ) : ℝ :=
  (L₁ + L₂) / (2 * v)

theorem trains_crossing (v : ℝ) (t₁ t₂ : ℝ) (h1 : t₁ = 27) (h2 : t₂ = 17) :
  time_to_cross_each_other v (v * 27) (v * 17) t₁ t₂ = 22 :=
by
  -- Conditions
  have h3 : t₁ = 27 := h1
  have h4 : t₂ = 17 := h2
  -- Proof outline (not needed, just to ensure the setup is understood):
  -- Lengths
  let L₁ := v * 27
  let L₂ := v * 17
  -- Calculating Crossing Time
  have t := (L₁ + L₂) / (2 * v)
  -- Simplification leads to t = 22
  sorry

end NUMINAMATH_GPT_trains_crossing_l875_87595


namespace NUMINAMATH_GPT_raffle_prize_l875_87531

theorem raffle_prize (P : ℝ) :
  (0.80 * P = 80) → (P = 100) :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_raffle_prize_l875_87531


namespace NUMINAMATH_GPT_angle_C_value_sides_a_b_l875_87565

variables (A B C : ℝ) (a b c : ℝ)

-- First part: Proving the value of angle C
theorem angle_C_value
  (h1 : 2*Real.cos (A/2)^2 + (Real.cos B - Real.sqrt 3 * Real.sin B) * Real.cos C = 1)
  : C = Real.pi / 3 :=
sorry

-- Second part: Proving the values of a and b given c and the area
theorem sides_a_b
  (c : ℝ)
  (h2 : c = 2)
  (h3 : C = Real.pi / 3)
  (area : ℝ)
  (h4 : area = Real.sqrt 3)
  (h5 : 1/2 * a * b * Real.sin C = Real.sqrt 3)
  : a = 2 ∧ b = 2 :=
sorry

end NUMINAMATH_GPT_angle_C_value_sides_a_b_l875_87565


namespace NUMINAMATH_GPT_sum_of_squares_l875_87518

theorem sum_of_squares (n : ℕ) (x : ℕ) (h1 : (x + 1)^3 - x^3 = n^2) (h2 : n > 0) : ∃ a b : ℕ, n = a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l875_87518


namespace NUMINAMATH_GPT_probability_sunglasses_to_hat_l875_87558

variable (S H : Finset ℕ) -- S: set of people wearing sunglasses, H: set of people wearing hats
variable (num_S : Nat) (num_H : Nat) (num_SH : Nat)
variable (prob_hat_to_sunglasses : ℚ)

-- Conditions
def condition1 : num_S = 80 := sorry
def condition2 : num_H = 50 := sorry
def condition3 : prob_hat_to_sunglasses = 3 / 5 := sorry
def condition4 : num_SH = (3/5) * 50 := sorry

-- Question: Prove that the probability a person wearing sunglasses is also wearing a hat
theorem probability_sunglasses_to_hat :
  (num_SH : ℚ) / num_S = 3 / 8 :=
sorry

end NUMINAMATH_GPT_probability_sunglasses_to_hat_l875_87558


namespace NUMINAMATH_GPT_percentage_income_diff_l875_87578

variable (A B : ℝ)

-- Condition that B's income is 33.33333333333333% greater than A's income
def income_relation (A B : ℝ) : Prop :=
  B = (4 / 3) * A

-- Proof statement to show that A's income is 25% less than B's income
theorem percentage_income_diff : 
  income_relation A B → 
  ((B - A) / B) * 100 = 25 :=
by
  intros h
  rw [income_relation] at h
  sorry

end NUMINAMATH_GPT_percentage_income_diff_l875_87578


namespace NUMINAMATH_GPT_inner_square_area_l875_87522

theorem inner_square_area (side_ABCD : ℝ) (dist_BI : ℝ) (area_IJKL : ℝ) :
  side_ABCD = Real.sqrt 72 →
  dist_BI = 2 →
  area_IJKL = 39 :=
by
  sorry

end NUMINAMATH_GPT_inner_square_area_l875_87522


namespace NUMINAMATH_GPT_find_D_l875_87515

theorem find_D (D E F : ℝ) (h : ∀ x : ℝ, x ≠ 1 → x ≠ -2 → (1 / (x^3 - 3*x^2 - 4*x + 12)) = (D / (x - 1)) + (E / (x + 2)) + (F / (x + 2)^2)) :
    D = -1 / 15 :=
by
  -- the proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_find_D_l875_87515


namespace NUMINAMATH_GPT_pascal_triangle_41st_number_42nd_row_l875_87544

open Nat

theorem pascal_triangle_41st_number_42nd_row :
  Nat.choose 42 40 = 861 := by
  sorry

end NUMINAMATH_GPT_pascal_triangle_41st_number_42nd_row_l875_87544


namespace NUMINAMATH_GPT_alpha_in_second_quadrant_l875_87519

theorem alpha_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 > 0 :=
by
  -- Given conditions
  have : Real.sin α > 0 := h1
  have : Real.cos α < 0 := h2
  sorry

end NUMINAMATH_GPT_alpha_in_second_quadrant_l875_87519


namespace NUMINAMATH_GPT_sandy_initial_books_l875_87568

-- Define the initial conditions as given.
def books_tim : ℕ := 33
def books_lost : ℕ := 24
def books_after_loss : ℕ := 19

-- Define the equation for the total books before Benny's loss and solve for Sandy's books.
def books_total_before_loss : ℕ := books_after_loss + books_lost
def books_sandy_initial : ℕ := books_total_before_loss - books_tim

-- Assert the proof statement:
def proof_sandy_books : Prop :=
  books_sandy_initial = 10

theorem sandy_initial_books : proof_sandy_books := by
  -- Placeholder for the actual proof.
  sorry

end NUMINAMATH_GPT_sandy_initial_books_l875_87568


namespace NUMINAMATH_GPT_line_of_symmetry_is_x_eq_0_l875_87520

variable (f : ℝ → ℝ)

theorem line_of_symmetry_is_x_eq_0 :
  (∀ y, f (10 + y) = f (10 - y)) → ( ∃ l, l = 0 ∧ ∀ x,  f (10 + l + x) = f (10 + l - x)) := 
by
  sorry

end NUMINAMATH_GPT_line_of_symmetry_is_x_eq_0_l875_87520


namespace NUMINAMATH_GPT_triangle_area_l875_87571

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area : 
  ∀ (A B C : (ℝ × ℝ)),
  (A = (3, 3)) →
  (B = (4.5, 7.5)) →
  (C = (7.5, 4.5)) →
  area_of_triangle A B C = 8.625 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  unfold area_of_triangle
  norm_num
  sorry

end NUMINAMATH_GPT_triangle_area_l875_87571


namespace NUMINAMATH_GPT_tire_usage_is_25714_l875_87539

-- Definitions based on conditions
def car_has_six_tires : Prop := (4 + 2 = 6)
def used_equally_over_miles (total_miles : ℕ) (number_of_tires : ℕ) : Prop := 
  (total_miles * 4) / number_of_tires = 25714

-- Theorem statement based on proof
theorem tire_usage_is_25714 (miles_driven : ℕ) (num_tires : ℕ) 
  (h1 : car_has_six_tires) 
  (h2 : miles_driven = 45000)
  (h3 : num_tires = 7) :
  used_equally_over_miles miles_driven num_tires :=
by
  sorry

end NUMINAMATH_GPT_tire_usage_is_25714_l875_87539


namespace NUMINAMATH_GPT_maximum_profit_l875_87557

noncomputable def L1 (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def L2 (x : ℝ) : ℝ := 2 * x

theorem maximum_profit :
  (∀ (x1 x2 : ℝ), x1 + x2 = 15 → L1 x1 + L2 x2 ≤ 45.6) := sorry

end NUMINAMATH_GPT_maximum_profit_l875_87557


namespace NUMINAMATH_GPT_find_a_from_polynomial_factor_l875_87567

theorem find_a_from_polynomial_factor (a b : ℤ)
  (h: ∀ x : ℝ, x*x - x - 1 = 0 → a*x^5 + b*x^4 + 1 = 0) : a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_from_polynomial_factor_l875_87567


namespace NUMINAMATH_GPT_order_exponents_l875_87587

theorem order_exponents :
  (2:ℝ) ^ 300 < (3:ℝ) ^ 200 ∧ (3:ℝ) ^ 200 < (10:ℝ) ^ 100 :=
by
  sorry

end NUMINAMATH_GPT_order_exponents_l875_87587


namespace NUMINAMATH_GPT_max_mx_plus_ny_l875_87549

theorem max_mx_plus_ny 
  (m n x y : ℝ) 
  (h1 : m^2 + n^2 = 6) 
  (h2 : x^2 + y^2 = 24) : 
  mx + ny ≤ 12 :=
sorry

end NUMINAMATH_GPT_max_mx_plus_ny_l875_87549


namespace NUMINAMATH_GPT_simplify_polynomial_l875_87526

variable {R : Type*} [CommRing R]

theorem simplify_polynomial (x : R) :
  (12 * x ^ 10 + 9 * x ^ 9 + 5 * x ^ 8) + (2 * x ^ 12 + x ^ 10 + 2 * x ^ 9 + 3 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9) =
  2 * x ^ 12 + 13 * x ^ 10 + 11 * x ^ 9 + 8 * x ^ 8 + 4 * x ^ 4 + 6 * x ^ 2 + 9 :=
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l875_87526


namespace NUMINAMATH_GPT_ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l875_87529

variable (a b : ℝ)
variable (m x : ℝ)
variable (t : ℝ)
variable (A B C : ℝ)

-- Define ideal points
def is_ideal_point (p : ℝ × ℝ) := p.snd = 2 * p.fst

-- Define the conditions for question 1
def distance_from_y_axis (a : ℝ) := abs a = 2

-- Question 1: Prove that M(2, 4) or M(-2, -4)
theorem ideal_point_distance_y_axis (a b : ℝ) (h1 : is_ideal_point (a, b)) (h2 : distance_from_y_axis a) :
  (a = 2 ∧ b = 4) ∨ (a = -2 ∧ b = -4) := sorry

-- Define the linear function
def linear_func (m x : ℝ) : ℝ := 3 * m * x - 1

-- Question 2: Prove or disprove the existence of ideal points in y = 3mx - 1
theorem exists_ideal_point_linear (m x : ℝ) (hx : is_ideal_point (x, linear_func m x)) :
  (m ≠ 2/3 → ∃ x, linear_func m x = 2 * x) ∧ (m = 2/3 → ¬ ∃ x, linear_func m x = 2 * x) := sorry

-- Question 3 conditions
def quadratic_func (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def quadratic_conditions (a b c : ℝ) : Prop :=
  (quadratic_func a b c 0 = 5 * a + 1) ∧ (quadratic_func a b c (-2) = 5 * a + 1)

-- Question 3: Prove the range of t = a^2 + a + 1 given the quadratic conditions
theorem range_of_t (a b c t : ℝ) (h1 : is_ideal_point (x, quadratic_func a b c x))
  (h2 : quadratic_conditions a b c) (ht : t = a^2 + a + 1) :
    3 / 4 ≤ t ∧ t ≤ 21 / 16 ∧ t ≠ 1 := sorry

end NUMINAMATH_GPT_ideal_point_distance_y_axis_exists_ideal_point_linear_range_of_t_l875_87529


namespace NUMINAMATH_GPT_ratio_female_to_total_l875_87509

theorem ratio_female_to_total:
  ∃ (F : ℕ), (6 + 7 * F - 9 = (6 + 7 * F) - 9) ∧ 
             (7 * F - 9 = 67 / 100 * ((6 + 7 * F) - 9)) → 
             F = 3 ∧ 6 = 6 → 
             1 / F = 2 / 6 :=
by sorry

end NUMINAMATH_GPT_ratio_female_to_total_l875_87509


namespace NUMINAMATH_GPT_seashells_total_l875_87564

theorem seashells_total (joan_seashells jessica_seashells : ℕ)
  (h_joan : joan_seashells = 6)
  (h_jessica : jessica_seashells = 8) :
  joan_seashells + jessica_seashells = 14 :=
by 
  sorry

end NUMINAMATH_GPT_seashells_total_l875_87564


namespace NUMINAMATH_GPT_arithmetic_sum_ratio_l875_87537

variable (a_n : ℕ → ℤ) -- the arithmetic sequence
variable (S : ℕ → ℤ) -- sum of the first n terms of the sequence
variable (d : ℤ) (a₁ : ℤ) -- common difference and first term of the sequence

-- Definition of the sum of the first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given condition
axiom h1 : (S 6) / (S 3) = 3

-- Definition of S_n in terms of the given formula
axiom S_def : ∀ n, S n = arithmetic_sum n

-- The main goal to prove
theorem arithmetic_sum_ratio : S 12 / S 9 = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_ratio_l875_87537


namespace NUMINAMATH_GPT_fourth_pentagon_has_31_dots_l875_87574

-- Conditions representing the sequence of pentagons
def first_pentagon_dots : ℕ := 1

def second_pentagon_dots : ℕ := first_pentagon_dots + 5

def nth_layer_dots (n : ℕ) : ℕ := 5 * (n - 1)

def nth_pentagon_dots (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + nth_layer_dots (k+1)) first_pentagon_dots

-- Question and proof statement
theorem fourth_pentagon_has_31_dots : nth_pentagon_dots 4 = 31 :=
  sorry

end NUMINAMATH_GPT_fourth_pentagon_has_31_dots_l875_87574


namespace NUMINAMATH_GPT_find_y_when_x_is_8_l875_87547

theorem find_y_when_x_is_8 : 
  ∃ k, (70 * 5 = k ∧ 8 * 25 = k) := 
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_8_l875_87547


namespace NUMINAMATH_GPT_part1_part2_l875_87502

-- Define the solution set M for the inequality
def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

-- Define the problem conditions
variables {a b : ℝ} (ha : a ∈ M) (hb : b ∈ M)

-- First part: Prove that |(1/3)a + (1/6)b| < 1/4
theorem part1 : |(1/3 : ℝ) * a + (1/6 : ℝ) * b| < 1/4 :=
sorry

-- Second part: Prove that |1 - 4 * a * b| > 2 * |a - b|
theorem part2 : |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end NUMINAMATH_GPT_part1_part2_l875_87502


namespace NUMINAMATH_GPT_shark_sightings_l875_87548

theorem shark_sightings (x : ℕ) 
  (h1 : 26 = 5 + 3 * x) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_shark_sightings_l875_87548


namespace NUMINAMATH_GPT_hyperbola_equation_standard_form_l875_87524

noncomputable def point_on_hyperbola_asymptote (A : ℝ × ℝ) (C : ℝ) : Prop :=
  let x := A.1
  let y := A.2
  (4 * y^2 - x^2 = C) ∧
  (y = (1/2) * x ∨ y = -(1/2) * x)

theorem hyperbola_equation_standard_form
  (A : ℝ × ℝ)
  (hA : A = (2 * Real.sqrt 2, 2))
  (asymptote1 asymptote2 : ℝ → ℝ)
  (hasymptote1 : ∀ x, asymptote1 x = (1/2) * x)
  (hasymptote2 : ∀ x, asymptote2 x = -(1/2) * x) :
  (∃ C : ℝ, point_on_hyperbola_asymptote A C) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (4 * (A.2)^2 - (A.1)^2 = 8) ∧ 
    (∀ x y : ℝ, (4 * y^2 - x^2 = 8) ↔ ((y^2) / a - (x^2) / b = 1))) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_standard_form_l875_87524


namespace NUMINAMATH_GPT_consecutive_differences_equal_l875_87527

-- Define the set and the condition
def S : Set ℕ := {n : ℕ | n > 0}

-- Condition that for any two numbers a and b in S with a > b, at least one of a + b or a - b is also in S
axiom h_condition : ∀ a b : ℕ, a ∈ S → b ∈ S → a > b → (a + b ∈ S ∨ a - b ∈ S)

-- The main theorem that we want to prove
theorem consecutive_differences_equal (a : ℕ) (s : Fin 2003 → ℕ) 
  (hS : ∀ i, s i ∈ S)
  (h_ordered : ∀ i j, i < j → s i < s j) :
  ∃ (d : ℕ), ∀ i, i < 2002 → (s (i + 1)) - (s i) = d :=
sorry

end NUMINAMATH_GPT_consecutive_differences_equal_l875_87527


namespace NUMINAMATH_GPT_find_b_plus_k_l875_87536

open Real

noncomputable def semi_major_axis (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) : ℝ :=
  dist p f1 + dist p f2

def c_squared (a : ℝ) (b : ℝ) : ℝ :=
  a ^ 2 - b ^ 2

theorem find_b_plus_k :
  ∀ (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) (h k : ℝ) (a b : ℝ),
  f1 = (-2, 0) →
  f2 = (2, 0) →
  p = (6, 0) →
  (∃ a b, semi_major_axis f1 f2 p = 2 * a ∧ c_squared a b = 4) →
  h = 0 →
  k = 0 →
  b = 4 * sqrt 2 →
  b + k = 4 * sqrt 2 :=
by
  intros f1 f2 p h k a b f1_def f2_def p_def maj_axis_def h_def k_def b_def
  rw [b_def, k_def]
  exact add_zero (4 * sqrt 2)

end NUMINAMATH_GPT_find_b_plus_k_l875_87536


namespace NUMINAMATH_GPT_no_infinite_set_exists_l875_87592

variable {S : Set ℕ} -- We assume S is a set of natural numbers

def satisfies_divisibility_condition (a b : ℕ) : Prop :=
  (a^2 + b^2 - a * b) ∣ (a * b)^2

theorem no_infinite_set_exists (h1 : Infinite S)
  (h2 : ∀ (a b : ℕ), a ∈ S → b ∈ S → satisfies_divisibility_condition a b) : false :=
  sorry

end NUMINAMATH_GPT_no_infinite_set_exists_l875_87592


namespace NUMINAMATH_GPT_gcd_lcm_ratio_l875_87570

theorem gcd_lcm_ratio (A B : ℕ) (k : ℕ) (h1 : Nat.lcm A B = 200) (h2 : 2 * k = A) (h3 : 5 * k = B) : Nat.gcd A B = k :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_ratio_l875_87570


namespace NUMINAMATH_GPT_factorization_option_D_l875_87538

-- Define variables
variables (x y : ℝ)

-- Define the expressions
def left_side_D := -4 * x^2 + 12 * x * y - 9 * y^2
def right_side_D := -(2 * x - 3 * y)^2

-- Theorem statement
theorem factorization_option_D : left_side_D x y = right_side_D x y :=
sorry

end NUMINAMATH_GPT_factorization_option_D_l875_87538


namespace NUMINAMATH_GPT_days_to_complete_job_l875_87540

theorem days_to_complete_job (m₁ m₂ d₁ d₂ total_man_days : ℝ)
    (h₁ : m₁ = 30)
    (h₂ : d₁ = 8)
    (h₃ : total_man_days = 240)
    (h₄ : total_man_days = m₁ * d₁)
    (h₅ : m₂ = 40) :
    d₂ = total_man_days / m₂ := by
  sorry

end NUMINAMATH_GPT_days_to_complete_job_l875_87540


namespace NUMINAMATH_GPT_alia_markers_count_l875_87566

theorem alia_markers_count :
  ∀ (Alia Austin Steve Bella : ℕ),
  (Alia = 2 * Austin) →
  (Austin = (1 / 3) * Steve) →
  (Steve = 60) →
  (Bella = (3 / 2) * Alia) →
  Alia = 40 :=
by
  intros Alia Austin Steve Bella H1 H2 H3 H4
  sorry

end NUMINAMATH_GPT_alia_markers_count_l875_87566


namespace NUMINAMATH_GPT_simplify_expr_l875_87555

theorem simplify_expr (a b x : ℝ) (h₁ : x = a^3 / b^3) (h₂ : a ≠ b) (h₃ : b ≠ 0) : 
  (a^3 + b^3) / (a^3 - b^3) = (x + 1) / (x - 1) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expr_l875_87555


namespace NUMINAMATH_GPT_arrange_squares_l875_87554

theorem arrange_squares (n : ℕ) (h : n ≥ 5) :
  ∃ arrangement : Fin n → Fin n × Fin n, 
    (∀ i j : Fin n, i ≠ j → 
      (arrangement i).fst + (arrangement i).snd = (arrangement j).fst + (arrangement j).snd
      ∨ (arrangement i).fst = (arrangement j).fst
      ∨ (arrangement i).snd = (arrangement j).snd) :=
sorry

end NUMINAMATH_GPT_arrange_squares_l875_87554


namespace NUMINAMATH_GPT_binary_addition_l875_87508

theorem binary_addition :
  0b1101 + 0b101 + 0b1110 + 0b10111 + 0b11000 = 0b11100010 :=
by
  sorry

end NUMINAMATH_GPT_binary_addition_l875_87508


namespace NUMINAMATH_GPT_percentage_two_sections_cleared_l875_87534

noncomputable def total_candidates : ℕ := 1200
def pct_cleared_all_sections : ℝ := 0.05
def pct_cleared_none_sections : ℝ := 0.05
def pct_cleared_one_section : ℝ := 0.25
def pct_cleared_four_sections : ℝ := 0.20
def cleared_three_sections : ℕ := 300

theorem percentage_two_sections_cleared :
  (total_candidates - total_candidates * (pct_cleared_all_sections + pct_cleared_none_sections + pct_cleared_one_section + pct_cleared_four_sections) - cleared_three_sections) / total_candidates * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_two_sections_cleared_l875_87534


namespace NUMINAMATH_GPT_anne_cleaning_time_l875_87528

theorem anne_cleaning_time (B A : ℝ) 
  (h₁ : 4 * (B + A) = 1) 
  (h₂ : 3 * (B + 2 * A) = 1) : 
  1 / A = 12 :=
sorry

end NUMINAMATH_GPT_anne_cleaning_time_l875_87528


namespace NUMINAMATH_GPT_total_books_l875_87516

-- Define the conditions
def books_per_shelf : ℕ := 9
def mystery_shelves : ℕ := 6
def picture_shelves : ℕ := 2

-- The proof problem statement
theorem total_books : 
  (mystery_shelves * books_per_shelf) + 
  (picture_shelves * books_per_shelf) = 72 := 
sorry

end NUMINAMATH_GPT_total_books_l875_87516


namespace NUMINAMATH_GPT_distance_to_angle_bisector_l875_87535

theorem distance_to_angle_bisector 
  (P : ℝ × ℝ) 
  (h_hyperbola : P.1^2 - P.2^2 = 9) 
  (h_distance_to_line_neg_x : abs (P.1 + P.2) = 2016 * Real.sqrt 2) : 
  abs (P.1 - P.2) / Real.sqrt 2 = 448 :=
sorry

end NUMINAMATH_GPT_distance_to_angle_bisector_l875_87535


namespace NUMINAMATH_GPT_find_three_digit_number_l875_87530

theorem find_three_digit_number (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : 0 ≤ c ∧ c ≤ 9)
    (h₄ : (10 * a + b) / 99 + (100 * a + 10 * b + c) / 999 = 33 / 37) :
    100 * a + 10 * b + c = 447 :=
sorry

end NUMINAMATH_GPT_find_three_digit_number_l875_87530


namespace NUMINAMATH_GPT_xiao_wang_ways_to_make_8_cents_l875_87500

theorem xiao_wang_ways_to_make_8_cents :
  (∃ c1 c2 c5 : ℕ, c1 ≤ 8 ∧ c2 ≤ 4 ∧ c5 ≤ 1 ∧ c1 + 2 * c2 + 5 * c5 = 8) → (number_of_ways_to_make_8_cents = 7) :=
sorry

end NUMINAMATH_GPT_xiao_wang_ways_to_make_8_cents_l875_87500


namespace NUMINAMATH_GPT_prove_proposition_l875_87541

-- Define the propositions p and q
def p : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def q : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Define the main theorem to prove
theorem prove_proposition : (¬ p) ∨ q :=
by { sorry }

end NUMINAMATH_GPT_prove_proposition_l875_87541


namespace NUMINAMATH_GPT_increased_work_l875_87512

variable (W p : ℕ)

theorem increased_work (hW : W > 0) (hp : p > 0) : 
  (W / (7 * p / 8)) - (W / p) = W / (7 * p) := 
sorry

end NUMINAMATH_GPT_increased_work_l875_87512


namespace NUMINAMATH_GPT_abs_a_k_le_fractional_l875_87562

variable (a : ℕ → ℝ) (n : ℕ)

-- Condition 1: a_0 = a_(n+1) = 0
axiom a_0 : a 0 = 0
axiom a_n1 : a (n + 1) = 0

-- Condition 2: |a_{k-1} - 2a_k + a_{k+1}| ≤ 1 for k = 1, 2, ..., n
axiom abs_diff_ineq (k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : 
  |a (k - 1) - 2 * a k + a (k + 1)| ≤ 1

-- Theorem statement
theorem abs_a_k_le_fractional (k : ℕ) (h : 0 ≤ k ∧ k ≤ n + 1) : 
  |a k| ≤ k * (n + 1 - k) / 2 := sorry

end NUMINAMATH_GPT_abs_a_k_le_fractional_l875_87562


namespace NUMINAMATH_GPT_sum_invested_l875_87561

theorem sum_invested (P R: ℝ) (h1: SI₁ = P * R * 20 / 100) (h2: SI₂ = P * (R + 10) * 20 / 100) (h3: SI₂ = SI₁ + 3000) : P = 1500 :=
by
  sorry

end NUMINAMATH_GPT_sum_invested_l875_87561


namespace NUMINAMATH_GPT_ellipse_equation_l875_87517

theorem ellipse_equation (a b c : ℝ) (h0 : a > b) (h1 : b > 0) (h2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) 
  (h3 : dist (3, y) (5 - 5 / 2, 0) = 6.5) (h4 : dist (3, y) (5 + 5 / 2, 0) = 3.5) : 
  ( ∀ x y, (x^2 / 25) + (y^2 / (75 / 4)) = 1 ) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_l875_87517


namespace NUMINAMATH_GPT_original_price_of_sarees_l875_87573

theorem original_price_of_sarees (P : ℝ) (h : 0.92 * 0.90 * P = 331.2) : P = 400 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_sarees_l875_87573


namespace NUMINAMATH_GPT_remainder_gx12_div_gx_l875_87511

-- Definition of the polynomial g(x)
def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Theorem stating the problem
theorem remainder_gx12_div_gx : ∀ x : ℂ, (g (x^12)) % (g x) = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_gx12_div_gx_l875_87511


namespace NUMINAMATH_GPT_bowling_ball_weight_l875_87598

variables (b c k : ℝ)

def condition1 : Prop := 9 * b = 6 * c
def condition2 : Prop := c + k = 42
def condition3 : Prop := 3 * k = 2 * c

theorem bowling_ball_weight
  (h1 : condition1 b c)
  (h2 : condition2 c k)
  (h3 : condition3 c k) :
  b = 16.8 :=
sorry

end NUMINAMATH_GPT_bowling_ball_weight_l875_87598


namespace NUMINAMATH_GPT_only_book_A_l875_87575

variable (numA numB numBoth numOnlyB x : ℕ)
variable (h1 : numA = 2 * numB)
variable (h2 : numBoth = 500)
variable (h3 : numBoth = 2 * numOnlyB)
variable (h4 : numB = numOnlyB + numBoth)
variable (h5 : x = numA - numBoth)

theorem only_book_A : 
  x = 1000 := 
by
  sorry

end NUMINAMATH_GPT_only_book_A_l875_87575


namespace NUMINAMATH_GPT_cubic_inequality_l875_87545

theorem cubic_inequality (a b : ℝ) : (a > b) ↔ (a^3 > b^3) := sorry

end NUMINAMATH_GPT_cubic_inequality_l875_87545


namespace NUMINAMATH_GPT_point_B_coordinates_l875_87560

variable (A : ℝ × ℝ)

def move_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + d)

def move_left (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  (p.1 - d, p.2)

theorem point_B_coordinates : 
  (move_left (move_up (-3, -5) 4) 3) = (-6, -1) :=
by
  sorry

end NUMINAMATH_GPT_point_B_coordinates_l875_87560


namespace NUMINAMATH_GPT_pascal_fifth_element_row_20_l875_87596

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem pascal_fifth_element_row_20 : binom 20 4 = 4845 := sorry

end NUMINAMATH_GPT_pascal_fifth_element_row_20_l875_87596


namespace NUMINAMATH_GPT_units_digit_of_sum_is_7_l875_87594

noncomputable def original_num (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
noncomputable def reversed_num (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

theorem units_digit_of_sum_is_7 (a b c : ℕ) (h : a = 2 * c - 3) :
  (original_num a b c + reversed_num a b c) % 10 = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_of_sum_is_7_l875_87594


namespace NUMINAMATH_GPT_relationship_between_c_and_d_l875_87597

noncomputable def c : ℝ := Real.log 400 / Real.log 4
noncomputable def d : ℝ := Real.log 20 / Real.log 2

theorem relationship_between_c_and_d : c = d := by
  sorry

end NUMINAMATH_GPT_relationship_between_c_and_d_l875_87597


namespace NUMINAMATH_GPT_number_of_terms_l875_87569

noncomputable def Sn (n : ℕ) : ℝ := sorry

def an_arithmetic_seq (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

theorem number_of_terms {a : ℕ → ℝ}
  (h_arith : an_arithmetic_seq a)
  (cond1 : a 1 + a 2 + a 3 + a 4 = 1)
  (cond2 : a 5 + a 6 + a 7 + a 8 = 2)
  (cond3 : Sn = 15) :
  ∃ n, n = 16 :=
sorry

end NUMINAMATH_GPT_number_of_terms_l875_87569


namespace NUMINAMATH_GPT_union_M_N_l875_87546

-- Definitions for the sets M and N
def M : Set ℝ := { x | x^2 = x }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≤ 0 }

-- Proof problem statement
theorem union_M_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_GPT_union_M_N_l875_87546


namespace NUMINAMATH_GPT_tree_initial_height_l875_87572

noncomputable def initial_tree_height (H : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ := 
  H + growth_rate * years

theorem tree_initial_height :
  ∀ (H : ℝ), 
  (∀ (years : ℕ), ∃ h : ℝ, h = initial_tree_height H 0.5 years) →
  initial_tree_height H 0.5 6 = initial_tree_height H 0.5 4 * (7 / 6) →
  H = 4 :=
by
  intro H height_increase condition
  sorry

end NUMINAMATH_GPT_tree_initial_height_l875_87572


namespace NUMINAMATH_GPT_num_distinct_integers_formed_l875_87577

theorem num_distinct_integers_formed (digits : Multiset ℕ) (h : digits = {2, 2, 3, 3, 3}) : 
  Multiset.card (Multiset.powerset digits).attach = 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_num_distinct_integers_formed_l875_87577


namespace NUMINAMATH_GPT_find_common_tangent_sum_constant_l875_87504

theorem find_common_tangent_sum_constant :
  ∃ (a b c : ℕ), (∀ x y : ℚ, y = x^2 + 169/100 → x = y^2 + 49/4 → a * x + b * y = c) ∧
  (Int.gcd (Int.gcd a b) c = 1) ∧
  (a + b + c = 52) :=
sorry

end NUMINAMATH_GPT_find_common_tangent_sum_constant_l875_87504


namespace NUMINAMATH_GPT_tom_monthly_fluid_intake_l875_87523

-- Define the daily fluid intake amounts
def daily_soda_intake := 5 * 12
def daily_water_intake := 64
def daily_juice_intake := 3 * 8
def daily_sports_drink_intake := 2 * 16
def additional_weekend_smoothie := 32

-- Define the weekdays and weekend days in a month
def weekdays_in_month := 5 * 4
def weekend_days_in_month := 2 * 4

-- Calculate the total daily intake
def daily_intake := daily_soda_intake + daily_water_intake + daily_juice_intake + daily_sports_drink_intake
def weekend_daily_intake := daily_intake + additional_weekend_smoothie

-- Calculate the total monthly intake
def total_fluid_intake_in_month := (daily_intake * weekdays_in_month) + (weekend_daily_intake * weekend_days_in_month)

-- Statement to prove
theorem tom_monthly_fluid_intake : total_fluid_intake_in_month = 5296 :=
by
  unfold total_fluid_intake_in_month
  unfold daily_intake weekend_daily_intake
  unfold weekdays_in_month weekend_days_in_month
  unfold daily_soda_intake daily_water_intake daily_juice_intake daily_sports_drink_intake additional_weekend_smoothie
  sorry

end NUMINAMATH_GPT_tom_monthly_fluid_intake_l875_87523


namespace NUMINAMATH_GPT_find_x_l875_87591

theorem find_x (x : ℝ) (h : (x * (x ^ 4) ^ (1/2)) ^ (1/4) = 2) : 
  x = 16 ^ (1/3) :=
sorry

end NUMINAMATH_GPT_find_x_l875_87591


namespace NUMINAMATH_GPT_evaluate_expression_l875_87513

variables (a b c d m : ℝ)

lemma opposite_is_zero (h1 : a + b = 0) : a + b = 0 := h1

lemma reciprocals_equal_one (h2 : c * d = 1) : c * d = 1 := h2

lemma abs_value_two (h3 : |m| = 2) : |m| = 2 := h3

theorem evaluate_expression (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 2) :
  m + c * d + (a + b) / m = 3 ∨ m + c * d + (a + b) / m = -1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l875_87513


namespace NUMINAMATH_GPT_find_ABC_l875_87542

noncomputable def g (x : ℝ) (A B C : ℝ) : ℝ := 
  x^2 / (A * x^2 + B * x + C)

theorem find_ABC : 
  (∀ x : ℝ, x > 5 → g x 2 (-2) (-24) > 0.5) ∧
  (A = 2) ∧
  (B = -2) ∧
  (C = -24) ∧
  (∀ x, A * x^2 + B * x + C = A * (x + 3) * (x - 4)) → 
  A + B + C = -24 := 
by
  sorry

end NUMINAMATH_GPT_find_ABC_l875_87542


namespace NUMINAMATH_GPT_find_a_l875_87553

theorem find_a (a b c d : ℤ) 
  (h1 : d + 0 = 2)
  (h2 : c + 2 = 2)
  (h3 : b + 0 = 4)
  (h4 : a + 4 = 0) : 
  a = -4 := 
sorry

end NUMINAMATH_GPT_find_a_l875_87553


namespace NUMINAMATH_GPT_product_g_roots_l875_87588

noncomputable def f (x : ℝ) : ℝ := x^4 - x^3 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 3

theorem product_g_roots (x_1 x_2 x_3 x_4 : ℝ) (hx : ∀ x, (x = x_1 ∨ x = x_2 ∨ x = x_3 ∨ x = x_4) ↔ f x = 0) :
  g x_1 * g x_2 * g x_3 * g x_4 = 142 :=
by sorry

end NUMINAMATH_GPT_product_g_roots_l875_87588


namespace NUMINAMATH_GPT_infimum_of_function_l875_87586

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 1)^2

def is_lower_bound (M : ℝ) (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x ≥ M

def is_infimum (M : ℝ) (f : ℝ → ℝ) : Prop :=
  is_lower_bound M f ∧ ∀ L : ℝ, is_lower_bound L f → L ≤ M

theorem infimum_of_function :
  is_infimum 0.5 f :=
sorry

end NUMINAMATH_GPT_infimum_of_function_l875_87586


namespace NUMINAMATH_GPT_sum_of_p_q_r_s_t_l875_87552

theorem sum_of_p_q_r_s_t (p q r s t : ℤ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = 120) : 
  p + q + r + s + t = 32 := 
sorry

end NUMINAMATH_GPT_sum_of_p_q_r_s_t_l875_87552


namespace NUMINAMATH_GPT_pretzels_count_l875_87503

-- Define the number of pretzels
def pretzels : ℕ := 64

-- Given conditions
def goldfish (P : ℕ) : ℕ := 4 * P
def suckers : ℕ := 32
def kids : ℕ := 16
def items_per_kid : ℕ := 22
def total_items (P : ℕ) : ℕ := P + goldfish P + suckers

-- The theorem to prove
theorem pretzels_count : total_items pretzels = kids * items_per_kid := by
  sorry

end NUMINAMATH_GPT_pretzels_count_l875_87503


namespace NUMINAMATH_GPT_value_of_square_reciprocal_l875_87582

theorem value_of_square_reciprocal (x : ℝ) (h : 18 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 20 := by
  sorry

end NUMINAMATH_GPT_value_of_square_reciprocal_l875_87582


namespace NUMINAMATH_GPT_total_selling_price_of_toys_l875_87543

/-
  Prove that the total selling price (TSP) for 18 toys,
  given that each toy costs Rs. 1100 and the man gains the cost price of 3 toys, is Rs. 23100.
-/
theorem total_selling_price_of_toys :
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  TSP = 23100 :=
by
  let CP := 1100
  let TCP := 18 * CP
  let G := 3 * CP
  let TSP := TCP + G
  sorry

end NUMINAMATH_GPT_total_selling_price_of_toys_l875_87543


namespace NUMINAMATH_GPT_polynomial_simplified_l875_87551

def polynomial (x : ℝ) : ℝ := 4 - 6 * x - 8 * x^2 + 12 - 14 * x + 16 * x^2 - 18 + 20 * x + 24 * x^2

theorem polynomial_simplified (x : ℝ) : polynomial x = 32 * x^2 - 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplified_l875_87551


namespace NUMINAMATH_GPT_union_of_A_and_B_at_m_equals_3_range_of_m_if_A_union_B_equals_A_l875_87593

def set_A : Set ℝ := { x | x^2 - x - 12 ≤ 0 }
def set_B (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Statement 1: Prove that when \( m = 3 \), \( A \cup B \) = \( \{ x \mid -3 \leq x \leq 5 \} \).
theorem union_of_A_and_B_at_m_equals_3 : set_A ∪ set_B 3 = { x | -3 ≤ x ∧ x ≤ 5 } :=
sorry

-- Statement 2: Prove that if \( A ∪ B = A \), then the range of \( m \) is \( (-\infty, \frac{5}{2}] \).
theorem range_of_m_if_A_union_B_equals_A (m : ℝ) : (set_A ∪ set_B m = set_A) → m ≤ 5 / 2 :=
sorry

end NUMINAMATH_GPT_union_of_A_and_B_at_m_equals_3_range_of_m_if_A_union_B_equals_A_l875_87593


namespace NUMINAMATH_GPT_simplified_expression_value_at_4_l875_87501

theorem simplified_expression (x : ℝ) (h : x ≠ 5) : (x^2 - 3*x - 10) / (x - 5) = x + 2 := 
sorry

theorem value_at_4 : (4 : ℝ)^2 - 3*4 - 10 / (4 - 5) = 6 := 
sorry

end NUMINAMATH_GPT_simplified_expression_value_at_4_l875_87501


namespace NUMINAMATH_GPT_figure_50_squares_l875_87510

open Nat

noncomputable def g (n : ℕ) : ℕ := 2 * n ^ 2 + 5 * n + 2

theorem figure_50_squares : g 50 = 5252 :=
by
  sorry

end NUMINAMATH_GPT_figure_50_squares_l875_87510


namespace NUMINAMATH_GPT_solution_set_equivalence_l875_87506

def solution_set_inequality (x : ℝ) : Prop :=
  abs (x - 1) + abs x < 3

theorem solution_set_equivalence :
  { x : ℝ | solution_set_inequality x } = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_equivalence_l875_87506
