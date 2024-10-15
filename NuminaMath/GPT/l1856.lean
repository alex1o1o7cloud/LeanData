import Mathlib

namespace NUMINAMATH_GPT_perpendicular_lines_slope_l1856_185611

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, x + a * y = 1 - a ∧ (a - 2) * x + 3 * y + 2 = 0) → a = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_perpendicular_lines_slope_l1856_185611


namespace NUMINAMATH_GPT_trailing_zeros_310_factorial_l1856_185695

def count_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem trailing_zeros_310_factorial :
  count_trailing_zeros 310 = 76 := by
sorry

end NUMINAMATH_GPT_trailing_zeros_310_factorial_l1856_185695


namespace NUMINAMATH_GPT_tricycles_count_l1856_185648

-- Define the variables for number of bicycles, tricycles, and scooters.
variables (b t s : ℕ)

-- Define the total number of children and total number of wheels conditions.
def children_condition := b + t + s = 10
def wheels_condition := 2 * b + 3 * t + 2 * s = 27

-- Prove that number of tricycles t is 4 under these conditions.
theorem tricycles_count : children_condition b t s → wheels_condition b t s → t = 4 := by
  sorry

end NUMINAMATH_GPT_tricycles_count_l1856_185648


namespace NUMINAMATH_GPT_gcd_lcm_product_l1856_185691

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 3 * 5^2) (h2 : b = 5^3) : 
  Nat.gcd a b * Nat.lcm a b = 9375 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l1856_185691


namespace NUMINAMATH_GPT_each_shopper_will_receive_amount_l1856_185685

/-- Definitions of the given conditions -/
def isabella_has_more_than_sam : ℕ := 45
def isabella_has_more_than_giselle : ℕ := 15
def giselle_money : ℕ := 120

/-- Calculation based on the provided conditions -/
def isabella_money : ℕ := giselle_money + isabella_has_more_than_giselle
def sam_money : ℕ := isabella_money - isabella_has_more_than_sam
def total_money : ℕ := isabella_money + sam_money + giselle_money

/-- The total amount each shopper will receive when the donation is shared equally -/
def money_each_shopper_receives : ℕ := total_money / 3

/-- Main theorem to prove the statement derived from the problem -/
theorem each_shopper_will_receive_amount :
  money_each_shopper_receives = 115 := by
  sorry

end NUMINAMATH_GPT_each_shopper_will_receive_amount_l1856_185685


namespace NUMINAMATH_GPT_average_of_possible_values_l1856_185633

theorem average_of_possible_values 
  (x : ℝ)
  (h : Real.sqrt (2 * x^2 + 5) = Real.sqrt 25) : 
  (x = Real.sqrt 10 ∨ x = -Real.sqrt 10) → (Real.sqrt 10 + (-Real.sqrt 10)) / 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_average_of_possible_values_l1856_185633


namespace NUMINAMATH_GPT_dealer_profit_percentage_l1856_185661

-- Define the conditions
def cost_price_kg : ℕ := 1000
def given_weight_kg : ℕ := 575

-- Define the weight saved by the dealer
def weight_saved : ℕ := cost_price_kg - given_weight_kg

-- Define the profit percentage formula
def profit_percentage : ℕ → ℕ → ℚ := λ saved total_weight => (saved : ℚ) / (total_weight : ℚ) * 100

-- The main theorem statement
theorem dealer_profit_percentage : profit_percentage weight_saved cost_price_kg = 42.5 :=
by
  sorry

end NUMINAMATH_GPT_dealer_profit_percentage_l1856_185661


namespace NUMINAMATH_GPT_sequence_an_solution_l1856_185684

theorem sequence_an_solution {a : ℕ → ℝ} (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → (1 / a (n + 1) = 1 / a n + 1)) : ∀ n : ℕ, 0 < n → (a n = 1 / n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_an_solution_l1856_185684


namespace NUMINAMATH_GPT_cos_sum_eq_one_l1856_185663

theorem cos_sum_eq_one (α β γ : ℝ) 
  (h1 : α + β + γ = Real.pi) 
  (h2 : Real.tan ((β + γ - α) / 4) + Real.tan ((γ + α - β) / 4) + Real.tan ((α + β - γ) / 4) = 1) :
  Real.cos α + Real.cos β + Real.cos γ = 1 :=
sorry

end NUMINAMATH_GPT_cos_sum_eq_one_l1856_185663


namespace NUMINAMATH_GPT_trigonometric_relationship_l1856_185686

theorem trigonometric_relationship (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = (1 - Real.sin β) / Real.cos β) : 
  2 * α + β = π / 2 := 
sorry

end NUMINAMATH_GPT_trigonometric_relationship_l1856_185686


namespace NUMINAMATH_GPT_triangle_base_second_l1856_185601

theorem triangle_base_second (base1 height1 height2 : ℝ) 
  (h_base1 : base1 = 15) (h_height1 : height1 = 12) (h_height2 : height2 = 18) :
  let area1 := (base1 * height1) / 2
  let area2 := 2 * area1
  let base2 := (2 * area2) / height2
  base2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_second_l1856_185601


namespace NUMINAMATH_GPT_fraction_pairs_l1856_185606

theorem fraction_pairs (n : ℕ) (h : n > 2009) : 
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 1 ≤ a ∧ a ≤ n ∧
  1 ≤ b ∧ b ≤ n ∧ 1 ≤ c ∧ c ≤ n ∧ 1 ≤ d ∧ d ≤ n ∧
  1/a + 1/b = 1/c + 1/d := 
sorry

end NUMINAMATH_GPT_fraction_pairs_l1856_185606


namespace NUMINAMATH_GPT_shaded_region_area_is_48pi_l1856_185620

open Real

noncomputable def small_circle_radius : ℝ := 4
noncomputable def small_circle_area : ℝ := π * small_circle_radius^2
noncomputable def large_circle_radius : ℝ := 2 * small_circle_radius
noncomputable def large_circle_area : ℝ := π * large_circle_radius^2
noncomputable def shaded_region_area : ℝ := large_circle_area - small_circle_area

theorem shaded_region_area_is_48pi :
  shaded_region_area = 48 * π := by
    sorry

end NUMINAMATH_GPT_shaded_region_area_is_48pi_l1856_185620


namespace NUMINAMATH_GPT_part_1_part_2_l1856_185675

-- Conditions and definitions
noncomputable def triangle_ABC (a b c S : ℝ) (A B C : ℝ) :=
  a * Real.sin B = -b * Real.sin (A + Real.pi / 3) ∧
  S = Real.sqrt 3 / 4 * c^2

-- 1. Prove A = 5 * Real.pi / 6
theorem part_1 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  A = 5 * Real.pi / 6 :=
  sorry

-- 2. Prove sin C = sqrt 7 / 14 given S = sqrt 3 / 4 * c^2
theorem part_2 (a b c S A B C : ℝ) (h : triangle_ABC a b c S A B C) :
  Real.sin C = Real.sqrt 7 / 14 :=
  sorry

end NUMINAMATH_GPT_part_1_part_2_l1856_185675


namespace NUMINAMATH_GPT_circle_equation_l1856_185673

theorem circle_equation :
  ∃ (a : ℝ) (x y : ℝ), 
    (2 * a + y - 1 = 0 ∧ (x = 3 ∧ y = 0) ∧ (x = 0 ∧ y = 1)) →
    (x - 1) ^ 2 + (y + 1) ^ 2 = 5 := by
  sorry

end NUMINAMATH_GPT_circle_equation_l1856_185673


namespace NUMINAMATH_GPT_multiply_repeating_decimals_l1856_185658

noncomputable def repeating_decimal_03 : ℚ := 1 / 33
noncomputable def repeating_decimal_8 : ℚ := 8 / 9

theorem multiply_repeating_decimals : repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by 
  sorry

end NUMINAMATH_GPT_multiply_repeating_decimals_l1856_185658


namespace NUMINAMATH_GPT_meaningful_sqrt_range_l1856_185654

theorem meaningful_sqrt_range (x : ℝ) (h : x - 1 ≥ 0) : x ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_meaningful_sqrt_range_l1856_185654


namespace NUMINAMATH_GPT_zero_in_interval_l1856_185607

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 2

theorem zero_in_interval : f 1 < 0 ∧ f 2 > 0 → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_zero_in_interval_l1856_185607


namespace NUMINAMATH_GPT_solve_system_l1856_185638

theorem solve_system (x y z : ℝ) 
  (h1 : 19 * (x + y) + 17 = 19 * (-x + y) - 21)
  (h2 : 5 * x - 3 * z = 11 * y - 7) : 
  x = -1 ∧ z = -11 * y / 3 + 2 / 3 :=
by sorry

end NUMINAMATH_GPT_solve_system_l1856_185638


namespace NUMINAMATH_GPT_outfit_count_l1856_185624

theorem outfit_count 
  (S P T J : ℕ) 
  (hS : S = 8) 
  (hP : P = 5) 
  (hT : T = 4) 
  (hJ : J = 3) : 
  S * P * (T + 1) * (J + 1) = 800 := by 
  sorry

end NUMINAMATH_GPT_outfit_count_l1856_185624


namespace NUMINAMATH_GPT_kims_morning_routine_total_time_l1856_185676

def time_spent_making_coffee := 5 -- in minutes
def time_spent_per_employee_status_update := 2 -- in minutes
def time_spent_per_employee_payroll_update := 3 -- in minutes
def number_of_employees := 9

theorem kims_morning_routine_total_time :
  time_spent_making_coffee +
  (time_spent_per_employee_status_update + time_spent_per_employee_payroll_update) * number_of_employees = 50 :=
by
  sorry

end NUMINAMATH_GPT_kims_morning_routine_total_time_l1856_185676


namespace NUMINAMATH_GPT_inequality_not_true_l1856_185600

theorem inequality_not_true (a b : ℝ) (h : a > b) : (a / (-2)) ≤ (b / (-2)) :=
sorry

end NUMINAMATH_GPT_inequality_not_true_l1856_185600


namespace NUMINAMATH_GPT_quadratic_root_properties_l1856_185669

theorem quadratic_root_properties (b : ℝ) (t : ℝ) :
  (∀ x : ℝ, x^2 + b*x - 2 = 0 → (x = 2 ∨ x = t)) →
  b = -1 ∧ t = -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_properties_l1856_185669


namespace NUMINAMATH_GPT_fractional_addition_l1856_185688

theorem fractional_addition : (2 : ℚ) / 5 + 3 / 8 = 31 / 40 :=
by
  sorry

end NUMINAMATH_GPT_fractional_addition_l1856_185688


namespace NUMINAMATH_GPT_gcd_4557_1953_5115_l1856_185666

theorem gcd_4557_1953_5115 : Nat.gcd (Nat.gcd 4557 1953) 5115 = 93 :=
by
  -- We use 'sorry' to skip the proof part as per the instructions.
  sorry

end NUMINAMATH_GPT_gcd_4557_1953_5115_l1856_185666


namespace NUMINAMATH_GPT_sum_divisible_by_5_and_7_l1856_185698

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_divisible_by_5_and_7 (A B : ℕ) (hA_prime : is_prime A) 
  (hB_prime : is_prime B) (hA_minus_3_prime : is_prime (A - 3)) 
  (hA_plus_3_prime : is_prime (A + 3)) (hB_eq_2 : B = 2) : 
  5 ∣ (A + B + (A - 3) + (A + 3)) ∧ 7 ∣ (A + B + (A - 3) + (A + 3)) := by 
  sorry

end NUMINAMATH_GPT_sum_divisible_by_5_and_7_l1856_185698


namespace NUMINAMATH_GPT_probability_two_most_expensive_l1856_185631

open Nat

noncomputable def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem probability_two_most_expensive :
  (combination 8 1) / (combination 10 3) = 1 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_most_expensive_l1856_185631


namespace NUMINAMATH_GPT_range_of_x_l1856_185618

noncomputable def f : ℝ → ℝ := sorry -- Define the function f

variable (f_increasing : ∀ x y, x < y → f x < f y) -- f is increasing
variable (f_at_2 : f 2 = 0) -- f(2) = 0

theorem range_of_x (x : ℝ) : f (x - 2) > 0 ↔ x > 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1856_185618


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1856_185696

def condition1 (a b : ℝ) : Prop :=
  a > b

def statement (a b : ℝ) : Prop :=
  a > b + 1

theorem necessary_but_not_sufficient (a b : ℝ) (h : condition1 a b) : 
  (∀ a b : ℝ, statement a b → condition1 a b) ∧ ¬ (∀ a b : ℝ, condition1 a b → statement a b) :=
by 
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1856_185696


namespace NUMINAMATH_GPT_hall_volume_l1856_185662

theorem hall_volume (length width : ℝ) (h : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 6) 
  (h_areas : 2 * (length * width) = 4 * (length * h)) :
  length * width * h = 108 :=
by
  sorry

end NUMINAMATH_GPT_hall_volume_l1856_185662


namespace NUMINAMATH_GPT_system_of_equations_solution_l1856_185665

theorem system_of_equations_solution :
  ∃ (a b : ℤ), (2 * (2 : ℤ) + b = a ∧ (2 : ℤ) + b = 3 ∧ a = 5 ∧ b = 1) :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1856_185665


namespace NUMINAMATH_GPT_payment_ways_l1856_185690

-- Define basic conditions and variables
variables {x y z : ℕ}

-- Define the main problem as a Lean statement
theorem payment_ways : 
  ∃ (n : ℕ), n = 9 ∧ 
             (∀ x y z : ℕ, 
              x + y + z ≤ 10 ∧ 
              x + 2 * y + 5 * z = 18 ∧ 
              x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ 
              (x > 0 ∨ y > 0) ∧ (y > 0 ∨ z > 0) ∧ (z > 0 ∨ x > 0) → 
              n = 9) := 
sorry

end NUMINAMATH_GPT_payment_ways_l1856_185690


namespace NUMINAMATH_GPT_inclination_angle_of_line_l1856_185644

theorem inclination_angle_of_line (θ : Real) 
  (h : θ = Real.tan 45) : θ = 90 :=
sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l1856_185644


namespace NUMINAMATH_GPT_extra_games_needed_l1856_185615

def initial_games : ℕ := 500
def initial_success_rate : ℚ := 0.49
def target_success_rate : ℚ := 0.5

theorem extra_games_needed :
  ∀ (x : ℕ),
  (245 + x) / (initial_games + x) = target_success_rate → x = 10 := 
by
  sorry

end NUMINAMATH_GPT_extra_games_needed_l1856_185615


namespace NUMINAMATH_GPT_find_x3_minus_y3_l1856_185610

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end NUMINAMATH_GPT_find_x3_minus_y3_l1856_185610


namespace NUMINAMATH_GPT_exists_word_D_l1856_185632

variable {α : Type} [Inhabited α] [DecidableEq α]

def repeats (D : List α) (w : List α) : Prop :=
  ∃ k : ℕ, w = List.join (List.replicate k D)

theorem exists_word_D (A B C : List α)
  (h : (A ++ A ++ B ++ B) = (C ++ C)) :
  ∃ D : List α, repeats D A ∧ repeats D B ∧ repeats D C :=
sorry

end NUMINAMATH_GPT_exists_word_D_l1856_185632


namespace NUMINAMATH_GPT_fraction_neither_cable_nor_vcr_l1856_185692

variable (T : ℕ)
variable (units_with_cable : ℕ := T / 5)
variable (units_with_vcrs : ℕ := T / 10)
variable (units_with_cable_and_vcrs : ℕ := (T / 5) / 3)

theorem fraction_neither_cable_nor_vcr (T : ℕ)
  (h1 : units_with_cable = T / 5)
  (h2 : units_with_vcrs = T / 10)
  (h3 : units_with_cable_and_vcrs = (units_with_cable / 3)) :
  (T - (units_with_cable + (units_with_vcrs - units_with_cable_and_vcrs))) / T = 7 / 10 := 
by
  sorry

end NUMINAMATH_GPT_fraction_neither_cable_nor_vcr_l1856_185692


namespace NUMINAMATH_GPT_no_odd_m_solution_l1856_185652

theorem no_odd_m_solution : ∀ (m n : ℕ), 0 < m → 0 < n → (5 * n = m * n - 3 * m) → ¬ Odd m :=
by
  intros m n hm hn h_eq
  sorry

end NUMINAMATH_GPT_no_odd_m_solution_l1856_185652


namespace NUMINAMATH_GPT_rosalina_received_21_gifts_l1856_185693

def Emilio_gifts : Nat := 11
def Jorge_gifts : Nat := 6
def Pedro_gifts : Nat := 4

def total_gifts : Nat :=
  Emilio_gifts + Jorge_gifts + Pedro_gifts

theorem rosalina_received_21_gifts : total_gifts = 21 := by
  sorry

end NUMINAMATH_GPT_rosalina_received_21_gifts_l1856_185693


namespace NUMINAMATH_GPT_number_of_periods_l1856_185670

-- Definitions based on conditions
def students : ℕ := 32
def time_per_student : ℕ := 5
def period_duration : ℕ := 40

-- Theorem stating the equivalent proof problem
theorem number_of_periods :
  (students * time_per_student) / period_duration = 4 :=
sorry

end NUMINAMATH_GPT_number_of_periods_l1856_185670


namespace NUMINAMATH_GPT_consecutive_numbers_sum_digits_l1856_185621

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem consecutive_numbers_sum_digits :
  ∃ n : ℕ, sum_of_digits n = 52 ∧ sum_of_digits (n + 4) = 20 := 
sorry

end NUMINAMATH_GPT_consecutive_numbers_sum_digits_l1856_185621


namespace NUMINAMATH_GPT_range_of_sum_abs_l1856_185647

variable {x y z : ℝ}

theorem range_of_sum_abs : 
  x^2 + y^2 + z = 15 → 
  x + y + z^2 = 27 → 
  xy + yz + zx = 7 → 
  7 ≤ |x + y + z| ∧ |x + y + z| ≤ 8 := by
  sorry

end NUMINAMATH_GPT_range_of_sum_abs_l1856_185647


namespace NUMINAMATH_GPT_cube_properties_l1856_185628

theorem cube_properties (y : ℝ) (s : ℝ) 
  (h_volume : s^3 = 6 * y)
  (h_surface_area : 6 * s^2 = 2 * y) :
  y = 5832 :=
by sorry

end NUMINAMATH_GPT_cube_properties_l1856_185628


namespace NUMINAMATH_GPT_good_subset_divisible_by_5_l1856_185634

noncomputable def num_good_subsets : ℕ :=
  (Nat.factorial 1000) / ((Nat.factorial 201) * (Nat.factorial (1000 - 201)))

theorem good_subset_divisible_by_5 : num_good_subsets / 5 = (1 / 5) * num_good_subsets := 
sorry

end NUMINAMATH_GPT_good_subset_divisible_by_5_l1856_185634


namespace NUMINAMATH_GPT_min_value_expression_l1856_185640

theorem min_value_expression (a b : ℝ) : ∃ v : ℝ, ∀ (a b : ℝ), (a^2 + a * b + b^2 - a - 2 * b) ≥ v ∧ v = -1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1856_185640


namespace NUMINAMATH_GPT_distance_focus_directrix_l1856_185616

theorem distance_focus_directrix (y x p : ℝ) (h : y^2 = 4 * x) (hp : 2 * p = 4) : p = 2 :=
by sorry

end NUMINAMATH_GPT_distance_focus_directrix_l1856_185616


namespace NUMINAMATH_GPT_flour_needed_l1856_185671

theorem flour_needed (flour_per_24_cookies : ℝ) (cookies_per_recipe : ℕ) (desired_cookies : ℕ) 
  (h : flour_per_24_cookies = 1.5) (h1 : cookies_per_recipe = 24) (h2 : desired_cookies = 72) : 
  flour_per_24_cookies / cookies_per_recipe * desired_cookies = 4.5 := 
  by {
    -- The proof is omitted
    sorry
  }

end NUMINAMATH_GPT_flour_needed_l1856_185671


namespace NUMINAMATH_GPT_minimum_number_of_circles_l1856_185687

-- Define the problem conditions
def conditions_of_problem (circles : ℕ) (n : ℕ) (highlighted_lines : ℕ) (sides_of_regular_2011_gon : ℕ) : Prop :=
  circles ≥ n ∧ highlighted_lines = sides_of_regular_2011_gon

-- The main theorem we need to prove
theorem minimum_number_of_circles :
  ∀ (n circles highlighted_lines sides_of_regular_2011_gon : ℕ),
    sides_of_regular_2011_gon = 2011 ∧ (highlighted_lines = sides_of_regular_2011_gon * 2) ∧ conditions_of_problem circles n highlighted_lines sides_of_regular_2011_gon → n = 504 :=
by
  sorry

end NUMINAMATH_GPT_minimum_number_of_circles_l1856_185687


namespace NUMINAMATH_GPT_video_files_initial_l1856_185612

theorem video_files_initial (V : ℕ) (h1 : 4 + V - 23 = 2) : V = 21 :=
by 
  sorry

end NUMINAMATH_GPT_video_files_initial_l1856_185612


namespace NUMINAMATH_GPT_jessica_can_mail_letter_l1856_185629

-- Define the constants
def paper_weight := 1/5 -- each piece of paper weighs 1/5 ounce
def envelope_weight := 2/5 -- envelope weighs 2/5 ounce
def num_papers := 8

-- Calculate the total weight
def total_weight := num_papers * paper_weight + envelope_weight

-- Define stamping rates
def international_rate := 2 -- $2 per ounce internationally

-- Calculate the required postage
def required_postage := total_weight * international_rate

-- Define the available stamp values
inductive Stamp
| one_dollar : Stamp
| fifty_cents : Stamp

-- Function to calculate the total value of a given stamp combination
def stamp_value : List Stamp → ℝ
| [] => 0
| (Stamp.one_dollar :: rest) => 1 + stamp_value rest
| (Stamp.fifty_cents :: rest) => 0.5 + stamp_value rest

-- State the theorem to be proved
theorem jessica_can_mail_letter :
  ∃ stamps : List Stamp, stamp_value stamps = required_postage := by
sorry

end NUMINAMATH_GPT_jessica_can_mail_letter_l1856_185629


namespace NUMINAMATH_GPT_cashback_percentage_l1856_185623

theorem cashback_percentage
  (total_cost : ℝ) (rebate : ℝ) (final_cost : ℝ)
  (H1 : total_cost = 150) (H2 : rebate = 25) (H3 : final_cost = 110) :
  (total_cost - rebate - final_cost) / (total_cost - rebate) * 100 = 12 := by
  sorry

end NUMINAMATH_GPT_cashback_percentage_l1856_185623


namespace NUMINAMATH_GPT_find_ratio_of_hyperbola_asymptotes_l1856_185677

theorem find_ratio_of_hyperbola_asymptotes (a b : ℝ) (h : a > b) (hyp : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → |(2 * b / a)| = 1) : 
  a / b = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_ratio_of_hyperbola_asymptotes_l1856_185677


namespace NUMINAMATH_GPT_roots_greater_than_two_l1856_185604

variable {x m : ℝ}

theorem roots_greater_than_two (h : ∀ x, x^2 - 2 * m * x + 4 = 0 → (∃ a b : ℝ, a > 2 ∧ b < 2 ∧ x = a ∨ x = b)) : 
  m > 2 :=
by
  sorry

end NUMINAMATH_GPT_roots_greater_than_two_l1856_185604


namespace NUMINAMATH_GPT_gcd_765432_654321_l1856_185619

-- Define the two integers 765432 and 654321
def a : ℕ := 765432
def b : ℕ := 654321

-- State the main theorem to prove the gcd
theorem gcd_765432_654321 : Nat.gcd a b = 3 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_765432_654321_l1856_185619


namespace NUMINAMATH_GPT_f_neg_def_l1856_185657

variable (f : ℝ → ℝ)
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_def_pos : ∀ x : ℝ, 0 < x → f x = x * (1 + x)

theorem f_neg_def (x : ℝ) (hx : x < 0) : f x = x * (1 - x) := by
  sorry

end NUMINAMATH_GPT_f_neg_def_l1856_185657


namespace NUMINAMATH_GPT_div_40_of_prime_ge7_l1856_185626

theorem div_40_of_prime_ge7 (p : ℕ) (hp_prime : Prime p) (hp_ge7 : p ≥ 7) : 40 ∣ (p^2 - 1) :=
sorry

end NUMINAMATH_GPT_div_40_of_prime_ge7_l1856_185626


namespace NUMINAMATH_GPT_alpha_beta_sum_l1856_185680

theorem alpha_beta_sum (α β : ℝ) (h1 : α^3 - 3 * α^2 + 5 * α - 17 = 0) (h2 : β^3 - 3 * β^2 + 5 * β + 11 = 0) : α + β = 2 := 
by
  sorry

end NUMINAMATH_GPT_alpha_beta_sum_l1856_185680


namespace NUMINAMATH_GPT_count_total_legs_l1856_185649

theorem count_total_legs :
  let tables4 := 4 * 4
  let sofa := 1 * 4
  let chairs4 := 2 * 4
  let tables3 := 3 * 3
  let table1 := 1 * 1
  let rocking_chair := 1 * 2
  let total_legs := tables4 + sofa + chairs4 + tables3 + table1 + rocking_chair
  total_legs = 40 :=
by
  sorry

end NUMINAMATH_GPT_count_total_legs_l1856_185649


namespace NUMINAMATH_GPT_tree_distance_l1856_185694

theorem tree_distance 
  (num_trees : ℕ) (dist_first_to_fifth : ℕ) (length_of_road : ℤ) 
  (h1 : num_trees = 8) 
  (h2 : dist_first_to_fifth = 100) 
  (h3 : length_of_road = (dist_first_to_fifth * (num_trees - 1)) / 4 + 3 * dist_first_to_fifth) 
  :
  length_of_road = 175 := 
sorry

end NUMINAMATH_GPT_tree_distance_l1856_185694


namespace NUMINAMATH_GPT_fraction_value_l1856_185668

theorem fraction_value : (1 + 3 + 5) / (10 + 6 + 2) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_fraction_value_l1856_185668


namespace NUMINAMATH_GPT_andy_wrong_questions_l1856_185659

variables (a b c d : ℕ)

theorem andy_wrong_questions 
  (h1 : a + b = c + d) 
  (h2 : a + d = b + c + 6) 
  (h3 : c = 7) : 
  a = 20 :=
sorry

end NUMINAMATH_GPT_andy_wrong_questions_l1856_185659


namespace NUMINAMATH_GPT_shaded_area_is_correct_l1856_185642

theorem shaded_area_is_correct : 
  ∀ (leg_length : ℕ) (total_partitions : ℕ) (shaded_partitions : ℕ) 
    (tri_area : ℕ) (small_tri_area : ℕ) (shaded_area : ℕ), 
  leg_length = 10 → 
  total_partitions = 25 →
  shaded_partitions = 15 →
  tri_area = (1 / 2 * leg_length * leg_length) → 
  small_tri_area = (tri_area / total_partitions) →
  shaded_area = (shaded_partitions * small_tri_area) →
  shaded_area = 30 :=
by
  intros leg_length total_partitions shaded_partitions tri_area small_tri_area shaded_area
  intros h_leg_length h_total_partitions h_shaded_partitions h_tri_area h_small_tri_area h_shaded_area
  sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l1856_185642


namespace NUMINAMATH_GPT_interior_angle_second_quadrant_l1856_185653

theorem interior_angle_second_quadrant (α : ℝ) (h1 : 0 < α ∧ α < π) (h2 : Real.sin α * Real.tan α < 0) : 
  π / 2 < α ∧ α < π :=
by
  sorry

end NUMINAMATH_GPT_interior_angle_second_quadrant_l1856_185653


namespace NUMINAMATH_GPT_f_f_f_f_f_3_eq_4_l1856_185602

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x + 1

theorem f_f_f_f_f_3_eq_4 : f (f (f (f (f 3)))) = 4 := 
  sorry

end NUMINAMATH_GPT_f_f_f_f_f_3_eq_4_l1856_185602


namespace NUMINAMATH_GPT_point_within_region_l1856_185678

theorem point_within_region (a : ℝ) (h : 2 * a + 2 < 4) : a < 1 := 
sorry

end NUMINAMATH_GPT_point_within_region_l1856_185678


namespace NUMINAMATH_GPT_complete_square_rewrite_l1856_185682

theorem complete_square_rewrite (j i : ℂ) :
  let c := 8
  let p := (3 * i / 8 : ℂ)
  let q := (137 / 8 : ℂ)
  (8 * j^2 + 6 * i * j + 16 = c * (j + p)^2 + q) →
  q / p = - (137 * i / 3) :=
by
  sorry

end NUMINAMATH_GPT_complete_square_rewrite_l1856_185682


namespace NUMINAMATH_GPT_problem_solution_l1856_185625

theorem problem_solution (a : ℝ) : 
  ( ∀ x : ℝ, (ax - 1) * (x + 1) < 0 ↔ (x ∈ Set.Iio (-1) ∨ x ∈ Set.Ioi (-1 / 2)) ) →
  a = -2 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1856_185625


namespace NUMINAMATH_GPT_max_rides_day1_max_rides_day2_l1856_185674

open List 

def daily_budget : ℤ := 10

def ride_prices_day1 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 5), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6)]

def ride_prices_day2 : List (String × ℤ) := 
  [("Ferris wheel", 4), ("Roller coaster", 7), ("Bumper cars", 3), ("Carousel", 2), ("Log flume", 6), ("Haunted house", 4)]

def max_rides (budget : ℤ) (prices : List (String × ℤ)) : ℤ :=
  sorry -- We'll assume this calculates the max number of rides correctly based on the given budget and prices.

theorem max_rides_day1 : max_rides daily_budget ride_prices_day1 = 3 := by
  sorry 

theorem max_rides_day2 : max_rides daily_budget ride_prices_day2 = 3 := by
  sorry 

end NUMINAMATH_GPT_max_rides_day1_max_rides_day2_l1856_185674


namespace NUMINAMATH_GPT_find_n_coordinates_l1856_185630

variables {a b : ℝ}

def is_perpendicular (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_n_coordinates (n : ℝ × ℝ) (h1 : is_perpendicular (a, b) n) (h2 : same_magnitude (a, b) n) :
  n = (b, -a) :=
sorry

end NUMINAMATH_GPT_find_n_coordinates_l1856_185630


namespace NUMINAMATH_GPT_relationship_among_vars_l1856_185655

theorem relationship_among_vars {a b c d : ℝ} (h : (a + 2 * b) / (b + 2 * c) = (c + 2 * d) / (d + 2 * a)) :
  b = 2 * a ∨ a + b + c + d = 0 :=
sorry

end NUMINAMATH_GPT_relationship_among_vars_l1856_185655


namespace NUMINAMATH_GPT_B_and_C_mutually_exclusive_but_not_complementary_l1856_185651

-- Define the sample space of the cube
def faces : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define events based on conditions
def event_A (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5
def event_B (n : ℕ) : Prop := n = 1 ∨ n = 2
def event_C (n : ℕ) : Prop := n = 4 ∨ n = 5 ∨ n = 6

-- Define mutually exclusive events
def mutually_exclusive (A B : ℕ → Prop) : Prop := ∀ n, A n → ¬ B n

-- Define complementary events (for events over finite sample spaces like faces)
-- Events A and B are complementary if they partition the sample space faces
def complementary (A B : ℕ → Prop) : Prop := (∀ n, n ∈ faces → A n ∨ B n) ∧ (∀ n, A n → ¬ B n) ∧ (∀ n, B n → ¬ A n)

theorem B_and_C_mutually_exclusive_but_not_complementary :
  mutually_exclusive event_B event_C ∧ ¬ complementary event_B event_C := 
by
  sorry

end NUMINAMATH_GPT_B_and_C_mutually_exclusive_but_not_complementary_l1856_185651


namespace NUMINAMATH_GPT_original_price_l1856_185608

theorem original_price (P : ℝ) (h1 : 0.76 * P = 820) : P = 1079 :=
by
  sorry

end NUMINAMATH_GPT_original_price_l1856_185608


namespace NUMINAMATH_GPT_tully_twice_kate_in_three_years_l1856_185627

-- Definitions for the conditions
def tully_was := 60
def kate_is := 29

-- Number of years from now when Tully will be twice as old as Kate
theorem tully_twice_kate_in_three_years : 
  ∃ (x : ℕ), (tully_was + 1 + x = 2 * (kate_is + x)) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_tully_twice_kate_in_three_years_l1856_185627


namespace NUMINAMATH_GPT_skateboarder_speed_l1856_185614

theorem skateboarder_speed :
  let distance := 293.33
  let time := 20
  let feet_per_mile := 5280
  let seconds_per_hour := 3600
  let speed_ft_per_sec := distance / time
  let speed_mph := speed_ft_per_sec * (feet_per_mile / seconds_per_hour)
  speed_mph = 21.5 :=
by
  sorry

end NUMINAMATH_GPT_skateboarder_speed_l1856_185614


namespace NUMINAMATH_GPT_hyperbola_center_l1856_185643

theorem hyperbola_center 
  (x y : ℝ)
  (h : 9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0) : 
  x = 2 ∧ y = 4 :=
sorry

end NUMINAMATH_GPT_hyperbola_center_l1856_185643


namespace NUMINAMATH_GPT_percentage_of_candidates_selected_in_State_A_is_6_l1856_185664

-- Definitions based on conditions
def candidates_appeared : ℕ := 8400
def candidates_selected_B : ℕ := (7 * candidates_appeared) / 100 -- 7% of 8400
def extra_candidates_selected : ℕ := 84
def candidates_selected_A : ℕ := candidates_selected_B - extra_candidates_selected

-- Definition based on the goal proof
def percentage_selected_A : ℕ := (candidates_selected_A * 100) / candidates_appeared

-- The theorem we need to prove
theorem percentage_of_candidates_selected_in_State_A_is_6 :
  percentage_selected_A = 6 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_candidates_selected_in_State_A_is_6_l1856_185664


namespace NUMINAMATH_GPT_ladybugs_calculation_l1856_185609

def total_ladybugs : ℕ := 67082
def ladybugs_with_spots : ℕ := 12170
def ladybugs_without_spots : ℕ := 54912

theorem ladybugs_calculation :
  total_ladybugs - ladybugs_with_spots = ladybugs_without_spots :=
by
  sorry

end NUMINAMATH_GPT_ladybugs_calculation_l1856_185609


namespace NUMINAMATH_GPT_half_of_1_point_6_times_10_pow_6_l1856_185689

theorem half_of_1_point_6_times_10_pow_6 : (1.6 * 10^6) / 2 = 8 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_half_of_1_point_6_times_10_pow_6_l1856_185689


namespace NUMINAMATH_GPT_terminal_side_in_second_quadrant_l1856_185667

theorem terminal_side_in_second_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.tan α < 0) : 
    0 < α ∧ α < π :=
sorry

end NUMINAMATH_GPT_terminal_side_in_second_quadrant_l1856_185667


namespace NUMINAMATH_GPT_sin_squared_plus_sin_double_eq_one_l1856_185650

variable (α : ℝ)
variable (h : Real.tan α = 1 / 2)

theorem sin_squared_plus_sin_double_eq_one : Real.sin α ^ 2 + Real.sin (2 * α) = 1 :=
by
  -- sorry to indicate the proof is skipped
  sorry

end NUMINAMATH_GPT_sin_squared_plus_sin_double_eq_one_l1856_185650


namespace NUMINAMATH_GPT_find_p_from_circle_and_parabola_tangency_l1856_185605

theorem find_p_from_circle_and_parabola_tangency :
  (∃ x y : ℝ, (x^2 + y^2 - 6*x - 7 = 0) ∧ (y^2 = 2*p * x) ∧ p > 0) →
  p = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_p_from_circle_and_parabola_tangency_l1856_185605


namespace NUMINAMATH_GPT_largest_circle_radius_l1856_185635

theorem largest_circle_radius 
  (h H : ℝ) (h_pos : h > 0) (H_pos : H > 0) :
  ∃ R, R = (h * H) / (h + H) :=
sorry

end NUMINAMATH_GPT_largest_circle_radius_l1856_185635


namespace NUMINAMATH_GPT_relative_error_comparison_l1856_185660

theorem relative_error_comparison :
  (0.05 / 25 = 0.002) ∧ (0.4 / 200 = 0.002) → (0.002 = 0.002) :=
by
  sorry

end NUMINAMATH_GPT_relative_error_comparison_l1856_185660


namespace NUMINAMATH_GPT_age_difference_l1856_185641

theorem age_difference (M T J X S : ℕ)
  (hM : M = 3)
  (hT : T = 4 * M)
  (hJ : J = T - 5)
  (hX : X = 2 * J)
  (hS : S = 3 * X - 1) :
  S - M = 38 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l1856_185641


namespace NUMINAMATH_GPT_Nora_to_Lulu_savings_ratio_l1856_185681

-- Definitions
def L : ℕ := 6
def T (N : ℕ) : Prop := N = 3 * (N / 3)
def total_savings (N : ℕ) : Prop := 6 + N + (N / 3) = 46

-- Theorem statement
theorem Nora_to_Lulu_savings_ratio (N : ℕ) (hN_T : T N) (h_total_savings : total_savings N) :
  N / L = 5 :=
by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_Nora_to_Lulu_savings_ratio_l1856_185681


namespace NUMINAMATH_GPT_max_value_7x_10y_z_l1856_185613

theorem max_value_7x_10y_z (x y z : ℝ) 
  (h : x^2 + 2 * x + (1 / 5) * y^2 + 7 * z^2 = 6) : 
  7 * x + 10 * y + z ≤ 55 := 
sorry

end NUMINAMATH_GPT_max_value_7x_10y_z_l1856_185613


namespace NUMINAMATH_GPT_coordinates_of_point_A_in_third_quadrant_l1856_185603

def point_in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def distance_to_x_axis (y : ℝ) : ℝ := abs y

def distance_to_y_axis (x : ℝ) : ℝ := abs x

theorem coordinates_of_point_A_in_third_quadrant 
  (x y : ℝ)
  (h1 : point_in_third_quadrant x y)
  (h2 : distance_to_x_axis y = 2)
  (h3 : distance_to_y_axis x = 3) :
  (x, y) = (-3, -2) :=
  sorry

end NUMINAMATH_GPT_coordinates_of_point_A_in_third_quadrant_l1856_185603


namespace NUMINAMATH_GPT_acid_solution_l1856_185697

theorem acid_solution (m x : ℝ) (h1 : 0 < m) (h2 : m > 50)
  (h3 : (m / 100) * m = (m - 20) / 100 * (m + x)) : x = 20 * m / (m + 20) := 
sorry

end NUMINAMATH_GPT_acid_solution_l1856_185697


namespace NUMINAMATH_GPT_add_fractions_l1856_185672

theorem add_fractions :
  (8:ℚ) / 19 + 5 / 57 = 29 / 57 :=
sorry

end NUMINAMATH_GPT_add_fractions_l1856_185672


namespace NUMINAMATH_GPT_average_fuel_efficiency_l1856_185622

theorem average_fuel_efficiency (d1 d2 : ℝ) (e1 e2 : ℝ) (fuel1 fuel2 : ℝ)
  (h1 : d1 = 150) (h2 : e1 = 35) (h3 : d2 = 180) (h4 : e2 = 18)
  (h_fuel1 : fuel1 = d1 / e1) (h_fuel2 : fuel2 = d2 / e2)
  (total_distance : ℝ := 330)
  (total_fuel : ℝ := fuel1 + fuel2) :
  total_distance / total_fuel = 23 := by
  sorry

end NUMINAMATH_GPT_average_fuel_efficiency_l1856_185622


namespace NUMINAMATH_GPT_total_animals_l1856_185645

namespace Zoo

def snakes := 15
def monkeys := 2 * snakes
def lions := monkeys - 5
def pandas := lions + 8
def dogs := pandas / 3

theorem total_animals : snakes + monkeys + lions + pandas + dogs = 114 := by
  -- definitions from conditions
  have h_snakes : snakes = 15 := rfl
  have h_monkeys : monkeys = 2 * snakes := rfl
  have h_lions : lions = monkeys - 5 := rfl
  have h_pandas : pandas = lions + 8 := rfl
  have h_dogs : dogs = pandas / 3 := rfl
  -- sorry is used as a placeholder for the proof
  sorry

end Zoo

end NUMINAMATH_GPT_total_animals_l1856_185645


namespace NUMINAMATH_GPT_find_f_8_5_l1856_185646

-- Conditions as definitions in Lean
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def segment_function (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 3 * x

-- The main theorem to prove
theorem find_f_8_5 (f : ℝ → ℝ) (h1 : even_function f) (h2 : periodic_function f 3) (h3 : segment_function f)
: f 8.5 = 1.5 :=
sorry

end NUMINAMATH_GPT_find_f_8_5_l1856_185646


namespace NUMINAMATH_GPT_M_k_max_l1856_185679

noncomputable def J_k (k : ℕ) : ℕ := 5^(k+3) * 2^(k+3) + 648

def M (k : ℕ) : ℕ := 
  if k < 3 then k + 3
  else 3

theorem M_k_max (k : ℕ) : M k = 3 :=
by sorry

end NUMINAMATH_GPT_M_k_max_l1856_185679


namespace NUMINAMATH_GPT_solve_g_eq_g_inv_l1856_185699

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5

noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem solve_g_eq_g_inv : 
  ∃ x : ℝ, g x = g_inv x ∧ x = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_g_eq_g_inv_l1856_185699


namespace NUMINAMATH_GPT_part1_part2_l1856_185617

theorem part1 (A B C a b c : ℝ) (h1 : 3 * a * Real.cos A = Real.sqrt 6 * (c * Real.cos B + b * Real.cos C)) :
    Real.tan (2 * A) = 2 * Real.sqrt 2 := sorry

theorem part2 (A B C a b c S : ℝ) 
  (h_sin_B : Real.sin (Real.pi / 2 + B) = 2 * Real.sqrt 2 / 3)
  (hc : c = 2 * Real.sqrt 2) :
    S = 2 * Real.sqrt 2 / 3 := sorry

end NUMINAMATH_GPT_part1_part2_l1856_185617


namespace NUMINAMATH_GPT_question1_question2_l1856_185639

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Problem 1: Prove the valid solution of x when f(x) = 3 and x ∈ [0, 4]
theorem question1 (h₀ : 0 ≤ 3) (h₁ : 4 ≥ 3) : 
  ∃ (x : ℝ), (f x = 3 ∧ 0 ≤ x ∧ x ≤ 4) → x = 3 :=
by
  sorry

-- Problem 2: Prove the range of f(x) when x ∈ [0, 4]
theorem question2 : 
  ∃ (a b : ℝ), (∀ x, 0 ≤ x ∧ x ≤ 4 → a ≤ f x ∧ f x ≤ b) → a = -1 ∧ b = 8 :=
by
  sorry

end NUMINAMATH_GPT_question1_question2_l1856_185639


namespace NUMINAMATH_GPT_janet_hourly_wage_l1856_185656

theorem janet_hourly_wage : 
  ∃ x : ℝ, 
    (20 * x + (5 * 20 + 7 * 20) = 1640) ∧ 
    x = 70 :=
by
  use 70
  sorry

end NUMINAMATH_GPT_janet_hourly_wage_l1856_185656


namespace NUMINAMATH_GPT_problem_f_neg2_equals_2_l1856_185637

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem_f_neg2_equals_2 (f : ℝ → ℝ) (b : ℝ) 
  (h_odd : is_odd_function f)
  (h_def : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 3 * x + b) 
  (h_b : b = 0) :
  f (-2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_f_neg2_equals_2_l1856_185637


namespace NUMINAMATH_GPT_find_x_l1856_185683

theorem find_x (x : ℝ) (h : 2500 - 1002 / x = 2450) : x = 20.04 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l1856_185683


namespace NUMINAMATH_GPT_lcm_18_35_l1856_185636

-- Given conditions: Prime factorizations of 18 and 35
def factorization_18 : Prop := (18 = 2^1 * 3^2)
def factorization_35 : Prop := (35 = 5^1 * 7^1)

-- The goal is to prove that the least common multiple of 18 and 35 is 630
theorem lcm_18_35 : factorization_18 ∧ factorization_35 → Nat.lcm 18 35 = 630 := by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_lcm_18_35_l1856_185636
