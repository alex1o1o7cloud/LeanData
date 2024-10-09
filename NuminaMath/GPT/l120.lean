import Mathlib

namespace number_multiplied_by_approx_l120_12004

variable (X : ℝ)

theorem number_multiplied_by_approx (h : (0.0048 * X) / (0.05 * 0.1 * 0.004) = 840) : X = 3.5 :=
by
  sorry

end number_multiplied_by_approx_l120_12004


namespace distinct_convex_quadrilaterals_l120_12030

open Nat

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem distinct_convex_quadrilaterals (n : ℕ) (h : n > 4) 
  (no_three_collinear : ℕ → Prop) :
  ∃ k, k ≥ combinations n 5 / (n - 4) :=
by
  sorry

end distinct_convex_quadrilaterals_l120_12030


namespace domain_of_transformed_function_l120_12060

theorem domain_of_transformed_function (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → True) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → True :=
sorry

end domain_of_transformed_function_l120_12060


namespace infinite_geometric_sum_l120_12036

noncomputable def geometric_sequence (n : ℕ) : ℝ := 3 * (-1 / 2)^(n - 1)

theorem infinite_geometric_sum :
  ∑' n, geometric_sequence n = 2 :=
sorry

end infinite_geometric_sum_l120_12036


namespace max_value_quadratic_max_value_quadratic_attained_l120_12015

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_value_quadratic : ∀ (x : ℝ), quadratic (-8) 32 (-1) x ≤ 31 :=
by
  sorry

theorem max_value_quadratic_attained : 
  quadratic (-8) 32 (-1) 2 = 31 :=
by
  sorry

end max_value_quadratic_max_value_quadratic_attained_l120_12015


namespace polygon_interior_exterior_relation_l120_12034

theorem polygon_interior_exterior_relation (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n ≥ 3) :
  n = 8 :=
by
  sorry

end polygon_interior_exterior_relation_l120_12034


namespace coloring_points_l120_12051

theorem coloring_points
  (A : ℤ × ℤ) (B : ℤ × ℤ) (C : ℤ × ℤ)
  (hA : A.fst % 2 = 1 ∧ A.snd % 2 = 1)
  (hB : (B.fst % 2 = 1 ∧ B.snd % 2 = 0) ∨ (B.fst % 2 = 0 ∧ B.snd % 2 = 1))
  (hC : C.fst % 2 = 0 ∧ C.snd % 2 = 0) :
  ∃ D : ℤ × ℤ,
    (D.fst % 2 = 1 ∧ D.snd % 2 = 0) ∨ (D.fst % 2 = 0 ∧ D.snd % 2 = 1) ∧
    (A.fst + C.fst = B.fst + D.fst) ∧
    (A.snd + C.snd = B.snd + D.snd) := 
sorry

end coloring_points_l120_12051


namespace train_pass_time_l120_12086

-- Definitions based on the conditions
def train_length : ℕ := 360   -- Length of the train in meters
def platform_length : ℕ := 190 -- Length of the platform in meters
def speed_kmh : ℕ := 45       -- Speed of the train in km/h
def speed_ms : ℚ := speed_kmh * (1000 / 3600) -- Speed of the train in m/s

-- Total distance to be covered
def total_distance : ℕ := train_length + platform_length 

-- Time taken to pass the platform
def time_to_pass_platform : ℚ := total_distance / speed_ms

-- Proof that the time taken is 44 seconds
theorem train_pass_time : time_to_pass_platform = 44 := 
by 
  -- this is where the detailed proof would go
  sorry  

end train_pass_time_l120_12086


namespace age_difference_l120_12013

/-- 
The overall age of x and y is some years greater than the overall age of y and z. Z is 12 years younger than X.
Prove: The overall age of x and y is 12 years greater than the overall age of y and z.
-/
theorem age_difference {X Y Z : ℕ} (h1: X + Y > Y + Z) (h2: Z = X - 12) : 
  (X + Y) - (Y + Z) = 12 :=
by 
  -- proof goes here
  sorry

end age_difference_l120_12013


namespace katie_earnings_l120_12067

def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

theorem katie_earnings : bead_necklaces + gemstone_necklaces * cost_per_necklace = 21 := 
by
  sorry

end katie_earnings_l120_12067


namespace units_digit_sum_of_factorials_l120_12091

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_of_factorials :
  ones_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
              factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) = 3 := 
sorry

end units_digit_sum_of_factorials_l120_12091


namespace actual_time_when_car_clock_shows_10PM_l120_12025

def car_clock_aligned (aligned_time wristwatch_time : ℕ) : Prop :=
  aligned_time = wristwatch_time

def car_clock_time (rate: ℚ) (hours_elapsed_real_time hours_elapsed_car_time : ℚ) : Prop :=
  rate = hours_elapsed_car_time / hours_elapsed_real_time

def actual_time (current_car_time car_rate : ℚ) : ℚ :=
  current_car_time / car_rate

theorem actual_time_when_car_clock_shows_10PM :
  let accurate_start_time := 9 -- 9:00 AM
  let car_start_time := 9 -- Synchronized at 9:00 AM
  let wristwatch_time_wristwatch := 13 -- 1:00 PM in hours
  let car_time_car := 13 + 48 / 60 -- 1:48 PM in hours
  let rate := car_time_car / wristwatch_time_wristwatch
  let current_car_time := 22 -- 10:00 PM in hours
  let real_time := actual_time current_car_time rate
  real_time = 19.8333 := -- which converts to 7:50 PM (Option B)
sorry

end actual_time_when_car_clock_shows_10PM_l120_12025


namespace traveler_meets_truck_at_15_48_l120_12057

noncomputable def timeTravelerMeetsTruck : ℝ := 15 + 48 / 60

theorem traveler_meets_truck_at_15_48 {S Vp Vm Vg : ℝ}
  (h_travel_covered : Vp = S / 4)
  (h_motorcyclist_catch : 1 = (S / 4) / (Vm - Vp))
  (h_motorcyclist_meet_truck : 1.5 = S / (Vm + Vg)) :
  (S / 4 + (12 / 5) * (Vg + Vp)) / (12 / 5) = timeTravelerMeetsTruck := sorry

end traveler_meets_truck_at_15_48_l120_12057


namespace solution_l120_12002

def is_prime (n : ℕ) : Prop := ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

noncomputable def find_pairs : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ is_prime (a * b^2 / (a + b)) ∧ ((a = 6 ∧ b = 2) ∨ (a = 2 ∧ b = 6))

theorem solution :
  find_pairs := sorry

end solution_l120_12002


namespace correct_operation_C_l120_12029

theorem correct_operation_C (m : ℕ) : m^7 / m^3 = m^4 := by
  sorry

end correct_operation_C_l120_12029


namespace pig_duck_ratio_l120_12050

theorem pig_duck_ratio (G C D P : ℕ)
(h₁ : G = 66)
(h₂ : C = 2 * G)
(h₃ : D = (G + C) / 2)
(h₄ : P = G - 33) :
  P / D = 1 / 3 :=
by {
  sorry
}

end pig_duck_ratio_l120_12050


namespace molecular_weight_correct_l120_12095

-- Define the atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01

-- Define the number of atoms of each element
def num_atoms_K : ℕ := 2
def num_atoms_Br : ℕ := 2
def num_atoms_O : ℕ := 4
def num_atoms_H : ℕ := 3
def num_atoms_N : ℕ := 1

-- Calculate the molecular weight
def molecular_weight : ℝ :=
  num_atoms_K * atomic_weight_K +
  num_atoms_Br * atomic_weight_Br +
  num_atoms_O * atomic_weight_O +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 319.04

-- The theorem stating that the calculated molecular weight matches the expected molecular weight
theorem molecular_weight_correct : molecular_weight = expected_molecular_weight :=
  by
  sorry -- Proof is skipped

end molecular_weight_correct_l120_12095


namespace cost_of_four_dozen_apples_l120_12074

-- Define the given conditions and problem
def half_dozen_cost : ℚ := 4.80 -- cost of half a dozen apples
def full_dozen_cost : ℚ := half_dozen_cost / 0.5
def four_dozen_cost : ℚ := 4 * full_dozen_cost

-- Statement of the theorem to prove
theorem cost_of_four_dozen_apples : four_dozen_cost = 38.40 :=
by
  sorry

end cost_of_four_dozen_apples_l120_12074


namespace greatest_value_of_n_l120_12053

theorem greatest_value_of_n (n : ℤ) (h : 101 * n ^ 2 ≤ 3600) : n ≤ 5 :=
by
  sorry

end greatest_value_of_n_l120_12053


namespace circle_properties_l120_12052

noncomputable def circle_center_and_radius (x y : ℝ) : ℝ × ℝ × ℝ :=
  let eq1 := x^2 - 4 * y - 18
  let eq2 := -y^2 + 6 * x + 26
  let lhs := x^2 - 6 * x + y^2 - 4 * y
  let rhs := 44
  let center_x := 3
  let center_y := 2
  let radius := Real.sqrt 57
  let target := 5 + radius
  (center_x, center_y, target)

theorem circle_properties
  (x y : ℝ) :
  let (a, b, r) := circle_center_and_radius x y 
  a + b + r = 5 + Real.sqrt 57 :=
by
  sorry

end circle_properties_l120_12052


namespace ball_hits_ground_at_correct_time_l120_12027

def initial_velocity : ℝ := 7
def initial_height : ℝ := 10

-- The height function as given by the condition
def height_function (t : ℝ) : ℝ := -4.9 * t^2 + initial_velocity * t + initial_height

-- Statement
theorem ball_hits_ground_at_correct_time :
  ∃ t : ℝ, height_function t = 0 ∧ t = 2313 / 1000 :=
by
  sorry

end ball_hits_ground_at_correct_time_l120_12027


namespace smallest_x_for_square_l120_12066

theorem smallest_x_for_square (N : ℕ) (h1 : ∃ x : ℕ, x > 0 ∧ 1260 * x = N^2) : ∃ x : ℕ, x = 35 :=
by
  sorry

end smallest_x_for_square_l120_12066


namespace number_of_b_values_l120_12028

theorem number_of_b_values (b : ℤ) :
  (∃ (x1 x2 x3 : ℤ), ∀ (x : ℤ), x^2 + b * x + 6 ≤ 0 ↔ x = x1 ∨ x = x2 ∨ x = x3) ↔ (b = -6 ∨ b = -5 ∨ b = 5 ∨ b = 6) :=
by
  sorry

end number_of_b_values_l120_12028


namespace insurance_payment_yearly_l120_12070

noncomputable def quarterly_payment : ℝ := 378
noncomputable def quarters_per_year : ℕ := 12 / 3
noncomputable def annual_payment : ℝ := quarterly_payment * quarters_per_year

theorem insurance_payment_yearly : annual_payment = 1512 := by
  sorry

end insurance_payment_yearly_l120_12070


namespace number_of_sturgeons_l120_12019

def number_of_fishes := 145
def number_of_pikes := 30
def number_of_herrings := 75

theorem number_of_sturgeons : (number_of_fishes - (number_of_pikes + number_of_herrings) = 40) :=
  by
  sorry

end number_of_sturgeons_l120_12019


namespace total_number_of_animals_l120_12010

-- Definitions based on conditions
def number_of_females : ℕ := 35
def males_outnumber_females_by : ℕ := 7
def number_of_males : ℕ := number_of_females + males_outnumber_females_by

-- Theorem to prove the total number of animals
theorem total_number_of_animals :
  number_of_females + number_of_males = 77 := by
  sorry

end total_number_of_animals_l120_12010


namespace number_of_dogs_with_both_tags_and_collars_l120_12054

-- Defining the problem
def total_dogs : ℕ := 80
def dogs_with_tags : ℕ := 45
def dogs_with_collars : ℕ := 40
def dogs_with_neither : ℕ := 1

-- Statement: Prove the number of dogs with both tags and collars
theorem number_of_dogs_with_both_tags_and_collars : 
  (dogs_with_tags + dogs_with_collars - total_dogs + dogs_with_neither) = 6 :=
by
  sorry

end number_of_dogs_with_both_tags_and_collars_l120_12054


namespace toms_dog_age_is_twelve_l120_12031

-- Definitions based on given conditions
def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := 3 * toms_rabbit_age

-- The statement to be proved
theorem toms_dog_age_is_twelve : toms_dog_age = 12 := by
  sorry

end toms_dog_age_is_twelve_l120_12031


namespace problem_solution_l120_12090

variable (x y z : ℝ)

theorem problem_solution
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + x * y = 8)
  (h2 : y + z + y * z = 15)
  (h3 : z + x + z * x = 35) :
  x + y + z + x * y = 15 :=
sorry

end problem_solution_l120_12090


namespace bags_on_monday_l120_12022

/-- Define the problem conditions -/
def t : Nat := 8  -- total number of bags
def f : Nat := 4  -- number of bags found the next day

-- Define the statement to be proven
theorem bags_on_monday : t - f = 4 := by
  -- Sorry to skip the proof
  sorry

end bags_on_monday_l120_12022


namespace prism_width_l120_12098

/-- A rectangular prism with dimensions l, w, h such that the diagonal length is 13
    and given l = 3 and h = 12, has width w = 4. -/
theorem prism_width (w : ℕ) 
  (h : ℕ) (l : ℕ) 
  (diag_len : ℕ) 
  (hl : l = 3) 
  (hh : h = 12) 
  (hd : diag_len = 13) 
  (h_diag : diag_len = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := 
  sorry

end prism_width_l120_12098


namespace sophie_one_dollar_bills_l120_12084

theorem sophie_one_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 55) 
  (h2 : x + 2 * y + 5 * z = 126) 
  : x = 18 := by
  sorry

end sophie_one_dollar_bills_l120_12084


namespace distance_between_trees_l120_12024

theorem distance_between_trees (l : ℕ) (n : ℕ) (d : ℕ) (h_length : l = 225) (h_trees : n = 26) (h_segments : n - 1 = 25) : d = 9 :=
sorry

end distance_between_trees_l120_12024


namespace infinitely_many_gt_sqrt_l120_12039

open Real

noncomputable def sequences := ℕ → ℕ × ℕ

def strictly_increasing_ratios (seq : sequences) : Prop :=
  ∀ n : ℕ, 0 < n → (seq (n + 1)).2 / (seq (n + 1)).1 > (seq n).2 / (seq n).1

theorem infinitely_many_gt_sqrt (seq : sequences) 
  (positive_integers : ∀ n : ℕ, (seq n).1 > 0 ∧ (seq n).2 > 0) 
  (inc_ratios : strictly_increasing_ratios seq) :
  ∃ᶠ n in at_top, (seq n).2 > sqrt n :=
sorry

end infinitely_many_gt_sqrt_l120_12039


namespace corrected_mean_l120_12068

theorem corrected_mean (n : ℕ) (incorrect_mean : ℝ) (incorrect_observation correct_observation : ℝ)
  (h_n : n = 50)
  (h_incorrect_mean : incorrect_mean = 30)
  (h_incorrect_observation : incorrect_observation = 23)
  (h_correct_observation : correct_observation = 48) :
  (incorrect_mean * n - incorrect_observation + correct_observation) / n = 30.5 :=
by
  sorry

end corrected_mean_l120_12068


namespace salary_increase_correct_l120_12047

noncomputable def old_average_salary : ℕ := 1500
noncomputable def number_of_employees : ℕ := 24
noncomputable def manager_salary : ℕ := 11500
noncomputable def new_total_salary := (number_of_employees * old_average_salary) + manager_salary
noncomputable def new_number_of_people := number_of_employees + 1
noncomputable def new_average_salary := new_total_salary / new_number_of_people
noncomputable def salary_increase := new_average_salary - old_average_salary

theorem salary_increase_correct : salary_increase = 400 := by
sorry

end salary_increase_correct_l120_12047


namespace find_f_prime_zero_l120_12083

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Condition given in the problem.
def f_def : ∀ x : ℝ, f x = x^2 + 2 * x * f' 1 := 
sorry

-- Statement we want to prove.
theorem find_f_prime_zero : f' 0 = -4 := 
sorry

end find_f_prime_zero_l120_12083


namespace pen_cost_price_l120_12085

-- Define the variables and assumptions
variable (x : ℝ)

-- Given conditions
def profit_one_pen (x : ℝ) := 10 - x
def profit_three_pens (x : ℝ) := 20 - 3 * x

-- Statement to prove
theorem pen_cost_price : profit_one_pen x = profit_three_pens x → x = 5 :=
by
  sorry

end pen_cost_price_l120_12085


namespace total_chapters_read_l120_12093

def books_read : ℕ := 12
def chapters_per_book : ℕ := 32

theorem total_chapters_read : books_read * chapters_per_book = 384 :=
by
  sorry

end total_chapters_read_l120_12093


namespace max_integer_value_of_k_l120_12021

theorem max_integer_value_of_k :
  ∀ x y k : ℤ,
    x - 4 * y = k - 1 →
    2 * x + y = k →
    x - y ≤ 0 →
    k ≤ 0 :=
by
  intros x y k h1 h2 h3
  sorry

end max_integer_value_of_k_l120_12021


namespace stratified_sampling_third_grade_l120_12063

theorem stratified_sampling_third_grade (total_students : ℕ) (first_grade_students : ℕ)
  (second_grade_students : ℕ) (third_grade_students : ℕ) (sample_size : ℕ)
  (h_total : total_students = 270000) (h_first : first_grade_students = 99000)
  (h_second : second_grade_students = 90000) (h_third : third_grade_students = 81000)
  (h_sample : sample_size = 3000) :
  third_grade_students * (sample_size / total_students) = 900 := 
by {
  sorry
}

end stratified_sampling_third_grade_l120_12063


namespace bank_exceeds_1600cents_in_9_days_after_Sunday_l120_12043

theorem bank_exceeds_1600cents_in_9_days_after_Sunday
  (a : ℕ)
  (r : ℕ)
  (initial_deposit : ℕ)
  (days_after_sunday : ℕ)
  (geometric_series : ℕ -> ℕ)
  (sum_geometric_series : ℕ -> ℕ)
  (geo_series_definition : ∀(n : ℕ), geometric_series n = 5 * 2^n)
  (sum_geo_series_definition : ∀(n : ℕ), sum_geometric_series n = 5 * (2^n - 1))
  (exceeds_condition : ∀(n : ℕ), sum_geometric_series n > 1600 -> n >= 9) :
  days_after_sunday = 9 → a = 5 → r = 2 → initial_deposit = 5 → days_after_sunday = 9 → geometric_series 1 = 10 → sum_geometric_series 9 > 1600 :=
by sorry

end bank_exceeds_1600cents_in_9_days_after_Sunday_l120_12043


namespace value_of_a_sub_b_l120_12079

theorem value_of_a_sub_b (a b : ℝ) (h1 : abs a = 8) (h2 : abs b = 5) (h3 : a > 0) (h4 : b < 0) : a - b = 13 := 
  sorry

end value_of_a_sub_b_l120_12079


namespace intersection_correct_l120_12096

-- Define sets M and N
def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log (2 * x + 1) > 0}

-- Define the intersection of M and N
def M_intersect_N := {x : ℝ | 0 < x ∧ x < 1}

-- Prove that M_intersect_N is the correct intersection
theorem intersection_correct : M ∩ N = M_intersect_N :=
by
  sorry

end intersection_correct_l120_12096


namespace diamond_value_l120_12062

def diamond (a b : ℕ) : ℚ := 1 / (a : ℚ) + 2 / (b : ℚ)

theorem diamond_value : ∀ (a b : ℕ), a + b = 10 ∧ a * b = 24 → diamond a b = 2 / 3 := by
  intros a b h
  sorry

end diamond_value_l120_12062


namespace integer_multiple_of_ten_l120_12041

theorem integer_multiple_of_ten (x : ℤ) :
  10 * x = 30 ↔ x = 3 :=
by
  sorry

end integer_multiple_of_ten_l120_12041


namespace sin_cos_difference_theorem_tan_theorem_l120_12037

open Real

noncomputable def sin_cos_difference (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5)

theorem sin_cos_difference_theorem (x : ℝ) (h : sin_cos_difference x) : 
  sin x - cos x = - 7 / 5 := by
  sorry

noncomputable def sin_cos_ratio (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5) ∧ (tan x = -3 / 4)

theorem tan_theorem (x : ℝ) (h : sin_cos_ratio x) :
  tan x = -3 / 4 := by
  sorry

end sin_cos_difference_theorem_tan_theorem_l120_12037


namespace y_intercept_of_line_l120_12008

theorem y_intercept_of_line (m : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 7) (h₃ : y₀ = 0) :
  ∃ (b : ℝ), (0, b) = (0, 21) :=
by
  -- Our goal is to prove the y-intercept is (0, 21)
  sorry

end y_intercept_of_line_l120_12008


namespace sufficient_but_not_necessary_condition_l120_12080

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (-2 ≤ x ∧ x ≤ 2) → (x ≤ a))
  → (∃ x : ℝ, (x ≤ a ∧ ¬((-2 ≤ x ∧ x ≤ 2))))
  → (a ≥ 2) :=
by
  intros h1 h2
  sorry

end sufficient_but_not_necessary_condition_l120_12080


namespace ratio_of_larger_to_smaller_l120_12017

theorem ratio_of_larger_to_smaller
  (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h3 : x + y = 6 * (x - y)) :
  x / y = 7 / 5 :=
by sorry

end ratio_of_larger_to_smaller_l120_12017


namespace range_of_reciprocal_sum_l120_12011

theorem range_of_reciprocal_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y + 1 / x + 1 / y = 10) :
  1 ≤ 1 / x + 1 / y ∧ 1 / x + 1 / y ≤ 9 := 
sorry

end range_of_reciprocal_sum_l120_12011


namespace max_value_of_f_l120_12044

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_of_f : ∃ max, max ∈ Set.image f (Set.Icc (-1 : ℝ) 1) ∧ max = Real.exp 1 - 1 :=
by
  sorry

end max_value_of_f_l120_12044


namespace find_a_value_l120_12087

theorem find_a_value 
  (a : ℝ)
  (h : abs (1 - (-1 / (4 * a))) = 2) :
  a = 1 / 4 ∨ a = -1 / 12 :=
sorry

end find_a_value_l120_12087


namespace selection_methods_l120_12012

-- Conditions
def volunteers : ℕ := 5
def friday_slots : ℕ := 1
def saturday_slots : ℕ := 2
def sunday_slots : ℕ := 1

-- Function to calculate combinatorial n choose k
def choose (n k : ℕ) : ℕ :=
(n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Function to calculate permutations of n P k
def perm (n k : ℕ) : ℕ :=
(n.factorial) / ((n - k).factorial)

-- The target proposition
theorem selection_methods : choose volunteers saturday_slots * perm (volunteers - saturday_slots) (friday_slots + sunday_slots) = 60 :=
by
  -- assumption here leads to the property required, usually this would be more detailed computation.
  sorry

end selection_methods_l120_12012


namespace largest_integer_m_l120_12082

theorem largest_integer_m (m n : ℕ) (h1 : ∀ n ≤ m, (2 * n + 1) / (3 * n + 8) < (Real.sqrt 5 - 1) / 2) 
(h2 : ∀ n ≤ m, (Real.sqrt 5 - 1) / 2 < (n + 7) / (2 * n + 1)) : 
  m = 27 :=
sorry

end largest_integer_m_l120_12082


namespace right_triangle_48_55_l120_12088

def right_triangle_properties (a b : ℕ) (ha : a = 48) (hb : b = 55) : Prop :=
  let area := 1 / 2 * a * b
  let hypotenuse := Real.sqrt (a ^ 2 + b ^ 2)
  area = 1320 ∧ hypotenuse = 73

theorem right_triangle_48_55 : right_triangle_properties 48 55 (by rfl) (by rfl) :=
  sorry

end right_triangle_48_55_l120_12088


namespace not_prime_sum_l120_12065

theorem not_prime_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_square : ∃ k : ℕ, a^2 - b * c = k^2) : ¬ Nat.Prime (2 * a + b + c) := 
sorry

end not_prime_sum_l120_12065


namespace inverse_proportion_first_third_quadrant_l120_12073

theorem inverse_proportion_first_third_quadrant (k : ℝ) :
  (∀ (x : ℝ), x ≠ 0 → ((x > 0 → (2 - k) / x > 0) ∧ (x < 0 → (2 - k) / x < 0))) → k < 2 :=
by
  sorry

end inverse_proportion_first_third_quadrant_l120_12073


namespace min_distance_sum_well_l120_12069

theorem min_distance_sum_well (A B C : ℝ) (h1 : B = A + 50) (h2 : C = B + 50) :
  ∃ X : ℝ, X = B ∧ (∀ Y : ℝ, (dist Y A + dist Y B + dist Y C) ≥ (dist B A + dist B B + dist B C)) :=
sorry

end min_distance_sum_well_l120_12069


namespace highest_number_paper_l120_12042

theorem highest_number_paper (n : ℕ) (h : (1 : ℝ) / n = 0.010526315789473684) : n = 95 :=
sorry

end highest_number_paper_l120_12042


namespace acrobat_count_range_l120_12007

def animal_legs (elephants monkeys acrobats : ℕ) : ℕ :=
  4 * elephants + 2 * monkeys + 2 * acrobats

def animal_heads (elephants monkeys acrobats : ℕ) : ℕ :=
  elephants + monkeys + acrobats

theorem acrobat_count_range (e m a : ℕ) (h1 : animal_heads e m a = 18)
  (h2 : animal_legs e m a = 50) : 0 ≤ a ∧ a ≤ 11 :=
by {
  sorry
}

end acrobat_count_range_l120_12007


namespace nate_total_time_l120_12099

/-- Definitions for the conditions -/
def sectionG : ℕ := 18 * 12
def sectionH : ℕ := 25 * 10
def sectionI : ℕ := 17 * 11
def sectionJ : ℕ := 20 * 9
def sectionK : ℕ := 15 * 13

def speedGH : ℕ := 8
def speedIJ : ℕ := 10
def speedK : ℕ := 6

/-- Compute the time spent in each section, rounding up where necessary -/
def timeG : ℕ := (sectionG + speedGH - 1) / speedGH
def timeH : ℕ := (sectionH + speedGH - 1) / speedGH
def timeI : ℕ := (sectionI + speedIJ - 1) / speedIJ
def timeJ : ℕ := (sectionJ + speedIJ - 1) / speedIJ
def timeK : ℕ := (sectionK + speedK - 1) / speedK

/-- Compute the total time spent -/
def totalTime : ℕ := timeG + timeH + timeI + timeJ + timeK

/-- The proof statement -/
theorem nate_total_time : totalTime = 129 := by
  -- the proof goes here
  sorry

end nate_total_time_l120_12099


namespace total_points_first_four_games_l120_12035

-- Define the scores for the first three games
def score1 : ℕ := 10
def score2 : ℕ := 14
def score3 : ℕ := 6

-- Define the score for the fourth game as the average of the first three games
def score4 : ℕ := (score1 + score2 + score3) / 3

-- Define the total points scored in the first four games
def total_points : ℕ := score1 + score2 + score3 + score4

-- State the theorem to prove
theorem total_points_first_four_games : total_points = 40 :=
  sorry

end total_points_first_four_games_l120_12035


namespace Jon_regular_bottle_size_is_16oz_l120_12056

noncomputable def Jon_bottle_size (x : ℝ) : Prop :=
  let daily_intake := 4 * x + 2 * 1.25 * x
  let weekly_intake := 7 * daily_intake
  weekly_intake = 728

theorem Jon_regular_bottle_size_is_16oz : ∃ x : ℝ, Jon_bottle_size x ∧ x = 16 :=
by
  use 16
  sorry

end Jon_regular_bottle_size_is_16oz_l120_12056


namespace Brenda_bakes_cakes_l120_12071

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end Brenda_bakes_cakes_l120_12071


namespace smallest_percentage_owning_90_percent_money_l120_12038

theorem smallest_percentage_owning_90_percent_money
  (P M : ℝ)
  (h1 : 0.2 * P = 0.8 * M) :
  (∃ x : ℝ, x = 0.6 * P ∧ 0.9 * M <= (0.2 * P + (x - 0.2 * P))) :=
sorry

end smallest_percentage_owning_90_percent_money_l120_12038


namespace equation_solution_system_solution_l120_12049

theorem equation_solution (x : ℚ) :
  (3 * x + 1) / 5 = 1 - (4 * x + 3) / 2 ↔ x = -7 / 26 :=
by sorry

theorem system_solution (x y : ℚ) :
  (3 * x - 4 * y = 14) ∧ (5 * x + 4 * y = 2) ↔
  (x = 2) ∧ (y = -2) :=
by sorry

end equation_solution_system_solution_l120_12049


namespace margo_walks_total_distance_l120_12018

theorem margo_walks_total_distance :
  let time_to_house := 15
  let time_to_return := 25
  let total_time_minutes := time_to_house + time_to_return
  let total_time_hours := (total_time_minutes : ℝ) / 60
  let avg_rate := 3  -- units: miles per hour
  (avg_rate * total_time_hours = 2) := 
sorry

end margo_walks_total_distance_l120_12018


namespace ceil_mul_eq_225_l120_12045

theorem ceil_mul_eq_225 {x : ℝ} (h₁ : ⌈x⌉ * x = 225) (h₂ : x > 0) : x = 15 :=
sorry

end ceil_mul_eq_225_l120_12045


namespace smallest_h_l120_12058

theorem smallest_h (h : ℕ) : 
  (∀ k, h = k → (k + 5) % 8 = 0 ∧ 
        (k + 5) % 11 = 0 ∧ 
        (k + 5) % 24 = 0) ↔ h = 259 :=
by
  sorry

end smallest_h_l120_12058


namespace range_function_1_l120_12020

theorem range_function_1 (y : ℝ) : 
  (∃ x : ℝ, x ≥ -1 ∧ y = (1/3) ^ x) ↔ (0 < y ∧ y ≤ 3) :=
sorry

end range_function_1_l120_12020


namespace problem1_problem2_l120_12081

variable (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)

theorem problem1 : 
  (a * b + a + b + 1) * (a * b + a * c + b * c + c ^ 2) ≥ 16 * a * b * c := 
by sorry

theorem problem2 : 
  (b + c - a) / a + (c + a - b) / b + (a + b - c) / c ≥ 3 := 
by sorry

end problem1_problem2_l120_12081


namespace regular_pentagon_diagonal_square_l120_12059

variable (a d : ℝ)
def is_regular_pentagon (a d : ℝ) : Prop :=
d ^ 2 = a ^ 2 + a * d

theorem regular_pentagon_diagonal_square :
  is_regular_pentagon a d :=
sorry

end regular_pentagon_diagonal_square_l120_12059


namespace trapezoid_base_difference_is_10_l120_12048

noncomputable def trapezoid_base_difference (AD BC AB : ℝ) (angle_BAD angle_ADC : ℝ) : ℝ :=
if angle_BAD = 60 ∧ angle_ADC = 30 ∧ AB = 5 then AD - BC else 0

theorem trapezoid_base_difference_is_10 (AD BC : ℝ) (angle_BAD angle_ADC : ℝ) (h_BAD : angle_BAD = 60)
(h_ADC : angle_ADC = 30) (h_AB : AB = 5) : trapezoid_base_difference AD BC AB angle_BAD angle_ADC = 10 :=
sorry

end trapezoid_base_difference_is_10_l120_12048


namespace min_value_xy_l120_12046

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end min_value_xy_l120_12046


namespace brick_height_l120_12009

theorem brick_height (length width : ℕ) (num_bricks : ℕ) (wall_length wall_width wall_height : ℕ) (h : ℕ) :
  length = 20 ∧ width = 10 ∧ num_bricks = 25000 ∧ wall_length = 2500 ∧ wall_width = 200 ∧ wall_height = 75 ∧
  ( 20 * 10 * h = (wall_length * wall_width * wall_height) / 25000 ) -> 
  h = 75 :=
by
  sorry

end brick_height_l120_12009


namespace range_of_p_l120_12026

noncomputable def proof_problem (p : ℝ) : Prop :=
  (∀ x : ℝ, (4 * x + p < 0) → (x < -1 ∨ x > 2)) → (p ≥ 4)

theorem range_of_p (p : ℝ) : proof_problem p :=
by
  intros h
  sorry

end range_of_p_l120_12026


namespace gravitational_equal_forces_point_l120_12061

variable (d M m : ℝ) (hM : 0 < M) (hm : 0 < m) (hd : 0 < d)

theorem gravitational_equal_forces_point :
  ∃ x : ℝ, (0 < x ∧ x < d) ∧ x = d / (1 + Real.sqrt (m / M)) :=
by
  sorry

end gravitational_equal_forces_point_l120_12061


namespace b_share_in_profit_l120_12094

theorem b_share_in_profit (A B C : ℝ) (p : ℝ := 4400) (x : ℝ)
  (h1 : A = 3 * B)
  (h2 : B = (2 / 3) * C)
  (h3 : C = x) :
  B / (A + B + C) * p = 800 :=
by
  sorry

end b_share_in_profit_l120_12094


namespace num_math_not_science_l120_12064

-- Definitions as conditions
def students_total : ℕ := 30
def both_clubs : ℕ := 2
def math_to_science_ratio : ℕ := 3

-- The proof we need to show
theorem num_math_not_science :
  ∃ x y : ℕ, (x + y + both_clubs = students_total) ∧ (y = math_to_science_ratio * (x + both_clubs) - 2 * (math_to_science_ratio - 1)) ∧ (y - both_clubs = 20) :=
by
  sorry

end num_math_not_science_l120_12064


namespace x4_y4_value_l120_12032

theorem x4_y4_value (x y : ℝ) (h1 : x^4 + x^2 = 3) (h2 : y^4 - y^2 = 3) : x^4 + y^4 = 7 := by
  sorry

end x4_y4_value_l120_12032


namespace find_g_of_one_fifth_l120_12097

variable {g : ℝ → ℝ}

theorem find_g_of_one_fifth (h₀ : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1)
    (h₁ : g 0 = 0)
    (h₂ : ∀ {x y}, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y)
    (h₃ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x)
    (h₄ : ∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2) :
  g (1 / 5) = 1 / 4 :=
by
  sorry

end find_g_of_one_fifth_l120_12097


namespace remaining_shape_perimeter_l120_12040

def rectangle_perimeter (L W : ℕ) : ℕ := 2 * (L + W)

theorem remaining_shape_perimeter (L W S : ℕ) (hL : L = 12) (hW : W = 5) (hS : S = 2) :
  rectangle_perimeter L W = 34 :=
by
  rw [hL, hW]
  rfl

end remaining_shape_perimeter_l120_12040


namespace shorter_piece_length_l120_12075

theorem shorter_piece_length (x : ℕ) (h1 : 177 = x + 2*x) : x = 59 :=
by sorry

end shorter_piece_length_l120_12075


namespace inequality_selection_l120_12014

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := 
by sorry

end inequality_selection_l120_12014


namespace sum_arithmetic_series_l120_12072

theorem sum_arithmetic_series :
  let a := -42
  let d := 2
  let l := 0
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = -462 := by
sorry

end sum_arithmetic_series_l120_12072


namespace problem_probability_ao_drawn_second_l120_12003

def is_ao_drawn_second (pair : ℕ × ℕ) : Bool :=
  pair.snd = 3

def random_pairs : List (ℕ × ℕ) := [
  (1, 3), (2, 4), (1, 2), (3, 2), (4, 3), (1, 4), (2, 4), (3, 2), (3, 1), (2, 1), 
  (2, 3), (1, 3), (3, 2), (2, 1), (2, 4), (4, 2), (1, 3), (3, 2), (2, 1), (3, 4)
]

def count_ao_drawn_second : ℕ :=
  (random_pairs.filter is_ao_drawn_second).length

def probability_ao_drawn_second : ℚ :=
  count_ao_drawn_second / random_pairs.length

theorem problem_probability_ao_drawn_second :
  probability_ao_drawn_second = 1 / 4 :=
by
  sorry

end problem_probability_ao_drawn_second_l120_12003


namespace jordan_book_pages_l120_12005

theorem jordan_book_pages (avg_first_4_days : ℕ)
                           (avg_next_2_days : ℕ)
                           (pages_last_day : ℕ)
                           (total_pages : ℕ) :
  avg_first_4_days = 42 → 
  avg_next_2_days = 38 → 
  pages_last_day = 20 → 
  total_pages = 4 * avg_first_4_days + 2 * avg_next_2_days + pages_last_day →
  total_pages = 264 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end jordan_book_pages_l120_12005


namespace tan_difference_l120_12016

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4 / 3) :
  Real.tan (α - β) = 1 / 3 :=
by
  sorry

end tan_difference_l120_12016


namespace smallest_value_l120_12000

theorem smallest_value 
  (x1 x2 x3 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2) 
  (hx3 : 0 < x3)
  (h : 2 * x1 + 3 * x2 + 4 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 = 10000 / 29 := by
  sorry

end smallest_value_l120_12000


namespace cone_section_volume_ratio_l120_12078

theorem cone_section_volume_ratio :
  ∀ (r h : ℝ), (h > 0 ∧ r > 0) →
  let V1 := ((75 / 3) * π * r^2 * h - (64 / 3) * π * r^2 * h)
  let V2 := ((64 / 3) * π * r^2 * h - (27 / 3) * π * r^2 * h)
  V2 / V1 = 37 / 11 :=
by
  intros r h h_pos
  sorry

end cone_section_volume_ratio_l120_12078


namespace no_counterexample_exists_l120_12033

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_counterexample_exists : ∀ n : ℕ, sum_of_digits n % 9 = 0 → n % 9 = 0 :=
by
  intro n h
  sorry

end no_counterexample_exists_l120_12033


namespace ratio_of_teaspoons_to_knives_is_2_to_1_l120_12076

-- Define initial conditions based on the problem
def initial_knives : ℕ := 24
def initial_teaspoons (T : ℕ) : Prop := 
  initial_knives + T + (1 / 3 : ℚ) * initial_knives + (2 / 3 : ℚ) * T = 112

-- Define the ratio to be proved
def ratio_teaspoons_to_knives (T : ℕ) : Prop :=
  initial_teaspoons T ∧ T = 48 ∧ 48 / initial_knives = 2

theorem ratio_of_teaspoons_to_knives_is_2_to_1 : ∃ T, ratio_teaspoons_to_knives T :=
by
  -- Proof would follow here
  sorry

end ratio_of_teaspoons_to_knives_is_2_to_1_l120_12076


namespace sets_equal_l120_12092

-- Defining the sets and proving their equality
theorem sets_equal : { x : ℝ | x^2 + 1 = 0 } = (∅ : Set ℝ) :=
  sorry

end sets_equal_l120_12092


namespace find_a_range_find_value_x1_x2_l120_12055

noncomputable def quadratic_equation_roots_and_discriminant (a : ℝ) :=
  ∃ x1 x2 : ℝ, 
      (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
      (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧
      (x1 ≠ x2) ∧ 
      (∀ Δ > 0, Δ = 9 - 8 * a - 4)

theorem find_a_range (a : ℝ) : 
  (quadratic_equation_roots_and_discriminant a) → a < 5 / 8 :=
sorry

theorem find_value_x1_x2 (a : ℤ) (h : a = 0) (x1 x2 : ℝ) :
  (x1^2 - 3 * x1 + 2 * a + 1 = 0) ∧ 
  (x2^2 - 3 * x2 + 2 * a + 1 = 0) ∧ 
  (x1 + x2 = 3) ∧ 
  (x1 * x2 = 1) → 
  (x1^2 * x2 + x1 * x2^2 = 3) :=
sorry

end find_a_range_find_value_x1_x2_l120_12055


namespace michael_total_fish_l120_12001

-- Definitions based on conditions
def michael_original_fish : ℕ := 31
def ben_fish_given : ℕ := 18

-- Theorem to prove the total number of fish Michael has now
theorem michael_total_fish : (michael_original_fish + ben_fish_given) = 49 :=
by sorry

end michael_total_fish_l120_12001


namespace part1_part2_part3_part4_l120_12006

section QuadraticFunction

variable {x : ℝ} {y : ℝ} 

-- 1. Prove that if a quadratic function y = x^2 + bx - 3 intersects the x-axis at (3, 0), 
-- then b = -2 and the other intersection point is (-1, 0).
theorem part1 (b : ℝ) : 
  ((3:ℝ) ^ 2 + b * (3:ℝ) - 3 = 0) → 
  b = -2 ∧ ∃ x : ℝ, (x = -1 ∧ x^2 + b * x - 3 = 0) := 
  sorry

-- 2. For the function y = x^2 + bx - 3 where b = -2, 
-- prove that when 0 < y < 5, x is in -2 < x < -1 or 3 < x < 4.
theorem part2 (b : ℝ) :
  b = -2 → 
  (0 < y ∧ y < 5 → ∃ x : ℝ, (x^2 + b * x - 3 = y) → (-2 < x ∧ x < -1) ∨ (3 < x ∧ x < 4)) :=
  sorry

-- 3. Prove that the value t such that y = x^2 + bx - 3 and y > t always holds for all x
-- is t < -((b ^ 2 + 12) / 4).
theorem part3 (b t : ℝ) :
  (∀ x : ℝ, (x ^ 2 + b * x - 3 > t)) → t < -(b ^ 2 + 12) / 4 :=
  sorry

-- 4. Given y = x^2 - 3x - 3 and 1 < x < 2, 
-- prove that m < y < n with n = -5, b = -3, and m ≤ -21 / 4.
theorem part4 (m n : ℝ) :
  (1 < x ∧ x < 2 → m < x^2 - 3 * x - 3 ∧ x^2 - 3 * x - 3 < n) →
  n = -5 ∧ -21 / 4 ≤ m :=
  sorry

end QuadraticFunction

end part1_part2_part3_part4_l120_12006


namespace num_choices_l120_12077

theorem num_choices (classes scenic_spots : ℕ) (h_classes : classes = 4) (h_scenic_spots : scenic_spots = 3) :
  (scenic_spots ^ classes) = 81 :=
by
  -- The detailed proof goes here
  sorry

end num_choices_l120_12077


namespace f_at_2_l120_12089

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4

-- State the theorem that we need to prove
theorem f_at_2 : f 2 = 2 := by
  -- the proof will go here
  sorry

end f_at_2_l120_12089


namespace values_of_x_l120_12023

theorem values_of_x (x : ℝ) : (x+2)*(x-9) < 0 ↔ -2 < x ∧ x < 9 := 
by
  sorry

end values_of_x_l120_12023
