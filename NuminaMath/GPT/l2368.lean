import Mathlib

namespace exist_common_divisor_l2368_236893

theorem exist_common_divisor (a : ℕ → ℕ) (m : ℕ) (h_positive : ∀ i, 1 ≤ i ∧ i ≤ m → 0 < a i)
  (p : ℕ → ℤ) (h_poly : ∀ n : ℕ, ∃ i, 1 ≤ i ∧ i ≤ m ∧ (a i : ℤ) ∣ p n) :
  ∃ j, 1 ≤ j ∧ j ≤ m ∧ ∀ n, (a j : ℤ) ∣ p n :=
by
  sorry

end exist_common_divisor_l2368_236893


namespace total_students_l2368_236817

theorem total_students (students_per_classroom : ℕ) (num_classrooms : ℕ) (h1 : students_per_classroom = 30) (h2 : num_classrooms = 13) : students_per_classroom * num_classrooms = 390 :=
by
  -- Begin the proof
  sorry

end total_students_l2368_236817


namespace mike_spent_l2368_236855

def trumpet_price : ℝ := 145.16
def song_book_price : ℝ := 5.84
def total_price : ℝ := 151.00

theorem mike_spent :
  trumpet_price + song_book_price = total_price :=
by
  sorry

end mike_spent_l2368_236855


namespace avg_GPA_is_93_l2368_236851

def avg_GPA_school (GPA_6th GPA_8th : ℕ) (GPA_diff : ℕ) : ℕ :=
  (GPA_6th + (GPA_6th + GPA_diff) + GPA_8th) / 3

theorem avg_GPA_is_93 :
  avg_GPA_school 93 91 2 = 93 :=
by
  -- The proof can be handled here 
  sorry

end avg_GPA_is_93_l2368_236851


namespace factor_polynomial_l2368_236845

theorem factor_polynomial (x y : ℝ) : 
  2*x^2 - x*y - 15*y^2 = (2*x - 5*y) * (x - 3*y) :=
sorry

end factor_polynomial_l2368_236845


namespace number_of_functions_l2368_236890

open Nat

theorem number_of_functions (f : Fin 15 → Fin 15)
  (h : ∀ x, (f (f x) - 2 * f x + x : Int) % 15 = 0) :
  ∃! n : Nat, n = 375 := sorry

end number_of_functions_l2368_236890


namespace not_integer_division_l2368_236836

def P : ℕ := 1
def Q : ℕ := 2

theorem not_integer_division : ¬ (∃ (n : ℤ), (P : ℤ) / (Q : ℤ) = n) := by
sorry

end not_integer_division_l2368_236836


namespace B_alone_can_do_work_in_9_days_l2368_236820

-- Define the conditions
def A_completes_work_in : ℕ := 15
def A_completes_portion_in (days : ℕ) : ℚ := days / 15
def portion_of_work_left (days : ℕ) : ℚ := 1 - A_completes_portion_in days
def B_completes_remaining_work_in_left_days (days_left : ℕ) : ℕ := 6
def B_completes_work_in (days_left : ℕ) : ℚ := B_completes_remaining_work_in_left_days days_left / (portion_of_work_left 5)

-- Define the theorem to be proven
theorem B_alone_can_do_work_in_9_days (days_left : ℕ) : B_completes_work_in days_left = 9 := by
  sorry

end B_alone_can_do_work_in_9_days_l2368_236820


namespace total_cost_maria_l2368_236847

-- Define the cost of the pencil
def cost_pencil : ℕ := 8

-- Define the cost of the pen as half the price of the pencil
def cost_pen : ℕ := cost_pencil / 2

-- Define the total cost for both the pen and the pencil
def total_cost : ℕ := cost_pencil + cost_pen

-- Prove that total cost is equal to 12
theorem total_cost_maria : total_cost = 12 := 
by
  -- skip the proof
  sorry

end total_cost_maria_l2368_236847


namespace calculate_expression_value_l2368_236864

theorem calculate_expression_value : 
  3 - ((-3 : ℚ) ^ (-3 : ℤ) * 2) = 83 / 27 := 
by
  sorry

end calculate_expression_value_l2368_236864


namespace fractional_sum_identity_l2368_236819

noncomputable def distinct_real_roots (f : ℝ → ℝ) (a b c : ℝ) : Prop :=
f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem fractional_sum_identity :
  ∀ (p q r A B C : ℝ),
  (x^3 - 22*x^2 + 80*x - 67 = (x - p) * (x - q) * (x - r)) →
  distinct_real_roots (λ x => x^3 - 22*x^2 + 80*x - 67) p q r →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 22*s^2 + 80*s - 67) = A / (s - p) + B / (s - q) + C / (s - r)) →
  (1 / (A) + 1 / (B) + 1 / (C) = 244) :=
by 
  intros p q r A B C h_poly h_distinct h_fractional
  sorry

end fractional_sum_identity_l2368_236819


namespace A_plays_D_third_day_l2368_236873

section GoTournament

variables (Player : Type) (A B C D : Player) 

-- Define the condition that each player competes with every other player exactly once.
def each_plays_once (P : Player → Player → Prop) : Prop :=
  ∀ x y, x ≠ y → (P x y ∨ P y x)

-- Define the tournament setup and the play conditions.
variables (P : Player → Player → Prop)
variable [∀ x y, Decidable (P x y)] -- Assuming decidability for the play relation

-- The given conditions of the problem
axiom A_plays_C_first_day : P A C
axiom C_plays_D_second_day : P C D
axiom only_one_match_per_day : ∀ x, ∃! y, P x y

-- We aim to prove that A will play against D on the third day.
theorem A_plays_D_third_day : P A D :=
sorry

end GoTournament

end A_plays_D_third_day_l2368_236873


namespace sum_of_selected_primes_divisible_by_3_probability_l2368_236872

def first_fifteen_primes : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def count_combinations_divisible_3 (nums : List ℕ) (k : ℕ) : ℕ :=
sorry -- Combines over the list to count combinations summing divisible by 3

noncomputable def probability_divisible_by_3 : ℚ :=
  let total_combinations := (Nat.choose 15 4)
  let favorable_combinations := count_combinations_divisible_3 first_fifteen_primes 4
  favorable_combinations / total_combinations

theorem sum_of_selected_primes_divisible_by_3_probability :
  probability_divisible_by_3 = 1/3 :=
sorry

end sum_of_selected_primes_divisible_by_3_probability_l2368_236872


namespace number_of_roses_picked_later_l2368_236833

-- Given definitions
def initial_roses : ℕ := 50
def sold_roses : ℕ := 15
def final_roses : ℕ := 56

-- Compute the number of roses left after selling.
def roses_left := initial_roses - sold_roses

-- Define the final goal: number of roses picked later.
def picked_roses_later := final_roses - roses_left

-- State the theorem
theorem number_of_roses_picked_later : picked_roses_later = 21 :=
by
  sorry

end number_of_roses_picked_later_l2368_236833


namespace value_of_xyz_l2368_236834

variable (x y z : ℝ)

theorem value_of_xyz (h1 : (x + y + z) * (x * y + x * z + y * z) = 36)
                     (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 22) 
                     : x * y * z = 14 / 3 := 
sorry

end value_of_xyz_l2368_236834


namespace problem_l2368_236859

theorem problem (a b c : ℂ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6) :
  (a - 1)^(2023) + (b - 1)^(2023) + (c - 1)^(2023) = 0 :=
by
  sorry

end problem_l2368_236859


namespace gcf_72_108_l2368_236843

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l2368_236843


namespace cupboard_cost_price_l2368_236806

theorem cupboard_cost_price (C : ℝ) 
  (h1 : ∀ C₀, C = C₀ → C₀ * 0.88 + 1500 = C₀ * 1.12) :
  C = 6250 := by
  sorry

end cupboard_cost_price_l2368_236806


namespace price_per_unit_l2368_236809

theorem price_per_unit (x y : ℝ) 
    (h1 : 2 * x + 3 * y = 690) 
    (h2 : x + 4 * y = 720) : 
    x = 120 ∧ y = 150 := 
by 
    sorry

end price_per_unit_l2368_236809


namespace platform_length_is_correct_l2368_236860

noncomputable def length_of_platform (time_to_pass_man : ℝ) (time_to_cross_platform : ℝ) (length_of_train : ℝ) : ℝ := 
  length_of_train * time_to_cross_platform / time_to_pass_man - length_of_train

theorem platform_length_is_correct : length_of_platform 8 20 178 = 267 := 
  sorry

end platform_length_is_correct_l2368_236860


namespace Robie_chocolates_left_l2368_236880

def initial_bags : ℕ := 3
def given_away : ℕ := 2
def additional_bags : ℕ := 3

theorem Robie_chocolates_left : (initial_bags - given_away) + additional_bags = 4 :=
by
  sorry

end Robie_chocolates_left_l2368_236880


namespace gcd_228_1995_l2368_236823

theorem gcd_228_1995 : Int.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l2368_236823


namespace min_value_x_plus_2_div_x_minus_2_l2368_236858

theorem min_value_x_plus_2_div_x_minus_2 (x : ℝ) (h : x > 2) : 
  ∃ m, m = 2 + 2 * Real.sqrt 2 ∧ x + 2/(x-2) ≥ m :=
by sorry

end min_value_x_plus_2_div_x_minus_2_l2368_236858


namespace opposite_of_2023_l2368_236899

def opposite (n : Int) : Int := -n

theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l2368_236899


namespace number_of_children_per_seat_l2368_236898

variable (children : ℕ) (seats : ℕ)

theorem number_of_children_per_seat (h1 : children = 58) (h2 : seats = 29) :
  children / seats = 2 := by
  sorry

end number_of_children_per_seat_l2368_236898


namespace smallest_integral_value_k_l2368_236813

-- Define the discriminant of the quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the quadratic equation
def quadratic (k : ℝ) (x : ℝ) : ℝ := 3 * x * (k * x - 5) - x^2 + 4

-- Define the condition for the quadratic equation having no real roots
def no_real_roots (k : ℝ) : Prop :=
  let a := 3 * k - 1
  let b := -15
  let c := 4
  discriminant a b c < 0

-- The Lean 4 statement to find the smallest integral value of k such that the quadratic has no real roots
theorem smallest_integral_value_k : ∃ (k : ℤ), no_real_roots k ∧ (∀ (m : ℤ), no_real_roots m → k ≤ m) :=
  sorry

end smallest_integral_value_k_l2368_236813


namespace tori_current_height_l2368_236875

theorem tori_current_height :
  let original_height := 4.4
  let growth := 2.86
  original_height + growth = 7.26 := 
by
  sorry

end tori_current_height_l2368_236875


namespace sum_of_primes_less_than_twenty_is_77_l2368_236816

-- Define prime numbers less than 20
def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the sum of those primes
def sum_primes_less_than_twenty : ℕ := primes_less_than_twenty.sum

-- The theorem to prove
theorem sum_of_primes_less_than_twenty_is_77 : sum_primes_less_than_twenty = 77 :=
by
  sorry

end sum_of_primes_less_than_twenty_is_77_l2368_236816


namespace rectangle_width_length_ratio_l2368_236885

theorem rectangle_width_length_ratio (w l P : ℕ) (hP : P = 30) (hl : l = 10) (h_perimeter : P = 2*l + 2*w) :
  w / l = 1 / 2 :=
by
  sorry

end rectangle_width_length_ratio_l2368_236885


namespace minimum_value_of_T_l2368_236848

theorem minimum_value_of_T (a b c : ℝ) (h1 : ∀ x : ℝ, (1 / a) * x^2 + b * x + c ≥ 0) (h2 : a * b > 1) :
  ∃ T : ℝ, T = 4 ∧ T = (1 / (2 * (a * b - 1))) + (a * (b + 2 * c) / (a * b - 1)) :=
by
  sorry

end minimum_value_of_T_l2368_236848


namespace larger_number_is_seventy_two_l2368_236856

def five_times_larger_is_six_times_smaller (x y : ℕ) : Prop := 5 * y = 6 * x
def difference_is_twelve (x y : ℕ) : Prop := y - x = 12

theorem larger_number_is_seventy_two (x y : ℕ) 
  (h1 : five_times_larger_is_six_times_smaller x y)
  (h2 : difference_is_twelve x y) : y = 72 :=
sorry

end larger_number_is_seventy_two_l2368_236856


namespace find_b_value_l2368_236818

theorem find_b_value
  (b : ℝ)
  (eq1 : ∀ y x, 3 * y - 3 * b = 9 * x)
  (eq2 : ∀ y x, y - 2 = (b + 9) * x)
  (parallel : ∀ y1 y2 x1 x2, 
    (3 * y1 - 3 * b = 9 * x1) ∧ (y2 - 2 = (b + 9) * x2) → 
    ((3 * x1 = (b + 9) * x2) ↔ (3 = b + 9)))
  : b = -6 := 
  sorry

end find_b_value_l2368_236818


namespace algebraic_expression_value_l2368_236838

theorem algebraic_expression_value (m x n : ℝ)
  (h1 : (m + 3) * x ^ (|m| - 2) + 6 * m = 0)
  (h2 : n * x - 5 = x * (3 - n))
  (h3 : |m| = 2)
  (h4 : (m + 3) ≠ 0) :
  (m + x) ^ 2000 * (-m ^ 2 * n + x * n ^ 2) + 1 = 1 := by
  sorry

end algebraic_expression_value_l2368_236838


namespace max_xy_under_constraint_l2368_236846

theorem max_xy_under_constraint (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : x > 0) (h3 : y > 0) : 
  xy ≤ 1 / 8 
  := sorry

end max_xy_under_constraint_l2368_236846


namespace range_of_function_l2368_236863

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, y = (1 / 2) ^ (x^2 + 2 * x - 1)) ↔ (0 < y ∧ y ≤ 4) :=
by
  sorry

end range_of_function_l2368_236863


namespace find_m_l2368_236888

theorem find_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = m) : m = 4 :=
sorry

end find_m_l2368_236888


namespace bases_with_final_digit_one_in_360_l2368_236803

theorem bases_with_final_digit_one_in_360 (b : ℕ) (h : 2 ≤ b ∧ b ≤ 9) : ¬(b ∣ 359) :=
by
  sorry

end bases_with_final_digit_one_in_360_l2368_236803


namespace jim_gas_tank_capacity_l2368_236886

/-- Jim has 2/3 of a tank left after a round-trip of 20 miles where he gets 5 miles per gallon.
    Prove that the capacity of Jim's gas tank is 12 gallons. --/
theorem jim_gas_tank_capacity
    (remaining_fraction : ℚ)
    (round_trip_distance : ℚ)
    (fuel_efficiency : ℚ)
    (used_fraction : ℚ)
    (used_gallons : ℚ)
    (total_capacity : ℚ)
    (h1 : remaining_fraction = 2/3)
    (h2 : round_trip_distance = 20)
    (h3 : fuel_efficiency = 5)
    (h4 : used_fraction = 1 - remaining_fraction)
    (h5 : used_gallons = round_trip_distance / fuel_efficiency)
    (h6 : used_gallons = used_fraction * total_capacity) :
  total_capacity = 12 :=
sorry

end jim_gas_tank_capacity_l2368_236886


namespace number_of_girls_in_first_year_l2368_236841

theorem number_of_girls_in_first_year
  (total_students : ℕ)
  (sample_size : ℕ)
  (boys_in_sample : ℕ)
  (girls_in_first_year : ℕ) :
  total_students = 2400 →
  sample_size = 80 →
  boys_in_sample = 42 →
  girls_in_first_year = total_students * (sample_size - boys_in_sample) / sample_size →
  girls_in_first_year = 1140 :=
by 
  intros h1 h2 h3 h4
  sorry

end number_of_girls_in_first_year_l2368_236841


namespace max_area_triangle_l2368_236812

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

noncomputable def line_eq (x y : ℝ) : Prop := 2 * Real.sqrt 2 * x - y - 1 = 0

theorem max_area_triangle (x1 y1 x2 y2 xp yp : ℝ) (h1 : circle_eq x1 y1) (h2 : circle_eq x2 y2) (h3 : circle_eq xp yp)
  (h4 : line_eq x1 y1) (h5 : line_eq x2 y2) (h6 : (xp, yp) ≠ (x1, y1)) (h7 : (xp, yp) ≠ (x2, y2)) :
  ∃ S : ℝ, S = 10 * Real.sqrt 5 / 9 :=
by
  sorry

end max_area_triangle_l2368_236812


namespace cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l2368_236866

-- 1) Cylinder
theorem cylinder_is_defined (R : ℝ) :
  ∀ (x y z : ℝ), x^2 + y^2 = R^2 → ∃ (r : ℝ), r = R ∧ x^2 + y^2 = r^2 :=
sorry

-- 2) Sphere
theorem sphere_is_defined (R : ℝ) :
  ∀ (x y z : ℝ), x^2 + y^2 + z^2 = R^2 → ∃ (r : ℝ), r = R ∧ x^2 + y^2 + z^2 = r^2 :=
sorry

-- 3) Hyperbolic Cylinder
theorem hyperbolic_cylinder_is_defined (m : ℝ) :
  ∀ (x y z : ℝ), xy = m → ∃ (k : ℝ), k = m ∧ xy = k :=
sorry

-- 4) Parabolic Cylinder
theorem parabolic_cylinder_is_defined :
  ∀ (x z : ℝ), z = x^2 → ∃ (k : ℝ), k = 1 ∧ z = k*x^2 :=
sorry

end cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l2368_236866


namespace fraction_sum_l2368_236876

theorem fraction_sum : (1 / 4 : ℚ) + (3 / 8) = 5 / 8 :=
by
  sorry

end fraction_sum_l2368_236876


namespace least_value_QGK_l2368_236854

theorem least_value_QGK :
  ∃ (G K Q : ℕ), (10 * G + G) * G = 100 * Q + 10 * G + K ∧ G ≠ K ∧ (10 * G + G) ≥ 10 ∧ (10 * G + G) < 100 ∧  ∃ x, x = 44 ∧ 100 * G + 10 * 4 + 4 = (100 * Q + 10 * G + K) ∧ 100 * 0 + 10 * 4 + 4 = 044  :=
by
  sorry

end least_value_QGK_l2368_236854


namespace john_marks_wrongly_entered_as_l2368_236894

-- Definitions based on the conditions
def john_correct_marks : ℤ := 62
def num_students : ℤ := 80
def avg_increase : ℤ := 1/2
def total_increase : ℤ := num_students * avg_increase

-- Statement to prove
theorem john_marks_wrongly_entered_as (x : ℤ) :
  (total_increase = (x - john_correct_marks)) → x = 102 :=
by {
  -- Placeholder for proof
  sorry
}

end john_marks_wrongly_entered_as_l2368_236894


namespace geometric_sequence_sum_of_first_four_terms_l2368_236862

theorem geometric_sequence_sum_of_first_four_terms 
  (a q : ℝ)
  (h1 : a * (1 + q) = 7)
  (h2 : a * (q^6 - 1) / (q - 1) = 91) :
  a * (1 + q + q^2 + q^3) = 28 := by
  sorry

end geometric_sequence_sum_of_first_four_terms_l2368_236862


namespace rationalize_denominator_eq_l2368_236850

noncomputable def rationalize_denominator : ℝ :=
  18 / (Real.sqrt 36 + Real.sqrt 2)

theorem rationalize_denominator_eq : rationalize_denominator = (54 / 17) - (9 * Real.sqrt 2 / 17) := 
by
  sorry

end rationalize_denominator_eq_l2368_236850


namespace initial_erasers_calculation_l2368_236874

variable (initial_erasers added_erasers total_erasers : ℕ)

theorem initial_erasers_calculation
  (total_erasers_eq : total_erasers = 270)
  (added_erasers_eq : added_erasers = 131) :
  initial_erasers = total_erasers - added_erasers → initial_erasers = 139 := by
  intro h
  rw [total_erasers_eq, added_erasers_eq] at h
  simp at h
  exact h

end initial_erasers_calculation_l2368_236874


namespace root_k_value_l2368_236878

theorem root_k_value
  (k : ℝ)
  (h : Polynomial.eval 4 (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 3 * Polynomial.X - Polynomial.C k) = 0) :
  k = 44 :=
sorry

end root_k_value_l2368_236878


namespace carpet_area_l2368_236821

def width : ℝ := 8
def length : ℝ := 1.5

theorem carpet_area : width * length = 12 := by
  sorry

end carpet_area_l2368_236821


namespace pictures_per_album_l2368_236822

-- Definitions based on the conditions
def phone_pics := 35
def camera_pics := 5
def total_pics := phone_pics + camera_pics
def albums := 5 

-- Statement that needs to be proven
theorem pictures_per_album : total_pics / albums = 8 := by
  sorry

end pictures_per_album_l2368_236822


namespace sequence_value_l2368_236849

theorem sequence_value : 
  ∃ (x y r : ℝ), 
    (4096 * r = 1024) ∧ 
    (1024 * r = 256) ∧ 
    (256 * r = x) ∧ 
    (x * r = y) ∧ 
    (y * r = 4) ∧  
    (4 * r = 1) ∧ 
    (x + y = 80) :=
by
  sorry

end sequence_value_l2368_236849


namespace find_room_length_l2368_236884

variable (w : ℝ) (C : ℝ) (r : ℝ)

theorem find_room_length (h_w : w = 4.75) (h_C : C = 29925) (h_r : r = 900) : (C / r) / w = 7 := by
  sorry

end find_room_length_l2368_236884


namespace logical_equivalence_l2368_236879

variables (P Q : Prop)

theorem logical_equivalence :
  (¬P → ¬Q) ↔ (Q → P) :=
sorry

end logical_equivalence_l2368_236879


namespace music_students_count_l2368_236804

open Nat

theorem music_students_count (total_students : ℕ) (art_students : ℕ) (both_music_art : ℕ) 
      (neither_music_art : ℕ) (M : ℕ) :
    total_students = 500 →
    art_students = 10 →
    both_music_art = 10 →
    neither_music_art = 470 →
    (total_students - neither_music_art) = 30 →
    (M + (art_students - both_music_art)) = 30 →
    M = 30 :=
by
  intros h_total h_art h_both h_neither h_music_art_total h_music_count
  sorry

end music_students_count_l2368_236804


namespace sum_of_squares_l2368_236869

theorem sum_of_squares (a b : ℝ) (h1 : a + b = 16) (h2 : a * b = 20) : a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l2368_236869


namespace symmetry_axis_of_sine_function_l2368_236840

theorem symmetry_axis_of_sine_function (x : ℝ) :
  (∃ k : ℤ, 2 * x + π / 4 = k * π + π / 2) ↔ x = π / 8 :=
by sorry

end symmetry_axis_of_sine_function_l2368_236840


namespace simplify_expression_l2368_236829

theorem simplify_expression :
  (Real.sqrt 15 + Real.sqrt 45 - (Real.sqrt (4/3) - Real.sqrt 108)) = 
  (Real.sqrt 15 + 3 * Real.sqrt 5 + 16 * Real.sqrt 3 / 3) :=
by
  sorry

end simplify_expression_l2368_236829


namespace intersection_of_sets_l2368_236882
-- Define the sets and the proof statement
theorem intersection_of_sets : 
  let A := { x : ℝ | x^2 - 3 * x - 4 < 0 }
  let B := {-4, 1, 3, 5}
  A ∩ B = {1, 3} :=
by
  sorry

end intersection_of_sets_l2368_236882


namespace allocate_teaching_positions_l2368_236887

theorem allocate_teaching_positions :
  ∃ (ways : ℕ), ways = 10 ∧ 
    (∃ (a b c : ℕ), a + b + c = 8 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 2 ≤ a) := 
sorry

end allocate_teaching_positions_l2368_236887


namespace solution_to_system_l2368_236857

theorem solution_to_system : ∃ x y : ℤ, (2 * x + 3 * y = -11 ∧ 6 * x - 5 * y = 9) ↔ (x = -1 ∧ y = -3) :=
by
  sorry

end solution_to_system_l2368_236857


namespace coal_removal_date_l2368_236808

theorem coal_removal_date (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : 25 * m + 9 * n = 0.5)
  (h4 : ∃ z : ℝ,  z * (n + m) = 0.5)
  (h5 : ∀ z : ℝ, z = 12 → (16 + z) * m = (9 + z) * n):
  ∃ t : ℝ, t = 28 := 
by 
{
  sorry
}

end coal_removal_date_l2368_236808


namespace units_digit_of_150_factorial_is_zero_l2368_236825

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l2368_236825


namespace son_l2368_236891

def woman's_age (W S : ℕ) : Prop := W = 2 * S + 3
def sum_of_ages (W S : ℕ) : Prop := W + S = 84

theorem son's_age_is_27 (W S : ℕ) (h1: woman's_age W S) (h2: sum_of_ages W S) : S = 27 :=
by
  sorry

end son_l2368_236891


namespace ring_stack_vertical_distance_l2368_236828

theorem ring_stack_vertical_distance :
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  ∃ n, (top_ring_outer_diameter - bottom_ring_outer_diameter) / decrement + 1 = n ∧
       n * ring_thickness = 260 :=
by {
  let ring_thickness := 2
  let top_ring_outer_diameter := 36
  let bottom_ring_outer_diameter := 12
  let decrement := 2
  sorry
}

end ring_stack_vertical_distance_l2368_236828


namespace z_sum_of_squares_eq_101_l2368_236802

open Complex

noncomputable def z_distances_sum_of_squares (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : ℝ :=
  abs (z - (1 + 1 * I)) ^ 2 + abs (z - (5 - 5 * I)) ^ 2

theorem z_sum_of_squares_eq_101 (z : ℂ) (h : abs (z - (3 + -3 * I)) = 3) : 
  z_distances_sum_of_squares z h = 101 :=
by
  sorry

end z_sum_of_squares_eq_101_l2368_236802


namespace sum_faces_edges_vertices_eq_26_l2368_236895

-- We define the number of faces, edges, and vertices of a rectangular prism.
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def num_vertices : ℕ := 8

-- The theorem we want to prove.
theorem sum_faces_edges_vertices_eq_26 :
  num_faces + num_edges + num_vertices = 26 :=
by
  -- This is where the proof would go.
  sorry

end sum_faces_edges_vertices_eq_26_l2368_236895


namespace sheets_borrowed_l2368_236892

theorem sheets_borrowed (pages sheets borrowed remaining_sheets : ℕ) 
  (h1 : pages = 70) 
  (h2 : sheets = 35)
  (h3 : remaining_sheets = sheets - borrowed)
  (h4 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ remaining_sheets -> 2*i-1 <= pages) 
  (h5 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ remaining_sheets -> i + 1 != borrowed ∧ i <= remaining_sheets)
  (avg : ℕ) (h6 : avg = 28)
  : borrowed = 17 := by
  sorry

end sheets_borrowed_l2368_236892


namespace repave_today_l2368_236815

theorem repave_today (total_repaved : ℕ) (repaved_before_today : ℕ) (repaved_today : ℕ) :
  total_repaved = 4938 → repaved_before_today = 4133 → repaved_today = total_repaved - repaved_before_today → repaved_today = 805 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end repave_today_l2368_236815


namespace cut_wire_l2368_236853

theorem cut_wire (x y : ℕ) : 
  15 * x + 12 * y = 102 ↔ (x = 2 ∧ y = 6) ∨ (x = 6 ∧ y = 1) :=
by
  sorry

end cut_wire_l2368_236853


namespace sum_of_integers_product_neg17_l2368_236832

theorem sum_of_integers_product_neg17 (a b c : ℤ) (h : a * b * c = -17) : a + b + c = -15 ∨ a + b + c = 17 :=
sorry

end sum_of_integers_product_neg17_l2368_236832


namespace find_y_l2368_236870

theorem find_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := 
sorry

end find_y_l2368_236870


namespace find_rate_l2368_236805

noncomputable def national_bank_interest_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ): ℚ :=
  (total_income - (investment_additional * additional_rate)) / investment_national

theorem find_rate (total_income: ℚ) (investment_national: ℚ) (investment_additional: ℚ) (additional_rate: ℚ) (total_investment_rate: ℚ) (correct_rate: ℚ):
  investment_national = 2400 → investment_additional = 600 → additional_rate = 0.10 → total_investment_rate = 0.06 → total_income = total_investment_rate * (investment_national + investment_additional) → correct_rate = 0.05 → national_bank_interest_rate total_income investment_national investment_additional additional_rate total_investment_rate = correct_rate :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end find_rate_l2368_236805


namespace gear_revolutions_difference_l2368_236824

noncomputable def gear_revolution_difference (t : ℕ) : ℕ :=
  let p := 10 * t
  let q := 40 * t
  q - p

theorem gear_revolutions_difference (t : ℕ) : gear_revolution_difference t = 30 * t :=
by
  sorry

end gear_revolutions_difference_l2368_236824


namespace rebus_solution_l2368_236896

-- We state the conditions:
variables (A B Γ D : ℤ)

-- Define the correct values
def A_correct := 2
def B_correct := 7
def Γ_correct := 1
def D_correct := 0

-- State the conditions as assumptions
axiom cond1 : A * B + 8 = 3 * B
axiom cond2 : Γ * D + B = 5  -- Adjusted assuming V = 5 from problem data
axiom cond3 : Γ * B + 3 = A * D

-- State the goal to be proved
theorem rebus_solution : A = A_correct ∧ B = B_correct ∧ Γ = Γ_correct ∧ D = D_correct :=
by
  sorry

end rebus_solution_l2368_236896


namespace problem_l2368_236811

variable (x y : ℝ)

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 18) : (x - y)^2 = 9 :=
by
  sorry

end problem_l2368_236811


namespace river_current_speed_l2368_236889

variable (c : ℝ)

def boat_speed_still_water : ℝ := 20
def round_trip_distance : ℝ := 182
def round_trip_time : ℝ := 10

theorem river_current_speed (h : (91 / (boat_speed_still_water - c)) + (91 / (boat_speed_still_water + c)) = round_trip_time) : c = 6 :=
sorry

end river_current_speed_l2368_236889


namespace solve_quadratic_eq_l2368_236897

theorem solve_quadratic_eq (x : ℝ) :
  x^2 - 7 * x + 6 = 0 ↔ x = 1 ∨ x = 6 :=
by
  sorry

end solve_quadratic_eq_l2368_236897


namespace ratio_of_girls_to_boys_l2368_236868

variables (g b : ℕ)

theorem ratio_of_girls_to_boys (h₁ : b = g - 6) (h₂ : g + b = 36) :
  (g / gcd g b) / (b / gcd g b) = 7 / 5 :=
by
  sorry

end ratio_of_girls_to_boys_l2368_236868


namespace quadratic_solution_l2368_236830

theorem quadratic_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : x + 2 / x = 4 :=
by sorry

end quadratic_solution_l2368_236830


namespace initial_apples_proof_l2368_236865

-- Define the variables and conditions
def initial_apples (handed_out: ℕ) (pies: ℕ) (apples_per_pie: ℕ): ℕ := 
  handed_out + pies * apples_per_pie

-- Define the proof statement
theorem initial_apples_proof : initial_apples 30 7 8 = 86 := by 
  sorry

end initial_apples_proof_l2368_236865


namespace christine_siri_total_money_l2368_236844

-- Define the conditions
def christine_has_more_than_siri : ℝ := 20 -- Christine has 20 rs more than Siri
def christine_amount : ℝ := 20.5 -- Christine has 20.5 rs

-- Define the proof problem
theorem christine_siri_total_money :
  (∃ (siri_amount : ℝ), christine_amount = siri_amount + christine_has_more_than_siri) →
  ∃ total : ℝ, total = christine_amount + (christine_amount - christine_has_more_than_siri) ∧ total = 21 :=
by sorry

end christine_siri_total_money_l2368_236844


namespace period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l2368_236814

noncomputable def f (x a : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x) ^ 2 + a

theorem period_of_f : ∀ a : ℝ, ∀ x : ℝ, f (x + π) a = f x a := 
by sorry

theorem minimum_value_zero_then_a_eq_one : (∀ x : ℝ, f x a ≥ 0) → a = 1 := 
by sorry

theorem maximum_value_of_f : a = 1 → (∀ x : ℝ, f x 1 ≤ 4) :=
by sorry

theorem axis_of_symmetry : a = 1 → ∃ k : ℤ, ∀ x : ℝ, 2 * x + π / 6 = k * π + π / 2 ↔ f x 1 = f 0 1 :=
by sorry

end period_of_f_minimum_value_zero_then_a_eq_one_maximum_value_of_f_axis_of_symmetry_l2368_236814


namespace quadratic_equal_roots_l2368_236827

theorem quadratic_equal_roots (a : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ (x * (x + 1) + a * x = 0) ∧ ((1 + a)^2 = 0)) →
  a = -1 :=
by
  sorry

end quadratic_equal_roots_l2368_236827


namespace find_a_plus_b_l2368_236801

variables (a b c d x : ℝ)

def conditions (a b c d x : ℝ) : Prop :=
  (a + b = x) ∧
  (b + c = 9) ∧
  (c + d = 3) ∧
  (a + d = 5)

theorem find_a_plus_b (a b c d x : ℝ) (h : conditions a b c d x) : a + b = 11 :=
by
  have h1 : a + b = x := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : a + d = 5 := h.2.2.2
  sorry

end find_a_plus_b_l2368_236801


namespace gcd_n_squared_plus_4_n_plus_3_l2368_236852

theorem gcd_n_squared_plus_4_n_plus_3 (n : ℕ) (hn_gt_four : n > 4) : 
  (gcd (n^2 + 4) (n + 3)) = if n % 13 = 10 then 13 else 1 := 
sorry

end gcd_n_squared_plus_4_n_plus_3_l2368_236852


namespace largest_6_digit_div_by_88_l2368_236867

theorem largest_6_digit_div_by_88 : ∃ n : ℕ, 100000 ≤ n ∧ n ≤ 999999 ∧ 88 ∣ n ∧ (∀ m : ℕ, 100000 ≤ m ∧ m ≤ 999999 ∧ 88 ∣ m → m ≤ n) ∧ n = 999944 :=
by
  sorry

end largest_6_digit_div_by_88_l2368_236867


namespace ratio_white_to_remaining_l2368_236807

def total_beans : ℕ := 572

def red_beans (total : ℕ) : ℕ := total / 4

def remaining_beans_after_red (total : ℕ) (red : ℕ) : ℕ := total - red

def green_beans : ℕ := 143

def remaining_beans_after_green (remaining : ℕ) (green : ℕ) : ℕ := remaining - green

def white_beans (remaining : ℕ) : ℕ := remaining / 2

theorem ratio_white_to_remaining (total : ℕ) (red : ℕ) (remaining : ℕ) (green : ℕ) (white : ℕ) 
  (H_total : total = 572)
  (H_red : red = red_beans total)
  (H_remaining : remaining = remaining_beans_after_red total red)
  (H_green : green = 143)
  (H_remaining_after_green : remaining_beans_after_green remaining green = white)
  (H_white : white = white_beans remaining) :
  (white : ℚ) / (remaining : ℚ) = (1 : ℚ) / 2 := 
by sorry

end ratio_white_to_remaining_l2368_236807


namespace rhombus_area_l2368_236881

theorem rhombus_area (x y : ℝ) (h : |x - 1| + |y - 1| = 1) : 
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end rhombus_area_l2368_236881


namespace cars_difference_proof_l2368_236842

theorem cars_difference_proof (U M : ℕ) :
  let initial_cars := 150
  let total_cars := 196
  let cars_from_uncle := U
  let cars_from_grandpa := 2 * U
  let cars_from_dad := 10
  let cars_from_auntie := U + 1
  let cars_from_mum := M
  let total_given_cars := cars_from_dad + cars_from_auntie + cars_from_uncle + cars_from_grandpa + cars_from_mum
  initial_cars + total_given_cars = total_cars ->
  (cars_from_mum - cars_from_dad = 5) := 
by
  sorry

end cars_difference_proof_l2368_236842


namespace find_t_l2368_236826

-- Definitions of the vectors and parallel condition
def a : ℝ × ℝ := (-1, 1)
def b (t : ℝ) : ℝ × ℝ := (3, t)
def is_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v ∨ v = k • u

-- The theorem statement
theorem find_t (t : ℝ) (h : is_parallel (b t) (a + b t)) : t = -3 := by
  sorry

end find_t_l2368_236826


namespace pythagorean_set_A_l2368_236837

theorem pythagorean_set_A : 
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  x^2 + y^2 = z^2 := 
by
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  let z := Real.sqrt 5
  sorry

end pythagorean_set_A_l2368_236837


namespace burger_cost_l2368_236871

theorem burger_cost
  (B P : ℝ)
  (h₁ : P = 2 * B)
  (h₂ : P + 3 * B = 45) :
  B = 9 := by
  sorry

end burger_cost_l2368_236871


namespace min_AB_DE_l2368_236883

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def line_through_focus (k x y : ℝ) : Prop := y = k * (x - 1)

theorem min_AB_DE 
(F : (ℝ × ℝ)) 
(A B D E : ℝ × ℝ) 
(k1 k2 : ℝ) 
(hF : F = (1, 0)) 
(hk : k1^2 + k2^2 = 1) 
(hAB : ∀ x y, parabola x y → line_through_focus k1 x y → A = (x, y) ∨ B = (x, y)) 
(hDE : ∀ x y, parabola x y → line_through_focus k2 x y → D = (x, y) ∨ E = (x, y)) 
: |(A.1 - B.1)| + |(D.1 - E.1)| ≥ 24 := 
sorry

end min_AB_DE_l2368_236883


namespace distance_travelled_l2368_236831

variables (S D : ℝ)

-- conditions
def cond1 : Prop := D = S * 7
def cond2 : Prop := D = (S + 12) * 5

-- Define the main theorem
theorem distance_travelled (h1 : cond1 S D) (h2 : cond2 S D) : D = 210 :=
by {
  sorry
}

end distance_travelled_l2368_236831


namespace average_expenditure_whole_week_l2368_236835

theorem average_expenditure_whole_week (a b : ℕ) (h₁ : a = 3 * 350) (h₂ : b = 4 * 420) : 
  (a + b) / 7 = 390 :=
by 
  sorry

end average_expenditure_whole_week_l2368_236835


namespace division_of_fractions_l2368_236839

theorem division_of_fractions :
  (10 / 21) / (4 / 9) = 15 / 14 :=
by
  -- Proof will be provided here 
  sorry

end division_of_fractions_l2368_236839


namespace phil_won_more_games_than_charlie_l2368_236800

theorem phil_won_more_games_than_charlie :
  ∀ (P D C Ph : ℕ),
  (P = D + 5) → (C = D - 2) → (Ph = 12) → (P = Ph + 4) →
  Ph - C = 3 :=
by
  intros P D C Ph hP hC hPh hPPh
  sorry

end phil_won_more_games_than_charlie_l2368_236800


namespace at_least_one_zero_of_product_zero_l2368_236810

theorem at_least_one_zero_of_product_zero (a b c : ℝ) (h : a * b * c = 0) : a = 0 ∨ b = 0 ∨ c = 0 := by
  sorry

end at_least_one_zero_of_product_zero_l2368_236810


namespace remainder_of_3_pow_19_mod_10_l2368_236861

-- Definition of the problem and conditions
def q := 3^19

-- Statement to prove
theorem remainder_of_3_pow_19_mod_10 : q % 10 = 7 :=
by
  sorry

end remainder_of_3_pow_19_mod_10_l2368_236861


namespace probability_red_or_white_l2368_236877

-- Definitions based on the conditions
def total_marbles := 20
def blue_marbles := 5
def red_marbles := 9
def white_marbles := total_marbles - (blue_marbles + red_marbles)

-- Prove that the probability of selecting a red or white marble is 3/4
theorem probability_red_or_white : (red_marbles + white_marbles : ℚ) / total_marbles = 3 / 4 :=
by sorry

end probability_red_or_white_l2368_236877
