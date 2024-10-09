import Mathlib

namespace oshea_bought_basil_seeds_l1808_180812

-- Define the number of large and small planters and their capacities.
def large_planters := 4
def seeds_per_large_planter := 20
def small_planters := 30
def seeds_per_small_planter := 4

-- The theorem statement: Oshea bought 200 basil seeds
theorem oshea_bought_basil_seeds :
  large_planters * seeds_per_large_planter + small_planters * seeds_per_small_planter = 200 :=
by sorry

end oshea_bought_basil_seeds_l1808_180812


namespace fastest_route_time_l1808_180898

theorem fastest_route_time (d1 d2 : ℕ) (s1 s2 : ℕ) (h1 : d1 = 1500) (h2 : d2 = 750) (h3 : s1 = 75) (h4 : s2 = 25) :
  min (d1 / s1) (d2 / s2) = 20 := by
  sorry

end fastest_route_time_l1808_180898


namespace find_a_from_circle_and_chord_l1808_180894

theorem find_a_from_circle_and_chord 
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x - 2*y + a = 0)
  (line_eq : ∀ x y : ℝ, x + y + 2 = 0)
  (chord_length : ∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 + 2*x1 - 2*y1 + a = 0 ∧ x2^2 + y2^2 + 2*x2 - 2*y2 + a = 0 ∧ x1 + y1 + 2 = 0 ∧ x2 + y2 + 2 = 0 → (x1 - x2)^2 + (y1 - y2)^2 = 16) :
  a = -4 :=
by
  sorry

end find_a_from_circle_and_chord_l1808_180894


namespace bill_milk_problem_l1808_180879

theorem bill_milk_problem 
  (M : ℚ) 
  (sour_cream_milk : ℚ := M / 4)
  (butter_milk : ℚ := M / 4)
  (whole_milk : ℚ := M / 2)
  (sour_cream_gallons : ℚ := sour_cream_milk / 2)
  (butter_gallons : ℚ := butter_milk / 4)
  (butter_revenue : ℚ := butter_gallons * 5)
  (sour_cream_revenue : ℚ := sour_cream_gallons * 6)
  (whole_milk_revenue : ℚ := whole_milk * 3)
  (total_revenue : ℚ := butter_revenue + sour_cream_revenue + whole_milk_revenue)
  (h : total_revenue = 41) :
  M = 16 :=
by
  sorry

end bill_milk_problem_l1808_180879


namespace percentage_owning_cats_percentage_owning_birds_l1808_180834

def total_students : ℕ := 500
def students_owning_cats : ℕ := 80
def students_owning_birds : ℕ := 120

theorem percentage_owning_cats : students_owning_cats * 100 / total_students = 16 := 
by 
  sorry

theorem percentage_owning_birds : students_owning_birds * 100 / total_students = 24 := 
by 
  sorry

end percentage_owning_cats_percentage_owning_birds_l1808_180834


namespace intersection_is_equilateral_triangle_l1808_180813

noncomputable def circle_eq (x y : ℝ) := x^2 + (y - 1)^2 = 1
noncomputable def ellipse_eq (x y : ℝ) := 9*x^2 + (y + 1)^2 = 9

theorem intersection_is_equilateral_triangle :
  ∀ A B C : ℝ × ℝ, circle_eq A.1 A.2 ∧ ellipse_eq A.1 A.2 ∧
                 circle_eq B.1 B.2 ∧ ellipse_eq B.1 B.2 ∧
                 circle_eq C.1 C.2 ∧ ellipse_eq C.1 C.2 → 
                 (dist A B = dist B C ∧ dist B C = dist C A) :=
by
  sorry

end intersection_is_equilateral_triangle_l1808_180813


namespace pelican_fish_count_l1808_180826

theorem pelican_fish_count 
(P K F : ℕ) 
(h1: K = P + 7) 
(h2: F = 3 * (P + K)) 
(h3: F = P + 86) : P = 13 := 
by 
  sorry

end pelican_fish_count_l1808_180826


namespace find_k_l1808_180865

-- Identifying conditions from the problem
def point (x : ℝ) : ℝ × ℝ := (x, x^3)  -- A point on the curve y = x^3
def tangent_slope (x : ℝ) : ℝ := 3 * x^2  -- The slope of the tangent to the curve y = x^3 at point (x, x^3)
def tangent_line (x k : ℝ) : ℝ := k * x + 2  -- The given tangent line equation

-- Question as a proof problem
theorem find_k (x : ℝ) (k : ℝ) (h : tangent_line x k = x^3) : k = 3 :=
by
  sorry

end find_k_l1808_180865


namespace polynomial_remainder_l1808_180828

def f (r : ℝ) : ℝ := r^15 - r + 3

theorem polynomial_remainder :
  f 2 = 32769 := by
  sorry

end polynomial_remainder_l1808_180828


namespace prime_sequence_constant_l1808_180863

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Condition: There exists a constant sequence of primes such that the given recurrence relation holds.
theorem prime_sequence_constant (p : ℕ) (k : ℤ) (n : ℕ) 
  (h1 : 1 ≤ n)
  (h2 : ∀ m ≥ 1, is_prime (p + m))
  (h3 : p + k = p + p + k) :
  ∀ m ≥ 1, p + m = p :=
sorry

end prime_sequence_constant_l1808_180863


namespace intersection_of_A_and_B_l1808_180848

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l1808_180848


namespace ratio_of_first_to_second_ball_l1808_180849

theorem ratio_of_first_to_second_ball 
  (x y z : ℕ) 
  (h1 : 3 * x = 27) 
  (h2 : y = 18) 
  (h3 : z = 3 * x) : 
  x / y = 1 / 2 := 
sorry

end ratio_of_first_to_second_ball_l1808_180849


namespace sandy_earnings_correct_l1808_180844

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end sandy_earnings_correct_l1808_180844


namespace largest_difference_rounding_l1808_180835

variable (A B : ℝ)
variable (estimate_A estimate_B : ℝ)
variable (within_A within_B : ℝ)
variable (diff : ℝ)

axiom est_A : estimate_A = 55000
axiom est_B : estimate_B = 58000
axiom cond_A : within_A = 0.15
axiom cond_B : within_B = 0.10

axiom bounds_A : 46750 ≤ A ∧ A ≤ 63250
axiom bounds_B : 52727 ≤ B ∧ B ≤ 64444

noncomputable def max_possible_difference : ℝ :=
  max (abs (B - A)) (abs (A - B))

theorem largest_difference_rounding :
  max_possible_difference A B = 18000 :=
by
  sorry

end largest_difference_rounding_l1808_180835


namespace minimum_sum_of_dimensions_l1808_180804

-- Define the problem as a Lean 4 statement
theorem minimum_sum_of_dimensions (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 2184) : 
  x + y + z = 36 := 
sorry

end minimum_sum_of_dimensions_l1808_180804


namespace binary_to_decimal_110101_l1808_180885

theorem binary_to_decimal_110101 :
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 = 53) :=
by
  sorry

end binary_to_decimal_110101_l1808_180885


namespace jessica_withdraw_fraq_l1808_180866

theorem jessica_withdraw_fraq {B : ℝ} (h : B - 200 + (1 / 2) * (B - 200) = 450) :
  (200 / B) = 2 / 5 := by
  sorry

end jessica_withdraw_fraq_l1808_180866


namespace h_h3_eq_3568_l1808_180838

def h (x : ℤ) := 3 * x ^ 2 + 3 * x - 2

theorem h_h3_eq_3568 : h (h 3) = 3568 := by
  sorry

end h_h3_eq_3568_l1808_180838


namespace solve_system_of_equations_l1808_180823

theorem solve_system_of_equations (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : 1 / (x * y) = x / z + 1)
  (h2 : 1 / (y * z) = y / x + 1)
  (h3 : 1 / (z * x) = z / y + 1) :
  x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2 :=
by
  sorry

end solve_system_of_equations_l1808_180823


namespace six_pow_2n_plus1_plus_1_div_by_7_l1808_180884

theorem six_pow_2n_plus1_plus_1_div_by_7 (n : ℕ) : (6^(2*n+1) + 1) % 7 = 0 := by
  sorry

end six_pow_2n_plus1_plus_1_div_by_7_l1808_180884


namespace unique_prime_n_l1808_180873

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_n (n : ℕ)
  (h1 : isPrime n)
  (h2 : isPrime (n^2 + 10))
  (h3 : isPrime (n^2 - 2))
  (h4 : isPrime (n^3 + 6))
  (h5 : isPrime (n^5 + 36)) : n = 7 :=
by
  sorry

end unique_prime_n_l1808_180873


namespace simplify_sin_formula_l1808_180858

theorem simplify_sin_formula : 2 * Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 2 := by
  -- Conditions and values used in the proof
  sorry

end simplify_sin_formula_l1808_180858


namespace water_consumed_l1808_180892

theorem water_consumed (traveler_water : ℕ) (camel_multiplier : ℕ) (ounces_in_gallon : ℕ) (total_water : ℕ)
  (h_traveler : traveler_water = 32)
  (h_camel : camel_multiplier = 7)
  (h_ounces_in_gallon : ounces_in_gallon = 128)
  (h_total : total_water = traveler_water + camel_multiplier * traveler_water) :
  total_water / ounces_in_gallon = 2 :=
by
  sorry

end water_consumed_l1808_180892


namespace smallest_n_divisibility_l1808_180809

theorem smallest_n_divisibility :
  ∃ n : ℕ, (n > 0) ∧ (∀ m : ℕ, (m > 0) ∧ (72 ∣ m^2) ∧ (1728 ∣ m^3) → (n ≤ m)) ∧
  (72 ∣ 12^2) ∧ (1728 ∣ 12^3) :=
by
  sorry

end smallest_n_divisibility_l1808_180809


namespace total_charge_for_3_6_miles_during_peak_hours_l1808_180861

-- Define the initial conditions as constants
def initial_fee : ℝ := 2.05
def charge_per_half_mile_first_2_miles : ℝ := 0.45
def charge_per_two_fifth_mile_after_2_miles : ℝ := 0.35
def peak_hour_surcharge : ℝ := 1.50

-- Define the function to calculate the total charge
noncomputable def total_charge (total_distance : ℝ) (is_peak_hour : Bool) : ℝ :=
  let first_2_miles_charge := if total_distance > 2 then 4 * charge_per_half_mile_first_2_miles else (total_distance / 0.5) * charge_per_half_mile_first_2_miles
  let remaining_distance := if total_distance > 2 then total_distance - 2 else 0
  let after_2_miles_charge := if total_distance > 2 then (remaining_distance / (2 / 5)) * charge_per_two_fifth_mile_after_2_miles else 0
  let surcharge := if is_peak_hour then peak_hour_surcharge else 0
  initial_fee + first_2_miles_charge + after_2_miles_charge + surcharge

-- Prove that total charge of 3.6 miles during peak hours is 6.75
theorem total_charge_for_3_6_miles_during_peak_hours : total_charge 3.6 true = 6.75 := by
  sorry

end total_charge_for_3_6_miles_during_peak_hours_l1808_180861


namespace total_students_in_class_is_15_l1808_180878

noncomputable def choose (n k : ℕ) : ℕ := sorry -- Define a function for combinations
noncomputable def permute (n k : ℕ) : ℕ := sorry -- Define a function for permutations

variables (x m n : ℕ) (hx : choose x 4 = m) (hn : permute x 2 = n) (hratio : m * 2 = n * 13)

theorem total_students_in_class_is_15 : x = 15 :=
sorry

end total_students_in_class_is_15_l1808_180878


namespace right_triangle_sides_l1808_180864

theorem right_triangle_sides (x y z : ℕ) (h1 : x + y + z = 30)
    (h2 : x^2 + y^2 + z^2 = 338) (h3 : x^2 + y^2 = z^2) :
    (x = 5 ∧ y = 12 ∧ z = 13) ∨ (x = 12 ∧ y = 5 ∧ z = 13) :=
by
  sorry

end right_triangle_sides_l1808_180864


namespace abs_diff_eq_implies_le_l1808_180852

theorem abs_diff_eq_implies_le {x y : ℝ} (h : |x - y| = y - x) : x ≤ y := 
by
  sorry

end abs_diff_eq_implies_le_l1808_180852


namespace money_problem_l1808_180840

variable (a b : ℝ)

theorem money_problem (h1 : 4 * a + b = 68) 
                      (h2 : 2 * a - b < 16) 
                      (h3 : a + b > 22) : 
                      a < 14 ∧ b > 12 := 
by 
  sorry

end money_problem_l1808_180840


namespace janet_dresses_pockets_l1808_180837

theorem janet_dresses_pockets :
  ∀ (x : ℕ), (∀ (dresses_with_pockets remaining_dresses total_pockets : ℕ),
  dresses_with_pockets = 24 / 2 →
  total_pockets = 32 →
  remaining_dresses = dresses_with_pockets - dresses_with_pockets / 3 →
  (dresses_with_pockets / 3) * x + remaining_dresses * 3 = total_pockets →
  x = 2) :=
by
  intros x dresses_with_pockets remaining_dresses total_pockets h1 h2 h3 h4
  sorry

end janet_dresses_pockets_l1808_180837


namespace parabola_focus_coordinates_l1808_180819

theorem parabola_focus_coordinates : 
  ∀ (x y : ℝ), x = 4 * y^2 → (∃ (y₀ : ℝ), (x, y₀) = (1/16, 0)) :=
by
  intro x y hxy
  sorry

end parabola_focus_coordinates_l1808_180819


namespace find_subtracted_value_l1808_180856

theorem find_subtracted_value (n x : ℕ) (h₁ : n = 36) (h₂ : ((n + 10) * 2 / 2 - x) = 44) : x = 2 :=
by
  sorry

end find_subtracted_value_l1808_180856


namespace ratio_of_female_democrats_l1808_180880

theorem ratio_of_female_democrats 
    (M F : ℕ) 
    (H1 : M + F = 990)
    (H2 : M / 4 + 165 = 330) 
    (H3 : 165 = 165) : 
    165 / F = 1 / 2 := 
sorry

end ratio_of_female_democrats_l1808_180880


namespace calc_expression_l1808_180871

theorem calc_expression :
  let a := 3^456
  let b := 9^5 / 9^3
  a - b = 3^456 - 81 :=
by
  let a := 3^456
  let b := 9^5 / 9^3
  sorry

end calc_expression_l1808_180871


namespace matt_worked_more_on_wednesday_l1808_180800

theorem matt_worked_more_on_wednesday :
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  minutes_wednesday - minutes_tuesday = 75 :=
by
  let minutes_monday := 450
  let minutes_tuesday := minutes_monday / 2
  let minutes_wednesday := 300
  show minutes_wednesday - minutes_tuesday = 75
  sorry

end matt_worked_more_on_wednesday_l1808_180800


namespace james_carrot_sticks_left_l1808_180806

variable (original_carrot_sticks : ℕ)
variable (eaten_before_dinner : ℕ)
variable (eaten_after_dinner : ℕ)
variable (given_away_during_dinner : ℕ)

theorem james_carrot_sticks_left 
  (h1 : original_carrot_sticks = 50)
  (h2 : eaten_before_dinner = 22)
  (h3 : eaten_after_dinner = 15)
  (h4 : given_away_during_dinner = 8) :
  original_carrot_sticks - eaten_before_dinner - eaten_after_dinner - given_away_during_dinner = 5 := 
sorry

end james_carrot_sticks_left_l1808_180806


namespace necessary_but_not_sufficient_l1808_180876

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a ≠ 0) → (ab ≠ 0) ↔ (a ≠ 0) :=
by sorry

end necessary_but_not_sufficient_l1808_180876


namespace build_wall_time_l1808_180843

theorem build_wall_time {d : ℝ} : 
  (15 * 1 + 3 * 2) * 3 = 63 ∧ 
  (25 * 1 + 5 * 2) * d = 63 → 
  d = 1.8 := 
by 
  sorry

end build_wall_time_l1808_180843


namespace find_natural_pairs_l1808_180807

theorem find_natural_pairs (a b : ℕ) :
  (∃ A, A * A = a ^ 2 + 3 * b) ∧ (∃ B, B * B = b ^ 2 + 3 * a) ↔ 
  (a = 1 ∧ b = 1) ∨ (a = 11 ∧ b = 11) ∨ (a = 16 ∧ b = 11) :=
by
  sorry

end find_natural_pairs_l1808_180807


namespace find_denomination_l1808_180867

def denomination_of_bills (num_tumblers : ℕ) (cost_per_tumbler change num_bills amount_paid bill_denomination : ℤ) : Prop :=
  num_tumblers * cost_per_tumbler + change = amount_paid ∧
  amount_paid = num_bills * bill_denomination

theorem find_denomination :
  denomination_of_bills
    10    -- num_tumblers
    45    -- cost_per_tumbler
    50    -- change
    5     -- num_bills
    500   -- amount_paid
    100   -- bill_denomination
:=
by
  sorry

end find_denomination_l1808_180867


namespace quadratic_roots_range_l1808_180860

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x^2 + (a^2 - 1) * x + a - 2 = 0 ∧ y^2 + (a^2 - 1) * y + a - 2 = 0 ∧ x ≠ y ∧ x > 1 ∧ y < 1) ↔ -2 < a ∧ a < 1 := 
sorry

end quadratic_roots_range_l1808_180860


namespace min_visible_sum_of_4x4x4_cube_l1808_180859

theorem min_visible_sum_of_4x4x4_cube (dice_capacity : ℕ) (opposite_sum : ℕ) (corner_dice edge_dice center_face_dice innermost_dice : ℕ) : 
  dice_capacity = 64 ∧ 
  opposite_sum = 7 ∧ 
  corner_dice = 8 ∧ 
  edge_dice = 24 ∧ 
  center_face_dice = 24 ∧ 
  innermost_dice = 8 → 
  ∃ min_sum, min_sum = 144 := by
  sorry

end min_visible_sum_of_4x4x4_cube_l1808_180859


namespace maximum_candies_purchase_l1808_180830

theorem maximum_candies_purchase (c1 : ℕ) (c4 : ℕ) (c7 : ℕ) (n : ℕ)
    (H_single : c1 = 1)
    (H_pack4  : c4 = 4)
    (H_cost4  : c4 = 3) 
    (H_pack7  : c7 = 7) 
    (H_cost7  : c7 = 4) 
    (H_budget : n = 10) :
    ∃ k : ℕ, k = 16 :=
by
    -- We'll skip the proof since the task requires only the statement
    sorry

end maximum_candies_purchase_l1808_180830


namespace determinant_inequality_solution_l1808_180816

theorem determinant_inequality_solution (a : ℝ) :
  (∀ x : ℝ, (x > -1 → x < (4 / a))) ↔ a = -4 := by
sorry

end determinant_inequality_solution_l1808_180816


namespace smallest_possible_value_of_d_l1808_180890

noncomputable def smallest_value_of_d : ℝ :=
  2 + Real.sqrt 2

theorem smallest_possible_value_of_d (c d : ℝ) (h1 : 2 < c) (h2 : c < d)
    (triangle_condition1 : ¬ (2 + c > d ∧ 2 + d > c ∧ c + d > 2))
    (triangle_condition2 : ¬ ( (2 / d) + (2 / c) > 2)) : d = smallest_value_of_d :=
  sorry

end smallest_possible_value_of_d_l1808_180890


namespace count_whole_numbers_in_interval_l1808_180820

theorem count_whole_numbers_in_interval : 
  let interval := Set.Ico (7 / 4 : ℝ) (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x ∈ interval, x ∈ Set.Icc 2 9 :=
by
  sorry

end count_whole_numbers_in_interval_l1808_180820


namespace teams_in_BIG_M_l1808_180821

theorem teams_in_BIG_M (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
sorry

end teams_in_BIG_M_l1808_180821


namespace intersection_P_Q_l1808_180814

open Set

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | abs x < 2}

theorem intersection_P_Q : P ∩ Q = {1} :=
by
  sorry

end intersection_P_Q_l1808_180814


namespace proportionality_problem_l1808_180883

noncomputable def find_x (z w : ℝ) (k : ℝ) : ℝ :=
  k / (z^(3/2) * w^2)

theorem proportionality_problem :
  ∃ k : ℝ, 
    (find_x 16 2 k = 5) ∧
    (find_x 64 4 k = 5 / 32) :=
by
  sorry

end proportionality_problem_l1808_180883


namespace solution_set_l1808_180855

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

variable {f : ℝ → ℝ}

-- Hypotheses
axiom odd_f : is_odd f
axiom increasing_f : is_increasing f
axiom f_of_neg_three : f (-3) = 0

-- Theorem statement
theorem solution_set (x : ℝ) : (x - 3) * f (x - 3) < 0 ↔ (0 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) :=
sorry

end solution_set_l1808_180855


namespace smallest_number_of_marbles_l1808_180874

theorem smallest_number_of_marbles :
  ∃ N : ℕ, N > 1 ∧ (N % 9 = 1) ∧ (N % 10 = 1) ∧ (N % 11 = 1) ∧ (∀ m : ℕ, m > 1 ∧ (m % 9 = 1) ∧ (m % 10 = 1) ∧ (m % 11 = 1) → N ≤ m) :=
sorry

end smallest_number_of_marbles_l1808_180874


namespace digit_at_1286th_position_l1808_180875

def naturally_written_sequence : ℕ → ℕ := sorry

theorem digit_at_1286th_position : naturally_written_sequence 1286 = 3 :=
sorry

end digit_at_1286th_position_l1808_180875


namespace intersection_M_N_l1808_180822

def M (x : ℝ) : Prop := (2 - x) / (x + 1) ≥ 0
def N (y : ℝ) : Prop := ∃ x : ℝ, y = Real.log x

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {y : ℝ | N y} = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end intersection_M_N_l1808_180822


namespace smallest_solution_x4_minus_50x2_plus_625_eq_0_l1808_180808

theorem smallest_solution_x4_minus_50x2_plus_625_eq_0 : ∃ x : ℝ, x^4 - 50 * x^2 + 625 = 0 ∧ ∀ y : ℝ, y^4 - 50 * y^2 + 625 = 0 → x ≤ y := 
sorry

end smallest_solution_x4_minus_50x2_plus_625_eq_0_l1808_180808


namespace pradeep_max_marks_l1808_180851

-- conditions
variables (M : ℝ)
variable (h1 : 0.40 * M = 220)

-- question and answer
theorem pradeep_max_marks : M = 550 :=
by
  sorry

end pradeep_max_marks_l1808_180851


namespace arithmetic_evaluation_l1808_180801

theorem arithmetic_evaluation : (64 / 0.08) - 2.5 = 797.5 :=
by
  sorry

end arithmetic_evaluation_l1808_180801


namespace system_of_equations_solve_l1808_180891

theorem system_of_equations_solve (x y : ℝ) 
  (h1 : 2 * x + y = 5)
  (h2 : x + 2 * y = 4) :
  x + y = 3 :=
by
  sorry

end system_of_equations_solve_l1808_180891


namespace taxi_ride_cost_is_five_dollars_l1808_180827

def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def miles_traveled : ℝ := 10.0
def total_cost : ℝ := base_fare + (cost_per_mile * miles_traveled)

theorem taxi_ride_cost_is_five_dollars : total_cost = 5.00 :=
by
  -- proof omitted
  sorry

end taxi_ride_cost_is_five_dollars_l1808_180827


namespace total_games_in_season_l1808_180810

theorem total_games_in_season :
  let num_teams := 100
  let num_sub_leagues := 5
  let teams_per_league := 20
  let games_per_pair := 6
  let teams_advancing := 4
  let playoff_teams := num_sub_leagues * teams_advancing
  let sub_league_games := (teams_per_league * (teams_per_league - 1) / 2) * games_per_pair
  let total_sub_league_games := sub_league_games * num_sub_leagues
  let playoff_games := (playoff_teams * (playoff_teams - 1)) / 2 
  let total_games := total_sub_league_games + playoff_games
  total_games = 5890 :=
by
  sorry

end total_games_in_season_l1808_180810


namespace ball_box_distribution_l1808_180899

theorem ball_box_distribution : (3^5 = 243) :=
by
  sorry

end ball_box_distribution_l1808_180899


namespace number_writing_number_reading_l1808_180836

def ten_million_place := 10^7
def hundred_thousand_place := 10^5
def ten_place := 10

def ten_million := 1 * ten_million_place
def three_hundred_thousand := 3 * hundred_thousand_place
def fifty := 5 * ten_place

def constructed_number := ten_million + three_hundred_thousand + fifty

def read_number := "ten million and thirty thousand and fifty"

theorem number_writing : constructed_number = 10300050 := by
  -- Sketch of proof goes here based on place values
  sorry

theorem number_reading : read_number = "ten million and thirty thousand and fifty" := by
  -- Sketch of proof goes here for the reading method
  sorry

end number_writing_number_reading_l1808_180836


namespace cars_more_than_trucks_l1808_180803

theorem cars_more_than_trucks (total_vehicles : ℕ) (trucks : ℕ) (h : total_vehicles = 69) (h' : trucks = 21) :
  (total_vehicles - trucks) - trucks = 27 :=
by
  sorry

end cars_more_than_trucks_l1808_180803


namespace courier_total_travel_times_l1808_180847

-- Define the conditions
variables (v1 v2 : ℝ) (t : ℝ)
axiom speed_condition_1 : v1 * (t + 16) = (v1 + v2) * t
axiom speed_condition_2 : v2 * (t + 9) = (v1 + v2) * t
axiom time_condition : t = 12

-- Define the total travel times
def total_travel_time_1 : ℝ := t + 16
def total_travel_time_2 : ℝ := t + 9

-- Proof problem statement
theorem courier_total_travel_times :
  total_travel_time_1 = 28 ∧ total_travel_time_2 = 21 :=
by
  sorry

end courier_total_travel_times_l1808_180847


namespace intersection_conditions_l1808_180831

-- Define the conditions
variables (c : ℝ) (k : ℝ) (m : ℝ) (n : ℝ) (p : ℝ)

-- Distance condition
def distance_condition (k : ℝ) (m : ℝ) (n : ℝ) (c : ℝ) : Prop :=
  (abs ((k^2 + 8 * k + c) - (m * k + n)) = 4)

-- Line passing through point (2, 7)
def passes_through_point (m : ℝ) (n : ℝ) : Prop :=
  (7 = 2 * m + n)

-- Definition of discriminants
def discriminant_1 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n - 4))

def discriminant_2 (m : ℝ) (c : ℝ) (n : ℝ) : ℝ :=
  ((8 - m)^2 - 4 * (c - n + 4))

-- Statement of the problem
theorem intersection_conditions (h₁ : n ≠ 0)
  (h₂ : passes_through_point m n)
  (h₃ : distance_condition k m n c)
  (h₄ : (discriminant_1 m c n = 0 ∨ discriminant_1 m c n < 0))
  (h₅ : (discriminant_2 m c n < 0)) :
  ∃ m n, n = 7 - 2 * m ∧ distance_condition k m n c :=
sorry

end intersection_conditions_l1808_180831


namespace clara_meeting_time_l1808_180802

theorem clara_meeting_time (d T : ℝ) :
  (d / 20 = T - 0.5) →
  (d / 12 = T + 0.5) →
  (d / T = 15) :=
by
  intros h1 h2
  sorry

end clara_meeting_time_l1808_180802


namespace ratio_as_percentage_l1808_180862

theorem ratio_as_percentage (x : ℝ) (h : (x / 2) / (3 * x / 5) = 3 / 5) : 
  (3 / 5) * 100 = 60 := 
sorry

end ratio_as_percentage_l1808_180862


namespace remainder_div_19_l1808_180842

theorem remainder_div_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
sorry

end remainder_div_19_l1808_180842


namespace first_day_is_sunday_l1808_180872

noncomputable def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem first_day_is_sunday :
  (day_of_week 18 = "Wednesday") → (day_of_week 1 = "Sunday") :=
by
  intro h
  -- proof would go here
  sorry

end first_day_is_sunday_l1808_180872


namespace fraction_order_l1808_180870

theorem fraction_order:
  let frac1 := (21 : ℚ) / 17
  let frac2 := (22 : ℚ) / 19
  let frac3 := (18 : ℚ) / 15
  let frac4 := (20 : ℚ) / 16
  frac2 < frac3 ∧ frac3 < frac1 ∧ frac1 < frac4 := 
sorry

end fraction_order_l1808_180870


namespace gcd_of_3150_and_9800_is_350_l1808_180868

-- Definition of the two numbers
def num1 : ℕ := 3150
def num2 : ℕ := 9800

-- The greatest common factor of num1 and num2 is 350
theorem gcd_of_3150_and_9800_is_350 : Nat.gcd num1 num2 = 350 := by
  sorry

end gcd_of_3150_and_9800_is_350_l1808_180868


namespace total_soccer_balls_l1808_180824

theorem total_soccer_balls (boxes : ℕ) (packages_per_box : ℕ) (balls_per_package : ℕ) 
  (h1 : boxes = 10) (h2 : packages_per_box = 8) (h3 : balls_per_package = 13) : 
  (boxes * packages_per_box * balls_per_package = 1040) :=
by 
  sorry

end total_soccer_balls_l1808_180824


namespace inequality_minus_x_plus_3_l1808_180811

variable (x y : ℝ)

theorem inequality_minus_x_plus_3 (h : x < y) : -x + 3 > -y + 3 :=
by {
  sorry
}

end inequality_minus_x_plus_3_l1808_180811


namespace luisa_trip_l1808_180881

noncomputable def additional_miles (d1: ℝ) (s1: ℝ) (s2: ℝ) (desired_avg_speed: ℝ) : ℝ := 
  let t1 := d1 / s1
  let t := (d1 * (desired_avg_speed - s1)) / (s2 * (s1 - desired_avg_speed))
  s2 * t

theorem luisa_trip :
  additional_miles 18 36 60 45 = 18 :=
by
  sorry

end luisa_trip_l1808_180881


namespace apple_percentage_is_23_l1808_180832

def total_responses := 70 + 80 + 50 + 30 + 70
def apple_responses := 70

theorem apple_percentage_is_23 :
  (apple_responses : ℝ) / (total_responses : ℝ) * 100 = 23 := 
by
  sorry

end apple_percentage_is_23_l1808_180832


namespace rose_paid_after_discount_l1808_180896

noncomputable def discount_percentage : ℝ := 0.1
noncomputable def original_price : ℝ := 10
noncomputable def discount_amount := discount_percentage * original_price
noncomputable def final_price := original_price - discount_amount

theorem rose_paid_after_discount : final_price = 9 := by
  sorry

end rose_paid_after_discount_l1808_180896


namespace correct_calculation_D_l1808_180853

theorem correct_calculation_D (m : ℕ) : 
  (2 * m ^ 3) * (3 * m ^ 2) = 6 * m ^ 5 :=
by
  sorry

end correct_calculation_D_l1808_180853


namespace remaining_dresses_pockets_count_l1808_180886

-- Definitions translating each condition in the problem.
def total_dresses : Nat := 24
def dresses_with_pockets : Nat := total_dresses / 2
def dresses_with_two_pockets : Nat := dresses_with_pockets / 3
def total_pockets : Nat := 32

-- Question translated into a proof problem using Lean's logic.
theorem remaining_dresses_pockets_count :
  (total_pockets - (dresses_with_two_pockets * 2)) / (dresses_with_pockets - dresses_with_two_pockets) = 3 := by
  sorry

end remaining_dresses_pockets_count_l1808_180886


namespace range_of_a_l1808_180818

-- Definitions derived from conditions
def is_ellipse_with_foci_on_x_axis (a : ℝ) : Prop := a^2 > a + 6 ∧ a + 6 > 0

-- Theorem representing the proof problem
theorem range_of_a (a : ℝ) (h : is_ellipse_with_foci_on_x_axis a) :
  (a > 3) ∨ (-6 < a ∧ a < -2) :=
sorry

end range_of_a_l1808_180818


namespace total_cost_l1808_180829

-- Definitions based on the problem's conditions
def cost_hamburger : ℕ := 4
def cost_milkshake : ℕ := 3

def qty_hamburgers : ℕ := 7
def qty_milkshakes : ℕ := 6

-- The proof statement
theorem total_cost :
  (qty_hamburgers * cost_hamburger + qty_milkshakes * cost_milkshake) = 46 :=
by
  sorry

end total_cost_l1808_180829


namespace triangle_count_l1808_180839

theorem triangle_count (a b c : ℕ) (hb : b = 2008) (hab : a ≤ b) (hbc : b ≤ c) (ht : a + b > c) : 
  ∃ n, n = 2017036 :=
by
  sorry

end triangle_count_l1808_180839


namespace evaluate_polynomial_l1808_180841

theorem evaluate_polynomial (x : ℤ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end evaluate_polynomial_l1808_180841


namespace larger_number_l1808_180895

theorem larger_number (A B : ℝ) (h1 : A - B = 1650) (h2 : 0.075 * A = 0.125 * B) : A = 4125 :=
sorry

end larger_number_l1808_180895


namespace Marcy_spears_l1808_180857

def makeSpears (saplings: ℕ) (logs: ℕ) (branches: ℕ) (trunks: ℕ) : ℕ :=
  3 * saplings + 9 * logs + 7 * branches + 15 * trunks

theorem Marcy_spears :
  makeSpears 12 1 6 0 - (3 * 2) + makeSpears 0 4 0 0 - (9 * 4) + makeSpears 0 0 6 1 - (7 * 0) + makeSpears 0 0 0 2 = 81 := by
  sorry

end Marcy_spears_l1808_180857


namespace vectors_not_coplanar_l1808_180888

def a : ℝ × ℝ × ℝ := (4, 1, 1)
def b : ℝ × ℝ × ℝ := (-9, -4, -9)
def c : ℝ × ℝ × ℝ := (6, 2, 6)

def scalarTripleProduct (u v w : ℝ × ℝ × ℝ) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

theorem vectors_not_coplanar : scalarTripleProduct a b c = -18 := by
  sorry

end vectors_not_coplanar_l1808_180888


namespace distance_walked_north_l1808_180882

-- Definition of the problem parameters
def distance_west : ℝ := 10
def total_distance : ℝ := 14.142135623730951

-- The theorem stating the result
theorem distance_walked_north (x : ℝ) (h : distance_west ^ 2 + x ^ 2 = total_distance ^ 2) : x = 10 :=
by sorry

end distance_walked_north_l1808_180882


namespace not_divisible_1978_1000_l1808_180887

theorem not_divisible_1978_1000 (m : ℕ) : ¬ ∃ m : ℕ, (1000^m - 1) ∣ (1978^m - 1) := sorry

end not_divisible_1978_1000_l1808_180887


namespace probability_X_equals_Y_l1808_180833

noncomputable def prob_X_equals_Y : ℚ :=
  let count_intersections : ℚ := 15
  let total_possibilities : ℚ := 15 * 15
  count_intersections / total_possibilities

theorem probability_X_equals_Y :
  (∀ (x y : ℝ), -15 * Real.pi ≤ x ∧ x ≤ 15 * Real.pi ∧ -15 * Real.pi ≤ y ∧ y ≤ 15 * Real.pi →
    (Real.cos (Real.cos x) = Real.cos (Real.cos y)) →
    prob_X_equals_Y = 1/15) :=
sorry

end probability_X_equals_Y_l1808_180833


namespace length_of_AB_l1808_180893

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem length_of_AB :
  let O := (0, 0)
  let A := (54^(1/3), 0)
  let B := (0, 54^(1/3))
  distance A B = 54^(1/3) * Real.sqrt 2 :=
by
  sorry

end length_of_AB_l1808_180893


namespace area_of_right_isosceles_triangle_l1808_180845

def is_right_isosceles (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

theorem area_of_right_isosceles_triangle (a b c : ℝ) (h : is_right_isosceles a b c) (h_hypotenuse : c = 10) :
  1/2 * a * b = 25 :=
by
  sorry

end area_of_right_isosceles_triangle_l1808_180845


namespace necklace_stand_capacity_l1808_180825

def necklace_stand_initial := 5
def ring_display_capacity := 30
def ring_display_current := 18
def bracelet_display_capacity := 15
def bracelet_display_current := 8
def cost_per_necklace := 4
def cost_per_ring := 10
def cost_per_bracelet := 5
def total_cost := 183

theorem necklace_stand_capacity : necklace_stand_current + (total_cost - (ring_display_capacity - ring_display_current) * cost_per_ring - (bracelet_display_capacity - bracelet_display_current) * cost_per_bracelet) / cost_per_necklace = 12 :=
by
  sorry

end necklace_stand_capacity_l1808_180825


namespace find_a_plus_c_l1808_180889

noncomputable def triangle_ABC (A B C a b c : ℝ) : Prop :=
  (b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B) ∧
  (b = 2) ∧
  ((1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 2) / 2)

theorem find_a_plus_c {A B C a b c : ℝ} (h : triangle_ABC A B C a b c) :
  a + c = 4 :=
by
  rcases h with ⟨hc1, hc2, hc3⟩
  sorry

end find_a_plus_c_l1808_180889


namespace notebook_width_l1808_180869

theorem notebook_width
  (circumference : ℕ)
  (length : ℕ)
  (width : ℕ)
  (H1 : circumference = 46)
  (H2 : length = 9)
  (H3 : circumference = 2 * (length + width)) :
  width = 14 :=
by
  sorry -- proof is omitted

end notebook_width_l1808_180869


namespace jason_less_than_jenny_l1808_180846

-- Definition of conditions

def grade_Jenny : ℕ := 95
def grade_Bob : ℕ := 35
def grade_Jason : ℕ := 2 * grade_Bob -- Bob's grade is half of Jason's grade

-- The theorem we need to prove
theorem jason_less_than_jenny : grade_Jenny - grade_Jason = 25 :=
by
  sorry

end jason_less_than_jenny_l1808_180846


namespace ellipse_parabola_common_point_l1808_180877

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ x y : ℝ, x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔  -1 ≤ a ∧ a ≤ 17 / 8 :=
by
  sorry

end ellipse_parabola_common_point_l1808_180877


namespace probability_of_B_l1808_180817

variables (A B : Prop)
variables (P : Prop → ℝ) -- Probability Measure

axiom A_and_B : P (A ∧ B) = 0.15
axiom not_A_and_not_B : P (¬A ∧ ¬B) = 0.6

theorem probability_of_B : P B = 0.15 :=
by
  sorry

end probability_of_B_l1808_180817


namespace julio_lost_15_fish_l1808_180897

def fish_caught_per_hour : ℕ := 7
def hours_fished : ℕ := 9
def fish_total_without_loss : ℕ := fish_caught_per_hour * hours_fished
def fish_total_actual : ℕ := 48
def fish_lost : ℕ := fish_total_without_loss - fish_total_actual

theorem julio_lost_15_fish : fish_lost = 15 := by
  sorry

end julio_lost_15_fish_l1808_180897


namespace find_common_difference_l1808_180854

variable {a : ℕ → ℤ} 
variable {S : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def problem_conditions (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 4 = 8 ∧ S 8 = 48

theorem find_common_difference :
  ∃ d, problem_conditions a S d ∧ is_arithmetic_sequence a d ∧ sum_of_first_n_terms a S ∧ d = 2 :=
by
  sorry

end find_common_difference_l1808_180854


namespace correct_quotient_divide_8_l1808_180805

theorem correct_quotient_divide_8 (N : ℕ) (Q : ℕ) 
  (h1 : N = 7 * 12 + 5) 
  (h2 : N / 8 = Q) : 
  Q = 11 := 
by
  sorry

end correct_quotient_divide_8_l1808_180805


namespace estimate_fish_population_l1808_180815

theorem estimate_fish_population (n m k : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) (h4 : k ≤ m) : 
  ∃ N : ℕ, N = m * n / k :=
by
  sorry

end estimate_fish_population_l1808_180815


namespace fraction_zero_implies_x_eq_neg3_l1808_180850

theorem fraction_zero_implies_x_eq_neg3 (x : ℝ) (h1 : x ≠ 3) (h2 : (x^2 - 9) / (x - 3) = 0) : x = -3 :=
sorry

end fraction_zero_implies_x_eq_neg3_l1808_180850
