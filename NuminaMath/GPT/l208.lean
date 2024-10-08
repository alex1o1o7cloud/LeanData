import Mathlib

namespace find_k_value_l208_208655

theorem find_k_value (S : ℕ → ℕ) (a : ℕ → ℕ) (k : ℤ) 
  (hS : ∀ n, S n = 5 * n^2 + k * n)
  (ha2 : a 2 = 18) :
  k = 3 := 
sorry

end find_k_value_l208_208655


namespace find_max_value_l208_208823

-- We define the conditions as Lean definitions and hypotheses
def is_distinct_digits (A B C D E F : ℕ) : Prop :=
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧
  (D ≠ E) ∧ (D ≠ F) ∧
  (E ≠ F)

def all_digits_in_range (A B C D E F : ℕ) : Prop :=
  (1 ≤ A) ∧ (A ≤ 8) ∧
  (1 ≤ B) ∧ (B ≤ 8) ∧
  (1 ≤ C) ∧ (C ≤ 8) ∧
  (1 ≤ D) ∧ (D ≤ 8) ∧
  (1 ≤ E) ∧ (E ≤ 8) ∧
  (1 ≤ F) ∧ (F ≤ 8)

def divisible_by_99 (n : ℕ) : Prop :=
  (n % 99 = 0)

theorem find_max_value (A B C D E F : ℕ) :
  is_distinct_digits A B C D E F →
  all_digits_in_range A B C D E F →
  divisible_by_99 (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F) →
  100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F = 87653412 :=
sorry

end find_max_value_l208_208823


namespace fescue_in_Y_l208_208813

-- Define the weight proportions of the mixtures
def weight_X : ℝ := 0.6667
def weight_Y : ℝ := 0.3333

-- Define the proportion of ryegrass in each mixture
def ryegrass_X : ℝ := 0.40
def ryegrass_Y : ℝ := 0.25

-- Define the proportion of ryegrass in the final mixture
def ryegrass_final : ℝ := 0.35

-- Define the proportion of ryegrass contributed by X and Y to the final mixture
def contrib_X : ℝ := weight_X * ryegrass_X
def contrib_Y : ℝ := weight_Y * ryegrass_Y

-- Define the total proportion of ryegrass in the final mixture
def total_ryegrass : ℝ := contrib_X + contrib_Y

-- The lean theorem stating that the percentage of fescue in Y equals 75%
theorem fescue_in_Y :
  total_ryegrass = ryegrass_final →
  (100 - (ryegrass_Y * 100)) = 75 := 
by
  intros h
  sorry

end fescue_in_Y_l208_208813


namespace solve_trig_equation_l208_208994
open Real

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (1 / 2) * abs (cos (2 * x) + (1 / 2)) = (sin (3 * x))^2 - (sin x) * (sin (3 * x))

-- Define the correct solution set 
def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (π / 6) + (k * (π / 2)) ∨ x = -(π / 6) + (k * (π / 2))

-- The theorem we need to prove
theorem solve_trig_equation : ∀ x : ℝ, original_equation x ↔ solution_set x :=
by sorry

end solve_trig_equation_l208_208994


namespace no_integers_abc_for_polynomial_divisible_by_9_l208_208897

theorem no_integers_abc_for_polynomial_divisible_by_9 :
  ¬ ∃ (a b c : ℤ), ∀ x : ℤ, 9 ∣ (x + a) * (x + b) * (x + c) - x ^ 3 - 1 :=
by
  sorry

end no_integers_abc_for_polynomial_divisible_by_9_l208_208897


namespace concert_ticket_revenue_l208_208649

theorem concert_ticket_revenue :
  let price_student : ℕ := 9
  let price_non_student : ℕ := 11
  let total_tickets : ℕ := 2000
  let student_tickets : ℕ := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  revenue_student + revenue_non_student = 20960 :=
by
  -- Definitions
  let price_student := 9
  let price_non_student := 11
  let total_tickets := 2000
  let student_tickets := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  -- Proof
  sorry  -- Placeholder for the proof

end concert_ticket_revenue_l208_208649


namespace geometric_sequence_problem_l208_208601

theorem geometric_sequence_problem
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h5 : a 5 * a 6 = 3)
  (h9 : a 9 * a 10 = 9) :
  a 7 * a 8 = 3 * Real.sqrt 3 :=
by
  sorry

end geometric_sequence_problem_l208_208601


namespace sum_of_squares_of_roots_l208_208543

theorem sum_of_squares_of_roots :
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂) →
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = 79 / 25) :=
by
  sorry

end sum_of_squares_of_roots_l208_208543


namespace average_distance_is_600_l208_208361

-- Definitions based on the given conditions
def distance_around_block := 200
def johnny_rounds := 4
def mickey_rounds := johnny_rounds / 2

-- The calculated distances
def johnny_distance := johnny_rounds * distance_around_block
def mickey_distance := mickey_rounds * distance_around_block

-- The average distance computation
def average_distance := (johnny_distance + mickey_distance) / 2

-- The theorem to prove that the average distance is 600 meters
theorem average_distance_is_600 : average_distance = 600 := by sorry

end average_distance_is_600_l208_208361


namespace solution_n_value_l208_208540

open BigOperators

noncomputable def problem_statement (a b n : ℝ) : Prop :=
  ∃ (A B : ℝ), A = Real.log a ∧ B = Real.log b ∧
    (7 * A + 15 * B) - (4 * A + 9 * B) = (11 * A + 20 * B) - (7 * A + 15 * B) ∧
    (4 + 135) * B = Real.log (b^n)

theorem solution_n_value (a b : ℝ) (h_pos : a > 0) (h_pos_b : b > 0) :
  problem_statement a b 139 :=
by
  sorry

end solution_n_value_l208_208540


namespace intersection_points_on_ellipse_l208_208319

theorem intersection_points_on_ellipse (s x y : ℝ)
  (h_line1 : s * x - 3 * y - 4 * s = 0)
  (h_line2 : x - 3 * s * y + 4 = 0) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 :=
by
  sorry

end intersection_points_on_ellipse_l208_208319


namespace intersection_of_sets_l208_208324

variable {x : ℝ}

def SetA : Set ℝ := {x | x + 1 > 0}
def SetB : Set ℝ := {x | x - 3 < 0}

theorem intersection_of_sets : SetA ∩ SetB = {x | -1 < x ∧ x < 3} :=
by sorry

end intersection_of_sets_l208_208324


namespace appropriate_sampling_methods_l208_208614
-- Import the entire Mathlib library for broader functionality

-- Define the conditions
def community_high_income_families : ℕ := 125
def community_middle_income_families : ℕ := 280
def community_low_income_families : ℕ := 95
def community_total_households : ℕ := community_high_income_families + community_middle_income_families + community_low_income_families

def student_count : ℕ := 12

-- Define the theorem to be proven
theorem appropriate_sampling_methods :
  (community_total_households = 500 → stratified_sampling) ∧
  (student_count = 12 → random_sampling) :=
by sorry

end appropriate_sampling_methods_l208_208614


namespace tom_overall_profit_l208_208610

def initial_purchase_cost : ℝ := 20 * 3 + 30 * 5 + 15 * 10
def purchase_commission : ℝ := 0.02 * initial_purchase_cost
def total_initial_cost : ℝ := initial_purchase_cost + purchase_commission

def sale_revenue_before_commission : ℝ := 10 * 4 + 20 * 7 + 5 * 12
def sales_commission : ℝ := 0.02 * sale_revenue_before_commission
def total_sales_revenue : ℝ := sale_revenue_before_commission - sales_commission

def remaining_stock_a_value : ℝ := 10 * (3 * 2)
def remaining_stock_b_value : ℝ := 10 * (5 * 1.20)
def remaining_stock_c_value : ℝ := 10 * (10 * 0.90)
def total_remaining_value : ℝ := remaining_stock_a_value + remaining_stock_b_value + remaining_stock_c_value

def overall_profit_or_loss : ℝ := total_sales_revenue + total_remaining_value - total_initial_cost

theorem tom_overall_profit : overall_profit_or_loss = 78 := by
  sorry

end tom_overall_profit_l208_208610


namespace value_of_expression_l208_208776

theorem value_of_expression : 1 + 3^2 = 10 :=
by
  sorry

end value_of_expression_l208_208776


namespace bicycle_speed_l208_208107

theorem bicycle_speed
  (dist : ℝ := 15) -- Distance between the school and the museum
  (bus_factor : ℝ := 1.5) -- Bus speed is 1.5 times the bicycle speed
  (time_diff : ℝ := 1 / 4) -- Bicycle students leave 1/4 hour earlier
  (x : ℝ) -- Speed of bicycles
  (h : (dist / x) - (dist / (bus_factor * x)) = time_diff) :
  x = 20 :=
sorry

end bicycle_speed_l208_208107


namespace cost_per_person_trip_trips_rental_cost_l208_208722

-- Define the initial conditions
def ticket_price_per_person := 60
def total_employees := 70
def small_car_seats := 4
def large_car_seats := 11
def extra_cost_small_car_per_person := 5
def extra_revenue_large_car := 50
def max_total_cost := 5000

-- Define the costs per person per trip for small and large cars
def large_car_cost_per_person := 10
def small_car_cost_per_person := large_car_cost_per_person + extra_cost_small_car_per_person

-- Define the number of trips for four-seater and eleven-seater cars
def four_seater_trips := 1
def eleven_seater_trips := 6

-- Prove the lean statements
theorem cost_per_person_trip : 
  (11 * large_car_cost_per_person) - (small_car_seats * small_car_cost_per_person) = extra_revenue_large_car := 
sorry

theorem trips_rental_cost (x y : ℕ) : 
  (small_car_seats * x + large_car_seats * y = total_employees) ∧
  ((total_employees * ticket_price_per_person) + (small_car_cost_per_person * small_car_seats * x) + (large_car_cost_per_person * large_car_seats * y) ≤ max_total_cost) :=
sorry

end cost_per_person_trip_trips_rental_cost_l208_208722


namespace length_of_QR_of_triangle_l208_208468

def length_of_QR (PQ PR PM : ℝ) : ℝ := sorry

theorem length_of_QR_of_triangle (PQ PR : ℝ) (PM : ℝ) (hPQ : PQ = 4) (hPR : PR = 7) (hPM : PM = 7 / 2) : length_of_QR PQ PR PM = 9 := by
  sorry

end length_of_QR_of_triangle_l208_208468


namespace find_t_l208_208854

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def tangent_point (t : ℝ) : ℝ × ℝ := (t, 0)

theorem find_t :
  (∀ (A : ℝ × ℝ), ellipse_eq A.1 A.2 → 
    ∃ (C : ℝ × ℝ),
      tangent_point 2 = C ∧
      -- C is tangent to the extended line of F1A
      -- C is tangent to the extended line of F1F2
      -- C is tangent to segment AF2
      true
  ) :=
sorry

end find_t_l208_208854


namespace find_a_l208_208941

def F (a b c : ℝ) : ℝ := a * b^3 + c

theorem find_a (a : ℝ) (h : F a 3 8 = F a 5 12) : a = -2 / 49 := by
  sorry

end find_a_l208_208941


namespace third_smallest_number_l208_208065

/-- 
  The third smallest two-decimal-digit number that can be made
  using the digits 3, 8, 2, and 7 each exactly once is 27.38.
-/
theorem third_smallest_number (digits : List ℕ) (h : digits = [3, 8, 2, 7]) : 
  ∃ x y, 
  x < y ∧
  x = 23.78 ∧
  y = 23.87 ∧
  ∀ z, z > x ∧ z < y → z = 27.38 :=
by 
  sorry

end third_smallest_number_l208_208065


namespace Aunt_Zhang_expenditure_is_negative_l208_208760

-- Define variables for the problem
def income_yuan : ℤ := 5
def expenditure_yuan : ℤ := 3

-- The theorem stating Aunt Zhang's expenditure in financial terms
theorem Aunt_Zhang_expenditure_is_negative :
  (- expenditure_yuan) = -3 :=
by
  sorry

end Aunt_Zhang_expenditure_is_negative_l208_208760


namespace not_all_perfect_squares_l208_208619

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem not_all_perfect_squares (x : ℕ) (hx : x > 0) :
  ¬ (is_perfect_square (2 * x - 1) ∧ is_perfect_square (5 * x - 1) ∧ is_perfect_square (13 * x - 1)) :=
by
  sorry

end not_all_perfect_squares_l208_208619


namespace initial_printing_presses_l208_208780

theorem initial_printing_presses (P : ℕ) 
  (h1 : 500000 / (9 * P) = 500000 / (12 * 30)) : 
  P = 40 :=
by
  sorry

end initial_printing_presses_l208_208780


namespace find_other_root_l208_208548

-- Definitions based on conditions
def quadratic_equation (k : ℝ) (x : ℝ) : Prop := x^2 + 2 * k * x + k - 1 = 0

def is_root (k : ℝ) (x : ℝ) : Prop := quadratic_equation k x = true

-- The theorem to prove
theorem find_other_root (k x t: ℝ) (h₁ : is_root k 0) : t = -2 :=
sorry

end find_other_root_l208_208548


namespace integer_solutions_for_xyz_eq_4_l208_208804

theorem integer_solutions_for_xyz_eq_4 :
  {n : ℕ // n = 48} :=
sorry

end integer_solutions_for_xyz_eq_4_l208_208804


namespace must_hold_inequality_l208_208784

variable (f : ℝ → ℝ)

noncomputable def condition : Prop := ∀ x > 0, x * (deriv^[2] f) x < 1

theorem must_hold_inequality (h : condition f) : f (Real.exp 1) < f 1 + 1 := 
sorry

end must_hold_inequality_l208_208784


namespace sheep_daddy_input_l208_208359

-- Conditions for black box transformations
def black_box (k : ℕ) : ℕ :=
  if k % 2 = 1 then 4 * k + 1 else k / 2

-- The transformation chain with three black boxes
def black_box_chain (k : ℕ) : ℕ :=
  black_box (black_box (black_box k))

-- Theorem statement capturing the problem:
-- Final output m is 2, and the largest input leading to this is 64.
theorem sheep_daddy_input : ∃ k : ℕ, ∀ (k1 k2 k3 k4 : ℕ), 
  black_box_chain k1 = 2 ∧ 
  black_box_chain k2 = 2 ∧ 
  black_box_chain k3 = 2 ∧ 
  black_box_chain k4 = 2 ∧ 
  k1 ≠ k2 ∧ k2 ≠ k3 ∧ k3 ≠ k4 ∧ k4 ≠ k1 ∧ 
  k = max k1 (max k2 (max k3 k4)) → k = 64 :=
sorry  -- Proof is not required

end sheep_daddy_input_l208_208359


namespace bob_ears_left_l208_208049

namespace CornProblem

-- Definitions of the given conditions
def initial_bob_bushels : ℕ := 120
def ears_per_bushel : ℕ := 15

def given_away_bushels_terry : ℕ := 15
def given_away_bushels_jerry : ℕ := 8
def given_away_bushels_linda : ℕ := 25
def given_away_ears_stacy : ℕ := 42
def given_away_bushels_susan : ℕ := 9
def given_away_bushels_tim : ℕ := 4
def given_away_ears_tim : ℕ := 18

-- Calculate initial ears of corn
noncomputable def initial_ears_of_corn : ℕ := initial_bob_bushels * ears_per_bushel

-- Calculate total ears given away in bushels
def total_ears_given_away_bushels : ℕ :=
  (given_away_bushels_terry + given_away_bushels_jerry + given_away_bushels_linda +
   given_away_bushels_susan + given_away_bushels_tim) * ears_per_bushel

-- Calculate total ears directly given away
def total_ears_given_away_direct : ℕ :=
  given_away_ears_stacy + given_away_ears_tim

-- Calculate total ears given away
def total_ears_given_away : ℕ :=
  total_ears_given_away_bushels + total_ears_given_away_direct

-- Calculate ears of corn Bob has left
noncomputable def ears_left : ℕ :=
  initial_ears_of_corn - total_ears_given_away

-- The proof statement
theorem bob_ears_left : ears_left = 825 := by
  sorry

end CornProblem

end bob_ears_left_l208_208049


namespace proof_problem_l208_208766

variables {x y z w : ℝ}

-- Condition given in the problem
def condition (x y z w : ℝ) : Prop :=
  (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3

-- The statement to be proven
theorem proof_problem (h : condition x y z w) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = 1 :=
by
  sorry

end proof_problem_l208_208766


namespace y_in_terms_of_x_l208_208746

theorem y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 5) : y = -2 * x + 5 :=
sorry

end y_in_terms_of_x_l208_208746


namespace prism_unique_triple_l208_208694

theorem prism_unique_triple :
  ∃! (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ b = 2000 ∧
                  (∃ b' c', b' = 2000 ∧ c' = 2000 ∧
                  (∃ k : ℚ, k = 1/2 ∧
                  (∃ x y z, x = a / 2 ∧ y = 1000 ∧ z = c / 2 ∧ a = 2000 ∧ c = 2000)))
/- The proof is omitted for this statement. -/
:= sorry

end prism_unique_triple_l208_208694


namespace direct_variation_exponent_l208_208490

variable {X Y Z : Type}

theorem direct_variation_exponent (k j : ℝ) (x y z : ℝ) 
  (h1 : x = k * y^4) 
  (h2 : y = j * z^3) : 
  ∃ m : ℝ, x = m * z^12 :=
by
  sorry

end direct_variation_exponent_l208_208490


namespace max_difference_is_correct_l208_208735

noncomputable def max_y_difference : ℝ := 
  let x1 := Real.sqrt (2 / 3)
  let y1 := 2 + (x1 ^ 2) + (x1 ^ 3)
  let x2 := -x1
  let y2 := 2 + (x2 ^ 2) + (x2 ^ 3)
  abs (y1 - y2)

theorem max_difference_is_correct : max_y_difference = 4 * Real.sqrt 2 / 9 := 
  sorry -- Proof is omitted

end max_difference_is_correct_l208_208735


namespace no_snow_five_days_l208_208741

noncomputable def prob_snow_each_day : ℚ := 2 / 3

noncomputable def prob_no_snow_one_day : ℚ := 1 - prob_snow_each_day

noncomputable def prob_no_snow_five_days : ℚ := prob_no_snow_one_day ^ 5

theorem no_snow_five_days:
  prob_no_snow_five_days = 1 / 243 :=
by
  sorry

end no_snow_five_days_l208_208741


namespace tan_135_eq_neg_one_l208_208476

theorem tan_135_eq_neg_one : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg_one_l208_208476


namespace plywood_perimeter_difference_l208_208920

theorem plywood_perimeter_difference :
  let l := 10
  let w := 6
  let n := 6
  ∃ p_max p_min, 
    (l * w) % n = 0 ∧
    (p_max = 24) ∧
    (p_min = 12.66) ∧
    p_max - p_min = 11.34 := 
by
  sorry

end plywood_perimeter_difference_l208_208920


namespace hexagon_triangle_count_l208_208140

-- Definitions based on problem conditions
def numPoints : ℕ := 7
def totalTriangles := Nat.choose numPoints 3
def collinearCases : ℕ := 3

-- Proof problem
theorem hexagon_triangle_count : totalTriangles - collinearCases = 32 :=
by
  -- Calculation is expected here
  sorry

end hexagon_triangle_count_l208_208140


namespace initial_money_in_wallet_l208_208816

theorem initial_money_in_wallet (x : ℝ) 
  (h1 : x = 78 + 16) : 
  x = 94 :=
by
  sorry

end initial_money_in_wallet_l208_208816


namespace ab_value_l208_208323

theorem ab_value (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 33) : a * b = 18 := 
by
  sorry

end ab_value_l208_208323


namespace remainder_when_divided_l208_208969

theorem remainder_when_divided (P D Q R D'' Q'' R'' : ℕ) (h1 : P = Q * D + R) (h2 : Q = D'' * Q'' + R'') :
  P % (2 * D * D'') = D * R'' + R := sorry

end remainder_when_divided_l208_208969


namespace least_possible_number_l208_208505

theorem least_possible_number :
  ∃ x : ℕ, (∃ q r : ℕ, x = 34 * q + r ∧ 0 ≤ r ∧ r < 34) ∧
            (∃ q' : ℕ, x = 5 * q' ∧ q' = r + 8) ∧
            x = 75 :=
by
  sorry

end least_possible_number_l208_208505


namespace infinite_bad_numbers_l208_208056

-- Define types for natural numbers
variables {a b : ℕ}

-- The theorem statement
theorem infinite_bad_numbers (a b : ℕ) : ∃ᶠ (n : ℕ) in at_top, n > 0 ∧ ¬ (n^b + 1 ∣ a^n + 1) :=
sorry

end infinite_bad_numbers_l208_208056


namespace first_digit_base12_1025_l208_208119

theorem first_digit_base12_1025 : (1025 : ℕ) / (12^2 : ℕ) = 7 := by
  sorry

end first_digit_base12_1025_l208_208119


namespace expenditure_on_digging_l208_208514

noncomputable def volume_of_cylinder (r h : ℝ) := 
  Real.pi * r^2 * h

noncomputable def rate_per_cubic_meter (cost : ℝ) (r h : ℝ) : ℝ := 
  cost / (volume_of_cylinder r h)

theorem expenditure_on_digging (d h : ℝ) (cost : ℝ) (r : ℝ) (π : ℝ) (rate : ℝ)
  (h₀ : d = 3) (h₁ : h = 14) (h₂ : cost = 1682.32) (h₃ : r = d / 2) (h₄ : π = Real.pi) 
  : rate_per_cubic_meter cost r h = 17 := sorry

end expenditure_on_digging_l208_208514


namespace average_score_is_7_stddev_is_2_l208_208475

-- Define the scores list
def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

-- Proof statement for average score
theorem average_score_is_7 : (scores.sum / scores.length) = 7 :=
by
  simp [scores]
  sorry

-- Proof statement for standard deviation
theorem stddev_is_2 : Real.sqrt ((scores.map (λ x => (x - (scores.sum / scores.length))^2)).sum / scores.length) = 2 :=
by
  simp [scores]
  sorry

end average_score_is_7_stddev_is_2_l208_208475


namespace solve_system_of_inequalities_l208_208199

theorem solve_system_of_inequalities (x y : ℤ) :
  (2 * x - y > 3 ∧ 3 - 2 * x + y > 0) ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := 
by { sorry }

end solve_system_of_inequalities_l208_208199


namespace vector_addition_l208_208828

-- Let vectors a and b be defined as
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -3)

-- Theorem statement to prove
theorem vector_addition : a + 2 • b = (4, -5) :=
by
  sorry

end vector_addition_l208_208828


namespace election_total_votes_l208_208024

theorem election_total_votes (V_A V_B V : ℕ) (H1 : V_A = V_B + 15/100 * V) (H2 : V_A + V_B = 80/100 * V) (H3 : V_B = 2184) : V = 6720 :=
sorry

end election_total_votes_l208_208024


namespace matrix_multiplication_correct_l208_208954

-- Define the matrices
def A : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, 0, -3],
    ![1, 3, -2],
    ![0, 2, 4]
  ]

def B : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![1, -1, 0],
    ![0, 2, -1],
    ![3, 0, 1]
  ]

def C : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![-7, -2, -3],
    ![-5, 5, -5],
    ![12, 4, 2]
  ]

-- Proof statement that multiplication of A and B gives C
theorem matrix_multiplication_correct : A * B = C := 
by
  sorry

end matrix_multiplication_correct_l208_208954


namespace complex_number_in_second_quadrant_l208_208185

theorem complex_number_in_second_quadrant 
  (a b : ℝ) 
  (h : ¬ (a ≥ 0 ∨ b ≤ 0)) : 
  (a < 0 ∧ b > 0) :=
sorry

end complex_number_in_second_quadrant_l208_208185


namespace num_triangles_in_circle_l208_208184

noncomputable def num_triangles (n : ℕ) : ℕ :=
  n.choose 3

theorem num_triangles_in_circle (n : ℕ) :
  num_triangles n = n.choose 3 :=
by
  sorry

end num_triangles_in_circle_l208_208184


namespace same_color_pair_exists_l208_208404

-- Define the coloring of a point on a plane
def is_colored (x y : ℝ) : Type := ℕ  -- Assume ℕ represents two colors 0 and 1

-- Prove there exists two points of the same color such that the distance between them is 2006 meters
theorem same_color_pair_exists (colored : ℝ → ℝ → ℕ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∧ y1 ≠ y2 ∧ colored x1 y1 = colored x2 y2 ∧ (x2 - x1)^2 + (y2 - y1)^2 = 2006^2) :=
sorry

end same_color_pair_exists_l208_208404


namespace expression_undefined_iff_l208_208223

theorem expression_undefined_iff (a : ℝ) : (a^2 - 9 = 0) ↔ (a = 3 ∨ a = -3) :=
sorry

end expression_undefined_iff_l208_208223


namespace problem_rewrite_expression_l208_208706

theorem problem_rewrite_expression (j : ℝ) : 
  ∃ (c p q : ℝ), (8 * j^2 - 6 * j + 20 = c * (j + p)^2 + q) ∧ (q / p = -77) :=
sorry

end problem_rewrite_expression_l208_208706


namespace original_price_l208_208462

theorem original_price (total_payment : ℝ) (num_units : ℕ) (discount_rate : ℝ) 
(h1 : total_payment = 500) (h2 : num_units = 18) (h3 : discount_rate = 0.20) : 
  (total_payment / (1 - discount_rate) * num_units) = 625.05 :=
by
  sorry

end original_price_l208_208462


namespace correct_propositions_l208_208967

-- Definitions based on the propositions
def prop1 := 
"Sampling every 20 minutes from a uniformly moving production line is stratified sampling."

def prop2 := 
"The stronger the correlation between two random variables, the closer the absolute value of the correlation coefficient is to 1."

def prop3 := 
"In the regression line equation hat_y = 0.2 * x + 12, the forecasted variable hat_y increases by 0.2 units on average for each unit increase in the explanatory variable x."

def prop4 := 
"For categorical variables X and Y, the smaller the observed value k of their statistic K², the greater the certainty of the relationship between X and Y."

-- Mathematical statements for propositions
def p1 : Prop := false -- Proposition ① is incorrect
def p2 : Prop := true  -- Proposition ② is correct
def p3 : Prop := true  -- Proposition ③ is correct
def p4 : Prop := false -- Proposition ④ is incorrect

-- The theorem we need to prove
theorem correct_propositions : (p2 = true) ∧ (p3 = true) :=
by 
  -- Details of the proof here
  sorry

end correct_propositions_l208_208967


namespace tan_product_identity_l208_208108

-- Lean statement for the mathematical problem
theorem tan_product_identity : 
  (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 := by
  sorry

end tan_product_identity_l208_208108


namespace journey_time_difference_l208_208707

theorem journey_time_difference :
  let speed := 40  -- mph
  let distance1 := 360  -- miles
  let distance2 := 320  -- miles
  (distance1 / speed - distance2 / speed) * 60 = 60 := 
by
  sorry

end journey_time_difference_l208_208707


namespace find_n_l208_208482

-- Define the conditions as hypothesis
variables (A B n : ℕ)

-- Hypothesis 1: This year, Ana's age is the square of Bonita's age.
-- A = B^2
#check (A = B^2) 

-- Hypothesis 2: Last year Ana was 5 times as old as Bonita.
-- A - 1 = 5 * (B - 1)
#check (A - 1 = 5 * (B - 1))

-- Hypothesis 3: Ana and Bonita were born n years apart.
-- A = B + n
#check (A = B + n)

-- Goal: The difference in their ages, n, should be 12.
theorem find_n (A B n : ℕ) (h1 : A = B^2) (h2 : A - 1 = 5 * (B - 1)) (h3 : A = B + n) : n = 12 :=
sorry

end find_n_l208_208482


namespace min_possible_value_box_l208_208880

theorem min_possible_value_box :
  ∃ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15 ∧ a^2 + b^2 = 61) ∧
  ∀ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15) → (a^2 + b^2 ≥ 61) :=
by {
  sorry
}

end min_possible_value_box_l208_208880


namespace sequence_formula_general_formula_l208_208032

open BigOperators

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2 * n + 2

def S_n (n : ℕ) : ℕ :=
  n^2 + 3 * n + 1

theorem sequence_formula :
  ∀ n, a_n n =
    if n = 1 then 5 else 2 * n + 2 := by
  sorry

theorem general_formula (n : ℕ) :
  a_n n =
    if n = 1 then S_n 1 else S_n n - S_n (n - 1) := by
  sorry

end sequence_formula_general_formula_l208_208032


namespace find_X_l208_208480

theorem find_X : 
  let M := 3012 / 4
  let N := M / 4
  let X := M - N
  X = 564.75 :=
by
  sorry

end find_X_l208_208480


namespace lcm_of_4_8_9_10_l208_208258

theorem lcm_of_4_8_9_10 : Nat.lcm (Nat.lcm 4 8) (Nat.lcm 9 10) = 360 := by
  sorry

end lcm_of_4_8_9_10_l208_208258


namespace distance_covered_l208_208064

noncomputable def boat_speed_still_water : ℝ := 6.5
noncomputable def current_speed : ℝ := 2.5
noncomputable def time_taken : ℝ := 35.99712023038157

noncomputable def effective_speed_downstream (boat_speed_still_water current_speed : ℝ) : ℝ :=
  boat_speed_still_water + current_speed

noncomputable def convert_kmph_to_mps (speed_in_kmph : ℝ) : ℝ :=
  speed_in_kmph * (1000 / 3600)

noncomputable def calculate_distance (speed_in_mps time_in_seconds : ℝ) : ℝ :=
  speed_in_mps * time_in_seconds

theorem distance_covered :
  calculate_distance (convert_kmph_to_mps (effective_speed_downstream boat_speed_still_water current_speed)) time_taken = 89.99280057595392 :=
by
  sorry

end distance_covered_l208_208064


namespace inequality_positive_reals_l208_208265

theorem inequality_positive_reals (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  (x / (1 + x^2)) + (y / (1 + y^2)) + (z / (1 + z^2)) ≤ (3 * Real.sqrt 3) / 4 := by
  sorry

end inequality_positive_reals_l208_208265


namespace rectangle_perimeter_l208_208171

variable (x : ℝ) (y : ℝ)

-- Definitions based on conditions
def area_of_rectangle : Prop := x * (x + 5) = 500
def side_length_relation : Prop := y = x + 5

-- The theorem we want to prove
theorem rectangle_perimeter (h_area : area_of_rectangle x) (h_side_length : side_length_relation x y) : 2 * (x + y) = 90 := by
  sorry

end rectangle_perimeter_l208_208171


namespace fraction_meaningful_l208_208382

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end fraction_meaningful_l208_208382


namespace sufficient_not_necessary_l208_208137

variables (A B : Prop)

theorem sufficient_not_necessary (h : B → A) : ¬(A → B) :=
by sorry

end sufficient_not_necessary_l208_208137


namespace yuna_has_biggest_number_l208_208570

theorem yuna_has_biggest_number (yoongi : ℕ) (jungkook : ℕ) (yuna : ℕ) (hy : yoongi = 7) (hj : jungkook = 6) (hn : yuna = 9) :
  yuna = 9 ∧ yuna > yoongi ∧ yuna > jungkook :=
by 
  sorry

end yuna_has_biggest_number_l208_208570


namespace factorize_expression_l208_208872

theorem factorize_expression (a b : ℝ) : 3 * a ^ 2 - 3 * b ^ 2 = 3 * (a + b) * (a - b) :=
by
  sorry

end factorize_expression_l208_208872


namespace arithmetic_geometric_relation_l208_208518

variable (a₁ a₂ b₁ b₂ b₃ : ℝ)

-- Conditions
def is_arithmetic_sequence (a₁ a₂ : ℝ) : Prop :=
  ∃ (d : ℝ), -2 + d = a₁ ∧ a₁ + d = a₂ ∧ a₂ + d = -8

def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ (r : ℝ), -2 * r = b₁ ∧ b₁ * r = b₂ ∧ b₂ * r = b₃ ∧ b₃ * r = -8

-- The problem statement
theorem arithmetic_geometric_relation (h₁ : is_arithmetic_sequence a₁ a₂) (h₂ : is_geometric_sequence b₁ b₂ b₃) :
  (a₂ - a₁) / b₂ = 1 / 2 := by
    sorry

end arithmetic_geometric_relation_l208_208518


namespace find_unit_price_B_l208_208590

variable (x : ℕ)

def unit_price_B := x
def unit_price_A := x + 50

theorem find_unit_price_B (h : (2000 / unit_price_A x = 1500 / unit_price_B x)) : unit_price_B x = 150 :=
by
  sorry

end find_unit_price_B_l208_208590


namespace sequence_propositions_l208_208777

theorem sequence_propositions (a : ℕ → ℝ) (h_seq : a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 > a 4 ∧ a 4 ≥ 0) 
  (h_sub : ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 4 → ∃ k, a i - a j = a k) :
  (∀ k, ∃ d, a k = a 1 - d * (k - 1)) ∧
  (∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ i * a i = j * a j) ∧
  (∃ i, a i = 0) :=
by
  sorry

end sequence_propositions_l208_208777


namespace harmonic_power_identity_l208_208635

open Real

theorem harmonic_power_identity (a b c : ℝ) (n : ℕ) (hn : n % 2 = 1) 
(h : (1 / a + 1 / b + 1 / c) = 1 / (a + b + c)) :
  (1 / (a ^ n) + 1 / (b ^ n) + 1 / (c ^ n) = 1 / (a ^ n + b ^ n + c ^ n)) :=
sorry

end harmonic_power_identity_l208_208635


namespace det_condition_l208_208923

theorem det_condition (a b c d : ℤ) 
    (h_exists : ∀ m n : ℤ, ∃ h k : ℤ, a * h + b * k = m ∧ c * h + d * k = n) :
    |a * d - b * c| = 1 :=
sorry

end det_condition_l208_208923


namespace chewbacca_gum_packs_l208_208859

theorem chewbacca_gum_packs (x : ℕ) :
  (30 - 2 * x) * (40 + 4 * x) = 1200 → x = 5 :=
by
  -- This is where the proof would go. We'll leave it as sorry for now.
  sorry

end chewbacca_gum_packs_l208_208859


namespace find_omega_l208_208993

noncomputable def omega_solution (ω : ℝ) : Prop :=
  ω > 0 ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) > 2 * Real.cos (ω * y)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) ≥ 1)

theorem find_omega : omega_solution (1 / 2) :=
sorry

end find_omega_l208_208993


namespace find_divisor_l208_208874

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 34) / y = 2) : y = 10 :=
by
  sorry

end find_divisor_l208_208874


namespace product_of_b_values_is_neg_12_l208_208410

theorem product_of_b_values_is_neg_12 (b : ℝ) (y1 y2 x1 : ℝ) (h1 : y1 = 3) (h2 : y2 = 7) (h3 : x1 = 2) (h4 : y2 - y1 = 4) (h5 : ∃ b1 b2, b1 = x1 - 4 ∧ b2 = x1 + 4) : 
  (b1 * b2 = -12) :=
by
  sorry

end product_of_b_values_is_neg_12_l208_208410


namespace combination_sum_l208_208693

noncomputable def combination (n r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_sum :
  combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + 
  combination 7 2 + combination 8 2 + combination 9 2 + combination 10 2 = 164 :=
by
  sorry

end combination_sum_l208_208693


namespace smallest_arithmetic_geometric_seq_sum_l208_208881

variable (A B C D : ℕ)

noncomputable def arithmetic_seq (A B C : ℕ) (d : ℕ) : Prop :=
  B - A = d ∧ C - B = d

noncomputable def geometric_seq (B C D : ℕ) : Prop :=
  C = (5 / 3) * B ∧ D = (25 / 9) * B

theorem smallest_arithmetic_geometric_seq_sum :
  ∃ A B C D : ℕ, 
    arithmetic_seq A B C 12 ∧ 
    geometric_seq B C D ∧ 
    (A + B + C + D = 104) :=
sorry

end smallest_arithmetic_geometric_seq_sum_l208_208881


namespace find_a1_l208_208063

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a (n + 1) = a n + d

theorem find_a1 (h_arith : is_arithmetic_sequence a 3) (ha2 : a 2 = -5) : a 1 = -8 :=
sorry

end find_a1_l208_208063


namespace alice_additional_cookies_proof_l208_208737

variable (alice_initial_cookies : ℕ)
variable (bob_initial_cookies : ℕ)
variable (cookies_thrown_away : ℕ)
variable (bob_additional_cookies : ℕ)
variable (total_edible_cookies : ℕ)

theorem alice_additional_cookies_proof 
    (h1 : alice_initial_cookies = 74)
    (h2 : bob_initial_cookies = 7)
    (h3 : cookies_thrown_away = 29)
    (h4 : bob_additional_cookies = 36)
    (h5 : total_edible_cookies = 93) :
  alice_initial_cookies + bob_initial_cookies - cookies_thrown_away + bob_additional_cookies + (93 - (74 + 7 - 29 + 36)) = total_edible_cookies :=
by
  sorry

end alice_additional_cookies_proof_l208_208737


namespace sum_digits_probability_l208_208017

noncomputable def sumOfDigits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def numInRange : ℕ := 1000000

noncomputable def coefficient : ℕ :=
  Nat.choose 24 5 - 6 * Nat.choose 14 5

noncomputable def probability : ℚ :=
  coefficient / numInRange

theorem sum_digits_probability :
  probability = 7623 / 250000 :=
by
  sorry

end sum_digits_probability_l208_208017


namespace find_a_from_roots_l208_208431

theorem find_a_from_roots (θ : ℝ) (a : ℝ) (h1 : ∀ x : ℝ, 4 * x^2 + 2 * a * x + a = 0 → (x = Real.sin θ ∨ x = Real.cos θ)) :
  a = 1 - Real.sqrt 5 :=
by
  sorry

end find_a_from_roots_l208_208431


namespace find_x_value_l208_208592

theorem find_x_value (A B C x : ℝ) (hA : A = 40) (hB : B = 3 * x) (hC : C = 2 * x) (hSum : A + B + C = 180) : x = 28 :=
by
  sorry

end find_x_value_l208_208592


namespace cans_collected_by_first_group_l208_208371

def class_total_students : ℕ := 30
def students_didnt_collect : ℕ := 2
def students_collected_4 : ℕ := 13
def total_cans_collected : ℕ := 232

theorem cans_collected_by_first_group :
  let remaining_students := class_total_students - (students_didnt_collect + students_collected_4)
  let cans_by_13_students := students_collected_4 * 4
  let cans_by_first_group := total_cans_collected - cans_by_13_students
  let cans_per_student := cans_by_first_group / remaining_students
  cans_per_student = 12 := by
  sorry

end cans_collected_by_first_group_l208_208371


namespace angle_bisector_inequality_l208_208398

theorem angle_bisector_inequality
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h_perimeter : (x + y + z) = 6) :
  (1 / x^2) + (1 / y^2) + (1 / z^2) ≥ 1 := by
  sorry

end angle_bisector_inequality_l208_208398


namespace option_C_correct_l208_208469

theorem option_C_correct : ∀ x : ℝ, x^2 + 1 ≥ 2 * |x| :=
by
  intro x
  sorry

end option_C_correct_l208_208469


namespace interest_difference_l208_208487

theorem interest_difference
  (principal : ℕ) (rate : ℚ) (time : ℕ) (interest : ℚ) (difference : ℚ)
  (h1 : principal = 600)
  (h2 : rate = 0.05)
  (h3 : time = 8)
  (h4 : interest = principal * (rate * time))
  (h5 : difference = principal - interest) :
  difference = 360 :=
by sorry

end interest_difference_l208_208487


namespace find_num_students_B_l208_208796

-- Given conditions as definitions
def num_students_A : ℕ := 24
def avg_weight_A : ℚ := 40
def avg_weight_B : ℚ := 35
def avg_weight_class : ℚ := 38

-- The total weight for sections A and B
def total_weight_A : ℚ := num_students_A * avg_weight_A
def total_weight_B (x: ℕ) : ℚ := x * avg_weight_B

-- The number of students in section B
noncomputable def num_students_B : ℕ := 16

-- The proof problem: Prove that number of students in section B is 16
theorem find_num_students_B (x: ℕ) (h: (total_weight_A + total_weight_B x) / (num_students_A + x) = avg_weight_class) : 
  x = 16 :=
by
  sorry

end find_num_students_B_l208_208796


namespace number_of_adult_female_alligators_l208_208512

-- Define the conditions
def total_alligators (females males: ℕ) : ℕ := females + males

def male_alligators : ℕ := 25
def female_alligators : ℕ := 25
def juvenile_percentage : ℕ := 40

-- Calculate the number of juveniles
def juvenile_count : ℕ := (juvenile_percentage * female_alligators) / 100

-- Calculate the number of adults
def adult_female_alligators : ℕ := female_alligators - juvenile_count

-- The main theorem statement
theorem number_of_adult_female_alligators : adult_female_alligators = 15 :=
by
    sorry

end number_of_adult_female_alligators_l208_208512


namespace stratified_sampling_yogurt_adult_milk_powder_sum_l208_208343

theorem stratified_sampling_yogurt_adult_milk_powder_sum :
  let liquid_milk_brands := 40
  let yogurt_brands := 10
  let infant_formula_brands := 30
  let adult_milk_powder_brands := 20
  let total_brands := liquid_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sample_size := 20
  let yogurt_sample := sample_size * yogurt_brands / total_brands
  let adult_milk_powder_sample := sample_size * adult_milk_powder_brands / total_brands
  yogurt_sample + adult_milk_powder_sample = 6 :=
by
  sorry

end stratified_sampling_yogurt_adult_milk_powder_sum_l208_208343


namespace quadratic_has_two_distinct_real_roots_iff_l208_208259

theorem quadratic_has_two_distinct_real_roots_iff (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 - 2 * x1 + k - 1 = 0 ∧ x2 * x2 - 2 * x2 + k - 1 = 0) ↔ k < 2 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_iff_l208_208259


namespace Vovochka_correct_pairs_count_l208_208248

def no_carry_pairs_count (digit_sum: ℕ → ℕ → Prop) : ℕ :=
  let count_pairs (lim: ℕ) : ℕ := (lim * (lim + 1)) / 2
  let digit_valid_pairs : ℕ := count_pairs 9
  (digit_valid_pairs * digit_valid_pairs) * 81

def digit_sum (x y: ℕ) : Prop := (x + y ≤ 9)

theorem Vovochka_correct_pairs_count :
  no_carry_pairs_count digit_sum = 244620 := by
  sorry

end Vovochka_correct_pairs_count_l208_208248


namespace right_triangle_condition_l208_208079

theorem right_triangle_condition (a d : ℝ) (h : d > 0) : 
  (a = d * (1 + Real.sqrt 7)) ↔ (a^2 + (a + 2 * d)^2 = (a + 4 * d)^2) := 
sorry

end right_triangle_condition_l208_208079


namespace fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l208_208739

theorem fixed_point_of_line (a : ℝ) (A : ℝ × ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> A = (1, 6)) :=
sorry

theorem range_of_a_to_avoid_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> x * y < 0 -> a ≤ -5) :=
sorry

end fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l208_208739


namespace average_production_is_correct_l208_208895

noncomputable def average_tv_production_last_5_days
  (daily_production : ℕ)
  (ill_workers : List ℕ)
  (decrease_rate : ℕ) : ℚ :=
  let productivity_decrease (n : ℕ) : ℚ := (1 - (decrease_rate * n) / 100 : ℚ) * daily_production
  let total_production := (ill_workers.map productivity_decrease).sum
  total_production / ill_workers.length

theorem average_production_is_correct :
  average_tv_production_last_5_days 50 [3, 5, 2, 4, 3] 2 = 46.6 :=
by
  -- proof needed here
  sorry

end average_production_is_correct_l208_208895


namespace intersecting_lines_l208_208521

theorem intersecting_lines (c d : ℝ) 
  (h1 : 3 = (1/3 : ℝ) * 0 + c)
  (h2 : 0 = (1/3 : ℝ) * 3 + d) :
  c + d = 2 := 
by {
  sorry
}

end intersecting_lines_l208_208521


namespace probability_value_l208_208768

noncomputable def P (k : ℕ) (c : ℚ) : ℚ := c / (k * (k + 1))

theorem probability_value (c : ℚ) (h : P 1 c + P 2 c + P 3 c + P 4 c = 1) : P 1 c + P 2 c = 5 / 6 := 
by
  sorry

end probability_value_l208_208768


namespace cut_half_meter_from_two_thirds_l208_208758

theorem cut_half_meter_from_two_thirds (L : ℝ) (hL : L = 2 / 3) : L - 1 / 6 = 1 / 2 :=
by
  rw [hL]
  norm_num

end cut_half_meter_from_two_thirds_l208_208758


namespace harold_savings_l208_208567

theorem harold_savings :
  let income_primary := 2500
  let income_freelance := 500
  let rent := 700
  let car_payment := 300
  let car_insurance := 125
  let electricity := 0.25 * car_payment
  let water := 0.15 * rent
  let internet := 75
  let groceries := 200
  let miscellaneous := 150
  let total_income := income_primary + income_freelance
  let total_expenses := rent + car_payment + car_insurance + electricity + water + internet + groceries + miscellaneous
  let amount_before_savings := total_income - total_expenses
  let retirement := (1/3) * amount_before_savings
  let emergency := (1/3) * amount_before_savings
  let amount_after_savings := amount_before_savings - retirement - emergency
  amount_after_savings = 423.34 := 
sorry

end harold_savings_l208_208567


namespace minimum_voters_needed_l208_208122

-- conditions
def num_voters := 135
def num_districts := 5
def precincts_per_district := 9
def voters_per_precinct := 3
def majority_precincts (n : ℕ) := (n + 1) / 2

-- definitions for quantities derived from conditions
def total_precincts := num_districts * precincts_per_district
def majority_districts := majority_precincts num_districts
def precincts_needed_for_district_win := majority_precincts precincts_per_district
def total_precincts_needed_for_win := majority_districts * precincts_needed_for_district_win
def votes_needed_per_precinct := majority_precincts voters_per_precinct

-- main statement
theorem minimum_voters_needed : (votes_needed_per_precinct * total_precincts_needed_for_win = 30) ∧ TallGiraffeWon :=
by sorry

end minimum_voters_needed_l208_208122


namespace factorize_9_minus_a_squared_l208_208114

theorem factorize_9_minus_a_squared (a : ℤ) : 9 - a^2 = (3 + a) * (3 - a) :=
by
  sorry

end factorize_9_minus_a_squared_l208_208114


namespace problem_statement_l208_208716

theorem problem_statement (x y : ℝ) (h₁ : |x| = 3) (h₂ : |y| = 4) (h₃ : x > y) : 2 * x - y = 10 := 
by {
  sorry
}

end problem_statement_l208_208716


namespace fred_carrots_l208_208023

-- Define the conditions
def sally_carrots : Nat := 6
def total_carrots : Nat := 10

-- Define the problem question and the proof statement
theorem fred_carrots : ∃ fred_carrots : Nat, fred_carrots = total_carrots - sally_carrots := 
by
  sorry

end fred_carrots_l208_208023


namespace total_watermelons_l208_208087

def watermelons_grown_by_jason : ℕ := 37
def watermelons_grown_by_sandy : ℕ := 11

theorem total_watermelons : watermelons_grown_by_jason + watermelons_grown_by_sandy = 48 := by
  sorry

end total_watermelons_l208_208087


namespace mean_transformation_l208_208912

variable {x1 x2 x3 : ℝ}
variable (s : ℝ)
variable (h_var : s^2 = (1 / 3) * (x1^2 + x2^2 + x3^2 - 12))

theorem mean_transformation :
  (x1 + 1 + x2 + 1 + x3 + 1) / 3 = 3 :=
by
  sorry

end mean_transformation_l208_208912


namespace max_distance_AB_l208_208073

-- Define curve C1 in Cartesian coordinates
def C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define curve C2 in Cartesian coordinates
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the problem to prove the maximum value of distance AB is 8
theorem max_distance_AB :
  ∀ (Ax Ay Bx By : ℝ),
    C1 Ax Ay →
    C2 Bx By →
    dist (Ax, Ay) (Bx, By) ≤ 8 :=
sorry

end max_distance_AB_l208_208073


namespace sum_of_twos_and_threes_3024_l208_208018

theorem sum_of_twos_and_threes_3024 : ∃ n : ℕ, n = 337 ∧ (∃ (a b : ℕ), 3024 = 2 * a + 3 * b) :=
sorry

end sum_of_twos_and_threes_3024_l208_208018


namespace base6_add_sub_l208_208386

theorem base6_add_sub (a b c : ℕ) (ha : a = 5 * 6^2 + 5 * 6^1 + 5 * 6^0)
  (hb : b = 6 * 6^1 + 5 * 6^0) (hc : c = 1 * 6^1 + 1 * 6^0) :
  (a + b - c) = 1 * 6^3 + 0 * 6^2 + 5 * 6^1 + 3 * 6^0 :=
by
  -- We should translate the problem context into equivalence
  -- but this part of the actual proof is skipped with sorry.
  sorry

end base6_add_sub_l208_208386


namespace regular_octahedron_vertices_count_l208_208898

def regular_octahedron_faces := 8
def regular_octahedron_edges := 12
def regular_octahedron_faces_shape := "equilateral triangle"
def regular_octahedron_vertices_meet := 4

theorem regular_octahedron_vertices_count :
  ∀ (F E V : ℕ),
    F = regular_octahedron_faces →
    E = regular_octahedron_edges →
    (∀ (v : ℕ), v = regular_octahedron_vertices_meet) →
    V = 6 :=
by
  intros F E V hF hE hV
  sorry

end regular_octahedron_vertices_count_l208_208898


namespace range_of_a_for_inequality_l208_208508

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 :=
by {
  sorry
}

end range_of_a_for_inequality_l208_208508


namespace sum_of_distances_l208_208616

theorem sum_of_distances (P : ℤ × ℤ) (hP : P = (-1, -2)) :
  abs P.1 + abs P.2 = 3 :=
sorry

end sum_of_distances_l208_208616


namespace angle_in_third_quadrant_half_l208_208028

theorem angle_in_third_quadrant_half {
  k : ℤ 
} (h1: (k * 360 + 180) < α) (h2 : α < k * 360 + 270) :
  (k * 180 + 90) < (α / 2) ∧ (α / 2) < (k * 180 + 135) :=
sorry

end angle_in_third_quadrant_half_l208_208028


namespace triangle_properties_l208_208736

theorem triangle_properties (A B C a b c : ℝ) (h1 : a * Real.tan C = 2 * c * Real.sin A)
  (h2 : C > 0 ∧ C < Real.pi)
  (h3 : a / Real.sin A = c / Real.sin C) :
  C = Real.pi / 3 ∧ (1 / 2 < Real.sin (A + Real.pi / 6) ∧ Real.sin (A + Real.pi / 6) ≤ 1) →
  (Real.sqrt 3 / 2 < Real.sin A + Real.sin B ∧ Real.sin A + Real.sin B ≤ Real.sqrt 3) :=
by
  intro h4
  sorry

end triangle_properties_l208_208736


namespace inverse_proportion_k_value_l208_208942

theorem inverse_proportion_k_value (k m : ℝ) 
  (h1 : m = k / 3) 
  (h2 : 6 = k / (m - 1)) 
  : k = 6 :=
by
  sorry

end inverse_proportion_k_value_l208_208942


namespace stanley_total_cost_l208_208000

theorem stanley_total_cost (n_tires : ℕ) (price_per_tire : ℝ) (h_n : n_tires = 4) (h_price : price_per_tire = 60) : n_tires * price_per_tire = 240 := by
  sorry

end stanley_total_cost_l208_208000


namespace possible_atomic_numbers_l208_208387

/-
Given the following conditions:
1. An element X is from Group IIA and exhibits a +2 charge.
2. An element Y is from Group VIIA and exhibits a -1 charge.
Prove that the possible atomic numbers for elements X and Y that can form an ionic compound with the formula XY₂ are 12 for X and 9 for Y.
-/

structure Element :=
  (atomic_number : Nat)
  (group : Nat)
  (charge : Int)

def GroupIIACharge := 2
def GroupVIIACharge := -1

axiom X : Element
axiom Y : Element

theorem possible_atomic_numbers (X_group_IIA : X.group = 2)
                                (X_charge : X.charge = GroupIIACharge)
                                (Y_group_VIIA : Y.group = 7)
                                (Y_charge : Y.charge = GroupVIIACharge) :
  (X.atomic_number = 12) ∧ (Y.atomic_number = 9) :=
sorry

end possible_atomic_numbers_l208_208387


namespace problem1_problem2_l208_208492

noncomputable def cos_alpha (α : ℝ) : ℝ := (Real.sqrt 2 + 4) / 6
noncomputable def cos_alpha_plus_half_beta (α β : ℝ) : ℝ := 5 * Real.sqrt 3 / 9

theorem problem1 {α : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) :
  Real.cos α = cos_alpha α :=
sorry

theorem problem2 {α β : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (hβ1 : -Real.pi / 2 < β) (hβ2 : β < 0) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) 
                 (h2 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α + β / 2) = cos_alpha_plus_half_beta α β :=
sorry

end problem1_problem2_l208_208492


namespace remainder_problem_l208_208659

theorem remainder_problem (n : ℤ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end remainder_problem_l208_208659


namespace math_problem_l208_208504

variables (a b c : ℤ)

theorem math_problem (h1 : a - (b - 2 * c) = 19) (h2 : a - b - 2 * c = 7) : a - b = 13 := by
  sorry

end math_problem_l208_208504


namespace crayons_erasers_difference_l208_208163

theorem crayons_erasers_difference
  (initial_erasers : ℕ) (initial_crayons : ℕ) (final_crayons : ℕ)
  (no_eraser_lost : initial_erasers = 457)
  (initial_crayons_condition : initial_crayons = 617)
  (final_crayons_condition : final_crayons = 523) :
  final_crayons - initial_erasers = 66 :=
by
  -- These would be assumptions in the proof; be aware that 'sorry' is used to skip the proof details.
  sorry

end crayons_erasers_difference_l208_208163


namespace ellipse_parameters_l208_208161

theorem ellipse_parameters 
  (x y : ℝ)
  (h : 2 * x^2 + y^2 + 42 = 8 * x + 36 * y) :
  ∃ (h k : ℝ) (a b : ℝ), 
    (h = 2) ∧ (k = 18) ∧ (a = Real.sqrt 290) ∧ (b = Real.sqrt 145) ∧ 
    ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1 :=
sorry

end ellipse_parameters_l208_208161


namespace max_AMC_expression_l208_208437

theorem max_AMC_expression (A M C : ℕ) (h : A + M + C = 15) : A * M * C + A * M + M * C + C * A ≤ 200 :=
by
  sorry

end max_AMC_expression_l208_208437


namespace distance_from_A_to_B_l208_208494

theorem distance_from_A_to_B (d C1A C1B C2A C2B : ℝ) (h1 : C1A + C1B = d)
  (h2 : C2A + C2B = d) (h3 : (C1A = 2 * C1B) ∨ (C1B = 2 * C1A)) 
  (h4 : (C2A = 3 * C2B) ∨ (C2B = 3 * C2A))
  (h5 : |C2A - C1A| = 10) : d = 120 ∨ d = 24 :=
sorry

end distance_from_A_to_B_l208_208494


namespace find_second_smallest_odd_number_l208_208652

theorem find_second_smallest_odd_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6) = 112)) : (x + 2 = 27) :=
sorry

end find_second_smallest_odd_number_l208_208652


namespace rachel_fathers_age_when_rachel_is_25_l208_208597

theorem rachel_fathers_age_when_rachel_is_25 (R G M F Y : ℕ) 
  (h1 : R = 12)
  (h2 : G = 7 * R)
  (h3 : M = G / 2)
  (h4 : F = M + 5)
  (h5 : Y = 25 - R) : 
  F + Y = 60 :=
by sorry

end rachel_fathers_age_when_rachel_is_25_l208_208597


namespace negation_true_l208_208847

theorem negation_true (a : ℝ) : ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) :=
sorry

end negation_true_l208_208847


namespace car_grid_probability_l208_208082

theorem car_grid_probability:
  let m := 11
  let n := 48
  100 * m + n = 1148 := by
  sorry

end car_grid_probability_l208_208082


namespace cost_of_each_ticket_l208_208671

theorem cost_of_each_ticket (x : ℝ) : 
  500 * x * 0.70 = 4 * 2625 → x = 30 :=
by 
  sorry

end cost_of_each_ticket_l208_208671


namespace specific_n_values_l208_208684

theorem specific_n_values (n : ℕ) : 
  ∃ m : ℕ, 
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → m % k = 0) ∧ 
    (m % (n + 1) ≠ 0) ∧ 
    (m % (n + 2) ≠ 0) ∧ 
    (m % (n + 3) ≠ 0) ↔ n = 1 ∨ n = 2 ∨ n = 6 := 
by
  sorry

end specific_n_values_l208_208684


namespace minimum_value_of_4a_plus_b_l208_208288

noncomputable def minimum_value (a b : ℝ) :=
  if a > 0 ∧ b > 0 ∧ a^2 + a*b - 3 = 0 then 4*a + b else 0

theorem minimum_value_of_4a_plus_b :
  ∀ (a b : ℝ), a > 0 → b > 0 → a^2 + a*b - 3 = 0 → 4*a + b ≥ 6 :=
by
  intros a b ha hb hab
  sorry

end minimum_value_of_4a_plus_b_l208_208288


namespace balcony_more_than_orchestra_l208_208714

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 340) 
  (h2 : 12 * x + 8 * y = 3320) : 
  y - x = 40 := 
sorry

end balcony_more_than_orchestra_l208_208714


namespace ralph_squares_count_l208_208031

def total_matchsticks := 50
def elvis_square_sticks := 4
def ralph_square_sticks := 8
def elvis_squares := 5
def leftover_sticks := 6

theorem ralph_squares_count : 
  ∃ R : ℕ, 
  (elvis_squares * elvis_square_sticks) + (R * ralph_square_sticks) + leftover_sticks = total_matchsticks ∧ R = 3 :=
by 
  sorry

end ralph_squares_count_l208_208031


namespace S4k_eq_32_l208_208411

-- Definition of the problem conditions
variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)
variable (k : ℕ)

-- Conditions: Arithmetic sequence sum properties
axiom sum_arithmetic_sequence : ∀ {n : ℕ}, S n = n * (a 1 + a n) / 2

-- Given conditions
axiom Sk_eq_2 : S k = 2
axiom S3k_eq_18 : S (3 * k) = 18

-- Prove the required statement
theorem S4k_eq_32 : S (4 * k) = 32 :=
by
  sorry

end S4k_eq_32_l208_208411


namespace solve_nested_function_l208_208486

def f (x : ℝ) : ℝ := x^2 + 12 * x + 30

theorem solve_nested_function :
  ∃ x : ℝ, f (f (f (f (f x)))) = 0 ↔ (x = -6 + 6^(1/32) ∨ x = -6 - 6^(1/32)) :=
by sorry

end solve_nested_function_l208_208486


namespace abs_sum_inequality_l208_208910

theorem abs_sum_inequality (x : ℝ) : (|x - 2| + |x + 3| < 7) ↔ (-6 < x ∧ x < 3) :=
sorry

end abs_sum_inequality_l208_208910


namespace correct_proposition_D_l208_208054

theorem correct_proposition_D (a b c : ℝ) (h : a > b) : a - c > b - c :=
by
  sorry

end correct_proposition_D_l208_208054


namespace vertical_throw_time_l208_208104

theorem vertical_throw_time (h v g t : ℝ)
  (h_def: h = v * t - (1/2) * g * t^2)
  (initial_v: v = 25)
  (gravity: g = 10)
  (target_h: h = 20) :
  t = 1 ∨ t = 4 := 
by
  sorry

end vertical_throw_time_l208_208104


namespace solve_exponential_eq_l208_208644

theorem solve_exponential_eq (x : ℝ) : 
  ((5 - 2 * x)^(x + 1) = 1) ↔ (x = -1 ∨ x = 2 ∨ x = 3) := by
  sorry

end solve_exponential_eq_l208_208644


namespace find_circle_radius_l208_208927

noncomputable def circle_radius (x y : ℝ) : ℝ :=
  (x - 1) ^ 2 + (y + 2) ^ 2

theorem find_circle_radius :
  (∀ x y : ℝ, 25 * x^2 - 50 * x + 25 * y^2 + 100 * y + 125 = 0 → circle_radius x y = 0) → radius = 0 :=
sorry

end find_circle_radius_l208_208927


namespace like_terms_implies_a_plus_2b_eq_3_l208_208869

theorem like_terms_implies_a_plus_2b_eq_3 (a b : ℤ) (h1 : 2 * a + b = 6) (h2 : a - b = 3) : a + 2 * b = 3 :=
sorry

end like_terms_implies_a_plus_2b_eq_3_l208_208869


namespace isaac_ribbon_length_l208_208173

variable (part_length : ℝ) (total_length : ℝ := part_length * 6) (unused_length : ℝ := part_length * 2)

theorem isaac_ribbon_length
  (total_parts : ℕ := 6)
  (used_parts : ℕ := 4)
  (not_used_parts : ℕ := total_parts - used_parts)
  (not_used_length : Real := 10)
  (equal_parts : total_length / total_parts = part_length) :
  total_length = 30 := by
  sorry

end isaac_ribbon_length_l208_208173


namespace convex_n_hedral_angle_l208_208317

theorem convex_n_hedral_angle (n : ℕ) 
  (sum_plane_angles : ℝ) (sum_dihedral_angles : ℝ) 
  (h1 : sum_plane_angles = sum_dihedral_angles)
  (h2 : sum_plane_angles < 2 * Real.pi)
  (h3 : sum_dihedral_angles > (n - 2) * Real.pi) :
  n = 3 := 
by 
  sorry

end convex_n_hedral_angle_l208_208317


namespace matrix_det_evaluation_l208_208702

noncomputable def matrix_det (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1,   x,     y,     z],
    ![1, x + y,   y,     z],
    ![1,   x, x + y,     z],
    ![1,   x,     y, x + y + z]
  ]

theorem matrix_det_evaluation (x y z : ℝ) :
  matrix_det x y z = y * x * x + y * y * x :=
by sorry

end matrix_det_evaluation_l208_208702


namespace problem_statement_l208_208943

-- We begin by stating the variables x and y with the given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : x - 2 * y = 3
axiom h2 : (x - 2) * (y + 1) = 2

-- The theorem to prove
theorem problem_statement : (x^2 - 2) * (2 * y^2 - 1) = -9 :=
by
  sorry

end problem_statement_l208_208943


namespace solve_eq1_solve_eq2_solve_eq3_solve_eq4_l208_208368

theorem solve_eq1 :
  ∀ x : ℝ, 6 * x - 7 = 4 * x - 5 ↔ x = 1 := by
  intro x
  sorry

theorem solve_eq2 :
  ∀ x : ℝ, 5 * (x + 8) - 5 = 6 * (2 * x - 7) ↔ x = 11 := by
  intro x
  sorry

theorem solve_eq3 :
  ∀ x : ℝ, x - (x - 1) / 2 = 2 - (x + 2) / 5 ↔ x = 11 / 7 := by
  intro x
  sorry

theorem solve_eq4 :
  ∀ x : ℝ, x^2 - 64 = 0 ↔ x = 8 ∨ x = -8 := by
  intro x
  sorry

end solve_eq1_solve_eq2_solve_eq3_solve_eq4_l208_208368


namespace terminal_side_angle_is_in_fourth_quadrant_l208_208882

variable (α : ℝ)
variable (tan_alpha cos_alpha : ℝ)

-- Given conditions
def in_second_quadrant := tan_alpha < 0 ∧ cos_alpha > 0

-- Conclusion to prove
theorem terminal_side_angle_is_in_fourth_quadrant 
  (h : in_second_quadrant tan_alpha cos_alpha) : 
  -- Here we model the "fourth quadrant" in a proof-statement context:
  true := sorry

end terminal_side_angle_is_in_fourth_quadrant_l208_208882


namespace area_parallelogram_proof_l208_208709

/-- We are given a rectangle with a length of 10 cm and a width of 8 cm.
    We transform it into a parallelogram with a height of 9 cm.
    We need to prove that the area of the parallelogram is 72 square centimeters. -/
def area_of_parallelogram_from_rectangle (length width height : ℝ) : ℝ :=
  width * height

theorem area_parallelogram_proof
  (length width height : ℝ)
  (h_length : length = 10)
  (h_width : width = 8)
  (h_height : height = 9) :
  area_of_parallelogram_from_rectangle length width height = 72 :=
by
  sorry

end area_parallelogram_proof_l208_208709


namespace inequality_transitive_l208_208342

theorem inequality_transitive {a b c d : ℝ} (h1 : a > b) (h2 : c > d) (h3 : c ≠ 0) (h4 : d ≠ 0) :
  a + c > b + d :=
by {
  sorry
}

end inequality_transitive_l208_208342


namespace parabola_focus_l208_208141

theorem parabola_focus (F : ℝ × ℝ) :
  (∀ (x y : ℝ), y^2 = 4 * x → (x + 1)^2 + y^2 = ((x - F.1)^2 + (y - F.2)^2)) → 
  F = (1, 0) :=
sorry

end parabola_focus_l208_208141


namespace min_value_expression_l208_208593

theorem min_value_expression (θ φ : ℝ) :
  ∃ (θ φ : ℝ), (3 * Real.cos θ + 6 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 6 * Real.cos φ - 18)^2 = 121 :=
sorry

end min_value_expression_l208_208593


namespace altitude_angle_bisector_inequality_l208_208700

theorem altitude_angle_bisector_inequality
  (h l R r : ℝ) 
  (triangle_condition : ∀ (h l : ℝ) (R r : ℝ), (h > 0 ∧ l > 0 ∧ R > 0 ∧ r > 0)) :
  h / l ≥ Real.sqrt (2 * r / R) :=
by
  sorry

end altitude_angle_bisector_inequality_l208_208700


namespace inequality_addition_l208_208731

theorem inequality_addition (a b : ℝ) (h : a > b) : a + 3 > b + 3 := by
  sorry

end inequality_addition_l208_208731


namespace min_value_ineq_least_3_l208_208027

noncomputable def min_value_ineq (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : ℝ :=
  1 / (x + y) + (x + y) / z

theorem min_value_ineq_least_3 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) :
  min_value_ineq x y z h1 h2 h3 h4 ≥ 3 :=
sorry

end min_value_ineq_least_3_l208_208027


namespace average_age_l208_208067

theorem average_age (women men : ℕ) (avg_age_women avg_age_men : ℝ) 
  (h_women : women = 12) 
  (h_men : men = 18) 
  (h_avg_women : avg_age_women = 28) 
  (h_avg_men : avg_age_men = 40) : 
  (12 * 28 + 18 * 40) / (12 + 18) = 35.2 :=
by {
  sorry
}

end average_age_l208_208067


namespace sum_of_squares_iff_double_sum_of_squares_l208_208886

theorem sum_of_squares_iff_double_sum_of_squares (n : ℕ) :
  (∃ a b : ℤ, n = a^2 + b^2) ↔ (∃ a b : ℤ, 2 * n = a^2 + b^2) :=
sorry

end sum_of_squares_iff_double_sum_of_squares_l208_208886


namespace small_circle_area_l208_208600

theorem small_circle_area (r R : ℝ) (n : ℕ)
  (h_n : n = 6)
  (h_area_large : π * R^2 = 120)
  (h_relation : r = R / 2) :
  π * r^2 = 40 :=
by
  sorry

end small_circle_area_l208_208600


namespace gcd_204_85_l208_208860

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l208_208860


namespace rectangle_area_perimeter_eq_l208_208273

theorem rectangle_area_perimeter_eq (x : ℝ) (h : 4 * x * (x + 4) = 2 * 4 * x + 2 * (x + 4)) : x = 1 / 2 :=
sorry

end rectangle_area_perimeter_eq_l208_208273


namespace chocolate_eggs_total_weight_l208_208247

def total_weight_after_discarding_box_b : ℕ :=
  let weight_large := 14
  let weight_medium := 10
  let weight_small := 6
  let box_A_weight := 4 * weight_large + 2 * weight_medium
  let box_B_weight := 6 * weight_small + 2 * weight_large
  let box_C_weight := 4 * weight_large + 3 * weight_medium
  let box_D_weight := 4 * weight_medium + 4 * weight_small
  let box_E_weight := 4 * weight_small + 2 * weight_medium
  box_A_weight + box_C_weight + box_D_weight + box_E_weight

theorem chocolate_eggs_total_weight : total_weight_after_discarding_box_b = 270 := by
  sorry

end chocolate_eggs_total_weight_l208_208247


namespace number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l208_208035

-- Define the conditions
def total_matches := 16
def played_matches := 9
def lost_matches := 2
def current_points := 19
def max_points_per_win := 3
def draw_points := 1
def remaining_matches := total_matches - played_matches
def required_points := 34

-- Statements to prove
theorem number_of_wins_in_first_9_matches :
  ∃ wins_in_first_9, 3 * wins_in_first_9 + draw_points * (played_matches - lost_matches - wins_in_first_9) = current_points :=
sorry

theorem highest_possible_points :
  current_points + remaining_matches * max_points_per_win = 40 :=
sorry

theorem minimum_wins_in_remaining_matches :
  ∃ min_wins_in_remaining_7, (min_wins_in_remaining_7 = 4 ∧ 3 * min_wins_in_remaining_7 + current_points + (remaining_matches - min_wins_in_remaining_7) * draw_points ≥ required_points) :=
sorry

end number_of_wins_in_first_9_matches_highest_possible_points_minimum_wins_in_remaining_matches_l208_208035


namespace total_money_l208_208790

theorem total_money (m c : ℝ) (hm : m = 5 / 8) (hc : c = 7 / 20) : m + c = 0.975 := sorry

end total_money_l208_208790


namespace square_area_inscribed_in_parabola_l208_208599

-- Declare the parabola equation
def parabola (x : ℝ) : ℝ := x^2 - 10 * x + 20

-- Declare the condition that we have a square inscribed to this parabola.
def is_inscribed_square (side_length : ℝ) : Prop :=
∀ (x : ℝ), (x = 5 - side_length/2 ∨ x = 5 + side_length/2) → (parabola x = 0)

-- Proof goal
theorem square_area_inscribed_in_parabola : ∃ (side_length : ℝ), is_inscribed_square side_length ∧ side_length^2 = 400 :=
by
  sorry

end square_area_inscribed_in_parabola_l208_208599


namespace flight_time_is_approximately_50_hours_l208_208643

noncomputable def flightTime (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  circumference / speed

theorem flight_time_is_approximately_50_hours :
  let radius := 4200
  let speed := 525
  abs (flightTime radius speed - 50) < 1 :=
by
  sorry

end flight_time_is_approximately_50_hours_l208_208643


namespace triangle_sine_value_l208_208473

-- Define the triangle sides and angles
variables {a b c A B C : ℝ}

-- Main theorem stating the proof problem
theorem triangle_sine_value (h : a^2 = b^2 + c^2 - bc) :
  (a * Real.sin B) / b = Real.sqrt 3 / 2 := sorry

end triangle_sine_value_l208_208473


namespace barkley_total_net_buried_bones_l208_208263

def monthly_bones_received (months : ℕ) : (ℕ × ℕ × ℕ) := (10 * months, 6 * months, 4 * months)

def burying_pattern_A (months : ℕ) : ℕ := 6 * months
def eating_pattern_A (months : ℕ) : ℕ := if months > 2 then 3 else 1

def burying_pattern_B (months : ℕ) : ℕ := if months = 5 then 0 else 4 * (months - 1)
def eating_pattern_B (months : ℕ) : ℕ := 2

def burying_pattern_C (months : ℕ) : ℕ := 2 * months
def eating_pattern_C (months : ℕ) : ℕ := 2

def total_net_buried_bones (months : ℕ) : ℕ :=
  let (received_A, received_B, received_C) := monthly_bones_received months
  let net_A := burying_pattern_A months - eating_pattern_A months
  let net_B := burying_pattern_B months - eating_pattern_B months
  let net_C := burying_pattern_C months - eating_pattern_C months
  net_A + net_B + net_C

theorem barkley_total_net_buried_bones : total_net_buried_bones 5 = 49 := by
  sorry

end barkley_total_net_buried_bones_l208_208263


namespace rectangle_area_l208_208875

theorem rectangle_area (p q : ℝ) (x : ℝ) (h1 : x^2 + (2 * x)^2 = (p + q)^2) : 
    2 * x^2 = (2 * (p + q)^2) / 5 := 
sorry

end rectangle_area_l208_208875


namespace union_of_sets_l208_208908

def M : Set ℝ := {x | x^2 + 2 * x = 0}

def N : Set ℝ := {x | x^2 - 2 * x = 0}

theorem union_of_sets : M ∪ N = {x | x = -2 ∨ x = 0 ∨ x = 2} := sorry

end union_of_sets_l208_208908


namespace problem_statement_l208_208318

noncomputable def m (α : ℝ) : ℝ := - (Real.sqrt 2) / 4

noncomputable def tan_alpha (α : ℝ) : ℝ := 2 * Real.sqrt 2

theorem problem_statement (α : ℝ) (P : (ℝ × ℝ)) (h1 : P = (m α, 1)) (h2 : Real.cos α = - 1 / 3) :
  (P.1 = - (Real.sqrt 2) / 4) ∧ (Real.tan α = 2 * Real.sqrt 2) :=
by
  sorry

end problem_statement_l208_208318


namespace problem_l208_208103

theorem problem (a : ℤ) (n : ℕ) : (a + 1) ^ (2 * n + 1) + a ^ (n + 2) ∣ a ^ 2 + a + 1 :=
sorry

end problem_l208_208103


namespace real_solutions_in_interval_l208_208030

noncomputable def problem_statement (x : ℝ) : Prop :=
  (x + 1 > 0) ∧ 
  (x ≠ -1) ∧
  (x^2 / (x + 1 - Real.sqrt (x + 1))^2 < (x^2 + 3 * x + 18) / (x + 1)^2)
  
theorem real_solutions_in_interval (x : ℝ) (h : problem_statement x) : -1 < x ∧ x < 3 :=
sorry

end real_solutions_in_interval_l208_208030


namespace parallel_lines_condition_l208_208198

theorem parallel_lines_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * m * x + y + 6 = 0 → (m - 3) * x - y + 7 = 0) → m = 1 :=
by
  sorry

end parallel_lines_condition_l208_208198


namespace simplify_expression_l208_208021

theorem simplify_expression :
  18 * (14 / 15) * (1 / 12) - (1 / 5) = 1 / 2 :=
by
  sorry

end simplify_expression_l208_208021


namespace simplify_expression_l208_208633

variable {R : Type} [AddCommGroup R] [Module ℤ R]

theorem simplify_expression (a b : R) :
  (25 • a + 70 • b) + (15 • a + 34 • b) - (12 • a + 55 • b) = 28 • a + 49 • b :=
by sorry

end simplify_expression_l208_208633


namespace system_solution_l208_208547

theorem system_solution (x y : ℝ) (h1 : 4 * x - y = 3) (h2 : x + 6 * y = 17) : x + y = 4 :=
by
  sorry

end system_solution_l208_208547


namespace sqrt_factorial_sq_l208_208129

theorem sqrt_factorial_sq : ((Real.sqrt (Nat.factorial 5 * Nat.factorial 4)) ^ 2) = 2880 := by
  sorry

end sqrt_factorial_sq_l208_208129


namespace sequence_polynomial_l208_208938

theorem sequence_polynomial (f : ℕ → ℤ) :
  (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 21 ∧ f 3 = 51) ↔ (∀ n, f n = n^3 + 2 * n^2 + n + 3) :=
by
  sorry

end sequence_polynomial_l208_208938


namespace negation_of_universal_l208_208946

theorem negation_of_universal {x : ℝ} : ¬ (∀ x > 0, x^2 - x ≤ 0) ↔ ∃ x > 0, x^2 - x > 0 :=
by
  sorry

end negation_of_universal_l208_208946


namespace probability_of_green_l208_208591

theorem probability_of_green : 
  ∀ (P_red P_orange P_yellow P_green : ℝ), 
    P_red = 0.25 → P_orange = 0.35 → P_yellow = 0.1 → 
    P_red + P_orange + P_yellow + P_green = 1 →
    P_green = 0.3 :=
by
  intros P_red P_orange P_yellow P_green h_red h_orange h_yellow h_total
  sorry

end probability_of_green_l208_208591


namespace mike_corvette_average_speed_l208_208690

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end mike_corvette_average_speed_l208_208690


namespace complex_number_identity_l208_208201

theorem complex_number_identity (a b : ℝ) (i : ℂ) (h : (a + i) * (1 + i) = b * i) : a + b * i = 1 + 2 * i := 
by
  sorry

end complex_number_identity_l208_208201


namespace Thabo_owns_25_hardcover_nonfiction_books_l208_208401

variable (H P F : ℕ)

-- Conditions
def condition1 := P = H + 20
def condition2 := F = 2 * P
def condition3 := H + P + F = 160

-- Goal
theorem Thabo_owns_25_hardcover_nonfiction_books (H P F : ℕ) (h1 : condition1 H P) (h2 : condition2 P F) (h3 : condition3 H P F) : H = 25 :=
by
  sorry

end Thabo_owns_25_hardcover_nonfiction_books_l208_208401


namespace correct_conclusions_l208_208363

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem correct_conclusions :
  (∃ (a b : ℝ), a < b ∧ f a < f b ∧ ∀ x, a < x ∧ x < b → f x < f (x+1)) ∧
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = (x₁ - 2012) ∧ f x₂ = (x₂ - 2012)) :=
by
  sorry

end correct_conclusions_l208_208363


namespace find_r_s_l208_208800

def is_orthogonal (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1 * v₂.1 + v₁.2.1 * v₂.2.1 + v₁.2.2 * v₂.2.2 = 0

def have_equal_magnitudes (v₁ v₂ : ℝ × ℝ × ℝ) : Prop :=
  v₁.1^2 + v₁.2.1^2 + v₁.2.2^2 = v₂.1^2 + v₂.2.1^2 + v₂.2.2^2

theorem find_r_s (r s : ℝ) :
  is_orthogonal (4, r, -2) (-1, 2, s) ∧
  have_equal_magnitudes (4, r, -2) (-1, 2, s) →
  r = -11 / 4 ∧ s = -19 / 4 :=
by
  intro h
  sorry

end find_r_s_l208_208800


namespace cubic_inequality_solution_l208_208587

theorem cubic_inequality_solution :
  ∀ x : ℝ, (x + 1) * (x + 2)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ -1 := 
by 
  sorry

end cubic_inequality_solution_l208_208587


namespace distance_between_foci_l208_208742

theorem distance_between_foci :
  let x := ℝ
  let y := ℝ
  ∀ (x y : ℝ), 9*x^2 + 36*x + 4*y^2 - 8*y + 1 = 0 →
  ∃ (d : ℝ), d = (Real.sqrt 351) / 3 :=
sorry

end distance_between_foci_l208_208742


namespace circle_form_eq_standard_form_l208_208891

theorem circle_form_eq_standard_form :
  ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 6 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 11 := 
by
  intro x y
  sorry

end circle_form_eq_standard_form_l208_208891


namespace remaining_sugar_l208_208770

-- Conditions as definitions
def total_sugar : ℝ := 9.8
def spilled_sugar : ℝ := 5.2

-- Theorem to prove the remaining sugar
theorem remaining_sugar : total_sugar - spilled_sugar = 4.6 := by
  sorry

end remaining_sugar_l208_208770


namespace jordan_field_area_l208_208977

theorem jordan_field_area
  (s l : ℕ)
  (h1 : 2 * (s + l) = 24)
  (h2 : l + 1 = 2 * (s + 1)) :
  3 * s * 3 * l = 189 := 
by
  sorry

end jordan_field_area_l208_208977


namespace volume_of_sphere_l208_208680

theorem volume_of_sphere
  (r : ℝ) (V : ℝ)
  (h₁ : r = 1/3)
  (h₂ : 2 * r = (16/9 * V)^(1/3)) :
  V = 1/6 :=
  sorry

end volume_of_sphere_l208_208680


namespace correlation_highly_related_l208_208370

-- Conditions:
-- Let corr be the linear correlation coefficient of product output and unit cost.
-- Let rel be the relationship between product output and unit cost.

def corr : ℝ := -0.87

-- Proof Goal:
-- If corr = -0.87, then the relationship is "highly related".

theorem correlation_highly_related (h : corr = -0.87) : rel = "highly related" := by
  sorry

end correlation_highly_related_l208_208370


namespace total_marbles_correct_l208_208102

variable (r : ℝ) -- number of red marbles
variable (b : ℝ) -- number of blue marbles
variable (g : ℝ) -- number of green marbles

-- Conditions
def red_blue_ratio : Prop := r = 1.5 * b
def green_red_ratio : Prop := g = 1.8 * r

-- Total number of marbles
def total_marbles (r b g : ℝ) : ℝ := r + b + g

theorem total_marbles_correct (r b g : ℝ) (h1 : red_blue_ratio r b) (h2 : green_red_ratio r g) : 
  total_marbles r b g = 3.467 * r :=
by 
  sorry

end total_marbles_correct_l208_208102


namespace isosceles_triangle_interior_angles_l208_208925

theorem isosceles_triangle_interior_angles (a b c : ℝ) 
  (h1 : b = c) (h2 : a + b + c = 180) (exterior : a + 40 = 180 ∨ b + 40 = 140) :
  (a = 40 ∧ b = 70 ∧ c = 70) ∨ (a = 100 ∧ b = 40 ∧ c = 40) :=
by
  sorry

end isosceles_triangle_interior_angles_l208_208925


namespace jebb_take_home_pay_is_4620_l208_208791

noncomputable def gross_salary : ℤ := 6500
noncomputable def federal_tax (income : ℤ) : ℤ :=
  let tax1 := min income 2000 * 10 / 100
  let tax2 := min (max (income - 2000) 0) 2000 * 15 / 100
  let tax3 := max (income - 4000) 0 * 25 / 100
  tax1 + tax2 + tax3

noncomputable def health_insurance : ℤ := 300
noncomputable def retirement_contribution (income : ℤ) : ℤ := income * 7 / 100

noncomputable def total_deductions (income : ℤ) : ℤ :=
  federal_tax income + health_insurance + retirement_contribution income

noncomputable def take_home_pay (income : ℤ) : ℤ :=
  income - total_deductions income

theorem jebb_take_home_pay_is_4620 : take_home_pay gross_salary = 4620 := by
  sorry

end jebb_take_home_pay_is_4620_l208_208791


namespace ladder_length_l208_208769

/-- The length of the ladder leaning against a wall when it forms
    a 60 degree angle with the ground and the foot of the ladder 
    is 9.493063650744542 m from the wall is 18.986127301489084 m. -/
theorem ladder_length (L : ℝ) (adjacent : ℝ) (θ : ℝ) (cosθ : ℝ) :
  θ = Real.pi / 3 ∧ adjacent = 9.493063650744542 ∧ cosθ = Real.cos θ →
  L = 18.986127301489084 :=
by
  intro h
  sorry

end ladder_length_l208_208769


namespace functional_equation_solution_l208_208588

theorem functional_equation_solution (f : ℝ → ℝ)
  (h : ∀ x, 2 * f (f x) = (x^2 - x) * f x + 4 - 2 * x) :
  f 2 = 2 ∧ (f 1 = 1 ∨ f 1 = 4) :=
sorry

end functional_equation_solution_l208_208588


namespace expression_equals_66069_l208_208174

-- Definitions based on the conditions
def numerator : Nat := 1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10
def denominator : Nat := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10
def expression : Rat := numerator / denominator

-- The main theorem to be proven
theorem expression_equals_66069 : expression = 66069 := by
  sorry

end expression_equals_66069_l208_208174


namespace coin_probability_l208_208743

theorem coin_probability (a r : ℝ) (h : r < a / 2) :
  let favorable_cells := 3
  let larger_cell_area := 9 * a^2
  let favorable_area_per_cell := (a - 2 * r)^2
  let favorable_area := favorable_cells * favorable_area_per_cell
  let probability := favorable_area / larger_cell_area
  probability = (a - 2 * r)^2 / (3 * a^2) :=
by
  sorry

end coin_probability_l208_208743


namespace min_value_expression_l208_208808

theorem min_value_expression (a : ℝ) (h : a > 2) : a + 4 / (a - 2) ≥ 6 :=
by
  sorry

end min_value_expression_l208_208808


namespace total_fencing_cost_is_correct_l208_208320

-- Defining the lengths of each side
def length1 : ℝ := 50
def length2 : ℝ := 75
def length3 : ℝ := 60
def length4 : ℝ := 80
def length5 : ℝ := 65

-- Defining the cost per unit length for each side
def cost_per_meter1 : ℝ := 2
def cost_per_meter2 : ℝ := 3
def cost_per_meter3 : ℝ := 4
def cost_per_meter4 : ℝ := 3.5
def cost_per_meter5 : ℝ := 5

-- Calculating the total cost for each side
def cost1 : ℝ := length1 * cost_per_meter1
def cost2 : ℝ := length2 * cost_per_meter2
def cost3 : ℝ := length3 * cost_per_meter3
def cost4 : ℝ := length4 * cost_per_meter4
def cost5 : ℝ := length5 * cost_per_meter5

-- Summing up the total cost for all sides
def total_cost : ℝ := cost1 + cost2 + cost3 + cost4 + cost5

-- The theorem to be proven
theorem total_fencing_cost_is_correct : total_cost = 1170 := by
  sorry

end total_fencing_cost_is_correct_l208_208320


namespace difference_between_perfect_and_cracked_l208_208559

def total_eggs : ℕ := 24
def broken_eggs : ℕ := 3
def cracked_eggs : ℕ := 2 * broken_eggs

def perfect_eggs : ℕ := total_eggs - broken_eggs - cracked_eggs
def difference : ℕ := perfect_eggs - cracked_eggs

theorem difference_between_perfect_and_cracked :
  difference = 9 := by
  sorry

end difference_between_perfect_and_cracked_l208_208559


namespace find_a_l208_208155

theorem find_a (a : ℝ) :
  (∃ x : ℝ, (a + 1) * x^2 - x + a^2 - 2*a - 2 = 0 ∧ x = 1) → a = 2 :=
by
  sorry

end find_a_l208_208155


namespace root_line_discriminant_curve_intersection_l208_208367

theorem root_line_discriminant_curve_intersection (a p q : ℝ) :
  (4 * p^3 + 27 * q^2 = 0) ∧ (ap + q + a^3 = 0) →
  (a = 0 ∧ ∀ p q, 4 * p^3 + 27 * q^2 = 0 → ap + q + a^3 = 0 → (p = 0 ∧ q = 0)) ∨
  (a ≠ 0 ∧ (∃ p1 q1 p2 q2, 
             4 * p1^3 + 27 * q1^2 = 0 ∧ ap + q1 + a^3 = 0 ∧ 
             4 * p2^3 + 27 * q2^2 = 0 ∧ ap + q2 + a^3 = 0 ∧ 
             (p1, q1) ≠ (p2, q2))) := 
sorry

end root_line_discriminant_curve_intersection_l208_208367


namespace charity_amount_l208_208541

theorem charity_amount (total : ℝ) (charities : ℕ) (amount_per_charity : ℝ) 
  (h1 : total = 3109) (h2 : charities = 25) : 
  amount_per_charity = 124.36 :=
by
  sorry

end charity_amount_l208_208541


namespace bobby_finishes_candies_in_weeks_l208_208899

def total_candies (packets: Nat) (candies_per_packet: Nat) : Nat := packets * candies_per_packet

def candies_eaten_per_week (candies_per_day_mon_fri: Nat) (days_mon_fri: Nat) (candies_per_day_weekend: Nat) (days_weekend: Nat) : Nat :=
  (candies_per_day_mon_fri * days_mon_fri) + (candies_per_day_weekend * days_weekend)

theorem bobby_finishes_candies_in_weeks :
  let packets := 2
  let candies_per_packet := 18
  let candies_per_day_mon_fri := 2
  let days_mon_fri := 5
  let candies_per_day_weekend := 1
  let days_weekend := 2

  total_candies packets candies_per_packet / candies_eaten_per_week candies_per_day_mon_fri days_mon_fri candies_per_day_weekend days_weekend = 3 :=
by
  sorry

end bobby_finishes_candies_in_weeks_l208_208899


namespace efficiency_ratio_l208_208939

theorem efficiency_ratio (A B : ℝ) (h1 : A ≠ B)
  (h2 : A + B = 1 / 7)
  (h3 : B = 1 / 21) :
  A / B = 2 :=
by
  sorry

end efficiency_ratio_l208_208939


namespace same_bill_at_300_minutes_l208_208271

def monthlyBillA (x : ℕ) : ℝ := 15 + 0.1 * x
def monthlyBillB (x : ℕ) : ℝ := 0.15 * x

theorem same_bill_at_300_minutes : monthlyBillA 300 = monthlyBillB 300 := 
by
  sorry

end same_bill_at_300_minutes_l208_208271


namespace value_of_expression_l208_208703

theorem value_of_expression (x : ℝ) (hx : 23 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 5 :=
by
  sorry

end value_of_expression_l208_208703


namespace intersection_A_B_l208_208325

open Set Real

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {x | abs (x - 2) ≥ 1}
def answer : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_A_B :
  A ∩ B = answer :=
sorry

end intersection_A_B_l208_208325


namespace abc_divisibility_l208_208436

theorem abc_divisibility (a b c : Nat) (h1 : a^3 ∣ b) (h2 : b^3 ∣ c) (h3 : c^3 ∣ a) :
  ∃ k : Nat, (a + b + c)^13 = k * a * b * c :=
by
  sorry

end abc_divisibility_l208_208436


namespace player_weekly_earnings_l208_208911

structure Performance :=
  (points assists rebounds steals : ℕ)

def base_pay (avg_points : ℕ) : ℕ :=
  if avg_points >= 30 then 10000 else 8000

def assists_bonus (total_assists : ℕ) : ℕ :=
  if total_assists >= 20 then 5000
  else if total_assists >= 10 then 3000
  else 1000

def rebounds_bonus (total_rebounds : ℕ) : ℕ :=
  if total_rebounds >= 40 then 5000
  else if total_rebounds >= 20 then 3000
  else 1000

def steals_bonus (total_steals : ℕ) : ℕ :=
  if total_steals >= 15 then 5000
  else if total_steals >= 5 then 3000
  else 1000

def total_payment (performances : List Performance) : ℕ :=
  let total_points := performances.foldl (λ acc p => acc + p.points) 0
  let total_assists := performances.foldl (λ acc p => acc + p.assists) 0
  let total_rebounds := performances.foldl (λ acc p => acc + p.rebounds) 0
  let total_steals := performances.foldl (λ acc p => acc + p.steals) 0
  let avg_points := total_points / performances.length
  base_pay avg_points + assists_bonus total_assists + rebounds_bonus total_rebounds + steals_bonus total_steals
  
theorem player_weekly_earnings :
  let performances := [
    Performance.mk 30 5 7 3,
    Performance.mk 28 6 5 2,
    Performance.mk 32 4 9 1,
    Performance.mk 34 3 11 2,
    Performance.mk 26 2 8 3
  ]
  total_payment performances = 23000 := by 
    sorry

end player_weekly_earnings_l208_208911


namespace largest_divisor_n_l208_208257

theorem largest_divisor_n (n : ℕ) (h₁ : n > 0) (h₂ : 650 ∣ n^3) : 130 ∣ n :=
sorry

end largest_divisor_n_l208_208257


namespace problem_inequality_l208_208691

variable {a b c d : ℝ}

theorem problem_inequality (h1 : 0 ≤ a) (h2 : 0 ≤ d) (h3 : 0 < b) (h4 : 0 < c) (h5 : b + c ≥ a + d) :
  (b / (c + d)) + (c / (b + a)) ≥ (Real.sqrt 2) - (1 / 2) := 
sorry

end problem_inequality_l208_208691


namespace team_air_conditioner_installation_l208_208683

theorem team_air_conditioner_installation (x : ℕ) (y : ℕ) 
  (h1 : 66 % x = 0) 
  (h2 : 60 % y = 0) 
  (h3 : x = y + 2) 
  (h4 : 66 / x = 60 / y) 
  : x = 22 ∧ y = 20 :=
by
  have h5 : x = 22 := sorry
  have h6 : y = 20 := sorry
  exact ⟨h5, h6⟩

end team_air_conditioner_installation_l208_208683


namespace solve_trig_eq_l208_208227

open Real

theorem solve_trig_eq (x a : ℝ) (hx1 : 0 < x) (hx2 : x < 2 * π) (ha : a > 0) :
    (sin (3 * x) + a * sin (2 * x) + 2 * sin x = 0) →
    (0 < a ∧ a < 2 → x = 0 ∨ x = π) ∧ 
    (a > 5 / 2 → ∃ α, (x = α ∨ x = 2 * π - α)) :=
by sorry

end solve_trig_eq_l208_208227


namespace value_of_x7_plus_64x2_l208_208956

-- Let x be a real number such that x^3 + 4x = 8.
def x_condition (x : ℝ) : Prop := x^3 + 4 * x = 8

-- We need to determine the value of x^7 + 64x^2.
theorem value_of_x7_plus_64x2 (x : ℝ) (h : x_condition x) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end value_of_x7_plus_64x2_l208_208956


namespace determine_A_value_l208_208596

noncomputable def solve_for_A (A B C : ℝ) : Prop :=
  (A = 1/16) ↔ 
  (∀ x : ℝ, (1 / ((x + 5) * (x - 3) * (x + 3))) = (A / (x + 5)) + (B / (x - 3)) + (C / (x + 3)))

theorem determine_A_value :
  solve_for_A (1/16) B C :=
by
  sorry

end determine_A_value_l208_208596


namespace find_a1_and_d_l208_208321

variable (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) (a5 : ℤ := -1) (a8 : ℤ := 2)

def arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem find_a1_and_d
  (h : arithmetic_sequence a d)
  (h_a5 : a 5 = -1)
  (h_a8 : a 8 = 2) :
  a 1 = -5 ∧ d = 1 :=
by
  sorry

end find_a1_and_d_l208_208321


namespace possible_degrees_of_remainder_l208_208327

theorem possible_degrees_of_remainder (p : Polynomial ℝ) :
  ∃ r q : Polynomial ℝ, p = q * (3 * X^3 - 4 * X^2 + 5 * X - 6) + r ∧ r.degree < 3 :=
sorry

end possible_degrees_of_remainder_l208_208327


namespace Angela_is_295_cm_l208_208419

noncomputable def Angela_height (Carl_height : ℕ) : ℕ :=
  let Becky_height := 2 * Carl_height
  let Amy_height := Becky_height + Becky_height / 5  -- 20% taller than Becky
  let Helen_height := Amy_height + 3
  let Angela_height := Helen_height + 4
  Angela_height

theorem Angela_is_295_cm : Angela_height 120 = 295 := 
by 
  sorry

end Angela_is_295_cm_l208_208419


namespace tangent_line_to_circle_l208_208558

theorem tangent_line_to_circle {c : ℝ} (h : c > 0) :
  (∀ x y : ℝ, x^2 + y^2 = 8 → x + y = c) ↔ c = 4 := sorry

end tangent_line_to_circle_l208_208558


namespace sri_lanka_population_problem_l208_208045

theorem sri_lanka_population_problem
  (P : ℝ)
  (h1 : 0.85 * (0.9 * P) = 3213) :
  P = 4200 :=
sorry

end sri_lanka_population_problem_l208_208045


namespace mr_brown_final_price_is_correct_l208_208574

noncomputable def mr_brown_final_purchase_price :
  Float :=
  let initial_price : Float := 100000
  let mr_brown_price  := initial_price * 1.12
  let improvement := mr_brown_price * 0.05
  let mr_brown_total_investment := mr_brown_price + improvement
  let mr_green_purchase_price := mr_brown_total_investment * 1.04
  let market_decline := mr_green_purchase_price * 0.03
  let value_after_decline := mr_green_purchase_price - market_decline
  let loss := value_after_decline * 0.10
  let ms_white_purchase_price := value_after_decline - loss
  let market_increase := ms_white_purchase_price * 0.08
  let value_after_increase := ms_white_purchase_price + market_increase
  let profit := value_after_increase * 0.05
  let final_price := value_after_increase + profit
  final_price

theorem mr_brown_final_price_is_correct :
  mr_brown_final_purchase_price = 121078.76 := by
  sorry

end mr_brown_final_price_is_correct_l208_208574


namespace calc_expression_eq_3_solve_quadratic_eq_l208_208290

-- Problem 1
theorem calc_expression_eq_3 :
  (-1 : ℝ) ^ 2020 + (- (1 / 2)⁻¹) - (3.14 - Real.pi) ^ 0 + abs (-3) = 3 :=
by
  sorry

-- Problem 2
theorem solve_quadratic_eq {x : ℝ} :
  (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2 / 3) :=
by
  sorry

end calc_expression_eq_3_solve_quadratic_eq_l208_208290


namespace T_bisects_broken_line_l208_208798

def midpoint_arc {α : Type*} [LinearOrderedField α] (A B C : α) : α := (A + B + C) / 2
def projection_perpendicular {α : Type*} [LinearOrderedField α] (F A B C : α) : α := sorry -- Define perpendicular projection T

theorem T_bisects_broken_line {α : Type*} [LinearOrderedField α]
  (A B C : α) (F := midpoint_arc A B C) (T := projection_perpendicular F A B C) :
  T = (A + B + C) / 2 :=
sorry

end T_bisects_broken_line_l208_208798


namespace fraction_distance_traveled_by_bus_l208_208922

theorem fraction_distance_traveled_by_bus (D : ℝ) (hD : D = 105.00000000000003)
    (distance_by_foot : ℝ) (h_foot : distance_by_foot = (1 / 5) * D)
    (distance_by_car : ℝ) (h_car : distance_by_car = 14) :
    (D - (distance_by_foot + distance_by_car)) / D = 2 / 3 := by
  sorry

end fraction_distance_traveled_by_bus_l208_208922


namespace systematic_sampling_missiles_l208_208417

theorem systematic_sampling_missiles (S : Set ℕ) (hS : S = {n | 1 ≤ n ∧ n ≤ 50}) :
  (∃ seq : Fin 5 → ℕ, (∀ i : Fin 4, seq (Fin.succ i) - seq i = 10) ∧ seq 0 = 3)
  → (∃ seq : Fin 5 → ℕ, seq = ![3, 13, 23, 33, 43]) :=
by
  sorry

end systematic_sampling_missiles_l208_208417


namespace no_such_integers_exist_l208_208182

theorem no_such_integers_exist (x y z : ℤ) (hx : x ≠ 0) :
  ¬ (2 * x ^ 4 + 2 * x ^ 2 * y ^ 2 + y ^ 4 = z ^ 2) :=
by
  sorry

end no_such_integers_exist_l208_208182


namespace cos_double_angle_l208_208785

-- Define the hypothesis
def cos_alpha (α : ℝ) : Prop := Real.cos α = 1 / 2

-- State the theorem
theorem cos_double_angle (α : ℝ) (h : cos_alpha α) : Real.cos (2 * α) = -1 / 2 := by
  sorry

end cos_double_angle_l208_208785


namespace min_omega_value_l208_208627

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) := 2 * Real.sin (ω * x + φ)

theorem min_omega_value
  (ω : ℝ) (φ : ℝ)
  (hω : ω > 0)
  (h1 : f (π / 3) ω φ = 0)
  (h2 : f (π / 2) ω φ = 2) :
  ω = 3 :=
sorry

end min_omega_value_l208_208627


namespace num_of_nickels_l208_208687

theorem num_of_nickels (x : ℕ) (hx_eq_dimes : ∀ n, n = x → n = x) (hx_eq_quarters : ∀ n, n = x → n = 2 * x) (total_value : 5 * x + 10 * x + 50 * x = 1950) : x = 30 :=
sorry

end num_of_nickels_l208_208687


namespace expression_simplification_l208_208662

theorem expression_simplification :
  (- (1 / 2)) ^ 2023 * 2 ^ 2024 = -2 :=
by
  sorry

end expression_simplification_l208_208662


namespace ratio_students_sent_home_to_remaining_l208_208443

theorem ratio_students_sent_home_to_remaining (total_students : ℕ) (students_taken_to_beach : ℕ)
    (students_still_in_school : ℕ) (students_sent_home : ℕ) 
    (h1 : total_students = 1000) (h2 : students_taken_to_beach = total_students / 2)
    (h3 : students_still_in_school = 250) 
    (h4 : students_sent_home = total_students / 2 - students_still_in_school) :
    (students_sent_home / students_still_in_school) = 1 := 
by
    sorry

end ratio_students_sent_home_to_remaining_l208_208443


namespace find_coordinates_of_P_l208_208460

-- Define points N and M with given symmetries.
structure Point where
  x : ℝ
  y : ℝ

def symmetric_about_x (P1 P2 : Point) : Prop :=
  P1.x = P2.x ∧ P1.y = -P2.y

def symmetric_about_y (P1 P2 : Point) : Prop :=
  P1.x = -P2.x ∧ P1.y = P2.y

-- Given conditions
def N : Point := ⟨1, 2⟩
def M : Point := ⟨-1, 2⟩ -- derived from symmetry about y-axis with N
def P : Point := ⟨-1, -2⟩ -- derived from symmetry about x-axis with M

theorem find_coordinates_of_P :
  symmetric_about_x M P ∧ symmetric_about_y N M → P = ⟨-1, -2⟩ :=
by
  sorry

end find_coordinates_of_P_l208_208460


namespace prime_in_A_l208_208096

def is_in_A (n : ℕ) : Prop :=
  ∃ a b : ℤ, n = a^2 + 2 * b^2 ∧ b ≠ 0

theorem prime_in_A (p : ℕ) (hp : Nat.Prime p) (h : is_in_A (p^2)) : is_in_A p :=
by
  sorry

end prime_in_A_l208_208096


namespace number_of_teams_l208_208055

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
sorry

end number_of_teams_l208_208055


namespace triple_layer_area_l208_208294

theorem triple_layer_area (A B C X Y : ℕ) 
  (h1 : A + B + C = 204) 
  (h2 : 140 = (A + B + C) - X - 2 * Y + X + Y)
  (h3 : X = 24) : 
  Y = 64 := by
  sorry

end triple_layer_area_l208_208294


namespace arrangement_meeting_ways_l208_208353

-- For convenience, define the number of members per school and the combination function.
def num_members_per_school : ℕ := 6
def num_schools : ℕ :=  4
def combination (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem arrangement_meeting_ways : 
  let host_ways := num_schools
  let host_reps_ways := combination num_members_per_school 2
  let non_host_schools := num_schools - 1
  let non_host_reps_ways := combination num_members_per_school 2
  let total_non_host_reps_ways := non_host_reps_ways ^ non_host_schools
  let total_ways := host_ways * host_reps_ways * total_non_host_reps_ways
  total_ways = 202500 :=
by 
  -- Definitions and computation is deferred to the steps,
  -- which are to be filled during the proof.
  sorry

end arrangement_meeting_ways_l208_208353


namespace john_small_planks_l208_208894

theorem john_small_planks (L S : ℕ) (h1 : L = 12) (h2 : L + S = 29) : S = 17 :=
by {
  sorry
}

end john_small_planks_l208_208894


namespace mutually_exclusive_event_of_hitting_target_at_least_once_l208_208261

-- Definitions from conditions
def two_shots_fired : Prop := true

def complementary_events (E F : Prop) : Prop :=
  E ∨ F ∧ ¬(E ∧ F)

def hitting_target_at_least_once : Prop := true -- Placeholder for the event of hitting at least one target
def both_shots_miss : Prop := true              -- Placeholder for the event that both shots miss

-- Statement to prove
theorem mutually_exclusive_event_of_hitting_target_at_least_once
  (h1 : two_shots_fired)
  (h2 : complementary_events hitting_target_at_least_once both_shots_miss) :
  hitting_target_at_least_once = ¬both_shots_miss := 
sorry

end mutually_exclusive_event_of_hitting_target_at_least_once_l208_208261


namespace max_tulips_l208_208195

theorem max_tulips (y r : ℕ) (h1 : (y + r) % 2 = 1) (h2 : r = y + 1 ∨ y = r + 1) (h3 : 50 * y + 31 * r ≤ 600) : y + r = 15 :=
by
  sorry

end max_tulips_l208_208195


namespace probability_six_distinct_numbers_l208_208992

theorem probability_six_distinct_numbers :
  let outcomes := 6^7
  let favorable_outcomes := 1 * 6 * (Nat.choose 7 2) * (Nat.factorial 5)
  let probability := favorable_outcomes / outcomes
  probability = (35 / 648) := 
by
  let outcomes := 6^7
  let favorable_outcomes := 1 * 6 * (Nat.choose 7 2) * (Nat.factorial 5)
  let probability := favorable_outcomes / outcomes
  have h : favorable_outcomes = 15120 := by sorry
  have h2 : outcomes = 279936 := by sorry
  have prob : probability = (15120 / 279936) := by sorry
  have gcd_calc : gcd 15120 279936 = 432 := by sorry
  have simplified_prob : (15120 / 279936) = (35 / 648) := by sorry
  exact simplified_prob

end probability_six_distinct_numbers_l208_208992


namespace child_sold_apples_correct_l208_208906

-- Definitions based on conditions
def initial_apples (children : ℕ) (apples_per_child : ℕ) : ℕ := children * apples_per_child
def eaten_apples (children_eating : ℕ) (apples_eaten_per_child : ℕ) : ℕ := children_eating * apples_eaten_per_child
def remaining_apples (initial : ℕ) (eaten : ℕ) : ℕ := initial - eaten
def sold_apples (remaining : ℕ) (final : ℕ) : ℕ := remaining - final

-- Given conditions
variable (children : ℕ := 5)
variable (apples_per_child : ℕ := 15)
variable (children_eating : ℕ := 2)
variable (apples_eaten_per_child : ℕ := 4)
variable (final_apples : ℕ := 60)

-- Theorem statement
theorem child_sold_apples_correct :
  sold_apples (remaining_apples (initial_apples children apples_per_child) (eaten_apples children_eating apples_eaten_per_child)) final_apples = 7 :=
by
  sorry -- Proof is omitted

end child_sold_apples_correct_l208_208906


namespace combinatorics_sum_l208_208759

theorem combinatorics_sum :
  (Nat.choose 20 6 + Nat.choose 20 5 = 62016) :=
by
  sorry

end combinatorics_sum_l208_208759


namespace original_price_l208_208385

theorem original_price (SP : ℝ) (gain_percent : ℝ) (P : ℝ) : SP = 1080 → gain_percent = 0.08 → SP = P * (1 + gain_percent) → P = 1000 :=
by
  intro hSP hGainPercent hEquation
  sorry

end original_price_l208_208385


namespace transport_capacity_l208_208980

-- Declare x and y as the amount of goods large and small trucks can transport respectively
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := 2 * x + 3 * y = 15.5
def condition2 : Prop := 5 * x + 6 * y = 35

-- The goal to prove
def goal : Prop := 3 * x + 5 * y = 24.5

-- Main theorem stating that given the conditions, the goal follows
theorem transport_capacity (h1 : condition1 x y) (h2 : condition2 x y) : goal x y :=
by sorry

end transport_capacity_l208_208980


namespace part_I_part_II_l208_208789

-- Define the conditions given in the problem
def set_A : Set ℝ := { x | -1 < x ∧ x < 3 }
def set_B (a b : ℝ) : Set ℝ := { x | x^2 - a * x + b < 0 }

-- Part I: Prove that if A = B, then a = 2 and b = -3
theorem part_I (a b : ℝ) (h : set_A = set_B a b) : a = 2 ∧ b = -3 :=
sorry

-- Part II: Prove that if b = 3 and A ∩ B ⊇ B, then the range of a is [-2√3, 4]
theorem part_II (a : ℝ) (b : ℝ := 3) (h : set_A ∩ set_B a b ⊇ set_B a b) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 :=
sorry

end part_I_part_II_l208_208789


namespace find_varphi_l208_208676

theorem find_varphi (ϕ : ℝ) (h1 : 0 < ϕ) (h2 : ϕ < π)
(h_symm : ∃ k : ℤ, ϕ = k * π + 2 * π / 3) :
ϕ = 2 * π / 3 :=
sorry

end find_varphi_l208_208676


namespace circle_relationship_l208_208699

noncomputable def f : ℝ × ℝ → ℝ := sorry

variables {x y x₁ y₁ x₂ y₂ : ℝ}
variables (h₁ : f (x₁, y₁) = 0) (h₂ : f (x₂, y₂) ≠ 0)

theorem circle_relationship :
  f (x, y) - f (x₁, y₁) - f (x₂, y₂) = 0 ↔ f (x, y) = f (x₂, y₂) :=
sorry

end circle_relationship_l208_208699


namespace larger_number_l208_208615

theorem larger_number (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end larger_number_l208_208615


namespace minimize_total_time_l208_208771

def exercise_time (s : ℕ → ℕ) : Prop :=
  ∀ i, s i < 45

def total_exercises (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 25

def minimize_time (a : ℕ → ℕ) (s : ℕ → ℕ) : Prop :=
  ∃ (j : ℕ), (1 ≤ j ∧ j ≤ 7 ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ 7 → if i = j then a i = 25 else a i = 0) ∧
  ∀ i, 1 ≤ i ∧ i ≤ 7 → s i ≥ s j)

theorem minimize_total_time
  (a : ℕ → ℕ) (s : ℕ → ℕ) 
  (h_exercise_time : exercise_time s)
  (h_total_exercises : total_exercises a) :
  minimize_time a s := by
  sorry

end minimize_total_time_l208_208771


namespace spring_length_relationship_maximum_mass_l208_208793

theorem spring_length_relationship (x y : ℝ) : 
  (y = 0.5 * x + 12) ↔ y = 12 + 0.5 * x := 
by sorry

theorem maximum_mass (x y : ℝ) : 
  (y = 0.5 * x + 12) → (y ≤ 20) → (x ≤ 16) :=
by sorry

end spring_length_relationship_maximum_mass_l208_208793


namespace question1_solution_question2_solution_l208_208650

noncomputable def f (x m : ℝ) : ℝ := x^2 - m * x + m - 1

theorem question1_solution (x : ℝ) :
  ∀ x, f x 3 ≤ 0 ↔ 1 ≤ x ∧ x ≤ 2 :=
sorry

theorem question2_solution (m : ℝ) :
  (∀ x, 2 ≤ x ∧ x ≤ 4 → f x m ≥ -1) ↔ m ≤ 4 :=
sorry

end question1_solution_question2_solution_l208_208650


namespace walking_time_difference_at_slower_speed_l208_208961

theorem walking_time_difference_at_slower_speed (T : ℕ) (v_s: ℚ) (h1: T = 32) (h2: v_s = 4/5) : 
  (T * (5/4) - T) = 8 :=
by
  sorry

end walking_time_difference_at_slower_speed_l208_208961


namespace number_of_workers_in_original_scenario_l208_208972

-- Definitions based on the given conditions
def original_days := 70
def alternative_days := 42
def alternative_workers := 50

-- The statement we want to prove
theorem number_of_workers_in_original_scenario : 
  (∃ (W : ℕ), W * original_days = alternative_workers * alternative_days) → ∃ (W : ℕ), W = 30 :=
by
  sorry

end number_of_workers_in_original_scenario_l208_208972


namespace min_cost_yogurt_l208_208409

theorem min_cost_yogurt (cost_per_box : ℕ) (boxes : ℕ) (promotion : ℕ → ℕ) (cost : ℕ) :
  cost_per_box = 4 → 
  boxes = 10 → 
  promotion 3 = 2 → 
  cost = 28 := 
by {
  -- The proof will go here
  sorry
}

end min_cost_yogurt_l208_208409


namespace determine_m_l208_208120

open Set Real

theorem determine_m (m : ℝ) : (∀ x, x ∈ { x | x ≥ 3 } ∪ { x | x < m }) ∧ (∀ x, x ∉ { x | x ≥ 3 } ∩ { x | x < m }) → m = 3 :=
by
  intros h
  sorry

end determine_m_l208_208120


namespace find_a_l208_208346

noncomputable def a : ℚ := ((68^3 - 65^3) * (32^3 + 18^3)) / ((32^2 - 32 * 18 + 18^2) * (68^2 + 68 * 65 + 65^2))

theorem find_a : a = 150 := 
  sorry

end find_a_l208_208346


namespace function_classification_l208_208085

theorem function_classification (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = f (f x) + f (y^2) + 2 * f (x * y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  sorry

end function_classification_l208_208085


namespace number_of_female_students_school_l208_208205

theorem number_of_female_students_school (T S G_s B_s B G : ℕ) (h1 : T = 1600)
    (h2 : S = 200) (h3 : G_s = B_s - 10) (h4 : G_s + B_s = 200) (h5 : B_s = 105) (h6 : G_s = 95) (h7 : B + G = 1600) : 
    G = 760 :=
by
  sorry

end number_of_female_students_school_l208_208205


namespace fish_ratio_bobby_sarah_l208_208987

-- Defining the conditions
variables (bobby sarah tony billy : ℕ)

-- Condition: Billy has 10 fish.
def billy_has_10_fish : billy = 10 := by sorry

-- Condition: Tony has 3 times as many fish as Billy.
def tony_has_3_times_billy : tony = 3 * billy := by sorry

-- Condition: Sarah has 5 more fish than Tony.
def sarah_has_5_more_than_tony : sarah = tony + 5 := by sorry

-- Condition: All 4 people have 145 fish together.
def total_fish : bobby + sarah + tony + billy = 145 := by sorry

-- The theorem we want to prove
theorem fish_ratio_bobby_sarah : (bobby : ℚ) / sarah = 2 / 1 := by
  -- You can write out the entire proof step by step here, but initially, we'll just put sorry.
  sorry

end fish_ratio_bobby_sarah_l208_208987


namespace compute_result_l208_208870

-- Define the operations a # b and b # c
def operation (a b : ℤ) : ℤ := a * b - b + b^2

-- Define the expression for (3 # 8) # z given the operations
def evaluate (z : ℤ) : ℤ := operation (operation 3 8) z

-- Prove that (3 # 8) # z = 79z + z^2
theorem compute_result (z : ℤ) : evaluate z = 79 * z + z^2 := 
by
  sorry

end compute_result_l208_208870


namespace smallest_positive_integer_l208_208316

theorem smallest_positive_integer (
    b : ℤ 
) : 
    (b % 4 = 1) → (b % 5 = 2) → (b % 6 = 3) → b = 21 := 
by
  intros h1 h2 h3
  sorry

end smallest_positive_integer_l208_208316


namespace tangent_line_at_P0_is_parallel_l208_208463

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_at_P0_is_parallel (x y : ℝ) (h_curve : y = curve x) (h_slope : tangent_slope x = 4) :
  (x, y) = (-1, -4) :=
sorry

end tangent_line_at_P0_is_parallel_l208_208463


namespace problem1_problem2_l208_208434

theorem problem1 :
  0.064 ^ (-1 / 3) - (-7 / 8) ^ 0 + 16 ^ 0.75 + 0.01 ^ (1 / 2) = 48 / 5 :=
by sorry

theorem problem2 :
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 
  - 25 ^ (Real.log 3 / Real.log 5) = -7 :=
by sorry

end problem1_problem2_l208_208434


namespace find_certain_number_l208_208394

theorem find_certain_number (h1 : 2994 / 14.5 = 173) (h2 : ∃ x, x / 1.45 = 17.3) : ∃ x, x = 25.085 :=
by
  -- Proof goes here
  sorry

end find_certain_number_l208_208394


namespace technicians_count_l208_208843

theorem technicians_count 
  (T R : ℕ) 
  (h1 : T + R = 14) 
  (h2 : 12000 * T + 6000 * R = 9000 * 14) : 
  T = 7 :=
by
  sorry

end technicians_count_l208_208843


namespace distance_between_centers_l208_208474

variable (P R r : ℝ)
variable (h_tangent : P = R - r)
variable (h_radius1 : R = 6)
variable (h_radius2 : r = 3)

theorem distance_between_centers : P = 3 := by
  sorry

end distance_between_centers_l208_208474


namespace ratio_of_ages_l208_208322

variable (x : Nat) -- The multiple of Marie's age
variable (marco_age marie_age : Nat) -- Marco's and Marie's ages

-- Conditions from (a)
axiom h1 : marie_age = 12
axiom h2 : marco_age = (12 * x) + 1
axiom h3 : marco_age + marie_age = 37

-- Statement to be proved
theorem ratio_of_ages : (marco_age : Nat) / (marie_age : Nat) = (25 / 12) :=
by
  -- Proof steps here
  sorry

end ratio_of_ages_l208_208322


namespace ratio_c_div_d_l208_208438

theorem ratio_c_div_d (a b d : ℝ) (h1 : 8 = 0.02 * a) (h2 : 2 = 0.08 * b) (h3 : d = 0.05 * a) (c : ℝ) (h4 : c = b / a) : c / d = 1 / 320 := 
sorry

end ratio_c_div_d_l208_208438


namespace complement_A_eq_l208_208004

def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1}

theorem complement_A_eq :
  U \ A = {0, 2} :=
by
  sorry

end complement_A_eq_l208_208004


namespace two_pow_n_minus_one_div_by_seven_iff_l208_208782

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ (2^n - 1)) ↔ (∃ k : ℕ, n = 3 * k) := by
  sorry

end two_pow_n_minus_one_div_by_seven_iff_l208_208782


namespace walking_rate_on_escalator_l208_208057

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 160)
  (time_taken : ℝ := 8)
  (distance_eq : escalator_length = (v + escalator_speed) * time_taken) :
  v = 8 :=
by
  sorry

end walking_rate_on_escalator_l208_208057


namespace distance_from_point_to_line_condition_l208_208235

theorem distance_from_point_to_line_condition (a : ℝ) : (|a - 2| = 3) ↔ (a = 5 ∨ a = -1) :=
by
  sorry

end distance_from_point_to_line_condition_l208_208235


namespace smallest_possible_value_of_d_l208_208451

theorem smallest_possible_value_of_d (c d : ℝ) (hc : 1 < c) (hd : c < d)
  (h_triangle1 : ¬(1 + c > d ∧ c + d > 1 ∧ 1 + d > c))
  (h_triangle2 : ¬(1 / c + 1 / d > 1 ∧ 1 / d + 1 > 1 / c ∧ 1 / c + 1 > 1 / d)) :
  d = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end smallest_possible_value_of_d_l208_208451


namespace place_b_left_of_a_forms_correct_number_l208_208315

noncomputable def form_three_digit_number (a b : ℕ) : ℕ :=
  100 * b + a

theorem place_b_left_of_a_forms_correct_number (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 1 ≤ b ∧ b < 10) :
  form_three_digit_number a b = 100 * b + a :=
by sorry

end place_b_left_of_a_forms_correct_number_l208_208315


namespace correct_statement_d_l208_208144

theorem correct_statement_d (x : ℝ) : 2 * (x + 1) = x + 7 → x = 5 :=
by
  sorry

end correct_statement_d_l208_208144


namespace tank_depth_l208_208266

theorem tank_depth (d : ℝ)
    (field_length : ℝ) (field_breadth : ℝ)
    (tank_length : ℝ) (tank_breadth : ℝ)
    (remaining_field_area : ℝ)
    (rise_in_field_level : ℝ)
    (field_area_eq : field_length * field_breadth = 4500)
    (tank_area_eq : tank_length * tank_breadth = 500)
    (remaining_field_area_eq : remaining_field_area = 4500 - 500)
    (earth_volume_spread_eq : remaining_field_area * rise_in_field_level = 2000)
    (volume_eq : tank_length * tank_breadth * d = 2000)
  : d = 4 := by
  sorry

end tank_depth_l208_208266


namespace sum_of_floors_of_square_roots_l208_208728

theorem sum_of_floors_of_square_roots : 
  (⌊Real.sqrt 1⌋ + ⌊Real.sqrt 2⌋ + ⌊Real.sqrt 3⌋ + 
   ⌊Real.sqrt 4⌋ + ⌊Real.sqrt 5⌋ + ⌊Real.sqrt 6⌋ + 
   ⌊Real.sqrt 7⌋ + ⌊Real.sqrt 8⌋ + ⌊Real.sqrt 9⌋ + 
   ⌊Real.sqrt 10⌋ + ⌊Real.sqrt 11⌋ + ⌊Real.sqrt 12⌋ + 
   ⌊Real.sqrt 13⌋ + ⌊Real.sqrt 14⌋ + ⌊Real.sqrt 15⌋ + 
   ⌊Real.sqrt 16⌋ + ⌊Real.sqrt 17⌋ + ⌊Real.sqrt 18⌋ + 
   ⌊Real.sqrt 19⌋ + ⌊Real.sqrt 20⌋ + ⌊Real.sqrt 21⌋ + 
   ⌊Real.sqrt 22⌋ + ⌊Real.sqrt 23⌋ + ⌊Real.sqrt 24⌋ + 
   ⌊Real.sqrt 25⌋) = 75 := 
sorry

end sum_of_floors_of_square_roots_l208_208728


namespace min_value_expression_l208_208008

open Real

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∃ (y : ℝ), y = x * sqrt 2 ∧ ∀ (u : ℝ), ∀ (hu : u > 0), 
     sqrt ((x^2 + u^2) * (4 * x^2 + u^2)) / (x * u) ≥ 3 * sqrt 2) := 
sorry

end min_value_expression_l208_208008


namespace Carla_pays_more_than_Bob_l208_208278

theorem Carla_pays_more_than_Bob
  (slices : ℕ := 12)
  (veg_slices : ℕ := slices / 2)
  (non_veg_slices : ℕ := slices / 2)
  (base_cost : ℝ := 10)
  (extra_cost : ℝ := 3)
  (total_cost : ℝ := base_cost + extra_cost)
  (per_slice_cost : ℝ := total_cost / slices)
  (carla_slices : ℕ := veg_slices + 2)
  (bob_slices : ℕ := 3)
  (carla_payment : ℝ := carla_slices * per_slice_cost)
  (bob_payment : ℝ := bob_slices * per_slice_cost) :
  (carla_payment - bob_payment) = 5.41665 :=
sorry

end Carla_pays_more_than_Bob_l208_208278


namespace num_div_divided_by_10_l208_208328

-- Given condition: the number divided by 10 equals 12
def number_divided_by_10_gives_12 (x : ℝ) : Prop :=
  x / 10 = 12

-- Lean statement for the mathematical problem
theorem num_div_divided_by_10 (x : ℝ) (h : number_divided_by_10_gives_12 x) : x = 120 :=
by
  sorry

end num_div_divided_by_10_l208_208328


namespace find_k_l208_208963

variable (k : ℝ)
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, 2)

theorem find_k 
  (h : (k * a.1 - b.1, k * a.2 - b.2) = (k - 1, k - 2)) 
  (perp_cond : (k * a.1 - b.1, k * a.2 - b.2).fst * (b.1 + a.1) + (k * a.1 - b.1, k * a.2 - b.2).snd * (b.2 + a.2) = 0) :
  k = 8 / 5 :=
sorry

end find_k_l208_208963


namespace total_amount_is_20_yuan_60_cents_l208_208607

-- Conditions
def ten_yuan_note : ℕ := 10
def five_yuan_notes : ℕ := 2 * 5
def twenty_cent_coins : ℕ := 3 * 20

-- Total amount calculation
def total_yuan : ℕ := ten_yuan_note + five_yuan_notes
def total_cents : ℕ := twenty_cent_coins

-- Conversion rates
def yuan_per_cent : ℕ := 100
def total_cents_in_yuan : ℕ := total_cents / yuan_per_cent
def remaining_cents : ℕ := total_cents % yuan_per_cent

-- Proof statement
theorem total_amount_is_20_yuan_60_cents : total_yuan = 20 ∧ total_cents_in_yuan = 0 ∧ remaining_cents = 60 :=
by
  sorry

end total_amount_is_20_yuan_60_cents_l208_208607


namespace total_area_painted_is_correct_l208_208602

noncomputable def barn_area_painted (width length height : ℝ) : ℝ :=
  let walls_area := 2 * (width * height + length * height) * 2
  let ceiling_and_roof_area := 2 * (width * length)
  walls_area + ceiling_and_roof_area

theorem total_area_painted_is_correct 
  (width length height : ℝ) 
  (h_w : width = 12) 
  (h_l : length = 15) 
  (h_h : height = 6) 
  : barn_area_painted width length height = 1008 :=
  by
  rw [h_w, h_l, h_h]
  -- Simplify steps omitted
  sorry

end total_area_painted_is_correct_l208_208602


namespace line_passes_through_fixed_point_l208_208190

theorem line_passes_through_fixed_point (m : ℝ) :
  (m-1) * 9 + (2*m-1) * (-4) = m - 5 :=
by
  sorry

end line_passes_through_fixed_point_l208_208190


namespace percentage_y_less_than_x_l208_208642

variable (x y : ℝ)
variable (h : x = 12 * y)

theorem percentage_y_less_than_x :
  (11 / 12) * 100 = 91.67 := by
  sorry

end percentage_y_less_than_x_l208_208642


namespace max_value_is_one_l208_208982

noncomputable def max_value (x y z : ℝ) : ℝ :=
  (x^2 - 2 * x * y + y^2) * (x^2 - 2 * x * z + z^2) * (y^2 - 2 * y * z + z^2)

theorem max_value_is_one :
  ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x + y + z = 3 →
  max_value x y z ≤ 1 :=
by sorry

end max_value_is_one_l208_208982


namespace markup_is_correct_l208_208545

-- The mathematical interpretation of the given conditions
def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.05
def net_profit : ℝ := 12

-- Define the overhead cost calculation
def overhead_cost : ℝ := overhead_percentage * purchase_price

-- Define the total cost calculation
def total_cost : ℝ := purchase_price + overhead_cost

-- Define the selling price calculation
def selling_price : ℝ := total_cost + net_profit

-- Define the markup calculation
def markup : ℝ := selling_price - purchase_price

-- The statement we want to prove
theorem markup_is_correct : markup = 14.40 :=
by
  -- We will eventually prove this, but for now we use sorry as a placeholder
  sorry

end markup_is_correct_l208_208545


namespace golden_section_AC_correct_l208_208046

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def segment_length := 20
noncomputable def golden_section_point (AB AC BC : ℝ) (h1 : AB = AC + BC) (h2 : AC > BC) (h3 : AB = segment_length) : Prop :=
  AC = (Real.sqrt 5 - 1) / 2 * AB

theorem golden_section_AC_correct :
  ∃ (AC BC : ℝ), (AC + BC = segment_length) ∧ (AC > BC) ∧ (AC = 10 * (Real.sqrt 5 - 1)) :=
by
  sorry

end golden_section_AC_correct_l208_208046


namespace find_divisor_value_l208_208151

theorem find_divisor_value (x : ℝ) (h : 63 / x = 63 - 42) : x = 3 :=
by
  sorry

end find_divisor_value_l208_208151


namespace even_function_a_equals_one_l208_208217

theorem even_function_a_equals_one (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (1 - x) * (-x - a)) → a = 1 :=
by
  intro h
  sorry

end even_function_a_equals_one_l208_208217


namespace seashells_unbroken_l208_208042

theorem seashells_unbroken (total_seashells broken_seashells unbroken_seashells : ℕ) 
  (h1 : total_seashells = 6) 
  (h2 : broken_seashells = 4) 
  (h3 : unbroken_seashells = total_seashells - broken_seashells) :
  unbroken_seashells = 2 :=
by
  sorry

end seashells_unbroken_l208_208042


namespace min_e1_plus_2e2_l208_208305

noncomputable def e₁ (r : ℝ) : ℝ := 2 / (4 - r)
noncomputable def e₂ (r : ℝ) : ℝ := 2 / (4 + r)

theorem min_e1_plus_2e2 (r : ℝ) (h₀ : 0 < r) (h₂ : r < 2) :
  e₁ r + 2 * e₂ r = (3 + 2 * Real.sqrt 2) / 4 :=
sorry

end min_e1_plus_2e2_l208_208305


namespace rival_awards_l208_208145

theorem rival_awards (S J R : ℕ) (h1 : J = 3 * S) (h2 : S = 4) (h3 : R = 2 * J) : R = 24 := 
by sorry

end rival_awards_l208_208145


namespace sum_A_B_C_zero_l208_208990

noncomputable def poly : Polynomial ℝ := Polynomial.X^3 - 16 * Polynomial.X^2 + 72 * Polynomial.X - 27

noncomputable def exists_real_A_B_C 
  (p q r: ℝ) (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) :
  ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r)))) := sorry

theorem sum_A_B_C_zero 
  {p q r: ℝ} (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) 
  (hABC: ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r))))) :
  ∀ A B C, A + B + C = 0 := sorry

end sum_A_B_C_zero_l208_208990


namespace subtract_value_l208_208424

theorem subtract_value (N x : ℤ) (h1 : (N - x) / 7 = 7) (h2 : (N - 6) / 8 = 6) : x = 5 := 
by 
  sorry

end subtract_value_l208_208424


namespace graphs_symmetric_l208_208989

noncomputable def exp2 : ℝ → ℝ := λ x => 2^x
noncomputable def log2 : ℝ → ℝ := λ x => Real.log x / Real.log 2

theorem graphs_symmetric :
  ∀ (x y : ℝ), (y = exp2 x) ↔ (x = log2 y) := sorry

end graphs_symmetric_l208_208989


namespace michael_brought_5000_rubber_bands_l208_208157

noncomputable def totalRubberBands
  (small_band_count : ℕ) (large_band_count : ℕ)
  (small_ball_count : ℕ := 22) (large_ball_count : ℕ := 13)
  (rubber_bands_per_small : ℕ := 50) (rubber_bands_per_large : ℕ := 300) 
: ℕ :=
small_ball_count * rubber_bands_per_small + large_ball_count * rubber_bands_per_large

theorem michael_brought_5000_rubber_bands :
  totalRubberBands 22 13 = 5000 := by
  sorry

end michael_brought_5000_rubber_bands_l208_208157


namespace find_y_positive_monotone_l208_208454

noncomputable def y (y : ℝ) : Prop :=
  0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 ∧ y = 12

theorem find_y_positive_monotone : ∃ y : ℝ, 0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 := by
  sorry

end find_y_positive_monotone_l208_208454


namespace bens_car_costs_l208_208957

theorem bens_car_costs :
  (∃ C_old C_2nd : ℕ,
    (2 * C_old = 4 * C_2nd) ∧
    (C_old = 1800) ∧
    (C_2nd = 900) ∧
    (2 * C_old = 3600) ∧
    (4 * C_2nd = 3600) ∧
    (1800 + 900 = 2700) ∧
    (3600 - 2700 = 900) ∧
    (2000 - 900 = 1100) ∧
    (900 * 0.05 = 45) ∧
    (45 * 2 = 90))
  :=
sorry

end bens_car_costs_l208_208957


namespace proportion_not_necessarily_correct_l208_208193

theorem proportion_not_necessarily_correct
  (a b c d : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : b ≠ 0)
  (h₃ : c ≠ 0)
  (h₄ : d ≠ 0)
  (h₅ : a * d = b * c) :
  ¬ ((a + 1) / b = (c + 1) / d) :=
by 
  sorry

end proportion_not_necessarily_correct_l208_208193


namespace initial_roses_in_vase_l208_208300

theorem initial_roses_in_vase (added_roses current_roses : ℕ) (h1 : added_roses = 8) (h2 : current_roses = 18) : 
  current_roses - added_roses = 10 :=
by
  sorry

end initial_roses_in_vase_l208_208300


namespace value_of_expression_l208_208523

theorem value_of_expression (x : ℝ) (h : x^2 - 3 * x = 4) : 3 * x^2 - 9 * x + 8 = 20 := 
by
  sorry

end value_of_expression_l208_208523


namespace distance_covered_by_train_l208_208269

-- Define the average speed and the total duration of the journey
def speed : ℝ := 10
def time : ℝ := 8

-- Use these definitions to state and prove the distance covered by the train
theorem distance_covered_by_train : speed * time = 80 := by
  sorry

end distance_covered_by_train_l208_208269


namespace distance_from_home_to_high_school_l208_208390

theorem distance_from_home_to_high_school 
  (total_mileage track_distance d : ℝ)
  (h_total_mileage : total_mileage = 10)
  (h_track : track_distance = 4)
  (h_eq : d + d + track_distance = total_mileage) :
  d = 3 :=
by sorry

end distance_from_home_to_high_school_l208_208390


namespace find_investment_sum_l208_208209

variable (P : ℝ)

def simple_interest (rate time : ℝ) (principal : ℝ) : ℝ :=
  principal * rate * time

theorem find_investment_sum (h : simple_interest 0.18 2 P - simple_interest 0.12 2 P = 240) :
  P = 2000 :=
by
  sorry

end find_investment_sum_l208_208209


namespace car_rental_daily_rate_l208_208921

theorem car_rental_daily_rate (x : ℝ) : 
  (x + 0.18 * 48 = 18.95 + 0.16 * 48) -> 
  x = 17.99 :=
by 
  sorry

end car_rental_daily_rate_l208_208921


namespace charge_per_action_figure_l208_208779

-- Definitions according to given conditions
def cost_of_sneakers : ℕ := 90
def saved_amount : ℕ := 15
def num_action_figures : ℕ := 10
def left_after_purchase : ℕ := 25

-- Theorem to prove the charge per action figure
theorem charge_per_action_figure : 
  (cost_of_sneakers - saved_amount + left_after_purchase) / num_action_figures = 10 :=
by 
  sorry

end charge_per_action_figure_l208_208779


namespace maximum_value_of_vectors_l208_208516

open Real EuclideanGeometry

variables (a b c : EuclideanSpace ℝ (Fin 3))

def unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ‖v‖ = 1

def given_conditions (a b c : EuclideanSpace ℝ (Fin 3)) : Prop :=
  unit_vector a ∧ unit_vector b ∧ ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖ ∧ ‖c‖ = 2

theorem maximum_value_of_vectors
  (ha : unit_vector a)
  (hb : unit_vector b)
  (hab : ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖)
  (hc : ‖c‖ = 2) :
  ‖a + b - c‖ ≤ sqrt 2 + 2 := 
by
  sorry

end maximum_value_of_vectors_l208_208516


namespace find_certain_number_l208_208997

open Real

noncomputable def certain_number (x : ℝ) : Prop :=
  0.75 * x = 0.50 * 900

theorem find_certain_number : certain_number 600 :=
by
  dsimp [certain_number]
  -- We need to show that 0.75 * 600 = 0.50 * 900
  sorry

end find_certain_number_l208_208997


namespace part_a_7_pieces_l208_208697

theorem part_a_7_pieces (grid : Fin 4 × Fin 4 → Prop) (h : ∀ i j, ∃ n, grid (i, j) → n < 7)
  (hnoTwoInSameCell : ∀ (i₁ i₂ : Fin 4) (j₁ j₂ : Fin 4), (i₁, j₁) ≠ (i₂, j₂) → grid (i₁, j₁) ≠ grid (i₂, j₂))
  : ∀ (rowsRemoved colsRemoved : Finset (Fin 4)), rowsRemoved.card = 2 → colsRemoved.card = 2
    → ∃ i j, ¬ grid (i, j) := by sorry

end part_a_7_pieces_l208_208697


namespace forty_percent_of_number_l208_208807

theorem forty_percent_of_number (N : ℝ) 
  (h : (1/4) * (1/3) * (2/5) * N = 35) : 0.4 * N = 420 :=
by
  sorry

end forty_percent_of_number_l208_208807


namespace initial_bees_l208_208135

theorem initial_bees (B : ℕ) (h : B + 7 = 23) : B = 16 :=
by {
  sorry
}

end initial_bees_l208_208135


namespace mike_total_money_l208_208501

theorem mike_total_money (num_bills : ℕ) (value_per_bill : ℕ) (h1 : num_bills = 9) (h2 : value_per_bill = 5) :
  (num_bills * value_per_bill) = 45 :=
by
  sorry

end mike_total_money_l208_208501


namespace area_of_sector_equals_13_75_cm2_l208_208399

noncomputable def radius : ℝ := 5 -- radius in cm
noncomputable def arc_length : ℝ := 5.5 -- arc length in cm
noncomputable def circumference : ℝ := 2 * Real.pi * radius -- circumference of the circle
noncomputable def area_of_circle : ℝ := Real.pi * radius^2 -- area of the entire circle

theorem area_of_sector_equals_13_75_cm2 :
  (arc_length / circumference) * area_of_circle = 13.75 :=
by sorry

end area_of_sector_equals_13_75_cm2_l208_208399


namespace parabola_axis_symmetry_value_p_l208_208106

theorem parabola_axis_symmetry_value_p (p : ℝ) (h_parabola : ∀ y x, y^2 = 2 * p * x) (h_axis_symmetry : ∀ (a: ℝ), a = -1 → a = -p / 2) : p = 2 :=
by 
  sorry

end parabola_axis_symmetry_value_p_l208_208106


namespace tan_rewrite_l208_208692

open Real

theorem tan_rewrite (α β : ℝ) 
  (h1 : tan (α + β) = 2 / 5)
  (h2 : tan (β - π / 4) = 1 / 4) : 
  (1 + tan α) / (1 - tan α) = 3 / 22 := 
by
  sorry

end tan_rewrite_l208_208692


namespace ratio_of_speeds_l208_208795

theorem ratio_of_speeds (a b v1 v2 S : ℝ)
  (h1 : S = a * (v1 + v2))
  (h2 : S = b * (v1 - v2)) :
  v2 / v1 = (a + b) / (b - a) :=
by
  sorry

end ratio_of_speeds_l208_208795


namespace value_of_x_plus_2y_l208_208377

theorem value_of_x_plus_2y (x y : ℝ) (h1 : (x + y) / 3 = 1.6666666666666667) (h2 : 2 * x + y = 7) : x + 2 * y = 8 := by
  sorry

end value_of_x_plus_2y_l208_208377


namespace collinear_c1_c2_l208_208641

def vec3 := (ℝ × ℝ × ℝ)

def a : vec3 := (8, 3, -1)
def b : vec3 := (4, 1, 3)

def c1 : vec3 := (2 * 8 - 4, 2 * 3 - 1, 2 * (-1) - 3) -- (12, 5, -5)
def c2 : vec3 := (2 * 4 - 4 * 8, 2 * 1 - 4 * 3, 2 * 3 - 4 * (-1)) -- (-24, -10, 10)

theorem collinear_c1_c2 : ∃ γ : ℝ, c1 = (γ * -24, γ * -10, γ * 10) :=
  sorry

end collinear_c1_c2_l208_208641


namespace num_men_scenario1_is_15_l208_208846

-- Definitions based on the conditions
def hours_per_day_scenario1 : ℕ := 9
def days_scenario1 : ℕ := 16
def men_scenario2 : ℕ := 18
def hours_per_day_scenario2 : ℕ := 8
def days_scenario2 : ℕ := 15
def total_work_done : ℕ := men_scenario2 * hours_per_day_scenario2 * days_scenario2

-- Definition of the number of men M in the first scenario
noncomputable def men_scenario1 : ℕ := total_work_done / (hours_per_day_scenario1 * days_scenario1)

-- Statement of desired proof: prove that the number of men in the first scenario is 15
theorem num_men_scenario1_is_15 :
  men_scenario1 = 15 := by
  sorry

end num_men_scenario1_is_15_l208_208846


namespace zongzi_cost_prices_l208_208563

theorem zongzi_cost_prices (a : ℕ) (n : ℕ)
  (h1 : n * a = 8000)
  (h2 : n * (a - 10) = 6000)
  : a = 40 ∧ a - 10 = 30 :=
by
  sorry

end zongzi_cost_prices_l208_208563


namespace total_cost_is_9_43_l208_208624

def basketball_game_cost : ℝ := 5.20
def racing_game_cost : ℝ := 4.23
def total_cost : ℝ := basketball_game_cost + racing_game_cost

theorem total_cost_is_9_43 : total_cost = 9.43 := by
  sorry

end total_cost_is_9_43_l208_208624


namespace quadratic_inequality_solution_l208_208040

theorem quadratic_inequality_solution
  (x : ℝ) :
  -2 * x^2 + x < -3 ↔ x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 2) := by
  sorry

end quadratic_inequality_solution_l208_208040


namespace add_to_fraction_l208_208160

theorem add_to_fraction (n : ℚ) : (4 + n) / (7 + n) = 7 / 9 → n = 13 / 2 :=
by
  sorry

end add_to_fraction_l208_208160


namespace sum_of_digits_B_equals_4_l208_208357

theorem sum_of_digits_B_equals_4 (A B : ℕ) (N : ℕ) (hN : N = 4444 ^ 4444)
    (hA : A = (N.digits 10).sum) (hB : B = (A.digits 10).sum) :
    (B.digits 10).sum = 4 := by
  sorry

end sum_of_digits_B_equals_4_l208_208357


namespace complement_union_result_l208_208279

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3})
variable (hA : A = {1, 2})
variable (hB : B = {2, 3})

theorem complement_union_result : compl A ∪ B = {0, 2, 3} :=
by
  -- Our proof steps would go here
  sorry

end complement_union_result_l208_208279


namespace collinear_probability_l208_208044

-- Define the rectangular array
def rows : ℕ := 4
def cols : ℕ := 5
def total_dots : ℕ := rows * cols
def chosen_dots : ℕ := 4

-- Define the collinear sets
def horizontal_lines : ℕ := rows
def vertical_lines : ℕ := cols
def collinear_sets : ℕ := horizontal_lines + vertical_lines

-- Define the total combinations of choosing 4 dots out of 20
def total_combinations : ℕ := Nat.choose total_dots chosen_dots

-- Define the probability
def probability : ℚ := collinear_sets / total_combinations

theorem collinear_probability : probability = 9 / 4845 := by
  sorry

end collinear_probability_l208_208044


namespace triangle_construction_condition_l208_208166

variable (varrho_a varrho_b m_c : ℝ)

theorem triangle_construction_condition :
  (∃ (triangle : Type) (ABC : triangle)
    (r_a : triangle → ℝ)
    (r_b : triangle → ℝ)
    (h_from_C : triangle → ℝ),
      r_a ABC = varrho_a ∧
      r_b ABC = varrho_b ∧
      h_from_C ABC = m_c)
  ↔ 
  (1 / m_c = 1 / 2 * (1 / varrho_a + 1 / varrho_b)) :=
sorry

end triangle_construction_condition_l208_208166


namespace max_x_plus_y_l208_208915

theorem max_x_plus_y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 9) (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 7 / 3 :=
sorry

end max_x_plus_y_l208_208915


namespace angles_on_x_axis_l208_208015

theorem angles_on_x_axis (α : ℝ) : 
  (∃ k : ℤ, α = 2 * k * Real.pi) ∨ (∃ k : ℤ, α = (2 * k + 1) * Real.pi) ↔ 
  ∃ k : ℤ, α = k * Real.pi :=
by
  sorry

end angles_on_x_axis_l208_208015


namespace product_of_two_primes_l208_208091

theorem product_of_two_primes (p q z : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) 
    (h_p_range : 2 < p ∧ p < 6) 
    (h_q_range : 8 < q ∧ q < 24) 
    (h_z_def : z = p * q) 
    (h_z_range : 15 < z ∧ z < 36) : 
    z = 33 := 
by 
    sorry

end product_of_two_primes_l208_208091


namespace relationship_among_f_values_l208_208598

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0)

theorem relationship_among_f_values (h₀ : 0 < 2) (h₁ : 2 < 3) :
  f 0 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end relationship_among_f_values_l208_208598


namespace hank_donates_90_percent_l208_208658

theorem hank_donates_90_percent (x : ℝ) : 
  (100 * x + 0.75 * 80 + 50 = 200) → (x = 0.9) :=
by
  intro h
  sorry

end hank_donates_90_percent_l208_208658


namespace maximum_value_l208_208582

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x : ℝ) : ℝ := -Real.log x / x

theorem maximum_value (x1 x2 t : ℝ) (h1 : 0 < t) (h2 : f x1 = t) (h3 : g x2 = t) : 
  ∃ x1 x2, (t > 0) ∧ (f x1 = t) ∧ (g x2 = t) ∧ ((x1 / (x2 * Real.exp t)) = 1 / Real.exp 1) := 
sorry

end maximum_value_l208_208582


namespace projectile_first_reaches_28_l208_208498

theorem projectile_first_reaches_28 (t : ℝ) (h_eq : ∀ t, -4.9 * t^2 + 23.8 * t = 28) : 
    t = 2 :=
sorry

end projectile_first_reaches_28_l208_208498


namespace correct_calculation_is_A_l208_208415

theorem correct_calculation_is_A : (1 + (-2)) = -1 :=
by 
  sorry

end correct_calculation_is_A_l208_208415


namespace equation_1_solution_equation_2_solution_l208_208916

theorem equation_1_solution (x : ℝ) :
  6 * (x - 2 / 3) - (x + 7) = 11 → x = 22 / 5 :=
by
  intro h
  -- The actual proof steps would go here; for now, we use sorry.
  sorry

theorem equation_2_solution (x : ℝ) :
  (2 * x - 1) / 3 = (2 * x + 1) / 6 - 2 → x = -9 / 2 :=
by
  intro h
  -- The actual proof steps would go here; for now, we use sorry.
  sorry

end equation_1_solution_equation_2_solution_l208_208916


namespace levels_for_blocks_l208_208625

theorem levels_for_blocks (S : ℕ → ℕ) (n : ℕ) (h1 : S n = n * (n + 1)) (h2 : S 10 = 110) : n = 10 :=
by {
  sorry
}

end levels_for_blocks_l208_208625


namespace number_of_valid_numbers_l208_208203

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def four_digit_number_conditions : Prop :=
  (∀ N : ℕ, 7000 ≤ N ∧ N < 9000 → 
    (N % 5 = 0) →
    (∃ a b c d : ℕ, 
      N = 1000 * a + 100 * b + 10 * c + d ∧
      (a = 7 ∨ a = 8) ∧
      (d = 0 ∨ d = 5) ∧
      3 ≤ b ∧ is_prime b ∧ b < c ∧ c ≤ 7))

theorem number_of_valid_numbers : four_digit_number_conditions → 
  (∃ n : ℕ, n = 24) :=
  sorry

end number_of_valid_numbers_l208_208203


namespace math_problem_l208_208051

theorem math_problem 
  (X : ℝ)
  (num1 : ℝ := 1 + 28/63)
  (num2 : ℝ := 8 + 7/16)
  (frac_sub1 : ℝ := 19/24 - 21/40)
  (frac_sub2 : ℝ := 1 + 28/63 - 17/21)
  (denom_calc : ℝ := 0.675 * 2.4 - 0.02) :
  0.125 * X / (frac_sub1 * num2) = (frac_sub2 * 0.7) / denom_calc → X = 5 := 
sorry

end math_problem_l208_208051


namespace evaluate_expression_l208_208019

theorem evaluate_expression :
  ( ( ( 5 / 2 : ℚ ) / ( 7 / 12 : ℚ ) ) - ( 4 / 9 : ℚ ) ) = ( 242 / 63 : ℚ ) :=
by
  sorry

end evaluate_expression_l208_208019


namespace bacon_suggestions_count_l208_208608

def mashed_potatoes_suggestions : ℕ := 324
def tomatoes_suggestions : ℕ := 128
def total_suggestions : ℕ := 826

theorem bacon_suggestions_count :
  total_suggestions - (mashed_potatoes_suggestions + tomatoes_suggestions) = 374 :=
by
  sorry

end bacon_suggestions_count_l208_208608


namespace inverse_of_73_mod_74_l208_208281

theorem inverse_of_73_mod_74 :
  73 * 73 ≡ 1 [MOD 74] :=
by
  sorry

end inverse_of_73_mod_74_l208_208281


namespace more_roses_than_orchids_l208_208070

theorem more_roses_than_orchids (roses orchids : ℕ) (h1 : roses = 12) (h2 : orchids = 2) : roses - orchids = 10 := by
  sorry

end more_roses_than_orchids_l208_208070


namespace a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l208_208878

theorem a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4 (a : ℝ) :
  (a < 2 → a^2 < 4) ∧ (a^2 < 4 → a < 2) :=
by
  -- Proof skipped
  sorry

end a_lt_2_is_necessary_but_not_sufficient_for_a_squared_lt_4_l208_208878


namespace simplify_expression_l208_208113

def operation (a b : ℚ) : ℚ := 2 * a - b

theorem simplify_expression (x y : ℚ) : 
  operation (operation (x - y) (x + y)) (-3 * y) = 2 * x - 3 * y :=
by
  sorry

end simplify_expression_l208_208113


namespace family_can_purchase_furniture_in_april_l208_208152

noncomputable def monthly_income : ℤ := 150000
noncomputable def monthly_expenses : ℤ := 115000
noncomputable def initial_savings : ℤ := 45000
noncomputable def furniture_cost : ℤ := 127000

theorem family_can_purchase_furniture_in_april : 
  ∃ (months : ℕ), months = 3 ∧ 
  (initial_savings + months * (monthly_income - monthly_expenses) >= furniture_cost) :=
by
  -- proof will be written here
  sorry

end family_can_purchase_furniture_in_april_l208_208152


namespace solution_l208_208412

theorem solution (t : ℝ) :
  let x := 3 * t
  let y := t
  let z := 0
  x^2 - 9 * y^2 = z^2 :=
by
  sorry

end solution_l208_208412


namespace birdseed_mixture_l208_208124

theorem birdseed_mixture (x : ℝ) (h1 : 0.40 * x + 0.65 * (100 - x) = 50) : x = 60 :=
by
  sorry

end birdseed_mixture_l208_208124


namespace boat_speed_in_still_water_l208_208441

theorem boat_speed_in_still_water  (b s : ℝ) (h1 : b + s = 13) (h2 : b - s = 9) : b = 11 :=
sorry

end boat_speed_in_still_water_l208_208441


namespace fundamental_disagreement_l208_208913

-- Definitions based on conditions
def represents_materialism (s : String) : Prop :=
  s = "Without scenery, where does emotion come from?"

def represents_idealism (s : String) : Prop :=
  s = "Without emotion, where does scenery come from?"

-- Theorem statement
theorem fundamental_disagreement :
  ∀ (s1 s2 : String),
  (represents_materialism s1 ∧ represents_idealism s2) →
  (∃ disagreement : String,
    disagreement = "Acknowledging whether the essence of the world is material or consciousness") :=
by
  intros s1 s2 h
  existsi "Acknowledging whether the essence of the world is material or consciousness"
  sorry

end fundamental_disagreement_l208_208913


namespace minimum_perimeter_is_12_l208_208481

noncomputable def minimum_perimeter_upper_base_frustum
  (a b : ℝ) (h : ℝ) (V : ℝ) : ℝ :=
if h = 3 ∧ V = 63 ∧ (a * b = 9) then
  2 * (a + b)
else
  0 -- this case will never be used

theorem minimum_perimeter_is_12 :
  ∃ a b : ℝ, a * b = 9 ∧ 2 * (a + b) = 12 :=
by
  existsi 3
  existsi 3
  sorry

end minimum_perimeter_is_12_l208_208481


namespace no_four_distinct_sum_mod_20_l208_208254

theorem no_four_distinct_sum_mod_20 (R : Fin 9 → ℕ) (h : ∀ i, R i < 19) :
  ¬ ∃ (a b c d : Fin 9), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (R a + R b) % 20 = (R c + R d) % 20 := sorry

end no_four_distinct_sum_mod_20_l208_208254


namespace linear_regression_decrease_l208_208679

theorem linear_regression_decrease (x : ℝ) (y : ℝ) :
  (h : ∃ c₀ c₁, (c₀ = 2) ∧ (c₁ = -1.5) ∧ y = c₀ - c₁ * x) →
  ( ∃ Δx, Δx = 1 → ∃ Δy, Δy = -1.5) :=
by 
  sorry

end linear_regression_decrease_l208_208679


namespace real_numbers_int_approximation_l208_208773

theorem real_numbers_int_approximation:
  ∀ (x y : ℝ), ∃ (m n : ℤ),
  (x - m) ^ 2 + (y - n) * (x - m) + (y - n) ^ 2 ≤ (1 / 3) :=
by
  intros x y
  sorry

end real_numbers_int_approximation_l208_208773


namespace psychiatrist_problem_l208_208628

theorem psychiatrist_problem 
  (x : ℕ)
  (h_total : 4 * 8 + x + (x + 5) = 25)
  : x = 2 := by
  sorry

end psychiatrist_problem_l208_208628


namespace square_area_from_hexagon_l208_208181

theorem square_area_from_hexagon (hex_side length square_side : ℝ) (h1 : hex_side = 4) (h2 : length = 6 * hex_side)
  (h3 : square_side = length / 4) : square_side ^ 2 = 36 :=
by 
  sorry

end square_area_from_hexagon_l208_208181


namespace sum_factors_of_30_l208_208544

theorem sum_factors_of_30 : (1 + 2 + 3 + 5 + 6 + 10 + 15 + 30) = 72 :=
by
  sorry

end sum_factors_of_30_l208_208544


namespace find_fraction_l208_208749

noncomputable def some_fraction_of_number_is (N f : ℝ) : Prop :=
  1 + f * N = 0.75 * N

theorem find_fraction (N : ℝ) (hN : N = 12.0) :
  ∃ f : ℝ, some_fraction_of_number_is N f ∧ f = 2 / 3 :=
by
  sorry

end find_fraction_l208_208749


namespace larger_number_is_391_l208_208636

theorem larger_number_is_391 (A B : ℕ) 
  (hcf : ∀ n : ℕ, n ∣ A ∧ n ∣ B ↔ n = 23)
  (lcm_factors : ∃ C D : ℕ, lcm A B = 23 * 13 * 17 ∧ C = 13 ∧ D = 17) :
  max A B = 391 :=
sorry

end larger_number_is_391_l208_208636


namespace solution_set_of_inequality_l208_208477

theorem solution_set_of_inequality (x : ℝ) : 
  abs ((x + 2) / x) < 1 ↔ x < -1 :=
by
  sorry

end solution_set_of_inequality_l208_208477


namespace ratio_is_correct_l208_208888

-- Define the constants
def total_students : ℕ := 47
def current_students : ℕ := 6 * 3
def girls_bathroom : ℕ := 3
def new_groups : ℕ := 2 * 4
def foreign_exchange_students : ℕ := 3 * 3

-- The total number of missing students
def missing_students : ℕ := girls_bathroom + new_groups + foreign_exchange_students

-- The number of students who went to the canteen
def students_canteen : ℕ := total_students - current_students - missing_students

-- The ratio of students who went to the canteen to girls who went to the bathroom
def canteen_to_bathroom_ratio : ℕ × ℕ := (students_canteen, girls_bathroom)

theorem ratio_is_correct : canteen_to_bathroom_ratio = (3, 1) :=
by
  -- Proof goes here
  sorry

end ratio_is_correct_l208_208888


namespace time_to_school_building_l208_208200

theorem time_to_school_building 
  (total_time : ℕ := 30) 
  (time_to_gate : ℕ := 15) 
  (time_to_room : ℕ := 9)
  (remaining_time := total_time - time_to_gate - time_to_room) : 
  remaining_time = 6 :=
by
  sorry

end time_to_school_building_l208_208200


namespace partial_fractions_sum_zero_l208_208453

theorem partial_fractions_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, 
     x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 →
     1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4)) →
  A + B + C + D + E = 0 :=
by
  intros h
  sorry

end partial_fractions_sum_zero_l208_208453


namespace carlos_gold_quarters_l208_208657

theorem carlos_gold_quarters (quarter_weight : ℚ) 
  (store_value_per_quarter : ℚ) 
  (melt_value_per_ounce : ℚ) 
  (quarters_per_ounce : ℚ := 1 / quarter_weight) 
  (spent_value : ℚ := quarters_per_ounce * store_value_per_quarter)
  (melted_value: ℚ := melt_value_per_ounce) :
  quarter_weight = 1/5 ∧ store_value_per_quarter = 0.25 ∧ melt_value_per_ounce = 100 → 
  melted_value / spent_value = 80 := 
by
  intros h
  sorry

end carlos_gold_quarters_l208_208657


namespace max_teams_in_chess_tournament_l208_208364

theorem max_teams_in_chess_tournament :
  ∃ n : ℕ, n * (n - 1) ≤ 500 / 9 ∧ ∀ m : ℕ, m * (m - 1) ≤ 500 / 9 → m ≤ n :=
sorry

end max_teams_in_chess_tournament_l208_208364


namespace find_increase_in_perimeter_l208_208260

variable (L B y : ℕ)

theorem find_increase_in_perimeter (h1 : 2 * (L + y + (B + y)) = 2 * (L + B) + 16) : y = 4 := by
  sorry

end find_increase_in_perimeter_l208_208260


namespace length_CF_is_7_l208_208449

noncomputable def CF_length
  (ABCD_rectangle : Prop)
  (triangle_ABE_right : Prop)
  (triangle_CDF_right : Prop)
  (area_triangle_ABE : ℝ)
  (length_AE length_DF : ℝ)
  (h1 : ABCD_rectangle)
  (h2 : triangle_ABE_right)
  (h3 : triangle_CDF_right)
  (h4 : area_triangle_ABE = 150)
  (h5 : length_AE = 15)
  (h6 : length_DF = 24) :
  ℝ :=
7

theorem length_CF_is_7
  (ABCD_rectangle : Prop)
  (triangle_ABE_right : Prop)
  (triangle_CDF_right : Prop)
  (area_triangle_ABE : ℝ)
  (length_AE length_DF : ℝ)
  (h1 : ABCD_rectangle)
  (h2 : triangle_ABE_right)
  (h3 : triangle_CDF_right)
  (h4 : area_triangle_ABE = 150)
  (h5 : length_AE = 15)
  (h6 : length_DF = 24) :
  CF_length ABCD_rectangle triangle_ABE_right triangle_CDF_right area_triangle_ABE length_AE length_DF h1 h2 h3 h4 h5 h6 = 7 :=
by
  sorry

end length_CF_is_7_l208_208449


namespace product_of_numbers_l208_208666

variable (x y z : ℝ)

theorem product_of_numbers :
  x + y + z = 36 ∧ x = 3 * (y + z) ∧ y = 6 * z → x * y * z = 268 := 
by
  sorry

end product_of_numbers_l208_208666


namespace remaining_balance_is_correct_l208_208626

def total_price (deposit amount sales_tax_rate discount_rate service_charge P : ℝ) :=
  let sales_tax := sales_tax_rate * P
  let price_after_tax := P + sales_tax
  let discount := discount_rate * price_after_tax
  let price_after_discount := price_after_tax - discount
  let total_price := price_after_discount + service_charge
  total_price

theorem remaining_balance_is_correct (deposit : ℝ) (amount_paid : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (service_charge : ℝ)
  (P : ℝ) : deposit = 0.10 * P →
         amount_paid = 110 →
         sales_tax_rate = 0.15 →
         discount_rate = 0.05 →
         service_charge = 50 →
         total_price deposit amount_paid sales_tax_rate discount_rate service_charge P - amount_paid = 1141.75 :=
by
  sorry

end remaining_balance_is_correct_l208_208626


namespace WR_eq_35_l208_208902

theorem WR_eq_35 (PQ ZY SX : ℝ) (hPQ : PQ = 30) (hZY : ZY = 15) (hSX : SX = 10) :
    let WS := ZY - SX
    let SR := PQ
    let WR := WS + SR
    WR = 35 := by
  sorry

end WR_eq_35_l208_208902


namespace page_shoes_count_l208_208188

theorem page_shoes_count (p_i : ℕ) (d : ℝ) (b : ℕ) (h1 : p_i = 120) (h2 : d = 0.45) (h3 : b = 15) : 
  (p_i - (d * p_i)) + b = 81 :=
by
  sorry

end page_shoes_count_l208_208188


namespace cupcakes_leftover_l208_208483

-- Definitions based on the conditions
def total_cupcakes : ℕ := 17
def num_children : ℕ := 3

-- Theorem proving the correct answer
theorem cupcakes_leftover : total_cupcakes % num_children = 2 := by
  sorry

end cupcakes_leftover_l208_208483


namespace remainder_of_n_div_7_l208_208090

theorem remainder_of_n_div_7 (n : ℕ) (h1 : n^2 % 7 = 3) (h2 : n^3 % 7 = 6) : n % 7 = 5 :=
sorry

end remainder_of_n_div_7_l208_208090


namespace no_integer_solutions_for_sum_of_squares_l208_208003

theorem no_integer_solutions_for_sum_of_squares :
  ∀ a b c : ℤ, a^2 + b^2 + c^2 ≠ 20122012 := 
by sorry

end no_integer_solutions_for_sum_of_squares_l208_208003


namespace cost_of_pears_l208_208283

theorem cost_of_pears 
  (initial_amount : ℕ := 55) 
  (left_amount : ℕ := 28) 
  (banana_count : ℕ := 2) 
  (banana_price : ℕ := 4) 
  (asparagus_price : ℕ := 6) 
  (chicken_price : ℕ := 11) 
  (total_spent : ℕ := 27) :
  initial_amount - left_amount - (banana_count * banana_price + asparagus_price + chicken_price) = 2 := 
by
  sorry

end cost_of_pears_l208_208283


namespace cookie_recipe_total_cups_l208_208787

theorem cookie_recipe_total_cups (r_butter : ℕ) (r_flour : ℕ) (r_sugar : ℕ) (sugar_cups : ℕ) 
  (h_ratio : r_butter = 1 ∧ r_flour = 2 ∧ r_sugar = 3) (h_sugar : sugar_cups = 9) : 
  r_butter * (sugar_cups / r_sugar) + r_flour * (sugar_cups / r_sugar) + sugar_cups = 18 := 
by 
  sorry

end cookie_recipe_total_cups_l208_208787


namespace isosceles_triangle_properties_l208_208865

noncomputable def isosceles_triangle_sides (a : ℝ) : ℝ × ℝ × ℝ :=
  let x := a * Real.sqrt 3
  let y := 2 * x / 3
  let z := (x + y) / 2
  (x, z, z)

theorem isosceles_triangle_properties (a x y z : ℝ) 
  (h1 : x * y = 2 * a ^ 2) 
  (h2 : x + y = 2 * z) 
  (h3 : y ^ 2 + (x / 2) ^ 2 = z ^ 2) : 
  x = a * Real.sqrt 3 ∧ 
  z = 5 * a * Real.sqrt 3 / 6 :=
by
-- Proof goes here
sorry

end isosceles_triangle_properties_l208_208865


namespace total_profit_is_2560_l208_208240

noncomputable def basicWashPrice : ℕ := 5
noncomputable def deluxeWashPrice : ℕ := 10
noncomputable def premiumWashPrice : ℕ := 15

noncomputable def basicCarsWeekday : ℕ := 50
noncomputable def deluxeCarsWeekday : ℕ := 40
noncomputable def premiumCarsWeekday : ℕ := 20

noncomputable def employeeADailyWage : ℕ := 110
noncomputable def employeeBDailyWage : ℕ := 90
noncomputable def employeeCDailyWage : ℕ := 100
noncomputable def employeeDDailyWage : ℕ := 80

noncomputable def operatingExpenseWeekday : ℕ := 200

noncomputable def totalProfit : ℕ := 
  let revenueWeekday := (basicCarsWeekday * basicWashPrice) + 
                        (deluxeCarsWeekday * deluxeWashPrice) + 
                        (premiumCarsWeekday * premiumWashPrice)
  let totalRevenue := revenueWeekday * 5
  let wageA := employeeADailyWage * 5
  let wageB := employeeBDailyWage * 2
  let wageC := employeeCDailyWage * 3
  let wageD := employeeDDailyWage * 2
  let totalWages := wageA + wageB + wageC + wageD
  let totalOperatingExpenses := operatingExpenseWeekday * 5
  totalRevenue - (totalWages + totalOperatingExpenses)

theorem total_profit_is_2560 : totalProfit = 2560 := by
  sorry

end total_profit_is_2560_l208_208240


namespace squared_expression_l208_208904

variable {x y : ℝ}

theorem squared_expression (x y : ℝ) : (-3 * x^2 * y)^2 = 9 * x^4 * y^2 :=
  by
  sorry

end squared_expression_l208_208904


namespace part1_correct_part2_correct_part3_correct_l208_208339

-- Example survival rates data (provided conditions)
def survivalRatesA : List (Option Float) := [some 95.5, some 92, some 96.5, some 91.6, some 96.3, some 94.6, none, none, none, none]
def survivalRatesB : List (Option Float) := [some 95.1, some 91.6, some 93.2, some 97.8, some 95.6, some 92.3, some 96.6, none, none, none]
def survivalRatesC : List (Option Float) := [some 97, some 95.4, some 98.2, some 93.5, some 94.8, some 95.5, some 94.5, some 93.5, some 98, some 92.5]

-- Define high-quality project condition
def isHighQuality (rate : Float) : Bool := rate > 95.0

-- Problem 1: Probability of two high-quality years from farm B
noncomputable def probabilityTwoHighQualityB : Float := (4.0 * 3.0) / (7.0 * 6.0)

-- Problem 2: Distribution of high-quality projects from farms A, B, and C
structure DistributionX := 
(P0 : Float) -- probability of 0 high-quality years
(P1 : Float) -- probability of 1 high-quality year
(P2 : Float) -- probability of 2 high-quality years
(P3 : Float) -- probability of 3 high-quality years

noncomputable def distributionX : DistributionX := 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
}

-- Problem 3: Inference of average survival rate from high-quality project probabilities
structure AverageSurvivalRates := 
(avgB : Float) 
(avgC : Float)
(probHighQualityB : Float)
(probHighQualityC : Float)
(canInfer : Bool)

noncomputable def avgSurvivalRates : AverageSurvivalRates := 
{ avgB := (95.1 + 91.6 + 93.2 + 97.8 + 95.6 + 92.3 + 96.6) / 7.0,
  avgC := (97 + 95.4 + 98.2 + 93.5 + 94.8 + 95.5 + 94.5 + 93.5 + 98 + 92.5) / 10.0,
  probHighQualityB := 4.0 / 7.0,
  probHighQualityC := 5.0 / 10.0,
  canInfer := false
}

-- Definitions for proof statements indicating correctness
theorem part1_correct : probabilityTwoHighQualityB = (2.0 / 7.0) := sorry

theorem part2_correct : distributionX = 
{ P0 := 3.0 / 28.0,
  P1 := 5.0 / 14.0,
  P2 := 11.0 / 28.0,
  P3 := 1.0 / 7.0 
} := sorry

theorem part3_correct : avgSurvivalRates.canInfer = false := sorry

end part1_correct_part2_correct_part3_correct_l208_208339


namespace inverse_prop_relation_l208_208022

theorem inverse_prop_relation (y₁ y₂ y₃ : ℝ) :
  (y₁ = (1 : ℝ) / (-1)) →
  (y₂ = (1 : ℝ) / (-2)) →
  (y₃ = (1 : ℝ) / (3)) →
  y₃ > y₂ ∧ y₂ > y₁ :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  constructor
  · norm_num
  · norm_num

end inverse_prop_relation_l208_208022


namespace probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l208_208937

-- Considering a die with 6 faces
def die_faces := 6

-- Total number of possible outcomes when rolling 3 dice
def total_outcomes := die_faces^3

-- 1. Probability of having exactly one die showing a 6 when rolling 3 dice
def prob_exactly_one_six : ℚ :=
  have favorable_outcomes := 3 * 5^2 -- 3 ways to choose which die shows 6, and 25 ways for others to not show 6
  favorable_outcomes / total_outcomes

-- Proof statement
theorem probability_exactly_one_six : prob_exactly_one_six = 25/72 := by 
  sorry

-- 2. Probability of having at least one die showing a 6 when rolling 3 dice
def prob_at_least_one_six : ℚ :=
  have no_six_outcomes := 5^3
  (total_outcomes - no_six_outcomes) / total_outcomes

-- Proof statement
theorem probability_at_least_one_six : prob_at_least_one_six = 91/216 := by 
  sorry

-- 3. Probability of having at most one die showing a 6 when rolling 3 dice
def prob_at_most_one_six : ℚ :=
  have no_six_probability := 125 / total_outcomes
  have one_six_probability := 75 / total_outcomes
  no_six_probability + one_six_probability

-- Proof statement
theorem probability_at_most_one_six : prob_at_most_one_six = 25/27 := by 
  sorry

end probability_exactly_one_six_probability_at_least_one_six_probability_at_most_one_six_l208_208937


namespace water_formed_from_reaction_l208_208006

-- Definitions
def mol_mass_water : ℝ := 18.015
def water_formed_grams (moles_water : ℝ) : ℝ := moles_water * mol_mass_water

-- Statement
theorem water_formed_from_reaction (moles_water : ℝ) :
  18 = water_formed_grams moles_water :=
by sorry

end water_formed_from_reaction_l208_208006


namespace hyperbola_eq_l208_208009

/-- Given a hyperbola with center at the origin, 
    one focus at (-√5, 0), and a point P on the hyperbola such that 
    the midpoint of segment PF₁ has coordinates (0, 2), 
    then the equation of the hyperbola is x² - y²/4 = 1. --/
theorem hyperbola_eq (x y : ℝ) (P F1 : ℝ × ℝ) 
  (hF1 : F1 = (-Real.sqrt 5, 0)) 
  (hMidPoint : (P.1 + -Real.sqrt 5) / 2 = 0 ∧ (P.2 + 0) / 2 = 2) 
  : x^2 - y^2 / 4 = 1 := 
sorry

end hyperbola_eq_l208_208009


namespace sandy_correct_sums_l208_208128

theorem sandy_correct_sums :
  ∃ x y : ℕ, x + y = 30 ∧ 3 * x - 2 * y = 60 ∧ x = 24 :=
by
  sorry

end sandy_correct_sums_l208_208128


namespace max_value_expr_l208_208617

open Real

theorem max_value_expr {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 9) : 
  (x / y + y / z + z / x) * (y / x + z / y + x / z) = 81 / 4 :=
sorry

end max_value_expr_l208_208617


namespace blocks_per_tree_l208_208286

def trees_per_day : ℕ := 2
def blocks_after_5_days : ℕ := 30
def days : ℕ := 5

theorem blocks_per_tree : (blocks_after_5_days / (trees_per_day * days)) = 3 :=
by
  sorry

end blocks_per_tree_l208_208286


namespace distinct_arrangements_TOOL_l208_208830

/-- The word "TOOL" consists of four letters where "O" is repeated twice. 
Prove that the number of distinct arrangements of the letters in the word is 12. -/
theorem distinct_arrangements_TOOL : 
  let total_letters := 4
  let repeated_O := 2
  (Nat.factorial total_letters / Nat.factorial repeated_O) = 12 := 
by
  sorry

end distinct_arrangements_TOOL_l208_208830


namespace line_through_point_inequality_l208_208369

theorem line_through_point_inequality
  (a b θ : ℝ)
  (h : (b * Real.cos θ + a * Real.sin θ = a * b)) :
  1 / a^2 + 1 / b^2 ≥ 1 := 
  sorry

end line_through_point_inequality_l208_208369


namespace neg_four_fifth_less_neg_two_third_l208_208528

theorem neg_four_fifth_less_neg_two_third : (-4 : ℚ) / 5 < (-2 : ℚ) / 3 :=
  sorry

end neg_four_fifth_less_neg_two_third_l208_208528


namespace crescents_area_eq_rectangle_area_l208_208177

noncomputable def rectangle_area (a b : ℝ) : ℝ := 4 * a * b

noncomputable def semicircle_area (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2

noncomputable def circumscribed_circle_area (a b : ℝ) : ℝ :=
  Real.pi * (a^2 + b^2)

noncomputable def combined_area (a b : ℝ) : ℝ :=
  rectangle_area a b + 2 * (semicircle_area a) + 2 * (semicircle_area b)

theorem crescents_area_eq_rectangle_area (a b : ℝ) : 
  combined_area a b - circumscribed_circle_area a b = rectangle_area a b :=
by
  unfold combined_area
  unfold circumscribed_circle_area
  unfold rectangle_area
  unfold semicircle_area
  sorry

end crescents_area_eq_rectangle_area_l208_208177


namespace find_f_2023_l208_208306

noncomputable def f : ℤ → ℤ := sorry

theorem find_f_2023 (h1 : ∀ x : ℤ, f (x+2) + f x = 3) (h2 : f 1 = 0) : f 2023 = 3 := sorry

end find_f_2023_l208_208306


namespace solveEquation_l208_208695

theorem solveEquation (x : ℝ) (hx : |x| ≥ 3) : (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (x₁ / 3 + x₁ / Real.sqrt (x₁ ^ 2 - 9) = 35 / 12) ∧ (x₂ / 3 + x₂ / Real.sqrt (x₂ ^ 2 - 9) = 35 / 12)) ∧ x₁ + x₂ = 8.75) :=
sorry

end solveEquation_l208_208695


namespace cos_triangle_inequality_l208_208341

theorem cos_triangle_inequality (α β γ : ℝ) (h_sum : α + β + γ = Real.pi) 
    (h_α : 0 < α) (h_β : 0 < β) (h_γ : 0 < γ) (h_α_lt : α < Real.pi) (h_β_lt : β < Real.pi) (h_γ_lt : γ < Real.pi) : 
    (Real.cos α * Real.cos β + Real.cos β * Real.cos γ + Real.cos γ * Real.cos α) ≤ 3 / 4 :=
by
  sorry

end cos_triangle_inequality_l208_208341


namespace no_int_satisfies_both_congruences_l208_208220

theorem no_int_satisfies_both_congruences :
  ¬ ∃ n : ℤ, (n ≡ 5 [ZMOD 6]) ∧ (n ≡ 1 [ZMOD 21]) :=
sorry

end no_int_satisfies_both_congruences_l208_208220


namespace parallel_vectors_sum_l208_208176

variable (x y : ℝ)
variable (k : ℝ)

theorem parallel_vectors_sum :
  (k * 3 = 2) ∧ (k * x = 4) ∧ (k * y = 5) → x + y = 27 / 2 :=
by
  sorry

end parallel_vectors_sum_l208_208176


namespace log_sum_l208_208581

theorem log_sum : 2 * Real.log 2 + Real.log 25 = 2 := 
by 
  sorry

end log_sum_l208_208581


namespace pyramid_surface_area_l208_208187

noncomputable def total_surface_area (a : ℝ) : ℝ :=
  a^2 * (6 + 3 * Real.sqrt 3 + Real.sqrt 7) / 2

theorem pyramid_surface_area (a : ℝ) :
  let hexagon_base_area := 3 * a^2 * Real.sqrt 3 / 2
  let triangle_area_1 := a^2 / 2
  let triangle_area_2 := a^2
  let triangle_area_3 := a^2 * Real.sqrt 7 / 4
  let lateral_area := 2 * (triangle_area_1 + triangle_area_2 + triangle_area_3)
  total_surface_area a = hexagon_base_area + lateral_area := 
sorry

end pyramid_surface_area_l208_208187


namespace replaced_person_weight_l208_208575

theorem replaced_person_weight :
  ∀ (old_avg_weight new_person_weight incr_weight : ℕ),
    old_avg_weight * 8 + incr_weight = new_person_weight →
    incr_weight = 16 →
    new_person_weight = 81 →
    (old_avg_weight - (new_person_weight - incr_weight) / 8) = 65 :=
by
  intros old_avg_weight new_person_weight incr_weight h1 h2 h3
  -- TODO: Proof goes here
  sorry

end replaced_person_weight_l208_208575


namespace difference_of_squares_l208_208471

theorem difference_of_squares (a b : ℕ) (h1: a = 630) (h2: b = 570) : a^2 - b^2 = 72000 :=
by
  sorry

end difference_of_squares_l208_208471


namespace number_of_planting_methods_l208_208727

noncomputable def num_planting_methods : ℕ :=
  -- Six different types of crops
  let crops := ['A', 'B', 'C', 'D', 'E', 'F']
  -- Six trial fields arranged in a row, numbered 1 through 6
  -- Condition: Crop A cannot be planted in the first two fields
  -- Condition: Crop B must not be adjacent to crop A
  -- Answer: 240 different planting methods
  240

theorem number_of_planting_methods :
  num_planting_methods = 240 :=
  by
    -- Proof omitted
    sorry

end number_of_planting_methods_l208_208727


namespace Matthias_fewer_fish_l208_208979

-- Define the number of fish Micah has
def Micah_fish : ℕ := 7

-- Define the number of fish Kenneth has
def Kenneth_fish : ℕ := 3 * Micah_fish

-- Define the total number of fish
def total_fish : ℕ := 34

-- Define the number of fish Matthias has
def Matthias_fish : ℕ := total_fish - (Micah_fish + Kenneth_fish)

-- State the theorem for the number of fewer fish Matthias has compared to Kenneth
theorem Matthias_fewer_fish : Kenneth_fish - Matthias_fish = 15 := by
  -- Proof goes here
  sorry

end Matthias_fewer_fish_l208_208979


namespace distribute_items_among_people_l208_208396

theorem distribute_items_among_people :
  (Nat.choose (10 + 3 - 1) 3) = 220 := 
by sorry

end distribute_items_among_people_l208_208396


namespace polynomial_value_l208_208458

theorem polynomial_value
  (x : ℝ)
  (h : x^2 + 2 * x - 2 = 0) :
  4 - 2 * x - x^2 = 2 :=
by
  sorry

end polynomial_value_l208_208458


namespace part1_x_values_part2_m_value_l208_208403

/-- 
Part 1: Given \(2x^2 + 3x - 5\) and \(-2x + 2\) are opposite numbers, 
prove that \(x = -\frac{3}{2}\) or \(x = 1\).
-/
theorem part1_x_values (x : ℝ)
  (hyp : 2 * x ^ 2 + 3 * x - 5 = -(-2 * x + 2)) :
  2 * x ^ 2 + 5 * x - 7 = 0 → (x = -3 / 2 ∨ x = 1) :=
by
  sorry

/-- 
Part 2: If \(\sqrt{m^2 - 6}\) and \(\sqrt{6m + 1}\) are of the same type, 
prove that \(m = 7\).
-/
theorem part2_m_value (m : ℝ)
  (hyp : m ^ 2 - 6 = 6 * m + 1) :
  7 ^ 2 - 6 = 6 * 7 + 1 → m = 7 :=
by
  sorry

end part1_x_values_part2_m_value_l208_208403


namespace sum_of_80th_equation_l208_208723

theorem sum_of_80th_equation : (2 * 80 + 1) + (5 * 80 - 1) = 560 := by
  sorry

end sum_of_80th_equation_l208_208723


namespace minimum_value_of_E_l208_208917

theorem minimum_value_of_E (x E : ℝ) (h : |x - 4| + |E| + |x - 5| = 12) : |E| = 11 :=
sorry

end minimum_value_of_E_l208_208917


namespace complement_P_l208_208230

def U : Set ℝ := Set.univ

def P : Set ℝ := {x | x^2 < 1}

theorem complement_P : (U \ P) = Set.Iic (-1) ∪ Set.Ici 1 := by
  sorry

end complement_P_l208_208230


namespace difference_of_cubes_divisible_by_8_l208_208452

theorem difference_of_cubes_divisible_by_8 (a b : ℤ) : 
  8 ∣ ((2 * a - 1) ^ 3 - (2 * b - 1) ^ 3) := 
by
  sorry

end difference_of_cubes_divisible_by_8_l208_208452


namespace points_on_ellipse_l208_208296

theorem points_on_ellipse (u : ℝ) :
  let x := (Real.cos u + Real.sin u)
  let y := (4 * (Real.cos u - Real.sin u))
  (x^2 / 2 + y^2 / 32 = 1) :=
by
  let x := (Real.cos u + Real.sin u)
  let y := (4 * (Real.cos u - Real.sin u))
  sorry

end points_on_ellipse_l208_208296


namespace number_of_posts_needed_l208_208169

-- Define the conditions
def length_of_field : ℕ := 80
def width_of_field : ℕ := 60
def distance_between_posts : ℕ := 10

-- Statement to prove the number of posts needed to completely fence the field
theorem number_of_posts_needed : 
  (2 * (length_of_field / distance_between_posts + 1) + 
   2 * (width_of_field / distance_between_posts + 1) - 
   4) = 28 := 
by
  -- Skipping the proof for this theorem
  sorry

end number_of_posts_needed_l208_208169


namespace mat_inverse_sum_l208_208634

theorem mat_inverse_sum (a b c d : ℝ)
  (h1 : -2 * a + 3 * d = 1)
  (h2 : a * c - 12 = 0)
  (h3 : -8 + b * d = 0)
  (h4 : 4 * c - 4 * b = 0)
  (abc : a = 3 * Real.sqrt 2)
  (bb : b = 2 * Real.sqrt 2)
  (cc : c = 2 * Real.sqrt 2)
  (dd : d = (1 + 6 * Real.sqrt 2) / 3) :
  a + b + c + d = 9 * Real.sqrt 2 + 1 / 3 := by
  sorry

end mat_inverse_sum_l208_208634


namespace greatest_multiple_of_4_less_than_100_l208_208682

theorem greatest_multiple_of_4_less_than_100 : ∃ n : ℕ, n % 4 = 0 ∧ n < 100 ∧ ∀ m : ℕ, (m % 4 = 0 ∧ m < 100) → m ≤ n 
:= by
  sorry

end greatest_multiple_of_4_less_than_100_l208_208682


namespace sqrt_10_integer_decimal_partition_l208_208238

theorem sqrt_10_integer_decimal_partition:
  let a := Int.floor (Real.sqrt 10)
  let b := Real.sqrt 10 - a
  (Real.sqrt 10 + a) * b = 1 :=
by
  sorry

end sqrt_10_integer_decimal_partition_l208_208238


namespace sqrt_equiv_1715_l208_208831

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end sqrt_equiv_1715_l208_208831


namespace total_area_of_house_is_2300_l208_208552

-- Definitions based on the conditions in the problem
def area_living_room_dining_room_kitchen : ℕ := 1000
def area_master_bedroom_suite : ℕ := 1040
def area_guest_bedroom : ℕ := area_master_bedroom_suite / 4

-- Theorem to state the total area of the house
theorem total_area_of_house_is_2300 :
  area_living_room_dining_room_kitchen + area_master_bedroom_suite + area_guest_bedroom = 2300 :=
by
  sorry

end total_area_of_house_is_2300_l208_208552


namespace value_range_of_f_l208_208638

open Set

def f (x : ℝ) : ℝ := x^2 - 2 * x

theorem value_range_of_f : {y : ℝ | ∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), f x = y} = Icc (-1 : ℝ) 8 := 
by
  sorry

end value_range_of_f_l208_208638


namespace cost_of_items_l208_208100

theorem cost_of_items (e t b : ℝ) 
    (h1 : 3 * e + 4 * t = 3.20)
    (h2 : 4 * e + 3 * t = 3.50)
    (h3 : 5 * e + 5 * t + 2 * b = 5.70) :
    4 * e + 4 * t + 3 * b = 5.20 :=
by
  sorry

end cost_of_items_l208_208100


namespace total_students_standing_committee_ways_different_grade_pairs_ways_l208_208928

-- Given conditions
def freshmen : ℕ := 5
def sophomores : ℕ := 6
def juniors : ℕ := 4

-- Proofs (statements only, no proofs provided)
theorem total_students : freshmen + sophomores + juniors = 15 :=
by sorry

theorem standing_committee_ways : freshmen * sophomores * juniors = 120 :=
by sorry

theorem different_grade_pairs_ways :
  freshmen * sophomores + sophomores * juniors + juniors * freshmen = 74 :=
by sorry

end total_students_standing_committee_ways_different_grade_pairs_ways_l208_208928


namespace sales_tax_amount_l208_208472

variable (T : ℝ := 25) -- Total amount spent
variable (y : ℝ := 19.7) -- Cost of tax-free items
variable (r : ℝ := 0.06) -- Tax rate

theorem sales_tax_amount : 
  ∃ t : ℝ, t = 0.3 ∧ (T - y) * r = t :=
by 
  sorry

end sales_tax_amount_l208_208472


namespace joe_probability_select_counsel_l208_208427

theorem joe_probability_select_counsel :
  let CANOE := ['C', 'A', 'N', 'O', 'E']
  let SHRUB := ['S', 'H', 'R', 'U', 'B']
  let FLOW := ['F', 'L', 'O', 'W']
  let COUNSEL := ['C', 'O', 'U', 'N', 'S', 'E', 'L']
  -- Probability of selecting C and O from CANOE
  let p_CANOE := 1 / (Nat.choose 5 2)
  -- Probability of selecting U, S, and E from SHRUB
  let comb_SHRUB := Nat.choose 5 3
  let count_USE := 3  -- Determined from the solution
  let p_SHRUB := count_USE / comb_SHRUB
  -- Probability of selecting L, O, W, F from FLOW
  let p_FLOW := 1 / 1
  -- Total probability
  let total_prob := p_CANOE * p_SHRUB * p_FLOW
  total_prob = 3 / 100 := by
    sorry

end joe_probability_select_counsel_l208_208427


namespace pete_mileage_l208_208826

def steps_per_flip : Nat := 100000
def flips : Nat := 50
def final_reading : Nat := 25000
def steps_per_mile : Nat := 2000

theorem pete_mileage :
  let total_steps := (steps_per_flip * flips) + final_reading
  let total_miles := total_steps.toFloat / steps_per_mile.toFloat
  total_miles = 2512.5 :=
by
  sorry

end pete_mileage_l208_208826


namespace train_crosses_pole_in_9_seconds_l208_208733

theorem train_crosses_pole_in_9_seconds
  (speed_kmh : ℝ) (train_length_m : ℝ) (time_s : ℝ) 
  (h1 : speed_kmh = 58) 
  (h2 : train_length_m = 145) 
  (h3 : time_s = train_length_m / (speed_kmh * 1000 / 3600)) :
  time_s = 9 :=
by
  sorry

end train_crosses_pole_in_9_seconds_l208_208733


namespace fruit_basket_combinations_l208_208571

theorem fruit_basket_combinations :
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples+1) * (oranges+1) * (bananas+1)
  let empty_basket := 1
  total_combinations - empty_basket = 159 :=
by
  let apples := 3
  let oranges := 7
  let bananas := 4
  let total_combinations := (apples + 1) * (oranges + 1) * (bananas + 1)
  let empty_basket := 1
  have h_total_combinations : total_combinations = 4 * 8 * 5 := by sorry
  have h_empty_basket : empty_basket = 1 := by sorry
  have h_subtract : 4 * 8 * 5 - 1 = 159 := by sorry
  exact h_subtract

end fruit_basket_combinations_l208_208571


namespace price_of_peas_l208_208143

theorem price_of_peas
  (P : ℝ) -- price of peas per kg in rupees
  (price_soybeans : ℝ) (price_mixture : ℝ)
  (ratio_peas_soybeans : ℝ) :
  price_soybeans = 25 →
  price_mixture = 19 →
  ratio_peas_soybeans = 2 →
  P = 16 :=
by
  intros h_price_soybeans h_price_mixture h_ratio
  sorry

end price_of_peas_l208_208143


namespace intersection_coords_perpendicular_line_l208_208277

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := x + y - 2 = 0

theorem intersection_coords : ∃ P : ℝ × ℝ, line1 P.1 P.2 ∧ line2 P.1 P.2 ∧ P = (1, 1) := by
  sorry

theorem perpendicular_line (x y : ℝ) (P : ℝ × ℝ) (hP: P = (1, 1)) : 
  (line2 P.1 P.2) → x - y = 0 := by
  sorry

end intersection_coords_perpendicular_line_l208_208277


namespace sequence_general_formula_l208_208502

theorem sequence_general_formula (n : ℕ) (hn : n > 0) 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS : ∀ n, S n = 1 - n * a n) 
  (hpos : ∀ n, a n > 0) : 
  (a n = 1 / (n * (n + 1))) :=
sorry

end sequence_general_formula_l208_208502


namespace hyperbola_eccentricity_l208_208710

-- Let's define the variables and conditions first
variables (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
variable (h_asymptote : b = a)

-- We need to prove the eccentricity
theorem hyperbola_eccentricity : eccentricity = Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l208_208710


namespace least_positive_integer_with_12_factors_l208_208489

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l208_208489


namespace marbles_difference_l208_208530

-- Conditions
def L : ℕ := 23
def F : ℕ := 9

-- Proof statement
theorem marbles_difference : L - F = 14 := by
  sorry

end marbles_difference_l208_208530


namespace benny_turnips_l208_208379

theorem benny_turnips (M B : ℕ) (h1 : M = 139) (h2 : M = B + 26) : B = 113 := 
by 
  sorry

end benny_turnips_l208_208379


namespace large_painting_area_l208_208093

theorem large_painting_area :
  ∃ (large_painting : ℕ),
  (3 * (6 * 6) + 4 * (2 * 3) + large_painting = 282) → large_painting = 150 := by
  sorry

end large_painting_area_l208_208093


namespace g_zero_eq_zero_l208_208948

noncomputable def g : ℝ → ℝ :=
  sorry

axiom functional_equation (a b : ℝ) :
  g (3 * a + 2 * b) + g (3 * a - 2 * b) = 2 * g (3 * a) + 2 * g (2 * b)

theorem g_zero_eq_zero : g 0 = 0 :=
by
  let a := 0
  let b := 0
  have eqn := functional_equation a b
  sorry

end g_zero_eq_zero_l208_208948


namespace original_price_of_petrol_l208_208909

theorem original_price_of_petrol (P : ℝ) (h : 0.9 * P * 190 / (0.9 * P) = 190 / P + 5) : P = 4.22 :=
by
  -- The proof goes here
  sorry

end original_price_of_petrol_l208_208909


namespace hats_cost_l208_208717

variables {week_days : ℕ} {weeks : ℕ} {cost_per_hat : ℕ}

-- Conditions
def num_hats (week_days : ℕ) (weeks : ℕ) : ℕ := week_days * weeks
def total_cost (num_hats : ℕ) (cost_per_hat : ℕ) : ℕ := num_hats * cost_per_hat

-- Proof problem
theorem hats_cost (h1 : week_days = 7) (h2 : weeks = 2) (h3 : cost_per_hat = 50) : 
  total_cost (num_hats week_days weeks) cost_per_hat = 700 :=
by 
  sorry

end hats_cost_l208_208717


namespace trajectory_of_moving_circle_l208_208681

-- Definitions for the given circles C1 and C2
def Circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Prove the trajectory of the center of the moving circle M
theorem trajectory_of_moving_circle (x y : ℝ) :
  ((∃ x_center y_center : ℝ, Circle1 x_center y_center ∧ Circle2 x_center y_center ∧ 
  -- Tangency conditions for Circle M
  (x - x_center)^2 + y^2 = (x_center - 2)^2 + y^2 ∧ (x - x_center)^2 + y^2 = (x_center + 2)^2 + y^2)) →
  (x = 0 ∨ x^2 - y^2 / 3 = 1) := 
sorry

end trajectory_of_moving_circle_l208_208681


namespace roots_cube_reciprocal_eqn_l208_208204

variable (a b c r s : ℝ)

def quadratic_eqn (r s : ℝ) : Prop :=
  3 * a * r ^ 2 + 5 * b * r + 7 * c = 0 ∧ 
  3 * a * s ^ 2 + 5 * b * s + 7 * c = 0

theorem roots_cube_reciprocal_eqn (h : quadratic_eqn a b c r s) :
  (1 / r^3 + 1 / s^3) = (-5 * b * (25 * b ^ 2 - 63 * c) / (343 * c^3)) :=
sorry

end roots_cube_reciprocal_eqn_l208_208204


namespace percentage_increase_from_second_to_third_building_l208_208216

theorem percentage_increase_from_second_to_third_building :
  let first_building_units := 4000
  let second_building_units := (2 / 5 : ℝ) * first_building_units
  let total_units := 7520
  let third_building_units := total_units - (first_building_units + second_building_units)
  (third_building_units - second_building_units) / second_building_units * 100 = 20 := by
  let first_building_units := 4000
  let second_building_units : ℝ := (2 / 5 : ℝ) * first_building_units
  let total_units := 7520
  let third_building_units := total_units - (first_building_units + second_building_units)
  have H : (third_building_units - second_building_units) / second_building_units * 100 = 20 := sorry
  exact H

end percentage_increase_from_second_to_third_building_l208_208216


namespace unique_solution_l208_208765

noncomputable def pair_satisfying_equation (m n : ℕ) : Prop :=
  2^m - 1 = 3^n

theorem unique_solution : ∀ (m n : ℕ), m > 0 → n > 0 → pair_satisfying_equation m n → (m, n) = (2, 1) :=
by
  intros m n m_pos n_pos h
  sorry

end unique_solution_l208_208765


namespace farmer_turkeys_l208_208134

variable (n c : ℝ)

theorem farmer_turkeys (h1 : n * c = 60) (h2 : (c + 0.10) * (n - 15) = 54) : n = 75 :=
sorry

end farmer_turkeys_l208_208134


namespace quadratic_is_perfect_square_l208_208568

theorem quadratic_is_perfect_square (c : ℝ) :
  (∃ b : ℝ, (3 * (x : ℝ) + b)^2 = 9 * x^2 - 24 * x + c) ↔ c = 16 :=
by sorry

end quadratic_is_perfect_square_l208_208568


namespace pizza_boxes_sold_l208_208439

variables (P : ℕ) -- Representing the number of pizza boxes sold

def pizza_price : ℝ := 12
def fries_price : ℝ := 0.30
def soda_price : ℝ := 2

def fries_sold : ℕ := 40
def soda_sold : ℕ := 25

def goal_amount : ℝ := 500
def more_needed : ℝ := 258
def current_amount : ℝ := goal_amount - more_needed

-- Total earnings calculation
def total_earnings : ℝ := (P : ℝ) * pizza_price + fries_sold * fries_price + soda_sold * soda_price

theorem pizza_boxes_sold (h : total_earnings P = current_amount) : P = 15 := 
by
  sorry

end pizza_boxes_sold_l208_208439


namespace rectangle_dimensions_l208_208811

theorem rectangle_dimensions (x y : ℝ) (h1 : x = 2 * y) (h2 : 2 * (x + y) = 2 * x * y) : 
  (x = 3 ∧ y = 1.5) :=
by
  sorry

end rectangle_dimensions_l208_208811


namespace find_initial_marbles_l208_208786

def initial_marbles (W Y H : ℕ) : Prop :=
  (W + 2 = 20) ∧ (Y - 5 = 20) ∧ (H + 3 = 20)

theorem find_initial_marbles (W Y H : ℕ) (h : initial_marbles W Y H) : W = 18 :=
  by
    sorry

end find_initial_marbles_l208_208786


namespace hall_length_l208_208326

theorem hall_length (L : ℝ) (H : ℝ) 
  (h1 : 2 * (L * 15) = 2 * (L * H) + 2 * (15 * H)) 
  (h2 : L * 15 * H = 1687.5) : 
  L = 15 :=
by 
  sorry

end hall_length_l208_208326


namespace remaining_sum_avg_l208_208936

variable (a b : ℕ → ℝ)
variable (h1 : 1 / 6 * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 2.5)
variable (h2 : 1 / 2 * (a 1 + a 2) = 1.1)
variable (h3 : 1 / 2 * (a 3 + a 4) = 1.4)

theorem remaining_sum_avg :
  1 / 2 * (a 5 + a 6) = 5 :=
by
  sorry

end remaining_sum_avg_l208_208936


namespace larger_integer_is_72_l208_208604

theorem larger_integer_is_72 (x y : ℤ) (h1 : y = 4 * x) (h2 : (x + 6) * 3 = y) : y = 72 :=
sorry

end larger_integer_is_72_l208_208604


namespace circle_center_l208_208020

theorem circle_center :
    ∃ (h k : ℝ), (x^2 - 10 * x + y^2 - 4 * y = -4) →
                 (x - h)^2 + (y - k)^2 = 25 ∧ h = 5 ∧ k = 2 :=
sorry

end circle_center_l208_208020


namespace range_of_a_l208_208061

theorem range_of_a (m a : ℝ) (h1 : m < a) (h2 : m ≤ -1) : a > -1 :=
by sorry

end range_of_a_l208_208061


namespace squares_on_grid_l208_208814

-- Defining the problem conditions
def grid_size : ℕ := 5
def total_points : ℕ := grid_size * grid_size
def used_points : ℕ := 20

-- Stating the theorem to prove the total number of squares formed
theorem squares_on_grid : 
  (total_points = 25) ∧ (used_points = 20) →
  (∃ all_squares : ℕ, all_squares = 21) :=
by
  intros
  sorry

end squares_on_grid_l208_208814


namespace prove_intersection_l208_208349

-- Defining the set M
def M : Set ℝ := { x | x^2 - 2 * x < 0 }

-- Defining the set N
def N : Set ℝ := { x | x ≥ 1 }

-- Defining the complement of N in ℝ
def complement_N : Set ℝ := { x | x < 1 }

-- The intersection M ∩ complement_N
def intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The statement to be proven
theorem prove_intersection : M ∩ complement_N = intersection :=
by
  sorry

end prove_intersection_l208_208349


namespace num_brownies_correct_l208_208170

-- Define the conditions (pan dimensions and brownie piece dimensions)
def pan_width : ℕ := 24
def pan_length : ℕ := 15
def piece_width : ℕ := 3
def piece_length : ℕ := 2

-- Define the area calculations for the pan and each piece
def pan_area : ℕ := pan_width * pan_length
def piece_area : ℕ := piece_width * piece_length

-- Define the problem statement to prove the number of brownies
def number_of_brownies : ℕ := pan_area / piece_area

-- The statement we need to prove
theorem num_brownies_correct : number_of_brownies = 60 :=
by
  sorry

end num_brownies_correct_l208_208170


namespace candy_division_l208_208099

theorem candy_division (total_candy num_students : ℕ) (h1 : total_candy = 344) (h2 : num_students = 43) : total_candy / num_students = 8 := by
  sorry

end candy_division_l208_208099


namespace ellipse_equation_l208_208864

theorem ellipse_equation (a : ℝ) (x y : ℝ) (h : (x, y) = (-3, 2)) :
  (∃ a : ℝ, ∀ x y : ℝ, x^2 / 15 + y^2 / 10 = 1) ↔ (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 15 + p.2^2 / 10 = 1 } :=
by
  have h1 : 15 = a^2 := by
    sorry
  have h2 : 10 = a^2 - 5 := by
    sorry
  sorry

end ellipse_equation_l208_208864


namespace simplify_product_l208_208998

theorem simplify_product (x y : ℝ) : 
  (x - 3 * y + 2) * (x + 3 * y + 2) = (x^2 + 4 * x + 4 - 9 * y^2) :=
by
  sorry

end simplify_product_l208_208998


namespace find_f_neg5_l208_208705

theorem find_f_neg5 (a b : ℝ) (Sin : ℝ → ℝ) (f : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x + b * (Sin x) ^ 3 + 1)
  (h_f5 : f 5 = 7) :
  f (-5) = -5 := 
by
  sorry

end find_f_neg5_l208_208705


namespace speed_of_goods_train_l208_208414

theorem speed_of_goods_train
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_crossing : ℕ)
  (h_length_train : length_train = 240)
  (h_length_platform : length_platform = 280)
  (h_time_crossing : time_crossing = 26)
  : (length_train + length_platform) / time_crossing * (3600 / 1000) = 72 := 
by sorry

end speed_of_goods_train_l208_208414


namespace ellipse_equation_correct_coordinates_c_correct_l208_208550

-- Definition of the ellipse Γ with given properties
def ellipse_properties (a b : ℝ) (ecc : ℝ) (c_len : ℝ) :=
  a > b ∧ b > 0 ∧ ecc = (Real.sqrt 2) / 2 ∧ c_len = Real.sqrt 2

-- Correct answer for the equation of the ellipse
def correct_ellipse_equation := ∀ x y : ℝ, (x^2) / 2 + y^2 = 1

-- Proving that given the properties of the ellipse, the equation is as stated
theorem ellipse_equation_correct (a b : ℝ) (h : ellipse_properties a b (Real.sqrt 2 / 2) (Real.sqrt 2)) :
  (x^2) / 2 + y^2 = 1 := 
  sorry

-- Definition of the conditions for points A, B, and C
def triangle_conditions (a b : ℝ) (area : ℝ) :=
  ∀ A B : ℝ × ℝ,
    A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
    B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
    area = 3 * Real.sqrt 6 / 4

-- Correct coordinates of point C given the conditions
def correct_coordinates_c (C : ℝ × ℝ) :=
  (C = (1, Real.sqrt 2 / 2) ∨ C = (2, 1))

-- Proving that given the conditions, the coordinates of point C are correct
theorem coordinates_c_correct (a b : ℝ) (h : triangle_conditions a b (3 * Real.sqrt 6 / 4)) (C : ℝ × ℝ) :
  correct_coordinates_c C :=
  sorry

end ellipse_equation_correct_coordinates_c_correct_l208_208550


namespace square_of_negative_eq_square_l208_208965

theorem square_of_negative_eq_square (a : ℝ) : (-a)^2 = a^2 :=
sorry

end square_of_negative_eq_square_l208_208965


namespace find_sister_candy_initially_l208_208839

-- Defining the initial pieces of candy Katie had.
def katie_candy : ℕ := 8

-- Defining the pieces of candy Katie's sister had initially.
def sister_candy_initially : ℕ := sorry -- To be determined

-- The total number of candy pieces they had after eating 8 pieces.
def total_remaining_candy : ℕ := 23

theorem find_sister_candy_initially : 
  (katie_candy + sister_candy_initially - 8 = total_remaining_candy) → (sister_candy_initially = 23) :=
by
  sorry

end find_sister_candy_initially_l208_208839


namespace range_of_3a_minus_b_l208_208115

theorem range_of_3a_minus_b (a b : ℝ) (h1 : 2 ≤ a + b ∧ a + b ≤ 5) (h2 : -2 ≤ a - b ∧ a - b ≤ 1) : 
    -2 ≤ 3 * a - b ∧ 3 * a - b ≤ 7 := 
by 
  sorry

end range_of_3a_minus_b_l208_208115


namespace percentage_of_volume_occupied_l208_208564

-- Define the dimensions of the block
def block_length : ℕ := 9
def block_width : ℕ := 7
def block_height : ℕ := 12

-- Define the dimension of the cube
def cube_side : ℕ := 4

-- Define the volumes
def block_volume : ℕ := block_length * block_width * block_height
def cube_volume : ℕ := cube_side * cube_side * cube_side

-- Define the count of cubes along each dimension
def cubes_along_length : ℕ := block_length / cube_side
def cubes_along_width : ℕ := block_width / cube_side
def cubes_along_height : ℕ := block_height / cube_side

-- Define the total number of cubes that fit into the block
def total_cubes : ℕ := cubes_along_length * cubes_along_width * cubes_along_height

-- Define the total volume occupied by the cubes
def occupied_volume : ℕ := total_cubes * cube_volume

-- Define the percentage of the block's volume occupied by the cubes (as a float for precision)
def volume_percentage : Float := (Float.ofNat occupied_volume / Float.ofNat block_volume) * 100

-- Statement to prove
theorem percentage_of_volume_occupied :
  volume_percentage = 50.79 := by
  sorry

end percentage_of_volume_occupied_l208_208564


namespace inverse_of_3_mod_199_l208_208197

theorem inverse_of_3_mod_199 : (3 * 133) % 199 = 1 :=
by
  sorry

end inverse_of_3_mod_199_l208_208197


namespace evaluate_fraction_l208_208871

theorem evaluate_fraction : 1 + 3 / (4 + 5 / (6 + 7 / 8)) = 85 / 52 :=
by sorry

end evaluate_fraction_l208_208871


namespace probability_of_two_green_apples_l208_208358

theorem probability_of_two_green_apples (total_apples green_apples choose_apples : ℕ)
  (h_total : total_apples = 8)
  (h_green : green_apples = 4)
  (h_choose : choose_apples = 2) 
: (Nat.choose green_apples choose_apples : ℚ) / (Nat.choose total_apples choose_apples) = 3 / 14 := 
by
  -- This part we would provide a proof, but for now we will use sorry
  sorry

end probability_of_two_green_apples_l208_208358


namespace dig_second_hole_l208_208822

theorem dig_second_hole (w1 h1 d1 w2 d2 : ℕ) (extra_workers : ℕ) (h2 : ℕ) :
  w1 = 45 ∧ h1 = 8 ∧ d1 = 30 ∧ extra_workers = 65 ∧
  w2 = w1 + extra_workers ∧ d2 = 55 →
  360 * d2 / d1 = w2 * h2 →
  h2 = 6 :=
by
  intros h cond
  sorry

end dig_second_hole_l208_208822


namespace x_intercept_of_line_l208_208892

theorem x_intercept_of_line : ∃ x : ℚ, (6 * x, 0) = (35 / 6, 0) :=
by
  use 35 / 6
  sorry

end x_intercept_of_line_l208_208892


namespace initial_speed_solution_l208_208077

def initial_speed_problem : Prop :=
  ∃ V : ℝ, 
    (∀ t t_new : ℝ, 
      t = 300 / V ∧ 
      t_new = t - 4 / 5 ∧ 
      (∀ d d_remaining : ℝ, 
        d = V * (5 / 4) ∧ 
        d_remaining = 300 - d ∧ 
        t_new = (5 / 4) + d_remaining / (V + 16)) 
    ) → 
    V = 60

theorem initial_speed_solution : initial_speed_problem :=
by
  unfold initial_speed_problem
  sorry

end initial_speed_solution_l208_208077


namespace tangents_parallel_l208_208950

variable {R : Type*} [Field R]

-- Let f be a function from ratios to slopes
variable (φ : R -> R)

-- Given points (x, y) and (x₁, y₁) with corresponding conditions
variable (x x₁ y y₁ : R)

-- Conditions
def corresponding_points := y / x = y₁ / x₁
def homogeneous_diff_eqn := ∀ x y, (y / x) = φ (y / x)

-- Prove that the tangents are parallel
theorem tangents_parallel (h_corr : corresponding_points x x₁ y y₁)
  (h_diff_eqn : ∀ (x x₁ y y₁ : R), y' = φ (y / x) ∧ y₁' = φ (y₁ / x₁)) :
  y' = y₁' :=
by
  sorry

end tangents_parallel_l208_208950


namespace find_johns_allowance_l208_208840

variable (A : ℝ)  -- John's weekly allowance

noncomputable def johns_allowance : Prop :=
  let arcade_spent := (3 / 5) * A
  let remaining_after_arcade := (2 / 5) * A
  let toy_store_spent := (1 / 3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
  let final_spent := 0.88
  final_spent = remaining_after_toy_store → A = 3.30

theorem find_johns_allowance : johns_allowance A := by
  sorry

end find_johns_allowance_l208_208840


namespace arithmetic_sequence_common_difference_l208_208255

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℕ)
  (d : ℚ)
  (h_arith_seq : ∀ (n m : ℕ), (n > 0) → (m > 0) → (a n) / n - (a m) / m = (n - m) * d)
  (h_a3 : a 3 = 2)
  (h_a9 : a 9 = 12) :
  d = 1 / 9 ∧ a 12 = 20 :=
by 
  sorry

end arithmetic_sequence_common_difference_l208_208255


namespace product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l208_208392

theorem product_of_two_numbers_less_than_the_smaller_of_the_two_factors
    (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  a * b < min a b := 
sorry

end product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l208_208392


namespace price_increase_decrease_l208_208252

theorem price_increase_decrease (P : ℝ) (h : 0.84 * P = P * (1 - (x / 100)^2)) : x = 40 := by
  sorry

end price_increase_decrease_l208_208252


namespace range_g_l208_208333

def f (x: ℝ) : ℝ := 4 * x - 3
def g (x: ℝ) : ℝ := f (f (f (f (f x))))

theorem range_g (x: ℝ) (h: 0 ≤ x ∧ x ≤ 3) : -1023 ≤ g x ∧ g x ≤ 2049 :=
by
  sorry

end range_g_l208_208333


namespace part1_l208_208332

def U : Set ℝ := Set.univ
def P (a : ℝ) : Set ℝ := {x | 4 ≤ x ∧ x ≤ 7}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem part1 (a : ℝ) (P_def : P 3 = {x | 4 ≤ x ∧ x ≤ 7}) :
  ((U \ P a) ∩ Q = {x | -2 ≤ x ∧ x < 4}) := by
  sorry

end part1_l208_208332


namespace vasya_drives_fraction_l208_208034

theorem vasya_drives_fraction {a b c d s : ℝ} 
  (h1 : a = b / 2) 
  (h2 : c = a + d) 
  (h3 : d = s / 10) 
  (h4 : a + b + c + d = s) : 
  b / s = 0.4 :=
by
  sorry

end vasya_drives_fraction_l208_208034


namespace ratio_spaghetti_pizza_l208_208879

/-- Define the number of students who participated in the survey and their preferences --/
def students_surveyed : ℕ := 800
def lasagna_pref : ℕ := 150
def manicotti_pref : ℕ := 120
def ravioli_pref : ℕ := 180
def spaghetti_pref : ℕ := 200
def pizza_pref : ℕ := 150

/-- Prove the ratio of students who preferred spaghetti to those who preferred pizza is 4/3 --/
theorem ratio_spaghetti_pizza : (200 / 150 : ℚ) = 4 / 3 :=
by sorry

end ratio_spaghetti_pizza_l208_208879


namespace taxi_faster_than_truck_l208_208890

noncomputable def truck_speed : ℝ := 2.1 / 1
noncomputable def taxi_speed : ℝ := 10.5 / 4

theorem taxi_faster_than_truck :
  taxi_speed / truck_speed = 1.25 :=
by
  sorry

end taxi_faster_than_truck_l208_208890


namespace greatest_number_of_bouquets_l208_208253

def sara_red_flowers : ℕ := 16
def sara_yellow_flowers : ℕ := 24

theorem greatest_number_of_bouquets : Nat.gcd sara_red_flowers sara_yellow_flowers = 8 := by
  rfl

end greatest_number_of_bouquets_l208_208253


namespace blueberry_pies_count_l208_208844

-- Definitions and conditions
def total_pies := 30
def ratio_parts := 10
def pies_per_part := total_pies / ratio_parts
def blueberry_ratio := 3

-- Problem statement
theorem blueberry_pies_count :
  blueberry_ratio * pies_per_part = 9 := by
  -- The solution step that leads to the proof
  sorry

end blueberry_pies_count_l208_208844


namespace composite_function_evaluation_l208_208978

def f (x : ℕ) : ℕ := x * x
def g (x : ℕ) : ℕ := x + 2

theorem composite_function_evaluation : f (g 3) = 25 := by
  sorry

end composite_function_evaluation_l208_208978


namespace lambda_range_l208_208110

noncomputable def lambda (S1 S2 S3 S4: ℝ) (S: ℝ) : ℝ :=
  4 * (S1 + S2 + S3 + S4) / S

theorem lambda_range (S1 S2 S3 S4: ℝ) (S: ℝ) (h_max: S = max (max S1 S2) (max S3 S4)) :
  2 < lambda S1 S2 S3 S4 S ∧ lambda S1 S2 S3 S4 S ≤ 4 :=
by
  sorry

end lambda_range_l208_208110


namespace rubert_james_ratio_l208_208818

-- Definitions and conditions from a)
def adam_candies : ℕ := 6
def james_candies : ℕ := 3 * adam_candies
def rubert_candies (total_candies : ℕ) : ℕ := total_candies - (adam_candies + james_candies)
def total_candies : ℕ := 96

-- Statement to prove the ratio
theorem rubert_james_ratio : 
  (rubert_candies total_candies) / james_candies = 4 :=
by
  -- Proof is not required, so we leave it as sorry.
  sorry

end rubert_james_ratio_l208_208818


namespace find_number_l208_208001

theorem find_number :
  ∃ (x : ℤ), 
  x * (x + 6) = -8 ∧ 
  x^4 + (x + 6)^4 = 272 :=
by
  sorry

end find_number_l208_208001


namespace candy_ratio_l208_208851

theorem candy_ratio (chocolate_bars M_and_Ms marshmallows total_candies : ℕ)
  (h1 : chocolate_bars = 5)
  (h2 : M_and_Ms = 7 * chocolate_bars)
  (h3 : total_candies = 25 * 10)
  (h4 : marshmallows = total_candies - chocolate_bars - M_and_Ms) :
  marshmallows / M_and_Ms = 6 :=
by
  sorry

end candy_ratio_l208_208851


namespace no_prime_sum_seventeen_l208_208645

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_sum_seventeen :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 17 := by
  sorry

end no_prime_sum_seventeen_l208_208645


namespace dante_age_l208_208495

def combined_age (D : ℕ) : ℕ := D + D / 2 + (D + 1)

theorem dante_age :
  ∃ D : ℕ, combined_age D = 31 ∧ D = 12 :=
by
  sorry

end dante_age_l208_208495


namespace count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l208_208365

theorem count_of_numbers_less_than_100_divisible_by_2_but_not_by_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

theorem count_of_numbers_less_than_100_divisible_by_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 = 0 ∨ n % 3 = 0) (Finset.range 100)) = 66 :=
sorry

theorem count_of_numbers_less_than_100_not_divisible_by_either_2_or_3 :
  Finset.card (Finset.filter (λ n => n % 2 ≠ 0 ∧ n % 3 ≠ 0) (Finset.range 100)) = 33 :=
sorry

end count_of_numbers_less_than_100_divisible_by_2_but_not_by_3_count_of_numbers_less_than_100_divisible_by_2_or_3_count_of_numbers_less_than_100_not_divisible_by_either_2_or_3_l208_208365


namespace find_four_digit_number_l208_208959

theorem find_four_digit_number :
  ∃ (N : ℕ), 1000 ≤ N ∧ N < 10000 ∧ 
    (N % 131 = 112) ∧ 
    (N % 132 = 98) ∧ 
    N = 1946 :=
by
  sorry

end find_four_digit_number_l208_208959


namespace samia_walking_distance_l208_208554

noncomputable def total_distance (x : ℝ) : ℝ := 4 * x
noncomputable def biking_distance (x : ℝ) : ℝ := 3 * x
noncomputable def walking_distance (x : ℝ) : ℝ := x
noncomputable def biking_time (x : ℝ) : ℝ := biking_distance x / 12
noncomputable def walking_time (x : ℝ) : ℝ := walking_distance x / 4
noncomputable def total_time (x : ℝ) : ℝ := biking_time x + walking_time x

theorem samia_walking_distance : ∀ (x : ℝ), total_time x = 1 → walking_distance x = 2 :=
by
  sorry

end samia_walking_distance_l208_208554


namespace solution_set_ineq_l208_208631

theorem solution_set_ineq (x : ℝ) : 
  x * (x + 2) > 0 → abs x < 1 → 0 < x ∧ x < 1 := by
sorry

end solution_set_ineq_l208_208631


namespace class_student_count_l208_208805

-- Statement: Prove that under the given conditions, the number of students in the class is 19.
theorem class_student_count (n : ℕ) (avg_students_age : ℕ) (teacher_age : ℕ) (avg_with_teacher : ℕ):
  avg_students_age = 20 → 
  teacher_age = 40 → 
  avg_with_teacher = 21 → 
  21 * (n + 1) = 20 * n + 40 → 
  n = 19 := 
by 
  intros h1 h2 h3 h4 
  sorry

end class_student_count_l208_208805


namespace leonards_age_l208_208356

variable (L N J : ℕ)

theorem leonards_age (h1 : L = N - 4) (h2 : N = J / 2) (h3 : L + N + J = 36) : L = 6 := 
by 
  sorry

end leonards_age_l208_208356


namespace calc_101_cubed_expression_l208_208297

theorem calc_101_cubed_expression : 101^3 + 3 * (101^2) - 3 * 101 + 9 = 1060610 := 
by
  sorry

end calc_101_cubed_expression_l208_208297


namespace minimum_triangle_perimeter_l208_208168

def fractional_part (x : ℚ) : ℚ := x - ⌊x⌋

theorem minimum_triangle_perimeter (l m n : ℕ) (h1 : l > m) (h2 : m > n)
  (h3 : fractional_part (3^l / 10^4) = fractional_part (3^m / 10^4)) 
  (h4 : fractional_part (3^m / 10^4) = fractional_part (3^n / 10^4)) :
   l + m + n = 3003 := 
sorry

end minimum_triangle_perimeter_l208_208168


namespace purchase_price_of_first_commodity_l208_208896

-- Define the conditions
variable (price_first price_second : ℝ)
variable (h1 : price_first - price_second = 127)
variable (h2 : price_first + price_second = 827)

-- Prove the purchase price of the first commodity is $477
theorem purchase_price_of_first_commodity : price_first = 477 :=
by
  sorry

end purchase_price_of_first_commodity_l208_208896


namespace revenue_from_full_price_tickets_l208_208150

theorem revenue_from_full_price_tickets (f h p : ℕ) (h1 : f + h = 160) (h2 : f * p + h * (p / 2) = 2400) : f * p = 1600 :=
by
  sorry

end revenue_from_full_price_tickets_l208_208150


namespace jane_performance_l208_208964

theorem jane_performance :
  ∃ (p w e : ℕ), 
  p + w + e = 15 ∧ 
  2 * p + 4 * w + 6 * e = 66 ∧ 
  e = p + 4 ∧ 
  w = 11 :=
by
  sorry

end jane_performance_l208_208964


namespace maximum_marks_l208_208126

theorem maximum_marks (M : ℝ)
  (pass_threshold_percentage : ℝ := 33)
  (marks_obtained : ℝ := 92)
  (marks_failed_by : ℝ := 40) :
  (marks_obtained + marks_failed_by) = (pass_threshold_percentage / 100) * M → M = 400 := by
  sorry

end maximum_marks_l208_208126


namespace range_of_a_l208_208284
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 1 + a
noncomputable def g (x : ℝ) : ℝ := 3 * Real.log x

theorem range_of_a (h : ∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f x a = -g x) : 
  0 ≤ a ∧ a ≤ Real.exp 3 - 4 := 
sorry

end range_of_a_l208_208284


namespace denom_asymptotes_sum_l208_208250

theorem denom_asymptotes_sum (A B C : ℤ)
  (h_denom : ∀ x, (x = -1 ∨ x = 3 ∨ x = 4) → x^3 + A * x^2 + B * x + C = 0) :
  A + B + C = 11 := 
sorry

end denom_asymptotes_sum_l208_208250


namespace general_solution_of_differential_eq_l208_208133

theorem general_solution_of_differential_eq (x y : ℝ) (C : ℝ) :
  (x^2 - y^2) * (y * (1 - C^2)) - 2 * (y * x) * (x) = 0 → (x^2 + y^2 = C * y) := by
  sorry

end general_solution_of_differential_eq_l208_208133


namespace percentage_of_A_l208_208618

-- Define variables and assumptions
variables (A B : ℕ)
def total_payment := 580
def payment_B := 232

-- Define the proofs of the conditions provided in the problem
axiom total_payment_eq : A + B = total_payment
axiom B_eq : B = payment_B
noncomputable def percentage_paid_to_A := (A / B) * 100

-- Theorem to prove the percentage of the payment to A compared to B
theorem percentage_of_A : percentage_paid_to_A = 150 :=
by
 sorry

end percentage_of_A_l208_208618


namespace find_m_of_parallel_lines_l208_208500

theorem find_m_of_parallel_lines
  (m : ℝ) 
  (parallel : ∀ x y, (x - 2 * y + 5 = 0 → 2 * x + m * y - 5 = 0)) :
  m = -4 :=
sorry

end find_m_of_parallel_lines_l208_208500


namespace relationship_A_B_l208_208455

variable (x y : ℝ)

noncomputable def A : ℝ := (x + y) / (1 + x + y)

noncomputable def B : ℝ := (x / (1 + x)) + (y / (1 + y))

theorem relationship_A_B (hx : 0 < x) (hy : 0 < y) : A x y < B x y := sorry

end relationship_A_B_l208_208455


namespace water_wasted_in_one_hour_l208_208566

theorem water_wasted_in_one_hour:
  let drips_per_minute : ℕ := 10
  let drop_volume : ℝ := 0.05 -- volume in mL
  let minutes_in_hour : ℕ := 60
  drips_per_minute * drop_volume * minutes_in_hour = 30 := by
  sorry

end water_wasted_in_one_hour_l208_208566


namespace simple_interest_years_l208_208456

theorem simple_interest_years (r1 r2 t2 P1 P2 S : ℝ) (hP1: P1 = 3225) (hP2: P2 = 8000) (hr1: r1 = 0.08) (hr2: r2 = 0.15) (ht2: t2 = 2) (hCI : S = 2580) :
    S / 2 = (P1 * r1 * t) / 100 → t = 5 :=
by
  sorry

end simple_interest_years_l208_208456


namespace certain_number_is_120_l208_208298

theorem certain_number_is_120 : ∃ certain_number : ℤ, 346 * certain_number = 173 * 240 ∧ certain_number = 120 :=
by
  sorry

end certain_number_is_120_l208_208298


namespace max_value_x_plus_2y_l208_208672

variable (x y : ℝ)
variable (h1 : 4 * x + 3 * y ≤ 12)
variable (h2 : 3 * x + 6 * y ≤ 9)

theorem max_value_x_plus_2y : x + 2 * y ≤ 3 := by
  sorry

end max_value_x_plus_2y_l208_208672


namespace find_4a_add_c_find_2a_sub_2b_sub_c_l208_208924

variables {R : Type*} [CommRing R]

theorem find_4a_add_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  4 * a + c = 12 :=
sorry

theorem find_2a_sub_2b_sub_c (a b c : ℝ) (h : ∀ x : ℝ, (x^3 + a * x^2 + b * x + c) = (x^2 + 3 * x - 4) * (x + (a - 3) - b + 4 - c)) :
  2 * a - 2 * b - c = 14 :=
sorry

end find_4a_add_c_find_2a_sub_2b_sub_c_l208_208924


namespace largest_divisor_of_product_l208_208062

theorem largest_divisor_of_product (n : ℕ) (h : n % 3 = 0) : ∃ d, d = 288 ∧ ∀ n (h : n % 3 = 0), d ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)) := 
sorry

end largest_divisor_of_product_l208_208062


namespace jonathan_needs_12_bottles_l208_208639

noncomputable def fl_oz_to_liters (fl_oz : ℝ) : ℝ :=
  fl_oz / 33.8

noncomputable def liters_to_ml (liters : ℝ) : ℝ :=
  liters * 1000

noncomputable def num_bottles_needed (ml : ℝ) : ℝ :=
  ml / 150

theorem jonathan_needs_12_bottles :
  num_bottles_needed (liters_to_ml (fl_oz_to_liters 60)) = 12 := 
by
  sorry

end jonathan_needs_12_bottles_l208_208639


namespace value_of_x_l208_208781

def condition (x : ℝ) : Prop :=
  3 * x = (20 - x) + 20

theorem value_of_x : ∃ x : ℝ, condition x ∧ x = 10 := 
by
  sorry

end value_of_x_l208_208781


namespace solve_system_of_equations_l208_208611

theorem solve_system_of_equations (x y z t : ℤ) :
  (3 * x - 2 * y + 4 * z + 2 * t = 19) ∧ (5 * x + 6 * y - 2 * z + 3 * t = 23) →
  (x = 16 * z - 18 * y - 11) ∧ (t = 28 * y - 26 * z + 26) :=
by {
  sorry
}

end solve_system_of_equations_l208_208611


namespace bookseller_loss_l208_208118

theorem bookseller_loss (C S : ℝ) (h : 20 * C = 25 * S) : (C - S) / C * 100 = 20 := by
  sorry

end bookseller_loss_l208_208118


namespace jerry_total_hours_at_field_l208_208637
-- Import the entire necessary library

-- Lean statement of the problem
theorem jerry_total_hours_at_field 
  (games_per_daughter : ℕ)
  (practice_hours_per_game : ℕ)
  (game_duration : ℕ)
  (daughters : ℕ)
  (h1: games_per_daughter = 8)
  (h2: practice_hours_per_game = 4)
  (h3: game_duration = 2)
  (h4: daughters = 2)
 : (game_duration * games_per_daughter * daughters + practice_hours_per_game * games_per_daughter * daughters) = 96 :=
by
  -- Proof not required, so we skip it with sorry
  sorry

end jerry_total_hours_at_field_l208_208637


namespace avg_decreased_by_one_l208_208289

noncomputable def avg_decrease (n : ℕ) (average_initial : ℝ) (obs_new : ℝ) : ℝ :=
  (n * average_initial + obs_new) / (n + 1)

theorem avg_decreased_by_one (init_avg : ℝ) (obs_new : ℝ) (num_obs : ℕ)
  (h₁ : num_obs = 6)
  (h₂ : init_avg = 12)
  (h₃ : obs_new = 5) :
  init_avg - avg_decrease num_obs init_avg obs_new = 1 :=
by
  sorry

end avg_decreased_by_one_l208_208289


namespace sequence_eventually_congruent_mod_l208_208303

theorem sequence_eventually_congruent_mod (n : ℕ) (hn : n ≥ 1) : 
  ∃ N, ∀ m ≥ N, ∃ k, m = k * n + N ∧ (2^N.succ - 2^k) % n = 0 :=
by
  sorry

end sequence_eventually_congruent_mod_l208_208303


namespace radius_of_sphere_l208_208857

theorem radius_of_sphere 
  (shadow_length_sphere : ℝ)
  (stick_height : ℝ)
  (stick_shadow : ℝ)
  (parallel_sun_rays : Prop) 
  (tan_θ : ℝ) 
  (h1 : tan_θ = stick_height / stick_shadow)
  (h2 : tan_θ = shadow_length_sphere / 20) :
  shadow_length_sphere / 20 = 1/4 → shadow_length_sphere = 5 := by
  sorry

end radius_of_sphere_l208_208857


namespace evaluate_sets_are_equal_l208_208380

theorem evaluate_sets_are_equal :
  (-3^5) = ((-3)^5) ∧
  ¬ ((-2^2) = ((-2)^2)) ∧
  ¬ ((-4 * 2^3) = (-4^2 * 3)) ∧
  ¬ ((- (-3)^2) = (- (-2)^3)) :=
by
  sorry

end evaluate_sets_are_equal_l208_208380


namespace number_of_zeros_l208_208432

-- Definitions based on the conditions
def five_thousand := 5 * 10 ^ 3
def one_hundred := 10 ^ 2

-- The main theorem that we want to prove
theorem number_of_zeros : (five_thousand ^ 50) * (one_hundred ^ 2) = 10 ^ 154 * 5 ^ 50 := 
by sorry

end number_of_zeros_l208_208432


namespace tangent_intersection_product_l208_208577

theorem tangent_intersection_product (R r : ℝ) (A B C : ℝ) :
  (AC * CB = R * r) :=
sorry

end tangent_intersection_product_l208_208577


namespace points_in_rectangle_distance_l208_208270

/-- In a 3x4 rectangle, if 4 points are randomly located, 
    then the distance between at least two of them is at most 25/8. -/
theorem points_in_rectangle_distance (a b : ℝ) (h₁ : a = 3) (h₂ : b = 4)
  {points : Fin 4 → ℝ × ℝ}
  (h₃ : ∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ a)
  (h₄ : ∀ i, 0 ≤ (points i).2 ∧ (points i).2 ≤ b) :
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 25 / 8 := 
by
  sorry

end points_in_rectangle_distance_l208_208270


namespace fraction_simplification_l208_208228

variable {x y z : ℝ}
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z - z / x ≠ 0)

theorem fraction_simplification :
  (x^2 - 1 / y^2) / (z - z / x) = x / z :=
by
  sorry

end fraction_simplification_l208_208228


namespace seating_arrangements_l208_208845

theorem seating_arrangements :
  ∀ (chairs people : ℕ), 
  chairs = 8 → 
  people = 3 → 
  (∃ gaps : ℕ, gaps = 4) → 
  (∀ pos, pos = Nat.choose 3 4) → 
  pos = 24 :=
by
  intros chairs people h1 h2 h3 h4
  have gaps := 4
  have pos := Nat.choose 4 3
  sorry

end seating_arrangements_l208_208845


namespace black_area_fraction_after_three_changes_l208_208249

theorem black_area_fraction_after_three_changes
  (initial_black_area : ℚ)
  (change_factor : ℚ)
  (h1 : initial_black_area = 1)
  (h2 : change_factor = 2 / 3)
  : (change_factor ^ 3) * initial_black_area = 8 / 27 := 
by
  sorry

end black_area_fraction_after_three_changes_l208_208249


namespace bench_cost_l208_208670

theorem bench_cost (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
by {
  sorry
}

end bench_cost_l208_208670


namespace no_valid_pairs_l208_208016

theorem no_valid_pairs : ∀ (a b : ℕ), (a > 0) → (b > 0) → (a ≥ b) → 
  a * b + 125 = 30 * Nat.lcm a b + 24 * Nat.gcd a b + a % b → 
  false := by
  sorry

end no_valid_pairs_l208_208016


namespace quadratic_real_roots_condition_l208_208153

theorem quadratic_real_roots_condition (a : ℝ) :
  (∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) ↔ (a ≥ 1 ∧ a ≠ 5) :=
by
  sorry

end quadratic_real_roots_condition_l208_208153


namespace remainder_5n_div_3_l208_208149

theorem remainder_5n_div_3 (n : ℤ) (h : n % 3 = 2) : (5 * n) % 3 = 1 := by
  sorry

end remainder_5n_div_3_l208_208149


namespace sum_of_remainders_l208_208884

theorem sum_of_remainders (n : ℤ) (h₁ : n % 12 = 5) (h₂ : n % 3 = 2) (h₃ : n % 4 = 1) : 2 + 1 = 3 := by
  sorry

end sum_of_remainders_l208_208884


namespace remainder_of_3_pow_20_mod_7_l208_208762

theorem remainder_of_3_pow_20_mod_7 : (3^20) % 7 = 2 := by
  sorry

end remainder_of_3_pow_20_mod_7_l208_208762


namespace Dans_placed_scissors_l208_208086

theorem Dans_placed_scissors (initial_scissors placed_scissors total_scissors : ℕ) 
  (h1 : initial_scissors = 39) 
  (h2 : total_scissors = initial_scissors + placed_scissors) 
  (h3 : total_scissors = 52) : placed_scissors = 13 := 
by 
  sorry

end Dans_placed_scissors_l208_208086


namespace cost_of_each_candy_bar_l208_208877

-- Definitions of the conditions
def initial_amount : ℕ := 20
def final_amount : ℕ := 12
def number_of_candy_bars : ℕ := 4

-- Statement of the proof problem: prove the cost of each candy bar
theorem cost_of_each_candy_bar :
  (initial_amount - final_amount) / number_of_candy_bars = 2 := by
  sorry

end cost_of_each_candy_bar_l208_208877


namespace linear_function_decreasing_y_l208_208792

theorem linear_function_decreasing_y (x1 y1 y2 : ℝ) :
  y1 = -2 * x1 - 7 → y2 = -2 * (x1 - 1) - 7 → y1 < y2 := by
  intros h1 h2
  sorry

end linear_function_decreasing_y_l208_208792


namespace find_x_l208_208005

theorem find_x :
  ∃ x : ℝ, (2020 + x)^2 = x^2 ∧ x = -1010 :=
sorry

end find_x_l208_208005


namespace f_g_relationship_l208_208586

def f (x : ℝ) : ℝ := 3 * x ^ 2 - x + 1
def g (x : ℝ) : ℝ := 2 * x ^ 2 + x - 1

theorem f_g_relationship (x : ℝ) : f x > g x :=
by
  -- proof goes here
  sorry

end f_g_relationship_l208_208586


namespace no_linear_term_implies_equal_l208_208817

theorem no_linear_term_implies_equal (m n : ℝ) (h : (x : ℝ) → (x + m) * (x - n) - x^2 - (- mn) = 0) : m = n :=
by
  sorry

end no_linear_term_implies_equal_l208_208817


namespace circle_equation_line_intersect_circle_l208_208660

theorem circle_equation (x y : ℝ) : 
  y = x^2 - 4*x + 3 → (x = 0 ∧ y = 3) ∨ (y = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

theorem line_intersect_circle (m : ℝ) :
  (∀ x y : ℝ, (x + y + m = 0) ∨ ((x - 2)^2 + (y - 2)^2 = 5)) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + y₁ + m = 0) → ((x₁ - 2)^2 + (y₁ - 2)^2 = 5) →
    (x₂ + y₂ + m = 0) → ((x₂ - 2)^2 + (y₂ - 2)^2 = 5) →
    ((x₁ * x₂ + y₁ * y₂ = 0) → (m = -1 ∨ m = -3))) :=
sorry

end circle_equation_line_intersect_circle_l208_208660


namespace find_original_number_l208_208829

theorem find_original_number (x : ℝ) : ((x - 3) / 6) * 12 = 8 → x = 7 :=
by
  intro h
  sorry

end find_original_number_l208_208829


namespace wire_cutting_l208_208952

theorem wire_cutting : 
  ∃ (n : ℕ), n = 33 ∧ (∀ (x y : ℕ), 3 * x + y = 100 → x > 0 ∧ y > 0 → ∃ m : ℕ, m = n) :=
by {
  sorry
}

end wire_cutting_l208_208952


namespace total_fruits_in_baskets_l208_208532

structure Baskets where
  mangoes : ℕ
  pears : ℕ
  pawpaws : ℕ
  kiwis : ℕ
  lemons : ℕ

def taniaBaskets : Baskets := {
  mangoes := 18,
  pears := 10,
  pawpaws := 12,
  kiwis := 9,
  lemons := 9
}

theorem total_fruits_in_baskets : taniaBaskets.mangoes + taniaBaskets.pears + taniaBaskets.pawpaws + taniaBaskets.kiwis + taniaBaskets.lemons = 58 :=
by
  sorry

end total_fruits_in_baskets_l208_208532


namespace mutually_coprime_divisors_l208_208820

theorem mutually_coprime_divisors (a x y : ℕ) (h1 : a = 1944) 
  (h2 : ∃ d1 d2 d3, d1 * d2 * d3 = a ∧ gcd x y = 1 ∧ gcd x (x + y) = 1 ∧ gcd y (x + y) = 1) : 
  (x = 1 ∧ y = 2 ∧ x + y = 3) ∨ 
  (x = 1 ∧ y = 8 ∧ x + y = 9) ∨ 
  (x = 1 ∧ y = 3 ∧ x + y = 4) :=
sorry

end mutually_coprime_divisors_l208_208820


namespace louie_monthly_payment_l208_208629

noncomputable def compound_interest_payment (P : ℝ) (r : ℝ) (n : ℕ) (t_months : ℕ) : ℝ :=
  let t_years := t_months / 12
  let A := P * (1 + r / ↑n)^(↑n * t_years)
  A / t_months

theorem louie_monthly_payment : compound_interest_payment 1000 0.10 1 3 = 444 :=
by
  sorry

end louie_monthly_payment_l208_208629


namespace eccentricity_of_hyperbola_l208_208686

noncomputable def hyperbola (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

noncomputable def foci_condition (a b : ℝ) (c : ℝ) : Prop :=
  c = Real.sqrt (a^2 + b^2)

noncomputable def trisection_condition (a b c : ℝ) : Prop :=
  2 * c = 6 * a^2 / c

theorem eccentricity_of_hyperbola (a b c e : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hc : c = Real.sqrt (a^2 + b^2)) (ht : 2 * c = 6 * a^2 / c) :
  e = Real.sqrt 3 :=
by
  apply sorry

end eccentricity_of_hyperbola_l208_208686


namespace train_pass_bridge_time_l208_208307

noncomputable def totalDistance (trainLength bridgeLength : ℕ) : ℕ :=
  trainLength + bridgeLength

noncomputable def speedInMPerSecond (speedInKmPerHour : ℕ) : ℝ :=
  (speedInKmPerHour * 1000) / 3600

noncomputable def timeToPass (totalDistance : ℕ) (speedInMPerSecond : ℝ) : ℝ :=
  totalDistance / speedInMPerSecond

theorem train_pass_bridge_time
  (trainLength : ℕ) (bridgeLength : ℕ) (speedInKmPerHour : ℕ)
  (h_train : trainLength = 300)
  (h_bridge : bridgeLength = 115)
  (h_speed : speedInKmPerHour = 35) :
  timeToPass (totalDistance trainLength bridgeLength) (speedInMPerSecond speedInKmPerHour) = 42.7 :=
by
  sorry

end train_pass_bridge_time_l208_208307


namespace enrollment_difference_l208_208139

theorem enrollment_difference :
  let Varsity := 1680
  let Northwest := 1170
  let Central := 1840
  let Greenbriar := 1090
  let Eastside := 1450
  Central - Greenbriar = 750 := 
by
  intros Varsity Northwest Central Greenbriar Eastside
  -- calculate the difference
  have h1 : 750 = 750 := rfl
  sorry

end enrollment_difference_l208_208139


namespace ice_cream_remaining_l208_208485

def total_initial_scoops : ℕ := 3 * 10
def ethan_scoops : ℕ := 1 + 1
def lucas_danny_connor_scoops : ℕ := 2 * 3
def olivia_scoops : ℕ := 1 + 1
def shannon_scoops : ℕ := 2 * olivia_scoops
def total_consumed_scoops : ℕ := ethan_scoops + lucas_danny_connor_scoops + olivia_scoops + shannon_scoops
def remaining_scoops : ℕ := total_initial_scoops - total_consumed_scoops

theorem ice_cream_remaining : remaining_scoops = 16 := by
  sorry

end ice_cream_remaining_l208_208485


namespace avg_annual_reduction_l208_208673

theorem avg_annual_reduction (x : ℝ) (hx : (1 - x)^2 = 0.64) : x = 0.2 :=
by
  sorry

end avg_annual_reduction_l208_208673


namespace rectangular_park_length_l208_208428

noncomputable def length_of_rectangular_park
  (P : ℕ) (B : ℕ) (L : ℕ) : Prop :=
  (P = 1000) ∧ (B = 200) ∧ (P = 2 * (L + B)) → (L = 300)

theorem rectangular_park_length : length_of_rectangular_park 1000 200 300 :=
by {
  sorry
}

end rectangular_park_length_l208_208428


namespace Julie_and_Matt_ate_cookies_l208_208533

def initial_cookies : ℕ := 32
def remaining_cookies : ℕ := 23

theorem Julie_and_Matt_ate_cookies : initial_cookies - remaining_cookies = 9 :=
by
  sorry

end Julie_and_Matt_ate_cookies_l208_208533


namespace linear_dependency_k_l208_208838

theorem linear_dependency_k (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
    (c1 * 1 + c2 * 4 = 0) ∧
    (c1 * 2 + c2 * k = 0) ∧
    (c1 * 3 + c2 * 6 = 0)) ↔ k = 8 :=
by
  sorry

end linear_dependency_k_l208_208838


namespace lowest_possible_price_l208_208726

def typeADiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 15 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 20 / 100
  discountedPrice - additionalDiscount

def typeBDiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 25 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 15 / 100
  discountedPrice - additionalDiscount

def typeCDiscountedPrice (msrp : ℕ) : ℕ :=
  let regularDiscount := msrp * 30 / 100
  let discountedPrice := msrp - regularDiscount
  let additionalDiscount := discountedPrice * 10 / 100
  discountedPrice - additionalDiscount

def finalPrice (discountedPrice : ℕ) : ℕ :=
  let tax := discountedPrice * 7 / 100
  discountedPrice + tax

theorem lowest_possible_price : 
  min (finalPrice (typeADiscountedPrice 4500)) 
      (min (finalPrice (typeBDiscountedPrice 5500)) 
           (finalPrice (typeCDiscountedPrice 5000))) = 3274 :=
by {
  sorry
}

end lowest_possible_price_l208_208726


namespace find_k_l208_208747

theorem find_k (x y k : ℝ) 
  (h1 : 4 * x + 2 * y = 5 * k - 4) 
  (h2 : 2 * x + 4 * y = -1) 
  (h3 : x - y = 1) : 
  k = 1 := 
by sorry

end find_k_l208_208747


namespace division_multiplication_result_l208_208849

theorem division_multiplication_result : (180 / 6) * 3 = 90 := by
  sorry

end division_multiplication_result_l208_208849


namespace sum_as_common_fraction_l208_208241

/-- The sum of 0.2 + 0.04 + 0.006 + 0.0008 + 0.00010 as a common fraction -/
theorem sum_as_common_fraction : (0.2 + 0.04 + 0.006 + 0.0008 + 0.00010) = (12345 / 160000) := by
  sorry

end sum_as_common_fraction_l208_208241


namespace reciprocal_is_1_or_neg1_self_square_is_0_or_1_l208_208809

theorem reciprocal_is_1_or_neg1 (x : ℝ) (hx : x = 1 / x) :
  x = 1 ∨ x = -1 :=
sorry

theorem self_square_is_0_or_1 (x : ℝ) (hx : x = x^2) :
  x = 0 ∨ x = 1 :=
sorry

end reciprocal_is_1_or_neg1_self_square_is_0_or_1_l208_208809


namespace problem_statement_l208_208740

variables (p1 p2 p3 p4 : Prop)

theorem problem_statement (h_p1 : p1 = True)
                         (h_p2 : p2 = False)
                         (h_p3 : p3 = False)
                         (h_p4 : p4 = True) :
  (p1 ∧ p4) = True ∧
  (p1 ∧ p2) = False ∧
  (¬p2 ∨ p3) = True ∧
  (¬p3 ∨ ¬p4) = True :=
by
  sorry

end problem_statement_l208_208740


namespace eugene_pencils_after_giving_l208_208914

-- Define Eugene's initial number of pencils and the number of pencils given away.
def initial_pencils : ℝ := 51.0
def pencils_given : ℝ := 6.0

-- State the theorem that should be proved.
theorem eugene_pencils_after_giving : initial_pencils - pencils_given = 45.0 :=
by
  -- We would normally provide the proof steps here, but as per instructions, we'll use "sorry" to skip it.
  sorry

end eugene_pencils_after_giving_l208_208914


namespace lionel_initial_boxes_crackers_l208_208172

/--
Lionel went to the grocery store and bought some boxes of Graham crackers and 15 packets of Oreos. 
To make an Oreo cheesecake, Lionel needs 2 boxes of Graham crackers and 3 packets of Oreos. 
After making the maximum number of Oreo cheesecakes he can with the ingredients he bought, 
he had 4 boxes of Graham crackers left over. 

The number of boxes of Graham crackers Lionel initially bought is 14.
-/
theorem lionel_initial_boxes_crackers (G : ℕ) (h1 : G - 4 = 10) : G = 14 := 
by sorry

end lionel_initial_boxes_crackers_l208_208172


namespace result_of_subtraction_l208_208751

theorem result_of_subtraction (N : ℝ) (h1 : N = 100) : 0.80 * N - 20 = 60 :=
by
  sorry

end result_of_subtraction_l208_208751


namespace g_neg_one_l208_208191

variables {F : Type*} [Field F]

def odd_function (f : F → F) := ∀ x, f (-x) = -f x

variables (f : F → F) (g : F → F)

-- Given conditions
lemma given_conditions :
  (∀ x, f (-x) + (-x)^2 = -(f x + x^2)) ∧
  f 1 = 1 ∧
  (∀ x, g x = f x + 2) :=
sorry

-- Prove that g(-1) = -1
theorem g_neg_one :
  g (-1) = -1 :=
sorry

end g_neg_one_l208_208191


namespace minimal_withdrawals_proof_l208_208720

-- Defining the conditions
def red_marbles : ℕ := 200
def blue_marbles : ℕ := 300
def green_marbles : ℕ := 400

def max_red_withdrawal_per_time : ℕ := 1
def max_blue_withdrawal_per_time : ℕ := 2
def max_total_withdrawal_per_time : ℕ := 5

-- The target minimal number of withdrawals
def minimal_withdrawals : ℕ := 200

-- Lean statement of the proof problem
theorem minimal_withdrawals_proof :
  ∃ (w : ℕ), w = minimal_withdrawals ∧ 
    (∀ n, n ≤ w →
      (n = 200 ∧ 
       (∀ r b g, r ≤ max_red_withdrawal_per_time ∧ b ≤ max_blue_withdrawal_per_time ∧ (r + b + g) ≤ max_total_withdrawal_per_time))) :=
sorry

end minimal_withdrawals_proof_l208_208720


namespace students_per_class_l208_208461

theorem students_per_class (total_cupcakes : ℕ) (num_classes : ℕ) (pe_students : ℕ) 
  (h1 : total_cupcakes = 140) (h2 : num_classes = 3) (h3 : pe_students = 50) : 
  (total_cupcakes - pe_students) / num_classes = 30 :=
by
  sorry

end students_per_class_l208_208461


namespace solution_set_inequality_l208_208132

open Real

theorem solution_set_inequality (f : ℝ → ℝ) (h1 : f e = 0) (h2 : ∀ x > 0, x * deriv f x < 2) :
    ∀ x, 0 < x → x ≤ e → f x + 2 ≥ 2 * log x :=
by
  sorry

end solution_set_inequality_l208_208132


namespace problem_statement_l208_208919

theorem problem_statement (M N : ℕ) 
  (hM : M = 2020 / 5) 
  (hN : N = 2020 / 20) : 10 * M / N = 40 := 
by
  sorry

end problem_statement_l208_208919


namespace add_fractions_l208_208783

theorem add_fractions : (1 / 6 : ℚ) + (5 / 12) = 7 / 12 := 
by
  sorry

end add_fractions_l208_208783


namespace inverse_proportion_quadrants_l208_208622

theorem inverse_proportion_quadrants (x : ℝ) (y : ℝ) (h : y = 6/x) : 
  (x > 0 -> y > 0) ∧ (x < 0 -> y < 0) := 
sorry

end inverse_proportion_quadrants_l208_208622


namespace find_c_l208_208674

/-- Define the conditions given in the problem --/
def parabola_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def vertex_condition (a b c : ℝ) : Prop := 
  ∀ x, parabola_equation a b c x = a * (x - 3)^2 - 1

def passes_through_point (a b c : ℝ) : Prop := 
  parabola_equation a b c 1 = 5

/-- The main statement -/
theorem find_c (a b c : ℝ) 
  (h_vertex : vertex_condition a b c) 
  (h_point : passes_through_point a b c) :
  c = 12.5 :=
sorry

end find_c_l208_208674


namespace student_B_speed_l208_208352

theorem student_B_speed (d : ℝ) (t_diff : ℝ) (k : ℝ) (x : ℝ) 
  (h_dist : d = 12) (h_time_diff : t_diff = 1 / 6) (h_speed_ratio : k = 1.2)
  (h_eq : (d / x) - (d / (k * x)) = t_diff) : x = 12 :=
sorry

end student_B_speed_l208_208352


namespace exchange_silver_cards_l208_208142

theorem exchange_silver_cards : 
  (∃ red gold silver : ℕ,
    (∀ (r g s : ℕ), ((2 * g = 5 * r) ∧ (g = r + s) ∧ (r = 3) ∧ (g = 3) → s = 7))) :=
by
  sorry

end exchange_silver_cards_l208_208142


namespace population_correct_individual_correct_sample_correct_sample_size_correct_l208_208420

-- Definitions based on the problem conditions
def Population : Type := {s : String // s = "all seventh-grade students in the city"}
def Individual : Type := {s : String // s = "each seventh-grade student in the city"}
def Sample : Type := {s : String // s = "the 500 students that were drawn"}
def SampleSize : ℕ := 500

-- Prove given conditions
theorem population_correct (p : Population) : p.1 = "all seventh-grade students in the city" :=
by sorry

theorem individual_correct (i : Individual) : i.1 = "each seventh-grade student in the city" :=
by sorry

theorem sample_correct (s : Sample) : s.1 = "the 500 students that were drawn" :=
by sorry

theorem sample_size_correct : SampleSize = 500 :=
by sorry

end population_correct_individual_correct_sample_correct_sample_size_correct_l208_208420


namespace prove_fraction_identity_l208_208388

-- Define the conditions and the entities involved
variables {x y : ℝ} (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : 3 * x + y / 3 ≠ 0)

-- Formulate the theorem statement
theorem prove_fraction_identity :
  (3 * x + y / 3)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹) = (x * y)⁻¹ :=
sorry

end prove_fraction_identity_l208_208388


namespace value_makes_expression_undefined_l208_208154

theorem value_makes_expression_undefined (a : ℝ) : 
    (a^2 - 9 * a + 20 = 0) ↔ (a = 4 ∨ a = 5) :=
by
  sorry

end value_makes_expression_undefined_l208_208154


namespace tangent_planes_of_surface_and_given_plane_l208_208970

-- Define the surface and the given plane
def surface (x y z : ℝ) := (x^2 + 4 * y^2 + 9 * z^2 = 1)
def given_plane (x y z : ℝ) := (x + y + 2 * z = 1)

-- Define the tangent plane equations to be proved
def tangent_plane_1 (x y z : ℝ) := (x + y + 2 * z - (109 / (6 * Real.sqrt 61)) = 0)
def tangent_plane_2 (x y z : ℝ) := (x + y + 2 * z + (109 / (6 * Real.sqrt 61)) = 0)

-- The statement to be proved
theorem tangent_planes_of_surface_and_given_plane :
  ∀ x y z, surface x y z ∧ given_plane x y z →
    tangent_plane_1 x y z ∨ tangent_plane_2 x y z :=
sorry

end tangent_planes_of_surface_and_given_plane_l208_208970


namespace rectangle_length_l208_208196

theorem rectangle_length (side_of_square : ℕ) (width_of_rectangle : ℕ) (same_wire_length : ℕ) 
(side_eq : side_of_square = 12) (width_eq : width_of_rectangle = 6) 
(square_perimeter : same_wire_length = 4 * side_of_square) :
  ∃ (length_of_rectangle : ℕ), 2 * (length_of_rectangle + width_of_rectangle) = same_wire_length ∧ length_of_rectangle = 18 :=
by
  sorry

end rectangle_length_l208_208196


namespace geometric_sum_S30_l208_208949

theorem geometric_sum_S30 (S : ℕ → ℝ) (h1 : S 10 = 10) (h2 : S 20 = 30) : S 30 = 70 := 
by 
  sorry

end geometric_sum_S30_l208_208949


namespace frequency_of_hits_l208_208013

theorem frequency_of_hits (n m : ℕ) (h_n : n = 20) (h_m : m = 15) : (m / n : ℚ) = 0.75 := by
  sorry

end frequency_of_hits_l208_208013


namespace problem_statement_l208_208685

theorem problem_statement
  (m : ℝ) 
  (h : m + (1/m) = 5) :
  m^2 + (1 / m^2) + 4 = 27 :=
by
  -- Parameter types are chosen based on the context and problem description.
  sorry

end problem_statement_l208_208685


namespace correct_value_of_a_l208_208665

namespace ProofProblem

-- Condition 1: Definition of set M
def M : Set ℤ := {x | x^2 ≤ 1}

-- Condition 2: Definition of set N dependent on a parameter a
def N (a : ℤ) : Set ℤ := {a, a * a}

-- Question translated: Correct value of a such that M ∪ N = M
theorem correct_value_of_a (a : ℤ) : (M ∪ N a = M) → a = -1 :=
by
  sorry

end ProofProblem

end correct_value_of_a_l208_208665


namespace no_solution_for_inequalities_l208_208244

theorem no_solution_for_inequalities (x : ℝ) :
  ¬(5 * x^2 - 7 * x + 1 < 0 ∧ x^2 - 9 * x + 30 < 0) :=
sorry

end no_solution_for_inequalities_l208_208244


namespace remainder_27_pow_482_div_13_l208_208285

theorem remainder_27_pow_482_div_13 :
  27^482 % 13 = 1 :=
sorry

end remainder_27_pow_482_div_13_l208_208285


namespace correct_operation_l208_208858

variable (m n : ℝ)

-- Define the statement to be proved
theorem correct_operation : (-2 * m * n) ^ 2 = 4 * m ^ 2 * n ^ 2 :=
by sorry

end correct_operation_l208_208858


namespace smallest_solution_x4_50x2_576_eq_0_l208_208136

theorem smallest_solution_x4_50x2_576_eq_0 :
  ∃ x : ℝ, (x^4 - 50*x^2 + 576 = 0) ∧ ∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y :=
sorry

end smallest_solution_x4_50x2_576_eq_0_l208_208136


namespace rectangle_width_l208_208192

-- Conditions
def length (w : Real) : Real := 4 * w
def area (w : Real) : Real := w * length w

-- Theorem stating that the width of the rectangle is 5 inches if the area is 100 square inches
theorem rectangle_width (h : area w = 100) : w = 5 :=
sorry

end rectangle_width_l208_208192


namespace circle_tangent_to_yaxis_and_line_l208_208268

theorem circle_tangent_to_yaxis_and_line :
  (∃ C : ℝ → ℝ → Prop, 
    (∀ x y r : ℝ, C x y ↔ (x - 3) ^ 2 + (y - 2) ^ 2 = 9 ∨ (x + 1 / 3) ^ 2 + (y - 2) ^ 2 = 1 / 9) ∧ 
    (∀ y : ℝ, C 0 y → y = 2) ∧ 
    (∀ x y: ℝ, C x y → (∃ x1 : ℝ, 4 * x - 3 * y + 9 = 0 → 4 * x1 + 3 = 0))) :=
sorry

end circle_tangent_to_yaxis_and_line_l208_208268


namespace exists_set_X_gcd_condition_l208_208123

theorem exists_set_X_gcd_condition :
  ∃ (X : Finset ℕ), X.card = 2022 ∧
  (∀ (a b c : ℕ) (n : ℕ) (ha : a ∈ X) (hb : b ∈ X) (hc : c ∈ X) (hn_pos : 0 < n)
    (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c),
  Nat.gcd (a^n + b^n) c = 1) :=
sorry

end exists_set_X_gcd_condition_l208_208123


namespace range_for_a_l208_208381

variable (a : ℝ)

theorem range_for_a (h : ∀ x : ℝ, x^2 + 2 * x + a > 0) : 1 < a := 
sorry

end range_for_a_l208_208381


namespace games_given_away_correct_l208_208280

-- Define initial and remaining games
def initial_games : ℕ := 50
def remaining_games : ℕ := 35

-- Define the number of games given away
def games_given_away : ℕ := initial_games - remaining_games

-- Prove that the number of games given away is 15
theorem games_given_away_correct : games_given_away = 15 := by
  -- This is a placeholder for the actual proof
  sorry

end games_given_away_correct_l208_208280


namespace quadratic_polynomial_half_coefficient_l208_208167

theorem quadratic_polynomial_half_coefficient :
  ∃ b c : ℚ, ∀ x : ℤ, ∃ k : ℤ, (1/2 : ℚ) * (x^2 : ℚ) + b * (x : ℚ) + c = (k : ℚ) :=
by
  sorry

end quadratic_polynomial_half_coefficient_l208_208167


namespace find_subtracted_value_l208_208348

theorem find_subtracted_value (N V : ℕ) (h1 : N = 1376) (h2 : N / 8 - V = 12) : V = 160 :=
by
  sorry

end find_subtracted_value_l208_208348


namespace inequality_abc_l208_208097

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + a))) ≥ 3 / (1 + a * b * c) :=
by 
  sorry

end inequality_abc_l208_208097


namespace range_of_omega_l208_208164

noncomputable def f (ω x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f ω x = 0 → 
      (∃ x₁ x₂, x₁ ≠ x₂ ∧ 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ 
        0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧ f ω x₁ = 0 ∧ f ω x₂ = 0)) ↔ 2 ≤ ω ∧ ω < 4 :=
sorry

end range_of_omega_l208_208164


namespace smallest_four_consecutive_numbers_l208_208824

theorem smallest_four_consecutive_numbers (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 4574880) : n = 43 :=
sorry

end smallest_four_consecutive_numbers_l208_208824


namespace quadrilateral_diagonals_inequality_l208_208194

theorem quadrilateral_diagonals_inequality (a b c d e f : ℝ) :
  e^2 + f^2 ≤ b^2 + d^2 + 2 * a * c :=
by
  sorry

end quadrilateral_diagonals_inequality_l208_208194


namespace value_of_N_l208_208219

theorem value_of_N (N : ℕ) (h : Nat.choose N 5 = 231) : N = 11 := sorry

end value_of_N_l208_208219


namespace all_xi_equal_l208_208350

theorem all_xi_equal (P : Polynomial ℤ) (n : ℕ) (hn : n % 2 = 1) (x : Fin n → ℤ) 
  (hP : ∀ i : Fin n, P.eval (x i) = x ⟨i + 1, sorry⟩) : 
  ∀ i j : Fin n, x i = x j :=
by
  sorry

end all_xi_equal_l208_208350


namespace cosine_value_of_angle_between_vectors_l208_208688

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def cosine_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_value_of_angle_between_vectors :
  cosine_angle a b = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end cosine_value_of_angle_between_vectors_l208_208688


namespace geometric_sequence_common_ratio_l208_208302

theorem geometric_sequence_common_ratio {a : ℕ → ℝ} (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_start : a 1 < 0)
  (h_increasing : ∀ n, a n < a (n + 1)) : 0 < q ∧ q < 1 :=
by
  sorry

end geometric_sequence_common_ratio_l208_208302


namespace determine_x_l208_208621

variable (n p : ℝ)

-- Definitions based on conditions
def x (n : ℝ) : ℝ := 4 * n
def percentage_condition (n p : ℝ) : Prop := 2 * n + 3 = (p / 100) * 25

-- Statement to be proven
theorem determine_x (h : percentage_condition n p) : x n = 4 * n := by
  sorry

end determine_x_l208_208621


namespace sum_of_coordinates_D_l208_208156

structure Point where
  x : ℝ
  y : ℝ

def is_midpoint (M C D : Point) : Prop :=
  M = ⟨(C.x + D.x) / 2, (C.y + D.y) / 2⟩

def sum_of_coordinates (P : Point) : ℝ :=
  P.x + P.y

theorem sum_of_coordinates_D :
  ∀ (C M : Point), C = ⟨1/2, 3/2⟩ → M = ⟨2, 5⟩ →
  ∃ D : Point, is_midpoint M C D ∧ sum_of_coordinates D = 12 :=
by
  intros C M hC hM
  sorry

end sum_of_coordinates_D_l208_208156


namespace probability_failed_both_tests_eq_l208_208701

variable (total_students pass_test1 pass_test2 pass_both : ℕ)

def students_failed_both_tests (total pass1 pass2 both : ℕ) : ℕ :=
  total - (pass1 + pass2 - both)

theorem probability_failed_both_tests_eq 
  (h_total : total_students = 100)
  (h_pass1 : pass_test1 = 60)
  (h_pass2 : pass_test2 = 40)
  (h_pass_both : pass_both = 20) :
  students_failed_both_tests total_students pass_test1 pass_test2 pass_both / (total_students : ℚ) = 0.2 :=
by
  sorry

end probability_failed_both_tests_eq_l208_208701


namespace original_cost_of_car_l208_208179

theorem original_cost_of_car (C : ℝ)
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (profit_percent : ℝ)
  (h1 : repairs_cost = 14000)
  (h2 : selling_price = 72900)
  (h3 : profit_percent = 17.580645161290324)
  (h4 : profit_percent = ((selling_price - (C + repairs_cost)) / C) * 100) :
  C = 50075 := 
sorry

end original_cost_of_car_l208_208179


namespace equivalent_exponentiation_l208_208354

theorem equivalent_exponentiation (h : 64 = 8^2) : 8^15 / 64^3 = 8^9 :=
by
  sorry

end equivalent_exponentiation_l208_208354


namespace problem_solution_l208_208335

theorem problem_solution
  (a b c d : ℕ)
  (h1 : a^6 = b^5)
  (h2 : c^4 = d^3)
  (h3 : c - a = 25) :
  d - b = 561 :=
sorry

end problem_solution_l208_208335


namespace problem_statement_l208_208292

noncomputable def f : ℝ → ℝ := sorry

axiom func_condition : ∀ a b : ℝ, b^2 * f a = a^2 * f b
axiom f2_nonzero : f 2 ≠ 0

theorem problem_statement : (f 6 - f 3) / f 2 = 27 / 4 := 
by 
  sorry

end problem_statement_l208_208292


namespace trajectory_is_parabola_l208_208509

def distance_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ :=
|p.1 - a|

noncomputable def distance_to_point (p q : ℝ × ℝ) : ℝ :=
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def parabola_condition (P : ℝ × ℝ) : Prop :=
distance_to_line P (-1) + 1 = distance_to_point P (2, 0)

theorem trajectory_is_parabola : ∀ (P : ℝ × ℝ), parabola_condition P ↔
(P.1 + 1)^2 = (Real.sqrt ((P.1 - 2)^2 + P.2^2))^2 := 
by 
  sorry

end trajectory_is_parabola_l208_208509


namespace jim_total_payment_l208_208337

def lamp_cost : ℕ := 7
def bulb_cost : ℕ := lamp_cost - 4
def num_lamps : ℕ := 2
def num_bulbs : ℕ := 6

def total_cost : ℕ := (num_lamps * lamp_cost) + (num_bulbs * bulb_cost)

theorem jim_total_payment : total_cost = 32 := by
  sorry

end jim_total_payment_l208_208337


namespace bills_head_circumference_l208_208653

/-- Jack is ordering custom baseball caps for him and his two best friends, and we need to prove the circumference of Bill's head. -/
theorem bills_head_circumference (Jack : ℝ) (Charlie : ℝ) (Bill : ℝ)
  (h1 : Jack = 12)
  (h2 : Charlie = (1 / 2) * Jack + 9)
  (h3 : Bill = (2 / 3) * Charlie) :
  Bill = 10 :=
by sorry

end bills_head_circumference_l208_208653


namespace paris_total_study_hours_semester_l208_208903

-- Definitions
def weeks_in_semester := 15
def weekday_study_hours_per_day := 3
def weekdays_per_week := 5
def saturday_study_hours := 4
def sunday_study_hours := 5

-- Theorem statement
theorem paris_total_study_hours_semester :
  weeks_in_semester * (weekday_study_hours_per_day * weekdays_per_week + saturday_study_hours + sunday_study_hours) = 360 := 
sorry

end paris_total_study_hours_semester_l208_208903


namespace mod_inverse_35_36_l208_208116

theorem mod_inverse_35_36 : ∃ a : ℤ, 0 ≤ a ∧ a < 36 ∧ (35 * a) % 36 = 1 :=
  ⟨35, by sorry⟩

end mod_inverse_35_36_l208_208116


namespace dice_game_probability_l208_208806

def is_valid_roll (d1 d2 : ℕ) : Prop :=
  1 ≤ d1 ∧ d1 ≤ 6 ∧ 1 ≤ d2 ∧ d2 ≤ 6

def score (d1 d2 : ℕ) : ℕ :=
  max d1 d2

def favorable_outcomes : List (ℕ × ℕ) :=
  [ (1, 1), (1, 2), (2, 1), (2, 2), 
    (1, 3), (2, 3), (3, 1), (3, 2), (3, 3) ]

def total_outcomes : ℕ := 36

def favorable_count : ℕ := favorable_outcomes.length

theorem dice_game_probability : 
  (favorable_count : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end dice_game_probability_l208_208806


namespace buffy_whiskers_l208_208162

theorem buffy_whiskers :
  ∀ (Puffy Scruffy Buffy Juniper : ℕ),
    Juniper = 12 →
    Puffy = 3 * Juniper →
    Puffy = Scruffy / 2 →
    Buffy = (Juniper + Puffy + Scruffy) / 3 →
    Buffy = 40 :=
by
  intros Puffy Scruffy Buffy Juniper hJuniper hPuffy hScruffy hBuffy
  sorry

end buffy_whiskers_l208_208162


namespace total_amount_earned_l208_208211

theorem total_amount_earned (avg_price_per_pair : ℝ) (number_of_pairs : ℕ) (price : avg_price_per_pair = 9.8 ) (pairs : number_of_pairs = 50 ) : 
avg_price_per_pair * number_of_pairs = 490 := by
  -- Given conditions
  sorry

end total_amount_earned_l208_208211


namespace almonds_walnuts_ratio_l208_208330

-- Define the given weights and parts
def w_a : ℝ := 107.14285714285714
def w_m : ℝ := 150
def p_a : ℝ := 5

-- Now we will formulate the statement to prove the ratio of almonds to walnuts
theorem almonds_walnuts_ratio : 
  ∃ (p_w : ℝ), p_a / p_w = 5 / 2 :=
by
  -- It is given that p_a / p_w = 5 / 2, we need to find p_w
  sorry

end almonds_walnuts_ratio_l208_208330


namespace students_minus_rabbits_l208_208551

-- Define the number of students per classroom
def students_per_classroom : ℕ := 24

-- Define the number of rabbits per classroom
def rabbits_per_classroom : ℕ := 3

-- Define the number of classrooms
def number_of_classrooms : ℕ := 5

-- Define the total number of students and rabbits
def total_students : ℕ := students_per_classroom * number_of_classrooms
def total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms

-- The main statement to prove
theorem students_minus_rabbits :
  total_students - total_rabbits = 105 :=
by
  sorry

end students_minus_rabbits_l208_208551


namespace find_missing_surface_area_l208_208815

noncomputable def total_surface_area (areas : List ℕ) : ℕ :=
  areas.sum

def known_areas : List ℕ := [148, 46, 72, 28, 88, 126, 58]

def missing_surface_area : ℕ := 22

theorem find_missing_surface_area (areas : List ℕ) (total : ℕ) (missing : ℕ) :
  total_surface_area areas + missing = total →
  missing = 22 :=
by
  sorry

end find_missing_surface_area_l208_208815


namespace prove_ln10_order_l208_208730

def ln10_order_proof : Prop :=
  let a := Real.log 10
  let b := Real.log 100
  let c := (Real.log 10) ^ 2
  c > b ∧ b > a

theorem prove_ln10_order : ln10_order_proof := 
sorry

end prove_ln10_order_l208_208730


namespace amplitude_five_phase_shift_minus_pi_over_4_l208_208165

noncomputable def f (x : ℝ) : ℝ := 5 * Real.cos (x + (Real.pi / 4))

theorem amplitude_five : ∀ x : ℝ, 5 * Real.cos (x + (Real.pi / 4)) = f x :=
by
  sorry

theorem phase_shift_minus_pi_over_4 : ∀ x : ℝ, f x = 5 * Real.cos (x + (Real.pi / 4)) :=
by
  sorry

end amplitude_five_phase_shift_minus_pi_over_4_l208_208165


namespace minimum_questions_two_l208_208605

structure Person :=
  (is_liar : Bool)

structure Decagon :=
  (people : Fin 10 → Person)

def minimumQuestionsNaive (d : Decagon) : Nat :=
  match d with 
  -- add the logic here later
  | _ => sorry

theorem minimum_questions_two (d : Decagon) : minimumQuestionsNaive d = 2 :=
  sorry

end minimum_questions_two_l208_208605


namespace value_of_f_5_l208_208764

variable (f : ℕ → ℕ) (x y : ℕ)

theorem value_of_f_5 (h1 : f 2 = 50) (h2 : ∀ x, f x = 2 * x ^ 2 + y) : f 5 = 92 :=
by
  sorry

end value_of_f_5_l208_208764


namespace find_f_g_3_l208_208245

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x - 2

theorem find_f_g_3 : f (g 3) = 51 := 
by 
  sorry

end find_f_g_3_l208_208245


namespace total_cost_is_46_8_l208_208907

def price_pork : ℝ := 6
def price_chicken : ℝ := price_pork - 2
def price_beef : ℝ := price_chicken + 4
def price_lamb : ℝ := price_pork + 3

def quantity_chicken : ℝ := 3.5
def quantity_pork : ℝ := 1.2
def quantity_beef : ℝ := 2.3
def quantity_lamb : ℝ := 0.8

def total_cost : ℝ :=
    (quantity_chicken * price_chicken) +
    (quantity_pork * price_pork) +
    (quantity_beef * price_beef) +
    (quantity_lamb * price_lamb)

theorem total_cost_is_46_8 : total_cost = 46.8 :=
by
  sorry

end total_cost_is_46_8_l208_208907


namespace machine_A_production_rate_l208_208215

theorem machine_A_production_rate :
  ∀ (A B T_A T_B : ℝ),
    500 = A * T_A →
    500 = B * T_B →
    B = 1.25 * A →
    T_A = T_B + 15 →
    A = 100 / 15 :=
by
  intros A B T_A T_B hA hB hRate hTime
  sorry

end machine_A_production_rate_l208_208215


namespace handrail_length_is_17_point_3_l208_208572

noncomputable def length_of_handrail (turn : ℝ) (rise : ℝ) (radius : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  let arc_length := (turn / 360) * circumference
  Real.sqrt (rise^2 + arc_length^2)

theorem handrail_length_is_17_point_3 : length_of_handrail 270 10 3 = 17.3 :=
by 
  sorry

end handrail_length_is_17_point_3_l208_208572


namespace age_of_Rahim_l208_208231

theorem age_of_Rahim (R : ℕ) (h1 : ∀ (a : ℕ), a = (R + 1) → (a + 5) = (2 * R)) (h2 : ∀ (a : ℕ), a = (R + 1) → a = R + 1) :
  R = 6 := by
  sorry

end age_of_Rahim_l208_208231


namespace klay_to_draymond_ratio_l208_208595

-- Let us define the points earned by each player
def draymond_points : ℕ := 12
def curry_points : ℕ := 2 * draymond_points
def kelly_points : ℕ := 9
def durant_points : ℕ := 2 * kelly_points

-- Total points of the Golden State Team
def total_points_team : ℕ := 69

theorem klay_to_draymond_ratio :
  ∃ klay_points : ℕ,
    klay_points = total_points_team - (draymond_points + curry_points + kelly_points + durant_points) ∧
    klay_points / draymond_points = 1 / 2 :=
by
  sorry

end klay_to_draymond_ratio_l208_208595


namespace solve_equation_l208_208450

theorem solve_equation (x : ℝ) : 
  (x ^ (Real.log x / Real.log 2) = x^5 / 32) ↔ (x = 2^((5 + Real.sqrt 5) / 2) ∨ x = 2^((5 - Real.sqrt 5) / 2)) := 
by 
  sorry

end solve_equation_l208_208450


namespace g_at_3_l208_208175

def g (x : ℝ) : ℝ := 5 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_at_3 : g 3 = 79 := 
by 
  -- proof placeholder
  sorry

end g_at_3_l208_208175


namespace winnie_keeps_balloons_l208_208578

theorem winnie_keeps_balloons (red white green chartreuse friends total remainder : ℕ) (hRed : red = 17) (hWhite : white = 33) (hGreen : green = 65) (hChartreuse : chartreuse = 83) (hFriends : friends = 10) (hTotal : total = red + white + green + chartreuse) (hDiv : total % friends = remainder) : remainder = 8 :=
by
  have hTotal_eq : total = 198 := by
    sorry -- This would be the computation of 17 + 33 + 65 + 83
  have hRemainder_eq : 198 % 10 = remainder := by
    sorry -- This would involve the computation of the remainder
  exact sorry -- This would be the final proof that remainder = 8, tying all parts together

end winnie_keeps_balloons_l208_208578


namespace days_c_worked_l208_208233

theorem days_c_worked 
    (days_a : ℕ) (days_b : ℕ) (wage_ratio_a : ℚ) (wage_ratio_b : ℚ) (wage_ratio_c : ℚ)
    (total_earnings : ℚ) (wage_c : ℚ) :
    days_a = 16 →
    days_b = 9 →
    wage_ratio_a = 3 →
    wage_ratio_b = 4 →
    wage_ratio_c = 5 →
    wage_c = 71.15384615384615 →
    total_earnings = 1480 →
    ∃ days_c : ℕ, (total_earnings = (wage_ratio_a / wage_ratio_c * wage_c * days_a) + 
                                 (wage_ratio_b / wage_ratio_c * wage_c * days_b) + 
                                 (wage_c * days_c)) ∧ days_c = 4 :=
by
  intros
  sorry

end days_c_worked_l208_208233


namespace calculate_adults_in_play_l208_208583

theorem calculate_adults_in_play :
  ∃ A : ℕ, (11 * A = 49 + 50) := sorry

end calculate_adults_in_play_l208_208583


namespace evaluate_exponential_operations_l208_208146

theorem evaluate_exponential_operations (a : ℝ) :
  (2 * a^2 - a^2 ≠ 2) ∧
  (a^2 * a^4 = a^6) ∧
  ((a^2)^3 ≠ a^5) ∧
  (a^6 / a^2 ≠ a^3) := by
  sorry

end evaluate_exponential_operations_l208_208146


namespace eight_pow_three_eq_two_pow_nine_l208_208210

theorem eight_pow_three_eq_two_pow_nine : 8^3 = 2^9 := by
  sorry -- Proof is skipped

end eight_pow_three_eq_two_pow_nine_l208_208210


namespace solve_for_x_l208_208206

theorem solve_for_x :
  ∃ x : ℝ, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 :=
by
  sorry

end solve_for_x_l208_208206


namespace true_proposition_p_and_q_l208_208433

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

-- Define the proposition q
def q : Prop := ∃ x : ℝ, x^3 = 1 - x^2

-- Statement to prove the conjunction p ∧ q
theorem true_proposition_p_and_q : p ∧ q := 
by 
    sorry

end true_proposition_p_and_q_l208_208433


namespace boys_belong_to_other_communities_l208_208304

/-- In a school of 300 boys, if 44% are Muslims, 28% are Hindus, and 10% are Sikhs,
then the number of boys belonging to other communities is 54. -/
theorem boys_belong_to_other_communities
  (total_boys : ℕ)
  (percentage_muslims percentage_hindus percentage_sikhs : ℕ)
  (b : total_boys = 300)
  (m : percentage_muslims = 44)
  (h : percentage_hindus = 28)
  (s : percentage_sikhs = 10) :
  total_boys * ((100 - (percentage_muslims + percentage_hindus + percentage_sikhs)) / 100) = 54 := 
sorry

end boys_belong_to_other_communities_l208_208304


namespace total_books_l208_208757

def books_per_shelf_mystery : ℕ := 7
def books_per_shelf_picture : ℕ := 5
def books_per_shelf_sci_fi : ℕ := 8
def books_per_shelf_biography : ℕ := 6

def shelves_mystery : ℕ := 8
def shelves_picture : ℕ := 2
def shelves_sci_fi : ℕ := 3
def shelves_biography : ℕ := 4

theorem total_books :
  (books_per_shelf_mystery * shelves_mystery) + 
  (books_per_shelf_picture * shelves_picture) + 
  (books_per_shelf_sci_fi * shelves_sci_fi) + 
  (books_per_shelf_biography * shelves_biography) = 114 :=
by
  sorry

end total_books_l208_208757


namespace determine_number_l208_208499

def is_divisible_by_9 (n : ℕ) : Prop :=
  (n.digits 10).sum % 9 = 0

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def ten_power (n p : ℕ) : ℕ :=
  n * 10 ^ p

theorem determine_number (a b : ℕ) (h₁ : b = 0 ∨ b = 5)
  (h₂ : is_divisible_by_9 (7 + 2 + a + 3 + b))
  (h₃ : is_divisible_by_5 (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b)) :
  (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72630 ∨ 
   7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72135) :=
by sorry

end determine_number_l208_208499


namespace problem_1_problem_2_l208_208999

def f (x : ℝ) : ℝ := abs (2 * x + 3) + abs (2 * x - 1)

theorem problem_1 (x : ℝ) : (f x ≤ 5) ↔ (-7/4 ≤ x ∧ x ≤ 3/4) :=
by sorry

theorem problem_2 (m : ℝ) : (∃ x, f x < abs (m - 1)) ↔ (m > 5 ∨ m < -3) :=
by sorry

end problem_1_problem_2_l208_208999


namespace grace_age_l208_208208

/-- Grace's age calculation based on given family ages. -/
theorem grace_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ)
  (h1 : mother_age = 80)
  (h2 : grandmother_age = 2 * mother_age)
  (h3 : grace_age = (3 * grandmother_age) / 8) : grace_age = 60 :=
by
  sorry

end grace_age_l208_208208


namespace tom_has_65_fruits_left_l208_208675

def initial_fruits : ℕ := 40 + 70 + 30 + 15

def sold_oranges : ℕ := (1 / 4) * 40
def sold_apples : ℕ := (2 / 3) * 70
def sold_bananas : ℕ := (5 / 6) * 30
def sold_kiwis : ℕ := (60 / 100) * 15

def fruits_remaining : ℕ :=
  40 - sold_oranges +
  70 - sold_apples +
  30 - sold_bananas +
  15 - sold_kiwis

theorem tom_has_65_fruits_left :
  fruits_remaining = 65 := by
  sorry

end tom_has_65_fruits_left_l208_208675


namespace right_angled_triangle_l208_208833

-- Define the lengths of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove that these lengths form a right-angled triangle
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l208_208833


namespace base_form_exists_l208_208466

-- Definitions for three-digit number and its reverse in base g
def N (a b c g : ℕ) : ℕ := a * g^2 + b * g + c
def N_reverse (a b c g : ℕ) : ℕ := c * g^2 + b * g + a

-- The problem statement in Lean
theorem base_form_exists (a b c g : ℕ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : 0 < g)
    (h₅ : N a b c g = 2 * N_reverse a b c g) : ∃ k : ℕ, g = 3 * k + 2 ∧ k > 0 :=
by
  sorry

end base_form_exists_l208_208466


namespace neither_necessary_nor_sufficient_l208_208402

theorem neither_necessary_nor_sufficient (x : ℝ) : 
  ¬ ((x = 0) ↔ (x^2 - 2 * x = 0) ∧ (x ≠ 0 → x^2 - 2 * x ≠ 0) ∧ (x = 0 → x^2 - 2 * x = 0)) := 
sorry

end neither_necessary_nor_sufficient_l208_208402


namespace train_crossing_time_l208_208848

theorem train_crossing_time :
  ∀ (length_train1 length_train2 : ℕ) 
    (speed_train1_kmph speed_train2_kmph : ℝ), 
  length_train1 = 420 →
  speed_train1_kmph = 72 →
  length_train2 = 640 →
  speed_train2_kmph = 36 →
  (length_train1 + length_train2) / ((speed_train1_kmph - speed_train2_kmph) * (1000 / 3600)) = 106 :=
by
  intros
  sorry

end train_crossing_time_l208_208848


namespace roots_pure_imaginary_if_negative_real_k_l208_208362

theorem roots_pure_imaginary_if_negative_real_k (k : ℝ) (h_neg : k < 0) :
  (∃ (z : ℂ), 10 * z^2 - 3 * Complex.I * z - (k : ℂ) = 0 ∧ z.im ≠ 0 ∧ z.re = 0) :=
sorry

end roots_pure_imaginary_if_negative_real_k_l208_208362


namespace Samanta_points_diff_l208_208947

variables (Samanta Mark Eric : ℕ)

/-- In a game, Samanta has some more points than Mark, Mark has 50% more points than Eric,
Eric has 6 points, and Samanta, Mark, and Eric have a total of 32 points. Prove that Samanta
has 8 more points than Mark. -/
theorem Samanta_points_diff 
    (h1 : Mark = Eric + Eric / 2) 
    (h2 : Eric = 6) 
    (h3 : Samanta + Mark + Eric = 32)
    : Samanta - Mark = 8 :=
sorry

end Samanta_points_diff_l208_208947


namespace sandy_total_money_l208_208336

def half_dollar_value := 0.5
def quarter_value := 0.25
def dime_value := 0.1
def nickel_value := 0.05
def dollar_value := 1.0

def monday_total := 12 * half_dollar_value + 5 * quarter_value + 10 * dime_value
def tuesday_total := 8 * half_dollar_value + 15 * quarter_value + 5 * dime_value
def wednesday_total := 3 * dollar_value + 4 * half_dollar_value + 10 * quarter_value + 7 * nickel_value
def thursday_total := 5 * dollar_value + 6 * half_dollar_value + 8 * quarter_value + 5 * dime_value + 12 * nickel_value
def friday_total := 2 * dollar_value + 7 * half_dollar_value + 20 * nickel_value + 25 * dime_value

def total_amount := monday_total + tuesday_total + wednesday_total + thursday_total + friday_total

theorem sandy_total_money : total_amount = 44.45 := by
  sorry

end sandy_total_money_l208_208336


namespace expression_eq_l208_208052

theorem expression_eq (x : ℝ) : 
    (x + 1)^4 + 4 * (x + 1)^3 + 6 * (x + 1)^2 + 4 * (x + 1) + 1 = (x + 2)^4 := 
  sorry

end expression_eq_l208_208052


namespace cone_dimensions_l208_208058

noncomputable def cone_height (r_sector : ℝ) (r_cone_base : ℝ) : ℝ :=
  Real.sqrt (r_sector^2 - r_cone_base^2)

noncomputable def cone_volume (radius : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * Real.pi * radius^2 * height

theorem cone_dimensions 
  (r_circle : ℝ) (num_sectors : ℕ) (r_cone_base : ℝ) :
  r_circle = 12 → num_sectors = 4 → r_cone_base = 3 → 
  cone_height r_circle r_cone_base = 3 * Real.sqrt 15 ∧ 
  cone_volume r_cone_base (cone_height r_circle r_cone_base) = 9 * Real.pi * Real.sqrt 15 :=
by
  intros
  sorry

end cone_dimensions_l208_208058


namespace problem1_problem2_l208_208010

-- Definition for the first problem: determine the number of arrangements when no box is empty and ball 3 is in box B
def arrangements_with_ball3_in_B_and_no_empty_box : ℕ :=
  12

theorem problem1 : arrangements_with_ball3_in_B_and_no_empty_box = 12 :=
  by
    sorry

-- Definition for the second problem: determine the number of arrangements when ball 1 is not in box A and ball 2 is not in box B
def arrangements_with_ball1_not_in_A_and_ball2_not_in_B : ℕ :=
  36

theorem problem2 : arrangements_with_ball1_not_in_A_and_ball2_not_in_B = 36 :=
  by
    sorry

end problem1_problem2_l208_208010


namespace solution_set_unique_line_l208_208125

theorem solution_set_unique_line (x y : ℝ) : 
  (x - 2 * y = 1 ∧ x^3 - 6 * x * y - 8 * y^3 = 1) ↔ (y = (x - 1) / 2) := 
by
  sorry

end solution_set_unique_line_l208_208125


namespace simplify_expression_l208_208835

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (a ^ (7 / 3) - 2 * a ^ (5 / 3) * b ^ (2 / 3) + a * b ^ (4 / 3)) / 
  (a ^ (5 / 3) - a ^ (4 / 3) * b ^ (1 / 3) - a * b ^ (2 / 3) + a ^ (2 / 3) * b) / 
  a ^ (1 / 3) =
  a ^ (1 / 3) + b ^ (1 / 3) :=
sorry

end simplify_expression_l208_208835


namespace find_z_add_inv_y_l208_208355

theorem find_z_add_inv_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 31) : z + 1 / y = 5 / 27 := by
  sorry

end find_z_add_inv_y_l208_208355


namespace squared_sum_l208_208340

theorem squared_sum (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 :=
by
  sorry

end squared_sum_l208_208340


namespace line_length_l208_208995

theorem line_length (n : ℕ) (d : ℤ) (h1 : n = 51) (h2 : d = 3) : 
  (n - 1) * d = 150 := sorry

end line_length_l208_208995


namespace find_quadruple_l208_208698

/-- Problem Statement:
Given distinct positive integers a, b, c, and d such that a + b = c * d and a * b = c + d,
find the quadruple (a, b, c, d) that meets these conditions.
-/

theorem find_quadruple :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
            0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
            (a + b = c * d) ∧ (a * b = c + d) ∧
            ((a, b, c, d) = (1, 5, 3, 2) ∨ (a, b, c, d) = (1, 5, 2, 3) ∨
             (a, b, c, d) = (5, 1, 3, 2) ∨ (a, b, c, d) = (5, 1, 2, 3) ∨
             (a, b, c, d) = (2, 3, 1, 5) ∨ (a, b, c, d) = (3, 2, 1, 5) ∨
             (a, b, c, d) = (2, 3, 5, 1) ∨ (a, b, c, d) = (3, 2, 5, 1)) :=
sorry

end find_quadruple_l208_208698


namespace class_average_correct_l208_208584

-- Define the constants as per the problem data
def total_students : ℕ := 30
def students_group_1 : ℕ := 24
def students_group_2 : ℕ := 6
def avg_score_group_1 : ℚ := 85 / 100  -- 85%
def avg_score_group_2 : ℚ := 92 / 100  -- 92%

-- Calculate total scores and averages based on the defined constants
def total_score_group_1 : ℚ := students_group_1 * avg_score_group_1
def total_score_group_2 : ℚ := students_group_2 * avg_score_group_2
def total_class_score : ℚ := total_score_group_1 + total_score_group_2
def class_average : ℚ := total_class_score / total_students

-- Goal: Prove that class_average is 86.4%
theorem class_average_correct : class_average = 86.4 / 100 := sorry

end class_average_correct_l208_208584


namespace intersection_eq_l208_208291

open Set

-- Define the sets M and N
def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℤ := {-3, -1, 1, 3, 5}

-- The goal is to prove that M ∩ N = {-1, 1, 3}
theorem intersection_eq : M ∩ N = {-1, 1, 3} :=
  sorry

end intersection_eq_l208_208291


namespace algebraic_expression_value_l208_208178

/-- Given \( x^2 - 5x - 2006 = 0 \), prove that the expression \(\frac{(x-2)^3 - (x-1)^2 + 1}{x-2}\) is equal to 2010. -/
theorem algebraic_expression_value (x : ℝ) (h: x^2 - 5 * x - 2006 = 0) :
  ( (x - 2)^3 - (x - 1)^2 + 1 ) / (x - 2) = 2010 :=
by
  sorry

end algebraic_expression_value_l208_208178


namespace hcf_of_numbers_l208_208095

theorem hcf_of_numbers (x y : ℕ) (hcf lcm : ℕ) 
    (h_sum : x + y = 45) 
    (h_lcm : lcm = 100)
    (h_reciprocal_sum : 1 / (x : ℝ) + 1 / (y : ℝ) = 0.3433333333333333) :
    hcf = 1 :=
by
  sorry

end hcf_of_numbers_l208_208095


namespace symmetric_line_eq_x_axis_l208_208934

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * (-y) + 5 = 0) :=
by
  sorry

end symmetric_line_eq_x_axis_l208_208934


namespace height_relationship_l208_208050

theorem height_relationship 
  (r₁ h₁ r₂ h₂ : ℝ)
  (h_volume : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius : r₂ = (6/5) * r₁) :
  h₁ = 1.44 * h₂ :=
by
  sorry

end height_relationship_l208_208050


namespace distance_between_foci_of_ellipse_l208_208565

-- Define the parameters a^2 and b^2 according to the problem
def a_sq : ℝ := 25
def b_sq : ℝ := 16

-- State the problem
theorem distance_between_foci_of_ellipse : 
  (2 * Real.sqrt (a_sq - b_sq)) = 6 := by
  -- Proof content is skipped 
  sorry

end distance_between_foci_of_ellipse_l208_208565


namespace fraction_of_project_completed_in_one_hour_l208_208262

noncomputable def fraction_of_project_completed_together (a b : ℝ) : ℝ :=
  (1 / a) + (1 / b)

theorem fraction_of_project_completed_in_one_hour (a b : ℝ) :
  fraction_of_project_completed_together a b = (1 / a) + (1 / b) := by
  sorry

end fraction_of_project_completed_in_one_hour_l208_208262


namespace jump_difference_l208_208738

-- Definitions based on conditions
def grasshopper_jump : ℕ := 13
def frog_jump : ℕ := 11

-- Proof statement
theorem jump_difference : grasshopper_jump - frog_jump = 2 := by
  sorry

end jump_difference_l208_208738


namespace area_of_Q1Q3Q5Q7_l208_208267

def regular_octagon_apothem : ℝ := 3

def area_of_quadrilateral (a : ℝ) : Prop :=
  let s := 6 * (1 - Real.sqrt 2)
  let side_length := s * Real.sqrt 2
  let area := side_length ^ 2
  area = 72 * (3 - 2 * Real.sqrt 2)

theorem area_of_Q1Q3Q5Q7 : area_of_quadrilateral regular_octagon_apothem :=
  sorry

end area_of_Q1Q3Q5Q7_l208_208267


namespace james_units_per_semester_l208_208440

theorem james_units_per_semester
  (cost_per_unit : ℕ)
  (total_cost : ℕ)
  (num_semesters : ℕ)
  (payment_per_semester : ℕ)
  (units_per_semester : ℕ)
  (H1 : cost_per_unit = 50)
  (H2 : total_cost = 2000)
  (H3 : num_semesters = 2)
  (H4 : payment_per_semester = total_cost / num_semesters)
  (H5 : units_per_semester = payment_per_semester / cost_per_unit) :
  units_per_semester = 20 :=
sorry

end james_units_per_semester_l208_208440


namespace tetrahedron_volume_correct_l208_208889

noncomputable def tetrahedron_volume (a b c : ℝ) : ℝ :=
  (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2))

theorem tetrahedron_volume_correct (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a^2 + b^2 = c^2) :
  tetrahedron_volume a b c = (1 / (6 * Real.sqrt 2)) * Real.sqrt ((a^2 + b^2 - c^2) * (b^2 + c^2 - a^2) * (c^2 + a^2 - b^2)) :=
by
  sorry

end tetrahedron_volume_correct_l208_208889


namespace probability_not_greater_than_two_l208_208072

theorem probability_not_greater_than_two : 
  let cards := [1, 2, 3, 4]
  let favorable_cards := [1, 2]
  let total_scenarios := cards.length
  let favorable_scenarios := favorable_cards.length
  let prob := favorable_scenarios / total_scenarios
  prob = 1 / 2 :=
by
  sorry

end probability_not_greater_than_two_l208_208072


namespace new_concentration_is_37_percent_l208_208788

-- Conditions
def capacity_vessel_1 : ℝ := 2 -- litres
def alcohol_concentration_vessel_1 : ℝ := 0.35

def capacity_vessel_2 : ℝ := 6 -- litres
def alcohol_concentration_vessel_2 : ℝ := 0.50

def total_poured_liquid : ℝ := 8 -- litres
def final_vessel_capacity : ℝ := 10 -- litres

-- Question: Prove the new concentration of the mixture
theorem new_concentration_is_37_percent :
  (alcohol_concentration_vessel_1 * capacity_vessel_1 + alcohol_concentration_vessel_2 * capacity_vessel_2) / final_vessel_capacity = 0.37 := by
  sorry

end new_concentration_is_37_percent_l208_208788


namespace k_greater_than_inv_e_l208_208060

theorem k_greater_than_inv_e (k : ℝ) (x : ℝ) (hx_pos : 0 < x) (hcond : k * (Real.exp (k * x) + 1) - (1 + (1 / x)) * Real.log x > 0) : 
  k > 1 / Real.exp 1 :=
sorry

end k_greater_than_inv_e_l208_208060


namespace restaurant_total_glasses_l208_208229

theorem restaurant_total_glasses (x y t : ℕ) 
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15)
  (h3 : t = 12 * x + 16 * y) : 
  t = 480 :=
by 
  -- Proof omitted
  sorry

end restaurant_total_glasses_l208_208229


namespace standard_deviation_of_applicants_l208_208251

theorem standard_deviation_of_applicants (σ : ℕ) 
  (h1 : ∃ avg : ℕ, avg = 30)
  (h2 : ∃ n : ℕ, n = 17)
  (h3 : ∃ range_count : ℕ, range_count = (30 + σ) - (30 - σ) + 1) :
  σ = 8 :=
by
  sorry

end standard_deviation_of_applicants_l208_208251


namespace kelly_initial_sony_games_l208_208885

def nintendo_games : ℕ := 46
def sony_games_given_away : ℕ := 101
def sony_games_left : ℕ := 31

theorem kelly_initial_sony_games :
  sony_games_given_away + sony_games_left = 132 :=
by
  sorry

end kelly_initial_sony_games_l208_208885


namespace anna_interest_l208_208158

noncomputable def interest_earned (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n - P

theorem anna_interest : interest_earned 2000 0.08 5 = 938.66 := by
  sorry

end anna_interest_l208_208158


namespace A_leaves_after_2_days_l208_208497

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 30
noncomputable def C_work_rate : ℚ := 1 / 10
noncomputable def C_days_work : ℚ := 4
noncomputable def total_days_work : ℚ := 15

theorem A_leaves_after_2_days (x : ℚ) : 
  2 / 5 + x / 12 + (15 - x) / 30 = 1 → x = 2 :=
by
  intro h
  sorry

end A_leaves_after_2_days_l208_208497


namespace andrea_still_needs_rhinestones_l208_208529

def total_rhinestones_needed : ℕ := 45
def rhinestones_bought : ℕ := total_rhinestones_needed / 3
def rhinestones_found : ℕ := total_rhinestones_needed / 5
def rhinestones_total_have : ℕ := rhinestones_bought + rhinestones_found
def rhinestones_still_needed : ℕ := total_rhinestones_needed - rhinestones_total_have

theorem andrea_still_needs_rhinestones : rhinestones_still_needed = 21 := by
  rfl

end andrea_still_needs_rhinestones_l208_208529


namespace first_ship_rescued_boy_l208_208375

noncomputable def river_speed : ℝ := 3 -- River speed is 3 km/h

-- Define the speeds of the ships
def ship1_speed_upstream : ℝ := 4 
def ship2_speed_upstream : ℝ := 6 
def ship3_speed_upstream : ℝ := 10 

-- Define the distance downstream where the boy was found
def boy_distance_from_bridge : ℝ := 6

-- Define the equation for the first ship
def first_ship_equation (c : ℝ) : Prop := (10 - c) / (4 + c) = 1 + 6 / c

-- The problem to prove:
theorem first_ship_rescued_boy : first_ship_equation river_speed :=
by sorry

end first_ship_rescued_boy_l208_208375


namespace evaluate_expression_l208_208400

theorem evaluate_expression : 
  (10^8 / (2.5 * 10^5) * 3) = 1200 :=
by
  sorry

end evaluate_expression_l208_208400


namespace sum_coeff_eq_neg_two_l208_208893

theorem sum_coeff_eq_neg_two (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ) :
  (1 - 2*x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 →
  a = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = -2 :=
by
  sorry

end sum_coeff_eq_neg_two_l208_208893


namespace pure_imaginary_implies_a_neg_one_l208_208630

theorem pure_imaginary_implies_a_neg_one (a : ℝ) 
  (h_pure_imaginary : ∃ (y : ℝ), z = 0 + y * I) : 
  z = a + 1 - a * I → a = -1 :=
by
  sorry

end pure_imaginary_implies_a_neg_one_l208_208630


namespace amy_total_score_l208_208373

theorem amy_total_score :
  let points_per_treasure := 4
  let treasures_first_level := 6
  let treasures_second_level := 2
  let score_first_level := treasures_first_level * points_per_treasure
  let score_second_level := treasures_second_level * points_per_treasure
  let total_score := score_first_level + score_second_level
  total_score = 32 := by
sorry

end amy_total_score_l208_208373


namespace average_speed_for_remaining_part_l208_208237

theorem average_speed_for_remaining_part (D : ℝ) (v : ℝ) 
  (h1 : 0.8 * D / 80 + 0.2 * D / v = D / 50) : v = 20 :=
sorry

end average_speed_for_remaining_part_l208_208237


namespace tangent_line_eq_extreme_values_interval_l208_208186

noncomputable def f (x : ℝ) (a b : ℝ) := a * x^3 + b * x + 2

theorem tangent_line_eq (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  9 * 1 + (f 1 a b) = 0 :=
sorry

theorem extreme_values_interval (a b : ℝ) (h1 : 3 * a * 2^2 + b = 0) (h2 : a * 2^3 + b * 2 + 2 = -14) :
  ∃ (min_val max_val : ℝ), 
    min_val = -14 ∧ f 2 a b = min_val ∧
    max_val = 18 ∧ f (-2) a b = max_val ∧
    ∀ x, (x ∈ Set.Icc (-3 : ℝ) 3 → f x a b ≥ min_val ∧ f x a b ≤ max_val) :=
sorry

end tangent_line_eq_extreme_values_interval_l208_208186


namespace circle_tangent_to_ellipse_l208_208841

theorem circle_tangent_to_ellipse {r : ℝ} 
  (h1: ∀ p: ℝ × ℝ, p ≠ (0, 0) → ((p.1 - r)^2 + p.2^2 = r^2 → p.1^2 + 4 * p.2^2 = 8))
  (h2: ∃ p: ℝ × ℝ, p ≠ (0, 0) ∧ ((p.1 - r)^2 + p.2^2 = r^2 ∧ p.1^2 + 4 * p.2^2 = 8)):
  r = Real.sqrt (3 / 2) :=
by
  sorry

end circle_tangent_to_ellipse_l208_208841


namespace solution_set_ln_inequality_l208_208661

noncomputable def f (x : ℝ) := Real.cos x - 4 * x^2

theorem solution_set_ln_inequality :
  {x : ℝ | 0 < x ∧ x < Real.exp (-Real.pi / 2)} ∪ {x : ℝ | x > Real.exp (Real.pi / 2)} =
  {x : ℝ | f (Real.log x) + Real.pi^2 > 0} :=
by
  sorry

end solution_set_ln_inequality_l208_208661


namespace max_constant_k_l208_208374

theorem max_constant_k (x y : ℤ) : 4 * x^2 + y^2 + 1 ≥ 3 * x * (y + 1) :=
sorry

end max_constant_k_l208_208374


namespace min_even_integers_least_one_l208_208459

theorem min_even_integers_least_one (x y a b m n o : ℤ) 
  (h1 : x + y = 29)
  (h2 : x + y + a + b = 47)
  (h3 : x + y + a + b + m + n + o = 66) :
  ∃ e : ℕ, (e = 1) := by
sorry

end min_even_integers_least_one_l208_208459


namespace problem_statement_l208_208389

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem problem_statement :
  ¬ is_pythagorean_triple 2 3 4 ∧ 
  is_pythagorean_triple 3 4 5 ∧ 
  is_pythagorean_triple 6 8 10 ∧ 
  is_pythagorean_triple 5 12 13 :=
by 
  constructor
  sorry
  constructor
  sorry
  constructor
  sorry
  sorry

end problem_statement_l208_208389


namespace total_blocks_fell_l208_208310

-- Definitions based on the conditions
def first_stack_height := 7
def second_stack_height := first_stack_height + 5
def third_stack_height := second_stack_height + 7

def first_stack_fallen_blocks := first_stack_height  -- All blocks fell down
def second_stack_fallen_blocks := second_stack_height - 2  -- 2 blocks left standing
def third_stack_fallen_blocks := third_stack_height - 3  -- 3 blocks left standing

-- Total fallen blocks
def total_fallen_blocks := first_stack_fallen_blocks + second_stack_fallen_blocks + third_stack_fallen_blocks

-- Theorem to prove the total number of fallen blocks
theorem total_blocks_fell : total_fallen_blocks = 33 :=
by
  -- Proof omitted, statement given as required
  sorry

end total_blocks_fell_l208_208310


namespace cats_in_house_l208_208214

-- Define the conditions
def total_cats (C : ℕ) : Prop :=
  let num_white_cats := 2
  let num_black_cats := C / 4
  let num_grey_cats := 10
  C = num_white_cats + num_black_cats + num_grey_cats

-- State the theorem
theorem cats_in_house : ∃ C : ℕ, total_cats C ∧ C = 16 := 
by
  sorry

end cats_in_house_l208_208214


namespace find_x_eq_eight_l208_208331

theorem find_x_eq_eight (x : ℕ) : 3^(x-2) = 9^3 → x = 8 := 
by
  sorry

end find_x_eq_eight_l208_208331


namespace sqrt_expression_identity_l208_208236

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 17 - 4

theorem sqrt_expression_identity : Real.sqrt ((-a)^3 + (b + 4)^2) = 4 :=
by
  -- Prove the statement

  sorry

end sqrt_expression_identity_l208_208236


namespace cost_of_two_burritos_and_five_quesadillas_l208_208287

theorem cost_of_two_burritos_and_five_quesadillas
  (b q : ℝ)
  (h1 : b + 4 * q = 3.50)
  (h2 : 4 * b + q = 4.10) :
  2 * b + 5 * q = 5.02 := 
sorry

end cost_of_two_burritos_and_five_quesadillas_l208_208287


namespace intersection_of_A_and_B_l208_208513

def A : Set ℝ := {x | 1 < x ∧ x < 7}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 5} := by
  sorry

end intersection_of_A_and_B_l208_208513


namespace number_of_sports_books_l208_208569

def total_books : ℕ := 58
def school_books : ℕ := 19
def sports_books (total_books school_books : ℕ) : ℕ := total_books - school_books

theorem number_of_sports_books : sports_books total_books school_books = 39 := by
  -- proof goes here
  sorry

end number_of_sports_books_l208_208569


namespace trajectory_of_Q_l208_208984

variables {P Q M : ℝ × ℝ}

-- Define the conditions as Lean predicates
def is_midpoint (M P Q : ℝ × ℝ) : Prop :=
  M = (0, 4) ∧ M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def point_on_line (P : ℝ × ℝ) : Prop :=
  P.1 + P.2 - 2 = 0

-- Define the theorem that needs to be proven
theorem trajectory_of_Q :
  (∃ P Q M : ℝ × ℝ, is_midpoint M P Q ∧ point_on_line P) →
  ∃ Q : ℝ × ℝ, (∀ P : ℝ × ℝ, point_on_line P → is_midpoint (0,4) P Q → Q.1 + Q.2 - 6 = 0) :=
by sorry

end trajectory_of_Q_l208_208984


namespace count_three_digit_numbers_increased_by_99_when_reversed_l208_208256

def countValidNumbers : Nat := 80

theorem count_three_digit_numbers_increased_by_99_when_reversed :
  ∃ (a b c : Nat), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧
   (100 * a + 10 * b + c + 99 = 100 * c + 10 * b + a) ∧
  (countValidNumbers = 80) :=
sorry

end count_three_digit_numbers_increased_by_99_when_reversed_l208_208256


namespace koala_fiber_consumption_l208_208531

theorem koala_fiber_consumption (x : ℝ) (H : 12 = 0.30 * x) : x = 40 :=
by
  sorry

end koala_fiber_consumption_l208_208531


namespace jessie_weight_before_jogging_l208_208448

-- Definitions: conditions from the problem statement
variables (lost_weight current_weight : ℤ)
-- Conditions
def condition_lost_weight : Prop := lost_weight = 126
def condition_current_weight : Prop := current_weight = 66

-- Proposition to be proved
theorem jessie_weight_before_jogging (W_before_jogging : ℤ) :
  condition_lost_weight lost_weight → condition_current_weight current_weight →
  W_before_jogging = current_weight + lost_weight → W_before_jogging = 192 :=
by
  intros
  sorry

end jessie_weight_before_jogging_l208_208448


namespace incorrect_statement_C_l208_208678

theorem incorrect_statement_C :
  (∀ r : ℚ, ∃ p : ℝ, p = r) ∧  -- Condition A: All rational numbers can be represented by points on the number line.
  (∀ x : ℝ, x = 1 / x → x = 1 ∨ x = -1) ∧  -- Condition B: The reciprocal of a number equal to itself is ±1.
  (∀ f : ℚ, ∃ q : ℝ, q = f) →  -- Condition C (negation of C as presented): Fractions cannot be represented by points on the number line.
  (∀ x : ℝ, abs x ≥ 0) ∧ (∀ x : ℝ, abs x = 0 ↔ x = 0) →  -- Condition D: The number with the smallest absolute value is 0.
  false :=                      -- Prove that statement C is incorrect
by
  sorry

end incorrect_statement_C_l208_208678


namespace solve_quadratic_l208_208159

theorem solve_quadratic (y : ℝ) :
  3 * y * (y - 1) = 2 * (y - 1) → y = 2 / 3 ∨ y = 1 :=
by
  sorry

end solve_quadratic_l208_208159


namespace minimum_weighings_for_counterfeit_coin_l208_208542

/-- Given 9 coins, where 8 have equal weight and 1 is heavier (the counterfeit coin), prove that the 
minimum number of weighings required on a balance scale without weights to find the counterfeit coin is 2. -/
theorem minimum_weighings_for_counterfeit_coin (n : ℕ) (coins : Fin n → ℝ) 
  (h_n : n = 9) 
  (h_real : ∃ w : ℝ, ∀ i : Fin n, i.val < 8 → coins i = w) 
  (h_counterfeit : ∃ i : Fin n, ∀ j : Fin n, j ≠ i → coins i > coins j) : 
  ∃ k : ℕ, k = 2 :=
by
  sorry

end minimum_weighings_for_counterfeit_coin_l208_208542


namespace each_friend_pays_18_l208_208407

theorem each_friend_pays_18 (total_bill : ℝ) (silas_share : ℝ) (tip_fraction : ℝ) (num_friends : ℕ) (silas : ℕ) (remaining_friends : ℕ) :
  total_bill = 150 →
  silas_share = total_bill / 2 →
  tip_fraction = 0.1 →
  num_friends = 6 →
  remaining_friends = num_friends - 1 →
  silas = 1 →
  (total_bill - silas_share + tip_fraction * total_bill) / remaining_friends = 18 :=
by
  intros
  sorry

end each_friend_pays_18_l208_208407


namespace quadratic_inequality_solution_set_l208_208612

theorem quadratic_inequality_solution_set (a b c : ℝ) (Δ : ℝ) (hΔ : Δ = b^2 - 4*a*c) :
  (∀ x : ℝ, a*x^2 + b*x + c > 0) ↔ (a > 0 ∧ Δ < 0) := by
  sorry

end quadratic_inequality_solution_set_l208_208612


namespace reciprocal_of_one_fifth_l208_208976

theorem reciprocal_of_one_fifth : (∃ x : ℚ, (1/5) * x = 1 ∧ x = 5) :=
by
  -- The proof goes here, for now we assume it with sorry
  sorry

end reciprocal_of_one_fifth_l208_208976


namespace michael_matchstick_houses_l208_208089

theorem michael_matchstick_houses :
  ∃ n : ℕ, n = (600 / 2) / 10 ∧ n = 30 := 
sorry

end michael_matchstick_houses_l208_208089


namespace find_f_of_2_l208_208109

theorem find_f_of_2 
  (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x - 1/x) = x^2 + 1/x^2) : f 2 = 6 :=
sorry

end find_f_of_2_l208_208109


namespace find_other_man_age_l208_208444

variable (avg_age_men inc_age_man other_man_age avg_age_women total_age_increase : ℕ)

theorem find_other_man_age 
    (h1 : inc_age_man = 2) 
    (h2 : ∀ m, m = 8 * (avg_age_men + inc_age_man))
    (h3 : ∃ y, y = 22) 
    (h4 : ∀ w, w = 29) 
    (h5 : total_age_increase = 2 * avg_age_women - (22 + other_man_age)) :
  total_age_increase = 16 → other_man_age = 20 :=
by
  intros
  sorry

end find_other_man_age_l208_208444


namespace initial_number_of_persons_l208_208827

-- Define the given conditions
def initial_weights (N : ℕ) : ℝ := 65 * N
def new_person_weight : ℝ := 80
def increased_average_weight : ℝ := 2.5
def weight_increase (N : ℕ) : ℝ := increased_average_weight * N

-- Mathematically equivalent proof problem
theorem initial_number_of_persons 
    (N : ℕ)
    (h : weight_increase N = new_person_weight - 65) : N = 6 :=
by
  -- Place proof here when necessary
  sorry

end initial_number_of_persons_l208_208827


namespace final_bug_population_is_zero_l208_208442

def initial_population := 400
def spiders := 12
def spider_consumption := 7
def ladybugs := 5
def ladybug_consumption := 6
def mantises := 8
def mantis_consumption := 4

def day1_population := initial_population * 80 / 100

def predators_consumption_day := (spiders * spider_consumption) +
                                 (ladybugs * ladybug_consumption) +
                                 (mantises * mantis_consumption)

def day2_population := day1_population - predators_consumption_day
def day3_population := day2_population - predators_consumption_day
def day4_population := max 0 (day3_population - predators_consumption_day)
def day5_population := max 0 (day4_population - predators_consumption_day)
def day6_population := max 0 (day5_population - predators_consumption_day)

def day7_population := day6_population * 70 / 100

theorem final_bug_population_is_zero: 
  day7_population = 0 :=
  by
  sorry

end final_bug_population_is_zero_l208_208442


namespace value_of_2alpha_minus_beta_l208_208651

theorem value_of_2alpha_minus_beta (a β : ℝ) (h1 : 3 * Real.sin a - Real.cos a = 0) 
    (h2 : 7 * Real.sin β + Real.cos β = 0) (h3 : 0 < a ∧ a < Real.pi / 2) 
    (h4 : Real.pi / 2 < β ∧ β < Real.pi) : 
    2 * a - β = -3 * Real.pi / 4 := 
sorry

end value_of_2alpha_minus_beta_l208_208651


namespace LittleRedHeightCorrect_l208_208664

noncomputable def LittleRedHeight : ℝ :=
let LittleMingHeight := 1.3 
let HeightDifference := 0.2 
LittleMingHeight - HeightDifference

theorem LittleRedHeightCorrect : LittleRedHeight = 1.1 := by
  sorry

end LittleRedHeightCorrect_l208_208664


namespace largest_integer_value_n_l208_208594

theorem largest_integer_value_n (n : ℤ) : 
  (n^2 - 9 * n + 18 < 0) → n ≤ 5 := sorry

end largest_integer_value_n_l208_208594


namespace opposite_of_2023_l208_208033

-- Defining the opposite (additive inverse) of a number
def opposite (a : ℤ) : ℤ := -a

-- Stating that the opposite of 2023 is -2023
theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  -- Proof goes here
  sorry

end opposite_of_2023_l208_208033


namespace probability_top_card_is_star_l208_208344

theorem probability_top_card_is_star :
  let total_cards := 65
  let suits := 5
  let ranks_per_suit := 13
  let star_cards := 13
  (star_cards / total_cards) = 1 / 5 :=
by
  sorry

end probability_top_card_is_star_l208_208344


namespace quadrilateral_angles_arith_prog_l208_208430

theorem quadrilateral_angles_arith_prog {x a b c : ℕ} (d : ℝ):
  (x^2 = 8^2 + 7^2 + 2 * 8 * 7 * Real.sin (3 * d)) →
  x = a + Real.sqrt b + Real.sqrt c →
  x = Real.sqrt 113 →
  a + b + c = 113 :=
by
  sorry

end quadrilateral_angles_arith_prog_l208_208430


namespace hexagon_perimeter_l208_208868

theorem hexagon_perimeter
  (A B C D E F : Type)  -- vertices of the hexagon
  (angle_A : ℝ) (angle_C : ℝ) (angle_E : ℝ)  -- nonadjacent angles
  (angle_B : ℝ) (angle_D : ℝ) (angle_F : ℝ)  -- adjacent angles
  (area_hexagon : ℝ)
  (side_length : ℝ)
  (h1 : angle_A = 120) (h2 : angle_C = 120) (h3 : angle_E = 120)
  (h4 : angle_B = 60) (h5 : angle_D = 60) (h6 : angle_F = 60)
  (h7 : area_hexagon = 24)
  (h8 : ∃ s, ∀ (u v : Type), side_length = s) :
  6 * side_length = 24 / (Real.sqrt 3 ^ (1/4)) :=
by
  sorry

end hexagon_perimeter_l208_208868


namespace relationship_xyz_l208_208393

theorem relationship_xyz (a b : ℝ) (x y z : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a > b) (hab_sum : a + b = 1) 
  (hx : x = Real.log b / Real.log a)
  (hy : y = Real.log (1 / b) / Real.log a)
  (hz : z = Real.log 3 / Real.log ((1 / a) + (1 / b))) : 
  y < z ∧ z < x := 
sorry

end relationship_xyz_l208_208393


namespace handshakes_count_l208_208121

def women := 6
def teams := 3
def shakes_per_woman := 4
def total_handshakes := (6 * 4) / 2

theorem handshakes_count : total_handshakes = 12 := by
  -- We provide this theorem directly.
  rfl

end handshakes_count_l208_208121


namespace phyllis_marbles_l208_208391

theorem phyllis_marbles (num_groups : ℕ) (num_marbles_per_group : ℕ) (h1 : num_groups = 32) (h2 : num_marbles_per_group = 2) : 
  num_groups * num_marbles_per_group = 64 :=
by
  sorry

end phyllis_marbles_l208_208391


namespace factor_polynomial_l208_208117

theorem factor_polynomial (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) := by
  sorry

end factor_polynomial_l208_208117


namespace frog_reaches_vertical_side_l208_208853

def P (x y : ℕ) : ℝ := 
  if (x = 3 ∧ y = 3) then 0 -- blocked cell
  else if (x = 0 ∨ x = 5) then 1 -- vertical boundary
  else if (y = 0 ∨ y = 5) then 0 -- horizontal boundary
  else sorry -- inner probabilities to be calculated

theorem frog_reaches_vertical_side : P 2 2 = 5 / 8 :=
by sorry

end frog_reaches_vertical_side_l208_208853


namespace tail_to_body_ratio_l208_208457

variables (B : ℝ) (tail : ℝ := 9) (total_length : ℝ := 30)
variables (head_ratio : ℝ := 1/6)

-- Condition: The overall length is 30 inches
def overall_length_eq : Prop := B + B * head_ratio + tail = total_length

-- Theorem: Ratio of tail length to body length is 1:2
theorem tail_to_body_ratio (h : overall_length_eq B) : tail / B = 1 / 2 :=
sorry

end tail_to_body_ratio_l208_208457


namespace emily_small_gardens_count_l208_208576

-- Definitions based on conditions
def initial_seeds : ℕ := 41
def seeds_planted_in_big_garden : ℕ := 29
def seeds_per_small_garden : ℕ := 4

-- Theorem statement
theorem emily_small_gardens_count (initial_seeds seeds_planted_in_big_garden seeds_per_small_garden : ℕ) :
  initial_seeds = 41 →
  seeds_planted_in_big_garden = 29 →
  seeds_per_small_garden = 4 →
  (initial_seeds - seeds_planted_in_big_garden) / seeds_per_small_garden = 3 :=
by
  intros
  sorry

end emily_small_gardens_count_l208_208576


namespace find_x_of_product_eq_72_l208_208794

theorem find_x_of_product_eq_72 (x : ℝ) (h : 0 < x) (hx : x * ⌊x⌋₊ = 72) : x = 9 :=
sorry

end find_x_of_product_eq_72_l208_208794


namespace largest_divisor_of_n_l208_208708

-- Definitions and conditions from the problem
def is_positive_integer (n : ℕ) := n > 0
def is_divisible_by (a b : ℕ) := ∃ k : ℕ, a = k * b

-- Lean 4 statement encapsulating the problem
theorem largest_divisor_of_n (n : ℕ) (h1 : is_positive_integer n) (h2 : is_divisible_by (n * n) 72) : 
  ∃ v : ℕ, v = 12 ∧ is_divisible_by n v := 
sorry

end largest_divisor_of_n_l208_208708


namespace twice_x_plus_one_third_y_l208_208850

theorem twice_x_plus_one_third_y (x y : ℝ) : 2 * x + (1 / 3) * y = 2 * x + (1 / 3) * y := 
by 
  sorry

end twice_x_plus_one_third_y_l208_208850


namespace graph_of_equation_is_line_and_hyperbola_l208_208763

theorem graph_of_equation_is_line_and_hyperbola :
  ∀ (x y : ℝ), ((x^2 - 1) * (x + y) = y^2 * (x + y)) ↔ (y = -x) ∨ ((x + y) * (x - y) = 1) := by
  intro x y
  sorry

end graph_of_equation_is_line_and_hyperbola_l208_208763


namespace choir_members_max_l208_208752

theorem choir_members_max (s x : ℕ) (h1 : s * x < 147) (h2 : s * x + 3 = (s - 3) * (x + 2)) : s * x = 84 :=
sorry

end choir_members_max_l208_208752


namespace regular_polygon_interior_angle_l208_208036

theorem regular_polygon_interior_angle (S : ℝ) (n : ℕ) (h1 : S = 720) (h2 : (n - 2) * 180 = S) : 
  (S / n) = 120 := 
by
  sorry

end regular_polygon_interior_angle_l208_208036


namespace night_crew_worker_fraction_l208_208446

noncomputable def box_fraction_day : ℝ := 5/7

theorem night_crew_worker_fraction
  (D N : ℝ) -- Number of workers in day and night crew
  (B : ℝ)  -- Number of boxes each worker in the day crew loads
  (H1 : ∀ day_boxes_loaded : ℝ, day_boxes_loaded = D * B)
  (H2 : ∀ night_boxes_loaded : ℝ, night_boxes_loaded = N * (B / 2))
  (H3 : (D * B) / ((D * B) + (N * (B / 2))) = box_fraction_day) :
  N / D = 4/5 := 
sorry

end night_crew_worker_fraction_l208_208446


namespace complex_magnitude_l208_208721

open Complex

theorem complex_magnitude {x y : ℝ} (h : (1 + Complex.I) * x = 1 + y * Complex.I) : abs (x + y * Complex.I) = Real.sqrt 2 :=
sorry

end complex_magnitude_l208_208721


namespace clock_rings_in_a_day_l208_208719

-- Define the conditions
def rings_every_3_hours : ℕ := 3
def first_ring : ℕ := 1 -- This is 1 A.M. in our problem
def total_hours_in_day : ℕ := 24

-- Define the theorem
theorem clock_rings_in_a_day (n_rings : ℕ) : 
  (∀ n : ℕ, n_rings = total_hours_in_day / rings_every_3_hours + 1) :=
by
  -- use sorry to skip the proof
  sorry

end clock_rings_in_a_day_l208_208719


namespace pounds_added_l208_208180

-- Definitions based on conditions
def initial_weight : ℝ := 5
def weight_increase_percent : ℝ := 1.5  -- 150% increase
def final_weight : ℝ := 28

-- Statement to prove
theorem pounds_added (w_initial w_final w_percent_added : ℝ) (h_initial: w_initial = 5) (h_final: w_final = 28)
(h_percent: w_percent_added = 1.5) :
  w_final - w_initial = 23 := 
by
  sorry

end pounds_added_l208_208180


namespace taxi_ride_cost_l208_208423

namespace TaxiFare

def baseFare : ℝ := 2.00
def costPerMile : ℝ := 0.30
def taxRate : ℝ := 0.10
def distance : ℝ := 8.0

theorem taxi_ride_cost :
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  total_fare = 4.84 := by
  let fare_without_tax := baseFare + (costPerMile * distance)
  let tax := taxRate * fare_without_tax
  let total_fare := fare_without_tax + tax
  sorry

end TaxiFare

end taxi_ride_cost_l208_208423


namespace point_P_coordinates_l208_208011

noncomputable def P_coordinates (θ : ℝ) : ℝ × ℝ :=
(3 * Real.cos θ, 4 * Real.sin θ)

theorem point_P_coordinates : 
  ∀ θ, (0 ≤ θ ∧ θ ≤ Real.pi ∧ 1 = (4 / 3) * Real.tan θ) →
  P_coordinates θ = (12 / 5, 12 / 5) :=
by
  intro θ h
  sorry

end point_P_coordinates_l208_208011


namespace evaluate_infinite_series_l208_208111

noncomputable def infinite_series (n : ℕ) : ℝ := (n^2) / (3^n)

theorem evaluate_infinite_series :
  (∑' k : ℕ, infinite_series (k+1)) = 4.5 :=
by sorry

end evaluate_infinite_series_l208_208111


namespace max_abs_diff_f_l208_208589

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2 * Real.exp x

theorem max_abs_diff_f {k x1 x2 : ℝ} (hk : -3 ≤ k ∧ k ≤ -1) 
    (hx1 : k ≤ x1 ∧ x1 ≤ k + 2) (hx2 : k ≤ x2 ∧ x2 ≤ k + 2) : 
    |f x1 - f x2| ≤ 4 * Real.exp 1 := 
sorry

end max_abs_diff_f_l208_208589


namespace richmond_tickets_l208_208183

theorem richmond_tickets (total_tickets : ℕ) (second_half_tickets : ℕ) (first_half_tickets : ℕ) :
  total_tickets = 9570 →
  second_half_tickets = 5703 →
  first_half_tickets = total_tickets - second_half_tickets →
  first_half_tickets = 3867 := by
  sorry

end richmond_tickets_l208_208183


namespace length_of_first_train_l208_208704

theorem length_of_first_train
  (speed1_kmph : ℝ) (speed2_kmph : ℝ)
  (time_s : ℝ) (length2_m : ℝ)
  (relative_speed_mps : ℝ := (speed1_kmph + speed2_kmph) * 1000 / 3600)
  (total_distance_m : ℝ := relative_speed_mps * time_s)
  (length1_m : ℝ := total_distance_m - length2_m) :
  speed1_kmph = 80 →
  speed2_kmph = 65 →
  time_s = 7.199424046076314 →
  length2_m = 180 →
  length1_m = 110 :=
by
  sorry

end length_of_first_train_l208_208704


namespace total_marbles_l208_208071

variables (y : ℝ) 

def first_friend_marbles : ℝ := 2 * y + 2
def second_friend_marbles : ℝ := y
def third_friend_marbles : ℝ := 3 * y - 1

theorem total_marbles :
  (first_friend_marbles y) + (second_friend_marbles y) + (third_friend_marbles y) = 6 * y + 1 :=
by
  sorry

end total_marbles_l208_208071


namespace total_flowers_in_vase_l208_208883

-- Conditions as definitions
def num_roses : ℕ := 5
def num_lilies : ℕ := 2

-- Theorem statement
theorem total_flowers_in_vase : num_roses + num_lilies = 7 :=
by
  sorry

end total_flowers_in_vase_l208_208883


namespace time_with_cat_total_l208_208534

def time_spent_with_cat (petting combing brushing playing feeding cleaning : ℕ) : ℕ :=
  petting + combing + brushing + playing + feeding + cleaning

theorem time_with_cat_total :
  let petting := 12
  let combing := 1/3 * petting
  let brushing := 1/4 * combing
  let playing := 1/2 * petting
  let feeding := 5
  let cleaning := 2/5 * feeding
  time_spent_with_cat petting combing brushing playing feeding cleaning = 30 := by
  sorry

end time_with_cat_total_l208_208534


namespace correct_solution_l208_208470

variable (x y : ℤ) (a b : ℤ) (h1 : 2 * x + a * y = 6) (h2 : b * x - 7 * y = 16)

theorem correct_solution : 
  (∃ x y : ℤ, 2 * x - 3 * y = 6 ∧ 5 * x - 7 * y = 16 ∧ x = 6 ∧ y = 2) :=
by
  use 6, 2
  constructor
  · exact sorry -- 2 * 6 - 3 * 2 = 6
  constructor
  · exact sorry -- 5 * 6 - 7 * 2 = 16
  constructor
  · exact rfl
  · exact rfl

end correct_solution_l208_208470


namespace find_abc_sum_l208_208983

theorem find_abc_sum :
  ∃ (a b c : ℤ), 2 * a + 3 * b = 52 ∧ 3 * b + c = 41 ∧ b * c = 60 ∧ a + b + c = 25 :=
by
  use 8, 12, 5
  sorry

end find_abc_sum_l208_208983


namespace jogger_ahead_distance_l208_208276

-- Definitions of conditions
def jogger_speed : ℝ := 9  -- km/hr
def train_speed : ℝ := 45  -- km/hr
def train_length : ℝ := 150  -- meters
def passing_time : ℝ := 39  -- seconds

-- The main statement that we want to prove
theorem jogger_ahead_distance : 
  let relative_speed := (train_speed - jogger_speed) * (5 / 18)  -- conversion to m/s
  let distance_covered := relative_speed * passing_time
  let jogger_ahead := distance_covered - train_length
  jogger_ahead = 240 :=
by
  sorry

end jogger_ahead_distance_l208_208276


namespace JulieCompletesInOneHour_l208_208900

-- Define conditions
def JuliePeelsIn : ℕ := 10
def TedPeelsIn : ℕ := 8
def TimeTogether : ℕ := 4

-- Define their respective rates
def JulieRate : ℚ := 1 / JuliePeelsIn
def TedRate : ℚ := 1 / TedPeelsIn

-- Define the task completion in 4 hours together
def TaskCompletedTogether : ℚ := (JulieRate * TimeTogether) + (TedRate * TimeTogether)

-- Define remaining task after working together
def RemainingTask : ℚ := 1 - TaskCompletedTogether

-- Define time for Julie to complete the remaining task
def TimeForJulieToComplete : ℚ := RemainingTask / JulieRate

-- The theorem statement
theorem JulieCompletesInOneHour :
  TimeForJulieToComplete = 1 := by
  sorry

end JulieCompletesInOneHour_l208_208900


namespace sum_m_n_l208_208515

-- We define the conditions and problem
variables (m n : ℕ)

-- Conditions
def conditions := m > 50 ∧ n > 50 ∧ Nat.lcm m n = 480 ∧ Nat.gcd m n = 12

-- Statement to prove
theorem sum_m_n : conditions m n → m + n = 156 := by sorry

end sum_m_n_l208_208515


namespace D_score_l208_208761

noncomputable def score_A : ℕ := 94

variables (A B C D E : ℕ)

-- Conditions
def A_scored : A = score_A := sorry
def B_highest : B > A := sorry
def C_average_AD : (C * 2) = A + D := sorry
def D_average_five : (D * 5) = A + B + C + D + E := sorry
def E_score_C2 : E = C + 2 := sorry

-- Question
theorem D_score : D = 96 :=
by {
  sorry
}

end D_score_l208_208761


namespace triangle_shortest_side_l208_208112

theorem triangle_shortest_side (x y z : ℝ) (h : x / y = 1 / 2) (h1 : x / z = 1 / 3) (hyp : x = 6) : z = 3 :=
sorry

end triangle_shortest_side_l208_208112


namespace latus_rectum_of_parabola_l208_208243

theorem latus_rectum_of_parabola : 
  ∀ x y : ℝ, x^2 = -y → y = 1/4 :=
by
  -- Proof omitted
  sorry

end latus_rectum_of_parabola_l208_208243


namespace num_men_in_second_group_l208_208539

-- Define the conditions
def numMen1 := 4
def hoursPerDay1 := 10
def daysPerWeek := 7
def earningsPerWeek1 := 1200

def hoursPerDay2 := 6
def earningsPerWeek2 := 1620

-- Define the earning per man-hour
def earningPerManHour := earningsPerWeek1 / (numMen1 * hoursPerDay1 * daysPerWeek)

-- Define the total man-hours required for the second amount of earnings
def totalManHours2 := earningsPerWeek2 / earningPerManHour

-- Define the number of men in the second group
def numMen2 := totalManHours2 / (hoursPerDay2 * daysPerWeek)

-- Theorem stating the number of men in the second group 
theorem num_men_in_second_group : numMen2 = 9 := by
  sorry

end num_men_in_second_group_l208_208539


namespace smallest_n_mod_equality_l208_208239

theorem smallest_n_mod_equality :
  ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 5]) ∧ (∀ m : ℕ, 0 < m ∧ m < n → (7^m ≡ m^7 [MOD 5]) = false) :=
  sorry

end smallest_n_mod_equality_l208_208239


namespace expression_evaluation_l208_208212

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end expression_evaluation_l208_208212


namespace tangent_line_eq_monotonic_intervals_l208_208488

noncomputable def f (x : ℝ) (a : ℝ) := x - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := 1 - (a / x)

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ a = 2) :
  y = f 1 2 → (x - 1) + (y - 1) - 2 * ((x - 1) + (y - 1)) = 0 := by sorry

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, f' x a > 0) ∧
  (a > 0 → ∀ x > 0, (x < a → f' x a < 0) ∧ (x > a → f' x a > 0)) := by sorry

end tangent_line_eq_monotonic_intervals_l208_208488


namespace min_balls_to_draw_l208_208078

theorem min_balls_to_draw (black white red : ℕ) (h_black : black = 10) (h_white : white = 9) (h_red : red = 8) :
  ∃ n, n = 20 ∧
  ∀ k, (k < 20) → ¬ (∃ b w r, b + w + r = k ∧ b ≤ black ∧ w ≤ white ∧ r ≤ red ∧ r > 0 ∧ w > 0) :=
by {
  sorry
}

end min_balls_to_draw_l208_208078


namespace second_largest_is_D_l208_208918

noncomputable def A := 3 * 3
noncomputable def C := 4 * A
noncomputable def B := C - 15
noncomputable def D := A + 19

theorem second_largest_is_D : 
    ∀ (A B C D : ℕ), 
      A = 9 → 
      B = 21 →
      C = 36 →
      D = 28 →
      D = 28 :=
by
  intros A B C D hA hB hC hD
  have h1 : A = 9 := by assumption
  have h2 : B = 21 := by assumption
  have h3 : C = 36 := by assumption
  have h4 : D = 28 := by assumption
  exact h4

end second_largest_is_D_l208_208918


namespace median_eq_altitude_eq_perp_bisector_eq_l208_208484

open Real

def point := ℝ × ℝ

def A : point := (1, 3)
def B : point := (3, 1)
def C : point := (-1, 0)

-- Median on BC
theorem median_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x, y) = ((1 + (-1))/2, (1 + 0)/2) → x = 1 :=
by
  intros x y h
  sorry

-- Altitude on BC
theorem altitude_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x - 1) / (y - 3) = -4 → 4*x + y - 7 = 0 :=
by
  intros x y h
  sorry

-- Perpendicular bisector of BC
theorem perp_bisector_eq : ∀ (x y : ℝ), (x = 1 ∧ y = 1/2) ∨ (x - 1) / (y - 1/2) = -4 
                          → 8*x + 2*y - 9 = 0 :=
by
  intros x y h
  sorry

end median_eq_altitude_eq_perp_bisector_eq_l208_208484


namespace find_x_collinear_l208_208426

-- Given vectors
def vec_a : ℝ × ℝ := (1, 2)
def vec_b : ℝ × ℝ := (1, -3)
def vec_c (x : ℝ) : ℝ × ℝ := (-2, x)

-- Definition of vectors being collinear
def collinear (v₁ v₂ : ℝ × ℝ) : Prop :=
∃ k : ℝ, v₁ = (k * v₂.1, k * v₂.2)

-- Question: What is the value of x such that vec_a + vec_b is collinear with vec_c(x)?
theorem find_x_collinear : ∃ x : ℝ, collinear (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_c x) ∧ x = 1 :=
by
  sorry

end find_x_collinear_l208_208426


namespace power_of_m_divisible_by_33_l208_208654

theorem power_of_m_divisible_by_33 (m : ℕ) (h : m > 0) (k : ℕ) (h_pow : (m ^ k) % 33 = 0) :
  ∃ n, n > 0 ∧ 11 ∣ m ^ n :=
by
  sorry

end power_of_m_divisible_by_33_l208_208654


namespace gina_total_cost_l208_208506

-- Define the constants based on the conditions
def total_credits : ℕ := 18
def reg_credits : ℕ := 12
def reg_cost_per_credit : ℕ := 450
def lab_credits : ℕ := 6
def lab_cost_per_credit : ℕ := 550
def num_textbooks : ℕ := 3
def textbook_cost : ℕ := 150
def num_online_resources : ℕ := 4
def online_resource_cost : ℕ := 95
def facilities_fee : ℕ := 200
def lab_fee_per_credit : ℕ := 75

-- Calculating the total cost
noncomputable def total_cost : ℕ :=
  (reg_credits * reg_cost_per_credit) +
  (lab_credits * lab_cost_per_credit) +
  (num_textbooks * textbook_cost) +
  (num_online_resources * online_resource_cost) +
  facilities_fee +
  (lab_credits * lab_fee_per_credit)

-- The proof problem to show that the total cost is 10180
theorem gina_total_cost : total_cost = 10180 := by
  sorry

end gina_total_cost_l208_208506


namespace number_of_valid_4_digit_integers_l208_208012

/-- 
Prove that the number of 4-digit positive integers that satisfy the following conditions:
1. Each of the first two digits must be 2, 3, or 5.
2. The last two digits cannot be the same.
3. Each of the last two digits must be 4, 6, or 9.
is equal to 54.
-/
theorem number_of_valid_4_digit_integers : 
  ∃ n : ℕ, n = 54 ∧ 
  ∀ d1 d2 d3 d4 : ℕ, 
    (d1 = 2 ∨ d1 = 3 ∨ d1 = 5) ∧ 
    (d2 = 2 ∨ d2 = 3 ∨ d2 = 5) ∧ 
    (d3 = 4 ∨ d3 = 6 ∨ d3 = 9) ∧ 
    (d4 = 4 ∨ d4 = 6 ∨ d4 = 9) ∧ 
    (d3 ≠ d4) → 
    n = 54 := 
sorry

end number_of_valid_4_digit_integers_l208_208012


namespace bianca_made_after_selling_l208_208264

def bianca_initial_cupcakes : ℕ := 14
def bianca_sold_cupcakes : ℕ := 6
def bianca_final_cupcakes : ℕ := 25

theorem bianca_made_after_selling :
  (bianca_initial_cupcakes - bianca_sold_cupcakes) + (bianca_final_cupcakes - (bianca_initial_cupcakes - bianca_sold_cupcakes)) = bianca_final_cupcakes :=
by
  sorry

end bianca_made_after_selling_l208_208264


namespace HorseKeepsPower_l208_208048

/-- If the Little Humpbacked Horse does not eat for seven days or does not sleep for seven days,
    he will lose his magic power. Suppose he did not eat or sleep for a whole week. 
    Prove that by the end of the seventh day, he must do the activity he did not do right before 
    the start of the first period of seven days in order to keep his power. -/
theorem HorseKeepsPower (eat sleep : ℕ → Prop) :
  (∀ (n : ℕ), (n ≥ 7 → ¬eat n) ∨ (n ≥ 7 → ¬sleep n)) →
  (∀ (n : ℕ), n < 7 → (¬eat n ∧ ¬sleep n)) →
  ∃ (t : ℕ), t > 7 → (eat t ∨ sleep t) :=
sorry

end HorseKeepsPower_l208_208048


namespace Question_D_condition_l208_208069

theorem Question_D_condition (P Q : Prop) (h : P → Q) : ¬ Q → ¬ P :=
by sorry

end Question_D_condition_l208_208069


namespace sequence_formula_l208_208725

theorem sequence_formula (a : ℕ → ℚ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_formula_l208_208725


namespace men_in_business_class_l208_208855

theorem men_in_business_class (total_passengers : ℕ) (percentage_men : ℝ)
  (fraction_business_class : ℝ) (num_men_in_business_class : ℕ) 
  (h1 : total_passengers = 160) 
  (h2 : percentage_men = 0.75) 
  (h3 : fraction_business_class = 1 / 4) 
  (h4 : num_men_in_business_class = total_passengers * percentage_men * fraction_business_class) : 
  num_men_in_business_class = 30 := 
  sorry

end men_in_business_class_l208_208855


namespace tony_gas_expense_in_4_weeks_l208_208933

theorem tony_gas_expense_in_4_weeks :
  let miles_per_gallon := 25
  let miles_per_round_trip_per_day := 50
  let travel_days_per_week := 5
  let tank_capacity_in_gallons := 10
  let cost_per_gallon := 2
  let weeks := 4
  let total_miles_per_week := miles_per_round_trip_per_day * travel_days_per_week
  let total_miles := total_miles_per_week * weeks
  let miles_per_tank := miles_per_gallon * tank_capacity_in_gallons
  let fill_ups_needed := total_miles / miles_per_tank
  let total_gallons_needed := fill_ups_needed * tank_capacity_in_gallons
  let total_cost := total_gallons_needed * cost_per_gallon
  total_cost = 80 :=
by
  sorry

end tony_gas_expense_in_4_weeks_l208_208933


namespace michael_lost_at_least_800_l208_208299

theorem michael_lost_at_least_800 
  (T F : ℕ) 
  (h1 : T + F = 15) 
  (h2 : T = F + 1 ∨ T = F - 1) 
  (h3 : 10 * T + 50 * F = 1270) : 
  1270 - (10 * T + 50 * F) = 800 :=
by
  sorry

end michael_lost_at_least_800_l208_208299


namespace stamens_in_bouquet_l208_208753

-- Define the number of pistils, leaves, stamens for black roses and crimson flowers
def pistils_black_rose : ℕ := 4
def stamens_black_rose : ℕ := 4
def leaves_black_rose : ℕ := 2

def pistils_crimson_flower : ℕ := 8
def stamens_crimson_flower : ℕ := 10
def leaves_crimson_flower : ℕ := 3

-- Define the number of black roses and crimson flowers (as variables x and y)
variables (x y : ℕ)

-- Define the total number of pistils and leaves in the bouquet
def total_pistils : ℕ := pistils_black_rose * x + pistils_crimson_flower * y
def total_leaves : ℕ := leaves_black_rose * x + leaves_crimson_flower * y

-- Condition: There are 108 fewer leaves than pistils
axiom leaves_pistils_relation : total_leaves = total_pistils - 108

-- Calculate the total number of stamens in the bouquet
def total_stamens : ℕ := stamens_black_rose * x + stamens_crimson_flower * y

-- The theorem to be proved
theorem stamens_in_bouquet : total_stamens = 216 :=
by
  sorry

end stamens_in_bouquet_l208_208753


namespace solve_math_problem_l208_208043

-- Math problem definition
def math_problem (A : ℝ) : Prop :=
  (0 < A ∧ A < (Real.pi / 2)) ∧ (Real.cos A = 3 / 5) →
  Real.sin (2 * A) = 24 / 25

-- Example theorem statement in Lean
theorem solve_math_problem (A : ℝ) : math_problem A :=
sorry

end solve_math_problem_l208_208043


namespace question_I_question_II_l208_208002

def f (x a : ℝ) : ℝ := |x - a| + 3 * x

theorem question_I (a : ℝ) (h_pos : a > 0) : 
  (f 1 x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) := by sorry

theorem question_II (a : ℝ) (h_pos : a > 0) : 
  (- (a / 2) = -1) ↔ (a = 2) := by sorry

end question_I_question_II_l208_208002


namespace determine_range_of_a_l208_208975

theorem determine_range_of_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ a * x^2 - x + 2 = 0 ∧ a * y^2 - y + 2 = 0) : 
  a < 1 / 8 ∧ a ≠ 0 :=
sorry

end determine_range_of_a_l208_208975


namespace ratio_of_side_lengths_l208_208734

theorem ratio_of_side_lengths (a b c : ℕ) (h : a * a * b * b = 18 * c * c * 50 * c * c) :
  (12 = 1800000) ->  (15 = 1500) -> (10 > 0):=
by
  sorry

end ratio_of_side_lengths_l208_208734


namespace probability_plane_contains_points_inside_octahedron_l208_208549

noncomputable def enhanced_octahedron_probability : ℚ :=
  let total_vertices := 18
  let total_ways := Nat.choose total_vertices 3
  let faces := 8
  let triangles_per_face := 4
  let unfavorable_ways := faces * triangles_per_face
  total_ways - unfavorable_ways

theorem probability_plane_contains_points_inside_octahedron :
  enhanced_octahedron_probability / (816 : ℚ) = 49 / 51 :=
sorry

end probability_plane_contains_points_inside_octahedron_l208_208549


namespace flag_pole_height_eq_150_l208_208837

-- Define the conditions
def tree_height : ℝ := 12
def tree_shadow_length : ℝ := 8
def flag_pole_shadow_length : ℝ := 100

-- Problem statement: prove the height of the flag pole equals 150 meters
theorem flag_pole_height_eq_150 :
  ∃ (F : ℝ), (tree_height / tree_shadow_length) = (F / flag_pole_shadow_length) ∧ F = 150 :=
by
  -- Setup the proof scaffold
  have h : (tree_height / tree_shadow_length) = (150 / flag_pole_shadow_length) := by sorry
  exact ⟨150, h, rfl⟩

end flag_pole_height_eq_150_l208_208837


namespace problem_solution_l208_208935

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l208_208935


namespace common_property_rhombus_rectangle_diagonals_l208_208232

-- Define a structure for Rhombus and its property
structure Rhombus (R : Type) :=
  (diagonals_perpendicular : Prop)
  (diagonals_bisect : Prop)

-- Define a structure for Rectangle and its property
structure Rectangle (R : Type) :=
  (diagonals_equal_length : Prop)
  (diagonals_bisect : Prop)

-- Define the theorem that states the common property between diagonals of both shapes
theorem common_property_rhombus_rectangle_diagonals (R : Type) 
  (rhombus_properties : Rhombus R) 
  (rectangle_properties : Rectangle R) :
  rhombus_properties.diagonals_bisect ∧ rectangle_properties.diagonals_bisect :=
by {
  -- Since the solution steps are not to be included, we conclude the proof with 'sorry'
  sorry
}

end common_property_rhombus_rectangle_diagonals_l208_208232


namespace find_vector_l208_208053

noncomputable def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + 3 * t, 3 + 2 * t)

noncomputable def line_m (s : ℝ) : ℝ × ℝ :=
  (-4 + 3 * s, 5 + 2 * s)

def vector_condition (v1 v2 : ℝ) : Prop :=
  v1 - v2 = 1

theorem find_vector :
  ∃ (v1 v2 : ℝ), vector_condition v1 v2 ∧ (v1, v2) = (3, 2) :=
sorry

end find_vector_l208_208053


namespace expected_value_boy_girl_adjacent_pairs_l208_208081

/-- Considering 10 boys and 15 girls lined up in a row, we need to show that
    the expected number of adjacent positions where a boy and a girl stand next to each other is 12. -/
theorem expected_value_boy_girl_adjacent_pairs :
  let boys := 10
  let girls := 15
  let total_people := boys + girls
  let total_adjacent_pairs := total_people - 1
  let p_boy_then_girl := (boys / total_people) * (girls / (total_people - 1))
  let p_girl_then_boy := (girls / total_people) * (boys / (total_people - 1))
  let expected_T := total_adjacent_pairs * (p_boy_then_girl + p_girl_then_boy)
  expected_T = 12 :=
by
  sorry

end expected_value_boy_girl_adjacent_pairs_l208_208081


namespace division_of_negatives_example_div_l208_208537

theorem division_of_negatives (a b : Int) (ha : a < 0) (hb : b < 0) (hb_neq : b ≠ 0) : 
  (-a) / (-b) = a / b :=
by sorry

theorem example_div : (-300) / (-50) = 6 :=
by
  apply division_of_negatives
  repeat { sorry }

end division_of_negatives_example_div_l208_208537


namespace greatest_prime_factor_341_l208_208131

theorem greatest_prime_factor_341 : ∃ p : ℕ, Nat.Prime p ∧ p = 17 ∧ p = Nat.gcd 341 (Nat.gcd 341 (Nat.gcd 341 341)) :=
by
  sorry

end greatest_prime_factor_341_l208_208131


namespace car_travel_distance_l208_208646

variable (b t : Real)
variable (h1 : b > 0)
variable (h2 : t > 0)

theorem car_travel_distance (b t : Real) (h1 : b > 0) (h2 : t > 0) :
  let rate := b / 4
  let inches_in_yard := 36
  let time_in_seconds := 5 * 60
  let distance_in_inches := (rate / t) * time_in_seconds
  let distance_in_yards := distance_in_inches / inches_in_yard
  distance_in_yards = (25 * b) / (12 * t) := by
  sorry

end car_travel_distance_l208_208646


namespace square_area_is_256_l208_208510

-- Definitions of the conditions
def rect_width : ℝ := 4
def rect_length : ℝ := 3 * rect_width
def side_of_square : ℝ := rect_length + rect_width

-- Proposition
theorem square_area_is_256 (rect_width : ℝ) (h1 : rect_width = 4) 
                           (rect_length : ℝ) (h2 : rect_length = 3 * rect_width) :
  side_of_square ^ 2 = 256 :=
by 
  sorry

end square_area_is_256_l208_208510


namespace cube_volume_of_surface_area_l208_208876

-- Define the condition: the surface area S is 864 square units
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- The proof problem: Given that the surface area of a cube is 864 square units,
-- prove that the volume of the cube is 1728 cubic units
theorem cube_volume_of_surface_area (S : ℝ) (hS : S = 864) : 
  ∃ V : ℝ, V = 1728 ∧ ∃ s : ℝ, surface_area s = S ∧ V = s^3 :=
by 
  sorry

end cube_volume_of_surface_area_l208_208876


namespace more_geese_than_ducks_l208_208383

def mallard_start := 25
def wood_start := 15
def geese_start := 2 * mallard_start - 10
def swan_start := 3 * wood_start + 8

def mallard_after_morning := mallard_start + 4
def wood_after_morning := wood_start + 8
def geese_after_morning := geese_start + 7
def swan_after_morning := swan_start

def mallard_after_noon := mallard_after_morning
def wood_after_noon := wood_after_morning - 6
def geese_after_noon := geese_after_morning - 5
def swan_after_noon := swan_after_morning - 9

def mallard_after_later := mallard_after_noon + 8
def wood_after_later := wood_after_noon + 10
def geese_after_later := geese_after_noon
def swan_after_later := swan_after_noon + 4

def mallard_after_evening := mallard_after_later + 5
def wood_after_evening := wood_after_later + 3
def geese_after_evening := geese_after_later + 15
def swan_after_evening := swan_after_later + 11

def mallard_final := 0
def wood_final := wood_after_evening - (3 / 4 : ℚ) * wood_after_evening
def geese_final := geese_after_evening - (1 / 5 : ℚ) * geese_after_evening
def swan_final := swan_after_evening - (1 / 2 : ℚ) * swan_after_evening

theorem more_geese_than_ducks :
  (geese_final - (mallard_final + wood_final)) = 38 :=
by sorry

end more_geese_than_ducks_l208_208383


namespace trouser_sale_price_l208_208553

theorem trouser_sale_price 
  (original_price : ℝ) 
  (percent_decrease : ℝ) 
  (sale_price : ℝ) 
  (h : original_price = 100) 
  (p : percent_decrease = 0.25) 
  (s : sale_price = original_price * (1 - percent_decrease)) : 
  sale_price = 75 :=
by 
  sorry

end trouser_sale_price_l208_208553


namespace solution_set_f_lt_g_l208_208275

noncomputable def f : ℝ → ℝ := sorry -- Assume f exists according to the given conditions

lemma f_at_one : f 1 = -2 := sorry

lemma f_derivative_neg (x : ℝ) : (deriv f x) < 0 := sorry

def g (x : ℝ) : ℝ := x - 3

lemma g_at_one : g 1 = -2 := sorry

theorem solution_set_f_lt_g :
  {x : ℝ | f x < g x} = {x : ℝ | 1 < x} :=
sorry

end solution_set_f_lt_g_l208_208275


namespace binom_1300_2_eq_844350_l208_208663

theorem binom_1300_2_eq_844350 : (Nat.choose 1300 2) = 844350 := by
  sorry

end binom_1300_2_eq_844350_l208_208663


namespace quadratic_sum_of_coefficients_l208_208825

theorem quadratic_sum_of_coefficients (x : ℝ) : 
  let a := 1
  let b := 1
  let c := -4
  a + b + c = -2 :=
by
  sorry

end quadratic_sum_of_coefficients_l208_208825


namespace find_value_of_f_at_1_l208_208613

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_value_of_f_at_1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 2 * f x - f (- x) = 3 * x + 1) : f 1 = 2 :=
by
  sorry

end find_value_of_f_at_1_l208_208613


namespace reservoir_water_l208_208821

-- Conditions definitions
def total_capacity (C : ℝ) : Prop :=
  ∃ (x : ℝ), x = C

def normal_level (C : ℝ) : ℝ :=
  C - 20

def water_end_of_month (C : ℝ) : ℝ :=
  0.75 * C

def condition_equation (C : ℝ) : Prop :=
  water_end_of_month C = 2 * normal_level C

-- The theorem proving the amount of water at the end of the month is 24 million gallons given the conditions
theorem reservoir_water (C : ℝ) (hC : total_capacity C) (h_condition : condition_equation C) : water_end_of_month C = 24 :=
by
  sorry

end reservoir_water_l208_208821


namespace sum_of_first_8_terms_l208_208059

noncomputable def sum_of_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
a * (1 - r^n) / (1 - r)

theorem sum_of_first_8_terms 
  (a r : ℝ)
  (h₁ : sum_of_geometric_sequence a r 4 = 5)
  (h₂ : sum_of_geometric_sequence a r 12 = 35) :
  sum_of_geometric_sequence a r 8 = 15 := 
sorry

end sum_of_first_8_terms_l208_208059


namespace problem_equivalent_statement_l208_208603

-- Define the operations provided in the problem
inductive Operation
| add
| sub
| mul
| div

open Operation

-- Represents the given equation with the specified operation
def applyOperation (op : Operation) (a b : ℕ) : ℕ :=
  match op with
  | add => a + b
  | sub => a - b
  | mul => a * b
  | div => a / b

theorem problem_equivalent_statement : 
  (∀ (op : Operation), applyOperation op 8 2 - 5 + 7 - (3^2 - 4) ≠ 6) → (¬ ∃ op : Operation, applyOperation op 8 2 = 9) := 
by
  sorry

end problem_equivalent_statement_l208_208603


namespace max_value_xy_xz_yz_l208_208098

theorem max_value_xy_xz_yz (x y z : ℝ) (h : x + 2 * y + z = 6) :
  xy + xz + yz ≤ 6 :=
sorry

end max_value_xy_xz_yz_l208_208098


namespace int_modulo_l208_208632

theorem int_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n < 17) (h3 : 38574 ≡ n [ZMOD 17]) : n = 1 :=
by
  sorry

end int_modulo_l208_208632


namespace rahul_work_days_l208_208520

theorem rahul_work_days
  (R : ℕ)
  (Rajesh_days : ℕ := 2)
  (total_payment : ℕ := 170)
  (rahul_share : ℕ := 68)
  (combined_work_rate : ℚ := 1) :
  (∃ R : ℕ, (1 / (R : ℚ) + 1 / (Rajesh_days : ℚ) = combined_work_rate) ∧ (68 / (total_payment - rahul_share) = 2 / R) ∧ R = 3) :=
sorry

end rahul_work_days_l208_208520


namespace length_of_second_platform_l208_208478

theorem length_of_second_platform (train_length first_platform_length : ℕ) (time_to_cross_first_platform time_to_cross_second_platform : ℕ) 
  (H1 : train_length = 110) (H2 : first_platform_length = 160) (H3 : time_to_cross_first_platform = 15) 
  (H4 : time_to_cross_second_platform = 20) : ∃ second_platform_length, second_platform_length = 250 := 
by
  sorry

end length_of_second_platform_l208_208478


namespace find_x_l208_208213

theorem find_x (number x : ℝ) (h1 : 24 * number = 173 * x) (h2 : 24 * number = 1730) : x = 10 :=
by
  sorry

end find_x_l208_208213


namespace cumulative_revenue_eq_l208_208493

-- Define the initial box office revenue and growth rate
def initial_revenue : ℝ := 3
def growth_rate (x : ℝ) : ℝ := x

-- Define the cumulative revenue equation after 3 days
def cumulative_revenue (x : ℝ) : ℝ :=
  initial_revenue + initial_revenue * (1 + growth_rate x) + initial_revenue * (1 + growth_rate x) ^ 2

-- State the theorem that proves the equation
theorem cumulative_revenue_eq (x : ℝ) :
  cumulative_revenue x = 10 :=
sorry

end cumulative_revenue_eq_l208_208493


namespace max_square_plots_l208_208944
-- Lean 4 statement for the equivalent math problem

theorem max_square_plots (w l f s : ℕ) (h₁ : w = 40) (h₂ : l = 60) 
                         (h₃ : f = 2400) (h₄ : s ≠ 0) (h₅ : 2400 - 100 * s ≤ 2400)
                         (h₆ : w % s = 0) (h₇ : l % s = 0) :
  (w * l) / (s * s) = 6 :=
by {
  sorry
}

end max_square_plots_l208_208944


namespace plus_signs_count_l208_208234

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end plus_signs_count_l208_208234


namespace find_a_l208_208014

theorem find_a (a : ℝ) (h : 2 * a + 2 * a / 4 = 4) : a = 8 / 5 := sorry

end find_a_l208_208014


namespace min_value_xy_l208_208378

-- Defining the operation ⊗
def otimes (a b : ℝ) : ℝ := a * b - a - b

theorem min_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : otimes x y = 3) : 9 ≤ x * y := by
  sorry

end min_value_xy_l208_208378


namespace new_average_height_is_184_l208_208715

-- Define the initial conditions
def original_num_students : ℕ := 35
def original_avg_height : ℕ := 180
def left_num_students : ℕ := 7
def left_avg_height : ℕ := 120
def joined_num_students : ℕ := 7
def joined_avg_height : ℕ := 140

-- Calculate the initial total height
def original_total_height := original_avg_height * original_num_students

-- Calculate the total height of the students who left
def left_total_height := left_avg_height * left_num_students

-- Calculate the new total height after the students left
def new_total_height1 := original_total_height - left_total_height

-- Calculate the total height of the new students who joined
def joined_total_height := joined_avg_height * joined_num_students

-- Calculate the new total height after the new students joined
def new_total_height2 := new_total_height1 + joined_total_height

-- Calculate the new average height
def new_avg_height := new_total_height2 / original_num_students

-- The theorem stating the result
theorem new_average_height_is_184 : new_avg_height = 184 := by
  sorry

end new_average_height_is_184_l208_208715


namespace solve_for_x_l208_208867

-- Definitions for the problem conditions
def perimeter_triangle := 14 + 12 + 12
def perimeter_rectangle (x : ℝ) := 2 * x + 16

-- Lean 4 statement for the proof problem 
theorem solve_for_x (x : ℝ) : 
  perimeter_triangle = perimeter_rectangle x → 
  x = 11 := 
by 
  -- standard placeholders
  sorry

end solve_for_x_l208_208867


namespace range_of_a_l208_208856

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) : a ≥ 1 :=
sorry

end range_of_a_l208_208856


namespace proof_problem_l208_208105

def p := 8 + 7 = 16
def q := Real.pi > 3

theorem proof_problem :
  (¬p ∧ q) ∧ ((p ∨ q) = true) ∧ ((p ∧ q) = false) ∧ ((¬p) = true) := sorry

end proof_problem_l208_208105


namespace proposition_a_proposition_b_proposition_c_proposition_d_l208_208988

variable (a b c : ℝ)

-- Proposition A: If ac^2 > bc^2, then a > b
theorem proposition_a (h : a * c^2 > b * c^2) : a > b := sorry

-- Proposition B: If a > b, then ac^2 > bc^2
theorem proposition_b (h : a > b) : ¬ (a * c^2 > b * c^2) := sorry

-- Proposition C: If a > b, then 1/a < 1/b
theorem proposition_c (h : a > b) : ¬ (1/a < 1/b) := sorry

-- Proposition D: If a > b > 0, then a^2 > ab > b^2
theorem proposition_d (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 := sorry

end proposition_a_proposition_b_proposition_c_proposition_d_l208_208988


namespace triangle_sides_inequality_l208_208127

-- Define the sides of a triangle and their sum
variables {a b c : ℝ}

-- Define the condition that they are sides of a triangle.
def triangle_sides (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the condition that their sum is 1
axiom sum_of_sides (a b c : ℝ) (h : triangle_sides a b c) : a + b + c = 1

-- Define the proof theorem for the inequality
theorem triangle_sides_inequality (h : triangle_sides a b c) (h_sum : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end triangle_sides_inequality_l208_208127


namespace no_natural_numbers_satisfying_conditions_l208_208526

theorem no_natural_numbers_satisfying_conditions :
  ¬ ∃ (a b : ℕ), a < b ∧ ∃ k : ℕ, b^2 + 4*a = k^2 := by
  sorry

end no_natural_numbers_satisfying_conditions_l208_208526


namespace total_alphabets_written_l208_208905

-- Define the number of vowels and the number of times each is written
def num_vowels : ℕ := 5
def repetitions : ℕ := 4

-- The theorem stating the total number of alphabets written on the board
theorem total_alphabets_written : num_vowels * repetitions = 20 := by
  sorry

end total_alphabets_written_l208_208905


namespace logical_equivalence_l208_208313

variables {α : Type} (A B : α → Prop)

theorem logical_equivalence :
  (∀ x, A x → B x) ↔
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, A x → B x) ∧
  (∀ x, ¬ B x → ¬ A x) :=
by sorry

end logical_equivalence_l208_208313


namespace factorize_polynomial_l208_208130

theorem factorize_polynomial (x y : ℝ) : x * y^2 - 16 * x = x * (y + 4) * (y - 4) := 
by
  sorry

end factorize_polynomial_l208_208130


namespace not_rain_probability_l208_208347

-- Define the probability of rain tomorrow
def prob_rain : ℚ := 3 / 10

-- Define the complementary probability (probability that it will not rain tomorrow)
def prob_no_rain : ℚ := 1 - prob_rain

-- Statement to prove: probability that it will not rain tomorrow equals 7/10 
theorem not_rain_probability : prob_no_rain = 7 / 10 := 
by sorry

end not_rain_probability_l208_208347


namespace emily_small_gardens_l208_208338

theorem emily_small_gardens 
  (total_seeds : Nat)
  (big_garden_seeds : Nat)
  (small_garden_seeds : Nat)
  (remaining_seeds : total_seeds = big_garden_seeds + (small_garden_seeds * 3)) :
  3 = (total_seeds - big_garden_seeds) / small_garden_seeds :=
by
  have h1 : total_seeds = 42 := by sorry
  have h2 : big_garden_seeds = 36 := by sorry
  have h3 : small_garden_seeds = 2 := by sorry
  have h4 : 6 = total_seeds - big_garden_seeds := by sorry
  have h5 : 3 = 6 / small_garden_seeds := by sorry
  sorry

end emily_small_gardens_l208_208338


namespace exists_q_r_polynomials_l208_208774

theorem exists_q_r_polynomials (n : ℕ) (p : Polynomial ℝ) 
  (h_deg : p.degree = n) 
  (h_monic : p.leadingCoeff = 1) :
  ∃ q r : Polynomial ℝ, 
    q.degree = n ∧ r.degree = n ∧ 
    (∀ x : ℝ, q.eval x = 0 → r.eval x = 0) ∧
    (∀ y : ℝ, r.eval y = 0 → q.eval y = 0) ∧
    q.leadingCoeff = 1 ∧ r.leadingCoeff = 1 ∧ 
    p = (q + r) / 2 := 
sorry

end exists_q_r_polynomials_l208_208774


namespace rectangular_plot_width_l208_208496

theorem rectangular_plot_width :
  ∀ (length width : ℕ), 
    length = 60 → 
    ∀ (poles spacing : ℕ), 
      poles = 44 → 
      spacing = 5 → 
      2 * length + 2 * width = poles * spacing →
      width = 50 :=
by
  intros length width h_length poles spacing h_poles h_spacing h_perimeter
  rw [h_length, h_poles, h_spacing] at h_perimeter
  linarith

end rectangular_plot_width_l208_208496


namespace birds_meeting_distance_l208_208308

theorem birds_meeting_distance :
  ∀ (d distance speed1 speed2: ℕ),
  distance = 20 →
  speed1 = 4 →
  speed2 = 1 →
  (d / speed1) = ((distance - d) / speed2) →
  d = 16 :=
by
  intros d distance speed1 speed2 hdist hspeed1 hspeed2 htime
  sorry

end birds_meeting_distance_l208_208308


namespace max_zoo_area_l208_208960

theorem max_zoo_area (length width x y : ℝ) (h1 : length = 16) (h2 : width = 8 - x) (h3 : y = x * (8 - x)) : 
  ∃ M, ∀ x, 0 < x ∧ x < 8 → y ≤ M ∧ M = 16 :=
by
  sorry

end max_zoo_area_l208_208960


namespace line_eq_x_1_parallel_y_axis_l208_208931

theorem line_eq_x_1_parallel_y_axis (P : ℝ × ℝ) (hP : P = (1, 0)) (h_parallel : ∀ y : ℝ, (1, y) = P ∨ P = (1, y)) :
  ∃ x : ℝ, (∀ y : ℝ, P = (x, y)) → x = 1 := 
by 
  sorry

end line_eq_x_1_parallel_y_axis_l208_208931


namespace jose_to_haylee_ratio_l208_208834

variable (J : ℕ)

def haylee_guppies := 36
def charliz_guppies := J / 3
def nicolai_guppies := 4 * (J / 3)
def total_guppies := haylee_guppies + J + charliz_guppies + nicolai_guppies

theorem jose_to_haylee_ratio :
  haylee_guppies = 36 ∧ total_guppies = 84 →
  J / haylee_guppies = 1 / 2 :=
by
  intro h
  sorry

end jose_to_haylee_ratio_l208_208834


namespace vertical_asymptote_x_value_l208_208376

theorem vertical_asymptote_x_value (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 :=
by
  sorry

end vertical_asymptote_x_value_l208_208376


namespace min_cubes_required_l208_208724

def volume_of_box (L W H : ℕ) : ℕ := L * W * H
def volume_of_cube (v_cube : ℕ) : ℕ := v_cube
def minimum_number_of_cubes (V_box V_cube : ℕ) : ℕ := V_box / V_cube

theorem min_cubes_required :
  minimum_number_of_cubes (volume_of_box 12 16 6) (volume_of_cube 3) = 384 :=
by sorry

end min_cubes_required_l208_208724


namespace average_mileage_first_car_l208_208406

theorem average_mileage_first_car (X Y : ℝ) 
  (h1 : X + Y = 75) 
  (h2 : 25 * X + 35 * Y = 2275) : 
  X = 35 :=
by 
  sorry

end average_mileage_first_car_l208_208406


namespace percentage_hate_german_l208_208314

def percentage_hate_math : ℝ := 0.01
def percentage_hate_english : ℝ := 0.02
def percentage_hate_french : ℝ := 0.01
def percentage_hate_all_four : ℝ := 0.08

theorem percentage_hate_german : (0.08 - (0.01 + 0.02 + 0.01)) = 0.04 :=
by
  -- Proof goes here
  sorry

end percentage_hate_german_l208_208314


namespace num_ways_to_choose_officers_same_gender_l208_208351

-- Definitions based on conditions
def num_members : Nat := 24
def num_boys : Nat := 12
def num_girls : Nat := 12
def num_officers : Nat := 3

-- Theorem statement using these definitions
theorem num_ways_to_choose_officers_same_gender :
  (num_boys * (num_boys-1) * (num_boys-2) * 2) = 2640 :=
by
  sorry

end num_ways_to_choose_officers_same_gender_l208_208351


namespace mod_equiv_n_l208_208732

theorem mod_equiv_n (n : ℤ) : 0 ≤ n ∧ n < 9 ∧ -1234 % 9 = n := 
by
  sorry

end mod_equiv_n_l208_208732


namespace problem_statement_l208_208606

open Real

noncomputable def f (x : ℝ) : ℝ := 10^x

theorem problem_statement : f (log 2) * f (log 5) = 10 :=
by {
  -- Note: Proof is omitted as indicated in the procedure.
  sorry
}

end problem_statement_l208_208606


namespace natural_numbers_partition_l208_208372

def isSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

def subsets_with_square_sum (n : ℕ) : Prop :=
  ∀ (A B : Finset ℕ), (A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅) →
  ∃ (a b : ℕ), a ≠ b ∧ isSquare (a + b) ∧ (a ∈ A ∨ a ∈ B) ∧ (b ∈ A ∨ b ∈ B)

theorem natural_numbers_partition (n : ℕ) : n ≥ 15 → subsets_with_square_sum n := 
sorry

end natural_numbers_partition_l208_208372


namespace Joey_age_l208_208812

theorem Joey_age (J B : ℕ) (h1 : J + 5 = B) (h2 : J - 4 = B - J) : J = 9 :=
by 
  sorry

end Joey_age_l208_208812


namespace boat_speed_in_still_water_l208_208745

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := by
  sorry

end boat_speed_in_still_water_l208_208745


namespace polygon_sides_eq_seven_l208_208083

-- Given conditions:
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360
def difference_in_angles (n : ℕ) : ℝ := sum_interior_angles n - sum_exterior_angles

-- Proof statement:
theorem polygon_sides_eq_seven (n : ℕ) (h : difference_in_angles n = 540) : n = 7 := sorry

end polygon_sides_eq_seven_l208_208083


namespace douglas_percent_votes_l208_208405

def percentageOfTotalVotesWon (votes_X votes_Y: ℕ) (percent_X percent_Y: ℕ) : ℕ :=
  let total_votes_Douglas : ℕ := (percent_X * 2 * votes_X + percent_Y * votes_Y)
  let total_votes_cast : ℕ := 3 * votes_Y
  (total_votes_Douglas * 100 / total_votes_cast)

theorem douglas_percent_votes (votes_X votes_Y : ℕ) (h_ratio : 2 * votes_X = votes_Y)
  (h_perc_X : percent_X = 64)
  (h_perc_Y : percent_Y = 46) :
  percentageOfTotalVotesWon votes_X votes_Y 64 46 = 58 := by
    sorry

end douglas_percent_votes_l208_208405


namespace all_are_knights_l208_208797

-- Definitions for inhabitants as either knights or knaves
inductive Inhabitant
| Knight : Inhabitant
| Knave : Inhabitant

open Inhabitant

-- Functions that determine if an inhabitant is a knight or a knave
def is_knight (x : Inhabitant) : Prop :=
  x = Knight

def is_knave (x : Inhabitant) : Prop :=
  x = Knave

-- Given conditions
axiom A : Inhabitant
axiom B : Inhabitant
axiom C : Inhabitant

axiom statement_A : is_knight A → is_knight B
axiom statement_B : is_knight B → (is_knight A → is_knight C)

-- The proof goal
theorem all_are_knights : is_knight A ∧ is_knight B ∧ is_knight C := by
  sorry

end all_are_knights_l208_208797


namespace simplify_expression_l208_208647

-- Defining the variables involved
variables (b : ℝ)

-- The theorem statement that needs to be proven
theorem simplify_expression : 3 * b * (3 * b^2 - 2 * b + 1) + 2 * b^2 = 9 * b^3 - 4 * b^2 + 3 * b :=
by
  sorry

end simplify_expression_l208_208647


namespace average_charge_proof_l208_208101

noncomputable def averageChargePerPerson
  (chargeFirstDay : ℝ)
  (chargeSecondDay : ℝ)
  (chargeThirdDay : ℝ)
  (chargeFourthDay : ℝ)
  (ratioFirstDay : ℝ)
  (ratioSecondDay : ℝ)
  (ratioThirdDay : ℝ)
  (ratioFourthDay : ℝ)
  : ℝ :=
  let totalRevenue := ratioFirstDay * chargeFirstDay + ratioSecondDay * chargeSecondDay + ratioThirdDay * chargeThirdDay + ratioFourthDay * chargeFourthDay
  let totalVisitors := ratioFirstDay + ratioSecondDay + ratioThirdDay + ratioFourthDay
  totalRevenue / totalVisitors

theorem average_charge_proof :
  averageChargePerPerson 25 15 7.5 2.5 3 7 11 19 = 7.75 := by
  simp [averageChargePerPerson]
  sorry

end average_charge_proof_l208_208101


namespace shuttle_speed_in_km_per_sec_l208_208491

variable (speed_mph : ℝ) (miles_to_km : ℝ) (hour_to_sec : ℝ)

theorem shuttle_speed_in_km_per_sec
  (h_speed_mph : speed_mph = 18000)
  (h_miles_to_km : miles_to_km = 1.60934)
  (h_hour_to_sec : hour_to_sec = 3600) :
  (speed_mph * miles_to_km) / hour_to_sec = 8.046 := by
sorry

end shuttle_speed_in_km_per_sec_l208_208491


namespace second_layer_ratio_l208_208418

theorem second_layer_ratio
  (first_layer_sugar third_layer_sugar : ℕ)
  (third_layer_factor : ℕ)
  (h1 : first_layer_sugar = 2)
  (h2 : third_layer_sugar = 12)
  (h3 : third_layer_factor = 3) :
  third_layer_sugar = third_layer_factor * (2 * first_layer_sugar) →
  second_layer_factor = 2 :=
by
  sorry

end second_layer_ratio_l208_208418


namespace find_a_in_subset_l208_208930

theorem find_a_in_subset 
  (A : Set ℝ)
  (B : Set ℝ)
  (hA : A = { x | x^2 ≠ 1 })
  (hB : ∃ a : ℝ, B = { x | a * x = 1 })
  (h_subset : B ⊆ A) : 
  ∃ a : ℝ, a = 0 ∨ a = 1 ∨ a = -1 := 
by
  sorry

end find_a_in_subset_l208_208930


namespace number_is_10_l208_208435

theorem number_is_10 (x : ℕ) (h : x * 15 = 150) : x = 10 :=
sorry

end number_is_10_l208_208435


namespace best_approximation_of_x_squared_l208_208955

theorem best_approximation_of_x_squared
  (x : ℝ) (A B C D E : ℝ)
  (h1 : -2 < -1)
  (h2 : -1 < 0)
  (h3 : 0 < 1)
  (h4 : 1 < 2)
  (hx : -1 < x ∧ x < 0)
  (hC : 0 < C ∧ C < 1) :
  x^2 = C :=
sorry

end best_approximation_of_x_squared_l208_208955


namespace fifth_number_in_pascal_row_l208_208246

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l208_208246


namespace product_remainder_mod_7_l208_208667

theorem product_remainder_mod_7 (a b c : ℕ) (ha : a % 7 = 2) (hb : b % 7 = 3) (hc : c % 7 = 5) :
    (a * b * c) % 7 = 2 :=
by
  sorry

end product_remainder_mod_7_l208_208667


namespace find_y_intercept_l208_208555

theorem find_y_intercept (m x y b : ℤ) (h_slope : m = 2) (h_point : (x, y) = (259, 520)) :
  y = m * x + b → b = 2 :=
by {
  sorry
}

end find_y_intercept_l208_208555


namespace badges_before_exchange_l208_208242

theorem badges_before_exchange (V T : ℕ) (h1 : V = T + 5) (h2 : 76 * V + 20 * T = 80 * T + 24 * V - 100) :
  V = 50 ∧ T = 45 :=
by
  sorry

end badges_before_exchange_l208_208242


namespace pure_imaginary_number_implies_x_eq_1_l208_208522

theorem pure_imaginary_number_implies_x_eq_1 (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x + 1 ≠ 0) : x = 1 :=
sorry

end pure_imaginary_number_implies_x_eq_1_l208_208522


namespace playground_length_l208_208803

theorem playground_length
  (P : ℕ)
  (B : ℕ)
  (h1 : P = 1200)
  (h2 : B = 500)
  (h3 : P = 2 * (100 + B)) :
  100 = 100 :=
 by sorry

end playground_length_l208_208803


namespace complement_inter_of_A_and_B_l208_208866

open Set

variable (U A B : Set ℕ)

theorem complement_inter_of_A_and_B:
  U = {1, 2, 3, 4, 5}
  ∧ A = {1, 2, 3}
  ∧ B = {2, 3, 4} 
  → U \ (A ∩ B) = {1, 4, 5} :=
by
  sorry

end complement_inter_of_A_and_B_l208_208866


namespace find_m_containing_2015_l208_208366

theorem find_m_containing_2015 : 
  ∃ n : ℕ, ∀ k, 0 ≤ k ∧ k < n → 2015 = n^3 → (1979 + 2*k < 2015 ∧ 2015 < 1979 + 2*k + 2*n) :=
by
  sorry

end find_m_containing_2015_l208_208366


namespace least_third_side_length_l208_208580

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l208_208580


namespace q_poly_correct_l208_208748

open Polynomial

noncomputable def q : Polynomial ℚ := 
  -(C 1) * X^6 + C 4 * X^4 + C 21 * X^3 + C 15 * X^2 + C 14 * X + C 3

theorem q_poly_correct : 
  ∀ x : Polynomial ℚ,
  q + (X^6 + 4 * X^4 + 5 * X^3 + 12 * X) = 
  (8 * X^4 + 26 * X^3 + 15 * X^2 + 26 * X + C 3) := by sorry

end q_poly_correct_l208_208748


namespace gcd_lcm_product_l208_208536

theorem gcd_lcm_product (a b: ℕ) (h1 : a = 36) (h2 : b = 210) :
  Nat.gcd a b * Nat.lcm a b = 7560 := 
by 
  sorry

end gcd_lcm_product_l208_208536


namespace gcd_47_pow6_plus_1_l208_208345

theorem gcd_47_pow6_plus_1 (h_prime : Prime 47) : 
  Nat.gcd (47^6 + 1) (47^6 + 47^3 + 1) = 1 := 
by 
  sorry

end gcd_47_pow6_plus_1_l208_208345


namespace handshakes_l208_208696

open Nat

theorem handshakes : ∃ x : ℕ, 4 + 3 + 2 + 1 + x = 10 ∧ x = 2 :=
by
  existsi 2
  simp
  sorry

end handshakes_l208_208696


namespace multiply_abs_value_l208_208511

theorem multiply_abs_value : -2 * |(-3 : ℤ)| = -6 := by
  sorry

end multiply_abs_value_l208_208511


namespace danny_bottle_cap_count_l208_208029

theorem danny_bottle_cap_count 
  (initial_caps : Int) 
  (found_caps : Int) 
  (final_caps : Int) 
  (h1 : initial_caps = 6) 
  (h2 : found_caps = 22) 
  (h3 : final_caps = initial_caps + found_caps) : 
  final_caps = 28 :=
by
  sorry

end danny_bottle_cap_count_l208_208029


namespace amy_7_mile_run_time_l208_208801

-- Define the conditions
variable (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ)

-- State the conditions
def conditions : Prop :=
  rachel_time_per_9_miles = 36 ∧
  amy_time_per_4_miles = 1 / 3 * rachel_time_per_9_miles ∧
  amy_time_per_mile = amy_time_per_4_miles / 4 ∧
  amy_time_per_7_miles = amy_time_per_mile * 7

-- The main statement to prove
theorem amy_7_mile_run_time (rachel_time_per_9_miles : ℕ) (amy_time_per_4_miles : ℕ) (amy_time_per_mile : ℕ) (amy_time_per_7_miles: ℕ) :
  conditions rachel_time_per_9_miles amy_time_per_4_miles amy_time_per_mile amy_time_per_7_miles → 
  amy_time_per_7_miles = 21 := 
by
  intros h
  sorry

end amy_7_mile_run_time_l208_208801


namespace value_of_xy_l208_208088

noncomputable def distinct_nonzero_reals (x y : ℝ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y

theorem value_of_xy (x y : ℝ) (h : distinct_nonzero_reals x y) (h_eq : x + 4 / x = y + 4 / y) :
  x * y = 4 :=
sorry

end value_of_xy_l208_208088


namespace quincy_sold_more_than_jake_l208_208926

theorem quincy_sold_more_than_jake :
  ∀ (T Jake : ℕ), Jake = 2 * T + 15 → 4000 = 100 * (T + Jake) → 4000 - Jake = 3969 :=
by
  intros T Jake hJake hQuincy
  sorry

end quincy_sold_more_than_jake_l208_208926


namespace expression_equals_answer_l208_208712

noncomputable def evaluate_expression : ℚ :=
  (2011^2 * 2012 - 2013) / Nat.factorial 2012 +
  (2013^2 * 2014 - 2015) / Nat.factorial 2014

theorem expression_equals_answer :
  evaluate_expression = 
  1 / Nat.factorial 2009 + 
  1 / Nat.factorial 2010 - 
  1 / Nat.factorial 2013 - 
  1 / Nat.factorial 2014 :=
by
  sorry

end expression_equals_answer_l208_208712


namespace min_value_expression_l208_208799

theorem min_value_expression (x y z : ℝ) (h : x - 2 * y + 2 * z = 5) : (x + 5) ^ 2 + (y - 1) ^ 2 + (z + 3) ^ 2 ≥ 36 :=
by
  sorry

end min_value_expression_l208_208799


namespace greatest_integer_l208_208744

theorem greatest_integer (m : ℕ) (h1 : 0 < m) (h2 : m < 150)
  (h3 : ∃ a : ℤ, m = 9 * a - 2) (h4 : ∃ b : ℤ, m = 5 * b + 4) :
  m = 124 := 
sorry

end greatest_integer_l208_208744


namespace block_of_flats_l208_208585

theorem block_of_flats :
  let total_floors := 12
  let half_floors := total_floors / 2
  let apartments_per_half_floor := 6
  let max_residents_per_apartment := 4
  let total_max_residents := 264
  let apartments_on_half_floors := half_floors * apartments_per_half_floor
  ∃ (x : ℝ), 
    4 * (apartments_on_half_floors + half_floors * x) = total_max_residents ->
    x = 5 :=
sorry

end block_of_flats_l208_208585


namespace equation_has_real_root_l208_208413

theorem equation_has_real_root (x : ℝ) : (x^3 + 3 = 0) ↔ (x = -((3:ℝ)^(1/3))) :=
sorry

end equation_has_real_root_l208_208413


namespace rice_mixed_grain_amount_l208_208384

theorem rice_mixed_grain_amount (total_rice : ℕ) (sample_size : ℕ) (mixed_in_sample : ℕ) (proportion : ℚ) 
    (h1 : total_rice = 1536) 
    (h2 : sample_size = 256)
    (h3 : mixed_in_sample = 18)
    (h4 : proportion = mixed_in_sample / sample_size) : 
    total_rice * proportion = 108 :=
  sorry

end rice_mixed_grain_amount_l208_208384


namespace failed_students_calculation_l208_208986

theorem failed_students_calculation (total_students : ℕ) (percentage_passed : ℕ)
  (h_total : total_students = 840) (h_passed : percentage_passed = 35) :
  (total_students * (100 - percentage_passed) / 100) = 546 :=
by
  sorry

end failed_students_calculation_l208_208986


namespace quadratic_distinct_real_roots_l208_208579

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + m - 1 = 0) ∧ (x2^2 - 4*x2 + m - 1 = 0)) → m < 5 := sorry

end quadratic_distinct_real_roots_l208_208579


namespace probability_sum_greater_than_six_l208_208556

variable (A : Finset ℕ) (B : Finset ℕ)
variable (balls_in_A : A = {1, 2}) (balls_in_B : B = {3, 4, 5, 6})

theorem probability_sum_greater_than_six : 
  (∃ selected_pair ∈ (A.product B), selected_pair.1 + selected_pair.2 > 6) →
  (Finset.filter (λ pair => pair.1 + pair.2 > 6) (A.product B)).card / 
  (A.product B).card = 3 / 8 := sorry

end probability_sum_greater_than_six_l208_208556


namespace inequality_three_integer_solutions_l208_208525

theorem inequality_three_integer_solutions (c : ℤ) :
  (∃ s1 s2 s3 : ℤ, s1 < s2 ∧ s2 < s3 ∧ 
    (∀ x : ℤ, x^2 + c * x + 1 ≤ 0 ↔ x = s1 ∨ x = s2 ∨ x = s3)) ↔ (c = -4 ∨ c = 4) := 
by 
  sorry

end inequality_three_integer_solutions_l208_208525


namespace fraction_value_l208_208713

theorem fraction_value :
  (0.02 ^ 2 + 0.52 ^ 2 + 0.035 ^ 2) / (0.002 ^ 2 + 0.052 ^ 2 + 0.0035 ^ 2) = 100 := by
    sorry

end fraction_value_l208_208713


namespace max_super_bishops_l208_208677

/--
A "super-bishop" attacks another "super-bishop" if they are on the
same diagonal, there are no pieces between them, and the next cell
along the diagonal after the "super-bishop" B is empty. Given these
conditions, prove that the maximum number of "super-bishops" that can
be placed on a standard 8x8 chessboard such that each one attacks at
least one other is 32.
-/
theorem max_super_bishops (n : ℕ) (chessboard : ℕ → ℕ → Prop) (super_bishop : ℕ → ℕ → Prop)
  (attacks : ∀ {x₁ y₁ x₂ y₂}, super_bishop x₁ y₁ → super_bishop x₂ y₂ →
            (x₁ - x₂ = y₁ - y₂ ∨ x₁ + y₁ = x₂ + y₂) →
            (∀ x y, super_bishop x y → (x < min x₁ x₂ ∨ x > max x₁ x₂ ∨ y < min y₁ y₂ ∨ y > max y₁ y₂)) →
            chessboard (x₂ + (x₁ - x₂)) (y₂ + (y₁ - y₂))) :
  ∃ k, k = 32 ∧ (∀ x y, super_bishop x y → x < 8 ∧ y < 8) → k ≤ n :=
sorry

end max_super_bishops_l208_208677


namespace decimal_to_binary_25_l208_208464

theorem decimal_to_binary_25 : (25 : Nat) = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_25_l208_208464


namespace technicians_count_l208_208560

/-- Given a workshop with 49 workers, where the average salary of all workers 
    is Rs. 8000, the average salary of the technicians is Rs. 20000, and the
    average salary of the rest is Rs. 6000, prove that the number of 
    technicians is 7. -/
theorem technicians_count (T R : ℕ) (h1 : T + R = 49) (h2 : 10 * T + 3 * R = 196) : T = 7 := 
by
  sorry

end technicians_count_l208_208560


namespace star_test_one_star_test_two_l208_208074

def star (x y : ℤ) : ℤ :=
  if x = 0 then Int.natAbs y
  else if y = 0 then Int.natAbs x
  else if (x < 0) = (y < 0) then Int.natAbs x + Int.natAbs y
  else -(Int.natAbs x + Int.natAbs y)

theorem star_test_one :
  star 11 (star 0 (-12)) = 23 :=
by
  sorry

theorem star_test_two (a : ℤ) :
  2 * (2 * star 1 a) - 1 = 3 * a ↔ a = 3 ∨ a = -5 :=
by
  sorry

end star_test_one_star_test_two_l208_208074


namespace vessel_base_length_l208_208524

noncomputable def volume_of_cube (side: ℝ) : ℝ :=
  side ^ 3

noncomputable def volume_displaced (length breadth height: ℝ) : ℝ :=
  length * breadth * height

theorem vessel_base_length
  (breadth : ℝ) 
  (cube_edge : ℝ)
  (water_rise : ℝ)
  (displaced_volume : ℝ) 
  (h1 : breadth = 30) 
  (h2 : cube_edge = 30) 
  (h3 : water_rise = 15) 
  (h4 : volume_of_cube cube_edge = displaced_volume) :
  volume_displaced (displaced_volume / (breadth * water_rise)) breadth water_rise = displaced_volume :=
  by
  sorry

end vessel_base_length_l208_208524


namespace correct_polynomial_multiplication_l208_208465

theorem correct_polynomial_multiplication (a b : ℤ) (x : ℝ)
  (h1 : 2 * b - 3 * a = 11)
  (h2 : 2 * b + a = -9) :
  (2 * x + a) * (3 * x + b) = 6 * x^2 - 19 * x + 10 := by
  sorry

end correct_polynomial_multiplication_l208_208465


namespace number_to_multiply_l208_208025

theorem number_to_multiply (a b x : ℝ) (h1 : x * a = 4 * b) (h2 : a * b ≠ 0) (h3 : a / 4 = b / 3) : x = 3 :=
sorry

end number_to_multiply_l208_208025


namespace intersection_A_B_l208_208272

-- Define the set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, x + 1) }

-- Define the set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, -2*x + 4) }

-- State the theorem to prove A ∩ B = {(1, 2)}
theorem intersection_A_B : A ∩ B = { (1, 2) } :=
by
  sorry

end intersection_A_B_l208_208272


namespace problem_solution_l208_208852

theorem problem_solution (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a + b = 1 :=
sorry

end problem_solution_l208_208852


namespace predict_height_at_age_10_l208_208047

def regression_line := fun (x : ℝ) => 7.19 * x + 73.93

theorem predict_height_at_age_10 :
  regression_line 10 = 145.83 :=
by
  sorry

end predict_height_at_age_10_l208_208047


namespace triangle_area_l208_208836

theorem triangle_area (A B C : ℝ × ℝ) (hA : A = (0, 0)) (hB : B = (0, 8)) (hC : C = (10, 15)) : 
  let base := 8
  let height := 10
  let area := 1 / 2 * base * height
  area = 40.0 :=
by
  sorry

end triangle_area_l208_208836


namespace distance_between_foci_of_hyperbola_l208_208421

theorem distance_between_foci_of_hyperbola :
  ∀ x y : ℝ, (x^2 - 8 * x - 16 * y^2 - 16 * y = 48) → (∃ c : ℝ, 2 * c = 2 * Real.sqrt 63.75) :=
by
  sorry

end distance_between_foci_of_hyperbola_l208_208421


namespace area_of_transformed_region_l208_208729

theorem area_of_transformed_region : 
  let T : ℝ := 15
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![6, -2]]
  (abs (Matrix.det A) * T = 450) := 
  sorry

end area_of_transformed_region_l208_208729


namespace simplified_value_of_sum_l208_208039

theorem simplified_value_of_sum :
  (-1)^(2004) + (-1)^(2005) + 1^(2006) - 1^(2007) = -2 := by
  sorry

end simplified_value_of_sum_l208_208039


namespace factor_sum_l208_208274

theorem factor_sum (P Q R : ℤ) (h : ∃ (b c : ℤ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + R*x + Q) : 
  P + Q + R = 11*P - 1 := 
sorry

end factor_sum_l208_208274


namespace functional_eq_implies_odd_l208_208445

variable (f : ℝ → ℝ)

def condition (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (x * f y) = y * f x

theorem functional_eq_implies_odd (h : condition f) : ∀ x : ℝ, f (-x) = -f x :=
sorry

end functional_eq_implies_odd_l208_208445


namespace remaining_pencils_l208_208966

-- Define the initial conditions
def initial_pencils : Float := 56.0
def pencils_given : Float := 9.0

-- Formulate the theorem stating that the remaining pencils = 47.0
theorem remaining_pencils : initial_pencils - pencils_given = 47.0 := by
  sorry

end remaining_pencils_l208_208966


namespace diagonal_length_of_regular_hexagon_l208_208940

-- Define a structure for the hexagon with a given side length
structure RegularHexagon (s : ℝ) :=
(side_length : ℝ := s)

-- Prove that the length of diagonal DB in a regular hexagon with side length 12 is 12√3
theorem diagonal_length_of_regular_hexagon (H : RegularHexagon 12) : 
  ∃ DB : ℝ, DB = 12 * Real.sqrt 3 :=
by
  sorry

end diagonal_length_of_regular_hexagon_l208_208940


namespace units_digit_of_power_17_l208_208519

theorem units_digit_of_power_17 (n : ℕ) (k : ℕ) (h_n4 : n % 4 = 3) : (17^n) % 10 = 3 :=
  by
  -- Since units digits of powers repeat every 4
  sorry

-- Specific problem instance
example : (17^1995) % 10 = 3 := units_digit_of_power_17 1995 17 (by norm_num)

end units_digit_of_power_17_l208_208519


namespace real_value_of_b_l208_208138

open Real

theorem real_value_of_b : ∃ x : ℝ, (x^2 - 2 * x + 1 = 0) ∧ (x^2 + x - 2 = 0) :=
by
  sorry

end real_value_of_b_l208_208138


namespace distance_from_P_to_focus_l208_208068

-- Define the parabola equation and the definition of the point P
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the given condition that P's distance to the x-axis is 12
def point_P (x y : ℝ) : Prop := parabola x y ∧ |y| = 12

-- The Lean proof problem statement
theorem distance_from_P_to_focus :
  ∃ (x y : ℝ), point_P x y → dist (x, y) (4, 0) = 13 :=
by {
  sorry   -- proof to be completed
}

end distance_from_P_to_focus_l208_208068


namespace sqrt_12_lt_4_l208_208301

theorem sqrt_12_lt_4 : Real.sqrt 12 < 4 := sorry

end sqrt_12_lt_4_l208_208301


namespace cost_of_bread_l208_208084

-- Definition of the conditions
def total_purchase_amount : ℕ := 205  -- in cents
def amount_given_to_cashier : ℕ := 700  -- in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def num_nickels_received : ℕ := 8

-- Statement of the problem
theorem cost_of_bread :
  (∃ (B C : ℕ), B + C = total_purchase_amount ∧
                  amount_given_to_cashier - total_purchase_amount = 
                  (quarter_value + dime_value + num_nickels_received * nickel_value + 420) ∧
                  B = 125) :=
by
  -- Skipping the proof
  sorry

end cost_of_bread_l208_208084


namespace find_f1_find_f8_inequality_l208_208076

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_pos : ∀ x : ℝ, 0 < x → 0 < f x
axiom f_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y
axiom f_multiplicative : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x * f y
axiom f_of_2 : f 2 = 4

-- Statements to prove
theorem find_f1 : f 1 = 1 := sorry
theorem find_f8 : f 8 = 64 := sorry
theorem inequality : ∀ x : ℝ, 3 < x → x ≤ 7 / 2 → 16 * f (1 / (x - 3)) ≥ f (2 * x + 1) := sorry

end find_f1_find_f8_inequality_l208_208076


namespace coefficient_of_x7_in_expansion_eq_15_l208_208750

open Nat

noncomputable def binomial (n k : ℕ) : ℕ := n.choose k

theorem coefficient_of_x7_in_expansion_eq_15 (a : ℝ) (hbinom : binomial 10 3 * (-a) ^ 3 = 15) : a = -1 / 2 := by
  sorry

end coefficient_of_x7_in_expansion_eq_15_l208_208750


namespace y1_mul_y2_eq_one_l208_208557

theorem y1_mul_y2_eq_one (x1 x2 y1 y2 : ℝ) (h1 : y1^2 = x1) (h2 : y2^2 = x2) 
  (h3 : y1 / (y1^2 - 1) = - (y2 / (y2^2 - 1))) (h4 : y1 + y2 ≠ 0) : y1 * y2 = 1 :=
sorry

end y1_mul_y2_eq_one_l208_208557


namespace Kylie_US_coins_left_l208_208962

-- Define the given conditions
def initial_US_coins : ℝ := 15
def Euro_coins : ℝ := 13
def Canadian_coins : ℝ := 8
def US_coins_given_to_Laura : ℝ := 21
def Euro_to_US_rate : ℝ := 1.18
def Canadian_to_US_rate : ℝ := 0.78

-- Define the conversions
def Euro_to_US : ℝ := Euro_coins * Euro_to_US_rate
def Canadian_to_US : ℝ := Canadian_coins * Canadian_to_US_rate
def total_US_before_giving : ℝ := initial_US_coins + Euro_to_US + Canadian_to_US
def US_left_with : ℝ := total_US_before_giving - US_coins_given_to_Laura

-- Statement of the problem to be proven
theorem Kylie_US_coins_left :
  US_left_with = 15.58 := by
  sorry

end Kylie_US_coins_left_l208_208962


namespace find_S9_l208_208718

-- Setting up basic definitions for arithmetic sequence and the sum of its terms
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d
def sum_arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := n * (a + arithmetic_seq a d n) / 2

-- Given conditions
variables (a d : ℤ)
axiom h : 2 * arithmetic_seq a d 3 = 3 + a

-- Theorem to prove
theorem find_S9 : sum_arithmetic_seq a d 9 = 27 :=
by {
  sorry
}

end find_S9_l208_208718


namespace sectorChordLength_correct_l208_208467

open Real

noncomputable def sectorChordLength (r α : ℝ) : ℝ :=
  2 * r * sin (α / 2)

theorem sectorChordLength_correct :
  ∃ (r α : ℝ), (1/2) * α * r^2 = 1 ∧ 2 * r + α * r = 4 ∧ sectorChordLength r α = 2 * sin 1 :=
by {
  sorry
}

end sectorChordLength_correct_l208_208467


namespace eight_is_100_discerning_nine_is_not_100_discerning_l208_208991

-- Define what it means to be b-discerning
def is_b_discerning (n b : ℕ) : Prop :=
  ∃ S : Finset ℕ, S.card = n ∧ (∀ (U V : Finset ℕ), U ≠ V ∧ U ⊆ S ∧ V ⊆ S → U.sum id ≠ V.sum id)

-- Prove that 8 is 100-discerning
theorem eight_is_100_discerning : is_b_discerning 8 100 :=
sorry

-- Prove that 9 is not 100-discerning
theorem nine_is_not_100_discerning : ¬is_b_discerning 9 100 :=
sorry

end eight_is_100_discerning_nine_is_not_100_discerning_l208_208991


namespace express_x_in_terms_of_y_l208_208648

theorem express_x_in_terms_of_y (x y : ℝ) (h : 3 * x - 4 * y = 8) : x = (4 * y + 8) / 3 :=
sorry

end express_x_in_terms_of_y_l208_208648


namespace student_percentage_to_pass_l208_208973

/-- A student needs to obtain 50% of the total marks to pass given the conditions:
    1. The student got 200 marks.
    2. The student failed by 20 marks.
    3. The maximum marks are 440. -/
theorem student_percentage_to_pass : 
  ∀ (student_marks : ℕ) (failed_by : ℕ) (max_marks : ℕ),
  student_marks = 200 → failed_by = 20 → max_marks = 440 →
  (student_marks + failed_by) / max_marks * 100 = 50 := 
by
  intros student_marks failed_by max_marks h1 h2 h3
  sorry

end student_percentage_to_pass_l208_208973


namespace number_of_people_going_on_trip_l208_208038

theorem number_of_people_going_on_trip
  (bags_per_person : ℕ)
  (weight_per_bag : ℕ)
  (total_luggage_capacity : ℕ)
  (additional_capacity : ℕ)
  (bags_per_additional_capacity : ℕ)
  (h1 : bags_per_person = 5)
  (h2 : weight_per_bag = 50)
  (h3 : total_luggage_capacity = 6000)
  (h4 : additional_capacity = 90) :
  (total_luggage_capacity + (bags_per_additional_capacity * weight_per_bag)) / (weight_per_bag * bags_per_person) = 42 := 
by
  simp [h1, h2, h3, h4]
  repeat { sorry }

end number_of_people_going_on_trip_l208_208038


namespace hexagon_probability_same_length_l208_208094

-- Define the problem
theorem hexagon_probability_same_length (T : Finset (Fin 15)) :
  let num_favorable_outcomes := 33
  let total_possible_outcomes := 105
  (num_favorable_outcomes / total_possible_outcomes) = (11 / 35) :=
by
  sorry

end hexagon_probability_same_length_l208_208094


namespace bread_slices_leftover_l208_208546

-- Definitions based on conditions provided in the problem
def total_bread_slices : ℕ := 2 * 20
def total_ham_slices : ℕ := 2 * 8
def sandwiches_made : ℕ := total_ham_slices
def bread_slices_needed : ℕ := sandwiches_made * 2

-- Theorem we want to prove
theorem bread_slices_leftover : total_bread_slices - bread_slices_needed = 8 :=
by 
    -- Insert steps of proof here
    sorry

end bread_slices_leftover_l208_208546


namespace students_not_skating_nor_skiing_l208_208689

theorem students_not_skating_nor_skiing (total_students skating_students skiing_students both_students : ℕ)
  (h_total : total_students = 30)
  (h_skating : skating_students = 20)
  (h_skiing : skiing_students = 9)
  (h_both : both_students = 5) :
  total_students - (skating_students + skiing_students - both_students) = 6 :=
by
  sorry

end students_not_skating_nor_skiing_l208_208689


namespace find_pairs_l208_208863

noncomputable def possibleValues (α β : ℝ) : Prop :=
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = -(Real.pi/3) + 2*l*Real.pi) ∨
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = (Real.pi/3) + 2*l*Real.pi)

theorem find_pairs (α β : ℝ) (h1 : Real.sin (α - β) = Real.sin α - Real.sin β)
  (h2 : Real.cos (α - β) = Real.cos α - Real.cos β) :
  possibleValues α β :=
sorry

end find_pairs_l208_208863


namespace equation_one_solution_equation_two_solution_l208_208295

theorem equation_one_solution (x : ℝ) : (6 * x - 7 = 4 * x - 5) ↔ (x = 1) := by
  sorry

theorem equation_two_solution (x : ℝ) : ((x + 1) / 2 - 1 = 2 + (2 - x) / 4) ↔ (x = 4) := by
  sorry

end equation_one_solution_equation_two_solution_l208_208295


namespace sequence_is_increasing_l208_208507

theorem sequence_is_increasing :
  ∀ n m : ℕ, n < m → (1 - 2 / (n + 1) : ℝ) < (1 - 2 / (m + 1) : ℝ) :=
by
  intro n m hnm
  have : (2 : ℝ) / (n + 1) > 2 / (m + 1) :=
    sorry
  linarith [this]

end sequence_is_increasing_l208_208507


namespace determine_p_and_q_l208_208189

noncomputable def find_p_and_q (a : ℝ) (p q : ℝ) : Prop :=
  (∀ x : ℝ, x = 1 ∨ x = -1 → (x^4 + p * x^2 + q * x + a^2 = 0))

theorem determine_p_and_q (a p q : ℝ) (h : find_p_and_q a p q) : p = -(a^2 + 1) ∧ q = 0 :=
by
  -- The proof would go here.
  sorry

end determine_p_and_q_l208_208189


namespace unique_sum_of_squares_l208_208503

theorem unique_sum_of_squares (p : ℕ) (k : ℕ) (x y a b : ℤ) 
  (hp : Prime p) (h1 : p = 4 * k + 1) (hx : x^2 + y^2 = p) (ha : a^2 + b^2 = p) :
  (x = a ∨ x = -a) ∧ (y = b ∨ y = -b) ∨ (x = b ∨ x = -b) ∧ (y = a ∨ y = -a) :=
sorry

end unique_sum_of_squares_l208_208503


namespace matthew_total_time_on_failure_day_l208_208755

-- Define the conditions as variables
def assembly_time : ℝ := 1 -- hours
def usual_baking_time : ℝ := 1.5 -- hours
def decoration_time : ℝ := 1 -- hours
def baking_factor : ℝ := 2 -- Factor by which baking time increased on that day

-- Prove that the total time taken is 5 hours
theorem matthew_total_time_on_failure_day : 
  (assembly_time + (usual_baking_time * baking_factor) + decoration_time) = 5 :=
by {
  sorry
}

end matthew_total_time_on_failure_day_l208_208755


namespace circle_area_conversion_l208_208901

-- Define the given diameter
def diameter (d : ℝ) := d = 8

-- Define the radius calculation
def radius (r : ℝ) := r = 4

-- Define the formula for the area of the circle in square meters
def area_sq_m (A : ℝ) := A = 16 * Real.pi

-- Define the conversion factor from square meters to square centimeters
def conversion_factor := 10000

-- Define the expected area in square centimeters
def area_sq_cm (A : ℝ) := A = 160000 * Real.pi

-- The theorem to prove
theorem circle_area_conversion (d r A_cm : ℝ) (h1 : diameter d) (h2 : radius r) (h3 : area_sq_cm A_cm) :
  A_cm = 160000 * Real.pi :=
by
  sorry

end circle_area_conversion_l208_208901


namespace smaller_cubes_total_l208_208397

theorem smaller_cubes_total (n : ℕ) (painted_edges_cubes : ℕ) 
  (h1 : ∀ (a b : ℕ), a ^ 3 = n) 
  (h2 : ∀ (c : ℕ), painted_edges_cubes = 12) 
  (h3 : ∀ (d e : ℕ), 12 <= 2 * d * e) 
  : n = 27 :=
by
  sorry

end smaller_cubes_total_l208_208397


namespace misha_needs_total_l208_208066

theorem misha_needs_total (
  current_amount : ℤ := 34
) (additional_amount : ℤ := 13) : 
  current_amount + additional_amount = 47 :=
by
  sorry

end misha_needs_total_l208_208066


namespace problem1_problem2_problem3_l208_208535

theorem problem1 : 999 * 999 + 1999 = 1000000 := by
  sorry

theorem problem2 : 9 * 72 * 125 = 81000 := by
  sorry

theorem problem3 : 416 - 327 + 184 - 273 = 0 := by
  sorry

end problem1_problem2_problem3_l208_208535


namespace b6_b8_equals_16_l208_208075

noncomputable def a_seq : ℕ → ℝ := sorry
noncomputable def b_seq : ℕ → ℝ := sorry

axiom a_arithmetic : ∃ d, ∀ n, a_seq (n + 1) = a_seq n + d
axiom b_geometric : ∃ r, ∀ n, b_seq (n + 1) = b_seq n * r
axiom a_nonzero : ∀ n, a_seq n ≠ 0
axiom a_eq : 2 * a_seq 3 - (a_seq 7)^2 + 2 * a_seq 11 = 0
axiom b7_eq_a7 : b_seq 7 = a_seq 7

theorem b6_b8_equals_16 : b_seq 6 * b_seq 8 = 16 := by
  sorry

end b6_b8_equals_16_l208_208075


namespace find_room_height_l208_208873

theorem find_room_height (l b d : ℕ) (h : ℕ) (hl : l = 12) (hb : b = 8) (hd : d = 17) :
  d = Int.sqrt (l^2 + b^2 + h^2) → h = 9 :=
by
  sorry

end find_room_height_l208_208873


namespace unique_common_tangent_l208_208147

noncomputable def f (x : ℝ) : ℝ := x ^ 2
noncomputable def g (a x : ℝ) : ℝ := a * Real.exp (x + 1)

theorem unique_common_tangent (a : ℝ) (h : a > 0) : 
  (∃ k x₁ x₂, k = 2 * x₁ ∧ k = a * Real.exp (x₂ + 1) ∧ k = (g a x₂ - f x₁) / (x₂ - x₁)) →
  a = 4 / Real.exp 3 :=
by
  sorry

end unique_common_tangent_l208_208147


namespace find_v_value_l208_208224

theorem find_v_value (x : ℝ) (v : ℝ) (h1 : x = 3.0) (h2 : 5 * x + v = 19) : v = 4 := by
  sorry

end find_v_value_l208_208224


namespace sum_of_values_satisfying_eq_l208_208756

theorem sum_of_values_satisfying_eq (x : ℝ) :
  (x^2 - 5 * x + 5 = 16) → ∀ r s : ℝ, (r + s = 5) :=
by
  sorry  -- Proof is omitted, looking to verify the structure only.

end sum_of_values_satisfying_eq_l208_208756


namespace polynomial_j_value_l208_208311

noncomputable def polynomial_roots_in_ap (a d : ℝ) : Prop :=
  let r1 := a
  let r2 := a + d
  let r3 := a + 2 * d
  let r4 := a + 3 * d
  ∀ (r : ℝ), r = r1 ∨ r = r2 ∨ r = r3 ∨ r = r4

theorem polynomial_j_value (a d : ℝ) (h_ap : polynomial_roots_in_ap a d)
  (h_poly : ∀ (x : ℝ), (x - (a)) * (x - (a + d)) * (x - (a + 2 * d)) * (x - (a + 3 * d)) = x^4 + j * x^2 + k * x + 256) :
  j = -80 :=
by
  sorry

end polynomial_j_value_l208_208311


namespace attendance_calculation_l208_208148

theorem attendance_calculation (total_students : ℕ) (attendance_rate : ℚ)
  (h1 : total_students = 120)
  (h2 : attendance_rate = 0.95) :
  total_students * attendance_rate = 114 := 
  sorry

end attendance_calculation_l208_208148


namespace jeremy_watermelons_l208_208802

theorem jeremy_watermelons :
  ∀ (total_watermelons : ℕ) (weeks : ℕ) (consumption_per_week : ℕ) (eaten_per_week : ℕ),
  total_watermelons = 30 →
  weeks = 6 →
  eaten_per_week = 3 →
  consumption_per_week = total_watermelons / weeks →
  (consumption_per_week - eaten_per_week) = 2 :=
by
  intros total_watermelons weeks consumption_per_week eaten_per_week h1 h2 h3 h4
  sorry

end jeremy_watermelons_l208_208802


namespace factorize_negative_quadratic_l208_208932

theorem factorize_negative_quadratic (x y : ℝ) : 
  -4 * x^2 + y^2 = (y - 2 * x) * (y + 2 * x) :=
by 
  sorry

end factorize_negative_quadratic_l208_208932


namespace initial_machines_l208_208429

theorem initial_machines (n x : ℕ) (hx : x > 0) (h : x / (4 * n) = x / 20) : n = 5 :=
by sorry

end initial_machines_l208_208429


namespace circle_sine_intersection_l208_208754

theorem circle_sine_intersection (h k r : ℝ) (hr : r > 0) :
  ∃ (n : ℕ), n > 16 ∧
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, (x - h)^2 + (2 * Real.sin x - k)^2 = r^2) ∧ xs.card = n :=
by
  sorry

end circle_sine_intersection_l208_208754


namespace consecutive_integers_eq_l208_208425

theorem consecutive_integers_eq (a b c d e: ℕ) (h1: b = a + 1) (h2: c = a + 2) (h3: d = a + 3) (h4: e = a + 4) (h5: a^2 + b^2 + c^2 = d^2 + e^2) : a = 10 :=
by
  sorry

end consecutive_integers_eq_l208_208425


namespace triangle_with_angle_ratio_obtuse_l208_208668

theorem triangle_with_angle_ratio_obtuse 
  (a b c : ℝ) 
  (h_sum : a + b + c = 180) 
  (h_ratio : a = 2 * d ∧ b = 2 * d ∧ c = 5 * d) : 
  90 < c :=
by
  sorry

end triangle_with_angle_ratio_obtuse_l208_208668


namespace tan_value_l208_208416

open Real

noncomputable def geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

noncomputable def arithmetic_seq (b : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem tan_value
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : geometric_seq a)
  (hb : arithmetic_seq b)
  (h_geom : a 0 * a 5 * a 10 = -3 * sqrt 3)
  (h_arith : b 0 + b 5 + b 10 = 7 * π) :
  tan ((b 2 + b 8) / (1 - a 3 * a 7)) = -sqrt 3 :=
sorry

end tan_value_l208_208416


namespace oliver_boxes_total_l208_208609

theorem oliver_boxes_total (initial_boxes : ℕ := 8) (additional_boxes : ℕ := 6) : initial_boxes + additional_boxes = 14 := 
by 
  sorry

end oliver_boxes_total_l208_208609


namespace inequality_proof_l208_208538

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a + b + c + d + 8 / (a*b + b*c + c*d + d*a) ≥ 6 := 
by
  sorry

end inequality_proof_l208_208538


namespace almond_walnut_ratio_is_5_to_2_l208_208810

-- Definitions based on conditions
variables (A W : ℕ)
def almond_ratio_to_walnut_ratio := A / (2 * W)
def weight_of_almonds := 250
def total_weight := 350
def weight_of_walnuts := total_weight - weight_of_almonds

-- Theorem to prove
theorem almond_walnut_ratio_is_5_to_2
  (h_ratio : almond_ratio_to_walnut_ratio A W = 250 / 100)
  (h_weights : weight_of_walnuts = 100) :
  A = 5 ∧ 2 * W = 2 := by
  sorry

end almond_walnut_ratio_is_5_to_2_l208_208810


namespace total_call_charges_l208_208527

-- Definitions based on conditions
def base_fee : ℝ := 39
def included_minutes : ℕ := 300
def excess_charge_per_minute : ℝ := 0.19

-- Given variables
variable (x : ℕ) -- excess minutes
variable (y : ℝ) -- total call charges

-- Theorem stating the relationship between y and x
theorem total_call_charges (h : x > 0) : y = 0.19 * x + 39 := 
by sorry

end total_call_charges_l208_208527


namespace inequality_proof_l208_208951

theorem inequality_proof (a b : ℤ) (ha : a > 0) (hb : b > 0) : a + b ≤ 1 + a * b :=
by
  sorry

end inequality_proof_l208_208951


namespace original_cylinder_weight_is_24_l208_208334

noncomputable def weight_of_original_cylinder (cylinder_weight cone_weight : ℝ) : Prop :=
  cylinder_weight = 3 * cone_weight

-- Given conditions in Lean 4
variables (cone_weight : ℝ) (h_cone_weight : cone_weight = 8)

-- Proof problem statement
theorem original_cylinder_weight_is_24 :
  weight_of_original_cylinder 24 cone_weight :=
by
  sorry

end original_cylinder_weight_is_24_l208_208334


namespace car_a_speed_l208_208929

theorem car_a_speed (d_A d_B v_B t v_A : ℝ)
  (h1 : d_A = 10)
  (h2 : v_B = 50)
  (h3 : t = 2.25)
  (h4 : d_A + 8 - d_B = v_A * t)
  (h5 : d_B = v_B * t) :
  v_A = 58 :=
by
  -- Work on the proof here
  sorry

end car_a_speed_l208_208929


namespace buyers_cake_and_muffin_l208_208887

theorem buyers_cake_and_muffin (total_buyers cake_buyers muffin_buyers neither_prob : ℕ) :
  total_buyers = 100 →
  cake_buyers = 50 →
  muffin_buyers = 40 →
  neither_prob = 26 →
  (cake_buyers + muffin_buyers - neither_prob) = 74 →
  90 - cake_buyers - muffin_buyers = neither_prob :=
by
  sorry

end buyers_cake_and_muffin_l208_208887


namespace fraction_zero_implies_x_is_minus_5_l208_208945

theorem fraction_zero_implies_x_is_minus_5 (x : ℝ) (h1 : (x + 5) / (x - 2) = 0) (h2 : x ≠ 2) : x = -5 := 
by
  sorry

end fraction_zero_implies_x_is_minus_5_l208_208945


namespace parabola_tangent_circle_radius_l208_208312

noncomputable def radius_of_tangent_circle : ℝ :=
  let r := 1 / 4
  r

theorem parabola_tangent_circle_radius :
  ∃ (r : ℝ), (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 1 - 4 * r) ∧ r = 1 / 4 :=
by
  use 1 / 4
  sorry

end parabola_tangent_circle_radius_l208_208312


namespace add_decimals_l208_208329

theorem add_decimals : 4.3 + 3.88 = 8.18 := 
sorry

end add_decimals_l208_208329


namespace percentage_increase_l208_208221

theorem percentage_increase (original_value : ℕ) (percentage_increase : ℚ) :  
  original_value = 1200 → 
  percentage_increase = 0.40 →
  original_value * (1 + percentage_increase) = 1680 :=
by
  intros h1 h2
  sorry

end percentage_increase_l208_208221


namespace cadence_old_company_salary_l208_208309

variable (S : ℝ)

def oldCompanyMonths : ℝ := 36
def newCompanyMonths : ℝ := 41
def newSalaryMultiplier : ℝ := 1.20
def totalEarnings : ℝ := 426000

theorem cadence_old_company_salary :
  (oldCompanyMonths * S) + (newCompanyMonths * newSalaryMultiplier * S) = totalEarnings → 
  S = 5000 :=
by
  sorry

end cadence_old_company_salary_l208_208309


namespace cargo_total_ship_l208_208041

-- Define the initial cargo and the additional cargo loaded
def initial_cargo := 5973
def additional_cargo := 8723

-- Define the total cargo the ship holds after loading additional cargo
def total_cargo := initial_cargo + additional_cargo

-- Statement of the problem
theorem cargo_total_ship (h1 : initial_cargo = 5973) (h2 : additional_cargo = 8723) : 
  total_cargo = 14696 := 
by
  sorry

end cargo_total_ship_l208_208041


namespace find_number_l208_208620

theorem find_number (x : ℝ) :
  (7 * (x + 10) / 5) - 5 = 44 → x = 25 :=
by
  sorry

end find_number_l208_208620


namespace profit_percent_l208_208408

theorem profit_percent (CP SP : ℕ) (h : CP * 5 = SP * 4) : 100 * (SP - CP) = 25 * CP :=
by
  sorry

end profit_percent_l208_208408


namespace evaporation_period_days_l208_208282

theorem evaporation_period_days
    (initial_water : ℝ)
    (daily_evaporation : ℝ)
    (evaporation_percentage : ℝ)
    (total_evaporated_water : ℝ)
    (number_of_days : ℝ) :
    initial_water = 10 ∧
    daily_evaporation = 0.06 ∧
    evaporation_percentage = 0.12 ∧
    total_evaporated_water = initial_water * evaporation_percentage ∧
    number_of_days = total_evaporated_water / daily_evaporation →
    number_of_days = 20 :=
by
  sorry

end evaporation_period_days_l208_208282


namespace multiplication_value_l208_208037

theorem multiplication_value (x : ℝ) (h : (2.25 / 3) * x = 9) : x = 12 :=
by
  sorry

end multiplication_value_l208_208037


namespace rope_length_total_l208_208767

theorem rope_length_total :
  let length1 := 24
  let length2 := 20
  let length3 := 14
  let length4 := 12
  length1 + length2 + length3 + length4 = 70 :=
by
  sorry

end rope_length_total_l208_208767


namespace zero_product_property_l208_208395

theorem zero_product_property {a b : ℝ} (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end zero_product_property_l208_208395


namespace correct_sample_size_l208_208656

variable {StudentScore : Type} {scores : Finset StudentScore} (extract_sample : Finset StudentScore → Finset StudentScore)

noncomputable def is_correct_statement : Prop :=
  ∀ (total_scores : Finset StudentScore) (sample_scores : Finset StudentScore),
  (total_scores.card = 1000) →
  (extract_sample total_scores = sample_scores) →
  (sample_scores.card = 100) →
  sample_scores.card = 100

theorem correct_sample_size (total_scores sample_scores : Finset StudentScore)
  (H_total : total_scores.card = 1000)
  (H_sample : extract_sample total_scores = sample_scores)
  (H_card : sample_scores.card = 100) :
  sample_scores.card = 100 :=
sorry

end correct_sample_size_l208_208656


namespace equivalent_resistance_is_15_l208_208968

-- Definitions based on conditions
def R : ℝ := 5 -- Resistance of each resistor in Ohms
def num_resistors : ℕ := 4

-- The equivalent resistance due to the short-circuit path removing one resistor
def simplified_circuit_resistance : ℝ := (num_resistors - 1) * R

-- The statement to prove
theorem equivalent_resistance_is_15 :
  simplified_circuit_resistance = 15 :=
by
  sorry

end equivalent_resistance_is_15_l208_208968


namespace number_of_digits_in_x_l208_208479

open Real

theorem number_of_digits_in_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (hxy_inequality : x > y)
  (hxy_prod : x * y = 490)
  (hlog_cond : (log x - log 7) * (log y - log 7) = -143/4) :
  ∃ n : ℕ, n = 8 ∧ (10^(n - 1) ≤ x ∧ x < 10^n) :=
by
  sorry

end number_of_digits_in_x_l208_208479


namespace arithmetic_sequence_n_value_l208_208026

theorem arithmetic_sequence_n_value (a_1 d a_nm1 n : ℤ) (h1 : a_1 = -1) (h2 : d = 2) (h3 : a_nm1 = 15) :
    a_nm1 = a_1 + (n - 2) * d → n = 10 :=
by
  intros h
  sorry

end arithmetic_sequence_n_value_l208_208026


namespace space_shuttle_new_orbital_speed_l208_208562

noncomputable def new_orbital_speed (v_1 : ℝ) (delta_v : ℝ) : ℝ :=
  let v_new := v_1 + delta_v
  v_new * 3600

theorem space_shuttle_new_orbital_speed : 
  new_orbital_speed 2 (500 / 1000) = 9000 :=
by 
  sorry

end space_shuttle_new_orbital_speed_l208_208562


namespace solve_for_x_l208_208775

theorem solve_for_x (x : ℝ) (h1 : x^2 - 9 ≠ 0) (h2 : x + 3 ≠ 0) :
  (20 / (x^2 - 9) - 3 / (x + 3) = 2) ↔ (x = (-3 + Real.sqrt 385) / 4 ∨ x = (-3 - Real.sqrt 385) / 4) :=
by
  sorry

end solve_for_x_l208_208775


namespace inequality1_inequality2_l208_208974

noncomputable def f (x : ℝ) := abs (x + 1 / 2) + abs (x - 3 / 2)

theorem inequality1 (x : ℝ) : 
  (f x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 2) := by
sorry

theorem inequality2 (a : ℝ) :
  (∀ x, f x ≥ 1 / 2 * abs (1 - a)) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end inequality1_inequality2_l208_208974


namespace division_quotient_l208_208819

theorem division_quotient (x : ℤ) (y : ℤ) (r : ℝ) (h1 : x > 0) (h2 : y = 96) (h3 : r = 11.52) :
  ∃ q : ℝ, q = (x - r) / y := 
sorry

end division_quotient_l208_208819


namespace geometric_series_sum_frac_l208_208207

open BigOperators

theorem geometric_series_sum_frac (q : ℚ) (a1 : ℚ) (a_list: List ℚ) (h_theta : q = 1 / 2) 
(h_a_list : a_list ⊆ [-4, -3, -2, 0, 1, 23, 4]) : 
  a1 * (1 + q^5) / (1 - q) = 33 / 4 := by
  sorry

end geometric_series_sum_frac_l208_208207


namespace remainder_73_to_73_plus73_div137_l208_208832

theorem remainder_73_to_73_plus73_div137 :
  ((73 ^ 73 + 73) % 137) = 9 := by
  sorry

end remainder_73_to_73_plus73_div137_l208_208832


namespace range_of_root_difference_l208_208080

variable (a b c d : ℝ)
variable (x1 x2 : ℝ)

def g (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem range_of_root_difference
  (h1 : a ≠ 0)
  (h2 : a + b + c = 0)
  (h3 : f a b c 0 * f a b c 1 > 0)
  (hroot1 : f a b c x1 = 0)
  (hroot2 : f a b c x2 = 0)
  : |x1 - x2| ∈ Set.Ico (Real.sqrt 3 / 3) (2 / 3) := sorry

end range_of_root_difference_l208_208080


namespace transform_equation_l208_208561

open Real

theorem transform_equation (m : ℝ) (x : ℝ) (h1 : x^2 + 4 * x = m) (h2 : (x + 2)^2 = 5) : m = 1 := by
  sorry

end transform_equation_l208_208561


namespace smallest_area_of_square_containing_rectangles_l208_208447

noncomputable def smallest_area_square : ℕ :=
  let side1 := 3
  let side2 := 5
  let side3 := 4
  let side4 := 6
  let smallest_side := side1 + side3
  let square_area := smallest_side * smallest_side
  square_area

theorem smallest_area_of_square_containing_rectangles : smallest_area_square = 49 :=
by
  sorry

end smallest_area_of_square_containing_rectangles_l208_208447


namespace max_f_value_l208_208971

open Real

noncomputable def f (x y : ℝ) : ℝ := min x (y / (x^2 + y^2))

theorem max_f_value : ∃ (x₀ y₀ : ℝ), (0 < x₀) ∧ (0 < y₀) ∧ (∀ (x y : ℝ), (0 < x) → (0 < y) → f x y ≤ f x₀ y₀) ∧ f x₀ y₀ = 1 / sqrt 2 :=
by 
  sorry

end max_f_value_l208_208971


namespace product_sum_condition_l208_208640

theorem product_sum_condition (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c > (1/a) + (1/b) + (1/c)) : 
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
sorry

end product_sum_condition_l208_208640


namespace unique_12_tuple_l208_208711

theorem unique_12_tuple : 
  ∃! (x : Fin 12 → ℝ), 
    ((1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + 
    (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 +
    (x 7 - x 8)^2 + (x 8 - x 9)^2 + (x 9 - x 10)^2 + (x 10 - x 11)^2 + 
    (x 11)^2 = 1 / 13) ∧ (x 0 + x 11 = 1 / 2) :=
by
  sorry

end unique_12_tuple_l208_208711


namespace balloons_difference_l208_208517

-- Define the balloons each person brought
def Allan_red := 150
def Allan_blue_total := 75
def Allan_forgotten_blue := 25
def Allan_green := 30

def Jake_red := 100
def Jake_blue := 50
def Jake_green := 45

-- Calculate the actual balloons Allan brought to the park
def Allan_blue := Allan_blue_total - Allan_forgotten_blue
def Allan_total := Allan_red + Allan_blue + Allan_green

-- Calculate the total number of balloons Jake brought
def Jake_total := Jake_red + Jake_blue + Jake_green

-- State the problem: Prove Allan distributed 35 more balloons than Jake
theorem balloons_difference : Allan_total - Jake_total = 35 := 
by
  sorry

end balloons_difference_l208_208517


namespace least_possible_sum_l208_208573

theorem least_possible_sum {c d : ℕ} (hc : c ≥ 2) (hd : d ≥ 2) (h : 3 * c + 6 = 6 * d + 3) : c + d = 5 :=
by
  sorry

end least_possible_sum_l208_208573


namespace median_in_interval_65_69_l208_208360

-- Definitions for student counts in each interval
def count_50_54 := 5
def count_55_59 := 7
def count_60_64 := 22
def count_65_69 := 19
def count_70_74 := 15
def count_75_79 := 10
def count_80_84 := 18
def count_85_89 := 5

-- Total number of students
def total_students := 101

-- Calculation of the position of the median
def median_position := (total_students + 1) / 2

-- Cumulative counts
def cumulative_up_to_59 := count_50_54 + count_55_59
def cumulative_up_to_64 := cumulative_up_to_59 + count_60_64
def cumulative_up_to_69 := cumulative_up_to_64 + count_65_69

-- Proof statement
theorem median_in_interval_65_69 :
  34 < median_position ∧ median_position ≤ cumulative_up_to_69 :=
by
  sorry

end median_in_interval_65_69_l208_208360


namespace swimmers_meeting_times_l208_208225

theorem swimmers_meeting_times (l : ℕ) (vA vB t : ℕ) (T : ℝ) :
  l = 120 →
  vA = 4 →
  vB = 3 →
  t = 15 →
  T = 21 :=
  sorry

end swimmers_meeting_times_l208_208225


namespace speed_of_A_l208_208222
-- Import necessary library

-- Define conditions
def initial_distance : ℝ := 25  -- initial distance between A and B
def speed_B : ℝ := 13  -- speed of B in kmph
def meeting_time : ℝ := 1  -- time duration in hours

-- The speed of A which is to be proven
def speed_A : ℝ := 12

-- The theorem to be proved
theorem speed_of_A (d : ℝ) (vB : ℝ) (t : ℝ) (vA : ℝ) : d = 25 → vB = 13 → t = 1 → 
  d = vA * t + vB * t → vA = 12 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Enforcing the statement to be proved
  have := Eq.symm h4
  simp [speed_A, *] at *
  sorry

end speed_of_A_l208_208222


namespace regular_price_per_can_l208_208981

variable (P : ℝ) -- Regular price per can

-- Condition: The regular price per can is discounted 15 percent when the soda is purchased in 24-can cases
def discountedPricePerCan (P : ℝ) : ℝ :=
  0.85 * P

-- Condition: The price of 72 cans purchased in 24-can cases is $18.36
def priceOf72CansInDollars : ℝ :=
  18.36

-- Predicate describing the condition that the price of 72 cans is 18.36
axiom h : (72 * discountedPricePerCan P) = priceOf72CansInDollars

theorem regular_price_per_can (P : ℝ) (h : (72 * discountedPricePerCan P) = priceOf72CansInDollars) : P = 0.30 :=
by
  sorry

end regular_price_per_can_l208_208981


namespace amy_total_spending_l208_208861

def initial_tickets : ℕ := 33
def cost_per_ticket : ℝ := 1.50
def additional_tickets : ℕ := 21
def total_cost : ℝ := 81.00

theorem amy_total_spending :
  (initial_tickets * cost_per_ticket + additional_tickets * cost_per_ticket) = total_cost := 
sorry

end amy_total_spending_l208_208861


namespace find_starting_number_of_range_l208_208985

theorem find_starting_number_of_range :
  ∃ x, (∀ n, 0 ≤ n ∧ n < 10 → 65 - 5 * n = x + 5 * (9 - n)) ∧ x = 15 := 
by
  sorry

end find_starting_number_of_range_l208_208985


namespace sum_terms_a1_a17_l208_208007

theorem sum_terms_a1_a17 (S : ℕ → ℤ) (a : ℕ → ℤ)
  (hS : ∀ n, S n = n^2 - 2 * n - 1)
  (ha : ∀ n, a n = if n = 1 then S 1 else S n - S (n - 1)) :
  a 1 + a 17 = 29 := by
  sorry

end sum_terms_a1_a17_l208_208007


namespace joe_paint_usage_l208_208772

theorem joe_paint_usage :
  let initial_paint := 360
  let first_week_usage := (1 / 3: ℝ) * initial_paint
  let remaining_after_first_week := initial_paint - first_week_usage
  let second_week_usage := (1 / 5: ℝ) * remaining_after_first_week
  let total_usage := first_week_usage + second_week_usage
  total_usage = 168 :=
by
  sorry

end joe_paint_usage_l208_208772


namespace compute_tensor_operation_l208_208218

def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

theorem compute_tensor_operation :
  tensor (tensor 8 4) 2 = 202 / 9 :=
by
  sorry

end compute_tensor_operation_l208_208218


namespace smallest_number_is_21_5_l208_208202

-- Definitions of the numbers in their respective bases
def num1 := 3 * 4^0 + 3 * 4^1
def num2 := 0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3
def num3 := 2 * 3^0 + 2 * 3^1 + 1 * 3^2
def num4 := 1 * 5^0 + 2 * 5^1

-- Statement asserting that num4 is the smallest number
theorem smallest_number_is_21_5 : num4 < num1 ∧ num4 < num2 ∧ num4 < num3 := by
  sorry

end smallest_number_is_21_5_l208_208202


namespace sin_neg_three_halves_pi_l208_208862

theorem sin_neg_three_halves_pi : Real.sin (-3 * Real.pi / 2) = 1 := sorry

end sin_neg_three_halves_pi_l208_208862


namespace curve_intersection_four_points_l208_208958

theorem curve_intersection_four_points (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 * a^2 ∧ y = a * x^2 - 2 * a) ∧ 
  (∃! (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ), 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
    y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4 ∧
    x1^2 + y1^2 = 4 * a^2 ∧ y1 = a * x1^2 - 2 * a ∧
    x2^2 + y2^2 = 4 * a^2 ∧ y2 = a * x2^2 - 2 * a ∧
    x3^2 + y3^2 = 4 * a^2 ∧ y3 = a * x3^2 - 2 * a ∧
    x4^2 + y4^2 = 4 * a^2 ∧ y4 = a * x4^2 - 2 * a) ↔ 
  a > 1 / 2 :=
by 
  sorry

end curve_intersection_four_points_l208_208958


namespace calculate_share_A_l208_208422

-- Defining the investments
def investment_A : ℕ := 7000
def investment_B : ℕ := 11000
def investment_C : ℕ := 18000
def investment_D : ℕ := 13000
def investment_E : ℕ := 21000
def investment_F : ℕ := 15000
def investment_G : ℕ := 9000

-- Defining B's share
def share_B : ℚ := 3600

-- Function to calculate total investment
def total_investment : ℕ :=
  investment_A + investment_B + investment_C + investment_D + investment_E + investment_F + investment_G

-- Ratio of B's investment to total investment
def ratio_B : ℚ :=
  investment_B / total_investment

-- Calculate total profit using B's share and ratio
def total_profit : ℚ :=
  share_B / ratio_B

-- Ratio of A's investment to total investment
def ratio_A : ℚ :=
  investment_A / total_investment

-- Calculate A's share based on the total profit
def share_A : ℚ :=
  total_profit * ratio_A

-- The theorem to prove the share of A is approximately $2292.34
theorem calculate_share_A : 
  abs (share_A - 2292.34) < 0.01 :=
by
  sorry

end calculate_share_A_l208_208422


namespace find_positive_integer_n_l208_208293

noncomputable def is_largest_prime_divisor (p n : ℕ) : Prop :=
  (∃ k, n = p * k) ∧ ∀ q, Prime q ∧ q ∣ n → q ≤ p

noncomputable def is_least_prime_divisor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n ∧ ∀ q, Prime q ∧ q ∣ n → p ≤ q

theorem find_positive_integer_n :
  ∃ n : ℕ, n > 0 ∧ 
    (∃ p, is_largest_prime_divisor p (n^2 + 3) ∧ is_least_prime_divisor p (n^4 + 6)) ∧
    ∀ m : ℕ, m > 0 ∧ 
      (∃ q, is_largest_prime_divisor q (m^2 + 3) ∧ is_least_prime_divisor q (m^4 + 6)) → m = 3 :=
by sorry

end find_positive_integer_n_l208_208293


namespace total_highlighters_l208_208953

-- Define the number of highlighters of each color
def pink_highlighters : ℕ := 10
def yellow_highlighters : ℕ := 15
def blue_highlighters : ℕ := 8

-- Prove the total number of highlighters
theorem total_highlighters : pink_highlighters + yellow_highlighters + blue_highlighters = 33 :=
by
  sorry

end total_highlighters_l208_208953


namespace first_year_students_sampled_equals_40_l208_208842

-- Defining the conditions
def num_first_year_students := 800
def num_second_year_students := 600
def num_third_year_students := 500
def num_sampled_third_year_students := 25
def total_students := num_first_year_students + num_second_year_students + num_third_year_students

-- Proving the number of first-year students sampled
theorem first_year_students_sampled_equals_40 :
  (num_first_year_students * num_sampled_third_year_students) / num_third_year_students = 40 := by
  sorry

end first_year_students_sampled_equals_40_l208_208842


namespace company_pays_per_month_l208_208092

theorem company_pays_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1.08 * 10^6)
  (h5 : cost_per_box = 0.6) :
  (total_volume / (length * width * height) * cost_per_box) = 360 :=
by
  -- sorry to skip proof
  sorry

end company_pays_per_month_l208_208092


namespace proof_problem_l208_208778

-- Definitions based on the conditions
def x := 70 + 0.11 * 70
def y := x + 0.15 * x
def z := y - 0.2 * y

-- The statement to prove
theorem proof_problem : 3 * z - 2 * x + y = 148.407 :=
by
  sorry

end proof_problem_l208_208778


namespace curve_is_circle_l208_208669

theorem curve_is_circle (s : ℝ) :
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  x^2 + y^2 = 1 :=
by
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  sorry

end curve_is_circle_l208_208669


namespace equilateral_triangle_perimeter_l208_208226

theorem equilateral_triangle_perimeter (p_ADC : ℝ) (h_ratio : ∀ s1 s2 : ℝ, s1 / s2 = 1 / 2) :
  p_ADC = 9 + 3 * Real.sqrt 3 → (3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3) :=
by
  intro h
  have h1 : 3 * (2 * (3 + Real.sqrt 3)) = 18 + 6 * Real.sqrt 3 := sorry
  exact h1

end equilateral_triangle_perimeter_l208_208226


namespace determine_k_for_intersection_l208_208996

theorem determine_k_for_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 3 = 2 * x + 5) ∧ 
  (∀ x₁ x₂ : ℝ, (k * x₁^2 + 2 * x₁ + 3 = 2 * x₁ + 5) ∧ 
                (k * x₂^2 + 2 * x₂ + 3 = 2 * x₂ + 5) → 
              x₁ = x₂) ↔ k = -1/2 :=
by
  sorry

end determine_k_for_intersection_l208_208996


namespace value_of_c_div_b_l208_208623

theorem value_of_c_div_b (a b c : ℕ) (h1 : a = 0) (h2 : a < b) (h3 : b < c) 
  (h4 : b ≠ a + 1) (h5 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end value_of_c_div_b_l208_208623
