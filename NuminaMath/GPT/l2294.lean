import Mathlib

namespace local_minimum_at_minus_one_l2294_229400

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem local_minimum_at_minus_one :
  (∃ δ > 0, ∀ x : ℝ, (x < -1 + δ ∧ x > -1 - δ) → f x ≥ f (-1)) :=
by
  sorry

end local_minimum_at_minus_one_l2294_229400


namespace isabel_weekly_distance_l2294_229467

def circuit_length : ℕ := 365
def morning_runs : ℕ := 7
def afternoon_runs : ℕ := 3
def days_per_week : ℕ := 7

def morning_distance := morning_runs * circuit_length
def afternoon_distance := afternoon_runs * circuit_length
def daily_distance := morning_distance + afternoon_distance
def weekly_distance := daily_distance * days_per_week

theorem isabel_weekly_distance : weekly_distance = 25550 := by
  sorry

end isabel_weekly_distance_l2294_229467


namespace fraction_computation_l2294_229458

theorem fraction_computation :
  (2 + 4 - 8 + 16 + 32 - 64) / (4 + 8 - 16 + 32 + 64 - 128) = 1 / 2 :=
by
  sorry

end fraction_computation_l2294_229458


namespace equilateral_triangle_l2294_229457

theorem equilateral_triangle (a b c : ℝ) (h1 : a + b - c = 2) (h2 : 2 * a * b - c^2 = 4) : a = b ∧ b = c ∧ a = c := 
by
  sorry

end equilateral_triangle_l2294_229457


namespace sin_identity_alpha_l2294_229403

theorem sin_identity_alpha (α : ℝ) (hα : α = Real.pi / 7) : 
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := 
by 
  sorry

end sin_identity_alpha_l2294_229403


namespace expression_value_l2294_229452

-- Define the problem statement
theorem expression_value (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : (x + y) / z = (y + z) / x) (h5 : (y + z) / x = (z + x) / y) :
  ∃ k : ℝ, k = 8 ∨ k = -1 := 
sorry

end expression_value_l2294_229452


namespace f_1982_value_l2294_229425

noncomputable def f (n : ℕ) : ℕ := sorry  -- placeholder for the function definition

axiom f_condition_2 : f 2 = 0
axiom f_condition_3 : f 3 > 0
axiom f_condition_9999 : f 9999 = 3333
axiom f_add_condition (m n : ℕ) : f (m+n) - f m - f n = 0 ∨ f (m+n) - f m - f n = 1

open Nat

theorem f_1982_value : f 1982 = 660 :=
by
  sorry  -- proof goes here

end f_1982_value_l2294_229425


namespace gift_sequences_count_l2294_229423

def num_students : ℕ := 11
def num_meetings : ℕ := 4
def sequences : ℕ := num_students ^ num_meetings

theorem gift_sequences_count : sequences = 14641 := by
  sorry

end gift_sequences_count_l2294_229423


namespace not_chosen_rate_l2294_229490

theorem not_chosen_rate (sum : ℝ) (interest_15_percent : ℝ) (extra_interest : ℝ) : 
  sum = 7000 ∧ interest_15_percent = 2100 ∧ extra_interest = 420 →
  ∃ R : ℝ, (sum * 0.15 * 2 = interest_15_percent) ∧ 
           (interest_15_percent - (sum * R / 100 * 2) = extra_interest) ∧ 
           R = 12 := 
by {
  sorry
}

end not_chosen_rate_l2294_229490


namespace students_on_field_trip_l2294_229441

theorem students_on_field_trip (vans: ℕ) (capacity_per_van: ℕ) (adults: ℕ) 
  (H_vans: vans = 3) 
  (H_capacity_per_van: capacity_per_van = 5) 
  (H_adults: adults = 3) : 
  (vans * capacity_per_van - adults = 12) :=
by
  sorry

end students_on_field_trip_l2294_229441


namespace sqrt_eighteen_simplifies_l2294_229481

open Real

theorem sqrt_eighteen_simplifies :
  sqrt 18 = 3 * sqrt 2 :=
by
  sorry

end sqrt_eighteen_simplifies_l2294_229481


namespace remainder_div_l2294_229433

theorem remainder_div (N : ℤ) (k : ℤ) (h : N = 35 * k + 25) : N % 15 = 10 := by
  sorry

end remainder_div_l2294_229433


namespace sequence_a_n_l2294_229447

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 1 then 1 else (1 : ℚ) / (2 * n - 1)

theorem sequence_a_n (n : ℕ) (hn : n ≥ 1) : 
  (a_n 1 = 1) ∧ 
  (∀ n, a_n n ≠ 0) ∧ 
  (∀ n, n ≥ 2 → a_n n + 2 * a_n n * a_n (n - 1) - a_n (n - 1) = 0) →
  a_n n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_a_n_l2294_229447


namespace sqrt_difference_inequality_l2294_229420

noncomputable def sqrt10 := Real.sqrt 10
noncomputable def sqrt6 := Real.sqrt 6
noncomputable def sqrt7 := Real.sqrt 7
noncomputable def sqrt3 := Real.sqrt 3

theorem sqrt_difference_inequality : sqrt10 - sqrt6 < sqrt7 - sqrt3 :=
by 
  sorry

end sqrt_difference_inequality_l2294_229420


namespace smaller_circle_area_l2294_229482

theorem smaller_circle_area (r R : ℝ) (hR : R = 3 * r)
  (hTangentLines : ∀ (P A B A' B' : ℝ), P = 5 ∧ A = 5 ∧ PA = 5 ∧ A' = 5 ∧ PA' = 5 ∧ AB = 5 ∧ A'B' = 5 ) :
  π * r^2 = 25 / 3 * π := by
  sorry

end smaller_circle_area_l2294_229482


namespace length_of_bridge_l2294_229405

theorem length_of_bridge (train_length : ℕ) (train_speed : ℕ) (cross_time : ℕ) 
  (h1 : train_length = 150) 
  (h2 : train_speed = 45) 
  (h3 : cross_time = 30) : 
  ∃ bridge_length : ℕ, bridge_length = 225 := sorry

end length_of_bridge_l2294_229405


namespace probability_both_in_picture_l2294_229475

-- Define the conditions
def completes_lap (laps_time: ℕ) (time: ℕ) : ℕ := time / laps_time

def position_into_lap (laps_time: ℕ) (time: ℕ) : ℕ := time % laps_time

-- Define the positions of Rachel and Robert
def rachel_position (time: ℕ) : ℚ :=
  let rachel_lap_time := 100
  let laps_completed := completes_lap rachel_lap_time time
  let time_into_lap := position_into_lap rachel_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / rachel_lap_time

def robert_position (time: ℕ) : ℚ :=
  let robert_lap_time := 70
  let laps_completed := completes_lap robert_lap_time time
  let time_into_lap := position_into_lap robert_lap_time time
  (laps_completed : ℚ) + (time_into_lap : ℚ) / robert_lap_time

-- Define the probability that both are in the picture
theorem probability_both_in_picture :
  let rachel_lap_time := 100
  let robert_lap_time := 70
  let start_time := 720
  let end_time := 780
  ∃ (overlap_time: ℚ) (total_time: ℚ),
    overlap_time / total_time = 1 / 16 :=
sorry

end probability_both_in_picture_l2294_229475


namespace find_number_l2294_229439

theorem find_number (x : ℤ) (h : 5 * x - 28 = 232) : x = 52 :=
by
  sorry

end find_number_l2294_229439


namespace dots_not_visible_l2294_229438

theorem dots_not_visible (visible_sum : ℕ) (total_faces_sum : ℕ) (num_dice : ℕ) (total_visible_faces : ℕ)
  (h1 : total_faces_sum = 21)
  (h2 : visible_sum = 22) 
  (h3 : num_dice = 3)
  (h4 : total_visible_faces = 7) :
  (num_dice * total_faces_sum - visible_sum) = 41 :=
sorry

end dots_not_visible_l2294_229438


namespace exists_c_gt_zero_l2294_229466

theorem exists_c_gt_zero (a b : ℝ) (h : a < b) : ∃ c > 0, a < b + c := 
sorry

end exists_c_gt_zero_l2294_229466


namespace unique_solution_l2294_229416

theorem unique_solution (x y z : ℕ) (h_x : x > 1) (h_y : y > 1) (h_z : z > 1) :
  (x + 1)^y - x^z = 1 → x = 2 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end unique_solution_l2294_229416


namespace problem_solution_l2294_229434

theorem problem_solution (x y z w : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : w > 0) 
  (h5 : x^2 + y^2 + z^2 + w^2 = 1) : 
  x^2 * y * z * w + x * y^2 * z * w + x * y * z^2 * w + x * y * z * w^2 ≤ 1 / 8 := 
by
  sorry

end problem_solution_l2294_229434


namespace bus_length_is_200_l2294_229440

def length_of_bus (distance_km distance_secs passing_secs : ℕ) : ℕ :=
  let speed_kms := distance_km / distance_secs
  let speed_ms := speed_kms * 1000
  speed_ms * passing_secs

theorem bus_length_is_200 
  (distance_km : ℕ) (distance_secs : ℕ) (passing_secs : ℕ)
  (h1 : distance_km = 12) (h2 : distance_secs = 300) (h3 : passing_secs = 5) : 
  length_of_bus distance_km distance_secs passing_secs = 200 := 
  by
    sorry

end bus_length_is_200_l2294_229440


namespace time_for_A_l2294_229498

theorem time_for_A (A B C : ℝ) 
  (h1 : 1/B + 1/C = 1/3) 
  (h2 : 1/A + 1/C = 1/2) 
  (h3 : 1/B = 1/30) : 
  A = 5/2 := 
by
  sorry

end time_for_A_l2294_229498


namespace solve_rational_equation_l2294_229459

theorem solve_rational_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 4/5) :
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 20*x - 40)/(5*x - 4) = -5 ↔ x = -3 :=
by 
  sorry

end solve_rational_equation_l2294_229459


namespace largest_n_binomial_l2294_229462

-- Definitions of binomial coefficients and properties
open Nat

-- Binomial coefficient function definition
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem statement: finding the largest integer n satisfying the equation with given conditions
theorem largest_n_binomial (n : ℕ) (h : binom 10 4 + binom 10 5 = binom 11 n) : n = 6 :=
  sorry

end largest_n_binomial_l2294_229462


namespace loan_difference_is_979_l2294_229493

noncomputable def compounded_interest (P r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

noncomputable def loan_difference (P : ℝ) : ℝ :=
  let compounded_7_years := compounded_interest P 0.08 12 7
  let half_payment := compounded_7_years / 2
  let remaining_balance := compounded_interest half_payment 0.08 12 8
  let total_compounded := half_payment + remaining_balance
  let total_simple := simple_interest P 0.10 15
  abs (total_compounded - total_simple)

theorem loan_difference_is_979 : loan_difference 15000 = 979 := sorry

end loan_difference_is_979_l2294_229493


namespace function_passes_through_point_l2294_229477

noncomputable def special_function (a : ℝ) (x : ℝ) := a^(x - 1) + 1

theorem function_passes_through_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  special_function a 1 = 2 :=
by
  -- skip the proof
  sorry

end function_passes_through_point_l2294_229477


namespace matrix_non_invertible_at_36_31_l2294_229407

-- Define the matrix A
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2 * x, 9], ![4 - x, 11]]

-- State the theorem
theorem matrix_non_invertible_at_36_31 :
  ∃ x : ℝ, (A x).det = 0 ∧ x = 36 / 31 :=
by {
  sorry
}

end matrix_non_invertible_at_36_31_l2294_229407


namespace sum_of_solutions_eq_9_l2294_229476

theorem sum_of_solutions_eq_9 (x_1 x_2 : ℝ) (h : x^2 - 9 * x + 20 = 0) :
  x_1 + x_2 = 9 :=
sorry

end sum_of_solutions_eq_9_l2294_229476


namespace rationalize_denominator_sum_l2294_229429

theorem rationalize_denominator_sum :
  let expr := 1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)
  ∃ (A B C D E F G H I : ℤ), 
    I > 0 ∧
    expr * (Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 11) /
    ((Real.sqrt 5 + Real.sqrt 3)^2 - (Real.sqrt 11)^2) = 
        (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + 
         G * Real.sqrt H) / I ∧
    (A + B + C + D + E + F + G + H + I) = 225 :=
by
  sorry

end rationalize_denominator_sum_l2294_229429


namespace bruce_bank_savings_l2294_229460

def aunt_gift : ℕ := 75
def grandfather_gift : ℕ := 150
def total_gift : ℕ := aunt_gift + grandfather_gift
def fraction_saved : ℚ := 1/5
def amount_saved : ℚ := total_gift * fraction_saved

theorem bruce_bank_savings : amount_saved = 45 := by
  sorry

end bruce_bank_savings_l2294_229460


namespace range_of_a_l2294_229444

theorem range_of_a (a : ℝ) :
  (a + 1 > 0 ∧ 3 - 2 * a > 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a < 0 ∧ a + 1 > 3 - 2 * a) ∨ (a + 1 < 0 ∧ 3 - 2 * a > 0)
  → (2 / 3 < a ∧ a < 3 / 2) ∨ (a < -1) :=
by
  sorry

end range_of_a_l2294_229444


namespace relay_scheme_count_l2294_229499

theorem relay_scheme_count
  (num_segments : ℕ)
  (num_torchbearers : ℕ)
  (first_choices : ℕ)
  (last_choices : ℕ) :
  num_segments = 6 ∧
  num_torchbearers = 6 ∧
  first_choices = 3 ∧
  last_choices = 2 →
  ∃ num_schemes : ℕ, num_schemes = 7776 :=
by
  intro h
  obtain ⟨h_segments, h_torchbearers, h_first_choices, h_last_choices⟩ := h
  exact ⟨7776, sorry⟩

end relay_scheme_count_l2294_229499


namespace inequality_proof_l2294_229453

theorem inequality_proof (a b c : ℝ) (hab : a > b) : a * |c| ≥ b * |c| := by
  sorry

end inequality_proof_l2294_229453


namespace passed_boys_count_l2294_229422

theorem passed_boys_count (P F : ℕ) 
  (h1 : P + F = 120) 
  (h2 : 37 * 120 = 39 * P + 15 * F) : 
  P = 110 :=
sorry

end passed_boys_count_l2294_229422


namespace length_first_train_l2294_229495

/-- Let the speeds of two trains be 120 km/hr and 80 km/hr, respectively. 
These trains cross each other in 9 seconds, and the length of the second train is 250.04 meters. 
Prove that the length of the first train is 250 meters. -/
theorem length_first_train
  (FirstTrainSpeed : ℝ := 120)  -- speed of the first train in km/hr
  (SecondTrainSpeed : ℝ := 80)  -- speed of the second train in km/hr
  (TimeToCross : ℝ := 9)        -- time to cross each other in seconds
  (LengthSecondTrain : ℝ := 250.04) -- length of the second train in meters
  : FirstTrainSpeed / 0.36 + SecondTrainSpeed / 0.36 * TimeToCross - LengthSecondTrain = 250 :=
by
  -- omitted proof
  sorry

end length_first_train_l2294_229495


namespace mary_books_end_of_year_l2294_229427

def total_books_end_of_year (books_start : ℕ) (book_club : ℕ) (lent_to_jane : ℕ) 
 (returned_by_alice : ℕ) (bought_5th_month : ℕ) (bought_yard_sales : ℕ) 
 (birthday_daughter : ℕ) (birthday_mother : ℕ) (received_sister : ℕ)
 (buy_one_get_one : ℕ) (donated_charity : ℕ) (borrowed_neighbor : ℕ)
 (sold_used_store : ℕ) : ℕ :=
  books_start + book_club - lent_to_jane + returned_by_alice + bought_5th_month + bought_yard_sales +
  birthday_daughter + birthday_mother + received_sister + buy_one_get_one - donated_charity - borrowed_neighbor - sold_used_store

theorem mary_books_end_of_year : total_books_end_of_year 200 (2 * 12) 10 5 15 8 1 8 6 4 30 5 7 = 219 := by
  sorry

end mary_books_end_of_year_l2294_229427


namespace term_value_in_sequence_l2294_229401

theorem term_value_in_sequence (a : ℕ → ℕ) (n : ℕ) (h : ∀ n, a n = n * (n + 2) / 2) (h_val : a n = 220) : n = 20 :=
  sorry

end term_value_in_sequence_l2294_229401


namespace largest_possible_A_l2294_229492

theorem largest_possible_A : ∃ A B : ℕ, 13 = 4 * A + B ∧ B < A ∧ A = 3 := by
  sorry

end largest_possible_A_l2294_229492


namespace probability_odd_sum_is_correct_l2294_229432

-- Define the set of the first twelve prime numbers.
def first_twelve_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

-- Define the problem statement.
noncomputable def probability_odd_sum : ℚ :=
  let even_prime_count := 1
  let odd_prime_count := 11
  let ways_to_pick_1_even_and_4_odd := (Nat.choose odd_prime_count 4)
  let total_ways := Nat.choose 12 5
  (ways_to_pick_1_even_and_4_odd : ℚ) / total_ways

theorem probability_odd_sum_is_correct :
  probability_odd_sum = 55 / 132 :=
by
  sorry

end probability_odd_sum_is_correct_l2294_229432


namespace distance_focus_directrix_l2294_229484

theorem distance_focus_directrix (p : ℝ) :
  (∀ (x y : ℝ), y^2 = 2 * p * x ∧ x = 6 ∧ dist (x, y) (p/2, 0) = 10) →
  abs (p) = 8 :=
by
  sorry

end distance_focus_directrix_l2294_229484


namespace four_digit_even_numbers_divisible_by_4_l2294_229448

noncomputable def number_of_4_digit_even_numbers_divisible_by_4 : Nat :=
  500

theorem four_digit_even_numbers_divisible_by_4 : 
  (∃ count : Nat, count = number_of_4_digit_even_numbers_divisible_by_4) :=
sorry

end four_digit_even_numbers_divisible_by_4_l2294_229448


namespace simplify_tan_cot_expr_l2294_229449

theorem simplify_tan_cot_expr :
  let tan_45 := 1
  let cot_45 := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 :=
by
  let tan_45 := 1
  let cot_45 := 1
  sorry

end simplify_tan_cot_expr_l2294_229449


namespace proof_cos_2x_cos_2y_l2294_229461

variable {θ x y : ℝ}

-- Conditions
def is_arith_seq (a b c : ℝ) := b = (a + c) / 2
def is_geom_seq (a b c : ℝ) := b^2 = a * c

-- Proving the given statement with the provided conditions
theorem proof_cos_2x_cos_2y (h_arith : is_arith_seq (Real.sin θ) (Real.sin x) (Real.cos θ))
                            (h_geom : is_geom_seq (Real.sin θ) (Real.sin y) (Real.cos θ)) :
  2 * Real.cos (2 * x) = Real.cos (2 * y) :=
sorry

end proof_cos_2x_cos_2y_l2294_229461


namespace simplify_radical_product_l2294_229487

theorem simplify_radical_product : 
  (32^(1/5)) * (8^(1/3)) * (4^(1/2)) = 8 := 
by
  sorry

end simplify_radical_product_l2294_229487


namespace monkeys_and_bananas_l2294_229451

theorem monkeys_and_bananas (m1 m2 t b1 b2 : ℕ) (h1 : m1 = 8) (h2 : t = 8) (h3 : b1 = 8) (h4 : b2 = 3) : m2 = 3 :=
by
  -- Here we will include the formal proof steps
  sorry

end monkeys_and_bananas_l2294_229451


namespace bags_already_made_l2294_229468

def bags_per_batch : ℕ := 10
def customer_order : ℕ := 60
def days_to_fulfill : ℕ := 4
def batches_per_day : ℕ := 1

theorem bags_already_made :
  (customer_order - (days_to_fulfill * batches_per_day * bags_per_batch)) = 20 :=
by
  sorry

end bags_already_made_l2294_229468


namespace world_grain_demand_l2294_229406

theorem world_grain_demand (S D : ℝ) (h1 : S = 1800000) (h2 : S = 0.75 * D) : D = 2400000 := by
  sorry

end world_grain_demand_l2294_229406


namespace problem_conditions_l2294_229479

noncomputable def f (x : ℝ) := x^2 - 2 * x * Real.log x
noncomputable def g (x : ℝ) := Real.exp x - (Real.exp 2 * x^2) / 4

theorem problem_conditions :
  (∀ x > 0, deriv f x > 0) ∧ 
  (∃! x, g x = 0) ∧ 
  (∃ x, f x = g x) :=
by
  sorry

end problem_conditions_l2294_229479


namespace system_solution_l2294_229463

theorem system_solution (x y z : ℝ) 
  (h1 : x - y ≥ z)
  (h2 : x^2 + 4 * y^2 + 5 = 4 * z) :
  (x = 2 ∧ y = -0.5 ∧ z = 2.5) :=
sorry

end system_solution_l2294_229463


namespace arithmetic_sequence_property_l2294_229426

variable {a : ℕ → ℝ} -- Let a be an arithmetic sequence
variable {S : ℕ → ℝ} -- Let S be the sum of the first n terms of the sequence

-- Conditions
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a 1 + (n - 1) * (a 2 - a 1) / 2)
axiom a_5 : a 5 = 3
axiom S_13 : S 13 = 91

-- Question to prove
theorem arithmetic_sequence_property : a 1 + a 11 = 10 :=
by
  sorry

end arithmetic_sequence_property_l2294_229426


namespace rectangle_ratio_l2294_229414

theorem rectangle_ratio 
  (s : ℝ) -- side length of the inner square
  (x y : ℝ) -- longer side and shorter side of the rectangle
  (h_inner_area : s^2 = (inner_square_area : ℝ))
  (h_outer_area : 9 * inner_square_area = outer_square_area)
  (h_outer_side_eq : (s + 2 * y)^2 = outer_square_area)
  (h_longer_side_eq : x + y = 3 * s) :
  x / y = 2 :=
by sorry

end rectangle_ratio_l2294_229414


namespace common_difference_l2294_229413

theorem common_difference (a : ℕ → ℤ) (d : ℤ) 
    (h1 : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
    (h2 : a 1 + a 3 + a 5 = 15)
    (h3 : a 4 = 3) : 
    d = -2 := 
sorry

end common_difference_l2294_229413


namespace shenille_points_l2294_229478

def shenille_total_points (x y : ℕ) : ℝ :=
  0.6 * x + 0.6 * y

theorem shenille_points (x y : ℕ) (h : x + y = 30) : 
  shenille_total_points x y = 18 := by
  sorry

end shenille_points_l2294_229478


namespace find_a_l2294_229485

noncomputable def ab (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a {a : ℝ} : ab a 6 = -3 → a = 23 :=
by
  sorry

end find_a_l2294_229485


namespace dana_pencils_more_than_jayden_l2294_229454

theorem dana_pencils_more_than_jayden :
  ∀ (Jayden_has_pencils : ℕ) (Marcus_has_pencils : ℕ) (Dana_has_pencils : ℕ),
    Jayden_has_pencils = 20 →
    Marcus_has_pencils = Jayden_has_pencils / 2 →
    Dana_has_pencils = Marcus_has_pencils + 25 →
    Dana_has_pencils - Jayden_has_pencils = 15 :=
by
  intros Jayden_has_pencils Marcus_has_pencils Dana_has_pencils
  intro h1
  intro h2
  intro h3
  sorry

end dana_pencils_more_than_jayden_l2294_229454


namespace complement_union_l2294_229483

open Set

variable (U : Set ℕ := {0, 1, 2, 3, 4}) (A : Set ℕ := {1, 2, 3}) (B : Set ℕ := {2, 4})

theorem complement_union (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) : 
  (U \ A ∪ B) = {0, 2, 4} :=
by
  sorry

end complement_union_l2294_229483


namespace find_r_from_tan_cosine_tangent_l2294_229415

theorem find_r_from_tan_cosine_tangent 
  (θ : ℝ) 
  (r : ℝ) 
  (htan : Real.tan θ = -7 / 24) 
  (hquadrant : π / 2 < θ ∧ θ < π) 
  (hr : 100 * Real.cos θ = r) : 
  r = -96 := 
sorry

end find_r_from_tan_cosine_tangent_l2294_229415


namespace Sam_balloon_count_l2294_229480

theorem Sam_balloon_count:
  ∀ (F M S : ℕ), F = 5 → M = 7 → (F + M + S = 18) → S = 6 :=
by
  intros F M S hF hM hTotal
  rw [hF, hM] at hTotal
  linarith

end Sam_balloon_count_l2294_229480


namespace curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l2294_229486

noncomputable def curve_C (a x y : ℝ) := a * x ^ 2 + a * y ^ 2 - 2 * x - 2 * y = 0

theorem curve_C_straight_line (a : ℝ) : a = 0 → ∃ x y : ℝ, curve_C a x y :=
by
  intro ha
  use (-1), 1
  rw [curve_C, ha]
  simp

theorem curve_C_not_tangent (a : ℝ) : a = 1 → ¬ ∀ x y, 3 * x + y = 0 → curve_C a x y :=
by
  sorry

theorem curve_C_fixed_point (x y a : ℝ) : curve_C a 0 0 :=
by
  rw [curve_C]
  simp

theorem curve_C_intersect (a : ℝ) : a = 1 → ∃ x y : ℝ, (x + 2 * y = 0) ∧ curve_C a x y :=
by
  sorry

end curve_C_straight_line_curve_C_not_tangent_curve_C_fixed_point_curve_C_intersect_l2294_229486


namespace jason_initial_cards_l2294_229418

theorem jason_initial_cards (cards_given_away cards_left : ℕ) (h1 : cards_given_away = 9) (h2 : cards_left = 4) :
  cards_given_away + cards_left = 13 :=
sorry

end jason_initial_cards_l2294_229418


namespace find_a_l2294_229496

/-- 
Given sets A and B defined by specific quadratic equations, 
if A ∪ B = A, then a ∈ (-∞, 0).
-/
theorem find_a :
  ∀ (a : ℝ),
    (A = {x : ℝ | x^2 - 3 * x + 2 = 0}) →
    (B = {x : ℝ | x^2 - 2 * a * x + a^2 - a = 0}) →
    (A ∪ B = A) →
    a < 0 :=
by
  sorry

end find_a_l2294_229496


namespace equivalent_single_discount_l2294_229442

theorem equivalent_single_discount (x : ℝ) : 
  (1 - 0.15) * (1 - 0.20) * (1 - 0.10) = 1 - 0.388 :=
by
  sorry

end equivalent_single_discount_l2294_229442


namespace find_y_l2294_229421

theorem find_y (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 :=
sorry

end find_y_l2294_229421


namespace sequence_general_formula_l2294_229469

theorem sequence_general_formula (a : ℕ → ℚ) (h₁ : a 1 = 2 / 3)
  (h₂ : ∀ n : ℕ, a (n + 1) = a n + a n * a (n + 1)) : 
  ∀ n : ℕ, a n = 2 / (5 - 2 * n) :=
by 
  sorry

end sequence_general_formula_l2294_229469


namespace radius_of_circle_with_square_and_chord_l2294_229443

theorem radius_of_circle_with_square_and_chord :
  ∃ (r : ℝ), 
    (∀ (chord_length square_side_length : ℝ), chord_length = 6 ∧ square_side_length = 2 → 
    (r = Real.sqrt 10)) :=
by
  sorry

end radius_of_circle_with_square_and_chord_l2294_229443


namespace surface_area_inequality_l2294_229489

theorem surface_area_inequality
  (a b c d e f S : ℝ) :
  S ≤ (Real.sqrt 3 / 6) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2) :=
sorry

end surface_area_inequality_l2294_229489


namespace total_land_l2294_229419

variable (land_house : ℕ) (land_expansion : ℕ) (land_cattle : ℕ) (land_crop : ℕ)

theorem total_land (h1 : land_house = 25) 
                   (h2 : land_expansion = 15) 
                   (h3 : land_cattle = 40) 
                   (h4 : land_crop = 70) : 
  land_house + land_expansion + land_cattle + land_crop = 150 := 
by 
  sorry

end total_land_l2294_229419


namespace find_number_of_students_l2294_229446

theorem find_number_of_students
  (n : ℕ)
  (average_marks : ℕ → ℚ)
  (wrong_mark_corrected : ℕ → ℕ → ℚ)
  (correct_avg_marks_pred : ℕ → ℚ → Prop)
  (h1 : average_marks n = 60)
  (h2 : wrong_mark_corrected 90 15 = 75)
  (h3 : correct_avg_marks_pred n 57.5) :
  n = 30 :=
sorry

end find_number_of_students_l2294_229446


namespace find_ellipse_eq_product_of_tangent_slopes_l2294_229497

variables {a b : ℝ} {x y x0 y0 : ℝ}

-- Given conditions
def ellipse (a b : ℝ) := a > 0 ∧ b > 0 ∧ a > b ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → y = 1 ∧ y = 3 / 2)

def eccentricity (a b : ℝ) := b = (1 / 2) * a

def passes_through (x y : ℝ) := x = 1 ∧ y = 3 / 2

-- Part 1: Prove the equation of the ellipse
theorem find_ellipse_eq (a b : ℝ) (h_ellipse : ellipse a b) (h_eccentricity : eccentricity a b) (h_point : passes_through 1 (3/2)) :
    (x^2) / 4 + (y^2) / 3 = 1 :=
sorry

-- Circle equation definition
def circle (x y : ℝ) := x^2 + y^2 = 7

-- Part 2: Prove the product of the slopes of the tangent lines is constant
theorem product_of_tangent_slopes (P : ℝ × ℝ) (h_circle : circle P.1 P.2) : 
    ∀ k1 k2 : ℝ, (4 - P.1^2) * k1^2 + 6 * P.1 * P.2 * k1 + 3 - P.2^2 = 0 → 
    (4 - P.1^2) * k2^2 + 6 * P.1 * P.2 * k2 + 3 - P.2^2 = 0 → k1 * k2 = -1 :=
sorry

end find_ellipse_eq_product_of_tangent_slopes_l2294_229497


namespace simplify_expression_l2294_229494

noncomputable def proof_problem (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : Prop :=
  (1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a)) = 1

theorem simplify_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) :
  proof_problem a b c h h_abc :=
by sorry

end simplify_expression_l2294_229494


namespace decagon_diagonals_l2294_229464

-- Number of diagonals calculation definition
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Proving the number of diagonals in a decagon
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end decagon_diagonals_l2294_229464


namespace max_sum_of_distinct_integers_l2294_229431

theorem max_sum_of_distinct_integers (A B C : ℕ) (hABC_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (hProduct : A * B * C = 1638) :
  A + B + C ≤ 126 :=
sorry

end max_sum_of_distinct_integers_l2294_229431


namespace vector_expression_result_l2294_229491

structure Vector2 :=
(x : ℝ)
(y : ℝ)

def vector_dot_product (v1 v2 : Vector2) : ℝ :=
  v1.x * v1.y + v2.x * v2.y

def vector_scalar_mul (c : ℝ) (v : Vector2) : Vector2 :=
  { x := c * v.x, y := c * v.y }

def vector_sub (v1 v2 : Vector2) : Vector2 :=
  { x := v1.x - v2.x, y := v1.y - v2.y }

noncomputable def a : Vector2 := { x := 2, y := -1 }
noncomputable def b : Vector2 := { x := 3, y := -2 }

theorem vector_expression_result :
  vector_dot_product
    (vector_sub (vector_scalar_mul 3 a) b)
    (vector_sub a (vector_scalar_mul 2 b)) = -15 := by
  sorry

end vector_expression_result_l2294_229491


namespace television_screen_horizontal_length_l2294_229471

theorem television_screen_horizontal_length :
  ∀ (d : ℝ) (r_l : ℝ) (r_h : ℝ), r_l / r_h = 4 / 3 → d = 27 → 
  let h := (3 / 5) * d
  let l := (4 / 5) * d
  l = 21.6 := by
  sorry

end television_screen_horizontal_length_l2294_229471


namespace sulfuric_acid_moles_used_l2294_229428

-- Definitions and conditions
def iron_moles : ℕ := 2
def iron_ii_sulfate_moles_produced : ℕ := 2
def sulfuric_acid_to_iron_ratio : ℕ := 1

-- Proof statement
theorem sulfuric_acid_moles_used {H2SO4_moles : ℕ} 
  (h_fe_reacts : H2SO4_moles = iron_moles * sulfuric_acid_to_iron_ratio) 
  (h_fe produces: iron_ii_sulfate_moles_produced = iron_moles) : H2SO4_moles = 2 :=
by
  sorry

end sulfuric_acid_moles_used_l2294_229428


namespace negation_p_l2294_229474

def nonneg_reals := { x : ℝ // 0 ≤ x }

def p := ∀ x : nonneg_reals, Real.exp x.1 ≥ 1

theorem negation_p :
  ¬ p ↔ ∃ x : nonneg_reals, Real.exp x.1 < 1 :=
by
  sorry

end negation_p_l2294_229474


namespace min_value_l2294_229404

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (1 / x + 4 / y) ≥ 9) :=
by
  sorry

end min_value_l2294_229404


namespace golden_section_length_l2294_229450

theorem golden_section_length (MN : ℝ) (MP NP : ℝ) (hMN : MN = 1) (hP : MP + NP = MN) (hgolden : MN / MP = MP / NP) (hMP_gt_NP : MP > NP) : MP = (Real.sqrt 5 - 1) / 2 :=
by sorry

end golden_section_length_l2294_229450


namespace ten_term_sequence_l2294_229402
open Real

theorem ten_term_sequence (a b : ℝ) 
    (h₁ : a + b = 1)
    (h₂ : a^2 + b^2 = 3)
    (h₃ : a^3 + b^3 = 4)
    (h₄ : a^4 + b^4 = 7)
    (h₅ : a^5 + b^5 = 11) :
    a^10 + b^10 = 123 :=
  sorry

end ten_term_sequence_l2294_229402


namespace segment_equality_l2294_229410

variables {Point : Type} [AddGroup Point]

-- Define the points A, B, C, D, E, F
variables (A B C D E F : Point)

-- Given conditions
variables (AC CE BD DF AD CF : Point)
variable (h1 : AC = CE)
variable (h2 : BD = DF)
variable (h3 : AD = CF)

-- Theorem statement
theorem segment_equality (h1 : A - C = C - E)
                         (h2 : B - D = D - F)
                         (h3 : A - D = C - F) :
  (C - D) = (A - B) ∧ (C - D) = (E - F) :=
by
  sorry

end segment_equality_l2294_229410


namespace rahul_matches_l2294_229424

variable (m : ℕ)

/-- Rahul's current batting average is 51, and if he scores 78 runs in today's match,
    his new batting average will become 54. Prove that the number of matches he had played
    in this season before today's match is 8. -/
theorem rahul_matches (h1 : (51 * m) / m = 51)
                      (h2 : (51 * m + 78) / (m + 1) = 54) : m = 8 := by
  sorry

end rahul_matches_l2294_229424


namespace xy_sufficient_but_not_necessary_l2294_229445

theorem xy_sufficient_but_not_necessary (x y : ℝ) : (x > 0 ∧ y > 0) → (xy > 0) ∧ ¬(xy > 0 → (x > 0 ∧ y > 0)) :=
by
  intros h
  sorry

end xy_sufficient_but_not_necessary_l2294_229445


namespace no_positive_integer_solutions_l2294_229473

theorem no_positive_integer_solutions (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * (x + 1) = p^(2 * n) * y * (y + 1) :=
by
  sorry

end no_positive_integer_solutions_l2294_229473


namespace min_blocks_to_remove_l2294_229409

theorem min_blocks_to_remove (n : ℕ) (h : n = 59) : 
  ∃ (k : ℕ), k = 32 ∧ (∃ m, n = m^3 + k ∧ m^3 ≤ n) :=
by {
  sorry
}

end min_blocks_to_remove_l2294_229409


namespace expression_evaluation_l2294_229430

theorem expression_evaluation : 4 * (9 - 6) / 2 - 3 = 3 := 
by
  sorry

end expression_evaluation_l2294_229430


namespace avg_salary_increase_l2294_229465

def initial_avg_salary : ℝ := 1700
def num_employees : ℕ := 20
def manager_salary : ℝ := 3800

theorem avg_salary_increase :
  ((num_employees * initial_avg_salary + manager_salary) / (num_employees + 1)) - initial_avg_salary = 100 :=
by
  sorry

end avg_salary_increase_l2294_229465


namespace find_number_of_girls_l2294_229417

-- Definitions for the number of candidates
variables (B G : ℕ)
variable (total_candidates : B + G = 2000)

-- Definitions for the percentages of passed candidates
variable (pass_rate_boys : ℝ := 0.34)
variable (pass_rate_girls : ℝ := 0.32)
variable (pass_rate_total : ℝ := 0.331)

-- Hypotheses based on the conditions
variables (P_B P_G : ℝ)
variable (pass_boys : P_B = pass_rate_boys * B)
variable (pass_girls : P_G = pass_rate_girls * G)
variable (pass_total_eq : P_B + P_G = pass_rate_total * 2000)

-- Goal: Prove that the number of girls (G) is 1800
theorem find_number_of_girls (B G : ℕ)
  (total_candidates : B + G = 2000)
  (pass_rate_boys : ℝ := 0.34)
  (pass_rate_girls : ℝ := 0.32)
  (pass_rate_total : ℝ := 0.331)
  (P_B P_G : ℝ)
  (pass_boys : P_B = pass_rate_boys * (B : ℝ))
  (pass_girls : P_G = pass_rate_girls * (G : ℝ))
  (pass_total_eq : P_B + P_G = pass_rate_total * 2000) : G = 1800 :=
sorry

end find_number_of_girls_l2294_229417


namespace age_ratio_is_4_over_3_l2294_229456

-- Define variables for ages
variable (R D : ℕ)

-- Conditions
axiom key_condition_R : R + 10 = 26
axiom key_condition_D : D = 12

-- Theorem statement: The ratio of Rahul's age to Deepak's age is 4/3
theorem age_ratio_is_4_over_3 (hR : R + 10 = 26) (hD : D = 12) : R / D = 4 / 3 :=
sorry

end age_ratio_is_4_over_3_l2294_229456


namespace andrei_cannot_ensure_victory_l2294_229411

theorem andrei_cannot_ensure_victory :
  ∀ (juice_andrew : ℝ) (juice_masha : ℝ),
    juice_andrew = 24 * 1000 ∧
    juice_masha = 24 * 1000 ∧
    ∀ (andrew_mug : ℝ) (masha_mug1 : ℝ) (masha_mug2 : ℝ),
      andrew_mug = 500 ∧
      masha_mug1 = 240 ∧
      masha_mug2 = 240 ∧
      (¬ (∃ (turns_andrew turns_masha : ℕ), 
        turns_andrew * andrew_mug > 48 * 1000 / 2 ∨
        turns_masha * (masha_mug1 + masha_mug2) > 48 * 1000 / 2)) := sorry

end andrei_cannot_ensure_victory_l2294_229411


namespace standard_colony_condition_l2294_229470

noncomputable def StandardBacterialColony : Prop := sorry

theorem standard_colony_condition (visible_mass_of_microorganisms : Prop) 
                                   (single_mother_cell : Prop) 
                                   (solid_culture_medium : Prop) 
                                   (not_multiple_types : Prop) 
                                   : StandardBacterialColony :=
sorry

end standard_colony_condition_l2294_229470


namespace max_sum_x_y_l2294_229488

theorem max_sum_x_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x^3 + y^3 + (x + y)^3 + 36 * x * y = 3456) : x + y ≤ 12 :=
sorry

end max_sum_x_y_l2294_229488


namespace eleven_place_unamed_racer_l2294_229435

theorem eleven_place_unamed_racer
  (Rand Hikmet Jack Marta David Todd : ℕ)
  (positions : Fin 15)
  (C_1 : Rand = Hikmet + 6)
  (C_2 : Marta = Jack + 1)
  (C_3 : David = Hikmet + 3)
  (C_4 : Jack = Todd + 3)
  (C_5 : Todd = Rand + 1)
  (C_6 : Marta = 8) :
  ∃ (x : Fin 15), (x ≠ Rand) ∧ (x ≠ Hikmet) ∧ (x ≠ Jack) ∧ (x ≠ Marta) ∧ (x ≠ David) ∧ (x ≠ Todd) ∧ x = 11 := 
sorry

end eleven_place_unamed_racer_l2294_229435


namespace defeated_candidate_percentage_l2294_229436

noncomputable def percentage_defeated_candidate (total_votes diff_votes invalid_votes : ℕ) : ℕ :=
  let valid_votes := total_votes - invalid_votes
  let P := 100 * (valid_votes - diff_votes) / (2 * valid_votes)
  P

theorem defeated_candidate_percentage (total_votes : ℕ) (diff_votes : ℕ) (invalid_votes : ℕ) :
  total_votes = 12600 ∧ diff_votes = 5000 ∧ invalid_votes = 100 → percentage_defeated_candidate total_votes diff_votes invalid_votes = 30 :=
by
  intros
  sorry

end defeated_candidate_percentage_l2294_229436


namespace quadratic_positive_imp_ineq_l2294_229472

theorem quadratic_positive_imp_ineq (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c > 0) → b^2 - 4 * c ≤ 0 :=
by 
  sorry

end quadratic_positive_imp_ineq_l2294_229472


namespace train_passes_jogger_in_40_seconds_l2294_229455

variable (speed_jogger_kmh : ℕ)
variable (speed_train_kmh : ℕ)
variable (head_start : ℕ)
variable (train_length : ℕ)

noncomputable def time_to_pass_jogger (speed_jogger_kmh speed_train_kmh head_start train_length : ℕ) : ℕ :=
  let speed_jogger_ms := (speed_jogger_kmh * 1000) / 3600
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  let relative_speed := speed_train_ms - speed_jogger_ms
  let total_distance := head_start + train_length
  total_distance / relative_speed

theorem train_passes_jogger_in_40_seconds : time_to_pass_jogger 9 45 280 120 = 40 := by
  sorry

end train_passes_jogger_in_40_seconds_l2294_229455


namespace quadratic_root_relation_l2294_229412

theorem quadratic_root_relation (m n p q : ℝ) (s₁ s₂ : ℝ) 
  (h1 : s₁ + s₂ = -p) 
  (h2 : s₁ * s₂ = q) 
  (h3 : 3 * s₁ + 3 * s₂ = -m) 
  (h4 : 9 * s₁ * s₂ = n) 
  (h_m : m ≠ 0) 
  (h_n : n ≠ 0) 
  (h_p : p ≠ 0) 
  (h_q : q ≠ 0) :
  n = 9 * q :=
by
  sorry

end quadratic_root_relation_l2294_229412


namespace xiao_hua_correct_answers_l2294_229437

theorem xiao_hua_correct_answers :
  ∃ (correct_answers wrong_answers : ℕ), 
    correct_answers + wrong_answers = 15 ∧
    8 * correct_answers - 4 * wrong_answers = 72 ∧
    correct_answers = 11 :=
by
  sorry

end xiao_hua_correct_answers_l2294_229437


namespace count_pairs_l2294_229408

theorem count_pairs (a b : ℤ) (ha : 1 ≤ a ∧ a ≤ 42) (hb : 1 ≤ b ∧ b ≤ 42) (h : a^9 % 43 = b^7 % 43) : (∃ (n : ℕ), n = 42) :=
  sorry

end count_pairs_l2294_229408
