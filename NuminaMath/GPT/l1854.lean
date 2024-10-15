import Mathlib

namespace NUMINAMATH_GPT_pirate_islands_probability_l1854_185485

open Finset

/-- There are 7 islands.
There is a 1/5 chance of finding an island with treasure only (no traps).
There is a 1/10 chance of finding an island with treasure and traps.
There is a 1/10 chance of finding an island with traps only (no treasure).
There is a 3/5 chance of finding an island with neither treasure nor traps.
We want to prove that the probability of finding exactly 3 islands
with treasure only and the remaining 4 islands with neither treasure
nor traps is 81/2225. -/
theorem pirate_islands_probability :
  (Nat.choose 7 3 : ℚ) * ((1/5)^3) * ((3/5)^4) = 81 / 2225 :=
by
  /- Here goes the proof -/
  sorry

end NUMINAMATH_GPT_pirate_islands_probability_l1854_185485


namespace NUMINAMATH_GPT_distinct_real_c_f_ff_ff_five_l1854_185479

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem distinct_real_c_f_ff_ff_five : 
  (∀ c : ℝ, f (f (f (f c))) = 5 → False) :=
by
  sorry

end NUMINAMATH_GPT_distinct_real_c_f_ff_ff_five_l1854_185479


namespace NUMINAMATH_GPT_c_share_is_160_l1854_185458

theorem c_share_is_160 (a b c : ℕ) (total : ℕ) (h1 : 4 * a = 5 * b) (h2 : 5 * b = 10 * c) (h_total : a + b + c = 880) : c = 160 :=
by
  sorry

end NUMINAMATH_GPT_c_share_is_160_l1854_185458


namespace NUMINAMATH_GPT_smallest_positive_omega_l1854_185441

theorem smallest_positive_omega (f g : ℝ → ℝ) (ω : ℝ) 
  (hf : ∀ x, f x = Real.cos (ω * x)) 
  (hg : ∀ x, g x = Real.sin (ω * x - π / 4)) 
  (heq : ∀ x, f (x - π / 2) = g x) :
  ω = 3 / 2 :=
sorry

end NUMINAMATH_GPT_smallest_positive_omega_l1854_185441


namespace NUMINAMATH_GPT_truck_weight_l1854_185478

theorem truck_weight (T R : ℝ) (h1 : T + R = 7000) (h2 : R = 0.5 * T - 200) : T = 4800 :=
by sorry

end NUMINAMATH_GPT_truck_weight_l1854_185478


namespace NUMINAMATH_GPT_coronavirus_case_ratio_l1854_185490

theorem coronavirus_case_ratio (n_first_wave_cases : ℕ) (total_second_wave_cases : ℕ) (n_days : ℕ) 
  (h1 : n_first_wave_cases = 300) (h2 : total_second_wave_cases = 21000) (h3 : n_days = 14) :
  (total_second_wave_cases / n_days) / n_first_wave_cases = 5 :=
by sorry

end NUMINAMATH_GPT_coronavirus_case_ratio_l1854_185490


namespace NUMINAMATH_GPT_area_increase_l1854_185418

theorem area_increase (r₁ r₂: ℝ) (A₁ A₂: ℝ) (side1 side2: ℝ) 
  (h1: side1 = 8) (h2: side2 = 12) (h3: r₁ = side2 / 2) (h4: r₂ = side1 / 2)
  (h5: A₁ = 2 * (1/2 * Real.pi * r₁ ^ 2) + 2 * (1/2 * Real.pi * r₂ ^ 2))
  (h6: A₂ = 4 * (Real.pi * r₂ ^ 2))
  (h7: A₁ = 52 * Real.pi) (h8: A₂ = 64 * Real.pi) :
  ((A₁ + A₂) - A₁) / A₁ * 100 = 123 :=
by
  sorry

end NUMINAMATH_GPT_area_increase_l1854_185418


namespace NUMINAMATH_GPT_solve_real_numbers_l1854_185439

theorem solve_real_numbers (x y : ℝ) :
  (x = 3 * x^2 * y - y^3) ∧ (y = x^3 - 3 * x * y^2) ↔
  ((x = 0 ∧ y = 0) ∨ 
   (x = (Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = (Real.sqrt (2 + Real.sqrt 2)) / 2) ∨
   (x = -(Real.sqrt (2 + Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 - Real.sqrt 2)) / 2) ∨
   (x = (Real.sqrt (2 - Real.sqrt 2)) / 2 ∧ y = -(Real.sqrt (2 + Real.sqrt 2)) / 2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_real_numbers_l1854_185439


namespace NUMINAMATH_GPT_find_numbers_l1854_185412

theorem find_numbers (x y z u n : ℤ)
  (h1 : x + y + z + u = 36)
  (h2 : x + n = y - n)
  (h3 : x + n = z * n)
  (h4 : x + n = u / n) :
  n = 1 ∧ x = 8 ∧ y = 10 ∧ z = 9 ∧ u = 9 :=
sorry

end NUMINAMATH_GPT_find_numbers_l1854_185412


namespace NUMINAMATH_GPT_parabolas_intersect_diff_l1854_185493

theorem parabolas_intersect_diff (a b c d : ℝ) (h1 : c ≥ a)
  (h2 : b = 3 * a^2 - 6 * a + 3)
  (h3 : d = 3 * c^2 - 6 * c + 3)
  (h4 : b = -2 * a^2 - 4 * a + 6)
  (h5 : d = -2 * c^2 - 4 * c + 6) :
  c - a = 1.6 :=
sorry

end NUMINAMATH_GPT_parabolas_intersect_diff_l1854_185493


namespace NUMINAMATH_GPT_find_angle_ACB_l1854_185449

-- Definitions corresponding to the conditions
def angleABD : ℝ := 145
def angleBAC : ℝ := 105
def supplementary (a b : ℝ) : Prop := a + b = 180
def triangleAngleSum (a b c : ℝ) : Prop := a + b + c = 180

theorem find_angle_ACB :
  ∃ (angleACB : ℝ), 
    supplementary angleABD angleABC ∧
    triangleAngleSum angleBAC angleABC angleACB ∧
    angleACB = 40 := 
sorry

end NUMINAMATH_GPT_find_angle_ACB_l1854_185449


namespace NUMINAMATH_GPT_age_of_youngest_child_l1854_185494

theorem age_of_youngest_child (x : ℕ) :
  (x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 55) → x = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_age_of_youngest_child_l1854_185494


namespace NUMINAMATH_GPT_third_consecutive_odd_integer_l1854_185468

theorem third_consecutive_odd_integer (x : ℤ) (h : 3 * x = 2 * (x + 4) + 3) : x + 4 = 15 :=
sorry

end NUMINAMATH_GPT_third_consecutive_odd_integer_l1854_185468


namespace NUMINAMATH_GPT_range_of_m_l1854_185459

theorem range_of_m (m : ℝ) (x y : ℝ)
  (h1 : x + y - 3 * m = 0)
  (h2 : 2 * x - y + 2 * m - 1 = 0)
  (h3 : x > 0)
  (h4 : y < 0) : 
  -1 < m ∧ m < 1/8 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1854_185459


namespace NUMINAMATH_GPT_polynomial_solution_l1854_185472

noncomputable def polynomial_form (P : ℝ → ℝ) : Prop :=
∀ (x y z : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ (2 * x * y * z = x + y + z) →
(P x / (y * z) + P y / (z * x) + P z / (x * y) = P (x - y) + P (y - z) + P (z - x))

theorem polynomial_solution (P : ℝ → ℝ) : polynomial_form P → ∃ c : ℝ, ∀ x : ℝ, P x = c * (x ^ 2 + 3) := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_solution_l1854_185472


namespace NUMINAMATH_GPT_initial_nickels_l1854_185489

theorem initial_nickels (N : ℕ) (h1 : N + 9 + 2 = 18) : N = 7 :=
by sorry

end NUMINAMATH_GPT_initial_nickels_l1854_185489


namespace NUMINAMATH_GPT_scientific_calculator_ratio_l1854_185476

theorem scientific_calculator_ratio (total : ℕ) (basic_cost : ℕ) (change : ℕ) (sci_ratio : ℕ → ℕ) (graph_ratio : ℕ → ℕ) : 
  total = 100 →
  basic_cost = 8 →
  sci_ratio basic_cost = 8 * x →
  graph_ratio (sci_ratio basic_cost) = 3 * sci_ratio basic_cost →
  change = 28 →
  8 + (8 * x) + (24 * x) = 72 →
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_scientific_calculator_ratio_l1854_185476


namespace NUMINAMATH_GPT_polygon_sides_eq_eight_l1854_185442

theorem polygon_sides_eq_eight (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polygon_sides_eq_eight_l1854_185442


namespace NUMINAMATH_GPT_unique_positive_integer_k_for_rational_solutions_l1854_185425

theorem unique_positive_integer_k_for_rational_solutions :
  ∃ (k : ℕ), (k > 0) ∧ (∀ (x : ℤ), x * x = 256 - 4 * k * k → x = 8) ∧ (k = 7) :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_integer_k_for_rational_solutions_l1854_185425


namespace NUMINAMATH_GPT_average_median_eq_l1854_185406

theorem average_median_eq (a b c : ℤ) (h1 : (a + b + c) / 3 = 4 * b)
  (h2 : a < b) (h3 : b < c) (h4 : a = 0) : c / b = 11 := 
by
  sorry

end NUMINAMATH_GPT_average_median_eq_l1854_185406


namespace NUMINAMATH_GPT_percentage_error_edge_percentage_error_edge_l1854_185401

open Real

-- Define the main context, E as the actual edge and E' as the calculated edge
variables (E E' : ℝ)

-- Condition: Error in calculating the area is 4.04%
axiom area_error : E' * E' = E * E * 1.0404

-- Statement: To prove that the percentage error in edge calculation is 2%
theorem percentage_error_edge : (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

-- Alternatively, include variable and condition definitions in the actual theorem statement
theorem percentage_error_edge' (E E' : ℝ) (h : E' * E' = E * E * 1.0404) : 
    (sqrt 1.0404 - 1) * 100 = 2 :=
by sorry

end NUMINAMATH_GPT_percentage_error_edge_percentage_error_edge_l1854_185401


namespace NUMINAMATH_GPT_fraction_of_phones_l1854_185434

-- The total number of valid 8-digit phone numbers (b)
def valid_phone_numbers_total : ℕ := 5 * 10^7

-- The number of valid phone numbers that begin with 5 and end with 2 (a)
def valid_phone_numbers_special : ℕ := 10^6

-- The fraction of phone numbers that begin with 5 and end with 2
def fraction_phone_numbers_special : ℚ := valid_phone_numbers_special / valid_phone_numbers_total

-- Prove that the fraction of such phone numbers is 1/50
theorem fraction_of_phones : fraction_phone_numbers_special = 1 / 50 := by
  sorry

end NUMINAMATH_GPT_fraction_of_phones_l1854_185434


namespace NUMINAMATH_GPT_sqrt_of_9_eq_pm_3_l1854_185462

theorem sqrt_of_9_eq_pm_3 : (∃ x : ℤ, x * x = 9) → (∃ x : ℤ, x = 3 ∨ x = -3) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_9_eq_pm_3_l1854_185462


namespace NUMINAMATH_GPT_apple_distribution_ways_l1854_185427

-- Definitions based on conditions
def distribute_apples (a b c : ℕ) : Prop := a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3

-- Non-negative integer solutions to a' + b' + c' = 21
def num_solutions := Nat.choose 23 2

-- Theorem to prove
theorem apple_distribution_ways : distribute_apples 10 10 10 → num_solutions = 253 :=
by
  intros
  sorry

end NUMINAMATH_GPT_apple_distribution_ways_l1854_185427


namespace NUMINAMATH_GPT_roger_shelves_l1854_185424

theorem roger_shelves (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : 
  total_books = 24 → 
  books_taken = 3 → 
  books_per_shelf = 4 → 
  Nat.ceil ((total_books - books_taken) / books_per_shelf) = 6 :=
by
  intros h_total h_taken h_per_shelf
  rw [h_total, h_taken, h_per_shelf]
  sorry

end NUMINAMATH_GPT_roger_shelves_l1854_185424


namespace NUMINAMATH_GPT_maximal_cards_taken_l1854_185496

theorem maximal_cards_taken (cards : Finset ℕ) (h_cards : ∀ n, n ∈ cards ↔ 1 ≤ n ∧ n ≤ 100)
                            (andriy_cards nick_cards : Finset ℕ)
                            (h_card_count : andriy_cards.card = nick_cards.card)
                            (h_card_relation : ∀ n, n ∈ andriy_cards → (2 * n + 2) ∈ nick_cards) :
                            andriy_cards.card + nick_cards.card ≤ 50 := 
sorry

end NUMINAMATH_GPT_maximal_cards_taken_l1854_185496


namespace NUMINAMATH_GPT_initial_bees_l1854_185438

theorem initial_bees (B : ℕ) (h : B + 8 = 24) : B = 16 := 
by {
  sorry
}

end NUMINAMATH_GPT_initial_bees_l1854_185438


namespace NUMINAMATH_GPT_painting_time_l1854_185471

noncomputable def bob_rate : ℕ := 120 / 8
noncomputable def alice_rate : ℕ := 150 / 10
noncomputable def combined_rate : ℕ := bob_rate + alice_rate
noncomputable def total_area : ℕ := 120 + 150
noncomputable def working_time : ℕ := total_area / combined_rate
noncomputable def lunch_break : ℕ := 1
noncomputable def total_time : ℕ := working_time + lunch_break

theorem painting_time : total_time = 10 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_painting_time_l1854_185471


namespace NUMINAMATH_GPT_bags_of_white_flour_l1854_185465

theorem bags_of_white_flour (total_flour wheat_flour : ℝ) (h1 : total_flour = 0.3) (h2 : wheat_flour = 0.2) : 
  total_flour - wheat_flour = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_bags_of_white_flour_l1854_185465


namespace NUMINAMATH_GPT_cos_2015_eq_neg_m_l1854_185414

variable (m : ℝ)

-- Given condition
axiom sin_55_eq_m : Real.sin (55 * Real.pi / 180) = m

-- The proof problem
theorem cos_2015_eq_neg_m : Real.cos (2015 * Real.pi / 180) = -m :=
by
  sorry

end NUMINAMATH_GPT_cos_2015_eq_neg_m_l1854_185414


namespace NUMINAMATH_GPT_field_length_is_112_l1854_185497

-- Define the conditions
def is_pond_side_length : ℕ := 8
def pond_area : ℕ := is_pond_side_length * is_pond_side_length
def pond_to_field_area_ratio : ℚ := 1 / 98

-- Define the field properties
def field_area (w l : ℕ) : ℕ := w * l

-- Expressing the condition given length is double the width
def length_double_width (w l : ℕ) : Prop := l = 2 * w

-- Equating the areas based on the ratio given
def area_condition (w l : ℕ) : Prop := pond_area = pond_to_field_area_ratio * field_area w l

-- The main theorem
theorem field_length_is_112 : ∃ w l, length_double_width w l ∧ area_condition w l ∧ l = 112 := by
  sorry

end NUMINAMATH_GPT_field_length_is_112_l1854_185497


namespace NUMINAMATH_GPT_min_guesses_correct_l1854_185486

noncomputable def min_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  min_guesses n k h = if n = 2 * k then 2 else 1 :=
by
  sorry

end NUMINAMATH_GPT_min_guesses_correct_l1854_185486


namespace NUMINAMATH_GPT_find_TS_l1854_185498

-- Definitions of the conditions as given:
def PQ : ℝ := 25
def PS : ℝ := 25
def QR : ℝ := 15
def RS : ℝ := 15
def PT : ℝ := 15
def ST_parallel_QR : Prop := true  -- ST is parallel to QR (used as a given fact)

-- Main statement in Lean:
theorem find_TS (h1 : PQ = 25) (h2 : PS = 25) (h3 : QR = 15) (h4 : RS = 15) (h5 : PT = 15)
               (h6 : ST_parallel_QR) : TS = 24 :=
by
  sorry

end NUMINAMATH_GPT_find_TS_l1854_185498


namespace NUMINAMATH_GPT_solution_set_inequality_l1854_185488

variable (f : ℝ → ℝ)

-- Conditions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_non_neg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x
def f_neg_half_eq_zero (f : ℝ → ℝ) : Prop := f (-1/2) = 0

-- Problem statement
theorem solution_set_inequality (f : ℝ → ℝ) 
  (hf_even : is_even_function f) 
  (hf_decreasing : is_decreasing_on_non_neg f) 
  (hf_neg_half_zero : f_neg_half_eq_zero f) : 
  {x : ℝ | f (Real.logb (1/4) x) < 0} = {x | x > 2} ∪ {x | 0 < x ∧ x < 1/2} :=
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1854_185488


namespace NUMINAMATH_GPT_jenna_round_trip_pay_l1854_185463

theorem jenna_round_trip_pay :
  let pay_per_mile := 0.40
  let one_way_miles := 400
  let round_trip_miles := 2 * one_way_miles
  let total_pay := round_trip_miles * pay_per_mile
  total_pay = 320 := 
by
  sorry

end NUMINAMATH_GPT_jenna_round_trip_pay_l1854_185463


namespace NUMINAMATH_GPT_balls_in_boxes_l1854_185454

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1854_185454


namespace NUMINAMATH_GPT_probability_sum_odd_correct_l1854_185469

noncomputable def probability_sum_odd : ℚ :=
  let total_ways := 10
  let ways_sum_odd := 6
  ways_sum_odd / total_ways

theorem probability_sum_odd_correct :
  probability_sum_odd = 3 / 5 :=
by
  unfold probability_sum_odd
  rfl

end NUMINAMATH_GPT_probability_sum_odd_correct_l1854_185469


namespace NUMINAMATH_GPT_cone_sphere_ratio_l1854_185491

theorem cone_sphere_ratio (r h : ℝ) (π_pos : 0 < π) (r_pos : 0 < r) :
  (1/3) * π * r^2 * h = (1/3) * (4/3) * π * r^3 → h / r = 4/3 :=
by
  sorry

end NUMINAMATH_GPT_cone_sphere_ratio_l1854_185491


namespace NUMINAMATH_GPT_pyramid_angles_sum_pi_over_four_l1854_185422

theorem pyramid_angles_sum_pi_over_four :
  ∃ (α β : ℝ), 
    α + β = Real.pi / 4 ∧ 
    α = Real.arctan ((Real.sqrt 17 - 3) / 4) ∧ 
    β = Real.pi / 4 - Real.arctan ((Real.sqrt 17 - 3) / 4) :=
by
  sorry

end NUMINAMATH_GPT_pyramid_angles_sum_pi_over_four_l1854_185422


namespace NUMINAMATH_GPT_exponential_function_example_l1854_185467

def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a > 0, a ≠ 1 ∧ ∀ x, f x = a ^ x

theorem exponential_function_example : is_exponential_function (fun x => 3 ^ x) :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_example_l1854_185467


namespace NUMINAMATH_GPT_value_of_x_plus_y_l1854_185456

theorem value_of_x_plus_y (x y : ℝ) (h : (x + 1)^2 + |y - 6| = 0) : x + y = 5 :=
by
sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l1854_185456


namespace NUMINAMATH_GPT_halfway_point_l1854_185495

theorem halfway_point (x1 x2 : ℚ) (h1 : x1 = 1 / 6) (h2 : x2 = 5 / 6) : 
  (x1 + x2) / 2 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_halfway_point_l1854_185495


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1854_185470

theorem solution_set_of_inequality (x : ℝ) :
  (3 * x + 5) / (x - 1) > x ↔ x < -1 ∨ (1 < x ∧ x < 5) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1854_185470


namespace NUMINAMATH_GPT_visits_exactly_two_friends_l1854_185426

theorem visits_exactly_two_friends (a_visits b_visits c_visits vacation_period : ℕ) (full_period days : ℕ)
(h_a : a_visits = 4)
(h_b : b_visits = 5)
(h_c : c_visits = 6)
(h_vacation : vacation_period = 30)
(h_full_period : full_period = Nat.lcm (Nat.lcm a_visits b_visits) c_visits)
(h_days : days = 360)
(h_start_vacation : ∀ n, ∃ k, n = k * vacation_period + 30):
  ∃ n, n = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_visits_exactly_two_friends_l1854_185426


namespace NUMINAMATH_GPT_mn_minus_7_is_negative_one_l1854_185492

def opp (x : Int) : Int := -x
def largest_negative_integer : Int := -1
def m := opp (-6)
def n := opp largest_negative_integer

theorem mn_minus_7_is_negative_one : m * n - 7 = -1 := by
  sorry

end NUMINAMATH_GPT_mn_minus_7_is_negative_one_l1854_185492


namespace NUMINAMATH_GPT_inequality_proof_l1854_185413

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  1 ≤ (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ∧
  (x + y) * (x^3 + y^3) / (x^2 + y^2)^2 ≤ 9 / 8 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1854_185413


namespace NUMINAMATH_GPT_rectangle_perimeter_l1854_185444

theorem rectangle_perimeter 
  (w : ℝ) (l : ℝ) (hw : w = Real.sqrt 3) (hl : l = Real.sqrt 6) : 
  2 * (w + l) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1854_185444


namespace NUMINAMATH_GPT_mowing_lawn_time_l1854_185445

def maryRate := 1 / 3
def tomRate := 1 / 4
def combinedRate := 7 / 12
def timeMaryAlone := 1
def lawnLeft := 1 - (timeMaryAlone * maryRate)

theorem mowing_lawn_time:
  (7 / 12) * (8 / 7) = (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_mowing_lawn_time_l1854_185445


namespace NUMINAMATH_GPT_part1_part2_l1854_185483

def A (x : ℝ) (a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def B (x : ℝ) : Prop := (x - 3) * (2 - x) ≥ 0

theorem part1 (a : ℝ) (ha1: a = 1) :
  ∀ x, (A x 1 ∧ B x) ↔ (2 ≤ x ∧ x < 3) :=
sorry

theorem part2 (a : ℝ) (ha1: a = 1) :
  ∀ x, (A x 1 ∨ B x) ↔ (1 < x ∧ x ≤ 3) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1854_185483


namespace NUMINAMATH_GPT_bunch_of_bananas_cost_l1854_185404

def cost_of_bananas (A : ℝ) : ℝ := 5 - A

theorem bunch_of_bananas_cost (A B T : ℝ) (h1 : A + B = 5) (h2 : 2 * A + B = T) : B = cost_of_bananas A :=
by
  sorry

end NUMINAMATH_GPT_bunch_of_bananas_cost_l1854_185404


namespace NUMINAMATH_GPT_line_equation_l1854_185475

noncomputable def arithmetic_sequence (n : ℕ) (a_1 d : ℝ) : ℝ :=
  a_1 + (n - 1) * d

theorem line_equation
  (a_2 a_4 a_5 : ℝ)
  (a_2_cond : a_2 = arithmetic_sequence 2 a_1 d)
  (a_4_cond : a_4 = arithmetic_sequence 4 a_1 d)
  (a_5_cond : a_5 = arithmetic_sequence 5 a_1 d)
  (sum_cond : a_2 + a_4 = 12)
  (a_5_val : a_5 = 10)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * y = 0 ↔ (x - 0)^2 + (y - 1)^2 = 1)
  : ∃ (line : ℝ → ℝ → Prop), line x y ↔ (6 * x - y + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l1854_185475


namespace NUMINAMATH_GPT_doughnuts_per_box_l1854_185447

theorem doughnuts_per_box (total_doughnuts : ℕ) (boxes : ℕ) (h_doughnuts : total_doughnuts = 48) (h_boxes : boxes = 4) : 
  total_doughnuts / boxes = 12 :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_doughnuts_per_box_l1854_185447


namespace NUMINAMATH_GPT_complement_union_l1854_185420

variable (U M N : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : M = {1, 2})
variable (hN : N = {3, 4})

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) :
  U \ (M ∪ N) = {5} :=
sorry

end NUMINAMATH_GPT_complement_union_l1854_185420


namespace NUMINAMATH_GPT_seating_arrangements_exactly_two_adjacent_empty_seats_l1854_185436

theorem seating_arrangements_exactly_two_adjacent_empty_seats : 
  (∃ (arrangements : ℕ), arrangements = 72) :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_exactly_two_adjacent_empty_seats_l1854_185436


namespace NUMINAMATH_GPT_hyperbola_foci_distance_l1854_185419

theorem hyperbola_foci_distance (c : ℝ) (h : c = Real.sqrt 2) : 
  let f1 := (c * Real.sqrt 2, c * Real.sqrt 2)
  let f2 := (-c * Real.sqrt 2, -c * Real.sqrt 2)
  Real.sqrt ((f2.1 - f1.1) ^ 2 + (f2.2 - f1.2) ^ 2) = 4 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_foci_distance_l1854_185419


namespace NUMINAMATH_GPT_simplify_and_evaluate_fraction_l1854_185448

theorem simplify_and_evaluate_fraction (x : ℤ) (hx : x = 5) :
  ((2 * x + 1) / (x - 1) - 1) / ((x + 2) / (x^2 - 2 * x + 1)) = 4 :=
by
  rw [hx]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_fraction_l1854_185448


namespace NUMINAMATH_GPT_cos_neg_13pi_over_4_l1854_185443

theorem cos_neg_13pi_over_4 : Real.cos (-13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_neg_13pi_over_4_l1854_185443


namespace NUMINAMATH_GPT_solution_eq_c_l1854_185461

variables (x : ℝ) (a : ℝ) 

def p := ∃ x0 : ℝ, (0 < x0) ∧ (3^x0 + x0 = 2016)
def q := ∃ a : ℝ, (0 < a) ∧ (∀ x : ℝ, (|x| - a * x) = (|(x)| - a * (-x)))

theorem solution_eq_c : p ∧ ¬q :=
by {
  sorry -- proof placeholder
}

end NUMINAMATH_GPT_solution_eq_c_l1854_185461


namespace NUMINAMATH_GPT_geometric_sequence_S_n_l1854_185411

-- Definitions related to the sequence
def a_n (n : ℕ) : ℕ := sorry  -- Placeholder for the actual sequence

-- Sum of the first n terms
def S_n (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms

-- Given conditions
axiom a1 : a_n 1 = 1
axiom Sn_eq_2an_plus1 : ∀ (n : ℕ), S_n n = 2 * a_n (n + 1)

-- Theorem to be proved
theorem geometric_sequence_S_n 
    (n : ℕ) (h : n > 1) 
    : S_n n = (3/2)^(n-1) := 
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_S_n_l1854_185411


namespace NUMINAMATH_GPT_opposite_of_9_is_neg_9_l1854_185410

-- Definition of opposite number according to the given condition
def opposite (n : Int) : Int := -n

-- Proof statement that the opposite of 9 is -9
theorem opposite_of_9_is_neg_9 : opposite 9 = -9 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_9_is_neg_9_l1854_185410


namespace NUMINAMATH_GPT_value_of_expression_l1854_185437

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  (a + b + c + d).sqrt + (a^2 - 2*a + 3 - b).sqrt - (b - c^2 + 4*c - 8).sqrt = 3

theorem value_of_expression (a b c d : ℝ) (h : proof_problem a b c d) : a - b + c - d = -7 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1854_185437


namespace NUMINAMATH_GPT_correct_operation_l1854_185481

theorem correct_operation (x y m c d : ℝ) : (5 * x * y - 4 * x * y = x * y) :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1854_185481


namespace NUMINAMATH_GPT_meaningful_expression_range_l1854_185466

theorem meaningful_expression_range (x : ℝ) :
  (x + 2 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ≥ -2) ∧ (x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_range_l1854_185466


namespace NUMINAMATH_GPT_original_circle_area_l1854_185477

theorem original_circle_area (A : ℝ) (h1 : ∃ sector_area : ℝ, sector_area = 5) (h2 : A / 64 = 5) : A = 320 := 
by sorry

end NUMINAMATH_GPT_original_circle_area_l1854_185477


namespace NUMINAMATH_GPT_travel_time_difference_is_58_minutes_l1854_185473

-- Define the distances and speeds for Minnie
def minnie_uphill_distance := 15
def minnie_uphill_speed := 10
def minnie_downhill_distance := 25
def minnie_downhill_speed := 40
def minnie_flat_distance := 30
def minnie_flat_speed := 25

-- Define the distances and speeds for Penny
def penny_flat_distance := 30
def penny_flat_speed := 35
def penny_downhill_distance := 25
def penny_downhill_speed := 50
def penny_uphill_distance := 15
def penny_uphill_speed := 15

-- Calculate Minnie's total travel time in hours
def minnie_time := (minnie_uphill_distance / minnie_uphill_speed) + 
                   (minnie_downhill_distance / minnie_downhill_speed) + 
                   (minnie_flat_distance / minnie_flat_speed)

-- Calculate Penny's total travel time in hours
def penny_time := (penny_flat_distance / penny_flat_speed) + 
                  (penny_downhill_distance / penny_downhill_speed) +
                  (penny_uphill_distance / penny_uphill_speed)

-- Calculate difference in minutes
def time_difference_minutes := (minnie_time - penny_time) * 60

-- The proof statement
theorem travel_time_difference_is_58_minutes :
  time_difference_minutes = 58 := by
  sorry

end NUMINAMATH_GPT_travel_time_difference_is_58_minutes_l1854_185473


namespace NUMINAMATH_GPT_janet_more_siblings_than_carlos_l1854_185432

-- Define the initial conditions
def masud_siblings := 60
def carlos_siblings := (3 / 4) * masud_siblings
def janet_siblings := 4 * masud_siblings - 60

-- The statement to be proved
theorem janet_more_siblings_than_carlos : janet_siblings - carlos_siblings = 135 :=
by
  sorry

end NUMINAMATH_GPT_janet_more_siblings_than_carlos_l1854_185432


namespace NUMINAMATH_GPT_chloe_fifth_test_score_l1854_185487

theorem chloe_fifth_test_score (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 84) (h2 : a2 = 87) (h3 : a3 = 78) (h4 : a4 = 90)
  (h_avg : (a1 + a2 + a3 + a4 + a5) / 5 ≥ 85) : 
  a5 ≥ 86 :=
by
  sorry

end NUMINAMATH_GPT_chloe_fifth_test_score_l1854_185487


namespace NUMINAMATH_GPT_population_growth_l1854_185435

theorem population_growth (scale_factor1 scale_factor2 : ℝ)
    (h1 : scale_factor1 = 1.2)
    (h2 : scale_factor2 = 1.26) :
    (scale_factor1 * scale_factor2) - 1 = 0.512 :=
by
  sorry

end NUMINAMATH_GPT_population_growth_l1854_185435


namespace NUMINAMATH_GPT_original_price_l1854_185402

variable (P : ℝ)

theorem original_price (h : 560 = 1.05 * (0.72 * P)) : P = 740.46 := 
by
  sorry

end NUMINAMATH_GPT_original_price_l1854_185402


namespace NUMINAMATH_GPT_find_N_l1854_185415

theorem find_N (N : ℕ) (h₁ : ∃ (d₁ d₂ : ℕ), d₁ + d₂ = 3333 ∧ N = max d₁ d₂ ∧ (max d₁ d₂) / (min d₁ d₂) = 2) : 
  N = 2222 := sorry

end NUMINAMATH_GPT_find_N_l1854_185415


namespace NUMINAMATH_GPT_total_sticks_used_l1854_185421

-- Definitions based on the conditions
def hexagons : Nat := 800
def sticks_for_first_hexagon : Nat := 6
def sticks_per_additional_hexagon : Nat := 5

-- The theorem to prove
theorem total_sticks_used :
  sticks_for_first_hexagon + (hexagons - 1) * sticks_per_additional_hexagon = 4001 := by
  sorry

end NUMINAMATH_GPT_total_sticks_used_l1854_185421


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1854_185451

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 3 → x^2 > 4) ∧ ¬(x^2 > 4 → x > 3) :=
by sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1854_185451


namespace NUMINAMATH_GPT_mittens_pairing_possible_l1854_185453

/--
In a kindergarten's lost and found basket, there are 30 mittens: 
10 blue, 10 green, 10 red, 15 right-hand, and 15 left-hand. 

Prove that it is always possible to create matching pairs of one right-hand 
and one left-hand mitten of the same color for 5 children.
-/
theorem mittens_pairing_possible : 
  (∃ (right_blue left_blue right_green left_green right_red left_red : ℕ), 
    right_blue + left_blue + right_green + left_green + right_red + left_red = 30 ∧
    right_blue ≤ 10 ∧ left_blue ≤ 10 ∧
    right_green ≤ 10 ∧ left_green ≤ 10 ∧
    right_red ≤ 10 ∧ left_red ≤ 10 ∧
    right_blue + right_green + right_red = 15 ∧
    left_blue + left_green + left_red = 15) →
  (∃ right_blue left_blue right_green left_green right_red left_red,
    min right_blue left_blue + 
    min right_green left_green + 
    min right_red left_red ≥ 5) :=
sorry

end NUMINAMATH_GPT_mittens_pairing_possible_l1854_185453


namespace NUMINAMATH_GPT_coin_change_count_ways_l1854_185428

theorem coin_change_count_ways :
  ∃ n : ℕ, (∀ q h : ℕ, (25 * q + 50 * h = 1500) ∧ q > 0 ∧ h > 0 → (1 ≤ h ∧ h < 30)) ∧ n = 29 :=
  sorry

end NUMINAMATH_GPT_coin_change_count_ways_l1854_185428


namespace NUMINAMATH_GPT_cost_ratio_l1854_185482

theorem cost_ratio (S J M : ℝ) (h1 : S = 4) (h2 : M = 0.75 * (S + J)) (h3 : S + J + M = 21) : J / S = 2 :=
by
  sorry

end NUMINAMATH_GPT_cost_ratio_l1854_185482


namespace NUMINAMATH_GPT_ones_digit_of_9_pow_47_l1854_185450

theorem ones_digit_of_9_pow_47 : (9 ^ 47) % 10 = 9 := 
by
  sorry

end NUMINAMATH_GPT_ones_digit_of_9_pow_47_l1854_185450


namespace NUMINAMATH_GPT_payment_required_l1854_185431

-- Definitions of the conditions
def price_suit : ℕ := 200
def price_tie : ℕ := 40
def num_suits : ℕ := 20
def discount_option_1 (x : ℕ) (hx : x > 20) : ℕ := price_suit * num_suits + (x - num_suits) * price_tie
def discount_option_2 (x : ℕ) (hx : x > 20) : ℕ := (price_suit * num_suits + x * price_tie) * 9 / 10

-- Theorem that needs to be proved
theorem payment_required (x : ℕ) (hx : x > 20) :
  discount_option_1 x hx = 40 * x + 3200 ∧ discount_option_2 x hx = 3600 + 36 * x :=
by sorry

end NUMINAMATH_GPT_payment_required_l1854_185431


namespace NUMINAMATH_GPT_speed_of_first_train_l1854_185405

noncomputable def length_of_first_train : ℝ := 280
noncomputable def speed_of_second_train_kmph : ℝ := 80
noncomputable def length_of_second_train : ℝ := 220.04
noncomputable def time_to_cross : ℝ := 9

noncomputable def relative_speed_mps := (length_of_first_train + length_of_second_train) / time_to_cross

noncomputable def relative_speed_kmph := relative_speed_mps * (3600 / 1000)

theorem speed_of_first_train :
  (relative_speed_kmph - speed_of_second_train_kmph) = 120.016 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_first_train_l1854_185405


namespace NUMINAMATH_GPT_intersects_negative_half_axis_range_l1854_185409

noncomputable def f (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * m * x + 2 * m - 6

theorem intersects_negative_half_axis_range (m : ℝ) :
  (1 ≤ m ∧ m < 2) ∨ (2 < m ∧ m < 3) ↔ (∃ x : ℝ, f m x < 0) :=
sorry

end NUMINAMATH_GPT_intersects_negative_half_axis_range_l1854_185409


namespace NUMINAMATH_GPT_two_a_minus_b_equals_four_l1854_185400

theorem two_a_minus_b_equals_four (a b : ℕ) 
    (consec_integers : b = a + 1)
    (min_a : min (Real.sqrt 30) a = a)
    (min_b : min (Real.sqrt 30) b = Real.sqrt 30) : 
    2 * a - b = 4 := 
sorry

end NUMINAMATH_GPT_two_a_minus_b_equals_four_l1854_185400


namespace NUMINAMATH_GPT_min_value_of_z_l1854_185460

theorem min_value_of_z : ∀ x : ℝ, ∃ z : ℝ, z = x^2 + 16 * x + 20 ∧ (∀ y : ℝ, y = x^2 + 16 * x + 20 → z ≤ y) → z = -44 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_z_l1854_185460


namespace NUMINAMATH_GPT_monthly_cost_per_iguana_l1854_185480

theorem monthly_cost_per_iguana
  (gecko_cost snake_cost annual_cost : ℕ)
  (monthly_cost_per_iguana : ℕ)
  (gecko_count iguana_count snake_count : ℕ)
  (annual_cost_eq : annual_cost = 1140)
  (gecko_count_eq : gecko_count = 3)
  (iguana_count_eq : iguana_count = 2)
  (snake_count_eq : snake_count = 4)
  (gecko_cost_eq : gecko_cost = 15)
  (snake_cost_eq : snake_cost = 10)
  (total_annual_cost_eq : gecko_count * gecko_cost + iguana_count * monthly_cost_per_iguana * 12 + snake_count * snake_cost * 12 = annual_cost) :
  monthly_cost_per_iguana = 5 :=
by
  sorry

end NUMINAMATH_GPT_monthly_cost_per_iguana_l1854_185480


namespace NUMINAMATH_GPT_work_together_l1854_185416

variable (W : ℝ) -- 'W' denotes the total work
variable (a_days b_days c_days : ℝ)

-- Conditions provided in the problem
axiom a_work : a_days = 18
axiom b_work : b_days = 6
axiom c_work : c_days = 12

-- The statement to be proved
theorem work_together :
  (W / a_days + W / b_days + W / c_days) * (36 / 11) = W := by
  sorry

end NUMINAMATH_GPT_work_together_l1854_185416


namespace NUMINAMATH_GPT_original_deck_size_l1854_185440

noncomputable def initial_red_probability (r b : ℕ) : Prop := r / (r + b) = 1 / 4
noncomputable def added_black_probability (r b : ℕ) : Prop := r / (r + (b + 6)) = 1 / 6

theorem original_deck_size (r b : ℕ) 
  (h1 : initial_red_probability r b) 
  (h2 : added_black_probability r b) : 
  r + b = 12 := 
sorry

end NUMINAMATH_GPT_original_deck_size_l1854_185440


namespace NUMINAMATH_GPT_alex_loan_difference_l1854_185408

theorem alex_loan_difference :
  let P := (15000 : ℝ)
  let r1 := (0.08 : ℝ)
  let n := (2 : ℕ)
  let t := (12 : ℕ)
  let r2 := (0.09 : ℝ)
  
  -- Calculate the amount owed after 6 years with compound interest (first option)
  let A1_half := P * (1 + r1 / n)^(n * t / 2)
  let half_payment := A1_half / 2
  let remaining_balance := A1_half / 2
  let A1_final := remaining_balance * (1 + r1 / n)^(n * t / 2)
  
  -- Total payment for the first option
  let total1 := half_payment + A1_final
  
  -- Total payment for the second option (simple interest)
  let simple_interest := P * r2 * t
  let total2 := P + simple_interest
  
  -- Compute the positive difference
  let difference := abs (total1 - total2)
  
  difference = 24.59 :=
  by
  sorry

end NUMINAMATH_GPT_alex_loan_difference_l1854_185408


namespace NUMINAMATH_GPT_M_in_fourth_quadrant_l1854_185433

-- Define the conditions
variables (a b : ℝ)

/-- Condition that point A(a, 3) and B(2, b) are symmetric with respect to the x-axis -/
def symmetric_points : Prop :=
  a = 2 ∧ 3 = -b

-- Define the point M and quadrant check
def in_fourth_quadrant (a b : ℝ) : Prop :=
  a > 0 ∧ b < 0

-- The theorem stating that if A(a, 3) and B(2, b) are symmetric wrt x-axis, M is in the fourth quadrant
theorem M_in_fourth_quadrant (a b : ℝ) (h : symmetric_points a b) : in_fourth_quadrant a b :=
by {
  sorry
}

end NUMINAMATH_GPT_M_in_fourth_quadrant_l1854_185433


namespace NUMINAMATH_GPT_b_investment_l1854_185484

theorem b_investment (A_invest C_invest total_profit A_profit x : ℝ) 
(h1 : A_invest = 2400) 
(h2 : C_invest = 9600) 
(h3 : total_profit = 9000) 
(h4 : A_profit = 1125)
(h5 : x = (8100000 / 1125)) : 
x = 7200 := by
  rw [h5]
  sorry

end NUMINAMATH_GPT_b_investment_l1854_185484


namespace NUMINAMATH_GPT_find_multiplier_l1854_185429

theorem find_multiplier (x y: ℤ) (h1: x = 127)
  (h2: x * y - 152 = 102): y = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_multiplier_l1854_185429


namespace NUMINAMATH_GPT_find_n_equal_roots_l1854_185430

theorem find_n_equal_roots (x n : ℝ) (hx : x ≠ 2) : n = -1 ↔
  let a := 1
  let b := -2
  let c := -(n^2 + 2 * n)
  b^2 - 4 * a * c = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_n_equal_roots_l1854_185430


namespace NUMINAMATH_GPT_cost_per_rug_proof_l1854_185455

noncomputable def cost_per_rug (price_sold : ℝ) (number_rugs : ℕ) (profit : ℝ) : ℝ :=
  let total_revenue := number_rugs * price_sold
  let total_cost := total_revenue - profit
  total_cost / number_rugs

theorem cost_per_rug_proof : cost_per_rug 60 20 400 = 40 :=
by
  -- Lean will need the proof steps here, which are skipped
  -- The solution steps illustrate how Lean would derive this in a proof
  sorry

end NUMINAMATH_GPT_cost_per_rug_proof_l1854_185455


namespace NUMINAMATH_GPT_total_value_of_goods_l1854_185446

theorem total_value_of_goods (V : ℝ)
  (h1 : 0 < V)
  (h2 : ∃ t, V - 600 = t ∧ 0.12 * t = 134.4) :
  V = 1720 := 
sorry

end NUMINAMATH_GPT_total_value_of_goods_l1854_185446


namespace NUMINAMATH_GPT_stratified_sampling_red_balls_l1854_185403

-- Define the conditions
def total_balls : ℕ := 1000
def red_balls : ℕ := 50
def sampled_balls : ℕ := 100

-- Prove that the number of red balls sampled using stratified sampling is 5
theorem stratified_sampling_red_balls :
  (red_balls : ℝ) / (total_balls : ℝ) * (sampled_balls : ℝ) = 5 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_red_balls_l1854_185403


namespace NUMINAMATH_GPT_highest_score_of_D_l1854_185452

theorem highest_score_of_D
  (a b c d : ℕ)
  (h1 : a + b = c + d)
  (h2 : b + d > a + c)
  (h3 : a > b + c) :
  d > a :=
by
  sorry

end NUMINAMATH_GPT_highest_score_of_D_l1854_185452


namespace NUMINAMATH_GPT_complement_of_M_l1854_185407

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 4 ≤ 0}

theorem complement_of_M :
  ∀ x, x ∈ U \ M ↔ x < -2 ∨ x > 2 :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_l1854_185407


namespace NUMINAMATH_GPT_quadratic_equation_completes_to_square_l1854_185464

theorem quadratic_equation_completes_to_square :
  ∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x + 2)^2 = 2 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_equation_completes_to_square_l1854_185464


namespace NUMINAMATH_GPT_calc_expression1_calc_expression2_l1854_185423

-- Problem 1
theorem calc_expression1 (x y : ℝ) : (1/2 * x * y)^2 * 6 * x^2 * y = (3/2) * x^4 * y^3 := 
sorry

-- Problem 2
theorem calc_expression2 (a b : ℝ) : (2 * a + b)^2 = 4 * a^2 + 4 * a * b + b^2 := 
sorry

end NUMINAMATH_GPT_calc_expression1_calc_expression2_l1854_185423


namespace NUMINAMATH_GPT_range_of_f_l1854_185457

open Set

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (2 * x - 1)

theorem range_of_f : range f = Ici (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1854_185457


namespace NUMINAMATH_GPT_max_m_eq_half_l1854_185417

noncomputable def f (x m : ℝ) : ℝ := (1/2) * x^2 + m * x + m * Real.log x

theorem max_m_eq_half :
  ∃ m : ℝ, (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 ≤ 2) → (1 ≤ x2 ∧ x2 ≤ 2) → 
  x1 < x2 → |f x1 m - f x2 m| < x2^2 - x1^2)) ∧ m = 1/2 :=
sorry

end NUMINAMATH_GPT_max_m_eq_half_l1854_185417


namespace NUMINAMATH_GPT_quadratic_root_relationship_l1854_185474

theorem quadratic_root_relationship (a b c : ℝ) (α β : ℝ)
  (h1 : a ≠ 0)
  (h2 : α + β = -b / a)
  (h3 : α * β = c / a)
  (h4 : β = 3 * α) : 
  3 * b^2 = 16 * a * c :=
sorry

end NUMINAMATH_GPT_quadratic_root_relationship_l1854_185474


namespace NUMINAMATH_GPT_original_price_l1854_185499

variables (p q d : ℝ)


theorem original_price (x : ℝ) (h : x * (1 + p / 100) * (1 - q / 100) = d) :
  x = 100 * d / (100 + p - q - p * q / 100) := 
sorry

end NUMINAMATH_GPT_original_price_l1854_185499
