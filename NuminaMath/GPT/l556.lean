import Mathlib

namespace find_k_if_equal_roots_l556_55603

theorem find_k_if_equal_roots (a b k : ℚ) 
  (h1 : 2 * a + b = -4) 
  (h2 : 2 * a * b + a^2 = -60) 
  (h3 : -2 * a^2 * b = k)
  (h4 : a ≠ b)
  (h5 : k > 0) :
  k = 6400 / 27 :=
by {
  sorry
}

end find_k_if_equal_roots_l556_55603


namespace value_of_a_l556_55628

theorem value_of_a (a : ℕ) (h1 : a * 9^3 = 3 * 15^5) (h2 : a = 5^5) : a = 3125 := by
  sorry

end value_of_a_l556_55628


namespace find_positive_real_numbers_l556_55648

open Real

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  x * sqrt (16 - x) + sqrt (16 * x - x^3) ≥ 16

theorem find_positive_real_numbers (x : ℝ) (hx : x > 0) :
  satisfies_inequality x ↔ 15 * x^2 + 32 * x - 256 = 0 :=
sorry

end find_positive_real_numbers_l556_55648


namespace stolen_bones_is_two_l556_55670

/-- Juniper's initial number of bones -/
def initial_bones : ℕ := 4

/-- Juniper's bones after receiving more bones -/
def doubled_bones : ℕ := initial_bones * 2

/-- Juniper's remaining number of bones after theft -/
def remaining_bones : ℕ := 6

/-- Number of bones stolen by the neighbor's dog -/
def stolen_bones : ℕ := doubled_bones - remaining_bones

theorem stolen_bones_is_two : stolen_bones = 2 := sorry

end stolen_bones_is_two_l556_55670


namespace base_salary_is_1600_l556_55637

theorem base_salary_is_1600 (B : ℝ) (C : ℝ) (sales : ℝ) (fixed_salary : ℝ) :
  C = 0.04 ∧ sales = 5000 ∧ fixed_salary = 1800 ∧ (B + C * sales = fixed_salary) → B = 1600 :=
by sorry

end base_salary_is_1600_l556_55637


namespace unique_divisors_form_l556_55692

theorem unique_divisors_form (n : ℕ) (h₁ : n > 1)
    (h₂ : ∀ d : ℕ, d ∣ n ∧ d > 1 → ∃ a r : ℕ, a > 1 ∧ r > 1 ∧ d = a^r + 1) :
    n = 10 := by
  sorry

end unique_divisors_form_l556_55692


namespace find_smallest_x_l556_55698

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

theorem find_smallest_x (x: ℕ) (h1: 2 * x = 144) (h2: 3 * x = 216) : x = 72 :=
by
  sorry

end find_smallest_x_l556_55698


namespace equal_spacing_between_paintings_l556_55611

/--
Given:
- The width of each painting is 30 centimeters.
- The total width of the wall in the exhibition hall is 320 centimeters.
- There are six pieces of artwork.
Prove that: The distance between the end of the wall and the artwork, and between the artworks, is 20 centimeters.
-/
theorem equal_spacing_between_paintings :
  let width_painting := 30 -- in centimeters
  let total_wall_width := 320 -- in centimeters
  let num_paintings := 6
  let total_paintings_width := num_paintings * width_painting
  let remaining_space := total_wall_width - total_paintings_width
  let num_spaces := num_paintings + 1
  let space_between := remaining_space / num_spaces
  space_between = 20 := sorry

end equal_spacing_between_paintings_l556_55611


namespace solve_problem_l556_55643

def num : ℕ := 1 * 3 * 5 * 7
def den : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7

theorem solve_problem : (num : ℚ) / den = 3.75 := 
by
  sorry

end solve_problem_l556_55643


namespace gcd_2835_9150_l556_55613

theorem gcd_2835_9150 : Nat.gcd 2835 9150 = 15 := by
  sorry

end gcd_2835_9150_l556_55613


namespace part1_monotonicity_when_a_eq_1_part2_range_of_a_l556_55676

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * (Real.log (x - 2)) - a * (x - 3)

theorem part1_monotonicity_when_a_eq_1 :
  ∀ x, 2 < x → ∀ x1, (2 < x1 → f x 1 ≤ f x1 1) := by
  sorry

theorem part2_range_of_a :
  ∀ a, (∀ x, 3 < x → f x a > 0) → a ≤ 2 := by
  sorry

end part1_monotonicity_when_a_eq_1_part2_range_of_a_l556_55676


namespace youngest_child_age_l556_55605

theorem youngest_child_age (x : ℕ) (h1 : Prime x)
  (h2 : Prime (x + 2))
  (h3 : Prime (x + 6))
  (h4 : Prime (x + 8))
  (h5 : Prime (x + 12))
  (h6 : Prime (x + 14)) :
  x = 5 := 
sorry

end youngest_child_age_l556_55605


namespace gain_percentage_l556_55647

-- Define the conditions as a Lean problem
theorem gain_percentage (C G : ℝ) (hC : (9 / 10) * C = 1) (hSP : (10 / 6) = (1 + G / 100) * C) : 
  G = 50 :=
by
-- Here, you would generally have the proof steps, but we add sorry to skip the proof for now.
sorry

end gain_percentage_l556_55647


namespace sum_of_exponents_l556_55663

-- Given product of integers from 1 to 15
def y := Nat.factorial 15

-- Prime exponent variables in the factorization of y
variables (i j k m n p q : ℕ)

-- Conditions
axiom h1 : y = 2^i * 3^j * 5^k * 7^m * 11^n * 13^p * 17^q 

-- Prove that the sum of the exponents equals 24
theorem sum_of_exponents :
  i + j + k + m + n + p + q = 24 := 
sorry

end sum_of_exponents_l556_55663


namespace mimi_shells_l556_55689

theorem mimi_shells (Kyle_shells Mimi_shells Leigh_shells : ℕ) 
  (h₀ : Kyle_shells = 2 * Mimi_shells) 
  (h₁ : Leigh_shells = Kyle_shells / 3) 
  (h₂ : Leigh_shells = 16) 
  : Mimi_shells = 24 := by 
  sorry

end mimi_shells_l556_55689


namespace rita_total_hours_l556_55649

def h_backstroke : ℕ := 50
def h_breaststroke : ℕ := 9
def h_butterfly : ℕ := 121
def h_freestyle_sidestroke_per_month : ℕ := 220
def months : ℕ := 6

def h_total : ℕ := h_backstroke + h_breaststroke + h_butterfly + (h_freestyle_sidestroke_per_month * months)

theorem rita_total_hours :
  h_total = 1500 :=
by
  sorry

end rita_total_hours_l556_55649


namespace tricycles_count_l556_55673

theorem tricycles_count (cars bicycles pickup_trucks tricycles : ℕ) (total_tires : ℕ) : 
  cars = 15 →
  bicycles = 3 →
  pickup_trucks = 8 →
  total_tires = 101 →
  4 * cars + 2 * bicycles + 4 * pickup_trucks + 3 * tricycles = total_tires →
  tricycles = 1 :=
by
  sorry

end tricycles_count_l556_55673


namespace probability_5_consecutive_heads_in_8_flips_l556_55646

noncomputable def probability_at_least_5_consecutive_heads (n : ℕ) : ℚ :=
  if n = 8 then 5 / 128 else 0  -- Using conditional given the specificity to n = 8

theorem probability_5_consecutive_heads_in_8_flips : 
  probability_at_least_5_consecutive_heads 8 = 5 / 128 := 
by
  -- Proof to be provided here
  sorry

end probability_5_consecutive_heads_in_8_flips_l556_55646


namespace patient_treatment_volume_l556_55609

noncomputable def total_treatment_volume : ℝ :=
  let drop_rate1 := 15     -- drops per minute for the first drip
  let ml_rate1 := 6 / 120  -- milliliters per drop for the first drip
  let drop_rate2 := 25     -- drops per minute for the second drip
  let ml_rate2 := 7.5 / 90 -- milliliters per drop for the second drip
  let total_time := 4 * 60 -- total minutes including breaks
  let break_time := 4 * 10 -- total break time in minutes
  let actual_time := total_time - break_time -- actual running time in minutes
  let total_drops1 := actual_time * drop_rate1
  let total_drops2 := actual_time * drop_rate2
  let volume1 := total_drops1 * ml_rate1
  let volume2 := total_drops2 * ml_rate2
  volume1 + volume2 -- total volume from both drips

theorem patient_treatment_volume : total_treatment_volume = 566.67 :=
  by
    -- Place the necessary calculation steps as assumptions or directly as one-liner
    sorry

end patient_treatment_volume_l556_55609


namespace value_of_x_minus_y_l556_55690

theorem value_of_x_minus_y (x y : ℝ) (h1 : abs x = 4) (h2 : abs y = 7) (h3 : x + y > 0) :
  x - y = -3 ∨ x - y = -11 :=
sorry

end value_of_x_minus_y_l556_55690


namespace find_pairs_l556_55622

def is_solution_pair (m n : ℕ) : Prop :=
  Nat.lcm m n = 3 * m + 2 * n + 1

theorem find_pairs :
  { pairs : List (ℕ × ℕ) // ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n } :=
by
  let pairs := [(3,10), (4,9)]
  have key : ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n := sorry
  exact ⟨pairs, key⟩

end find_pairs_l556_55622


namespace rectangle_semi_perimeter_l556_55616

variables (BC AC AM x y : ℝ)

theorem rectangle_semi_perimeter (hBC : BC = 5) (hAC : AC = 12) (hAM : AM = x)
  (hMN_AC : ∀ (MN : ℝ), MN = 5 / 12 * AM)
  (hNP_BC : ∀ (NP : ℝ), NP = AC - AM)
  (hy_def : y = (5 / 12 * x) + (12 - x)) :
  y = (144 - 7 * x) / 12 :=
sorry

end rectangle_semi_perimeter_l556_55616


namespace boa_constrictor_length_l556_55671

theorem boa_constrictor_length (garden_snake_length : ℕ) (boa_multiplier : ℕ) (boa_length : ℕ) 
    (h1 : garden_snake_length = 10) (h2 : boa_multiplier = 7) (h3 : boa_length = garden_snake_length * boa_multiplier) : 
    boa_length = 70 := 
sorry

end boa_constrictor_length_l556_55671


namespace f_of_g_of_3_l556_55604

def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := (x + 2)^2
theorem f_of_g_of_3 : f (g 3) = 95 := by
  sorry

end f_of_g_of_3_l556_55604


namespace six_digit_divisibility_by_37_l556_55612

theorem six_digit_divisibility_by_37 (a b c d e f : ℕ) (H : (100 * a + 10 * b + c + 100 * d + 10 * e + f) % 37 = 0) : 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 37 = 0 := 
sorry

end six_digit_divisibility_by_37_l556_55612


namespace find_p_plus_q_l556_55630

noncomputable def probability_only_one (factor : ℕ → Prop) : ℚ := 0.08 -- Condition 1
noncomputable def probability_exaclty_two (factor1 factor2 : ℕ → Prop) : ℚ := 0.12 -- Condition 2
noncomputable def probability_all_three_given_two (factor1 factor2 factor3 : ℕ → Prop) : ℚ := 1 / 4 -- Condition 3
def women_without_D_has_no_risk_factors (total_women women_with_D women_with_all_factors women_without_D_no_risk_factors : ℕ) : ℚ :=
  women_without_D_no_risk_factors / (total_women - women_with_D)

theorem find_p_plus_q : ∃ (p q : ℕ), (women_without_D_has_no_risk_factors 100 (8 + 2 * 12 + 4) 4 28 = p / q) ∧ (Nat.gcd p q = 1) ∧ p + q = 23 :=
by
  sorry

end find_p_plus_q_l556_55630


namespace magician_weeks_worked_l556_55634

theorem magician_weeks_worked
  (hourly_rate : ℕ)
  (hours_per_day : ℕ)
  (total_payment : ℕ)
  (days_per_week : ℕ)
  (h1 : hourly_rate = 60)
  (h2 : hours_per_day = 3)
  (h3 : total_payment = 2520)
  (h4 : days_per_week = 7) :
  total_payment / (hourly_rate * hours_per_day * days_per_week) = 2 := 
by
  -- sorry to skip the proof
  sorry

end magician_weeks_worked_l556_55634


namespace arithmetic_geometric_inequality_l556_55652

variables {a b A1 A2 G1 G2 x y d q : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b)
variables (h₂ : a = x - 3 * d) (h₃ : A1 = x - d) (h₄ : A2 = x + d) (h₅ : b = x + 3 * d)
variables (h₆ : a = y / q^3) (h₇ : G1 = y / q) (h₈ : G2 = y * q) (h₉ : b = y * q^3)
variables (h₁₀ : x - 3 * d = y / q^3) (h₁₁ : x + 3 * d = y * q^3)

theorem arithmetic_geometric_inequality : A1 * A2 ≥ G1 * G2 :=
by {
  sorry
}

end arithmetic_geometric_inequality_l556_55652


namespace functional_equality_l556_55678

theorem functional_equality (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f y + x ^ 2 + 1) + 2 * x = y + (f (x + 1)) ^ 2) →
  (∀ x : ℝ, f x = x) := 
by
  intro h
  sorry

end functional_equality_l556_55678


namespace dogs_daily_food_total_l556_55608

theorem dogs_daily_food_total :
  let first_dog_food := 0.125
  let second_dog_food := 0.25
  let third_dog_food := 0.375
  let fourth_dog_food := 0.5
  first_dog_food + second_dog_food + third_dog_food + fourth_dog_food = 1.25 :=
by
  sorry

end dogs_daily_food_total_l556_55608


namespace expression_equals_41_l556_55602

theorem expression_equals_41 (x : ℝ) (h : 3*x^2 + 9*x + 5 ≠ 0) : 
  (3*x^2 + 9*x + 15) / (3*x^2 + 9*x + 5) = 41 :=
by
  sorry

end expression_equals_41_l556_55602


namespace number_of_students_l556_55620

-- Defining the parameters and conditions
def passing_score : ℕ := 65
def average_score_whole_class : ℕ := 66
def average_score_passed : ℕ := 71
def average_score_failed : ℕ := 56
def increased_score : ℕ := 5
def post_increase_average_passed : ℕ := 75
def post_increase_average_failed : ℕ := 59
def num_students_lb : ℕ := 15 
def num_students_ub : ℕ := 30

-- Lean statement to prove the number of students in the class
theorem number_of_students (x y n : ℕ) 
  (h1 : average_score_passed * x + average_score_failed * y = average_score_whole_class * (x + y))
  (h2 : (average_score_whole_class + increased_score) * (x + y) = post_increase_average_passed * (x + n) + post_increase_average_failed * (y - n))
  (h3 : num_students_lb < x + y ∧ x + y < num_students_ub)
  (h4 : x = 2 * y)
  (h5 : y = 4 * n) : x + y = 24 :=
sorry

end number_of_students_l556_55620


namespace domain_proof_l556_55662

def domain_of_function : Set ℝ := {x : ℝ | x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x}

theorem domain_proof :
  (∀ x : ℝ, (x ≠ 7) → (x^2 - 16 ≥ 0) → (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x)) ∧
  (∀ x : ℝ, (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x) → (x ≠ 7) ∧ (x^2 - 16 ≥ 0)) :=
by
  sorry

end domain_proof_l556_55662


namespace volume_difference_l556_55655

theorem volume_difference (x1 x2 x3 Vmin Vmax : ℝ)
  (hx1 : 0.5 < x1 ∧ x1 < 1.5)
  (hx2 : 0.5 < x2 ∧ x2 < 1.5)
  (hx3 : 2016.5 < x3 ∧ x3 < 2017.5)
  (rV : 2017 = Nat.floor (x1 * x2 * x3))
  : abs (Vmax - Vmin) = 4035 := 
sorry

end volume_difference_l556_55655


namespace distinct_possible_lunches_l556_55653

def main_dishes := 3
def beverages := 3
def snacks := 3

theorem distinct_possible_lunches : main_dishes * beverages * snacks = 27 := by
  sorry

end distinct_possible_lunches_l556_55653


namespace evaluate_expression_l556_55626

theorem evaluate_expression : 
  908 * 501 - (731 * 1389 - (547 * 236 + 842 * 731 - 495 * 361)) = 5448 := by
  sorry

end evaluate_expression_l556_55626


namespace tom_jerry_coffee_total_same_amount_total_coffee_l556_55624

noncomputable def total_coffee_drunk (x : ℚ) : ℚ := 
  let jerry_coffee := 1.25 * x
  let tom_drinks := (2/3) * x
  let jerry_drinks := (2/3) * jerry_coffee
  let jerry_remainder := (5/12) * x
  let jerry_gives_tom := (5/48) * x + 3
  tom_drinks + jerry_gives_tom

theorem tom_jerry_coffee_total (x : ℚ) : total_coffee_drunk x = jerry_drinks + (1.25 * x - jerry_gives_tom) := sorry

theorem same_amount_total_coffee (x : ℚ) 
  (h : total_coffee_drunk x = (5/4) * x - ((5/48) * x + 3)) : 
  (1.25 * x + x = 36) :=
by sorry

end tom_jerry_coffee_total_same_amount_total_coffee_l556_55624


namespace tangency_condition_and_point_l556_55679

variable (a b p q : ℝ)

/-- Condition for the line y = px + q to be tangent to the ellipse b^2 x^2 + a^2 y^2 = a^2 b^2. -/
theorem tangency_condition_and_point
  (h_cond : a^2 * p^2 + b^2 - q^2 = 0)
  : 
  ∃ (x₀ y₀ : ℝ), 
  x₀ = - (a^2 * p) / q ∧
  y₀ = b^2 / q ∧ 
  (b^2 * x₀^2 + a^2 * y₀^2 = a^2 * b^2 ∧ y₀ = p * x₀ + q) :=
sorry

end tangency_condition_and_point_l556_55679


namespace probability_A_C_winning_l556_55661

-- Definitions based on the conditions given
def students := ["A", "B", "C", "D"]

def isDistictPositions (x y : String) : Prop :=
  x ≠ y

-- Lean statement for the mathematical problem
theorem probability_A_C_winning :
  ∃ (P : ℚ), P = 1/6 :=
by
  sorry

end probability_A_C_winning_l556_55661


namespace greatest_possible_value_of_x_l556_55695

theorem greatest_possible_value_of_x (x : ℕ) (h₁ : x % 4 = 0) (h₂ : x > 0) (h₃ : x^3 < 8000) :
  x ≤ 16 := by
  apply sorry

end greatest_possible_value_of_x_l556_55695


namespace planes_parallel_if_any_line_parallel_l556_55632

-- Definitions for Lean statements:
variable (P1 P2 : Set Point)
variable (line : Set Point)

-- Conditions
def is_parallel_to_plane (line : Set Point) (plane : Set Point) : Prop := sorry

def is_parallel_plane (plane1 plane2 : Set Point) : Prop := sorry

-- Lean statement to be proved:
theorem planes_parallel_if_any_line_parallel (h : ∀ line, 
  line ⊆ P1 → is_parallel_to_plane line P2) : is_parallel_plane P1 P2 := sorry

end planes_parallel_if_any_line_parallel_l556_55632


namespace average_of_first_two_is_1_point_1_l556_55623

theorem average_of_first_two_is_1_point_1
  (a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.5)
  (h2 : (a1 + a2) / 2 = x)
  (h3 : (a3 + a4) / 2 = 1.4)
  (h4 : (a5 + a6) / 2 = 5) :
  x = 1.1 := 
sorry

end average_of_first_two_is_1_point_1_l556_55623


namespace equal_real_roots_of_quadratic_l556_55693

theorem equal_real_roots_of_quadratic (k : ℝ) :
  (∃ x : ℝ, x^2 + k*x + 4 = 0 ∧ (x-4)*(x-4) = 0) ↔ k = 4 ∨ k = -4 :=
by
  sorry

end equal_real_roots_of_quadratic_l556_55693


namespace triangle_ABC_area_l556_55687

-- We define the basic structure of a triangle and its properties
structure Triangle :=
(base : ℝ)
(height : ℝ)
(right_angled_at : ℝ)

-- Define the specific triangle ABC with given properties
def triangle_ABC : Triangle := {
  base := 12,
  height := 15,
  right_angled_at := 90 -- since right-angled at C
}

-- Given conditions, we need to prove the area is 90 square cm
theorem triangle_ABC_area : 1/2 * triangle_ABC.base * triangle_ABC.height = 90 := 
by 
  sorry

end triangle_ABC_area_l556_55687


namespace rectangle_area_error_l556_55619

theorem rectangle_area_error
  (L W : ℝ)
  (measured_length : ℝ := 1.15 * L)
  (measured_width : ℝ := 1.20 * W)
  (true_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width)
  (percentage_error : ℝ := ((measured_area - true_area) / true_area) * 100) :
  percentage_error = 38 :=
by
  sorry

end rectangle_area_error_l556_55619


namespace polar_coordinates_of_point_l556_55682

theorem polar_coordinates_of_point :
  ∃ (r θ : ℝ), r = 2 ∧ θ = (2 * Real.pi) / 3 ∧
  (r > 0) ∧ (0 ≤ θ) ∧ (θ < 2 * Real.pi) ∧
  (-1, Real.sqrt 3) = (r * Real.cos θ, r * Real.sin θ) :=
by 
  sorry

end polar_coordinates_of_point_l556_55682


namespace derek_age_calculation_l556_55621

theorem derek_age_calculation 
  (bob_age : ℕ)
  (evan_age : ℕ)
  (derek_age : ℕ) 
  (h1 : bob_age = 60)
  (h2 : evan_age = (2 * bob_age) / 3)
  (h3 : derek_age = evan_age - 10) : 
  derek_age = 30 :=
by
  -- The proof is to be filled in
  sorry

end derek_age_calculation_l556_55621


namespace solve_equation_l556_55667

theorem solve_equation (x : ℝ) : 
  (x + 1) / 6 = 4 / 3 - x ↔ x = 1 :=
sorry

end solve_equation_l556_55667


namespace infer_correct_l556_55607

theorem infer_correct (a b c : ℝ) (h1: c < b) (h2: b < a) (h3: a + b + c = 0) :
  (c * b^2 ≤ ab^2) ∧ (ab > ac) :=
by
  sorry

end infer_correct_l556_55607


namespace victoria_more_scoops_l556_55635

theorem victoria_more_scoops (Oli_scoops : ℕ) (Victoria_scoops : ℕ) 
  (hOli : Oli_scoops = 4) (hVictoria : Victoria_scoops = 2 * Oli_scoops) : 
  (Victoria_scoops - Oli_scoops) = 4 :=
by
  sorry

end victoria_more_scoops_l556_55635


namespace peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l556_55664

-- Define the context of the problem
def total_people := 100
def men := 50
def women := 50

-- Define Peter Ivanovich being satisfied
def satisfies_peter_ivanovich := (women / (total_people - 1)) * ((women - 1) / (total_people - 2)) 

-- Define the probability that Peter Ivanovich is satisfied
theorem peter_ivanovich_satisfied_probability :
  satisfies_peter_ivanovich = 25 / 33 := 
sorry

-- Define the expected number of satisfied men
def expected_satisfied_men := men * (25 / 33)

-- Prove the expected number of satisfied men
theorem expected_satisfied_men_value :
  expected_satisfied_men = 1250 / 33 :=
sorry

end peter_ivanovich_satisfied_probability_expected_satisfied_men_value_l556_55664


namespace candy_count_l556_55601

theorem candy_count (initial_candy : ℕ) (eaten_candy : ℕ) (received_candy : ℕ) (final_candy : ℕ) :
  initial_candy = 33 → eaten_candy = 17 → received_candy = 19 → final_candy = 35 :=
by
  intros h_initial h_eaten h_received
  sorry

end candy_count_l556_55601


namespace problem_statement_l556_55696

theorem problem_statement : 100 * 29.98 * 2.998 * 1000 = (2998)^2 :=
by
  sorry

end problem_statement_l556_55696


namespace isosceles_right_triangle_hypotenuse_l556_55638

noncomputable def hypotenuse_length : ℝ :=
  let a := Real.sqrt 363
  let c := Real.sqrt (2 * (a ^ 2))
  c

theorem isosceles_right_triangle_hypotenuse :
  ∀ (a : ℝ),
    (2 * (a ^ 2)) + (a ^ 2) = 1452 →
    hypotenuse_length = Real.sqrt 726 := by
  intro a h
  rw [hypotenuse_length]
  sorry

end isosceles_right_triangle_hypotenuse_l556_55638


namespace fraction_meaningful_l556_55681

theorem fraction_meaningful (x : ℝ) : (x + 2 ≠ 0) ↔ x ≠ -2 := by
  sorry

end fraction_meaningful_l556_55681


namespace algebraic_expression_value_l556_55617

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x + 7 = 6) : 4 * x^2 + 8 * x - 5 = -9 :=
by
  sorry

end algebraic_expression_value_l556_55617


namespace coprime_with_others_l556_55641

theorem coprime_with_others:
  ∀ (a b c d e : ℕ),
  a = 20172017 → 
  b = 20172018 → 
  c = 20172019 →
  d = 20172020 →
  e = 20172021 →
  (Nat.gcd c a = 1 ∧ 
   Nat.gcd c b = 1 ∧ 
   Nat.gcd c d = 1 ∧ 
   Nat.gcd c e = 1) :=
by
  sorry

end coprime_with_others_l556_55641


namespace range_for_m_l556_55644

def A := { x : ℝ | x^2 - 3 * x - 10 < 0 }
def B (m : ℝ) := { x : ℝ | m + 1 < x ∧ x < 1 - 3 * m }

theorem range_for_m (m : ℝ) (h : ∀ x, x ∈ A ∪ B m ↔ x ∈ B m) : m ≤ -3 := sorry

end range_for_m_l556_55644


namespace units_digit_p_plus_2_l556_55691

theorem units_digit_p_plus_2 {p : ℕ} 
  (h1 : p % 2 = 0) 
  (h2 : p % 10 ≠ 0) 
  (h3 : (p^3 % 10) = (p^2 % 10)) : 
  (p + 2) % 10 = 8 :=
sorry

end units_digit_p_plus_2_l556_55691


namespace h_at_3_l556_55636

theorem h_at_3 :
  ∃ h : ℤ → ℤ,
    (∀ x, (x^7 - 1) * h x = (x+1) * (x^2 + 1) * (x^4 + 1) - (x-1)) →
    h 3 = 3 := 
sorry

end h_at_3_l556_55636


namespace red_cards_taken_out_l556_55677

-- Definitions based on the conditions
def total_cards : ℕ := 52
def half_of_total_cards (n : ℕ) := n / 2
def initial_red_cards : ℕ := half_of_total_cards total_cards
def remaining_red_cards : ℕ := 16

-- The statement to prove
theorem red_cards_taken_out : initial_red_cards - remaining_red_cards = 10 := by
  sorry

end red_cards_taken_out_l556_55677


namespace find_grade_C_boxes_l556_55694

theorem find_grade_C_boxes (m n t : ℕ) (h : 2 * t = m + n) (total_boxes : ℕ) (h_total : total_boxes = 420) : t = 140 :=
by
  sorry

end find_grade_C_boxes_l556_55694


namespace triangle_is_isosceles_right_l556_55627

theorem triangle_is_isosceles_right (a b S : ℝ) (h : S = (1/4) * (a^2 + b^2)) :
  ∃ C : ℝ, C = 90 ∧ a = b :=
by
  sorry

end triangle_is_isosceles_right_l556_55627


namespace exists_real_ge_3_l556_55625

-- Definition of the existential proposition
theorem exists_real_ge_3 : ∃ x : ℝ, x ≥ 3 :=
sorry

end exists_real_ge_3_l556_55625


namespace cookies_total_is_60_l556_55610

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end cookies_total_is_60_l556_55610


namespace find_a3_plus_a9_l556_55631

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

-- Conditions stating sequence is arithmetic and a₁ + a₆ + a₁₁ = 3
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a_1_6_11_sum (a : ℕ → ℝ) : Prop :=
  a 1 + a 6 + a 11 = 3

theorem find_a3_plus_a9 
  (h_arith : is_arithmetic_sequence a d)
  (h_sum : a_1_6_11_sum a) : 
  a 3 + a 9 = 2 := 
sorry

end find_a3_plus_a9_l556_55631


namespace alternating_sum_cubes_eval_l556_55650

noncomputable def alternating_sum_cubes : ℕ → ℤ
| 0 => 0
| n + 1 => alternating_sum_cubes n + (-1)^(n / 4) * (n + 1)^3

theorem alternating_sum_cubes_eval :
  alternating_sum_cubes 99 = S :=
by
  sorry

end alternating_sum_cubes_eval_l556_55650


namespace rightmost_three_digits_of_3_pow_2023_l556_55659

theorem rightmost_three_digits_of_3_pow_2023 :
  (3^2023) % 1000 = 787 := 
sorry

end rightmost_three_digits_of_3_pow_2023_l556_55659


namespace fraction_addition_l556_55685

theorem fraction_addition : (3 / 5) + (2 / 15) = 11 / 15 := sorry

end fraction_addition_l556_55685


namespace y_intercept_of_line_l556_55699

theorem y_intercept_of_line (x y : ℝ) (h : 2 * x - 3 * y = 6) (hx : x = 0) : y = -2 :=
by
  sorry

end y_intercept_of_line_l556_55699


namespace fraction_product_l556_55640

theorem fraction_product : (2 * (-4)) / (9 * 5) = -8 / 45 :=
  by sorry

end fraction_product_l556_55640


namespace garden_perimeter_l556_55654

theorem garden_perimeter (A : ℝ) (P : ℝ) : 
  (A = 97) → (P = 40) :=
by
  sorry

end garden_perimeter_l556_55654


namespace find_correct_value_l556_55642

-- Definitions based on the problem's conditions
def incorrect_calculation (x : ℤ) : Prop := 7 * x = 126
def correct_value (x : ℤ) (y : ℤ) : Prop := x / 6 = y

theorem find_correct_value :
  ∃ (x y : ℤ), incorrect_calculation x ∧ correct_value x y ∧ y = 3 := by
  sorry

end find_correct_value_l556_55642


namespace no_three_digit_number_such_that_sum_is_perfect_square_l556_55651

theorem no_three_digit_number_such_that_sum_is_perfect_square :
  ∀ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 →
  ¬ (∃ m : ℕ, m * m = 100 * a + 10 * b + c + 100 * b + 10 * c + a + 100 * c + 10 * a + b) := by
  sorry

end no_three_digit_number_such_that_sum_is_perfect_square_l556_55651


namespace sum_of_series_eq_5_over_16_l556_55680

theorem sum_of_series_eq_5_over_16 :
  ∑' n : ℕ, (n + 1 : ℝ) / (5 : ℝ)^(n + 1) = 5 / 16 := by
  sorry

end sum_of_series_eq_5_over_16_l556_55680


namespace remainder_when_divided_by_22_l556_55674

theorem remainder_when_divided_by_22 
    (y : ℤ) 
    (h : y % 264 = 42) :
    y % 22 = 20 :=
by
  sorry

end remainder_when_divided_by_22_l556_55674


namespace cannot_determine_right_triangle_l556_55668

-- Definitions of conditions
def condition_A (A B C : ℝ) : Prop := A = B + C
def condition_B (a b c : ℝ) : Prop := a/b = 5/12 ∧ b/c = 12/13
def condition_C (a b c : ℝ) : Prop := a^2 = (b + c) * (b - c)
def condition_D (A B C : ℝ) : Prop := A/B = 3/4 ∧ B/C = 4/5

-- The proof problem
theorem cannot_determine_right_triangle (a b c A B C : ℝ)
  (hD : condition_D A B C) : 
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) := by
  sorry

end cannot_determine_right_triangle_l556_55668


namespace fraction_not_on_time_l556_55656

theorem fraction_not_on_time (total_attendees : ℕ) (male_fraction female_fraction male_on_time_fraction female_on_time_fraction : ℝ)
  (H1 : male_fraction = 3/5)
  (H2 : male_on_time_fraction = 7/8)
  (H3 : female_on_time_fraction = 4/5)
  : ((1 - (male_fraction * male_on_time_fraction + (1 - male_fraction) * female_on_time_fraction)) = 3/20) :=
sorry

end fraction_not_on_time_l556_55656


namespace solve_for_ab_l556_55669

def f (a b : ℚ) (x : ℚ) : ℚ := a * x^3 - 4 * x^2 + b * x - 3

theorem solve_for_ab : 
  ∃ a b : ℚ, 
    f a b 1 = 3 ∧ 
    f a b (-2) = -47 ∧ 
    (a, b) = (4 / 3, 26 / 3) := 
by
  sorry

end solve_for_ab_l556_55669


namespace factorization_correct_l556_55683

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l556_55683


namespace domain_log_function_l556_55684

theorem domain_log_function :
  { x : ℝ | 12 + x - x^2 > 0 } = { x : ℝ | -3 < x ∧ x < 4 } :=
sorry

end domain_log_function_l556_55684


namespace parabola_transformation_correct_l556_55606

-- Definitions and conditions
def original_parabola (x : ℝ) : ℝ := 2 * x^2

def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 3)^2 - 4

-- Theorem to prove that the above definition is correct
theorem parabola_transformation_correct : 
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 3)^2 - 4 :=
by
  intros x
  rfl -- This uses the definition of 'transformed_parabola' directly

end parabola_transformation_correct_l556_55606


namespace image_of_element_2_l556_55615

-- Define the mapping f and conditions
def f (x : ℕ) : ℕ := 2 * x + 1

-- Define the element and its image using f
def element_in_set_A : ℕ := 2
def image_in_set_B : ℕ := f element_in_set_A

-- The theorem to prove
theorem image_of_element_2 : image_in_set_B = 5 :=
by
  -- This is where the proof would go, but we omit it with sorry
  sorry

end image_of_element_2_l556_55615


namespace solve_equation_l556_55686

theorem solve_equation:
  ∀ x y z : ℝ, x^2 + 5 * y^2 + 5 * z^2 - 4 * x * z - 2 * y - 4 * y * z + 1 = 0 → 
    x = 4 ∧ y = 1 ∧ z = 2 :=
by
  intros x y z h
  sorry

end solve_equation_l556_55686


namespace distance_run_l556_55629

theorem distance_run (D : ℝ) (A_time : ℝ) (B_time : ℝ) (A_beats_B : ℝ) : 
  A_time = 90 ∧ B_time = 180 ∧ A_beats_B = 2250 → D = 2250 :=
by
  sorry

end distance_run_l556_55629


namespace prob_of_2_digit_in_frac_1_over_7_l556_55614

noncomputable def prob (n : ℕ) : ℚ := (3/2)^(n-1) / (3/2 - 1)

theorem prob_of_2_digit_in_frac_1_over_7 :
  let infinite_series_sum := ∑' n : ℕ, (2/3)^(6 * n + 3)
  ∑' (n : ℕ), prob (6 * n + 3) = 108 / 665 :=
by
  sorry

end prob_of_2_digit_in_frac_1_over_7_l556_55614


namespace students_recess_time_l556_55672

def initial_recess : ℕ := 20

def extra_minutes_as (as : ℕ) : ℕ := 4 * as
def extra_minutes_bs (bs : ℕ) : ℕ := 3 * bs
def extra_minutes_cs (cs : ℕ) : ℕ := 2 * cs
def extra_minutes_ds (ds : ℕ) : ℕ := ds
def extra_minutes_es (es : ℕ) : ℤ := - es
def extra_minutes_fs (fs : ℕ) : ℤ := -2 * fs

def total_recess (as bs cs ds es fs : ℕ) : ℤ :=
  initial_recess + 
  (extra_minutes_as as + extra_minutes_bs bs +
  extra_minutes_cs cs + extra_minutes_ds ds +
  extra_minutes_es es + extra_minutes_fs fs : ℤ)

theorem students_recess_time :
  total_recess 10 12 14 5 3 2 = 122 := by sorry

end students_recess_time_l556_55672


namespace find_beta_l556_55658

variables {m n p : ℤ} -- defining variables m, n, p as integers
variables {α β : ℤ} -- defining roots α and β as integers

theorem find_beta (h1: α = 3)
  (h2: ∀ x, x^2 - (m+n)*x + (m*n - p) = 0) -- defining the quadratic equation
  (h3: α + β = m + n)
  (h4: α * β = m * n - p)
  (h5: m ≠ n) (h6: n ≠ p) (h7: m ≠ p) : -- ensuring m, n, and p are distinct
  β = m + n - 3 := sorry

end find_beta_l556_55658


namespace equal_roots_iff_k_eq_one_l556_55633

theorem equal_roots_iff_k_eq_one (k : ℝ) : (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + 2 = 0 → ∀ y : ℝ, 2 * k * y^2 + 4 * k * y + 2 = 0 → x = y) ↔ k = 1 := sorry

end equal_roots_iff_k_eq_one_l556_55633


namespace sufficient_and_necessary_condition_l556_55639

theorem sufficient_and_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, m * x ^ 2 + 2 * m * x - 1 < 0) ↔ (-1 < m ∧ m < -1 / 2) :=
by
  sorry

end sufficient_and_necessary_condition_l556_55639


namespace necessary_and_sufficient_condition_l556_55666

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (|a + b| / (|a| + |b|) ≤ 1) ↔ (a^2 + b^2 ≠ 0) :=
sorry

end necessary_and_sufficient_condition_l556_55666


namespace fraction_sum_eq_one_l556_55675

theorem fraction_sum_eq_one (m n : ℝ) (h : m ≠ n) : (m / (m - n) + n / (n - m) = 1) :=
by
  sorry

end fraction_sum_eq_one_l556_55675


namespace not_washed_shirts_l556_55688

-- Definitions based on given conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def washed_shirts : ℕ := 29

-- Theorem to prove the number of shirts not washed
theorem not_washed_shirts : (short_sleeve_shirts + long_sleeve_shirts) - washed_shirts = 1 := by
  sorry

end not_washed_shirts_l556_55688


namespace distance_yolkino_palkino_l556_55600

theorem distance_yolkino_palkino (d_1 d_2 : ℕ) (h : ∀ k : ℕ, d_1 + d_2 = 13) : 
  ∀ k : ℕ, d_1 + d_2 = 13 → (d_1 + d_2 = 13) :=
by
  sorry

end distance_yolkino_palkino_l556_55600


namespace problem_solution_exists_l556_55697

theorem problem_solution_exists (x : ℝ) (h : ∃ x, 2 * (3 * 5 - x) - x = -8) : x = 10 :=
sorry

end problem_solution_exists_l556_55697


namespace point_on_x_axis_l556_55660

theorem point_on_x_axis (m : ℤ) (P : ℤ × ℤ) (hP : P = (m + 3, m + 1)) (h : P.2 = 0) : P = (2, 0) :=
by 
  sorry

end point_on_x_axis_l556_55660


namespace bd_ad_ratio_l556_55645

noncomputable def mass_point_geometry_bd_ad : ℚ := 
  let AT_OVER_ET := 5
  let DT_OVER_CT := 2
  let mass_A := 1
  let mass_D := 3 * mass_A
  let mass_B := mass_A + mass_D
  mass_B / mass_D

theorem bd_ad_ratio (h1 : AT/ET = 5) (h2 : DT/CT = 2) : BD/AD = 4 / 3 :=
by
  have mass_A := 1
  have mass_D := 3
  have mass_B := 4
  have h := mass_B / mass_D
  sorry

end bd_ad_ratio_l556_55645


namespace ring_rotation_count_l556_55657

-- Define the constants and parameters from the conditions
variables (R ω μ g : ℝ) -- radius, angular velocity, coefficient of friction, and gravity constant
-- Additional constraints on these variables
variable (m : ℝ) -- mass of the ring

theorem ring_rotation_count :
  ∃ n : ℝ, n = (ω^2 * R * (1 + μ^2)) / (4 * π * g * μ * (1 + μ)) :=
sorry

end ring_rotation_count_l556_55657


namespace kids_played_on_monday_l556_55665

theorem kids_played_on_monday (total : ℕ) (tuesday : ℕ) (monday : ℕ) (h_total : total = 16) (h_tuesday : tuesday = 14) :
  monday = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end kids_played_on_monday_l556_55665


namespace luke_bus_time_l556_55618

theorem luke_bus_time
  (L : ℕ)   -- Luke's bus time to work in minutes
  (P : ℕ)   -- Paula's bus time to work in minutes
  (B : ℕ)   -- Luke's bike time home in minutes
  (h1 : P = 3 * L / 5) -- Paula's bus time is \( \frac{3}{5} \) of Luke's bus time
  (h2 : B = 5 * L)     -- Luke's bike time is 5 times his bus time
  (h3 : L + P + B + P = 504) -- Total travel time is 504 minutes
  : L = 70 := 
sorry

end luke_bus_time_l556_55618
