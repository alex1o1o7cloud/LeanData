import Mathlib

namespace NUMINAMATH_GPT_steven_more_peaches_l2064_206402

variable (Jake Steven Jill : ℕ)

-- Conditions
axiom h1 : Jake + 6 = Steven
axiom h2 : Jill = 5
axiom h3 : Jake = 17

-- Goal
theorem steven_more_peaches : Steven - Jill = 18 := by
  sorry

end NUMINAMATH_GPT_steven_more_peaches_l2064_206402


namespace NUMINAMATH_GPT_part1_part2_l2064_206477

-- Define the initial conditions and the given inequality.
def condition1 (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def condition2 (m : ℝ) (x : ℝ) : Prop := x = (1/2)^(m - 1) ∧ 1 < m ∧ m < 2

-- Definitions of the correct ranges
def range_x (x : ℝ) : Prop := 1/2 < x ∧ x < 3/4
def range_a (a : ℝ) : Prop := 1/3 ≤ a ∧ a ≤ 1/2

-- Mathematical equivalent proof problem
theorem part1 {x : ℝ} (h1 : condition1 x (1/4)) (h2 : ∃ (m : ℝ), condition2 m x) : range_x x :=
sorry

theorem part2 {a : ℝ} (h : ∀ x : ℝ, (1/2 < x ∧ x < 1) → condition1 x a) : range_a a :=
sorry

end NUMINAMATH_GPT_part1_part2_l2064_206477


namespace NUMINAMATH_GPT_modulo_remainder_product_l2064_206446

theorem modulo_remainder_product :
  let a := 2022
  let b := 2023
  let c := 2024
  let d := 2025
  let n := 17
  (a * b * c * d) % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_modulo_remainder_product_l2064_206446


namespace NUMINAMATH_GPT_find_a_l2064_206484

theorem find_a (a : ℝ) (h : (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) + (1 / Real.log 7 / Real.log a) = 1) : 
  a = 105 := 
sorry

end NUMINAMATH_GPT_find_a_l2064_206484


namespace NUMINAMATH_GPT_multiple_of_Mel_weight_l2064_206449

/-- Given that Brenda weighs 10 pounds more than a certain multiple of Mel's weight,
    and given that Brenda weighs 220 pounds and Mel's weight is 70 pounds,
    show that the multiple is 3. -/
theorem multiple_of_Mel_weight 
    (Brenda_weight Mel_weight certain_multiple : ℝ) 
    (h1 : Brenda_weight = Mel_weight * certain_multiple + 10)
    (h2 : Brenda_weight = 220)
    (h3 : Mel_weight = 70) :
  certain_multiple = 3 :=
by 
  sorry

end NUMINAMATH_GPT_multiple_of_Mel_weight_l2064_206449


namespace NUMINAMATH_GPT_number_of_cows_l2064_206456

variable (C H : ℕ)

section
-- Condition 1: Cows have 4 legs each
def cows_legs := C * 4

-- Condition 2: Chickens have 2 legs each
def chickens_legs := H * 2

-- Condition 3: The number of legs was 10 more than twice the number of heads
def total_legs := cows_legs C + chickens_legs H = 2 * (C + H) + 10

theorem number_of_cows : total_legs C H → C = 5 :=
by
  intros h
  sorry

end

end NUMINAMATH_GPT_number_of_cows_l2064_206456


namespace NUMINAMATH_GPT_July_husband_age_l2064_206475

namespace AgeProof

variable (HannahAge JulyAge HusbandAge : ℕ)

def double_age_condition (hannah_age : ℕ) (july_age : ℕ) : Prop :=
  hannah_age = 2 * july_age

def twenty_years_later (current_age : ℕ) : ℕ :=
  current_age + 20

def two_years_older (age : ℕ) : ℕ :=
  age + 2

theorem July_husband_age :
  ∃ (hannah_age july_age : ℕ), double_age_condition hannah_age july_age ∧
    twenty_years_later hannah_age = 26 ∧
    twenty_years_later july_age = 23 ∧
    two_years_older (twenty_years_later july_age) = 25 :=
by
  sorry
end AgeProof

end NUMINAMATH_GPT_July_husband_age_l2064_206475


namespace NUMINAMATH_GPT_expression_value_l2064_206441

theorem expression_value 
  (x : ℝ)
  (h : x = 1/5) :
  (x^2 - 4) / (x^2 - 2 * x) = 11 :=
  by
  rw [h]
  sorry

end NUMINAMATH_GPT_expression_value_l2064_206441


namespace NUMINAMATH_GPT_polygon_of_T_has_4_sides_l2064_206448

def T (b : ℝ) (x y : ℝ) : Prop :=
  b ≤ x ∧ x ≤ 4 * b ∧
  b ≤ y ∧ y ≤ 4 * b ∧
  x + y ≥ 3 * b ∧
  x + 2 * b ≥ 2 * y ∧
  2 * y ≥ x + b

noncomputable def sides_of_T (b : ℝ) : ℕ :=
  if b > 0 then 4 else 0

theorem polygon_of_T_has_4_sides (b : ℝ) (hb : b > 0) : sides_of_T b = 4 := by
  sorry

end NUMINAMATH_GPT_polygon_of_T_has_4_sides_l2064_206448


namespace NUMINAMATH_GPT_ratio_of_speeds_l2064_206403

theorem ratio_of_speeds (v_A v_B : ℝ) (d_A d_B t : ℝ) (h1 : d_A = 100) (h2 : d_B = 50) (h3 : v_A = d_A / t) (h4 : v_B = d_B / t) : 
  v_A / v_B = 2 := 
by sorry

end NUMINAMATH_GPT_ratio_of_speeds_l2064_206403


namespace NUMINAMATH_GPT_find_number_l2064_206437

theorem find_number (x : ℕ) (h : x + 56 = 110) : x = 54 :=
sorry

end NUMINAMATH_GPT_find_number_l2064_206437


namespace NUMINAMATH_GPT_original_price_of_second_pair_l2064_206468

variable (P : ℝ) -- original price of the second pair of shoes
variable (discounted_price : ℝ := P / 2)
variable (total_before_discount : ℝ := 40 + discounted_price)
variable (final_payment : ℝ := (3 / 4) * total_before_discount)
variable (payment : ℝ := 60)

theorem original_price_of_second_pair (h : final_payment = payment) : P = 80 :=
by
  -- Skipping the proof with sorry.
  sorry

end NUMINAMATH_GPT_original_price_of_second_pair_l2064_206468


namespace NUMINAMATH_GPT_triangle_sum_is_19_l2064_206419

-- Defining the operation on a triangle
def triangle_op (a b c : ℕ) := a * b - c

-- Defining the vertices of the two triangles
def triangle1 := (4, 2, 3)
def triangle2 := (3, 5, 1)

-- Statement that the sum of the operation results is 19
theorem triangle_sum_is_19 :
  triangle_op (4) (2) (3) + triangle_op (3) (5) (1) = 19 :=
by
  -- Triangle 1 calculation: 4 * 2 - 3 = 8 - 3 = 5
  -- Triangle 2 calculation: 3 * 5 - 1 = 15 - 1 = 14
  -- Sum of calculations: 5 + 14 = 19
  sorry

end NUMINAMATH_GPT_triangle_sum_is_19_l2064_206419


namespace NUMINAMATH_GPT_relationship_between_x_x2_and_x3_l2064_206494

theorem relationship_between_x_x2_and_x3 (x : ℝ) (h : -1 < x ∧ x < 0) :
  x ^ 3 < x ∧ x < x ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_x_x2_and_x3_l2064_206494


namespace NUMINAMATH_GPT_calculate_number_of_girls_l2064_206421

-- Definitions based on the conditions provided
def ratio_girls_to_boys : ℕ := 3
def ratio_boys_to_girls : ℕ := 4
def total_students : ℕ := 35

-- The proof statement
theorem calculate_number_of_girls (k : ℕ) (hk : ratio_girls_to_boys * k + ratio_boys_to_girls * k = total_students) :
  ratio_girls_to_boys * k = 15 :=
by sorry

end NUMINAMATH_GPT_calculate_number_of_girls_l2064_206421


namespace NUMINAMATH_GPT_gcd_sum_product_pairwise_coprime_l2064_206450

theorem gcd_sum_product_pairwise_coprime 
  (a b c : ℤ) 
  (h1 : Int.gcd a b = 1)
  (h2 : Int.gcd b c = 1)
  (h3 : Int.gcd a c = 1) : 
  Int.gcd (a * b + b * c + a * c) (a * b * c) = 1 := 
sorry

end NUMINAMATH_GPT_gcd_sum_product_pairwise_coprime_l2064_206450


namespace NUMINAMATH_GPT_largest_perimeter_l2064_206452

noncomputable def triangle_largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : ℕ :=
7 + 8 + y

theorem largest_perimeter (y : ℕ) (h1 : 1 < y) (h2 : y < 15) : triangle_largest_perimeter y h1 h2 = 29 :=
sorry

end NUMINAMATH_GPT_largest_perimeter_l2064_206452


namespace NUMINAMATH_GPT_black_grid_probability_l2064_206417

theorem black_grid_probability : 
  (let n := 4
   let unit_squares := n * n
   let pairs := unit_squares / 2
   let probability_each_pair := (1:ℝ) / 4
   let total_probability := probability_each_pair ^ pairs
   total_probability = (1:ℝ) / 65536) :=
by
  let n := 4
  let unit_squares := n * n
  let pairs := unit_squares / 2
  let probability_each_pair := (1:ℝ) / 4
  let total_probability := probability_each_pair ^ pairs
  sorry

end NUMINAMATH_GPT_black_grid_probability_l2064_206417


namespace NUMINAMATH_GPT_box_weight_l2064_206434

theorem box_weight (W : ℝ) (h : 7 * (W - 20) = 3 * W) : W = 35 := by
  sorry

end NUMINAMATH_GPT_box_weight_l2064_206434


namespace NUMINAMATH_GPT_last_digit_of_189_in_base_3_is_0_l2064_206412

theorem last_digit_of_189_in_base_3_is_0 : 
  (189 % 3 = 0) :=
sorry

end NUMINAMATH_GPT_last_digit_of_189_in_base_3_is_0_l2064_206412


namespace NUMINAMATH_GPT_people_counted_l2064_206459

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end NUMINAMATH_GPT_people_counted_l2064_206459


namespace NUMINAMATH_GPT_total_amount_l2064_206444

theorem total_amount (T_pq r : ℝ) (h1 : r = 2/3 * T_pq) (h2 : r = 1600) : T_pq + r = 4000 :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_total_amount_l2064_206444


namespace NUMINAMATH_GPT_largest_divisor_for_consecutive_seven_odds_l2064_206471

theorem largest_divisor_for_consecutive_seven_odds (n : ℤ) (h_even : 2 ∣ n) (h_pos : 0 < n) : 
  105 ∣ ((n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)) :=
sorry

end NUMINAMATH_GPT_largest_divisor_for_consecutive_seven_odds_l2064_206471


namespace NUMINAMATH_GPT_paul_initial_crayons_l2064_206425

-- Define the variables for the crayons given away, lost, and left
def crayons_given_away : ℕ := 563
def crayons_lost : ℕ := 558
def crayons_left : ℕ := 332

-- Define the total number of crayons Paul got for his birthday
def initial_crayons : ℕ := 1453

-- The proof statement
theorem paul_initial_crayons :
  initial_crayons = crayons_given_away + crayons_lost + crayons_left :=
sorry

end NUMINAMATH_GPT_paul_initial_crayons_l2064_206425


namespace NUMINAMATH_GPT_area_of_triangle_tangent_line_l2064_206435

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

def tangent_line_at_1 (y x : ℝ) : Prop := y = x - 1

theorem area_of_triangle_tangent_line :
  let tangent_intercept_x : ℝ := 1
  let tangent_intercept_y : ℝ := -1
  let area_of_triangle : ℝ := 1 / 2 * tangent_intercept_x * -tangent_intercept_y
  area_of_triangle = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_tangent_line_l2064_206435


namespace NUMINAMATH_GPT_smallest_int_ends_in_3_div_by_11_l2064_206430

theorem smallest_int_ends_in_3_div_by_11 :
  ∃ k : ℕ, k > 0 ∧ k % 10 = 3 ∧ k % 11 = 0 ∧ k = 33 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_int_ends_in_3_div_by_11_l2064_206430


namespace NUMINAMATH_GPT_complex_magnitude_condition_l2064_206487

noncomputable def magnitude_of_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem complex_magnitude_condition (z : ℂ) (i : ℂ) (h : i * i = -1) (h1 : z - 2 * i = 1 + z * i) :
  magnitude_of_z z = Real.sqrt (10) / 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_complex_magnitude_condition_l2064_206487


namespace NUMINAMATH_GPT_problem_one_l2064_206476

def S_n (n : Nat) : Nat := 
  List.foldl (fun acc x => acc * 10 + 2) 0 (List.replicate n 2)

theorem problem_one : ∃ n ∈ Finset.range 2011, S_n n % 2011 = 0 := 
  sorry

end NUMINAMATH_GPT_problem_one_l2064_206476


namespace NUMINAMATH_GPT_change_in_us_volume_correct_l2064_206474

-- Definition: Change in the total import and export volume of goods in a given year
def change_in_volume (country : String) : Float :=
  if country = "China" then 7.5
  else if country = "United States" then -6.4
  else 0

-- Theorem: The change in the total import and export volume of goods in the United States is correctly represented.
theorem change_in_us_volume_correct :
  change_in_volume "United States" = -6.4 := by
  sorry

end NUMINAMATH_GPT_change_in_us_volume_correct_l2064_206474


namespace NUMINAMATH_GPT_number_of_hikers_in_the_morning_l2064_206458

theorem number_of_hikers_in_the_morning (H : ℕ) :
  41 + 26 + H = 71 → H = 4 :=
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_number_of_hikers_in_the_morning_l2064_206458


namespace NUMINAMATH_GPT_proof_problem_l2064_206443

-- Define the problem conditions
variables (x y : ℝ)

-- State the theorem
theorem proof_problem (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 :=
by sorry

end NUMINAMATH_GPT_proof_problem_l2064_206443


namespace NUMINAMATH_GPT_solve_equation_l2064_206466

theorem solve_equation :
  ∀ x : ℝ, (1 + 2 * x ^ (1/2) - x ^ (1/3) - 2 * x ^ (1/6) = 0) ↔ (x = 1 ∨ x = 1 / 64) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2064_206466


namespace NUMINAMATH_GPT_max_value_of_e_l2064_206445

theorem max_value_of_e (a b c d e : ℝ) 
  (h₁ : a + b + c + d + e = 8) 
  (h₂ : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ 16 / 5 :=
sorry

end NUMINAMATH_GPT_max_value_of_e_l2064_206445


namespace NUMINAMATH_GPT_checkerboard_black_squares_count_l2064_206439

namespace Checkerboard

def is_black (n : ℕ) : Bool :=
  -- Define the alternating pattern of the checkerboard
  (n % 2 = 0)

def black_square_count (n : ℕ) : ℕ :=
  -- Calculate the number of black squares in a checkerboard of size n x n
  if n % 2 = 0 then n * n / 2 else n * n / 2 + n / 2 + 1

def additional_black_squares (n : ℕ) : ℕ :=
  -- Calculate the additional black squares due to modification of every 33rd square in every third row
  ((n - 1) / 3 + 1)

def total_black_squares (n : ℕ) : ℕ :=
  -- Calculate the total black squares considering the modified hypothesis
  black_square_count n + additional_black_squares n

theorem checkerboard_black_squares_count : total_black_squares 33 = 555 := 
  by sorry

end Checkerboard

end NUMINAMATH_GPT_checkerboard_black_squares_count_l2064_206439


namespace NUMINAMATH_GPT_problem_1_problem_2_l2064_206473

-- Problem (1): Proving the solutions for \( x^2 - 3x = 0 \)
theorem problem_1 : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ (x = 0 ∨ x = 3) :=
by
  intro x
  sorry

-- Problem (2): Proving the solutions for \( 5x + 2 = 3x^2 \)
theorem problem_2 : ∀ x : ℝ, 5 * x + 2 = 3 * x^2 ↔ (x = -1/3 ∨ x = 2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2064_206473


namespace NUMINAMATH_GPT_find_a5_l2064_206422

variable {a : ℕ → ℝ}

-- Condition 1: {a_n} is an arithmetic sequence
def arithmetic_sequence (a: ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Condition 2: a1 + a9 = 10
axiom a1_a9_sum : a 1 + a 9 = 10

theorem find_a5 (h_arith : arithmetic_sequence a) : a 5 = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a5_l2064_206422


namespace NUMINAMATH_GPT_sin_double_angle_given_condition_l2064_206498

open Real

variable (x : ℝ)

theorem sin_double_angle_given_condition :
  sin (π / 4 - x) = 3 / 5 → sin (2 * x) = 7 / 25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sin_double_angle_given_condition_l2064_206498


namespace NUMINAMATH_GPT_sin_double_angle_l2064_206455

theorem sin_double_angle (α : ℝ) (h1 : Real.sin α = 1 / 3) (h2 : (π / 2) < α ∧ α < π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := sorry

end NUMINAMATH_GPT_sin_double_angle_l2064_206455


namespace NUMINAMATH_GPT_largest_square_area_l2064_206416

theorem largest_square_area (a b c : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (square_area_sum : a^2 + b^2 + c^2 = 450)
  (area_a : a^2 = 100) :
  c^2 = 225 :=
by
  sorry

end NUMINAMATH_GPT_largest_square_area_l2064_206416


namespace NUMINAMATH_GPT_pendulum_faster_17_seconds_winter_l2064_206499

noncomputable def pendulum_period (l g : ℝ) : ℝ :=
  2 * Real.pi * Real.sqrt (l / g)

noncomputable def pendulum_seconds_faster_in_winter (T : ℝ) (l : ℝ) (g : ℝ) (shorten : ℝ) (hours : ℝ) : ℝ :=
  let summer_period := T
  let winter_length := l - shorten
  let winter_period := pendulum_period winter_length g
  let summer_cycles := (hours * 60 * 60) / summer_period
  let winter_cycles := (hours * 60 * 60) / winter_period
  winter_cycles - summer_cycles

theorem pendulum_faster_17_seconds_winter :
  let T := 1
  let l := 980 * (1 / (4 * Real.pi ^ 2))
  let g := 980
  let shorten := 0.01 / 100
  let hours := 24
  pendulum_seconds_faster_in_winter T l g shorten hours = 17 :=
by
  sorry

end NUMINAMATH_GPT_pendulum_faster_17_seconds_winter_l2064_206499


namespace NUMINAMATH_GPT_smallest_total_marbles_l2064_206415

-- Definitions based on conditions in a)
def urn_contains_marbles : Type := ℕ → ℕ
def red_marbles (u : urn_contains_marbles) := u 0
def white_marbles (u : urn_contains_marbles) := u 1
def blue_marbles (u : urn_contains_marbles) := u 2
def green_marbles (u : urn_contains_marbles) := u 3
def yellow_marbles (u : urn_contains_marbles) := u 4
def total_marbles (u : urn_contains_marbles) := u 0 + u 1 + u 2 + u 3 + u 4

-- Probabilities of selection events
def prob_event_a (u : urn_contains_marbles) := (red_marbles u).choose 5
def prob_event_b (u : urn_contains_marbles) := (white_marbles u).choose 1 * (red_marbles u).choose 4
def prob_event_c (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (red_marbles u).choose 3
def prob_event_d (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (green_marbles u).choose 1 * (red_marbles u).choose 2
def prob_event_e (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (green_marbles u).choose 1 * (yellow_marbles u).choose 1 * (red_marbles u).choose 1

-- Proof that the smallest total number of marbles satisfying the conditions is 33
theorem smallest_total_marbles : ∃ u : urn_contains_marbles, 
    (prob_event_a u = prob_event_b u) ∧ 
    (prob_event_b u = prob_event_c u) ∧ 
    (prob_event_c u = prob_event_d u) ∧ 
    (prob_event_d u = prob_event_e u) ∧ 
    total_marbles u = 33 := sorry

end NUMINAMATH_GPT_smallest_total_marbles_l2064_206415


namespace NUMINAMATH_GPT_number_of_buses_l2064_206400

-- Define the conditions
def vans : ℕ := 9
def people_per_van : ℕ := 8
def people_per_bus : ℕ := 27
def total_people : ℕ := 342

-- Translate the mathematical proof problem
theorem number_of_buses : ∃ buses : ℕ, (vans * people_per_van + buses * people_per_bus = total_people) ∧ (buses = 10) :=
by
  -- calculations to prove the theorem
  sorry

end NUMINAMATH_GPT_number_of_buses_l2064_206400


namespace NUMINAMATH_GPT_odd_function_property_l2064_206492

-- Define that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- The theorem statement
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd_function f) : ∀ x : ℝ, f x * f (-x) ≤ 0 :=
by
  -- The proof is omitted as per the instruction
  sorry

end NUMINAMATH_GPT_odd_function_property_l2064_206492


namespace NUMINAMATH_GPT_least_common_multiple_xyz_l2064_206429

theorem least_common_multiple_xyz (x y z : ℕ) 
  (h1 : Nat.lcm x y = 18) 
  (h2 : Nat.lcm y z = 20) : 
  Nat.lcm x z = 90 := 
sorry

end NUMINAMATH_GPT_least_common_multiple_xyz_l2064_206429


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l2064_206486

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) = 358800) : 
  n + (n + 1) + (n + 2) + (n + 3) = 98 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l2064_206486


namespace NUMINAMATH_GPT_smallest_value_of_3a_plus_2_l2064_206447

theorem smallest_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 10 * a + 6 = 2) : 3 * a + 2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_of_3a_plus_2_l2064_206447


namespace NUMINAMATH_GPT_express_in_scientific_notation_l2064_206404

-- Definitions based on problem conditions
def GDP_first_quarter : ℝ := 27017800000000

-- Main theorem statement that needs to be proved
theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), (GDP_first_quarter = a * 10 ^ b) ∧ (a = 2.70178) ∧ (b = 13) :=
by
  sorry -- Placeholder to indicate the proof is omitted

end NUMINAMATH_GPT_express_in_scientific_notation_l2064_206404


namespace NUMINAMATH_GPT_car_B_speed_90_l2064_206488

def car_speed_problem (distance : ℝ) (ratio_A : ℕ) (ratio_B : ℕ) (time_minutes : ℝ) : Prop :=
  let x := distance / (ratio_A + ratio_B) * (60 / time_minutes)
  (ratio_B * x = 90)

theorem car_B_speed_90 
  (distance : ℝ := 88)
  (ratio_A : ℕ := 5)
  (ratio_B : ℕ := 6)
  (time_minutes : ℝ := 32)
  : car_speed_problem distance ratio_A ratio_B time_minutes :=
by
  sorry

end NUMINAMATH_GPT_car_B_speed_90_l2064_206488


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_l2064_206470

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 6) (hθ : θ = π / 3) (hz : z = -3) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, -3) :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_l2064_206470


namespace NUMINAMATH_GPT_exists_m_such_that_m_poly_is_zero_mod_p_l2064_206462

theorem exists_m_such_that_m_poly_is_zero_mod_p (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 7 = 1) :
  ∃ m : ℕ, m > 0 ∧ (m^3 + m^2 - 2*m - 1) % p = 0 := 
sorry

end NUMINAMATH_GPT_exists_m_such_that_m_poly_is_zero_mod_p_l2064_206462


namespace NUMINAMATH_GPT_john_mean_score_l2064_206467

-- Define John's quiz scores as a list
def johnQuizScores := [95, 88, 90, 92, 94, 89]

-- Define the function to calculate the mean of a list of integers
def mean_scores (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

-- Prove that the mean of John's quiz scores is 91.3333 
theorem john_mean_score :
  mean_scores johnQuizScores = 91.3333 := by
  -- sorry is a placeholder for the missing proof
  sorry

end NUMINAMATH_GPT_john_mean_score_l2064_206467


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l2064_206472

def set_M : Set ℝ := {x | 0 ≤ x ∧ x < 2}
def set_N : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def intersection_M_N : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = intersection_M_N := 
by sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l2064_206472


namespace NUMINAMATH_GPT_nth_inequality_l2064_206485

theorem nth_inequality (x : ℝ) (n : ℕ) (h_x_pos : 0 < x) : x + (n^n / x^n) ≥ n + 1 := 
sorry

end NUMINAMATH_GPT_nth_inequality_l2064_206485


namespace NUMINAMATH_GPT_largest_three_digit_base7_to_decimal_l2064_206418

theorem largest_three_digit_base7_to_decimal :
  (6 * 7^2 + 6 * 7^1 + 6 * 7^0) = 342 :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_base7_to_decimal_l2064_206418


namespace NUMINAMATH_GPT_positive_reals_condition_l2064_206465

theorem positive_reals_condition (a : ℝ) (h_pos : 0 < a) : a < 2 :=
by
  -- Problem conditions:
  -- There exists a positive integer n and n pairwise disjoint infinite sets A_i
  -- such that A_1 ∪ ... ∪ A_n = ℕ* and for any two numbers b > c in each A_i,
  -- b - c ≥ a^i.

  sorry

end NUMINAMATH_GPT_positive_reals_condition_l2064_206465


namespace NUMINAMATH_GPT_find_positive_integers_n_l2064_206428

open Real Int

noncomputable def satisfies_conditions (x y z : ℝ) (n : ℕ) : Prop :=
  sqrt x + sqrt y + sqrt z = 1 ∧ 
  (∃ m : ℤ, sqrt (x + n) + sqrt (y + n) + sqrt (z + n) = m)

theorem find_positive_integers_n (n : ℕ) :
  (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ satisfies_conditions x y z n) ↔
  (∃ k : ℤ, k ≥ 1 ∧ (k % 9 = 1 ∨ k % 9 = 8) ∧ n = (k^2 - 1) / 9) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integers_n_l2064_206428


namespace NUMINAMATH_GPT_gcd_determinant_l2064_206490

theorem gcd_determinant (a b : ℤ) (h : Int.gcd a b = 1) :
  Int.gcd (a + b) (a^2 + b^2 - a * b) = 1 ∨ Int.gcd (a + b) (a^2 + b^2 - a * b) = 3 :=
sorry

end NUMINAMATH_GPT_gcd_determinant_l2064_206490


namespace NUMINAMATH_GPT_find_value_l2064_206431

-- Define the theorem with the given conditions and the expected result
theorem find_value (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + a * c^2 + a + b + c = 2 * (a * b + b * c + a * c)) :
  c^2017 / (a^2016 + b^2018) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_value_l2064_206431


namespace NUMINAMATH_GPT_temperature_on_Monday_l2064_206432

theorem temperature_on_Monday 
  (M T W Th F : ℝ) 
  (h1 : (M + T + W + Th) / 4 = 48) 
  (h2 : (T + W + Th + F) / 4 = 46) 
  (h3 : F = 36) : 
  M = 44 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_temperature_on_Monday_l2064_206432


namespace NUMINAMATH_GPT_minimum_value_l2064_206409

theorem minimum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_condition : 2 * a + 3 * b = 1) :
  ∃ min_value : ℝ, min_value = 65 / 6 ∧ (∀ c d : ℝ, (0 < c) → (0 < d) → (2 * c + 3 * d = 1) → (1 / c + 1 / d ≥ min_value)) :=
sorry

end NUMINAMATH_GPT_minimum_value_l2064_206409


namespace NUMINAMATH_GPT_limit_expr_at_pi_l2064_206408

theorem limit_expr_at_pi :
  (Real.exp π - Real.exp x) / (Real.sin (5*x) - Real.sin (3*x)) = 1 / 2 * Real.exp π :=
by
  sorry

end NUMINAMATH_GPT_limit_expr_at_pi_l2064_206408


namespace NUMINAMATH_GPT_min_gennadies_l2064_206454

theorem min_gennadies 
  (n_Alexanders : ℕ) (n_Borises : ℕ) (n_Vasilies : ℕ) 
  (x_Gennadies : ℕ) 
  (h_Alexanders : n_Alexanders = 45) 
  (h_Borises   : n_Borises = 122) 
  (h_Vasilies  : n_Vasilies = 27) 
  (h_condition : ∀ p : ℕ, p = n_Borises - 1 → p = 121) 
  (h_total     : ∀ q : ℕ, q = 45 + 27 → q = 72)
  : x_Gennadies = 49 := 
sorry

end NUMINAMATH_GPT_min_gennadies_l2064_206454


namespace NUMINAMATH_GPT_largest_n_satisfying_inequality_l2064_206461

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, (1/4 + n/6 < 3/2) ∧ (∀ m : ℕ, m > n → 1/4 + m/6 ≥ 3/2) :=
by
  sorry

end NUMINAMATH_GPT_largest_n_satisfying_inequality_l2064_206461


namespace NUMINAMATH_GPT_remainder_division_l2064_206497

theorem remainder_division (x : ℝ) :
  (x ^ 2021 + 1) % (x ^ 12 - x ^ 9 + x ^ 6 - x ^ 3 + 1) = -x ^ 4 + 1 :=
sorry

end NUMINAMATH_GPT_remainder_division_l2064_206497


namespace NUMINAMATH_GPT_number_of_oranges_l2064_206464

def bananas : ℕ := 7
def apples : ℕ := 2 * bananas
def pears : ℕ := 4
def grapes : ℕ := apples / 2
def total_fruits : ℕ := 40

theorem number_of_oranges : total_fruits - (bananas + apples + pears + grapes) = 8 :=
by sorry

end NUMINAMATH_GPT_number_of_oranges_l2064_206464


namespace NUMINAMATH_GPT_compare_fractions_l2064_206460

theorem compare_fractions : (31 : ℚ) / 11 > (17 : ℚ) / 14 := 
by
  sorry

end NUMINAMATH_GPT_compare_fractions_l2064_206460


namespace NUMINAMATH_GPT_fraction_numerator_l2064_206495

theorem fraction_numerator (x : ℚ) : 
  (∃ y : ℚ, y = 4 * x + 4 ∧ x / y = 3 / 7) → x = -12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_numerator_l2064_206495


namespace NUMINAMATH_GPT_find_k_l2064_206480

-- Define the conditions
variables (k : ℝ) -- the variable k
variables (x1 : ℝ) -- x1 coordinate of point A on the graph y = k/x
variable (AREA_ABCD : ℝ := 10) -- the area of the quadrilateral ABCD

-- The statement to be proven
theorem find_k (k : ℝ) (h1 : ∀ x1 : ℝ, (0 < x1 ∧ 2 * abs k = AREA_ABCD → x1 * abs k * 2 = AREA_ABCD)) : k = -5 :=
sorry

end NUMINAMATH_GPT_find_k_l2064_206480


namespace NUMINAMATH_GPT_product_mod_32_is_15_l2064_206453

-- Define the odd primes less than 32
def oddPrimes : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the product of odd primes
def M : ℕ := List.foldl (· * ·) 1 oddPrimes

-- The theorem to prove the remainder when M is divided by 32 is 15
theorem product_mod_32_is_15 : M % 32 = 15 := by
  sorry

end NUMINAMATH_GPT_product_mod_32_is_15_l2064_206453


namespace NUMINAMATH_GPT_general_term_of_seq_l2064_206483

open Nat

noncomputable def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 3 * 2^n

theorem general_term_of_seq (a : ℕ → ℕ) :
  seq a → ∀ n, a n = (3 * n - 1) * 2^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_of_seq_l2064_206483


namespace NUMINAMATH_GPT_total_players_on_ground_l2064_206491

def cricket_players : ℕ := 15
def hockey_players : ℕ := 12
def football_players : ℕ := 13
def softball_players : ℕ := 15

theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + softball_players = 55 := 
by
  sorry

end NUMINAMATH_GPT_total_players_on_ground_l2064_206491


namespace NUMINAMATH_GPT_sample_size_correct_l2064_206433

-- Define the total number of students in a certain grade.
def total_students : ℕ := 500

-- Define the number of students selected for statistical analysis.
def selected_students : ℕ := 30

-- State the theorem to prove the selected students represent the sample size.
theorem sample_size_correct : selected_students = 30 := by
  -- The proof would go here, but we use sorry to indicate it is skipped.
  sorry

end NUMINAMATH_GPT_sample_size_correct_l2064_206433


namespace NUMINAMATH_GPT_evaluate_N_l2064_206423

theorem evaluate_N (N : ℕ) :
    988 + 990 + 992 + 994 + 996 = 5000 - N → N = 40 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_N_l2064_206423


namespace NUMINAMATH_GPT_find_width_of_sheet_of_paper_l2064_206401

def width_of_sheet_of_paper (W : ℝ) : Prop :=
  let margin := 1.5
  let length_of_paper := 10
  let area_covered := 38.5
  let width_of_picture := W - 2 * margin
  let length_of_picture := length_of_paper - 2 * margin
  width_of_picture * length_of_picture = area_covered

theorem find_width_of_sheet_of_paper : ∃ W : ℝ, width_of_sheet_of_paper W ∧ W = 8.5 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_width_of_sheet_of_paper_l2064_206401


namespace NUMINAMATH_GPT_line_through_midpoint_l2064_206440

theorem line_through_midpoint (x y : ℝ) (P : x = 2 ∧ y = -1) :
  (∃ l : ℝ, ∀ t : ℝ, 
  (1 + 5 * Real.cos t = x) ∧ (5 * Real.sin t = y) →
  (x - y = 3)) :=
by
  sorry

end NUMINAMATH_GPT_line_through_midpoint_l2064_206440


namespace NUMINAMATH_GPT_base_conversion_subtraction_l2064_206482

theorem base_conversion_subtraction :
  (4 * 6^4 + 3 * 6^3 + 2 * 6^2 + 1 * 6^1 + 0 * 6^0) - (3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0) = 4776 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_conversion_subtraction_l2064_206482


namespace NUMINAMATH_GPT_percentage_shaded_l2064_206463

theorem percentage_shaded (total_squares shaded_squares : ℕ) (h1 : total_squares = 5 * 5) (h2 : shaded_squares = 9) :
  (shaded_squares:ℚ) / total_squares * 100 = 36 :=
by
  sorry

end NUMINAMATH_GPT_percentage_shaded_l2064_206463


namespace NUMINAMATH_GPT_maximum_garden_area_l2064_206481

theorem maximum_garden_area (l w : ℝ) (h_perimeter : 2 * l + 2 * w = 400) : 
  l * w ≤ 10000 :=
by {
  -- proving the theorem
  sorry
}

end NUMINAMATH_GPT_maximum_garden_area_l2064_206481


namespace NUMINAMATH_GPT_root_iff_coeff_sum_zero_l2064_206478

theorem root_iff_coeff_sum_zero (a b c : ℝ) :
    (a * 1^2 + b * 1 + c = 0) ↔ (a + b + c = 0) := sorry

end NUMINAMATH_GPT_root_iff_coeff_sum_zero_l2064_206478


namespace NUMINAMATH_GPT_calculation_correct_l2064_206493

theorem calculation_correct :
  (Int.ceil ((15 : ℚ) / 8 * ((-35 : ℚ) / 4)) - 
  Int.floor (((15 : ℚ) / 8) * Int.floor ((-35 : ℚ) / 4 + (1 : ℚ) / 4))) = 1 := by
  sorry

end NUMINAMATH_GPT_calculation_correct_l2064_206493


namespace NUMINAMATH_GPT_shadow_problem_l2064_206426

-- Define the conditions
def cube_edge_length : ℝ := 2
def shadow_area_outside : ℝ := 147
def total_shadow_area : ℝ := shadow_area_outside + cube_edge_length^2

-- The main statement to prove
theorem shadow_problem :
  let x := 4 / (Real.sqrt total_shadow_area - cube_edge_length)
  (⌊1000 * x⌋ : ℤ) = 481 :=
by
  let x := 4 / (Real.sqrt total_shadow_area - cube_edge_length)
  have h : (⌊1000 * x⌋ : ℤ) = 481 := sorry
  exact h

end NUMINAMATH_GPT_shadow_problem_l2064_206426


namespace NUMINAMATH_GPT_count_students_with_green_eyes_l2064_206496

-- Definitions for the given conditions
def total_students := 50
def students_with_both := 10
def students_with_neither := 5

-- Let the number of students with green eyes be y
variable (y : ℕ) 

-- There are twice as many students with brown hair as with green eyes
def students_with_brown := 2 * y

-- There are y - 10 students with green eyes only
def students_with_green_only := y - students_with_both

-- There are 2y - 10 students with brown hair only
def students_with_brown_only := students_with_brown - students_with_both

-- Proof statement
theorem count_students_with_green_eyes (y : ℕ) 
  (h1 : (students_with_green_only) + (students_with_brown_only) + students_with_both + students_with_neither = total_students) : y = 15 := 
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_count_students_with_green_eyes_l2064_206496


namespace NUMINAMATH_GPT_last_digit_of_3_to_2010_is_9_l2064_206406

theorem last_digit_of_3_to_2010_is_9 : (3^2010 % 10) = 9 := by
  -- Given that the last digits of powers of 3 cycle through 3, 9, 7, 1
  -- We need to prove that the last digit of 3^2010 is 9
  sorry

end NUMINAMATH_GPT_last_digit_of_3_to_2010_is_9_l2064_206406


namespace NUMINAMATH_GPT_area_of_triangle_l2064_206438

theorem area_of_triangle (p : ℝ) (h_p : 0 < p ∧ p < 10) : 
    let C := (0, p)
    let O := (0, 0)
    let B := (10, 0)
    (1/2) * 10 * p = 5 * p := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_l2064_206438


namespace NUMINAMATH_GPT_value_of_b7b9_l2064_206479

-- Define arithmetic sequence and geometric sequence with given conditions
variable (a : ℕ → ℝ) (b : ℕ → ℝ)

-- The given conditions in the problem
def a_seq_arithmetic (a : ℕ → ℝ) := ∀ n, a n = a 1 + (n - 1) • (a 2 - a 1)
def b_seq_geometric (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n, b (n + 1) = r * b n
def given_condition (a : ℕ → ℝ) := 2 * a 5 - (a 8)^2 + 2 * a 11 = 0
def b8_eq_a8 (a b : ℕ → ℝ) := b 8 = a 8

-- The statement to prove
theorem value_of_b7b9 : a_seq_arithmetic a → b_seq_geometric b → given_condition a → b8_eq_a8 a b → b 7 * b 9 = 4 := by
  intros a_arith b_geom cond b8a8
  sorry

end NUMINAMATH_GPT_value_of_b7b9_l2064_206479


namespace NUMINAMATH_GPT_greatest_constant_triangle_l2064_206457

theorem greatest_constant_triangle (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  ∃ N : ℝ, (∀ (a b c : ℝ), a + b > c → b + c > a → c + a > b → (a^2 + b^2 + a * b) / c^2 > N) ∧ N = 3 / 4 :=
  sorry

end NUMINAMATH_GPT_greatest_constant_triangle_l2064_206457


namespace NUMINAMATH_GPT_smallest_total_squares_l2064_206442

theorem smallest_total_squares (n : ℕ) (h : 4 * n - 4 = 2 * n) : n^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_total_squares_l2064_206442


namespace NUMINAMATH_GPT_negate_existential_l2064_206413

theorem negate_existential (p : Prop) : (¬(∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0)) ↔ ∀ x : ℝ, x^2 - 2 * x + 2 > 0 :=
by sorry

end NUMINAMATH_GPT_negate_existential_l2064_206413


namespace NUMINAMATH_GPT_total_selling_price_correct_l2064_206424

-- Defining the given conditions
def profit_per_meter : ℕ := 5
def cost_price_per_meter : ℕ := 100
def total_meters_sold : ℕ := 85

-- Using the conditions to define the total selling price
def total_selling_price := total_meters_sold * (cost_price_per_meter + profit_per_meter)

-- Stating the theorem without the proof
theorem total_selling_price_correct : total_selling_price = 8925 := by
  sorry

end NUMINAMATH_GPT_total_selling_price_correct_l2064_206424


namespace NUMINAMATH_GPT_tan_alpha_eq_two_l2064_206407

theorem tan_alpha_eq_two (α : ℝ) (h1 : α ∈ Set.Ioc 0 (Real.pi / 2))
    (h2 : Real.sin ((Real.pi / 4) - α) * Real.sin ((Real.pi / 4) + α) = -3 / 10) :
    Real.tan α = 2 := by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_two_l2064_206407


namespace NUMINAMATH_GPT_product_of_radii_l2064_206410

-- Definitions based on the problem conditions
def passes_through (a : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - a)^2 + (C.2 - a)^2 = a^2

def tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def circle_radii_roots (a b : ℝ) : Prop :=
  a^2 - 14 * a + 25 = 0 ∧ b^2 - 14 * b + 25 = 0

-- Theorem statement to prove the product of the radii
theorem product_of_radii (a r1 r2 : ℝ) (h1 : passes_through a (3, 4)) (h2 : tangent_to_axes a) (h3 : circle_radii_roots r1 r2) : r1 * r2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_product_of_radii_l2064_206410


namespace NUMINAMATH_GPT_monotonicity_and_extrema_l2064_206427

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

theorem monotonicity_and_extrema :
  (∀ x, -3 / 2 < x ∧ x < -1 → f x < f (x + 0.0001)) ∧
  (∀ x, -1 < x ∧ x < -1 / 2 → f x > f (x + 0.0001)) ∧
  (∀ x, -1 / 2 < x ∧ x < (Real.exp 2 - 3) / 2 → f x < f (x + 0.0001)) ∧
  ∀ x, x ∈ Set.Icc (-1 : ℝ) ((Real.exp 2 - 3) / 2) →
     (f (x) ≥ Real.log 2 + 1 / 4 → x = -1 / 2) ∧
     (f (x) ≤ 2 + (Real.exp 2 - 3)^2 / 4 → x = (Real.exp 2 - 3) / 2) :=
sorry

end NUMINAMATH_GPT_monotonicity_and_extrema_l2064_206427


namespace NUMINAMATH_GPT_total_cost_of_fruits_l2064_206451

theorem total_cost_of_fruits (h_orange_weight : 12 * 2 = 24)
                             (h_apple_weight : 8 * 3.75 = 30)
                             (price_orange : ℝ := 1.5)
                             (price_apple : ℝ := 2.0) :
  (5 * 2 * price_orange + 4 * 3.75 * price_apple) = 45 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_fruits_l2064_206451


namespace NUMINAMATH_GPT_find_q_l2064_206411

def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h: d = 3) (h1: -p / 3 = -d) (h2: -p / 3 = 1 + p + q + d) : q = -16 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l2064_206411


namespace NUMINAMATH_GPT_total_days_in_month_eq_l2064_206420

-- Definition of the conditions
def took_capsules_days : ℕ := 27
def forgot_capsules_days : ℕ := 4

-- The statement to be proved
theorem total_days_in_month_eq : took_capsules_days + forgot_capsules_days = 31 := by
  sorry

end NUMINAMATH_GPT_total_days_in_month_eq_l2064_206420


namespace NUMINAMATH_GPT_smallest_common_multiple_gt_50_l2064_206414

theorem smallest_common_multiple_gt_50 (a b : ℕ) (h1 : a = 15) (h2 : b = 20) : 
    ∃ x, x > 50 ∧ Nat.lcm a b = x := by
  have h_lcm : Nat.lcm a b = 60 := by sorry
  use 60
  exact ⟨by decide, h_lcm⟩

end NUMINAMATH_GPT_smallest_common_multiple_gt_50_l2064_206414


namespace NUMINAMATH_GPT_trig_identity_l2064_206469

theorem trig_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l2064_206469


namespace NUMINAMATH_GPT_find_a_for_cubic_sum_l2064_206436

theorem find_a_for_cubic_sum (a : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - a * x1 + a + 2 = 0 ∧ 
    x2^2 - a * x2 + a + 2 = 0 ∧
    x1 + x2 = a ∧
    x1 * x2 = a + 2 ∧
    x1^3 + x2^3 = -8) ↔ a = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_for_cubic_sum_l2064_206436


namespace NUMINAMATH_GPT_smallest_y_for_perfect_fourth_power_l2064_206489

-- Define the conditions
def x : ℕ := 7 * 24 * 48
def y : ℕ := 6174

-- The theorem we need to prove
theorem smallest_y_for_perfect_fourth_power (x y : ℕ) 
  (hx : x = 7 * 24 * 48) 
  (hy : y = 6174) : ∃ k : ℕ, (∃ z : ℕ, z * z * z * z = x * y) :=
sorry

end NUMINAMATH_GPT_smallest_y_for_perfect_fourth_power_l2064_206489


namespace NUMINAMATH_GPT_citizen_income_l2064_206405

noncomputable def income (I : ℝ) : Prop :=
  let P := 0.11 * 40000
  let A := I - 40000
  P + 0.20 * A = 8000

theorem citizen_income (I : ℝ) (h : income I) : I = 58000 := 
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_citizen_income_l2064_206405
