import Mathlib

namespace NUMINAMATH_GPT_emma_missing_coins_l1743_174338

theorem emma_missing_coins (x : ℤ) (h₁ : x > 0) :
  let lost := (1 / 3 : ℚ) * x
  let found := (2 / 3 : ℚ) * lost
  let remaining := x - lost + found
  let missing := x - remaining
  missing / x = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_emma_missing_coins_l1743_174338


namespace NUMINAMATH_GPT_sally_baseball_cards_l1743_174309

theorem sally_baseball_cards (initial_cards sold_cards : ℕ) (h1 : initial_cards = 39) (h2 : sold_cards = 24) :
  (initial_cards - sold_cards = 15) :=
by
  -- Proof needed
  sorry

end NUMINAMATH_GPT_sally_baseball_cards_l1743_174309


namespace NUMINAMATH_GPT_max_value_of_6_f_x_plus_2012_l1743_174336

noncomputable def f (x : ℝ) : ℝ :=
  min (min (4*x + 1) (x + 2)) (-2*x + 4)

theorem max_value_of_6_f_x_plus_2012 : ∃ x : ℝ, 6 * f x + 2012 = 2028 :=
sorry

end NUMINAMATH_GPT_max_value_of_6_f_x_plus_2012_l1743_174336


namespace NUMINAMATH_GPT_calories_per_serving_l1743_174345

theorem calories_per_serving (x : ℕ) (total_calories bread_calories servings : ℕ)
    (h1: total_calories = 500) (h2: bread_calories = 100) (h3: servings = 2)
    (h4: total_calories = bread_calories + (servings * x)) :
    x = 200 :=
by
  sorry

end NUMINAMATH_GPT_calories_per_serving_l1743_174345


namespace NUMINAMATH_GPT_value_2x_y_l1743_174304

theorem value_2x_y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y + 5 = 0) : 2*x + y = 0 := 
by
  sorry

end NUMINAMATH_GPT_value_2x_y_l1743_174304


namespace NUMINAMATH_GPT_andrew_total_hours_l1743_174371

theorem andrew_total_hours (days_worked : ℕ) (hours_per_day : ℝ)
    (h1 : days_worked = 3) (h2 : hours_per_day = 2.5) : 
    days_worked * hours_per_day = 7.5 := by
  sorry

end NUMINAMATH_GPT_andrew_total_hours_l1743_174371


namespace NUMINAMATH_GPT_handshake_problem_l1743_174352

theorem handshake_problem (n : ℕ) (h : n * (n - 1) / 2 = 1770) : n = 60 :=
sorry

end NUMINAMATH_GPT_handshake_problem_l1743_174352


namespace NUMINAMATH_GPT_problem_l1743_174381

def pair_eq (a b c d : ℝ) : Prop := (a = c) ∧ (b = d)

def op_a (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, b * c - a * d)
def op_o (a b c d : ℝ) : ℝ × ℝ := (a + c, b + d)

theorem problem (x y : ℝ) :
  op_a 3 4 x y = (11, -2) →
  op_o 3 4 x y = (4, 6) :=
sorry

end NUMINAMATH_GPT_problem_l1743_174381


namespace NUMINAMATH_GPT_compute_fraction_equation_l1743_174354

theorem compute_fraction_equation :
  (8 * (2 / 3: ℚ)^4 + 2 = 290 / 81) :=
sorry

end NUMINAMATH_GPT_compute_fraction_equation_l1743_174354


namespace NUMINAMATH_GPT_sqrt_sum_eq_five_l1743_174314

theorem sqrt_sum_eq_five
  (x : ℝ)
  (h1 : -Real.sqrt 15 ≤ x ∧ x ≤ Real.sqrt 15)
  (h2 : Real.sqrt (25 - x^2) - Real.sqrt (15 - x^2) = 2) :
  Real.sqrt (25 - x^2) + Real.sqrt (15 - x^2) = 5 := by
  sorry

end NUMINAMATH_GPT_sqrt_sum_eq_five_l1743_174314


namespace NUMINAMATH_GPT_min_value_expression_l1743_174307

noncomputable def f (t : ℝ) : ℝ :=
  (1 / (t + 1)) + (2 * t / (2 * t + 1))

theorem min_value_expression (x y : ℝ) (h : x * y > 0) :
  ∃ t, (x / y = t) ∧ t > 0 ∧ f t = 4 - 2 * Real.sqrt 2 := 
  sorry

end NUMINAMATH_GPT_min_value_expression_l1743_174307


namespace NUMINAMATH_GPT_g_eval_l1743_174379

-- Define the function g
def g (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := (2 * a + b) / (c - a)

-- Theorem to prove g(2, 4, -1) = -8 / 3
theorem g_eval :
  g 2 4 (-1) = -8 / 3 := 
by
  sorry

end NUMINAMATH_GPT_g_eval_l1743_174379


namespace NUMINAMATH_GPT_sum_of_coefficients_l1743_174331

theorem sum_of_coefficients 
  (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℕ)
  (h : (3 * x - 1) ^ 10 = a + a_1 * x + a_2 * x ^ 2 + a_3 * x ^ 3 + a_4 * x ^ 4 + a_5 * x ^ 5 + a_6 * x ^ 6 + a_7 * x ^ 7 + a_8 * x ^ 8 + a_9 * x ^ 9 + a_10 * x ^ 10) :
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1023 := 
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1743_174331


namespace NUMINAMATH_GPT_no_square_sum_l1743_174389

theorem no_square_sum (x y : ℕ) (hxy_pos : 0 < x ∧ 0 < y)
  (hxy_gcd : Nat.gcd x y = 1)
  (hxy_perf : ∃ k : ℕ, x + 3 * y^2 = k^2) : ¬ ∃ z : ℕ, x^2 + 9 * y^4 = z^2 :=
by
  sorry

end NUMINAMATH_GPT_no_square_sum_l1743_174389


namespace NUMINAMATH_GPT_total_cost_correct_l1743_174313

def bun_price : ℝ := 0.1
def buns_count : ℝ := 10
def milk_price : ℝ := 2
def milk_count : ℝ := 2
def egg_price : ℝ := 3 * milk_price

def total_cost : ℝ := (buns_count * bun_price) + (milk_count * milk_price) + egg_price

theorem total_cost_correct : total_cost = 11 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l1743_174313


namespace NUMINAMATH_GPT_product_equals_permutation_l1743_174372

-- Definitions and conditions
def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Given product sequence
def product_seq (n k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldr (λ x y => x * y) 1

-- Problem statement: The product of numbers from 18 to 9 is equivalent to A_{18}^{10}
theorem product_equals_permutation :
  product_seq 18 10 = perm 18 10 :=
by
  sorry

end NUMINAMATH_GPT_product_equals_permutation_l1743_174372


namespace NUMINAMATH_GPT_percent_decrease_l1743_174329

def original_price : ℝ := 100
def sale_price : ℝ := 60

theorem percent_decrease : (original_price - sale_price) / original_price * 100 = 40 := by
  sorry

end NUMINAMATH_GPT_percent_decrease_l1743_174329


namespace NUMINAMATH_GPT_john_probability_l1743_174308

/-- John arrives at a terminal which has sixteen gates arranged in a straight line with exactly 50 feet between adjacent gates. His departure gate is assigned randomly. After waiting at that gate, John is informed that the departure gate has been changed to another gate, chosen randomly again. Prove that the probability that John walks 200 feet or less to the new gate is \(\frac{4}{15}\), and find \(4 + 15 = 19\) -/
theorem john_probability :
  let n_gates := 16
  let dist_between_gates := 50
  let max_walk_dist := 200
  let total_possibilities := n_gates * (n_gates - 1)
  let valid_cases :=
    4 * (2 + 2 * (4 - 1))
  let probability_within_200_feet := valid_cases / total_possibilities
  let fraction := probability_within_200_feet * (15 / 4)
  fraction = 1 → 4 + 15 = 19 := by
  sorry -- Proof goes here 

end NUMINAMATH_GPT_john_probability_l1743_174308


namespace NUMINAMATH_GPT_initial_customers_correct_l1743_174375

def initial_customers (remaining : ℕ) (left : ℕ) : ℕ := remaining + left

theorem initial_customers_correct :
  initial_customers 12 9 = 21 :=
by
  sorry

end NUMINAMATH_GPT_initial_customers_correct_l1743_174375


namespace NUMINAMATH_GPT_value_of_f_at_2_l1743_174384

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem value_of_f_at_2 : f 2 = 3 := by
  -- Definition of the function f.
  -- The goal is to prove that f(2) = 3.
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l1743_174384


namespace NUMINAMATH_GPT_proof_problem_l1743_174358

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

-- The two conditions
def condition1 (x y : ℝ) : Prop := f x + f y ≤ 0
def condition2 (x y : ℝ) : Prop := f x - f y ≥ 0

-- Equivalent description
def circle_condition (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 ≤ 8
def region1 (x y : ℝ) : Prop := y ≤ x ∧ y ≥ 6 - x
def region2 (x y : ℝ) : Prop := y ≥ x ∧ y ≤ 6 - x

-- The proof statement
theorem proof_problem (x y : ℝ) :
  (condition1 x y ∧ condition2 x y) ↔ 
  (circle_condition x y ∧ (region1 x y ∨ region2 x y)) :=
sorry

end NUMINAMATH_GPT_proof_problem_l1743_174358


namespace NUMINAMATH_GPT_probability_correct_l1743_174321

-- Define the problem conditions.
def num_balls : ℕ := 8
def possible_colors : ℕ := 2

-- Probability calculation for a specific arrangement (either configuration of colors).
def probability_per_arrangement : ℚ := (1/2) ^ num_balls

-- Number of favorable arrangements with 4 black and 4 white balls.
def favorable_arrangements : ℕ := Nat.choose num_balls 4

-- The required probability for the solution.
def desired_probability : ℚ := favorable_arrangements * probability_per_arrangement

-- The proof statement to be provided.
theorem probability_correct :
  desired_probability = 35 / 128 := 
by
  sorry

end NUMINAMATH_GPT_probability_correct_l1743_174321


namespace NUMINAMATH_GPT_number_composition_l1743_174355

theorem number_composition :
  5 * 100000 + 6 * 100 + 3 * 10 + 6 * 0.01 = 500630.06 := 
by 
  sorry

end NUMINAMATH_GPT_number_composition_l1743_174355


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1743_174302

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, x^2 - 2 * a * x + a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1743_174302


namespace NUMINAMATH_GPT_train_length_l1743_174369

noncomputable def length_of_train (time_sec : ℕ) (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000 / 3600) * time_sec

theorem train_length (h_time : 21 = 21) (h_speed : 75.6 = 75.6) :
  length_of_train 21 75.6 = 441 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l1743_174369


namespace NUMINAMATH_GPT_expression_value_l1743_174346

theorem expression_value (x : ℝ) (h : x = 3) : x^4 - 4 * x^2 = 45 := by
  sorry

end NUMINAMATH_GPT_expression_value_l1743_174346


namespace NUMINAMATH_GPT_trig_equation_solution_l1743_174356

open Real

theorem trig_equation_solution (x : ℝ) (k n : ℤ) :
  (sin (2 * x)) ^ 4 + (sin (2 * x)) ^ 3 * (cos (2 * x)) -
  8 * (sin (2 * x)) * (cos (2 * x)) ^ 3 - 8 * (cos (2 * x)) ^ 4 = 0 ↔
  (∃ k : ℤ, x = -π / 8 + (π * k) / 2) ∨ 
  (∃ n : ℤ, x = (1 / 2) * arctan 2 + (π * n) / 2) := sorry

end NUMINAMATH_GPT_trig_equation_solution_l1743_174356


namespace NUMINAMATH_GPT_ratio_of_areas_l1743_174366

theorem ratio_of_areas (R_A R_B : ℝ) 
  (h1 : (1 / 6) * 2 * Real.pi * R_A = (1 / 9) * 2 * Real.pi * R_B) :
  (Real.pi * R_A^2) / (Real.pi * R_B^2) = (4 : ℝ) / 9 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1743_174366


namespace NUMINAMATH_GPT_man_speed_with_the_stream_l1743_174391

def speed_with_the_stream (V_m V_s : ℝ) : Prop :=
  V_m + V_s = 2

theorem man_speed_with_the_stream (V_m V_s : ℝ) (h1 : V_m - V_s = 2) (h2 : V_m = 2) : speed_with_the_stream V_m V_s :=
by
  sorry

end NUMINAMATH_GPT_man_speed_with_the_stream_l1743_174391


namespace NUMINAMATH_GPT_avg_eggs_per_nest_l1743_174378

/-- In the Caribbean, loggerhead turtles lay three million eggs in twenty thousand nests. 
On average, show that there are 150 eggs in each nest. -/

theorem avg_eggs_per_nest 
  (total_eggs : ℕ) 
  (total_nests : ℕ) 
  (h1 : total_eggs = 3000000) 
  (h2 : total_nests = 20000) :
  total_eggs / total_nests = 150 := 
by {
  sorry
}

end NUMINAMATH_GPT_avg_eggs_per_nest_l1743_174378


namespace NUMINAMATH_GPT_balls_in_boxes_l1743_174395

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), (balls = 6) → (boxes = 3) → 
  (∃ ways : ℕ, ways = 7) :=
by
  sorry

end NUMINAMATH_GPT_balls_in_boxes_l1743_174395


namespace NUMINAMATH_GPT_symmetric_to_origin_l1743_174359

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_to_origin (p : ℝ × ℝ) (h : p = (3, -1)) : symmetric_point p = (-3, 1) :=
by
  -- This is just the statement; the proof is not provided.
  sorry

end NUMINAMATH_GPT_symmetric_to_origin_l1743_174359


namespace NUMINAMATH_GPT_f_sum_2018_2019_l1743_174398

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom even_shifted_function (x : ℝ) : f (x + 1) = f (-x + 1)
axiom f_neg1 : f (-1) = -1

theorem f_sum_2018_2019 : f 2018 + f 2019 = -1 :=
by sorry

end NUMINAMATH_GPT_f_sum_2018_2019_l1743_174398


namespace NUMINAMATH_GPT_no_integer_solutions_l1743_174351

theorem no_integer_solutions (m n : ℤ) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2011) :=
by sorry

end NUMINAMATH_GPT_no_integer_solutions_l1743_174351


namespace NUMINAMATH_GPT_largest_n_in_base10_l1743_174361

-- Definitions corresponding to the problem conditions
def n_eq_base8_expr (A B C : ℕ) : ℕ := 64 * A + 8 * B + C
def n_eq_base12_expr (A B C : ℕ) : ℕ := 144 * C + 12 * B + A

-- Problem statement translated into Lean
theorem largest_n_in_base10 (n A B C : ℕ) (h1 : n = n_eq_base8_expr A B C) 
    (h2 : n = n_eq_base12_expr A B C) (hA : A < 8) (hB : B < 8) (hC : C < 12) (h_pos: n > 0) : 
    n ≤ 509 :=
sorry

end NUMINAMATH_GPT_largest_n_in_base10_l1743_174361


namespace NUMINAMATH_GPT_fraction_of_walls_not_illuminated_l1743_174363

-- Define given conditions
def point_light_source : Prop := true
def rectangular_room : Prop := true
def flat_mirror_on_wall : Prop := true
def full_height_of_room : Prop := true

-- Define the fraction not illuminated
def fraction_not_illuminated := 17 / 32

-- State the theorem to prove
theorem fraction_of_walls_not_illuminated :
  point_light_source ∧ rectangular_room ∧ flat_mirror_on_wall ∧ full_height_of_room →
  fraction_not_illuminated = 17 / 32 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_fraction_of_walls_not_illuminated_l1743_174363


namespace NUMINAMATH_GPT_sum_of_ages_is_14_l1743_174347

/-- Kiana has two older twin brothers and the product of their three ages is 72.
    Prove that the sum of their three ages is 14. -/
theorem sum_of_ages_is_14 (kiana_age twin_age : ℕ) (htwins : twin_age > kiana_age) (h_product : kiana_age * twin_age * twin_age = 72) :
  kiana_age + twin_age + twin_age = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_is_14_l1743_174347


namespace NUMINAMATH_GPT_num_valid_sequences_10_transformations_l1743_174317

/-- Define the transformations: 
    L: 90° counterclockwise rotation,
    R: 90° clockwise rotation,
    H: reflection across the x-axis,
    V: reflection across the y-axis. -/
inductive Transformation
| L | R | H | V

/-- Define a function to get the number of valid sequences of transformations
    that bring the vertices E, F, G, H back to their original positions.-/
def countValidSequences : ℕ :=
  56

/-- The theorem to prove that the number of valid sequences
    of 10 transformations resulting in the identity transformation is 56. -/
theorem num_valid_sequences_10_transformations : 
  countValidSequences = 56 :=
sorry

end NUMINAMATH_GPT_num_valid_sequences_10_transformations_l1743_174317


namespace NUMINAMATH_GPT_contrapositive_honor_roll_l1743_174301

variable (Student : Type) (scores_hundred : Student → Prop) (honor_roll_qualifies : Student → Prop)

theorem contrapositive_honor_roll (s : Student) :
  (¬ honor_roll_qualifies s) → (¬ scores_hundred s) := 
sorry

end NUMINAMATH_GPT_contrapositive_honor_roll_l1743_174301


namespace NUMINAMATH_GPT_cricketer_new_average_l1743_174337

variable (A : ℕ) (runs_19th_inning : ℕ) (avg_increase : ℕ)
variable (total_runs_after_18 : ℕ)

theorem cricketer_new_average
  (h1 : runs_19th_inning = 98)
  (h2 : avg_increase = 4)
  (h3 : total_runs_after_18 = 18 * A)
  (h4 : 18 * A + 98 = 19 * (A + 4)) :
  A + 4 = 26 :=
by sorry

end NUMINAMATH_GPT_cricketer_new_average_l1743_174337


namespace NUMINAMATH_GPT_ellipse_major_minor_axis_l1743_174334

theorem ellipse_major_minor_axis (m : ℝ) :
  (∀ x y : ℝ, x^2 + m*y^2 = 1) ∧
  (∃ a b : ℝ, a = 2 * b ∧ b^2 = 1 ∧ a^2 = 1/m) →
  m = 1/4 :=
by {
  sorry
}

end NUMINAMATH_GPT_ellipse_major_minor_axis_l1743_174334


namespace NUMINAMATH_GPT_expansion_coefficient_l1743_174397

theorem expansion_coefficient :
  ∀ (x : ℝ), (∃ (a₀ a₁ a₂ b : ℝ), x^6 + x^4 = a₀ + a₁ * (x + 2) + a₂ * (x + 2)^2 + b * (x + 2)^3) →
  (a₀ = 0 ∧ a₁ = 0 ∧ a₂ = 0 ∧ b = -168) :=
by
  sorry

end NUMINAMATH_GPT_expansion_coefficient_l1743_174397


namespace NUMINAMATH_GPT_value_of_x_in_terms_of_z_l1743_174370

variable {z : ℝ} {x y : ℝ}
  
theorem value_of_x_in_terms_of_z (h1 : y = z + 50) (h2 : x = 0.70 * y) : x = 0.70 * z + 35 := 
  sorry

end NUMINAMATH_GPT_value_of_x_in_terms_of_z_l1743_174370


namespace NUMINAMATH_GPT_graph_does_not_pass_through_fourth_quadrant_l1743_174322

def linear_function (x : ℝ) : ℝ := x + 1

theorem graph_does_not_pass_through_fourth_quadrant : 
  ¬ ∃ x : ℝ, x > 0 ∧ linear_function x < 0 :=
sorry

end NUMINAMATH_GPT_graph_does_not_pass_through_fourth_quadrant_l1743_174322


namespace NUMINAMATH_GPT_glass_sphere_wall_thickness_l1743_174350

/-- Mathematically equivalent proof problem statement:
Given a hollow glass sphere with outer diameter 16 cm such that 3/8 of its surface remains dry,
and specific gravity of glass s = 2.523. The wall thickness of the sphere is equal to 0.8 cm. -/
theorem glass_sphere_wall_thickness 
  (outer_diameter : ℝ) (dry_surface_fraction : ℝ) (specific_gravity : ℝ) (required_thickness : ℝ) 
  (uniform_thickness : outer_diameter = 16)
  (dry_surface : dry_surface_fraction = 3 / 8)
  (s : specific_gravity = 2.523) :
  required_thickness = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_glass_sphere_wall_thickness_l1743_174350


namespace NUMINAMATH_GPT_two_digit_number_with_tens_5_l1743_174392

-- Definitions and conditions
variable (A : Nat)

-- Problem statement as a Lean theorem
theorem two_digit_number_with_tens_5 (hA : A < 10) : (10 * 5 + A) = 50 + A := by
  sorry

end NUMINAMATH_GPT_two_digit_number_with_tens_5_l1743_174392


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l1743_174316

theorem arithmetic_sequence_a5
  (a : ℕ → ℤ) -- a is the arithmetic sequence function
  (S : ℕ → ℤ) -- S is the sum of the first n terms of the sequence
  (h1 : S 5 = 2 * S 4) -- Condition S_5 = 2S_4
  (h2 : a 2 + a 4 = 8) -- Condition a_2 + a_4 = 8
  (hS : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) -- Definition of S_n
  (ha : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) -- Definition of a_n
: a 5 = 10 := 
by
  -- proof
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l1743_174316


namespace NUMINAMATH_GPT_different_rhetorical_device_in_optionA_l1743_174330

def optionA_uses_metaphor : Prop :=
  -- Here, define the condition explaining that Option A uses metaphor
  true -- This will denote that Option A uses metaphor 

def optionsBCD_use_personification : Prop :=
  -- Here, define the condition explaining that Options B, C, and D use personification
  true -- This will denote that Options B, C, and D use personification

theorem different_rhetorical_device_in_optionA :
  optionA_uses_metaphor ∧ optionsBCD_use_personification → 
  (∃ (A P : Prop), A ≠ P) :=
by
  -- No proof is required as per instructions
  intro h
  exact Exists.intro optionA_uses_metaphor (Exists.intro optionsBCD_use_personification sorry)

end NUMINAMATH_GPT_different_rhetorical_device_in_optionA_l1743_174330


namespace NUMINAMATH_GPT_intersection_y_sum_zero_l1743_174303

theorem intersection_y_sum_zero :
  ∀ (x1 y1 x2 y2 : ℝ), (y1 = 2 * x1) ∧ (y1 = 2 / x1) ∧ (y2 = 2 * x2) ∧ (y2 = 2 / x2) →
  (x2 = -x1) ∧ (y2 = -y1) →
  y1 + y2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_intersection_y_sum_zero_l1743_174303


namespace NUMINAMATH_GPT_problem_statement_l1743_174368

-- Definitions for given conditions
variables (a b m n x : ℤ)

-- Assuming conditions: a = -b, mn = 1, and |x| = 2
axiom opp_num : a = -b
axiom recip : m * n = 1
axiom abs_x : |x| = 2

-- Problem statement to prove
theorem problem_statement :
  -2 * m * n + (a + b) / 2023 + x * x = 2 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1743_174368


namespace NUMINAMATH_GPT_largest_is_B_l1743_174365

noncomputable def A : ℚ := ((2023:ℚ) / 2022) + ((2023:ℚ) / 2024)
noncomputable def B : ℚ := ((2024:ℚ) / 2023) + ((2026:ℚ) / 2023)
noncomputable def C : ℚ := ((2025:ℚ) / 2024) + ((2025:ℚ) / 2026)

theorem largest_is_B : B > A ∧ B > C := by
  sorry

end NUMINAMATH_GPT_largest_is_B_l1743_174365


namespace NUMINAMATH_GPT_laura_saves_more_with_promotion_A_l1743_174380

def promotion_A_cost (pair_price : ℕ) : ℕ :=
  let second_pair_price := pair_price / 2
  pair_price + second_pair_price

def promotion_B_cost (pair_price : ℕ) : ℕ :=
  let discount := pair_price * 20 / 100
  pair_price + (pair_price - discount)

def savings (pair_price : ℕ) : ℕ :=
  promotion_B_cost pair_price - promotion_A_cost pair_price

theorem laura_saves_more_with_promotion_A :
  savings 50 = 15 :=
  by
  -- The detailed proof will be added here
  sorry

end NUMINAMATH_GPT_laura_saves_more_with_promotion_A_l1743_174380


namespace NUMINAMATH_GPT_difference_divisible_by_18_l1743_174348

theorem difference_divisible_by_18 (a b : ℤ) : 18 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_difference_divisible_by_18_l1743_174348


namespace NUMINAMATH_GPT_find_a4_l1743_174341

noncomputable def a (n : ℕ) : ℕ := sorry -- Define the arithmetic sequence
def S (n : ℕ) : ℕ := sorry -- Define the sum function for the sequence

theorem find_a4 (h1 : S 5 = 25) (h2 : a 2 = 3) : a 4 = 7 := by
  sorry

end NUMINAMATH_GPT_find_a4_l1743_174341


namespace NUMINAMATH_GPT_mass_percentage_Ca_in_CaI2_l1743_174306

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I

theorem mass_percentage_Ca_in_CaI2 :
  (molar_mass_Ca / molar_mass_CaI2) * 100 = 13.63 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_Ca_in_CaI2_l1743_174306


namespace NUMINAMATH_GPT_toys_per_hour_computation_l1743_174339

noncomputable def total_toys : ℕ := 20500
noncomputable def monday_hours : ℕ := 8
noncomputable def tuesday_hours : ℕ := 7
noncomputable def wednesday_hours : ℕ := 9
noncomputable def thursday_hours : ℕ := 6

noncomputable def total_hours_worked : ℕ := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
noncomputable def toys_produced_each_hour : ℚ := total_toys / total_hours_worked

theorem toys_per_hour_computation :
  toys_produced_each_hour = 20500 / (8 + 7 + 9 + 6) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_toys_per_hour_computation_l1743_174339


namespace NUMINAMATH_GPT_integer_sum_of_squares_power_l1743_174324

theorem integer_sum_of_squares_power (a p q : ℤ) (k : ℕ) (h : a = p^2 + q^2) : 
  ∃ c d : ℤ, a^k = c^2 + d^2 := 
sorry

end NUMINAMATH_GPT_integer_sum_of_squares_power_l1743_174324


namespace NUMINAMATH_GPT_volume_ratio_of_cubes_l1743_174373

theorem volume_ratio_of_cubes :
  (4^3 / 10^3 : ℚ) = 8 / 125 := by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_cubes_l1743_174373


namespace NUMINAMATH_GPT_sibling_age_difference_l1743_174394

theorem sibling_age_difference 
  (x : ℕ) 
  (h : 3 * x + 2 * x + 1 * x = 90) : 
  3 * x - x = 30 := 
by 
  sorry

end NUMINAMATH_GPT_sibling_age_difference_l1743_174394


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1743_174399

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) : 1 / x + 1 / y = 8 / 75 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1743_174399


namespace NUMINAMATH_GPT_solve_for_x_l1743_174340

theorem solve_for_x : ∃ x : ℝ, (2010 + x)^3 = -x^3 ∧ x = -1005 := 
by
  use -1005
  sorry

end NUMINAMATH_GPT_solve_for_x_l1743_174340


namespace NUMINAMATH_GPT_binary_to_decimal_1010101_l1743_174376

def bin_to_dec (bin : List ℕ) (len : ℕ): ℕ :=
  List.foldl (λ acc (digit, idx) => acc + digit * 2^idx) 0 (List.zip bin (List.range len))

theorem binary_to_decimal_1010101 : bin_to_dec [1, 0, 1, 0, 1, 0, 1] 7 = 85 :=
by
  simp [bin_to_dec, List.range, List.zip]
  -- Detailed computation can be omitted and sorry used here if necessary
  sorry

end NUMINAMATH_GPT_binary_to_decimal_1010101_l1743_174376


namespace NUMINAMATH_GPT_geometric_sequence_a_11_l1743_174305

-- Define the geometric sequence with given terms
variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def is_geometric_sequence := ∀ n : ℕ, a (n + 1) = a n * q

axiom a_5 : a 5 = -16
axiom a_8 : a 8 = 8

-- Question to prove
theorem geometric_sequence_a_11 (h : is_geometric_sequence a q) : a 11 = -4 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a_11_l1743_174305


namespace NUMINAMATH_GPT_range_of_a_l1743_174386

theorem range_of_a (a : ℝ) :
  (¬ ( ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0 ) 
    ∨ 
   ¬ ( ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0 )) 
→ a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1743_174386


namespace NUMINAMATH_GPT_composite_probability_l1743_174360

/--
Given that a number selected at random from the first 50 natural numbers,
where 1 is neither prime nor composite,
the probability of selecting a composite number is 34/49.
-/
theorem composite_probability :
  let total_numbers := 50
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  let num_primes := primes.length
  let num_composites := total_numbers - num_primes - 1
  let probability_composite := (num_composites : ℚ) / (total_numbers - 1)
  probability_composite = 34 / 49 :=
by {
  sorry
}

end NUMINAMATH_GPT_composite_probability_l1743_174360


namespace NUMINAMATH_GPT_gcd_max_digits_l1743_174383

theorem gcd_max_digits (a b : ℕ) (h_a : a < 10^7) (h_b : b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) : Nat.gcd a b < 10^3 :=
by
  sorry

end NUMINAMATH_GPT_gcd_max_digits_l1743_174383


namespace NUMINAMATH_GPT_unique_three_digit_numbers_unique_three_digit_odd_numbers_l1743_174332

def num_digits: ℕ := 10

theorem unique_three_digit_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 648 ∧ n = (num_digits - 1) * (num_digits - 1) * (num_digits - 2) + 2 * (num_digits - 1) * (num_digits - 1) :=
  sorry

theorem unique_three_digit_odd_numbers (num_digits : ℕ) :
  ∃ n : ℕ, n = 320 ∧ ∀ odd_digit_nums : ℕ, odd_digit_nums ≥ 1 → odd_digit_nums = 5 → 
  n = odd_digit_nums * (num_digits - 2) * (num_digits - 2) :=
  sorry

end NUMINAMATH_GPT_unique_three_digit_numbers_unique_three_digit_odd_numbers_l1743_174332


namespace NUMINAMATH_GPT_parabola_focus_l1743_174374

theorem parabola_focus (p : ℝ) (h : 4 = 2 * p * 1^2) : (0, 1 / (4 * 2 * p)) = (0, 1 / 16) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l1743_174374


namespace NUMINAMATH_GPT_walter_bus_time_l1743_174367

/--
Walter wakes up at 6:30 a.m., leaves for the bus at 7:30 a.m., attends 7 classes that each last 45 minutes,
enjoys a 40-minute lunch, and spends 2.5 hours of additional time at school for activities.
He takes the bus home and arrives at 4:30 p.m.
Prove that Walter spends 35 minutes on the bus.
-/
theorem walter_bus_time : 
  let total_time_away := 9 * 60 -- in minutes
  let class_time := 7 * 45 -- in minutes
  let lunch_time := 40 -- in minutes
  let additional_school_time := 2.5 * 60 -- in minutes
  total_time_away - (class_time + lunch_time + additional_school_time) = 35 := 
by
  sorry

end NUMINAMATH_GPT_walter_bus_time_l1743_174367


namespace NUMINAMATH_GPT_num_ways_to_divide_friends_l1743_174364

theorem num_ways_to_divide_friends :
  (4 : ℕ) ^ 8 = 65536 := by
  sorry

end NUMINAMATH_GPT_num_ways_to_divide_friends_l1743_174364


namespace NUMINAMATH_GPT_lily_coffee_budget_l1743_174388

variable (initial_amount celery_price cereal_original_price bread_price milk_original_price potato_price : ℕ)
variable (cereal_discount milk_discount number_of_potatoes : ℕ)

theorem lily_coffee_budget 
  (h_initial_amount : initial_amount = 60)
  (h_celery_price : celery_price = 5)
  (h_cereal_original_price : cereal_original_price = 12)
  (h_bread_price : bread_price = 8)
  (h_milk_original_price : milk_original_price = 10)
  (h_potato_price : potato_price = 1)
  (h_number_of_potatoes : number_of_potatoes = 6)
  (h_cereal_discount : cereal_discount = 50)
  (h_milk_discount : milk_discount = 10) :
  initial_amount - (celery_price + (cereal_original_price * cereal_discount / 100) + bread_price + (milk_original_price - (milk_original_price * milk_discount / 100)) + (potato_price * number_of_potatoes)) = 26 :=
by
  sorry

end NUMINAMATH_GPT_lily_coffee_budget_l1743_174388


namespace NUMINAMATH_GPT_knitting_time_total_l1743_174343

-- Define knitting times for each item
def hat_knitting_time : ℕ := 2
def scarf_knitting_time : ℕ := 3
def mitten_knitting_time : ℕ := 1
def sock_knitting_time : ℕ := 3 / 2
def sweater_knitting_time : ℕ := 6

-- Define the number of grandchildren
def grandchildren_count : ℕ := 3

-- Total knitting time calculation
theorem knitting_time_total : 
  hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time = 16 ∧ 
  (hat_knitting_time + scarf_knitting_time + 2 * mitten_knitting_time + 2 * sock_knitting_time + sweater_knitting_time) * grandchildren_count = 48 :=
by 
  sorry

end NUMINAMATH_GPT_knitting_time_total_l1743_174343


namespace NUMINAMATH_GPT_range_of_a_l1743_174390

-- Define the sets A, B, and C
def set_A (x : ℝ) : Prop := -3 < x ∧ x ≤ 2
def set_B (x : ℝ) : Prop := -1 < x ∧ x < 3
def set_A_int_B (x : ℝ) : Prop := -1 < x ∧ x ≤ 2
def set_C (x : ℝ) (a : ℝ) : Prop := a < x ∧ x < a + 1

-- The target theorem to prove
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, set_C x a → set_A_int_B x) → 
  (-1 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1743_174390


namespace NUMINAMATH_GPT_valid_B_sets_l1743_174312

def A : Set ℝ := {x | 0 < x ∧ x < 2}

theorem valid_B_sets (B : Set ℝ) : A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = A :=
by
  sorry

end NUMINAMATH_GPT_valid_B_sets_l1743_174312


namespace NUMINAMATH_GPT_normal_line_at_point_l1743_174323

noncomputable def curve (x : ℝ) : ℝ := (4 * x - x ^ 2) / 4

theorem normal_line_at_point (x0 : ℝ) (h : x0 = 2) :
  ∃ (L : ℝ → ℝ), ∀ (x : ℝ), L x = (2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_normal_line_at_point_l1743_174323


namespace NUMINAMATH_GPT_smallest_possible_a_plus_b_l1743_174357

theorem smallest_possible_a_plus_b :
  ∃ (a b : ℕ), (0 < a ∧ 0 < b) ∧ (2^10 * 7^3 = a^b) ∧ (a + b = 350753) :=
sorry

end NUMINAMATH_GPT_smallest_possible_a_plus_b_l1743_174357


namespace NUMINAMATH_GPT_number_of_pages_in_contract_l1743_174387

theorem number_of_pages_in_contract (total_pages_copied : ℕ) (copies_per_person : ℕ) (number_of_people : ℕ)
  (h1 : total_pages_copied = 360) (h2 : copies_per_person = 2) (h3 : number_of_people = 9) :
  total_pages_copied / (copies_per_person * number_of_people) = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pages_in_contract_l1743_174387


namespace NUMINAMATH_GPT_probability_tenth_ball_black_l1743_174310

theorem probability_tenth_ball_black :
  let total_balls := 30
  let black_balls := 4
  let red_balls := 7
  let yellow_balls := 5
  let green_balls := 6
  let white_balls := 8
  (black_balls / total_balls) = 4 / 30 :=
by sorry

end NUMINAMATH_GPT_probability_tenth_ball_black_l1743_174310


namespace NUMINAMATH_GPT_total_distance_biked_l1743_174353

-- Definitions of the given conditions
def biking_time_to_park : ℕ := 15
def biking_time_return : ℕ := 25
def average_speed : ℚ := 6 -- miles per hour

-- Total biking time in minutes, then converted to hours
def total_biking_time_minutes : ℕ := biking_time_to_park + biking_time_return
def total_biking_time_hours : ℚ := total_biking_time_minutes / 60

-- Prove that the total distance biked is 4 miles
theorem total_distance_biked : total_biking_time_hours * average_speed = 4 := 
by
  -- proof will be here
  sorry

end NUMINAMATH_GPT_total_distance_biked_l1743_174353


namespace NUMINAMATH_GPT_population_increase_rate_l1743_174300

theorem population_increase_rate (persons : ℕ) (minutes : ℕ) (seconds_per_person : ℕ) 
  (h1 : persons = 240) 
  (h2 : minutes = 60) 
  (h3 : seconds_per_person = (minutes * 60) / persons) 
  : seconds_per_person = 15 :=
by 
  sorry

end NUMINAMATH_GPT_population_increase_rate_l1743_174300


namespace NUMINAMATH_GPT_polynomial_root_expression_l1743_174335

theorem polynomial_root_expression (a b : ℂ) 
  (h₁ : a + b = 5) (h₂ : a * b = 6) : 
  a^4 + a^5 * b^3 + a^3 * b^5 + b^4 = 2905 := by
  sorry

end NUMINAMATH_GPT_polynomial_root_expression_l1743_174335


namespace NUMINAMATH_GPT_number_of_licenses_l1743_174396

-- We define the conditions for the problem
def number_of_letters : ℕ := 3  -- B, C, or D
def number_of_digits : ℕ := 4   -- Four digits following the letter
def choices_per_digit : ℕ := 10 -- Each digit can range from 0 to 9

-- We define the total number of licenses that can be generated
def total_licenses : ℕ := number_of_letters * (choices_per_digit ^ number_of_digits)

-- We now state the theorem to be proved
theorem number_of_licenses : total_licenses = 30000 :=
by
  sorry

end NUMINAMATH_GPT_number_of_licenses_l1743_174396


namespace NUMINAMATH_GPT_simplify_and_find_ratio_l1743_174326

theorem simplify_and_find_ratio (k : ℤ) : (∃ (c d : ℤ), (∀ x y : ℤ, c = 1 ∧ d = 2 ∧ x = c ∧ y = d → ((6 * k + 12) / 6 = k + 2) ∧ (c / d = 1 / 2))) :=
by
  use 1
  use 2
  sorry

end NUMINAMATH_GPT_simplify_and_find_ratio_l1743_174326


namespace NUMINAMATH_GPT_rectangle_area_l1743_174319

theorem rectangle_area (length diagonal : ℝ) (h_length : length = 16) (h_diagonal : diagonal = 20) : 
  ∃ width : ℝ, (length * width = 192) :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l1743_174319


namespace NUMINAMATH_GPT_percent_employed_females_in_employed_population_l1743_174320

def percent_employed (population: ℝ) : ℝ := 0.64 * population
def percent_employed_males (population: ℝ) : ℝ := 0.50 * population
def percent_employed_females (population: ℝ) : ℝ := percent_employed population - percent_employed_males population

theorem percent_employed_females_in_employed_population (population: ℝ) : 
  (percent_employed_females population / percent_employed population) * 100 = 21.875 :=
by
  sorry

end NUMINAMATH_GPT_percent_employed_females_in_employed_population_l1743_174320


namespace NUMINAMATH_GPT_fraction_decomposition_l1743_174328
noncomputable def A := (48 : ℚ) / 17
noncomputable def B := (-(25 : ℚ) / 17)

theorem fraction_decomposition (A : ℚ) (B : ℚ) :
  ( ∀ x : ℚ, x ≠ -5 ∧ x ≠ 2/3 →
    (7 * x - 13) / (3 * x^2 + 13 * x - 10) = A / (x + 5) + B / (3 * x - 2) ) ↔ 
    (A = (48 : ℚ) / 17 ∧ B = (-(25 : ℚ) / 17)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_decomposition_l1743_174328


namespace NUMINAMATH_GPT_triangle_area_l1743_174315

variables {A B C D M N: Type}

-- Define the conditions and the proof 
theorem triangle_area
  (α β : ℝ)
  (CD : ℝ)
  (sin_Ratio : ℝ)
  (C_angle : ℝ)
  (MCN_Area : ℝ)
  (M_distance : ℝ)
  (N_distance : ℝ)
  (hCD : CD = Real.sqrt 13)
  (hSinRatio : (Real.sin α) / (Real.sin β) = 4 / 3)
  (hC_angle : C_angle = 120)
  (hMCN_Area : MCN_Area = 3 * Real.sqrt 3)
  (hDistance : M_distance = 2 * N_distance)
  : ∃ ABC_Area, ABC_Area = 27 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_triangle_area_l1743_174315


namespace NUMINAMATH_GPT_shaded_region_area_is_correct_l1743_174362

noncomputable def area_of_shaded_region : ℝ :=
  let R := 6 -- radius of the larger circle
  let r := R / 2 -- radius of each smaller circle
  let area_large_circle := Real.pi * R^2
  let area_two_small_circles := 2 * Real.pi * r^2
  area_large_circle - area_two_small_circles

theorem shaded_region_area_is_correct :
  area_of_shaded_region = 18 * Real.pi :=
sorry

end NUMINAMATH_GPT_shaded_region_area_is_correct_l1743_174362


namespace NUMINAMATH_GPT_cost_per_box_of_cookies_l1743_174349

-- Given conditions
def initial_money : ℝ := 20
def mother_gift : ℝ := 2 * initial_money
def total_money : ℝ := initial_money + mother_gift
def cupcake_price : ℝ := 1.50
def num_cupcakes : ℝ := 10
def cost_cupcakes : ℝ := num_cupcakes * cupcake_price
def money_after_cupcakes : ℝ := total_money - cost_cupcakes
def remaining_money : ℝ := 30
def num_boxes_cookies : ℝ := 5
def money_spent_on_cookies : ℝ := money_after_cupcakes - remaining_money

-- Theorem: Calculate the cost per box of cookies
theorem cost_per_box_of_cookies : (money_spent_on_cookies / num_boxes_cookies) = 3 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_box_of_cookies_l1743_174349


namespace NUMINAMATH_GPT_ana_final_salary_l1743_174311

def initial_salary : ℝ := 2500
def june_raise : ℝ := initial_salary * 0.15
def june_bonus : ℝ := 300
def salary_after_june : ℝ := initial_salary + june_raise + june_bonus
def july_pay_cut : ℝ := salary_after_june * 0.25
def final_salary : ℝ := salary_after_june - july_pay_cut

theorem ana_final_salary :
  final_salary = 2381.25 := by
  -- sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_ana_final_salary_l1743_174311


namespace NUMINAMATH_GPT_circle_area_polar_eq_l1743_174327

theorem circle_area_polar_eq (r θ : ℝ) :
    (r = 3 * Real.cos θ - 4 * Real.sin θ) → (π * (5 / 2) ^ 2 = 25 * π / 4) :=
by 
  sorry

end NUMINAMATH_GPT_circle_area_polar_eq_l1743_174327


namespace NUMINAMATH_GPT_calculate_ab_l1743_174382

theorem calculate_ab {a b c : ℝ} (hc : c ≠ 0) (h1 : (a * b) / c = 4) (h2 : a * (b / c) = 12) : a * b = 12 :=
by
  sorry

end NUMINAMATH_GPT_calculate_ab_l1743_174382


namespace NUMINAMATH_GPT_parabola_hyperbola_focus_vertex_l1743_174377

theorem parabola_hyperbola_focus_vertex (p : ℝ) : 
  (∃ (focus_vertex : ℝ × ℝ), focus_vertex = (2, 0) 
    ∧ focus_vertex = (p / 2, 0)) → p = 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_hyperbola_focus_vertex_l1743_174377


namespace NUMINAMATH_GPT_integer_sum_19_l1743_174325

variable (p q r s : ℤ)

theorem integer_sum_19 (h1 : p - q + r = 4) 
                       (h2 : q - r + s = 5) 
                       (h3 : r - s + p = 7) 
                       (h4 : s - p + q = 3) :
                       p + q + r + s = 19 :=
by
  sorry

end NUMINAMATH_GPT_integer_sum_19_l1743_174325


namespace NUMINAMATH_GPT_solution_set_inequality_l1743_174318

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1
axiom f_condition2 : ∀ x : ℝ, deriv f x < 1 / 2

theorem solution_set_inequality : {x : ℝ | 0 < x ∧ x < 2} = {x : ℝ | f (Real.log x / Real.log 2) > (Real.log x / Real.log 2 + 1) / 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1743_174318


namespace NUMINAMATH_GPT_max_quadratic_function_l1743_174344

theorem max_quadratic_function :
  ∃ M, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → (x^2 - 2*x - 1 ≤ M)) ∧
       (∀ y : ℝ, y = (x : ℝ) ^ 2 - 2 * x - 1 → x = 3 → y = M) :=
by
  use 2
  sorry

end NUMINAMATH_GPT_max_quadratic_function_l1743_174344


namespace NUMINAMATH_GPT_maximum_value_of_objective_function_l1743_174342

variables (x y : ℝ)

def objective_function (x y : ℝ) := 3 * x + 2 * y

theorem maximum_value_of_objective_function : 
  (∀ x y, (x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4) → objective_function x y ≤ 12) 
  ∧ 
  (∃ x y, x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4 ∧ objective_function x y = 12) :=
sorry

end NUMINAMATH_GPT_maximum_value_of_objective_function_l1743_174342


namespace NUMINAMATH_GPT_mean_of_sequence_l1743_174385

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem mean_of_sequence :
  mean [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 2] = 17.75 := by
sorry

end NUMINAMATH_GPT_mean_of_sequence_l1743_174385


namespace NUMINAMATH_GPT_quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l1743_174393

theorem quadrant_606 (θ : ℝ) : θ = 606 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

theorem quadrant_minus_950 (θ : ℝ) : θ = -950 → (90 < (θ % 360) ∧ (θ % 360) < 180) := by
  sorry

theorem same_terminal_side (α k : ℤ) : (α = -457 + k * 360) ↔ (∃ n : ℤ, α = -457 + n * 360) := by
  sorry

theorem quadrant_minus_97 (θ : ℝ) : θ = -97 → (180 < (θ % 360) ∧ (θ % 360) < 270) := by
  sorry

end NUMINAMATH_GPT_quadrant_606_quadrant_minus_950_same_terminal_side_quadrant_minus_97_l1743_174393


namespace NUMINAMATH_GPT_snow_leopards_arrangement_l1743_174333

theorem snow_leopards_arrangement :
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  end_positions * factorial_six = 1440 :=
by
  let leopards := 8
  let end_positions := 2
  let remaining_leopards := 6
  let factorial_six := Nat.factorial remaining_leopards
  show end_positions * factorial_six = 1440
  sorry

end NUMINAMATH_GPT_snow_leopards_arrangement_l1743_174333
