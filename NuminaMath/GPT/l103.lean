import Mathlib

namespace half_angle_quadrant_l103_10303

theorem half_angle_quadrant (k : ℤ) (α : ℝ) (h : 2 * k * π + π < α ∧ α < 2 * k * π + (3 * π / 2)) : 
  (∃ j : ℤ, j * π + (π / 2) < (α / 2) ∧ (α / 2) < j * π + (3 * π / 4)) :=
  by sorry

end half_angle_quadrant_l103_10303


namespace value_of_t_l103_10398

def vec (x y : ℝ) := (x, y)

def p := vec 3 3
def q := vec (-1) 2
def r := vec 4 1

noncomputable def t := 3

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem value_of_t (t : ℝ) : (dot_product (vec (6 + 4 * t) (6 + t)) q = 0) ↔ t = 3 :=
by
  sorry

end value_of_t_l103_10398


namespace calculate_expression_l103_10352

theorem calculate_expression :
  ((7 / 9) - (5 / 6) + (5 / 18)) * 18 = 4 :=
by
  -- proof to be filled in later.
  sorry

end calculate_expression_l103_10352


namespace count_even_thousands_digit_palindromes_l103_10364

-- Define the set of valid digits
def valid_A : Finset ℕ := {2, 4, 6, 8}
def valid_B : Finset ℕ := Finset.range 10

-- Define the condition of a four-digit palindrome ABBA where A is even and non-zero
def is_valid_palindrome (a b : ℕ) : Prop :=
  a ∈ valid_A ∧ b ∈ valid_B

-- The proof problem: Prove that the total number of valid palindromes ABBA is 40
theorem count_even_thousands_digit_palindromes :
  (valid_A.card) * (valid_B.card) = 40 :=
by
  -- Skipping the proof itself
  sorry

end count_even_thousands_digit_palindromes_l103_10364


namespace positive_difference_of_two_numbers_l103_10328

theorem positive_difference_of_two_numbers
  (x y : ℝ)
  (h₁ : x + y = 10)
  (h₂ : x^2 - y^2 = 24) :
  |x - y| = 12 / 5 :=
sorry

end positive_difference_of_two_numbers_l103_10328


namespace sum_x_y_z_eq_3_or_7_l103_10304

theorem sum_x_y_z_eq_3_or_7 (x y z : ℝ) (h1 : x + y / z = 2) (h2 : y + z / x = 2) (h3 : z + x / y = 2) : x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end sum_x_y_z_eq_3_or_7_l103_10304


namespace lisa_additional_marbles_l103_10354

theorem lisa_additional_marbles (n : ℕ) (f : ℕ) (m : ℕ) (current_marbles : ℕ) : 
  n = 12 ∧ f = n ∧ m = (n * (n + 1)) / 2 ∧ current_marbles = 34 → 
  m - current_marbles = 44 :=
by
  intros
  sorry

end lisa_additional_marbles_l103_10354


namespace price_of_cork_l103_10372

theorem price_of_cork (C : ℝ) 
  (h₁ : ∃ (bottle_with_cork bottle_without_cork : ℝ), bottle_with_cork = 2.10 ∧ bottle_without_cork = C + 2.00 ∧ bottle_with_cork = C + bottle_without_cork) :
  C = 0.05 :=
by
  obtain ⟨bottle_with_cork, bottle_without_cork, hwc, hwoc, ht⟩ := h₁
  sorry

end price_of_cork_l103_10372


namespace min_value_expression_l103_10327

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + ((b / a) - 1)^2 + ((c / b) - 1)^2 + ((5 / c) - 1)^2 ≥ 20 - 8 * Real.sqrt 5 := 
by
  sorry

end min_value_expression_l103_10327


namespace weight_of_b_l103_10392

variable {A B C : ℤ}

def condition1 (A B C : ℤ) : Prop := (A + B + C) / 3 = 45
def condition2 (A B : ℤ) : Prop := (A + B) / 2 = 42
def condition3 (B C : ℤ) : Prop := (B + C) / 2 = 43

theorem weight_of_b (A B C : ℤ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B) 
  (h3 : condition3 B C) : 
  B = 35 := 
by
  sorry

end weight_of_b_l103_10392


namespace min_value_of_M_l103_10379

noncomputable def f (x : ℝ) : ℝ := Real.log x

theorem min_value_of_M (M : ℝ) (hM : M = Real.sqrt 2) :
  ∀ (a b c : ℝ), a > M → b > M → c > M → a^2 + b^2 = c^2 → 
  (f a) + (f b) > f c ∧ (f a) + (f c) > f b ∧ (f b) + (f c) > f a :=
by
  sorry

end min_value_of_M_l103_10379


namespace house_transaction_l103_10335

variable (initial_value : ℝ) (loss_rate : ℝ) (gain_rate : ℝ) (final_loss : ℝ)

theorem house_transaction
  (h_initial : initial_value = 12000)
  (h_loss : loss_rate = 0.15)
  (h_gain : gain_rate = 0.15)
  (h_final_loss : final_loss = 270) :
  let selling_price := initial_value * (1 - loss_rate)
  let buying_price := selling_price * (1 + gain_rate)
  (initial_value - buying_price) = final_loss :=
by
  simp only [h_initial, h_loss, h_gain, h_final_loss]
  sorry

end house_transaction_l103_10335


namespace system_of_equations_abs_diff_l103_10314

theorem system_of_equations_abs_diff 
  (x y m n : ℝ) 
  (h₁ : 2 * x - y = m)
  (h₂ : x + m * y = n)
  (hx : x = 2)
  (hy : y = 1) : 
  |m - n| = 2 :=
by
  sorry

end system_of_equations_abs_diff_l103_10314


namespace solution_system_of_equations_l103_10344

theorem solution_system_of_equations : 
  ∃ (x y : ℝ), (2 * x - y = 3 ∧ x + y = 3) ∧ (x = 2 ∧ y = 1) := 
by
  sorry

end solution_system_of_equations_l103_10344


namespace original_length_l103_10378

-- Definitions based on conditions
def length_sawed_off : ℝ := 0.33
def remaining_length : ℝ := 0.08

-- The problem statement translated to a Lean 4 theorem
theorem original_length (L : ℝ) (h1 : L = length_sawed_off + remaining_length) : 
  L = 0.41 :=
by
  sorry

end original_length_l103_10378


namespace sum_of_common_ratios_l103_10394

noncomputable def geometric_sequence (m x : ℝ) : ℝ × ℝ × ℝ := (m, m * x, m * x^2)

theorem sum_of_common_ratios
  (m x y : ℝ)
  (h1 : x ≠ y)
  (h2 : m ≠ 0)
  (h3 : ∃ c3 c2 d3 d2 : ℝ, geometric_sequence m x = (m, c2, c3) ∧ geometric_sequence m y = (m, d2, d3) ∧ c3 - d3 = 3 * (c2 - d2)) :
  x + y = 3 := by
  sorry

end sum_of_common_ratios_l103_10394


namespace volume_of_first_bottle_l103_10386

theorem volume_of_first_bottle (V_2 V_3 : ℕ) (V_total : ℕ):
  V_2 = 750 ∧ V_3 = 250 ∧ V_total = 3 * 1000 →
  (V_total - V_2 - V_3) / 1000 = 2 :=
by
  sorry

end volume_of_first_bottle_l103_10386


namespace translate_down_three_units_l103_10353

def original_function (x : ℝ) : ℝ := 3 * x + 2

def translated_function (x : ℝ) : ℝ := 3 * x - 1

theorem translate_down_three_units :
  ∀ x : ℝ, translated_function x = original_function x - 3 :=
by
  intro x
  simp [original_function, translated_function]
  sorry

end translate_down_three_units_l103_10353


namespace area_of_region_l103_10305

theorem area_of_region (x y : ℝ) (h : x^2 + y^2 + 6 * x - 8 * y - 5 = 0) : 
  ∃ (r : ℝ), (π * r^2 = 30 * π) :=
by -- Starting the proof, skipping the detailed steps
sorry -- Proof placeholder

end area_of_region_l103_10305


namespace distance_behind_l103_10324

-- Given conditions
variables {A B E : ℝ} -- Speed of Anusha, Banu, and Esha
variables {Da Db De : ℝ} -- distances covered by Anusha, Banu, and Esha

axiom const_speeds : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)

-- The proof to be established
theorem distance_behind (h : Da = 100 ∧ Db = 90 ∧ Db / Da = De / Db ∧ De = 90 * (Db / 100)) :
  100 - De = 19 :=
by sorry

end distance_behind_l103_10324


namespace problem_inequality_l103_10341

variable {n : ℕ}
variable (S_n : Finset (Fin n)) (f : Finset (Fin n) → ℝ)

axiom pos_f : ∀ A : Finset (Fin n), 0 < f A
axiom cond_f : ∀ (A : Finset (Fin n)) (x y : Fin n), x ≠ y → f (A ∪ {x}) * f (A ∪ {y}) ≤ f (A ∪ {x, y}) * f A

theorem problem_inequality (A B : Finset (Fin n)) : f A * f B ≤ f (A ∪ B) * f (A ∩ B) := sorry

end problem_inequality_l103_10341


namespace kangaroo_meetings_l103_10395

/-- 
Two kangaroos, A and B, start at point A and jump in specific sequences:
- Kangaroo A jumps in the sequence A, B, C, D, E, F, G, H, I, A, B, C, ... in a loop every 9 jumps.
- Kangaroo B jumps in the sequence A, B, D, E, G, H, A, B, D, ... in a loop every 6 jumps.
They start at point A together. Prove that they will land on the same point 226 times after 2017 jumps.
-/
theorem kangaroo_meetings (n : Nat) (ka : Fin 9 → Fin 9) (kb : Fin 6 → Fin 6)
  (hka : ∀ i, ka i = (i + 1) % 9) (hkb : ∀ i, kb i = (i + 1) % 6) :
  n = 2017 →
  -- Prove that the two kangaroos will meet 226 times after 2017 jumps
  ∃ k, k = 226 :=
by
  sorry

end kangaroo_meetings_l103_10395


namespace cost_unit_pen_max_profit_and_quantity_l103_10388

noncomputable def cost_pen_A : ℝ := 5
noncomputable def cost_pen_B : ℝ := 10
noncomputable def profit_pen_A : ℝ := 2
noncomputable def profit_pen_B : ℝ := 3
noncomputable def spent_on_A : ℝ := 400
noncomputable def spent_on_B : ℝ := 800
noncomputable def total_pens : ℝ := 300

theorem cost_unit_pen : (spent_on_A / cost_pen_A) = (spent_on_B / (cost_pen_A + 5)) := by
  sorry

theorem max_profit_and_quantity
    (xa xb : ℝ)
    (h1 : xa ≥ 4 * xb)
    (h2 : xa + xb = total_pens)
    : ∃ (wa : ℝ), wa = 2 * xa + 3 * xb ∧ xa = 240 ∧ xb = 60 ∧ wa = 660 := by
  sorry

end cost_unit_pen_max_profit_and_quantity_l103_10388


namespace find_a_values_l103_10351

theorem find_a_values (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 27 * x^3) (h₃ : a - b = 2 * x) :
  a = 3.041 * x ∨ a = -1.041 * x :=
by
  sorry

end find_a_values_l103_10351


namespace product_price_interval_l103_10322

def is_too_high (price guess : ℕ) : Prop := guess > price
def is_too_low  (price guess : ℕ) : Prop := guess < price

theorem product_price_interval 
    (price : ℕ)
    (h1 : is_too_high price 2000)
    (h2 : is_too_low price 1000)
    (h3 : is_too_high price 1500)
    (h4 : is_too_low price 1250)
    (h5 : is_too_low price 1375) :
    1375 < price ∧ price < 1500 :=
    sorry

end product_price_interval_l103_10322


namespace problem1_problem2_l103_10362

-- Define the propositions
def S (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 2 * m * x + 2 - m = 0

def p (m : ℝ) : Prop := 0 < m ∧ m < 2

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * m * x + 1 > 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ 1 ≤ m := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hpq : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end problem1_problem2_l103_10362


namespace determine_constants_l103_10348

theorem determine_constants :
  ∃ P Q R : ℚ, (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → (x^2 - 4 * x + 8) / ((x - 1) * (x - 4) * (x - 6)) = P / (x - 1) + Q / (x - 4) + R / (x - 6)) ∧ 
  P = 1 / 3 ∧ Q = - 4 / 3 ∧ R = 2 :=
by
  -- Proof is left as a placeholder
  sorry

end determine_constants_l103_10348


namespace problem_l103_10346

theorem problem (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -1/2 ∨ x > 1/3)) : 
  a + b = -14 :=
sorry

end problem_l103_10346


namespace jessica_cut_roses_l103_10302

/-- There were 13 roses and 84 orchids in the vase. Jessica cut some more roses and 
orchids from her flower garden. There are now 91 orchids and 14 roses in the vase. 
How many roses did she cut? -/
theorem jessica_cut_roses :
  let initial_roses := 13
  let new_roses := 14
  ∃ cut_roses : ℕ, new_roses = initial_roses + cut_roses ∧ cut_roses = 1 :=
by
  sorry

end jessica_cut_roses_l103_10302


namespace find_expression_l103_10331

theorem find_expression (E a : ℝ) 
  (h1 : (E + (3 * a - 8)) / 2 = 69) 
  (h2 : a = 26) : 
  E = 68 :=
sorry

end find_expression_l103_10331


namespace estimate_pi_simulation_l103_10359

theorem estimate_pi_simulation :
  let side := 2
  let radius := 1
  let total_seeds := 1000
  let seeds_in_circle := 778
  (π : ℝ) * radius^2 / side^2 = (seeds_in_circle : ℝ) / total_seeds → π = 3.112 :=
by
  intros
  sorry

end estimate_pi_simulation_l103_10359


namespace base8_to_base10_l103_10329

theorem base8_to_base10 {a b : ℕ} (h1 : 3 * 64 + 7 * 8 + 4 = 252) (h2 : 252 = a * 10 + b) :
  (a + b : ℝ) / 20 = 0.35 :=
sorry

end base8_to_base10_l103_10329


namespace original_percent_acid_l103_10317

open Real

variables (a w : ℝ)

theorem original_percent_acid 
  (h1 : (a + 2) / (a + w + 2) = 1 / 4)
  (h2 : (a + 2) / (a + w + 4) = 1 / 5) :
  a / (a + w) = 1 / 5 :=
sorry

end original_percent_acid_l103_10317


namespace power_inequality_l103_10321

open Nat

theorem power_inequality (a b : ℝ) (n : ℕ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a) + (1 / b) = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
  sorry

end power_inequality_l103_10321


namespace projections_relationship_l103_10375

theorem projections_relationship (a b r : ℝ) (h : r ≠ 0) :
  (∃ α β : ℝ, a = r * Real.cos α ∧ b = r * Real.cos β ∧ (Real.cos α)^2 + (Real.cos β)^2 = 1) → (a^2 + b^2 = r^2) :=
by
  sorry

end projections_relationship_l103_10375


namespace find_square_side_length_l103_10313

open Nat

def original_square_side_length (s : ℕ) : Prop :=
  let length := s + 8
  let breadth := s + 4
  (2 * (length + breadth)) = 40 → s = 4

theorem find_square_side_length (s : ℕ) : original_square_side_length s := by
  sorry

end find_square_side_length_l103_10313


namespace find_x_l103_10374

variable (x : ℝ)
variable (l : ℝ) (w : ℝ)

def length := 4 * x + 1
def width := x + 7

theorem find_x (h1 : l = length x) (h2 : w = width x) (h3 : l * w = 2 * (2 * l + 2 * w)) :
  x = (-9 + Real.sqrt 481) / 8 :=
by
  subst_vars
  sorry

end find_x_l103_10374


namespace M_intersection_P_l103_10363

namespace IntersectionProof

-- Defining the sets M and P with given conditions
def M : Set ℝ := {y | ∃ x : ℝ, y = 3 ^ x}
def P : Set ℝ := {y | y ≥ 1}

-- The theorem that corresponds to the problem statement
theorem M_intersection_P : (M ∩ P) = {y | y ≥ 1} :=
sorry

end IntersectionProof

end M_intersection_P_l103_10363


namespace math_problem_l103_10330

variable (p q r : ℝ)

theorem math_problem
  (h1: p + q + r = 5)
  (h2: 1 / (p + q) + 1 / (q + r) + 1 / (p + r) = 9) :
  r / (p + q) + p / (q + r) + q / (p + r) = 42 := by
  sorry

end math_problem_l103_10330


namespace dog_bones_l103_10376

theorem dog_bones (initial_bones found_bones : ℕ) (h₁ : initial_bones = 15) (h₂ : found_bones = 8) : initial_bones + found_bones = 23 := by
  sorry

end dog_bones_l103_10376


namespace least_people_to_complete_job_on_time_l103_10306

theorem least_people_to_complete_job_on_time
  (total_duration : ℕ)
  (initial_days : ℕ)
  (initial_people : ℕ)
  (initial_work_done : ℚ)
  (efficiency_multiplier : ℚ)
  (remaining_work_fraction : ℚ)
  (remaining_days : ℕ)
  (resulting_people : ℕ)
  (work_rate_doubled : ℕ → ℚ → ℚ)
  (final_resulting_people : ℚ)
  : initial_work_done = 1/4 →
    efficiency_multiplier = 2 →
    remaining_work_fraction = 3/4 →
    total_duration = 40 →
    initial_days = 10 →
    initial_people = 12 →
    remaining_days = 20 →
    work_rate_doubled 12 2 = 24 →
    final_resulting_people = (1/2) →
    resulting_people = 6 :=
sorry

end least_people_to_complete_job_on_time_l103_10306


namespace students_per_bench_l103_10360

theorem students_per_bench (num_male num_benches : ℕ) (h₁ : num_male = 29) (h₂ : num_benches = 29) (h₃ : ∀ num_female, num_female = 4 * num_male) : 
  ((29 + 4 * 29) / 29) = 5 :=
by
  sorry

end students_per_bench_l103_10360


namespace find_speed_B_l103_10300

def distance_to_location : ℝ := 12
def A_speed_is_1_2_times_B (speed_B speed_A : ℝ) : Prop := speed_A = 1.2 * speed_B
def A_arrives_1_6_hour_earlier (speed_B speed_A : ℝ) : Prop :=
  (distance_to_location / speed_B) - (distance_to_location / speed_A) = 1 / 6

theorem find_speed_B (speed_B : ℝ) (speed_A : ℝ) :
  A_speed_is_1_2_times_B speed_B speed_A →
  A_arrives_1_6_hour_earlier speed_B speed_A →
  speed_B = 12 :=
by
  intros h1 h2
  sorry

end find_speed_B_l103_10300


namespace acute_triangle_l103_10301

theorem acute_triangle (r R : ℝ) (h : R < r * (Real.sqrt 2 + 1)) : 
  ∃ (α β γ : ℝ), α + β + γ = π ∧ (0 < α) ∧ (0 < β) ∧ (0 < γ) ∧ (α < π / 2) ∧ (β < π / 2) ∧ (γ < π / 2) := 
sorry

end acute_triangle_l103_10301


namespace intersection_points_l103_10340

def f(x : ℝ) : ℝ := x^2 + 3*x + 2
def g(x : ℝ) : ℝ := 4*x^2 + 6*x + 2

theorem intersection_points : {p : ℝ × ℝ | ∃ x, f x = p.2 ∧ g x = p.2 ∧ p.1 = x} = { (0, 2), (-1, 0) } := 
by {
  sorry
}

end intersection_points_l103_10340


namespace toothpicks_needed_l103_10366

-- Defining the number of rows in the large equilateral triangle.
def rows : ℕ := 10

-- Formula to compute the total number of smaller equilateral triangles.
def total_small_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

-- Number of small triangles in this specific case.
def num_small_triangles : ℕ := total_small_triangles rows

-- Total toothpicks without sharing sides.
def total_sides_no_sharing (n : ℕ) : ℕ := 3 * num_small_triangles

-- Adjust for shared toothpicks internally.
def shared_toothpicks (n : ℕ) : ℕ := (total_sides_no_sharing n - 3 * rows) / 2 + 3 * rows

-- Total boundary toothpicks.
def boundary_toothpicks (n : ℕ) : ℕ := 3 * rows

-- Final total number of toothpicks required.
def total_toothpicks (n : ℕ) : ℕ := shared_toothpicks n + boundary_toothpicks n

-- The theorem to be proved
theorem toothpicks_needed : total_toothpicks rows = 98 :=
by
  -- You can complete the proof.
  sorry

end toothpicks_needed_l103_10366


namespace intersection_correct_union_correct_l103_10370

variable (U A B : Set Nat)

def U_set : U = {1, 2, 3, 4, 5, 6} := by sorry
def A_set : A = {2, 4, 5} := by sorry
def B_set : B = {1, 2, 5} := by sorry

theorem intersection_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∩ B) = {2, 5} := by sorry

theorem union_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∪ (U \ B)) = {2, 3, 4, 5, 6} := by sorry

end intersection_correct_union_correct_l103_10370


namespace solve_for_F_l103_10332

theorem solve_for_F (F C : ℝ) (h₁ : C = 4 / 7 * (F - 40)) (h₂ : C = 25) : F = 83.75 :=
sorry

end solve_for_F_l103_10332


namespace division_multiplication_result_l103_10385

theorem division_multiplication_result :
  (7.5 / 6) * 12 = 15 := by
  sorry

end division_multiplication_result_l103_10385


namespace how_many_more_red_balls_l103_10377

def r_packs : ℕ := 12
def y_packs : ℕ := 9
def r_balls_per_pack : ℕ := 24
def y_balls_per_pack : ℕ := 20

theorem how_many_more_red_balls :
  (r_packs * r_balls_per_pack) - (y_packs * y_balls_per_pack) = 108 :=
by
  sorry

end how_many_more_red_balls_l103_10377


namespace find_f_3_l103_10345

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x : ℝ) : f x + 3 * f (1 - x) = 4 * x ^ 2

theorem find_f_3 : f 3 = 3 / 2 := 
by
  sorry

end find_f_3_l103_10345


namespace mrs_heine_dogs_treats_l103_10319

theorem mrs_heine_dogs_treats (heart_biscuits_per_dog puppy_boots_per_dog total_items : ℕ)
  (h_biscuits : heart_biscuits_per_dog = 5)
  (h_boots : puppy_boots_per_dog = 1)
  (total : total_items = 12) :
  (total_items / (heart_biscuits_per_dog + puppy_boots_per_dog)) = 2 :=
by
  sorry

end mrs_heine_dogs_treats_l103_10319


namespace margo_total_distance_l103_10356

theorem margo_total_distance (time_to_friend : ℝ) (time_back_home : ℝ) (average_rate : ℝ)
  (total_time_hours : ℝ) (total_miles : ℝ) :
  time_to_friend = 12 / 60 ∧
  time_back_home = 24 / 60 ∧
  total_time_hours = (12 / 60) + (24 / 60) ∧
  average_rate = 3 ∧
  total_miles = average_rate * total_time_hours →
  total_miles = 1.8 :=
by
  sorry

end margo_total_distance_l103_10356


namespace area_of_hexagon_l103_10349

-- Definitions of the angles and side lengths
def angle_A := 120
def angle_B := 120
def angle_C := 120
def angle_D := 150

def FA := 2
def AB := 2
def BC := 2
def CD := 3
def DE := 3
def EF := 3

-- Theorem statement for the area of hexagon ABCDEF
theorem area_of_hexagon : 
  (angle_A = 120 ∧ angle_B = 120 ∧ angle_C = 120 ∧ angle_D = 150 ∧
   FA = 2 ∧ AB = 2 ∧ BC = 2 ∧ CD = 3 ∧ DE = 3 ∧ EF = 3) →
  (∃ area : ℝ, area = 7.5 * Real.sqrt 3) :=
by
  sorry

end area_of_hexagon_l103_10349


namespace original_number_l103_10312

theorem original_number (x : ℕ) (h : ∃ k, 14 * x = 112 * k) : x = 8 :=
sorry

end original_number_l103_10312


namespace magic_square_proof_l103_10399

theorem magic_square_proof
    (a b c d e S : ℕ)
    (h1 : 35 + e + 27 = S)
    (h2 : 30 + c + d = S)
    (h3 : a + 32 + b = S)
    (h4 : 35 + c + b = S)
    (h5 : a + c + 27 = S)
    (h6 : 35 + c + b = S)
    (h7 : 35 + c + 27 = S)
    (h8 : a + c + d = S) :
  d + e = 35 :=
  sorry

end magic_square_proof_l103_10399


namespace lemons_for_10_gallons_l103_10384

noncomputable def lemon_proportion : Prop :=
  ∃ x : ℝ, (36 / 48) = (x / 10) ∧ x = 7.5

theorem lemons_for_10_gallons : lemon_proportion :=
by
  sorry

end lemons_for_10_gallons_l103_10384


namespace closed_polygonal_chain_exists_l103_10323

theorem closed_polygonal_chain_exists (n m : ℕ) : 
  ((n % 2 = 1 ∨ m % 2 = 1) ↔ 
   ∃ (length : ℕ), length = (n + 1) * (m + 1) ∧ length % 2 = 0) :=
by sorry

end closed_polygonal_chain_exists_l103_10323


namespace find_w_l103_10357

variable (p j t : ℝ) (w : ℝ)

-- Definitions based on conditions
def j_less_than_p : Prop := j = 0.75 * p
def j_less_than_t : Prop := j = 0.80 * t
def t_less_than_p : Prop := t = p * (1 - w / 100)

-- Objective: Prove that given these conditions, w = 6.25
theorem find_w (h1 : j_less_than_p p j) (h2 : j_less_than_t j t) (h3 : t_less_than_p t p w) : 
  w = 6.25 := 
by 
  sorry

end find_w_l103_10357


namespace simplify_expression_l103_10381

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (15 * x^2) * (6 * x) * (1 / (3 * x)^2) = 10 * x := 
by
  sorry

end simplify_expression_l103_10381


namespace unit_digit_calc_l103_10380

theorem unit_digit_calc : (8 * 19 * 1981 - 8^3) % 10 = 0 := by
  sorry

end unit_digit_calc_l103_10380


namespace percent_of_juniors_involved_in_sports_l103_10318

theorem percent_of_juniors_involved_in_sports
  (total_students : ℕ)
  (percent_juniors : ℝ)
  (juniors_in_sports : ℕ)
  (h1 : total_students = 500)
  (h2 : percent_juniors = 0.40)
  (h3 : juniors_in_sports = 140) :
  (juniors_in_sports : ℝ) / (total_students * percent_juniors) * 100 = 70 := 
by
  -- By conditions h1, h2, h3:
  sorry

end percent_of_juniors_involved_in_sports_l103_10318


namespace probability_below_8_l103_10334

theorem probability_below_8 (p10 p9 p8 : ℝ) (h1 : p10 = 0.20) (h2 : p9 = 0.30) (h3 : p8 = 0.10) : 
  1 - (p10 + p9 + p8) = 0.40 :=
by 
  rw [h1, h2, h3]
  sorry

end probability_below_8_l103_10334


namespace part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l103_10308

-- Part (1)
theorem part1_coordinates_on_x_axis (a : ℝ) (h : a + 5 = 0) : (2*a - 2, a + 5) = (-12, 0) :=
by sorry

-- Part (2)
theorem part2_coordinates_parallel_y_axis (a : ℝ) (h : 2*a - 2 = 4) : (2*a - 2, a + 5) = (4, 8) :=
by sorry

-- Part (3)
theorem part3_distances_equal_second_quadrant (a : ℝ) 
  (h1 : 2*a-2 < 0) (h2 : a+5 > 0) (h3 : abs (2*a - 2) = abs (a + 5)) : a^(2022 : ℕ) + 2022 = 2023 :=
by sorry

end part1_coordinates_on_x_axis_part2_coordinates_parallel_y_axis_part3_distances_equal_second_quadrant_l103_10308


namespace triangle_segments_l103_10367

def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_segments (a : ℕ) (h : a > 0) :
  ¬ triangle_inequality 1 2 3 ∧
  ¬ triangle_inequality 4 5 10 ∧
  triangle_inequality 5 10 13 ∧
  ¬ triangle_inequality (2 * a) (3 * a) (6 * a) :=
by
  -- Proof goes here
  sorry

end triangle_segments_l103_10367


namespace current_number_of_women_is_24_l103_10393

-- Define initial person counts based on the given ratio and an arbitrary factor x.
variables (x : ℕ)
def M_initial := 4 * x
def W_initial := 5 * x
def C_initial := 3 * x
def E_initial := 2 * x

-- Define the changes that happened to the room.
def men_after_entry := M_initial x + 2
def women_after_leaving := W_initial x - 3
def women_after_doubling := 2 * women_after_leaving x
def children_after_leaving := C_initial x - 5
def elderly_after_leaving := E_initial x - 3

-- Define the current counts after all changes.
def men_current := 14
def children_current := 7
def elderly_current := 6

-- Prove that the current number of women is 24.
theorem current_number_of_women_is_24 :
  men_after_entry x = men_current ∧
  children_after_leaving x = children_current ∧
  elderly_after_leaving x = elderly_current →
  women_after_doubling x = 24 :=
by
  sorry

end current_number_of_women_is_24_l103_10393


namespace runners_speed_ratio_l103_10396

/-- Two runners, 20 miles apart, start at the same time, aiming to meet. 
    If they run in the same direction, they meet in 5 hours. 
    If they run towards each other, they meet in 1 hour.
    Prove that the ratio of the speed of the faster runner to the slower runner is 3/2. -/
theorem runners_speed_ratio (v1 v2 : ℝ) (h1 : v1 > v2)
  (h2 : 20 = 5 * (v1 - v2)) 
  (h3 : 20 = (v1 + v2)) : 
  v1 / v2 = 3 / 2 :=
sorry

end runners_speed_ratio_l103_10396


namespace hannahs_adblock_not_block_l103_10358

theorem hannahs_adblock_not_block (x : ℝ) (h1 : 0.8 * x = 0.16) : x = 0.2 :=
by {
  sorry
}

end hannahs_adblock_not_block_l103_10358


namespace cylinder_volume_l103_10365

theorem cylinder_volume (r h V: ℝ) (r_pos: r = 4) (lateral_area: 2 * 3.14 * r * h = 62.8) : 
    V = 125600 :=
by
  sorry

end cylinder_volume_l103_10365


namespace total_money_is_twenty_l103_10338

-- Define Henry's initial money
def henry_initial_money : Nat := 5

-- Define the money Henry earned
def henry_earned_money : Nat := 2

-- Define Henry's total money
def henry_total_money : Nat := henry_initial_money + henry_earned_money

-- Define friend's money
def friend_money : Nat := 13

-- Define the total combined money
def total_combined_money : Nat := henry_total_money + friend_money

-- The main statement to prove
theorem total_money_is_twenty : total_combined_money = 20 := sorry

end total_money_is_twenty_l103_10338


namespace find_analytical_expression_of_f_l103_10316

-- Given conditions: f(1/x) = 1/(x+1)
def f (x : ℝ) : ℝ := sorry

-- Domain statement (optional for additional clarity):
def domain (x : ℝ) := x ≠ 0 ∧ x ≠ -1

-- Proof obligation: Prove that f(x) = x / (x + 1)
theorem find_analytical_expression_of_f :
  ∀ x : ℝ, domain x → f x = x / (x + 1) := sorry

end find_analytical_expression_of_f_l103_10316


namespace sequence_sum_identity_l103_10342

theorem sequence_sum_identity 
  (a_n b_n : ℕ → ℕ) 
  (S_n T_n : ℕ → ℕ)
  (h1 : ∀ n, b_n n - a_n n = 2^n + 1)
  (h2 : ∀ n, S_n n + T_n n = 2^(n+1) + n^2 - 2) : 
  ∀ n, 2 * T_n n = n * (n - 1) :=
by sorry

end sequence_sum_identity_l103_10342


namespace ebay_ordered_cards_correct_l103_10390

noncomputable def initial_cards := 4
noncomputable def father_cards := 13
noncomputable def cards_given_to_dexter := 29
noncomputable def cards_kept := 20
noncomputable def bad_cards := 4

theorem ebay_ordered_cards_correct :
  let total_before_ebay := initial_cards + father_cards
  let total_after_giving_and_keeping := cards_given_to_dexter + cards_kept
  let ordered_before_bad := total_after_giving_and_keeping - total_before_ebay
  let ebay_ordered_cards := ordered_before_bad + bad_cards
  ebay_ordered_cards = 36 :=
by
  sorry

end ebay_ordered_cards_correct_l103_10390


namespace find_number_l103_10397

theorem find_number (x : ℕ) (h : x + 8 = 500) : x = 492 :=
by sorry

end find_number_l103_10397


namespace total_quarters_l103_10343

-- Definitions from conditions
def initial_quarters : ℕ := 49
def quarters_given_by_dad : ℕ := 25

-- Theorem to prove the total quarters is 74
theorem total_quarters : initial_quarters + quarters_given_by_dad = 74 :=
by sorry

end total_quarters_l103_10343


namespace circumcenter_rational_l103_10315

theorem circumcenter_rational {a1 b1 a2 b2 a3 b3 : ℚ} 
  (h1 : a1 ≠ a2 ∨ b1 ≠ b2) 
  (h2 : a1 ≠ a3 ∨ b1 ≠ b3) 
  (h3 : a2 ≠ a3 ∨ b2 ≠ b3) :
  ∃ (x y : ℚ), 
    (x - a1)^2 + (y - b1)^2 = (x - a2)^2 + (y - b2)^2 ∧
    (x - a1)^2 + (y - b1)^2 = (x - a3)^2 + (y - b3)^2 :=
sorry

end circumcenter_rational_l103_10315


namespace mod_remainder_7_10_20_3_20_l103_10368

theorem mod_remainder_7_10_20_3_20 : (7 * 10^20 + 3^20) % 9 = 7 := sorry

end mod_remainder_7_10_20_3_20_l103_10368


namespace simple_interest_calculation_l103_10311

-- Define the known quantities
def principal : ℕ := 400
def rate_of_interest : ℕ := 15
def time : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Statement to be proved
theorem simple_interest_calculation :
  simple_interest principal rate_of_interest time = 60 :=
by
  -- This space is used for the proof, We assume the user will complete it
  sorry

end simple_interest_calculation_l103_10311


namespace time_for_A_to_complete_race_l103_10382

open Real

theorem time_for_A_to_complete_race (V_A V_B : ℝ) (T_A : ℝ) :
  (V_B = 4) →
  (V_B = 960 / T_A) →
  T_A = 1000 / V_A →
  T_A = 240 := by
  sorry

end time_for_A_to_complete_race_l103_10382


namespace perfect_square_trinomial_m_l103_10369

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2*(m-3)*x + 16) = (x + a)^2) ↔ (m = 7 ∨ m = -1) := 
sorry

end perfect_square_trinomial_m_l103_10369


namespace triangle_properties_equivalence_l103_10333

-- Define the given properties for the two triangles
variables {A B C A' B' C' : Type}

-- Triangle side lengths and properties
def triangles_equal (b b' c c' : ℝ) : Prop :=
  (b = b') ∧ (c = c')

def equivalent_side_lengths (a a' b b' c c' : ℝ) : Prop :=
  a = a'

def equivalent_medians (ma ma' b b' c c' a a' : ℝ) : Prop :=
  ma = ma'

def equivalent_altitudes (ha ha' Δ Δ' a a' : ℝ) : Prop :=
  ha = ha'

def equivalent_angle_bisectors (ta ta' b b' c c' a a' : ℝ) : Prop :=
  ta = ta'

def equivalent_circumradii (R R' a a' b b' c c' : ℝ) : Prop :=
  R = R'

def equivalent_areas (Δ Δ' b b' c c' A A' : ℝ) : Prop :=
  Δ = Δ'

-- Main theorem statement
theorem triangle_properties_equivalence
  (b b' c c' a a' ma ma' ha ha' ta ta' R R' Δ Δ' : ℝ)
  (A A' : ℝ)
  (eq_b : b = b')
  (eq_c : c = c') :
  equivalent_side_lengths a a' b b' c c' ∧ 
  equivalent_medians ma ma' b b' c c' a a' ∧ 
  equivalent_altitudes ha ha' Δ Δ' a a' ∧ 
  equivalent_angle_bisectors ta ta' b b' c c' a a' ∧ 
  equivalent_circumradii R R' a a' b b' c c' ∧ 
  equivalent_areas Δ Δ' b b' c c' A A'
:= by
  sorry

end triangle_properties_equivalence_l103_10333


namespace range_of_a_l103_10307

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 - 3 * a * x + 9 ≤ 0) → a ∈ Set.Ico 0 4 := by
  sorry

end range_of_a_l103_10307


namespace find_dividend_l103_10383

theorem find_dividend (D Q R dividend : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) (h4 : dividend = D * Q + R) :
  dividend = 5336 :=
by
  -- We will complete the proof using the provided conditions
  sorry

end find_dividend_l103_10383


namespace diamond_3_7_l103_10309

def star (a b : ℕ) : ℕ := a^2 + 2*a*b + b^2
def diamond (a b : ℕ) : ℕ := star a b - a * b

theorem diamond_3_7 : diamond 3 7 = 79 :=
by 
  sorry

end diamond_3_7_l103_10309


namespace tangential_circle_radius_l103_10326

theorem tangential_circle_radius (R r x : ℝ) (hR : R > r) (hx : x = 4 * R * r / (R + r)) :
  ∃ x, x = 4 * R * r / (R + r) := by
sorry

end tangential_circle_radius_l103_10326


namespace value_of_D_l103_10387

variable (L E A D : ℤ)

-- given conditions
def LEAD := 41
def DEAL := 45
def ADDED := 53

-- condition that L = 15
axiom hL : L = 15

-- equations from the problem statement
def eq1 := L + E + A + D = 41
def eq2 := D + E + A + L = 45
def eq3 := A + 3 * D + E = 53

-- stating the problem as proving that D = 4 given the conditions
theorem value_of_D : D = 4 :=
by
  sorry

end value_of_D_l103_10387


namespace minimum_value_of_expression_l103_10355

theorem minimum_value_of_expression {x : ℝ} (hx : x > 0) : (2 / x + x / 2) ≥ 2 :=
by sorry

end minimum_value_of_expression_l103_10355


namespace factorize_x_squared_minus_nine_l103_10373

theorem factorize_x_squared_minus_nine : ∀ (x : ℝ), x^2 - 9 = (x - 3) * (x + 3) :=
by
  intro x
  exact sorry

end factorize_x_squared_minus_nine_l103_10373


namespace yanna_afternoon_baking_l103_10391

noncomputable def butter_cookies_in_afternoon (B : ℕ) : Prop :=
  let biscuits_afternoon := 20
  let butter_cookies_morning := 20
  let biscuits_morning := 40
  (biscuits_afternoon = B + 30) → B = 20

theorem yanna_afternoon_baking (h : butter_cookies_in_afternoon 20) : 20 = 20 :=
by {
  sorry
}

end yanna_afternoon_baking_l103_10391


namespace product_of_sequence_l103_10339

theorem product_of_sequence :
  (1 + 1 / 1) * (1 + 1 / 2) * (1 + 1 / 3) * (1 + 1 / 4) * (1 + 1 / 5) *
  (1 + 1 / 6) * (1 + 1 / 7) * (1 + 1 / 8) = 9 :=
by sorry

end product_of_sequence_l103_10339


namespace rational_sum_of_cubes_l103_10325

theorem rational_sum_of_cubes (t : ℚ) : 
    ∃ (a b c : ℚ), t = (a^3 + b^3 + c^3) :=
by
  sorry

end rational_sum_of_cubes_l103_10325


namespace no_matching_option_for_fraction_l103_10371

theorem no_matching_option_for_fraction (m n : ℕ) (h : m = 16 ^ 500) : 
  (m / 8 ≠ 8 ^ 499) ∧ 
  (m / 8 ≠ 4 ^ 999) ∧ 
  (m / 8 ≠ 2 ^ 1998) ∧ 
  (m / 8 ≠ 4 ^ 498) ∧ 
  (m / 8 ≠ 2 ^ 1994) := 
by {
  sorry
}

end no_matching_option_for_fraction_l103_10371


namespace prob_2_out_of_5_exactly_A_and_B_l103_10361

noncomputable def probability_exactly_A_and_B_selected (students : List String) : ℚ :=
  if students = ["A", "B", "C", "D", "E"] then 1 / 10 else 0

theorem prob_2_out_of_5_exactly_A_and_B :
  probability_exactly_A_and_B_selected ["A", "B", "C", "D", "E"] = 1 / 10 :=
by 
  sorry

end prob_2_out_of_5_exactly_A_and_B_l103_10361


namespace relationship_x_y_l103_10320

theorem relationship_x_y (x y m : ℝ) (h1 : x + m = 4) (h2 : y - 5 = m) : x + y = 9 := 
by 
  sorry

end relationship_x_y_l103_10320


namespace nearest_integer_pow_l103_10389

noncomputable def nearest_integer_to_power : ℤ := 
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_pow : nearest_integer_to_power = 7414 := 
  by
    unfold nearest_integer_to_power
    sorry -- Proof skipped

end nearest_integer_pow_l103_10389


namespace solution_to_eq_l103_10350

def eq1 (x y z t : ℕ) : Prop := x * y - x * z + y * t = 182
def cond_numbers (n : ℕ) : Prop := n = 12 ∨ n = 14 ∨ n = 37 ∨ n = 65

theorem solution_to_eq 
  (x y z t : ℕ) 
  (hx : cond_numbers x) 
  (hy : cond_numbers y) 
  (hz : cond_numbers z) 
  (ht : cond_numbers t) 
  (h : eq1 x y z t) : 
  (x = 12 ∧ y = 37 ∧ z = 65 ∧ t = 14) ∨ 
  (x = 37 ∧ y = 12 ∧ z = 14 ∧ t = 65) := 
sorry

end solution_to_eq_l103_10350


namespace molecular_weight_CaOH2_l103_10337

def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_CaOH2 :
  (atomic_weight_Ca + 2 * atomic_weight_O + 2 * atomic_weight_H = 74.10) := 
by 
  sorry

end molecular_weight_CaOH2_l103_10337


namespace abc_sum_71_l103_10336

theorem abc_sum_71 (a b c : ℝ) (h₁ : ∀ x, (x ≤ -3 ∨ 23 ≤ x ∧ x < 27) ↔ ( (x - a) * (x - b) / (x - c) ≥ 0)) (h₂ : a < b) : 
  a + 2 * b + 3 * c = 71 :=
sorry

end abc_sum_71_l103_10336


namespace price_difference_l103_10347

theorem price_difference (P : ℝ) :
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  difference = 0.24 * P := by
  let new_price := 1.20 * P
  let discounted_price := 0.96 * P
  let difference := new_price - discounted_price
  sorry

end price_difference_l103_10347


namespace line_segment_AB_length_l103_10310

noncomputable def length_AB (xA yA xB yB : ℝ) : ℝ :=
  Real.sqrt ((xA - xB)^2 + (yA - yB)^2)

theorem line_segment_AB_length :
  ∀ (xA yA xB yB : ℝ),
    (xA - yA = 0) →
    (xB + yB = 0) →
    (∃ k : ℝ, yA = k * (xA + 1) ∧ yB = k * (xB + 1)) →
    (-1 ≤ xA ∧ xA ≤ 0) →
    (xA + xB = 2 * k ∧ yA + yB = 2 * k) →
    length_AB xA yA xB yB = (4/3) * Real.sqrt 5 :=
by
  intros xA yA xB yB h1 h2 h3 h4 h5
  sorry

end line_segment_AB_length_l103_10310
