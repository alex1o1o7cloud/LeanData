import Mathlib

namespace triangle_areas_l192_192514

theorem triangle_areas (S₁ S₂ : ℝ) :
  ∃ (ABC : ℝ), ABC = Real.sqrt (S₁ * S₂) :=
sorry

end triangle_areas_l192_192514


namespace sum_of_longest_altitudes_l192_192964

-- Defines the sides of the triangle
def side_a : ℕ := 9
def side_b : ℕ := 12
def side_c : ℕ := 15

-- States it is a right triangle (by Pythagorean triple)
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the altitude lengths in a right triangle
def altitude_a (a b c : ℕ) (h : is_right_triangle a b c) : ℕ := a
def altitude_b (a b c : ℕ) (h : is_right_triangle a b c) : ℕ := b

-- Problem statement
theorem sum_of_longest_altitudes :
  ∃ (a b c : ℕ), is_right_triangle a b c ∧ a = side_a ∧ b = side_b ∧ c = side_c ∧
  altitude_a a b c sorry + altitude_b a b c sorry = 21 :=
by
  use side_a, side_b, side_c
  split
  sorry -- Proof that 9, 12, and 15 form a right triangle.
  split; refl
  split; refl
  sorry -- Proof that the sum of altitudes is 21.

end sum_of_longest_altitudes_l192_192964


namespace number_of_integer_values_l192_192328

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192328


namespace frequency_in_interval_l192_192585

open Real

theorem frequency_in_interval (n : ℕ) (f1 f2 f3 f4 f5 f6 : ℕ) :
  n = 20 →
  f1 = 2 →
  f2 = 3 →
  f3 = 4 →
  f4 = 5 →
  f5 = 4 →
  f6 = 2 →
  let total_freq_below_50 := f1 + f2 + f3 + f4 in
  let relative_freq_below_50 := (total_freq_below_50 : ℝ) / n in
  let freq_above_50 := 1 - relative_freq_below_50 in
  freq_above_50 = 0.3 :=
by
  intros h_n h_f1 h_f2 h_f3 h_f4 h_f5 h_f6 total_freq_below_50 relative_freq_below_50 freq_above_50
  sorry

end frequency_in_interval_l192_192585


namespace positive_divisors_of_8_factorial_l192_192832

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192832


namespace num_divisors_8_fact_l192_192865

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192865


namespace correct_statements_count_l192_192642

open Nat

noncomputable def circles (k : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - k + 1)^2 + (p.2 - 3 * k)^2 = 2 * k^4}

def statement_1 : Prop := ∃ (L : ℝ → ℝ), ∀ k ∈ ℕ, is_tangent_to_circle L (circles k)
def statement_2 : Prop := ∃ (L : ℝ → ℝ), ∀ k ∈ ℕ, intersects_circle L (circles k)
def statement_3 : Prop := ∃ (L : ℝ → ℝ), ∀ k ∈ ℕ, ∀ (p ∈ circles k), L p = False
def statement_4 : Prop := ∀ k ∈ ℕ, (0, 0) ∉ circles k

def number_of_correct_statements : ℕ :=
  [statement_1, statement_2, statement_3, statement_4].count (λ st, st)

theorem correct_statements_count : number_of_correct_statements = 2 :=
  sorry

end correct_statements_count_l192_192642


namespace floor_abs_T_eq_3015_l192_192467

theorem floor_abs_T_eq_3015 (x : Fin 2010 → ℝ) 
  (h : ∀ n : Fin 2010, (x n) + (↑n + 1) = ∑ i : Fin 2010, x i + 4020) : 
  Int.floor (abs (∑ i : Fin 2010, x i)) = 3015 := 
by sorry

end floor_abs_T_eq_3015_l192_192467


namespace partial_fraction_decomposition_l192_192453

theorem partial_fraction_decomposition (n : ℕ) (x : ℤ) :
  (n.factorial : ℤ) / (list.range (n + 1)).prod (λ i, x + i) =
  ∑ k in (finset.range (n + 1)), (-1 : ℤ) ^ k * (nat.choose n k : ℤ) / (x + k) :=
by {
  sorry
}

end partial_fraction_decomposition_l192_192453


namespace grandma_vasya_cheapest_option_l192_192246

/-- Constants and definitions for the cost calculations --/
def train_ticket_cost : ℕ := 200
def collected_berries_kg : ℕ := 5
def market_berries_cost_per_kg : ℕ := 150
def sugar_cost_per_kg : ℕ := 54
def jam_made_per_kg_combination : ℕ := 15 / 10  -- representing 1.5 kg (as ratio 15/10)
def ready_made_jam_cost_per_kg : ℕ := 220

/-- Compute the cost per kg of jam for different methods --/
def cost_per_kg_jam_option1 : ℕ := (train_ticket_cost / collected_berries_kg + sugar_cost_per_kg)
def cost_per_kg_jam_option2 : ℕ := market_berries_cost_per_kg + sugar_cost_per_kg
def cost_per_kg_jam_option3 : ℕ := ready_made_jam_cost_per_kg

/-- Numbers converted to per 1.5 kg --/
def cost_for_1_5_kg (cost_per_kg: ℕ) : ℕ := cost_per_kg * (15 / 10)

/-- Theorem stating option 1 is the cheapest --/
theorem grandma_vasya_cheapest_option :
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option2 ∧
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option3 :=
by sorry

end grandma_vasya_cheapest_option_l192_192246


namespace two_vectors_less_than_45_deg_angle_l192_192373

theorem two_vectors_less_than_45_deg_angle (n : ℕ) (h : n = 30) (v : Fin n → ℝ → ℝ → ℝ) :
  ∃ i j : Fin n, i ≠ j ∧ ∃ θ : ℝ, θ < (45 * Real.pi / 180) :=
  sorry

end two_vectors_less_than_45_deg_angle_l192_192373


namespace range_of_a_l192_192364

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 := by
  sorry

end range_of_a_l192_192364


namespace number_of_integer_values_l192_192275

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192275


namespace hydrogen_atoms_in_compound_l192_192568

theorem hydrogen_atoms_in_compound :
  ∀ (H_atoms Br_atoms O_atoms total_molecular_weight weight_H weight_Br weight_O : ℝ),
  Br_atoms = 1 ∧ O_atoms = 3 ∧ total_molecular_weight = 129 ∧ 
  weight_H = 1 ∧ weight_Br = 79.9 ∧ weight_O = 16 →
  H_atoms = 1 :=
by
  sorry

end hydrogen_atoms_in_compound_l192_192568


namespace incorrect_pair_l192_192031

def roots_of_polynomial (x : ℝ) : Prop := x^2 - 3*x + 2 = 0

theorem incorrect_pair : ¬ ∃ x : ℝ, (y = x - 1 ∧ y = x + 1 ∧ roots_of_polynomial x) :=
by
  sorry

end incorrect_pair_l192_192031


namespace num_pos_divisors_fact8_l192_192774

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192774


namespace geom_prog_235_l192_192140

theorem geom_prog_235 (q : ℝ) (k n : ℕ) (hk : 1 < k) (hn : k < n) : 
  ¬ (q > 0 ∧ q ≠ 1 ∧ 3 = 2 * q^(k - 1) ∧ 5 = 2 * q^(n - 1)) := 
by 
  sorry

end geom_prog_235_l192_192140


namespace total_bricks_used_l192_192442

def numberOfCoursesPerWall := 6
def bricksPerCourse := 10
def numberOfWalls := 4
def incompleteCourses := 2

theorem total_bricks_used :
  (numberOfCoursesPerWall * bricksPerCourse * (numberOfWalls - 1)) + ((numberOfCoursesPerWall - incompleteCourses) * bricksPerCourse) = 220 :=
by
  -- Proof goes here
  sorry

end total_bricks_used_l192_192442


namespace probability_elena_harry_diagonal_l192_192374

-- Define our setup and problem
def seating_arrangement : Type :=
  {p : Fin 4 → Fin 4 // bijective p}

def count_total_arrangements : ℕ :=
  factorial 4

def count_favorable_arrangements (E H F G : Fin 4) : ℕ :=
  if (E = 0 ∧ H = 2) ∨ (E = 2 ∧ H = 0) ∨ (E = 1 ∧ H = 3) ∨ (E = 3 ∧ H = 1) then 4 else 0

def probability_diagonal_opposite (E H F G : Fin 4) : ℚ :=
  if h : count_total_arrangements ≠ 0 then
    (count_favorable_arrangements E H F G) / (count_total_arrangements : ℚ) else 0

-- Prove that the probability is indeed 2/3
theorem probability_elena_harry_diagonal (E H F G : Fin 4) :
  probability_diagonal_opposite E H F G = 2 / 3 := by
  simp [count_total_arrangements, count_favorable_arrangements, probability_diagonal_opposite]
  sorry

end probability_elena_harry_diagonal_l192_192374


namespace polynomial_root_sum_l192_192021

theorem polynomial_root_sum : 
  ∀ (r1 r2 r3 r4 : ℝ), 
  (r1^4 - r1 - 504 = 0) ∧ 
  (r2^4 - r2 - 504 = 0) ∧ 
  (r3^4 - r3 - 504 = 0) ∧ 
  (r4^4 - r4 - 504 = 0) → 
  r1^4 + r2^4 + r3^4 + r4^4 = 2016 := by
sorry

end polynomial_root_sum_l192_192021


namespace ellipse_standard_form_and_fixed_points_l192_192706

variables {a b c : ℝ} {A B M F1 F2 : ℝ × ℝ}
variables {l : ℝ → ℝ}

-- Given conditions
def ellipse (a b : ℝ) : Prop :=
a > b ∧ ∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

def short_axis_length (b : ℝ) : Prop :=
2 * b = 2

def distance_left_vertex_left_focus (a c : ℝ) : Prop :=
a - c = sqrt 2 - 1

def point_M : Prop :=
M = (0, 1 / 4)

def line_l_intersection_points (A B : ℝ × ℝ) (l : ℝ → ℝ) : Prop :=
∃ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | x^2 / 2 + y^2 = 1} ∧ y = l x

def constant_dot_product (A B M : ℝ × ℝ) : Prop :=
(0, 1 / 4) = M ∧ 
∃ k : ℝ, ∀ A B : ℝ × ℝ, (A ≠ B) → (A.1 = 0 ∨ B.1 = 0) → 
   ((A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = k)

-- Prove the standard equation of ellipse and fixed points with constant value
theorem ellipse_standard_form_and_fixed_points 
  (hc : ellipse a b)
  (hl : short_axis_length b)
  (hd : distance_left_vertex_left_focus a c)
  (hp : point_M)
  (line_cond : line_l_intersection_points A B l)
  (const_dot : constant_dot_product A B M) : 
  (a = sqrt 2 ∧ b = 1 ∧ c = 1 ∧ 
  (const_dot → l 0 = -1/2 ∨ l 0 = 2/3 ∧ k = -15/16)) :=
sorry

end ellipse_standard_form_and_fixed_points_l192_192706


namespace num_divisors_8_factorial_l192_192768

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192768


namespace perp_tangent_points_abscissa_l192_192226

theorem perp_tangent_points_abscissa :
  let curve1 (x : ℝ) := x * real.log x
      curve2 (x : ℝ) := 4 / x
      derivative1 (x : ℝ) := 1 + real.log x
      derivative2 (x : ℝ) := -4 / x^2
      P := (x_0, y_0)
      tangent1_slope := 1 -- slope of tangent at (1, 0) of curve1
      tangent2_slope := derivative2 x_0
  in (1 * tangent2_slope = -1) → (x_0^2 = 4) → (x_0 = 2 ∨ x_0 = -2) :=
by
  sorry

end perp_tangent_points_abscissa_l192_192226


namespace num_divisors_8_factorial_l192_192919

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192919


namespace no_real_solution_for_k_l192_192166

theorem no_real_solution_for_k : 
  ∀ k : ℝ, ∥k • ⟨3, -4⟩ - ⟨5, 8⟩∥ ≠ 3 * real.sqrt 13 :=
by
  intro k
  simp only [sub_eq_add_neg, smul_neg, norm_eq_sqrt_inner, real_inner, sq, norm]
  sorry

end no_real_solution_for_k_l192_192166


namespace count_positive_integers_l192_192184

theorem count_positive_integers (n : ℕ) :
    {x : ℕ | (x > 0) ∧ (20 < x^2 + 6*x + 9) ∧ (x^2 + 6*x + 9 < 40)}.card = 2 :=
by
  sorry

end count_positive_integers_l192_192184


namespace min_combinations_to_open_locker_l192_192111

theorem min_combinations_to_open_locker :
  (∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (n.to_string.contains "23") ∧ (n.to_string.contains "37")) →
  ∃ (k : ℕ), k = 356 := by
  sorry

end min_combinations_to_open_locker_l192_192111


namespace caterpillar_weight_gain_in_days_l192_192040

noncomputable def time_to_gain_weight (max_weight : ℕ) : ℕ :=
  ∑ n in finset.range (max_weight + 1), n * 2 ^ n / 2

theorem caterpillar_weight_gain_in_days : time_to_gain_weight 10 = 9217 := by
  sorry

end caterpillar_weight_gain_in_days_l192_192040


namespace number_of_divisors_of_8_fact_l192_192892

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192892


namespace proof_problem_l192_192239

-- Define a universe of discourse for animals
universe u
constant Animal : Type u

-- Define predicates for being a cat, being a dog, being playful, and being lazy
constant Cat : Animal → Prop
constant Dog : Animal → Prop
constant Playful : Animal → Prop
constant Lazy : Animal → Prop

-- Given conditions
axiom all_cats_playful : ∀ x : Animal, Cat x → Playful x
axiom some_cats_playful : ∃ x : Animal, Cat x ∧ Playful x
axiom no_dogs_playful : ∀ x : Animal, Dog x → ¬ Playful x
axiom all_dogs_lazy : ∀ x : Animal, Dog x → Lazy x
axiom at_least_one_dog_lazy : ∃ x : Animal, Dog x ∧ Lazy x

-- Statement (6) that we need to negate
constant all_dogs_playful : ∀ x : Animal, Dog x → Playful x

-- The goal: prove that no_dogs_playful negates all_dogs_playful
theorem proof_problem : ∀ x : Animal, Dog x → ¬ Playful x := by
  exact no_dogs_playful

end proof_problem_l192_192239


namespace ratio_of_boys_l192_192995

theorem ratio_of_boys (p : ℝ) (h : p = (3 / 4) * (1 - p)) :
  p = 3 / 7 :=
by
  have h1: p + (3 / 4) * p = 3 / 4 := by 
    rw [h, add_mul, one_mul, add_sub_assoc]
  have h2: (7 / 4) * p = 3 / 4 := by
    rw [←add_div, mul_div_cancel' _ (ne_of_gt (by norm_num))]
  have h3: p = 3 / 7 := by
    rw [←mul_div_assoc, ←div_div, div_self (ne_of_gt (by norm_num)), mul_one]
  exact h3

end ratio_of_boys_l192_192995


namespace find_ordered_pair_l192_192492

-- We need to define the variables and conditions first.
variables (a c : ℝ)

-- Now we state the conditions.
def quadratic_has_one_solution : Prop :=
  a * c = 25 ∧ a + c = 12 ∧ a < c

-- Finally, we state the main goal to prove.
theorem find_ordered_pair (ha : quadratic_has_one_solution a c) :
  a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11 :=
by sorry

end find_ordered_pair_l192_192492


namespace percent_time_in_meetings_l192_192433

-- Define the conditions
def work_day_minutes : ℕ := 10 * 60  -- Total minutes in a 10-hour work day is 600 minutes
def first_meeting_minutes : ℕ := 60  -- The first meeting took 60 minutes
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes  -- The second meeting took three times as long as the first meeting

-- Total time spent in meetings
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes  -- 60 + 180 = 240 minutes

-- The task is to prove that Makarla spent 40% of her work day in meetings.
theorem percent_time_in_meetings : (total_meeting_minutes / work_day_minutes : ℚ) * 100 = 40 := by
  sorry

end percent_time_in_meetings_l192_192433


namespace minimum_sum_of_arithmetic_sequence_l192_192380

noncomputable def a_n (a_1 d : ℝ) (n : ℕ) : ℝ := a_1 + (n - 1) * d

theorem minimum_sum_of_arithmetic_sequence
  (a_1 d : ℝ)
  (pos_a1 : 0 < a_1)
  (pos_d : 0 < d)
  (h : (a_n a_1 d 4) * (a_n a_1 d 9) = 36) :
  72 ≤ 12 * (a_1 + a_n a_1 d 12) :=
begin
  sorry
end

end minimum_sum_of_arithmetic_sequence_l192_192380


namespace sqrt_floor_8_integer_count_l192_192321

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192321


namespace number_of_integer_values_l192_192326

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192326


namespace prob_conditional_l192_192491

def prob_live_to_20 : ℝ := 0.8
def prob_live_to_25 : ℝ := 0.5
def prob_live_to_25_given_20 : ℝ := prob_live_to_25 / prob_live_to_20

theorem prob_conditional :
  prob_live_to_25_given_20 = 0.625 :=
by sorry

end prob_conditional_l192_192491


namespace work_completion_l192_192098

theorem work_completion (B_completion_days : ℕ) (B_work_rate : ℚ) (A_work_days : ℕ) (B_remaining_work_days : ℕ) :
  B_completion_days = 30 →
  B_work_rate = 1 / 30 →
  A_work_days = 10 →
  B_remaining_work_days = 15 →
  ∃ A_completion_days : ℚ, A_completion_days = 20 :=
by
  intros hB_completion_days hB_work_rate hA_work_days hB_remaining_work_days
  use 20
  sorry

end work_completion_l192_192098


namespace trigonometric_identity_l192_192637

theorem trigonometric_identity :
  (tan (40 * Real.pi / 180))^2 - (sin (40 * Real.pi / 180))^2 / 
  ((tan (40 * Real.pi / 180))^2 * (sin (40 * Real.pi / 180))^2) = 1 :=
by
  sorry

end trigonometric_identity_l192_192637


namespace determine_set_M_l192_192505

universe u

variable (U : set ℕ) (M : set ℕ)
variable (U_def : U = {0, 1, 2, 3})
variable (comp_M : (U \ M) = {2})

theorem determine_set_M : M = {0, 1, 3} :=
by
  sorry

end determine_set_M_l192_192505


namespace find_a_l192_192627

-- Define the positive constant b
variables (b : ℝ) (hb : 0 < b)

-- Define the function y = a * csc(b * x)
def y (a b x : ℝ) := a * (1 / Real.sin(b * x))

-- The condition from the graph that the minimum positive value when y > 0 is 2
theorem find_a (a : ℝ) (ha_pos : 0 < a) (hx : ∃ x : ℝ, 0 < y a b x ∧ y a b x = 2) : a = 2 :=
  sorry

end find_a_l192_192627


namespace num_divisors_8_factorial_l192_192921

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192921


namespace num_positive_divisors_8_factorial_l192_192942

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192942


namespace num_whole_numbers_between_cuberoots_l192_192256

open Real

theorem num_whole_numbers_between_cuberoots : 
  (20 : ℝ) ^ (1 / 3) < 3 ∧ 6 < (300 : ℝ) ^ (1 / 3) → 
  ∃ (k : ℕ), k = 4 :=
by 
  intros h
  obtain ⟨h1, h2⟩ := h
  have h3 : (3 : ℝ) ≤ (300 : ℝ) ^ (1 / 3),
  { sorry }
  have h4 : (20 : ℝ) ^ (1 / 3) ≤ 2,
  { sorry }
  use k = 4
  sorry

end num_whole_numbers_between_cuberoots_l192_192256


namespace count_possible_integer_values_l192_192285

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192285


namespace find_principal_l192_192551

/-
Conditions:
1. Equal amounts of money, denoted P
2. Both banks have an interest rate of 15% per annum
3. Time for first bank: 3.5 years
4. Time for second bank: 10 years
5. Difference in interests: Rs. 144
-/

def interest1 (P : ℝ) : ℝ := (P * 15 * 3.5) / 100
def interest2 (P : ℝ) : ℝ := (P * 15 * 10) / 100

theorem find_principal (P : ℝ) : interest2 P - interest1 P = 144 → P = 14400 / 97.5 :=
by
  intro h
  have h1 : interest1 P = (P * 52.5) / 100 := by sorry
  have h2 : interest2 P = (P * 150) / 100 := by sorry
  have h3 : (P * 150) / 100 - (P * 52.5) / 100 = 144 := by sorry
  have h4 : (97.5 * P) / 100 = 144 := by sorry
  exact congrArg ((· / (97.5 * 1))) (by sorry)

end find_principal_l192_192551


namespace projection_of_a_on_b_l192_192694

-- Definitions for vectors a and b
def vector_a : ℝ × ℝ := (2, 1)
def vector_b : ℝ × ℝ := (3, 4)

-- Dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Magnitude function
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Projection function
def projection (a b : ℝ × ℝ) : ℝ := dot_product a b / magnitude b

-- Proof statement
theorem projection_of_a_on_b : projection vector_a vector_b = 2 :=
by
  -- Add the proof here
  sorry

end projection_of_a_on_b_l192_192694


namespace cistern_problem_l192_192105

theorem cistern_problem (fill_rate empty_rate net_rate : ℝ) (T : ℝ) : 
  fill_rate = 1 / 3 →
  net_rate = 7 / 30 →
  empty_rate = 1 / T →
  net_rate = fill_rate - empty_rate →
  T = 10 :=
by
  intros
  sorry

end cistern_problem_l192_192105


namespace num_positive_divisors_8_factorial_l192_192938

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192938


namespace machine_tool_comparison_l192_192524

def defective_A : List ℕ := [1, 0, 2, 0, 2]
def defective_B : List ℕ := [1, 0, 1, 0, 3]

-- Calculate the total pairs for machine tool A
def total_pairs_A : List (ℕ × ℕ) :=
  [(1, 0), (1, 2), (1, 0), (1, 2), (0, 2), (0, 0), (0, 2), (2, 0), (2, 2), (0, 2)]

-- Calculate the number of pairs where the defective count is ≤ 1
def favorable_pairs_A : List (ℕ × ℕ) :=
  [(1, 0), (1, 0), (0, 0)]

-- Calculate mean of defective components
def mean (l : List ℕ) : ℝ :=
  (l.sum.toReal) / (l.length.toReal)

-- Calculate variance of defective components
def variance (l : List ℕ) : ℝ :=
  let m := mean l in
  (l.map (λ x => (x - m)^2)).sum.toReal / (l.length.toReal)

-- Prove the probability and variance comparison
theorem machine_tool_comparison :
  (favorable_pairs_A.length = 3 ∧ total_pairs_A.length = 10 ∧
   (favorable_pairs_A.length.toReal / total_pairs_A.length.toReal) = 3 / 10) ∧
  variance defective_A < variance defective_B :=
by
  sorry

end machine_tool_comparison_l192_192524


namespace smithNumberCount_l192_192982

def isSmithNumber (n : ℕ) : Prop :=
  let primeFactorsSumDigits := (∑ p in (Nat.factorization n).keys, digitSum p)
  primeFactorsSumDigits = digitSum n

def digitSum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum 

def numSmithNumbers (lst : List ℕ) : ℕ :=
  lst.filter isSmithNumber |>.length

theorem smithNumberCount : numSmithNumbers [4, 32, 58, 65, 94] = 3 :=
by
  sorry

end smithNumberCount_l192_192982


namespace more_than_10_numbers_l192_192001

theorem more_than_10_numbers (seq : List ℕ) 
  (sum_eq_20 : seq.sum = 20) 
  (no_num_or_sum_eq_3 : ∀ n, n ∈ seq → n ≠ 3 ∧ 
    ∀ (start len : ℕ), start + len ≤ seq.length → (seq.slice start len).sum ≠ 3) :
  seq.length > 10 :=
  sorry

end more_than_10_numbers_l192_192001


namespace degree_of_product_l192_192008

-- Define the degrees of the polynomials
def degree_f : ℕ := 3
def degree_g : ℕ := 4

-- State the main theorem
theorem degree_of_product (f g : ℕ → ℕ) (degree_f : ℕ) (degree_g : ℕ) (h1 : degree f = degree_f) (h2 : degree g = degree_g) : 
  degree (λ x, f (x * x) * g (x * x * x)) = 18 := 
sorry

end degree_of_product_l192_192008


namespace arithmetic_sequence_first_term_l192_192419

theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (k : ℕ) (hk : k ≥ 2)
  (hS : S k = 5)
  (ha_k2_p1 : a (k^2 + 1) = -45)
  (ha_sum : (Finset.range (2 * k + 1) \ Finset.range (k + 1)).sum a = -45) :
  a 1 = 5 := 
sorry

end arithmetic_sequence_first_term_l192_192419


namespace num_subsets_pos_reals_of_A_l192_192236

noncomputable def A : Set ℂ := {1 / (2 * complex.I), complex.I ^ 2, abs (5 * (complex.I ^ 2)), (1 + complex.I ^ 2) / complex.I, -complex.I ^ 2 / 2}
noncomputable def PositiveReals : Set ℝ := {x | 0 < x}

theorem num_subsets_pos_reals_of_A : 
  let intersection := {x : ℂ | x ∈ A ∧ x.re > 0}
  ∃ (n : ℕ), n = 4 :=
begin
  sorry
end

end num_subsets_pos_reals_of_A_l192_192236


namespace triangle_largest_angle_l192_192153

theorem triangle_largest_angle (u : ℝ) (h1 : 4u - 2 > 0) (h2 : 4u + 2 > 0) (h3 : 2u > 0) :
  let a := Real.sqrt (4u - 2)
  let b := Real.sqrt (4u + 2)
  let c := 2 * Real.sqrt (2u)
  a^2 + b^2 = c^2 →
  ∃ θ : ℝ, θ = 90 :=
by
  sorry

end triangle_largest_angle_l192_192153


namespace num_divisors_8_factorial_l192_192929

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192929


namespace pos_divisors_8_factorial_l192_192910

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192910


namespace double_integral_parabolas_eqn_l192_192137

-- the double integral of (x + 2y) over D, where D is bounded by the given parabolas and the y-axis, equals 2/3
theorem double_integral_parabolas_eqn : 
  let D := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ (p.1 - p.1^2) ≤ p.2 ∧ p.2 ≤ (1 - p.1^2)}
  ∫∫ (x, y) in D, x + 2 * y = 2 / 3 :=
by
  sorry

end double_integral_parabolas_eqn_l192_192137


namespace sum_of_longest_altitudes_l192_192963

-- Defines the sides of the triangle
def side_a : ℕ := 9
def side_b : ℕ := 12
def side_c : ℕ := 15

-- States it is a right triangle (by Pythagorean triple)
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the altitude lengths in a right triangle
def altitude_a (a b c : ℕ) (h : is_right_triangle a b c) : ℕ := a
def altitude_b (a b c : ℕ) (h : is_right_triangle a b c) : ℕ := b

-- Problem statement
theorem sum_of_longest_altitudes :
  ∃ (a b c : ℕ), is_right_triangle a b c ∧ a = side_a ∧ b = side_b ∧ c = side_c ∧
  altitude_a a b c sorry + altitude_b a b c sorry = 21 :=
by
  use side_a, side_b, side_c
  split
  sorry -- Proof that 9, 12, and 15 form a right triangle.
  split; refl
  split; refl
  sorry -- Proof that the sum of altitudes is 21.

end sum_of_longest_altitudes_l192_192963


namespace july14_2030_is_sunday_l192_192401

-- Define the given condition that July 3, 2030 is a Wednesday. 
def july3_2030_is_wednesday : Prop := true -- Assume the existence and correctness of this statement.

-- Define the proof problem that July 14, 2030 is a Sunday given the above condition.
theorem july14_2030_is_sunday : july3_2030_is_wednesday → (14 % 7 = 0) := 
sorry

end july14_2030_is_sunday_l192_192401


namespace pyramid_volume_l192_192626

theorem pyramid_volume (a b : ℝ) (φ : ℝ) (h : 0 < φ < π)
  (ha : a > 2 * b) :
  let h_base : ℝ := sqrt ((a - b) * (a + b)),
      S_base : ℝ := 0.5 * (a + b) * h_base,
      tan_phi_half := tan (φ / 2),
      FP : ℝ := (sqrt (a * (a - 2 * b)) * tan_phi_half) / 2,
      V := (1 / 3) * S_base * FP
  in V = ((a + b) ^ 2 * (tan_phi_half ^ 2) * sqrt (a * (a - 2 * b))) / 24 :=
by
  sorry

end pyramid_volume_l192_192626


namespace problem_solution_l192_192711

-- Define f(x) under the given conditions
def f (x : ℝ) : ℝ := 
  if x ∈ set.Icc 0 2 then (Real.exp x - 1)
  else if x < 0 then -f (-x)
  else f (x - 2 * (Real.floor (x / 2) : ℝ))

-- State the theorem
theorem problem_solution : f 2013 + f (-2014) = Real.exp 1 - 1 :=
by {
  sorry
}

end problem_solution_l192_192711


namespace isosceles_triangles_with_perimeter_25_l192_192742

/-- Prove that there are 6 distinct isosceles triangles with integer side lengths 
and a perimeter of 25 -/
theorem isosceles_triangles_with_perimeter_25 :
  ∃ (count : ℕ), 
    count = 6 ∧ 
    (∀ (a b : ℕ), 
      let a1 := a,
          a2 := a,
          b3 := b in
      2 * a + b = 25 → 
      2 * a > b ∧ a + b > a ∧ b < 2 * a ∧
      a > 0 ∧ b > 0 ∧ a ∈ finset.Icc 7 12) :=
by sorry

end isosceles_triangles_with_perimeter_25_l192_192742


namespace number_of_good_sets_l192_192570

def is_good_set (S : set ℕ) : Prop :=
  ∑ k in S, 2^k = 2004

noncomputable def good_sets_num : ℕ :=
  (∑ S in (finset.powerset (finset.range 11)), 
   if is_good_set S.to_set then 1 else 0)

theorem number_of_good_sets :
  good_sets_num = 1006009 :=
  sorry

end number_of_good_sets_l192_192570


namespace num_divisors_8_factorial_l192_192928

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192928


namespace summation_inequality_l192_192164

theorem summation_inequality (n : ℕ) (h : n > 1)
  (x : Fin n → ℝ) : 
  (∑ i : Fin n, x i ^ 2 ≥ x (Fin.last n) * ∑ i : Fin (n - 1), x i) ↔ n ≤ 5 := 
sorry

end summation_inequality_l192_192164


namespace angle_equal_l192_192450

-- Define the conditions
variables {A B C A' B' C' H M : Type}

-- Points A', B', C' are the midpoints of sides BC, CA, and AB of triangle ABC, respectively
def is_midpoint (P Q R : Type) : Prop := sorry -- placeholder definition

axiom midpoints : is_midpoint B C A' ∧ is_midpoint C A B' ∧ is_midpoint A B C'

-- BH is the altitude of triangle ABC
def is_altitude (B H C : Type) : Prop := sorry -- placeholder definition

axiom altitude_BH : is_altitude B H C

-- The circumcircles of triangles AHC' and CHA' pass through point M
def passes_through_circumcircle (T M : Type) : Prop := sorry -- placeholder definition

axiom circumcircles_M : passes_through_circumcircle (A, H, C') M ∧ passes_through_circumcircle (C, H, A') M

-- Prove that angle ABM = angle CBB'
theorem angle_equal (A B C A' B' C' H M : Type) :
  midpoints ∧ altitude_BH ∧ circumcircles_M →
  ∠ ABM = ∠ CBB' :=
sorry

end angle_equal_l192_192450


namespace number_of_integers_in_double_inequality_l192_192751

noncomputable def pi_approx : ℝ := 3.14
noncomputable def sqrt_pi_approx : ℝ := Real.sqrt pi_approx
noncomputable def lower_bound : ℝ := -12 * sqrt_pi_approx
noncomputable def upper_bound : ℝ := 15 * pi_approx

theorem number_of_integers_in_double_inequality : 
  ∃ n : ℕ, n = 13 ∧ ∀ k : ℤ, lower_bound ≤ (k^2 : ℝ) ∧ (k^2 : ℝ) ≤ upper_bound → (-6 ≤ k ∧ k ≤ 6) :=
by
  sorry

end number_of_integers_in_double_inequality_l192_192751


namespace perfect_cube_die_roll_l192_192595

theorem perfect_cube_die_roll :
  let p := 1 in let q := 36 in (p + q = 37) :=
by
  sorry

end perfect_cube_die_roll_l192_192595


namespace number_of_divisors_8_factorial_l192_192794

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192794


namespace num_divisors_of_8_factorial_l192_192866

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192866


namespace num_divisors_8_factorial_l192_192763

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192763


namespace find_values_a_b_l192_192688

theorem find_values_a_b :
  let a b : ℝ
  in (|a| + |b| = (2 / Real.sqrt 3)) ∧ (a = 2 * b ∨ a = -2 * b) →
  (a = 4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) ∨ 
  (a = -4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨ 
  (a = 4 / (3 * Real.sqrt 3) ∧ b = -2 / (3 * Real.sqrt 3)) ∨ 
  (a = -4 / (3 * Real.sqrt 3) ∧ b = 2 / (3 * Real.sqrt 3)) :=
begin
  sorry
end

end find_values_a_b_l192_192688


namespace frankie_pets_l192_192692

variable {C S P D : ℕ}

theorem frankie_pets (h1 : S = C + 6) (h2 : P = C - 1) (h3 : C + D = 6) (h4 : C + S + P + D = 19) : 
  C + S + P + D = 19 :=
  by sorry

end frankie_pets_l192_192692


namespace trapezoid_axyd_relationship_l192_192547

theorem trapezoid_axyd_relationship (a b : ℝ) (h_square : ∀ (P : ℝ), P ∈ {a, b} -> (4032 - P) ∉ {a, b})
  (h_area : 2016 = (1 / 2) * (4032 + a + (4032 - b)) * 4032) : b - a = 8063 := sorry

end trapezoid_axyd_relationship_l192_192547


namespace num_positive_divisors_8_factorial_l192_192934

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192934


namespace erick_total_earnings_l192_192060

theorem erick_total_earnings
    (original_lemon_price : ℕ)
    (lemon_price_increase : ℕ)
    (original_grape_price : ℕ)
    (lemons_count : ℕ)
    (grapes_count : ℕ) :
    let new_lemon_price := original_lemon_price + lemon_price_increase,
        total_lemons_earning := lemons_count * new_lemon_price,
        grape_price_increase := lemon_price_increase / 2,
        new_grape_price := original_grape_price + grape_price_increase,
        total_grapes_earning := grapes_count * new_grape_price,
        total_earning := total_lemons_earning + total_grapes_earning
    in total_earning = 2220 := by
  sorry

end erick_total_earnings_l192_192060


namespace euler_formula_planar_graph_edge_bound_planar_graph_no_triangle_bound_l192_192977

-- Definition of a planar graph and its properties
axiom planar_graph (G : Type) : Prop

-- Definitions of vertices, edges, and faces
noncomputable def vertices (G : Type) : ℕ := sorry
noncomputable def edges (G : Type) : ℕ := sorry
noncomputable def faces (G : Type) : ℕ := sorry

-- Euler's formula statement
theorem euler_formula {G : Type} (h : planar_graph G) : 
  vertices G - edges G + faces G = 2 :=
sorry

-- Additional inequalities for planar graphs
theorem planar_graph_edge_bound {G : Type} (h : planar_graph G) : 
  edges G ≤ 3 * vertices G - 6 :=
sorry

theorem planar_graph_no_triangle_bound {G : Type} (h : planar_graph G) 
  (no_triangles : ∀ f, ¬ (∃ (x y z : vertices G), 
    -- Predicate here asserting that any face f with vertices x, y, z does not form a triangle
    sorry)) : edges G ≤ 2 * vertices G - 4 :=
sorry

end euler_formula_planar_graph_edge_bound_planar_graph_no_triangle_bound_l192_192977


namespace apples_left_over_proof_l192_192249

noncomputable def greg_apples : ℝ := 9
noncomputable def sarah_apples : ℝ := 9
noncomputable def susan_apples : ℝ := 2 * greg_apples
noncomputable def mark_apples : ℝ := susan_apples - 5
noncomputable def emily_apples : ℝ := sqrt mark_apples + 3/2
noncomputable def total_apples : ℝ := greg_apples + sarah_apples + susan_apples + mark_apples + emily_apples
noncomputable def apples_needed_for_pie : ℝ := 38.5
noncomputable def apples_left_over : ℝ := total_apples - apples_needed_for_pie

theorem apples_left_over_proof : apples_left_over ≈ 15.61 := by
  sorry

end apples_left_over_proof_l192_192249


namespace number_of_divisors_8_factorial_l192_192798

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192798


namespace sum_of_angles_DAC_and_ADE_l192_192523

theorem sum_of_angles_DAC_and_ADE :
  ∀ (A B C D E F : Type)
  (AB AC DE DF : ℝ)
  (BAC EDF DAC ADE : ℝ),
  is_isosceles_triangle AB AC ∧ is_isosceles_triangle DE DF ∧
  BAC = 25 ∧ EDF = 40 →
  DAC = (180 - BAC) / 2 ∧ ADE = (180 - EDF) / 2 →
  DAC + ADE = 147.5 := 
by 
  sorry

def is_isosceles_triangle (x y : ℝ) : Prop := x = y

end sum_of_angles_DAC_and_ADE_l192_192523


namespace solution_theorem_l192_192165

open Nat

noncomputable def solution_prob : List (ℕ × ℕ) :=
[(0, n) | n ∈ List.range 10000] ++ [(1, n) | n ∈ List.range 10000]

theorem solution_theorem (m n : ℕ) :
  (1 + (m + n) * m) ∣ ((n + 1) * (m + n) - 1) ↔ (m = 0 ∧ n ∈ ℕ) ∨ (m = 1 ∧ n ∈ ℕ) :=
by
  sorry

end solution_theorem_l192_192165


namespace fraction_of_ninth_triangle_shaded_l192_192987

theorem fraction_of_ninth_triangle_shaded :
  let shaded (n : ℕ) := 2 * n - 1
  let total (n : ℕ) := 4 ^ (n - 1)
  fraction (9 : ℕ) := 17 / 65536
  (shaded 9 / total 9) = fraction 9 := 
sorry

end fraction_of_ninth_triangle_shaded_l192_192987


namespace sum_sublist_eq_100_l192_192501

theorem sum_sublist_eq_100 {l : List ℕ}
  (h_len : l.length = 2 * 31100)
  (h_max : ∀ x ∈ l, x ≤ 100)
  (h_sum : l.sum = 200) :
  ∃ (s : List ℕ), s ⊆ l ∧ s.sum = 100 := 
sorry

end sum_sublist_eq_100_l192_192501


namespace two_real_values_for_k_l192_192649

-- Define the problem conditions
variables (a b c : ℝ)

-- Statement of the problem
theorem two_real_values_for_k (a b c : ℝ) :
  ∃ k1 k2 : ℝ, (a * x^2 + b * x + c + k * (x^2 + 1) = ((v: ℝ) -> v ^ 2)) 
    ∧ (4 * k^2 + 4 * (a + c) * k + 4 * a * c - b^2 = 0) := 
sorry

end two_real_values_for_k_l192_192649


namespace prob_divisors_8_fact_l192_192817

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192817


namespace num_divisors_8_fact_l192_192857

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192857


namespace correct_quadratic_eq_l192_192376

theorem correct_quadratic_eq 
  (h1 : ∃ b c, (2 * x^2 + b * x + c = 0 ∧ {root1 := 3, root2 := 1})) 
  (h2 : ∃ b c, (x^2 + b * x + c = 0 ∧ {root1 := -6, root2 := -2})) 
  : x^2 - 8x + 12 = 0 :=
sorry

end correct_quadratic_eq_l192_192376


namespace prob_divisors_8_fact_l192_192810

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192810


namespace orthocentric_tetrahedron_l192_192994

theorem orthocentric_tetrahedron 
  (a b c d : EuclideanSpace ℝ (Fin 3))
  (medians_bases_midpoints_form_equal_edges_polyhedron : 
      ∀ (x y : EuclideanSpace ℝ (Fin 3)), 
      (∃ m1 m2 m3 m4 : EuclideanSpace ℝ (Fin 3),
      m1 = (a + b + c) / 3 ∨ m1 = (a + b + d) / 3 ∨ m1 = (a + c + d) / 3 ∨ m1 = (b + c + d) / 3
      ∧ m2 = (a + b + c) / 3 ∨ m2 = (a + b + d) / 3 ∨ m2 = (a + c + d) / 3 ∨ m2 = (b + c + d) / 3
      ∧ m3 = (a + b + c) / 3 ∨ m3 = (a + b + d) / 3 ∨ m3 = (a + c + d) / 3 ∨ m3 = (b + c + d) / 3
      ∧ m4 = (a + b + c) / 3 ∨ m4 = (a + b + d) / 3 ∨ m4 = (a + c + d) / 3 ∨ m4 = (b + c + d) / 3
      ∧ ( ∥(a + b + c + d) / 4 - m1∥ = ∥(a + b + c + d) / 4 - m2∥ 
          ∧ ∥(a + b + c + d) / 4 - m2∥ = ∥(a + b + c + d) / 4 - m3∥ 
          ∧ ∥(a + b + c + d) / 4 - m3∥ = ∥(a + b + c + d) / 4 - m4∥ 
        )
      )) :
  ∀ (x y z w : EuclideanSpace ℝ (Fin 3)), 
  x = a - c ∧ y = b - d ∧ 
  (x ∙ y = 0 ∨ x ∙ z = 0 ∨ x ∙ w = 0 ∨ y ∙ z = 0 ∨ y ∙ w = 0 ∨ z ∙ w = 0)
 :=
sorry

end orthocentric_tetrahedron_l192_192994


namespace polynomial_exists_l192_192155

open Polynomial

noncomputable def exists_polynomial_2013 : Prop :=
  ∃ (f : Polynomial ℤ), (∀ (n : ℕ), n ≤ f.natDegree → (coeff f n = 1 ∨ coeff f n = -1))
                         ∧ ((X - 1) ^ 2013 ∣ f)

theorem polynomial_exists : exists_polynomial_2013 :=
  sorry

end polynomial_exists_l192_192155


namespace positive_divisors_of_8_factorial_l192_192819

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192819


namespace exponent_properties_l192_192973

variables (a : ℝ) (m n : ℕ)
hypothesis (h1 : a^m = 2) (h2 : a^n = 3)

theorem exponent_properties : (a^(m+n) = 6) ∧ (a^(m-2*n) = 2 / 9) :=
by
  sorry

end exponent_properties_l192_192973


namespace count_integer_values_l192_192312

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192312


namespace num_divisors_of_8_factorial_l192_192875

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192875


namespace Whitney_total_cost_l192_192543

-- Conditions
def whaleBooks := 15
def whaleBookCost := 14
def fishBooks := 12
def fishBookCost := 13
def sharkBooks := 5
def sharkBookCost := 10
def magazines := 8
def magazineCost := 3
def fishBookDiscount := 0.10
def salesTax := 0.05

-- Definitions based on conditions
def totalWhaleBookCost := whaleBooks * whaleBookCost
def totalFishBookCost := fishBooks * fishBookCost
def totalSharkBookCost := sharkBooks * sharkBookCost
def totalMagazineCost := magazines * magazineCost

def discountOnFishBooks := totalFishBookCost * fishBookDiscount
def fishBooksAfterDiscount := totalFishBookCost - discountOnFishBooks
def totalCostBeforeTax := totalWhaleBookCost + fishBooksAfterDiscount + totalSharkBookCost + totalMagazineCost
def taxAmount := totalCostBeforeTax * salesTax
def totalFinalCost := totalCostBeforeTax + taxAmount

-- Theorem statement to prove the final cost
theorem Whitney_total_cost : totalFinalCost = 445.62 :=
by sorry

end Whitney_total_cost_l192_192543


namespace max_value_z_minus_one_i_l192_192196

theorem max_value_z_minus_one_i (z : ℂ) (hz : abs z = 1) : abs (z - 1 - complex.I) ≤ sqrt 2 + 1 :=
sorry

end max_value_z_minus_one_i_l192_192196


namespace symmetric_center_and_sum_l192_192719

def f (x : ℝ) : ℝ := x^3 + Real.sin x + 2

theorem symmetric_center_and_sum :
  (∀ x1 x2 : ℝ, x1 + x2 = 0 → f x1 + f x2 = 4) ∧
  (f (-1) + f (-19/20) + f (-18/20) + f (-17/20) + f (-16/20) + f (-15/20) + f (-14/20) + f (-13/20) + f (-12/20) +
     f (-11/20) + f (-10/20) + f (-9/20) + f (-8/20) + f (-7/20) + f (-6/20) + f (-5/20) + f (-4/20) + f (-3/20) + 
     f (-2/20) + f (-1/20) + f 0 + f (1/20) + f (2/20) + f (3/20) + f (4/20) + f (5/20) + f (6/20) + f (7/20) + 
     f (8/20) + f (9/20) + f (10/20) + f (11/20) + f (12/20) + f (13/20) + f (14/20) + f (15/20) + f (16/20) + 
     f (17/20) + f (18/20) + f (19/20) + f 1) = 82 :=
by
  sorry

end symmetric_center_and_sum_l192_192719


namespace participants_in_robbery_l192_192611

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l192_192611


namespace num_divisors_8_factorial_l192_192918

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192918


namespace cheapest_option_is_1_l192_192248

-- Definitions of the costs and amounts
def cost_train_ticket : ℝ := 200
def berries_collected : ℝ := 5
def cost_per_kg_berries_market : ℝ := 150
def cost_per_kg_sugar : ℝ := 54
def jam_production_rate : ℝ := 1.5
def cost_per_kg_jam_market : ℝ := 220

-- Calculations for cost per kg of jam for each option
def cost_per_kg_berries_collect := cost_train_ticket / berries_collected
def cost_per_kg_jam_collect := cost_per_kg_berries_collect + cost_per_kg_sugar
def cost_for_1_5_kg_jam_collect := cost_per_kg_jam_collect
def cost_for_1_5_kg_jam_market := cost_per_kg_berries_market + cost_per_kg_sugar
def cost_for_1_5_kg_jam_ready := cost_per_kg_jam_market * jam_production_rate

-- Proof that Option 1 is the cheapest
theorem cheapest_option_is_1 : (cost_for_1_5_kg_jam_collect < cost_for_1_5_kg_jam_market ∧ cost_for_1_5_kg_jam_collect < cost_for_1_5_kg_jam_ready) :=
by
  sorry

end cheapest_option_is_1_l192_192248


namespace number_of_divisors_of_8_fact_l192_192883

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192883


namespace num_divisors_8_fact_l192_192856

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192856


namespace trig_expr_value_l192_192686

theorem trig_expr_value :
  (sin (330 * real.pi / 180) * tan (-13 * real.pi / 3)) / 
  (cos (-19 * real.pi / 6) * cos (690 * real.pi / 180)) = -2 * real.sqrt 3 / 3 := 
sorry

end trig_expr_value_l192_192686


namespace simple_interest_years_l192_192033

theorem simple_interest_years (P : ℝ) (r : ℝ) (SI : ℝ) (t : ℝ) 
  (hP : P = 1750) (hr : r = 0.08) (hSI : SI = 420) :
  P * r * t = SI → t = 3 :=
by
  intros h
  rw [hP, hr, hSI] at h
  linarith

end simple_interest_years_l192_192033


namespace num_possible_integer_values_l192_192306

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192306


namespace modulus_z_eq_sqrt_five_l192_192190

theorem modulus_z_eq_sqrt_five (i : ℂ) (z : ℂ) (h : (1 + i) * z = 3 - i) : abs(z) = sqrt(5) :=
sorry

end modulus_z_eq_sqrt_five_l192_192190


namespace count_integer_values_l192_192310

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192310


namespace find_a_for_critical_point_inequality_for_a_zero_l192_192731

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x + 1 / x

def is_critical_point (f : ℝ → ℝ) (a x : ℝ) : Prop :=
  (deriv (fun x => f x a) x = 0)

theorem find_a_for_critical_point :
  is_critical_point f a 1 → a = 0 :=
by
  sorry

theorem inequality_for_a_zero (x : ℝ) (h : 0 < x) :
  let a := 0; f x a ≤ x * Real.exp x - x + 1 / x - 1 :=
by
  sorry

end find_a_for_critical_point_inequality_for_a_zero_l192_192731


namespace movies_in_first_box_l192_192655

theorem movies_in_first_box (x : ℕ) 
  (cost_first : ℕ) (cost_second : ℕ) 
  (num_second : ℕ) (avg_price : ℕ)
  (h_cost_first : cost_first = 2)
  (h_cost_second : cost_second = 5)
  (h_num_second : num_second = 5)
  (h_avg_price : avg_price = 3)
  (h_total_eq : cost_first * x + cost_second * num_second = avg_price * (x + num_second)) :
  x = 5 :=
by
  sorry

end movies_in_first_box_l192_192655


namespace num_divisors_of_8_factorial_l192_192881

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192881


namespace solve_system_of_equations_l192_192004

theorem solve_system_of_equations :
  ∃ (x y : ℝ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end solve_system_of_equations_l192_192004


namespace num_pos_divisors_fact8_l192_192779

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192779


namespace smallest_x_value_l192_192178

open Real

theorem smallest_x_value (x : ℝ) 
  (h : x * abs x = 3 * x + 2) : 
  x = -2 ∨ (∀ y, y * abs y = 3 * y + 2 → y ≥ -2) := sorry

end smallest_x_value_l192_192178


namespace train_speed_correct_l192_192117

noncomputable def train_speed_kmph (train_length bridge_length : ℕ) (time_to_cross_seconds : ℝ) (conversion_factor : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_mps := (total_distance : ℝ) / time_to_cross_seconds
  speed_mps * conversion_factor

theorem train_speed_correct :
  train_speed_kmph 100 150 21.42685727998903 3.6 ≈ 41.9904 :=
by
  -- By definition and calculations, the speed of the train is approximately 41.9904 km/h
  sorry

end train_speed_correct_l192_192117


namespace num_divisors_fact8_l192_192953

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192953


namespace probability_of_odd_numbered_ball_first_l192_192092

/-- Problem Statement: -/
theorem probability_of_odd_numbered_ball_first :
  let balls := finset.range 1 101 -- Balls numbered from 1 to 100
  let odd_balls := finset.filter (λ n, n % 2 = 1) balls -- Odd-numbered balls
  let even_balls := finset.filter (λ n, n % 2 = 0) balls -- Even-numbered balls
  let total_balls := 100 -- Total number of balls
  let odd_count := finset.card odd_balls -- Count of odd-numbered balls
  let even_count := finset.card even_balls -- Count of even-numbered balls
  let total_count := odd_count + even_count -- Total count of odd and even balls
  let probabilty_of_odd := (odd_count / total_count : ℚ) -- Desired probability of selecting an odd-numbered ball first
  (2 / 3 : ℚ) = probabilty_of_odd :=
by
  sorry

end probability_of_odd_numbered_ball_first_l192_192092


namespace percentage_decrease_second_year_l192_192093

variable (I I1 T : ℝ)

theorem percentage_decrease_second_year : 
  ∀ (I I1 T : ℝ),
  I = 100 → 
  I1 = I + 0.80 * I → 
  T = I + 0.26 * I → 
  (I1 - (I1 * (30 / 100)) = T) :=
by
  intros I I1 T hI hI1 hT
  rw [hI, hI1, hT]
  sorry

end percentage_decrease_second_year_l192_192093


namespace system_of_equations_soln_l192_192678

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l192_192678


namespace solution_l192_192721

noncomputable def f : ℝ → ℝ := sorry

axiom new_condition1 : ∀ x : ℝ, f(x + π) = f(x) + sin x
axiom new_condition2 : ∀ x : ℝ, 0 ≤ x ∧ x < π → f(x) = 0

theorem solution : f (23 * π / 6) = 1 / 2 :=
by
  -- Prove using the given axioms and conditions
  sorry

end solution_l192_192721


namespace max_board_size_l192_192400

theorem max_board_size : ∀ (n : ℕ), 
  (∃ (board : Fin n → Fin n → Prop),
    ∀ i j k l : Fin n,
      (i ≠ k ∧ j ≠ l) → board i j ≠ board k l) ↔ n ≤ 4 :=
by sorry

end max_board_size_l192_192400


namespace num_divisors_fact8_l192_192948

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192948


namespace count_whole_numbers_l192_192258

theorem count_whole_numbers (a b : ℝ) (ha : a = real.cbrt 20) (hb : b = real.cbrt 300) :
  ∃ n : ℕ, n = 4 ∧ 3 ≤ (⌈ a ⌉ : ℤ) ∧ (⌊ b ⌋ : ℤ) = 6 :=
by
  sorry

end count_whole_numbers_l192_192258


namespace math_problem_l192_192260

variable (x Q : ℝ)

theorem math_problem (h : 5 * (6 * x - 3 * Real.pi) = Q) :
  15 * (18 * x - 9 * Real.pi) = 9 * Q := 
by
  sorry

end math_problem_l192_192260


namespace num_divisors_of_8_factorial_l192_192872

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192872


namespace sum_of_x_values_l192_192664

-- Define the equation as a condition
def equation (x y : ℤ) : Prop :=
  7 * x * y - 13 * x + 15 * y - 37 = 0

-- Define the main theorem to prove
theorem sum_of_x_values : (∑ (x, y) in {(-2, 11), (-1, 3), (7, 2)}, x) = 4 :=
by
  sorry

end sum_of_x_values_l192_192664


namespace prob_divisors_8_fact_l192_192812

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192812


namespace num_divisors_of_8_factorial_l192_192873

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192873


namespace number_of_integer_values_l192_192277

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192277


namespace find_point_N_with_equal_angles_l192_192235

-- Define point struct for better readability in geometric context
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the right triangle ABC and midpoint M of hypotenuse AB
variables {A B C N : Point}

-- Define the angles function
def angle (P Q R : Point) : ℝ := sorry  -- Angle calculation placeholder

-- Given: triangle ABC is a right triangle with a right angle at B
axiom right_triangle (A B C : Point) : angle A B C = 90

-- Given: N is a point such that ∠NAB = ∠NBC = ∠NCA
axiom equal_angles (N A B C : Point) : angle N A B = angle N B C ∧ angle N B C = angle N C A

-- We'll prove N exists and ∠NAB = ∠NBC = 45°
theorem find_point_N_with_equal_angles (A B C N : Point)
  (h1 : right_triangle A B C)
  (h2 : equal_angles N A B C) : 
  angle N A B = 45 ∧ angle N B C = 45 ∧ angle N C A = 45 := 
sorry

end find_point_N_with_equal_angles_l192_192235


namespace second_recipe_cup_count_l192_192005

theorem second_recipe_cup_count (bottle_ounces : ℕ) (ounces_per_cup : ℕ)
  (first_recipe_cups : ℕ) (third_recipe_cups : ℕ) (bottles_needed : ℕ)
  (total_ounces : bottle_ounces = 16)
  (ounce_to_cup : ounces_per_cup = 8)
  (first_recipe : first_recipe_cups = 2)
  (third_recipe : third_recipe_cups = 3)
  (bottles : bottles_needed = 3) :
  (bottles_needed * bottle_ounces) / ounces_per_cup - first_recipe_cups - third_recipe_cups = 1 :=
by
  sorry

end second_recipe_cup_count_l192_192005


namespace number_of_12_digit_integers_l192_192740

open Finset

-- Definition of the set U (all 12-digit numbers with digits either 1 or 2)
def U := { n : ℕ | ∀ i < 12, (Bit0 1 ≤ n.digits 2 ![i] ∧ n.digits 2 ![i] ≤ Bit1 0) }

-- Definition of the subset A (12-digit numbers with no two consecutive digits alike)
def A := { n ∈ U | ∀ i < 11, (n.digits 2 ![i] ≠ n.digits 2 ![i + 1]) }

-- Prove the cardinality of U \ A is 4094
theorem number_of_12_digit_integers : (U.card - A.card) = 4094 :=
by
  sorry

end number_of_12_digit_integers_l192_192740


namespace dihedral_angle_cosine_l192_192525

theorem dihedral_angle_cosine (r₁ r₂ : ℝ) (h₁ : r₂ = 1.5 * r₁)
    (d : ℝ) (h₂ : d = r₁ + r₂) : 
    let θ : ℝ := 30 * Real.pi / 180 in
    cos θ = 0.68 :=
by
  sorry

end dihedral_angle_cosine_l192_192525


namespace integer_count_in_range_l192_192752

theorem integer_count_in_range :
  {x : ℤ | -10 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 9}.card = 7 :=
sorry

end integer_count_in_range_l192_192752


namespace correct_relationship_l192_192211

theorem correct_relationship (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) : (1 / log a) > (1 / log b) :=
sorry

end correct_relationship_l192_192211


namespace park_area_l192_192490

theorem park_area (P : ℝ) (w l : ℝ) (hP : P = 120) (hL : l = 3 * w) (hPerimeter : 2 * l + 2 * w = P) : l * w = 675 :=
by
  sorry

end park_area_l192_192490


namespace circle_equation_through_origin_l192_192700

theorem circle_equation_through_origin (focus : ℝ × ℝ) (radius : ℝ) (x y : ℝ) 
  (h1 : focus = (1, 0)) 
  (h2 : (x - 1)^2 + y^2 = radius^2) : 
  x^2 + y^2 - 2*x = 0 :=
by
  sorry

end circle_equation_through_origin_l192_192700


namespace Luke_money_at_end_of_June_l192_192993

def initial_amount_luke : ℝ := 48
def february_expenditure (initial_amount : ℝ) : ℝ := 0.3 * initial_amount
def money_after_february (initial_amount : ℝ) : ℝ := initial_amount - february_expenditure(initial_amount)
def march_expenditure : ℝ := 11
def money_received_march : ℝ := 21
def money_after_march (money_february : ℝ) : ℝ := money_february - march_expenditure + money_received_march
def monthly_savings (money: ℝ) : ℝ := 0.1 * money
def money_after_saving (money: ℝ) : ℝ := money - monthly_savings(money)

theorem Luke_money_at_end_of_June : 
  money_after_saving (money_after_saving (money_after_saving (money_after_march (money_after_february initial_amount_luke)))) = 31.79 :=
by
  sorry

end Luke_money_at_end_of_June_l192_192993


namespace sequence_general_formula_l192_192205

theorem sequence_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+2) = 2 * a (n+1) / (2 + a (n+1))) :
  (a 1 = 1) → ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end sequence_general_formula_l192_192205


namespace quadratic_monotonic_range_l192_192225

theorem quadratic_monotonic_range {a : ℝ} :
  (∀ x1 x2 : ℝ, (2 < x1 ∧ x1 < x2 ∧ x2 < 3) → (x1^2 - 2*a*x1 + 1) ≤ (x2^2 - 2*a*x2 + 1) ∨ (x1^2 - 2*a*x1 + 1) ≥ (x2^2 - 2*a*x2 + 1)) → (a ≤ 2 ∨ a ≥ 3) := 
sorry

end quadratic_monotonic_range_l192_192225


namespace robbery_participants_l192_192607

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l192_192607


namespace no_real_solution_for_k_l192_192167

theorem no_real_solution_for_k : 
  ∀ k : ℝ, ∥k • ⟨3, -4⟩ - ⟨5, 8⟩∥ ≠ 3 * real.sqrt 13 :=
by
  intro k
  simp only [sub_eq_add_neg, smul_neg, norm_eq_sqrt_inner, real_inner, sq, norm]
  sorry

end no_real_solution_for_k_l192_192167


namespace KL_parallel_AD_l192_192457

-- Define the conditions of the problem
variables {A B C D K L : Type}
variable [Geometry A]
variable [InCircle B C D : Circle]
variable [InscribedQuadrilateral A B C D]

-- Points K and L on diagonals AC and BD respectively
variable (K_on_AC : On K A C)
variable (L_on_BD : On L B D)

-- Here AK = AB and DL = DC
variable (AK_eq_AB : Equal AK AB)
variable (DL_eq_DC : Equal DL DC)

-- Prove KL is parallel to AD
theorem KL_parallel_AD : Parallel KL AD :=
by
  sorry -- Proof is omitted

end KL_parallel_AD_l192_192457


namespace radius_of_circle_C1_l192_192636

theorem radius_of_circle_C1 (C1 C2 : Circle) (O X Z Y W : Point) (r R : ℝ)
  (h1 : C1.center = O)
  (h2 : O ∈ line O X)
  (h3 : C1.radius = r)
  (h4 : C2.radius = R)
  (h5 : X ∈ C1)
  (h6 : X ∈ C2)
  (h7 : Z ∈ C2)
  (h8 : ¬ (Z ∈ C1))
  (h9 : XZ = 15)
  (h10 : OZ = 17)
  (h11 : YZ = 8)
  (h12 : ¬ (W ∈ C1))
  (h13 : ¬ (W ∈ C2))
  (h14 : XW = 20)
  (h15 : OW = 25) :
  r = 5 := by
  sorry

end radius_of_circle_C1_l192_192636


namespace positive_divisors_of_8_factorial_l192_192826

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192826


namespace tan_theta_equation_l192_192648

theorem tan_theta_equation (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 6) :
  Real.tan θ + Real.tan (4 * θ) + Real.tan (6 * θ) = 0 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  sorry

end tan_theta_equation_l192_192648


namespace range_of_x_in_function_sqrt_x_minus_2_l192_192495

theorem range_of_x_in_function_sqrt_x_minus_2 (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 2)) → x ≥ 2 :=
by
  sorry

end range_of_x_in_function_sqrt_x_minus_2_l192_192495


namespace num_divisors_8_fact_l192_192858

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192858


namespace prob_divisors_8_fact_l192_192806

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192806


namespace num_divisors_8_fact_l192_192859

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192859


namespace abs_x_minus_2_eq_5_sum_of_distances_eq_3_abs_x_minus_2_plus_abs_x_plus_3_min_val_l192_192561

-- Statement for problem 1
theorem abs_x_minus_2_eq_5 (x : ℤ) : (|x - 2| = 5) → (x = 7 ∨ x = -3) :=
sorry

-- Statement for problem 2
theorem sum_of_distances_eq_3 (x : ℤ) : (|x - 2| + |x + 1| = 3) → (-1 ≤ x ∧ x ≤ 2) :=
sorry

-- Statement for problem 3
theorem abs_x_minus_2_plus_abs_x_plus_3_min_val (x : ℚ) : ∃ m, m = 5 ∧ ∀ y, (|y - 2| + |y + 3|) ≥ m :=
sorry

end abs_x_minus_2_eq_5_sum_of_distances_eq_3_abs_x_minus_2_plus_abs_x_plus_3_min_val_l192_192561


namespace number_of_divisors_of_8_fact_l192_192886

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192886


namespace current_population_is_513_l192_192071

-- Defining the initial population
def initial_population : ℕ := 684

-- Defining the percentage increase
def growth_rate : ℝ := 0.25

-- Defining the percentage decrease
def move_away_rate : ℝ := 0.40

-- Calculate the new population after the increase
def new_population : ℕ := initial_population + (initial_population * growth_rate).to_nat

-- Calculate the decrease in population
def decrease : ℕ := (new_population * move_away_rate).to_nat

-- Calculate the current population
def current_population : ℕ := new_population - decrease

-- The theorem to prove that the current population is 513
theorem current_population_is_513 : current_population = 513 := by
  sorry

end current_population_is_513_l192_192071


namespace relationship_between_abc_l192_192192

theorem relationship_between_abc 
  (a b c : ℝ) 
  (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c) 
  (ha : Real.exp a = 9 * a * Real.log 11)
  (hb : Real.exp b = 10 * b * Real.log 10)
  (hc : Real.exp c = 11 * c * Real.log 9) : 
  a < b ∧ b < c :=
sorry

end relationship_between_abc_l192_192192


namespace cheapest_option_is_1_l192_192247

-- Definitions of the costs and amounts
def cost_train_ticket : ℝ := 200
def berries_collected : ℝ := 5
def cost_per_kg_berries_market : ℝ := 150
def cost_per_kg_sugar : ℝ := 54
def jam_production_rate : ℝ := 1.5
def cost_per_kg_jam_market : ℝ := 220

-- Calculations for cost per kg of jam for each option
def cost_per_kg_berries_collect := cost_train_ticket / berries_collected
def cost_per_kg_jam_collect := cost_per_kg_berries_collect + cost_per_kg_sugar
def cost_for_1_5_kg_jam_collect := cost_per_kg_jam_collect
def cost_for_1_5_kg_jam_market := cost_per_kg_berries_market + cost_per_kg_sugar
def cost_for_1_5_kg_jam_ready := cost_per_kg_jam_market * jam_production_rate

-- Proof that Option 1 is the cheapest
theorem cheapest_option_is_1 : (cost_for_1_5_kg_jam_collect < cost_for_1_5_kg_jam_market ∧ cost_for_1_5_kg_jam_collect < cost_for_1_5_kg_jam_ready) :=
by
  sorry

end cheapest_option_is_1_l192_192247


namespace num_divisors_of_8_factorial_l192_192876

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192876


namespace martin_marks_l192_192434

theorem martin_marks 
  (M : ℕ) -- Maximum marks
  (P : ℕ) -- Pass percentage
  (S : ℕ) -- Marks shortfall
  (hM : M = 500)
  (hP : P = 80)
  (hS : S = 200)
  :
  let required_to_pass := (P * M) / 100 in
  let m := required_to_pass - S in
  m = 200 :=
by {
  have h1 : required_to_pass = 400, from sorry,
  have h2 : m = 200, from sorry,
  exact h2
}

end martin_marks_l192_192434


namespace number_of_x_values_l192_192270

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192270


namespace distinct_numbers_board_l192_192412

theorem distinct_numbers_board (m n : ℕ) (h_m : 4 ≤ m) (h_n : 4 ≤ n)
  (board : Fin m → Fin n → ℤ)
  (mean_condition : ∀ i j : Fin m, ∀ neighbors : list (Fin m × Fin n),
    (∀ (i' j') : Fin m × Fin n, (i', j') ∈ neighbors → (|i'.1 - i.1| + |i'.2 - j.2| = 1)) →
    ∃ a b, (a, b) ∈ neighbors ∧ board i j = (board a.1 a.2 + board b.1 b.2) / 2) :
  ∃ x : ℤ, ∀ i j, board i j = x := by
  sorry

end distinct_numbers_board_l192_192412


namespace sum_of_ages_is_42_l192_192502

-- Define the variables for present ages of the son (S) and the father (F)
variables (S F : ℕ)

-- Define the conditions:
-- 1. 6 years ago, the father's age was 4 times the son's age.
-- 2. After 6 years, the son's age will be 18 years.

def son_age_condition := S + 6 = 18
def father_age_6_years_ago_condition := F - 6 = 4 * (S - 6)

-- Theorem statement to prove:
theorem sum_of_ages_is_42 (S F : ℕ)
  (h1 : son_age_condition S)
  (h2 : father_age_6_years_ago_condition F S) :
  S + F = 42 :=
sorry

end sum_of_ages_is_42_l192_192502


namespace amoeba_count_after_two_weeks_l192_192122

theorem amoeba_count_after_two_weeks :
  let initial_day_count := 1
  let days_double_split := 7
  let days_triple_split := 7
  let end_of_first_phase := initial_day_count * 2 ^ days_double_split
  let final_amoeba_count := end_of_first_phase * 3 ^ days_triple_split
  final_amoeba_count = 279936 :=
by
  sorry

end amoeba_count_after_two_weeks_l192_192122


namespace right_triangle_iff_l192_192121

variable {A B C : ℝ} -- Angles of the triangle

-- Condition ①: ∠A + ∠B = ∠C
def Condition1 (A B C : ℝ) : Prop :=
  A + B = C

-- Condition ②: ∠A : ∠B : ∠C = 1 : 2 : 3
def Condition2 (A B C : ℝ) : Prop :=
  (A / B) = (1 / 2) ∧ (B / C) = (2 / 3)

-- Condition ③: ∠A = ∠B = ∠C
def Condition3 (A B C : ℝ) : Prop :=
  A = B ∧ B = C

-- Condition ④: ∠A = 90° - ∠B
def Condition4 (A B C : ℝ) : Prop :=
  A = 90 - B

-- Prove that ∠C = 90° if and only if Conditions ①, ②, and ④ hold
theorem right_triangle_iff (A B C : ℝ) :
  (Condition1 A B C ∨ Condition2 A B C ∨ Condition4 A B C) ↔ C = 90 :=
begin
  sorry -- Proof goes here
end

end right_triangle_iff_l192_192121


namespace area_fourth_triangle_sequence_30_60_90_l192_192115

open Real

theorem area_fourth_triangle_sequence_30_60_90 
  (hypotenuse_AB : ℝ)
  (h1 : hypotenuse_AB = 12)
  (triangle_1 : 30_60_90_triangle)
  (h2 : triangle_1.hypotenuse = hypotenuse_AB)
  (triangle_2 : 30_60_90_triangle)
  (h3 : triangle_2.hypotenuse = triangle_1.longest_leg)
  (triangle_3 : 30_60_90_triangle)
  (h4 : triangle_3.hypotenuse = triangle_2.longest_leg)
  (triangle_4 : 30_60_90_triangle)
  (h5 : triangle_4.hypotenuse = triangle_3.longest_leg) :
  triangle_4.area = 7.59 := sorry

end area_fourth_triangle_sequence_30_60_90_l192_192115


namespace find_x_plus_y_l192_192218

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3000)
  (h2 : x + 3000 * Real.sin y = 2999) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2999 := by
  sorry

end find_x_plus_y_l192_192218


namespace experimental_fertilizer_height_is_correct_l192_192392

/-- Define the static heights and percentages for each plant's growth conditions. -/
def control_plant_height : ℝ := 36
def bone_meal_multiplier : ℝ := 1.25
def cow_manure_multiplier : ℝ := 2
def experimental_fertilizer_multiplier : ℝ := 1.5

/-- Define each plant's height based on the given multipliers and conditions. -/
def bone_meal_plant_height : ℝ := bone_meal_multiplier * control_plant_height
def cow_manure_plant_height : ℝ := cow_manure_multiplier * bone_meal_plant_height
def experimental_fertilizer_plant_height : ℝ := experimental_fertilizer_multiplier * cow_manure_plant_height

/-- Proof that the height of the experimental fertilizer plant is 135 inches. -/
theorem experimental_fertilizer_height_is_correct :
  experimental_fertilizer_plant_height = 135 := by
    sorry

end experimental_fertilizer_height_is_correct_l192_192392


namespace Kates_hair_length_l192_192403

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end Kates_hair_length_l192_192403


namespace log_a_lt_a_sq_lt_2_a_l192_192187

theorem log_a_lt_a_sq_lt_2_a (a : ℝ) (h : 0 < a ∧ a < 1) : 
  log 2 a < a^2 ∧ a^2 < 2^a := 
sorry

end log_a_lt_a_sq_lt_2_a_l192_192187


namespace alex_shirts_count_l192_192600

theorem alex_shirts_count (j a b : ℕ) (h1 : j = a + 3) (h2 : b = j + 8) (h3 : b = 15) : a = 4 :=
by
  sorry

end alex_shirts_count_l192_192600


namespace arrangement_count_correct_l192_192154

def people : Type := {A, B, C, D, E} -- Define the five people

def valid_arrangement (arrangement : list people) : Prop :=
  -- Check if A and B are not next to C
  ∀ i, 
    (i > 0 ∧ i < arrangement.length - 1) → 
    (arrangement.get_opt i = some C →
     (arrangement.get_opt (i - 1) ≠ some A ∧ arrangement.get_opt (i - 1) ≠ some B) ∧
     (arrangement.get_opt (i + 1) ≠ some A ∧ arrangement.get_opt (i + 1) ≠ some B))

def num_valid_arrangements : ℕ :=
  -- Calculate the number of valid arrangements (this is the problem statement)
  48

theorem arrangement_count_correct :
  ∃ arrangements : list (list people),
    valid_arrangement ∧ list.length arrangements = num_valid_arrangements :=
sorry

end arrangement_count_correct_l192_192154


namespace right_triangle_area_l192_192381

variable (AB AC : ℝ) (angle_A : ℝ)

def is_right_triangle (AB AC : ℝ) (angle_A : ℝ) : Prop :=
  angle_A = 90

def area_of_triangle (AB AC : ℝ) : ℝ :=
  0.5 * AB * AC

theorem right_triangle_area :
  is_right_triangle AB AC angle_A →
  AB = 35 →
  AC = 15 →
  area_of_triangle AB AC = 262.5 :=
by
  intros
  simp [is_right_triangle, area_of_triangle]
  sorry

end right_triangle_area_l192_192381


namespace log_fraction_squared_l192_192451

noncomputable def Log : ℝ → ℝ → ℝ := real.logb

theorem log_fraction_squared
  (x y : ℝ)
  (h1 : 0 < x ∧ x ≠ 1)
  (h2 : 0 < y ∧ y ≠ 1)
  (h3 : Log 3 x = Log y 81)
  (h4 : x * y = 243) :
  (Log 3 (x / y))^2 = 9 :=
sorry

end log_fraction_squared_l192_192451


namespace number_of_integer_values_l192_192273

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192273


namespace num_divisors_8_factorial_l192_192754

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192754


namespace boat_travel_distance_downstream_l192_192564

-- Define the basic parameters
def boat_speed_in_still_water : ℝ := 5 -- km/hr
def stream_speed : ℝ := 5 -- km/hr
def travel_time_downstream : ℝ := 10 -- hours

-- Calculate the downstream speed and distance
def effective_speed_downstream : ℝ := boat_speed_in_still_water + stream_speed

def distance_downstream : ℝ := effective_speed_downstream * travel_time_downstream

-- Prove the correct answer
theorem boat_travel_distance_downstream :
  distance_downstream = 100 := by
  -- Definitions lead to
  -- effective_speed_downstream = 10 km/hr
  -- distance_downstream = 100 km
  -- Therefore the theorem proves the distance equals 100 km
  rfl

end boat_travel_distance_downstream_l192_192564


namespace robbery_proof_l192_192614

variables (A B V G : Prop)

-- Define the conditions as Lean propositions
def condition1 : Prop := ¬G → (B ∧ ¬A)
def condition2 : Prop := V → (¬A ∧ ¬B)
def condition3 : Prop := G → B
def condition4 : Prop := B → (A ∨ V)

-- The statement we want to prove based on conditions
theorem robbery_proof (h1 : condition1 A B G) 
                      (h2 : condition2 A B V) 
                      (h3 : condition3 B G) 
                      (h4 : condition4 A B V) : 
                      A ∧ B ∧ G :=
begin
  sorry
end

end robbery_proof_l192_192614


namespace color_subsets_l192_192423

noncomputable def S : Set ℕ := {i | i < 2002}

def P : Set (Set ℕ) := set.powerset S

theorem color_subsets (n : ℕ) (hn : n ≤ 2^2002) :
  ∃ (white black : Set (Set ℕ)),
  white ∪ black = P ∧
  white ∩ black = ∅ ∧
  white.card = n ∧
  ∀ A B ∈ white, (A ∪ B) ∈ white ∧
  ∀ A B ∈ black, (A ∪ B) ∈ black :=
sorry

end color_subsets_l192_192423


namespace DEKN_is_cyclic_l192_192556

open EuclideanGeometry

variable {A B C D E F T M N K : Point}

def cyclic_quadrilateral (A B C D : Point) : Prop := ∃ O : Circle, A ∈ O ∧ B ∈ O ∧ C ∈ O ∧ D ∈ O

theorem DEKN_is_cyclic (hA_outside : ¬(A ∈ ω))
  (hBC_intersect : ∀ (hBC : Line), ∃ B C, hBC_through_A : A ∈ hBC ∧ intersect_circle_line B C ω hBC ∧ B ≠ C)
  (hDE_intersect : ∀ (hDE : Line), ∃ D E, hDE_through_A : A ∈ hDE ∧ intersect_circle_line D E ω hDE ∧ D ≠ E ∧ D ∈ segment A E)
  (hDF_parallel_BC : ∀ (hDF: Line), D ∈ hDF ∧ parallel hDF (line_through B C) ∧ intersect_circle_line D F ω hDF ∧ F ≠ D)
  (hAF_intersect : ∃ T, T ≠ F ∧ intersect_circle_line A T ω (line_through A F))
  (hM_defined : ∃ M, ∃ hBC (hET : Line), M = intersect_lines hBC hET ∧ intersect_circle_line B C ω hBC ∧ intersect_circle_line E T ω hET)
  (hN_symmetric : ∃ M, N = symmetric_point A M)
  (hK_midpoint : K = midpoint B C) :
  cyclic_quadrilateral D E K N := by
  sorry

end DEKN_is_cyclic_l192_192556


namespace average_of_remaining_two_numbers_l192_192078

theorem average_of_remaining_two_numbers (S a₁ a₂ a₃ a₄ : ℝ)
    (h₁ : S / 6 = 3.95)
    (h₂ : (a₁ + a₂) / 2 = 3.8)
    (h₃ : (a₃ + a₄) / 2 = 3.85) :
    (S - (a₁ + a₂ + a₃ + a₄)) / 2 = 4.2 := 
sorry

end average_of_remaining_two_numbers_l192_192078


namespace positive_divisors_8_factorial_l192_192838

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192838


namespace typing_service_cost_is_five_l192_192102

def typing_cost (x : ℝ) : ℝ :=
  100 * x + 30 * 4 + 20 * 4 * 2

theorem typing_service_cost_is_five :
  ∃ x : ℝ, typing_cost x = 780 ∧ x = 5 :=
by
  use 5
  simp [typing_cost]
  norm_num
  sorry

end typing_service_cost_is_five_l192_192102


namespace smallest_n_rotation_identity_l192_192650

noncomputable def rotationMatrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ], ![Real.sin θ, Real.cos θ]]

theorem smallest_n_rotation_identity :
  ∃ n : ℕ, 0 < n ∧ (rotationMatrix (160 * Real.pi / 180)) ^ n = Matrix.identity (Fin 2) ∧ n = 9 :=
by
  sorry

end smallest_n_rotation_identity_l192_192650


namespace pos_divisors_8_factorial_l192_192899

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192899


namespace count_integer_values_l192_192313

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192313


namespace positive_divisors_of_8_factorial_l192_192825

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192825


namespace integral_problem_solution_l192_192658

open Real

noncomputable def integral_problem : ℝ :=
  ∫ x in -2..2, sqrt (4 - x ^ 2) - x ^ 2017

theorem integral_problem_solution :
  integral_problem = 2 * π :=
by
  sorry

end integral_problem_solution_l192_192658


namespace resort_employees_l192_192133

theorem resort_employees (B D S BD BS DS BDS : ℕ) 
  (hB : B = 15) 
  (hD : D = 18) 
  (hS : S = 12) 
  (hBD : BD + BS + DS = 4) 
  (hBDS : BDS = 1) : 
  B + D + S - BD - BS - DS - 2 * BDS = 39 := 
by
  rw [hB, hD, hS, hBD, hBDS]
  sorry

end resort_employees_l192_192133


namespace num_divisors_8_factorial_l192_192758

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192758


namespace fixed_point_of_mn_l192_192733

theorem fixed_point_of_mn (F : ℝ × ℝ) (M N : ℝ × ℝ) 
  (parabola : ℝ → ℝ → Prop) (chord_AB : ℝ × ℝ → ℝ × ℝ → Prop)
  (chord_CD : ℝ × ℝ → ℝ × ℝ → Prop)
  (midpoint : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ))
  (MN : (ℝ × ℝ) → (ℝ × ℝ) → ℝ)
  (focus : F = (1, 0))
  (on_parabola : ∀ P, parabola P.1 P.2 ↔ P.2^2 = 4 * P.1)
  (through_focus : ∀ A B, chord_AB A B → chord_AB A B → A.1 = focus.1 ∨ B.1 = focus.1)
  (perpendicular : ∀ A B C D, chord_AB A B → chord_CD C D → (A.1 - B.1) * (C.1 - D.1) + (A.2 - B.2) * (C.2 - D.2) = 0)
  (mid_M : ∀ A B, chord_AB A B → midpoint A B = M)
  (mid_N : ∀ C D, chord_CD C D → midpoint C D = N) :
  (MN M N) = (3, 0) := by
sory

end fixed_point_of_mn_l192_192733


namespace value_of_frac_l192_192351

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end value_of_frac_l192_192351


namespace isosceles_triangle_count_l192_192750

theorem isosceles_triangle_count : 
  ∃ (count : ℕ), count = 6 ∧ 
  ∀ (a b c : ℕ), a + b + c = 25 → 
  (a = b ∨ a = c ∨ b = c) → 
  a ≠ b ∨ c ≠ b ∨ a ≠ c → 
  ∃ (x y z : ℕ), x = a ∧ y = b ∧ z = c := 
sorry

end isosceles_triangle_count_l192_192750


namespace painting_cost_l192_192596

-- Define the arithmetic sequences and how to get the nth term
def arith_term (a d n : ℕ) : ℕ := a + (n - 1) * d

-- Define the cost function per digit of a number
def digit_cost (n : ℕ) : ℕ := String.length (n.repr)

-- Prove the total cost for painting house numbers
theorem painting_cost (n : ℕ := 30) :
  let east_side : List ℕ := List.map (arith_term 5 7) (List.range n).map (· + 1),
      west_side : List ℕ := List.map (arith_term 7 8) (List.range n).map (· + 1),
      total_houses : List ℕ := east_side ++ west_side,
      total_cost : ℕ := total_houses.map digit_cost |> List.sum
  in total_cost = 149 :=
by
  sorry

end painting_cost_l192_192596


namespace factorization_correct_l192_192540

theorem factorization_correct : 
  ¬(∃ x : ℝ, -x^2 + 4 * x = -x * (x + 4)) ∧
  ¬(∃ x y: ℝ, x^2 + x * y + x = x * (x + y)) ∧
  (∀ x y: ℝ, x * (x - y) + y * (y - x) = (x - y)^2) ∧
  ¬(∃ x : ℝ, x^2 - 4 * x + 4 = (x + 2) * (x - 2)) :=
by
  sorry

end factorization_correct_l192_192540


namespace segments_sum_l192_192472

-- Definitions and conditions
variables {A B C I D M P N : Type}
variables (I : Incenter ABC)
variables (D : Point_circumcircle_bisector_angleA ABC)
variables (M : Midpoint BC)
variables (P : ReflectionI_over_M I M)
variables (N : Intersect_DP_circumcircle D P)

-- Proof problem statement
theorem segments_sum (AN BN CN : ℝ) :
  (N ∈ Circumcircle ABC) ∧ (D ∈ Circumcircle ABC) ∧ (M = Midpoint BC) ∧ (P = Reflection I M) → 
  AN = BN + CN ∨ BN = AN + CN ∨ CN = AN + BN :=
sorry

end segments_sum_l192_192472


namespace is_isosceles_triangle_l192_192992

theorem is_isosceles_triangle (A B C : Type) [fintype A] [fintype B] [fintype C]
  {AB AC : ℝ} (h : AB = AC) : 
  isosceles A B C :=
sorry

end is_isosceles_triangle_l192_192992


namespace num_possible_integer_values_l192_192298

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192298


namespace pattern_of_balls_l192_192090

-- Define the set of balls and properties
def balls : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

-- Define the properties of odd and even balls
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the selection of 3 balls with replacement
def select_balls : (balls × balls × balls) → Prop :=
  λ (x : balls × balls × balls), true  -- This defines a generic selection

-- Define the probability of selecting an odd ball first
def probability_first_odd : ℝ := 2 / 3

-- Theorem statement
theorem pattern_of_balls (b1 b2 b3 : ℕ) (h1 : b1 ∈ balls) (h2 : b2 ∈ balls) (h3 : b3 ∈ balls) :
  (select_balls (b1, b2, b3) →
  (prob_first_odd := probability_first_odd → 
  (is_odd b1 ∧ is_odd b2 ∧ is_even b3) ∨
  (is_odd b1 ∧ is_even b2 ∧ is_odd b3) ∨
  (is_even b1 ∧ is_odd b2 ∧ is_odd b3))) := sorry

end pattern_of_balls_l192_192090


namespace lost_marble_count_l192_192399

def initial_marble_count : ℕ := 16
def remaining_marble_count : ℕ := 9

theorem lost_marble_count : initial_marble_count - remaining_marble_count = 7 := by
  -- Proof goes here
  sorry

end lost_marble_count_l192_192399


namespace count_integer_values_l192_192314

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192314


namespace tunnel_length_eq_distance_travelled_l192_192503

-- We will define the necessary conditions and then state our result

/-- Definition of the train length in miles -/
def train_length : ℝ := 2

/-- Definition of the time in minutes after which the tail exits the tunnel -/
def time_tail_exits : ℝ := 4

/-- Definition of the speed of the train in miles per hour -/
def train_speed_mph : ℝ := 30

/-- Conversion from hours to minutes -/
def time_conversion_factor : ℝ := 60

/-- Compute train_speed in miles per minute -/
def train_speed_mpm : ℝ := train_speed_mph / time_conversion_factor

/-- The length of the tunnel equals the distance traveled by the train front within the given time -/
theorem tunnel_length_eq_distance_travelled : 
  (train_speed_mpm * time_tail_exits) = train_length := 
by sorry

end tunnel_length_eq_distance_travelled_l192_192503


namespace ab_plus_c_condition_l192_192195

noncomputable def lowest_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x y, y = f x → y ≥ p.2

theorem ab_plus_c_condition (a b c : ℝ) :
  lowest_point (λ x, a * Real.sin x + b * Real.cos x + c) (11 * Real.pi / 6, 1) →
  (∀ x, (λ x, a * Real.sin x + b * Real.cos x + c) (x * 3 / Real.pi + 1) = (λ x, (c - 1) * (Real.sin x) + c) x) →
  ∀ x₁ x₂, (f : ℝ → ℝ) := (λ x, (c - 1) * Real.sin (Real.pi / 3 * x) + c) →
  (f x₁ = 3 ∧ f x₂ = 3) → |x₂ - x₁| = 3 →
  a + b + c = 4 - Real.sqrt 3 := 
sorry

end ab_plus_c_condition_l192_192195


namespace monotonic_decreasing_intervals_l192_192487

noncomputable def f (x : ℝ) : ℝ := (x + 1) / x

theorem monotonic_decreasing_intervals :
  {x : ℝ | f' x < 0} = {x : ℝ | x ∈ Iio 0 ∪ Ioi 0} :=
sorry

end monotonic_decreasing_intervals_l192_192487


namespace Kates_hair_length_l192_192404

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end Kates_hair_length_l192_192404


namespace domino_finite_intersections_l192_192446

theorem domino_finite_intersections :
  (∃ arrangement : (ℤ × ℤ) → bool, 
  (∀ x : ℤ, ∃ finite_dominos_x : set (ℤ × ℤ), finite finite_dominos_x ∧ 
    ∀ k, arrangement (x, k) = (k ∈ finite_dominos_x).to_bool) ∧ 
  (∀ y : ℤ, ∃ finite_dominos_y : set (ℤ × ℤ), finite finite_dominos_y ∧ 
    ∀ k, arrangement (k, y) = (k ∈ finite_dominos_y).to_bool)) :=
sorry

end domino_finite_intersections_l192_192446


namespace num_divisors_8_factorial_l192_192769

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192769


namespace product_of_all_terms_l192_192125

noncomputable def geometric_sequence_product (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ i, a (i + 1) = a 1 * (a (n - 1)) ^ (i / (n - 1))) ∧
  a 1 * a 2 * a 3 * a 4 = 1 / 128 ∧
  a n * a (n - 1) * a (n - 2) * a (n - 3) = 512 ∧
  n = 20

theorem product_of_all_terms (a : ℕ → ℝ) :
  geometric_sequence_product a 20 → ∏ i in finset.range 20, a (i + 1) = 32 :=
by
  sorry

end product_of_all_terms_l192_192125


namespace john_change_received_is_7_l192_192396

def cost_per_orange : ℝ := 0.75
def num_oranges : ℝ := 4
def amount_paid : ℝ := 10.0
def total_cost : ℝ := num_oranges * cost_per_orange
def change_received : ℝ := amount_paid - total_cost

theorem john_change_received_is_7 : change_received = 7 :=
by
  sorry

end john_change_received_is_7_l192_192396


namespace length_GH_of_tetrahedron_l192_192034

noncomputable def tetrahedron_edge_length : ℕ := 24

theorem length_GH_of_tetrahedron
  (a b c d e f : ℕ)
  (h1 : a = 8) 
  (h2 : b = 16) 
  (h3 : c = 24) 
  (h4 : d = 35) 
  (h5 : e = 45) 
  (h6 : f = 55)
  (hEF : f = 55)
  (hEGF : e + b > f)
  (hEHG: e + c > a ∧ e + c > d) 
  (hFHG : b + c > a ∧ b + f > c ∧ c + a > b):
   tetrahedron_edge_length = c := 
sorry

end length_GH_of_tetrahedron_l192_192034


namespace number_of_x_values_l192_192269

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192269


namespace participants_in_robbery_l192_192610

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l192_192610


namespace num_positive_divisors_8_factorial_l192_192940

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192940


namespace num_positive_divisors_8_factorial_l192_192944

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192944


namespace seed_mixture_x_percentage_l192_192549

theorem seed_mixture_x_percentage (x y : ℝ) (h : 0.40 * x + 0.25 * y = 0.30 * (x + y)) : 
  (x / (x + y)) * 100 = 33.33 := sorry

end seed_mixture_x_percentage_l192_192549


namespace percentage_increase_of_gross_sales_l192_192571

theorem percentage_increase_of_gross_sales 
  (P R : ℝ) 
  (orig_gross new_price new_qty new_gross : ℝ)
  (h1 : new_price = 0.8 * P)
  (h2 : new_qty = 1.8 * R)
  (h3 : orig_gross = P * R)
  (h4 : new_gross = new_price * new_qty) :
  ((new_gross - orig_gross) / orig_gross) * 100 = 44 :=
by sorry

end percentage_increase_of_gross_sales_l192_192571


namespace sum_of_longest_altitudes_l192_192965

theorem sum_of_longest_altitudes (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  let h1 := a,
      h2 := b,
      h := (a * b) / c in
  h1 + h2 = 21 := by
{
  sorry
}

end sum_of_longest_altitudes_l192_192965


namespace positive_divisors_8_factorial_l192_192846

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192846


namespace Wendy_bouquets_l192_192557

def num_flowers_before : ℕ := 45
def num_wilted_flowers : ℕ := 35
def flowers_per_bouquet : ℕ := 5

theorem Wendy_bouquets : (num_flowers_before - num_wilted_flowers) / flowers_per_bouquet = 2 := by
  sorry

end Wendy_bouquets_l192_192557


namespace area_of_isosceles_triangle_l192_192520

open Real

theorem area_of_isosceles_triangle 
  (PQ PR QR : ℝ) (PQ_eq_PR : PQ = PR) (PQ_val : PQ = 13) (QR_val : QR = 10) : 
  1 / 2 * QR * sqrt (PQ^2 - (QR / 2)^2) = 60 := 
by 
sorry

end area_of_isosceles_triangle_l192_192520


namespace num_divisors_fact8_l192_192946

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192946


namespace costPrice_of_bat_is_152_l192_192586

noncomputable def costPriceOfBatForA (priceC : ℝ) (profitA : ℝ) (profitB : ℝ) : ℝ :=
  priceC / (1 + profitB) / (1 + profitA)

theorem costPrice_of_bat_is_152 :
  costPriceOfBatForA 228 0.20 0.25 = 152 :=
by
  -- Placeholder for the proof
  sorry

end costPrice_of_bat_is_152_l192_192586


namespace total_amc8_students_l192_192131

theorem total_amc8_students (germain_class newton_class young_class gauss_class : ℕ) 
  (h_germain : germain_class = 12) 
  (h_newton : newton_class = 10) 
  (h_young : young_class = 9) 
  (h_gauss : gauss_class = 7) : 
  germain_class + newton_class + young_class + gauss_class = 38 := 
by 
  rw [h_germain, h_newton, h_young, h_gauss] 
  sorry

end total_amc8_students_l192_192131


namespace angle_range_l192_192737

variables {ℝ : Type*} [linear_ordered_field ℝ] {a b : ℝ^3}

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * ∥a∥ * x^2 + 6 * (a ⬝ b) * x + 7

theorem angle_range (h₀ : ∥a∥ = 2 * real.sqrt 2 * ∥b∥) (h₁ : ∀ x : ℝ, deriv (f x) ≥ 0) :
  ∀ θ ∈ set.Icc 0 (real.pi / 4), ∀ a b, real.angle a b = θ := sorry

end angle_range_l192_192737


namespace sqrt_floor_8_integer_count_l192_192316

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192316


namespace integer_values_of_x_l192_192292

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192292


namespace complex_pure_imaginary_l192_192476

theorem complex_pure_imaginary (a : ℝ) : 
  (a - 2 * complex.I) / (1 + 2 * complex.I) ∈ {z : ℂ | z.re = 0} ↔ a = 4 :=
by { sorry }

end complex_pure_imaginary_l192_192476


namespace limit_of_function_l192_192632

noncomputable def limit_expr (x : ℝ) : ℝ := (1 - real.sqrt (real.cos x)) / (x * real.sin x)
noncomputable def L : ℝ := 1/4

theorem limit_of_function : 
  filter.tendsto limit_expr (nhds 0) (nhds L) :=
sorry

end limit_of_function_l192_192632


namespace problem_equivalent_proof_l192_192183

theorem problem_equivalent_proof :
  let count_N := (finset.range 2500).filter (λ N, (Nat.gcd (N^2 + 5) (N + 4)) > 1) in
  count_N.card = 1310 :=
by
  let count_N := (finset.range 2500).filter (λ N, (Nat.gcd (N^2 + 5) (N + 4)) > 1) in
  exact (count_N.card = 1310)
  sorry

end problem_equivalent_proof_l192_192183


namespace sum_of_consecutive_numbers_mod_13_l192_192177

theorem sum_of_consecutive_numbers_mod_13 :
  ((8930 + 8931 + 8932 + 8933 + 8934) % 13) = 5 :=
by
  sorry

end sum_of_consecutive_numbers_mod_13_l192_192177


namespace num_divisors_of_8_factorial_l192_192877

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192877


namespace percentage_difference_l192_192534

theorem percentage_difference (x : ℝ) (h1 : 0.38 * 80 = 30.4) (h2 : 30.4 - (x / 100) * 160 = 11.2) :
    x = 12 :=
by
  sorry

end percentage_difference_l192_192534


namespace num_divisors_8_fact_l192_192852

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192852


namespace positive_divisors_8_factorial_l192_192843

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192843


namespace num_possible_integer_values_x_l192_192339

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192339


namespace num_pos_divisors_fact8_l192_192773

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192773


namespace number_of_integer_values_l192_192274

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192274


namespace scientific_notation_of_308000000_l192_192656

theorem scientific_notation_of_308000000 :
  ∃ (a : ℝ) (n : ℤ), (a = 3.08) ∧ (n = 8) ∧ (308000000 = a * 10 ^ n) :=
by
  sorry

end scientific_notation_of_308000000_l192_192656


namespace integer_values_of_x_l192_192297

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192297


namespace grandma_vasya_cheapest_option_l192_192245

/-- Constants and definitions for the cost calculations --/
def train_ticket_cost : ℕ := 200
def collected_berries_kg : ℕ := 5
def market_berries_cost_per_kg : ℕ := 150
def sugar_cost_per_kg : ℕ := 54
def jam_made_per_kg_combination : ℕ := 15 / 10  -- representing 1.5 kg (as ratio 15/10)
def ready_made_jam_cost_per_kg : ℕ := 220

/-- Compute the cost per kg of jam for different methods --/
def cost_per_kg_jam_option1 : ℕ := (train_ticket_cost / collected_berries_kg + sugar_cost_per_kg)
def cost_per_kg_jam_option2 : ℕ := market_berries_cost_per_kg + sugar_cost_per_kg
def cost_per_kg_jam_option3 : ℕ := ready_made_jam_cost_per_kg

/-- Numbers converted to per 1.5 kg --/
def cost_for_1_5_kg (cost_per_kg: ℕ) : ℕ := cost_per_kg * (15 / 10)

/-- Theorem stating option 1 is the cheapest --/
theorem grandma_vasya_cheapest_option :
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option2 ∧
  cost_for_1_5_kg cost_per_kg_jam_option1 ≤ cost_for_1_5_kg cost_per_kg_jam_option3 :=
by sorry

end grandma_vasya_cheapest_option_l192_192245


namespace min_k_63_l192_192449

def L_shape (b : fin 9 × fin 9 → bool) (x y : fin 9) : Prop :=
  (b (x, y) = tt ∧ (b (x + 1, y) = tt ∧ b (x + 2, y) = tt) ∨
   b (x, y) = tt ∧ (b (x, y + 1) = tt ∧ b (x, y + 2) = tt) ∨
   b (x, y) = tt ∧ (b (x + 1, y) = tt ∧ b (x, y + 1) = tt) ∨
   b (x, y) = tt ∧ (b (x + 1, y) = tt ∧ b (x, y - 1) = tt))

theorem min_k_63 : ∃ k, (k = 63) ∧ ∀ (b : fin 9 × fin 9 → bool),
  ((∑ i, ∑ j, if b (i, j) then 1 else 0) = k) →
  (∀ x y, ¬ L_shape b x y) :=
begin
  sorry
end

end min_k_63_l192_192449


namespace smallest_five_digit_number_divisible_by_4_is_13492_l192_192054

noncomputable def smallestFiveDigitNumberDivisibleBy4 : ℕ :=
  13492

theorem smallest_five_digit_number_divisible_by_4_is_13492 :
  ∀ (digits : List ℕ), 
    (digits = [1, 2, 3, 4, 9]) →
    ∃ (n : ℕ), 
      (list.perm digits (Int.to_digits 10 n)) ∧
      (10000 ≤ n) ∧ (n < 100000) ∧ (n % 4 = 0) ∧ (n = smallestFiveDigitNumberDivisibleBy4) := 
by
  sorry

end smallest_five_digit_number_divisible_by_4_is_13492_l192_192054


namespace max_volume_height_correct_max_volume_value_correct_l192_192560

def sheet_length := 8
def sheet_width := 5

def volume (x : ℝ) : ℝ := (sheet_length - 2 * x) * (sheet_width - 2 * x) * x

noncomputable def max_volume_height := argmax volume (Icc 0 (min (sheet_length / 2) (sheet_width / 2)))

theorem max_volume_height_correct : max_volume_height = 1 :=
sorry

theorem max_volume_value_correct : volume max_volume_height = 18 :=
sorry

end max_volume_height_correct_max_volume_value_correct_l192_192560


namespace num_pos_divisors_fact8_l192_192785

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192785


namespace num_divisors_8_factorial_l192_192916

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192916


namespace number_of_divisors_8_factorial_l192_192800

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192800


namespace num_divisors_8_factorial_l192_192927

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192927


namespace eval_complex_sum_l192_192161

noncomputable def complex_sum : ℂ :=
∑ n in Finset.range 41, (complex.i ^ n) * (Real.sin (30 + 60 * n)).toReal

theorem eval_complex_sum :
  complex_sum = 1 / 2 + 15 * complex.i :=
by
  sorry

end eval_complex_sum_l192_192161


namespace total_cost_price_of_all_items_l192_192590

def SP_A := 100
def profit_A := 0.15

def SP_B := 120
def profit_B := 0.20

def SP_C := 200
def profit_C := 0.10

noncomputable def CP_A := SP_A / (1 + profit_A)
noncomputable def CP_B := SP_B / (1 + profit_B)
noncomputable def CP_C := SP_C / (1 + profit_C)

noncomputable def total_CP : Float := CP_A + CP_B + CP_C

theorem total_cost_price_of_all_items :
    total_CP ≈ 368.78 :=
by
    sorry

end total_cost_price_of_all_items_l192_192590


namespace num_divisors_8_fact_l192_192855

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192855


namespace area_of_rectangle_l192_192566

/-- Representation of the problem conditions and statement of the theorem -/
theorem area_of_rectangle
  (r : ℝ) (h : r = 6)
  (w : ℝ) (h_w : w = 2 * r)
  (ratio : ℝ) (h_ratio : ratio = 3)
  (l : ℝ) (h_l : l = ratio * w) :
  l * w = 432 :=
by
  -- Definitions based on the given conditions
  have h_r₀ : r = 6 := h,
  have h_w₀ : w = 2 * r := h_w,
  have h_l₀ : l = ratio * w := h_l,
  -- Calculate area
  rw [h_r₀, h_w₀, h_l₀], -- Replace r, w and l with their defined values
  calculate_area : l * w = 432 := sorry -- Proof of the final step
  exact calculate_area

end area_of_rectangle_l192_192566


namespace question1_question2_l192_192730

noncomputable def f1 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2
noncomputable def f2 (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x^2 - 2*x

theorem question1 (a : ℝ) : 
  (∀ x : ℝ, f1 a x = 0 → ∀ y : ℝ, f1 a y = 0 → x = y) ↔ (a = 0 ∨ a < -4 / Real.exp 2) :=
sorry -- Proof of theorem 1

theorem question2 (a m n x0 : ℝ) (h : a ≠ 0) :
  (f2 a x0 = f2 a ((x0 + m) / 2) * (x0 - m) + n ∧ x0 ≠ m) → False :=
sorry -- Proof of theorem 2

end question1_question2_l192_192730


namespace integer_values_of_x_l192_192295

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192295


namespace functional_sum_l192_192979

theorem functional_sum :
  (∃ f : ℕ → ℝ, (∀ a b : ℕ, f (a + b) = f a * f b) ∧ f 1 = 1) →
  (∑ k in finset.range (2005 - 1 + 1), (λ k, f (k + 1) / f k) = 2004) :=
by {
  sorry
}

end functional_sum_l192_192979


namespace distance_midpoint_BC_to_A_square_folded_along_BD_l192_192591

theorem distance_midpoint_BC_to_A_square_folded_along_BD :
  let A := (0, 0)
      B := (4, 0)
      C := (4, 4)
      D := (0, 4)
      E := (3, 2)
      OA := 2 * Real.sqrt 2
      AC := 2 * Real.sqrt 2
  in dist (E.1, E.2) A = 2 * Real.sqrt 2 :=
by sorry

end distance_midpoint_BC_to_A_square_folded_along_BD_l192_192591


namespace positive_divisors_8_factorial_l192_192835

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192835


namespace sin_pi_theta_sin_half_pi_theta_l192_192695

theorem sin_pi_theta_sin_half_pi_theta (θ : ℝ) (h1 : sin θ = 1 / 3) (h2 : θ ∈ set.Ioo (-(real.pi / 2)) (real.pi / 2)) :
  sin (real.pi - θ) * sin (real.pi / 2 - θ) = 2 * real.sqrt 2 / 9 := 
by
  sorry

end sin_pi_theta_sin_half_pi_theta_l192_192695


namespace range_of_t_l192_192209

open Real

noncomputable def f (x : ℝ) : ℝ := x / log x
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - exp 1 * x + exp 1 ^ 2)

theorem range_of_t :
  (∀ x > 1, ∀ t > 0, (t + 1) * g x ≤ t * f x)
  ↔ (∀ t > 0, t ≥ 1 / (exp 1 ^ 2 - 1)) :=
by
  sorry

end range_of_t_l192_192209


namespace pattern_of_balls_l192_192089

-- Define the set of balls and properties
def balls : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

-- Define the properties of odd and even balls
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the selection of 3 balls with replacement
def select_balls : (balls × balls × balls) → Prop :=
  λ (x : balls × balls × balls), true  -- This defines a generic selection

-- Define the probability of selecting an odd ball first
def probability_first_odd : ℝ := 2 / 3

-- Theorem statement
theorem pattern_of_balls (b1 b2 b3 : ℕ) (h1 : b1 ∈ balls) (h2 : b2 ∈ balls) (h3 : b3 ∈ balls) :
  (select_balls (b1, b2, b3) →
  (prob_first_odd := probability_first_odd → 
  (is_odd b1 ∧ is_odd b2 ∧ is_even b3) ∨
  (is_odd b1 ∧ is_even b2 ∧ is_odd b3) ∨
  (is_even b1 ∧ is_odd b2 ∧ is_odd b3))) := sorry

end pattern_of_balls_l192_192089


namespace sum_diagonals_and_sides_dodecagon_l192_192990

def num_diagonals (n : ℕ) := n * (n - 3) / 2

theorem sum_diagonals_and_sides_dodecagon : 
  let n := 12 in (num_diagonals n + n) = 66 :=
by
  let n := 12
  have : num_diagonals n = 54 := 
    by
      unfold num_diagonals
      sorry
  show num_diagonals n + n = 66
  sorry

end sum_diagonals_and_sides_dodecagon_l192_192990


namespace nikola_ants_l192_192439

-- Define the conditions
def ounces_per_ant : ℝ := 2
def cost_per_ounce : ℝ := 0.1
def charge_per_job : ℝ := 5
def cost_per_leaf : ℝ := 0.01
def leaves_raked : ℕ := 6000
def jobs_completed : ℕ := 4
def total_money_saved : ℝ :=
  (leaves_raked * cost_per_leaf) + (jobs_completed * charge_per_job)

-- Define the cost to feed one ant
def cost_per_ant : ℝ :=
  ounces_per_ant * cost_per_ounce

-- Define the number of ants Nikola can feed
def num_ants : ℕ :=
  total_money_saved / cost_per_ant

theorem nikola_ants : 
  num_ants = 400 := by
sorry

end nikola_ants_l192_192439


namespace johns_change_l192_192394

theorem johns_change
  (num_oranges : ℤ) 
  (cost_per_orange : ℝ) 
  (amount_paid : ℝ)
  (h_oranges : num_oranges = 4)
  (h_cost : cost_per_orange = 0.75)
  (h_paid : amount_paid = 10.00) :
  amount_paid - num_oranges * cost_per_orange = 7.00 :=
by 
  rw [h_oranges, h_cost, h_paid]
  norm_num
  sorry

end johns_change_l192_192394


namespace count_possible_integer_values_l192_192283

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192283


namespace swimming_pool_people_count_l192_192441

theorem swimming_pool_people_count :
  let child_price := 1.50
  let adult_price := 2.25
  let total_receipts := 1422.00
  let number_of_children := 388
  let total_from_children := number_of_children * child_price
  let total_from_adults := total_receipts - total_from_children
  let number_of_adults := total_from_adults / adult_price
  number_of_children + ⌊number_of_adults⌋ = 761 := sorry

end swimming_pool_people_count_l192_192441


namespace solution_set_f_leq_f3_l192_192726

noncomputable def f (a x : ℝ) : ℝ := log a (x^2) + a^(abs x)

theorem solution_set_f_leq_f3 (a x : ℝ) (h : a > 1) :
  (f a (-3) < f a 4) →
  (f a (x^2 - 2*x) ≤ f a 3) ↔ x ∈ [-1, 0) ∪ (0, 3] :=
by
  sorry

end solution_set_f_leq_f3_l192_192726


namespace pos_divisors_8_factorial_l192_192908

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192908


namespace perimeter_triangle_APR_l192_192045

-- Given conditions
variables (A B C Q P R : Point)
variable (circle : Circle)
variable (AB AC PQ QR : Line)
variable (hA_B_tangent : Tangent A circle B)
variable (hA_C_tangent : Tangent A circle C)
variable (hQ_tangent : Tangent Q circle)
variable (hA_eq : distance A B = 24)
variable (hPQ_intersect : intersects PQ AB P)
variable (hQR_intersect : intersects QR (line_through C R) R)

theorem perimeter_triangle_APR :
  perimeter (triangle A P R) = 48 :=
sorry

end perimeter_triangle_APR_l192_192045


namespace divide_ratio_of_E_l192_192384

noncomputable def point_divides (A B C : Type) [ordered_ring A] :=
  ∀ (F E G : A),
  divides F (A, C) (3, 2) → intersects E (B, C) (AG) (G, divides G (B, F) (2, 1)) →
  (E : divides (B, C) (2, 5))

theorem divide_ratio_of_E (A B C : Type) [ordered_ring A]
  (F E G : A)
  (h1 : divides F (A, C) (3, 2))
  (h2 : intersects E (B, C) (AG) (G, divides G (B, F) (2, 1))) :
  divides (E, (B, C)) (2, 5) :=
sorry

end divide_ratio_of_E_l192_192384


namespace sqrt_floor_8_integer_count_l192_192323

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192323


namespace num_pos_divisors_fact8_l192_192783

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192783


namespace value_of_frac_l192_192350

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end value_of_frac_l192_192350


namespace num_possible_integer_values_x_l192_192335

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192335


namespace dihedral_angle_cosine_l192_192526

theorem dihedral_angle_cosine (r₁ r₂ : ℝ) (h₁ : r₂ = 1.5 * r₁)
    (d : ℝ) (h₂ : d = r₁ + r₂) : 
    let θ : ℝ := 30 * Real.pi / 180 in
    cos θ = 0.68 :=
by
  sorry

end dihedral_angle_cosine_l192_192526


namespace prob_divisors_8_fact_l192_192802

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192802


namespace infinite_solutions_l192_192454

theorem infinite_solutions (M : ℕ) : ∃ (a : ℕ), (∃ (xy_set : Finset (ℕ × ℕ)), (xy_set.card ≥ 1980 ∧ ∀ (x y : ℕ), (x, y) ∈ xy_set → (floor (x ^ (3/2)) + floor (y ^ (3/2)) = a))) :=
by
  sorry

end infinite_solutions_l192_192454


namespace pos_divisors_8_factorial_l192_192906

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192906


namespace triangle_area_l192_192629

noncomputable def semiPerimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heronsFormula (a b c : ℝ) : ℝ := 
  let s := semiPerimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area {a b c : ℝ} (ha : a = 30) (hb : b = 21) (hc : c = 10) : 
  heronsFormula a b c ≈ 54.52 := by
  sorry

end triangle_area_l192_192629


namespace more_than_10_numbers_with_sum_20_l192_192000

theorem more_than_10_numbers_with_sum_20
    (a : ℕ → ℕ)
    (len : ℕ)
    (sum_eq_20 : ∑ i in finset.range len, a i = 20)
    (no_elem_eq_3 : ∀ i < len, a i ≠ 3)
    (no_consec_sum_eq_3 : ∀ i j, 0 ≤ i → i < j → j ≤ len → (∑ k in finset.range (j - i), a (i + k)) ≠ 3) :
  len > 10 :=
begin
  sorry
end

end more_than_10_numbers_with_sum_20_l192_192000


namespace chessboard_painting_l192_192562

theorem chessboard_painting (n : ℕ) (m : ℕ) : n = 4 → m = 4 → 
  (∃ f : fin n × fin n → Prop, 
  (∀ r, (finset.univ.filter (λ c, f (r, c))).card = 2) ∧ 
  (∀ c, (finset.univ.filter (λ r, f (r, c))).card = 2)) → 
  (nat.choose 4 2 + nat.choose 4 2 * nat.choose 4 2) + nat.choose 4 2 * 4 * 2 + nat.choose 4 2 * 1 = 90 := by
  intros h_n h_m h_paint
  sorry

end chessboard_painting_l192_192562


namespace triangle_cos_b_l192_192991

-- Definitions according to the conditions
variables {a b c : ℝ}
variables (h1 : b^2 = a * c) (h2 : c = 2 * a)

-- The theorem to be proven
theorem triangle_cos_b (h1 : b^2 = a * c) (h2 : c = 2 * a) : 
  Real.cos_angle (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 3 / 4 :=
  sorry

end triangle_cos_b_l192_192991


namespace solve_quadratics_l192_192003

noncomputable theory
open Classical

theorem solve_quadratics :
  (∃ x : ℝ, 2 * x * (x - 1) = 1 ∧ (x = (1 + Real.sqrt 3) / 2 ∨ x = (1 - Real.sqrt 3) / 2)) ∧
  (∃ x : ℝ, x^2 + 8 * x + 7 = 0 ∧ (x = -7 ∨ x = -1)) :=
by
  sorry

end solve_quadratics_l192_192003


namespace number_of_good_functions_l192_192414

-- Define the context of the problem
def p : ℕ := 2017

def F_p := zmod p

def is_good_function (f : ℤ → F_p) (α : F_p) : Prop :=
∀ x y : ℤ, f x * f y = f (x + y) + (α ^ y) * f (x - y)

def is_periodic (f : ℤ → F_p) (n : ℕ) : Prop :=
∀ x : ℤ, f (x + n) = f x

-- Define the main theorem to be proved
theorem number_of_good_functions (α : F_p) (hα : α ≠ 0)
  (h_period : ∀ (f : ℤ → F_p), is_periodic f 2016 → is_good_function f α) :
∃ f : ℕ, f = 1327392 :=
begin
  sorry
end

end number_of_good_functions_l192_192414


namespace KP_bisects_CD_iff_LC_eq_LD_l192_192998

variables (A B C D P K L : Point)
(non_parallel : ¬ parallel AD BC)
(exists_circle : ∃ (Γ : Circle), tangent Γ BC C ∧ tangent Γ AD D ∧ intersects Γ AB K ∧ intersects Γ AB L)
(intersect_AC_BD : intersection AC BD = some P)

theorem KP_bisects_CD_iff_LC_eq_LD 
  (line_KP_bisects_CD : line KP bisects segment CD) : LC = LD ↔ KP bisects segment CD := 
sorry

end KP_bisects_CD_iff_LC_eq_LD_l192_192998


namespace total_expenditure_now_l192_192999

-- Define the conditions in Lean
def original_student_count : ℕ := 100
def additional_students : ℕ := 25
def decrease_in_average_expenditure : ℤ := 10
def increase_in_total_expenditure : ℤ := 500

-- Let's denote the original average expenditure per student as A rupees
variable (A : ℤ)

-- Define the old and new expenditures
def original_total_expenditure := original_student_count * A
def new_average_expenditure := A - decrease_in_average_expenditure
def new_total_expenditure := (original_student_count + additional_students) * new_average_expenditure

-- The theorem to prove
theorem total_expenditure_now :
  new_total_expenditure A - original_total_expenditure A = increase_in_total_expenditure →
  new_total_expenditure A = 7500 :=
by
  sorry

end total_expenditure_now_l192_192999


namespace ellipse_equation_and_max_area_l192_192206

noncomputable def ellipse (a b : Float) (x y : Float) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def slope (x1 y1 x2 y2 : Float) : Float := 
  (y2 - y1) / (x2 - x1)

theorem ellipse_equation_and_max_area {a b : Float} (h1 : a > b) (h2 : b > 0)
  (h3 : a^2 - b^2 = 3) 
  (h4 : ∀ x1 y1 x2 y2, (slope x1 y1 x2 y2) * (slope x2 y2 (-x1) (-y1)) = -1) :
  (ellipse 2 1) ∧ (∀ (x1 y1 : Float) (h5 : |(x1 * y1)| ≤ 1 / 2 * 4 * y1^2 + 1), 
  (|x1| = 2 * |y1|) → (1 = (x1^2 / 4) + y1^2) → 
  max_area : Float := 9/8) := 
sorry

end ellipse_equation_and_max_area_l192_192206


namespace percentage_of_time_spent_watching_comet_l192_192250

noncomputable def time_spent_shopping_telescope : ℕ := 2 * 60
noncomputable def time_spent_buying_binoculars : ℕ := 1 * 60 + 15
noncomputable def time_spent_setting_up_stargazing : ℕ := (3 / 2 : ℚ) * 60
noncomputable def time_spent_preparing_snacks : ℚ := 3 * time_spent_setting_up_stargazing
noncomputable def time_spent_observing_before_comet : ℚ := (40 * 0.75 : ℚ)
noncomputable def time_spent_watching_comet : ℕ := 20
noncomputable def time_spent_observing_after_comet : ℕ := 50

noncomputable def total_time_spent : ℚ := time_spent_shopping_telescope + time_spent_buying_binoculars + time_spent_setting_up_stargazing + time_spent_preparing_snacks + time_spent_observing_before_comet + time_spent_watching_comet + time_spent_observing_after_comet

noncomputable def percentage_time_watching_comet : ℚ := (time_spent_watching_comet / total_time_spent) * 100

theorem percentage_of_time_spent_watching_comet : percentage_time_watching_comet ≈ 3 :=
sorry

end percentage_of_time_spent_watching_comet_l192_192250


namespace math_problem_proof_l192_192067

variable {α : Type} [LinearOrderedField α]

def statement_A (a : α) : Prop :=
  (∀ x y : α, x = 3 → y = 2 → (y = a * (x - 3) + 2))

def statement_B (m : α) (x1 y1 : α) : Prop :=
  (m = -Real.sqrt 3 → x1 = 2 → y1 = -1 → (y1 + 1 = m * (x1 - 2)))

def statement_C (m b : α) : Prop :=
  (m = -2 → b = 3 → (∀ x y : α, (y = m * x + b → ¬ (y = m * x ± b))))

def statement_D (x y : α) : Prop :=
  (x = 1 → y = 1 → (∀ a : α, x/a + y/a = 1 → (a = 2 → (x + y - 2 = 0))))

def correct_statements (a m x1 y1 b x y : α) : Prop :=
 (statement_A a ∧ statement_B m x1 y1 ∧ ¬ statement_C m b ∧ statement_D x y)

theorem math_problem_proof (a m x1 y1 b x y : α) :
  correct_statements a m x1 y1 b x y → (statement_A a ∧ statement_B m x1 y1) :=
by
  intro h
  sorry

end math_problem_proof_l192_192067


namespace find_angle_B_l192_192383

-- Definitions for the problem statement
def in_triangle (A B C a b c : ℝ) := 
  A + B + C = π ∧ 
  a = b * sin(C) / sin(A) ∧ 
  c = b * sin(A) / sin(C)

-- The theorem statement that needs to be proved
theorem find_angle_B (A B C a b c : ℝ) (h_triangle : in_triangle A B C a b c)
  (h_condition : a * cos C + c * cos A = 2 * b * cos B) : B = π / 3 :=
by
  sorry

end find_angle_B_l192_192383


namespace johns_change_l192_192395

theorem johns_change
  (num_oranges : ℤ) 
  (cost_per_orange : ℝ) 
  (amount_paid : ℝ)
  (h_oranges : num_oranges = 4)
  (h_cost : cost_per_orange = 0.75)
  (h_paid : amount_paid = 10.00) :
  amount_paid - num_oranges * cost_per_orange = 7.00 :=
by 
  rw [h_oranges, h_cost, h_paid]
  norm_num
  sorry

end johns_change_l192_192395


namespace num_divisors_8_factorial_l192_192755

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192755


namespace grey_rectangles_area_l192_192448

-- The side length of the blue square
def blue_square_side : ℝ := Real.sqrt 36

-- The area of the mixed center rectangle (black, blue, and white parts)
def mixed_rectangle_area : ℝ := 78

-- The side length of the red square
def red_square_side : ℝ := Real.sqrt 49

-- The entire side length of the canvas
def canvas_side_length : ℝ := blue_square_side + red_square_side

-- The total area of the canvas
def canvas_area : ℝ := canvas_side_length ^ 2

-- The area of the right half of the canvas
def right_half_area : ℝ := canvas_area / 2

-- The sum of the areas of the grey rectangles
def sum_grey_areas : ℝ := (6 * 13) - mixed_rectangle_area

theorem grey_rectangles_area :
  sum_grey_areas = 42 := by
  sorry

end grey_rectangles_area_l192_192448


namespace positive_divisors_8_factorial_l192_192840

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192840


namespace count_possible_integer_values_l192_192282

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192282


namespace probability_of_odd_numbered_ball_first_l192_192091

/-- Problem Statement: -/
theorem probability_of_odd_numbered_ball_first :
  let balls := finset.range 1 101 -- Balls numbered from 1 to 100
  let odd_balls := finset.filter (λ n, n % 2 = 1) balls -- Odd-numbered balls
  let even_balls := finset.filter (λ n, n % 2 = 0) balls -- Even-numbered balls
  let total_balls := 100 -- Total number of balls
  let odd_count := finset.card odd_balls -- Count of odd-numbered balls
  let even_count := finset.card even_balls -- Count of even-numbered balls
  let total_count := odd_count + even_count -- Total count of odd and even balls
  let probabilty_of_odd := (odd_count / total_count : ℚ) -- Desired probability of selecting an odd-numbered ball first
  (2 / 3 : ℚ) = probabilty_of_odd :=
by
  sorry

end probability_of_odd_numbered_ball_first_l192_192091


namespace num_divisors_fact8_l192_192954

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192954


namespace quadrilateral_angles_and_sides_correct_l192_192169

-- Define the isosceles trapezoid and its properties
structure IsoscelesTrapezoid (A B C D : Type) :=
(diagonals_equal : Real)
(diagonals_angle : Real)

-- Define the quadrilateral formed by the midpoints
def Quadrilateral (A B C D : Type) :=
  { quadrilateral : Type // true }

-- Define the quadrilateral properties we need to prove
theorem quadrilateral_angles_and_sides_correct (A B C D : Type) 
  (T : IsoscelesTrapezoid A B C D)
  (midpoints : Quadrilateral A B C D) :
  midpoints.quadrilateral.angles = [40, 140, 40, 140] ∧
  midpoints.quadrilateral.sides = [5, 5, 5, 5] := 
sorry

end quadrilateral_angles_and_sides_correct_l192_192169


namespace slope_tangent_at_pi_over_4_l192_192035

noncomputable def f (x : ℝ) := Real.sin x

theorem slope_tangent_at_pi_over_4 : 
  (Real.deriv f) (Real.pi / 4) = Real.sqrt 2 / 2 :=
sorry

end slope_tangent_at_pi_over_4_l192_192035


namespace min_distance_point_P_l192_192716

def point_A : ℝ × ℝ := (-3, 2)
def parabola (x y : ℝ) : Prop := y^2 = -4 * x
def focus_F : ℝ × ℝ := (-1, 0)

theorem min_distance_point_P : 
  ∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧ ∀ Q, parabola Q.1 Q.2 → 
  |dist point_A P + dist P focus_F| = 4 → 
  P = (-1, 2) := 
sorry

end min_distance_point_P_l192_192716


namespace fraction_of_prize_money_l192_192088

theorem fraction_of_prize_money 
  (total_prize_money : ℝ)
  (remaining_prize_money : ℝ)
  (individual_prize : ℝ)
  (f : ℝ)
  (H1 : total_prize_money = 2400)
  (H2 : remaining_prize_money = total_prize_money - f * total_prize_money)
  (H3 : individual_prize = remaining_prize_money / 10)
  (H4 : individual_prize = 160) :
  f = (1 / 3) :=
by
  have H5 : total_prize_money - f * total_prize_money = 1600, from sorry,
  have H6 : 2400 * f = 800, from sorry,
  have H7 : f = 800 / 2400, from sorry,
  exact sorry

end fraction_of_prize_money_l192_192088


namespace robbery_proof_l192_192617

variables (A B V G : Prop)

-- Define the conditions as Lean propositions
def condition1 : Prop := ¬G → (B ∧ ¬A)
def condition2 : Prop := V → (¬A ∧ ¬B)
def condition3 : Prop := G → B
def condition4 : Prop := B → (A ∨ V)

-- The statement we want to prove based on conditions
theorem robbery_proof (h1 : condition1 A B G) 
                      (h2 : condition2 A B V) 
                      (h3 : condition3 B G) 
                      (h4 : condition4 A B V) : 
                      A ∧ B ∧ G :=
begin
  sorry
end

end robbery_proof_l192_192617


namespace number_of_integer_values_l192_192276

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192276


namespace num_divisors_8_fact_l192_192862

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192862


namespace num_pos_divisors_fact8_l192_192784

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192784


namespace rabbit_hopping_time_l192_192581

noncomputable def rabbit_time_in_minutes (distance : ℝ) (speed : ℝ) : ℝ := (distance / speed) * 60

theorem rabbit_hopping_time (distance : ℝ) (speed : ℝ) (h_distance : distance = 2) (h_speed : speed = 5) :
  rabbit_time_in_minutes distance speed = 24 :=
by
  rw [h_distance, h_speed]
  unfold rabbit_time_in_minutes
  norm_num
  sorry

end rabbit_hopping_time_l192_192581


namespace series_evaluation_l192_192425

theorem series_evaluation : 
  let n := 1990 in 
  (1 / (2^n : ℝ)) * (∑ k in finset.range (n // 2 + 1), (-3)^k * (nat.choose n (2 * k))) = -1 / 2 := 
by 
  let n := 1990 
  have h : ∑ k in finset.range (n // 2 + 1), (-3)^k * (nat.choose n (2 * k)) = -((2^1990 * -1 / 2) : ℝ) 
  sorry 
  rw h 
  ring 
  have h2 : 2^1990 ≠ 0 := pow_ne_zero 1990 (by norm_num) 
  field_simp [h2] 
  norm_num 
  repeat { rw pow_add } 
  norm_num 
  sorry 

end series_evaluation_l192_192425


namespace kate_hair_length_l192_192407

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end kate_hair_length_l192_192407


namespace total_number_of_drivers_l192_192039

theorem total_number_of_drivers (N : ℕ) (A_drivers : ℕ) (B_sample : ℕ) (C_sample : ℕ) (D_sample : ℕ)
  (A_sample : ℕ)
  (hA : A_drivers = 96)
  (hA_sample : A_sample = 12)
  (hB_sample : B_sample = 21)
  (hC_sample : C_sample = 25)
  (hD_sample : D_sample = 43) :
  N = 808 :=
by
  -- skipping the proof here
  sorry

end total_number_of_drivers_l192_192039


namespace common_tangent_problem_l192_192224

theorem common_tangent_problem
  {x1 y1 x2 y2 p : ℝ}
  (h_curve : y1 = -1 / x1)
  (h_parabola : y2^2 = 2 * p * x2)
  (h_tangent_points : p > 0 ∧ x2 > 0 ∧ (dist (x1, y1) (x2, y2) = 3 * sqrt 10 / 2))
  (h_x1y1 : x1 * y1 = -1)
  (h_p:
    p = sqrt 2 ∨ 
    p = 8 * sqrt 2)
  : x1 * y1 = -1 ∧
    (p = sqrt 2 ∨ p = 8 * sqrt 2) := 
  sorry

end common_tangent_problem_l192_192224


namespace abs_diff_51st_terms_correct_l192_192042

-- Definition of initial conditions for sequences A and C
def seqA_first_term : ℤ := 40
def seqA_common_difference : ℤ := 8

def seqC_first_term : ℤ := 40
def seqC_common_difference : ℤ := -5

-- Definition of the nth term function for an arithmetic sequence
def nth_term (a₁ d n : ℤ) : ℤ := a₁ + d * (n - 1)

-- 51st term of sequence A
def a_51 : ℤ := nth_term seqA_first_term seqA_common_difference 51

-- 51st term of sequence C
def c_51 : ℤ := nth_term seqC_first_term seqC_common_difference 51

-- Absolute value of the difference
def abs_diff_51st_terms : ℤ := Int.natAbs (a_51 - c_51)

-- The theorem to be proved
theorem abs_diff_51st_terms_correct : abs_diff_51st_terms = 650 := by
  sorry

end abs_diff_51st_terms_correct_l192_192042


namespace two_bedroom_units_l192_192095

theorem two_bedroom_units (x y : ℕ) (h1 : x + y = 12) (h2 : 360 * x + 450 * y = 4950) : y = 7 :=
by
  sorry

end two_bedroom_units_l192_192095


namespace flower_shop_options_l192_192436

theorem flower_shop_options:
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, 2 * p.1 + 3 * p.2 = 30 ∧ p.1 > 0 ∧ p.2 > 0) ∧ S.card = 4 :=
sorry

end flower_shop_options_l192_192436


namespace num_ways_sum_ordered_2010_l192_192175

theorem num_ways_sum_ordered_2010 :
  ∃ n : ℕ, n = 2010 ∧ ∀ (a : ℕ → ℕ) (length : ℕ),
    length = 2010 →
    (∀ i, 0 < i ∧ i ≤ length → a i ≤ a (i + 1)) ∧
    (a length - a 1 ≤ 1) →
    ∃k (k ∈ (list.range 2011)), k = n ∧
    (∀ i, 0 < i ∧ i ≤ k → a i = ⌊2010 / ↑length⌋ ∨ a i = ⌊2010 / ↑length⌋ + 1) :=
sorry

end num_ways_sum_ordered_2010_l192_192175


namespace not_valid_base_five_l192_192016

theorem not_valid_base_five (k : ℕ) (h₁ : k = 5) : ¬(∀ d ∈ [3, 2, 5, 0, 1], d < k) :=
by
  sorry

end not_valid_base_five_l192_192016


namespace total_arts_students_l192_192114

theorem total_arts_students (total_students : ℕ) (sample_drawn : ℕ) (selection_prob : ℝ) 
  (stratified_sample : ℕ) (science_students : ℕ) (arts_students : ℕ)
  (h_total_students : sample_drawn = 60 ∧ selection_prob = 0.05 ∧ total_students = 1200)
  (h_stratified_sample : stratified_sample = 30 ∧ science_students = 24)
  (h_arts_students : arts_students = stratified_sample - science_students) :
  ((total_students : ℝ) * (arts_students : ℝ / stratified_sample : ℝ) = 240) :=
begin
  sorry
end

end total_arts_students_l192_192114


namespace num_divisors_8_factorial_l192_192920

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192920


namespace number_of_divisors_of_8_fact_l192_192887

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192887


namespace positive_divisors_of_8_factorial_l192_192821

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192821


namespace dilation_result_l192_192018

noncomputable def dilation (c a : ℂ) (k : ℝ) : ℂ := k * (c - a) + a

theorem dilation_result :
  dilation (3 - 1* I) (1 + 2* I) 4 = 9 + 6* I :=
by
  sorry

end dilation_result_l192_192018


namespace number_of_x_values_l192_192263

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192263


namespace area_grazing_l192_192573

-- Define the field dimensions
def length_of_field : ℝ := 42
def width_of_field : ℝ := 26

-- Define the length of the rope
def length_of_rope : ℝ := 16

-- Define π
def pi : ℝ := Real.pi

-- Define the radius of the grazing area (same as the length of the rope)
def radius_of_grazing_area : ℝ := length_of_rope

-- Define the area of the quarter-circle grazing area
def grazing_area : ℝ := (1 / 4) * pi * radius_of_grazing_area ^ 2

-- Define the approximate value of the grazing area using π approximation
def approximate_grazing_area : ℝ := 64 * 3.14159

-- Prove that the grazing area is approximately 201.06 square meters
theorem area_grazing : grazing_area ≈ 201.06 :=
by
  sorry

end area_grazing_l192_192573


namespace ferry_speed_difference_l192_192186

variable (v_P v_Q d_P d_Q t_P t_Q x : ℝ)

-- Defining the constants and conditions provided in the problem
axiom h1 : v_P = 8 
axiom h2 : t_P = 2 
axiom h3 : d_P = t_P * v_P 
axiom h4 : d_Q = 3 * d_P 
axiom h5 : t_Q = t_P + 2
axiom h6 : d_Q = v_Q * t_Q 
axiom h7 : x = v_Q - v_P 

-- The theorem that corresponds to the solution
theorem ferry_speed_difference : x = 4 := by
  sorry

end ferry_speed_difference_l192_192186


namespace num_divisors_8_fact_l192_192853

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192853


namespace range_of_m_l192_192228

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), -2 ≤ x ∧ x ≤ 3 ∧ m * x + 6 = 0) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end range_of_m_l192_192228


namespace g_sum_l192_192689

def g (n : ℕ) : ℝ := Real.log (n ^ 2) / Real.log 3003

theorem g_sum :
  g 7 + g 11 + g 13 = 2 :=
by
  sorry

end g_sum_l192_192689


namespace ellipse_equation_and_m_value_l192_192705

variable {a b : ℝ}
variable (e : ℝ) (F : ℝ × ℝ) (h1 : e = Real.sqrt 2 / 2) (h2 : F = (1, 0))

theorem ellipse_equation_and_m_value (h3 : a > b) (h4 : b > 0) 
  (h5 : (x y : ℝ) → (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1 → (x - 1) ^ 2 + y ^ 2 = 1) :
  (a = Real.sqrt 2 ∧ b = 1) ∧
  (∀ m : ℝ, (y = x + m) → 
  ((∃ A B : ℝ × ℝ, A = (x₁, x₁ + m) ∧ B = (x₂, x₂ + m) ∧
  (x₁ ^ 2) / 2 + (x₁ + m) ^ 2 = 1 ∧ (x₂ ^ 2) / 2 + (x₂ + m) ^ 2 = 1 ∧
  x₁ * x₂ + (x₁ + m) * (x₂ + m) = -1) ↔ m = Real.sqrt 3 / 3 ∨ m = - Real.sqrt 3 / 3))
  :=
sorry

end ellipse_equation_and_m_value_l192_192705


namespace max_angle_APB_is_60_degrees_l192_192422

noncomputable def max_angle_APB : ℝ :=
  let P_line : set (ℝ × ℝ) := {p | p.1 - p.2 = 0}
  let circle_center : ℝ × ℝ := (4, 0)
  let circle_radius : ℝ := real.sqrt 2
  let tangent_points (p : ℝ × ℝ) : Prop :=
    (p.1 - 4) ^ 2 + p.2 ^ 2 = circle_radius ^ 2 × (p ∈ P_line)
  let P : (ℝ × ℝ) := classical.some tangents_points
  ∠ (classical.some tangent_points P) (classical.some tangent_points P) P

theorem max_angle_APB_is_60_degrees : max_angle_APB = 60 :=
sorry

end max_angle_APB_is_60_degrees_l192_192422


namespace num_divisors_8_factorial_l192_192766

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192766


namespace problem_l192_192646

noncomputable def f (x : ℝ) : ℝ :=
if x > -1 ∧ x < 0 then 2^x + 1/5 
else if ∃ n : ℤ, x = n + 4 * (x - n) / 4 then f(x - 4)
else if ∃ n : ℤ, x = -n then -f(n)
else x

theorem problem (x : ℝ) (hx : x = Real.log 20 / Real.log 2) : 
  (f x = f (x - 4)) ∧ (f (-x) = -f x) 
  → f (Real.log 20 / Real.log 2) = -1 := 
sorry

end problem_l192_192646


namespace maximum_b_value_l192_192220

theorem maximum_b_value 
  (a b : ℤ) 
  (h : 127 * b - 16 * a = a * b) 
  (hb : ∀ a' b': ℤ, 127 * b' - 16 * a' = a' * b' → b ≤ b') : 
  b = 2016 :=
begin
  sorry
end

end maximum_b_value_l192_192220


namespace optimal_production_l192_192127

noncomputable def optimal_strategy_natural_gas := 3032
noncomputable def optimal_strategy_liquefied_gas := 2954

theorem optimal_production (mild_natural mild_liquefied severe_natural severe_liquefied cost_natural cost_liquefied price_natural price_liquefied : ℕ) :
  (mild_natural = 2200 ∧ mild_liquefied = 3500 ∧ severe_natural = 3800 ∧ severe_liquefied = 2450) →
  (cost_natural = 19 ∧ cost_liquefied = 25 ∧ price_natural = 35 ∧ price_liquefied = 58) →
  (optimal_strategy_natural_gas = 3032 ∧ optimal_strategy_liquefied_gas = 2954) :=
by
  intros,
  sorry

end optimal_production_l192_192127


namespace evaluate_expression_at_minus_two_l192_192538

theorem evaluate_expression_at_minus_two : ∀ (x : ℝ), x = -2 → x^2 + 6 * x - 7 = -15 := by
  intros x hx
  rw [hx]
  calc
    (-2)^2 + 6 * (-2) - 7 = 4 - 12 - 7 := by norm_num
    ... = -15 := by norm_num

end evaluate_expression_at_minus_two_l192_192538


namespace number_of_x_values_l192_192264

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192264


namespace smallest_a2_l192_192589

def sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ ∃ A2 : ℝ, A2 > 0 ∧ (∀ n, 1 ≤ n ∧ n ≤ 7 → a (n+2) * a n * a (n-1) = a (n+2) + a n + a (n-1))

theorem smallest_a2 : 
  (∀ a : ℕ → ℝ, sequence a →  ¬∃ a10 : ℝ, a10 * a 8 * a 7 = a10 + a 8 + a 7 → a 2 = sqrt 2 - 1) :=
sorry

end smallest_a2_l192_192589


namespace cosine_dihedral_angle_l192_192528

/-- Two spheres of radius r1 and 1.5 * r1 are inscribed in a dihedral angle
    and touch each other. The line connecting their centers forms a 30-degree angle
    with the edge of the dihedral angle. Prove that the cosine of this dihedral
    angle is approximately 0.68. -/
theorem cosine_dihedral_angle (r1 : ℝ) (r2 : ℝ) (h : r2 = 1.5 * r1) 
    (angle_between_centers_edge : ℝ) (h_angle : angle_between_centers_edge = 30) : 
    (cos (dihedral_angle r1 r2 h_angle)) = 0.68 :=
sorry

end cosine_dihedral_angle_l192_192528


namespace tiles_needed_to_cover_room_l192_192113

open Real

-- Given the dimensions of the room
def room_length : ℝ := 10
def room_width : ℝ := 15

-- Given the tile dimensions converted to feet
def tile_length : ℝ := 1 / 4
def tile_width : ℝ := 5 / 12

-- Function to calculate the area of a rectangle
def area (length width : ℝ) : ℝ := length * width

theorem tiles_needed_to_cover_room :
  let room_area := area room_length room_width,
      tile_area := area tile_length tile_width in
  room_area / tile_area = 1440 :=
by
  sorry

end tiles_needed_to_cover_room_l192_192113


namespace double_summation_eval_l192_192659

theorem double_summation_eval :
  (∑ i in Finset.range 50, ∑ j in Finset.range 50, 2 * (i + 1 + j + 1)) = 255000 := 
by
  sorry

end double_summation_eval_l192_192659


namespace number_of_divisors_8_factorial_l192_192793

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192793


namespace remainder_is_four_l192_192063

theorem remainder_is_four (a b : ℤ) (p q r s : ℤ) :
  1108 + a ≡ b [MOD 23] ∧
  1453 ≡ b [MOD 23] ∧
  1844 + 2 * a ≡ b [MOD 23] ∧
  2281 ≡ b [MOD 23] →
  b = 4 :=
by
  intro h
  sorry

end remainder_is_four_l192_192063


namespace locus_of_moving_point_is_line_segment_l192_192026

-- Definitions based on the conditions
structure Point where
  x : ℝ
  y : ℝ

def F1 : Point := ⟨1, 0⟩
def F2 : Point := ⟨-1, 0⟩

def distance (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def on_line_segment (P : Point) (A B : Point) : Prop :=
  (distance A P) + (distance P B) = distance A B

-- The actual theorem statement
theorem locus_of_moving_point_is_line_segment (M : Point) :
  distance M F1 + distance M F2 = 2 → on_line_segment M F1 F2 :=
by
  sorry

end locus_of_moving_point_is_line_segment_l192_192026


namespace fraction_simplification_l192_192460

theorem fraction_simplification (x : ℚ) : 
  (3 / 4) * 60 - x * 60 + 63 = 12 → 
  x = (8 / 5) :=
by
  sorry

end fraction_simplification_l192_192460


namespace find_x9_y9_l192_192242

theorem find_x9_y9 (x y : ℝ) (h1 : x^3 + y^3 = 7) (h2 : x^6 + y^6 = 49) : x^9 + y^9 = 343 :=
by
  sorry

end find_x9_y9_l192_192242


namespace bc_approx_A_l192_192028

theorem bc_approx_A (A B C D E : ℝ) 
    (hA : 0 < A ∧ A < 1) (hB : 0 < B ∧ B < 1) (hC : 0 < C ∧ C < 1)
    (hD : 0 < D ∧ D < 1) (hE : 1 < E ∧ E < 2)
    (hA_val : A = 0.2) (hB_val : B = 0.4) (hC_val : C = 0.6) (hD_val : D = 0.8) :
    abs (B * C - A) < abs (B * C - B) ∧ abs (B * C - A) < abs (B * C - C) ∧ abs (B * C - A) < abs (B * C - D) := 
by 
  sorry

end bc_approx_A_l192_192028


namespace oven_clock_actual_time_l192_192368

theorem oven_clock_actual_time :
  ∀ (h : ℕ), (oven_time : h = 10) →
  (oven_gains : ℕ) = 8 →
  (initial_time : ℕ) = 18 →          
  (initial_wall_time : ℕ) = 18 →
  (wall_time_after_one_hour : ℕ) = 19 →
  (oven_time_after_one_hour : ℕ) = 19 + 8/60 →
  ℕ := sorry

end oven_clock_actual_time_l192_192368


namespace sum_of_longest_altitudes_l192_192962

-- Defines the sides of the triangle
def side_a : ℕ := 9
def side_b : ℕ := 12
def side_c : ℕ := 15

-- States it is a right triangle (by Pythagorean triple)
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the altitude lengths in a right triangle
def altitude_a (a b c : ℕ) (h : is_right_triangle a b c) : ℕ := a
def altitude_b (a b c : ℕ) (h : is_right_triangle a b c) : ℕ := b

-- Problem statement
theorem sum_of_longest_altitudes :
  ∃ (a b c : ℕ), is_right_triangle a b c ∧ a = side_a ∧ b = side_b ∧ c = side_c ∧
  altitude_a a b c sorry + altitude_b a b c sorry = 21 :=
by
  use side_a, side_b, side_c
  split
  sorry -- Proof that 9, 12, and 15 form a right triangle.
  split; refl
  split; refl
  sorry -- Proof that the sum of altitudes is 21.

end sum_of_longest_altitudes_l192_192962


namespace geometric_sequence_properties_l192_192724

/--
Given a sequence {a_n} that is a monotonically increasing geometric sequence
with the first term a_1 = 1, and the condition that (a_3, 5/3 * a_4, a_5) forms
an arithmetic sequence, we need to prove two things:
1. The general term formula for the sequence {a_n} is a_n = 3^(n-1).
2. The sum of the first n terms of the sequence {n * a_n} is 
   S_n = (1/4) * (3^n - 1) - (2n-1)/4 * 3^n.
-/
theorem geometric_sequence_properties (a : ℕ → ℕ) (n : ℕ) 
  (h1 : ∀ n, a n ≤ a (n + 1)) 
  (h2 : a 1 = 1) 
  (h3 : 2 * (5/3 * a 4) = a 3 + a 5) :
  (a n = 3^(n-1)) ∧ 
  (∑ i in finset.range n, i * a i = (1/4) * (3^n - 1) - (2n - 1)/4 * 3^n) :=
sorry

end geometric_sequence_properties_l192_192724


namespace participants_in_robbery_l192_192612

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l192_192612


namespace circle_center_l192_192193

open Real

theorem circle_center (a : ℝ) 
    (h_circle : ∀ x y : ℝ, a^2 * x^2 + (a + 2) * y^2 + 4 * x + 8 * y + 5 * a = 0) : 
    (a = -1) → center_of_circle h_circle = (-2, -4) :=
sorry

end circle_center_l192_192193


namespace pos_divisors_8_factorial_l192_192909

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192909


namespace positive_divisors_8_factorial_l192_192837

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192837


namespace rad_to_deg_eq_l192_192147

theorem rad_to_deg_eq : (4 / 3) * 180 = 240 := by
  sorry

end rad_to_deg_eq_l192_192147


namespace number_of_divisors_of_8_fact_l192_192897

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192897


namespace cube_edge_length_close_to_six_l192_192989

theorem cube_edge_length_close_to_six
  (a V S : ℝ)
  (h1 : V = a^3)
  (h2 : S = 6 * a^2)
  (h3 : V = S + 1) : abs (a - 6) < 1 :=
by
  sorry

end cube_edge_length_close_to_six_l192_192989


namespace second_year_interest_rate_l192_192126

noncomputable def investment_problem (initial_investment first_year_rate final_value : ℝ) : ℝ :=
  let first_year_value := initial_investment * (1 + first_year_rate / 100)
  in (final_value / first_year_value - 1) * 100
  
theorem second_year_interest_rate : investment_problem 15000 5 16380 = 4 :=
by
  sorry

end second_year_interest_rate_l192_192126


namespace problem_equivalent_l192_192647

-- Define the function f on ℝ satisfying the conditions
variable (f : ℝ → ℝ)

-- Define the conditions
def cond1 : Prop := ∀ x : ℝ, f(x) + (deriv f x) > 1
def cond2 : Prop := f(0) = 4

-- Define the inequality question we need to solve
def inequality_holds : Prop := ∀ x : ℝ, (e^x * f(x) > e^x + 3) ↔ (x > 0)

-- Main theorem statement combining all together
theorem problem_equivalent (cond1 : cond1 f) (cond2 : cond2 f) : inequality_holds f :=
by sorry

end problem_equivalent_l192_192647


namespace arc_length_parametric_curve_l192_192553

-- Define the parametric equations
def x(t : ℝ) : ℝ := 2 * (t - sin t)
def y(t : ℝ) : ℝ := 2 * (1 - cos t)

-- Define the bounds for t
def t1 : ℝ := 0
def t2 : ℝ := (π / 2)

-- Define the derivatives
def dxdt(t : ℝ) : ℝ := 2 * (1 - cos t)
def dydt(t : ℝ) : ℝ := 2 * sin t

-- Arc length function
def arcLength : ℝ :=
  ∫ t in t1..t2, real.sqrt (dxdt t ^ 2 + dydt t ^ 2)

-- The proof statement: Calculating the arc length
theorem arc_length_parametric_curve : arcLength = 8 - 4 * real.sqrt 2 :=
by
  sorry

end arc_length_parametric_curve_l192_192553


namespace hyperbola_asymptotes_l192_192574

theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) (e : ℝ) 
  (he : e = sqrt 13 / 2) 
  (h : (a * e)^2 = a^2 + b^2)
  : asymptotes_eq (C := hyperbola_form) a b = "y = ± (3/2) x" :=
sorry

end hyperbola_asymptotes_l192_192574


namespace problems_per_page_l192_192080

def total_problems : ℕ := 60
def finished_problems : ℕ := 20
def remaining_pages : ℕ := 5

theorem problems_per_page :
  (total_problems - finished_problems) / remaining_pages = 8 :=
by
  sorry

end problems_per_page_l192_192080


namespace single_digit_number_transformation_l192_192578

theorem single_digit_number_transformation (n : ℕ) (h : n = 222222222) :
  let op : ℕ → ℕ := λ n, let (a, b) := (n / 10, n % 10) in 4 * a + b in
  ∃! d : ℕ, d < 10 ∧ (∀ k, n = op k → k = d) ∧ d = 6 :=
by sorry

end single_digit_number_transformation_l192_192578


namespace num_possible_integer_values_l192_192299

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192299


namespace num_divisors_of_8_factorial_l192_192868

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192868


namespace sqrt_floor_8_integer_count_l192_192319

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192319


namespace joker_then_spade_probability_correct_l192_192107

-- Defining the conditions of the deck
def deck_size : ℕ := 60
def joker_count : ℕ := 4
def suit_count : ℕ := 4
def cards_per_suit : ℕ := 15

-- The probability of drawing a Joker first and then a spade
def prob_joker_then_spade : ℚ :=
  (joker_count * (cards_per_suit - 1) + (deck_size - joker_count) * cards_per_suit) /
  (deck_size * (deck_size - 1))

-- The expected probability according to the solution
def expected_prob : ℚ := 224 / 885

theorem joker_then_spade_probability_correct :
  prob_joker_then_spade = expected_prob :=
by
  -- Skipping the actual proof steps
  sorry

end joker_then_spade_probability_correct_l192_192107


namespace quadratic_real_roots_l192_192690

variable (a b : ℝ)

theorem quadratic_real_roots (h : ∀ a : ℝ, ∃ x : ℝ, x^2 - 2*a*x - a + 2*b = 0) : b ≤ -1/8 :=
by
  sorry

end quadratic_real_roots_l192_192690


namespace yellow_pairs_proof_l192_192134

variables {total_students blue_students yellow_students pairs blue_pairs yellow_pairs : ℕ}

-- Conditions
def total_students := 132
def blue_students := 63
def yellow_students := 69
def pairs := 66
def blue_pairs := 27
def yellow_student_pairs := pairs - blue_pairs - 9

theorem yellow_pairs_proof :
    2 * blue_pairs = 2 * 27 ∧
    blue_pairs = 27 ∧
    (blue_students - 2 * blue_pairs) + (yellow_students - 2 * yellow_student_pairs) = total_students - 2 * pairs →
    yellow_pairs = 30 :=
begin
  sorry
end

end yellow_pairs_proof_l192_192134


namespace probability_positive_product_is_correct_l192_192518

-- The set consists of the given elements
def my_set : Set ℤ := {-3, -2, -1, 0, 1, 3, 4}

-- Define the total number of ways to choose 3 elements
def total_ways : ℕ := Nat.choose 7 3

-- Define the function to check if the product is positive
def is_positive (l : List ℤ) : Prop := 
  l.product > 0

-- Define the favorable outcomes
def favorable_outcomes : ℕ := 1 + (Nat.choose 3 2) * (Nat.choose 3 1)

-- Probability calculation
def probability_positive_product : ℚ := favorable_outcomes / total_ways

-- The theorem to prove the desired probability
theorem probability_positive_product_is_correct :
  probability_positive_product = 2 / 7 := by
sorry

end probability_positive_product_is_correct_l192_192518


namespace sum_of_two_longest_altitudes_l192_192970

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def altitude (a b c : ℝ) (side : ℝ) : ℝ :=
  (2 * heron_area a b c) / side

theorem sum_of_two_longest_altitudes (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let ha := altitude a b c a
  let hb := altitude a b c b
  let hc := altitude a b c c
  ha + hb = 21 ∨ ha + hc = 21 ∨ hb + hc = 21 := by
  sorry

end sum_of_two_longest_altitudes_l192_192970


namespace sequences_from_confidence_l192_192251

-- setting up known conditions
def letters := ["C", "O", "N", "F", "I", "D", "E", "N", "C", "E"]
def total_letters := 10
def remove_letter (letter: String) (multiset: List String) : List String :=
  nat.pred <$> multiset  -- reducing available letters

def count_fixed_sequence (seq_length: Nat) :=
  let remaining := remove_letter "N" letters
  let exclude_E := remaining.filter (λ x => x ≠ "E")
  let differences := (remaining.length - 2) -- exclude 2 Es after removing one N
  let choices3 := (differences.choose 3) * factorial 3 -- ways to pick 3 mid letters
  let last_letter_choices := differences - 3 -- excluding 3 chosen, avoiding E
  choices3 * last_letter_choices

-- Theorem statement
theorem sequences_from_confidence :
  count_fixed_sequence 5 = 360 :=
  by sorry

end sequences_from_confidence_l192_192251


namespace value_of_fraction_l192_192348

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end value_of_fraction_l192_192348


namespace alcohol_percentage_after_mixing_l192_192086

namespace AlcoholMixture

def initial_volume : ℝ := 15
def initial_alcohol_percentage : ℝ := 0.25
def additional_water : ℝ := 3

def initial_alcohol_volume : ℝ := initial_volume * initial_alcohol_percentage
def new_total_volume : ℝ := initial_volume + additional_water
def new_alcohol_percentage : ℝ := (initial_alcohol_volume / new_total_volume) * 100

theorem alcohol_percentage_after_mixing :
  new_alcohol_percentage = 20.83 := by
  sorry

end AlcoholMixture

end alcohol_percentage_after_mixing_l192_192086


namespace binary_difference_l192_192644

theorem binary_difference (n : ℕ) (b_2 : List ℕ) (x y : ℕ) (h1 : n = 157)
  (h2 : b_2 = [1, 0, 0, 1, 1, 1, 0, 1])
  (hx : x = b_2.count 0)
  (hy : y = b_2.count 1) : y - x = 2 := by
  sorry

end binary_difference_l192_192644


namespace problem_z_value_l192_192431

theorem problem_z_value (k m : ℕ) (n : ℤ) 
  (h_k : 15^k ∣ 1031525) 
  (h_m : 20^m ∣ 1031525) 
  (h_n : 10^n ∣ 1031525) 
  (h_k_val : k = 2) 
  (h_m_val : m = 0) 
  (h_n_val : n = -1) :
  let x := 3^k - k^3,
      y := 5^m - m^5,
      z := (x + y) - 2^(n + 2)
  in z = 0 :=
by
  sorry

end problem_z_value_l192_192431


namespace part1_part2_l192_192204

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

/-- Given sequence properties -/
axiom h1 : a 1 = 5
axiom h2 : ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1) + 2^n - 1

/-- Part (I): Proving the sequence is arithmetic -/
theorem part1 (n : ℕ) : ∃ d, (∀ m ≥ 1, (a (m + 1) - 1) / 2^(m + 1) - (a m - 1) / 2^m = d)
∧ ((a 1 - 1) / 2 = 2) := sorry

/-- Part (II): Sum of the first n terms -/
theorem part2 (n : ℕ) : S n = n * 2^(n+1) := sorry

end part1_part2_l192_192204


namespace gcd_lcm_product_150_180_l192_192138

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcd_lcm_product_150_180 : (gcd 150 180) * (lcm 150 180) = 54000 := by
  -- Proof is omitted
  sorry

end gcd_lcm_product_150_180_l192_192138


namespace denom_asymptotes_sum_l192_192481

theorem denom_asymptotes_sum (A B C : ℤ)
  (h_denom : ∀ x, (x = -1 ∨ x = 3 ∨ x = 4) → x^3 + A * x^2 + B * x + C = 0) :
  A + B + C = 11 := 
sorry

end denom_asymptotes_sum_l192_192481


namespace incorrect_statements_l192_192068

-- Given the binomial random variable and transformation
def xi : ℕ → ℝ := λ n, (3 / 4) * n
def eta : ℕ → ℝ := λ n, 2 * xi n + 1

-- Given the average of sample data
def avg_x : ℝ := 2
def new_avg (x : ℕ → ℝ) (n : ℕ) := (3 * avg_x) + 2

theorem incorrect_statements :
  ¬(D (η := 2 * ξ + 1) = 9) ∧ ¬(avg_{x * n} = 6) := 
by
  sorry

end incorrect_statements_l192_192068


namespace limit_calc_l192_192630

noncomputable def limit_expr (x : ℝ) : ℝ := (1 - real.sqrt (real.cos x)) / (x * real.sin x)

theorem limit_calc : filter.tendsto limit_expr (nhds 0) (nhds (1 / 4)) :=
sorry

end limit_calc_l192_192630


namespace coeff_of_linear_term_l192_192017

def quadratic_eqn (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem coeff_of_linear_term :
  ∀ (x : ℝ), (quadratic_eqn x = 0) → (∃ c_b : ℝ, quadratic_eqn x = x^2 + c_b * x + 3 ∧ c_b = -2) :=
by
  sorry

end coeff_of_linear_term_l192_192017


namespace arrange_knights_l192_192624

-- Definitions based on conditions
def Knight (n : ℕ) : Type :=
  { i // i < 2 * n }

def enemies (n : ℕ) (a b : Knight n) : Prop :=
  sorry -- some way to define enemy relation

-- The theorem statement
theorem arrange_knights (n : ℕ) (knights : List (Knight n)) (enemy_rel : ∀ k : Knight n, (List.filter (enemies n k) knights).length ≤ n - 1) :
  ∃ permuted_knights : List (Knight n), ∀ i : ℕ, enemies n (permuted_knights.get i) (permuted_knights.get (i + 1) % (2 * n)) = false :=
sorry

end arrange_knights_l192_192624


namespace prob_divisors_8_fact_l192_192803

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192803


namespace pupils_correct_l192_192515

def totalPeople : ℕ := 676
def numberOfParents : ℕ := 22
def numberOfPupils : ℕ := totalPeople - numberOfParents

theorem pupils_correct :
  numberOfPupils = 654 := 
by
  sorry

end pupils_correct_l192_192515


namespace more_than_10_numbers_l192_192002

theorem more_than_10_numbers (seq : List ℕ) 
  (sum_eq_20 : seq.sum = 20) 
  (no_num_or_sum_eq_3 : ∀ n, n ∈ seq → n ≠ 3 ∧ 
    ∀ (start len : ℕ), start + len ≤ seq.length → (seq.slice start len).sum ≠ 3) :
  seq.length > 10 :=
  sorry

end more_than_10_numbers_l192_192002


namespace ordered_pairs_count_l192_192753

theorem ordered_pairs_count :
  (∃ (A B : ℕ), 0 < A ∧ 0 < B ∧ A % 2 = 0 ∧ B % 2 = 0 ∧ (A / 8) = (8 / B))
  → (∃ (n : ℕ), n = 5) :=
by {
  sorry
}

end ordered_pairs_count_l192_192753


namespace james_1500th_day_is_thursday_l192_192390

theorem james_1500th_day_is_thursday 
  (born_on_tuesday : ∀ n, n % 7 = 0 → n + 1 = 0 + 1): 
  (∀ k,  k = 1500 → (born_on_tuesday 1500 = 4 + 1)) :=
by 
  intros k hk
  rw hk
  sorry

end james_1500th_day_is_thursday_l192_192390


namespace range_of_h_l192_192176

-- Define the function h
def h (t : ℝ) : ℝ := (t^2 + (5 / 4) * t) / (t^2 + 1)

-- Define the statement that the range of h is the interval [ (4 - sqrt(41)) / 8, (4 + sqrt(41)) / 8 ]
theorem range_of_h :
  set.range h = set.Icc ((4 - Real.sqrt 41) / 8) ((4 + Real.sqrt 41) / 8) := sorry

end range_of_h_l192_192176


namespace marble_count_l192_192997

variable (r b g : ℝ)

-- Conditions
def condition1 : b = r / 1.3 := sorry
def condition2 : g = 1.5 * r := sorry

-- Theorem statement
theorem marble_count (h1 : b = r / 1.3) (h2 : g = 1.5 * r) :
  r + b + g = 3.27 * r :=
by sorry

end marble_count_l192_192997


namespace angular_regions_bounds_l192_192638

-- Given 1996 lines on the plane such that any two of them intersect, but no three are concurrent
-- Define the minimum and maximum number of angular regions formed by these lines

theorem angular_regions_bounds (n : ℕ) (h : n = 1996) :
  3 ≤ number_of_angular_regions n ∧ number_of_angular_regions n ≤ 1995 := 
sorry

end angular_regions_bounds_l192_192638


namespace find_smallest_m_l192_192701

noncomputable def S (n : ℕ) : ℝ := 1/2 * (1 + 1/2 - 1/(n+1) - 1/(n+2))

def is_solution (m : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → S n < 1/4 * m^2 - 1/2 * m

theorem find_smallest_m :
  ∃ m : ℕ, m > 0 ∧ is_solution m ∧ ∀ m' : ℕ, m' > 0 ∧ is_solution m' → m ≤ m' :=
begin
  use 3,
  split,
  { norm_num },
  split,
  { intros n hn,
    sorry },
  { intros m' hm',
    sorry }
end

end find_smallest_m_l192_192701


namespace count_integer_values_l192_192307

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192307


namespace greatest_value_expression_l192_192065

variable (x y z : ℕ)

def expr1 := 4 * x^2 - 3 * y + 2 * z
def expr2 := 6 * x - 2 * y^3 + 3 * z^2
def expr3 := 2 * x^3 - y^2 * z
def expr4 := x * y^3 - z^2

theorem greatest_value_expression : 
  x = 3 → 
  y = 2 → 
  z = 1 → 
  expr3 x y z = 50 ∧
  expr3 x y z > expr1 x y z ∧
  expr3 x y z > expr2 x y z ∧
  expr3 x y z > expr4 x y z :=
by
  intros hx hy hz
  have h_expr1 : expr1 3 2 1 = 32 := by norm_num
  have h_expr2 : expr2 3 2 1 = 5 := by norm_num
  have h_expr3 : expr3 3 2 1 = 50 := by norm_num
  have h_expr4 : expr4 3 2 1 = 23 := by norm_num
  rw [hx, hy, hz]
  exact ⟨h_expr3, by linarith [h_expr1, h_expr2, h_expr4]⟩

end greatest_value_expression_l192_192065


namespace positive_divisors_8_factorial_l192_192836

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192836


namespace num_possible_integer_values_l192_192301

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192301


namespace sum_consecutive_numbers_last_digit_diff_l192_192635

theorem sum_consecutive_numbers_last_digit_diff (a : ℕ) : 
    (2015 * (a + 1007) % 10) ≠ (2019 * (a + 3024) % 10) := 
by 
  sorry

end sum_consecutive_numbers_last_digit_diff_l192_192635


namespace value_of_c_l192_192056

theorem value_of_c (c : ℝ) : (∀ x : ℝ, x * (4 * x + 1) < c ↔ x > -5 / 2 ∧ x < 3) → c = 27 :=
by
  intros h
  sorry

end value_of_c_l192_192056


namespace system_of_equations_soln_l192_192679

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l192_192679


namespace semicircle_perimeter_approx_l192_192587

def radius : ℝ := 9
def pi_approx : ℝ := 3.14159

theorem semicircle_perimeter_approx :
  let curved_part := pi_approx * radius in
  let diameter := 2 * radius in
  let perimeter := curved_part + diameter in
  abs(perimeter - 46.27) < 0.01 :=
by
  let curved_part := pi_approx * radius
  let diameter := 2 * radius
  let perimeter := curved_part + diameter
  sorry

end semicircle_perimeter_approx_l192_192587


namespace num_divisors_8_factorial_l192_192760

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192760


namespace forty_mn_equals_PQ_l192_192466

theorem forty_mn_equals_PQ (m n : ℤ) (P Q : ℤ) (hP : P = 2^m) (hQ : Q = 5^n) : 40^(m*n) = P^(3*n) * Q^m :=
by
  sorry

end forty_mn_equals_PQ_l192_192466


namespace product_of_all_snug_integers_congruent_to_one_l192_192426

def is_snug (n k : ℕ) : Prop :=
  1 ≤ k ∧ k < n ∧ Nat.gcd k n = Nat.gcd (k+1) n

theorem product_of_all_snug_integers_congruent_to_one (n : ℕ) (h : n > 3) :
  (∏ k in Finset.filter (is_snug n) (Finset.range n), k) % n = 1 :=
sorry

end product_of_all_snug_integers_congruent_to_one_l192_192426


namespace geometric_progression_first_term_proof_l192_192201

noncomputable def geometric_progression_first_term 
  (a_1 q : ℚ) (S : ℕ → ℚ) : ℚ :=
if q^4 = 16 then
  if q = 2 then if S 4 = 1 ∧ S 8 = 17 then a_1 = 1/15 else a_1
  else if q = -2 then if S 4 = 1 ∧ S 8 = 17 then a_1 = -1/5 else a_1
  else a_1
else a_1

theorem geometric_progression_first_term_proof 
  (S : ℕ → ℚ) 
  (h1 : S 4 = 1) 
  (h2 : S 8 = 17) 
  (a_1 : ℚ) :
  (∃ a_1, geometric_progression_first_term a_1 2 S = 1/15 ∨ geometric_progression_first_term a_1 (-2) S = -1/5) :=
sorry

end geometric_progression_first_term_proof_l192_192201


namespace eq_30_apples_n_7_babies_min_3_max_6_l192_192509

theorem eq_30_apples_n_7_babies_min_3_max_6 (x : ℕ) 
    (h1 : 30 = x + 7 * 4)
    (h2 : 21 ≤ 30) 
    (h3 : 30 ≤ 42) 
    (h4 : x = 2) :
  x = 2 :=
by
  sorry

end eq_30_apples_n_7_babies_min_3_max_6_l192_192509


namespace num_divisors_of_8_factorial_l192_192867

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192867


namespace cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l192_192651

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

end cylinder_is_defined_sphere_is_defined_hyperbolic_cylinder_is_defined_parabolic_cylinder_is_defined_l192_192651


namespace remainder_n_plus_1008_l192_192057

variable (n : ℕ)

theorem remainder_n_plus_1008 (h1 : n % 4 = 1) (h2 : n % 5 = 3) : (n + 1008) % 4 = 1 := by
  sorry

end remainder_n_plus_1008_l192_192057


namespace ratio_of_speeds_l192_192517

variables (v_A v_B v_C : ℝ)

-- Conditions definitions
def condition1 : Prop := v_A - v_B = 5
def condition2 : Prop := v_A + v_C = 15

-- Theorem statement (the mathematically equivalent proof problem)
theorem ratio_of_speeds (h1 : condition1 v_A v_B) (h2 : condition2 v_A v_C) : (v_A / v_B) = 3 :=
sorry

end ratio_of_speeds_l192_192517


namespace robbery_participants_l192_192609

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l192_192609


namespace find_m_for_eccentricity_l192_192207

-- Define the problem statement in Lean
theorem find_m_for_eccentricity :
  ∃ m : ℝ, (∀ x y : ℝ, m * x^2 + y^2 = 1) ∧ (∀ a b : ℝ, (a > b) ∧ (∃ e : ℝ, e = 1/2)) :=
begin
  use 3/4,
  sorry
end

end find_m_for_eccentricity_l192_192207


namespace integer_values_of_x_l192_192296

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192296


namespace range_of_x_in_sqrt_x_minus_2_l192_192494

theorem range_of_x_in_sqrt_x_minus_2 :
  {x : ℝ // x ≥ 2} = {x : ℝ // ∃ y : ℝ, y = sqrt (x - 2)} :=
by
  sorry

end range_of_x_in_sqrt_x_minus_2_l192_192494


namespace num_positive_divisors_8_factorial_l192_192945

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192945


namespace molecular_weight_of_compound_l192_192052

theorem molecular_weight_of_compound :
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  molecular_weight = 156.22615 :=
by
  -- conditions
  let atomic_weight_ca := 40.08
  let atomic_weight_o := 16.00
  let atomic_weight_h := 1.008
  let atomic_weight_n := 14.01
  let atomic_weight_c12 := 12.00
  let atomic_weight_c13 := 13.003
  let average_atomic_weight_c := (0.95 * atomic_weight_c12) + (0.05 * atomic_weight_c13)
  let molecular_weight := 
    (2 * atomic_weight_ca) +
    (3 * atomic_weight_o) +
    (2 * atomic_weight_h) +
    (1 * atomic_weight_n) +
    (1 * average_atomic_weight_c)
  -- prove statement
  have h1 : average_atomic_weight_c = 12.05015 := by sorry
  have h2 : molecular_weight = 156.22615 := by sorry
  exact h2

end molecular_weight_of_compound_l192_192052


namespace solve_problem_l192_192663

noncomputable def problem_statement : Prop :=
  ( ( (2 : ℚ) / (3 : ℚ) )^6 * ( (5 : ℚ) / (6 : ℚ) )^(-4) = ( 82944 : ℚ ) / ( 456375 : ℚ )

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l192_192663


namespace smallest_m_for_probability_condition_l192_192458

noncomputable def smallest_m (P : ℕ → Prop) : ℕ :=
if h : ∃ m, P m then Nat.find h else 0

def satisfies_condition (m : ℕ) : Prop :=
((m - 4)^3 > (m^3 / 2 : ℝ))

theorem smallest_m_for_probability_condition :
  smallest_m satisfies_condition = 15 :=
sorry

end smallest_m_for_probability_condition_l192_192458


namespace jack_walked_distance_proof_l192_192389

-- Define conditions as given in the problem.
def walking_rate := 4.8  -- miles per hour
def total_time := 1.25   -- hours

-- Define the claim which needs to be proved.
def distance_jack_walked : ℕ := 6

-- The theorem statement including the conditions.
theorem jack_walked_distance_proof (rate: ℝ) (time: ℝ) (distance: ℕ) 
  (h_rate: rate = walking_rate) (h_time: time = total_time) 
  (h_distance: distance = distance_jack_walked) :
  rate * time = distance := sorry

end jack_walked_distance_proof_l192_192389


namespace num_possible_integer_values_l192_192305

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192305


namespace relationship_xyz_l192_192191

theorem relationship_xyz (a b : ℝ) (x y z : ℝ) (ha : a > 0) (hb : b > 0) 
  (hab : a > b) (hab_sum : a + b = 1) 
  (hx : x = Real.log b / Real.log a)
  (hy : y = Real.log (1 / b) / Real.log a)
  (hz : z = Real.log 3 / Real.log ((1 / a) + (1 / b))) : 
  y < z ∧ z < x := 
sorry

end relationship_xyz_l192_192191


namespace part1_solution_part2_solution_l192_192597

section IsoscelesTriangle

variable (base_leg_ratio : ℝ := 2) -- The ratio of the length of each leg to the base.

def is_isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (c = a ∧ c ≠ b)

def can_form_triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def part1_triangle_sides (total_length : ℝ) (ratio : ℝ) : (ℝ × ℝ × ℝ) :=
  let x := total_length / (2 * ratio + 1)
  let leg := ratio * x
  (x, leg, leg)

theorem part1_solution :
  part1_triangle_sides 18 base_leg_ratio = (18 / 5, 36 / 5, 36 / 5) := 
begin
  sorry
end

noncomputable def part2_triangle_sides (total_length : ℝ) (fixed_side : ℝ) : Prop :=
  ∃ (a b c : ℝ), is_isosceles_triangle a b c ∧ a + b + c = total_length ∧ (a = fixed_side ∨ b = fixed_side ∨ c = fixed_side) ∧ can_form_triangle a b c

theorem part2_solution :
  part2_triangle_sides 18 4 = ∃ a b c, a = 7 ∧ b = 7 ∧ c = 4 :=
begin
  sorry
end

end IsoscelesTriangle

end part1_solution_part2_solution_l192_192597


namespace pos_divisors_8_factorial_l192_192901

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192901


namespace mass_percentage_ca_in_compound_l192_192051

noncomputable def mass_percentage_ca_in_cac03 : ℝ :=
  let mm_ca := 40.08
  let mm_c := 12.01
  let mm_o := 16.00
  let mm_caco3 := mm_ca + mm_c + 3 * mm_o
  (mm_ca / mm_caco3) * 100

theorem mass_percentage_ca_in_compound (mp : ℝ) (h : mp = mass_percentage_ca_in_cac03) : mp = 40.04 := by
  sorry

end mass_percentage_ca_in_compound_l192_192051


namespace sum_of_two_longest_altitudes_l192_192969

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def altitude (a b c : ℝ) (side : ℝ) : ℝ :=
  (2 * heron_area a b c) / side

theorem sum_of_two_longest_altitudes (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let ha := altitude a b c a
  let hb := altitude a b c b
  let hc := altitude a b c c
  ha + hb = 21 ∨ ha + hc = 21 ∨ hb + hc = 21 := by
  sorry

end sum_of_two_longest_altitudes_l192_192969


namespace arrangements_where_A_and_B_are_not_next_to_each_other_l192_192623

-- Definitions for products
variables (A B C D E : Type)

-- Definition for the arrangement of products
def number_of_arrangements (AB_not_next_to_each_other : Prop) : Nat :=
  if AB_not_next_to_each_other then 72 else 0

-- We state our theorem which needs no computational proofs
theorem arrangements_where_A_and_B_are_not_next_to_each_other :
  ∀ (A B C D E : Type), number_of_arrangements (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ A) = 72 :=
by simp [number_of_arrangements]; sorry

end arrangements_where_A_and_B_are_not_next_to_each_other_l192_192623


namespace sum_of_longest_altitudes_l192_192967

theorem sum_of_longest_altitudes (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  let h1 := a,
      h2 := b,
      h := (a * b) / c in
  h1 + h2 = 21 := by
{
  sorry
}

end sum_of_longest_altitudes_l192_192967


namespace Kates_hair_length_l192_192402

theorem Kates_hair_length (L E K : ℕ) (h1 : K = E / 2) (h2 : E = L + 6) (h3 : L = 20) : K = 13 :=
by
  sorry

end Kates_hair_length_l192_192402


namespace johns_weekly_allowance_l192_192075

theorem johns_weekly_allowance
    (A : ℝ)
    (h1 : ∃ A, (4/15) * A = 0.64) :
    A = 2.40 :=
by
  sorry

end johns_weekly_allowance_l192_192075


namespace number_of_divisors_of_8_fact_l192_192894

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192894


namespace subtract_decimal_nearest_hundredth_l192_192006

theorem subtract_decimal_nearest_hundredth :
  ∀ (a b : ℝ), a = 888.88 → b = 444.44 → a - b = 444.44 := by
  intros a b ha hb
  rw [ha, hb]
  norm_num
  -- The proof can be completed here or marked as sorry if only the statement is required.
  sorry

end subtract_decimal_nearest_hundredth_l192_192006


namespace william_initial_marbles_l192_192069

theorem william_initial_marbles (shared_with_theresa : ℕ) (left_with_william : ℕ) (h_shared : shared_with_theresa = 3) (h_left : left_with_william = 7) :
  shared_with_theresa + left_with_william = 10 :=
by
  rw [h_shared, h_left]
  rfl

end william_initial_marbles_l192_192069


namespace length_of_AC_l192_192985

noncomputable def sqrt5 : ℝ := Real.sqrt 5
noncomputable def golden_ratio_reciprocal : ℝ := (sqrt5 - 1) / 2

def AB : ℝ := 100
def BC : ℝ := golden_ratio_reciprocal * AB
def AC : ℝ := AB - BC

theorem length_of_AC : AC = 61.8 :=
by
  -- Proof omitted
  sorry

end length_of_AC_l192_192985


namespace problem_statement_l192_192353

variables {a b x : ℝ}

theorem problem_statement (h1 : x = a / b + 2) (h2 : a ≠ b) (h3 : b ≠ 0) : 
  (a + 2 * b) / (a - 2 * b) = x / (x - 4) := 
sorry

end problem_statement_l192_192353


namespace pos_divisors_8_factorial_l192_192905

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192905


namespace olivia_spent_89_l192_192041

-- Define initial and subsequent amounts
def initial_amount : ℕ := 100
def atm_amount : ℕ := 148
def after_supermarket : ℕ := 159

-- Total amount before supermarket
def total_before_supermarket : ℕ := initial_amount + atm_amount

-- Amount spent
def amount_spent : ℕ := total_before_supermarket - after_supermarket

-- Proof that Olivia spent 89 dollars
theorem olivia_spent_89 : amount_spent = 89 := sorry

end olivia_spent_89_l192_192041


namespace positive_divisors_of_8_factorial_l192_192833

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192833


namespace prob_divisors_8_fact_l192_192808

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192808


namespace group_size_systematic_sampling_l192_192628

-- Define the total number of viewers
def total_viewers : ℕ := 10000

-- Define the number of viewers to be selected
def selected_viewers : ℕ := 10

-- Lean statement to prove the group size for systematic sampling
theorem group_size_systematic_sampling (n_total n_selected : ℕ) : n_total = total_viewers → n_selected = selected_viewers → (n_total / n_selected) = 1000 :=
by
  intros h_total h_selected
  rw [h_total, h_selected]
  sorry

end group_size_systematic_sampling_l192_192628


namespace locus_of_Q_is_curve_E_line_passing_focus_F_l192_192709

-- Definitions and conditions
def parabola_C (x y : ℝ) : Prop := y^2 = 4 * x

def is_focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Locus equation proof
theorem locus_of_Q_is_curve_E :
  ∀ (F : ℝ × ℝ) (P Q : ℝ × ℝ), is_focus F →
  P.2^2 = 4 * P.1 → Q.1 = 3 * P.1 / 2 + 1 / 2 ∧ Q.2 = P.2 / 3 →
  9 * Q.2^2 = 12 * Q.1 - 8 :=
by sorry

-- Line equation proof
theorem line_passing_focus_F :
  ∀ (F : ℝ × ℝ) (m : ℝ) (x y : ℝ), is_focus F →
  x = m * y + 1 → 9 * y^2 - 12 * m * y - 4 = 0 →
  (y = (2 * (3 * m + 1) / 3 : ℝ) ∨ y = (2 * (3 * m - 1) / 3 : ℝ)) →
  |((2 * (3 * m + 1) / 3 : ℝ) - (2 * (3 * m - 1) / 3 : ℝ))| = 4 →
  (x = sqrt 2 * y + 1 ∨ x = - sqrt 2 * y + 1) :=
by sorry

end locus_of_Q_is_curve_E_line_passing_focus_F_l192_192709


namespace bank_robbery_participants_l192_192604

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l192_192604


namespace pentagon_triangle_ratio_l192_192582

theorem pentagon_triangle_ratio (p t s : ℝ) 
  (h₁ : 5 * p = 30) 
  (h₂ : 3 * t = 30)
  (h₃ : 4 * s = 30) : 
  p / t = 3 / 5 := by
  sorry

end pentagon_triangle_ratio_l192_192582


namespace num_divisors_8_factorial_l192_192764

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192764


namespace num_pos_divisors_fact8_l192_192770

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192770


namespace a_is_geometric_b_general_formula_T_n_correct_l192_192498

-- Define sequence a_n
def a (n : ℕ) : ℝ := sorry

-- Conditions related to the sequence {a_n}
axiom a_1 : a 1 = 1
axiom S_n (n : ℕ) (t : ℝ) (hn : n ≥ 2) : 2 * t * (∑ i in finset.range (n + 1), a i) - (2 * t + 1) * (∑ i in finset.range n, a i) = 2 * t

-- Common ratio of the sequence {a_n}
def common_ratio_a (t : ℝ) := 1 + 1 / (2 * t)

-- Define sequence b_n
def b (n : ℕ) : ℝ := if n = 1 then 1 else (common_ratio_a (1 / (b (n - 1) + 2))) - 2

-- General formula for b_n
axiom b_general (n : ℕ) : b n = (1 / 2)^(n - 1)

-- Define sequence c_n
def c (n : ℕ) : ℝ := n * b n

-- Sum of the first n terms of {c_n}
def T (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), c i

-- Target sum formula for T_n
axiom T_n_formula (n : ℕ) : T n = 4 - (2 + n) / 2^(n - 1)

-- Mathematical properties to be proven as Lean statements:

-- {a_n} is geometric sequence:
theorem a_is_geometric (n : ℕ) (t : ℝ) (hn : n ≥ 2) : a (n + 1) / a n = 1 + 1 / (2 * t) := sorry

-- General formula for b_n
theorem b_general_formula (n : ℕ) : b n = (1 / 2)^(n - 1) := sorry

-- Sum formula for T_n
theorem T_n_correct (n : ℕ) : T n = 4 - (2 + n) / 2^(n - 1) := sorry

end a_is_geometric_b_general_formula_T_n_correct_l192_192498


namespace smallest_number_of_eggs_l192_192070

theorem smallest_number_of_eggs (e : ℕ) (d : ℕ) (h1 : e > 200) 
                                 (h2 : e = 15 * d - 3) : 
  e = 207 :=
by {
  have h3 : d > 13, from sorry, -- Derived from solving 15d - 3 > 200 and d being an integer
  have h4 : d = 14, from sorry, -- Since integer d > 13, the smallest such d is 14
  rw [h4, ←h2],
  norm_num,
  exact h1.trans (by norm_num),
}

end smallest_number_of_eggs_l192_192070


namespace find_missing_values_l192_192163

-- Conditions: Define the equations and the given solution
def system_of_equations (x y : ℝ) (⊗ ⊙ : ℝ) : Prop :=
  x + 2 * y = ⊗ ∧ x - 2 * y = 2 ∧ x = 4 ∧ y = ⊙

-- Proof statement: Prove the specific values for ⊗ and ⊙
theorem find_missing_values : ∃ (⊗ ⊙ : ℝ), ⊗ = 6 ∧ ⊙ = 1 :=
  by
    use [6, 1]
    sorry

end find_missing_values_l192_192163


namespace alcohol_concentration_after_additions_l192_192470

theorem alcohol_concentration_after_additions
  (initial_volume : ℚ) (initial_concentration : ℚ)
  (addition1 : ℚ) (addition2 : ℚ) (addition3 : ℚ) :
  initial_volume = 15 → initial_concentration = 0.20 →
  addition1 = 5 → addition2 = 8 → addition3 = 12 →
  let total_volume1 := initial_volume + addition1,
      concentration1 := 3 / total_volume1,
      total_volume2 := total_volume1 + addition2,
      concentration2 := 3 / total_volume2,
      total_volume3 := total_volume2 + addition3,
      concentration3 := 3 / total_volume3 in
  concentration1 = 0.15 ∧ concentration2 ≈ 0.1071 ∧ concentration3 = 0.075 :=
by
  sorry

end alcohol_concentration_after_additions_l192_192470


namespace number_of_divisors_of_8_fact_l192_192890

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192890


namespace number_of_divisors_of_8_fact_l192_192882

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192882


namespace num_divisors_8_factorial_l192_192765

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192765


namespace correct_option_l192_192541

variable (a b : ℤ)

-- Propositions for each operation
def prop_A : Prop := a^2 * a = a^3
def prop_B : Prop := a^6 / a^2 = a^3
def prop_C : Prop := 3 * a - a = 3
def prop_D : Prop := (a - b)^2 = a^2 - b^2

-- Main theorem to verify the correctness of Option A
theorem correct_option : prop_A ∧ ¬prop_B ∧ ¬prop_C ∧ ¬prop_D :=
by
  sorry

end correct_option_l192_192541


namespace trajectory_of_point_l192_192360

theorem trajectory_of_point (x y : ℝ)
  (h1 : (x - 1)^2 + (y - 1)^2 = ((3 * x + y - 4)^2) / 10) :
  x - 3 * y + 2 = 0 :=
sorry

end trajectory_of_point_l192_192360


namespace functional_sum_l192_192978

theorem functional_sum :
  (∃ f : ℕ → ℝ, (∀ a b : ℕ, f (a + b) = f a * f b) ∧ f 1 = 1) →
  (∑ k in finset.range (2005 - 1 + 1), (λ k, f (k + 1) / f k) = 2004) :=
by {
  sorry
}

end functional_sum_l192_192978


namespace solution_set_is_circle_with_exclusion_l192_192677

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l192_192677


namespace robbery_participants_l192_192608

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l192_192608


namespace pos_divisors_8_factorial_l192_192903

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192903


namespace symmetric_circle_line_l192_192474

theorem symmetric_circle_line (a b : ℝ) (h1 : (x^2 + y^2 + 2x + 6y + 5a = 0) → true) 
(h2 : (y = x + 2b) → true) (h3 : (-3 = -1 + 2b)) (h4 : a < 2) : 
b - a > -3 :=
sorry

end symmetric_circle_line_l192_192474


namespace impossible_end_with_2048_on_board_l192_192484

theorem impossible_end_with_2048_on_board :
  let S := (Finset.range 2022).sum
  ∀ l : List ℤ, (∀ n, n ∈ l → n ∈ (Finset.range 2022)) →
  l.sum = S →
  (∀ x y, x ∈ l → y ∈ l → l ≠ [2048]) :=
by
  let S := (Finset.range 2022).sum
  intro l hmem hsum x y hx hy hne
  sorry

end impossible_end_with_2048_on_board_l192_192484


namespace number_of_divisors_8_factorial_l192_192792

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192792


namespace lcm_gcd_eq_product_l192_192462

theorem lcm_gcd_eq_product {a b : ℕ} (h : Nat.lcm a b + Nat.gcd a b = a * b) : a = 2 ∧ b = 2 :=
  sorry

end lcm_gcd_eq_product_l192_192462


namespace sqrt_floor_8_integer_count_l192_192318

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192318


namespace find_three_digit_numbers_l192_192669

theorem find_three_digit_numbers :
  ∃ A, (100 ≤ A ∧ A ≤ 999) ∧ (A^2 % 1000 = A) ↔ (A = 376) ∨ (A = 625) :=
by
  sorry

end find_three_digit_numbers_l192_192669


namespace largest_n_for_triangle_property_l192_192142

-- Definitions and assumptions from the condition of the problem
def consecutive_set (n : ℕ) : Set ℕ := {k | 3 ≤ k ∧ k ≤ n}

def triangle_property (s : Set ℕ) : Prop :=
  ∀ {a b c : ℕ}, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c →
    a + b > c ∧ a + c > b ∧ b + c > a

-- The statement of the problem respecting the given solution steps and answer
theorem largest_n_for_triangle_property :
  ∃ n : ℕ, (∀ s : Set ℕ, (s ⊆ consecutive_set n) ∧ s.card = 10 → triangle_property s) ∧
          (¬ (∀ s : Set ℕ, (s ⊆ consecutive_set (n + 1)) ∧ s.card = 10 → triangle_property s)) :=
  ⟨198, sorry, sorry⟩

end largest_n_for_triangle_property_l192_192142


namespace kate_hair_length_l192_192405

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end kate_hair_length_l192_192405


namespace prob_divisors_8_fact_l192_192811

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192811


namespace johns_profit_l192_192398

noncomputable def profit_made 
  (trees_chopped : ℕ)
  (planks_per_tree : ℕ)
  (planks_per_table : ℕ)
  (price_per_table : ℕ)
  (labor_cost : ℕ) : ℕ :=
(trees_chopped * planks_per_tree / planks_per_table) * price_per_table - labor_cost

theorem johns_profit : profit_made 30 25 15 300 3000 = 12000 :=
by sorry

end johns_profit_l192_192398


namespace integral_ln_10_l192_192382

def binomial_expansion_coefficient : ℕ := 5.choose 3

theorem integral_ln_10 (a : ℝ) (h : a = binomial_expansion_coefficient) : 
  ∫ x in 1..a, x⁻¹ = Real.log 10 :=
by {
  rw [h],
  simp [Set.Icc, measure_theory.integral_set_integral_real],
  exact @measure_theory.interval_integral.integral_const_inv (measure_theory.integrable_on_inv_Icc (by norm_num)),
}

end integral_ln_10_l192_192382


namespace strictly_increasing_intervals_l192_192174

theorem strictly_increasing_intervals (k : ℤ) :
  ∀ x, (k * π - 3 * π / 4 < x ∧ x < k * π + π / 4) → 
        strict_mono (λ x : ℝ, Real.tan(x + π / 4)) :=
sorry

end strictly_increasing_intervals_l192_192174


namespace last_digit_of_exponents_l192_192485

theorem last_digit_of_exponents : 
  (∃k, 2011 = 4 * k + 3 ∧ 
         (2^2011 % 10 = 8) ∧ 
         (3^2011 % 10 = 7)) → 
  ((2^2011 + 3^2011) % 10 = 5) := 
by 
  sorry

end last_digit_of_exponents_l192_192485


namespace problem1_problem2_min_value_l192_192559

theorem problem1 (x : ℝ) : |x + 1| + |x - 2| ≥ 3 := sorry

theorem problem2 (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  x^2 + y^2 + z^2 ≥ 1 / 14 := sorry

theorem min_value (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) :
  ∃ x y z, x^2 + y^2 + z^2 = 1 / 14 := sorry

end problem1_problem2_min_value_l192_192559


namespace toothbrushes_difference_l192_192654

theorem toothbrushes_difference
  (total : ℕ)
  (jan : ℕ)
  (feb : ℕ)
  (mar : ℕ)
  (apr_may_sum : total = jan + feb + mar + 164)
  (apr_may_half : 164 / 2 = 82)
  (busy_month_given : feb = 67)
  (slow_month_given : mar = 46) :
  feb - mar = 21 :=
by
  sorry

end toothbrushes_difference_l192_192654


namespace hyperbola_focal_length_l192_192717

theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) (h : C : hyperbola ℝ → Prop)  
  (asymptote : ∀ x y: ℝ, √2 * x + m * y = 0 → C x y) : 
  (2 * √(m + 1)) = 2 * √3 :=
by
  sorry

end hyperbola_focal_length_l192_192717


namespace integer_values_of_x_l192_192294

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192294


namespace area_of_set_S_l192_192410

-- Definitions based on the conditions
def Point := (ℝ × ℝ)
def LineSegment (A B : Point) : Set Point := { P : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))}

-- Main theorem statement
theorem area_of_set_S : 
  let A := (0, 0) in
  let B := (2, 0) in
  let S := { P : Point | ∃ X : Point, X ∈ LineSegment A B ∧ (dist A X) = 2 * (dist P X) } in 
  ∃ area : ℝ, area = sqrt 3 + (π / 6) :=
sorry

end area_of_set_S_l192_192410


namespace total_marks_prove_total_marks_l192_192077

def average_marks : ℝ := 40
def number_of_candidates : ℕ := 50

theorem total_marks (average_marks : ℝ) (number_of_candidates : ℕ) : Real :=
  average_marks * number_of_candidates

theorem prove_total_marks : total_marks average_marks number_of_candidates = 2000 := 
by
  sorry

end total_marks_prove_total_marks_l192_192077


namespace number_of_divisors_of_8_fact_l192_192895

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192895


namespace number_of_integer_values_l192_192272

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192272


namespace concylic_QO1O2O3_l192_192043

-- Definitions for circles and concurrency
noncomputable def Circle (center : Point) (radius : ℝ) : Set Point := sorry

-- Assume the existence of three circles C1, C2, C3 sharing a common point Q
variables (Q O1 O2 O3 A B C : Point)
variables (C1 : Circle O1) (C2 : Circle O2) (C3 : Circle O3)

-- Assume the circles meet again pairwise at points A, B, and C
axiom meet_pairwise : Set.Point =
  (C1 ∩ C2 = {Q, A}) ∧ (C2 ∩ C3 = {Q, B}) ∧ (C3 ∩ C1 = {Q, C})

-- Collinearity of A, B, C
axiom collinear_ABC : collinear {A, B, C}

-- Goal: Prove Q, O1, O2, O3 are concyclic
theorem concylic_QO1O2O3 :
  ∃ (circle : Circle) (radius : ℝ), {Q, O1, O2, O3} ⊆ circle :=
sorry

end concylic_QO1O2O3_l192_192043


namespace fraction_of_grid_covered_by_triangle_l192_192599

-- Define the vertices of the triangle
structure Point where
  x : ℕ
  y : ℕ

def A : Point := ⟨2, 2⟩
def B : Point := ⟨5, 5⟩
def C : Point := ⟨3, 6⟩

-- Define the grid dimensions
def gridWidth : ℕ := 8
def gridHeight : ℕ := 7

-- Lean statement to prove the fraction of the grid covered by the triangle
theorem fraction_of_grid_covered_by_triangle :
  let area_of_triangle := 1 / 2 * |2 * (5 - 6) + 5 * (6 - 2) + 3 * (2 - 5)|
  let area_of_grid := gridWidth * gridHeight
  (area_of_triangle / area_of_grid) = 9 / 112 :=
by
  sorry

end fraction_of_grid_covered_by_triangle_l192_192599


namespace num_divisors_8_factorial_l192_192767

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192767


namespace locus_of_P_l192_192237

-- Define the triangle vertices as points in the plane
variables {P : Type} [metric_space P]

-- Define points A, B, C in the plane
variables {A B C : P}

-- Define a function to calculate the area of a triangle given three points
noncomputable def triangle_area (A B C : P) : ℝ := 
  -- Suppose we have some function for calculating the area
  sorry

-- Define the lines e_1 and e_2 through B such that they satisfy the given ratio conditions
def on_line_e1 (P : P) : Prop :=
  ∃ M1 : P, M1 ∈ line_through A C ∧ (dist A M1) = 2 * (dist M1 C) ∧ P ∈ line_through B M1

def on_line_e2 (P : P) : Prop :=
  ∃ M2 : P, M2 ∈ line_through A C ∧ (dist C M2) = 2 * (dist M2 A) ∧ P ∈ line_through B M2

-- The statement of the problem
theorem locus_of_P (P : P) :
  triangle_area A B P = 2 * triangle_area B C P ↔ on_line_e1 P ∨ on_line_e2 P :=
begin
  sorry
end

end locus_of_P_l192_192237


namespace number_of_divisors_8_factorial_l192_192786

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192786


namespace pineapple_more_expensive_by_3_3_yuan_l192_192545

-- Definitions based on conditions
def watermelon_kg := 5 : ℝ
def watermelon_total_yuan := 9.5 : ℝ
def watermelon_unit_price := watermelon_total_yuan / watermelon_kg

def pineapple_kg := 3 : ℝ
def pineapple_total_yuan := 15.6 : ℝ
def pineapple_unit_price := pineapple_total_yuan / pineapple_kg

-- Define the statement to be proved
theorem pineapple_more_expensive_by_3_3_yuan :
  (pineapple_unit_price - watermelon_unit_price) = 3.3 :=
by
  sorry

end pineapple_more_expensive_by_3_3_yuan_l192_192545


namespace number_of_divisors_of_8_fact_l192_192888

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192888


namespace ball_count_l192_192369

theorem ball_count (white red blue : ℕ)
  (h_ratio : white = 4 ∧ red = 3 ∧ blue = 2)
  (h_white : white = 16) :
  red = 12 ∧ blue = 8 :=
by
  -- Proof skipped
  sorry

end ball_count_l192_192369


namespace optimal_production_l192_192128

noncomputable def optimal_strategy_natural_gas := 3032
noncomputable def optimal_strategy_liquefied_gas := 2954

theorem optimal_production (mild_natural mild_liquefied severe_natural severe_liquefied cost_natural cost_liquefied price_natural price_liquefied : ℕ) :
  (mild_natural = 2200 ∧ mild_liquefied = 3500 ∧ severe_natural = 3800 ∧ severe_liquefied = 2450) →
  (cost_natural = 19 ∧ cost_liquefied = 25 ∧ price_natural = 35 ∧ price_liquefied = 58) →
  (optimal_strategy_natural_gas = 3032 ∧ optimal_strategy_liquefied_gas = 2954) :=
by
  intros,
  sorry

end optimal_production_l192_192128


namespace ratio_of_surface_areas_l192_192362

theorem ratio_of_surface_areas (r_large r_small : ℝ) (h_large : r_large = 6) (h_small : r_small = 3) :
  (4 * real.pi * r_large^2) / (4 * real.pi * r_small^2) = 4 :=
by
  rw [h_large, h_small] -- substitute given values
  -- calculate and simplify
  sorry

end ratio_of_surface_areas_l192_192362


namespace num_possible_integer_values_x_l192_192336

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192336


namespace gcd_8251_6105_l192_192483

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 :=
by
  sorry

end gcd_8251_6105_l192_192483


namespace cyclic_pentagon_area_ratio_l192_192108

noncomputable def cyclic_pentagon_ratio (R : ℝ) (α : ℕ → ℝ) : Prop :=
  let area := (1 / 2) * R^2 * (α 1).sin + (α 2).sin + (α 3).sin + (α 4).sin + (α 5).sin in
  let diagonals_sum := 2 * R * ((α 1 + α 2) / 2).sin + ((α 2 + α 3) / 2).sin + ((α 3 + α 4) / 2).sin + ((α 4 + α 5) / 2).sin + ((α 5 + α 1) / 2).sin in
  area ≤ (R / 4) * diagonals_sum

theorem cyclic_pentagon_area_ratio (R : ℝ) (α : ℕ → ℝ) 
  (hα : ∀ i, 1 ≤ i ∧ i ≤ 5 → 0 < α i ∧ α i < π) : cyclic_pentagon_ratio R α := 
sorry

end cyclic_pentagon_area_ratio_l192_192108


namespace unique_triple_l192_192670

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def find_triples (x y z : ℕ) : Prop :=
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  is_prime x ∧ is_prime y ∧ is_prime z ∧
  is_prime (x - y) ∧ is_prime (y - z) ∧ is_prime (x - z)

theorem unique_triple :
  ∀ (x y z : ℕ), find_triples x y z → (x, y, z) = (7, 5, 2) :=
by
  sorry

end unique_triple_l192_192670


namespace find_number_l192_192465

theorem find_number (x : ℤ) (h : x - 254 + 329 = 695) : x = 620 :=
sorry

end find_number_l192_192465


namespace limit_calc_l192_192631

noncomputable def limit_expr (x : ℝ) : ℝ := (1 - real.sqrt (real.cos x)) / (x * real.sin x)

theorem limit_calc : filter.tendsto limit_expr (nhds 0) (nhds (1 / 4)) :=
sorry

end limit_calc_l192_192631


namespace percentage_increase_in_consumption_l192_192037

theorem percentage_increase_in_consumption 
  (T C : ℝ) 
  (h1 : 0.8 * T * C * (1 + P / 100) = 0.88 * T * C)
  : P = 10 := 
by 
  sorry

end percentage_increase_in_consumption_l192_192037


namespace circumcenter_of_triangle_l192_192199

-- Definitions and conditions from the problem:
variable (O O1 O2 : Type) -- The three circles
variable {l1 l2 : Line} -- The parallel lines
variable {A B C D E Q : Point} -- The points
variable {AD BC : Line} -- The lines AD and BC

-- Hypotheses translating the given conditions:
-- Circle O is tangent to l1 and l2
def tangent_to_lines (O : Type) (l1 l2 : Line) : Prop := sorry

-- Circle O1 is tangent to l1 at A and externally tangent to circle O at C
def tangent_at_point (O1 : Type) (l1 : Line) (A : Point) : Prop := sorry
def external_tangent (O1 O : Type) (C : Point) : Prop := sorry

-- Circle O2 is tangent to l2 at B, externally tangent to O at D, and
-- externally tangent to O1 at E
def tangent_to_point_and_circles (O2 : Type) (l2 : Line) (B D E : Point) (O O1 : Type) : Prop := sorry

-- Lines AD and BC describe the intersections
def line_through_points (A D B C : Point) (AD BC : Line) (Q: Point): Prop := sorry

-- Formal theorem statement:
theorem circumcenter_of_triangle (
  hc1: tangent_to_lines O l1 l2, 
  hc2: tangent_at_point O1 l1 A, 
  hc3: external_tangent O1 O C, 
  hc4: tangent_to_point_and_circles O2 l2 B D E O O1, 
  hl: line_through_points A D B C AD BC Q):
(Q = circumcenter C D E) := sorry

end circumcenter_of_triangle_l192_192199


namespace num_positive_divisors_8_factorial_l192_192932

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192932


namespace medians_form_right_triangle_l192_192386

theorem medians_form_right_triangle
  (A B C H M K L N T : Type)
  [HG: geometric_structure A B C H M K L N T]
  (H_part : ∀ A B C H M, divides (altitude A H) (median B M) → equal_parts B M):
  right_triangle (medians_of_triangle (triangle A B M)) :=
begin
  sorry
end

end medians_form_right_triangle_l192_192386


namespace number_of_integer_values_l192_192271

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192271


namespace expression_for_g_odd_function_f_range_a_range_k_l192_192725

noncomputable def g (x : ℝ) := 2^x

noncomputable def f (x : ℝ) := (1 - 2^x) / (2 + 2^(x + 1))

theorem expression_for_g : g 3 = 8 := sorry

theorem odd_function_f (x : ℝ) : f (-x) = -f x := sorry

theorem range_a (a : ℝ) (h : ∃ x ∈ Ioo (-1 : ℝ) (1 : ℝ), f x + a = 0) : -1/6 < a ∧ a < 1/6 := sorry

theorem range_k (k : ℝ) (h : ∀ t ∈ Ioo (-4 : ℝ) (4 : ℝ), f (6 * t - 3) + f (t^2 - k) < 0) : k < -12 := sorry

end expression_for_g_odd_function_f_range_a_range_k_l192_192725


namespace positive_divisors_8_factorial_l192_192844

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192844


namespace cos_pi_minus_2alpha_l192_192355

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (Real.pi / 2 - α) = Real.sqrt 2 / 3) : 
  Real.cos (Real.pi - 2 * α) = -5 / 9 := by
  sorry

end cos_pi_minus_2alpha_l192_192355


namespace erick_total_revenue_l192_192058

def lemon_price_increase := 4
def grape_price_increase := lemon_price_increase / 2
def original_lemon_price := 8
def original_grape_price := 7
def lemons_sold := 80
def grapes_sold := 140

def new_lemon_price := original_lemon_price + lemon_price_increase -- $12 per lemon
def new_grape_price := original_grape_price + grape_price_increase -- $9 per grape

def revenue_from_lemons := lemons_sold * new_lemon_price -- $960
def revenue_from_grapes := grapes_sold * new_grape_price -- $1260

def total_revenue := revenue_from_lemons + revenue_from_grapes

theorem erick_total_revenue : total_revenue = 2220 := by
  -- Skipping proof with sorry
  sorry

end erick_total_revenue_l192_192058


namespace num_pos_divisors_fact8_l192_192781

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192781


namespace find_x0_l192_192200

-- Define the function f and its properties.
variables {α β : Type*} [linear_ordered_field α]
variable (f : α → α)
variable (f_inv : α → α)

-- Conditions given in the problem.
axiom f_inv_0_1 : ∀ y ∈ set.Ioc 0 (1 : α), f_inv y ∈ set.Ico 0 2
axiom f_inv_2_inf : ∀ y ∈ set.Ioi (2 : α), f_inv y ∈ set.Ioo 0 1
axiom f_inverse : ∀ y, f (f_inv y) = y ∧ f_inv (f y) = y

-- Prove the question.
theorem find_x0 : ∃ x : α, x ∈ set.Ioo 0 1 ∧ f x = x :=
sorry

end find_x0_l192_192200


namespace milk_powder_sample_l192_192377

theorem milk_powder_sample :
  ∃ (bags : Finset ℕ), 
  (∀ x ∈ bags, 1 ≤ x ∧ x ≤ 50) ∧ 
  bags.card = 5 ∧ 
  ∃ d, bags = {6, 16, 26, 36, 46} :=
by
  sorry

end milk_powder_sample_l192_192377


namespace cosine_dihedral_angle_l192_192527

/-- Two spheres of radius r1 and 1.5 * r1 are inscribed in a dihedral angle
    and touch each other. The line connecting their centers forms a 30-degree angle
    with the edge of the dihedral angle. Prove that the cosine of this dihedral
    angle is approximately 0.68. -/
theorem cosine_dihedral_angle (r1 : ℝ) (r2 : ℝ) (h : r2 = 1.5 * r1) 
    (angle_between_centers_edge : ℝ) (h_angle : angle_between_centers_edge = 30) : 
    (cos (dihedral_angle r1 r2 h_angle)) = 0.68 :=
sorry

end cosine_dihedral_angle_l192_192527


namespace value_of_fraction_l192_192349

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end value_of_fraction_l192_192349


namespace solution_set_circle_l192_192685

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l192_192685


namespace number_of_divisors_of_8_fact_l192_192891

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192891


namespace sum_of_series_equals_negative_682_l192_192141

noncomputable def geometric_sum : ℤ :=
  let a := 2
  let r := -2
  let n := 10
  (a * (r ^ n - 1)) / (r - 1)

theorem sum_of_series_equals_negative_682 : geometric_sum = -682 := 
by sorry

end sum_of_series_equals_negative_682_l192_192141


namespace proof_problem_l192_192213

variable {a b m n x : ℝ}

theorem proof_problem (h1 : a = -b) (h2 : m * n = 1) (h3 : m ≠ n) (h4 : |x| = 2) :
    (-2 * m * n + (b + a) / (m - n) - x = -4 ∧ x = 2) ∨
    (-2 * m * n + (b + a) / (m - n) - x = 0 ∧ x = -2) :=
by
  sorry

end proof_problem_l192_192213


namespace trader_gain_percentage_l192_192136

-- Definition of the given conditions
def cost_per_pen (C : ℝ) := C
def num_pens_sold := 90
def gain_from_sale (C : ℝ) := 15 * C
def total_cost (C : ℝ) := 90 * C

-- Statement of the problem
theorem trader_gain_percentage (C : ℝ) : 
  (((gain_from_sale C) / (total_cost C)) * 100) = 16.67 :=
by
  -- This part will contain the step-by-step proof, omitted here
  sorry

end trader_gain_percentage_l192_192136


namespace ethanol_in_full_tank_l192_192124

theorem ethanol_in_full_tank:
  ∀ (capacity : ℕ) (vol_A : ℕ) (vol_B : ℕ) (eth_A_perc : ℝ) (eth_B_perc : ℝ) (eth_A : ℝ) (eth_B : ℝ),
  capacity = 208 →
  vol_A = 82 →
  vol_B = (capacity - vol_A) →
  eth_A_perc = 0.12 →
  eth_B_perc = 0.16 →
  eth_A = vol_A * eth_A_perc →
  eth_B = vol_B * eth_B_perc →
  eth_A + eth_B = 30 :=
by
  intros capacity vol_A vol_B eth_A_perc eth_B_perc eth_A eth_B h1 h2 h3 h4 h5 h6 h7
  sorry

end ethanol_in_full_tank_l192_192124


namespace find_equation_of_line_l192_192223

open Real

noncomputable def equation_of_line : Prop :=
  ∃ c : ℝ, (∀ (x y : ℝ), (3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 → 2 * x + 3 * y + c = 0)) ∧
  ∃ x y : ℝ, 3 * x + 5 * y - 4 = 0 ∧ 6 * x - y + 3 = 0 ∧
              (2 * x + 3 * y + c = 0 → 6 * x + 9 * y - 7 = 0)

theorem find_equation_of_line : equation_of_line :=
sorry

end find_equation_of_line_l192_192223


namespace grouping_of_guides_and_tourists_l192_192046

theorem grouping_of_guides_and_tourists :
  ∑ k in finset.range (8 - 1) + 1, nat.choose 8 k = 254 := 
by 
  sorry

end grouping_of_guides_and_tourists_l192_192046


namespace complex_division_identity_l192_192475

noncomputable def complex_example : Prop :=
  (2 : ℂ) / (1 + (0 : ℂ) + I) = (1 : ℂ) - I

theorem complex_division_identity : complex_example :=
  by sorry

end complex_division_identity_l192_192475


namespace count_integer_values_l192_192315

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192315


namespace num_divisors_fact8_l192_192960

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192960


namespace packsOfRedBalls_l192_192432

variable (R : ℕ)

def packsYellow := 8
def packsGreen := 4
def ballsPerPack := 10
def totalBalls := 160
def totalYellowBalls := packsYellow * ballsPerPack
def totalGreenBalls := packsGreen * ballsPerPack
def totalOtherBalls := totalYellowBalls + totalGreenBalls
def totalRedBalls := totalBalls - totalOtherBalls

theorem packsOfRedBalls : R = 4 :=
by
  have : totalYellowBalls = 80 := rfl
  have : totalGreenBalls = 40 := rfl
  have : totalOtherBalls = 120 := by rw [totalYellowBalls, totalGreenBalls]
  have : totalRedBalls = 40 := by rw [totalBalls, totalOtherBalls]
  show R = 4
  sorry

end packsOfRedBalls_l192_192432


namespace increasing_intervals_length_BC_l192_192739

open Real

def vec_m (x : ℝ) : ℝ × ℝ := (2 * cos x, sin x)
def vec_n (x : ℝ) : ℝ × ℝ := (cos x, 2 * sqrt 3 * cos x)
def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2 - 1

-- Prove that f(x) is monotonically increasing in the specified intervals
theorem increasing_intervals :
  ∀ k : ℤ, ∀ x : ℝ, (k * π - π / 3) < x ∧ x < (k * π + π / 6) → deriv f x > 0 :=
sorry

-- Prove the length of side BC given the conditions
theorem length_BC (A : ℝ) (B : ℝ) (AB : ℝ) (hA : f A = 2) (hB : B = π / 4) (hAB : AB = 3) :
  side_length_BC A B AB = (3 * (sqrt 6 - sqrt 2)) / 2 :=
sorry

end increasing_intervals_length_BC_l192_192739


namespace additional_hours_equal_five_l192_192619

-- The total hovering time constraint over two days
def total_time : ℕ := 24

-- Hovering times for each zone on the first day
def day1_mountain_time : ℕ := 3
def day1_central_time : ℕ := 4
def day1_eastern_time : ℕ := 2

-- Additional hours on the second day (variables M, C, E)
variables (M C E : ℕ)

-- The main proof statement
theorem additional_hours_equal_five 
  (h : day1_mountain_time + M + day1_central_time + C + day1_eastern_time + E = total_time) :
  M = 5 ∧ C = 5 ∧ E = 5 :=
by
  sorry

end additional_hours_equal_five_l192_192619


namespace gcd_lcm_of_45_and_150_l192_192482

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem gcd_lcm_of_45_and_150 :
  GCD 45 150 = 15 ∧ LCM 45 150 = 450 :=
by
  sorry

end gcd_lcm_of_45_and_150_l192_192482


namespace line_passes_through_fixed_point_l192_192243

theorem line_passes_through_fixed_point :
  ∃ (M : ℝ × ℝ), (∀ (m : ℝ), (2 + m) * M.1 + (1 - 2 * m) * M.2 + 4 - 3 * m = 0) :=
begin
  use (-1, -2),
  intros m,
  sorry
end

end line_passes_through_fixed_point_l192_192243


namespace equation_infinitely_many_solutions_l192_192687

theorem equation_infinitely_many_solutions
  (a b c : ℝ)
  (h : ∀ x : ℝ, x + a * real.sqrt x + b = c^2 ∨ x + a * real.sqrt x + b = c^2)
  : a = -2 * c ∧ b = c^2 ∧ c > 0 := 
sorry

end equation_infinitely_many_solutions_l192_192687


namespace range_of_omega_l192_192416

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x) - cos (ω * x)

theorem range_of_omega (ω : ℝ) (hω : 5/4 < ω ∧ ω ≤ 9/4) :
  ∃ (a b : ℝ), a < b ∧ (0 < a ∧ a < π) ∧ (0 < b ∧ b < π) ∧ f ω a = 0 ∧ f ω b = 0 ∧ ∀ (c : ℝ), (0 < c ∧ c < π) → f ω c = 0 → (c = a ∨ c = b) :=
by sorry

end range_of_omega_l192_192416


namespace radius_of_sphere_eq_l192_192715

-- Definitions
variable (O : Type) [MetricSpace O] (P A B C : O)
variable (r : ℝ)
variable (PA PB PC : ℝ) -- lengths of the chords
variable [Nonempty O]

-- Conditions
axiom P_on_sphere : dist P O = r
axiom PA_perpendicular_PB : ∥PA - P∥ ⬝ ∥PB - P∥ = 0
axiom PB_perpendicular_PC : ∥PB - P∥ ⬝ ∥PC - P∥ = 0
axiom PC_perpendicular_PA : ∥PC - P∥ ⬝ ∥PA - P∥ = 0
axiom max_distance_to_plane : dist P (plane A B C) = 1

-- Theorem to prove
theorem radius_of_sphere_eq : r = 3 / 2 :=
sorry

end radius_of_sphere_eq_l192_192715


namespace brownies_total_l192_192435

theorem brownies_total :
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  after_mooney_ate + additional_brownies = 36 :=
by
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  show after_mooney_ate + additional_brownies = 36
  sorry

end brownies_total_l192_192435


namespace count_possible_integer_values_l192_192288

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192288


namespace triangle_third_side_lengths_l192_192984

theorem triangle_third_side_lengths (a b : ℕ) (hx1 : a = 10) (hx2 : b = 15) : 
  {x : ℕ | 5 < x ∧ x < 25}.card = 19 := 
by
  sorry

end triangle_third_side_lengths_l192_192984


namespace translation_maintains_double_root_distance_l192_192640

-- Assume definitions and conditions
variables (a b c p : ℝ)
variables (h_a_pos : a > 0)
variables (h_distance_roots : (b^2 - 4 * a * c) / a^2 = p^2)

-- Prove that translating the function downward by (3b^2 / (4 * a) - 3 * c) results in distance 2p between the roots
theorem translation_maintains_double_root_distance :
  let c' := c - (3 * b^2 / (4 * a) - 3 * c) in
  (b^2 - 4 * a * c') / a^2 = (2 * p)^2 := 
sorry

end translation_maintains_double_root_distance_l192_192640


namespace pos_divisors_8_factorial_l192_192902

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192902


namespace convert_base_6_l192_192148

-- Definitions of powers of 6 up to 6^3
def pow_6_0 := 1
def pow_6_1 := 6
def pow_6_2 := 36
def pow_6_3 := 216

-- Relevant conditions
def largest_power_under_512 := pow_6_3

theorem convert_base_6 : nat.base_repr 512 6 = [2, 2, 1, 2] :=
by
  sorry

end convert_base_6_l192_192148


namespace sqrt_three_is_infinite_non_repeating_decimal_l192_192455

theorem sqrt_three_is_infinite_non_repeating_decimal :
  (¬ ∃ p q : ℤ, q ≠ 0 ∧ (p:ℝ) / (q:ℝ) = real.sqrt 3) ∧
  (¬ ∃ (m n : ℕ), (real.sqrt 3) = (m / n : ℝ)) :=
begin
  sorry
end

end sqrt_three_is_infinite_non_repeating_decimal_l192_192455


namespace range_of_a_for_empty_solution_set_l192_192363

theorem range_of_a_for_empty_solution_set :
  {a : ℝ | ∀ x : ℝ, (a^2 - 9) * x^2 + (a + 3) * x - 1 < 0} = 
  {a : ℝ | -3 ≤ a ∧ a < 9 / 5} :=
sorry

end range_of_a_for_empty_solution_set_l192_192363


namespace shortest_and_longest_paths_l192_192118

theorem shortest_and_longest_paths (O A B C : Point) (r : ℝ)
  (h_circle : Adva.circle(O, r))
  (h_diameter : Points_on_diameter_equal_distance(O, A, B, r)) :
  shortest_path(O, A, C, B, 2 * r) ∧ longest_path(O, A, C, B) := by
  -- Prove that the shortest path from A to C and then to B is 2r
  -- Prove that the path can be extended using an ellipse
  sorry

end shortest_and_longest_paths_l192_192118


namespace triple_angle_sum_formula_l192_192145

theorem triple_angle_sum_formula (α : ℝ) (n : ℕ) (h : ∀ θ : ℝ, sin (3 * θ) = 3 * sin θ - 4 * (sin θ)^3) :
  (∑ k in Finset.range n, 3^k * (sin (α / 3^k))^3) = (1 / 4) * (3^n * (sin (α / 3^n)) - sin α) :=
by sorry

end triple_angle_sum_formula_l192_192145


namespace find_zero_sequence_l192_192464

def sequences_condition (a b c d : ℕ) (n : ℕ) :=
  (|a - b|, |b - c|, |c - d|, |d - a|)

def is_zero_sequence (a_1 b_1 c_1 d_1 : ℕ) : Prop :=
  ∃ k : ℕ, let (a_k, b_k, c_k, d_k) := Nat.iterate sequences_condition k (a_1, b_1, c_1, d_1) in 
  a_k = 0 ∧ b_k = 0 ∧ c_k = 0 ∧ d_k = 0

-- Given four initial integers a1, b1, c1, d1, we need to prove the following:
theorem find_zero_sequence (a_1 b_1 c_1 d_1 : ℕ) : 
  is_zero_sequence a_1 b_1 c_1 d_1 :=
sorry

end find_zero_sequence_l192_192464


namespace prove_y_identity_l192_192469

theorem prove_y_identity (y : ℤ) (h1 : y^2 = 2209) : (y + 2) * (y - 2) = 2205 :=
by
  sorry

end prove_y_identity_l192_192469


namespace retailer_selling_price_l192_192577

theorem retailer_selling_price
  (cost_price_manufacturer : ℝ)
  (manufacturer_profit_rate : ℝ)
  (wholesaler_profit_rate : ℝ)
  (retailer_profit_rate : ℝ)
  (manufacturer_selling_price : ℝ)
  (wholesaler_selling_price : ℝ)
  (retailer_selling_price : ℝ)
  (h1 : cost_price_manufacturer = 17)
  (h2 : manufacturer_profit_rate = 0.18)
  (h3 : wholesaler_profit_rate = 0.20)
  (h4 : retailer_profit_rate = 0.25)
  (h5 : manufacturer_selling_price = cost_price_manufacturer + (manufacturer_profit_rate * cost_price_manufacturer))
  (h6 : wholesaler_selling_price = manufacturer_selling_price + (wholesaler_profit_rate * manufacturer_selling_price))
  (h7 : retailer_selling_price = wholesaler_selling_price + (retailer_profit_rate * wholesaler_selling_price)) :
  retailer_selling_price = 30.09 :=
by {
  sorry
}

end retailer_selling_price_l192_192577


namespace problem_1_problem_2_l192_192697

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

theorem problem_1 (a b : ℝ) (h_odd : ∀ x : ℝ, f a b x = - f a b (-x))
                      (h : f a b (1/2) = 4/5) :
                      f a b x = 2 * x / (1 + x^2) :=
begin
  sorry
end

theorem problem_2 : ∀ x₁ x₂ : ℝ, x₁ ∈ set.Ioo (-1) 1 → x₂ ∈ set.Ioo (-1) 1 → x₁ < x₂ →
                      (2 * x₁ / (1 + x₁^2)) < (2 * x₂ / (1 + x₂^2)) :=
begin
  sorry
end

end problem_1_problem_2_l192_192697


namespace solve_for_x_l192_192983

theorem solve_for_x (x : ℚ) :  (1/2) * (12 * x + 3) = 3 * x + 2 → x = 1/6 := by
  intro h
  sorry

end solve_for_x_l192_192983


namespace sum_max_min_on_interval_l192_192036

noncomputable def y (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem sum_max_min_on_interval : 
  ∃ (a b : ℝ), (∀ x ∈ (Icc 0 3 : set ℝ), y(x) ≥ a ∧ y(x) ≤ b) ∧ a + b = 0 :=
  sorry

end sum_max_min_on_interval_l192_192036


namespace values_of_a_for_single_root_l192_192736

theorem values_of_a_for_single_root (a : ℝ) :
  (∃ (x : ℝ), ax^2 - 4 * x + 2 = 0) ∧ (∀ (x1 x2 : ℝ), ax^2 - 4 * x1 + 2 = 0 → ax^2 - 4 * x2 + 2 = 0 → x1 = x2) ↔ a = 0 ∨ a = 2 :=
sorry

end values_of_a_for_single_root_l192_192736


namespace ABM_collinear_l192_192044

variables {k : Type*} [EuclideanSpace k]

-- Define points A, B where two circles intersect
variables (A B C D M : k)

-- Define the circles with given intersection points A and B
noncomputable def circle1 := sorry -- representation of the first circle
noncomputable def circle2 := sorry -- representation of the second circle

-- Conditions:
-- The circles intersect at A and B
axiom circles_intersect_at_A_and_B : A ≠ B ∧ A ∈ circle1 ∧ B ∈ circle1 ∧ A ∈ circle2 ∧ B ∈ circle2

-- A common tangent to both circles touching at points C and D
axiom common_tangent : tangent circle1 C ∧ tangent circle2 D

-- M is the midpoint of segment CD
axiom M_is_midpoint_CD : midpoint C D M

-- Theorem: Points A, B, and M are collinear
theorem ABM_collinear : collinear {A, B, M} :=
by
  apply sorry

end ABM_collinear_l192_192044


namespace num_positive_divisors_8_factorial_l192_192930

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192930


namespace probability_even_sum_l192_192657

-- Defining the probabilities for the first wheel
def P_even_1 : ℚ := 2/3
def P_odd_1 : ℚ := 1/3

-- Defining the probabilities for the second wheel
def P_even_2 : ℚ := 1/2
def P_odd_2 : ℚ := 1/2

-- Prove that the probability that the sum of the two selected numbers is even is 1/2
theorem probability_even_sum : 
  P_even_1 * P_even_2 + P_odd_1 * P_odd_2 = 1/2 :=
by
  sorry

end probability_even_sum_l192_192657


namespace number_of_paths_below_diagonal_l192_192427

noncomputable def catalan_number (n : ℕ) : ℕ :=
  (1 / (n + 1) : ℚ * (Nat.binomial (2 * n) n)).to_nat

theorem number_of_paths_below_diagonal (n : ℕ) :
  ∑ (path : ℕ → ℕ),
    (path 0 = 0) ∧
    (path n = n) ∧
    (∀ m : ℕ, m ≤ n → path m ≤ m ∧ (path (m + 1) = path m ∨ path (m + 1) = path m + 1)) ∧
    (∑ m in Finset.range n, (path (m+1) - path m) = 1) =
  catalan_number n := sorry

end number_of_paths_below_diagonal_l192_192427


namespace value_of_fraction_l192_192347

theorem value_of_fraction (x y z w : ℕ) (h₁ : x = 4 * y) (h₂ : y = 3 * z) (h₃ : z = 5 * w) :
  x * z / (y * w) = 20 := by
  sorry

end value_of_fraction_l192_192347


namespace max_sum_of_arithmetic_sequence_l192_192702

theorem max_sum_of_arithmetic_sequence :
  ∃ n : ℕ, (∀ m : ℕ, m ≠ n → 
    (let a : ℕ → ℝ := λ n, 43 - 3 * n in 
     let S : ℕ → ℝ := λ n, n / 2 * (2 * a 1 + (n - 1) * (-3)) in 
     S n > S m)) ∧ n = 14 :=
by 
  sorry

end max_sum_of_arithmetic_sequence_l192_192702


namespace imaginary_part_of_z_l192_192699

-- Define the complex number z given in the problem
def z : ℂ := 1 - 2 * complex.I

-- State the theorem for the imaginary part of z
theorem imaginary_part_of_z : im z = -2 :=
by
  sorry

end imaginary_part_of_z_l192_192699


namespace problem_1_problem_2_l192_192085

-- Problem 1
theorem problem_1 : 
  sqrt 3 + sqrt 8 < 2 + sqrt 7 :=
by
  -- Proof by analysis, skipped
  sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : 
  1 / a + 1 / b + 1 / c ≥ 9 :=
by
  -- Proof by AM-GM inequality, skipped
  sorry

end problem_1_problem_2_l192_192085


namespace geometric_series_sum_l192_192539

noncomputable def geometric_sum : ℚ :=
  let a := (2^3 : ℚ) / (3^3)
  let r := (2 : ℚ) / 3
  let n := 12 - 3 + 1
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  geometric_sum = 1440600 / 59049 :=
by
  sorry

end geometric_series_sum_l192_192539


namespace number_of_isosceles_triangles_with_perimeter_25_l192_192745

def is_isosceles_triangle (a b : ℕ) : Prop :=
  a + 2 * b = 25 ∧ 2 * b > a ∧ a < 2 * b

theorem number_of_isosceles_triangles_with_perimeter_25 :
  (finset.filter (λ b, ∃ a, is_isosceles_triangle a b)
                 (finset.range 13)).card = 6 := by
sorry

end number_of_isosceles_triangles_with_perimeter_25_l192_192745


namespace total_bricks_used_l192_192443

def numberOfCoursesPerWall := 6
def bricksPerCourse := 10
def numberOfWalls := 4
def incompleteCourses := 2

theorem total_bricks_used :
  (numberOfCoursesPerWall * bricksPerCourse * (numberOfWalls - 1)) + ((numberOfCoursesPerWall - incompleteCourses) * bricksPerCourse) = 220 :=
by
  -- Proof goes here
  sorry

end total_bricks_used_l192_192443


namespace triangle_ratios_sum_l192_192385

open Classical

theorem triangle_ratios_sum (A B C D E F : Point) (BC_midpoint : Midpoint D B C) 
  (E_on_AB : E ∈ segment A B) (AE_EB_ratio : ratio A E E B 2 1) 
  (F_on_AD : F ∈ segment A D) (AF_FD_ratio : ratio A F F D 3 2) : 
  ratio_sum EF FC + ratio_sum AF FD = 7 / 6 :=
  sorry

end triangle_ratios_sum_l192_192385


namespace distance_speed_proof_l192_192583

noncomputable def distance_from_A_to_B : ℝ := 6
noncomputable def speed_of_rider : ℝ := 7.2
noncomputable def speed_of_pedestrian : ℝ := 3.6
noncomputable def round_trip_time : ℝ := 100 / 60  -- in hours
noncomputable def time_difference : ℝ := 50 / 60  -- in hours
noncomputable def meeting_distance : ℝ := 2

theorem distance_speed_proof :
  let t_rider := round_trip_time / 2 in
  let t_pedestrian := t_rider + time_difference in
  let distance := distance_from_A_to_B in
  let speed_rider := speed_of_rider in
  let speed_pedestrian := speed_of_pedestrian in
  distance = 6 ∧
  speed_rider = distance / t_rider ∧
  speed_pedestrian = speed_rider / 2 :=
by
  sorry

end distance_speed_proof_l192_192583


namespace monotonic_interval_a_eq_3_smallest_integer_a_inequality_no_common_tangent_line_l192_192229

-- Condition definitions
def f (x : ℝ) (a : ℝ) := 2 * Real.log x - a * x
def h (x : ℝ) (a : ℝ) := f x a + 1/2 * x^2

-- Question (Ⅰ) when a = 3
theorem monotonic_interval_a_eq_3 :
  (∀ x ∈ Ioo 0 (2/3 : ℝ), (derivative (λ x, f x 3) x > 0)) ∧
  (∀ x ∈ Ioo (2/3 : ℝ) ⊤, (derivative (λ x, f x 3) x < 0)) :=
sorry

-- Question (Ⅱ) smallest integer value of a
theorem smallest_integer_a_inequality :
  ∃ a : ℤ, (∀ x > 0, f x a ≤ a * x^2 + (a - 2) * x - 2) ∧ (a = 2) :=
sorry

-- Question (Ⅲ) existence of the tangent line at two distinct points
theorem no_common_tangent_line :
  ¬∃ x1 x2 : ℝ, x1 ≠ x2 ∧
  (derivative (λ x, h x a) x1 = derivative (λ x, h x a) x2) ∧
  (h x1 a - x1 * derivative (λ x, h x a) x1 = h x2 a - x2 * derivative (λ x, h x a) x2) :=
sorry

end monotonic_interval_a_eq_3_smallest_integer_a_inequality_no_common_tangent_line_l192_192229


namespace sum_of_squares_of_prime_divisors_1728_l192_192537

theorem sum_of_squares_of_prime_divisors_1728 :
  let primes := [2, 3] in
  (2^2 + 3^2 = 13) -> 
  sum_of_squares_of_prime_divisors (1728) = 13 :=
by
  sorry

end sum_of_squares_of_prime_divisors_1728_l192_192537


namespace Q_ratio_l192_192643

-- Define the polynomial g(x)
def g (x : ℂ) : ℂ := x^2011 + 13 * x^2010 + 1

-- Define the polynomial Q with the given condition
noncomputable def Q : ℂ → ℂ :=
  let s := {s : ℂ | g s = 0}
  assume z, ∏ s in s, (z - (s + 1/s))

-- Define the statement to be proved
theorem Q_ratio :
  ∀ (Q : ℂ → ℂ), (∀ s ∈ {s : ℂ | g s = 0}, Q (s + 1/s) = 0) → Q 1 / Q (-1) = 169 / 170 :=
by
  sorry

end Q_ratio_l192_192643


namespace sum_of_two_longest_altitudes_l192_192968

noncomputable def semi_perimeter (a b c : ℝ) := (a + b + c) / 2

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def altitude (a b c : ℝ) (side : ℝ) : ℝ :=
  (2 * heron_area a b c) / side

theorem sum_of_two_longest_altitudes (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) :
  let ha := altitude a b c a
  let hb := altitude a b c b
  let hc := altitude a b c c
  ha + hb = 21 ∨ ha + hc = 21 ∨ hb + hc = 21 := by
  sorry

end sum_of_two_longest_altitudes_l192_192968


namespace kayla_scores_on_sixth_level_l192_192408

-- Define the sequence of points scored in each level
def points (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 3
  | 2 => 5
  | 3 => 8
  | 4 => 12
  | n + 5 => points (n + 4) + (n + 1) + 1

-- Statement to prove that Kayla scores 17 points on the sixth level
theorem kayla_scores_on_sixth_level : points 5 = 17 :=
by
  sorry

end kayla_scores_on_sixth_level_l192_192408


namespace num_divisors_fact8_l192_192957

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192957


namespace Mike_investment_approximation_l192_192393

theorem Mike_investment_approximation : 
  ∃ (M : ℝ), M ≈ 77.78 ∧ 
    let total_profit := 3000.0000000000005 in
    let john_investment := 700 in
    let mike_share_effort := total_profit / 3 in
    let remaining_profit := total_profit - 2 * mike_share_effort in
    let john_share_ratio := (john_investment / (john_investment + M)) * remaining_profit in
    let mike_share_ratio := (M / (john_investment + M)) * remaining_profit in
    (john_share_ratio - mike_share_ratio = 800) :=
by
  sorry

end Mike_investment_approximation_l192_192393


namespace number_of_divisors_8_factorial_l192_192788

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192788


namespace number_of_integer_values_l192_192279

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192279


namespace prob_divisors_8_fact_l192_192815

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192815


namespace girls_select_same_color_probability_l192_192073

theorem girls_select_same_color_probability :
  let total_marbles := 6
  let marbles_white := 3
  let marbles_black := 3
  let total_girls := 3
  let total_boys := 3
  let probability_all_white := (3 / total_marbles) * ((marbles_white - 1) / (total_marbles - 1)) * ((marbles_white - 2) / (total_marbles - 2))
  let probability_all_black := (3 / total_marbles) * ((marbles_black - 1) / (total_marbles - 1)) * ((marbles_black - 2) / (total_marbles - 2))
  in probability_all_white + probability_all_black = 1 / 20 :=
by
  sorry

end girls_select_same_color_probability_l192_192073


namespace hyperbola_eccentricity_perpendicular_asymptotes_l192_192365

theorem hyperbola_eccentricity_perpendicular_asymptotes {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (hyp : (⟨1, 0⟩:ℝ × ℝ) ∙ ⟨0, 1⟩ = 0): 
  let e := (λ (a b : ℝ), real.sqrt(1 + (b^2 / a^2))) in
  e a b = real.sqrt(2) :=
by
  sorry

end hyperbola_eccentricity_perpendicular_asymptotes_l192_192365


namespace number_of_divisors_8_factorial_l192_192789

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192789


namespace ratio_netbooks_is_one_third_l192_192437

open Nat

def total_computers (total : ℕ) : Prop := total = 72
def laptops_sold (laptops : ℕ) (total : ℕ) : Prop := laptops = total / 2
def desktops_sold (desktops : ℕ) : Prop := desktops = 12
def netbooks_sold (netbooks : ℕ) (total laptops desktops : ℕ) : Prop :=
  netbooks = total - (laptops + desktops)
def ratio_netbooks_total (netbooks total : ℕ) : Prop :=
  netbooks * 3 = total

theorem ratio_netbooks_is_one_third
  (total laptops desktops netbooks : ℕ)
  (h_total : total_computers total)
  (h_laptops : laptops_sold laptops total)
  (h_desktops : desktops_sold desktops)
  (h_netbooks : netbooks_sold netbooks total laptops desktops) :
  ratio_netbooks_total netbooks total :=
by
  sorry

end ratio_netbooks_is_one_third_l192_192437


namespace negation_of_p_is_neg_p_l192_192210

def p (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

def neg_p (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0

theorem negation_of_p_is_neg_p (f : ℝ → ℝ) : ¬ p f ↔ neg_p f :=
by
  sorry -- Proof of this theorem

end negation_of_p_is_neg_p_l192_192210


namespace find_x_l192_192112

variables (x : ℝ)

def equidistant_walk : Prop :=
  let position := (-x + 2 * sqrt 3, -2) in
  real.sqrt ((-x + 2 * sqrt 3)^2 + (-2)^2) = 2

theorem find_x (h : equidistant_walk x) :
  x = 2 * sqrt 3 + 2 ∨ x = 2 * sqrt 3 - 2 :=
sorry

end find_x_l192_192112


namespace pos_divisors_8_factorial_l192_192911

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192911


namespace min_sum_of_three_integers_l192_192029

theorem min_sum_of_three_integers (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_product : a * b * c = 5^3) : 
  a + b + c ≥ 15 :=
begin
  sorry
end

end min_sum_of_three_integers_l192_192029


namespace inequality_solution_l192_192424

theorem inequality_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b + c = 1) : (1 / (b * c + a + 1 / a) + 1 / (c * a + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31) :=
by sorry

end inequality_solution_l192_192424


namespace seq_determinant_G_784_786_minus_G_785_2_l192_192387

structure Seq :=
  (G : ℕ → ℤ)
  (G0 : G 0 = 1)
  (G1 : G 1 = 2)
  (Gn : ∀ n ≥ 2, G n = 2 * G (n - 1) + G (n - 2))

def matrix_power (A : Matrix (Fin 2) (Fin 2) ℤ) (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  if n = 0 then 1 else (List.repeat A n).prod

def matrix_eq_G (seq : Seq) (n : ℕ) :=
  matrix_power ![![2, 1], ![1, 1]] n = ![![seq.G (n + 1), seq.G n], ![seq.G n, seq.G (n - 1)]]

theorem seq_determinant (seq : Seq) : ∀ n, seq.G (n + 1) * seq.G (n - 1) - seq.G n ^ 2 = 1 :=
by
  intro n
  sorry

theorem G_784_786_minus_G_785_2 (seq : Seq) : seq.G 784 * seq.G 786 - seq.G 785 ^ 2 = 1 :=
by
  have h := seq_determinant seq 785
  rwa [Nat.sub_self] at h
  sorry

end seq_determinant_G_784_786_minus_G_785_2_l192_192387


namespace number_of_divisors_of_8_fact_l192_192884

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192884


namespace max_tiles_l192_192076

def size := { length: ℕ, width: ℕ }

def tile_size : size := { length := 25, width := 16 }
def floor_size : size := { length := 180, width := 120 }

def can_fit (floor: size) (tile: size) :=
  (floor.length / tile.length) * (floor.width / tile.width)

theorem max_tiles (floor : size) (tile : size) :
  can_fit floor tile ≤ 49 :=
  sorry

end max_tiles_l192_192076


namespace value_of_frac_l192_192352

theorem value_of_frac (x y z w : ℕ) 
  (hz : z = 5 * w) 
  (hy : y = 3 * z) 
  (hx : x = 4 * y) : 
  x * z / (y * w) = 20 := 
  sorry

end value_of_frac_l192_192352


namespace num_possible_integer_values_l192_192304

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192304


namespace positive_divisors_of_8_factorial_l192_192822

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192822


namespace mag_2a_sub_b_eq_sqrt_13_l192_192714

variables (a b : ℝ^3)
variables (hab : real.angle a b = real.pi * 5 / 6)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = real.sqrt 3)

theorem mag_2a_sub_b_eq_sqrt_13
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = real.sqrt 3)
  (hab : real.angle a b = real.pi * 5 / 6) :
  ‖2 • a - b‖ = real.sqrt 13 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end mag_2a_sub_b_eq_sqrt_13_l192_192714


namespace pos_divisors_8_factorial_l192_192913

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192913


namespace monotonically_increasing_interval_l192_192488

def f (x : ℝ) : ℝ := sqrt (-x^2 + 6*x - 5)

theorem monotonically_increasing_interval :
  ∀ x, 1 ≤ x ∧ x ≤ 3 → ∃ y, (f y) = f x :=
by
  sorry

end monotonically_increasing_interval_l192_192488


namespace polar_and_cartesian_eq_of_curves_l192_192009

noncomputable def C1_parametric_eq := (α : ℝ) → (α ∈ Set.Ico 0 Real.pi) → (cos α, sin α)

noncomputable def C2_polar_eq := (θ : ℝ) → -2 * sin θ

theorem polar_and_cartesian_eq_of_curves :
  (∀ θ ∈ Set.Ico 0 Real.pi, (1 : ℝ)) ∧ (∀ (x y : ℝ), x^2 + (y + 1)^2 = 1) ∧
  (∀ (x₀ y₀ : ℝ) (h₀ : 0 ≤ y₀ ∧ y₀ ≤ 1) (α : ℝ) (hα : α ∈ Set.Ico 0 Real.pi)
     (t : ℝ), (x₀, y₀) = C1_parametric_eq α hα →
    ∃ (PM PN : ℝ), PM * PN = abs (1 + 2 * y₀) ∧ (1 ≤ abs (1 + 2 * y₀)) ∧ (abs (1 + 2 * y₀) ≤ 3)) :=
by
  sorry

end polar_and_cartesian_eq_of_curves_l192_192009


namespace square_perimeter_l192_192593

theorem square_perimeter (s : ℝ) (r_perimeter : ℝ) (h1 : 4 * s = 4 * s) (h2 : r_perimeter = 32) : 4 * (64 / 5) = 51.2 :=
by
  have h3 : 2 * (s + s / 4) = 32 := by sorry -- This is derived from the perimeter condition of the rectangle
  have h4 : (5 / 2) * s = 32 := by sorry -- Simplifying h3
  let s := 64 / 5 -- Solving for s from h4
  have h5 : s = 12.8 := by sorry -- Simplified value of s
  calc
    4 * s = 4 * 12.8 : by rw h5
    ... = 51.2 : by norm_num


end square_perimeter_l192_192593


namespace pos_divisors_8_factorial_l192_192912

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192912


namespace probability_is_pi_over_32_l192_192580

open Set MeasureTheory ProbabilityTheory

-- Define the rectangle as a set in ℝ²
def rectangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define the event as a set in ℝ² where x² + y² < y
def event : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 < p.2}

-- The measure of the rectangle is the Lebesgue measure in 2 dimensions
def rect_measure := volume (rectangle : Set (ℝ × ℝ))

-- The measure of the event in the rectangle
def event_measure := volume (event ∩ rectangle)

-- Probability that x² + y² < y given (x, y) is randomly picked from the rectangle
def prob : ℝ := event_measure / rect_measure

theorem probability_is_pi_over_32 : prob = π / 32 :=
by
  sorry

end probability_is_pi_over_32_l192_192580


namespace robbery_proof_l192_192616

variables (A B V G : Prop)

-- Define the conditions as Lean propositions
def condition1 : Prop := ¬G → (B ∧ ¬A)
def condition2 : Prop := V → (¬A ∧ ¬B)
def condition3 : Prop := G → B
def condition4 : Prop := B → (A ∨ V)

-- The statement we want to prove based on conditions
theorem robbery_proof (h1 : condition1 A B G) 
                      (h2 : condition2 A B V) 
                      (h3 : condition3 B G) 
                      (h4 : condition4 A B V) : 
                      A ∧ B ∧ G :=
begin
  sorry
end

end robbery_proof_l192_192616


namespace length_of_conjugate_axis_hyperbola_l192_192723

theorem length_of_conjugate_axis_hyperbola :
  ∃ (a : ℝ), (c = sqrt(13)) ∧ (a ^ 2 = 4) ∧ (2 * sqrt(a^2) = 4) :=
begin
  -- Hyperbola equation condition
  let h : ℝ := 0,
  let k : ℝ := 0,
  let a2 : ℝ := 9,
  -- Right focus condition
  let c : ℝ := sqrt(13),
  -- Length of conjugate axis
  use sqrt(4),
  exact sorry
end

end length_of_conjugate_axis_hyperbola_l192_192723


namespace tangent_line_value_l192_192504

theorem tangent_line_value (f : ℝ → ℝ) (hf : differentiable ℝ f) :
  (∃ (a b : ℝ), ∀ x, f'(5) = -1 ∧ f 5 = 3 ∧ f(5) + f'(5) = 2) :=
by
  sorry

end tangent_line_value_l192_192504


namespace positive_divisors_8_factorial_l192_192839

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192839


namespace pos_divisors_8_factorial_l192_192904

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192904


namespace probability_two_checkpoints_no_red_then_red_l192_192010

theorem probability_two_checkpoints_no_red_then_red (p_red : ℚ) (p_no_red : ℚ) (independent : Prop) (total_checkpoints : ℕ) :
  p_red = 1/3 →
  p_no_red = 2/3 →
  independent →
  total_checkpoints = 6 →
  (p_no_red * p_no_red * p_red) = 4/27 :=
by
  intros h_red h_no_red h_indep h_total
  rw [h_red, h_no_red, h_total]
  simp
  sorry

end probability_two_checkpoints_no_red_then_red_l192_192010


namespace integer_values_of_x_l192_192293

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192293


namespace fundamental_theorem_of_calculus_l192_192471

theorem fundamental_theorem_of_calculus {f F : ℝ → ℝ} {a b : ℝ} 
  (h_cont : ContinuousOn f (interval a b)) 
  (h_deriv : ∀ x ∈  set.Icc a b, has_deriv_at F (f x) x) :
  ∫ x in a..b, f x = F b - F a :=
by
  sorry

end fundamental_theorem_of_calculus_l192_192471


namespace num_divisors_fact8_l192_192947

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192947


namespace num_divisors_of_8_factorial_l192_192874

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192874


namespace number_of_integer_values_l192_192327

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192327


namespace number_of_divisors_8_factorial_l192_192796

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192796


namespace august_five_times_wednesday_l192_192007

theorem august_five_times_wednesday (N : ℕ) (july_has_31_days : ∀ d ∈ (finset.range 31).image (λ x, x + 1), true)
  (july_has_five_tuesdays : ∃ days : finset ℕ, 
    days.card = 5 ∧ 
    ∀ d ∈ days, (d - 1) % 7 = 1) 
  (august_has_31_days : ∀ d ∈ (finset.range 31).image (λ x, x + 1), true) :
  ∃ days : finset ℕ, 
    days.card = 5 ∧ 
    ∀ d ∈ days, 
      (n -> ((d - n) % 7 = 3)) := -- Using modular arithmetic to represent days, where 3 represents Wednesday
sorry

end august_five_times_wednesday_l192_192007


namespace pos_divisors_8_factorial_l192_192898

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192898


namespace number_of_almost_centers_of_symmetry_at_most_three_l192_192558

/- 
Consider a finite set M of points on the plane. 
A point O is called an "almost center of symmetry" of the set M if it is possible to remove one point from M such that O becomes a center of symmetry for the remaining set.
We need to prove that the number of "almost centers of symmetry" of M is an element of the set {0, 1, 2, 3}.
-/

def is_center_of_symmetry (O : Point) (M : Finset Point) : Prop := 
  ∀ (A B : Point), A ∈ M → B ∈ M → O = midpoint A B

def is_almost_center_of_symmetry (O : Point) (M : Finset Point) : Prop :=
  ∃ (P : Point), P ∈ M ∧ is_center_of_symmetry O (M.erase P)

theorem number_of_almost_centers_of_symmetry_at_most_three (M : Finset Point) :
  let centers := {O : Point | is_almost_center_of_symmetry O M}
  centers.card ≤ 3 :=
sorry 

end number_of_almost_centers_of_symmetry_at_most_three_l192_192558


namespace derivative_of_sin3x_at_pi_div_9_is_3_div_2_l192_192531

noncomputable def derivative_sin3x_at_pi_div_9 : ℝ := 
  (deriv (λ x, Real.sin (3 * x)) (Real.pi / 9))

-- The theorem to prove
theorem derivative_of_sin3x_at_pi_div_9_is_3_div_2 :
  derivative_sin3x_at_pi_div_9 = 3 / 2 :=
sorry

end derivative_of_sin3x_at_pi_div_9_is_3_div_2_l192_192531


namespace sqrt_floor_8_integer_count_l192_192324

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192324


namespace num_positive_divisors_8_factorial_l192_192943

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192943


namespace sum_of_perpendiculars_limit_l192_192708

theorem sum_of_perpendiculars_limit (a b : ℝ) (h : a ≠ b) (h_a_gt_b : a > b) :
  let r := b / a in
  (∑' n, a * r^n) = (a^2) / (a - b) :=
by
  sorry

end sum_of_perpendiculars_limit_l192_192708


namespace number_of_integer_values_l192_192278

theorem number_of_integer_values (x : ℤ) (h : ⌊Real.sqrt x⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192278


namespace max_value_expr_l192_192447

open_locale big_operators

/-- 
  Given nine variables represented by Chinese characters from 1 to 9 such that each character represents a unique digit.
  Calculate the maximum value of the expression:
    盼 × 望 + 树 × 翠绿 + 天空 × 湛蓝
  and show it is equal to 8569.
-/
theorem max_value_expr :
  ∃ 盼 望 树 翠绿 天空 湛蓝 : ℕ, 
  (盼 ∈ {1,2,3,4,5,6,7,8,9} ∧ 望 ∈ {1,2,3,4,5,6,7,8,9} ∧ 树 ∈ {1,2,3,4,5,6,7,8,9} ∧ 
   翠绿 ∈ {1,2,3,4,5,6,7,8,9} ∧ 天空 ∈ {1,2,3,4,5,6,7,8,9} ∧ 湛蓝 ∈ {1,2,3,4,5,6,7,8,9}) ∧
  (盼 ≠ 望 ∧ 盼 ≠ 树 ∧ 盼 ≠ 翠绿 ∧ 盼 ≠ 天空 ∧ 盼 ≠ 湛蓝 ∧ 望 ≠ 树 ∧ 
  望 ≠ 翠绿 ∧ 望 ≠ 天空 ∧ 望 ≠ 湛蓝 ∧ 树 ≠ 翠绿 ∧ 树 ≠ 天空 ∧ 树 ≠ 湛蓝 ∧ 
  翠绿 ≠ 天空 ∧ 翠绿 ≠ 湛蓝 ∧ 天空 ≠ 湛蓝) ∧
  盼 * 望 + 树 * 翠绿 + 天空 * 湛蓝 = 8569 := 
sorry

end max_value_expr_l192_192447


namespace parabola_vertex_coordinates_l192_192023

def parabola_vertex_x (a b c : ℝ) : ℝ :=
  -(b / (2 * a))

theorem parabola_vertex_coordinates (a b c : ℝ)
  (h₀ : (2:ℝ) * 2 * a + 2 * b + c = 8)
  (h₁ : (4:ℝ) * 4 * a + 4 * b + c = 8)
  (h₂ : c = 3) :
  parabola_vertex_x a b c = 3 :=
sorry

end parabola_vertex_coordinates_l192_192023


namespace max_log_xy_l192_192713

theorem max_log_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 8) :
  ∃ u : ℝ, (u = log x + log y) ∧ (u ≤ 4 * log 2) := sorry

end max_log_xy_l192_192713


namespace tan_theta_expression_l192_192417

theorem tan_theta_expression (θ x : ℝ) (h_cos : cos (θ / 3) = sqrt ((x + 2) / (3 * x))) :
  tan θ = (sqrt (1 - ((4 * ((x + 2) ^ (3/2)) - 3 * sqrt (3 * x) * sqrt (x + 2)) / (3 * sqrt (3 * x ^ 3))) ^ 2)) /
         ((4 * ((x + 2) ^ (3/2)) - 3 * sqrt (3 * x) * sqrt (x + 2)) / (3 * sqrt (3 * x ^ 3))) := 
by
  sorry

end tan_theta_expression_l192_192417


namespace students_drawn_from_grade10_l192_192099

-- Define the initial conditions
def total_students_grade12 : ℕ := 750
def total_students_grade11 : ℕ := 850
def total_students_grade10 : ℕ := 900
def sample_size : ℕ := 50

-- Prove the number of students drawn from grade 10 is 18
theorem students_drawn_from_grade10 : 
  total_students_grade12 = 750 ∧
  total_students_grade11 = 850 ∧
  total_students_grade10 = 900 ∧
  sample_size = 50 →
  (sample_size * total_students_grade10 / 
  (total_students_grade12 + total_students_grade11 + total_students_grade10) = 18) :=
by
  sorry

end students_drawn_from_grade10_l192_192099


namespace positive_divisors_of_8_factorial_l192_192823

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192823


namespace compute_expression_l192_192544

theorem compute_expression : 1013^2 - 987^2 - 1007^2 + 993^2 = 24000 := by
  sorry

end compute_expression_l192_192544


namespace proj_u_equals_six_neg_three_twelve_l192_192415

section ProjectionProblem

variables (v w u : ℝ^3)
variables (proj_w_v : ℝ^3 := ⟨2, -1, 4⟩)
variables (u_eq_3v : u = 3 • v)

theorem proj_u_equals_six_neg_three_twelve 
  (proj_eq : ∃ w : ℝ^3, ∀ v : ℝ^3, proj_w_v = proj w v) :
  ∃ w : ℝ^3, ∀ u : ℝ^3, proj w u = ⟨6, -3, 12⟩ :=
by
  intro w h
  have hu : u = 3 • v := by assumption
  sorry

end proj_u_equals_six_neg_three_twelve_l192_192415


namespace encryption_of_hope_is_correct_l192_192159

def shift_letter (c : Char) : Char :=
  if 'a' ≤ c ∧ c ≤ 'z' then
    Char.ofNat ((c.toNat - 'a'.toNat + 4) % 26 + 'a'.toNat)
  else 
    c

def encrypt (s : String) : String :=
  s.map shift_letter

theorem encryption_of_hope_is_correct : encrypt "hope" = "lsti" :=
by
  sorry

end encryption_of_hope_is_correct_l192_192159


namespace no_strictly_increasing_sequence_with_finite_primes_in_shifted_sequences_l192_192157

theorem no_strictly_increasing_sequence_with_finite_primes_in_shifted_sequences :
  ¬ ∃ (a : ℕ → ℕ), (∀ n m, n < m → a n < a m) ∧ (∀ c : ℤ, ∀ᶠ n in at_top, ¬ (c + a n : ℤ).prime) :=
sorry

end no_strictly_increasing_sequence_with_finite_primes_in_shifted_sequences_l192_192157


namespace problem_solution_l192_192378

noncomputable def curve_parametric (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * Real.cos α, Real.sin α)

noncomputable def line_polar (θ : ℝ) : ℝ :=
  2 * sqrt 2 / Real.cos (θ - Real.pi / 4)

lemma Q_coordinates {α : ℝ} :
  ∃ Qx Qy, 
    (Qx, Qy) = curve_parametric α ∧
    Qx = -3 / 2 ∧ Qy = -1 / 2 :=
sorry

theorem problem_solution :
  (∀ θ, by let ρ := line_polar θ; ∃ x y, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ x + y = 4) ∧
  (∃ α : ℝ, 
    let (Qx, Qy) := curve_parametric α in
    let d := abs (sqrt 3 * Real.cos α + Real.sin α - 4) / sqrt 2 in
    d = 3 * sqrt 2 ∧
    Q_coordinates) :=
sorry

end problem_solution_l192_192378


namespace robbery_participants_l192_192606

variables (A B V G : Prop)

-- Conditions
axiom cond1 : ¬G → (B ∧ ¬A)
axiom cond2 : V → ¬A ∧ ¬B
axiom cond3 : G → B
axiom cond4 : B → (A ∨ V)

-- Theorem to be proved
theorem robbery_participants : A ∧ B ∧ G :=
by 
  sorry

end robbery_participants_l192_192606


namespace range_of_x_in_function_sqrt_x_minus_2_l192_192496

theorem range_of_x_in_function_sqrt_x_minus_2 (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 2)) → x ≥ 2 :=
by
  sorry

end range_of_x_in_function_sqrt_x_minus_2_l192_192496


namespace naomi_total_time_l192_192438

-- Definitions
def time_to_parlor : ℕ := 60
def speed_ratio : ℕ := 2 -- because her returning speed is half of the going speed
def first_trip_delay : ℕ := 15
def coffee_break : ℕ := 10
def second_trip_delay : ℕ := 20
def detour_time : ℕ := 30

-- Calculate total round trip times
def first_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + first_trip_delay + coffee_break
def second_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + second_trip_delay + detour_time

-- Hypothesis
def total_round_trip_time : ℕ := first_round_trip_time + second_round_trip_time

-- Main theorem statement
theorem naomi_total_time : total_round_trip_time = 435 := by
  sorry

end naomi_total_time_l192_192438


namespace sqrt_floor_8_integer_count_l192_192322

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192322


namespace prob_divisors_8_fact_l192_192809

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192809


namespace moscow_probability_higher_l192_192555

def total_combinations : ℕ := 64 * 63

def invalid_combinations_ural : ℕ := 8 * 7 + 8 * 7

def valid_combinations_moscow : ℕ := total_combinations

def valid_combinations_ural : ℕ := total_combinations - invalid_combinations_ural

def probability_moscow : ℚ := valid_combinations_moscow / total_combinations

def probability_ural : ℚ := valid_combinations_ural / total_combinations

theorem moscow_probability_higher :
  probability_moscow > probability_ural :=
by
  unfold probability_moscow probability_ural
  unfold valid_combinations_moscow valid_combinations_ural invalid_combinations_ural total_combinations
  sorry

end moscow_probability_higher_l192_192555


namespace find_c_squared_ab_l192_192489

theorem find_c_squared_ab (a b c : ℝ) (h1 : a^2 * (b + c) = 2008) (h2 : b^2 * (a + c) = 2008) (h3 : a ≠ b) : 
  c^2 * (a + b) = 2008 :=
sorry

end find_c_squared_ab_l192_192489


namespace positive_divisors_8_factorial_l192_192848

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192848


namespace positive_divisors_of_8_factorial_l192_192827

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192827


namespace concyclic_B1_to_B5_l192_192370

variable {Point : Type}
variable (A1 A2 A3 A4 A5 C1 C2 C3 C4 C5 B1 B2 B3 B4 B5 : Point)

-- Assuming necessary geometric axioms and definitions
axiom pentagram_intersections :
  ∀ (A1 A2 A3 A4 A5 C1 C2 C3 C4 C5 B1 B2 B3 B4 B5 : Point),
  let triangles := [(A1, C1, C2), (A2, C2, C3), (A3, C3, C4), (A4, C4, C5), (A5, C5, C1)] in
  let circumcircles := triangles.map (λ t, circumcircle t) in
  ∃ (B1 B2 B3 B4 B5 : Point), 
  ∀ i, B_i ∈ circumcircle (triangles[i]) ∧ B_i ≠ pentagon_vertex_i

def are_concyclic (B1 B2 B3 B4 B5 : Point) : Prop :=
  ∃ (c : Circle), ∀ B ∈ {B1, B2, B3, B4, B5}, B ∈ c

theorem concyclic_B1_to_B5 :
  pentagram_intersections A1 A2 A3 A4 A5 C1 C2 C3 C4 C5 B1 B2 B3 B4 B5 →
  are_concyclic B1 B2 B3 B4 B5 :=
by
  sorry

end concyclic_B1_to_B5_l192_192370


namespace number_of_integer_values_l192_192333

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192333


namespace isosceles_triangle_count_l192_192748

theorem isosceles_triangle_count : 
  ∃ (count : ℕ), count = 6 ∧ 
  ∀ (a b c : ℕ), a + b + c = 25 → 
  (a = b ∨ a = c ∨ b = c) → 
  a ≠ b ∨ c ≠ b ∨ a ≠ c → 
  ∃ (x y z : ℕ), x = a ∧ y = b ∧ z = c := 
sorry

end isosceles_triangle_count_l192_192748


namespace solution_set_circle_l192_192683

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l192_192683


namespace sum_of_x_values_l192_192665

-- Define the equation as a condition
def equation (x y : ℤ) : Prop :=
  7 * x * y - 13 * x + 15 * y - 37 = 0

-- Define the main theorem to prove
theorem sum_of_x_values : (∑ (x, y) in {(-2, 11), (-1, 3), (7, 2)}, x) = 4 :=
by
  sorry

end sum_of_x_values_l192_192665


namespace arithmetic_sequence_difference_l192_192621

theorem arithmetic_sequence_difference (a : ℕ → ℕ) (d : ℕ) (n : ℕ)
  (h : ∀ (i : ℕ), a i = (a 0 + i * d ∧ (a i ≠ 0 → ∃ k, a i = 2 * (10 + k)) ∧ (∀ m, (a (2 * m) ⊕ a (2 * m + 1)) ⊕ a (2 * m + 2) = a (2 * m) ⊕ (a (2 * m + 1) + a (2 * m + 2)) ) )
  (h_sum_odd : ∑ i in range n, if odd i then a i else 0 = 100) :
  99 * ∑ i in range n, if odd i then a i else 0 = 9900 := by
  sorry

end arithmetic_sequence_difference_l192_192621


namespace num_divisors_8_fact_l192_192851

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192851


namespace find_r_l192_192506

noncomputable def a : ℝ^3 := ![3, 1, -2]
noncomputable def b : ℝ^3 := ![0, 2, -1]
noncomputable def c : ℝ^3 := ![4, 1, -4]
noncomputable def cross_ab : ℝ^3 := ![3, 3, 6]

theorem find_r (p q r : ℝ) :
  c = p • a + q • b + r • cross_ab →
  r = -1/6 :=
by
  sorry

end find_r_l192_192506


namespace num_divisors_8_factorial_l192_192756

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192756


namespace A_can_finish_alone_in_28_days_l192_192563

theorem A_can_finish_alone_in_28_days
  (W : ℝ)  -- total work
  (A B : ℝ)  -- A's and B's work rate
  (h1 : A + B = W / 40)  -- combined work rate of A and B
  (h2 : 21 * A = 3 * W / 4)  -- A's work for 21 days is the remaining 3/4 work
  : (A = W / 28) :=  -- A's work rate in days to finish the whole work alone
begin
  sorry -- Proof to be filled in
end

end A_can_finish_alone_in_28_days_l192_192563


namespace diophantine_infinite_solutions_l192_192087

theorem diophantine_infinite_solutions :
  ∃ (a b c x y : ℤ), (a + b + c = x + y) ∧ (a^3 + b^3 + c^3 = x^3 + y^3) ∧ 
  ∃ (d : ℤ), (a = b - d) ∧ (c = b + d) :=
sorry

end diophantine_infinite_solutions_l192_192087


namespace num_divisors_fact8_l192_192955

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192955


namespace solve_for_y_l192_192461

def solution (y : ℝ) : Prop :=
  2 * Real.arctan (1/3) - Real.arctan (1/5) + Real.arctan (1/y) = Real.pi / 4

theorem solve_for_y (y : ℝ) : solution y → y = 31 / 9 :=
by
  intro h
  sorry

end solve_for_y_l192_192461


namespace positive_divisors_of_8_factorial_l192_192824

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192824


namespace sum_of_possible_h_values_l192_192144

theorem sum_of_possible_h_values :
  (∃ Q b c d e f g h : ℕ, 24 * b * c = Q ∧ d * e * f = Q ∧ g * h * 4 = Q ∧
   24 * e * 4 = Q ∧ b * e * h = Q ∧ c * e * 4 = Q ∧ 24 * e * g = Q ∧
   c * f * 4 = Q ∧ ∀ e : ℕ, e ≥ 1 → h = 6 * e) →
   ∑ (e in {1, 2, 3}.filter (λ e, 6 * e ≤ Q / 24)), 6 * e = 108 :=
begin
  sorry
end

end sum_of_possible_h_values_l192_192144


namespace positive_divisors_8_factorial_l192_192842

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192842


namespace num_divisors_fact8_l192_192952

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192952


namespace max_selection_of_children_can_be_arranged_in_a_circle_with_restriction_l192_192510

noncomputable def max_children (n : ℕ) : ℕ := 18

theorem max_selection_of_children_can_be_arranged_in_a_circle_with_restriction :
  ∃ (S : finset ℕ), S.card = 18 ∧
  ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → a * b < 100 :=
sorry

end max_selection_of_children_can_be_arranged_in_a_circle_with_restriction_l192_192510


namespace robbery_proof_l192_192615

variables (A B V G : Prop)

-- Define the conditions as Lean propositions
def condition1 : Prop := ¬G → (B ∧ ¬A)
def condition2 : Prop := V → (¬A ∧ ¬B)
def condition3 : Prop := G → B
def condition4 : Prop := B → (A ∨ V)

-- The statement we want to prove based on conditions
theorem robbery_proof (h1 : condition1 A B G) 
                      (h2 : condition2 A B V) 
                      (h3 : condition3 B G) 
                      (h4 : condition4 A B V) : 
                      A ∧ B ∧ G :=
begin
  sorry
end

end robbery_proof_l192_192615


namespace num_pos_divisors_fact8_l192_192776

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192776


namespace isosceles_triangle_count_l192_192749

theorem isosceles_triangle_count : 
  ∃ (count : ℕ), count = 6 ∧ 
  ∀ (a b c : ℕ), a + b + c = 25 → 
  (a = b ∨ a = c ∨ b = c) → 
  a ≠ b ∨ c ≠ b ∨ a ≠ c → 
  ∃ (x y z : ℕ), x = a ∧ y = b ∧ z = c := 
sorry

end isosceles_triangle_count_l192_192749


namespace eccentricity_of_ellipse_l192_192486

-- Define the problem conditions and restate the problem in Lean.

-- Ellipse definition and conditions
variables {a b c e : ℝ} (h₁ : a > b) (h₂ : b > 0) (h₃ : b^2 = a^2 - c^2)

-- Symmetric point A of F with respect to the given line
variables {m n : ℝ} (h₄ : (n / (m + c)) * (-sqrt 3) = -1)
variables :noinspection   (h₅ : sqrt 3 * ((m - c) / 2) + n / 2 = 0)
(h₆ : m = c / 2) (h₇ : n = sqrt 3 / 2 * c)

-- Point A on the ellipse
variables (h₈ : (m^2 / a^2) + (n^2 / b^2) = 1)

-- Prove the eccentricity is sqrt 3 - 1
theorem eccentricity_of_ellipse :
  e = sqrt 3 - 1 := sorry

end eccentricity_of_ellipse_l192_192486


namespace limit_of_function_l192_192633

noncomputable def limit_expr (x : ℝ) : ℝ := (1 - real.sqrt (real.cos x)) / (x * real.sin x)
noncomputable def L : ℝ := 1/4

theorem limit_of_function : 
  filter.tendsto limit_expr (nhds 0) (nhds L) :=
sorry

end limit_of_function_l192_192633


namespace find_sixth_number_l192_192015

variable (nums : Fin 11 → ℕ)

def avg_11 := 60
def avg_first_6 := 98
def avg_last_6 := 65
def sum_11 := 660
def sum_first_6 := 588
def sum_last_6 := 390

theorem find_sixth_number (H1 : (Array.sum (Array.mk nums) / 11 : ℕ) = avg_11)
 (H2 : (Array.sum (Array.mk (nums ∘ Fin.castSucc) ∘ Fin.cast (Fin.castSucc 6)) / 6 : ℕ) = avg_first_6)
 (H3 : (Array.sum (Array.mk (nums ∘ Fin.castSucc) ∘ Fin.cast (Fin.castSucc (6:ℕ)) ∘ Fin.shift 5) / 6 : ℕ) = avg_last_6)
 (H4 : Array.sum (Array.mk nums) = sum_11)
 (H5 : Array.sum (Array.mk (nums ∘ Fin.castSucc) ∘ Fin.cast (Fin.castSucc 6)) = sum_first_6)
 (H6 : Array.sum (Array.mk (nums ∘ Fin.castSucc) ∘ Fin.cast (Fin.castSucc (6: ℕ)) ∘ Fin.shift 5) = sum_last_6) :
 nums 5 = 318 := by
sorry

end find_sixth_number_l192_192015


namespace repeating_decimal_sum_proof_l192_192662

noncomputable def repeating_decimal_sum : Prop :=
  (0.33333333... : ℚ) + (0.22222222... : ℚ) = (5 / 9 : ℚ)

theorem repeating_decimal_sum_proof : repeating_decimal_sum :=
by
  sorry

end repeating_decimal_sum_proof_l192_192662


namespace prob_divisors_8_fact_l192_192804

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192804


namespace solve_log_eq_l192_192463

theorem solve_log_eq {x : ℝ} (hx : x > 0) : 
  2 * log x = log 192 + log 3 - log 4 → x = 12 :=
by
  sorry

end solve_log_eq_l192_192463


namespace num_positive_divisors_8_factorial_l192_192937

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192937


namespace min_max_f_l192_192188

noncomputable def f (x : ℝ) : ℝ := 4^(x - 1 / 2) - 3 * 2^x + 5

theorem min_max_f : 
  ∃ (m M : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → m ≤ f x ∧ f x ≤ M) ∧ 
               (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → m = 1 / 2 ∧ M = 5 / 2) :=
by 
  let f := λ (x : ℝ), 4^(x - 1 / 2) - 3 * 2^x + 5
  use (1 / 2, 5 / 2)
  sorry

end min_max_f_l192_192188


namespace solution_set_circle_l192_192682

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l192_192682


namespace sum_of_x_values_l192_192666

def eqn (x y : ℤ) : Prop := 7 * x * y - 13 * x + 15 * y - 37 = 0

theorem sum_of_x_values : 
  (∑ x in {x : ℤ | ∃ y : ℤ, eqn x y}, x) = 4 :=
sorry

end sum_of_x_values_l192_192666


namespace not_always_possible_to_mark_exactly_one_point_per_unit_circle_l192_192511

theorem not_always_possible_to_mark_exactly_one_point_per_unit_circle 
  (circles : set (metric.sphere ℝ 1)) : 
  ¬ ∀ (points : set ℝ × ℝ),
  (∀ c ∈ circles, ∃! p ∈ points, metric.ball c.1 1 p.1 p.2) :=
sorry

end not_always_possible_to_mark_exactly_one_point_per_unit_circle_l192_192511


namespace vector_magnitude_parallel_l192_192241

def vector_parallel : ℝ × ℝ → ℝ × ℝ → Prop
| (a₁, b₁), (a₂, b₂) := a₁ * b₂ = b₁ * a₂

theorem vector_magnitude_parallel 
  (p q : ℝ × ℝ) 
  (hp : p = (2, -3)) 
  (hq : ∃ x, q = (x, 6) ∧ vector_parallel p q) 
  : |p.1 + q.1, p.2 + q.2| = real.sqrt 13 :=
by
  sorry

end vector_magnitude_parallel_l192_192241


namespace find_length_of_brick_l192_192179

-- Definitions given in the problem
def w : ℕ := 4
def h : ℕ := 2
def SA : ℕ := 112
def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

-- Lean 4 statement for the proof problem
theorem find_length_of_brick (l : ℕ) (h w SA : ℕ) (h_w : w = 4) (h_h : h = 2) (h_SA : SA = 112) :
  surface_area l w h = SA → l = 8 := by
  intros H
  simp [surface_area, h_w, h_h, h_SA] at H
  sorry

end find_length_of_brick_l192_192179


namespace inequality_am_gm_l192_192217

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (y * z) + y / (z * x) + z / (x * y)) ≥ (1 / x + 1 / y + 1 / z) := 
by
  sorry

end inequality_am_gm_l192_192217


namespace number_of_elements_ge_m_squared_exists_set_with_partitions_divisible_by_3_l192_192221

variable {X : Type} (A B C : Finset (Finset X)) (m n : ℕ)

-- Condition 1: The sets A_1, A_2, ..., A_m; B_1, B_2, ..., B_m; C_1, C_2, ..., C_m are partitions of X.
-- For every group of i, j, k, it holds that:
axiom condition (i j k : ℕ) (h₁ : i < m) (h₂ : j < m) (h₃ : k < m) :
  (A i ∩ B j).card + (A i ∩ C k).card + (B j ∩ C k).card ≥ m

-- Question 1: Prove that the number of elements in X, n, is at least m^2.
theorem number_of_elements_ge_m_squared :
  n ≥ m^2 :=
  sorry

-- Question 2: When m is divisible by 3, show that there exists a set X with n = m^3 / 3 elements with three partitions satisfying the given condition.
theorem exists_set_with_partitions_divisible_by_3 (h : m % 3 = 0) :
  ∃ (X : Finset X), X.card = m^3 / 3 ∧ (∀ i j k, i < m → j < m → k < m → 
  (A i ∩ B j).card + (A i ∩ C k).card + (B j ∩ C k).card ≥ m) :=
  sorry

end number_of_elements_ge_m_squared_exists_set_with_partitions_divisible_by_3_l192_192221


namespace num_possible_integer_values_x_l192_192342

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192342


namespace tan_alpha_equation_l192_192698

-- Define the given condition
def condition (α : ℝ) := 
  sin (2 * α + π / 4) - 7 * sin (2 * α + 3 * π / 4) = 5 * √2

-- State the theorem
theorem tan_alpha_equation (α : ℝ) (h : condition α) : 
  tan α = 2 :=
sorry

end tan_alpha_equation_l192_192698


namespace erick_total_revenue_l192_192059

def lemon_price_increase := 4
def grape_price_increase := lemon_price_increase / 2
def original_lemon_price := 8
def original_grape_price := 7
def lemons_sold := 80
def grapes_sold := 140

def new_lemon_price := original_lemon_price + lemon_price_increase -- $12 per lemon
def new_grape_price := original_grape_price + grape_price_increase -- $9 per grape

def revenue_from_lemons := lemons_sold * new_lemon_price -- $960
def revenue_from_grapes := grapes_sold * new_grape_price -- $1260

def total_revenue := revenue_from_lemons + revenue_from_grapes

theorem erick_total_revenue : total_revenue = 2220 := by
  -- Skipping proof with sorry
  sorry

end erick_total_revenue_l192_192059


namespace valid_permutations_count_l192_192253

/-- Adjacent letter pairs in the alphabet for the given problem -/
def adjacent_pairs : List (Char × Char) :=
  [ ('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e') ]

/-- Insert your configurations -/
def is_valid_permutation (l : List Char) : Prop :=
  (l ~ ['a', 'b', 'c', 'd', 'e']) ∧ ∀ (i : Nat), i < l.length - 1 → 
  ¬((l.nth i).getD ' ' , (l.nth (i + 1)).getD ' ') ∈ adjacent_pairs

theorem valid_permutations_count : 
  (List.filter is_valid_permutation (List.permutations ['a', 'b', 'c', 'd', 'e'])).length = 6 :=
sorry

end valid_permutations_count_l192_192253


namespace expected_number_of_adjacent_black_pairs_l192_192473

theorem expected_number_of_adjacent_black_pairs :
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_per_black_card := black_cards * adjacent_probability / total_cards
  let expected_total := black_cards * adjacent_probability
  expected_total = 650 / 51 := 
by
  let total_cards := 52
  let black_cards := 26
  let adjacent_probability := (black_cards - 1) / (total_cards - 1)
  let expected_total := black_cards * adjacent_probability
  sorry

end expected_number_of_adjacent_black_pairs_l192_192473


namespace solution_set_is_circle_with_exclusion_l192_192674

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l192_192674


namespace count_possible_integer_values_l192_192287

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192287


namespace number_of_integer_values_l192_192330

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192330


namespace exists_distinct_i_j_l192_192411

def S (x : ℕ) : ℕ :=
  -- S(x) is the sum of the 2013-th powers of the digits of x
  (x.digits 10).sum (λ d, d ^ 2013)

def a : ℕ → ℕ
| 0     := 2013
| (n+1) := S (a n)

theorem exists_distinct_i_j : ∃ (i j : ℕ), i ≠ j ∧ a i = a j :=
sorry -- proof to be filled in

end exists_distinct_i_j_l192_192411


namespace solve_equation_l192_192500

theorem solve_equation (x : ℝ) : x^2 = 5 * x → x = 0 ∨ x = 5 := 
by
  sorry

end solve_equation_l192_192500


namespace count_integer_values_l192_192311

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192311


namespace mul_exponent_result_l192_192053

theorem mul_exponent_result : 112 * (5^4) = 70000 := 
by 
  sorry

end mul_exponent_result_l192_192053


namespace train_speed_l192_192598

theorem train_speed :
  (length : ℝ) (time : ℝ) (speed : ℝ)
  (h_length : length = 250)
  (h_time : time = 18) :
  speed = (length / (time / 3600)) / 1000 := 
by
  rw [h_length, h_time]
  -- intermediate steps to simplify the statement can be added here
  sorry

end train_speed_l192_192598


namespace minimum_value_f_l192_192027

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / 2 + 2 / (Real.sin x)

theorem minimum_value_f (x : ℝ) (h : 0 < x ∧ x ≤ Real.pi / 2) :
  ∃ y, (∀ z, 0 < z ∧ z ≤ Real.pi / 2 → f z ≥ y) ∧ y = 5 / 2 :=
sorry

end minimum_value_f_l192_192027


namespace sqrt_floor_8_integer_count_l192_192320

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192320


namespace propositions_A_and_D_are_correct_l192_192066

theorem propositions_A_and_D_are_correct :
  (∀ (a b : ℝ), (-2 < a ∧ a < 3) → (1 < b ∧ b < 2) → -4 < a - b ∧ a - b < 2) ∧
  ¬ (∀ x : ℝ, ∃ y : ℝ, y = sqrt (x ^ 2 + 2) + 1 / sqrt (x ^ 2 + 2) ∧ y = 2) ∧
  ¬ (∀ a b : ℝ, a > b → 1 / a < 1 / b) ∧
  (∃ a : ℝ, a + 1 / a ≤ 2) :=
by sorry

end propositions_A_and_D_are_correct_l192_192066


namespace max_value_of_M_for_cone_bottomed_f_l192_192720

-- Definitions from the conditions
def is_cone_bottomed (f : ℝ → ℝ) (M : ℝ) : Prop :=
  ∀ x : ℝ, |f x| ≥ M * |x|

-- Function given in the problem
noncomputable
def f (x : ℝ) : ℝ := x^2 + 1

-- The theorem stating the mathematically equivalent proof problem
theorem max_value_of_M_for_cone_bottomed_f : ∃ M : ℝ, is_cone_bottomed f M ∧ (∀ N : ℝ, is_cone_bottomed f N → N ≤ 2) :=
begin
  sorry,
end

end max_value_of_M_for_cone_bottomed_f_l192_192720


namespace num_divisors_of_8_factorial_l192_192870

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192870


namespace weighted_average_score_l192_192151

theorem weighted_average_score :
  let english := 76
  let math := 65
  let physics := 82
  let chemistry := 67
  let biology := 85
  let ratio_english := 2
  let ratio_math := 3
  let ratio_science := 3
  let weighted_sum := (english * ratio_english) + (math * ratio_math) + (physics + chemistry + biology)
  let total_weight := ratio_english + ratio_math + ratio_science in
  weighted_sum / total_weight = 72.625 :=
sorry

end weighted_average_score_l192_192151


namespace correct_option_is_D_l192_192532

noncomputable def option_A := 230
noncomputable def option_B := [251, 260]
noncomputable def option_B_average := 256
noncomputable def option_C := [21, 212, 256]
noncomputable def option_C_average := 163
noncomputable def option_D := [210, 240, 250]
noncomputable def option_D_average := 233

theorem correct_option_is_D :
  ∃ (correct_option : String), correct_option = "D" :=
  sorry

end correct_option_is_D_l192_192532


namespace num_divisors_fact8_l192_192956

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192956


namespace total_surface_area_of_cylinder_l192_192584

theorem total_surface_area_of_cylinder 
  (r h : ℝ) 
  (hr : r = 3) 
  (hh : h = 8) : 
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 66 * Real.pi := by
  sorry

end total_surface_area_of_cylinder_l192_192584


namespace three_digit_natural_numbers_perfect_square_count_l192_192254

theorem three_digit_natural_numbers_perfect_square_count :
  {n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n - 1 = k^2}.to_finset.size = 22 := 
sorry

end three_digit_natural_numbers_perfect_square_count_l192_192254


namespace lines_in_same_plane_l192_192202

variables {P : Type} [InnerProductSpace ℝ P]

-- Define the three rays originating from point P
variables (a b c : P) 

-- Conditions
-- Angles between any two rays are acute
variable (h_ab : ⟪a, b⟫ > 0)
variable (h_bc : ⟪b, c⟫ > 0)
variable (h_ac : ⟪a, c⟫ > 0)

-- Define the lines a_0, b_0, c_0 that are perpendicular to the respective planes
-- Note: These are idealizations in this context as we assume perpendicular projections
variables (a_0 b_0 c_0 : P)

-- Each line is perpendicular to the corresponding plane formed by the other two rays
variable (h_a0 : ⟪a_0, b⟫ = 0 ∧ ⟪a_0, c⟫ = 0)
variable (h_b0 : ⟪b_0, a⟫ = 0 ∧ ⟪b_0, c⟫ = 0)
variable (h_c0 : ⟪c_0, a⟫ = 0 ∧ ⟪c_0, b⟫ = 0)

-- We need to show that a_0, b_0, and c_0 lie in the same plane
theorem lines_in_same_plane :
  ∃ (plane : Subspace ℝ P), a_0 ∈ plane ∧ b_0 ∈ plane ∧ c_0 ∈ plane :=
sorry

end lines_in_same_plane_l192_192202


namespace system_of_equations_soln_l192_192680

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l192_192680


namespace find_a_plus_b_l192_192215

-- Given conditions
variable (a b : ℝ)

-- The imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Condition equation
def equation := (a + i) * i = b - 2 * i

-- Define the lean statement
theorem find_a_plus_b (h : equation a b) : a + b = -3 :=
by sorry

end find_a_plus_b_l192_192215


namespace count_integer_values_l192_192308

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192308


namespace ratio_of_flour_to_eggs_l192_192645

theorem ratio_of_flour_to_eggs (F E : ℕ) (h1 : E = 60) (h2 : F + E = 90) : F / 30 = 1 ∧ E / 30 = 2 := by
  sorry

end ratio_of_flour_to_eggs_l192_192645


namespace number_of_isosceles_triangles_with_perimeter_25_l192_192747

def is_isosceles_triangle (a b : ℕ) : Prop :=
  a + 2 * b = 25 ∧ 2 * b > a ∧ a < 2 * b

theorem number_of_isosceles_triangles_with_perimeter_25 :
  (finset.filter (λ b, ∃ a, is_isosceles_triangle a b)
                 (finset.range 13)).card = 6 := by
sorry

end number_of_isosceles_triangles_with_perimeter_25_l192_192747


namespace derivative_quotient_eq_quotient_derivatives_l192_192522

noncomputable def f (x : ℝ) := Real.exp (4 * x)
noncomputable def g (x : ℝ) := Real.exp (2 * x)

theorem derivative_quotient_eq_quotient_derivatives (x : ℝ) :
  ∃ f g : ℝ → ℝ, 
  Differentiable ℝ f ∧ Differentiable ℝ g ∧ (deriv g x ≠ 0) ∧ 
  ( ∀ x, deriv (λ x, (f x) / (g x)) x = (deriv f x) / (deriv g x) ) := 
by
  use [f, g]
  split
  · exact differentiable_exp.comp (differentiable_id.smul_const 4)
  split
  · exact differentiable_exp.comp (differentiable_id.smul_const 2)
  split
  · intros x
    apply ne_of_gt
    apply exp_pos
  sorry

end derivative_quotient_eq_quotient_derivatives_l192_192522


namespace number_of_integer_values_l192_192325

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192325


namespace isosceles_triangles_with_perimeter_25_l192_192743

/-- Prove that there are 6 distinct isosceles triangles with integer side lengths 
and a perimeter of 25 -/
theorem isosceles_triangles_with_perimeter_25 :
  ∃ (count : ℕ), 
    count = 6 ∧ 
    (∀ (a b : ℕ), 
      let a1 := a,
          a2 := a,
          b3 := b in
      2 * a + b = 25 → 
      2 * a > b ∧ a + b > a ∧ b < 2 * a ∧
      a > 0 ∧ b > 0 ∧ a ∈ finset.Icc 7 12) :=
by sorry

end isosceles_triangles_with_perimeter_25_l192_192743


namespace solve_sqrt_eq_13_l192_192168

theorem solve_sqrt_eq_13 {z : ℝ} (h : sqrt (10 + 3 * z) = 13) : z = 53 := 
sorry

end solve_sqrt_eq_13_l192_192168


namespace average_speed_approx_l192_192971

-- Define the constants and conditions of the problem:
def outward_speed := 110
def tailwind := 15
def return_speed := 72
def headwind := 10

def effective_outward_speed := outward_speed + tailwind -- Speed on outward journey
def effective_return_speed := return_speed - headwind -- Speed on return journey

theorem average_speed_approx : 
  let D := (1 : ℝ) in -- Assume the distance 'D' is 1 mile for simplicity
  let total_distance := 2 * D in
  let outward_time := D / effective_outward_speed in
  let return_time := D / effective_return_speed in
  let total_time := outward_time + return_time in
  let avg_speed := total_distance / total_time in
  avg_speed ≈ 82.89 := 
by {
  let D := (1 : ℝ) -- Assume the distance 'D' is 1 mile for simplicity
  let total_distance := 2 * D
  let outward_time := D / effective_outward_speed
  let return_time := D / effective_return_speed
  let total_time := outward_time + return_time
  let avg_speed := total_distance / total_time
  -- Use Lean's 'approx' for floating point comparison
  exact (approx avg_speed 82.89 sorry) -- sorry is used to skip the actual proof.
}

end average_speed_approx_l192_192971


namespace number_of_divisors_of_8_fact_l192_192885

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192885


namespace number_of_divisors_8_factorial_l192_192797

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192797


namespace count_valid_7_digit_numbers_l192_192180

def is_strictly_increasing (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i < l.nth j

def is_strictly_decreasing (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i > l.nth j

def valid_7_digit_number (n : ℕ) : Prop :=
  let digits := (n / 10^6 % 10), (n / 10^5 % 10), (n / 10^4 % 10), (n / 10^3 % 10), (n / 10^2 % 10), (n / 10^1 % 10), (n % 10) in
  n > 999999 ∧ n < 10000000 ∧ 
  digits.length = 7 ∧
  (∃ m ∈ Set.ofList [4, 5, 6, 7, 8, 9], 
    let (incr_part, decr_part) := digits.splitAt 4 in
    is_strictly_increasing incr_part ∧ 
    is_strictly_decreasing decr_part ∧
    digits.nth 3 = m)

theorem count_valid_7_digit_numbers : 
  (Finset.filter valid_7_digit_number (Finset.range 10000000)).card = 7608 :=
sorry

end count_valid_7_digit_numbers_l192_192180


namespace prob_divisors_8_fact_l192_192813

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192813


namespace num_divisors_fact8_l192_192959

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192959


namespace num_divisors_8_fact_l192_192850

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192850


namespace number_of_x_values_l192_192262

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192262


namespace range_of_m_for_decreasing_interval_l192_192727

def function_monotonically_decreasing_in_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a < x → x < y → y < b → f y ≤ f x

def f (x : ℝ) : ℝ := x ^ 3 - 12 * x

theorem range_of_m_for_decreasing_interval :
  ∀ m : ℝ, function_monotonically_decreasing_in_interval f (2 * m) (m + 1) → -1 ≤ m ∧ m < 1 :=
by
  sorry

end range_of_m_for_decreasing_interval_l192_192727


namespace mary_money_after_purchase_l192_192119

def mary_initial_money : ℕ := 58
def pie_cost : ℕ := 6
def mary_friend_money : ℕ := 43  -- This is an extraneous condition, included for completeness.

theorem mary_money_after_purchase : mary_initial_money - pie_cost = 52 := by
  sorry

end mary_money_after_purchase_l192_192119


namespace alcohol_solution_equivalence_l192_192252

noncomputable def liters_of_pure_alcohol_added : ℝ := 14.285714285714286

def initial_solution_volume : ℝ := 100
def initial_alcohol_volume : ℝ := initial_solution_volume * 0.2

def desired_percentage : ℝ := 0.3

theorem alcohol_solution_equivalence :
  let final_solution_volume := initial_solution_volume + liters_of_pure_alcohol_added;
      final_alcohol_volume := initial_alcohol_volume + liters_of_pure_alcohol_added;
      final_percentage := final_alcohol_volume / final_solution_volume
  in final_percentage ≈ desired_percentage :=
by
  sorry

end alcohol_solution_equivalence_l192_192252


namespace max_possible_hop_sum_l192_192413

theorem max_possible_hop_sum (n : ℕ) (hn : n > 0) : 
  ∃ (S : ℕ), 
    (∀ i, (0 < i) → (i < 2^n) → (¬ visited_before i)) ∧ 
    (∀ len, (len ∈ {2^i | i : ℕ}) → (S = sum_of_hops len)) ∧ 
    (S = (4^n - 1) / 3) :=
sorry

end max_possible_hop_sum_l192_192413


namespace count_whole_numbers_l192_192257

theorem count_whole_numbers (a b : ℝ) (ha : a = real.cbrt 20) (hb : b = real.cbrt 300) :
  ∃ n : ℕ, n = 4 ∧ 3 ≤ (⌈ a ⌉ : ℤ) ∧ (⌊ b ⌋ : ℤ) = 6 :=
by
  sorry

end count_whole_numbers_l192_192257


namespace optimal_strategy_l192_192129

def mild_winter : ℕ := 2200
def mild_liquefied : ℕ := 3500
def severe_winter : ℕ := 3800
def severe_liquefied : ℕ := 2450
def cost_natural_gas : ℕ := 19
def cost_liquefied_gas : ℕ := 25
def price_natural_gas : ℕ := 35
def price_liquefied_gas : ℕ := 58

noncomputable def optimal_natural_gas : ℕ := 3032
noncomputable def optimal_liquefied_gas : ℕ := 2954

theorem optimal_strategy : 
  proof 
    (mild_winter = 2200) 
    ∧ (mild_liquefied = 3500) 
    ∧ (severe_winter = 3800) 
    ∧ (severe_liquefied = 2450) 
    ∧ (cost_natural_gas = 19) 
    ∧ (cost_liquefied_gas = 25) 
    ∧ (price_natural_gas = 35) 
    ∧ (price_liquefied_gas = 58)
  implies ((optimal_natural_gas = 3032) 
    ∧ (optimal_liquefied_gas = 2954)) :=
sorry

end optimal_strategy_l192_192129


namespace gcd_of_2146_1813_horner_value_at_2_l192_192084

-- Define the problem of finding GCD
def gcd_problem (a b : ℕ) : ℕ :=
  let rec gcd_aux : ℕ → ℕ → ℕ
  | x, 0 => x
  | x, y => gcd_aux y (x % y)
  in gcd_aux a b

-- Define the polynomial evaluation using Horner's method
def horner_eval (x : ℤ) : ℤ :=
  let f := [2, 3, 2, 0, -4, 5]  -- Coefficients of the polynomial 2x^5 + 3x^4 + 2x^3 + 0x^2 - 4x + 5
  f.foldl (λ acc coeff => acc * x + coeff) 0

-- Lean 4 statements
theorem gcd_of_2146_1813 : gcd_problem 2146 1813 = 37 := by {
  sorry
}

theorem horner_value_at_2 : horner_eval 2 = 60 := by {
  sorry
}

end gcd_of_2146_1813_horner_value_at_2_l192_192084


namespace prob_divisors_8_fact_l192_192805

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192805


namespace num_positive_divisors_8_factorial_l192_192931

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192931


namespace cannot_be_simultaneous_squares_l192_192634

theorem cannot_be_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + y = a^2 ∧ y^2 + x = b^2) :=
by
  sorry

end cannot_be_simultaneous_squares_l192_192634


namespace analytical_expression_of_f_range_of_m_l192_192208

-- Definitions of the problem
def f (x : ℝ) : ℝ :=
  if h : x ∈ Icc (-4) 0 then 1 / (4^x) - 1 / (3^x)
  else - (1 / (4^(-x)) - 1 / (3^(-x)))

theorem analytical_expression_of_f :
  ∀ x, f(x) =
    if h : x ∈ Icc (-4) 0 then 1 / (4^x) - 1 / (3^x)
    else 3^x - 4^x :=
sorry

theorem range_of_m (m : ℝ) :
  (∃ x ∈ Icc (-2) (-1), f x ≤ m / (2^x) - 1 / (3^(x-1))) →
  5 ≤ m :=
sorry

end analytical_expression_of_f_range_of_m_l192_192208


namespace find_x_l192_192261

noncomputable def G (a b c d : ℕ) : ℝ := a^b + (c / d)

theorem find_x (x : ℝ) : G 3 x 48 8 = 310 ↔ x = (Real.log 304) / (Real.log 3) := by
  sorry

end find_x_l192_192261


namespace digit_sum_even_numbers_count_l192_192478

def is_even (n : ℕ) : Prop := n % 2 = 0

noncomputable def count_valid_numbers : ℕ :=
  (Finset.filter (λ n : ℕ, is_even n ∧ ∑ d in (Nat.digits 10 n), d = 26) 
     (Finset.range 1000 \ Finset.range 100)).card

theorem digit_sum_even_numbers_count :
  count_valid_numbers = 1 := by
  sorry

end digit_sum_even_numbers_count_l192_192478


namespace number_of_divisors_8_factorial_l192_192799

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192799


namespace positive_divisors_8_factorial_l192_192834

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192834


namespace train_arrival_probability_l192_192601

theorem train_arrival_probability :
  let in_range (t : ℕ) := t >= 0 ∧ t <= 120
  let train_at_station (train_time alex_time : ℕ) := alex_time >= train_time ∧ alex_time <= train_time + 20
  ∃ (probability : ℚ),
    probability = 11 / 72 ∧
    (∀ (train_time alex_time : ℕ),
      in_range train_time →
      in_range alex_time →
      train_at_station train_time alex_time →
      true
    ) :=
begin
  sorry
end

end train_arrival_probability_l192_192601


namespace steel_rod_length_l192_192100

-- Definitions for weights and lengths.
def weight_rod (length : ℝ) : ℝ := 22.8 / 6 * length

theorem steel_rod_length (h : weight_rod L = 42.75) : L = 11.25 :=
by
  sorry

end steel_rod_length_l192_192100


namespace num_divisors_fact8_l192_192949

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192949


namespace min_time_to_complete_distance_l192_192533

/-- Prove that the minimal time for both Vasya and Petya to complete a 3 km distance 
is 1/2 hour, given their running and skating speeds and the ability to exchange skates 
without losing time. -/
theorem min_time_to_complete_distance (D : ℝ) (v_run_V v_skate_V v_run_P v_skate_P : ℝ) 
  (hV_skate : v_run_V ≤ v_skate_V) (hP_skate : v_run_P ≤ v_skate_P) (h_distance : D = 3)
  (hV_speeds : v_run_V = 4 ∧ v_skate_V = 8) (hP_speeds : v_run_P = 5 ∧ v_skate_P = 10) :
  let t1 := (2/3) / v_skate_V + (3 - (2/3)) / v_run_V,
      t2 := (2/3) / v_skate_P + (3 - (2/3)) / v_run_P in
  min t1 t2 = 1 / 2 :=
by
  -- Placeholder for proof
  sorry

end min_time_to_complete_distance_l192_192533


namespace optimal_strategy_l192_192130

def mild_winter : ℕ := 2200
def mild_liquefied : ℕ := 3500
def severe_winter : ℕ := 3800
def severe_liquefied : ℕ := 2450
def cost_natural_gas : ℕ := 19
def cost_liquefied_gas : ℕ := 25
def price_natural_gas : ℕ := 35
def price_liquefied_gas : ℕ := 58

noncomputable def optimal_natural_gas : ℕ := 3032
noncomputable def optimal_liquefied_gas : ℕ := 2954

theorem optimal_strategy : 
  proof 
    (mild_winter = 2200) 
    ∧ (mild_liquefied = 3500) 
    ∧ (severe_winter = 3800) 
    ∧ (severe_liquefied = 2450) 
    ∧ (cost_natural_gas = 19) 
    ∧ (cost_liquefied_gas = 25) 
    ∧ (price_natural_gas = 35) 
    ∧ (price_liquefied_gas = 58)
  implies ((optimal_natural_gas = 3032) 
    ∧ (optimal_liquefied_gas = 2954)) :=
sorry

end optimal_strategy_l192_192130


namespace num_positive_divisors_8_factorial_l192_192935

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192935


namespace forest_area_relationship_l192_192480

variable (a b c x : ℝ)

theorem forest_area_relationship
    (hb : b = a * (1 + x))
    (hc : c = a * (1 + x) ^ 2) :
    a * c = b ^ 2 := by
  sorry

end forest_area_relationship_l192_192480


namespace sequence_general_formula_l192_192219

-- Define the sequence S_n and the initial conditions
def S (n : ℕ) : ℕ := 3^(n + 1) - 1

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 8 else 2 * 3^n

-- Theorem statement proving the general formula
theorem sequence_general_formula (n : ℕ) : 
  a n = if n = 1 then 8 else 2 * 3^n := by
  -- This is where the proof would go
  sorry

end sequence_general_formula_l192_192219


namespace number_of_divisors_8_factorial_l192_192787

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192787


namespace polynomial_roots_l192_192668

theorem polynomial_roots : ∃ (r : List ℤ), (Polynomial.eval₂ (λ (n : ℤ), n) (Polynomial.C (x^3 - 2 * x^2 - 5 * x + 6)) (↑r.head) = 0) ∧
  ∃ (r : List ℤ), r = [1, -2, 3] :=
begin
  sorry
end

end polynomial_roots_l192_192668


namespace num_positive_divisors_8_factorial_l192_192941

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192941


namespace num_divisors_fact8_l192_192950

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192950


namespace eighth_triangular_number_l192_192012

def triangular_number (n: ℕ) : ℕ := n * (n + 1) / 2

theorem eighth_triangular_number : triangular_number 8 = 36 :=
by
  -- Proof here
  sorry

end eighth_triangular_number_l192_192012


namespace num_divisors_8_factorial_l192_192917

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192917


namespace sin_sq_half_cos_le_eighth_l192_192456

theorem sin_sq_half_cos_le_eighth (x y : ℝ) (h : 0 ≤ x ∧ x ≤ y ∧ y ≤ π / 2) : 
  (sin (x / 2))^2 * cos y ≤ 1 / 8 :=
by
  sorry

end sin_sq_half_cos_le_eighth_l192_192456


namespace system_of_equations_soln_l192_192681

theorem system_of_equations_soln :
  {p : ℝ × ℝ | ∃ a : ℝ, (a * p.1 + p.2 = 2 * a + 3) ∧ (p.1 - a * p.2 = a + 4)} =
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 5} \ {⟨2, -1⟩} :=
by
  sorry

end system_of_equations_soln_l192_192681


namespace problem_statement_l192_192198

noncomputable def a : ℝ := 2 ^ 0.5
noncomputable def b : ℝ := Real.log 3 / Real.log π
noncomputable def c : ℝ := Real.log 0.5 / Real.log 2

theorem problem_statement : a > b ∧ b > c :=
by {
    -- Proof steps go here.
    sorry
}

end problem_statement_l192_192198


namespace true_count_p_and_q_l192_192343

variables (p q : Prop)

theorem true_count_p_and_q : ((p ∧ q) → (true_count [p ∨ q, p, ¬q, (¬p) ∨ (¬q)] = 2)) :=
by {
  sorry
}

-- Helper function to count true propositions
def true_count (props : List Prop) : Nat :=
  props.count (λ p, p = True)

end true_count_p_and_q_l192_192343


namespace erick_total_earnings_l192_192061

theorem erick_total_earnings
    (original_lemon_price : ℕ)
    (lemon_price_increase : ℕ)
    (original_grape_price : ℕ)
    (lemons_count : ℕ)
    (grapes_count : ℕ) :
    let new_lemon_price := original_lemon_price + lemon_price_increase,
        total_lemons_earning := lemons_count * new_lemon_price,
        grape_price_increase := lemon_price_increase / 2,
        new_grape_price := original_grape_price + grape_price_increase,
        total_grapes_earning := grapes_count * new_grape_price,
        total_earning := total_lemons_earning + total_grapes_earning
    in total_earning = 2220 := by
  sorry

end erick_total_earnings_l192_192061


namespace expression_value_l192_192139

theorem expression_value : 
  (2 ^ 1501 + 5 ^ 1502) ^ 2 - (2 ^ 1501 - 5 ^ 1502) ^ 2 = 20 * 10 ^ 1501 := 
by
  sorry

end expression_value_l192_192139


namespace sum_of_roots_eq_four_l192_192536

open Real

theorem sum_of_roots_eq_four : 
  ∀ (a b c : ℝ), a ≠ 0 ∧ a = 3 ∧ b = -12 ∧ c = 9 ∧ (3 * a * (x : ℝ)^2 - 12 * (x : ℝ) + 9 = 0) → 
  (let sum_of_roots := (-b / a) in sum_of_roots = 4) :=
by {
  intros a b c,
  intro h,
  have ha : a = 3 := h.2.1,
  have hb : b = -12 := h.2.2.1,
  rw [ha, hb],
  sorry
}

end sum_of_roots_eq_four_l192_192536


namespace pairs_condition_l192_192152

def div_by (d n : ℕ) (x : ℤ) : Prop := x % d = 0

theorem pairs_condition (a b : ℤ) :
  (∃ d : ℕ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → div_by d (a^n + b^n + 1)) ↔
  ((∃ d : ℕ, d = 2 ∧ (a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0)) ∨
   (∃ d : ℕ, d = 3 ∧ ((a % 3 = 1 ∧ b % 3 = 1) ∨ (a % 3 = 2 ∧ b % 3 = 2))) ∨
   (∃ d : ℕ, d = 6 ∧ ((a % 6 = 1 ∧ b % 6 = 4) ∨ (a % 6 = 4 ∧ b % 6 = 1)))) :=
by
  sorry

end pairs_condition_l192_192152


namespace potassium_iodide_formation_l192_192672

-- Define the types and variables
variable (KOH NH4I KI NH3 H2O: Type)
variable (moles : Type)
variable (reaction : KOH → NH4I → KI → NH3 → H2O → Prop)

-- Define the given conditions
variables (three_moles_KOH : moles) (three_moles_NH4I : moles)
variables [OfNat moles 3]

-- The reaction equation
axiom reaction_eq : ∀ (k : KOH) (a : NH4I) (i : KI) (n : NH3) (w : H2O), reaction k a i n w

-- The goal: prove 3 moles of KI are formed
theorem potassium_iodide_formation (three_moles_KOH: moles) (three_moles_NH4I: moles)
  (h1: three_moles_KOH = 3) (h2: three_moles_NH4I = 3)
  (eq: ∀ (k: KOH) (a: NH4I), reaction k a KI NH3 H2O): 
  three_moles_KOH = 3 ∧ three_moles_NH4I = 3 → three_moles_KI = 3 :=
by
  sorry

end potassium_iodide_formation_l192_192672


namespace find_min_p_q_r_s_l192_192430

theorem find_min_p_q_r_s :
  ∃ p q r s : ℕ, 
    (p > 0) ∧ (q > 0) ∧ (r > 0) ∧ (s > 0) ∧
    ((matrix.mul (matrix.vec_cons (20:ℕ) [27]) (λ _ _, 4:ℕ)) = matrix.mul (λ _ _,4:ℕ) (matrix.vec_cons (20:ℕ) [27])) ∧
    ((matrix.mul (matrix.vec_cons (27:ℕ) [0]) (λ _ _, 3:ℕ)) = matrix.mul (λ _ _,3:ℕ) (matrix.vec_cons (27:ℕ) [0])) ∧
    p + q + r + s = 63 :=
by
  sorry

end find_min_p_q_r_s_l192_192430


namespace area_between_concentric_circles_l192_192521

theorem area_between_concentric_circles (r1 r2 : ℝ) (h1 : r1 = 12) (h2 : r2 = 7) : 
  let A_large := π * r1^2
  let A_small := π * r2^2
  let A_ring := A_large - A_small
  A_ring = 95 * π :=
by
  rw [h1, h2]
  let A_large := π * 12^2
  let A_small := π * 7^2
  let A_ring := A_large - A_small
  calc
    A_ring = π * 12^2 - π * 7^2 : by rw [A_large, A_small]
    ...    = π * (12^2 - 7^2)   : by rw [mul_sub]
    ...    = π * (144 - 49)     : by norm_num
    ...    = π * 95             : by norm_num
    ...    = 95 * π             : by ring

end area_between_concentric_circles_l192_192521


namespace num_divisors_8_fact_l192_192854

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192854


namespace find_f_prime_2_l192_192718

def f (x : ℝ) : ℝ := 2 * (f' 2) * x + x^3
def f' (x : ℝ) := 2 * (f' 2) + 3 * x^2

theorem find_f_prime_2 : f' 2 = -12 := by
  sorry

end find_f_prime_2_l192_192718


namespace cannon_hit_probability_l192_192101

theorem cannon_hit_probability {P2 P3 : ℝ} (hP1 : 0.5 <= P2) (hP2 : P2 = 0.2) (hP3 : P3 = 0.3) (h_none_hit : (1 - 0.5) * (1 - P2) * (1 - P3) = 0.28) :
  0.5 = 0.5 :=
by sorry

end cannon_hit_probability_l192_192101


namespace even_odd_equal_l192_192048

-- Define a 3x3 grid of 0s and 1s
def Grid := Matrix (Fin 3) (Fin 3) ℕ 

-- Define a function to compute the parity (even or odd) of the sum of rows, columns, and diagonals
def isOddSum (grid : Grid) (f : Fin 3 → ℕ) : Bool :=
  (∑ i, grid (f i)) % 2 = 1

-- Define a function to count points for a grid
def scoreGrid (grid : Grid) : ℕ :=
  let rows := (Fin 3).toList.map (λ i => isOddSum grid (λ j => ⟨i.val, by decide⟩))
  let cols := (Fin 3).toList.map (λ j => isOddSum grid (λ i => ⟨j.val, by decide⟩))
  let diag1 := isOddSum grid (λ i => ⟨i.val, by decide⟩)
  let diag2 := isOddSum grid (λ i => ⟨2 - i.val, by decide⟩)
  (rows.filter id).length + (cols.filter id).length + [diag1, diag2].filter id.length

-- Prove that the number of grids with an even number of points is equal to the number of grids with an odd number of points
theorem even_odd_equal : 
  let grids := (Fin 2) ^ (Fin 3 × Fin 3) -- All 3x3 grids, each cell being 0 or 1
  let E := grids.filter (λ grid => scoreGrid grid % 2 = 0).length
  let O := grids.filter (λ grid => scoreGrid grid % 2 = 1).length
  E = O :=
sorry

end even_odd_equal_l192_192048


namespace find_rectangle_length_l192_192550

theorem find_rectangle_length (L W : ℕ) (h_area : L * W = 300) (h_perimeter : 2 * L + 2 * W = 70) : L = 20 :=
by
  sorry

end find_rectangle_length_l192_192550


namespace order_of_nums_l192_192696

variable (a b : ℝ)

theorem order_of_nums (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := 
sorry

end order_of_nums_l192_192696


namespace convex_numbers_count_l192_192356

def is_convex_number (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 < d2 ∧ d3 < d2

theorem convex_numbers_count : 
  (Finset.filter is_convex_number (Finset.range 1000)).card = 240 := by sorry

end convex_numbers_count_l192_192356


namespace count_possible_integer_values_l192_192281

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192281


namespace positive_divisors_of_8_factorial_l192_192828

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192828


namespace num_divisors_8_fact_l192_192861

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192861


namespace range_of_a_l192_192452

variable {a : ℝ} {x : ℝ}

-- Definitions of propositions p and q
def p (a : ℝ) : Prop := ∀ x > 0, x + 1/x > a
def q (a : ℝ) : Prop := ∃ x, x^2 - 2 * a * x + 1 ≤ 0

-- The theorem to prove the range of values for a
theorem range_of_a (h1 : ¬¬q a = true) (h2 : ¬(p a ∧ q a)) : a ≥ 2 :=
begin
  sorry
end

end range_of_a_l192_192452


namespace cone_volume_l192_192499

theorem cone_volume :
  ∀ (l h : ℝ) (r : ℝ), l = 15 ∧ h = 9 ∧ h = 3 * r → 
  (1 / 3) * Real.pi * r^2 * h = 27 * Real.pi :=
by
  intros l h r
  intro h_eqns
  sorry

end cone_volume_l192_192499


namespace smallest_positive_period_of_f_range_and_decreasing_interval_of_f_l192_192227

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * (cos x ^ 2 - sin x ^ 2) + 2 * sin x * cos x

theorem smallest_positive_period_of_f :
  (∀ x, f (x + π) = f x) ∧ (∀ T, T > 0 → (∀ x, f (x + T) = f x) → T = π) := sorry

theorem range_and_decreasing_interval_of_f :
  (∀ x ∈ Icc (-π/3) (π/3), -sqrt 3 ≤ f x ∧ f x ≤ 2) ∧
  (∀ x, x ∈ Icc (π/12) (7 * π/12) → ∀ h : f (x + dx) < f x) := sorry

end smallest_positive_period_of_f_range_and_decreasing_interval_of_f_l192_192227


namespace range_of_a_l192_192975

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + 9| > a) → a < 8 :=
by
  sorry

end range_of_a_l192_192975


namespace prob_divisors_8_fact_l192_192816

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192816


namespace positive_divisors_of_8_factorial_l192_192818

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192818


namespace num_divisors_8_factorial_l192_192759

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192759


namespace num_divisors_of_8_factorial_l192_192869

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192869


namespace plane_difference_correct_l192_192653

noncomputable def max_planes : ℕ := 27
noncomputable def min_planes : ℕ := 7
noncomputable def diff_planes : ℕ := max_planes - min_planes

theorem plane_difference_correct : diff_planes = 20 := by
  sorry

end plane_difference_correct_l192_192653


namespace num_divisors_fact8_l192_192958

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192958


namespace total_height_correct_l192_192594

-- Stack and dimensions setup
def height_of_disc_stack (top_diameter bottom_diameter disc_thickness : ℕ) : ℕ :=
  let num_discs := (top_diameter - bottom_diameter) / 2 + 1
  num_discs * disc_thickness

def total_height (top_diameter bottom_diameter disc_thickness cylinder_height : ℕ) : ℕ :=
  height_of_disc_stack top_diameter bottom_diameter disc_thickness + cylinder_height

-- Given conditions
def top_diameter := 15
def bottom_diameter := 1
def disc_thickness := 2
def cylinder_height := 10
def correct_answer := 26

-- Proof problem
theorem total_height_correct :
  total_height top_diameter bottom_diameter disc_thickness cylinder_height = correct_answer :=
by
  sorry

end total_height_correct_l192_192594


namespace find_parabola_equation_prove_distances_constant_find_minimum_sum_of_areas_l192_192233

section

variable {P : ℝ × ℝ}

def parabola_vertex_origin (x y : ℝ) : Prop :=
  x^2 = 4 * y

def point_on_directrix_condition (x y : ℝ) (p : ℝ) : Prop :=
  abs (y + p) = 5

def circle (x y : ℝ) : Prop :=
  x^2 + (y - 1)^2 = 1

def parabola_equation (x y : ℝ) : Prop :=
  parabola_vertex_origin x y

def distances_condition (A B C D : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (x4, y4) := D
  abs ((x1 - x3) * (x2 - x4) + (y1 - y3) * (y2 - y4)) = 0  -- Placeholder for showing the constant product

theorem find_parabola_equation (m p : ℝ) :
  point_on_directrix_condition m 4 p →
  0 < p →
  parabola_equation 4 p := sorry

theorem prove_distances_constant (A B C D : ℝ × ℝ) :
  ∀ (x1 y1 x2 y2 x3 y3 x4 y4 k : ℝ),
  A = (x1, y1) ∧ B = (x2, y2) ∧ C = (x3, k * x3 + 1) ∧ D = (x4, k * x4 + 1) →
  distances_condition A B C D := sorry

theorem find_minimum_sum_of_areas (A B C D M : ℝ × ℝ) :
  ∀ (t : ℝ),
  t = sqrt (k^2 + 1) →
  1 ≤ t →
  let area_ACM := 4 * t^3 - 2 * t in
  area_ACM = 2 := sorry

end

end find_parabola_equation_prove_distances_constant_find_minimum_sum_of_areas_l192_192233


namespace find_expression_l192_192014

theorem find_expression 
  (E a : ℤ) 
  (h1 : (E + (3 * a - 8)) / 2 = 74) 
  (h2 : a = 28) : 
  E = 72 := 
by
  sorry

end find_expression_l192_192014


namespace time_stopped_per_hour_correct_l192_192661

variables (speed_excluding_stoppages : ℝ) (speed_including_stoppages : ℝ) 

def time_stopped_per_hour (speed_excluding_stoppages speed_including_stoppages : ℝ) : ℝ :=
  let distance_lost := speed_excluding_stoppages - speed_including_stoppages in
  let time_in_hours := distance_lost / speed_excluding_stoppages in
  time_in_hours * 60

theorem time_stopped_per_hour_correct
  (h1 : speed_excluding_stoppages = 80)
  (h2 : speed_including_stoppages = 60) :
  time_stopped_per_hour speed_excluding_stoppages speed_including_stoppages = 15 :=
by
  unfold time_stopped_per_hour
  rw [h1, h2]
  norm_num
  sorry

end time_stopped_per_hour_correct_l192_192661


namespace Kelly_carrots_weight_l192_192388

theorem Kelly_carrots_weight :
  let carrots_from_first_bed := 55
  let carrots_from_second_bed := 101
  let carrots_from_third_bed := 78
  let carrots_per_pound := 6
  let total_carrots := carrots_from_first_bed + carrots_from_second_bed + carrots_from_third_bed
  let total_pounds := total_carrots / carrots_per_pound
  total_pounds = 39 :=
by {
  let carrots_from_first_bed := 55,
  let carrots_from_second_bed := 101,
  let carrots_from_third_bed := 78,
  let carrots_per_pound := 6,
  let total_carrots := carrots_from_first_bed + carrots_from_second_bed + carrots_from_third_bed,
  let total_pounds := total_carrots / carrots_per_pound,
  sorry
}

end Kelly_carrots_weight_l192_192388


namespace rectangle_perimeter_l192_192592

-- Define the conditions
variables (z w : ℕ)
-- Define the side lengths of the rectangles
def rectangle_long_side := z - w
def rectangle_short_side := w

-- Theorem: The perimeter of one of the four rectangles
theorem rectangle_perimeter : 2 * (rectangle_long_side z w) + 2 * (rectangle_short_side w) = 2 * z :=
by sorry

end rectangle_perimeter_l192_192592


namespace kate_hair_length_l192_192406

theorem kate_hair_length :
  ∀ (logans_hair : ℕ) (emilys_hair : ℕ) (kates_hair : ℕ),
  logans_hair = 20 →
  emilys_hair = logans_hair + 6 →
  kates_hair = emilys_hair / 2 →
  kates_hair = 13 :=
by
  intros logans_hair emilys_hair kates_hair
  sorry

end kate_hair_length_l192_192406


namespace spanish_but_not_german_l192_192372

-- Given conditions
variables (total_students : ℕ) (both_languages : ℕ) (spanish_to_german_ratio : ℕ)

-- Initial conditions defined
def students_in_each_language := total_students = 30
def students_in_both := both_languages = 2
def language_ratio := spanish_to_german_ratio = 3

-- Goal: number of students taking Spanish but not German is 20.
theorem spanish_but_not_german :
  (∃ (x y : ℕ), x + y + both_languages = total_students ∧ y = spanish_to_german_ratio * (x + both_languages) - 2 ∧ y - both_languages = 20) :=
by 
  simp [students_in_each_language, students_in_both, language_ratio]
  -- Suppose x = number of German students excluding Ben and Alice
  -- Suppose y = number of Spanish students excluding Ben and Alice
  use 6, 22
  simp [total_students, both_languages, spanish_to_german_ratio]
  -- Verifying the conditions and solving accordingly.
  sorry

end spanish_but_not_german_l192_192372


namespace sum_of_coefficients_of_factors_is_4_l192_192020

theorem sum_of_coefficients_of_factors_is_4 :
  ∃ (a b c d e f g h j k : ℤ),
    (8 * x^4 - 125 * y^4 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2))
    ∧ (a + b + c + d + e + f + g + h + j + k = 4) :=
begin
  sorry
end

end sum_of_coefficients_of_factors_is_4_l192_192020


namespace binary_divide_and_double_is_4_l192_192062

theorem binary_divide_and_double_is_4 : 
  let n := 0b111011010010 in
  (n / 4) * 2 = 4 :=
by
  sorry

end binary_divide_and_double_is_4_l192_192062


namespace num_divisors_8_fact_l192_192864

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192864


namespace probability_of_prime_sum_l192_192064

/-- 
  We will define a function that returns the probability that the sum of the results 
  of two six-sided dice is a prime number.
-/

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def dice_sums_prime_probability : ℚ :=
  let outcomes : Finset (ℕ × ℕ) := (Finset.range 6).product (Finset.range 6)
  let prime_sums : Finset ℕ := outcomes.image (λ p, p.1 + p.2 + 2).filter is_prime
  (prime_sums.card : ℚ) / (outcomes.card : ℚ)

theorem probability_of_prime_sum :
  dice_sums_prime_probability = 5 / 12 :=
by 
  sorry

end probability_of_prime_sum_l192_192064


namespace first_part_is_7613_l192_192579

theorem first_part_is_7613 :
  ∃ (n : ℕ), ∃ (d : ℕ), d = 3 ∧ (761 * 10 + d) * 1000 + 829 = n ∧ (n % 9 = 0) ∧ (761 * 10 + d = 7613) := 
by
  sorry

end first_part_is_7613_l192_192579


namespace number_of_x_values_l192_192266

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192266


namespace product_of_c_values_l192_192512

theorem product_of_c_values :
  ∃ (c1 c2 : ℕ), (c1 > 0 ∧ c2 > 0) ∧
  (∃ (x1 x2 : ℚ), (7 * x1^2 + 15 * x1 + c1 = 0) ∧ (7 * x2^2 + 15 * x2 + c2 = 0)) ∧
  (c1 * c2 = 16) :=
sorry

end product_of_c_values_l192_192512


namespace eight_pow_mn_eq_p_pow_3n_l192_192468

theorem eight_pow_mn_eq_p_pow_3n (m n : ℤ) (P Q : ℤ) (hP : P = 2 ^ m) (hQ : Q = 5 ^ n) : 
  8 ^ (m * n) = P ^ (3 * n) :=
by
  sorry

end eight_pow_mn_eq_p_pow_3n_l192_192468


namespace quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l192_192735

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 2

-- Problem 1: Prove that the quadratic function passes through the origin for m = 1 or m = -2
theorem quadratic_passes_through_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -2) ∧ quadratic m 0 = 0 := by
  sorry

-- Problem 2: Prove that the quadratic function is symmetric about the y-axis for m = 0
theorem quadratic_symmetric_about_y_axis :
  ∃ m : ℝ, m = 0 ∧ ∀ x : ℝ, quadratic m x = quadratic m (-x) := by
  sorry

end quadratic_passes_through_origin_quadratic_symmetric_about_y_axis_l192_192735


namespace find_f_2016_l192_192420

noncomputable def f : ℝ → ℝ :=
sorry

axiom f_0_eq_2016 : f 0 = 2016

axiom f_x_plus_2_minus_f_x_leq : ∀ x : ℝ, f (x + 2) - f x ≤ 3 * 2 ^ x

axiom f_x_plus_6_minus_f_x_geq : ∀ x : ℝ, f (x + 6) - f x ≥ 63 * 2 ^ x

theorem find_f_2016 : f 2016 = 2015 + 2 ^ 2020 :=
sorry

end find_f_2016_l192_192420


namespace num_possible_integer_values_x_l192_192341

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192341


namespace compute_fg_l192_192232

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem compute_fg : f (g (-3)) = 3 := by
  sorry

end compute_fg_l192_192232


namespace sum_of_alternating_angles_is_1200_l192_192507

-- Problem Statement
theorem sum_of_alternating_angles_is_1200 :
  (∃ (n : ℕ) (z : ℂ), (z^24 - z^6 = 1) ∧ (norm z = 1) ∧ 
    let thetas := (filter_map (λ (m : ℕ), angle_normalized (z ^ m)) (range (2*n)))
      in (sum_even_indices thetas) = 1200) :=
sorry

-- Define essential auxiliary functions 
def angle_normalized (z : ℂ) : ℝ :=
  real.arccos (z.re)

def sum_even_indices (l : list ℝ) : ℝ :=
  l.enum.filter_map (λ (i, x), if i.even then some x else none).sum

end sum_of_alternating_angles_is_1200_l192_192507


namespace num_divisors_of_8_factorial_l192_192871

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192871


namespace least_k_exists_A_B_l192_192409

-- Definitions for the conditions
def is_set_of_non_neg_integers (S : Set ℕ) : Prop :=
  ∀ x ∈ S, 0 ≤ x

def possible_sums (A B : Set ℕ) : Set ℕ :=
  {z | ∃ a ∈ A, ∃ b ∈ B, z = a + b}

-- The statement we need to prove
theorem least_k_exists_A_B :
  ∃ (k : ℕ) (A B : Set ℕ), 
    is_set_of_non_neg_integers A ∧ is_set_of_non_neg_integers B ∧ 
    A.card = k ∧ B.card = 2 * k ∧
    possible_sums A B = {n | 0 ≤ n ∧ n ≤ 2020} ∧
    k = 32 :=
  sorry

end least_k_exists_A_B_l192_192409


namespace fx_properties_l192_192022

-- Definition of the function
def f (x : ℝ) : ℝ := x * |x|

-- Lean statement for the proof problem
theorem fx_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) :=
by
  -- Definition used directly from the conditions
  sorry

end fx_properties_l192_192022


namespace find_x_l192_192361

def operation (x y : ℕ) : ℕ := 2 * x * y

theorem find_x : 
  (operation 4 5 = 40) ∧ (operation x 40 = 480) → x = 6 :=
by
  sorry

end find_x_l192_192361


namespace isosceles_triangles_with_perimeter_25_l192_192744

/-- Prove that there are 6 distinct isosceles triangles with integer side lengths 
and a perimeter of 25 -/
theorem isosceles_triangles_with_perimeter_25 :
  ∃ (count : ℕ), 
    count = 6 ∧ 
    (∀ (a b : ℕ), 
      let a1 := a,
          a2 := a,
          b3 := b in
      2 * a + b = 25 → 
      2 * a > b ∧ a + b > a ∧ b < 2 * a ∧
      a > 0 ∧ b > 0 ∧ a ∈ finset.Icc 7 12) :=
by sorry

end isosceles_triangles_with_perimeter_25_l192_192744


namespace find_k_l192_192732

theorem find_k
  (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : ∃ (x y : ℝ), (x - k * y - 5 = 0 ∧ x^2 + y^2 = 10 ∧ (A = (x, y) ∨ B = (x, y))))
  (h2 : (A.fst^2 + A.snd^2 = 10) ∧ (B.fst^2 + B.snd^2 = 10))
  (h3 : (A.fst - k * A.snd - 5 = 0) ∧ (B.fst - k * B.snd - 5 = 0))
  (h4 : A.fst * B.fst + A.snd * B.snd = 0) :
  k = 2 ∨ k = -2 :=
by
  sorry

end find_k_l192_192732


namespace least_time_for_4_horses_sum_of_digits_S_is_6_l192_192508

-- Definition of horse run intervals
def horse_intervals : List Nat := List.range' 1 9 |>.map (λ k => 2 * k)

-- Function to compute LCM of a set of numbers
def lcm_set (s : List Nat) : Nat :=
  s.foldl Nat.lcm 1

-- Proving that 4 of the horse intervals have an LCM of 24
theorem least_time_for_4_horses : 
  ∃ S > 0, (S = 24 ∧ (lcm_set [2, 4, 6, 8] = S)) ∧
  (List.length (horse_intervals.filter (λ t => S % t = 0)) ≥ 4) := 
by
  sorry

-- Proving the sum of the digits of S (24) is 6
theorem sum_of_digits_S_is_6 : 
  let S := 24
  (S / 10 + S % 10 = 6) :=
by
  sorry

end least_time_for_4_horses_sum_of_digits_S_is_6_l192_192508


namespace bank_robbery_participants_l192_192603

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l192_192603


namespace sum_of_possible_values_l192_192259

theorem sum_of_possible_values (x : ℝ) (h : (x + 3) * (x - 5) = 20) : x = -2 ∨ x = 7 :=
sorry

end sum_of_possible_values_l192_192259


namespace prob_divisors_8_fact_l192_192814

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192814


namespace log_sum_equals_half_impl_x_eq_k_pow_one_over_k_l192_192546

theorem log_sum_equals_half_impl_x_eq_k_pow_one_over_k (k : ℕ) (hk : k ≥ 1) (x : ℝ) :
    7.318 * Real.logBase k x + Real.logBase (Real.sqrt k) x + Real.logBase (k^(1/3)) x + Real.logBase (k^(1/4)) x + ... + Real.logBase (k^(1/k)) x = (k + 1) / 2 →
    x = k^(1/k) :=
sorry

end log_sum_equals_half_impl_x_eq_k_pow_one_over_k_l192_192546


namespace function_properties_l192_192618

-- Define the function f as x -> x^(-2)
def f (x : ℝ) : ℝ := x^(-2)

-- The proof statement to show f is an even function and monotonically decreasing on (0, ∞)
theorem function_properties :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by 
  sorry

end function_properties_l192_192618


namespace value_of_fraction_l192_192345

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end value_of_fraction_l192_192345


namespace enclosed_region_area_l192_192013

def area_of_enclosed_region : ℝ :=
  ∫ x in 0..1, x^{(1:ℝ)/2} - x^2

theorem enclosed_region_area :
  area_of_enclosed_region = 1 / 3 := by
  sorry

end enclosed_region_area_l192_192013


namespace num_pos_divisors_fact8_l192_192782

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192782


namespace num_divisors_fact8_l192_192961

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192961


namespace max_evolving_path_length_mul_count_l192_192639

-- Definition of the points in the 5x5 grid and the distance function
def distance (p1 p2 : ℕ × ℕ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Evolving path: a sequence of points where the distance is non-decreasing
def is_evolving_path (path : list (ℕ × ℕ)) : Prop :=
  ∀ i, i < path.length - 1 → distance (path.nth_le i (by linarith)) (path.nth_le (i+1) (by linarith)) ≤ distance (path.nth_le (i+1) (by linarith)) (path.nth_le (i+2) (by linarith))

-- The grid coordinates
def grid_points : list (ℕ × ℕ) :=
  [(x, y) | x ← list.range 5, y ← list.range 5]

-- Proof problem
theorem max_evolving_path_length_mul_count :
  ∃ (n s : ℕ), (n = 11) ∧ (s = _) → n * s = 11 * s :=
by sorry

end max_evolving_path_length_mul_count_l192_192639


namespace value_of_ab_l192_192230

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a - real.sqrt 3 * real.tan (2 * x)

theorem value_of_ab (a b : ℝ) (hmax : f (-π/6) a = 7) (hmin : ∃ b, f b a = 3 ∧ -π/6 ≤ b ∧ b ≤ π/4) : a * b = π/3 :=
by
  sorry

end value_of_ab_l192_192230


namespace modulus_z_eq_sqrt_five_l192_192189

theorem modulus_z_eq_sqrt_five (i : ℂ) (z : ℂ) (h : (1 + i) * z = 3 - i) : abs(z) = sqrt(5) :=
sorry

end modulus_z_eq_sqrt_five_l192_192189


namespace num_possible_integer_values_l192_192303

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192303


namespace num_divisors_8_factorial_l192_192926

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192926


namespace number_of_two_bedroom_units_l192_192097

-- Definitions based on the conditions
def is_solution (x y : ℕ) : Prop :=
  (x + y = 12) ∧ (360 * x + 450 * y = 4950)

theorem number_of_two_bedroom_units : ∃ y : ℕ, is_solution (12 - y) y ∧ y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l192_192097


namespace num_positive_divisors_8_factorial_l192_192933

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192933


namespace integer_values_of_x_l192_192289

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192289


namespace num_possible_integer_values_x_l192_192340

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192340


namespace five_fourths_of_eight_thirds_is_correct_l192_192671

-- Define the given fractions
def five_fourths : ℚ := 5 / 4
def eight_thirds : ℚ := 8 / 3

-- Define the expected result
def expected_result : ℚ := 10 / 3

-- Theorem to prove correctness of the computation
theorem five_fourths_of_eight_thirds_is_correct : five_fourths * eight_thirds = expected_result := by
  sorry

end five_fourths_of_eight_thirds_is_correct_l192_192671


namespace value_of_6_inch_cube_is_1688_l192_192569

noncomputable def cube_value (side_length : ℝ) : ℝ :=
  let volume := side_length ^ 3
  (volume / 64) * 500

-- Main statement
theorem value_of_6_inch_cube_is_1688 :
  cube_value 6 = 1688 := by
  sorry

end value_of_6_inch_cube_is_1688_l192_192569


namespace point_to_spherical_l192_192150

def point_rectangular : ℝ × ℝ × ℝ := (4, -4 * Real.sqrt 3, 4)

def spherical_coordinates : ℝ × ℝ × ℝ := 
  (4 * Real.sqrt 5, 4 * Real.pi / 3, Real.arccos (1 / Real.sqrt 5))

theorem point_to_spherical :
  let (x, y, z) := point_rectangular in
  let (ρ, θ, φ) := spherical_coordinates in
  ρ = Real.sqrt (x^2 + y^2 + z^2) ∧ φ = Real.arccos (z / ρ) ∧ θ = Real.pi + Real.arctan (y / x) :=
by
  sorry

end point_to_spherical_l192_192150


namespace mawangdui_age_l192_192722

theorem mawangdui_age (a b k x : Real) 
  (h1 : b = a * Real.exp (-k * x))
  (h_half_life : 1 / 2 = Real.exp (-5730 * k))
  (h_mawangdui : 0.767 = Real.exp (-k * x))
  (log2_0_767 : Real.log 0.767 / Real.log 2 ≈ -0.4) :
  x ≈ 2292 := 
sorry

end mawangdui_age_l192_192722


namespace common_root_value_l192_192185

theorem common_root_value (p : ℝ) (hp : p > 0) : 
  (∃ x : ℝ, 3 * x ^ 2 - 4 * p * x + 9 = 0 ∧ x ^ 2 - 2 * p * x + 5 = 0) ↔ p = 3 :=
by {
  sorry
}

end common_root_value_l192_192185


namespace positive_divisors_8_factorial_l192_192849

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192849


namespace num_divisors_of_8_factorial_l192_192880

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192880


namespace greatest_A_satisfies_condition_l192_192173

theorem greatest_A_satisfies_condition :
  ∃ (A : ℝ), A = 64 ∧ ∀ (s : Fin₇ → ℝ), (∀ i, 1 ≤ s i ∧ s i ≤ A) →
  ∃ (i j : Fin₇), i ≠ j ∧ (1 / 2 ≤ s i / s j ∧ s i / s j ≤ 2) :=
by 
  sorry

end greatest_A_satisfies_condition_l192_192173


namespace num_pos_divisors_fact8_l192_192777

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192777


namespace min_intersection_value_l192_192421

theorem min_intersection_value {A B C : Set (Fin 102)} 
  (h1 : A.card = 100) 
  (h2 : B.card = 100) 
  (h3 : (2^A.card + 2^B.card + 2^C.card = 2^(A ∪ B ∪ C).card)) :
  (A ∩ B ∩ C).card = 97 := 
sorry

end min_intersection_value_l192_192421


namespace positive_divisors_of_8_factorial_l192_192820

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192820


namespace chord_length_perpendicular_bisector_of_radius_l192_192104

theorem chord_length_perpendicular_bisector_of_radius (r : ℝ) (h : r = 15) :
  ∃ (CD : ℝ), CD = 15 * Real.sqrt 3 :=
by
  sorry

end chord_length_perpendicular_bisector_of_radius_l192_192104


namespace participants_in_robbery_l192_192613

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery_l192_192613


namespace exists_equidistant_point_l192_192371

-- Defining the conditions
variables {l1 l2 l3 : Line}

-- These definitions are used to describe the conditions and the problem.
def at_most_two_parallel (l1 l2 l3 : Line) : Prop := 
  (¬parallel l1 l2) ∨ (¬parallel l2 l3) ∨ (¬parallel l3 l1)

theorem exists_equidistant_point (l1 l2 l3 : Line) 
  (h : at_most_two_parallel l1 l2 l3) : ∃ P : Point, equidistant P l1 ∧ equidistant P l2 ∧ equidistant P l3 := 
sorry

end exists_equidistant_point_l192_192371


namespace packaging_combinations_l192_192565

-- Conditions
def wrapping_paper_choices : ℕ := 10
def ribbon_colors : ℕ := 5
def gift_tag_styles : ℕ := 6

-- Question and proof
theorem packaging_combinations : wrapping_paper_choices * ribbon_colors * gift_tag_styles = 300 := by
  sorry

end packaging_combinations_l192_192565


namespace number_of_integer_values_l192_192332

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192332


namespace retailer_selling_price_l192_192576

theorem retailer_selling_price
  (cost_price_manufacturer : ℝ)
  (manufacturer_profit_rate : ℝ)
  (wholesaler_profit_rate : ℝ)
  (retailer_profit_rate : ℝ)
  (manufacturer_selling_price : ℝ)
  (wholesaler_selling_price : ℝ)
  (retailer_selling_price : ℝ)
  (h1 : cost_price_manufacturer = 17)
  (h2 : manufacturer_profit_rate = 0.18)
  (h3 : wholesaler_profit_rate = 0.20)
  (h4 : retailer_profit_rate = 0.25)
  (h5 : manufacturer_selling_price = cost_price_manufacturer + (manufacturer_profit_rate * cost_price_manufacturer))
  (h6 : wholesaler_selling_price = manufacturer_selling_price + (wholesaler_profit_rate * manufacturer_selling_price))
  (h7 : retailer_selling_price = wholesaler_selling_price + (retailer_profit_rate * wholesaler_selling_price)) :
  retailer_selling_price = 30.09 :=
by {
  sorry
}

end retailer_selling_price_l192_192576


namespace initial_amount_l192_192660

theorem initial_amount (A : ℝ) (annual_increase : ℝ) (final_value : ℝ) (h1 : annual_increase = 1/8) (h2 : final_value = 64_800) 
  (h3 : (81/64) * A = final_value) : A = 51_200 := 
by
  sorry

end initial_amount_l192_192660


namespace num_possible_integer_values_x_l192_192338

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192338


namespace num_divisors_8_factorial_l192_192925

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192925


namespace product_of_slopes_l192_192240

theorem product_of_slopes (m n : ℝ) (φ₁ φ₂ : ℝ) 
  (h1 : ∀ x, y = m * x)
  (h2 : ∀ x, y = n * x)
  (h3 : φ₁ = 2 * φ₂) 
  (h4 : m = 3 * n)
  (h5 : m ≠ 0 ∧ n ≠ 0)
  : m * n = 3 / 5 :=
sorry

end product_of_slopes_l192_192240


namespace solve_trigonometric_equation_l192_192072

-- Conditions: cos x ≠ 0 and sin 3x ≠ 0
variables {x k n : ℝ}
variables {l : ℤ}

def satisfies_equation (x : ℝ) : Prop :=
  2 * Real.sin(3 * x)^2 + Real.sin(6 * x)^2 = (Real.sin(2 * x) + Real.sin(4 * x)) * (1 / Real.cos x) * (1 / Real.sin(3 * x))

def cos_ne_zero (x : ℝ) : Prop := Real.cos x ≠ 0
def sin3x_ne_zero (x : ℝ) : Prop := Real.sin(3 * x) ≠ 0

theorem solve_trigonometric_equation :
  (∀ x, cos_ne_zero x → sin3x_ne_zero x → 
    (x = π / 12 * (2 * k + 1) ∨ 
     x = π / 6 * (2 * n + 1))) :=
sorry

end solve_trigonometric_equation_l192_192072


namespace probability_of_arithmetic_progression_l192_192691

-- Defining the problem context
def four_dice_tossed : Prop :=
  ∃ (d1 d2 d3 d4 : ℕ), 
    d1 ∈ {1, 2, 3, 4, 5, 6} ∧
    d2 ∈ {1, 2, 3, 4, 5, 6} ∧
    d3 ∈ {1, 2, 3, 4, 5, 6} ∧
    d4 ∈ {1, 2, 3, 4, 5, 6}

-- Defining the condition for forming an arithmetic progression with a common difference of 2
def forms_arithmetic_progression (d1 d2 d3 d4 : ℕ) : Prop :=
  ∃ (a : ℕ), 
    (d1 = a ∨ d1 = a + 2 ∨ d1 = a + 4 ∨ d1 = a + 6) ∧
    (d2 = a ∨ d2 = a + 2 ∨ d2 = a + 4 ∨ d2 = a + 6) ∧
    (d3 = a ∨ d3 = a + 2 ∨ d3 = a + 4 ∨ d3 = a + 6) ∧
    (d4 = a ∨ d4 = a + 2 ∨ d4 = a + 4 ∨ d4 = a + 6)

noncomputable def probability_arithmetic_progression : ℚ :=
  (2 * 24) / 1296

theorem probability_of_arithmetic_progression : probability_arithmetic_progression = 1 / 27 := by
  sorry

end probability_of_arithmetic_progression_l192_192691


namespace num_divisors_of_8_factorial_l192_192878

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192878


namespace number_of_two_bedroom_units_l192_192096

-- Definitions based on the conditions
def is_solution (x y : ℕ) : Prop :=
  (x + y = 12) ∧ (360 * x + 450 * y = 4950)

theorem number_of_two_bedroom_units : ∃ y : ℕ, is_solution (12 - y) y ∧ y = 7 :=
by
  sorry

end number_of_two_bedroom_units_l192_192096


namespace incircle_angle_b_l192_192024

open Real

theorem incircle_angle_b
    (α β γ : ℝ)
    (h1 : α + β + γ = 180)
    (angle_AOC_eq_4_MKN : ∀ (MKN : ℝ), 4 * MKN = 180 - (180 - γ) / 2 - (180 - α) / 2) :
    β = 108 :=
by
  -- Proof will be handled here.
  sorry

end incircle_angle_b_l192_192024


namespace f_sum_2018_2019_l192_192019

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom even_shifted_function (x : ℝ) : f (x + 1) = f (-x + 1)
axiom f_neg1 : f (-1) = -1

theorem f_sum_2018_2019 : f 2018 + f 2019 = -1 :=
by sorry

end f_sum_2018_2019_l192_192019


namespace compound_interest_correct_l192_192171

noncomputable def compound_interest (P : ℝ) (rates : List (ℝ × ℕ × ℕ)) : ℝ :=
  rates.foldl (λ acc (rate, n, t), acc * (1 + rate / n) ^ (n * t)) P

noncomputable def principal : ℝ := 5000
noncomputable def periods : List (ℝ × ℕ × ℕ) := 
  [(0.035, 1, 3), (0.04, 2, 3), (0.05, 4, 4)]

theorem compound_interest_correct :
  compound_interest principal periods - principal = 2605.848 := by
  sorry

end compound_interest_correct_l192_192171


namespace part1_part2_part3_l192_192703

variable (a b : ℕ → ℝ)

-- Condition: a_1 = 1 and a_{n+1} = (n+2)/n * a_n + 1
noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 0 then 1 else ((n + 2) / n.to_real()) * a (n - 1) + 1

/-- 
  1. Prove that the sequence {a_n / n} is an arithmetic sequence.
-/
theorem part1 : ∀ n : ℕ, 1 ≤ n → (a n / n.to_real() - a (n-1) / (n-1).to_real() = constant)
  := sorry

/-- 
  2. Find a general formula for the sequence {a_n}.
-/
theorem part2 : ∀ n : ℕ, a n = (n: ℝ)^2 := sorry

/-- 
  3. Prove the sum b_n = 2 * √n / a_n < 6.
-/
theorem part3 : ∀ (n : ℕ), (b n = 2 * real.sqrt n / (a n)) → 
  ∑ i in (finset.range n), b i < 6 
  := sorry

end part1_part2_part3_l192_192703


namespace count_integer_values_l192_192309

theorem count_integer_values (x : ℕ) (h : 64 ≤ x ∧ x ≤ 80) :
    ({x : ℕ | 64 ≤ x ∧ x ≤ 80}).card = 17 := by
  sorry

end count_integer_values_l192_192309


namespace distance_point_to_plane_example_l192_192049

noncomputable def distance_from_point_to_plane (x0 y0 z0 A B C D : ℝ) : ℝ :=
  real.abs(A * x0 + B * y0 + C * z0 + D) / real.sqrt(A * A + B * B + C * C)

theorem distance_point_to_plane_example :
  distance_from_point_to_plane 2 4 1 1 2 3 3 = 8 * real.sqrt 14 / 7 :=
by
  sorry

end distance_point_to_plane_example_l192_192049


namespace age_of_B_present_l192_192552

theorem age_of_B_present (A B C : ℕ) (h1 : A + B + C = 90)
  (h2 : (A - 10) * 2 = (B - 10))
  (h3 : (B - 10) * 3 = (C - 10) * 2) :
  B = 30 := 
sorry

end age_of_B_present_l192_192552


namespace time_at_2010_minutes_after_3pm_is_930pm_l192_192477

def time_after_2010_minutes (current_time : Nat) (minutes_passed : Nat) : Nat :=
  sorry

theorem time_at_2010_minutes_after_3pm_is_930pm :
  time_after_2010_minutes 900 2010 = 1290 :=
by
  sorry

end time_at_2010_minutes_after_3pm_is_930pm_l192_192477


namespace number_of_integer_values_l192_192329

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192329


namespace Ajay_and_Vijay_work_together_l192_192976

variable (Ajay Vijay : ℕ)
variable (work_rate : ℕ → ℚ)

-- Define individual work rates
def Ajay_work_rate : ℚ := work_rate Ajay
def Vijay_work_rate : ℚ := work_rate Vijay
def combined_work_rate : ℚ := Ajay_work_rate + Vijay_work_rate

-- Define the time it takes to complete the work when both work together
def combined_time : ℚ := 1 / combined_work_rate

-- Assign Ajay and Vijay's specific work times
def Ajay_days : ℕ := 8
def Vijay_days : ℕ := 24

-- Assign the specific work rates based on days
noncomputable def work_rate := fun (days : ℕ) => (1 : ℚ) / days

theorem Ajay_and_Vijay_work_together : 
Ajay_days = 8 → Vijay_days = 24 → combined_time Ajay_days Vijay_days = 6 :=
by
  intros
  sorry

end Ajay_and_Vijay_work_together_l192_192976


namespace points_per_draw_l192_192011

-- Definitions based on conditions
def total_games : ℕ := 20
def wins : ℕ := 14
def losses : ℕ := 2
def total_points : ℕ := 46
def points_per_win : ℕ := 3
def points_per_loss : ℕ := 0

-- Calculation of the number of draws and points per draw
def draws : ℕ := total_games - wins - losses
def points_wins : ℕ := wins * points_per_win
def points_draws : ℕ := total_points - points_wins

-- Theorem statement
theorem points_per_draw : points_draws / draws = 1 := by
  sorry

end points_per_draw_l192_192011


namespace no_opsolves_7_op_4_eq_zero_l192_192519

/-- Prove there is no arithmetic operation (?, which can be +, -, ×, ÷) that makes (7 ? 4) = 0,
given the initial equation (7 ? 4) + 5 - (3 - 2) = 4. -/
theorem no_opsolves_7_op_4_eq_zero :
  ¬ (∃ (op : ℕ → ℕ → ℕ), ((∀ a b : ℕ, (op = has_add.add ∨ op = has_sub.sub ∨ op = has_mul.mul ∨ op = has_div.div) 
  ∧ ((op 7 4) + 5 - 1 = 4)) ∧ ((op 7 4) = 0))) :=
by 
  sorry

end no_opsolves_7_op_4_eq_zero_l192_192519


namespace perimeter_of_ABC_l192_192548

theorem perimeter_of_ABC {AC CD AD x : ℝ} 
  (h1 : AC = x)
  (h2 : CD = x * real.sqrt 3)
  (h3 : AD = 2 * x)
  (h4 : AC + CD + AD = 9 + 3 * real.sqrt 3) :
  3 * (2 * (9 / 4)) = 13.5 :=
by
  sorry

end perimeter_of_ABC_l192_192548


namespace eccentricity_of_ellipse_equation_of_ellipse_l192_192203

-- Definitions
variable {a b : ℝ} (M : ℝ × ℝ) (r : ℝ)
variable (h1 : 0 < b)
variable (h2 : b < a)
variable (h3 : M.fst^2 / a^2 + M.snd^2 / b^2 = 1)
variable (h4 : M.fst = sqrt (a^2 - b^2))
variable (h5 : r = abs M.snd)
variable (h6 : M.snd^2 = b^4 / a^2) -- derived from r = b^2 / a

-- Theorem 1: Eccentricity of the ellipse
theorem eccentricity_of_ellipse : sqrt (a^2 - b^2) / a = (sqrt 5 - 1) / 2 :=
by
  sorry

-- Additional definitions for part (2)
variable (side_length_abc : ℝ) (h7 : side_length_abc = 2)
variable (h8 : M.fst = sqrt 3)
variable (h9 : r = side_length_abc / 2)

-- Theorem 2: Equation of the ellipse with given conditions
theorem equation_of_ellipse : a = 3 ∧ b^2 = 6 :=
by
  sorry

end eccentricity_of_ellipse_equation_of_ellipse_l192_192203


namespace antenna_height_l192_192620

variable (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variable (H_AC : dist A C = 5)
variable (H_AD : dist A D = 3)
variable (H_DE : dist D E = 1.75)
variable (H_similar : ∀ x y z v w u, ∠x y z = ∠v w u ∧ ∠y z x = ∠w u v → similar (triangle x y z) (triangle v w u))

theorem antenna_height (dist : A → B → ℝ) : (dist A B = 4.375) :=
by
  sorry

end antenna_height_l192_192620


namespace count_possible_integer_values_l192_192280

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192280


namespace positive_divisors_8_factorial_l192_192845

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192845


namespace average_divisible_by_3_between_40_and_80_l192_192170

-- Define the condition for the set of numbers between 40 and 80 divisible by 3
def numbers_divisible_by_3_between_40_and_80 : List Int :=
  List.filter (λ n => n % 3 = 0) (List.range 81).filter (λ n => n >= 40)

-- Define the average calculation
def average (l : List Int) : Int :=
  (l.sum) / (l.length)

-- State the theorem
theorem average_divisible_by_3_between_40_and_80 :
  average numbers_divisible_by_3_between_40_and_80 = 63 :=
by
  -- The proof will go here
  sorry

end average_divisible_by_3_between_40_and_80_l192_192170


namespace ufo_convention_attendees_l192_192132

theorem ufo_convention_attendees (f m total : ℕ) 
  (h1 : m = 62) 
  (h2 : m = f + 4) : 
  total = 120 :=
by
  sorry

end ufo_convention_attendees_l192_192132


namespace find_f_2010_l192_192497

noncomputable def f : ℕ → ℤ := sorry

theorem find_f_2010 (f_prop : ∀ {a b n : ℕ}, a + b = 3 * 2^n → f a + f b = 2 * n^2) :
  f 2010 = 193 :=
sorry

end find_f_2010_l192_192497


namespace scientific_notation_500_billion_l192_192135

theorem scientific_notation_500_billion :
  ∃ (a : ℝ), 500000000000 = a * 10 ^ 10 ∧ 1 ≤ a ∧ a < 10 :=
by
  sorry

end scientific_notation_500_billion_l192_192135


namespace students_called_back_l192_192516

theorem students_called_back (g b d t c : ℕ) (h1 : g = 9) (h2 : b = 14) (h3 : d = 21) (h4 : t = g + b) (h5 : c = t - d) : c = 2 := by 
  sorry

end students_called_back_l192_192516


namespace total_bricks_used_l192_192444

def numCoursesPerWall : Nat := 6
def bricksPerCourse : Nat := 10
def numWalls : Nat := 4
def unfinishedCoursesLastWall : Nat := 2

theorem total_bricks_used : 
  let totalCourses := numWalls * numCoursesPerWall
  let bricksRequired := totalCourses * bricksPerCourse
  let bricksMissing := unfinishedCoursesLastWall * bricksPerCourse
  let bricksUsed := bricksRequired - bricksMissing
  bricksUsed = 220 := 
by
  sorry

end total_bricks_used_l192_192444


namespace maximum_sum_is_42_l192_192641

-- Definitions according to the conditions in the problem

def initial_faces : ℕ := 7 -- 2 pentagonal + 5 rectangular
def initial_vertices : ℕ := 10 -- 5 at the top and 5 at the bottom
def initial_edges : ℕ := 15 -- 5 for each pentagon and 5 linking them

def added_faces : ℕ := 5 -- 5 new triangular faces
def added_vertices : ℕ := 1 -- 1 new vertex at the apex of the pyramid
def added_edges : ℕ := 5 -- 5 new edges connecting the new vertex to the pentagon's vertices

-- New quantities after adding the pyramid
def new_faces : ℕ := initial_faces - 1 + added_faces
def new_vertices : ℕ := initial_vertices + added_vertices
def new_edges : ℕ := initial_edges + added_edges

-- Sum of the new shape's characteristics
def sum_faces_vertices_edges : ℕ := new_faces + new_vertices + new_edges

-- Statement to be proved
theorem maximum_sum_is_42 : sum_faces_vertices_edges = 42 := by
  sorry

end maximum_sum_is_42_l192_192641


namespace number_of_integer_values_l192_192331

theorem number_of_integer_values (x : ℕ) (h : ⌊ Real.sqrt x ⌋ = 8) : ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_integer_values_l192_192331


namespace num_whole_numbers_between_cuberoots_l192_192255

open Real

theorem num_whole_numbers_between_cuberoots : 
  (20 : ℝ) ^ (1 / 3) < 3 ∧ 6 < (300 : ℝ) ^ (1 / 3) → 
  ∃ (k : ℕ), k = 4 :=
by 
  intros h
  obtain ⟨h1, h2⟩ := h
  have h3 : (3 : ℝ) ≤ (300 : ℝ) ^ (1 / 3),
  { sorry }
  have h4 : (20 : ℝ) ^ (1 / 3) ≤ 2,
  { sorry }
  use k = 4
  sorry

end num_whole_numbers_between_cuberoots_l192_192255


namespace pq_greater_than_4s_l192_192652

theorem pq_greater_than_4s {ABCD : ConvexQuadrilateral} (P Q : ℝ) (S_ABCD : ℝ)
  (hP : P = perimeter ABCD)
  (hQ : Q = perimeter (QuadrilateralFormedByInscribedCircleCenters ABCD)) :
  P * Q > 4 * S_ABCD := 
sorry

end pq_greater_than_4s_l192_192652


namespace remaining_grass_area_eq_l192_192567

theorem remaining_grass_area_eq :
  let diameter := 12
  let radius := diameter / 2
  let path_width := 3
  let circle_area := π * radius^2
  let semicircle_area := circle_area / 2
  let chord_length := 2 * radius * real.sin(60 * real.pi / 180)
  let sector_area := (120 / 360) * circle_area
  let triangle_area := (1 / 2) * chord_length * (radius * real.sin(30 * real.pi / 180))
  let segment_area := sector_area - triangle_area
  let remaining_area := semicircle_area + (semicircle_area - segment_area)
  in remaining_area = 30 * π - 9 * real.sqrt 3 := by
  sorry

end remaining_grass_area_eq_l192_192567


namespace num_divisors_8_factorial_l192_192922

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192922


namespace num_pos_divisors_fact8_l192_192775

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192775


namespace two_bedroom_units_l192_192094

theorem two_bedroom_units (x y : ℕ) (h1 : x + y = 12) (h2 : 360 * x + 450 * y = 4950) : y = 7 :=
by
  sorry

end two_bedroom_units_l192_192094


namespace chocolate_chip_cookie_price_l192_192572

noncomputable def price_of_chocolate_chip_cookies :=
  let total_boxes := 1585
  let total_revenue := 1586.75
  let plain_boxes := 793.375
  let price_of_plain := 0.75
  let revenue_plain := plain_boxes * price_of_plain
  let choco_boxes := total_boxes - plain_boxes
  (993.71875 - revenue_plain) / choco_boxes

theorem chocolate_chip_cookie_price :
  price_of_chocolate_chip_cookies = 1.2525 :=
by sorry

end chocolate_chip_cookie_price_l192_192572


namespace mode_is_3_5_of_salaries_l192_192358

def salaries : List ℚ := [30, 14, 9, 6, 4, 3.5, 3]
def frequencies : List ℕ := [1, 2, 3, 4, 5, 6, 4]

noncomputable def mode_of_salaries (salaries : List ℚ) (frequencies : List ℕ) : ℚ :=
by
  sorry

theorem mode_is_3_5_of_salaries :
  mode_of_salaries salaries frequencies = 3.5 :=
by
  sorry

end mode_is_3_5_of_salaries_l192_192358


namespace positive_divisors_8_factorial_l192_192841

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192841


namespace sqrt_sum_eq_eight_l192_192055

theorem sqrt_sum_eq_eight :
  Real.sqrt (24 - 8 * Real.sqrt 3) + Real.sqrt (24 + 8 * Real.sqrt 3) = 8 := by
  sorry

end sqrt_sum_eq_eight_l192_192055


namespace determinant_of_matrixA_l192_192162

variables (x y a : ℝ)

def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![[1, x^2, y],
    [1, a * x + y, y^2],
    [1, x^2, a * x + y]]

theorem determinant_of_matrixA :
  Matrix.det matrixA = a^2 * x^2 + 2 * a * x * y + y^2 - a * x^3 - x * y^2 := by
  sorry

end determinant_of_matrixA_l192_192162


namespace positive_divisors_8_factorial_l192_192847

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def num_divisors (n : ℕ) : ℕ :=
  let factor_count : List (ℕ × ℕ) := n.factorization.to_list
  factor_count.foldr (λ p acc => (p.snd + 1) * acc) 1

theorem positive_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end positive_divisors_8_factorial_l192_192847


namespace problem1_problem2_problem3_problem4_problem5_problem6_l192_192513

-- (1) If teachers A and B must stand at the two ends, there are 48 arrangements.
theorem problem1 : 
  let A := "TeacherA"
  let B := "TeacherB"
  let students := ["Student1", "Student2", "Student3", "Student4"]
  in (endArrangements A B students).length = 48 := 
sorry

-- (2) If teachers A and B must stand next to each other, there are 240 arrangements.
theorem problem2 : 
  let A := "TeacherA"
  let B := "TeacherB"
  let students := ["Student1", "Student2", "Student3", "Student4"]
  in (adjacentArrangements A B students).length = 240 := 
sorry

-- (3) If teachers A and B cannot stand next to each other, there are 480 arrangements.
theorem problem3 : 
  let A := "TeacherA"
  let B := "TeacherB"
  let students := ["Student1", "Student2", "Student3", "Student4"]
  in (nonAdjacentArrangements A B students).length = 480 := 
sorry

-- (4) If there must be two students standing between teachers A and B, there are 144 arrangements.
theorem problem4 : 
  let A := "TeacherA"
  let B := "TeacherB"
  let students := ["Student1", "Student2", "Student3", "Student4"]
  in (twoStudentsBetweenArrangements A B students).length = 144 := 
sorry

-- (5) If teacher A cannot stand at the first position and teacher B cannot stand at the last position, there are 504 arrangements.
theorem problem5 : 
  let A := "TeacherA"
  let B := "TeacherB"
  let students := ["Student1", "Student2", "Student3", "Student4"]
  in (positionalConstraintArrangements A B students).length = 504 :=
sorry

-- (6) If student C cannot stand next to either teacher A or teacher B, there are 288 arrangements.
theorem problem6 : 
  let A := "TeacherA"
  let B := "TeacherB"
  let C := "StudentC"
  let students := ["Student1", "Student2", "Student3", "Student4"]
  in (nonAdjacentStudentArrangements A B C students).length = 288 := 
sorry

end problem1_problem2_problem3_problem4_problem5_problem6_l192_192513


namespace num_divisors_8_factorial_l192_192761

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192761


namespace total_bricks_used_l192_192445

def numCoursesPerWall : Nat := 6
def bricksPerCourse : Nat := 10
def numWalls : Nat := 4
def unfinishedCoursesLastWall : Nat := 2

theorem total_bricks_used : 
  let totalCourses := numWalls * numCoursesPerWall
  let bricksRequired := totalCourses * bricksPerCourse
  let bricksMissing := unfinishedCoursesLastWall * bricksPerCourse
  let bricksUsed := bricksRequired - bricksMissing
  bricksUsed = 220 := 
by
  sorry

end total_bricks_used_l192_192445


namespace dolphin_prob_within_2m_of_edge_l192_192109

theorem dolphin_prob_within_2m_of_edge :
  let length := 30
  let width := 20
  let total_area := length * width
  let inner_length := length - 4
  let inner_width := width - 4
  let inner_area := inner_length * inner_width
  let shaded_area := total_area - inner_area
  let probability := shaded_area.toRational / total_area.toRational
  probability = 23 / 75 := 
by
  sorry

end dolphin_prob_within_2m_of_edge_l192_192109


namespace triangle_area_28_26_10_l192_192988

noncomputable def semi_perimeter (a b c : ℕ) : ℕ := (a + b + c) / 2

noncomputable def triangle_area (a b c : ℕ) : ℕ := 
  let s := semi_perimeter a b c
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_28_26_10 : triangle_area 28 26 10 = 130 :=
by
  sorry

end triangle_area_28_26_10_l192_192988


namespace number_of_x_values_l192_192268

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192268


namespace integer_values_of_x_l192_192290

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192290


namespace bank_robbery_participants_l192_192605

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l192_192605


namespace area_triangle_correct_angle_C_correct_l192_192366

noncomputable def area_triangle {a b c : ℝ} (B : ℝ) (cosB : ℝ) (dot_product_AB_BC : ℝ) : ℝ :=
  if h : cosB = 3 / 5 ∧ dot_product_AB_BC = -21 then
    let ac := 35 in
    let sinB := 4 / 5 in
    let area := 1 / 2 * ac * sinB in
    area
  else 0

noncomputable def angle_C {a b c : ℝ} (B : ℝ) (cosB : ℝ) (a_giv : ℝ) (dot_product_AB_BC : ℝ) : Real :=
  if h : cosB = 3 / 5 ∧ dot_product_AB_BC = -21 ∧ a_giv = 7 then
    let c := 5 in
    let b := 4 * Real.sqrt 2 in
    let sinC := (c * (4 / 5)) / b in
    if sinC = Real.sqrt 2 / 2 then
      Real.pi / 4
    else
      0
  else 0

theorem area_triangle_correct (a b c B : ℝ) (cosB : ℝ) (dot_product_AB_BC : ℝ) (h_cosB : cosB = 3 / 5) (h_dot_product_AB_BC : dot_product_AB_BC = -21) :
  area_triangle B cosB dot_product_AB_BC = 14 :=
by sorry

theorem angle_C_correct (a b c B : ℝ) (cosB : ℝ) (dot_product_AB_BC : ℝ) (a_giv : ℝ) (h_cosB : cosB = 3 / 5) (h_dot_product_AB_BC : dot_product_AB_BC = -21) (h_a : a_giv = 7) :
  angle_C B cosB a_giv dot_product_AB_BC = Real.pi / 4 :=
by sorry

end area_triangle_correct_angle_C_correct_l192_192366


namespace point_A_polar_coords_max_value_MB_MC_diff_l192_192234

-- Noncomputable due to sqrt and trigonometric functions
noncomputable def circle_in_polar (theta : ℝ) : ℝ := 4 * sin theta

noncomputable def line_polar_angle : ℝ := 3 * π / 4

noncomputable def point_A_polar : ℝ × ℝ := (2 * real.sqrt 2, 3 * π / 4)

-- Definitions for the maximum norm differences
noncomputable def max_MB_MC_diff (alpha : ℝ) : ℝ := 2 * real.sqrt 2 * abs (sin (alpha + π / 4))

-- Proof of statement 1: Point A in polar coordinates
theorem point_A_polar_coords (theta : ℝ) (h0 : 0 ≤ theta) (h1 : theta < 2 * π) :
  circle_in_polar line_polar_angle = 2 * real.sqrt 2 ∧ line_polar_angle == 3 * π / 4 :=
sorry

-- Proof of statement 2: Maximum value of MB - MC
theorem max_value_MB_MC_diff :
  (∃ (alpha : ℝ), max_MB_MC_diff alpha = 2 * real.sqrt 2) :=
sorry

end point_A_polar_coords_max_value_MB_MC_diff_l192_192234


namespace rungs_to_climb_l192_192391

-- Definitions and conditions
def tree_height : ℝ := 50
def wind_increase : ℝ := 0.20
def previous_tree_height : ℝ := 6
def previous_rungs : ℕ := 12

-- Effective height calculation
def effective_tree_height : ℝ := tree_height * (1 + wind_increase)
def rungs_per_foot : ℝ := previous_rungs / previous_tree_height

-- Theorem to prove the number of rungs Jamie has to climb
theorem rungs_to_climb : 
  effective_tree_height * rungs_per_foot = 120 :=
by
  calc effective_tree_height * rungs_per_foot
       = (tree_height * (1 + wind_increase)) * (previous_rungs / previous_tree_height) : by sorry
    ... = 120 : by sorry

end rungs_to_climb_l192_192391


namespace number_of_x_values_l192_192265

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192265


namespace quotient_is_33_minus_G_l192_192197

variable (a b c d : ℕ)
variable (h_a : a < 10) (h_b : b < 10) (h_c : c < 10) (h_d : d < 10)
variable (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

def S : ℕ := a + b + c + d
def G : ℕ := Nat.gcd (Nat.gcd a b) (Nat.gcd c d)

theorem quotient_is_33_minus_G :
  (33 * S - G * S) / S = 33 - G :=
by
  dsimp [S, G]
  sorry

end quotient_is_33_minus_G_l192_192197


namespace john_change_received_is_7_l192_192397

def cost_per_orange : ℝ := 0.75
def num_oranges : ℝ := 4
def amount_paid : ℝ := 10.0
def total_cost : ℝ := num_oranges * cost_per_orange
def change_received : ℝ := amount_paid - total_cost

theorem john_change_received_is_7 : change_received = 7 :=
by
  sorry

end john_change_received_is_7_l192_192397


namespace num_divisors_8_factorial_l192_192757

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192757


namespace tangent_locus_is_circle_l192_192529

-- Define the ellipse as a set of points (x, y) satisfying the equation
def is_ellipse (x y : ℝ) : Prop := 4 * x^2 + 5 * y^2 = 20

-- Define the condition that point P has perpendicular tangents to the ellipse
def has_perpendicular_tangents (P : ℝ × ℝ) : Prop :=
  ∃ (S T : ℝ × ℝ), 
  is_ellipse S.1 S.2 ∧ 
  is_ellipse T.1 T.2 ∧ 
  P = (S.1 + T.1) / 2 ∧
  P = (S.2 + T.2) / 2 ∧
  (S.1 - P.1) * (T.1 - P.1) + (S.2 - P.2) * (T.2 - P.2) = 0

-- Define the locus to be proved
def locus_P (P : ℝ × ℝ) : Prop := P.1 ^ 2 + P.2 ^ 2 = 9

-- The final proposition that needs to be proved
theorem tangent_locus_is_circle : 
  ∀ (P : ℝ × ℝ), has_perpendicular_tangents P → locus_P P := 
sorry

end tangent_locus_is_circle_l192_192529


namespace ellipse_equation_l192_192479

-- Definitions to establish conditions
def initial_ellipse : Prop := 9 * x^2 + 4 * y^2 = 36
def minor_axis_length : ℝ := 4 * real.sqrt 5

-- Final statement to prove the equivalence
theorem ellipse_equation (x y : ℝ) (c : ℝ) (a' b' : ℝ) :
  initial_ellipse → minor_axis_length = 4 * real.sqrt 5 → c = real.sqrt 5 →
  b' = 2 * real.sqrt 5 → c^2 = a'^2 - b'^2 → a' = 5 →
  (a' = 5 ∧ b' = 2 * real.sqrt 5) → 
  ∃ (a' b' : ℝ), (x^2 / 25 + y^2 / 20 = 1) :=
by sorry

end ellipse_equation_l192_192479


namespace rods_in_one_mile_l192_192212

theorem rods_in_one_mile
  (h1 : 1 = 10 * (1 : ℝ))
  (h2 : 1 = 40 * (1 : ℝ)) :
  10 * 40 = 400 :=
by
  rw [h1, h2]
  sorry

end rods_in_one_mile_l192_192212


namespace num_divisors_8_factorial_l192_192915

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192915


namespace number_of_x_values_l192_192267

theorem number_of_x_values (x : ℤ) (h1 : 64 ≤ x) (h2 : x < 81) :
  fintype.card {x : ℤ // 64 ≤ x ∧ x < 81} = 17 :=
by
  sorry

end number_of_x_values_l192_192267


namespace sin_cos_equation_solution_l192_192972

theorem sin_cos_equation_solution (x : ℝ) (h : sin (4 * x) * sin (5 * x) = cos (4 * x) * cos (5 * x)) : x = 0 :=
sorry

end sin_cos_equation_solution_l192_192972


namespace locus_of_C_bounds_on_S_l192_192379

-- Part (a): Prove the equation of the locus E
theorem locus_of_C (x y : ℝ) (h : x ≠ 0) 
  (h1 : ∃ (G M : ℝ × ℝ), 
    G = (x / 3, y / 3) ∧ 
    M = (x / 3, 0) ∧ 
    (G.1, G.2) + (M.1, M.2) = (0, 0)) :
  x^2 / 3 + y^2 = 1 := 
sorry

-- Part (b): Prove the bounds on the area S
theorem bounds_on_S (k : ℝ) (h1 : k ≠ 0) 
  (h2 : k ≠ (1 / ℝ.sqrt 2)) (h3 : k ≠ - (1 / ℝ.sqrt 2)) 
  (h4 : ∃ (P Q R N F : ℝ × ℝ), 
    F = (ℝ.sqrt 2, 0) ∧ 
    P.1 = Q.1 ∧ P.2 = Q.2 ∧ 
    R.1 = N.1 ∧ R.2 = N.2 ∧ 
    (P.1 - F.1) * (R.1 - F.1) + (P.2 - F.2) * (R.2 - F.2) = 0) :
  3 / 2 ≤ S ∧ S ≤ 2 := 
sorry

end locus_of_C_bounds_on_S_l192_192379


namespace num_positive_divisors_8_factorial_l192_192936

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192936


namespace sequence_arithmetic_sequence_sequence_general_formula_smallest_n_for_Sn_l192_192704

theorem sequence_arithmetic_sequence {a : ℕ → ℝ} 
  (h1 : ∀ n, log 3 (a (2*n)) = a (2*n - 1) + a (2*n + 1))
  (h2 : ∀ n, a (2*n + 2) * a (2*n) = 8 * 1 ^ (a (2*n + 1)))
  (h3 : a 1 = 1)
  (h4 : a 2 = 27) :
  ∃ d : ℝ, ∀ n, a (2*n - 1) = 1 + d * (n - 1) :=
sorry

theorem sequence_general_formula {a : ℕ → ℝ}
  (h1 : ∀ n, log 3 (a (2*n)) = a (2*n - 1) + a (2*n + 1))
  (h2 : ∀ n, a (2*n + 2) * a (2*n) = 8 * 1 ^ (a (2*n + 1)))
  (h3 : a 1 = 1)
  (h4 : a 2 = 27) :
  ∀ n, a n = if n % 2 = 1 then ↑((n + 1) / 2) else 3^((n + 1) / 2) :=
sorry

theorem smallest_n_for_Sn {a : ℕ → ℝ}
  (h1 : ∀ n, log 3 (a (2*n)) = a (2*n - 1) + a (2*n + 1))
  (h2 : ∀ n, a (2*n + 2) * a (2*n) = 8 * 1 ^ (a (2*n + 1)))
  (h3 : a 1 = 1)
  (h4 : a 2 = 27)
  (h5 : ∀ n, a n = if n % 2 = 1 then ↑((n + 1) / 2) else 3^((n + 1) / 2)) :
  ∃ n, (∑ i in finset.range (n + 1), a i) > 2023 ∧ (∑ i in finset.range n, a i) ≤ 2023 :=
sorry

end sequence_arithmetic_sequence_sequence_general_formula_smallest_n_for_Sn_l192_192704


namespace incorrect_statement_l192_192116

-- Define what it means to be a sharp angle
def is_sharp_angle (θ : ℝ) : Prop := θ > 0 ∧ θ < 90

-- Define what it means to be in the first quadrant
def is_in_first_quadrant (θ : ℝ) : Prop :=
  let θ_mod := θ % 360
  θ_mod > 0 ∧ θ_mod < 90

-- The conditions provided in a)
axiom sharp_angle_def (θ : ℝ) : is_sharp_angle θ ↔ is_in_first_quadrant θ
axiom first_quadrant_def (θ : ℝ) : is_in_first_quadrant θ → is_sharp_angle θ

-- The incorrect statement to prove
theorem incorrect_statement :
  ¬ (∀ θ : ℝ, is_in_first_quadrant θ → is_sharp_angle θ) :=
by
  intro h
  have h1 : is_in_first_quadrant 390, from -- example counterexample from the solution
  { sorry },
  have h2 : ¬ is_sharp_angle 390, from -- example counterexample from the solution
  { sorry },
  exact h2 (h 390 h1)

end incorrect_statement_l192_192116


namespace positive_divisors_of_8_factorial_l192_192831

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192831


namespace hyperbola_eccentricity_l192_192710

open Real

def is_foci (F1 F2 : ℝ × ℝ) (h : (+,+)) := -- This is a placeholder definition, adjust as needed
 sorry

def is_line_perpendicular_real_axis (line : ℝ × ℝ → ℝ × ℝ) (F : ℝ × ℝ) := -- Another placeholder
 sorry

def intersects_hyperbola (P Q : ℝ × ℝ) (line : ℝ × ℝ → ℝ × ℝ) := -- Another placeholder
 sorry

def hyperbola_properties (a b c e : ℝ) :=
  c^2 = a^2 + b^2 ∧ b^2 = a^2 * (e^2 - 1) ∧ b^4 = 4 * a^2 * c^2

theorem hyperbola_eccentricity (F1 F2 : ℝ × ℝ) (P Q : ℝ × ℝ)
  (a b c e : ℝ) (h : Algebra ℝ ℝ ℝ → Prop):
  is_foci F1 F2 h →
  is_line_perpendicular_real_axis (λ t, (a, t)) F2 →
  intersects_hyperbola P Q (λ t, (a, t)) →
  angle P F1 Q = π / 2 →
  hyperbola_properties a b c e →
  e = 1 + sqrt 2 :=
by sorry

end hyperbola_eccentricity_l192_192710


namespace problem_inequality_goal_inequality_l192_192083

-- Define the function f
def f (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 1 then 1 else
  if h : x > 1 then x^2 else 0

-- Lean 4 statement
theorem problem_inequality (x : ℝ) (hx : x > 0) : 2 * f (x^2) ≥ x * f x + x := 
sorry

theorem goal_inequality (x : ℝ) (hx : x > 1) : f (x^3) ≥ x^2 := 
sorry

end problem_inequality_goal_inequality_l192_192083


namespace num_divisors_8_fact_l192_192860

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192860


namespace functional_eq_series_sum_l192_192980

theorem functional_eq_series_sum (f : ℕ → ℝ)
  (h_add : ∀ a b : ℕ, f(a + b) = f(a) * f(b))
  (h_one : f(1) = 1) :
  (∑ k in Finset.range 2004, f(k + 2) / f(k + 1)) = 2004 := 
by 
  sorry

end functional_eq_series_sum_l192_192980


namespace triangle_area_is_180_l192_192032

theorem triangle_area_is_180 {a b c : ℕ} (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) 
  (h_right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2 : ℚ) * a * b = 180 :=
by
  sorry

end triangle_area_is_180_l192_192032


namespace count_possible_integer_values_l192_192284

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192284


namespace parallel_condition_perpendicular_condition_l192_192738

theorem parallel_condition (x : ℝ) (a b : ℝ × ℝ) :
  (a = (x, x + 2)) → (b = (1, 2)) → a.1 * b.2 = a.2 * b.1 → x = 2 := 
sorry

theorem perpendicular_condition (x : ℝ) (a b : ℝ × ℝ) :
  (a = (x, x + 2)) → (b = (1, 2)) → ((a.1 - b.1) * b.1 + (a.2 - b.2) * b.2) = 0 → x = 1 / 3 :=
sorry

end parallel_condition_perpendicular_condition_l192_192738


namespace solve_cubic_eq_solve_quadratic_eq_l192_192181

theorem solve_cubic_eq (x : ℝ) (h : 8 * x^3 = 27) : x = 3 / 2 :=
sorry

theorem solve_quadratic_eq (x : ℝ) (h : (x - 2)^2 = 3) : x = sqrt 3 + 2 ∨ x = - sqrt 3 + 2 :=
sorry

end solve_cubic_eq_solve_quadratic_eq_l192_192181


namespace pos_divisors_8_factorial_l192_192907

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192907


namespace predict_height_of_10_year_old_l192_192030

theorem predict_height_of_10_year_old :
  ∀ x y : ℝ, (∀ x : ℝ, y = 7.2 * x + 74) → (x = 10) → y = 146 := 
by 
  intros x y model hx
  have : y = 7.2 * 10 + 74 := by rw [hx]
  linarith 

  sorry

end predict_height_of_10_year_old_l192_192030


namespace number_of_divisors_8_factorial_l192_192795

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192795


namespace combination_count_30_cents_l192_192741

theorem combination_count_30_cents : 
  ∃ (pennies nickels dimes : ℕ), 
    (pennies + 5 * nickels + 10 * dimes = 30) → 
    (15 = (card {n | ∃ p k d : ℕ, n = p + 5 * k + 10 * d ∧ n = 30})) :=
begin
  sorry
end

end combination_count_30_cents_l192_192741


namespace geometric_sequence_sum_l192_192707

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Conditions
axiom a1 : (a 1) = 1
axiom a2 : ∀ (n : ℕ), n ≥ 2 → 2 * a (n + 1) + 2 * a (n - 1) = 5 * a n
axiom increasing : ∀ (n m : ℕ), n < m → a n < a m

-- Target
theorem geometric_sequence_sum : S 5 = 31 := by
  sorry

end geometric_sequence_sum_l192_192707


namespace arithmetic_sequence_general_formula_l192_192123

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h₁ : a 1 = 39) (h₂ : a 1 + a 3 = 74) : 
  ∀ n, a n = 41 - 2 * n :=
sorry

end arithmetic_sequence_general_formula_l192_192123


namespace positive_divisors_of_8_factorial_l192_192829

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192829


namespace find_common_ratio_find_number_of_terms_l192_192622

-- Define the arithmetic-geometric sequence with sum S_n
def S (a q : ℚ) (n : ℕ) : ℚ :=
  a * (1 - q^n) / (1 - q)

open Classical

-- Problem statement 1: Finding common ratio q
theorem find_common_ratio
  (a : ℚ)
  (q : ℚ)
  (h_S2 : S a q 2)
  (h_S4 : S a q 4)
  (h_S3 : S a q 3)
  (h_arithmetic : 2 * h_S4 = h_S2 + h_S3) :
  q = -1/2 ∨ q = 1 :=
by sorry

-- Problem statement 2: Finding number of terms n given specific sum condition
theorem find_number_of_terms
  (a q : ℚ)
  (h_q : q = -1/2)
  (h_a1 : a = 4)
  (h_sum : S a q n = 21/8) :
  n = 6 :=
by sorry

end find_common_ratio_find_number_of_terms_l192_192622


namespace range_of_f_l192_192231

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem range_of_f : set.range (f ∘ (λ x, x)) = set.Ioo (-4 : ℝ) 5 := sorry

end range_of_f_l192_192231


namespace count_possible_integer_values_l192_192286

theorem count_possible_integer_values (x : ℕ) (h : ⌊real.sqrt x⌋ = 8) : 
  ∃ n, n = 17 := by
sorry

end count_possible_integer_values_l192_192286


namespace find_cd_product_l192_192182

open Complex

theorem find_cd_product :
  let u : ℂ := -3 + 4 * I
  let v : ℂ := 2 - I
  let c : ℂ := -5 + 5 * I
  let d : ℂ := -5 - 5 * I
  c * d = 50 :=
by
  sorry

end find_cd_product_l192_192182


namespace solution_set_is_circle_with_exclusion_l192_192676

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l192_192676


namespace count_ordered_triples_solution_number_of_ordered_triples_l192_192428

theorem count_ordered_triples_solution :
  ∃ x y n : ℕ, (n > 1) ∧ (x ^ n - y ^ n = 2 ^ 100) ∧ x > 0 ∧ y > 0 :=
sorry

theorem number_of_ordered_triples :
  (finset.card {p : ℕ × ℕ × ℕ | let (x, y, n) := p in n > 1 ∧ x ^ n - y ^ n = 2 ^ 100 ∧ x > 0 ∧ y > 0}) = 49 :=
sorry

end count_ordered_triples_solution_number_of_ordered_triples_l192_192428


namespace number_drawn_from_group_3_is_23_l192_192530

theorem number_drawn_from_group_3_is_23 
  (total_students : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (number_from_group_20 : ℕ) :
  (total_students = 180 ∧ total_groups = 20 ∧ 
   group_size = 9 ∧ number_from_group_20 = 176) → 
  ∃ x : ℕ, x = 23 :=
by {
  intros h,
  have total_students_eq : total_students = 180 := h.1,
  have total_groups_eq : total_groups = 20 := h.2.1,
  have group_size_eq : group_size = 9 := h.2.2.1,
  have number_from_group_20_eq : number_from_group_20 = 176 := h.2.2.2,
  
  -- Here, you can proceed to perform the exact arithmetic and logical steps
  -- based on the given conditions (if proof is required).

  use 23,
  sorry -- Proof steps are skipped as per the requirement.
}

end number_drawn_from_group_3_is_23_l192_192530


namespace num_divisors_8_fact_l192_192863

noncomputable def fac8 : ℕ := 8!

def num_divisors (n : ℕ) : ℕ :=
∏ p in (nat.factors n).to_finset, (nat.factor_multiset n p + 1)

theorem num_divisors_8_fact : num_divisors fac8 = 96 := by
  sorry

end num_divisors_8_fact_l192_192863


namespace num_divisors_8_factorial_l192_192924

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192924


namespace fish_to_corn_value_l192_192375

/-- In an island kingdom, five fish can be traded for three jars of honey, 
    and a jar of honey can be traded for six cobs of corn. 
    Prove that one fish is worth 3.6 cobs of corn. -/

theorem fish_to_corn_value (f h c : ℕ) (h1 : 5 * f = 3 * h) (h2 : h = 6 * c) : f = 18 * c / 5 := by
  sorry

end fish_to_corn_value_l192_192375


namespace num_possible_integer_values_x_l192_192337

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192337


namespace sqrt_floor_8_integer_count_l192_192317

theorem sqrt_floor_8_integer_count :
  {x : ℕ // (64 ≤ x) ∧ (x ≤ 80)}.card = 17 :=
by
  sorry

end sqrt_floor_8_integer_count_l192_192317


namespace complete_the_square_example_l192_192149

theorem complete_the_square_example : ∀ x m n : ℝ, (x^2 - 12 * x + 33 = 0) → 
  (x + m)^2 = n → m = -6 ∧ n = 3 :=
by
  sorry

end complete_the_square_example_l192_192149


namespace number_of_divisors_of_8_fact_l192_192893

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192893


namespace max_missed_questions_is_12_l192_192625

-- Define the number of questions and the passing rate
def totalQuestions : ℕ := 50
def passingRate : ℚ := 0.75

-- Calculate the minimum required correct answers
noncomputable def minCorrectAnswers : ℕ := 
  (passingRate * totalQuestions).ceil.to_nat

-- Define the maximum number of questions a student can miss and still pass
noncomputable def maxMissedQuestions : ℕ := totalQuestions - minCorrectAnswers

-- The theorem stating the maximum number of questions that can be missed
theorem max_missed_questions_is_12 : maxMissedQuestions = 12 := 
  sorry

end max_missed_questions_is_12_l192_192625


namespace number_of_isosceles_triangles_with_perimeter_25_l192_192746

def is_isosceles_triangle (a b : ℕ) : Prop :=
  a + 2 * b = 25 ∧ 2 * b > a ∧ a < 2 * b

theorem number_of_isosceles_triangles_with_perimeter_25 :
  (finset.filter (λ b, ∃ a, is_isosceles_triangle a b)
                 (finset.range 13)).card = 6 := by
sorry

end number_of_isosceles_triangles_with_perimeter_25_l192_192746


namespace number_of_divisors_8_factorial_l192_192790

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192790


namespace value_of_fraction_l192_192344

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end value_of_fraction_l192_192344


namespace number_of_divisors_of_8_fact_l192_192889

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192889


namespace positive_divisors_of_8_factorial_l192_192830

theorem positive_divisors_of_8_factorial : 
  ∃ (n : ℕ), n = 8! ∧ ∀ d : ℕ, d ∣ 40320 → 
  (finset.card (finset.filter (λ x, x ∣ 40320) (finset.range 40321)) = 96) :=
sorry

end positive_divisors_of_8_factorial_l192_192830


namespace num_possible_integer_values_l192_192300

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192300


namespace solution_set_circle_l192_192684

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l192_192684


namespace N_bisects_AH_l192_192554

open EuclideanGeometry

-- Definitions of triangle, orthocenter, altitude feet, and intersection points.
def acute_triangle (A B C H D E F P Q N : Point) : Prop :=
  Triangle ABC ∧
  is_orthocenter H ABC ∧
  feet_of_altitudes D E F A B C ∧
  altitudes_intersect_at H D E F ∧
  intersect DF (altitude_from B) = P ∧
  is_perpendicular (line_through P BC) ∧
  intersect_perpendicular BC P AB = Q ∧
  intersect EQ (altitude_from A) = N

-- Prove that N bisects the segment AH
theorem N_bisects_AH (A B C H D E F P Q N : Point)
  (hAcuteTriangle : acute_triangle A B C H D E F P Q N) : 
  midpoint N A H :=
sorry

end N_bisects_AH_l192_192554


namespace sum_of_x_values_l192_192667

def eqn (x y : ℤ) : Prop := 7 * x * y - 13 * x + 15 * y - 37 = 0

theorem sum_of_x_values : 
  (∑ x in {x : ℤ | ∃ y : ℤ, eqn x y}, x) = 4 :=
sorry

end sum_of_x_values_l192_192667


namespace gcd_1729_1337_l192_192050

theorem gcd_1729_1337 : Nat.gcd 1729 1337 = 7 := 
by
  sorry

end gcd_1729_1337_l192_192050


namespace values_of_a_l192_192081

noncomputable def condition_eq (a x : ℝ) : Prop :=
  7.68 * log a (sqrt (4 + x)) + 
  3 * log (a ^ 2) (4 - x) - 
  log (a ^ 4) ((16 - x ^ 2) ^ 2) = 2

theorem values_of_a (a x : ℝ) :
  condition_eq a x →
  (0 < a ∧ a < 1) ∨ (1 < a ∧ a < 2 * Real.sqrt 2) :=
sorry

end values_of_a_l192_192081


namespace complex_number_real_implies_m_eq_neg_one_l192_192986

theorem complex_number_real_implies_m_eq_neg_one 
    (z : ℂ)
    (m : ℝ)
    (h : z = (m + complex.I) / (1 - complex.I))
    (h_real : z.im = 0) :
    m = -1 :=
sorry

end complex_number_real_implies_m_eq_neg_one_l192_192986


namespace num_pos_divisors_fact8_l192_192771

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192771


namespace solution_set_is_circle_with_exclusion_l192_192675

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l192_192675


namespace domain_of_func_l192_192214

def func (x : ℝ) : ℝ := 1 / real.sqrt (real.logb 0.5 (2 * x - 1))

theorem domain_of_func :
  {x : ℝ | (2 * x - 1) > 0 ∧ real.logb 0.5 (2 * x - 1) > 0} = Ioo (1/2 : ℝ) 1 :=
by
  sorry

end domain_of_func_l192_192214


namespace number_of_divisors_8_factorial_l192_192791

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192791


namespace true_propositions_not_three_l192_192734

theorem true_propositions_not_three (p q : Prop) :
  let original := p → q,
      converse := q → p,
      inverse := ¬p → ¬q,
      contrapositive := ¬q → ¬p in
  ¬((to_bool original + to_bool converse + to_bool inverse + to_bool contrapositive) = 3) :=
sorry

end true_propositions_not_three_l192_192734


namespace num_possible_integer_values_x_l192_192334

theorem num_possible_integer_values_x (x : ℕ) (h : 8 ≤ Real.sqrt x ∧ Real.sqrt x < 9) : 
  ∃ n, n = 17 ∧ n = (Finset.card (Finset.filter (λ y, 64 ≤ y ∧ y < 81) (Finset.range 81))) :=
by
  sorry

end num_possible_integer_values_x_l192_192334


namespace equation_of_line_through_center_parallel_to_given_line_l192_192172

noncomputable def center_of_circle : (ℝ × ℝ) :=
  let c := Classical.choose (exists_center (x^2 + y^2 - 6*x + 4*y - 3 = 0))
  (c : ℝ × ℝ)

theorem equation_of_line_through_center_parallel_to_given_line 
  (h_circle : x^2 + y^2 - 6*x + 4*y - 3 = 0) (h_parallel : x + 2*y + 11 = 0) :
  (center_of_circle h_circle) = (3, -2) → (∀ (x y : ℝ), x + 2*y + 1 = 0) :=
sorry

end equation_of_line_through_center_parallel_to_given_line_l192_192172


namespace gosha_max_initial_number_l192_192244

theorem gosha_max_initial_number :
  ∀ (n : ℕ), (⌊⌊⌊real.sqrt (n : ℝ)⌋.sqrt⌋.sqrt⌋ = 1) → (n ≤ 255) :=
sorry

end gosha_max_initial_number_l192_192244


namespace prob_divisors_8_fact_l192_192807

theorem prob_divisors_8_fact : 
  let n := 8!
  let prime_factors := [(2,7), (3,2), (5,1), (7,1)]
  n = 40320 → 
  (prime_factors.reduce (λ acc p, acc * (p.snd + 1)) 1) = 96 :=
by
  intro h1,
  have h2 : prime_factors = [(2,7), (3,2), (5,1), (7,1)] := rfl,
  sorry

end prob_divisors_8_fact_l192_192807


namespace num_pos_divisors_fact8_l192_192778

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192778


namespace pos_divisors_8_factorial_l192_192900

open Nat

theorem pos_divisors_8_factorial : 
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (5 + 1) * (7 + 1)
  f = 2^7 * 3^2 * 5^1 * 7^1 →
  numberOfDivisors f = factorization :=
by
  let f := factorial 8
  let factorization := (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  sorry

end pos_divisors_8_factorial_l192_192900


namespace yellow_preference_l192_192158

theorem yellow_preference :
  let red := 80
  let blue := 150
  let green := 70
  let yellow := 200
  let purple := 50
  let total := red + blue + green + yellow + purple
  total = 550 →
  ((yellow * 100) / total : ℚ) ≈ 36 :=
by
  sorry

end yellow_preference_l192_192158


namespace minimum_value_of_angle_l192_192222

theorem minimum_value_of_angle
  (α : ℝ)
  (h : ∃ x y : ℝ, (x, y) = (Real.sin (2 * Real.pi / 3), Real.cos (2 * Real.pi / 3))) :
  α = 11 * Real.pi / 6 :=
sorry

end minimum_value_of_angle_l192_192222


namespace num_possible_integer_values_l192_192302

-- Define a predicate for the given condition
def satisfies_condition (x : ℕ) : Prop :=
  floor (Real.sqrt x) = 8

-- Main theorem statement
theorem num_possible_integer_values : 
  {x : ℕ // satisfies_condition x}.card = 17 :=
by
  sorry

end num_possible_integer_values_l192_192302


namespace correct_answer_is_C_l192_192542

-- Definitions based on the conditions
def A : Prop := ∀ (event : Event), 0 ≤ probability event ∧ probability event ≤ 1
def B : Prop := ∀ (freq : Frequency), freq is objective ∧ independent_of_trial_number freq
def C : Prop := ∀ (trials : ℕ) (freq : Frequency), as trials increases, freq tends to probability
def D : Prop := ∀ (prob : Probability), prob is random ∧ cannot be determined before an experiment

-- The problem we need to prove: that (C) is the correct statement
theorem correct_answer_is_C : C :=
by
  sorry

end correct_answer_is_C_l192_192542


namespace isosceles_right_triangle_perimeter_l192_192025

theorem isosceles_right_triangle_perimeter {a : ℝ} 
  (h_median_length : (a / 2) = 15) : 
  2 * (a / 2) * sqrt 2 + a = 30 + 30 * sqrt 2 :=
by
  sorry

end isosceles_right_triangle_perimeter_l192_192025


namespace functional_eq_series_sum_l192_192981

theorem functional_eq_series_sum (f : ℕ → ℝ)
  (h_add : ∀ a b : ℕ, f(a + b) = f(a) * f(b))
  (h_one : f(1) = 1) :
  (∑ k in Finset.range 2004, f(k + 2) / f(k + 1)) = 2004 := 
by 
  sorry

end functional_eq_series_sum_l192_192981


namespace monotonically_increasing_intervals_max_and_min_values_l192_192729

-- Define the function
def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin (x / 2) * cos (x / 2) + 2 * cos (x / 2) ^ 2

-- Problem (I): Prove that the function is monotonically increasing in the given intervals
theorem monotonically_increasing_intervals (k : ℤ) :
  ∀ x, 2 * k * π - (2 * π / 3) ≤ x ∧ x ≤ 2 * k * π + (π / 3) → f x ≤ f (x + 1) :=
sorry -- proof

-- Define the interval [ -π, 0]
def interval : Set ℝ := Set.Icc (-π) 0

-- Problem (II): Prove the max and min values within the interval [-π, 0]
theorem max_and_min_values :
  (∀ x ∈ interval, f x ≤ f 0) ∧ (∀ x ∈ interval, f (-2 * π / 3) ≤ f x) :=
sorry -- proof

end monotonically_increasing_intervals_max_and_min_values_l192_192729


namespace sum_of_new_dimensions_l192_192106

theorem sum_of_new_dimensions (s : ℕ) (h₁ : s^2 = 36) (h₂ : s' = s - 1) : s' + s' + s' = 15 :=
sorry

end sum_of_new_dimensions_l192_192106


namespace f_period_2_f3_equals_2_l192_192082

-- Define the period of the function
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f(x) = f(x + T)

-- Define the function f as stated in the conditions
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ set.Icc 0 1 then 2 ^ x else sorry  -- Function definition for [0,1]

-- Prove the specific case for f(3)
theorem f_period_2_f3_equals_2 : is_periodic f 2 → (∀ x ∈ set.Icc 0 1, f x = 2 ^ x) → f 3 = 2 :=
  by
  -- Assuming that f satisfies periodicity and functional conditions, we conclude
  sorry

end f_period_2_f3_equals_2_l192_192082


namespace max_salary_single_player_l192_192588

theorem max_salary_single_player
  (n : ℕ) (k : ℕ) (min_salary : ℕ) (total_salary : ℕ) (player_salary : ℕ → ℕ)
  (h_team_size : n = 25)
  (h_min_salary : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → player_salary i ≥ min_salary)
  (h_salary_sum : ∑ i in finset.range n, player_salary i = total_salary)
  (h_min_salary_value : min_salary = 20000)
  (h_total_salary_value : total_salary = 800000)
  (h_other_players_salary : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 24 → player_salary i = 20000) :
  ∃ x, (x ≥ 20000) ∧ (player_salary 25 = x) ∧ x = 320000 :=
by
  sorry

end max_salary_single_player_l192_192588


namespace bank_robbery_participants_l192_192602

variables (Alexey Boris Veniamin Grigory : Prop)

axiom h1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom h2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom h3 : Grigory → Boris
axiom h4 : Boris → (Alexey ∨ Veniamin)

theorem bank_robbery_participants : Alexey ∧ Boris ∧ Grigory :=
by
  sorry

end bank_robbery_participants_l192_192602


namespace part_a_l192_192429

variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)

def b_n (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), (1 - a (k - 1) / a k) / real.sqrt (a k)

axiom a_nonneg : ∀ n : ℕ, 0 ≤ a n
axiom a_seq : ∀ n m : ℕ, n ≤ m → a n ≤ a m
axiom a_0_1 : a 0 = 1

theorem part_a (n : ℕ) : 0 ≤ b_n n ∧ b_n n < 2 := 
sorry

end part_a_l192_192429


namespace median_interval_of_histogram_l192_192146

theorem median_interval_of_histogram (s1 s2 s3 s4 s5 s6 : ℕ)
  (h1 : s1 = 15) (h2 : s2 = 20) (h3 : s3 = 25) (h4 : s4 = 18) (h5 : s5 = 12) (h6 : s6 = 10) :
  let total_students := s1 + s2 + s3 + s4 + s5 + s6 in
  total_students = 100 → 
  let median_position := (total_students + 1) / 2 in
  let cumulative_sum_first_three_intervals := s1 + s2 + s3 in
  50 ≤ median_position ∧ median_position ≤ cumulative_sum_first_three_intervals :=
by {
  sorry
}

end median_interval_of_histogram_l192_192146


namespace sum_of_longest_altitudes_l192_192966

theorem sum_of_longest_altitudes (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c = 15) :
  let h1 := a,
      h2 := b,
      h := (a * b) / c in
  h1 + h2 = 21 := by
{
  sorry
}

end sum_of_longest_altitudes_l192_192966


namespace integer_values_of_x_l192_192291

theorem integer_values_of_x (x : ℕ) (h : (⌊ sqrt x ⌋ : ℕ) = 8) :
  ∃ (n : ℕ), n = 17 ∧ ∀ (x : ℕ), 64 ≤ x ∧ x < 81 → ∃ (k : ℕ), k = 17 :=
by sorry

end integer_values_of_x_l192_192291


namespace quadrilateral_divided_into_trapezoids_l192_192459

theorem quadrilateral_divided_into_trapezoids (A B C D M N K : Point) (quadrilateral : quadrilateral ABCD) 
  (BM_parallel_AD : parallel (Line B M) (Line A D))
  (MN_parallel_BC : parallel (Line M N) (Line B C))
  (MK_parallel_CD : parallel (Line M K) (Line C D)) :
  exists T₁ T₂ T₃ : trapezoid, divides_into_three_trapezoids quadrilateral T₁ T₂ T₃ := 
sorry

end quadrilateral_divided_into_trapezoids_l192_192459


namespace num_divisors_8_factorial_l192_192762

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def num_divisors (n : ℕ) : ℕ :=
  (finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem num_divisors_8_factorial : num_divisors (factorial 8) = 96 := by
  sorry

end num_divisors_8_factorial_l192_192762


namespace value_of_y_l192_192974

theorem value_of_y (x y : ℝ) (h1 : x ^ (2 * y) = 81) (h2 : x = 9) : y = 1 :=
sorry

end value_of_y_l192_192974


namespace pure_imaginary_complex_l192_192359

theorem pure_imaginary_complex (a : ℝ) (i : ℂ) (h : i * i = -1) (p : (1 + a * i) / (1 - i) = (0 : ℂ) + b * i) :
  a = 1 := 
sorry

end pure_imaginary_complex_l192_192359


namespace num_divisors_of_8_factorial_l192_192879

theorem num_divisors_of_8_factorial : 
  ∀ (n : ℕ), 
  n = 8! → 
  ( ∃ (p1 p2 p3 p4 : ℕ),
    p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 7 ∧
    n = 2^7 * 3^2 * 5^1 * 7^1 ) →
  ( ∏ (d ∈ finset.divisors n), d) = 96 := 
by 
  intros n h1 h2
  cases' h2 with p1 h2
  cases' h2 with p2 h2
  cases' h2 with p3 h2
  cases' h2 with p4 h2
  cases' h2 with hp1 h2
  cases' h2 with hp2 h2
  cases' h2 with hp3 h2
  cases' h2 with hp4 hn
  have factorial_8_eq : n = 40320 := by sorry
  have prime_factorization :
    40320 = 2^7 * 3^2 * 5^1 * 7^1 := by sorry
  have num_divisors_formula :
    (∏ (d ∈ finset.divisors 40320), d) = 96 := by sorry
  rw [h1] at factorial_8_eq
  rw [←factorial_8_eq, prime_factorization, num_divisors_formula]
  exact sorry -- Proof to complete

end num_divisors_of_8_factorial_l192_192879


namespace addition_problem_l192_192143

theorem addition_problem :
  (C A S H M O S2 I D E : ℕ) →
  0 < C ∧ 0 < M ∧ 0 < O →
  C ≠ M ∧ C ≠ O ∧ M ≠ O →
  (C * 1000 + A * 100 + S * 10 + H) + (M * 10 + E) = 
  O * 10000 + S2 * 1000 + I * 100 + D * 10 + E →
  0 ∈ ∅ :=
by
  sorry

end addition_problem_l192_192143


namespace num_divisors_8_factorial_l192_192923

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192923


namespace num_pos_divisors_fact8_l192_192780

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192780


namespace number_of_hens_is_correct_l192_192110

-- Definitions of given conditions
def total_chickens := 500
def black_copper_marans := 0.25 * total_chickens
def rhode_island_reds := 0.40 * total_chickens
def leghorns := 0.35 * total_chickens

def black_copper_marans_hens := 0.65 * black_copper_marans
def rhode_island_reds_hens := 0.55 * rhode_island_reds
def leghorns_hens := 0.60 * leghorns

-- Theorem statement to prove number of hens for each breed
theorem number_of_hens_is_correct :
  black_copper_marans_hens = 81 ∧
  rhode_island_reds_hens = 110 ∧ 
  leghorns_hens = 105 :=
sorry

end number_of_hens_is_correct_l192_192110


namespace num_divisors_fact8_l192_192951

-- Helper function to compute factorial
def factorial (n : ℕ) : ℕ :=
  if hn : n = 0 then 1 else n * factorial (n - 1)

-- Defining 8!
def fact8 := factorial 8

-- Prime factorization related definitions
def prime_factors_8! := (2 ^ 7) * (3 ^ 2) * (5 ^ 1) * (7 ^ 1)
def number_of_divisors (n : ℕ) := n.factors.length

-- Statement of the theorem
theorem num_divisors_fact8 : number_of_divisors fact8 = 96 := 
sorry

end num_divisors_fact8_l192_192951


namespace num_pos_divisors_fact8_l192_192772

theorem num_pos_divisors_fact8 : 
  let fact8 := 8.factorial in -- Define the factorial of 8
  let prime_decomp := (2^7) * (3^2) * (5^1) * (7^1) in -- Define the prime decomposition
  fact8 = prime_decomp → 
  (List.prod (([7, 2, 1, 1].map (λ e => e + 1)))) = 96 := -- Calculate the number of divisors from prime powers
by 
  intro fact8 
  intro prime_decomp 
  intro h
  sorry

end num_pos_divisors_fact8_l192_192772


namespace intersection_domains_l192_192728

def domain_f : Set ℝ := {x : ℝ | x < 1}
def domain_g : Set ℝ := {x : ℝ | x > -1}

theorem intersection_domains : {x : ℝ | x < 1} ∩ {x : ℝ | x > -1} = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end intersection_domains_l192_192728


namespace problem_solution_l192_192712

-- Define f(x) under the given conditions
def f (x : ℝ) : ℝ := 
  if x ∈ set.Icc 0 2 then (Real.exp x - 1)
  else if x < 0 then -f (-x)
  else f (x - 2 * (Real.floor (x / 2) : ℝ))

-- State the theorem
theorem problem_solution : f 2013 + f (-2014) = Real.exp 1 - 1 :=
by {
  sorry
}

end problem_solution_l192_192712


namespace complete_decomposition_time_l192_192693

-- Define the decomposition rate function
def decomposition_rate (a b t : ℝ) : ℝ := a * b ^ t

-- Given conditions
axiom condition1 : decomposition_rate a b 6 = 0.05
axiom condition2 : decomposition_rate a b 12 = 0.1

-- The goal to prove
theorem complete_decomposition_time (a b : ℝ) (h1 : decomposition_rate a b 6 = 0.05) (h2 : decomposition_rate a b 12 = 0.1) : 
  t = 32 :=
sorry

end complete_decomposition_time_l192_192693


namespace num_divisors_8_factorial_l192_192914

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem num_divisors_8_factorial :
  let n := 8 in
  let fact_8 := factorial n in
  let prime_factors := (2^7 * 3^2 * 5^1 * 7^1) in
  fact_8 = prime_factors → 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 :=
by sorry

end num_divisors_8_factorial_l192_192914


namespace exists_100_distinct_sums_l192_192156

theorem exists_100_distinct_sums : ∃ (a : Fin 100 → ℕ), (∀ i j k l : Fin 100, i ≠ j → k ≠ l → (i, j) ≠ (k, l) → a i + a j ≠ a k + a l) ∧ (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 25000) :=
by
  sorry

end exists_100_distinct_sums_l192_192156


namespace evaluate_log_l192_192160

theorem evaluate_log (
  h1 : 64 = 4^3,
  h2 : sqrt 16 = 4
) : log 4 (64 * sqrt 16) = 4 :=
by 
sorry

end evaluate_log_l192_192160


namespace evaluate_x_squared_plus_y_squared_l192_192216

theorem evaluate_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : x + y = 12)
  (h2 : 3 * x + y = 20) :
  x^2 + y^2 = 80 := by
  sorry

end evaluate_x_squared_plus_y_squared_l192_192216


namespace num_positive_divisors_8_factorial_l192_192939

theorem num_positive_divisors_8_factorial :
  ∃ t, t = (8! : ℕ) ∧ t = (2^7 * 3^2 * 5^1 * 7^1 : ℕ) ∧
  ∃ d, number_of_positive_divisors(8!) = d ∧ d = 96 :=
by
  sorry

end num_positive_divisors_8_factorial_l192_192939


namespace number_of_divisors_8_factorial_l192_192801

theorem number_of_divisors_8_factorial :
  let n := (8!)
  let prime_fact := 2^7 * 3^2 * 5 * 7
  (40320 = prime_fact) →
  (8 * 3 * 2 * 2 = 96) →
  ∃ d : ℕ, nat.num_divisors n = d ∧ d = 96 :=
by
  intro n prime_fact
  intros h_fact h_divisors
  use 96
  rw [← h_fact, ← h_divisors]
  sorry

end number_of_divisors_8_factorial_l192_192801


namespace smallest_five_digit_divisible_by_15_32_54_l192_192535

theorem smallest_five_digit_divisible_by_15_32_54 : 
  ∃ n : ℤ, n >= 10000 ∧ n < 100000 ∧ (15 ∣ n) ∧ (32 ∣ n) ∧ (54 ∣ n) ∧ n = 17280 :=
  sorry

end smallest_five_digit_divisible_by_15_32_54_l192_192535


namespace probability_of_correct_match_l192_192575

noncomputable def probability_correct_match : ℚ :=
  1 / (Finset.univ : Finset (Equiv.Perm (Fin 4))).card

theorem probability_of_correct_match :
  probability_correct_match = 1 / 24 := by
  sorry

end probability_of_correct_match_l192_192575


namespace circumference_of_tire_l192_192357

theorem circumference_of_tire (rotations_per_minute : ℕ) (speed_kmh : ℕ) 
  (h1 : rotations_per_minute = 400) (h2 : speed_kmh = 72) :
  let speed_mpm := speed_kmh * 1000 / 60
  let circumference := speed_mpm / rotations_per_minute
  circumference = 3 :=
by
  sorry

end circumference_of_tire_l192_192357


namespace max_dot_product_l192_192367

theorem max_dot_product (a b : ℝ) (h : a^2 + b^2 - a * b = 3) :
  (∀ CA CB : ℝ, (CA = a * b / 2) → CA * CB ≤ 3 / 2) :=
begin
  sorry
end

end max_dot_product_l192_192367


namespace teacher_sequences_l192_192038

theorem teacher_sequences : 
  let original_sequence := [1,4,7,10,13,16,19,22,25,28,31,34,37,40,43],
      remaining_numbers := [1,19,43],
      divisors := [1,2,3,6]
  in ∃ d ∈ divisors, ∀ k m, 
      (19 = 1 + (k-1)*d ∧ 43 = 19 + (m-k)*d) -> d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 6 :=
sorry

end teacher_sequences_l192_192038


namespace range_of_a_l192_192194

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * real.log x + 0.5 * x^2

theorem range_of_a (a : ℝ) (h₁ : 0 < a)
  (h₂ : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 0 < x₁ ∧ 0 < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 2) :
  1 ≤ a :=
sorry

end range_of_a_l192_192194


namespace rental_space_cost_l192_192120

theorem rental_space_cost :
  (∀ (earnings_per_day profit : ℕ) (days_in_week : ℕ) (C : ℕ),
      (earnings_per_day = 8) →
      (profit = 36) →
      (days_in_week = 7) →
      (profit = earnings_per_day * days_in_week - C) →
      (C = 20)) :=
by {
  intros earnings_per_day profit days_in_week C earnings_per_day_eq profit_eq days_in_week_eq profit_def,
  rw [earnings_per_day_eq, profit_eq, days_in_week_eq] at profit_def,
  linarith,
}

end rental_space_cost_l192_192120


namespace geometry_problem_l192_192238

noncomputable def EquationOfLineAB (A B : ℝ × ℝ) : ℝ × ℝ × ℝ :=
let (x1, y1) := A
let (x2, y2) := B
(3, -1, 5)

noncomputable def LengthOfMedianAM (A M : ℝ × ℝ) : ℝ :=
let (x1, y1) := A
let (x2, y2) := M
Real.sqrt $ (x2 - x1)^2 + (y2 - y1)^2

theorem geometry_problem 
  (A B C : ℝ × ℝ)
  (hA : A = (0, 5))
  (hB : B = (-2, -1))
  (hC : C = (4, 3))
  (M : ℝ × ℝ)
  (hM : M = ((fst B + fst C) / 2, (snd B + snd C) / 2))
:
  EquationOfLineAB A B = (3, -1, 5) ∧ LengthOfMedianAM A M = Real.sqrt 17 := by
  sorry

end geometry_problem_l192_192238


namespace value_of_fraction_l192_192346

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end value_of_fraction_l192_192346


namespace find_y_l192_192354

theorem find_y (x y: ℝ) (h1: x = 680) (h2: 0.25 * x = 0.20 * y - 30) : y = 1000 :=
by 
  sorry

end find_y_l192_192354


namespace range_of_x_in_sqrt_x_minus_2_l192_192493

theorem range_of_x_in_sqrt_x_minus_2 :
  {x : ℝ // x ≥ 2} = {x : ℝ // ∃ y : ℝ, y = sqrt (x - 2)} :=
by
  sorry

end range_of_x_in_sqrt_x_minus_2_l192_192493


namespace trapezoid_and_inscribed_circle_ratio_trapezoid_and_circumscribed_circle_ratio_l192_192079

-- Define the given variables and conditions
variables {r R : ℝ} {a b c : ℝ} {α : ℝ}
variable is_isosceles_trapezoid : True
variable (BM_eq_h : BM = 2 * r)
variable (height_eq_h : h = 2 * r)
variable (α_eq : α = 30 * Real.pi / 180) -- converting degrees to radians
variable (side_condition_1 : 2 * c = a + b)
variable (side_condition_2 : a + b = 8 * r)
variable (sin_eq : Real.sin α = 1 / 2)
variable (h_area_eq : S = (a + b) * r)
variable (inscribed_area_eq : S1 = π * r^2)
variable (circumscribed_area_eq : S2 = π * (c * Real.sqrt 5 / 2)^2)
variable (trapezoid_area_eq : S = c^2 / 2)

-- Proof statements
theorem trapezoid_and_inscribed_circle_ratio (h_area_eq_inscribed : S = (a + b) * r) (inscribed_area_eq : S1 = π * r^2) 
(h_s : S = 8 * r^2) : S / S1 = 8 / π :=
by
  have h_s1 : S1 = π * r ^ 2, from inscribed_area_eq,
  have h_s2 : S = 8 * r ^ 2, from h_s,
  rw [h_s2, h_s1],
  field_simp,
  linarith [h_s1, h_s2]

theorem trapezoid_and_circumscribed_circle_ratio (circumscribed_area_eq : S2 = π * (c * Real.sqrt 5 / 2)^2)
(trapezoid_area_eq : S = c^2 / 2) : S / S2 = 2 / (5 * π) :=
by
  have S2_eq : S2 = π * (c * Real.sqrt 5 / 2)^2, from circumscribed_area_eq,
  have S_eq : S = c ^ 2 / 2, from trapezoid_area_eq,
  rw [S_eq, S2_eq],
  field_simp,
  linarith [S_eq, S2_eq]


end trapezoid_and_inscribed_circle_ratio_trapezoid_and_circumscribed_circle_ratio_l192_192079


namespace total_votes_l192_192074

theorem total_votes (V : ℝ) (h1 : 0.35 * V + (0.35 * V + 1650) = V) : V = 5500 := 
by 
  sorry

end total_votes_l192_192074


namespace pirate_coins_l192_192103

theorem pirate_coins (x : ℕ) (h : x % (10^9 / 362880) = 0) : 
  let n := 9! in
  n = 362880 :=
by
  let ratio := (10 : ℕ)^9 / n
  have h_ratio : ratio = 10^9 / 362880 := by sorry
  have h_mod : x % ratio = 0 := h
  have h_n : n = 362880 := by sorry
  exact h_n

end pirate_coins_l192_192103


namespace solution_count_l192_192673

def equation (x : ℝ) : Prop :=
  cos (6 * x) + cos (4 * x) ^ 2 + cos (3 * x) ^ 3 + cos (2 * x) ^ 4 = 0

def condition (x : ℝ) : Prop :=
  -π ≤ x ∧ x ≤ π ∧ cos (3 * x) ≤ 1 / 2

def valid_solution_count : ℕ := 3

theorem solution_count :
  {x : ℝ | equation x ∧ condition x}.finite.card = valid_solution_count :=
by
  sorry

end solution_count_l192_192673


namespace max_area_triangle_l192_192418

theorem max_area_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : Real.sin B ^ 2 + Real.sin C ^ 2 - Real.sin A ^ 2 = Real.sin B * Real.sin C)
  (h2 : b * Real.cos C + c * Real.cos B = 2)
  (h3 : a = 2)  -- This is an implicit condition derived from the steps
  (h4 : B + C + A = π)
  (h5 : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) -- Angles are in proper range
  : ∃ S : ℝ, S = (sqrt 3) ∧ S = (1 / 2) * b * c * (Real.sin A) :=
begin
  sorry
end

end max_area_triangle_l192_192418


namespace cricket_lovers_l192_192996

theorem cricket_lovers (B C Both : ℕ) (hB : B = 7) (hBoth : Both = 3) (hUnion : B + C - Both = 9) : C = 5 := by
  -- Use the given conditions to form the proof statement.
  have h_eq : 9 = 7 + C - 3 := by rw [hB, hBoth]; exact hUnion
  -- Solve for C and conclude that C = 5.
  sorry

end cricket_lovers_l192_192996


namespace GeetaSpeed_is_correct_l192_192440

noncomputable def LataSpeed_kmh : ℝ := 4.2
noncomputable def TrackLength_m : ℝ := 640
noncomputable def MeetingTime_min : ℝ := 4.8

theorem GeetaSpeed_is_correct : 
  let LataSpeed_mmin := (LataSpeed_kmh * 1000 / 60)
  let GeetaDistance_m := TrackLength_m - (LataSpeed_mmin * MeetingTime_min)
  (GeetaDistance_m / MeetingTime_min * 60 / 1000) ≈ 3.8 := 
by
  sorry

end GeetaSpeed_is_correct_l192_192440


namespace number_of_divisors_of_8_fact_l192_192896

theorem number_of_divisors_of_8_fact: 
  let n := 8
  let fact := Nat.factorial n
  fact = (2^7) * (3^2) * (5^1) * (7^1) -> 
  (7 + 1) * (2 + 1) * (1 + 1) * (1 + 1) = 96 := by
  sorry

end number_of_divisors_of_8_fact_l192_192896


namespace root_interval_l192_192047

def f (x : ℝ) : ℝ := x^3 - x - 1

theorem root_interval : 
  f 1 < 0 ∧ f 1.25 < 0 ∧ f 1.5 > 0 ∧ f 1.75 > 0 ∧ f 2 > 0 →
  ∃ x, x ∈ set.Icc 1.25 1.5 ∧ f x = 0 :=
by
  intro h
  sorry

end root_interval_l192_192047
