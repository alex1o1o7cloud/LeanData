import Mathlib

namespace total_amount_is_175_l676_67687

noncomputable def calc_total_amount (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
x + y + z

theorem total_amount_is_175 (x y z : ℝ) 
  (h1 : y = 0.45 * x)
  (h2 : z = 0.30 * x)
  (h3 : y = 45) :
  calc_total_amount x y z = 175 :=
by
  -- sorry to skip the proof
  sorry

end total_amount_is_175_l676_67687


namespace complex_sum_l676_67617

noncomputable def omega : ℂ := sorry
axiom omega_power_five : omega^5 = 1
axiom omega_not_one : omega ≠ 1

theorem complex_sum :
  (omega^20 + omega^25 + omega^30 + omega^35 + omega^40 + omega^45 + omega^50 + omega^55 + omega^60 + omega^65 + omega^70) = 11 :=
by
  sorry

end complex_sum_l676_67617


namespace largest_possible_last_digit_l676_67663

theorem largest_possible_last_digit (D : Fin 3003 → Nat) :
  D 0 = 2 →
  (∀ i : Fin 3002, (10 * D i + D (i + 1)) % 17 = 0 ∨ (10 * D i + D (i + 1)) % 23 = 0) →
  D 3002 = 9 :=
sorry

end largest_possible_last_digit_l676_67663


namespace total_dolls_l676_67631

-- Definitions based on the given conditions
def grandmother_dolls : Nat := 50
def sister_dolls : Nat := grandmother_dolls + 2
def rene_dolls : Nat := 3 * sister_dolls

-- Statement we want to prove
theorem total_dolls : grandmother_dolls + sister_dolls + rene_dolls = 258 := by
  sorry

end total_dolls_l676_67631


namespace dot_product_is_five_l676_67693

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 3)

-- Define the condition that involves a and b
def condition : Prop := 2 • a - b = (3, 1)

-- Prove that the dot product of a and b equals 5 given the condition
theorem dot_product_is_five : condition → (a.1 * b.1 + a.2 * b.2) = 5 :=
by
  sorry

end dot_product_is_five_l676_67693


namespace math_problem_l676_67630

noncomputable def m : ℕ := 294
noncomputable def n : ℕ := 81
noncomputable def d : ℕ := 3

axiom circle_radius (r : ℝ) : r = 42
axiom chords_length (l : ℝ) : l = 78
axiom intersection_distance (d : ℝ) : d = 18

theorem math_problem :
  let m := 294
  let n := 81
  let d := 3
  m + n + d = 378 :=
by {
  -- Proof omitted
  sorry
}

end math_problem_l676_67630


namespace roots_equation_sum_and_product_l676_67672

theorem roots_equation_sum_and_product (x1 x2 : ℝ) (h1 : x1 ^ 2 - 3 * x1 - 5 = 0) (h2 : x2 ^ 2 - 3 * x2 - 5 = 0) :
  x1 + x2 - x1 * x2 = 8 :=
sorry

end roots_equation_sum_and_product_l676_67672


namespace find_number_l676_67641

theorem find_number (n : ℝ) : (1 / 2) * n + 6 = 11 → n = 10 := by
  sorry

end find_number_l676_67641


namespace range_of_m_l676_67608

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := -x^2 + x + m + 2

theorem range_of_m (m : ℝ) : 
  (∃! x : ℤ, f x m ≥ |x|) ↔ -2 ≤ m ∧ m < -1 :=
by
  sorry

end range_of_m_l676_67608


namespace sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l676_67696

-- Part a) Prove that if a sequence has a limit, then it is bounded.
theorem sequence_with_limit_is_bounded (x : ℕ → ℝ) (x0 : ℝ) (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) :
  ∃ C, ∀ n, |x n| ≤ C := by
  sorry

-- Part b) Is the converse statement true?
theorem bounded_sequence_does_not_imply_limit :
  ∃ (x : ℕ → ℝ), (∃ C, ∀ n, |x n| ≤ C) ∧ ¬(∃ x0, ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) := by
  sorry

end sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l676_67696


namespace aleksey_divisible_l676_67657

theorem aleksey_divisible
  (x y a b S : ℤ)
  (h1 : x + y = S)
  (h2 : S ∣ (a * x + b * y)) :
  S ∣ (b * x + a * y) := 
sorry

end aleksey_divisible_l676_67657


namespace calculate_expression_l676_67636

theorem calculate_expression :
  let s1 := 3 + 6 + 9
  let s2 := 4 + 8 + 12
  s1 = 18 → s2 = 24 → (s1 / s2 + s2 / s1) = 25 / 12 :=
by
  intros
  sorry

end calculate_expression_l676_67636


namespace odd_expression_proof_l676_67642

theorem odd_expression_proof (n : ℤ) : Odd (n^2 + n + 5) :=
by 
  sorry

end odd_expression_proof_l676_67642


namespace pow_congruence_modulus_p_squared_l676_67699

theorem pow_congruence_modulus_p_squared (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) (h : a ≡ b [ZMOD p]) : a^p ≡ b^p [ZMOD p^2] :=
sorry

end pow_congruence_modulus_p_squared_l676_67699


namespace roots_negative_reciprocals_l676_67670

theorem roots_negative_reciprocals (a b c r s : ℝ) (h1 : a * r^2 + b * r + c = 0)
    (h2 : a * s^2 + b * s + c = 0) (h3 : r = -1 / s) (h4 : s = -1 / r) :
    a = -c :=
by
  -- Insert clever tricks to auto-solve or reuse axioms here
  sorry

end roots_negative_reciprocals_l676_67670


namespace sequence_2007th_number_l676_67607

-- Defining the sequence according to the given rule
def a (n : ℕ) : ℕ := 2 ^ n

theorem sequence_2007th_number : a 2007 = 2 ^ 2007 :=
by
  -- Proof is omitted
  sorry

end sequence_2007th_number_l676_67607


namespace max_value_of_f_l676_67660

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^3 + Real.cos (2 * x) - (Real.cos x)^2 - Real.sin x

theorem max_value_of_f :
  ∃ x : ℝ, f x = 5 / 27 ∧ ∀ y : ℝ, f y ≤ 5 / 27 :=
sorry

end max_value_of_f_l676_67660


namespace mother_stickers_given_l676_67615

-- Definitions based on the conditions
def initial_stickers : ℝ := 20.0
def bought_stickers : ℝ := 26.0
def birthday_stickers : ℝ := 20.0
def sister_stickers : ℝ := 6.0
def total_stickers : ℝ := 130.0

-- Statement of the problem to be proved in Lean 4.
theorem mother_stickers_given :
  initial_stickers + bought_stickers + birthday_stickers + sister_stickers + 58.0 = total_stickers :=
by
  sorry

end mother_stickers_given_l676_67615


namespace sandwiches_per_day_l676_67644

theorem sandwiches_per_day (S : ℕ) 
  (h1 : ∀ n, n = 4 * S)
  (h2 : 7 * 4 * S = 280) : S = 10 := 
by
  sorry

end sandwiches_per_day_l676_67644


namespace find_y_l676_67662

def star (a b : ℝ) : ℝ := 4 * a + 2 * b

theorem find_y (y : ℝ) : star 3 (star 4 y) = -2 → y = -11.5 :=
by
  sorry

end find_y_l676_67662


namespace identify_solids_with_identical_views_l676_67633

def has_identical_views (s : Type) : Prop := sorry

def sphere : Type := sorry
def triangular_pyramid : Type := sorry
def cube : Type := sorry
def cylinder : Type := sorry

theorem identify_solids_with_identical_views :
  (has_identical_views sphere) ∧
  (¬ has_identical_views triangular_pyramid) ∧
  (has_identical_views cube) ∧
  (¬ has_identical_views cylinder) :=
sorry

end identify_solids_with_identical_views_l676_67633


namespace product_of_roots_of_polynomial_l676_67648

theorem product_of_roots_of_polynomial : 
  ∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x^2 - x - 34 = 0) ∧ (a * b = -34) :=
by
  sorry

end product_of_roots_of_polynomial_l676_67648


namespace cubic_root_sum_cubed_l676_67655

theorem cubic_root_sum_cubed
  (p q r : ℂ)
  (h1 : 3 * p^3 - 9 * p^2 + 27 * p - 6 = 0)
  (h2 : 3 * q^3 - 9 * q^2 + 27 * q - 6 = 0)
  (h3 : 3 * r^3 - 9 * r^2 + 27 * r - 6 = 0)
  (hpq : p ≠ q)
  (hqr : q ≠ r)
  (hrp : r ≠ p) :
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 585 := 
  sorry

end cubic_root_sum_cubed_l676_67655


namespace kostyas_table_prime_l676_67604

theorem kostyas_table_prime (n : ℕ) (h₁ : n > 3) 
    (h₂ : ¬ ∃ r s : ℕ, r ≥ 3 ∧ s ≥ 3 ∧ n = r * s - (r + s)) : 
    Prime (n + 1) := 
sorry

end kostyas_table_prime_l676_67604


namespace largest_positive_integer_n_l676_67622

 

theorem largest_positive_integer_n (n : ℕ) :
  (∀ p : ℕ, Nat.Prime p ∧ 2 < p ∧ p < n → Nat.Prime (n - p)) →
  ∀ m : ℕ, (∀ q : ℕ, Nat.Prime q ∧ 2 < q ∧ q < m → Nat.Prime (m - q)) → n ≥ m → n = 10 :=
by
  sorry

end largest_positive_integer_n_l676_67622


namespace smallest_positive_m_condition_l676_67661

theorem smallest_positive_m_condition
  (p q : ℤ) (m : ℤ) (h_prod : p * q = 42) (h_diff : |p - q| ≤ 10) 
  (h_roots : 15 * (p + q) = m) : m = 195 :=
sorry

end smallest_positive_m_condition_l676_67661


namespace intersection_of_lines_l676_67691

theorem intersection_of_lines :
  ∃ x y : ℚ, (8 * x - 3 * y = 9) ∧ (6 * x + 2 * y = 20) ∧ (x = 39 / 17) ∧ (y = 53 / 17) :=
by
  sorry

end intersection_of_lines_l676_67691


namespace heal_time_l676_67616

theorem heal_time (x : ℝ) (hx_pos : 0 < x) (h_total : 2.5 * x = 10) : x = 4 := 
by {
  -- Lean proof will be here
  sorry
}

end heal_time_l676_67616


namespace rectangle_ratio_l676_67645

noncomputable def ratio_of_length_to_width (w : ℝ) : ℝ :=
  40 / w

theorem rectangle_ratio (w : ℝ) 
  (hw1 : 35 * (w + 5) = 40 * w + 75) : 
  ratio_of_length_to_width w = 2 :=
by
  sorry

end rectangle_ratio_l676_67645


namespace part_a_part_b_l676_67678

-- Part (a)
theorem part_a (f : ℚ → ℝ) (h_add : ∀ x y : ℚ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℚ, f (x * y) = f x * f y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = 0) :=
sorry

-- Part (b)
theorem part_b (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℝ, f (x * y) = f x * f y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 0) :=
sorry

end part_a_part_b_l676_67678


namespace conveyor_belt_sampling_l676_67671

noncomputable def sampling_method (interval : ℕ) (total_items : ℕ) : String :=
  if interval = 5 ∧ total_items > 0 then "systematic sampling" else "unknown"

theorem conveyor_belt_sampling :
  ∀ (interval : ℕ) (total_items : ℕ),
  interval = 5 ∧ total_items > 0 →
  sampling_method interval total_items = "systematic sampling" :=
sorry

end conveyor_belt_sampling_l676_67671


namespace smallest_t_for_circle_covered_l676_67638

theorem smallest_t_for_circle_covered:
  ∃ t, (∀ θ, 0 ≤ θ → θ ≤ t → (∃ r, r = Real.sin θ)) ∧
         (∀ t', (∀ θ, 0 ≤ θ → θ ≤ t' → (∃ r, r = Real.sin θ)) → t' ≥ t) :=
sorry

end smallest_t_for_circle_covered_l676_67638


namespace roots_of_quadratic_l676_67679

theorem roots_of_quadratic (a b c : ℝ) (h1 : a ≠ 0) (h2 : a + b + c = 0) (h3 : a - b + c = 0) : 
  (a * 1 ^2 + b * 1 + c = 0) ∧ (a * (-1) ^2 + b * (-1) + c = 0) :=
sorry

end roots_of_quadratic_l676_67679


namespace hash_op_example_l676_67666

def hash_op (a b c : ℤ) : ℤ := (b + 1)^2 - 4 * a * (c - 1)

theorem hash_op_example : hash_op 2 3 4 = -8 := by
  -- The proof can be added here, but for now, we use sorry to skip it
  sorry

end hash_op_example_l676_67666


namespace num_solutions_l676_67649

theorem num_solutions :
  ∃ n, (∀ a b c : ℤ, (|a + b| + c = 21 ∧ a * b + |c| = 85) ↔ n = 12) :=
sorry

end num_solutions_l676_67649


namespace cos_C_eq_3_5_l676_67609

theorem cos_C_eq_3_5 (A B C : ℝ) (hABC : A^2 + B^2 = C^2) (hRight : B ^ 2 + C ^ 2 = A ^ 2) (hTan : B / C = 4 / 3) : B / A = 3 / 5 :=
by
  sorry

end cos_C_eq_3_5_l676_67609


namespace find_k_value_l676_67605

-- Define the condition that point A(3, -5) lies on the graph of the function y = k / x
def point_on_inverse_proportion (k : ℝ) : Prop :=
  (3 : ℝ) ≠ 0 ∧ (-5) = k / (3 : ℝ)

-- The theorem to prove that k = -15 given the point on the graph
theorem find_k_value (k : ℝ) (h : point_on_inverse_proportion k) : k = -15 :=
by
  sorry

end find_k_value_l676_67605


namespace divisible_by_7_iff_l676_67682

variable {x y : ℤ}

theorem divisible_by_7_iff :
  7 ∣ (2 * x + 3 * y) ↔ 7 ∣ (5 * x + 4 * y) :=
by
  sorry

end divisible_by_7_iff_l676_67682


namespace vector_calculation_l676_67647

def a :ℝ × ℝ := (1, 2)
def b :ℝ × ℝ := (1, -1)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_calculation : scalar_mult (1/3) a - scalar_mult (4/3) b = (-1, 2) :=
by sorry

end vector_calculation_l676_67647


namespace cube_volume_given_surface_area_l676_67677

theorem cube_volume_given_surface_area (s : ℝ) (h₀ : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_given_surface_area_l676_67677


namespace puzzles_and_board_games_count_l676_67667

def num_toys : ℕ := 200
def num_action_figures : ℕ := num_toys / 4
def num_dolls : ℕ := num_toys / 3

theorem puzzles_and_board_games_count :
  num_toys - num_action_figures - num_dolls = 84 := 
  by
    -- TODO: Prove this theorem
    sorry

end puzzles_and_board_games_count_l676_67667


namespace find_x_find_a_l676_67628

-- Definitions based on conditions
def inversely_proportional (p q : ℕ) (k : ℕ) := p * q = k

-- Given conditions for (x, y)
def x1 : ℕ := 36
def y1 : ℕ := 4
def k1 : ℕ := x1 * y1 -- or 144
def y2 : ℕ := 9

-- Given conditions for (a, b)
def a1 : ℕ := 50
def b1 : ℕ := 5
def k2 : ℕ := a1 * b1 -- or 250
def b2 : ℕ := 10

-- Proof statements
theorem find_x (x : ℕ) : inversely_proportional x y2 k1 → x = 16 := by
  sorry

theorem find_a (a : ℕ) : inversely_proportional a b2 k2 → a = 25 := by
  sorry

end find_x_find_a_l676_67628


namespace incorrect_proposition_l676_67624

theorem incorrect_proposition (p q : Prop) :
  ¬(¬(p ∧ q) → ¬p ∧ ¬q) := sorry

end incorrect_proposition_l676_67624


namespace regular_bike_wheels_eq_two_l676_67635

-- Conditions
def regular_bikes : ℕ := 7
def childrens_bikes : ℕ := 11
def wheels_per_childrens_bike : ℕ := 4
def total_wheels_seen : ℕ := 58

-- Define the problem
theorem regular_bike_wheels_eq_two 
  (w : ℕ)
  (h1 : total_wheels_seen = regular_bikes * w + childrens_bikes * wheels_per_childrens_bike) :
  w = 2 :=
by
  -- Proof steps would go here
  sorry

end regular_bike_wheels_eq_two_l676_67635


namespace dividend_is_686_l676_67637

theorem dividend_is_686 (divisor quotient remainder : ℕ) (h1 : divisor = 36) (h2 : quotient = 19) (h3 : remainder = 2) :
  divisor * quotient + remainder = 686 :=
by
  sorry

end dividend_is_686_l676_67637


namespace sum_of_adjacents_to_15_l676_67684

-- Definitions of the conditions
def divisorsOf225 : Set ℕ := {3, 5, 9, 15, 25, 45, 75, 225}

-- Definition of the adjacency relationship
def isAdjacent (x y : ℕ) (s : Set ℕ) : Prop :=
  x ∈ s ∧ y ∈ s ∧ Nat.gcd x y > 1

-- Problem statement in Lean 4
theorem sum_of_adjacents_to_15 :
  ∃ x y : ℕ, isAdjacent 15 x divisorsOf225 ∧ isAdjacent 15 y divisorsOf225 ∧ x + y = 120 :=
by
  sorry

end sum_of_adjacents_to_15_l676_67684


namespace standard_equation_of_circle_l676_67619

theorem standard_equation_of_circle :
  (∃ a r, r^2 = (a + 1)^2 + (a - 1)^2 ∧ r^2 = (a - 1)^2 + (a - 3)^2 ∧ a = 1 ∧ r^2 = 4) →
  ∃ r, (x - 1)^2 + (y - 1)^2 = r^2 :=
by
  intro h
  sorry

end standard_equation_of_circle_l676_67619


namespace ones_digit_34_pow_34_pow_17_pow_17_l676_67658

-- Definitions from the conditions
def ones_digit (n : ℕ) : ℕ := n % 10

-- Translation of the original problem statement
theorem ones_digit_34_pow_34_pow_17_pow_17 :
  ones_digit (34 ^ (34 * 17 ^ 17)) = 4 :=
sorry

end ones_digit_34_pow_34_pow_17_pow_17_l676_67658


namespace omega_bound_l676_67692

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - Real.sin (ω * x)

theorem omega_bound (ω : ℝ) (h₁ : ω > 0)
  (h₂ : ∀ x : ℝ, -π / 2 < x ∧ x < π / 2 → (f ω x) ≤ (f ω (-π / 2))) :
  ω ≤ 1 / 2 :=
sorry

end omega_bound_l676_67692


namespace intersection_point_on_circle_l676_67651

theorem intersection_point_on_circle :
  ∀ (m : ℝ) (x y : ℝ),
  (m * x - y = 0) → 
  (x + m * y - m - 2 = 0) → 
  (x - 1)^2 + (y - 1 / 2)^2 = 5 / 4 :=
by
  intros m x y h1 h2
  sorry

end intersection_point_on_circle_l676_67651


namespace remainder_when_divided_by_x_minus_2_l676_67646

def f (x : ℝ) : ℝ := x^5 - 6 * x^4 + 11 * x^3 + 21 * x^2 - 17 * x + 10

theorem remainder_when_divided_by_x_minus_2 : (f 2) = 84 := by
  sorry

end remainder_when_divided_by_x_minus_2_l676_67646


namespace quadratic_inequality_l676_67632

theorem quadratic_inequality
  (a b c : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by
  sorry

end quadratic_inequality_l676_67632


namespace hall_breadth_l676_67626

theorem hall_breadth (l : ℝ) (w_s l_s b : ℝ) (n : ℕ)
  (hall_length : l = 36)
  (stone_width : w_s = 0.4)
  (stone_length : l_s = 0.5)
  (num_stones : n = 2700)
  (area_paving : l * b = n * (w_s * l_s)) :
  b = 15 := by
  sorry

end hall_breadth_l676_67626


namespace travel_time_difference_l676_67643

theorem travel_time_difference 
  (speed : ℝ) (d1 d2 : ℝ) (h_speed : speed = 50) (h_d1 : d1 = 475) (h_d2 : d2 = 450) : 
  (d1 - d2) / speed * 60 = 30 := 
by 
  sorry

end travel_time_difference_l676_67643


namespace sin_pi_minus_alpha_l676_67634

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi - α) = -1/3) : Real.sin α = -1/3 :=
sorry

end sin_pi_minus_alpha_l676_67634


namespace infinite_series_equals_l676_67614

noncomputable def infinite_series : Real :=
  ∑' n, if h : (n : ℕ) ≥ 2 then (n^4 + 2 * n^3 + 8 * n^2 + 8 * n + 8) / (2^n * (n^4 + 4)) else 0

theorem infinite_series_equals : infinite_series = 11 / 10 :=
  sorry

end infinite_series_equals_l676_67614


namespace find_angle_C_60_find_min_value_of_c_l676_67640

theorem find_angle_C_60 (a b c : ℝ) (A B C : ℝ)
  (h_cos_eq : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) : 
  C = 60 := 
sorry

theorem find_min_value_of_c (a b c : ℝ) (A B C : ℝ)
  (h_cos_eq : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h_area : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :
  c ≥ 2 * Real.sqrt 2 :=
sorry

end find_angle_C_60_find_min_value_of_c_l676_67640


namespace deepak_investment_l676_67611

theorem deepak_investment (D : ℝ) (A : ℝ) (P : ℝ) (Dp : ℝ) (Ap : ℝ) 
  (hA : A = 22500)
  (hP : P = 13800)
  (hDp : Dp = 5400)
  (h_ratio : Dp / P = D / (A + D)) :
  D = 15000 := by
  sorry

end deepak_investment_l676_67611


namespace purely_imaginary_complex_iff_l676_67688

theorem purely_imaginary_complex_iff (m : ℝ) :
  (m + 2 = 0) → (m = -2) :=
by
  sorry

end purely_imaginary_complex_iff_l676_67688


namespace speed_of_stream_l676_67639

-- Define the problem conditions
variables (b s : ℝ)
axiom cond1 : 21 = b + s
axiom cond2 : 15 = b - s

-- State the theorem
theorem speed_of_stream : s = 3 :=
sorry

end speed_of_stream_l676_67639


namespace binomial_expansion_fraction_l676_67694

theorem binomial_expansion_fraction 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 1)
    (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 243) :
    (a_0 + a_2 + a_4) / (a_1 + a_3 + a_5) = -122 / 121 :=
by
  sorry

end binomial_expansion_fraction_l676_67694


namespace subset_m_values_l676_67603

theorem subset_m_values
  {A B : Set ℝ}
  (hA : A = { x | x^2 + x - 6 = 0 })
  (hB : ∃ m, B = { x | m * x + 1 = 0 })
  (h_subset : ∀ {x}, x ∈ B → x ∈ A) :
  (∃ m, m = -1/2 ∨ m = 0 ∨ m = 1/3) :=
sorry

end subset_m_values_l676_67603


namespace find_missing_value_l676_67698

theorem find_missing_value :
  300 * 2 + (12 + 4) * 1 / 8 = 602 :=
by
  sorry

end find_missing_value_l676_67698


namespace relationship_among_variables_l676_67664

theorem relationship_among_variables (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (h1 : a^2 = 2) (h2 : b^3 = 3) (h3 : c^4 = 4) (h4 : d^5 = 5) : a = c ∧ a < d ∧ d < b := 
by
  sorry

end relationship_among_variables_l676_67664


namespace land_division_possible_l676_67652

-- Define the basic properties and conditions of the plot
structure Plot :=
  (is_square : Prop)
  (has_center_well : Prop)
  (has_four_trees : Prop)
  (has_four_gates : Prop)

-- Define a section of the plot
structure Section :=
  (contains_tree : Prop)
  (contains_gate : Prop)
  (equal_fence_length : Prop)
  (unrestricted_access_to_well : Prop)

-- Define the property that indicates a valid division of the plot
def valid_division (p : Plot) (sections : List Section) : Prop :=
  sections.length = 4 ∧
  (∀ s ∈ sections, s.contains_tree) ∧
  (∀ s ∈ sections, s.contains_gate) ∧
  (∀ s ∈ sections, s.equal_fence_length) ∧
  (∀ s ∈ sections, s.unrestricted_access_to_well)

-- Define the main theorem to prove
theorem land_division_possible (p : Plot) : 
  p.is_square ∧ p.has_center_well ∧ p.has_four_trees ∧ p.has_four_gates → 
  ∃ sections : List Section, valid_division p sections :=
by
  sorry

end land_division_possible_l676_67652


namespace james_profit_l676_67613

def cattle_profit (num_cattle : ℕ) (purchase_price total_feed_increase : ℝ)
    (weight_per_cattle : ℝ) (selling_price_per_pound : ℝ) : ℝ :=
  let feed_cost := purchase_price * (1 + total_feed_increase)
  let total_cost := purchase_price + feed_cost
  let revenue_per_cattle := weight_per_cattle * selling_price_per_pound
  let total_revenue := revenue_per_cattle * num_cattle
  total_revenue - total_cost

theorem james_profit : cattle_profit 100 40000 0.20 1000 2 = 112000 := by
  sorry

end james_profit_l676_67613


namespace smallest_integer_rel_prime_to_1020_l676_67654

theorem smallest_integer_rel_prime_to_1020 : ∃ n : ℕ, n > 1 ∧ n = 7 ∧ gcd n 1020 = 1 := by
  -- Here we state the theorem
  sorry

end smallest_integer_rel_prime_to_1020_l676_67654


namespace four_students_three_classes_l676_67674

-- Define the function that calculates the number of valid assignments
def valid_assignments (students : ℕ) (classes : ℕ) : ℕ :=
  if students = 4 ∧ classes = 3 then 36 else 0  -- Using given conditions to return 36 when appropriate

-- Define the theorem to prove that there are 36 valid ways
theorem four_students_three_classes : valid_assignments 4 3 = 36 :=
  by
  -- The proof is not required, so we use sorry to skip it
  sorry

end four_students_three_classes_l676_67674


namespace k_value_for_polynomial_l676_67681

theorem k_value_for_polynomial (k : ℤ) :
  (3 : ℤ)^3 + k * (3 : ℤ) - 18 = 0 → k = -3 :=
by
  sorry

end k_value_for_polynomial_l676_67681


namespace cost_of_cheaper_feed_l676_67675

theorem cost_of_cheaper_feed (C : ℝ)
  (total_weight : ℝ) (weight_cheaper : ℝ) (price_expensive : ℝ) (total_value : ℝ) : 
  total_weight = 35 → 
  total_value = 0.36 * total_weight → 
  weight_cheaper = 17 → 
  price_expensive = 0.53 →
  (total_value = weight_cheaper * C + (total_weight - weight_cheaper) * price_expensive) →
  C = 0.18 := 
by
  sorry

end cost_of_cheaper_feed_l676_67675


namespace two_pow_n_plus_one_square_or_cube_l676_67673

theorem two_pow_n_plus_one_square_or_cube (n : ℕ) :
  (∃ a : ℕ, 2^n + 1 = a^2) ∨ (∃ a : ℕ, 2^n + 1 = a^3) → n = 3 :=
by
  sorry

end two_pow_n_plus_one_square_or_cube_l676_67673


namespace least_deletions_to_square_l676_67680

theorem least_deletions_to_square (l : List ℕ) (h : l = [10, 20, 30, 40, 50, 60, 70, 80, 90]) : 
  ∃ d, d.card ≤ 2 ∧ ∀ (lp : List ℕ), lp = l.diff d → 
  ∃ k, lp.prod = k^2 :=
by
  sorry

end least_deletions_to_square_l676_67680


namespace bridge_length_l676_67695

def train_length : ℕ := 120
def train_speed : ℕ := 45
def crossing_time : ℕ := 30

theorem bridge_length :
  let speed_m_per_s := (train_speed * 1000) / 3600
  let total_distance := speed_m_per_s * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 255 := by
  sorry

end bridge_length_l676_67695


namespace pizza_volume_one_piece_l676_67683

theorem pizza_volume_one_piece :
  ∀ (h t: ℝ) (d: ℝ) (n: ℕ), d = 16 → t = 1/2 → n = 8 → h = 8 → 
  ( (π * (d / 2)^2 * t) / n = 4 * π ) :=
by 
  intros h t d n hd ht hn hh
  sorry

end pizza_volume_one_piece_l676_67683


namespace mass_percentage_C_in_CuCO3_l676_67627

def molar_mass_Cu := 63.546 -- g/mol
def molar_mass_C := 12.011 -- g/mol
def molar_mass_O := 15.999 -- g/mol
def molar_mass_CuCO3 := molar_mass_Cu + molar_mass_C + 3 * molar_mass_O

theorem mass_percentage_C_in_CuCO3 : 
  (molar_mass_C / molar_mass_CuCO3) * 100 = 9.72 :=
by
  sorry

end mass_percentage_C_in_CuCO3_l676_67627


namespace kate_money_ratio_l676_67690

-- Define the cost of the pen and the amount Kate needs
def pen_cost : ℕ := 30
def additional_money_needed : ℕ := 20

-- Define the amount of money Kate has
def kate_savings : ℕ := pen_cost - additional_money_needed

-- Define the ratio of Kate's money to the cost of the pen
def ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- The target property: the ratio of Kate's savings to the cost of the pen
theorem kate_money_ratio : ratio kate_savings pen_cost = (1, 3) :=
by
  sorry

end kate_money_ratio_l676_67690


namespace lcm_of_three_numbers_l676_67659

theorem lcm_of_three_numbers :
  ∀ (a b c : ℕ) (hcf : ℕ), hcf = Nat.gcd (Nat.gcd a b) c → a = 136 → b = 144 → c = 168 → hcf = 8 →
  Nat.lcm (Nat.lcm a b) c = 411264 :=
by
  intros a b c hcf h1 h2 h3 h4
  rw [h2, h3, h4]
  sorry

end lcm_of_three_numbers_l676_67659


namespace find_sum_of_x_and_reciprocal_l676_67601

theorem find_sum_of_x_and_reciprocal (x : ℝ) (hx_condition : x^3 + 1/x^3 = 110) : x + 1/x = 5 := 
sorry

end find_sum_of_x_and_reciprocal_l676_67601


namespace M_eq_N_l676_67668

noncomputable def M (a : ℝ) : ℝ :=
  a^2 + (a + 3)^2 + (a + 5)^2 + (a + 6)^2

noncomputable def N (a : ℝ) : ℝ :=
  (a + 1)^2 + (a + 2)^2 + (a + 4)^2 + (a + 7)^2

theorem M_eq_N (a : ℝ) : M a = N a :=
by
  sorry

end M_eq_N_l676_67668


namespace train_length_l676_67610

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (h1 : speed_kmh = 90) (h2 : time_s = 12) : 
  ∃ length_m : ℕ, length_m = 300 := 
by
  sorry

end train_length_l676_67610


namespace third_side_length_not_4_l676_67629

theorem third_side_length_not_4 (x : ℕ) : 
  (5 < x + 9) ∧ (9 < x + 5) ∧ (x + 5 < 14) → ¬ (x = 4) := 
by
  intros h
  sorry

end third_side_length_not_4_l676_67629


namespace find_a_plus_b_l676_67600

def satisfies_conditions (a b : ℝ) :=
  ∀ x : ℝ, 3 * (a * x + b) - 8 = 4 * x + 7

theorem find_a_plus_b (a b : ℝ) (h : satisfies_conditions a b) : a + b = 19 / 3 :=
  sorry

end find_a_plus_b_l676_67600


namespace complement_of_A_l676_67650

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set A
def A : Set ℕ := {2, 4, 5}

-- Define the complement of A with respect to U
def CU : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- State the theorem that the complement of A with respect to U is {1, 3, 6, 7}
theorem complement_of_A : CU = {1, 3, 6, 7} := by
  sorry

end complement_of_A_l676_67650


namespace crocodile_can_move_anywhere_iff_even_l676_67676

def is_even (n : ℕ) : Prop := n % 2 = 0

def can_move_to_any_square (N : ℕ) : Prop :=
∀ (x1 y1 x2 y2 : ℤ), ∃ (k : ℕ), 
(x1 + k * (N + 1) = x2 ∨ y1 + k * (N + 1) = y2)

theorem crocodile_can_move_anywhere_iff_even (N : ℕ) : can_move_to_any_square N ↔ is_even N :=
sorry

end crocodile_can_move_anywhere_iff_even_l676_67676


namespace greatest_radius_of_circle_area_lt_90pi_l676_67669

theorem greatest_radius_of_circle_area_lt_90pi : ∃ (r : ℤ), (∀ (r' : ℤ), (π * (r':ℝ)^2 < 90 * π ↔ (r' ≤ r))) ∧ (π * (r:ℝ)^2 < 90 * π) ∧ (r = 9) :=
sorry

end greatest_radius_of_circle_area_lt_90pi_l676_67669


namespace range_of_a_l676_67625

noncomputable def matrix_det_2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem range_of_a : 
  {a : ℝ | matrix_det_2x2 (a^2) 1 3 2 < matrix_det_2x2 a 0 4 1} = {a : ℝ | -1 < a ∧ a < 3/2} :=
by
  sorry

end range_of_a_l676_67625


namespace prob1_prob2_l676_67602

-- Define lines l1 and l2
def l1 (x y m : ℝ) : Prop := x + m * y + 1 = 0
def l2 (x y m : ℝ) : Prop := (m - 3) * x - 2 * y + (13 - 7 * m) = 0

-- Perpendicular condition
def perp_cond (m : ℝ) : Prop := 1 * (m - 3) - 2 * m = 0

-- Parallel condition
def parallel_cond (m : ℝ) : Prop := m * (m - 3) + 2 = 0

-- Distance between parallel lines when m = 1
def distance_between_parallel_lines (d : ℝ) : Prop := d = 2 * Real.sqrt 2

-- Problem 1: Prove that if l1 ⊥ l2, then m = -3
theorem prob1 (m : ℝ) (h : perp_cond m) : m = -3 := sorry

-- Problem 2: Prove that if l1 ∥ l2, the distance d is 2√2
theorem prob2 (m : ℝ) (h1 : parallel_cond m) (d : ℝ) (h2 : m = 1 ∨ m = -2) (h3 : m = 1) (h4 : distance_between_parallel_lines d) : d = 2 * Real.sqrt 2 := sorry

end prob1_prob2_l676_67602


namespace number_of_tables_l676_67606

theorem number_of_tables (c t : ℕ) (h1 : c = 8 * t) (h2 : 4 * c + 3 * t = 759) : t = 22 := by
  sorry

end number_of_tables_l676_67606


namespace solve_quadratic_equations_l676_67620

noncomputable def E1 := ∀ x : ℝ, x^2 - 14 * x + 21 = 0 ↔ (x = 7 + 2 * Real.sqrt 7 ∨ x = 7 - 2 * Real.sqrt 7)

noncomputable def E2 := ∀ x : ℝ, x^2 - 3 * x + 2 = 0 ↔ (x = 1 ∨ x = 2)

theorem solve_quadratic_equations :
  (E1) ∧ (E2) :=
by
  sorry

end solve_quadratic_equations_l676_67620


namespace ice_cream_cost_l676_67686

-- Define the given conditions
def cost_brownie : ℝ := 2.50
def cost_syrup_per_unit : ℝ := 0.50
def cost_nuts : ℝ := 1.50
def cost_total : ℝ := 7.00
def scoops_ice_cream : ℕ := 2
def syrup_units : ℕ := 2

-- Define the hot brownie dessert cost equation
def hot_brownie_cost (cost_ice_cream_per_scoop : ℝ) : ℝ :=
  cost_brownie + (cost_syrup_per_unit * syrup_units) + cost_nuts + (scoops_ice_cream * cost_ice_cream_per_scoop)

-- Define the theorem we want to prove
theorem ice_cream_cost : hot_brownie_cost 1 = cost_total :=
by sorry

end ice_cream_cost_l676_67686


namespace binomial_log_inequality_l676_67621

theorem binomial_log_inequality (n : ℤ) :
  n * Real.log 2 ≤ Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ∧ 
  Real.log (Nat.choose (2 * n.natAbs) n.natAbs) ≤ n * Real.log 4 :=
by sorry

end binomial_log_inequality_l676_67621


namespace f_sum_neg_l676_67665

def f : ℝ → ℝ := sorry

theorem f_sum_neg (x₁ x₂ : ℝ)
  (h1 : ∀ x, f (4 - x) = - f x)
  (h2 : ∀ x, x < 2 → ∀ y, y < x → f y < f x)
  (h3 : x₁ + x₂ > 4)
  (h4 : (x₁ - 2) * (x₂ - 2) < 0)
  : f x₁ + f x₂ < 0 := 
sorry

end f_sum_neg_l676_67665


namespace transistors_in_2010_l676_67653

theorem transistors_in_2010 (initial_transistors: ℕ) 
    (doubling_period_years: ℕ) (start_year: ℕ) (end_year: ℕ) 
    (h_initial: initial_transistors = 500000)
    (h_period: doubling_period_years = 2) 
    (h_start: start_year = 1992) 
    (h_end: end_year = 2010) :
  let years_passed := end_year - start_year
  let number_of_doublings := years_passed / doubling_period_years
  let transistors_in_end_year := initial_transistors * 2^number_of_doublings
  transistors_in_end_year = 256000000 := by
    sorry

end transistors_in_2010_l676_67653


namespace fraction_scaling_l676_67685

theorem fraction_scaling (x y : ℝ) :
  ((5 * x - 5 * 5 * y) / ((5 * x) ^ 2 + (5 * y) ^ 2)) = (1 / 5) * ((x - 5 * y) / (x ^ 2 + y ^ 2)) :=
by
  sorry

end fraction_scaling_l676_67685


namespace num_hens_in_caravan_l676_67623

variable (H G C K : ℕ)  -- number of hens, goats, camels, keepers
variable (total_heads total_feet : ℕ)

-- Defining the conditions
def num_goats := 35
def num_camels := 6
def num_keepers := 10
def heads := H + G + C + K
def feet := 2 * H + 4 * G + 4 * C + 2 * K
def relation := feet = heads + 193

theorem num_hens_in_caravan :
  G = num_goats → C = num_camels → K = num_keepers → relation → 
  H = 60 :=
by 
  intros _ _ _ _
  sorry

end num_hens_in_caravan_l676_67623


namespace find_coefficients_sum_l676_67697

theorem find_coefficients_sum :
  let f := (2 * x - 1) ^ 5 + (x + 2) ^ 4
  let a_0 := 15
  let a_1 := 42
  let a_2 := -16
  let a_5 := 32
  (|a_0| + |a_1| + |a_2| + |a_5| = 105) := 
by {
  sorry
}

end find_coefficients_sum_l676_67697


namespace city_population_l676_67612

theorem city_population (p : ℝ) (hp : 0.85 * (p + 2000) = p + 2050) : p = 2333 :=
by
  sorry

end city_population_l676_67612


namespace problem_result_l676_67689

noncomputable def max_value (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) : ℝ :=
  2 * x^2 + x * y + y^2

theorem problem (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) :
  max_value x y hx = (75 + 60 * Real.sqrt 2) / 7 :=
sorry

theorem result : 75 + 60 + 2 + 7 = 144 :=
by norm_num

end problem_result_l676_67689


namespace tan_value_of_point_on_exp_graph_l676_67656

theorem tan_value_of_point_on_exp_graph (a : ℝ) (h1 : (a, 9) ∈ {p : ℝ × ℝ | ∃ x, p = (x, 3^x)}) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := by
  sorry

end tan_value_of_point_on_exp_graph_l676_67656


namespace arithmetic_sequence_sum_condition_l676_67618

noncomputable def sum_first_n_terms (a_1 : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_sum_condition (a_1 d : ℤ) :
  sum_first_n_terms a_1 d 3 = 3 →
  sum_first_n_terms a_1 d 6 = 15 →
  (a_1 + 9 * d) + (a_1 + 10 * d) + (a_1 + 11 * d) = 30 :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_condition_l676_67618
