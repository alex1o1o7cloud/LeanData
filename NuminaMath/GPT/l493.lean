import Mathlib

namespace cylinder_volume_ratio_l493_493119

noncomputable def volume_of_cylinder (height : ℝ) (circumference : ℝ) : ℝ := 
  let r := circumference / (2 * Real.pi)
  Real.pi * r^2 * height

theorem cylinder_volume_ratio :
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  max V1 V2 / min V1 V2 = 5 / 3 :=
by
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  have hV1 : V1 = 90 / Real.pi := sorry
  have hV2 : V2 = 150 / Real.pi := sorry
  sorry

end cylinder_volume_ratio_l493_493119


namespace units_digit_factorial_150_zero_l493_493443

def units_digit (n : ℕ) : ℕ :=
  (nat.factorial n) % 10

theorem units_digit_factorial_150_zero :
  units_digit 150 = 0 :=
sorry

end units_digit_factorial_150_zero_l493_493443


namespace sequence_100th_term_l493_493595

theorem sequence_100th_term (a : ℕ → ℕ)
  (h : ∀ n : ℕ, 0 < n → (finset.range n).sum a / n = n^2) :
  a 100 = 29701 := 
sorry

end sequence_100th_term_l493_493595


namespace pints_needed_for_9_pancakes_l493_493070

theorem pints_needed_for_9_pancakes :
  ∀ (quarts_per_pancakes : ℚ) (pints_per_quart : ℚ) (pancakes_needed : ℕ),
  (quarts_per_pancakes = 3 / 18) →
  (pints_per_quart = 2) →
  (pancakes_needed = 9) →
  (pint_qty : ℚ) (pint_qty = pancakes_needed * quarts_per_pancakes * pints_per_quart) →
  pint_qty = 3 :=
by
  intros quarts_per_pancakes pints_per_quart pancakes_needed
  intros h1 h2 h3 h4
  sorry

end pints_needed_for_9_pancakes_l493_493070


namespace statement_a_is_false_statement_b_is_true_statement_c_is_false_statement_d_is_true_l493_493861

theorem statement_a_is_false :
  ¬ (∀ x : ℝ, x ≠ 0 → (abs x) / x = if x ≥ 0 then 1 else -1) := sorry

theorem statement_b_is_true :
  ∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → x₁ + (1 / x₁) < x₂ + (1 / x₂) := sorry

theorem statement_c_is_false :
  ¬ (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x < y → (x - 1/x) < (y - 1/y)) := sorry

theorem statement_d_is_true :
  (| (1 / 2) - 1 | - (1 / 2) |> -1 | = 1 := sorry

end statement_a_is_false_statement_b_is_true_statement_c_is_false_statement_d_is_true_l493_493861


namespace find_d_sin_theta_l493_493531

-- Definitions and conditions.
def length_AB : ℝ := 4
def length_BC : ℝ := 2 * Real.sqrt 2
def length_CC1 : ℝ := 2 * Real.sqrt 2

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (p1 p2 : Point) : Point :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2, z := (p1.z + p2.z) / 2 }

def B : Point := {x := length_AB, y := 0, z := 0}
def C : Point := {x := length_AB + length_BC, y := 0, z := 0}
def C1 : Point := {x := length_AB + length_BC, y := 0, z := length_CC1}
def M : Point := midpoint C C1
def N : Point := midpoint M C1

-- Proof objective.
theorem find_d_sin_theta :
  ∃ (d θ : ℝ), θ = Real.pi / 2 ∧ d = 4 / 5 ∧ d * Real.sin θ = 4 / 5 :=
by
  use 4 / 5
  use Real.pi / 2
  have h1 : Real.sin (Real.pi / 2) = 1 := Real.sin_pi_div_two
  split
  rfl
  split
  rfl
  conv {
    to_lhs,
    rw [h1],
  }
  ring
  sorry

end find_d_sin_theta_l493_493531


namespace equations_of_asymptotes_l493_493622

noncomputable def asymptotes_of_hyperbola (m : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.snd = (4 / 3) * p.fst) ∨ (p.snd = -(4 / 3) * p.fst)}

theorem equations_of_asymptotes (m : ℝ) (f : ℝ × ℝ) 
  (hf : f ∈ {p | p.1 + p.2 = 5}) :
  (f = (5, 0) ∧ m = 16) → 
  asymptotes_of_hyperbola m = {p | p.snd = (4 / 3) * p.fst ∨ p.snd = -(4 / 3) * p.fst} :=
by
  intro h
  cases h with hfoc hm
  have h1 : f = (5, 0) := hfoc
  have h2 : m = 16 := hm
  sorry

end equations_of_asymptotes_l493_493622


namespace sum_of_lengths_of_legs_of_larger_triangle_l493_493422

theorem sum_of_lengths_of_legs_of_larger_triangle
  (area_small : ℝ) (area_large : ℝ) (hypo_small : ℝ)
  (h_area_small : area_small = 18) (h_area_large : area_large = 288) (h_hypo_small : hypo_small = 10) :
  ∃ (sum_legs_large : ℝ), sum_legs_large = 52 :=
by
  sorry

end sum_of_lengths_of_legs_of_larger_triangle_l493_493422


namespace sin_diff_identity_l493_493060

theorem sin_diff_identity : sin(3 * Real.pi / 8)^2 - sin(Real.pi / 8)^2 = Real.sqrt 2 / 2 :=
by
  sorry

end sin_diff_identity_l493_493060


namespace prize_distribution_l493_493299

/--
In a best-of-five competition where two players of equal level meet in the final, 
with a score of 2:1 after the first three games and the total prize money being 12,000 yuan, 
the prize awarded to the player who has won 2 games should be 9,000 yuan.
-/
theorem prize_distribution (prize_money : ℝ) 
  (A_wins : ℕ) (B_wins : ℕ) (prob_A : ℝ) (prob_B : ℝ) (total_games : ℕ) : 
  total_games = 5 → 
  prize_money = 12000 → 
  A_wins = 2 → 
  B_wins = 1 → 
  prob_A = 1/2 → 
  prob_B = 1/2 → 
  ∃ prize_for_A : ℝ, prize_for_A = 9000 :=
by
  intros
  sorry

end prize_distribution_l493_493299


namespace average_time_per_other_class_l493_493695

theorem average_time_per_other_class (school_hours : ℚ) (num_classes : ℕ) (hist_chem_hours : ℚ)
  (total_school_time_minutes : ℕ) (hist_chem_time_minutes : ℕ) (num_other_classes : ℕ)
  (other_classes_time_minutes : ℕ) (average_time_other_classes : ℕ) :
  school_hours = 7.5 →
  num_classes = 7 →
  hist_chem_hours = 1.5 →
  total_school_time_minutes = school_hours * 60 →
  hist_chem_time_minutes = hist_chem_hours * 60 →
  other_classes_time_minutes = total_school_time_minutes - hist_chem_time_minutes →
  num_other_classes = num_classes - 2 →
  average_time_other_classes = other_classes_time_minutes / num_other_classes →
  average_time_other_classes = 72 :=
by
  intros
  sorry

end average_time_per_other_class_l493_493695


namespace exists_line_passing_through_O_with_ratios_and_max_angle_l493_493525

-- We assume two planes S1 and S2 as sets of points in ℝ³
structure Plane (S : Set ℝ^3) : Prop := 
  (is_plane : True) -- A plane definition placeholder. In real scenario, S should satisfy plane equations.

-- Points in 3D space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition of a line in 3D space
structure Line :=
  (origin : Point3D)
  (direction : Point3D)
  
-- Placeholder condition that a line intersects a plane at a certain point
def intersects (l : Line) (P : Plane) (intersect_point : Point3D) : Prop := True

-- Points A and B, and the line g passing through O
axioms (O A B : Point3D) (S1 S2 : Plane)
  (m n : ℝ)
  (g : Line)

-- Definitions to capture the intersection properties
axiom intersects_A : intersects g S1 A
axiom intersects_B : intersects g S2 B

-- Length function definition placeholder
noncomputable def length (P Q : Point3D) : ℝ := 
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2

-- Condition for the lengths ratio
axiom ratio_cond : (length O A) / (length O B) = m / n

-- Goal: Prove the existence of such a line g
theorem exists_line_passing_through_O_with_ratios_and_max_angle :
  ∃ g, intersects g S1 A ∧ intersects g S2 B ∧ (length O A) / (length O B) = m / n :=
by
  sorry

end exists_line_passing_through_O_with_ratios_and_max_angle_l493_493525


namespace part_a_l493_493492

theorem part_a (N : ℕ) (h1 : ∀ (S : Finset ℕ), S.card = 10 → (S.pairwise (λ a b, a ≠ b))) (h2 : ∀ (S : Finset ℕ), S.card = 10 → S ∈ F) (h3 : F.card = 40) : N > 60 := 
sorry

end part_a_l493_493492


namespace problem_1_problem_2_l493_493351

noncomputable theory

def vector_a : ℝ×ℝ := (Real.cos (23*Real.pi/180), Real.cos (67*Real.pi/180))
def vector_b : ℝ×ℝ := (Real.cos (68*Real.pi/180), Real.cos (22*Real.pi/180))
def dot_product (u v : ℝ×ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ×ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

theorem problem_1 : dot_product vector_a vector_b = Real.sqrt 2 / 2 := 
by 
  sorry

def collinear (u v : ℝ×ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem problem_2 (m : ℝ×ℝ) (h : collinear vector_b m) : 
  ∃ (u : ℝ×ℝ), u = (vector_a.1 + m.1, vector_a.2 + m.2) ∧ magnitude u = Real.sqrt 2 / 2 :=
by
  sorry

end problem_1_problem_2_l493_493351


namespace problem_statement_l493_493653

noncomputable section

def a : ℝ := log 3 (1 / 2)
def b : ℝ := log (1 / 4) 10
def c : ℝ := log 2 (1 / 3)

theorem problem_statement : b < c ∧ c < a :=
by
  sorry

end problem_statement_l493_493653


namespace smallest_positive_period_cos_l493_493884

theorem smallest_positive_period_cos (T : ℝ) (x : ℝ) :
  (∀ x, cos ((π / 2) - (x + T)) = cos ((π / 2) - x)) → (T = 2 * π) :=
sorry

end smallest_positive_period_cos_l493_493884


namespace num_of_integers_l493_493927

theorem num_of_integers (n : ℤ) (h : n ≠ 25) : card { n : ℤ | ∃ k : ℤ, k^2 = n / (25 - n) } = 2 :=
by
  sorry

end num_of_integers_l493_493927


namespace distinct_root_exists_l493_493226

theorem distinct_root_exists (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) :
  ∃ k : ℝ, (k = a ∧ (has_two_distinct_real_roots (λ x => a * x^2 + 2 * b * x + c)))
         ∨ (k = b ∧ (has_two_distinct_real_roots (λ x => b * x^2 + 2 * c * x + a)))
         ∨ (k = c ∧ (has_two_distinct_real_roots (λ x => c * x^2 + 2 * a * x + b))) := 
sorry

end distinct_root_exists_l493_493226


namespace principal_arg_conjugate_is_correct_l493_493243

noncomputable def principal_arg_conjugate (θ : ℝ) (h1 : (π/2) < θ) (h2 : θ < π) : ℝ :=
  let z := Complex.mk (1 - Real.sin θ) (Real.cos θ)
  let conjugate_z := Complex.conj z
  Complex.arg conjugate_z

theorem principal_arg_conjugate_is_correct (θ : ℝ) (h1 : (π/2) < θ) (h2 : θ < π) :
  principal_arg_conjugate θ h1 h2 = (3/4) * π - θ :=
sorry

end principal_arg_conjugate_is_correct_l493_493243


namespace max_value_f_l493_493761

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (-x)

theorem max_value_f : 
  ∃ (x ∈ Icc 1 3), f x = 4 * Real.exp (-2) ∧
  ∀ y ∈ Icc 1 3, f y ≤ 4 * Real.exp (-2) :=
begin
  sorry,
end

end max_value_f_l493_493761


namespace quadratic_eq_geometric_progression_l493_493557

theorem quadratic_eq_geometric_progression (a b c : ℝ) (h : 36 * b^2 - 36 * a * c = 0) :
    b^2 = a * c → a ≠ 0 → b ≠ 0 → c ≠ 0 → (∃ k : ℝ, b = a * k ∧ c = b * k) :=
by
  intros h_eq ha hb hc
  use b / a
  have hb_ne_a : a * b ≠ 0 := by { apply mul_ne_zero; assumption }
  have hc_ne_b : b * c ≠ 0 := by { apply mul_ne_zero; assumption }
  split
  · field_simp [ha]
    exact eq.symm (eq_div_of_mul_eq ha h_eq)
  · field_simp [ha,hc]
    rw [mul_comm b (b / a), mul_div_cancel' _ hb]
    exact (eq_div_iff_mul_eq hc_ne_b).mpr h_eq

end quadratic_eq_geometric_progression_l493_493557


namespace cylinder_volume_ratio_l493_493113

theorem cylinder_volume_ratio (r1 r2 V1 V2 : ℝ) (h1 : 2 * Real.pi * r1 = 6) (h2 : 2 * Real.pi * r2 = 10) (hV1 : V1 = Real.pi * r1^2 * 10) (hV2 : V2 = Real.pi * r2^2 * 6) :
  V1 < V2 → (V2 / V1) = 5 / 3 :=
by
  sorry

end cylinder_volume_ratio_l493_493113


namespace complex_power_sum_l493_493180

theorem complex_power_sum :
  (Complex.I ^ 9) + (Complex.I ^ 13) + (Complex.I ^ (-24)) = 2 * Complex.I + 1 := by
  sorry

end complex_power_sum_l493_493180


namespace exists_plane_with_at_least_4_colors_l493_493569

theorem exists_plane_with_at_least_4_colors
  (colors : Set Point → Fin 5)
  (h1 : ∀ c : Fin 5, ∃ p : Point, colors p = c) :
  ∃ (Π : Plane), ∃ S : Set Point, (S ⊆ Π) ∧ (Π ⊆ S) ∧ (4 ≤ S.card) :=
begin
  sorry
end

end exists_plane_with_at_least_4_colors_l493_493569


namespace more_blue_blocks_than_red_l493_493667

theorem more_blue_blocks_than_red 
  (red_blocks : ℕ) 
  (yellow_blocks : ℕ) 
  (blue_blocks : ℕ) 
  (total_blocks : ℕ) 
  (h_red : red_blocks = 18) 
  (h_yellow : yellow_blocks = red_blocks + 7) 
  (h_total : total_blocks = red_blocks + yellow_blocks + blue_blocks) 
  (h_total_given : total_blocks = 75) :
  blue_blocks - red_blocks = 14 :=
by sorry

end more_blue_blocks_than_red_l493_493667


namespace length_of_AB_l493_493309

-- Definitions and conditions
variables {A B C D : Type}
variables [linear_ordered_field A]

def isosceles_triangle (a b c : A) : Prop :=
  a = b ∨ b = c ∨ a = c

def perimeter (a b c : A) : A :=
  a + b + c

variables (AB AC BC CD BD : A)

-- Given conditions
axiom isosceles_ABC : isosceles_triangle AB BC AC
axiom isosceles_CBD : isosceles_triangle BC CD BD
axiom perimeter_CBD : perimeter BC CD BD = 24
axiom perimeter_ABC : perimeter AB AC BC = 23
axiom length_BD : BD = 10

-- Goal: Prove the length of AB is 9
theorem length_of_AB : AB = 9 :=
sorry

end length_of_AB_l493_493309


namespace matrix_pow_A_50_l493_493323

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 2], ![-16, -6]]

theorem matrix_pow_A_50 :
  A ^ 50 = ![![301, 100], ![-800, -249]] :=
by
  sorry

end matrix_pow_A_50_l493_493323


namespace part_I_geometric_sequence_part_II_sum_first_n_terms_l493_493951

def sequence {α : Type*} (f : ℕ → α) := ℕ → α

variable {a : ℕ → ℝ}

-- Given conditions
axiom S_n (n : ℕ) (h : n > 0) : ℝ
axiom h1 (n : ℕ) (h : n > 0) : S_n n = (3 / 2) * a n + n - 3

-- Part I: Prove that {a_n - 1} forms a geometric sequence
theorem part_I_geometric_sequence (n : ℕ) (h : n > 0) :
  ∃ r : ℝ, ∀ n > 0, a n - 1 = r * (a (n - 1) - 1) :=
sorry

-- Part II: Find the sum of the first n terms T_n of the sequence {n * a_n}
noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (i + 1) * a (i + 1)

theorem part_II_sum_first_n_terms (n : ℕ) (h : n > 0) :
  T_n n = (3 / 4) + ((2 * n - 1) * 3 ^ (n + 1)) / 4 + (n * (n + 1)) / 2 :=
sorry

end part_I_geometric_sequence_part_II_sum_first_n_terms_l493_493951


namespace num_remainders_of_square_prime_gt_7_l493_493170

theorem num_remainders_of_square_prime_gt_7 (p : ℕ) (hp_prime : Prime p) (hp_gt_7 : p > 7) :
    ∃ n = 12, n = finset.card (finset.image (λ x, (x^2 % 840)) (finset.filter (λ p, Prime p ∧ p > 7) finset.range(10000))) := sorry

end num_remainders_of_square_prime_gt_7_l493_493170


namespace solution_eq_l493_493974

theorem solution_eq (a x : ℚ) :
  (2 * (x - 2 * (x - a / 4)) = 3 * x) ∧ ((x + a) / 9 - (1 - 3 * x) / 12 = 1) → 
  a = 65 / 11 ∧ x = 13 / 11 :=
by
  sorry

end solution_eq_l493_493974


namespace units_digit_150_factorial_is_zero_l493_493455

theorem units_digit_150_factorial_is_zero :
  (nat.trailing_digits (nat.factorial 150) 1) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l493_493455


namespace correct_equation_l493_493670

theorem correct_equation (x : ℕ) (h : x ≤ 26) :
    let a_parts := 2100
    let b_parts := 1200
    let total_workers := 26
    let a_rate := 30
    let b_rate := 20
    let type_a_time := (a_parts : ℚ) / (a_rate * x)
    let type_b_time := (b_parts : ℚ) / (b_rate * (total_workers - x))
    type_a_time = type_b_time :=
by
    sorry

end correct_equation_l493_493670


namespace scientific_notation_example_l493_493852

theorem scientific_notation_example : 
  ∃ a n : ℝ, 1 ≤ |a| ∧ |a| < 10 ∧ (10,460,000 = a * 10^n) ∧ (a = 1.046) ∧ (n = 7) :=
by
  use 1.046
  use 7
  simp [Real.pow]
  norm_num
  sorry

end scientific_notation_example_l493_493852


namespace equilateral_triangle_third_vertex_y_coordinate_l493_493864

theorem equilateral_triangle_third_vertex_y_coordinate :
  ∀ (a b : ℝ), a = (0, 7) → b = (10, 7) → 
  ∃ c : ℝ, c = (5, 7 + 5 * Real.sqrt 3) :=
by
  sorry

end equilateral_triangle_third_vertex_y_coordinate_l493_493864


namespace find_a_for_quadratic_roots_l493_493756

theorem find_a_for_quadratic_roots :
  ∀ (a x₁ x₂ : ℝ), 
    (x₁ ≠ x₂) →
    (x₁ * x₁ + a * x₁ + 6 = 0) →
    (x₂ * x₂ + a * x₂ + 6 = 0) →
    (x₁ - (72 / (25 * x₂^3)) = x₂ - (72 / (25 * x₁^3))) →
    (a = 9 ∨ a = -9) :=
by
  sorry

end find_a_for_quadratic_roots_l493_493756


namespace percentage_increase_soda_price_l493_493919

theorem percentage_increase_soda_price
  (C_new : ℝ) (S_new : ℝ) (C_increase : ℝ) (C_total_before : ℝ)
  (h1 : C_new = 20)
  (h2: S_new = 6)
  (h3: C_increase = 0.25)
  (h4: C_new * (1 - C_increase) + S_new * (1 + (S_new / (S_new * (1 + (S_new / (S_new * 0.5)))))) = C_total_before) : 
  (S_new - S_new * (1 - C_increase) * 100 / (S_new * (1 + 0.5)) * C_total_before) = 50 := 
by 
  -- This is where the proof would go.
  sorry

end percentage_increase_soda_price_l493_493919


namespace bus_stops_for_45_minutes_per_hour_l493_493094

-- Define the conditions
def speed_excluding_stoppages : ℝ := 48 -- in km/hr
def speed_including_stoppages : ℝ := 12 -- in km/hr

-- Define the statement to be proven
theorem bus_stops_for_45_minutes_per_hour :
  let speed_reduction := speed_excluding_stoppages - speed_including_stoppages
  let time_stopped : ℝ := (speed_reduction / speed_excluding_stoppages) * 60
  time_stopped = 45 :=
by
  sorry

end bus_stops_for_45_minutes_per_hour_l493_493094


namespace intersection_of_A_and_B_l493_493265

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 1 ≤ 0}
noncomputable def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_of_A_and_B_l493_493265


namespace correct_remainder_l493_493588

noncomputable def f : Polynomial ℚ := X^5 + X^3 - X^2 - X - 1
noncomputable def g : Polynomial ℚ := (X^2 - 1) * (X - 2)
noncomputable def remainder : Polynomial ℚ := (34 / 3) * X^2 - (37 / 3)

theorem correct_remainder : f % g = remainder :=
by
  sorry

end correct_remainder_l493_493588


namespace min_value_1abc_l493_493489

theorem min_value_1abc (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 9) (h₂ : 0 ≤ b ∧ b ≤ 9) (h₃ : c = 0) 
    (h₄ : (1000 + 100 * a + 10 * b + c) % 2 = 0) 
    (h₅ : (1000 + 100 * a + 10 * b + c) % 3 = 0) 
    (h₆ : (1000 + 100 * a + 10 * b + c) % 5 = 0)
  : 1000 + 100 * a + 10 * b + c = 1020 :=
by
  sorry

end min_value_1abc_l493_493489


namespace incircle_radius_independence_of_j_l493_493327

-- Definitions
def on_line (A : ℕ → ℝ × ℝ) (ℓ : set (ℝ × ℝ)) : Prop :=
  ∀ i, A i ∈ ℓ

def same_incircle_radius (P : ℝ × ℝ) (A : ℕ → ℝ × ℝ) (r : ℝ) : Prop :=
  ∀ i, 
    let in_circle_radius := 
      (0.5 * dist (A i) (A (i + 1))) / 
      (0.5 * (dist P (A i) + dist P (A (i + 1)) + dist (A i) (A (i + 1)))) in
    in_circle_radius = r

-- Constants and Theorem
variable {ℓ : set (ℝ × ℝ)}
variable {P : ℝ × ℝ}
variable {A : ℕ → ℝ × ℝ}
variable {r : ℝ}
variable {k : ℕ}

theorem incircle_radius_independence_of_j
  (hP : P ∉ ℓ)
  (hA : on_line A ℓ)
  (h_radius : same_incircle_radius P A r) :
  ∀ j, 
    let r_k := 
      (0.5 * dist (A j) (A (j + k))) / 
      (0.5 * (dist P (A j) + dist P (A (j + k)) + dist (A j) (A (j + k)))) in
    r_k = (1 - (1 - 2 * r) ^ k) / 2 :=
sorry

end incircle_radius_independence_of_j_l493_493327


namespace gift_wrapping_combinations_l493_493510

theorem gift_wrapping_combinations : 
  let wrappers := 10 in
  let ribbons := 3 in
  let gift_cards := 4 in
  let gift_tags := 5 in
  wrappers * ribbons * gift_cards * gift_tags = 600 :=
by
  sorry

end gift_wrapping_combinations_l493_493510


namespace area_of_field_l493_493137

-- Define the conditions: length, width, and total fencing
def length : ℕ := 40
def fencing : ℕ := 74

-- Define the property being proved: the area of the field
theorem area_of_field : ∃ (width : ℕ), 2 * width + length = fencing ∧ length * width = 680 :=
by
  -- Proof omitted
  sorry

end area_of_field_l493_493137


namespace incorrect_statement_about_rational_numbers_l493_493472

theorem incorrect_statement_about_rational_numbers :
    (∀ x : ℝ, 0 ≤ |x|) → -- The absolute value of any real number is non-negative
    (∀ r : ℚ, ∃ r' : ℚ, r = r' ∨ ∃ irr : ℝ, ¬(irr ∈ ℚ)) → -- Real numbers include rational numbers and irrational numbers
    (∀ (a b : ℝ), a = b → ∃ p : ℝ, (p = a ∧ p = b)) → -- There is a one-to-one correspondence between real numbers and points on the number line
    ¬(∀ x : ℝ, ¬∃ y : ℝ, y*y = x → x ∈ ℚ) := -- Numbers without square roots are rational numbers is incorrect
    sorry

end incorrect_statement_about_rational_numbers_l493_493472


namespace arith_seq_general_formula_sum_of_b_terms_l493_493615

-- Definitions and conditions
def a (n : ℕ) : ℤ := 2 * n - 1
def b (n : ℕ) : ℤ := (2 * n + 1) * 2^n

noncomputable def sum_b_terms (n : ℕ) : ℤ :=
  (finset.range n).sum (λ k, b (k + 1))

-- Stating the propositions as a proof problem
theorem arith_seq_general_formula (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 + a 6 = 14) :
  a n = 2 * n - 1 :=
sorry

theorem sum_of_b_terms (n : ℕ) (h : ∀ k, ∑ i in finset.range k, b (i + 1) / 2^(i + 1) = a k + k^2 + 1) :
  sum_b_terms n = (2 * n - 1) * 2^(n + 1) + 2 :=
sorry

end arith_seq_general_formula_sum_of_b_terms_l493_493615


namespace triangle_incircle_and_circumscribed_circle_properties_l493_493416

theorem triangle_incircle_and_circumscribed_circle_properties (DE DF FE : ℝ) 
  (h_triangle : DE ^ 2 = DF ^ 2 + FE ^ 2)
  (h_angleD : ∃ (α : ℝ), α = 60 * (π / 180))
  (h_DF : DF = 6)
  (h30_60_90 : FE = DF * sqrt 3 ∧ DE = 2 * DF) :
  let s := (DE + DF + FE) / 2 in
  let area := (1 / 2) * DF * FE in
  let r := area / s in
  r = 3 * (sqrt 3 - 1) ∧ 2 * π * (DE / 2) = 12 * π :=
by
  sorry

end triangle_incircle_and_circumscribed_circle_properties_l493_493416


namespace max_length_sequence_309_l493_493901

def sequence (a₁ a₂ : ℤ) : ℕ → ℤ
| 0     := a₁
| 1     := a₂
| (n+2) := sequence n - sequence (n+1)

theorem max_length_sequence_309 :
  ∃ x : ℤ, x = 309 ∧
  (let a₁ := 500 in 
  let a₂ := x in
  sequence a₁ a₂ 9 > 0 ∧
  sequence a₁ a₂ 10 > 0) :=
sorry

end max_length_sequence_309_l493_493901


namespace part_I_part_II_l493_493266

-- Definition of the vectors required
def vector_a (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (2 * x), cos (2 * x))
def vector_b (x : ℝ) : ℝ × ℝ := (cos (2 * x), - cos (2 * x))

-- Dot product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Realization of part (I)
theorem part_I (x : ℝ) (h1 : x ∈ Ioo (7 * π / 24) (5 * π / 12)) 
    (h2 : dot_product (vector_a x) (vector_b x) + 1 / 2 = -3 / 5) : 
    cos (4 * x) = (3 - 4 * sqrt 3) / 10 :=
sorry

-- Realization of part (II)
theorem part_II (m : ℝ) (h1 : ∃ x ∈ Ioc 0 (π / 3), dot_product (vector_a x) (vector_b x) + 1 / 2 = m ∧ 
    ∀ y ∈ Ioc 0 (π / 3), dot_product (vector_a y) (vector_b y) + 1 / 2 = m → y = x) :
    m = -1 / 2 ∨ m = 1 :=
sorry

end part_I_part_II_l493_493266


namespace integer_solutions_eq_400_l493_493994

theorem integer_solutions_eq_400 : 
  ∃ (s : Finset (ℤ × ℤ)), (∀ x y, (x, y) ∈ s ↔ |3 * x + 2 * y| + |2 * x + y| = 100) ∧ s.card = 400 :=
sorry

end integer_solutions_eq_400_l493_493994


namespace trailing_zeroes_base_81_l493_493275

theorem trailing_zeroes_base_81 (n : ℕ) : (n = 15) → num_trailing_zeroes_base 81 (factorial n) = 1 :=
by
  sorry

end trailing_zeroes_base_81_l493_493275


namespace integer_condition_l493_493596

theorem integer_condition (k n : ℤ) (h1 : 1 ≤ k) (h2 : k < n) :
  (∃ m : ℤ, (n + 4) = m * (k + 2)) ↔ ∃ c : ℤ, (n - 3 * k - 2) * Nat.choose (n.nat_abs) (k.nat_abs) = c * (k + 2) :=
by
  sorry

end integer_condition_l493_493596


namespace find_m_l493_493268

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (1, 0)
def c : ℝ × ℝ := (1, -2)

-- Define the condition that a is parallel to m * b - c
def is_parallel (a : ℝ × ℝ) (v : ℝ × ℝ) : Prop :=
  a.1 * v.2 = a.2 * v.1

-- The main theorem we want to prove
theorem find_m (m : ℝ) (h : is_parallel a (m * b.1 - c.1, m * b.2 - c.2)) : m = -3 :=
by {
  -- This will be filled in with the appropriate proof
  sorry
}

end find_m_l493_493268


namespace cadence_total_earnings_l493_493540

/-- Cadence's earning details over her employment period -/
def cadence_old_company_months := 3 * 12
def cadence_new_company_months := cadence_old_company_months + 5
def cadence_old_company_salary_per_month := 5000
def cadence_salary_increase_rate := 0.20
def cadence_new_company_salary_per_month := 
  cadence_old_company_salary_per_month * (1 + cadence_salary_increase_rate)

def cadence_old_company_earnings := 
  cadence_old_company_months * cadence_old_company_salary_per_month

def cadence_new_company_earnings := 
  cadence_new_company_months * cadence_new_company_salary_per_month
  
def total_earnings := 
  cadence_old_company_earnings + cadence_new_company_earnings

theorem cadence_total_earnings :
  total_earnings = 426000 := 
by
  sorry

end cadence_total_earnings_l493_493540


namespace error_percentage_l493_493675

def r := 5
def R := 10
def pi := Real.pi

def circumference (r : ℝ) := 2 * pi * r
def area (r : ℝ) := pi * r^2

def mean (x y : ℝ) := (x + y) / 2

theorem error_percentage :
  let C1 := circumference r
  let C2 := circumference R
  let C_mean := mean C1 C2
  let mean_radius := C_mean / (2 * pi)
  let A1 := area r
  let A2 := area R
  let A_mean := mean A1 A2
  let A_mean_circ := area mean_radius
  let error := A_mean - A_mean_circ
  let error_percentage := (error / A_mean) * 100
  error_percentage = 10 := by
  sorry

end error_percentage_l493_493675


namespace coin_flip_sequences_l493_493505

/--
A coin is flipped ten times. In any two consecutive flips starting from the third flip, 
the outcomes are the same (both heads or both tails). Prove that the number of distinct 
sequences is 64.
-/
theorem coin_flip_sequences : 
  (∃ seq : vector (fin 2) 10, 
    (∀ i, 2 ≤ i → seq[i-1] = seq[i]) →
    fintype.card {seq : vector (fin 2) 10 // 
      (∀ i, 2 ≤ i → seq[i-1] = seq[i])}) = 64 :=
by sorry

end coin_flip_sequences_l493_493505


namespace right_angle_triangle_sets_l493_493148

theorem right_angle_triangle_sets :
  (∀ (a b c : ℝ), a = 2 → b = 3 → c = 4 → a^2 + b^2 ≠ c^2) ∧ 
  (∀ (a b c : ℝ), a = sqrt 3 → b = sqrt 4 → c = sqrt 5 → a^2 + b^2 ≠ c^2) ∧
  (∀ (a b c : ℝ), a = 1 → b = sqrt 2 → c = 3 → a^2 + b^2 ≠ c^2) ∧
  (∀ (a b c : ℝ), a = 5 → b = 12 → c = 13 → a^2 + b^2 = c^2) :=
by
  sorry

end right_angle_triangle_sets_l493_493148


namespace rectangle_area_l493_493043

noncomputable def length (w : ℝ) : ℝ := 4 * w

noncomputable def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area :
  ∀ (l w : ℝ), 
  l = length w ∧ perimeter l w = 200 → area l w = 1600 :=
by
  intros l w h
  cases h with h1 h2
  rw [length, perimeter, area] at *
  sorry

end rectangle_area_l493_493043


namespace tourist_groupings_l493_493423

theorem tourist_groupings :
  (∑ i in finset.range(7) \ finset.singleton(0), nat.choose 8 i) = 246 :=
by sorry

end tourist_groupings_l493_493423


namespace color_5x5_grid_excluding_two_corners_l493_493063

-- Define the total number of ways to color a 5x5 grid with each row and column having exactly one colored cell
def total_ways : Nat := 120

-- Define the number of ways to color a 5x5 grid excluding one specific corner cell such that each row and each column has exactly one colored cell
def ways_excluding_one_corner : Nat := 96

-- Prove the number of ways to color the grid excluding two specific corner cells is 78
theorem color_5x5_grid_excluding_two_corners : total_ways - (ways_excluding_one_corner + ways_excluding_one_corner - 6) = 78 := by
  -- We state our given conditions directly as definitions
  -- Now we state our theorem explicitly and use the correct answer we derived
  sorry

end color_5x5_grid_excluding_two_corners_l493_493063


namespace Kira_away_time_l493_493319

/-- 
Problem: Given two cats with different eating rates and initial conditions on the amount
of kibble in the bowl, prove that Kira was away from home for 7.2 hours.
-/
theorem Kira_away_time :
  (∀ (t : ℝ), t ≥ 0 → (4 * (t / 4) + 6 * (t / 6) - 4) + 1 = t) →
  ∃ (t : ℝ), t = 7.2 :=
begin
  sorry
end

end Kira_away_time_l493_493319


namespace fraction_sum_59_l493_493398

theorem fraction_sum_59 :
  ∃ (a b : ℕ), (0.84375 = (a : ℚ) / b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 59) :=
sorry

end fraction_sum_59_l493_493398


namespace find_b_l493_493906

theorem find_b (b : ℝ) (h : log b 625 = -2) : b = 1 / 25 :=
sorry

end find_b_l493_493906


namespace find_points_on_plane_l493_493717

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
  distance A B = 1 ∧ distance B C = 1 ∧ distance C A = 1

def circle (center : ℝ × ℝ) (radius : ℝ) (P : ℝ × ℝ) : Prop :=
  distance center P = radius * radius

def arc (center : ℝ × ℝ) (radius : ℝ) (angle : ℝ) (P : ℝ × ℝ) : Prop :=
  circle center radius P ∧ 
  (some_angle_condition) -- assume that we have some predicate here that defines the "arc-ness".

theorem find_points_on_plane (A B C : ℝ × ℝ) : 
  equilateral_triangle A B C → 
  (∀ P : ℝ × ℝ, max (distance P A) (max (distance P B) (distance P C)) = 1) ↔ 
  (∃ P : ℝ × ℝ, 
  (arc A 1 (2 * Real.pi / 3) P ∨ arc B 1 (2 * Real.pi / 3) P ∨ arc C 1 (2 * Real.pi / 3) P)) := 
by sorry

end find_points_on_plane_l493_493717


namespace race_time_difference_l493_493301

/-- In a kilometer race, A beats B by 40 meters. A takes 192 seconds to complete the race.
    Prove that A beats B by 7.68 seconds. -/
theorem race_time_difference :
  ∃ T : ℝ, 
  (A_time = 192) ∧ (A_distance = 1000) ∧ (B_distance = 960) → 
  (T - A_time = 7.68) :=
begin
  sorry
end

end race_time_difference_l493_493301


namespace initial_amount_7_years_ago_l493_493738

noncomputable def initial_investment : ℝ :=
  1000 / (1.08 ^ 7)

theorem initial_amount_7_years_ago :
  initial_investment ≈ 583.49 := sorry

end initial_amount_7_years_ago_l493_493738


namespace difference_of_extreme_numbers_l493_493427

def largest_number (digits : List ℕ) : ℕ :=
  digits.reverse.joinDigits

def least_number (digits : List ℕ) : ℕ :=
  digits.joinDigits

theorem difference_of_extreme_numbers (digits : List ℕ)
  (h_digits : digits = [1, 2, 3, 9]) :
  largest_number digits - least_number digits = 8082 := by
  sorry

end difference_of_extreme_numbers_l493_493427


namespace range_of_a_l493_493261

theorem range_of_a (n : ℕ) (hn : n > 0) (a : ℝ) :
  (\sum k in Finset.range n, (1 : ℝ) / (k * (k + 1))) > Real.log (a - 1) / Real.log 2 + a - 7 / 2 → 
  1 < a ∧ a < 3 :=
sorry

end range_of_a_l493_493261


namespace sequence_extremes_l493_493983

theorem sequence_extremes :
  ∀ n : ℕ, (a : ℝ) (h1 : a_n = (n-2017.5)/(n-2016.5)) (h2 : a_n = 1 - 1/(n-2016.5)), 
  (∀ n = 2017, a_n = -1) ∧ (∀ n = 2016, a_n = 3) ∧ 
  (∀ (n : ℕ) (hn : n > 2016.5), a_n < 1) ∧ (∀ (n : ℕ) (hn : n < 2016.5), a_n > 1) :=
by
  sorry

end sequence_extremes_l493_493983


namespace add_decimals_l493_493808

theorem add_decimals : 4.3 + 3.88 = 8.18 := 
sorry

end add_decimals_l493_493808


namespace minimum_cuts_to_unit_cubes_l493_493936

def cubes := List (ℕ × ℕ × ℕ)

def cube_cut (c : cubes) (n : ℕ) (dim : ℕ) : cubes :=
  sorry -- Function body not required for the statement

theorem minimum_cuts_to_unit_cubes (c : cubes) (s : ℕ) (dim : ℕ) :
  c = [(4,4,4)] ∧ s = 64 ∧ dim = 3 →
  ∃ (n : ℕ), n = 9 ∧
    (∀ cuts : cubes, cube_cut cuts n dim = [(1,1,1)]) :=
sorry

end minimum_cuts_to_unit_cubes_l493_493936


namespace sum_of_form_eq_l493_493770

-- Define the sum function representing the sum of the series
def sum_of_form (n : ℕ) : ℕ :=
  ∑ k in Finset.range (n + 1), (2 * k + 1)

-- Establish the theorem statement
theorem sum_of_form_eq (n: ℕ) : sum_of_form n = n * (n + 2) := 
by
  sorry

end sum_of_form_eq_l493_493770


namespace correct_option_is_C_l493_493806

theorem correct_option_is_C : 
  (∀ x : ℝ, y = 4 * (x - 1) - 2 → ¬ (y = -2)) ∧
  (∀ y : ℝ, y = x - 1 → x ≠ -1) ∧
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → (y = -2 * x + 3) ∧
  (∀ m n, y = (-m^2 - 1) * x + 3 * x + n → (m < -√2 ∨ m > √2) ∨ -m^2 + 2 < 0) →
  ∃ C : Prop, C :=
begin
  sorry,
end

end correct_option_is_C_l493_493806


namespace number_of_possible_values_for_s_l493_493513

noncomputable def is_four_place_decimal (s : ℚ) := 
  ∃ w x y z : ℕ, w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧ s = (w / 10 + x / 10^2 + y / 10^3 + z / 10^4)

theorem number_of_possible_values_for_s : 
  (∀ s : ℚ, (is_four_place_decimal s ∧ (s ≥ 2614 / 10000) ∧ (s ≤ 2792 / 10000)) → 
    (s - 3/11).abs < ((s - 1/4).abs ∧ (s - 3/11).abs < (s - 2/7).abs ∧ (s - 3/11).abs < (s - 3/10).abs) → 
      (finset.range (2792 - 2614 + 1)).card = 179) :=
sorry

end number_of_possible_values_for_s_l493_493513


namespace brocard_inequality_l493_493699

def point := (ℝ × ℝ)
def triangle := (point × point × point)

variables (A B C Ω : point)

-- Assume these points form a triangle and Ω is the Brocard point
axiom is_brocard_point (A B C Ω : point) : Prop

theorem brocard_inequality 
  (A B C Ω : point) 
  (h₁ : is_brocard_point A B C Ω) :
  ( (dist A Ω / dist B C)^2 
  + (dist B Ω / dist A C)^2 
  + (dist C Ω / dist A B)^2) 
  ≥ 1 :=
sorry

end brocard_inequality_l493_493699


namespace collinear_and_distance_relation_l493_493004

-- Definitions of points in a triangle
variables {A B C G H O : Type} [Euclidean_plane ABC]
def centroid (A B C : Point) : Point := G
def orthocenter (A B C : Point) : Point := H
def circumcenter (A B C : Point) : Point := O

-- Statement asserting that the centroid, orthocenter, and circumcenter are collinear
-- And the distance condition |OG| = 1/2 |GH|
theorem collinear_and_distance_relation 
  (triangle : Triangle A B C) 
  (G := centroid A B C) 
  (H := orthocenter A B C) 
  (O := circumcenter A B C) :
  collinear G H O ∧ distance O G = (1/2) * distance G H := 
sorry

end collinear_and_distance_relation_l493_493004


namespace greatest_distance_from_origin_l493_493729

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem greatest_distance_from_origin
  (a b r : ℝ)
  (h_center : a = 6 ∧ b = 8)
  (h_radius : r = 15) :
  greatest_distance (distance 0 0 a b) r = 25 :=
  sorry

end greatest_distance_from_origin_l493_493729


namespace polynomial_coefficients_equivalence_l493_493331

theorem polynomial_coefficients_equivalence
    {a0 a1 a2 a3 a4 a5 : ℤ}
    (h_poly : (2*x-1)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5):
    (a0 + a1 + a2 + a3 + a4 + a5 = 1) ∧
    (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| = 243) ∧
    (a1 + a3 + a5 = 122) ∧
    ((a0 + a2 + a4)^2 - (a1 + a3 + a5)^2 = -243) :=
    sorry

end polynomial_coefficients_equivalence_l493_493331


namespace part1_part2_l493_493978

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 + a * x) - 2 * x / (x + 2)

theorem part1 (h : ∀ x : ℝ, f (1 / 2) x ≥ f (1 / 2) 2) : 
  f (1 / 2) 2 = Real.log 2 - 1 ∧ ∀ x, f (1 / 2) x ≤ f (1 / 2) 2 :=
sorry

theorem part2 (a : ℝ) (h1 : 1 / 2 < a) (h2 : a < 1) (x1 x2 : ℝ) 
  (hx1 : is_extreme_point (f a) x1) (hx2 : is_extreme_point (f a) x2) :
  f a x1 + f a x2 > f a 0 :=
sorry

end part1_part2_l493_493978


namespace number_of_spotted_blue_fish_l493_493775

def total_fish := 60
def blue_fish := total_fish / 3
def spotted_blue_fish := blue_fish / 2

theorem number_of_spotted_blue_fish : spotted_blue_fish = 10 :=
by
  -- Proof is omitted
  sorry

end number_of_spotted_blue_fish_l493_493775


namespace count_valid_5_digit_numbers_l493_493073

def valid5DigitNumbers : Nat :=
  let digits := {1, 2, 3, 4, 5}
  let countDigits (f : ℕ) : ℕ :=
    if f = 3 then 24 else 54
  countDigits 3 + countDigits 2 + countDigits 4 + countDigits 5

theorem count_valid_5_digit_numbers :
  valid5DigitNumbers = 78 := 
by sorry

end count_valid_5_digit_numbers_l493_493073


namespace distance_point_line_l493_493192

def distance_from_point_to_line (a b c d e f: ℝ) : ℝ :=
  let point := (2, 3, 5)
  let line_point := (4, 9, 8)
  let line_direction := (5, 1, -3)
  let t := -1/5
  let closest_point := (5 * t + 4, t + 9, -3 * t + 8)
  let vector := (closest_point.1 - point.1, closest_point.2 - point.2, closest_point.3 - point.3)
  sqrt (vector.1^2 + vector.2^2 + vector.3^2)

theorem distance_point_line (a b c d e f : ℝ) : distance_from_point_to_line a b c d e f = sqrt 47.6 :=
by sorry

end distance_point_line_l493_493192


namespace not_exists_n_with_sum_of_remainders_eq_2012_l493_493481

theorem not_exists_n_with_sum_of_remainders_eq_2012 
  (a : Fin 11 → ℕ) (h_distinct : Function.Injective a)
  (h_ge_2 : ∀ i, 2 ≤ a i) (h_sum_407 : ∑ i, a i = 407) :
  ¬ ∃ n, (∑ i, n % a i + ∑ i, n % (4 * a i)) = 2012 :=
by
  sorry

end not_exists_n_with_sum_of_remainders_eq_2012_l493_493481


namespace probability_product_multiple_of_sixteen_l493_493664

def set_of_numbers := {3, 4, 8, 16}

def choose_two_without_replacement (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def is_multiple_of_sixteen (n : ℕ) : Prop :=
  16 ∣ n

theorem probability_product_multiple_of_sixteen :
  let eligible_pairs := choose_two_without_replacement set_of_numbers
  let total_pairs := eligible_pairs.card
  let favorable_pairs := (eligible_pairs.filter (λ (p : ℕ × ℕ), is_multiple_of_sixteen (p.1 * p.2))).card
  (favorable_pairs : ℚ) / total_pairs = 1 / 3 := by
  sorry

end probability_product_multiple_of_sixteen_l493_493664


namespace cartesian_to_polar_l493_493971

theorem cartesian_to_polar (x y : ℝ) (hx : x = 1) (hy : y = -Real.sqrt 3) :
    ∃ (ρ θ : ℝ), (ρ = 2) ∧ (θ = -Real.pi / 3) ∧ (ρ = Real.sqrt (x^2 + y^2)) ∧ (Real.tan θ = y / x) := 
by
  use 2, -Real.pi / 3
  split
  · rfl
  split
  · rfl
  split
  · rw [hx, hy]
    simp
  · rw [hx, hy]
    simp
    sorry

end cartesian_to_polar_l493_493971


namespace classics_section_books_l493_493391

-- Define the number of authors
def num_authors : Nat := 6

-- Define the number of books per author
def books_per_author : Nat := 33

-- Define the total number of books
def total_books : Nat := num_authors * books_per_author

-- Prove that the total number of books is 198
theorem classics_section_books : total_books = 198 := by
  sorry

end classics_section_books_l493_493391


namespace possible_values_of_x_l493_493304

theorem possible_values_of_x (x : ℕ) (score : ℤ) :
  (∀ q : ℕ, q = 25) → (∀ a : ℕ, a = 4) →
  (∀ b : ℕ, b = -2) → (∀ y : ℕ, y = 0) →
  (∀ answers : ℕ, answers = 25) →
  (score = 4 * x + b * (25 - x)) →
  (score ≥ 70) →
  20 ≤ x ∧ x ≤ 25 :=
by
  intros q hq a ha b hb y hy answers hansw score_eq score_ge
  sorry

end possible_values_of_x_l493_493304


namespace carpet_width_l493_493530

theorem carpet_width
  (carpet_percentage : ℝ)
  (living_room_area : ℝ)
  (carpet_length : ℝ) :
  carpet_percentage = 0.30 →
  living_room_area = 120 →
  carpet_length = 9 →
  carpet_percentage * living_room_area / carpet_length = 4 :=
by
  sorry

end carpet_width_l493_493530


namespace product_mb_gt_one_l493_493033

theorem product_mb_gt_one (m b : ℝ) (hm : m = 3 / 4) (hb : b = 2) : m * b = 3 / 2 := by
  sorry

end product_mb_gt_one_l493_493033


namespace max_xy_l493_493212

theorem max_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_eq : 2 * x + 3 * y = 6) : 
  xy ≤ (3/2) :=
sorry

end max_xy_l493_493212


namespace coprime_pos_addition_l493_493350

noncomputable def X_log_Y_Z (X Y Z : ℕ) : Prop :=
  X * (Real.log 2 / Real.log 1000) + Y * (Real.log 3 / Real.log 1000) = Z

theorem coprime_pos_addition (X Y Z : ℕ) (h1 : Nat.Coprime X Y) (h2 : Nat.Coprime X Z) (h3 : Nat.Coprime Y Z)
  (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z) (hXYZ : X_log_Y_Z X Y Z) : 
  X + Y + Z = 4 := 
sorry

end coprime_pos_addition_l493_493350


namespace triple_overlap_area_correct_l493_493055

-- Define the dimensions of the auditorium and carpets
def auditorium_dim : ℕ × ℕ := (10, 10)
def carpet1_dim : ℕ × ℕ := (6, 8)
def carpet2_dim : ℕ × ℕ := (6, 6)
def carpet3_dim : ℕ × ℕ := (5, 7)

-- The coordinates and dimensions of the overlap regions are derived based on the given positions
-- Here we assume derivations as described in the solution steps without recalculating them

-- Overlap area of the second and third carpets
def overlap23 : ℕ × ℕ := (5, 3)

-- Intersection of this overlap with the first carpet
def overlap_all : ℕ × ℕ := (2, 3)

-- Calculate the area of the region where all three carpets overlap
def triple_overlap_area : ℕ :=
  (overlap_all.1 * overlap_all.2)

theorem triple_overlap_area_correct :
  triple_overlap_area = 6 := by
  -- Expected result should be 6 square meters
  sorry

end triple_overlap_area_correct_l493_493055


namespace rate_of_interest_per_annum_l493_493662

variable {P : ℕ} (SI : ℕ)

-- Given conditions
def condition1 : Prop := SI = P / 5
def condition2 (R : ℕ) : Prop := SI = P * R * 5 / 100

-- Proof statement
theorem rate_of_interest_per_annum (P : ℕ) (SI : ℕ) : condition1 SI → (∃ R, condition2 SI R) → (∃ R, R = 4) := by
  intros h1 h2
  sorry

end rate_of_interest_per_annum_l493_493662


namespace area_OAB_coordinates_C_l493_493676

-- Defining the points
structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def A : Point := ⟨2, 4⟩
def B : Point := ⟨6, -2⟩

-- Problem 1: Prove that the area of triangle OAB is 14
theorem area_OAB : (1 / 2) * (abs (A.x * B.y + B.x * O.y + O.x * A.y - A.y * B.x - B.y * O.x - O.y * A.x)) = 14 :=
 by sorry

-- Problem 2: Prove that given OA parallel to BC and OA equals BC, the coordinates of point C are either (4,-6) or (8, 2)
theorem coordinates_C (C : Point) (h1 : (A.x - O.x) / (A.y - O.y) = (C.x - B.x) / (C.y - B.y)) 
    (h2 : sqrt ((A.x - O.x)^2 + (A.y - O.y)^2) = sqrt ((C.x - B.x)^2 + (C.y - B.y)^2)): 
    (C = ⟨4, -6⟩ ∨ C = ⟨8, 2⟩) :=
 by sorry

end area_OAB_coordinates_C_l493_493676


namespace find_angle_KML_l493_493413

noncomputable def angle_KML (r : ℝ) (S_OKL : ℝ) (S_KLM : ℝ) : ℝ :=
  if r = 5 ∧ S_OKL = 12 ∧ S_KLM > 30 then
    real.arccos (3/5)
  else
    0 -- default value if conditions are not met

-- Formal statement in Lean 4
theorem find_angle_KML :
  ∀ (O K L M : Type) (radius : ℝ)
  (area_OKL : ℝ) (area_KLM : ℝ),
  (radius = 5) →
  (area_OKL = 12) →
  (area_KLM > 30) →
  angle_KML radius area_OKL area_KLM = real.arccos (3 / 5) :=
begin
  intros O K L M r S_OKL S_KLM hr hS_OKL hS_KLM,
  simp [angle_KML],
  split_ifs,
  { exact rfl },
  { sorry }, -- default value case skipped
end

end find_angle_KML_l493_493413


namespace find_first_factor_of_LCM_l493_493018

-- Conditions
def HCF : ℕ := 23
def Y : ℕ := 14
def largest_number : ℕ := 322

-- Statement
theorem find_first_factor_of_LCM
  (A B : ℕ)
  (H : Nat.gcd A B = HCF)
  (max_num : max A B = largest_number)
  (lcm_eq : Nat.lcm A B = HCF * X * Y) :
  X = 23 :=
sorry

end find_first_factor_of_LCM_l493_493018


namespace N_satisfies_condition_l493_493579

open Matrix

def vector2d := Vector ℝ 2

def N : Matrix (Fin 2) (Fin 2) ℝ := of 2 2 ![![3, 0], ![0, 3]]

theorem N_satisfies_condition (u : vector2d) :
  (mulVec N u = 3 • u) := by
  sorry

end N_satisfies_condition_l493_493579


namespace remaining_water_in_bucket_l493_493537

theorem remaining_water_in_bucket :
  let starting_amount := 15 / 8
  let poured_out := 9 / 8
  starting_amount - poured_out = 3 / 4 :=
by
  let starting_amount := 15 / 8
  let poured_out := 9 / 8
  show starting_amount - poured_out = 3 / 4
  calc
    starting_amount - poured_out
      = (15 / 8) - (9 / 8) : by rw [starting_amount, poured_out]
  ... = (15 - 9) / 8 : by congr
  ... = 6 / 8 : by norm_num
  ... = 3 / 4 : by norm_num

end remaining_water_in_bucket_l493_493537


namespace train_speed_correct_l493_493811

noncomputable def train_speed 
  (l_t : ℕ) (l_b : ℕ) (t : ℕ) : ℝ :=
  (l_t + l_b : ℕ) / (t : ℕ : ℝ)

theorem train_speed_correct 
  (h_lt : l_t = 500) 
  (h_lb : l_b = 300) 
  (h_t : t = 45) : 
  train_speed l_t l_b t ≈ 17.78 := 
by
  sorry

end train_speed_correct_l493_493811


namespace problem_statement_l493_493221

-- Define the constants and variables
variables (x y z a b c : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := x / a + y / b + z / c = 4
def condition2 : Prop := a / x + b / y + c / z = 1

-- State the theorem that proves the question equals the correct answer
theorem problem_statement (h1 : condition1 x y z a b c) (h2 : condition2 x y z a b c) :
    x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end problem_statement_l493_493221


namespace max_length_sequence_l493_493893

noncomputable def max_length_x : ℕ := 309

theorem max_length_sequence :
  let a : ℕ → ℤ := λ n, match n with
    | 0 => 500
    | 1 => max_length_x
    | n + 2 => a n - a (n + 1)
  in
  ∀ n : ℕ, (∀ m < n, a m ≥ 0) →
    (a n < 0) →
    (309 = max_length_x) :=
by
  intro a
  intro n
  intro h_pos
  intro h_neg
  sorry

end max_length_sequence_l493_493893


namespace mean_of_remaining_students_l493_493300

theorem mean_of_remaining_students (n : ℕ) (h1 : n > 15) (mean_class mean_15 : ℝ)
  (h2 : mean_class = 10) (h3 : mean_15 = 16) : 
  (mean_remaining : ℝ) :=
  mean_remaining = (10 * n - 240) / (n - 15) :=
sorry

end mean_of_remaining_students_l493_493300


namespace sequence_a_general_term_sequence_b_sum_of_first_n_terms_l493_493625

variable {n : ℕ}

def a (n : ℕ) : ℕ := 2 * n

def b (n : ℕ) : ℕ := 3^(n-1) + 2 * n

def T (n : ℕ) : ℕ := (3^n - 1) / 2 + n^2 + n

theorem sequence_a_general_term :
  (∀ n, a n = 2 * n) :=
by
  intro n
  sorry

theorem sequence_b_sum_of_first_n_terms :
  (∀ n, T n = (3^n - 1) / 2 + n^2 + n) :=
by
  intro n
  sorry

end sequence_a_general_term_sequence_b_sum_of_first_n_terms_l493_493625


namespace ratio_of_ages_in_two_years_l493_493132

theorem ratio_of_ages_in_two_years
  (S M : ℕ) 
  (h1 : M = S + 22)
  (h2 : S = 20)
  (h3 : ∃ k : ℕ, M + 2 = k * (S + 2)) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_in_two_years_l493_493132


namespace ratio_equality_l493_493705

variables {A B C D P Q R S T U : Type}
variables [Geometry A B C D] [Circle Γ]
variables (D P Q R S T U : Point)
variable (ABC : CyclicQuadrilateral A B C D Γ)

def condition_1 : Affine (D, P) (BC) := sorry -- D to BC parallelism
def condition_2 : Intersects (P, Q, R) := sorry -- P, Q, R intersection with given lines and circle
def condition_3 : Affine (D, S) (AB) := sorry -- D to AB parallelism
def condition_4 : Intersects (S, T, U) := sorry -- S, T, U intersection with given lines and circle

theorem ratio_equality (h : condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4) :
  PQ / QR = TU / ST := by sorry

end ratio_equality_l493_493705


namespace rectangle_width_decrease_l493_493760

theorem rectangle_width_decrease (L W : ℝ) (x : ℝ) (h : L * W = x) :
  let L' := (1.40 : ℝ) * L,
      p := 1 - (1 / (1.40 : ℝ)),
      W' := (1 - p) * W in
  L' * W' = x ∧ p * 100 = 28.57 :=
by
  sorry

end rectangle_width_decrease_l493_493760


namespace find_a_l493_493105

theorem find_a (a : ℤ) 
  (hA : A = {0, 2, a})
  (hB : B = {1, a^2})
  (hU : A ∪ B = {0, 1, 2, 4, 16}) :
  a = 4 :=
sorry

end find_a_l493_493105


namespace fraction_of_loss_is_correct_l493_493517

def selling_price : ℝ := 20
def cost_price : ℝ := 21
def loss : ℝ := cost_price - selling_price
def fraction_of_loss : ℝ := loss / cost_price

theorem fraction_of_loss_is_correct : fraction_of_loss = 1 / 21 := 
by
  -- Proof goes here
  sorry

end fraction_of_loss_is_correct_l493_493517


namespace number_of_shapes_after_4_folds_sum_of_areas_after_n_folds_l493_493469

noncomputable def number_of_shapes (k : ℕ) : ℕ :=
  k + 1

noncomputable def total_area (k : ℕ) : ℚ :=
  240 * (k + 1) / (2 ^ k)

noncomputable def sum_of_areas (n : ℕ) : ℚ :=
  240 * (3 - (n + 3) / (2 ^ n))

theorem number_of_shapes_after_4_folds : number_of_shapes 4 = 5 := by
  sorry

theorem sum_of_areas_after_n_folds (n : ℕ) : ∑ k in Finset.range (n + 1), total_area k = sum_of_areas n := by
  sorry

end number_of_shapes_after_4_folds_sum_of_areas_after_n_folds_l493_493469


namespace expand_binomials_l493_493185

theorem expand_binomials (x : ℝ) : (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := 
by 
  sorry

end expand_binomials_l493_493185


namespace select_team_ways_l493_493728

-- Define the number of boys and girls
def num_boys : ℕ := 6
def num_girls : ℕ := 8

-- Define the size of the team
def team_size : ℕ := 4

-- Define a predicate for choosing at least 2 boys
def at_least_two_boys (team : list (bool × ℕ)) : Prop :=
  2 ≤ team.countp (λ person, person.1)

-- Define the total number of ways to select the team
noncomputable def total_ways : ℕ :=
  (num_boys.choose 2) * (num_girls.choose 2) + 
  (num_boys.choose 3) * (num_girls.choose 1) + 
  (num_boys.choose 4)

theorem select_team_ways : total_ways = 595 := by
  sorry

end select_team_ways_l493_493728


namespace units_digit_factorial_150_zero_l493_493439

def units_digit (n : ℕ) : ℕ :=
  (nat.factorial n) % 10

theorem units_digit_factorial_150_zero :
  units_digit 150 = 0 :=
sorry

end units_digit_factorial_150_zero_l493_493439


namespace units_digit_of_150_factorial_is_zero_l493_493435

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l493_493435


namespace trueConverseB_l493_493804

noncomputable def conditionA : Prop :=
  ∀ (x y : ℝ), -- "Vertical angles are equal"
  sorry -- Placeholder for vertical angles equality

noncomputable def conditionB : Prop :=
  ∀ (l₁ l₂ : ℝ), -- "If the consecutive interior angles are supplementary, then the two lines are parallel."
  sorry -- Placeholder for supplementary angles imply parallel lines

noncomputable def conditionC : Prop :=
  ∀ (a b : ℝ), -- "If \(a = b\), then \(a^2 = b^2\)"
  a = b → a^2 = b^2

noncomputable def conditionD : Prop :=
  ∀ (a b : ℝ), -- "If \(a > 0\) and \(b > 0\), then \(a^2 + b^2 > 0\)"
  a > 0 ∧ b > 0 → a^2 + b^2 > 0

theorem trueConverseB (hB: conditionB) : -- Proposition (B) has a true converse
  ∀ (l₁ l₂ : ℝ), 
  (∃ (a1 a2 : ℝ), -- Placeholder for angles
  sorry) → (l₁ = l₂) := -- Placeholder for consecutive interior angles are supplementary
  sorry

end trueConverseB_l493_493804


namespace total_rainfall_l493_493538

theorem total_rainfall
  (monday : ℝ)
  (tuesday : ℝ)
  (wednesday : ℝ)
  (h_monday : monday = 0.17)
  (h_tuesday : tuesday = 0.42)
  (h_wednesday : wednesday = 0.08) :
  monday + tuesday + wednesday = 0.67 :=
by
  sorry

end total_rainfall_l493_493538


namespace non_consecutive_heads_probability_l493_493076

-- Define the total number of basic events (n).
def total_events : ℕ := 2^4

-- Define the number of events where heads do not appear consecutively (m).
def non_consecutive_heads_events : ℕ := 1 + (Nat.choose 4 1) + (Nat.choose 3 2)

-- Define the probability of heads not appearing consecutively.
def probability_non_consecutive_heads : ℚ := non_consecutive_heads_events / total_events

-- The theorem we seek to prove
theorem non_consecutive_heads_probability :
  probability_non_consecutive_heads = 1 / 2 :=
by
  sorry

end non_consecutive_heads_probability_l493_493076


namespace monotonic_intervals_l493_493255

def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1 / x) - abs (x - 1 / x)

theorem monotonic_intervals (a : ℝ) (h : a = 1 / 2) :
  (strict_mono_in (f a) 0 1) ∧ (strict_mono_decreasing (f a) 1) ∧
  (strict_mono_decreasing (f a) (-1,0)) ∧ (strict_mono_in (f a) (-∞,-1)) := sorry

end monotonic_intervals_l493_493255


namespace volume_of_quadrilateral_prism_l493_493751

variable (a h V : ℝ)

-- Define the conditions given in the problem
def is_regular_quadrilateral_prism_volume_correct (a h V : ℝ) : Prop :=
  let V1 := a^2 * h in
  let V_pyramid := (1/3) * (a^2 / 2) * h in
  V = V_pyramid →
  V1 = 6 * V

theorem volume_of_quadrilateral_prism (a h V : ℝ) (h1 : is_regular_quadrilateral_prism_volume_correct a h V) : 
  a^2 * h = 6 * V := by
  sorry

end volume_of_quadrilateral_prism_l493_493751


namespace find_principal_l493_493092

variable {P R : ℝ}
axiom original_si : SI₁ = (P * R * 2) / 100
axiom increased_si : SI₂ = (P * (R + 4) * 2) / 100
axiom interest_difference : SI₂ = SI₁ + 60

theorem find_principal : P = 750 :=
by
  -- original_si, increased_si, and interest_difference are used as assumptions
  sorry

end find_principal_l493_493092


namespace count_valid_n_l493_493647

theorem count_valid_n :
  let n_values := [50, 550, 1050, 1550, 2050]
  ( ∀ n : ℤ, (50 * ((n + 500) / 50) - 500 = n) ∧ (Int.floor (Real.sqrt (2 * n : ℝ)) = (n + 500) / 50) → n ∈ n_values ) ∧
  ((∀ n : ℤ, ∃ k : ℤ, (n = 50 * k - 500) ∧ (k = Int.floor (Real.sqrt (2 * (50 * k - 500) : ℝ))) ∧ 0 < n ) → n_values.length = 5) :=
by
  sorry

end count_valid_n_l493_493647


namespace problem_statement_l493_493343

theorem problem_statement (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : abc = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ (3 / 2) := 
  sorry

end problem_statement_l493_493343


namespace max_value_expression_l493_493428

-- Define the real numbers x, y, z
variables (x y z : ℝ)

-- Define the expression we're examining
def expression := sin x * cos y + sin y * cos z + sin z * cos x

-- The theorem statement
theorem max_value_expression : ∃ x y z : ℝ, expression x y z ≤ 3/2 :=
by sorry

end max_value_expression_l493_493428


namespace practice_time_for_Friday_l493_493348

variables (M T W Th F : ℕ)

def conditions : Prop :=
  (M = 2 * T) ∧
  (T = W - 10) ∧
  (W = Th + 5) ∧
  (Th = 50) ∧
  (M + T + W + Th + F = 300)

theorem practice_time_for_Friday (h : conditions M T W Th F) : F = 60 :=
sorry

end practice_time_for_Friday_l493_493348


namespace binomial_sum_mod_9_l493_493402

noncomputable def binomial_sum_33 : ℕ :=
  ∑ k in finset.range 34, nat.choose 33 k

theorem binomial_sum_mod_9 : binomial_sum_33 % 9 = 7 := 
sorry

end binomial_sum_mod_9_l493_493402


namespace average_weight_increase_l493_493022

noncomputable def initial_average_weight (A : ℝ) := A
noncomputable def replaced_weight := 75
noncomputable def new_person_weight := 102
noncomputable def num_persons := 6

theorem average_weight_increase (A X: ℝ) (h1: X = (new_person_weight - replaced_weight) / num_persons) : 
  X = 4.5 :=
by
  calc
    X = (new_person_weight - replaced_weight) / num_persons : h1
    ... = (102 - 75) / 6 : by rfl
    ... = 27 / 6 : by rfl
    ... = 4.5 : by norm_num

end average_weight_increase_l493_493022


namespace solution_mix_percentage_l493_493370

theorem solution_mix_percentage
  (x y z : ℝ)
  (hx1 : x + y + z = 100)
  (hx2 : 0.40 * x + 0.50 * y + 0.30 * z = 46)
  (hx3 : z = 100 - x - y) :
  x = 40 ∧ y = 60 ∧ z = 0 :=
by
  sorry

end solution_mix_percentage_l493_493370


namespace six_inch_cube_value_eq_844_l493_493838

-- Definition of the value of a cube in lean
noncomputable def cube_value (s₁ s₂ : ℕ) (value₁ : ℕ) : ℕ :=
  let volume₁ := s₁ ^ 3
  let volume₂ := s₂ ^ 3
  (value₁ * volume₂) / volume₁

-- Theorem stating the equivalence between the volumes and values.
theorem six_inch_cube_value_eq_844 :
  cube_value 4 6 250 = 844 :=
by
  sorry

end six_inch_cube_value_eq_844_l493_493838


namespace simplify_and_evaluate_evaluate_when_x_is_zero_l493_493010

def expr_1 (x : ℝ) : ℝ := (1 / (1 - x)) + 1
def expr_2 (x : ℝ) : ℝ := (x ^ 2 - 4 * x + 4) / (x ^ 2 - 1)
def simplified_expr (x : ℝ) : ℝ := (x + 1) / (x - 2)

theorem simplify_and_evaluate (x : ℝ) (h1 : -2 < x) (h2 : x < 3) (h3 : x ≠ 1) (h4 : x ≠ -1) (h5 : x ≠ 2) :
  (expr_1 x) / (expr_2 x) = simplified_expr x :=
by
  sorry

theorem evaluate_when_x_is_zero :
  (expr_1 0) / (expr_2 0) = -1 / 2 :=
by
  sorry

end simplify_and_evaluate_evaluate_when_x_is_zero_l493_493010


namespace midpoint_sum_and_distance_l493_493384

theorem midpoint_sum_and_distance :
  let (x1, y1) := (-1 : ℤ, 2 : ℤ);
      (x2, y2) := (5 : ℤ, 10 : ℤ);
      mx := (x1 + x2) / 2;
      my := (y1 + y2) / 2;
  (mx + my = 8) ∧ (real.sqrt ((mx:ℝ)^2 + (my:ℝ)^2) = 2 * real.sqrt 10) := by
    sorry

end midpoint_sum_and_distance_l493_493384


namespace company_percentage_increase_l493_493548

/-- Company P had 426.09 employees in January and 490 employees in December.
    Prove that the percentage increase in employees from January to December is 15%. --/
theorem company_percentage_increase :
  ∀ (employees_jan employees_dec : ℝ),
  employees_jan = 426.09 → 
  employees_dec = 490 → 
  ((employees_dec - employees_jan) / employees_jan) * 100 = 15 :=
by
  intros employees_jan employees_dec h_jan h_dec
  sorry

end company_percentage_increase_l493_493548


namespace distinct_roots_quadratic_l493_493757

theorem distinct_roots_quadratic (a x₁ x₂ : ℝ) (h₁ : x^2 + a*x + 8 = 0) 
  (h₂ : x₁ ≠ x₂) (h₃ : x₁ - 64 / (17 * x₂^3) = x₂ - 64 / (17 * x₁^3)) : 
  a = 12 ∨ a = -12 := 
sorry

end distinct_roots_quadratic_l493_493757


namespace distance_from_O_to_points_l493_493177

noncomputable def inradius_equilateral_triangle (s : ℝ) : ℝ :=
  (s * real.sqrt 3) / 6

noncomputable def circumradius_equilateral_triangle (s : ℝ) : ℝ :=
  (s * real.sqrt 3) / 3

theorem distance_from_O_to_points :
  let s := 300
  let R := circumradius_equilateral_triangle s
  let d := R * (real.sqrt 2 - 1)
  (∀ O A B C P Q : Euc3,
    is_equilateral_triangle A B C ∧
    side_length A B C = s ∧
    outside_plane P Q A B C ∧
    PA = PB ∧ PB = PC ∧
    QA = QB ∧ QB = QC ∧
    dihedral_angle PAB QAB = 90 ∧
    equidistant O A B C P Q ⟹
    distance O A = d ∧
    distance O B = d ∧
    distance O C = d ∧
    distance O P = d ∧
    distance O Q = d) :=
by { sorry }

end distance_from_O_to_points_l493_493177


namespace efficiency_relation_l493_493484

-- Definitions and conditions
def eta0 (Q34 Q12 : ℝ) : ℝ := 1 - Q34 / Q12
def eta1 (Q13 Q12 : ℝ) : ℝ := 1 - Q13 / Q12
def eta2 (Q34 Q13 : ℝ) : ℝ := 1 - Q34 / Q13

theorem efficiency_relation 
    (Q12 Q13 Q34 α : ℝ)
    (h_eta0 : η₀ = 1 - Q34 / Q12)
    (h_eta1 : η₁ = 1 - Q13 / Q12)
    (h_eta2 : η₂ = 1 - Q34 / Q13)
    (h_relation : η₂ = (η₀ - η₁) / (1 - η₁))
    (h_eta10 : η₁ < η₀)
    (h_eta20 : η₂ < η₀)
    (h_eta0_lt : η₀ < 1)
    (h_eta1_lt : η₁ < 1)
    (h_eta1_def : η₁ = (1 - 0.01 * α) * η₀) :
    η₂ = α / (100 - (100 - α) * η₀) :=
begin
  sorry
end

end efficiency_relation_l493_493484


namespace sequence_not_strictly_monotone_l493_493700

open Nat

def d (k : ℕ) : ℕ := (List.range (k + 1)).count (λ i => i > 0 ∧ k % i = 0)

theorem sequence_not_strictly_monotone (n0 : ℕ) : ¬StrictMono (λ n, d (n^2 + 1)) := 
by
  sorry

end sequence_not_strictly_monotone_l493_493700


namespace limit_sum_of_areas_l493_493504

-- Define initial conditions
def initial_rectangle_length : ℝ := 2 * m
def initial_rectangle_width : ℝ := m

-- Define the recursive sum of circle areas
noncomputable def circle_area (n : ℕ) : ℝ := (π * m^2 / 4) * (1 / 2)^(n - 1)

noncomputable def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, circle_area (k + 1)

theorem limit_sum_of_areas (m : ℝ) (hm : m > 0) :
  tendsto (λ n, S_n n) atTop (𝓝 (π * m^2 / 2)) :=
by
  sorry

end limit_sum_of_areas_l493_493504


namespace impossible_painting_l493_493875

/--
Consider an infinite grid \(\mathcal{G}\) of unit square cells.
A chessboard polygon is a simple polygon whose sides lie along the gridlines of \(\mathcal{G}\).
Prove that there exists a chessboard polygon \(F\) such that any congruent copy of \(F\) on \(\mathcal{G}\)
will intersect at least one green cell, and it is impossible to paint no more than 2020 cells green.
--/

theorem impossible_painting (G : grid) (F : polygon) (green_cells : finset (cell G)) :
  (F.is_chessboard_polygon ∧ (∀ (G' : grid) (F' : polygon), F'.is_congruent_to F → F'.intersects G' green_cells ∧ green_cells.card ≤ 2020)) → false :=
sorry

end impossible_painting_l493_493875


namespace part_one_part_two_l493_493222

open Real

structure Point :=
  (x : ℝ)
  (y : ℝ)

def parabola (p : ℝ) (positive_p : p > 0) : set Point :=
  {A | ∃ x, A.y = x^2 / (2*p)}

def is_on_parabola (p : ℝ) (positive_p : p > 0) (A : Point) : Prop :=
  A.y = A.x^2 / (2 * p)

def orthogonal (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

def line_through (A B : Point) (P : Point) : Prop :=
  (B.x - A.x) * (P.y - A.y) = (B.y - A.y) * (P.x - A.x)

theorem part_one (p : ℝ) (positive_p : p > 0) (A B : Point):
  is_on_parabola p positive_p A → (A.x * B.x + A.y * B.y = 0) → (∃ (P : Point), P.y = 2*p ∧ ∀ (A B : Point), line_through A B P):=
by
  sorry

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def distance_to_line (P : Point) : ℝ :=
  abs (P.y - 2 * P.x) / sqrt 5

theorem part_two (p : ℝ) (positive_p : p > 0) (A B : Point):
  is_on_parabola p positive_p A → is_on_parabola p positive_p B → orthogonal A B → distance_to_line (midpoint A B) = 2*sqrt 5 / 5 → p = 2 :=
by
  sorry

end part_one_part_two_l493_493222


namespace brady_work_hours_l493_493157

theorem brady_work_hours (A : ℕ) :
    (A * 30 + 5 * 30 + 8 * 30 = 3 * 190) → 
    A = 6 :=
by sorry

end brady_work_hours_l493_493157


namespace range_of_m_l493_493607

noncomputable def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
noncomputable def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) (h₀ : m > 0) (h₁ : ∀ x : ℝ, q x m → p x) : m ≥ 9 :=
sorry

end range_of_m_l493_493607


namespace hat_p_at_1_l493_493850

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^2 - (1 + 1)*x + 1

-- Definition of displeased polynomial
def isDispleased (p : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 x4 : ℝ), p (p x1) = 0 ∧ p (p x2) = 0 ∧ p (p x3) = 0 ∧ p (p x4) = 0

-- Define the specific polynomial hat_p
def hat_p (x : ℝ) : ℝ := p x

-- Theorem statement
theorem hat_p_at_1 : isDispleased hat_p → hat_p 1 = 0 :=
by
  sorry

end hat_p_at_1_l493_493850


namespace xy_relationship_l493_493284

theorem xy_relationship (x y : ℤ) (h1 : 2 * x - y > x + 1) (h2 : x + 2 * y < 2 * y - 3) :
  x < -3 ∧ y < -4 ∧ x > y + 1 :=
sorry

end xy_relationship_l493_493284


namespace units_digit_150_factorial_is_zero_l493_493450

theorem units_digit_150_factorial_is_zero :
  (nat.trailing_digits (nat.factorial 150) 1) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l493_493450


namespace initial_owls_l493_493015

theorem initial_owls (n_0 : ℕ) (h : n_0 + 2 = 5) : n_0 = 3 :=
by 
  sorry

end initial_owls_l493_493015


namespace tower_remainder_l493_493124

variable (f : ℕ → ℕ)
variable S : ℕ

-- Conditions translated into Lean definitions
def is_valid_tower (tower : List ℕ) : Prop :=
  ∀ i, i < tower.length - 1 → tower[i + 1] ≤ tower[i] + 3

def number_of_towers (n : ℕ) : ℕ :=
  List.permutations (List.range n).succ.filter is_valid_tower.length

-- Statement we want to prove
theorem tower_remainder (S := number_of_towers 9) : S % 1000 = 0 :=
sorry

end tower_remainder_l493_493124


namespace units_digit_of_expression_l493_493173

theorem units_digit_of_expression
  (A B : ℝ)
  (hA : A = 17 + real.sqrt 224)
  (hB : B = 17 - real.sqrt 224) :
  let S := A^21 + A^85 in
  ( ∃ (d : ℕ), d < 10 ∧ S % 10 = (d : ℝ) ) :=
  sorry

end units_digit_of_expression_l493_493173


namespace track_circumference_l493_493388

-- Definitions based on conditions in Step a)
def speed_Deepak := 4.5 -- km/hr
def speed_Wife := 3.75 -- km/hr
def time_meet := 4.56 / 60 -- hours (conversion from minutes)

-- The Lean statement asserting the circumference of the track
theorem track_circumference :
  let Distance_Deepak := speed_Deepak * time_meet in
  let Distance_Wife := speed_Wife * time_meet in
  let Circumference := Distance_Deepak + Distance_Wife in
  Circumference = 0.627 :=
by
  have time_meet_def : time_meet = 4.56 / 60 := rfl
  have Distance_Deepak_def : Distance_Deepak = speed_Deepak * time_meet := rfl
  have Distance_Wife_def : Distance_Wife = speed_Wife * time_meet := rfl
  have Circumference_def : Circumference = Distance_Deepak + Distance_Wife := rfl
  have calculation : Circumference = (4.5 * (4.56 / 60)) + (3.75 * (4.56 / 60)) := by
    simp [Distance_Deepak_def, Distance_Wife_def, speed_Deepak, speed_Wife, time_meet_def]
  have result : Circumference = 0.627 := by
    simp [calculation]; norm_num
  exact result

end track_circumference_l493_493388


namespace number_of_hexagons_fitting_in_triangle_l493_493512

theorem number_of_hexagons_fitting_in_triangle :
  let A_large := (sqrt 3 / 4) * 12^2,
      A_small := (3 * sqrt 3 / 2) * 1^2 in
  (A_large / A_small) = 24 :=
by
  let A_large := (sqrt 3 / 4) * 12^2
  let A_small := (3 * sqrt 3 / 2) * 1^2
  have h : (A_large / A_small) = 24
  exact h
  sorry

end number_of_hexagons_fitting_in_triangle_l493_493512


namespace problem_l493_493960

variables {a b c : ℝ}

-- Given positive numbers a, b, c
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c

-- Given conditions
axiom h1 : a * b + a + b = 3
axiom h2 : b * c + b + c = 3
axiom h3 : a * c + a + c = 3

-- Goal statement
theorem problem : (a + 1) * (b + 1) * (c + 1) = 8 := 
by 
  sorry

end problem_l493_493960


namespace calc_correct_operation_l493_493829

theorem calc_correct_operation (a : ℕ) :
  (2 : ℕ) * a + (3 : ℕ) * a = (5 : ℕ) * a :=
by
  -- Proof
  sorry

end calc_correct_operation_l493_493829


namespace display_window_configurations_l493_493476

theorem display_window_configurations : 
  (factorial 3) * (factorial 3) = 36 := by
  sorry

end display_window_configurations_l493_493476


namespace pyramid_height_l493_493139

theorem pyramid_height (a : ℝ) (h : ℝ) (d : ℝ) (p : ℝ) (height : ℝ)
  (h1 : 4 * a = 40)                                 -- Condition: Perimeter of the square base is 40 cm
  (h2 : d = real.sqrt (a^2 + a^2))                  -- Step: Calculate the diagonal of the square base
  (h3 : h = d / 2)                                  -- Step: Half of the diagonal
  (h4 : p = 12)                                     -- Condition: Distance from apex to each vertex
  (h5 : height = real.sqrt (p^2 - h^2)) :           -- Step: Pythagorean theorem for height
  height = real.sqrt 94 :=                          -- Conclusion: Height of the pyramid
sorry

end pyramid_height_l493_493139


namespace sum_y_l493_493711

def y (m k : ℕ) : ℕ :=
  if k = 0 then 0
  else if k = 1 then 1
  else ∀ (k ≥ 2), y m (k + 2) = ((m + 1) * y m (k + 1) - (m - k) * y m k) / (k + 1)

theorem sum_y (m : ℕ) (hk : ∀ k, y m k < 2 ^ (m + 1)) :
  (∑ k in range (m + 2), y m k) = 2 ^ (m + 1) :=
  sorry

end sum_y_l493_493711


namespace find_p_l493_493908

theorem find_p (p : ℕ) : 18^3 = (16^2 / 4) * 2^(8 * p) → p = 0 := 
by 
  sorry

end find_p_l493_493908


namespace angle_DAE_l493_493529

theorem angle_DAE (A B C D E F : Type*) [norm_triplet : IsEquilateralTriangle A B C]
[reg_pent : IsRegularPentagon B C D E F] :
  ∠DAE = 42 :=
sorry

end angle_DAE_l493_493529


namespace f_monotonic_intervals_f_zero_range_l493_493256

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  m * (2 * Real.log x - x) + 1 / (x ^ 2) - 1 / x

theorem f_monotonic_intervals (m : ℕ) :
  (if m ≤ 0 then ∀ x, x ∈ (0, 2) → f m x < f m (2 - x) ∧ ∀ x, x ∈ (2, +∞) → f m x > f m (2 + x)
   else if 0 < m ∧ m < 1 / 4 then
     ∀ x, x ∈ (0, 2) → f m x < f m (2 - x) ∧ ∀ x, x ∈ (2, +∞) → f m x > f m (2 + x) ∧
     ∀ x, x ∈(0,  sqrt m / m)or  x ∈ (2, +∞)  → f m x <f m ( sqrt m/x)∧→ ∀ x, x ∈( sqrt m/ m, 2) f m x > f m (2 ) 
   else if m = 1 / 4 then ∀ x, x ∈ (0, +∞) → f m x < f m (x)
   else if m > 1 / 4 then
      ∀ x, x ∈ (0, sqrt m  / m) → f m x < f m (sqrt m / x) ∧
      ∀ x, x ∈ (sqrt m / m, 2) → f m x > f m (sqrt m / x) ∧
      ∀ x, x ∈ (2, +∞) → f m x < f m (2 + x)
  sorry

theorem f_zero_range (m : ℝ) :
  (∀ x, 0 < x ∧ x < ∞ → f m x = 0) → (m ∈ (left_of (1 / (8 * (Real.log 2 - 1))), 0)) :=
sorry

end f_monotonic_intervals_f_zero_range_l493_493256


namespace jay_more_points_than_tobee_l493_493298

-- Declare variables.
variables (x J S : ℕ)

-- Given conditions
def Tobee_points := 4
def Jay_points := Tobee_points + x -- Jay_score is 4 + x
def Sean_points := (Tobee_points + Jay_points) - 2 -- Sean_score is 4 + Jay - 2

-- The total score condition
def total_score_condition := Tobee_points + Jay_points + Sean_points = 26

-- The main statement to be proven
theorem jay_more_points_than_tobee (h : total_score_condition) : J - Tobee_points = 6 :=
sorry

end jay_more_points_than_tobee_l493_493298


namespace part_one_part_two_part_three_l493_493635

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + 0.5 * (a - x) ^ 2

-- Part (1) Tangent Line
theorem part_one (a : ℝ) (h : a = 1) : ∃ (y : ℝ), y = 1 * (1 - x) - 0 :=
begin
  sorry
end

-- Part (2) Monotonicity
theorem part_two (a : ℝ) : 
  (a ≤ 2 → ∀ x > 0, f' x a ≥ 0) ∧ 
  (a > 2 → ∃ (x1 x2 : ℝ), 0 < x1 ∧ x1 < x2 ∧ 
    (∀ x ∈ Ioo 0 x1, f' x a > 0) ∧ 
    (∀ x ∈ Ioo x1 x2, f' x a < 0) ∧ 
    (∀ x ∈ Ioo x2 ⊤, f' x a > 0)) :=
begin
  sorry
end

-- Part (3) Range of a
theorem part_three (x1 x2 a : ℝ) (h1 : x1 < x2) (h2 : |f (x2) a - f (x1) a| ∈ Ioo (3/4 - Real.log 2) (15/8 - 2 * Real.log 2)) :
  (3 * Real.sqrt 2 / 2 < a ∧ a < 5 / 2) :=
begin
  sorry
end

end part_one_part_two_part_three_l493_493635


namespace arithmetic_sequence_sum_3m_l493_493407

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (a₁ d : α) (n : ℕ) : α :=
  n • a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum_3m (a₁ d : α) (m : ℕ) (h₁ : arithmetic_sequence_sum a₁ d m = 30) (h₂ : arithmetic_sequence_sum a₁ d (2 * m) = 100) :
  arithmetic_sequence_sum a₁ d (3 * m) = 210 :=
sorry

end arithmetic_sequence_sum_3m_l493_493407


namespace angle_between_apothem_and_plane_l493_493387

-- Definitions based on the problem conditions
def regular_hexagonal_pyramid (a : ℝ) :=
  ∃ (P M K Q : ℝ×ℝ×ℝ), 
    (P.2 = a ∧ P.3 = a) ∧
    (K.1 = 0 ∧ K.2 = a * (sqrt 3 / 2)) ∧
    (M.1 = 0 ∧ M.2 = 0 ∧ M.3 = a) ∧
    (Q.1 = 0 ∧ Q.2 = a * (sqrt 3 / 2 / sqrt 7 / 2)) ∧
    (a > 0)

theorem angle_between_apothem_and_plane (a : ℝ) (h : regular_hexagonal_pyramid a) : 
  ∃ φ : ℝ, φ = Real.arcsin (sqrt 3 / 7) :=
begin
  sorry
end

end angle_between_apothem_and_plane_l493_493387


namespace highest_powers_sum_in_factorial_twenty_l493_493194

theorem highest_powers_sum_in_factorial_twenty :
  let pow2 := Nat.div (20 / 2^1) + Nat.div (20 / 2^2) + Nat.div (20 / 2^3) + Nat.div (20 / 2^4) in
  let pow5 := Nat.div (20 / 5^1) + Nat.div (20 / 5^2) in
  let pow3 := Nat.div (20 / 3^1) + Nat.div (20 / 3^2) in
  let pow10 := min pow2 pow5 in
  let pow6 := min pow2 pow3 in
  pow10 + pow6 = 12 :=
by
  let pow2 := Nat.div (20 / 2) + Nat.div (20 / 4) + Nat.div (20 / 8) + Nat.div (20 / 16)
  let pow5 := Nat.div (20 / 5) + Nat.div (20 / 25)
  let pow3 := Nat.div (20 / 3) + Nat.div (20 / 9)
  let pow10 := min pow2 pow5
  let pow6 := min pow2 pow3
  have h_pow2 : pow2 = 18 := by sorry
  have h_pow5 : pow5 = 4 := by sorry
  have h_pow3 : pow3 = 8 := by sorry
  have h_pow10 : pow10 = 4 := by sorry
  have h_pow6 : pow6 = 8 := by sorry
  have h_sum : pow10 + pow6 = 12 := by sorry
  exact h_sum

end highest_powers_sum_in_factorial_twenty_l493_493194


namespace area_of_ABCD_l493_493367

-- Define the quadrilateral and its properties.
variables (A B C D E : Type*) [EuclideanGeometry A B C D E]

-- Given conditions as definitions.
def quad_properties (A B C D E : Point) : Prop :=
  Angle ABC = π / 2 ∧
  Angle ACD = π / 2 ∧
  Dist A C = 24 ∧
  Dist C D = 36 ∧
  Dist A E = 6 ∧
  (∃ E, SameLine A C E ∧ SameLine B D E)

-- Define the theorem statement, based on given conditions and known solution.
theorem area_of_ABCD (A B C D E : Point) (h : quad_properties A B C D E) : 
  area_of_quadrilateral A B C D = 576 := sorry

end area_of_ABCD_l493_493367


namespace pink_crayons_proof_l493_493125

open Nat Real

def total_crayons : Nat := 48
def red_crayons : Nat := 12
def blue_crayons : Nat := 8
def green_crayons : Nat := 3 * blue_crayons / 4
def yellow_crayons : Nat := (0.15 * total_crayons).floor
def known_crayons : Nat := red_crayons + blue_crayons + green_crayons + yellow_crayons
def remaining_crayons : Nat := total_crayons - known_crayons
def pink_crayons : Nat := remaining_crayons / 2
def purple_crayons : Nat := remaining_crayons / 2

theorem pink_crayons_proof : pink_crayons = 8 :=
by
  sorry

end pink_crayons_proof_l493_493125


namespace problem_statement_l493_493158

theorem problem_statement :
  75 * ((4 + 1/3) - (5 + 1/4)) / ((3 + 1/2) + (2 + 1/5)) = -5/31 := 
by
  sorry

end problem_statement_l493_493158


namespace ratio_cars_to_dogs_is_two_l493_493879

-- Definitions of the conditions
def initial_dogs : ℕ := 90
def initial_cars : ℕ := initial_dogs / 3
def additional_cars : ℕ := 210
def current_dogs : ℕ := 120
def current_cars : ℕ := initial_cars + additional_cars

-- The statement to be proven
theorem ratio_cars_to_dogs_is_two :
  (current_cars : ℚ) / (current_dogs : ℚ) = 2 := by
  sorry

end ratio_cars_to_dogs_is_two_l493_493879


namespace birds_seen_wednesday_more_than_tuesday_l493_493832

/--
A bird watcher records the number of birds he sees each day. One Monday he sees 70 birds.
On Tuesday he sees half as many birds as he did on Monday.
On Wednesday he sees some more birds than he did on Tuesday.
The bird watcher saw a total of 148 birds from Monday to Wednesday.
We want to prove that the bird watcher saw 8 more birds on Wednesday compared to Tuesday.
-/
theorem birds_seen_wednesday_more_than_tuesday :
  let monday_birds := 70
  let tuesday_birds := monday_birds / 2
  let total_birds := 148
  let wednesday_birds := total_birds - (monday_birds + tuesday_birds)
  wednesday_birds - tuesday_birds = 8 :=
by
  let monday_birds := 70
  let tuesday_birds := monday_birds / 2
  let total_birds := 148
  let wednesday_birds := total_birds - (monday_birds + tuesday_birds)
  show wednesday_birds - tuesday_birds = 8 from sorry

end birds_seen_wednesday_more_than_tuesday_l493_493832


namespace units_digit_of_150_factorial_is_zero_l493_493446

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l493_493446


namespace park_area_l493_493766

theorem park_area (P : ℝ) (l w : ℝ) (A : ℝ) 
  (h1 : P = 80) 
  (h2 : l = 3 * w) 
  (h3 : 2 * l + 2 * w + 5 = 80) :
  A = l * w := 
sorry

noncomputable def calculated_area : ℝ :=
  let w := 37.5 / 4 in
  let l := 3 * w in
  l * w

example : calculated_area = 263.671875 := 
by
  -- Proof omitted, focus is on the Lean statement
  sorry

end park_area_l493_493766


namespace greatest_possible_value_l493_493789

theorem greatest_possible_value (x : ℝ) : 
  (∃ (k : ℝ), k = (5 * x - 25) / (4 * x - 5) ∧ k^2 + k = 20) → x ≤ 2 := 
sorry

end greatest_possible_value_l493_493789


namespace sum_of_distances_perpendicular_l493_493637

-- Define the hyperbola condition
def hyperbola (P : ℝ×ℝ) := P.1^2 - P.2^2 = 1

-- Define the foci of the hyperbola
def focus1 : ℝ×ℝ := (√2, 0)
def focus2 : ℝ×ℝ := (-√2, 0)

-- Define the point on the hyperbola
def is_on_hyperbola (P : ℝ×ℝ) : Prop := hyperbola P

-- Define the perpendicularity condition
def perpendicular_distance (P : ℝ×ℝ) (F1 F2 : ℝ×ℝ) : Prop := 
  let PF1 := (P.1 - F1.1, P.2 - F1.2) in
  let PF2 := (P.1 - F2.1, P.2 - F2.2) in
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 0

-- Define the main theorem
theorem sum_of_distances_perpendicular (P : ℝ×ℝ) (hP : is_on_hyperbola P) 
  (perp : perpendicular_distance P focus1 focus2) : 
  dist P focus1 + dist P focus2 = 2*√3 := 
sorry

end sum_of_distances_perpendicular_l493_493637


namespace number_of_true_propositions_l493_493859

def proposition1 : Prop :=
  -- description of proposition 1
  false -- This represents that statement 1 is false. Adjust if necessary

def proposition2 : Prop :=
  -- description of proposition 2
  true

def proposition3 : Prop :=
  -- description of proposition 3
  false -- This represents that statement 3 is false. Adjust if necessary

def proposition4 : Prop :=
  -- description of proposition 4
  true

theorem number_of_true_propositions : 
  (cond : List Prop := [proposition1, proposition2, proposition3, proposition4]) →
  List.count cond true = 2 :=
by
  sorry

end number_of_true_propositions_l493_493859


namespace heartsuit_properties_l493_493552

def heartsuit (x y : ℝ) : ℝ := abs (x - y)

theorem heartsuit_properties (x y : ℝ) :
  (heartsuit x y ≥ 0) ∧ (heartsuit x y > 0 ↔ x ≠ y) := by
  -- Proof will go here 
  sorry

end heartsuit_properties_l493_493552


namespace eccentricity_of_ellipse_l493_493968

variables {a b m n c : ℝ}
variables {F1 F2 P : ℝ}
variables {e1 e2 : ℝ}

-- Conditions
def ellipse_equation := a^2 - b^2 = c^2
def hyperbola_equation := m^2 + n^2 = c^2
def eccentricities_reciprocal := e1 * e2 = 1
def intersection_point_angle := ∠F1PF2 = π / 3  -- 60 degrees in radians

-- Definitions
def e1_definition := e1 = c / a
def e2_definition := e2 = c / m

-- Proof statement
theorem eccentricity_of_ellipse :
  ellipse_equation ∧ hyperbola_equation ∧ eccentricities_reciprocal ∧ intersection_point_angle
  → e1 = √3 / 3 :=
by
  sorry

end eccentricity_of_ellipse_l493_493968


namespace part_a_l493_493493

theorem part_a (N : ℕ) (h1 : ∀ (S : Finset ℕ), S.card = 10 → (S.pairwise (λ a b, a ≠ b))) (h2 : ∀ (S : Finset ℕ), S.card = 10 → S ∈ F) (h3 : F.card = 40) : N > 60 := 
sorry

end part_a_l493_493493


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l493_493998

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l493_493998


namespace f_f1_eq_4_l493_493636

def f (x : ℝ) : ℝ :=
  if x < 2 then 2^x else x + 2

theorem f_f1_eq_4 : f (f 1) = 4 := 
  by
    sorry

end f_f1_eq_4_l493_493636


namespace value_at_2_l493_493468

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

theorem value_at_2 : f 2 = 0 := by
  sorry

end value_at_2_l493_493468


namespace units_digit_of_150_factorial_is_zero_l493_493447

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l493_493447


namespace cadence_total_earnings_l493_493541

/-- Cadence's earning details over her employment period -/
def cadence_old_company_months := 3 * 12
def cadence_new_company_months := cadence_old_company_months + 5
def cadence_old_company_salary_per_month := 5000
def cadence_salary_increase_rate := 0.20
def cadence_new_company_salary_per_month := 
  cadence_old_company_salary_per_month * (1 + cadence_salary_increase_rate)

def cadence_old_company_earnings := 
  cadence_old_company_months * cadence_old_company_salary_per_month

def cadence_new_company_earnings := 
  cadence_new_company_months * cadence_new_company_salary_per_month
  
def total_earnings := 
  cadence_old_company_earnings + cadence_new_company_earnings

theorem cadence_total_earnings :
  total_earnings = 426000 := 
by
  sorry

end cadence_total_earnings_l493_493541


namespace wire_attachment_distance_l493_493355

theorem wire_attachment_distance :
  ∃ x : ℝ, 
    (∀ z y : ℝ, z = Real.sqrt (x ^ 2 + 3.6 ^ 2) ∧ y = Real.sqrt ((x + 5) ^ 2 + 3.6 ^ 2) →
      z + y = 13) ∧
    abs ((x : ℝ) - 2.7) < 0.01 := -- Assuming numerical closeness within a small epsilon for practical solutions.
sorry -- Proof not provided.

end wire_attachment_distance_l493_493355


namespace player_catches_ball_in_5_seconds_l493_493023

theorem player_catches_ball_in_5_seconds
    (s_ball : ℕ → ℝ) (s_player : ℕ → ℝ)
    (t_ball : ℕ)
    (t_player : ℕ)
    (d_player_initial : ℝ)
    (d_sideline : ℝ) :
  (∀ t, s_ball t = (4.375 * t - 0.375 * t^2)) →
  (∀ t, s_player t = (3.25 * t + 0.25 * t^2)) →
  (d_player_initial = 10) →
  (d_sideline = 23) →
  t_player = 5 →
  s_player t_player + d_player_initial = s_ball t_player ∧ s_ball t_player < d_sideline := 
by sorry

end player_catches_ball_in_5_seconds_l493_493023


namespace triangle_area_midpoints_l493_493178

/-- Equilateral triangle ABC has sides of length 3. T is the set of all line segments
that have length 3 and whose endpoints are on adjacent sides of the triangle. The 
midpoints of the line segments in set T enclose a region whose area to the nearest 
hundredth is m. Prove that 100m = 353. -/
theorem triangle_area_midpoints (ABC : Triangle ℝ)
  (h1 : ABC.equilateral)
  (h2 : ∀ a b : Segment ℝ, a ∈ T → b ∈ T → a.length = 3 ∧ b.length = 3 ∧ a.endpoints ⊆ ABC.vertices ∧ b.endpoints ⊆ ABC.vertices)
  : ∃ m : ℝ, 100 * m = 353 :=
sorry

end triangle_area_midpoints_l493_493178


namespace value_of_f_m_plus_one_is_negative_l493_493938

-- Definitions for function and condition
def f (x a : ℝ) := x^2 - x + a 

-- Problem statement: Given that 'f(-m) < 0', prove 'f(m+1) < 0'
theorem value_of_f_m_plus_one_is_negative (a m : ℝ) (h : f (-m) a < 0) : f (m + 1) a < 0 :=
by 
  sorry

end value_of_f_m_plus_one_is_negative_l493_493938


namespace primes_pq_pq11_7p_q_l493_493623

theorem primes_pq_pq11_7p_q (p q : ℕ) (hp : p.prime) (hq : q.prime) (h1 : (7 * p + q).prime) (h2 : (p * q + 11).prime) : p^q + q^p = 17 := 
sorry

end primes_pq_pq11_7p_q_l493_493623


namespace three_digit_number_l493_493143

theorem three_digit_number (m : ℕ) : (300 * m + 10 * m + (m - 1)) = (311 * m - 1) :=
by 
  sorry

end three_digit_number_l493_493143


namespace length_of_train_correct_l493_493854

-- Given conditions
def speed_kmph : ℝ := 120
def time_seconds : ℝ := 9
def speed_mps := (speed_kmph * 1000) / 3600

-- The length of the train should be
def length_of_train : ℝ := speed_mps * time_seconds

-- Proving the length of the train
theorem length_of_train_correct : length_of_train = 299.97 :=
by
  -- Sorry is used to indicate where the proof would be written.
  sorry

end length_of_train_correct_l493_493854


namespace trailing_zeroes_base_81_l493_493276

theorem trailing_zeroes_base_81 (n : ℕ) : (n = 15) → num_trailing_zeroes_base 81 (factorial n) = 1 :=
by
  sorry

end trailing_zeroes_base_81_l493_493276


namespace rectangle_area_l493_493035

theorem rectangle_area :
  ∃ (l w : ℝ), l = 4 * w ∧ 2 * l + 2 * w = 200 ∧ l * w = 1600 :=
by
  use [80, 20]
  split; norm_num
  split; norm_num
  sorry

end rectangle_area_l493_493035


namespace congruence_of_triangles_stereometric_theorem_l493_493731

-- Definitions for necessary conditions
structure Line :=
  (point1 point2 : pts)

structure Plane :=
  (line1 line2 : Line)
  (intersection : point)

def Perpendicular (l : Line) (π : Plane) : Prop :=
  ∀ line_in_plane (intersection : point), line_in_plane ∈ π → Perpendicular l line_in_plane

-- Theorem on which the proof relies
theorem congruence_of_triangles (a b c d e f : pts) :
  congruent_triangle a b c d e f → congruent_triangle a b c d e f :=
  sorry

-- The stereometric theorem proof as required
theorem stereometric_theorem (l : Line) (π : Plane) 
  (h1 : Perpendicular l π.line1)
  (h2 : Perpendicular l π.line2)
  (h3 : π.line1.point1 = π.line2.point1 ∧ π.line1.point2 = π.line2.point2) :
  Perpendicular l π :=
  by
    sorry

end congruence_of_triangles_stereometric_theorem_l493_493731


namespace find_m_l493_493970

theorem find_m (m : ℝ) 
    (h1 : ∃ (m: ℝ), ∀ x y : ℝ, x - m * y + 2 * m = 0) 
    (h2 : ∃ (m: ℝ), ∀ x y : ℝ, x + 2 * y - m = 0) 
    (perpendicular : (1/m) * (-1/2) = -1) : m = 1/2 :=
sorry

end find_m_l493_493970


namespace medians_divide_triangle_into_six_equal_areas_l493_493735

/-- Prove that the medians divide a triangle into six triangles of equal area. -/
theorem medians_divide_triangle_into_six_equal_areas
    (A B C M N P G : ℝ) 
    (is_median_AM : segment G A = 2 * segment G M)
    (is_median_BN : segment G B = 2 * segment G N)
    (is_median_CP : segment G C = 2 * segment G P)
    (is_centroid : G = (A + B + C) / 3) :
  area (triangle A B N) = area (triangle B N G) ∧
  area (triangle B N G) = area (triangle B C P) ∧
  area (triangle B C P) = area (triangle C P G) ∧
  area (triangle C P G) = area (triangle C A G) ∧
  area (triangle C A G) = area (triangle A P G) ∧
  area (triangle A P G) = area (triangle A B N) :=
sorry

end medians_divide_triangle_into_six_equal_areas_l493_493735


namespace log_base_conversion_l493_493825

theorem log_base_conversion {x : ℝ} (h : 3 ^ x = 4) : x = log 3 4 :=
sorry

end log_base_conversion_l493_493825


namespace sum_of_k_values_l493_493885

theorem sum_of_k_values (k : ℤ) :
  (∃ (r s : ℤ), (r ≠ s) ∧ (3 * r * s = 9) ∧ (r + s = k / 3)) → k = 0 :=
by sorry

end sum_of_k_values_l493_493885


namespace xiaobo_probability_not_home_l493_493089

theorem xiaobo_probability_not_home :
  let r1 := 1 / 2
  let r2 := 1 / 4
  let area_circle := Real.pi
  let area_greater_r1 := area_circle * (1 - r1^2)
  let area_less_r2 := area_circle * r2^2
  let area_favorable := area_greater_r1 + area_less_r2
  let probability_not_home := area_favorable / area_circle
  probability_not_home = 13 / 16 := by
  sorry

end xiaobo_probability_not_home_l493_493089


namespace minimum_value_of_f_solve_kt_single_real_root_l493_493940

noncomputable def f (x t : ℝ) : ℝ := (Real.sin (2 * x - Real.pi / 4))^2 - 2 * t * (Real.sin (2 * x - Real.pi / 4)) + t^2 - 6 * t + 1

def g (t : ℝ) : ℝ :=
  if t < -1/2 then t^2 - 5 * t + 5/4
  else if t ≤ 1 then -6 * t + 1
  else t^2 - 8 * t + 2

theorem minimum_value_of_f (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 24) (Real.pi / 2)) (t : ℝ) :
  g t = 
    if t < -1/2 then 
      have : -1/2 ∈ [-1/2, 1] ∧ true := ⟨by simp [-/, ⟨⟩⟩, _sorry⟩,
      t^2 - 5 * t + 5/4
    else if t ≤ 1 then 
      have : t ∈ Set.Icc (-1/2 : ℝ) 1 := ⟨by bound_apply_, _sorry⟩, 
      -6 * t + 1
    else 
      have : 1/2 ∈ [ (- 1 / 2) ,  1] ∧ true := ⟨by simp [-/, ⟨1⟩⟩, _sorry⟩,
      t^2 - 8*t + 2
:= sorry

theorem solve_kt_single_real_root (t k : ℝ) (ht : -1/2 ≤ t ∧ t ≤ 1) :
  (g t = k * t) ↔ (k ≤ -8 ∨ k ≥ -5) := sorry

end minimum_value_of_f_solve_kt_single_real_root_l493_493940


namespace find_matrix_N_l493_493578

theorem find_matrix_N :
  ∃ (N : matrix (fin 2) (fin 2) ℝ),
    (∀ (u : vector ℝ 2), (matrix.mul_vec N u) = (3 • u)) :=
begin
  use ![![3, 0], ![0, 3]],
  intro u,
  ext i,
  fin_cases i;
  simp,
  sorry
end

end find_matrix_N_l493_493578


namespace acute_angle_solution_l493_493189

-- Let x be an acute angle such that 0 < x < π / 2 and the given condition holds.
theorem acute_angle_solution
  (x : ℝ) (h : 0 < x ∧ x < π / 2)
  (condition : 2 * sin x * sin x + sin x - sin (2 * x) = 3 * cos x) : 
  x = π / 3 := 
sorry

end acute_angle_solution_l493_493189


namespace smallest_angle_in_15_sided_polygon_l493_493753

theorem smallest_angle_in_15_sided_polygon : 
  ∀ (a d : ℕ), (15 : ℕ).convex_polygon_with_increasing_sequence a d
  → ∀ (angles : Fin 15 → ℕ), increasing_arithmetic_sequence angles a d
  → ∀ (largest_angle less_than 172), angles 7 < 172
  → smallest_angle angles = 142 := by
    sorry

/-
Conditions encoded as definitions:
1. convex_polygon_with_increasing_sequence
2. increasing_arithmetic_sequence
3. angles 7 < 172
4. smallest_angle angles = 142
-/

end smallest_angle_in_15_sided_polygon_l493_493753


namespace PCQ_Angle_45_l493_493217

-- Definitions based on conditions
variable {A B C D P Q : Type}
variable [Square ABCD : A]
variable [OnSide P AB : B]
variable [OnSide Q AD : D]
variable [SideLength ABCD 1 : ℝ]
variable [Perimeter APQ 2 : ℝ]

-- Given the conditions, prove the angle is 45 degrees
theorem PCQ_Angle_45 : angle PCQ = 45 :=
by sorry

end PCQ_Angle_45_l493_493217


namespace cylinder_volume_ratio_l493_493117

noncomputable def volume_of_cylinder (height : ℝ) (circumference : ℝ) : ℝ := 
  let r := circumference / (2 * Real.pi)
  Real.pi * r^2 * height

theorem cylinder_volume_ratio :
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  max V1 V2 / min V1 V2 = 5 / 3 :=
by
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  have hV1 : V1 = 90 / Real.pi := sorry
  have hV2 : V2 = 150 / Real.pi := sorry
  sorry

end cylinder_volume_ratio_l493_493117


namespace probability_event_in_single_trial_l493_493294

theorem probability_event_in_single_trial (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : (1 - p)^4 = 16 / 81) : 
  p = 1 / 3 :=
sorry

end probability_event_in_single_trial_l493_493294


namespace length_PE_l493_493532

variable {A B C D E F M N P : Type}
variable {AB AD PE : ℝ}
variable [rect : Rectangle A B C D]
variable [midpoints : Midpoint E (A, D) ∧ Midpoint F (B, C)]
variable [points : Point M A B ∧ Point N C F]
variable (MN12 : SegmentLength M N = 12)
variable (AD2AB : AD = 2 * AB)
variable (PE_perp_MN : Perpendicular PE (M, N))

theorem length_PE : SegmentLength P E = 6 :=
by
  sorry

end length_PE_l493_493532


namespace conjugate_z_quadrant_l493_493609

theorem conjugate_z_quadrant (z : ℂ) (h : (z / (1 + complex.I) = complex.abs (2 - complex.I))) :
  ∃ x y : ℝ, complex.conj z = x - y * complex.I ∧ x > 0 ∧ y < 0 := 
sorry

end conjugate_z_quadrant_l493_493609


namespace base_of_hill_depth_l493_493377

theorem base_of_hill_depth : 
  ∀ (H : ℕ), 
  (H = 900) → 
  (1 / 4 * H = 225) :=
by
  intros H h
  sorry

end base_of_hill_depth_l493_493377


namespace cadence_total_earnings_l493_493543

noncomputable def total_earnings (old_years : ℕ) (old_monthly : ℕ) (new_increment : ℤ) (extra_months : ℕ) : ℤ :=
  let old_months := old_years * 12
  let old_earnings := old_monthly * old_months
  let new_monthly := old_monthly + ((old_monthly * new_increment) / 100)
  let new_months := old_months + extra_months
  let new_earnings := new_monthly * new_months
  old_earnings + new_earnings

theorem cadence_total_earnings :
  total_earnings 3 5000 20 5 = 426000 :=
by
  sorry

end cadence_total_earnings_l493_493543


namespace problem_a_part1_problem_a_part2_l493_493631

theorem problem_a_part1 (h : ∀ x : ℝ, f x = log 2 (4^x + 1) - a * x) 
  (hf_even : ∀ x : ℝ, f (-x) = f x) : a = 1 := 
sorry

theorem problem_a_part2 (h: ∀ x : ℝ, f x = log 2 (4^x + 1) - 4 * x) 
  (a_eq_4: a = 4) : ∃ x : ℝ, f x = 0 ↔ x = log 4 ((1 + sqrt 5) / 2) :=
sorry

end problem_a_part1_problem_a_part2_l493_493631


namespace area_of_shaded_region_l493_493680

noncomputable def r2 : ℝ := Real.sqrt 20
noncomputable def r1 : ℝ := 3 * r2

theorem area_of_shaded_region :
  let area := π * (r1 ^ 2) - π * (r2 ^ 2)
  area = 160 * π :=
by
  sorry

end area_of_shaded_region_l493_493680


namespace matrix_power_50_l493_493321

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := ![(5 : ℤ), 2; -16, -6]

-- The target matrix we want to prove A^50 equals to
def target : Matrix (Fin 2) (Fin 2) ℤ := ![(-301 : ℤ), -100; 800, 299]

-- Prove that A^50 equals to the target matrix
theorem matrix_power_50 : A^50 = target := 
by {
  sorry
}

end matrix_power_50_l493_493321


namespace part1_monotonicity_part2_range_a_l493_493976

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x + 1

theorem part1_monotonicity (a : ℝ) :
  (∀ x > 0, (0 : ℝ) < x → 0 < 1 / x - a) ∨
  (a > 0 → ∀ x > 0, (0 : ℝ) < x ∧ x < 1 / a → 0 < 1 / x - a ∧ 1 / a < x → 1 / x - a < 0) := sorry

theorem part2_range_a (a : ℝ) :
  (∀ x > 0, Real.log x - a * x + 1 ≤ 0) → 1 ≤ a := sorry

end part1_monotonicity_part2_range_a_l493_493976


namespace breadth_of_landscape_l493_493813

-- Definitions used in the proof
variable (L B : ℕ)
variable (playground_area : ℕ := 3200)
variable (landscape_area : ℕ := playground_area * 9)

-- Conditions
def breadth_condition : Prop := B = 8 * L
def area_condition : Prop := landscape_area = L * B
def playground_condition : Prop := playground_area = 3200

-- Main statement to prove
theorem breadth_of_landscape (h1 : breadth_condition) (h2 : area_condition) (h3 : playground_condition) : B = 480 :=
by
  sorry

end breadth_of_landscape_l493_493813


namespace square_division_l493_493142

structure Square (s : ℝ) :=
  (side_length : tl.s_nonneg s)   -- positive side length

def is_division (s : ℝ) (method : ℕ) : Prop :=
  method = 0 ∨ method = 1 ∨ method = 2

theorem square_division (s : ℝ) (sq: Square s) :
  ∃ method : ℕ, is_division s method :=
by
  -- Sorry is added here to skip the proof
  sorry

end square_division_l493_493142


namespace percent_not_participating_music_sports_l493_493053

theorem percent_not_participating_music_sports
  (total_students : ℕ) 
  (both : ℕ) 
  (music_only : ℕ) 
  (sports_only : ℕ) 
  (not_participating : ℕ)
  (percentage_not_participating : ℝ) :
  total_students = 50 →
  both = 5 →
  music_only = 15 →
  sports_only = 20 →
  not_participating = total_students - (both + music_only + sports_only) →
  percentage_not_participating = (not_participating : ℝ) / (total_students : ℝ) * 100 →
  percentage_not_participating = 20 :=
by
  sorry

end percent_not_participating_music_sports_l493_493053


namespace tulip_problem_l493_493410

theorem tulip_problem 
  (total_tulips : ℕ) 
  (red_fraction : ℚ) 
  (blue_fraction : ℚ) 
  (h_total : total_tulips = 56) 
  (h_red_fraction : red_fraction = 3 / 7) 
  (h_blue_fraction : blue_fraction = 3 / 8) 
  (h_red_tulips : 24 = red_fraction * total_tulips) 
  (h_blue_tulips : 21 = blue_fraction * total_tulips) :
  let red_tulips := red_fraction * total_tulips
  let blue_tulips := blue_fraction * total_tulips
  let pink_tulips := total_tulips - (red_tulips + blue_tulips)
  in pink_tulips = 11 := 
by
  sorry

end tulip_problem_l493_493410


namespace irrational_pi_l493_493802

def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem irrational_pi : is_irrational π := by
  sorry

end irrational_pi_l493_493802


namespace sum_is_45_l493_493390

noncomputable def sum_of_numbers (a b c : ℝ) : ℝ :=
  a + b + c

theorem sum_is_45 {a b c : ℝ} (h1 : ∃ a b c, (a ≤ b ∧ b ≤ c) ∧ b = 10)
  (h2 : (a + b + c) / 3 = a + 20)
  (h3 : (a + b + c) / 3 = c - 25) :
  sum_of_numbers a b c = 45 := 
sorry

end sum_is_45_l493_493390


namespace units_digit_of_150_factorial_is_zero_l493_493464

theorem units_digit_of_150_factorial_is_zero : 
  ∃ k : ℕ, (150! = k * 10) :=
begin
  -- We need to prove that there exists a natural number k such that 150! is equal to k times 10
  sorry
end

end units_digit_of_150_factorial_is_zero_l493_493464


namespace weight_of_brick_l493_493499

def brick_weight_problem (x : ℝ) : Prop :=
  x = 2 + x / 2

theorem weight_of_brick : ∃ (x : ℝ), brick_weight_problem x ∧ x = 4 :=
by
  use 4
  split
  · simp [brick_weight_problem]
  · simp

end weight_of_brick_l493_493499


namespace cannot_form_triangle_can_form_triangle_B_can_form_triangle_C_can_form_triangle_D_l493_493862

theorem cannot_form_triangle :
  ¬ (∀ a b c : ℕ, a = 4 ∧ b = 4 ∧ c = 9 → a + b > c ∧ a + c > b ∧ b + c > a) :=
by
  intro h
  specialize h 4 4 9 ⟨rfl, rfl, rfl⟩
  cases h
  exact lt_irrefl _ (lt_of_le_of_lt (nat.add_le_add (le_refl 4) (le_refl 4)) (by norm_num))

theorem can_form_triangle_B :
  ∀ a b c : ℕ, a = 3 ∧ b = 5 ∧ c = 6 → a + b > c ∧ a + c > b ∧ b + c > a :=
by
  rintro a b c ⟨rfl, rfl, rfl⟩
  norm_num

theorem can_form_triangle_C :
  ∀ a b c : ℕ, a = 6 ∧ b = 8 ∧ c = 10 → a + b > c ∧ a + c > b ∧ b + c > a :=
by
  rintro a b c ⟨rfl, rfl, rfl⟩
  norm_num

theorem can_form_triangle_D :
  ∀ a b c : ℕ, a = 5 ∧ b = 12 ∧ c = 13 → a + b > c ∧ a + c > b ∧ b + c > a :=
by
  rintro a b c ⟨rfl, rfl, rfl⟩
  norm_num

end cannot_form_triangle_can_form_triangle_B_can_form_triangle_C_can_form_triangle_D_l493_493862


namespace parabolic_arch_height_at_10_inch_from_center_l493_493135

theorem parabolic_arch_height_at_10_inch_from_center :
  ∀ (a : ℝ) (x : ℝ) (y : ℝ), 
    (y = a * x^2 + 20) →
    ((∀ x, 0 = a * (30)^2 + 20) ∧ (x = 10)) →
    y = 17.78 :=
by
  intros a x y h_eq h_cond
  sorry

end parabolic_arch_height_at_10_inch_from_center_l493_493135


namespace number_of_BMWs_sold_l493_493837

theorem number_of_BMWs_sold (total_cars : ℕ) (Audi_percent Toyota_percent Acura_percent Ford_percent : ℝ)
  (h_total_cars : total_cars = 250) 
  (h_percentages : Audi_percent = 0.10 ∧ Toyota_percent = 0.20 ∧ Acura_percent = 0.15 ∧ Ford_percent = 0.25) :
  ∃ (BMWs_sold : ℕ), BMWs_sold = 75 := 
by
  sorry

end number_of_BMWs_sold_l493_493837


namespace seventy_fifth_digit_of_1_to_50_consecutive_is_2_l493_493287

theorem seventy_fifth_digit_of_1_to_50_consecutive_is_2 :
  let sequence := (List.range (50 + 1)).drop 1.map (λ n => n.toString).join;
  sequence.nth 74 = '2' :=
by
  sorry

end seventy_fifth_digit_of_1_to_50_consecutive_is_2_l493_493287


namespace log_ordering_l493_493655

-- Definitions of the logarithmic expressions.
def a : ℝ := log 3 (1 / 2)
def b : ℝ := log (1 / 4) 10
def c : ℝ := log 2 (1 / 3)

-- Statement of the theorem we aim to prove.
theorem log_ordering : b < c < a := sorry

end log_ordering_l493_493655


namespace find_abc_l493_493373

noncomputable def abc_satisfies (a b c : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a + b + c = 29 ∧
  (1 / a.toReal + 1 / b.toReal + 1 / c.toReal + 399 / (a * b * c).toReal = 1)

theorem find_abc :
  ∃ (a b c : ℤ), abc_satisfies a b c ∧ a * b * c = 992 :=
by
  sorry

end find_abc_l493_493373


namespace find_triples_l493_493910

-- Define the main theorem to be proven
theorem find_triples (a m n : ℕ) (h1 : a > 1) (h2 : m < n) 
  (h3 : ∀ p : ℕ, p.prime → (p ∣ (a ^ m - 1) ↔ p ∣ (a ^ n - 1))) : 
  (a, m, n) = (3, 1, 2) := 
sorry

end find_triples_l493_493910


namespace factorial_multiple_l493_493364

theorem factorial_multiple (m n : ℕ) : (2 * m)! * (2 * n)! % (m! * n! * (m + n)!) = 0 := by
  sorry

end factorial_multiple_l493_493364


namespace no_pos_int_mult_5005_in_form_l493_493273

theorem no_pos_int_mult_5005_in_form (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 49) :
  ¬ ∃ k : ℕ, 5005 * k = 10^j - 10^i := by
  sorry

end no_pos_int_mult_5005_in_form_l493_493273


namespace calories_in_250g_mixed_drink_l493_493315

def calories_in_mixed_drink (grams_cranberry : ℕ) (grams_honey : ℕ) (grams_water : ℕ)
  (calories_per_100g_cranberry : ℕ) (calories_per_100g_honey : ℕ) (calories_per_100g_water : ℕ)
  (total_grams : ℕ) (portion_grams : ℕ) : ℚ :=
  ((grams_cranberry * calories_per_100g_cranberry + grams_honey * calories_per_100g_honey + grams_water * calories_per_100g_water) : ℚ)
  / (total_grams * portion_grams)

theorem calories_in_250g_mixed_drink :
  calories_in_mixed_drink 150 50 300 30 304 0 100 250 = 98.5 := by
  -- The proof will involve arithmetic operations
  sorry

end calories_in_250g_mixed_drink_l493_493315


namespace find_numbers_with_double_digit_sum_one_l493_493909

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

def double_digit_sum (n : ℕ) : ℕ :=
  digit_sum (digit_sum n)

theorem find_numbers_with_double_digit_sum_one :
  {n : ℕ | 90 ≤ n ∧ n ≤ 150 ∧ double_digit_sum(n) = 1} =
  {91, 100, 109, 118, 127, 136, 145} := sorry

end find_numbers_with_double_digit_sum_one_l493_493909


namespace minimum_value_proof_l493_493992

noncomputable def min_value (m n : ℝ) : ℝ := m + n

theorem minimum_value_proof (m n : ℝ) (h_pos_m : m > 0) (h_pos_n : n > 0) 
  (h_dot_product : (1 : ℝ) + 2 * n * (m + n) = 1) :
  min_value m n = sqrt 3 - 1 :=
  sorry

end minimum_value_proof_l493_493992


namespace range_f_on_interval_l493_493944

noncomputable def f (x : ℝ) : ℝ := sorry

theorem range_f_on_interval :
  (∀ x : ℝ, f (-x) + f x = 0) ∧
  (∀ x : ℝ, f (x + 1) + f x = 0) ∧
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x = Real.sqrt x) →
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ -5 / 2 → f x ∈ (-1, -(Real.sqrt 2)/2] ∪ {0}) :=
sorry

end range_f_on_interval_l493_493944


namespace units_digit_factorial_150_l493_493456

theorem units_digit_factorial_150 : (nat.factorial 150) % 10 = 0 :=
sorry

end units_digit_factorial_150_l493_493456


namespace food_cost_max_l493_493500

theorem food_cost_max (x : ℝ) (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_total : ℝ) (food_cost_max : ℝ) :
  total_cost = x * (1 + tax_rate + tip_rate) →
  tax_rate = 0.07 →
  tip_rate = 0.15 →
  max_total = 50 →
  total_cost ≤ max_total →
  food_cost_max = 50 / 1.22 →
  x ≤ food_cost_max :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end food_cost_max_l493_493500


namespace cannot_have_1970_minus_signs_in_grid_l493_493296

theorem cannot_have_1970_minus_signs_in_grid :
  ∀ (k l : ℕ), k ≤ 100 → l ≤ 100 → (k+l)*50 - k*l ≠ 985 :=
by
  intros k l hk hl
  sorry

end cannot_have_1970_minus_signs_in_grid_l493_493296


namespace maximum_sequence_length_positive_integer_x_l493_493902

/-- Define the sequence terms based on the problem statement -/
def sequence_term (n : ℕ) (a₁ a₂ : ℤ) : ℤ :=
  if n = 1 then a₁
  else if n = 2 then a₂
  else sequence_term (n - 2) a₁ a₂ - sequence_term (n - 1) a₁ a₂

/-- Define the main problem with the conditions -/
theorem maximum_sequence_length_positive_integer_x :
  ∃ x : ℕ, 0 < x ∧ (309 = x) ∧ 
  (∀ n, sequence_term n 500 x ≥ 0) ∧
  (sequence_term 11 500 x < 0) :=
by
  sorry

end maximum_sequence_length_positive_integer_x_l493_493902


namespace series_sum_imaginary_unit_l493_493340

theorem series_sum_imaginary_unit :
  let i : ℂ := complex.I in
  (∑ k in finset.range 2006, i ^ k) = i :=
begin
  sorry
end

end series_sum_imaginary_unit_l493_493340


namespace javier_five_attractions_orders_l493_493314

theorem javier_five_attractions_orders :
  ∃ (n : ℕ), n = 5! ∧ n = 120 :=
begin
  use 120,
  split,
  { simp [factorial],
    norm_num, },
  { refl, }
end

end javier_five_attractions_orders_l493_493314


namespace rationalize_denominator_l493_493736

theorem rationalize_denominator :
  let A := 20
  let B := 7
  let C := 15
  let D := 3
  let F := 5
  let G := 2
  let E := 137
  (5 / (4 * Real.sqrt 7 + 3 * Real.sqrt 3 - Real.sqrt 2) = (20 * Real.sqrt 7 + 15 * Real.sqrt 3 + 5 * Real.sqrt(2)) / 137) ∧
  (A + B + C + D + F + G + E = 199) :=
by {
  let A := 20
  let B := 7
  let C := 15
  let D := 3
  let F := 5
  let G := 2
  let E := 137
  sorry
}

end rationalize_denominator_l493_493736


namespace prove_sequences_l493_493553

open Int

def seq_a : ℕ → ℤ
| 0       := 1
| (n + 1) := seq_a n + 3 * seq_b n + 3 * seq_c n

def seq_b : ℕ → ℤ
| 0       := 1
| (n + 1) := seq_a n + seq_b n + 3 * seq_c n

def seq_c : ℕ → ℤ
| 0       := 1
| (n + 1) := seq_a n + seq_b n + seq_c n

theorem prove_sequences (n : ℕ) (h : n = 13 ^ 4) :
  (seq_a n) % 13 = 1 ∧ (seq_b n) % 13 = 3 ∧ (seq_c n) % 13 = 9 :=
sorry

end prove_sequences_l493_493553


namespace number_of_boxes_l493_493131

-- Define the given conditions
def total_chocolates : ℕ := 442
def chocolates_per_box : ℕ := 26

-- Prove the number of small boxes in the large box
theorem number_of_boxes : (total_chocolates / chocolates_per_box) = 17 := by
  sorry

end number_of_boxes_l493_493131


namespace convert_distance_convert_time_l493_493110

theorem convert_distance (d_km : Float) (d_m : Float) (conv_rate_km_m : Float) :
  d_km = 70 → d_m = 50 → conv_rate_km_m = 1000 → d_km + d_m / conv_rate_km_m = 70.05 :=
by
  intros
  sorry

theorem convert_time (t_hr : Float) (conv_rate_hr_min : Float) :
  t_hr = 3.6 → conv_rate_hr_min = 60 → (Int.floor t_hr, (t_hr - Int.floor t_hr) * conv_rate_hr_min) = (3, 36) :=
by
  intros
  sorry

end convert_distance_convert_time_l493_493110


namespace max_meeting_days_l493_493497

theorem max_meeting_days (n : ℕ) (h_n : n = 8) : ∃ d, d = 128 ∧ 
  (∀ (days : list (set (fin n))), (∀ (s : set (fin n)), s ∈ days → s ≠ ∅) ∧ 
  (∀ (i j : ℕ), i ≠ j → (days.nth i).get_or_else ∅ ≠ (days.nth j).get_or_else ∅) ∧ 
  (∀ (N : ℕ), 1 ≤ N → N < list.length days → 
    ∃ (k : ℕ), 1 ≤ k → k < N → ∃ (p : fin n), p ∈ (days.nth N).get_or_else ∅ ∧ p ∈ (days.nth k).get_or_else ∅)) :=
begin
  sorry
end

end max_meeting_days_l493_493497


namespace ratio_of_sheep_to_horses_l493_493154

noncomputable def number_of_horses (total_food : ℕ) (food_per_horse : ℕ) : ℕ :=
  total_food / food_per_horse

def gcd (x y : ℕ) : ℕ :=
  if y = 0 then x else gcd y (x % y)

def simplified_ratio (a b : ℕ) : ℕ × ℕ :=
  let d := gcd a b in (a / d, b / d)

theorem ratio_of_sheep_to_horses :
  let sheep := 24
  let total_food := 12880
  let food_per_horse := 230
  let horses := number_of_horses total_food food_per_horse
  let (simplified_sheep, simplified_horses) := simplified_ratio sheep horses
  simplified_sheep = 3 ∧ simplified_horses = 7 := by
  sorry

end ratio_of_sheep_to_horses_l493_493154


namespace impossible_event_abs_lt_zero_l493_493800

theorem impossible_event_abs_lt_zero (a : ℝ) : ¬ (|a| < 0) :=
sorry

end impossible_event_abs_lt_zero_l493_493800


namespace conic_section_eccentricity_l493_493241

theorem conic_section_eccentricity (m : ℝ) (h_geo_seq : 1 * m = m * 9) :
  (m = 3 ∨ m = -3) →
  (m = 3 → let e := Real.sqrt (1 - 1 / 3) in e = Real.sqrt 2 / 3) ∧
  (m = -3 → let e := Real.sqrt (1 + 1 / 3) in e = 2) := by
  sorry

end conic_section_eccentricity_l493_493241


namespace perpendicular_bisector_eq_l493_493547

-- Define the circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- Prove that the perpendicular bisector of line segment AB has the equation 3x - y - 9 = 0
theorem perpendicular_bisector_eq :
  (∀ x y : ℝ, C1 x y → C2 x y → 3 * x - y - 9 = 0) :=
by
  sorry

end perpendicular_bisector_eq_l493_493547


namespace angle_D_is_90_l493_493207

theorem angle_D_is_90 (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : B = 130) (h5 : C + D = 180) :
  D = 90 :=
by
  sorry

end angle_D_is_90_l493_493207


namespace maria_spent_l493_493867

theorem maria_spent : 
  let price_rose := 6
  let price_daisy := 4
  let price_bouquet := 10
  let price_vase := 8
  let num_roses := 7
  let num_daisies := 3
  let num_bouquets := 2
  let num_vases := 1
  let total_cost := (num_roses * price_rose) + (num_daisies * price_daisy) + (num_bouquets * price_bouquet) + (num_vases * price_vase)
  in total_cost = 82 :=
sorry

end maria_spent_l493_493867


namespace monthly_payment_correct_l493_493356

-- Definitions of the conditions
def purchase_price : ℝ := 130
def down_payment : ℝ := 30
def interest_rate : ℝ := 23.076923076923077 / 100
def number_of_payments : ℕ := 12

-- Definition of the interest paid
def interest_paid : ℝ := interest_rate * purchase_price

-- Definition of the total amount paid by the customer
def total_paid : ℝ := purchase_price + interest_paid

-- Definition of the remaining amount to be paid in monthly installments
def remaining_amount : ℝ := total_paid - down_payment

-- Definition of the monthly payment amount
def monthly_payment : ℝ := remaining_amount / (number_of_payments:ℝ)

-- Statement to be proved
theorem monthly_payment_correct : monthly_payment = 10.833333333333334 :=
by
  -- The proof would go here
  sorry

end monthly_payment_correct_l493_493356


namespace intersection_at_one_point_l493_493069

-- Definitions for points and lines in the context of affine geometry
structure Point :=
(x : ℝ)
(y : ℝ)

structure Line :=
(p1 : Point)
(p2 : Point)

-- Parallelogram configuration and the given conditions
variables {A B C D M P Q R S : Point}

-- Line definitions, ensuring assertions of parallel lines and intersections
def line_PR_through_M_parallel_BC (M : Point) (BC : Line) : Line := sorry -- Line PR through M parallel to BC
def line_QS_through_M_parallel_AB (M : Point) (AB : Line) : Line := sorry -- Line QS through M parallel to AB

def line_BS {B S: Point} : Line := sorry -- Line BS
def line_PD {P D: Point} : Line := sorry -- Line PD
def line_MC {M C: Point} : Line := sorry -- Line MC

-- Proof statement
theorem intersection_at_one_point
  (h_parallelogram : ∀ (A B C D : Point), -- Points A, B, C, D form a parallelogram
      AB ∥ CD ∧ BC ∥ AD)
  (h_lines : ∀ (M P Q R S : Point) (BC AB : Line),
      (PR := line_PR_through_M_parallel_BC M BC) ∧ 
      (QS := line_QS_through_M_parallel_AB M AB) ∧ 
      (P ∈ AB) ∧ (Q ∈ BC) ∧ (R ∈ CD) ∧ (S ∈ DA)) :
  ∃ (N : Point), N ∈ line_BS ∧ N ∈ line_PD ∧ N ∈ line_MC := sorry

end intersection_at_one_point_l493_493069


namespace classics_section_books_l493_493392

-- Define the number of authors
def num_authors : Nat := 6

-- Define the number of books per author
def books_per_author : Nat := 33

-- Define the total number of books
def total_books : Nat := num_authors * books_per_author

-- Prove that the total number of books is 198
theorem classics_section_books : total_books = 198 := by
  sorry

end classics_section_books_l493_493392


namespace largest_prime_factor_13231_l493_493196

-- Define the conditions
def is_prime (n : ℕ) : Prop := ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

-- State the problem as a theorem in Lean 4
theorem largest_prime_factor_13231 (H1 : 13231 = 121 * 109) 
    (H2 : is_prime 109)
    (H3 : 121 = 11^2) :
    ∃ p, is_prime p ∧ p ∣ 13231 ∧ ∀ q, is_prime q ∧ q ∣ 13231 → q ≤ p :=
by
  sorry

end largest_prime_factor_13231_l493_493196


namespace numerators_count_in_lowest_terms_l493_493334

/-- 
Given the set S of all rational numbers r, 0 < r < 1, with repeating decimal expansion 0.defdefdef...
where d, e, and f are digits such that d + e + f is not divisible by 3, the number of different
numerators when these numbers are written as fractions in lowest terms is 972.
-/
theorem numerators_count_in_lowest_terms 
  (S : Set ℚ) 
  (hS : ∀ r ∈ S, ∃ def : ℕ, r = def / 999 ∧ (r > 0 ∧ r < 1) ∧ (def % 999 ≠ 0) ∧ ¬ (def % 3 = 0)) : 
  ∃ n, n = 972 ∧ ∀ r ∈ S, ∃ numer: ℕ, numer / 999 = r ∧ ∀ numer' / 999 = r, numer = numer' :=
sorry

end numerators_count_in_lowest_terms_l493_493334


namespace increasing_iff_a_ge_half_l493_493981

noncomputable def f (a x : ℝ) : ℝ := (2 / 3) * x ^ 3 + (1 / 2) * (a - 1) * x ^ 2 + a * x + 1

theorem increasing_iff_a_ge_half (a : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → (2 * x ^ 2 + (a - 1) * x + a) ≥ 0) ↔ a ≥ -1 / 2 :=
sorry

end increasing_iff_a_ge_half_l493_493981


namespace abs_nonneg_l493_493797

theorem abs_nonneg (a : ℝ) : 0 ≤ |a| :=
sorry

end abs_nonneg_l493_493797


namespace units_digit_150_factorial_is_zero_l493_493453

theorem units_digit_150_factorial_is_zero :
  (nat.trailing_digits (nat.factorial 150) 1) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l493_493453


namespace probability_midpoint_in_T_l493_493335

def T : set (ℤ × ℤ × ℤ) := 
  { p | let ⟨x, y, z⟩ := p in 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 5 ∧ 0 ≤ z ∧ z ≤ 6 }

def is_midpoint_in_T (p1 p2 : ℤ × ℤ × ℤ) : bool :=
  let ⟨x1, y1, z1⟩ := p1 in
  let ⟨x2, y2, z2⟩ := p2 in
  let x_mid := (x1 + x2) / 2 in
  let y_mid := (y1 + y2) / 2 in
  let z_mid := (z1 + z2) / 2 in
  (x1 + x2) % 2 = 0 ∧ (y1 + y2) % 2 = 0 ∧ (z1 + z2) % 2 = 0 ∧
    x_mid ∈ (set.range (0:ℤ) (5:ℤ)) ∧ 
    y_mid ∈ (set.range (0:ℤ) (6:ℤ)) ∧ 
    z_mid ∈ (set.range (0:ℤ) (7:ℤ))

theorem probability_midpoint_in_T (p q : ℕ) (hpq_rel_prime : nat.coprime p q) 
  (hprob : ∀ p1 p2 ∈ T, is_true (is_midpoint_in_T p1 p2) → (p / q : ℚ) = 30 / 1045) :
  p + q = 1075 := 
sorry

end probability_midpoint_in_T_l493_493335


namespace range_for_m_l493_493882

-- Definition of f(x)
def f (x : ℝ) : ℝ := x^2 - 4 * x + 5

-- Define the range for m
def correct_range (m : ℝ) : Prop := 2 ≤ m ∧ m ≤ 4

-- Define the conditions
def has_max_and_min (m : ℝ) : Prop :=
  (∀ x ∈ (set.Icc 0 m), f x ≤ 5) ∧
  (∀ x ∈ (set.Icc 0 m), f x ≥ 1)

-- The theorem we want to prove
theorem range_for_m (m : ℝ) : has_max_and_min m → correct_range m :=
by {
  sorry
}

end range_for_m_l493_493882


namespace smallest_residue_l493_493649

open Nat

theorem smallest_residue (c p : ℕ) (hc : 0 < c) (hp : Nat.Prime p) (hpodd : p % 2 = 1) :
  let binom := λ n, (Nat.choose (2 * n) n) in
  let sum := ∑ n in Finset.range ((p - 1) / 2 + 1), binom n * c ^ n in
  (if c % p = 4 ^ (-1 : ℤ) % p then sum % p = 0 % p
  else if (∃ k, k ^ 2 % p = (1 - 4 * c) % p) then sum % p = 1 % p
  else sum % p = (p - 1) % p) := sorry

end smallest_residue_l493_493649


namespace solve_Diamond_l493_493280

theorem solve_Diamond :
  ∀ (Diamond : ℕ), (Diamond * 7 + 4 = Diamond * 8 + 1) → Diamond = 3 :=
by
  intros Diamond h
  sorry

end solve_Diamond_l493_493280


namespace soccer_ball_complex_numbers_l493_493107

theorem soccer_ball_complex_numbers (nodes : Finset ℕ) (polygons : Finset (Finset ℕ)) 
  (colors : Finset (Fin 3))
  (threads : ∀ (v : ℕ), Fin 3)
  (H1 : ∀ e ∈ polygons, ∀ v ∈ e, ∃ u ∈ e, threads v = threads u)
  (H2 : ∀ (v : ℕ), 3 = Finset.card (Finset.image (λ e, threads e) (nodes.filter (λ u, u = v))))
  (H3 : ∀ (v : ℕ), ∃ (z : ℂ), z ≠ 1 ∧ ∀ (p : Finset ℕ) (H4 : p ∈ polygons) (v ∈ p), ∏ v ∈ p, z = 1) :
  ∃ (f : ℕ → ℂ), (∀ v, f v ≠ 1) ∧ ∀ p ∈ polygons, ∏ v in p, f v = 1 := sorry

end soccer_ball_complex_numbers_l493_493107


namespace units_digit_factorial_150_zero_l493_493442

def units_digit (n : ℕ) : ℕ :=
  (nat.factorial n) % 10

theorem units_digit_factorial_150_zero :
  units_digit 150 = 0 :=
sorry

end units_digit_factorial_150_zero_l493_493442


namespace n_A_values_l493_493324

-- Given conditions
def n_A (A : ℝ) : ℕ :=
  {x : ℝ | x^6 - 2 * x^4 + x^2 = A}.to_finset.card

-- Proof problem statement
theorem n_A_values : {k | ∃ A : ℝ, n_A A = k} = {0, 2, 3, 4, 6} :=
by
  sorry

end n_A_values_l493_493324


namespace min_guesses_correct_l493_493817

noncomputable def min_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  min_guesses n k h = if n = 2 * k then 2 else 1 :=
by
  sorry

end min_guesses_correct_l493_493817


namespace volume_of_pyramid_l493_493024

variables (c α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)

theorem volume_of_pyramid : 
  let V := (c^3 / 24) * sin (2 * α) * tan β in
    V = (1/3) * (1/4 * c^2 * sin (2 * α)) * (c / 2 * tan β) :=
by
  sorry

end volume_of_pyramid_l493_493024


namespace M_minus_N_l493_493328

-- Define the set of roots \omega_k
noncomputable def omega (k : ℕ) : ℂ := complex.exp (2 * real.pi * complex.I * k / 101)

-- Define the set S
noncomputable def S : set ℂ := {omega k ^ k | k in finset.range 100}

-- Define M as the max unique values in S
def M : ℕ := (S.to_finset.size)

-- Define N as the min unique values in S
def N : ℕ := 1  -- Given by result that when all elements are equal

-- Theorem statement
theorem M_minus_N : M - N = 99 := 
by
  -- noncomputable instances and simplified proof placeholder
  unfold M N
  sorry

end M_minus_N_l493_493328


namespace leading_coefficient_of_polynomial_l493_493050

theorem leading_coefficient_of_polynomial :
  (∃ f : ℤ[X], (∀ x : ℤ, f.eval (x + 1) - f.eval x = 8 * x^2 + 2 * x + 6) ∧ leading_coeff f = 16 / 3) :=
begin
  sorry
end

end leading_coefficient_of_polynomial_l493_493050


namespace log_ordering_l493_493654

-- Definitions of the logarithmic expressions.
def a : ℝ := log 3 (1 / 2)
def b : ℝ := log (1 / 4) 10
def c : ℝ := log 2 (1 / 3)

-- Statement of the theorem we aim to prove.
theorem log_ordering : b < c < a := sorry

end log_ordering_l493_493654


namespace area_of_roof_l493_493403

noncomputable def roof_area (w l : ℝ) (h : ∀ (w l : ℝ), l = 4 * w ∧ l - w = 32) : ℝ :=
  w * l

theorem area_of_roof (w l : ℝ) (h : l = 4 * w ∧ l - w = 32) :
  h w l → roof_area w l h = 4096 / 9 :=
by
  sorry

end area_of_roof_l493_493403


namespace benny_seashells_l493_493156

-- Defining the conditions
def initial_seashells : ℕ := 66
def given_away_seashells : ℕ := 52

-- Statement of the proof problem
theorem benny_seashells : (initial_seashells - given_away_seashells) = 14 :=
by
  sorry

end benny_seashells_l493_493156


namespace initial_oranges_planned_l493_493012

-- Define the initial amount of oranges they planned to buy
def initial_amount := x : ℕ

-- Define the total oranges they would buy after three weeks
def total_oranges (x : ℕ) : ℕ := (x + 5) + 2 * x + 2 * x

-- Write the Lean 4 statement to prove the initial plan given the total
theorem initial_oranges_planned : ∃ x : ℕ, total_oranges x = 75 → x = 14 :=
by {
  intro h,
  use 14,
  sorry
}

end initial_oranges_planned_l493_493012


namespace coordinates_of_a_l493_493989

variables (a b : ℝ × ℝ)
variable (c : ℝ)

-- Given conditions
def magnitude_a : Prop := real.sqrt (a.1 ^ 2 + a.2 ^ 2) = 2 * real.sqrt 5
def vector_b : ℕ × ℕ := (1, 2)
def parallel_a_b : Prop := ∃ k : ℝ, a = k • vector_b

-- Main theorem to prove
theorem coordinates_of_a (h1 : magnitude_a a) (h2 : parallel_a_b a c):
  a = (2, 4) ∨ a = (-2, -4) :=
sorry

end coordinates_of_a_l493_493989


namespace number_of_correct_statements_l493_493602

variable {a b c d : ℝ}

theorem number_of_correct_statements (h1 : a > 0 ∧ 0 > b ∧ b > -a)
    (h2 : c < d ∧ d < 0) :
    (¬(a * d < b * c) ∧
    (a / d + b / c < 0) ∧
    (a - c > b - d) ∧
    (a * (d - c) > b * (d - c))) →
    3 :=
by
  sorry

end number_of_correct_statements_l493_493602


namespace num_of_int_square_fraction_l493_493925

theorem num_of_int_square_fraction :
  {n : ℤ | ∃ k : ℤ, n / (25 - n) = k^2}.to_finset.card = 2 :=
by
  sorry

end num_of_int_square_fraction_l493_493925


namespace units_digit_of_150_factorial_is_zero_l493_493436

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l493_493436


namespace standard_ellipse_equation_l493_493528

noncomputable def ellipse_equation (b e : ℝ) := 
  b = 8 ∧ e = 3/5 → 
  (x y : ℝ), x^2 / 25 + y^2 / 16 = 1 ∨ x^2 / 16 + y^2 / 25 = 1

theorem standard_ellipse_equation :
  ellipse_equation 8 (3/5) :=
by
  sorry

end standard_ellipse_equation_l493_493528


namespace average_marks_l493_493021

theorem average_marks (avg1 avg2 : ℝ) (n1 n2 : ℕ) 
  (h_avg1 : avg1 = 40) 
  (h_avg2 : avg2 = 60) 
  (h_n1 : n1 = 25) 
  (h_n2 : n2 = 30) : 
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 50.91 := 
by
  sorry

end average_marks_l493_493021


namespace triangle_side_lengths_relation_l493_493955

-- Given a triangle ABC with side lengths a, b, c
variables (a b c R d : ℝ)
-- Given orthocenter H and circumcenter O, and the radius of the circumcircle is R,
-- and distance between O and H is d.
-- Prove that a² + b² + c² = 9R² - d²

theorem triangle_side_lengths_relation (a b c R d : ℝ) (H O : Type) (orthocenter : H) (circumcenter : O)
  (radius_circumcircle : O → ℝ)
  (distance_OH : O → H → ℝ) :
  a^2 + b^2 + c^2 = 9 * R^2 - d^2 :=
sorry

end triangle_side_lengths_relation_l493_493955


namespace arith_seq_property_min_value_property_final_problem_l493_493805

-- Define the arithmetic sequence condition and the subsequent property
theorem arith_seq_property (a₁ d : ℝ) (n : ℕ) : 
  (a₃ = a₁ + 2 * d) →
  (a₇ = a₁ + 6 * d) →
  (a₅ = a₁ + 4 * d) →
  a₃ + a₇ = 2 * a₅ := by
  intros h₃ h₇ h₅
  rw [h₃, h₇, h₅]
  ring

-- Define the condition and property for minimum value
theorem min_value_property (m n : ℝ) : 
  (m > 0) →
  (n > 0) →
  (m + n = 1) →
  ∃ (m_val : ℝ) (n_val : ℝ), 
    m_val = 1/5 ∧ n_val = 4/5 ∧ (1/m_val + 4/n_val = 9) := by
  intros hm hn hsum
  use [1/5, 4/5]
  constructor
  • rfl
  constructor
  • rfl
  ring_nf
  norm_num

-- Combine the results of both theorems to state the final problem
theorem final_problem :
  -- statement B is correct.
  arith_seq_property ∧ 
  -- statement C is correct.
  min_value_property := 
by
  constructor
  • exact arith_seq_property
  • exact min_value_property

end arith_seq_property_min_value_property_final_problem_l493_493805


namespace car_original_cost_price_l493_493133

noncomputable def original_cost_price_approx := 63125.47

theorem car_original_cost_price
    (C : ℝ)
    (H : 1.30 * (0.92115 * C + 7250) = 85000) :
    abs (C - original_cost_price_approx) < 1 :=
begin
  sorry
end

end car_original_cost_price_l493_493133


namespace a_seq_general_term_max_m_l493_493263

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 1 then 1 / 2
  else (1 / (3 ^ n - 1))

noncomputable def b_seq (n : ℕ) : ℝ :=
  1 + (1 / a_seq n)

theorem a_seq_general_term :
  ∀ n : ℕ, n ≥ 1 → a_seq n = (1 / (3 ^ n - 1)) := by
  -- Proof omitted
  sorry

variable (f : ℕ → ℝ)
def f_sum (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, 1 / (n + log 3 (b_seq k))

theorem max_m :
  ∃ m : ℕ, (∀ n : ℕ, n ≥ 2 → f_sum n > m / 24) ∧ m = 13 := by
  -- Proof omitted
  sorry

end a_seq_general_term_max_m_l493_493263


namespace mirror_wall_area_ratio_l493_493519

noncomputable def area_of_mirror (side_length : ℝ) : ℝ :=
  side_length ^ 2

noncomputable def area_of_wall (width length : ℝ) : ℝ :=
  width * length

theorem mirror_wall_area_ratio 
  (side_length : ℝ) (width : ℝ) (length : ℝ)
  (h_side_length : side_length = 54)
  (h_width : width = 68)
  (h_length : length = 85.76470588235294) :
  (area_of_mirror side_length) / (area_of_wall width length) ≈ 0.5 :=
by
  sorry

end mirror_wall_area_ratio_l493_493519


namespace student_survey_l493_493810

-- Define the conditions given in the problem
theorem student_survey (S F : ℝ) (h1 : F = 25 + 65) (h2 : F = 0.45 * S) : S = 200 :=
by
  sorry

end student_survey_l493_493810


namespace units_digit_factorial_150_zero_l493_493438

def units_digit (n : ℕ) : ℕ :=
  (nat.factorial n) % 10

theorem units_digit_factorial_150_zero :
  units_digit 150 = 0 :=
sorry

end units_digit_factorial_150_zero_l493_493438


namespace elevation_area_not_possible_l493_493943

-- Given conditions
def edge_length : ℝ := 1
def plan_view_area : ℝ := 1
def min_elevation_area : ℝ := 1
def max_elevation_area : ℝ := Real.sqrt 2

-- Predicate for the exclusion of a specific area for elevation view
def elevation_area_invalid (area : ℝ) : Prop :=
  area < min_elevation_area ∨ area > max_elevation_area

-- The elevation view area in question
def specific_elevation_area : ℝ := (Real.sqrt 2 - 1) / 2

-- The problem statement
theorem elevation_area_not_possible :
  elevation_area_invalid specific_elevation_area :=
sorry

end elevation_area_not_possible_l493_493943


namespace ellipse_equation_and_max_triangle_area_l493_493218

theorem ellipse_equation_and_max_triangle_area
  (a b c : ℝ)
  (eccentricity : ℝ)
  (eq1 : a > b ∧ b > 0)
  (eq2 : c = eccentricity * a)
  (eq3 : a^2 = b^2 + c^2)
  (eq4 : rfl : sqrt 6 = | a - c |): 
  (h1 : ∀ x y, (x:ℝ)^2 / a^2 + (y:ℝ)^2 / b^2 = 1 ↔ ∃ x y, (x^2 / 6 + y^2 / 2 = 1)) ∧
  (h2 : ∃ t : ℝ, (t = 1 ∨ t = -1) → (S : ℝ)--> (S = 2 * (sqrt 3))): 
    sorry


end ellipse_equation_and_max_triangle_area_l493_493218


namespace find_original_expression_l493_493174

theorem find_original_expression (a b c X : ℤ) :
  X + (a * b - 2 * b * c + 3 * a * c) = 2 * b * c - 3 * a * c + 2 * a * b →
  X = 4 * b * c - 6 * a * c + a * b :=
by
  sorry

end find_original_expression_l493_493174


namespace dorothy_profit_l493_493563

-- Define the conditions
def expense := 53
def number_of_doughnuts := 25
def price_per_doughnut := 3

-- Define revenue and profit calculations
def revenue := number_of_doughnuts * price_per_doughnut
def profit := revenue - expense

-- Prove the profit calculation
theorem dorothy_profit : profit = 22 := by
  sorry

end dorothy_profit_l493_493563


namespace max_length_sequence_309_l493_493898

def sequence (a₁ a₂ : ℤ) : ℕ → ℤ
| 0     := a₁
| 1     := a₂
| (n+2) := sequence n - sequence (n+1)

theorem max_length_sequence_309 :
  ∃ x : ℤ, x = 309 ∧
  (let a₁ := 500 in 
  let a₂ := x in
  sequence a₁ a₂ 9 > 0 ∧
  sequence a₁ a₂ 10 > 0) :=
sorry

end max_length_sequence_309_l493_493898


namespace arrangements_count_l493_493561

theorem arrangements_count : 
  let teachers := 2
  let students := 4
  let groups := 2
  choose_1_teacher_for_A := nat.choose teachers 1
  choose_2_students_for_A := nat.choose students 2
  total_arrangements := choose_1_teacher_for_A * choose_2_students_for_A
  in
  groups = 2 → teachers = 2 → students = 4 → total_arrangements = 12 :=
by
  sorry

end arrangements_count_l493_493561


namespace sequence_max_length_x_l493_493897

theorem sequence_max_length_x (x : ℕ) : 
  (∀ n, a_n = 500 ∧ a_{n+1} = x → (a_{n+2} = a_n - a_{n+1})) →
  (a_{11} > 0 ∧ a_{10} > 0 → x = 500) :=
by
  sorry

end sequence_max_length_x_l493_493897


namespace power_of_two_divides_factorial_iff_l493_493002

theorem power_of_two_divides_factorial_iff (n : ℕ) (k : ℕ) : 2^(n - 1) ∣ n! ↔ n = 2^k := sorry

end power_of_two_divides_factorial_iff_l493_493002


namespace solve_for_x_l493_493371

theorem solve_for_x (y z x : ℝ) (h1 : 2 / 3 = y / 90) (h2 : 2 / 3 = (y + z) / 120) (h3 : 2 / 3 = (x - z) / 150) : x = 120 :=
by
  sorry

end solve_for_x_l493_493371


namespace units_digit_factorial_150_l493_493460

theorem units_digit_factorial_150 : (nat.factorial 150) % 10 = 0 :=
sorry

end units_digit_factorial_150_l493_493460


namespace cube_sum_eq_eleven_l493_493713

-- Definitions based on the conditions
variables (p q r : ℝ)
-- Assumption that p, q, r are roots of the polynomial x^3 - 2x^2 + x - 3 = 0
def root_condition : Prop :=
  polynomial.eval p (polynomial.mk [ -3, 1, -2, 1 ]) = 0 ∧ 
  polynomial.eval q (polynomial.mk [ -3, 1, -2, 1 ]) = 0 ∧ 
  polynomial.eval r (polynomial.mk [ -3, 1, -2, 1 ]) = 0

-- Using Vieta's formulas in the conditions
def vieta_condition_1 : Prop := p + q + r = 2
def vieta_condition_2 : Prop := p * q + q * r + r * p = 1
def vieta_condition_3 : Prop := p * q * r = 3

-- Main goal: Prove p^3 + q^3 + r^3 = 11
theorem cube_sum_eq_eleven 
  (hpqr : root_condition p q r)
  (hv1 : vieta_condition_1 p q r)
  (hv2 : vieta_condition_2 p q r)
  (hv3 : vieta_condition_3 p q r) : 
  p^3 + q^3 + r^3 = 11 :=
sorry

end cube_sum_eq_eleven_l493_493713


namespace fuel_cost_l493_493933

theorem fuel_cost:
  ∃ (y : ℚ), (y / 4 - y / 7 = 8) ∧ y = 224 / 3 :=
by
  use 224 / 3
  split
  sorry
  sorry

end fuel_cost_l493_493933


namespace general_formula_arithmetic_sequence_and_sum_l493_493616

theorem general_formula_arithmetic_sequence_and_sum :
  (∀ {a_n : ℕ → ℤ} {S_n : ℕ → ℤ},
    (∀ n, a_n n = a_n 1 + (n - 1) * 2) →
    (∀ n, S_n n = (n * (a_n 1) + n * (n - 1))) →
    (∃ a₁ : ℤ, S_n 2 ^ 2 = S_n 1 * S_n 4 → a_n n = 2 * n - 1)) 
∧
  (∀ {b_n : ℕ → ℤ} {T_n : ℕ → ℤ} {a_n : ℕ → ℤ},
    (∀ n, a_n n = 2 * n - 1) →
    (∀ n, b_n n = (-1)^(n-1) * (4 * n) / (a_n n * a_n (n + 1))) →
    (∀ n, T_n n = (finset.range n).sum b_n) →
      ((∀ n, nat.even n → T_n n = 2 * n / (2 * n + 1)) 
      ∧ (∀ n, nat.odd n → T_n n = (2 * n + 2) / (2 * n + 1)))) :=
sorry

end general_formula_arithmetic_sequence_and_sum_l493_493616


namespace area_of_triangle_l493_493980

def f (x : ℝ) := x + 1 / (x - 1)

theorem area_of_triangle :
  let P := (2, f 2)
  let tangent_line_y := f 2
  let vertices := [(1, 1), (1, tangent_line_y), (tangent_line_y, tangent_line_y)]
  let area := 1 / 2 * (tangent_line_y - 1) * (tangent_line_y - 1)
  area = 2 :=
by
  -- definitions
  have def_f2 : f 2 = 3 := by sorry
  have def_vertices : [(1, 1), (1, 3), (3, 3)] = vertices := by sorry
  have def_area : 1 / 2 * 2 * 2 = 2 := by sorry
  sorry

end area_of_triangle_l493_493980


namespace rotated_ordinate_l493_493223

theorem rotated_ordinate 
  (O : ℝ × ℝ) 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ)
  (hO : O = (0, 0))
  (hA : A = (3, -4)) 
  (h_rotate : ∃ θ : ℝ, θ = π / 2 ∧ B = (A.1 * real.cos θ - A.2 * real.sin θ, A.1 * real.sin θ + A.2 * real.cos θ)) :
  B.2 = 3 :=
sorry

end rotated_ordinate_l493_493223


namespace highway_length_proof_l493_493564

variable (L : ℝ) (v1 v2 : ℝ) (t : ℝ)

def highway_length : Prop :=
  v1 = 55 ∧ v2 = 35 ∧ t = 1 / 15 ∧ (L / v2 - L / v1 = t) ∧ L = 6.42

theorem highway_length_proof : highway_length L 55 35 (1 / 15) := by
  sorry

end highway_length_proof_l493_493564


namespace tan_alpha_neg_seven_l493_493231

noncomputable def tan_value (α : Real) : Prop :=
  α ∈ Ioo (Real.pi / 2) Real.pi ∧ (Real.cos α)^2 + Real.sin (Real.pi + 2 * α) = 3 / 10 → 
  Real.tan α = -7

-- Theorem statement asserting the above definition is satisfied
theorem tan_alpha_neg_seven (α : Real) : tan_value α := 
begin
  sorry
end

end tan_alpha_neg_seven_l493_493231


namespace exists_matrix_M_l493_493584

variable {R : Type} [CommRing R]

def matrix_M (M : Matrix (Fin 2) (Fin 2) R) := 
  ∀ (A : Matrix (Fin 2) (Fin 2) R),
  M ⬝ A = Matrix.of 2 2 ![![A 0 0, 2 * A 0 1], ![A 1 0, 3 * A 1 1]]

theorem exists_matrix_M :
  ∃ (M : Matrix (Fin 2) (Fin 2) R), matrix_M M := by
  use ![![1, 0], ![0, 3]]
  unfold matrix_M
  intro A
  ext i j
  fin_cases i
  fin_cases j
  { simp [Matrix.mul_apply, Matrix.of], }
  { simp [Matrix.mul_apply, Matrix.of], ring }
  { simp [Matrix.mul_apply, Matrix.of], }
  { simp [Matrix.mul_apply, Matrix.of], ring }

end exists_matrix_M_l493_493584


namespace exists_mn_coprime_l493_493739

theorem exists_mn_coprime (a b : ℤ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gcd : Int.gcd a b = 1) :
  ∃ (m n : ℕ), 1 ≤ m ∧ 1 ≤ n ∧ (a^m + b^n) % (a * b) = 1 % (a * b) :=
sorry

end exists_mn_coprime_l493_493739


namespace geometric_progression_fourth_term_l493_493029

theorem geometric_progression_fourth_term (a b c : ℝ) (h1 : a = 2^6) (h2 : b = 2^3) (h3 : c = 2^(3/2)) :
  ∃ d : ℝ, d = 2^(3/4) :=
begin
  use 2^(3/4),
  sorry
end

end geometric_progression_fourth_term_l493_493029


namespace range_of_AB_l493_493292

variable (AB BC AC : ℝ)
variable (θ : ℝ)
variable (B : ℝ)

-- Conditions
axiom angle_condition : θ = 150
axiom length_condition : AC = 2

-- Theorem to prove
theorem range_of_AB (h_θ : θ = 150) (h_AC : AC = 2) : (0 < AB) ∧ (AB ≤ 4) :=
sorry

end range_of_AB_l493_493292


namespace factors_of_5_pow_30_minus_1_between_90_and_100_l493_493558

theorem factors_of_5_pow_30_minus_1_between_90_and_100 :
  ∃ (a b : ℕ), 90 < a ∧ a < 100 ∧ 90 < b ∧ b < 100 ∧ a ≠ b ∧
    a ∣ (5^30 - 1) ∧ b ∣ (5^30 - 1) ∧ {a, b} = {91, 97} :=
by {
  -- We state the existence of two numbers a and b
  use 91,
  use 97,
  -- We state the necessary conditions:
  -- a and b are between 90 and 100
  repeat { split }; norm_num,
  apply (Nat.Prime.dvd_iff_le_or_dvd).1,
  { -- a ≠ b
    norm_num },
  -- a | (5^30 - 1)
  { apply Num.dvd_nat_of_prime_pow_sub_one,
    -- 5 is a prime number
    exact Nat.prime_5 },
  -- b | (5^30 - 1)
  { apply Num.dvd_nat_of_prime_pow_sub_one,
    -- 5 is a prime number
    exact Nat.prime_5 },
  -- {a, b} = {91, 97}
  sorry
}

end factors_of_5_pow_30_minus_1_between_90_and_100_l493_493558


namespace students_entry_rate_l493_493533

variable (x : ℝ) -- x represents the number of students entering every 3 minutes

theorem students_entry_rate :
  (20 + 14 * x - 32 = 27) → x = 3 :=
by
  intro h
  linarith

end students_entry_rate_l493_493533


namespace intersection_of_A_and_B_l493_493704

def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

theorem intersection_of_A_and_B : (A ∩ B) = {x | 2 < x ∧ x < 3} := by
  sorry

end intersection_of_A_and_B_l493_493704


namespace f_monotonic_intervals_f_zero_range_l493_493257

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  m * (2 * Real.log x - x) + 1 / (x ^ 2) - 1 / x

theorem f_monotonic_intervals (m : ℕ) :
  (if m ≤ 0 then ∀ x, x ∈ (0, 2) → f m x < f m (2 - x) ∧ ∀ x, x ∈ (2, +∞) → f m x > f m (2 + x)
   else if 0 < m ∧ m < 1 / 4 then
     ∀ x, x ∈ (0, 2) → f m x < f m (2 - x) ∧ ∀ x, x ∈ (2, +∞) → f m x > f m (2 + x) ∧
     ∀ x, x ∈(0,  sqrt m / m)or  x ∈ (2, +∞)  → f m x <f m ( sqrt m/x)∧→ ∀ x, x ∈( sqrt m/ m, 2) f m x > f m (2 ) 
   else if m = 1 / 4 then ∀ x, x ∈ (0, +∞) → f m x < f m (x)
   else if m > 1 / 4 then
      ∀ x, x ∈ (0, sqrt m  / m) → f m x < f m (sqrt m / x) ∧
      ∀ x, x ∈ (sqrt m / m, 2) → f m x > f m (sqrt m / x) ∧
      ∀ x, x ∈ (2, +∞) → f m x < f m (2 + x)
  sorry

theorem f_zero_range (m : ℝ) :
  (∀ x, 0 < x ∧ x < ∞ → f m x = 0) → (m ∈ (left_of (1 / (8 * (Real.log 2 - 1))), 0)) :=
sorry

end f_monotonic_intervals_f_zero_range_l493_493257


namespace peter_total_spent_l493_493360

/-
Peter bought a scooter for a certain sum of money. He spent 5% of the cost on the first round of repairs, another 10% on the second round of repairs, and 7% on the third round of repairs. After this, he had to pay a 12% tax on the original cost. Also, he offered a 15% holiday discount on the scooter's selling price. Despite the discount, he still managed to make a profit of $2000. How much did he spend in total, including repairs, tax, and discount if his profit percentage was 30%?
-/

noncomputable def total_spent (C S P : ℝ) : Prop :=
    (0.3 * C = P) ∧
    (0.85 * S = 1.34 * C + P) ∧
    (C = 2000 / 0.3) ∧
    (1.34 * C = 8933.33)

theorem peter_total_spent
  (C S P : ℝ)
  (h1 : 0.3 * C = P)
  (h2 : 0.85 * S = 1.34 * C + P)
  (h3 : C = 2000 / 0.3)
  : 1.34 * C = 8933.33 := by 
  sorry

end peter_total_spent_l493_493360


namespace stationary_points_cubic_l493_493886

variables (p q : ℝ)
-- Assume p and q are positive
axiom hp : 0 < p
axiom hq : 0 < q

theorem stationary_points_cubic :
  ∃ x1 x2 : ℝ, x1 = (-p + sqrt (p^2 - 3*q)) / 3 ∧ x2 = (-p - sqrt (p^2 - 3*q)) / 3 ∧
  ∀ x, deriv (fun x : ℝ => x^3 + p * x^2 + q * x) x = 0 ↔ x = x1 ∨ x = x2 :=
by
  sorry

end stationary_points_cubic_l493_493886


namespace problem_solution_l493_493264

theorem problem_solution (a b : ℝ) (h1 : {a, b / a, 1} = {a^2, a + b, 0}) : a^(2015) + b^(2016) = -1 :=
by
  have h2 : b = 0 := sorry
  have h3 : a^2 = 1 := sorry
  /- From h3 we have two cases a = 1 or a = -1,
     since a = 1 does not satisfy the set equality we get a = -1 -/
  sorry

end problem_solution_l493_493264


namespace monotonic_intervals_of_f_range_of_m_for_two_zeros_l493_493258

noncomputable def f (m x : ℝ) : ℝ := m * (2 * Real.log x - x) + (1 / x^2) - (1 / x)

theorem monotonic_intervals_of_f (m : ℝ) :
  (∀ x ∈ Ioo (0 : ℝ) 2, deriv (f m) x < 0) ∧
  (∀ x ∈ Ioi 2, deriv (f m) x > 0) ∨
  (0 < m ∧ m < (1 / 4) ∧
    (∀ x ∈ Ioo (0 : ℝ) 2, deriv (f m) x < 0) ∧
    (∀ x ∈ Ioo (2 : ℝ) (sqrt m / m), deriv (f m) x > 0) ∧
    (∀ x ∈ Ioi (sqrt m / m), deriv (f m) x < 0)) ∨
  (m = 1 / 4 ∧ (∀ x ∈ Ioi (0 : ℝ), deriv (f m) x < 0)) ∨
  (m > 1 / 4 ∧ (∀ x ∈ Ioo (0 : ℝ) (sqrt m / m), deriv (f m) x < 0) ∧
    (∀ x ∈ Ioo (sqrt m / m) 2, deriv (f m) x > 0) ∧
    (∀ x ∈ Ioi 2, deriv (f m) x < 0))
  :=
  sorry

theorem range_of_m_for_two_zeros :
  (∃ m : ℝ, (1 / (8 * (Real.log 2 - 1)) < m) ∧ m < 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f m x₁ = 0 ∧ f m x₂ = 0))
  :=
  sorry

end monotonic_intervals_of_f_range_of_m_for_two_zeros_l493_493258


namespace store_owner_cases_in_April_l493_493520

theorem store_owner_cases_in_April 
  (bottles_per_case : ℕ) 
  (cases_in_May : ℕ) 
  (total_bottles : ℕ) 
  (bottles_in_April_and_May : cases_in_May * bottles_per_case + total_bottles) 
  (total_cases_in_April_and_May_bottles_eq : 20 * x + 600 = 1000) 
  (h_total_bottles : total_bottles = 1000) : 
  ∃ x : ℕ, x = 20 :=
by
  sorry

end store_owner_cases_in_April_l493_493520


namespace angle_B_45_l493_493733

variables {A B C : Type} [inhabited A] [inhabited B] [inhabited C]

-- Variables for points
variables (O : Type) [inhabited O] (E F : Type) [inhabited E] [inhabited F]

-- Variables for triangles, angles and circle
variables (ABC : Type) [inhabited ABC] (AOC : Type) [inhabited AOC]

-- Definitions for circumcenter and circumcircle
def circumcenter (O : Type) (ABC : Type) : Prop := sorry
def circumcircle (AOC : Type) (E F : Type) : Prop := sorry
def bisects_area (EF : Type) (ABC : Type) : Prop := sorry
def angle_B (A B C : Type) : ℝ := sorry -- measure of angle B in radians

-- Given conditions
axiom (h1 : circumcenter O ABC)
axiom (h2 : circumcircle AOC E F)
axiom (h3 : bisects_area EF ABC)

-- The proof goal
theorem angle_B_45 (O : Type) (ABC AOC : Type) (E F : Type) [inhabited O] [inhabited E] [inhabited F] [inhabited ABC] [inhabited AOC] :
  circumcenter O ABC → circumcircle AOC E F → bisects_area EF ABC → angle_B A B C = π / 4 :=
by
  intros
  sorry

end angle_B_45_l493_493733


namespace repeating_decimal_to_fraction_l493_493915

/-- The repeating decimal 0.565656... equals the fraction 56/99. -/
theorem repeating_decimal_to_fraction : 
  let a := 56 / 100
      r := 1 / 100
  in (a / (1 - r) = 56 / 99) := 
by
  let a := 56 / 100
  let r := 1 / 100
  have h1 : 0 < r, by norm_num
  have h2 : r < 1, by norm_num
  have sum_inf_geo_series : a / (1 - r) = 56 / 99 := by sorry
  use sum_inf_geo_series
  sorry

end repeating_decimal_to_fraction_l493_493915


namespace cylinder_volume_ratio_l493_493112

theorem cylinder_volume_ratio (r1 r2 V1 V2 : ℝ) (h1 : 2 * Real.pi * r1 = 6) (h2 : 2 * Real.pi * r2 = 10) (hV1 : V1 = Real.pi * r1^2 * 10) (hV2 : V2 = Real.pi * r2^2 * 6) :
  V1 < V2 → (V2 / V1) = 5 / 3 :=
by
  sorry

end cylinder_volume_ratio_l493_493112


namespace closest_point_is_correct_l493_493197

def point := (ℝ × ℝ × ℝ)

noncomputable def closest_point (A : point) (plane : point → Prop) : point :=
  let normal := (4, 3, -2) in
  let t := (27 : ℝ) / 29 in
  (2 + 4 * t, 1 + 3 * t, -1 - 2 * t)

def is_on_plane (p : point) : Prop :=
  let (x, y, z) := p in
  4 * x + 3 * y - 2 * z = 40

theorem closest_point_is_correct :
  let A := (2, 1, -1) in
  let plane := is_on_plane in
  closest_point A plane = (134 / 29, 82 / 29, -55 / 29) := by
  sorry

end closest_point_is_correct_l493_493197


namespace fixed_point_AB_intersect_l493_493219

noncomputable def a_squared : ℝ := 2
noncomputable def b_squared : ℝ := 1
noncomputable def P : ℝ × ℝ := (1, real.sqrt 2 / 2)
noncomputable def k1 : ℝ := sorry -- The slope of line MA
noncomputable def k2 : ℝ := sorry -- The slope of line MB

theorem fixed_point_AB_intersect :
  k1 + k2 = 2 →
  ∃ (x y : ℝ), 
    (x, y) = (-1, -1) ∧
    (∀ (A B : ℝ × ℝ), 
      (A ∈ set_of (λ p, (p.1^2 / a_squared) + (p.2^2 / b_squared) = 1) ∧
       B ∈ set_of (λ p, (p.1^2 / a_squared) + (p.2^2 / b_squared) = 1) ∧
       A ≠ B ∧ 
       slopes of line segments (A to top vertex M) and (B to M)) implies the line segment AB passes through (x, y)) :=
begin
  sorry
end

end fixed_point_AB_intersect_l493_493219


namespace num_of_int_square_fraction_l493_493924

theorem num_of_int_square_fraction :
  {n : ℤ | ∃ k : ℤ, n / (25 - n) = k^2}.to_finset.card = 2 :=
by
  sorry

end num_of_int_square_fraction_l493_493924


namespace alice_wrong_questions_l493_493857

theorem alice_wrong_questions :
  ∃ a b e : ℕ,
    (a + b = 6 + 8 + e) ∧
    (a + 8 = b + 6 + 3) ∧
    a = 9 :=
by {
  sorry
}

end alice_wrong_questions_l493_493857


namespace geometric_sequence_sine_l493_493234

theorem geometric_sequence_sine (α β γ : ℝ) (h1: β = 2 * α) (h2: γ = 4 * α) (h3: α ≥ 0 ∧ α ≤ 2 * Real.pi) : 
    (sin β - sin α = sin γ - sin β) →
    (α = 2 * Real.pi / 3 ∧ β = 4 * Real.pi / 3 ∧ γ = 8 * Real.pi / 3) ∨
    (α = 4 * Real.pi / 3 ∧ β = 8 * Real.pi / 3 ∧ γ = 16 * Real.pi / 3) :=
by
  sorry

end geometric_sequence_sine_l493_493234


namespace sqrt_condition_l493_493283

theorem sqrt_condition (x y : ℝ) (h : x * y ≠ 0) :
  (sqrt (4 * x^2 * y^3) = -2 * x * y * sqrt y) ↔ (x < 0 ∧ y > 0) :=
by
  sorry

end sqrt_condition_l493_493283


namespace isosceles_triangle_perimeter_l493_493152

-- Definitions and conditions
-- Define the lengths of the three sides of the triangle
def a : ℕ := 3
def b : ℕ := 8

-- Define that the triangle is isosceles
def is_isosceles_triangle := 
  (a = a) ∨ (b = b) ∨ (a = b)

-- Perimeter of the triangle
def perimeter (x y z : ℕ) := x + y + z

-- The theorem we need to prove
theorem isosceles_triangle_perimeter : is_isosceles_triangle → (a + b + b = 19) :=
by
  intro h
  sorry

end isosceles_triangle_perimeter_l493_493152


namespace greatestSymmetry_l493_493123

def linesOfSymmetry (shape : String) : ℕ∞ :=
  match shape with
  | "Circle" => ⊤  -- representing infinity
  | "Isosceles Right Triangle" => 1
  | "Regular Pentagon" => 5
  | "Regular Hexagon" => 6
  | _ => 0

theorem greatestSymmetry (shapes : List String) : 
  "Circle" ∈ shapes → ∀ shape ∈ shapes, linesOfSymmetry "Circle" ≥ linesOfSymmetry shape :=
by {
  intros h shapes nh,
  cases h,
  exact ⟨_, shapes, linesOfSymmetry "Circle" ≥ linesOfSymmetry shape⟩,
  sorry
}

end greatestSymmetry_l493_493123


namespace problem_solution_l493_493634

theorem problem_solution :
  ∀ (f : ℝ → ℝ) (c : ℝ),
    (0 < c ∧ c < 1) ∧ 
    (∀ x, 0 < x ∧ x < c → f x = c * x + 1) ∧
    (∀ x, c ≤ x ∧ x < 1 → f x = 2 * x / c^2 + 1) ∧
    (f (c^2) = 9 / 8)
    → (c = 1 / 2) ∧ 
       (∀ x, f x > (Real.sqrt 2) / 8 + 1 ↔ (Real.sqrt 2) / 4 < x ∧ x < 1) :=
by
  intros f c conditions,
  sorry

end problem_solution_l493_493634


namespace problem1_problem2_l493_493828

section Problem1

-- Given: a, b, c ∈ ℝ, Prove: a^2 + b^2 + c^2 ≥ ab + bc + ca
theorem problem1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end Problem1

section Problem2

-- Given: a = x^2 - 2y + π/2, b = y^2 - 2z + π/3, c = z^2 - 2x + π/6
-- Prove: at least one of a, b, c is greater than 0
theorem problem2 (x y z : ℝ) :
  let a := x^2 - 2*y + (Real.pi / 2),
      b := y^2 - 2*z + (Real.pi / 3),
      c := z^2 - 2*x + (Real.pi / 6)
  in a > 0 ∨ b > 0 ∨ c > 0 :=
by
  sorry

end Problem2

end problem1_problem2_l493_493828


namespace sequence_max_length_x_l493_493896

theorem sequence_max_length_x (x : ℕ) : 
  (∀ n, a_n = 500 ∧ a_{n+1} = x → (a_{n+2} = a_n - a_{n+1})) →
  (a_{11} > 0 ∧ a_{10} > 0 → x = 500) :=
by
  sorry

end sequence_max_length_x_l493_493896


namespace max_length_sequence_l493_493890

noncomputable def max_length_x : ℕ := 309

theorem max_length_sequence :
  let a : ℕ → ℤ := λ n, match n with
    | 0 => 500
    | 1 => max_length_x
    | n + 2 => a n - a (n + 1)
  in
  ∀ n : ℕ, (∀ m < n, a m ≥ 0) →
    (a n < 0) →
    (309 = max_length_x) :=
by
  intro a
  intro n
  intro h_pos
  intro h_neg
  sorry

end max_length_sequence_l493_493890


namespace extremum_at_one_l493_493288

-- Definition for the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x

-- Statement to prove
theorem extremum_at_one (a : ℝ) (h : deriv (λ x => f x a) 1 = 0) : a = Real.exp 1 :=
by
  sorry

end extremum_at_one_l493_493288


namespace problem_l493_493245

noncomputable def f (a₁ a₂ a₃ a₄ : ℤ) (b₁ b₂ b₃ b₄ : ℤ) : ℤ :=
  b₁ - b₂ + b₃ - b₄

theorem problem (b₁ b₂ b₃ b₄ : ℤ) :
  (∀ x : ℤ, x^4 + 2 * x^3 + x + 6 = (x + 1)^4 + b₁ * (x + 1)^3 + b₂ * (x + 1)^2 + b₃ * (x + 1) + b₄) →
  f 2 0 1 6 b₁ b₂ b₃ b₄ = -3 :=
begin
  intros h,
  -- Proof steps would go here
  sorry,
end

end problem_l493_493245


namespace required_speed_l493_493353

noncomputable def distance_travelled_late (d: ℝ) (t: ℝ) : ℝ :=
  50 * (t + 1/12)

noncomputable def distance_travelled_early (d: ℝ) (t: ℝ) : ℝ :=
  70 * (t - 1/12)

theorem required_speed :
  ∃ (s: ℝ), s = 58 ∧ 
  (∀ (d t: ℝ), distance_travelled_late d t = d ∧ distance_travelled_early d t = d → 
  d / t = s) :=
by
  sorry

end required_speed_l493_493353


namespace area_ratio_bounds_l493_493211

variables {a : ℝ}

/-- Given conditions for the quadrilateral inscribed in a unit circle -/
def conditions (a : ℝ) : Prop :=
  a * real.sqrt 2 < a ∧ a < 2 ∧
  let S_ABCD_min := real.sqrt (4 - a^2) in
  ∃ (ABCD : Type) [quadrilateral ABCD] (unit_circle : unit_circle) (center_within : center_within ABCD unit_circle),
    max_side_length ABCD = a ∧ min_side_length ABCD = S_ABCD_min

/-- The minimum and maximum values of the area ratio S_A'B'C'D'/S_ABCD -/
theorem area_ratio_bounds {a : ℝ} (h : conditions a) :
  ∃ (min_ratio max_ratio : ℝ),
    min_ratio = 4 / (a * real.sqrt (4 - a^2)) ∧
    max_ratio = 8 / (a^2 * (4 - a^2)) :=
by
  sorry

end area_ratio_bounds_l493_493211


namespace sum_mod_seven_l493_493198

theorem sum_mod_seven : 
    let sum := 5000 + 5001 + 5002 + 5003 + 5004 in
    sum % 7 = 0 :=
by
    sorry

end sum_mod_seven_l493_493198


namespace c_share_l493_493824

theorem c_share (A B C : ℝ) 
  (h1 : A = (1 / 2) * B)
  (h2 : B = (1 / 2) * C)
  (h3 : A + B + C = 392) : 
  C = 224 :=
by
  sorry

end c_share_l493_493824


namespace sum_of_consecutive_odds_eq_power_l493_493764

theorem sum_of_consecutive_odds_eq_power (n : ℕ) (k : ℕ) (hn : n > 0) (hk : k ≥ 2) :
  ∃ a : ℤ, n * (2 * a + n) = n^k ∧
            (∀ i : ℕ, i < n → 2 * a + 2 * (i : ℤ) + 1 = 2 * a + 1 + 2 * i) :=
by
  sorry

end sum_of_consecutive_odds_eq_power_l493_493764


namespace cylinder_volume_ratio_l493_493116

-- First define the problem: a 6 x 10 rectangle rolled to form two different cylinders
theorem cylinder_volume_ratio : 
  let r1 := 3 / Real.pi in
  let V1 := Real.pi * r1^2 * 10 in
  let r2 := 5 / Real.pi in
  let V2 := Real.pi * r2^2 * 6 in
  V2 / V1 = 5 / 3 :=
by
  -- The proof steps are omitted as the theorem states only.
  sorry

end cylinder_volume_ratio_l493_493116


namespace square_area_inscribed_triangle_l493_493141

-- Definitions from the conditions of the problem
variable (EG : ℝ) (hF : ℝ)

-- Since EG = 12 inches and the altitude from F to EG is 7 inches
theorem square_area_inscribed_triangle 
(EG_eq : EG = 12) 
(hF_eq : hF = 7) :
  ∃ (AB : ℝ), AB ^ 2 = 36 :=
by 
  sorry

end square_area_inscribed_triangle_l493_493141


namespace sphere_surface_area_l493_493376

theorem sphere_surface_area (r : ℝ) (h : π * r^2 = 81 * π) : 4 * π * r^2 = 324 * π :=
  sorry

end sphere_surface_area_l493_493376


namespace total_tickets_sold_correct_total_tickets_sold_is_21900_l493_493565

noncomputable def total_tickets_sold : ℕ := 5400 + 16500

theorem total_tickets_sold_correct :
    total_tickets_sold = 5400 + 5 * (16500 / 5) :=
by
  rw [Nat.div_mul_cancel]
  sorry

-- The following theorem states the main proof equivalence:
theorem total_tickets_sold_is_21900 :
    total_tickets_sold = 21900 :=
by
  sorry

end total_tickets_sold_correct_total_tickets_sold_is_21900_l493_493565


namespace cylinder_volume_ratio_l493_493111

theorem cylinder_volume_ratio (r1 r2 V1 V2 : ℝ) (h1 : 2 * Real.pi * r1 = 6) (h2 : 2 * Real.pi * r2 = 10) (hV1 : V1 = Real.pi * r1^2 * 10) (hV2 : V2 = Real.pi * r2^2 * 6) :
  V1 < V2 → (V2 / V1) = 5 / 3 :=
by
  sorry

end cylinder_volume_ratio_l493_493111


namespace units_digit_factorial_150_zero_l493_493441

def units_digit (n : ℕ) : ℕ :=
  (nat.factorial n) % 10

theorem units_digit_factorial_150_zero :
  units_digit 150 = 0 :=
sorry

end units_digit_factorial_150_zero_l493_493441


namespace pipe_q_fills_cistern_in_15_minutes_l493_493421

theorem pipe_q_fills_cistern_in_15_minutes :
  ∃ T : ℝ, 
    (1/12 * 2 + 1/T * 2 + 1/T * 10.5 = 1) → 
    T = 15 :=
by {
  -- Assume the conditions and derive T = 15
  sorry
}

end pipe_q_fills_cistern_in_15_minutes_l493_493421


namespace largest_m_is_795_l493_493047

noncomputable def largest_possible_m : ℕ :=
  let prime_less_than_10 := [2, 3, 5, 7] in
  let all_combinations := [(2, 3), (5, 3), (7, 3), (3, 7)] in
  all_combinations.foldr (λ (p : ℕ × ℕ) acc,
    let (x, y) := p,
    let candidate := x * y * (10 * x + y) in
    if candidate < 1000 ∧ candidate > acc then candidate else acc) 0

theorem largest_m_is_795 : largest_possible_m = 795 := by
  sorry

end largest_m_is_795_l493_493047


namespace units_digit_of_150_factorial_is_zero_l493_493445

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l493_493445


namespace bradley_travel_time_l493_493539

theorem bradley_travel_time (T : ℕ) (h1 : T / 4 = 20) (h2 : T / 3 = 45) : T - 20 = 280 :=
by
  -- Placeholder for proof
  sorry

end bradley_travel_time_l493_493539


namespace distinct_arrangements_l493_493784

open Nat

theorem distinct_arrangements (n k : ℕ) (n_i : Fin k → ℕ)
    (h_sum : (Finset.univ.sum (λ i, n_i i)) = n) :
    ∃ (arrangements : ℕ), arrangements = factorial n / (Finset.univ.prod (λ i, factorial (n_i i))) :=
by
  use factorial n / (Finset.univ.prod (λ i, factorial (n_i i)))
  sorry

end distinct_arrangements_l493_493784


namespace extra_apples_proof_l493_493054

def total_apples (red_apples : ℕ) (green_apples : ℕ) : ℕ :=
  red_apples + green_apples

def apples_taken_by_students (students : ℕ) : ℕ :=
  students

def extra_apples (total_apples : ℕ) (apples_taken : ℕ) : ℕ :=
  total_apples - apples_taken

theorem extra_apples_proof
  (red_apples : ℕ) (green_apples : ℕ) (students : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : students = 21) :
  extra_apples (total_apples red_apples green_apples) (apples_taken_by_students students) = 35 :=
by
  sorry

end extra_apples_proof_l493_493054


namespace cylinder_volume_ratio_l493_493118

noncomputable def volume_of_cylinder (height : ℝ) (circumference : ℝ) : ℝ := 
  let r := circumference / (2 * Real.pi)
  Real.pi * r^2 * height

theorem cylinder_volume_ratio :
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  max V1 V2 / min V1 V2 = 5 / 3 :=
by
  let V1 := volume_of_cylinder 10 6
  let V2 := volume_of_cylinder 6 10
  have hV1 : V1 = 90 / Real.pi := sorry
  have hV2 : V2 = 150 / Real.pi := sorry
  sorry

end cylinder_volume_ratio_l493_493118


namespace max_product_of_three_numbers_l493_493171

-- Definitions of the set and the constraint of selecting three different numbers
def number_set : set ℤ := { -5, -4, -1, 2, 6 }

-- Function to calculate the product of three numbers
def product_of_three (a b c : ℤ) : ℤ := a * b * c

-- Define a condition to check if a set contains three elements
def has_three_elements (s : set ℤ) : Prop := s.card = 3

-- State and statement of the problem to be proved
theorem max_product_of_three_numbers : 
  ∃ (a b c : ℤ), a ∈ number_set ∧ b ∈ number_set ∧ c ∈ number_set ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ product_of_three a b c = 120 :=
sorry

end max_product_of_three_numbers_l493_493171


namespace votes_for_crow_l493_493102

noncomputable def num_votes_crow (P V K : ℕ) : Prop :=
  let total_votes := 59
  let P_V_range := 2 ≤ P + V ∧ P + V ≤ 28
  let V_K_range := 5 ≤ V + K ∧ V + K ≤ 31
  let K_P_range := 7 ≤ K + P ∧ K + P ≤ 33
  let total_range := 46 ≤ total_votes ∧ total_votes ≤ 72 in
  total_range ∧ P_V_range ∧ V_K_range ∧ K_P_range ∧ (P + V + K = 46) ∧ (V = 13)

theorem votes_for_crow :
  ∃ (P V K : ℕ), num_votes_crow P V K :=
sorry

end votes_for_crow_l493_493102


namespace characterize_functional_equation_l493_493575

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y

theorem characterize_functional_equation (f : ℝ → ℝ) (h : satisfies_condition f) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end characterize_functional_equation_l493_493575


namespace analytical_expression_increasing_function_solve_inequality_l493_493238

-- Definitions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := (a * x + b) / (1 + x^2)

-- Conditions
axiom f_odd : odd_function f
axiom f_value_at_half : f (1 / 2) = 2 / 5

-- Statements to Prove
theorem analytical_expression (h_odd : odd_function f) (h_half : f (1 / 2) = 2 / 5) : ∀ x, f x = x / (1 + x^2) := sorry

theorem increasing_function (h_odd : odd_function f) : ∀ m n : ℝ, -1 < m ∧ m < n ∧ n < 1 → f m < f n := sorry

theorem solve_inequality (h_odd : odd_function f) (h_increasing : ∀ m n : ℝ, -1 < m ∧ m < n ∧ n < 1 → f m < f n) :
  ∀ t : ℝ, 0 < t ∧ t < 1 / 2 → f (t - 1) + f t < 0 := sorry

end analytical_expression_increasing_function_solve_inequality_l493_493238


namespace determine_alpha_set_l493_493169

noncomputable def satisfies_condition (α : ℂ) :=
  ∀ (z₁ z₂ : ℂ), abs z₁ < 1 → abs z₂ < 1 → z₁ ≠ z₂ → 
  (z₁ + α)^2 + α * conj z₁ ≠ (z₂ + α)^2 + α * conj z₂

theorem determine_alpha_set :
  {α : ℂ | satisfies_condition α} = {α | abs α ≥ 2} :=
sorry

end determine_alpha_set_l493_493169


namespace exist_two_lines_with_angle_less_than_26_degrees_l493_493210

theorem exist_two_lines_with_angle_less_than_26_degrees 
  (L : Fin 7 → Set (ℝ × ℝ)) 
  (h1 : ∀ i j : Fin 7, i ≠ j → ¬(Parallel (L i) (L j))) :
  ∃ i j : Fin 7, i ≠ j ∧ angle_between (L i) (L j) < (26 : ℝ) :=
begin
  sorry
end

end exist_two_lines_with_angle_less_than_26_degrees_l493_493210


namespace minimal_sum_of_distances_l493_493380

theorem minimal_sum_of_distances :
  ∀ (x_1 x_2 x_3 x_4 x_5 : ℕ), 
    x_1 + x_2 + x_3 + x_4 + x_5 = 99999 →
    x_2 + x_3 + x_4 + x_5 ≥ 9999 →
    x_3 + x_4 + x_5 ≥ 999 →
    x_4 + x_5 ≥ 99 →
    x_5 ≥ 9 →
    (x_1 + 2 * x_2 + 3 * x_3 + 4 * x_4 + 5 * x_5) ≥ 101105 :=
begin
  sorry
end

end minimal_sum_of_distances_l493_493380


namespace find_positive_real_number_l493_493912

theorem find_positive_real_number (x : ℝ) (hx : x = 25 + 2 * Real.sqrt 159) :
  1 / 2 * (3 * x ^ 2 - 1) = (x ^ 2 - 50 * x - 10) * (x ^ 2 + 25 * x + 5) :=
by
  sorry

end find_positive_real_number_l493_493912


namespace binary_to_decimal_110101_l493_493877

theorem binary_to_decimal_110101 : 
  binary_to_decimal "110101" = 53 :=
sorry

end binary_to_decimal_110101_l493_493877


namespace find_xyz_l493_493709

open Complex

-- Definitions of the variables and conditions
variables {a b c x y z : ℂ} (h_a_ne_zero : a ≠ 0) (h_b_ne_zero : b ≠ 0) (h_c_ne_zero : c ≠ 0)
  (h_x_ne_zero : x ≠ 0) (h_y_ne_zero : y ≠ 0) (h_z_ne_zero : z ≠ 0)
  (h1 : a = (b - c) * (x + 2))
  (h2 : b = (a - c) * (y + 2))
  (h3 : c = (a - b) * (z + 2))
  (h4 : x * y + x * z + y * z = 12)
  (h5 : x + y + z = 6)

-- Statement of the theorem
theorem find_xyz : x * y * z = 7 := 
by
  -- Proof steps to be filled in
  sorry

end find_xyz_l493_493709


namespace find_b_l493_493535

theorem find_b (a b c d : ℝ) (h : ∃ k : ℝ, 2 * k = π ∧ k * (b / 2) = π) : b = 4 :=
by
  sorry

end find_b_l493_493535


namespace no_solution_exists_l493_493576

theorem no_solution_exists :
  ∀ (x y p : ℕ), 
    x > 0 → 
    y > 0 → 
    p.prime → 
    (real.cbrt x + real.cbrt y = real.cbrt p) → 
    false :=
by sorry

end no_solution_exists_l493_493576


namespace probability_blue_given_popped_is_18_over_53_l493_493120

section PopcornProblem

/-- Representation of probabilities -/
def prob_white : ℚ := 1 / 2
def prob_yellow : ℚ := 1 / 4
def prob_blue : ℚ := 1 / 4

def pop_white_given_white : ℚ := 1 / 2
def pop_yellow_given_yellow : ℚ := 3 / 4
def pop_blue_given_blue : ℚ := 9 / 10

/-- Joint probabilities of kernel popping -/
def prob_white_popped : ℚ := prob_white * pop_white_given_white
def prob_yellow_popped : ℚ := prob_yellow * pop_yellow_given_yellow
def prob_blue_popped : ℚ := prob_blue * pop_blue_given_blue

/-- Total probability of popping -/
def prob_popped : ℚ := prob_white_popped + prob_yellow_popped + prob_blue_popped

/-- Conditional probability of being a blue kernel given that it popped -/
def prob_blue_given_popped : ℚ := prob_blue_popped / prob_popped

/-- The main theorem to prove the final probability -/
theorem probability_blue_given_popped_is_18_over_53 :
  prob_blue_given_popped = 18 / 53 :=
by sorry

end PopcornProblem

end probability_blue_given_popped_is_18_over_53_l493_493120


namespace solve_inequality_l493_493883

theorem solve_inequality :
  {x : ℝ | x^2 - 9 * x + 14 < 0} = {x : ℝ | 2 < x ∧ x < 7} := sorry

end solve_inequality_l493_493883


namespace period_tan_2x_l493_493791

theorem period_tan_2x (x : ℝ) : (∃ T : ℝ, ∀ x : ℝ, tan (2 * (x + T)) = tan (2 * x)) → T = π / 2 :=
by
  sorry

end period_tan_2x_l493_493791


namespace geometric_sequence_property_l493_493612

noncomputable def S_n (n : ℕ) (a_n : ℕ → ℕ) : ℕ := 3 * 2^n - 3

noncomputable def a_n (n : ℕ) : ℕ := 3 * 2^(n-1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n-1)

noncomputable def T_n (n : ℕ) : ℕ := 2^n - 1

theorem geometric_sequence_property (n : ℕ) (hn : n ≥ 0) :
  T_n n < b_n (n+1) :=
by
  sorry

end geometric_sequence_property_l493_493612


namespace trapezoid_area_even_integer_l493_493865

-- Definitions
variables (AB CD BC AD r : ℕ)
variables (trapezoid_is_isosceles : Prop)
variables (AD_perp_BC : Prop)
variables (AD_is_tangent : Prop)
variables (BC_passes_center : Prop)

-- Lean 4 statement
theorem trapezoid_area_even_integer
  (trapezoid_is_isosceles : trapezoid_is_isosceles)
  (AD_perp_BC : AD_perp_BC)
  (AD_is_tangent : AD_is_tangent)
  (BC_passes_center : BC_passes_center) :
  (AB = 4 ∧ CD = 2 ∨ AB = 8 ∧ CD = 4 ∨ AB = 12 ∧ CD = 6) →
  even ((AB + CD) * r) :=
begin
  intros h,
  cases h,
  { have h_even_0 : even ((4 + 2) * r), sorry, 
    exact h_even_0 },
  cases h,
  { have h_even_1 : even ((8 + 4) * r), sorry,
    exact h_even_1 },
  { have h_even_2 : even ((12 + 6) * r), sorry, 
    exact h_even_2 },
end

end trapezoid_area_even_integer_l493_493865


namespace circles_C1_C2_intersect_C1_C2_l493_493969

noncomputable def center1 : (ℝ × ℝ) := (5, 3)
noncomputable def radius1 : ℝ := 3

noncomputable def center2 : (ℝ × ℝ) := (2, -1)
noncomputable def radius2 : ℝ := Real.sqrt 14

noncomputable def distance : ℝ := Real.sqrt ((5 - 2)^2 + (3 + 1)^2)

def circles_intersect : Prop :=
  radius2 - radius1 < distance ∧ distance < radius2 + radius1

theorem circles_C1_C2_intersect_C1_C2 : circles_intersect :=
by
  -- The proof of this theorem is to be worked out using the given conditions and steps.
  sorry

end circles_C1_C2_intersect_C1_C2_l493_493969


namespace neither_necessary_nor_sufficient_l493_493235

def p (x y : ℝ) : Prop := x > 1 ∧ y > 1
def q (x y : ℝ) : Prop := x + y > 3

theorem neither_necessary_nor_sufficient :
  ¬ (∀ x y, q x y → p x y) ∧ ¬ (∀ x y, p x y → q x y) :=
by
  sorry

end neither_necessary_nor_sufficient_l493_493235


namespace number_of_subsets_of_246_l493_493048

theorem number_of_subsets_of_246 : (set.finite.to_finset {2, 4, 6}).powerset.card = 8 := 
by
  sorry

end number_of_subsets_of_246_l493_493048


namespace eval_first_expr_eval_second_expr_l493_493104

-- Calculating the value of the first expression
theorem eval_first_expr :
    (2 + 1 / 4) ^ (1 / 2) - (-9.6) ^ 0 - (3 + 3 / 8) ^ (2 / 3) + 1.5 ^ 2 + (Real.sqrt 2 * 43) ^ 4 = 5 / 4 + 4 * 43 ^ 4 :=
by
  sorry

-- Calculating the value of the second expression
theorem eval_second_expr (K : ℝ) (hK : K = log ((8 * Real.sqrt (27)) / (Real.sqrt 1000))) :
    (log (Real.sqrt 27) + log 8 - log (Real.sqrt 1000)) / (1 / 2 * log 0.3 + log 2) + (Real.sqrt 5 - 2) ^ 0 + 0.027 ^ (-1 / 3) * (-1 / 3) ^ (-2) = K + 4 :=
by
  sorry

end eval_first_expr_eval_second_expr_l493_493104


namespace calculate_expression_l493_493544

theorem calculate_expression :
  let num1 := 2468
  let base5_to_decimal := 50   -- 200_{5} = 50_{10}
  let base7_to_decimal := 1261 -- 3451_{7} = 1261_{10}
  let num2 := 7891
  (num1 / base5_to_decimal).to_int - base7_to_decimal + num2 = 6679 := by
  let num1 := 2468
  let base5_to_decimal := 50
  let base7_to_decimal := 1261
  let num2 := 7891
  sorry

end calculate_expression_l493_493544


namespace shortest_path_l493_493412
noncomputable theory

variables (a d : ℝ) 

open Real

theorem shortest_path (h : d / a < 0.746) : 
  let direct_path := a * sqrt 2 in
  let around_path := 2 * a - d + (d * pi / 4) in
  let through_path := a * sqrt 2 - d + (d * pi / 2) in
  through_path < min direct_path around_path :=
sorry

end shortest_path_l493_493412


namespace find_points_on_line_of_tangents_at_60_degree_l493_493357

noncomputable def is_tangent (f : ℝ → ℝ) (p : ℝ × ℝ) (k : ℝ) : Prop :=
  ∀ x, f x = p.2 + k * (x - p.1) ↔ (x = p.1)

theorem find_points_on_line_of_tangents_at_60_degree :
  ∃ y : ℝ, ((is_tangent (λ x : ℝ, x^2 / 4) (sqrt 3, y) _) ∧
            (is_tangent (λ x : ℝ, x^2 / 4) (sqrt 3, y) _)) ∧
            (| atan (sqrt 3 / (1 + y)) | = π / 3 ∨ y = 0 ∨ y = -10 / 3) :=
  sorry

end find_points_on_line_of_tangents_at_60_degree_l493_493357


namespace problem_proof_l493_493411

theorem problem_proof (N : ℤ) (h : N / 5 = 4) : ((N - 10) * 3) - 18 = 12 :=
by
  -- proof goes here
  sorry

end problem_proof_l493_493411


namespace units_digit_factorial_150_l493_493457

theorem units_digit_factorial_150 : (nat.factorial 150) % 10 = 0 :=
sorry

end units_digit_factorial_150_l493_493457


namespace sum_multiple_of_3_probability_l493_493067

noncomputable def probability_sum_multiple_of_3 (faces : List ℕ) (rolls : ℕ) (multiple : ℕ) : ℚ :=
  if rolls = 3 ∧ multiple = 3 ∧ faces = [1, 2, 3, 4, 5, 6] then 1 / 3 else 0

theorem sum_multiple_of_3_probability :
  probability_sum_multiple_of_3 [1, 2, 3, 4, 5, 6] 3 3 = 1 / 3 :=
by
  sorry

end sum_multiple_of_3_probability_l493_493067


namespace sum_consecutive_integers_l493_493225

theorem sum_consecutive_integers (a b : ℤ) (h1 : b = a + 1) (h2 : a < real.sqrt 13) (h3 : real.sqrt 13 < b) :
  a + b = 7 :=
sorry

end sum_consecutive_integers_l493_493225


namespace externally_tangent_circles_m_l493_493991

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2_eqn (x y m : ℝ) : Prop := x^2 + y^2 - 2 * m * x + m^2 - 1 = 0

theorem externally_tangent_circles_m (m : ℝ) :
  (∀ x y : ℝ, circle1_eqn x y) →
  (∀ x y : ℝ, circle2_eqn x y m) →
  m = 3 ∨ m = -3 :=
by sorry

end externally_tangent_circles_m_l493_493991


namespace prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l493_493000

def person_A_hits : ℚ := 1 / 2
def person_B_hits : ℚ := 1 / 3

def person_A_misses : ℚ := 1 - person_A_hits
def person_B_misses : ℚ := 1 - person_B_hits

def exactly_one_hits : ℚ := (person_A_hits * person_B_misses) + (person_B_hits * person_A_misses)
def at_least_one_hits : ℚ := 1 - (person_A_misses * person_B_misses)

theorem prob_exactly_one_hits_is_one_half : exactly_one_hits = 1 / 2 := sorry

theorem prob_at_least_one_hits_is_two_thirds : at_least_one_hits = 2 / 3 := sorry

end prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l493_493000


namespace smallest_k_l493_493918

theorem smallest_k :
  ∃ k : ℕ, 0 < k ∧ (∀ z : \complex, (z^{12} + z^{11} + z^7 + z^6 + z^5 + z + 1) ∣ (z^k - 1)) ∧ k = 91 :=
  sorry

end smallest_k_l493_493918


namespace correct_forecast_interpretation_l493_493045

/-- The probability of precipitation in the area tomorrow is 80%. -/
def prob_precipitation_tomorrow : ℝ := 0.8

/-- Multiple choice options regarding the interpretation of the probability of precipitation. -/
inductive forecast_interpretation
| A : forecast_interpretation
| B : forecast_interpretation
| C : forecast_interpretation
| D : forecast_interpretation

/-- The correct interpretation is Option C: "There is an 80% chance of rain in the area tomorrow." -/
def correct_interpretation : forecast_interpretation :=
forecast_interpretation.C

theorem correct_forecast_interpretation :
  (prob_precipitation_tomorrow = 0.8) → (correct_interpretation = forecast_interpretation.C) :=
by
  sorry

end correct_forecast_interpretation_l493_493045


namespace point_on_terminal_side_of_240_l493_493663

theorem point_on_terminal_side_of_240 (a : ℝ) (h : tan (240 * real.pi / 180) = a / -4) : a = -4 * sqrt 3 :=
sorry

end point_on_terminal_side_of_240_l493_493663


namespace maximum_sequence_length_positive_integer_x_l493_493904

/-- Define the sequence terms based on the problem statement -/
def sequence_term (n : ℕ) (a₁ a₂ : ℤ) : ℤ :=
  if n = 1 then a₁
  else if n = 2 then a₂
  else sequence_term (n - 2) a₁ a₂ - sequence_term (n - 1) a₁ a₂

/-- Define the main problem with the conditions -/
theorem maximum_sequence_length_positive_integer_x :
  ∃ x : ℕ, 0 < x ∧ (309 = x) ∧ 
  (∀ n, sequence_term n 500 x ≥ 0) ∧
  (sequence_term 11 500 x < 0) :=
by
  sorry

end maximum_sequence_length_positive_integer_x_l493_493904


namespace fraction_sum_59_l493_493397

theorem fraction_sum_59 :
  ∃ (a b : ℕ), (0.84375 = (a : ℚ) / b) ∧ (Nat.gcd a b = 1) ∧ (a + b = 59) :=
sorry

end fraction_sum_59_l493_493397


namespace reroll_two_dice_maximize_winning_chances_l493_493302

/-- 
  In a modified game, Jason rolls three fair standard six-sided dice. 
  He can decide to re-roll any subset of them based on optimizing his winning chances. 
  He wins if the sum of the numbers face up on the three dice after re-rolling (if any) is exactly 9.
  We want to prove that the probability that he chooses to re-roll exactly two of the dice to maximize his winning chances is 7/36.
-/
theorem reroll_two_dice_maximize_winning_chances :
  let outcomes : List (ℕ × ℕ × ℕ) := List.product (List.product (List.range 1 7) (List.range 1 7)) (List.range 1 7),
      winning_outcomes := outcomes.filter (fun (d1, d2, d3) => d1 + d2 + d3 = 9)
  in 
    (winning_outcomes.filter (λ (d1, d2, d3) => optimal_re_roll_strategy (d1, d2, d3) = 2)).length / outcomes.length = 7 / 36 := 
sorry

/-- 
  Define the optimal re-roll strategy which returns the number of dice Jason should re-roll 
  given the current state of the dice 
-/
def optimal_re_roll_strategy (state : ℕ × ℕ × ℕ) : ℕ := 
sorry

end reroll_two_dice_maximize_winning_chances_l493_493302


namespace keaton_apple_earnings_l493_493694

theorem keaton_apple_earnings
  (orange_harvest_interval : ℕ)
  (orange_income_per_harvest : ℕ)
  (total_yearly_income : ℕ)
  (orange_harvests_per_year : ℕ)
  (orange_yearly_income : ℕ)
  (apple_yearly_income : ℕ) :
  orange_harvest_interval = 2 →
  orange_income_per_harvest = 50 →
  total_yearly_income = 420 →
  orange_harvests_per_year = 12 / orange_harvest_interval →
  orange_yearly_income = orange_harvests_per_year * orange_income_per_harvest →
  apple_yearly_income = total_yearly_income - orange_yearly_income →
  apple_yearly_income = 120 :=
by
  sorry

end keaton_apple_earnings_l493_493694


namespace sum_of_numerator_and_denominator_is_nine_l493_493669

theorem sum_of_numerator_and_denominator_is_nine :
  ∀ (A B C D E : Type) [metric_space A] [metric_space B]
  (distance : A → B → C)
  (right_triangle_ABC : Prop) (right_triangle_ABD : Prop)
  (right_angle_C : right_triangle_ABC)
  (right_angle_A : right_triangle_ABD)
  (C_D_same_side : Prop)
  (AC_eq_4 : distance A C = 4)
  (BC_eq_3 : distance B C = 3)
  (AD_eq_15 : distance A D = 15)
  (DE_parallel_AC : D → E → Prop), 
  (DE_parallel_AC D E) → 
  let AB := distance A B in
  let BD := distance B D in
  let DE := distance D E in
  AB = 5 → BD = 5 * real.sqrt 10 → DE = 4 * real.sqrt 10 →
  (4 + 5 = 9) :=
by
  sorry

end sum_of_numerator_and_denominator_is_nine_l493_493669


namespace efficiency_relation_l493_493485

-- Definitions and conditions
def eta0 (Q34 Q12 : ℝ) : ℝ := 1 - Q34 / Q12
def eta1 (Q13 Q12 : ℝ) : ℝ := 1 - Q13 / Q12
def eta2 (Q34 Q13 : ℝ) : ℝ := 1 - Q34 / Q13

theorem efficiency_relation 
    (Q12 Q13 Q34 α : ℝ)
    (h_eta0 : η₀ = 1 - Q34 / Q12)
    (h_eta1 : η₁ = 1 - Q13 / Q12)
    (h_eta2 : η₂ = 1 - Q34 / Q13)
    (h_relation : η₂ = (η₀ - η₁) / (1 - η₁))
    (h_eta10 : η₁ < η₀)
    (h_eta20 : η₂ < η₀)
    (h_eta0_lt : η₀ < 1)
    (h_eta1_lt : η₁ < 1)
    (h_eta1_def : η₁ = (1 - 0.01 * α) * η₀) :
    η₂ = α / (100 - (100 - α) * η₀) :=
begin
  sorry
end

end efficiency_relation_l493_493485


namespace book_cost_l493_493888

theorem book_cost (b : ℝ) : (11 * b < 15) ∧ (12 * b > 16.20) → b = 1.36 :=
by
  intros h
  sorry

end book_cost_l493_493888


namespace waiter_initial_tables_l493_493523

theorem waiter_initial_tables
  (T : ℝ)
  (H1 : (T - 12.0) * 8.0 = 256) :
  T = 44.0 :=
sorry

end waiter_initial_tables_l493_493523


namespace general_formula_a_n_smallest_m_l493_493626

-- Given conditions
axiom S_n_def {n : ℕ} (hn : n > 0) : Σ (n : ℕ), n > 0 → ℕ := λ n (hn : n > 0), 3 * n ^ 2 - 2 * n

-- Part (1): General formula for the sequence {a_n}
theorem general_formula_a_n (n : ℕ) (hn : n > 0) : ∀ n, (a_n : ℕ), a_n = (λ n, if n = 1 then 1 else 6 * n - 5) := sorry

-- Part (2): Smallest integer m such that T_n < m / 20 for all n > 0
theorem smallest_m (m : ℕ) (hn_m : m > 0) : ∀ n, T_n < m / 20 → m = 10 := sorry

end general_formula_a_n_smallest_m_l493_493626


namespace number_of_designs_with_shaded_fraction_eq_three_fifths_l493_493567

theorem number_of_designs_with_shaded_fraction_eq_three_fifths :
  let designs := [ (3 / 8 : ℚ), (12 / 20 : ℚ), (2 / 3 : ℚ), (15 / 25 : ℚ), (4 / 8 : ℚ) ]
  in (designs.count (λ x, x = 3 / 5 : ℚ)) = 2 := by
  sorry

end number_of_designs_with_shaded_fraction_eq_three_fifths_l493_493567


namespace no_regular_n_gon_with_integer_diagonals_l493_493957

theorem no_regular_n_gon_with_integer_diagonals (n : ℕ) (h : n ≥ 4) :
  ¬ ∃ (vertices : ℕ → ℝ × ℝ),
    (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → 
      2 * sin (π * (j - i) / n)).nat_abs :=
sorry

end no_regular_n_gon_with_integer_diagonals_l493_493957


namespace max_value_g_l493_493881

def g (x : ℝ) : ℝ := 4 * x - x^3

theorem max_value_g : 
  ∃ x ∈ Icc (0:ℝ) 2, g x = (16 * Real.sqrt 3) / 9 :=
sorry

end max_value_g_l493_493881


namespace population_difference_l493_493101

variable (A B C : ℝ)

-- Conditions
def population_condition (A B C : ℝ) : Prop := A + B = B + C + 5000

-- The proof statement
theorem population_difference (h : population_condition A B C) : A - C = 5000 :=
by sorry

end population_difference_l493_493101


namespace ellipse_foci_y_axis_range_l493_493244

theorem ellipse_foci_y_axis_range (k : ℝ) : 
  (2*k - 1 > 2 - k) → (2 - k > 0) → (1 < k ∧ k < 2) := 
by 
  intros h1 h2
  -- We use the assumptions to derive the target statement.
  sorry

end ellipse_foci_y_axis_range_l493_493244


namespace cost_of_each_pair_of_socks_eq_2_l493_493127

-- Definitions and conditions
def cost_of_shoes : ℤ := 74
def cost_of_bag : ℤ := 42
def paid_amount : ℤ := 118
def discount_rate : ℚ := 0.10

-- Given the conditions
def total_cost (x : ℚ) : ℚ := cost_of_shoes + 2 * x + cost_of_bag
def discount (x : ℚ) : ℚ := if total_cost x > 100 then discount_rate * (total_cost x - 100) else 0
def total_cost_after_discount (x : ℚ) : ℚ := total_cost x - discount x

-- Theorem to prove
theorem cost_of_each_pair_of_socks_eq_2 : 
  ∃ x : ℚ, total_cost_after_discount x = paid_amount ∧ 2 * x = 4 :=
by
  sorry

end cost_of_each_pair_of_socks_eq_2_l493_493127


namespace number_of_sheep_l493_493100

def ratio := 3 / 7
def food_per_horse := 230
def total_food := 12880
def num_horses := total_food / food_per_horse
def num_sheep : ℕ := sorry

theorem number_of_sheep (S H : ℕ) (ratio_cond : (S : ℚ) / H = ratio)
                       (food_cond : H * food_per_horse = total_food) :
                       S = 24 :=
by
  have H_eq := num_horses
  rw [←H_eq] at food_cond
  sorry

end number_of_sheep_l493_493100


namespace average_and_variance_l493_493953

-- Definitions based on the given conditions
variables {x : ℕ → ℝ} {n : ℕ}

-- Condition 1: Data transformation
def transformed_data (x : ℕ → ℝ) (n : ℕ) := 
  ∀ i, i < n → 3 * x i + 7

-- Condition 2: Average of transformed data
def average_transformed_data (x : ℕ → ℝ) (n : ℕ) := 
  (1 / n) * (∑ i in finset.range n, 3 * x i + 7) = 22

-- Condition 3: Variance of transformed data
def variance_transformed_data (x : ℕ → ℝ) (n : ℕ) := 
  (1 / n) * (∑ i in finset.range n, (3 * x i + 7 - 22)^2) = 36

-- Proof problem: determining and proving average and variance
theorem average_and_variance (x : ℕ → ℝ) (n : ℕ) 
  (ht: transformed_data x n)
  (avg_trans: average_transformed_data x n)
  (var_trans: variance_transformed_data x n) : 
  (∑ i in finset.range n, x i) / n = 5 ∧ 
  (1 / n) * (∑ i in finset.range n, (x i - 5)^2) = 4 :=
sorry

end average_and_variance_l493_493953


namespace equilateral_triangles_perpendicular_and_ratio_l493_493571

theorem equilateral_triangles_perpendicular_and_ratio
  (A B C K M N P Q: Point)
  (hABC : equilateral_triangle A B C)
  (hKMN : equilateral_triangle K M N)
  (hAKNB : vec_eq (vector A K) (vector N B))
  (hPlane : co_planar [A, B, C, K, M, N]) :
  (∃ P, is_midpoint P A B ∧ is_midpoint P N K ∧ angle_right (line C P) (line A B)
    ∧ P = midpoint A B)
  → (∃ Q, Q = rotate (line A P) 90 A ∧ similar_triangle Q C P ∧ 
    dilation Q C (√3) A
    ∧ perpendicular (line C M) (line A N)
    ∧ length_ratio (segment C M) (segment A N) = √3) := sorry

end equilateral_triangles_perpendicular_and_ratio_l493_493571


namespace extreme_point_inequality_l493_493260

theorem extreme_point_inequality (a : ℝ) (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) (h₃ : x₂ < 1) (h₄ : x₁ + x₂ = 1) (h₅ : x₁ * x₂ = a / 2) (h₆ : a < 1/2) :
  let f := λ x : ℝ, x^2 - 2 * x + 1 + a * log x in
  f(x₂) > (1 - 2 * log 2) / 4 :=
by
  sorry

end extreme_point_inequality_l493_493260


namespace max_length_sequence_l493_493892

noncomputable def max_length_x : ℕ := 309

theorem max_length_sequence :
  let a : ℕ → ℤ := λ n, match n with
    | 0 => 500
    | 1 => max_length_x
    | n + 2 => a n - a (n + 1)
  in
  ∀ n : ℕ, (∀ m < n, a m ≥ 0) →
    (a n < 0) →
    (309 = max_length_x) :=
by
  intro a
  intro n
  intro h_pos
  intro h_neg
  sorry

end max_length_sequence_l493_493892


namespace altitude_of_isosceles_triangle_on_square_side_l493_493518

theorem altitude_of_isosceles_triangle_on_square_side
  (s : ℝ) : 
  let side_length_square := s
  let side_length_triangle := sqrt 2 * s
  let area_square := s^2
  let h := 2 * s -- Hypothesized altitude to be proven
  let area_triangle := (1/2) * s * h
in area_square = area_triangle → h = 2 * s := 
sorry

end altitude_of_isosceles_triangle_on_square_side_l493_493518


namespace denominator_of_simplest_form_l493_493786

theorem denominator_of_simplest_form : 
  let frac := (625 : ℚ) / 1000000 in
  frac = 1 / 1600 :=
by
  sorry

end denominator_of_simplest_form_l493_493786


namespace value_of_number_l493_493109

theorem value_of_number (x : ℤ) (number : ℚ) (h₁ : x = 32) (h₂ : 35 - (23 - (15 - x)) = 12 * number / (1/2)) : number = -5/6 :=
by
  sorry

end value_of_number_l493_493109


namespace exists_set_B_l493_493698

open Finset

theorem exists_set_B (A : Finset ℕ) (hA : ∀ x ∈ A, 0 < x) : 
  ∃ B : Finset ℕ, A ⊆ B ∧ (∏ x in B, x = ∑ x in B, x^2) :=
sorry

end exists_set_B_l493_493698


namespace problem_1_parity_monotonicity_l493_493252

def f (x : ℝ) : ℝ := x^3 + x

theorem problem_1_parity_monotonicity :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, f' x > 0) := sorry

lemma problem_2_sign_of_f_sum
  (a b c : ℝ) (h1 : a + b > 0) (h2 : b + c > 0) (h3 : c + a > 0) :
  f a + f b + f c > 0 := sorry

end problem_1_parity_monotonicity_l493_493252


namespace solve_k_and_min_cost_l493_493071

noncomputable def k_value (k : ℝ) : Prop :=
  1270 = (16000000 + ((k + 800) + (2 * k + 800) + (3 * k + 800) + (4 * k + 800) + (5 * k + 800)) * 10000) / (50000)

noncomputable def minimum_cost_n (n : ℕ) (min_cost : ℝ) : Prop :=
  min_cost = 1225 ∧ n = 8 ∧
  ∀ n' > 0, 
    let cost := 1600 / ↑n' + 25 * ↑n' + 825 
    in cost ≥ 1225

theorem solve_k_and_min_cost : 
  ∃ k : ℝ, k_value k ∧ ∃ n : ℕ, ∃ min_cost : ℝ, minimum_cost_n n min_cost :=
begin
  sorry
end

end solve_k_and_min_cost_l493_493071


namespace integer_roots_condition_l493_493491

theorem integer_roots_condition (n : ℕ) (hn : n > 0) :
  (∃ x : ℤ, x^2 - 4 * x + n = 0) ↔ (n = 3 ∨ n = 4) := 
by
  sorry

end integer_roots_condition_l493_493491


namespace fraction_six_power_l493_493712

theorem fraction_six_power (n : ℕ) (hyp : n = 6 ^ 2024) : n / 6 = 6 ^ 2023 :=
by sorry

end fraction_six_power_l493_493712


namespace Emily_used_10_dimes_l493_493570

theorem Emily_used_10_dimes
  (p n d : ℕ)
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 10 := by
  sorry

end Emily_used_10_dimes_l493_493570


namespace lassis_from_mangoes_l493_493872

theorem lassis_from_mangoes (L M : ℕ) (h : 2 * L = 11 * M) : 12 * L = 66 :=
by sorry

end lassis_from_mangoes_l493_493872


namespace more_than_60_people_l493_493495

theorem more_than_60_people (N : ℕ) (C : ℕ -> set (fin 10)): 
    (∀ i j, i < j -> (C i ∩ C j = ∅)) -> 
    (∀ k, fin 10 → N) -> 
    40 * (10 * (10 - 1) / 2) ≤ N * (N - 1) / 2 -> 
    N > 60 := 
by
  sorry

end more_than_60_people_l493_493495


namespace coloring_problem_solution_l493_493425

noncomputable def num_possible_colorings : ℕ :=
  let squares : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let colors : Finset ℕ := {1, 2, 3} -- 1: red, 2: yellow, 3: blue
  let adjacent_pairs : Finset (ℕ × ℕ) := {(1, 2), (1, 4), (2, 3), (2, 5), (3, 6), (4, 5), (4, 7), (5, 6), (5, 8), (6, 9), (7, 8), (8, 9)}
  let must_be_same := {3, 5, 7}
  -- Counting all valid colorings
  108

theorem coloring_problem_solution :
  num_possible_colorings = 108 :=
by
  sorry

end coloring_problem_solution_l493_493425


namespace total_votes_proof_l493_493671

-- Define the terms and conditions
def total_votes (V : ℝ) : Prop :=
  let valid_votes := 0.8 * V in
  let candidate1_votes := 0.55 * valid_votes in
  candidate1_votes + 2700 = valid_votes

-- The theorem we need to prove
theorem total_votes_proof : ∃ V : ℝ, total_votes V ∧ V = 7500 :=
by
  sorry -- Proof to be filled in later

end total_votes_proof_l493_493671


namespace initial_sum_of_money_l493_493521

theorem initial_sum_of_money (A2 A7 : ℝ) (H1 : A2 = 520) (H2 : A7 = 820) :
  ∃ P : ℝ, P = 400 :=
by
  -- Proof starts here
  sorry

end initial_sum_of_money_l493_493521


namespace youngest_boy_age_l493_493748

-- Definitions arising directly from conditions
variables {A B C : ℝ}
variable (x : ℝ)

-- Conditions
def ages_in_proportion : Prop := A = 2 * x ∧ B = 6 * x ∧ C = 8 * x
def average_age_is_120 : Prop := (A + B + C) / 3 = 120

-- Theorem to prove
theorem youngest_boy_age (x_val : ℝ) (hst1 : ages_in_proportion x_val) (hst2 : average_age_is_120 x_val) : A = 45 :=
by {
  -- Skipping proof by "sorry"
  sorry
}

end youngest_boy_age_l493_493748


namespace greatest_sum_of_base_eight_digits_l493_493788

theorem greatest_sum_of_base_eight_digits 
  (n : ℕ) 
  (h1 : 0 < n) 
  (h2 : n < 1728) 
  : ∃ d : ℕ, (d = 6 ∧ 
    ∑ (i : ℕ) in (coe : list ℕ → finset ℕ) (n.digits 8), i = d) :=
sorry

end greatest_sum_of_base_eight_digits_l493_493788


namespace cards_return_to_initial_order_l493_493215

def perform_operation (cards : List ℕ) : List ℕ :=
  let evens := cards.zipWithIndex.filterMap (λ ⟨c, i⟩ => if (i+1) % 2 = 0 then some c else none)
  let odds := cards.zipWithIndex.filterMap (λ ⟨c, i⟩ => if (i+1) % 2 = 1 then some c else none)
  odds ++ evens

theorem cards_return_to_initial_order (n : ℕ) (h : 0 < n) :
  ∃ k ≤ 2 * n - 2, (Nat.iterate perform_operation k (List.range (2 * n).map (λ i => i + 1))) = List.range (2 * n).map (λ i => i + 1) :=
sorry

end cards_return_to_initial_order_l493_493215


namespace hours_per_week_summer_l493_493858

noncomputable def winter_hours_per_week := 45
noncomputable def winter_weeks := 8
noncomputable def winter_earnings := 3600
noncomputable def summer_earnings := 4500
noncomputable def summer_weeks := 20

theorem hours_per_week_summer :
  let hourly_rate := winter_earnings / (winter_hours_per_week * winter_weeks)
  let total_hours_summer := summer_earnings / hourly_rate
  let weekly_hours_summer := total_hours_summer / summer_weeks
  weekly_hours_summer = 22.5 :=
by
  sorry

end hours_per_week_summer_l493_493858


namespace compare_numbers_l493_493874

theorem compare_numbers :
  3 * 10^5 < 2 * 10^6 ∧ -2 - 1 / 3 > -3 - 1 / 2 := by
  sorry

end compare_numbers_l493_493874


namespace closest_years_l493_493772

theorem closest_years (a b c d : ℕ) (h1 : 10 * a + b + 10 * c + d = 10 * b + c) :
  (a = 1 ∧ b = 8 ∧ c = 6 ∧ d = 8) ∨ (a = 2 ∧ b = 3 ∧ c = 0 ∧ d =7) ↔
  ((10 * 1 + 8 + 10 * 6 + 8 = 10 * 8 + 6) ∧ (10 * 2 + 3 + 10 * 0 + 7 = 10 * 3 + 0)) :=
sorry

end closest_years_l493_493772


namespace infinite_solutions_of_linear_system_l493_493549

theorem infinite_solutions_of_linear_system :
  ∃ (f : ℝ → ℝ), (∀ x y : ℝ, y = f x → 3 * x - 4 * y = 10) ∧
                 (∀ x y : ℝ, y = f x → 9 * x - 12 * y = 30) :=
by
  -- Provide the function f and the proof of the conditions
  let f : ℝ → ℝ := λ x, (3 * x - 10) / 4
  use f
  split
  -- Proof that the function satisfies the first equation
  { intros x y hy,
    rw [hy],
    -- Algebraic simplification goes here
    sorry },
  -- Proof that the function satisfies the second equation
  { intros x y hy,
    rw [hy],
    -- Algebraic simplification goes here
    sorry }

end infinite_solutions_of_linear_system_l493_493549


namespace max_length_sequence_309_l493_493900

def sequence (a₁ a₂ : ℤ) : ℕ → ℤ
| 0     := a₁
| 1     := a₂
| (n+2) := sequence n - sequence (n+1)

theorem max_length_sequence_309 :
  ∃ x : ℤ, x = 309 ∧
  (let a₁ := 500 in 
  let a₂ := x in
  sequence a₁ a₂ 9 > 0 ∧
  sequence a₁ a₂ 10 > 0) :=
sorry

end max_length_sequence_309_l493_493900


namespace cost_of_pencil_l493_493842

theorem cost_of_pencil (s n c : ℕ) (h_majority : s > 15) (h_pencils : n > 1) (h_cost : c > n)
  (h_total_cost : s * c * n = 1771) : c = 11 :=
sorry

end cost_of_pencil_l493_493842


namespace perpendicular_vectors_l493_493645

variables {λ : ℝ}

def vector_m : ℝ × ℝ := (λ + 1, 1)
def vector_n : ℝ × ℝ := (λ + 2, 2)

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem perpendicular_vectors :
  dot_product (vector_add vector_m vector_n) (vector_sub vector_m vector_n) = 0 → λ = -3 :=
by
  sorry

end perpendicular_vectors_l493_493645


namespace crabapple_sequences_l493_493726

theorem crabapple_sequences (students : ℕ) (meetings : ℕ) (num_sequences : ℕ) :
  students = 13 → meetings = 3 → num_sequences = students^meetings → num_sequences = 2197 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end crabapple_sequences_l493_493726


namespace root_product_is_27_l493_493870

open Real

noncomputable def cube_root (x : ℝ) := x ^ (1 / 3 : ℝ)
noncomputable def fourth_root (x : ℝ) := x ^ (1 / 4 : ℝ)
noncomputable def square_root (x : ℝ) := x ^ (1 / 2 : ℝ)

theorem root_product_is_27 : 
  (cube_root 27) * (fourth_root 81) * (square_root 9) = 27 := 
by
  sorry

end root_product_is_27_l493_493870


namespace divide_parallelogram_into_five_equal_parts_l493_493312

theorem divide_parallelogram_into_five_equal_parts (ABCD : parallelogram) :
  ∃ (lines : list line), divides_into_equal_area_parts ABCD lines 5 := 
sorry

end divide_parallelogram_into_five_equal_parts_l493_493312


namespace student_selection_sequences_l493_493155

-- Define the conditions
def students := 10
def days_in_week := 5

-- Theorem statement
theorem student_selection_sequences : 
  (Finset.univ.finset_of_card students).permutations days_in_week = 30240 := by sorry

end student_selection_sequences_l493_493155


namespace inequality_proof_l493_493721

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_cond : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
by
  sorry

end inequality_proof_l493_493721


namespace rectangle_area_l493_493039

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) 
  : l * w = 1600 := 
by 
  sorry

end rectangle_area_l493_493039


namespace range_of_a_l493_493975

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem range_of_a {a : ℝ} (h : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  a > 1/7 ∧ a < 1/3 :=
sorry

end range_of_a_l493_493975


namespace convert_deg_mins_to_decimal_l493_493823

-- Definition of degrees and minutes
def degrees_mins_to_decimal (deg : ℝ) (mins : ℝ) : ℝ :=
  deg + (mins / 60)

-- The theorem to be proven
theorem convert_deg_mins_to_decimal :
  degrees_mins_to_decimal 120 45 = 120.75 :=
by
  sorry

end convert_deg_mins_to_decimal_l493_493823


namespace zac_strawberries_l493_493693

theorem zac_strawberries (J M Z : ℕ) 
  (h1 : J + M + Z = 550) 
  (h2 : J + M = 350) 
  (h3 : M + Z = 250) : 
  Z = 200 :=
sorry

end zac_strawberries_l493_493693


namespace probability_three_primes_out_of_five_l493_493536

def probability_of_prime (p : ℚ) : Prop := ∃ k, k = 4 ∧ p = 4/10

def probability_of_not_prime (p : ℚ) : Prop := ∃ k, k = 6 ∧ p = 6/10

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_three_primes_out_of_five :
  ∀ p_prime p_not_prime : ℚ, 
  probability_of_prime p_prime →
  probability_of_not_prime p_not_prime →
  (combinations 5 3 * (p_prime^3 * p_not_prime^2) = 720/3125) :=
by
  intros p_prime p_not_prime h_prime h_not_prime
  sorry

end probability_three_primes_out_of_five_l493_493536


namespace contractor_work_done_l493_493506

def initial_people : ℕ := 10
def remaining_people : ℕ := 8
def total_days : ℕ := 100
def remaining_days : ℕ := 75
def fraction_done : ℚ := 1/4
def total_work : ℚ := 1

theorem contractor_work_done (x : ℕ) 
  (h1 : initial_people * x = fraction_done * total_work) 
  (h2 : remaining_people * remaining_days = (1 - fraction_done) * total_work) :
  x = 60 :=
by
  sorry

end contractor_work_done_l493_493506


namespace domain_of_f_f_is_odd_l493_493249

noncomputable def f (a x : ℝ) : ℝ := log a (x + 1) - log a (1 - x)

axiom a_pos (a : ℝ) : 0 < a
axiom a_ne_one (a : ℝ) : a ≠ 1

theorem domain_of_f (a : ℝ) (h1 : a_pos a) (h2 : a_ne_one a) : 
  ∀ x, -1 < x ∧ x < 1 ↔ ∃ y, f a y = f a x ∧ -1 < y ∧ y < 1 :=
sorry

theorem f_is_odd (a : ℝ) (h1 : a_pos a) (h2 : a_ne_one a) :
  ∀ x, -1 < x ∧ x < 1 → f a (-x) = -f a (x) :=
sorry

end domain_of_f_f_is_odd_l493_493249


namespace trig_identity_l493_493593

theorem trig_identity :
  (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) + 
   Real.cos (20 * Real.pi / 180) * Real.sin (40 * Real.pi / 180)) = 
  (Real.sqrt 3 / 2) :=
by
  sorry

end trig_identity_l493_493593


namespace range_of_m_for_g_ge_2_l493_493979

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (a * x - a + 2)

theorem range_of_m_for_g_ge_2
  (a : ℝ)
  (h_local_min : ∀ f', (∀ x, f' x = deriv (f a) x) → f' (-2) = 0)
  (h_tangent_line : ∀ x y, 2 * x - y + 1 = 0 ↔ y = f 1 x)
  (g : ℝ → ℝ := λ x, m * f 1 x - x * (x + 4))
  (h_ge_2 : ∀ x ∈ Icc (-2 : ℝ) ∞, g x ≥ 2) :
  (2 : ℝ) ≤ m ∧ m ≤ 2 * Real.exp 2
:= by sorry

end range_of_m_for_g_ge_2_l493_493979


namespace symmetric_line_proof_l493_493583

-- Define the given lines
def line_l (x y : ℝ) : Prop := 3 * x - 4 * y + 5 = 0
def axis_of_symmetry (x y : ℝ) : Prop := x + y = 0

-- Define the final symmetric line to be proved
def symmetric_line (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0

-- State the theorem
theorem symmetric_line_proof (x y : ℝ) : 
  (line_l (-y) (-x)) → 
  axis_of_symmetry x y → 
  symmetric_line x y := 
sorry

end symmetric_line_proof_l493_493583


namespace cylinder_volume_ratio_l493_493114

-- First define the problem: a 6 x 10 rectangle rolled to form two different cylinders
theorem cylinder_volume_ratio : 
  let r1 := 3 / Real.pi in
  let V1 := Real.pi * r1^2 * 10 in
  let r2 := 5 / Real.pi in
  let V2 := Real.pi * r2^2 * 6 in
  V2 / V1 = 5 / 3 :=
by
  -- The proof steps are omitted as the theorem states only.
  sorry

end cylinder_volume_ratio_l493_493114


namespace problem_l493_493613

variable (S a : ℕ → ℤ)

-- Given condition: ∀ (n: ℕ), (S n) = (a n - 1) / 3, where n ∈ (ℕ*) 
def condition (n : ℕ) [fact (0 < n)] : Prop := (S n) = (a n - 1) / 3

theorem problem (n : ℕ) [fact (0 < n)] (h_n1 : condition S a 1) (h_n2 : condition S a 2) :
  a 1 = -1/2 ∧ a 2 = 1/4 ∧ (∃ q, q = -1/2 ∧ ∀ n ≥ 2, a n = q * a (n - 1)) :=
sorry

end problem_l493_493613


namespace greater_than_reciprocal_l493_493807

theorem greater_than_reciprocal (x : ℚ) : 
  x ∈ {-3/2, -1, 1/3, 2, 3} → (x = 2 ∨ x = 3) ↔ x > 1 / x :=
by 
  intros hx
  finish 

end greater_than_reciprocal_l493_493807


namespace f_k_plus_1_eq_f_k_plus_terms_l493_493605

def f (n : ℕ) : ℝ := ∑ i in Finset.Icc (n+1) (3*n+1), (1 / (i : ℝ))

theorem f_k_plus_1_eq_f_k_plus_terms (k : ℕ) :
  f (k + 1) = f k + (1 / (3 * k + 2 : ℝ)) + (1 / (3 * k + 3 : ℝ)) + (1 / (3 * k + 4 : ℝ)) - (1 / (k + 1 : ℝ)) :=
by
  sorry

end f_k_plus_1_eq_f_k_plus_terms_l493_493605


namespace sum_of_magnitudes_l493_493474

noncomputable def complex_absolute_value (z : ℂ) : ℝ :=
complex.abs z

theorem sum_of_magnitudes (x y : ℝ) :
  let z1 := x + complex.sqrt 11 + y * complex.I,
      z6 := x - complex.sqrt 11 + y * complex.I in
  complex_absolute_value z1 + complex_absolute_value z6 = 30 * (real.sqrt 2 + 1) :=
begin
  sorry
end

end sum_of_magnitudes_l493_493474


namespace radius_range_l493_493240

-- Conditions:
-- r1 is the radius of circle O1
-- r2 is the radius of circle O2
-- d is the distance between centers of circles O1 and O2
-- PO1 is the distance from a point P on circle O2 to the center of circle O1

variables (r1 r2 d PO1 : ℝ)

-- Given r1 = 1, d = 5, PO1 = 2
axiom r1_def : r1 = 1
axiom d_def : d = 5
axiom PO1_def : PO1 = 2

-- To prove: 3 ≤ r2 ≤ 7
theorem radius_range (r2 : ℝ) (h : d = 5 ∧ r1 = 1 ∧ PO1 = 2 ∧ (∃ P : ℝ, P = r2)) : 3 ≤ r2 ∧ r2 ≤ 7 :=
by {
  sorry
}

end radius_range_l493_493240


namespace max_tan_A_l493_493608

theorem max_tan_A 
  (A B : ℝ)
  (h1 : 0 < A)
  (h2 : A < π / 2)
  (h3 : 0 < B)
  (h4 : B < π / 2)
  (h5 : sin A / sin B = sin (A + B))
  : tan A ≤ 4 / 3 := 
sorry

end max_tan_A_l493_493608


namespace minimum_intersection_l493_493332

theorem minimum_intersection (A B C : Set) (hA : |A| = 150) (hB : |B| = 150) (hn : 2^|A| + 2^|B| + 2^|C| = 2^|A ∪ B ∪ C|) (hAB : |A ∩ B| ≥ 145) :
  ∃ ℕ, minimum (|A ∩ B ∩ C|) = -305 :=
by sorry

end minimum_intersection_l493_493332


namespace assembly_line_arrangements_l493_493151

-- Declaring task types
inductive Task
| E  -- Install the engine
| A  -- Add axles
| W  -- Add wheels to the axles
| I  -- Install the windshield
| P  -- Install the instrument panel
| S  -- Install the steering wheel
| D  -- Mount the doors
| IS -- Install the interior seating

-- Conditions as predicates
def valid_sequence (tasks : List Task) : Prop :=
  (tasks.indexOf Task.A < tasks.indexOf Task.W) ∧
  (tasks.indexOf Task.E < tasks.indexOf Task.P) ∧
  (tasks.indexOf Task.E < tasks.indexOf Task.S) ∧
  (tasks.indexOf Task.IS < tasks.indexOf Task.D)

-- The number of valid ways to arrange the tasks
def arrangement_count : Nat :=
  1920

-- Proof statement
theorem assembly_line_arrangements :
  (∃ (tasks : List Task), valid_sequence tasks) ∧ 
  (List.permutations (List.factorial 8)).count (λ tasks, valid_sequence tasks) = arrangement_count :=
  sorry

end assembly_line_arrangements_l493_493151


namespace minimum_modulus_l493_493952

open Classical Real

noncomputable def arithmetic_vector_sequence (a b : ℕ) (a₁ a₃ : ℤ × ℤ) : ℕ → ℤ × ℤ :=
  λ n, (a₁.1 + (n - 1) * a, a₁.2 + (n - 1) * b)

theorem minimum_modulus (a₁ a₃ : ℤ × ℤ) (n₄ n₅ : ℕ) (h₁ : a₁ = (-20, 13))
  (h₃ : a₃ = (-18, 15))
  (h₄ : ∃ a b, ∀ n, arithmetic_vector_sequence a b a₁ a₃ n = (n₄, n₅)) :
  n₄ = 4 ∨ n₄ = 5 :=
by
  sorry

end minimum_modulus_l493_493952


namespace sqrt_14400_eq_120_l493_493741

theorem sqrt_14400_eq_120 : Real.sqrt 14400 = 120 :=
by
  sorry

end sqrt_14400_eq_120_l493_493741


namespace transformed_sin_2x_eq_sin_4x_l493_493415

theorem transformed_sin_2x_eq_sin_4x (x : ℝ) :
  let f := λ x, sin (2 * x),
      g := λ x, sin (2 * (x + π / 2)),
      h := λ x, sin (4 * x)
  in h x = g (x / 2) := by sorry

end transformed_sin_2x_eq_sin_4x_l493_493415


namespace cost_per_square_inch_l493_493725

def length : ℕ := 9
def width : ℕ := 12
def total_cost : ℕ := 432

theorem cost_per_square_inch :
  total_cost / ((length * width) / 2) = 8 := 
by 
  sorry

end cost_per_square_inch_l493_493725


namespace sum_valid_primes_l493_493363

open Nat

theorem sum_valid_primes:
  ∃ (a b c : ℕ)
   (p : ℕ)
   [Fact (Prime p)],
  (a + b + c + 1) % p = 0 ∧
  (a^2 + b^2 + c^2 + 1) % p = 0 ∧
  (a^3 + b^3 + c^3 + 1) % p = 0 ∧
  (a^4 + b^4 + c^4 + 7459) % p = 0 ∧
  p < 1000 →
  p = 2 ∨ p = 3 ∨ p = 13 ∨ p = 41 →
  (2 + 3 + 13 + 41 = 59) := by
  sorry

end sum_valid_primes_l493_493363


namespace solve_phi_l493_493650

noncomputable def find_phi (phi : ℝ) : Prop :=
  2 * Real.cos phi - Real.sin phi = Real.sqrt 3 * Real.sin (20 / 180 * Real.pi)

theorem solve_phi (phi : ℝ) :
  find_phi phi ↔ (phi = 140 / 180 * Real.pi ∨ phi = 40 / 180 * Real.pi) :=
sorry

end solve_phi_l493_493650


namespace measure_brick_diagonal_l493_493064

-- Definitions of conditions
variable (brick : Type) [HasDiagonal brick]
variable (ruler : Type) [MeasureDistance ruler]
variable (A B : brick)

-- Proof statement: Given three bricks and a ruler, if bricks are arranged correctly,
-- the distance measured equals to the diagonal of one brick.
theorem measure_brick_diagonal (bricks : Fin 3 → brick) (r : ruler) :
  has_correct_arrangement bricks A B →
  MeasureDistance.distance r A B = HasDiagonal.diagonal A :=
sorry

end measure_brick_diagonal_l493_493064


namespace boat_speed_in_still_water_l493_493475

theorem boat_speed_in_still_water (V_b : ℝ) (D : ℝ) (V_s : ℝ) 
  (h1 : V_s = 3) 
  (h2 : D = (V_b + V_s) * 1) 
  (h3 : D = (V_b - V_s) * 1.5) : 
  V_b = 15 := 
by 
  sorry

end boat_speed_in_still_water_l493_493475


namespace flour_for_each_cupcake_l493_493935

noncomputable def flour_per_cupcake (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ) : ℝ :=
  remaining_flour / num_cupcakes

theorem flour_for_each_cupcake :
  ∀ (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ),
    total_flour = 6 →
    remaining_flour = 2 →
    cake_flour_per_cake = 0.5 →
    cake_price = 2.5 →
    cupcake_price = 1 →
    total_revenue = 30 →
    num_cakes = 4 / 0.5 →
    num_cupcakes = 10 →
    flour_per_cupcake total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes = 0.2 :=
by intros; sorry

end flour_for_each_cupcake_l493_493935


namespace find_point_A_equidistant_l493_493913

theorem find_point_A_equidistant :
  ∃ (x : ℝ), (∃ A : ℝ × ℝ × ℝ, A = (x, 0, 0)) ∧
              (∃ B : ℝ × ℝ × ℝ, B = (4, 0, 5)) ∧
              (∃ C : ℝ × ℝ × ℝ, C = (5, 4, 2)) ∧
              (dist (x, 0, 0) (4, 0, 5) = dist (x, 0, 0) (5, 4, 2)) ∧ 
              (x = 2) :=
by
  sorry

end find_point_A_equidistant_l493_493913


namespace monotonically_increasing_interval_l493_493556

noncomputable def f (x : ℝ) : ℝ := x^2 - 8 * Real.log x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 0 → (∀ y : ℝ, y ∈ Set.Ici 2 → f' y ≥ 0) →
  (∀ x y : ℝ, x ∈ Set.Ici 2 → y ∈ Set.Ici 2 → x ≤ y → f x ≤ f y) :=
by
  sorry

end monotonically_increasing_interval_l493_493556


namespace tangent_line_at_1_range_of_c_for_x_gt_1_l493_493633

noncomputable def f (x : ℝ) : ℝ := 3 * x * Real.log x + 2

theorem tangent_line_at_1 :
  let f' := (fun x => 3 * Real.log x + 3)
  let x0 := 1 in
  let y0 := f x0 in
  let slope := f' x0 in
  ∃ a b c, slope = a ∧ -1 = b ∧ (a * x0 + b * y0 + c = 0) := 
by 
  sorry

theorem range_of_c_for_x_gt_1 : 
  ∀ x > 1, ∃ c, 3 * x * Real.log x + 2 ≤ x^2 - c * x → c ≤ 1 - 3 * Real.log 2 := 
by 
  sorry

end tangent_line_at_1_range_of_c_for_x_gt_1_l493_493633


namespace factorial_divisibility_l493_493345

theorem factorial_divisibility (n : ℕ) (hn : 0 < n) : 
  ∃ k : ℕ, 2 * (fact (3 * n)) = k * (fact n * fact (n + 1) * fact (n + 2)) :=
by
  sorry

end factorial_divisibility_l493_493345


namespace area_of_ABC_l493_493025

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  sqrt (1 / 4 * (c^2 * a^2 - ( (c^2 + a^2 - b^2) / 2 )^2))

theorem area_of_ABC :
  let a := 4
  let b := 6
  let c := 2 * sqrt 7
  area_of_triangle a b c = 6 * sqrt 3 :=
by
  sorry

end area_of_ABC_l493_493025


namespace percentage_of_carnations_is_correct_l493_493509

noncomputable def total_flowers (F : ℕ) : ℕ := F
noncomputable def pink_flowers (F : ℕ) : ℕ := (7 * F) / 10
noncomputable def red_flowers (F : ℕ) : ℕ := (3 * F) / 10
noncomputable def pink_roses (F : ℕ) : ℕ := (7 * F) / 40
noncomputable def pink_carnations (F : ℕ) : ℕ := pink_flowers F - pink_roses F
noncomputable def red_carnations (F : ℕ) : ℕ := (3 * F) / 20
noncomputable def total_carnations (F : ℕ) : ℕ := pink_carnations F + red_carnations F
noncomputable def percentage_carnations (F : ℕ) : ℚ := (total_carnations F) / F * 100

theorem percentage_of_carnations_is_correct (F : ℕ) (hF : F ≠ 0) :
  percentage_carninations F = 67.5 := by
  sorry

end percentage_of_carnations_is_correct_l493_493509


namespace variance_proof_l493_493488

variables {X : Type} [random_variable X]
variables {E : X → ℝ} {D : X → ℝ}

-- Assume conditions as definitions in Lean
axiom D_linear : ∀ (a : ℝ) (X : X), D (a • X) = a^2 * D X
axiom expectation_squared_constant : ∀ (X : X), constant (E X)^2
axiom variance_constant : ∀ (X : X), constant (D X)

theorem variance_proof (a : ℝ) (X : X) : 
  D (a • X + E (X^2) - D X) = a^2 * D X :=
by
  sorry

end variance_proof_l493_493488


namespace more_than_60_people_l493_493494

theorem more_than_60_people (N : ℕ) (C : ℕ -> set (fin 10)): 
    (∀ i j, i < j -> (C i ∩ C j = ∅)) -> 
    (∀ k, fin 10 → N) -> 
    40 * (10 * (10 - 1) / 2) ≤ N * (N - 1) / 2 -> 
    N > 60 := 
by
  sorry

end more_than_60_people_l493_493494


namespace correct_factorization_l493_493801

theorem correct_factorization {x y : ℝ} :
  (2 * x ^ 2 - 8 * y ^ 2 = 2 * (x + 2 * y) * (x - 2 * y)) ∧
  ¬(x ^ 2 + 3 * x * y + 9 * y ^ 2 = (x + 3 * y) ^ 2)
    ∧ ¬(2 * x ^ 2 - 4 * x * y + 9 * y ^ 2 = (2 * x - 3 * y) ^ 2)
    ∧ ¬(x * (x - y) + y * (y - x) = (x - y) * (x + y)) := 
by sorry

end correct_factorization_l493_493801


namespace number_of_students_in_third_group_l493_493389

-- Definitions based on given conditions
def students_group1 : ℕ := 9
def students_group2 : ℕ := 10
def tissues_per_box : ℕ := 40
def total_tissues : ℕ := 1200

-- Define the number of students in the third group as a variable
variable {x : ℕ}

-- Prove that the number of students in the third group is 11
theorem number_of_students_in_third_group (h : 360 + 400 + 40 * x = 1200) : x = 11 :=
by sorry

end number_of_students_in_third_group_l493_493389


namespace hexagon_side_length_l493_493027

theorem hexagon_side_length (hexagon : Type) [regular_hexagon hexagon]
  (A B C O : hexagon)
  (h_OA : distance O A = 1)
  (h_OB : distance O B = 1)
  (h_OC : distance O C = 2)
  (consecutive : consecutive_vertices A B C) :
  side_length hexagon = sqrt 3 := 
sorry

end hexagon_side_length_l493_493027


namespace constant_term_binomial_expansion_l493_493677

theorem constant_term_binomial_expansion :
  (∃ k : ℤ, (∀ x : ℚ, x ≠ 0 → ( ∑ r in finset.range 7, (choose 6 r) * (2/x)^(6-r) * (-x)^r) = k) ∧ k = -160) :=
sorry

end constant_term_binomial_expansion_l493_493677


namespace two_students_solve_all_problems_l493_493061

theorem two_students_solve_all_problems
    (students : Fin 15 → Fin 6 → Prop)
    (h : ∀ (p : Fin 6), (∃ (s1 s2 s3 s4 s5 s6 s7 s8 : Fin 15), 
          students s1 p ∧ students s2 p ∧ students s3 p ∧ students s4 p ∧ 
          students s5 p ∧ students s6 p ∧ students s7 p ∧ students s8 p)) :
    ∃ (s1 s2 : Fin 15), ∀ (p : Fin 6), students s1 p ∨ students s2 p := 
by
    sorry

end two_students_solve_all_problems_l493_493061


namespace sum_greater_than_threshold_eq_answer_l493_493078

def numbers := {0.8, 1/2, 0.9}
def threshold := 0.4
def answer := 2.2

theorem sum_greater_than_threshold_eq_answer:
  (∑ x in numbers, if x > threshold then x else 0) = answer := 
sorry

end sum_greater_than_threshold_eq_answer_l493_493078


namespace fraction_is_integer_l493_493347

theorem fraction_is_integer (a b : ℤ) (n : ℕ) (h : n > 0) : 
  ∃ k : ℤ, k * n.factorial = b^(n-1) * (List.prod (List.map (λ k, a + k * b) (List.range n))) :=
sorry

end fraction_is_integer_l493_493347


namespace mn_squared_eq_l493_493404

-- Definition of the problem
variables {A B C D M N : Type} {a b c MN : ℝ}

-- The conditions as hypotheses
def MN_parallel_CD (MN CD : Set A) : Prop := ∀ (x y : A), MN x y = CD x y
def divides_area_in_half (ABCD MN : Set A) : Prop := area MN = area ABCD / 2
def M_on_BC (M BC : Set A) : Prop := M ∈ BC
def N_on_AD (N AD : Set A) : Prop := N ∈ AD
def segment_lengths (A BC AD : Set A) (a b : ℝ) : Prop :=
  length (parallel_segment_through A BC) = a ∧ length (parallel_segment_through B AD) = b
  
-- The proof statement
theorem mn_squared_eq (h1 : MN_parallel_CD MN CD)
                      (h2 : divides_area_in_half ABCD MN)
                      (h3 : M_on_BC M BC)
                      (h4 : N_on_AD N AD)
                      (h5 : segment_lengths A BC AD a b)
                      (h6 : c = length CD):
  MN^2 = (a * b + c^2) / 2 := 
sorry

end mn_squared_eq_l493_493404


namespace period_tan_2x_l493_493794

theorem period_tan_2x : (∃ T, ∀ x, tan (2 * (x + T)) = tan (2 * x)) → T = π / 2 :=
by
  sorry

end period_tan_2x_l493_493794


namespace construct_angle_approx_l493_493740
-- Use a broader import to bring in the entirety of the necessary library

-- Define the problem 
theorem construct_angle_approx (α : ℝ) (m : ℕ) (h : ∃ l : ℕ, (l : ℝ) / 2^m * 90 ≤ α ∧ α ≤ ((l+1) : ℝ) / 2^m * 90) :
  ∃ β : ℝ, β ∈ { β | ∃ l : ℕ, β = (l : ℝ) / 2^m * 90} ∧ |α - β| ≤ 90 / 2^m :=
sorry

end construct_angle_approx_l493_493740


namespace combined_average_score_l493_493727

theorem combined_average_score (M A : ℝ) (m a : ℝ)
  (hM : M = 78) (hA : A = 85) (h_ratio : m = 2 * a / 3) :
  (78 * (2 * a / 3) + 85 * a) / ((2 * a / 3) + a) = 82 := by
  sorry

end combined_average_score_l493_493727


namespace age_problem_l493_493295

theorem age_problem 
  (x y z u : ℕ)
  (h1 : x + 6 = 3 * (y - u))
  (h2 : x = y + z - u)
  (h3: y = x - u) 
  (h4 : x + 19 = 2 * z):
  x = 69 ∧ y = 47 ∧ z = 44 :=
by
  sorry

end age_problem_l493_493295


namespace operation_value_l493_493923

def operation1 (y : ℤ) : ℤ := 8 - y
def operation2 (y : ℤ) : ℤ := y - 8

theorem operation_value : operation2 (operation1 15) = -15 := by
  sorry

end operation_value_l493_493923


namespace units_digit_factorial_150_zero_l493_493440

def units_digit (n : ℕ) : ℕ :=
  (nat.factorial n) % 10

theorem units_digit_factorial_150_zero :
  units_digit 150 = 0 :=
sorry

end units_digit_factorial_150_zero_l493_493440


namespace inequality_and_equality_condition_l493_493720

theorem inequality_and_equality_condition (m n : ℕ) (h_mn : m ≠ n) (h_m_pos : 0 < m) (h_n_pos : 0 < n) :
  (nat.gcd m n) + (nat.gcd (m+1) (n+1)) + (nat.gcd (m+2) (n+2)) ≤ 2 * |m - n| + 1 ∧
  ((∃ k ∈ ℕ, k = |m - n| ∧ ((m, n) = (k, k+1) ∨ (m, n) = (k+1, k) ∨ (m, n) = (2*k, 2*k+2) ∨ (m, n) = (2*k+2, 2*k)))) :=
by
  sorry

end inequality_and_equality_condition_l493_493720


namespace price_of_70_cans_l493_493052

noncomputable def discounted_price (regular_price : ℝ) (discount_percent : ℝ) : ℝ :=
  regular_price * (1 - discount_percent / 100)

noncomputable def total_price (regular_price : ℝ) (discount_percent : ℝ) (total_cans : ℕ) (cans_per_case : ℕ) : ℝ :=
  let price_per_can := discounted_price regular_price discount_percent
  let full_cases := total_cans / cans_per_case
  let remaining_cans := total_cans % cans_per_case
  full_cases * cans_per_case * price_per_can + remaining_cans * price_per_can

theorem price_of_70_cans :
  total_price 0.55 25 70 24 = 28.875 :=
by
  sorry

end price_of_70_cans_l493_493052


namespace quadrilateral_midpoint_property_l493_493008

structure Point :=
(x : ℝ)
(y : ℝ)

structure Quadrilateral :=
(A B C D : Point)

def midpoint (P Q : Point) : Point :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2 }

theorem quadrilateral_midpoint_property (q : Quadrilateral) :
  let E := midpoint q.A q.B,
      F := midpoint q.B q.C,
      G := midpoint q.C q.D,
      H := midpoint q.D q.A,
      M := midpoint q.A q.C,
      N := midpoint q.B q.D,
      HE := (H.x, E.x), (H.y, E.y),
      GF := (G.x, F.x), (G.y, F.y),
      R := intersection_point_of_lines HE GF in
  segment R MN bisects MN := sorry

end quadrilateral_midpoint_property_l493_493008


namespace percentage_increase_correct_l493_493477

-- Definitions based on the conditions
def cost_repair : ℝ := 10.50
def duration_repair : ℝ := 1.0
def cost_new : ℝ := 30.00
def duration_new : ℝ := 2.0

-- Define the average costs per year
def avg_cost_repair := cost_repair / duration_repair
def avg_cost_new := cost_new / duration_new

-- Define the percentage increase calculation
def percentage_increase := ((avg_cost_new - avg_cost_repair) / avg_cost_repair) * 100

-- The theorem statement: 
theorem percentage_increase_correct : percentage_increase ≈ 42.86 := sorry

end percentage_increase_correct_l493_493477


namespace company_KW_price_percentage_l493_493163

theorem company_KW_price_percentage
  (A B : ℝ)
  (h1 : ∀ P: ℝ, P = 1.9 * A)
  (h2 : ∀ P: ℝ, P = 2 * B) :
  Price = 131.034 / 100 * (A + B) := 
by
  sorry

end company_KW_price_percentage_l493_493163


namespace intersection_AF_BF_inv_sum_l493_493639

noncomputable def parametric_x (t : ℝ) : ℝ := 1 + (1 / 2) * t
noncomputable def parametric_y (t : ℝ) : ℝ := (Real.sqrt 3 / 2) * t

def polar_C2 (θ : ℝ) : ℝ := Real.sqrt (12 / (3 + Real.sin θ ^ 2))

theorem intersection_AF_BF_inv_sum :
  let x (t : ℝ) := 1 + (1 / 2) * t
  let y (t : ℝ) := (Real.sqrt 3 / 2) * t
  let C1 := y = Real.sqrt 3 * (x - 1)
  let C2 := (x^2 / 4) + (y^2 / 3) = 1
  let F : ℝ × ℝ := (1, 0)
  ∃ (t1 t2 : ℝ), let A : ℝ × ℝ := (parametric_x t1, parametric_y t1)
  let B : ℝ × ℝ := (parametric_x t2, parametric_y t2)
  t1 + t2 = - (4 / 5) ∧ t1 * t2 = - (12 / 5) ∧ 
  (1 / Real.dist A F) + (1 / Real.dist B F) = 4 / 3 := sorry

end intersection_AF_BF_inv_sum_l493_493639


namespace sequence_max_length_x_l493_493894

theorem sequence_max_length_x (x : ℕ) : 
  (∀ n, a_n = 500 ∧ a_{n+1} = x → (a_{n+2} = a_n - a_{n+1})) →
  (a_{11} > 0 ∧ a_{10} > 0 → x = 500) :=
by
  sorry

end sequence_max_length_x_l493_493894


namespace hotel_room_mistake_l493_493400

theorem hotel_room_mistake (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  100 * a + 10 * b + c = (a + 1) * (b + 1) * c → false := by sorry

end hotel_room_mistake_l493_493400


namespace circumcircle_eq_l493_493232

open Real

def point := (ℝ × ℝ)

def O : point := (0, 0)
def A : point := (4, 0)
def B : point := (0, 3)

theorem circumcircle_eq :
  ∃ (D E F : ℝ), 
  (x^2 + y^2 + D*x + E*y + F = 0) ∧ 
  (4 * D + 16 + F = 0) ∧
  (3 * E + 9 + F = 0) ∧ 
  F = 0 ∧ 
  D = -4 ∧ 
  E = -3 :=
by {
  rcases (4 * D + 16 + F) with _ | _,
  rcases (3 * E + 9 + F) with _ | _,
  ring_nf at *,
  sorry
}

end circumcircle_eq_l493_493232


namespace remainder_correct_l493_493917

noncomputable def p : Polynomial ℝ := Polynomial.C 3 * Polynomial.X^5 + Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X + Polynomial.C 8
noncomputable def d : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) ^ 2
noncomputable def r : Polynomial ℝ := Polynomial.C 16 * Polynomial.X - Polynomial.C 8 

theorem remainder_correct : (p % d) = r := by sorry

end remainder_correct_l493_493917


namespace median_of_data_set_is_5_5_l493_493834

theorem median_of_data_set_is_5_5 : 
  let data := [3, 4, 5, 6, 6, 7];
  ∃ m, m = 5.5 ∧ m = (data.nth_le 2 sorry + data.nth_le 3 sorry) / 2 :=
by {
  -- Arrange the data in ascending order (already sorted in this case)
  let data_sorted := data.sorted (λ a b, a ≤ b);
  -- Find the middle elements for even length
  let m1 := data_sorted.nth_le 2 sorry;
  let m2 := data_sorted.nth_le 3 sorry;
  -- Calculate the median
  let median := (m1 + m2)/2;
  -- Assert the result
  use median;
  sorry
}

end median_of_data_set_is_5_5_l493_493834


namespace area_triangle_PNF_is_correct_l493_493306

-- Define the given conditions
def PQ : ℝ := 10
def QR : ℝ := 8
def N_midpoint_PR : Bool := true
def NF_perpendicular_PR : Bool := true
def PF : ℝ := PQ / 3

-- Define the goal, which is to prove the area of triangle PNF given the conditions
theorem area_triangle_PNF_is_correct :
  let PR := Real.sqrt (PQ^2 + QR^2)
  let PN := PR / 2
  let NF := Real.sqrt ((PN^2) - (PF^2))
  (1/2) * PF * NF = 5 * Real.sqrt 246 / 9 :=
by
  sorry

end area_triangle_PNF_is_correct_l493_493306


namespace perpendicular_vectors_x_value_l493_493267

theorem perpendicular_vectors_x_value :
  let a := (4, 2)
  let b := (x, 3)
  a.1 * b.1 + a.2 * b.2 = 0 -> x = -3/2 :=
by
  intros
  sorry

end perpendicular_vectors_x_value_l493_493267


namespace units_digit_of_150_factorial_is_zero_l493_493466

theorem units_digit_of_150_factorial_is_zero : 
  ∃ k : ℕ, (150! = k * 10) :=
begin
  -- We need to prove that there exists a natural number k such that 150! is equal to k times 10
  sorry
end

end units_digit_of_150_factorial_is_zero_l493_493466


namespace second_derivative_parametric_l493_493199

noncomputable theory

def x (t : ℝ) := Real.sin t
def y (t : ℝ) := Real.log (Real.cos t)

theorem second_derivative_parametric:
  ∀ t : ℝ, 
  diff2_yx :=
    -((1 + (Real.sin t)^2) / (Real.cos t)^4) :=
begin
  sorry
end

end second_derivative_parametric_l493_493199


namespace qin_jiushao_value_l493_493424

def polynomial (x : ℤ) : ℤ :=
  2 * x^5 + 5 * x^4 + 8 * x^3 + 7 * x^2 - 6 * x + 11

def step1 (x : ℤ) : ℤ := 2 * x + 5
def step2 (x : ℤ) (v : ℤ) : ℤ := v * x + 8
def step3 (x : ℤ) (v : ℤ) : ℤ := v * x + 7
def step_v3 (x : ℤ) (v : ℤ) : ℤ := v * x - 6

theorem qin_jiushao_value (x : ℤ) (v3 : ℤ) (h1 : x = 3) (h2 : v3 = 130) :
  step_v3 3 (step3 3 (step2 3 (step1 3))) = v3 :=
by {
  sorry
}

end qin_jiushao_value_l493_493424


namespace cylinder_volume_ratio_l493_493115

-- First define the problem: a 6 x 10 rectangle rolled to form two different cylinders
theorem cylinder_volume_ratio : 
  let r1 := 3 / Real.pi in
  let V1 := Real.pi * r1^2 * 10 in
  let r2 := 5 / Real.pi in
  let V2 := Real.pi * r2^2 * 6 in
  V2 / V1 = 5 / 3 :=
by
  -- The proof steps are omitted as the theorem states only.
  sorry

end cylinder_volume_ratio_l493_493115


namespace only_integer_solution_l493_493743

theorem only_integer_solution (a b c d : ℤ) (h : a^2 + b^2 = 3 * (c^2 + d^2)) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := 
by
  sorry

end only_integer_solution_l493_493743


namespace FK_passes_through_fixed_point_l493_493325

-- Step d) Rewrite the math proof problem in Lean 4 statement.
theorem FK_passes_through_fixed_point (A B C D E F K : Type) [Triangle A B C] (h : AB < AC)
  (hD : on_segment D A B) (hE : parallel_through D B C E A C) 
  (hF : perpendicular_bisector D E F B C) (hK : intersect_circles B D F C E F K) 
  : passes_through_fixed_point (line_through F K) :=
sorry

end FK_passes_through_fixed_point_l493_493325


namespace find_x_l493_493681

/-
In the diagram, given the following conditions:
- \( \angle ABC = 60^\circ \)
- \( \angle ACB = 90^\circ \)
- \( \angle CDE = 48^\circ \)
- \( \angle ADC = 180^\circ \)
- \( \angle AEB = 180^\circ \)

Prove that \( x = 162^\circ \).
-/

theorem find_x 
  (angle_ABC : ℝ = 60)
  (angle_ACB : ℝ = 90)
  (angle_CDE : ℝ = 48)
  (angle_ADC : ℝ = 180)
  (angle_AEB : ℝ = 180)
  : ℝ := sorry

end find_x_l493_493681


namespace diane_postage_problem_l493_493560

-- Definition of stamps
def stamps : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]

-- Define a function to compute the number of arrangements that sums to a target value
def arrangements_sum_to (target : ℕ) (stamps : List (ℕ × ℕ)) : ℕ :=
  sorry -- Implementation detail is skipped

-- The main theorem to prove
theorem diane_postage_problem :
  arrangements_sum_to 15 stamps = 271 :=
by sorry

end diane_postage_problem_l493_493560


namespace total_students_participated_l493_493668

open Set

variables (A B C : Set α) 
  (hA : card (A) = 203)
  (hB : card (B) = 179)
  (hC : card (C) = 165)
  (hAB : card (A ∩ B) = 143)
  (hBC : card (B ∩ C) = 97)
  (hCA : card (C ∩ A) = 116)
  (hABC : card (A ∩ B ∩ C) = 89)

theorem total_students_participated : card (A ∪ B ∪ C) = 280 :=
  by
  sorry

end total_students_participated_l493_493668


namespace binary_add_sub_l493_493856

theorem binary_add_sub :
  (binary.mk [1, 0, 1, 1, 0, 1]) + (binary.mk [1, 1, 1]) + (binary.mk [1, 1, 0, 0, 1, 1, 0]) - (binary.mk [1, 0, 1, 0]) 
  = binary.mk [1, 1, 0, 1, 1, 1, 0, 1] :=
sorry

end binary_add_sub_l493_493856


namespace solve_equation_l493_493911

theorem solve_equation (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) :
  ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) + 1 )
  ↔ x = 5 / 4 ∨ x = -2 :=
by sorry

end solve_equation_l493_493911


namespace pool_balls_pyramid_arrangement_l493_493673

/-- In how many distinguishable ways can 10 distinct pool balls be arranged in a pyramid
    (6 on the bottom, 3 in the middle, 1 on the top), assuming that all rotations of the pyramid are indistinguishable? -/
def pyramid_pool_balls_distinguishable_arrangements : Nat :=
  let total_arrangements := Nat.factorial 10
  let indistinguishable_rotations := 9
  total_arrangements / indistinguishable_rotations

theorem pool_balls_pyramid_arrangement :
  pyramid_pool_balls_distinguishable_arrangements = 403200 :=
by
  -- Proof will be added here
  sorry

end pool_balls_pyramid_arrangement_l493_493673


namespace coprime_product_consecutive_l493_493368

theorem coprime_product_consecutive (n : ℕ) (h : n > 0) : 
  ∃ (a : Fin n → ℕ), (∀ i, 2 ≤ a i) ∧ (∀ i j, i ≠ j → Nat.coprime (a i) (a j)) ∧ 
  (∃ k, (∏ i, a i) - 1 = k * (k + 1)) :=
by sorry

end coprime_product_consecutive_l493_493368


namespace math_problem_l493_493921

theorem math_problem
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x = z * (1 / y)) : 
  (x - z / x) * (y + 1 / (z * y)) = (x^4 - z^3 + x^2 * (z^2 - z)) / (z * x^2) :=
by
  sorry

end math_problem_l493_493921


namespace polygon_interior_angle_increase_l493_493081

theorem polygon_interior_angle_increase (n : ℕ) (h : 3 ≤ n) :
  ((n + 1 - 2) * 180 - (n - 2) * 180 = 180) :=
by sorry

end polygon_interior_angle_increase_l493_493081


namespace smallest_points_in_T_l493_493140

def symmetric_origin (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (-a, -b) ∈ T

def symmetric_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (a, -b) ∈ T

def symmetric_y (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (-a, b) ∈ T

def symmetric_y_eq_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (b, a) ∈ T

def symmetric_y_eq_neg_x (T : Set (ℝ × ℝ)) : Prop :=
  ∀ {a b : ℝ}, (a, b) ∈ T → (-b, -a) ∈ T

def point_in_T (T : Set (ℝ × ℝ)) : Prop :=
  (3, 4) ∈ T

theorem smallest_points_in_T (T : Set (ℝ × ℝ)) 
  (h_origin : symmetric_origin T)
  (h_x : symmetric_x T)
  (h_y : symmetric_y T)
  (h_y_eq_x : symmetric_y_eq_x T)
  (h_y_eq_neg_x : symmetric_y_eq_neg_x T)
  (h_point_in_T : point_in_T T) :
  ∃ (S : Set (ℝ × ℝ)), S ⊆ T ∧ S.finite ∧ S.card = 8 :=
sorry

end smallest_points_in_T_l493_493140


namespace problem_1_tangent_problem_2_range_problem_3_max_value_l493_493715

open Real

noncomputable def f (x a : ℝ) : ℝ :=
  log x + 0.5 * x^2 - (a + 2) * x

def tangent_vertical_at_P1 (a : ℝ) : Prop :=
  let df := (1 / (1 : ℝ)) + 1 - (a + 2) in
  df = 0 → a = 0

def range_of_f_sum (a : ℝ) : Prop :=
  ∀ m n : ℝ, m < n ∧ (m + n) = a + 2 ∧ m * n = 1 →
    (f m a) + (f n a) < -3

def max_of_f_diff (a n : ℝ) : ℝ :=
  if h : n ≥ sqrt e + 1 / sqrt e - 2 then
    let m := 1 / n in
    let df := log (n / m) + 0.5 * (n^2 - m^2) - (a + 2) * (n - m) in
    let df' := log (n / m) - 0.5 * (n - m) in
    df' = 1 - e / 2 + 1 / (2 * e)
  else 0

theorem problem_1_tangent : ∀ a : ℝ, tangent_vertical_at_P1 a :=
sorry

theorem problem_2_range : ∀ a : ℝ, range_of_f_sum a :=
sorry

theorem problem_3_max_value : ∀ a n : ℝ, n ≥ sqrt e + 1 / sqrt e - 2 → max_of_f_diff a n = 1 - e / 2 + 1 / (2 * e) :=
sorry

end problem_1_tangent_problem_2_range_problem_3_max_value_l493_493715


namespace parabola2_intersection_l493_493845

variables {a b : ℝ}

def parabola1 (x : ℝ) : ℝ := a * (x - 10) * (x - 13)
def vertex_pi1_x := (10 + 13) / 2

def vertex_pi2_x := 2 * vertex_pi1_x

theorem parabola2_intersection :
  vertex_pi2_x = 23 →
  (2 * 23) - 13 = 33 :=
by
  intros h_vertex
  calc
    (2 * 23) - 13 = 46 - 13 : by ring
    ... = 33          : by norm_num

end parabola2_intersection_l493_493845


namespace non_empty_subsets_count_l493_493272

theorem non_empty_subsets_count:
  let S := finset.range 20 in
  let non_consecutive_subset (T : finset ℕ) : Prop :=
    (∀ x y ∈ T, (x ≠ y) → (abs (x - y) > 1)) ∧
    (T ≠ ∅) ∧
    (T.card = k → ∀ x ∈ T, x ≥ k) in
  (finset.univ.filter non_consecutive_subset).card = 656 :=
sorry

end non_empty_subsets_count_l493_493272


namespace intersection_proof_l493_493683

noncomputable def pointP : ℝ × ℝ := (0, real.sqrt 3)

noncomputable def curveC (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - real.sqrt 3) ^ 2 = 4

noncomputable def lineL (t : ℝ) : ℝ × ℝ :=
  (-real.sqrt 2 / 2 * t, real.sqrt 3 + real.sqrt 2 / 2 * t)

theorem intersection_proof :
  ∃ t1 t2 : ℝ, 
    ∀ P A B : ℝ × ℝ,
    P = pointP → 
    curveC (fst P) (snd P) → 
    curveC (fst A) (snd A) → 
    curveC (fst B) (snd B) → 
    A = lineL t1 → 
    B = lineL t2 → 
    (t1 + t2 = real.sqrt 2 ∧ t1 * t2 = -3) →
    (abs (PA.val / PB.val) + abs (PB.val / PA.val)) = 8 / 3 :=
begin
  intros t1 t2 P A B hP hC1 hC2 hC3 hA hB h,
  sorry
end

end intersection_proof_l493_493683


namespace system_of_equations_solution_l493_493014

theorem system_of_equations_solution (C₁ C₂ C₃ : ℝ) :
  let x := λ t : ℝ, C₁ * Real.exp(2*t) + C₂ * Real.exp(3*t) + C₃ * Real.exp(6*t)
  let y := λ t : ℝ, C₂ * Real.exp(3*t) - 2 * C₃ * Real.exp(6*t)
  let z := λ t : ℝ, -C₁ * Real.exp(2*t) + C₂ * Real.exp(3*t) + C₃ * Real.exp(6*t)
  ∀ t : ℝ,
  let x' := (deriv x) in
  let y' := (deriv y) in
  let z' := (deriv z) in
  (x' t = 3 * x t - y t + z t) ∧
  (y' t = -x t + 5 * y t - z t) ∧
  (z' t = x t - y t + 3 * z t) := by
  sorry

end system_of_equations_solution_l493_493014


namespace find_y_intercept_l493_493405

-- Definitions for conditions
def slope : ℝ := 6
def x_intercept : ℝ × ℝ := (8, 0)

-- The theorem stating that the y-intercept is (0, -48)
theorem find_y_intercept (m : ℝ) (x1 y1 : ℝ) (h_slope : m = slope) (h_point : (x1, y1) = x_intercept) : (0, y1 - m * x1) = (0, -48) :=
by {
  -- Substituting known values and showing the result
  sorry
}

end find_y_intercept_l493_493405


namespace prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l493_493001

def person_A_hits : ℚ := 1 / 2
def person_B_hits : ℚ := 1 / 3

def person_A_misses : ℚ := 1 - person_A_hits
def person_B_misses : ℚ := 1 - person_B_hits

def exactly_one_hits : ℚ := (person_A_hits * person_B_misses) + (person_B_hits * person_A_misses)
def at_least_one_hits : ℚ := 1 - (person_A_misses * person_B_misses)

theorem prob_exactly_one_hits_is_one_half : exactly_one_hits = 1 / 2 := sorry

theorem prob_at_least_one_hits_is_two_thirds : at_least_one_hits = 2 / 3 := sorry

end prob_exactly_one_hits_is_one_half_prob_at_least_one_hits_is_two_thirds_l493_493001


namespace cosA_proof_area_proof_l493_493293

-- Define the problem conditions
variables {A B C : ℝ} -- angles of triangle ABC
variables {a b c : ℝ} -- lengths of sides opposite to angles A B C respectively

-- Conditions
-- condition (1): c * cos A + a * cos C = 2 * b * cos A
def condition1 (a b c : ℝ) (cosA cosC : ℝ) : Prop :=
  c * cosA + a * cosC = 2 * b * cosA

-- condition (2): a = sqrt(7)
def condition2 (a : ℝ) : Prop :=
  a = Real.sqrt 7

-- condition (3): b + c = 4
def condition3 (b c : ℝ) : Prop :=
  b + c = 4

-- Intended to prove cos A
theorem cosA_proof (a b c A C : ℝ) (cosA cosC : ℝ) 
  (h1 : condition1 a b c cosA cosC)
  (h2 : condition2 a)
  (h3 : condition3 b c) : cosA = 1 / 2 :=
sorry

-- Define area of the triangle
def area_of_triangle (a b c cosA : ℝ) : ℝ :=
  let bc := b * c in
  let sinA := Real.sqrt (1 - cosA * cosA) in
  (1/2) * b * c * sinA

-- Intended to find and prove the area of triangle
theorem area_proof (a b c A C : ℝ) (cosA cosC : ℝ)
  (h1 : condition1 a b c cosA cosC)
  (h2 : condition2 a)
  (h3 : condition3 b c) (hCosA : cosA = 1 / 2) :
  area_of_triangle a b c cosA = (3 * Real.sqrt 3) / 4 :=
sorry

end cosA_proof_area_proof_l493_493293


namespace ball_hits_ground_l493_493534

def height (t : ℝ) : ℝ := -6 * t^2 - 30 * t + 180

theorem ball_hits_ground (t : ℝ) : height t = 0 → t = 5 := 
begin
  intros h_eq,
  sorry
end

end ball_hits_ground_l493_493534


namespace total_dogs_l493_493815

axiom brown_dogs : ℕ
axiom white_dogs : ℕ
axiom black_dogs : ℕ

theorem total_dogs (b w bl : ℕ) (h1 : b = 20) (h2 : w = 10) (h3 : bl = 15) : (b + w + bl) = 45 :=
by {
  sorry
}

end total_dogs_l493_493815


namespace arithmetic_sequence_30th_term_l493_493030

theorem arithmetic_sequence_30th_term (a1 a2 a3 d a30 : ℤ) 
 (h1 : a1 = 3) (h2 : a2 = 12) (h3 : a3 = 21) 
 (h4 : d = a2 - a1) (h5 : a3 = a1 + 2 * d) 
 (h6 : a30 = a1 + 29 * d) : 
 a30 = 264 :=
by
  sorry

end arithmetic_sequence_30th_term_l493_493030


namespace households_surveyed_l493_493847

theorem households_surveyed 
  (h_both : Nat := 120) 
  (h_gasoline_no_electricity : Nat := 60) 
  (h_neither : Nat := 24)
  (h_electricity_no_gasoline : Nat := 4 * h_neither) :
  h_both + h_gasoline_no_electricity + h_electricity_no_gasoline + h_neither = 300 :=
begin
  sorry
end

end households_surveyed_l493_493847


namespace smallest_possible_value_l493_493080

theorem smallest_possible_value (n : ℕ) (h1 : ∀ m, (Nat.lcm 60 m / Nat.gcd 60 m = 24) → m = n) (h2 : ∀ m, (m % 5 = 0) → m = n) : n = 160 :=
sorry

end smallest_possible_value_l493_493080


namespace random_point_between_R_S_l493_493362

theorem random_point_between_R_S {P Q R S : ℝ} (PQ PR RS : ℝ) (h1 : PQ = 4 * PR) (h2 : PQ = 8 * RS) :
  let PS := PR + RS
  let probability := RS / PQ
  probability = 5 / 8 :=
by
  let PS := PR + RS
  let probability := RS / PQ
  sorry

end random_point_between_R_S_l493_493362


namespace number_of_valid_mappings_l493_493840

def valid_mappings (f : {a, b, c, d} → {1, 2, 3}) :
  Prop :=  (10 < f a * f b) ∧ (f c * f d < 20)

theorem number_of_valid_mappings :
  ∃ (f : {a, b, c, d} → {1, 2, 3}), valid_mappings f ∧ (fine.count_valid_mappings f = 25) :=
by
  sorry

end number_of_valid_mappings_l493_493840


namespace complex_problem_l493_493716

theorem complex_problem (z : ℂ) (h : 8 * complex.abs z ^ 2 = 3 * complex.abs (z + 3) ^ 2 + complex.abs (z^2 + 2) ^ 2 + 50) :
  z + 9 / z = -4 :=
by
  sorry

end complex_problem_l493_493716


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l493_493996

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l493_493996


namespace bisection_method_interval_l493_493781

theorem bisection_method_interval :
  let f (x : ℝ) := log x - 3 + x in
  (f 2) * (f 3) < 0 :=
by
  let f (x : ℝ) := log x - 3 + x
  have f2 : f 2 = log 2 - 1 := by rfl
  have f3 : f 3 = log 3 := by rfl
  calc
    (log 2 - 1) * log 3 < 0 : sorry

end bisection_method_interval_l493_493781


namespace range_of_x0_l493_493972

open Real

-- Define the problem statement using Lean 4
theorem range_of_x0 (x0 y0 : ℝ) (hP : x0 = y0 + 2)
    (hC : x0^2 + y0^2 = 1)
    (existsQ : ∃ Q : ℝ × ℝ, (Q.1^2 + Q.2^2 = 1) ∧ (∠(0,0) P Q = 30)) :
    0 ≤ x0 ∧ x0 ≤ 2 :=
by
  sorry -- Proof not required

end range_of_x0_l493_493972


namespace correct_conclusions_l493_493632

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def f (x : ℝ) : ℝ := Real.sin (Real.floor (Real.cos x)) + Real.cos (Real.floor (Real.sin x))

theorem correct_conclusions : 
  is_periodic f (2 * Real.pi) ∧ 
  ¬ (is_odd f ∨ is_even f) ∧ 
  ¬ ∀ x, 0 < x ∧ x < Real.pi → f x < f (x + 1) ∧ 
  ∃ x, f x > Real.sqrt 2 :=
by
  sorry

end correct_conclusions_l493_493632


namespace find_min_value_l493_493214

def point (α : Type*) := α × α
 
def dist (P Q : point ℝ) : ℝ :=
real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def parabola (x y : ℝ) : Prop :=
x * x = -4 * y

def min_value_condition (P : point ℝ) (Q : point ℝ) (y : ℝ) : ℝ :=
abs y + dist P Q

def min_value (P : point ℝ) (Q : point ℝ) (y : ℝ) : Prop :=
P.2 = y ∧ parabola P.1 y ∧ Q = (-2 * real.sqrt 2, 0) ∧ (∀ (z : ℝ), min_value_condition P Q z ≥ min_value_condition P Q y ∧ min_value_condition P Q y = 2)

theorem find_min_value : 
  ∃ (P : point ℝ) (y : ℝ), min_value P (-2 * real.sqrt 2, 0) y :=
sorry

end find_min_value_l493_493214


namespace pen_refill_count_l493_493279

noncomputable
def pen_refills_needed : ℕ :=
  let letters_per_core := 3 + (2 / 3) in  -- Assuming each pen core lasts for 3 and 2/3 letters.
  (16 / letters_per_core).ceil

theorem pen_refill_count :
  pen_refills_needed = 5 :=
by
  unfold pen_refills_needed
  -- Perform inner calculations to show this equivalently results to 5.
  sorry  -- Proof steps are not required as per instructions.

end pen_refill_count_l493_493279


namespace segment_displacement_arbitrarily_large_l493_493851

-- Definitions based on conditions:
structure Segment (α : Type) :=
(A B : α)

structure Line (α : Type) :=
(points : set α)

variable {α : Type}

-- Conditions based on the problem statement:
def on_line (s : Segment α) (l : Line α) := s.A ∈ l.points ∧ s.B ∈ l.points

def parallel_movement (s : Segment α) (l : Line α) (new_s : Segment α) :=
  (∀ x ∈ l.points, x ∉ set.range (λ t, (s.A, new_s.A + t)) ∧ x ∉ set.range (λ t, (s.B, new_s.B + t))) ∧
  (s.A - s.B) = (new_s.A - new_s.B)

-- Theorem statement to prove that point A can be displaced arbitrarily far:
theorem segment_displacement_arbitrarily_large (s : Segment α) (l : Line α) :
  on_line s l → (∃ δ : α, ∀ ε > 0, ∃ new_s : Segment α, parallel_movement s l new_s ∧ (new_s.A - s.A) > δ) :=
by
  intros h_on_line
  sorry

end segment_displacement_arbitrarily_large_l493_493851


namespace point_to_focus_distance_l493_493638

def parabola : Set (ℝ × ℝ) := { p | p.2^2 = 4 * p.1 }

def point_P : ℝ × ℝ := (3, 2) -- Since y^2 = 4*3 hence y = ±2 and we choose one of the (3, 2) or (3, -2)

def focus_F : ℝ × ℝ := (1, 0) -- Focus of y^2 = 4x is (1, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem point_to_focus_distance : distance point_P focus_F = 4 := by
  sorry -- Proof goes here

end point_to_focus_distance_l493_493638


namespace rectangle_area_l493_493037

theorem rectangle_area :
  ∃ (l w : ℝ), l = 4 * w ∧ 2 * l + 2 * w = 200 ∧ l * w = 1600 :=
by
  use [80, 20]
  split; norm_num
  split; norm_num
  sorry

end rectangle_area_l493_493037


namespace find_excluded_number_l493_493479

-- Definition of the problem conditions
def avg (nums : List ℕ) : ℕ := (nums.sum / nums.length)

-- Problem condition: the average of 5 numbers is 27
def condition1 (nums : List ℕ) : Prop :=
  nums.length = 5 ∧ avg nums = 27

-- Problem condition: excluding one number, the average of remaining 4 numbers is 25
def condition2 (nums : List ℕ) (x : ℕ) : Prop :=
  let nums' := nums.filter (λ n => n ≠ x)
  nums.length = 5 ∧ nums'.length = 4 ∧ avg nums' = 25

-- Proof statement: finding the excluded number
theorem find_excluded_number (nums : List ℕ) (x : ℕ) (h1 : condition1 nums) (h2 : condition2 nums x) : x = 35 := 
by
  sorry

end find_excluded_number_l493_493479


namespace range_of_z_l493_493961

theorem range_of_z (x y : ℝ) (h1 : -4 ≤ x - y ∧ x - y ≤ -1) (h2 : -1 ≤ 4 * x - y ∧ 4 * x - y ≤ 5) :
  ∃ (z : ℝ), z = 9 * x - y ∧ -1 ≤ z ∧ z ≤ 20 :=
sorry

end range_of_z_l493_493961


namespace distance_from_Zlatoust_to_Miass_l493_493776

variables {x g m k : ℝ}

axioms (h1 : (x + 18) / k = (x - 18) / m)
       (h2 : (x + 25) / k = (x - 25) / g)
       (h3 : (x + 8) / m = (x - 8) / g)

theorem distance_from_Zlatoust_to_Miass : x = 60 :=
by sorry

end distance_from_Zlatoust_to_Miass_l493_493776


namespace weight_of_new_person_l493_493750

theorem weight_of_new_person (avg_increase : ℝ) (num_persons : ℕ) (old_weight : ℝ) (new_weight : ℝ) :
  avg_increase = 2.5 → num_persons = 8 → old_weight = 60 → 
  new_weight = old_weight + num_persons * avg_increase → new_weight = 80 :=
  by
    intros
    sorry

end weight_of_new_person_l493_493750


namespace neg_p_implies_neg_q_sufficient_but_not_necessary_l493_493958

variables (x : ℝ) (p : Prop) (q : Prop)

def p_condition := (1 < x ∨ x < -3)
def q_condition := (5 * x - 6 > x ^ 2)

theorem neg_p_implies_neg_q_sufficient_but_not_necessary :
  p_condition x → q_condition x → ((¬ p_condition x) → (¬ q_condition x)) :=
by 
  intro h1 h2
  sorry

end neg_p_implies_neg_q_sufficient_but_not_necessary_l493_493958


namespace keegan_other_class_average_time_l493_493696

def total_school_time_hours := 9
def total_classes := 10
def history_chemistry_hours := 2
def mathematics_hours := 1.5

theorem keegan_other_class_average_time :
  let total_school_time_minutes := total_school_time_hours * 60
  let history_chemistry_minutes := history_chemistry_hours * 60
  let mathematics_minutes := mathematics_hours * 60
  let total_subject_minutes := history_chemistry_minutes + mathematics_minutes
  let remaining_minutes := total_school_time_minutes - total_subject_minutes
  let num_other_classes := total_classes - 3
  let average_other_class_time := remaining_minutes / num_other_classes
  average_other_class_time = 47.14 :=
by
  sorry

end keegan_other_class_average_time_l493_493696


namespace polynomial_roots_absolute_sum_l493_493597

theorem polynomial_roots_absolute_sum (p q r : ℤ) (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2027) :
  |p| + |q| + |r| = 98 := 
sorry

end polynomial_roots_absolute_sum_l493_493597


namespace directrix_of_parabola_l493_493582

theorem directrix_of_parabola (a b c : ℝ) (parabola_eqn : ∀ x : ℝ, y = 3 * x^2 - 6 * x + 2)
  (vertex : ∃ h k : ℝ, h = 1 ∧ k = -1)
  : ∃ y : ℝ, y = -13 / 12 := 
sorry

end directrix_of_parabola_l493_493582


namespace triangle_KBC_area_l493_493682

-- Define the conditions
def FE : ℝ := 7
def BC : ℝ := FE
def JB : ℝ := 5
def BK : ℝ := JB
def angle_CBK : ℝ := 15 * Real.pi / 180 -- converting degrees to radians

-- Calculate the area of triangle KBC
def triangle_area (BC BK : ℝ) (angle_CBK : ℝ) : ℝ :=
  0.5 * BC * BK * Real.sin angle_CBK

-- State the theorem
theorem triangle_KBC_area :
  triangle_area BC BK angle_CBK = 4.5 :=
by
  -- Proof skipped
  sorry

end triangle_KBC_area_l493_493682


namespace twelve_star_three_eq_four_star_eight_eq_star_assoc_l493_493074

def star (a b : ℕ) : ℕ := 10^a * 10^b

theorem twelve_star_three_eq : star 12 3 = 10^15 :=
by 
  -- Proof here
  sorry

theorem four_star_eight_eq : star 4 8 = 10^12 :=
by 
  -- Proof here
  sorry

theorem star_assoc (a b c : ℕ) : star (a + b) c = star a (b + c) :=
by 
  -- Proof here
  sorry

end twelve_star_three_eq_four_star_eight_eq_star_assoc_l493_493074


namespace units_digit_of_150_factorial_is_zero_l493_493465

theorem units_digit_of_150_factorial_is_zero : 
  ∃ k : ℕ, (150! = k * 10) :=
begin
  -- We need to prove that there exists a natural number k such that 150! is equal to k times 10
  sorry
end

end units_digit_of_150_factorial_is_zero_l493_493465


namespace cost_of_pencil_l493_493599

theorem cost_of_pencil (x y : ℕ) (h1 : 4 * x + 3 * y = 224) (h2 : 2 * x + 5 * y = 154) : y = 12 := 
by
  sorry

end cost_of_pencil_l493_493599


namespace compute_d_l493_493233

noncomputable def d_value (c : ℚ) : ℚ :=
let root1 := (3 : ℚ) + real.sqrt 2 in
let root2 := (3 : ℚ) - real.sqrt 2 in
let root3 := (-36 : ℚ) / (root1 * root2) in
root1 * root2 + (root1 * root3) + (root2 * root3)

theorem compute_d (c : ℚ) (d : ℚ) (h : polynomial.aeval (3 + real.sqrt 2) (X^3 + c * X^2 + d * X - 36) = 0 ∧ c ∈ ℚ ∧ d ∈ ℚ) : 
  d = -23 - (6/7) :=
by
  have eq1 : (3:ℚ) + real.sqrt 2 ≠ (3:ℚ) - real.sqrt 2 := by sorry
  have eq2 : 3 + sqrt 2 ≠ 3 - sqrt 2 := by apply eq1 
  calc
    d = d_value c : by sorry
       ... = -23 - (6/7) : by sorry

end compute_d_l493_493233


namespace units_digit_of_150_factorial_is_zero_l493_493432

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l493_493432


namespace smallest_head_tail_number_maximum_value_head_tail_number_l493_493920

def is_head_tail_number (a b c : ℕ) : Prop :=
  (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (b - (a + c) = 1)

def is_valid_transformation (a b c : ℕ) : Prop :=
  let N := 100 * a + 10 * b + c in
  let N' := 100 * c + 10 * b + a in
  c ≠ 0 → (∃ k : ℕ, 9 * abs (a - c) = k^2)

theorem smallest_head_tail_number :
  ∃ a b c : ℕ, is_head_tail_number a b c ∧ (100 * a + 10 * b + c = 120) := sorry

theorem maximum_value_head_tail_number :
  ∃ a b c : ℕ, is_head_tail_number a b c ∧ is_valid_transformation a b c ∧ (100 * a + 10 * b + c = 692) := sorry

end smallest_head_tail_number_maximum_value_head_tail_number_l493_493920


namespace probability_james_wins_l493_493313

open Classical

def outcomes : Finset (ℕ × ℕ) := 
  Finset.product (Finset.range 6).succ (Finset.range 6).succ

def winning_pairs (x y : ℕ) : Prop := 
  abs (x - y) ≤ 2

def winning_outcomes : Finset (ℕ × ℕ) := 
  outcomes.filter (λ p, winning_pairs p.1 p.2)

theorem probability_james_wins : 
  winning_outcomes.card / outcomes.card = 2 / 3 :=
sorry

end probability_james_wins_l493_493313


namespace smallest_circle_area_l493_493077

noncomputable def function_y (x : ℝ) : ℝ := 6 / x - 4 * x / 3

theorem smallest_circle_area :
  ∃ r : ℝ, (∀ x : ℝ, r * r = x^2 + (function_y x)^2) → r^2 * π = 4 * π :=
sorry

end smallest_circle_area_l493_493077


namespace total_notes_l493_493091

theorem total_notes (total_amount : ℕ) (note_count : ℕ) (denom1 denom5 denom10 : ℕ) 
  (h1 : denom1 = denom5)
  (h2 : denom5 = denom10)
  (h3 : denom10 = note_count)
  (h4 : total_amount = denom1 * 1 + denom5 * 5 + denom10 * 10) : 
  note_count * 3 = 90 := 
by 
  have h5 : denom1 = note_count := by rw [h1, h2, h3]
  have h6 : 16 * note_count = total_amount := by rw [h4, h5]; ring
  have h7 : note_count = 30 := by linarith
  sorry

end total_notes_l493_493091


namespace loan_payment_difference_l493_493526

noncomputable def compounded_amount (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest_amount (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P + P * r * t

noncomputable def loan1_payment (P : ℝ) (r : ℝ) (n : ℝ) (t1 : ℝ) (t2 : ℝ) : ℝ :=
  let A1 := compounded_amount P r n t1
  let one_third_payment := A1 / 3
  let remaining := A1 - one_third_payment
  one_third_payment + compounded_amount remaining r n t2

noncomputable def loan2_payment (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  simple_interest_amount P r t

noncomputable def positive_difference (x y : ℝ) : ℝ :=
  if x > y then x - y else y - x

theorem loan_payment_difference: 
  ∀ P : ℝ, ∀ r1 r2 : ℝ, ∀ n : ℝ, ∀ t1 t2 : ℝ,
  P = 12000 → r1 = 0.08 → r2 = 0.09 → n = 12 → t1 = 7 → t2 = 8 →
  positive_difference 
    (loan2_payment P r2 (t1 + t2)) 
    (loan1_payment P r1 n t1 t2) = 2335 := 
by
  intros
  sorry

end loan_payment_difference_l493_493526


namespace area_outside_chords_l493_493419

theorem area_outside_chords (r chord_distance : ℝ) (h_radius : r = 10) (h_distance : chord_distance = 10) :
  let area_total := π * r^2
  let sector_area := 2 * (r^2 * π * 1/6)
  let triangle_area := 2 * (1/2 * (r/2) * (r * sqrt 3 / 2))
  let area_between_chords := sector_area - triangle_area
  let area_outside := area_total - area_between_chords
  area_outside = (200 * π / 3 - 25 * sqrt 3) :=
by
  sorry

end area_outside_chords_l493_493419


namespace bing_dwen_dwen_cost_minimum_bing_dwen_dwen_toys_l493_493747

-- Part 1: Prove the cost of each toy

theorem bing_dwen_dwen_cost (a b : ℕ) 
  (h1 : 4 * a + 5 * b = 1000) 
  (h2 : 5 * a + 10 * b = 1550) :
  a = 150 ∧ b = 80 :=
by
  sorry

-- Part 2: Prove the minimum number of "Bing Dwen Dwen" toys

theorem minimum_bing_dwen_dwen_toys (x : ℕ)
  (h1 : 180 - 150 = 30) -- Profit per "Bing Dwen Dwen" toy
  (h2 : 100 - 80 = 20)  -- Profit per "Shuey Rongrong" toy
  (total_toys : 180)
  (minimum_profit : 4600) :
  30 * x + 20 * (total_toys - x) ≥ minimum_profit → x ≥ 100 :=
by
  sorry

end bing_dwen_dwen_cost_minimum_bing_dwen_dwen_toys_l493_493747


namespace total_teams_is_correct_l493_493515

/-- The total number of different teams that can be formed from 10 engineers and 6 designers, 
    given that the team must consist of at least one person and the absolute difference 
    between the number of engineers and designers must be exactly 3 or a prime number. -/
def team_count : ℕ :=
  calc
    let primes := {2, 3, 5, 7}
    let total_teams := (∑ diff in primes, ∑ e in finset.range 11, ∑ d in finset.range 7, if |e - d| = diff then (@choose ℕ _ _ (10) (e)) * (@choose ℕ _ _ (6) (d)) else 0) - 1
  total_teams

/-- Prove the total number of different teams -/
theorem total_teams_is_correct : team_count = /* calculated value */ :=
by sorry

end total_teams_is_correct_l493_493515


namespace number_of_elements_in_A_intersection_N_l493_493641

def A := { x : ℝ | |x - 2| < 3 }
def N := { n : ℕ | True }

theorem number_of_elements_in_A_intersection_N : 
  finset.card ((finset.filter (λ x, x ∈ A) (finset.coe N : finset ℝ))) = 5 :=
sorry

end number_of_elements_in_A_intersection_N_l493_493641


namespace angle_B_in_triangle_l493_493311

open Real

theorem angle_B_in_triangle (a b c : ℝ) (h : a^2 + c^2 = b^2 + sqrt 3 * a * c) :
  angle (180 - arccos (sqrt 3 / 2)) = 30 :=
sorry

end angle_B_in_triangle_l493_493311


namespace least_positive_integer_with_exactly_10_factors_l493_493790

theorem least_positive_integer_with_exactly_10_factors : ∃ k : ℕ, (k > 0) ∧ (number_of_factors k = 10) ∧ (∀ m : ℕ, (m > 0) ∧ (number_of_factors m = 10) → k ≤ m) :=
sorry

end least_positive_integer_with_exactly_10_factors_l493_493790


namespace atomic_weight_F_l493_493586

noncomputable def molecular_weight_BaF2 : ℝ := 175
noncomputable def atomic_weight_Ba : ℝ := 137.33

theorem atomic_weight_F : 
  (∃ (x : ℝ), molecular_weight_BaF2 = atomic_weight_Ba + 2 * x ∧ x ≈ 18.835) := 
sorry

end atomic_weight_F_l493_493586


namespace simplify_expression1_simplify_expression2_l493_493742

-- Problem 1
theorem simplify_expression1 (a : ℝ) : 
  (a^2)^3 + 3 * a^4 * a^2 - a^8 / a^2 = 3 * a^6 :=
by sorry

-- Problem 2
theorem simplify_expression2 (x : ℝ) : 
  (x - 3) * (x + 4) - x * (x + 3) = -2 * x - 12 :=
by sorry

end simplify_expression1_simplify_expression2_l493_493742


namespace find_area_of_triangle_AEB_l493_493674

noncomputable def triangle_AEB_area : ℚ := 48 / 5

variables {A B C D F G E : Type}
variables (ABCD : A) 
variables (AB BC DF GC: ℝ)
variables (intersect_AF_BG : A → B → G → F → Type)

-- Conditions from the problem
def rectangle_configuration := 
  AB = 8 ∧ BC = 4 ∧ DF = 2 ∧ GC = 3 ∧ 
  intersect_AF_BG (intersect_AF_BG intersect_AF_BG intersect_AF_BG)

-- The statement we need to prove
theorem find_area_of_triangle_AEB
  (h : rectangle_configuration AB BC DF GC) :
  triangle_AEB_area = 48 / 5 :=
sorry

end find_area_of_triangle_AEB_l493_493674


namespace probability_same_number_l493_493527

theorem probability_same_number (h1: ∀ n, 0 < n ∧ n < 300 → (n % 15 = 0 ↔ n % 20 = 0)) : 
  let total_combinations := 20 * 15 in
  let common_multiples := 5 in
  ((common_multiples : ℚ) / total_combinations = (1 : ℚ) / 60) :=
by
  sorry

end probability_same_number_l493_493527


namespace class_size_relative_error_hall_people_relative_error_price_relative_error_print_sheet_relative_error_l493_493679

theorem class_size_relative_error (x Δx : ℕ) (hx : x = 40) (hΔx : Δx = 5) : Δx / x = 0.125 := by
  have h : 5 / 40 = 0.125 := sorry
  exact h

theorem hall_people_relative_error (x Δx : ℕ) (hx : x = 1500) (hΔx : Δx = 100) : Δx / x = 0.067 := by
  have h : 100 / 1500 = 0.067 := sorry
  exact h

theorem price_relative_error (x Δx : ℕ) (hx : x = 100) (hΔx : Δx = 5) : Δx / x = 0.05 := by
  have h : 5 / 100 = 0.05 := sorry
  exact h

theorem print_sheet_relative_error (x Δx : ℕ) (hx : x = 40000) (hΔx : Δx = 500) : Δx / x = 0.0125 := by
  have h : 500 / 40000 = 0.0125 := sorry
  exact h

end class_size_relative_error_hall_people_relative_error_price_relative_error_print_sheet_relative_error_l493_493679


namespace ratio_as_percentage_l493_493660

theorem ratio_as_percentage (x : ℝ) (h : (x / 2) / (3 * x / 5) = 3 / 5) : 
  (3 / 5) * 100 = 60 := 
sorry

end ratio_as_percentage_l493_493660


namespace perpendicular_chords_exist_l493_493486

theorem perpendicular_chords_exist 
  (circle : Type) [metric_space circle]
  (red_points : fin 100 → circle)
  (arc_lengths : fin 100 → ℕ)
  (h_distinct_lengths : ∀ (i j : fin 100), i ≠ j → arc_lengths i ≠ arc_lengths j)
  (h_arc_sum : (finset.univ.sum arc_lengths) = (finset.range 101).sum) :
  ∃ p1 p2 p3 p4 : fin 100, 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    dist (red_points p1) (red_points p2) = dist (red_points p3) (red_points p4) ∧
    dist (red_points p1) (red_points p3) ≠ dist (red_points p2) (red_points p4) := 
  sorry

end perpendicular_chords_exist_l493_493486


namespace triangle_isosceles_or_right_l493_493209

theorem triangle_isosceles_or_right (a b c : ℝ) (A B C : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (triangle_abc : A + B + C = 180)
  (opposite_sides : ∀ {x y}, x ≠ y → x + y < 180) 
  (condition : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = 90) :=
by {
  sorry
}

end triangle_isosceles_or_right_l493_493209


namespace min_value_of_f_l493_493916

noncomputable def f (x : ℝ) : ℝ := x^2 + 10*x + 100/(x^2)

theorem min_value_of_f : ∃ x > 0, f(x) = 79 ∧ ∀ y > 0, f(y) ≥ 79 :=
begin
  sorry,
end

end min_value_of_f_l493_493916


namespace ordered_pair_sol_l493_493046

noncomputable def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![5, d]]

noncomputable def is_inverse_scalar_mul (d k : ℝ) : Prop :=
  (A d)⁻¹ = k • (A d)

theorem ordered_pair_sol (d k : ℝ) :
  is_inverse_scalar_mul d k → (d = -2 ∧ k = 1 / 19) :=
by
  intros h
  sorry

end ordered_pair_sol_l493_493046


namespace roof_difference_l493_493814

theorem roof_difference 
  (w l : ℝ) 
  (h_length : l = 8 * w) 
  (h_area : l * w = 847) :
  l - w ≈ 72.03 :=
by
  sorry -- proof goes here

end roof_difference_l493_493814


namespace second_grade_students_sampled_l493_493511

-- Definitions corresponding to conditions in a)
def total_students := 2000
def mountain_climbing_fraction := 2 / 5
def running_ratios := (2, 3, 5)
def sample_size := 200

-- Calculation of total running participants based on ratio
def total_running_students :=
  total_students * (1 - mountain_climbing_fraction)

def a := 2 * (total_running_students / (2 + 3 + 5))
def b := 3 * (total_running_students / (2 + 3 + 5))
def c := 5 * (total_running_students / (2 + 3 + 5))

def running_sample_size := sample_size * (3 / 5) --since the ratio is 3:5

-- The statement to prove
theorem second_grade_students_sampled : running_sample_size * (3 / (2+3+5)) = 36 :=
by
  sorry

end second_grade_students_sampled_l493_493511


namespace range_of_a_minus_b_l493_493603

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 2 < b ∧ b < 4) : -4 < a - b ∧ a - b < -1 :=
by
  sorry

end range_of_a_minus_b_l493_493603


namespace find_b_l493_493907

theorem find_b (b : ℝ) (h : log b 625 = -2) : b = 1 / 25 :=
sorry

end find_b_l493_493907


namespace product_value_eq_l493_493079

theorem product_value_eq :
  ∏ n in Finset.range 98, (n + 1) * (n + 3) / ((n + 2) * (n + 2)) = 50 / 99 :=
sorry

end product_value_eq_l493_493079


namespace line_through_intersection_parallel_l493_493193

theorem line_through_intersection_parallel
  (x y k : ℝ)
  (h1 : 2 * x - y - 3 = 0)
  (h2 : 4 * x - 3 * y - 5 = 0)
  (h3 : 2 * x + 3 * y + k = 0) :
  k = -7 :=
by
  have h_inter : (x, y) = (2, 1), from sorry,
  have h_eq : 2 * 2 + 3 * 1 + k = 0, from sorry,
  show k = -7, from sorry

end line_through_intersection_parallel_l493_493193


namespace proposition_and_implication_l493_493734

theorem proposition_and_implication
  (m : ℝ)
  (h1 : 5/4 * (m^2 + m) > 0)
  (h2 : 1 + 9 - 4 * (5/4 * (m^2 + m)) > 0)
  (h3 : m + 3/2 ≥ 0)
  (h4 : m - 1/2 ≤ 0) :
  (-3/2 ≤ m ∧ m < -1) ∨ (0 < m ∧ m ≤ 1/2) :=
sorry

end proposition_and_implication_l493_493734


namespace monotonic_intervals_of_f_range_of_m_for_two_zeros_l493_493259

noncomputable def f (m x : ℝ) : ℝ := m * (2 * Real.log x - x) + (1 / x^2) - (1 / x)

theorem monotonic_intervals_of_f (m : ℝ) :
  (∀ x ∈ Ioo (0 : ℝ) 2, deriv (f m) x < 0) ∧
  (∀ x ∈ Ioi 2, deriv (f m) x > 0) ∨
  (0 < m ∧ m < (1 / 4) ∧
    (∀ x ∈ Ioo (0 : ℝ) 2, deriv (f m) x < 0) ∧
    (∀ x ∈ Ioo (2 : ℝ) (sqrt m / m), deriv (f m) x > 0) ∧
    (∀ x ∈ Ioi (sqrt m / m), deriv (f m) x < 0)) ∨
  (m = 1 / 4 ∧ (∀ x ∈ Ioi (0 : ℝ), deriv (f m) x < 0)) ∨
  (m > 1 / 4 ∧ (∀ x ∈ Ioo (0 : ℝ) (sqrt m / m), deriv (f m) x < 0) ∧
    (∀ x ∈ Ioo (sqrt m / m) 2, deriv (f m) x > 0) ∧
    (∀ x ∈ Ioi 2, deriv (f m) x < 0))
  :=
  sorry

theorem range_of_m_for_two_zeros :
  (∃ m : ℝ, (1 / (8 * (Real.log 2 - 1)) < m) ∧ m < 0 ∧
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ f m x₁ = 0 ∧ f m x₂ = 0))
  :=
  sorry

end monotonic_intervals_of_f_range_of_m_for_two_zeros_l493_493259


namespace max_x_plus_reciprocal_l493_493062

theorem max_x_plus_reciprocal (n : ℕ) (sum_n : ℝ) (sum_reciprocals : ℝ) (x : ℝ) (hne_zero : 2023 = n)
  (hsum_nn : 2024 = sum_n) (hsum_rec : 2024 = sum_reciprocals)
  (hx_pos : 0 < x) : x + 1/x ≤ 4049 / 2024 :=
begin
  sorry
end

end max_x_plus_reciprocal_l493_493062


namespace find_prime_pairs_l493_493188

open Nat

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem as a theorem in Lean
theorem find_prime_pairs :
  ∀ (p n : ℕ), is_prime p ∧ n > 0 ∧ p^3 - 2*p^2 + p + 1 = 3^n ↔ (p = 2 ∧ n = 1) ∨ (p = 5 ∧ n = 4) :=
by
  sorry

end find_prime_pairs_l493_493188


namespace vector_equation_l493_493617

variables {Point : Type} [AddCommGroup Point] [Module ℝ Point]

-- Given points on the plane
variables (O A B C : Point)

-- The definition of vector AC and CB
variables (vecAC : Point) (vecCB : Point)

-- The condition 2 * vecAC + vecCB = 0
axiom condition : 2 • vecAC + vecCB = 0

def vecOA := A - O
def vecOB := B - O
def vecOC := C - O

def vecAC := C - A
def vecCB := B - C

-- The statement to be proved
theorem vector_equation : vecOC = 2 • vecOA - vecOB := sorry

end vector_equation_l493_493617


namespace ellipse_foci_distance_l493_493600

noncomputable def distance_between_foci : ℝ :=
  2 * Real.sqrt (4.75)

theorem ellipse_foci_distance :
  let points : List (ℝ × ℝ) := [(1, 3), (10, 3), (4, -2), (4, 8)]
  let ellipse_center : ℝ × ℝ := (4, 3)
  let semi_major_axis_length : ℝ := 4.5
  let semi_minor_axis_length : ℝ := 5
  distance_between_foci = 
  2 * Real.sqrt (semi_minor_axis_length^2 - semi_major_axis_length^2) := 
  begin
    sorry
  end

end ellipse_foci_distance_l493_493600


namespace relation_among_a_b_c_l493_493604

noncomputable def a : ℝ := 2^(-1/3)
noncomputable def b : ℝ := (2^Real.log2 3)^(-1/2)
noncomputable def c : ℝ := 1/4 * ∫ x in 0..Real.pi, Real.sin x

theorem relation_among_a_b_c : a > b ∧ b > c := by
  sorry

end relation_among_a_b_c_l493_493604


namespace deepak_and_wife_meet_time_l493_493099

noncomputable def deepak_speed_kmph : ℝ := 20
noncomputable def wife_speed_kmph : ℝ := 12
noncomputable def track_circumference_m : ℝ := 1000

noncomputable def speed_to_m_per_min (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 60

noncomputable def deepak_speed_m_per_min : ℝ := speed_to_m_per_min deepak_speed_kmph
noncomputable def wife_speed_m_per_min : ℝ := speed_to_m_per_min wife_speed_kmph

noncomputable def combined_speed_m_per_min : ℝ :=
  deepak_speed_m_per_min + wife_speed_m_per_min

noncomputable def meeting_time_minutes : ℝ :=
  track_circumference_m / combined_speed_m_per_min

theorem deepak_and_wife_meet_time :
  abs (meeting_time_minutes - 1.875) < 0.01 :=
by
  sorry

end deepak_and_wife_meet_time_l493_493099


namespace max_length_sequence_309_l493_493899

def sequence (a₁ a₂ : ℤ) : ℕ → ℤ
| 0     := a₁
| 1     := a₂
| (n+2) := sequence n - sequence (n+1)

theorem max_length_sequence_309 :
  ∃ x : ℤ, x = 309 ∧
  (let a₁ := 500 in 
  let a₂ := x in
  sequence a₁ a₂ 9 > 0 ∧
  sequence a₁ a₂ 10 > 0) :=
sorry

end max_length_sequence_309_l493_493899


namespace max_sum_condition_l493_493568

theorem max_sum_condition 
  (a b c d e f g h : ℕ) 
  (h_distinct : list.nodup [a, b, c, d, e, f, g, h]) 
  (h_range : list.perm [a, b, c, d, e, f] [3, 6, 9, 12, 15, 18]) 
  (sum_eq1 : e + f + h = c + d + h) 
  (sum_eq2 : c + d + h = e + f + h) 
  (sum_eq3 : c + d + g = a + b + g)
  (sum_eq4 : a + g = b + d) :
  a + g + (b + d + e + f + h - g - h) = 39 :=
sorry

end max_sum_condition_l493_493568


namespace remarkable_seven_digit_count_l493_493723

/-- A natural number is remarkable if and only if all of its digits are distinct, 
  it does not start with the digit 2, and by removing some of its digits, the number 2018 can be obtained -/
def is_remarkable (n : ℕ) : Prop :=
  let digits := to_digits 10 n in
  (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) ∧
  (digits.head ≠ 2) ∧
  (∃ (subseq : List Nat), subseq ~ 2018 ∧ subseq ⊆ digits)

theorem remarkable_seven_digit_count :
  ∃ n, n = 1800 ∧ ∀ m, (m < 10^7) → (m ≥ 10^6) → is_remarkable m → n = n.succ - 1.succ + 1 := sorry

end remarkable_seven_digit_count_l493_493723


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l493_493997

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l493_493997


namespace sum_of_nonempty_subset_sums_of_1_to_16_l493_493346

noncomputable def sum_of_nonempty_subset_sums (S : Finset ℕ) : ℕ :=
  ∑ k in S.subsets \ {∅}, ∑ x in k, x

theorem sum_of_nonempty_subset_sums_of_1_to_16 :
  let S := (Finset.range 16).map Nat.succ in
  sum_of_nonempty_subset_sums S = 4456448 :=
by
  let S := (Finset.range 16).map Nat.succ
  sorry

end sum_of_nonempty_subset_sums_of_1_to_16_l493_493346


namespace average_weight_of_remaining_carrots_l493_493830

/-- 
Given the conditions:
1. 20 carrots on a scale weigh 3.64 kg.
2. 4 carrots are removed.
3. The average weight of the 4 removed carrots is 190 grams.
Prove that the average weight of the remaining 16 carrots is 180 grams.
-/
theorem average_weight_of_remaining_carrots:
  (total_weight_20_carrots : ℝ) (total_weight_20_carrots = 3.64) 
  (average_weight_4_carrots : ℝ) (average_weight_4_carrots = 190)
  (converted_units : ℝ) (converted_units = 1000) :
  let total_weight_4_removed := 4 * average_weight_4_carrots,
      total_weight_20_carrots_in_grams := total_weight_20_carrots * converted_units,
      total_weight_16_remaining := total_weight_20_carrots_in_grams - total_weight_4_removed,
      average_weight_16_remaining := total_weight_16_remaining / 16
  in average_weight_16_remaining = 180 :=
begin
  sorry
end

end average_weight_of_remaining_carrots_l493_493830


namespace unique_odd_and_monotonically_decreasing_power_function_l493_493962

-- Define a predicate for a function to be odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

-- Define a predicate for a function to be monotonically decreasing over (0, +∞)
def is_monotonically_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f(x) > f(y)

-- Define the function f(x) = x^α
def power_function (α : ℝ) (x : ℝ) : ℝ :=
  x^α

-- Problem conditions
def α_values : set ℝ := {-2, -1, -1/2, 1/2, 1, 2}

-- Lean theorem to state the translated math proof problem
theorem unique_odd_and_monotonically_decreasing_power_function :
  ∃! α ∈ α_values, is_odd_function (power_function α) ∧ is_monotonically_decreasing_on_pos (power_function α) :=
sorry

end unique_odd_and_monotonically_decreasing_power_function_l493_493962


namespace total_students_in_school_l493_493097

theorem total_students_in_school (s : ℕ) (below_8 above_8 : ℕ) (students_8 : ℕ)
  (h1 : below_8 = 20 * s / 100) 
  (h2 : above_8 = 2 * students_8 / 3) 
  (h3 : students_8 = 48) 
  (h4 : s = students_8 + above_8 + below_8) : 
  s = 100 := 
by 
  sorry 

end total_students_in_school_l493_493097


namespace survey_support_percentage_l493_493134

def menSurveyed : ℕ := 150
def womenSurveyed : ℕ := 750
def menSupportPercentage : ℝ := 0.55
def womenSupportPercentage : ℝ := 0.85

def menSupporters : ℕ := (menSupportPercentage * menSurveyed).to_nat
def womenSupporters : ℕ := (womenSupportPercentage * womenSurveyed).to_nat

def totalSupporters : ℕ := menSupporters + womenSupporters
def totalSurveyed : ℕ := menSurveyed + womenSurveyed

def percentageSupporters : ℝ := (totalSupporters.to_real / totalSurveyed.to_real) * 100

theorem survey_support_percentage : percentageSupporters = 80 := by
  sorry

end survey_support_percentage_l493_493134


namespace sum_of_coefficients_of_a_plus_b_sum_of_coefficients_of_a_minus_b_sum_of_coefficients_even_positions_l493_493871

theorem sum_of_coefficients_of_a_plus_b (n : ℕ) : 
  ∑ i in finset.range (n + 1), (nat.choose n i) * (1 : ℕ) ^ (n - i) * (1 : ℕ) ^ i = 2 ^ n :=
by sorry

theorem sum_of_coefficients_of_a_minus_b (n : ℕ) :
  ∑ i in finset.range (n + 1), (nat.choose n i) * (1 : ℕ) ^ (n - i) * (-1 : ℕ) ^ i = 0 :=
by sorry

theorem sum_of_coefficients_even_positions (n : ℕ) : 
  ∑ i in finset.range (n/2 + 1), (nat.choose n (2 * i)) * (1 : ℕ) ^ (n - (2 * i)) * (1 : ℕ) ^ (2 * i) = 2 ^ (n - 1) :=
by sorry

end sum_of_coefficients_of_a_plus_b_sum_of_coefficients_of_a_minus_b_sum_of_coefficients_even_positions_l493_493871


namespace probability_sum_1997_roots_unity_l493_493702

theorem probability_sum_1997_roots_unity :
  let v w : ℂ := sorry in
  if v ∈ {z : ℂ | z^1997 = 1} ∧ w ∈ {z : ℂ | z^1997 = 1} ∧ v ≠ w
  then 
    let p := ((∑ (i j : ℕ) in finset.range 1997, if |zeta i + zeta j| ≥ sqrt (2 + sqrt 3) then 1 else 0) : ℕ) /
              ((1997 * (1997 - 1)) / 2) : ℚ in
    p.numerator + p.denominator = 2312047 := sorry

end probability_sum_1997_roots_unity_l493_493702


namespace negation_of_existential_l493_493763

theorem negation_of_existential (h : ∃ x_0 : ℝ, x_0^3 - x_0^2 + 1 ≤ 0) : 
  ∀ x : ℝ, x^3 - x^2 + 1 > 0 :=
sorry

end negation_of_existential_l493_493763


namespace solution_set_l493_493406

theorem solution_set (x : ℝ) : 
  (-2 * x ≤ 6) ∧ (x + 1 < 0) ↔ (-3 ≤ x) ∧ (x < -1) := by
  sorry

end solution_set_l493_493406


namespace total_fingers_folded_l493_493090

theorem total_fingers_folded (yoojung_fingers yuna_fingers : ℕ) (hyoojung : yoojung_fingers = 2) (hyuna : yuna_fingers = 5) : yoojung_fingers + yuna_fingers = 7 := 
by
  rw [hyoojung, hyuna]
  rfl

end total_fingers_folded_l493_493090


namespace find_matrix_N_l493_493577

theorem find_matrix_N :
  ∃ (N : matrix (fin 2) (fin 2) ℝ),
    (∀ (u : vector ℝ 2), (matrix.mul_vec N u) = (3 • u)) :=
begin
  use ![![3, 0], ![0, 3]],
  intro u,
  ext i,
  fin_cases i;
  simp,
  sorry
end

end find_matrix_N_l493_493577


namespace number_of_books_in_box_l493_493822

theorem number_of_books_in_box (total_weight : ℕ) (weight_per_book : ℕ) 
  (h1 : total_weight = 42) (h2 : weight_per_book = 3) : total_weight / weight_per_book = 14 :=
by sorry

end number_of_books_in_box_l493_493822


namespace uncool_family_members_l493_493774

namespace ScienceClass

variable (students : ℕ)
variable (coolDads coolMoms coolSiblings coolDadsAndMoms coolMomsAndSiblings coolDadsAndSiblings coolDadsMomsAndSiblings : ℕ)

def total_students : Prop := students = 40
def have_cool_dads : Prop := coolDads = 18
def have_cool_moms : Prop := coolMoms = 20
def have_cool_siblings : Prop := coolSiblings = 10
def have_cool_dads_and_moms : Prop := coolDadsAndMoms = 8
def have_cool_moms_and_siblings : Prop := coolMomsAndSiblings = 4
def have_cool_dads_and_siblings : Prop := coolDadsAndSiblings = 3
def have_cool_dads_moms_and_siblings : Prop := coolDadsMomsAndSiblings = 2

theorem uncool_family_members :
 (total_students ∧ have_cool_dads ∧ have_cool_moms ∧ have_cool_siblings ∧
  have_cool_dads_and_moms ∧ have_cool_moms_and_siblings ∧ have_cool_dads_and_siblings ∧
  have_cool_dads_moms_and_siblings) → 
  students - (coolDads + coolMoms + coolSiblings - coolDadsAndMoms - coolMomsAndSiblings - coolDadsAndSiblings + coolDadsMomsAndSiblings) = 5 :=
by
  intros h
  sorry -- proof will go here

end ScienceClass

end uncool_family_members_l493_493774


namespace jerome_problem_odd_jerome_problem_even_l493_493818

theorem jerome_problem_odd (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ (s1 s2 : ℕ), s1 ≠ s2 → intersects s1 s2) 
  (h3 : ∀ (s1 s2 s3 : ℕ), (intersects s1 s2) → (intersects s1 s3) → 
   (intersects s2 s3) → ∃ p1 p2 p3, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
   (on_point s1 p1) ∧ (on_point s2 p2) ∧ (on_point s3 p3)) 
  (no_inters_three : ∀ (s1 s2 s3 : ℕ), n ≤ 3 → s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 → 
   ¬∃ p1 p2 p3, (on_point s1 p1 ∧ on_point s2 p2 ∧ on_point s3 p3 ∧ p1 = p2 ∧ p2 = p3)) 
  (all_steps_odd : ∀ (s : ℕ), ∀ (i < n), ∀ (intersection : point), 
    n % 2 = 1 → frog_steps s intersection i → 
    frog_steps s intersection (i + n - 1) ≠ intersection) :
  ∃ (p : point), ∀ s, finitely_jumpable p := sorry

theorem jerome_problem_even (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ (s1 s2 : ℕ), s1 ≠ s2 → intersects s1 s2) 
  (h3 : ∀ (s1 s2 s3 : ℕ), (intersects s1 s2) → (intersects s1 s3) → 
   (intersects s2 s3) → ∃ p1 p2 p3, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
   (on_point s1 p1) ∧ (on_point s2 p2) ∧ (on_point s3 p3)) 
  (no_inters_three : ∀ (s1 s2 s3 : ℕ), n ≤ 3 → s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 → 
   ¬∃ p1 p2 p3, (on_point s1 p1 ∧ on_point s2 p2 ∧ on_point s3 p3 ∧ p1 = p2 ∧ p2 = p3)) 
  (any_steps_encounter : ∀ (s : ℕ), ∀ (i < n), ∀ (intersection : point), 
    n % 2 = 0 → frog_steps s intersection i → 
    frog_steps s intersection (i + n - 1) = intersection) :
  ∀ (p : point), ¬finitely_jumpable p := sorry

end jerome_problem_odd_jerome_problem_even_l493_493818


namespace sheena_sewing_hours_weekly_l493_493007

theorem sheena_sewing_hours_weekly
  (hours_per_dress : ℕ)
  (number_of_dresses : ℕ)
  (weeks_to_complete : ℕ)
  (total_sewing_hours : ℕ)
  (hours_per_week : ℕ) :
  hours_per_dress = 12 →
  number_of_dresses = 5 →
  weeks_to_complete = 15 →
  total_sewing_hours = number_of_dresses * hours_per_dress →
  hours_per_week = total_sewing_hours / weeks_to_complete →
  hours_per_week = 4 := by
  intros h1 h2 h3 h4 h5
  sorry

end sheena_sewing_hours_weekly_l493_493007


namespace point_transform_l493_493767

def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p in (-x, y, -z)

def reflect_yz_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p in (-x, y, z)

def reflect_xz_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p in (x, -y, z)

theorem point_transform :
  let p := (1, 1, 1)
  let p1 := rotate_y_180 p
  let p2 := reflect_yz_plane p1
  let p3 := reflect_xz_plane p2
  let p4 := rotate_y_180 p3
  let p5 := reflect_xz_plane p4
  p5 = (-1, 1, 1) :=
by
  let p := (1, 1, 1)
  let p1 := rotate_y_180 p
  let p2 := reflect_yz_plane p1
  let p3 := reflect_xz_plane p2
  let p4 := rotate_y_180 p3
  let p5 := reflect_xz_plane p4
  show p5 = (-1, 1, 1)
  sorry

end point_transform_l493_493767


namespace least_possible_unit_squares_l493_493795

theorem least_possible_unit_squares (n : ℕ) (h : n = 25) : 
  ∃ k : ℕ, k ≥ (n^2 - 1) / 2 ∧ k = 312 := 
by {
  have n_pos : 0 < n := by linarith,
  use 312,
  split,
  { 
    calc 312 ≥ (n^2 - 1) / 2 : by linarith,
  },
  { 
    exact rfl,
  }};
sorry

end least_possible_unit_squares_l493_493795


namespace exactly_one_envelope_with_both_surcharges_l493_493019

-- Define the lengths and heights of the envelopes
def length_E := 7
def height_E := 5
def length_F := 10
def height_F := 4
def length_G := 8
def height_G := 8
def length_H := 12
def height_H := 4
def length_I := 13
def height_I := 3

-- Compute the ratios for each envelope
def ratio (length height : ℕ) := (length: ℚ) / (height: ℚ)

-- Compute the areas for each envelope
def area (length height : ℕ) := length * height

-- Define the surcharges conditions
def surcharge_condition (length height : ℕ) : bool :=
  (ratio length height < 1.2 ∨ ratio length height > 2.8) ||
  (area length height > 50)

-- Determine the number of envelopes incurring both charges
def count_envelopes_with_surcharge : ℕ :=
  let E_surcharge := surcharge_condition length_E height_E
  let F_surcharge := surcharge_condition length_F height_F
  let G_surcharge := surcharge_condition length_G height_G
  let H_surcharge := surcharge_condition length_H height_H
  let I_surcharge := surcharge_condition length_I height_I
  [E_surcharge, F_surcharge, G_surcharge, H_surcharge, I_surcharge].count id

theorem exactly_one_envelope_with_both_surcharges :
  count_envelopes_with_surcharge = 1 := by
  sorry

end exactly_one_envelope_with_both_surcharges_l493_493019


namespace problem1_problem2_l493_493878

noncomputable def op (a b : ℝ) := 2 * a - (3 / 2) * (a + b)

theorem problem1 (x : ℝ) (h : op x 4 = 0) : x = 12 :=
by sorry

theorem problem2 (x m : ℝ) (h : op x m = op (-2) (x + 4)) (hnn : x ≥ 0) : m ≥ 14 / 3 :=
by sorry

end problem1_problem2_l493_493878


namespace part1_part2_part3_l493_493684

section

variables (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℚ)

-- Given conditions
axiom h₀ : a 1 = 1
axiom h₁ : ∀ n ≥ 2, S n ^ 2 = a n * (S n - 1) ∧ S n ≠ 0
axiom h₂ : ∀ n : ℕ, 0 < n → (∑ i in finset.range n, b (i + 1) / S (i + 1)) = (n - 1) * 2^(n+1) + 2

-- 1. Prove that the sequence {1 / S_n} is an arithmetic sequence
theorem part1 : ∃ d, ∀ n ≥ 2, 1 / S n = 1 / S (n-1) + d := sorry

-- 2. Find the general formula for the sequence {b_n}
theorem part2 : ∀ n ≥ 2, b n = 2^n := sorry

-- 3. Prove Tₙ < 7 / 6
theorem part3 : ∀ n, T n < 7 / 6 := sorry

end

end part1_part2_part3_l493_493684


namespace find_a_l493_493987

variable (a : ℝ)

def A := ({1, 2, a} : Set ℝ)
def B := ({1, a^2 - a} : Set ℝ)

theorem find_a (h : B a ⊆ A a) : a = -1 ∨ a = 0 :=
  sorry

end find_a_l493_493987


namespace determine_remaining_sides_l493_493126

variables (A B C D E : Type)

def cyclic_quadrilateral (A B C D : Type) : Prop := sorry

def known_sides (AB CD : ℝ) : Prop := AB > 0 ∧ CD > 0

def known_ratio (m n : ℝ) : Prop := m > 0 ∧ n > 0

theorem determine_remaining_sides
  {A B C D : Type}
  (h_cyclic : cyclic_quadrilateral A B C D)
  (AB CD : ℝ) (h_sides : known_sides AB CD)
  (m n : ℝ) (h_ratio : known_ratio m n) :
  ∃ (BC AD : ℝ), BC / AD = m / n ∧ BC > 0 ∧ AD > 0 :=
sorry

end determine_remaining_sides_l493_493126


namespace probability_X_eq_Y_l493_493147

open Real

-- Define the conditions
def condition (x y : ℝ) : Prop := sin (sin x) = sin (sin y)

def in_range (x : ℝ) : Prop := -10 * π ≤ x ∧ x ≤ 10 * π

-- Define the problem:
theorem probability_X_eq_Y :
  ∀ (X Y : ℝ), (condition X Y ∧ in_range X ∧ in_range Y) →
  (probability (X = Y) = 1 / 20) :=
by
  sorry

end probability_X_eq_Y_l493_493147


namespace max_obtuse_angles_of_99_rays_l493_493326

-- Definitions
def rays (n : ℕ) := { S : set (set ℝ × set ℝ) // S.card = n }

def common_endpoint (M : set (set ℝ × set ℝ)) : Prop := 
  ∀ (r1 r2 : set ℝ × set ℝ), r1 ∈ M → r2 ∈ M → (∃ p, p ∈ r1 ∧ p ∈ r2)

def obtuse_angle_without_rays (M : set (set ℝ × set ℝ)) : Prop :=
  ∃ (r1 r2 : set ℝ × set ℝ), r1 ∈ M ∧ r2 ∈ M ∧ obtuse_angle r1 r2 ∧ 
  ∀ r ∈ M, r ≠ r1 → r ≠ r2 → ¬inside_angle r r1 r2

-- Question to be proven
theorem max_obtuse_angles_of_99_rays (M : rays 99) 
  (h_comm : common_endpoint M.val) 
  (h_obtuse : obtuse_angle_without_rays M.val) : 
  ∃ n, n = 3267 ∧ ∀ obtuse_angle (r1 r2 : set ℝ × set ℝ), r1 ∈ M.val → r2 ∈ M.val → obtuse_angle r1 r2 → n = 3267 :=
sorry

end max_obtuse_angles_of_99_rays_l493_493326


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l493_493999

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l493_493999


namespace excircle_side_formula_l493_493186

theorem excircle_side_formula 
  (a b c r_a r_b r_c : ℝ)
  (h1 : r_c = Real.sqrt (r_a * r_b)) :
  c = (a^2 + b^2) / (a + b) :=
sorry

end excircle_side_formula_l493_493186


namespace incorrect_statement_d_l493_493085

variable (x : ℝ)
variables (p q : Prop)

-- Proving D is incorrect given defined conditions
theorem incorrect_statement_d :
  ∀ (x : ℝ), (¬ (x = 1) → ¬ (x^2 - 3 * x + 2 = 0)) ∧
  ((x > 2) → (x^2 - 3 * x + 2 > 0) ∧
  (¬ (x^2 + x + 1 = 0))) ∧
  ((p ∨ q) → ¬ (p ∧ q)) :=
by
  -- A detailed proof would be required here
  sorry

end incorrect_statement_d_l493_493085


namespace median_on_AB_has_equation_l493_493990

theorem median_on_AB_has_equation :
  ∀ (A B C : ℝ × ℝ)
    (hA : A = (-2, 4))
    (hB : B = (4, -6))
    (hC : C = (5, 1)),
  let M := ((-2 + 4) / 2, (4 + (-6)) / 2),
  let k := (1 - (-1)) / (5 - 1),
  let eqn := (2 : ℝ) * λ y => y + (-1) = k * λ x => x + (-1),
  eqn = (λ x y => x - 2 * y - 3 = 0)
:= sorry

end median_on_AB_has_equation_l493_493990


namespace find_polynomial_g_l493_493710

theorem find_polynomial_g 
  (f g : Polynomial ℝ) 
  (hf : f ≠ 0) 
  (hg : g ≠ 0)
  (H : f.eval (g.eval x) = (f.eval x) * (g.eval x)) 
  (H3 : g.eval 3 = 10) :
  g = Polynomial.x^2 + Polynomial.x - 1 := 
sorry

end find_polynomial_g_l493_493710


namespace find_BP_l493_493826

theorem find_BP
  (A B C D P : Type)
  (h_circle : A B C D lie_on_circle)
  (h_intersect : segments_intersect_at AC BD P)
  (h_AP : AP = 5)
  (h_PC : PC = 3)
  (h_BD : BD = 10)
  (x : ℝ)
  (h_x_gt_5 : x > 5)
  (h_power_of_point : AP * PC = BP * PD) :
  BP = 5 + Real.sqrt 10 :=
sorry

end find_BP_l493_493826


namespace sqrt_164_between_12_and_13_l493_493827

theorem sqrt_164_between_12_and_13 : 12 < Real.sqrt 164 ∧ Real.sqrt 164 < 13 :=
sorry

end sqrt_164_between_12_and_13_l493_493827


namespace is_quadratic_equation_l493_493083

open Real

-- Define the candidate equations as statements in Lean 4
def equation_A (x : ℝ) : Prop := 3 * x^2 = 1 - 1 / (3 * x)
def equation_B (x m : ℝ) : Prop := (m - 2) * x^2 - m * x + 3 = 0
def equation_C (x : ℝ) : Prop := (x^2 - 3) * (x - 1) = 0
def equation_D (x : ℝ) : Prop := x^2 = 2

-- Prove that among the given equations, equation_D is the only quadratic equation
theorem is_quadratic_equation (x : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_A x = (a * x^2 + b * x + c = 0)) ∨
  (∃ m a b c : ℝ, a ≠ 0 ∧ equation_B x m = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_C x = (a * x^2 + b * x + c = 0)) ∨
  (∃ a b c : ℝ, a ≠ 0 ∧ equation_D x = (a * x^2 + b * x + c = 0)) := by
  sorry

end is_quadratic_equation_l493_493083


namespace evaluate_expression_l493_493183

theorem evaluate_expression :
  (\frac{(3^3 * 3^{-2})}{(3^{-1} * 3^{4})}) = \frac{1}{9} :=
by
  simp [pow_add, pow_sub]
  sorry

end evaluate_expression_l493_493183


namespace inverse_of_f_l493_493429

-- Define f(x) = 3 + 4x
def f (x : ℝ) : ℝ := 3 + 4 * x

-- Define the inverse function g(x) = (x - 3) / 4
def g (x : ℝ) : ℝ := (x - 3) / 4

-- Prove that g is the inverse of f
theorem inverse_of_f : ∀ x : ℝ, f(g(x)) = x :=
by
  -- Insert proof here
  sorry

end inverse_of_f_l493_493429


namespace rectangle_area_l493_493040

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) 
  : l * w = 1600 := 
by 
  sorry

end rectangle_area_l493_493040


namespace distance_between_parallel_lines_l493_493963

theorem distance_between_parallel_lines (a : ℝ) 
  (l1_parallel_l2 : is_parallel (line_eqn a 2 (-10)) (line_eqn 2 (a + 3) 5)) :
  distance_between_parallel_lines (line_eqn a 2 (-10)) (line_eqn 2 (a + 3) 5) = 5 * sqrt 5 / 2 :=
sorry

def line_eqn (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

def is_parallel (l1 l2 : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ l1.1 / l2.1 = k ∧ l1.2 / l2.2 = k ∧ l1.3 / l2.3 ≠ k

noncomputable def distance_between_parallel_lines (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  let (a1, b1, c1) := l1
  let (a2, b2, c2) := l2
  abs (c2 - c1) / sqrt (a1^2 + b1^2)

end distance_between_parallel_lines_l493_493963


namespace ad_length_theorem_l493_493666

noncomputable def find_ad_length (AB CD BE F A : Type) 
  [AB ⊥ CD] [midpoint F BE] [angle A = 45°] [DF = 3] [BD = 4] : Prop :=
  AD = 10

theorem ad_length_theorem 
  (AB CD BE F A : Type) 
  [AB ⊥ CD] [midpoint F BE] [angle A = 45°] [DF = 3] [BD = 4] : 
  find_ad_length AB CD BE F A :=
by sorry

end ad_length_theorem_l493_493666


namespace original_rent_is_1600_l493_493749

theorem original_rent_is_1600
  (avg_rent_before_increase : ℝ)
  (number_of_friends : ℕ)
  (avg_rent_after_increase : ℝ)
  (percentage_increase : ℝ)
  (original_total_rent : ℝ)
  (new_total_rent : ℝ) : ℝ :=
  -- Conditions
  (avg_rent_before_increase = 800) → 
  (number_of_friends = 4) → 
  (avg_rent_after_increase = 880) → 
  (percentage_increase = 0.20) →
  (original_total_rent = 4 * avg_rent_before_increase) →
  (new_total_rent = 4 * avg_rent_after_increase) →
  -- Prove
  (R : ℝ) (original_rent_of_increased_friend : ℝ) 
  (original_rent_of_increased_friend = original_total_rent + new_total_rent / percentage_increase) → 
  -- Answer
  (original_rent_of_increased_friend = 1600)
:= sorry

end original_rent_is_1600_l493_493749


namespace remaining_volume_of_cube_l493_493507

theorem remaining_volume_of_cube :
  let s := 6
  let r := 3
  let h := 6
  let V_cube := s^3
  let V_cylinder := Real.pi * (r^2) * h
  V_cube - V_cylinder = 216 - 54 * Real.pi :=
by
  sorry

end remaining_volume_of_cube_l493_493507


namespace prove_equation_l493_493841

variable {a b x1 x2 x3 : ℝ}

-- Assuming non-zero conditions
variable [NeZero x1] [NeZero x2] [NeZero x3]

-- Given conditions
def line_eq (x : ℝ) := a * x + b

/-- Definitions of intersection conditions -/
def intersection_parabola (x : ℝ) := x^2 = line_eq x

/-- Definitions of x1 and x2 as roots of the quadratic equation -/
def roots_of_quadratic := x1 + x2 = a ∧ x1 * x2 = -b

/-- Definition of x3 being the x-intercept of the line -/
def x_intercept := line_eq x3 = -b

theorem prove_equation (h1 : intersection_parabola x1) (h2 : intersection_parabola x2)
  (h3 : roots_of_quadratic) (h4 : x_intercept) :
  (1 / x1) + (1 / x2) = 1 / x3 := 
sorry

end prove_equation_l493_493841


namespace circle_properties_l493_493614

theorem circle_properties : 
  ∀ (A B : ℝ × ℝ) (C_2 : ℝ → ℝ → Prop),
    A = (0, 2) → 
    B = (2, -2) →
    C_2 = λ x y, x^2 + y^2 - 6*x - 2*y + 5 = 0 →
    (∃ (C_1 : ℝ → ℝ → Prop), 
      (C_1 = λ x y, (x - 1)^2 + y^2 = 5) ∧ 
      (∀ C D, C_2 C.1 C.2 → C_2 D.1 D.2 → 
        dist C D = sqrt 15)) := sorry

end circle_properties_l493_493614


namespace chord_length_and_segment_area_l493_493503

noncomputable def length_of_chord (r d : ℝ) : ℝ := 2 * real.sqrt (r^2 - d^2)
noncomputable def area_of_segment (r d : ℝ) : ℝ :=
  let theta := 2 * real.arcsin (d / r * real.sqrt (r^2 - d^2)) in
  r^2 * (theta - real.sin theta) / 2

theorem chord_length_and_segment_area :
  let r := 5
  let d := 4
  (length_of_chord r d = 6) ∧ (abs (area_of_segment r d - 9.6125) < 0.001) :=
by {
  let r := 5,
  let d := 4,
  suffices : length_of_chord r d = 6 ∧ abs (area_of_segment r d - 9.6125) < 0.001,
    from this,
  split,
  { -- Prove the length of chord EF is 6 inches.
    apply eq_of_h_eq,
    let rhs := 2 * real.sqrt (5^2 - 4^2),
    calc
      2 * real.sqrt (5^2 - 4^2) = ... : sorry
  },
  { -- Prove the area of the segment is approximately 9.6125 square inches.
    apply abs_sub_lt_of_le,
    calc
      area_of_segment r d = ... : sorry,
    exact sorry, -- numerical approximation check
  }
}

end chord_length_and_segment_area_l493_493503


namespace part1_part2_l493_493216

noncomputable def S (n : ℕ) : ℕ := 3 * n ^ 2 + 8 * n
noncomputable def a_nat (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)
noncomputable def a (n : ℕ) : ℕ := if hn : n > 1 then 6 * n + 5 else if n = 1 then 11 else 0
noncomputable def b (n : ℕ) : ℕ := 3 * n + 1
noncomputable def c (n : ℕ) : ℕ := (a n + 1) ^ (n + 1) / (b n + 2) ^ n
noncomputable def T (n : ℕ) : ℕ := 
  6 * (List.range n).sum (λ k, (k + 2) * 2 ^ (k + 1))

theorem part1 (n : ℕ) : b n = 3 * n + 1 :=
by
  sorry

theorem part2 (n : ℕ) : T n = 3 * n * 2 ^ (n + 2) :=
by
  sorry

end part1_part2_l493_493216


namespace max_colors_in_cube_l493_493330

-- Given conditions and problem setup.
variables (n : ℕ) (h : n > 1)

-- Definition of the cube and constraints.
def isColorfulCube (colors : ℕ → ℕ → ℕ → ℕ) : Prop :=
  ∀ (x y : ℕ), 
    x < n → y < n → 
    set.equiv (set.range (λ k, colors x y k)) 
      (set.range (λ k, colors k x y)) ∧
    set.equiv (set.range (λ k, colors x y k)) 
      (set.range (λ k, colors k y x))

-- Statement: The proof problem
theorem max_colors_in_cube : ∃ (maxColors : ℕ), 
  (∀ (colors : ℕ → ℕ → ℕ → ℕ), 
    isColorfulCube n h colors → 
    maxColors ≤ {c : ℕ | ∃ x y z, colors x y z = c}.card) ∧ 
  maxColors = (n * (n + 1) * (2 * n + 1)) / 6 := 
sorry

end max_colors_in_cube_l493_493330


namespace triangle_right_angle_l493_493291

theorem triangle_right_angle (A B C : ℝ) (h1 : 0 ≤ A ∧ A ≤ π) (h2 : 0 ≤ B ∧ B ≤ π) 
  (h3 : 0 ≤ C ∧ C ≤ π) (h4 : A + B + C = π) 
  (h5 : sin A + sin B = sin C * (cos A + cos B)) : C = π / 2 :=
sorry

end triangle_right_angle_l493_493291


namespace line_through_point_with_equal_intercepts_l493_493385

theorem line_through_point_with_equal_intercepts {x y : ℝ} :
  (∀ x₁ y₁ : ℝ, (x₁, y₁) = (3, -2) → 
    (∃ a : ℝ, (x₁ / a) + (y₁ / a) = 1 ∨ y₁ = - (2 / 3) * x₁)) → 
  (∀ y, y = - (2 / 3) * 3 ↔ 2 * 3 + 3 * (-2) = 0) :=
begin
  assume h,
  sorry
end

end line_through_point_with_equal_intercepts_l493_493385


namespace work_duration_17_333_l493_493812

def work_done (rate: ℚ) (days: ℕ) : ℚ := rate * days

def combined_work_done (rate1: ℚ) (rate2: ℚ) (days: ℕ) : ℚ :=
  (rate1 + rate2) * days

def total_work_done (rate1: ℚ) (rate2: ℚ) (rate3: ℚ) (days: ℚ) : ℚ :=
  (rate1 + rate2 + rate3) * days

noncomputable def total_days_work_last (rate_p rate_q rate_r: ℚ) : ℚ :=
  have work_p := 8 * rate_p
  have work_pq := combined_work_done rate_p rate_q 4
  have remaining_work := 1 - (work_p + work_pq)
  have days_all_together := remaining_work / (rate_p + rate_q + rate_r)
  8 + 4 + days_all_together

theorem work_duration_17_333 (rate_p rate_q rate_r: ℚ) : total_days_work_last rate_p rate_q rate_r = 17.333 :=
  by 
  have hp := 1/40
  have hq := 1/24
  have hr := 1/30
  sorry -- proof omitted

end work_duration_17_333_l493_493812


namespace rectangle_area_l493_493042

noncomputable def length (w : ℝ) : ℝ := 4 * w

noncomputable def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area :
  ∀ (l w : ℝ), 
  l = length w ∧ perimeter l w = 200 → area l w = 1600 :=
by
  intros l w h
  cases h with h1 h2
  rw [length, perimeter, area] at *
  sorry

end rectangle_area_l493_493042


namespace functions_satisfy_condition_l493_493803

def f_B (x : ℝ) := - 2 / x
def f_C (x : ℝ) := x^2 + 4 * x + 3
def f_D (x : ℝ) := x - 1 / x

theorem functions_satisfy_condition (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) :
  (f_B x1 - f_B x2) / (x1 - x2) > 0 ∧
  (f_C x1 - f_C x2) / (x1 - x2) > 0 ∧
  (f_D x1 - f_D x2) / (x1 - x2) > 0 :=
by
  sorry

end functions_satisfy_condition_l493_493803


namespace max_intersection_points_l493_493204

theorem max_intersection_points (circles : List Circle) (h_coplanar : ∀ c ∈ circles, coplanar c)
    (h_four : circles.length = 4)
    (h_middle_intersect : ∃ c1 c2 ∈ circles, c1 ≠ c2 ∧ (c1 ∩ c2).nonempty)
    (h_side_intersect : ∃ c1 c2 c3 ∈ circles, c1 ≠ c3 ∧ (c1 ∩ c3).nonempty ∧ c2 ≠ c3 ∧ (c2 ∩ c3).nonempty)
    (h_non_intersecting_sides : ∃ c3 c4 ∈ circles, c3 ≠ c4 ∧ (c3 ∩ c4).empty) :
  6 ≤ number_of_points_a_line_can_touch circles ∧ number_of_points_a_line_can_touch circles ≤ 8 := by
    sorry

end max_intersection_points_l493_493204


namespace product_of_zero_multiples_is_equal_l493_493051

theorem product_of_zero_multiples_is_equal :
  (6000 * 0 = 0) ∧ (6 * 0 = 0) → (6000 * 0 = 6 * 0) :=
by sorry

end product_of_zero_multiples_is_equal_l493_493051


namespace gcd_of_polynomial_and_multiple_of_350_l493_493619

theorem gcd_of_polynomial_and_multiple_of_350 (b : ℕ) (hb : 350 ∣ b) :
  gcd (2 * b^3 + 3 * b^2 + 5 * b + 70) b = 70 := by
  sorry

end gcd_of_polynomial_and_multiple_of_350_l493_493619


namespace find_number_l493_493068

theorem find_number : ∃ x : ℤ, 3 * x = 2 * x - 7 ∧ x = -7 :=
by {
  use -7,
  split,
  {
    norm_num,
  },
  {
    norm_num,
  }
}

end find_number_l493_493068


namespace ellipse_equation_and_fixed_point_l493_493973

theorem ellipse_equation_and_fixed_point :
  ∃ (a b c : ℝ) (E : Set (ℝ × ℝ)),
  (a > b ∧ b > 0) ∧
  (b / c = Real.tan (Real.pi / 3)) ∧
  (a + b + c = 3 + Real.sqrt 3) ∧
  (a^2 = b^2 + c^2) ∧
  (E = {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1}) ∧
  (E = {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}) ∧
  let A := (2, 0) in
  (∀ (l : Set (ℝ × ℝ)), ∃ (P Q : Set (ℝ × ℝ)),
    P ≠ A ∧ Q ≠ A ∧
    P ∈ E ∧ Q ∈ E ∧
    (∀ (PQ_circle : Set (ℝ × ℝ)), PQ_circle ∈ circle P Q → A ∈ PQ_circle) →
    ∀ (fixed_point : ℝ × ℝ), fixed_point = (2 / 7, 0) → l fixed_point) :=
sorry

end ellipse_equation_and_fixed_point_l493_493973


namespace quaternary_to_decimal_l493_493550

theorem quaternary_to_decimal : 
  let n := 123 in
  (1 * 4^2 + 2 * 4^1 + 3 * 4^0) = 27 := by
  sorry

end quaternary_to_decimal_l493_493550


namespace remainder_of_2357916_div_8_l493_493431

theorem remainder_of_2357916_div_8 : (2357916 % 8) = 4 := by
  sorry

end remainder_of_2357916_div_8_l493_493431


namespace total_dividends_received_l493_493843

theorem total_dividends_received
  (investment : ℝ)
  (share_price : ℝ)
  (nominal_value : ℝ)
  (dividend_rate_year1 : ℝ)
  (dividend_rate_year2 : ℝ)
  (dividend_rate_year3 : ℝ)
  (num_shares : ℝ)
  (total_dividends : ℝ) :
  investment = 14400 →
  share_price = 120 →
  nominal_value = 100 →
  dividend_rate_year1 = 0.07 →
  dividend_rate_year2 = 0.09 →
  dividend_rate_year3 = 0.06 →
  num_shares = investment / share_price → 
  total_dividends = (dividend_rate_year1 * nominal_value * num_shares) +
                    (dividend_rate_year2 * nominal_value * num_shares) +
                    (dividend_rate_year3 * nominal_value * num_shares) →
  total_dividends = 2640 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end total_dividends_received_l493_493843


namespace mean_of_sequence_is_26_point_5_l493_493545

-- Define the arithmetic sequence starting from 7 with 40 successive terms
def arithmetic_sequence (n : ℕ) : ℕ := n + 6

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (n : ℕ) : ℕ := n * (7 + (n + 6)) / 2

-- Define the arithmetic mean of the sequence
def arithmetic_mean (sum : ℕ) (num_terms : ℕ) : ℕ := sum / num_terms

-- Prove that the arithmetic mean of the sequence of forty successive positive integers starting from 7 is 26.5
theorem mean_of_sequence_is_26_point_5 : arithmetic_mean (sum_arithmetic_sequence 40) 40 = 26.5 := by
  sorry

end mean_of_sequence_is_26_point_5_l493_493545


namespace fraction_not_covered_l493_493417

/--
Given that frame X has a diameter of 16 cm and frame Y has a diameter of 12 cm,
prove that the fraction of the surface of frame X that is not covered by frame Y is 7/16.
-/
theorem fraction_not_covered (dX dY : ℝ) (hX : dX = 16) (hY : dY = 12) : 
  let rX := dX / 2
  let rY := dY / 2
  let AX := Real.pi * rX^2
  let AY := Real.pi * rY^2
  let uncovered_area := AX - AY
  let fraction_not_covered := uncovered_area / AX
  fraction_not_covered = 7 / 16 :=
by
  sorry

end fraction_not_covered_l493_493417


namespace initial_pokemon_cards_l493_493687

theorem initial_pokemon_cards (x : ℤ) (h : x - 9 = 4) : x = 13 :=
by
  sorry

end initial_pokemon_cards_l493_493687


namespace jason_initial_cards_l493_493689

theorem jason_initial_cards (cards_given_away cards_left : ℕ) (h1 : cards_given_away = 9) (h2 : cards_left = 4) :
  cards_given_away + cards_left = 13 :=
sorry

end jason_initial_cards_l493_493689


namespace octagon_diagonal_ratio_l493_493303

theorem octagon_diagonal_ratio (P : ℝ → ℝ → Prop) (d1 d2 : ℝ) (h1 : P d1 d2) : d1 / d2 = Real.sqrt 2 / 2 :=
sorry

end octagon_diagonal_ratio_l493_493303


namespace units_digit_150_factorial_is_zero_l493_493451

theorem units_digit_150_factorial_is_zero :
  (nat.trailing_digits (nat.factorial 150) 1) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l493_493451


namespace cube_divided_by_five_tetrahedrons_l493_493430

-- Define the minimum number of tetrahedrons needed to divide a cube
def min_tetrahedrons_to_divide_cube : ℕ := 5

-- State the theorem
theorem cube_divided_by_five_tetrahedrons : min_tetrahedrons_to_divide_cube = 5 :=
by
  -- The proof is skipped, as instructed
  sorry

end cube_divided_by_five_tetrahedrons_l493_493430


namespace range_of_x_and_f_values_l493_493620

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log (x / 2) / Real.log 2) * (Real.log (Real.sqrt x / 2) / Real.log (Real.sqrt 2))

theorem range_of_x_and_f_values :
  (∀ x : ℝ, 9 ^ x - 4 * 3 ^ (x + 1) + 27 ≤ 0 → (1 ≤ x ∧ x ≤ 2)) ∧
  (∀ x : ℝ, 1 ≤ x → x ≤ 2 → ∃ a b : ℝ, 
    (f x = a → (a = 0 ∧ x = 2) ∨ (a = 2 ∧ x = 1)) ∧ 
    (f x = b → (b = 0 ∧ x = 2) ∨ (b = 2 ∧ x = 1))) :=
by sorry

end range_of_x_and_f_values_l493_493620


namespace smaller_triangle_area_correct_l493_493846

-- Define the initial conditions
def right_angle_triangle (A : ℝ) : Prop :=
  A = 34

def similar_triangle_reduction (H' H : ℝ) : Prop :=
  H' = 0.65 * H

-- Define the goal
def smaller_triangle_area (A A' : ℝ) (H' H : ℝ) (h1 : right_angle_triangle A)
  (h2 : similar_triangle_reduction H' H) : Prop :=
  A' = A * (H' / H) ^ 2

-- Prove that the area of the smaller triangle is 14.365 square inches
theorem smaller_triangle_area_correct : ∃ (A' : ℝ) (H H' : ℝ),
  right_angle_triangle 34 ∧ similar_triangle_reduction H' H ∧
  smaller_triangle_area 34 A' H' H ∧ A' = 14.365 :=
by
  let A := 34
  let H := 1 -- This value is arbitrary; we just need a placeholder
  let H' := 0.65 * H
  let A' := 34 * (0.65) ^ 2
  use [A', H, H']
  split
  · -- Proof that A is the area of the original triangle
    exact eq.refl 34
  split
  · -- Proof that H' equals 0.65 times H
    exact eq.refl H'
  split
  · -- Proof that A' equals the area computed by the ratio rule
    unfold smaller_triangle_area
    exact eq.refl A'
  -- Conclude that A' correctly matches the expected area
  exact eq.refl 14.365

end smaller_triangle_area_correct_l493_493846


namespace gcd_three_digit_palindromes_multiple_of_3_l493_493787

-- Definitions used in the proof problem
def is_digit (n : ℕ) := n ≥ 0 ∧ n < 10

def is_palindrome (n : ℕ) := ∃ a b : ℕ, is_digit a ∧ is_digit b ∧ a ≠ 0 ∧ n = 101 * a + 10 * b + a ∧ (2 * a + b) % 3 = 0

-- Main statement
theorem gcd_three_digit_palindromes_multiple_of_3 : 
  ∃ d, d = gcd (set.filter (λ n, is_palindrome n) {n : ℕ | n ≥ 100 ∧ n < 1000}) 3 :=
sorry

end gcd_three_digit_palindromes_multiple_of_3_l493_493787


namespace relation_empty_plate_action_gender_l493_493777

theorem relation_empty_plate_action_gender : 
  let a := 45
  let b := 10
  let c := 30
  let d := 15
  let n := 100
  let chi_squared := n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))
  2.706 < chi_squared ∧ chi_squared < 3.841 :=
by {
  let a := 45
  let b := 10
  let c := 30
  let d := 15
  let n := 100
  let chi_squared := n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))
  show 2.706 < chi_squared ∧ chi_squared < 3.841, from sorry
}

end relation_empty_plate_action_gender_l493_493777


namespace min_sum_of_distances_eqn_l493_493965

open Real

def parabola (x y : ℝ) : Prop := y^2 = 2 * x

def point_D : ℝ × ℝ := (2, (3 / 2) * sqrt 3)

def distance_to_y_axis (P : ℝ × ℝ) : ℝ := abs P.1

def distance (P Q : ℝ × ℝ) : ℝ := sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

noncomputable def min_distance_sum : ℝ :=
  let f : ℝ × ℝ := (1/2, 0)
  let sum_distance (P : ℝ × ℝ) := distance P point_D + distance_to_y_axis P
  let min_distance := min (sum_distance f) ((distance f point_D) + distance_to_y_axis f - 1/2)
  min_distance

theorem min_sum_of_distances_eqn :
    ∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧
    min_distance_sum = 5 / 2 :=
sorry

end min_sum_of_distances_eqn_l493_493965


namespace maintenance_building_width_l493_493065

theorem maintenance_building_width (side_length building_length : ℝ)
  (uncovered_area : ℝ) (playground_area : ℝ) (building_area : ℝ) (W : ℝ) 
  (h1 : side_length = 12) 
  (h2 : building_length = 8)
  (h3 : uncovered_area = 104)
  (h4 : playground_area = side_length * side_length)
  (h5 : playground_area - uncovered_area = building_area)
  (h6 : building_area = building_length * W) : 
  W = 5 := 
begin
  sorry
end

end maintenance_building_width_l493_493065


namespace find_m_when_circle_tangent_to_line_l493_493559

theorem find_m_when_circle_tangent_to_line 
    (m : ℝ)
    (circle_eq : (x y : ℝ) → (x - 1)^2 + (y - 1)^2 = 4 * m)
    (line_eq : (x y : ℝ) → x + y = 2 * m) :
    (m = 2 + Real.sqrt 3) ∨ (m = 2 - Real.sqrt 3) :=
sorry

end find_m_when_circle_tangent_to_line_l493_493559


namespace range_of_t_l493_493722

theorem range_of_t (x y : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : x + y > 1) (h4 : x + 1 > y) (h5 : y + 1 > x) :
    1 ≤ max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) ∧
    max (1 / x) (max (x / y) y) * min (1 / x) (min (x / y) y) < (1 + Real.sqrt 5) / 2 := 
sorry

end range_of_t_l493_493722


namespace find_n_l493_493967

noncomputable def f : ℝ → ℝ := sorry

axiom periodicity : ∀ x : ℝ, f(x + 2) = 2 - f(x)
axiom even_func : ∀ x : ℝ, f(2 - 3*x) = f(2 + 3*x)
axiom f_zero : f 0 = 0
axiom sum_eq_123 : ∃ n : ℕ, ∑ k in finset.range n, f k = 123

theorem find_n : ∃ n : ℕ, n = 122 ∧ ∑ k in finset.range n, f k = 123 :=
sorry

end find_n_l493_493967


namespace units_digit_of_150_factorial_is_zero_l493_493448

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l493_493448


namespace ab_divisible_by_6_l493_493941

theorem ab_divisible_by_6
  (n : ℕ) (a b : ℕ)
  (h1 : 2^n = 10 * a + b)
  (h2 : n > 3)
  (h3 : b < 10) :
  (a * b) % 6 = 0 :=
sorry

end ab_divisible_by_6_l493_493941


namespace sum_of_cubes_of_roots_eq_seven_l493_493201

-- Definition of the polynomial equation given in the conditions.
def poly : Polynomial ℂ := Polynomial.C (3 : ℂ) * Polynomial.X ^ 0 +
                            Polynomial.C (-1 : ℂ) * Polynomial.X ^ 1 +
                            Polynomial.C (2 : ℂ) * Polynomial.X ^ 2 +
                            Polynomial.C (1 : ℂ) * Polynomial.X ^ 3

theorem sum_of_cubes_of_roots_eq_seven :
  let roots := (Polynomial.roots poly).val,
  roots.sum (λ x, x^3) = 7 :=
by
  sorry

end sum_of_cubes_of_roots_eq_seven_l493_493201


namespace HarrysNumber_l493_493176

-- Definition for students' skipping pattern
def skippedNumbers (n : ℕ) : ℕ → ℕ :=
  λ k, 3 * k - 1

def remainingNumbers (n : ℕ) : ℕ → List ℕ :=
  λ k, List.filter (λ x, x % 3 ≠ 2) (List.range' 1 n)

theorem HarrysNumber : (remainingNumbers 500).nth 300 = some 301 :=
by 
  sorry

end HarrysNumber_l493_493176


namespace find_x_l493_493708

def diamond (a b : ℝ) : ℝ := 3 * a * b - a + b

theorem find_x : ∃ x : ℝ, diamond 3 x = 24 ∧ x = 2.7 :=
by
  sorry

end find_x_l493_493708


namespace find_integer_n_l493_493195

noncomputable def valid_cosine_angle (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 270 ∧ cos (n * Real.pi / 180) = cos (890 * Real.pi / 180)

theorem find_integer_n : ∃ n, valid_cosine_angle n ∧ n = 10 :=
by
  use 10
  split
  · simp
  · split
    · linarith
    · norm_num
      rw [Real.cos_eq_cos_iff, ←Real.to_nat 170]
      norm_num
      sorry

end find_integer_n_l493_493195


namespace acute_angle_solution_l493_493190

-- Let x be an acute angle such that 0 < x < π / 2 and the given condition holds.
theorem acute_angle_solution
  (x : ℝ) (h : 0 < x ∧ x < π / 2)
  (condition : 2 * sin x * sin x + sin x - sin (2 * x) = 3 * cos x) : 
  x = π / 3 := 
sorry

end acute_angle_solution_l493_493190


namespace light_round_trip_distance_l493_493383

theorem light_round_trip_distance :
  let distance_per_year := 5.87 * 10^12
  let years := 50
  let star_distance := 25
in 2 * (distance_per_year * years) = 5.87 * 10^14 := 
sorry

end light_round_trip_distance_l493_493383


namespace magnitude_w_is_one_l493_493697

noncomputable def z : ℂ := ((5 - 3 * complex.I) ^ 5 * (18 + 8 * complex.I) ^ 3) / (4 - complex.I)
noncomputable def w : ℂ := (conj z) / z

theorem magnitude_w_is_one : complex.abs w = 1 := by
  sorry

end magnitude_w_is_one_l493_493697


namespace probability_sum_l493_493780

theorem probability_sum (n : ℕ) (points : Finset (ℕ)) 
    (h_points_15 : points.card = 15)
    (distinct_pairs: (A B : ℕ) -> A ≠ B -> (A, B) ∈ points -> (B, A) ∈ points)
    (h_probability : ∀ (A B : ℕ), 
        A ≠ B → 
        (A ∈ points ∧ B ∈ points) → 
        let angle_AOB := 24 * (A + B) in 
        angle_AOB < 120 → (m : ℕ) (n : ℕ), 
          (m / n : ℝ) = (4 / 7 : ℝ)) : 
    m + n = 11 := 
sorry

end probability_sum_l493_493780


namespace sequence_b_properties_l493_493956

theorem sequence_b_properties (a : ℤ → ℝ) (h_inc : ∀ i, a i < a (i + 1)) :
  ∃ N, ∀ k, (k < N → b k = k) ∧ (k ≥ N → b k = N) ∨ (∀ k, b k = k) where 
  b : ℕ → ℕ := λ k, Inf { n : ℕ | ∀ i, (∑ j in finset.Ico i (i + k), a j) / a (i + k - 1) ≤ n } :=
sorry

end sequence_b_properties_l493_493956


namespace particular_proposition_correct_l493_493470

-- Define the propositions as given in the problem
def propositionA : Prop := ∀ (f : ℝ → ℝ), even_function f → symmetric_about_y_axis f
def propositionB : Prop := ∀ (P : Set ℝ), square_prism P → parallelepiped P
def propositionC : Prop := ∀ (l1 l2 : Line), (¬ intersects l1 l2) → parallel l1 l2
def propositionD : Prop := ∃ (x : ℝ), x ≥ 3

-- Define what it means to be a particular proposition (using existential quantifier)
def is_particular_proposition (P : Prop) : Prop := ∃ x, P = (∃ y, y = x)

-- The main theorem: option D is the particular proposition
theorem particular_proposition_correct : is_particular_proposition propositionD :=
sorry

end particular_proposition_correct_l493_493470


namespace area_of_triangle_perpendicular_lines_l493_493420

/-- Two perpendicular lines intersect at the point (4,10). 
    The sum of the y-intercepts of these lines is 2. 
    Prove that the area of the triangle formed by the intersection point 
    and the y-intercepts of these lines is 4. -/
theorem area_of_triangle_perpendicular_lines 
  (m1 m2 b1 b2 : ℝ)
  (h1 : m1 * m2 = -1)
  (h2 : b1 + b2 = 2)
  (h3 : (4, 10) ∈ set_of (λ (x : ℝ), 10 = m1 * 4 + b1)) 
  (h4 : (4, 10) ∈ set_of (λ (x : ℝ), 10 = m2 * 4 + b2)) : 
  let P := (0, b1), Q := (0, b2) in
  let height := 4 in
  let base := abs (b1 - b2) in
  1 / 2 * height * base = 4 :=
sorry

end area_of_triangle_perpendicular_lines_l493_493420


namespace selection_count_l493_493724

def countSelections : ℕ :=
  let voucher : ℕ := 100
  let efficiency_threshold : ℕ := 95
  let prices := [18, 30, 39]

  let outcomes := [(x, y, z) | x y z : ℕ, 18 * x + 30 * y + 39 * z = voucher, 18 * x + 30 * y + 39 * z > efficiency_threshold]

  outcomes.length

theorem selection_count : countSelections = 3 := sorry

end selection_count_l493_493724


namespace fraction_sum_simplest_l493_493396

theorem fraction_sum_simplest (a b : ℕ) (h : 0.84375 = (a : ℝ) / b) (ha : a = 27) (hb : b = 32) : a + b = 59 :=
by
  rw [ha, hb]
  norm_num
  sorry

end fraction_sum_simplest_l493_493396


namespace geometric_sequence_formula_lambda_value_lambda_eq_1_l493_493621

theorem geometric_sequence_formula :
  (∀ {a : ℕ → ℕ}, (∀ n, ∃ q, a (n+1) = q * a n) → a 2 = 2 →
    (a 4 + a 5) = 6 * a 3 →
    ∀ n, a n = 2^(n-1)) :=
by sorry

theorem lambda_value_lambda_eq_1 :
  (∀ {a : ℕ → ℕ} {λ : ℕ},
    (∀ n, ∃ q, a (n+1) = q * a n) → a 2 = 2 →
    (a 4 + a 5) = 6 * a 3 →
    (∑ i in finset.range (n + 1), (a (i + 1) - λ * a i)) = 2^n - 1 → λ = 1) :=
by sorry

end geometric_sequence_formula_lambda_value_lambda_eq_1_l493_493621


namespace percent_of_people_with_university_diploma_no_job_choice_l493_493305

noncomputable def percent_univ_dipl_no_job_choice (no_dipl_has_job : ℝ) (has_job : ℝ) (has_dipl : ℝ) : ℝ :=
  let no_job := 100 - has_job
  let has_job_and_dipl := has_job - no_dipl_has_job
  let has_dipl_no_job := has_dipl - has_job_and_dipl
  (has_dipl_no_job / no_job) * 100

theorem percent_of_people_with_university_diploma_no_job_choice :
  percent_univ_dipl_no_job_choice 12 40 43 = 25 :=
by
  rw [percent_univ_dipl_no_job_choice]
  have no_job : ℝ := 100 - 40
  have has_job_and_dipl : ℝ := 40 - 12
  have has_dipl_no_job : ℝ := 43 - has_job_and_dipl
  have result : ℝ := (has_dipl_no_job / no_job) * 100
  have expected_result : ℝ := 25
  exact congrArg (λ x, x) rfl

end percent_of_people_with_university_diploma_no_job_choice_l493_493305


namespace matrix_pow_A_50_l493_493322

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 2], ![-16, -6]]

theorem matrix_pow_A_50 :
  A ^ 50 = ![![301, 100], ![-800, -249]] :=
by
  sorry

end matrix_pow_A_50_l493_493322


namespace sunny_probability_l493_493758

/-- Probability definitions and conditions --/
def P_rain : ℝ := 0.8
def P_no_rain : ℝ := 1.0 - P_rain
def P_sunny_given_no_rain : ℝ := 0.5
def P_sunny : ℝ := P_no_rain * P_sunny_given_no_rain
def P_cloudy_given_no_rain : ℝ := 0.5
def P_cloudy : ℝ := P_no_rain * P_cloudy_given_no_rain
def P_non_sunny : ℝ := 1.0 - P_sunny
def num_days : ℕ := 3
def num_ways_to_choose_sunny_day : ℕ := nat.choose num_days 1
def prob_exactly_one_sunny_day : ℝ := num_ways_to_choose_sunny_day * (P_sunny * (P_non_sunny ^ (num_days - 1)))

/-- The main theorem, stating the probability of exactly one sunny day is 0.243 --/
theorem sunny_probability : prob_exactly_one_sunny_day = 0.243 := by
  sorry

end sunny_probability_l493_493758


namespace determine_k_of_symmetry_axis_l493_493237

theorem determine_k_of_symmetry_axis 
  (f : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = sin (2 * x) + k * cos (2 * x)) → 
  (∃ x0 : ℝ, x0 = π / 6 ∧ (∃ θ, f x0 = sqrt (1 + k^2) * sin (2 * x0 + θ) ∧ tan θ = k)) → 
  k = sqrt 3 / 3 :=
by
  intros h1 h2
  sorry

end determine_k_of_symmetry_axis_l493_493237


namespace find_ellipse_eq_l493_493590

theorem find_ellipse_eq (a b : ℝ) (h1 : (sqrt 3, -sqrt 5) ∈ set_of (λ p, (p.2 ^ 2) / a^2 + (p.1 ^ 2) / b^2 = 1))
  (h2 : a^2 - b^2 = 16)
  (h3 : ∀ x y : ℝ, (x^2) / 9 + (y^2) / 25 = 1 → abs y = 4) :
  ∀ x y : ℝ, (y^2) / 20 + (x^2) / 4 = 1 :=
sorry

end find_ellipse_eq_l493_493590


namespace arithmetic_statement_not_basic_l493_493471

-- Define the basic algorithmic statements as a set
def basic_algorithmic_statements : Set String := 
  {"Input statement", "Output statement", "Assignment statement", "Conditional statement", "Loop statement"}

-- Define the arithmetic statement
def arithmetic_statement : String := "Arithmetic statement"

-- Prove that arithmetic statement is not a basic algorithmic statement
theorem arithmetic_statement_not_basic :
  arithmetic_statement ∉ basic_algorithmic_statements :=
sorry

end arithmetic_statement_not_basic_l493_493471


namespace coin_toss_sequences_count_l493_493640

theorem coin_toss_sequences_count :
  ∃ (seq : List (List Bool)), 
    seq.length = 16 ∧
    (seq.count (== [tt, tt]) = 5) ∧
    (seq.count (== [tt, ff]) = 4) ∧
    (seq.count (== [ff, tt]) = 3) ∧
    (seq.count (== [ff, ff]) = 3) → 
    560 := 
by
  sorry

end coin_toss_sequences_count_l493_493640


namespace necessary_but_not_sufficient_l493_493206

noncomputable def α : Type := sorry  -- Placeholder for the plane α
noncomputable def β : Type := sorry  -- Placeholder for the plane β
noncomputable def m : Type := sorry  -- Placeholder for the line m 

axiom α_perpendicular_to_β : α ≠ β → Prop        -- α ⟂ β
axiom m_in_α : m → α → Prop                           -- m ∈ α
axiom m_perpendicular_to_β : m → β → Prop      -- m ⟂ β

theorem necessary_but_not_sufficient (α β m : Type) :
  α ≠ β → m ∈ α → (α ⟂ β → m ⟂ β) ∧ ¬ (m ⟂ β → α ⟂ β) := 
begin 
  intros hαβ hminα hαperpβ,
  split,
  { sorry },  -- Proof that α ⟂ β is a necessary condition for m ⟂ β
  { sorry },  -- Proof that α ⟂ β is not a sufficient condition for m ⟂ β
end

end necessary_but_not_sufficient_l493_493206


namespace badminton_game_ninth_l493_493088

-- Define the number of games played by each player and who rested in which game
variables 
  (total_games : ℕ)
  (zhao_rested : ℕ)
  (qian_played : ℕ)
  (sun_played : ℕ)

-- Set the conditions given in the problem
def conditions : Prop :=
  total_games = 9 ∧ 
  zhao_rested = 2 ∧ 
  qian_played = 8 ∧
  sun_played = 5

-- Define the proposition that needs to be proven: Xiao Zhao and Xiao Qian played in the 9th game
def played_in_9th_game : Prop :=
  "Xiao Zhao" ∈ players_in_9th_game ∧ "Xiao Qian" ∈ players_in_9th_game

-- The statement that the conditions imply the conclusion
theorem badminton_game_ninth
  (h : conditions) :
  played_in_9th_game :=
by {
  sorry
}

end badminton_game_ninth_l493_493088


namespace find_m_l493_493658

theorem find_m (m : ℝ) (h : 2 / m = (m + 1) / 3) : m = -3 := by
  sorry

end find_m_l493_493658


namespace N_satisfies_condition_l493_493580

open Matrix

def vector2d := Vector ℝ 2

def N : Matrix (Fin 2) (Fin 2) ℝ := of 2 2 ![![3, 0], ![0, 3]]

theorem N_satisfies_condition (u : vector2d) :
  (mulVec N u = 3 • u) := by
  sorry

end N_satisfies_condition_l493_493580


namespace max_length_sequence_l493_493891

noncomputable def max_length_x : ℕ := 309

theorem max_length_sequence :
  let a : ℕ → ℤ := λ n, match n with
    | 0 => 500
    | 1 => max_length_x
    | n + 2 => a n - a (n + 1)
  in
  ∀ n : ℕ, (∀ m < n, a m ≥ 0) →
    (a n < 0) →
    (309 = max_length_x) :=
by
  intro a
  intro n
  intro h_pos
  intro h_neg
  sorry

end max_length_sequence_l493_493891


namespace sum_of_solutions_eq_neg8_l493_493719

def g (x : ℝ) : ℝ := 3 * x - 2
def g_inv (x : ℝ) : ℝ := (x + 2) / 3

theorem sum_of_solutions_eq_neg8 :
  let f := λ (x : ℝ), (x + 2) / 3 = 3 * x⁻² - 2 in
  ∑ (x : ℝ) in {x | f x}, x = -8 :=
sorry

end sum_of_solutions_eq_neg8_l493_493719


namespace f_x_plus_1_l493_493251

-- Given function definition
def f (x : ℝ) := x^2

-- Statement to prove
theorem f_x_plus_1 (x : ℝ) : f (x + 1) = x^2 + 2 * x + 1 := 
by
  rw [f]
  -- This simplifies to:
  -- (x + 1)^2 = x^2 + 2 * x + 1
  sorry

end f_x_plus_1_l493_493251


namespace vacation_cost_l493_493059

theorem vacation_cost (C : ℝ) (h : C / 3 - C / 4 = 60) : C = 720 := 
by sorry

end vacation_cost_l493_493059


namespace cyclist_climbing_speed_l493_493508

theorem cyclist_climbing_speed :
  ∃ v t : ℝ, t = 400 / v ∧ 400 = v * t ∧ 400 = 2 * v * (30 - t) ∧ v = 20 :=
by
  use 20
  use 20
  split
  split
  sorry
  sorry
  split
  sorry
  rfl

end cyclist_climbing_speed_l493_493508


namespace students_prefer_windows_l493_493831

def total_students := 210
def preferred_mac := 60
def equally_preferred_both := (preferred_mac / 3)
def no_preference := 90

theorem students_prefer_windows :
  total_students - preferred_mac - equally_preferred_both - no_preference = 40 := 
by
  have eq_mac_pref: equally_preferred_both = 20 := by sorry
  calc
    total_students - preferred_mac - equally_preferred_both - no_preference
      = 210 - 60 - equally_preferred_both - 90 : by rfl
  ... = 210 - 60 - 20 - 90  : by rw eq_mac_pref
  ... = 40 : by norm_num

end students_prefer_windows_l493_493831


namespace efficiency_relationship_l493_493483

variable (Q₁₂ Q₃₄ Q₁₃ : ℝ)
variable (η₀ η₁ η₂ α : ℝ)

-- Define the given conditions
def efficiency_Lomonosov : Prop := η₀ = 1 - (Q₃₄ / Q₁₂)
def efficiency_Avogadro : Prop := η₁ = (1 - 0.01 * α) * η₀
def efficiency_Boltzmann : Prop := η₂ = (η₀ - η₁) / (1 - η₁)
def efficiency_bounds : Prop := η₁ < η₀ ∧ η₀ < 1

-- Target proposition that we need to prove
theorem efficiency_relationship (h₀ : efficiency_Lomonosov) (h₁ : efficiency_Avogadro) (h₂ : efficiency_Boltzmann) (h₃ : efficiency_bounds) :
  η₂ = α / (100 - (100 - α) * η₀) := sorry

end efficiency_relationship_l493_493483


namespace smallest_domain_of_ffx_l493_493285

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5)

theorem smallest_domain_of_ffx :
  ∃ x : ℝ, x >= 30 ∧ (∀ y : ℝ, y >= 30 → y ∈ {x | ∃ x, f(f x) = f(f y)}) :=
sorry

end smallest_domain_of_ffx_l493_493285


namespace anna_clara_age_l493_493866

theorem anna_clara_age :
  ∃ x : ℕ, (54 - x) * 3 = 80 - x ∧ x = 41 :=
by
  sorry

end anna_clara_age_l493_493866


namespace growingPathProduct_l493_493165

def distance (p1 p2 : ℕ × ℕ) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

def isGrowingPath (path : List (ℕ × ℕ)) : Prop :=
  ∀ i j, i < j → distance (path.nthLe i (by sorry)) (path.nthLe j (by sorry)) >
         distance (path.nthLe (i - 1) (by sorry)) (path.nthLe i (by sorry))

def maximumGrowingPathLength (gridSize : ℕ) : ℕ := sorry

def numberOfMaximumGrowingPaths (gridSize : ℕ) (maxLength : ℕ) : ℕ := sorry

theorem growingPathProduct (gridSize : ℕ := 5) :
  let m := maximumGrowingPathLength gridSize
  let r := numberOfMaximumGrowingPaths gridSize m
  ∃ m_max r_paths, 
    m = m_max ∧ r = r_paths ∧
    r_paths = numberOfMaximumGrowingPaths gridSize m_max →
    m_max * r_paths = sorry 
:= by
  sorry

end growingPathProduct_l493_493165


namespace area_within_C2_not_intersecting_tangency_is_π_over_4_l493_493601

theorem area_within_C2_not_intersecting_tangency_is_π_over_4 :
  let C1 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4},
      C2 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1},
      points_tangency (P : ℝ × ℝ) : set (ℝ × ℝ) := { Q | ∃ θ : ℝ, 2 * Q.1 * cos θ + 2 * Q.2 * sin θ = 1 }
  in 
    ∀ P ∈ C1,
    let non_intersecting_area := {Q | Q ∈ C2 ∧ ∀ θ : ℝ, 2 * Q.1 * cos θ + 2 * Q.2 * sin θ ≠ 1} in
    ∀ Q ∈ non_intersecting_area, 
    (λ (R : ℝ × ℝ), R.1^2 + R.2^2 < (1 / 2)^2) Q → ∃ area, area = π / 4 :=
by
  sorry

end area_within_C2_not_intersecting_tangency_is_π_over_4_l493_493601


namespace route_Y_quicker_than_route_X_l493_493352

theorem route_Y_quicker_than_route_X :
    let dist_X := 9  -- distance of Route X in miles
    let speed_X := 45  -- speed of Route X in miles per hour
    let dist_Y := 8  -- total distance of Route Y in miles
    let normal_dist_Y := 6.5  -- normal speed distance of Route Y in miles
    let construction_dist_Y := 1.5  -- construction zone distance of Route Y in miles
    let normal_speed_Y := 50  -- normal speed of Route Y in miles per hour
    let construction_speed_Y := 25  -- construction zone speed of Route Y in miles per hour
    let time_X := (dist_X / speed_X) * 60  -- time for Route X in minutes
    let time_Y1 := (normal_dist_Y / normal_speed_Y) * 60  -- time for normal speed segment of Route Y in minutes
    let time_Y2 := (construction_dist_Y / construction_speed_Y) * 60  -- time for construction zone segment of Route Y in minutes
    let time_Y := time_Y1 + time_Y2  -- total time for Route Y in minutes
    time_X - time_Y = 0.6 :=  -- the difference in time between Route X and Route Y in minutes
by
  sorry

end route_Y_quicker_than_route_X_l493_493352


namespace johns_donation_l493_493095

theorem johns_donation (n : ℕ) (A : ℝ) (J : ℝ) 
  (h1 : ∀ n, n = 1 -> A = 75 / 1.5) 
  (h2 : ∀ n, n = 1 -> J = 75 * (n + 1) - A * n)
  (h3 : n = 1) : J = 100 :=
by
  have hA : A = 50 := by rw [h1, h3]; norm_num
  rw [h2] at h3
  norm_num [hA, h3]
  sorry

end johns_donation_l493_493095


namespace parabola_focus_distance_l493_493985

theorem parabola_focus_distance (p m : ℝ) (hp : p > 0)
  (P_on_parabola : m^2 = 2 * p)
  (PF_dist : (1 + p / 2) = 3) : p = 4 := 
  sorry

end parabola_focus_distance_l493_493985


namespace line_parameterization_l493_493044

theorem line_parameterization (s m : ℝ) :
  (∃ t : ℝ, ∀ x y : ℝ, (x = s + 2 * t ∧ y = 3 + m * t) ↔ y = 5 * x - 7) →
  s = 2 ∧ m = 10 :=
by
  intro h_conditions
  sorry

end line_parameterization_l493_493044


namespace units_digit_of_150_factorial_is_zero_l493_493462

theorem units_digit_of_150_factorial_is_zero : 
  ∃ k : ℕ, (150! = k * 10) :=
begin
  -- We need to prove that there exists a natural number k such that 150! is equal to k times 10
  sorry
end

end units_digit_of_150_factorial_is_zero_l493_493462


namespace find_A_range_f_l493_493270

variables {A : ℝ} (x : ℝ)

-- Defining the vectors and given conditions
def vec_m := (sin A, cos A)
def vec_n := (sqrt 3, -1 : ℝ)

-- Given conditions
def dot_product_eq : (vec_m.1 * vec_n.1 + vec_m.2 * vec_n.2 = 1) :=
  sqrt 3 * sin A - cos A = 1

def A_acute := 0 < A ∧ A < (π / 2)

-- Result 1: Finding the value of A
theorem find_A (h1 : dot_product_eq) (h2 : A_acute) : A = π / 3 :=
sorry

-- Result 2: Finding the range of the function f(x)
def f := cos (2 * x) + 4 * cos A * sin x

theorem range_f (h1 : dot_product_eq) (hx : x ∈ ℝ) : 
  ∀ x, -3 ≤ f x ∧ f x ≤ 3 / 2 :=
sorry

end find_A_range_f_l493_493270


namespace root_sum_value_l493_493714

theorem root_sum_value (r s t : ℝ) (h1: r + s + t = 24) (h2: r * s + s * t + t * r = 50) (h3: r * s * t = 24) :
  r / (1/r + s * t) + s / (1/s + t * r) + t / (1/t + r * s) = 19.04 :=
sorry

end root_sum_value_l493_493714


namespace units_digit_of_150_factorial_is_zero_l493_493433

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l493_493433


namespace triangle_area_ABC_l493_493191

theorem triangle_area_ABC :
  let A := (0, 0)
  let B := (1424233, 2848467)
  let C := (1424234, 2848469)
  area_of_triangle A B C = 0.50 :=
by
  -- Proof goes here
  sorry

end triangle_area_ABC_l493_493191


namespace find_k_l493_493759

theorem find_k (k l : ℝ) (C : ℝ × ℝ) (OC : ℝ) (A B D : ℝ × ℝ)
  (hC_coords : C = (0, 3))
  (hl_val : l = 3)
  (line_eqn : ∀ x, y = k * x + l)
  (intersect_eqn : ∀ x, y = 1 / x)
  (hA_coords : A = (1 / 6, 6))
  (hD_coords : D = (1 / 6, 6))
  (dist_ABC : dist A B = dist B C)
  (dist_BCD : dist B C = dist C D)
  (OC_val : OC = 3) :
  k = 18 := 
sorry

end find_k_l493_493759


namespace transformations_count_to_original_position_l493_493707

def vertex1 := (0, 0)
def vertex2 := (6, 0)
def vertex3 := (0, 4)

inductive Transformation
| Rotation90
| Rotation180
| Rotation270
| ReflectX
| ReflectY

open Transformation

def applyTransformation : Transformation → (ℤ × ℤ) → (ℤ × ℤ)
| Rotation90, (x, y)   => (-y, x)
| Rotation180, (x, y)  => (-x, -y)
| Rotation270, (x, y)  => (y, -x)
| ReflectX, (x, y)     => (x, -y)
| ReflectY, (x, y)     => (-x, y)

def applySequence (seq : List Transformation) (p : ℤ × ℤ) : (ℤ × ℤ) :=
  seq.foldl (fun acc t => applyTransformation t acc) p

def triangleTransformed (seq : List Transformation) : List (ℤ × ℤ) :=
  [applySequence seq vertex1, applySequence seq vertex2, applySequence seq vertex3]

def returnsToOriginal (seq : List Transformation) : Prop :=
  triangleTransformed seq = [vertex1, vertex2, vertex3]

def sequences : List (List Transformation) :=
  List.replicateM 3 [Rotation90, Rotation180, Rotation270, ReflectX, ReflectY]

def countReturningSequences : ℕ :=
  (sequences.filter returnsToOriginal).length

theorem transformations_count_to_original_position : countReturningSequences = 7 := 
by sorry

end transformations_count_to_original_position_l493_493707


namespace smallest_possible_elements_l493_493341

-- Let S be a set, and let there be a sequence of mutually distinct nonempty subsets X1, X2, ..., X100 of S
-- such that for all i in {1, ..., 99}, Xi and Xi+1 are disjoint and their union is not the whole set S.
-- Prove the smallest possible number of elements in S is 8.
theorem smallest_possible_elements {S : Type} (X : ℕ → set S) :
  (∀ i, 1 ≤ i ∧ i ≤ 99 → X i ≠ ∅ ∧ X (i + 1) ≠ ∅ ∧ X i ≠ X (i + 1) ∧
              (X i ∩ X (i + 1) = ∅) ∧ (X i ∪ X (i + 1) ≠ set.univ)) →
  (nat.card S ≥ 8) :=
sorry

end smallest_possible_elements_l493_493341


namespace find_pairs_l493_493329

theorem find_pairs (a b : ℕ) (h1: a > 0) (h2: b > 0) (q r : ℕ)
  (h3: a^2 + b^2 = q * (a + b) + r) (h4: q^2 + r = 1977) : 
  (a = 50 ∧ b = 37) ∨ (a = 37 ∧ b = 50) :=
sorry

end find_pairs_l493_493329


namespace identify_knight_and_liar_l493_493819

-- Definitions and assumptions
def Knight (p : Prop) : Prop := p
def Liar (p : Prop) : Prop := ¬p

variables A B : Prop

-- Condition: A's statement "If B is a knight, then I am a liar"
variable h : Knight B → Liar A

-- Theorem: Determine the identities of A and B
theorem identify_knight_and_liar : Knight A ∧ Liar B :=
by
  -- Proof would go here
  sorry

end identify_knight_and_liar_l493_493819


namespace train_speed_l493_493144

/-- Given: 
1. A train travels a distance of 80 km in 40 minutes. 
2. We need to prove that the speed of the train is 120 km/h.
-/
theorem train_speed (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ) (speed : ℝ) 
  (h_distance : distance = 80) 
  (h_time_minutes : time_minutes = 40) 
  (h_time_hours : time_hours = 40 / 60) 
  (h_speed : speed = distance / time_hours) : 
  speed = 120 :=
sorry

end train_speed_l493_493144


namespace number_of_zeros_of_quadratic_function_l493_493049

-- Given the quadratic function y = x^2 + x - 1
def quadratic_function (x : ℝ) : ℝ := x^2 + x - 1

-- Prove that the number of zeros of the quadratic function y = x^2 + x - 1 is 2
theorem number_of_zeros_of_quadratic_function : 
  ∃ x1 x2 : ℝ, quadratic_function x1 = 0 ∧ quadratic_function x2 = 0 ∧ x1 ≠ x2 :=
by
  sorry

end number_of_zeros_of_quadratic_function_l493_493049


namespace sum_of_c_values_l493_493591

def is_rational (x : ℚ) := true

def quadratic_has_two_rational_roots (a b c : ℤ) : Prop :=
  let Δ := b^2 - 4 * a * c in
  ∃ x₁ x₂ : ℚ, is_rational x₁ ∧ is_rational x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂

theorem sum_of_c_values : 
  (∑ c in {c : ℤ | c ≤ 40 ∧ quadratic_has_two_rational_roots 1 (-9) (-c^2)}, c) = 0 :=
sorry

end sum_of_c_values_l493_493591


namespace race_heartbeats_l493_493149

def heart_rate : ℕ := 140
def pace : ℕ := 6
def distance : ℕ := 30
def total_time (pace distance : ℕ) : ℕ := pace * distance
def total_heartbeats (heart_rate total_time : ℕ) : ℕ := heart_rate * total_time pace distance

theorem race_heartbeats : total_heartbeats heart_rate (total_time pace distance) = 25200 := by
  sorry

end race_heartbeats_l493_493149


namespace classics_books_l493_493393

theorem classics_books (authors : ℕ) (books_per_author : ℕ) (h_authors : authors = 6) (h_books_per_author : books_per_author = 33) :
  authors * books_per_author = 198 := 
by { rw [h_authors, h_books_per_author], norm_num }

end classics_books_l493_493393


namespace regular_octahedron_axes_count_l493_493338

def is_rotation_symmetry (O : Type) (axis : O → O → Prop) : Prop := 
  ∃ (rotation : ℝ), abs rotation ≤ 180 ∧ O = rotation • O

theorem regular_octahedron_axes_count 
  (O : Type) [regular_octahedron O] :
  (number_of_rotation_symmetrically_axes O 180) = 13 := 
sorry

end regular_octahedron_axes_count_l493_493338


namespace odd_function_period_pi_l493_493876

def f (x : ℝ) : ℝ := (Real.sin (x + Real.pi / 4))^2 - (Real.cos (x + Real.pi / 4))^2

theorem odd_function_period_pi (x : ℝ) : 
    (f(-x) = -f(x)) ∧ (∀p > 0, (∀y, f (y + p) = f y) → p ≥ Real.pi / 2) := 
by 
    sorry

end odd_function_period_pi_l493_493876


namespace shoe_combination_l493_493153

theorem shoe_combination (friends : ℕ) (sizes : ℕ) (designs : ℕ) (colors : ℕ) (choose_colors : ℕ) :
  friends = 35 →
  sizes = 11 →
  designs = 20 →
  colors = 4 →
  choose_colors = 3 →
  (nat.choose colors choose_colors) * designs * friends = 2800 :=
by
  intros h_friends h_sizes h_designs h_colors h_choose_colors
  have : nat.choose 4 3 = 4 := by sorry
  rw [h_friends, h_designs, this]
  norm_num

end shoe_combination_l493_493153


namespace range_of_phi_midpoint_trajectory_l493_493262

noncomputable theory

-- Definitions for problem conditions
def parametric_equation_of_line (t φ : ℝ) : ℝ × ℝ :=
  (t * cos φ, -2 + t * sin φ)

def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Questions transformed into statements in Lean

-- 1. Proving the range of φ
theorem range_of_phi (φ : ℝ) (hφ : 0 ≤ φ ∧ φ < π) :
  (|sin φ| > √3/2) ↔ (φ > π/3 ∧ φ < 2*π/3) :=
sorry

-- 2. Parametric equation for the midpoint
theorem midpoint_trajectory (φ : ℝ) (hφ : φ > π/3 ∧ φ < 2*π/3) :
  ∃ t1 t2 : ℝ, 
  let (x, y) := (parametric_equation_of_line t1 φ, parametric_equation_of_line t2 φ) in
  (x + t1*cos φ / 2 = sin 2φ) ∧
  (y + (-2 + t2*sin φ) / 2 = -1 - cos 2φ) :=
sorry

end range_of_phi_midpoint_trajectory_l493_493262


namespace length_DE_proof_l493_493179

open EuclideanGeometry
noncomputable theory

def equilateral_triangle (A B C : Point) (s : ℝ) : Prop :=
equilateral A B C ∧ dist A B = s

def circle_through_two_points (ω : Circle) (A B : Point) : Prop :=
on ω A ∧ on ω B

def tangents_to_circle (C : Point) (A B : Point) (ω : Circle) : Prop :=
tangent (Line.mk A C) ω ∧ tangent (Line.mk B C) ω

def point_on_circle_with_distance (D : Point) (C : Point) (ω : Circle) (dist_CD : ℝ) : Prop :=
on ω D ∧ dist C D = dist_CD

def intersection_of_line_and_segment (C D : Point) (A B : Segment) : Point :=
sorry -- Intersection defined, proof skipped.

-- The main theorem:
theorem length_DE_proof (A B C D E : Point) (ω : Circle) :
  equilateral_triangle A B C 6 →
  circle_through_two_points ω A B →
  tangents_to_circle C A B ω →
  point_on_circle_with_distance D C ω 4 →
  intersection_of_line_and_segment C D (Segment.mk A B) = E →
  dist D E = 20 / 13 :=
by
  sorry -- Proof skipped.

end length_DE_proof_l493_493179


namespace find_smallest_n_l493_493606

theorem find_smallest_n (i : ℂ) (h_i : i = complex.I) : 
  ∃ n : ℕ, 0 < n ∧ (1 + i)^n ∈ ℝ ∧ ∀ m : ℕ, 0 < m ∧ (1 + i)^m ∈ ℝ → m ≥ 4 :=
by
  sorry

end find_smallest_n_l493_493606


namespace gcd_binom_integer_l493_493366

open Nat

theorem gcd_binom_integer (n m : ℤ) (hnm : n ≥ m)
  (hm1 : m ≥ 1) :
  (gcd m n : ℤ) / n * (nat.binomial (n.nat_abs) (m.nat_abs)) ∈ ℤ := 
sorry

end gcd_binom_integer_l493_493366


namespace efficiency_relationship_l493_493482

variable (Q₁₂ Q₃₄ Q₁₃ : ℝ)
variable (η₀ η₁ η₂ α : ℝ)

-- Define the given conditions
def efficiency_Lomonosov : Prop := η₀ = 1 - (Q₃₄ / Q₁₂)
def efficiency_Avogadro : Prop := η₁ = (1 - 0.01 * α) * η₀
def efficiency_Boltzmann : Prop := η₂ = (η₀ - η₁) / (1 - η₁)
def efficiency_bounds : Prop := η₁ < η₀ ∧ η₀ < 1

-- Target proposition that we need to prove
theorem efficiency_relationship (h₀ : efficiency_Lomonosov) (h₁ : efficiency_Avogadro) (h₂ : efficiency_Boltzmann) (h₃ : efficiency_bounds) :
  η₂ = α / (100 - (100 - α) * η₀) := sorry

end efficiency_relationship_l493_493482


namespace ratio_a3_a2_l493_493657

theorem ratio_a3_a2 (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℝ)
  (h : (1 - 2 * x)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  a_3 / a_2 = -2 :=
sorry

end ratio_a3_a2_l493_493657


namespace subset_condition_for_A_B_l493_493642

open Set

theorem subset_condition_for_A_B {a : ℝ} (A B : Set ℝ) 
  (hA : A = {x | abs (x - 2) < a}) 
  (hB : B = {x | x^2 - 2 * x - 3 < 0}) :
  B ⊆ A ↔ 3 ≤ a :=
  sorry

end subset_condition_for_A_B_l493_493642


namespace remainder_zero_l493_493589

theorem remainder_zero (x : Polynomial ℤ) :
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 0 :=
sorry

end remainder_zero_l493_493589


namespace prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l493_493988

theorem prime_in_form_x_squared_plus_16y_squared (p : ℕ) (hprime : Prime p) (h1 : p % 8 = 1) :
  ∃ x y : ℤ, p = x^2 + 16 * y^2 :=
by
  sorry

theorem prime_in_form_4x_squared_plus_4xy_plus_5y_squared (p : ℕ) (hprime : Prime p) (h5 : p % 8 = 5) :
  ∃ x y : ℤ, p = 4 * x^2 + 4 * x * y + 5 * y^2 :=
by
  sorry

end prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l493_493988


namespace correct_statement_is_D_l493_493086

-- Definitions based on conditions
def virusesNotLifeSystem : Prop := ¬ LifeSystem Virus
def dnaLigaseSpecific : Prop := Specific DNA_Ligase
def chromosomesToChromatinByDNAUnwinding : Prop := ∀ endOfMitosis, ChromosomesToChromatinBy DNAUnwinding endOfMitosis
def heritableVariationHaploidCornTriploidWatermelons : Prop := HeritableVariation HaploidCorn ∧ HeritableVariation TriploidWatermelons

-- Proof problem
theorem correct_statement_is_D : 
  ¬ virusesNotLifeSystem ∧ ¬ dnaLigaseSpecific ∧ ¬ chromosomesToChromatinByDNAUnwinding ∧ heritableVariationHaploidCornTriploidWatermelons → 
  (¬ virusesNotLifeSystem = false) ∧ (¬ dnaLigaseSpecific = false) ∧ (¬ chromosomesToChromatinByDNAUnwinding = false) ∧ (heritableVariationHaploidCornTriploidWatermelons = true) → true
:=
by sorry

end correct_statement_is_D_l493_493086


namespace complex_integral_cosh_l493_493159

theorem complex_integral_cosh (R : ℝ) (hR : R = 2) :
  ∮ (|z|=R) (λ z, (cosh (I * z)) / (z^2 + 4 * z + 3)) = real.pi * I * real.cos 1 :=
by
  sorry

end complex_integral_cosh_l493_493159


namespace line_of_intersection_in_standard_form_l493_493821

noncomputable def plane1 (x y z : ℝ) := 3 * x + 4 * y - 2 * z = 5
noncomputable def plane2 (x y z : ℝ) := 2 * x + 3 * y - z = 3

theorem line_of_intersection_in_standard_form :
  (∃ x y z : ℝ, plane1 x y z ∧ plane2 x y z ∧ (∀ t : ℝ, (x, y, z) = 
  (3 + 2 * t, -1 - t, t))) :=
by {
  sorry
}

end line_of_intersection_in_standard_form_l493_493821


namespace problem_statement_l493_493643

variables (α : ℝ) (a b : ℝ)

def p := (sin α = 1/2 → α = π/6)
def q := (a > b → 1/a < 1/b)

theorem problem_statement : ¬ (p ∨ q) :=
by
  -- The proof goes here
  sorry

end problem_statement_l493_493643


namespace irene_to_becky_age_ratio_l493_493887

theorem irene_to_becky_age_ratio 
(Eddie_age : ℕ) (h1 : Eddie_age = 92)
(Becky_age : ℕ) (h2 : Becky_age * 4 = Eddie_age)
(Irene_age : ℕ) (h3 : Irene_age = 46) :
Irene_age / Becky_age = 2 :=
by
  rw [h1, h3] at h2
  sorry

end irene_to_becky_age_ratio_l493_493887


namespace distance_between_rhombus_centers_l493_493863

theorem distance_between_rhombus_centers 
  (a α : ℝ) 
  (hα : α < 90) : 
  let PQ := a * real.sqrt (1 + (real.sqrt 3 / 2) * real.sin α) in 
  PQ = a * real.sqrt(1 + (real.sqrt 3 / 2) * real.sin α) :=
by
  sorry

end distance_between_rhombus_centers_l493_493863


namespace number_of_questions_in_exam_l493_493297

theorem number_of_questions_in_exam :
  ∀ (typeA : ℕ) (typeB : ℕ) (timeA : ℝ) (timeB : ℝ) (totalTime : ℝ),
    typeA = 100 →
    timeA = 1.2 →
    timeB = 0.6 →
    totalTime = 180 →
    120 = typeA * timeA →
    totalTime - 120 = typeB * timeB →
    typeA + typeB = 200 :=
by
  intros typeA typeB timeA timeB totalTime h_typeA h_timeA h_timeB h_totalTime h_timeA_calc h_remaining_time
  sorry

end number_of_questions_in_exam_l493_493297


namespace modulus_of_complex_l493_493242

  open Complex

  theorem modulus_of_complex (z : ℂ) (h : (1 - I) * z = 4 * I) : |z| = 2 * Real.sqrt 2 :=
  by
    sorry
  
end modulus_of_complex_l493_493242


namespace total_spent_l493_493317

theorem total_spent (puppy_cost dog_food_cost treats_cost_per_bag toys_cost crate_cost bed_cost collar_leash_cost bags_of_treats discount_rate : ℝ) :
  puppy_cost = 20 →
  dog_food_cost = 20 →
  treats_cost_per_bag = 2.5 →
  toys_cost = 15 →
  crate_cost = 20 →
  bed_cost = 20 →
  collar_leash_cost = 15 →
  bags_of_treats = 2 →
  discount_rate = 0.2 →
  (dog_food_cost + treats_cost_per_bag * bags_of_treats + toys_cost + crate_cost + bed_cost + collar_leash_cost) * (1 - discount_rate) + puppy_cost = 96 :=
by sorry

end total_spent_l493_493317


namespace rain_at_least_one_day_l493_493594

-- Define the probabilities
def P_A1 : ℝ := 0.30
def P_A2 : ℝ := 0.40
def P_A2_given_A1 : ℝ := 0.70

-- Define complementary probabilities
def P_not_A1 : ℝ := 1 - P_A1
def P_not_A2 : ℝ := 1 - P_A2
def P_not_A2_given_A1 : ℝ := 1 - P_A2_given_A1

-- Calculate probabilities of no rain on both days under different conditions
def P_no_rain_both_days_if_no_rain_first : ℝ := P_not_A1 * P_not_A2
def P_no_rain_both_days_if_rain_first : ℝ := P_A1 * P_not_A2_given_A1

-- Total probability of no rain on both days
def P_no_rain_both_days : ℝ := P_no_rain_both_days_if_no_rain_first + P_no_rain_both_days_if_rain_first

-- Probability of rain on at least one of the two days
def P_rain_one_or_more_days : ℝ := 1 - P_no_rain_both_days

-- Expressing the result as a percentage
def result_percentage : ℝ := P_rain_one_or_more_days * 100

-- Theorem statement
theorem rain_at_least_one_day : result_percentage = 49 := by
  -- We skip the proof
  sorry

end rain_at_least_one_day_l493_493594


namespace zeroes_in_base_81_l493_493277

-- Definitions based on the conditions:
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question: How many zeroes does 15! end with in base 81?
-- Lean 4 proof statement:
theorem zeroes_in_base_81 (n : ℕ) : n = 15 → Nat.factorial n = 
  (81 : ℕ) ^ k * m → k = 1 :=
by
  sorry

end zeroes_in_base_81_l493_493277


namespace frog_reaches_vertical_side_l493_493839

def P (x y : ℕ) : ℝ := 
  if (x = 3 ∧ y = 3) then 0 -- blocked cell
  else if (x = 0 ∨ x = 5) then 1 -- vertical boundary
  else if (y = 0 ∨ y = 5) then 0 -- horizontal boundary
  else sorry -- inner probabilities to be calculated

theorem frog_reaches_vertical_side : P 2 2 = 5 / 8 :=
by sorry

end frog_reaches_vertical_side_l493_493839


namespace correct_statements_are_B_and_C_l493_493644

def vec2D := ℝ × ℝ

def magnitude (v : vec2D) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v1 v2 : vec2D) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def angle_cosine (v1 v2 : vec2D) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

def is_perpendicular (v1 v2 : vec2D) : Prop :=
  dot_product v1 v2 = 0

variable (a b : vec2D)
theorem correct_statements_are_B_and_C :
  a = (2,0) ∧ b = (1,1) →
  (real.arccos (angle_cosine a b) = π / 4 ∧ 
   is_perpendicular (a - b) b) :=
by
  sorry

end correct_statements_are_B_and_C_l493_493644


namespace impossible_grouping_l493_493103

/-- Given a set of stones weighing 1 to 77 grams, prove that for none 
    of the values k in {9, 10, 11, 12} can the stones be divided into k groups such that the total weights 
    of each group are different and each group contains fewer stones than groups with smaller total weights. -/
theorem impossible_grouping (k : ℕ) (h_k : k ∈ {9, 10, 11, 12}) :
  ¬ ∃ (groups : fin k → finset ℕ), 
    (∀ i j : fin k, i ≠ j → (groups i).sum ≠ (groups j).sum) ∧
    (∀ i : fin k, ∃ m : ℕ, 1 ≤ groups i.card ∧ groups i.card ≤ m ∧ groups i.card < groups (i.succ % k).card) :=
sorry

end impossible_grouping_l493_493103


namespace sum_of_distinct_factorials_l493_493274

/-- 
  Prove the number of positive integers less than or equal to 240 that can be expressed 
  as a sum of distinct factorials is 39
-/
theorem sum_of_distinct_factorials: 
  (finset.filter (fun n => ∃ (a b c d e f : ℕ), a ≤ 1 ∧ b ≤ 1 ∧ c ≤ 1 ∧ d ≤ 1 ∧ e ≤ 1 ∧ f ≤ 1 ∧
    n = (a! + b! + c! + d! + e! + f!) ∧ n ≤ 240 ∧ n > 0) 
    (finset.range 241)).card = 39 :=
by
  sorry

end sum_of_distinct_factorials_l493_493274


namespace problem_statement_l493_493342

theorem problem_statement (a b c : ℝ) (h : a^6 + b^6 + c^6 = 3) : a^7 * b^2 + b^7 * c^2 + c^7 * a^2 ≤ 3 := 
  sorry

end problem_statement_l493_493342


namespace incorrect_equation_a_neq_b_l493_493796

theorem incorrect_equation_a_neq_b (a b : ℝ) (h : a ≠ b) : a - b ≠ b - a :=
  sorry

end incorrect_equation_a_neq_b_l493_493796


namespace prove_money_given_l493_493161

-- Declaring the initial state and given conditions.
variable (money_left : ℕ) (bread_loaves : ℕ) (milk_cartons : ℕ) (cost_bread : ℕ) (cost_milk : ℕ)
variable (money_given : ℕ)

-- Assigning the values given in the problem.
def problem_conditions : Prop :=
  bread_loaves = 4 ∧
  milk_cartons = 2 ∧
  cost_bread = 2 ∧
  cost_milk = 2 ∧
  money_left = 35

-- Defining the cost calculations based on the given conditions.
def total_spent (bread_loaves cost_bread milk_cartons cost_milk : ℕ) : ℕ :=
  (bread_loaves * cost_bread) + (milk_cartons * cost_milk)

-- The statement to be proved, which should match the given mathematical conditions.
theorem prove_money_given :
  problem_conditions →
  money_given = total_spent bread_loaves cost_bread milk_cartons cost_milk + money_left →
  money_given = 47 :=
by
  intros h1 h2
  sorry

end prove_money_given_l493_493161


namespace exists_line_l_intersects_g_at_angle_l493_493947

variable (P : Plane)
variable (g : Line)
variable (α : Real) -- Assuming α is given in radians

theorem exists_line_l_intersects_g_at_angle :
  ∃ l : Line, (l ∈ P) ∧ (angle_between l g = α) :=
sorry

end exists_line_l_intersects_g_at_angle_l493_493947


namespace moles_of_NaOH_combined_l493_493587

-- Define the reaction conditions
variable (moles_NH4NO3 : ℕ) (moles_NaNO3 : ℕ)

-- Define a proof problem that asserts the number of moles of NaOH combined
theorem moles_of_NaOH_combined
  (h1 : moles_NH4NO3 = 3)  -- 3 moles of NH4NO3 are combined
  (h2 : moles_NaNO3 = 3)  -- 3 moles of NaNO3 are formed
  : ∃ moles_NaOH : ℕ, moles_NaOH = 3 :=
by {
  -- Proof skeleton to be filled
  sorry
}

end moles_of_NaOH_combined_l493_493587


namespace exists_combination_bounded_length_l493_493701

theorem exists_combination_bounded_length
  (n : ℕ) (n_pos : n > 0)
  (v : Fin n → EuclideanSpace ℝ (Fin 2))
  (hv : ∀ i, ‖v i‖ ≤ 1) :
  ∃ ξ : Fin n → ℤ, (∀ i, ξ i = 1 ∨ ξ i = -1) ∧ ‖∑ i, (ξ i) • v i‖ ≤ Real.sqrt 2 :=
sorry

end exists_combination_bounded_length_l493_493701


namespace colton_stickers_l493_493162

theorem colton_stickers :
  (original : ℕ) (friends : ℕ) (stickers_per_friend : ℕ) (stickers_mandy_more : ℕ) (stickers_justin_less : ℕ) :
  original = 72 →
  friends = 3 →
  stickers_per_friend = 4 →
  stickers_mandy_more = 2 →
  stickers_justin_less = 10 →
  (remaining_stickers : ℕ) 
  (stickers_given_to_three_friends := friends * stickers_per_friend)
  (stickers_given_to_mandy := stickers_given_to_three_friends + stickers_mandy_more)
  (stickers_given_to_justin := stickers_given_to_mandy - stickers_justin_less)
  (total_stickers_given := stickers_given_to_three_friends + stickers_given_to_mandy + stickers_given_to_justin)
  (remaining_stickers = original - total_stickers_given) :=
  remaining_stickers = 42 :=
sorry

end colton_stickers_l493_493162


namespace num_of_integers_l493_493926

theorem num_of_integers (n : ℤ) (h : n ≠ 25) : card { n : ℤ | ∃ k : ℤ, k^2 = n / (25 - n) } = 2 :=
by
  sorry

end num_of_integers_l493_493926


namespace h_at_8_l493_493344

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

noncomputable def h (x : ℝ) : ℝ :=
  let a := 1
  let b := 1
  let c := 2
  (1/2) * (x - a^3) * (x - b^3) * (x - c^3)

theorem h_at_8 : h 8 = 147 := 
by 
  sorry

end h_at_8_l493_493344


namespace sum_x_coordinates_eq_4_l493_493200

theorem sum_x_coordinates_eq_4 :
  let f := λ x : ℝ, |x^2 - 4 * x + 3|
  let g := λ x : ℝ, 5 - 2 * x
  let solutions := { x : ℝ | f x = g x }
  (∑ x in solutions, x) = 4 :=
by
  have H1 : ∀ x, (x^2 - 4 * x + 3) = (x - 3) * (x - 1), sorry
  have H2 : ∀ x, x ≤ 1 ∨ x ≥ 3 → |x^2 - 4 * x + 3| = x^2 - 4 * x + 3, sorry
  have H3 : ∀ x, 1 < x ∧ x < 3 → |x^2 - 4 * x + 3| = -(x^2 - 4 * x + 3), sorry
  sorry

end sum_x_coordinates_eq_4_l493_493200


namespace shaded_area_calculation_l493_493072

-- Definitions based on conditions in the problem
def right_triangle_area (a b : ℝ) : ℝ := (a * b) / 2

def rectangle_area (a b : ℝ) : ℝ := a * b

-- Problem hypotheses
variables (a b : ℝ)
variables (A B D C : ℝ)

-- Condition 1: Two congruent right triangles
def congruent_right_triangles (a b : ℝ) (c d e f : ℝ) : Prop :=
a = c ∧ b = d ∧ (a^2 + b^2 = e^2) ∧ (c^2 + d^2 = f^2)

-- Condition 2: First figure arrangement
def first_figure (a b : ℝ) : ℝ := (3 / 4) * (rectangle_area a b)

-- Condition 3: Second figure arrangement
def second_figure (hyp : ℝ) : ℝ := 2 * (hyp / 2) + 14

-- Shaded Area in both figures
theorem shaded_area_calculation :
  congruent_right_triangles 4 7 4 7 (sqrt (4^2 + 7^2)) (sqrt (4^2 + 7^2)) →
  A = 7 ∧ B = 4 →
  first_figure 7 4 = 21 ∧
  second_figure (33 / 14) = 131 / 7 :=
  by sorry

end shaded_area_calculation_l493_493072


namespace num_unique_two_digit_numbers_l493_493246

-- Define the available digits and the rule that no digit is repeated
def digits : List Nat := [2, 4, 7, 8]
def unique_two_digit_numbers : Nat :=
  (digits.length) * (digits.length - 1)

theorem num_unique_two_digit_numbers : unique_two_digit_numbers = 12 :=
by
  -- Formalize the calculation as described in the solution
  have h1 : digits.length = 4 := rfl
  have h2 : 4 - 1 = 3 := rfl
  show unique_two_digit_numbers = 12
  unfold unique_two_digit_numbers
  rw [h1, h2]
  rfl

end num_unique_two_digit_numbers_l493_493246


namespace tessa_initial_apples_l493_493375

-- Define conditions as variables
variable (initial_apples anita_gave : ℕ)
variable (apples_needed_for_pie : ℕ := 10)
variable (apples_additional_now_needed : ℕ := 1)

-- Define the current amount of apples Tessa has
noncomputable def current_apples :=
  apples_needed_for_pie - apples_additional_now_needed

-- Define the initial apples Tessa had before Anita gave her 5 apples
noncomputable def initial_apples_calculated :=
  current_apples - anita_gave

-- Lean statement to prove the initial number of apples Tessa had
theorem tessa_initial_apples (h_initial_apples : anita_gave = 5) : initial_apples_calculated = 4 :=
by
  -- Here is where the proof would go; we use sorry to indicate it's not provided
  sorry

end tessa_initial_apples_l493_493375


namespace relationship_among_a_b_c_l493_493227

def a : ℝ := 2 ^ 0.1
def b : ℝ := (1 / 2) ^ (-0.4)
def c : ℝ := 2 * Real.log 2 / Real.log 7

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  have ha : a = 2 ^ 0.1 := rfl
  have hb : b = (1 / 2) ^ (-0.4) := rfl
  have hc : c = 2 * Real.log 2 / Real.log 7 := rfl
  sorry

end relationship_among_a_b_c_l493_493227


namespace sindy_expression_value_l493_493011

theorem sindy_expression_value :
  let ns := ((List.range 200).filter (λ n, n % 10 ≠ 0)),
      alternated_sum := ns.enum.foldl (λ acc ⟨i, n⟩, if i % 2 = 0 then acc + n else acc - n) 0
  in alternated_sum = 109 :=
by
  -- Definitions and setup for the terms, filtering, and sum calculation
  let ns := (List.range 200).filter (λ n, n % 10 ≠ 0)
  let alternated_sum := ns.enum.foldl
    (λ acc ⟨i, n⟩, if i % 2 = 0 then acc + n else acc - n) 0
  show alternated_sum = 109
  sorry

end sindy_expression_value_l493_493011


namespace units_digit_factorial_150_l493_493458

theorem units_digit_factorial_150 : (nat.factorial 150) % 10 = 0 :=
sorry

end units_digit_factorial_150_l493_493458


namespace min_value_2x_minus_y_l493_493659

open Real

theorem min_value_2x_minus_y : ∀ (x y : ℝ), |x| ≤ y ∧ y ≤ 2 → ∃ (c : ℝ), c = 2 * x - y ∧ ∀ z, z = 2 * x - y → z ≥ -6 := sorry

end min_value_2x_minus_y_l493_493659


namespace number_of_possible_rational_roots_l493_493514

theorem number_of_possible_rational_roots (b_3 b_2 b_1 : ℤ) : 
  -- Given polynomial
  let p := 8 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + 18
  -- Potential rational roots according to the theorem
  let possible_roots := {rational : ℚ | p.eval rational = 0}
  -- Size of the unique rational roots
  possible_roots.to_finset.card = 24 :=
sorry

end number_of_possible_rational_roots_l493_493514


namespace perimeter_quadrilateral_ABCD_l493_493308

-- Define the necessary points and values
noncomputable def A : ℝ × ℝ := (0, 20.785)
noncomputable def B : ℝ × ℝ := (0, 0)
noncomputable def C : ℝ × ℝ := (9, -5.196)
noncomputable def D : ℝ × ℝ := (13.5, -2.598)
noncomputable def E : ℝ × ℝ := (12, 0)

-- Define the lengths based on the given conditions
def AE : ℝ := 30
def BE : ℝ := AE / 2
def AB : ℝ := AE * (Real.sqrt 3 / 2)
def EC : ℝ := BE / 2
def BC : ℝ := BE * (Real.sqrt 3 / 2)
def DE : ℝ := EC / 2
def CD : ℝ := EC * (Real.sqrt 3 / 2)
def EA : ℝ := AE
def DA : ℝ := DE + EA

-- Prove the perimeter of quadrilateral ABCD
theorem perimeter_quadrilateral_ABCD : 
  AB + BC + CD + DA = 26.25 * Real.sqrt 3 + 33.75 :=
  by
    have h1 : AB = 15 * Real.sqrt 3 := by sorry
    have h2 : BC = 7.5 * Real.sqrt 3 := by sorry
    have h3 : CD = 3.75 * Real.sqrt 3 := by sorry
    have h4 : DA = 33.75 := by sorry
    have h5 : AB + BC + CD + DA = 26.25 * Real.sqrt 3 + 33.75 := by sorry
    exact h5

end perimeter_quadrilateral_ABCD_l493_493308


namespace planA_charge_for_8_minutes_eq_48_cents_l493_493122

theorem planA_charge_for_8_minutes_eq_48_cents
  (X : ℝ)
  (hA : ∀ t : ℝ, t ≤ 8 → X = X)
  (hB : ∀ t : ℝ, 6 * 0.08 = 0.48)
  (hEqual : 6 * 0.08 = X) :
  X = 0.48 := by
  sorry

end planA_charge_for_8_minutes_eq_48_cents_l493_493122


namespace average_book_width_l493_493316

noncomputable def book_widths : List ℚ := [7, 3/4, 1.25, 3, 8, 2.5, 12]
def number_of_books : ℕ := 7
def total_sum_of_widths : ℚ := 34.5

theorem average_book_width :
  ((book_widths.sum) / number_of_books) = 241/49 :=
by
  sorry

end average_book_width_l493_493316


namespace car_travel_time_on_regular_road_l493_493121

theorem car_travel_time_on_regular_road :
  (∀ (x : ℝ), 60 * x = 2 * 100 * (2.2 - x) → x = 1) :=
by
  intro x
  have h : 60 * x = 2 * 100 * (2.2 - x)
  sorry

end car_travel_time_on_regular_road_l493_493121


namespace at_least_one_large_abs_l493_493034

theorem at_least_one_large_abs (a b c d : ℤ) (h_not_all_equal : ¬ (a = b ∧ b = c ∧ c = d)) :
  ∃ n, ∃ m > n, (let (a_n, b_n, c_n, d_n) := iterate (λ (t : ℤ × ℤ × ℤ × ℤ), (t.1 - t.2, t.2 - t.3, t.3 - t.4, t.4 - t.1)) m (a, b, c, d) in
  abs a_n > |a| ∨ abs b_n > |b| ∨ abs c_n > |c| ∨ abs d_n > |d|) :=
sorry

end at_least_one_large_abs_l493_493034


namespace race_heartbeats_l493_493150

def heart_rate : ℕ := 140
def pace : ℕ := 6
def distance : ℕ := 30
def total_time (pace distance : ℕ) : ℕ := pace * distance
def total_heartbeats (heart_rate total_time : ℕ) : ℕ := heart_rate * total_time pace distance

theorem race_heartbeats : total_heartbeats heart_rate (total_time pace distance) = 25200 := by
  sorry

end race_heartbeats_l493_493150


namespace no_natural_numbers_for_squares_l493_493562

theorem no_natural_numbers_for_squares :
  ∀ x y : ℕ, ¬(∃ k m : ℕ, k^2 = x^2 + y ∧ m^2 = y^2 + x) :=
by sorry

end no_natural_numbers_for_squares_l493_493562


namespace notification_probability_l493_493516

theorem notification_probability
  (num_students : ℕ)
  (num_notified_Li : ℕ)
  (num_notified_Zhang : ℕ)
  (prob_Li : ℚ)
  (prob_Zhang : ℚ)
  (h1 : num_students = 10)
  (h2 : num_notified_Li = 4)
  (h3 : num_notified_Zhang = 4)
  (h4 : prob_Li = (4 : ℚ) / 10)
  (h5 : prob_Zhang = (4 : ℚ) / 10) :
  prob_Li + prob_Zhang - prob_Li * prob_Zhang = (16 : ℚ) / 25 := 
by 
  sorry

end notification_probability_l493_493516


namespace taxi_ride_cost_l493_493522

theorem taxi_ride_cost (initial_cost : ℝ) (cost_first_3_miles : ℝ) (rate_first_3_miles : ℝ) (rate_after_3_miles : ℝ) (total_miles : ℝ) (remaining_miles : ℝ) :
  initial_cost = 2.00 ∧ rate_first_3_miles = 0.30 ∧ rate_after_3_miles = 0.40 ∧ total_miles = 8 ∧ total_miles - 3 = remaining_miles →
  initial_cost + 3 * rate_first_3_miles + remaining_miles * rate_after_3_miles = 4.90 :=
sorry

end taxi_ride_cost_l493_493522


namespace possible_double_roots_l493_493848

theorem possible_double_roots (b1 b2 b3 s : ℤ) :
  (x : ℝ) → (x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 50 = 0) →
  (x - s)^2 ∣ (x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 50) →
  s = 1 ∨ s = -1 ∨ s = 5 ∨ s = -5 :=
begin
  sorry
end

end possible_double_roots_l493_493848


namespace general_term_formula_sum_of_reciprocal_d_l493_493744

-- Define the sequence and its properties
variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}

-- Given conditions
axiom sum_of_geometric_sequence
  (n : ℕ) : S n = ∑ i in (Finset.range n), a i

axiom geometric_recurrence
  (n : ℕ) : a (n + 1) = 2 * S n + 2

-- Inserting n numbers between a_n and a_(n+1) forms an arithmetic sequence
variable {d : ℕ → ℚ}

axiom arithmetic_sequence
  (n : ℕ) : a (n + 1) = a n + (n + 1) * d n

-- Prove the general term formula for the sequence
theorem general_term_formula (n : ℕ) : a n = 2 * 3 ^ (n - 1) := sorry

-- Prove the sum of the first n terms of the sequence 1/d_n
theorem sum_of_reciprocal_d (n : ℕ) :
  (∑ i in (Finset.range n), 1 / d i) = 15 / 16 - (2 * n + 5) / (16 * 3 ^ (n - 1)) := sorry

end general_term_formula_sum_of_reciprocal_d_l493_493744


namespace log_sequence_equality_l493_493949

theorem log_sequence_equality (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 1) (h2: a 2 + a 4 + a 6 = 18) : 
  Real.logb 3 (a 5 + a 7 + a 9) = 3 := 
by
  sorry

end log_sequence_equality_l493_493949


namespace flat_rate_first_night_l493_493130

-- Definitions of conditions
def total_cost_sarah (f n : ℕ) := f + 3 * n = 210
def total_cost_mark (f n : ℕ) := f + 7 * n = 450

-- Main theorem to be proven
theorem flat_rate_first_night : 
  ∃ f n : ℕ, total_cost_sarah f n ∧ total_cost_mark f n ∧ f = 30 :=
by
  sorry

end flat_rate_first_night_l493_493130


namespace problem_value_expression_l493_493005

theorem problem_value_expression 
  (x y : ℝ)
  (h₁ : x + y = 4)
  (h₂ : x * y = -2) : 
  x + (x^3 / y^2) + (y^3 / x^2) + y = 440 := 
sorry

end problem_value_expression_l493_493005


namespace exists_monomial_with_coefficient_and_degree_l493_493087

theorem exists_monomial_with_coefficient_and_degree :
  ∃ (f : ℤ[X]) (a : ℤ), a = -5 ∧ (degree (monomial 2 (a) : ℤ[X]) = 2) := 
sorry

end exists_monomial_with_coefficient_and_degree_l493_493087


namespace area_of_tangent_segments_l493_493942

theorem area_of_tangent_segments (r : ℝ) (h : r = 3) : 
  ∃ A : ℝ, A = 4 * Real.pi ∧ ∀ AB : ℝ, AB = 4 → is_tangent_mid AB r :=
sorry

end area_of_tangent_segments_l493_493942


namespace find_first_term_l493_493031

-- Define the geometric sequence properties
variables {α : Type*} [field α] [semiring α]

open_locale big_operators

noncomputable def first_term_of_geometric_sequence
  (t_4 t_5 : α) 
  (h1 : t_4 = 24) 
  (h2 : t_5 = 48) : α :=
  let r := t_5 / t_4 in
  let a := t_4 / r^3 in
  a

-- The theorem statement
theorem find_first_term
  (t_4 t_5 : ℝ)
  (h1 : t_4 = 24)
  (h2 : t_5 = 48) : 
  first_term_of_geometric_sequence t_4 t_5 h1 h2 = 3 := 
sorry

end find_first_term_l493_493031


namespace classics_books_l493_493394

theorem classics_books (authors : ℕ) (books_per_author : ℕ) (h_authors : authors = 6) (h_books_per_author : books_per_author = 33) :
  authors * books_per_author = 198 := 
by { rw [h_authors, h_books_per_author], norm_num }

end classics_books_l493_493394


namespace weight_loss_clothes_percentage_l493_493473

theorem weight_loss_clothes_percentage (W : ℝ) : 
  let initial_weight := W
  let weight_after_loss := 0.89 * initial_weight
  let final_weight_with_clothes := 0.9078 * initial_weight
  let added_weight_percentage := (final_weight_with_clothes / weight_after_loss - 1) * 100
  added_weight_percentage = 2 :=
by
  sorry

end weight_loss_clothes_percentage_l493_493473


namespace rose_bush_cost_correct_l493_493868

-- Definitions of the given conditions
def total_rose_bushes : ℕ := 20
def gardener_rate : ℕ := 30
def gardener_hours_per_day : ℕ := 5
def gardener_days : ℕ := 4
def gardener_cost : ℕ := gardener_rate * gardener_hours_per_day * gardener_days
def soil_cubic_feet : ℕ := 100
def soil_cost_per_cubic_foot : ℕ := 5
def soil_cost : ℕ := soil_cubic_feet * soil_cost_per_cubic_foot
def total_cost : ℕ := 4100

-- Result computed given the conditions
def rose_bush_cost : ℕ := 150

-- The proof goal (statement only, no proof)
theorem rose_bush_cost_correct : 
  total_cost - gardener_cost - soil_cost = total_rose_bushes * rose_bush_cost :=
by
  sorry

end rose_bush_cost_correct_l493_493868


namespace order_of_6_l493_493281

def f (x : ℤ) : ℤ := (x^2) % 13

theorem order_of_6 :
  ∀ n : ℕ, (∀ k < n, f^[k] 6 ≠ 6) → f^[n] 6 = 6 → n = 72 :=
by
  sorry

end order_of_6_l493_493281


namespace total_tickets_sold_correct_total_tickets_sold_is_21900_l493_493566

noncomputable def total_tickets_sold : ℕ := 5400 + 16500

theorem total_tickets_sold_correct :
    total_tickets_sold = 5400 + 5 * (16500 / 5) :=
by
  rw [Nat.div_mul_cancel]
  sorry

-- The following theorem states the main proof equivalence:
theorem total_tickets_sold_is_21900 :
    total_tickets_sold = 21900 :=
by
  sorry

end total_tickets_sold_correct_total_tickets_sold_is_21900_l493_493566


namespace projection_of_v2_on_b_l493_493849

-- Define the initial vectors
def u := (4 : ℝ, 4 : ℝ)
def v1 := (60 / 13 : ℝ, 12 / 13 : ℝ)
def v2 := (-2 : ℝ, 2 : ℝ)
def b := (5 : ℝ, 1 : ℝ)

-- Define necessary operations
noncomputable def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_prod := λ x y : ℝ × ℝ, x.1 * y.1 + x.2 * y.2
  let proj_scalar := (dot_prod a b) / (dot_prod b b)
  (proj_scalar * b.1, proj_scalar * b.2)

-- The statement of the proof
theorem projection_of_v2_on_b : proj v2 b = (-20 / 13 : ℝ, -4 / 13 : ℝ) :=
by
  sorry

end projection_of_v2_on_b_l493_493849


namespace park_area_l493_493138

theorem park_area (l w : ℝ) (h1 : l + w = 40) (h2 : l = 3 * w) : l * w = 300 :=
by
  sorry

end park_area_l493_493138


namespace tims_total_equals_toms_total_tims_total_minus_toms_total_is_zero_l493_493678

-- Define constants based on the problem
def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25

-- Define calculations for Tim's total
def tims_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)

-- Define calculations for Tom's total
def toms_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

-- Lean theorem statement proving both totals are equal
theorem tims_total_equals_toms_total : tims_total = toms_total := 
by
  calc
    tims_total = (original_price * (1 + sales_tax_rate)) * (1 - discount_rate) : by rfl
           ... = (original_price * 1.08) * 0.75 : by rfl
           ... = (original_price * 0.75) * 1.08 : by ring
           ... = toms_total : by rfl

-- Assert that the difference between Tim's and Tom's total is 0
theorem tims_total_minus_toms_total_is_zero : tims_total - toms_total = 0 := 
by 
  rw [tims_total_equals_toms_total] 
  exact sub_self toms_total

end tims_total_equals_toms_total_tims_total_minus_toms_total_is_zero_l493_493678


namespace probability_in_D_l493_493247

noncomputable def f (x : ℝ) : ℝ := if x < 0 then 2 ^ x else 0

def D : set ℝ := {y : ℝ | 0 < y ∧ y < 1}

def interval (a b : ℝ) : set ℝ := {x : ℝ | a < x ∧ x < b}

theorem probability_in_D :
  let A := interval (-1) 2
      B := interval (-1) 0 in
  (measure_theory.measure_space.volume B / measure_theory.measure_space.volume A) = 1 / 3 :=
by 
  sorry

end probability_in_D_l493_493247


namespace units_digit_of_150_factorial_is_zero_l493_493467

theorem units_digit_of_150_factorial_is_zero : 
  ∃ k : ℕ, (150! = k * 10) :=
begin
  -- We need to prove that there exists a natural number k such that 150! is equal to k times 10
  sorry
end

end units_digit_of_150_factorial_is_zero_l493_493467


namespace line_AP_passes_through_orthocenter_l493_493220

variables {R : Type*} [Field R]
variables {A B C P : R^3} -- points in the plane
variables {λ : R} {AB AC : R^3} -- vectors in the plane

def is_orthocenter (A B C P : R^3) : Prop :=
  ∃ λ : R, λ ≠ 0 ∧ (P - A) = λ * ((B - A) / (‖B - A‖ * real.cos(∠ B A C)) + (C - A) / (‖C - A‖ * real.cos(∠ A C B)))

theorem line_AP_passes_through_orthocenter (h : (P - A) = λ * ((B - A) / (‖B - A‖ * real.cos(∠ B A C)) + (C - A) / (‖C - A‖ * real.cos(∠ A C B)))) :
  is_orthocenter A B C P :=
begin
  sorry
end

end line_AP_passes_through_orthocenter_l493_493220


namespace arithmetic_sequence_common_difference_l493_493307

theorem arithmetic_sequence_common_difference :
  let a : ℕ → ℤ := λ n, 2 - 3 * (n : ℤ)
  ∃ d : ℤ, d = -3 ∧ (∀ n : ℕ, a (n + 1) - a n = d) :=
by
  let a : ℕ → ℤ := λ n, 2 - 3 * (n : ℤ)
  use -3
  split
  · rfl
  · intros n
    simp [a]
    ring
  ⟩ sorry

end arithmetic_sequence_common_difference_l493_493307


namespace necessary_and_sufficient_condition_l493_493224

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b > a * b ↔ (a = 1 ∨ b = 1) :=
sorry

end necessary_and_sufficient_condition_l493_493224


namespace vodka_concentration_18_l493_493498

-- Define the volumes and concentrations of vodka in each vessel
def volume_A : ℝ := 3
def conc_vodka_A : ℝ := 0.40
def volume_B : ℝ := 5
def conc_vodka_B : ℝ := 0.20
def volume_C : ℝ := 6
def conc_vodka_C : ℝ := 0.10
def volume_D : ℝ := 8
def conc_vodka_D : ℝ := 0.10

-- Define the total volume of the cocktail
def total_volume_cocktail : ℝ := 20

-- Calculate the total amount of vodka
def total_vodka : ℝ :=
  (volume_A * conc_vodka_A) +
  (volume_B * conc_vodka_B) +
  (volume_C * conc_vodka_C) +
  (volume_D * conc_vodka_D)

-- Calculate the concentration of vodka in the "Challenge Cocktail"
def conc_vodka_cocktail : ℝ := total_vodka / total_volume_cocktail

-- Theorem: The concentration of vodka in the "Challenge Cocktail" is 18%
theorem vodka_concentration_18 :
  conc_vodka_cocktail = 0.18 :=
  by sorry

end vodka_concentration_18_l493_493498


namespace proof_sqrt_of_123454321_eq_11111_l493_493075
-- Import necessary Mathlib modules

noncomputable def sqrt_of_123454321_eq_11111 : Prop :=
  sqrt 123454321 = 11111

-- Proof placeholder
theorem proof_sqrt_of_123454321_eq_11111 : sqrt_of_123454321_eq_11111 :=
  sorry

end proof_sqrt_of_123454321_eq_11111_l493_493075


namespace toad_difference_l493_493414

variables (Tim_toads Jim_toads Sarah_toads : ℕ)

theorem toad_difference (h1 : Tim_toads = 30) 
                        (h2 : Jim_toads > Tim_toads) 
                        (h3 : Sarah_toads = 2 * Jim_toads) 
                        (h4 : Sarah_toads = 100) :
  Jim_toads - Tim_toads = 20 :=
by
  -- The next lines are placeholders for the logical steps which need to be proven
  sorry

end toad_difference_l493_493414


namespace delphi_population_2070_l493_493572

theorem delphi_population_2070 (P_2020 : ℕ) (doubles_every : ℕ) (years_diff : ℕ) 
  (P_2020_eq : P_2020 = 350) (doubles_every_eq : doubles_every = 30) (years_diff_eq : years_diff = 50) :
  let P_2050 := P_2020 * 2 in
  let P_2070 := P_2050 * 2^(2/3 : ℝ) in
  P_2070 = 700 * 2^(2/3 : ℝ) :=
by 
  have h1 : P_2020 = 350 := P_2020_eq,
  have h2 : doubles_every = 30 := doubles_every_eq,
  have h3 : years_diff = 50 := years_diff_eq,
  let P_2050 := P_2020 * 2,
  let P_2070 := P_2050 * 2^(2/3 : ℝ),
  sorry

end delphi_population_2070_l493_493572


namespace range_of_a_l493_493984

theorem range_of_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5})
  (hB : B = {x | 3 ≤ x ∧ x ≤ 22}) :
  A ⊆ (A ∩ B) ↔ (1 ≤ a ∧ a ≤ 9) :=
by
  sorry

end range_of_a_l493_493984


namespace goldfish_cost_graph_is_finite_set_of_points_l493_493993

theorem goldfish_cost_graph_is_finite_set_of_points :
  ∀ (n : ℤ), (1 ≤ n ∧ n ≤ 12) → ∃ (C : ℤ), C = 15 * n ∧ ∀ m ≠ n, C ≠ 15 * m :=
by
  -- The proof goes here
  sorry

end goldfish_cost_graph_is_finite_set_of_points_l493_493993


namespace factorial_square_gt_power_l493_493365

theorem factorial_square_gt_power {n : ℕ} (h : n > 2) : (n! * n!) > n^n :=
sorry

end factorial_square_gt_power_l493_493365


namespace conjugate_of_complex_l493_493627

open Complex

theorem conjugate_of_complex (z : ℂ) (h : z = 2 / (1 - I)) : conj z = 1 - I :=
by
  sorry

end conjugate_of_complex_l493_493627


namespace impossible_event_abs_lt_zero_l493_493799

theorem impossible_event_abs_lt_zero (a : ℝ) : ¬ (|a| < 0) :=
sorry

end impossible_event_abs_lt_zero_l493_493799


namespace angle_BPC_eq_80_l493_493685

theorem angle_BPC_eq_80 (A B C P : Type) (hPAC : ∠ PAC = 10) (hPCA : ∠ PCA = 20) (hPAB : ∠ PAB = 30) (hABC : ∠ ABC = 40) : ∠ BPC = 80 :=
sorry

end angle_BPC_eq_80_l493_493685


namespace reciprocal_condition_l493_493661

theorem reciprocal_condition (m : ℤ) (h : 1 / (-0.5) = -(m + 4)) : m = 2 := by
  sorry

end reciprocal_condition_l493_493661


namespace fishing_probability_correct_l493_493779

-- Definitions for probabilities
def P_sunny : ℝ := 0.3
def P_rainy : ℝ := 0.5
def P_cloudy : ℝ := 0.2

def P_fishing_given_sunny : ℝ := 0.7
def P_fishing_given_rainy : ℝ := 0.3
def P_fishing_given_cloudy : ℝ := 0.5

-- The total probability function
def P_fishing : ℝ :=
  P_sunny * P_fishing_given_sunny +
  P_rainy * P_fishing_given_rainy +
  P_cloudy * P_fishing_given_cloudy

theorem fishing_probability_correct : P_fishing = 0.46 :=
by 
  sorry -- Proof goes here

end fishing_probability_correct_l493_493779


namespace at_least_15_pairs_l493_493128

def min_socks_to_guarantee_pairs (red green blue yellow purple : ℕ) (pairs_needed : ℕ) : ℕ :=
  118

theorem at_least_15_pairs (red green blue yellow purple : ℕ) :
  red = 120 → green = 100 → blue = 90 → yellow = 70 → purple = 50 →
  min_socks_to_guarantee_pairs red green blue yellow purple 15 = 118 :=
by
  intros
  simp [min_socks_to_guarantee_pairs]
  sorry

end at_least_15_pairs_l493_493128


namespace integral_evaluation_l493_493480

noncomputable def definiteIntegral : ℝ :=
  ∫ x in 0..(Real.pi / 2), (Real.sin x) ^ 2 / (1 + Real.cos x + Real.sin x) ^ 2

theorem integral_evaluation : definiteIntegral = (1 / 2) - (1 / 2) * Real.log 2 := by
  sorry

end integral_evaluation_l493_493480


namespace units_digit_of_150_factorial_is_zero_l493_493463

theorem units_digit_of_150_factorial_is_zero : 
  ∃ k : ℕ, (150! = k * 10) :=
begin
  -- We need to prove that there exists a natural number k such that 150! is equal to k times 10
  sorry
end

end units_digit_of_150_factorial_is_zero_l493_493463


namespace Susan_roses_ratio_l493_493374

theorem Susan_roses_ratio (total_roses given_roses vase_roses remaining_roses : ℕ) 
  (H1 : total_roses = 3 * 12)
  (H2 : vase_roses = total_roses - given_roses)
  (H3 : remaining_roses = vase_roses * 2 / 3)
  (H4 : remaining_roses = 12) :
  given_roses / gcd given_roses total_roses = 1 ∧ total_roses / gcd given_roses total_roses = 2 :=
by
  sorry

end Susan_roses_ratio_l493_493374


namespace infinite_series_sum_l493_493592

theorem infinite_series_sum :
  (∑' n : ℕ, (2 * n + 1) * (1 / 2023)^n) = 1.002472 := 
sorry

end infinite_series_sum_l493_493592


namespace optimal_speed_minimizes_cost_l493_493408

noncomputable def transportation_cost (x : ℝ) : ℝ :=
  let base_rate := 0.5 * 500
  let speed_surcharge := if x > 30 then 0.03 * (x - 30) * 500 else 0
  let time_surcharge := let t := 500 / x in if t > 10 then 60 * (t - 10) else 0
  base_rate + speed_surcharge + time_surcharge

theorem optimal_speed_minimizes_cost : ∃ x : ℝ, x = real.sqrt 2000 ∧ 
  ∀ y : ℝ, y > 0 → (transportation_cost x ≤ transportation_cost y) :=
sorry

end optimal_speed_minimizes_cost_l493_493408


namespace log9_8000_nearest_integer_l493_493785

theorem log9_8000_nearest_integer :
  (Real.log 8000 / Real.log 9).round = 4 :=
by
  have h1 : Real.log 6561 / Real.log 9 = 4 := by sorry
  have h2 : 6561 < 8000 ∧ 8000 < 59049 := by sorry
  sorry

end log9_8000_nearest_integer_l493_493785


namespace magnitude_of_z_l493_493228

open Complex

def z : ℂ := 1 / (2 + I)

theorem magnitude_of_z : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end magnitude_of_z_l493_493228


namespace AnaMinimumSpeed_l493_493778

def AnaDistance : ℝ := 5 -- Ana needs to complete 5 kilometers
def TimeInMinutes : ℝ := 20 -- in under 20 minutes
def TimeInHours : ℝ := TimeInMinutes / 60 -- Convert time to hours
def MinimumSpeed : ℝ := 15 -- Minimum speed in km/h

theorem AnaMinimumSpeed : 
  ∃ v : ℝ, (AnaDistance / TimeInHours) = v ∧ v = MinimumSpeed :=
by
  sorry

end AnaMinimumSpeed_l493_493778


namespace sufficient_condition_l493_493922

variable (x : ℝ) (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, |x| + |x - 1| ≥ 1) : a < 1 → ∀ x : ℝ, a ≤ |x| + |x - 1| :=
by
  sorry

end sufficient_condition_l493_493922


namespace line_intersects_circle_l493_493361

-- Definitions
def point_outside_circle (x0 y0 R : ℝ) : Prop := x0^2 + y0^2 > R^2

def line (x0 y0 R : ℝ) : set (ℝ × ℝ) := {p | x0 * p.1 + y0 * p.2 = R^2}

def circle (R : ℝ) : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = R^2}

def intersects (S1 S2 : set (ℝ × ℝ)) : Prop := ∃ p, p ∈ S1 ∧ p ∈ S2

-- Theorem statement
theorem line_intersects_circle
  (x0 y0 R : ℝ)
  (h : point_outside_circle x0 y0 R) :
  intersects (line x0 y0 R) (circle R) :=
sorry

end line_intersects_circle_l493_493361


namespace sum_adjacent_odd_l493_493108

/-
  Given 2020 natural numbers written in a circle, prove that the sum of any two adjacent numbers is odd.
-/

noncomputable def numbers_in_circle : Fin 2020 → ℕ := sorry

theorem sum_adjacent_odd (k : Fin 2020) :
  (numbers_in_circle k + numbers_in_circle (k + 1)) % 2 = 1 :=
sorry

end sum_adjacent_odd_l493_493108


namespace rectangle_area_l493_493036

theorem rectangle_area :
  ∃ (l w : ℝ), l = 4 * w ∧ 2 * l + 2 * w = 200 ∧ l * w = 1600 :=
by
  use [80, 20]
  split; norm_num
  split; norm_num
  sorry

end rectangle_area_l493_493036


namespace transform_sine_graph_l493_493880

theorem transform_sine_graph (x : ℝ) :
  sin (-2 * x + π / 4) = sin (2 * (x - π / 8)) :=
by
  sorry

end transform_sine_graph_l493_493880


namespace bounded_set_decomposable_l493_493167

theorem bounded_set_decomposable {n : ℕ} (S : set (fin n → ℝ)) (r : ℝ) (B : set (fin n → ℝ)) :
  bounded S ∧ (∃ x : fin n → ℝ, metric.closed_ball x r ⊆ S) →
  (∃ T : set (fin n → ℝ), bounded T ∧ (∃ y : fin n → ℝ, metric.closed_ball y 1 ⊆ T) ∧ (T ⊆ S)) ∧
  (∃ U : set (fin n → ℝ), bounded U ∧ (∃ z : fin n → ℝ, metric.closed_ball z r ⊆ U) ∧ (U ⊆ S)) :=
begin
  sorry
end

end bounded_set_decomposable_l493_493167


namespace maximum_sequence_length_positive_integer_x_l493_493903

/-- Define the sequence terms based on the problem statement -/
def sequence_term (n : ℕ) (a₁ a₂ : ℤ) : ℤ :=
  if n = 1 then a₁
  else if n = 2 then a₂
  else sequence_term (n - 2) a₁ a₂ - sequence_term (n - 1) a₁ a₂

/-- Define the main problem with the conditions -/
theorem maximum_sequence_length_positive_integer_x :
  ∃ x : ℕ, 0 < x ∧ (309 = x) ∧ 
  (∀ n, sequence_term n 500 x ≥ 0) ∧
  (sequence_term 11 500 x < 0) :=
by
  sorry

end maximum_sequence_length_positive_integer_x_l493_493903


namespace bus_speed_including_stoppages_l493_493184

theorem bus_speed_including_stoppages 
  (speed_without_stoppages : ℕ) 
  (stoppage_time_per_hour : ℕ) 
  (correct_speed_including_stoppages : ℕ) :
  speed_without_stoppages = 54 →
  stoppage_time_per_hour = 10 →
  correct_speed_including_stoppages = 45 :=
by
sorry

end bus_speed_including_stoppages_l493_493184


namespace trains_meet_distance_l493_493093

noncomputable def time_difference : ℝ :=
  5 -- Time difference between two departures in hours

noncomputable def speed_train_a : ℝ :=
  30 -- Speed of Train A in km/h

noncomputable def speed_train_b : ℝ :=
  40 -- Speed of Train B in km/h

noncomputable def distance_train_a : ℝ :=
  speed_train_a * time_difference -- Distance covered by Train A before Train B starts

noncomputable def relative_speed : ℝ :=
  speed_train_b - speed_train_a -- Relative speed of Train B with respect to Train A

noncomputable def catch_up_time : ℝ :=
  distance_train_a / relative_speed -- Time taken for Train B to catch up with Train A

noncomputable def distance_from_delhi : ℝ :=
  speed_train_b * catch_up_time -- Distance from Delhi where the two trains will meet

theorem trains_meet_distance :
  distance_from_delhi = 600 := by
  sorry

end trains_meet_distance_l493_493093


namespace probability_multiple_of_4_l493_493409

def spinner_1_numbers : List ℕ := [1, 4, 6]
def spinner_2_numbers : List ℕ := [3, 5, 7]

def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def valid_sums : List ℕ :=
  [x + y | x ← spinner_1_numbers, y ← spinner_2_numbers]

def multiples_of_4 (ns : List ℕ) : List ℕ :=
  ns.filter is_multiple_of_4

theorem probability_multiple_of_4 :
  (multiples_of_4 valid_sums).length / valid_sums.length = 2 / 9 := 
by
  sorry

end probability_multiple_of_4_l493_493409


namespace evaluate_fraction_l493_493182

theorem evaluate_fraction : (1 / (2 + (1 / (3 + (1 / 4))))) = 13 / 30 :=
by
  sorry

end evaluate_fraction_l493_493182


namespace negation_of_proposition_l493_493286

theorem negation_of_proposition {x : ℝ} :
  (¬ ∃ x_0 > 0, |x_0| ≤ 1) ↔ (∀ x > 0, |x| > 1) :=
by
  sorry

end negation_of_proposition_l493_493286


namespace testing_methods_l493_493930

noncomputable def num_possible_methods : ℕ :=
  let genuine_items := 6
  let defective_items := 4
  (defective_items * (genuine_items * (3!)) * (4!))

theorem testing_methods : num_possible_methods = 576 := by
  sorry

end testing_methods_l493_493930


namespace cadence_total_earnings_l493_493542

noncomputable def total_earnings (old_years : ℕ) (old_monthly : ℕ) (new_increment : ℤ) (extra_months : ℕ) : ℤ :=
  let old_months := old_years * 12
  let old_earnings := old_monthly * old_months
  let new_monthly := old_monthly + ((old_monthly * new_increment) / 100)
  let new_months := old_months + extra_months
  let new_earnings := new_monthly * new_months
  old_earnings + new_earnings

theorem cadence_total_earnings :
  total_earnings 3 5000 20 5 = 426000 :=
by
  sorry

end cadence_total_earnings_l493_493542


namespace sum_of_g_f_values_l493_493490

/-- Definition of the functions f and g with their respective domains and ranges -/
def f : ℕ → ℕ :=
  λ x, match x with
  | 0 => 1
  | 1 => 3
  | 2 => 5
  | 3 => 7
  | _ => 0 -- This should never be reached since the domain is restricted to {0, 1, 2, 3}
  end

def g : ℕ → ℕ :=
  λ x, match x with
  | 2 => x + 2
  | 3 => x + 2
  | 4 => x + 2
  | 5 => x + 2
  | _ => 0 -- This should never be reached since the domain is restricted to {2, 3, 4, 5}
  end

/-- The sum of all possible values of g(f(x)), given the specified conditions on f and g -/
theorem sum_of_g_f_values : (g (f 1)) + (g (f 2)) = 12 :=
  by
  -- Proof will involve checking the values individually
  sorry

end sum_of_g_f_values_l493_493490


namespace solve_complex_division_l493_493229

noncomputable def z : ℂ := 1 + complex.i 

theorem solve_complex_division (hz : z = 1 + complex.i) : (2 / z) = 1 - complex.i :=
by
  sorry

end solve_complex_division_l493_493229


namespace solution_in_range_for_fraction_l493_493931

theorem solution_in_range_for_fraction (a : ℝ) : 
  (∃ x : ℝ, (2 * x + a) / (x + 1) = 1 ∧ x < 0) ↔ (a > 1 ∧ a ≠ 2) :=
by
  sorry

end solution_in_range_for_fraction_l493_493931


namespace sum_first_n_terms_l493_493032

noncomputable def seq (n : ℕ) : ℕ := n

noncomputable def sum_seq (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), seq i

theorem sum_first_n_terms (n : ℕ) : sum_seq n = n * (n + 1) / 2 :=
by sorry

end sum_first_n_terms_l493_493032


namespace area_of_triangle_bounded_by_given_lines_l493_493581

-- Define the lines and find their area bounded with y-axis
def line1 (x : ℝ) := 2 * x - 1
def line2 (x : ℝ) := (16 - x) / 4

-- Point of intersection of the lines
def intersection : ℝ × ℝ := (20 / 9, 31 / 9)

-- Base and height of the triangle
def base : ℝ := 5
def height (intersect : ℝ × ℝ) : ℝ := intersect.1

-- Calculate the area of the triangle
def triangle_area (b h : ℝ) : ℝ := (1 / 2) * b * h

theorem area_of_triangle_bounded_by_given_lines : triangle_area base (height intersection) = 50 / 9 := by
  sorry

end area_of_triangle_bounded_by_given_lines_l493_493581


namespace laurie_shells_l493_493730

def alan_collected : ℕ := 48
def ben_collected (alan : ℕ) : ℕ := alan / 4
def laurie_collected (ben : ℕ) : ℕ := ben * 3

theorem laurie_shells (a : ℕ) (b : ℕ) (l : ℕ) (h1 : alan_collected = a)
  (h2 : ben_collected a = b) (h3 : laurie_collected b = l) : l = 36 := 
by
  sorry

end laurie_shells_l493_493730


namespace min_value_of_vector_sum_l493_493611

noncomputable def min_vector_sum_magnitude (P Q: (ℝ×ℝ)) : ℝ :=
  let x := P.1
  let y := P.2
  let a := Q.1
  let b := Q.2
  Real.sqrt ((x + a)^2 + (y + b)^2)

theorem min_value_of_vector_sum :
  ∃ P Q, 
  (P.1 - 2)^2 + (P.2 - 2)^2 = 1 ∧ 
  Q.1 + Q.2 = 1 ∧ 
  min_vector_sum_magnitude P Q = (5 * Real.sqrt 2 - 2) / 2 :=
by
  sorry

end min_value_of_vector_sum_l493_493611


namespace rectangle_area_l493_493041

noncomputable def length (w : ℝ) : ℝ := 4 * w

noncomputable def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area :
  ∀ (l w : ℝ), 
  l = length w ∧ perimeter l w = 200 → area l w = 1600 :=
by
  intros l w h
  cases h with h1 h2
  rw [length, perimeter, area] at *
  sorry

end rectangle_area_l493_493041


namespace probability_multiple_of_7_condition_l493_493418

theorem probability_multiple_of_7_condition :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b ∧ (ab + a + b + 1) % 7 = 0 → 
  (1295 / 4950 = 259 / 990) :=
sorry

end probability_multiple_of_7_condition_l493_493418


namespace abs_nonneg_l493_493798

theorem abs_nonneg (a : ℝ) : 0 ≤ |a| :=
sorry

end abs_nonneg_l493_493798


namespace evaluate_expression_l493_493487

theorem evaluate_expression (x : ℕ) (h : x = 3) : 5^3 - 2^x * 3 + 4^2 = 117 :=
by
  rw [h]
  sorry

end evaluate_expression_l493_493487


namespace sum_indexed_coeff_leq_half_l493_493718

theorem sum_indexed_coeff_leq_half (n : ℕ) (a : ℕ → ℝ)
  (h1 : ∑ i in Finset.range n, a i = 0)
  (h2 : ∑ i in Finset.range n, |a i| = 1) :
  | ∑ i in Finset.range n, i * a i | ≤ (n - 1) / 2 :=
sorry

end sum_indexed_coeff_leq_half_l493_493718


namespace maximum_sequence_length_positive_integer_x_l493_493905

/-- Define the sequence terms based on the problem statement -/
def sequence_term (n : ℕ) (a₁ a₂ : ℤ) : ℤ :=
  if n = 1 then a₁
  else if n = 2 then a₂
  else sequence_term (n - 2) a₁ a₂ - sequence_term (n - 1) a₁ a₂

/-- Define the main problem with the conditions -/
theorem maximum_sequence_length_positive_integer_x :
  ∃ x : ℕ, 0 < x ∧ (309 = x) ∧ 
  (∀ n, sequence_term n 500 x ≥ 0) ∧
  (sequence_term 11 500 x < 0) :=
by
  sorry

end maximum_sequence_length_positive_integer_x_l493_493905


namespace limestone_mass_l493_493358

theorem limestone_mass (mass_fraction_HCl : ℝ) (total_solution_mass : ℝ)
    (molar_mass_HCl : ℝ) (molar_mass_CaCO3 : ℝ) (purity_limestone : ℝ) :
    mass_fraction_HCl = 0.20 →
    total_solution_mass = 150 →
    molar_mass_HCl = 36.5 →
    molar_mass_CaCO3 = 100 →
    purity_limestone = 0.97 →
    (let mass_HCl := total_solution_mass * mass_fraction_HCl,
         moles_HCl := mass_HCl / molar_mass_HCl,
         moles_CaCO3 := moles_HCl / 2,
         mass_CaCO3 := moles_CaCO3 * molar_mass_CaCO3,
         total_mass_limestone := mass_CaCO3 / purity_limestone
      in total_mass_limestone = 42.27) :=
by
  intros
  unfold let mass_HCl moles_HCl moles_CaCO3 mass_CaCO3 total_mass_limestone
  sorry

end limestone_mass_l493_493358


namespace find_f_of_neg3_l493_493282

-- Definitions based on problem conditions
def g (x : ℝ) : ℝ := 2 * x - 1
def f_comp (x : ℝ) : ℝ := (1 + x^2) / (3 * x^2)

-- The theorem we need to prove
theorem find_f_of_neg3 : (f_comp (-1) : ℝ) = (f_comp ((g (-1)))) := by
  sorry

end find_f_of_neg3_l493_493282


namespace minimal_period_f_monotonic_intervals_f_range_f_on_interval_l493_493630

noncomputable def f (x : ℝ) : ℝ := (cos x) ^ 4 - 2 * (sin x) * (cos x) - (sin x) ^ 4

theorem minimal_period_f :
  function.periodic (λ x, f x) π := sorry

theorem monotonic_intervals_f :
  ∀ k : ℤ, (∀ x : ℝ, x ∈ set.Icc (-5 * π / 8 + k * π) (-π / 8 + k * π) → monotone_increasing (λ x, f x)) := sorry

theorem range_f_on_interval :
  set.range (λ x, f x) (set.Icc 0 (π / 2)) = set.Icc (-√2) 1 := sorry

end minimal_period_f_monotonic_intervals_f_range_f_on_interval_l493_493630


namespace sequence_term_position_l493_493768

theorem sequence_term_position (n : ℕ) (h : 2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) : n = 7 :=
sorry

end sequence_term_position_l493_493768


namespace vlad_taller_than_sister_l493_493782

theorem vlad_taller_than_sister : 
  ∀ (vlad_height sister_height : ℝ), 
  vlad_height = 190.5 → sister_height = 86.36 → vlad_height - sister_height = 104.14 :=
by
  intros vlad_height sister_height vlad_height_eq sister_height_eq
  rw [vlad_height_eq, sister_height_eq]
  sorry

end vlad_taller_than_sister_l493_493782


namespace symmetry_axis_l493_493172

theorem symmetry_axis (x : ℝ) : (∀ x, 3 * sin (2 * x + π/4) = 3 * sin (2 * (π/8) - (2 * x + π/4) + π/4)) ↔ x = π / 8 :=
by
  sorry

end symmetry_axis_l493_493172


namespace rectangle_area_l493_493038

theorem rectangle_area (l w : ℕ) 
  (h1 : l = 4 * w) 
  (h2 : 2 * l + 2 * w = 200) 
  : l * w = 1600 := 
by 
  sorry

end rectangle_area_l493_493038


namespace units_digit_of_150_factorial_is_zero_l493_493437

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l493_493437


namespace part1_part2_l493_493253

-- Define f(x) = |x-4|
def f (x : ℝ) : ℝ := |x - 4|

-- Part (Ⅰ): if f(x) ≤ 2, then the range of x is [2, 6]
theorem part1 (x : ℝ) : f(x) ≤ 2 → 2 ≤ x ∧ x ≤ 6 :=
by sorry

-- Define g(x) = 2 * sqrt(|x-2|) + sqrt(|x-6|)
def g (x : ℝ) : ℝ := 2 * Real.sqrt (|x - 2|) + Real.sqrt (|x - 6|)

-- Part (Ⅱ): given 2 ≤ x ≤ 6, the maximum value of g(x) is 2√5
theorem part2 (x : ℝ) : 2 ≤ x → x ≤ 6 → g(x) ≤ 2 * Real.sqrt 5 :=
by sorry

end part1_part2_l493_493253


namespace john_trip_time_l493_493692

theorem john_trip_time (normal_distance : ℕ) (normal_time : ℕ) (extra_distance : ℕ) 
  (double_extra_distance : ℕ) (same_speed : ℕ) 
  (h1: normal_distance = 150) 
  (h2: normal_time = 3) 
  (h3: extra_distance = 50)
  (h4: double_extra_distance = 2 * extra_distance)
  (h5: same_speed = normal_distance / normal_time) : 
  normal_time + double_extra_distance / same_speed = 5 :=
by 
  sorry

end john_trip_time_l493_493692


namespace part1_part2_l493_493977

variable (a b c : ℝ)
def f (x : ℝ) := -x^3 + a*x^2 + b*x + c
def f' (x : ℝ) := -3*x^2 + 2*a*x + b

theorem part1 (h1 : f' 1 = -3) (h2 : f 1 = -2) (h3 : f' (-2) = 0) :
  f = λ x, -x^3 - 2*x^2 + 4*x - 3 := sorry

theorem part2 (h_incr: ∀ x ∈ Icc (-2 : ℝ) 0, f' x ≥ 0) : b ≥ 4 := sorry

end part1_part2_l493_493977


namespace equation_one_solution_equation_two_solution_l493_493372

-- Define the first problem and expected solutions
theorem equation_one_solution (x : ℝ) : x^2 + 8 * x - 9 = 0 ↔ x = -9 ∨ x = 1 := 
begin
  sorry
end

-- Define the second problem and expected solutions
theorem equation_two_solution (x : ℝ) : x * (x - 1) + 3 * (x - 1) = 0 ↔ x = -3 ∨ x = 1 := 
begin
  sorry
end

end equation_one_solution_equation_two_solution_l493_493372


namespace range_of_a_l493_493651

-- Define the condition set
def condition_set (a : ℝ) := {x : ℝ | x^2 ≤ a}

-- Statement of the problem in Lean 4
theorem range_of_a (a : ℝ) (h : ∅ ⊂ condition_set a) : a ∈ set.Ici 0 :=
sorry

end range_of_a_l493_493651


namespace complex_conjugate_magnitude_l493_493629

open Complex

-- Condition
def z : ℂ := 2 * I / (1 - I)

-- Question statement to be proved
theorem complex_conjugate_magnitude : abs (conj z + 3 * I) = Real.sqrt 5 := sorry

end complex_conjugate_magnitude_l493_493629


namespace units_digit_of_150_factorial_is_zero_l493_493449

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l493_493449


namespace dodecagon_diagonals_intersection_l493_493551

open Point Geometry

theorem dodecagon_diagonals_intersection {P : ℕ → Point} {O : Point} 
  (h1 : ∀ n, distance (P n) O = distance (P (n+1)) O)
  (h2 : ∀ n, angle (P n) O (P (n+1)) = π / 6) :
    let P1 := P 1
    let P9 := P 9
    let P2 := P 2
    let P11 := P 11
    let P4 := P 4
    let P12 := P 12
    line_through_intersection (diagonal P1 P9) (diagonal P2 P11) (diagonal P4 P12) = O := 
sorry

end dodecagon_diagonals_intersection_l493_493551


namespace four_leaf_area_l493_493006

theorem four_leaf_area (a : ℝ) : 
  let radius := a / 2
  let semicircle_area := (π * radius ^ 2) / 2
  let triangle_area := (a / 2) * (a / 2) / 2
  let half_leaf_area := semicircle_area - triangle_area
  let leaf_area := 2 * half_leaf_area
  let total_area := 4 * leaf_area
  total_area = a ^ 2 / 2 * (π - 2) := 
by
  sorry

end four_leaf_area_l493_493006


namespace evaluate_expression_l493_493181

theorem evaluate_expression : 
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end evaluate_expression_l493_493181


namespace intersection_lines_l493_493914

theorem intersection_lines (x y : ℝ) :
  (2 * x - y - 10 = 0) ∧ (3 * x + 4 * y - 4 = 0) → (x = 4) ∧ (y = -2) :=
by
  -- The proof is provided here
  sorry

end intersection_lines_l493_493914


namespace quadrant_third_l493_493610

-- Define the complex number z
def z : ℂ := (1 - complex.i) / (1 + complex.i)

-- Define the target complex number (z / (1 + complex.i))
def target : ℂ := z / (1 + complex.i)

-- Prove that target is in the third quadrant
theorem quadrant_third : target.re < 0 ∧ target.im < 0 := by
  -- The expanded proof body with steps would go here, but we'll use sorry for now.
  sorry

end quadrant_third_l493_493610


namespace coupon1_greatest_discount_at_229_95_l493_493836

variable (x : ℝ)
variable (discount1 discount2 discount3 : ℝ)

/-- 
Conditions:
1. Coupon 1: 15% off the listed price if the listed price is at least $80.
2. Coupon 2: $30 off the listed price if the listed price is at least $150.
3. Coupon 3: 25% off the amount by which the listed price exceeds $150.
4. We are examining the price $229.95.
-/

def coupon1_discount (x : ℝ) : ℝ :=
  if x ≥ 80 then 0.15 * x else 0

def coupon2_discount (x : ℝ) : ℝ :=
  if x ≥ 150 then 30 else 0

def coupon3_discount (x : ℝ) : ℝ :=
  if x > 150 then 0.25 * (x - 150) else 0

theorem coupon1_greatest_discount_at_229_95 (x : ℝ) (h1 : x = 229.95) :
  coupon1_discount x > coupon2_discount x ∧ coupon1_discount x > coupon3_discount x := by
  sorry

end coupon1_greatest_discount_at_229_95_l493_493836


namespace modulus_of_complex_eq_sqrt2_l493_493628

theorem modulus_of_complex_eq_sqrt2 (z : ℂ) : (2 - complex.i) * z = 3 + complex.i → complex.norm z = real.sqrt 2 := by
  sorry

end modulus_of_complex_eq_sqrt2_l493_493628


namespace units_digit_of_150_factorial_is_zero_l493_493434

theorem units_digit_of_150_factorial_is_zero : (Nat.factorial 150) % 10 = 0 := by
sorry

end units_digit_of_150_factorial_is_zero_l493_493434


namespace minimum_N_for_set_conditions_l493_493106

theorem minimum_N_for_set_conditions (k : ℕ) (hk : 0 < k) :
  ∃ (a : Fin (2*k + 1) → ℕ), 
    (∑ i, a i > 2*k^3 + 3*k^2 + 3*k) ∧
    (∀ (S : Finset (Fin (2*k + 1))), S.card = k → ∑ i in S, a i ≤ (2*k^3 + 3*k^2 + 3*k) / 2) :=
sorry

end minimum_N_for_set_conditions_l493_493106


namespace negation_of_proposition_l493_493986

theorem negation_of_proposition (m a b : ℝ) (h : m ≠ 0) :
  ¬ (2 ^ a > 2 ^ b → a * m ^ 2 > b * m ^ 2) ↔ (2 ^ a ≤ 2 ^ b → a * m ^ 2 ≤ b * m ^ 2) :=
sorry

end negation_of_proposition_l493_493986


namespace simplify_to_ap_minus_b_l493_493369

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((7*p + 3) - 3*p * 2) * 4 + (5 - 2 / 4) * (8*p - 12)

theorem simplify_to_ap_minus_b (p : ℝ) :
  simplify_expression p = 40 * p - 42 :=
by
  -- Proof steps would go here
  sorry

end simplify_to_ap_minus_b_l493_493369


namespace not_divisible_by_121_l493_493399

theorem not_divisible_by_121 (n : ℤ) : ¬ ∃ t : ℤ, (n^2 + 3*n + 5) = 121 * t ∧ (n^2 - 3*n + 5) = 121 * t := sorry

end not_divisible_by_121_l493_493399


namespace units_produced_today_l493_493598

theorem units_produced_today (n : ℕ) (P : ℕ) (T : ℕ) 
  (h1 : n = 14)
  (h2 : P = 60 * n)
  (h3 : (P + T) / (n + 1) = 62) : 
  T = 90 :=
by
  sorry

end units_produced_today_l493_493598


namespace length_of_wall_l493_493646

-- Define the dimensions of a brick
def brick_length : ℝ := 40
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the dimensions of the wall
def wall_height : ℝ := 600
def wall_width : ℝ := 22.5

-- Define the required number of bricks
def required_bricks : ℝ := 4000

-- Calculate the volume of a single brick
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Calculate the volume of the wall
def volume_wall (length : ℝ) : ℝ := length * wall_height * wall_width

-- The theorem to prove
theorem length_of_wall : ∃ (L : ℝ), required_bricks * volume_brick = volume_wall L → L = 800 :=
sorry

end length_of_wall_l493_493646


namespace matrix_power_50_l493_493320

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := ![(5 : ℤ), 2; -16, -6]

-- The target matrix we want to prove A^50 equals to
def target : Matrix (Fin 2) (Fin 2) ℤ := ![(-301 : ℤ), -100; 800, 299]

-- Prove that A^50 equals to the target matrix
theorem matrix_power_50 : A^50 = target := 
by {
  sorry
}

end matrix_power_50_l493_493320


namespace part1_part2_axis_of_symmetry_part2_center_of_symmetry_l493_493269

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x) ^ 2, Real.sin x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem part1 (x : ℝ) (h1 : 0 < x ∧ x < π) (h2 : perpendicular (m x) (n x)) :
  x = π / 2 ∨ x = 3 * π / 4 :=
sorry

theorem part2_axis_of_symmetry (k : ℤ) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = f (2 * c - x) ∧ 
    ((2 * x + π / 4) = k * π + π / 2 → x = k * π / 2 + π / 8) :=
sorry

theorem part2_center_of_symmetry (k : ℤ) :
  ∃ x c : ℝ, f x = 1 ∧ ((2 * x + π / 4) = k * π → x = k * π / 2 - π / 8) :=
sorry

end part1_part2_axis_of_symmetry_part2_center_of_symmetry_l493_493269


namespace eccentricity_of_hyperbola_l493_493205

-- Defining the hyperbola and its conditions
variables (a b k : ℝ) (ha : a > 0) (hb : b > 0) (hk : k > 0)

def hyperbola (x y : ℝ) := x^2 / a^2 - y^2 / b^2 = 1

-- Distance relationships for points P, Q, and foci F1, F2
variables (PQ PF2 F1F2 : ℝ)
variables (hPQ : PQ = 3 * k) (hPF2 : PF2 = 4 * k)
variables (hF1F2 : F1F2 = 2 * c)

-- Defining c such that F1F2 is the distance between the foci
noncomputable def c := sqrt (a^2 + b^2)

-- Conditions and relationship for perpendicular distances within the circle
variables (hF1_F2_perp_PQ : PF2^2 + (F1F2 / 2)^2 = PQ^2)
variables (hF1 : |PF1| = sqrt (F1F2/2)^2)
variables (hEccentricity : e = F1F2 / 2 / a)

theorem eccentricity_of_hyperbola :
  let e := F1F2 / 2 / a in
  e = sqrt 17 / 3 :=
by
  sorry

end eccentricity_of_hyperbola_l493_493205


namespace find_angle_C_max_area_of_triangle_l493_493954

variables {A B C : ℝ} {a b c : ℝ}
variable (R : ℝ := sqrt 2)

theorem find_angle_C 
  (h1 : 2 * sqrt 2 * (sin A ^ 2 - sin C ^ 2) = (a - b) * sin B)
  (h2 : R = sqrt 2) :
  C = π / 3 := sorry

theorem max_area_of_triangle
  (h1 : 2 * sqrt 2 * (sin A ^ 2 - sin C ^ 2) = (a - b) * sin B)
  (h2 : R = sqrt 2)
  (h3 : C = π / 3) :
  ∃ S_max, S_max = 3 * sqrt(3) / 2 := sorry

end find_angle_C_max_area_of_triangle_l493_493954


namespace tangent_line_at_1_extreme_points_range_of_a_l493_493254

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + a * (x ^ 2 - 3 * x + 2)

theorem tangent_line_at_1 (a : ℝ) (h : a = 0) :
  ∃ m b, ∀ x, f x a = m * x + b ∧ m = 1 ∧ b = -1 := sorry

theorem extreme_points (a : ℝ) :
  (0 < a ∧ a <= 8 / 9 → ∀ x, 0 < x → f x a = 0) ∧
  (a > 8 / 9 → ∃ x1 x2, x1 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x1 → f x a = 0) ∧
   (∀ x, x1 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) ∧
  (a < 0 → ∃ x1 x2, x1 < 0 ∧ 0 < x2 ∧ f x1 a = 0 ∧ f x2 a = 0 ∧
   (∀ x, 0 < x ∧ x < x2 → f x a = 0) ∧
   (∀ x, x2 < x → f x a = 0)) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → f x a >= 0) ↔ 0 ≤ a ∧ a ≤ 1 := sorry

end tangent_line_at_1_extreme_points_range_of_a_l493_493254


namespace intervals_of_monotonicity_a_eq_e_minimum_value_of_f_number_of_zeros_of_f_l493_493339

noncomputable def f (a x : ℝ) := a^x + x^2 - x * Real.log a - a

-- Statement 1: Monotonicity intervals for a = e
theorem intervals_of_monotonicity_a_eq_e :
  let a := Real.exp 1
  in (∀ x > 0, deriv (f a) x > 0) ∧ (∀ x < 0, deriv (f a) x < 0) :=
  by sorry

-- Statement 2: Minimum value of the function
theorem minimum_value_of_f (a : ℝ) (h : a > 0) (h' : a ≠ 1) :
  ∃ x_min, (∀ x, f a x ≥ f a x_min) ∧ f a x_min = 1 - a :=
  by sorry

-- Statement 3: Number of zeros of the function f
theorem number_of_zeros_of_f (a : ℝ) (h : a > 0) (h' : a ≠ 1) :
  (0 < a ∧ a < 1 → ∀ x, f a x ≠ 0) ∧ (a > 1 → ∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
  by sorry

end intervals_of_monotonicity_a_eq_e_minimum_value_of_f_number_of_zeros_of_f_l493_493339


namespace solve_for_S_l493_493732

-- Defining the conditions based on the problem
def one_half (a : ℝ) := (1 / 2) * a
def one_seventh (a : ℝ) := (1 / 7) * a
def one_fourth (a : ℝ) := (1 / 4) * a
def one_sixth (a : ℝ) := (1 / 6) * a

-- Stating the main theorem to prove
theorem solve_for_S : 
  ∃ S : ℝ, one_half (one_seventh S) = one_fourth (one_sixth 120) ∧ S = 70 := 
sorry

end solve_for_S_l493_493732


namespace rain_on_tuesday_l493_493686

theorem rain_on_tuesday 
  (rain_monday : ℝ)
  (rain_less : ℝ) 
  (h1 : rain_monday = 0.9) 
  (h2 : rain_less = 0.7) : 
  (rain_monday - rain_less) = 0.2 :=
by
  sorry

end rain_on_tuesday_l493_493686


namespace functionalEquation_l493_493337

noncomputable def validFunc : ℝ⁺ → ℝ⁺ := λ x, x^2 + (1/x^2)

theorem functionalEquation (f : ℝ⁺ → ℝ⁺) :
  (∀ x y z : ℝ⁺, 
    f (x * y * z) + f (x) + f (y) + f (z) = f (real.sqrt (x * y)) * f (real.sqrt (y * z)) * f (real.sqrt (z * x))) ∧
  (∀ x y : ℝ⁺, 1 ≤ x → x < y → f(x) < f(y))
  → f = validFunc :=
begin
  sorry
end

end functionalEquation_l493_493337


namespace problem_l493_493096

noncomputable def a := sorry
noncomputable def b := sorry
noncomputable def c := sorry

theorem problem (a b c : ℝ) 
  (h1 : a = (1/3) * b)
  (h2 : b = (1/4) * c)
  (h3 : a + b + c = 1440) : b = 202.5 :=
by
  sorry

end problem_l493_493096


namespace angle_ABC_measure_l493_493401

theorem angle_ABC_measure (O A B C : Point) 
  (hO : is_circumcenter O A B C) 
  (hBOC : ∠ B O C = 150) 
  (hAOB : ∠ A O B = 130) : 
  ∠ A B C = 40 := 
sorry

end angle_ABC_measure_l493_493401


namespace triangle_area_0_0_2_5_9_5_l493_493426

def point := (ℝ × ℝ) -- Define a point as a tuple of two real numbers

def triangle_area (A B C : point) : ℝ :=
  -- Formula for the area of a triangle given its vertices
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_0_0_2_5_9_5 :
  triangle_area (0, 0) (2, 5) (9, 5) = 17.5 :=
by
  -- Lean proof steps would go here
  sorry

end triangle_area_0_0_2_5_9_5_l493_493426


namespace intersection_P_Q_l493_493706

def P := {-3, 0, 2, 4}
def Q := {x : ℝ | -1 < x ∧ x < 3}

theorem intersection_P_Q : P ∩ Q = {0, 2} :=
by
  sorry

end intersection_P_Q_l493_493706


namespace count_numbers_with_D_eq_3_l493_493929

def D (n : ℕ) : ℕ :=
  let b := n.toDigits 2
  (List.zip b (List.tail b)).count (λ (a, b) => a ≠ b)

theorem count_numbers_with_D_eq_3 :
  (Finset.range 201).filter (λ n => D n = 3).card = 33 :=
by sorry

end count_numbers_with_D_eq_3_l493_493929


namespace olympic_numbers_l493_493028

-- Define the functions under consideration
def f_2 (x : ℝ) : ℝ := 2008 * x^3
def f_4 (x : ℝ) : ℝ := log (2008 * x)

-- Define the Olympic number property
def olympic_number_property (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ x1 ∈ D, ∃! x2 ∈ D, f x1 + f x2 = 2008

-- Define the domains for which the property is checked
def D : set ℝ := set.univ -- considering the domain to be all real numbers as specified by the problem

-- Prove that only f_2 and f_4 satisfy the Olympic number property
theorem olympic_numbers :
  olympic_number_property f_2 D ∧ olympic_number_property f_4 D :=
by
  sorry

end olympic_numbers_l493_493028


namespace cos_angles_difference_cos_angles_sum_l493_493820

-- Part (a)
theorem cos_angles_difference: 
  (Real.cos (36 * Real.pi / 180) - Real.cos (72 * Real.pi / 180) = 1 / 2) :=
sorry

-- Part (b)
theorem cos_angles_sum: 
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7) = 1 / 2) :=
sorry

end cos_angles_difference_cos_angles_sum_l493_493820


namespace jason_initial_cards_l493_493690

theorem jason_initial_cards (cards_given_away cards_left : ℕ) (h1 : cards_given_away = 9) (h2 : cards_left = 4) :
  cards_given_away + cards_left = 13 :=
sorry

end jason_initial_cards_l493_493690


namespace problem_part1_problem_part2_l493_493982

noncomputable def f (x : ℝ) : ℝ := 2 * x ^ 2 - 12 * x + 10

theorem problem_part1 :
  ∃ f : ℝ → ℝ, (∀ x, f x = 2 * x ^ 2 - 12 * x + 10) :=
begin
  use f,
  intro x,
  refl,
end

theorem problem_part2 :
  ∀ t : ℝ, (∀ x ∈ set.Icc (1 : ℝ) 3, 2 * x ^ 2 - 12 * x + 10 ≤ 2 + t) → t ≥ -10 :=
begin
  intros t h,
  have key : ∀ x ∈ set.Icc (1 : ℝ) 3, 2 * x ^ 2 - 12 * x + 8 ≤ t,
  { intros x hx,
    specialize h x hx,
    linarith, },
  suffices : 2 * 3 ^ 2 - 12 * 3 + 8 ≤ t,
  { linarith, },
  exact key 3 (by norm_num),
  sorry, -- skipping further proof details
end

end problem_part1_problem_part2_l493_493982


namespace smallest_set_of_circular_handshakes_l493_493496

def circular_handshake_smallest_set (n : ℕ) : ℕ :=
  if h : n % 2 = 0 then n / 2 else (n / 2) + 1

theorem smallest_set_of_circular_handshakes :
  circular_handshake_smallest_set 36 = 18 :=
by
  sorry

end smallest_set_of_circular_handshakes_l493_493496


namespace find_m_l493_493945

noncomputable def point (t m : ℝ) : ℝ × ℝ :=
  (m + (real.sqrt 3) / 2 * t, 1 / 2 * t)

theorem find_m (m : ℝ) :
  (∀ t : ℝ, (m + (real.sqrt 3) / 2 * t, 1 / 2 * t) ∈ ({p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 2 * p.1} : set (ℝ × ℝ)))
  → (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧
     (m + (real.sqrt 3) / 2 * t₁, 1 / 2 * t₁) ∈ ({p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 2 * p.1} : set (ℝ × ℝ)) ∧
     (m + (real.sqrt 3) / 2 * t₂, 1 / 2 * t₂) ∈ ({p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 2 * p.1} : set (ℝ × ℝ)) ∧
     (m * m - m = 2)) → (m = 2 ∨ m = -1) :=
sorry

end find_m_l493_493945


namespace P_lies_on_polar_of_C_wrt_S1_l493_493026

variable (S1 S2 : Circle)
variable (A B O P C : Point)
variable (htwo_intersect : ∀{x : Point}, x ∈ S1 ∧ x ∈ S2 ↔ x = A ∨ x = B)
variable (center_S1_O : O ∈ S1)
variable (O_on_S2 : O ∈ S2)
variable (line_through_O : line_through O P)
variable (P_on_AB : P ∈ line_segment A B)
variable (C_on_S2 : C ∈ S2)
variable (P_on_line_OC : point_on_line_through C O P)

theorem P_lies_on_polar_of_C_wrt_S1 :
  P ∈ polar C S1 := sorry

end P_lies_on_polar_of_C_wrt_S1_l493_493026


namespace marks_exam_scores_l493_493672

theorem marks_exam_scores 
  (first_four_scores : list ℕ := [92, 78, 85, 80])
  (average_score : ℕ := 84)
  (below_95 : ∀ score ∈ first_four_scores, score < 95)
  (unique_scores : ∀ score1 score2 ∈ first_four_scores, score1 = score2 → score1 = score2)
  (new_scores : list ℕ := [94, 75]) :
  (let total_score := (list.sum first_four_scores) + (list.sum new_scores) in
  total_score = 504 ∧ 
  list.sum first_four_scores = 335 ∧
  list.sum new_scores = 169 ∧
  ∀ score ∈ new_scores, score < 95 ∧
  ∀ score1 score2 ∈ new_scores, score1 ≠ score2) →
  list.of_fn (λ i, (list.nth_le ([94, 92, 85, 80, 78, 75]) (i) (by norm_num))) = [94, 92, 85, 80, 78, 75] := 
sorry

end marks_exam_scores_l493_493672


namespace work_completion_days_l493_493502

/-- 
  Given that worker A can complete a job in 15 days, and worker B can complete the same job in 20 days,
  prove that if both work together, they can complete the job in 60/7 days.
-/
theorem work_completion_days (a_days b_days : ℕ) (h_a : a_days = 15) (h_b : b_days = 20) :
  let a_rate := 1 / (a_days : ℚ),
      b_rate := 1 / (b_days : ℚ),
      combined_rate := a_rate + b_rate
  in 1 / combined_rate = 60 / 7 :=
by
  -- Insert proof here
  sorry

end work_completion_days_l493_493502


namespace sum_of_integers_in_factorization_l493_493573

theorem sum_of_integers_in_factorization (x y : ℤ) :
  let expr := 81 * x^4 - 256 * y^4 in
  ∀ a b c d e f : ℤ,
  ((a * x^2 + b * x * y + c * y^2) * (d * x^2 + e * x * y + f * y^2) = expr) →
  (a + b + c + d + e + f = 31) :=
by
  intros
  have h1 : 81 * x^4 - 256 * y^4 = (9 * x^2 - 16 * y^2) * (9 * x^2 + 16 * y^2), from sorry
  have h2 : 9 * x^2 - 16 * y^2 = (3 * x - 4 * y) * (3 * x + 4 * y), from sorry
  have h3 : 9 * x^2 + 16 * y^2 = 9 * x^2 + 16 * y^2, from sorry
  have h4 : (3 * x - 4 * y) * (3 * x + 4 * y) * (9 * x^2 + 16 * y^2) = expr, from sorry
  have h5 : a = 3 ∧ b = -4 ∧ c = 4 ∧ d = 3 ∧ e = 9 ∧ f = 16, from sorry
  rw [h5.1, h5.2.1, h5.2.2.1, h5.2.2.2.1, h5.2.2.2.2]
  calc
    3 + (-4) + 3 + 4 + 9 + 16 = 31 : by norm_num


end sum_of_integers_in_factorization_l493_493573


namespace count_75_ray_not_45_ray_partitional_points_l493_493333

-- Defining the problem conditions
def is_n_ray_partitional (n : ℕ) (Y : ℝ × ℝ) (S : Set (ℝ × ℝ)) : Prop := sorry
def count_n_ray_partitional_points (n : ℕ) (S : Set (ℝ × ℝ)) : ℕ := sorry
def overlap_count (a b : ℕ) (S : Set (ℝ × ℝ)) : ℕ := sorry

axiom unit_square : Set (ℝ × ℝ) -- The unit square region

theorem count_75_ray_not_45_ray_partitional_points :
  let A_75 := count_n_ray_partitional_points 75 unit_square,
      A_45 := count_n_ray_partitional_points 45 unit_square,
      O := overlap_count 75 45 unit_square in
  A_75 - O = 5280 :=
by
  -- conditions for 75-ray partitional points
  have A_75_def : count_n_ray_partitional_points 75 unit_square = 5476 := sorry,
  -- conditions for 45-ray partitional points
  have A_45_def : count_n_ray_partitional_points 45 unit_square = 1936 := sorry,
  -- conditions for overlapping points
  have O_def : overlap_count 75 45 unit_square = 196 := sorry,
  rw [A_75_def, A_45_def, O_def],
  apply Nat.sub_eq_of_eq_add,
  simp [Nat.add_sub_of_le, Nat.sub_add_cancel],
  norm_num,
  apply Nat.le.intro rfl

end count_75_ray_not_45_ray_partitional_points_l493_493333


namespace range_of_f_l493_493386

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 5

theorem range_of_f : set.range (λ x : {x : ℝ // -3 ≤ x ∧ x ≤ 0}, f x) = set.Icc (-6) (-2) :=
by
  sorry

end range_of_f_l493_493386


namespace fraction_sum_simplest_l493_493395

theorem fraction_sum_simplest (a b : ℕ) (h : 0.84375 = (a : ℝ) / b) (ha : a = 27) (hb : b = 32) : a + b = 59 :=
by
  rw [ha, hb]
  norm_num
  sorry

end fraction_sum_simplest_l493_493395


namespace larger_segment_of_triangle_l493_493145

theorem larger_segment_of_triangle (x y : ℝ) (h1 : 40^2 = x^2 + y^2) 
  (h2 : 90^2 = (100 - x)^2 + y^2) :
  100 - x = 82.5 :=
by {
  sorry
}

end larger_segment_of_triangle_l493_493145


namespace solve_system_of_equations_l493_493013

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 1 ∧ x^3 + y^3 + z^3 = 1 ∧ xyz = -16 ↔ 
  (x = 1 ∧ y = 4 ∧ z = -4) ∨ (x = 1 ∧ y = -4 ∧ z = 4) ∨ 
  (x = 4 ∧ y = 1 ∧ z = -4) ∨ (x = 4 ∧ y = -4 ∧ z = 1) ∨ 
  (x = -4 ∧ y = 1 ∧ z = 4) ∨ (x = -4 ∧ y = 4 ∧ z = 1) := 
by
  sorry

end solve_system_of_equations_l493_493013


namespace intersection_A_B_l493_493703

def A : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = cos (arccos x)}
def B : set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = arccos (cos x)}

theorem intersection_A_B :
  ∀ (x y : ℝ), ((x, y) ∈ A ∧ (x, y) ∈ B) ↔ ((x, y) ∈ {p | ∃ x y, p = (x, y) ∧ y = x ∧ -1 ≤ x ∧ x ≤ 1}) :=
by
  sorry

end intersection_A_B_l493_493703


namespace truck_speed_l493_493769

/-- Given the speeds of cars A and B, and the time taken for them to meet a truck, prove that the speed of the truck is 52 km/h. -/
theorem truck_speed
  (speed_A speed_B : ℕ)
  (time_A time_B : ℕ)
  (distance_A distance_B : ℕ)
  (time_difference : ℕ)
  (truck_speed : ℕ)
  (h1 : speed_A = 102)
  (h2 : speed_B = 80)
  (h3 : time_A = 6)
  (h4 : time_B = 7)
  (h5 : distance_A = speed_A * time_A)
  (h6 : distance_B = speed_B * time_B)
  (h7 : time_difference = time_B - time_A)
  (h8 : truck_speed = (distance_A - distance_B) / time_difference) :
  truck_speed = 52 := 
begin
  sorry
end

end truck_speed_l493_493769


namespace option_b_not_valid_l493_493624

theorem option_b_not_valid (a b c d : ℝ) (h_arith_seq : b - a = d ∧ c - b = d ∧ d ≠ 0) : 
  a^3 * b + b^3 * c + c^3 * a < a^4 + b^4 + c^4 :=
by sorry

end option_b_not_valid_l493_493624


namespace find_reflection_matrix_l493_493168

noncomputable def plane_reflection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  -- Input the matrix S
  !![!![1/9, 4/9, -8/9],
     !![4/9, 10/9, 4/9],
     !![-8/9, 4/9, 1/9]]

def normal_vector : Fin 3 → ℚ := ![2, -1, 2]

def plane_passing_origin (u : Fin 3 → ℚ) : Prop :=
  ∃ a b c : ℚ, u = ![a, b, c]

def reflection_condition (u : Fin 3 → ℚ) (S : Matrix (Fin 3) (Fin 3) ℚ) : Prop :=
  S.mul_vec u = vec_sub u ((4 * u 0 - 2 * u 1 + 4 * u 2) * (9⁻¹ : ℚ) • normal_vector)

theorem find_reflection_matrix (u : Fin 3 → ℚ) :
  plane_passing_origin u → reflection_condition u plane_reflection_matrix :=
by
  intros
  sorry

end find_reflection_matrix_l493_493168


namespace light_coloured_blocks_in_tower_l493_493771

theorem light_coloured_blocks_in_tower :
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  total_light_coloured_blocks = 20 :=
by
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  show total_light_coloured_blocks = 20
  sorry

end light_coloured_blocks_in_tower_l493_493771


namespace zeroes_in_base_81_l493_493278

-- Definitions based on the conditions:
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question: How many zeroes does 15! end with in base 81?
-- Lean 4 proof statement:
theorem zeroes_in_base_81 (n : ℕ) : n = 15 → Nat.factorial n = 
  (81 : ℕ) ^ k * m → k = 1 :=
by
  sorry

end zeroes_in_base_81_l493_493278


namespace correct_propositions_l493_493860

noncomputable def proposition1 : Prop :=
  (∀ x : ℝ, x^2 - 3 * x + 2 = 0 -> x = 1) ->
  (∀ x : ℝ, x ≠ 1 -> x^2 - 3 * x + 2 ≠ 0)

noncomputable def proposition2 : Prop :=
  (∀ p q : Prop, p ∨ q -> p ∧ q) ->
  (∀ p q : Prop, p ∧ q -> p ∨ q)

noncomputable def proposition3 : Prop :=
  (∀ p q : Prop, ¬(p ∧ q) -> ¬p ∧ ¬q)

noncomputable def proposition4 : Prop :=
  (∃ x : ℝ, x^2 + x + 1 < 0) ->
  (∀ x : ℝ, x^2 + x + 1 ≥ 0)

theorem correct_propositions :
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l493_493860


namespace shorter_piece_length_l493_493809

-- Definitions for the conditions
def total_length : ℕ := 70
def ratio (short long : ℕ) : Prop := long = (5 * short) / 2

-- The proof problem statement
theorem shorter_piece_length (x : ℕ) (h1 : total_length = x + (5 * x) / 2) : x = 20 :=
sorry

end shorter_piece_length_l493_493809


namespace find_original_number_l493_493754

theorem find_original_number (r : ℝ) (h : 1.15 * r - 0.7 * r = 40) : r = 88.88888888888889 :=
by
  sorry

end find_original_number_l493_493754


namespace a_n_is_base_conversion_l493_493336

noncomputable def a_sequence (p : ℕ) [hp : Nat.Prime p] : ℕ → ℕ 
| 0       => 0
| 1       => 1
| ...     => sorry -- for 2 to p-2
| n       => sorry -- for n ≥ p-1, the least positive integer condition...

theorem a_n_is_base_conversion (p : ℕ) [hp : Nat.Prime p] (hodd : p % 2 = 1) (n : ℕ) : 
  a_sequence p n = sorry -- Transform n to base p-1, then read it as base p
:= sorry

end a_n_is_base_conversion_l493_493336


namespace diagonals_per_vertex_total_diagonals_l493_493855

variable (n : ℕ) (hn : n ≥ 3)

-- Prove a vertex of an n-sided polygon forms (n-3) diagonals.
theorem diagonals_per_vertex (h : n ≥ 3) : ∀ (n : ℕ), n - 3 := 
sorry

-- Prove the total number of diagonals in an n-sided polygon is (n(n-3))/2.
theorem total_diagonals (h : n ≥ 3) : ∀ (n : ℕ), n * (n - 3) / 2 :=
sorry

end diagonals_per_vertex_total_diagonals_l493_493855


namespace boat_distance_against_water_flow_l493_493833

variable (a : ℝ) -- speed of the boat in still water

theorem boat_distance_against_water_flow 
  (speed_boat_still_water : ℝ := a)
  (speed_water_flow : ℝ := 3)
  (time_travel : ℝ := 3) :
  (speed_boat_still_water - speed_water_flow) * time_travel = 3 * (a - 3) := 
by
  sorry

end boat_distance_against_water_flow_l493_493833


namespace daily_distance_zoo_to_train_l493_493382

def daily_kilometers (d n : ℕ) : ℕ :=
  2 * d * n

theorem daily_distance_zoo_to_train (d : ℕ) (n : ℕ) (h_d : d = 33) (h_n : n = 5) :
  daily_kilometers d n = 330 :=
by 
  rw [h_d, h_n]
  simp [daily_kilometers]
  sorry

end daily_distance_zoo_to_train_l493_493382


namespace time_to_cross_man_l493_493853

def train_length : ℝ := 120
def man_speed_kmph : ℝ := 5
def train_speed_kmph : ℝ := 67

def relative_speed_kmph : ℝ := train_speed_kmph + man_speed_kmph
def relative_speed_mps : ℝ := relative_speed_kmph * (1000 / 3600)

theorem time_to_cross_man :
  (train_length / relative_speed_mps) = 6 :=
by
  sorry

end time_to_cross_man_l493_493853


namespace find_a_l493_493016

theorem find_a (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 20) (h2 : (56831742 - a) % 17 = 0) : a = 2 :=
by
  sorry

end find_a_l493_493016


namespace M_invertible_iff_square_free_l493_493202

noncomputable def d (a b : ℕ) : ℕ :=
if a % b = 0 then 1 else 0

def M (n : ℕ) : Matrix (Fin n) (Fin n) ℕ :=
λ i j, d (i + 1) (j + 1)

def square_free (n : ℕ) : Prop :=
∀ m : ℕ, m > 1 → m * m ∣ n → False

theorem M_invertible_iff_square_free (n : ℕ) :
  Invertible (M n) ↔ square_free (n + 1) := sorry

end M_invertible_iff_square_free_l493_493202


namespace finite_division_ring_partition_l493_493009

variable {K : Type*} [DivisionRing K] [Fintype K] (hK : Fintype.card K ≥ 4)

theorem finite_division_ring_partition (hK : Fintype.card K ≥ 4) :
  ∃ (A B : Finset K), 
  A ≠ ∅ ∧ B ≠ ∅ ∧ A ∪ B = (Finset.univ.erase 0) ∧ 
  (∑ x in A, x) = (∏ y in B, y) :=
sorry

end finite_division_ring_partition_l493_493009


namespace probability_odd_number_probability_forming_35_l493_493066

-- Condition: Cards are labeled with numbers 2, 3, and 5
def cards : List ℕ := [2, 3, 5]

-- Proof Problem ①: Prove the probability of drawing an odd number
theorem probability_odd_number : 
  (∃ n ∈ {2, 3, 5}, n % 2 = 1) → 
  (finset.card {n | n ∈ {2, 3, 5} ∧ n % 2 = 1}.toFinset).toFloat / 
  (finset.card {2, 3, 5}.toFinset).toFloat = 2/3 :=
    sorry

-- Proof Problem ②: Prove the probability of forming the number 35
theorem probability_forming_35 :
  (cards.erase 3 = [2, 5]) → -- drawing two cards without replacement
  (finset.card (finset.filter (λ n:ℕ, n = 35) (finset.from_list ([23, 32, 25, 52, 35, 53] : List ℕ)))).toFloat /
  (finset.card {n | ∀ x ∈ cards, ∀ y ∈ cards, x ≠ y -> (x * 10 + y ∈ {23, 32, 25, 52, 35, 53})}.toFinset).toFloat = 1/6 := 
    sorry

end probability_odd_number_probability_forming_35_l493_493066


namespace fair_share_of_bill_l493_493816

noncomputable def total_bill : Real := 139.00
noncomputable def tip_percent : Real := 0.10
noncomputable def num_people : Real := 6
noncomputable def expected_amount_per_person : Real := 25.48

theorem fair_share_of_bill :
  (total_bill + (tip_percent * total_bill)) / num_people = expected_amount_per_person :=
by
  sorry

end fair_share_of_bill_l493_493816


namespace correct_transformation_l493_493524

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : 
  (a / b) = ((a + 2 * a) / (b + 2 * b)) :=
by 
  sorry

end correct_transformation_l493_493524


namespace range_of_a_l493_493290

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 1| > Real.logb 2 a) →
  0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l493_493290


namespace sqrt_linear_combination_l493_493271

theorem sqrt_linear_combination (a b : ℝ) (h1 : sqrt 18 - sqrt 2 = a * sqrt 2 - sqrt 2) (h2 : sqrt 18 - sqrt 2 = b * sqrt 2) : a * b = 6 :=
by sorry

end sqrt_linear_combination_l493_493271


namespace paul_sugar_amount_l493_493889

theorem paul_sugar_amount :
  let initial_sugar := 24 : ℝ in
  let second_week_sugar := initial_sugar / 2 in
  let third_week_sugar := second_week_sugar / 2 in
  let fourth_week_sugar := third_week_sugar / 2 in
  fourth_week_sugar = 3 :=
by
  sorry

end paul_sugar_amount_l493_493889


namespace coin_toss_random_event_l493_493129

noncomputable def coin_toss_event := 
  -- The event that a fair coin is tossed 3 times and lands on heads exactly 1 time
  ∃ head_count : ℕ, head_count = 1 ∧ (∃ (sequence : vector bool 3), sequence.to_list.count tt = head_count)

theorem coin_toss_random_event : 
  coin_toss_event → "random event" := 
sorry

end coin_toss_random_event_l493_493129


namespace basketball_probability_l493_493746

open Real

theorem basketball_probability :
  let n := 5 in
  let p := 0.5 in
  let P (k : ℕ) := (choose n k) * (p ^ k) * ((1 - p) ^ (n - k)) in
  ∑ k in {3, 4, 5}, P k = 1 / 2 :=
by
  let n := 5
  let p := 0.5
  let P (k : ℕ) := (choose n k) * (p ^ k) * ((1 - p) ^ (n - k))
  have h0 : P 3 = (choose n 3) * (p ^ 3) * ((1 - p) ^ (n - 3)) := rfl
  have h1 : P 4 = (choose n 4) * (p ^ 4) * ((1 - p) ^ (n - 4)) := rfl
  have h2 : P 5 = (choose n 5) * (p ^ 5) * ((1 - p) ^ (n - 5)) := rfl
  have hsum : ∑ k in {3, 4, 5}, P k = P 3 + P 4 + P 5 := sorry
  have htotal : P 3 + P 4 + P 5 = 1 / 2 := sorry
  exact calc 
    ∑ k in {3, 4, 5}, P k = P 3 + P 4 + P 5 : hsum
    ... = 1 / 2 : htotal

end basketball_probability_l493_493746


namespace sequence_periodic_l493_493950

theorem sequence_periodic (a : ℕ → ℚ) (h1 : a 1 = 4 / 5)
  (h2 : ∀ n, 0 ≤ a n ∧ a n ≤ 1 → 
    (a (n + 1) = if a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1)) :
  a 2017 = 4 / 5 :=
sorry

end sequence_periodic_l493_493950


namespace sum_of_products_of_three_numbers_l493_493057

theorem sum_of_products_of_three_numbers
    (a b c : ℝ)
    (h1 : a^2 + b^2 + c^2 = 179)
    (h2 : a + b + c = 21) :
  ab + bc + ac = 131 :=
by
  -- Proof goes here
  sorry

end sum_of_products_of_three_numbers_l493_493057


namespace parallel_lines_a_value_l493_493239

theorem parallel_lines_a_value (a : ℝ) :
  (∀ x y : ℝ, x - a * y + a = 0 ∧ 2 * x + y + 2 = 0 → (a + 2 = 0)) :=
begin
  intro h,
  sorry -- proof goes here
end

end parallel_lines_a_value_l493_493239


namespace problem_l493_493250

noncomputable def fx (a b c : ℝ) (x : ℝ) : ℝ := a * x + b / x + c

theorem problem 
  (a b c : ℝ) 
  (h_odd : ∀ x, fx a b c x = -fx a b c (-x))
  (h_f1 : fx a b c 1 = 5 / 2)
  (h_f2 : fx a b c 2 = 17 / 4) :
  (a = 2) ∧ (b = 1 / 2) ∧ (c = 0) ∧ (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / 2 → fx a b c x₁ > fx a b c x₂) := 
sorry

end problem_l493_493250


namespace current_mixture_hcl_percentage_l493_493835

/-- Total volume of the current mixture in milliliters -/
def total_volume_current_mixture := 300

/-- Percentage of water in the current mixture -/
def percent_water_current_mixture := 0.60

/-- Volume of water to be added in milliliters -/
def volume_added_water := 100

/-- Target percentage of water in the final mixture -/
def target_percent_water := 0.70

/-- Target percentage of hydrochloric acid in the final mixture -/
def target_percent_hcl := 0.30

/-- The percentage of hydrochloric acid in the current mixture -/
def percent_hcl_current_mixture : Prop := 
  let volume_water_current := percent_water_current_mixture * total_volume_current_mixture in
  let volume_hcl_current := total_volume_current_mixture - volume_water_current in
  let total_volume_final := total_volume_current_mixture + volume_added_water in
  let volume_water_final := volume_water_current + volume_added_water in
  let percent_water_final := volume_water_final / total_volume_final in
  let percent_hcl_final := volume_hcl_current / total_volume_final in
  percent_hcl_current_mixture = 0.40 ∧ percent_water_final = target_percent_water ∧ percent_hcl_final = target_percent_hcl

theorem current_mixture_hcl_percentage :
  percent_hcl_current_mixture := 
by
  sorry

end current_mixture_hcl_percentage_l493_493835


namespace sum_of_intersection_points_l493_493056

noncomputable def f (x : ℝ) : ℝ := 1 / x - 3 * Real.sin (π * x)

theorem sum_of_intersection_points :
  (∑ (p : ℝ × ℝ) in (set_of (λ (x : ℝ), -1 ≤ x ∧ x ≤ 1 ∧ f x = 0)), (p.1 + p.2)) = 0 := 
sorry

end sum_of_intersection_points_l493_493056


namespace find_norm_a_l493_493236

variables {ℝ : Type*} [linear_ordered_field ℝ] [decidable_eq ℝ] [module ℝ ℝ]

-- Definitions according to conditions
variables (a b : ℝ^3) (x : ℝ)
def angle_between_vectors : ℝ := 2 * π / 3
def norm_b : ℝ := 1
def inequality_holds := ∀ x : ℝ, ∥a + x • b∥ ≥ ∥a + b∥

-- Assumptions derived from conditions
axiom angle_condition : real.angle a b = angle_between_vectors
axiom norm_b_condition : ∥b∥ = norm_b
axiom inequality_condition : inequality_holds x

-- Theorem to prove the question == answer
theorem find_norm_a (a b : ℝ^3) (x : ℝ) 
  (angle_condition : real.angle a b = 2 * π / 3) 
  (norm_b_condition : ∥b∥ = 1)
  (inequality_condition : ∀ x : ℝ, ∥a + x • b∥ ≥ ∥a + b∥) :
  ∥a∥ = 2 := 
sorry

end find_norm_a_l493_493236


namespace impossible_chain_of_dominoes_l493_493175

def even_numbers (tiles : List (ℕ × ℕ)) : ℕ :=
  tiles.foldr (λ t acc => acc + if t.1 % 2 = 0 then 1 else 0 + if t.2 % 2 = 0 then 1 else 0) 0

def odd_numbers (tiles : List (ℕ × ℕ)) : ℕ :=
  tiles.foldr (λ t acc => acc + if t.1 % 2 ≠ 0 then 1 else 0 + if t.2 % 2 ≠ 0 then 1 else 0) 0

theorem impossible_chain_of_dominoes : ¬ ∃ (tiles : List (ℕ × ℕ)),
  length tiles = 28 ∧
  even_numbers tiles = 32 ∧
  odd_numbers tiles = 24 ∧
  (∀ (i : ℕ) (hi : i < length tiles),
    abs (tiles.nthLe i hi).1 - (tiles.nthLe ((i + 1) % length tiles) sorry).1 = 1 ∨
    abs (tiles.nthLe i hi).1 - (tiles.nthLe ((i + 1) % length tiles) sorry).2 = 1 ∨
    abs (tiles.nthLe i hi).2 - (tiles.nthLe ((i + 1) % length tiles) sorry).2 = 1 ∨
    abs (tiles.nthLe i hi).2 - (tiles.nthLe ((i + 1) % length tiles) sorry).1 = 1) :=
by
  sorry

end impossible_chain_of_dominoes_l493_493175


namespace count_numbers_with_D_eq_2_l493_493203

-- Define a function D which counts pairs of different adjacent digits in the binary representation of n.
def D (n : ℕ) : ℕ := 
  (nat.binary_repr n).adjacency_pairs.count (λ (p : ℕ × ℕ), p.1 ≠ p.2)

-- Define a predicate for the main problem.
def problem_condition (n : ℕ) : Prop := (n ≤ 97 ∧ D n = 2)

-- Main statement of the problem.
theorem count_numbers_with_D_eq_2 : finset.card (finset.filter problem_condition (finset.range 98)) = 26 := 
by { sorry }

end count_numbers_with_D_eq_2_l493_493203


namespace nadia_total_cost_l493_493354

noncomputable def calculate_total_cost_after_discount : ℝ :=
  let number_of_roses : ℕ := 20
  let number_of_lilies : ℕ := (3 / 4 : ℝ) * number_of_roses
  let cost_per_rose : ℝ := 5
  let cost_per_lily : ℝ := 2 * cost_per_rose
  let cost_of_roses : ℝ := number_of_roses * cost_per_rose
  let cost_of_lilies : ℝ := number_of_lilies * cost_per_lily
  let total_cost_before_discount : ℝ := cost_of_roses + cost_of_lilies
  let total_number_of_flowers : ℕ := number_of_roses + number_of_lilies
  let discount_rate : ℝ := (total_number_of_flowers / 5) * 0.01
  let discount_amount : ℝ := total_cost_before_discount * discount_rate
  let total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount 
  total_cost_after_discount

-- We assert that the total amount of money Nadia used is $232.50.
theorem nadia_total_cost : calculate_total_cost_after_discount = 232.50 := sorry

end nadia_total_cost_l493_493354


namespace QiangqiangWinsAndLeleCorrect_l493_493932

-- Define the participants
inductive Participant
  | Baby
  | Star
  | Lele
  | Qiangqiang

open Participant

-- Define the conditions (predictions)
def StarPrediction (winner : Participant) : Prop := winner = Lele
def BabyPrediction (winner : Participant) : Prop := winner = Star
def LelePrediction (winner : Participant) : Prop := winner ≠ Lele
def QiangqiangPrediction (winner : Participant) : Prop := winner ≠ Qiangqiang

-- Define that only one of the predictions is correct
def OnlyOneCorrectPrediction (winner : Participant) : Prop :=
  (StarPrediction winner ∧ ¬BabyPrediction winner ∧ ¬LelePrediction winner ∧ ¬QiangqiangPrediction winner) ∨
  (¬StarPrediction winner ∧ BabyPrediction winner ∧ ¬LelePrediction winner ∧ ¬QiangqiangPrediction winner) ∨
  (¬StarPrediction winner ∧ ¬BabyPrediction winner ∧ LelePrediction winner ∧ ¬QiangqiangPrediction winner) ∨
  (¬StarPrediction winner ∧ ¬BabyPrediction winner ∧ ¬LelePrediction winner ∧ QiangqiangPrediction winner)

-- Prove that Qiangqiang is the first place and only Lele's prediction is correct
theorem QiangqiangWinsAndLeleCorrect :
  ∃ winner, winner = Qiangqiang ∧ OnlyOneCorrectPrediction winner :=
by
  sorry

end QiangqiangWinsAndLeleCorrect_l493_493932


namespace rhombus_area_3cm_45deg_l493_493555

noncomputable def rhombusArea (a : ℝ) (theta : ℝ) : ℝ :=
  a * (a * Real.sin theta)

theorem rhombus_area_3cm_45deg :
  rhombusArea 3 (Real.pi / 4) = 9 * Real.sqrt 2 / 2 := 
by
  sorry

end rhombus_area_3cm_45deg_l493_493555


namespace find_quadratic_tangent_to_yx_unique_solution_composition_l493_493478

theorem find_quadratic_tangent_to_yx :
  ∃ a : ℝ, ∀ x : ℝ, (x^2 + a) = x ↔ a = 1 / 4 :=
begin
  sorry
end

theorem unique_solution_composition 
  (P : ℝ → ℝ)
  (hP : ∀ x : ℝ, P x = x^2 + 1 / 4) 
  (unique_x0 : ∃! x0 : ℝ, P x0 = x0) :
  ∃! x0 : ℝ, P (P x0) = x0 :=
begin
  sorry
end

end find_quadratic_tangent_to_yx_unique_solution_composition_l493_493478


namespace units_digit_factorial_150_l493_493461

theorem units_digit_factorial_150 : (nat.factorial 150) % 10 = 0 :=
sorry

end units_digit_factorial_150_l493_493461


namespace midpoint_trajectory_l493_493946

-- Define the parabola and line intersection conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def line_through_focus (A B : ℝ × ℝ) (focus : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, (∀ P ∈ [A, B, focus], P.2 = m * P.1 + b)

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_trajectory (A B M : ℝ × ℝ) (focus : ℝ × ℝ):
  (parabola A.1 A.2) ∧ (parabola B.1 B.2) ∧ (line_through_focus A B focus) ∧ (midpoint A B M)
  → (M.1 ^ 2 = 2 * M.2 - 2) :=
by
  sorry

end midpoint_trajectory_l493_493946


namespace count_integers_product_zero_l493_493928

def integer_product_zero (n : ℕ) : Prop :=
  ∏ k in finset.range n, ((1 + complex.exp (2 * real.pi * complex.I * k / n))^n + 1) = 0

theorem count_integers_product_zero :
  ∑ n in finset.range 3001, (integer_product_zero n ∧ n % 5 = 0) = 100 :=
sorry

end count_integers_product_zero_l493_493928


namespace incorrect_comparison_l493_493082

theorem incorrect_comparison :
  ¬ (- (2 / 3) < - (4 / 5)) :=
by
  sorry

end incorrect_comparison_l493_493082


namespace number_of_barrels_l493_493359

def capacity (barrel : Type) := 7 -- 7 gallons
def flow_rate := 3.5 -- 3.5 gallons per minute
def time := 8 -- 8 minutes

theorem number_of_barrels :
  let total_water_dispensed := flow_rate * time in
  let number_of_barrels := total_water_dispensed / capacity (Unit) in
  number_of_barrels = 4 :=
by
  sorry

end number_of_barrels_l493_493359


namespace swim_team_more_people_l493_493058

theorem swim_team_more_people :
  let car1_people := 5
  let car2_people := 4
  let van1_people := 3
  let van2_people := 3
  let van3_people := 5
  let minibus_people := 10

  let car_max_capacity := 6
  let van_max_capacity := 8
  let minibus_max_capacity := 15

  let actual_people := car1_people + car2_people + van1_people + van2_people + van3_people + minibus_people
  let max_capacity := 2 * car_max_capacity + 3 * van_max_capacity + minibus_max_capacity
  (max_capacity - actual_people : ℕ) = 21 := 
  by
    sorry

end swim_team_more_people_l493_493058


namespace coeff_x3y3_correct_l493_493752

def coeff_x3y3_in_expansion : ℤ :=
  let binomial_coeff := (n k : ℕ) (n.choose k : ℕ) in
  let term1 := 2^2 * (-1)^2 * (binomial_coeff 5 3) in
  let term2 := 2^3 * 1 * (binomial_coeff 5 2) in
  term1 + term2

theorem coeff_x3y3_correct :
  coeff_x3y3_in_expansion = 40 :=
sorry

end coeff_x3y3_correct_l493_493752


namespace max_value_of_a_l493_493289

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f y ≤ f x

theorem max_value_of_a (f : ℝ → ℝ) (a : ℝ) :
  is_odd_function f →
  is_decreasing f →
  (∀ x ∈ ℝ, f (cos (2 * x) + sin x) + f (sin x - a) ≤ 0) →
  a ≤ -3 :=
by
  sorry

end max_value_of_a_l493_493289


namespace probability_ex_one_odd_l493_493934

theorem probability_ex_one_odd (s : Finset ℕ) (h_s : s = {2, 3, 4, 5, 6, 7, 8}) :
  (∃ t ⊆ s, t.card = 2 ∧ (∃ a b ∈ t, (a % 2 = 1 ∧ b % 2 = 0) ∨ (a % 2 = 0 ∧ b % 2 = 1))) →
  ((∑ t in s.powerset.filter (λ t, t.card = 2 ∧ (∃ a b ∈ t, (a % 2 = 1 ∧ b % 2 = 0) ∨ (a % 2 = 0 ∧ b % 2 = 1))), 1) : ℚ) / 
  (s.card.choose 2 : ℚ) = 4 / 7 :=
by
  sorry

end probability_ex_one_odd_l493_493934


namespace period_tan_2x_l493_493793

theorem period_tan_2x : (∃ T, ∀ x, tan (2 * (x + T)) = tan (2 * x)) → T = π / 2 :=
by
  sorry

end period_tan_2x_l493_493793


namespace sequence_max_length_x_l493_493895

theorem sequence_max_length_x (x : ℕ) : 
  (∀ n, a_n = 500 ∧ a_{n+1} = x → (a_{n+2} = a_n - a_{n+1})) →
  (a_{11} > 0 ∧ a_{10} > 0 → x = 500) :=
by
  sorry

end sequence_max_length_x_l493_493895


namespace Alan_current_wings_per_minute_l493_493318

-- Definitions of conditions
def Kevin_wings := 64
def Kevin_time := 8
def Alan_extra_wings_per_minute := 4

-- Theorem statement to prove Alan's current wings per minute rate
theorem Alan_current_wings_per_minute :
  let wings_per_minute_to_beat_kevin := Kevin_wings / Kevin_time in
  let Alan_current_rate := wings_per_minute_to_beat_kevin - Alan_extra_wings_per_minute in
  Alan_current_rate = 4 :=
by
  -- Sorry is used here to skip the proof itself
  sorry

end Alan_current_wings_per_minute_l493_493318


namespace factor_difference_of_squares_l493_493187

theorem factor_difference_of_squares (t : ℝ) : 4 * t^2 - 81 = (2 * t - 9) * (2 * t + 9) := 
by
  sorry

end factor_difference_of_squares_l493_493187


namespace min_abs_sum_l493_493762

theorem min_abs_sum (x : ℝ) : ∃ y : ℝ, y = min ((|x+1| + |x-2| + |x-3|)) 4 :=
sorry

end min_abs_sum_l493_493762


namespace solve_dot_product_l493_493230

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))

def abs_a_eq_4 : Prop := ‖a‖ = 4
def abs_b_eq_2 : Prop := ‖b‖ = 2
def angle_between_ab_eq_120 : Prop := ∃ θ : ℝ, θ = π / 3 ∧ θ = real.angle a b

theorem solve_dot_product (h₁ : abs_a_eq_4 a) (h₂ : abs_b_eq_2 b) (h₃ : angle_between_ab_eq_120 a b) :
  (a + 2 • b) ⬝ (a + b) = 12 :=
sorry

end solve_dot_product_l493_493230


namespace probability_even_sums_is_correct_l493_493765

noncomputable def probability_even_sums : ℚ :=
  let factorial : ℕ → ℕ
      | 0 => 1
      | n + 1 => (n + 1) * factorial n
  let total_arrangements : ℚ := factorial 16
  let valid_placements : ℚ := 36
  valid_placements / total_arrangements

theorem probability_even_sums_is_correct :
  probability_even_sums = 36 / 20922789888000 := by
  sorry

end probability_even_sums_is_correct_l493_493765


namespace find_tangent_points_minimum_circle_area_l493_493959

section

variable (a : ℝ) 

-- (I): Find x1 and x2
theorem find_tangent_points (a : ℝ):
  ∃ x1 x2 : ℝ, x1 = a - sqrt (a^2 + 1) ∧ x2 = a + sqrt (a^2 + 1) 
  ∧ x1 < x2 :=
sorry

-- (II): Find the minimum area of circle E
theorem minimum_circle_area (a : ℝ):
  ∃ r : ℝ, r = (2 * a^2 + 2) / sqrt (4 * a^2 + 1) 
  ∧ (π * r^2 = 3 * π) :=
sorry

end

end find_tangent_points_minimum_circle_area_l493_493959


namespace boys_passed_l493_493098

theorem boys_passed (total_boys : ℕ) (avg_marks : ℕ) (avg_passed : ℕ) (avg_failed : ℕ) (P : ℕ) 
    (h1 : total_boys = 120) (h2 : avg_marks = 36) (h3 : avg_passed = 39) (h4 : avg_failed = 15)
    (h5 : P + (total_boys - P) = 120) 
    (h6 : P * avg_passed + (total_boys - P) * avg_failed = total_boys * avg_marks) :
    P = 105 := 
sorry

end boys_passed_l493_493098


namespace regular_graph_has_small_2_dominating_set_l493_493783

open_locale big_operators

variables {G : Type*} {n r : ℕ} [fintype G]

-- Definition of r-regular graph.
def is_regular_graph (G : Type*) (r : ℕ) [fintype G] [decidable_eq G] :=
  ∀ v : G, (finset.filter (λ u : G, has_edge G u v) finset.univ).card = r

-- Definition of 2-dominating set.
def is_2_dominating_set (G : Type*) (S : finset G) :=
  ∀ v ∈ (finset.univ \ S), (finset.filter (λ u : G, u ∈ S ∧ u ≠ v ∧ has_edge G u v) finset.univ).card ≥ 2

noncomputable def ln : ℕ → ℝ
| r => real.log r

theorem regular_graph_has_small_2_dominating_set
  (G : Type*) [fintype G] [decidable_eq G]
  (r n : ℕ) (hG : fintype.card G = n)
  (hr_valid : r ≥ 3)
  (hr_reg : is_regular_graph G r)
  : ∃ (S : finset G), is_2_dominating_set G S ∧ S.card ≤ (n * (1 + ln r) / r) :=
begin
  sorry
end

end regular_graph_has_small_2_dominating_set_l493_493783


namespace units_digit_of_150_factorial_is_zero_l493_493444

-- Define the conditions for the problem
def is_units_digit_zero_of_factorial (n : ℕ) : Prop :=
  n = 150 → (Nat.factorial n % 10 = 0)

-- The statement of the proof problem
theorem units_digit_of_150_factorial_is_zero : is_units_digit_zero_of_factorial 150 :=
  sorry

end units_digit_of_150_factorial_is_zero_l493_493444


namespace distance_to_first_museum_l493_493691

theorem distance_to_first_museum (x : ℝ) 
  (dist_second_museum : ℝ) 
  (total_distance : ℝ) 
  (h1 : dist_second_museum = 15) 
  (h2 : total_distance = 40) 
  (h3 : 2 * x + 2 * dist_second_museum = total_distance) : x = 5 :=
by 
  sorry

end distance_to_first_museum_l493_493691


namespace units_digit_150_factorial_is_zero_l493_493454

theorem units_digit_150_factorial_is_zero :
  (nat.trailing_digits (nat.factorial 150) 1) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l493_493454


namespace ratio_flowers_l493_493146

theorem ratio_flowers (flowers_monday flowers_tuesday flowers_week total_flowers flowers_friday : ℕ)
    (h_monday : flowers_monday = 4)
    (h_tuesday : flowers_tuesday = 8)
    (h_total : total_flowers = 20)
    (h_week : total_flowers = flowers_monday + flowers_tuesday + flowers_friday) :
    flowers_friday / flowers_monday = 2 :=
by
  sorry

end ratio_flowers_l493_493146


namespace minimum_eccentricity_l493_493966

-- Definitions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def circle (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

-- Main theorem stating the minimum eccentricity
theorem minimum_eccentricity (a b : ℝ) (P : ℝ × ℝ) (M N : ℝ × ℝ) :
  (∃ (xP yP : ℝ), ellipse a b xP yP ∧ P = (xP, yP) ∧
  ∃ (xM yM xN yN : ℝ), circle b xM yM ∧ circle b xN yN ∧ 
  (angle (xM, yM) P (xN, yN) = π / 2)) → 
  sqrt 2 / 2 ≤ sqrt (1 - b^2 / a^2) :=
sorry

end minimum_eccentricity_l493_493966


namespace initial_pokemon_cards_l493_493688

theorem initial_pokemon_cards (x : ℤ) (h : x - 9 = 4) : x = 13 :=
by
  sorry

end initial_pokemon_cards_l493_493688


namespace clock_angle_8_30_l493_493379

theorem clock_angle_8_30 
  (angle_per_hour_mark : ℝ := 30)
  (angle_per_minute_mark : ℝ := 6)
  (hour_hand_angle_8 : ℝ := 8 * angle_per_hour_mark)
  (half_hour_movement : ℝ := 0.5 * angle_per_hour_mark)
  (hour_hand_angle_8_30 : ℝ := hour_hand_angle_8 + half_hour_movement)
  (minute_hand_angle_30 : ℝ := 30 * angle_per_minute_mark) :
  abs (hour_hand_angle_8_30 - minute_hand_angle_30) = 75 :=
by
  sorry

end clock_angle_8_30_l493_493379


namespace find_nested_f_value_l493_493248

def f (x : ℝ) : ℝ :=
  if x > 0 then real.log x / real.log 2 else 3^x

theorem find_nested_f_value : f (f (1 / 8)) = 1 / 27 := 
  by 
  sorry

end find_nested_f_value_l493_493248


namespace other_endpoint_of_diameter_l493_493873

theorem other_endpoint_of_diameter (center endpoint : ℝ × ℝ) (hc : center = (1, 2)) (he : endpoint = (4, 6)) :
    ∃ other_endpoint : ℝ × ℝ, other_endpoint = (-2, -2) :=
by
  sorry

end other_endpoint_of_diameter_l493_493873


namespace proof_P3_5_l493_493844

noncomputable def P : ℕ × ℕ → ℚ
| (0, 0)       := 1
| (a + 1, 0)   := 0
| (0, b + 1)   := 0
| (a + 1, b + 1) :=
  1 / 2 * P (a, b + 1) + 1 / 4 * P (a + 1, b) + 1 / 4 * P (a, b)

def Q (a b : ℕ) (m n : ℕ) := 
  P (a, b) = m / 4 ^ n ∧ gcd m 4 = 1

theorem proof_P3_5 (m n : ℕ) :
  ∃ m n, Q 3 5 m n := 
sorry

end proof_P3_5_l493_493844


namespace hawks_prob_win_at_least_four_l493_493745

-- Define the parameters
def p : ℝ := 0.5
def n : ℕ := 7

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k 

-- Define the binomial probability function
def binom_prob (k : ℕ) (p : ℝ) (n : ℕ) : ℝ := (binom n k) * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the total probability of winning at least k matches
def prob_at_least (k : ℕ) (p : ℝ) (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), if i ≥ k then binom_prob i p n else 0

-- The main statement we want to prove
theorem hawks_prob_win_at_least_four : prob_at_least 4 p n = 1 / 2 := by
   sorry

end hawks_prob_win_at_least_four_l493_493745


namespace problem_statement_l493_493652

noncomputable section

def a : ℝ := log 3 (1 / 2)
def b : ℝ := log (1 / 4) 10
def c : ℝ := log 2 (1 / 3)

theorem problem_statement : b < c ∧ c < a :=
by
  sorry

end problem_statement_l493_493652


namespace units_digit_150_factorial_is_zero_l493_493452

theorem units_digit_150_factorial_is_zero :
  (nat.trailing_digits (nat.factorial 150) 1) = 0 :=
by sorry

end units_digit_150_factorial_is_zero_l493_493452


namespace min_satisfies_condition_only_for_x_eq_1_div_4_l493_493937

theorem min_satisfies_condition_only_for_x_eq_1_div_4 (x : ℝ) (hx_nonneg : 0 ≤ x) :
  (min (Real.sqrt x) (min (x^2) x) = 1/16) ↔ (x = 1/4) :=
by sorry

end min_satisfies_condition_only_for_x_eq_1_div_4_l493_493937


namespace statement_a_correct_statement_c_correct_statement_d_correct_combined_statements_l493_493084

-- Definitions from the problem conditions
def statement_a (p : Prop) : Prop :=
  ∃ x : ℝ, x^2 + 2 * x + 2 < 0

def statement_c (a : ℝ) : Prop :=
  ∀ x ∈ Set.Ioo 2 3, 3 * x - a < 0

def statement_d (m : ℝ) : Prop :=
  ∃x, x^2 - 2*x + m = 0 ∧  x > 0 ∧ ∃y, y^2 - 2*y + m = 0 ∧ y < 0

-- The proof problem statements
theorem statement_a_correct :
  (∃ x : ℝ, x^2 + 2 * x + 2 < 0) = ¬(∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0) := 
  sorry

theorem statement_c_correct (a : ℝ) :
  (∀ x ∈ Set.Ioo 2 3, 3 * x - a < 0) → a ≥ 9 := 
  sorry

theorem statement_d_correct (m : ℝ) :
  (∃ x, x^2 - 2 * x + m = 0 ∧  x > 0 ∧ ∃ y, y^2 - 2 * y + m = 0 ∧ y < 0) ↔ m < 0 :=
  sorry

-- Combined results
theorem combined_statements :
  let A := statement_a_correct
  let C := statement_c_correct
  let D := statement_d_correct
  A ∧ C ∧ D :=
  sorry

end statement_a_correct_statement_c_correct_statement_d_correct_combined_statements_l493_493084


namespace captain_age_eq_your_age_l493_493648

-- Represent the conditions as assumptions
variables (your_age : ℕ) -- You, the captain, have an age as a natural number

-- Define the statement
theorem captain_age_eq_your_age (H_cap : ∀ captain, captain = your_age) : ∀ captain, captain = your_age := by
  sorry

end captain_age_eq_your_age_l493_493648


namespace hexagon_centroids_are_symmetric_l493_493166

noncomputable def centroid (A B C : Point) : Point := 
  -- definition of centroid

structure Hexagon :=
  (A B C D E F : Point)
  (convex : convex_hexagon A B C D E F)

def centroids_of_diagonals (hex : Hexagon) : (Point × Point × Point × Point × Point × Point) :=
  let S₁ := centroid hex.A hex.C hex.B
  let S₂ := centroid hex.B hex.D hex.C
  let S₃ := centroid hex.C hex.E hex.D
  let S₄ := centroid hex.D hex.F hex.E
  let S₅ := centroid hex.E hex.A hex.F
  let S₆ := centroid hex.F hex.B hex.A
  (S₁, S₂, S₃, S₄, S₅, S₆)

def is_central_symmetric_about (P Q R S T U O : Point) : Prop :=
  -- definition of central symmetry about the point O

theorem hexagon_centroids_are_symmetric (hex : Hexagon) :
  let (S₁, S₂, S₃, S₄, S₅, S₆) := centroids_of_diagonals hex in
  ∃ O : Point, is_central_symmetric_about S₁ S₂ S₃ S₄ S₅ S₆ O := 
sorry

end hexagon_centroids_are_symmetric_l493_493166


namespace P_subsetneq_M_l493_493349

def M := {x : ℝ | x > 1}
def P := {x : ℝ | x^2 - 6*x + 9 = 0}

theorem P_subsetneq_M : P ⊂ M := by
  sorry

end P_subsetneq_M_l493_493349


namespace perimeter_of_square_l493_493020

theorem perimeter_of_square (s : ℝ) (h : s^2 = s * Real.sqrt 2) (h_ne_zero : s ≠ 0) :
    4 * s = 4 * Real.sqrt 2 := by
  sorry

end perimeter_of_square_l493_493020


namespace work_problem_l493_493501

theorem work_problem (A B C : ℝ) (hB : B = 3) (h1 : 1 / B + 1 / C = 1 / 2) (h2 : 1 / A + 1 / C = 1 / 2) : A = 3 := by
  sorry

end work_problem_l493_493501


namespace count_valid_three_digit_numbers_l493_493995

def digits_sum_to_14 (n : ℕ) : Prop := 
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 + d2 + d3 = 14

def not_divisible_by_5 (n : ℕ) : Prop := 
  n % 5 ≠ 0

def first_digit_equals_last_digit (n : ℕ) : Prop :=
  let d1 := n / 100
  let d3 := n % 10
  d1 = d3

def valid_three_digit_number (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000 ∧ digits_sum_to_14 n ∧ not_divisible_by_5 n ∧ first_digit_equals_last_digit n

theorem count_valid_three_digit_numbers : 
  {n : ℕ | valid_three_digit_number n}.to_finset.card = 4 := 
sorry

end count_valid_three_digit_numbers_l493_493995


namespace remainder_is_90_l493_493755

theorem remainder_is_90:
  let larger_number := 2982
  let smaller_number := 482
  let quotient := 6
  (larger_number - smaller_number = 2500) ∧ 
  (larger_number = quotient * smaller_number + r) →
  (r = 90) :=
by
  sorry

end remainder_is_90_l493_493755


namespace hyperbola_solution_l493_493964

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : Prop :=
  ∃ (x y : ℝ), (x = 2) ∧ (y = 0) ∧ 
    (∀ (x y : ℝ), y = 2 * x - 4 → 
      ∀ (x y : ℝ), (x ≠ 2 ∨ y ≠ 0) → False) ∧
      (b = 2 * a)

theorem hyperbola_solution : hyperbola_equation (sqrt (4/5)) (sqrt (16/5)) (by norm_num) (by norm_num) :=
  sorry

end hyperbola_solution_l493_493964


namespace find_k_eq_9_l493_493574

theorem find_k_eq_9 (α β : ℝ) : 
  (2 * sin (α + β) + csc (α + β))^2 + (2 * cos (α + β) + sec (α + β))^2 = 
  9 + tan (α + β)^2 + cot (α + β)^2 := 
sorry

end find_k_eq_9_l493_493574


namespace ribbon_needed_for_gate_l493_493378

noncomputable def radius (area : ℝ) (pi : ℝ) : ℝ :=
  (area * 7 / pi).sqrt

noncomputable def circumference (r : ℝ) (pi : ℝ) : ℝ :=
  2 * pi * r

noncomputable def ribbon_length (area : ℝ) (pi : ℝ) (additional_length : ℝ) : ℝ :=
  let r := radius area pi
  let c := circumference r pi
  c + additional_length

theorem ribbon_needed_for_gate :
  ribbon_length 246 (22 / 7) 5 ≈ 60.57 ≈ 60.57 :=
by
  sorry

end ribbon_needed_for_gate_l493_493378


namespace polyhedron_property_l493_493213

-- Step 1: Define a centrally symmetric, convex polyhedron type
structure Polyhedron :=
  (vertices : Set Point)
  (faces : Set (Set Point))
  (is_convex : MConvexHull vertices = some vertices)
  (is_symmetric : ∀ v ∈ vertices, (-v) ∈ vertices)

-- Step 2: Define the condition for any two vertices
axiom faces_condition : ∀ (P : Polyhedron) (X Y : Point), 
  X ∈ P.vertices → Y ∈ P.vertices →
  (Y = -X ∨ ∃ face ∈ P.faces, X ∈ face ∧ Y ∈ face)

-- We need to show the conclusion based on the given conditions
theorem polyhedron_property (P : Polyhedron)
  (H1 : ∀ X Y, X ∈ P.vertices → Y ∈ P.vertices → (Y = -X ∨ ∃ face ∈ P.faces, X ∈ face ∧ Y ∈ face)) :
  (∃ edges : Set (Point × Point), IsParallelepiped P.edges) ∨
  (∃ Q : Polyhedron, Q.vertices = Set.midpoints (faces_vertices of P)) := 
sorry

end polyhedron_property_l493_493213


namespace multiplication_simplify_l493_493164

theorem multiplication_simplify :
  12 * (1 / 8) * 32 = 48 := 
sorry

end multiplication_simplify_l493_493164


namespace quadratic_distinct_roots_l493_493618

theorem quadratic_distinct_roots (p q : ℝ) (hp : 0 < p) (hq : 0 < q) (hpq_pos : p^2 - 4 * q > 0) (hqp_pos : q^2 - 4 * p > 0) : 
  let a := p + q in
  (a^2 - 8 * a) > 0 :=
by sorry

end quadratic_distinct_roots_l493_493618


namespace blocks_given_by_father_l493_493869

theorem blocks_given_by_father :
  ∀ (blocks_original total_blocks blocks_given : ℕ), 
  blocks_original = 2 →
  total_blocks = 8 →
  blocks_given = total_blocks - blocks_original →
  blocks_given = 6 :=
by
  intros blocks_original total_blocks blocks_given h1 h2 h3
  sorry

end blocks_given_by_father_l493_493869


namespace period_tan_2x_l493_493792

theorem period_tan_2x (x : ℝ) : (∃ T : ℝ, ∀ x : ℝ, tan (2 * (x + T)) = tan (2 * x)) → T = π / 2 :=
by
  sorry

end period_tan_2x_l493_493792


namespace exponentiation_addition_logarithmic_exponentiation_l493_493546

theorem exponentiation_addition :
  32 * 2^(2 / 3) + (1 / 4)^(-1 / 2) = 4 := 
  sorry

theorem logarithmic_exponentiation :
  2^(Real.log 3 / Real.log 2 + (Real.log 9 / Real.log 4)) = 9 :=
  sorry

end exponentiation_addition_logarithmic_exponentiation_l493_493546


namespace directrices_distance_l493_493381

theorem directrices_distance (a b : ℝ) (h₀ : a^2 = 9) (h₁ : b^2 = 8) :
  let c := real.sqrt (a^2 - b^2)
  in abs ((a^2) / c - (-(a^2) / c)) = 18 :=
by
  sorry

end directrices_distance_l493_493381


namespace increasing_function_range_l493_493939

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < -1 then (-a + 4) * x - 3 * a else x^2 + a * x - 8

theorem increasing_function_range :
  (∀ x1 x2, x1 < x2 → f x1 a ≤ f x2 a) ↔ (3 ≤ a ∧ a < 4) := 
sorry

end increasing_function_range_l493_493939


namespace measure_angle_C_l493_493665

theorem measure_angle_C (a b c : ℝ) (S : ℝ) (h1 : S = 1 / 4 * (a^2 + b^2 - c^2)) (h2 : S = 1 / 2 * a * b * Real.sin 45) :
  ∠C = 45 :=
sorry

end measure_angle_C_l493_493665


namespace find_first_number_l493_493017

theorem find_first_number 
  (first_number second_number hcf lcm : ℕ) 
  (hCF_condition : hcf = 12) 
  (lCM_condition : lcm = 396) 
  (one_number_condition : first_number = 99) 
  (relation_condition : first_number * second_number = hcf * lcm) : 
  second_number = 48 :=
by
  sorry

end find_first_number_l493_493017


namespace units_digit_factorial_150_l493_493459

theorem units_digit_factorial_150 : (nat.factorial 150) % 10 = 0 :=
sorry

end units_digit_factorial_150_l493_493459


namespace product_divisible_by_four_l493_493737

theorem product_divisible_by_four (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 2017) (h2 : 1 ≤ b ∧ b ≤ 2017) (h3 : 1 ≤ c ∧ c ≤ 2017) 
(h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  let p : ℚ := /- the computed probability -/ in
  1 / 8 < p ∧ p < 1 / 3 :=
sorry

end product_divisible_by_four_l493_493737


namespace tan_beta_minus_2alpha_l493_493656

theorem tan_beta_minus_2alpha (α β : Real) 
  (h1 : Real.cot α = 2) 
  (h2 : Real.tan (α - β) = -2 / 5) : 
  Real.tan (β - 2 * α) = -1 / 12 := 
sorry

end tan_beta_minus_2alpha_l493_493656


namespace total_volume_of_cubes_l493_493160

theorem total_volume_of_cubes 
  (Carl_cubes : ℕ)
  (Carl_side_length : ℕ)
  (Kate_cubes : ℕ)
  (Kate_side_length : ℕ)
  (h1 : Carl_cubes = 8)
  (h2 : Carl_side_length = 2)
  (h3 : Kate_cubes = 3)
  (h4 : Kate_side_length = 3) :
  Carl_cubes * Carl_side_length ^ 3 + Kate_cubes * Kate_side_length ^ 3 = 145 :=
by
  sorry

end total_volume_of_cubes_l493_493160


namespace length_BC_l493_493310

-- Conditions as definitions
def AB : ℝ := 5
def AC : ℝ := 8
def AM : ℝ := 5
def height_from_A : ℝ := 4

-- Proof statement to prove BC = sqrt(78)
theorem length_BC :
  ∃ BC : ℝ, (AM = (1/2) * real.sqrt(2 * AB^2 + 2 * AC^2 - BC^2)) ∧
            (2 * BC) = (1/2 * BC * height_from_A) ∧
            BC = real.sqrt 78 :=
sorry

end length_BC_l493_493310


namespace square_perimeter_l493_493136

-- First, declare the side length of the square (rectangle)
variable (s : ℝ)

-- State the conditions: the area is 484 cm^2 and it's a square
axiom area_condition : s^2 = 484
axiom is_square : ∀ (s : ℝ), s > 0

-- Define the perimeter of the square
def perimeter (s : ℝ) : ℝ := 4 * s

-- State the theorem: perimeter == 88 given the conditions
theorem square_perimeter : perimeter s = 88 :=
by 
  -- Prove the statement given the axiom 'area_condition'
  sorry

end square_perimeter_l493_493136


namespace boys_under_six_ratio_l493_493773

theorem boys_under_six_ratio (total_students : ℕ) (two_third_boys : (2/3 : ℚ) * total_students = 25) (boys_under_six : ℕ) (boys_under_six_eq : boys_under_six = 19) :
  boys_under_six / 25 = 19 / 25 :=
by
  sorry

end boys_under_six_ratio_l493_493773


namespace dilation_matrix_l493_493585

theorem dilation_matrix {R : Type*} [Field R] : 
  let T : Matrix (Fin 3) (Fin 3) R := !![
    [1/2, 0, 0],
    [0, 1/2, 0],
    [0, 0, 1/2]]
  in ∀ (v : Matrix (Fin 3) (Fin 1) R), (T.mul_vec v) = (fun i => (1/2) * v i) :=
by
  sorry

end dilation_matrix_l493_493585


namespace find_lambda_l493_493208

variable (V : Type) [AddCommGroup V] [Module ℝ V]
variables (A B C : V)

theorem find_lambda (h1 : A - B = 2 * (A - C)) (h2 : A - B = λ * (B - C)) : λ = -2 :=
sorry

end find_lambda_l493_493208


namespace max_S_n_sum_b_n_terms_l493_493948

-- Given sequence {a_n}
def a_n (n : ℕ) : ℤ := 17 - 3 * n

-- Sum of first n terms of sequence {a_n}
def S_n (n : ℕ) : ℤ := (n * (a_n 1 + a_n n)) / 2

-- Maximum value of S_n
theorem max_S_n : ∃ n : ℕ, S_n n = 40 :=
sorry

-- Given sequence {b_n}
def b_n (n : ℕ) : ℤ := abs (a_n n)

-- Sum of first n terms of sequence {b_n}
def T_n (n : ℕ) : ℤ :=
if n ≤ 5 then
    -(3 / 2) * n^2 + (31 / 2) * n
else
    (3 / 2) * n^2 - (31 / 2) * n + 80

-- Theorem for T_n
theorem sum_b_n_terms (n : ℕ) : T_n n = 
if n ≤ 5 then
    -(3 / 2) * n^2 + (31 / 2) * n
else
    (3 / 2) * n^2 - (31 / 2) * n + 80 :=
sorry

end max_S_n_sum_b_n_terms_l493_493948


namespace sequence_is_integer_sequence_divisible_by_3_l493_493003

def sequence (n : ℤ) : ℤ :=
  ((2 + Real.sqrt 3)^n - (2 - Real.sqrt 3)^n) / (2 * Real.sqrt 3)

theorem sequence_is_integer (n : ℤ) : ∃ k : ℤ, sequence n = k :=
sorry

theorem sequence_divisible_by_3 (n : ℤ) : (sequence n % 3 = 0) ↔ (n % 3 = 0) :=
sorry

end sequence_is_integer_sequence_divisible_by_3_l493_493003


namespace solution_is_zero_constant_l493_493554

noncomputable def is_solution (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f ((int.floor x : ℝ) * y) = f x * (int.floor (f y) : ℝ)

theorem solution_is_zero_constant {f : ℝ → ℝ} (h : is_solution f) :
  f = (λ x, 0) ∨ ∃ c : ℝ, f = (λ x, c) ∧ int.floor c = 1 :=
sorry

end solution_is_zero_constant_l493_493554
