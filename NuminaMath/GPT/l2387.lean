import Mathlib

namespace red_peaches_l2387_238727

theorem red_peaches (R G : ℕ) (h1 : G = 11) (h2 : G = R + 6) : R = 5 :=
by {
  sorry
}

end red_peaches_l2387_238727


namespace tan_seven_pi_over_four_l2387_238702

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 := 
by
  -- In this case, we are proving a specific trigonometric identity
  sorry

end tan_seven_pi_over_four_l2387_238702


namespace sum_of_areas_of_triangles_l2387_238777

theorem sum_of_areas_of_triangles 
  (AB BG GE DE : ℕ) 
  (A₁ A₂ : ℕ)
  (H1 : AB = 2) 
  (H2 : BG = 3) 
  (H3 : GE = 4) 
  (H4 : DE = 5) 
  (H5 : 3 * A₁ + 4 * A₂ = 48)
  (H6 : 9 * A₁ + 5 * A₂ = 102) : 
  1 * AB * A₁ / 2 + 1 * DE * A₂ / 2 = 23 :=
by
  sorry

end sum_of_areas_of_triangles_l2387_238777


namespace interest_difference_correct_l2387_238707

-- Define the basic parameters and constants
def principal : ℝ := 147.69
def rate : ℝ := 0.15
def time1 : ℝ := 3.5
def time2 : ℝ := 10
def interest1 : ℝ := principal * rate * time1
def interest2 : ℝ := principal * rate * time2
def difference : ℝ := 143.998

-- Theorem statement: The difference between the interests is approximately Rs. 143.998
theorem interest_difference_correct :
  interest2 - interest1 = difference := sorry

end interest_difference_correct_l2387_238707


namespace min_value_l2387_238716

theorem min_value (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (m n : ℝ) (h3 : m > 0) (h4 : n > 0) 
(h5 : m + 4 * n = 1) : 
  1 / m + 4 / n ≥ 25 :=
by
  sorry

end min_value_l2387_238716


namespace part_1_part_2_1_part_2_2_l2387_238784

variable {k x : ℝ}
def y (k : ℝ) (x : ℝ) := k * x^2 - 2 * k * x + 2 * k - 1

theorem part_1 (k : ℝ) : (∀ x, y k x ≥ 4 * k - 2) ↔ (0 ≤ k ∧ k ≤ 1 / 3) := by
  sorry

theorem part_2_1 (k : ℝ) : ¬∃ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1^2 + x2^2 = 3 * x1 * x2 - 4 := by
  sorry

theorem part_2_2 (k : ℝ) : (∀ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1 > 0 ∧ x2 > 0) ↔ (1 / 2 < k ∧ k < 1) := by
  sorry

end part_1_part_2_1_part_2_2_l2387_238784


namespace perimeter_of_ABCD_l2387_238766

theorem perimeter_of_ABCD
  (AD BC AB CD : ℕ)
  (hAD : AD = 4)
  (hAB : AB = 5)
  (hBC : BC = 10)
  (hCD : CD = 7)
  (hAD_lt_BC : AD < BC) :
  AD + AB + BC + CD = 26 :=
by
  -- Proof will be provided here.
  sorry

end perimeter_of_ABCD_l2387_238766


namespace find_weight_A_l2387_238703

noncomputable def weight_of_A (a b c d e : ℕ) : Prop :=
  (a + b + c) / 3 = 84 ∧
  (a + b + c + d) / 4 = 80 ∧
  e = d + 5 ∧
  (b + c + d + e) / 4 = 79 →
  a = 77

theorem find_weight_A (a b c d e : ℕ) : weight_of_A a b c d e :=
by
  sorry

end find_weight_A_l2387_238703


namespace max_value_expression_l2387_238778

theorem max_value_expression : ∃ s_max : ℝ, 
  (∀ s : ℝ, -3 * s^2 + 24 * s - 7 ≤ -3 * s_max^2 + 24 * s_max - 7) ∧
  (-3 * s_max^2 + 24 * s_max - 7 = 41) :=
sorry

end max_value_expression_l2387_238778


namespace largest_house_number_l2387_238737

theorem largest_house_number (house_num : ℕ) : 
  house_num ≤ 981 :=
  sorry

end largest_house_number_l2387_238737


namespace scientific_notation_of_3395000_l2387_238744

theorem scientific_notation_of_3395000 :
  3395000 = 3.395 * 10^6 :=
sorry

end scientific_notation_of_3395000_l2387_238744


namespace min_value_m_n_l2387_238736

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem min_value_m_n 
  (a : ℝ) (m n : ℝ)
  (h_a_pos : a > 0) (h_a_ne1 : a ≠ 1)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_line_eq : 2 * m + n = 1) :
  m + n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_m_n_l2387_238736


namespace find_x_l2387_238713

noncomputable def approx_equal (a b : ℝ) (ε : ℝ) : Prop := abs (a - b) < ε

theorem find_x :
  ∃ x : ℝ, x + Real.sqrt 68 = 24 ∧ approx_equal x 15.753788749 0.0001 :=
sorry

end find_x_l2387_238713


namespace polygon_sides_l2387_238787

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) (h2 : n > 2) : n = 8 := by
  -- Conditions given:
  -- h1: (n - 2) * 180 = 3 * 360
  -- h2: n > 2
  sorry

end polygon_sides_l2387_238787


namespace nell_initial_cards_l2387_238745

theorem nell_initial_cards (cards_given cards_left total_cards : ℕ)
  (h1 : cards_given = 301)
  (h2 : cards_left = 154)
  (h3 : total_cards = cards_given + cards_left) :
  total_cards = 455 := by
  rw [h1, h2] at h3
  exact h3

end nell_initial_cards_l2387_238745


namespace grace_pennies_l2387_238767

theorem grace_pennies :
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  dimes * dime_value + coins * coin_value = 150 :=
by
  let dime_value := 10
  let coin_value := 5
  let dimes := 10
  let coins := 10
  sorry

end grace_pennies_l2387_238767


namespace largest_y_coordinate_ellipse_l2387_238718

theorem largest_y_coordinate_ellipse:
  (∀ x y : ℝ, (x^2 / 49) + ((y + 3)^2 / 25) = 1 → y ≤ 2)  ∧ 
  (∃ x : ℝ, (x^2 / 49) + ((2 + 3)^2 / 25) = 1) := sorry

end largest_y_coordinate_ellipse_l2387_238718


namespace minimum_degree_g_l2387_238763

-- Define the degree function for polynomials
noncomputable def degree (p : Polynomial ℤ) : ℕ := p.natDegree

-- Declare the variables and conditions for the proof
variables (f g h : Polynomial ℤ)
variables (deg_f : degree f = 10) (deg_h : degree h = 12)
variable (eqn : 2 * f + 5 * g = h)

-- State the main theorem for the problem
theorem minimum_degree_g : degree g ≥ 12 :=
    by sorry -- Proof to be provided

end minimum_degree_g_l2387_238763


namespace concentration_of_concentrated_kola_is_correct_l2387_238712

noncomputable def concentration_of_concentrated_kola_added 
  (initial_volume : ℝ) (initial_pct_sugar : ℝ)
  (sugar_added : ℝ) (water_added : ℝ)
  (required_pct_sugar : ℝ) (new_sugar_volume : ℝ) : ℝ :=
  let initial_sugar := initial_volume * initial_pct_sugar / 100
  let total_sugar := initial_sugar + sugar_added
  let new_total_volume := initial_volume + sugar_added + water_added
  let total_volume_with_kola := new_total_volume + (new_sugar_volume / required_pct_sugar * 100 - total_sugar) / (100 / required_pct_sugar - 1)
  total_volume_with_kola - new_total_volume

noncomputable def problem_kola : ℝ :=
  concentration_of_concentrated_kola_added 340 7 3.2 10 7.5 27

theorem concentration_of_concentrated_kola_is_correct : 
  problem_kola = 6.8 :=
by
  unfold problem_kola concentration_of_concentrated_kola_added
  sorry

end concentration_of_concentrated_kola_is_correct_l2387_238712


namespace part1_part2_l2387_238747

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | 1 < x ∧ x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 ≤ 0}
def H (a : ℝ) : Set ℝ := {x | abs (x - a) <= 2}

def symdiff (A B : Set ℝ) : Set ℝ := A ∩ (U \ B)

theorem part1 :
  symdiff M N = {x | 1 < x ∧ x < 2} ∧
  symdiff N M = {x | 3 ≤ x ∧ x ≤ 4} :=
by
  sorry

theorem part2 (a : ℝ) :
  symdiff (symdiff N M) (H a) =
    if a ≥ 4 ∨ a ≤ -1 then {x | 1 < x ∧ x < 2}
    else if 3 < a ∧ a < 4 then {x | 1 < x ∧ x < a - 2}
    else if -1 < a ∧ a < 0 then {x | a + 2 < x ∧ x < 2}
    else ∅ :=
by
  sorry

end part1_part2_l2387_238747


namespace squared_sum_inverse_l2387_238722

theorem squared_sum_inverse (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 :=
by
  sorry

end squared_sum_inverse_l2387_238722


namespace incorrect_statement_D_l2387_238733

/-
Define the conditions for the problem:
- A prism intersected by a plane.
- The intersection of a sphere and a plane when the plane is less than the radius.
- The intersection of a plane parallel to the base of a circular cone.
- The geometric solid formed by rotating a right triangle around one of its sides.
- The incorrectness of statement D.
-/

noncomputable def intersect_prism_with_plane (prism : Type) (plane : Type) : Prop := sorry

noncomputable def sphere_intersection (sphere_radius : ℝ) (distance_to_plane : ℝ) : Type := sorry

noncomputable def cone_intersection (cone : Type) (plane : Type) : Type := sorry

noncomputable def rotation_result (triangle : Type) (side : Type) : Type := sorry

theorem incorrect_statement_D :
  ¬(rotation_result RightTriangle Side = Cone) :=
sorry

end incorrect_statement_D_l2387_238733


namespace calculate_total_area_l2387_238776

theorem calculate_total_area :
  let height1 := 7
  let width1 := 6
  let width2 := 4
  let height2 := 5
  let height3 := 1
  let width3 := 2
  let width4 := 5
  let height4 := 6
  let area1 := width1 * height1
  let area2 := width2 * height2
  let area3 := height3 * width3
  let area4 := width4 * height4
  area1 + area2 + area3 + area4 = 94 := by
  sorry

end calculate_total_area_l2387_238776


namespace solve_equation_l2387_238760

theorem solve_equation :
  ∃ x : ℝ, (20 / (x^2 - 9) - 3 / (x + 3) = 1) ↔ (x = -8) ∨ (x = 5) :=
by
  sorry

end solve_equation_l2387_238760


namespace wilted_flowers_are_18_l2387_238726

def picked_flowers := 53
def flowers_per_bouquet := 7
def bouquets_after_wilted := 5

def flowers_left := bouquets_after_wilted * flowers_per_bouquet
def flowers_wilted : ℕ := picked_flowers - flowers_left

theorem wilted_flowers_are_18 : flowers_wilted = 18 := by
  sorry

end wilted_flowers_are_18_l2387_238726


namespace orange_gumdrops_after_replacement_l2387_238782

noncomputable def total_gumdrops : ℕ :=
  100

noncomputable def initial_orange_gumdrops : ℕ :=
  10

noncomputable def initial_blue_gumdrops : ℕ :=
  40

noncomputable def replaced_blue_gumdrops : ℕ :=
  initial_blue_gumdrops / 3

theorem orange_gumdrops_after_replacement : 
  (initial_orange_gumdrops + replaced_blue_gumdrops) = 23 :=
by
  sorry

end orange_gumdrops_after_replacement_l2387_238782


namespace poly_sequence_correct_l2387_238770

-- Sequence of polynomials defined recursively
def f : ℕ → ℕ → ℕ 
| 0, x => 1
| 1, x => 1 + x 
| (k + 1), x => ((x + 1) * f (k) (x) - (x - k) * f (k - 1) (x)) / (k + 1)

-- Prove f(k, k) = 2^k for all k ≥ 0
theorem poly_sequence_correct (k : ℕ) : f k k = 2 ^ k := by
  sorry

end poly_sequence_correct_l2387_238770


namespace total_number_of_posters_l2387_238773

theorem total_number_of_posters : 
  ∀ (P : ℕ), 
  (2 / 5 : ℚ) * P + (1 / 2 : ℚ) * P + 5 = P → 
  P = 50 :=
by
  intro P
  intro h
  sorry

end total_number_of_posters_l2387_238773


namespace find_original_price_l2387_238798

variable (original_price : ℝ)
variable (final_price : ℝ) (first_reduction_rate : ℝ) (second_reduction_rate : ℝ)

theorem find_original_price :
  final_price = 15000 →
  first_reduction_rate = 0.30 →
  second_reduction_rate = 0.40 →
  0.42 * original_price = final_price →
  original_price = 35714 := by
  intros h1 h2 h3 h4
  sorry

end find_original_price_l2387_238798


namespace geometric_sequence_common_ratio_l2387_238743

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a n = a 2 * q ^ (n - 2)) ∧ a 2 = 2 ∧ a 6 = 1 / 8 →
  (q = 1 / 2 ∨ q = -1 / 2) :=
by
  sorry

end geometric_sequence_common_ratio_l2387_238743


namespace average_of_remaining_three_numbers_l2387_238758

noncomputable def avg_remaining_three_numbers (avg_12 : ℝ) (avg_4 : ℝ) (avg_3 : ℝ) (avg_2 : ℝ) : ℝ :=
  let sum_12 := 12 * avg_12
  let sum_4 := 4 * avg_4
  let sum_3 := 3 * avg_3
  let sum_2 := 2 * avg_2
  let sum_9 := sum_4 + sum_3 + sum_2
  let sum_remaining_3 := sum_12 - sum_9
  sum_remaining_3 / 3

theorem average_of_remaining_three_numbers :
  avg_remaining_three_numbers 6.30 5.60 4.90 7.25 = 8 :=
by {
  sorry
}

end average_of_remaining_three_numbers_l2387_238758


namespace exists_primes_sum_2024_with_one_gt_1000_l2387_238742

open Nat

-- Definition of primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Conditions given in the problem
def sum_primes_eq_2024 (p q : ℕ) : Prop :=
  p + q = 2024 ∧ is_prime p ∧ is_prime q

def at_least_one_gt_1000 (p q : ℕ) : Prop :=
  p > 1000 ∨ q > 1000

-- The theorem to be proved
theorem exists_primes_sum_2024_with_one_gt_1000 :
  ∃ (p q : ℕ), sum_primes_eq_2024 p q ∧ at_least_one_gt_1000 p q :=
sorry

end exists_primes_sum_2024_with_one_gt_1000_l2387_238742


namespace arithmetic_sequence_sum_l2387_238746

noncomputable def first_21_sum (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) : ℝ :=
  let a1 := a 1
  let a21 := a 21
  21 * (a1 + a21) / 2

theorem arithmetic_sequence_sum
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_symmetry : ∀ x, f (x + 1) = f (-(x + 1)))
  (h_monotonic : ∀ x y, 1 < x → x < y → f x < f y)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_f_eq : f (a 4) = f (a 18))
  (h_non_zero_diff : d ≠ 0) :
  first_21_sum f a d = 21 := by
  sorry

end arithmetic_sequence_sum_l2387_238746


namespace fractional_eq_solution_l2387_238729

theorem fractional_eq_solution (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) :
  (1 / (x - 1) = 2 / (x - 2)) → (x = 2) :=
by
  sorry

end fractional_eq_solution_l2387_238729


namespace initial_value_subtract_perfect_square_l2387_238709

theorem initial_value_subtract_perfect_square :
  ∃ n : ℕ, n^2 = 308 - 139 :=
by
  sorry

end initial_value_subtract_perfect_square_l2387_238709


namespace polygon_interior_angles_540_implies_pentagon_l2387_238735

theorem polygon_interior_angles_540_implies_pentagon
  (n : ℕ) (H: 180 * (n - 2) = 540) : n = 5 :=
sorry

end polygon_interior_angles_540_implies_pentagon_l2387_238735


namespace points_A_B_D_collinear_l2387_238771

variable (a b : ℝ)

theorem points_A_B_D_collinear
  (AB : ℝ × ℝ := (a, 5 * b))
  (BC : ℝ × ℝ := (-2 * a, 8 * b))
  (CD : ℝ × ℝ := (3 * a, -3 * b)) :
  AB = (BC.1 + CD.1, BC.2 + CD.2) := 
by
  sorry

end points_A_B_D_collinear_l2387_238771


namespace trapezium_area_l2387_238762

theorem trapezium_area (a b h : ℝ) (h_a : a = 20) (h_b : b = 18) (h_h : h = 10) : 
  (1 / 2) * (a + b) * h = 190 := 
by
  -- We provide the conditions:
  rw [h_a, h_b, h_h]
  -- The proof steps will be skipped using 'sorry'
  sorry

end trapezium_area_l2387_238762


namespace jose_is_21_l2387_238750

-- Define the ages of the individuals based on the conditions
def age_of_inez := 12
def age_of_zack := age_of_inez + 4
def age_of_jose := age_of_zack + 5

-- State the proposition we want to prove
theorem jose_is_21 : age_of_jose = 21 := 
by 
  sorry

end jose_is_21_l2387_238750


namespace correct_option_l2387_238769

-- Definitions of the options as Lean statements
def optionA : Prop := (-1 : ℝ) / 6 > (-1 : ℝ) / 7
def optionB : Prop := (-4 : ℝ) / 3 < (-3 : ℝ) / 2
def optionC : Prop := (-2 : ℝ)^3 = -2^3
def optionD : Prop := -(-4.5 : ℝ) > abs (-4.6 : ℝ)

-- Theorem stating that optionC is the correct statement among the provided options
theorem correct_option : optionC :=
by
  unfold optionC
  rw [neg_pow, neg_pow, pow_succ, pow_succ]
  sorry  -- The proof is omitted as per instructions

end correct_option_l2387_238769


namespace combination_identity_l2387_238774

theorem combination_identity : (Nat.choose 5 3 + Nat.choose 5 4 = Nat.choose 6 4) := 
by 
  sorry

end combination_identity_l2387_238774


namespace solution_set_for_x_l2387_238757

theorem solution_set_for_x (x : ℝ) (h : ⌊x⌋ + ⌈x⌉ = 7) : 3 < x ∧ x < 4 :=
sorry

end solution_set_for_x_l2387_238757


namespace value_of_a_l2387_238705

theorem value_of_a :
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - -1)^2 + (y - 1)^2 = 4) := sorry

end value_of_a_l2387_238705


namespace triangle_is_isosceles_l2387_238720

theorem triangle_is_isosceles (A B C : ℝ)
  (h : Real.log (Real.sin A) - Real.log (Real.cos B) - Real.log (Real.sin C) = Real.log 2) :
  ∃ a b c : ℝ, a = b ∨ b = c ∨ a = c := 
sorry

end triangle_is_isosceles_l2387_238720


namespace sequence_to_geometric_l2387_238741

variable (a : ℕ → ℝ)

def seq_geom (a : ℕ → ℝ) : Prop :=
∀ m n, a (m + n) = a m * a n

def condition (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 2) = a n * a (n + 1)

theorem sequence_to_geometric (a1 a2 : ℝ) (h1 : a 1 = a1) (h2 : a 2 = a2) (h : ∀ n, a (n + 2) = a n * a (n + 1)) :
  a1 = 1 → a2 = 1 → seq_geom a :=
by
  intros ha1 ha2
  have h_seq : ∀ n, a n = 1 := sorry
  intros m n
  sorry

end sequence_to_geometric_l2387_238741


namespace possible_marks_l2387_238748

theorem possible_marks (n : ℕ) : n = 3 ∨ n = 6 ↔
  ∃ (m : ℕ), n = (m * (m - 1)) / 2 ∧ (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → ∃ (i j : ℕ), i < j ∧ j - i = k ∧ (∀ (x y : ℕ), x < y → x ≠ i ∨ y ≠ j)) :=
by sorry

end possible_marks_l2387_238748


namespace miniature_model_to_actual_statue_scale_l2387_238788

theorem miniature_model_to_actual_statue_scale (height_actual : ℝ) (height_model : ℝ) : 
  height_actual = 90 → height_model = 6 → 
  (height_actual / height_model = 15) := 
by
  intros h_actual h_model
  rw [h_actual, h_model]
  sorry

end miniature_model_to_actual_statue_scale_l2387_238788


namespace a_n_is_perfect_square_l2387_238768

def sequence_c (n : ℕ) : ℤ :=
  if n = 0 then 1
  else if n = 1 then 0
  else if n = 2 then 2005
  else -3 * sequence_c (n - 2) - 4 * sequence_c (n - 3) + 2008

def sequence_a (n : ℕ) :=
  if n < 2 then 0
  else 5 * (sequence_c (n + 2) - sequence_c n) * (502 - sequence_c (n - 1) - sequence_c (n - 2)) + (4 ^ n) * 2004 * 501

theorem a_n_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, sequence_a n = k^2 :=
sorry

end a_n_is_perfect_square_l2387_238768


namespace point_in_plane_region_l2387_238789

theorem point_in_plane_region :
  (2 * 0 + 1 - 6 < 0) ∧ ¬(2 * 5 + 0 - 6 < 0) ∧ ¬(2 * 0 + 7 - 6 < 0) ∧ ¬(2 * 2 + 3 - 6 < 0) :=
by
  -- Proof detail goes here.
  sorry

end point_in_plane_region_l2387_238789


namespace total_kilometers_ridden_l2387_238714

theorem total_kilometers_ridden :
  ∀ (d1 d2 d3 d4 : ℕ),
    d1 = 40 →
    d2 = 50 →
    d3 = d2 - d2 / 2 →
    d4 = d1 + d3 →
    d1 + d2 + d3 + d4 = 180 :=
by 
  intros d1 d2 d3 d4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_kilometers_ridden_l2387_238714


namespace geometric_progression_x_value_l2387_238723

noncomputable def geometric_progression_solution (x : ℝ) : Prop :=
  let a := -30 + x
  let b := -10 + x
  let c := 40 + x
  b^2 = a * c

theorem geometric_progression_x_value :
  ∃ x : ℝ, geometric_progression_solution x ∧ x = 130 / 3 :=
by
  sorry

end geometric_progression_x_value_l2387_238723


namespace increasing_on_1_to_infty_min_value_on_1_to_e_l2387_238761

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := (2 * x^2 - a) / x

-- Proof that f(x) is increasing on (1, +∞) when a = 2
theorem increasing_on_1_to_infty (x : ℝ) (h : x > 1) : f' x 2 > 0 := 
  sorry

-- Proof for minimum value of f(x) on [1, e]
theorem min_value_on_1_to_e (a : ℝ) :
  if a ≤ 2 then ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = 1
  else if 2 < a ∧ a < 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = a / 2 - (a / 2) * Real.log (a / 2)
  else if a ≥ 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = Real.exp 2 - a
  else False := 
  sorry

end increasing_on_1_to_infty_min_value_on_1_to_e_l2387_238761


namespace average_value_f_l2387_238731

def f (x : ℝ) : ℝ := (1 + x)^3

theorem average_value_f : (1 / (4 - 2)) * (∫ x in (2:ℝ)..(4:ℝ), f x) = 68 :=
by
  sorry

end average_value_f_l2387_238731


namespace intersection_M_N_l2387_238752

open Set

def M : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def N : Set ℝ := { x | x >= 1 }

theorem intersection_M_N : M ∩ N = { x | 1 <= x ∧ x < 3 } :=
by
  sorry

end intersection_M_N_l2387_238752


namespace proposition_A_iff_proposition_B_l2387_238793

-- Define propositions
def Proposition_A (A B C : ℕ) : Prop := (A = 60 ∨ B = 60 ∨ C = 60)
def Proposition_B (A B C : ℕ) : Prop :=
  (A + B + C = 180) ∧ 
  (2 * B = A + C)

-- The theorem stating the relationship between Proposition_A and Proposition_B
theorem proposition_A_iff_proposition_B (A B C : ℕ) :
  Proposition_A A B C ↔ Proposition_B A B C :=
sorry

end proposition_A_iff_proposition_B_l2387_238793


namespace no_divisibility_condition_by_all_others_l2387_238751

theorem no_divisibility_condition_by_all_others 
  {p : ℕ → ℕ} 
  (h_distinct_odd_primes : ∀ i j, i ≠ j → Nat.Prime (p i) ∧ Nat.Prime (p j) ∧ p i ≠ p j ∧ p i % 2 = 1 ∧ p j % 2 = 1)
  (h_ordered : ∀ i j, i < j → p i < p j) :
  ¬ ∀ i j, i ≠ j → (∀ k ≠ i, k ≠ j → p k ∣ (p i ^ 8 - p j ^ 8)) :=
by
  sorry

end no_divisibility_condition_by_all_others_l2387_238751


namespace find_number_l2387_238781

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 11) : x = 5.5 :=
by
  sorry

end find_number_l2387_238781


namespace fraction_sum_equals_zero_l2387_238715

theorem fraction_sum_equals_zero :
  (1 / 12) + (2 / 12) + (3 / 12) + (4 / 12) + (5 / 12) + (6 / 12) + (7 / 12) + (8 / 12) + (9 / 12) - (45 / 12) = 0 :=
by
  sorry

end fraction_sum_equals_zero_l2387_238715


namespace youseff_time_difference_l2387_238755

noncomputable def walking_time (blocks : ℕ) (time_per_block : ℕ) : ℕ := blocks * time_per_block
noncomputable def biking_time (blocks : ℕ) (time_per_block_seconds : ℕ) : ℕ := (blocks * time_per_block_seconds) / 60

theorem youseff_time_difference : walking_time 6 1 - biking_time 6 20 = 4 := by
  sorry

end youseff_time_difference_l2387_238755


namespace ratio_PR_QS_l2387_238734

/-- Given points P, Q, R, and S on a straight line in that order with
    distances PQ = 3 units, QR = 7 units, and PS = 20 units,
    the ratio of PR to QS is 1. -/
theorem ratio_PR_QS (P Q R S : ℝ) (PQ QR PS : ℝ) (hPQ : PQ = 3) (hQR : QR = 7) (hPS : PS = 20) :
  let PR := PQ + QR
  let QS := PS - PQ - QR
  PR / QS = 1 :=
by
  -- Definitions from conditions
  let PR := PQ + QR
  let QS := PS - PQ - QR
  -- Proof not required, hence sorry
  sorry

end ratio_PR_QS_l2387_238734


namespace johnny_marble_choice_l2387_238701

/-- Johnny has 9 different colored marbles and always chooses 1 specific red marble.
    Prove that the number of ways to choose four marbles from his bag is 56. -/
theorem johnny_marble_choice : (Nat.choose 8 3) = 56 := 
by
  sorry

end johnny_marble_choice_l2387_238701


namespace blue_to_red_marble_ratio_l2387_238749

-- Define the given conditions and the result.
theorem blue_to_red_marble_ratio (total_marble yellow_marble : ℕ) 
  (h1 : total_marble = 19)
  (h2 : yellow_marble = 5)
  (red_marble : ℕ)
  (h3 : red_marble = yellow_marble + 3) : 
  ∃ blue_marble : ℕ, (blue_marble = total_marble - (yellow_marble + red_marble)) 
  ∧ (blue_marble / (gcd blue_marble red_marble)) = 3 
  ∧ (red_marble / (gcd blue_marble red_marble)) = 4 :=
by {
  --existence of blue_marble and the ratio
  sorry
}

end blue_to_red_marble_ratio_l2387_238749


namespace father_age_is_30_l2387_238711

theorem father_age_is_30 {M F : ℝ} 
  (h1 : M = (2 / 5) * F) 
  (h2 : M + 6 = (1 / 2) * (F + 6)) :
  F = 30 :=
sorry

end father_age_is_30_l2387_238711


namespace find_side_length_l2387_238732

theorem find_side_length
  (a b : ℝ)
  (S : ℝ)
  (h1 : a = 4)
  (h2 : b = 5)
  (h3 : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end find_side_length_l2387_238732


namespace power_of_three_l2387_238738

theorem power_of_three (a b : ℕ) (h1 : 360 = (2^3) * (3^2) * (5^1))
  (h2 : 2^a ∣ 360 ∧ ∀ n, 2^n ∣ 360 → n ≤ a)
  (h3 : 5^b ∣ 360 ∧ ∀ n, 5^n ∣ 360 → n ≤ b) :
  (1/3 : ℝ)^(b - a) = 9 :=
by sorry

end power_of_three_l2387_238738


namespace maximal_subset_with_property_A_l2387_238704

-- Define property A for a subset S ⊆ {0, 1, 2, ..., 99}
def has_property_A (S : Finset ℕ) : Prop := 
  ∀ a b c : ℕ, (a * 10 + b ∈ S) → (b * 10 + c ∈ S) → False

-- Define the set of integers {0, 1, 2, ..., 99}
def numbers_set := Finset.range 100

-- The main statement to be proven
theorem maximal_subset_with_property_A :
  ∃ S : Finset ℕ, S ⊆ numbers_set ∧ has_property_A S ∧ S.card = 25 := 
sorry

end maximal_subset_with_property_A_l2387_238704


namespace train_speed_l2387_238764

noncomputable def original_speed_of_train (v d : ℝ) : Prop :=
  (120 ≤ v / (5/7)) ∧
  (2 * d) / (5 * v) = 65 / 60 ∧
  (2 * (d - 42)) / (5 * v) = 45 / 60

theorem train_speed (v d : ℝ) (h : original_speed_of_train v d) : v = 50.4 :=
by sorry

end train_speed_l2387_238764


namespace bob_paid_correctly_l2387_238756

-- Define the variables involved
def alice_acorns : ℕ := 3600
def price_per_acorn : ℕ := 15
def multiplier : ℕ := 9
def total_amount_alice_paid : ℕ := alice_acorns * price_per_acorn

-- Define Bob's payment amount
def bob_payment : ℕ := total_amount_alice_paid / multiplier

-- The main theorem
theorem bob_paid_correctly : bob_payment = 6000 := by
  sorry

end bob_paid_correctly_l2387_238756


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l2387_238790

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l2387_238790


namespace fraction_problem_l2387_238740

-- Definitions translated from conditions
variables (m n p q : ℚ)
axiom h1 : m / n = 20
axiom h2 : p / n = 5
axiom h3 : p / q = 1 / 15

-- Statement to prove
theorem fraction_problem : m / q = 4 / 15 :=
by
  sorry

end fraction_problem_l2387_238740


namespace shaniqua_earnings_l2387_238775

noncomputable def shaniqua_total_earnings : ℕ :=
  let haircut_rate := 12
  let style_rate := 25
  let coloring_rate := 35
  let treatment_rate := 50
  let haircuts := 8
  let styles := 5
  let colorings := 10
  let treatments := 6
  (haircuts * haircut_rate) +
  (styles * style_rate) +
  (colorings * coloring_rate) +
  (treatments * treatment_rate)

theorem shaniqua_earnings : shaniqua_total_earnings = 871 := by
  sorry

end shaniqua_earnings_l2387_238775


namespace simplify_expression_l2387_238730

theorem simplify_expression (x y : ℝ) (h : x = -3) : 
  x * (x - 4) * (x + 4) - (x + 3) * (x^2 - 6 * x + 9) + 5 * x^3 * y^2 / (x^2 * y^2) = -66 :=
by
  sorry

end simplify_expression_l2387_238730


namespace ratio_A_to_B_l2387_238708

/--
Proof problem statement:
Given that A and B together can finish the work in 4 days,
and B alone can finish the work in 24 days,
prove that the ratio of the time A takes to finish the work to the time B takes to finish the work is 1:5.
-/
theorem ratio_A_to_B
  (A_time B_time working_together_time : ℝ) 
  (h1 : working_together_time = 4)
  (h2 : B_time = 24)
  (h3 : 1 / A_time + 1 / B_time = 1 / working_together_time) :
  A_time / B_time = 1 / 5 :=
sorry

end ratio_A_to_B_l2387_238708


namespace recliner_price_drop_l2387_238796

theorem recliner_price_drop
  (P : ℝ) (N : ℝ)
  (N' : ℝ := 1.8 * N)
  (G : ℝ := P * N)
  (G' : ℝ := 1.44 * G) :
  (P' : ℝ) → P' = 0.8 * P → (P - P') / P * 100 = 20 :=
by
  intros
  sorry

end recliner_price_drop_l2387_238796


namespace bicyclist_speed_remainder_l2387_238799

noncomputable def speed_of_bicyclist (total_distance first_distance remaining_distance time_for_first_distance total_time : ℝ) : ℝ :=
  remaining_distance / (total_time - time_for_first_distance)

theorem bicyclist_speed_remainder 
  (total_distance : ℝ)
  (first_distance : ℝ)
  (remaining_distance : ℝ)
  (first_speed : ℝ)
  (average_speed : ℝ)
  (correct_speed : ℝ) :
  total_distance = 250 → 
  first_distance = 100 →
  remaining_distance = total_distance - first_distance →
  first_speed = 20 →
  average_speed = 16.67 →
  correct_speed = 15 →
  speed_of_bicyclist total_distance first_distance remaining_distance (first_distance / first_speed) (total_distance / average_speed) = correct_speed :=
by
  sorry

end bicyclist_speed_remainder_l2387_238799


namespace sin_double_angle_l2387_238700

theorem sin_double_angle (θ : Real) (h : Real.sin θ = 3/5) : Real.sin (2*θ) = 24/25 :=
by
  sorry

end sin_double_angle_l2387_238700


namespace graph_symmetric_about_x_eq_pi_div_8_l2387_238724

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem graph_symmetric_about_x_eq_pi_div_8 :
  ∀ x, f (π / 8 - x) = f (π / 8 + x) :=
sorry

end graph_symmetric_about_x_eq_pi_div_8_l2387_238724


namespace area_of_rectangle_A_is_88_l2387_238780

theorem area_of_rectangle_A_is_88 
  (lA lB lC w wC : ℝ)
  (h1 : lB = lA + 2)
  (h2 : lB * w = lA * w + 22)
  (h3 : wC = w - 4)
  (AreaB : ℝ := lB * w)
  (AreaC : ℝ := lB * wC)
  (h4 : AreaC = AreaB - 40) : 
  (lA * w = 88) :=
sorry

end area_of_rectangle_A_is_88_l2387_238780


namespace total_bananas_in_collection_l2387_238785

-- Definitions based on the conditions
def group_size : ℕ := 18
def number_of_groups : ℕ := 10

-- The proof problem statement
theorem total_bananas_in_collection : group_size * number_of_groups = 180 := by
  sorry

end total_bananas_in_collection_l2387_238785


namespace sum_of_roots_l2387_238772

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem sum_of_roots (m : ℝ) (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 < 2 * Real.pi)
  (h3 : 0 ≤ x2) (h4 : x2 < 2 * Real.pi) (h_distinct : x1 ≠ x2)
  (h_eq1 : f x1 = m) (h_eq2 : f x2 = m) : x1 + x2 = Real.pi / 2 ∨ x1 + x2 = 5 * Real.pi / 2 :=
by
  sorry

end sum_of_roots_l2387_238772


namespace simplify_expression_l2387_238783

theorem simplify_expression (x y : ℝ) : 3 * y - 5 * x + 2 * y + 4 * x = 5 * y - x :=
by
  sorry

end simplify_expression_l2387_238783


namespace coopers_age_l2387_238795

theorem coopers_age (C D M E : ℝ) 
  (h1 : D = 2 * C) 
  (h2 : M = 2 * C + 1) 
  (h3 : E = 3 * C)
  (h4 : C + D + M + E = 62) : 
  C = 61 / 8 := 
by 
  sorry

end coopers_age_l2387_238795


namespace incorrect_statement_l2387_238728

-- Define the relationship between the length of the spring and the mass of the object
def spring_length (mass : ℝ) : ℝ := 2.5 * mass + 10

-- Formalize statements A, B, C, and D
def statementA : Prop := spring_length 0 = 10

def statementB : Prop :=
  ¬ ∃ (length : ℝ) (mass : ℝ), (spring_length mass = length ∧ mass = (length - 10) / 2.5)

def statementC : Prop :=
  ∀ m : ℝ, spring_length (m + 1) = spring_length m + 2.5

def statementD : Prop := spring_length 4 = 20

-- The Lean statement to prove that statement B is incorrect
theorem incorrect_statement (hA : statementA) (hC : statementC) (hD : statementD) : ¬ statementB := by
  sorry

end incorrect_statement_l2387_238728


namespace pencils_left_l2387_238710

def initial_pencils : Nat := 127
def pencils_from_joyce : Nat := 14
def pencils_per_friend : Nat := 7

theorem pencils_left : ((initial_pencils + pencils_from_joyce) % pencils_per_friend) = 1 := by
  sorry

end pencils_left_l2387_238710


namespace savings_fraction_l2387_238717

theorem savings_fraction 
(P : ℝ) 
(f : ℝ) 
(h1 : P > 0) 
(h2 : 12 * f * P = 5 * (1 - f) * P) : 
    f = 5 / 17 :=
by
  sorry

end savings_fraction_l2387_238717


namespace coefficients_sum_l2387_238754

theorem coefficients_sum (a0 a1 a2 a3 a4 : ℝ) (h : (1 - 2*x)^4 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) : 
  a0 + a4 = 17 :=
by
  sorry

end coefficients_sum_l2387_238754


namespace sqrt_360000_eq_600_l2387_238719

theorem sqrt_360000_eq_600 : Real.sqrt 360000 = 600 := 
sorry

end sqrt_360000_eq_600_l2387_238719


namespace math_problem_l2387_238739

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem math_problem
  (omega phi : ℝ)
  (h1 : omega > 0)
  (h2 : |phi| < Real.pi / 2)
  (h3 : ∀ x, f x = Real.sin (omega * x + phi))
  (h4 : ∀ k : ℤ, f (k * Real.pi) = f 0) 
  (h5 : f 0 = 1 / 2) :
  (omega = 2) ∧
  (∀ x, f (x + Real.pi / 6) = f (-x + Real.pi / 6)) ∧
  (∀ k : ℤ, 
    ∀ x, x ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    ∀ y, y ∈ Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6) → 
    x < y → f x ≤ f y) :=
by
  sorry

end math_problem_l2387_238739


namespace scientific_notation_470000000_l2387_238725

theorem scientific_notation_470000000 : 470000000 = 4.7 * 10^8 :=
by
  sorry

end scientific_notation_470000000_l2387_238725


namespace ticket_value_unique_l2387_238791

theorem ticket_value_unique (x : ℕ) (h₁ : ∃ n, n > 0 ∧ x * n = 60)
  (h₂ : ∃ m, m > 0 ∧ x * m = 90)
  (h₃ : ∃ p, p > 0 ∧ x * p = 49) : 
  ∃! x, x = 1 :=
by
  sorry

end ticket_value_unique_l2387_238791


namespace number_subtracted_l2387_238786

theorem number_subtracted (x : ℝ) : 3 + 2 * (8 - x) = 24.16 → x = -2.58 :=
by
  intro h
  sorry

end number_subtracted_l2387_238786


namespace q_simplified_l2387_238706

noncomputable def q (a b c x : ℝ) : ℝ :=
  (x + a)^4 / ((a - b) * (a - c)) +
  (x + b)^4 / ((b - a) * (b - c)) +
  (x + c)^4 / ((c - a) * (c - b)) - 3 * x * (
      1 / ((a - b) * (a - c)) + 
      1 / ((b - a) * (b - c)) +
      1 / ((c - a) * (c - b))
  )

theorem q_simplified (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q a b c x = a^2 + b^2 + c^2 + 4*x^2 - 4*(a + b + c)*x + 12*x :=
sorry

end q_simplified_l2387_238706


namespace find_higher_selling_price_l2387_238753

def cost_price := 200
def selling_price_low := 340
def gain_low := selling_price_low - cost_price
def gain_high := gain_low + (5 / 100) * gain_low
def higher_selling_price := cost_price + gain_high

theorem find_higher_selling_price : higher_selling_price = 347 := 
by 
  sorry

end find_higher_selling_price_l2387_238753


namespace kendra_change_is_correct_l2387_238794

-- Define the initial conditions
def price_wooden_toy : ℕ := 20
def price_hat : ℕ := 10
def kendra_initial_money : ℕ := 100
def num_wooden_toys : ℕ := 2
def num_hats : ℕ := 3

-- Calculate the total costs
def total_wooden_toys_cost : ℕ := price_wooden_toy * num_wooden_toys
def total_hats_cost : ℕ := price_hat * num_hats
def total_cost : ℕ := total_wooden_toys_cost + total_hats_cost

-- Calculate the change Kendra received
def kendra_change : ℕ := kendra_initial_money - total_cost

theorem kendra_change_is_correct : kendra_change = 30 := by
  sorry

end kendra_change_is_correct_l2387_238794


namespace B_pow_5_eq_r_B_add_s_I_l2387_238797

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![ -2,  3 ], 
                                      ![  4,  5 ]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem B_pow_5_eq_r_B_add_s_I :
  ∃ r s : ℤ, (r = 425) ∧ (s = 780) ∧ (B^5 = r • B + s • I) :=
by
  sorry

end B_pow_5_eq_r_B_add_s_I_l2387_238797


namespace polygon_perimeter_l2387_238779

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l2387_238779


namespace find_k_l2387_238759

-- Definitions based on the problem conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (k : ℝ) : ℝ × ℝ := (k, 4)

-- Property of parallel vectors
def parallel (u v : ℝ × ℝ) : Prop := ∃ c : ℝ, u.1 = c * v.1 ∧ u.2 = c * v.2

-- Theorem statement equivalent to the problem
theorem find_k (k : ℝ) (h : parallel vector_a (vector_b k)) : k = -2 :=
sorry

end find_k_l2387_238759


namespace arithmetic_mean_calc_l2387_238792

theorem arithmetic_mean_calc (x a : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) :
  ( ( (x + a)^2 / x ) + ( (x - a)^2 / x ) ) / 2 = x + (a^2 / x) :=
sorry

end arithmetic_mean_calc_l2387_238792


namespace tate_total_years_proof_l2387_238765

def highSchoolYears: ℕ := 4 - 1
def gapYear: ℕ := 2
def bachelorYears (highSchoolYears: ℕ): ℕ := 2 * highSchoolYears
def workExperience: ℕ := 1
def phdYears (highSchoolYears: ℕ) (bachelorYears: ℕ): ℕ := 3 * (highSchoolYears + bachelorYears)
def totalYears (highSchoolYears: ℕ) (gapYear: ℕ) (bachelorYears: ℕ) (workExperience: ℕ) (phdYears: ℕ): ℕ :=
  highSchoolYears + gapYear + bachelorYears + workExperience + phdYears

theorem tate_total_years_proof : totalYears highSchoolYears gapYear (bachelorYears highSchoolYears) workExperience (phdYears highSchoolYears (bachelorYears highSchoolYears)) = 39 := by
  sorry

end tate_total_years_proof_l2387_238765


namespace equivalence_of_statements_l2387_238721

-- Variables used in the statements
variable (P Q : Prop)

-- Proof problem statement
theorem equivalence_of_statements : (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end equivalence_of_statements_l2387_238721
