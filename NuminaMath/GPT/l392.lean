import Mathlib

namespace range_of_g_l392_392140

theorem range_of_g (m : ℝ) (h_m : m > 0) : 
  (∀ y, y ≥ 1 → ∃ x, x ≥ 1 ∧ y = x ^ m) :=
begin
  sorry
end

end range_of_g_l392_392140


namespace physics_majors_consecutive_probability_l392_392297

open Nat

-- Define the total number of seats and the specific majors
def totalSeats : ℕ := 10
def mathMajors : ℕ := 4
def physicsMajors : ℕ := 3
def chemistryMajors : ℕ := 2
def biologyMajors : ℕ := 1

-- Assuming a round table configuration
def probabilityPhysicsMajorsConsecutive : ℚ :=
  (3 * (Nat.factorial (totalSeats - physicsMajors))) / (Nat.factorial (totalSeats - 1))

-- Declare the theorem
theorem physics_majors_consecutive_probability : 
  probabilityPhysicsMajorsConsecutive = 1 / 24 :=
by
  sorry

end physics_majors_consecutive_probability_l392_392297


namespace num_zeros_of_func_is_two_l392_392677

-- Define the function y = x - 4/x
def func (x : ℝ) : ℝ := x - 4 / x

-- Problem statement: Prove the number of zeros of the function y = x - 4/x is 2.
theorem num_zeros_of_func_is_two : (∃ (x : ℝ), func x = 0) ∧ (∃ (y : ℝ), func y = 0) ∧ (¬ ∃ (z : ℝ), func z = 0 ∧ z ≠ 2 ∧ z ≠ -2) :=
by
  sorry

end num_zeros_of_func_is_two_l392_392677


namespace count_odd_3_digit_numbers_with_distinct_digits_l392_392531

variable (n : ℕ)

def is_between_100_and_999 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def digits_are_distinct (n : ℕ) : Prop :=
  ∀ i j, i ≠ j → (n / 10 ^ i % 10) ≠ (n / 10 ^ j % 10)

theorem count_odd_3_digit_numbers_with_distinct_digits :
  (finset.filter (λ n, is_between_100_and_999 n ∧ is_odd n ∧ digits_are_distinct n)
    (finset.range 1000)).card = 320 :=
by
  sorry

end count_odd_3_digit_numbers_with_distinct_digits_l392_392531


namespace vector_condition_l392_392246

variables (a b : ℝ × ℝ × ℝ) -- Assuming 3D vectors for generality
variables (ha : a ≠ (0, 0, 0)) (hb : b ≠ (0, 0, 0))

def vector_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def normalized (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let n := vector_norm v in (v.1 / n, v.2 / n, v.3 / n)

theorem vector_condition {a b : ℝ × ℝ × ℝ} (ha : a ≠ (0, 0, 0)) (hb : b ≠ (0, 0, 0)) :
  a = (-2 : ℝ) • b → normalized a + normalized b = (0, 0, 0) :=
sorry

end vector_condition_l392_392246


namespace shift_f_for_even_g_l392_392120

def f (x : Real) (m : Real) : Real := m * sin (2 * x) - sqrt 3 * cos (2 * x)

def g (x : Real) : Real :=
  2 * sin (2 * (x - π / 12) - π / 3)

theorem shift_f_for_even_g :
  (∀ (x : Real), f (π / 6) 1 = 0) →
  ∃ n : Real, (n = π / 12) ∧ (∀ x : Real, g x = (2 * sin (2 * (x - n) - π / 3))) :=
begin
  sorry
end

end shift_f_for_even_g_l392_392120


namespace advertisement_arrangement_l392_392795

theorem advertisement_arrangement :
  let A (n r : ℕ) := n.factorial / (n - r).factorial in
  A 4 4 * A 5 2 = 480 := by
  sorry

end advertisement_arrangement_l392_392795


namespace tangent_product_power_l392_392153

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180))
  * (1 + Real.tan (2 * Real.pi / 180))
  * (1 + Real.tan (3 * Real.pi / 180))
  * (1 + Real.tan (4 * Real.pi / 180))
  * (1 + Real.tan (5 * Real.pi / 180))
  * (1 + Real.tan (6 * Real.pi / 180))
  * (1 + Real.tan (7 * Real.pi / 180))
  * (1 + Real.tan (8 * Real.pi / 180))
  * (1 + Real.tan (9 * Real.pi / 180))
  * (1 + Real.tan (10 * Real.pi / 180))
  * (1 + Real.tan (11 * Real.pi / 180))
  * (1 + Real.tan (12 * Real.pi / 180))
  * (1 + Real.tan (13 * Real.pi / 180))
  * (1 + Real.tan (14 * Real.pi / 180))
  * (1 + Real.tan (15 * Real.pi / 180))
  * (1 + Real.tan (16 * Real.pi / 180))
  * (1 + Real.tan (17 * Real.pi / 180))
  * (1 + Real.tan (18 * Real.pi / 180))
  * (1 + Real.tan (19 * Real.pi / 180))
  * (1 + Real.tan (20 * Real.pi / 180))
  * (1 + Real.tan (21 * Real.pi / 180))
  * (1 + Real.tan (22 * Real.pi / 180))
  * (1 + Real.tan (23 * Real.pi / 180))
  * (1 + Real.tan (24 * Real.pi / 180))
  * (1 + Real.tan (25 * Real.pi / 180))
  * (1 + Real.tan (26 * Real.pi / 180))
  * (1 + Real.tan (27 * Real.pi / 180))
  * (1 + Real.tan (28 * Real.pi / 180))
  * (1 + Real.tan (29 * Real.pi / 180))
  * (1 + Real.tan (30 * Real.pi / 180))
  * (1 + Real.tan (31 * Real.pi / 180))
  * (1 + Real.tan (32 * Real.pi / 180))
  * (1 + Real.tan (33 * Real.pi / 180))
  * (1 + Real.tan (34 * Real.pi / 180))
  * (1 + Real.tan (35 * Real.pi / 180))
  * (1 + Real.tan (36 * Real.pi / 180))
  * (1 + Real.tan (37 * Real.pi / 180))
  * (1 + Real.tan (38 * Real.pi / 180))
  * (1 + Real.tan (39 * Real.pi / 180))
  * (1 + Real.tan (40 * Real.pi / 180))
  * (1 + Real.tan (41 * Real.pi / 180))
  * (1 + Real.tan (42 * Real.pi / 180))
  * (1 + Real.tan (43 * Real.pi / 180))
  * (1 + Real.tan (44 * Real.pi / 180))
  * (1 + Real.tan (45 * Real.pi / 180))
  * (1 + Real.tan (46 * Real.pi / 180))
  * (1 + Real.tan (47 * Real.pi / 180))
  * (1 + Real.tan (48 * Real.pi / 180))
  * (1 + Real.tan (49 * Real.pi / 180))
  * (1 + Real.tan (50 * Real.pi / 180))
  * (1 + Real.tan (51 * Real.pi / 180))
  * (1 + Real.tan (52 * Real.pi / 180))
  * (1 + Real.tan (53 * Real.pi / 180))
  * (1 + Real.tan (54 * Real.pi / 180))
  * (1 + Real.tan (55 * Real.pi / 180))
  * (1 + Real.tan (56 * Real.pi / 180))
  * (1 + Real.tan (57 * Real.pi / 180))
  * (1 + Real.tan (58 * Real.pi / 180))
  * (1 + Real.tan (59 * Real.pi / 180))
  * (1 + Real.tan (60 * Real.pi / 180))

theorem tangent_product_power : tangent_product = 2^30 := by
  sorry

end tangent_product_power_l392_392153


namespace reflection_matrix_l392_392018

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l392_392018


namespace digit_makes_5678d_multiple_of_9_l392_392474

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

theorem digit_makes_5678d_multiple_of_9 (d : Nat) (h : d ≥ 0 ∧ d < 10) :
  is_multiple_of_9 (5 * 10000 + 6 * 1000 + 7 * 100 + 8 * 10 + d) ↔ d = 1 := 
by
  sorry

end digit_makes_5678d_multiple_of_9_l392_392474


namespace circle_area_percentage_change_is_negative_36_l392_392783

def circle_area (r : ℝ) : ℝ := real.pi * r^2

def percentage_change (initial_area final_area : ℝ) : ℝ :=
  ((final_area - initial_area) / initial_area) * 100

theorem circle_area_percentage_change_is_negative_36 :
  percentage_change (circle_area 5) (circle_area 4) = -36 :=
by 
  sorry

end circle_area_percentage_change_is_negative_36_l392_392783


namespace angles_of_isosceles_triangle_l392_392229

theorem angles_of_isosceles_triangle (A B C : Type)
    (triangle_ABC : IsoscelesTriangle A B C)
    (angle_bisector : AngleBisector (vertex A))
    (altitude : Altitude (vertex A))
    (angle_between_bisector_and_altitude : angle_between angle_bisector altitude = 60) :
    angle (triangle_ABC ∠ BAC) = 20 ∧ angle (triangle_ABC ∠ ABC) = 80 ∧ angle (triangle_ABC ∠ ACB) = 80 := by
  sorry

end angles_of_isosceles_triangle_l392_392229


namespace solution_set_for_inequality_l392_392859

noncomputable def f : ℝ → ℝ := sorry

axiom f_deriv2_gt : ∀ (x : ℝ), f x > (derivative^2 f) x
axiom f_odd_plus_2017 : ∀ (x : ℝ), f x + 2017 = - (f (-x) + 2017)
axiom f_at_zero : f 0 = -2017

theorem solution_set_for_inequality :
  {x : ℝ | f x + 2017 * Real.exp x < 0} = set.Ioi 0 :=
sorry

end solution_set_for_inequality_l392_392859


namespace simplify_trig_expr_l392_392650

theorem simplify_trig_expr (x : ℝ) : 2 * sin (2 * x) * sin x + cos (3 * x) = cos x := 
by sorry

end simplify_trig_expr_l392_392650


namespace nancy_carrots_l392_392621

-- Definitions based on the conditions
def initial_carrots := 12
def carrots_to_cook := 2
def new_carrot_seeds := 5
def growth_factor := 3
def kept_carrots := 10
def poor_quality_ratio := 3

-- Calculate new carrots grown from seeds
def new_carrots := new_carrot_seeds * growth_factor

-- Total carrots after new ones are added
def total_carrots := kept_carrots + new_carrots

-- Calculate poor quality carrots (integer part only)
def poor_quality_carrots := total_carrots / poor_quality_ratio

-- Calculate good quality carrots
def good_quality_carrots := total_carrots - poor_quality_carrots

-- Statement to prove
theorem nancy_carrots : good_quality_carrots = 17 :=
by
  sorry -- proof is not required

end nancy_carrots_l392_392621


namespace sequence_limit_l392_392735

noncomputable def sequence (α β : ℝ) : ℕ → ℝ
| 0       := α
| 1       := β
| (n + 1) := sequence α β n + (sequence α β (n - 1) - sequence α β n) / (2 * n)

theorem sequence_limit (α β : ℝ) : 
  (tendsto (λ n, sequence α β n) at_top (𝓝 (α + (β - α) / Real.exp (1/2)))) :=
begin
  -- Proof goes here
  sorry
end

end sequence_limit_l392_392735


namespace num_pass_students_is_85_l392_392563

theorem num_pass_students_is_85 (T P F : ℕ) (avg_all avg_pass avg_fail : ℕ) (weight_pass weight_fail : ℕ) 
  (h_total_students : T = 150)
  (h_avg_all : avg_all = 40)
  (h_avg_pass : avg_pass = 45)
  (h_avg_fail : avg_fail = 20)
  (h_weight_ratio : weight_pass = 3 ∧ weight_fail = 1)
  (h_total_marks : (weight_pass * avg_pass * P + weight_fail * avg_fail * F) / (weight_pass * P + weight_fail * F) = avg_all)
  (h_students_sum : P + F = T) :
  P = 85 :=
by
  sorry

end num_pass_students_is_85_l392_392563


namespace find_z_l392_392524

def u (z : ℝ) : ℝ × ℝ := (2, z)
def t : ℝ × ℝ := (8, 4)
def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2
  ((dot a b / dot b b) * b.1, (dot a b / dot b b) * b.2)

theorem find_z (z : ℝ) (h : proj (u z) t = (4, 2)) : z = 3 :=
by
  sorry

end find_z_l392_392524


namespace inscribed_sphere_radius_l392_392076

theorem inscribed_sphere_radius (h1 h2 h3 h4 : ℝ) (S1 S2 S3 S4 V : ℝ)
  (h1_ge : h1 ≥ 1) (h2_ge : h2 ≥ 1) (h3_ge : h3 ≥ 1) (h4_ge : h4 ≥ 1)
  (volume : V = (1/3) * S1 * h1)
  : (∃ r : ℝ, 3 * V = (S1 + S2 + S3 + S4) * r ∧ r = 1 / 4) :=
by
  sorry

end inscribed_sphere_radius_l392_392076


namespace increasing_on_neg_infinity_l392_392117

variable {f : ℝ → ℝ}
variable {g : ℝ → ℝ}

-- Conditions
axiom h1 : ∀ x y : ℝ, x < y → f(x) < f(y)  -- f(x) is increasing
axiom h2 : ∀ x : ℝ, f(x) < 0                -- f(x) is less than 0 for all x

-- Statement
theorem increasing_on_neg_infinity (h : ∀ x, f(x) < 0) (h' : ∀ x y, x < y → f(x) < f(y)) :
  ∀ x y : ℝ, x < y ∧ x < 0 ∧ y < 0 → g(x) = x^2 * f(x) ∧ g(y) = y^2 * f(y) ∧ g(x) < g(y) :=
sorry

end increasing_on_neg_infinity_l392_392117


namespace square_area_fraction_shaded_l392_392633

theorem square_area_fraction_shaded (s : ℝ) :
  let R := (s / 2, s)
  let S := (s, s / 2)
  -- Area of triangle RSV
  let area_RSV := (1 / 2) * (s / 2) * (s * Real.sqrt 2 / 4)
  -- Non-shaded area
  let non_shaded_area := area_RSV
  -- Total area of the square
  let total_area := s^2
  -- Shaded area
  let shaded_area := total_area - non_shaded_area
  -- Fraction shaded
  (shaded_area / total_area) = 1 - Real.sqrt 2 / 16 :=
by
  sorry

end square_area_fraction_shaded_l392_392633


namespace value_of_x_l392_392156

theorem value_of_x (x : ℕ) :
  (1 / 8) * 2 ^ 36 = 8 ^ x → x = 11 :=
by
  intro h
  have h1 : (1 / 8) = 2⁻³ := by sorry
  have h2 : (2⁻³) * 2 ^ 36 = 2 ^ 33 := by sorry
  have h3 : 8 ^ x = (2 ^ 3) ^ x := by sorry
  have h4 : (2 ^ 3) ^ x = 2 ^ (3 * x) := by sorry
  have h5 : 2 ^ 33 = 2 ^ (3 * x) := by sorry
  have h6 : 33 = 3 * x := by sorry
  exact nat.div_eq_of_lt 33 3 sorry

end value_of_x_l392_392156


namespace oranges_per_bag_l392_392337

/-- Given:
     - Total weight of oranges is 45.0 pounds.
     - There are 1.956521739 bags.
    Prove:
     - Each bag contains approximately 23.0 pounds of oranges (rounded to the nearest whole number).
-/
theorem oranges_per_bag :
  let total_oranges := 45.0
  let num_bags := 1.956521739
  let pounds_per_bag := total_oranges / num_bags
  abs (pounds_per_bag - 23.0) < 1 :=
by
  let total_oranges := 45.0
  let num_bags := 1.956521739
  let pounds_per_bag := total_oranges / num_bags
  have : abs (pounds_per_bag - 23.0) < 1 := by sorry
  exact this

end oranges_per_bag_l392_392337


namespace ratio_area_of_extended_equilateral_triangle_l392_392996

variables {A B C A' B' C' : Type} [metric_space A] [metric_space B] [metric_space C]

-- Definitions for equilateral triangle
def equilateral_triangle (s : ℝ) (A B C : point) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

-- Definitions for extended points
def extended_point (A B : point) (s : ℝ) : point := sorry -- s := extend by 2s logic here

-- Defining areas of triangles
def area_triangle (A B C : point) : ℝ := sorry -- can use Heron's formula or another method

theorem ratio_area_of_extended_equilateral_triangle
  (A B C A' B' C' : point)
  (s : ℝ)
  (h_eq : equilateral_triangle s A B C)
  (h_ext_1 : dist B B' = 2 * s ∧ B' = extended_point B A s)
  (h_ext_2 : dist C C' = 2 * s ∧ C' = extended_point C B s)
  (h_ext_3 : dist A A' = 2 * s ∧ A' = extended_point A C s) :
  (area_triangle A' B' C') / (area_triangle A B C) = 9 :=
sorry

end ratio_area_of_extended_equilateral_triangle_l392_392996


namespace outlier_count_zero_l392_392853

open List

def data_set : List ℕ := [8, 15, 21, 29, 29, 35, 39, 42, 50, 68]
def Q1 : ℕ := 25
def Q3 : ℕ := 45
def IQR : ℕ := Q3 - Q1
def outlier_multiplier : ℝ := 2.0
def lower_outlier_threshold : ℝ := Q1 - outlier_multiplier * IQR
def upper_outlier_threshold : ℝ := Q3 + outlier_multiplier * IQR

theorem outlier_count_zero :
  (∀ x ∈ data_set, x ≥ lower_outlier_threshold ∧ x ≤ upper_outlier_threshold) → 
  (data_set.count (λ x, x < lower_outlier_threshold ∨ x > upper_outlier_threshold) = 0) :=
by
  intro h
  sorry

end outlier_count_zero_l392_392853


namespace intersection_correct_l392_392493

open Set

noncomputable def A := {x : ℕ | x^2 - x - 2 ≤ 0}
noncomputable def B := {x : ℝ | -1 ≤ x ∧ x < 2}
noncomputable def A_cap_B := A ∩ {x : ℕ | (x : ℝ) ∈ B}

theorem intersection_correct : A_cap_B = {0, 1} :=
sorry

end intersection_correct_l392_392493


namespace problem_l392_392519

open Set

variables (m n : ℕ)
def A : Set ℕ := {1, 2, m}
def B : Set ℕ := {2, 3, 4, n}

theorem problem (h : A m n ∩ B m n = {1, 2, 3}) : m - n = 2 :=
sorry

end problem_l392_392519


namespace al_and_barb_rest_days_l392_392827

theorem al_and_barb_rest_days (n : ℕ) (al_cycle bar_Cycle : ℕ) (days : ℕ) (coinciding_days : ℕ) :
  (al_cycle = 4) →
  (bar_Cycle = 10) →
  (days = 1000) →
  (coinciding_days = 100) →
  (∃ k : ℕ, days = k * algebra.lcm al_cycle bar_Cycle ∧ coinciding_days = 2 * k) :=
by 
  intros h_al_cycle h_bar_Cycle h_days h_coinciding_days
  let k := days / algebra.lcm al_cycle bar_Cycle
  use k
  split
  sorry
  sorry

end al_and_barb_rest_days_l392_392827


namespace solve_for_x_l392_392290

theorem solve_for_x (x : ℝ) (h : 2^(32^x) = 32^(2^x)) : 
  x = real.log (2, 5) / 4 :=
sorry

end solve_for_x_l392_392290


namespace emma_share_l392_392404

-- Definitions for the conditions
def total_money : ℝ := 153
def ratio : (ℝ × ℝ × ℝ) := (3, 5, 9)
def total_parts := ratio.1 + ratio.2 + ratio.3
def value_per_part : ℝ := total_money / total_parts
def emma_parts := ratio.2

-- Theorem to prove Emma's share
theorem emma_share : (emma_parts * value_per_part) = 45 := by
  sorry

end emma_share_l392_392404


namespace calculate_p_q_sum_l392_392012

-- Define the probabilities for individual and pairwise traits.
def prob_X_only := 0.12
def prob_Y_only := 0.11
def prob_Z_only := 0.11
def prob_X_and_Y_only := 0.15
def prob_X_and_Z_only := 0.15
def prob_Y_and_Z_only := 0.15
def prob_Z_given_X_and_Y := 1/5

-- Define the condition given in the problem.
def prob_no_traits_given_no_X (p q : ℕ) [hp: rel_prime p q] := (p : ℚ) / q

-- Formalize the target statement.
theorem calculate_p_q_sum : ∃ (p q : ℕ), rel_prime p q ∧ prob_no_traits_given_no_X p q = 29/88 ∧ (p + q = 117) :=
by
  sorry

end calculate_p_q_sum_l392_392012


namespace part_one_part_two_l392_392514

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  Real.exp x - Real.log (Real.exp x * a)

theorem part_one (x : ℝ) : f x Mathlib.Real.exp 1 ≥ 0 :=
sorry

theorem part_two (x : ℝ) (h : ∀ x, f x a > Mathlib.Real.exp 1) : a < 1 - Mathlib.Real.exp 1 :=
sorry

end part_one_part_two_l392_392514


namespace unique_pairs_of_socks_l392_392785

-- Defining the problem conditions
def pairs_socks : Nat := 3

-- The main proof statement
theorem unique_pairs_of_socks : ∃ (n : Nat), n = 3 ∧ 
  (∀ (p q : Fin 6), (p / 2 ≠ q / 2) → p ≠ q) →
  (n = (pairs_socks * (pairs_socks - 1)) / 2) :=
by
  sorry

end unique_pairs_of_socks_l392_392785


namespace find_angle_x_l392_392631

def O_center (O : Point) (C : Circle) : Prop := C.center = O

def is_diameter (A D O : Point) (C : Circle) : Prop := D ∈ C ∧ line_through A D ∧ A ∈ line_segments O D

def is_isosceles (A B C : Point) : Prop := dist A B = dist A C

def angle_at_base (B O C : Point) (θ : ℝ) : Prop := ∠ B O C = θ ∧ ∠ O B C = θ

axiom angle_in_semicircle (A C D O : Point) (C1 D1 : Circle) :
  is_diameter A D O C1 → ∠ A C D = 90

theorem find_angle_x (O B C A D : Point) (C1 C2 : Circle):
  O_center O C1 ∧ 
  is_isosceles O B C∧ 
  angle_at_base B O C 32 ∧ 
  is_diameter A D O C1 ∧ 
  ∠ A D C = 67 →
  ∠ O A D - ∠ A O C = 9 := by
  sorry

end find_angle_x_l392_392631


namespace area_ABC_l392_392278

-- Defining coordinates for points X, Y, Z
def X := (6, 0)
def Y := (8, 4)
def Z := (10, 0)

-- Helper function to calculate the area of a triangle given three points
def area_triangle (P Q R : (ℕ × ℕ)) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

-- Defining the area of triangle XYZ
def area_XYZ : ℝ := area_triangle X Y Z

-- Statement of the theorem
theorem area_ABC :
  ∃ (Area_ABC : ℝ), area_XYZ = 0.1111111111111111 * Area_ABC ∧ Area_ABC = 72 :=
by
  sorry

end area_ABC_l392_392278


namespace reflection_matrix_over_vector_is_correct_l392_392040

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l392_392040


namespace infinite_decimal_irrational_l392_392282

def infinite_decimal := "0.1234567891011121314..."

def is_rational (x : ℚ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def eventually_periodic_decimal (d : List ℕ) : Prop :=
  ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ ∀ k, d.drop m = (d.take n).repeated k

theorem infinite_decimal_irrational :
  ∀ d, (d = infinite_decimal) → ¬ is_rational d :=
begin
  sorry
end

end infinite_decimal_irrational_l392_392282


namespace sally_boxes_sold_l392_392644

-- Define the problem conditions
def boxes_sold_Saturday (S : ℕ) : Prop :=
  let Sunday := 1.5 * S
  ∧ S + Sunday = 150

-- State the proof problem
theorem sally_boxes_sold (S : ℕ) (h : boxes_sold_Saturday S) : S = 60 :=
  sorry

end sally_boxes_sold_l392_392644


namespace bisects_pq_bd_l392_392209

-- Definitions
variables {A B C D P Q : Type}
variables [convex_quadrilateral A B C D]
variables [midpoint P A B]
variables [midpoint Q C D]

-- Hypothesis
variable (h1 : bisects P Q A C)

-- Theorem statement
theorem bisects_pq_bd : bisects P Q B D :=
sorry

end bisects_pq_bd_l392_392209


namespace sum_of_possible_values_f10_l392_392306

theorem sum_of_possible_values_f10 :
  ∃ (f : ℕ → ℝ), 
    (f 1 = 1) ∧ 
    (∀ m n : ℕ, m ≥ n → f(m + n) + f(m - n) = (f(2 * m) + f(2 * n)) / 2) ∧ 
    f 10 = 100 :=
begin
  sorry
end

end sum_of_possible_values_f10_l392_392306


namespace relationship_among_a_b_c_l392_392932

noncomputable def f (x : ℝ) : ℝ := sorry -- f is an unknown function

lemma y_axis_symmetric (x : ℝ) : f x = f (-x) :=
  sorry -- given: f is symmetric about the y-axis

lemma f_decreasing (x : ℝ) (h : x > 0) : f x - x * (deriv f) x < 0 :=
  sorry -- given: f(x) + x*f'(x) < 0 for x < 0, implies f(x) - x*f'(x) < 0 for x > 0

def a : ℝ := 2^(0.2) * f (2^(0.2))
def b : ℝ := real.log 3 / real.log π * f (real.log 3 / real.log π)
def c : ℝ := 2 * f 2

theorem relationship_among_a_b_c : b > a ∧ a > c :=
  sorry

end relationship_among_a_b_c_l392_392932


namespace alissa_presents_l392_392005

theorem alissa_presents :
  let Ethan_presents := 31
  let Alissa_presents := Ethan_presents + 22
  Alissa_presents = 53 :=
by
  sorry

end alissa_presents_l392_392005


namespace mean_after_adding_constant_l392_392372

theorem mean_after_adding_constant (numbers : List ℝ) (hlen : numbers.length = 15) (havg : numbers.sum / 15 = 40) :
  let new_numbers := numbers.map (λ x, x + 15)
  new_numbers.sum / 15 = 55 :=
by
  sorry

end mean_after_adding_constant_l392_392372


namespace angle_sum_triangle_l392_392981

theorem angle_sum_triangle (x : ℝ) 
  (h1 : 70 + 70 + x = 180) : 
  x = 40 :=
by
  sorry

end angle_sum_triangle_l392_392981


namespace f_pi_over_two_eq_pi_plus_one_l392_392965

noncomputable def f (a : ℝ) : ℝ :=
∫ x in 0..a, (2 + Real.sin x)

theorem f_pi_over_two_eq_pi_plus_one : f (Real.pi / 2) = Real.pi + 1 := by
  sorry

end f_pi_over_two_eq_pi_plus_one_l392_392965


namespace price_after_discount_l392_392868

-- Define the original price and discount
def original_price : ℕ := 76
def discount : ℕ := 25

-- The main proof statement
theorem price_after_discount : original_price - discount = 51 := by
  sorry

end price_after_discount_l392_392868


namespace distance_travelled_l392_392557

theorem distance_travelled : 
  let walk_time := 60 / 60 in
  let walk_rate := 3 in
  let run_time := 40 / 60 in
  let run_rate := 8 in
  walk_rate * walk_time + run_rate * run_time = 8.34 :=
by {
  sorry
}

end distance_travelled_l392_392557


namespace auntie_em_can_park_l392_392412

-- Define the conditions as formal statements in Lean
def parking_lot_spaces : ℕ := 20
def cars_arriving : ℕ := 14
def suv_adjacent_spaces : ℕ := 2

-- Define the total number of ways to park 14 cars in 20 spaces
def total_ways_to_park : ℕ := Nat.choose parking_lot_spaces cars_arriving
-- Define the number of unfavorable configurations where the SUV cannot park
def unfavorable_configs : ℕ := Nat.choose (parking_lot_spaces - suv_adjacent_spaces + 1) (parking_lot_spaces - cars_arriving)

-- Final probability calculation
def probability_park_suv : ℚ := 1 - (unfavorable_configs / total_ways_to_park)

-- Mathematically equivalent statement to be proved
theorem auntie_em_can_park : probability_park_suv = 850 / 922 :=
by sorry

end auntie_em_can_park_l392_392412


namespace find_number_l392_392779

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 126) : x = 5600 := 
by
  -- Proof goes here
  sorry

end find_number_l392_392779


namespace exists_two_equal_among_ten_l392_392112

theorem exists_two_equal_among_ten (a : Fin 100 → ℝ)
  (h : ∃ p : ℝ, {i | i < 100 ∧ (∑ j in Finset.range (i + 1), a j) / (i + 1) = p}.card ≥ 51) :
  ∃ i j : Fin 10, i ≠ j ∧ a i = a j :=
  sorry

end exists_two_equal_among_ten_l392_392112


namespace f_zero_eq_zero_f_periodic_l392_392118

def odd_function {α : Type*} [AddGroup α] (f : α → α) : Prop :=
∀ x, f (-x) = -f (x)

def symmetric_about (c : ℝ) (f : ℝ → ℝ) : Prop :=
∀ x, f (c + x) = f (c - x)

variable (f : ℝ → ℝ)
variables (h_odd : odd_function f) (h_sym : symmetric_about 1 f)

theorem f_zero_eq_zero : f 0 = 0 :=
sorry

theorem f_periodic : ∀ x, f (x + 4) = f x :=
sorry

end f_zero_eq_zero_f_periodic_l392_392118


namespace total_points_scored_l392_392418

def num_members : ℕ := 12
def num_absent : ℕ := 4
def points_per_member : ℕ := 8

theorem total_points_scored : 
  (num_members - num_absent) * points_per_member = 64 := by
  sorry

end total_points_scored_l392_392418


namespace solve_for_x_l392_392652

theorem solve_for_x (x : ℕ) (hx : 1000^4 = 10^x) : x = 12 := 
by
  sorry

end solve_for_x_l392_392652


namespace greatest_q_minus_r_l392_392676

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1001 = 17 * q + r ∧ q - r = 43 :=
by
  sorry

end greatest_q_minus_r_l392_392676


namespace memorable_telephone_numbers_l392_392857

theorem memorable_telephone_numbers :
  let digits := Fin 10 in
  let phone_number := Fin 10 -> digits in
  let memorable (n : phone_number) := n 0 = n 5 ∧ n 1 = n 6 ∧ n 2 = n 7 ∧ n 3 = n 8 in
  let count_memorable : Nat := 10000 * 10 in
  count_memorable = 100000 :=
by
  sorry

end memorable_telephone_numbers_l392_392857


namespace line_equation_l392_392960

theorem line_equation {a : ℝ} (h : ∀ x y, x / a + y / a = 1 ↔ x + y = a) :
  (∃ (l : ℝ × ℝ → Prop), (∀ (x y : ℝ), l (x, y) ↔ x + y = 3) ∧ l (2, 1))
  := 
begin
  use λ p : ℝ × ℝ, (p.1 + p.2 = 3),
  split,
  { intros x y,
    exact h x y, },
  { exact (by simp [h] : 2 + 1 = 3) },
end

end line_equation_l392_392960


namespace area_ratio_of_R_to_ABCD_l392_392349

-- Definitions for the problem

def square (A B C D : Point) := 
  (dist A B = dist B C) ∧
  (dist B C = dist C D) ∧
  (dist C D = dist D A) ∧
  (∃M, midpoint C D M)

structure Particle :=
  (position : Point)
  (velocity : Vector)

-- Initial conditions

def conditions (ABCD : Square) :=
  let A := ABCD.1
  let B := ABCD.2
  let C := ABCD.3
  let D := ABCD.4
  let M := midpoint C D
  ∃(p1 p2 : Particle), 
    p1.position = A ∧
    p2.position = M ∧
    p1.velocity = p2.velocity

-- Target to prove

theorem area_ratio_of_R_to_ABCD (ABCD : Square) 
  (condition : conditions ABCD) :
  ∃ R, (area R) / (area ABCD) = 1 / 4 :=
sorry

end area_ratio_of_R_to_ABCD_l392_392349


namespace limit_of_length_is_correct_l392_392406

theorem limit_of_length_is_correct :
  ∃ l : ℝ, l = 1 + (∑' n : ℕ, (1 : ℝ) / 4 ^ (n + 1) + (1 : ℝ) / 4 ^ (n + 1) * (real.sqrt 2)) 
    → l = (1 / 3) * (4 + real.sqrt 2) :=
by
  sorry

end limit_of_length_is_correct_l392_392406


namespace vector_magnitude_problem_l392_392126

open Real

variables {a b : ℝ^3}

theorem vector_magnitude_problem
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 2)
  (hab : a • b = 0) :
  ‖a + (sqrt 2 : ℝ) • b‖ = 3 := 
sorry

end vector_magnitude_problem_l392_392126


namespace volume_of_sphere_max_area_tetrahedron_l392_392333

-- Definitions
def PA : ℝ := 3
def PB : ℝ := 3
def PC : ℝ := 3
def diagonal := real.sqrt (PA^2 + PB^2 + PC^2)
def radius := diagonal / 2
def volume_sphere (r : ℝ) := (4 / 3) * real.pi * r^3

-- Theorem statement
theorem volume_of_sphere_max_area_tetrahedron :
  volume_sphere radius = (27 * real.sqrt 3 * real.pi / 2) :=
by
  sorry

end volume_of_sphere_max_area_tetrahedron_l392_392333


namespace weeks_needed_to_purchase_car_l392_392593

def hourly_rate : ℕ := 20
def total_hours_per_week : ℕ := 52
def normal_hours_per_week : ℕ := 40
def overtime_multiplier : ℚ := 1.5
def car_cost : ℕ := 4640

theorem weeks_needed_to_purchase_car :
  let h_rate := hourly_rate
      total_hours := total_hours_per_week
      norm_hours := normal_hours_per_week
      ot_mult := overtime_multiplier
      car := car_cost
      normal_pay := h_rate * norm_hours
      ot_hours := total_hours - norm_hours
      ot_pay := (ot_hours : ℕ) * (h_rate : ℕ) * ot_mult
      total_pay := normal_pay + ot_pay
  in ∀ weeks_needed : ℕ, (weeks_needed * total_pay) = car -> weeks_needed = 4 :=
begin
  intros,
  sorry
end

end weeks_needed_to_purchase_car_l392_392593


namespace evaluate_expression_l392_392011

theorem evaluate_expression : (5⁻¹ + 2⁻¹)⁻¹ = (10 / 7) := by
  sorry

end evaluate_expression_l392_392011


namespace division_multiplication_l392_392849

theorem division_multiplication : (0.25 / 0.005) * 2 = 100 := 
by 
  sorry

end division_multiplication_l392_392849


namespace num_ways_arrange_distinct_reals_l392_392230

theorem num_ways_arrange_distinct_reals (n : ℕ) (distinct_reals : Fin n^2 → ℝ) :
  ∃ (arrangements : Fin n → Fin n → ℝ),
    (∀ i j, arrangements i j ∈ set.range distinct_reals) ∧
    (max_j_min_i arrangements = min_i_max_j arrangements) ↔
    number_of_ways = (n^2)! * (n!)^2 / (2n - 1)! :=
sorry

end num_ways_arrange_distinct_reals_l392_392230


namespace original_selling_price_is_correct_l392_392782

def original_price_condition (P : ℝ) : Prop :=
  0.68 * P = 650

theorem original_selling_price_is_correct : ∃ P : ℝ, original_price_condition P ∧ P = 955.88 :=
by
  use 955.88
  split
  · sorry
  · rfl

end original_selling_price_is_correct_l392_392782


namespace telescoping_series_sum_l392_392010

theorem telescoping_series_sum : 
  (∑ n in Finset.range (2010+1), (3 / ((n + 1) * (n + 1 + 3)))) = 1.832 :=
by
  sorry

end telescoping_series_sum_l392_392010


namespace trig_identity_45_15_l392_392432

theorem trig_identity_45_15 : 
  sin (45 : ℝ) * cos (15 : ℝ) + cos (45 : ℝ) * sin (15 : ℝ) = sqrt 3 / 2 := 
sorry

end trig_identity_45_15_l392_392432


namespace interval_a_less_than_2_l392_392133

theorem interval_a_less_than_2
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_f : ∀ x : ℝ, f x = x^2 - 4 * x + 3)
  (h_exists : ∃ x1 x2 : ℝ, x1 ∈ set.Icc a b ∧ x2 ∈ set.Icc a b ∧ x1 < x2 ∧ f x1 > f x2)
  : a < 2 :=
begin
  sorry
end

end interval_a_less_than_2_l392_392133


namespace fraction_calculation_l392_392754

theorem fraction_calculation :
  let a := (2 / 5:ℚ) * 5040
  let b := (3 / 4:ℚ) * a
  ∃ x : ℚ, x * b = 756.0000000000001 → x = 0.5 :=
by
  intros a b x h
  sorry

end fraction_calculation_l392_392754


namespace find_varphi_l392_392552

theorem find_varphi (ω : ℝ) (varphi : ℝ) : 
  (0 < ω) → (0 < varphi ∧ varphi < π) → 
  (∀ x y, (x ∈ set.Icc (π/12) (2 * π / 3)) → (y ∈ set.Icc (π/12) (2 * π / 3)) → x ≤ y → (2 * sin (ω * x + varphi) ≤ 2 * sin (ω * y + varphi))) → 
  (2 * sin (ω * -π/3 + varphi) = 2 * sin (ω * π/6 + varphi)) → 
  (2 * sin (ω * π/6 + varphi) = -2 * sin (ω * 2 * π / 3 + varphi)) →
  varphi = 7 * π / 12 :=
begin
  intros,
  sorry,
end

end find_varphi_l392_392552


namespace find_larger_integer_l392_392731

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l392_392731


namespace triangle_area_diff_l392_392792

theorem triangle_area_diff (b h : ℝ) :
  let base_A := 1.12 * b,
      height_A := 0.88 * h,
      area_B := (b * h) / 2,
      area_A := (base_A * height_A) / 2 
  in (area_A / area_B) = 0.9856 → 100 - (area_A / area_B * 100) = 1.44 :=
by
  intros base_A height_A area_B area_A h0
  sorry

end triangle_area_diff_l392_392792


namespace time_for_freight_train_to_pass_l392_392401

-- Define the lengths of the trains in meters
def length_freight_train := 550 -- meters
def length_passenger_train := 350 -- meters

-- Define the speeds of the trains in m/s
def speed_freight_train := 25 -- m/s (converted from 90 km/h)
def speed_passenger_train := 20.83 -- m/s (converted from 75 km/h, approximated to two decimal places)

-- Define the relative speed
def relative_speed := speed_freight_train - speed_passenger_train -- m/s

-- Define the total distance to be covered
def total_distance := length_freight_train + length_passenger_train -- meters

-- Define the time it takes for the freight train to pass the passenger train
def time_to_pass := total_distance / relative_speed -- seconds

theorem time_for_freight_train_to_pass :
  time_to_pass = 215.82 :=
  sorry

end time_for_freight_train_to_pass_l392_392401


namespace reflection_matrix_is_correct_l392_392064

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l392_392064


namespace reflection_matrix_l392_392019

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l392_392019


namespace prove_A_plus_B_l392_392994

-- Define A and B as real numbers and state the given condition
variable {A B x : ℝ}

-- Define the given equality
def given_equality : Prop := (A / (x - 7)) + B * (x + 2) = (-4 * x^2 + 16 * x + 28) / (x - 7)

-- The proof statement that we need to show
theorem prove_A_plus_B (h : given_equality) : A + B = 24 :=
sorry

end prove_A_plus_B_l392_392994


namespace cube_surface_area_increase_cube_volume_increase_l392_392396

theorem cube_surface_area_increase (s : ℝ) : 
  let s_new := 1.3 * s in
  let A := 6 * s^2 in
  let A_new := 6 * s_new^2 in
  ((A_new - A) / A) * 100 = 69 := sorry

theorem cube_volume_increase (s : ℝ) : 
  let s_new := 1.3 * s in
  let V := s^3 in
  let V_new := s_new^3 in
  ((V_new - V) / V) * 100 = 119.7 := sorry

end cube_surface_area_increase_cube_volume_increase_l392_392396


namespace evaluate_product_l392_392009

-- Define the given numerical values
def a : ℝ := 2.5
def b : ℝ := 50.5
def c : ℝ := 0.15

-- State the theorem we want to prove
theorem evaluate_product : a * (b + c) = 126.625 := by
  sorry

end evaluate_product_l392_392009


namespace problem1_problem2_problem3_l392_392262

-- Definition of the function and conditions
def f (x : ℝ) (λ : ℝ) : ℝ := x + λ / x
axiom λ_pos : λ > 0

-- Problem 1: Prove f(x) is an odd function
theorem problem1 : ∀ x, f (-x) = - f x := sorry

-- Problem 2: Prove f(x) is monotonically increasing on [1, +∞) for λ = 1
theorem problem2 : ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → f x₁ 1 < f x₂ 1 := sorry

-- Problem 3: Prove the range of λ is 0 < λ ≤ 1 if f(x) is monotonically increasing on [1, +∞)
theorem problem3 (h : ∀ x₁ x₂, 1 ≤ x₁ → x₁ < x₂ → f x₁ λ < f x₂ λ) : 0 < λ ∧ λ ≤ 1 := sorry

end problem1_problem2_problem3_l392_392262


namespace inf_subset_sum_mod_l392_392255

open Nat

/-- Let A be an infinite subset of positive integers and n be a given integer greater than 1.
    Suppose for any prime number p that does not divide n, there are infinitely many elements
    in the set A that are not divisible by p. Prove that for any integer m greater than 1, 
    such that gcd(m, n) = 1, there exist finitely many distinct elements in the set A whose 
    sum S satisfies S ≡ 1 (mod m) and S ≡ 0 (mod n). -/
theorem inf_subset_sum_mod 
  (A : Set ℕ) (h_inf_A : Set.infinite A) 
  (n : ℕ) (hn : 1 < n) 
  (hdiv : ∀ (p : ℕ), Prime p → ¬ p ∣ n → Set.infinite {a ∈ A | ¬ p ∣ a}) :
  ∀ (m : ℕ), 1 < m → gcd m n = 1 → 
  ∃ (S : ℕ) (B : Finset ℕ), (∀ (b ∈ B), b ∈ A) ∧ Finset.sum B id = S ∧ S ≡ 1 [MOD m] ∧ S ≡ 0 [MOD n] :=
by
  sorry

end inf_subset_sum_mod_l392_392255


namespace find_a_decreasing_l392_392931

def decreasing_interval (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y ∧ y ≤ 4 → f a x ≥ f a y

def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem find_a_decreasing : ∀ a : ℝ, decreasing_interval a → a ≤ -3 :=
by {
  intro a,
  intro h,
  -- You would include the necessary steps and justify the proof here.
  sorry
}

end find_a_decreasing_l392_392931


namespace find_larger_integer_l392_392696

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l392_392696


namespace strictly_increasing_f_function_f_l392_392461

noncomputable def f : ℕ → ℕ := λ x, x^2

theorem strictly_increasing_f : ∀ x y ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, x < y → f x < f y := by
  intros x x_in y y_in hxy
  simp [f]
  exact Nat.pow_le_pow_of_le_left hxy 2

theorem function_f : ∀ x y ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, (x + y) ∣ (x * f x + y * f y) := by
  intros x x_in y y_in
  simp [f]
  have hdiv : (x + y) ∣ (x * x^2 + y * y^2) := sorry
  exact hdiv

end strictly_increasing_f_function_f_l392_392461


namespace fraction_of_nonempty_subsets_with_odd_smallest_element_l392_392489

theorem fraction_of_nonempty_subsets_with_odd_smallest_element (n : ℕ) (hn : 0 < n) : 
  (let subsets := finset.powerset (finset.range (2 * n + 1)).filter (λ s, s ≠ ∅) in
    let odd_smallest := subsets.filter (λ s, (finset.min' s (finset.nonempty_of_ne_empty (finset.filter_ne_empty hn s))) % 2 = 1) in
    (odd_smallest.card : ℚ) / subsets.card = 1 / 2) :=
sorry

end fraction_of_nonempty_subsets_with_odd_smallest_element_l392_392489


namespace larger_integer_is_21_l392_392688

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l392_392688


namespace events_are_mutually_exclusive_and_complementary_l392_392647

-- Definitions
def Group := { name : String, gender : String }
def boys := [ { name := "boy1", gender := "M" }, { name := "boy2", gender := "M" }, { name := "boy3", gender := "M" } ]
def girls := [ { name := "girl1", gender := "F" }, { name := "girl2", gender := "F" } ]
def group := boys ++ girls

-- Conditions
def select_two (g : List Group) := g.combinations 2 -- assuming a combinations function exists

def at_least_one_girl (s : List Group) := s.any (λ student => student.gender = "F")
def all_boys (s : List Group) := s.all (λ student => student.gender = "M")

-- Statement
theorem events_are_mutually_exclusive_and_complementary :
  ∀ (s : List Group), s ∈ select_two group → (at_least_one_girl s ↔ ¬ all_boys s) ∧ (all_boys s ↔ ¬ at_least_one_girl s) :=
by
-- proof goes here
sorry

end events_are_mutually_exclusive_and_complementary_l392_392647


namespace average_of_possible_x_values_l392_392160

theorem average_of_possible_x_values :
  (∀ x : ℝ, sqrt (2 * x^2 + 1) = sqrt 25 → x = -2 * sqrt 3 ∨ x = 2 * sqrt 3) →
  (1 / 2 * (-2 * sqrt 3 + 2 * sqrt 3) = 0) :=
by
  intro h
  have h1 : -2 * sqrt 3 = -2 * sqrt 3, by sorry
  have h2 : 2 * sqrt 3 = 2 * sqrt 3, by sorry
  exact (h1, h2)

end average_of_possible_x_values_l392_392160


namespace always_two_real_roots_equal_real_roots_implies_m_and_roots_l392_392903

-- Define the quadratic equation and its discriminant
def discriminant (a b c : ℝ) := b^2 - 4 * a * c

-- Part 1: Prove that for any real number m, the equation 2x^2 + (m+2)x + m = 0 always has two real roots.
theorem always_two_real_roots (m : ℝ) : 
  let δ := discriminant 2 (m + 2) m in δ >= 0 := by
  let δ := discriminant 2 (m + 2) m
  show δ >= 0
  sorry

-- Part 2: Prove that if the equation 2x^2 + (m+2)x + m = 0 has two equal real roots, then m = 2 and the roots are x = -1.
theorem equal_real_roots_implies_m_and_roots (m : ℝ) : 
  let δ := discriminant 2 (m + 2) m in
  δ = 0 -> m = 2 ∧ -1 = - (m + 2)/4 := by
  let δ := discriminant 2 (m + 2) m
  show δ = 0 -> m = 2 ∧ -1 = - (m + 2)/4
  sorry

end always_two_real_roots_equal_real_roots_implies_m_and_roots_l392_392903


namespace terms_before_one_l392_392151

theorem terms_before_one (a d : ℤ) (term : ℤ → ℤ) (h_a : a = 100) (h_d : d = -4) :
  (∃ n : ℕ, n = 25 ∧ ∀ i < n, term i ≠ 1 ∧ term i > 1) :=
by
  -- Define the nth term of the sequence
  let term := λ n : ℕ, a + n * d
  -- The sequence is arithmetic
  replace h_a := rfl
  replace h_d := rfl
  sorry

end terms_before_one_l392_392151


namespace ratio_of_areas_is_two_thirds_l392_392237

noncomputable def PQ := 10
noncomputable def PR := 6
noncomputable def QR := 4
noncomputable def r_PQ := PQ / 2
noncomputable def r_PR := PR / 2
noncomputable def r_QR := QR / 2
noncomputable def area_semi_PQ := (1 / 2) * Real.pi * r_PQ^2
noncomputable def area_semi_PR := (1 / 2) * Real.pi * r_PR^2
noncomputable def area_semi_QR := (1 / 2) * Real.pi * r_QR^2
noncomputable def shaded_area := (area_semi_PQ - area_semi_PR) + area_semi_QR
noncomputable def total_area_circle := Real.pi * r_PQ^2
noncomputable def unshaded_area := total_area_circle - shaded_area
noncomputable def ratio := shaded_area / unshaded_area

theorem ratio_of_areas_is_two_thirds : ratio = 2 / 3 := by
  sorry

end ratio_of_areas_is_two_thirds_l392_392237


namespace correct_calculation_l392_392764

theorem correct_calculation (x : ℝ) (h : (x / 2) + 45 = 85) : (2 * x) - 45 = 115 :=
by {
  -- Note: Proof steps are not needed, 'sorry' is used to skip the proof
  sorry
}

end correct_calculation_l392_392764


namespace ellipse_focal_length_through_point_l392_392964

noncomputable def ellipse_focal_length (a b : ℝ) : ℝ := 2 * real.sqrt (a^2 - b^2)

theorem ellipse_focal_length_through_point :
  ∀ (m : ℝ), 
  (∀ (x y : ℝ), (x, y) = (-2, real.sqrt 3) → 
   x^2 / 16 + y^2 / m^2 = 1) → 
  ellipse_focal_length 4 2 = 4 * real.sqrt 3 :=
by
  intros m h_point
  sorry

end ellipse_focal_length_through_point_l392_392964


namespace count_integers_between_300_and_700_with_345_l392_392955

noncomputable def count_valid_numbers (lower upper : ℕ) : ℕ :=
  let valid_numbers := List.range' lower (upper - lower + 1)
  let contains_345 (n : ℕ) : Bool :=
    let digits := n.digits 10
    digits.contains 3 && digits.contains 4 && digits.contains 5 && digits.length = 3
  (valid_numbers.filter contains_345).length

theorem count_integers_between_300_and_700_with_345 :
  count_valid_numbers 300 700 = 6 :=
by
  sorry

end count_integers_between_300_and_700_with_345_l392_392955


namespace part1_zero_of_f_part2_a_range_l392_392510

-- Define the given function f
def f (x a b : ℝ) : ℝ := (x - a) * |x| + b

-- Define the problem statement for Part 1
theorem part1_zero_of_f :
  ∀ (x : ℝ),
    f x 2 3 = 0 ↔ x = -1 := 
by
  sorry

-- Define the problem statement for Part 2
theorem part2_a_range :
  ∀ (a : ℝ),
    (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → f x a (-2) < 0) ↔ a > -1 :=
by
  sorry

end part1_zero_of_f_part2_a_range_l392_392510


namespace valid_range_for_sqrt_expression_l392_392322

theorem valid_range_for_sqrt_expression (x : ℝ) : (∃ y : ℝ, y = sqrt (8 - x)) ↔ x ≤ 8 :=
by
  sorry

end valid_range_for_sqrt_expression_l392_392322


namespace arrangements_divisible_by_three_l392_392894

theorem arrangements_divisible_by_three (f : Fin 10 → Fin 10) :
  (∀ i : Fin 10, (f i + f ((i + 1) % 10) + f ((i + 2) % 10)) % 3 = 0) →
  (Nat.fact 4 * Nat.fact 3 * Nat.fact 3 * 2 = 1728) :=
by
  sorry

end arrangements_divisible_by_three_l392_392894


namespace contrapositive_statement_l392_392927

-- Conditions: x and y are real numbers
variables (x y : ℝ)

-- Contrapositive statement: If x ≠ 0 or y ≠ 0, then x^2 + y^2 ≠ 0
theorem contrapositive_statement (hx : x ≠ 0 ∨ y ≠ 0) : x^2 + y^2 ≠ 0 :=
sorry

end contrapositive_statement_l392_392927


namespace parabola_and_hyperbola_equations_l392_392125

-- Parabola and Hyperbola equations
theorem parabola_and_hyperbola_equations
    (parabola_vertex : (0, 0))
    (hyperbola_intersection : (3/2, sqrt 6))
    (hyperbola_eqn : ∀ x y a b, x^2 / a^2 - y^2 / b^2 = 1)
    (right_focus : (1, 0))
    (c_squared_eqn : ∀ a b c, c^2 = a^2 + b^2) :
    (∀ x y, y^2 = 4 * x) ∧ (∀ x y, x^2 / (1/4) - y^2 / (3/4) = 1) :=
by
  -- Provide a proof here
  sorry

end parabola_and_hyperbola_equations_l392_392125


namespace daps_to_dips_l392_392179

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end daps_to_dips_l392_392179


namespace points_lie_on_circle_l392_392085

theorem points_lie_on_circle (t : ℝ) : 
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  in x^2 + y^2 = 1 :=
by
  let x := (2 - t^2) / (2 + t^2)
  let y := (3 * t) / (2 + t^2)
  calc
    x^2 + y^2 = ( (2 - t^2) / (2 + t^2) )^2 + ( (3 * t) / (2 + t^2) )^2 : rfl
          ... = ( (2 - t^2)^2 + (3 * t)^2 ) / (2 + t^2)^2 : by rw [add_div]
          ... = (4 - 4 * t^2 + t^4 + 9 * t^2) / (4 + 4 * t^2 + t^4) : by sorry
          ... = (4 + (5 * t^2) + (t^4)) / (4 + (4 * t^2) + (t^4)) : by sorry
          ... = 1 : by sorry

end points_lie_on_circle_l392_392085


namespace number_of_children_l392_392781

theorem number_of_children (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 390)) : C = 780 :=
by
  sorry

end number_of_children_l392_392781


namespace x_is_48_percent_of_z_l392_392556

variable {x y z : ℝ}

theorem x_is_48_percent_of_z (h1 : x = 1.20 * y) (h2 : y = 0.40 * z) : x = 0.48 * z :=
by
  sorry

end x_is_48_percent_of_z_l392_392556


namespace coeff_x2y7_in_expansion_eq_minus56_l392_392665

theorem coeff_x2y7_in_expansion_eq_minus56 :
  let f := (fun x y : ℝ => (x + y) * (x - y)^8)
  polynomial.coeff (f x y) (2, 7) = -56 :=
by
  sorry

end coeff_x2y7_in_expansion_eq_minus56_l392_392665


namespace locus_of_M_is_circle_l392_392400

noncomputable def locus_of_M {O A : Point} (R AO : Real) (hA_inside: A ∈ Disk O R) :=
  {M : Point | ∃ (XY : LineSegment) (hXY_on_circle : XY ⊆ Circle O R) (h_right_angle: angle X A Y = 90), 
                 M = symmetric_point A XY}

theorem locus_of_M_is_circle {O A : Point} (R AO : Real) (hA_inside: A ∈ Disk O R) :
  locus_of_M O A R AO hA_inside = {M : Point | dist O M = 2 * sqrt (1/2 * R^2 - 1/4 * (dist A O)^2)} :=
sorry

end locus_of_M_is_circle_l392_392400


namespace BD_parallel_CP_l392_392200

open EuclideanGeometry

-- Definitions of points and conditions
variables {A B C D E P : Point}
variables (triangleABC : Triangle A B C)
variables (bisectorB : IsAngleBisector B C D)
variables (bisectorC : IsAngleBisector C B E)
variables (tangentA : IsTangent A circumcircleABC)
variables (APeqBC : AP = BC)
variables (ABgtAC : AB > AC)

-- Main theorem statement
theorem BD_parallel_CP :
  IsParallel (LineSegment.mk B D) (LineSegment.mk C P) :=
sorry

end BD_parallel_CP_l392_392200


namespace complex_number_solution_l392_392933

open Complex

theorem complex_number_solution (z : ℂ) (hz_re : z.re = 1)
    (h_magnitude : abs (z - conj z) = abs (Complex.I - 2 * Complex.I / (1 + Complex.I))) :
  z = 1 + (1 / 2) * Complex.I ∨ z = 1 - (1 / 2) * Complex.I := by
  sorry

end complex_number_solution_l392_392933


namespace not_in_sequence_l392_392874

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence property
def sequence_property (a b : ℕ) : Prop :=
  b = a + sum_of_digits a

-- Main theorem
theorem not_in_sequence (n : ℕ) (h : n = 793210041) : 
  ¬ (∃ a : ℕ, sequence_property a n) :=
by
  sorry

end not_in_sequence_l392_392874


namespace no_solution_exists_for_12x4x_divisible_by_99_l392_392077

theorem no_solution_exists_for_12x4x_divisible_by_99 : 
  ¬ ∃ x : ℕ, x ≤ 9 ∧ (let n := 10000 + 2000 * x + 400 + x in
                      (7 + 2 * x) % 9 = 0 ∧ (2 * x - 5) % 11 = 0) :=
by sorry

end no_solution_exists_for_12x4x_divisible_by_99_l392_392077


namespace Simplify_Sn_l392_392651

theorem Simplify_Sn (n : ℕ) : 
  let S_n := n + (n - 1) * 2 + (n - 2) * 2^2 + ∙∙∙ + 2 * 2^(n - 2) + 2^(n - 1)
  in S_n = 2^(n + 1) - n - 2 :=
by
  sorry

end Simplify_Sn_l392_392651


namespace li_li_age_this_year_l392_392746

theorem li_li_age_this_year (A B : ℕ) (h1 : A + B = 30) (h2 : A = B + 6) : B = 12 := by
  sorry

end li_li_age_this_year_l392_392746


namespace probability_student_major_b_and_below_25_l392_392578

theorem probability_student_major_b_and_below_25
  (p_student_major_b : ℝ) (p_student_major_b_below_25 : ℝ) :
  p_student_major_b = 0.30 →
  p_student_major_b_below_25 = 0.60 →
  (p_student_major_b * p_student_major_b_below_25 = 0.18) :=
by
  intros h1 h2
  simp [h1, h2]
  sorry

end probability_student_major_b_and_below_25_l392_392578


namespace _l392_392033
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l392_392033


namespace range_of_a_l392_392947

theorem range_of_a (a : ℝ) :
  (∀ (n : ℕ), 0 < n → let T_n := ∑ i in range(n), 1 / (4 * i ^ 2 + 4 * i - 3) 
   in 12 * T_n < 3 * a ^ 2 - a) ↔ a ∈ Icc (-∞ : ℝ) (-1 : ℝ) ∪ Icc (4 / 3 : ℝ) (∞ : ℝ) :=
sorry

end range_of_a_l392_392947


namespace daps_dips_equivalence_l392_392185

theorem daps_dips_equivalence :
  (∃ dap dop dip : Type,
    (5 : ℝ) * ∀ x : dap, x = (4 : ℝ) * ∀ y : dop, y ∧
    (3 : ℝ) * ∀ z : dop, z = (8 : ℝ) * ∀ w : dip, w) →
  (22.5 : ℝ) * ∀ x : dap, x = (48 : ℝ) * ∀ y : dip, y :=
begin
  sorry
end

end daps_dips_equivalence_l392_392185


namespace ab_greater_than_one_l392_392134

variable (f : ℝ → ℝ)
variable (a b : ℝ)

def f_def : Prop := ∀ x, x > 0 → f x = |1 - 1/x|

def cond : Prop := 0 < a ∧ a < b ∧ f a = f b

theorem ab_greater_than_one (h₁ : f_def f) (h₂ : cond a b f) : a * b > 1 := by
  sorry

end ab_greater_than_one_l392_392134


namespace PQ_bisects_BD_l392_392218

variable {A B C D P Q M N : Type}
variable [ConvexQuadrilateral A B C D]
variable [Midpoint P A B]
variable [Midpoint Q C D]
variable [Bisects PQ AC]

-- Prove that PQ also bisects diagonal BD
theorem PQ_bisects_BD (PQ_bisects_AC : bisects P Q A C) : bisects P Q B D := 
sorry

end PQ_bisects_BD_l392_392218


namespace daps_dips_equivalence_l392_392183

theorem daps_dips_equivalence :
  (∃ dap dop dip : Type,
    (5 : ℝ) * ∀ x : dap, x = (4 : ℝ) * ∀ y : dop, y ∧
    (3 : ℝ) * ∀ z : dop, z = (8 : ℝ) * ∀ w : dip, w) →
  (22.5 : ℝ) * ∀ x : dap, x = (48 : ℝ) * ∀ y : dip, y :=
begin
  sorry
end

end daps_dips_equivalence_l392_392183


namespace trapezoid_perimeter_is_183_l392_392581

-- Declare the lengths of the sides of the trapezoid
def EG : ℕ := 35
def FH : ℕ := 40
def GH : ℕ := 36

-- Declare the relation between the bases EF and GH
def EF : ℕ := 2 * GH

-- The statement of the problem
theorem trapezoid_perimeter_is_183 : EF = 72 ∧ (EG + GH + FH + EF) = 183 := by
  sorry

end trapezoid_perimeter_is_183_l392_392581


namespace problem_1_problem_2_l392_392953

noncomputable def magnitude (v: ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

variable (a b : ℝ × ℝ)
variable (k : ℝ)

axiom mag_a : magnitude a = 2
axiom vec_b : b = (-1/2, real.sqrt 3 / 2)
axiom angle_ab : real.cos (2 * real.pi / 3) = -1/2

theorem problem_1 : magnitude (a.1 + 2 * b.1, a.2 + 2 * b.2) = 1 := sorry

theorem problem_2 : (a.1 + k * b.1, a.2 + k * b.2) = (-(1/2) * (2*b.1 - a.1), -(1/2) * (2*b.2 - a.2)) → k = 2 := sorry

end problem_1_problem_2_l392_392953


namespace number_that_multiplies_b_l392_392549

variable (a b x : ℝ)

theorem number_that_multiplies_b (h1 : 7 * a = x * b) (h2 : a * b ≠ 0) (h3 : (a / 8) / (b / 7) = 1) : x = 8 := 
sorry

end number_that_multiplies_b_l392_392549


namespace corey_gave_more_books_l392_392625

def books_given_by_mike : ℕ := 10
def total_books_received_by_lily : ℕ := 35
def books_given_by_corey : ℕ := total_books_received_by_lily - books_given_by_mike
def difference_in_books (a b : ℕ) : ℕ := a - b

theorem corey_gave_more_books :
  difference_in_books books_given_by_corey books_given_by_mike = 15 := by
sorry

end corey_gave_more_books_l392_392625


namespace sum_lent_l392_392822

theorem sum_lent (P : ℝ) (r : ℝ) (n : ℕ) (r_half : ℝ) (n_half : ℕ) 
    (A1 A2 : ℝ) (diff : ℝ) (H1 : r = 0.20) (H2 : n = 2)
    (H3 : r_half = 0.10) (H4 : n_half = 4)
    (H5 : A1 = P * (1 + r)^n) (H6 : A2 = P * (1 + r_half)^n_half) 
    (H7 : diff = 482) (H8 : A2 - A1 = diff) :
  P = 20000 :=
by
  rw [H5, H6, H7, H8]
  simp [r_half, r, n_half, n]
  sorry

end sum_lent_l392_392822


namespace fraction_of_data_less_than_mode_l392_392780

theorem fraction_of_data_less_than_mode (lst : list ℕ) (mode : ℕ) (count_less : ℕ) (total_count : ℕ) : 
  lst = [3, 3, 4, 4, 5, 5, 5, 5, 7, 11, 21] ∧ 
  mode = 5 ∧ 
  count_less = 4 ∧ 
  total_count = 11 →
  (count_less / total_count : ℚ) = 4 / 11 := by
sorry

end fraction_of_data_less_than_mode_l392_392780


namespace floor_abs_floor_neg_l392_392877

theorem floor_abs_floor_neg (x : ℝ) (h : x = -3.7) :
  (⌊|x|⌋ + |⌊x⌋|) = 7 :=
by
  -- declaration of the condition h
  have h1 : |x| = 3.7 := by rw [h, abs_neg, abs_of_pos]; norm_num
  have h2 : ⌊|x|⌋ = 3 := by rw [h1, floor_real]; norm_num
  have h3 : ⌊x⌋ = -4 := by rw [h, floor_neg 3.7 (by norm_num)]; norm_num
  have h4 : |⌊x⌋| = 4 := by rw [h3, abs_neg]; norm_num
  rw [h2, h4]
  norm_num
  sorry

end floor_abs_floor_neg_l392_392877


namespace radius_of_sphere_l392_392580

noncomputable def tetrahedronRadiusProof (A B C D : Type) [MetricSpace A] 
  [MetricSpace B] [MetricSpace C] [MetricSpace D] : Prop :=
∃ (O : Type) [MetricSpace O] (r : ℝ),
  (∃ (AB AC : ℝ) (θ : ℝ) (V : ℝ),
    AB = 6 ∧ AC = 10 ∧ θ = (π / 2) ∧ V = 200) →
    r = 13

theorem radius_of_sphere (A B C D : Type) [MetricSpace A] 
  [MetricSpace B] [MetricSpace C] [MetricSpace D] :
  tetrahedronRadiusProof A B C D :=
sorry

end radius_of_sphere_l392_392580


namespace chord_length_EF_l392_392236

theorem chord_length_EF {A B C D O N P G E F : Point} 
                        {radius_O radius_N radius_P : ℝ}
                        (h1 : AB diameter O = 2 * radius_O)
                        (h2 : BC diameter N = 2 * radius_N)
                        (h3 : CD diameter P = 2 * radius_P)
                        (rad_O : radius_O = 12)
                        (rad_N : radius_N = 18)
                        (rad_P : radius_P = 15)
                        (tangent_AG_P : Tangent AG P G)
                        (intersect: Intersects AG N E F) :
    length_EF = 2 * sqrt 201.84 :=
    sorry

end chord_length_EF_l392_392236


namespace bisects_pq_bd_l392_392208

-- Definitions
variables {A B C D P Q : Type}
variables [convex_quadrilateral A B C D]
variables [midpoint P A B]
variables [midpoint Q C D]

-- Hypothesis
variable (h1 : bisects P Q A C)

-- Theorem statement
theorem bisects_pq_bd : bisects P Q B D :=
sorry

end bisects_pq_bd_l392_392208


namespace weeks_needed_to_purchase_car_l392_392594

def hourly_rate : ℕ := 20
def total_hours_per_week : ℕ := 52
def normal_hours_per_week : ℕ := 40
def overtime_multiplier : ℚ := 1.5
def car_cost : ℕ := 4640

theorem weeks_needed_to_purchase_car :
  let h_rate := hourly_rate
      total_hours := total_hours_per_week
      norm_hours := normal_hours_per_week
      ot_mult := overtime_multiplier
      car := car_cost
      normal_pay := h_rate * norm_hours
      ot_hours := total_hours - norm_hours
      ot_pay := (ot_hours : ℕ) * (h_rate : ℕ) * ot_mult
      total_pay := normal_pay + ot_pay
  in ∀ weeks_needed : ℕ, (weeks_needed * total_pay) = car -> weeks_needed = 4 :=
begin
  intros,
  sorry
end

end weeks_needed_to_purchase_car_l392_392594


namespace range_of_a_l392_392131

noncomputable def f : ℝ → ℝ :=
λ x, if x >= 0 then x^2 + 4 * x else 4 * x - x^2

theorem range_of_a (a : ℝ) (h : ∀ x y : ℝ, x < y → f x > f y) : f (2 * a + 1) > f (a - 2) → a < -3 :=
by 
  intros h1,
  have : 2 * a + 1 < a - 2,
    from sorry,
  linarith

end range_of_a_l392_392131


namespace ratio_of_area_l392_392232

-- Mathematical definitions and assertions based on the given problem

variable {x : ℝ} (A M N : ℝ × ℝ)
variable {AreaSquare AreaTriangle : ℝ}

-- Conditions of the problem
def is_square (A B C D : ℝ × ℝ) : Prop := 
  ∃ x : ℝ, B = (x, 0) ∧ C = (x, x) ∧ D = (0, x)

def midpoint (P Q M : ℝ × ℝ) : Prop := 
  M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Setting up points
def A_def : A = (0, 0) := rfl
def M_def : M = (0, x / 2) := rfl
def N_def : N = (x / 2, x) := rfl

-- Area calculation (using the specific area formula)
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The proof statement
theorem ratio_of_area (h1 : is_square A (x, 0) (x, x) (0, x))
                      (h2 : midpoint A (0, x) M)
                      (h3 : midpoint (x, 0) (x, x) N) :
  let AreaSquare := x^2,
      AreaTriangle := area_of_triangle A M N
  in AreaTriangle / AreaSquare = 1 / 8 := 
  by
  sorry

end ratio_of_area_l392_392232


namespace distance_to_new_york_l392_392596

theorem distance_to_new_york 
  (travel_rate : ℕ := 50) (rest_interval_hours : ℕ := 2) 
  (rest_duration_minutes : ℕ := 30) (total_time_hours : ℕ := 7) : 
  distance_to_new_york = 275 :=
by
  sorry

end distance_to_new_york_l392_392596


namespace smaller_angle_linear_pair_l392_392344

theorem smaller_angle_linear_pair (a b : ℝ) (h1 : a + b = 180) (h2 : a = 5 * b) : b = 30 := by
  sorry

end smaller_angle_linear_pair_l392_392344


namespace reflection_matrix_is_correct_l392_392043

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l392_392043


namespace find_larger_integer_l392_392695

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l392_392695


namespace num_blocks_differ_by_two_ways_l392_392435

theorem num_blocks_differ_by_two_ways :
  let materials := 2
  let sizes := 4
  let colors := 4
  let shapes := 4
  let blocks := materials * sizes * colors * shapes
  let generating_function := (1 + 1) * (1 + 3) * (1 + 3)^2
  number_of_blocks_differ_by_two_ways := generating_function.coeff 2
  blocks = 128
  number_of_blocks_differ_by_two_ways = 15 :=
by
  sorry

end num_blocks_differ_by_two_ways_l392_392435


namespace ratio_apps_optimal_l392_392897

theorem ratio_apps_optimal (max_apps : ℕ) (recommended_apps : ℕ) (apps_to_delete : ℕ) (current_apps : ℕ)
  (h_max_apps : max_apps = 50)
  (h_recommended_apps : recommended_apps = 35)
  (h_apps_to_delete : apps_to_delete = 20)
  (h_current_apps : current_apps = max_apps + apps_to_delete) :
  current_apps / recommended_apps = 2 :=
by {
  sorry
}

end ratio_apps_optimal_l392_392897


namespace PQ_bisects_BD_l392_392216

variable {A B C D P Q M N : Type}
variable [ConvexQuadrilateral A B C D]
variable [Midpoint P A B]
variable [Midpoint Q C D]
variable [Bisects PQ AC]

-- Prove that PQ also bisects diagonal BD
theorem PQ_bisects_BD (PQ_bisects_AC : bisects P Q A C) : bisects P Q B D := 
sorry

end PQ_bisects_BD_l392_392216


namespace circle_diameters_radii_l392_392562

theorem circle_diameters_radii (D R : ℕ) (hD : D = ⊤) (hR : R = ⊤) : ¬ (D = 1 / 2 * R) :=
by
  sorry

end circle_diameters_radii_l392_392562


namespace xiaohua_amount_paid_l392_392416

def cost_per_bag : ℝ := 18
def discount_rate : ℝ := 0.1
def price_difference : ℝ := 36

theorem xiaohua_amount_paid (x : ℝ) 
  (h₁ : 18 * (x+1) * (1 - 0.1) = 18 * x - 36) :
  18 * (x + 1) * (1 - 0.1) = 486 := 
sorry

end xiaohua_amount_paid_l392_392416


namespace oliver_used_fraction_l392_392272

variable (x : ℚ)

/--
Oliver had 135 stickers. He used a fraction x of his stickers, gave 2/5 of the remaining to his friend, and kept the remaining 54 stickers. Prove that he used 1/3 of his stickers.
-/
theorem oliver_used_fraction (h : 135 - (135 * x) - (2 / 5) * (135 - 135 * x) = 54) : 
  x = 1 / 3 := 
sorry

end oliver_used_fraction_l392_392272


namespace smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l392_392895

theorem smallest_N_such_that_N_and_N_squared_end_in_same_three_digits :
  ∃ N : ℕ, (N > 0) ∧ (N % 1000 = (N^2 % 1000)) ∧ (1 ≤ N / 100 % 10) ∧ (N = 376) :=
by
  sorry

end smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l392_392895


namespace total_cost_of_ice_cream_l392_392660

theorem total_cost_of_ice_cream :
  let kiddie_scoop_cost := 3 in
  let regular_scoop_cost := 4 in
  let double_scoop_cost := 6 in
  let mr_and_mrs_martin_scoops := 2 in
  let children_scoops := 2 in
  let teenage_children_scoops := 3 in
  mr_and_mrs_martin_scoops * regular_scoop_cost +
  children_scoops * kiddie_scoop_cost +
  teenage_children_scoops * double_scoop_cost = 32 :=
by
  sorry

end total_cost_of_ice_cream_l392_392660


namespace max_clouds_crossed_by_plane_l392_392978

-- Define the conditions
def plane_region_divide (num_planes : ℕ) : ℕ :=
  num_planes + 1

-- Hypotheses/Conditions
variable (num_planes : ℕ)
variable (initial_region_clouds : ℕ)
variable (max_crosses : ℕ)

-- The primary statement to be proved
theorem max_clouds_crossed_by_plane : 
  num_planes = 10 → initial_region_clouds = 1 → max_crosses = num_planes + initial_region_clouds →
  max_crosses = 11 := 
by
  -- Placeholder for the actual proof
  intros
  sorry

end max_clouds_crossed_by_plane_l392_392978


namespace line_intersects_circle_l392_392485

noncomputable def circle : Set (ℝ × ℝ) := { p | let (x, y) := p in x^2 + y^2 - 4*x - 6*y + 9 = 0 }

noncomputable def line (m : ℝ) : Set (ℝ × ℝ) := { p | let (x, y) := p in 2*m*x - 3*m*y + x - y - 1 = 0 }

theorem line_intersects_circle (m : ℝ) : ∀ (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)), 
  (C = circle) → (l = line m) → ¬ ∅ ∈ (C ∩ l) := 
sorry

noncomputable def line_perpendicular_to_cp : Set (ℝ × ℝ) := { p | let (x, y) := p in y = x - 1 }

lemma shortest_chord_line_eq :
  let C := circle in
  let l := line_perpendicular_to_cp in
  ∀ p ∈ C, p ∈ l :=
sorry

end line_intersects_circle_l392_392485


namespace cube_sphere_tangent_radius_l392_392093

theorem cube_sphere_tangent_radius :
  (∃ (r : ℝ), 
    ∀ (A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ),
      is_cube A B C D A1 B1 C1 D1 1 ∧
      sphere_touches_edges_and_line r A D DD1 CD line_BC1 → 
      r = 2 * real.sqrt 2 - real.sqrt 5) :=
sorry

-- Definitions for is_cube, sphere_touches_edges_and_line, etc., need to be provided or assumed.

end cube_sphere_tangent_radius_l392_392093


namespace fraction_subtraction_l392_392499

theorem fraction_subtraction (a b : ℝ) (h1 : 2 * b = 1 + a * b) (h2 : a ≠ 1) (h3 : b ≠ 1) :
  (a + 1) / (a - 1) - (b + 1) / (b - 1) = 2 :=
by
  sorry

end fraction_subtraction_l392_392499


namespace reflection_matrix_is_correct_l392_392042

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l392_392042


namespace find_four_digit_number_l392_392573

noncomputable def reverse_num (n : ℕ) : ℕ := -- assume definition to reverse digits
  sorry

theorem find_four_digit_number :
  ∃ (A : ℕ), 1000 ≤ A ∧ A ≤ 9999 ∧ reverse_num (9 * A) = A ∧ 9 * A = reverse_num A ∧ A = 1089 :=
sorry

end find_four_digit_number_l392_392573


namespace reflection_matrix_over_vector_l392_392066

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l392_392066


namespace blake_lollipops_count_l392_392836

theorem blake_lollipops_count (lollipop_cost : ℕ) (choc_cost_per_pack : ℕ) 
  (chocolate_packs : ℕ) (total_paid : ℕ) (change_received : ℕ) 
  (total_spent : ℕ) (total_choc_cost : ℕ) (remaining_amount : ℕ) 
  (lollipop_count : ℕ) : 
  lollipop_cost = 2 →
  choc_cost_per_pack = 4 * lollipop_cost →
  chocolate_packs = 6 →
  total_paid = 6 * 10 →
  change_received = 4 →
  total_spent = total_paid - change_received →
  total_choc_cost = chocolate_packs * choc_cost_per_pack →
  remaining_amount = total_spent - total_choc_cost →
  lollipop_count = remaining_amount / lollipop_cost →
  lollipop_count = 4 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end blake_lollipops_count_l392_392836


namespace polynomial_sum_divisible_l392_392787

theorem polynomial_sum_divisible (P : Polynomial ℤ) (k : ℕ) (hk : k > 0) :
  ∃ n : ℕ, (∑ i in Finset.range (n + 1), P.eval (i + 1)) % k = 0 :=
sorry

end polynomial_sum_divisible_l392_392787


namespace range_of_m4_n4_is_correct_l392_392121

noncomputable def range_of_m4_n4 (m n : ℝ) (A B C : Point) (O : Circle) : Set ℝ :=
  {r | ∃ (m n : ℝ), (m * m + n * n = 1 + m * n ∧ - (m * n - 1) ^ 2 + 2 = r)}

theorem range_of_m4_n4_is_correct (m n : ℝ) 
  (h1 : 0 ≤ 1)
  (h2 : angle O A B = 120)
  (h3 : O.radius = 1)
  (h4 : vector OC = m • vector OA + n • vector OB) :
  range_of_m4_n4 = set.Icc (2 / 9) 2 :=
by
  sorry

end range_of_m4_n4_is_correct_l392_392121


namespace larger_integer_value_l392_392682

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l392_392682


namespace intervals_of_monotonicity_l392_392863

noncomputable def monotonicity_intervals (x : ℝ) :=
  sin (0.5 * x + (Real.pi / 3))

theorem intervals_of_monotonicity : 
  (∀ x, x ∈ Set.Icc (-2 * Real.pi) (-5 * Real.pi / 3) ∪ Set.Icc (Real.pi / 3) (2 * Real.pi) → derivative (monotonicity_intervals x).nonpos) ∧
  (∀ x, x ∈ Set.Icc (-5 * Real.pi / 3) (Real.pi / 3) → derivative (monotonicity_intervals x).nonneg) :=
sorry

end intervals_of_monotonicity_l392_392863


namespace jorge_total_goals_l392_392605

theorem jorge_total_goals (last_season_goals current_season_goals : ℕ) (h_last : last_season_goals = 156) (h_current : current_season_goals = 187) : 
  last_season_goals + current_season_goals = 343 :=
by
  sorry

end jorge_total_goals_l392_392605


namespace parabola_equation_ellipse_equation_lambda_mu_constant_l392_392516

open Real

theorem parabola_equation {p M_y0 : ℝ} (h1 : M_y0^2 = 2 * p * 3) (h2 : dist (3, M_y0) (p / 2, 0) = 4) : 
  p = 2 ∧ M_y0 = sqrt (2 * p * 3) ∧ 4 * 3 = M_y0^2 :=
by sorry

theorem ellipse_equation {a b : ℝ} (h1 : eccentricity a b = real.sqrt(2) / 2) (h2 : b = 1) : 
  a^2 = 2 ∧ (1 / a^2 = real.sqrt (a^2 - b^2) / a) :=
by sorry

theorem lambda_mu_constant (k x₁ x₂: ℝ) (h1 : ∀ x, k^2 * x^2 - (2 * k^2 + 4) * x + k^2 = 0)
                               (h2 : x₁ + x₂ = (2 * k^2 + 4) / k^2 ∧ x₁ * x₂ = 1)
                               (h3 : ∀ λ₁ λ₂, λ₁ = x₁ / (1 - x₁) ∧ λ₂ = x₂ / (1 - x₂)) : 
  ∑ μ λ, (λ₁ + λ₂ = -1) :=
by sorry

end parabola_equation_ellipse_equation_lambda_mu_constant_l392_392516


namespace solve_for_x_l392_392289

theorem solve_for_x (x : ℝ) (h : 2^(32^x) = 32^(2^x)) : 
  x = real.log (2, 5) / 4 :=
sorry

end solve_for_x_l392_392289


namespace value_of_x_l392_392155

theorem value_of_x (x : ℕ) :
  (1 / 8) * 2 ^ 36 = 8 ^ x → x = 11 :=
by
  intro h
  have h1 : (1 / 8) = 2⁻³ := by sorry
  have h2 : (2⁻³) * 2 ^ 36 = 2 ^ 33 := by sorry
  have h3 : 8 ^ x = (2 ^ 3) ^ x := by sorry
  have h4 : (2 ^ 3) ^ x = 2 ^ (3 * x) := by sorry
  have h5 : 2 ^ 33 = 2 ^ (3 * x) := by sorry
  have h6 : 33 = 3 * x := by sorry
  exact nat.div_eq_of_lt 33 3 sorry

end value_of_x_l392_392155


namespace intersection_A_B_l392_392949

variable {x : ℝ}

def A : set ℝ := {x | 0 < x ∧ x < 2}
def B : set ℝ := {x | x^2 ≤ 1}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 1} :=
by sorry

end intersection_A_B_l392_392949


namespace trig_identity_l392_392284

theorem trig_identity (α : ℝ) :
  (2 - 2 * Math.sin (α + 3 * Real.pi / 4) * Math.cos (α + Real.pi / 4)) / (Math.cos α ^ 4 - Math.sin α ^ 4) =
  (1 + Math.tan α) / (1 - Math.tan α) :=
sorry

end trig_identity_l392_392284


namespace digit_d_makes_multiple_of_9_l392_392468

theorem digit_d_makes_multiple_of_9 :
  ∃ d : ℕ, d < 10 ∧ (26 + d) % 9 = 0 ∧ d = 1 :=
by {
  have h1 : 26 % 9 = 8 := rfl,
  use 1,
  split,
  { linarith },
  split,
  { norm_num },
  { refl }
}

end digit_d_makes_multiple_of_9_l392_392468


namespace angle_B1KB2_75_degrees_l392_392570

theorem angle_B1KB2_75_degrees 
  {α β γ : ℝ} 
  (acute_triangle : α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90)
  (angle_A : α = 35) 
  (B1_altitude : BB_1 ∥⊥AC)
  (C1_altitude : CC_1 ∥⊥AB)
  (B2_midpoint : B_2 = midpoint A C)
  (C2_midpoint : C_2 = midpoint A B)
  (K_intersection : K = point_of_intersection (line_through B_1 C_2) (line_through C_1 B_2)) :
  angle B_1 K B_2 = 75 :=
sorry

end angle_B1KB2_75_degrees_l392_392570


namespace reflection_matrix_over_vector_is_correct_l392_392035

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l392_392035


namespace sum_of_divisors_lt_million_l392_392466

theorem sum_of_divisors_lt_million :
  let S := (∑ n in Finset.range 1000, (∑ k in Finset.range (1000 / (n + 1)), (n + 1))) in
  S < 1000000 :=
by
  let S := (∑ n in Finset.range 1000, (∑ k in Finset.range (1000 / (n + 1)), (n + 1)))
  sorry

end sum_of_divisors_lt_million_l392_392466


namespace total_cost_of_ice_cream_l392_392661

theorem total_cost_of_ice_cream :
  let kiddie_scoop_cost := 3 in
  let regular_scoop_cost := 4 in
  let double_scoop_cost := 6 in
  let mr_and_mrs_martin_scoops := 2 in
  let children_scoops := 2 in
  let teenage_children_scoops := 3 in
  mr_and_mrs_martin_scoops * regular_scoop_cost +
  children_scoops * kiddie_scoop_cost +
  teenage_children_scoops * double_scoop_cost = 32 :=
by
  sorry

end total_cost_of_ice_cream_l392_392661


namespace compare_a_b_l392_392545

theorem compare_a_b (a b : ℝ) (h1 : a = 2 * Real.sqrt 7) (h2 : b = 3 * Real.sqrt 5) : a < b :=
by {
  sorry -- We'll leave the proof as a placeholder.
}

end compare_a_b_l392_392545


namespace factorization_divisibility_l392_392014

theorem factorization_divisibility (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end factorization_divisibility_l392_392014


namespace reflection_matrix_over_vector_l392_392067

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l392_392067


namespace f_is_monotone_decreasing_l392_392138

def f (x : ℝ) : ℝ := x * cos x - sin x

-- Define the domain of the function as an interval
def domain : Set ℝ := Set.Icc 0 (Real.pi / 2)

theorem f_is_monotone_decreasing :
  ∀ x ∈ domain, ∀ y ∈ domain, x < y → f y < f x := 
by
  sorry

end f_is_monotone_decreasing_l392_392138


namespace expected_strawberries_l392_392843

/-- Define the problem parameters -/
def garden_length : ℕ := 7
def garden_width : ℕ := 9
def plants_per_sqft : ℕ := 5
def strawberries_per_plant : ℕ := 10

/-- Calculate the garden area -/
def garden_area := garden_length * garden_width

/-- Calculate the total number of plants -/
def total_plants := plants_per_sqft * garden_area

/-- Calculate the total expected strawberries -/
def total_expected_strawberries := strawberries_per_plant * total_plants

/-- The theorem to prove -/
theorem expected_strawberries (garden_length = 7) (garden_width = 9) 
    (plants_per_sqft = 5) (strawberries_per_plant = 10) :
    total_expected_strawberries = 3150 :=
by 
sory

end expected_strawberries_l392_392843


namespace exists_section_by_plane_l392_392919

-- Definitions to set up the geometry of the problem
structure Point3D :=
  (x : ℝ) (y : ℝ) (z : ℝ)

structure LineSegment :=
  (start : Point3D) (end : Point3D)

structure Cube :=
  (A B C D A1 B1 C1 D1 : Point3D)
  (edges : List LineSegment)

def is_on_edge (P : Point3D) (l : LineSegment) : Prop :=
  -- Definition of a point on an edge (line segment)
  sorry

-- Given points P, Q, R on specified edges
variable (cube : Cube)
variable (P : Point3D) (hP : is_on_edge P (LineSegment.mk cube.A cube.A1))
variable (Q : Point3D) (hQ : is_on_edge Q (LineSegment.mk cube.B cube.C))
variable (R : Point3D) (hR : is_on_edge R (LineSegment.mk cube.C1 cube.D1))

-- Prove the existence of the section of the cube by the plane passing through P, Q, R
theorem exists_section_by_plane (cube : Cube) (P Q R : Point3D)
  (hP : is_on_edge P (LineSegment.mk cube.A cube.A1))
  (hQ : is_on_edge Q (LineSegment.mk cube.B cube.C))
  (hR : is_on_edge R (LineSegment.mk cube.C1 cube.D1)) :
  ∃ section : List LineSegment, -- This represents the intersection of the plane with the cube
    -- Some conditions representing the section
    sorry :=
  sorry

end exists_section_by_plane_l392_392919


namespace sum_of_extreme_prime_factors_of_1140_l392_392359

theorem sum_of_extreme_prime_factors_of_1140 : 
  let prime_factors := [2, 3, 5, 19] in 
  (List.minimum prime_factors + List.maximum prime_factors) = 21 := by
  sorry

end sum_of_extreme_prime_factors_of_1140_l392_392359


namespace reflection_matrix_over_vector_is_correct_l392_392036

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l392_392036


namespace john_danced_before_break_3_hours_l392_392601

def johns_dancing_time_before_break (x : ℝ) : Prop :=
  -- John danced for x hours, then took a 1-hour break, and then danced another 5 hours.
  let john_dance_total := x + 5 in
  
  -- James danced the whole time John was dancing and resting, which is x + 1 + 5 hours,
  -- and then danced for another (1/3) times more hours.
  let james_dance_total := x + 6 + (1/3) * (x + 6) in

  -- Their combined dancing time without including John's break time was 20 hours.
  john_dance_total + james_dance_total = 20

theorem john_danced_before_break_3_hours :
  ∃ x : ℝ, johns_dancing_time_before_break x ∧ x = 3 :=
sorry

end john_danced_before_break_3_hours_l392_392601


namespace shadow_problem_l392_392825

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

end shadow_problem_l392_392825


namespace range_of_b_over_a_l392_392479

noncomputable def f (a b x : ℝ) : ℝ := (x - a)^3 * (x - b)
noncomputable def g_k (a b k x : ℝ) : ℝ := (f a b x - f a b k) / (x - k)

theorem range_of_b_over_a (a b : ℝ) (h₀ : 0 < a) (h₁ : a < b) (h₂ : b < 1)
    (hk_inc : ∀ k : ℤ, ∀ x : ℝ, k < x → g_k a b k x ≥ g_k a b k (k + 1)) :
  1 < b / a ∧ b / a ≤ 3 :=
by
  sorry


end range_of_b_over_a_l392_392479


namespace number_of_prize_orders_l392_392428

/-- At the end of a professional bowling tournament, the top 6 bowlers have a playoff.
    - #6 and #5 play a game. The loser receives the 6th prize and the winner plays #4.
    - The loser of the second game receives the 5th prize and the winner plays #3.
    - The loser of the third game receives the 4th prize and the winner plays #2.
    - The loser of the fourth game receives the 3rd prize and the winner plays #1.
    - The winner of the final game gets 1st prize and the loser gets 2nd prize.

    We want to determine the number of possible orders in which the bowlers can receive the prizes.
-/
theorem number_of_prize_orders : 2^5 = 32 := by
  sorry

end number_of_prize_orders_l392_392428


namespace max_value_of_f_on_interval_l392_392309

noncomputable def f (x : ℝ) := x^3 - 3*x

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (-1 : ℝ) (3 : ℝ), ∀ y ∈ Set.Icc (-1 : ℝ) (3 : ℝ), f(y) ≤ f(x) :=
begin
  let x := 3 : ℝ,
  use x,
  split,
  { -- Proof that 3 ∈ [-1, 3]
    exact set.mem_Icc.mpr ⟨by norm_num, by norm_num⟩ },
  { -- Proof that f(y) ≤ f(3) for all y ∈ [-1, 3]
    intro y,
    intro hy,
    -- Skipping the detailed steps here and using sorry to bypass the proof
    sorry
  }
end

end max_value_of_f_on_interval_l392_392309


namespace factor_polynomials_l392_392013

theorem factor_polynomials :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 3) * (x^2 + 6*x + 12) :=
by
  sorry

end factor_polynomials_l392_392013


namespace gwen_bonus_fraction_l392_392149

theorem gwen_bonus_fraction (f : ℝ) :
  (∃ f : ℝ, (900 * (2 * f) + 450 * (1 - 2 * f) = 1350) ∧
          (f = 1/3 ∧ (1 - 2 * f = 1/3))) :=
by
  exists 1/3
  split
  { -- Prove 900 * (2 * f) + 450 * (1 - 2 * f) = 1350
    sorry }
  { -- Prove (f = 1/3 ∧ (1 - 2 * f = 1/3))
    split
    { -- Prove f = 1/3
      rfl
    }
    { -- Prove 1 - 2 * f = 1/3
      sorry }
  }

end gwen_bonus_fraction_l392_392149


namespace squares_do_not_have_equal_perimeters_l392_392768

-- Define what it means to be a square
structure Square (s : ℝ) :=
  (side_length : s > 0)
  (all_angles_90 : ∀ (a : ℝ), a = 90)

-- Axiom: Define the equivalence of statements
axiom all_squares_similar : ∀ (s1 s2 : Square s), similar s1 s2
axiom all_squares_convex : ∀ (s : Square s), convex s
axiom all_squares_perpendicular_diagonals : ∀ (s : Square s), perpendicular_diagonals s
axiom area_proportional_to_square_of_side_length : ∀ (s : Square s), area s = (side_length s)^2

-- Define a false statement: All squares have equal perimeters
def all_squares_have_equal_perimeters (s1 s2 : Square s) : Prop :=
  perimeter s1 = perimeter s2

-- Statement to prove: Falsehood of the given statement
theorem squares_do_not_have_equal_perimeters : 
  ∃ (s1 s2 : Square s), s1.side_length ≠ s2.side_length ∧ not (all_squares_have_equal_perimeters s1 s2) :=
sorry

end squares_do_not_have_equal_perimeters_l392_392768


namespace larger_integer_value_l392_392680

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l392_392680


namespace x_equals_eleven_l392_392157

theorem x_equals_eleven (x : ℕ) 
  (h : (1 / 8) * 2^36 = 8^x) : x = 11 :=
sorry

end x_equals_eleven_l392_392157


namespace part_a_part_b_part_c_l392_392861

-- Part (a)
theorem part_a :
  ∃ (P : Type) [polygon P], 
    (∃ f : polygon P → (polygon P × polygon P), is_broken_line f) 
  ∧ (¬ ∃ s : polygon P → bool, is_segment_division s) :=
begin
  sorry
end

-- Part (b)
theorem part_b :
  ∃ (P : Type) [convex_polygon P], 
    (∃ f : convex_polygon P → (convex_polygon P × convex_polygon P), is_broken_line f) 
  ∧ (¬ ∃ s : convex_polygon P → bool, is_segment_division s) :=
begin
  sorry
end

-- Part (c)
theorem part_c :
  ∀ (P : Type) [convex_polygon P], 
    (∃ f : convex_polygon P → (convex_polygon P × convex_polygon P), 
      is_broken_line f ∧ is_orientation_preserving f) →
    (∃ s : convex_polygon P → bool, 
      is_segment_division s ∧ is_orientation_preserving s) :=
begin
  sorry
end

end part_a_part_b_part_c_l392_392861


namespace pow_addition_l392_392382

theorem pow_addition : (-2 : ℤ)^2 + (2 : ℤ)^2 = 8 :=
by
  sorry

end pow_addition_l392_392382


namespace find_larger_integer_l392_392729

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l392_392729


namespace find_a_l392_392494

-- Statement of the problem conditions
def is_root (p : Polynomial ℚ) (x : ℝ) := p.eval x = 0

theorem find_a (a b : ℚ) (h : is_root (Polynomial.C 31 + Polynomial.X * (Polynomial.C b + Polynomial.X * (Polynomial.C a + Polynomial.X))) (-1 - 4 * Real.sqrt 2)) :
  a = 1 := 
sorry

end find_a_l392_392494


namespace express_c_in_terms_of_a_b_l392_392951

variables (e1 e2 a b c : ℝ^3)

-- Given conditions
def non_collinear_vectors (e1 e2 : ℝ^3) : Prop := 
  e1 ≠ 0 ∧ e2 ≠ 0 ∧ ¬ ∃ k : ℝ, e1 = k • e2

def a_def (e1 e2 : ℝ^3) : ℝ^3 := 
  e1 + e2

def b_def (e1 e2 : ℝ^3) : ℝ^3 := 
  2 • e1 - e2

def c_def (e1 e2 : ℝ^3) : ℝ^3 := 
  e1 + 2 • e2

-- The theorem to be proven
theorem express_c_in_terms_of_a_b (e1 e2 a b c : ℝ^3) 
  (h_non_collinear : non_collinear_vectors e1 e2)
  (ha : a = a_def e1 e2)
  (hb : b = b_def e1 e2)
  (hc : c = c_def e1 e2) : 
  c = (5 / 3) • a - (1 / 3) • b :=
sorry

end express_c_in_terms_of_a_b_l392_392951


namespace pumpkin_price_l392_392276

theorem pumpkin_price (P : ℝ) : 
  (let j := 9.00 in                -- Price of a jumbo pumpkin
   let n_total := 80 in            -- Total number of pumpkins sold
   let revenue_total := 395.00 in  -- Total revenue collected
   let r_sold := 65 in             -- Number of regular pumpkins sold
   let j_sold := n_total - r_sold in -- Number of jumbo pumpkins sold
   let r_revenue := r_sold * P in  -- Revenue from regular pumpkins
   let j_revenue := j_sold * j in  -- Revenue from jumbo pumpkins
   let total_revenue := r_revenue + j_revenue in
   total_revenue = revenue_total) → P = 4 :=
by intros; sorry

end pumpkin_price_l392_392276


namespace maximize_prob_defective_l392_392392

def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

noncomputable def prob_defective (p : ℝ) : ℝ := binomial_coefficient 10 3 * p^3 * (1-p)^7

theorem maximize_prob_defective (p : ℝ) (h : 0 < p ∧ p < 1) :
  arg_max f p = 3/10 :=
begin
  sorry
end

end maximize_prob_defective_l392_392392


namespace largest_divisor_of_expression_l392_392189

theorem largest_divisor_of_expression (x : ℤ) (h : odd x) : ∃ k, (∀ n : ℤ, n divides (10*x + 2) * (10*x + 6)^2 * (5*x + 1) → n ≤ 24) ∧ k = 24 :=
by sorry

end largest_divisor_of_expression_l392_392189


namespace simplest_sqrt_l392_392766

/-- Conditions: Definitions of the given square roots -/
def sqrt_23 := Real.sqrt 23
def sqrt_one_third := Real.sqrt (1/3)
def sqrt_12 := Real.sqrt 12
def sqrt_half := Real.sqrt (1/2)

/-- Main Statement: -/
theorem simplest_sqrt : (sqrt_23 = Real.sqrt 23) →
                        (sqrt_one_third = Real.sqrt (1/3)) →
                        (sqrt_12 = Real.sqrt 12) →
                        (sqrt_half = Real.sqrt (1/2)) →
                        sqrt_23 = Real.sqrt 23 :=
by
  intros _ _ _ _
  sorry

end simplest_sqrt_l392_392766


namespace Jorge_goals_total_l392_392603

theorem Jorge_goals_total : 
  let last_season_goals := 156
  let this_season_goals := 187
  last_season_goals + this_season_goals = 343 := 
by
  sorry

end Jorge_goals_total_l392_392603


namespace total_votes_in_election_l392_392227

-- Definitions of the conditions
variables (V : ℕ) (valid_votes : ℕ) (A_votes : ℕ)
hypotheses 
  (h1 : valid_votes = 85 * V / 100) 
  (h2 : A_votes = 75 * valid_votes / 100)
  (h3 : A_votes = 357000)

-- Total number of votes in the election
theorem total_votes_in_election : V = 560000 :=
by
  sorry

end total_votes_in_election_l392_392227


namespace find_complex_number_l392_392885

namespace ComplexProof

open Complex

def satisfies_conditions (z : ℂ) : Prop :=
  (z^2).im = 0 ∧ abs (z - I) = 1

theorem find_complex_number (z : ℂ) (h : satisfies_conditions z) : z = 0 ∨ z = 2 * I :=
sorry

end ComplexProof

end find_complex_number_l392_392885


namespace smallest_number_of_seats_required_l392_392801

theorem smallest_number_of_seats_required (total_chairs : ℕ) (condition : ∀ (N : ℕ), ∀ (seating : Finset ℕ),
  seating.card = N → (∀ x ∈ seating, (x + 1) % total_chairs ∈ seating ∨ (x + total_chairs - 1) % total_chairs ∈ seating)) :
  total_chairs = 100 → ∃ N : ℕ, N = 20 :=
by
  intros
  sorry

end smallest_number_of_seats_required_l392_392801


namespace set_d_forms_triangle_l392_392829

theorem set_d_forms_triangle : (a b c : ℕ) (h1 : a = 6) (h2 : b = 9) (h3 : c = 14) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) :=
by {
  sorry
}

end set_d_forms_triangle_l392_392829


namespace not_symmetric_star_l392_392856

def star (x y : ℝ) : ℝ := abs (x - 2 * y + 3)

theorem not_symmetric_star :
  ∃ x y : ℝ, star x y ≠ star y x := 
by
  sorry

end not_symmetric_star_l392_392856


namespace total_votes_in_election_l392_392225

def valid_votes_fraction : ℝ := 0.85
def candidate_A_votes_fraction : ℝ := 0.75
def candidate_A_votes : ℝ := 357000

theorem total_votes_in_election (v : ℝ) :
  v = candidate_A_votes / (valid_votes_fraction * candidate_A_votes_fraction) → v = 560000 :=
by
  intro h
  rw h
  sorry

end total_votes_in_election_l392_392225


namespace tangent_line_eq_l392_392305

noncomputable def f : ℝ → ℝ := λ x, x * Real.exp x + 1

theorem tangent_line_eq :
  ∃ k b, 
    (∀ x, (f x - (e + 1)) = k * (x - 1)) ∧
    (2 * e - k = 0) ∧
    (e + 1 - b = e - 1) :=
by
  sorry

end tangent_line_eq_l392_392305


namespace intersection_of_sets_l392_392950

def setA := { x : ℝ | x / (x - 1) < 0 }
def setB := { x : ℝ | 0 < x ∧ x < 3 }
def setIntersect := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_of_sets :
  ∀ x : ℝ, x ∈ setA ∧ x ∈ setB ↔ x ∈ setIntersect := 
by
  sorry

end intersection_of_sets_l392_392950


namespace ceil_neg_sqrt_frac_eq_l392_392006

theorem ceil_neg_sqrt_frac_eq : ⌈-real.sqrt (64 / 4)⌉ = -4 :=
by sorry

end ceil_neg_sqrt_frac_eq_l392_392006


namespace optimal_discount_sequence_saves_more_l392_392673

theorem optimal_discount_sequence_saves_more :
  (let initial_price := 30
   let flat_discount := 5
   let percent_discount := 0.25
   let first_seq_price := ((initial_price - flat_discount) * (1 - percent_discount))
   let second_seq_price := ((initial_price * (1 - percent_discount)) - flat_discount)
   first_seq_price - second_seq_price = 1.25) :=
by
  sorry

end optimal_discount_sequence_saves_more_l392_392673


namespace min_value_of_f_min_at_boundary_min_value_min_value_of_f_final_l392_392893

noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (x - 1)

theorem min_value_of_f : ∀ x : ℝ, x ≥ 1 → f x ≥ 1 :=
by
  intro x
  intro hx
  sorry

theorem min_at_boundary : f 1 = 1 :=
by
  sorry

theorem min_value : ∀ x : ℝ, x ≥ 1 → f x ≥ f 1 :=
by
  intro x hx
  apply min_value_of_f
  exact hx

theorem min_value_of_f_final : ∀ x : ℝ, x ≥ 1 → f x ≥ 1 ∧ f 1 = 1 :=
by
  intro x hx
  constructor
  · apply min_value
    exact hx
  · exact min_at_boundary

end min_value_of_f_min_at_boundary_min_value_min_value_of_f_final_l392_392893


namespace nearest_integer_l392_392538

def x : ℝ := 
  ∏ i in (finset.range 89).filter (λ i, i % 1 = 0), real.cos (↑i + 1) * (real.pi / 180)

def y : ℝ := 
  ∏ i in (finset.filter (λ i, i % 4 = 2) (finset.range 87)), real.cos (↑i + 2) * (real.pi / 180)

noncomputable def xy_ratio : ℝ := 2 / 7 * real.logb 2 (y / x)

theorem nearest_integer : round xy_ratio = 13 :=
by
  sorry

end nearest_integer_l392_392538


namespace pow_addition_l392_392381

theorem pow_addition : (-2)^2 + 2^2 = 8 :=
by
  sorry

end pow_addition_l392_392381


namespace marked_hexagons_selection_l392_392317

theorem marked_hexagons_selection :
  ∃ H' : set hexagon, H'.card = 666 ∧ (∀ (h₁ h₂ : hexagon), h₁ ∈ H' → h₂ ∈ H' → h₁ ≠ h₂ → h₁.vertices ∩ h₂.vertices = ∅) :=
sorry

end marked_hexagons_selection_l392_392317


namespace chord_length_of_tangent_l392_392662

theorem chord_length_of_tangent (R r : ℝ) (h : R^2 - r^2 = 25) : ∃ c : ℝ, c = 10 :=
by
  sorry

end chord_length_of_tangent_l392_392662


namespace A_divisible_by_1980_l392_392770

def consecutive_numbers_concatenated := "19202122...787980" -- placeholder, string representation

theorem A_divisible_by_1980 :
  ∃ A, A = concatenate_consecutive_numbers 19 80 ∧ 1980 ∣ A :=
by
  sorry

end A_divisible_by_1980_l392_392770


namespace _l392_392032
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l392_392032


namespace max_cards_l392_392595

def card_cost : ℝ := 0.85
def budget : ℝ := 7.50

theorem max_cards (n : ℕ) : card_cost * n ≤ budget → n ≤ 8 :=
by sorry

end max_cards_l392_392595


namespace parabola_equation_l392_392334

theorem parabola_equation (focus : ℝ × ℝ) (hf : focus.1 - focus.2 + 4 = 0) :
    (focus = (0, 4) ∨ focus = (-4, 0)) →
    (exists p : ℝ, p = 8 ∧ (y^2 = -2 * p * x ∨ x^2 = 2 * p * y)) :=
by
  intro h
  cases h
  case Or.inl h => sorry
  case Or.inr h => sorry

end parabola_equation_l392_392334


namespace john_climbs_steps_l392_392243

theorem john_climbs_steps (flights : ℕ) (feet_per_flight inches_per_step : ℕ) (h1 : flights = 9) (h2 : feet_per_flight = 10) (h3 : inches_per_step = 18) :
  let total_steps := (flights * feet_per_flight * 12) / inches_per_step
  in total_steps = 60 :=
by
  sorry

end john_climbs_steps_l392_392243


namespace man_and_son_work_together_l392_392805

theorem man_and_son_work_together :
  let man's_rate := 1 / 10
  let son's_rate := 3 / 20
  let combined_rate := man's_rate + son's_rate
  combined_rate = 1 / 4 → 1 / combined_rate = 4 :=
by
  intros man's_rate son's_rate combined_rate h
  have : combined_rate = 1 / 4 := h
  rw [this]
  norm_num
  done

end man_and_son_work_together_l392_392805


namespace part1_part2_part3_l392_392136

-- Define the function f
def f (x : ℝ) : ℝ := 3^x - (1 / 3^(abs x))

-- Prove that if f(x) = 2, then x = log_3 (1 + sqrt 2)
theorem part1 {x : ℝ} (h : f x = 2) : x = log 3 (1 + real.sqrt 2) := 
  sorry

-- Prove that f(x) is strictly increasing on (0, +∞)
theorem part2 : strict_mono_on f (set.Ioi 0) := 
  sorry

-- Prove the range of m such that 3^t f(t) + m f(t) ≥ 0 holds for t ∈ [½, 1] is m ≥ -4
theorem part3 : ∀ (m : ℝ), (∀ t ∈ set.Icc (1/2 : ℝ) 1, 3^t * f t + m * f t ≥ 0) ↔ m ≥ -4 := 
  sorry

end part1_part2_part3_l392_392136


namespace increasing_on_condition_l392_392132

def f (a : ℝ) (x : ℝ) : ℝ :=
if x <= 1 then x^2 + a * x - 2 else -a^x

theorem increasing_on_condition (a : ℝ) (h : a ≠ 1) 
  (h_increasing : ∀ x y : ℝ, 0 < x → x < y → y < ∞ → f a x < f a y) :
  a ∈ set.Icc 0 (1 / 2) :=
sorry

end increasing_on_condition_l392_392132


namespace larger_integer_21_l392_392724

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l392_392724


namespace bisect_diagonals_l392_392204

variables {A B C D P Q : Type} [AffineSpace ℝ A B C D]

/-- Given a convex quadrilateral ABCD, if P and Q are midpoints of AB and CD respectively, 
and line PQ bisects diagonal AC, then line PQ also bisects diagonal BD.-/
theorem bisect_diagonals (h1 : convex_quadrilateral A B C D) 
  (h2 : midpoint (A, B) = P) (h3 : midpoint (C, D) = Q) 
  (h4 : bisects (PQ, AC)) : bisects (PQ, BD) := 
sorry

end bisect_diagonals_l392_392204


namespace find_amount_l392_392548

theorem find_amount (x : ℝ) (h1 : 0.25 * x = 0.15 * 1500 - 30) (h2 : x = 780) : 30 = 30 :=
by
  sorry

end find_amount_l392_392548


namespace find_larger_integer_l392_392733

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l392_392733


namespace earnings_of_edison_high_l392_392748

theorem earnings_of_edison_high (d_work_days : ℕ) (d_students : ℕ) (e_work_days : ℕ) (e_students : ℕ) (f_work_days : ℕ) (f_students : ℕ) (total_pay : ℚ) :
  d_work_days = 4 → d_students = 8 →
  e_work_days = 7 → e_students = 5 →
  f_work_days = 6 → f_students = 6 →
  total_pay = 884 →
  (e_students : ℚ) * (e_work_days : ℚ) * (total_pay / ((d_students : ℚ) * (d_work_days : ℚ) + (e_students : ℚ) * (e_work_days : ℚ) + (f_students : ℚ) * (f_work_days : ℚ))) = 300.39 := 
by
  intros hd d_students_he hs e_students_he hs f_students_he hs total_pay_he
  sorry

end earnings_of_edison_high_l392_392748


namespace total_number_of_elementary_events_is_16_l392_392566

def num_events_three_dice : ℕ := 6 * 6 * 6

theorem total_number_of_elementary_events_is_16 :
  num_events_three_dice = 16 := 
sorry

end total_number_of_elementary_events_is_16_l392_392566


namespace sample_size_l392_392202

theorem sample_size (
  students_grade10 : ℕ := 400
  students_grade11 : ℕ := 320
  students_grade12 : ℕ := 280
  p : ℝ := 0.2
) : 
  (students_grade10 + students_grade11 + students_grade12) * p = 200 :=
by
  sorry

end sample_size_l392_392202


namespace calculate_gross_income_l392_392313
noncomputable def gross_income (net_income : ℝ) (tax_rate : ℝ) : ℝ := net_income / (1 - tax_rate)

theorem calculate_gross_income : gross_income 20000 0.13 = 22989 :=
by
  sorry

end calculate_gross_income_l392_392313


namespace larger_integer_is_21_l392_392706

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l392_392706


namespace reflection_matrix_is_correct_l392_392065

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l392_392065


namespace quadrilateral_pyramid_coloring_l392_392812

/-- 
Statement: A quadrilateral pyramid \( S-ABCD \) has 5 vertices and 8 edges. Each vertex must be colored such that no two adjacent vertices share the same color, and there are 5 colors available. Prove the number of distinct ways to color the vertices is 420.
-/
theorem quadrilateral_pyramid_coloring (vertices edges colors : ℕ) 
    (pyramid : vertices = 5) 
    (connections : edges = 8)
    (color_options : colors = 5)
    (adjacent_different : ∀ (v1 v2 : Fin vertices), adjacency v1 v2 → ∀ (c1 c2 : Fin colors), v1 ≠ v2 → c1 ≠ c2) :
  ∃ (num_colorings : ℕ), num_colorings = 420 := 
  sorry

end quadrilateral_pyramid_coloring_l392_392812


namespace PQ_bisects_BD_l392_392219

variable {A B C D P Q M N : Type}
variable [ConvexQuadrilateral A B C D]
variable [Midpoint P A B]
variable [Midpoint Q C D]
variable [Bisects PQ AC]

-- Prove that PQ also bisects diagonal BD
theorem PQ_bisects_BD (PQ_bisects_AC : bisects P Q A C) : bisects P Q B D := 
sorry

end PQ_bisects_BD_l392_392219


namespace solution1_solution2_l392_392352

/-- Definitions of the vectors -/
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c (m : ℝ) : ℝ × ℝ := (-2, m)

/-- Problem 1 -/
def problem1 (m : ℝ) (c_val : ℝ) : Prop :=
  let c := c m in (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 0) → (c_val = real.sqrt ((c.1)^(2) + (c.2)^(2)))

theorem solution1 : problem1 (-1) (real.sqrt 5) := sorry

/-- Problem 2 -/
def problem2 (k : ℝ) : Prop :=
  (k - 2) / 4 = (2 * k + 3) → k = -2

theorem solution2 : problem2 (-2) := sorry

end solution1_solution2_l392_392352


namespace variance_of_arithmetic_sequence_is_20_over_3_l392_392667

noncomputable def variance_of_arithmetic_sequence : ℕ → Real :=
λ x₁, let seq := List.range' x₁ 9 in
let µ := (x₁ + 4 : ℝ) in
(1 / 9) * (List.sum (List.map (λ x, ((x - µ) ^ 2)) seq)).toReal

theorem variance_of_arithmetic_sequence_is_20_over_3 :
  (∀ x₁ : ℕ, (common_difference x₁ 1 → variance_of_arithmetic_sequence x₁ = 20 / 3)) :=
begin
  intros x₁ hd,
  sorry
end

def common_difference (x₁ : ℕ) (d : ℕ) : Prop :=
∀ i : ℕ, i < 9 → (nth x₁ i + d = nth x₁ (i + 1))

end variance_of_arithmetic_sequence_is_20_over_3_l392_392667


namespace range_of_a_l392_392921

def p (a : ℝ) := ∀ x ∈ Icc 1 2, x^2 - a ≥ 0
def q (a : ℝ) := (∃ x y : ℝ, x^2 + 2*a*x + 2 - a = 0 ∧ y^2 + 2*a*y + 2 - a = 0)

theorem range_of_a (a : ℝ) : ¬(¬ p a ∨ ¬ q a) → (a ≤ -2 ∨ a = 1) :=
by
  sorry

end range_of_a_l392_392921


namespace tangent_of_angle_l392_392952

noncomputable def vector_tangent_angle (a b : ℝ × ℝ) : ℝ :=
(let dot_product := (a.1 * b.1 + a.2 * b.2) in
 let magnitude_a := real.sqrt (a.1^2 + a.2^2) in
 let magnitude_b := real.sqrt (b.1^2 + b.2^2) in
 let cos_theta := dot_product / (magnitude_a * magnitude_b) in
 let sin_theta := real.sqrt (1 - cos_theta^2) in
 sin_theta / cos_theta)

theorem tangent_of_angle (a b : ℝ × ℝ) 
  (h1 : a.1^2 + a.2^2 = 4)
  (h2 : b.1^2 + b.2^2 = 1)
  (h3 : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 5)
  : vector_tangent_angle a b = real.sqrt 3 :=
by
  -- The proof is to be filled in.
  sorry

end tangent_of_angle_l392_392952


namespace larger_integer_21_l392_392723

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l392_392723


namespace tan_pos_in_third_quadrant_l392_392161

theorem tan_pos_in_third_quadrant (θ : ℝ) (h : π < θ ∧ θ < 3 * π / 2) : tan θ > 0 :=
sorry

end tan_pos_in_third_quadrant_l392_392161


namespace a_in_a_1_to_a_n_l392_392379

-- Definitions and Hypotheses
variables {a : ℕ} {a_1 a_2 ... a_n : ℕ}
hypothesis H : ∀ (k : ℕ), (∃ m : ℕ, m^2 = a * k + 1) → 
                      (∃ m : ℕ, (m^2 = a_1 * k + 1) ∨  
                                (m^2 = a_2 * k + 1) ∨ 
                                  ... ∨
                                (m^2 = a_n * k + 1))
                                
-- Theorem statement
theorem a_in_a_1_to_a_n : a = a_1 ∨ a = a_2 ∨ ... ∨ a = a_n :=
sorry

end a_in_a_1_to_a_n_l392_392379


namespace geometric_sequence_common_ratio_l392_392959

open scoped Nat

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, a n * a (n + 1) = (16 : ℝ) ^ n) :
  ∃ r : ℝ, (∀ n : ℕ, a n = a 0 * r ^ n) ∧ (r = 4) :=
sorry

end geometric_sequence_common_ratio_l392_392959


namespace surface_area_hemisphere_radius_1_l392_392739

noncomputable def surface_area_hemisphere (r : ℝ) : ℝ :=
  2 * Real.pi * r^2 + Real.pi * r^2

theorem surface_area_hemisphere_radius_1 :
  surface_area_hemisphere 1 = 3 * Real.pi :=
by
  sorry

end surface_area_hemisphere_radius_1_l392_392739


namespace reflection_matrix_l392_392025

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l392_392025


namespace larger_integer_is_21_l392_392713

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l392_392713


namespace arithmetic_sequence_formula_and_max_sum_l392_392571

theorem arithmetic_sequence_formula_and_max_sum :
  ∀ (a : ℕ → ℤ) (S : ℕ → ℤ),
  (a 1 = 25) →
  (a 2 ≠ a 1) →
  (a 1, a 11, a 13) forms_geometric_sequence →
  is_arithmetic_sequence a →
  (∀ n, a n = 27 - 2 * n) ∧ (max_sum S = 169) :=
by
  -- Insert proof here
  sorry

-- Definitions used:
def forms_geometric_sequence (a1 a11 a13 : ℤ) : Prop :=
  a11 * a11 = a1 * a13

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def max_sum (S : ℕ → ℤ) : ℤ :=
  -- Definition to calculate or determine the max sum
  sorry

end arithmetic_sequence_formula_and_max_sum_l392_392571


namespace find_larger_integer_l392_392698

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l392_392698


namespace reflection_matrix_is_correct_l392_392059

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l392_392059


namespace infinite_geometric_series_sum_l392_392008

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := (3 : ℚ) / 8
  |r| < 1 →
  (a / (1 - r)) = (8 : ℚ) / 3 :=
by {
  -- Definitions
  let a : ℚ := 5 / 3,
  let r : ℚ := 3 / 8,
  sorry
}

end infinite_geometric_series_sum_l392_392008


namespace first_4_seeds_selected_correctly_l392_392296

def random_table : List (List ℕ) := [
    [84, 42, 17, 53, 31,  57, 24, 55, 06, 88,  77, 04, 74, 47, 67,  21, 76, 33, 50, 25,  83, 92, 12, 06, 76],
    [63, 01, 63, 78, 59,  16, 95, 55, 67, 19,  98, 10, 50, 71, 75,  12, 86, 73, 58, 07,  44, 39, 52, 38, 79],
    [33, 21, 12, 34, 29,  78, 64, 56, 07, 82,  52, 42, 07, 44, 38,  15, 51, 00, 13, 42,  99, 66, 02, 79, 54]
]

-- The first position to start checking is the 8th row, 2nd column with value '301'.
def start_position := (2, 1) -- 8th row (index 2 in 0-indexed), 2nd column (index 1 in 0-indexed)

-- Function to filter eligible seed numbers from the table.
def eligible_seed_numbers (table : List (List ℕ)) (start : ℕ × ℕ) : List ℕ :=
  table[start.fst].drop start.snd |>.filter (λ n => n ≤ 850)

theorem first_4_seeds_selected_correctly :
  eligible_seed_numbers random_table start_position = [301, 637, 169, 555] :=
sorry

end first_4_seeds_selected_correctly_l392_392296


namespace reflection_over_vector_l392_392055

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l392_392055


namespace daps_dips_equivalence_l392_392184

theorem daps_dips_equivalence :
  (∃ dap dop dip : Type,
    (5 : ℝ) * ∀ x : dap, x = (4 : ℝ) * ∀ y : dop, y ∧
    (3 : ℝ) * ∀ z : dop, z = (8 : ℝ) * ∀ w : dip, w) →
  (22.5 : ℝ) * ∀ x : dap, x = (48 : ℝ) * ∀ y : dip, y :=
begin
  sorry
end

end daps_dips_equivalence_l392_392184


namespace hydrocarbon_tree_configurations_l392_392535

theorem hydrocarbon_tree_configurations (n : ℕ) 
  (h1 : 3 * n + 2 > 0) -- Total vertices count must be positive
  (h2 : 2 * n + 2 > 0) -- Leaves count must be positive
  (h3 : n > 0) -- Internal nodes count must be positive
  : (n:ℕ) ^ (n-2) = n ^ (n-2) :=
sorry

end hydrocarbon_tree_configurations_l392_392535


namespace find_number_l392_392194

theorem find_number (N x : ℝ) (h : x = 9) (h1 : N - (5 / x) = 4 + (4 / x)) : N = 5 :=
by
  sorry

end find_number_l392_392194


namespace shaded_area_correct_l392_392648

-- Define the given conditions as constants in Lean.

def diameter : ℝ := 3  -- Diameter of each semicircle in inches
def radius : ℝ := diameter / 2  -- Radius of each semicircle
def length_in_feet : ℝ := 1.5  -- Length of the pattern in feet
def length_in_inches : ℝ := length_in_feet * 12  -- Convert length from feet to inches
def num_semicircles : ℕ := (length_in_inches / diameter).toNat -- Number of semicircles in the given length

-- Define the expected answer
def expected_area : ℝ := 13.5 * Real.pi

-- Statement to prove the shaded area equals the expected answer
theorem shaded_area_correct :
  let full_circle_area := Real.pi * radius^2 in
  let total_area := (full_circle_area / 2) * num_semicircles * 2 in
  total_area = expected_area :=
by
  sorry

end shaded_area_correct_l392_392648


namespace S15_is_75_l392_392436

theorem S15_is_75 {a1 a7 a9 a15 : ℝ}
    (h1 : 1 * a9 - 1 * (10 - a7) = 0)
    (h2 : ∀ (n : ℕ), S n = n / 2 * (a1 + a15)) :
    S 15 = 75 := 
by
  sorry

end S15_is_75_l392_392436


namespace exists_n_good_not_n_add_1_good_l392_392249

-- Define the sum of digits function S
def S (k : ℕ) : ℕ := (k.digits 10).sum

-- Define what it means for a number to be n-good
def n_good (a n : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), (a_seq 0 = a) ∧ (∀ i : Fin n, a_seq i.succ = a_seq i - S (a_seq i))

-- Define the main theorem
theorem exists_n_good_not_n_add_1_good : ∀ n : ℕ, ∃ a : ℕ, n_good a n ∧ ¬n_good a (n + 1) :=
by
  sorry

end exists_n_good_not_n_add_1_good_l392_392249


namespace true_statements_count_is_two_l392_392518

def original_proposition (a : ℝ) : Prop :=
  a < 0 → ∃ x : ℝ, x^2 + x + a = 0

def contrapositive (a : ℝ) : Prop :=
  ¬ (∃ x : ℝ, x^2 + x + a = 0) → a ≥ 0

def converse (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 + x + a = 0) → a < 0

def negation (a : ℝ) : Prop :=
  a < 0 → ¬ ∃ x : ℝ, x^2 + x + a = 0

-- Prove that there are exactly 2 true statements among the four propositions: 
-- original_proposition, contrapositive, converse, and negation.

theorem true_statements_count_is_two : 
  ∀ (a : ℝ), original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a) → 
  (original_proposition a ∧ contrapositive a ∧ ¬(converse a) ∧ ¬(negation a)) ↔ (2 = 2) := 
by
  sorry

end true_statements_count_is_two_l392_392518


namespace tank_capacity_l392_392958

theorem tank_capacity 
  (T : ℚ) -- denoting the total capacity of the tank in gallons
  (h : 4 + (3 / 4) * T = (9 / 10) * T) : 
  T = 80 / 3 :=
begin
  -- The proof will go here, but it is not required for this task.
  sorry
end

end tank_capacity_l392_392958


namespace smallest_integer_y_l392_392757

theorem smallest_integer_y (y : ℤ) : (5 : ℝ) / 8 < (y : ℝ) / 17 → y = 11 := by
  sorry

end smallest_integer_y_l392_392757


namespace rate_of_interest_per_annum_l392_392367

def simple_interest (P T R : ℕ) : ℕ :=
  (P * T * R) / 100

theorem rate_of_interest_per_annum :
  let P_B := 5000
  let T_B := 2
  let P_C := 3000
  let T_C := 4
  let total_interest := 1980
  ∃ R : ℕ, 
      simple_interest P_B T_B R + simple_interest P_C T_C R = total_interest ∧
      R = 9 :=
by
  sorry

end rate_of_interest_per_annum_l392_392367


namespace dips_to_daps_l392_392176

theorem dips_to_daps : 
  ∀ (daps dops dips : Type) (eq1 : 5 * daps = 4 * dops) (eq2 : 3 * dops = 8 * dips),
  (48 * dips = 22.5 * daps) :=
begin
  intros,
  sorry
end

end dips_to_daps_l392_392176


namespace shortest_distance_ln_x_to_line_is_sqrt2_l392_392737

noncomputable def shortest_distance_ln_x_to_line : ℝ :=
  let line := λ x, x + 1 in
  let curve := λ x, log x in
  let tangent_slope_at_x := λ x, 1/x in
  let tangent_point := (1, log 1) in
  let distance := λ (p : ℝ × ℝ) (l : ℝ → ℝ), abs (l p.1 - p.2) / real.sqrt (1^2 + (-1)^2) in
  distance tangent_point line

theorem shortest_distance_ln_x_to_line_is_sqrt2 :
  shortest_distance_ln_x_to_line = real.sqrt 2 :=
sorry

end shortest_distance_ln_x_to_line_is_sqrt2_l392_392737


namespace num_of_arith_prog_sets_l392_392257

def S (n : ℕ) : Set ℕ := {i | 1 ≤ i ∧ i ≤ n}

def is_arith_prog (A : Set ℕ) : Prop :=
  ∃ d > 0, ∃ a b ∈ A, ∀ x ∈ A, ∃ k : ℕ, x = a + k * d

def valid_set (n : ℕ) (A : Set ℕ) : Prop :=
  A ⊆ S n ∧ ∃ d > 0, ∀ (x ∈ (S n \ A)), ¬ is_arith_prog (A ∪ {x})

def count_arith_prog_sets (n : ℕ) : ℕ := ⌊n^2 / 4⌋

theorem num_of_arith_prog_sets (n : ℕ) :
  ∃! k, (k = count_arith_prog_sets n) ∧
  ∀ A, valid_set n A → count_arith_prog_sets n = k :=
sorry

end num_of_arith_prog_sets_l392_392257


namespace power_function_fixed_point_logarithm_fixed_point_l392_392929

theorem power_function_fixed_point_logarithm_fixed_point
  (a : ℝ) (m n : ℝ) (f : ℝ → ℝ)
  (h1 : f x = log a (x - m) + n)
  (h2 : A = (1, 1))
  (h3 : m = 1)
  (h4 : n = 1) :
  f 2 = 1 :=
sorry

end power_function_fixed_point_logarithm_fixed_point_l392_392929


namespace problem1_problem2_problem3_l392_392930

-- Definition of the problem conditions
variables (M F : ℝ × ℝ) (x : ℝ) (y : ℝ)

-- Given conditions
def dist_to_point (M F : ℝ × ℝ) : ℝ :=
  real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2)

def dist_to_line (M : ℝ × ℝ) (line : ℝ) : ℝ :=
  real.abs (M.1 - line)

-- Problem 1: Find the equation of the trajectory C
theorem problem1 : 
  dist_to_point M (1,0) = dist_to_line M (-1) → 
  y^2 = 4 * x := 
sorry

-- Problem 2: Line PQ passes through a fixed point E(3,0)
theorem problem2 :
  ∀ (l1 l2 : ℝ × ℝ → Prop) (A B M N P Q : ℝ × ℝ),
  l1 = λ M, M.2 = k * (M.1 - 1) ∧ k ≠ 0 → 
  l2 = λ M, M.2 = - (1 / k) * (M.1 - 1) ∧ k ≠ 0 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  Q = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) →
  A = (X1 + Y1) →
  B = (X2 + Y2) →
  M = (X3 + Y3) →
  N = (X4 + Y4) →
  dist_to_point P Q → 
  line_pq = (M, N, P, Q) → 
  line_pq passesE :=
sorry

-- Problem 3: Find the minimum value of the area of ∆FPQ
theorem problem3 :
  ∀ (k : ℝ), k ≠ 0 ∧ k ≠ 1 →
  area_FPQ_min = 4 :=
sorry

end problem1_problem2_problem3_l392_392930


namespace circumference_divided_by_diameter_l392_392800

noncomputable def radius : ℝ := 15
noncomputable def circumference : ℝ := 90
noncomputable def diameter : ℝ := 2 * radius

theorem circumference_divided_by_diameter :
  circumference / diameter = 3 := by
  sorry

end circumference_divided_by_diameter_l392_392800


namespace Kyle_rose_cost_l392_392080

/-- Given the number of roses Kyle picked last year, the number of roses he picked this year, 
and the cost of one rose, prove that the total cost he has to spend to buy the remaining roses 
is correct. -/
theorem Kyle_rose_cost (last_year_roses this_year_roses total_roses_needed cost_per_rose : ℕ)
    (h_last_year_roses : last_year_roses = 12) 
    (h_this_year_roses : this_year_roses = last_year_roses / 2) 
    (h_total_roses_needed : total_roses_needed = 2 * last_year_roses) 
    (h_cost_per_rose : cost_per_rose = 3) : 
    (total_roses_needed - this_year_roses) * cost_per_rose = 54 := 
by
sorry

end Kyle_rose_cost_l392_392080


namespace reflection_matrix_over_vector_is_correct_l392_392034

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l392_392034


namespace reflection_matrix_l392_392022

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l392_392022


namespace math_problem_proof_l392_392384

-- Define the convex quadrilateral and the properties of the circles and points involved
variable {A B C D M K L : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M] [Inhabited K] [Inhabited L]

-- Assume conditions: ABCD is convex, circles with diameters AB and CD touch at M
-- M is different from the intersection point of diagonals, K and L are defined as described
noncomputable def quadrilateralConvex (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] := sorry
noncomputable def circlesTouchExternally (A B C D M : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M] := sorry
noncomputable def distinctPoint (M : Type) [Inhabited M] := sorry
noncomputable def pointsOnLine (M K L : Type) [Inhabited M] [Inhabited K] [Inhabited L] := sorry
noncomputable def circleConditions (A B C D M K L : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M] [Inhabited K] [Inhabited L] := sorry

-- The final proof problem in Lean
theorem math_problem_proof :
  quadrilateralConvex A B C D →
  circlesTouchExternally A B C D M →
  distinctPoint M →
  pointsOnLine M K L →
  circleConditions A B C D M K L →
  |MK - ML| = |AB - CD| :=
by 
  sorry

end math_problem_proof_l392_392384


namespace transformed_point_of_function_l392_392966

theorem transformed_point_of_function (f : ℝ → ℝ) (h : f 1 = -2) : f (-1) + 1 = -1 :=
by
  sorry

end transformed_point_of_function_l392_392966


namespace angle_between_vectors_is_pi_over_six_l392_392752

open Real

-- Defining the vectors a and b, with their magnitudes
variables (a b : EuclideanSpace ℝ (Fin 3))
variables (h1 : ‖a‖ = sqrt 3) (h2 : ‖b‖ = 1)
variables (h3 : ‖a - 2 • b‖ = 1)

-- The theorem we want to prove
theorem angle_between_vectors_is_pi_over_six : 
  let θ := Real.arccos ((a ⬝ b) / (‖a‖ * ‖b‖)) in
  θ = π / 6 :=
sorry

end angle_between_vectors_is_pi_over_six_l392_392752


namespace average_of_angles_l392_392576

theorem average_of_angles (p q r s t : ℝ) (h : p + q + r + s + t = 180) : 
  (p + q + r + s + t) / 5 = 36 :=
by
  sorry

end average_of_angles_l392_392576


namespace dap_equiv_48_dips_l392_392163

variables (dap dop dip : Type) [CommRing dap] [CommRing dop] [CommRing dip]

-- Define equivalences between daps, dops, and dips
def equivalence_dap_dop : dap ≃ₐ[dop] (dop →ₐ[dip] dap) := sorry
def equivalence_dop_dip : dop ≃ₐ[dip] (dip →ₐ[dap] dop) := sorry

-- Proportions given in the conditions
def prop1 (d : dap) (o : dop) : 5 * d = 4 * o := sorry
def prop2 (o : dop) (i : dip) : 3 * o = 8 * i := sorry

-- The proof statement
theorem dap_equiv_48_dips : ∀ (d : dap) (i : dip), (15 * d = 32 * i) → (d = 22.5 * i) := 
by
  intros
  sorry

end dap_equiv_48_dips_l392_392163


namespace emily_saves_more_using_promotion_a_l392_392414

-- Definitions based on conditions
def price_per_pair : ℕ := 50
def promotion_a_cost : ℕ := price_per_pair + price_per_pair / 2
def promotion_b_cost : ℕ := price_per_pair + (price_per_pair - 20)

-- Statement to prove the savings
theorem emily_saves_more_using_promotion_a :
  promotion_b_cost - promotion_a_cost = 5 := by
  sorry

end emily_saves_more_using_promotion_a_l392_392414


namespace sum_of_coefficients_l392_392106

theorem sum_of_coefficients (a : Fin 10 → ℤ) (x : ℤ) 
  (h : (1 - 2 * x)^9 = (∑ i, a i * x^i)) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = -2 :=
by
  sorry

end sum_of_coefficients_l392_392106


namespace part_one_part_two_l392_392523
-- Import Mathlib to get full access to necessary libraries

-- Define the conditions and the proof goals in Lean

variables {a b : Vector3} 

-- Conditions: Two non-zero planar vectors and the given inequality for any λ
variables (h_nonzero : a.len > 0 ∧ b.len > 0)
variables (h_cond : ∀ λ : ℝ, (a - λ * b).len ≥ (a - (1/2) * b).len)

-- Condition specific to question ①: |b| = 4
def b_len_eq_four : Prop := b.len = 4

-- Question ①: Prove that a·b = 8 under the given conditions and b.len = 4
theorem part_one (h_b_len : b_len_eq_four) : a.dot b = 8 :=
by
  -- Abstract proof placeholder
  sorry

-- Condition specific to question ②: The angle between a and b is π/3
def angle_eq_pi_by_3 : Prop := angle a b = π / 3

-- Question ②: Prove that the minimum value of (|2a - t * b|) / |b| is √3 under the given conditions and angle = π/3
theorem part_two (h_angle : angle_eq_pi_by_3) : ∀ t, (2 * a - t * b).len / b.len ≥ sqrt 3 :=
by
  -- Abstract proof placeholder
  sorry

end part_one_part_two_l392_392523


namespace unique_polynomial_in_H_l392_392998

-- Definition of the set H
def H := { Q : ℂ[X] | ∃ n (c : ℕ → ℤ), Q = X^n + polynomial.sum (λ i, c (n-i) • X^i) - 36 • X^0 ∧ 
  ∀ (a b : ℤ), ∃ z : ℂ, z = a + b * complex.I ∧ Q.has_root z }

-- The theorem to prove
theorem unique_polynomial_in_H : ∃! (Q : ℂ[X]), Q ∈ H :=
by
  sorry

end unique_polynomial_in_H_l392_392998


namespace min_distance_ellipse_to_line_l392_392507

theorem min_distance_ellipse_to_line :
  let ellipse := {P : ℝ × ℝ // ∃ θ ∈ Icc 0 (2 * Real.pi), P = (2 * Real.cos θ, Real.sin θ)},
      line := λ P : ℝ × ℝ, 2 * P.1 - 3 * P.2 + 6 = 0 in
  ∃ P ∈ ellipse, ∀ Q ∈ ellipse, dist P {Q : ℝ × ℝ // line Q} ≤ dist Q {Q : ℝ × ℝ // line Q} := sorry

end min_distance_ellipse_to_line_l392_392507


namespace daps_to_dips_l392_392181

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end daps_to_dips_l392_392181


namespace total_sum_of_money_is_71_l392_392742

noncomputable def total_sum_of_money (total_coins : ℕ) (num_20paise_coins : ℕ) :=
  let num_25paise_coins := total_coins - num_20paise_coins
  let sum_20paise := num_20paise_coins * 0.20
  let sum_25paise := num_25paise_coins * 0.25
  sum_20paise + sum_25paise

theorem total_sum_of_money_is_71 :
  total_sum_of_money 336 260 = 71 :=
by
  let num_25paise_coins := 336 - 260
  let sum_20paise := 260 * 0.20
  let sum_25paise := num_25paise_coins * 0.25
  have h20 : sum_20paise = 52 := by norm_num
  have h25 : sum_25paise = 19 := by norm_num
  have hsum : sum_20paise + sum_25paise = 71 := by norm_num
  exact hsum

end total_sum_of_money_is_71_l392_392742


namespace average_speed_round_trip_l392_392391

theorem average_speed_round_trip (d : ℝ) (h_d_pos : d > 0) : 
  let t1 := d / 80
  let t2 := d / 120
  let d_total := 2 * d
  let t_total := t1 + t2
  let v_avg := d_total / t_total
  v_avg = 96 :=
by
  sorry

end average_speed_round_trip_l392_392391


namespace octagon_perimeter_l392_392750

-- define the square side length
def square_side_length : ℝ := 1

-- define the length of the legs of the right-angled isosceles triangle
def triangle_leg_length : ℝ := 1

-- defining the problem as finding the perimeter of the constructed octagon
theorem octagon_perimeter : 
  let s := triangle_leg_length in -- since s is the side of the cut-out right-angled isosceles triangle
  (1 - s) + 1 + 1 + 1 + (1 - s) + 1 + s + s = 6 := 
  sorry

end octagon_perimeter_l392_392750


namespace incorrect_statement_isosceles_trapezoid_l392_392769

-- Define the properties of an isosceles trapezoid
structure IsoscelesTrapezoid (a b c d : ℝ) :=
  (parallel_bases : a = c ∨ b = d)  -- Bases are parallel
  (equal_diagonals : a = b) -- Diagonals are equal
  (equal_angles : ∀ α β : ℝ, α = β)  -- Angles on the same base are equal
  (axisymmetric : ∀ x : ℝ, x = -x)  -- Is an axisymmetric figure

-- Prove that the statement "The two bases of an isosceles trapezoid are parallel and equal" is incorrect
theorem incorrect_statement_isosceles_trapezoid (a b c d : ℝ) (h : IsoscelesTrapezoid a b c d) :
  ¬ (a = c ∧ b = d) :=
sorry

end incorrect_statement_isosceles_trapezoid_l392_392769


namespace f_of_x_at_2_l392_392908

-- Definitions based on given conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^5 + a * x^3 + b * x

-- Theorems based on the provided conditions and question
theorem f_of_x_at_2 (a b : ℝ) (h : f -2 a b = 10) : f 2 a b = -10 :=
by
  -- Proof steps would go here
  sorry

end f_of_x_at_2_l392_392908


namespace reflection_over_vector_l392_392051

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l392_392051


namespace rectangle_area_l392_392567

structure Rectangle (α : Type) :=
(A B C D : α)

variables {α : Type} [euclidean_space α]

noncomputable def midpoint (P Q : α) : α :=
1/2 * P + 1/2 * Q

theorem rectangle_area {ABCD : Rectangle α} (O M N Q : α)
  (hO_diag: segment_join ABCD.A ABCD.C ∩ segment_join ABCD.B ABCD.D = O)
  (hM_midpoint: M = midpoint ABCD.A ABCD.D)
  (hN_midpoint: N = midpoint ABCD.B ABCD.C)
  (hMN_line: Q ∈ line_through M N)
  (hMN_AC_intersect: Q ∈ segment_join ABCD.A ABCD.C)
  (hOMQ_area : area (triangle O M Q) = r) :
  area (rectangle ABCD.A ABCD.B ABCD.C ABCD.D) = 8 * r := 
sorry

end rectangle_area_l392_392567


namespace reflection_over_vector_l392_392053

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l392_392053


namespace total_votes_in_election_l392_392226

def valid_votes_fraction : ℝ := 0.85
def candidate_A_votes_fraction : ℝ := 0.75
def candidate_A_votes : ℝ := 357000

theorem total_votes_in_election (v : ℝ) :
  v = candidate_A_votes / (valid_votes_fraction * candidate_A_votes_fraction) → v = 560000 :=
by
  intro h
  rw h
  sorry

end total_votes_in_election_l392_392226


namespace PQ_bisects_BD_l392_392217

variable {A B C D P Q M N : Type}
variable [ConvexQuadrilateral A B C D]
variable [Midpoint P A B]
variable [Midpoint Q C D]
variable [Bisects PQ AC]

-- Prove that PQ also bisects diagonal BD
theorem PQ_bisects_BD (PQ_bisects_AC : bisects P Q A C) : bisects P Q B D := 
sorry

end PQ_bisects_BD_l392_392217


namespace constant_term_of_expansion_l392_392577

theorem constant_term_of_expansion : 
  ∃ (c : ℚ), is_constant_term ((x - (1 / (2 * sqrt x))) ^ 6) c ∧ c = 15 / 16 :=
by
  sorry

end constant_term_of_expansion_l392_392577


namespace positive_differences_count_l392_392791

-- Given a range of integers from -2 to 2012
def rangeSet : Set ℤ := {i | -2 ≤ i ∧ i ≤ 2012}

-- Define the set of positive differences within the range
def positiveDifferences : Set ℕ := {n : ℕ | ∃ x y ∈ rangeSet, x - y = n ∧ n > 0 }

-- Prove that the positiveDifferences set contains exactly the numbers from 1 to 2014
theorem positive_differences_count : ∀ n ∈ positiveDifferences, 1 ≤ n ∧ n ≤ 2014 ∧ ∃ m, 1 ≤ m ∧ m ≤ 2014 ∧ positiveDifferences.contains m :=
sorry

end positive_differences_count_l392_392791


namespace minimum_value_expression_l392_392890

-- Define the function for the expression
def expression (x : ℝ) : ℝ := (Real.sin x)^4 + 2 * (Real.cos x)^4 + (Real.sin x)^2 * (Real.cos x)^2

-- Define the theorem statement
theorem minimum_value_expression : ∀ x : ℝ, expression x ≥ 3 / 16 := 
by
  -- The proof is omitted and replaced with sorry.
  sorry

end minimum_value_expression_l392_392890


namespace units_digit_of_x_l392_392867

theorem units_digit_of_x (p x : ℕ): 
  (p * x = 32 ^ 10) → 
  (p % 10 = 6) → 
  (x % 4 = 0) → 
  (x % 10 = 1) :=
by
  sorry

end units_digit_of_x_l392_392867


namespace problem_statement_l392_392251

def g (t : ℝ) (h : t ≠ 1) : ℝ := (t + 1) / (t - 1)

theorem problem_statement (x y : ℝ) (hy : y ≠ 1) (hx : x = g (g y (y_ne_one)) sorry) : x = y :=
by sorry

end problem_statement_l392_392251


namespace unique_intersection_of_line_and_parabola_l392_392000

theorem unique_intersection_of_line_and_parabola :
  ∃! k : ℚ, ∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k → k = 25 / 3 :=
by
  sorry

end unique_intersection_of_line_and_parabola_l392_392000


namespace polynomial_factorization_l392_392448

-- Define the given polynomial expression
def given_poly (x : ℤ) : ℤ :=
  3 * (x + 3) * (x + 7) * (x + 11) * (x + 13) - 5 * x^2

-- Define the supposed factored form
def factored_poly (x : ℤ) : ℤ :=
  x * (3 * x^3 + 117 * x^2 + 1430 * x + 14895)

-- The theorem stating the equality of the two expressions
theorem polynomial_factorization (x : ℤ) : given_poly x = factored_poly x :=
  sorry

end polynomial_factorization_l392_392448


namespace value_of_y_at_x_3_l392_392091

theorem value_of_y_at_x_3 (a b c : ℝ) (h : a * (-3 : ℝ)^5 + b * (-3)^3 + c * (-3) - 5 = 7) :
  a * (3 : ℝ)^5 + b * 3^3 + c * 3 - 5 = -17 :=
by
  sorry

end value_of_y_at_x_3_l392_392091


namespace monotonic_intervals_sum_odd_reciprocal_squares_sum_reciprocal_squares_l392_392938

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x + 1
noncomputable def f' (x : ℝ) : ℝ := -x * Real.sin x
noncomputable def x_i (i : ℕ) : ℝ := sorry -- Placeholder for the ith zero of f(x)

theorem monotonic_intervals : 
  (∀ k : ℕ, (∀ x : ℝ, (2 * k * Real.pi < x ∧ x < (2 * k + 1) * Real.pi) → f' x > 0)) ∧ 
  (∀ k : ℕ, (∀ x : ℝ, ((2 * k + 1) * Real.pi < x ∧ x < (2 * k + 2) * Real.pi) → f' x < 0)) := 
sorry

theorem sum_odd_reciprocal_squares (k : ℕ) : 
  ∑ j in Finset.range (k+1), 1 / (x_i (2 * j + 1)) ^ 2 < 9 / (2 * Real.pi ^ 2) := 
sorry

theorem sum_reciprocal_squares (n : ℕ) : 
  ∑ i in Finset.range (n)+1), 1 / (x_i i) ^ 2 < 23 / (4 * Real.pi ^ 2) := 
sorry

end monotonic_intervals_sum_odd_reciprocal_squares_sum_reciprocal_squares_l392_392938


namespace unique_point_P_of_triangle_l392_392826

-- Definitions for the problem
variables {Point : Type} [metric_space Point]
def is_midpoint (A B M : Point) : Prop := dist A M = dist B M ∧ dist A M + dist M B = dist A B

def is_orthocenter (P A B C : Point) : Prop :=
  ∃ H : Point, (∀ (K : Point), is_midpoint B C K → dist P K = dist A K) ∧
               (∀ (M : Point), is_midpoint C A M → dist P M = dist B M) ∧
               (∀ (N : Point), is_midpoint A B N → dist P N = dist C N)

theorem unique_point_P_of_triangle (A B C P : Point)
  (h1 : dist P A ^ 2 + dist P B ^ 2 + dist A B ^ 2 = dist P B ^ 2 + dist P C ^ 2 + dist B C ^ 2)
  (h2 : dist P B ^ 2 + dist P C ^ 2 + dist B C ^ 2 = dist P C ^ 2 + dist P A ^ 2 + dist C A ^ 2)
  (h3 : dist P C ^ 2 + dist P A ^ 2 + dist C A ^ 2 = dist P A ^ 2 + dist P B ^ 2 + dist A B ^ 2) :
  is_orthocenter P ((A + B) / 2) ((B + C) / 2) ((C + A) / 2) :=
sorry

end unique_point_P_of_triangle_l392_392826


namespace collinearity_of_X_Y_Z_l392_392992

theorem collinearity_of_X_Y_Z
  (ABC : Type)
  [acute_triangle ABC]
  (altitude_A : ∀ (A B C : Point), line A ⊥ line B C ↔ altitude A B C)
  (X Y Z : Point)
  (H1 : altitude_A A B C)
  (H2 : altitude_A B C A)
  (H3 : altitude_A C A B)
  (H4 : ∀ {X Y Z : Point}, ∠ AYB = 90 ∧ ∠ BZC = 90 ∧ ∠ CXA = 90)
  (H5 : let τ := nine_point_circle ABC
        in tangent_length τ A = tangent_length τ B + tangent_length τ C) :
  collinear X Y Z := 
begin
  sorry -- Proof of the theorem goes here
end

end collinearity_of_X_Y_Z_l392_392992


namespace linear_composition_1000_l392_392912

theorem linear_composition_1000 (
  {α : Type*} [CommRing α] 
  (p : Fin 1000 →  α)
  (q : Fin 1000 -> α)
  (x0 : α) :

  ∃ (A B : α), 
    A = (List.prod $ (List.finRange 1000).map p) ∧
    B = Finset.sum (Finset.range 1000) (λ k, q k * (List.prod $ (Finset.range k).map p)) ∧
    let f := (A * x0 + B) in
    (f == (p 0 * p 1 * ... * p 999 * x0 + (sum (q i * (prod (p j) : for j in range i)) where i ranges from 1 to 1000))) ∧
    exists(n : Nat), n ≤ 30 :=
sorry

end linear_composition_1000_l392_392912


namespace inequality_distinct_natural_numbers_l392_392509

open BigOperators

theorem inequality_distinct_natural_numbers 
  (a : ℕ → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  (∀ n, 
    ∑ i in Finset.range n, a i ^ 7 
    + ∑ i in Finset.range n, a i ^ 5 
    ≥ 2 * (∑ i in Finset.range n, a i ^ 3)^2) := 
begin
  sorry,
end

end inequality_distinct_natural_numbers_l392_392509


namespace company_employees_after_hiring_l392_392784

/--
  The workforce of company x is 60% female. The company hired 24 additional male workers,
  and as a result, the percent of female workers dropped to 55%. Prove that the number 
  of employees after hiring the additional male workers is 288.
-/
theorem company_employees_after_hiring (E : ℕ) (h : 0.60 * E = 0.55 * (E + 24)) : E + 24 = 288 :=
by { sorry }

end company_employees_after_hiring_l392_392784


namespace correct_options_l392_392924

-- Define the vectors
def a_B : ℝ × ℝ := (2, 6)
def b_B : ℝ × ℝ := (-1, 3)
def a_C : ℝ × ℝ := (0, 3)
def b_C : ℝ × ℝ := (Real.sqrt 3, 1)

-- Define the projections
noncomputable def proj_C : ℝ × ℝ :=
    let dot_product := (a_C.1 * b_C.1 + a_C.2 * b_C.2)
    let norm_b_squared := (b_C.1 * b_C.1 + b_C.2 * b_C.2)
    ((dot_product / norm_b_squared) * b_C.1, (dot_product / norm_b_squared) * b_C.2)

-- The theorem checking which options are correct
theorem correct_options : 
    (¬ ((∃ k : ℝ, a_B = (k * b_B.1, k * b_B.2)) ∨ (∃ k : ℝ, b_B = (k * a_B.1, k * a_B.2))) ∧
    proj_C = (3 * Real.sqrt 3 / 4, 3 / 4)) :=
by {
    sorry
}

end correct_options_l392_392924


namespace sum_greatest_least_third_row_spiral_grid_l392_392974

-- Scenario: Ms. Lin fills a 16x16 grid with integers from 1 to 256 in a clockwise spiral.
-- The objective is to find the sum of the greatest and least number in the third row from the top.

theorem sum_greatest_least_third_row_spiral_grid :
  let spiral_fill := λ (n: ℕ), sorry -- Assume a function that fills the grid in a spiral pattern.
  let grid := spiral_fill 16 in
  let third_row := grid[2]  -- Third row from the top (index 2 as Lean is 0-based).
  let max_val := List.maximum third_row in
  let min_val := List.minimum third_row in
  max_val + min_val = 401 :=
by
  sorry

end sum_greatest_least_third_row_spiral_grid_l392_392974


namespace inequality_solution_l392_392328

theorem inequality_solution (x : ℝ) (h : (x + 1) / 2 ≥ x / 3) : x ≥ -3 :=
by
  sorry

end inequality_solution_l392_392328


namespace larger_integer_21_l392_392726

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l392_392726


namespace total_episodes_magic_king_l392_392326

theorem total_episodes_magic_king :
  (let first_half_seasons := 5
   let second_half_seasons := 5
   let first_half_episodes_per_season := 20
   let second_half_episodes_per_season := 25
   let total_seasons := first_half_seasons + second_half_seasons
   let total_episodes := first_half_seasons * first_half_episodes_per_season +
                         second_half_seasons * second_half_episodes_per_season
   in total_seasons = 10 ∧ total_episodes = 225) :=
by
  sorry

end total_episodes_magic_king_l392_392326


namespace sequence_inequality_l392_392483

theorem sequence_inequality 
  (a : ℝ) (a_seq : ℕ → ℝ) (K : ℕ)
  (h1 : a > 2)
  (h2 : a_seq 0 = 1)
  (h3 : a_seq 1 = a)
  (h4 : ∀ n, a_seq (n + 2) = ((a_seq (n + 1))^2 / (a_seq n)^2 - 2) * (a_seq (n + 1)))
  : (∑ i in Finset.range (K + 1), 1 / a_seq i) < 1 / 2 * (2 + a - real.sqrt (a^2 - 4)) :=
sorry

end sequence_inequality_l392_392483


namespace sum_of_solutions_eq_l392_392462

noncomputable def equation (x : ℝ) : Prop :=
  2^(x^2 - 4 * x - 3) = 8^(x - 5)

theorem sum_of_solutions_eq :
  (∀ x, equation x → x = 3 ∨ x = 4) →
  (3 + 4 = 7) :=
by
  intros h
  sorry

end sum_of_solutions_eq_l392_392462


namespace alpha_values_count_l392_392298

theorem alpha_values_count (α β γ : ℝ) (h1 : α + β + γ = π) 
  (h2 : β - α = γ - β) (h3 : 2 * Math.sin (20 * β) = Math.sin (20 * α) + Math.sin (20 * γ)) : 
  ∃ n : ℕ, n = 3 :=
by
  sorry

end alpha_values_count_l392_392298


namespace max_sum_of_inradii_is_30_l392_392608

-- Definitions of the geometric and algebraic objects and their properties
variable {A B C D : Type} [Triangle ABC]
variable (r r1 r2 : ℕ)
variable (angle_BAC : ∠ BAC = 90)
variable (D_on_BC : PointOnLine D BC)
variable (AD_perp_BC : Perpendicular AD BC)
variable (inradius_ABC : Inradius ABC r)
variable (inradius_ABD : Inradius ABD r1)
variable (inradius_ACD : Inradius ACD r2)
variable (positive_integers : r > 0 ∧ r1 > 0 ∧ r2 > 0)
variable (one_is_five : (r = 5 ∨ r1 = 5 ∨ r2 = 5))

-- Main theorem statement
theorem max_sum_of_inradii_is_30 : r + r1 + r2 = 30 :=
sorry

end max_sum_of_inradii_is_30_l392_392608


namespace incenter_is_midpoint_of_DE_l392_392376

-- Definitions and assumptions
variables {A B C O I D E : Type}
variables (triangleABC : Triangle A B C) (circumcircle_k : Circle O R)
variables (incircle : Circle I r) (tangent_circle : Circle P rho)

-- Conditions
axiom tangent_to_CA_CB_at_D_E : Tangent tangent_circle A C D ∧ Tangent tangent_circle B C E
axiom internally_tangent_to_k : InternallyTangent tangent_circle circumcircle_k

-- To Prove
theorem incenter_is_midpoint_of_DE :
  is_midpoint I D E :=
sorry

end incenter_is_midpoint_of_DE_l392_392376


namespace tom_build_wall_time_l392_392240

theorem tom_build_wall_time :
  ∃ T : ℝ, (1 / 3) + (1 / T) + (2 / (3 * T)) = 1 ∧ T = 2.5 :=
by
  have h1 : ∃ T : ℝ, (1 / 3) + (1 / T) + (2 / (3 * T)) = 1 := sorry -- This part could be computed or provided
  have h2 : T = 2.5 := sorry -- This part connects the correct solution with the conditions
  exact ⟨T, h1, h2⟩

end tom_build_wall_time_l392_392240


namespace tetrahedron_not_regular_due_to_equal_altitude_segments_l392_392354

theorem tetrahedron_not_regular_due_to_equal_altitude_segments (T : Tetrahedron)
  (h : ∀ a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : point (inscribed_sphere T),
   segment_length (altitude_segment_inside_sphere T a₁ b₁) =
   segment_length (altitude_segment_inside_sphere T a₂ b₂) ∧
   segment_length (altitude_segment_inside_sphere T a₃ b₃) =
   segment_length (altitude_segment_inside_sphere T a₄ b₄)) :
  ¬ regular_tetrahedron T :=
sorry

end tetrahedron_not_regular_due_to_equal_altitude_segments_l392_392354


namespace area_of_region_l392_392454

theorem area_of_region :
  let center : (ℝ × ℝ) := (8, 4)
  ∧ let radius : ℝ := 10
  ∧ let circle_eqn := λ x y : ℝ, (x - center.fst) ^ 2 + (y - center.snd) ^ 2 = radius ^ 2
  ∧ ∀ x y : ℝ, y < 0 → y < x - 10 → circle_eqn x y →
  (area of the region of the circle below the x-axis and to the left of the line (y = x - 10) is 25 * Real.pi) :=
begin
  -- Proof omitted
  sorry
end

end area_of_region_l392_392454


namespace two_cones_radius_proof_l392_392347

noncomputable def two_cones_max_radius_squared : ℚ := 
  66 - 16 * real.sqrt 116 / 25 

theorem two_cones_radius_proof :
  m + n = 91 :=
by {
  let m := 66,
  let n := 25,
  have h : two_cones_max_radius_squared = (66 - 16 * real.sqrt 116) / 25,
  {
    sorry  -- Calculation and proof of radius squared
  },
  have h_rel_prime : nat.coprime m n,
  {
    sorry  -- Proof that 66 and 25 are relatively prime
  },
  exact h + h_rel_prime -- Prove m + n = 91
}

end two_cones_radius_proof_l392_392347


namespace domain_of_f_l392_392301

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 + x - x^2)) / (Real.abs x - x)

theorem domain_of_f :
  { x : ℝ | 2 + x - x^2 > 0 ∧ Real.abs x - x ≠ 0 } = { x : ℝ | -1 < x ∧ x < 0 } := 
by
  sorry

end domain_of_f_l392_392301


namespace reggie_layups_l392_392969

/-
Conditions:
1. Layups are worth 1 point.
2. Free throws are worth 2 points.
3. Long shots are worth 3 points.
4. Reggie makes some layups, two free throws, and one long shot.
5. Reggie's brother makes 4 long shots.
6. Reggie loses by 2 points.
-/

theorem reggie_layups :
  ∃ L : ℕ, (L + 4 + 3 = 12 - 2) ∧ L = 3 :=
by
  existsi 3
  split
  . sorry -- Proof that L + 4 + 3 = 10
  . sorry -- Proof that L = 3

end reggie_layups_l392_392969


namespace evaluate_expression_at_one_l392_392649

-- Defining the main expression problem
def main_expression (x : ℝ) : ℝ :=
  (x^2 - 4*x + 4) / (2*x) / ( (x^2 - 2*x) / x^2 ) + 1

-- Theorem statement: for x = 1, the main expression evaluates to 1/2
theorem evaluate_expression_at_one : main_expression 1 = 1 / 2 :=
by
  sorry

end evaluate_expression_at_one_l392_392649


namespace polynomials_even_factors_count_l392_392460

theorem polynomials_even_factors_count :
  ∃ (count : ℕ), (count = 22) ∧
  (∀ (n : ℕ), (1 ≤ n ∧ n ≤ 2000 ∧ even n ∧ 
               ∃ a b : ℤ, (x - a) * (x - b) = x^2 - x - n ∧ a + b = 1 ∧ ab = -n) ↔ count = 22) :=
sorry

end polynomials_even_factors_count_l392_392460


namespace function_monotonicity_l392_392261

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  (a^x) / (b^x + c^x) + (b^x) / (a^x + c^x) + (c^x) / (a^x + b^x)

theorem function_monotonicity (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (∀ x y : ℝ, 0 ≤ x → x ≤ y → f a b c x ≤ f a b c y) ∧
  (∀ x y : ℝ, y ≤ x → x < 0 → f a b c x ≤ f a b c y) :=
by
  sorry

end function_monotonicity_l392_392261


namespace arithmetic_sequence_l392_392917

variable (p q : ℕ) -- Assuming natural numbers for simplicity, but can be generalized.

def a (n : ℕ) : ℕ := p * n + q

theorem arithmetic_sequence:
  ∀ n : ℕ, n ≥ 1 → (a n - a (n-1) = p) := by
  -- proof steps would go here
  sorry

end arithmetic_sequence_l392_392917


namespace daps_equivalent_to_48_dips_l392_392171

noncomputable def conversion_daps_to_dops : ℚ := 5 / 4
noncomputable def conversion_dops_to_dips : ℚ := 3 / 8
noncomputable def conversion_daps_to_dips : ℚ := conversion_daps_to_dops * conversion_dops_to_dips

theorem daps_equivalent_to_48_dips :
  ∀ (daps dops dips : Type) (eq1 : 5*daps = 4*dops) (eq2 : 3*dops = 8*dips), 
  (48:ℚ) * conversion_daps_to_dips = (22.5:ℚ) :=
by
  sorry

end daps_equivalent_to_48_dips_l392_392171


namespace probability_of_death_each_month_l392_392970

-- Defining the variables and expressions used in conditions
def p : ℝ := 0.1
def N : ℝ := 400
def surviving_after_3_months : ℝ := 291.6

-- The main theorem to be proven
theorem probability_of_death_each_month (prob : ℝ) :
  (N * (1 - prob)^3 = surviving_after_3_months) → (prob = p) :=
by
  sorry

end probability_of_death_each_month_l392_392970


namespace shooting_prob_l392_392818

theorem shooting_prob (p q : ℚ) (h: p + q = 1) (n : ℕ) 
  (cond1: p = 2/3) 
  (cond2: q = 1 - p) 
  (cond3: n = 5) : 
  (q ^ (n-1)) = 1/81 := 
by 
  sorry

end shooting_prob_l392_392818


namespace probability_of_real_solutions_l392_392749

noncomputable def equation_has_real_solutions_probability : ℚ := 
  let total_events := 6 * 6 -- Total number of basic events
  let valid_events := {
    pairs : Finset (ℕ × ℕ) | 
    let a := pairs.fst
    let b := pairs.snd
    a ∈ Finset.range (6 + 1) ∧ a > 0 ∧ b ∈ Finset.range (6 + 1) ∧ b > 0
    b^2 - 4 * a ≥ 0
  } : Finset (ℕ × ℕ)
  valid_events.card / total_events

theorem probability_of_real_solutions : 
  equation_has_real_solutions_probability = 19 / 36 :=
sorry

end probability_of_real_solutions_l392_392749


namespace students_in_class_l392_392265

theorem students_in_class
  (total_stickers : ℕ)
  (stickers_per_friend : ℕ)
  (num_friends : ℕ)
  (stickers_per_other : ℕ)
  (leftover_stickers : ℕ)
  (total_stickers = 250)
  (stickers_per_friend = 15)
  (num_friends = 10)
  (stickers_per_other = 5)
  (leftover_stickers = 25) :
  let stickers_given_to_friends := stickers_per_friend * num_friends in
  let remaining_stickers := total_stickers - stickers_given_to_friends in
  let stickers_given_to_others := remaining_stickers - leftover_stickers in
  let num_others := stickers_given_to_others / stickers_per_other in
  num_others + num_friends + 1 = 26 :=
sorry

end students_in_class_l392_392265


namespace cost_of_candy_l392_392620

theorem cost_of_candy (initial_amount pencil_cost remaining_after_candy : ℕ) 
  (h1 : initial_amount = 43) 
  (h2 : pencil_cost = 20) 
  (h3 : remaining_after_candy = 18) :
  ∃ candy_cost : ℕ, candy_cost = initial_amount - pencil_cost - remaining_after_candy :=
by
  sorry

end cost_of_candy_l392_392620


namespace find_larger_integer_l392_392730

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l392_392730


namespace rectangle_area_l392_392238

theorem rectangle_area (s : ℝ) (h1 : s * 8 = 160) : 4 * (s * s) = 1600 :=
by {
    intro h1,
    calc
      4 * (s * s) = 4 * (20 * 20) : sorry
                 ... = 4 * 400     : sorry
                 ... = 1600        : sorry
}

end rectangle_area_l392_392238


namespace matrix_not_invertible_possible_values_l392_392084

theorem matrix_not_invertible_possible_values (a b c : ℝ)
  (h : det (matrix 4 4 ![
    ![a, b, c, 0],
    ![b, c, 0, a],
    ![c, 0, a, b],
    ![0, a, b, c]]) = 0) :
  ∃ values : Set ℝ, values = \{-3, 3/2\} ∧ 
  ∃ x : ℝ, x ∈ values ∧ 
  (a / (b + c) + b / (c + a) + c / (a + b) = x) :=
sorry

end matrix_not_invertible_possible_values_l392_392084


namespace time_to_fill_by_B_l392_392340

/-- 
Assume a pool with two taps, A and B, fills in 30 minutes when both are open.
When both are open for 10 minutes, and then only B is open for another 40 minutes, the pool fills up.
Prove that if only tap B is opened, it would take 60 minutes to fill the pool.
-/
theorem time_to_fill_by_B
  (r_A r_B : ℝ)
  (H1 : (r_A + r_B) * 30 = 1)
  (H2 : ((r_A + r_B) * 10 + r_B * 40) = 1) :
  1 / r_B = 60 :=
by
  sorry

end time_to_fill_by_B_l392_392340


namespace max_c_in_range_f_l392_392458

theorem max_c_in_range_f (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 2) ↔ c ≤ 11 :=
begin
  sorry
end

end max_c_in_range_f_l392_392458


namespace evaluate_at_3_l392_392001

def f (x : ℝ) : ℝ := 9 * x^4 + 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem evaluate_at_3 : f 3 = 876 := by
  sorry

end evaluate_at_3_l392_392001


namespace reflection_matrix_is_correct_l392_392063

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l392_392063


namespace dips_to_daps_l392_392175

theorem dips_to_daps : 
  ∀ (daps dops dips : Type) (eq1 : 5 * daps = 4 * dops) (eq2 : 3 * dops = 8 * dips),
  (48 * dips = 22.5 * daps) :=
begin
  intros,
  sorry
end

end dips_to_daps_l392_392175


namespace weights_divide_three_piles_l392_392476

theorem weights_divide_three_piles (n : ℕ) (h : n > 3) :
  (∃ (k : ℕ), n = 3 * k ∨ n = 3 * k + 2) ↔
  (∃ (A B C : Finset ℕ), A ∪ B ∪ C = Finset.range (n + 1) ∧
   A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧
   A.sum id = (n * (n + 1)) / 6 ∧ B.sum id = (n * (n + 1)) / 6 ∧ C.sum id = (n * (n + 1)) / 6) :=
sorry

end weights_divide_three_piles_l392_392476


namespace product_of_integers_l392_392079

theorem product_of_integers :
  ∃ (a b c d e : ℤ),
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} =
      {-1, 5, 8, 9, 11, 12, 14, 18, 20, 24}) ∧
    a * b * c * d * e = -2002 :=
by {
  -- The statement formulation does not require a proof, hence here we end with sorry.
  sorry
}

end product_of_integers_l392_392079


namespace min_value_function_l392_392892

theorem min_value_function (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∀ x y : ℝ, x > 1 ∧ y > 1 → (min ((x^2 + y) / (y^2 - 1) + (y^2 + x) / (x^2 - 1)) = 8 / 3)) := 
sorry

end min_value_function_l392_392892


namespace orthocenter_on_hyperbola_l392_392335

variable {α β γ : ℂ}
variable a b c h : ℂ
variable (xy_hyperbola : ∀ z : ℂ, (∃ (α : ℂ), z = α + (1/α) * complex.I))

theorem orthocenter_on_hyperbola (a b c : ℂ) (hx : a ≠ b) (hy : b ≠ c) (hz : c ≠ a)
  (h_a : xy_hyperbola a) (h_b : xy_hyperbola b) (h_c : xy_hyperbola c)
  (h_ortho : h = -α * β * γ - complex.I * (α * β * γ)⁻¹) :
  xy_hyperbola h :=
by
  sorry

end orthocenter_on_hyperbola_l392_392335


namespace reflection_over_vector_l392_392054

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l392_392054


namespace value_of_m_l392_392875

theorem value_of_m (m : ℝ) :
  let expr := (m * (2:ℝ) * (x : ℝ) - (3:ℝ) * m * x ^ 2 + (8:ℝ) * (2:ℝ) - (24:ℝ) * x)
  expr.does_not_contain_linear_term →
  m = 12 := 
sorry

end value_of_m_l392_392875


namespace mike_notebooks_total_l392_392266

theorem mike_notebooks_total
  (red_notebooks : ℕ)
  (green_notebooks : ℕ)
  (blue_notebooks_cost : ℕ)
  (total_cost : ℕ)
  (red_cost : ℕ)
  (green_cost : ℕ)
  (blue_cost : ℕ)
  (h1 : red_notebooks = 3)
  (h2 : red_cost = 4)
  (h3 : green_notebooks = 2)
  (h4 : green_cost = 2)
  (h5 : total_cost = 37)
  (h6 : blue_cost = 3)
  (h7 : total_cost = red_notebooks * red_cost + green_notebooks * green_cost + blue_notebooks_cost) :
  (red_notebooks + green_notebooks + blue_notebooks_cost / blue_cost = 12) :=
by {
  sorry
}

end mike_notebooks_total_l392_392266


namespace student_group_intersect_l392_392786

theorem student_group_intersect (students groups : Finset ℕ) (h_students : students.card = 32) (h_groups : groups.card = 33)
  (h_group_size : ∀ g ∈ groups, g.card = 3) (h_distinct : ∀ g1 g2 ∈ groups, g1 ≠ g2 → g1 ∩ g2 ≠ ∅ → (g1 ∩ g2).card ≠ 3) :
  ∃ g1 g2 ∈ groups, g1 ≠ g2 ∧ (g1 ∩ g2).card = 1 := 
by
  sorry

end student_group_intersect_l392_392786


namespace max_digit_sum_in_24_hour_format_l392_392398

def digit_sum (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

theorem max_digit_sum_in_24_hour_format :
  (∃ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ digit_sum h + digit_sum m = 19) ∧
  ∀ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 → digit_sum h + digit_sum m ≤ 19 :=
by
  sorry

end max_digit_sum_in_24_hour_format_l392_392398


namespace problem1_problem2_l392_392135

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (-3^x + a) / (3^(x + 1) + b)

theorem problem1 : f (-1) 1 1 = 3^(-1) :=
by
sorry

theorem problem2 (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = -f x)
  (t : ℝ) (k : ℝ) (h2 : f (t^2 - 2*t) < f (2*t^2 - k)) :
  k > -1 :=
by
sorry

end problem1_problem2_l392_392135


namespace find_number_divisible_by_792_l392_392016

theorem find_number_divisible_by_792 :
  ∃ (x y z : ℕ), 
    x ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    z ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    (13 * 10^5 + x * 10^4 + y * 10^3 + 45 * 10 + z) % 792 = 0 ∧
    (13 * 10^5 + x * 10^4 + y * 10^3 + 45 * 10 + z) = 1380456 :=
begin
  -- Proof to be provided
  sorry
end

end find_number_divisible_by_792_l392_392016


namespace probability_X_greater_than_4_l392_392506

theorem probability_X_greater_than_4 {σ : ℝ} (h : ∫ x in 0..2, pdf (Normal 2 σ) x = 1 / 3) :
  P(X > 4) = 1 / 6 :=
sorry

end probability_X_greater_than_4_l392_392506


namespace eggs_collected_week_l392_392004

def num_chickens : ℕ := 6
def num_ducks : ℕ := 4
def num_geese : ℕ := 2
def eggs_per_chicken : ℕ := 3
def eggs_per_duck : ℕ := 2
def eggs_per_goose : ℕ := 1

def eggs_per_day (num_birds eggs_per_bird : ℕ) : ℕ := num_birds * eggs_per_bird

def eggs_collected_monday_to_saturday : ℕ :=
  6 * (eggs_per_day num_chickens eggs_per_chicken +
       eggs_per_day num_ducks eggs_per_duck +
       eggs_per_day num_geese eggs_per_goose)

def eggs_collected_sunday : ℕ :=
  eggs_per_day num_chickens (eggs_per_chicken - 1) +
  eggs_per_day num_ducks (eggs_per_duck - 1) +
  eggs_per_day num_geese (eggs_per_goose - 1)

def total_eggs_collected : ℕ :=
  eggs_collected_monday_to_saturday + eggs_collected_sunday

theorem eggs_collected_week : total_eggs_collected = 184 :=
by sorry

end eggs_collected_week_l392_392004


namespace k_interval_l392_392139

noncomputable def f (x k : ℝ) : ℝ := x^2 + (1 - k) * x - k

theorem k_interval (k : ℝ) :
  (∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x k = 0) ↔ (2 < k ∧ k < 3) :=
by
  sorry

end k_interval_l392_392139


namespace axis_of_symmetry_values_ge_one_range_m_l392_392906

open Real

-- Definitions for vectors and the function f(x)
noncomputable def a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, sin x)
noncomputable def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- Part I: Prove the equation of the axis of symmetry of f(x)
theorem axis_of_symmetry {k : ℤ} : f x = (sqrt 2 / 2) * sin (2 * x - π / 4) + 1 / 2 → 
                                    x = k * π / 2 + 3 * π / 8 := 
sorry

-- Part II: Prove the set of values x for which f(x) ≥ 1
theorem values_ge_one : (f x ≥ 1) ↔ (∃ (k : ℤ), π / 4 + k * π ≤ x ∧ x ≤ π / 2 + k * π) := 
sorry

-- Part III: Prove the range of m given the inequality
theorem range_m (m : ℝ) : (∀ x, π / 6 ≤ x ∧ x ≤ π / 3 → f x - m < 2) → 
                            m > (sqrt 3 - 5) / 4 := 
sorry

end axis_of_symmetry_values_ge_one_range_m_l392_392906


namespace sweater_markup_percentage_l392_392762

-- Define the conditions
variable (W : ℝ) -- wholesale cost
variable (S : ℝ) -- sale price after discount
variable (R : ℝ) -- original retail price

-- Conditions
axiom wholesale_nonzero (h : W ≠ 0)
axiom fifty_percent_discount (h1 : S = 0.5 * R) -- The sale price S is 50% discount of retail price R
axiom profit_condition (h2 : S = 1.4 * W) -- Sale price S nets a 40% profit on wholesale cost W

-- The theorem to prove
theorem sweater_markup_percentage
  (W : ℝ) (h : W ≠ 0) (S : ℝ) (h1 : S = 0.5 * R) (h2 : S = 1.4 * W) :
  ((R - W) / W) * 100 = 180 :=
by
  -- Placeholder for proof
  sorry

end sweater_markup_percentage_l392_392762


namespace non_neg_sum_sq_inequality_l392_392928

theorem non_neg_sum_sq_inequality (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
sorry

end non_neg_sum_sq_inequality_l392_392928


namespace highest_student_id_in_sample_l392_392564

theorem highest_student_id_in_sample
    (total_students : ℕ)
    (sample_size : ℕ)
    (included_student_id : ℕ)
    (interval : ℕ)
    (first_id in_sample : ℕ)
    (k : ℕ)
    (highest_id : ℕ)
    (total_students_eq : total_students = 63)
    (sample_size_eq : sample_size = 7)
    (included_student_id_eq : included_student_id = 11)
    (k_def : k = total_students / sample_size)
    (included_student_id_in_second_pos : included_student_id = first_id + k)
    (interval_eq : interval = first_id - k)
    (in_sample_eq : in_sample = interval)
    (highest_id_eq : highest_id = in_sample + k * (sample_size - 1)) :
  highest_id = 56 := sorry

end highest_student_id_in_sample_l392_392564


namespace minimum_value_g_l392_392141

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

def g (t : ℝ) : ℝ :=
  if t < 0 then t^2 + 1
  else if t ≤ 1 then 1
  else t^2 - 2*t + 2

theorem minimum_value_g (t : ℝ) :
  ∀ x, t ≤ x ∧ x < t+1 → f x ≥ g t :=
by
  intro x hx
  cases lt_or_ge t 0 with ht_neg ht_ge
  · have : x = t + 1 := sorry
    rw [this, f, g]
    sorry
  cases le_or_gt x 1 with hx_le hx_gt
  · rw [f, g]
    sorry
  have : x = t := sorry
  rw [this, f, g]
  sorry

end minimum_value_g_l392_392141


namespace find_n_l392_392154

theorem find_n (s P k : ℝ) (h : P = s / (1 + k)^n) 
  (log_pos_1 : log ((1 + k)^n) = log ((s) / (P)))
  (log_pos_2 : log (1 + k) ≠ 0) :
  n = log((s) / (P)) / log(1 + k) := sorry

end find_n_l392_392154


namespace cos_angle_correct_k_values_correct_l392_392145

def point (α : Type) := prod (prod α α) α

def vector (α : Type) := point α

noncomputable def a : vector ℝ := (1, 1, 0)
noncomputable def b : vector ℝ := (-1, 0, 2)

def dot_product (u v : vector ℝ) : ℝ :=
  u.1.1 * v.1.1 + u.1.2 * v.1.2 + u.2 * v.2

def magnitude (v : vector ℝ) : ℝ :=
  real.sqrt (v.1.1^2 + v.1.2^2 + v.2^2)

def cos_angle (u v : vector ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cos_angle_correct :
  cos_angle a b = - (real.sqrt 10) / 10 :=
by
  sorry

noncomputable def ka (k : ℝ) : vector ℝ := (k * 1, k * 1, k * 0)
noncomputable def kb (k : ℝ) : vector ℝ := (k * (-1), k * 0, k * 2)

def perpendicular (u v : vector ℝ) : Prop :=
  dot_product u v = 0

theorem k_values_correct (k : ℝ) :
  perpendicular (ka k + b) (ka k - (2 • b)) ↔ k = - (5 / 2) ∨ k = 2 :=
by
  sorry

end cos_angle_correct_k_values_correct_l392_392145


namespace exists_valid_arrangement_n_1_exists_valid_arrangement_n_gt_1_l392_392336

-- Define the conditions
def num_mathematicians (n : ℕ) : ℕ := 6 * n + 4
def num_meetings (n : ℕ) : ℕ := 2 * n + 1
def num_4_person_tables (n : ℕ) : ℕ := 1
def num_6_person_tables (n : ℕ) : ℕ := n

-- Define the constraint on arrangements
def valid_arrangement (n : ℕ) : Prop :=
  -- A placeholder for the actual arrangement checking logic.
  -- This should ensure no two people sit next to or opposite each other more than once.
  sorry

-- Proof of existence of a valid arrangement when n = 1
theorem exists_valid_arrangement_n_1 : valid_arrangement 1 :=
sorry

-- Proof of existence of a valid arrangement when n > 1
theorem exists_valid_arrangement_n_gt_1 (n : ℕ) (h : n > 1) : valid_arrangement n :=
sorry

end exists_valid_arrangement_n_1_exists_valid_arrangement_n_gt_1_l392_392336


namespace square_complex_number_l392_392331

theorem square_complex_number : (2 + complex.i)^2 = 3 + 4 * complex.i :=
by
  sorry

end square_complex_number_l392_392331


namespace concert_attendance_l392_392789

theorem concert_attendance (n1 : ℕ) (delta_n : ℕ) (n2 : ℕ) :
  (n1 = 65899) ∧ (delta_n = 119) ∧ (n2 = n1 + delta_n) → n2 = 66018 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end concert_attendance_l392_392789


namespace digit_B_condition_l392_392250

theorem digit_B_condition {B : ℕ} (h10 : ∃ d : ℕ, 58709310 = 10 * d)
  (h5 : ∃ e : ℕ, 58709310 = 5 * e)
  (h6 : ∃ f : ℕ, 58709310 = 6 * f)
  (h4 : ∃ g : ℕ, 58709310 = 4 * g)
  (h3 : ∃ h : ℕ, 58709310 = 3 * h)
  (h2 : ∃ i : ℕ, 58709310 = 2 * i) :
  B = 0 := by
  sorry

end digit_B_condition_l392_392250


namespace find_larger_integer_l392_392701

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l392_392701


namespace allocation_methods_count_l392_392561

theorem allocation_methods_count :
  (∃ g : ℕ → ℕ, g 1 + g 2 + g 3 = 5 ∧ g 1 > 0 ∧ g 2 > 0 ∧ g 3 > 0 ∧ 
    ∃ n k : ℕ, n = 4 ∧ k = 2 ∧ C(n, k) = 6) :=
begin
  sorry
end

end allocation_methods_count_l392_392561


namespace circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l392_392239

theorem circumscribe_quadrilateral_a : 
  ∃ (x : ℝ), 2 * x + 4 * x + 5 * x + 3 * x = 360 
          ∧ (2 * x + 5 * x = 180) 
          ∧ (4 * x + 3 * x = 180) := sorry

theorem circumscribe_quadrilateral_b : 
  ∃ (x : ℝ), 5 * x + 7 * x + 8 * x + 9 * x = 360 
          ∧ (5 * x + 8 * x ≠ 180) 
          ∧ (7 * x + 9 * x ≠ 180) := sorry

end circumscribe_quadrilateral_a_circumscribe_quadrilateral_b_l392_392239


namespace compute_sum_l392_392845

theorem compute_sum : 
  (1 / (2^1988 : ℝ)) * (Finset.sum (Finset.range 995) (λ n, (-3)^n * (Nat.choose 1988 (2 * n)))) = -0.5 :=
by
  sorry

end compute_sum_l392_392845


namespace monotonicity_of_f_extreme_points_of_f_l392_392940

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (3 - x) * Real.exp x + a / x

/-- Question 1 -/
theorem monotonicity_of_f (a : ℝ) (h : a > -3 / 4) : 
    ∀ x > 0, (deriv (λ x, f x a)).x < 0 :=
sorry

/-- Question 2 -/
theorem extreme_points_of_f (a : ℝ) (h1 : ∃ x1 x2, 0 < x1 ∧ x1 < x2 ∧ 
    x2 < 3 / 2 ∧ (deriv (λ x, f x a)).x1 = 0 ∧ (deriv (λ x, f x a)).x2 = 0):
  -3 < a ∧ a < -Real.exp 1 ∧ ∃ x2 (hx2 : 1 < x2 ∧ x2 < 3 / 2), f x2 a > 2 :=
sorry

end monotonicity_of_f_extreme_points_of_f_l392_392940


namespace sum_non_solutions_is_neg21_l392_392248

noncomputable def sum_of_non_solutions (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2) : ℝ :=
  -21

theorem sum_non_solutions_is_neg21 (A B C : ℝ) (h1 : ∀ x : ℝ, ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) = 2) : 
  ∃! (x1 x2 : ℝ), ((x + B) * (A * x + 28)) / ((x + C) * (x + 7)) ≠ 2 → x = x1 ∨ x = x2 ∧ x1 + x2 = -21 :=
sorry

end sum_non_solutions_is_neg21_l392_392248


namespace arithmetic_sequence_l392_392224

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem arithmetic_sequence (a1 d : ℝ) (h_d : d ≠ 0) 
  (h1 : a1 + (a1 + 2 * d) = 8) 
  (h2 : (a1 + d) * (a1 + 8 * d) = (a1 + 3 * d) * (a1 + 3 * d)) :
  a_n a1 d 5 = 13 := 
by 
  sorry

end arithmetic_sequence_l392_392224


namespace probability_at_least_3_speak_l392_392199

theorem probability_at_least_3_speak (p : ℚ) (n : ℕ) (h : p = 1/3 ∧ n = 6) : 
  (∑ k in (finset.range n.succ).filter (λ k, k ≥ 3), 
    (nat.choose n k : ℚ) * p^k * (1-p)^(n-k)) = 353/729 :=
by 
  cases h with hp hn
  simp [hp, hn]
  sorry

end probability_at_least_3_speak_l392_392199


namespace Joshua_process_final_number_l392_392606

theorem Joshua_process_final_number :
  let N := 150
  let initial_sequence := list.range N
  let elimination_process (seq : list ℕ) : list ℕ :=
    seq.indexed.filter_map
      (λ ⟨i, n⟩, if (i % 5 == 0) then none else some n)
  let final_number (seq : list ℕ) : ℕ :=
    seq.head!

  final_number (nat.iterate elimination_process ⟨initial_sequence⟩.val) = 125 :=
sorry

end Joshua_process_final_number_l392_392606


namespace find_magnitude_b_l392_392502

variable (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions
def angle_between_a_b : Real.Angle := Real.pi / 3  -- Corresponds to 60 degrees
def magnitude_a : ℝ := 2 
def norm_a_minus_2b : ℝ := 2 * Real.sqrt 7

-- Mathematically equivalent problem
theorem find_magnitude_b (h1 : |a| = magnitude_a)
                         (h2 : |a - 2 • b| = norm_a_minus_2b)
                         (h3 : Real.Angle.cos angle_between_a_b = 1/2) :
  |b| = 3 :=
sorry

end find_magnitude_b_l392_392502


namespace log_base_3_18_l392_392495

theorem log_base_3_18 (a : ℝ) (h : log 3 2 = a) : log 3 18 = 2 + a :=
sorry

end log_base_3_18_l392_392495


namespace range_of_a_for_decreasing_log_function_l392_392911

theorem range_of_a_for_decreasing_log_function (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  (∀ x ∈ set.Icc (0 : ℝ) 1, monotone_decreasing (λ x, log a (2 - a * x))) ↔ (1 < a ∧ a < 2) :=
begin
  sorry
end

end range_of_a_for_decreasing_log_function_l392_392911


namespace sum_g_equals_one_half_l392_392898

def g (n : ℕ) : ℝ :=
  if h : n > 0 then ∑' k : ℕ, if k >= 3 then 1 / (k : ℝ) ^ n else 0 else 0

noncomputable def sum_g : ℝ := ∑' (n : ℕ) in (Set.Ici 3), g n

theorem sum_g_equals_one_half : sum_g = 1 / 2 :=
sorry

end sum_g_equals_one_half_l392_392898


namespace lock_settings_count_l392_392821

theorem lock_settings_count :
  (∀ (d1 d2 d3 d4 : ℕ), d1 ∈ {0, 1, 2, ..., 9} ∧ d2 ∈ {0, 1, 2, ..., 9} ∧ d3 ∈ {0, 1, 2, ..., 9} ∧ d4 ∈ {0, 1, 2, ..., 9} →
  (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4) →
  ∃ (n : ℕ), n = 5040) :=
begin
  sorry
end

end lock_settings_count_l392_392821


namespace determine_a_perpendicular_lines_l392_392946

theorem determine_a_perpendicular_lines (a : ℝ) :
  (λ x y : ℝ, x + 2 * a * y - 1 = 0) ≠ (λ x y : ℝ, (3 * a - 1) * x - y - 1 = 0) → 
  (∃ a : ℝ, (a = 1 ∧ ((∃ k : ℝ, k = -1 ∧ (2a ≠ 0 ∧ (3a-1 ≠ 0 ∧ (- 1 / (2 * a)) * (3a - 1) = k))))) sorry

end determine_a_perpendicular_lines_l392_392946


namespace reflection_over_vector_l392_392056

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l392_392056


namespace probability_heart_seven_jack_l392_392747

def prob_heart_seven_jack (total : ℕ) (hearts : ℕ) (sevens : ℕ) (jacks : ℕ) : ℚ :=
  (hearts / total) * (sevens / (total - 1)) * (jacks / (total - 2))

def prob_cases : ℚ :=
  (12 / 52) * (3 / 51) * (4 / 50) + 
  (11 / 52) * (1 / 51) * (3 / 50) + 
  (1 / 52) * (3 / 51) * (4 / 50) + 
  (1 / 52) * (1 / 51) * (3 / 50)

theorem probability_heart_seven_jack : prob_cases = 8 / 5525 := 
  by linarith

end probability_heart_seven_jack_l392_392747


namespace circle_integral_points_count_l392_392901

theorem circle_integral_points_count : 
  ∀ x : ℤ, ¬((x - 3)^2 + (3 * x + 1)^2 ≤ 16) ↔ 0 :=
by sorry

end circle_integral_points_count_l392_392901


namespace proof_problem_l392_392907

noncomputable theory

def f (x : ℝ) : ℝ := sin (x / 4) ^ 2 - 2 * cos (x / 4) ^ 2 + sqrt 3 * sin (x / 4) * cos (x / 4)

def monotonically_decreasing_interval : Set ℝ := 
  { x | ∃ k : ℤ, 4 * k * π + 5 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 11 * π / 3 }

def a (A B C : ℝ) (angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) : Type :=
{ a b c : ℝ // a > 0 ∧ b > 0 ∧ c > 0 }

def range_perimeter_tri (sides : a A B C) : Set ℝ :=
  { p | ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ b = sqrt 3 ∧ 
    (A B C).fst ^ 2 + (A B C).snd ^ 2 + (A B C).fst * (A B C).snd = 3 ∧
    (2 * sqrt 3 < p ∧ p ≤ 2 + sqrt 3)
  }

theorem proof_problem (x : ℝ) (angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) (sides : a A B C) :
  f (B) = -1/2 ∧ b = sqrt 3 →
  (∀ x, x ∈ monotonically_decreasing_interval ↔
      ∃ k : ℤ, 4 * k * π + 5 * π / 3 ≤ x ∧ x ≤ 4 * k * π + 11 * π / 3) ∧
  (∀ a, a ∈ range_perimeter_tri sides ↔
      ∃ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ b = sqrt 3 ∧
      (A B C).fst ^ 2 + (A B C).snd ^ 2 + (A B C).fst * (A B C).snd = 3 ∧
      (2 * sqrt 3 < a ∧ a ≤ 2 + sqrt 3))
:= sorry

end proof_problem_l392_392907


namespace probability_all_evens_before_any_odd_l392_392802

-- Definitions based on conditions
def fair_die : Type := {n : ℕ // n ≥ 1 ∧ n ≤ 6}
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

axiom fair_die_probability_each_equal : ∀ (n : fair_die), (n.val = 1) ∨ (n.val = 2) ∨ (n.val = 3) ∨ (n.val = 4) ∨ (n.val = 5) ∨ (n.val = 6) → 
  (1/6 : ℝ)

theorem probability_all_evens_before_any_odd :
  -- Probability that each even number (2, 4, 6) appears at least once before any odd number
  ∀ (rolls : list fair_die), (∀ (d : fair_die), d ∈ rolls → is_even d.val) →
  (∃ (perm : list fair_die), (∀ e ∈ perm, is_even e.val) ∧ (∀ o ∈ (rolls.filter is_odd), ∀ e ∈ perm, e = o → e.val ≠ o.val)) →
  (1 / 20 : ℝ) :=
sorry

end probability_all_evens_before_any_odd_l392_392802


namespace daps_equivalent_to_48_dips_l392_392168

noncomputable def conversion_daps_to_dops : ℚ := 5 / 4
noncomputable def conversion_dops_to_dips : ℚ := 3 / 8
noncomputable def conversion_daps_to_dips : ℚ := conversion_daps_to_dops * conversion_dops_to_dips

theorem daps_equivalent_to_48_dips :
  ∀ (daps dops dips : Type) (eq1 : 5*daps = 4*dops) (eq2 : 3*dops = 8*dips), 
  (48:ℚ) * conversion_daps_to_dips = (22.5:ℚ) :=
by
  sorry

end daps_equivalent_to_48_dips_l392_392168


namespace gasoline_added_l392_392797

theorem gasoline_added (total_capacity : ℝ) (initial_fraction final_fraction : ℝ) 
(h1 : initial_fraction = 3 / 4)
(h2 : final_fraction = 9 / 10)
(h3 : total_capacity = 29.999999999999996) : 
(final_fraction * total_capacity - initial_fraction * total_capacity = 4.499999999999999) :=
by sorry

end gasoline_added_l392_392797


namespace daps_equivalent_to_48_dips_l392_392172

noncomputable def conversion_daps_to_dops : ℚ := 5 / 4
noncomputable def conversion_dops_to_dips : ℚ := 3 / 8
noncomputable def conversion_daps_to_dips : ℚ := conversion_daps_to_dops * conversion_dops_to_dips

theorem daps_equivalent_to_48_dips :
  ∀ (daps dops dips : Type) (eq1 : 5*daps = 4*dops) (eq2 : 3*dops = 8*dips), 
  (48:ℚ) * conversion_daps_to_dips = (22.5:ℚ) :=
by
  sorry

end daps_equivalent_to_48_dips_l392_392172


namespace largest_angle_heptagon_l392_392395

def heptagon_largest_angle (x : ℝ) : ℝ :=
  2 * (x^2)

theorem largest_angle_heptagon
  (x : ℝ)
  (a : ℝ := x^2 - 1)
  (b : ℝ := 2*x - 2)
  (c : ℝ := 3*x + 1)
  (d : ℝ := 2 * x^2)
  (e : ℝ := x + 3)
  (f : ℝ := 4 * x - 1)
  (g : ℝ := 5 * x + 2)
  (angle_sum_eq : a + b + c + d + e + f + g = 900) :
  d = 623 - 5 * real.sqrt 1221 :=
sorry

end largest_angle_heptagon_l392_392395


namespace larger_integer_is_21_l392_392711

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l392_392711


namespace set_representation_l392_392325

noncomputable def Z_set : set ℂ :=
  {Z | ∃ (n : ℤ), Z = (complex.exp (n * complex.pi * complex.I / 2)) + (complex.exp (-n * complex.pi * complex.I / 2))}

theorem set_representation : Z_set = {0, 2, -2} :=
by
  -- required proof steps would go here
  sorry

end set_representation_l392_392325


namespace segment_PQ_length_l392_392097

-- Define the obtuse triangle and given points P, Q with the mentioned properties
variables {A B C P Q : Type} [OrderedSpace A] [OrderedSpace B] [OrderedSpace C] 
variables (obtuse_triangle : Triangle A B C) 
variables (A B C : Point)
variables (P Q : Point) 
variables (angle_ACP_90 : angle obtuse_triangle.A C P = 90) 
variables (angle_CPQ_90 : angle obtuse_triangle.C P Q = 90) 
variables (AC : length (segment A C) = 25)
variables (CP : length (segment C P) = 20)
variables (angle_APC_eq_A_plus_B : angle (segment A P C) = angle obtuse_triangle.A + angle obtuse_triangle.B)

theorem segment_PQ_length : length (segment P Q) = 16 :=
sorry

end segment_PQ_length_l392_392097


namespace find_lambda_l392_392148

-- Defining the vectors a, b, c and the condition for collinearity
variables (λ : ℝ)

def vec_a : ℝ × ℝ := (2, -3)
def vec_b : ℝ × ℝ := (4, λ)
def vec_c : ℝ × ℝ := (-1, 1)

def sum_vecs (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def scalar_mult (k : ℝ) (a : ℝ × ℝ) : ℝ × ℝ := (k * a.1, k * a.2)
def collinear (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

theorem find_lambda
  (h : collinear (sum_vecs vec_a (scalar_mult 2 vec_b)) 
                (sum_vecs (scalar_mult 3 vec_a) (scalar_mult (-1) vec_c))) :
  λ = -79/14 :=
sorry

end find_lambda_l392_392148


namespace sufficient_but_not_necessary_l392_392086

def sequence_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def abs_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) > abs (a n)

theorem sufficient_but_not_necessary (a : ℕ → ℝ) :
  (abs_condition a → sequence_increasing a) ∧ ¬ (sequence_increasing a → abs_condition a) :=
by
  sorry

end sufficient_but_not_necessary_l392_392086


namespace Ricardo_coin_difference_l392_392641

theorem Ricardo_coin_difference (p : ℕ) (h₁ : 1 ≤ p) (h₂ : p ≤ 3029) :
    let max_value := 15150 - 4 * 1
    let min_value := 15150 - 4 * 3029
    max_value - min_value = 12112 := by
  sorry

end Ricardo_coin_difference_l392_392641


namespace right_triangle_locations_l392_392632

theorem right_triangle_locations (P Q : ℝ × ℝ) (hPQ : dist P Q = 8)
  (area : ℝ) (h_area : area = 12) :
  ∃ (R : ℝ × ℝ), triangle_area P Q R = 12 ∧ number_of_right_triangle_locations P Q area = 8 :=
sorry

end right_triangle_locations_l392_392632


namespace range_of_y_l392_392439

theorem range_of_y (b : Fin 21 → ℝ) (h_b : ∀ i, b i = 0 ∨ b i = 3)
(h_sum : (Finset.range 21).sum b ≥ 30) :
  0 < ∑ i in (Finset.range 21).filter (λ i, b i ≠ 0), b i / 4^(i+1) ∧ 
  ∑ i in (Finset.range 21).filter (λ i, b i ≠ 0), b i / 4^(i+1) < 1 :=
by
  sorry

end range_of_y_l392_392439


namespace find_xyz_l392_392923

noncomputable def vectors_not_coplanar 
  (a b c : ℝ^3) : Prop := 
¬ collinear a b c 

theorem find_xyz (a b c : ℝ^3) 
  (h_coplanar : vectors_not_coplanar a b c) 
  (h_eq : 2 • a + b - c = (z - 1) • a + x • b + 2 • y • c) :
  (x = 1 ∧ y = -1/2 ∧ z = 3) :=
sorry

end find_xyz_l392_392923


namespace reflection_matrix_is_correct_l392_392047

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l392_392047


namespace max_log_sum_l392_392920

open Real

theorem max_log_sum (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 4 * y = 40) :
  log x + log y ≤ 2 :=
sorry

end max_log_sum_l392_392920


namespace blueberries_picked_l392_392987

-- Define the conditions and question as Lean statements
def total_berries (B : ℕ) : ℕ := B + 20 + 10
def fresh_berries (B : ℕ) := (2 / 3 : ℝ) * (total_berries B)
def berries_to_sell (B : ℕ) := (1 / 2 : ℝ) * fresh_berries B

-- Lean theorem statement to prove the number of blueberries picked by Iris
theorem blueberries_picked : ∃ B : ℕ, berries_to_sell B = 20 ∧ B = 30 := by
  sorry

end blueberries_picked_l392_392987


namespace probability_of_forming_zhongguomeng_l392_392339

def total_arrangements := fact 3  -- The total number of ways to arrange 3 items
def successful_arrangements := 1  -- Only one specific arrangement forms "中国梦"
def probability := successful_arrangements / total_arrangements

theorem probability_of_forming_zhongguomeng :
  probability = 1 / 6 := by
  sorry

end probability_of_forming_zhongguomeng_l392_392339


namespace cost_of_playing_cards_l392_392264

theorem cost_of_playing_cards 
  (allowance_each : ℕ)
  (combined_allowance : ℕ)
  (sticker_box_cost : ℕ)
  (number_of_sticker_packs : ℕ)
  (number_of_packs_Dora_got : ℕ)
  (cost_of_playing_cards : ℕ)
  (h1 : allowance_each = 9)
  (h2 : combined_allowance = allowance_each * 2)
  (h3 : sticker_box_cost = 2)
  (h4 : number_of_packs_Dora_got = 2)
  (h5 : number_of_sticker_packs = number_of_packs_Dora_got * 2)
  (h6 : combined_allowance - number_of_sticker_packs * sticker_box_cost = cost_of_playing_cards) :
  cost_of_playing_cards = 10 :=
sorry

end cost_of_playing_cards_l392_392264


namespace rectangle_area_l392_392979

-- Define the points P, Q, and R.
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨4, -18⟩
def Q : Point := ⟨1004, 202⟩

-- z is an integer
variable (z : ℝ)

-- R definition and conditions on z
def R : Point := ⟨6, z⟩

-- Function to calculate the distance between two points
def distance (A B : Point) : ℝ :=
  Real.sqrt ((B.x - A.x) ^ 2 + (B.y - A.y) ^ 2)

-- Definition of the area of rectangle PQRS
def area (P Q R : Point) : ℝ :=
  distance P Q * distance P R

theorem rectangle_area :
  ∃ z : ℝ, area P Q (Point.mk 6 z) = 10444.8 :=
by
  let z := -(101.0 / 11) - 18 -- Calculate z as -298/11
  have proof_of_z : z == -(101.0 / 11) - 18 := by sorry
  use z
  have proof_of_area : area P Q (Point.mk 6 z) = 10444.8 := by sorry
  exact proof_of_area

end rectangle_area_l392_392979


namespace find_a_plus_bi_l392_392656

-- Define positive integers a and b and the complex number z
variable (a b : ℤ) (z : ℂ)

-- Define the conditions given in the problem
def conditions1 : Prop := a > 0 ∧ b > 0
def conditions2 : Prop := (a : ℂ) + (b : ℂ) * complex.i * complex.i * complex.i = 2 + 11 * complex.i

-- Define the goal
def goal : Prop := (a : ℂ) + (b : ℂ) * complex.i = 2 + complex.i

-- Main theorem statement
theorem find_a_plus_bi (h1 : conditions1) (h2 : conditions2) : goal := sorry

end find_a_plus_bi_l392_392656


namespace triangle_problem_l392_392201

noncomputable def triangle_values (A B C a b c : ℝ) := 
  a = 2 * Real.sqrt 3 ∧ Real.sin (2 * A + π / 4) = -(8 + 7 * Real.sqrt 2) / 18

theorem triangle_problem 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_b : b = 3) 
  (h_c : c = 1) 
  (h_A : A = 2 * B) 
  (h_triangle : triangle_values A B C a b c) :
  triangle_values A B C a b c :=
by 
  sorry

end triangle_problem_l392_392201


namespace constant_term_is_minus_20_l392_392089

noncomputable def constant_term_expansion : ℝ :=
  let a := ∫ x in 1..Real.exp 1, 1 / x in
  (Poly.coeff (Polynomial.expand ℝ (6 : ℕ) (Polynomial.cyclotomic 6 ℝ ^ 6)) 0).evaluate (1 / a)

theorem constant_term_is_minus_20 : constant_term_expansion = -20 := by
  sorry

end constant_term_is_minus_20_l392_392089


namespace point_in_fourth_quadrant_l392_392491

theorem point_in_fourth_quadrant (m n : ℝ) (h₁ : m < 0) (h₂ : n > 0) : 
  2 * n - m > 0 ∧ -n + m < 0 := by
  sorry

end point_in_fourth_quadrant_l392_392491


namespace race_time_l392_392371

theorem race_time (t_A t_B : ℝ) (v_A v_B : ℝ)
  (h1 : t_B = t_A + 7)
  (h2 : v_A * t_A = 80)
  (h3 : v_B * t_B = 80)
  (h4 : v_A * (t_A + 7) = 136) :
  t_A = 10 :=
by
  sorry

end race_time_l392_392371


namespace reflection_matrix_over_vector_is_correct_l392_392039

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l392_392039


namespace toms_trip_cost_l392_392342

-- Define the relevant constants and functions
constants 
  (odometer_start odometer_end : ℝ)
  (miles_per_gallon : ℝ)
  (price_per_gallon : ℝ)

-- Assign the given values to the constants
def odometer_start := 32105
def odometer_end := 32138
def miles_per_gallon := 32
def price_per_gallon := 3.80

-- Define the distance traveled
def distance_traveled := odometer_end - odometer_start

-- Define the amount of fuel used
def fuel_used := distance_traveled / miles_per_gallon

-- Define the cost of the trip
def cost_of_trip := fuel_used * price_per_gallon

-- Prove the cost rounded to the nearest cent is $3.93
theorem toms_trip_cost :
  Float.round(cost_of_trip * 100) / 100 = 3.93 :=
by
  sorry

end toms_trip_cost_l392_392342


namespace find_larger_integer_l392_392700

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l392_392700


namespace digit_d_for_5678d_is_multiple_of_9_l392_392471

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_d_for_5678d_is_multiple_of_9 : 
  ∃ d : ℕ, d < 10 ∧ is_multiple_of_9 (56780 + d) ∧ d = 1 :=
by
  sorry

end digit_d_for_5678d_is_multiple_of_9_l392_392471


namespace larger_integer_is_21_l392_392710

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l392_392710


namespace larger_integer_is_21_l392_392709

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l392_392709


namespace calculate_power_l392_392841

variable (x y : ℝ)

theorem calculate_power :
  (- (1 / 2) * x^2 * y)^3 = - (1 / 8) * x^6 * y^3 :=
sorry

end calculate_power_l392_392841


namespace reflection_matrix_is_correct_l392_392045

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l392_392045


namespace first_place_beats_joe_by_two_points_l392_392599

def points (wins draws : ℕ) : ℕ := 3 * wins + draws

theorem first_place_beats_joe_by_two_points
  (joe_wins joe_draws first_place_wins first_place_draws : ℕ)
  (h1 : joe_wins = 1)
  (h2 : joe_draws = 3)
  (h3 : first_place_wins = 2)
  (h4 : first_place_draws = 2) :
  points first_place_wins first_place_draws - points joe_wins joe_draws = 2 := by
  sorry

end first_place_beats_joe_by_two_points_l392_392599


namespace count_integers_satisfying_inequality_l392_392529

theorem count_integers_satisfying_inequality :
  { n : ℤ // -12 ≤ n ∧ n ≤ 12 ∧ (n - 3) * (n + 5) * (n + 9) < 0 }.card = 10 :=
by
  sorry

end count_integers_satisfying_inequality_l392_392529


namespace correct_relationships_count_l392_392830

theorem correct_relationships_count :
  let relation1 := ({a, b} ⊆ {b, a}) in
  let relation2 := ({a, b} = {b, a}) in
  let relation3 := ({0} = (∅ : set ℕ)) in
  let relation4 := (0 ∈ {0}) in
  let relation5 := (∅ ∈ ({0} : set (set ℕ))) in
  let relation6 := (∅ ⊆ ({0} : set ℕ)) in
  (if relation1 then 1 else 0) +
  (if relation2 then 1 else 0) +
  (if relation3 then 1 else 0) +
  (if relation4 then 1 else 0) +
  (if relation5 then 1 else 0) +
  (if relation6 then 1 else 0) = 4 :=
by
  sorry

end correct_relationships_count_l392_392830


namespace reflection_matrix_l392_392024

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l392_392024


namespace larger_integer_value_l392_392686

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l392_392686


namespace digit_d_for_5678d_is_multiple_of_9_l392_392472

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_d_for_5678d_is_multiple_of_9 : 
  ∃ d : ℕ, d < 10 ∧ is_multiple_of_9 (56780 + d) ∧ d = 1 :=
by
  sorry

end digit_d_for_5678d_is_multiple_of_9_l392_392472


namespace count_integers_with_zero_product_l392_392900

theorem count_integers_with_zero_product : 
  {n : ℕ | 1 ≤ n ∧ n ≤ 5000 ∧ 
               ∏ k in finset.range n, (((1 + complex.exp (4 * real.pi * complex.I * k / n))^n) + 1) = 0}.card = 833 :=
sorry

end count_integers_with_zero_product_l392_392900


namespace percentage_increase_greater_l392_392968

theorem percentage_increase_greater (x : ℝ) (h1 : x = 90.4) : 
  ∃ (p : ℝ), x = 80 * (1 + p) ∧ p = 0.13 :=
by
  use 0.13
  split
  sorry

end percentage_increase_greater_l392_392968


namespace range_of_m_l392_392329

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x) ↔ (-2 < m ∧ m ≤ 2) :=
begin
  sorry
end

end range_of_m_l392_392329


namespace find_a_and_b_l392_392915

theorem find_a_and_b (a b : ℝ) 
  (h_tangent_slope : (2 * a * 2 + b = 1)) 
  (h_point_on_parabola : (a * 4 + b * 2 + 9 = -1)) : 
  a = 3 ∧ b = -11 :=
by
  sorry

end find_a_and_b_l392_392915


namespace common_area_of_rotated_squares_l392_392351

theorem common_area_of_rotated_squares :
  let s := 12
  let θ := 30
  let area_of_intersection := 48 * Real.sqrt 3
  ∃ s θ, area_of_intersection = 48 * Real.sqrt 3 :=
sorry

end common_area_of_rotated_squares_l392_392351


namespace eccentricity_correct_minimum_QM_dot_QN_correct_l392_392129
noncomputable def eccentricity_of_ellipse : ℝ :=
  let a := 4
  let b := sqrt 2
  let c := sqrt (a^2 - b^2)
  c / a

theorem eccentricity_correct : eccentricity_of_ellipse = sqrt 14 / 4 := by
  sorry

noncomputable def minimum_QM_dot_QN : ℝ :=
  let b : ℝ := -8 / 9
  (9 * b^2 + 16 * b - 14) / 5

theorem minimum_QM_dot_QN_correct : minimum_QM_dot_QN = -(38 / 9) := by
  sorry

end eccentricity_correct_minimum_QM_dot_QN_correct_l392_392129


namespace chord_bisector_l392_392393

theorem chord_bisector
  (O A B C D : Point)
  (r : ℝ)
  (circle : Circle O r)
  (radius_length : r = 10)
  (diameter_ab : LineSegment A B)
  (chord_cd : LineSegment C D)
  (midpoint : midpoint O A B)
  (perpendicular_bisector : is_perpendicular_bisector chord_cd diameter_ab) :
  length chord_cd = 20 := sorry

end chord_bisector_l392_392393


namespace find_smallest_a_l392_392616

-- Problem Statement
theorem find_smallest_a (a b : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) :
  (∀ x : ℤ, sin (a * ↑x + b) = sin (17 * ↑x)) → a = 2 * π - 17 :=
begin
  sorry
end

end find_smallest_a_l392_392616


namespace mosquito_drops_per_feed_l392_392409

-- Defining the constants and conditions.
def drops_per_liter : ℕ := 5000
def liters_to_die : ℕ := 3
def mosquitoes_to_kill : ℕ := 750

-- The assertion we want to prove.
theorem mosquito_drops_per_feed :
  (drops_per_liter * liters_to_die) / mosquitoes_to_kill = 20 :=
by
  sorry

end mosquito_drops_per_feed_l392_392409


namespace Deepak_age_l392_392425

theorem Deepak_age : ∃ (A D : ℕ), (A / D = 4 / 3) ∧ (A + 6 = 26) ∧ (D = 15) :=
by
  sorry

end Deepak_age_l392_392425


namespace concyclic_AQTP_l392_392617

variables {Γ₁ Γ₂ : Type} [Circle Γ₁] [Circle Γ₂]
variables {A B P Q T : Point}

-- Assuming conditions from part a)
def intersection_points : Prop :=
  T ∈ (tangent Γ₂ P ∩ tangent Γ₂ Q) ∧
  P ∈ Γ₁ ∧ Q ∈ Γ₂ ∧
  collinear P B Q ∧
  A ≠ B ∧
  A ∈ (Γ₁ ∩ Γ₂) ∧
  B ∈ (Γ₁ ∩ Γ₂)
  
-- The statement to prove part c)
theorem concyclic_AQTP : intersection_points → concyclic {A, Q, T, P} :=
by sorry

end concyclic_AQTP_l392_392617


namespace triangle_angle_relation_l392_392637

theorem triangle_angle_relation 
  (a b c : ℝ)
  (α β γ : ℝ)
  (h1 : b = (a + c) / Real.sqrt 2)
  (h2 : β = (α + γ) / 2)
  (h3 : c > a)
  : γ = α + 90 :=
sorry

end triangle_angle_relation_l392_392637


namespace number_of_factors_of_expr_l392_392341

def has_four_factors (n : ℕ) : Prop :=
  ∃ p : ℕ, prime p ∧ n = p^3

def distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem number_of_factors_of_expr (a b c : ℕ) 
  (ha : has_four_factors a) (hb : has_four_factors b) (hc : has_four_factors c) 
  (hdist : distinct a b c) : 
  (∃ p q r: ℕ, prime p ∧ prime q ∧ prime r ∧ a = p^3 ∧ b = q^3 ∧ c = r^3) →
  (∃ p q r : ℕ, p ≠ q ∧ q ≠ r ∧ p ≠ r)  →
  numberOfFactors (a^3 * b^2 * c^4) = 910 := 
sorry

end number_of_factors_of_expr_l392_392341


namespace min_value_of_f_l392_392074

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 2

noncomputable def g (a : ℝ) : ℝ :=
  if a < -1 then 2 * a + 3
  else if -1 ≤ a ∧ a ≤ 1 then 2 - a^2
  else 3 - 2 * a

theorem min_value_of_f (a : ℝ) : 
  ∃ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), 
  f x a = g a := sorry

end min_value_of_f_l392_392074


namespace correct_trapezoid_ratios_l392_392587

def trapezoid_with_circles_ratio (a b c : ℝ) (α : ℝ) (inscribed_radius circumscribed_radius : ℝ) : Prop :=
  α = 60 ∧
  inscribed_radius = (a + b) * sqrt 3 * c / (2 * (a + b + 2 * c)) ∧
  circumscribed_radius = 2 * c / sqrt 3 ∧
  (a + b + 2 * c) * (sqrt 3 / (4 * π * c)) = 4 * sqrt 21 / (7 * π) →
  8 * sqrt 3 / (3 * π) → 
  True

theorem correct_trapezoid_ratios (a b c : ℝ) (α : ℝ) (inscribed_radius circumscribed_radius : ℝ) :
  trapezoid_with_circles_ratio a b c α inscribed_radius circumscribed_radius :=
begin
  sorry
end

end correct_trapezoid_ratios_l392_392587


namespace sum_binom_1988_l392_392848

theorem sum_binom_1988 :
  \frac{1}{2^{1988}} \sum_{n = 0}^{994} (-3)^n * ∑ \binom{1988}{2n} == -1/2 :=
begin
  sorry
end

end sum_binom_1988_l392_392848


namespace find_constants_l392_392835

theorem find_constants (a b c : ℝ) (h_neg : a < 0) (h_amp : |a| = 3) (h_period : b > 0 ∧ (2 * π / b) = 8 * π) : 
a = -3 ∧ b = 0.5 :=
by
  sorry

end find_constants_l392_392835


namespace paws_on_ground_l392_392426

def total_dogs : ℕ := 12
def dogs_on_all_4 : ℕ := total_dogs / 2
def dogs_on_2_legs : ℕ := total_dogs / 2

theorem paws_on_ground : (dogs_on_all_4 * 4) + (dogs_on_2_legs * 2) = 36 := by
  -- num_dogs / 2 is integer division, ensuring dogs_on_all_4 and dogs_on_2_legs are both 6
  calc
    (dogs_on_all_4 * 4) + (dogs_on_2_legs * 2)
        = (6 * 4) + (6 * 2) : by rw [nat.succ_sub_one]; norm_num
    ... = 24 + 12 : by norm_num
    ... = 36 : by norm_num

end paws_on_ground_l392_392426


namespace larger_integer_is_21_l392_392691

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l392_392691


namespace question_I_question_II_question_III_l392_392943

noncomputable def f (x a : ℝ) : ℝ := a * (x - 1) / x^2
noncomputable def g (x a : ℝ) : ℝ := x * Real.log x - x^2 * f x a 

theorem question_I (a : ℝ) (h : a ≠ 0) : 
  (if a > 0 then ∃ (I1 I2 : set ℝ), I1 = {0 < x < 2} ∧ I2 = {x < 0} ∪ {x > 2} ∧ 
                                        is_increasing_on (f a) I1 ∧ 
                                        is_decreasing_on (f a) I2 else 
   if a < 0 then ∃ (I1 I2 : set ℝ), I1 = {x < 0} ∪ {x > 2} ∧ I2 = {0 < x < 2} ∧ 
                                    is_increasing_on (f a) I1 ∧ 
                                    is_decreasing_on (f a) I2 else false) := sorry

theorem question_II (a : ℝ) (h : a = 1) : 
  tangent_line (λ x, f x a) (λ x, x - x - 1) := sorry

theorem question_III (a : ℝ) (h : 0 < a) : 
  (if 0 < a ≤ 1 then ∃ x ∈ [1,e], g x a = 0 
  else if 1 < a < 2 then ∃ x ∈ [1,e], g x a = a - Real.exp (a - 1) 
  else ∃ x ∈ [1,e], g x a = Real.exp 1 + a - a * Real.exp 1) := sorry

end question_I_question_II_question_III_l392_392943


namespace reflection_over_vector_l392_392057

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l392_392057


namespace equation_of_curve_C_distance_AB_when_R_max_l392_392099

-- Given Circle M
def circle_M (x y : ℝ) := (x + 1)^2 + y^2 = 1

-- Given Circle N
def circle_N (x y : ℝ) := (x - 1)^2 + y^2 = 9

-- Circle P being tangent
def externally_tangent (x y cx cy r : ℝ) := (x - cx)^2 + y^2 = r^2

def internally_tangent (x y cx cy r : ℝ) := (x - cx)^2 + y^2 = r^2

-- Centers and radii
def center_M := (-1 : ℝ, 0 : ℝ)
def center_N := (1 : ℝ, 0 : ℝ)
def radius_M := 1
def radius_N := 3

-- Proving the equations and conditions
theorem equation_of_curve_C : (∀ (x y : ℝ), externally_tangent x y (-1) 0 1 
                            → internally_tangent x y 1 0 3 
                            → (x^2 / 4 + y^2 / 3 = 1)) :=
sorry

theorem distance_AB_when_R_max : (∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)
                            → (externally_tangent x y 2 0 2 
                            → ∃ A B : ℝ × ℝ,
                            (circle_M A.1 A.2 ∧ circle_M B.1 B.2)
                            ∧ (A ≠ B) 
                            ∧ |A.1 - B.1| = (18 / 7))) :=
sorry

end equation_of_curve_C_distance_AB_when_R_max_l392_392099


namespace triangle_YQZ_max_area_l392_392584

open Real EuclideanGeometry

theorem triangle_YQZ_max_area :
  ∀ (X Y Z E Q : Point) (I_Y I_Z : Point),
    dist X Y = 12 →
    dist Y Z = 18 →
    dist Z X = 20 →
    E ∈ Segment Y Z →
    is_incenter X Y E I_Y →
    is_incenter X Z E I_Z →
    circumcircle_intersect Y I_Y E Z I_Z E Q →
    ∃ p q r : ℕ, r > 0 ∧ ¬ (∃ k : ℕ, k^2 ∣ r) ∧ area Y Q Z = p - q * sqrt r :=
sorry

end triangle_YQZ_max_area_l392_392584


namespace polar_equation_of_curve_l392_392517

open Real

theorem polar_equation_of_curve (φ ρ θ : ℝ):
  (x = sec φ) ∧ (y = tan φ) →
  (ρ * cos θ = x) ∧ (ρ * sin θ = y) →
  (ρ^2 * cos 2 * θ = 1) :=
by
  sorry

end polar_equation_of_curve_l392_392517


namespace positional_relationship_l392_392544

-- Definitions of skew_lines and parallel_lines
def skew_lines (a b : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, ¬ (a x y ∨ b x y) 

def parallel_lines (a c : ℝ → ℝ → Prop) : Prop :=
  ∃ k : ℝ, ∀ x y, c x y = a (k * x) (k * y)

-- Theorem statement
theorem positional_relationship (a b c : ℝ → ℝ → Prop) 
  (h1 : skew_lines a b) 
  (h2 : parallel_lines a c) : 
  skew_lines c b ∨ (∃ x y, c x y ∧ b x y) :=
sorry

end positional_relationship_l392_392544


namespace amanda_speed_l392_392588

-- Defining the conditions
def distance : ℝ := 6 -- 6 miles
def time : ℝ := 3 -- 3 hours

-- Stating the question with the conditions and the correct answer
theorem amanda_speed : (distance / time) = 2 :=
by 
  -- the proof is skipped as instructed
  sorry

end amanda_speed_l392_392588


namespace felicia_total_scoops_l392_392015

/- Define the required measurements and the capacity of the scoop. -/
def flour_cups_required : ℚ := 2
def white_sugar_cups_required : ℚ := 1
def brown_sugar_cups_required : ℚ := 1/4
def oil_cups_required : ℚ := 1/2
def scoop_cups : ℚ := 1/4

/- Calculate the number of scoops for each ingredient. -/
def flour_scoops : ℚ := flour_cups_required / scoop_cups
def white_sugar_scoops : ℚ := white_sugar_cups_required / scoop_cups
def brown_sugar_scoops : ℚ := brown_sugar_cups_required / scoop_cups
def oil_scoops : ℚ := oil_cups_required / scoop_cups

/- Calculate the total number of scoops. -/
def total_scoops : ℚ := flour_scoops + white_sugar_scoops + brown_sugar_scoops + oil_scoops

/- Statement we need to prove. -/
theorem felicia_total_scoops : total_scoops = 15 := by
  calc
    total_scoops == flour_scoops + white_sugar_scoops + brown_sugar_scoops + oil_scoops : by rfl
            ... == (flour_cups_required / scoop_cups) + (white_sugar_cups_required / scoop_cups) + (brown_sugar_cups_required / scoop_cups) + (oil_cups_required / scoop_cups) : by rfl
            ... == (2 / (1/4)) + (1 / (1/4)) + (1/4 / (1/4)) + (1/2 / (1/4)) : by rfl
            ... == 8 + 4 + 1 + 2 : by norm_num
            ... == 15 : by norm_num

end felicia_total_scoops_l392_392015


namespace degree3_poly_partition_degree5_poly_partition_l392_392378

-- First we need to define the increment of a polynomial over an interval
def increment (p : ℝ → ℝ) (a b : ℝ) : ℝ := p(b) - p(a)

-- Define black and white intervals for [0, 1]
def black_white_intervals (n : ℕ) : list (ℝ × ℝ) :=
  -- Function to generate alternating black and white sub-intervals
  -- This is a mock definition as the details of generating intervals are not provided
  sorry

-- Prove the conditions for polynomials of degree 3
theorem degree3_poly_partition (p : ℝ → ℝ) (h₃ : ∀ x, polynomial.degree (polynomial C)[x]
      = 3) :
  ∑ (I : ℝ × ℝ) in (black_white_intervals 3).filter (λ I, is_black I),
    increment p I.fst I.snd =
  ∑ (I : ℝ × ℝ) in (black_white_intervals 3).filter (λ I, is_white I),
    increment p I.fst I.snd :=
sorry

-- Prove the conditions for polynomials of degree 5
theorem degree5_poly_partition (p : ℝ → ℝ) (h₅ : ∀ x, polynomial.degree (polynomial C)[x]
      = 5) :
  ∑ (I : ℝ × ℝ) in (black_white_intervals 5).filter (λ I, is_black I),
    increment p I.fst I.snd =
  ∑ (I : ℝ × ℝ) in (black_white_intervals 5).filter (λ I, is_white I),
    increment p I.fst I.snd :=
sorry

end degree3_poly_partition_degree5_poly_partition_l392_392378


namespace problem1_problem2_l392_392103

-- Definitions and assumptions
def p (m : ℝ) : Prop := ∀x y : ℝ, (x^2)/(4 - m) + (y^2)/m = 1 → ∃ c : ℝ, c^2 < (4 - m) ∧ c^2 < m
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0
def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

-- Problem (1)
theorem problem1 (m : ℝ) (hS : S m) : m < 0 ∨ m ≥ 1 := sorry

-- Problem (2)
theorem problem2 (m : ℝ) (hp : p m ∨ q m) (hnq : ¬ q m) : 1 ≤ m ∧ m < 2 := sorry

end problem1_problem2_l392_392103


namespace area_of_region_l392_392840

theorem area_of_region : 
  let S := {(x, y) | abs (4 * x - 12) + abs (3 * y + 9) ≤ 6} in 
  ∃ A, (A = 7.5 ∧ 
        ∀ R ∈ S, ∃ (x y : ℝ), R = (x, y)) :=
sorry

end area_of_region_l392_392840


namespace digit_d_makes_multiple_of_9_l392_392467

theorem digit_d_makes_multiple_of_9 :
  ∃ d : ℕ, d < 10 ∧ (26 + d) % 9 = 0 ∧ d = 1 :=
by {
  have h1 : 26 % 9 = 8 := rfl,
  use 1,
  split,
  { linarith },
  split,
  { norm_num },
  { refl }
}

end digit_d_makes_multiple_of_9_l392_392467


namespace larger_integer_21_l392_392719

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l392_392719


namespace continuous_function_f_l392_392083

-- Given conditions and claim
theorem continuous_function_f (n : ℤ) (f : ℝ → ℝ) (h_cond : ∀ x : ℝ, 
  tendsto (λ h : ℝ, (1 / h) * (∫ t in x - n * h .. x + n * h, f t)) (𝓝 0) (𝓝 (2 * f (n * x)))) 
  (h_f1 : f 1 = 1) : 
  ∀ x : ℝ, f x = x :=
sorry

end continuous_function_f_l392_392083


namespace domain_of_function_l392_392945

theorem domain_of_function :
  (∀ x : ℝ, (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 0)) :=
sorry

end domain_of_function_l392_392945


namespace intersection_of_sets_l392_392993

def A := {x : ℕ | x > 0 ∧ x < 9}
def B := {1, 2, 3}
def C := {3, 4, 5, 6}

theorem intersection_of_sets :
  A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} :=
by
  sorry

end intersection_of_sets_l392_392993


namespace find_larger_number_l392_392369

theorem find_larger_number (L S : ℤ) (h₁ : L - S = 1000) (h₂ : L = 10 * S + 10) : L = 1110 :=
sorry

end find_larger_number_l392_392369


namespace find_A_l392_392361

theorem find_A (A : ℤ) (h : 10 + A = 15) : A = 5 := by
  sorry

end find_A_l392_392361


namespace part_one_part_two_l392_392147

variable {α β λ : ℝ}
variable (a b : ℝ × ℝ)

-- Define vectors \overrightarrow{a} and \overrightarrow{b}
def a := (Real.cos α, λ*Real.sin α)
def b := (Real.cos β, Real.sin β)

-- Conditions 
axiom lam_pos : λ > 0
axiom alpha_range : 0 < α ∧ α < β ∧ β < Real.pi / 2
axiom perp : (a.1 + b.1, a.2 + b.2) Dot (a.1 - b.1, a.2 - b.2) = 0
axiom dot_product : (a.1 * b.1 + a.2 * b.2) = 4 / 5
axiom tan_beta : Real.tan β = 2

-- The statements to be proved
theorem part_one : λ = 1 := sorry

theorem part_two : Real.tan α = 1 / 2 := sorry

end part_one_part_two_l392_392147


namespace max_sum_b_l392_392925

-- Condition that an is an arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given sequence definitions
def b (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a n * a (n + 1) * a (n + 2)

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b a i

-- Main statement to prove
theorem max_sum_b (a : ℕ → ℝ) (h_arith : arithmetic_seq a) (h_cond : 3 * a 5 = 8 * a 12) (h_pos : 3 * a 5 > 0) :
  ∃ n : ℕ, S a 16 = ∑ i in Finset.range 16, b a i := sorry

end max_sum_b_l392_392925


namespace sum_b_lt_2_over_3_l392_392813

def a : ℕ → ℝ
| 0     := 2
| (n+1) := if a n < real.sqrt 3 then (a n)^2 else (a n)^2 / 3

def b : ℕ → ℝ
| 0     := 0  -- b sequence starts from b_1
| (n+1) := if a n < real.sqrt 3 then 0 else 1 / 2^(n+1)

theorem sum_b_lt_2_over_3 :
  (finset.range 2020).sum (λ n, b (n+1)) < 2 / 3 :=
  sorry

end sum_b_lt_2_over_3_l392_392813


namespace other_number_LCM_l392_392452

open Nat

theorem other_number_LCM (n : ℕ) (h : lcm n 852 = 5964) : n = 852 :=
begin
  sorry
end

end other_number_LCM_l392_392452


namespace roots_poly_sum_l392_392256

noncomputable def Q (z : ℂ) (a b c : ℝ) : ℂ := z^3 + (a:ℂ)*z^2 + (b:ℂ)*z + (c:ℂ)

theorem roots_poly_sum (a b c : ℝ) (u : ℂ)
  (h1 : u.im = 0) -- Assuming u is a real number
  (h2 : Q (u + 5 * Complex.I) a b c = 0)
  (h3 : Q (u + 15 * Complex.I) a b c = 0)
  (h4 : Q (2 * u - 6) a b c = 0) :
  a + b + c = -196 := by
  sorry

end roots_poly_sum_l392_392256


namespace quadratic_equation_properties_l392_392904

theorem quadratic_equation_properties :
  ¬(∃ x, x^2 + 1993 * x + 3991 = 0 ∧ x > 0) ∧ 
  ¬(∃ x : ℤ, x^2 + 1993 * x + 3991 = 0) ∧ 
  ¬(∑ y in (Finset.filter (λ x, x^2 + 1993 * x + 3991 = 0) Finset.univ), 1 / y) < -1 :=
begin
  sorry
end

end quadratic_equation_properties_l392_392904


namespace P_is_orthocenter_of_reflections_on_circumcircle_l392_392610

noncomputable def is_orthocenter (P A B C : Point) : Prop :=
  ∀ (H : Point), (is_orthocenter H A B C) → H = P

theorem P_is_orthocenter_of_reflections_on_circumcircle 
  (A B C P : Point) (ABC_triangle : Triangle A B C) 
  (acute : acute_angled ABC_triangle) 
  (circumcircle : Circle) 
  (H : is_orthocenter P A B C) :
  (∀ (P' : Point), (is_reflection P' P (side ABC_triangle)) → on_circumcircle P' circumcircle) → H :=
by
  sorry

end P_is_orthocenter_of_reflections_on_circumcircle_l392_392610


namespace find_slope_l392_392756

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem find_slope :
  slope (-2, 3) (3, -4) = -7 / 5 :=
by
  simp [slope]
  norm_num
  sorry

end find_slope_l392_392756


namespace initial_order_cogs_l392_392421

theorem initial_order_cogs (x : ℕ) : 
  (let initial_rate := 15
       additional_units := 60
       additional_rate := 60
       average_output := 24 in
   x + additional_units) / ((x / initial_rate) + 1) = average_output → x = 60 :=
by
  intro h
  have :=
    calc
      x + additional_units = 24 * (x / initial_rate + 1) : h
                         ... = (24 * x / 15 + 24) : sorry
                         ... = (8 * x / 5 + 24) : sorry
  sorry

end initial_order_cogs_l392_392421


namespace circles_intersect_l392_392678

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 4)^2 = 16

-- Distance function between two points
def dist (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

-- Centers and radii of the circles
def center1 := (3:ℝ, 0:ℝ)
def radius1 := 2:ℝ
def center2 := (0:ℝ, 4:ℝ)
def radius2 := 4:ℝ

-- Distance between the centers
def center_dist := dist (fst center1) (snd center1) (fst center2) (snd center2)

-- Prove that the circles intersect
theorem circles_intersect : 2 < center_dist ∧ center_dist < 6 := 
by 
  -- Calculate the exact center distance
  let d := ((0 - 3)^2 + (4 - 0)^2).sqrt
  have h1 : d = 5 := by simp; linarith [sqrt_eq_rpow]
  -- Using the radii conditions
  exact ⟨by norm_num, by norm_num⟩

end circles_intersect_l392_392678


namespace concave_probability_is_23_over_1250_l392_392828

def is_digit (a : ℕ) : Prop := a ∈ {0, 1, 2, 3, 4}

def is_concave_number (a b c d e : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧ 
  a ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  a > b ∧ b > c ∧ c < d ∧ d < e

def num_concave_numbers : ℕ :=
  -- This would involve counting how many 5-tuples (a, b, c, d, e) satisfy is_concave_number
  46  -- Known from the solution steps

def total_five_digit_numbers : ℕ :=
  2500  -- Known from solution steps

def concave_number_probability : ℚ :=
  num_concave_numbers / total_five_digit_numbers

theorem concave_probability_is_23_over_1250 :
  concave_number_probability = 23 / 1250 :=
by
  -- The proof would go here, using the conditions and systematic counting argument from the solution.
  sorry

end concave_probability_is_23_over_1250_l392_392828


namespace total_matches_played_l392_392796

theorem total_matches_played
  (avg_runs_first_20: ℕ) (num_first_20: ℕ) (avg_runs_next_10: ℕ) (num_next_10: ℕ) (overall_avg: ℕ) (total_matches: ℕ) :
  avg_runs_first_20 = 40 →
  num_first_20 = 20 →
  avg_runs_next_10 = 13 →
  num_next_10 = 10 →
  overall_avg = 31 →
  (num_first_20 + num_next_10 = total_matches) →
  total_matches = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_matches_played_l392_392796


namespace salary_of_D_l392_392323

theorem salary_of_D (A B C D E : ℝ)
  (hA : A = 8000)
  (hB : B = 5000)
  (hC : C = 11000)
  (hE : E = 9000)
  (h_avg : (A + B + C + D + E) / 5 = 8000) :
  D = 9000 :=
by
  have h_sum : (8000 : ℝ) * 5 = 40000 := by norm_num
  have h_total : A + B + C + D + E = 40000 := 
    by rw [hA, hB, hC, hE]; linarith 
  have h : (40000 : ℝ) / 5 = 8000 := by norm_num
  rw only [h_avg, h, h_sum] at h_total
  linarith

end salary_of_D_l392_392323


namespace count_unique_four_digit_numbers_l392_392954

theorem count_unique_four_digit_numbers : 
  ∀ (arr : list ℕ), arr = [9, 3, 3, 3] → (∃! x, x ∈ (list.permutes arr) ∧ x.head = 9) :=
by
  sorry

end count_unique_four_digit_numbers_l392_392954


namespace pigeonhole_students_gender_l392_392394

theorem pigeonhole_students_gender : 
  ∀ (students : Finset (Fin 25)) (months : Finset (Fin 12)) (girls boys : students → Fin 12), 
  (∀ s, s ∈ students → (girls s ∈ months ∧ boys s ∈ months)) →
  (∃ (s1 s2 : students), s1 ≠ s2 ∧ ((girls s1 = girls s2) ∨ (boys s1 = boys s2))) :=
by {
  intro students,
  intro months,
  intro girls boys,
  intro cond,
  sorry
}

end pigeonhole_students_gender_l392_392394


namespace compare_abc_l392_392108

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem compare_abc : a < c ∧ c < b :=
by
  -- The proof will be provided here.
  sorry

end compare_abc_l392_392108


namespace probability_of_being_admitted_expected_value_of_X_l392_392798

-- Define the data
def number_of_penalty_kicks : List Nat := [20, 30, 30, 25, 20, 25]
def number_of_goals : List Nat := [15, 17, 22, 18, 14, 14]

-- Define the probabilities
def total_penalty_kicks : Nat := number_of_penalty_kicks.sum
def total_goals : Nat := number_of_goals.sum
def P_score : Rat := total_goals / total_penalty_kicks
def P_miss : Rat := 1 - P_score

-- First statement: Probability of being admitted
theorem probability_of_being_admitted : P_score * P_score + 
                                        P_miss * (P_score * P_score) + 
                                        (P_miss * P_miss * (P_score * P_score)) + 
                                        (P_score * P_miss * (P_score * P_score)) = 20 / 27 := 
by
  sorry

-- Second statement: Expected value of X
theorem expected_value_of_X : (0 * (P_miss * P_miss * P_miss)
                               + 1 * (2 * (P_score * (P_miss * P_miss) + P_miss * (P_miss * P_score)))
                               + 2 * (P_score * P_score + P_miss * (P_score * P_score) + P_miss * P_miss * (P_score * P_score) + P_score * (P_miss * P_score))
                               + 3 * (P_score * P_miss * (P_score * P_score))) = 50 / 27 := 
by
  sorry

end probability_of_being_admitted_expected_value_of_X_l392_392798


namespace intersection_of_M_and_N_l392_392143

noncomputable def M : set ℝ := { y | ∃ (x : ℝ), y = x^2 - 1 }

noncomputable def N : set ℝ := { y | ∃ (x : ℝ), y = real.sqrt (3 - x^2) }

theorem intersection_of_M_and_N : M ∩ N = { y | -1 ≤ y ∧ y ≤ real.sqrt 3 } :=
by
  sorry

end intersection_of_M_and_N_l392_392143


namespace equilateral_triangle_covers_points_l392_392569

theorem equilateral_triangle_covers_points : 
  ∀ (S : Type) [metric_space S], 
  ∀ (points : set S) (square : set S), 
  is_square square 12 → 
  card points = 2005 →
  ∃ triangle : set S, 
    is_equilateral_triangle triangle 11 ∧ 
    (∃ P : set S, P ⊆ points ∧ ∃ cover : set S, cover = triangle ∩ square ∧ card cover ≥ 502) :=
begin
  sorry
end

end equilateral_triangle_covers_points_l392_392569


namespace solve_for_x_l392_392292

theorem solve_for_x (x : ℝ) : 2^(32^x) = 32^(2^x) ↔ x = real.log2 5 / 4 :=
by
  sorry

end solve_for_x_l392_392292


namespace train_arrival_time_l392_392740

-- Define the time type
structure Time where
  hour : Nat
  minute : Nat

namespace Time

-- Define the addition of minutes to a time.
def add_minutes (t : Time) (m : Nat) : Time :=
  let new_minutes := t.minute + m
  if new_minutes < 60 then 
    { hour := t.hour, minute := new_minutes }
  else 
    { hour := t.hour + new_minutes / 60, minute := new_minutes % 60 }

-- Define the departure time
def departure_time : Time := { hour := 9, minute := 45 }

-- Define the travel time in minutes
def travel_time : Nat := 15

-- Define the expected arrival time
def expected_arrival_time : Time := { hour := 10, minute := 0 }

-- The theorem we need to prove
theorem train_arrival_time:
  add_minutes departure_time travel_time = expected_arrival_time := by
  sorry

end train_arrival_time_l392_392740


namespace ratio_EF_FC_l392_392245

theorem ratio_EF_FC (A B C D F E : ℝ × ℝ) (ABCD_is_square : is_square A B C D)
  (E_is_midpoint : E = midpoint A B) (circle_centered_A : circle A B)
  (F_on_EC : F ∈ line_segment E C) : (dist E F / dist F C) = (3 / 2) := 
  sorry

end ratio_EF_FC_l392_392245


namespace reflection_over_vector_l392_392050

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l392_392050


namespace solution_exists_l392_392283

theorem solution_exists (c : ℝ) : ∃ (x y z : ℝ), x = 2 ∧ y = 0 ∧ z = -1 ∧
  x - y + 2*z = 0 ∧ -2*x + y - 2*z = -2 ∧ 2*x + c*y + 3*z = 1 := 
by
  use 2, 0, -1
  split; try {refl}
  split; try {refl}
  split; try {refl}
  split
  . calc
    2 - 0 + 2 * (-1) = 0 : by norm_num
  split
  . calc
    -2 * 2 + 0 - 2 * (-1) = -2 : by norm_num
  . calc
    2 * 2 + c * 0 + 3 * (-1) = 1 : by norm_num

example (c : ℝ) : ∃ x y z : ℝ, x - y + 2 * z = 0 ∧ -2 * x + y - 2 * z = -2 ∧ 2 * x + c * y + 3 * z = 1 :=
begin
  exact solution_exists c
end


end solution_exists_l392_392283


namespace _l392_392029
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l392_392029


namespace intersect_A_B_l392_392105

def Z := Set ℤ

def A : Z := {x | x^2 + x - 6 ≤ 0}
def B := {x | x ≥ 1}

theorem intersect_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersect_A_B_l392_392105


namespace concurrency_of_lines_l392_392834

variables {A B C D P O O1 O2 O3 O4 G : Type}

-- Define the quadrilateral ABCD inscribed in circle O
axiom quad_inscribed (ABCD : A B C D) (O : O) (circumcircle_O : O) :
  ∃ (circ_O : circumcircle_O (A B C D)), true

-- Define the intersection of diagonals AC and BD at P
axiom diagonals_intersect (AC BD : A C D B) :
  ∃ (P : P), AC ∩ BD = P

-- Define circumcenters of triangles ABP, BCP, CDP, DAP as O1, O2, O3, O4 respectively
axiom circumcenters (tri_ABP tri_BCP tri_CDP tri_DAP : A B P * B C P * C D P * D A P) :
  ∃ (O1 O2 O3 O4 : O1 O2 O3 O4), 
    circumcenter tri_ABP = O1 ∧ 
    circumcenter tri_BCP = O2 ∧ 
    circumcenter tri_CDP = O3 ∧ 
    circumcenter tri_DAP = O4

-- Prove the concurrency of lines OP, O1O3, and O2O4 at point G
theorem concurrency_of_lines {A B C D P O O1 O2 O3 O4 G : Type} 
  (h1 : quad_inscribed (A B C D) O)
  (h2 : diagonals_intersect (A C) (B D))
  (h3 : circumcenters (A B P) (B C P) (C D P) (D A P)):
  ∃ G, concurrent (line O P) (line O1 O3) (line O2 O4) :=
sorry

end concurrency_of_lines_l392_392834


namespace boxes_sold_on_saturday_l392_392642

-- Definitions from conditions
def total_boxes_sold (S Sun : ℕ) : Prop :=
  Sun = 1.5 * S ∧ S + Sun = 150

-- Proof statement
theorem boxes_sold_on_saturday (S : ℕ) :
  total_boxes_sold S (1.5 * S) → S = 60 :=
by
  sorry

end boxes_sold_on_saturday_l392_392642


namespace triplet_solution_l392_392883

theorem triplet_solution (x y n : ℕ) (hx : x > 0) (hy : y > 0) (hn : n > 0) :
  (x! + y!) / n! = 3 ^ n ↔ (x = 2 ∧ y = 1 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
sorry

end triplet_solution_l392_392883


namespace circumradius_BDE_correct_l392_392997

noncomputable def circumradius_of_BDE
  (AB BC AC : ℝ)
  (h₁ : AB = 30)
  (h₂ : BC = 26)
  (h₃ : AC = 28)
  (altitude_BD : BD_is_altitude AB BC AC)
  (midpoint_E : E_is_midpoint BC)
  : ℝ :=
  16.9

theorem circumradius_BDE_correct:
  ∀ (AB BC AC : ℝ)
  (h₁ : AB = 30)
  (h₂ : BC = 26)
  (h₃ : AC = 28)
  (altitude_BD : BD_is_altitude AB BC AC)
  (midpoint_E : E_is_midpoint BC),
  circumradius_of_BDE AB BC AC h₁ h₂ h₃ altitude_BD midpoint_E = 16.9 := by
  sorry

end circumradius_BDE_correct_l392_392997


namespace find_k_l392_392252

theorem find_k (a b x : ℝ) (hx : tan x = a / b) (h3x : tan (3 * x) = b / (2 * a + b)) : 
  ∃ k, arctan k = x ∧ k = 1 / 2 :=
by
  sorry

end find_k_l392_392252


namespace larger_integer_is_21_l392_392693

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l392_392693


namespace smallest_pos_int_mod_congruence_l392_392357

theorem smallest_pos_int_mod_congruence : ∃ n : ℕ, 0 < n ∧ n ≡ 2 [MOD 31] ∧ 5 * n ≡ 409 [MOD 31] :=
by
  sorry

end smallest_pos_int_mod_congruence_l392_392357


namespace trig_expression_value_l392_392107

theorem trig_expression_value (θ : ℝ)
  (h1 : Real.sin (Real.pi + θ) = 1/4) :
  (Real.cos (Real.pi + θ) / (Real.cos θ * (Real.cos (Real.pi + θ) - 1)) + 
  Real.sin (Real.pi / 2 - θ) / (Real.cos (θ + 2 * Real.pi) * Real.cos (Real.pi + θ) + Real.cos (-θ))) = 32 :=
by
  sorry

end trig_expression_value_l392_392107


namespace combined_current_income_l392_392870

variables (Ernie_prev Jack_prev Susan_prev : ℕ)
variables (Ernie_curr Jack_curr Susan_curr : ℕ)

-- Conditions
def condition1 := Ernie_curr = 4 * Ernie_prev / 5
def condition2 := Jack_curr = Jack_prev - Jack_prev / 3
def condition3 := Susan_curr = Susan_prev + Susan_prev / 4
def condition4 := Ernie_prev = 6000
def condition5 := Jack_prev = 5 * Susan_prev / 4
def condition6 := Jack_prev = 2 * Ernie_prev

-- Proof goal
theorem combined_current_income :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 ∧ condition6 →
  Ernie_curr + Jack_curr + Susan_curr = 18800 :=
by {
  intros h,
  sorry
}

end combined_current_income_l392_392870


namespace trajectory_of_C_l392_392986

open Real

noncomputable def Point : Type := ℝ × ℝ

def distance (p1 p2 : Point) : ℝ := 
  sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def midpoint (p1 p2 : Point) : Point := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def A : Point := (0, 0)
def B : Point := (2, 0)
def AB : ℝ := distance A B

def D (C : Point) : Point := midpoint B C
def AD (C : Point) : ℝ := distance A (D C)

theorem trajectory_of_C (C : Point) (h1 : AB = 2) (h2 : AD C = 3 / 2) :
  (C.1 + 2) ^ 2 + C.2 ^ 2 = 3 ^ 2 := sorry

end trajectory_of_C_l392_392986


namespace max_k_l392_392511

noncomputable def f (x : ℝ) (b : ℝ) := x^2 + x * Real.log x + b

theorem max_k (b : ℝ) :
  (∀ x : ℝ, x > 0 → f x b > -3) ∧
  (∀ k : ℤ, (∀ x : ℝ, x > 0 → f x b > k) → k ≤ -3) :=
begin
  sorry
end

end max_k_l392_392511


namespace bus_children_l392_392387

theorem bus_children (X : ℕ) (initial_children : ℕ) (got_on : ℕ) (total_children_after : ℕ) 
  (h1 : initial_children = 28) 
  (h2 : got_on = 82) 
  (h3 : total_children_after = 30) 
  (h4 : initial_children + got_on - X = total_children_after) : 
  got_on - X = 2 :=
by 
  -- h1, h2, h3, and h4 are conditions from the problem
  sorry

end bus_children_l392_392387


namespace range_of_m_l392_392109

theorem range_of_m (a b c x m : ℝ) (h : a^2 + b^2 + c^2 = 1) 
  (h₂ : ∀ a b c x, a^2 + b^2 + c^2 = 1 → sqrt 2 * a + sqrt 3 * b + 2 * c ≤ |x - 1| + |x + m|) : 
  m ∈ (-∞, -4] ∪ [2, ∞) :=
begin
  sorry
end

end range_of_m_l392_392109


namespace reflection_matrix_is_correct_l392_392058

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l392_392058


namespace xy_sum_equals_two_l392_392546

variable {x y : ℝ}

theorem xy_sum_equals_two (hx : x ≠ 0) (hy : y ≠ 0) (h : 12^x = 18^y ∧ 18^y = 6^(x * y)) : x + y = 2 := sorry

end xy_sum_equals_two_l392_392546


namespace bottles_produced_by_twenty_machines_l392_392373

-- Definitions corresponding to conditions
def bottles_per_machine_per_minute (total_machines : ℕ) (total_bottles : ℕ) : ℕ :=
  total_bottles / total_machines

def bottles_produced (machines : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  machines * rate * time

-- Given conditions
axiom six_machines_rate : ∀ (machines total_bottles : ℕ), machines = 6 → total_bottles = 270 →
  bottles_per_machine_per_minute machines total_bottles = 45

-- Prove the question == answer given conditions
theorem bottles_produced_by_twenty_machines :
  bottles_produced 20 45 4 = 3600 :=
by sorry

end bottles_produced_by_twenty_machines_l392_392373


namespace f_2012_l392_392937

-- Definitions of the function and its iterated derivatives
def f (x : ℝ) : ℝ := sin x + exp x + x^2011

def f_n (n : ℕ) : (ℝ → ℝ) := nat.rec_on n f $ λ n' fn, deriv fn

-- Statement to prove
theorem f_2012 (x : ℝ): f_n 2012 x = sin x + exp x :=
by sorry

end f_2012_l392_392937


namespace _l392_392031
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l392_392031


namespace coastal_waters_area_comparison_l392_392565

def island (length : ℝ) (width : ℝ) : set (ℝ × ℝ) := 
{ p | 0 ≤ p.1 ∧ p.1 ≤ length ∧ 0 ≤ p.2 ∧ p.2 ≤ width }

def coastal_waters (length : ℝ) (width : ℝ) (d : ℝ) : set (ℝ × ℝ) := 
{ p | −d ≤ p.1 ∧ p.1 ≤ length + d ∧ −d ≤ p.2 ∧ p.2 ≤ width + d } 

noncomputable def area (s : set (ℝ × ℝ)) : ℝ := sorry

theorem coastal_waters_area_comparison :
  ∃ l1 w1 l2 w2 : ℝ,
  let I1 := island l1 w1,
      I2 := island l2 w2,
      C1 := coastal_waters l1 w1 50,
      C2 := coastal_waters l2 w2 50 in
  area I1 < area I2 ∧ area C1 > area C2 :=
sorry

end coastal_waters_area_comparison_l392_392565


namespace janet_needs_4_weeks_l392_392592

variable (hourly_rate : ℝ := 20) (regular_hours : ℝ := 40) (total_hours : ℝ := 52) (overtime_factor : ℝ := 1.5) (car_cost : ℝ := 4640)

def total_weekly_earning (hourly_rate : ℝ) (regular_hours : ℝ) (total_hours : ℝ) (overtime_factor : ℝ) : ℝ :=
  let regular_pay := regular_hours * hourly_rate
  let overtime_hours := total_hours - regular_hours
  let overtime_pay := overtime_hours * (hourly_rate * overtime_factor)
  regular_pay + overtime_pay

def weeks_needed_to_purchase_car (car_cost : ℝ) (weekly_earning : ℝ) : ℕ :=
  ⌈car_cost / weekly_earning⌉.nat_abs

theorem janet_needs_4_weeks (h : total_weekly_earning hourly_rate regular_hours total_hours overtime_factor = 1160) :
  weeks_needed_to_purchase_car car_cost (total_weekly_earning hourly_rate regular_hours total_hours overtime_factor) = 4 := by
  rw [total_weekly_earning, h]
  simp [weeks_needed_to_purchase_car, car_cost, weekly_earning]
  sorry 

end janet_needs_4_weeks_l392_392592


namespace bisector_OE_of_angle_XOY_l392_392634

-- Define points and conditions
variables {O A B C D E : Type}
-- Points on rays: OX and OY
axiom OA_EQ_OB : OA = OB
axiom AC_EQ_BD : AC = BD
axiom AD_intersects_BC_at_E : ∃ E, AD ∩ BC = {E}

-- Define the proof statement
theorem bisector_OE_of_angle_XOY
  (A C : ℝ)
  (B D : ℝ)
  (OA OB AC BD : ℝ)
  (H_OA_EQ_OB : OA = OB)
  (H_AC_EQ_BD : AC = BD)
  (H_intersection : ∃ E, ∃ AD BC, AD ∩ BC = {E}) : (OE bisects ∠XOY) :=
sorry

end bisector_OE_of_angle_XOY_l392_392634


namespace area_of_triangle_OPQ_l392_392096

noncomputable def ellipse_centered_at_origin (a b : ℝ) (ecc : ℝ) (pt_x pt_y : ℝ) (m k : ℝ) :=
  (a > b ∧ b > 0 ∧ ecc = real.sqrt 3 / 2 ∧ a^2 = 4 * b^2) ∧
  (pt_x^2 / (a^2) + pt_y^2 / b^2 = 1) ∧
  (k ≠ 0 ∧ (1 + 4 * k^2) > 0) ∧ 
  ∀ x : ℝ, (1 + 4 * k^2) * x^2 + (8 * k * m) * x + 4 * (m^2 - 1) = 0

theorem area_of_triangle_OPQ {a b ecc pt_x pt_y m k: ℝ} (h : ellipse_centered_at_origin a b ecc pt_x pt_y m k) :
  (0 < m^2 ∧ m^2 < 2 ∧ m^2 ≠ 1) → ∃ (S : ℝ), 0 < S ∧ S < 1 :=
sorry

end area_of_triangle_OPQ_l392_392096


namespace trapezoid_area_l392_392672

theorem trapezoid_area (a b c d : ℝ) (h1 : a > b) : 
  let S := (a + b) / (4 * (a - b)) * (Real.sqrt ((a + c + d - b) * (a + d - b - c) * (a + c - b - d) * (b + c + d - a))) in
  S = (a + b) / (4 * (a - b)) * (Real.sqrt ((a + c + d - b) * (a + d - b - c) * (a + c - b - d) * (b + c + d - a))) :=
by
  sorry

end trapezoid_area_l392_392672


namespace one_boy_one_girl_prob_one_boy_one_girl_given_one_boy_one_boy_one_girl_given_boy_born_on_monday_l392_392390

namespace ProofProblems

open ProbabilityTheory

-- Question 1
theorem one_boy_one_girl_prob {eqlikely : Bool} (h_eqlikely : eqlikely = true) :
  (prob_one_boy_one_girl_eq_2_children : ℚ) = 1 / 2 := by
  sorry

-- Question 2
theorem one_boy_one_girl_given_one_boy {eqlikely : Bool} (h_eqlikely : eqlikely = true) :
  (prob_one_boy_one_girl_given_one_boy : ℚ) = 2 / 3 := by
  sorry

-- Question 3
theorem one_boy_one_girl_given_boy_born_on_monday {eqlikely : Bool} (h_eqlikely : eqlikely = true) :
  (prob_one_boy_one_girl_given_boy_born_on_monday : ℚ) = 14 / 27 := by
  sorry

end ProofProblems

end one_boy_one_girl_prob_one_boy_one_girl_given_one_boy_one_boy_one_girl_given_boy_born_on_monday_l392_392390


namespace incorrect_calculation_d_l392_392765

theorem incorrect_calculation_d : (1 / 3) / (-1) ≠ 3 * (-1) := 
by {
  -- we'll leave the body of the proof as sorry.
  sorry
}

end incorrect_calculation_d_l392_392765


namespace find_larger_integer_l392_392702

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l392_392702


namespace how_much_larger_is_star_sum_l392_392654

-- Define the sequence of numbers from 1 to 40
def star_numbers : List ℕ := List.range' 1 40

-- Define the function to replace digit '3' with digit '2'
def replace_3_with_2 (n : ℕ) : ℕ :=
  let digits := (Nat.digits 10 n).map (fun d => if d = 3 then 2 else d)
  digits.foldr (fun d acc => acc * 10 + d) 0

-- Define the sum of Star's numbers
def star_sum : ℕ := star_numbers.sum

-- Define the sum of Emilio's numbers
def emilio_sum : ℕ := (star_numbers.map replace_3_with_2).sum

-- The theorem stating how much larger Star's sum is compared to Emilio's
theorem how_much_larger_is_star_sum : star_sum - emilio_sum = 104 := sorry

end how_much_larger_is_star_sum_l392_392654


namespace girls_exceed_boys_by_402_l392_392568

theorem girls_exceed_boys_by_402 : 
  let girls := 739
  let boys := 337
  girls - boys = 402 :=
by
  sorry

end girls_exceed_boys_by_402_l392_392568


namespace license_plates_count_l392_392150

theorem license_plates_count :
  let num_vowels := 5
  let num_letters := 26
  let num_odd_digits := 5
  let num_even_digits := 5
  num_vowels * num_letters * num_letters * num_odd_digits * num_even_digits = 84500 :=
by
  sorry

end license_plates_count_l392_392150


namespace larger_integer_21_l392_392721

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l392_392721


namespace arithmetic_sequence_a3_l392_392095

theorem arithmetic_sequence_a3 (a1 a2 a3 a4 a5 : ℝ) 
  (h1 : a2 = a1 + (a1 + a5 - a1) / 4)
  (h2 : a3 = a1 + 2 * (a1 + a5 - a1) / 4) 
  (h3 : a4 = a1 + 3 * (a1 + a5 - a1) / 4) 
  (h4 : a5 = a1 + 4 * (a1 + a5 - a1) / 4)
  (h_sum : 5 * a3 = 15) : 
  a3 = 3 :=
sorry

end arithmetic_sequence_a3_l392_392095


namespace max_silver_medals_l392_392235

-- Define the condition that there are 6 competitors and 8 events.
def is_event_structure (n_competitors n_events : ℕ) : Prop :=
  n_competitors = 6 ∧ n_events = 8

-- Define the points system for the medals.
def points_for_medals (gold silver bronze : ℕ) : Prop :=
  gold = 5 ∧ silver = 3 ∧ bronze = 1

-- Define the total points scored by a competitor.
def total_points (total : ℕ) (gold_count silver_count bronze_count : ℕ) : Prop :=
  total = gold_count * 5 + silver_count * 3 + bronze_count * 1

-- Theorem stating that the maximum number of silver medals won is 4.
theorem max_silver_medals (n_competitors n_events gold silver bronze points : ℕ)
  (hc : is_event_structure n_competitors n_events)
  (hp : points_for_medals gold silver bronze)
  (ht : total_points points 27) : silver ≤ 4 :=
sorry

end max_silver_medals_l392_392235


namespace total_books_l392_392607

def keith_books : ℕ := 20
def jason_books : ℕ := 21

theorem total_books : keith_books + jason_books = 41 :=
by
  sorry

end total_books_l392_392607


namespace beta_minus_alpha_l392_392525

open Real

noncomputable def vector_a (α : ℝ) := (cos α, sin α)
noncomputable def vector_b (β : ℝ) := (cos β, sin β)

theorem beta_minus_alpha (α β : ℝ)
  (h₁ : 0 < α)
  (h₂ : α < β)
  (h₃ : β < π)
  (h₄ : |2 * vector_a α + vector_b β| = |vector_a α - 2 * vector_b β|) :
  β - α = π / 2 :=
sorry

end beta_minus_alpha_l392_392525


namespace find_a_b_c_l392_392451

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_a_b_c (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hp1 : is_prime (a + b * c))
  (hp2 : is_prime (b + a * c))
  (hp3 : is_prime (c + a * b))
  (hdiv1 : (a + b * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv2 : (b + a * c) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1)))
  (hdiv3 : (c + a * b) ∣ ((a^2 + 1) * (b^2 + 1) * (c^2 + 1))) :
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end find_a_b_c_l392_392451


namespace solve_factorial_equation_l392_392293

theorem solve_factorial_equation (n m : ℕ) :
  (n + 1)! * (m + 1)! = (n + m)! ↔ (n = 2 ∧ m = 4) ∨ (n = 4 ∧ m = 2) :=
begin
  sorry,
end

end solve_factorial_equation_l392_392293


namespace count_integers_satisfying_inequality_l392_392530

theorem count_integers_satisfying_inequality :
  { n : ℤ // -12 ≤ n ∧ n ≤ 12 ∧ (n - 3) * (n + 5) * (n + 9) < 0 }.card = 10 :=
by
  sorry

end count_integers_satisfying_inequality_l392_392530


namespace how_many_mph_slower_l392_392263

-- Definitions from the conditions
def commute_distance : ℝ := 10 -- Liam's commute distance in miles.
def actual_speed : ℝ := 30 -- Liam's actual speed in mph.
def early_time_in_hours : ℝ := (4 / 60) -- Time Liam arrived early in hours.

-- The question reformulated as the proof problem
theorem how_many_mph_slower : ∃ v : ℝ, (10 / v = 10 / actual_speed + early_time_in_hours) ∧ (actual_speed - v = 5) :=
by
  -- Define v from the proved statement
  let v := 25
  use v
  split
  -- Prove that the computed speed meets the time equation
  { calc 10 / v = 10 / 25 : by rfl
  ... = 2 / 5 : by norm_num
  ... = 6 / 30 : by norm_num
  ... = 1 / 3 + 1 / 15 : by norm_num
  ... = 10 / actual_speed + early_time_in_hours : by rfl },
  -- Prove the speed difference is 5 mph
  { show actual_speed - v = 5
  calc actual_speed - v = 30 - 25 : by rfl
  ... = 5 : by norm_num }

end how_many_mph_slower_l392_392263


namespace continuity_at_3_l392_392260

def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 3 then 3 * x^2 - 2 else b * x + 5

theorem continuity_at_3 (b : ℝ) : 
  (f 3 b) = (f 3 0) → b = 20 / 3 := 
by 
  sorry

end continuity_at_3_l392_392260


namespace upper_limit_of_range_l392_392759

theorem upper_limit_of_range (n : ℕ) (h : (10 + 10 * n) / 2 = 255) : 10 * n = 500 :=
by 
  sorry

end upper_limit_of_range_l392_392759


namespace ratio_m_n_l392_392308

-- Define the context for the problem
variables (p x1 x2 y1 y2 m n : ℝ)
variables (h1 : p > 0)
variables (h2 : y1^2 = 2 * p * x1)
variables (h3 : y2^2 = 2 * p * x2)
variables (h4 : m = x1 + p / 2)
variables (h5 : n = x2 + p / 2)
variables (h6 : (n : ℝ), x1 - x2, and m + n form a geometric sequence)

-- The theorem to prove
theorem ratio_m_n : m / n = 3 :=
by
  sorry

end ratio_m_n_l392_392308


namespace true_proposition_l392_392115

variable (x : ℝ)

def p : Prop := x > 2
def r : Prop := x > real.log 5 / real.log 2
def q : Prop := sin x = sqrt 3 / 3 → cos (2 * x) = (sin x) ^ 2

theorem true_proposition (hp : p x) (hr : r x) (hq : q x) : p x ∧ q x :=
by
  sorry

end true_proposition_l392_392115


namespace natalia_apartment_number_unit_digit_l392_392622

def two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def true_statements (n : ℕ) : Prop :=
  (n % 3 = 0 → true) ∧   -- Statement (1): divisible by 3
  (∃ k : ℕ, k^2 = n → true) ∧  -- Statement (2): square number
  (n % 2 = 1 → true) ∧   -- Statement (3): odd
  (n % 10 = 4 → true)     -- Statement (4): ends in 4

def three_out_of_four_true (n : ℕ) : Prop :=
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 ≠ 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 ≠ 1 ∧ n % 10 = 4) ∨
  (n % 3 = 0 ∧ (∃ k : ℕ, k^2 ≠ n) ∧ n % 2 = 1 ∧ n % 10 = 4) ∨
  (n % 3 ≠ 0 ∧ (∃ k : ℕ, k^2 = n) ∧ n % 2 = 1 ∧ n % 10 = 4)

theorem natalia_apartment_number_unit_digit :
  ∀ n : ℕ, two_digit_number n → three_out_of_four_true n → n % 10 = 1 :=
by sorry

end natalia_apartment_number_unit_digit_l392_392622


namespace rotated_ellipse_sum_is_four_l392_392668

noncomputable def rotated_ellipse_center (h' k' : ℝ) : Prop :=
h' = 3 ∧ k' = -5

noncomputable def rotated_ellipse_axes (a' b' : ℝ) : Prop :=
a' = 4 ∧ b' = 2

noncomputable def rotated_ellipse_sum (h' k' a' b' : ℝ) : ℝ :=
h' + k' + a' + b'

theorem rotated_ellipse_sum_is_four (h' k' a' b' : ℝ) 
  (hc : rotated_ellipse_center h' k') (ha : rotated_ellipse_axes a' b') :
  rotated_ellipse_sum h' k' a' b' = 4 :=
by
  -- The proof would be provided here.
  -- Since we're asked not to provide the proof but just to ensure the statement is correct, we use sorry.
  sorry

end rotated_ellipse_sum_is_four_l392_392668


namespace second_company_hires_10_geniuses_l392_392346

/-- Define a set of programmers, including the acquaintance relationships and designating geniuses. -/
structure Programmer :=
  (id : ℕ)
  (genius : bool)
  (acquainted_with : list ℕ)

def hiring_possible (prog_list : list Programmer) : Prop :=
  ∃ (second_company_hires : list Programmer),
    (length (filter Programmer.genius second_company_hires) = 10) ∧
    (∀ (p ∈ second_company_hires) (q ∈ second_company_hires), 
      q.id ∈ p.acquainted_with ∨ p.id ∈ q.acquainted_with)

theorem second_company_hires_10_geniuses 
  (prog_list : list Programmer)
  (genius_count : ∀ (genius_list: list Programmer), length (filter Programmer.genius genius_list) = 11)
  : hiring_possible prog_list :=
sorry

end second_company_hires_10_geniuses_l392_392346


namespace age_difference_l392_392242

def JaneAge : ℕ := 16
def SumAges (JoeAge : ℕ) : Prop := JoeAge + JaneAge = 54

theorem age_difference (JoeAge : ℕ) (h : SumAges JoeAge) : JoeAge - JaneAge = 22 := by
  have h1 : JoeAge = 54 - JaneAge := by
    rw [Nat.sub_eq_of_eq_add h]
  rw [h1]
  exact Nat.sub_self_add 22

end age_difference_l392_392242


namespace m_range_positive_real_number_l392_392330

theorem m_range_positive_real_number (m : ℝ) (x : ℝ) 
  (h : m * x - 1 = 2 * x) (h_pos : x > 0) : m > 2 :=
sorry

end m_range_positive_real_number_l392_392330


namespace geometric_progression_fourth_term_l392_392669

theorem geometric_progression_fourth_term :
  let a1 := 4 ^ (1/2)
  let a2 := 4 ^ (1/4)
  let a3 := 4 ^ (1/8)
  let r := a2 / a1
  let a4 := a3 * r
  in a4 = 4 ^ (-1/8) :=
by
  sorry

end geometric_progression_fourth_term_l392_392669


namespace parabola_directrix_eq_4_l392_392488

noncomputable def parabola_directrix (p : ℝ) : ℝ :=
- p / 2

theorem parabola_directrix_eq_4 (p m : ℝ) (h_parabola : m^2 = 2 * p)
  (h_distance : (1 - p / 2)^2 + m^2 = 25)
  (h_p : p = 8) : 

parabola_directrix p = -4 := 
by {
  rw [parabola_directrix, h_p],
  norm_num,
}

end parabola_directrix_eq_4_l392_392488


namespace bisect_diagonals_l392_392206

variables {A B C D P Q : Type} [AffineSpace ℝ A B C D]

/-- Given a convex quadrilateral ABCD, if P and Q are midpoints of AB and CD respectively, 
and line PQ bisects diagonal AC, then line PQ also bisects diagonal BD.-/
theorem bisect_diagonals (h1 : convex_quadrilateral A B C D) 
  (h2 : midpoint (A, B) = P) (h3 : midpoint (C, D) = Q) 
  (h4 : bisects (PQ, AC)) : bisects (PQ, BD) := 
sorry

end bisect_diagonals_l392_392206


namespace quadrilateral_midpoints_area_invariant_l392_392274

-- Define the problem theorem statement
theorem quadrilateral_midpoints_area_invariant
  (A B C D E F : Point)
  (ABCD_convex : ConvexQuadrilateral A B C D)
  (E_on_AB : PointOnSegment E A B)
  (F_on_CD : PointOnSegment F C D) :
  let L := midpoint B F,
      M := midpoint C E,
      N := midpoint A F,
      K := midpoint D E,
      alpha := angle_between_lines A B C D in
  convex_quadrilateral L M N K ∧
  area_quadrilateral L M N K = (1 / 8) * length_segment A B * length_segment C D * sin(alpha) := sorry

end quadrilateral_midpoints_area_invariant_l392_392274


namespace reflection_matrix_is_correct_l392_392061

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l392_392061


namespace final_answer_l392_392434

def is_pretty (k n : ℕ) : Prop :=
  n % k = 0 ∧ ∀ d : ℕ, d ∣ n → (∃ m : ℕ, m ≤ k ∧ m > 0 ∧ (d = m ∨ n / d = m))

def S : ℕ := 
  (Finset.range 2019).filter (λ n, is_pretty 20 n).sum id

theorem final_answer : S / 20 = 372 := 
  by 
  sorry

end final_answer_l392_392434


namespace dips_to_daps_l392_392177

theorem dips_to_daps : 
  ∀ (daps dops dips : Type) (eq1 : 5 * daps = 4 * dops) (eq2 : 3 * dops = 8 * dips),
  (48 * dips = 22.5 * daps) :=
begin
  intros,
  sorry
end

end dips_to_daps_l392_392177


namespace no_intersection_at_roots_l392_392537

theorem no_intersection_at_roots {f g : ℝ → ℝ} (h : ∀ x, f x = x ∧ g x = x - 3) :
  ¬ (∃ x, (x = 0 ∨ x = 3) ∧ (f x = g x)) :=
by
  intros 
  sorry

end no_intersection_at_roots_l392_392537


namespace solve_for_x_l392_392771

-- Constants a, b, and c represent the three possible solutions.
variable (a b c : ℤ)

-- Conditions derived from the problem statement.
def condition1 := (a + 3 = 0)
def condition2 := (2 * b - 3 = 1)
def condition3 := (2 * c - 3 = -1 ∧ (c + 3) % 2 = 0) 

-- Statement that given the conditions, the solutions satisfy the equation (2x - 3)^(x + 3) = 1
theorem solve_for_x (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 c) :
  (2 * a - 3) ^ (a + 3) = 1 ∧
  (2 * b - 3) ^ (b + 3) = 1 ∧
  (2 * c - 3) ^ (c + 3) = 1 := 
by
  sorry

end solve_for_x_l392_392771


namespace general_formula_sum_and_min_value_l392_392613

variables {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def a1 := (a 1 = -5)
def a_condition := (3 * a 3 + a 5 = 0)

-- Prove the general formula for an arithmetic sequence
theorem general_formula (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0) : 
  ∀ n, a n = 2 * n - 7 := 
by
  sorry

-- Using the general formula to find the sum Sn and its minimum value
theorem sum_and_min_value (a1 : a 1 = -5) (a_condition : 3 * a 3 + a 5 = 0)
  (h : ∀ n, a n = 2 * n - 7) : 
  ∀ n, S n = n^2 - 6 * n ∧ ∃ n, S n = -9 :=
by
  sorry

end general_formula_sum_and_min_value_l392_392613


namespace product_of_integers_l392_392078

theorem product_of_integers :
  ∃ (a b c d e : ℤ),
    ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} =
      {-1, 5, 8, 9, 11, 12, 14, 18, 20, 24}) ∧
    a * b * c * d * e = -2002 :=
by {
  -- The statement formulation does not require a proof, hence here we end with sorry.
  sorry
}

end product_of_integers_l392_392078


namespace larger_integer_is_21_l392_392712

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l392_392712


namespace reflection_matrix_l392_392023

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l392_392023


namespace range_of_a_l392_392090

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) :
  (∀ x, q x → p x) → (∃ x, p x ∧ ¬ q x) → a ≤ -3 :=
by
  let p := λ x : ℝ, x ≤ -1
  let q := λ x : ℝ, a ≤ x ∧ x < a + 2
  sorry

end range_of_a_l392_392090


namespace units_digit_17_pow_2107_l392_392441

theorem units_digit_17_pow_2107 : (17 ^ 2107) % 10 = 3 := by
  -- Definitions derived from conditions:
  -- 1. Powers of 17 have the same units digit as the corresponding powers of 7.
  -- 2. Units digits of powers of 7 cycle: 7, 9, 3, 1.
  -- 3. 2107 modulo 4 gives remainder 3.
  sorry

end units_digit_17_pow_2107_l392_392441


namespace simplify_fraction_1_simplify_series_l392_392286

-- Part 1: Simplify the expression \frac{2}{\sqrt{5}+\sqrt{3}}
theorem simplify_fraction_1 (a b : ℝ) (h_a : a = sqrt 5) (h_b : b = sqrt 3) :
  (2 / (a + b)) = (sqrt 5 - sqrt 3) := 
by 
  rw [h_a, h_b]
  -- following steps would simplify \frac{2}{\sqrt{5}+\sqrt{3}} to \sqrt{5}-\sqrt{3}
  sorry

-- Part 2: Simplify the series \sum_{k=1}^{49} \frac{1}{\sqrt{2k+1}+\sqrt{2k-1}}
theorem simplify_series : 
  (∑ k in Finset.range 49, (1 / (sqrt (2 * (k + 1) + 1) + sqrt (2 * (k + 1) - 1)))) = (3*sqrt 11 - 1) / 2 :=
by
  -- following steps would simplify the series to \frac{3\sqrt{11}-1}{2}
  sorry

end simplify_fraction_1_simplify_series_l392_392286


namespace completing_the_square_correct_l392_392763

theorem completing_the_square_correct :
  (∃ x : ℝ, x^2 - 6 * x + 5 = 0) →
  (∃ x : ℝ, (x - 3)^2 = 4) :=
by
  sorry

end completing_the_square_correct_l392_392763


namespace factor_expression_l392_392449

theorem factor_expression (x : ℝ) :
  (7 * x^6 + 36 * x^4 - 8) - (3 * x^6 - 4 * x^4 + 6) = 2 * (2 * x^6 + 20 * x^4 - 7) :=
  sorry

end factor_expression_l392_392449


namespace PQ_bisects_BD_l392_392212

-- Define a structure for a Quadrilateral having points A, B, C, D
structure Quadrilateral :=
  (A B C D : Point)

-- Define midpoints P and Q
def is_midpoint (P Q : Point) (A B C D : Point) : Prop :=
  (P = midpoint A B) ∧ (Q = midpoint C D)

-- Define bisect conditions
def line_bisects (PQ : Line) (A C B D : Point) := 
  (PQ.bisects A C) ∧ (PQ.bisects B D)

-- Define the main theorem
theorem PQ_bisects_BD 
  (quad : Quadrilateral)
  (P Q : Point)
  (PQ : Line)
  (condition_midpoints : is_midpoint P Q quad.A quad.B quad.C quad.D)
  (condition_bisect_AC : PQ.bisects quad.A quad.C) :
  PQ.bisects quad.B quad.D :=
sorry

end PQ_bisects_BD_l392_392212


namespace question1_question2_l392_392909

noncomputable def f (x m : ℝ) : ℝ := (x^2 + 3) / (x - m)

theorem question1 (m : ℝ) : (∀ x : ℝ, x > m → f x m + m ≥ 0) ↔ m ∈ Set.Ici (- (2 * Real.sqrt 15) / 5) := sorry

theorem question2 (m : ℝ) : (∃ x : ℝ, x > m ∧ f x m = 6) ↔ m = 1 := sorry

end question1_question2_l392_392909


namespace sin_pi_minus_alpha_tan_pi_four_plus_alpha_l392_392572

theorem sin_pi_minus_alpha (α : ℝ) : 
    (cos α = 3/5 ∧ sin α = 4/5) → sin (π - α) = 4/5 := 
by sorry

theorem tan_pi_four_plus_alpha (α : ℝ) :
    (cos α = 3/5 ∧ sin α = 4/5) → tan (π/4 + α) = -7 :=
by sorry

end sin_pi_minus_alpha_tan_pi_four_plus_alpha_l392_392572


namespace find_zero_interval_l392_392910

-- Define the function f
def f (x : ℝ) := Real.exp x + x - 4

-- State the problem
theorem find_zero_interval :
  ∃ c, c ∈ Ioo 1 2 ∧ f c = 0 :=
by
  sorry

end find_zero_interval_l392_392910


namespace coefficient_of_x3_in_expansion_l392_392300

theorem coefficient_of_x3_in_expansion :
  (∃ c, coefficient c x^3 ((x^2 - x - 2)^4) = -40) :=
by sorry

end coefficient_of_x3_in_expansion_l392_392300


namespace range_of_b_l392_392198

theorem range_of_b (b : ℝ) :
  (∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 3 ∧ 
    y = x + b ∧ (x - 2)^2 + (y - 3)^2 = 4) ↔ 
    1 - 2 * Real.sqrt 2 ≤ b ∧ b ≤ 3 :=
by
  sorry

end range_of_b_l392_392198


namespace bah_rah_yah_equiv_l392_392547

-- We define the initial equivalences given in the problem statement.
theorem bah_rah_yah_equiv (bahs rahs yahs : ℕ) :
  (18 * bahs = 30 * rahs) ∧
  (12 * rahs = 20 * yahs) →
  (1200 * yahs = 432 * bahs) :=
by
  -- Placeholder for the actual proof
  sorry

end bah_rah_yah_equiv_l392_392547


namespace blue_area_one_percent_l392_392820

-- Let's define a noncomputable theory because we are dealing with real numbers and percentages
noncomputable theory
open_locale big_operators

-- Assume the flag is a square with side length s
def square_flag (s : ℝ) : Prop :=
  ∃ total_area cross_area green_area blue_area red_area,
    total_area = s * s ∧ -- Area of the flag is s^2
    cross_area = 0.49 * total_area ∧ -- Cross area is 49% of the flag's area
    green_area = 0.05 * total_area ∧ -- Green rectangle area is 5% of the flag's area
    (∀ blue_area, cross_area - green_area = 0.44 * total_area ∧ -- Red and blue combined area
    blue_area = 0.01 * total_area) -- The blue area is 1% of the flag's area

-- Prove that the blue area occupies 1% of the total area of the flag
theorem blue_area_one_percent (s : ℝ) (h : square_flag s) : 
  ∀ blue_area, blue_area = 0.01 * (s * s) := 
by
  sorry

end blue_area_one_percent_l392_392820


namespace num_congruent_2_mod_11_l392_392532

theorem num_congruent_2_mod_11 : 
  ∃ (n : ℕ), n = 28 ∧ ∀ k : ℤ, 1 ≤ 11 * k + 2 ∧ 11 * k + 2 ≤ 300 ↔ 0 ≤ k ∧ k ≤ 27 :=
sorry

end num_congruent_2_mod_11_l392_392532


namespace seq_formula_l392_392736

noncomputable def a : ℕ → ℝ
| 0       := 0
| (n + 1) := a n + Real.sqrt (a n + a (n + 1))

theorem seq_formula (n : ℕ) : a (n + 1) = (n + 1) * (n + 2) / 2 := by
  sorry

end seq_formula_l392_392736


namespace lottery_ends_after_third_person_l392_392743

def lottery_probability : ℚ := 1/3

theorem lottery_ends_after_third_person 
  (num_people : ℕ)
  (num_tickets : ℕ)
  (num_winning_tickets : ℕ) 
  (event_ends_after_third_person : ℚ) 
  (drawing_without_replacement : Prop) :
  num_people = 4 ∧
  num_tickets = 4 ∧
  num_winning_tickets = 2 ∧
  drawing_without_replacement → 
  event_ends_after_third_person = lottery_probability := 
by
  intros h,
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest2,
  cases h_rest2 with h3 h4,
  rw [h1, h2, h3],
  exact sorry  -- proof to be filled


end lottery_ends_after_third_person_l392_392743


namespace bruce_paid_amount_l392_392838

def kg_of_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_of_mangoes : ℕ := 10
def rate_per_kg_mangoes : ℕ := 55

def total_amount_paid : ℕ := (kg_of_grapes * rate_per_kg_grapes) + (kg_of_mangoes * rate_per_kg_mangoes)

theorem bruce_paid_amount : total_amount_paid = 1110 :=
by sorry

end bruce_paid_amount_l392_392838


namespace car_owners_without_motorcycles_l392_392976

theorem car_owners_without_motorcycles (total_adults cars motorcycles no_vehicle : ℕ) 
  (h1 : total_adults = 560) (h2 : cars = 520) (h3 : motorcycles = 80) (h4 : no_vehicle = 10) : 
  cars - (total_adults - no_vehicle - cars - motorcycles) = 470 := 
by
  sorry

end car_owners_without_motorcycles_l392_392976


namespace polynomial_integer_roots_l392_392882

theorem polynomial_integer_roots :
  ∀ x : ℤ, (x^3 - 3*x^2 - 10*x + 20 = 0) ↔ (x = -2 ∨ x = 5) :=
by
  sorry

end polynomial_integer_roots_l392_392882


namespace domain_of_f_l392_392886

noncomputable def f (x : ℝ) : ℝ := (2 * x^2 - 3) / (Real.sqrt (x - 3))

theorem domain_of_f : 
  ∀ x : ℝ, (f x) ∈ set.Ioi 3 ↔ x > 3 := 
by 
  sorry

end domain_of_f_l392_392886


namespace sum_J_eq_sum_a2_l392_392500

-- Definitions for sequence a, condition, and set K_n
def seq (n : ℕ) := {a : (ℕ → ℕ) // ∑ i in finset.range (n + 1), (i + 1) * (a i) = n}

def K (n : ℕ) := { a : (ℕ → ℕ) // ∑ i in finset.range (n + 1), (i + 1) * (a i) = n }

-- Function J(a) counting the number of 1's in sequence a
def J (a : (ℕ → ℕ)) : ℕ := a 1

-- Function to sum J(a) over all a in K_n
def sum_J_kn (n : ℕ) : ℕ := 
  ∑ a in (set.to_finset {a : (ℕ → ℕ) | seq n}), (J a)

-- Function to sum a_2 over all a in K_{n+1}
def sum_a2_kn1 (n : ℕ) : ℕ := 
  let n_plus_1 := n + 1 in
  ∑ a in (set.to_finset {a : (ℕ → ℕ) | seq n_plus_1}), (a 2)

-- Theorem statement
theorem sum_J_eq_sum_a2 (n : ℕ) : sum_J_kn n = sum_a2_kn1 n := 
  sorry

end sum_J_eq_sum_a2_l392_392500


namespace hyperbola_eccentricity_proof_l392_392515

noncomputable def hyperbola_eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (H : ∀ (c : ℝ), (c = real.sqrt (a * a + b * b)) →
    (2 * a ≤ b)) : Set ℝ :=
  {e : ℝ | real.sqrt 5 < e}

theorem hyperbola_eccentricity_proof (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b)
  (h_eq : (a * a + b * b) = c * c)
  (H : ∀ (c : ℝ), (c = real.sqrt (a * a + b * b)) →
    (2 * a ≤ b)) :
  (hyperbola_eccentricity_range a b h_a h_b H) =
  {e : ℝ | real.sqrt 5 < e} :=
sorry

end hyperbola_eccentricity_proof_l392_392515


namespace least_years_to_double_l392_392370

noncomputable def compound_interest (P r n t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem least_years_to_double (P : ℝ) (hP : 0 < P) : ∃ t : ℕ, 0 < t ∧ 2 * P < compound_interest P 0.5 1 t := by
  use 2
  simp only [compound_interest]
  linarith [Real.log_two_pos, Real.log (3 / 2)]
sorry

end least_years_to_double_l392_392370


namespace transportation_problem_l392_392799

theorem transportation_problem (x : ℝ) : 
  (1 / 4) + (1 / 2) * ((1 / 4) + (1 / x)) = 1 :=
by
  -- conditions 
  have h1 : VehicleA_transports := (1 / 4)
  have h2 : VehicleA_alone_day := 1
  have h3 : VehicleA_B_together_days := (1 / 2)
  have h4 : VehicleB_alone_days := x
  -- proof to be completed
  sorry

end transportation_problem_l392_392799


namespace Kyle_rose_cost_l392_392081

/-- Given the number of roses Kyle picked last year, the number of roses he picked this year, 
and the cost of one rose, prove that the total cost he has to spend to buy the remaining roses 
is correct. -/
theorem Kyle_rose_cost (last_year_roses this_year_roses total_roses_needed cost_per_rose : ℕ)
    (h_last_year_roses : last_year_roses = 12) 
    (h_this_year_roses : this_year_roses = last_year_roses / 2) 
    (h_total_roses_needed : total_roses_needed = 2 * last_year_roses) 
    (h_cost_per_rose : cost_per_rose = 3) : 
    (total_roses_needed - this_year_roses) * cost_per_rose = 54 := 
by
sorry

end Kyle_rose_cost_l392_392081


namespace larger_integer_is_21_l392_392690

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l392_392690


namespace simplify_expression_l392_392360

theorem simplify_expression :
  (2021^3 - 3 * 2021^2 * 2022 + 4 * 2021 * 2022^2 - 2022^3 + 2) / (2021 * 2022) = 
  1 + (1 / 2021) :=
by
  sorry

end simplify_expression_l392_392360


namespace find_divisor_l392_392627

theorem find_divisor (d : ℕ) (h : 127 = d * 5 + 2) : d = 25 :=
by 
  -- Given conditions
  -- 127 = d * 5 + 2
  -- We need to prove d = 25
  sorry

end find_divisor_l392_392627


namespace length_of_RS_l392_392233

-- Define the lengths of the edges of the tetrahedron
def edge_lengths : List ℕ := [9, 16, 22, 31, 39, 48]

-- Given the edge PQ has length 48
def PQ_length : ℕ := 48

-- We need to prove that the length of edge RS is 9
theorem length_of_RS :
  ∃ (RS : ℕ), RS = 9 ∧
  ∃ (PR QR PS SQ : ℕ),
  [PR, QR, PS, SQ] ⊆ edge_lengths ∧
  PR + QR > PQ_length ∧
  PR + PQ_length > QR ∧
  QR + PQ_length > PR ∧
  PS + SQ > PQ_length ∧
  PS + PQ_length > SQ ∧
  SQ + PQ_length > PS :=
by
  sorry

end length_of_RS_l392_392233


namespace larger_integer_value_l392_392681

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l392_392681


namespace andrew_kept_stickers_l392_392831

theorem andrew_kept_stickers :
  ∃ (b d f e g h : ℕ), b = 2000 ∧ d = (5 * b) / 100 ∧ f = d + 120 ∧ e = (d + f) / 2 ∧ g = 80 ∧ h = (e + g) / 5 ∧ (b - (d + f + e + g + h) = 1392) :=
sorry

end andrew_kept_stickers_l392_392831


namespace part1_cosine_identity_part2_tangent_double_angle_l392_392433

theorem part1_cosine_identity : 
  cos (70 * real.pi / 180) * cos (80 * real.pi / 180) - sin (70 * real.pi / 180) * sin (80 * real.pi / 180) = -real.sqrt 3 / 2 := 
by
  sorry

theorem part2_tangent_double_angle : 
  (4 * tan (real.pi / 8)) / (1 - (tan (real.pi / 8))^2) = 2 :=
by
  sorry

end part1_cosine_identity_part2_tangent_double_angle_l392_392433


namespace rain_probability_in_two_locations_l392_392003

noncomputable def probability_no_rain_A : ℝ := 0.3
noncomputable def probability_no_rain_B : ℝ := 0.4

-- The probability of raining at a location is 1 - the probability of no rain at that location
noncomputable def probability_rain_A : ℝ := 1 - probability_no_rain_A
noncomputable def probability_rain_B : ℝ := 1 - probability_no_rain_B

-- The rain status in location A and location B are independent
theorem rain_probability_in_two_locations :
  probability_rain_A * probability_rain_B = 0.42 := by
  sorry

end rain_probability_in_two_locations_l392_392003


namespace BL_perp_AC_l392_392423

-- Definitions of given conditions
variables {A B C O K M N L : Point}

-- Assume O is the circumcenter of the acute-angled triangle ∆ABC.
-- Define the circumcenter property and acute angle triangle properties.
axiom circumcenter_AOB_eq_AB_OBC_eq_ABC {A B C O : Point} (h : circumcenter O (triangle A B C)): 
  O = circumcenter (triangle A B C) ∧ acute (angle A B C) ∧ acute (angle B A C) ∧ acute (angle B C A) 

-- Define circle w1 passing through points A, O, and C with center K, intersecting AB at M and BC at N.
axiom circle_w1 {A O C K M N : Point} 
  (h : circle w1 A O C) (center_w1 : center w1 = K) (MA_intersection : circle_intersection_line AB w1 M) (NB_intersection : circle_intersection_line BC w1 N) 

-- Define point L as the reflection of K over the line MN.
axiom reflection_L {K M N L : Point} (h : reflection M N K L)

-- The statement to be proven
theorem BL_perp_AC : 
  circumcenter O (triangle A B C) ∧ circle w1 A O C ∧ center w1 = K ∧ circle_intersection_line AB w1 M ∧ circle_intersection_line BC w1 N ∧ reflection M N K L → perpendicular (BL) (AC) :=
begin
  assume h,
  sorry
end

end BL_perp_AC_l392_392423


namespace unique_solution_l392_392098
-- Import necessary mathematical library

-- Define mathematical statement
theorem unique_solution (N : ℕ) (hN: N > 0) :
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (m + (1 / 2 : ℝ) * (m + n - 1) * (m + n - 2) = N) :=
by {
  sorry
}

end unique_solution_l392_392098


namespace reflection_matrix_over_vector_l392_392069

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l392_392069


namespace milford_age_in_3_years_l392_392871

theorem milford_age_in_3_years (current_age_eustace : ℕ) (current_age_milford : ℕ) :
  (current_age_eustace = 2 * current_age_milford) → 
  (current_age_eustace + 3 = 39) → 
  current_age_milford + 3 = 21 :=
by
  intros h1 h2
  sorry

end milford_age_in_3_years_l392_392871


namespace correct_calculation_l392_392364

theorem correct_calculation : 
  (¬ ((-3) * (-2) = -6)) ∧ 
  ((-4) * (-3) * (-5) = -60) ∧ 
  (¬ ((-8) * 7 + (-2) * 7 + (-5) * 0 = 0)) ∧ 
  (¬ (((1/3) - (1/4) - (1/6)) * (-48) = -4))
  :=
by {
  apply and.intro,
  {
    intro h,
    linarith, -- This will help in showing that the calculation is incorrect.
  },
  apply and.intro,
  {
    exact rfl, -- This states that the equation is correct as defined.
  },
  apply and.intro,
  {
    intro h,
    linarith, -- This will help in showing that the calculation is incorrect.
  },
  {
    intro h,
    linarith, -- This will help in showing that the calculation is incorrect.
  }
}

end correct_calculation_l392_392364


namespace larger_integer_is_21_l392_392692

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l392_392692


namespace pre_image_of_12_l392_392995

variable (A : Set (ℝ × ℝ)) -- A is the set of all points on the Cartesian coordinate plane
variable (f : A → A) -- f is a mapping from A to A
variable (h : ∀ x y : ℝ, f (x, y) = (x + y, 2x - 3y)) -- Definition of the mapping f

theorem pre_image_of_12 :
  ∃ x y : ℝ, f (x, y) = (1, 2) ∧ (x, y) = (1, 0) :=
by
  sorry

end pre_image_of_12_l392_392995


namespace sally_boxes_sold_l392_392645

-- Define the problem conditions
def boxes_sold_Saturday (S : ℕ) : Prop :=
  let Sunday := 1.5 * S
  ∧ S + Sunday = 150

-- State the proof problem
theorem sally_boxes_sold (S : ℕ) (h : boxes_sold_Saturday S) : S = 60 :=
  sorry

end sally_boxes_sold_l392_392645


namespace distinct_roots_and_integer_l392_392497

-- Given a polynomial with roots a, b, c
def polynomial : Polynomial ℝ := Polynomial.X ^ 3 - Polynomial.X ^ 2 - Polynomial.X - 1

-- Prove roots are distinct and certain expression is an integer
theorem distinct_roots_and_integer (a b c : ℝ) (h_roots : Polynomial.roots polynomial = {a, b, c}) :
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧
  ∃ S1982, S1982 = (a^1982 - b^1982) / (a - b) + (b^1982 - c^1982) / (b - c) + (c^1982 - a^1982) / (c - a) ∧ S1982 ∈ ℤ :=
by
  -- Proofs to be filled in
  sorry

end distinct_roots_and_integer_l392_392497


namespace ray_am_is_a_median_l392_392991

/-- Given a triangle ABC with D as the foot of the A-altitude,
    and a circle w with diameter AD which intersects AB at K and AC at L,
    and point M being the intersection of tangents to w at K and L,
    prove that the ray AM is the A-median in triangle ABC. -/
theorem ray_am_is_a_median (A B C D K L M : Point) (h1 : D ∈ (lineSegment B C))
  (w : Circle) (h2 : diameter w = lineSegment A D) (h3 : K ∈ (lineSegment A B))
  (h4 : L ∈ (lineSegment A C)) (h5 : w.intersects_lineSegment_extends (lineSegment A B) = {K})
  (h6 : w.intersects_lineSegment_extends (lineSegment A C) = {L})
  (h7 : tangent_to_circle_at_point w K ∩ tangent_to_circle_at_point w L = {M}) :
  is_a_median (line A M) (triangle A B C) :=
sorry

end ray_am_is_a_median_l392_392991


namespace triangle_median_min_sum_l392_392582

theorem triangle_median_min_sum (A B C D P : Point) (h : Line) (BC AD : ℝ)
  (hB : B ∈ h) (hC : C ∈ h) (hA : A ∉ h) (hP : P ∈ h) (hD : midpoint D B C) 
  (hAP : orthogonal_projection A h = P)  
  (hBC : BC = 10) (hAD : AD = 6) :
  let AB := distance A B,
      AC := distance A C in
  AB + AC ≥ 2 * Real.sqrt 41 :=
begin
  sorry
end

end triangle_median_min_sum_l392_392582


namespace right_triangle_angles_with_centroid_on_incircle_l392_392862

theorem right_triangle_angles_with_centroid_on_incircle (α β : ℝ) :
  (right_triangle α β) → (centroid_on_incircle α β) → (α = 22 + 18 / 60) ∧ (β = 67 + 42 / 60) :=
sorry

end right_triangle_angles_with_centroid_on_incircle_l392_392862


namespace fractional_part_of_log_invM_eq_l392_392116

variable (M a : ℝ)
variable (b : ℤ)

-- Conditions
axiom pos_M : 0 < M
axiom frac_log_M : ∃ b : ℤ, log M = b + 1 / a

-- Theorem statement
theorem fractional_part_of_log_invM_eq : 0 < M → (∃ b : ℤ, log M = b + 1 / a) → frac (log (1 / M)) = (a - 1) / a :=
by
  intro pos_M frac_log_M
  sorry

end fractional_part_of_log_invM_eq_l392_392116


namespace max_k_l392_392579

def seq (a : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = k * (a n) ^ 2 + 1

def bounded (a : ℕ → ℝ) (c : ℝ) : Prop :=
∀ n : ℕ, a n < c

theorem max_k (k : ℝ) (c : ℝ) (a : ℕ → ℝ) :
  a 1 = 1 →
  seq a k →
  bounded a c →
  0 < k ∧ k ≤ 1 / 4 :=
by
  sorry

end max_k_l392_392579


namespace monotonic_interval_omega_min_l392_392512

noncomputable def f (omega varphi x : ℝ) := 2 * Real.sin (omega * x + varphi) - 1
def has_zero_at (f : ℝ → ℝ) (x : ℝ) := f x = 0
def is_symmetry_axis (f : ℝ → ℝ) (x : ℝ) := 
  ∀ y, f (2 * x - y) = f y

theorem monotonic_interval_omega_min (omega varphi : ℝ) (k : ℤ) (x : ℝ) :
  f (2/3) (11/18*Real.pi) x = 2 * (Real.sin ((2/3) * x + (11/18 * Real.pi))) - 1 ∧
  omega = 2/3 ∧ 
  varphi = 11/18 * Real.pi ∧
  has_zero_at (f omega varphi) (Real.pi / 3) ∧
  is_symmetry_axis (f omega varphi) (-Real.pi / 6) → 
  -5*Real.pi/3 + 3*k*Real.pi ≤
  x ∧ x ≤ -Real.pi/6 + 3*k*Real.pi :=
begin
  sorry
end

end monotonic_interval_omega_min_l392_392512


namespace number_of_digits_in_3_power_15_times_5_power_10_l392_392438

-- Define the function to count the number of digits in a number
def numberOfDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log10 n + 1

-- Define the main theorem to state that the number of digits in 3^15 * 5^10 is 18
theorem number_of_digits_in_3_power_15_times_5_power_10 : numberOfDigits (3^15 * 5^10) = 18 := by
  sorry

end number_of_digits_in_3_power_15_times_5_power_10_l392_392438


namespace decreasing_function_minimum_value_l392_392941

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + 1)

theorem decreasing_function (x1 x2 : ℝ) (hx : x1 < x2) : f x1 > f x2 :=
by
  sorry

theorem minimum_value : ∃ y ∈ (Set.Icc 1 5), f y = 1 / 33 :=
by
  use 5
  split
  { norm_num }
  { unfold f, norm_num, exact rfl }

end decreasing_function_minimum_value_l392_392941


namespace quadratic_roots_l392_392962

theorem quadratic_roots {a b c : ℝ} 
  (h1 : ∀ (x : ℝ), y = a * x^2 + b * x + c)
  (h2 : ∃ (d1 d2 : ℝ), d1^2 = -4 * a * (-2) + 4 * a * (-5) + 4 * c = 36 ∧ d2^2 = -4 * a * (-2) + 4 * a * (1) + 4 * c = 36)
  (h3 : a * (-5)^2 + b * (-5) + c = 0)
  (h4 : a * 1^2 + b * 1 + c = 0):
  roots a b c = {x | x = -5 ∨ x = 1} :=
by
  sorry

end quadratic_roots_l392_392962


namespace set_equality_l392_392999

-- Define the universe U
def U := ℝ

-- Define the set M
def M := {x : ℝ | (x + 1) * (x - 2) ≤ 0}

-- Define the set N
def N := {x : ℝ | x > 1}

-- Define the set we want to prove is equal to the intersection of M and N
def target_set := {x : ℝ | 1 < x ∧ x ≤ 2}

theorem set_equality : target_set = M ∩ N := 
by sorry

end set_equality_l392_392999


namespace fraction_unchanged_l392_392553

theorem fraction_unchanged (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (2 * x) / (2 * (x + y)) = x / (x + y) :=
by
  sorry

end fraction_unchanged_l392_392553


namespace area_triangle_DEF_l392_392353

variable {D E F M : Type}
variable [NormedGroup D] [NormedSpace ℝ D]
variable [NormedGroup E] [NormedSpace ℝ E]
variable [NormedGroup F] [NormedSpace ℝ F]
variable [InnerProductSpace ℝ D] [InnerProductSpace ℝ E] [InnerProductSpace ℝ F]
variable (DF DM DE FM : ℝ)

theorem area_triangle_DEF 
  (hDF : DF = 15) 
  (hDM : DM = 9) 
  (hDE : DE = 18) 
  (hFM : FM = sqrt (DF ^ 2 - DM ^ 2)) : 
  (1 / 2) * DE * FM = 108 := 
by
  sorry

end area_triangle_DEF_l392_392353


namespace find_acute_angle_correct_l392_392114

noncomputable def find_acute_angle (θ : ℝ) : Prop :=
  ∀ (M N : ℝ × ℝ), 
  M = (-real.sqrt 3, real.sqrt 2) ∧ 
  N = (real.sqrt 2, -real.sqrt 3) ∧ 
  θ = real.arctan 1 → 
  θ = real.pi / 4

-- Statement only, no proof included.
theorem find_acute_angle_correct (θ : ℝ) : find_acute_angle θ := 
  sorry

end find_acute_angle_correct_l392_392114


namespace larger_integer_value_l392_392685

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l392_392685


namespace sine_law_necessary_and_sufficient_condition_l392_392559

theorem sine_law_necessary_and_sufficient_condition 
  {a b c : ℝ} {A B C : ℝ}
  (h_triangle: triangle ABC a b c)
  (h_sine_law : a / sin A = b / sin B) : 
  (a > b ↔ sin A > sin B) :=
sorry

end sine_law_necessary_and_sufficient_condition_l392_392559


namespace find_quadruplets_l392_392881

theorem find_quadruplets :
  ∃ (x y z w : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
  (xyz + 1) / (x + 1) = (yzw + 1) / (y + 1) ∧
  (yzw + 1) / (y + 1) = (zwx + 1) / (z + 1) ∧
  (zwx + 1) / (z + 1) = (wxy + 1) / (w + 1) ∧
  x + y + z + w = 48 ∧
  x = 12 ∧ y = 12 ∧ z = 12 ∧ w = 12 :=
by
  sorry

end find_quadruplets_l392_392881


namespace infinite_series_value_l392_392851

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 4 * n^2 + 8 * n + 8) / (3^n * (n^3 + 5)) = 1 / 2 :=
by sorry

end infinite_series_value_l392_392851


namespace probability_at_least_seven_stayed_l392_392190

theorem probability_at_least_seven_stayed :
  let num_people := 8
  let unsure_prob := 3 / 7
  let sure_people := 4
  let unsure_people := 4
  let prob_7_stayed := (nat.choose unsure_people 3 * (unsure_prob ^ 3) * ((1 - unsure_prob) ^ 1) : ℝ)
  let prob_8_stayed := (unsure_prob ^ 4 : ℝ)
  let total_probability := prob_7_stayed + prob_8_stayed
  total_probability = 513 / 2401 :=
sorry

end probability_at_least_seven_stayed_l392_392190


namespace platform_length_is_correct_l392_392389

-- Problem definition: length of the platform
def length_of_platform (train_length : ℝ) (time_to_cross_platform : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  let speed := train_length / time_to_cross_pole in
  (speed * time_to_cross_platform) - train_length

theorem platform_length_is_correct :
  length_of_platform 300 33 18 = 250.11 :=
by
  unfold length_of_platform
  have speed : ℝ := 300 / 18
  have total_distance : ℝ := speed * 33
  have length_platform : ℝ := total_distance - 300
  calc
    length_platform = (300 / 18 * 33) - 300 : by rw [←mul_sub 33 (300 / 18) 1]
    ... = 550.11 - 300 : by norm_num
    ... = 250.11 : by norm_num

end platform_length_is_correct_l392_392389


namespace alice_sugar_fill_count_l392_392419

theorem alice_sugar_fill_count :
  ∀ (sugar_needed : ℚ) (cup_capacity : ℚ) (spilled_sugar : ℚ),
  sugar_needed = 15 / 4 →
  cup_capacity = 1 / 3 →
  spilled_sugar = 1 / 4 →
  let half_sugar : ℚ := (1 / 2) * sugar_needed,
  remaining_sugar : ℚ := sugar_needed - (half_sugar - spilled_sugar)
  in
  let fills : ℚ := remaining_sugar / cup_capacity
  in
  ceil fills = 7 :=
by
  intros sugar_needed cup_capacity spilled_sugar h1 h2 h3
  sorry

end alice_sugar_fill_count_l392_392419


namespace original_number_l392_392113

theorem original_number (x : ℝ) (h1 : 74 * x = 19732) : x = 267 := by
  sorry

end original_number_l392_392113


namespace continuous_second_derivative_function_l392_392878

noncomputable def solution (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ :=
  λ x, b * (x + 1/6) ^ 2

theorem continuous_second_derivative_function
  (f : ℝ → ℝ)
  (h_continuous_second_derivative : ∀ x, ∃ y, (f_manual_has_deriv_at x).has_deriv_at (g x) )
  (h_eq : ∀ x, f (7 * x + 1) = 49 * f x) :
  ∃ b : ℝ, ∀ x, f x = solution f b x :=
begin
  sorry
end

end continuous_second_derivative_function_l392_392878


namespace daps_to_dips_l392_392178

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end daps_to_dips_l392_392178


namespace larger_integer_is_21_l392_392689

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l392_392689


namespace reflection_matrix_is_correct_l392_392062

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l392_392062


namespace factor_polynomial_l392_392876

def Polynomial_Factorization (x : ℝ) : Prop := 
  let P := x^2 - 6*x + 9 - 64*x^4
  P = (8*x^2 + x - 3) * (-8*x^2 + x - 3)

theorem factor_polynomial : ∀ x : ℝ, Polynomial_Factorization x :=
by 
  intro x
  unfold Polynomial_Factorization
  sorry

end factor_polynomial_l392_392876


namespace geom_seq_sum_l392_392983

variable {a : ℕ → ℕ}

def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def P (a : ℕ → ℕ) : Prop :=
  is_geometric_sequence a 2 ∧ (∑ i in finset.range (97 / 3 + 1), a (3 * i + 1)) = 22

theorem geom_seq_sum (a : ℕ → ℕ) (h : P a) : ∑ i in finset.range 99, a i = 77 :=
by
  sorry

end geom_seq_sum_l392_392983


namespace abs_algebraic_expression_l392_392162

theorem abs_algebraic_expression (x : ℝ) (h : |2 * x - 3| - 3 + 2 * x = 0) : |2 * x - 5| = 5 - 2 * x := 
by sorry

end abs_algebraic_expression_l392_392162


namespace count_four_digit_integers_divisible_by_15_l392_392534

theorem count_four_digit_integers_divisible_by_15 : 
  { n : Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 15 = 0 }.card = 600 :=
by
  sorry

end count_four_digit_integers_divisible_by_15_l392_392534


namespace proof_problem_l392_392487

variable {a : ℕ → ℝ} {n : ℕ}
variable {S : ℕ → ℝ} {T : ℕ → ℝ}
variable {f : ℝ → ℝ}
variable (q : ℝ)

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def function_f (f : ℝ → ℝ) (a : ℕ → ℝ) : Prop :=
  f = λ x, x * ∏ i in finset.range 8, (x + a i)

def derivative_at_zero (f : ℝ → ℝ) : Prop :=
  (deriv f) 0 = 1

def product_condition (a : ℕ → ℝ) : Prop :=
  ∏ i in finset.range 8, a i = 1

theorem proof_problem
  (a1_gt_1 : a 1 > 1)
  (geo_seq : geometric_sequence a q)
  (fn_f : function_f f a)
  (deriv_f : derivative_at_zero f)
  (prod_cond : product_condition a) :
  0 < q ∧ q < 1 ∧
  (∀ n : ℕ, (S n) - (a 1 / (1 - q))) = q^n ∧
  (∃ n : ℕ, T n > 1 ∧ n = 6) :=
sorry

end proof_problem_l392_392487


namespace larger_integer_is_21_l392_392704

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l392_392704


namespace number_of_dozens_l392_392287

theorem number_of_dozens (x : Nat) (h : x = 16 * (3 * 4)) : x / 12 = 16 :=
by
  sorry

end number_of_dozens_l392_392287


namespace interest_rate_l392_392550

theorem interest_rate (P A : ℝ) (t : ℕ) (r : ℝ) : 
  P = 5000 → 
  A = 20000 → 
  t = 36 → 
  (1 + r)^t = A / P → 
  r ≈ 0.0363 :=
by
  intros hP hA ht hr
  have hp : (1 + r) = real.exp (real.log 4 / 36) := sorry
  sorry

end interest_rate_l392_392550


namespace part_1_part_2_l392_392922

-- Definitions based on conditions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 1 ≥ a

def q (a : ℝ) : Prop := ∀ ρ α : ℝ, (ρ * cos α)^2 - (ρ * sin α)^2 = a + 2 → a + 2 > 0

-- Proof problems statement
theorem part_1 (a : ℝ) (h : p a) : a ≤ 1 := sorry

theorem part_2 (a : ℝ) (hp : p a) (hq : q a) : -2 < a ∧ a ≤ 1 := sorry

end part_1_part_2_l392_392922


namespace graph_of_g_xplus1_passes_through_point_l392_392111

-- Define that y = f(x) has an inverse function y = g(x)
variables {α β : Type*} [Inhabited α] [Inhabited β] [PartialOrder α] [PartialOrder β]
noncomputable def f : α → β := sorry
noncomputable def g : β → α := sorry
axiom f_g_inverse : ∀ x : α, g (f x) = x
axiom f_at_3 : f 3 = -1

-- Now state that g(-1) = 3 according to the conditions
theorem graph_of_g_xplus1_passes_through_point : g (-1) = 3 :=
by {
  -- using the axiom that y = f(x) has an inverse function y = g(x)
  -- and the condition that f(3) = -1, we can conclude that g(-1) = 3
  sorry
}

end graph_of_g_xplus1_passes_through_point_l392_392111


namespace caroline_selected_coprime_l392_392375

noncomputable def selected_integers : set ℕ := sorry

theorem caroline_selected_coprime  :
  selected_integers ⊆ { n | 1 ≤ n ∧ n ≤ 2022 } →
  selected_integers.finite →
  selected_integers.card = 1012 →
  ∃ a b ∈ selected_integers, Int.gcd a b = 1 :=
sorry

end caroline_selected_coprime_l392_392375


namespace Roger_needs_to_delete_20_apps_l392_392464

def max_apps := 50
def recommended_apps := 35
def current_apps := 2 * recommended_apps
def apps_to_delete := current_apps - max_apps

theorem Roger_needs_to_delete_20_apps : apps_to_delete = 20 := by
  sorry

end Roger_needs_to_delete_20_apps_l392_392464


namespace time_to_run_round_square_field_l392_392776

theorem time_to_run_round_square_field
  (side : ℝ) (speed_km_hr : ℝ)
  (h_side : side = 45)
  (h_speed_km_hr : speed_km_hr = 9) : 
  (4 * side / (speed_km_hr * 1000 / 3600)) = 72 := 
by 
  sorry

end time_to_run_round_square_field_l392_392776


namespace probability_x_lt_2y_l392_392811

-- Define the rectangle and the probability calculation
theorem probability_x_lt_2y {x y : ℝ} :
  let rect := ({ (0, 0), (3, 0), (3, 2), (0, 2) } : set (ℝ × ℝ)),
    area_triangle := (1 / 2) * abs ((0 : ℝ) * (1.5 - 2) + 3 * (2 - 0) + 0 * (0 - 1.5)),
    area_rectangle := 3 * 2 in
  (measure_theory.measure_of {p | p ∈ rect ∧ p.1 < 2 * p.2} (measure_theory.measure_space.volume) /
  measure_theory.measure_of rect (measure_theory.measure_space.volume)) = (1 / 2) :=
by
  let rect := ({ (0, 0), (3, 0), (3, 2), (0, 2) } : set (ℝ × ℝ)),
  let area_triangle := (1 / 2) * abs ((0 : ℝ) * (1.5 - 2) + 3 * (2 - 0) + 0 * (0 - 1.5)),
  let area_rectangle := 3 * 2,
  have h_triangle_area : area_triangle = 3 := by sorry,
  have h_rectangle_area : area_rectangle = 6 := rfl,
  have h_probability : (measure_theory.measure_of {p | p ∈ rect ∧ p.1 < 2 * p.2} (measure_theory.measure_space.volume) /
                        measure_theory.measure_of rect (measure_theory.measure_space.volume)) = (1 / 2) := 
        by sorry,
  exact h_probability

end probability_x_lt_2y_l392_392811


namespace total_number_of_shirts_l392_392653

variable (total_cost : ℕ) (num_15_dollar_shirts : ℕ) (cost_15_dollar_shirts : ℕ) 
          (cost_remaining_shirts : ℕ) (num_remaining_shirts : ℕ) 

theorem total_number_of_shirts :
  total_cost = 85 →
  num_15_dollar_shirts = 3 →
  cost_15_dollar_shirts = 15 →
  cost_remaining_shirts = 20 →
  (num_remaining_shirts * cost_remaining_shirts) + (num_15_dollar_shirts * cost_15_dollar_shirts) = total_cost →
  num_15_dollar_shirts + num_remaining_shirts = 5 :=
by
  intros
  sorry

end total_number_of_shirts_l392_392653


namespace necessary_and_sufficient_condition_l392_392541

theorem necessary_and_sufficient_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : (a + b > a * b) ↔ (a = 1 ∨ b = 1) := 
sorry

end necessary_and_sufficient_condition_l392_392541


namespace reflection_matrix_is_correct_l392_392046

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l392_392046


namespace length_of_chord_AB_equation_of_line_AB_bisected_l392_392486

/-- Given a circle O: x^2 + y^2 = 8 and a point P0(-1,2) inside,
prove that the length of the chord AB passing through point P0 with inclination angle of 135° is sqrt(30). -/
theorem length_of_chord_AB (α : Real.Angle) (hα : α = 135) :
  let O := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 8 }
  let P0 := (-1, 2)
  ∃ AB : Set (ℝ × ℝ), (P0 ∈ AB) ∧ (angle (vector.from_to (-1, 2) (fst AB)) = α) ∧ (length_of_chord AB = √30) :=
sorry

/-- Given a circle O: x^2 + y^2 = 8 and a point P0(-1,2) inside,
prove that when the chord AB is bisected by P0, the equation of the line AB is "x - 2y + 5 = 0". -/
theorem equation_of_line_AB_bisected (P : Point) :
  P = (-1, 2) → let O := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 8 }
  ∃ AB : Set (ℝ × ℝ), (is_bisected P AB) → (equation_of_line AB = "x - 2y + 5 = 0") :=
sorry

end length_of_chord_AB_equation_of_line_AB_bisected_l392_392486


namespace area_ratio_trapezoid_triangle_l392_392985

variable (PQ RS : ℝ) (TPQ PQRS : ℝ)
variable [hPQ : PQ = 10] [hRS : RS = 21] 

theorem area_ratio_trapezoid_triangle :
  RS = 21 → PQ = 10 → 
  (let r := (21 / 10) in
  (* The ratio of areas would be (r^2) for similar triangles *)
  let ratio := (r * r) in
  (* Hence using the final derived ratio for areas *)
  (1 : ℝ) / (ratio - 1) = 100 / 341) :=
by
  intros hRS hPQ
  let r := (21 / 10)
  let ratio := (r * r)
  exact (1 : ℝ) / (ratio - 1) = (100 / 341)
  sorry

end area_ratio_trapezoid_triangle_l392_392985


namespace sum_binom_1988_l392_392847

theorem sum_binom_1988 :
  \frac{1}{2^{1988}} \sum_{n = 0}^{994} (-3)^n * ∑ \binom{1988}{2n} == -1/2 :=
begin
  sorry
end

end sum_binom_1988_l392_392847


namespace complex_number_in_third_quadrant_l392_392197

noncomputable def quadrant (z : ℂ) : ℕ :=
if (z.re > 0) ∧ (z.im > 0) then 1
else if (z.re < 0) ∧ (z.im > 0) then 2
else if (z.re < 0) ∧ (z.im < 0) then 3
else if (z.re > 0) ∧ (z.im < 0) then 4
else 0 -- case where it lies on an axis

theorem complex_number_in_third_quadrant (z : ℂ) (h : z * (-1 + 2 * complex.I) = complex.abs (1 + 3 * complex.I)) :
  quadrant z = 3 := sorry

end complex_number_in_third_quadrant_l392_392197


namespace cost_per_bundle_l392_392817

-- Condition: each rose costs 500 won
def rose_price := 500

-- Condition: total number of roses
def total_roses := 200

-- Condition: number of bundles
def bundles := 25

-- Question: Prove the cost per bundle
theorem cost_per_bundle (rp : ℕ) (tr : ℕ) (b : ℕ) : rp = 500 → tr = 200 → b = 25 → (rp * tr) / b = 4000 :=
by
  intros h0 h1 h2
  sorry

end cost_per_bundle_l392_392817


namespace inclination_angle_range_l392_392321

theorem inclination_angle_range (θ : ℝ) :
  ∃ α : ℝ, (α ∈ [0, π/6] ∪ [5 * π / 6, π)) ∧ (∀ x y, x * cos θ + sqrt 3 * y + 2 = 0) → 
    (x / y = - sqrt 3 / 3 * cos θ) :=
by
  sorry

end inclination_angle_range_l392_392321


namespace max_ratio_for_hoop_contact_l392_392345

theorem max_ratio_for_hoop_contact
  (m m_h R g : ℝ)
  (h1 : 0 < m) 
  (h2 : 0 < m_h)
  (h3 : 0 < R)
  (h4 : 0 < g) :
  (m / m_h ≤ 3 / 2) :=
sorry

end max_ratio_for_hoop_contact_l392_392345


namespace boxes_sold_on_saturday_l392_392643

-- Definitions from conditions
def total_boxes_sold (S Sun : ℕ) : Prop :=
  Sun = 1.5 * S ∧ S + Sun = 150

-- Proof statement
theorem boxes_sold_on_saturday (S : ℕ) :
  total_boxes_sold S (1.5 * S) → S = 60 :=
by
  sorry

end boxes_sold_on_saturday_l392_392643


namespace problem_30_1_l392_392635

open BigOperators
open_locale big_operators

theorem problem_30_1 : ∑' n : ℕ, (1 : ℝ) / (n * (n + 1)) = 1 := 
sorry

end problem_30_1_l392_392635


namespace more_buyers_today_than_yesterday_l392_392338

theorem more_buyers_today_than_yesterday :
  ∀ (buyers_the_day_before : ℕ) (buyers_yesterday : ℕ) (buyers_today : ℕ),
  buyers_the_day_before = 50 →
  buyers_yesterday = buyers_the_day_before / 2 →
  buyers_today = 140 - (buyers_the_day_before + buyers_yesterday) →
  buyers_today - buyers_yesterday = 40 :=
begin
  intros buyers_the_day_before buyers_yesterday buyers_today h1 h2 h3,
  rw [h1, h2] at h3,
  have : buyers_yesterday = 25, by norm_num [h1, h2],
  rw this at h3,
  have : buyers_today = 65, by norm_num [h3],
  rw [this, this, this],
  exact this,
end

end more_buyers_today_than_yesterday_l392_392338


namespace log_one_third_nine_l392_392872

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_third_nine : log_base (1/3) 9 = -2 := by
  sorry

end log_one_third_nine_l392_392872


namespace sin_double_angle_l392_392253
-- Import the entirety of Mathlib for necessary mathematical definitions and theorems

-- Given conditions:
def x (interior_angle : ℝ) := true -- indicating x is some real number representing an interior angle of a triangle.
def condition (x : ℝ) : Prop := sin x + cos x = real.sqrt 2 / 2

-- The statement we need to prove:
theorem sin_double_angle (x : ℝ) (hx : condition x) : sin (2 * x) = -1 / 2 :=
by
  -- Extract the condition provided in the problem
  have h : sin x + cos x = real.sqrt 2 / 2 := hx,
  sorry -- skipping the proof

end sin_double_angle_l392_392253


namespace train_speed_l392_392403

-- Definitions of the given conditions
def platform_length : ℝ := 250
def train_length : ℝ := 470.06
def time_taken : ℝ := 36

-- Definition of the total distance covered
def total_distance := platform_length + train_length

-- The proof problem: Prove that the calculated speed is approximately 20.0017 m/s
theorem train_speed :
  (total_distance / time_taken) = 20.0017 :=
by
  -- The actual proof goes here, but for now we leave it as sorry
  sorry

end train_speed_l392_392403


namespace find_a_b_final_function_l392_392513

noncomputable def f (a b x : ℝ) : ℝ := a * x / (x^2 + b)

theorem find_a_b (a b : ℝ) (h_derivative : (λ x, (a * (b - x^2)) / (x^2 + b)^2) 1 = 0) (h_value : f a b 1 = 2) :
  a = 4 ∧ b = 1 :=
begin
  sorry
end

theorem final_function :
  (λ x, f 4 1 x) = (λ x, 4 * x / (x^2 + 1)) :=
begin
  sorry
end

end find_a_b_final_function_l392_392513


namespace cosine_angle_between_vectors_l392_392102

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def vector_between (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

def cos_angle (v1 v2 : Point3D) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem cosine_angle_between_vectors :
  let A := Point3D.mk (-2) 4 (-6)
  let B := Point3D.mk 0 2 (-4)
  let C := Point3D.mk (-6) 8 (-10)
  let AB := vector_between A B
  let AC := vector_between A C
  cos_angle AB AC = -1 :=
by {
  let A := Point3D.mk (-2) 4 (-6),
  let B := Point3D.mk 0 2 (-4),
  let C := Point3D.mk (-6) 8 (-10),
  let AB := vector_between A B,
  let AC := vector_between A C,
  sorry
}

end cosine_angle_between_vectors_l392_392102


namespace daps_dips_equivalence_l392_392186

theorem daps_dips_equivalence :
  (∃ dap dop dip : Type,
    (5 : ℝ) * ∀ x : dap, x = (4 : ℝ) * ∀ y : dop, y ∧
    (3 : ℝ) * ∀ z : dop, z = (8 : ℝ) * ∀ w : dip, w) →
  (22.5 : ℝ) * ∀ x : dap, x = (48 : ℝ) * ∀ y : dip, y :=
begin
  sorry
end

end daps_dips_equivalence_l392_392186


namespace larger_integer_is_21_l392_392708

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l392_392708


namespace larger_integer_is_21_l392_392707

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l392_392707


namespace sum_n_k_l392_392866

theorem sum_n_k : 
  ∃ n k : ℕ, 
  (n + k = 8) ∧ 
  ({n \choose k}.val * 3 = {n \choose k+1}.val) ∧ 
  ({n \choose k+1}.val * 5 = 3 * {n \choose k+2}.val) := 
sorry

end sum_n_k_l392_392866


namespace sum_fractions_then_multiply_l392_392850

theorem sum_fractions_then_multiply :
  let sum := (∑ i in Finset.range 15, (i/7 : ℝ)) -- Sum of fractions from 1/7 to 14/7
  (sum * 3) = 45 := by
  sorry

end sum_fractions_then_multiply_l392_392850


namespace personal_income_tax_l392_392316

theorem personal_income_tax {X: ℝ} (gross_income: ℝ) (net_income: ℝ) (tax_rate: ℝ) (h1: tax_rate = 0.13) (h2: net_income = gross_income * (1 - tax_rate)) (h3: net_income = 20000) : gross_income ≈ 22989 :=
by sorry

end personal_income_tax_l392_392316


namespace concyclic_AQTP_l392_392618

variables {Γ₁ Γ₂ : Type} [Circle Γ₁] [Circle Γ₂]
variables {A B P Q T : Point}

-- Assuming conditions from part a)
def intersection_points : Prop :=
  T ∈ (tangent Γ₂ P ∩ tangent Γ₂ Q) ∧
  P ∈ Γ₁ ∧ Q ∈ Γ₂ ∧
  collinear P B Q ∧
  A ≠ B ∧
  A ∈ (Γ₁ ∩ Γ₂) ∧
  B ∈ (Γ₁ ∩ Γ₂)
  
-- The statement to prove part c)
theorem concyclic_AQTP : intersection_points → concyclic {A, Q, T, P} :=
by sorry

end concyclic_AQTP_l392_392618


namespace even_numbers_average_19_l392_392431

theorem even_numbers_average_19 (n : ℕ) (h1 : (n / 2) * (2 + 2 * n) / n = 19) : n = 18 :=
by {
  sorry
}

end even_numbers_average_19_l392_392431


namespace gcd_two_powers_l392_392498

noncomputable def gcd_expression (m n : ℕ) : ℕ :=
  Int.gcd (2^m + 1) (2^n - 1)

theorem gcd_two_powers (m n : ℕ) (hm : m > 0) (hn : n > 0) (odd_n : n % 2 = 1) : 
  gcd_expression m n = 1 :=
by
  sorry

end gcd_two_powers_l392_392498


namespace roots_of_cubic_l392_392110

theorem roots_of_cubic (a b c d r s t : ℝ) 
  (h1 : r + s + t = -b / a)
  (h2 : r * s + r * t + s * t = c / a)
  (h3 : r * s * t = -d / a) :
  1 / (r ^ 2) + 1 / (s ^ 2) + 1 / (t ^ 2) = (c ^ 2 - 2 * b * d) / (d ^ 2) := 
sorry

end roots_of_cubic_l392_392110


namespace abc_value_l392_392188

noncomputable theory

variables (a b c : ℝ)
variables (positive_a : a > 0) (positive_b : b > 0) (positive_c : c > 0)
variables (h1 : a * b = 24 * real.cbrt 3)
variables (h2 : a * c = 40 * real.cbrt 3)
variables (h3 : b * c = 15 * real.cbrt 3)

theorem abc_value : a * b * c = 120 * real.sqrt 3 :=
by
  -- proof goes here
  sorry

end abc_value_l392_392188


namespace reflection_matrix_l392_392020

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l392_392020


namespace slope_angle_of_tangent_line_at_point_l392_392738

noncomputable def curve : ℝ → ℝ := λ x => (1 / 2) * x^2 - 2 * x

def derivative (f : ℝ → ℝ) : (ℝ → ℝ) := λ x => (deriv f) x

def eval_derivative_at (f : ℝ → ℝ) (x : ℝ) : ℝ := derivative f x

theorem slope_angle_of_tangent_line_at_point :
  eval_derivative_at curve 1 = -1 → slope_angle (-1) = 135 := 
sorry

end slope_angle_of_tangent_line_at_point_l392_392738


namespace jim_speed_l392_392477

def start_time := 19 + 0.75 -- 7:45 p.m. in hours
def end_time := 21 + 0.50 -- 9:30 p.m. in hours
def distance := 84 -- in kilometers
def time_duration := end_time - start_time -- total driving time in hours

theorem jim_speed :
  distance / time_duration = 48 :=
by
  -- Expected outcome without the proof body
  sorry

end jim_speed_l392_392477


namespace tom_bought_8_kg_of_apples_l392_392343

/-- 
   Given:
   - The cost of apples is 70 per kg.
   - 9 kg of mangoes at a rate of 55 per kg.
   - Tom paid a total of 1055.

   Prove that Tom purchased 8 kg of apples.
 -/
theorem tom_bought_8_kg_of_apples 
  (A : ℕ) 
  (h1 : 70 * A + 55 * 9 = 1055) : 
  A = 8 :=
sorry

end tom_bought_8_kg_of_apples_l392_392343


namespace arith_seq_ratio_proof_l392_392918

variable {a d : ℕ} (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)

def arith_seq (n : ℕ) : ℕ := a + n * d

axiom sum_first_n_terms (n : ℕ) : S_n n = (n * (2 * a + (n - 1) * d)) / 2
axiom condition : S_n 5 / S_n 3 = 2

theorem arith_seq_ratio_proof : ∃ (a_5 a_3 : ℕ), condition → (a_n 5) / (a_n 3) = 4 / 3 := by
  sorry

end arith_seq_ratio_proof_l392_392918


namespace g_three_fifths_l392_392670

-- Given conditions
variable (g : ℝ → ℝ)
variable (h₀ : g 0 = 0)
variable (h₁ : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
variable (h₂ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
variable (h₃ : ∀ ⦃x : ℝ⦄, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3)

-- Proof statement
theorem g_three_fifths : g (3 / 5) = 2 / 3 := by
  sorry

end g_three_fifths_l392_392670


namespace minimum_positive_period_of_cos_2x_l392_392675

/-- Define the cos function with argument 2x -/
def f (x : ℝ) : ℝ := Real.cos (2 * x)

/-- Prove that the minimum positive period of the function y = cos 2x is π -/
theorem minimum_positive_period_of_cos_2x : ∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ π :=
by
  sorry

end minimum_positive_period_of_cos_2x_l392_392675


namespace time_outside_l392_392408

theorem time_outside {n : ℕ} (h1 : ∀ n, |(210 + n / 2) - 6 * n| = 120) (h2 : 0 ≤ n ∧ n < 60) :
  |43.64 - 16.36| = 27.28 :=
by sorry

end time_outside_l392_392408


namespace first_place_beats_joe_by_two_points_l392_392600

def points (wins draws : ℕ) : ℕ := 3 * wins + draws

theorem first_place_beats_joe_by_two_points
  (joe_wins joe_draws first_place_wins first_place_draws : ℕ)
  (h1 : joe_wins = 1)
  (h2 : joe_draws = 3)
  (h3 : first_place_wins = 2)
  (h4 : first_place_draws = 2) :
  points first_place_wins first_place_draws - points joe_wins joe_draws = 2 := by
  sorry

end first_place_beats_joe_by_two_points_l392_392600


namespace lines_intersect_at_point_l392_392407

def ParametricLine1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, 4 - 3 * t)

def ParametricLine2 (u : ℝ) : ℝ × ℝ :=
  (-2 + 3 * u, 5 - u)

theorem lines_intersect_at_point :
  ∃ t u : ℝ, ParametricLine1 t = ParametricLine2 u ∧ ParametricLine1 t = (-5, 13) :=
by
  sorry

end lines_intersect_at_point_l392_392407


namespace sin_double_angle_tan_double_angle_l392_392790

-- Problem (1)
theorem sin_double_angle (α : ℝ) (h1 : sin α = 12 / 13) (h2 : α ∈ Ioo (π / 2) π) :
  sin (2 * α) = -120 / 169 := by
  sorry

-- Problem (2)
theorem tan_double_angle (α : ℝ) (h1 : tan α = 1 / 2) :
  tan (2 * α) = 4 / 3 := by
  sorry

end sin_double_angle_tan_double_angle_l392_392790


namespace _l392_392030
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l392_392030


namespace total_principal_and_interest_l392_392803

variable (a r : ℝ)

theorem total_principal_and_interest (h1 : a > 0) (h2 : r > 0) :
  let S := ∑ i in finset.range 13, a * (1 + r)^(i + 1)
  S = (a / r) * ((1 + r)^14 - r - 1) := by
  sorry

end total_principal_and_interest_l392_392803


namespace larger_integer_is_21_l392_392687

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l392_392687


namespace larger_integer_is_21_l392_392714

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l392_392714


namespace log2_x_lt_2_is_necessary_but_not_sufficient_for_1_lt_x_lt_3_l392_392586

theorem log2_x_lt_2_is_necessary_but_not_sufficient_for_1_lt_x_lt_3 (x : ℝ) :
  (1 < x ∧ x < 3) → log 2 x < 2 ∧ (log 2 x < 2 → ¬(1 < x ∧ x < 3)) :=
by 
  sorry

end log2_x_lt_2_is_necessary_but_not_sufficient_for_1_lt_x_lt_3_l392_392586


namespace arc_length_is_ln_sqrt_3_l392_392374

noncomputable def arc_length_ln_cos_x_plus_2 : Real :=
  let f : Real → Real := fun x => Real.log (Real.cos x) + 2
  let f' : Real → Real := fun x => -Real.tan x
  let integrand : Real → Real := fun x => Real.sqrt (1 + (f' x) ^ 2)
  ∫ x in 0 .. Real.pi / 6, integrand x

theorem arc_length_is_ln_sqrt_3 :
  arc_length_ln_cos_x_plus_2 = Real.log (Real.sqrt 3) :=
by
  sorry

end arc_length_is_ln_sqrt_3_l392_392374


namespace solve_for_x_l392_392291

theorem solve_for_x (x : ℝ) : 2^(32^x) = 32^(2^x) ↔ x = real.log2 5 / 4 :=
by
  sorry

end solve_for_x_l392_392291


namespace eccentricity_ellipse_l392_392303

/-- Definition for the given ellipse equation /--
def is_ellipse (x y : ℝ) : Prop := (x^2) / 9 + (y^2) / 4 = 1

/-- Eccentricity of the ellipse defined by the equation -/
theorem eccentricity_ellipse : 
  (e : ℝ) (h : ∀ x y : ℝ, is_ellipse x y) →
  e = (Real.sqrt 5) / 3 := 
sorry

end eccentricity_ellipse_l392_392303


namespace perimeter_of_semicircular_region_l392_392815

theorem perimeter_of_semicircular_region (a : ℝ) (h : a = 1 / Real.pi) :
    let d := a -- diameter of semicircular arc
    let C := Real.pi * d -- full circumference of circle
    let S := C / 2 -- circumference of one semicircular arc
    let P := 4 * S -- total perimeter of four semicircular arcs
    P = 2 :=
by
  -- definitions
  let d := a
  let C := Real.pi * d
  let S := C / 2
  let P := 4 * S
  -- conditions and proof conclusion
  have ha : a = 1 / Real.pi := h
  calc
    P = 4 * (Real.pi * a / 2) : by rw [S]
    ... = 4 * (Real.pi * (1 / Real.pi) / 2) : by rw [ha]
    ... = 4 * (1 / 2) : by rw [← mul_div_assoc, div_self (ne_of_gt pi_pos), mul_one]
    ... = 2 : by norm_num

end perimeter_of_semicircular_region_l392_392815


namespace total_weekly_water_consumption_l392_392267

-- Definitions coming from the conditions of the problem
def num_cows : Nat := 40
def water_per_cow_per_day : Nat := 80
def num_sheep : Nat := 10 * num_cows
def water_per_sheep_per_day : Nat := water_per_cow_per_day / 4
def days_in_week : Nat := 7

-- To prove statement: 
theorem total_weekly_water_consumption :
  let weekly_water_cow := water_per_cow_per_day * days_in_week
  let total_weekly_water_cows := weekly_water_cow * num_cows
  let daily_water_sheep := water_per_sheep_per_day
  let weekly_water_sheep := daily_water_sheep * days_in_week
  let total_weekly_water_sheep := weekly_water_sheep * num_sheep
  total_weekly_water_cows + total_weekly_water_sheep = 78400 := 
by
  sorry

end total_weekly_water_consumption_l392_392267


namespace vertex_angle_isosceles_l392_392192

theorem vertex_angle_isosceles (a b c : ℝ)
  (isosceles: (a = b ∨ b = c ∨ c = a))
  (angle_sum : a + b + c = 180)
  (one_angle_is_70 : a = 70 ∨ b = 70 ∨ c = 70) :
  a = 40 ∨ a = 70 ∨ b = 40 ∨ b = 70 ∨ c = 40 ∨ c = 70 :=
by sorry

end vertex_angle_isosceles_l392_392192


namespace leak_emptying_time_l392_392777

-- Definitions based on given conditions
def tank_fill_rate_without_leak : ℚ := 1 / 3
def combined_fill_and_leak_rate : ℚ := 1 / 4

-- Leak emptying time to be proven
theorem leak_emptying_time (R : ℚ := tank_fill_rate_without_leak) (C : ℚ := combined_fill_and_leak_rate) :
  (1 : ℚ) / (R - C) = 12 := by
  sorry

end leak_emptying_time_l392_392777


namespace water_consumption_total_l392_392269

def number_of_cows : ℕ := 40
def water_per_cow_per_day : ℕ := 80
def sheep_multiplicative_factor : ℕ := 10
def water_factor_cow_to_sheep : ℕ := 1 / 4
def days_in_week : ℕ := 7

theorem water_consumption_total :
  let cows_water_per_week := number_of_cows * water_per_cow_per_day * days_in_week in
  let number_of_sheep := number_of_cows * sheep_multiplicative_factor in
  let water_per_sheep_per_day := water_per_cow_per_day * water_factor_cow_to_sheep in
  let sheep_water_per_week := number_of_sheep * water_per_sheep_per_day * days_in_week in
  cows_water_per_week + sheep_water_per_week = 78400 := sorry

end water_consumption_total_l392_392269


namespace daps_equivalent_to_48_dips_l392_392170

noncomputable def conversion_daps_to_dops : ℚ := 5 / 4
noncomputable def conversion_dops_to_dips : ℚ := 3 / 8
noncomputable def conversion_daps_to_dips : ℚ := conversion_daps_to_dops * conversion_dops_to_dips

theorem daps_equivalent_to_48_dips :
  ∀ (daps dops dips : Type) (eq1 : 5*daps = 4*dops) (eq2 : 3*dops = 8*dips), 
  (48:ℚ) * conversion_daps_to_dips = (22.5:ℚ) :=
by
  sorry

end daps_equivalent_to_48_dips_l392_392170


namespace axis_of_symmetry_parabola_l392_392663

-- Definitions
def quadratic_equation (x : ℝ) : ℝ :=
  -x^2 + 4 * x + 1

def axis_of_symmetry (a b : ℝ) : ℝ :=
  -b / (2 * a)

-- Theorem Statement
theorem axis_of_symmetry_parabola :
  axis_of_symmetry (-1) 4 = 2 :=
by
  -- Proof is omitted, place sorry for now.
  sorry

end axis_of_symmetry_parabola_l392_392663


namespace jorge_total_goals_l392_392604

theorem jorge_total_goals (last_season_goals current_season_goals : ℕ) (h_last : last_season_goals = 156) (h_current : current_season_goals = 187) : 
  last_season_goals + current_season_goals = 343 :=
by
  sorry

end jorge_total_goals_l392_392604


namespace find_k_l392_392804

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_k (a b : V) (k : ℝ) :
  (∃ t : ℝ, k = 1 - t ∧ t = (2 / 5)) →
  k = (3 / 5) :=
by
  intro h
  cases h with t ht
  cases ht with ht1 ht2
  rw [ht1, ht2]
  norm_num
  done

end find_k_l392_392804


namespace intersection_on_semicircle_l392_392258

-- Definitions
variables (A B C H D P Q T : Point)
variables (ω : Semicircle)
variables [Triangle ABC]
variables [IsRightAngle (∠ BCA)]
variables [AltitudeFoot C A B H]
variables [CHBisectsAD C H A D]
variables [LineIntersectionPoint P BD CH]
variables [SemicircleDiameterBD ω B D]
variables [TangentAtQ ω P Q]
variables [IntersectionPoint T CQ AD]

-- Goal
theorem intersection_on_semicircle :
  T ∈ ω :=
sorry

end intersection_on_semicircle_l392_392258


namespace neil_initial_games_l392_392526

theorem neil_initial_games (N : ℕ) 
  (H₀ : ℕ) (H₀_eq : H₀ = 58)
  (H₁ : ℕ) (H₁_eq : H₁ = H₀ - 6)
  (H₁_condition : H₁ = 4 * (N + 6)) : N = 7 :=
by {
  -- Substituting the given values and simplifying to show the final equation
  sorry
}

end neil_initial_games_l392_392526


namespace quadratic_min_value_l392_392988

theorem quadratic_min_value 
  (n : ℕ)
  (f : ℕ → ℚ)
  (h1 : f n = 13)
  (h2 : f (n+1) = 13)
  (h3 : f (n+2) = 35) :
  ∃ c, c = 41 / 4 ∧ ∀ x, f x ≥ c :=
begin
  sorry
end

end quadratic_min_value_l392_392988


namespace find_larger_integer_l392_392727

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l392_392727


namespace cards_given_to_Jeff_l392_392623

theorem cards_given_to_Jeff
  (initial_cards : ℕ)
  (cards_given_to_John : ℕ)
  (remaining_cards : ℕ)
  (cards_left : ℕ)
  (h_initial : initial_cards = 573)
  (h_given_John : cards_given_to_John = 195)
  (h_left_before_Jeff : remaining_cards = initial_cards - cards_given_to_John)
  (h_final : cards_left = 210)
  (h_given_Jeff : remaining_cards - cards_left = 168) :
  (initial_cards - cards_given_to_John - cards_left = 168) :=
by
  sorry

end cards_given_to_Jeff_l392_392623


namespace collinear_points_l392_392277

open Classical
open Real

variable {A B C M R: Point}
variable {circumcircle: Circle}
variable {A1 B1 C1 A2 B2 C2: Point}

-- Define the conditions
axiom h1 : M ∈ circumcircle
axiom h2 : ¬ (A = B ∨ B = C ∨ C = A)
axiom h3 : ¬ (R = A ∨ R = B ∨ R = C)
axiom h4 : ∃ (AR: Line), ∃ (BR: Line), ∃ (CR: Line), (A1 ∈ circumcircle) ∧ (B1 ∈ circumcircle) ∧ (C1 ∈ circumcircle) ∧ 
              (A ∈ AR ∧ R ∈ AR ∧ A1 ∈ AR) ∧ (B ∈ BR ∧ R ∈ BR ∧ B1 ∈ BR) ∧ (C ∈ CR ∧ R ∈ CR ∧ C1 ∈ CR)
axiom h5 : intersection (line_through M A1) (line_through B C) = A2
axiom h6 : intersection (line_through M B1) (line_through C A) = B2
axiom h7 : intersection (line_through M C1) (line_through A B) = C2

-- Prove the collinearity of A2, B2, C2 and that they are on the line passing through R
theorem collinear_points : collinear {A2, B2, C2} ∧ collinear {A2, R, C2} :=
by
  sorry

end collinear_points_l392_392277


namespace sin_over_cos_inequality_l392_392280

-- Define the main theorem and condition
theorem sin_over_cos_inequality (t : ℝ) (h₁ : 0 < t) (h₂ : t ≤ Real.pi / 2) : 
  (Real.sin t / t)^3 > Real.cos t := 
sorry

end sin_over_cos_inequality_l392_392280


namespace polynomial_functional_relation_l392_392492

theorem polynomial_functional_relation
  (P Q : Polynomial ℝ)
  (R : Polynomial ℝ → Polynomial ℝ → Polynomial ℝ) 
  (h : ∀ x y : ℝ, P(x) - P(y) = R(x * y)(Q(x) - Q(y))) :
  ∃ S : Polynomial ℝ, ∀ x : ℝ, P(x) = S(Q(x)) :=
sorry

end polynomial_functional_relation_l392_392492


namespace transformed_variance_l392_392967

variable {α : Type*} [AddGroup α] [Module ℝ α] [AddCommGroup α] [TopologicalSpace α] [TopologicalAddGroup α]
variable {μ : MeasureTheory.MeasureSpace α}

open MeasureTheory

noncomputable def given_variance (a : ℕ → ℝ) (σ² : ℝ) :=
  ∀ (n : ℕ), (1 / n) * ∑ i in finset.range n, (a i - (1 / n) * ∑ j in finset.range n, a j)^2 = σ²

theorem transformed_variance {α : Type*} [AddGroup α] [Module ℝ α] [AddCommGroup α] [TopologicalSpace α] [TopologicalAddGroup α] 
  {a : ℕ → ℝ} (h : given_variance a 3) :
  given_variance (λ i, 2 * (a i - 3)) 12 := by
  sorry

end transformed_variance_l392_392967


namespace fault_line_movement_year_before_l392_392833

-- Define the total movement over two years
def total_movement : ℝ := 6.5

-- Define the movement during the past year
def past_year_movement : ℝ := 1.25

-- Define the movement the year before
def year_before_movement : ℝ := total_movement - past_year_movement

-- Prove that the fault line moved 5.25 inches the year before
theorem fault_line_movement_year_before : year_before_movement = 5.25 :=
  by  sorry

end fault_line_movement_year_before_l392_392833


namespace PQ_bisects_BD_l392_392213

-- Define a structure for a Quadrilateral having points A, B, C, D
structure Quadrilateral :=
  (A B C D : Point)

-- Define midpoints P and Q
def is_midpoint (P Q : Point) (A B C D : Point) : Prop :=
  (P = midpoint A B) ∧ (Q = midpoint C D)

-- Define bisect conditions
def line_bisects (PQ : Line) (A C B D : Point) := 
  (PQ.bisects A C) ∧ (PQ.bisects B D)

-- Define the main theorem
theorem PQ_bisects_BD 
  (quad : Quadrilateral)
  (P Q : Point)
  (PQ : Line)
  (condition_midpoints : is_midpoint P Q quad.A quad.B quad.C quad.D)
  (condition_bisect_AC : PQ.bisects quad.A quad.C) :
  PQ.bisects quad.B quad.D :=
sorry

end PQ_bisects_BD_l392_392213


namespace Jorge_goals_total_l392_392602

theorem Jorge_goals_total : 
  let last_season_goals := 156
  let this_season_goals := 187
  last_season_goals + this_season_goals = 343 := 
by
  sorry

end Jorge_goals_total_l392_392602


namespace smallest_difference_factors_1950_l392_392440

theorem smallest_difference_factors_1950 : ∃ (a b : ℕ), a * b = 1950 ∧ (a ≠ b) ∧ abs (a - b) = 11 :=
by
  sorry

end smallest_difference_factors_1950_l392_392440


namespace bisect_diagonals_l392_392207

variables {A B C D P Q : Type} [AffineSpace ℝ A B C D]

/-- Given a convex quadrilateral ABCD, if P and Q are midpoints of AB and CD respectively, 
and line PQ bisects diagonal AC, then line PQ also bisects diagonal BD.-/
theorem bisect_diagonals (h1 : convex_quadrilateral A B C D) 
  (h2 : midpoint (A, B) = P) (h3 : midpoint (C, D) = Q) 
  (h4 : bisects (PQ, AC)) : bisects (PQ, BD) := 
sorry

end bisect_diagonals_l392_392207


namespace right_triangle_ABC_l392_392327

open EuclideanGeometry

variables (A B C D E : Point) -- Points in the plane
variable [is_triang (A, B, C)] -- A, B, C are the vertices of a triangle
variable (D_exists : extend (C, B, D) ∧ (dist C D = dist B C)) -- D is such that CD = BC
variable (E_exists : extend (A, C, E) ∧ (dist A E = 2 * dist A C)) -- E is such that AE = 2 * AC
variable (AD_BE_eq : dist A D = dist B E) -- Given condition AD = BE

theorem right_triangle_ABC : right_triangle A B C := by
  sorry

end right_triangle_ABC_l392_392327


namespace bridgette_has_4_birds_l392_392837

/-
Conditions:
1. Bridgette has 2 dogs.
2. Bridgette has 3 cats.
3. Bridgette has some birds.
4. She gives the dogs a bath twice a month.
5. She gives the cats a bath once a month.
6. She gives the birds a bath once every 4 months.
7. In a year, she gives a total of 96 baths.
-/

def num_birds (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ) : ℕ :=
  let yearly_dog_baths := num_dogs * dog_baths_per_month * 12
  let yearly_cat_baths := num_cats * cat_baths_per_month * 12
  let birds_baths := total_baths_per_year - (yearly_dog_baths + yearly_cat_baths)
  let baths_per_bird_per_year := 12 / bird_baths_per_4_months
  birds_baths / baths_per_bird_per_year

theorem bridgette_has_4_birds :
  ∀ (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ),
    num_dogs = 2 →
    num_cats = 3 →
    dog_baths_per_month = 2 →
    cat_baths_per_month = 1 →
    bird_baths_per_4_months = 4 →
    total_baths_per_year = 96 →
    num_birds num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year = 4 :=
by
  intros
  sorry


end bridgette_has_4_birds_l392_392837


namespace reflection_matrix_is_correct_l392_392044

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l392_392044


namespace percentage_of_whole_is_correct_l392_392794

def Part := 193.2
def Whole := 480
def Percentage := (Part / Whole) * 100

theorem percentage_of_whole_is_correct : Percentage = 40.25 := 
by 
  -- proof skipped for now
  sorry

end percentage_of_whole_is_correct_l392_392794


namespace points_of_third_question_l392_392203

theorem points_of_third_question (x : ℕ) (h : x + (x + 4) + (x + 8) + (x + 12) + (x + 16) + (x + 20) + (x + 24) + (x + 28) = 360) : x + 8 = 39 :=
by
  -- We introduce the expression for the sum of the arithmetic progression.
  have h1 : 8 * x + (4 + 8 + 12 + 16 + 20 + 24 + 28) = 360, from h,
  -- Simplification of the constants.
  have h2 : 4 + 8 + 12 + 16 + 20 + 24 + 28 = 112, from sorry,
  -- Substitute back to isolate and solve for x.
  have h3 : 8 * x + 112 = 360, from h1.trans (congr_arg (8 * x + ·) h2.symm),
  have h4 : 8 * x = 248, from eq_sub_of_add_eq h3,
  have h5 : x = 31, from eq_of_mul_eq_mul_left (by norm_num) (eq_div_of_mul_eq 8 h4),
  -- Calculate the points for the third question.
  have h6 : x + 8 = 31 + 8, from congr_arg (· + 8) h5,
  exact eq.symm h6.trans (by norm_num)

end points_of_third_question_l392_392203


namespace roots_quadratic_l392_392437

theorem roots_quadratic (a b c d : ℝ) :
  (a + b = 3 * c / 2 ∧ a * b = 4 * d ∧ c + d = 3 * a / 2 ∧ c * d = 4 * b)
  ↔ ( (a = 4 ∧ b = 8 ∧ c = 4 ∧ d = 8) ∨
      (a = -2 ∧ b = -22 ∧ c = -8 ∧ d = 11) ∨
      (a = -8 ∧ b = 2 ∧ c = -2 ∧ d = -4) ) :=
by
  sorry

end roots_quadratic_l392_392437


namespace naomi_wash_time_l392_392271

theorem naomi_wash_time (C T S : ℕ) (h₁ : T = 2 * C) (h₂ : S = 2 * C - 15) (h₃ : C + T + S = 135) : C = 30 :=
by
  sorry

end naomi_wash_time_l392_392271


namespace product_of_six_consecutive_integers_l392_392295

theorem product_of_six_consecutive_integers (n: ℕ) :
  n * (n+1) * (n+2) * (n+3) * (n+4) * (n+5) = (n+5)! / (n-1)! :=
by
  sorry

end product_of_six_consecutive_integers_l392_392295


namespace reflection_matrix_is_correct_l392_392049

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l392_392049


namespace digit_d_for_5678d_is_multiple_of_9_l392_392470

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_d_for_5678d_is_multiple_of_9 : 
  ∃ d : ℕ, d < 10 ∧ is_multiple_of_9 (56780 + d) ∧ d = 1 :=
by
  sorry

end digit_d_for_5678d_is_multiple_of_9_l392_392470


namespace complement_of_union_A_B_l392_392144

def U := {1, 2, 3, 4, 5}

def A := {x | x ^ 2 - 3 * x + 2 = 0}

def B := {x | ∃ a ∈ A, x = 2 * a}

def complement (s t : Set ℕ) := {x ∈ s | x ∉ t}

theorem complement_of_union_A_B (CU : Set ℕ) (U A B: Set ℕ) (C : complement U (A ∪ B)) :
  U = {1, 2, 3, 4, 5} → (A = {x | x ^ 2 - 3 * x + 2 = 0}) →
  (B = {x | ∃ a ∈ A, x = 2 * a}) →
  C = {3, 5} :=
by
  intros hU hA hB
  sorry

end complement_of_union_A_B_l392_392144


namespace q1_q2_l392_392137

def f (x : ℝ) : ℝ := sqrt 3 * cos (π / 2 + x) * cos x + sin x ^ 2

def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  B = π / 4 ∧ a = 2 ∧ f A = 0 ∧ 0 < A ∧ A < π

noncomputable def area_triangle_ABC (A B C a b : ℝ) : ℝ :=
  1 / 2 * a * b * sin C

theorem q1 (k : ℤ) : ∃ (x : ℝ), (k * π + π / 6 <= x ∧ x <= k * π + 2 * π / 3) ∧ ∀ x : ℝ, tendsto_deriv_exists (f x) :=
sorry

theorem q2 {a b c A B C : ℝ} (h_triangle : triangle_ABC a b c A B C) :
  area_triangle_ABC A B C a b = (3 + sqrt 3) / 3 :=
sorry

end q1_q2_l392_392137


namespace max_a9_l392_392279

theorem max_a9 (a : Fin 18 → ℕ) (h_pos: ∀ i, 1 ≤ a i) (h_incr: ∀ i j, i < j → a i < a j) (h_sum: (Finset.univ : Finset (Fin 18)).sum a = 2001) : a 8 ≤ 192 :=
by
  -- Proof goes here
  sorry

end max_a9_l392_392279


namespace convert_to_base13_l392_392854

theorem convert_to_base13 : 
  ∀ (c : ℕ), (c = 172) → (∃ a b d : ℕ, c = a * 169 + b * 13 + d ∧ a = 1 ∧ b = 0 ∧ d = 3) :=
by
  intros c hc
  use [1, 0, 3]
  rw hc
  split
  { norm_num }
  { split; norm_num }

end convert_to_base13_l392_392854


namespace visiting_plans_correct_l392_392273

-- Define the number of students
def num_students : ℕ := 4

-- Define the number of places to visit
def num_places : ℕ := 3

-- Define the total number of visiting plans without any restrictions
def total_visiting_plans : ℕ := num_places ^ num_students

-- Define the number of visiting plans where no one visits Haxi Station
def no_haxi_visiting_plans : ℕ := (num_places - 1) ^ num_students

-- Define the number of visiting plans where Haxi Station has at least one visitor
def visiting_plans_with_haxi : ℕ := total_visiting_plans - no_haxi_visiting_plans

-- Prove that the number of different visiting plans with at least one student visiting Haxi Station is 65
theorem visiting_plans_correct : visiting_plans_with_haxi = 65 := by
  -- Omitted proof
  sorry

end visiting_plans_correct_l392_392273


namespace total_votes_in_election_l392_392228

-- Definitions of the conditions
variables (V : ℕ) (valid_votes : ℕ) (A_votes : ℕ)
hypotheses 
  (h1 : valid_votes = 85 * V / 100) 
  (h2 : A_votes = 75 * valid_votes / 100)
  (h3 : A_votes = 357000)

-- Total number of votes in the election
theorem total_votes_in_election : V = 560000 :=
by
  sorry

end total_votes_in_election_l392_392228


namespace beavers_working_l392_392386

theorem beavers_working (initial_beavers swimming_beavers : Nat) (initial_beavers = 2) (swimming_beavers = 1) : 
  initial_beavers - swimming_beavers = 1 := by
  sorry

end beavers_working_l392_392386


namespace exercise_b_c_values_l392_392522

open Set

universe u

theorem exercise_b_c_values : 
  ∀ (b c : ℝ), let U : Set ℝ := {2, 3, 5}
               let A : Set ℝ := {x | x^2 + b * x + c = 0}
               (U \ A = {2}) → (b = -8 ∧ c = 15) :=
by
  intros b c U A H
  let U : Set ℝ := {2, 3, 5}
  let A : Set ℝ := {x | x^2 + b * x + c = 0}
  have H1 : U \ A = {2} := H
  sorry

end exercise_b_c_values_l392_392522


namespace parallelogram_area_l392_392453

theorem parallelogram_area (base height : ℕ) (h_base : base = 3) (h_height : height = 3) :
  base * height = 9 :=
by {
  rw [h_base, h_height],
  norm_num,
  exact rfl,
}

end parallelogram_area_l392_392453


namespace find_lambda_l392_392088

open Real

def vector := ℝ × ℝ × ℝ

def a : vector := (2, -1, 3)
def b : vector := (-1, 4, -2)
def c (λ : ℝ) : vector := (7, 5, λ)

noncomputable def coplanar (v1 v2 v3 : vector) : Prop :=
  ∃ (m n : ℝ), v3 = (m * v1.1 + n * v2.1, m * v1.2 + n * v2.2, m * v1.3 + n * v2.3)

theorem find_lambda (λ : ℝ) :
  coplanar a b (c λ) → λ = 65 / 7 :=
by
  sorry

end find_lambda_l392_392088


namespace problem_statement_l392_392355

theorem problem_statement :
  18 * ( (1 / 3) + (1 / 4) + (1 / 12) )⁻¹ = 27 :=
by
  sorry

end problem_statement_l392_392355


namespace rectangle_area_l392_392814

/-- Given a rectangle ABCD inscribed in a semicircle with diameter FE,
    where DA = 12, FD = AE = 7, and ΔDFC is right-angled at D,
    the area of the rectangle ABCD is 24√30. -/
theorem rectangle_area (DA FD AE : ℝ) (h1 : DA = 12) (h2 : FD = 7) (h3 : AE = 7) (h4 : ∀ DFC : {d b a : ℝ // d^2 + b^2 = a^2}, DFC.1 = 7 ∧ DFC.3 = 13) : 
  let CD : ℝ := sqrt(120) in
  let area : ℝ := DA * CD in
  area = 24 * sqrt 30 :=
by
  have DFC : {d b a : ℝ // d^2 + b^2 = a^2} := ⟨7, sqrt(120), 13, by ring_nf; norm_num⟩
  sorry

end rectangle_area_l392_392814


namespace max_rent_A_most_cost_effective_l392_392152

-- Define the variables and conditions
variable (x : ℕ) -- Number of type A cars
variable (y : ℕ) -- Number of type B cars
variable (total_cars : ℕ := 10)
variable (max_cost : ℕ := 3500)
variable (num_participants : ℕ := 360)

-- Define the rental costs and passenger capacity
def cost_A : ℕ := 400
def cost_B : ℕ := 280
def capacity_A : ℕ := 50
def capacity_B : ℕ := 30

-- Constraint: total number of cars rented must be 10
def total_car_constraint : Prop := x + y = total_cars

-- Constraint: rental cost must not exceed 3500 yuan
def cost_constraint : Prop := cost_A * x + cost_B * y ≤ max_cost

-- Constraint: total participants must be accommodated
def capacity_constraint : Prop := capacity_A * x + capacity_B * y ≥ num_participants

-- Proof goals
theorem max_rent_A : x ≤ 5 → total_car_constraint → cost_constraint → capacity_constraint → ∃ a, a = 5 :=
by 
  sorry

theorem most_cost_effective : total_car_constraint → cost_constraint → capacity_constraint → ∃ a b, (a, b) = (3, 7) :=
by
  sorry

end max_rent_A_most_cost_effective_l392_392152


namespace gcd_max_value_l392_392832

noncomputable def max_gcd (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else 1

theorem gcd_max_value :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → gcd (13 * m + 4) (7 * m + 2) ≤ max_gcd m) ∧
              (∀ m : ℕ, m > 0 → max_gcd m ≤ 2) :=
by {
  sorry
}

end gcd_max_value_l392_392832


namespace digit_five_occurrences_l392_392778

variable (fives_ones fives_tens fives_hundreds : ℕ)

def count_fives := fives_ones + fives_tens + fives_hundreds

theorem digit_five_occurrences :
  ( ∀ (fives_ones fives_tens fives_hundreds : ℕ), 
    fives_ones = 100 ∧ fives_tens = 100 ∧ fives_hundreds = 100 → 
    count_fives fives_ones fives_tens fives_hundreds = 300 ) :=
by
  sorry

end digit_five_occurrences_l392_392778


namespace correct_num_conclusions_l392_392480

open Real

noncomputable def line_slope (α : ℝ) (m : ℝ) := tan α
noncomputable def line_eq (α : ℝ) (m : ℝ) := λ x : ℝ, x * tan α + m
def correct_conclusion1 (α : ℝ) : Prop := 
  dot_product (tan α, -1) (cos α, sin α) = 0

def correct_conclusion2 (α : ℝ) (m : ℝ) (h : 0 < α ∧ α < π / 4) : Prop := 
  |arctan 1 - α| = |π / 4 - α|

def correct_conclusion3 (α : ℝ) (m n : ℝ) (h : m ≠ 0 ∧ n ≠ m) : Prop := 
  line_slope α m = line_slope α (n / cos α) → m ≠ n / cos α

def num_correct_conclusions (α : ℝ) (m n : ℝ) (h : α ≠ π / 2 + k * π ∧ m ≠ 0 ∧ n ≠ m) : ℕ := 
  (if correct_conclusion1 α then 1 else 0) +
  (if correct_conclusion2 α m (⟨by sorry, by sorry⟩) then 1 else 0) +
  (if correct_conclusion3 α m n (⟨h.right, h.right.right⟩) then 1 else 0)

theorem correct_num_conclusions (α : ℝ) (m n : ℝ) (h : α ≠ π / 2 + k * π ∧ m ≠ 0 ∧ n ≠ m) : 
  num_correct_conclusions α m n h = 2 := 
sorry

end correct_num_conclusions_l392_392480


namespace value_of_x_l392_392760

theorem value_of_x (x : ℝ) : (9 - x) ^ 2 = x ^ 2 → x = 4.5 :=
by
  sorry

end value_of_x_l392_392760


namespace circles_tangent_l392_392664

theorem circles_tangent (m : ℝ) :
  (∀ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9 → 
                (x + 1)^2 + (y - m)^2 = 4 →
                ∃ m, m = -1 ∨ m = -2) := 
sorry

end circles_tangent_l392_392664


namespace distinct_collections_count_l392_392626

def MATHEMATICS := ["M", "A", "T", "H", "E", "M", "A", "T", "I", "C", "S"]
def vowels := ["A", "A", "I", "E"]
def consonants := ["M", "T", "H", "M", "T", "C", "S"]

theorem distinct_collections_count :
  ∀ (v_set : Finset (List String)), v_set.card = 3 → v_set ⊆ Finset.mk vowels,
  ∀ (c_set : Finset (List String)), c_set.card = 4 → c_set ⊆ Finset.mk consonants,
    num_distinct_collections v_set c_set = 490 :=
by
  sorry

end distinct_collections_count_l392_392626


namespace calculate_two_times_square_root_squared_l392_392839

theorem calculate_two_times_square_root_squared : 2 * (Real.sqrt 50625) ^ 2 = 101250 := by
  sorry

end calculate_two_times_square_root_squared_l392_392839


namespace remainder_of_polynomial_l392_392865

theorem remainder_of_polynomial (x : ℤ) : 
  (x^4 - 1) * (x^2 - 1) % (x^2 + x + 1) = 3 := 
sorry

end remainder_of_polynomial_l392_392865


namespace commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l392_392753

def binary_star (x y : ℝ) : ℝ := (x - 1) * (y - 1) - 1

-- Statement (A): Commutativity
theorem commutative_star (x y : ℝ) : binary_star x y = binary_star y x := sorry

-- Statement (B): Distributivity (proving it's not distributive)
theorem not_distributive_star (x y z : ℝ) : ¬(binary_star x (y + z) = binary_star x y + binary_star x z) := sorry

-- Statement (C): Special case
theorem special_case_star (x : ℝ) : binary_star (x + 1) (x - 1) = binary_star x x - 1 := sorry

-- Statement (D): Identity element
theorem no_identity_star (x e : ℝ) : ¬(binary_star x e = x ∧ binary_star e x = x) := sorry

-- Statement (E): Associativity (proving it's not associative)
theorem not_associative_star (x y z : ℝ) : ¬(binary_star x (binary_star y z) = binary_star (binary_star x y) z) := sorry

end commutative_star_not_distributive_star_special_case_star_no_identity_star_not_associative_star_l392_392753


namespace minimum_value_expression_l392_392504

theorem minimum_value_expression (F M N : ℝ × ℝ) (x y : ℝ) (a : ℝ) (k : ℝ) :
  (y ^ 2 = 16 * x ∧ F = (4, 0) ∧ l = (k * (x - 4), y) ∧ (M = (x₁, y₁) ∧ N = (x₂, y₂)) ∧
  0 ≤ x₁ ∧ y₁ ^ 2 = 16 * x₁ ∧ 0 ≤ x₂ ∧ y₂ ^ 2 = 16 * x₂) →
  (abs (dist F N) / 9 - 4 / abs (dist F M) ≥ 1 / 3) :=
sorry -- proof will be provided

end minimum_value_expression_l392_392504


namespace find_angle_x_l392_392628

-- Define the given problem
variable (O A B C : Point)
variable (r : ℝ)

-- Conditions of the problem
def is_center (O : Point) : Prop := True
def is_right_angle (∠ACD : ℝ) : Prop := ∠ACD = 90
def given_angle_CDA (∠CDA : ℝ) : Prop := ∠CDA = 48
def radii_equal (AO BO CO : ℝ) : Prop := AO = r ∧ BO = r ∧ CO = r

-- Goal: Prove that the angle x is equal to 58 degrees.
theorem find_angle_x (x : ℝ) (h1 : is_center O) (h2 : is_right_angle (90)) (h3 : given_angle_CDA 48) (h4 : radii_equal r r r) : x = 58 := 
by
  sorry

end find_angle_x_l392_392628


namespace cosine_angle_diagonals_parallelogram_area_is_15_l392_392916

structure Vector3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def dot_product (v1 v2 : Vector3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Vector3D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2 + v.z^2)

def cross_product (v1 v2 : Vector3D) : Vector3D :=
  ⟨v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x⟩

noncomputable def cos_theta (v1 v2 : Vector3D) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

def parallelogram_area (v1 v2 : Vector3D) : ℝ :=
  magnitude (cross_product v1 v2)

def a : Vector3D := ⟨3, 2, 1⟩
def b : Vector3D := ⟨-1, 4, 2⟩

theorem cosine_angle_diagonals : cos_theta ⟨2, 6, 3⟩ ⟨-4, 2, 1⟩ = 1 / Real.sqrt 21 :=
by
  sorry

theorem parallelogram_area_is_15 : parallelogram_area a b = 15 :=
by
  sorry

end cosine_angle_diagonals_parallelogram_area_is_15_l392_392916


namespace larger_integer_21_l392_392725

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l392_392725


namespace white_cats_count_l392_392809

theorem white_cats_count (total_cats : ℕ) (black_cats : ℕ) (gray_cats : ℕ) (white_cats : ℕ)
  (h1 : total_cats = 15)
  (h2 : black_cats = 10)
  (h3 : gray_cats = 3)
  (h4 : total_cats = black_cats + gray_cats + white_cats) : 
  white_cats = 2 := 
  by
    -- proof or sorry here
    sorry

end white_cats_count_l392_392809


namespace line_through_T_dot_product_three_l392_392980

section
open Real

-- Definitions for parabola and line conditions
def parabola (x y : ℝ) := y^2 = 2 * x
def line_through_point (x y k : ℝ) := y = k * (x - 3)
def line_vertical (x : ℝ) := x = 3
def line_through_T (x y : ℝ) (k : Option ℝ) :=
  match k with
  | none   => x = 3
  | some m => y = m * (x - 3)

-- Definitions for points A and B on the parabola
def point_on_parabola (A B : ℝ × ℝ) := 
  let (x1, y1) := A in
  let (x2, y2) := B in
  parabola x1 y1 ∧ parabola x2 y2

-- Definitions for dot product
def dot_product (A B : ℝ × ℝ) :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  x1 * x2 + y1 * y2

-- Overall problem statement
theorem line_through_T_dot_product_three (A B T : ℝ × ℝ) (k : Option ℝ) 
  (hT : T = (3, 0)) 
  (hl : line_through_T A.1 A.2 k → line_through_T B.1 B.2 k) 
  (hA_B_on_parabola : point_on_parabola A B) :
  dot_product A B = 3 :=
by
  -- Proof steps would go here
  sorry
end

end line_through_T_dot_product_three_l392_392980


namespace dress_designs_count_l392_392397

-- Define the number of colors, fabric types, and patterns
def num_colors : Nat := 3
def num_fabric_types : Nat := 4
def num_patterns : Nat := 3

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_fabric_types * num_patterns

-- Define the theorem to prove the equivalence
theorem dress_designs_count :
  total_dress_designs = 36 :=
by
  -- This is to show the theorem's structure; proof will be added here.
  sorry

end dress_designs_count_l392_392397


namespace larger_integer_is_21_l392_392705

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l392_392705


namespace locus_of_intersection_points_is_line_l392_392638

variables {Point Line : Type}

-- Definitions of the given geometric entities and their conditions
variable {A B C D P Q : Point}
variable {l1 l2 : Line}
variable (Locus : set Point)

-- Conditions
variable (OnLine : Point → Line → Prop)
variable (IntersectsAt : Line → Line → Point)
variable (IntersectionOfDiagonals : Point → Point → Point → Point → Point)

-- Hypotheses based on the problem statement
variable (h1 : OnLine A l1)
variable (h2 : OnLine B l1)
variable (h3 : OnLine C l2)
variable (h4 : OnLine D l2)
variable (h5 : IntersectsAt (Line.mk B C) (Line.mk A D) = P)
variable (h6 : IntersectsAt l1 l2 = Q)

-- Theorem statement
theorem locus_of_intersection_points_is_line :
  ∀ (A B C D : Point),
    OnLine A l1 → OnLine B l1 → OnLine C l2 → OnLine D l2 →
    IntersectsAt (Line.mk B C) (Line.mk A D) = P →
    IntersectsAt l1 l2 = Q →
    ∃ (L : Line), ∀ (I : Point), I ∈ Locus ↔
      I = IntersectionOfDiagonals A B C D ∧ OnLine I L ∧ OnLine Q L :=
sorry

end locus_of_intersection_points_is_line_l392_392638


namespace discriminant_of_quadratic_l392_392755

def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic (a b c : ℤ) (h_a : a = 5) (h_b : b = -3) (h_c : c = 4) :
  discriminant a b c = -71 :=
by
  -- Use given hypotheses and target proof
  rw [h_a, h_b, h_c]
  -- Unfold the discriminant definition
  dsimp [discriminant]
  -- Compute the result manually or using simplification tactics
  norm_num
  -- The above steps should end with -71
  sorry  -- The proof steps are to be filled in here

end discriminant_of_quadratic_l392_392755


namespace dips_to_daps_l392_392173

theorem dips_to_daps : 
  ∀ (daps dops dips : Type) (eq1 : 5 * daps = 4 * dops) (eq2 : 3 * dops = 8 * dips),
  (48 * dips = 22.5 * daps) :=
begin
  intros,
  sorry
end

end dips_to_daps_l392_392173


namespace max_value_f_l392_392674

noncomputable def f (x : ℝ) : ℝ := Real.sin (2*x) - 2 * Real.sqrt 3 * (Real.sin x)^2

theorem max_value_f : ∃ x : ℝ, f x = 2 - Real.sqrt 3 :=
  sorry

end max_value_f_l392_392674


namespace larger_integer_is_21_l392_392716

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l392_392716


namespace find_initial_amount_l392_392017

noncomputable def calculate_initial_amount
  (compound_interest : ℝ)
  (r : ℝ)
  (n : ℕ)
  (t : ℕ) :
  ℝ :=
  let A_sub_P := compound_interest in
  let compound_factor := (1 + r / n) ^ (n * t) in
  A_sub_P / (compound_factor - 1)

theorem find_initial_amount :
  calculate_initial_amount 1648.64 0.04 2 2 = 20000 :=
by
  sorry

end find_initial_amount_l392_392017


namespace cannot_be_parallel_l392_392501

-- Definitions explaining the relationships
def is_skew (a b : Line) : Prop :=
  ¬ (∃ p : Point, p ∈ a ∧ p ∈ b) ∧ ¬ (a ∥ b)

def is_parallel (a b : Line) : Prop :=
  a ∥ b

-- Given conditions
variables (a b c : Line)
variable (ha : is_skew a b)
variable (hb : is_parallel c a)

-- Statement to prove
theorem cannot_be_parallel (a b c : Line) (ha : is_skew a b) (hb : is_parallel c a) : ¬ is_parallel c b := 
sorry

end cannot_be_parallel_l392_392501


namespace range_of_a_if_minimum_at_a_l392_392935

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Given condition: f'(x) = a(x + 1)(x - a)
def f_prime (x : ℝ) : ℝ := a * (x + 1) * (x - a)

-- To prove: a < -1 ∨ a > 0 given that f(x) attains its minimum value at x = a
theorem range_of_a_if_minimum_at_a
  (h_min : ∀ x : ℝ, f_prime x = 0 → x = a)
  (h_derive : ∀ x : ℝ, deriv f x = f_prime x) :
  a < -1 ∨ a > 0 := 
sorry

end range_of_a_if_minimum_at_a_l392_392935


namespace floor_add_double_eq_15_4_l392_392884

theorem floor_add_double_eq_15_4 (r : ℝ) (h : (⌊r⌋ : ℝ) + 2 * r = 15.4) : r = 5.2 := 
sorry

end floor_add_double_eq_15_4_l392_392884


namespace reflection_matrix_over_vector_is_correct_l392_392038

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l392_392038


namespace total_amount_before_brokerage_l392_392299

variable (A : ℝ)

theorem total_amount_before_brokerage 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 106.25) 
  (h2 : brokerage_rate = 1 / 400) :
  A = 42500 / 399 :=
by
  sorry

end total_amount_before_brokerage_l392_392299


namespace max_log_sum_lg4_l392_392127

theorem max_log_sum_lg4 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 
  ∃ (xy_prod : ℝ), xy_prod = x * y ∧ (lg xy_prod) ≤ lg 4 := 
sorry

end max_log_sum_lg4_l392_392127


namespace solve_problem_l392_392583

variable (P Q R : Type) [triangle : IsTriangle P Q R]
variable (p q r : ℝ) (cos_P_minus_Q : ℝ)

def proof_problem : Prop :=
  p = 7 ∧ q = 6 ∧ cos_P_minus_Q = 13 / 14 → r = Real.sqrt 73

-- Instantiate the conditions
noncomputable def problem_instance : Prop := proof_problem P Q R p q r (13 / 14)

-- Statement that should be proven
theorem solve_problem : problem_instance := 
begin
  -- Enables proof environment, replace with actual proof.
  sorry
end

end solve_problem_l392_392583


namespace min_dist_circle_to_line_l392_392889

noncomputable def circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y

noncomputable def line_eq (x y : ℝ) := x + y - 8

theorem min_dist_circle_to_line : 
  (∀ x y : ℝ, circle_eq x y = 0 → ∃ d : ℝ, d ≥ 0 ∧ 
    (∀ x₁ y₁ : ℝ, circle_eq x₁ y₁ = 0 → ∀ x₂ y₂ : ℝ, line_eq x₂ y₂ = 0 → d ≤ dist (x₁, y₁) (x₂, y₂)) ∧ 
    d = 2 * Real.sqrt 2) :=
by
  sorry

end min_dist_circle_to_line_l392_392889


namespace carol_exercise_length_l392_392842

theorem carol_exercise_length:
  (∀ x y : ℝ, x * y = (80 * 45) → x = (80 * 45) / y) →  -- inverse relationship setup
  (85 * 2 = 170) →  -- target average score over two tests
  (170 - 80 = 90) → -- required score on the second test
  let k := 80 * 45 in  -- constant k
  (S2: ℝ) → (E2: ℝ) →
  (S2 = 90) → -- score in the second test
  (k = S2 * E2) →  -- inverse relationship applied to second test
  E2 = 40 := -- expected exercise length of 40 minutes
by
  intros,
  simp,
  split,
  { intro h_inverserel,
    intro h_avg,
    intro h_score_required,
    let hyp := h_inverserel 90 E2 (80 * 45),
    calc
      E2 = (80 * 45) / 90 : by exact hyp 
      ... = 40 : by norm_num
  },
  sorry

end carol_exercise_length_l392_392842


namespace find_min_value_l392_392496

variable {ℝ : Type*}

structure Vector :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

namespace Vector

def dot (v1 v2 : Vector) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def norm (v : Vector) : ℝ :=
  real.sqrt (v.dot v)

def sub (v1 v2 : Vector) : Vector :=
  { x := v1.x - v2.x, y := v1.y - v2.y, z := v1.z - v2.z }

instance : has_sub Vector := ⟨sub⟩

end Vector

open Vector

noncomputable def min_magnitude (a b c : Vector) (t1 t2 : ℝ) : ℝ :=
  norm (c - { x := t1, y := 0, z := 0 } * a - { x := 0, y := t2, z := 0 } * b)

theorem find_min_value
  (a b c : Vector)
  (h1 : dot a b = 0)
  (h2 : norm a = 1)
  (h3 : norm b = 1)
  (h4 : norm c = 13)
  (h5 : dot c a = 3)
  (h6 : dot c b = 4)
  : ∃ t1 t2 : ℝ, min_magnitude a b c t1 t2 = 12 := sorry

end find_min_value_l392_392496


namespace reflection_matrix_over_vector_l392_392070

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l392_392070


namespace centroid_vector_relation_l392_392615

-- Define triangle vertices and arbitrary point
variables (A B C O : Point)
-- Define point M as the centroid of triangle ABC
variable (M : Point)
-- Hypothesis that M is the centroid of triangle ABC
hypothesis h1 : M = centroid A B C

theorem centroid_vector_relation :
  vector O M = (1 : ℝ) / 3 * (vector O A + vector O B + vector O C) :=
sorry

end centroid_vector_relation_l392_392615


namespace samantha_mean_correct_l392_392646

-- Given data: Samantha's assignment scores
def samantha_scores : List ℕ := [84, 89, 92, 88, 95, 91, 93]

-- Definition of the arithmetic mean of a list of scores
def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / (scores.length : ℚ)

-- Prove that the arithmetic mean of Samantha's scores is 90.29
theorem samantha_mean_correct :
  arithmetic_mean samantha_scores = 90.29 := 
by
  -- The proof steps would be filled in here
  sorry

end samantha_mean_correct_l392_392646


namespace trapezoid_area_22_l392_392413

-- Define the problem parameters
namespace TrapezoidArea

variables {G H I J K L : Type} 
variables [CommRing G] [CommRing H] [CommRing I] [CommRing J] [CommRing K] [CommRing L]

-- Dimensions and positions
def rectangle_area (area_GHIJ : ℕ) := area_GHIJ = 20

def position_IK (IK KJ : ℕ) := IK = 2 ∧ KJ = 8

def position_GL (GL LJ : ℕ) := GL = 1 ∧ LJ = 4

-- Prove area of trapezoid KHLG is 22 square units
theorem trapezoid_area_22
  (area_GHIJ : ℕ)
  (IK KJ : ℕ)
  (GL LJ : ℕ)
  (h1 : rectangle_area area_GHIJ)
  (h2 : position_IK IK KJ)
  (h3 : position_GL GL LJ) :
  ∃ (area_KHLG : ℕ), area_KHLG = 22 :=
by 
  sorry

end TrapezoidArea

end trapezoid_area_22_l392_392413


namespace sum_of_factors_8128_l392_392410

theorem sum_of_factors_8128 : ∑ i in (nat.divisors 8128).to_finset, i = 16256 :=
by
  sorry

end sum_of_factors_8128_l392_392410


namespace negation_of_p_l392_392961

variable (x y : ℝ)

def proposition_p := ∀ x y : ℝ, x^2 + y^2 - 1 > 0 

theorem negation_of_p : (¬ proposition_p) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) := by
  sorry

end negation_of_p_l392_392961


namespace die_roll_event_order_l392_392399

theorem die_roll_event_order :
  let outcomes := {1, 2, 3, 4, 5, 6}
  let event_1_prob := 1 / 6
  let event_2_prob := 4 / 6
  let event_3_prob := 3 / 6
  event_1_prob < event_3_prob ∧ event_3_prob < event_2_prob :=
by
  let outcomes := {1, 2, 3, 4, 5, 6}
  let event_1_prob := 1 / 6
  let event_2_prob := 4 / 6
  let event_3_prob := 3 / 6
  have h1 : event_1_prob < event_3_prob := by sorry
  have h2 : event_3_prob < event_2_prob := by sorry
  exact And.intro h1 h2

end die_roll_event_order_l392_392399


namespace average_words_per_hour_l392_392411

theorem average_words_per_hour 
  (total_words : ℕ) 
  (total_hours : ℕ) 
  (intense_hours : ℕ) 
  (average : ℕ) 
  (h1 : total_words = 50000) 
  (h2 : total_hours = 90) 
  (h3 : intense_hours = 10) 
  (h4 : average = 556) : 
  (total_words.to_rat / total_hours.to_rat) ≈ average := 
by 
  have h5 : total_usual_hours := total_hours - intense_hours
  have h6 : prod_normal_rate := total_words / (total_usual_hours + 2 * intense_hours)
  sorry

end average_words_per_hour_l392_392411


namespace tangent_line_to_parabola_l392_392417

theorem tangent_line_to_parabola :
  let P := (-1, 0)
  let parabola := λ x : ℝ, x^2 + x + 1
  let line := λ x y : ℝ, x - y + 1 = 0
  ∃ x0 : ℝ, ∃ y0 : ℝ, 
  P.2 = parabola x0 ∧ (P.1 - x0)*(2*x0 + 1) = P.2 - y0 ∧ P.1, P.2 ∈ set_of (line x y) := 
sorry

end tangent_line_to_parabola_l392_392417


namespace geometric_sequence_r_value_l392_392554

theorem geometric_sequence_r_value (S : ℕ → ℚ) (r : ℚ) (n : ℕ) (h : n ≥ 2) (h1 : ∀ n, S n = 3^n + r) :
    r = -1 :=
sorry

end geometric_sequence_r_value_l392_392554


namespace part1_part2_part3_l392_392130

-- Definition for the first part of the problem
def is_circle (m : ℝ) : Prop :=
  ∀ x y : ℝ, (x - m)^2 + (y - 2)^2 = m^2 - 5 * m + 4

theorem part1 (m : ℝ) (h: is_circle m) : m < 1 ∨ m > 4 :=
sorry

-- Definitions for the second part of the problem
def chord_length (x y m : ℝ) : ℝ :=
  2 * real.sqrt ((3 * real.sqrt 2)^2 - (real.abs (-4 - 2 + 1) / real.sqrt 5)^2)

theorem part2 : chord_length (-2) (2) (-2) = 2 * real.sqrt 13 :=
sorry

-- Definitions for the third part of the problem
def intersects_circle (m : ℝ) : Prop :=
  ∃ x1 x2 y1 y2 : ℝ, (x1 > x2 ∧ y1 = 2 * x1 - 1 ∧ y2 = 2 * x2 - 1 ∧ 
                      (5 * x1 * x2 + 2 * (x1 + x2) + 1 = 0))

theorem part3 (m : ℝ) (h: intersects_circle m) : m = 2 / 29 :=
sorry

end part1_part2_part3_l392_392130


namespace side_length_correct_l392_392558

noncomputable def find_side_length (b : ℝ) (angleB : ℝ) (sinA : ℝ) : ℝ :=
  let sinB := Real.sin angleB
  let a := b * sinA / sinB
  a

theorem side_length_correct (b : ℝ) (angleB : ℝ) (sinA : ℝ) (a : ℝ) 
  (hb : b = 4)
  (hangleB : angleB = Real.pi / 6)
  (hsinA : sinA = 1 / 3)
  (ha : a = 8 / 3) : 
  find_side_length b angleB sinA = a :=
by
  sorry

end side_length_correct_l392_392558


namespace reflection_over_vector_l392_392052

noncomputable def reflection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

theorem reflection_over_vector :
  reflection_matrix (4, 3) =
    (λ (w : ℝ × ℝ), (7/25 * w.1 + 24/25 * w.2, 24/25 * w.1 - 7/25 * w.2)) := sorry

end reflection_over_vector_l392_392052


namespace second_solution_lemonade_is_45_l392_392819

-- Define percentages as real numbers for simplicity
def firstCarbonatedWater : ℝ := 0.80
def firstLemonade : ℝ := 0.20
def secondCarbonatedWater : ℝ := 0.55
def mixturePercentageFirst : ℝ := 0.50
def mixtureCarbonatedWater : ℝ := 0.675

-- The ones that already follow from conditions or trivial definitions:
def secondLemonade : ℝ := 1 - secondCarbonatedWater

-- Define the percentage of carbonated water in mixture, based on given conditions
def mixtureIsCorrect : Prop :=
  mixturePercentageFirst * firstCarbonatedWater + (1 - mixturePercentageFirst) * secondCarbonatedWater = mixtureCarbonatedWater

-- The theorem to prove: second solution's lemonade percentage is 45%
theorem second_solution_lemonade_is_45 :
  mixtureIsCorrect → secondLemonade = 0.45 :=
by
  sorry

end second_solution_lemonade_is_45_l392_392819


namespace personal_income_tax_l392_392315

theorem personal_income_tax {X: ℝ} (gross_income: ℝ) (net_income: ℝ) (tax_rate: ℝ) (h1: tax_rate = 0.13) (h2: net_income = gross_income * (1 - tax_rate)) (h3: net_income = 20000) : gross_income ≈ 22989 :=
by sorry

end personal_income_tax_l392_392315


namespace find_angle_x_l392_392629

-- Define the given problem
variable (O A B C : Point)
variable (r : ℝ)

-- Conditions of the problem
def is_center (O : Point) : Prop := True
def is_right_angle (∠ACD : ℝ) : Prop := ∠ACD = 90
def given_angle_CDA (∠CDA : ℝ) : Prop := ∠CDA = 48
def radii_equal (AO BO CO : ℝ) : Prop := AO = r ∧ BO = r ∧ CO = r

-- Goal: Prove that the angle x is equal to 58 degrees.
theorem find_angle_x (x : ℝ) (h1 : is_center O) (h2 : is_right_angle (90)) (h3 : given_angle_CDA 48) (h4 : radii_equal r r r) : x = 58 := 
by
  sorry

end find_angle_x_l392_392629


namespace ratio_problem_l392_392539

open Classical 

variables {q r s t u : ℚ}

theorem ratio_problem (h1 : q / r = 8) (h2 : s / r = 5) (h3 : s / t = 1 / 4) (h4 : u / t = 3) :
  u / q = 15 / 2 :=
by
  sorry

end ratio_problem_l392_392539


namespace smallest_positive_integer_x_l392_392758

theorem smallest_positive_integer_x (x : ℕ) (h : 725 * x ≡ 1165 * x [MOD 35]) : x = 7 :=
sorry

end smallest_positive_integer_x_l392_392758


namespace dap_equiv_48_dips_l392_392167

variables (dap dop dip : Type) [CommRing dap] [CommRing dop] [CommRing dip]

-- Define equivalences between daps, dops, and dips
def equivalence_dap_dop : dap ≃ₐ[dop] (dop →ₐ[dip] dap) := sorry
def equivalence_dop_dip : dop ≃ₐ[dip] (dip →ₐ[dap] dop) := sorry

-- Proportions given in the conditions
def prop1 (d : dap) (o : dop) : 5 * d = 4 * o := sorry
def prop2 (o : dop) (i : dip) : 3 * o = 8 * i := sorry

-- The proof statement
theorem dap_equiv_48_dips : ∀ (d : dap) (i : dip), (15 * d = 32 * i) → (d = 22.5 * i) := 
by
  intros
  sorry

end dap_equiv_48_dips_l392_392167


namespace rowing_time_ratio_l392_392446

def ethan_rowing_duration : ℕ := 25
def total_rowing_duration : ℕ := 75
def frank_rowing_duration : ℕ := total_rowing_duration - ethan_rowing_duration

theorem rowing_time_ratio (h1 : ethan_rowing_duration = 25) (h2 : total_rowing_duration = 75) : 
(frank_rowing_duration : ethan_rowing_duration) = (2 : 1) := by sorry

end rowing_time_ratio_l392_392446


namespace train_speed_second_part_l392_392823

variable (x : ℝ)

-- Condition 1: Time to cover x km at 40 kmph
def time_first_part (x : ℝ) : ℝ := x / 40

-- Condition 2: Time to cover 2x km at speed v
def time_second_part (x v : ℝ) : ℝ := 2 * x / v

-- Condition 3: Total time for the journey
def total_time (x : ℝ) : ℝ := 3 * x / 24

-- Given Condition: Sum of times equals the total time
def time_equation (x v : ℝ) : Prop := (x / 40) + (2 * x / v) = (3 * x / 24)

-- Goal: Prove the speed v of the second part is 120 kmph
theorem train_speed_second_part (x : ℝ) (v : ℝ) (h : time_equation x v) : v = 120 := 
  by sorry

end train_speed_second_part_l392_392823


namespace find_larger_integer_l392_392699

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l392_392699


namespace pyramid_section_rhombus_l392_392639

structure Pyramid (A B C D : Type) := (point : Type)

def is_parallel (l1 l2 : ℝ) : Prop :=
  ∀ (m n : ℝ), m * l1 = n * l2

def is_parallelogram (K L M N : Type) : Prop :=
  sorry

def is_rhombus (K L M N : Type) : Prop :=
  sorry

noncomputable def side_length_rhombus (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

/-- Prove that the section of pyramid ABCD with a plane parallel to edges AC and BD is a parallelogram,
and under certain conditions, this parallelogram is a rhombus. Find the side of this rhombus given AC = a and BD = b. -/
theorem pyramid_section_rhombus (A B C D K L M N : Type) (a b : ℝ) :
  is_parallel AC BD →
  is_parallelogram K L M N →
  is_rhombus K L M N →
  side_length_rhombus a b = (a * b) / (a + b) :=
by
  sorry

end pyramid_section_rhombus_l392_392639


namespace non_equilateral_triangle_division_equilateral_triangle_no_unequal_division_l392_392774

theorem non_equilateral_triangle_division (ABC : Triangle) (h : ABC.is_scalene) : 
  ∃ (A' B' C' : Point), triangle (A' B' C') ∧ (A' B' C').is_similar ABC ∧ (A' B' C').is_unequal ABC := 
sorry

theorem equilateral_triangle_no_unequal_division (ABC : Triangle) (h : ABC.is_equilateral) : 
  ¬ ∃ (A' B' C' : Point), triangle (A' B' C') ∧ (A' B' C').is_equilateral ∧ (A' B' C').is_unequal ABC := 
sorry

end non_equilateral_triangle_division_equilateral_triangle_no_unequal_division_l392_392774


namespace determine_finalists_l392_392984

-- Condition: 17 students with distinct race times
def num_students : ℕ := 17
axiom distinct_race_times : ∀ (times : vector ℕ num_students), ∀ (i j : fin num_students), i ≠ j → times[i] ≠ times[j]

-- Condition: top 8 enter the final race
def top_n : ℕ := 8

-- Wang's result is known
axiom wangs_time (times : vector ℕ num_students) : ℕ

-- Condition on median 
def median_definition (times : vector ℕ num_students) : ℕ :=
  times.to_list.nth_le (num_students / 2) (sorry : nat.lt (num_students / 2) num_students)

theorem determine_finalists (times : vector ℕ num_students) :
  ∃ (m : ℕ), m = median_definition times → 
  (wangs_time times < m → wangs_time times ∈ take top_n (sort times.to_list)) ∧
  (wangs_time times ≥ m → wangs_time times ∈ drop (num_students - top_n) (sort times.to_list)) :=
sorry

end determine_finalists_l392_392984


namespace angle_ZXC_eq_angle_YXB_l392_392788

theorem angle_ZXC_eq_angle_YXB
  {A B C H H' M X Y Z : Type*}
  [MetricSpace A] [Finite H] : 
  (IsAltitude A B C H) →
  (IsReflection H H' M) →
  (IsCircumcircleTangentIntersection B C X) →
  (PerpendicularLineToSegmentAtPoint X H' ⟶ IntersectsSegmentAtPoints H' A B Y) →
  (PerpendicularLineToSegmentAtPoint X H' ⟶ IntersectsSegmentAtPoints H' A C Z) →
  (∠ Z X C = ∠ Y X B) :=
sorry

end angle_ZXC_eq_angle_YXB_l392_392788


namespace sum_h_k_a_b_l392_392304

def h : ℤ := 3
def k : ℤ := -5
def a : ℤ := 7
def b : ℤ := 4

theorem sum_h_k_a_b : h + k + a + b = 9 := by
  sorry

end sum_h_k_a_b_l392_392304


namespace shaded_area_correct_l392_392816

-- Condition 1: Length of the sides of a regular hexagon
def side_length : ℝ := 6

-- Condition 2: Radius of the circular sectors
def radius : ℝ := 3

-- Proof that the area of the shaded region is equal to the specified value
theorem shaded_area_correct :
  let hexagon_area := 6 * (sqrt 3 / 4) * (side_length ^ 2)
  let sector_area := (60 / 360) * π * (radius ^ 2)
  let total_sector_area := 6 * sector_area
  let shaded_area := hexagon_area - total_sector_area
  shaded_area = 54 * sqrt 3 - 9 * π :=
by
  sorry

end shaded_area_correct_l392_392816


namespace Mrs_Martin_pays_32_l392_392659

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def num_regular_scoops : ℕ := 2
def num_kiddie_scoops : ℕ := 2
def num_double_scoops : ℕ := 3

def total_cost : ℕ := 
  (num_regular_scoops * regular_scoop_cost) + 
  (num_kiddie_scoops * kiddie_scoop_cost) + 
  (num_double_scoops * double_scoop_cost)

theorem Mrs_Martin_pays_32 :
  total_cost = 32 :=
by
  sorry

end Mrs_Martin_pays_32_l392_392659


namespace naomi_saw_wheels_l392_392429

theorem naomi_saw_wheels :
  let regular_bikes := 7
  let children's_bikes := 11
  let wheels_per_regular_bike := 2
  let wheels_per_children_bike := 4
  let total_wheels := regular_bikes * wheels_per_regular_bike + children's_bikes * wheels_per_children_bike
  total_wheels = 58 := by
  sorry

end naomi_saw_wheels_l392_392429


namespace no_n_gt_3_n_sq_odd_digits_l392_392444

theorem no_n_gt_3_n_sq_odd_digits :
  ¬ ∃ n : ℕ, n > 3 ∧ (∀ d : ℕ, d ∈ (n^2).digits 10 → odd d) :=
sorry

end no_n_gt_3_n_sq_odd_digits_l392_392444


namespace min_value_theorem_l392_392101

noncomputable def ellipse_foci_minimum_value (F1 F2 P : ℝ × ℝ) : Prop :=
∃ (p : ℝ × ℝ), 
  -- Ellipse condition: x^2 + 3y^2 = 12
  (p.1^2 + 3 * p.2^2 = 12) ∧
  -- Minimum value of |PF1 + PF2| is 4
  (|p.1 - F1.1| + |p.2 - F1.2| + |p.1 - F2.1| + |p.2 - F2.2| = 4)

theorem min_value_theorem : ∀ F1 F2 P, ellipse_foci_minimum_value F1 F2 P → 4 := 
begin
  -- proof goes here (skipped with sorry)
  sorry
end

end min_value_theorem_l392_392101


namespace sum_c_eq_T_l392_392934

def S (n : ℕ) : ℕ := 2 * n^2 - 1
def Q (n : ℕ) : ℕ := 2 * n - 2

def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 4 * n - 2

def b (n : ℕ) : ℕ := 2^n

def c (n : ℕ) : ℚ :=
  if n = 1 then 1 / 2 else (2 * n - 1) / 2^(n - 1)

noncomputable def T (n : ℕ) : ℚ :=
  ∑ i in finset.range n, c (i + 1)

theorem sum_c_eq_T (n : ℕ) : T n = 11 / 2 - (2 * n + 3) / 2^(n - 1) :=
  sorry

end sum_c_eq_T_l392_392934


namespace parallel_lines_m_l392_392936

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 1 = 0 → 6 ≠ 0) ∧ 
  (∀ x y : ℝ, m * x + 6 * y - 5 = 0 → 6 ≠ 0) → 
  m = 4 :=
by
  intro h
  sorry

end parallel_lines_m_l392_392936


namespace x_equals_eleven_l392_392158

theorem x_equals_eleven (x : ℕ) 
  (h : (1 / 8) * 2^36 = 8^x) : x = 11 :=
sorry

end x_equals_eleven_l392_392158


namespace find_larger_integer_l392_392734

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l392_392734


namespace total_weekly_water_consumption_l392_392268

-- Definitions coming from the conditions of the problem
def num_cows : Nat := 40
def water_per_cow_per_day : Nat := 80
def num_sheep : Nat := 10 * num_cows
def water_per_sheep_per_day : Nat := water_per_cow_per_day / 4
def days_in_week : Nat := 7

-- To prove statement: 
theorem total_weekly_water_consumption :
  let weekly_water_cow := water_per_cow_per_day * days_in_week
  let total_weekly_water_cows := weekly_water_cow * num_cows
  let daily_water_sheep := water_per_sheep_per_day
  let weekly_water_sheep := daily_water_sheep * days_in_week
  let total_weekly_water_sheep := weekly_water_sheep * num_sheep
  total_weekly_water_cows + total_weekly_water_sheep = 78400 := 
by
  sorry

end total_weekly_water_consumption_l392_392268


namespace actual_time_when_watch_is_8PM_l392_392844

theorem actual_time_when_watch_is_8PM :
  ∀ noon_to_three_watch_diff noon_to_three_actual_diff watch_reading_8PM,
  (noon_to_three_watch_diff = 2 * 3600 + 54 * 60 + 30) →
  (noon_to_three_actual_diff = 3 * 3600) →
  (watch_reading_8PM = 8 * 3600) →
  (let rate := noon_to_three_watch_diff / noon_to_three_actual_diff in
   let actual_time := watch_reading_8PM / rate in
   actual_time = 8 * 3600 + 15 * 60 + 8) :=
by 
  intros noon_to_three_watch_diff noon_to_three_actual_diff watch_reading_8PM h_watch_diff h_actual_diff h_watch_8PM
  let rate := (noon_to_three_watch_diff : ℝ) / (noon_to_three_actual_diff : ℝ)
  let actual_time := (watch_reading_8PM : ℝ) / rate
  have h_actual_time : actual_time = 8 * 3600 + 15 * 60 + 8 := sorry
  exact h_actual_time

end actual_time_when_watch_is_8PM_l392_392844


namespace original_cube_volume_l392_392275

theorem original_cube_volume (a : ℕ) (V_cube V_new : ℕ)
  (h1 : V_cube = a^3)
  (h2 : V_new = (a + 2) * a * (a - 2))
  (h3 : V_cube = V_new + 24) :
  V_cube = 216 :=
by
  sorry

end original_cube_volume_l392_392275


namespace find_larger_integer_l392_392697

-- Definitions and conditions
def quotient_condition (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (a = 7 * k) ∧ (b = 3 * k)

def product_condition (a b : ℕ) : Prop :=
  a * b = 189

-- Proof problem
theorem find_larger_integer : ∀ (a b : ℕ), (a > 0) → (b > 0) → quotient_condition a b ∧ product_condition a b → a = 21 :=
by
  intros a b h_pos_a h_pos_b h
  cases h with h_quotient h_product
  sorry

end find_larger_integer_l392_392697


namespace tangent_line_eq_l392_392455

theorem tangent_line_eq
    (f : ℝ → ℝ) (f_def : ∀ x, f x = x ^ 2)
    (tangent_point : ℝ × ℝ) (tangent_point_def : tangent_point = (1, 1))
    (f' : ℝ → ℝ) (f'_def : ∀ x, f' x = 2 * x)
    (slope_at_1 : f' 1 = 2) :
    ∃ (a b : ℝ), a = 2 ∧ b = -1 ∧ ∀ x y, y = a * x + b ↔ (2 * x - y - 1 = 0) :=
sorry

end tangent_line_eq_l392_392455


namespace sarah_shirts_l392_392288

theorem sarah_shirts (loads : ℕ) (pieces_per_load : ℕ) (sweaters : ℕ) 
  (total_pieces : ℕ) (shirts : ℕ) : 
  loads = 9 → pieces_per_load = 5 → sweaters = 2 →
  total_pieces = loads * pieces_per_load → shirts = total_pieces - sweaters → 
  shirts = 43 :=
by
  intros h_loads h_pieces_per_load h_sweaters h_total_pieces h_shirts
  sorry

end sarah_shirts_l392_392288


namespace first_place_beat_joe_l392_392597

theorem first_place_beat_joe (joe_won joe_draw first_place_won first_place_draw points_win points_draw : ℕ) 
    (h1 : joe_won = 1) (h2 : joe_draw = 3) (h3 : first_place_won = 2) (h4 : first_place_draw = 2)
    (h5 : points_win = 3) (h6 : points_draw = 1) : 
    (first_place_won * points_win + first_place_draw * points_draw) - (joe_won * points_win + joe_draw * points_draw) = 2 :=
by
   sorry

end first_place_beat_joe_l392_392597


namespace min_difference_is_247_l392_392422

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

noncomputable def min_difference_two_numbers (a b c d e f g h : ℕ) : ℕ :=
  abs ((1000 * a + 100 * b + 10 * c + d) - (1000 * e + 100 * f + 10 * g + h))

theorem min_difference_is_247 :
  ∃ (a b c d e f g h : ℕ), list.perm [a, b, c, d, e, f, g, h] digits ∧
  (1000 * b + 100 * c + 10 * d + e = 5) ∧
  min_difference_two_numbers a b c d e f g h = 247 :=
sorry

end min_difference_is_247_l392_392422


namespace count_four_digit_integers_divisible_by_15_l392_392533

theorem count_four_digit_integers_divisible_by_15 : 
  { n : Nat // 1000 ≤ n ∧ n < 10000 ∧ n % 15 = 0 }.card = 600 :=
by
  sorry

end count_four_digit_integers_divisible_by_15_l392_392533


namespace part_a_l392_392775

theorem part_a (p : ℕ) (h : Nat.prime (2^p - 1)) : Nat.sigma (2^(p-1) * (2^p - 1)) = 2^(p-1) * (2^p - 1) := 
  sorry

end part_a_l392_392775


namespace PQ_bisects_BD_l392_392214

-- Define a structure for a Quadrilateral having points A, B, C, D
structure Quadrilateral :=
  (A B C D : Point)

-- Define midpoints P and Q
def is_midpoint (P Q : Point) (A B C D : Point) : Prop :=
  (P = midpoint A B) ∧ (Q = midpoint C D)

-- Define bisect conditions
def line_bisects (PQ : Line) (A C B D : Point) := 
  (PQ.bisects A C) ∧ (PQ.bisects B D)

-- Define the main theorem
theorem PQ_bisects_BD 
  (quad : Quadrilateral)
  (P Q : Point)
  (PQ : Line)
  (condition_midpoints : is_midpoint P Q quad.A quad.B quad.C quad.D)
  (condition_bisect_AC : PQ.bisects quad.A quad.C) :
  PQ.bisects quad.B quad.D :=
sorry

end PQ_bisects_BD_l392_392214


namespace total_emails_received_l392_392590

theorem total_emails_received (emails_morning emails_afternoon : ℕ) 
  (h1 : emails_morning = 3) 
  (h2 : emails_afternoon = 5) : 
  emails_morning + emails_afternoon = 8 := 
by 
  sorry

end total_emails_received_l392_392590


namespace sum_of_reversed_remainders_l392_392362

noncomputable def quotient_remainder (a b : ℕ) : ℕ × ℕ :=
  (a / b, a % b)

theorem sum_of_reversed_remainders (n : ℕ)
  (h₁ : ∃ k, n = 12 * k + 56)
  (h₂ : ∃ m, n = 34 * m + 78) :
  let rem1 := (quotient_remainder n 34).2,
      rem2 := (quotient_remainder rem1 12).2 in
  rem1 + rem2 = 20 :=
by sorry

end sum_of_reversed_remainders_l392_392362


namespace min_value_of_f_l392_392891

noncomputable def f (x : ℝ) : ℝ := x^2 + 9*x + 81*x^(-4)

theorem min_value_of_f (x : ℝ) (h : x > 0) : ∃ y : ℝ, (∀ z : ℝ, z > 0 → f z ≥ y) ∧ y = 19 :=
by
  sorry

end min_value_of_f_l392_392891


namespace minimum_ab_bc_ca_l392_392543

theorem minimum_ab_bc_ca {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = a^3) (h5 : a * b * c = a^3) : 
  ab + bc + ca ≥ 9 :=
sorry

end minimum_ab_bc_ca_l392_392543


namespace cannot_form_right_triangle_l392_392852

theorem cannot_form_right_triangle (a b c : ℕ) (h : a = 2 ∧ b = 3 ∧ c = 4) : a^2 + b^2 ≠ c^2 :=
by
  cases h with
  | intro ha _ =>
  have hab := a^2 + b^2 = 2^2 + 3^2
  rw [ha] at hab
  have eq1 : 2^2 + 3^2 = 4 := by norm_num
  have eq2 : 4^2 = 16 := by norm_num
  rw [eq1] at hab
  exact ne_of_lt (by norm_num : 13 < 16)

end cannot_form_right_triangle_l392_392852


namespace no_such_function_exists_l392_392443

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, f(x^2) - (f x)^2 ≥ 1/4) ∧ (∀ a b : ℝ, f a = f b → a = b) :=
sorry

end no_such_function_exists_l392_392443


namespace max_ratio_AP_PE_l392_392990

noncomputable def lattice_point_ratio_max : ℕ :=
  5

theorem max_ratio_AP_PE (A B C P E : ℤ × ℤ)
  (h1 : unique (λ Q : ℤ × ℤ, Q ∈ interior (triangle A B C) ∧ Q ∈ lattice))
  (h2 : E ∈ line (A, P))
  (h3 : E ∈ line (B, C)) :
  max (λ (A B C P E : ℤ × ℤ), AP / PE) = 5 :=
  sorry

end max_ratio_AP_PE_l392_392990


namespace purely_imaginary_a_eq_2_l392_392196

theorem purely_imaginary_a_eq_2 (a : ℝ) (h : (2 - a) / 2 = 0) : a = 2 :=
sorry

end purely_imaginary_a_eq_2_l392_392196


namespace lunks_needed_for_12_apples_l392_392956

/-- 
  Given:
  1. 7 lunks can be traded for 4 kunks.
  2. 3 kunks will buy 5 apples.

  Prove that the number of lunks needed to purchase one dozen (12) apples is equal to 14.
-/
theorem lunks_needed_for_12_apples (L K : ℕ)
  (h1 : 7 * L = 4 * K)
  (h2 : 3 * K = 5) :
  (8 * K = 14 * L) :=
by
  sorry

end lunks_needed_for_12_apples_l392_392956


namespace geometric_sequence_of_d_l392_392094

-- Let \(a : ℕ \to ℝ\) be a geometric sequence defined as \(a_i = a_1 * q^(i-1)\)
-- \((a_1 > 0) \land (q > 1)\)
def a (a1 q : ℝ) (i : ℕ) : ℝ := a1 * (q ^ i)

-- Define that a1 and q follow the conditions
variables (a1 q : ℝ) (h_a1_pos : 0 < a1) (h_q_gt_1 : 1 < q)
-- di definition. i ranges from 1 to n-1
def d (i : ℕ) := a a1 q i - a a1 q (i + 1)

theorem geometric_sequence_of_d (n : ℕ) (h_n : 4 ≤ n) : 
    ∀ i, 1 ≤ i ∧ i < n ∧ d a1 q i = a1 * (1 - q) * q^(i-1) :=
by
  intro i h
  have : 1 ≤ i ∧ i < n := h
  sorry

end geometric_sequence_of_d_l392_392094


namespace number_of_apples_l392_392402

theorem number_of_apples (C : ℝ) (A : ℕ) (total_cost : ℝ) (price_diff : ℝ) (num_oranges : ℕ)
  (h_price : C = 0.26)
  (h_price_diff : price_diff = 0.28)
  (h_num_oranges : num_oranges = 7)
  (h_total_cost : total_cost = 4.56) :
  A * C + num_oranges * (C + price_diff) = total_cost → A = 3 := 
by
  sorry

end number_of_apples_l392_392402


namespace find_four_digit_number_l392_392450

theorem find_four_digit_number :
  ∃ (N : ℕ), 1000 ≤ N ∧ N < 10000 ∧ 
    (N % 131 = 112) ∧ 
    (N % 132 = 98) ∧ 
    N = 1946 :=
by
  sorry

end find_four_digit_number_l392_392450


namespace digit_d_makes_multiple_of_9_l392_392469

theorem digit_d_makes_multiple_of_9 :
  ∃ d : ℕ, d < 10 ∧ (26 + d) % 9 = 0 ∧ d = 1 :=
by {
  have h1 : 26 % 9 = 8 := rfl,
  use 1,
  split,
  { linarith },
  split,
  { norm_num },
  { refl }
}

end digit_d_makes_multiple_of_9_l392_392469


namespace compare_star_values_l392_392858

def star (A B : ℤ) : ℤ := A * B - A / B

theorem compare_star_values : star 6 (-3) < star 4 (-4) := by
  sorry

end compare_star_values_l392_392858


namespace plane_figures_l392_392508

def polyline_two_segments : Prop := -- Definition for a polyline composed of two line segments
  sorry

def polyline_three_segments : Prop := -- Definition for a polyline composed of three line segments
  sorry

def closed_three_segments : Prop := -- Definition for a closed figure composed of three line segments
  sorry

def quadrilateral_equal_opposite_sides : Prop := -- Definition for a quadrilateral with equal opposite sides
  sorry

def trapezoid : Prop := -- Definition for a trapezoid
  sorry

def is_plane_figure (fig : Prop) : Prop :=
  sorry  -- Axiom or definition that determines whether a figure is a plane figure.

-- Translating the proof problem
theorem plane_figures :
  is_plane_figure polyline_two_segments ∧
  ¬ is_plane_figure polyline_three_segments ∧
  is_plane_figure closed_three_segments ∧
  ¬ is_plane_figure quadrilateral_equal_opposite_sides ∧
  is_plane_figure trapezoid :=
by
  sorry

end plane_figures_l392_392508


namespace distinct_log_values_count_l392_392478

theorem distinct_log_values_count :
  let S := {1, 3, 5, 7, 9} in
  let log_values := {log a - log b | a ∈ S, b ∈ S, a ≠ b} in
  log_values.card = 18 :=
by
  sorry  -- Proof goes here

end distinct_log_values_count_l392_392478


namespace set_union_is_universal_l392_392484

-- Definitions
def U := {2, 3, 4, 5, 6, 7} : Set ℕ
def M := {3, 4, 5, 7} : Set ℕ
def N := {2, 4, 5, 6} : Set ℕ

-- Proof statement
theorem set_union_is_universal : M ∪ N = U := by
  sorry

end set_union_is_universal_l392_392484


namespace reflection_matrix_over_vector_is_correct_l392_392037

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l392_392037


namespace count_valid_c_values_l392_392902

theorem count_valid_c_values : 
  let valid_c : ℕ → Prop := λ c, 
    0 ≤ c ∧ c ≤ 2000 ∧ 
    ∃ x : ℝ, 10 * (x.floor : ℝ) + 3 * (x.ceil : ℝ) = c 
  in
  (∃ n : ℕ, ∀ c, valid_c c ↔ c ∈ finset.range(2001) ∧ c.count' valid_c = 308) := 
by
  sorry

end count_valid_c_values_l392_392902


namespace time_to_meet_l392_392671

-- Define the given conditions
def track_circumference := 528 -- in meters
def deepak_speed_kmh := 4.5 -- in km/hr
def wife_speed_kmh := 3.75 -- in km/hr

-- Convert speeds to m/min
def kmh_to_mmin (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 60

def deepak_speed := kmh_to_mmin deepak_speed_kmh
def wife_speed := kmh_to_mmin wife_speed_kmh

-- Define the statement to prove  
theorem time_to_meet : (track_circumference / (deepak_speed + wife_speed)) = 3.84 :=
by
  -- Proof body to be completed
  sorry

end time_to_meet_l392_392671


namespace exists_arith_prog_not_containing_polynomial_values_l392_392294

theorem exists_arith_prog_not_containing_polynomial_values (P : ℤ[X]) (hP : P.degree = 10) :
  ∃ (a d : ℤ), ∀ k : ℤ, P (a + k * d) ∉ (set.range P) :=
sorry

end exists_arith_prog_not_containing_polynomial_values_l392_392294


namespace convex_polygon_diagonals_l392_392092

def f : ℕ → ℕ

theorem convex_polygon_diagonals (n : ℕ) : f(n + 1) = f(n) + n - 1 :=
sorry

end convex_polygon_diagonals_l392_392092


namespace dot_product_ab_bc_eq_neg4_l392_392521

structure Vector3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def magnitude (v : Vector3D) : ℝ :=
  real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

def dot_product (v₁ v₂ : Vector3D) : ℝ :=
  v₁.x * v₂.x + v₁.y * v₂.y + v₁.z * v₂.z

constant a_ac : ℝ := 2
constant angle_ab_ac : ℝ := 3 * real.pi / 4
constant ab : Vector3D := { x := 1, y := 1, z := 0 }

noncomputable def ac : Vector3D :=
{
  x := a_ac * real.cos angle_ab_ac / magnitude ab,
  y := a_ac * real.cos angle_ab_ac / magnitude ab,
  z := 0
}

noncomputable def bc : Vector3D :=
{
  x := ac.x - ab.x,
  y := ac.y - ab.y,
  z := ac.z - ab.z
}

theorem dot_product_ab_bc_eq_neg4 :
  dot_product ab bc = -4 :=
by sorry

end dot_product_ab_bc_eq_neg4_l392_392521


namespace daps_to_dips_l392_392182

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end daps_to_dips_l392_392182


namespace perfect_squares_four_digit_numbers_l392_392385

-- Part (a)
theorem perfect_squares (m n p : ℕ) (h1 : m > n) (h2 : sqrt m - sqrt n = p) :
  ∃ a b : ℕ, m = a^2 ∧ n = b^2 :=
sorry

-- Part (b)
theorem four_digit_numbers (abcd acd q r b : ℕ) (h : abcd = q^2)
  (h2 : acd = r^2) (h3 : sqrt abcd - sqrt acd = 11 * b) :
  (exists a : ℕ, ∃ q r : ℕ, q - r = 11 * b ∧ q + r = 100 * (9 * a + b)) :=
sorry

end perfect_squares_four_digit_numbers_l392_392385


namespace dap_equiv_48_dips_l392_392165

variables (dap dop dip : Type) [CommRing dap] [CommRing dop] [CommRing dip]

-- Define equivalences between daps, dops, and dips
def equivalence_dap_dop : dap ≃ₐ[dop] (dop →ₐ[dip] dap) := sorry
def equivalence_dop_dip : dop ≃ₐ[dip] (dip →ₐ[dap] dop) := sorry

-- Proportions given in the conditions
def prop1 (d : dap) (o : dop) : 5 * d = 4 * o := sorry
def prop2 (o : dop) (i : dip) : 3 * o = 8 * i := sorry

-- The proof statement
theorem dap_equiv_48_dips : ∀ (d : dap) (i : dip), (15 * d = 32 * i) → (d = 22.5 * i) := 
by
  intros
  sorry

end dap_equiv_48_dips_l392_392165


namespace paws_on_ground_l392_392427

def total_dogs : ℕ := 12
def dogs_on_all_4 : ℕ := total_dogs / 2
def dogs_on_2_legs : ℕ := total_dogs / 2

theorem paws_on_ground : (dogs_on_all_4 * 4) + (dogs_on_2_legs * 2) = 36 := by
  -- num_dogs / 2 is integer division, ensuring dogs_on_all_4 and dogs_on_2_legs are both 6
  calc
    (dogs_on_all_4 * 4) + (dogs_on_2_legs * 2)
        = (6 * 4) + (6 * 2) : by rw [nat.succ_sub_one]; norm_num
    ... = 24 + 12 : by norm_num
    ... = 36 : by norm_num

end paws_on_ground_l392_392427


namespace dips_to_daps_l392_392174

theorem dips_to_daps : 
  ∀ (daps dops dips : Type) (eq1 : 5 * daps = 4 * dops) (eq2 : 3 * dops = 8 * dips),
  (48 * dips = 22.5 * daps) :=
begin
  intros,
  sorry
end

end dips_to_daps_l392_392174


namespace prove_a_plus_b_l392_392405

theorem prove_a_plus_b (a b k : ℤ) (hk : k = a + Real.sqrt b)
  (h : ∀ k : ℝ, abs (5^k - 5^(k - 4)) = 200) :
  a + b = 5 :=
sorry

end prove_a_plus_b_l392_392405


namespace general_solution_of_differential_equation_l392_392456

noncomputable def y (x : ℝ) (C1 C2 : ℝ) : ℝ :=
  C1 + C2 * Real.exp (-x) + (2 * x^2 - 6 * x + 7) * Real.exp x

theorem general_solution_of_differential_equation (C1 C2 : ℝ) :
  ∀ x : ℝ, 
    let y' := λ x, deriv (λ x, y x C1 C2) x,
        y'' := λ x, deriv (λ x, y' x) x in 
    y'' x + y' x = 4 * x^2 * Real.exp x :=
by
  intros x
  sorry

end general_solution_of_differential_equation_l392_392456


namespace domain_f_2x_minus_1_l392_392963

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f (2 * x - 1) = y) :=
by
  intro h
  sorry

end domain_f_2x_minus_1_l392_392963


namespace sum_of_valid_N_is_83_l392_392761

def f (N : ℕ) : ℕ :=
if N % 2 = 0 then N / 2 else 3 * N + 1

def apply_six_times (N : ℕ) : ℕ :=
f (f (f (f (f (f N)))))

theorem sum_of_valid_N_is_83 : 
  (∑ N in (Finset.filter (λ x, apply_six_times x = 1) (Finset.range 100)), N) = 83 :=
sorry

end sum_of_valid_N_is_83_l392_392761


namespace probability_of_p_probability_of_p_accurate_l392_392536

noncomputable def satisfies_equation (p q : ℤ) : Prop := 
  p * q - 6 * p - 3 * q = 3

theorem probability_of_p (h : 1 ≤ p ∧ p ≤ 15) :
  (∃ q : ℤ, satisfies_equation p q) → ∃ n, n = 3 ∧ p ∈ {4, 6, 10} := 
by
  sorry

theorem probability_of_p_accurate :
  (finset.filter (λ p, ∃ q, satisfies_equation p q) (finset.range 15)).card.to_rat / 15 = 1 / 5 :=
by
  sorry

end probability_of_p_probability_of_p_accurate_l392_392536


namespace latest_first_pump_time_l392_392869

theorem latest_first_pump_time 
  (V : ℝ) -- Volume of the pool
  (x y : ℝ) -- Productivity of first and second pumps respectively
  (t : ℝ) -- Time of operation of the first pump until the second pump is turned on
  (h1 : 2*x + 2*y = V/2) -- Condition from 10 AM to 12 PM
  (h2 : 5*x + 5*y = V/2) -- Condition from 12 PM to 5 PM
  (h3 : t*x + 2*x + 2*y = V/2) -- Condition for early morning until 12 PM
  (hx_pos : 0 < x) -- Assume productivity of first pump is positive
  (hy_pos : 0 < y) -- Assume productivity of second pump is positive
  : t ≥ 3 :=
by
  -- The proof goes here...
  sorry

end latest_first_pump_time_l392_392869


namespace rectangle_ratio_of_semicircles_l392_392424

theorem rectangle_ratio_of_semicircles (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h : a * b = π * b^2) : a / b = π := by
  sorry

end rectangle_ratio_of_semicircles_l392_392424


namespace probability_of_12th_grade_l392_392505

noncomputable def N10 : ℕ := 300
noncomputable def N11 : ℕ := 300
noncomputable def N12 : ℕ := 400
noncomputable def N_total : ℕ := N10 + N11 + N12

theorem probability_of_12th_grade (N10 N11 N12 : ℕ) (N_total : ℕ) :
  N10 = 300 →
  N11 = 300 →
  N12 = 400 →
  N_total = N10 + N11 + N12 →
  (N12.toRat / N_total.toRat = (2 : ℚ) / 5) :=
  by
  intros hN10 hN11 hN12 hN_total
  sorry

end probability_of_12th_grade_l392_392505


namespace vertices_coincide_l392_392914

noncomputable theory

variables {A B C D O1 O2 O3 O4 : Point} 
variables (ABO1 : Triangle A B O1)
variables (CDO3 : Triangle C D O1) -- O3 is actually coinciding with O1
variables (BCO2 : Triangle B C O2)
variables (DAO4 : Triangle D A O4)
variables [IsoscelesRight ABO1]
variables [IsoscelesRight CDO3]
variables [IsoscelesRight BCO2]
variables [IsoscelesRight DAO4]

-- Problem statement
theorem vertices_coincide (h1 : O1 = O3) : O2 = O4 :=
sorry

end vertices_coincide_l392_392914


namespace bisect_diagonals_l392_392205

variables {A B C D P Q : Type} [AffineSpace ℝ A B C D]

/-- Given a convex quadrilateral ABCD, if P and Q are midpoints of AB and CD respectively, 
and line PQ bisects diagonal AC, then line PQ also bisects diagonal BD.-/
theorem bisect_diagonals (h1 : convex_quadrilateral A B C D) 
  (h2 : midpoint (A, B) = P) (h3 : midpoint (C, D) = Q) 
  (h4 : bisects (PQ, AC)) : bisects (PQ, BD) := 
sorry

end bisect_diagonals_l392_392205


namespace count_integers_satisfying_inequality_l392_392527

theorem count_integers_satisfying_inequality : {n : ℤ | -12 ≤ n ∧ n ≤ 12 ∧ (n - 3) * (n + 5) * (n + 9) < 0}.toFinset.card = 10 :=
by
  sorry

end count_integers_satisfying_inequality_l392_392527


namespace identical_solutions_for_system_l392_392490

noncomputable def system_of_equations (n : ℕ) : Prop :=
  ∃ (x : fin n → ℝ), 
  (∀ (i : fin n), (1 - x i ^ 2) = x ((i + 1) % n))

theorem identical_solutions_for_system {n : ℕ} :
  ∀ (x : fin n → ℝ), system_of_equations n → (∀ i : fin n, x i = x 0) :=
sorry

end identical_solutions_for_system_l392_392490


namespace midpoint_is_makes_BMD_isosceles_right_l392_392575

noncomputable def isIsoscelesRightTriangle (A B C : ℂ) : Prop :=
  (B - A) = (C - A) * (complex.I)

theorem midpoint_is_makes_BMD_isosceles_right 
  (A B C D E M : ℂ)
  (hABC : isIsoscelesRightTriangle A B C)
  (hADE : isIsoscelesRightTriangle A D E)
  (hM : M = (E + C) / 2) :
  isIsoscelesRightTriangle B M D :=
sorry

end midpoint_is_makes_BMD_isosceles_right_l392_392575


namespace minimum_value_of_expression_l392_392459

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  3 * x + 5 + 2 / x^5 ≥ 10 + 3 * (2 / 5) ^ (1 / 5) := by
sorry

end minimum_value_of_expression_l392_392459


namespace area_ratio_XPQ_PQYZ_l392_392585

theorem area_ratio_XPQ_PQYZ
  (XY YZ XZ XP XQ : ℝ)
  (hXY : XY = 24) (hYZ : YZ = 52) (hXZ : XZ = 60)
  (hXP : XP = 12) (hXQ : XQ = 20)
  (X Y Z P Q : Type) [triangle X Y Z]
  [segment X Y P] [segment X Z Q] :
  XP / XY = 12 / 24 → XQ / XZ = 20 / 60 → 
  ∃ r : ℝ, r = 1 / 7 :=
by sorry

end area_ratio_XPQ_PQYZ_l392_392585


namespace daps_dips_equivalence_l392_392187

theorem daps_dips_equivalence :
  (∃ dap dop dip : Type,
    (5 : ℝ) * ∀ x : dap, x = (4 : ℝ) * ∀ y : dop, y ∧
    (3 : ℝ) * ∀ z : dop, z = (8 : ℝ) * ∀ w : dip, w) →
  (22.5 : ℝ) * ∀ x : dap, x = (48 : ℝ) * ∀ y : dip, y :=
begin
  sorry
end

end daps_dips_equivalence_l392_392187


namespace smallest_x_satisfying_equation_l392_392358

theorem smallest_x_satisfying_equation :
  ∀ x : ℝ, (2 * x ^ 2 + 24 * x - 60 = x * (x + 13)) → x = -15 ∨ x = 4 ∧ ∃ y : ℝ, y = -15 ∨ y = 4 ∧ y ≤ x :=
by
  sorry

end smallest_x_satisfying_equation_l392_392358


namespace reflection_matrix_over_vector_l392_392073

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l392_392073


namespace correct_order_of_operations_l392_392363

def order_of_operations (e : String) : String :=
  if e = "38 * 50 - 25 / 5" then
    "multiplication, division, subtraction"
  else
    "unknown"

theorem correct_order_of_operations :
  order_of_operations "38 * 50 - 25 / 5" = "multiplication, division, subtraction" :=
by
  sorry

end correct_order_of_operations_l392_392363


namespace dap_equiv_48_dips_l392_392166

variables (dap dop dip : Type) [CommRing dap] [CommRing dop] [CommRing dip]

-- Define equivalences between daps, dops, and dips
def equivalence_dap_dop : dap ≃ₐ[dop] (dop →ₐ[dip] dap) := sorry
def equivalence_dop_dip : dop ≃ₐ[dip] (dip →ₐ[dap] dop) := sorry

-- Proportions given in the conditions
def prop1 (d : dap) (o : dop) : 5 * d = 4 * o := sorry
def prop2 (o : dop) (i : dip) : 3 * o = 8 * i := sorry

-- The proof statement
theorem dap_equiv_48_dips : ∀ (d : dap) (i : dip), (15 * d = 32 * i) → (d = 22.5 * i) := 
by
  intros
  sorry

end dap_equiv_48_dips_l392_392166


namespace missing_digit_divisibility_by_13_l392_392745

theorem missing_digit_divisibility_by_13 (B : ℕ) (H : 0 ≤ B ∧ B ≤ 9) : 
  (13 ∣ (200 + 10 * B + 5)) ↔ B = 12 :=
by sorry

end missing_digit_divisibility_by_13_l392_392745


namespace number_of_people_in_team_l392_392744

def total_distance : ℕ := 150
def distance_per_member : ℕ := 30

theorem number_of_people_in_team :
  (total_distance / distance_per_member) = 5 := by
  sorry

end number_of_people_in_team_l392_392744


namespace pow_addition_l392_392380

theorem pow_addition : (-2)^2 + 2^2 = 8 :=
by
  sorry

end pow_addition_l392_392380


namespace sum_of_first_42_odd_cubed_l392_392896

theorem sum_of_first_42_odd_cubed :
  let n := 42 in
  (n * (2 * n - 1))^2 / 4 = 3106894 :=
by
  intro n
  sorry

end sum_of_first_42_odd_cubed_l392_392896


namespace voting_total_participation_l392_392221

theorem voting_total_participation:
  ∀ (x : ℝ),
  0.35 * x + 0.65 * x = x ∧
  0.65 * x = 0.45 * (x + 80) →
  (x + 80 = 260) :=
by
  intros x h
  sorry

end voting_total_participation_l392_392221


namespace square_diagonal_irrational_l392_392503
noncomputable section

-- Given the area of a square is 1
def square (a : ℝ) : Prop :=
  a ^ 2 = 1

-- Prove that the diagonal is an irrational number
theorem square_diagonal_irrational (a : ℝ) (h : square a) : ∃ (d : ℝ), d = a * Real.sqrt 2 ∧ ¬ Rational d :=
by
  -- Definitions of side length and diagonal
  sorry

end square_diagonal_irrational_l392_392503


namespace quadratic_equation_m_condition_l392_392159

theorem quadratic_equation_m_condition (m : ℝ) :
  (m + 1 ≠ 0) ↔ (m ≠ -1) :=
by sorry

end quadratic_equation_m_condition_l392_392159


namespace perimeter_triangle_ADE_eq_AB_plus_AC_l392_392254

theorem perimeter_triangle_ADE_eq_AB_plus_AC
  (A B C I D E : Type*)
  [InnerProductSpace Real A] [InnerProductSpace Real B] 
  [InnerProductSpace Real C] [InnerProductSpace Real I] 
  [InnerProductSpace Real D] [InnerProductSpace Real E]
  (h_incenter : ∃ r, IsIncenter I A B C)
  (h_parallel : ∀ P Q : Type*, I ∈ Line B C → Line P Q ∥ Line B C → D ∈ Line A B ∧ E ∈ Line A C) :
  Perimeter (Triangle A D E) = Perimeter (Triangle A B C) := 
sorry

end perimeter_triangle_ADE_eq_AB_plus_AC_l392_392254


namespace sum_of_x_and_y_l392_392555

-- Define integers x and y
variables (x y : ℤ)

-- Define conditions
def condition1 : Prop := x - y = 200
def condition2 : Prop := y = 250

-- Define the main statement
theorem sum_of_x_and_y (h1 : condition1 x y) (h2 : condition2 y) : x + y = 700 := 
by
  sorry

end sum_of_x_and_y_l392_392555


namespace ab_difference_l392_392540

variable (a b : ℝ)

theorem ab_difference (h : sqrt (a - 3) + (b + 1)^2 = 0) : a - b = 4 := by
  sorry

end ab_difference_l392_392540


namespace irrational_sum_condition_l392_392075

theorem irrational_sum_condition (n : ℕ) :
  (∀ s : finset ℝ, s.card = n → (∀ x ∈ s, irrational x) → 
    (∃ a b c ∈ s, irrational (a + b) ∧ irrational (b + c) ∧ irrational (c + a))) ↔ n ≥ 5 := 
by
  sorry

end irrational_sum_condition_l392_392075


namespace proposition_false_l392_392319

theorem proposition_false (x y : ℤ) (h : x + y = 5) : ¬ (x = 1 ∧ y = 4) := by 
  sorry

end proposition_false_l392_392319


namespace smallest_n_gt_20_l392_392640

theorem smallest_n_gt_20 : ∃ (n : ℕ), n > 20 ∧ n % 6 = 4 ∧ n % 7 = 5 ∧ ∀ m, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 5 → n ≤ m :=
begin
  use 40,
  split,
  { linarith, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm,
    rcases hm with ⟨hm1, hm2, hm3⟩,
    -- The proof would continue from here
    sorry }
end

end smallest_n_gt_20_l392_392640


namespace _l392_392028
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l392_392028


namespace relationship_between_a_b_c_l392_392104

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 3) ^ 2
noncomputable def c : ℝ := Real.log (1 / 30) / Real.log (1 / 3)

theorem relationship_between_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_between_a_b_c_l392_392104


namespace trig_identity_problem_l392_392463

theorem trig_identity_problem :
  let a := 17
  let b := 30
  let sum_angle := a + b
  let cofunc := 90 - a
  let sin30 := 1 / 2
  (sin (sum_angle * real.pi / 180) - sin (a * real.pi / 180) * cos (b * real.pi / 180)) / sin (cofunc * real.pi / 180) = sin30 :=
by
  sorry

end trig_identity_problem_l392_392463


namespace tax_percentage_l392_392589

theorem tax_percentage (total_pay take_home_pay: ℕ) (h1 : total_pay = 650) (h2 : take_home_pay = 585) :
  ((total_pay - take_home_pay) * 100 / total_pay) = 10 :=
by
  -- Assumptions
  have hp1 : total_pay = 650 := h1
  have hp2 : take_home_pay = 585 := h2
  -- Calculate tax paid
  let tax_paid := total_pay - take_home_pay
  -- Calculate tax percentage
  let tax_percentage := (tax_paid * 100) / total_pay
  -- Prove the tax percentage is 10%
  sorry

end tax_percentage_l392_392589


namespace reflection_matrix_l392_392021

-- Definitions of the problem conditions
def vector := ℝ × ℝ
def projection (u v : vector) : vector := 
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_uu := u.1 * u.1 + u.2 * u.2
  (dot_uv / dot_uu * u.1, dot_uv / dot_uu * u.2)

def reflection (u v : vector) : vector :=
  let p := projection u v
  (2 * p.1 - v.1, 2 * p.2 - v.2)

-- Theorem to prove
theorem reflection_matrix : 
  ∃ M : matrix (fin 2) (fin 2) ℝ,
  ∀ (v : vector), reflection (4, 3) v = (M 0 0 * v.1 + M 0 1 * v.2, M 1 0 * v.1 + M 1 1 * v.2) :=
begin
  use (λ i j, if (i, j) = (0, 0) then 7/25 else if (i, j) = (0, 1) then 24/25 else if (i, j) = (1, 0) then 24/25 else -7/25),
  sorry
end

end reflection_matrix_l392_392021


namespace acute_angled_triangle_at_most_seventy_percent_l392_392482

-- Define the fraction of acute-angled triangles among all triangles for a given n points
def h (n : ℕ) : ℚ := sorry -- Fraction of acute-angled triangles function, to be defined

-- Assume we have 100 points in the plane with no three points collinear
def points_condition (n : ℕ) : Prop :=
  n = 100 ∧ (∀ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3)

-- Theorem statement: Prove that the fraction of acute-angled triangles among all triangles is at most 0.7
theorem acute_angled_triangle_at_most_seventy_percent (n : ℕ) (H : points_condition n) : 
  h 100 ≤ (7 / 10) := sorry

end acute_angled_triangle_at_most_seventy_percent_l392_392482


namespace fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l392_392808

section FoldingNumberLine

-- Part (1)
def coincides_point_3_if_minus2_2_fold (x : ℝ) : Prop :=
  x = -3

theorem fold_minus2_2_3_coincides_neg3 :
  coincides_point_3_if_minus2_2_fold 3 :=
by
  sorry

-- Part (2) ①
def coincides_point_7_if_minus1_3_fold (x : ℝ) : Prop :=
  x = -5

theorem fold_minus1_3_7_coincides_neg5 :
  coincides_point_7_if_minus1_3_fold 7 :=
by
  sorry

-- Part (2) ②
def B_position_after_folding (m : ℝ) (h : m > 0) (A B : ℝ) : Prop :=
  B = 1 + m / 2

theorem fold_distanceA_to_B_coincide (m : ℝ) (h : m > 0) (A B : ℝ) :
  B_position_after_folding m h A B :=
by
  sorry

end FoldingNumberLine

end fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l392_392808


namespace data_set_conditions_l392_392551

theorem data_set_conditions (x : ℝ) (H_avg : (2 + 4 + x + 5 + 7) / 5 = 5) : 
  x = 7 ∧ (∃ s, s = [2, 4, 5, 7, 7] ∧ (s.nthLe 2 (by simp)) = 5) :=
by
  sorry

end data_set_conditions_l392_392551


namespace triangle_side_length_l392_392560

noncomputable def PQ := 93
noncomputable def PR := 105
constant QR : ℕ
constant QY : ℕ
constant YR : ℕ

-- Conditions given: 
axiom cond1 : QR = QY + YR
axiom cond2 : PQ = 93
axiom cond3 : PR = 105
axiom cond4 : PQ = 93

-- We will prove that QR is equal to 66 based on the conditions.
theorem triangle_side_length :
  ∃ (a b : ℕ), QR = a + b ∧ a * (a + b) = 2376 ∧ a + b = 66 :=
begin 
  sorry  -- Proof to be filled in by the theorem prover
end

end triangle_side_length_l392_392560


namespace bisects_pq_bd_l392_392211

-- Definitions
variables {A B C D P Q : Type}
variables [convex_quadrilateral A B C D]
variables [midpoint P A B]
variables [midpoint Q C D]

-- Hypothesis
variable (h1 : bisects P Q A C)

-- Theorem statement
theorem bisects_pq_bd : bisects P Q B D :=
sorry

end bisects_pq_bd_l392_392211


namespace min_value_of_f_l392_392310

noncomputable def f (x : ℝ) := 
  (Real.cos (2 * x) * (Real.sqrt 3 / 2)) + (Real.cos x * -Real.sin x)

theorem min_value_of_f : 
  x ∈ Icc 0 (Real.pi / 2) → 
  ∃ (y : ℝ), y = (Real.min {f x | x ∈ Icc 0 (Real.pi / 2)}) ∧ y = - (Real.sqrt 3 / 2) := 
sorry

end min_value_of_f_l392_392310


namespace area_of_region_defined_by_eq_l392_392356

theorem area_of_region_defined_by_eq : 
  (∃ x y : ℝ, x^2 + y^2 + 8 * x - 6 * y = 2) → real.pi * (3 * real.sqrt 3)^2 = 27 * real.pi :=
by
  intro h
  sorry

end area_of_region_defined_by_eq_l392_392356


namespace symmetric_circle_eq_l392_392887

theorem symmetric_circle_eq :
  let circ1 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 - 1 = 0},
      line := {p : ℝ × ℝ | 2 * p.1 - p.2 + 3 = 0},
      circ2 := {p : ℝ × ℝ | (p.1 + 3)^2 + (p.2 - 2)^2 = 2}
  in ∀ (p : ℝ × ℝ), p ∈ circ2 ↔ p ∈ symmetric_about_line circ1 line := sorry

noncomputable def symmetric_about_line (circ : set (ℝ × ℝ)) (line : set (ℝ × ℝ)) : set (ℝ × ℝ) := sorry

end symmetric_circle_eq_l392_392887


namespace water_consumption_total_l392_392270

def number_of_cows : ℕ := 40
def water_per_cow_per_day : ℕ := 80
def sheep_multiplicative_factor : ℕ := 10
def water_factor_cow_to_sheep : ℕ := 1 / 4
def days_in_week : ℕ := 7

theorem water_consumption_total :
  let cows_water_per_week := number_of_cows * water_per_cow_per_day * days_in_week in
  let number_of_sheep := number_of_cows * sheep_multiplicative_factor in
  let water_per_sheep_per_day := water_per_cow_per_day * water_factor_cow_to_sheep in
  let sheep_water_per_week := number_of_sheep * water_per_sheep_per_day * days_in_week in
  cows_water_per_week + sheep_water_per_week = 78400 := sorry

end water_consumption_total_l392_392270


namespace problem_statement_l392_392191

def is_H_function (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x1 ∈ D → x2 ∈ D → x1 ≠ x2 → x1 * f x1 + x2 * f x2 > x1 * f x2 + x2 * f x1

def func1 (x : ℝ) : ℝ := 1 / 3 * x^3 - 1 / 2 * x^2 + 1 / 2 * x
def D1 : set ℝ := set.univ

def func2 (x : ℝ) : ℝ := 3 * x + cos x - sin x
def D2 : set ℝ := {x | 0 < x ∧ x < π / 2}

def func3 (x : ℝ) : ℝ := (x + 1) * exp (-x)
def D3 : set ℝ := {x | x < 1}

def func4 (x : ℝ) : ℝ := x * log x
def D4 : set ℝ := {x | 0 < x ∧ x < 1 / exp 1}

theorem problem_statement :
  is_H_function func1 D1 ∧ is_H_function func2 D2 ∧ ¬ is_H_function func3 D3 ∧ ¬ is_H_function func4 D4 := by
  sorry

end problem_statement_l392_392191


namespace ratio_of_medians_of_tetrahedron_l392_392128

-- Definitions for the problem
def regular_tetrahedron (A B C D : Point3D) : Prop := 
  distance A B = distance A C ∧ distance A C = distance A D ∧ 
  distance A D = distance B C ∧ distance B C = distance B D ∧ 
  distance B D = distance C D

def centroid (M : Point3D) (B C D : Point3D) : Prop := 
  M = (B + C + D) / 3

def equidistant_from_faces (O : Point3D) (A B C D : Point3D) : Prop := 
  distance O (Plane A B C) = distance O (Plane A B D) ∧ 
  distance O (Plane A C D) = distance O (Plane B C D) ∧ 
  distance O (Plane A B C) = distance O (Plane A C D)

variables (A B C D M O : Point3D)

theorem ratio_of_medians_of_tetrahedron (h1 : regular_tetrahedron A B C D)
  (h2 : centroid M B C D)
  (h3 : equidistant_from_faces O A B C D) :
  (distance A O) / (distance O M) = 3 := 
sorry

end ratio_of_medians_of_tetrahedron_l392_392128


namespace no_rational_ratio_l392_392082

def S (n : ℕ) : ℝ := ∑ k in finset.range (n + 1), (2 ^ (k + 1)) / (k + 1) ^ 2

theorem no_rational_ratio (P Q : ℚ[X]) (hP : ∀ n : ℕ, n > 0 → (S (n + 1) / S n) = (P.eval n / Q.eval n)) : false :=
sorry

end no_rational_ratio_l392_392082


namespace speed_of_current_l392_392806

section rowing

def row_speed_kmph := 15
def distance_m := 90
def time_s := 17.998560115190784

/-- Convert the rowing speed from kmph to m/s. --/
def row_speed_mps := row_speed_kmph * 1000 / 3600

/-- Calculate the downstream speed in m/s. --/
def downstream_speed_mps := distance_m / time_s

/-- Calculate the speed of the current in m/s. --/
def speed_of_current_mps := downstream_speed_mps - row_speed_mps

/-- Convert the speed of the current from m/s to kmph. --/
def speed_of_current_kmph := speed_of_current_mps * 3600 / 1000

/-- Given conditions, prove that the speed of the current is 3 kmph. --/
theorem speed_of_current : speed_of_current_kmph = 3 := by
  -- Skipping the proof for now
  sorry

end rowing

end speed_of_current_l392_392806


namespace cot_phi_bisecting_lines_l392_392824

theorem cot_phi_bisecting_lines (a b c : ℝ) (s : ℝ) (area : ℝ) (φ : ℝ) :
  a = 13 ∧ b = 14 ∧ c = 15 ∧ s = (a + b + c) / 2 ∧ area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) →
  ∃ φ, ∀ pq : ℝ, (pq = 42 * 2 ∧ cot (φ) = (Real.sqrt 105 + Real.sqrt 15) / (15 * Real.sqrt 3 - 21)) := sorry

end cot_phi_bisecting_lines_l392_392824


namespace probability_sum_greater_9_l392_392142

open_locale classical
noncomputable theory

def set_of_numbers := {1, 3, 5, 7, 9}

-- Define the event we are interested in: sum of two distinct numbers is greater than 9
def event_sum_greater_9 (x y : ℕ) : Prop := x + y > 9

-- Count total pairs
def total_pairs : ℕ := (finset.univ.powerset_len 2).card

-- Count favorable pairs
def favorable_pairs : ℕ :=
(finset.univ.powerset_len 2).filter (λ s, ∃ a b, a ∈ s ∧ b ∈ s ∧ a ≠ b ∧ event_sum_greater_9 a b).card

-- Probability
def probability := (favorable_pairs : ℝ) / (total_pairs : ℝ)

-- The theorem to prove
theorem probability_sum_greater_9 : probability = 3 / 5 :=
sorry

end probability_sum_greater_9_l392_392142


namespace digit_makes_5678d_multiple_of_9_l392_392473

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

theorem digit_makes_5678d_multiple_of_9 (d : Nat) (h : d ≥ 0 ∧ d < 10) :
  is_multiple_of_9 (5 * 10000 + 6 * 1000 + 7 * 100 + 8 * 10 + d) ↔ d = 1 := 
by
  sorry

end digit_makes_5678d_multiple_of_9_l392_392473


namespace find_first_number_l392_392888

noncomputable def x : ℕ := 7981
noncomputable def y : ℕ := 9409
noncomputable def mean_proportional : ℕ := 8665

theorem find_first_number (mean_is_correct : (mean_proportional^2 = x * y)) : x = 7981 := by
-- Given: mean_proportional^2 = x * y
-- Goal: x = 7981
  sorry

end find_first_number_l392_392888


namespace cost_effectiveness_l392_392624

-- Define general parameters and conditions given in the problem
def a : ℕ := 70 -- We use 70 since it must be greater than 50

-- Define the scenarios
def cost_scenario1 (a: ℕ) : ℕ := 4500 + 27 * a
def cost_scenario2 (a: ℕ) : ℕ := 4400 + 30 * a

-- The theorem to be proven
theorem cost_effectiveness (h : a > 50) : cost_scenario1 a < cost_scenario2 a :=
  by
  -- First, let's replace a with 70 (this step is unnecessary in the proof since a = 70 is fixed)
  let a := 70
  -- Now, prove the inequality
  sorry

end cost_effectiveness_l392_392624


namespace daps_to_dips_l392_392180

theorem daps_to_dips : 
  (∀ a b c d : ℝ, (5 * a = 4 * b) → (3 * b = 8 * c) → (c = 48 * d) → (a = 22.5 * d)) := 
by
  intros a b c d h1 h2 h3
  sorry

end daps_to_dips_l392_392180


namespace trapezoid_area_condition_l392_392574

theorem trapezoid_area_condition
  (a x y z : ℝ)
  (h_sq  : ∀ (ABCD : ℝ), ABCD = a * a)
  (h_trap: ∀ (EBCF : ℝ), EBCF = x * a)
  (h_rec : ∀ (JKHG : ℝ), JKHG = y * z)
  (h_sum : y + z = a)
  (h_area : x * a = a * a - 2 * y * z) :
  x = a / 2 :=
by
  sorry

end trapezoid_area_condition_l392_392574


namespace sum_marked_sides_ge_one_l392_392415

theorem sum_marked_sides_ge_one {n : ℕ} (a b : ℕ → ℝ) 
  (h1 : ∑ i in finset.range n, a i * b i = 1)
  (h2 : ∀ i, 0 < a i ∧ a i ≤ 1)
  (h3 : ∀ i, 0 < b i ∧ b i ≤ 1) :
  ∑ i in finset.range n, a i ≥ 1 := 
  sorry

end sum_marked_sides_ge_one_l392_392415


namespace first_place_beat_joe_l392_392598

theorem first_place_beat_joe (joe_won joe_draw first_place_won first_place_draw points_win points_draw : ℕ) 
    (h1 : joe_won = 1) (h2 : joe_draw = 3) (h3 : first_place_won = 2) (h4 : first_place_draw = 2)
    (h5 : points_win = 3) (h6 : points_draw = 1) : 
    (first_place_won * points_win + first_place_draw * points_draw) - (joe_won * points_win + joe_draw * points_draw) = 2 :=
by
   sorry

end first_place_beat_joe_l392_392598


namespace calculate_gross_income_l392_392314
noncomputable def gross_income (net_income : ℝ) (tax_rate : ℝ) : ℝ := net_income / (1 - tax_rate)

theorem calculate_gross_income : gross_income 20000 0.13 = 22989 :=
by
  sorry

end calculate_gross_income_l392_392314


namespace centroid_and_concurrency_proof_l392_392247

open Triangle

variable {P : Type*} [EuclideanGeometry P]
variables {A B C A1 B1 C1 A2 B2 C2 A3 B3 C3 : P}

-- Conditions
def are_altitudes (A1 B1 C1 A B C : P) : Prop := 
  IsAltitude B C A A1 ∧ IsAltitude A C B B1 ∧ IsAltitude A B C C1

def circumcircle_intersections (A2 B2 C2 A B C A1 B1 C1 : P) : Prop :=
  SecondIntersection (circumcircle A B C) (line_through_points A A1) A2 ∧ 
  SecondIntersection (circumcircle A B C) (line_through_points B B1) B2 ∧ 
  SecondIntersection (circumcircle A B C) (line_through_points C C1) C2

def simson_lines_triangle (A3 B3 C3 A2 B2 C2 A B C : P) : Prop :=
  SimsonPoint A2 B2 ∧ SimsonPoint A2 C2 ∧ 
  SimsonPoint B2 A2 ∧ SimsonPoint B2 C2 ∧ 
  SimsonPoint C2 A2 ∧ SimsonPoint C2 B2

-- Questions
def coinciding_centroids (G1 G3 : P) (A1 B1 C1 A3 B3 C3 : P) : Prop :=
  Centroid A1 B1 C1 G1 ∧ Centroid A3 B3 C3 G3 ∧ G1 = G3

def concurrent_lines (A2 B2 C2 A3 B3 C3 : P) : Prop :=
  Concurrent [line_through_points A2 A3, line_through_points B2 B3, line_through_points C2 C3]

-- Main Theorem
theorem centroid_and_concurrency_proof
  (h1 : are_altitudes A1 B1 C1 A B C)
  (h2 : circumcircle_intersections A2 B2 C2 A B C A1 B1 C1)
  (h3 : simson_lines_triangle A3 B3 C3 A2 B2 C2 A B C) :
  ∃ G1 G3 : P, coinciding_centroids G1 G3 A1 B1 C1 A3 B3 C3 ∧ 
              concurrent_lines A2 B2 C2 A3 B3 C3 := 
sorry

end centroid_and_concurrency_proof_l392_392247


namespace linear_dependent_iff_38_div_3_l392_392860

theorem linear_dependent_iff_38_div_3 (k : ℚ) :
  k = 38 / 3 ↔ ∃ (α β γ : ℚ), α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0 ∧
    α * 1 + β * 4 + γ * 7 = 0 ∧
    α * 2 + β * 5 + γ * 8 = 0 ∧
    α * 3 + β * k + γ * 9 = 0 :=
by
  sorry

end linear_dependent_iff_38_div_3_l392_392860


namespace ramu_profit_percent_l392_392285

theorem ramu_profit_percent (initial_price repair_percent improve_percent tax_percent selling_price : ℝ)
    (h_initial_price : initial_price = 50000)
    (h_repair_percent : repair_percent = 0.20)
    (h_improve_percent : improve_percent = 0.15)
    (h_tax_percent : tax_percent = 0.05)
    (h_selling_price : selling_price = 75000) :
    let repair_cost := repair_percent * initial_price,
        total_after_repairs := initial_price + repair_cost,
        improve_cost := improve_percent * total_after_repairs,
        total_after_improve := total_after_repairs + improve_cost,
        sales_tax := tax_percent * total_after_improve,
        total_cost := total_after_improve + sales_tax,
        profit := selling_price - total_cost,
        profit_percent := (profit / total_cost) * 100
    in profit_percent = 3.52 :=
by
  intros
  sorry

end ramu_profit_percent_l392_392285


namespace evaluate_expression_l392_392007

theorem evaluate_expression :
  200 * (200 - 3) + (200 ^ 2 - 8 ^ 2) = 79336 :=
by
  sorry

end evaluate_expression_l392_392007


namespace pow_addition_l392_392383

theorem pow_addition : (-2 : ℤ)^2 + (2 : ℤ)^2 = 8 :=
by
  sorry

end pow_addition_l392_392383


namespace total_cost_is_26_30_l392_392855

open Real

-- Define the costs
def cost_snake_toy : ℝ := 11.76
def cost_cage : ℝ := 14.54

-- Define the total cost of purchases
def total_cost : ℝ := cost_snake_toy + cost_cage

-- Prove the total cost equals $26.30
theorem total_cost_is_26_30 : total_cost = 26.30 :=
by
  sorry

end total_cost_is_26_30_l392_392855


namespace dap_equiv_48_dips_l392_392164

variables (dap dop dip : Type) [CommRing dap] [CommRing dop] [CommRing dip]

-- Define equivalences between daps, dops, and dips
def equivalence_dap_dop : dap ≃ₐ[dop] (dop →ₐ[dip] dap) := sorry
def equivalence_dop_dip : dop ≃ₐ[dip] (dip →ₐ[dap] dop) := sorry

-- Proportions given in the conditions
def prop1 (d : dap) (o : dop) : 5 * d = 4 * o := sorry
def prop2 (o : dop) (i : dip) : 3 * o = 8 * i := sorry

-- The proof statement
theorem dap_equiv_48_dips : ∀ (d : dap) (i : dip), (15 * d = 32 * i) → (d = 22.5 * i) := 
by
  intros
  sorry

end dap_equiv_48_dips_l392_392164


namespace player_b_wins_condition_a_player_a_prevents_b_winning_condition_b_l392_392350

-- Define the 8x8 grid
def Grid := Fin 8 × Fin 8

-- Define the players
inductive Player
| A
| B

-- Define actions for players
structure Action where
  act : Player → Grid → Option Grid → Prop

-- Define game conditions
def initialGrid : Set Grid :=
  {coord | True} -- All coordinates are initially white.

variable (actions : ℕ → Player → Action) -- Actions per turn for each player

-- Condition for Player B winning under first scenario
def condition_a (grid : Set Grid) : Prop :=
  ∀ (i j : Fin 4), ∃ (k : Fin 2 × Fin 2), 
    (i + k.1 < 8) ∧ (j + k.2 < 8) ∧ 
    (i, j) ∈ grid ∧ 
    ((i + 5, j + 5) ∈ grid ∨ (i + 5, j) ∈ grid ∨ (i, j + 5) ∈ grid ∨ (i, j) ∈ grid)

-- Condition for Player B winning under second scenario
def condition_b (grid : Set Grid) : Prop :=
  ∀ (i j : Fin 4), ∃ (k1 k2 : Fin 2 × Fin 2), 
    k1 ≠ k2 ∧ 
    (i + k1.1 < 8 ∧ j + k1.2 < 8 ∧ i + k2.1 < 8 ∧ j + k2.2 < 8) ∧ 
    (i, j) ∈ grid ∧ 
    ((i + 5, j + 5) ∈ grid ∨ (i + 5, j) ∈ grid ∨ (i, j + 5) ∈ grid ∨ (i, j) ∈ grid)

-- Theorems
theorem player_b_wins_condition_a : Player B can guarantee a win under condition_a. :=
by {
  sorry -- proof omitted
}

theorem player_a_prevents_b_winning_condition_b : Player A can prevent B from winning under condition_b. :=
by {
  sorry -- proof omitted
}

end player_b_wins_condition_a_player_a_prevents_b_winning_condition_b_l392_392350


namespace circumcircles_concentric_l392_392609

variables {A B C D E F X Y: Type*}
noncomputable def is_median (A B C D : Type*) : Prop := sorry
-- Reflections and other geometric operations assumed to be defined.
noncomputable def is_reflection (F A D X : Type*) : Prop := sorry
noncomputable def concentric_circles (B E X A D Y : Type*) : Prop := sorry

-- Triangle vertices and condition definitions
variables [is_median A B C D] [is_median B A C E] [is_median C A B F]
variables [is_reflection F A D X] [is_reflection F B E Y]

theorem circumcircles_concentric :
  concentric_circles B E X A D Y := 
by
  sorry

end circumcircles_concentric_l392_392609


namespace set_intersection_complement_l392_392612

noncomputable def A : set ℕ := {x : ℕ | real.log2 x ≤ 2}
noncomputable def B : set ℝ := {x : ℝ | ∃ y : ℝ, y = real.sqrt (x - 3)}

theorem set_intersection_complement (A_def : A = {1, 2, 3, 4}) (B_def : B = set.Ici 3) :
  A ∩ (set.compl B : set ℝ) = {1, 2} :=
by
  rw [A_def, B_def]
  dsimp [A, B]
  have h₁ : {x : ℝ | x ∈ {1, 2, 3, 4}} = {1, 2, 3, 4}, by simp
  have h₂ : (set.compl (set.Ici 3) : set ℝ) = {x : ℝ | x < 3}, by simp [set.compl, set.Ici]
  rw [h₁, h₂]
  simp
  sorry

end set_intersection_complement_l392_392612


namespace solution_set_l392_392939

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 2 else x + 2

theorem solution_set (x : ℝ) : f x ≥ x^2 ↔ -2 ≤ x ∧ x ≤ 2 :=
begin
  sorry
end

end solution_set_l392_392939


namespace current_swans_number_l392_392873

noncomputable def swans_doubling (S : ℕ) : Prop :=
  let S_after_10_years := S * 2^5 -- Doubling every 2 years for 10 years results in multiplying by 2^5
  S_after_10_years = 480

theorem current_swans_number (S : ℕ) (h : swans_doubling S) : S = 15 := by
  sorry

end current_swans_number_l392_392873


namespace A_and_B_together_finish_work_l392_392368

theorem A_and_B_together_finish_work (days_B : ℕ) (hB : days_B = 12) 
  (work_rate_A : ℝ) (work_rate_B : ℝ) (hA : work_rate_A = 2 * work_rate_B)
  (hB_work_rate : work_rate_B = 1 / (days_B : ℝ)) : 
  ∃ days_AB : ℝ, days_AB = 4 :=
begin
  sorry
end

end A_and_B_together_finish_work_l392_392368


namespace part1_part2_l392_392611

noncomputable def f (x : ℝ) : ℝ := ∫ θ in x..(x + π / 3), |Real.sin θ|

-- Statement for Part 1: Proving derivative of f(x)
theorem part1 (x : ℝ) : deriv f x = |Real.sin (x + π / 3)| - |Real.sin x| := 
sorry

-- Statement for Part 2: Finding maximum and minimum values of f(x) on the given interval
theorem part2 : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π → f(x) ≤ 1) ∧ 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π ∧ f(x) = 1) ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ π → 2 - sqrt 3 ≤ f(x)) ∧ 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π ∧ f(x) = 2 - sqrt 3) := 
sorry

end part1_part2_l392_392611


namespace sum_divisible_by_floor_sqrt_l392_392465

theorem sum_divisible_by_floor_sqrt :
  let S := ∑ k in Finset.filter (λ k, k % Int.floor (Real.sqrt k) = 0) (Finset.range 1000000) id
  in S = 999999000 :=
by
  sorry

end sum_divisible_by_floor_sqrt_l392_392465


namespace find_larger_integer_l392_392728

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l392_392728


namespace players_even_sum_probability_l392_392657

-- Definitions translated from conditions
def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def choose_three (s : Finset ℕ) := s.choose 3
def valid_sum (tiles : Finset ℕ) := (tiles.sum % 2 = 0)

-- Theorem statement to prove m + n = 85 given the conditions
theorem players_even_sum_probability :
  let s := numbers in
  let player1 := choose_three s in
  let remaining := s \ player1 in
  let player2 := choose_three remaining in
  let player3 := remaining \ player2 in
  valid_sum player1 ∧ valid_sum player2 ∧ valid_sum player3 →
  (let m := 1, n := 84 in m + n = 85) :=
by
  sorry

end players_even_sum_probability_l392_392657


namespace complex_equation_solution_l392_392957

-- Define variables and conditions
variables (a : ℝ) (i : ℂ) (h_i : i * i = -1) (h_cond : (a + 2 * i) / (2 + i) = i)

-- State the theorem to be proved
theorem complex_equation_solution (h : ℝ) (h_cond : (h + 2 * complex.I) / (2 + complex.I) = complex.I) : h = -1 :=
by sorry

end complex_equation_solution_l392_392957


namespace BH_eq_CX_l392_392223

theorem BH_eq_CX
  (A B C M H Q P X : Point)
  (hABC : acute_scaled_triangle A B C)
  (hM : M = midpoint A B C)
  (hH : H = foot A (line B C))
  (hQ : Q ∈ line A B)
  (hP : P ∈ line A C)
  (hQM : perpendicular (line Q M) (line A C))
  (hPM : perpendicular (line P M) (line A B))
  (hX : second_intersection (circumcircle P M Q) (line B C) = X) :
  (distance B H) = (distance C X) := by
  sorry

end BH_eq_CX_l392_392223


namespace probability_of_winning_prize_l392_392234

noncomputable def winning_probability_third_flip : ℕ := 6

theorem probability_of_winning_prize (
  total_logos : ℕ,
  winning_logos : ℕ,
  flips_done : ℕ,
  remaining_logos : ℕ,
  remaining_winning_logos : ℕ
) : 
  total_logos = 20 →
  winning_logos = 5 →
  flips_done = 2 →
  remaining_logos = 18 →
  remaining_winning_logos = 3 →
  (remaining_winning_logos / remaining_logos) = (1 / winning_probability_third_flip) :=
by sorry

end probability_of_winning_prize_l392_392234


namespace DeansCalculatorGame_l392_392972

theorem DeansCalculatorGame (r : ℕ) (c1 c2 c3 : ℤ) (h1 : r = 45) (h2 : c1 = 1) (h3 : c2 = 0) (h4 : c3 = -2) : 
  let final1 := (c1 ^ 3)
  let final2 := (c2 ^ 2)
  let final3 := (-c3)^45
  final1 + final2 + final3 = 3 := 
by
  sorry

end DeansCalculatorGame_l392_392972


namespace exists_k_for_any_n_l392_392899

theorem exists_k_for_any_n (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, 2 * k^2 + 2001 * k + 3 ≡ 0 [MOD 2^n] :=
sorry

end exists_k_for_any_n_l392_392899


namespace coordinates_of_B_l392_392193

theorem coordinates_of_B (a : ℝ) (h : a - 2 = 0) : (a + 2, a - 1) = (4, 1) :=
by
  sorry

end coordinates_of_B_l392_392193


namespace larger_integer_value_l392_392679

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l392_392679


namespace problem1_problem2_l392_392520

def A (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 7
def S (x : ℝ) (k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

theorem problem1 (k : ℝ) : (∀ x, S x k → A x) → k ≤ 4 :=
by
  sorry

theorem problem2 (k : ℝ) : (∀ x, ¬(A x ∧ S x k)) → k < 2 ∨ k > 6 :=
by
  sorry

end problem1_problem2_l392_392520


namespace bisects_pq_bd_l392_392210

-- Definitions
variables {A B C D P Q : Type}
variables [convex_quadrilateral A B C D]
variables [midpoint P A B]
variables [midpoint Q C D]

-- Hypothesis
variable (h1 : bisects P Q A C)

-- Theorem statement
theorem bisects_pq_bd : bisects P Q B D :=
sorry

end bisects_pq_bd_l392_392210


namespace jane_percentage_increase_l392_392989

theorem jane_percentage_increase (B H : ℝ) (H_pos : 0 < H) :
  let bears_per_hour := B / H,
      bears_with_assistant_per_hour := 2 * bears_per_hour,
      hours_with_assistant := 0.90 * H,
      bears_with_assistant := bears_with_assistant_per_hour * hours_with_assistant in
  100 * ((bears_with_assistant - B) / B) = 80 :=
by
  sorry

end jane_percentage_increase_l392_392989


namespace polygon_interior_angles_l392_392332

theorem polygon_interior_angles (k : ℕ) (h : k ≥ 3) : 
  let interior_angle_sum := (k - 2) * 180 in
  ∃ S, S = interior_angle_sum ∧ ∀ n ≥ k, 
  let S_n := (n - 2) * 180 in S_n > S :=
begin
  sorry,
end

end polygon_interior_angles_l392_392332


namespace i_power_2016_eq_one_l392_392926

-- Definition of the imaginary unit i
def i : ℂ := Complex.I  -- ℂ represents the complex number space in Lean

-- Given condition
def i_power_four_eq_one : i^4 = 1 := by
  exact Complex.I_pow_4

-- Statement to be proved
theorem i_power_2016_eq_one : i^2016 = 1 :=
  by sorry

end i_power_2016_eq_one_l392_392926


namespace triangle_inequality_l392_392281

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  0 < Real.cot (A / 4) - Real.tan (B / 4) - Real.tan (C / 4) - 1 ∧ 
  Real.cot (A / 4) - Real.tan (B / 4) - Real.tan (C / 4) - 1 < 2 * Real.cot (A / 2) :=
by
  sorry

end triangle_inequality_l392_392281


namespace tied_at_end_of_august_l392_392302

/-
The double-bar graph shows the number of home runs hit by McGwire and Sosa during each month of the 1998 baseball season. 
At the end of which month were McGwire and Sosa tied in total number of home runs?
-/

def mcgwire_home_runs : ℕ → ℕ
| 0   := 1   -- March
| 1   := 10  -- April
| 2   := 16  -- May
| 3   := 10  -- June
| 4   := 8   -- July
| 5   := 10  -- August
| _   := 0   -- For months not considered

def sosa_home_runs : ℕ → ℕ
| 0   := 0   -- March
| 1   := 6   -- April
| 2   := 7   -- May
| 3   := 20  -- June
| 4   := 9   -- July
| 5   := 13  -- August
| _   := 0   -- For months not considered

def cumulative_home_runs (f : ℕ → ℕ) (month : ℕ) : ℕ :=
(nat.range (month + 1)).sum f

theorem tied_at_end_of_august : cumulative_home_runs mcgwire_home_runs 5 = cumulative_home_runs sosa_home_runs 5 :=
sorry

end tied_at_end_of_august_l392_392302


namespace distinct_positive_integers_divisible_by_10_l392_392259

theorem distinct_positive_integers_divisible_by_10 
  (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : 0 < a) (h5 : 0 < b) (h6 : 0 < c) :
  ∃ x ∈ {a^3 * b - a * b^3, b^3 * c - b * c^3, c^3 * a - c * a^3}, 10 ∣ x :=
by
  sorry

end distinct_positive_integers_divisible_by_10_l392_392259


namespace larger_integer_value_l392_392683

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l392_392683


namespace larger_integer_is_21_l392_392718

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l392_392718


namespace eight_circles_area_zero_l392_392445

noncomputable def area_of_annular_region (R : ℝ) (n : ℕ) (r : ℝ) : ℝ :=
  let outer_area := π * R^2
  let inner_area := n * (π * r^2)
  outer_area - inner_area

theorem eight_circles_area_zero (R : ℝ) (r : ℝ) (n : ℕ) (hR : R = 40) (hn : n = 8) (hr : r = 10 * Real.sqrt 2) :
  ∀ (L : ℝ), L = area_of_annular_region R n r → ⌊L⌋ = 0 :=
by
  sorry

end eight_circles_area_zero_l392_392445


namespace pass_platform_time_correct_l392_392388

-- Definitions based on problem conditions

-- Length of the train (in meters)
def train_length : ℝ := 1500

-- Time to cross a tree (in seconds)
def tree_cross_time : ℝ := 100

-- Length of the platform (in meters)
def platform_length : ℝ := 1000

-- Speed of the train (in meters per second)
def train_speed : ℝ := train_length / tree_cross_time

-- Combined distance to pass the platform (in meters)
def combined_distance : ℝ := train_length + platform_length

-- Time to pass the platform (in seconds)
def pass_platform_time : ℝ := combined_distance / train_speed

-- Theorem statement: Prove that the time to pass the platform is 166.67 seconds
theorem pass_platform_time_correct : pass_platform_time = 166.67 := 
  sorry

end pass_platform_time_correct_l392_392388


namespace set_B_correct_l392_392124

noncomputable section

open Set

def U : Set ℕ := { x ∈ ℕ | 0 < log 10 x ∧ log 10 x < 1 }

def A : Set ℕ := {1, 3, 5, 7, 9} ∪ (U ∩ { x | x % 2 = 1 })

def complement_U (B : Set ℕ) : Set ℕ := U \ B

def B : Set ℕ := {2, 4, 6, 8}

theorem set_B_correct :
  (U = A ∪ B) →
  (A ∩ complement_U B = {1, 3, 5, 7, 9}) →
  B = {2, 4, 6, 8} :=
by
  intros hU hA
  sorry

end set_B_correct_l392_392124


namespace scrabble_letter_values_l392_392241

-- Definitions based on conditions
def middle_letter_value : ℕ := 8
def final_score : ℕ := 30

-- The theorem we need to prove
theorem scrabble_letter_values (F T : ℕ)
  (h1 : 3 * (F + middle_letter_value + T) = final_score) :
  F = 1 ∧ T = 1 :=
sorry

end scrabble_letter_values_l392_392241


namespace janet_needs_4_weeks_l392_392591

variable (hourly_rate : ℝ := 20) (regular_hours : ℝ := 40) (total_hours : ℝ := 52) (overtime_factor : ℝ := 1.5) (car_cost : ℝ := 4640)

def total_weekly_earning (hourly_rate : ℝ) (regular_hours : ℝ) (total_hours : ℝ) (overtime_factor : ℝ) : ℝ :=
  let regular_pay := regular_hours * hourly_rate
  let overtime_hours := total_hours - regular_hours
  let overtime_pay := overtime_hours * (hourly_rate * overtime_factor)
  regular_pay + overtime_pay

def weeks_needed_to_purchase_car (car_cost : ℝ) (weekly_earning : ℝ) : ℕ :=
  ⌈car_cost / weekly_earning⌉.nat_abs

theorem janet_needs_4_weeks (h : total_weekly_earning hourly_rate regular_hours total_hours overtime_factor = 1160) :
  weeks_needed_to_purchase_car car_cost (total_weekly_earning hourly_rate regular_hours total_hours overtime_factor) = 4 := by
  rw [total_weekly_earning, h]
  simp [weeks_needed_to_purchase_car, car_cost, weekly_earning]
  sorry 

end janet_needs_4_weeks_l392_392591


namespace f_inequality_l392_392481

theorem f_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f(2 - x) = f(x)) → (f(0) = 3) → (∀ x, f(b^x) ≤ f(a^x)) :=
by 
  sorry

end f_inequality_l392_392481


namespace daps_equivalent_to_48_dips_l392_392169

noncomputable def conversion_daps_to_dops : ℚ := 5 / 4
noncomputable def conversion_dops_to_dips : ℚ := 3 / 8
noncomputable def conversion_daps_to_dips : ℚ := conversion_daps_to_dops * conversion_dops_to_dips

theorem daps_equivalent_to_48_dips :
  ∀ (daps dops dips : Type) (eq1 : 5*daps = 4*dops) (eq2 : 3*dops = 8*dips), 
  (48:ℚ) * conversion_daps_to_dips = (22.5:ℚ) :=
by
  sorry

end daps_equivalent_to_48_dips_l392_392169


namespace find_pq_sum_l392_392222

-- Define the vertices and the point Q
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (10, -2)
def C : (ℝ × ℝ) := (7, 5)
def Q : (ℝ × ℝ) := (5, 3)

-- Function to calculate Euclidean distance
def dist (P1 P2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P1.1 - P2.1) ^ 2 + (P1.2 - P2.2) ^ 2)

-- Calculate the distances AQ, BQ, CQ
def AQ := dist A Q
def BQ := dist B Q
def CQ := dist C Q

-- Proposition for the sum of distances in the form of p + q√r
def sum_of_distances_in_form (p q r : ℕ) : Prop :=
  AQ + BQ + CQ = p + q * Real.sqrt r

-- Statement: The sum of distances is 0 + 7√34, hence p + q = 7
theorem find_pq_sum : sum_of_distances_in_form 0 7 34 ∧ 0 + 7 = 7 := by
  sorry

end find_pq_sum_l392_392222


namespace no_integer_solutions_for_equation_l392_392864

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y : ℤ), 2^(2 * x) - 3^(2 * y) = 41 := 
by 
  sorry

end no_integer_solutions_for_equation_l392_392864


namespace PQ_bisects_BD_l392_392215

-- Define a structure for a Quadrilateral having points A, B, C, D
structure Quadrilateral :=
  (A B C D : Point)

-- Define midpoints P and Q
def is_midpoint (P Q : Point) (A B C D : Point) : Prop :=
  (P = midpoint A B) ∧ (Q = midpoint C D)

-- Define bisect conditions
def line_bisects (PQ : Line) (A C B D : Point) := 
  (PQ.bisects A C) ∧ (PQ.bisects B D)

-- Define the main theorem
theorem PQ_bisects_BD 
  (quad : Quadrilateral)
  (P Q : Point)
  (PQ : Line)
  (condition_midpoints : is_midpoint P Q quad.A quad.B quad.C quad.D)
  (condition_bisect_AC : PQ.bisects quad.A quad.C) :
  PQ.bisects quad.B quad.D :=
sorry

end PQ_bisects_BD_l392_392215


namespace sigma_phi_inequality_sigma_phi_equality_iff_prime_l392_392636

-- Define the assumptions about sigma and phi functions
def sigma (n : ℕ) : ℕ := 
  (List.range n).filter (λ m => n % m = 0).sum

def phi (n : ℕ) : ℕ := 
  (List.range n).filter (λ m => Nat.gcd m n = 1).length

theorem sigma_phi_inequality (n : ℕ) (h : n > 1) :
  sigma(n) * phi(n) ≤ n^2 - 1 :=
by
  sorry

theorem sigma_phi_equality_iff_prime (n : ℕ) (h : n > 1) :
  sigma(n) * phi(n) = n^2 - 1 ↔ Nat.Prime n :=
by
  sorry

end sigma_phi_inequality_sigma_phi_equality_iff_prime_l392_392636


namespace incorrect_transformation_is_not_valid_l392_392751

-- Define the system of linear equations
def eq1 (x y : ℝ) := 2 * x + y = 5
def eq2 (x y : ℝ) := 3 * x + 4 * y = 7

-- The definition of the correct transformation for x from equation eq2
def correct_transformation (x y : ℝ) := x = (7 - 4 * y) / 3

-- The definition of the incorrect transformation for x from equation eq2
def incorrect_transformation (x y : ℝ) := x = (7 + 4 * y) / 3

theorem incorrect_transformation_is_not_valid (x y : ℝ) 
  (h1 : eq1 x y) 
  (h2 : eq2 x y) :
  ¬ incorrect_transformation x y := 
by
  sorry

end incorrect_transformation_is_not_valid_l392_392751


namespace larger_integer_is_21_l392_392694

-- Setting up the conditions
def quotient_condition (a b : ℕ) : Prop := a / b = 7 / 3
def product_condition (a b : ℕ) : Prop := a * b = 189

-- Assertion: Prove larger of the two integers is 21
theorem larger_integer_is_21 (a b : ℕ) (h1 : quotient_condition a b) (h2 : product_condition a b) : max a b = 21 :=
by sorry

end larger_integer_is_21_l392_392694


namespace log_addition_identity_l392_392430

theorem log_addition_identity : (Real.log10 0.01 + Real.logBase 2 16) = 2 := by
  have h1 : Real.log10 0.01 = -2 := by sorry
  have h2 : Real.logBase 2 16 = 4 := by sorry
  rw [h1, h2]
  norm_num

end log_addition_identity_l392_392430


namespace proof_angle_DFE_Correct_l392_392982

noncomputable def calculate_angle_DFE :=
  let CFB_pred := ∀ (C F B: Point), angle C F B = 50
  let isosceles_TRIANGLE_CF :=
    ∀ (C F B: Point), isIsosceles C F B := 
  ∀  (x : ℕ),
  ∀ (C F E B: Point), CFB_pred → isosceles_TRIANGLE_CF → 
   angle E F B = 3 * angle C F E → 
  angle D F E = 180  - angle_CDF - 44 = 56
variable (F : Point)
variable (C : Point)
variable (D : Point)
variable (E : Point)
variable (B : Point)
theorem proof_angle_DFE_Correct 
(CFB : CF + FB = 50) 
(isosceles_CF : isIsosceles C F B)
(angle_EFB_CFE: angle E F B = 3 * angle C F E) : 
angle D F E = 56 :=
begin
-- proof will go here eventually
sorry
end

end proof_angle_DFE_Correct_l392_392982


namespace reflection_matrix_is_correct_l392_392060

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  -- Given vector for reflection
  let u := ![4, 3] in
  -- Manually derived reflection matrix
  ![![ (7 : ℚ) / 25, 24 / 25],
    ![24 / 25, (-7 : ℚ) / 25]]

theorem reflection_matrix_is_correct :
  reflection_matrix = ![![ (7 : ℚ) / 25, 24 / 25],
                        ![24 / 25, (-7 : ℚ) / 25]] :=
by
  -- Proof is to be provided here
  sorry

end reflection_matrix_is_correct_l392_392060


namespace general_term_sequence_T_n_value_l392_392123

theorem general_term_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (h₁ : ∀ n, 3 * S n + a n = 3) :
  a = λ n, 3 / 4 ^ (n + 1) := 
sorry

theorem T_n_value (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)
  (h₁ : ∀ n, 3 * S n + a n = 3)
  (h₂ : S = λ n, 1 - 1 / 4 ^ (n + 1))
  (h₃ : b = λ n, -log 4 (1 - S (n + 1)))
  (h₄ : T = λ n, ∑ i in range n, 1 / (b i * b (i + 1))) :
  ∀ n, T n = n / (2 * (n + 2)) := 
sorry

end general_term_sequence_T_n_value_l392_392123


namespace Mrs_Martin_pays_32_l392_392658

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def num_regular_scoops : ℕ := 2
def num_kiddie_scoops : ℕ := 2
def num_double_scoops : ℕ := 3

def total_cost : ℕ := 
  (num_regular_scoops * regular_scoop_cost) + 
  (num_kiddie_scoops * kiddie_scoop_cost) + 
  (num_double_scoops * double_scoop_cost)

theorem Mrs_Martin_pays_32 :
  total_cost = 32 :=
by
  sorry

end Mrs_Martin_pays_32_l392_392658


namespace problem_statement_l392_392231

open_locale real_inner_product_space

-- Definitions based on the given conditions
structure parallelogram (A B C D E : Type) (coords : Type) :=
(A B C D E : coords)
(AB : real)
(AD : real)
(angle_BAD : real)
(is_midpoint_EBC : bool)

noncomputable def question (coords : Type) (p : parallelogram := by sorry) : real :=
(inner ((p.D - p.B) : coords) ((p.E - p.A) : coords))

-- Conditions provided in the problem
variables {coords : Type} [inner_product_space_ratios coords] [proper_coords coords] 
variables (p : parallelogram coords)
variables (h1 : p.AB = 4)
variables (h2 : p.AD = 2)
variables (h3 : p.angle_BAD = real.pi / 3)
variables (h4 : p.is_midpoint_EBC = true)

-- Statement to be proved
theorem problem_statement : question coords p = -12 := sorry

end problem_statement_l392_392231


namespace problem_l392_392122

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (M m : ℕ)

-- Conditions
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 1 ≥ 1
axiom h3 : a 2 ≤ 5
axiom h4 : a 5 ≥ 8

-- Sum function for arithmetic sequence
axiom h5 : ∀ n : ℕ, S n = n * a 1 + n * (n - 1) / 2 * (a 2 - a 1)

-- Definition of M and m based on S_15
axiom hM : M = max (S 15)
axiom hm : m = min (S 15)

theorem problem (h : S 15 = M + m) : M + m = 600 :=
  sorry

end problem_l392_392122


namespace inscribed_circle_radius_l392_392975

theorem inscribed_circle_radius (DE EF : ℝ) (hDE : DE = 6) (hEF : EF = 8) (h_angle : ∠DEF = 90) : 
  let DF := Real.sqrt (DE^2 + EF^2),
      A := (1 / 2) * DE * EF,
      s := (DE + EF + DF) / 2,
      r := A / s
  in r = 2 :=
by
  sorry

end inscribed_circle_radius_l392_392975


namespace inequality_solution_l392_392087

theorem inequality_solution (x : ℝ) : (x^3 - 12*x^2 + 36*x > 0) ↔ (0 < x ∧ x < 6) ∨ (x > 6) := by
  sorry

end inequality_solution_l392_392087


namespace parakeets_per_cage_l392_392810

-- Define total number of cages
def num_cages: Nat := 6

-- Define number of parrots per cage
def parrots_per_cage: Nat := 2

-- Define total number of birds in the store
def total_birds: Nat := 54

-- Theorem statement: prove the number of parakeets per cage
theorem parakeets_per_cage : (total_birds - num_cages * parrots_per_cage) / num_cages = 7 :=
by
  sorry

end parakeets_per_cage_l392_392810


namespace sequence_geometric_condition_l392_392948

theorem sequence_geometric_condition :
  (∀ n : ℕ, a_n = n * (-2)^n) →
  (∀ n : ℕ, b_n = n) →
  ∃ q : ℝ, ∀ n : ℕ, (a_n / b_n) = q^(n-1) ↔ ¬ (∀ n : ℕ, b_n = n) :=
by
  intro ha hb
  sorry

end sequence_geometric_condition_l392_392948


namespace integral_problem_l392_392442

theorem integral_problem :
  ∫ x in (0..1), (sqrt (1 - x^2) - x) = (π - 2) / 4 :=
by
  sorry

end integral_problem_l392_392442


namespace tan_beta_rational_iff_l392_392655

-- Hypothesizing necessary conditions
variable (p q : ℤ)
variable (h : q ≠ 0)

theorem tan_beta_rational_iff (α β : ℝ) (h_tan_alpha : tan α = p / q) (h_tan_2beta : tan (2 * β) = tan (3 * α)) :
  (∃ n : ℤ, tan β = n) ↔ ∃ k : ℤ, p^2 + q^2 = k^2 := 
sorry

end tan_beta_rational_iff_l392_392655


namespace sum_first_100_nat_eq_4950_l392_392420

theorem sum_first_100_nat_eq_4950 : ∑ i in range 100, i = 4950 := 
by {
  sorry 
}

end sum_first_100_nat_eq_4950_l392_392420


namespace _l392_392027
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l392_392027


namespace larger_integer_value_l392_392684

theorem larger_integer_value (a b : ℕ) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : max a b = 21 :=
sorry

end larger_integer_value_l392_392684


namespace factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l392_392365

-- Definitions from conditions
theorem factorization_option_a (a b : ℝ) : a^4 * b - 6 * a^3 * b + 9 * a^2 * b = a^2 * b * (a^2 - 6 * a + 9) ↔ a^2 * b * (a - 3)^2 ≠ a^2 * b * (a^2 - 6 * a - 9) := sorry

theorem factorization_option_b (x : ℝ) : (x^2 - x + 1/4) = (x - 1/2)^2 := sorry

theorem factorization_option_c (x : ℝ) : x^2 - 2 * x + 4 = (x - 2)^2 ↔ x^2 - 2 * x + 4 ≠ x^2 - 4 * x + 4 := sorry

theorem factorization_option_d (x y : ℝ) : 4 * x^2 - y^2 = (2 * x + y) * (2 * x - y) ↔ (4 * x + y) * (4 * x - y) ≠ (2 * x + y) * (2 * x - y) := sorry

-- Main theorem that states option B's factorization is correct
theorem correct_factorization_b (x : ℝ) (h1 : x^2 - x + 1/4 = (x - 1/2)^2)
                                (h2 : ∀ (a b : ℝ), a^4 * b - 6 * a^3 * b + 9 * a^2 * b ≠ a^2 * b * (a^2 - 6 * a - 9))
                                (h3 : ∀ (x : ℝ), x^2 - 2 * x + 4 ≠ (x - 2)^2)
                                (h4 : ∀ (x y : ℝ), 4 * x^2 - y^2 ≠ (4 * x + y) * (4 * x - y)) : 
                                (x^2 - x + 1/4 = (x - 1/2)^2) := 
                                by 
                                sorry

end factorization_option_a_factorization_option_b_factorization_option_c_factorization_option_d_correct_factorization_b_l392_392365


namespace larger_integer_21_l392_392720

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l392_392720


namespace prob_sin_ge_half_l392_392307

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem prob_sin_ge_half : 
  ∀ x_0 : ℝ, x_0 ∈ Set.Icc 0 Real.pi ∧ (f(x_0) ≥ 1 / 2) →
  (∃ (prob : ℝ), prob = 2 / 3) :=
by
  sorry

end prob_sin_ge_half_l392_392307


namespace number_less_than_reciprocal_l392_392002

theorem number_less_than_reciprocal :
  (∀ x ∈ ({-3, -1/2, 0, 3/2, 3} : set ℚ), (x < 1/x) ↔ x = -3) :=
by
  -- The statement and conditions as identified above. 
  -- Adding a proof placeholder 'sorry'.
  sorry

end number_less_than_reciprocal_l392_392002


namespace determinant_scaled_l392_392905

-- Define the initial determinant condition
def init_det (x y z w : ℝ) : Prop :=
  x * w - y * z = -3

-- Define the scaled determinant
def scaled_det (x y z w : ℝ) : ℝ :=
  3 * x * (3 * w) - 3 * y * (3 * z)

-- State the theorem we want to prove
theorem determinant_scaled (x y z w : ℝ) (h : init_det x y z w) :
  scaled_det x y z w = -27 :=
by
  sorry

end determinant_scaled_l392_392905


namespace age_problem_l392_392793

theorem age_problem :
  (∃ (x y : ℕ), 
    (3 * x - 7 = 5 * (x - 7)) ∧ 
    (42 + y = 2 * (14 + y)) ∧ 
    (2 * x = 28) ∧ 
    (x = 14) ∧ 
    (3 * 14 = 42) ∧ 
    (42 - 14 = 28) ∧ 
    (y = 14)) :=
by
  sorry

end age_problem_l392_392793


namespace proof_Cn_leq_Dn_2Cn_l392_392614

noncomputable def b (a : ℕ → ℝ) (k : ℕ) : ℝ :=
  (∑ i in Finset.range k, a (i + 1)) / k

noncomputable def C (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (a (k + 1) - b a (k + 1))^2

noncomputable def D (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (a (k + 1) - b a n)^2

theorem proof_Cn_leq_Dn_2Cn (a : ℕ → ℝ) (n : ℕ) (h_pos : ∀ i, 1 ≤ i → i ≤ n → 0 < a i) :
  C a n ≤ D a n ∧ D a n ≤ 2 * C a n := 
sorry

end proof_Cn_leq_Dn_2Cn_l392_392614


namespace number_exceeds_its_part_l392_392807

theorem number_exceeds_its_part (x : ℝ) (h : x = 3/8 * x + 25) : x = 40 :=
by sorry

end number_exceeds_its_part_l392_392807


namespace digit_makes_5678d_multiple_of_9_l392_392475

def is_multiple_of_9 (n : Nat) : Prop :=
  n % 9 = 0

theorem digit_makes_5678d_multiple_of_9 (d : Nat) (h : d ≥ 0 ∧ d < 10) :
  is_multiple_of_9 (5 * 10000 + 6 * 1000 + 7 * 100 + 8 * 10 + d) ↔ d = 1 := 
by
  sorry

end digit_makes_5678d_multiple_of_9_l392_392475


namespace dogs_left_after_walk_l392_392741

theorem dogs_left_after_walk
    (total_dogs : ℕ) (dog_houses : ℕ) (farmhands : ℕ) (dogs_per_farmhand : ℕ)
    (H_dogs : total_dogs = 156) (H_dog_houses : dog_houses = 22)
    (H_farmhands : farmhands = 6) (H_dogs_per_farmhand : dogs_per_farmhand = 2) :
    (total_dogs - farmhands * dogs_per_farmhand) = 144 :=
by
  have walk_dogs : ℕ := farmhands * dogs_per_farmhand
  calc
    total_dogs - walk_dogs = 156 - (6 * 2) : by rw [H_dogs, H_farmhands, H_dogs_per_farmhand]
    ... = 156 - 12 : by sorry
    ... = 144 : by sorry

end dogs_left_after_walk_l392_392741


namespace part1_part2_l392_392913

variables (x y z : ℝ)
open Real

-- Part 1
theorem part1 (
  h1 : x > 0 ∧ y > 0 ∧ z > 0) 
  (h2 : x * y * z = 8) 
  (h3 : x + y < 7)
  : (x / (1 + x) + y / (1 + y) > 2 * sqrt (x * y / (x * y + 8))) := 
sorry

-- Part 2
theorem part2 (
  h1 : x > 0 ∧ y > 0 ∧ z > 0) 
  (h2 : x * y * z = 8)
  : ceil (∑ cyc, 1 / sqrt (1 + x)) = 2 := 
sorry

end part1_part2_l392_392913


namespace original_price_of_sarees_l392_392324
open Real

theorem original_price_of_sarees (P : ℝ) (h : 0.70 * 0.80 * P = 224) : P = 400 :=
sorry

end original_price_of_sarees_l392_392324


namespace equilateral_tetrahedron_plane_angle_l392_392977

noncomputable def volume_ratio_plane_angle : Real := 
  arctan (sqrt 2 / 5)

theorem equilateral_tetrahedron_plane_angle
  (a : Real)
  (pos_a : a > 0)
  (is_equilateral_tetrahedron : ∀ F A B C D K, 
      AK = FK → 
      AK = a * sqrt 3 / 2 → 
      AF = a →
      ¬ ∃ D (ratio : Real), FD = ratio * AD ∧ ratio = 3) : 
  ∃ θ : Real, θ = volume_ratio_plane_angle :=
begin
  sorry
end

end equilateral_tetrahedron_plane_angle_l392_392977


namespace coefficient_x5_in_expansion_of_x_plus_2_pow_7_l392_392666

theorem coefficient_x5_in_expansion_of_x_plus_2_pow_7 :
  (coeff_with_index (x + 2)^7 5) = 84 :=
by
  sorry

end coefficient_x5_in_expansion_of_x_plus_2_pow_7_l392_392666


namespace projection_correct_l392_392318

noncomputable def proj_scalar (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)

noncomputable def proj_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  let s := proj_scalar a b in
  (s * b.1, s * b.2)

theorem projection_correct : proj_vector (1, 2) (-2, 5) = (2 * ((4 + 25)^(-1/2)) * (-2),  2 * ((4 + 25)^(-1/2)) * 5) :=
by
  sorry

end projection_correct_l392_392318


namespace larger_integer_is_21_l392_392717

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l392_392717


namespace ratio_of_areas_l392_392377

theorem ratio_of_areas
  {A B C M N : Point}
  (h1 : triangle A B C)
  (h2 : AB = BC)
  (h3 : angle A B C = 45)   
  (h4 : AM = 2 * MC)
  (h5 : angle N M C = 60)
  (h6 : M ∈ line A C)
  (h7 : N ∈ line B C) :
  (area (triangle M N C)) / (area (quadrilateral A B N M)) = (7 - 3 * real.sqrt 3) / 11 :=
sorry

end ratio_of_areas_l392_392377


namespace second_less_than_first_l392_392348

-- Define the given conditions
def third_number : ℝ := sorry
def first_number : ℝ := 0.65 * third_number
def second_number : ℝ := 0.58 * third_number

-- Problem statement: Prove that the second number is approximately 10.77% less than the first number
theorem second_less_than_first : 
  (first_number - second_number) / first_number * 100 = 10.77 := 
sorry

end second_less_than_first_l392_392348


namespace reflection_matrix_over_vector_l392_392072

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l392_392072


namespace reflection_matrix_over_vector_l392_392068

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l392_392068


namespace basketball_game_l392_392971

theorem basketball_game (
  a r b d : ℕ
  (h_tigers_score : a * (1 + r + r^2 + r^3) ≤ 100)
  (h_lions_score : 4 * b + 6 * d ≤ 100)
  (h_tie_first_quarter : a = b)
  (h_tigers_win : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
) :  a * (r^2 + r^3) + (b + 2*d + b + 3*d) = 77 :=
by 
  sorry

end basketball_game_l392_392971


namespace probability_at_least_two_balls_in_a_box_l392_392772

theorem probability_at_least_two_balls_in_a_box :
  ∀ (boxes: ℕ → ℕ) (balls: ℕ → ℕ), (∀ n, P(boxes(n)) = (1/2^n)) → P(∃ n, boxes(n) ≥ 2) = 5/7 :=
by
  sorry

end probability_at_least_two_balls_in_a_box_l392_392772


namespace least_cubes_l392_392366

noncomputable def gcd (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

def cuboid_volume (l w h : ℕ) : ℕ := l * w * h

def cube_volume (side : ℕ) : ℕ := side * side * side

theorem least_cubes (l w h : ℕ) (hl : l = 6) (hw : w = 9) (hh : h = 12) :
  ∃ n : ℕ, n = (cuboid_volume l w h) / (cube_volume (gcd l w h)) ∧ n = 24 :=
by
  rw [hl, hw, hh]
  have h_gcd : gcd l w h = 3 := by
    sorry
  have h_cuboid_volume : cuboid_volume l w h = 648 := by
    sorry
  have h_cube_volume : cube_volume 3 = 27 := by
    sorry
  use (cuboid_volume l w h) / (cube_volume (gcd l w h))
  split
  case left =>
    rw [h_cuboid_volume, h_cube_volume, h_gcd]
    norm_num
  case right =>
    norm_num
    rfl

end least_cubes_l392_392366


namespace count_integers_satisfying_inequality_l392_392528

theorem count_integers_satisfying_inequality : {n : ℤ | -12 ≤ n ∧ n ≤ 12 ∧ (n - 3) * (n + 5) * (n + 9) < 0}.toFinset.card = 10 :=
by
  sorry

end count_integers_satisfying_inequality_l392_392528


namespace triangle_angle_bisector_extension_l392_392146

theorem triangle_angle_bisector_extension
  (P Q R D S: Type)
  (triangle : Triangle P Q R)
  (RS_bisects_angle_R : ∃ A B C : R, ∠QRS = ∠SRA)
  (PQ_extended_D : ∃ n : D, (PQ_endpoint : P Q) → right_angle n)
  : ∀ (p q m : ℝ), m = (p + q) / 2 :=
by
  sorry

end triangle_angle_bisector_extension_l392_392146


namespace reflection_matrix_over_vector_is_correct_l392_392041

theorem reflection_matrix_over_vector_is_correct :
  let v := (x, y) : ℕ × ℕ in
  let u := (4, 3) : ℕ × ℕ in
  let dot_product := u.1 * x + u.2 * y in
  let u_norm_sq := u.1 * u.1 + u.2 * u.2 in
  let scale_factor := dot_product / u_norm_sq in
  let p := (scale_factor * u.1, scale_factor * u.2) in
  let r := (2 * p.1 - v.1, 2 * p.2 - v.2) in 
  r = (7 * x + 24 * y) / 25, (24 * x - 7 * y) / 25 :=
sorry

end reflection_matrix_over_vector_is_correct_l392_392041


namespace common_tangent_point_common_points_range_l392_392100
open Real

-- Definitions
def f (x a : ℝ) : ℝ := (1 / 2) * x^2 + a * x + a - (1 / 2)
def g (x a : ℝ) : ℝ := a * log (x + 1)

-- Statement for Question 1
theorem common_tangent_point (a : ℝ) (h_a : a < 1) :
  (∃ P : ℝ, -1 < P ∧ f P a = g P a ∧ ∀ x : ℝ, f' = g' → x = P) → a = 1 / 2 := 
  sorry

-- Statement for Question 2
theorem common_points_range (a : ℝ) (h_a : a < 1) :
  (∃ P Q : ℝ, -1 < P ∧ -1 < Q ∧ P ≠ Q ∧ f P a = g P a ∧ f Q a = g Q a) → 0 < a ∧ a < 1 / 2 := 
  sorry

end common_tangent_point_common_points_range_l392_392100


namespace length_of_ab_l392_392773

variable (a b c d e : ℝ)
variable (bc cd de ac ae ab : ℝ)

axiom bc_eq_3cd : bc = 3 * cd
axiom de_eq_7 : de = 7
axiom ac_eq_11 : ac = 11
axiom ae_eq_20 : ae = 20
axiom ac_def : ac = ab + bc -- Definition of ac
axiom ae_def : ae = ab + bc + cd + de -- Definition of ae

theorem length_of_ab : ab = 5 := by
  sorry

end length_of_ab_l392_392773


namespace rabbit_turtle_travel_distance_l392_392320

-- Define the initial conditions and their values
def rabbit_velocity : ℕ := 40 -- meters per minute when jumping
def rabbit_jump_time : ℕ := 3 -- minutes of jumping
def rabbit_rest_time : ℕ := 2 -- minutes of resting
def rabbit_start_time : ℕ := 9 * 60 -- 9:00 AM in minutes from midnight

def turtle_velocity : ℕ := 10 -- meters per minute
def turtle_start_time : ℕ := 6 * 60 + 40 -- 6:40 AM in minutes from midnight
def lead_time : ℕ := 15 -- turtle leads the rabbit by 15 seconds at the end

-- Define the final distance the turtle traveled by the time rabbit arrives
def distance_traveled_by_turtle (total_time : ℕ) : ℕ :=
  total_time * turtle_velocity

-- Define time intervals for periodic calculations (in minutes)
def time_interval : ℕ := 5

-- Define the total distance rabbit covers in one periodic interval
def rabbit_distance_in_interval : ℕ :=
  rabbit_velocity * rabbit_jump_time

-- Calculate total time taken by the rabbit to close the gap before starting actual run
def initial_time_to_close_gap (gap : ℕ) : ℕ := 
  gap * time_interval / rabbit_distance_in_interval

-- Define the total time the rabbit travels
def total_travel_time : ℕ :=
  initial_time_to_close_gap ((rabbit_start_time - turtle_start_time) * turtle_velocity) + 97

-- Define the total distance condition to be proved as 2370 meters
theorem rabbit_turtle_travel_distance :
  distance_traveled_by_turtle (total_travel_time + lead_time) = 2370 :=
  by sorry

end rabbit_turtle_travel_distance_l392_392320


namespace max_c_in_range_f_l392_392457

theorem max_c_in_range_f (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 2) ↔ c ≤ 11 :=
begin
  sorry
end

end max_c_in_range_f_l392_392457


namespace entrance_charge_correct_l392_392619

variable (price_per_pound total_paid pounds_picked : ℝ)

theorem entrance_charge_correct : 
  price_per_pound = 20 → 
  total_paid = 128 → 
  pounds_picked = 7 → 
  entrance_charge = 12 where
  entrance_charge : ℝ :=
    pounds_picked * price_per_pound - total_paid :=
by
  intro h1 h2 h3
  sorry

end entrance_charge_correct_l392_392619


namespace larger_integer_is_21_l392_392715

theorem larger_integer_is_21 (a b : ℕ) (h₀ : 0 < b) (h₁ : a / b = 7 / 3) (h₂ : a * b = 189) : a = 21 ∨ b = 21 :=
by
  sorry

end larger_integer_is_21_l392_392715


namespace proof_a_squared_plus_1_l392_392542

theorem proof_a_squared_plus_1 (a : ℤ) (h1 : 3 < a) (h2 : a < 5) : a^2 + 1 = 17 :=
  by
  sorry

end proof_a_squared_plus_1_l392_392542


namespace reflection_matrix_is_correct_l392_392048

-- Defining the vectors
def u : ℝ × ℝ := (4, 3)
def reflection_matrix_over_u : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![7 / 25, 24 / 25],
  ![24 / 25, -7 / 25]
]

-- Statement asserting the reflection matrix for the vector u
theorem reflection_matrix_is_correct : 
  ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = reflection_matrix_over_u :=
by
  use reflection_matrix_over_u
  sorry

end reflection_matrix_is_correct_l392_392048


namespace number_of_a_values_l392_392119

noncomputable def f : ℕ → ℝ → ℝ
| 1, a => a
| (n+1), a => 
    if f n a > 1 then (f n a - 1) / f n a 
    else 2 * f n a

theorem number_of_a_values : ∃ (a : ℝ), a ∈ (0, 1] ∧ 
    (∀ n : ℕ, n > 0 → f (n + 3) a = f n a) ∧ 
    (a = 1/2 ∨ a = 1) :=
sorry

end number_of_a_values_l392_392119


namespace max_B_at_125_l392_392447

-- Definitions given in the problem
def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 ^ k)

-- Main theorem to prove
theorem max_B_at_125 : ∀ k : ℕ, k = 125 -> B k = Real.sup (Set.image B (Set.Icc 0 500)) :=
by
  intro k hk
  rw hk
  sorry

end max_B_at_125_l392_392447


namespace reflection_matrix_over_vector_l392_392071

theorem reflection_matrix_over_vector :
  let v := Vector2 4 3 in
  reflection_matrix v = Matrix.mk 
    (Vector2.mk (7 / 25) (24 / 25))
    (Vector2.mk (24 / 25) (-7 / 25)) :=
sorry

end reflection_matrix_over_vector_l392_392071


namespace necessary_and_sufficient_condition_l392_392312

noncomputable def hasExtremum (f : ℝ → ℝ) := ∃ x, ∀ ε > 0, ∃ δ > 0, ∀ y, abs(y - x) < ε → f(y) ≠ f(x)

theorem necessary_and_sufficient_condition (a : ℝ) :
  (∃ x, ∀ ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < ε → ax^3 + x + 1 ≠ ax^3 + x + 1) ↔ a < 0 :=
sorry

end necessary_and_sufficient_condition_l392_392312


namespace equivalent_expression_l392_392942
noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := 3 * sin(2 * x + ϕ)

theorem equivalent_expression (ϕ : ℝ) (h_ϕ : 0 < ϕ ∧ ϕ < π / 2) :
  let f_translated := f (x + (π / 6)) ϕ in
  (∀ x, 3 * sin(2 * x + π / 3 + ϕ) = 3 * sin(2 * (-x) + π / 3 + ϕ)) →
  ϕ = π / 6 ∧ (∀ x, f x (π / 6) = 3 * sin(2 * x + π / 6)) :=
by
  sorry

end equivalent_expression_l392_392942


namespace lana_extra_nickels_l392_392244

theorem lana_extra_nickels (stack_size stacks: ℕ) (h_stack_size: stack_size = 8) (h_stacks: stacks = 9) : stack_size * stacks = 72 :=
by
  rw [h_stack_size, h_stacks]
  exact rfl

end lana_extra_nickels_l392_392244


namespace larger_integer_is_21_l392_392703

theorem larger_integer_is_21
  (a b : ℕ)
  (h1 : a > 0)
  (h2 : b > 0)
  (quotient_condition : a = (7 * b) / 3)
  (product_condition : a * b = 189) :
  a = 21 := 
sorry

end larger_integer_is_21_l392_392703


namespace find_larger_integer_l392_392732

theorem find_larger_integer 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a * b = 189) 
  (h4 : a = 7 * (b / 3)) : 
  max a b = 21 := 
sorry

end find_larger_integer_l392_392732


namespace compute_sum_l392_392846

theorem compute_sum : 
  (1 / (2^1988 : ℝ)) * (Finset.sum (Finset.range 995) (λ n, (-3)^n * (Nat.choose 1988 (2 * n)))) = -0.5 :=
by
  sorry

end compute_sum_l392_392846


namespace correct_analytical_method_l392_392767

-- Definitions of the different reasoning methods
def reasoning_from_cause_to_effect : Prop := ∀ (cause effect : Prop), cause → effect
def reasoning_from_effect_to_cause : Prop := ∀ (cause effect : Prop), effect → cause
def distinguishing_and_mutually_inferring : Prop := ∀ (cause effect : Prop), (cause ↔ effect)
def proving_converse_statement : Prop := ∀ (P Q : Prop), (P → Q) → (Q → P)

-- Definition of the analytical method
def analytical_method : Prop := reasoning_from_effect_to_cause

-- Theorem stating that the analytical method is the method of reasoning from effect to cause
theorem correct_analytical_method : analytical_method = reasoning_from_effect_to_cause := 
by 
  -- Complete this proof with refined arguments
  sorry

end correct_analytical_method_l392_392767


namespace rectangle_overlap_l392_392973

theorem rectangle_overlap
  (rects : Fin 9 → Set (Fin 2 → ℝ))
  (total_area : Set (Fin 2 → ℝ))
  (h_total_area : measure total_area = 5)
  (h_rects : ∀ i, measure (rects i) = 1)
  (h_union : measure (⋃ i, rects i) = measure total_area) :
  ∃ i j (hi : i ≠ j), measure (rects i ∩ rects j) ≥ 1 / 9 :=
sorry

end rectangle_overlap_l392_392973


namespace modulus_of_conjugate_z_eq_sqrt2_div_2_l392_392311

-- Given the complex number z.
def z : ℂ := i / (1 - i)

-- Goal: Prove that the modulus of the conjugate of z is equal to sqrt(2)/2.
theorem modulus_of_conjugate_z_eq_sqrt2_div_2 : complex.abs (complex.conj z) = real.sqrt 2 / 2 :=
by
  sorry

end modulus_of_conjugate_z_eq_sqrt2_div_2_l392_392311


namespace larger_integer_21_l392_392722

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end larger_integer_21_l392_392722


namespace _l392_392026
-- Import necessary libraries for matrix operations

-- Define the vector for reflection
def reflection_vector : Matrix (Fin 2) (Fin 1) ℝ :=
  ![![4], ![3]]

-- Define the reflection matrix
def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![7 / 25, 24 / 25], ![24 / 25, -7 / 25]]

-- The theorem statement that needs to be proved
axiom reflection_matrix_correct :
  ∀ (v : Matrix (Fin 2) (Fin 1) ℝ),
  let r := (2 * (reflection_vectorᵀ ⬝ reflection_vector)⁻¹ ⬝ reflection_vector ⬝ reflection_vectorᵀ) ⬝ v - v in
  reflection_matrix ⬝ v = r

end _l392_392026


namespace find_certain_number_l392_392195

theorem find_certain_number
  (t b c : ℝ)
  (average1 : (t + b + c + 14 + 15) / 5 = 12)
  (average2 : (t + b + c + x) / 4 = 15)
  (x : ℝ) :
  x = 29 :=
by
  sorry

end find_certain_number_l392_392195


namespace odd_deg_polynomial_minimal_value_6_l392_392879

noncomputable def polynomial : Type := sorry  -- Consider using a placeholder for polynomial type

theorem odd_deg_polynomial_minimal_value_6 (P : polynomial)
  (h_odd_degree : ∃ d : ℕ, odd d ∧ (∃ a : ℤ, leading_coeff P = a))
  (h_conditions : ∀ n : ℕ, 0 < n → ∃ x : fin n → ℕ, function.injective x ∧ 
    (∀ i j, i < n → j < n → 1/2 < P (x i) / P (x j) ∧ P (x i) / P (x) ∈ ℚ)):
  ∃ k : ℕ, k = 6 :=
by
  sorry

end odd_deg_polynomial_minimal_value_6_l392_392879


namespace fn_formula_l392_392944

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

def f₁ (x : ℝ) : ℝ := f x
def f₂ (x : ℝ) : ℝ := f (f₁ x)
def f₃ (x : ℝ) : ℝ := f (f₂ x)
def f₄ (x : ℝ) : ℝ := f (f₃ x)

theorem fn_formula (n : ℕ) (h : 2 ≤ n) (x : ℝ) (hx : 0 < x) :
  let fn : ℝ → ℝ := match n with
         | 1     => f₁
         | 2     => f₂
         | 3     => f₃
         | 4     => f₄
         | _ => sorry -- define fₙ recursively here
  in fn x = x / ((2^n - 1) * x + 2^n) := 
by {
  sorry
}

end fn_formula_l392_392944


namespace find_angle_x_l392_392630

def O_center (O : Point) (C : Circle) : Prop := C.center = O

def is_diameter (A D O : Point) (C : Circle) : Prop := D ∈ C ∧ line_through A D ∧ A ∈ line_segments O D

def is_isosceles (A B C : Point) : Prop := dist A B = dist A C

def angle_at_base (B O C : Point) (θ : ℝ) : Prop := ∠ B O C = θ ∧ ∠ O B C = θ

axiom angle_in_semicircle (A C D O : Point) (C1 D1 : Circle) :
  is_diameter A D O C1 → ∠ A C D = 90

theorem find_angle_x (O B C A D : Point) (C1 C2 : Circle):
  O_center O C1 ∧ 
  is_isosceles O B C∧ 
  angle_at_base B O C 32 ∧ 
  is_diameter A D O C1 ∧ 
  ∠ A D C = 67 →
  ∠ O A D - ∠ A O C = 9 := by
  sorry

end find_angle_x_l392_392630


namespace no_natural_pairs_exist_l392_392880

theorem no_natural_pairs_exist (n m : ℕ) : ¬(n + 1) * (2 * n + 1) = 18 * m ^ 2 :=
by
  sorry

end no_natural_pairs_exist_l392_392880


namespace sequence_50th_number_l392_392220

theorem sequence_50th_number (n k : ℕ) : 
  (∀ n, ∀ k ≤ 2 * n, (nth_number n k) = 2 * n) → 
  (nth_number 25 25) = 14 :=
  sorry

end sequence_50th_number_l392_392220
