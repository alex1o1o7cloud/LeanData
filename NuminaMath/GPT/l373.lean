import Mathlib

namespace base_angle_isosceles_triangle_l373_373996

theorem base_angle_isosceles_triangle (x : ℝ) :
    (∀ (Δ : Triangle) (a b c : ℝ),
        Δ.Angles = [a, b, c] ∧ a = 80 ∧ IsIsosceles Δ → 
        (b = 80 ∨ b = 50)) :=
by
    -- Assuming AngleSum and IsIsosceles are defined appropriately in the context
    sorry

end base_angle_isosceles_triangle_l373_373996


namespace proposition_B_proposition_C_proposition_D_l373_373757

theorem proposition_B (I : Set ℝ) (f : ℝ → ℝ) (h_mono : ∀ x1 x2 ∈ I, x1 < x2 → f x1 < f x2):
  ∀ x1 x2 ∈ I, f x1 = f x2 → x1 = x2 :=
by sorry

theorem proposition_C (f g : ℝ → ℝ)
  (hf : ∀ x, f x = (x^4 - 1) / (x^2 + 1))
  (hg : ∀ x, g x = x^2 - 1):
  ∀ x, f x = g x :=
by sorry

theorem proposition_D (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0):
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (a * c) :=
by sorry

def main : String := "Propositions B, C, and D are true."

end proposition_B_proposition_C_proposition_D_l373_373757


namespace new_computer_price_l373_373589

-- Define the initial conditions
def initial_price_condition (x : ℝ) : Prop := 2 * x = 540

-- Define the calculation for the new price after a 30% increase
def new_price (x : ℝ) : ℝ := x * 1.30

-- Define the final proof problem statement
theorem new_computer_price : ∃ x : ℝ, initial_price_condition x ∧ new_price x = 351 :=
by sorry

end new_computer_price_l373_373589


namespace proof_part1_proof_part2_l373_373474

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373474


namespace bill_cooking_time_l373_373351

def total_time_spent 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
num_peppers * chop_pepper_time + 
num_onions * chop_onion_time + 
num_omelets * grate_cheese_time + 
num_omelets * cook_omelet_time

theorem bill_cooking_time 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ)
  (chop_pepper_time_eq : chop_pepper_time = 3)
  (chop_onion_time_eq : chop_onion_time = 4)
  (grate_cheese_time_eq : grate_cheese_time = 1)
  (cook_omelet_time_eq : cook_omelet_time = 5)
  (num_peppers_eq : num_peppers = 4)
  (num_onions_eq : num_onions = 2)
  (num_omelets_eq : num_omelets = 5) :
  total_time_spent chop_pepper_time chop_onion_time grate_cheese_time cook_omelet_time num_peppers num_onions num_omelets = 50 :=
by {
  sorry
}

end bill_cooking_time_l373_373351


namespace urea_formation_l373_373406

theorem urea_formation
  (CO2 NH3 Urea : ℕ) 
  (h_CO2 : CO2 = 1)
  (h_NH3 : NH3 = 2) :
  Urea = 1 := by
  sorry

end urea_formation_l373_373406


namespace find_n_positive_integers_l373_373016

theorem find_n_positive_integers :
  ∀ n : ℕ, 0 < n →
  (∃ k : ℕ, (n^2 + 11 * n - 4) * n! + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by
  sorry

end find_n_positive_integers_l373_373016


namespace us_supermarkets_count_l373_373281

variable (C : ℕ) -- C is the number of supermarkets in Canada
variable (US Supermarkets : ℕ) -- US Supermarkets is the number of supermarkets in the US

-- Given conditions
def total_supermarkets : ℕ := 84
def us_more_than_canada : ℕ := 14
def total_sums_to_84 (C : ℕ) : Prop := C + (C + us_more_than_canada) = total_supermarkets

-- Target statement to prove
theorem us_supermarkets_count (h : total_sums_to_84 C) : US Supermarkets = C + us_more_than_canada := 
by {
  sorry -- proof will follow from conditions
}

end us_supermarkets_count_l373_373281


namespace remaining_cube_height_l373_373690

/-- Given a cube with side length 2 units, where a corner is chopped off such that the cut runs
    through points on the three edges adjacent to a selected vertex, each at 1 unit distance
    from that vertex, the height of the remaining portion of the cube when the freshly cut face 
    is placed on a table is equal to (5 * sqrt 3) / 3. -/
theorem remaining_cube_height (s : ℝ) (h : ℝ) : 
    s = 2 → h = 1 → 
    ∃ height : ℝ, height = (5 * Real.sqrt 3) / 3 := 
by
    sorry

end remaining_cube_height_l373_373690


namespace solve_system_l373_373206

variable {a b c : ℝ}
variable {x y z : ℝ}
variable {e1 e2 e3 : ℤ} -- Sign variables should be integers to express ±1 more easily 

axiom ax1 : x * (x + y) + z * (x - y) = a
axiom ax2 : y * (y + z) + x * (y - z) = b
axiom ax3 : z * (z + x) + y * (z - x) = c

theorem solve_system :
  (e1 = 1 ∨ e1 = -1) ∧ (e2 = 1 ∨ e2 = -1) ∧ (e3 = 1 ∨ e3 = -1) →
  x = (1/2) * (e1 * Real.sqrt (a + b) - e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) ∧
  y = (1/2) * (e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) - e3 * Real.sqrt (c + a)) ∧
  z = (1/2) * (-e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) :=
sorry -- proof goes here

end solve_system_l373_373206


namespace common_area_rectangle_circle_l373_373781

open Real

theorem common_area_rectangle_circle
  (l w : ℝ) (r : ℝ) (h1 : l = 10) (h2 : w = 4) (h3 : r = 3) 
  (c : ℝ) (h4 : c = π * r ^ 2) :
  c = 9 * π :=
by
  rw [h3] at h4
  rw [← mul_assoc, pow_two] at h4
  simp at h4
  exact h4

end common_area_rectangle_circle_l373_373781


namespace sum_two_digit_divisors_of_105_l373_373158

theorem sum_two_digit_divisors_of_105 : 
  (∑ d in (Finset.filter (λ d, 10 ≤ d ∧ d < 100) (Finset.divisors 105)), d) = 71 :=
by
  sorry

end sum_two_digit_divisors_of_105_l373_373158


namespace nice_number_bound_l373_373285

def is_nice (k : ℕ) (ℓ : ℕ) : Prop :=
  ∃ m : ℕ, k! + ℓ = m^2

theorem nice_number_bound (ℓ : ℕ) (n : ℕ) (hℓ_pos : ℓ > 0) (hn_ell : n ≥ ℓ) :
  (finset.filter (λ k, is_nice k ℓ) (finset.range (n^2 + 1))).card ≤ n^2 - n + ℓ := 
sorry

end nice_number_bound_l373_373285


namespace problem_conditions_question1_question2_min_max_l373_373448

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 1) * (2 * Real.log (Real.sqrt x) / Real.log 2 - 2)

theorem problem_conditions (x : ℝ) : 9^x - 4 * 3^(x+1) + 27 ≤ 0 :=
sorry

theorem question1 (x : ℝ) (h : 9^x - 4 * 3^(x+1) + 27 ≤ 0) : 1 ≤ x ∧ x ≤ 2 :=
sorry

theorem question2_min_max (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
(∀ x ∈ Set.Icc 1 2, 0 ≤ f x ∧ f x ≤ 2) ∧ f 2 = 0 ∧ f 1 = 2 :=
sorry

end problem_conditions_question1_question2_min_max_l373_373448


namespace compute_expression_l373_373410

variables (a b c d e : ℝ)
variables (h1 : a < d) (h2 : d < c) (h3 : c < e) (h4 : e < b)

def M (x y : ℝ) := max x y
def m (x y : ℝ) := min x y

theorem compute_expression : M (m b (M a d)) (m (M a e) c) = c :=
by {
  have h5 : M a d = d := max_eq_right (le_of_lt h1),
  have h6 : m b d = d := min_eq_right (le_of_lt (lt_trans h2 (lt_trans h3 h4))),
  have h7 : M a e = e := max_eq_right (le_of_lt (lt_trans h2 (lt_trans h3 h4))),
  have h8 : m e c = c := min_eq_left (le_of_lt h3),
  rw [h5, h6, h7, h8],
  exact max_eq_right (le_of_lt h2),
}

end compute_expression_l373_373410


namespace g_property_multiplication_l373_373650

theorem g_property_multiplication :
  (∃ g : ℝ → ℝ, (∀ x y z : ℝ, g(x^2 + y + z * g(z)) = x*y + g(x) + z * g(y)) ∧ 
   let g3_values := {g 3 | ∀ x y z : ℝ, g(x^2 + y + z * g(z)) = x * y + g(x) + z * g(y)} in
   g3_values.card = 2 ∧ g3_values.sum = 3) →
  6 :=
by
  sorry

end g_property_multiplication_l373_373650


namespace area_of_triangle_ABC_l373_373389

theorem area_of_triangle_ABC (A B C : Type) [HasDist A B]
  (hypotenuse_length : Real) (angleB angleC : Real)
  (h_B : angleB = 90) (h_C : angleC = 45) (h_hypotenuse : hypotenuse_length = 14) :
  let leg_length := (hypotenuse_length / Real.sqrt 2),
  let area := (1 / 2) * leg_length * leg_length
  in area = 49 :=
by
  sorry

end area_of_triangle_ABC_l373_373389


namespace sqrt_subtraction_l373_373260

theorem sqrt_subtraction:
  ∀ a b c d : ℝ, a = 49 + 121 → b = 36 - 9 → sqrt a - sqrt b = sqrt 170 - 3 * sqrt 3 :=
by
  intros a b c d ha hb
  rw [ha, hb]
  sorry

end sqrt_subtraction_l373_373260


namespace max_diff_in_february_l373_373189

noncomputable def max_sales_percentage_difference_month (D B : ℕ → ℕ) : ℕ :=
  let perc_diff (a b : ℕ) : ℝ := (↑(max a b) - ↑(min a b)) / ↑(min a b) * 100
  let months := [{ month := 1, D := D 1, B := B 1},
                 { month := 2, D := D 2, B := B 2},
                 { month := 3, D := D 3, B := B 3},
                 { month := 4, D := D 4, B := B 4},
                 { month := 5, D := D 5, B := B 5}]
  let max_diff_month := months.maxBy? (λ m => perc_diff m.D m.B)
  match max_diff_month with
  | some m => m.month
  | none   => 0 -- default case, should not happen in this example

def D (m : ℕ) : ℕ :=
  match m with
  | 1 => 4 -- January
  | 2 => 5 -- February
  | 3 => 4 -- March
  | 4 => 3 -- April
  | 5 => 2 -- May
  | _ => 0

def B (m : ℕ) : ℕ :=
  match m with
  | 1 => 3 -- January
  | 2 => 3 -- February
  | 3 => 4 -- March
  | 4 => 4 -- April
  | 5 => 3 -- May
  | _ => 0

theorem max_diff_in_february : max_sales_percentage_difference_month D B = 2 := by
  sorry

end max_diff_in_february_l373_373189


namespace peaches_total_l373_373293

def peaches_in_basket (a b : Nat) : Nat :=
  a + b 

theorem peaches_total (a b : Nat) (h1 : a = 20) (h2 : b = 25) : peaches_in_basket a b = 45 := 
by
  sorry

end peaches_total_l373_373293


namespace range_of_a_l373_373045

def x : ℝ := arbitrary ℝ  -- Define x as an arbitrary real number

def p (x : ℝ) : Prop := |x + 1| > 2  -- Condition p

def q (x a : ℝ) : Prop := x > a  -- Condition q

theorem range_of_a (a : ℝ) : ∀ x : ℝ, (¬(p x) → ¬(q x a)) → 1 ≤ a :=
by
  assume x,
  intro H,
  sorry  -- Proof goes here

end range_of_a_l373_373045


namespace equal_angles_l373_373772

open Classical
open Geometry

variables (ABC : Triangle) (H K L : Point)
variable [AcuteTriangle ABC]

-- Given that AH is the altitude of the acute triangle ABC
variable [Altitude H ABC]

-- K and L are the feet of the perpendiculars from point H on sides AB and AC respectively
variable (foot1 : FootOfPerpendicular H (side AB ABC) = K)
variable (foot2 : FootOfPerpendicular H (side AC ABC) = L)

theorem equal_angles (ABC : Triangle) (H K L : Point)
  [Altitude H ABC] [FootOfPerpendicular H (side AB ABC) = K] [FootOfPerpendicular H (side AC ABC) = L] :
  ∠ BKC = ∠ BLC :=
sorry

end equal_angles_l373_373772


namespace sally_onions_proof_l373_373674

theorem sally_onions_proof :
  ∀ (Sara Fred Total Sally : ℕ), Sara = 4 → Fred = 9 → Total = 18 → Sally = Total - (Sara + Fred) → Sally = 5 :=
by
  intros Sara Fred Total Sally hSara hFred hTotal hSally,
  rw [hSara, hFred, hTotal] at hSally,
  exact hSally

end sally_onions_proof_l373_373674


namespace cylinder_volume_scaling_l373_373756

theorem cylinder_volume_scaling (r h : ℝ) (V : ℝ) (V' : ℝ) 
  (h_original : V = π * r^2 * h) 
  (h_new : V' = π * (1.5 * r)^2 * (3 * h)) :
  V' = 6.75 * V := by
  sorry

end cylinder_volume_scaling_l373_373756


namespace ratio_and_lengths_l373_373611

variables (ABCD : Type) [IsTrapezoid ABCD]
variables (A B C D E : Point ABCD)
variables (AC BD : Line ABCD)
variables (BC AD BE ED EC AE : ℝ)

def perpendicular_diag : Prop := IsPerpendicular AC BD
def given_conditions : Prop := 
  BC = 10 ∧ 
  AD = 30 ∧ 
  perpendicular_diag AC BD ∧ 
  IsOnTrapezoid ABCD A B C D ∧ 
  IsOnDiag AC A C ∧ 
  IsOnDiag BD B D

theorem ratio_and_lengths 
  (cond : given_conditions) 
  (h1 : AE * EC = 108) : 
  BE / ED = 1 / 3 ∧ 
  EC = sqrt (100 - BE^2) ∧ 
  AE = 3 * sqrt(100 - BE^2) ∧ 
  ED = 3 * BE ∧ 
  BE * ED = 192 := 
  by {
    sorry
  }

end ratio_and_lengths_l373_373611


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373496

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373496


namespace f_is_odd_g_formula_l373_373944

-- Problem 1: Prove that if g(x) = f(x) + x is an odd function, then f is odd.
theorem f_is_odd (g f : ℝ → ℝ) (h : ∀ x, g x = f x + x)
  (h_odd_g : ∀ x, g (-x) = -g x) : ∀ x, f (-x) = -f x := by
  -- Proof omitted
  sorry

-- Problem 2: Given particular f, prove the formula for g(x) when x < 0.
theorem g_formula (f g : ℝ → ℝ) 
  (h_f : ∀ x > 0, f x = real.log x / real.log 2)
  (h_odd_f : ∀ x, f (-x) = -f x)
  (h_g : ∀ x, g x = f x + x) 
  (x : ℝ) (hx : x < 0) : g x = x - (real.log (-x) / real.log 2) := by
  sorry

end f_is_odd_g_formula_l373_373944


namespace largest_angle_in_hexagon_l373_373702

theorem largest_angle_in_hexagon (x : ℝ) 
  (h_ratio : [2, 2, 2, 3, 3, 4] = [2, 2, 2, 3, 3, 4].map (λ r, r * x))
  (h_sum : 2 * x + 2 * x + 2 * x + 3 * x + 3 * x + 4 * x = 720) :
  4 * x = 180 := 
sorry

end largest_angle_in_hexagon_l373_373702


namespace equal_radius_tangent_circles_l373_373913

structure Circle := (center : Point) (radius : ℝ)
structure Line := (point : Point) (direction : Vector)

structure Construction (L1 M1 : Point) (r : ℝ) :=
  (tangency_k : ∃ k : Circle, k.center = L1 ∧ k.radius = r ∧ PointOnCircle(k, P))
  (tangency_e : ∃ e : Line, e.point = Q ∧ IsTangentTo(e, M1, r))
  (common_tangency : dist(L1, M1) = 2 * r)
  (similar_quad : IsSimilarQuadrilateral(P, Q, M1, L1))

theorem equal_radius_tangent_circles (k : Circle) (e : Line) (P : Point) (Q : Point) :
  ∃ L1 M1 r, Construction L1 M1 r := 
begin
  sorry
end

end equal_radius_tangent_circles_l373_373913


namespace range_of_b_if_f_increasing_g_has_one_zero_l373_373454

-- Define function f
def f (x : ℝ) (b : ℝ) : ℝ := log x + x^2 - b * x

-- Condition (1) to prove the range of b if f is increasing on its domain
theorem range_of_b_if_f_increasing (b : ℝ) :
  (∀ x > 0, (1 / x + 2 * x - b) ≥ 0) → b ≤ 2 :=
sorry

-- Define function g when b = -1
def g (x : ℝ) : ℝ := log x - x^2 + x

-- Condition (2) to prove g has only one zero at x = 1 / sqrt 3
theorem g_has_one_zero :
  ∃! x > 0, g x = 0 :=
sorry

end range_of_b_if_f_increasing_g_has_one_zero_l373_373454


namespace remainder_of_sum_1_to_20_divided_by_9_is_3_l373_373750

-- Lean statement to prove the given problem
theorem remainder_of_sum_1_to_20_divided_by_9_is_3 :
  (∑ k in Finset.range 21, k) % 9 = 3 :=
by
  sorry

end remainder_of_sum_1_to_20_divided_by_9_is_3_l373_373750


namespace find_x_l373_373299

theorem find_x (n x : ℚ) (h1 : 3 * n + x = 6 * n - 10) (h2 : n = 25 / 3) : x = 15 :=
by
  sorry

end find_x_l373_373299


namespace find_a_if_g_even_l373_373068

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then x - 1 else if -2 ≤ x ∧ x ≤ 0 then -1 else 0

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (f x) + a * x

theorem find_a_if_g_even (a : ℝ) : (∀ x : ℝ, f x + a * x = f (-x) + a * (-x)) → a = -1/2 :=
by
  intro h
  sorry

end find_a_if_g_even_l373_373068


namespace remainder_2468135792_mod_101_l373_373743

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end remainder_2468135792_mod_101_l373_373743


namespace proof_problem_l373_373451

-- Given conditions
def ellipse (a b : ℝ) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
def focus (F : ℝ × ℝ) := F = (1, 0)
def point_on_ellipse (a b : ℝ) := (-1, sqrt 2 / 2)
def a_gt_b (a b : ℝ) := a > b ∧ b > 0

-- Definitions inferred from conditions
def ellipse_definition := ellipse 2 1
def focus_definition := focus (1, 0)
def point_on_ellipse_definition := point_on_ellipse 2 1
def a_gt_b_definition := a_gt_b 2 1

-- Main theorem statement
theorem proof_problem :
  (ellipse 2 1) = ellipse_definition ∧
  (focus (1, 0)) = focus_definition ∧
  (point_on_ellipse 2 1) = point_on_ellipse_definition ∧
  (∃ (Q : ℝ × ℝ), Q = (5 / 4, 0) ∧ 
    ∀ (A B : ℝ × ℝ), (A.1 - Q.1, A.2) • (B.1 - Q.1, B.2) = -7 / 16) :=
by sorry

end proof_problem_l373_373451


namespace arithmetic_sequence_a1_range_l373_373430

theorem arithmetic_sequence_a1_range
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_d_range : d ∈ set.Ico (-1 : ℝ) 0)
  (h_eq : (sin (a 6))^2 * (cos (a 9))^2 - (sin (a 9))^2 * (cos (a 6))^2 = sin (a 7 + a 8)) :
  (4 * real.pi / 3 < a 1 ∧ a 1 < 3 * real.pi / 2) := 
sorry

end arithmetic_sequence_a1_range_l373_373430


namespace modulus_complex_solution_l373_373952

theorem modulus_complex_solution (x y : ℝ) (h : (1 + 2 * complex.I) * x = 2 + y * complex.I) : complex.abs (x + y * complex.I) = 2 * Real.sqrt 5 :=
sorry

end modulus_complex_solution_l373_373952


namespace range_of_g_le_2_minus_x_l373_373417

noncomputable def f (x : ℝ) : ℝ := x^2

noncomputable def g (x : ℝ) : ℝ :=
if x ≥ 0 then f x else -f (-x)

theorem range_of_g_le_2_minus_x : {x : ℝ | g x ≤ 2 - x} = {x : ℝ | x ≤ 1} :=
by sorry

end range_of_g_le_2_minus_x_l373_373417


namespace number_of_valid_arrangements_l373_373890

theorem number_of_valid_arrangements :
  let students := ["A", "B", "C", "D", "E"]
  let total_students := 5
  let end_positions := [0, total_students - 1]
  let adjacent_positions := λ l : List Nat, ∃ s1 s2, l.get! s1 = "C" ∧ l.get! s2 = "D" ∧ (s1 = s2 + 1 ∨ s2 = s1 + 1)
  let valid_positions := λ l : List Nat, ∀ i ∈ end_positions, l.get! i ≠ "A"
  students.nodup ∧ adjacent_positions students ∧ valid_positions students ↔ length (List.permutations students) = 24
:= sorry

end number_of_valid_arrangements_l373_373890


namespace tetrahedron_inequality_l373_373648

-- Define the types and objects for our problem
variables {T : Type} [tetrahedron T]
variables (A1 A2 A3 A4 : T) (I : point T)
variables (S : fin 4 → ℝ) (S' : fin 4 × fin 4 → ℝ)

-- State the theorem
theorem tetrahedron_inequality (A_incenter : is_incenter I (A1, A2, A3, A4))
    (S_face_opposite : ∀ i, S i = area_face_opposite I (A1, A2, A3, A4) i)
    (S'_triangle_area : ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 → S' (i, j) = area_triangle (triangle I (vertex_of i) (vertex_of j))) :
    (∑ 1 ≤ k < j ≤ 4, S' (k, j)) ≤ ( √6 / 4 ) * ( ∑ i, S i ) :=
by sorry

end tetrahedron_inequality_l373_373648


namespace parabola_focus_distance_max_area_triangle_l373_373557

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373557


namespace population_size_is_120_l373_373315

-- The total number of individuals in the population
constants (A B : Type) -- represent the layers
constants (population : Type) -- represent the entire population
constants (total_population_size sample_size : Nat)
constants (prob_layer_B : ℚ)

-- Given conditions
axiom population_partitions : A ⊕ B = population -- partition of the population into two layers
axiom stratified_sampling : true -- using stratified sampling method
axiom sample_capacity : sample_size = 10 -- sample capacity of 10
axiom prob_individual_B : prob_layer_B = 1 / 12 -- probability of each individual in layer B being drawn

-- Proof problem statement
theorem population_size_is_120 :
  (sample_size / prob_layer_B).num = 120 :=
  sorry

end population_size_is_120_l373_373315


namespace unique_non_zero_b_for_unique_x_solution_l373_373370

theorem unique_non_zero_b_for_unique_x_solution (c : ℝ) (hc : c ≠ 0) :
  c = 3 / 2 ↔ ∃! b : ℝ, b ≠ 0 ∧ ∃ x : ℝ, (x^2 + (b + 3 / b) * x + c = 0) ∧ 
  ∀ x1 x2 : ℝ, (x1^2 + (b + 3 / b) * x1 + c = 0) ∧ (x2^2 + (b + 3 / b) * x2 + c = 0) → x1 = x2 :=
sorry

end unique_non_zero_b_for_unique_x_solution_l373_373370


namespace exist_plane_parallel_line_l373_373027

open_locale classical

-- Definitions for non-intersecting lines a and b.
variables {a b : Type*} [affine_space ℝ ℝ a] [affine_space ℝ ℝ b]

-- Non-intersecting space lines definition
def non_intersecting_space_lines (a b : Type*) [affine_space ℝ ℝ a] [affine_space ℝ ℝ b] : Prop :=
  ∀ (p : a) (q : b), p ≠ q

-- Hypothesis that a and b are non-intersecting
axiom non_intersecting_ab : non_intersecting_space_lines a b

-- Proof statement
theorem exist_plane_parallel_line (a b : Type*) [affine_space ℝ ℝ a] [affine_space ℝ ℝ b] 
  (h : non_intersecting_space_lines a b) : ∃ α : set (affine_space ℝ ℝ), a ⊆ α ∧ (b ∥ α) :=
sorry

end exist_plane_parallel_line_l373_373027


namespace expected_disease_count_l373_373192

/-- Define the probability of an American suffering from the disease. -/
def probability_of_disease := 1 / 3

/-- Define the sample size of Americans surveyed. -/
def sample_size := 450

/-- Calculate the expected number of individuals suffering from the disease in the sample. -/
noncomputable def expected_number := probability_of_disease * sample_size

/-- State the theorem: the expected number of individuals suffering from the disease is 150. -/
theorem expected_disease_count : expected_number = 150 :=
by
  -- Proof is required but skipped using sorry.
  sorry

end expected_disease_count_l373_373192


namespace sum_repeating_decimals_l373_373382

-- Definitions
def x : ℝ := 4 / 9
def y : ℝ := 2 / 3

-- Theorem statement
theorem sum_repeating_decimals : x + y = 10 / 9 := by
  sorry

end sum_repeating_decimals_l373_373382


namespace point_distance_units_l373_373191

theorem point_distance_units (d : ℝ) (h : |d| = 4) : d = 4 ∨ d = -4 := 
sorry

end point_distance_units_l373_373191


namespace max_area_triangle_PAB_l373_373548

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373548


namespace isosceles_base_angle_l373_373994

theorem isosceles_base_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = B ∨ A = C) (h3 : A = 80 ∨ B = 80 ∨ C = 80) : (A = 80 ∧ B = 80) ∨ (A = 80 ∧ C = 80) ∨ (B = 80 ∧ C = 50) ∨ (C = 80 ∧ B = 50) :=
sorry

end isosceles_base_angle_l373_373994


namespace omega_range_l373_373346

theorem omega_range (ω : ℝ) (hω : 0 < ω) : local_min_points (λ x, cos (ω * x)) (Icc 0 (π / 2)) = 2 ↔ 6 ≤ ω ∧ ω < 10 := sorry

end omega_range_l373_373346


namespace cos_alpha_value_l373_373033

theorem cos_alpha_value (α : ℝ) (h1 : cos (α - π / 6) = 15 / 17)
  (h2 : α ∈ Set.Ioo (π / 6) (π / 2)) :
  cos α = (15 * Real.sqrt 3 - 8) / 34 := 
sorry

end cos_alpha_value_l373_373033


namespace problem_statement_l373_373202

open Probability

variables {Ω : Type*} {P : Measure Ω} {A B C : Set Ω}

-- Conditions given in the problem
axiom condition1 : condProb P A C > condProb P B C
axiom condition2 : condProb P A Cᶜ > condProb P B Cᶜ
axiom condition3 : P C ≠ 0
axiom condition4 : P Cᶜ ≠ 0

-- Statement to prove
theorem problem_statement : P A > P B :=
by
  sorry

end problem_statement_l373_373202


namespace find_length_of_other_wall_l373_373146

theorem find_length_of_other_wall (area : ℝ) (width : ℝ) (h_area : area = 12.0) (h_width : width = 8) : 
  ∃ length : ℝ, length = 1.5 ∧ area = width * length :=
by
  use 1.5
  have h_length : area = width * 1.5, by
    calc
      area = 12.0 : by rw h_area
      ... = 8 * 1.5 : by rw [h_width, mul_comm 8 1.5]
  exact ⟨1.5, h_length⟩

end find_length_of_other_wall_l373_373146


namespace maximum_value_of_L_in_triangle_l373_373021

def L (x y : ℝ) : ℝ := -2 * x + y

def A : ℝ × ℝ := (-2, -1)
def B : ℝ × ℝ := (0, 1)
def C : ℝ × ℝ := (2, -1)

theorem maximum_value_of_L_in_triangle :
  L A.1 A.2 ≤ 3 ∧ L B.1 B.2 ≤ 3 ∧ L C.1 C.2 ≤ 3 ∧
  (L A.1 A.2 = 3 → (L A.1 A.2 ≥ L B.1 B.2) ∧ (L A.1 A.2 ≥ L C.1 C.2)) :=
begin
  sorry
end

end maximum_value_of_L_in_triangle_l373_373021


namespace magnitude_of_5a_minus_b_l373_373940

def mag (u : V) : ℝ := sqrt (u • u)

variables (a b : V) (ha : ∥a∥ = 1) (hb : ∥b∥ = 3)
variables (angle_ab : real.angle.between (a : ℝ × ℝ) (b : ℝ × ℝ) = π * 2 / 3)

theorem magnitude_of_5a_minus_b : ∥(5 : ℝ) • a - b∥ = 7 :=
by
  let dot_product := real_inner (5 • a - b) (5 • a - b)
  have h1 : ⟪a, b⟫ = -3 / 2 := by sorry
  have h2 : ⟪a, a⟫ = 1 := by sorry
  have h3 : ⟪b, b⟫ = 9 := by sorry
  show = sqrt (25 * 1 + 9 + 15) = 7 := by sorry

-- Placeholders for variables, needs further elaboration based on the real vector space
variables (V : Type*) [inner_product_space ℝ V] ℝ angle_ab V a b mag_angle_ab V ha hb mag_angle_ab vady sorry ℝ angle_ab

end magnitude_of_5a_minus_b_l373_373940


namespace dice_probability_l373_373727

/-- The probability that the magnitude of the vector representing the points
    facing up on two dice is less than 5 is 13/36. --/
theorem dice_probability :
  ∃ (count : ℕ), 
    (count = 13 ∧ 
      let total := 36 in 
      ∀ (m n : ℕ), 
      (1 ≤ m ∧ m ≤ 6) → 
      (1 ≤ n ∧ n ≤ 6) → 
      (m * m + n * n < 25) → 
      (count.toReal / total.toReal = (13 / 36))) :=
sorry

end dice_probability_l373_373727


namespace combined_cost_is_107_l373_373840

def wallet_cost : ℕ := 22
def purse_cost (wallet_price : ℕ) : ℕ := 4 * wallet_price - 3
def combined_cost (wallet_price : ℕ) (purse_price : ℕ) : ℕ := wallet_price + purse_price

theorem combined_cost_is_107 : combined_cost wallet_cost (purse_cost wallet_cost) = 107 := 
by 
  -- Proof
  sorry

end combined_cost_is_107_l373_373840


namespace cubic_roots_real_parts_neg_l373_373669

variable {a0 a1 a2 a3 : ℝ}

theorem cubic_roots_real_parts_neg (h_same_signs : (a0 > 0 ∧ a1 > 0 ∧ a2 > 0 ∧ a3 > 0) ∨ (a0 < 0 ∧ a1 < 0 ∧ a2 < 0 ∧ a3 < 0)) 
  (h_root_condition : a1 * a2 - a0 * a3 > 0) : 
    ∀ (x : ℝ), (a0 * x^3 + a1 * x^2 + a2 * x + a3 = 0 → x < 0 ∨ (∃ (z : ℂ), z.re < 0 ∧ z.im ≠ 0 ∧ z^2 = x)) :=
sorry

end cubic_roots_real_parts_neg_l373_373669


namespace find_m_l373_373933

noncomputable def a (n : ℕ) : ℝ := Real.log (n+2) / Real.log (n+1)

theorem find_m (m : ℕ) (H : ∏ i in Finset.range m, a i = 2016) : m = 2^2016 - 2 :=
by
  sorry

end find_m_l373_373933


namespace parabola_condition_max_area_triangle_l373_373484

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373484


namespace jar_filling_fraction_l373_373012

-- Define the capacities and the initial fractions.
variables (S L W : ℝ)
variables (h_eq_water : W = 1/8 * S ∧ W = 1/6 * L)

-- Define the proportion of S and L.
def relation_S_L : Prop := S = 4/3 * L

-- Prove the final fraction in the larger jar.
theorem jar_filling_fraction (h_S_L : relation_S_L S L) :
  (W + W) / L = 1/3 :=
by {
  -- The conditions given ensure that W is consistent, hence both filling 
  -- amounts are calculated as described in the solution.
  have h_total_water : (W + W) = (1/6 * L + 1/8 * S),
  { sorry }, -- We encapsulate the equality of W into relation.
  rw h_total_water,
  rw h_S_L,
  ring, -- Simplify the arithmetic.
  }

end jar_filling_fraction_l373_373012


namespace reciprocal_addition_l373_373292

noncomputable theory
open Real

-- Definition of the reciprocals
def a : ℝ := 0.03
def b : ℝ := 0.37

-- Statement of the theorem to prove the given equivalence
theorem reciprocal_addition :
  1 / ((1 / a) + (1 / b)) ≈ 0.02775 :=
by
  sorry -- Proof omitted

end reciprocal_addition_l373_373292


namespace cubic_polynomial_root_sum_cube_value_l373_373364

noncomputable def α : ℝ := (17 : ℝ)^(1 / 3)
noncomputable def β : ℝ := (67 : ℝ)^(1 / 3)
noncomputable def γ : ℝ := (137 : ℝ)^(1 / 3)

theorem cubic_polynomial_root_sum_cube_value
    (p q r : ℝ)
    (h1 : (p - α) * (p - β) * (p - γ) = 1)
    (h2 : (q - α) * (q - β) * (q - γ) = 1)
    (h3 : (r - α) * (r - β) * (r - γ) = 1) :
    p^3 + q^3 + r^3 = 218 := 
by
  sorry

end cubic_polynomial_root_sum_cube_value_l373_373364


namespace bright_mirrors_probability_l373_373126

theorem bright_mirrors_probability :
  (finset.card (finset.filter (λ (p : ℕ × ℕ), abs (p.1 - p.2) ≤ 1)
    ((finset.fin_range 6).product (finset.fin_range 6)))) / 36 = 4 / 9 :=
sorry

end bright_mirrors_probability_l373_373126


namespace balls_in_box_l373_373358

def num_blue : Nat := 6
def num_red : Nat := 4
def num_green : Nat := 3 * num_blue
def num_yellow : Nat := 2 * num_red
def num_total : Nat := num_blue + num_red + num_green + num_yellow

theorem balls_in_box : num_total = 36 := by
  sorry

end balls_in_box_l373_373358


namespace probability_of_A_l373_373220

noncomputable theory
open Classical

variables {Ω : Type} {P : Ω → Prop}

theorem probability_of_A (A B : Ω → Prop) [decidable_pred A] [decidable_pred B]
  (h1 : ∀ ω, P (A ω) ∧ P (B ω) → P (A ω ∧ B ω))
  (h2 : ∃ ω, P (A ω) > 0)
  (h3 : ∀ ω, P (A ω) = 2 * P (B ω))
  (h4 : ∀ ω, P (A ω ∨ B ω) = 3 * P (A ω ∧ B ω)) :
  ∀ ω, P (A ω) = 3 / 4 :=
sorry

end probability_of_A_l373_373220


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373495

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373495


namespace least_n_exists_2_same_color_triangles_l373_373903

def Point := ℕ

-- Points P₁, P₂, ..., Pₙ
variable {n : ℕ}
variable P : Fin n → Point

-- Conditions: No three points are collinear, each point is colored red or blue
axiom no_three_collinear (i j k : Fin n) : i ≠ j → i ≠ k → j ≠ k → ¬ collinear P i j k
axiom color (i : Fin n) : Bool -- true for red, false for blue

-- Statement to be proved: the least n such that there exist two triangles with all vertices of the same color
theorem least_n_exists_2_same_color_triangles (h : n = 8) : 
  ∃ (T₁ T₂ : Finset (Fin n)), 
    (∀ i, i ∈ T₁ → color i = color (some T₁.elem)).to_set ∧ 
    (T₁.card = 3 ∧ T₁ ⊆ Finset.univ ∧ 
    ∀ i, i ∈ T₂ → color i = color (some T₂.elem)).to_set ∧ 
    (T₂.card = 3 ∧ T₂ ⊆ Finset.univ) :=
sorry

end least_n_exists_2_same_color_triangles_l373_373903


namespace juniors_in_sports_count_l373_373234

-- Definitions for given conditions
def total_students : ℕ := 500
def percent_juniors : ℝ := 0.40
def percent_juniors_in_sports : ℝ := 0.70

-- Definition to calculate the number of juniors
def number_juniors : ℕ := (percent_juniors * total_students : ℝ).toNat

-- Definition to calculate the number of juniors involved in sports
def number_juniors_in_sports : ℕ := (percent_juniors_in_sports * number_juniors : ℝ).toNat

-- Statement to prove the calculated number of juniors involved in sports
theorem juniors_in_sports_count : number_juniors_in_sports = 140 :=
sorry

end juniors_in_sports_count_l373_373234


namespace problem_C_l373_373758

theorem problem_C (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a * b :=
by sorry

end problem_C_l373_373758


namespace sum_of_powers_eq_123_l373_373660

section

variables {a b : Real}

-- Conditions provided in the problem
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7

-- Define the theorem to be proved
theorem sum_of_powers_eq_123 : a^10 + b^10 = 123 :=
sorry

end

end sum_of_powers_eq_123_l373_373660


namespace angle_A_area_triangle_l373_373125

noncomputable theory

variables {A B C a b c : Real}
variables (h_eq : b^2 + c^2 - a^2 = 2 * b * c * Real.sin (B + C))
variables (h_angle_sum : A + B + C = Real.pi)
variables (h_a : a = 2)
variables (h_B : B = Real.pi / 3)

theorem angle_A :
  A = Real.pi / 4 :=
by
  sorry

theorem area_triangle :
  let b := a / Real.sin A * Real.sin B
  let sin_C := Real.sin (A + B)
  (1 / 2 * a * b * sin_C) = (3 + Real.sqrt 3) / 2 :=
by
  sorry

end angle_A_area_triangle_l373_373125


namespace angle_between_a_b_l373_373566

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1 ^ 2 + a.2 ^ 2) * Real.sqrt (b.1 ^ 2 + b.2 ^ 2)))

theorem angle_between_a_b : 
  let a := (2, 1) in
  let b := (1, 3) in
  angle_between a b = π / 4 :=
by
  sorry

end angle_between_a_b_l373_373566


namespace sqrt_subtraction_l373_373259

theorem sqrt_subtraction:
  ∀ a b c d : ℝ, a = 49 + 121 → b = 36 - 9 → sqrt a - sqrt b = sqrt 170 - 3 * sqrt 3 :=
by
  intros a b c d ha hb
  rw [ha, hb]
  sorry

end sqrt_subtraction_l373_373259


namespace area_of_isosceles_right_triangle_l373_373244

-- Definitions of the given conditions
variables (A B C O : Type) [isIsoscelesRightTriangle : is_isosceles_right_triangle A B C]
variable (inscribed_circle : circle O (sqrt 2)) -- since area = 2π, radius r = sqrt(2)
variable (radius_eq_sqrt_2 : inscribed_circle.radius = sqrt 2)

-- The goal
theorem area_of_isosceles_right_triangle
  (h1 : is_isosceles_right_triangle A B C)
  (h2 : inscribed_circle.radius = sqrt 2)
  : triangle.area A B C = 4 := by
  sorry

end area_of_isosceles_right_triangle_l373_373244


namespace eval_polynomial_at_point_l373_373966

def is_fourth_degree_trinomial (m : ℤ) (p : ℤ × ℤ → ℚ) : Prop :=
  ∀ x y : ℤ, x + y = 4

noncomputable def polynomial (m : ℤ) : (ℚ × ℚ) → ℚ :=
  λ (xy : ℚ × ℚ), (m - 3) * xy.1^(|m|-2) * xy.2^(3) + xy.1^2 * xy.2 - 2 * xy.1 * xy.2^2

theorem eval_polynomial_at_point (m : ℤ) (x y : ℚ)
  (hp : is_fourth_degree_trinomial m (polynomial m)) (hm : m = -3) :
  polynomial m (x, y) = 15 / 4 :=
by
  -- Proof omitted. sorry as placeholder.
  sorry

end eval_polynomial_at_point_l373_373966


namespace unique_factors_of_a_l373_373335

theorem unique_factors_of_a (a : ℕ) (h1 : 1 < a) 
  (h2 : (∏ d in finset.filter (∣ a) (finset.range (a + 1)), d) = a ^ 5) :
  finset.card (finset.filter (∣ a) (finset.range (a + 1))) = 10 :=
sorry

end unique_factors_of_a_l373_373335


namespace variance_linear_transform_l373_373319

noncomputable def binomial_var (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

noncomputable def linear_transform_var (a b : ℝ) (var_xi : ℝ) : ℝ := a^2 * var_xi

theorem variance_linear_transform (ξ : ℝ) (hξ : ξ ∈ binomial_var 100 0.2):
  linear_transform_var 4 3 hξ = 256 := by
  sorry

end variance_linear_transform_l373_373319


namespace find_v_squared_l373_373307

def radius : ℝ := 4
def height : ℝ := 10
def side_length : ℝ := 8

def volume_cylinder : ℝ := Real.pi * radius^2 * height
def volume_cube : ℝ := side_length^3
def volume_displaced : ℝ := volume_cube

theorem find_v_squared : volume_displaced^2 = 262144 := by
  -- proof to be written here
  sorry

end find_v_squared_l373_373307


namespace probability_of_both_types_probability_distribution_and_expectation_of_X_l373_373002

-- Definitions
def total_zongzi : ℕ := 8
def red_bean_paste_zongzi : ℕ := 2
def date_zongzi : ℕ := 6
def selected_zongzi : ℕ := 3

-- Part 1: The probability of selecting both red bean paste and date zongzi
theorem probability_of_both_types :
  let total_combinations := Nat.choose total_zongzi selected_zongzi
  let one_red_two_date := Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2
  let two_red_one_date := Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1
  (one_red_two_date + two_red_one_date) / total_combinations = 9 / 14 :=
by sorry

-- Part 2: The probability distribution and expectation of X
theorem probability_distribution_and_expectation_of_X :
  let P_X_0 := (Nat.choose red_bean_paste_zongzi 0 * Nat.choose date_zongzi 3) / Nat.choose total_zongzi selected_zongzi
  let P_X_1 := (Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2) / Nat.choose total_zongzi selected_zongzi
  let P_X_2 := (Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1) / Nat.choose total_zongzi selected_zongzi
  P_X_0 = 5 / 14 ∧ P_X_1 = 15 / 28 ∧ P_X_2 = 3 / 28 ∧
  (0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 = 3 / 4) :=
by sorry

end probability_of_both_types_probability_distribution_and_expectation_of_X_l373_373002


namespace total_books_l373_373190

theorem total_books (books_taken : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : 
  (books_taken = 10) ∧ (books_per_shelf = 4) ∧ (shelves_needed = 9) → 
  ((shelves_needed * books_per_shelf) + books_taken = 46) :=
by
  intros h,
  cases h with ht h1,
  cases h1 with hp hs,
  rw [ht, hp, hs],
  have h_remain : shelves_needed * books_per_shelf = 36 := by norm_num,
  rw h_remain,
  have h_total : 36 + books_taken = 46 := by norm_num,
  exact h_total

end total_books_l373_373190


namespace document_completion_time_l373_373728

-- Define the typing rates for different typists
def fast_typist_rate := 1 / 4
def slow_typist_rate := 1 / 9
def additional_typist_rate := 1 / 4

-- Define the number of typists
def num_fast_typists := 2
def num_slow_typists := 3
def num_additional_typists := 2

-- Define the distraction time loss per typist every 30 minutes
def distraction_loss := 1 / 6

-- Define the combined rate without distractions
def combined_rate : ℚ :=
  (num_fast_typists * fast_typist_rate) +
  (num_slow_typists * slow_typist_rate) +
  (num_additional_typists * additional_typist_rate)

-- Define the distraction rate loss per hour (two distractions per hour)
def distraction_rate_loss_per_hour := 2 * distraction_loss

-- Define the effective combined rate considering distractions
def effective_combined_rate : ℚ := combined_rate - distraction_rate_loss_per_hour

-- Prove that the document is completed in 1 hour with the effective rate
theorem document_completion_time :
  effective_combined_rate = 1 :=
sorry

end document_completion_time_l373_373728


namespace probability_shift_l373_373653

open ProbabilityTheory

variable {Ω : Type*} [MeasureSpace Ω]

def normal_dist_standard : ProbabilityDistribution ℝ :=
  normalDist 0 1

variables (X : Ω → ℝ) [RandomVariable X normal_dist_standard]

theorem probability_shift (h_p : ∀ p : ℝ, P(X > 1) = p):
  P(X > -1) = 1 - p :=
sorry

end probability_shift_l373_373653


namespace find_y_in_similar_triangles_l373_373322

-- Define the variables and conditions of the problem
def is_similar (a1 b1 a2 b2 : ℚ) : Prop :=
  a1 / b1 = a2 / b2

-- Problem statement
theorem find_y_in_similar_triangles
  (a1 b1 a2 b2 : ℚ)
  (h1 : a1 = 15)
  (h2 : b1 = 12)
  (h3 : b2 = 10)
  (similarity_condition : is_similar a1 b1 a2 b2) :
  a2 = 25 / 2 :=
by
  rw [h1, h2, h3, is_similar] at similarity_condition
  sorry

end find_y_in_similar_triangles_l373_373322


namespace square_of_99_is_9801_l373_373266

theorem square_of_99_is_9801 : 99 ^ 2 = 9801 := 
by
  sorry

end square_of_99_is_9801_l373_373266


namespace average_speed_30_l373_373298

theorem average_speed_30 (v : ℝ) (h₁ : 0 < v) (h₂ : 210 / v - 1 = 210 / (v + 5)) : v = 30 :=
sorry

end average_speed_30_l373_373298


namespace sqrt_subtraction_proof_l373_373258

def sqrt_subtraction_example : Real := 
  let a := 49 + 121
  let b := 36 - 9
  Real.sqrt a - Real.sqrt b

theorem sqrt_subtraction_proof : sqrt_subtraction_example = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end sqrt_subtraction_proof_l373_373258


namespace michael_painting_price_l373_373186

theorem michael_painting_price :
  ∃ x : ℝ, (5 * 100 + 8 * x = 1140) ∧ x = 80 :=
begin
  sorry
end

end michael_painting_price_l373_373186


namespace b_gives_c_start_l373_373762

variable (Va Vb Vc : ℝ)

-- Conditions given in the problem
def condition1 : Prop := Va / Vb = 1000 / 930
def condition2 : Prop := Va / Vc = 1000 / 800
def race_distance : ℝ := 1000

-- Proposition to prove
theorem b_gives_c_start (h1 : condition1 Va Vb) (h2 : condition2 Va Vc) :
  ∃ x : ℝ, (1000 - x) / 1000 = (930 / 800) :=
sorry

end b_gives_c_start_l373_373762


namespace employed_females_percentage_l373_373279

def population_employed (total_population : ℕ) : ℕ := 120 * total_population / 100
def employed_males (total_population : ℕ) : ℕ := 80 * total_population / 100
def employed_females (total_population : ℕ) : ℕ := population_employed total_population - employed_males total_population
def percent_employed_females (total_population : ℕ) : ℕ := 100 * employed_females total_population / population_employed total_population

theorem employed_females_percentage {total_population : ℕ} (h : total_population > 0) : percent_employed_females total_population = 33.33 :=
by
  sorry

end employed_females_percentage_l373_373279


namespace decreasing_function_range_l373_373460

noncomputable def f (a : ℝ) (x : ℝ) := if x ≤ 1 then (3 * a - 2) * x + 1 else a ^ x

theorem decreasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f(a, y) < f(a, x)) ↔ (1 / 2 ≤ a ∧ a < 2 / 3) :=
sorry

end decreasing_function_range_l373_373460


namespace find_a_l373_373048

open Set

theorem find_a (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = {1, 2})
  (hB : B = {-a, a^2 + 3})
  (hUnion : A ∪ B = {1, 2, 4}) :
  a = -1 :=
sorry

end find_a_l373_373048


namespace parabola_focus_distance_max_area_triangle_l373_373556

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373556


namespace variance_stability_l373_373272

theorem variance_stability (S2_A S2_B : ℝ) (hA : S2_A = 1.1) (hB : S2_B = 2.5) : ¬(S2_B < S2_A) :=
by {
  sorry
}

end variance_stability_l373_373272


namespace sqrt_subtraction_l373_373263

theorem sqrt_subtraction : 
  sqrt (49 + 121) - sqrt (36 - 9) = sqrt 170 - sqrt 27 :=
by
  sorry

end sqrt_subtraction_l373_373263


namespace max_k_a_k_neg_l373_373906

theorem max_k_a_k_neg :
  ∀ (a : ℕ → ℕ) (x : ℝ), 
    (∀ k : ℕ, k ∈ {0..100} → (1 + 2023 * x)^100 + (2023 - x)^100 = 
    ∑ i in range 101, a i * x^i) → 
    ∃ k : ℕ, k = 49 ∧ a k < 0 := 
  by
  sorry

end max_k_a_k_neg_l373_373906


namespace volume_remaining_after_modifications_l373_373786

def diameter_bowling_ball := 24 -- diameter in cm
def radius_bowling_ball := diameter_bowling_ball / 2

def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
def volume_cylinder (r h : ℝ) : ℝ := Real.pi * r^2 * h

def radius_hole1 := (2.5 : ℝ) / 2
def radius_hole2 := (2.5 : ℝ) / 2
def radius_hole3 := (4 : ℝ) / 2

def depth_hole1 := 5 -- depth in cm
def depth_hole2 := 5 -- depth in cm
def depth_hole3 := 5 -- depth in cm

def volume_bowling_ball := volume_sphere radius_bowling_ball
def volume_hole1 := volume_cylinder radius_hole1 depth_hole1
def volume_hole2 := volume_cylinder radius_hole2 depth_hole2
def volume_hole3 := volume_cylinder radius_hole3 depth_hole3

def total_hole_volume := volume_hole1 + volume_hole2 + volume_hole3

def remaining_volume := volume_bowling_ball - total_hole_volume

theorem volume_remaining_after_modifications :
  remaining_volume = 2268.375 * Real.pi :=
by
  simp [volume_sphere, volume_cylinder, radius_bowling_ball, radius_hole1, radius_hole2, radius_hole3, depth_hole1, depth_hole2, depth_hole3, volume_bowling_ball, volume_hole1, volume_hole2, volume_hole3, total_hole_volume, remaining_volume]
  sorry

end volume_remaining_after_modifications_l373_373786


namespace find_p_max_area_triangle_l373_373528

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373528


namespace blue_hat_cost_l373_373248

theorem blue_hat_cost :
  ∀ (total_hats green_hats total_price green_hat_price blue_hat_price) 
  (B : ℕ),
  total_hats = 85 →
  green_hats = 30 →
  total_price = 540 →
  green_hat_price = 7 →
  blue_hat_price = B →
  (30 * 7) + (55 * B) = 540 →
  B = 6 := sorry

end blue_hat_cost_l373_373248


namespace base4_to_base2_equality_l373_373271

-- Define the decimal conversion from base 4
def convert_base4_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 1010 => 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 0 * 4^0
  | _ => 0

-- Define the decimal conversion from base 2
def convert_base2_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 1000100 => 1 * 2^6 + 1 * 2^2
  | _ => 0

-- The theorem to prove that the numbers are equal in decimal when converted
theorem base4_to_base2_equality : convert_base4_to_decimal 1010 = convert_base2_to_decimal 1000100 :=
by
  -- First we simplify the left-hand side which is base 4 to decimal
  unfold convert_base4_to_decimal
  -- Convert 1010_{(4)} to its decimal equivalent
  have h1 : 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 0 * 4^0 = 68 := by norm_num
  -- Simplify the right-hand side which is base 2 to decimal
  unfold convert_base2_to_decimal
  -- Convert 1000100_{(2)} to its decimal equivalent
  have h2 : 1 * 2^6 + 1 * 2^2 = 68 := by norm_num
  -- Use the equalities h1 and h2 to conclude the proof
  rw [h1, h2]
  rfl

end base4_to_base2_equality_l373_373271


namespace total_rainfall_in_2004_l373_373593

noncomputable def average_monthly_rainfall_2003 : ℝ := 35.0
noncomputable def average_monthly_rainfall_2004 : ℝ := average_monthly_rainfall_2003 + 4.0
noncomputable def total_rainfall_2004 : ℝ := 
  let regular_months := 11 * average_monthly_rainfall_2004
  let daily_rainfall_feb := average_monthly_rainfall_2004 / 30
  let feb_rain := daily_rainfall_feb * 29 
  regular_months + feb_rain

theorem total_rainfall_in_2004 : total_rainfall_2004 = 466.7 := by
  sorry

end total_rainfall_in_2004_l373_373593


namespace sample_space_correct_events_A_and_B_not_independent_most_likely_sum_is_5_l373_373598

/-- Conditions: A bag contains 4 balls labeled 1, 2, 3, 4. Two balls are drawn without replacement. 
Event A: drawing the ball labeled 2 on the first draw. Event B: the sum of the numbers on the two balls drawn is 5. -/
def sampleSpace : Set (Nat × Nat) :=
  {(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)}

def eventA : Set (Nat × Nat) :=
  {(2, 1), (2, 3), (2, 4)}

def eventB : Set (Nat × Nat) :=
  {(1, 4), (2, 3), (3, 2), (4, 1)}

theorem sample_space_correct :
  sampleSpace = {(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)} :=
sorry

theorem events_A_and_B_not_independent :
  ¬ (ProbSpace.independence eventA eventB) :=
sorry

theorem most_likely_sum_is_5 :
  (∃ (subset : Set (Nat × Nat)), subset ⊆ sampleSpace ∧
  (∀ (sums : Nat), (sums ∈ (5 :: List.nil) → (ProbSpace.probability_of subset) > 
  (ProbSpace.probability_of {p | p.fst + p.snd = sums}) → sums = 5))) := sorry

end sample_space_correct_events_A_and_B_not_independent_most_likely_sum_is_5_l373_373598


namespace original_prop_and_contrapositive_l373_373964

theorem original_prop_and_contrapositive (m : ℝ) (h : m > 0) : 
  (∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 + x - m = 0 ∨ ∃ x y : ℝ, x^2 + x - m = 0 ∧ y^2 + y - m = 0) :=
by
  sorry

end original_prop_and_contrapositive_l373_373964


namespace upstream_distance_is_88_l373_373785

noncomputable def distance_upstream 
(distance_downstream : ℕ) -- 96 km
(time_downstream : ℕ) -- 3 hours
(time_upstream : ℕ) -- 11 hours
(speed_still_water : ℕ) -- 20 km/h
: ℕ :=
let downstream_speed := distance_downstream / time_downstream in
let speed_current := downstream_speed - speed_still_water in
let upstream_speed := speed_still_water - speed_current in
upstream_speed * time_upstream

theorem upstream_distance_is_88 :
  distance_upstream 96 3 11 20 = 88 :=
by {
  -- We could go through the steps here, but for now place fillers
  sorry
}

end upstream_distance_is_88_l373_373785


namespace exists_d_with_three_real_roots_l373_373160

noncomputable def g (d : ℝ) (x : ℝ) : ℝ :=
  x^2 + 4*x + d

noncomputable def g_compose (d : ℝ) (x : ℝ) : ℝ :=
  g d (g d x)

theorem exists_d_with_three_real_roots :
  ∃ d : ℝ, d = (3 - Real.sqrt 57) / 2 ∧
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    g_compose d a = 0 ∧ g_compose d b = 0 ∧ g_compose d c = 0) :=
begin
  sorry
end

end exists_d_with_three_real_roots_l373_373160


namespace polar_coordinates_of_point_l373_373210

noncomputable def polar_coordinates (x : ℝ) (y : ℝ) : ℝ × ℝ :=
  let r := real.sqrt (x^2 + y^2)
  let theta := real.arctan (y / x)
  (r, theta)

theorem polar_coordinates_of_point (x y : ℝ) (hx : x = -real.sqrt 2) (hy : y = real.sqrt 2) :
  polar_coordinates x y = (2, 3 * real.pi / 4) :=
by
  sorry

end polar_coordinates_of_point_l373_373210


namespace base_angle_isosceles_triangle_l373_373995

theorem base_angle_isosceles_triangle (x : ℝ) :
    (∀ (Δ : Triangle) (a b c : ℝ),
        Δ.Angles = [a, b, c] ∧ a = 80 ∧ IsIsosceles Δ → 
        (b = 80 ∨ b = 50)) :=
by
    -- Assuming AngleSum and IsIsosceles are defined appropriately in the context
    sorry

end base_angle_isosceles_triangle_l373_373995


namespace minimum_top_block_number_is_119_l373_373317

noncomputable def minimum_top_block_number : ℕ :=
let bottom_layer := fin 15 → ℕ,
    second_layer := fin 9 → ℕ,
    third_layer := fin 6 → ℕ,
    top_block : ℕ
in
    ∃ b : bottom_layer, 
    ∃ s : second_layer, 
    ∃ t : third_layer, 
    ∃ top : ℕ, 
        (∀ i : fin 9, s i = b (3*i.val) + b (3*i.val + 1) + b (3*i.val + 2)) ∧
        (∀ j : fin 6, t j = s (3*j.val) + s (3*j.val + 1) + s (3*j.val + 2)) ∧
        (top = t 0 + t 1 + t 2) ∧
        (∀ k : fin 15, b k ∈ finset.range 1 16) ∧
        top = 119

theorem minimum_top_block_number_is_119 : minimum_top_block_number = 119 := 
by 
    sorry

end minimum_top_block_number_is_119_l373_373317


namespace line_bisects_l373_373685

-- Definitions corresponding to the problem conditions:
variables {A B C A1 C1 H M : Type*}

-- Given conditions
axiom altitude_AA1 : ∃ (H : Point), is_altitude A B C A1
axiom altitude_CC1 : ∃ (H : Point), is_altitude C B A C1
axiom circumcircle_intersection : ∃ (M : Point), M ≠ B ∧ on_circumcircle A B C M ∧ on_circumcircle A1 B C1 M

-- Conclusion we need to prove
theorem line_bisects : bisects_side (line_through M H) A C :=
sorry

end line_bisects_l373_373685


namespace arithmetic_sequence_general_term_max_sum_S_n_l373_373931

theorem arithmetic_sequence_general_term:
  ∃ a₁ d : ℤ, ∀ n : ℤ,
  (a₁ + (a₁ + 2 * d) = 16 ∧ 4 * a₁ + 6 * d = 28) →
  (set_of (λ n, a₁ + (n - 1) * d) = set_of (λ n, 12 - 2 * n)) :=
by
  sorry

theorem max_sum_S_n:
  ∃ (n : ℕ) (S : ℤ), ∀ m : ℕ,
  (a₁ + (a₁ + 2 * d) = 16 ∧ 4 * a₁ + 6 * d = 28) →
  ((∀ k, k ≤ 6 → k * (a₁ + (12 - 2 * k)) / 2 ≤ S) ∧ S = 30 ∧ n = 6) :=
by
  sorry

end arithmetic_sequence_general_term_max_sum_S_n_l373_373931


namespace max_rounds_proof_l373_373715

-- Definition of the child and their genders (boy as 1 and girl as -1)
inductive Gender
| boy
| girl

-- Define the initial setup and the game transformation rules
def initialChildren : List Gender :=
  [Gender.girl, Gender.girl, Gender.girl, Gender.girl, Gender.girl, Gender.girl, Gender.boy, Gender.boy, Gender.boy, Gender.boy, Gender.boy]

-- Function to represent the next state's gender based on adjacent children
def nextGender (g1 g2 : Gender) : Gender :=
  match g1, g2 with
  | Gender.boy, Gender.boy => Gender.boy
  | Gender.girl, Gender.girl => Gender.boy
  | _, _ => Gender.girl

-- Function to get the configuration after a round based on current round
def nextConfiguration (config : List Gender) : List Gender := sorry

-- The maximum number of rounds until all children are boys
def maxRounds : ℕ := 4

-- The theorem to prove the maximum number of rounds
theorem max_rounds_proof (initial : List Gender) :
  initial = initialChildren →
  ∃ rounds, (nextConfiguration^[rounds] initial).all (λ c => c = Gender.boy) ∧ rounds = maxRounds :=
sorry

end max_rounds_proof_l373_373715


namespace find_k_l373_373880

theorem find_k (k : ℝ) : -x^2 - (k + 10) * x - 8 = -(x - 2) * (x - 4) → k = -16 := by
  intro h
  sorry

end find_k_l373_373880


namespace smallest_r_minus_p_l373_373717

theorem smallest_r_minus_p (p q r : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0)
  (h4 : p * q * r = Nat.factorial 9) (h5 : p < q) (h6 : q < r) :
  r - p = 396 :=
sorry

end smallest_r_minus_p_l373_373717


namespace quadratic_roots_property_l373_373166

theorem quadratic_roots_property (m n : ℝ) 
  (h1 : ∀ x, x^2 - 2 * x - 2025 = (x - m) * (x - n))
  (h2 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 := 
by 
  sorry

end quadratic_roots_property_l373_373166


namespace mainTheorem_l373_373919

noncomputable def rightTriangularPrism :
Prop :=
  ∃ (A B C A₁ B₁ C₁ P Q T: Point) (BC_radius: ℝ),
    isRightTriangularPrism A B C A₁ B₁ C₁ ∧
    sphereWithDiameter BC A₁ B₁ A C B C₁ P Q T BC_radius ∧
    segmentIntersect (\A₁ P \ B₁) (\C₁ Q \ T) ∧
    C₁ Q = 2 ∧
    Angle T P A = 90 ∧
    AP / CP = 2 / 1 ∧
    AC = 3 ∧
    volumePrism A B C A₁ B₁ C₁ = 15

theorem mainTheorem :
rightTriangularPrism :=
sorry

end mainTheorem_l373_373919


namespace number_of_possible_values_l373_373703

-- Define the decimal number s and its representation
def s (e f g h : ℕ) : ℚ := e / 10 + f / 100 + g / 1000 + h / 10000

-- Define the condition that the closest fraction is 2/9
def closest_to_2_9 (s : ℚ) : Prop :=
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 1 / 6)) ∧
  abs (s - 2 / 9) < min (abs (s - 1 / 5)) (abs (s - 2 / 11))

-- The main theorem stating the number of possible values for s
theorem number_of_possible_values :
  (∃ e f g h : ℕ, 0 ≤ e ∧ e ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9 ∧ 0 ≤ g ∧ g ≤ 9 ∧ 0 ≤ h ∧ h ≤ 9 ∧
    closest_to_2_9 (s e f g h)) → (∃ n : ℕ, n = 169) :=
by
  sorry

end number_of_possible_values_l373_373703


namespace apple_juice_fraction_correct_l373_373729

def problem_statement : Prop :=
  let pitcher1_capacity := 800
  let pitcher2_capacity := 500
  let pitcher1_apple_fraction := 1 / 4
  let pitcher2_apple_fraction := 1 / 5
  let pitcher1_apple_volume := pitcher1_capacity * pitcher1_apple_fraction
  let pitcher2_apple_volume := pitcher2_capacity * pitcher2_apple_fraction
  let total_apple_volume := pitcher1_apple_volume + pitcher2_apple_volume
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_apple_volume / total_volume = 3 / 13

theorem apple_juice_fraction_correct : problem_statement := 
  sorry

end apple_juice_fraction_correct_l373_373729


namespace sophia_book_pages_l373_373683

theorem sophia_book_pages:
  ∃ (P : ℕ), (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30 ∧ P = 90 :=
by
  sorry

end sophia_book_pages_l373_373683


namespace find_t_l373_373055

noncomputable def z1 : ℂ := 3 + 4 * complex.i
noncomputable def z2 (t : ℝ) : ℂ := t + complex.i
noncomputable def z2_conjugate (t : ℝ) : ℂ := (t : ℂ) - complex.i

theorem find_t (t : ℝ) : (z1 * z2_conjugate t).im = 0 → t = 3 / 4 :=
by
  sorry

end find_t_l373_373055


namespace parabola_circle_distance_l373_373470

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373470


namespace travel_times_either_24_or_72_l373_373005

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end travel_times_either_24_or_72_l373_373005


namespace minimum_shots_to_hit_ship_l373_373240

-- Definitions based on conditions
def grid_size : ℕ := 7
def ship_squares : ℕ := 4

-- This is to formalize the main problem in Lean
theorem minimum_shots_to_hit_ship : 
  ∃ (shots : ℕ), shots = 20 ∧ 
  (∀ (attack_positions : finset (fin grid_size × fin grid_size)), 
     attack_positions.card < shots → 
     ∃ (ship_position : set (fin grid_size × fin grid_size)), 
       ship_position.card = ship_squares ∧
       ∀ (deck : fin grid_size × fin grid_size), 
         deck ∈ ship_position →
         ∀ (attack : fin grid_size × fin grid_size), 
           attack ∉ attack_positions) :=
begin
  sorry -- Proof goes here
end

end minimum_shots_to_hit_ship_l373_373240


namespace cos_of_angle_complement_l373_373117

theorem cos_of_angle_complement (α : ℝ) (h : 90 - α = 30) : Real.cos α = 1 / 2 :=
by
  sorry

end cos_of_angle_complement_l373_373117


namespace pirate_coins_l373_373664

theorem pirate_coins (x : ℕ) : 
  (x * (x + 1)) / 2 = 3 * x → 4 * x = 20 := by
  sorry

end pirate_coins_l373_373664


namespace find_p_max_area_of_triangle_l373_373506

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373506


namespace price_of_shoes_on_tuesday_is_correct_l373_373656

theorem price_of_shoes_on_tuesday_is_correct :
  let price_thursday : ℝ := 30
  let price_friday : ℝ := price_thursday * 1.2
  let price_monday : ℝ := price_friday - price_friday * 0.15
  let price_tuesday : ℝ := price_monday - price_monday * 0.1
  price_tuesday = 27.54 := 
by
  sorry

end price_of_shoes_on_tuesday_is_correct_l373_373656


namespace find_triplets_l373_373284

noncomputable def conditions (p q n : ℕ) : Prop :=
  q^(n+2) % p^n = 3^(n+2) % p^n ∧
  p^(n+2) % q^n = 3^(n+2) % q^n ∧
  p > 2 ∧
  q > 2 ∧
  Nat.prime p ∧
  Nat.prime q ∧
  n > 1

theorem find_triplets (p q n : ℕ) (h : conditions p q n) : 
  p = 3 ∧ q = 3 ∧ n ≥ 2 :=
begin
  sorry
end

end find_triplets_l373_373284


namespace incenters_concyclic_of_quad_l373_373600

open geometry

-- Definitions and conditions
variables {A B C D E F G H I J K L : Point}
variable [circle (A, B)]
variable [circle (C, D)]
variable cyclic_quadrilateral (ABCD : quadrilateral A B C D)
variable tangent_to (AE : line) (E) : line such that AE passes through points A and B and is tangent to CD at E
variable tangent_to (CF : line) (F) : line such that CF passes through points C and D and is tangent to AB at F
variable intersection_point (AE, DF : line) : Point such that G is the intersection of AE and DF
variable intersection_point (BE, CF : line) : Point such that H is the intersection of BE and CF
variable incenter_triangle (AGF : triangle) : Point such that I is the incenter of triangle AGF
variable incenter_triangle (BHF : triangle) : Point such that J is the incenter of triangle BHF
variable incenter_triangle (CHE : triangle) : Point such that K is the incenter of triangle CHE
variable incenter_triangle (DGE : triangle) : Point such that L is the incenter of triangle DGE

-- Theorem statement
theorem incenters_concyclic_of_quad:
  ∀ (ABCD : quadrilateral A B C D) (E F G H I J K L : Point),
    cyclic_quadrilateral ABCD →
    tangent_to AE E → tangent_to CF F → 
    intersection_point (AE, DF) = G →
    intersection_point (BE, CF) = H →
    incenter_triangle (AGF) = I →
    incenter_triangle (BHF) = J →
    incenter_triangle (CHE) = K →
    incenter_triangle (DGE) = L →
    concyclic {I, J, K, L} :=
by
sorry

end incenters_concyclic_of_quad_l373_373600


namespace p_necessary_not_sufficient_for_q_l373_373930

variables {α : Type*} [inner_product_space ℝ α]
variables {a b : α}

def p (a b : α) : Prop := inner a b > 0
def q (a b : α) : Prop := real.angle a b < real.pi / 2

theorem p_necessary_not_sufficient_for_q (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(p a b ↔ q a b) ∧ (p a b → q a b) :=
by
  split
  · intro h
    have : ¬(q a b → p a b),
      { intro hq
        obtain ⟨x, hx⟩ := real.angle_eq_iff.1 (lt_add_of_lt_of_nonneg hq (zero_le_one)),
        exact hx },
    exact (iff_iff_implies_and_implies.1 h).2 this
  · intro hp
    exact (real.angle_cos_eq_inner_div_norm_mul_norm _ _).mpr (by simp [hp, lt_add_of_pos_right _ real.pi_div_two_pos, real_inner_self_nonneg])
  sorry

end p_necessary_not_sufficient_for_q_l373_373930


namespace girl_scouts_signed_permission_slips_percentage_is_correct_l373_373795

-- Definitions from problem statement
def total_scouts : ℕ := 100
def boy_scouts : ℕ := 30
def girl_scouts : ℕ := 35
def cub_scouts : ℕ := 20
def brownies : ℕ := 15
def total_signed_permission_slips : ℕ := 65
def boy_scouts_signed_permission_slips : ℕ := 17
def cub_scouts_signed_permission_slips : ℕ := 9
def brownies_signed_permission_slips : ℕ := 12

-- Condition for Girl Scouts with signed permission slips
def girl_scouts_signed_permission_slips := total_signed_permission_slips - (boy_scouts_signed_permission_slips + cub_scouts_signed_permission_slips + brownies_signed_permission_slips)

-- Calculate percentage of Girl Scouts with signed permission slips
def girl_scouts_signed_permission_percentage := (girl_scouts_signed_permission_slips * 100) / girl_scouts

theorem girl_scouts_signed_permission_slips_percentage_is_correct :
  girl_scouts_signed_permission_percentage = 77 :=
by {
  have h := (65 - (17 + 9 + 12)) * 100 / 35,
  norm_num at h,
  exact h,
}

end girl_scouts_signed_permission_slips_percentage_is_correct_l373_373795


namespace volumes_in_ascending_order_volume_V₄_between_V₂_and_V₃_volume_V₅_between_V₁_and_V₃_l373_373242

-- Conditions: Definitions of radii and heights for the three cylinders
def R₁ : ℝ := 10
def h₁ : ℝ := 10
def R₂ : ℝ := 5
def h₂ : ℝ := 10
def R₃ : ℝ := 5
def h₃ : ℝ := 20

-- Definitions of volumes for the three cylinders
def V₁ : ℝ := Real.pi * R₁^2 * h₁
def V₂ : ℝ := Real.pi * R₂^2 * h₂
def V₃ : ℝ := Real.pi * R₃^2 * h₃

-- a) Statement: Arrange the volumes in ascending order.
theorem volumes_in_ascending_order :
  V₂ < V₃ ∧ V₃ < V₁ := by
  sorry

-- b) Statement: Finding dimensions for V₄ such that V₂ < V₄ < V₃.
def R₄ : ℝ := 5
def h₄ : ℝ := 15
def V₄ : ℝ := Real.pi * R₄^2 * h₄

theorem volume_V₄_between_V₂_and_V₃ :
  V₂ < V₄ ∧ V₄ < V₃ := by
  sorry

-- c) Statement: Finding dimensions for V₅ such that V₁ < V₅ < V₃.
def R₅ : ℝ := 8
def h₅ : ℝ := 10
def V₅ : ℝ := Real.pi * R₅^2 * h₅

theorem volume_V₅_between_V₁_and_V₃ :
  V₃ < V₅ ∧ V₅ < V₁ := by
  sorry

end volumes_in_ascending_order_volume_V₄_between_V₂_and_V₃_volume_V₅_between_V₁_and_V₃_l373_373242


namespace sum_of_multiples_l373_373207

def smallest_two_digit_multiple_of_seven : ℕ := 14
def smallest_three_digit_multiple_of_five : ℕ := 100

theorem sum_of_multiples : smallest_two_digit_multiple_of_seven + smallest_three_digit_multiple_of_five = 114 :=
by
  let c := smallest_two_digit_multiple_of_seven
  let d := smallest_three_digit_multiple_of_five
  have hc : c = 14 := rfl
  have hd : d = 100 := rfl
  show c + d = 114 from
    calc
      c + d = 14 + 100 := by rw [hc, hd]
      ...     = 114     := by norm_num

end sum_of_multiples_l373_373207


namespace problem_l373_373947

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1 else S n - S (n - 1)

def sum_abs_a_10 : ℤ :=
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|)

theorem problem : sum_abs_a_10 = 67 := by
  sorry

end problem_l373_373947


namespace jane_weekly_pages_l373_373624

-- Define the daily reading amounts
def monday_wednesday_morning_pages : ℕ := 5
def monday_wednesday_evening_pages : ℕ := 10
def tuesday_thursday_morning_pages : ℕ := 7
def tuesday_thursday_evening_pages : ℕ := 8
def friday_morning_pages : ℕ := 10
def friday_evening_pages : ℕ := 15
def weekend_morning_pages : ℕ := 12
def weekend_evening_pages : ℕ := 20

-- Define the number of days
def monday_wednesday_days : ℕ := 2
def tuesday_thursday_days : ℕ := 2
def friday_days : ℕ := 1
def weekend_days : ℕ := 2

-- Function to calculate weekly pages
def weekly_pages :=
  (monday_wednesday_days * (monday_wednesday_morning_pages + monday_wednesday_evening_pages)) +
  (tuesday_thursday_days * (tuesday_thursday_morning_pages + tuesday_thursday_evening_pages)) +
  (friday_days * (friday_morning_pages + friday_evening_pages)) +
  (weekend_days * (weekend_morning_pages + weekend_evening_pages))

-- Proof statement
theorem jane_weekly_pages : weekly_pages = 149 := by
  unfold weekly_pages
  norm_num
  sorry

end jane_weekly_pages_l373_373624


namespace remainder_2468135792_div_101_l373_373735

theorem remainder_2468135792_div_101 : (2468135792 % 101) = 52 := 
by 
  -- Conditions provided in the problem
  have decompose_num : 2468135792 = 24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92, 
  from sorry,
  
  -- Assert large powers of 10 modulo properties
  have ten_to_pow2 : (10^2 - 1) % 101 = 0, from sorry,
  have ten_to_pow4 : (10^4 - 1) % 101 = 0, from sorry,
  have ten_to_pow6 : (10^6 - 1) % 101 = 0, from sorry,
  have ten_to_pow8 : (10^8 - 1) % 101 = 0, from sorry,
  
  -- Summing coefficients
  have coefficients_sum : 24 + 68 + 13 + 57 + 92 = 254, from
  by linarith,
  
  -- Calculating modulus
  calc 
    2468135792 % 101
        = (24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92) % 101 : by rw decompose_num
    ... = (24 + 68 + 13 + 57 + 92) % 101 : by sorry
    ... = 254 % 101 : by rw coefficients_sum
    ... = 52 : by norm_num,

  sorry

end remainder_2468135792_div_101_l373_373735


namespace length_of_segment_AB_l373_373655

-- Definitions corresponding to the given conditions
variables {C1 C2 : Circle} -- Define two circles C1 and C2
variables {ℓ1 ℓ2 : Line} -- Define two lines ℓ1 and ℓ2 which are external tangents to C1 and C2

-- Assume the distance between the points of tangency on ℓ1 and ℓ2 is a
variable {a : ℝ} -- (distance between tangency points on each of the lines)

-- Internal tangent intersects ℓ1 and ℓ2 at points A and B respectively
variables {A B : Point}

-- Theorem statement
theorem length_of_segment_AB 
  (tangent_to_C1 : IsTangent ℓ1 C1)
  (tangent_to_C2 : IsTangent ℓ2 C2)
  (internal_tangent_intersects_A : IsIntersection (InternalTangent C1 C2) ℓ1 A)
  (internal_tangent_intersects_B : IsIntersection (InternalTangent C1 C2) ℓ2 B)
  (distance_tangency_points : Distance (TangencyPoint ℓ1 C1) (TangencyPoint ℓ2 C1) = a) :
  Distance A B = a :=
sorry

end length_of_segment_AB_l373_373655


namespace probability_of_50_hits_in_100_trials_l373_373340

theorem probability_of_50_hits_in_100_trials :
  (let prob (hits : ℕ) (trials : ℕ) : ℚ := 1 / (trials - 1) in
   prob 50 100 = 1 / 99) :=
begin
  sorry
end

end probability_of_50_hits_in_100_trials_l373_373340


namespace circle_radius_6_l373_373413

theorem circle_radius_6 (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 6*y - k = 0 ↔ (x + 5)^2 + (y + 3)^2 = 36) → k = 2 :=
by
  sorry

end circle_radius_6_l373_373413


namespace two_S1_gt_S2_l373_373851

def M : set ℕ := { n | 1 ≤ n ∧ n ≤ 2008 }

def color (n : ℕ) : string := "color_function" -- Placeholder for the actual coloring function

def same_color (x y z : ℕ) : Prop := color x = color y ∧ color y = color z

def different_colors (x y z : ℕ) : Prop := color x ≠ color y ∧ color y ≠ color z ∧ color z ≠ color x

def S1 : set (ℕ × ℕ × ℕ) :=
{ (x, y, z) | x ∈ M ∧ y ∈ M ∧ z ∈ M ∧ same_color x y z ∧ 2008 ∣ (x + y + z) }

def S2 : set (ℕ × ℕ × ℕ) :=
{ (x, y, z) | x ∈ M ∧ y ∈ M ∧ z ∈ M ∧ different_colors x y z ∧ 2008 ∣ (x + y + z) }

theorem two_S1_gt_S2 : 2 * S1.size > S2.size := 
sorry

end two_S1_gt_S2_l373_373851


namespace cantor_bernstein_l373_373173

variables {E F : Type} (f : E → F) (g : F → E)

theorem cantor_bernstein : function.injective f → function.injective g → ∃ (h : E → F), function.bijective h :=
by
  intros hf hg
  sorry

end cantor_bernstein_l373_373173


namespace cost_per_sqft_is_6_l373_373693

-- Define the dimensions of the room
def room_length : ℕ := 25
def room_width : ℕ := 15
def room_height : ℕ := 12

-- Define the dimensions of the door
def door_height : ℕ := 6
def door_width : ℕ := 3

-- Define the dimensions of the windows
def window_height : ℕ := 4
def window_width : ℕ := 3
def number_of_windows : ℕ := 3

-- Define the total cost of whitewashing
def total_cost : ℕ := 5436

-- Calculate areas
def area_one_pair_of_walls : ℕ :=
  (room_length * room_height) * 2

def area_other_pair_of_walls : ℕ :=
  (room_width * room_height) * 2

def total_wall_area : ℕ :=
  area_one_pair_of_walls + area_other_pair_of_walls

def door_area : ℕ :=
  door_height * door_width

def window_area : ℕ :=
  window_height * window_width

def total_window_area : ℕ :=
  window_area * number_of_windows

def area_to_be_whitewashed : ℕ :=
  total_wall_area - (door_area + total_window_area)

def cost_per_sqft : ℕ :=
  total_cost / area_to_be_whitewashed

-- The theorem statement proving the cost per square foot is 6
theorem cost_per_sqft_is_6 : cost_per_sqft = 6 := 
  by
  -- Proof goes here
  sorry

end cost_per_sqft_is_6_l373_373693


namespace c_less_than_a_l373_373901

variable (a b c : ℝ)

-- Conditions definitions
def are_negative : Prop := a < 0 ∧ b < 0 ∧ c < 0
def eq1 : Prop := c = 2 * (a + b)
def eq2 : Prop := c = 3 * (b - a)

-- Theorem statement
theorem c_less_than_a (h_neg : are_negative a b c) (h_eq1 : eq1 a b c) (h_eq2 : eq2 a b c) : c < a :=
  sorry

end c_less_than_a_l373_373901


namespace scientists_group_of_five_exists_l373_373601

/-- 
 In a meeting, there are 2011 scientists. Every scientist knows at least 1509 other ones. 
 Prove that a group of five scientists can be formed so that each one in this group knows 4 people in his group. 
--/
theorem scientists_group_of_five_exists (n : ℕ) (knows : ℕ → ℕ → Prop)
  (h : n = 2011) (h_knows : ∀ s : ℕ, s < n → (∑ i in finset.filter (λ s', knows s s') (finset.range n), 1) ≥ 1509) :
  ∃ (G : simple_graph (fin n)), (G.clique_card (fin n)) ≥ 5 :=
by 
  sorry

end scientists_group_of_five_exists_l373_373601


namespace correct_average_92_l373_373687

theorem correct_average_92:
  let a := 65
      b := 106
      c := 197
      d := 74
      e := 190
      r := 125
      s := 186
      t := 287
      u := 144
      v := 230
      n := 20
      incorrect_avg := 75 in
  (n * incorrect_avg - (a + b + c + d + e) + (r + s + t + u + v)) / n = 92 :=
by
  sorry

end correct_average_92_l373_373687


namespace find_old_weight_l373_373212

variable (avg_increase : ℝ) (num_persons : ℕ) (W_new : ℝ) (total_increase : ℝ) (W_old : ℝ)

theorem find_old_weight (h1 : avg_increase = 3.5) 
                        (h2 : num_persons = 7) 
                        (h3 : W_new = 99.5) 
                        (h4 : total_increase = num_persons * avg_increase) 
                        (h5 : W_new = W_old + total_increase) 
                        : W_old = 75 :=
by
  sorry

end find_old_weight_l373_373212


namespace minimum_packs_needed_l373_373204

theorem minimum_packs_needed (cans_needed : ℕ) (packs_available : list ℕ) : 
  (cans_needed = 120) → 
  (packs_available = [6, 12, 24, 30]) → 
  ∃ num_packs : ℕ, (num_packs = 4) 
  ∧ ∃ packs_list : list ℕ, (∀ p ∈ packs_list, p ∈ packs_available)
  ∧ list.sum packs_list = cans_needed 
  ∧ list.length packs_list = num_packs :=
by
  sorry

end minimum_packs_needed_l373_373204


namespace max_value_of_expression_l373_373225

theorem max_value_of_expression (m : ℝ) : 4 - |2 - m| ≤ 4 :=
by 
  sorry

end max_value_of_expression_l373_373225


namespace count_integers_with_at_most_three_different_digits_l373_373096

theorem count_integers_with_at_most_three_different_digits :
  let count : ℕ := 
    let single_digit := 9
    let two_different_digits := 
      let combinations_no_zero := 36
      let non_repetitive_arrangements := 2 + 6 + 14 + 30
      let including_zero := 9
      let arrangements_including_zero := 1 + 3 + 7 + 15
      (combinations_no_zero * non_repetitive_arrangements) + (including_zero * arrangements_including_zero)
    let three_different_digits := 
      let combinations_three_digits_no_zero := 84
      let arrangements_no_zero := 15 + 66 + 222
      let including_zero_combinations := 36
      let arrangements_including_zero := 6 + 20 + 56
      (combinations_three_digits_no_zero * arrangements_no_zero) + (including_zero_combinations * arrangements_including_zero)

    single_digit + two_different_digits + three_different_digits
  in count = 29555 :=
by
  sorry

end count_integers_with_at_most_three_different_digits_l373_373096


namespace quadratic_roots_problem_l373_373168

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end quadratic_roots_problem_l373_373168


namespace total_exercise_time_l373_373145

theorem total_exercise_time :
  let Javier_exercise := 50 * 10
  let Sanda_exercise := 90 * 3 + 75 * 2 + 45 * 4
  let Luis_exercise := 60 * 5 + 30 * 3
  let Nita_exercise := 100 * 2 + 55 * 4
  Javier_exercise + Sanda_exercise + Luis_exercise + Nita_exercise = 1910 := by {
  let Javier_exercise := 50 * 10
  let Sanda_exercise := 90 * 3 + 75 * 2 + 45 * 4
  let Luis_exercise := 60 * 5 + 30 * 3
  let Nita_exercise := 100 * 2 + 55 * 4
  have t1 : Javier_exercise = 500 := by
    rw [mul_comm, mul_comm, mul_assoc, mul_comm, mul_assoc]
    rfl
  have t2 : Sanda_exercise = 600 := by
    rw [add_assoc, add_assoc, mul_assoc, add_comm, mul_comm, mul_assoc, mul_assoc, add_comm]
    rfl
  have t3 : Luis_exercise = 390 := by
    rw [mul_comm, mul_comm, mul_assoc, mul_comm, mul_assoc]
    rfl
  have t4 : Nita_exercise = 420 := by
    rw [mul_comm, mul_comm, mul_assoc, mul_comm, mul_assoc]
    rfl
  simp only [t1, t2, t3, t4]
  rfl
} sorry

end total_exercise_time_l373_373145


namespace option_A_correct_option_C_correct_l373_373957

noncomputable def f (x : ℝ) : ℝ :=
  cos (2 * x) - 2 * real.sqrt 3 * sin x * cos x

theorem option_A_correct :
  ∃ x₁ x₂, x₁ - x₂ = real.pi ∧ f x₁ = f x₂ :=
by
  use [0, real.pi]
  sorry

theorem option_C_correct :
  f (real.pi / 12) = 0 :=
by
  have h : f (real.pi / 12) = 2 * cos (2 * (real.pi / 12) + real.pi / 3) := by 
    sorry
  have h2 : 2 * (real.pi / 12) + real.pi / 3 = real.pi / 2 := by
    sorry
  rw [h, h2]
  exact cos_pi_div_two

end option_A_correct_option_C_correct_l373_373957


namespace twigs_per_branch_l373_373622

/-- Definitions -/
def total_branches : ℕ := 30
def total_leaves : ℕ := 12690
def percentage_4_leaves : ℝ := 0.30
def leaves_per_twig_4_leaves : ℕ := 4
def percentage_5_leaves : ℝ := 0.70
def leaves_per_twig_5_leaves : ℕ := 5

/-- Given conditions translated to Lean -/
def hypothesis (T : ℕ) : Prop :=
  (percentage_4_leaves * T * leaves_per_twig_4_leaves) +
  (percentage_5_leaves * T * leaves_per_twig_5_leaves) = total_leaves

/-- The main theorem to prove -/
theorem twigs_per_branch
  (T : ℕ)
  (h : hypothesis T) :
  (T / total_branches) = 90 :=
sorry

end twigs_per_branch_l373_373622


namespace coin_balance_possible_l373_373038

theorem coin_balance_possible (n : ℕ) (heads tails : Fin (2^n) → Bool) :
  (∀ i, (heads i = true ∨ heads i = false) ∧ (tails i = true ∨ tails i = false)) →
  ∃ flips : ℕ, flips ≤ n ∧
  (∀ f : Fin (2^n), heads f = true ↔ tails f = false) ∧
  (∑ i, if heads i = true then 1 else 0) = (∑ i, if tails i = true then 1 else 0) :=
by
  sorry

end coin_balance_possible_l373_373038


namespace bricks_in_wall_l373_373826

theorem bricks_in_wall (h : ℕ) 
  (brenda_rate : ℕ := h / 8)
  (brandon_rate : ℕ := h / 12)
  (combined_rate : ℕ := (5 * h) / 24)
  (decreased_combined_rate : ℕ := combined_rate - 15)
  (work_time : ℕ := 6) :
  work_time * decreased_combined_rate = h → h = 360 := by
  intros h_eq
  sorry

end bricks_in_wall_l373_373826


namespace problem1_problem2_l373_373776

-- Problem 1
theorem problem1 (P : ℝ × ℝ) (hP : P = (4, -3)) : 
  let α : ℝ := real.angle_of_point P
  in 2 * real.sin α + real.cos α = -2 / 5 :=
by sorry

-- Problem 2
theorem problem2 (P : ℝ × ℝ) (m : ℝ) (hm : m ≠ 0) 
  (hP : P = (-real.sqrt 3, m)) 
  (hα : real.sin (real.angle_of_point P) = (real.sqrt 2 * m) / 4) :
  let α : ℝ := real.angle_of_point P
  in real.cos α = -real.sqrt 6 / 4 ∧ (real.tan α = -real.sqrt 15 / 3 ∨ real.tan α = real.sqrt 15 / 3) :=
by sorry

end problem1_problem2_l373_373776


namespace sqrt_subtraction_l373_373264

theorem sqrt_subtraction : 
  sqrt (49 + 121) - sqrt (36 - 9) = sqrt 170 - sqrt 27 :=
by
  sorry

end sqrt_subtraction_l373_373264


namespace tom_strokes_over_par_l373_373724

noncomputable def total_strokes (strokes_per_hole : list ℝ) (holes_per_round : ℝ) : ℝ :=
  strokes_per_hole.sum * holes_per_round

noncomputable def total_par (par_per_hole : ℝ) (holes_per_round : ℝ) (rounds : ℝ) : ℝ :=
  par_per_hole * holes_per_round * rounds

theorem tom_strokes_over_par
  (average_strokes : list ℝ)
  (holes_per_round : ℝ)
  (par_per_hole : ℝ)
  (rounds : ℝ) 
  (h_average_strokes_len : average_strokes.length = 5)
  (h_average_strokes_values : average_strokes = [4, 3.5, 5, 3, 4.5])
  (h_holes_per_round : holes_per_round = 9)
  (h_par_per_hole : par_per_hole = 3)
  (h_rounds : rounds = 5) : 
  total_strokes average_strokes holes_per_round - total_par par_per_hole holes_per_round rounds = 45 :=
by
  sorry

end tom_strokes_over_par_l373_373724


namespace remainder_of_large_number_l373_373747

theorem remainder_of_large_number (n : ℕ) (r : ℕ) (h : n = 2468135792) :
  (n % 101) = 52 := 
by
  have h1 : (10 ^ 8 - 1) % 101 = 0 := sorry
  have h2 : (10 ^ 6 - 1) % 101 = 0 := sorry
  have h3 : (10 ^ 4 - 1) % 101 = 0 := sorry
  have h4 : (10 ^ 2 - 1) % 101 = 99 % 101 := sorry

  -- Using these properties to simplify n
  have n_decomposition : 2468135792 = 24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 := sorry
  have div_property : 
    (24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 - (24 + 68 + 13 + 57 + 92)) % 101 = 0 := sorry

  have simplified_sum : (24 + 68 + 13 + 57 + 92 = 254 := by norm_num) := sorry
  have resulting_mod : 254 % 101 = 52 := by norm_num

  -- Thus n % 101 = 52
  exact resulting_mod

end remainder_of_large_number_l373_373747


namespace reflected_ray_equation_l373_373651

theorem reflected_ray_equation (x y : ℝ) (incident_ray : y = 2 * x + 1) (reflecting_line : y = x) :
  x - 2 * y - 1 = 0 :=
sorry

end reflected_ray_equation_l373_373651


namespace find_p_max_area_triangle_l373_373524

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373524


namespace probability_two_defective_out_of_three_l373_373950

theorem probability_two_defective_out_of_three :
  let total_items := 100 in
  let defective_items := 10 in
  let selected_items := 3 in
  (let favorable := (Nat.choose defective_items 2) * (Nat.choose (total_items - defective_items) 1) in
   let total := Nat.choose total_items selected_items in
   (favorable : ℚ) / total = 27 / 1078) :=
by
  -- Assume the variables and intermediate computations
  let total_items := 100
  let defective_items := 10
  let selected_items := 3
  let favorable := (Nat.choose defective_items 2) * (Nat.choose (total_items - defective_items) 1)
  let total := Nat.choose total_items selected_items
  -- State the expected result
  have : (favorable : ℚ) / total = 27 / 1078 := sorry
  exact have

end probability_two_defective_out_of_three_l373_373950


namespace symmetric_parabola_equation_l373_373219

theorem symmetric_parabola_equation (x y : ℝ) (h : y^2 = 2 * x) : (y^2 = -2 * (x + 2)) :=
by
  sorry

end symmetric_parabola_equation_l373_373219


namespace earnings_for_5_hours_l373_373333

theorem earnings_for_5_hours (rate_per_hour earnings_three_hours : ℝ) (hours_one := 3) 
                             (earnings_one := 45) (hours_two := 5) : 
                             earnings_three_hours = hours_one * rate_per_hour → 
                             rate_per_hour = earnings_one / hours_one →
                             earnings_two = hours_two * rate_per_hour →
                             earnings_two = 75 :=
by
  intros h1 h2 h3
  rw [h2, h1] at h3
  sorry -- proof is omitted

end earnings_for_5_hours_l373_373333


namespace vanya_dima_probability_l373_373794

-- Defining the probability that two specific people out of ten sit next to each other in a circular arrangement
theorem vanya_dima_probability :
  let n := 10 in
  (2:ℚ) / (n-1) = 2 / 9 :=
by
  have h1 : (n-1) = 9 := by rfl
  simp [h1]
  norm_num
  sorry

end vanya_dima_probability_l373_373794


namespace number_of_arrangements_l373_373889

-- Define the students
inductive Student
| A | B | C | D | E

open Student

def is_not_at_end (l : List Student) : Prop :=
  match l with
  | [] | [_] => False
  | x::y::xs => (x ≠ A) ∧ (List.last? (y::xs) ≠ some A)

def are_adjacent_C_D (l : List Student) : Prop :=
  l == [C, D] ++ l.drop 2 ∨ l == D::C::l.drop 2

def count_valid_arrangements (students : List Student) : Nat :=
  -- Count arrangements where A is not at either end and C and D are adjacent
  sorry

theorem number_of_arrangements : count_valid_arrangements [A, B, C, D, E] = 24 :=
  sorry

end number_of_arrangements_l373_373889


namespace scientific_notation_correct_l373_373661

theorem scientific_notation_correct :
  52000000 = 5.2 * 10^7 :=
sorry

end scientific_notation_correct_l373_373661


namespace remainder_when_P_divided_by_ab_l373_373900

-- Given conditions
variables {P a b c Q Q' R R' : ℕ}

-- Provided equations as conditions
def equation1 : P = a * Q + R :=
sorry

def equation2 : Q = (b + c) * Q' + R' :=
sorry

-- Proof problem statement
theorem remainder_when_P_divided_by_ab :
  P % (a * b) = (a * c * Q' + a * R' + R) % (a * b) :=
by
  sorry

end remainder_when_P_divided_by_ab_l373_373900


namespace constant_term_exists_l373_373286

theorem constant_term_exists (n : ℕ) (h : n = 6) : 
  (∃ r : ℕ, 2 * n - 3 * r = 0) ∧ 
  (∃ n' r' : ℕ, n' ≠ 6 ∧ 2 * n' - 3 * r' = 0) := by
  sorry

end constant_term_exists_l373_373286


namespace inequality_holds_l373_373115

-- Define the function f and the domain D
variable {D : Type} [Nonempty D] [OrderedRing D]
variable (f : D → ℝ)
variable h_positive : ∀ x, 0 < f x

-- Define the condition on the function f 
axiom h_condition : ∀ (x y : D), 
  f(x) * f(y) ≤ (f((x + y) / 2)) ^ 2
  ∧ (f(x) * f(y) = (f((x + y) / 2)) ^ 2 ↔ x = y)

-- State the theorem to be proven
theorem inequality_holds {n : ℕ} (x : Fin n → D) :
  (∏ i, f (x i)) ≤ (f (∑ i, x i / n)) ^ n
  ∧ ((∏ i, f (x i) = (f (∑ i, x i / n)) ^ n) ↔ ∀ i j, x i = x j) :=
sorry

end inequality_holds_l373_373115


namespace find_a_n_l373_373613

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n, a (n + 2) = 7 * a (n + 1) - 12 * a n

theorem find_a_n (a : ℕ → ℤ) (h : sequence a) : 
  ∀ n, a n = 2 * 3^(n-1) - 4^(n-1) := 
sorry

end find_a_n_l373_373613


namespace tan_add_tan_plus_one_eq_six_tan_cot_add_eq_six_l373_373052

theorem tan_add_tan_plus_one_eq_six_tan_cot_add_eq_six
    (x y : ℝ) 
    (h1 : Real.tan x + Real.tan y + 1 = 6) 
    (h2 : Real.cot x + Real.cot y = 6) : 
    Real.tan (x + y) = 30 := 
sorry

end tan_add_tan_plus_one_eq_six_tan_cot_add_eq_six_l373_373052


namespace pears_left_l373_373144

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) (total_pears : ℕ) (pears_left : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) 
  (h4 : total_pears = jason_pears + keith_pears) 
  (h5 : pears_left = total_pears - mike_ate) 
  : pears_left = 81 :=
by
  sorry

end pears_left_l373_373144


namespace work_done_together_in_one_day_l373_373273

-- Defining the conditions
def time_to_finish_a : ℕ := 12
def time_to_finish_b : ℕ := time_to_finish_a / 2

-- Defining the work done in one day
def work_done_by_a_in_one_day : ℚ := 1 / time_to_finish_a
def work_done_by_b_in_one_day : ℚ := 1 / time_to_finish_b

-- The proof statement
theorem work_done_together_in_one_day : 
  work_done_by_a_in_one_day + work_done_by_b_in_one_day = 1 / 4 := by
  sorry

end work_done_together_in_one_day_l373_373273


namespace parabola_circle_distance_l373_373466

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373466


namespace find_p_max_area_triangle_l373_373522

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373522


namespace P_implies_Q_Q_not_implies_P_sufficient_but_not_necessary_l373_373443

variable {a b c : ℝ}

/-- Proposition P states that a, b, and c can form the sides of a triangle -/
def P (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

/-- Proposition Q states the inequality a^2 + b^2 + c^2 < 2(ab + bc + ca) -/
def Q (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + c * a)

/-- Theorem stating that if P holds then Q also holds -/
theorem P_implies_Q (h : P a b c) : Q a b c := by
  sorry

/-- Theorem stating that if Q holds then P might not hold -/
theorem Q_not_implies_P (h : Q a b c) : ¬ P a b c := by
  sorry

/- Theoreme stating that P is a sufficient but not necessary condition for Q -/
theorem sufficient_but_not_necessary : (P a b c → Q a b c) ∧ ¬ (Q a b c → P a b c) := by
  exact ⟨P_implies_Q, Q_not_implies_P⟩

end P_implies_Q_Q_not_implies_P_sufficient_but_not_necessary_l373_373443


namespace smallest_r_minus_p_l373_373718

theorem smallest_r_minus_p (p q r : ℕ) (h1 : p > 0) (h2 : q > 0) (h3 : r > 0)
  (h4 : p * q * r = Nat.factorial 9) (h5 : p < q) (h6 : q < r) :
  r - p = 396 :=
sorry

end smallest_r_minus_p_l373_373718


namespace square_banner_total_shaded_area_correct_l373_373802

noncomputable def square_banner_total_shaded_area : ℝ :=
  let S := 3 in
  let T := 3 / 4 in
  let large_shaded_area := S * S in
  let small_shaded_area := (T * T) * 12 in
  large_shaded_area + small_shaded_area

theorem square_banner_total_shaded_area_correct :
  square_banner_total_shaded_area = 15.75 :=
begin
  unfold square_banner_total_shaded_area,
  norm_num,
  sorry,
end

end square_banner_total_shaded_area_correct_l373_373802


namespace trapezoid_shorter_base_length_l373_373701

theorem trapezoid_shorter_base_length (longer_base : ℕ) (segment_length : ℕ) (shorter_base : ℕ) 
  (h1 : longer_base = 120) (h2 : segment_length = 7)
  (h3 : segment_length = (longer_base - shorter_base) / 2) : 
  shorter_base = 106 := by
  sorry

end trapezoid_shorter_base_length_l373_373701


namespace shaded_area_correct_l373_373879

def semicircle_area_shaded (R : ℝ) (α : ℝ) : ℝ :=
  if α = real.pi / 6 then
    real.pi * R^2 / 3
  else
    0

theorem shaded_area_correct (R : ℝ) (hR : 0 ≤ R) :
  semicircle_area_shaded R (real.pi / 6) = real.pi * R^2 / 3 :=
by sorry

end shaded_area_correct_l373_373879


namespace both_events_interval_l373_373447

noncomputable def probability_a : ℝ := 5 / 6
noncomputable def probability_b : ℝ := 1 / 2

theorem both_events_interval :
  ∃ (p : ℝ), p ∈ set.Icc (1 / 3) (1 / 2) :=
by
  have hA : probability_a = 5 / 6 := by rfl
  have hB : probability_b = 1 / 2 := by rfl
  sorry

end both_events_interval_l373_373447


namespace average_books_correct_l373_373599

-- Definitions corresponding to the problem conditions
def total_students : ℕ := 40
def students_borrowed_0_books : ℕ := 2
def students_borrowed_1_book : ℕ := 12
def students_borrowed_2_books : ℕ := 12
def students_borrowed_at_least_3_books : ℕ :=
  total_students - (students_borrowed_0_books + students_borrowed_1_book + students_borrowed_2_books)

-- Calculate the total number of books borrowed, assuming the minimum for the "at least 3" group
def total_books_borrowed : ℕ :=
  (students_borrowed_0_books * 0) +
  (students_borrowed_1_book * 1) +
  (students_borrowed_2_books * 2) +
  (students_borrowed_at_least_3_books * 3)

-- Average number of books per student
def average_books_per_student : ℚ :=
  total_books_borrowed.to_rat / total_students.to_rat

theorem average_books_correct :
  average_books_per_student = 1.95 :=
by
  -- This is the point where the proof would generally be constructed.
  -- The actual mathematical proof steps would validate that
  -- average_books_per_student indeed equals 1.95, matching the problem's solution.
  sorry

end average_books_correct_l373_373599


namespace solve_k_l373_373969

theorem solve_k (x y k : ℝ) (h1 : x + 2 * y = k - 1) (h2 : 2 * x + y = 5 * k + 4) (h3 : x + y = 5) :
  k = 2 :=
sorry

end solve_k_l373_373969


namespace triangle_side_length_l373_373616

theorem triangle_side_length (A B C D E F : Type) [Triangle A B C] [AngleBisector AD] [Median CE]
  (h1 : ∃ F, ∠ADF = 90 ∧ ∠CDF = 90)
  (h2 : E = midpoint B C) :
  AB = 2 * AC :=
sorry

end triangle_side_length_l373_373616


namespace polynomial_P1_l373_373215

theorem polynomial_P1 :
  ∃ (a_4 a_3 a_2 a_1 a_0 : ℕ), 
    a_4 < 100 ∧ a_3 < 100 ∧ a_2 < 100 ∧ a_1 < 100 ∧ a_0 < 100 ∧
    (a_4 * 10^4 + a_3 * 10^3 + a_2 * 10^2 + a_1 * 10 + a_0 = 331633) ∧
    (a_4 * 10^4 - a_3 * 10^3 + a_2 * 10^2 - a_1 * 10 + a_0 = 273373) ∧
    (a_4 + a_3 + a_2 + a_1 + a_0 = 100) :=
begin
  sorry
end

end polynomial_P1_l373_373215


namespace tangent_line_eqn_at_point_max_min_interval_l373_373961

noncomputable def f (x : ℝ) : ℝ := (2 / 3) * x^3 - 2 * x^2 + 3

-- Given: A function f
-- 1. Prove the equation of the tangent line at the point (1, 5/3)
theorem tangent_line_eqn_at_point :
  let y := (5 / 3 : ℝ)
  let tangent_line_eqn := (6 * (1 : ℝ) + 3 * y - 11 = 0) 
  tangent_line_eqn := by 
    sorry

-- 2. Prove the maximum and minimum values of the function on the interval [-1, 3]
theorem max_min_interval :
  let max_value := (3 : ℝ)
  let min_value := (1 / 3 : ℝ)
  (∃ x ∈ Ici (-1 : ℝ) ∩ Iic 3, f x = max_value) ∧
  (∃ x ∈ Ici (-1 : ℝ) ∩ Iic 3, f x = min_value) := by
    sorry

end tangent_line_eqn_at_point_max_min_interval_l373_373961


namespace solve_for_x_l373_373205

theorem solve_for_x (x : ℚ) : (1 / 3) + (1 / x) = (3 / 4) → x = 12 / 5 :=
by
  intro h
  -- Proof goes here
  sorry

end solve_for_x_l373_373205


namespace proof_problem_l373_373928

-- Declare x, y as real numbers
variables (x y : ℝ)

-- Define the condition given in the problem
def condition (k : ℝ) : Prop :=
  (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k

-- The main conclusion we need to prove given the condition
theorem proof_problem (k : ℝ) (h : condition x y k) :
  (x^8 + y^8) / (x^8 - y^8) + (x^8 - y^8) / (x^8 + y^8) = (k^4 + 24 * k^2 + 16) / (4 * k^3 + 16 * k) :=
sorry

end proof_problem_l373_373928


namespace number_of_valid_arrangements_l373_373891

theorem number_of_valid_arrangements :
  let students := ["A", "B", "C", "D", "E"]
  let total_students := 5
  let end_positions := [0, total_students - 1]
  let adjacent_positions := λ l : List Nat, ∃ s1 s2, l.get! s1 = "C" ∧ l.get! s2 = "D" ∧ (s1 = s2 + 1 ∨ s2 = s1 + 1)
  let valid_positions := λ l : List Nat, ∀ i ∈ end_positions, l.get! i ≠ "A"
  students.nodup ∧ adjacent_positions students ∧ valid_positions students ↔ length (List.permutations students) = 24
:= sorry

end number_of_valid_arrangements_l373_373891


namespace AF_perpendicular_to_BC_l373_373345

variables {A B C D E F : Type}
variables {α β γ δ ε : ℝ}
variables {BAC ABC CBD BCE BAF FCB : ℝ}
variables [triangle ABC]
variables [point D on AC]
variables [point E on AB]
variables [point F where BD and CE intersect]

theorem AF_perpendicular_to_BC 
  (h1 : ∠BAC = 40)
  (h2 : ∠ABC = 60)
  (h3 : ∠CBD = 40)
  (h4 : ∠BCE = 70)
  (h5 : F = intersection lines BD CE)
  : is_perpendicular (line_through A F) (line_through B C) :=
sorry

end AF_perpendicular_to_BC_l373_373345


namespace fitting_function_l373_373128

theorem fitting_function :
  (∀ x y, (x = 0.50 ∧ y = -0.99) ∨ (x = 0.99 ∧ y = 0.01) ∨ (x = 2.01 ∧ y = 0.98) ∨ (x = 3.98 ∧ y = 2.00) →
  (y = Math.log2 x)) :=
by
  sorry

end fitting_function_l373_373128


namespace find_k_l373_373977

def vector_collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, v1 = (λ * v2.1, λ * v2.2)

theorem find_k (a b c : ℝ × ℝ) (k : ℝ) 
  (ha : a = (1, 3)) (hb : b = (-2, 1)) (hc : c = (3, 2)) :
  vector_collinear c (k * a.1 + b.1, k * a.2 + b.2) ↔ k = -1 :=
by
  sorry

end find_k_l373_373977


namespace find_p_max_area_of_triangle_l373_373508

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373508


namespace sum_of_digits_of_base9_product_of_base7_34_and_52_l373_373829

def sum_of_base9_digits (n : ℕ) : ℕ :=
  (n.digits 9).sum

def base7_to_decimal (n : ℕ) : ℕ :=
  (n.digits 7).to_nat

def base9_product_of_base7_nums (a b : ℕ) : ℕ :=
  let dec_a := base7_to_decimal a
  let dec_b := base7_to_decimal b
  let prod := dec_a * dec_b
  prod

theorem sum_of_digits_of_base9_product_of_base7_34_and_52 :
  sum_of_base9_digits (base9_product_of_base7_nums 34 52) = 10 :=
by
  sorry

end sum_of_digits_of_base9_product_of_base7_34_and_52_l373_373829


namespace gcd_max_possible_value_l373_373401

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end gcd_max_possible_value_l373_373401


namespace smallest_prime_factor_in_C_is_68_l373_373676

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) : ℕ :=
if is_prime n then n else
  Nat.find (λ p, p ∣ n ∧ is_prime p)

def C : set ℕ := {65, 67, 68, 71, 73}

theorem smallest_prime_factor_in_C_is_68 :
  ∃ x ∈ C, smallest_prime_factor x = 2 :=
by
  use 68
  sorry

end smallest_prime_factor_in_C_is_68_l373_373676


namespace max_sequence_length_l373_373867

def valid_sequence (s : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, 0 ≤ n ∧ n < 7 → ∑ i in (Finset.range 7), s (n + i) > 0) ∧
  (∀ n : ℕ, 0 ≤ n ∧ n < 11 → ∑ i in (Finset.range 11), s (n + i) < 0)

theorem max_sequence_length {s : ℕ → ℤ} (h : valid_sequence s) : 
  ∃ (n : ℕ), n = 16 ∧ (¬ ∃ m > 16, valid_sequence (λ i, s (i % m))) :=
sorry

end max_sequence_length_l373_373867


namespace dot_product_two_a_plus_b_with_a_l373_373084

-- Define vector a
def a : ℝ × ℝ := (2, -1)

-- Define vector b
def b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication of vector a by 2
def two_a : ℝ × ℝ := (2 * a.1, 2 * a.2)

-- Define the vector addition of 2a and b
def two_a_plus_b : ℝ × ℝ := (two_a.1 + b.1, two_a.2 + b.2)

-- Define dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that the dot product of (2 * a + b) and a equals 6
theorem dot_product_two_a_plus_b_with_a :
  dot_product two_a_plus_b a = 6 :=
by
  sorry

end dot_product_two_a_plus_b_with_a_l373_373084


namespace natural_numbers_exist_l373_373388

noncomputable def problem_statement : Prop :=
  ∃ a b c : ℕ, (∀ n : ℕ, n > 2 →
    (b - c / (n-2)!) < (∑ k in (Finset.range n).filter (≥2), (k^3 - a) / k!) ∧
    (∑ k in (Finset.range n).filter (≥2), (k^3 - a) / k!) < b) ∧
    a = 5 ∧ b = 9 ∧ c = 4

theorem natural_numbers_exist : problem_statement :=
  sorry

end natural_numbers_exist_l373_373388


namespace proof_part1_proof_part2_l373_373479

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373479


namespace convert_to_spherical_l373_373366

noncomputable def spherical_coordinates (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let ρ := Real.sqrt (x * x + y * y + z * z)
  let φ := Real.arccos (z / ρ)
  let r := Real.sqrt (x * x + y * y)
  let θ := if r = 0 then 0 else Real.atan2 y x
  (ρ, θ, φ)

theorem convert_to_spherical :
  spherical_coordinates 4 (4 * Real.sqrt 2) 4 = (8, π / 4, π / 3) :=
  by
    sorry

end convert_to_spherical_l373_373366


namespace no_two_digit_numbers_satisfy_condition_l373_373344

theorem no_two_digit_numbers_satisfy_condition :
  ¬ ∃ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
  (10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d :=
by
  sorry

end no_two_digit_numbers_satisfy_condition_l373_373344


namespace count_three_digit_perfect_squares_divisible_by_4_l373_373104

theorem count_three_digit_perfect_squares_divisible_by_4 :
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), 10 ≤ k ∧ k ≤ 31 → (∃ m : ℕ, m^2 = k^2 ∧ 100 ≤ m^2 ∧ m^2 ≤ 999 ∧ m^2 % 4 = 0) := 
sorry

end count_three_digit_perfect_squares_divisible_by_4_l373_373104


namespace problem1_proof_l373_373287

-- Define the mathematical conditions and problems
def problem1_expression (x y : ℝ) : ℝ := y * (4 * x - 3 * y) + (x - 2 * y) ^ 2

-- State the theorem with the simplified form as the conclusion
theorem problem1_proof (x y : ℝ) : problem1_expression x y = x^2 + y^2 :=
by
  sorry

end problem1_proof_l373_373287


namespace max_convex_polygon_sides_with_5_obtuse_angles_l373_373305

theorem max_convex_polygon_sides_with_5_obtuse_angles (n : ℕ) (hconvex : convex_polygon n)
  (hobtuse : ∃ k : ℕ, k = 5 ∧ obtuse_angles k n) (hnge2 : n ≥ 3) : n ≤ 8 :=
by
  -- given conditions for convex polygon
  -- sum of interior angles formula validity
  -- Not solving, just setting up the theorem.
  sorry

-- Definitions for convex polygon, obtuse angles, etc.
def convex_polygon (n : ℕ) : Prop :=
-- a placeholder definition
  n ≥ 3 ∧ n < 9

def obtuse_angles (k : ℕ) (n : ℕ) : Prop :=
-- a placeholder definition
  k = 5 ∧ ∀ i < k, i > 90

end max_convex_polygon_sides_with_5_obtuse_angles_l373_373305


namespace unique_positive_integer_n_l373_373847

-- Definitions based on conditions
def is_divisor (n a : ℕ) : Prop := a % n = 0

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- The main theorem statement
theorem unique_positive_integer_n : ∃ (n : ℕ), n > 0 ∧ is_divisor n 1989 ∧
    is_perfect_square (n^2 - 1989 / n) ∧ n = 13 :=
by
  sorry

end unique_positive_integer_n_l373_373847


namespace AC_length_l373_373866

variable {A B C D : Type}

structure Point := (x : ℝ) (y : ℝ)

structure Trapezoid (A B C D : Point) : Prop :=
(parallel : AB.y = DC.y)
(length_AB : AB.x - DC.x = 8)
(length_BC : (BC.x - AB.x) ^ 2 + (BC.y - AB.y) ^ 2 = 4 * (4))
(angle_BCD : ∠BCD = 45)
(angle_CDA : ∠CDA = 45)

def diagonal_length_AC {A B C D : Point} (h : Trapezoid A B C D) : ℝ :=
sqrt ((C.x - A.x) ^ 2 + (C.y - A.y) ^ 2)

theorem AC_length {A B C D : Point} (h : Trapezoid A B C D) :
  diagonal_length_AC h = 4 * sqrt 5 := 
sorry

end AC_length_l373_373866


namespace triangle_DEF_is_right_angled_and_isosceles_l373_373633

noncomputable def is_midpoint (F B C : ℝ × ℝ) : Prop := F = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

noncomputable def is_isosceles_right_triangle (A B D E : ℝ × ℝ) : Prop :=
  (dist A D = dist D B) ∧ (dist A D = dist A B * sqrt 2)

theorem triangle_DEF_is_right_angled_and_isosceles
  {A B C D E F : ℝ × ℝ}
  (hF_mid : is_midpoint F B C)
  (hABD_isosceles_right : is_isosceles_right_triangle A B D E)
  (hACE_isosceles_right : is_isosceles_right_triangle A C E D) :
  (∃ F, is_right_angle_triangle D E F ∧ is_isosceles_triangle D E F) :=
sorry

end triangle_DEF_is_right_angled_and_isosceles_l373_373633


namespace no_such_A_exists_l373_373884

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_A_exists :
  ¬ ∃ A : ℕ, 0 < A ∧ digit_sum A = 16 ∧ digit_sum (2 * A) = 17 :=
by 
  sorry

end no_such_A_exists_l373_373884


namespace sqrt_div_eq_five_over_two_l373_373014

theorem sqrt_div_eq_five_over_two
  (x y : ℝ)
  (h : ( ( 1 / 3 : ℝ ) ^ 2 + ( 1 / 4 ) ^ 2 ) / ( ( 1 / 5 ) ^ 2 + ( 1 / 6 ) ^ 2 ) = 25 * x / (73 * y)) :
  (sqrt x) / (sqrt y) = 5 / 2 := 
by 
  sorry

end sqrt_div_eq_five_over_two_l373_373014


namespace fixed_point_exists_l373_373325

variables {S : Type*} [fintype S]
variables (n : ℕ) (hn : odd n) (S : finset (ℤ × ℤ)) (hS : S.card = n) (f : (ℤ × ℤ) → (ℤ × ℤ))
variables (h_inj : function.injective f)
variables (h_dist : ∀ (A B : (ℤ × ℤ)), (A ∈ S ∧ B ∈ S) → dist (f A) (f B) ≥ dist A B)

-- Main statement
theorem fixed_point_exists : ∃ X ∈ S, f X = X := sorry

end fixed_point_exists_l373_373325


namespace only_integer_n_l373_373090

def P (n : ℤ) : ℤ := n^4 + 2 * n^3 + 2 * n^2 + 2 * n + 1

theorem only_integer_n (n : ℤ) : (Nat.prime (Int.natAbs (P n)) → n = -2) := by
  sorry

end only_integer_n_l373_373090


namespace wax_current_amount_l373_373087

theorem wax_current_amount (wax_needed wax_total : ℕ) (h : wax_needed + 11 = wax_total) : 11 = wax_total - wax_needed :=
by
  sorry

end wax_current_amount_l373_373087


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373493

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373493


namespace problem_sequence_k_term_l373_373922

theorem problem_sequence_k_term (a : ℕ → ℤ) (S : ℕ → ℤ) (h₀ : ∀ n, S n = n^2 - 9 * n)
    (h₁ : ∀ n, a n = S n - S (n - 1)) (h₂ : 5 < a 8 ∧ a 8 < 8) : 8 = 8 :=
sorry

end problem_sequence_k_term_l373_373922


namespace line_intersection_l373_373797

theorem line_intersection (t u : ℝ) :
  (∃ t u : ℝ, (1 + 2 * t = 5 + 4 * u) ∧ (1 - 3 * t = -9 + 2 * u)) →
  ∃ x y : ℝ, (x = 1 + 2 * 3) ∧ (y = 1 - 3 * 3) ∧ (x = 7) ∧ (y = -8) :=
by
  intro h
  use 7, -8
  split
  sorry
  sorry

end line_intersection_l373_373797


namespace setA_def_setB_def_union_AB_intersection_AB_l373_373073

def f (x : ℝ) : ℝ := Real.log (x^2 - 5*x + 6)
def g (x : ℝ) : ℝ := Real.sqrt ((4 / x) - 1)

def setA : Set ℝ := {x | x > 3 ∨ x < 2}
def setB : Set ℝ := {x | 0 < x ∧ x ≤ 4}

theorem setA_def : setA = {x | x > 3 ∨ x < 2} := by
  sorry

theorem setB_def : setB = {x | 0 < x ∧ x ≤ 4} := by
  sorry

theorem union_AB : setA ∪ setB = Set.univ := by
  sorry

theorem intersection_AB : setA ∩ setB = {x | (0 < x ∧ x < 2) ∨ (3 < x ∧ x ≤ 4)} := by
  sorry

end setA_def_setB_def_union_AB_intersection_AB_l373_373073


namespace find_p_max_area_triangle_l373_373540

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373540


namespace shaded_area_correct_l373_373132

def first_rectangle_area (w l : ℕ) : ℕ := w * l
def second_rectangle_area (w l : ℕ) : ℕ := w * l
def overlap_triangle_area (b h : ℕ) : ℕ := (b * h) / 2
def total_shaded_area (area1 area2 overlap : ℕ) : ℕ := area1 + area2 - overlap

theorem shaded_area_correct :
  let w1 := 4
  let l1 := 12
  let w2 := 5
  let l2 := 10
  let b := 4
  let h := 5
  let area1 := first_rectangle_area w1 l1
  let area2 := second_rectangle_area w2 l2
  let overlap := overlap_triangle_area b h
  total_shaded_area area1 area2 overlap = 88 := 
by
  sorry

end shaded_area_correct_l373_373132


namespace stack_map_front_view_l373_373849

theorem stack_map_front_view (r1 r2 r3 : list ℕ)
  (h1 : r1 = [4, 1])
  (h2 : r2 = [1, 2, 4])
  (h3 : r3 = [3, 1]) :
  front_view (r1, r2, r3) = [4, 2, 4] :=
by
  sorry

end stack_map_front_view_l373_373849


namespace suff_but_not_nec_l373_373654

def M (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 > 0
def N (a : ℝ) : Prop := ∃ x : ℝ, (a - 3) * x + 1 = 0

theorem suff_but_not_nec (a : ℝ) : M a → N a ∧ ¬(N a → M a) := by
  sorry

end suff_but_not_nec_l373_373654


namespace max_area_triangle_PAB_l373_373545

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373545


namespace number_of_smallest_squares_l373_373782

-- Conditions
def length_cm : ℝ := 28
def width_cm : ℝ := 48
def total_lines_cm : ℝ := 6493.6

-- The main question is about the number of smallest squares
theorem number_of_smallest_squares (d : ℝ) (h_d : d = 0.4) :
  ∃ n : ℕ, n = (length_cm / d - 2) * (width_cm / d - 2) ∧ n = 8024 :=
by
  sorry

end number_of_smallest_squares_l373_373782


namespace area_bounded_region_l373_373953

theorem area_bounded_region : 
  (∃ x y : ℝ, x^2 + y^2 = 2 * abs (x - y) + 2 * abs (x + y)) →
  (bounded_area : ℝ) = 16 * Real.pi :=
by
  sorry

end area_bounded_region_l373_373953


namespace part_I_1_part_I_2_part_II_l373_373949

noncomputable def universal_set := set.univ
def set_A : set ℝ := {x | ∃ y, y = real.sqrt (x-1) + real.sqrt (3-x)}
def set_B : set ℝ := {x | real.log x / real.log 2 > 1}
def set_C (a : ℝ) : set ℝ := {x | 1 < x ∧ x < a}

theorem part_I_1 : set_A ∩ set_B = {x | 2 < x ∧ x ≤ 3} :=
sorry

theorem part_I_2 : (set.univ \ set_B) ∪ set_A = {x | x ≤ 3} :=
sorry

theorem part_II : ∀ a : ℝ, set_C a ⊆ set_A → a ≤ 3 :=
sorry

end part_I_1_part_I_2_part_II_l373_373949


namespace remainder_of_large_number_l373_373746

theorem remainder_of_large_number (n : ℕ) (r : ℕ) (h : n = 2468135792) :
  (n % 101) = 52 := 
by
  have h1 : (10 ^ 8 - 1) % 101 = 0 := sorry
  have h2 : (10 ^ 6 - 1) % 101 = 0 := sorry
  have h3 : (10 ^ 4 - 1) % 101 = 0 := sorry
  have h4 : (10 ^ 2 - 1) % 101 = 99 % 101 := sorry

  -- Using these properties to simplify n
  have n_decomposition : 2468135792 = 24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 := sorry
  have div_property : 
    (24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 - (24 + 68 + 13 + 57 + 92)) % 101 = 0 := sorry

  have simplified_sum : (24 + 68 + 13 + 57 + 92 = 254 := by norm_num) := sorry
  have resulting_mod : 254 % 101 = 52 := by norm_num

  -- Thus n % 101 = 52
  exact resulting_mod

end remainder_of_large_number_l373_373746


namespace max_difference_primes_l373_373229

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even_integer : ℕ := 138

theorem max_difference_primes (p q : ℕ) :
  is_prime p ∧ is_prime q ∧ p + q = even_integer ∧ p ≠ q →
  (q - p) = 124 :=
by
  sorry

end max_difference_primes_l373_373229


namespace find_p_max_area_triangle_l373_373516

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373516


namespace geometric_sequence_general_formula_arithmetic_sequence_bn_l373_373037

theorem geometric_sequence_general_formula (a : ℕ → ℝ) (A : a 1 * a 2 = a 3) (B : 27 * a 2 - a 5 = 0) :
  ∀ n, a n = 3 ^ n :=
by { sorry }

theorem arithmetic_sequence_bn (a : ℕ → ℝ) (b : ℕ → ℝ) (A : ∀ n, a n = 3 ^ n)
  (B : b = λ n, 3 * real.log 3 (a n) + 3) :
  ∃ d, ∀ n, b (n + 1) - b n = d :=
by { sorry }

end geometric_sequence_general_formula_arithmetic_sequence_bn_l373_373037


namespace integral_indefinite_integral_definite_l373_373291

noncomputable def indefinite_integral (x : ℝ) : ℝ := 
  - (1 / 5) * (Real.exp (-x)) * (Real.sin (2 * x) + 2 * Real.cos (2 * x))

theorem integral_indefinite : 
  ∫ (x : ℝ) in Set.univ, Real.exp (-x) * Real.sin (2 * x) = indefinite_integral x + C := 
sorry

theorem integral_definite : 
  ∫ (x : ℝ) in (0 : ℝ)..π, Real.exp (-x) * Real.abs (Real.sin (2 * x)) = (2 / 5) * (1 + Real.exp (-π)) :=
sorry

end integral_indefinite_integral_definite_l373_373291


namespace planes_parallel_if_perpendicular_to_common_line_l373_373689

/-- If two planes are both perpendicular to a common line, then the two planes are parallel. -/
theorem planes_parallel_if_perpendicular_to_common_line
  (P1 P2 : Plane) (l : Line) 
  (h1 : P1.perpendicular_to l) (h2 : P2.perpendicular_to l) :
  P1.parallel_to P2 :=
sorry

end planes_parallel_if_perpendicular_to_common_line_l373_373689


namespace find_largest_number_l373_373020

theorem find_largest_number 
  (a b c : ℕ) 
  (h1 : a + b = 16) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) : 
  c = 19 := 
sorry

end find_largest_number_l373_373020


namespace find_a_plus_b_l373_373986

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem find_a_plus_b (a b : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (1 + i) * (2 + i) = a + b * i) : a + b = 4 :=
by sorry

end find_a_plus_b_l373_373986


namespace max_radius_of_inscribable_circle_l373_373850

theorem max_radius_of_inscribable_circle
  (AB BC CD DA : ℝ) (x y z w : ℝ)
  (h1 : AB = 10) (h2 : BC = 12) (h3 : CD = 8) (h4 : DA = 14)
  (h5 : x + y = 10) (h6 : y + z = 12)
  (h7 : z + w = 8) (h8 : w + x = 14)
  (h9 : x + z = y + w) :
  ∃ r : ℝ, r = Real.sqrt 24.75 :=
by
  sorry

end max_radius_of_inscribable_circle_l373_373850


namespace tokens_distribution_l373_373170

theorem tokens_distribution (n : ℕ) (h : n ≥ 1) :
  (∃ steps : ℕ, ∀ i, tokens_after_steps i steps = 1) ↔ n % 2 = 1 :=
sorry

end tokens_distribution_l373_373170


namespace juniors_in_sports_count_l373_373235

-- Definitions for given conditions
def total_students : ℕ := 500
def percent_juniors : ℝ := 0.40
def percent_juniors_in_sports : ℝ := 0.70

-- Definition to calculate the number of juniors
def number_juniors : ℕ := (percent_juniors * total_students : ℝ).toNat

-- Definition to calculate the number of juniors involved in sports
def number_juniors_in_sports : ℕ := (percent_juniors_in_sports * number_juniors : ℝ).toNat

-- Statement to prove the calculated number of juniors involved in sports
theorem juniors_in_sports_count : number_juniors_in_sports = 140 :=
sorry

end juniors_in_sports_count_l373_373235


namespace complement_intersection_l373_373563

open Set

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}

noncomputable def N : Set ℝ := {y | ∃ x ∈ M, y = 2^x}

theorem complement_intersection :
  compl ((λ x, 2^x) '' {x | -1 < x ∧ x < 1}) = {y | y ≤ 1/2 ∨ 1 ≤ y} :=
by
  sorry

end complement_intersection_l373_373563


namespace find_biology_marks_l373_373368

theorem find_biology_marks (english math physics chemistry : ℕ) (avg_marks : ℕ) (biology : ℕ)
  (h_english : english = 86) (h_math : math = 89) (h_physics : physics = 82)
  (h_chemistry : chemistry = 87) (h_avg_marks : avg_marks = 85) :
  (english + math + physics + chemistry + biology) = avg_marks * 5 →
  biology = 81 :=
by
  sorry

end find_biology_marks_l373_373368


namespace last_non_zero_digit_of_40_l373_373882

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def last_non_zero_digit (n : ℕ) : ℕ :=
  let p := factorial n
  let digits : List ℕ := List.filter (λ d => d ≠ 0) (p.digits 10)
  digits.headD 0

theorem last_non_zero_digit_of_40 : last_non_zero_digit 40 = 6 := by
  sorry

end last_non_zero_digit_of_40_l373_373882


namespace range_a_for_inequality_l373_373897

theorem range_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (a-2) * x^2 - 2 * (a-2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
by
  sorry

end range_a_for_inequality_l373_373897


namespace simplify_fraction_l373_373679

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
  ((Real.sqrt 3) + 2 * (Real.sqrt 5) - 1) / (2 + 4 * Real.sqrt 5) := 
by 
  sorry

end simplify_fraction_l373_373679


namespace chessboard_equal_area_l373_373721

theorem chessboard_equal_area (closed_polygon : set (ℕ × ℕ))
  (h_closed: ∀ x ∈ closed_polygon, ∃ y ∈ closed_polygon, adj x y)
  (adj: (ℕ × ℕ) → (ℕ × ℕ) → Prop)
  (h_adj_def: ∀ {x y : ℕ × ℕ}, adj x y ↔ (x.1 - y.1).nat_abs ≤ 1 ∧ (x.2 - y.2).nat_abs ≤ 1) :
  ∑ x in closed_polygon, area_black x = ∑ x in closed_polygon, area_white x :=
sorry

end chessboard_equal_area_l373_373721


namespace find_p_max_area_triangle_l373_373538

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373538


namespace shortest_chord_line_l373_373044

theorem shortest_chord_line (x y : ℝ) (P : (ℝ × ℝ)) (C : ℝ → ℝ → Prop) (h₁ : C x y) (hx : P = (1, 1)) (hC : ∀ x y, C x y ↔ x^2 + y^2 = 4) : 
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -2 ∧ a * x + b * y + c = 0 :=
by
  sorry

end shortest_chord_line_l373_373044


namespace cos_double_angle_l373_373032

theorem cos_double_angle 
  (α : ℝ) 
  (h : 3 * sin (α - π / 6) = sin (α + π / 6)) : 
  cos (2 * α) = -1 / 7 :=
by 
  sorry

end cos_double_angle_l373_373032


namespace persimmons_count_l373_373984

theorem persimmons_count (x : ℕ) (h : x - 5 = 12) : x = 17 :=
by
  sorry

end persimmons_count_l373_373984


namespace smallest_value_of_expressions_l373_373337

theorem smallest_value_of_expressions :
  let A := sin (50 * Real.pi / 180) * cos (39 * Real.pi / 180) - sin (40 * Real.pi / 180) * cos (51 * Real.pi / 180)
  let B := -2 * (sin (40 * Real.pi / 180))^2 + 1
  let C := 2 * sin (6 * Real.pi / 180) * cos (6 * Real.pi / 180)
  let D := (sqrt 3 / 2) * sin (43 * Real.pi / 180) - (1 / 2) * cos (43 * Real.pi / 180)
  A = sin (11 * Real.pi / 180) → 
  B = sin (10 * Real.pi / 180) →
  C = sin (12 * Real.pi / 180) →
  D = sin (13 * Real.pi / 180) →
  min (min A B) (min C D) = B := 
by
  sorry

end smallest_value_of_expressions_l373_373337


namespace matrix_inverse_l373_373640

variable (N : Matrix (Fin 2) (Fin 2) ℚ) 
variable (I : Matrix (Fin 2) (Fin 2) ℚ)
variable (c d : ℚ)

def M1 : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

def M2 : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem matrix_inverse (hN : N = M1) 
                       (hI : I = M2) 
                       (hc : c = 1/12) 
                       (hd : d = 1/12) :
                       N⁻¹ = c • N + d • I := by
  sorry

end matrix_inverse_l373_373640


namespace product_of_roots_l373_373361

theorem product_of_roots (r1 r2 r3 : ℝ) : 
  (∀ x : ℝ, 2 * x^3 - 24 * x^2 + 96 * x + 56 = 0 → x = r1 ∨ x = r2 ∨ x = r3) →
  r1 * r2 * r3 = -28 :=
by
  sorry

end product_of_roots_l373_373361


namespace statements_evaluation_l373_373815

-- Define the statements A, B, C, D, E as propositions
def A : Prop := ∀ (A B C D E : Prop), (A → ¬B ∧ ¬C ∧ ¬D ∧ ¬E)
def B : Prop := sorry  -- Assume we have some way to read the statement B under special conditions
def C : Prop := ∀ (A B C D E : Prop), (A ∧ B ∧ C ∧ D ∧ E)
def D : Prop := sorry  -- Assume we have some way to read the statement D under special conditions
def E : Prop := A

-- Prove the conditions
theorem statements_evaluation : ¬ A ∧ ¬ C ∧ ¬ E ∧ B ∧ D :=
by
  sorry

end statements_evaluation_l373_373815


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373501

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373501


namespace find_p_max_area_triangle_l373_373527

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373527


namespace min_distance_eq_3_l373_373956

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (Real.pi / 3 * x + Real.pi / 4)

theorem min_distance_eq_3 (x₁ x₂ : ℝ) 
  (h₁ : f x₁ ≤ f x) (h₂ : f x ≤ f x₂) 
  (x : ℝ) :
  |x₁ - x₂| = 3 :=
by
  -- Sorry placeholder for proof.
  sorry

end min_distance_eq_3_l373_373956


namespace train_speed_in_km_per_hr_l373_373811

-- Define the conditions
def length_of_train : ℝ := 100 -- length in meters
def time_to_cross_pole : ℝ := 6 -- time in seconds

-- Define the conversion factor from meters/second to kilometers/hour
def conversion_factor : ℝ := 18 / 5

-- Define the formula for speed calculation
def speed_of_train := (length_of_train / time_to_cross_pole) * conversion_factor

-- The theorem to be proven
theorem train_speed_in_km_per_hr : speed_of_train = 50 := by
  sorry

end train_speed_in_km_per_hr_l373_373811


namespace translation_vector_l373_373585

open Real

noncomputable def original_function (x : ℝ) : ℝ := log (x - 2) / log 2 + 3
noncomputable def new_function (x : ℝ) : ℝ := log (x + 1) / log 2 - 1

theorem translation_vector :
  ∃ a b : ℝ, (∀ x : ℝ, original_function (x + a) = new_function x + b) ∧ (a, b) = (-3, -4) := 
by
  use -3, -4
  intro x
  sorry

end translation_vector_l373_373585


namespace parabola_circle_distance_l373_373471

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373471


namespace induction_transition_factor_l373_373247

theorem induction_transition_factor (k : ℕ) (h : k > 0) :
  let lhs_k := (list.range k).map (λ i, k + 1 + i).prod
  let lhs_k1 := (list.range (k+1)).map (λ i, k + 2 + i).prod
  lhs_k1 = (lhs_k * (2*k + 1) * (2*k + 2)) / (k + 1) :=
by
  sorry

end induction_transition_factor_l373_373247


namespace least_number_of_square_tiles_l373_373275

-- Definitions based on conditions
def room_length_cm : ℕ := 672
def room_width_cm : ℕ := 432

-- Correct Answer is 126 tiles

-- Lean Statement for the proof problem
theorem least_number_of_square_tiles : 
  ∃ tile_size tiles_needed, 
    (tile_size = Int.gcd room_length_cm room_width_cm) ∧
    (tiles_needed = (room_length_cm / tile_size) * (room_width_cm / tile_size)) ∧
    tiles_needed = 126 := 
by
  sorry

end least_number_of_square_tiles_l373_373275


namespace find_value_of_2a_minus_b_l373_373967

def A : Set ℝ := {x | x < 1 ∨ x > 5}
def B (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}

theorem find_value_of_2a_minus_b (a b : ℝ) (h1 : A ∪ B a b = Set.univ) (h2 : A ∩ B a b = {x | 5 < x ∧ x ≤ 6}) : 2 * a - b = -4 :=
by
  sorry

end find_value_of_2a_minus_b_l373_373967


namespace exists_subset_sum_square_l373_373034

theorem exists_subset_sum_square (S : Finset ℕ) (hS_card : S.card = 50) (hS_max : ∀ x ∈ S, x ≤ 100) :
  ∃ T ⊆ S, is_square (T.sum) :=
by
  sorry

end exists_subset_sum_square_l373_373034


namespace roots_eq_solution_l373_373163

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end roots_eq_solution_l373_373163


namespace only_solution_2_pow_eq_y_sq_plus_y_plus_1_l373_373387

theorem only_solution_2_pow_eq_y_sq_plus_y_plus_1 {x y : ℕ} (h1 : 2^x = y^2 + y + 1) : x = 0 ∧ y = 0 := 
by {
  sorry -- proof goes here
}

end only_solution_2_pow_eq_y_sq_plus_y_plus_1_l373_373387


namespace Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l373_373028

def Q (x : ℂ) (n : ℕ) : ℂ := (x + 1)^n + x^n + 1
def P (x : ℂ) : ℂ := x^2 + x + 1

-- Part a) Q(x) is divisible by P(x) if and only if n ≡ 2 (mod 6) or n ≡ 4 (mod 6)
theorem Q_divisible_by_P (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 2 ∨ n % 6 = 4) := sorry

-- Part b) Q(x) is divisible by P(x)^2 if and only if n ≡ 4 (mod 6)
theorem Q_divisible_by_P_squared (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 4 := sorry

-- Part c) Q(x) is never divisible by P(x)^3
theorem Q_not_divisible_by_P_cubed (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^3 ≠ 0 := sorry

end Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l373_373028


namespace largest_integer_n_l373_373394

theorem largest_integer_n (n : ℕ) :
  (∑ r in Finset.range (n + 1), r * (Nat.choose n r)) < 500 ↔ n = 7 :=
by sorry

end largest_integer_n_l373_373394


namespace remainder_of_large_number_l373_373749

theorem remainder_of_large_number (n : ℕ) (r : ℕ) (h : n = 2468135792) :
  (n % 101) = 52 := 
by
  have h1 : (10 ^ 8 - 1) % 101 = 0 := sorry
  have h2 : (10 ^ 6 - 1) % 101 = 0 := sorry
  have h3 : (10 ^ 4 - 1) % 101 = 0 := sorry
  have h4 : (10 ^ 2 - 1) % 101 = 99 % 101 := sorry

  -- Using these properties to simplify n
  have n_decomposition : 2468135792 = 24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 := sorry
  have div_property : 
    (24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 - (24 + 68 + 13 + 57 + 92)) % 101 = 0 := sorry

  have simplified_sum : (24 + 68 + 13 + 57 + 92 = 254 := by norm_num) := sorry
  have resulting_mod : 254 % 101 = 52 := by norm_num

  -- Thus n % 101 = 52
  exact resulting_mod

end remainder_of_large_number_l373_373749


namespace limit_value_l373_373935

variable {f : ℝ → ℝ} {x0 : ℝ}

theorem limit_value (h : HasDerivAt f 3 x0) :
  tendsto (fun (Δx : ℝ) => (f (x0 + 2 * Δx) - f x0) / (3 * Δx)) (𝓝 0) (𝓝 2) :=
sorry

end limit_value_l373_373935


namespace number_is_composite_l373_373197

theorem number_is_composite : ∃ k l : ℕ, k * l = 53 * 83 * 109 + 40 * 66 * 96 ∧ k > 1 ∧ l > 1 :=
by
  have h1 : 53 + 96 = 149 := by norm_num
  have h2 : 83 + 66 = 149 := by norm_num
  have h3 : 109 + 40 = 149 := by norm_num
  sorry

end number_is_composite_l373_373197


namespace inequality_hold_l373_373195

-- Definitions for sequences a and b with given conditions
variables {n : ℕ} (a b : Fin n → ℝ)

-- Conditions for sequences (strictly increasing and positive)
def increasing_seq (a : Fin n → ℝ) :=
  ∀ i j : Fin n, i < j → a i < a j

def positive_seq (a : Fin n → ℝ) :=
  ∀ i : Fin n, 0 < a i

-- The main statement
theorem inequality_hold
  (ha_increasing : increasing_seq a)
  (hb_increasing : increasing_seq b)
  (ha_positive : positive_seq a)
  (hb_positive : positive_seq b) :
  n * (∑ i, a i * b i) > (∑ i, a i) * (∑ i, b i) :=
sorry

end inequality_hold_l373_373195


namespace tom_mileage_per_gallon_l373_373725

-- Definitions based on the given conditions
def daily_mileage : ℕ := 75
def cost_per_gallon : ℕ := 3
def amount_spent_in_10_days : ℕ := 45
def days : ℕ := 10

-- Main theorem to prove
theorem tom_mileage_per_gallon : 
  (amount_spent_in_10_days / cost_per_gallon) * 75 * days = 50 :=
by
  sorry

end tom_mileage_per_gallon_l373_373725


namespace greatest_N_l373_373371

noncomputable def greatest_constant (a b c : ℝ) (h : a + b > c) (h' : a + c > b) (h'' : b + c > a) : ℝ :=
  real.sup {N | ∀ (a b c : ℝ), a + b > c → a + c > b → b + c > a → (a^2 + c^2) / b^2 > N}

theorem greatest_N : 
  greatest_constant a b c (by linarith) (by linarith) (by linarith) = 1 := sorry

end greatest_N_l373_373371


namespace problem_proof_l373_373435

theorem problem_proof (p q : Prop) 
  (hp : ∀ x : ℝ, 2^x < 3^x → False) 
  (hq : ∃ x : ℝ, x^3 = 1 - x^2) : ¬p ∧ q :=
by
  have hnp : ¬p :=
    fun hp_true => 
      hp 0 (by linarith)
  have hq_true : q := hq
  exact ⟨hnp, hq_true⟩

end problem_proof_l373_373435


namespace red_ball_probability_l373_373241

variable {k : ℕ+} -- positive integers for bins

-- Given conditions
def probability_of_bin (k : ℕ+) : ℝ := 1 / (2^k : ℝ)

-- Main theorem to be proven
theorem red_ball_probability :
  (∑' k : ℕ+, probability_of_bin k) = 2 / 7 := 
sorry

end red_ball_probability_l373_373241


namespace range_of_m_l373_373118

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * 4^x - 3 * 2^(x + 1) - 2

theorem range_of_m (m : ℝ) : (∃ x : ℝ, f m x = 0) → m > 0 :=
begin
  sorry
end

end range_of_m_l373_373118


namespace sum_seq_formula_l373_373920

open Nat

def seq (n : ℕ) : ℕ :=
  Nat.recOn n 2 (λ _ a_n, 4 * a_n - 3 * n + 1)

noncomputable def sum_seq (n : ℕ) : ℕ :=
  (range n).sum (seq ∘ (λ i, i + 1))

theorem sum_seq_formula (n : ℕ) : 
  sum_seq n = (range n).sum (λ i, binomial n i * 3^(n - i) + i + 1) := 
by
  sorry

end sum_seq_formula_l373_373920


namespace find_time_interval_l373_373602

-- Definitions for conditions
def birthRate : ℕ := 4
def deathRate : ℕ := 2
def netIncreaseInPopulationPerInterval (T : ℕ) : ℕ := birthRate - deathRate
def totalTimeInOneDay : ℕ := 86400
def netIncreaseInOneDay (T : ℕ) : ℕ := (totalTimeInOneDay / T) * (netIncreaseInPopulationPerInterval T)

-- Theorem statement
theorem find_time_interval (T : ℕ) (h1 : netIncreaseInPopulationPerInterval T = 2) (h2 : netIncreaseInOneDay T = 86400) : T = 2 :=
sorry

end find_time_interval_l373_373602


namespace exists_99_consecutive_divisible_numbers_l373_373859

theorem exists_99_consecutive_divisible_numbers :
  ∃ (a : ℕ → ℕ), 
    (∀ n, 0 < n → n ≤ 99 → a (n - 1) = n! - n) ∧
    ((∀ m, 1 ≤ m ∧ m ≤ 99 → (m! - m) % (100 - m + 1) = 0) := 
begin
  -- This is the structure for the theorem
  -- Proof is omitted
  sorry
end

end exists_99_consecutive_divisible_numbers_l373_373859


namespace find_p_max_area_triangle_l373_373523

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373523


namespace first_number_less_than_twice_second_l373_373230

theorem first_number_less_than_twice_second : 
  ∀ (x y : ℕ), x + y = 50 ∧ y = 31 → 2 * y - x = 43 :=
by
  intros x y h,
  cases h with h_sum h_y,
  have h1: y = 31 := h_y,
  have h2: x + y = 50 := h_sum,
  rw h1 at h2,
  have h3: x = 50 - 31 := by linarith,
  rw [h1, h3],
  linarith,

end first_number_less_than_twice_second_l373_373230


namespace train_speed_excluding_stoppages_l373_373013

noncomputable def speed_excluding_stoppages (speed_including_stops : ℝ) (stoppage_time_min : ℝ) : ℝ :=
  let stoppage_time_hr := stoppage_time_min / 60
  let moving_time_per_hour := 1 - stoppage_time_hr
  speed_including_stops / moving_time_per_hour

theorem train_speed_excluding_stoppages:
  speed_excluding_stoppages 27 21.428571428571423 ≈ 42 :=
by
  sorry

end train_speed_excluding_stoppages_l373_373013


namespace triangulation_count_eq_catalan_l373_373107

noncomputable def catalan (n : ℕ) : ℕ :=
  if n = 0 then 1 else (2 * (2 * n - 1) * catalan (n - 1)) / (n + 1)

theorem triangulation_count_eq_catalan (n : ℕ) : 
  let C_n_minus_1 := catalan (n - 1) in
  ∃ (k : ℕ), k = C_n_minus_1 ∧ k = (number of ways to triangulate an (n + 2)-gon) :=
by sorry

end triangulation_count_eq_catalan_l373_373107


namespace exists_sphere_containing_all_projections_l373_373425

-- Define the problem conditions in Lean 4
variables (O : Point) (n : ℕ) (lines : Fin n → Line)
-- Assume the lines are pairwise non-parallel
axiom non_parallel_lines : ∀ i j : Fin n, i ≠ j → ¬ parallel (lines i) (lines j)

-- The theorem statement: There exists a sphere containing all points obtained
-- by the specified iterative orthogonal projections
theorem exists_sphere_containing_all_projections (hO : Point) (l: Fin n → Line) : 
  ∃ R: ℝ, ∀ P : Point, (P ∈ generated_points hO l) → (distance hO P ≤ R) :=
  sorry -- proof not required

end exists_sphere_containing_all_projections_l373_373425


namespace least_number_to_add_l373_373768

theorem least_number_to_add (n : ℕ) (h₁ : n = 1054) :
  ∃ k : ℕ, (n + k) % 23 = 0 ∧ k = 4 :=
by
  use 4
  have h₂ : n % 23 = 19 := by sorry
  have h₃ : (n + 4) % 23 = 0 := by sorry
  exact ⟨h₃, rfl⟩

end least_number_to_add_l373_373768


namespace magnitude_of_vector_l373_373080

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)

-- Conditions
def dot_product_zero : Prop := inner_product_space.inner a b = 0
def norm_a : Prop := ∥a∥ = 2
def norm_b : Prop := ∥b∥ = 3

theorem magnitude_of_vector (h1 : dot_product_zero a b) (h2 : norm_a a) (h3 : norm_b b) : 
  ∥(3 : ℝ) • a - (2 : ℝ) • b∥ = 6 * real.sqrt 2 :=
by
  sorry

end magnitude_of_vector_l373_373080


namespace standing_arrangements_l373_373790

-- Definitions for conditions in a)
def students := {A, B, C, D, E, F}
def males := {A, B, C}
def females := {D, E, F}

def adjacent (x y : students) (ll : List students) : Prop :=
  ∃ n, ll.nth n = some x ∧ ll.nth (n+1) = some y ∨ ll.nth n = some y ∧ ll.nth (n+1) = some x

def different_genders (x y : students) : Prop :=
  (x ∈ males ∧ y ∈ females) ∨ (x ∈ females ∧ y ∈ males)

def not_at_ends (x : students) (ll : List students) : Prop :=
  ¬(ll.head? = some x ∨ ll.last? = some x)

-- Given conditions in the problem
def conditions (ll : List students) : Prop :=
  ll.length = 6 ∧
  ∀ (i : Nat), i < 5 → different_genders (ll.nth_le i (by linarith)) (ll.nth_le (i+1) (by linarith)) ∧
  adjacent A B ll ∧
  not_at_ends A ll ∧
  not_at_ends B ll

-- Proof statement
theorem standing_arrangements : ∃ (ll : List students), conditions ll ∧ 
                                (List.permutations students).countp conditions = 24 := 
by
  sorry

end standing_arrangements_l373_373790


namespace matrix_inequality_l373_373136

theorem matrix_inequality (n : ℕ) (a : Fin n → ℕ) 
  (h1 : ∀ (i : Fin n), a i ∈ {(i.val * n + 1)..(i.val * n + n)})
  (h2 : Function.Injective a) : 
  ∑ i in Finset.range n, (i + 1) ^ 2 / a ⟨i, sorry⟩ ≥ (n + 2) / 2 - 1 / (n ^ 2 + 1) := 
sorry

end matrix_inequality_l373_373136


namespace find_BC_l373_373122

variables {A B C X : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace X]
variables (AB AC BC BX CX : ℝ)

-- Given conditions
def in_triangle := (AB = 72) ∧ (AC = 84)
def circle_intersects_BC := (BX = CX)
def intersection_B_to_X := (AB = 72) -- Radius of circle centered at A

theorem find_BC (h1 : in_triangle) (h2 : circle_intersects_BC) (h3 : intersection_B_to_X) : 
  BC = 60 :=
sorry

end find_BC_l373_373122


namespace correct_statements_l373_373053

open Real

-- Function f : ℝ → ℝ satisfying the given conditions
def f : ℝ → ℝ := sorry

-- Condition 1: f is an even function
axiom f_even : ∀ x : ℝ, f(x) = f(-x)

-- Condition 2: For x ≥ 0, f(x+1) = -f(x)
axiom f_shift : ∀ x : ℝ, x ≥ 0 → f(x + 1) = -f(x)

-- Condition 3: For x in [0,1), f(x) = log2(x + 1)
axiom f_base : ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f(x) = log2 (x + 1)

theorem correct_statements :
  (f(2016) + f(-2017) = 0) ∧ (∀ y, y ∈ Set.range f ↔ -1 < y ∧ y < 1) :=
by
  sorry

end correct_statements_l373_373053


namespace sqrt_50_value_l373_373180

def f (x : ℝ) : ℝ :=
  if x ∈ Int then 7 * x + 3 else Real.floor x + 6

theorem sqrt_50_value : f (Real.sqrt 50) = 13 :=
by
  sorry

end sqrt_50_value_l373_373180


namespace derivative_at_point_l373_373063

variable {f : ℝ → ℝ}

-- The tangent line at (1, f(1)) is 2x - y + 2 = 0
def tangent_line (x y : ℝ) := 2 * x - y + 2 = 0

theorem derivative_at_point :
  tangent_line 1 (f 1) →
  (∀ x, DifferentiableAt ℝ f x) →
  Deriv f 1 = 2 :=
by
  intros
  sorry

end derivative_at_point_l373_373063


namespace smallest_r_minus_p_l373_373719

theorem smallest_r_minus_p (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (prod_eq : p * q * r = nat.factorial 9) (h_lt : p < q ∧ q < r) :
  r - p = 219 := 
sorry

end smallest_r_minus_p_l373_373719


namespace inner_circumference_correct_l373_373789

def outer_radius : ℝ := 84.02817496043394
def track_width : ℝ := 14
def inner_radius : ℝ := outer_radius - track_width
def inner_circumference_approx : ℝ := 2 * Real.pi * inner_radius

theorem inner_circumference_correct :
  inner_circumference_approx ≈ 439.82 :=
begin
  sorry
end

end inner_circumference_correct_l373_373789


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373494

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373494


namespace problem_statement_l373_373369

def S (a b : ℤ) : ℤ := 4 * a + 6 * b
def T (a b : ℤ) : ℤ := 2 * a - 3 * b

theorem problem_statement : T (S 8 3) 4 = 88 := by
  sorry

end problem_statement_l373_373369


namespace expression_simplification_l373_373288

theorem expression_simplification (a b : ℤ) : 
  2 * (2 * a - 3 * b) - 3 * (2 * b - 3 * a) = 13 * a - 12 * b :=
by
  sorry

end expression_simplification_l373_373288


namespace sin_d_eq_8_over_9_l373_373582

variable (u v : ℝ)
variable (area : ℝ := 100)
variable (geo_mean : ℝ := 15)

theorem sin_d_eq_8_over_9 :
  (1 / 2 * u * v * Real.sin (Real.pi / 2) = area) →
  (√(u * v) = geo_mean) →
  Real.sin (Real.pi / 2) = 8 / 9 := by
  sorry

end sin_d_eq_8_over_9_l373_373582


namespace max_sequence_length_l373_373868

def valid_sequence (s : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, 0 ≤ n ∧ n < 7 → ∑ i in (Finset.range 7), s (n + i) > 0) ∧
  (∀ n : ℕ, 0 ≤ n ∧ n < 11 → ∑ i in (Finset.range 11), s (n + i) < 0)

theorem max_sequence_length {s : ℕ → ℤ} (h : valid_sequence s) : 
  ∃ (n : ℕ), n = 16 ∧ (¬ ∃ m > 16, valid_sequence (λ i, s (i % m))) :=
sorry

end max_sequence_length_l373_373868


namespace f_neg_five_value_l373_373954

noncomputable def f : ℝ → ℝ
| x if x > 0     := log x / log 2 + 2
| x := 2 * f (x + 3)

theorem f_neg_five_value : f (-5) = 8 := by
  sorry

end f_neg_five_value_l373_373954


namespace find_value_l373_373929

theorem find_value (a b : ℝ) (h1 : 2^a = real.sqrt 10) (h2 : 5^b = real.sqrt 10) : 
  (1 / a) + (1 / b) = 2 :=
sorry

end find_value_l373_373929


namespace math_courses_schedule_ways_l373_373804

def num_ways_to_schedule_courses 
  (math_courses : ℕ) (total_periods : ℕ) (literature_periods : ℕ) 
  (math_periods : ℕ) (no_consec_math : Prop) (lit_position_restriction : Prop) : ℕ :=
  if no_consec_math ∧ lit_position_restriction then 16 else 0

theorem math_courses_schedule_ways : ∀ (math_courses total_periods literature_periods math_periods : ℕ)
    (no_consec_math lit_position_restriction : Prop),
    math_courses = 3 →
    total_periods = 7 →
    literature_periods = 1 →
    math_periods = total_periods - literature_periods →
    no_consec_math →
    lit_position_restriction →
    num_ways_to_schedule_courses math_courses total_periods literature_periods math_periods no_consec_math lit_position_restriction = 16 :=
by
  assume (math_courses total_periods literature_periods math_periods : ℕ) (no_consec_math lit_position_restriction : Prop)
  assume hc : math_courses = 3
  assume hp : total_periods = 7
  assume hl : literature_periods = 1
  assume hm : math_periods = total_periods - literature_periods
  assume hnc : no_consec_math
  assume hlr : lit_position_restriction
  exact sorry

end math_courses_schedule_ways_l373_373804


namespace temperature_rise_l373_373713

variable (t : ℝ)

theorem temperature_rise (initial final : ℝ) (h : final = t) : final = 5 + t := by
  sorry

end temperature_rise_l373_373713


namespace John_cycles_distance_l373_373626

-- Define the rate and time as per the conditions in the problem
def rate : ℝ := 8 -- miles per hour
def time : ℝ := 2.25 -- hours

-- The mathematical statement to prove: distance = rate * time
theorem John_cycles_distance : rate * time = 18 := by
  sorry

end John_cycles_distance_l373_373626


namespace digit_5_count_in_300_to_699_l373_373108

theorem digit_5_count_in_300_to_699 :
  (count (λ n, 300 ≤ n ∧ n < 700 ∧ (∃ d ∈ [n % 10, (n / 10) % 10, (n / 100)], d = 5)) (list.range' 300 (700 - 300))) = 157 := 
sorry

end digit_5_count_in_300_to_699_l373_373108


namespace parallel_vectors_implies_xz_eq_nine_l373_373567

-- Given the context of the problem
variables {x z λ : ℝ}
variables (a b : ℝ × ℝ × ℝ)
def a := (x, 4, 3)
def b := (3, 2, z)

-- Assuming the vectors are parallel
axiom parallel_vectors : ∃ λ : ℝ, a = (λ * 3, λ * 2, λ * z)

theorem parallel_vectors_implies_xz_eq_nine
  (h1 : 4 = 2 * λ)
  (h2 : 3 = λ * z)
  (h3 : x = 3 * λ) :
  x * z = 9 :=
sorry

end parallel_vectors_implies_xz_eq_nine_l373_373567


namespace function_range_sin_cos_l373_373112

theorem function_range_sin_cos (x : ℝ) (hx1 : 0 < x) (hx2 : x ≤ π / 3) :
  let y := (sin x * cos x + 1) / (sin x + cos x) in 1 < y ∧ y ≤ 3 * sqrt 2 / 4 :=
by
  sorry

end function_range_sin_cos_l373_373112


namespace calculate_expression_l373_373831

theorem calculate_expression :
  (3^2 - 4 + 6^2 - 1) ^ (-2) * 7 = 7 / 1600 :=
by
  sorry

end calculate_expression_l373_373831


namespace proof_part1_proof_part2_l373_373475

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373475


namespace total_cutting_schemes_l373_373722

theorem total_cutting_schemes (length_wire : ℕ) (length_10cm : ℕ) (length_20cm : ℕ) (n : ℕ) : 
  length_wire = 150 → length_10cm = 10 → length_20cm = 20 → 
  (∃ x y : ℕ, x + 2 * y = n ∧ n = 15 ∧ x > 0 ∧ y > 0) → 
  (∃ y : ℕ, 1 ≤ y ∧ y ≤ 7 ∧ n = 15 ∧ 7 = length y) :=
by sorry

end total_cutting_schemes_l373_373722


namespace proof_part1_proof_part2_l373_373472

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373472


namespace strictly_increasing_b_range_l373_373960

theorem strictly_increasing_b_range (b : ℝ) :
  (∀ x : ℝ, (1 / 3) * x^3 + b * x^2 + (b + 2) * x + 3) →
  (∀ x : ℝ, x^2 + 2 * b * x + b + 2 ≥ 0) →
  (-1 ≤ b ∧ b ≤ 2) :=
sorry

end strictly_increasing_b_range_l373_373960


namespace find_x_l373_373277
-- Import the broad Mathlib module to bring in necessary functions and properties

-- Define the problem statement as a theorem in Lean
theorem find_x (x : ℤ) (h : x + 1 = 3) : x = 2 :=
begin
  -- Provide a placeholder for the proof part
  sorry
end

end find_x_l373_373277


namespace sum_terms_2012_l373_373431

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- Placeholder for the arithmetic sequence function
def S_n (n : ℕ) : ℝ := (n * (a_n 1 + a_n n)) / 2

theorem sum_terms_2012 (h1 : a_n 4 + a_n 2009 = 1) : S_n 2012 = 1006 :=
by
  sorry

end sum_terms_2012_l373_373431


namespace line_MN_pass_through_fixed_point_maximized_area_of_triangle_FMN_l373_373452

open Real

def ellipse (x y : ℝ) := x^2 / 5 + y^2 / 4 = 1
def right_focus : ℝ × ℝ := (1, 0)
def fixed_point : ℝ × ℝ := (5 / 9, 0)

theorem line_MN_pass_through_fixed_point 
  (A B C D M N : ℝ × ℝ) (n : ℝ)
  (hA : ellipse A.1 A.2)
  (hB : ellipse B.1 B.2)
  (hC : ellipse C.1 C.2)
  (hD : ellipse D.1 D.2)
  (hM : M = midpoint A B)
  (hN : N = midpoint C D)
  (h_perpendicular : (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0) : 
  line_through M N fixed_point := sorry

theorem maximized_area_of_triangle_FMN 
  (A B C D M N : ℝ × ℝ) (n : ℝ)
  (hA : ellipse A.1 A.2)
  (hB : ellipse B.1 B.2)
  (hC : ellipse C.1 C.2)
  (hD : ellipse D.1 D.2)
  (hM : M = midpoint A B)
  (hN : N = midpoint C D)
  (h_perpendicular : (B.1 - A.1) * (D.1 - C.1) + (B.2 - A.2) * (D.2 - C.2) = 0) :
  triangle_area right_focus M N ≤ 16 / 81 := sorry

end line_MN_pass_through_fixed_point_maximized_area_of_triangle_FMN_l373_373452


namespace max_gcd_2015xy_l373_373397

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end max_gcd_2015xy_l373_373397


namespace problem_l373_373156

-- Conditions
def a_n (n : ℕ) : ℚ := (1/3)^(n-1)

def b_n (n : ℕ) : ℚ := n * (1/3)^n

-- Sums over the first n terms
def S_n (n : ℕ) : ℚ := (3/2) - (1/2) * (1/3)^n

def T_n (n : ℕ) : ℚ := (3/4) - (1/4) * (1/3)^n - (n/2) * (1/3)^n

-- Problem: Prove T_n < S_n / 2
theorem problem (n : ℕ) : T_n n < S_n n / 2 :=
by sorry

end problem_l373_373156


namespace complement_U_A_l373_373564

def U : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def A : Set ℝ := {x | 3 ≤ 2 * x - 1 ∧ 2 * x - 1 < 5}

theorem complement_U_A : (U \ A) = {x | (0 ≤ x ∧ x < 2) ∨ (3 ≤ x)} := sorry

end complement_U_A_l373_373564


namespace solve_equation_l373_373682

theorem solve_equation (x : ℂ) :
  (x^3 + 3 * x^2 * complex.sqrt 3 + 9 * x + 3 * complex.sqrt 3) + (x + complex.sqrt 3) = 0 ↔ 
    (x = -complex.sqrt 3 ∨ x = -complex.sqrt 3 + complex.I ∨ x = -complex.sqrt 3 - complex.I) :=
by sorry

end solve_equation_l373_373682


namespace expressible_polynomial_iff_odd_l373_373151

theorem expressible_polynomial_iff_odd (n : ℕ) (hn : n > 0) :
  (∃ p : polynomial ℝ, ∀ x : ℝ, x^n - (1 / x)^n = p.eval (x - 1/x)) ↔ odd n :=
by
  sorry

end expressible_polynomial_iff_odd_l373_373151


namespace distance_of_hyperbola_vertices_l373_373391

-- Define the hyperbola equation condition
def hyperbola : Prop := ∃ (y x : ℝ), (y^2 / 16) - (x^2 / 9) = 1

-- Define a variable for the distance between the vertices
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

-- The main statement to be proved
theorem distance_of_hyperbola_vertices :
  hyperbola → distance_between_vertices 4 = 8 :=
by
  intro h
  sorry

end distance_of_hyperbola_vertices_l373_373391


namespace inverse_of_f_l373_373223

def f (x : ℝ) := 1 / (x - 2)

theorem inverse_of_f : ∀ x ≠ 0, (∀ y, f(x) = y ↔ x = (1 + 2 * y) / y) :=
by
  sorry

end inverse_of_f_l373_373223


namespace parabola_circle_distance_l373_373469

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373469


namespace max_good_set_size_l373_373910

def is_good_set (A : set ℕ) : Prop :=
  (∀ (λ1 λ2 λ3 : ℤ) (x1 x2 x3 : ℕ), 
    λ1 ∈ {-1, 0, 1} ∧ λ2 ∈ {-1, 0, 1} ∧ λ3 ∈ {-1, 0, 1} ∧ ¬ (λ1 = 0 ∧ λ2 = 0 ∧ λ3 = 0) 
    ∧ x1 ∈ A ∧ x2 ∈ A ∧ x3 ∈ A 
    → (x1 * x2 * x3 % 2014 ≠ 0 ∧ (λ1 * x1 + λ2 * x2 + λ3 * x3) % 2014 ≠ 0))

theorem max_good_set_size : 
  ∃ (A : set ℕ), (A ⊆ {n | n ∈ finset.range 2015} ∧ is_good_set A ∧ finset.card A = 503) := sorry

end max_good_set_size_l373_373910


namespace rationalize_denominator_l373_373198

def cuberoot3 : ℝ := real.cbrt 3
def cuberoot9 : ℝ := real.cbrt 9

theorem rationalize_denominator : 
  (1 / (cuberoot3 - 1)) = ((1 + cuberoot3 + cuberoot9) / 2) :=
by
  sorry

end rationalize_denominator_l373_373198


namespace smallest_num_conditions_l373_373314

theorem smallest_num_conditions :
  ∃ n : ℕ, (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 4 = 3) ∧ n = 11 :=
by
  sorry

end smallest_num_conditions_l373_373314


namespace boat_travel_times_l373_373010

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end boat_travel_times_l373_373010


namespace hyperbola_asymptotes_slope_l373_373942

theorem hyperbola_asymptotes_slope (a b c : ℝ) (h : a ≠ 0) (eccentricity_sqrt3 : c / a = sqrt 3) (c_squared_eq : c^2 = a^2 + b^2) :
  ∃ k : ℝ, k = sqrt 2 ∧ (∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 → (y = k * x ∨ y = -k * x)) :=
by {
  sorry
}

end hyperbola_asymptotes_slope_l373_373942


namespace find_t_l373_373855

-- Define the function f(x)
def f (x: ℝ) : ℝ := -x - 4

-- State the main theorem with appropriate conditions
theorem find_t (t : ℝ) :
  (∀ (k : ℝ), k > 0 → f(x + k) < f(x)) → t = -2 → 
  {x | |f(x - t) + 2| < 4 } = set.Ioo (-4 : ℝ) (4 : ℝ) :=
begin
  sorry
end

end find_t_l373_373855


namespace proof_math_problem_l373_373294

variables (initial_birds joined_storks joined_birds total_birds total_storks diff : ℕ)

def math_problem (initial_birds joined_storks joined_birds : ℕ) : Prop :=
  let total_birds := initial_birds + joined_birds in
  let total_storks := joined_storks in
  let diff := total_storks - total_birds in
  diff = 1

theorem proof_math_problem : math_problem 3 6 2 :=
by
  let initial_birds := 3
  let joined_storks := 6
  let joined_birds := 2
  let total_birds := initial_birds + joined_birds
  let total_storks := joined_storks
  let diff := total_storks - total_birds
  sorry

end proof_math_problem_l373_373294


namespace quadratic_real_roots_find_k_l373_373412

-- Part 1: Prove the equation always has two real roots
theorem quadratic_real_roots (k : ℝ) : 
  let δ := (k - 1)^2 in δ ≥ 0 :=
by
  sorry

-- Part 2: Find the value of k
theorem find_k (k : ℝ) :
  let α := k + 3
  let β := 2*k + 2
  α^2 + β^2 + α*β = 4 → (k = -1 ∨ k = -3) :=
by
  sorry

end quadratic_real_roots_find_k_l373_373412


namespace circumcenter_equidistant_l373_373231

-- Given three points A, B, and C forming a triangle, show that the circumcenter is equidistant from all three points.
theorem circumcenter_equidistant (A B C : Point) :
  let circ := circumcenter A B C in
  dist circ A = dist circ B ∧ dist circ B = dist circ C :=
by
  sorry

end circumcenter_equidistant_l373_373231


namespace max_gcd_coprime_l373_373399

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end max_gcd_coprime_l373_373399


namespace tangent_line_to_parabola_parallel_to_line_l373_373697

theorem tangent_line_to_parabola_parallel_to_line (x : ℝ) (hx : x = 1) :
  ∃ (y : ℝ), y = 2 * x - 1 ∧ 2 * x - y - 1 = 0 :=
by
  use (2 * x - 1)
  split
  {
    exact rfl,
  }
  {
    rw hx
    norm_num
  }

end tangent_line_to_parabola_parallel_to_line_l373_373697


namespace satisfactory_fraction_l373_373133

-- Define the number of each grade
def num_A : Nat := 7
def num_B : Nat := 6
def num_C : Nat := 5
def num_D : Nat := 4
def num_satisfactory : Nat := num_A + num_B + num_C + num_D

-- Define the number of unsatisfactory grades (combined E's and F's)
def num_unsatisfactory : Nat := 8

-- Define the total number of students
def total_students : Nat := num_satisfactory + num_unsatisfactory

-- Prove that the fraction of satisfactory grades is 11/15
theorem satisfactory_fraction : (num_satisfactory : ℚ) / (total_students : ℚ) = 11 / 15 := by sorry

end satisfactory_fraction_l373_373133


namespace parallel_necessary_but_not_sufficient_l373_373046

-- Given two non-zero vectors a and b
variables {α : Type*} [add_comm_group α] [vector_space ℝ α] 
  (a b : α) (h1 : a ≠ 0) (h2 : b ≠ 0)

-- Define the notion of parallel vectors
def parallel (a b : α) : Prop := ∃ (k : ℝ), k ≠ 0 ∧ a = k • b

-- Statement: proving the necessary but not sufficient condition
theorem parallel_necessary_but_not_sufficient :
  parallel a b ↔ (∃ k : ℝ, k > 0 ∧ a = k • b) :=
sorry

end parallel_necessary_but_not_sufficient_l373_373046


namespace boat_travel_times_l373_373011

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end boat_travel_times_l373_373011


namespace find_p_max_area_triangle_l373_373520

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373520


namespace first_division_percentage_l373_373604

theorem first_division_percentage (total_students : ℕ) (second_division_percentage : ℝ) (just_passed_students : ℕ) 
  (no_student_failed : total_students = (300 : ℕ)) : 
  total_students = 300 → second_division_percentage = 54 → just_passed_students = 57 → 
  (100 - second_division_percentage - (just_passed_students / total_students.to_real * 100)) = 27 :=
by 
  intro h1 h2 h3
  have h4 : second_division_percentage = 54 := h2
  have h5 : just_passed_students = 57 := h3
  have total_students_300 := h1
  sorry

end first_division_percentage_l373_373604


namespace value_of_m_l373_373116

theorem value_of_m (m α : ℝ) (hP : P = (-1, m)) (htrig : sin α * cos α = sqrt 3 / 4) : 
  m = -sqrt 3 ∨ m = -sqrt 3 / 3 :=
by
  -- Definitions based on point P(-1, m) on the terminal side
  have hsin : sin α = m / sqrt (1 + m^2), from sorry,
  have hcos : cos α = -1 / sqrt (1 + m^2), from sorry,
  -- Substitution into the trigonometric identity
  have hsubst : (m / sqrt (1 + m^2)) * (-1 / sqrt (1 + m^2)) = sqrt 3 / 4, from sorry,
  -- Solving for m
  have heq : (-m) / (1 + m^2) = sqrt 3 / 4, from sorry,
  -- Solving the quadractic equation
  have hsolve : m = -sqrt 3 ∨ m = -sqrt 3 / 3, from sorry,
  exact hsolve

#check value_of_m 123 4 -- This is to ensure Lean statement is valid.

end value_of_m_l373_373116


namespace max_gcd_coprime_l373_373400

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end max_gcd_coprime_l373_373400


namespace speed_of_train_in_kmh_l373_373807

-- Define the conditions
def time_to_cross_pole : ℝ := 6
def length_of_train : ℝ := 100
def conversion_factor : ℝ := 18 / 5

-- Using the conditions to assert the speed of the train
theorem speed_of_train_in_kmh (t : ℝ) (d : ℝ) (conv_factor : ℝ) : 
  t = time_to_cross_pole → 
  d = length_of_train → 
  conv_factor = conversion_factor → 
  (d / t) * conv_factor = 50 := 
by 
  intros h_t h_d h_conv_factor
  sorry

end speed_of_train_in_kmh_l373_373807


namespace count_non_divisible_5_9_l373_373856

theorem count_non_divisible_5_9 (n : ℕ) (h : n = 1199) :
  ∃ k, (k = n + 1 - (div_floor n 5 + div_floor n 9 - div_floor n 45)) ∧ k = 853 :=
by
  let div_floor := λ n d, n / d
  have h1 : div_floor 1199 5 = 239 := by sorry
  have h2 : div_floor 1199 9 = 133 := by sorry
  have h3 : div_floor 1199 45 = 26 := by sorry
  let lhs := 1199 - (239 + 133 - 26)
  have h4 : lhs = 853 := by sorry
  exact ⟨853, by simp [lhs, h4]⟩

end count_non_divisible_5_9_l373_373856


namespace distinct_ways_to_cover_circle_l373_373865

namespace FarmerJamesCircle

-- Define the conditions
def circle_circumference : ℝ := 10 * Real.pi
def arc_length (l: ℝ) : Prop := l = Real.pi ∨ l = 2 * Real.pi
def arc_colors : List String := ["red", "green", "blue"]
def adjacent_arcs_different_colors (c1 c2 : String) : Prop := c1 ≠ c2

-- Define the function to count distinct ways to cover the circle
noncomputable def count_distinct_coverings (circumference : ℝ) (lengths : List ℝ) (colors : List String)
  (adjacent_different : ∀ (c1 c2 : String), Prop) : ℕ := sorry

-- Problem statement
theorem distinct_ways_to_cover_circle : count_distinct_coverings circle_circumference [Real.pi, 2 * Real.pi] arc_colors adjacent_arcs_different_colors = 93 := sorry

end FarmerJamesCircle

end distinct_ways_to_cover_circle_l373_373865


namespace determinant_roots_l373_373157

theorem determinant_roots (a b c p q : ℝ) (h : Polynomial.roots (Polynomial.C (1 : ℝ) * Polynomial.X^3 - 4 * Polynomial.X^2 + p * Polynomial.X + q) = [a, b, c]) :
  Matrix.det (Matrix.of (λ (i j : Fin 3), ![a, b, c] (i.val + j.val) % 3)) = -64 + 8 * p :=
by {
  -- The detailed proof steps will go here.
  sorry
}

end determinant_roots_l373_373157


namespace total_preparation_and_cooking_time_l373_373349

def time_to_chop_pepper : Nat := 3
def time_to_chop_onion : Nat := 4
def time_to_grate_cheese_per_omelet : Nat := 1
def time_to_cook_omelet : Nat := 5
def num_peppers : Nat := 4
def num_onions : Nat := 2
def num_omelets : Nat := 5

theorem total_preparation_and_cooking_time :
  num_peppers * time_to_chop_pepper +
  num_onions * time_to_chop_onion +
  num_omelets * (time_to_grate_cheese_per_omelet + time_to_cook_omelet) = 50 := 
by
  sorry

end total_preparation_and_cooking_time_l373_373349


namespace travel_time_l373_373007

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end travel_time_l373_373007


namespace find_symmetry_point_l373_373421

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def perpendicular_bisector (A B : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ P, (2 * P.1 - P.2 = 2 * (A.1 + B.1) / 2 - (A.2 + B.2) / 2)

noncomputable def symmetric_point (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ :=
let ⟨a, b⟩ := P in ⟨2 * ⟦a, b⟧.1 - ⟦a, b⟧.2, -⟦a, b⟧⟩

theorem find_symmetry_point :
  let A := (10, 0)
  let B := (-6, 8)
  let C := (-4, 2)
  let M := midpoint A B
  let l := perpendicular_bisector A B
  let P := symmetric_point C l
  P = (4, -2) :=
by
  sorry

end find_symmetry_point_l373_373421


namespace inequality_problem_l373_373434

open Real

theorem inequality_problem 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h1 : x + y^2016 ≥ 1) : 
  x^2016 + y > 1 - 1/100 :=
by
  sorry

end inequality_problem_l373_373434


namespace exists_n_no_rational_solution_l373_373915

noncomputable def quadratic_polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem exists_n_no_rational_solution {a b c : ℝ} (h : ∃ n : ℕ, ¬∃ x : ℚ, quadratic_polynomial a b c x = (1 : ℝ) / n) :
  ∃ n : ℕ, ¬∃ x : ℚ, quadratic_polynomial a b c x = 1 / n :=
begin
  sorry
end

end exists_n_no_rational_solution_l373_373915


namespace gcd_max_possible_value_l373_373402

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end gcd_max_possible_value_l373_373402


namespace convert_base_5_to_base_10_l373_373852

theorem convert_base_5_to_base_10 :
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  a3 + a2 + a1 + a0 = 302 := by
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  show a3 + a2 + a1 + a0 = 302
  sorry

end convert_base_5_to_base_10_l373_373852


namespace find_p_max_area_triangle_l373_373535

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373535


namespace imo_geometry_problem_l373_373652

-- Definitions from conditions
variable (A B C A1 B1 C1 A2 B2 C2 O : Point)
variable [h1 : InscribedInCircle (Triangle.mk A B C) O]
variable [h2 : PerpendicularFromVertexIntersectsCircle A A1 A2 O]
variable [h3 : PerpendicularFromVertexIntersectsCircle B B1 B2 O]
variable [h4 : PerpendicularFromVertexIntersectsCircle C C1 C2 O]

-- Equivalent proof problem in Lean 4
theorem imo_geometry_problem :
  (AA_2 / AA_1) + (BB_2 / BB_1) + (CC_2 / CC_1) = 4 :=
by sorry

end imo_geometry_problem_l373_373652


namespace car_A_overtakes_B_and_C_l373_373835

-- Define the speeds of the cars
def speed_A : ℝ := 58 -- mph
def speed_B : ℝ := 50 -- mph
def speed_C : ℝ := 54 -- mph

-- Define the initial distances
def distance_A_B : ℝ := 24 -- miles (Car A is 24 miles behind Car B)
def distance_A_C : ℝ := 12 -- miles (Car A is 12 miles behind Car C)
def distance_C_B : ℝ := 12 -- miles (Car C is 12 miles behind Car B)

-- Define the relative speeds
def relative_speed_A_B : ℝ := speed_A - speed_B -- relative speed of Car A to Car B
def relative_speed_A_C : ℝ := speed_A - speed_C -- relative speed of Car A to Car C

-- Define the times to overtake
def time_to_overtake_B : ℝ := distance_A_B / relative_speed_A_B
def time_to_overtake_C : ℝ := distance_A_C / relative_speed_A_C

-- Main statement
theorem car_A_overtakes_B_and_C : time_to_overtake_B = 3 ∧ time_to_overtake_C = 3 :=
by
  -- Proof is skipped
  sorry

end car_A_overtakes_B_and_C_l373_373835


namespace max_area_triangle_PAB_l373_373546

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373546


namespace expected_baby_hawks_l373_373796

noncomputable def avg (a b : ℝ) := (a + b) / 2

theorem expected_baby_hawks (n_kettles : ℕ) (hawks_min hawks_max : ℝ) (preg_min preg_max : ℝ)
    (babies_min babies_max : ℝ) (loss_min loss_max : ℝ) :
  n_kettles = 15 →
  hawks_min = 18 → hawks_max = 25 →
  preg_min = 0.40 → preg_max = 0.60 →
  babies_min = 5 → babies_max = 8 →
  loss_min = 0.20 → loss_max = 0.40 →
  let avg_hawks := avg hawks_min hawks_max in
  let avg_preg := avg preg_min preg_max in
  let avg_babies := avg babies_min babies_max in
  let avg_loss := avg loss_min loss_max in
  let avg_survival := 1 - avg_loss in
  let total_hawks := avg_hawks * n_kettles in
  let pregnant_hawks := total_hawks * avg_preg in
  let total_babies := pregnant_hawks * avg_babies in
  let surviving_babies := total_babies * avg_survival in
  surviving_babies = 732 :=
by
  intros n_kettles_eq ...
  sorry

end expected_baby_hawks_l373_373796


namespace expression_I_expression_II_l373_373833

-- Proof problem for Expression I
theorem expression_I (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  a * sqrt a * ((1 / a) - (1 / sqrt (a^3 * b))) / ((a - b^(-1)) / (sqrt a + b^(-1/2))) = 1 :=
by
  sorry

-- Proof problem for Expression II
theorem expression_II :
  log 9 (sqrt 27) + sqrt (4 * log 2 3) - log 2 12 + log 2 3 = 7 / 4 :=
by
  sorry

end expression_I_expression_II_l373_373833


namespace triangle_acute_angles_5_integers_l373_373411

theorem triangle_acute_angles_5_integers :
  ∃ (xs : Finset ℤ), 
  (∀ x ∈ xs, 18 < x ∧ x < 42 ∧ real.sqrt 756 < x ∧ x < real.sqrt 1044) ∧
  xs.card = 5 :=
by
  sorry

end triangle_acute_angles_5_integers_l373_373411


namespace count_triangles_l373_373574

-- Setup the definitions and the condition
def rectangle := Type
def grid (n m : ℕ) := matrix.rectangle n m
def diagonal_lines (r : rectangle) := true
def crisscross_center (r : rectangle) := true

-- Given conditions
def given_conditions (r : rectangle) : Prop :=
  grid 2 6 ∧ diagonal_lines r ∧ crisscross_center r

-- Theorem statement
theorem count_triangles (r : rectangle) (h : given_conditions r) : number_of_triangles r = 88 := 
sorry

end count_triangles_l373_373574


namespace exists_nat_no_rational_solution_l373_373918

theorem exists_nat_no_rational_solution (p : ℝ → ℝ) (hp : ∃ a b c : ℝ, ∀ x, p x = a*x^2 + b*x + c) :
  ∃ n : ℕ, ∀ q : ℚ, p q ≠ 1 / (n : ℝ) :=
by
  sorry

end exists_nat_no_rational_solution_l373_373918


namespace hyperbola_eccentricity_l373_373054

theorem hyperbola_eccentricity (a b : ℝ) (h_asymptote : a = 3 * b) : 
    (a^2 + b^2) / a^2 = 10 / 9 := 
by
    sorry

end hyperbola_eccentricity_l373_373054


namespace monotonicity_of_f_l373_373415

theorem monotonicity_of_f (a : ℝ) (h : a > 0) : 
  ∀ x : ℝ, 0 < x ∧ x ≤ Real.sqrt a → (f_derivative x a < 0) 
  ∧ (∀ y : ℝ, 0 < y ∧ y ≤ Real.sqrt a → (x < y → f(x) > f(y))) 
  where f x := x + a / x
    f_derivative x a := 1 - a / x^2
:= sorry

end monotonicity_of_f_l373_373415


namespace first_player_wins_optimal_l373_373793

-- Define the structure of the game for clarity
def digit := Nat -- In the range 0 to 9, uniqueness is implied in the game rules
def op := digit → digit → digit
def add : op := Nat.add
def mul : op := Nat.mul

/-- The game board is transformed into an expression where operations and digits are inserted. -/
inductive expr
| digit (n : digit)
| binop (left : expr) (op : op) (right : expr)

open expr

/-- The resulting value of an expression -/
def eval_expr : expr → digit
| digit n => n
| binop left op right => op (eval_expr left) (eval_expr right)

/-- Determine if a number is even -/
def is_even (n : digit) : Prop := n % 2 = 0

/-- Determine if a number is odd -/
def is_odd (n : digit) : Prop := ¬ is_even n

/-- Determine the winner based on the evaluation of the expression -/
def winner (e : expr) : string :=
  if is_even (eval_expr e) then "First Player" else "Second Player"

/-- The first player wins under optimal play conditions. -/
theorem first_player_wins_optimal : 
  ∀ (e : expr), is_even (eval_expr e) :=
sorry

end first_player_wins_optimal_l373_373793


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373497

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373497


namespace sin_alpha_of_point_P_l373_373939

theorem sin_alpha_of_point_P (α : ℝ) 
  (h1 : ∃ P : ℝ × ℝ, P = (Real.cos (π / 3), 1) ∧ P = (Real.cos α, Real.sin α) ) :
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
  sorry

end sin_alpha_of_point_P_l373_373939


namespace sin_cos_range_l373_373991

theorem sin_cos_range (x : ℝ) (h : 0 < x ∧ x ≤ Real.pi / 3) :
  1 < Real.sin x + Real.cos x ∧ Real.sin x + Real.cos x ≤ Real.sqrt 2 :=
begin
  sorry
end

end sin_cos_range_l373_373991


namespace combined_cost_l373_373838

theorem combined_cost (wallet_cost : ℕ) (purse_cost : ℕ)
    (h_wallet_cost : wallet_cost = 22)
    (h_purse_cost : purse_cost = 4 * wallet_cost - 3) :
    wallet_cost + purse_cost = 107 :=
by
  rw [h_wallet_cost, h_purse_cost]
  norm_num
  sorry

end combined_cost_l373_373838


namespace find_k_l373_373971

-- Define the problem parameters
variables {x y k : ℝ}

-- The conditions given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (x + 2 * y = k - 1) ∧ (2 * x + y = 5 * k + 4)

def solution_condition (x y : ℝ) : Prop :=
  x + y = 5

-- The proof statement
theorem find_k (x y k : ℝ) (h1 : system_of_equations x y k) (h2 : solution_condition x y) :
  k = 2 :=
sorry

end find_k_l373_373971


namespace find_constants_l373_373638

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

noncomputable def N_inv : Matrix (Fin 2) (Fin 2) ℚ := N⁻¹

theorem find_constants :
  ∃ c d : ℚ, N_inv = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end find_constants_l373_373638


namespace treehouse_total_planks_l373_373359

theorem treehouse_total_planks (T : ℕ) 
    (h1 : T / 4 + T / 2 + 20 + 30 = T) : T = 200 :=
sorry

end treehouse_total_planks_l373_373359


namespace stock_remaining_percentage_l373_373627

theorem stock_remaining_percentage :
  let initial_stock := 1000
  let sold_Monday := (5 * initial_stock) / 100
  let remaining_after_Monday := initial_stock - sold_Monday
  let sold_Tuesday := (10 * remaining_after_Monday) / 100
  let remaining_after_Tuesday := remaining_after_Monday - sold_Tuesday
  let sold_Wednesday := (15 * remaining_after_Tuesday) / 100
  let remaining_after_Wednesday := remaining_after_Tuesday - (sold_Wednesday.floor : Int)
  let sold_Thursday := (20 * remaining_after_Wednesday) / 100
  let remaining_after_Thursday := remaining_after_Wednesday - (sold_Thursday.floor : Int)
  let sold_Friday := (25 * remaining_after_Thursday) / 100
  let remaining_after_Friday := remaining_after_Thursday - (sold_Friday.floor : Int)
  let final_percentage_not_sold := (remaining_after_Friday * 100) / initial_stock
  final_percentage_not_sold = 43.7 := by
  sorry

end stock_remaining_percentage_l373_373627


namespace count_integers_abs_less_than_3_14_l373_373573

theorem count_integers_abs_less_than_3_14 :
  {n : ℤ | abs n < 3.14}.toFinset.card = 7 :=
by
  -- The proof goes here
  sorry

end count_integers_abs_less_than_3_14_l373_373573


namespace average_speed_of_journey_l373_373311

theorem average_speed_of_journey :
  let D := D,    -- Assuming a distance D for each leg of the journey
  let time_to_office := (D / 20) / 2 + (D / 25) / 2 in
  let time_to_friend := (D / 30) / 2 + (D / 35) / 2 in
  let time_to_home := (D / 40) / 2 + (D / 45) / 2 in
  let total_time := time_to_office + time_to_friend + time_to_home in
  let total_distance := 3 * D in
  (total_distance / total_time) = 12.31 :=
sorry

end average_speed_of_journey_l373_373311


namespace correct_system_of_equations_l373_373608

theorem correct_system_of_equations (x y : ℕ) :
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔
  (∃ x y, (x / 3 = y - 2) ∧ ((x - 9) / 2 = y)) :=
by
  sorry

end correct_system_of_equations_l373_373608


namespace f_of_f_neg2_eq_half_l373_373453

def f (x : ℝ) : ℝ :=
  if x < 0 then 2^x else 1 - real.sqrt x

theorem f_of_f_neg2_eq_half : (f (f (-2))) = 1 / 2 :=
  sorry

end f_of_f_neg2_eq_half_l373_373453


namespace H_locus_on_AB_H_locus_in_triangle_l373_373592

section Geometry

variables {O A B M P Q H : Point}
variables (OA OB AB AB' λ : Real)
variables (acute_angle_AOB : ∠ AOB < 90)

-- Define the conditions in Lean
variables (in_triangle_OAB : ∀ M, M ∈ triangle O A B → 
                             (MP ⟂ OA) ∧ (MQ ⟂ OB))

-- Define the points C and D (feet of the perpendiculars)
noncomputable def C : Point := foot_of_perpendicular B OA
noncomputable def D : Point := foot_of_perpendicular A OB

-- Define the orthocenter of the triangle OPQ
noncomputable def orthocenter_OPQ (P Q : Point) : Point := 
  by sorry -- definition of orthocenter goes here

-- Define the movable point M on segment AB
noncomputable def M_on_AB (λ : Real) : Point := 
  (1 - λ) • A + λ • B

-- Prove the trajectory of H in each case

-- Part 1: When M is on segment AB
theorem H_locus_on_AB 
  (H : Point) 
  (locus_H : ∀ M ∈ segment A B, H = orthocenter_OPQ M Q P) :
    ∀ λ, H = (λ • C + (1 - λ) • D) :=
begin
  sorry
end

-- Part 2: When M moves within triangle OAB
theorem H_locus_in_triangle 
  (H : Point) 
  (locus_H : ∀ M, M ∈ triangle O A B → H = orthocenter_OPQ M Q P) :
    ∀ M, M ∈ triangle O A B → H ∈ triangle O C D :=
begin
  sorry
end

end Geometry

end H_locus_on_AB_H_locus_in_triangle_l373_373592


namespace backpack_cost_is_15_l373_373853

variable (total_spent : ℕ) (backpack_cost : ℕ) (pen_cost : ℕ) (pencil_cost : ℕ)
variable (notebook_cost : ℕ) (notebooks_bought : ℕ)

-- Assign values to the variables based on the conditions
def total_spent := 32
def pen_cost := 1
def pencil_cost := 1
def notebook_cost := 3
def notebooks_bought := 5

-- The cost of the pens and pencils
def pen_and_pencil_cost := pen_cost + pencil_cost

-- The cost of the notebooks
def total_notebook_cost := notebooks_bought * notebook_cost

-- The total cost of the pens, pencils, and notebooks
def other_items_cost := pen_and_pencil_cost + total_notebook_cost

-- The statement that we need to prove
theorem backpack_cost_is_15 :
  (total_spent = backpack_cost + other_items_cost) → backpack_cost = 15 :=
by
  simp [total_spent, backpack_cost, other_items_cost, pen_and_pencil_cost, total_notebook_cost]
  sorry

end backpack_cost_is_15_l373_373853


namespace total_number_of_toys_l373_373858

theorem total_number_of_toys (average_cost_Dhoni_toys : ℕ) (number_Dhoni_toys : ℕ) 
    (price_David_toy : ℕ) (new_avg_cost : ℕ) 
    (h1 : average_cost_Dhoni_toys = 10) (h2 : number_Dhoni_toys = 5) 
    (h3 : price_David_toy = 16) (h4 : new_avg_cost = 11) : 
    (number_Dhoni_toys + 1) = 6 := 
by
  sorry

end total_number_of_toys_l373_373858


namespace quadratic_roots_problem_l373_373169

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end quadratic_roots_problem_l373_373169


namespace geometric_sequence_sum_of_squares_l373_373060

theorem geometric_sequence_sum_of_squares (a : ℝ) (n : ℕ) (Sn : ℕ → ℝ) (an : ℕ → ℝ) 
  (hSn : ∀ n, Sn n = 2^n - a) 
  (h_an : ∀ n, an n = (if n = 1 then 2 - a else 2^(n-1))) :
  (∑ k in Finset.range n, (an k)^2) = 1 / 3 * (4^n - 1) :=
sorry

end geometric_sequence_sum_of_squares_l373_373060


namespace prime_saturated_value_96_l373_373313

def is_prime_saturated (g : ℕ) : Prop :=
  let prime_factors := (Multiset.to_finset (Nat.factors g)).to_list
  let product_of_prime_factors := prime_factors.prod id
  product_of_prime_factors < g

theorem prime_saturated_value_96 : 
  is_prime_saturated 96 ∧ 
  (let prime_factors := (Multiset.to_finset (Nat.factors 96)).to_list in 
   prime_factors.prod id < 96) := 
sorry

end prime_saturated_value_96_l373_373313


namespace sum_mod_remainder_l373_373752

theorem sum_mod_remainder :
  (∑ i in Finset.range 21, i % 9) = 3 :=
by sorry

end sum_mod_remainder_l373_373752


namespace rounding_1_6954_l373_373343

noncomputable def round_to_nearest_hundredth (x : ℝ) : ℝ :=
  (Real.Floor (x * 100 + 0.5)) / 100

theorem rounding_1_6954 :
  round_to_nearest_hundredth 1.6954 = 1.70 :=
by
  sorry

end rounding_1_6954_l373_373343


namespace probability_even_distinct_l373_373341
-- Importing Mathlib for necessary mathematical tools and tactics.

-- Define the range of integers.
def is_in_range (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define the condition of being an even integer with distinct digits.
def is_even_and_distinct (n : ℕ) : Prop :=
  (n % 2 = 0) ∧ ((n / 1000) ≠ ((n % 1000) / 100)) ∧ ((n / 1000) ≠ (n % 10)) ∧ ((n / 1000) ≠ ((n % 100) / 10)) ∧
  (((n % 1000) / 100) ≠ ((n % 100) / 10)) ∧ (((n % 1000) / 100) ≠ (n % 10)) ∧ (((n % 100) / 10) ≠ (n % 10))

-- Define the proof problem as a theorem.
theorem probability_even_distinct : 
  let favorable_count := ∑ n in (finset.range 9999).filter is_in_range, if is_even_and_distinct n then 1 else 0
  let total_count := 9000
  (favorable_count : ℚ) / total_count = 382 / 1500 := sorry

end probability_even_distinct_l373_373341


namespace count_three_digit_perfect_squares_divisible_by_4_l373_373106

theorem count_three_digit_perfect_squares_divisible_by_4 :
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), 10 ≤ k ∧ k ≤ 31 → (∃ m : ℕ, m^2 = k^2 ∧ 100 ≤ m^2 ∧ m^2 ≤ 999 ∧ m^2 % 4 = 0) := 
sorry

end count_three_digit_perfect_squares_divisible_by_4_l373_373106


namespace intersect_at_one_point_l373_373668

open Classical

variables {A B C D : Type} [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D]
variables (AB : segment A B) (BC : segment B C) (CD : segment C D) (DA : segment D A)
variables [cyclic AB BC CD DA]
variables (M : midpoint AB) (N : midpoint BC) (P : midpoint CD) (Q : midpoint DA)
variables (L1 : line M (perpendicular CD)) (L2 : line N (perpendicular DA)) (L3 : line P (perpendicular AB)) (L4 : line Q (perpendicular BC))

theorem intersect_at_one_point :
  ∃ P : point, P ∈ L1 ∧ P ∈ L2 ∧ P ∈ L3 ∧ P ∈ L4 :=
sorry

end intersect_at_one_point_l373_373668


namespace perfectutation_pairs_modulo_2011_rem_l373_373644

-- Definitions
def S : Set ℕ := { i | 1 ≤ i ∧ i ≤ 2012 }
def isPerfectutation (h : S → S) : Prop :=
  Function.Bijective h ∧ 
  ∃ a ∈ S, h a ≠ a ∧ 
    ∀ a b ∈ S, h a ≠ a → h b ≠ b → ∃ k : ℕ, h^[k] a = b

theorem perfectutation_pairs_modulo_2011_rem (f g : S → S) 
  (hf : isPerfectutation f) (hg : isPerfectutation g) 
  (h_comm : ∀ i ∈ S, f (g i) = g (f i)) 
  (h_neq : f ≠ g) :
  (number_of_pairs S / 2011) % 2011 = 2 :=
  sorry

end perfectutation_pairs_modulo_2011_rem_l373_373644


namespace total_games_attended_l373_373625

theorem total_games_attended 
  (games_this_month : ℕ)
  (games_last_month : ℕ)
  (games_next_month : ℕ)
  (total_games : ℕ) 
  (h : games_this_month = 11)
  (h2 : games_last_month = 17)
  (h3 : games_next_month = 16) 
  (htotal : total_games = 44) :
  games_this_month + games_last_month + games_next_month = total_games :=
by sorry

end total_games_attended_l373_373625


namespace count_three_digit_perfect_squares_divisible_by_4_l373_373105

theorem count_three_digit_perfect_squares_divisible_by_4 :
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), 10 ≤ k ∧ k ≤ 31 → (∃ m : ℕ, m^2 = k^2 ∧ 100 ≤ m^2 ∧ m^2 ≤ 999 ∧ m^2 % 4 = 0) := 
sorry

end count_three_digit_perfect_squares_divisible_by_4_l373_373105


namespace sum_of_numbers_greater_than_1_1_is_3_9_l373_373902

theorem sum_of_numbers_greater_than_1_1_is_3_9 :
  let numbers := {1.4, 9 / 10, 1.2, 0.5, 13 / 10}
  let filtered_numbers := {x : ℝ | x ∈ numbers ∧ x > 1.1}
  (∑ x in filtered_numbers, x) = 3.9 := by
{
  sorry -- The proof is omitted as per the instructions
}

end sum_of_numbers_greater_than_1_1_is_3_9_l373_373902


namespace log2_is_integer_probability_l373_373316

theorem log2_is_integer_probability :
  let four_digit_numbers := {N : ℕ | 1000 ≤ N ∧ N ≤ 9999},
      valid_numbers := {N : ℕ | ∃ k : ℕ, N = 2^k ∧ 1000 ≤ N ∧ N ≤ 9999},
      probability := (finset.card valid_numbers.to_finset : ℚ) / (finset.card four_digit_numbers.to_finset : ℚ)
  in probability = 1 / 2250 :=
by
  sorry

end log2_is_integer_probability_l373_373316


namespace solve_k_l373_373968

theorem solve_k (x y k : ℝ) (h1 : x + 2 * y = k - 1) (h2 : 2 * x + y = 5 * k + 4) (h3 : x + y = 5) :
  k = 2 :=
sorry

end solve_k_l373_373968


namespace remaining_area_is_correct_l373_373309

-- Define the large rectangle's side lengths
def large_rectangle_length1 (x : ℝ) := x + 7
def large_rectangle_length2 (x : ℝ) := x + 5

-- Define the hole's side lengths
def hole_length1 (x : ℝ) := x + 1
def hole_length2 (x : ℝ) := x + 4

-- Calculate the areas
def large_rectangle_area (x : ℝ) := large_rectangle_length1 x * large_rectangle_length2 x
def hole_area (x : ℝ) := hole_length1 x * hole_length2 x

-- Define the remaining area after subtracting the hole area from the large rectangle area
def remaining_area (x : ℝ) := large_rectangle_area x - hole_area x

-- Problem statement: prove that the remaining area is 7x + 31
theorem remaining_area_is_correct (x : ℝ) : remaining_area x = 7 * x + 31 :=
by 
  -- The proof should be provided here, but for now we use 'sorry' to omit it
  sorry

end remaining_area_is_correct_l373_373309


namespace cubic_polynomial_root_form_l373_373363

theorem cubic_polynomial_root_form :
  ∃ a b c : ℤ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 
    (∃ x : ℝ, 27 * x^3 - 12 * x^2 - 12 * x - 4 = 0 ∧ x = (real.cbrt a + real.cbrt b + 2) / c) ∧ 
    a + b + c = 12 :=
sorry

end cubic_polynomial_root_form_l373_373363


namespace slant_height_l373_373058

-- Define the variables and conditions
variables (r A : ℝ)
-- Assume the given conditions
def radius := r = 5
def area := A = 60 * Real.pi

-- Statement of the theorem to prove the slant height
theorem slant_height (r A l : ℝ) (h_r : r = 5) (h_A : A = 60 * Real.pi) : l = 12 :=
sorry

end slant_height_l373_373058


namespace boat_travel_times_l373_373009

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end boat_travel_times_l373_373009


namespace find_p_max_area_triangle_l373_373513

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373513


namespace cube_root_of_square_root_eight_l373_373583

theorem cube_root_of_square_root_eight :
  ∀ (x : ℝ), (sqrt x = 8) → real.cbrt x = 4 :=
by
  intro x h
  have h1 : x = 8 ^ 2 := by
    rw [←sq_eq_of_eq_sqrt h x]
  rw [h1, real.cbrt_pow (64 : ℝ) 4]
  sorry

end cube_root_of_square_root_eight_l373_373583


namespace sum_of_roots_of_quadratic_eqn_l373_373377

theorem sum_of_roots_of_quadratic_eqn (A B : ℝ) 
  (h₁ : 3 * A ^ 2 - 9 * A + 6 = 0)
  (h₂ : 3 * B ^ 2 - 9 * B + 6 = 0)
  (h_distinct : A ≠ B):
  A + B = 3 := by
  sorry

end sum_of_roots_of_quadratic_eqn_l373_373377


namespace min_max_sum_distances_l373_373036

noncomputable def cube_edge_length := 2

def P1 (α1 : ℝ) : ℝ × ℝ × ℝ := (1, cos α1, sin α1)
def P2 (α2 : ℝ) : ℝ × ℝ × ℝ := (sin α2, 1, cos α2)
def P3 (α3 : ℝ) : ℝ × ℝ × ℝ := (cos α3, sin α3, 1)

def distance (X Y : ℝ × ℝ × ℝ) : ℝ :=
  (Real.sqrt ((X.1 - Y.1) ^ 2 + (X.2 - Y.2) ^ 2 + (X.3 - Y.3) ^ 2))

def sum_distances (α1 α2 α3 : ℝ) : ℝ :=
  distance (P1 α1) (P2 α2) + distance (P2 α2) (P3 α3) + distance (P3 α3) (P1 α1)

theorem min_max_sum_distances:
  ∃ (min max : ℝ), min = 3 * Real.sqrt 2 - 3 ∧ max = 3 * Real.sqrt 2 ∧
    (∀ α1 α2 α3 : ℝ, min ≤ sum_distances α1 α2 α3 ∧ sum_distances α1 α2 α3 ≤ max) :=
sorry

end min_max_sum_distances_l373_373036


namespace combined_cost_is_107_l373_373841

def wallet_cost : ℕ := 22
def purse_cost (wallet_price : ℕ) : ℕ := 4 * wallet_price - 3
def combined_cost (wallet_price : ℕ) (purse_price : ℕ) : ℕ := wallet_price + purse_price

theorem combined_cost_is_107 : combined_cost wallet_cost (purse_cost wallet_cost) = 107 := 
by 
  -- Proof
  sorry

end combined_cost_is_107_l373_373841


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373498

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373498


namespace center_of_symmetry_max_value_of_f_l373_373959

noncomputable def f (x : ℝ) : ℝ := Real.sin(2 * x) - (Real.cos(x) ^ 2) * Real.sin(2 * x)

theorem center_of_symmetry (x : ℝ) : 
  f(π / 2 - x) + f(π / 2 + x) = 0 := 
sorry

theorem max_value_of_f : 
  ∃ x : ℝ, f(x) = 3 * Real.sqrt(3) / 8 := 
sorry

end center_of_symmetry_max_value_of_f_l373_373959


namespace area_of_right_angled_triangle_l373_373764

theorem area_of_right_angled_triangle (base hypotenuse : ℝ) (h1 : base = 12) (h2 : hypotenuse = 13) : 
  let height := Real.sqrt (hypotenuse^2 - base^2)
  in 1 / 2 * base * height = 30 :=
by
  sorry

end area_of_right_angled_triangle_l373_373764


namespace triangle_area_sum_l373_373591

theorem triangle_area_sum :
  ∀ (A B C D E : Type) 
    (AC_length : ℝ) 
    (m∠BAC m∠ABC m∠ACB m∠DEC : ℝ)
    (AD DC : ℝ)
    (cos sin : ℝ → ℝ) 
    (sqrt : ℝ → ℝ), 
    E = midpoint B C →
    D ∈ segment A C →
    AD = DC → 
    AC_length = 1 →
    m∠BAC = 50 →
    m∠ABC = 70 →
    m∠ACB = 60 →
    m∠DEC = 100 →
    AD = 0.5 →
    DC = 0.5 →
    (area_of_triangle A B C + 2 * area_of_triangle C D E) = 
    (sqrt 3 / 4) * ((sin 70 / sin 60) + 0.5) := 
by
  sorry

end triangle_area_sum_l373_373591


namespace like_term_exponents_l373_373587

theorem like_term_exponents (m n : ℕ) (h1 : 3x^m y = -5 x^3 y^n) : m + n = 4 := by
  sorry

end like_term_exponents_l373_373587


namespace find_a_value_l373_373376

theorem find_a_value :
  (∀ y : ℝ, y ∈ Set.Ioo (-3/2 : ℝ) 4 → y * (2 * y - 3) < (12 : ℝ)) ↔ (12 = 12) := 
by 
  sorry

end find_a_value_l373_373376


namespace skylar_age_started_l373_373203

variable (annual_donation : ℕ) (total_donation : ℕ) (current_age : ℕ)

def years_of_donation := total_donation / annual_donation

def age_when_started_donating := current_age - years_of_donation

theorem skylar_age_started (annual_donation : ℕ) (total_donation : ℕ) (current_age : ℕ) :
  annual_donation = 5000 → total_donation = 105000 → current_age = 33 →
  age_when_started_donating annual_donation total_donation current_age = 12 :=
by
  intros annual_donation_eq total_donation_eq current_age_eq
  rw [annual_donation_eq, total_donation_eq, current_age_eq]
  have h1 : years_of_donation 5000 105000 = 21 := by
    unfold years_of_donation
    exact Nat.div_eq_of_eq_mul_right (by norm_num) (by norm_num)
  unfold age_when_started_donating
  rw h1
  norm_num
  sorry

end skylar_age_started_l373_373203


namespace find_p_max_area_triangle_l373_373530

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373530


namespace relationship_among_values_l373_373441

noncomputable def f : ℝ → ℝ := sorry
def a := f (1 / 2)
def b := f 1
def c := f (-3)

-- f is even
axiom even_f : ∀ x : ℝ, f x = f (-x)

-- f is increasing on (-∞, 0]
axiom increasing_f : ∀ x y : ℝ, x < y ∧ y ≤ 0 → f x < f y

theorem relationship_among_values : a > b ∧ b > c :=
by
  -- This is where the proof would go; the statement is the formulation
  -- of the problem as a theorem in Lean.
  sorry

end relationship_among_values_l373_373441


namespace isosceles_base_angle_l373_373993

theorem isosceles_base_angle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = B ∨ A = C) (h3 : A = 80 ∨ B = 80 ∨ C = 80) : (A = 80 ∧ B = 80) ∨ (A = 80 ∧ C = 80) ∨ (B = 80 ∧ C = 50) ∨ (C = 80 ∧ B = 50) :=
sorry

end isosceles_base_angle_l373_373993


namespace cube_root_equation_solutions_l373_373876

theorem cube_root_equation_solutions (x : ℝ) :
  (∃ (y : ℝ), y = real.cbrt x ∧ y = 15 / (8 - y)) ↔ (x = 27 ∨ x = 125) :=
by
  sorry

end cube_root_equation_solutions_l373_373876


namespace mixture_cost_l373_373691

theorem mixture_cost (C : ℝ) (h1 : C = 1) :
  let coffee_price_july := 4 * C,
      milk_price_july := 0.2 * C,
      mixture_weight := 3,
      milk_powder_cost := 0.2,
      coffee_cost := 4
  in mixture_weight / 2 * milk_powder_cost + mixture_weight / 2 * coffee_cost = 6.3 :=
by
  sorry

end mixture_cost_l373_373691


namespace ratio_sum_div_c_l373_373576

theorem ratio_sum_div_c (a b c : ℚ) (h : a / 3 = b / 4 ∧ b / 4 = c / 5) : (a + b + c) / c = 12 / 5 :=
by
  sorry

end ratio_sum_div_c_l373_373576


namespace tangent_line_at_one_l373_373423

-- Given conditions
def f : ℝ → ℝ := sorry
axiom f_prop : ∀ x : ℝ, f(1 + x) = 2 * f(1 - x) - x^2 + 3 * x + 1

-- Statement of the problem
theorem tangent_line_at_one :
  let y := f 1 in
  let m := 1 in -- From the derivative
  ∀ x : ℝ, x - (m * (x - 1) + y) - 2 = 0 :=
sorry

end tangent_line_at_one_l373_373423


namespace ratio_of_segments_of_inscribed_circle_l373_373788

theorem ratio_of_segments_of_inscribed_circle 
  (a b c : ℕ) (h : set.insert a (set.insert b {c}) = {12, 16, 20}) 
  (r s : ℕ) (hs : r + s = 12) (hr_lt_hs : r < s) : r = 4 ∧ s = 8 :=
sorry

end ratio_of_segments_of_inscribed_circle_l373_373788


namespace two_bishops_placement_l373_373606

theorem two_bishops_placement :
  let squares := 64
  let white_squares := 32
  let black_squares := 32
  let first_bishop_white_positions := 32
  let second_bishop_black_positions := 32 - 8
  first_bishop_white_positions * second_bishop_black_positions = 768 := by
  sorry

end two_bishops_placement_l373_373606


namespace Newville_Academy_fraction_l373_373821

theorem Newville_Academy_fraction :
  let total_students := 100
  let enjoy_sports := 0.7 * total_students
  let not_enjoy_sports := 0.3 * total_students
  let say_enjoy_right := 0.75 * enjoy_sports
  let say_not_enjoy_wrong := 0.25 * enjoy_sports
  let say_not_enjoy_right := 0.85 * not_enjoy_sports
  let say_enjoy_wrong := 0.15 * not_enjoy_sports
  let say_not_enjoy_total := say_not_enjoy_wrong + say_not_enjoy_right
  let say_not_enjoy_but_enjoy := say_not_enjoy_wrong
  (say_not_enjoy_but_enjoy / say_not_enjoy_total) = (7 / 17) := by
  sorry

end Newville_Academy_fraction_l373_373821


namespace num_zeros_in_expansion_l373_373109

noncomputable def bigNum := (10^11 - 2) ^ 2

theorem num_zeros_in_expansion : ∀ n : ℕ, bigNum = n ↔ (n = 9999999999900000000004) := sorry

end num_zeros_in_expansion_l373_373109


namespace inequality_of_distinct_positives_l373_373440

variable {a b c : ℝ}

theorem inequality_of_distinct_positives (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
(habc : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  2 * (a^3 + b^3 + c^3) > a^2 * (b + c) + b^2 * (a + c) + c^2 * (a + b) :=
by
  sorry

end inequality_of_distinct_positives_l373_373440


namespace proof_part1_proof_part2_l373_373477

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373477


namespace proof_part1_proof_part2_l373_373476

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373476


namespace at_most_three_digits_count_l373_373095

theorem at_most_three_digits_count : 
  ∃ n, (n < 100000) ∧ 
       (∀ m < n, m ∈ ( ℕ.filter (λ x, (distinct_digits x ≤ 3)))) ∧
       (count_at_most_three_digits n) = 6435 :=
sorry

noncomputable def distinct_digits (x: ℕ): ℕ :=
sorry

noncomputable def count_at_most_three_digits (x: ℕ): ℕ :=
sorry

end at_most_three_digits_count_l373_373095


namespace product_of_roots_l373_373937

theorem product_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 :=
sorry

end product_of_roots_l373_373937


namespace range_of_x_satisfying_inequality_l373_373456

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 1 else 1

theorem range_of_x_satisfying_inequality :
  {x : ℝ | f (1 - x^2) > f (2 * x)} = set.Ioo (-1) (real.sqrt 2 - 1) :=
by
  sorry

end range_of_x_satisfying_inequality_l373_373456


namespace necessary_and_sufficient_conditions_l373_373932

-- Define the function f
def f (a x : ℝ) : ℝ := a * x - x^2

-- Define the domain of x
def dom_x (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

theorem necessary_and_sufficient_conditions {a : ℝ} (ha : a > 0) :
  (∀ x : ℝ, dom_x x → f a x ≤ 1) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end necessary_and_sufficient_conditions_l373_373932


namespace combined_cost_is_107_l373_373842

def wallet_cost : ℕ := 22
def purse_cost (wallet_price : ℕ) : ℕ := 4 * wallet_price - 3
def combined_cost (wallet_price : ℕ) (purse_price : ℕ) : ℕ := wallet_price + purse_price

theorem combined_cost_is_107 : combined_cost wallet_cost (purse_cost wallet_cost) = 107 := 
by 
  -- Proof
  sorry

end combined_cost_is_107_l373_373842


namespace ratio_a_b_c_l373_373031

-- Given condition 14(a^2 + b^2 + c^2) = (a + 2b + 3c)^2
theorem ratio_a_b_c (a b c : ℝ) (h : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : 
  a / b = 1 / 2 ∧ b / c = 2 / 3 :=
by 
  sorry

end ratio_a_b_c_l373_373031


namespace cube_remaining_faces_sum_l373_373698

theorem cube_remaining_faces_sum {a b c d e f : ℕ} (h1 : {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6})
  (h2 : {a, b, c} = {1, 3, 5}) :
  (d + e + f) = 12 := by 
  sorry

end cube_remaining_faces_sum_l373_373698


namespace least_positive_integer_to_add_l373_373731

theorem least_positive_integer_to_add (n : ℕ) (h1 : n > 0) (h2 : (624 + n) % 5 = 0) : n = 1 := 
by
  sorry

end least_positive_integer_to_add_l373_373731


namespace travel_times_either_24_or_72_l373_373004

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end travel_times_either_24_or_72_l373_373004


namespace intersection_A_complement_B_l373_373076

def U := ℝ

def A : set ℝ := { x | 0 < x }

def B : set ℝ := { y | 1 ≤ y }

def B_complement : set ℝ := { y | y < 1 }

theorem intersection_A_complement_B :
  (A ∩ B_complement) = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_complement_B_l373_373076


namespace parabola_condition_max_area_triangle_l373_373483

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373483


namespace interval_of_decrease_l373_373372

def quadratic (x : ℝ) := 3 * x^2 - 7 * x + 2

def decreasing_interval (y : ℝ) := y < 2 / 3

theorem interval_of_decrease :
  {x : ℝ | x < (1 / 3)} = {x : ℝ | x < (1 / 3)} :=
by sorry

end interval_of_decrease_l373_373372


namespace bus_routes_count_l373_373214

variables (n N : Nat)

-- Given conditions as definitions
def condition1 : Prop := ∀ x y : Nat, (x ≠ y) → (∃ p : Nat, p ≠ x ∧ p ≠ y)
def condition2 : Prop := ∀ a b : Nat, (a ≠ b) → (∃ s : Nat, ∃ r : Nat, s ≠ r ∧ s ≠ a ∧ s ≠ b)
def condition3 : Prop := ∀ r : Nat, ∃ A B C : Nat, (A ≠ B ∧ B ≠ C ∧ A ≠ C)

-- Number of stops is given as 3
def stops_condition : n = 3 := rfl

theorem bus_routes_count : 
  condition1 ∧ condition2 ∧ condition3 ∧ stops_condition → N = 7 := 
by
  sorry

end bus_routes_count_l373_373214


namespace smallest_m_l373_373803

noncomputable theory
open_locale big_operators

def is_integer (x : ℝ) : Prop := ∃ (n : ℤ), x = n

theorem smallest_m (m : ℕ) (h : ∀ m, (0 < m ∧ is_integer (10000 * m / 107) → m = 107)) : m = 107 :=
sorry

end smallest_m_l373_373803


namespace certain_number_proof_l373_373780

noncomputable def certain_number : ℝ := 30

theorem certain_number_proof (h1: 0.60 * 50 = 30) (h2: 30 = 0.40 * certain_number + 18) : 
  certain_number = 30 := 
sorry

end certain_number_proof_l373_373780


namespace cost_of_bananas_and_cantaloupe_l373_373200

-- Define prices for different items
variables (a b c d e : ℝ)

-- Define the conditions as hypotheses
theorem cost_of_bananas_and_cantaloupe (h1 : a + b + c + d + e = 30)
    (h2 : d = 3 * a) (h3 : c = a - b) (h4 : e = a + b) :
    b + c = 5 := 
by 
  -- Initial proof setup
  sorry

end cost_of_bananas_and_cantaloupe_l373_373200


namespace convex_enclosed_by_centrally_symmetric_no_centrally_symmetric_enclosing_triangle_l373_373778

-- Define the notion of a convex set
def is_convex (s : Set ℝ → ℝ) : Prop := 
  ∀ (x y ∈ s) (t ∈ Icc 0 1), t * x + (1 - t) * y ∈ s

-- Define the notion of centrally symmetric
def is_centrally_symmetric (s : Set ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x ∈ s, 2 * c - x ∈ s

-- Problem (a): Proving existence of centrally symmetric set enclosing a convex set
theorem convex_enclosed_by_centrally_symmetric (Φ : Set ℝ → ℝ) (hΦ : is_convex Φ) :
  ∃ Ψ, is_centrally_symmetric Ψ 0 ∧ area Ψ ≤ 2 * area Φ := 
sorry

-- Problem (b): Proving non-existence of centrally symmetric set enclosing a triangle with area less than twice its area
theorem no_centrally_symmetric_enclosing_triangle (ABC : Set ℝ → ℝ) (hABC : is_convex ABC ∧ (∃ A B C : ℝ, ABC = {A, B, C})) :
  ∀ Ψ, is_centrally_symmetric Ψ 0 → area Ψ < 2 * area ABC → False := 
sorry

end convex_enclosed_by_centrally_symmetric_no_centrally_symmetric_enclosing_triangle_l373_373778


namespace chessboard_values_not_equal_l373_373239

theorem chessboard_values_not_equal (n : ℕ) (h_n : n > 100) :
  (∃ M : ℕ → ℕ → ℤ, (∀ i j, M i j ∈ {0, 1}) ∧ 
  (∃ count_1 : ℕ, count_1 = n - 1 ∧ ∑ i j, (if M i j = 1 then 1 else 0) = count_1) ∧
  (∀ i j, ∃ M' : ℕ → ℕ → ℤ, M' i j = M i j - 1 ∧ 
   ∀ k, if k ≠ j then (M' i k = M i k + 1) ∧ 
   ∀ l, if l ≠ i then (M' l j = M l j + 1)) ∧ 
  (∀ t, ¬ ∀ u v, M t u = M t v)) :=
sorry

end chessboard_values_not_equal_l373_373239


namespace stock_remaining_percentage_l373_373628

theorem stock_remaining_percentage :
  let initial_stock := 1000
  let sold_Monday := (5 * initial_stock) / 100
  let remaining_after_Monday := initial_stock - sold_Monday
  let sold_Tuesday := (10 * remaining_after_Monday) / 100
  let remaining_after_Tuesday := remaining_after_Monday - sold_Tuesday
  let sold_Wednesday := (15 * remaining_after_Tuesday) / 100
  let remaining_after_Wednesday := remaining_after_Tuesday - (sold_Wednesday.floor : Int)
  let sold_Thursday := (20 * remaining_after_Wednesday) / 100
  let remaining_after_Thursday := remaining_after_Wednesday - (sold_Thursday.floor : Int)
  let sold_Friday := (25 * remaining_after_Thursday) / 100
  let remaining_after_Friday := remaining_after_Thursday - (sold_Friday.floor : Int)
  let final_percentage_not_sold := (remaining_after_Friday * 100) / initial_stock
  final_percentage_not_sold = 43.7 := by
  sorry

end stock_remaining_percentage_l373_373628


namespace necessary_condition_ac_eq_bc_l373_373024

theorem necessary_condition_ac_eq_bc {a b c : ℝ} (hc : c ≠ 0) : (ac = bc ↔ a = b) := by
  sorry

end necessary_condition_ac_eq_bc_l373_373024


namespace monotonic_increasing_intervals_g_max_value_l373_373075

noncomputable def f (x : ℝ) : ℝ := sin (2 * x)

noncomputable def g (x a : ℝ) : ℝ := f (x - a)

theorem monotonic_increasing_intervals (k : ℤ) :
  monotone_on f (set.Icc (k * π - π / 4) (k * π + π / 4)) :=
sorry

theorem g_max_value (a : ℝ) (k : ℤ) :
  ∃ x : ℝ, g x a = 1 ∧ x = k * π + π / 2 - a :=
sorry

end monotonic_increasing_intervals_g_max_value_l373_373075


namespace daniel_takeaway_pieces_l373_373367

theorem daniel_takeaway_pieces (total_pieces : ℕ) (friends : ℕ) (remainder : ℕ) (parity : total_pieces % friends = remainder) (least_takeaway : total_pieces - remainder) :
  total_pieces = 24 → friends = 5 → remainder = 4 → least_takeaway = 4 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end daniel_takeaway_pieces_l373_373367


namespace sin_3x_zero_implies_x_is_60_l373_373221

theorem sin_3x_zero_implies_x_is_60 (x : ℝ) (h : sin (3 * x * (π/180)) = 0) : x = 60 :=
by sorry

end sin_3x_zero_implies_x_is_60_l373_373221


namespace exists_nat_no_rational_solution_l373_373917

theorem exists_nat_no_rational_solution (p : ℝ → ℝ) (hp : ∃ a b c : ℝ, ∀ x, p x = a*x^2 + b*x + c) :
  ∃ n : ℕ, ∀ q : ℚ, p q ≠ 1 / (n : ℝ) :=
by
  sorry

end exists_nat_no_rational_solution_l373_373917


namespace number_of_elements_M_l373_373912

open Real

def M := { (x : ℝ) × (y : ℝ) | x^2 + 8 * x * sin (1 / 4 * x + y) * π + 16 = 0 ∧ 0 ≤ y ∧ y ≤ 5 }

theorem number_of_elements_M : (set.to_finset M).card = 5 := 
sorry

end number_of_elements_M_l373_373912


namespace tangent_line_perpendicular_l373_373455

noncomputable def f (x k : ℝ) : ℝ := x^3 - (k^2 - 1) * x^2 - k^2 + 2

theorem tangent_line_perpendicular (k : ℝ) (b : ℝ) (a : ℝ)
  (h1 : ∀ (x : ℝ), f x k = x^3 - (k^2 - 1) * x^2 - k^2 + 2)
  (h2 : (3 - 2 * (k^2 - 1)) = -1) :
  a = -2 := sorry

end tangent_line_perpendicular_l373_373455


namespace problem_statement_l373_373458

noncomputable def f (a x : ℝ) := a * (x ^ 2 + 1) + Real.log x

theorem problem_statement (a m : ℝ) (x : ℝ) 
  (h_a : -4 < a) (h_a' : a < -2) (h_x1 : 1 ≤ x) (h_x2 : x ≤ 3) :
  (m * a - f a x > a ^ 2) ↔ (m ≤ -2) :=
by
  sorry

end problem_statement_l373_373458


namespace convex_polyhedron_has_triangle_edges_l373_373194

variables {n : ℕ} {x : Fin n → ℝ}

def edges_satisfy (x : Fin n → ℝ) : Prop :=
  ∀ (i : Fin n), 2 ≤ i → x i + x (i - 2) ≤ x (i - 1)

theorem convex_polyhedron_has_triangle_edges (h_cond : edges_satisfy x) :
  ∃ (a b c : Fin n), a < b ∧ b < c ∧ (x a + x b > x c) := sorry

end convex_polyhedron_has_triangle_edges_l373_373194


namespace sum_first_10_terms_of_abs_seq_eq_105_l373_373948

noncomputable def a_n (n : ℕ) : ℤ := 3*n - 7

def sum_first_three_terms : Prop := (a_n 1 + a_n 2 + a_n 3 = -3)
def product_first_three_terms : Prop := (a_n 1 * a_n 2 * a_n 3 = 8)
def condition_a3_squared_eq_a1a2 : Prop := (a_n 3)^2 = a_n 1 * a_n 2

def abs_term (n : ℕ) : ℤ := abs (a_n n)

def abs_sum (n : ℕ) : ℤ := (List.range n).map abs_term |>.sum

theorem sum_first_10_terms_of_abs_seq_eq_105 : 
  sum_first_three_terms ∧ product_first_three_terms ∧ condition_a3_squared_eq_a1a2 → 
  abs_sum 10 = 105 := by
  sorry

end sum_first_10_terms_of_abs_seq_eq_105_l373_373948


namespace total_potatoes_l373_373659

theorem total_potatoes (Nancy_potatoes : ℕ) (Sandy_potatoes : ℕ) (Andy_potatoes : ℕ) 
  (h1 : Nancy_potatoes = 6) (h2 : Sandy_potatoes = 7) (h3 : Andy_potatoes = 9) : 
  Nancy_potatoes + Sandy_potatoes + Andy_potatoes = 22 :=
by
  -- The proof can be written here
  sorry

end total_potatoes_l373_373659


namespace maximal_length_sequence_l373_373870

theorem maximal_length_sequence :
  ∃ (a : ℕ → ℤ) (n : ℕ), (∀ i, 1 ≤ i → i + 6 ≤ n → (a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) + a (i + 6) > 0)) ∧ 
                          (∀ j, 1 ≤ j → j + 10 ≤ n → (a j + a (j + 1) + a (j + 2) + a (j + 3) + a (j + 4) + a (j + 5) + a (j + 6) + a (j + 7) + a (j + 8) + a (j + 9) + a (j + 10) < 0)) ∧ 
                          n = 16 :=
sorry

end maximal_length_sequence_l373_373870


namespace find_p_max_area_triangle_l373_373518

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373518


namespace sheep_speed_l373_373184

/-- Given:
    * The sheepdog runs at 20 feet per second.
    * The sheep is initially 160 feet away from the sheepdog.
    * The sheepdog catches the sheep in 20 seconds.

   Prove:
    The speed of the sheep is 12 feet per second.
-/
theorem sheep_speed (sheepdog_speed : ℝ) (initial_distance : ℝ) (catch_time : ℝ) : 
    sheepdog_speed = 20 → 
    initial_distance = 160 → 
    catch_time = 20 → 
    (initial_distance + (sheep_speed * catch_time)) = (sheepdog_speed * catch_time) → 
    sheep_speed = 12 :=
by
  intro sheepdog_speed_eq
  intro initial_distance_eq
  intro catch_time_eq
  intro distance_eq
  have h1 : (initial_distance + (sheep_speed * catch_time)) = (sheepdog_speed * catch_time) := distance_eq
  sorry

end sheep_speed_l373_373184


namespace isosceles_trapezoid_theorem_l373_373570

open EuclideanGeometry

variables {Point : Type*} [affine_space Point (euclidean_space ℝ (fin 2))]
variables {A B C D E : Point}
variables (isosceles_trapezoid : isosceles_trapezoid A B C D)
variables (parallels : parallel A D C B)
variables (AB_DC : dist2 A B = dist2 D C)
variables (line_parallel : parallel E D A C)

theorem isosceles_trapezoid_theorem
  (isosceles_trapezoid A B C D)
  (parallels : parallel A D C B)
  (AB_DC : dist2 A B = dist2 D C)
  (line_parallel : parallel E D A C) :
  (triangle_cong A B C B C D) ∧ (dist2 E D * dist2 D C = dist2 A E * dist2 B D) :=
sorry

end isosceles_trapezoid_theorem_l373_373570


namespace quadrilateral_ABCD_r_plus_s_l373_373137

noncomputable def AB_is (AB : Real) (r s : Nat) : Prop :=
  AB = r + Real.sqrt s

theorem quadrilateral_ABCD_r_plus_s :
  ∀ (BC CD AD : Real) (mA mB : ℕ) (r s : ℕ), 
  BC = 7 → 
  CD = 10 → 
  AD = 8 → 
  mA = 60 → 
  mB = 60 → 
  AB_is AB r s →
  r + s = 99 :=
by intros BC CD AD mA mB r s hBC hCD hAD hMA hMB hAB_is
   sorry

end quadrilateral_ABCD_r_plus_s_l373_373137


namespace circle_equation_condition1_circle_equation_condition2_l373_373886

def dist (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem circle_equation_condition1 :
  ∃ r : ℝ, r = dist 1 (-3) (-3) (-1) ∧ ∀ x y : ℝ, (x - 1) ^ 2 + (y + 3) ^ 2 = r^2 :=
by
  sorry

theorem circle_equation_condition2 :
  ∃ r : ℝ, r = dist 2 (-3) 0 (-4) ∧ ∀ x y : ℝ, (x - 2) ^ 2 + (y + 3) ^ 2 = r^2 :=
by
  sorry

end circle_equation_condition1_circle_equation_condition2_l373_373886


namespace find_distance_OA_l373_373965

-- Define the parabola, its properties and the points involved.
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- Assume point A is on the parabola, F is the focus, K is where the directrix intersects x-axis.
-- Given conditions
variable (A F K O : ℝ × ℝ) 
variable (H_A_on_parabola : parabola A.1 A.2)
variable (H_dist_AK_AF : ∥(A.1 - K.1, A.2 - K.2)∥ = real.sqrt 2 * ∥(A.1 - F.1, A.2 - F.2)∥) 
variable (H_O_is_origin : O = (0, 0))
variable (H_K_is_on_x_axis : K.2 = 0)
variable (H_F_is_focus : F = (2, 0)) -- Because the focus of y^2 = 8x is at (2, 0)

-- Prove that |OA| = 2√5.
theorem find_distance_OA : ∥(O.1 - A.1, O.2 - A.2)∥ = 2 * real.sqrt 5 :=
by
  sorry

end find_distance_OA_l373_373965


namespace has_two_extreme_values_l373_373216

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2

theorem has_two_extreme_values : ∃ a b : ℝ, (a ≠ b) ∧ f.derivative a = 0 ∧ f.derivative b = 0 :=
by
  sorry

end has_two_extreme_values_l373_373216


namespace slope_of_l3_l373_373182

noncomputable def point := (ℝ × ℝ)

def line (a b c : ℝ) (p : point) : Prop := a * p.1 + b * p.2 = c

-- Line l1 equation: 4x - 5y = 2
def l1 (p : point) : Prop := line 4 (-5) 2 p

-- Point A
def A : point := (-1, -2)

-- Line l2 equation: y = 3
def l2 (p : point) : Prop := line 0 1 3 p

-- Point B is the intersection of l1 and l2
def is_intersection (L1 L2 : point → Prop) (p : point) : Prop :=
L1 p ∧ L2 p

-- Line l3 with a positive slope, passes through point A
-- We encode a generic line through one point and matching slope
def l3 (m : ℝ) (p : point) : Prop :=
p.2 = m * (p.1 + 1) - 2

-- Area of triangle ABC
def triangle_area (A B C : point) : ℝ :=
(1/2) * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem slope_of_l3 :
(∃ B C : point, is_intersection l1 l2 B ∧ l2 C ∧
(∃ m > 0, l3 m A ∧ l3 m C ∧ triangle_area A B C = 5)) →
  ∃ m, m = 20/29 :=
sorry

end slope_of_l3_l373_373182


namespace smallest_degree_p_l373_373375

def f (x : ℝ) : ℝ := 3 * x ^ 8 + 5 * x ^ 7 - 2 * x ^ 3 + x - 4

theorem smallest_degree_p : ∃ (p : ℝ[X]), (degree p = 8) ∧ (∃ (k : ℝ), ∀ x, f(x) / (p.eval x) → k) :=
    sorry

end smallest_degree_p_l373_373375


namespace new_people_in_country_l373_373149

-- Statement of the problem in Lean 4
theorem new_people_in_country (number_born: ℕ) (number_immigrated: ℕ) (total_new_people: ℕ) :
  number_born = 90171 → 
  number_immigrated = 16320 → 
  total_new_people = number_born + number_immigrated →
  total_new_people = 106491 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3
  -- skip the proof for now
  sorry

end new_people_in_country_l373_373149


namespace number_of_two_digit_integers_l373_373065

def digits : Finset ℕ := {2, 4, 6, 7, 8}

theorem number_of_two_digit_integers : 
  (digits.card * (digits.card - 1)) = 20 := 
by
  sorry

end number_of_two_digit_integers_l373_373065


namespace older_brother_catches_up_l373_373860

theorem older_brother_catches_up :
  ∃ (x : ℝ), 0 ≤ x ∧ 6 * x = 2 + 2 * x ∧ x + 1 < 1.75 :=
by
  sorry

end older_brother_catches_up_l373_373860


namespace at_most_three_digits_count_l373_373094

theorem at_most_three_digits_count : 
  ∃ n, (n < 100000) ∧ 
       (∀ m < n, m ∈ ( ℕ.filter (λ x, (distinct_digits x ≤ 3)))) ∧
       (count_at_most_three_digits n) = 6435 :=
sorry

noncomputable def distinct_digits (x: ℕ): ℕ :=
sorry

noncomputable def count_at_most_three_digits (x: ℕ): ℕ :=
sorry

end at_most_three_digits_count_l373_373094


namespace rhombus_area_l373_373878

theorem rhombus_area (r1 r2 s : ℝ) (h1 : r1 = 15) (h2 : r2 = 30) (h3 : s = 10 * Real.sqrt 5) :
  let x := Real.sqrt ((s^2 * 5) / (1 + Real.pow (s / r1) 2))
  let y := 2 * x
  let area := x * y / 2
  area = 100 :=
sorry

end rhombus_area_l373_373878


namespace total_preparation_and_cooking_time_l373_373350

def time_to_chop_pepper : Nat := 3
def time_to_chop_onion : Nat := 4
def time_to_grate_cheese_per_omelet : Nat := 1
def time_to_cook_omelet : Nat := 5
def num_peppers : Nat := 4
def num_onions : Nat := 2
def num_omelets : Nat := 5

theorem total_preparation_and_cooking_time :
  num_peppers * time_to_chop_pepper +
  num_onions * time_to_chop_onion +
  num_omelets * (time_to_grate_cheese_per_omelet + time_to_cook_omelet) = 50 := 
by
  sorry

end total_preparation_and_cooking_time_l373_373350


namespace domain_of_function_l373_373218

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x
noncomputable def logb (b x : ℝ) : ℝ := Real.log x / Real.log b

def original_function (x : ℝ) : ℝ := sqrt (logb 2 (1 / (x - 3)))

theorem domain_of_function : 
  {x : ℝ | x > 3 ∧ x ≤ 4} = {x | ∃ (y : ℝ), original_function x = y} :=
sorry

end domain_of_function_l373_373218


namespace parabola_circle_distance_l373_373467

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373467


namespace cube_root_equation_solutions_l373_373877

theorem cube_root_equation_solutions (x : ℝ) :
  (∃ (y : ℝ), y = real.cbrt x ∧ y = 15 / (8 - y)) ↔ (x = 27 ∨ x = 125) :=
by
  sorry

end cube_root_equation_solutions_l373_373877


namespace volume_of_solid_rotated_around_y_axis_l373_373429

-- Defining points based on the given conditions
structure Point (α : Type _) := (x : α) (y : α) (z : α)

def O : Point ℝ := ⟨0, 0, 0⟩
def A : Point ℝ := ⟨1, 0, 0⟩
def B : Point ℝ := ⟨1, 1, 0⟩

-- Defining the theorem to prove the required volume
theorem volume_of_solid_rotated_around_y_axis :
  let V := cone_from_rotating_triangle_about_xaxis O A B in
  volume (solid_of_rotating_cone_about_yaxis V) = (8 * Real.pi) / 3 :=
by 
  -- All the required mathematical reasoning would go here
  sorry

end volume_of_solid_rotated_around_y_axis_l373_373429


namespace parabola_focus_distance_max_area_triangle_l373_373559

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373559


namespace eliminate_denominators_l373_373269

theorem eliminate_denominators (x : ℝ) : 
  (6 * (x / 2 - 1) = 6 * ((x - 1) / 3)) → (3 * x - 6 = 2 * (x - 1)) :=
by
  intro h
  have h₁ : 6 * (x / 2) - 6 = 6 * ((x - 1) / 3), from h
  have h₂ : 3 * x - 6 = 2 * (x - 1), from sorry
  exact h₂

end eliminate_denominators_l373_373269


namespace train_speed_kph_l373_373808

-- Define conditions as inputs
def train_time_to_cross_pole : ℝ := 6 -- seconds
def train_length : ℝ := 100 -- meters

-- Conversion factor from meters per second to kilometers per hour
def mps_to_kph : ℝ := 3.6

-- Define and state the theorem to be proved
theorem train_speed_kph : (train_length / train_time_to_cross_pole) * mps_to_kph = 50 :=
by
  sorry

end train_speed_kph_l373_373808


namespace ab_passes_through_fixed_point_min_area_triangle_abc_l373_373424

open Real

-- Definitions for mathematical conditions in Lean
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def point_C := (1, 2)
def on_parabola (A : ℝ × ℝ) (y0 : ℝ) : Prop :=
  A = (y0^2 / 4, y0) ∧ y0 ≠ 2
def line_AC (A : ℝ × ℝ) (x y : ℝ) : Prop :=
  ∃ y0, A = (y0^2 / 4, y0) ∧ y - 2 = (4 (y0 - 2) / (y0^2 - 4)) * (x - 1)
def intersects_P (A : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ x y, y = x + 3 ∧
         y - 2 = (4 * (A.2 - 2) / (A.1 * 4 - 1)) * (x - 1) ∧
         P = (x, y)
def line_parallel_x (P : ℝ × ℝ) (B : ℝ × ℝ) (y0 : ℝ) : Prop :=
  P.2 = B.2 ∧ B = ((y0 - 6)^2 / (y0 - 2)^2, 2*y0 - 12 / (y0 - 2))

-- First proof: line AB passes through a fixed point Q (3, 2)
theorem ab_passes_through_fixed_point :
  ∀ (p : ℝ) (A P B : ℝ × ℝ) (y0 : ℝ),
    parabola p 1 2 →
    on_parabola A y0 →
    line_AC A P.1 P.2 →
    intersects_P A P →
    line_parallel_x P B y0 →
    ∃ Q, Q = (3, 2) ∧ ∀ x y, line_AC B Q.1 Q.2 :=
by sorry

-- Second proof: minimum area of triangle ABC
theorem min_area_triangle_abc :
  ∀ (p : ℝ) (A P B : ℝ × ℝ) (y0 m : ℝ),
    parabola p 1 2 →
    on_parabola A y0 →
    line_AC A P.1 P.2 →
    intersects_P A P →
    line_parallel_x P B y0 →
    ∃ min_area, min_area = 4 * sqrt 2 :=
by sorry

end ab_passes_through_fixed_point_min_area_triangle_abc_l373_373424


namespace roots_eq_solution_l373_373161

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end roots_eq_solution_l373_373161


namespace sum_of_coordinates_B_l373_373666

theorem sum_of_coordinates_B :
  ∃ (x y : ℝ), (3, 5) = ((x + 6) / 2, (y + 8) / 2) ∧ x + y = 2 := by
  sorry

end sum_of_coordinates_B_l373_373666


namespace no_integer_solutions_l373_373386

theorem no_integer_solutions (x y : ℤ) (hx : x ≠ 1) : (x^7 - 1) / (x - 1) ≠ y^5 - 1 :=
by
  sorry

end no_integer_solutions_l373_373386


namespace constant_function_l373_373872

noncomputable def f : ℕ → ℤ := sorry

theorem constant_function (H1 : ∀ a b : ℕ, a ≠ 0 → b ≠ 0 → a ∣ b → f a ≥ f b)
                         (H2 : ∀ a b : ℕ, a ≠ 0  → b ≠ 0 → f(a * b) + f(a^2 + b^2) = f(a) + f(b)) :
                         ∃ m : ℤ, ∀ n : ℕ, n ≠ 0 → f n = m :=
begin
  sorry
end

end constant_function_l373_373872


namespace find_p_max_area_of_triangle_l373_373504

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373504


namespace grooming_time_l373_373619

theorem grooming_time (time_per_dog : ℕ) (num_dogs : ℕ) (days : ℕ) (minutes_per_hour : ℕ) :
  time_per_dog = 20 →
  num_dogs = 2 →
  days = 30 →
  minutes_per_hour = 60 →
  (time_per_dog * num_dogs * days) / minutes_per_hour = 20 := 
by
  intros
  exact sorry

end grooming_time_l373_373619


namespace find_q_l373_373594

noncomputable def p (b : ℝ) : ℝ := b - 9
def q : ℝ := 1  -- Given from the solution that q must be 1
def r : ℝ → ℝ := sorry  -- Radius will be assumed to be a positive real number

axiom area_of_semicircle (r : ℝ) : ℝ := (1/2) * π * r^2
axiom area_of_quadrant (r : ℝ) : ℝ := π * r^2

theorem find_q (b : ℝ) (x q r : ℝ) (h1 : p b = 1) (h2 : x + q = (1/2) * π * r^2) (h3 : 2 * x + p b + q = π * r^2) : q = 1 :=
by
  sorry

end find_q_l373_373594


namespace sqrt_factorial_div_l373_373885

theorem sqrt_factorial_div (h1 : (9!) = 362880)
                           (h2 : 105 = 3 * 5 * 7) :
  Real.sqrt ((9!) / 105) = 24 * Real.sqrt 3 :=
by
  sorry

end sqrt_factorial_div_l373_373885


namespace area_of_parallelogram_is_20_l373_373828

def Point := (ℝ × ℝ)

def base_length (A B : Point) : ℝ :=
  (B.1 - A.1).abs

def height (A C : Point) : ℝ :=
  (C.2 - A.2).abs

def area_parallelogram (A B C D : Point) : ℝ :=
  base_length A B * height A C

theorem area_of_parallelogram_is_20 :
  let A := (0, 0)
  let B := (4, 0)
  let C := (3, 5)
  let D := (7, 5)
  area_parallelogram A B C D = 20 :=
by
  let A := (0, 0)
  let B := (4, 0)
  let C := (3, 5)
  let D := (7, 5)
  show area_parallelogram A B C D = 20
  sorry

end area_of_parallelogram_is_20_l373_373828


namespace three_digit_squares_div_by_4_count_l373_373103

theorem three_digit_squares_div_by_4_count : 
  (finset.card ((finset.filter (λ x, 
    x % 4 = 0) 
    (finset.image (λ n : ℕ, n * n) 
      (finset.range 32)).filter 
        (λ x, 100 ≤ x ∧ x < 1000))) = 11) := 
by 
  sorry

end three_digit_squares_div_by_4_count_l373_373103


namespace min_distance_feasible_region_line_l373_373941

def point (x y : ℝ) : Type := ℝ × ℝ 

theorem min_distance_feasible_region_line :
  ∃ (M N : ℝ × ℝ),
    (2 * M.1 + M.2 - 4 >= 0) ∧
    (M.1 - M.2 - 2 <= 0) ∧
    (M.2 - 3 <= 0) ∧
    (N.2 = -2 * N.1 + 2) ∧
    (dist M N = (2 * Real.sqrt 5)/5) :=
by 
  sorry

end min_distance_feasible_region_line_l373_373941


namespace find_original_width_l373_373143

variable (w : ℝ)

def original_length : ℝ := 13
def increase : ℝ := 2
def new_length := original_length + increase
def new_width := w + increase
def new_area := new_length * new_width
def total_area := 4 * new_area + 2 * new_area
def target_area := 1800

theorem find_original_width (h : total_area = target_area) : w = 18 :=
sorry

end find_original_width_l373_373143


namespace identify_quadratic_l373_373270

/- Definitions for the given equations -/
def eq_A (x : ℝ) : Prop := 2 * x + 1 = 0
def eq_B (x : ℝ) : Prop := x^2 - 3 * x + 1 = 0
def eq_C (x y : ℝ) : Prop := x^2 + y = 1
def eq_D (x : ℝ) : Prop := 1 / (x^2) = 1

/- Conditions for being a quadratic equation -/
def is_quadratic (p : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, p x ↔ a * x^2 + b * x + c = 0)

/- The problem statement -/
theorem identify_quadratic : is_quadratic eq_B ∧ ¬is_quadratic eq_A ∧ ¬is_quadratic (λ x y, eq_C x y) ∧ ¬is_quadratic eq_D :=
by
  sorry

end identify_quadratic_l373_373270


namespace given_eqn_simplification_l373_373904

theorem given_eqn_simplification (x : ℝ) (h : 6 * x^2 - 4 * x - 3 = 0) : 
  (x - 1)^2 + x * (x + 2 / 3) = 2 :=
by
  sorry

end given_eqn_simplification_l373_373904


namespace length_of_bridge_l373_373763

-- Defining the conditions
def speed : ℝ := 10   -- 10 km/hr
def time : ℝ := 1 / 6 -- 10 minutes converted to hours

-- Statement to be proven
theorem length_of_bridge : speed * time = 5 / 3 := by
  sorry

end length_of_bridge_l373_373763


namespace infinite_geometric_series_sum_l373_373863

theorem infinite_geometric_series_sum :
  let a := 4 / 3
  let r := -5 / 16
  |r| < 1 →
  (a / (1 - r) = 64 / 63) :=
by
  intro _
  let a := (4 : ℚ) / 3
  let r := -(5 : ℚ) / 16
  calc
    a / (1 - r) = 64 / 63 : by sorry

end infinite_geometric_series_sum_l373_373863


namespace three_digit_squares_div_by_4_count_l373_373101

theorem three_digit_squares_div_by_4_count : 
  (finset.card ((finset.filter (λ x, 
    x % 4 = 0) 
    (finset.image (λ n : ℕ, n * n) 
      (finset.range 32)).filter 
        (λ x, 100 ≤ x ∧ x < 1000))) = 11) := 
by 
  sorry

end three_digit_squares_div_by_4_count_l373_373101


namespace minutes_to_seconds_l373_373982

theorem minutes_to_seconds (m : ℝ) (hm : m = 6.5) : m * 60 = 390 := by
  sorry

end minutes_to_seconds_l373_373982


namespace BA_eq_M_l373_373154

variable (A B : Matrix (Fin 2) (Fin 2) ℝ)

-- Condition 1: A + B = A * B
axiom h1 : A + B = A ⬝ B

-- Condition 2: A * B = given matrix
def M : Matrix (Fin 2) (Fin 2) ℝ := !![ (10 : ℝ), 5; -4, 6 ]
axiom h2 : A ⬝ B = M

-- Prove B * A = M
theorem BA_eq_M : B ⬝ A = M := by
  sorry

end BA_eq_M_l373_373154


namespace travel_time_l373_373006

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end travel_time_l373_373006


namespace period_of_y_l373_373733

-- Define the function y
def y (x : Real) : Real :=
  Real.sin (2 * x) + Real.cos (2 * x)

-- State the theorem to prove that the period of y is pi
theorem period_of_y : ∃ p > 0, ∀ x : Real, y (x + p) = y x := by
  use Real.pi
  sorry

end period_of_y_l373_373733


namespace range_of_k_l373_373998

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x - 2| + |x - 3| > |k - 1|) → 0 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l373_373998


namespace laptop_price_l373_373183

theorem laptop_price (upfront_percent : ℝ) (upfront_payment full_price : ℝ)
  (h1 : upfront_percent = 0.20)
  (h2 : upfront_payment = 240)
  (h3 : upfront_payment = upfront_percent * full_price) :
  full_price = 1200 := 
sorry

end laptop_price_l373_373183


namespace complex_i_power_l373_373714

theorem complex_i_power (i : ℂ) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : i^2015 = -i := 
by
  sorry

end complex_i_power_l373_373714


namespace num_even_three_digit_l373_373088

def count_even_three_digit_numbers (digits : List ℕ) (is_even : ℕ → Prop) : ℕ :=
  let evens := digits.filter is_even
  let odd := digits.filter (λ x, ¬is_even x)
  evens.length * (odd.permutations.length / 2)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

theorem num_even_three_digit (digits : List ℕ) :
  digits = [1, 2, 3, 4] →
  count_even_three_digit_numbers digits is_even = 12 :=
by
  intros
  -- proof here
  sorry

end num_even_three_digit_l373_373088


namespace find_sin_B_of_ABC_l373_373124

noncomputable theory

variables {A B C a b c : ℝ}
variables {ABC : ∀ A B C a b c, (a + c = 2 * b) ∧ (A - C = Real.pi / 3)}

theorem find_sin_B_of_ABC
  (h1 : a + c = 2 * b)
  (h2 : A - C = Real.pi / 3) :
  Real.sin B = Real.sqrt 39 / 8 := 
sorry

end find_sin_B_of_ABC_l373_373124


namespace Nadia_distance_is_18_l373_373187

-- Variables and conditions
variables (x : ℕ)

-- Definitions based on conditions
def Hannah_walked (x : ℕ) : ℕ := x
def Nadia_walked (x : ℕ) : ℕ := 2 * x
def total_distance (x : ℕ) : ℕ := Hannah_walked x + Nadia_walked x

-- The proof statement
theorem Nadia_distance_is_18 (h : total_distance x = 27) : Nadia_walked x = 18 :=
by
  sorry

end Nadia_distance_is_18_l373_373187


namespace projection_of_difference_eq_l373_373992

noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vec_projection (v w : ℝ × ℝ) : ℝ :=
vec_dot (v - w) v / vec_magnitude v

variables (a b : ℝ × ℝ)
  (congruence_cond : vec_magnitude a / vec_magnitude b = Real.cos θ)

theorem projection_of_difference_eq (h : vec_magnitude a / vec_magnitude b = Real.cos θ) :
  vec_projection (a - b) a = (vec_dot a a - vec_dot b b) / vec_magnitude a :=
sorry

end projection_of_difference_eq_l373_373992


namespace f_sqrt_50_l373_373177

def f (x : ℝ) : ℝ := 
  if x ∈ (Set.Icc 7.07 7.08) 
  then ⌊real.sqrt 50⌋ + 6 
  else 7 * x + 3

theorem f_sqrt_50 : f (real.sqrt 50) = 13 := 
by
  sorry

end f_sqrt_50_l373_373177


namespace dot_product_is_4_l373_373568

-- Define the vectors a and b
def a (k : ℝ) : ℝ × ℝ := (1, k)
def b : ℝ × ℝ := (2, 2)

-- Define collinearity condition
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Define k based on the collinearity condition
def k_value : ℝ := 1 -- derived from solving the collinearity condition in the problem

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of a and b is 4 when k = 1
theorem dot_product_is_4 {k : ℝ} (h : k = k_value) : dot_product (a k) b = 4 :=
by
  rw [h]
  sorry

end dot_product_is_4_l373_373568


namespace angle_B_contradiction_l373_373730

variables (A B C : Type) [is_triangle : triangle A B C]
variable (H1 : side_eq A B C) -- condition AB = AC
noncomputable def angle_B_lt_90 : Prop :=
  angle B < 90

theorem angle_B_contradiction (H : angle B ≥ 90) : false :=
sorry

example : angle_B_lt_90 A B C :=
by
  intro H
  exact H1
  apply not_le
  contradiction
  sorry

end angle_B_contradiction_l373_373730


namespace max_area_of_rotating_lines_l373_373139

noncomputable def max_triangle_area : ℕ := sorry

theorem max_area_of_rotating_lines :
  let A := (0, 0)
  let B := (8, 0)
  let C := (15, 0)
  let theta := 45 * (π/180) -- converting 45 degrees to radians for Lean
  let l_A_slope := -1
  let l_B_vertical := B.fst
  let l_C_slope := 1
  let l_A := fun x => - x * Real.tan(theta)
  let l_B := fun y => 8
  let l_C := fun x => x * Real.tan(theta) + 15 * Real.tan(theta)
  let Z := (8, -8 * Real.tan(theta))
  let Y := (-15 / 2, (15 / 2) * Real.tan(theta))
  let X := (8, 23 * Real.tan(theta))
  let area := abs(8 * (23 * Real.tan(theta) - (-8 * Real.tan(theta))) / 2)
  in area = 124 := sorry

end max_area_of_rotating_lines_l373_373139


namespace faye_total_crayons_l373_373384

theorem faye_total_crayons (rows crayons_per_row : ℕ) (h1 : rows = 16) (h2 : crayons_per_row = 6) : rows * crayons_per_row = 96 := by
  simp [h1, h2]
  sorry

end faye_total_crayons_l373_373384


namespace absolute_value_exponent_cosine_l373_373775

theorem absolute_value_exponent_cosine : 
  (|(-3 : ℝ)| + 2 ^ (-1 : ℝ) - real.cos (real.pi / 3) = 3) :=
by
  have h1 : |(-3 : ℝ)| = 3 := abs_neg (3 : ℝ)
  have h2 : 2 ^ (-1 : ℝ) = 1 / 2 := rfl
  have h3 : real.cos (real.pi / 3) = 1 / 2 := real.cos_pi_div_three
  sorry

end absolute_value_exponent_cosine_l373_373775


namespace tony_age_in_6_years_l373_373620

theorem tony_age_in_6_years (jacob_age : ℕ) (tony_age : ℕ) (h : jacob_age = 24) (h_half : tony_age = jacob_age / 2) : (tony_age + 6) = 18 :=
by
  sorry

end tony_age_in_6_years_l373_373620


namespace proof_part1_proof_part2_l373_373481

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373481


namespace simple_interest_rate_l373_373823

theorem simple_interest_rate (P A T : ℝ) (H1 : P = 1750) (H2 : A = 2000) (H3 : T = 4) :
  ∃ R : ℝ, R = 3.57 ∧ A = P * (1 + (R * T) / 100) :=
by
  sorry

end simple_interest_rate_l373_373823


namespace number_of_arrangements_l373_373887

-- Define the students
inductive Student
| A | B | C | D | E

open Student

def is_not_at_end (l : List Student) : Prop :=
  match l with
  | [] | [_] => False
  | x::y::xs => (x ≠ A) ∧ (List.last? (y::xs) ≠ some A)

def are_adjacent_C_D (l : List Student) : Prop :=
  l == [C, D] ++ l.drop 2 ∨ l == D::C::l.drop 2

def count_valid_arrangements (students : List Student) : Nat :=
  -- Count arrangements where A is not at either end and C and D are adjacent
  sorry

theorem number_of_arrangements : count_valid_arrangements [A, B, C, D, E] = 24 :=
  sorry

end number_of_arrangements_l373_373887


namespace rate_of_stream_l373_373798

-- Definitions from problem conditions
def rowing_speed_still_water : ℕ := 24

-- Assume v is the rate of the stream
variable (v : ℕ)

-- Time taken to row up is three times the time taken to row down
def rowing_time_condition : Prop :=
  1 / (rowing_speed_still_water - v) = 3 * (1 / (rowing_speed_still_water + v))

-- The rate of the stream (v) should be 12 kmph
theorem rate_of_stream (h : rowing_time_condition v) : v = 12 :=
  sorry

end rate_of_stream_l373_373798


namespace ordered_pairs_satisfy_equation_l373_373092

theorem ordered_pairs_satisfy_equation :
  {p : ℤ × ℤ | p.1^2 + p.2^2 = 2 * (p.1 + p.2) + p.1 * p.2}.to_finset.card = 6 :=
sorry

end ordered_pairs_satisfy_equation_l373_373092


namespace collinear_iff_collinear_sum_diff_l373_373433

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (a b : V) : Prop :=
  ∃ (λ : ℝ), b = λ • a

variables (a b : V)

theorem collinear_iff_collinear_sum_diff (ha : a ≠ 0) (hb : b ≠ 0) :
  collinear a b ↔ collinear (a + b) (a - b) :=
sorry

end collinear_iff_collinear_sum_diff_l373_373433


namespace range_of_alpha_minus_beta_l373_373581

open Real

theorem range_of_alpha_minus_beta (
    α β : ℝ) 
    (h1 : -π / 2 < α) 
    (h2 : α < 0)
    (h3 : 0 < β)
    (h4 : β < π / 3)
  : -5 * π / 6 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l373_373581


namespace guess_number_l373_373814

/-- 
  Given three individuals A, B, and C with statements about a number:

  - A states: "The number is even and less than 6."
  - B states: "The number is less than 7 and it is a two-digit number."
  - C states: "The first part of A's statement is true, and the second part is false."

  We know that among A, B, and C:
  - One tells two truths,
  - One tells two lies,
  - One tells one truth and one lie.

  Prove: The number is 8.
-/
theorem guess_number :
  ∃ (n : ℕ),
  (1 ≤ n ∧ n ≤ 99) ∧ -- n is a natural number between 1 and 99
  ((n % 2 = 0 ∧ n < 6) ∨ (¬ (n % 2 = 0) ∨ ¬ (n < 6))) ∧ -- A's statement
  ((n < 7 ∧ 10 ≤ n ∧ n < 100) ∨ (¬ (n < 7) ∨ ¬ (10 ≤ n ∧ n < 100))) ∧ -- B's statement
  ((n % 2 = 0 ∧ n ≥ 6) ∨ (¬ (n % 2 = 0) ∨ ¬ (n ≥ 6))) ∧ -- C's statement
  (n = 8) :=   -- We need to prove the number is 8
by
  intro n,
  sorry

end guess_number_l373_373814


namespace range_of_x_for_inequality_l373_373934

variable {f : ℝ → ℝ}

-- f is an odd function: f(-x) = -f(x) for all x
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Conditions given in the problem
variable {cond1 : is_odd_function f}
variable {cond2 : ∀ x, 0 < x → ln x * (deriv^[2] f x) < -((1 / x) * f x)}

-- The range of values of x for which (x^2 - 4) * f x > 0
theorem range_of_x_for_inequality :
  (∀ x, x ∈ (-∞, -2) ∪ (0, 2) → (x^2 - 4) * f x > 0) :=
by
  sorry

end range_of_x_for_inequality_l373_373934


namespace f_monotonically_increasing_on_interval_l373_373226

-- Define the function f(x) = sin x + cos x
def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Define the interval of interest
def interval_of_interest : Set ℝ := Set.Icc 0 Real.pi

-- Define the monotonically increasing interval we want to prove
def monotonically_increasing_interval : Set ℝ := Set.Icc 0 (Real.pi / 4)

-- The theorem to prove
theorem f_monotonically_increasing_on_interval :
    ∀ x y : ℝ, x ∈ interval_of_interest → y ∈ interval_of_interest → x ≤ y → f x ≤ f y :=
sorry

end f_monotonically_increasing_on_interval_l373_373226


namespace melanie_turnips_l373_373185

theorem melanie_turnips (benny_turnips total_turnips melanie_turnips : ℕ) 
  (h1 : benny_turnips = 113) 
  (h2 : total_turnips = 252) 
  (h3 : total_turnips = benny_turnips + melanie_turnips) : 
  melanie_turnips = 139 :=
by
  sorry

end melanie_turnips_l373_373185


namespace proof_of_f_inverse_l373_373072

noncomputable def f : ℝ → ℝ :=
λ x, if 0 < x ∧ x < 1 then x ^ (1 / 3) else if x ≥ 1 then 4 * (x - 1) else 0

theorem proof_of_f_inverse (a : ℝ) (h1 : 0 < a ∧ a < 1) (h2 : f a = f (a + 1)) : f (1 / a) = 28 := 
by 
  sorry

end proof_of_f_inverse_l373_373072


namespace parabola_circle_distance_l373_373468

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373468


namespace minimum_loadings_to_prove_first_ingot_weighs_1kg_l373_373148

theorem minimum_loadings_to_prove_first_ingot_weighs_1kg :
  ∀ (w : Fin 11 → ℕ), 
    (∀ i, w i = i + 1) →
    (∃ s₁ s₂ : Finset (Fin 11), 
       s₁.card ≤ 6 ∧ s₂.card ≤ 6 ∧ 
       s₁.sum w = 11 ∧ s₂.sum w = 11 ∧ 
       (∀ s : Finset (Fin 11), s.sum w = 11 → s ≠ s₁ ∧ s ≠ s₂) ∧
       (w 0 = 1)) := sorry -- Fill in the proof here

end minimum_loadings_to_prove_first_ingot_weighs_1kg_l373_373148


namespace smallest_positive_period_and_monotonic_interval_max_value_in_interval_l373_373070

def f (x : ℝ) : ℝ := (sin (π + x) - sqrt 3 * cos x * sin (2 * x)) / (2 * cos (π - x)) - 1 / 2

-- Proof Problem 1: Smallest positive period and monotonically decreasing intervals
theorem smallest_positive_period_and_monotonic_interval :
  (∀ x, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x ∈ [π / 3 + k * π, π / 2 + k * π], f x ≤ f (π / 3 + k * π)) ∧
  (∀ k : ℤ, ∀ x ∈ [π / 2 + k * π, 5 * π / 6 + k * π], f x ≤ f (π / 2 + k * π)) :=
by sorry

-- Proof Problem 2: Maximum value and corresponding x in the interval
theorem max_value_in_interval :
  (∀ x ∈ (0, π / 2), f x ≤ 1) ∧ (f (π / 3) = 1) :=
by sorry

end smallest_positive_period_and_monotonic_interval_max_value_in_interval_l373_373070


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373492

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373492


namespace total_number_of_cats_l373_373985

def Cat := Type -- Define a type of Cat.

variable (A B C: Cat) -- Declaring three cats A, B, and C.

variable (kittens_A: Fin 4 → {gender : Bool // (2 : Fin 4).val = 2 ∧ (2 : Fin 4).val = 2}) -- 4 kittens: 2 males, 2 females.
variable (kittens_B: Fin 3 → {gender : Bool // (1 : Fin 3).val = 1 ∧ (2 : Fin 3).val = 2}) -- 3 kittens: 1 male, 2 females.
variable (kittens_C: Fin 5 → {gender : Bool // (3 : Fin 5).val = 3 ∧ (2 : Fin 5).val = 2}) -- 5 kittens: 3 males, 2 females.

variable (extra_kittens: Fin 2 → {gender : Bool // (1 : Fin 2).val = 1 ∧ (1 : Fin 2).val = 1}) -- 2 kittens of the additional female kitten of Cat A.

theorem total_number_of_cats : 
  3 + 4 + 2 + 3 + 5 = 17 :=
by
  sorry

end total_number_of_cats_l373_373985


namespace ellipse_equation_l373_373695

variable {a b c : ℝ}
variable (eccentricity : ℝ := 1 / 2)
variable (directrix : ℝ := 4)
variable (origin_center : Prop := true)

-- Define the conditions
def semi_major_rel_directrix (a c : ℝ) : Prop := a^2 = 4 * c
def semi_major_rel_eccentricity (a c : ℝ) : Prop := c / a = 1 / 2

-- The ellipse equation given the correct semi-major and semi-minor axes
theorem ellipse_equation : 
  origin_center → 
  (∃ a c, semi_major_rel_directrix a c ∧ semi_major_rel_eccentricity a c ∧ a = 2 * c ∧ c = 1) → 
  ∃ b, b = sqrt (a^2 - c^2) → (a = 2 → b = sqrt 3) → 
  (a = 2 ∧ b = sqrt 3) → (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → (x^2 / 4 + y^2 / 3 = 1)) := 
by
  intros h₀ ⟨a, c, h₁, h₂, h₃, h₄⟩ ⟨b, h₅, h₆⟩ h₇
  sorry

end ellipse_equation_l373_373695


namespace tagged_fish_in_second_catch_l373_373129

-- Definitions and conditions
def total_fish_in_pond : ℕ := 1750
def tagged_fish_initial : ℕ := 70
def fish_caught_second_time : ℕ := 50
def ratio_tagged_fish : ℚ := tagged_fish_initial / total_fish_in_pond

-- Theorem statement
theorem tagged_fish_in_second_catch (T : ℕ) : (T : ℚ) / fish_caught_second_time = ratio_tagged_fish → T = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l373_373129


namespace hyperbola_vertices_distance_l373_373393

theorem hyperbola_vertices_distance :
  ∀ (x y : ℝ), ((y^2 / 16) - (x^2 / 9) = 1) → 
    2 * real.sqrt 16 = 8 :=
by
  intro x y h
  have a2 : real.sqrt 16 = 4 := by norm_num
  have h2a : 2 * 4 = 8 := by norm_num
  rw [←a2, h2a]
  sorry

end hyperbola_vertices_distance_l373_373393


namespace find_f_log_value_l373_373422

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_log_value :
  (∀ x: ℝ, f(-x) = -f(x)) →                  -- f is an odd function
  (∀ x: ℝ, f(x + 2) = -f(x)) →              -- f(x + 2) = -f(x)
  (∀ x ∈ Icc (0:ℝ) 1, f x = 2^x - 1) →      -- f(x) = 2^x - 1 for x in [0,1]
  f (Real.logb (1/2) 24) = -1/2 := 
by
  intros h_odd h_shift h_xrange
  sorry

end find_f_log_value_l373_373422


namespace ellipse_related_proof_l373_373064

noncomputable def ellipseEquation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) : Prop :=
  ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 → true

noncomputable def isEccentricity (a e : ℝ) (c : ℝ) : Prop :=
  e = c / a

noncomputable def tangentCondition (E : ℝ × ℝ) (a b : ℝ) : Prop :=
  (E = (-real.sqrt 7, 0)) →
  ∀ (m1 m2 : ℝ), (m1 * m2 = -1) ∧ (∃ (x y : ℝ), (y = x + real.sqrt 7) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1))

noncomputable def lineIntersectingEllipse (t : ℝ) (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), (x = m * y + t) → (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def perpendicularCondition (t : ℝ) (t_valid : Prop) : Prop :=
  ∀ (F : ℝ × ℝ) (A B : ℝ × ℝ),
  (F = (1, 0)) →
  t ∈ (set.Icc (-(real.sqrt(7))/2) (real.sqrt(7)/2)) → -- This assumed range
  A ≠ B →
  ∀ (x1 y1 x2 y2 : ℝ), (A = (x1, y1)) ∧ (B = (x2, y2)) ∧
  (FA = (x1 - 1, y1)) ∧ (FB = (x2 - 1, y2)) →
  (FA.1 * FB.1 + FA.2 * FB.2 = 0)

theorem ellipse_related_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (h4 : isEccentricity a (1 / 2) c)
  (h5 : tangentCondition (-real.sqrt 7, 0) a b)
  (t : ℝ)
  (t_valid : Prop) :
  ellipseEquation 2 (real.sqrt 3) h1 h2 h3 ∧
  perpendicularCondition t t_valid := by sorry

end ellipse_related_proof_l373_373064


namespace at_least_one_female_team_l373_373029

open Classical

namespace Probability

-- Define the Problem
noncomputable def prob_at_least_one_female (females males : ℕ) (team_size : ℕ) :=
  let total_students := females + males
  let total_ways := Nat.choose total_students team_size
  let ways_all_males := Nat.choose males team_size
  1 - (ways_all_males / total_ways : ℝ)

-- Verify the given problem against the expected answer
theorem at_least_one_female_team :
  prob_at_least_one_female 1 3 2 = 1 / 2 := by
  sorry

end Probability

end at_least_one_female_team_l373_373029


namespace tan_sq_sum_geq_l373_373437

theorem tan_sq_sum_geq : 
  ∀ (α β γ : ℝ), 0 < α ∧ α < β ∧ β < γ ∧ γ < π / 2 ∧ 
                ∑ i in [α, β, γ], sin i ^ 3 = 1 
  →
  ( ∑ i in [α, β, γ], (sin i ^ (2/3)) / (1 - sin i ^ (2/3)) ) 
  ≥ (3 / (real.cbrt 9 - 1)) :=
by
  intros
  sorry

end tan_sq_sum_geq_l373_373437


namespace divisible_by_4_l373_373175

theorem divisible_by_4 (n : ℕ) (x : ℕ → ℤ) 
  (h1 : ∀i, x i = 1 ∨ x i = -1)
  (h2 : (∑ i in finset.range n, x i * x (i + 1) * x (i + 2) * x (i + 3)) = 0) 
  : ∃ k, n = 4 * k := 
by
  sorry

end divisible_by_4_l373_373175


namespace remainder_of_large_number_l373_373748

theorem remainder_of_large_number (n : ℕ) (r : ℕ) (h : n = 2468135792) :
  (n % 101) = 52 := 
by
  have h1 : (10 ^ 8 - 1) % 101 = 0 := sorry
  have h2 : (10 ^ 6 - 1) % 101 = 0 := sorry
  have h3 : (10 ^ 4 - 1) % 101 = 0 := sorry
  have h4 : (10 ^ 2 - 1) % 101 = 99 % 101 := sorry

  -- Using these properties to simplify n
  have n_decomposition : 2468135792 = 24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 := sorry
  have div_property : 
    (24 * 10 ^ 8 + 68 * 10 ^ 6 + 13 * 10 ^ 4 + 57 * 10 ^ 2 + 92 - (24 + 68 + 13 + 57 + 92)) % 101 = 0 := sorry

  have simplified_sum : (24 + 68 + 13 + 57 + 92 = 254 := by norm_num) := sorry
  have resulting_mod : 254 % 101 = 52 := by norm_num

  -- Thus n % 101 = 52
  exact resulting_mod

end remainder_of_large_number_l373_373748


namespace find_p_max_area_triangle_l373_373539

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373539


namespace triangle_is_isosceles_find_perimeter_l373_373432

theorem triangle_is_isosceles 
  (A B C a b c : ℝ) 
  (h1 : (B = 3.14159 / 6)) 
  (h2 : (a = c))
  (h3 : (1 / 2 * a * c * real.sin B = 4))
  (h4 : real.tan B * (real.cos A - real.cos C) = real.sin C - real.sin A) : 
  is_isosceles A B C :=
sorry -- Proof to be filled

theorem find_perimeter
  (A B C a b c : ℝ)
  (h1 : (B = 3.14159 / 6)) 
  (h2 : (a = c))
  (h3 : (1 / 2 * a * c * real.sin B = 4))
  (h4 : real.tan B * (real.cos A - real.cos C) = real.sin C - real.sin A) :
  ∃ P, P = 8 + 2 * real.sqrt 6 - 2 * real.sqrt 2 :=
sorry -- Proof to be filled

end triangle_is_isosceles_find_perimeter_l373_373432


namespace find_p_max_area_of_triangle_l373_373502

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373502


namespace complement_of_A_l373_373972

open Set

noncomputable def U : Set ℝ := Icc 0 1
noncomputable def A : Set ℝ := Ioo 0 (1/3)

theorem complement_of_A :
  (U \ A) = {x | x = 0 ∨ (1 / 3 ≤ x ∧ x ≤ 1)} := by
sory

end complement_of_A_l373_373972


namespace monotonic_intervals_f_monotonic_intervals_g_max_min_values_h_number_of_real_roots_l373_373945

variable (a m : ℝ) (n : ℕ)
variable (x : ℝ)
variable (f g h : ℝ → ℝ)
variable (h : (ℝ → ℝ))

-- Define f and g functions
def f (x : ℝ) : ℝ := x^2 + a / x^2
def g (x : ℝ) : ℝ := x^n + a / x^n

-- Define h function
noncomputable def h (x : ℝ) : ℝ := (x^2 + 1/x)^3 + (x + 1/x^2)^3

-- Assumptions
axiom a_pos : a > 0
axiom n_geq_3 : n ≥ 3
axiom m_in_range : 0 < m ∧ m ≤ 30
axiom x_in_range : x ∈ set.Icc (1/2 : ℝ) 2

-- Questions as Lean statements
-- Part 1: Prove monotonic intervals for f and g
theorem monotonic_intervals_f : (∀ x, x ∈ set.Ioo 0 (real.sqrt (real.sqrt a)) → f x < f (real.sqrt (real.sqrt a))) ∧ 
                                 (∀ x, x ∈ set.Ioi (real.sqrt (real.sqrt a)) → f x > f (real.sqrt (real.sqrt a))) := sorry

theorem monotonic_intervals_g : (∀ x, x ∈ set.Ioo 0 (real.sqrt (a^(1/(2*n)))) → g x < g (real.sqrt (a^(1/(2*n))))) ∧ 
                                 (∀ x, x ∈ set.Ioi (real.sqrt (a^(1/(2*n)))) → g x > g (real.sqrt (a^(1/(2*n))))) := sorry

-- Part 2: Prove maximum and minimum values of h in the interval
theorem max_min_values_h : (∀ x ∈ set.Icc (1/2 : ℝ) 2, h x ≥ 16 ∧ h x ≤ 6561 / 64) ∧
                          (∃ x, x = 1 ∧ h x = 16) ∧
                          (∃ x, x ∈ {1/2, 2} ∧ h x = 6561 / 64) := sorry

-- Part 3: Number of real roots
theorem number_of_real_roots : (0 < m < 8 → (∀ x, h x ≠ m ∧ h x ≠ 2*m)) ∧ 
                               (m = 8 → (∃ x, h x = m ∧ h x ≠ 2*m) ∨ (h x = 2*m ∧ h x ≠ m)) ∧
                               (8 < m < 16 → (∃ x1 x2, h x1 = m ∧ h x2 = 2*m)) ∧
                               (m = 16 → (∃ x1 x2 x3, h x1 = m ∧ h x2 = m ∧ h x3 = 2*m)) ∧
                               (16 < m ≤ 30 → (∃ x1 x2 x3 x4, h x1 = m ∧ h x2 = m ∧ h x3 = 2*m ∧ h x4 = 2*m)) := sorry

end monotonic_intervals_f_monotonic_intervals_g_max_min_values_h_number_of_real_roots_l373_373945


namespace second_share_interest_rate_is_11_l373_373339

noncomputable def calculate_interest_rate 
    (total_investment : ℝ)
    (amount_in_second_share : ℝ)
    (interest_rate_first : ℝ)
    (total_interest : ℝ) : ℝ := 
  let A := total_investment - amount_in_second_share
  let interest_first := (interest_rate_first / 100) * A
  let interest_second := total_interest - interest_first
  (100 * interest_second) / amount_in_second_share

theorem second_share_interest_rate_is_11 :
  calculate_interest_rate 100000 12499.999999999998 9 9250 = 11 := 
by
  sorry

end second_share_interest_rate_is_11_l373_373339


namespace juniors_involved_in_sports_l373_373237

theorem juniors_involved_in_sports 
    (total_students : ℕ) (percentage_juniors : ℝ) (percentage_sports : ℝ) 
    (H1 : total_students = 500) 
    (H2 : percentage_juniors = 0.40) 
    (H3 : percentage_sports = 0.70) : 
    total_students * percentage_juniors * percentage_sports = 140 := 
by
  sorry

end juniors_involved_in_sports_l373_373237


namespace pages_in_book_l373_373819

theorem pages_in_book (P : ℕ) 
  (h1 : 150 < P)
  (h2 : 0.80 * 0.70 * (P - 150) = 196) : P = 500 :=
sorry

end pages_in_book_l373_373819


namespace find_solutions_l373_373875

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem find_solutions :
    {x : ℝ | cuberoot x = 15 / (8 - cuberoot x)} = {125, 27} :=
by
  sorry

end find_solutions_l373_373875


namespace magnitude_of_vector_a_l373_373078

theorem magnitude_of_vector_a : 
  let a : ℝ × ℝ := (1, 2)
  in real.sqrt (a.1 ^ 2 + a.2 ^ 2) = real.sqrt 5 := 
by
  -- Proof goes here
  sorry

end magnitude_of_vector_a_l373_373078


namespace total_number_of_notes_l373_373312

theorem total_number_of_notes (x : ℕ) (h₁ : 37 * 50 + x * 500 = 10350) : 37 + x = 54 :=
by
  -- We state that the total value of 37 Rs. 50 notes plus x Rs. 500 notes equals Rs. 10350.
  -- According to this information, we prove that the total number of notes is 54.
  sorry

end total_number_of_notes_l373_373312


namespace min_x_plus_y_l373_373909

theorem min_x_plus_y (x y : ℝ) (h1 : x * y = 2 * x + y + 2) (h2 : x > 1) :
  x + y ≥ 7 :=
sorry

end min_x_plus_y_l373_373909


namespace combined_cost_l373_373839

theorem combined_cost (wallet_cost : ℕ) (purse_cost : ℕ)
    (h_wallet_cost : wallet_cost = 22)
    (h_purse_cost : purse_cost = 4 * wallet_cost - 3) :
    wallet_cost + purse_cost = 107 :=
by
  rw [h_wallet_cost, h_purse_cost]
  norm_num
  sorry

end combined_cost_l373_373839


namespace parabola_circle_distance_l373_373464

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373464


namespace largest_prime_factor_of_891_l373_373251

theorem largest_prime_factor_of_891 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 891 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ 891 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_891_l373_373251


namespace sum_of_all_5_digit_integers_l373_373255

theorem sum_of_all_5_digit_integers : 
  ∑ n in finset.filter (λ x : ℕ, 10000 ≤ x ∧ x < 100000 ∧ ∀ d ∈ digits 10 x, d ∈ {1, 2, 3, 4, 5, 6}) finset.Ico 10000 100000 = 30176496 :=
sorry

end sum_of_all_5_digit_integers_l373_373255


namespace max_gcd_2015xy_l373_373395

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end max_gcd_2015xy_l373_373395


namespace probability_X_in_interval_l373_373883

noncomputable def CDF : ℝ → ℝ
| x := if x < 0 then 0 else if x ≤ 2 then x / 2 else 1

theorem probability_X_in_interval : (CDF 1) - (CDF 0) = 1 / 2 := 
by
  -- definitions/conditions from part a)
  have h1 : CDF 1 = 1 / 2 := by simp [CDF]; norm_num
  have h0 : CDF 0 = 0 := by simp [CDF]; norm_num
  -- conclude the result
  rw [h1, h0]
  norm_num
  --skip proof with sorry
  sorry

end probability_X_in_interval_l373_373883


namespace number_of_whole_numbers_in_interval_l373_373575

theorem number_of_whole_numbers_in_interval :
  let lb := Real.sqrt 3
  let ub := 3 * Real.exp 1
  lb < ub ∧
  lb.ceil ≤ ub.floor ∧
  (∀ x, lb.ceil ≤ x ∧ x ≤ ub.floor → x ∈ Int) →
  (ub.floor - lb.ceil + 1) = 7 :=
by
  let lb := Real.sqrt 3
  let ub := 3 * Real.exp 1
  have h1 : lb < ub := by sorry
  have h2 : lb.ceil ≤ ub.floor := by sorry
  have h3 : ∀ x, lb.ceil ≤ x ∧ x ≤ ub.floor → x ∈ Int := by sorry
  have h4 : (ub.floor - lb.ceil + 1) = 7 := by sorry
  exact ⟨h1, h2, h3, h4⟩

end number_of_whole_numbers_in_interval_l373_373575


namespace problem_value_l373_373754

theorem problem_value :
  4 * (8 - 3) / 2 - 7 = 3 := 
by
  sorry

end problem_value_l373_373754


namespace cg_length_l373_373617

-- Define the geometric setup and conditions
variables {A B C D E F G : Point}
variables {ω : Circle}

-- Assume the essential conditions
variables (h1 : ∠ BAC = 60)
variables (h2 : ω.tangent_to_segment AB D)
variables (h3 : ω.tangent_to_segment AC E)
variables (h4 : ω.intersects_segment_at BC F G ∧ F.between B G)
variables (h5 : AD = 4)
variables (h6 : FG = 4)
variables (h7 : BF = 1/2)

-- Define the proof problem equivalent statement
theorem cg_length :
  CG = 16/5 :=
sorry

end cg_length_l373_373617


namespace percentage_is_40_l373_373114

variables (num : ℕ) (perc : ℕ)

-- Conditions
def ten_percent_eq_40 : Prop := 10 * num = 400
def certain_percentage_eq_160 : Prop := perc * num = 160 * 100

-- Statement to prove
theorem percentage_is_40 (h1 : ten_percent_eq_40 num) (h2 : certain_percentage_eq_160 num perc) : perc = 40 :=
sorry

end percentage_is_40_l373_373114


namespace parabola_condition_max_area_triangle_l373_373487

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373487


namespace modulus_of_z_l373_373951

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the given complex number z
def z : ℂ := i * (2 - i)

-- State the theorem to prove the modulus |z| of the complex number z is √5
theorem modulus_of_z : complex.abs z = real.sqrt 5 :=
  sorry

end modulus_of_z_l373_373951


namespace cost_price_of_a_ball_l373_373662

variables (C : ℝ) (selling_price : ℝ) (cost_price_20_balls : ℝ) (loss_on_20_balls : ℝ)

def cost_price_per_ball (C : ℝ) := (20 * C - 720 = 5 * C)

theorem cost_price_of_a_ball :
  (∃ C : ℝ, 20 * C - 720 = 5 * C) -> (C = 48) := 
by
  sorry

end cost_price_of_a_ball_l373_373662


namespace juniors_involved_in_sports_l373_373236

theorem juniors_involved_in_sports 
    (total_students : ℕ) (percentage_juniors : ℝ) (percentage_sports : ℝ) 
    (H1 : total_students = 500) 
    (H2 : percentage_juniors = 0.40) 
    (H3 : percentage_sports = 0.70) : 
    total_students * percentage_juniors * percentage_sports = 140 := 
by
  sorry

end juniors_involved_in_sports_l373_373236


namespace probability_of_picking_red_ball_l373_373238

noncomputable def basketA_white := 10
noncomputable def basketA_red := 5
noncomputable def basketB_yellow := 4
noncomputable def basketB_red := 6
noncomputable def basketB_black := 5
noncomputable def prob_A := 0.6
noncomputable def prob_B := 0.4

theorem probability_of_picking_red_ball :
  let P_A_red := basketA_red / (basketA_white + basketA_red)
  let P_B_red := basketB_red / (basketB_yellow + basketB_red + basketB_black)
  let P_red := P_A_red * prob_A + P_B_red * prob_B
  P_red = 0.36 :=
by
  sorry

end probability_of_picking_red_ball_l373_373238


namespace number_of_men_in_first_group_l373_373300

variable (M : ℕ)

-- Given conditions
axiom cond1 : ∀ M : ℕ, (M * 96 = 40 * 60)

-- Proof problem
theorem number_of_men_in_first_group (M : ℕ) : M = 25 :=
by
  -- condition: M * 96 = 40 * 60
  have h1 : M * 96 = 40 * 60 := cond1 M
  -- since M * 96 = 40 * 60, we can derive the value of M as follows
  calc
    M = 40 * 60 / 96 : by sorry
    ... = 25        : by sorry

end number_of_men_in_first_group_l373_373300


namespace main_inequality_equality_condition_l373_373645

variable {a b c : ℝ}
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem main_inequality 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 3 / (1 + a * b * c)) :=
  sorry

theorem equality_condition 
  (hpos_a : 0 < a) 
  (hpos_b : 0 < b) 
  (hpos_c : 0 < c) :
  (1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) = 3 / (1 + a * b * c) ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end main_inequality_equality_condition_l373_373645


namespace train_speed_kph_l373_373810

-- Define conditions as inputs
def train_time_to_cross_pole : ℝ := 6 -- seconds
def train_length : ℝ := 100 -- meters

-- Conversion factor from meters per second to kilometers per hour
def mps_to_kph : ℝ := 3.6

-- Define and state the theorem to be proved
theorem train_speed_kph : (train_length / train_time_to_cross_pole) * mps_to_kph = 50 :=
by
  sorry

end train_speed_kph_l373_373810


namespace min_value_inequality_l373_373649

theorem min_value_inequality (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 9)
  : (a^2 + b^2 + c^2)/(a + b + c) + (b^2 + c^2)/(b + c) + (c^2 + a^2)/(c + a) + (a^2 + b^2)/(a + b) ≥ 12 :=
by
  sorry

end min_value_inequality_l373_373649


namespace february_has_greatest_percentage_diff_l373_373380

/-- Data for sales of Drummers, Bugle players, and Clarinetists for each month --/
structure SalesData :=
  (D_J : ℕ) (B_J : ℕ) (C_J : ℕ) -- January
  (D_F : ℕ) (B_F : ℕ) (C_F : ℕ) -- February
  (D_M : ℕ) (B_M : ℕ) (C_M : ℕ) -- March
  (D_A : ℕ) (B_A : ℕ) (C_A : ℕ) -- April
  (D_Ma : ℕ) (B_Ma : ℕ) (C_Ma : ℕ) -- May

/-- Function to calculate the percentage difference given sales of three groups --/
def percentage_difference (D B C : ℕ) : ℝ :=
  ((max D (max B C) - min D (min B C)) / (min D (min B C)).toReal) * 100

/-- The Lean statement proving February has the greatest percentage difference in sales --/
theorem february_has_greatest_percentage_diff (s : SalesData) :
  percentage_difference s.D_F s.B_F s.C_F ≥ 
  (percentage_difference s.D_J s.B_J s.C_J) ∧ -- January
  percentage_difference s.D_F s.B_F s.C_F ≥ 
  (percentage_difference s.D_A s.B_A s.C_A) ∧ -- April
  percentage_difference s.D_F s.B_F s.C_F ≥ 
  (percentage_difference s.D_M s.B_M s.C_M) ∧ -- March
  percentage_difference s.D_F s.B_F s.C_F ≥ 
  (percentage_difference s.D_Ma s.B_Ma s.C_Ma)  -- May
  :=
sorry

end february_has_greatest_percentage_diff_l373_373380


namespace find_p_max_area_triangle_l373_373536

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373536


namespace matrix_to_system_solution_l373_373450

theorem matrix_to_system_solution :
  ∀ (x y : ℝ),
  (2 * x + y = 5) ∧ (x - 2 * y = 0) →
  3 * x - y = 5 :=
by
  sorry

end matrix_to_system_solution_l373_373450


namespace part1_part2_l373_373914

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x
def F (x : ℝ) (λ : ℝ) : ℝ := g x - λ * f x

theorem part1
  (h1 : f 1 = 3)
  (h2 : ∀ x, f (-1 + x) = f (-1 - x))
  (symmetry_condition : ∀ x, g x = -f (-x))
  : f (x) = x^2 + 2x ∧ g (x) = -x^2 + 2x := 
  sorry

theorem part2
  (λ : ℝ)
  (increasing_condition : ∀ x ∈ Icc (-1 : ℝ) (1 : ℝ), 0 ≤ deriv (λ x, F x λ))
  : λ ≤ -1 :=
  sorry

end part1_part2_l373_373914


namespace reflection_point_in_quadrilateral_boundary_or_inside_l373_373671

theorem reflection_point_in_quadrilateral_boundary_or_inside 
  (N : Type) [convex_quadrilateral N] (A B C D reflectionA reflectionB reflectionC reflectionD : N)
  (h_mid_A : reflectionA = reflect_over_midpoint_connected_vertices N A B D)
  (h_mid_B : reflectionB = reflect_over_midpoint_connected_vertices N B A C)
  (h_mid_C : reflectionC = reflect_over_midpoint_connected_vertices N C B D)
  (h_mid_D : reflectionD = reflect_over_midpoint_connected_vertices N D A C) :
  reflectionA ∈ closure N ∨ reflectionB ∈ closure N ∨ reflectionC ∈ closure N ∨ reflectionD ∈ closure N :=
sorry

end reflection_point_in_quadrilateral_boundary_or_inside_l373_373671


namespace matrix_inverse_l373_373641

variable (N : Matrix (Fin 2) (Fin 2) ℚ) 
variable (I : Matrix (Fin 2) (Fin 2) ℚ)
variable (c d : ℚ)

def M1 : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

def M2 : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 0], ![0, 1]]

theorem matrix_inverse (hN : N = M1) 
                       (hI : I = M2) 
                       (hc : c = 1/12) 
                       (hd : d = 1/12) :
                       N⁻¹ = c • N + d • I := by
  sorry

end matrix_inverse_l373_373641


namespace road_renovation_condition_l373_373861

-- Define the conditions
def actual_length_repaired_each_day (x : ℝ) := x
def equation := ∀ x : ℝ, 1500 / (x - 5) - 1500 / x = 10

-- Define the condition we need to prove
def condition := "Repairing an extra 5m each day compared to the original plan results in completing 10 days ahead of schedule"

-- The statement we need to prove
theorem road_renovation_condition (x : ℝ) (h₁ : actual_length_repaired_each_day x) (h₂ : equation x) :
  condition :=
sorry

end road_renovation_condition_l373_373861


namespace area_enclosed_is_one_third_l373_373211

theorem area_enclosed_is_one_third :
  ∫ x in (0:ℝ)..1, (x^(1/2) - x^2 : ℝ) = (1/3 : ℝ) :=
by
  sorry

end area_enclosed_is_one_third_l373_373211


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373500

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373500


namespace number_of_valid_arrangements_l373_373892

theorem number_of_valid_arrangements :
  let students := ["A", "B", "C", "D", "E"]
  let total_students := 5
  let end_positions := [0, total_students - 1]
  let adjacent_positions := λ l : List Nat, ∃ s1 s2, l.get! s1 = "C" ∧ l.get! s2 = "D" ∧ (s1 = s2 + 1 ∨ s2 = s1 + 1)
  let valid_positions := λ l : List Nat, ∀ i ∈ end_positions, l.get! i ≠ "A"
  students.nodup ∧ adjacent_positions students ∧ valid_positions students ↔ length (List.permutations students) = 24
:= sorry

end number_of_valid_arrangements_l373_373892


namespace parabola_condition_max_area_triangle_l373_373488

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373488


namespace unique_b_positive_solution_l373_373404

theorem unique_b_positive_solution (c : ℝ) (h : c ≠ 0) : 
  (∃ b : ℝ, b > 0 ∧ ∀ b : ℝ, b ≠ 0 → 
    ∀ x : ℝ, x^2 + (b + 1 / b) * x + c = 0 → x = - (b + 1 / b) / 2) 
  ↔ c = (5 + Real.sqrt 21) / 2 ∨ c = (5 - Real.sqrt 21) / 2 := 
by {
  sorry
}

end unique_b_positive_solution_l373_373404


namespace simplify_fraction_l373_373678

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 3 + 1) + 2 / (Real.sqrt 5 - 1))) = 
  ((Real.sqrt 3) + 2 * (Real.sqrt 5) - 1) / (2 + 4 * Real.sqrt 5) := 
by 
  sorry

end simplify_fraction_l373_373678


namespace problem_statement_l373_373681

noncomputable def a : ℝ := Real.sqrt 3 - Real.sqrt 11
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 11

theorem problem_statement : (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 :=
by
  -- conditions
  let a := Real.sqrt 3 - Real.sqrt 11
  let b := Real.sqrt 3 + Real.sqrt 11
  have h1 : a = Real.sqrt 3 - Real.sqrt 11 := rfl
  have h2 : b = Real.sqrt 3 + Real.sqrt 11 := rfl
  -- question statement
  sorry

end problem_statement_l373_373681


namespace cos_alpha_value_l373_373989
open Real

theorem cos_alpha_value (α : ℝ) (h0 : 0 < α ∧ α < π / 2) 
  (h1 : sin (α - π / 6) = 1 / 3) : 
  cos α = (2 * sqrt 6 - 1) / 6 := 
by 
  sorry

end cos_alpha_value_l373_373989


namespace test_question_count_l373_373357

theorem test_question_count (n : ℕ) (h1 : 0.5 * n = 13 + 0.25 * (n - 20)) : n = 32 :=
by
  sorry

end test_question_count_l373_373357


namespace inequality_proof_l373_373946

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a * b * c = 1) : 
  (1 / (a^2 * (b + c))) + (1 / (b^2 * (c + a))) + (1 / (c^2 * (a + b))) ≥ 3 / 2 :=
sorry

end inequality_proof_l373_373946


namespace max_value_l373_373409

theorem max_value (y : ℝ) (h : y ≠ 0) : 
  ∃ M, M = 1 / 25 ∧ 
       ∀ y ≠ 0,  ∀ value, value = y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) 
       → value ≤ M :=
sorry

end max_value_l373_373409


namespace sausage_thickness_correct_l373_373121

noncomputable def earth_radius := 6000 -- in km
noncomputable def distance_to_sun := 150000000 -- in km
noncomputable def sausage_thickness := 44 -- in km

theorem sausage_thickness_correct :
  let R := earth_radius
  let L := distance_to_sun
  let r := Real.sqrt ((4 * R^3) / (3 * L))
  abs (r - sausage_thickness) < 10 * sausage_thickness :=
by
  sorry

end sausage_thickness_correct_l373_373121


namespace area_quadruple_l373_373643

noncomputable def area (A B C : ℝ × ℝ) : ℝ :=
  (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)

structure ConvexQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
  (convex : true)  -- Placeholder, include detailed convexity checks if necessary

def vectorAdd (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

def pointO : ℝ × ℝ := (0, 0) -- Assume O is at origin for simplification or redefine as needed

theorem area_quadruple (A B C D : ℝ × ℝ) (h : ConvexQuadrilateral A B C D) :
  let A' := vectorAdd pointO (B.1 - A.1, B.2 - A.2),
      B' := vectorAdd pointO (C.1 - B.1, C.2 - B.2),
      C' := vectorAdd pointO (D.1 - C.1, D.2 - C.2),
      D' := vectorAdd pointO (A.1 - D.1, A.2 - D.2) in
  area A' B' C' + area A' C' D' = 2 * (area A B C + area A C D) :=
by
  sorry

end area_quadruple_l373_373643


namespace arc_length_of_sector_l373_373426

theorem arc_length_of_sector (r S : ℝ) (h1 : r = 2) (h2 : S = 4) : 
  ∃ l : ℝ, l = 4 ∧ S = (1 / 2) * r * l :=
by
  use 4
  split
  case left => rfl
  case right =>
    rw [h1, h2]
    norm_num
    rw [(1 / 2) * 2 * 4]
    norm_num
    sorry

end arc_length_of_sector_l373_373426


namespace sqrt_subtraction_proof_l373_373256

def sqrt_subtraction_example : Real := 
  let a := 49 + 121
  let b := 36 - 9
  Real.sqrt a - Real.sqrt b

theorem sqrt_subtraction_proof : sqrt_subtraction_example = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end sqrt_subtraction_proof_l373_373256


namespace angle_quadrant_2_radians_l373_373227

theorem angle_quadrant_2_radians (θ : ℝ) (h1 : θ = 2) : (real.pi / 2 < θ ∧ θ < real.pi) :=
by
  sorry

end angle_quadrant_2_radians_l373_373227


namespace find_richards_score_l373_373672

variable (R B : ℕ)

theorem find_richards_score (h1 : B = R - 14) (h2 : B = 48) : R = 62 := by
  sorry

end find_richards_score_l373_373672


namespace average_speed_proof_l373_373799

-- Definitions from conditions
def biking_time := 2 / 3           -- in hours
def biking_speed := 18             -- in km/h (converted from 5 m/s)
def walking_time := 2              -- in hours
def walking_speed := 5             -- in km/h

-- Calculations based on the conditions
def biking_distance := biking_time * biking_speed  -- 12 km
def walking_distance := walking_time * walking_speed  -- 10 km

def total_distance := biking_distance + walking_distance  -- 22 km
def total_time := biking_time + walking_time  -- 8 / 3 hours

-- The average speed to be proven
def average_speed := total_distance / total_time         

-- The theorem to be proven
theorem average_speed_proof : average_speed = 8.25 := 
by
sorry

end average_speed_proof_l373_373799


namespace simple_interest_rate_l373_373824

theorem simple_interest_rate (P A T : ℝ) (hP : P = 25000) (hA : A = 42500) (hT : T = 12) : 
  ∃ R, R ≈ 5.83 := 
by
  let SI := A - P
  let R := (SI * 100) / (P * T)
  use R
  field_simp [hP, hA, hT]
  sorry

end simple_interest_rate_l373_373824


namespace train_speed_in_km_per_hr_l373_373813

-- Define the conditions
def length_of_train : ℝ := 100 -- length in meters
def time_to_cross_pole : ℝ := 6 -- time in seconds

-- Define the conversion factor from meters/second to kilometers/hour
def conversion_factor : ℝ := 18 / 5

-- Define the formula for speed calculation
def speed_of_train := (length_of_train / time_to_cross_pole) * conversion_factor

-- The theorem to be proven
theorem train_speed_in_km_per_hr : speed_of_train = 50 := by
  sorry

end train_speed_in_km_per_hr_l373_373813


namespace coeff_x5_l373_373250

theorem coeff_x5:
  let p1 := (x^4 - 2 * x^3 + 4 * x^2 - 5 * x + 3)
  let p2 := (3 * x^3 - 4 * x^2 + 6 * x - 8)
  (coeff (p1 * p2) 5) = 14 :=
by {
  sorry -- Proof goes here
}

end coeff_x5_l373_373250


namespace find_p_max_area_of_triangle_l373_373511

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373511


namespace range_f_l373_373709

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (cos x ^ 2 - 3 / 4) + sin x

theorem range_f :
  ∀ x : ℝ, cos x ^ 2 - 3 / 4 ≥ 0 → 
  set.range (λ x, sqrt (cos x ^ 2 - 3 / 4) + sin x) = set.interval (-1/2) (real.sqrt 2 / 2) :=
sorry

end range_f_l373_373709


namespace find_x_l373_373908

theorem find_x (x : ℝ) (i : ℂ) (hi : i = complex.I) (h : x^2 + i * x + 6 = 2 * i + 5 * x) : x = 2 :=
by
  sorry

end find_x_l373_373908


namespace capital_fraction_C_l373_373330

open Real

theorem capital_fraction_C (T p a : ℝ) (h1: a = 810) (h2: p = 2430) (hT: T ≠ 0) : 
  (c : ℝ) = 5 / 24 :=
by 
  -- Let's assume the proportions of the capital
  let A_share := 1 / 3 * T
  let B_share := 1 / 4 * T
  let C_share := c * T
  let D_share := (1 - (1/3 + 1/4 + c)) * T
  
  -- Use the provided profit information
  have a_share_by_profit : 1 / 3 = a / p := by 
    field_simp [h2, h1]
    linarith
  
  -- Puts conditions on capital contribution sums
  have sum_of_shares : (1 / 3 + 1 / 4 + c + (1 - (1 / 3 + 1 / 4 + c))) = 1 := by
    linarith

  -- Solve for c
  have solve_for_c : 2 * c = 1 - (1 / 3 + 1 / 4) := by
    simp [show (4/12 + 3/12) = 7/12, by field_simp]
    ring
  
  have fractional_c : c = 5 / 24 := by
    field_simp [solve_for_c]
    linarith
  
  apply fractional_c
  sorry

end capital_fraction_C_l373_373330


namespace exists_person_with_girls_as_neighbors_l373_373245

theorem exists_person_with_girls_as_neighbors (boys girls : Nat) (sitting : Nat) 
  (h_boys : boys = 25) (h_girls : girls = 25) (h_sitting : sitting = boys + girls) :
  ∃ p : Nat, p < sitting ∧ (p % 2 = 1 → p.succ % sitting % 2 = 0) := 
by
  sorry

end exists_person_with_girls_as_neighbors_l373_373245


namespace combined_cost_of_items_l373_373844

theorem combined_cost_of_items (wallet_cost : ℕ) 
  (purse_cost : ℕ) (combined_cost : ℕ) :
  wallet_cost = 22 →
  purse_cost = 4 * wallet_cost - 3 →
  combined_cost = wallet_cost + purse_cost →
  combined_cost = 107 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end combined_cost_of_items_l373_373844


namespace function_is_identity_l373_373385

theorem function_is_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, (Finset.range n).sum f ∣ (Finset.range n).sum id) :
  ∀ n : ℕ+, f n = n :=
sorry

end function_is_identity_l373_373385


namespace johns_website_visits_l373_373631

theorem johns_website_visits (c: ℝ) (d: ℝ) (days: ℕ) (h1: c = 0.01) (h2: d = 10) (h3: days = 30) :
  d / c * days = 30000 :=
by
  sorry

end johns_website_visits_l373_373631


namespace find_omega_l373_373056

theorem find_omega 
  (ω : ℝ) 
  (h1 : ∃ k : ℤ, (3 * ω * π) / 4 = π / 2 + k * π)
  (h2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 2 * π / 3 → cos (ω * x1) > cos (ω * x2))
  (h3 : ω > 0) : ω = 2 / 3 :=
by
  sorry

end find_omega_l373_373056


namespace principal_amount_l373_373326

theorem principal_amount (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) : 
  SI = 4016.25 ∧ R = 0.13 ∧ T = 5 → P = SI / (R * T) → P = 6180 :=
by
  intro h₁ h₂,
  cases h₁ with hSI hR,
  cases hR with hR hT,
  rw [hSI, hR, hT] at h₂,
  exact h₂

end principal_amount_l373_373326


namespace common_root_implies_remaining_roots_l373_373176

variables {R : Type*} [LinearOrderedField R]

theorem common_root_implies_remaining_roots
  (a b c x1 x2 x3 : R) 
  (h_non_zero_a : a ≠ 0)
  (h_non_zero_b : b ≠ 0)
  (h_non_zero_c : c ≠ 0)
  (h_a_ne_b : a ≠ b)
  (h_common_root1 : x1^2 + a*x1 + b*c = 0)
  (h_common_root2 : x1^2 + b*x1 + c*a = 0)
  (h_root2_eq : x2^2 + a*x2 + b*c = 0)
  (h_root3_eq : x3^2 + b*x3 + c*a = 0)
  : x2^2 + c*x2 + a*b = 0 ∧ x3^2 + c*x3 + a*b = 0 :=
sorry

end common_root_implies_remaining_roots_l373_373176


namespace problem_I_problem_II_l373_373153

-- Definition of transformation matrix for counterclockwise rotation of π/2
def M1 : Matrix (Fin 2) (Fin 2) ℝ := ![![0, -1], ![1, 0]]

-- Definition of the second transformation matrix
def M2 : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 1], ![0, 1]]

-- The point P(2, 1)
def P : Fin 2 → ℝ := ![2, 1]

-- Applying T1 to P(2, 1)
def P' := M1.mulVec P

-- Problem I: Proving the coordinates after applying T1 to P are (-1, 2)
theorem problem_I : P' = ![-1, 2] := sorry

-- Combining M1 and M2 to form the transformation matrix M
def M := M2 ⬝ M1

-- Equation of the original curve y = x^2
def original_curve (x : ℝ) : ℝ := x^2

-- Points on the transformed curve
def transformed_points (x0 y0 : ℝ) := λ (M.mulVec ![x0, y0])

-- The transformed curve equation
def transformed_curve (x y : ℝ) := (x0 : ℝ) (y0 : ℝ) (H : transformed_points x0 y0) := y - x = y^2

-- Problem II: Proving the equation of the transformed curve
theorem problem_II : ∀ (x y: ℝ), 
  (exists x0 y0: ℝ, transformed_points x0 y0 = ![x, y] ∧ original_curve x0 = y0) -> 
  y - x = y^2 := sorry

end problem_I_problem_II_l373_373153


namespace force_with_18_inch_wrench_l373_373700

theorem force_with_18_inch_wrench :
  ∀ (F L : ℕ), (F * L = 3600) → (F = 200) :=
by
  assume F L hFL
  have h1 : 300 * 12 = 3600 := by norm_num
  have h2 : L = 18 := sorry
  have h3 : 3600 / 18 = 200 := by norm_num
  exact sorry

end force_with_18_inch_wrench_l373_373700


namespace area_of_trapezoid_PQRS_l373_373603

variable (PQ RS : Set Point) -- Given, implicit
variable (PR QS : Set Point) -- Given, implicit
variable (P QT T R : Point) -- Given, implicit

def is_parallel (a b : Set Point) : Prop := sorry -- Assume definition for parallel

axiom condition_1 : is_parallel PQ RS
axiom condition_2 : ∃ T, T = (PR ∩ QS)
axiom condition_3 : area (triangle P Q T) = 60
axiom condition_4 : area (triangle P R T) = 30

theorem area_of_trapezoid_PQRS (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) (h4 : condition_4):
  area (trapezoid P Q R S) = 120 :=
  sorry

end area_of_trapezoid_PQRS_l373_373603


namespace problem1_problem2_l373_373026

theorem problem1 :
  ∃ a b : ℝ, (∀ x : ℂ, x^2 - (6 + complex.i) * x + 9 + a * complex.i = 0 → x = b) ∧ a = 3 ∧ b = 3 :=
by sorry

theorem problem2 :
  let z1 := 2 / (1 + complex.i) in
  let a := 3 in
  let b := 3 in
  ∃ z : ℂ, |z - a - b * complex.i| = |z1| ∧ |z| = 2 * real.sqrt 2 :=
by sorry

end problem1_problem2_l373_373026


namespace find_a_monotonicity_intervals_l373_373442

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (1 + x) + x^2 - 10 * x

theorem find_a (a : ℝ) (h : ∃ c, (f a 3 = c ∧ ∂ x, ∂' (f a x) = 0)) : a = 16 :=
sorry

theorem monotonicity_intervals :
  (monotone_on (f 16) Set.Ioi (-1)) ∧ 
  (monotone_on (f 16) Set.Ici (1) ∧ 
  (monotone_on (f 16) Set.Ioi (3)) :=
sorry

end find_a_monotonicity_intervals_l373_373442


namespace factor_polynomial_l373_373383

theorem factor_polynomial :
  4 * (λ x : ℝ, (x + 3) * (x + 7) * (x + 8) * (x + 10)) + 2 * (λ x : ℝ, x^2) =
  (λ x : ℝ, (x^2 + 16 * x + 72) * (2 * x + 36) * (2 * x + 9)) :=
sorry

end factor_polynomial_l373_373383


namespace foci_and_real_axis_of_hyperbola_equation_of_line_intersecting_hyperbola_l373_373963

variables {a : ℝ} (ha : a > 0)

def hyperbola := {p : ℝ × ℝ | p.1 ^ 2 / a ^ 2 - p.2 ^ 2 / 4 = 1}
def ellipse := {p : ℝ × ℝ | p.1 ^ 2 / 16 + p.2 ^ 2 / 8 = 1}

theorem foci_and_real_axis_of_hyperbola (h_foci_coincide : ∀ p ∈ ellipse, p.fst = 4 ∨ p.fst = -4) :
  (∀ p ∈ hyperbola ha, p = (4, 0) ∨ p = (-4, 0)) ∧ (4 * sqrt 3) = 4*sqrt(3) := sorry

variables {l : ℝ → ℝ} {A B : ℝ × ℝ} (h_midpoint : (A.1 + B.1) / 2 = 6 ∧ (A.2 + B.2) / 2 = 1)
           (h_intersection_A : A ∈ hyperbola ha) (h_intersection_B : B ∈ hyperbola ha)

theorem equation_of_line_intersecting_hyperbola :
  ∃ k b, (∀ x, l x = k * x + b) ∧ k = 2 ∧ b = -11 := sorry

end foci_and_real_axis_of_hyperbola_equation_of_line_intersecting_hyperbola_l373_373963


namespace motorcycle_price_l373_373791

variable (x : ℝ) -- selling price of each motorcycle
variable (car_cost material_car material_motorcycle : ℝ)

theorem motorcycle_price
  (h1 : car_cost = 100)
  (h2 : material_car = 4 * 50)
  (h3 : material_motorcycle = 250)
  (h4 : 8 * x - material_motorcycle = material_car - car_cost + 50)
  : x = 50 := 
sorry

end motorcycle_price_l373_373791


namespace simplest_proper_fraction_36_l373_373324

theorem simplest_proper_fraction_36 (n d : ℕ) (h₁ : n < d) (h₂ : n.coprime d) (h₃ : n * d = 36) :
  (n = 1 ∧ d = 36) ∨ (n = 4 ∧ d = 9) :=
by
  sorry

end simplest_proper_fraction_36_l373_373324


namespace problem_proof_l373_373438

theorem problem_proof (x y : ℝ) (h_cond : (x + 3)^2 + |y - 2| = 0) : (x + y)^y = 1 :=
by
  sorry

end problem_proof_l373_373438


namespace find_p_max_area_triangle_l373_373517

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373517


namespace union_of_sets_l373_373414

-- Definition for set M
def M : Set ℝ := {x | x^2 - 4 * x + 3 < 0}

-- Definition for set N
def N : Set ℝ := {x | 2 * x + 1 < 5}

-- The theorem linking M and N
theorem union_of_sets : M ∪ N = {x | x < 3} :=
by
  -- Proof goes here
  sorry

end union_of_sets_l373_373414


namespace remainder_of_sum_1_to_20_divided_by_9_is_3_l373_373751

-- Lean statement to prove the given problem
theorem remainder_of_sum_1_to_20_divided_by_9_is_3 :
  (∑ k in Finset.range 21, k) % 9 = 3 :=
by
  sorry

end remainder_of_sum_1_to_20_divided_by_9_is_3_l373_373751


namespace same_function_A_not_same_function_B_not_same_function_C_not_same_function_D_l373_373338

section
  variable (x : ℝ)

  def f_A (x : ℝ) := |x|
  def g_A (x : ℝ) := Real.sqrt (x^2)
  def f_B (x : ℝ) := Real.log (x^2)
  def g_B (x : ℝ) := 2 * Real.log x
  def f_C (x : ℝ) := if x ≠ 1 then (x^2 - 1) / (x - 1) else 0
  def g_C (x : ℝ) := x + 1
  def f_D (x : ℝ) := Real.sqrt (x + 1) * Real.sqrt (x - 1)
  def g_D (x : ℝ) := Real.sqrt (x^2 - 1)

  theorem same_function_A : ∀ x : ℝ, f_A x = g_A x := 
  by
    sorry

  theorem not_same_function_B : ∃ x : ℝ, f_B x ≠ g_B x :=
  by
    sorry

  theorem not_same_function_C : ∃ x : ℝ, f_C x ≠ g_C x :=
  by
    sorry

  theorem not_same_function_D : ∃ x : ℝ, f_D x ≠ g_D x := 
  by
    sorry
end

end same_function_A_not_same_function_B_not_same_function_C_not_same_function_D_l373_373338


namespace segment_division_game_l373_373208

theorem segment_division_game (k l : ℝ) (hk : k > 0) (hl : l > 0) :
  (k / l > 1 → "Person A wins") ∧ (k / l ≤ 1 → "Person B wins") :=
by
  sorry

end segment_division_game_l373_373208


namespace x_is_irrational_l373_373854
open Nat

def num_divisors (n : ℕ) : ℕ := (range n).filter (λ d, d > 0 ∧ n % d = 0).length

def a_n (n : ℕ) : ℕ :=
  if num_divisors n % 2 = 1 then 0 else 1

def x : ℝ :=
  let coefficients := (λ n, (a_n n : ℝ))
  (0 + (Nat.sum (list.range 0 upto n)

theorem x_is_irrational : ¬ is_rat x := 
begin
  -- Proof omitted
  sorry
end

end x_is_irrational_l373_373854


namespace count_integer_pairs_l373_373091

theorem count_integer_pairs :
  { pairs : ℤ × ℤ // 0 ≤ pairs.1 ∧ pairs.1 < 31 ∧ 0 ≤ pairs.2 ∧ pairs.2 < 31 ∧ ((pairs.1 ^ 2 - 18) ^ 2) % 31 = (pairs.2 ^ 2) % 31 }.card = 60 :=
by
  sorry

end count_integer_pairs_l373_373091


namespace percentage_increase_to_restore_salary_l373_373710

theorem percentage_increase_to_restore_salary (S : ℝ) (hS : S > 0) :
  ∃ (P : ℝ), S * 0.86 * (1 + P / 100) = S ∧ P ≈ 16.28 :=
begin
  sorry
end

end percentage_increase_to_restore_salary_l373_373710


namespace combined_cost_of_items_l373_373845

theorem combined_cost_of_items (wallet_cost : ℕ) 
  (purse_cost : ℕ) (combined_cost : ℕ) :
  wallet_cost = 22 →
  purse_cost = 4 * wallet_cost - 3 →
  combined_cost = wallet_cost + purse_cost →
  combined_cost = 107 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end combined_cost_of_items_l373_373845


namespace isosceles_trapezoid_side_length_l373_373686

theorem isosceles_trapezoid_side_length (A b1 b2 h half_diff s : ℝ) (h0 : A = 44) (h1 : b1 = 8) (h2 : b2 = 14) 
    (h3 : A = 0.5 * (b1 + b2) * h)
    (h4 : h = 4) 
    (h5 : half_diff = (b2 - b1) / 2) 
    (h6 : half_diff = 3)
    (h7 : s^2 = h^2 + half_diff^2)
    (h8 : s = 5) : 
    s = 5 :=
by 
    apply h8

end isosceles_trapezoid_side_length_l373_373686


namespace find_x_l373_373278

theorem find_x (x : ℝ) (h : 0.25 * x = 200 - 30) : x = 680 := 
by
  sorry

end find_x_l373_373278


namespace pentagonal_pyramid_edges_l373_373089

theorem pentagonal_pyramid_edges :
  ∀ (pyramid : Type) 
  (base : Type) 
  [has_edges: ∀ (t : Type), has_coe_to_sort t (with_size uint8)] 
  [has_base : has_coe pyramid base]
  [is_pentagon : base → Prop] 
  [has_5_edges: ∀ (b : base), is_pentagon b → b.edge_count = 5]
  [has_triangular_faces: pyramid → Prop] 
  [triangle_count: ∀ (p : pyramid), has_triangular_faces p → p.triangle_count = 5]
  [has_common_vertex: ∀ (p : pyramid), has_triangular_faces p → has_common_vertex p]
  [non_shared_edges: ∀ (p : pyramid), has_triangular_faces p → ∀ t, t ∈ p.triangles → non_shared_edge t],
  ∀ (p : pyramid), (is_pentagon p.base) → (has_triangular_faces p) → pyramid.edge_count = 10 :=
begin
  intros,
  sorry
end

end pentagonal_pyramid_edges_l373_373089


namespace translation_symmetric_graphs_l373_373142

/-- The graph of the function f(x)=sin(x/π + φ) is translated to the right by θ (θ>0) units to obtain the graph of the function g(x).
    On the graph of f(x), point A is translated to point B, let x_A and x_B be the abscissas of points A and B respectively.
    If the axes of symmetry of the graphs of f(x) and g(x) coincide, then the real values that can be taken as x_A - x_B are -2π² or -π². -/
theorem translation_symmetric_graphs (θ : ℝ) (hθ : θ > 0) (x_A x_B : ℝ) (φ : ℝ) :
  ((x_A - x_B = -2 * π^2) ∨ (x_A - x_B = -π^2)) :=
sorry

end translation_symmetric_graphs_l373_373142


namespace magnitude_of_difference_l373_373083

variables (a b : ℝ × ℝ) (y : ℝ)

def vec_a : ℝ × ℝ := (2, 1)
def vec_b (y : ℝ) : ℝ × ℝ := (1 - y, 2 + y)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := dot_product v1 v2 = 0
def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem magnitude_of_difference (a b : ℝ × ℝ) (y : ℝ)
  (h1: a = (2, 1))
  (h2: b = (1 - y, 2 + y))
  (h3: perpendicular a b) :
  magnitude (a.1 + b.1, a.2 - b.2) = 5 * real.sqrt 2 :=
by
  sorry

end magnitude_of_difference_l373_373083


namespace train_length_l373_373766

noncomputable def speed_kmph := 80
noncomputable def time_seconds := 5

 noncomputable def speed_mps := (speed_kmph * 1000) / 3600

 noncomputable def length_train : ℝ := speed_mps * time_seconds

theorem train_length : length_train = 111.1 := by
  sorry

end train_length_l373_373766


namespace max_area_triangle_PAB_l373_373549

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373549


namespace divisor_of_3825_is_15_l373_373265

theorem divisor_of_3825_is_15 : ∃ d, 3830 - 5 = 3825 ∧ 3825 % d = 0 ∧ d = 15 := by
  sorry

end divisor_of_3825_is_15_l373_373265


namespace find_solutions_l373_373874

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem find_solutions :
    {x : ℝ | cuberoot x = 15 / (8 - cuberoot x)} = {125, 27} :=
by
  sorry

end find_solutions_l373_373874


namespace investment_ratio_l373_373228

theorem investment_ratio (P_i Q_i : ℝ) (h1 : 5 * P_i / (14 * Q_i) = 1 / 2) : P_i / Q_i = 7 / 5 :=
by
  have h_eq : 2 * (5 * P_i) = 14 * Q_i,
  { sorry },
  have h_eq2 : 10 * P_i = 14 * Q_i,
  { sorry },
  have h_rat : P_i / Q_i = 14 / 10,
  { sorry },
  have h_simplified : 14 / 10 = 7 / 5,
  { sorry },
  exact (eq.trans h_rat h_simplified)

end investment_ratio_l373_373228


namespace number_of_ordered_triples_l373_373801

theorem number_of_ordered_triples :
  let b := 2023
  let n := (b ^ 2)
  ∀ (a c : ℕ), a * c = n ∧ a ≤ b ∧ b ≤ c → (∃ (k : ℕ), k = 7) :=
by
  sorry

end number_of_ordered_triples_l373_373801


namespace sum_with_signs_always_odd_l373_373665

-- Define the sum from 1 to 2009
def sum_1_to_2009 : ℕ := (2009 * (2009 + 1)) / 2

-- Define the problem statement
theorem sum_with_signs_always_odd :
  ∀ (f : ℕ → ℤ), (∀ n, 1 ≤ n ∧ n ≤ 2009 → f n = n ∨ f n = -n) →
    (∑ n in finset.range 2009, f (n + 1)) % 2 = 1 :=
sorry

end sum_with_signs_always_odd_l373_373665


namespace product_eval_l373_373381

theorem product_eval (a : ℤ) (h : a = 3) : (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  sorry

end product_eval_l373_373381


namespace percentage_difference_l373_373274

variable (x y : ℝ)
variable (hxy : x = 6 * y)

theorem percentage_difference : ((x - y) / x) * 100 = 83.33 := by
  sorry

end percentage_difference_l373_373274


namespace find_a4_l373_373921
open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n / (2 * a n + 3)

theorem find_a4 (a : ℕ → ℚ) (h : seq a) : a 4 = 1 / 53 :=
by
  obtain ⟨h1, h_rec⟩ := h
  have a2 := h_rec 1 (by decide)
  have a3 := h_rec 2 (by decide)
  have a4 := h_rec 3 (by decide)
  -- Proof steps would go here
  sorry

end find_a4_l373_373921


namespace find_vector_a_find_magnitude_l373_373449

open Real

-- Define vectors as tuples in Lean
structure Vec2 where
  x : ℝ
  y : ℝ

-- Conditions
def collinear (a b : Vec2) : Prop :=
  ∃ λ : ℝ, a = ⟨λ * b.x, λ * b.y⟩

def dot_product (a b : Vec2) : ℝ :=
  a.x * b.x + a.y * b.y

def vector_b : Vec2 := ⟨1, -2⟩
def dot_product_a_b : ℝ := -10

-- Statements to prove
theorem find_vector_a : ∃ a : Vec2, collinear a vector_b ∧ dot_product a vector_b = dot_product_a_b ∧ a = ⟨-2, 4⟩ :=  
sorry

def vector_c : Vec2 := ⟨6, -7⟩
def vector_a : Vec2 := ⟨-2, 4⟩

def vector_add (a b : Vec2) : Vec2 :=
  ⟨a.x + b.x, a.y + b.y⟩

def vector_magnitude (a : Vec2) : ℝ :=
  sqrt (a.x ^ 2 + a.y ^ 2)

theorem find_magnitude : vector_magnitude (vector_add vector_a vector_c) = 5 := 
sorry

end find_vector_a_find_magnitude_l373_373449


namespace number_of_safe_ints_l373_373408

/-- A number n is p-safe if it is at a distance more than 3 from all multiples of p. -/
def p_safe (n p : ℕ) : Prop :=
  ∀ k : ℕ, abs (n - k * p) > 3

/-- A set of positive integers less than or equal to 20000 is p-safe if every element is p-safe. -/
def safe_set (m p : ℕ) (s : List ℕ) : Prop :=
  ∀ n ∈ s, p_safe n p

/-- Main theorem statement: Determine the number of integers n ∈ [1, 20000] which are simultaneously 
8-safe, 10-safe, and 12-safe. -/
theorem number_of_safe_ints : 
  ∃ (count : ℕ),
    count = 2664 ∧
    safe_set 20000 8 (List.filter (λ n, p_safe n 8) (List.range 20001)) ∧
    safe_set 20000 10 (List.filter (λ n, p_safe n 10) (List.range 20001)) ∧
    safe_set 20000 12 (List.filter (λ n, p_safe n 12) (List.range 20001)) :=
begin
  use 2664,
  split,
  { refl, },
  split,
  { sorry, },
  split,
  { sorry, },
  { sorry, },
end

end number_of_safe_ints_l373_373408


namespace general_term_sum_Tn_correct_l373_373043

def Sn (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

def an_formula (n : ℕ) : ℤ := 2 * n - 7

def bn (n : ℕ) : ℤ :=
  have anp1 := 2 * (n + 1) - 7
  (2 * n - 7) * (2 : ℤ) ^ anp1

def Tn (n : ℕ) : ℚ :=
  (23 / 72 : ℚ) + (n / 3 : ℚ - 23 / 18 : ℚ) * (2 : ℚ) ^ (2 * n - 2 : ℤ)

theorem general_term (a1 a22 Sn22 : ℤ) (d : ℕ) (h_a22 : a22 = 37)
  (h_Sn22 : Sn a1 d 22 = 352) :
  an_formula 22 = 37 ∧ Sn a1 d 22 = 352 :=
by {
  sorry
}

theorem sum_Tn_correct (h_an_formula : ∀ n, (a_n n = 2 * n - 7))
  (h_bn : ∀ n, b_n n = (2 * n - 7) * 2^(2 * (n + 1) - 7)) :
  Tn n = \(23 / 72) + (n / 3 - (23 / 18)) * 2^(2 * n - 2) :=
by {
  sorry
}

end general_term_sum_Tn_correct_l373_373043


namespace num_sequences_of_8_digits_l373_373983

theorem num_sequences_of_8_digits : 
  let same_parity_adjacent (x y : ℕ) := (x % 2 = y % 2) in
  let alternate_parity (x y : ℕ) := (x % 2 ≠ y % 2) in
  let valid_sequence (seq : Fin 8 → ℕ) := 
    same_parity_adjacent (seq 0) (seq 1) ∧ 
    alternate_parity (seq 1) (seq 2) ∧
    same_parity_adjacent (seq 2) (seq 3) ∧
    alternate_parity (seq 3) (seq 4) ∧
    same_parity_adjacent (seq 4) (seq 5) ∧
    alternate_parity (seq 5) (seq 6) ∧
    same_parity_adjacent (seq 6) (seq 7) in

  (∃ (seq : Fin 8 → ℕ), valid_sequence seq).to_set.card = 300125 :=
by {
  sorry
}

end num_sequences_of_8_digits_l373_373983


namespace maximize_profit_marginal_profit_decreasing_l373_373323

noncomputable def revenue (x : ℕ) : ℝ := 3700 * x + 45 * x^2 - 10 * x^3

noncomputable def cost (x : ℕ) : ℝ := 460 * x + 5000

noncomputable def profit (x : ℕ) : ℝ := revenue x - cost x

@[simp] lemma profit_def (x : ℕ) : profit x = -10 * x^3 + 45 * x^2 + 3240 * x - 5000 := by
  unfold profit revenue cost
  ring

noncomputable def marginal_function (f : ℕ → ℝ) (x : ℕ) : ℝ := f (x + 1) - f x

@[simp] lemma marginal_profit_eq (x : ℕ) : marginal_function profit x = -30 * x^2 + 60 * x + 3275 := by
  unfold marginal_function profit
  ring

-- Define the statement to prove that building 12 ships annually maximizes profit
theorem maximize_profit : ∃ (x : ℕ), x = 12 ∧ ∀ y, 1 ≤ y ∧ y ≤ 20 → profit y ≤ profit 12 := sorry

-- Define the statement to prove that marginal profit function is monotonously decreasing
theorem marginal_profit_decreasing : ∀ x, 1 < x ∧ x ≤ 20 → marginal_function profit x < marginal_function profit (x - 1) := sorry

end maximize_profit_marginal_profit_decreasing_l373_373323


namespace parabola_focus_distance_max_area_triangle_l373_373553

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373553


namespace problem_solution_l373_373099

-- Define the set
def my_set : Finset ℕ := {89, 95, 99, 132, 166, 173}

-- Noncomputable because we are using finite combinatorial enumeration
noncomputable def num_even_sum_subsets : ℕ :=
  (Finset.filter (λ s : Finset ℕ, s.sum % 2 = 0) (my_set.powerset.filter (λ s, s.card = 3))).card

-- The theorem we want to prove
theorem problem_solution:
  num_even_sum_subsets = 12 :=
sorry

end problem_solution_l373_373099


namespace find_p_max_area_triangle_l373_373532

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373532


namespace BX_div_CX_l373_373634

theorem BX_div_CX (A B C P X : Point) (hABC : Triangle A B C)
  (hAB : AB = 5) (hBC : BC = 6) (hCA : CA = 7)
  (hP_inside : P ∈ interior A B C)
  (h_similar : Similar (Triangle B P A) (Triangle A P C))
  (h_intersect : Line A P ∩ Line B C = X) : 
  BX / CX = 25 / 49 :=
sorry

end BX_div_CX_l373_373634


namespace SeedMixtureWeights_l373_373675

theorem SeedMixtureWeights (x y z : ℝ) (h1 : x + y + z = 8) (h2 : x / 3 = y / 2) (h3 : x / 3 = z / 3) :
  x = 3 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end SeedMixtureWeights_l373_373675


namespace binary_to_decimal_l373_373773

theorem binary_to_decimal : ∀ (n : ℕ), n = 0b11111 → n = 31 := by
  intros n h
  rw h
  norm_num
  -- Initially use 'sorry' as this is only to construct the statement
  -- sorry

end binary_to_decimal_l373_373773


namespace problem1_problem2_l373_373356

theorem problem1 : ( (2 / 3 - 1 / 4 - 5 / 6) * 12 = -5 ) :=
by sorry

theorem problem2 : ( (-3)^2 * 2 + 4 * (-3) - 28 / (7 / 4) = -10 ) :=
by sorry

end problem1_problem2_l373_373356


namespace find_r_s_l373_373578

theorem find_r_s (r s : ℚ) :
  (-3)^5 - 2*(-3)^4 + 3*(-3)^3 - r*(-3)^2 + s*(-3) - 8 = 0 ∧
  2^5 - 2*(2^4) + 3*(2^3) - r*(2^2) + s*2 - 8 = 0 →
  (r, s) = (-482/15, -1024/15) :=
by
  sorry

end find_r_s_l373_373578


namespace remainder_of_large_number_div_by_101_l373_373738

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end remainder_of_large_number_div_by_101_l373_373738


namespace ratio_expression_l373_373579

variable (a b c : ℚ)
variable (h1 : a / b = 6 / 5)
variable (h2 : b / c = 8 / 7)

theorem ratio_expression (a b c : ℚ) (h1 : a / b = 6 / 5) (h2 : b / c = 8 / 7) :
  (7 * a + 6 * b + 5 * c) / (7 * a - 6 * b + 5 * c) = 751 / 271 := by
  sorry

end ratio_expression_l373_373579


namespace expr_equals_4094552_l373_373646

-- Define x
def x : ℤ := -2023

-- Define the expression
def expr := abs(abs(x.natAbs * x.natAbs - x) - x.natAbs) - x

-- The theorem we want to prove
theorem expr_equals_4094552 : expr = 4094552 := by
  sorry

end expr_equals_4094552_l373_373646


namespace min_value_f_range_m_l373_373755

-- Part I: Prove that the minimum value of f(a) = a^2 + 2/a for a > 0 is 3
theorem min_value_f (a : ℝ) (h : a > 0) : a^2 + 2 / a ≥ 3 :=
sorry

-- Part II: Prove the range of m given the inequality for any positive real number a
theorem range_m (m : ℝ) : (∀ (a : ℝ), a > 0 → a^3 + 2 ≥ 3 * a * (|m - 1| - |2 * m + 3|)) → (m ≤ -3 ∨ m ≥ -1) :=
sorry

end min_value_f_range_m_l373_373755


namespace length_of_bridge_is_correct_l373_373327

noncomputable def train_length : ℝ := 150
noncomputable def crossing_time : ℝ := 29.997600191984642
noncomputable def train_speed_kmph : ℝ := 36
noncomputable def kmph_to_mps (v : ℝ) : ℝ := (v * 1000) / 3600
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is_correct :
  bridge_length = 149.97600191984642 := by
  sorry

end length_of_bridge_is_correct_l373_373327


namespace centroid_positions_l373_373209

-- Defining the problem conditions and the proof goal
theorem centroid_positions (points : set (ℝ × ℝ)) (h : points ⊆ {(0,0), (15,0), (15,15), (0,15)} ∪ 
    { (x, 0) | 1 ≤ x ∧ x ≤ 15 } ∪ { (15, y) | 1 ≤ y ∧ y ≤ 15 } ∪ 
    { (x, 15) | 1 ≤ x ∧ x ≤ 15 } ∪ { (0, y) | 1 ≤ y ∧ y ≤ 15 }):
    (∀ P Q R ∈ points, P ≠ Q ∧ Q ≠ R ∧ P ≠ R ∧ ¬ collinear ℝ (λ P : ℝ × ℝ, P.fst) ![P, Q, R]) → 
    ∃ centroids : set (ℝ × ℝ), centroids = { (m/3, n/3) | 1 ≤ m ∧ m ≤ 44 ∧ 1 ≤ n ∧ n ≤ 44 } ∧ centroids.card == 1849 :=
sorry

end centroid_positions_l373_373209


namespace min_value_distance_complex_number_l373_373062

theorem min_value_distance_complex_number (z : ℂ) (h : complex.abs (z - 1) = complex.abs (z + 2 * complex.I)) :
  ∃ m : ℝ, m = abs (z - (1 + complex.I)) ∧ m = 9 * real.sqrt 5 / 10 :=
begin
  sorry
end

end min_value_distance_complex_number_l373_373062


namespace quadrilateral_is_parallelogram_l373_373711

theorem quadrilateral_is_parallelogram
  (AB BC CD DA : ℝ)
  (K L M N : ℝ)
  (H₁ : K = (AB + BC) / 2)
  (H₂ : L = (BC + CD) / 2)
  (H₃ : M = (CD + DA) / 2)
  (H₄ : N = (DA + AB) / 2)
  (H : K + M + L + N = (AB + BC + CD + DA) / 2)
  : ∃ P Q R S : ℝ, P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P ∧ 
    (P + R = AB) ∧ (Q + S = CD)  := 
sorry

end quadrilateral_is_parallelogram_l373_373711


namespace max_students_l373_373596

def height_variant_1 : ℝ := 1.60
def height_variant_2 : ℝ := 1.22

def num_students : ℕ := 30
def num_tall_students : ℕ := 15
def num_short_students : ℕ := 15

def valid_arrangement (n : ℕ) (heights : list ℝ) : Prop :=
  (∀ i, i + 3 < n → (heights.drop i).take 4.sum > 4 * 1.50) ∧
  (∀ i, i + 6 < n → (heights.drop i).take 7.sum < 7 * 1.50)

theorem max_students (heights : list ℝ)
  (h_len : heights.length = num_students)
  (h_tall : heights.count height_variant_1 = num_tall_students)
  (h_short : heights.count height_variant_2 = num_short_students)
  (n : ℕ) (hn : valid_arrangement n heights) : n ≤ 9 := 
sorry

end max_students_l373_373596


namespace measure_of_PQ_is_p_plus_r_l373_373138

-- Definitions and conditions
variables (PQ RS PS Q S : Type) [metric_space PQ] [metric_space RS]
variables (p r : ℝ)
variables (θ : ℝ) -- angle measure variable

-- Conditions
def segments_parallel : Prop := parallel PQ RS
def angle_S_triple_angle_Q : Prop := ∠S = 3 * ∠Q
def segment_PS_measure : Prop := dist PS Q = p
def segment_RS_measure : Prop := dist RS S = r

-- Target conclusion
def measure_PQ : Prop := dist PQ RS = p + r

-- Final theorem statement
theorem measure_of_PQ_is_p_plus_r :
  segments_parallel ∧ angle_S_triple_angle_Q ∧ segment_PS_measure ∧ segment_RS_measure → measure_PQ :=
by 
    sorry

end measure_of_PQ_is_p_plus_r_l373_373138


namespace num_arrangements_l373_373893

theorem num_arrangements (A B C D E : Type) : 
  ∃ l : list (list Type), 
  ((A :: l) <|> (l ++ [A])) ∧ 
  (permutations l (C :: D :: [])) ∧
  (permutations l (D :: C :: [])) ∧
  (((A :: l) <|> (l ++ [A])).length = 120) → 
  (24 = list.countp (λ lst, lst.nth 0 ≠ A ∧ lst.nth 4 ≠ A) [A, B, C, D, E].permutations) := by sorry

end num_arrangements_l373_373893


namespace difference_is_8_l373_373657

noncomputable def difference_flour_sugar (flour_needed : ℕ) (flour_recipe : ℕ) (sugar_recipe : ℕ) (initial_sugar : ℕ) : ℕ :=
  flour_needed - sugar_recipe

theorem difference_is_8 :
  ∀ flour_needed flour_recipe sugar_recipe initial_sugar, 
    flour_needed = 21 ∧ flour_recipe = 6 ∧ sugar_recipe = 13 ∧ initial_sugar = 0 → 
    difference_flour_sugar 21 6 13 0 = 8 :=
by 
  assume flour_needed flour_recipe sugar_recipe initial_sugar,
  intro h_conditions,
  cases h_conditions with h1 h_rest,
  cases h_rest with h2 h_rest2,
  cases h_rest2 with h3 h4,
  sorry

end difference_is_8_l373_373657


namespace find_p_max_area_triangle_l373_373537

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373537


namespace integer_condition_exists_l373_373873

theorem integer_condition_exists (a : ℚ) : 
  (∀ n : ℕ, (a * n * (n + 2) * (n + 3) * (n + 4)).denom = 1) ↔ 
  ∃ k : ℤ, a = k / 6 := 
sorry

end integer_condition_exists_l373_373873


namespace complex_solutions_count_l373_373405

noncomputable def num_complex_solutions : ℂ := 2

theorem complex_solutions_count :
  let numerator := λ z : ℂ, z^4 - 1
  let denominator := λ z : ℂ, z^3 - 3z + 2
  let equation := λ z : ℂ, numerator z / denominator z
  let sol_set := {z : ℂ | numerator z = 0 ∧ denominator z ≠ 0}
  nat.card sol_set = num_complex_solutions := by
sorry

end complex_solutions_count_l373_373405


namespace log_expression_evaluation_l373_373827

open Real

theorem log_expression_evaluation : log 5 * log 20 + (log 2) ^ 2 = 1 := 
sorry

end log_expression_evaluation_l373_373827


namespace shortest_distance_from_curve_to_line_l373_373705

noncomputable def circle_center : (ℝ × ℝ) := (0, 1)
noncomputable def circle_radius : ℝ := 1

noncomputable def line : (ℝ → ℝ × ℝ) := λ t, (sqrt 3 * t + sqrt 3, -3 * t + 2)

def distance_from_point_to_line (point : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * point.fst + b * point.snd + c) / sqrt (a^2 + b^2)

theorem shortest_distance_from_curve_to_line :
  distance_from_point_to_line circle_center (sqrt 3) 1 (-5) - circle_radius = 1 :=
sorry

end shortest_distance_from_curve_to_line_l373_373705


namespace max_area_triangle_PAB_l373_373543

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373543


namespace part1_l373_373077

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

theorem part1 (U_eq : U = univ) 
  (A_eq : A = {x | (x - 5) / (x - 2) ≤ 0}) 
  (B_eq : B = {x | 1 < x ∧ x < 3}) :
  compl A ∩ compl B = {x | x ≤ 1 ∨ x > 5} := 
  sorry

end part1_l373_373077


namespace parabola_focus_distance_max_area_triangle_l373_373561

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373561


namespace sqrt_expression_is_perfect_square_l373_373150

theorem sqrt_expression_is_perfect_square (n : ℕ) (h1 : n > 0) (h2 : ∃ m : ℕ, 2 + 2 * Real.sqrt (28 * n^2 + 1) = m) :
  ∃ k : ℕ, 2 + 2 * Real.sqrt (28 * n^2 + 1) = k^2 :=
sorry

end sqrt_expression_is_perfect_square_l373_373150


namespace oranges_to_make_200_cents_profit_l373_373297

theorem oranges_to_make_200_cents_profit 
  (buy_price : ℝ) (sell_price : ℝ) (cost_oranges : ℝ) (sell_oranges : ℝ) (target_profit : ℝ) :
  buy_price = 15 →
  cost_oranges = 4 →
  sell_price = 25 →
  sell_oranges = 6 →
  target_profit = 200 →
  ∃ n : ℕ, n = 477 :=
by
  intros h1 h2 h3 h4 h5
  have cost_per_orange : ℝ := buy_price / cost_oranges
  have sell_per_orange : ℝ := sell_price / sell_oranges
  have profit_per_orange : ℝ := sell_per_orange - cost_per_orange
  have oranges_needed : ℝ := target_profit / profit_per_orange
  have rounded_oranges : ℕ := oranges_needed.ceil.to_nat
  use rounded_oranges
  sorry

-- Auxiliary lemmas and properties required to complete the proof would go here

end oranges_to_make_200_cents_profit_l373_373297


namespace weight_of_B_l373_373767

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by sorry

end weight_of_B_l373_373767


namespace divisible_2010_by_2_count_middle_zeros_1005_l373_373110

-- Define the quotient of the division
def quotient := 2010 / 2

-- Define the function to count the middle zeros in an integer
def count_middle_zeros (n : ℕ) : ℕ :=
  let s := n.toString |>.dropWhile (λ d => d = '0') |>.reverse.dropWhile (λ d => d = '0') |>.reverse
  s.filter (λ d => d = '0').length

-- Given conditions
def is_divisible (x y : ℕ) : Prop := x % y = 0
theorem divisible_2010_by_2 : is_divisible 2010 2 := by sorry

-- Main statement to prove
theorem count_middle_zeros_1005 : quotient = 1005 ∧ count_middle_zeros 1005 = 2 := by
  sorry

end divisible_2010_by_2_count_middle_zeros_1005_l373_373110


namespace parabola_focus_distance_max_area_triangle_l373_373554

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373554


namespace range_of_omega_l373_373962

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 3)

theorem range_of_omega (ω : ℝ) (hω : ω > 0) :
  (∃ x ∈ Ioo (0:ℝ) 2, is_local_max_on (f ω) (set.univ ∩ Ioo (0:ℝ) 2) x) ∧
  (∃ x ∈ Ioo (0:ℝ) 2, is_local_min_on (f ω) (set.univ ∩ Ioo (0:ℝ) 2) x) ↔
  (7*Real.pi/12 < ω ∧ ω ≤ 13*Real.pi/12) :=
sorry

end range_of_omega_l373_373962


namespace C_share_of_rent_l373_373761

noncomputable def A_oxen := 10
noncomputable def A_months := 7
noncomputable def B_oxen := 12
noncomputable def B_months := 5
noncomputable def C_oxen := 15
noncomputable def C_months := 3
noncomputable def total_rent := 140

theorem C_share_of_rent : 
  let A_usage := A_oxen * A_months,
      B_usage := B_oxen * B_months,
      C_usage := C_oxen * C_months,
      total_usage := A_usage + B_usage + C_usage,
      cost_per_ox_month := total_rent / total_usage in
  C_share = C_usage * cost_per_ox_month :=
by 
  let A_usage := A_oxen * A_months
  let B_usage := B_oxen * B_months
  let C_usage := C_oxen * C_months
  let total_usage := A_usage + B_usage + C_usage
  let cost_per_ox_month := total_rent / total_usage
  have hC_share : C_share = C_usage * cost_per_ox_month := by sorry
  exact hC_share

end C_share_of_rent_l373_373761


namespace remainder_of_large_number_div_by_101_l373_373740

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end remainder_of_large_number_div_by_101_l373_373740


namespace find_p_max_area_triangle_l373_373515

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373515


namespace decreasing_interval_iff_l373_373374

theorem decreasing_interval_iff (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → deriv (λ x, x^2 + 2*(a - 1)*x + 2) x ≤ 0) ↔ a ≤ 0 := by
  sorry

end decreasing_interval_iff_l373_373374


namespace three_digit_squares_div_by_4_count_l373_373102

theorem three_digit_squares_div_by_4_count : 
  (finset.card ((finset.filter (λ x, 
    x % 4 = 0) 
    (finset.image (λ n : ℕ, n * n) 
      (finset.range 32)).filter 
        (λ x, 100 ≤ x ∧ x < 1000))) = 11) := 
by 
  sorry

end three_digit_squares_div_by_4_count_l373_373102


namespace earl_initial_money_l373_373379

theorem earl_initial_money :
  ∃ E : ℝ, let fred_initial := 48,
               greg_initial := 36,
               debt_earl_to_fred := 28,
               debt_fred_to_greg := 32,
               debt_greg_to_earl := 40,
               earl_final := E - debt_earl_to_fred + debt_greg_to_earl,
               fred_final := fred_initial + debt_earl_to_fred - debt_fred_to_greg,
               greg_final := greg_initial + debt_fred_to_greg - debt_greg_to_earl
           in earl_final + greg_final = 130 ∧ E = 90 :=
by {
  sorry
}

end earl_initial_money_l373_373379


namespace speed_of_train_in_kmh_l373_373806

-- Define the conditions
def time_to_cross_pole : ℝ := 6
def length_of_train : ℝ := 100
def conversion_factor : ℝ := 18 / 5

-- Using the conditions to assert the speed of the train
theorem speed_of_train_in_kmh (t : ℝ) (d : ℝ) (conv_factor : ℝ) : 
  t = time_to_cross_pole → 
  d = length_of_train → 
  conv_factor = conversion_factor → 
  (d / t) * conv_factor = 50 := 
by 
  intros h_t h_d h_conv_factor
  sorry

end speed_of_train_in_kmh_l373_373806


namespace find_p_max_area_of_triangle_l373_373503

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373503


namespace equivalent_eq_l373_373113

variable {x y : ℝ}

theorem equivalent_eq (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ (y = 6 * x / (x - 9)) :=
by
  sorry

end equivalent_eq_l373_373113


namespace sqrt_subtraction_l373_373261

theorem sqrt_subtraction:
  ∀ a b c d : ℝ, a = 49 + 121 → b = 36 - 9 → sqrt a - sqrt b = sqrt 170 - 3 * sqrt 3 :=
by
  intros a b c d ha hb
  rw [ha, hb]
  sorry

end sqrt_subtraction_l373_373261


namespace first_week_gain_l373_373663

variables (x : ℝ)

def initial_investment := 400
def final_investment := 750
def first_week_investment (x : ℝ) := initial_investment + (initial_investment * (x / 100))
def second_week_investment (x : ℝ) := first_week_investment x + (0.5 * first_week_investment x)

theorem first_week_gain :
  second_week_investment x = final_investment ↔ x = 25 :=
by
  sorry

end first_week_gain_l373_373663


namespace find_p_max_area_triangle_l373_373529

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373529


namespace slope_acute_l373_373590

noncomputable def curve (a : ℤ) : ℝ → ℝ := λ x => x^3 - 2 * a * x^2 + 2 * a * x

noncomputable def tangent_slope (a : ℤ) : ℝ → ℝ := λ x => 3 * x^2 - 4 * a * x + 2 * a

theorem slope_acute (a : ℤ) : (∀ x : ℝ, (tangent_slope a x > 0)) ↔ (a = 1) := sorry

end slope_acute_l373_373590


namespace parabola_circle_distance_l373_373462

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373462


namespace problem_four_points_problem_if_four_points_l373_373232

variables {n : ℕ} {points : Fin n → ℝ × ℝ} {λ : Fin n → ℝ}
  (hλ_non_zero : ∀ i, λ i ≠ 0)
  (h_distinct_points : Function.injective points)
  (h_distance : ∀ i j, i ≠ j → (Euclidean.dist (points i) (points j))^2 = λ i + λ j)

theorem problem_four_points:
  n ≤ 4 :=
sorry

theorem problem_if_four_points (h_n_eq_4 : n = 4) :
  (1 / λ 0 + 1 / λ 1 + 1 / λ 2 + 1 / λ 3 = 0) :=
sorry

end problem_four_points_problem_if_four_points_l373_373232


namespace percentage_decrease_y_z_l373_373172

variable (k j : ℝ) {x y z x' y' z' : ℝ}

-- Given conditions
def inversely_proportional_xy (x y : ℝ) := x * y = k
def directly_inversely_proportional_xz (x z : ℝ) := x * z = j

-- New value of x
def new_value_x (x : ℝ) := 1.2 * x

-- New value of y using inverse proportionality
def new_value_y (x y : ℝ) : ℝ := y / 1.2

-- New value of z using direct proportionality
def new_value_z (x z : ℝ) : ℝ := z / 1.2

-- Percentage decrease formula
def percentage_decrease (original new : ℝ) : ℝ := (1 - new / original) * 100

-- Main theorem to prove
theorem percentage_decrease_y_z (x y z : ℝ) (h : inversely_proportional_xy x y) (h' : directly_inversely_proportional_xz x z) :
  percentage_decrease y (new_value_y x y) = 16.67 ∧ percentage_decrease z (new_value_z x z) = 16.67 :=
by sorry

end percentage_decrease_y_z_l373_373172


namespace tangent_line_equation_monotonic_intervals_l373_373457

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  - x^2 / 2 + (a - 1) * x + (2 - a) * Real.log x + 3 / 2

-- Tangent line problem statement
theorem tangent_line_equation (a : ℝ) (h : a < 3) : 
  let x := (1 : ℝ)
  let y := f x a in 
  y = a :=
sorry

-- Monotonic intervals problem statement
theorem monotonic_intervals (a : ℝ) (h : a < 3) : 
  (if 2 < a then 
    (∃ I1 I2 I3 : Set ℝ, 
      I1 = Set.Ioc (a - 2) 1 ∧
      I2 = Set.Ico 0 (a - 2) ∧
      I3 = Set.Ioi 1 ∧ 
      (∀ x : ℝ, x ∈ I1 → differentiable_on ℝ (f x) x) ∧   -- Optional: specifying differentiability
      (∀ x : ℝ, x ∈ I2 → differentiable_on ℝ (f x) x) ∧   -- Optional
      (∀ x : ℝ, x ∈ I3 → differentiable_on ℝ (f x) x)) ∧   -- Optional
  else 
    (∃ I1 I2 : Set ℝ, 
      I1 = Set.Ioc 0 1 ∧
      I2 = Set.Ioi 1 ∧
      (∀ x : ℝ, x ∈ I1 → differentiable_on ℝ (f x) x) ∧   -- Optional
      (∀ x : ℝ, x ∈ I2 → differentiable_on ℝ (f x) x)) ∧   -- Optional
) :=
sorry

end tangent_line_equation_monotonic_intervals_l373_373457


namespace S_11_l373_373925

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
-- Define that {a_n} is an arithmetic sequence.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) := ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) := ∀ n, S n = n * (a 1 + a n) / 2

-- Given condition: a_5 + a_7 = 14
def sum_condition (a : ℕ → ℕ) := a 5 + a 7 = 14

-- Prove S_{11} = 77
theorem S_11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (d : ℕ)
  (h1 : arithmetic_sequence a d)
  (h2 : sum_arithmetic_sequence S a)
  (h3 : sum_condition a) :
  S 11 = 77 := by
  -- The proof steps would follow here.
  sorry

end S_11_l373_373925


namespace find_m_l373_373978

theorem find_m {m : ℝ} (a b : ℝ × ℝ) (h₁ : a = (1, m)) (h₂ : b = (3, -2))
  (h₃ : (fst a + fst b, snd a + snd b) = (4, m - 2))
  (h₄ : ∀ k : ℝ, b = (k * (fst (fst a + fst b), snd (fst a + snd b)))) :
  m = -2 / 3 :=
sorry

end find_m_l373_373978


namespace num_arrangements_l373_373895

theorem num_arrangements (A B C D E : Type) : 
  ∃ l : list (list Type), 
  ((A :: l) <|> (l ++ [A])) ∧ 
  (permutations l (C :: D :: [])) ∧
  (permutations l (D :: C :: [])) ∧
  (((A :: l) <|> (l ++ [A])).length = 120) → 
  (24 = list.countp (λ lst, lst.nth 0 ≠ A ∧ lst.nth 4 ≠ A) [A, B, C, D, E].permutations) := by sorry

end num_arrangements_l373_373895


namespace parabola_condition_max_area_triangle_l373_373486

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373486


namespace line_through_fixed_point_l373_373926

theorem line_through_fixed_point 
    (a b F : ℝ) (h₁ : a > b) (h₂ : F > 0) 
    (h₃ : F = 3) (h₄ : 2 * b = 2) 
    (x₁ y₁ x₂ y₂ : ℝ)
    (h₅ : x₁ ≠ x₂)
    (h₆ : (x₁^2 / a^2 + y₁^2 = 1))
    (h₇ : (x₂^2 / a^2 + y₂^2 = 1))
    (h₈ : y₁ * y₂ > 0)
    (h₉ : ∠(A - F M) = ∠(B - F N)) :
    ∃ (P : ℝ × ℝ), P = (10 / 3, 0) ∧
    (line_through A B) :=
sorry

end line_through_fixed_point_l373_373926


namespace bill_cooking_time_l373_373352

def total_time_spent 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ) : ℕ :=
num_peppers * chop_pepper_time + 
num_onions * chop_onion_time + 
num_omelets * grate_cheese_time + 
num_omelets * cook_omelet_time

theorem bill_cooking_time 
  (chop_pepper_time : ℕ) (chop_onion_time : ℕ)
  (grate_cheese_time : ℕ) (cook_omelet_time : ℕ)
  (num_peppers : ℕ) (num_onions : ℕ)
  (num_omelets : ℕ)
  (chop_pepper_time_eq : chop_pepper_time = 3)
  (chop_onion_time_eq : chop_onion_time = 4)
  (grate_cheese_time_eq : grate_cheese_time = 1)
  (cook_omelet_time_eq : cook_omelet_time = 5)
  (num_peppers_eq : num_peppers = 4)
  (num_onions_eq : num_onions = 2)
  (num_omelets_eq : num_omelets = 5) :
  total_time_spent chop_pepper_time chop_onion_time grate_cheese_time cook_omelet_time num_peppers num_onions num_omelets = 50 :=
by {
  sorry
}

end bill_cooking_time_l373_373352


namespace weight_gain_ratio_l373_373348

variable (J O F : ℝ)

theorem weight_gain_ratio :
  O = 5 ∧ F = (1/2) * J - 3 ∧ 5 + J + F = 20 → J / O = 12 / 5 :=
by
  intros h
  cases' h with hO h'
  cases' h' with hF hTotal
  sorry

end weight_gain_ratio_l373_373348


namespace disjoint_arithmetic_sequences_exists_l373_373419

theorem disjoint_arithmetic_sequences_exists
  (n1 n2 n3 ... n10000 : ℕ)
  (distinct_n : ∀ i j, 1 ≤ i ∧ i ≤ 10000 → 1 ≤ j ∧ j ≤ 10000 → i ≠ j → n1 ≠ n2)
  (largest_prime_power : ∃ p a, p.prime ∧ (∀ i, 1 ≤ i ∧ i ≤ 10000 → ∃ m1 m2 ... mt, n_i = m1 * m2 * ... * mt ∧ max_pow m1 m2 ... mt = p^a)) :
  ∃ a_1 a_2 ... a_10000 : ℕ, pairwise_disjoint (λi, λ k, a_i + k * n_i) :=
begin
  sorry
end

end disjoint_arithmetic_sequences_exists_l373_373419


namespace problem_proof_l373_373615

variables {A B C M N D E : Type} [add_comm_group A] [vector_space ℚ A]

-- Midpoints condition
variables (AB AC : A)
variables (M N : A)
variables {AM MB : ℚ} (hM : M = AM • AB + MB • AC) (hmidM : AM = 1/2) (hmidMB : MB = 1/2)

variables {AN NC : ℚ} (hN : N = AN • AB + NC • AC) (hmidN : AN = 1/2) (hmidNC : NC = 1/2)

-- Points on line segment BN and parallel condition
variables (BN BD BE : A)
variables {CD ME : A} (hME : ME = CD) (hCD_parallel_ME : is_parallel CD ME)
variables {BD_lt_BE : BD < BE}

-- Conclusion
theorem problem_proof : BD = 2 • (EN) :=
by 
  sorry  -- Proof goes here

end problem_proof_l373_373615


namespace order_of_magnitude_l373_373439

variables (a b c : ℝ)

noncomputable def x := real.sqrt (a^2 + (b + c)^2)
noncomputable def y := real.sqrt (b^2 + (c + a)^2)
noncomputable def z := real.sqrt (c^2 + (a + b)^2)

theorem order_of_magnitude (h1 : a > b) (h2 : b > c) (h3 : c > 0) : z > y ∧ y > x := 
by sorry

end order_of_magnitude_l373_373439


namespace water_left_in_bucket_l373_373267

-- Definitions of the conditions
def Jimin_drinks_ml := 150
def Taehyung_drinks_L := 1.65
def initial_amount_L := 30
def conversion_factor := 1000

-- Conversion and statement to prove
def conversion_ml_to_L (ml : ℝ) : ℝ := ml / conversion_factor

-- The amount Jimin drinks in liters
def Jimin_drinks_L := conversion_ml_to_L Jimin_drinks_ml

-- The statement to prove
theorem water_left_in_bucket : initial_amount_L - (Jimin_drinks_L + Taehyung_drinks_L) = 28.20 :=
by
symmetric
sorry

end water_left_in_bucket_l373_373267


namespace calculate_product_l373_373353

theorem calculate_product : (97 * 103) = 9991 := by
  have h1 : 97 = 100 - 3 := by rfl
  have h2 : 103 = 100 + 3 := by rfl
  calc 97 * 103 = (100 - 3) * (100 + 3) : by rw [h1, h2]
          ... = 100^2 - 3^2 : by sorry
          ... = 10000 - 9 : by sorry
          ... = 9991 : by sorry

end calculate_product_l373_373353


namespace additional_charge_per_2_5_mile_l373_373147

theorem additional_charge_per_2_5_mile (x : ℝ) : 
  (∀ (total_charge distance charge_per_segment initial_fee : ℝ),
    total_charge = 5.65 →
    initial_fee = 2.5 →
    distance = 3.6 →
    charge_per_segment = (3.6 / (2/5)) →
    total_charge = initial_fee + charge_per_segment * x → 
    x = 0.35) :=
by
  intros total_charge distance charge_per_segment initial_fee
  intros h_total_charge h_initial_fee h_distance h_charge_per_segment h_eq
  sorry

end additional_charge_per_2_5_mile_l373_373147


namespace find_number_l373_373777

theorem find_number (x : ℕ) (h : x + 18 = 44) : x = 26 :=
by
  sorry

end find_number_l373_373777


namespace remainder_of_large_number_div_by_101_l373_373739

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end remainder_of_large_number_div_by_101_l373_373739


namespace parabola_focus_distance_max_area_triangle_l373_373555

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373555


namespace find_p_max_area_triangle_l373_373514

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373514


namespace agent_capacities_l373_373836

-- Define capacities of the agents as variables
variables (jan john missy nancy peter : ℕ)

-- Define the conditions
def conditions : Prop :=
  jan = 25 ∧
  john = jan + jan * 3 / 10 ∧
  missy = john + 20 ∧
  nancy = (jan + john + missy) / 3 - ((jan + john + missy) / 3) * 1 / 10 ∧
  peter = 2 * (nancy + jan) / 3

-- Define the correct answers
def correct_answers : Prop :=
  jan = 25 ∧ john = 33 ∧ missy = 53 ∧ nancy = 33 ∧ peter = 39

-- The proof problem: proving that given the conditions, the capacities match the correct answers
theorem agent_capacities : conditions jan john missy nancy peter → correct_answers jan john missy nancy peter :=
by
  intro h
  rcases h with ⟨h_jan, h_john, h_missy, h_nancy, h_peter⟩
  split
  { exact h_jan }
  split
  { exact nat.eq_of_le_of_lt_succ (by linarith) (by linarith) }
  split
  { exact h_missy }
  split
  { exact nat.eq_of_le_of_lt_succ (by linarith) (by linarith) }
  { exact nat.eq_of_le_of_lt_succ (by linarith) (by linarith) }

end agent_capacities_l373_373836


namespace problem_a_l373_373276

noncomputable def convex_polyhedron (P : Type*) :=
  ∃ (V E F : ℕ), V - E + F = 2 ∧ (∀ k, k ≥ 3 → ∑ (γ : ℕ) in range k, γ < 2 * F)

theorem problem_a (P : Type*) [convex_polyhedron P] : 
  (∃ F : ℕ, F < 4) ∨ (∃ V : ℕ, V < 4) :=
sorry

end problem_a_l373_373276


namespace chord_intersects_diameter_angle_45_l373_373001
noncomputable theory

open Real
open_locale real

/-- Given a circle with center O and radius r, and a chord AC intersecting the
diameter at point B with angle AC makes with the diameter being 45 degrees,
prove that AB^2 + BC^2 = 2r^2. -/
theorem chord_intersects_diameter_angle_45 
  (O A C B : Point) (r : ℝ)
  (h_circle : dist O A = r ∧ dist O C = r)
  (h_chord : ∠ A O B = ∠ B O C)
  (h_angle : ∠ A B C = π / 4) :
  dist A B ^ 2 + dist B C ^ 2 = 2 * r^2 :=
sorry

end chord_intersects_diameter_angle_45_l373_373001


namespace person_is_not_sane_l373_373770

-- Definitions
def Person : Type := sorry
def sane : Person → Prop := sorry
def human : Person → Prop := sorry
def vampire : Person → Prop := sorry
def declares (p : Person) (s : String) : Prop := sorry

-- Conditions
axiom transylvanian_declares_vampire (p : Person) : declares p "I am a vampire"
axiom sane_human_never_claims_vampire (p : Person) : sane p → human p → ¬ declares p "I am a vampire"
axiom sane_vampire_never_admits_vampire (p : Person) : sane p → vampire p → ¬ declares p "I am a vampire"
axiom insane_human_might_claim_vampire (p : Person) : ¬ sane p → human p → declares p "I am a vampire"
axiom insane_vampire_might_admit_vampire (p : Person) : ¬ sane p → vampire p → declares p "I am a vampire"

-- Proof statement
theorem person_is_not_sane (p : Person) : declares p "I am a vampire" → ¬ sane p :=
by
  intros h
  sorry

end person_is_not_sane_l373_373770


namespace identical_closed_curves_on_surfaces_l373_373976

theorem identical_closed_curves_on_surfaces
  (S1 S2 : Type) [TopologicalSurface S1] [TopologicalSurface S2] :
  ∃ C : Set ℝ, ClosedCurveOnSurface C S1 ∧ ClosedCurveOnSurface C S2 ∧ IdenticalCurves C C :=
sorry

end identical_closed_curves_on_surfaces_l373_373976


namespace intersection_unique_l373_373093

theorem intersection_unique : ∃! (x y : ℝ), y = abs (3 * x + 6) ∧ y = -abs (2 * x - 1) :=
by
  sorry

end intersection_unique_l373_373093


namespace triangle_perimeter_not_78_l373_373246

theorem triangle_perimeter_not_78 (x : ℝ) (h1 : 11 < x) (h2 : x < 37) : 13 + 24 + x ≠ 78 :=
by
  -- Using the given conditions to show the perimeter is not 78
  intro h
  have h3 : 48 < 13 + 24 + x := by linarith
  have h4 : 13 + 24 + x < 74 := by linarith
  linarith

end triangle_perimeter_not_78_l373_373246


namespace mary_average_speed_l373_373658

noncomputable def average_speed (d1 d2 : ℝ) (t1 t2 : ℝ) : ℝ :=
  (d1 + d2) / ((t1 + t2) / 60)

theorem mary_average_speed :
  average_speed 1.5 1.5 45 15 = 3 := by
  sorry

end mary_average_speed_l373_373658


namespace affine_vector_preservation_l373_373769

variables {Point : Type} [AffineSpace Point ℝ]

def affine_transformation := Point → Point

variables (f : affine_transformation)

variables {A B C D A1 B1 C1 D1 : Point}

-- Conditions
variables (h_transformation : f A = A1 ∧ f B = B1 ∧ f C = C1 ∧ f D = D1)
variables (h_equality : (B -ᵥ A : Point) = (D -ᵥ C))

-- Lean statement to prove
theorem affine_vector_preservation : 
  (B1 -ᵥ A1 : Point) = (D1 -ᵥ C1) :=
by
  sorry

end affine_vector_preservation_l373_373769


namespace highest_grade_25_lowest_grade_0_high_low_grades_same_l373_373243

-- Define the function representing the grade after punishment
def grade_after_punishment (x : ℝ) : ℝ :=
  x - (x^2) / 100

-- (a) Prove that the highest grade is 25
theorem highest_grade_25 :
  ∃ x ∈ set.Icc 0 100, grade_after_punishment x = 25 :=
sorry

-- (b) Prove that the lowest grade is 0
theorem lowest_grade_0 :
  ∀ x ∈ set.Icc 0 100, grade_after_punishment x = 0 :=
sorry

-- (c) Confirm students with scores near 0 or 100 end up with the same grade
theorem high_low_grades_same:
  ∀ x ∈ {0, 100}, grade_after_punishment x = 0 :=
sorry

end highest_grade_25_lowest_grade_0_high_low_grades_same_l373_373243


namespace sqrt_subtraction_proof_l373_373257

def sqrt_subtraction_example : Real := 
  let a := 49 + 121
  let b := 36 - 9
  Real.sqrt a - Real.sqrt b

theorem sqrt_subtraction_proof : sqrt_subtraction_example = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end sqrt_subtraction_proof_l373_373257


namespace max_area_triangle_PAB_l373_373550

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373550


namespace trajectory_of_P_l373_373588

def point := ℝ × ℝ
def distance (P Q : point) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem trajectory_of_P :
  ∀ (P M N : point), M = (3, 0) → N = (1, 0) → abs (distance P M - distance P N) = 2 → 
  ∃ (ray_origin : point) (direc : point → point), ∀ (t : ℝ), direc (ray_origin.1 + t * (direc P).1, ray_origin.2 + t * (direc P).2) := 
by
  intro P M N hM hN hDist
  sorry

end trajectory_of_P_l373_373588


namespace four_digit_numbers_thousands_digit_5_div_by_5_l373_373981

theorem four_digit_numbers_thousands_digit_5_div_by_5 :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 5000 ≤ x ∧ x ≤ 5999 ∧ x % 5 = 0) ∧ s.card = 200 :=
by
  sorry

end four_digit_numbers_thousands_digit_5_div_by_5_l373_373981


namespace quadrilateral_area_l373_373188

theorem quadrilateral_area (S1 S2 S3 : Circle) (r a : ℝ) 
(h1 : S1 ≠ S2) (h2 : S2 ≠ S3) (h3 : S3 ≠ S1)
(h4 : inscribed_angle S1 S2 S3 (60 : ℝ)) 
(h5 : radius S2 = r) 
(h6 : radius S3 - radius S1 = a) :
  area_quadrilateral_tangents S1 S2 S3 = a * r * √3 :=
sorry

end quadrilateral_area_l373_373188


namespace find_length_AP_l373_373140

-- Define the lengths as constants
def side_length_square := 8
def length_ZY := 12
def length_XY := 8
def shaded_area := (1/3) * (length_ZY * length_XY)

-- Conditions and Question
def AD := side_length_square
def WX := length_XY
def perpendicular_AD_WX := true  -- representing that AD and WX are perpendicular

-- To find: Length of AP such that the shaded area is one third of WXYZ
def AP : ℝ := AD - (shaded_area / side_length_square)

theorem find_length_AP :
  AP = 4 :=
by
  -- actual proof steps would go here
  sorry

end find_length_AP_l373_373140


namespace remainder_2468135792_mod_101_l373_373742

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end remainder_2468135792_mod_101_l373_373742


namespace remainder_of_large_number_div_by_101_l373_373741

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end remainder_of_large_number_div_by_101_l373_373741


namespace sum_of_palindromes_522729_is_1366_l373_373708

def is_palindrome (n : Nat) : Prop :=
  let s := toString n
  s == s.reverse

def is_three_digit (n : Nat) : Prop := n >= 100 ∧ n < 1000

theorem sum_of_palindromes_522729_is_1366 :
  ∃ (x y : Nat), is_three_digit x ∧ is_three_digit y ∧ is_palindrome x ∧ is_palindrome y ∧ x * y = 522729 ∧ x + y = 1366 := 
sorry

end sum_of_palindromes_522729_is_1366_l373_373708


namespace exists_n_no_rational_solution_l373_373916

noncomputable def quadratic_polynomial (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem exists_n_no_rational_solution {a b c : ℝ} (h : ∃ n : ℕ, ¬∃ x : ℚ, quadratic_polynomial a b c x = (1 : ℝ) / n) :
  ∃ n : ℕ, ¬∃ x : ℚ, quadratic_polynomial a b c x = 1 / n :=
begin
  sorry
end

end exists_n_no_rational_solution_l373_373916


namespace sum_mod_remainder_l373_373753

theorem sum_mod_remainder :
  (∑ i in Finset.range 21, i % 9) = 3 :=
by sorry

end sum_mod_remainder_l373_373753


namespace find_constants_l373_373639

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 0], ![2, -4]]

noncomputable def N_inv : Matrix (Fin 2) (Fin 2) ℚ := N⁻¹

theorem find_constants :
  ∃ c d : ℚ, N_inv = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

end find_constants_l373_373639


namespace quadratic_roots_problem_l373_373167

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end quadratic_roots_problem_l373_373167


namespace range_of_a_l373_373181

noncomputable def f (x a : ℝ) : ℝ := ((Real.log x) / x) + x - a

def curve (x : ℝ) : ℝ := (2 * Real.exp (x + 1)) / (Real.exp (2 * x) + 1)

theorem range_of_a (a : ℝ) :
  (∃ (x y : ℝ), y = curve x ∧ f (f y a) a = y) →
  a ∈ Iic (1 / Real.exp 1) :=
by
  intro hyp
  sorry

end range_of_a_l373_373181


namespace polynomial_divisibility_l373_373017

theorem polynomial_divisibility (P : Polynomial ℝ) (h_nonconstant : ∃ n : ℕ, P.degree = n ∧ n ≥ 1)
  (h_div : ∀ x : ℝ, P.eval (x^3 + 8) = 0 → P.eval (x^2 - 2*x + 4) = 0) :
  ∃ a : ℝ, ∃ n : ℕ, a ≠ 0 ∧ P = Polynomial.C a * Polynomial.X ^ n :=
sorry

end polynomial_divisibility_l373_373017


namespace parabola_focus_distance_max_area_triangle_l373_373560

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373560


namespace max_sales_on_40th_day_l373_373321

-- Define the conditions
def g (t : ℕ) : ℝ := -t + 110
def f (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 40 then t + 8
  else if 41 ≤ t ∧ t ≤ 100 then -0.5 * t + 69
  else 0

-- Define the piecewise function for daily sales S
noncomputable def S : ℕ → ℝ
| t =>
  if 1 ≤ t ∧ t ≤ 40 then (-t * t + 102 * t + 880)
  else if 41 ≤ t ∧ t ≤ 100 then (0.5 * t * t - 124 * t + 7590)
  else 0

-- State the theorem
theorem max_sales_on_40th_day : 
  ∃ t, 1 ≤ t ∧ t ≤ 100 ∧ S t = 3360 ∧ (∀ u, 1 ≤ u ∧ u ≤ 100 → S u ≤ S 40) :=
begin
  use 40,
  split,
  { -- prove 1 ≤ 40 ≤ 100 
    exact ⟨le_refl 1, le_refl 40⟩, -- obvious properties about 40
  },
  split,
  { -- prove S 40 = 3360
    sorry,
  },
  { -- prove ∀ u, 1 ≤ u ≤ 100 →  S u ≤ S 40
    sorry,
  }
end

end max_sales_on_40th_day_l373_373321


namespace train_speed_in_km_per_hr_l373_373812

-- Define the conditions
def length_of_train : ℝ := 100 -- length in meters
def time_to_cross_pole : ℝ := 6 -- time in seconds

-- Define the conversion factor from meters/second to kilometers/hour
def conversion_factor : ℝ := 18 / 5

-- Define the formula for speed calculation
def speed_of_train := (length_of_train / time_to_cross_pole) * conversion_factor

-- The theorem to be proven
theorem train_speed_in_km_per_hr : speed_of_train = 50 := by
  sorry

end train_speed_in_km_per_hr_l373_373812


namespace smallest_m_plus_n_l373_373035

theorem smallest_m_plus_n (m n : ℕ) (h1 : m > n) (h2 : n ≥ 1) 
(h3 : 1000 ∣ 1978^m - 1978^n) : m + n = 106 :=
sorry

end smallest_m_plus_n_l373_373035


namespace novel_pages_l373_373760

theorem novel_pages (x : ℕ) (pages_per_day_in_reality : ℕ) (planned_days actual_days : ℕ)
  (h1 : planned_days = 20)
  (h2 : actual_days = 15)
  (h3 : pages_per_day_in_reality = x + 20)
  (h4 : pages_per_day_in_reality * actual_days = x * planned_days) :
  x * planned_days = 1200 :=
by
  sorry

end novel_pages_l373_373760


namespace min_distance_squared_l373_373444

noncomputable def graph_function1 (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_graph1 (a b : ℝ) : Prop := b = graph_function1 a

noncomputable def graph_function2 (x : ℝ) : ℝ := x + 2

noncomputable def point_on_graph2 (c d : ℝ) : Prop := d = graph_function2 c

theorem min_distance_squared (a b c d : ℝ) 
  (hP : point_on_graph1 a b)
  (hQ : point_on_graph2 c d) :
  (a - c)^2 + (b - d)^2 = 8 := 
sorry

end min_distance_squared_l373_373444


namespace product_of_three_numbers_l373_373712

theorem product_of_three_numbers 
  (a b c : ℕ) 
  (h1 : a + b + c = 300) 
  (h2 : 9 * a = b - 11) 
  (h3 : 9 * a = c + 15) : 
  a * b * c = 319760 := 
  sorry

end product_of_three_numbers_l373_373712


namespace find_p_max_area_triangle_l373_373531

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373531


namespace fruit_basket_problem_l373_373111

noncomputable def numberOfFruitBaskets (maxApples maxOranges maxFruits : ℕ) : ℕ :=
  (finset.range (maxApples + 1)).sum
    (λ a => (finset.range (maxOranges + 1)).filter (λ o => a + o ≤ maxFruits).card)

theorem fruit_basket_problem :
  numberOfFruitBaskets 6 8 10 = 56 :=
by
  sorry

end fruit_basket_problem_l373_373111


namespace smallest_non_consecutive_triplet_value_l373_373254

def non_consecutive (x y : ℕ) : Prop :=
  x ≠ y + 1 ∧ x + 1 ≠ y

def selected_set : set ℕ := {2, 3, 5, 7, 11, 13}

def valid_triplet (a b c : ℕ) : Prop :=
  a ∈ selected_set ∧ b ∈ selected_set ∧ c ∈ selected_set ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  non_consecutive a b ∧ non_consecutive b c ∧ non_consecutive a c

def smallest_result := min { (a + b) * c | a b c, valid_triplet a b c }

theorem smallest_non_consecutive_triplet_value :
  smallest_result = 24 :=
by
  sorry

end smallest_non_consecutive_triplet_value_l373_373254


namespace parabola_expression_and_point_check_l373_373057

theorem parabola_expression_and_point_check :
  (∀ (a b : ℝ), (y = a * x^2 - b * x + 3) ∧ ((2 : ℝ) = a - b + 3) ∧ ((3 : ℝ) = 4 * a - 2 * b + 3) ->
    (a = 0 ∧ b = 1 ∧ y = x^2 - 2 * x + 3) ∧ 
    ¬((5 : ℝ) = (-1 : ℝ)^2 - 2 * (-1 : ℝ) + 3)) := 
begin
  sorry
end

end parabola_expression_and_point_check_l373_373057


namespace AndyCoordinatesAfter1500Turns_l373_373820

/-- Definition for Andy's movement rules given his starting position. -/
def AndyPositionAfterTurns (turns : ℕ) : ℤ × ℤ :=
  let rec move (x y : ℤ) (length : ℤ) (dir : ℕ) (remainingTurns : ℕ) : ℤ × ℤ :=
    match remainingTurns with
    | 0 => (x, y)
    | n+1 => 
        let (dx, dy) := match dir % 4 with
                        | 0 => (0, 1)
                        | 1 => (1, 0)
                        | 2 => (0, -1)
                        | _ => (-1, 0)
        move (x + dx * length) (y + dy * length) (length + 1) (dir + 1) n
  move (-30) 25 2 0 turns

theorem AndyCoordinatesAfter1500Turns :
  AndyPositionAfterTurns 1500 = (-280141, 280060) :=
by
  sorry

end AndyCoordinatesAfter1500Turns_l373_373820


namespace f_sqrt_50_l373_373178

def f (x : ℝ) : ℝ := 
  if x ∈ (Set.Icc 7.07 7.08) 
  then ⌊real.sqrt 50⌋ + 6 
  else 7 * x + 3

theorem f_sqrt_50 : f (real.sqrt 50) = 13 := 
by
  sorry

end f_sqrt_50_l373_373178


namespace triangle_area_base_3_height_4_l373_373249

theorem triangle_area_base_3_height_4 :
  ∀ (b h : ℝ), b = 3 → h = 4 → (1 / 2) * b * h = 6 :=
by
  intros b h hb hh
  rw [hb, hh]
  norm_num
  sorry

end triangle_area_base_3_height_4_l373_373249


namespace tan_2x_eq_sin_x_has_three_solutions_l373_373098

theorem tan_2x_eq_sin_x_has_three_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.sin x) ∧ S.card = 3 :=
by
  sorry

end tan_2x_eq_sin_x_has_three_solutions_l373_373098


namespace find_number_of_each_coin_l373_373623

-- Define the number of coins
variables (n d q : ℕ)

-- Given conditions
axiom twice_as_many_nickels_as_quarters : n = 2 * q
axiom same_number_of_dimes_as_quarters : d = q
axiom total_value_of_coins : 5 * n + 10 * d + 25 * q = 1520

-- Statement to prove
theorem find_number_of_each_coin :
  q = 304 / 9 ∧
  n = 2 * (304 / 9) ∧
  d = 304 / 9 :=
sorry

end find_number_of_each_coin_l373_373623


namespace expansion_three_times_expansion_six_times_l373_373975

-- Definition for the rule of expansion
def expand (a b : Nat) : Nat := a * b + a + b

-- Problem 1: Expansion with a = 1, b = 3 for 3 times results in 255.
theorem expansion_three_times : expand (expand (expand 1 3) 7) 31 = 255 := sorry

-- Problem 2: After 6 operations, the expanded number matches the given pattern.
theorem expansion_six_times (p q : ℕ) (hp : p > q) (hq : q > 0) : 
  ∃ m n, m = 8 ∧ n = 13 ∧ (expand (expand (expand (expand (expand (expand q (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) = (q + 1) ^ m * (p + 1) ^ n - 1 :=
sorry

end expansion_three_times_expansion_six_times_l373_373975


namespace intersection_range_l373_373074

noncomputable def f (a : ℝ) (x : ℝ) := a * x
noncomputable def g (x : ℝ) := Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := f a x - g x

theorem intersection_range (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = g x1 ∧ f a x2 = g x2) ↔
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

end intersection_range_l373_373074


namespace problem2_problem3_combined_problem_l373_373049

/-- Definitions of sets P and Q based on given conditions -/
def P (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def Q (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

/-- Problem 1: Define the complement of P in real numbers. -/
def complement_P : set ℝ := {x : ℝ | x < -2 ∨ x > 10}

/-- Problem 2: Define the condition P ⊆ Q and find the range for m. -/
theorem problem2 (m : ℝ) : (∀ x, P x → Q x m) ↔ 9 ≤ m := sorry

/-- Problem 3: Define the condition P ∩ Q = Q and find the range for m. -/
theorem problem3 (m : ℝ) : (∀ x, Q x m → P x) ↔ (0 ≤ m ∧ m ≤ 9) := sorry

/-- Final problem combining the results -/
theorem combined_problem (m : ℝ) : 
  (∀ x, P x → Q x m) ∧ (∀ x, Q x m → P x) ↔ (m ∈ set.Icc 9 (⊤ : ℝ)) := sorry

end problem2_problem3_combined_problem_l373_373049


namespace count_a_less_than_100_satisfying_condition_l373_373174

theorem count_a_less_than_100_satisfying_condition :
  {a : ℕ // a < 100 ∧ (a^3 + 23) % 24 = 0}.card = 9 :=
begin
  sorry,
end

end count_a_less_than_100_satisfying_condition_l373_373174


namespace area_PQR_eq_S_l373_373692

variables {Point : Type} [EuclideanGeometry Point]

-- Given variables and conditions
variable (J K L M P Q R : Point)
variable (S : ℝ)
variable (h_area_JKL : area J K L = S)
variable (h_mid_M : midpoint M K L)
variable (h_JP : dist J P = 2 * dist J L)
variable (h_JQ : dist J Q = 3 * dist J M)
variable (h_JR : dist J R = 4 * dist J K)

-- Goal: Prove that the area of triangle PQR is S
theorem area_PQR_eq_S 
  (h_area_JKL : area J K L = S)
  (h_mid_M : midpoint M K L)
  (h_JP : dist J P = 2 * dist J L)
  (h_JQ : dist J Q = 3 * dist J M)
  (h_JR : dist J R = 4 * dist J K) :
  area P Q R = S :=
by
  sorry

end area_PQR_eq_S_l373_373692


namespace number_of_occurrences_of_1973_l373_373347

-- Definitions for the problem
def initial_segment : list ℕ := [1, 1]
def steps : ℕ := 1973
def target_number : ℕ := 1973

-- The process to insert sums between neighboring numbers for a given number of steps
def generate_sequence (initial : list ℕ) (n : ℕ) : list ℕ :=
  -- A placeholder definition, in reality, this should implement the summing logic.
  sorry

-- Euler's totient function for a given number
def euler_totient (n : ℕ) : ℕ :=
  -- A placeholder definition, in reality, this should implement the totient function logic.
  sorry

-- The main theorem to prove the number of occurrences of 1973 in the sequence after 1973 steps
theorem number_of_occurrences_of_1973 :
  ∀ seq, seq = generate_sequence initial_segment steps → list.count_seq target_number seq = euler_totient target_number :=
begin
  sorry
end

end number_of_occurrences_of_1973_l373_373347


namespace parabola_circle_distance_l373_373463

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373463


namespace find_p_max_area_triangle_l373_373541

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373541


namespace express_An_l373_373171

noncomputable def A_n (A : ℝ) (n : ℤ) : ℝ :=
  (1 / 2^n) * ((A + (A^2 - 4).sqrt)^n + (A - (A^2 - 4).sqrt)^n)

theorem express_An (a : ℝ) (A : ℝ) (n : ℤ) (h : a + a⁻¹ = A) :
  (a^n + a^(-n)) = A_n A n := 
sorry

end express_An_l373_373171


namespace no_primes_between_factorial_plus_3_and_factorial_plus_2n_l373_373023

theorem no_primes_between_factorial_plus_3_and_factorial_plus_2n (n : ℕ) (h : n > 2) :
  ∀ k, n! + 3 ≤ k → k ≤ n! + 2n → ¬ prime k :=
by
  sorry

end no_primes_between_factorial_plus_3_and_factorial_plus_2n_l373_373023


namespace solve_for_y_l373_373000

theorem solve_for_y : ∀ (y : ℝ), (∛ (y * (y^5)^(1/2))) = 4 → y = 4^(6/7) :=
begin
    intro y,
    intro h,
    sorry
end

end solve_for_y_l373_373000


namespace max_area_triangle_PAB_l373_373547

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373547


namespace no_enemies_seated_next_to_each_other_l373_373289

theorem no_enemies_seated_next_to_each_other (n : ℕ) (ambassadors : Finset (Fin (2 * n))) (is_enemy : Fin (2 * n) → Fin (2 * n) → Prop) :
  (∀ a, ∀ b, is_enemy a b → a ≠ b ∧ (∃ k, 0 < k ∧ k < n ∧ is_enemy a b)) →
  (∃ arrangement : List (Fin (2 * n)),
    arrangement.Nodup ∧
    (∀ i, 0 ≤ i ∧ i < 2 * n → ¬ is_enemy (arrangement.nthLe i (by linarith)) (arrangement.nthLe ((i + 1) % (2 * n)) sorry))) := 
sorry

end no_enemies_seated_next_to_each_other_l373_373289


namespace max_area_triangle_PAB_l373_373544

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373544


namespace complex_expression_evaluation_l373_373355

theorem complex_expression_evaluation (i : ℂ) (h : i^2 = -1) : i^3 * (1 - i)^2 = -2 :=
by
  -- Placeholder for the actual proof which is skipped here
  sorry

end complex_expression_evaluation_l373_373355


namespace min_value_of_a_l373_373943

theorem min_value_of_a : ∃ a : ℝ, (∀ x : ℝ, 9 * x - (4 + a) * 3 * x + 4 = 0 → ∃ x1 x2 : ℝ, 9 * x1 - (4 + a) * 3 * x1 + 4 = 0 ∧ 9 * x2 - (4 + a) * 3 * x2 + 4 = 0 ∧ x1 ≠ x2) ∧ a = 0 :=
begin
  sorry -- Proof steps omitted
end

end min_value_of_a_l373_373943


namespace y_intercept_range_l373_373224

open Real

theorem y_intercept_range (k : ℝ) (b : ℝ) :
  (∃ (x1 x2 y1 y2 : ℝ), 
    x1 ≠ x2 ∧ 
    (y1 = k * x1 + 1 ∧ y2 = k * x2 + 1) ∧ 
    (x1^2 - y1^2 = 1 ∧ x2^2 - y2^2 = 1) ∧ 
    (-2, 0) ∈ l ∧ 
    ((x1 + x2) / 2, (y1 + y2) / 2) ∈ l ∧ 
    b = y-value of line l at x = 0) →
  b ∈ (- ∞, -2 - sqrt 2) ∪ (2, ∞) := sorry

end y_intercept_range_l373_373224


namespace math_problem_l373_373025

-- Condition: For every x, the action [x] is defined as the greatest integer less than or equal to x.
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Given x = 6.5
def x : ℝ := 6.5

-- Evaluate the expression based on the conditions given
theorem math_problem :
  floor x * floor (2 / 3) + floor 2 * 7.2 + floor 8.3 - 6.6 = 15.8 :=
by
  sorry

end math_problem_l373_373025


namespace max_cardinality_valid_subset_l373_373428

-- Define the conditions as Lean 4 definitions
def I : Set (ℕ × ℕ × ℕ × ℕ) :=
  { x ∣ ∀ i, 1 ≤ (x.fst).fst ∧ (x.fst).fst ≤ 11 }

-- Define the condition for subset A
def valid_subset_A (A : Set (ℕ × ℕ × ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ × ℕ × ℕ × ℕ) (i j : ℕ), 
    x ∈ A → y ∈ A → 1 ≤ i → i < j → j ≤ 4 → (x.fst.fst - x.fst.snd) * (y.fst.fst - y.fst.snd) < 0

-- Define the maximum cardinality of a valid subset
theorem max_cardinality_valid_subset :
  ∃ A : Set (ℕ × ℕ × ℕ × ℕ), A ⊆ I ∧ valid_subset_A A ∧ #A = 24 := sorry

end max_cardinality_valid_subset_l373_373428


namespace factorial_expression_simplification_l373_373846

theorem factorial_expression_simplification (N : ℕ) :
  (N^2-1)! * N^2 / (N+2)! = 
  1 / ((N^2 + 1) * (N^2 + 2) * … * (N^2 + N + 2)) := 
by
  sorry

end factorial_expression_simplification_l373_373846


namespace masha_can_obtain_upto_1093_l373_373614

theorem masha_can_obtain_upto_1093 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1093 →
  ∃ (c0 c1 c2 c3 c4 c5 c6 : ℤ),
      n = c0 * 1 + c1 * 3 + c2 * 9 + c3 * 27 + c4 * 81 + c5 * 243 + c6 * 729 ∧
      (c0 = 0 ∨ c0 = 1 ∨ c0 = -1) ∧
      (c1 = 0 ∨ c1 = 1 ∨ c1 = -1) ∧
      (c2 = 0 ∨ c2 = 1 ∨ c2 = -1) ∧
      (c3 = 0 ∨ c3 = 1 ∨ c3 = -1) ∧
      (c4 = 0 ∨ c4 = 1 ∨ c4 = -1) ∧
      (c5 = 0 ∨ c5 = 1 ∨ c5 = -1) ∧
      (c6 = 0 ∨ c6 = 1 ∨ c6 = -1) :=
begin
  sorry
end

end masha_can_obtain_upto_1093_l373_373614


namespace max_gcd_coprime_l373_373398

theorem max_gcd_coprime (x y : ℤ) (h : Int.gcd x y = 1) : 
  Int.gcd (x + 2015 * y) (y + 2015 * x) ≤ 4060224 :=
sorry

end max_gcd_coprime_l373_373398


namespace tangent_line_at_origin_tangent_line_passing_through_neg1_neg3_l373_373459

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2

theorem tangent_line_at_origin :
  (∀ x y : ℝ, y = f x → x = 0 → y = 2 * x) := by
  sorry

theorem tangent_line_passing_through_neg1_neg3 :
  (∀ x y : ℝ, y = f x → (x, y) ≠ (-1, -3) → y = 5 * x + 2) := by
  sorry

end tangent_line_at_origin_tangent_line_passing_through_neg1_neg3_l373_373459


namespace translation_of_complex_l373_373329

def translation (z w : Complex) : Complex := z + w

theorem translation_of_complex :
  ∃ w : Complex,
    (translation (-3 + 2 * Complex.i) w = -7 - Complex.i) ∧
    (translation (-4 + 5 * Complex.i) w = -8 + 2 * Complex.i) :=
by
  let w := -4 - 3 * Complex.i
  use w
  split
  {
    -- Proof for first translation condition
    sorry
  }
  {
    -- Proof for second translation condition
    sorry
  }

end translation_of_complex_l373_373329


namespace general_formula_a_sum_T_n_l373_373039

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (n : ℕ)

-- Given conditions
axiom a_pos : ∀ n, a n > 0
axiom S_def : ∀ n, sqrt (S n) = (1 + a n) / 2

-- Proving a_n is given by the specific formula
theorem general_formula_a (n : ℕ) : a n = 2 * n - 1 :=
sorry

-- Proving T_n is given by the specific formula
noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (2 / (a i * (a (i + 1))))

theorem sum_T_n (n : ℕ) : T n = (2 * n) / (2 * n + 1) :=
sorry

end general_formula_a_sum_T_n_l373_373039


namespace common_point_circumcircles_l373_373152

variables {α : Type*} [euclidean_geometry α]

def triangle (A B C : α) : Prop := 
A ≠ B ∧ B ≠ C ∧ C ≠ A

def on_segment (P Q R : α) : Prop :=
collinear P Q R ∧ between P Q R

def circumcircle (Δ : Type*) [triangle Δ] : set α := 
{P : α | ∃ A B C : α, circumcircle_constraint}

theorem common_point_circumcircles
  {A B C D E F : α} (hABC : triangle A B C)
  (hD : on_segment D A B) (hE : on_segment E B C) (hF : on_segment F C A) :
  ∃ M : α, (M ∈ circumcircle (BDE)) ∧ (M ∈ circumcircle (CEF)) ∧ (M ∈ circumcircle (AFD)) :=
sorry

end common_point_circumcircles_l373_373152


namespace find_p_max_area_triangle_l373_373519

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373519


namespace sum_of_positive_integers_with_base4_reverse_base9_is_zero_l373_373022

theorem sum_of_positive_integers_with_base4_reverse_base9_is_zero :
  (∑ n in {n | ∃ a b, 0 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 8 ∧ 3 * a = 5 * b ∧ n = a * 4 + b ∧ n = b * 9 + a ∧ n > 0}, n) = 0 :=
by
  sorry

end sum_of_positive_integers_with_base4_reverse_base9_is_zero_l373_373022


namespace parabola_focus_distance_max_area_triangle_l373_373552

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373552


namespace parabola_condition_max_area_triangle_l373_373485

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373485


namespace parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373499

noncomputable def parabola (p : ℝ) : set (ℝ × ℝ) := {p | p.1^2 = 2 * p * p.2}
noncomputable def circle : set (ℝ × ℝ) := {q | q.1^2 + (q.2 + 4)^2 = 1}
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
noncomputable def dist (a b : ℝ × ℝ) : ℝ := 
    real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)

theorem parabola_point_distance_eq_four 
  (p : ℝ)
  (hp : p > 0) 
  (h : ∃ (q : ℝ × ℝ), q ∈ circle ∧ dist (focus p) q = 4 + 1) :
  p = 2 := 
sorry

theorem maximum_area_triangle_PAB
  (k b : ℝ)
  (hP : (2 * k, -b) ∈ circle)
  (p := 2) :
  4 * (k^2 + b)^(3/2) = 20 * real.sqrt 5 :=
sorry

end parabola_point_distance_eq_four_maximum_area_triangle_PAB_l373_373499


namespace coefficients_expression_l373_373987

theorem coefficients_expression :
  let a_0 := (2 + sqrt 3)^4,
      a_2 := sorry,  -- Placeholder for (1+ sqrt 3)^4
      a_4 := sorry,  -- Placeholder for (2 + sqrt 3), raised to an appropriate power
      a_1 := sorry,  -- Placeholder for appropriate calculation
      a_3 := sorry   -- Placeholder for appropriate calculation
  in (a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 2 := sorry

end coefficients_expression_l373_373987


namespace collinear_vectors_l373_373565

-- Definitions
def a : ℝ × ℝ := (2, 4)
def b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Proof statement
theorem collinear_vectors (x : ℝ) (h : ∃ k : ℝ, b x = k • a) : x = 3 :=
by sorry

end collinear_vectors_l373_373565


namespace boys_in_schoolA_boys_in_schoolB_boys_in_schoolC_l373_373605

noncomputable def schoolA_boys (x_A: ℝ) : Prop :=
  x_A + (x_A / 100) * 900 = 900

noncomputable def schoolB_boys (x_B: ℝ) : Prop :=
  x_B + (x_B / 100) * 1200 = 1200

noncomputable def schoolC_boys (x_C: ℝ) : Prop :=
  x_C + (x_C / 100) * 1500 = 1500

theorem boys_in_schoolA : ∃ x_A : ℝ, schoolA_boys x_A ∧ x_A = 90 :=
by {
  -- This is a placeholder for the actual proof
  sorry
}

theorem boys_in_schoolB : ∃ x_B : ℝ, schoolB_boys x_B ∧ x_B ≈ 92 :=
by {
  -- This is a placeholder for the actual proof
  sorry
}

theorem boys_in_schoolC : ∃ x_C : ℝ, schoolC_boys x_C ∧ x_C ≈ 94 :=
by {
  -- This is a placeholder for the actual proof
  sorry
}

end boys_in_schoolA_boys_in_schoolB_boys_in_schoolC_l373_373605


namespace number_of_arrangements_l373_373888

-- Define the students
inductive Student
| A | B | C | D | E

open Student

def is_not_at_end (l : List Student) : Prop :=
  match l with
  | [] | [_] => False
  | x::y::xs => (x ≠ A) ∧ (List.last? (y::xs) ≠ some A)

def are_adjacent_C_D (l : List Student) : Prop :=
  l == [C, D] ++ l.drop 2 ∨ l == D::C::l.drop 2

def count_valid_arrangements (students : List Student) : Nat :=
  -- Count arrangements where A is not at either end and C and D are adjacent
  sorry

theorem number_of_arrangements : count_valid_arrangements [A, B, C, D, E] = 24 :=
  sorry

end number_of_arrangements_l373_373888


namespace product_of_real_roots_eqn_l373_373373

/--
  Determine the product of all real roots of the equation \( x^{\ln x} = e \).

  Theorem: The product of all real roots of the equation \( x^{\ln x} = e \) is 1.
-/
theorem product_of_real_roots_eqn (x : ℝ) (h : x > 0 ∧ x^{Real.ln x} = Real.exp 1) :
  x = Real.exp 1 ∨ x = Real.exp (-1) ∧ (Real.exp 1 * Real.exp (-1) = 1) := 
begin
  sorry
end

end product_of_real_roots_eqn_l373_373373


namespace monotonic_range_of_b_l373_373445

noncomputable def f (b x : ℝ) : ℝ := x^3 - b * x^2 + 3 * x - 5

theorem monotonic_range_of_b (b : ℝ) : (∀ x y: ℝ, (f b x) ≤ (f b y) → x ≤ y) ↔ -3 ≤ b ∧ b ≤ 3 :=
sorry

end monotonic_range_of_b_l373_373445


namespace parabola_condition_max_area_triangle_l373_373490

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373490


namespace angle_equality_l373_373378

-- Definitions and conditions
variables (P A B C D Q : Point)
variables (circle : Circle)
variables (tangent1 : Tangent)
variables (tangent2 : Tangent)
variables (secant : Secant)
variables (chord : Chord)

-- Conditions
hypothesis h1 : P ⋖ circle
hypothesis h2 : tangent1 P A ∧ tangent1 P B
hypothesis h3 : secant P D ∧ secant C D
hypothesis h4 : C between P and D
hypothesis h5 : Q ∈ chord CD
hypothesis h6 : ∠DAQ = ∠PBC

-- Question to prove
theorem angle_equality : ∠DBQ = ∠PAC := 
by 
  sorry

end angle_equality_l373_373378


namespace simplify_expr1_simplify_expr2_l373_373680

theorem simplify_expr1 : 
  (1:ℝ) * (-3:ℝ) ^ 0 + (- (1/2:ℝ)) ^ (-2:ℝ) - (-3:ℝ) ^ (-1:ℝ) = 16 / 3 :=
by
  sorry

theorem simplify_expr2 (x : ℝ) : 
  ((-2 * x^3) ^ 2 * (-x^2)) / ((-x)^2) ^ 3 = -4 * x^2 :=
by
  sorry

end simplify_expr1_simplify_expr2_l373_373680


namespace sarah_loan_difference_l373_373201

noncomputable def compounded_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := 
  P + (P * r * t)

theorem sarah_loan_difference : 
  let P := 12000 in
  let r1 := 0.08 in
  let n := 2 in
  let t_half := 6 in
  let r2 := 0.09 in
  let payment1 := 3000 in
  let compounded_after_6 := compounded_amount P r1 n t_half in
  let remaining1 := compounded_after_6 - compounded_after_6 / 3 in
  let final_compounded := compounded_amount remaining1 r1 n t_half in
  let total_compounded := compounded_after_6 / 3 + final_compounded in
  let simple_after_6 := simple_interest P r2 t_half in
  let remaining2 := simple_after_6 - payment1 in
  let final_simple := simple_interest remaining2 r2 t_half in
  let total_simple := payment1 + final_simple in
  abs (total_compounded - total_simple) = 2427 :=
by
  sorry

end sarah_loan_difference_l373_373201


namespace purely_imaginary_real_parts_l373_373997

noncomputable def is_imaginary (z : ℂ) : Prop :=
∃ m : ℝ, z = m * complex.I

theorem purely_imaginary_real_parts (z : ℂ) (h : is_imaginary z) :
  (z + complex.conj z).im = 0 ∧ (z^2).im = 0 ∧ (complex.conj z * complex.I).im = 0 := sorry

end purely_imaginary_real_parts_l373_373997


namespace diagonal_sum_le_M_l373_373196

open Set

theorem diagonal_sum_le_M (M : ℝ) (a : Fin n → Fin n → ℝ) :
  (∀ x : Fin n → ℤ, (∀ i : Fin n, x i = 1 ∨ x i = -1) → 
    ∑ j, |∑ i, a j i * (x i : ℝ)| ≤ M) → 
  ∑ k, |a k k| ≤ M := by
    sorry

end diagonal_sum_le_M_l373_373196


namespace blue_balls_in_bag_l373_373784

variable (r : ℕ) -- Number of red balls
variable (b : ℕ) -- Number of blue balls
variable (total_balls : ℕ) -- Total number of balls in the bag

-- The conditions of the problem
def conditions : Prop :=
  total_balls = 12 ∧
  (r * (r - 1) / 132 = 1 / 22)

-- The statement to prove
theorem blue_balls_in_bag {r b : ℕ} (h : conditions r b 12) : b = 9 := by
  sorry

end blue_balls_in_bag_l373_373784


namespace find_diameter_of_well_l373_373018

noncomputable def well_volume (cost_per_cubic_meter : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / cost_per_cubic_meter

noncomputable def diameter_of_well (v : ℝ) (h : ℝ) : ℝ :=
  2 * real.sqrt (v / (real.pi * h))

theorem find_diameter_of_well: 
  ∀ (cost_per_cubic_meter total_cost depth : ℝ), 
  cost_per_cubic_meter = 16 → 
  total_cost = 1583.3626974092558 → 
  depth = 14 → 
  diameter_of_well (well_volume cost_per_cubic_meter total_cost) depth ≈ 2.995 :=
by
  intros cost_per_cubic_meter total_cost depth h1 h2 h3
  sorry

end find_diameter_of_well_l373_373018


namespace equivalent_fraction_denominator_l373_373302

theorem equivalent_fraction_denominator : 
  (∃ (d : ℕ), (1 : ℝ) / 3 = (12 : ℝ) / d) ↔ (d = 36) :=
by
  split
  { intro h
    cases h with d hd
    have h1 := (1 : ℝ) / 3 * 3 = (12 : ℝ) / d * 3,
    rw hd at h1
    simp at h1,
    use (12 * 3),
    norm_cast,
    assumption }
  { intro h,
    use 36,
    norm_cast,
    exact h }
  

end equivalent_fraction_denominator_l373_373302


namespace travel_times_either_24_or_72_l373_373003

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end travel_times_either_24_or_72_l373_373003


namespace combined_cost_l373_373837

theorem combined_cost (wallet_cost : ℕ) (purse_cost : ℕ)
    (h_wallet_cost : wallet_cost = 22)
    (h_purse_cost : purse_cost = 4 * wallet_cost - 3) :
    wallet_cost + purse_cost = 107 :=
by
  rw [h_wallet_cost, h_purse_cost]
  norm_num
  sorry

end combined_cost_l373_373837


namespace true_inverse_propositions_count_l373_373699

-- Let P1, P2, P3, P4 denote the original propositions
def P1 := "Supplementary angles are congruent, and two lines are parallel."
def P2 := "If |a| = |b|, then a = b."
def P3 := "Right angles are congruent."
def P4 := "Congruent angles are vertical angles."

-- Let IP1, IP2, IP3, IP4 denote the inverse propositions
def IP1 := "Two lines are parallel, and supplementary angles are congruent."
def IP2 := "If a = b, then |a| = |b|."
def IP3 := "Congruent angles are right angles."
def IP4 := "Vertical angles are congruent angles."

-- Counting the number of true inverse propositions
def countTrueInversePropositions : ℕ :=
  let p1_inverse_true := true  -- IP1 is true
  let p2_inverse_true := true  -- IP2 is true
  let p3_inverse_true := false -- IP3 is false
  let p4_inverse_true := true  -- IP4 is true
  [p1_inverse_true, p2_inverse_true, p4_inverse_true].length

-- The statement to be proved
theorem true_inverse_propositions_count : countTrueInversePropositions = 3 := by
  sorry

end true_inverse_propositions_count_l373_373699


namespace find_p_max_area_triangle_l373_373521

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373521


namespace total_amount_l373_373800

theorem total_amount (ratio_a ratio_b ratio_c : ℕ) (share_b : ℝ) (h1 : ratio_a = 2) (h2 : ratio_b = 3) (h3 : ratio_c = 4) (h4 : share_b = 1200) :
  let total_parts := ratio_a + ratio_b + ratio_c in
  let value_per_part := share_b / ratio_b in
  let total_amount := value_per_part * total_parts in
  total_amount = 3600 :=
by
  sorry

end total_amount_l373_373800


namespace remainder_2468135792_mod_101_l373_373745

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end remainder_2468135792_mod_101_l373_373745


namespace parabola_circle_distance_l373_373465

theorem parabola_circle_distance (p : ℝ) (hp : 0 < p) :
  let F := (0, p / 2)
  let M := EuclideanDistance
  let dist_F_M := ∀ (Q : ℝ × ℝ), Q ∈ {P : ℝ × ℝ | P.1^2 + (P.2 + 4)^2 = 1 } → ((0 - Q.1)^2 + (p / 2 - Q.2)^2).sqrt - 1 = 4 → p = 2
:= sorry

end parabola_circle_distance_l373_373465


namespace find_imaginary_part_l373_373936

noncomputable def i : ℂ := complex.I

theorem find_imaginary_part : complex.imag ((1 + i)^2 / (1 - i)) = 1 := 
by
  sorry

end find_imaginary_part_l373_373936


namespace inequality_part1_minimum_value_part2_l373_373927

open Real
open BigOperators

variable {x y z : ℝ}

-- Condition that x, y, z are positive and greater than 1 and that their sum equals 3√3:
variable (hx : x > 1) (hy : y > 1) (hz : z > 1) (hsum : x + y + z = 3 * sqrt 3)

-- Statement for part (1)
theorem inequality_part1 (hx : x > 1) (hy : y > 1) (hz : z > 1) (hsum : x + y + z = 3 * sqrt 3) :
  x^2 / (x + 2 * y + 3 * z) + y^2 / (y + 2 * z + 3 * x) + z^2 / (z + 2 * x + 3 * y) ≥ sqrt 3 / 2 :=
  sorry

-- Statement for part (2)
theorem minimum_value_part2 (hx : x > 1) (hy : y > 1) (hz : z > 1) (hsum : x + y + z = 3 * sqrt 3) :
  (1 / (log 3 x + log 3 y)) + (1 / (log 3 y + log 3 z)) + (1 / (log 3 z + log 3 x)) = 3 :=
  sorry

end inequality_part1_minimum_value_part2_l373_373927


namespace angle_relation_l373_373612

/-
Given an isosceles triangle ABC with AB = AC and an inscribed equilateral
triangle PQR, if ∠BQP = u, ∠ARP = v, and ∠CQR = w, prove that:
u = (v + w) / 2
-/

variables (A B C P Q R : Type) [IsoscelesTriangle A B C] [EquilateralTriangle P Q R]
variables (u v w : ℝ)
axiom isosceles_tria : AB = AC
axiom equilateral_tria : ∀ (x y z : Type), ∠PQR = 60 ∧ ∠QRP = 60 ∧ ∠RPQ = 60
axiom angles_def : (BQP = u) ∧ (ARP = v) ∧ (CQR = w)

theorem angle_relation : u = (v + w) / 2 :=
by
  sorry

end angle_relation_l373_373612


namespace parallel_vectors_x_value_l373_373085

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (6, x)

-- Define what it means for vectors to be parallel (they are proportional)
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem to prove
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel a (b x) → x = 9 :=
by
  intros x h
  sorry

end parallel_vectors_x_value_l373_373085


namespace pascal_triangle_row_sum_pascal_triangle_row_10_sum_l373_373597

theorem pascal_triangle_row_sum (n : ℕ) : (∑ k in Finset.range (n + 1), Nat.choose n k) = 2^n :=
by sorry

theorem pascal_triangle_row_10_sum : (∑ k in Finset.range 11, Nat.choose 10 k) = 1024 :=
by sorry

end pascal_triangle_row_sum_pascal_triangle_row_10_sum_l373_373597


namespace distinct_d_not_possible_l373_373848

theorem distinct_d_not_possible :
  ∀ (a : Fin 2010 → ℕ), 
    (Function.Injective a) → 
    (∀ i, 1 ≤ a i ∧ a i ≤ 2010) → 
    ¬ (Function.Injective (λ (i : Fin 2010), abs ((a i) - (i + 1)))) :=
by 
  -- Assume necessary types and imports
  intros a a_inj a_range,
  sorry

end distinct_d_not_possible_l373_373848


namespace remainder_2468135792_div_101_l373_373737

theorem remainder_2468135792_div_101 : (2468135792 % 101) = 52 := 
by 
  -- Conditions provided in the problem
  have decompose_num : 2468135792 = 24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92, 
  from sorry,
  
  -- Assert large powers of 10 modulo properties
  have ten_to_pow2 : (10^2 - 1) % 101 = 0, from sorry,
  have ten_to_pow4 : (10^4 - 1) % 101 = 0, from sorry,
  have ten_to_pow6 : (10^6 - 1) % 101 = 0, from sorry,
  have ten_to_pow8 : (10^8 - 1) % 101 = 0, from sorry,
  
  -- Summing coefficients
  have coefficients_sum : 24 + 68 + 13 + 57 + 92 = 254, from
  by linarith,
  
  -- Calculating modulus
  calc 
    2468135792 % 101
        = (24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92) % 101 : by rw decompose_num
    ... = (24 + 68 + 13 + 57 + 92) % 101 : by sorry
    ... = 254 % 101 : by rw coefficients_sum
    ... = 52 : by norm_num,

  sorry

end remainder_2468135792_div_101_l373_373737


namespace max_a_b_c_sum_91_l373_373635

theorem max_a_b_c_sum_91 (a b c : ℕ) (h1 : a ≠ b ∧ a ≠ c ∧ b ≠ c) 
  (h2 : ∃ k1 k2 k3 : ℕ, b + c - a = k1^2 ∧ c + a - b = k2^2 ∧ a + b - c = k3^2) 
  (h3 : a + b + c < 100) : 
  a + b + c ≤ 91 ∧ (∃ a' b' c' : ℕ, a' ≠ b' ∧ a' ≠ c' ∧ b' ≠ c' ∧ 
  (∃ k1 k2 k3 : ℕ, b' + c' - a' = k1^2 ∧ c' + a' - b' = k2^2 ∧ a' + b' - c' = k3^2) ∧ 
  a' + b' + c' = 91) :=
begin
  sorry
end

end max_a_b_c_sum_91_l373_373635


namespace proof_part1_proof_part2_l373_373478

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373478


namespace train_speed_kph_l373_373809

-- Define conditions as inputs
def train_time_to_cross_pole : ℝ := 6 -- seconds
def train_length : ℝ := 100 -- meters

-- Conversion factor from meters per second to kilometers per hour
def mps_to_kph : ℝ := 3.6

-- Define and state the theorem to be proved
theorem train_speed_kph : (train_length / train_time_to_cross_pole) * mps_to_kph = 50 :=
by
  sorry

end train_speed_kph_l373_373809


namespace greatest_integer_b_l373_373019

def quadratic_no_real_roots (b : ℤ) : Prop :=
  b^2 < 68

theorem greatest_integer_b :
  ∃ b : ℤ, quadratic_no_real_roots b ∧ ∀ k : ℤ, quadratic_no_real_roots k → k ≤ b :=
begin
  use 8,
  split,
  { show (8 : ℤ)^2 < 68,
    norm_num,
    exact lt_of_le_of_lt (show 64 ≤ 68, by norm_num) (by norm_num) },
  { intros k hk,
    show k ≤ 8,
    have : (k : ℤ)^2 < 68 := hk,
    calc k ≤ 8 : by { sorry }
  }
end

end greatest_integer_b_l373_373019


namespace game_final_score_l373_373365

noncomputable def final_score (n : ℕ) : ℕ :=
  if n = 1 then 0 else (n * (n - 1)) / 2

theorem game_final_score (M m : ℕ) (hM : M = final_score 20) (hm : m = final_score 20) :
  M - m = 0 := by
  sorry

end game_final_score_l373_373365


namespace max_intersections_of_fifth_degree_polynomials_l373_373253

theorem max_intersections_of_fifth_degree_polynomials 
  (a b : ℝ → ℝ)
  (ha : ∀ x, a x = x^5 + x^4 + x^3 + x^2 + x + 1)
  (hb : ∀ x, b x = x^5 + x^4 - x^3 + x^2 + x + 1) :
  ∃! x : ℝ, a x = b x :=
by
sory

end max_intersections_of_fifth_degree_polynomials_l373_373253


namespace inscribed_circle_radius_eq_l373_373974

noncomputable def inscribed_circle_radius (r s t : ℝ) : ℝ :=
  Real.sqrt (r * s * t / (r + s + t))

theorem inscribed_circle_radius_eq (r s t : ℝ) :
  r > 0 ∧ s > 0 ∧ t > 0 →
  (inscribed_circle_radius r s t = Real.sqrt (r * s * t / (r + s + t))) :=
begin
  assume h : r > 0 ∧ s > 0 ∧ t > 0,
  sorry
end

end inscribed_circle_radius_eq_l373_373974


namespace lambda_range_l373_373047

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3^n

noncomputable def b (n : ℕ) : ℕ :=
  n - 3

theorem lambda_range (λ : ℝ) : (∀ n : ℕ, n > 0 → (2 * λ - 1) * a n > 36 * b n) → λ > 13 / 18 :=
by
  intros h
  sorry

end lambda_range_l373_373047


namespace minimize_abs_diff_l373_373577

theorem minimize_abs_diff (x y : ℤ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h : x * y - 4 * x + 3 * y = 204) : ∃ x y : ℤ, 0 < x ∧ 0 < y ∧ |x - y| = 11 :=
begin
  sorry
end

end minimize_abs_diff_l373_373577


namespace fixed_point_pass_through_l373_373607

open EuclideanGeometry

noncomputable def circumscribed_circle (A B C : Point) : Circle := sorry

noncomputable def projection (P : Point) (L : Line) : Point := sorry

noncomputable def orthocenter (A B C : Point) : Point := sorry

theorem fixed_point_pass_through (A B C : Point) 
    (acute: is_acute (Triangle A B C)) :
    let H := orthocenter A B C,
    ∀ (X : Point), X ∈ minor_arc (circumscribed_circle A B C) B C →
    let P := projection X (line C A),
    let Q := projection X (line B C),
    let R := line_inter (line_through B (perpendicular_line (line A C))) (line P Q),
    let l := parallel_line_through (line X R) P
    in passes_through l H :=
begin
    intros,
    sorry
end

end fixed_point_pass_through_l373_373607


namespace find_p_max_area_triangle_l373_373526

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373526


namespace max_area_triangle_PAB_l373_373551

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373551


namespace speed_of_train_in_kmh_l373_373805

-- Define the conditions
def time_to_cross_pole : ℝ := 6
def length_of_train : ℝ := 100
def conversion_factor : ℝ := 18 / 5

-- Using the conditions to assert the speed of the train
theorem speed_of_train_in_kmh (t : ℝ) (d : ℝ) (conv_factor : ℝ) : 
  t = time_to_cross_pole → 
  d = length_of_train → 
  conv_factor = conversion_factor → 
  (d / t) * conv_factor = 50 := 
by 
  intros h_t h_d h_conv_factor
  sorry

end speed_of_train_in_kmh_l373_373805


namespace circumsphere_radius_l373_373684

theorem circumsphere_radius 
  (A B C D : Point) 
  (volume_ABCD : Volume A B C D = 1)
  (AB_eq : dist A B = 2)
  (AC_eq : dist A C = 2)
  (AD_eq : dist A D = 2)
  (BC_CD_DB_eq : dist B C * dist C D * dist D B = 16) :
  circumsphere_radius A B C D = 5 / 3 := 
sorry

end circumsphere_radius_l373_373684


namespace min_colors_to_distinguish_keys_l373_373334

def min_colors_needed (n : Nat) : Nat :=
  if n <= 2 then n
  else if n >= 6 then 2
  else 3

theorem min_colors_to_distinguish_keys (n : Nat) :
  (n ≤ 2 → min_colors_needed n = n) ∧
  (3 ≤ n ∧ n ≤ 5 → min_colors_needed n = 3) ∧
  (n ≥ 6 → min_colors_needed n = 2) :=
by
  sorry

end min_colors_to_distinguish_keys_l373_373334


namespace sum_of_coefficients_l373_373354

theorem sum_of_coefficients :
  let p := 3 * (λ x : ℝ, x^6 - x^3 + 4 * x^2 - 7) - 5 * (λ x : ℝ, x^4 + 3 * x) + 2 * (λ x : ℝ, x^7 - 6)
  (p 1) = -39 :=
by
  let p := 3 * (λ x : ℝ, x^6 - x^3 + 4 * x^2 - 7) - 5 * (λ x : ℝ, x^4 + 3 * x) + 2 * (λ x : ℝ, x^7 - 6)
  have sum := p 1
  sorry

end sum_of_coefficients_l373_373354


namespace problem_conditions_l373_373069

def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem problem_conditions (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) : 
    (f (-x) = f x) ∧ 
    (x ≠ 0 → f (1/x) = -f x) ∧ 
    (f 0 + (Finset.sum (Finset.range (19)) (λ n, f (1/(n + 2)) + f (n + 2))) ≠ 0 ) ∧
    (∀ y, y = f x → (y < -1 ∨ y ≥ 1)) :=
by
    sorry

end problem_conditions_l373_373069


namespace vector_eq_l373_373569

variables (a b c : ℝ × ℝ) (m λ : ℝ)

-- Given definitions
def a := (2, 1)
def b := (3, 4)
def c := (1, m)

-- Problem statement
theorem vector_eq : a + b = λ • c → λ + m = 6 :=
by 
  sorry

end vector_eq_l373_373569


namespace prove_sum_of_roots_l373_373706

lemma sum_of_roots (M : ℝ) (h : M * (M - 8) = 7): M = 4 + real.sqrt(23) ∨ M = 4 - real.sqrt(23) :=
by 
  let q := (polynomial.X^2 - 8 * polynomial.X - 7 : polynomial ℝ)
  have key : q.eval M = 0 := by simp only [polynomial.eval_X, polynomial.eval_C, polynomial.eval_Mul]; exact h
  have sum : polynomial.root_sum q = 8 := by simp
  sorry

theorem prove_sum_of_roots : 
  ∃ M : ℝ, M * (M - 8) = 7 ∧ (sum_of_roots M (sorry) = 8) :=
by
  have M := sum_of_roots M sorry
  use M
  exact ⟨M, by assumption⟩
    sorry

end prove_sum_of_roots_l373_373706


namespace find_p_max_area_triangle_l373_373525

-- Define the given conditions
def parabola (p : ℝ) := λ (x y : ℝ), x^2 = 2 * p * y
def circle (M : ℝ × ℝ) := λ (x y : ℝ), x^2 + (y + 4)^2 = 1

-- Focus and minimum distance condition
def focus (p : ℝ) := (0, p / 2)
def min_distance (F : ℝ × ℝ) (M : ℝ × ℝ) := dist F M = 4

-- The two main goals to prove
theorem find_p (p : ℝ) (x y : ℝ) :
  parabola p x y → circle (x, y) x y → min_distance (focus p) (x, y) → p = 2 :=
by
  sorry

theorem max_area_triangle (P A B : ℝ × ℝ) (k b : ℝ) :
  parabola 2 (A.1) (A.2) → parabola 2 (B.1) (B.2) →
  circle P.1 P.2 → P = (2 * k, -b) →
  (∃ y_P, y_P ∈ [-5, -3] ∧
    max_area k b = 20 * real.sqrt 5) :=
by
  sorry

end find_p_max_area_triangle_l373_373525


namespace find_y_l373_373609

/-- 
  Given: The sum of angles around a point is 360 degrees, 
  and those angles are: 6y, 3y, 4y, and 2y.
  Prove: y = 24 
-/ 
theorem find_y (y : ℕ) (h : 6 * y + 3 * y + 4 * y + 2 * y = 360) : y = 24 :=
sorry

end find_y_l373_373609


namespace alice_weight_l373_373688

theorem alice_weight (a c : ℝ) (h1 : a + c = 200) (h2 : a - c = a / 3) : a = 120 :=
by
  sorry

end alice_weight_l373_373688


namespace correct_answer_l373_373637

def sum_squares_of_three_consecutive_even_integers (n : ℤ) : ℤ :=
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  a * a + b * b + c * c

def T : Set ℤ :=
  {t | ∃ n : ℤ, t = sum_squares_of_three_consecutive_even_integers n}

theorem correct_answer : (∀ t ∈ T, t % 4 = 0) ∧ (∀ t ∈ T, t % 7 ≠ 0) :=
sorry

end correct_answer_l373_373637


namespace Sandy_age_l373_373673

variable (S M : ℕ)

def condition1 (S M : ℕ) : Prop := M = S + 18
def condition2 (S M : ℕ) : Prop := S * 9 = M * 7

theorem Sandy_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 63 := sorry

end Sandy_age_l373_373673


namespace find_p_max_area_of_triangle_l373_373505

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373505


namespace three_digit_perfect_cubes_divisible_by_27_l373_373100

theorem three_digit_perfect_cubes_divisible_by_27 : 
    ∃ n, (100 ≤ 27 * n^3) ∧ (27 * n^3 ≤ 999) ∧ (27 * n^3 = 216 ∨ 27 * n^3 = 729) :=
begin
  sorry
end

end three_digit_perfect_cubes_divisible_by_27_l373_373100


namespace greatest_vertex_product_sum_l373_373336

theorem greatest_vertex_product_sum :
  ∀ (x y z x' y' z' : ℕ),
    {0, 1, 2, 3, 4, 5} = {x, y, z, x', y', z'} →
    x + x' = 5 →
    y + y' = 5 →
    z + z' = 5 →
    (x * y * z) + (x * y * z') + (x * y' * z) + (x * y' * z') +
    (x' * y * z) + (x' * y * z') + (x' * y' * z) + (x' * y' * z') = 125 :=
begin
  -- Proof goes here, but will be skipped with:
  sorry
end

end greatest_vertex_product_sum_l373_373336


namespace grain_store_loses_l373_373308

theorem grain_store_loses 
  (a b : ℝ) (ha : a ≠ b) :
  let m₁ := 5 * b / a in
  let m₂ := 5 * a / b in
  m₁ + m₂ > 10 :=
by 
  sorry

end grain_store_loses_l373_373308


namespace sqrt_subtraction_l373_373262

theorem sqrt_subtraction : 
  sqrt (49 + 121) - sqrt (36 - 9) = sqrt 170 - sqrt 27 :=
by
  sorry

end sqrt_subtraction_l373_373262


namespace sqrt_inequality_l373_373051

theorem sqrt_inequality (a b : ℝ) (h_a_pos : 0 < a) (h_cond : 1 / b - 1 / a > 1) : 
  sqrt (1 + a) > 1 / sqrt (1 - b) :=
by {
  sorry
}

end sqrt_inequality_l373_373051


namespace milk_total_correct_l373_373899

def chocolate_milk : Nat := 2
def strawberry_milk : Nat := 15
def regular_milk : Nat := 3
def total_milk : Nat := chocolate_milk + strawberry_milk + regular_milk

theorem milk_total_correct : total_milk = 20 := by
  sorry

end milk_total_correct_l373_373899


namespace car_race_course_distance_l373_373787

theorem car_race_course_distance 
  (v : ℝ) 
  (d : ℝ) 
  (h1 : 0 < v)
  (h2 : 0 < d)
  (h3 : d/2 > 26)
  (h4 : d/2 > 4)
  (initial_speed_ratio : (5 / 6) * v)
  (speed_decrease_uphill : (3 / 4) * v)
  (speed_increase_downhill : (5 / 4) * v)
  (time_equal : 
    (30 / v + (d - 30) / (3 / 4 * v) + (d - 30) / (5 / 4 * v) =
     36 / (5 / 6 * v) + (d - 30) / (5 / 6 * 5 / 4 * v) + (d - 30) / (5 / 6 * 3 / 4 * v))) :
  d = 92 := 
sorry

end car_race_course_distance_l373_373787


namespace num_broadcasting_methods_l373_373783

theorem num_broadcasting_methods : 
  let n := 6
  let commercials := 4
  let public_services := 2
  (public_services * commercials!) = 48 :=
by
  let n := 6
  let commercials := 4
  let public_services := 2
  have total_methods : (public_services * commercials!) = 48 := sorry
  exact total_methods

end num_broadcasting_methods_l373_373783


namespace max_area_triangle_PAB_l373_373542

-- Define given parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2 * p * y

-- Define the circle M
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1

-- Define the focus F of the parabola
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

-- Define the minimum distance from F to any point on the circle
def min_distance_from_focus_to_circle (p : ℝ) : Prop := 
  abs ((p / 2) + 4 - 1) = 4

-- Define point P on the circle
def point_on_circle (x y : ℝ) : Prop := circle x y

-- Define tangents PA and PB from point P to the parabola with points of tangency A and B
def tangents (x1 y1 x2 y2 p : ℝ) : Prop :=
  parabola p x1 y1 ∧ parabola p x2 y2

-- Define area of triangle PAB
def area_triangle_PAB (x1 y1 x2 y2 : ℝ) : ℝ := 
  let k := (x1 + x2) / 2
  let b := -y1
  4 * ((k^2 + b) ^ (3 / 2))

-- Prove that for p = 2, the maximum area of the triangle PAB is 20 sqrt 5
theorem max_area_triangle_PAB : ∃ (x1 y1 x2 y2 : ℝ), 
  parabola 2 x1 y1 ∧ parabola 2 x2 y2 ∧ point_on_circle 2 (-(x2 / 2)^2 / 4) ∧
  area_triangle_PAB x1 y1 x2 y2 = 20 * real.sqrt 5 :=
sorry

end max_area_triangle_PAB_l373_373542


namespace find_n_l373_373636

theorem find_n (S : ℕ → ℚ) (hS : ∀ n, S n = ∑ i in finset.range (n+1), 1 / ((i+1:ℚ) * (i+2:ℚ))) 
(hS_mul : ∀ n, S n * S (n+1) = 3 / 4) : ∃ n, n = 6 :=
begin
  use 6,
  sorry
end

end find_n_l373_373636


namespace divisibility_by_3_l373_373618

theorem divisibility_by_3 (x y z : ℤ) (h : x^3 + y^3 = z^3) : 3 ∣ x ∨ 3 ∣ y ∨ 3 ∣ z := 
sorry

end divisibility_by_3_l373_373618


namespace propositions_correct_l373_373461

theorem propositions_correct
  (f : ℝ → ℝ)
  (h1 : ∀ x, f(1 + 2 * x) = f(1 - 2 * x))
  (h2 : ∀ x, f(x - 1) = f(1 - x))
  (h3a : ∀ x, f(x) = f(-x))
  (h3b : ∀ x, f(1 + x) = -f(x))
  (h4a : ∀ x, f(-x) = -f(x))
  (h4b : ∀ x, f(x) = f(-x - 2)) :
  (∀ x, f(1 + 2 * x) = f(1 - 2 * x)) ∧
  (∀ x, f(x - 1) = f(1 - x)) ∧
  (∀ x, f(x) = f(-x) → f(1 + x) = -f(x) → ∀ x, f(x) = f(2 - x)) ∧
  (∀ x, f(x) = f(-x - 2) → ∀ x, f(x) = -f(x) → ∀ x, f(x + 2) = f(-x)) :=
by sorry

end propositions_correct_l373_373461


namespace real_condition_of_complex_l373_373584

theorem real_condition_of_complex (m : ℝ) (H : (∃ (re : ℝ), (m^2 + complex.i : ℂ) / (1 - m * complex.i : ℂ) = re)) : m = -1 :=
sorry

end real_condition_of_complex_l373_373584


namespace square_side_length_l373_373283

theorem square_side_length
  (r : ℝ)
  (h1 : r = 2 + real.sqrt (5 - real.sqrt 5))
  (h2 : ∀ θ : ℝ, θ = 36 → real.sin θ = real.sqrt (5 - real.sqrt 5) / (2 * real.sqrt 2))
  (angle_tangents : θ = 72)
  : ∃ s : ℝ, s = (real.sqrt (real.sqrt 5 - 1) * real.sqrt (real.sqrt 125)) / 5 :=
by
  sorry

end square_side_length_l373_373283


namespace angle_relation_l373_373213

open_locale real

variables {A B C D E : Type}

noncomputable def altitude (A : Type) (AD : Type) : Type := sorry

noncomputable def external_angle_bisector (BAC : Type) (BC : Type) (E : Type) : Type := sorry

noncomputable def twice_length (x : Type) : Type := sorry

theorem angle_relation (A B C D E : Type) 
  (triangle_ABC : triangle A B C)
  (AD_altitude : altitude A D)
  (AE_ext_bisector : external_angle_bisector A B A E)
  (twice_AE_AD : twice_length D = E) :
  ∃ B C : Type, (B = C) ∨ (B = C + 60) ∨ (C = B + 60) :=
sorry

end angle_relation_l373_373213


namespace simplify_expression_l373_373834

variable {a : ℝ} (h1 : a ≠ -3) (h2 : a ≠ 3) (h3 : a ≠ 2) (h4 : 2 * a + 6 ≠ 0)

theorem simplify_expression : (1 / (a + 3) + 1 / (a ^ 2 - 9)) / ((a - 2) / (2 * a + 6)) = 2 / (a - 3) :=
by
  sorry

end simplify_expression_l373_373834


namespace min_ab_value_l373_373416

noncomputable def prove_min_ab : Prop :=
  ∀ (a b : ℝ), 
    a > 1 → b > 1 → ab + 2 = 2 * (a + b) → 
    ab ≥ 6 + 4 * Real.sqrt 2

theorem min_ab_value : prove_min_ab :=
by
  sorry

end min_ab_value_l373_373416


namespace median_salary_company_l373_373332

noncomputable def median_salary (salaries : List ℕ) : ℕ :=
  let sorted := salaries.sort
  sorted.get (sorted.length / 2) -- Using zero-based index

def company_salaries : List (ℕ × ℕ) :=
  [(1, 130000), (15, 90000), (10, 80000), (8, 50000), (37, 25000)]

def expanded_salaries : List ℕ :=
  company_salaries.bind (λ ⟨num, sal⟩, List.replicate num sal)

theorem median_salary_company :
  median_salary expanded_salaries = 25000 :=
by
  have company's_salaries_construction : expanded_salaries =
    List.concat (List.replicate 1 130000) (List.replicate 15 90000)
    (List.replicate 10 80000) (List.replicate 8 50000) (List.replicate 37 25000)
    := sorry
  rw [company's_salaries_construction, median_salary]
  sorry

end median_salary_company_l373_373332


namespace f_is_odd_l373_373704

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 2 * x

-- State the problem
theorem f_is_odd :
  ∀ x : ℝ, f (-x) = -f x := 
by
  sorry

end f_is_odd_l373_373704


namespace part1_part2_l373_373973

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)
variable (m : ℝ) [decidable_pred (λ x, x ∈ U)] [decidable_pred (λ x, x ∈ A)] [decidable_pred (λ x, x ∈ B)]

-- Condition definitions
def U := {x : ℝ | true}
def A := {x : ℝ | 1 < x ∧ x < 4}
def B (m : ℝ) := {x : ℝ | x ≤ 3 * m - 4 ∨ x ≥ 8 + m}

-- Question 1
theorem part1 (H : m = 2) : A ∩ ∁(B 2) = {x : ℝ | 2 < x ∧ x < 4} :=
by 
  sorry

-- Question 2
theorem part2 (H : A ∩ ∁(B m) = ∅) (H' : m < 6) : -4 ≤ m ∧ m ≤ 5 / 3 :=
by 
  sorry

end part1_part2_l373_373973


namespace calculate_final_number_l373_373732

theorem calculate_final_number (initial increment times : ℕ) (h₀ : initial = 540) (h₁ : increment = 10) (h₂ : times = 6) : initial + increment * times = 600 :=
by
  sorry

end calculate_final_number_l373_373732


namespace remaining_stock_correct_l373_373630

variable (initial_stock : ℕ)
variable (sold_percent_monday sold_percent_tuesday sold_percent_wednesday sold_percent_thursday sold_percent_friday : ℝ)

def remaining_stock_after_week (initial_stock : ℕ) 
  (sold_percent_monday sold_percent_tuesday sold_percent_wednesday sold_percent_thursday sold_percent_friday : ℝ) : ℝ :=
let stock_after_monday := initial_stock - (initial_stock * sold_percent_monday)
let stock_after_tuesday := stock_after_monday - (stock_after_monday * sold_percent_tuesday)
let stock_after_wednesday := stock_after_tuesday - (stock_after_tuesday * sold_percent_wednesday)
let stock_after_thursday := stock_after_wednesday - (stock_after_wednesday * sold_percent_thursday)
let stock_after_friday := stock_after_thursday - (stock_after_thursday * sold_percent_friday)
in (stock_after_friday / initial_stock) * 100

theorem remaining_stock_correct : 
  initial_stock = 1000 ∧ 
  sold_percent_monday = 0.05 ∧ 
  sold_percent_tuesday = 0.10 ∧ 
  sold_percent_wednesday = 0.15 ∧ 
  sold_percent_thursday = 0.20 ∧ 
  sold_percent_friday = 0.25 →
  remaining_stock_after_week initial_stock 
    sold_percent_monday 
    sold_percent_tuesday 
    sold_percent_wednesday 
    sold_percent_thursday 
    sold_percent_friday = 43.7 := 
sorry

end remaining_stock_correct_l373_373630


namespace teamA_teamB_repair_eq_l373_373282

-- conditions
def teamADailyRepair (x : ℕ) := x -- represent Team A repairing x km/day
def teamBDailyRepair (x : ℕ) := x + 3 -- represent Team B repairing x + 3 km/day
def timeTaken (distance rate: ℕ) := distance / rate -- time = distance / rate

-- Proof problem statement
theorem teamA_teamB_repair_eq (x : ℕ) (hx : x > 0) (hx_plus_3 : x + 3 > 0) :
  timeTaken 6 (teamADailyRepair x) = timeTaken 8 (teamBDailyRepair x) → (6 / x = 8 / (x + 3)) :=
by
  intros h
  sorry

end teamA_teamB_repair_eq_l373_373282


namespace ratio_girls_boys_l373_373595

theorem ratio_girls_boys (g b : ℕ) (h1 : b = g - 4) (h2 : g + b = 30) : g / b = 17 / 13 :=
by
  have h3 : g = 17 := sorry
  have h4 : b = 13 := sorry
  rw [h3, h4]
  exact rfl

end ratio_girls_boys_l373_373595


namespace least_positive_integer_l373_373252

theorem least_positive_integer :
  ∃ N : ℕ, 
    (N % 7 = 5) ∧ 
    (N % 8 = 6) ∧ 
    (N % 9 = 7) ∧ 
    (N % 10 = 8) ∧
    N = 2518 :=
begin
  use 2518,
  split, { exact Nat.mod_eq_of_lt dec_trivial (by norm_num) },
  split, { exact Nat.mod_eq_of_lt dec_trivial (by norm_num) },
  split, { exact Nat.mod_eq_of_lt dec_trivial (by norm_num) },
  split, { exact Nat.mod_eq_of_lt dec_trivial (by norm_num) },
  refl
end

end least_positive_integer_l373_373252


namespace player_A_not_losing_l373_373130

theorem player_A_not_losing (P_A_wins P_draw : ℝ) (h1 : P_A_wins = 0.4) (h2 : P_draw = 0.2) :
  P_A_wins + P_draw = 0.6 :=
by
  rw [h1, h2]
  rfl

end player_A_not_losing_l373_373130


namespace find_p_max_area_of_triangle_l373_373509

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373509


namespace unique_x_intersect_l373_373907

theorem unique_x_intersect (m : ℝ) (h : ∀ x : ℝ, (m - 4) * x^2 - 2 * m * x - m - 6 = 0 → ∀ y : ℝ, (m - 4) * y^2 - 2 * m * y - m - 6 = 0 → x = y) :
  m = -4 ∨ m = 3 ∨ m = 4 :=
sorry

end unique_x_intersect_l373_373907


namespace James_tins_collected_l373_373621

theorem James_tins_collected :
  let first_day := 50
  let second_day := 3 * first_day
  let third_day := second_day - 50
  let remaining_days := 4 * first_day
  in first_day + second_day + third_day + remaining_days = 500 :=
by
  let first_day := 50
  let second_day := 3 * first_day
  let third_day := second_day - 50
  let remaining_days := 4 * first_day
  have : first_day + second_day + third_day + remaining_days = 500 := sorry
  exact this

end James_tins_collected_l373_373621


namespace range_of_M_l373_373905

theorem range_of_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a + b + c = 1) :
    ( (1 / a - 1) * (1 / b - 1) * (1 / c - 1) )  ≥ 8 := 
  sorry

end range_of_M_l373_373905


namespace hyperbola_eccentricity_l373_373694

-- Define the conditions given in the problem
def asymptote_equation_related (a b : ℝ) : Prop := a / b = 3 / 4
def hyperbola_eccentricity_relation (a c : ℝ) : Prop := c^2 / a^2 = 25 / 9

-- Define the proof problem
theorem hyperbola_eccentricity (a b c e : ℝ)
  (h1 : asymptote_equation_related a b)
  (h2 : hyperbola_eccentricity_relation a c)
  (he : e = c / a) :
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l373_373694


namespace min_value_frac_l373_373958

noncomputable def f (x : ℝ) : ℝ := real.log (real.sqrt (x^2 + 1) + x)

theorem min_value_frac {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : f (2 * a - 2) + f b = 0) :
  ∃ (m : ℝ), m = 4 ∧ (∀ a b, 0 < a → 0 < b → f (2 * a - 2) + f b = 0 → frac_value a b = m) :=
begin
  let frac_value := λ a b, ((2 * a + b) / (a * b)),
  have pos_a : 0 < a := ha,
  have pos_b : 0 < b := hb,
  have cond_f : f (2 * a - 2) + f b = 0 := h,
  let m := 4,
  use m,
  split,
  { refl },
  { intros a b ha hb h,
    sorry }
end

end min_value_frac_l373_373958


namespace regular_hours_l373_373825

variable (R : ℕ)

theorem regular_hours (h1 : 5 * R + 6 * (44 - R) + 5 * R + 6 * (48 - R) = 472) : R = 40 :=
by
  sorry

end regular_hours_l373_373825


namespace smallest_m_for_probability_l373_373670

-- Define the conditions in Lean
def nonWithInTwoUnits (x y z : ℝ) : Prop :=
  abs (x - y) ≥ 2 ∧ abs (y - z) ≥ 2 ∧ abs (z - x) ≥ 2

def probabilityCondition (m : ℝ) : Prop :=
  (m - 4)^3 / m^3 > 2/3

-- The theorem statement
theorem smallest_m_for_probability : ∃ m : ℕ, 0 < m ∧ (∀ x y z : ℝ, 0 ≤ x ∧ x ≤ m ∧ 0 ≤ y ∧ y ≤ m ∧ 0 ≤ z ∧ z ≤ m → nonWithInTwoUnits x y z) → probabilityCondition m ∧ m = 14 :=
by sorry

end smallest_m_for_probability_l373_373670


namespace num_roots_of_equation_l373_373407

noncomputable def equation_roots : ℝ → ℝ :=
  λ x, arctan (tan (sqrt (13 * π^2 + 12 * π * x - 12 * x^2))) -
       arcsin (sin (sqrt ((13 * π^2) / 4 + 3 * π * x - 3 * x^2)))

theorem num_roots_of_equation : ∃ n, n = 9 :=
by
  use 9
  sorry

end num_roots_of_equation_l373_373407


namespace giselle_paint_l373_373030

theorem giselle_paint (x : ℚ) (h1 : 5/7 = x/21) : x = 15 :=
by
  sorry

end giselle_paint_l373_373030


namespace series_bounds_l373_373938

def floor (x : ℝ) : ℕ := int.to_nat (Real.floor x)

def sumSeries (n : ℕ) : ℝ :=
  ∑ k in (Finset.range (n + 1)).filter (λ k, k ≥ 4), (-1) ^ (floor (Real.sqrt k)) / k

theorem series_bounds (n : ℕ) (h : 4 ≤ n) : 
  0 < sumSeries n ∧ sumSeries n < 1 :=
sorry

end series_bounds_l373_373938


namespace angle_between_is_60_degrees_l373_373155

open Real

variables {V : Type*} [inner_product_space ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

def is_linearly_independent (a b c : V) : Prop :=
  linear_independent ℝ ![a, b, c]

theorem angle_between_is_60_degrees
  (a b c : V)
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hc : is_unit_vector c) 
  (h_triple_prod : a × (b × c) = (2 • b - c) / 2)
  (h_linear_indep : is_linearly_independent a b c) :
  real.angle.of_real (inner_product_space.angle a c) = 60 :=
sorry

end angle_between_is_60_degrees_l373_373155


namespace isosceles_trapezoid_axes_square_axes_angle_symmetry_l373_373342

variables (IsoscelesTrapezoid : Type) (Square : Type) (Angle : Type)

-- Definition and axioms based on geometric conditions
def has_axes_of_symmetry (shape : Type) (n : ℕ) : Prop := sorry

axiom isosceles_trapezoid_has_one_axis_symmetry : has_axes_of_symmetry IsoscelesTrapezoid 1
axiom square_has_four_axes_symmetry : has_axes_of_symmetry Square 4
axiom angle_bisector_is_axis_symmetry : ∀ (a : Angle), is_axis_of_symmetry (bisector a)

-- Statement of the problem in Lean 4
theorem isosceles_trapezoid_axes : has_axes_of_symmetry IsoscelesTrapezoid 1 :=
  isosceles_trapezoid_has_one_axis_symmetry

theorem square_axes : has_axes_of_symmetry Square 4 :=
  square_has_four_axes_symmetry

theorem angle_symmetry (a : Angle) : is_axis_of_symmetry (bisector a) :=
  angle_bisector_is_axis_symmetry a

-- Definitions of bisector and is_axis_of_symmetry to support the proofs
def bisector (a : Angle) : Line := sorry
def is_axis_of_symmetry (line : Line) : Prop := sorry

end isosceles_trapezoid_axes_square_axes_angle_symmetry_l373_373342


namespace volume_original_cone_l373_373306

-- Given conditions
def V_cylinder : ℝ := 21
def V_truncated_cone : ℝ := 91

-- To prove: The volume of the original cone is 94.5
theorem volume_original_cone : 
    (∃ (H R h r : ℝ), (π * r^2 * h = V_cylinder) ∧ (1 / 3 * π * (R^2 + R * r + r^2) * (H - h) = V_truncated_cone)) →
    (1 / 3 * π * R^2 * H = 94.5) :=
by
  sorry

end volume_original_cone_l373_373306


namespace min_value_of_f_min_value_at_x_1_l373_373990

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - 2 * x) + 1 / (2 - 3 * x)

theorem min_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 35 :=
by
  sorry

-- As an additional statement, we can check the specific case at x = 1
theorem min_value_at_x_1 :
  f 1 = 35 :=
by
  sorry

end min_value_of_f_min_value_at_x_1_l373_373990


namespace shaded_area_perimeter_l373_373135

/-- Define a square of side length 1 and quarter circles at each vertex. -/
structure SquareWithQuarterCircles :=
  (A B C D : ℝ)     -- vertices of the square
  (side_length : ℝ)
  (quarter_circle_centers : Set (ℝ × ℝ))

/-- The perimeter of the shaded area in a square ABCD with side length 1 meter
    and quarter circles of radius 1 meter centered at each vertex. -/
theorem shaded_area_perimeter 
  (s : SquareWithQuarterCircles) 
  (h1 : s.side_length = 1) 
  (h2 : s.quarter_circle_centers = { (0,0), (1,0), (1,1), (0,1) } )
  : s.shaded_perimeter = (2 / 3 * Real.pi) :=
sorry

end shaded_area_perimeter_l373_373135


namespace unit_disk_contains_three_points_l373_373041

-- Definition of a set of points in the plane and the condition on unit disks
def point_set := set (ℝ × ℝ)

def unit_disk_contains_at_least_one_point (P : point_set) :=
  ∀ (c : ℝ × ℝ),
  ∃ (p : ℝ × ℝ) (hyp : p ∈ P), (p.1 - c.1)^2 + (p.2 - c.2)^2 ≤ 1

-- The proof statement
theorem unit_disk_contains_three_points (P : point_set)
  (h : unit_disk_contains_at_least_one_point P) :
  ∃ (c : ℝ × ℝ), ∃ (p1 p2 p3 : ℝ × ℝ) (h1 : p1 ∈ P) (h2 : p2 ∈ P) (h3 : p3 ∈ P),
  (p1.1 - c.1)^2 + (p1.2 - c.2)^2 ≤ 1 ∧
  (p2.1 - c.1)^2 + (p2.2 - c.2)^2 ≤ 1 ∧
  (p3.1 - c.1)^2 + (p3.2 - c.2)^2 ≤ 1 ∧
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 :=
by sorry

end unit_disk_contains_three_points_l373_373041


namespace parabola_focus_distance_max_area_triangle_l373_373558

-- Part 1: Prove that p = 2
theorem parabola_focus_distance (p : ℝ) (h₀ : 0 < p) 
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> dist (0, p/2) (x, y) ≥ 4 ) : 
  p = 2 := 
by {
  sorry -- Proof will be filled in
}

-- Part 2: Prove the maximum area of ∆ PAB is 20√5
theorem max_area_triangle (p : ℝ)
  (h₀ : p = 2)
  (h₁ : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 -> 
           ∃ A B : ℝ × ℝ, A ∈ parabola_tangents C P ∧ B ∈ parabola_tangents C P ∧
                       ∃ P : ℝ × ℝ, point_on_circle M P)
  : ∃ P A B : ℝ × ℝ, area_of_triangle P A B = 20 * real.sqrt 5 :=
by {
  sorry
}

end parabola_focus_distance_max_area_triangle_l373_373558


namespace triangle_BC_length_l373_373123

/-
  In ΔABC, AB = 95, AC = 105. 
  A circle centered at A with radius AB intersects BC at points B and X. 
  Additionally, BX and CX have integer lengths. 
  Determine the length of BC.
-/

theorem triangle_BC_length 
  (A B C X : Type*)
  (dist_AB AC : ℤ) 
  (hAB : dist_AB = 95) 
  (hAC : AC = 105) 
  (circle_center_A : A)
  (circle_radius_AB : ℤ := dist_AB) 
  (h_circle : ∀ P : Type*, ∃ X : Type*, P ∈ C → dist_PX = dist_AB)
  (h_BX_integer : ∃ BX : ℤ, BX)
  (h_CX_integer : ∃ CX : ℤ, CX)
  : BC = 50 := 
sorry

end triangle_BC_length_l373_373123


namespace count_six_digit_integers_l373_373980

theorem count_six_digit_integers : 
  (finset.univ.filter 
    (λ (s : list ℕ), 
      s.length = 6 ∧ 
      s.count 1 = 2 ∧ 
      s.count 3 = 3 ∧ 
      s.count 6 = 1)).card = 60 := 
sorry

end count_six_digit_integers_l373_373980


namespace quadratic_roots_property_l373_373164

theorem quadratic_roots_property (m n : ℝ) 
  (h1 : ∀ x, x^2 - 2 * x - 2025 = (x - m) * (x - n))
  (h2 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 := 
by 
  sorry

end quadratic_roots_property_l373_373164


namespace unique_bijective_function_l373_373871

theorem unique_bijective_function (f : ℕ+ → ℕ+) (hf_bij : Function.Bijective f)
    (h_ineq : ∀ n : ℕ+, f (f n) ≤ (n + f n) / 2) : ∀ n : ℕ+, f n = n := by
  sorry

end unique_bijective_function_l373_373871


namespace sum_of_consecutive_page_numbers_l373_373707

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 283 := 
sorry

end sum_of_consecutive_page_numbers_l373_373707


namespace angle_PDO_45_degrees_l373_373362

-- Define the square configuration
variables (A B C D L P Q M N O : Type)
variables (a : ℝ) -- side length of the square ABCD

-- Conditions as hypothesized in the problem
def is_square (v₁ v₂ v₃ v₄ : Type) := true -- Placeholder for the square property
def on_diagonal_AC (L : Type) := true -- Placeholder for L being on diagonal AC
def common_vertex_L (sq1_v1 sq1_v2 sq1_v3 sq1_v4 sq2_v1 sq2_v2 sq2_v3 sq2_v4 : Type) := true -- Placeholder for common vertex L
def point_on_side (P AB_side: Type) := true -- Placeholder for P on side AB of ABCD
def square_center (center sq_v1 sq_v2 sq_v3 sq_v4 : Type) := true -- Placeholder for square's center

-- Prove the angle PDO is 45 degrees
theorem angle_PDO_45_degrees 
  (h₁ : is_square A B C D)
  (h₂ : on_diagonal_AC L)
  (h₃ : is_square A P L Q)
  (h₄ : is_square C M L N)
  (h₅ : common_vertex_L A P L Q C M L N)
  (h₆ : point_on_side P B)
  (h₇ : square_center O C M L N)
  : ∃ θ : ℝ, θ = 45 := 
  sorry

end angle_PDO_45_degrees_l373_373362


namespace determine_h_l373_373857

noncomputable def h (x : ℝ) : ℝ := -4 * x^5 - 3 * x^3 - 4 * x^2 + 12 * x + 2

theorem determine_h (x : ℝ) :
  4 * x^5 + 5 * x^3 - 3 * x + h x = 2 * x^3 - 4 * x^2 + 9 * x + 2 :=
by
  sorry

end determine_h_l373_373857


namespace alto_saxophone_ratio_l373_373822

-- Definition of the problem
def total_students : ℕ := 600
def marching_band_students : ℕ := total_students / 5
def brass_instrument_players : ℕ := marching_band_students / 2
def saxophone_players : ℕ := brass_instrument_players / 5
def alto_saxophone_players : ℕ := 4

-- Theorem to prove the ratio is 1:3
theorem alto_saxophone_ratio :
  alto_saxophone_players.toRat / saxophone_players.toRat = 1 / 3  :=
by
  sorry

end alto_saxophone_ratio_l373_373822


namespace glorias_ratio_l373_373086

variable (Q : ℕ) -- total number of quarters
variable (dimes : ℕ) -- total number of dimes, given as 350
variable (quarters_left : ℕ) -- number of quarters left

-- Given conditions
def conditions (Q dimes quarters_left : ℕ) : Prop :=
  dimes = 350 ∧
  quarters_left = (3 * Q) / 5 ∧
  (dimes + quarters_left = 392)

-- The ratio of dimes to quarters left
def ratio_of_dimes_to_quarters_left (dimes quarters_left : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd dimes quarters_left
  (dimes / gcd, quarters_left / gcd)

theorem glorias_ratio (Q : ℕ) (quarters_left : ℕ) : conditions Q 350 quarters_left → ratio_of_dimes_to_quarters_left 350 quarters_left = (25, 3) := by 
  sorry

end glorias_ratio_l373_373086


namespace round_6723_499_l373_373199

theorem round_6723_499 : round_nearest 6723.499 = 6723 := 
  sorry

end round_6723_499_l373_373199


namespace density_decrease_by_approximate_percentage_l373_373217

noncomputable def initialVolume : ℝ := 8  -- Initial volume in cubic meters.
def initialSideLength (V : ℝ) : ℝ := V^(1/3)  -- Side length function.
def sideIncrease : ℝ := 4 / 1000  -- Increase in side length in meters (4 mm converted to meters).
def newSideLength (a : ℝ) (inc : ℝ) : ℝ := a + inc  -- New side length function.
def volume (a : ℝ) : ℝ := a^3  -- Volume function based on side length.
def densityChange (V V' : ℝ) : ℝ := ((V - V') / V) * 100  -- Percentage change in volume.

theorem density_decrease_by_approximate_percentage :
  let a := initialSideLength initialVolume in
  let a' := newSideLength a sideIncrease in
  let V' := volume a' in
  densityChange initialVolume V' ≈ -0.6 :=
sorry

end density_decrease_by_approximate_percentage_l373_373217


namespace average_age_of_girls_l373_373134

theorem average_age_of_girls :
  ∀ (n_students : ℕ) (n_girls : ℕ) (avg_age_boys : ℚ) (avg_age_school : ℚ),
  n_students = 632 →
  n_girls = 158 →
  avg_age_boys = 12 →
  avg_age_school = 47 / 4 →
  let n_boys := n_students - n_girls in
  let total_age_boys := n_boys * avg_age_boys in
  let total_age_students := n_students * avg_age_school in
  let total_age_girls := total_age_students - total_age_boys in
  avg_age_girls = total_age_girls / n_girls →
  avg_age_girls ≈ 11 :=
by sorry

end average_age_of_girls_l373_373134


namespace remaining_stock_correct_l373_373629

variable (initial_stock : ℕ)
variable (sold_percent_monday sold_percent_tuesday sold_percent_wednesday sold_percent_thursday sold_percent_friday : ℝ)

def remaining_stock_after_week (initial_stock : ℕ) 
  (sold_percent_monday sold_percent_tuesday sold_percent_wednesday sold_percent_thursday sold_percent_friday : ℝ) : ℝ :=
let stock_after_monday := initial_stock - (initial_stock * sold_percent_monday)
let stock_after_tuesday := stock_after_monday - (stock_after_monday * sold_percent_tuesday)
let stock_after_wednesday := stock_after_tuesday - (stock_after_tuesday * sold_percent_wednesday)
let stock_after_thursday := stock_after_wednesday - (stock_after_wednesday * sold_percent_thursday)
let stock_after_friday := stock_after_thursday - (stock_after_thursday * sold_percent_friday)
in (stock_after_friday / initial_stock) * 100

theorem remaining_stock_correct : 
  initial_stock = 1000 ∧ 
  sold_percent_monday = 0.05 ∧ 
  sold_percent_tuesday = 0.10 ∧ 
  sold_percent_wednesday = 0.15 ∧ 
  sold_percent_thursday = 0.20 ∧ 
  sold_percent_friday = 0.25 →
  remaining_stock_after_week initial_stock 
    sold_percent_monday 
    sold_percent_tuesday 
    sold_percent_wednesday 
    sold_percent_thursday 
    sold_percent_friday = 43.7 := 
sorry

end remaining_stock_correct_l373_373629


namespace probability_woman_lawyer_l373_373295

theorem probability_woman_lawyer (total_members : ℕ) (p_women p_lawyers : ℚ)
  (h_women : p_women = 0.70)
  (h_lawyers : p_lawyers = 0.40)
  (h_total_members : total_members = 100) :
  (40 * 70) / total_members = 0.28 := by
  sorry

end probability_woman_lawyer_l373_373295


namespace hyperbola_vertices_distance_l373_373392

theorem hyperbola_vertices_distance :
  ∀ (x y : ℝ), ((y^2 / 16) - (x^2 / 9) = 1) → 
    2 * real.sqrt 16 = 8 :=
by
  intro x y h
  have a2 : real.sqrt 16 = 4 := by norm_num
  have h2a : 2 * 4 = 8 := by norm_num
  rw [←a2, h2a]
  sorry

end hyperbola_vertices_distance_l373_373392


namespace largest_even_number_l373_373999

theorem largest_even_number (x : ℕ) (h : x + (x+2) + (x+4) = 1194) : x + 4 = 400 :=
by
  have : 3*x + 6 = 1194 := by linarith
  have : 3*x = 1188 := by linarith
  have : x = 396 := by linarith
  linarith

end largest_even_number_l373_373999


namespace flight_cost_A_to_B_l373_373193

-- Definitions based on conditions in the problem
def distance_AB : ℝ := 2000
def flight_cost_per_km : ℝ := 0.10
def booking_fee : ℝ := 100

-- Statement: Given the distances and cost conditions, the flight cost from A to B is $300
theorem flight_cost_A_to_B : distance_AB * flight_cost_per_km + booking_fee = 300 := by
  sorry

end flight_cost_A_to_B_l373_373193


namespace find_x_l373_373015

open Real

theorem find_x (x : ℝ) (hx : 0 < x) (h : 4 * log 3 x = log 3 (6 * x)) : x = (6 : ℝ) ^ (1 / 3 : ℝ) := sorry

end find_x_l373_373015


namespace power_of_xy_l373_373418

-- Problem statement: Given a condition on x and y, find x^y.
theorem power_of_xy (x y : ℝ) (h : x^2 + y^2 + 4 * x - 6 * y + 13 = 0) : x^y = -8 :=
by {
  -- Proof will be added here
  sorry
}

end power_of_xy_l373_373418


namespace remainder_2468135792_mod_101_l373_373744

theorem remainder_2468135792_mod_101 : 
  2468135792 % 101 = 47 := 
sorry

end remainder_2468135792_mod_101_l373_373744


namespace parabola_condition_max_area_triangle_l373_373482

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373482


namespace toys_sold_in_first_week_l373_373818

/-
  Problem statement:
  An online toy store stocked some toys. It sold some toys at the first week and 26 toys at the second week.
  If it had 19 toys left and there were 83 toys in stock at the beginning, how many toys were sold in the first week?
-/

theorem toys_sold_in_first_week (initial_stock toys_left toys_sold_second_week : ℕ) 
  (h_initial_stock : initial_stock = 83) 
  (h_toys_left : toys_left = 19) 
  (h_toys_sold_second_week : toys_sold_second_week = 26) : 
  (initial_stock - toys_left - toys_sold_second_week) = 38 :=
by
  -- Proof goes here
  sorry

end toys_sold_in_first_week_l373_373818


namespace find_point_D_l373_373923

noncomputable def point_on_segment (A C : ℝ×ℝ) (t : ℝ) : ℝ×ℝ :=
  (A.1 + t * (C.1 - A.1), A.2 + t * (C.2 - A.2))

-- Define the vertices of the triangle
variables (A B C : ℝ×ℝ)

-- Assume D is a point on AC, parametrized by a scalar t ∈ [0, 1]
variables (t : ℝ) (h_t : 0 ≤ t ∧ t ≤ 1)

-- Define the point D on AC
def D := point_on_segment A C t

-- Define the distances (we can use the Euclidean distance)
def dist (P Q : ℝ×ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Perimeter of triangle ABD
def perimeter_ABD : ℝ := dist A B + dist B D + dist A D

-- Length of side BC
def length_BC : ℝ := dist B C

-- The theorem we need to prove
theorem find_point_D : perimeter_ABD A B C t = length_BC A B C :=
sorry

end find_point_D_l373_373923


namespace animal_legs_in_farm_l373_373233

theorem animal_legs_in_farm (total_animals ducks : ℕ) (legs_duck legs_dog : ℕ) (h1 : total_animals = 11) (h2 : ducks = 6) (h3 : legs_duck = 2) (h4 : legs_dog = 4) : (6 * 2 + (11 - 6) * 4) = 32 :=
by
  rw [h1, h2, h3, h4]
  simp
  norm_num
  sorry

end animal_legs_in_farm_l373_373233


namespace parabola_condition_max_area_triangle_l373_373489

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373489


namespace length_of_place_mat_l373_373304

noncomputable def radius : ℝ := 6
noncomputable def width : ℝ := 1.5
def inner_corner_touch (n : ℕ) : Prop := n = 6

theorem length_of_place_mat (y : ℝ) (h1 : radius = 6) (h2 : width = 1.5) (h3 : inner_corner_touch 6) :
  y = (Real.sqrt 141.75 + 1.5) / 2 :=
sorry

end length_of_place_mat_l373_373304


namespace wire_length_in_metres_l373_373779

theorem wire_length_in_metres (V : ℝ) (d : ℝ) (h : ℝ) (π : ℝ) (r : ℝ) :
  V = 44 ∧ d = 1 →
  r = d / 20 →
  V = π * r^2 * h →
  h = 5602.54 →
  h / 100 = 56.0254 :=
by
  intros _ _ _ _ _
  sorry

end wire_length_in_metres_l373_373779


namespace train_crossing_time_approx_l373_373280

noncomputable def train_length : ℝ := 90 -- in meters
noncomputable def speed_kmh : ℝ := 124 -- in km/hr
noncomputable def conversion_factor : ℝ := 1000 / 3600 -- km/hr to m/s conversion factor
noncomputable def speed_ms : ℝ := speed_kmh * conversion_factor -- speed in m/s
noncomputable def time_to_cross : ℝ := train_length / speed_ms -- time in seconds

theorem train_crossing_time_approx :
  abs (time_to_cross - 2.61) < 0.01 := 
by 
  sorry

end train_crossing_time_approx_l373_373280


namespace log_exp_inequality_solution_l373_373771

theorem log_exp_inequality_solution (x : ℝ) (h : x > 0) : 
  (|log x / log 2 - 3| + |2 ^ x - 8| ≥ 9) ↔ (0 < x ∧ x ≤ 1 ∨ 4 ≤ x) := 
sorry

end log_exp_inequality_solution_l373_373771


namespace max_sum_value_n_l373_373562

noncomputable def sequence_max_n (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  (a 1 = 13) ∧ (a 2 = 11) ∧ 
  (∀ n : ℕ, a n + a (n + 2) = 2 * a (n + 1)) ∧ 
  (∃ n : ℕ, S n = list.sum (list.map a (list.range n + 1)) ∧ n = 7)

-- Example proof structure, proof omitted
theorem max_sum_value_n (a : ℕ → ℤ) (S : ℕ → ℤ) :
  sequence_max_n a S :=
sorry

end max_sum_value_n_l373_373562


namespace find_vertex_D_l373_373079

/- Define points A, B, C with given coordinates -/ 
def A := (2, 0, -3 : ℝ × ℝ × ℝ)
def B := (5, -1, 2 : ℝ × ℝ × ℝ)
def C := (4, 4, -1 : ℝ × ℝ × ℝ)

/- Define the midpoint function -/
def midpoint (P Q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2, (P.3 + Q.3) / 2)

/- Assert D exists with the required coordinates -/
noncomputable def D : ℝ × ℝ × ℝ := (1, 5, -6)

/- Define the statement for the proof -/
theorem find_vertex_D :
  midpoint A C = midpoint B D :=
by
  unfold A B C D midpoint
  sorry

end find_vertex_D_l373_373079


namespace distance_of_hyperbola_vertices_l373_373390

-- Define the hyperbola equation condition
def hyperbola : Prop := ∃ (y x : ℝ), (y^2 / 16) - (x^2 / 9) = 1

-- Define a variable for the distance between the vertices
def distance_between_vertices (a : ℝ) : ℝ := 2 * a

-- The main statement to be proved
theorem distance_of_hyperbola_vertices :
  hyperbola → distance_between_vertices 4 = 8 :=
by
  intro h
  sorry

end distance_of_hyperbola_vertices_l373_373390


namespace min_x_squared_plus_y_squared_l373_373120

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) : x^2 + y^2 ≥ 50 := by
  sorry

end min_x_squared_plus_y_squared_l373_373120


namespace find_p_max_area_of_triangle_l373_373507

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373507


namespace hallie_reads_121_pages_on_fifth_day_l373_373571

-- Definitions for the given conditions.
def book_length : ℕ := 480
def pages_day_one : ℕ := 63
def pages_day_two : ℕ := 95 -- Rounded from 94.5
def pages_day_three : ℕ := 115
def pages_day_four : ℕ := 86 -- Rounded from 86.25

-- Total pages read from day one to day four
def pages_read_first_four_days : ℕ :=
  pages_day_one + pages_day_two + pages_day_three + pages_day_four

-- Conclusion: the number of pages read on the fifth day.
def pages_day_five : ℕ := book_length - pages_read_first_four_days

-- Proof statement: Hallie reads 121 pages on the fifth day.
theorem hallie_reads_121_pages_on_fifth_day :
  pages_day_five = 121 :=
by
  -- Proof omitted
  sorry

end hallie_reads_121_pages_on_fifth_day_l373_373571


namespace travel_time_l373_373008

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end travel_time_l373_373008


namespace exists_connected_subset_l373_373898

theorem exists_connected_subset (E V : Type) [Finite E] [Finite V] 
  (e_star : E) (x_star y_star : V) (F : ℕ → Subgraph V)
  (h1 : e_star ∈ E) 
  (h2 : e_star ∉ (E ∩ ⋃ (i : ℕ), (F i).edges)) : 
  ∃ U : Set V, 
    (∀ i : ℕ, ∃ path : List (F i).vertices, path.head = x_star ∧ path.tail = y_star ∧ (∀ v ∈ path, v ∈ U)) := 
begin
  sorry
end

end exists_connected_subset_l373_373898


namespace total_amount_invested_l373_373817

-- Define the problem details: given conditions
def interest_rate_share1 : ℚ := 9 / 100
def interest_rate_share2 : ℚ := 11 / 100
def total_interest_rate : ℚ := 39 / 400
def amount_invested_share2 : ℚ := 3750

-- Define the total amount invested (A), the amount invested at the 9% share (x)
variable (A x : ℚ)

-- Conditions
axiom condition1 : x + amount_invested_share2 = A
axiom condition2 : interest_rate_share1 * x + interest_rate_share2 * amount_invested_share2 = total_interest_rate * A

-- Prove that the total amount invested in both types of shares is Rs. 10,000
theorem total_amount_invested : A = 10000 :=
by {
  -- proof goes here
  sorry
}

end total_amount_invested_l373_373817


namespace acute_triangle_third_side_range_l373_373924

-- Given a triangle with sides 2, 3, and x, prove that if the triangle is acute-angled, 
-- then √5 < x < √13.
theorem acute_triangle_third_side_range (x : ℝ) :
  (2^2 + 3^2 > x^2) ∧ (2^2 + x^2 > 3^2) ∧ (3^2 + x^2 > 2^2) → sqrt 5 < x ∧ x < sqrt 13 :=
by
  sorry

end acute_triangle_third_side_range_l373_373924


namespace line_through_two_points_line_with_intercept_sum_l373_373881

theorem line_through_two_points (a b x1 y1 x2 y2: ℝ) : 
  (x1 = 2) → (y1 = 1) → (x2 = 0) → (y2 = -3) → (2 * x - y - 3 = 0) :=
by
                
  sorry

theorem line_with_intercept_sum (a b : ℝ) (x y : ℝ) :
  (x = 0) → (y = 5) → (a + b = 2) → (b = 5) → (5 * x - 3 * y + 15 = 0) :=
by
  sorry

end line_through_two_points_line_with_intercept_sum_l373_373881


namespace num_arrangements_l373_373894

theorem num_arrangements (A B C D E : Type) : 
  ∃ l : list (list Type), 
  ((A :: l) <|> (l ++ [A])) ∧ 
  (permutations l (C :: D :: [])) ∧
  (permutations l (D :: C :: [])) ∧
  (((A :: l) <|> (l ++ [A])).length = 120) → 
  (24 = list.countp (λ lst, lst.nth 0 ≠ A ∧ lst.nth 4 ≠ A) [A, B, C, D, E].permutations) := by sorry

end num_arrangements_l373_373894


namespace smallest_r_minus_p_l373_373720

theorem smallest_r_minus_p (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (prod_eq : p * q * r = nat.factorial 9) (h_lt : p < q ∧ q < r) :
  r - p = 219 := 
sorry

end smallest_r_minus_p_l373_373720


namespace perpendicular_os_bc_l373_373647

variable {A B C O S : Type}

noncomputable def acute_triangle (A B C : Type) := true -- Placeholder definition for acute triangle.

noncomputable def circumcenter (O : Type) (A B C : Type) := true -- Placeholder definition for circumcenter.

noncomputable def line_intersects_circumcircle_second_time (AC : Type) (circ : Type) (S : Type) := true -- Placeholder def.

-- Define the problem in Lean
theorem perpendicular_os_bc
  (ABC_is_acute : acute_triangle A B C)
  (O_is_circumcenter : circumcenter O A B C)
  (AC_intersects_AOB_circumcircle_at_S : line_intersects_circumcircle_second_time (A → C) (A → B → O) S) :
  true := -- Place for the proof that OS ⊥ BC
sorry

end perpendicular_os_bc_l373_373647


namespace find_p_max_area_triangle_l373_373533

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373533


namespace max_gcd_2015xy_l373_373396

theorem max_gcd_2015xy (x y : ℤ) (coprime : Int.gcd x y = 1) :
    ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
sorry

end max_gcd_2015xy_l373_373396


namespace find_k_l373_373970

-- Define the problem parameters
variables {x y k : ℝ}

-- The conditions given in the problem
def system_of_equations (x y k : ℝ) : Prop :=
  (x + 2 * y = k - 1) ∧ (2 * x + y = 5 * k + 4)

def solution_condition (x y : ℝ) : Prop :=
  x + y = 5

-- The proof statement
theorem find_k (x y k : ℝ) (h1 : system_of_equations x y k) (h2 : solution_condition x y) :
  k = 2 :=
sorry

end find_k_l373_373970


namespace hypotenuse_length_l373_373268

theorem hypotenuse_length 
  (x y : ℝ)
  (h1 : (1/3) * π * y * x^2 = 972 * π)
  (h2 : (1/3) * π * x * y^2 = 1458 * π) :
  (hypotenuse : ℝ) :=
hypotenuse = 12 * sqrt 5
proof := sorry

end hypotenuse_length_l373_373268


namespace product_of_roots_l373_373988

theorem product_of_roots (x : ℝ) (h : (x - 1) * (x + 4) = 22) : ∃ a b, (x^2 + 3*x - 26 = 0) ∧ a * b = -26 :=
by
  -- Given the equation (x - 1) * (x + 4) = 22,
  -- We want to show that the roots of the equation when simplified are such that
  -- their product is -26.
  sorry

end product_of_roots_l373_373988


namespace electronics_sale_negation_l373_373119

variables (E : Type) (storeElectronics : E → Prop) (onSale : E → Prop)

theorem electronics_sale_negation
  (H : ¬ ∀ e, storeElectronics e → onSale e) :
  (∃ e, storeElectronics e ∧ ¬ onSale e) ∧ ¬ ∀ e, storeElectronics e → onSale e :=
by
  -- Proving that at least one electronic is not on sale follows directly from the negation of the universal statement
  sorry

end electronics_sale_negation_l373_373119


namespace area_of_annulus_l373_373816

open Real

/-- Defining the radii of the concentric circles -/
variables (b c f : ℝ)
-- b is radius of the larger circle, c is radius of the smaller circle, and f is the length of the tangent segment from the circle foci

/-- Conditions of the problem -/
variable (h_b_gt_c : b > c)
variable (h_tangent : b^2 = c^2 + f^2)

/-- Statement to prove the area of the annulus in terms of π and f -/
theorem area_of_annulus : π * (b^2 - c^2) = π * f^2 :=
by 
  rw [← h_tangent]
  sorry

end area_of_annulus_l373_373816


namespace max_N_k_l373_373896

def I_k (k : ℕ) : ℕ :=
  10^(k + 2) + 64 -- Representation of I_k in the form as specified in the problem

def factors_of_2 (n : ℕ) : ℕ :=
  nat.find_greatest (λ d, 2^d ∣ n) (n.log 2 + 1) -- Finding the highest power of 2 that divides n

def N (k : ℕ) : ℕ :=
  factors_of_2 (I_k k) -- N(k) is the number of factors of 2 in the prime factorization of I_k

theorem max_N_k (k : ℕ) (h : k > 0) : ∃ k, N k = 7 :=
  sorry -- Placeholder for the proof

end max_N_k_l373_373896


namespace minimum_x_plus_2y_l373_373911

variable (x y : ℝ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : (1 / (2 * x + y)) + (1 / (y + 1)) = 1)

theorem minimum_x_plus_2y : x + 2 * y = sqrt 3 + 1 / 2 :=
sorry

end minimum_x_plus_2y_l373_373911


namespace proof_part1_proof_part2_l373_373473

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373473


namespace fastest_growth_in_1995_to_2000_l373_373331

variable (LivingArea : ℕ → ℝ)
variable (is_valid_year : ℕ → Prop)

-- Assumptions on valid years and the specific period of interest
def valid_year_range := ∀ y, is_valid_year y ↔ (1985 ≤ y ∧ y ≤ 2000)

-- Definition for the relative growth in a five-year period
def growth_rate (start : ℕ) : ℝ := LivingArea (start + 5) - LivingArea start

-- Specific periods to compare
def periods := [(1985,1990), (1990,1995), (1995,2000)]

-- The period with the fastest growth
def fastest_growth_period : (ℕ × ℕ) :=
  if growth_rate 1985 < growth_rate 1990 && growth_rate 1990 < growth_rate 1995
  then (1995, 2000) else (0,0)  -- Simplified for illustration

theorem fastest_growth_in_1995_to_2000 : 
  fastest_growth_period LivingArea = (1995, 2000) := 
begin
  sorry
end

end fastest_growth_in_1995_to_2000_l373_373331


namespace find_p_max_area_triangle_l373_373534

-- Definitions and conditions for Part 1
def parabola (p : ℝ) := p > 0 ∧ ∀ x y : ℝ, y = x^2 / (2 * p)
def focus (p : ℝ) : ℝ × ℝ := (0, p / 2)
def circle (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def minimum_distance (F : ℝ × ℝ) (d : ℝ) : Prop := ∃ x y, circle x y ∧ sqrt((x - F.1)^2 + (y - F.2)^2) = d
def distance_condition := minimum_distance (focus 2) 4

-- Statement for Part 1 proof
theorem find_p : ∃ p, parabola p ∧ distance_condition := by
  sorry

-- Definitions and conditions for Part 2
def tangent_to_parabola (P : ℝ × ℝ) (p : ℝ) := -- Assume suitable definition for tangents
def P_on_circle := ∃ x y, circle x y
def tangents_P (P : ℝ × ℝ) (tangents : list (ℝ × ℝ)) := ∀ p, tangent_to_parabola P p
def area_PAB (P : ℝ) (A B : ℝ × ℝ) : ℝ := -- Assume suitable definition for area

-- Statement for Part 2 proof
theorem max_area_triangle (P : ℝ × ℝ) (A B : ℝ × ℝ) : P_on_circle ∧ tangents_P P [A, B] → area_PAB P A B = 20 * sqrt 5 := by
  sorry

end find_p_max_area_triangle_l373_373534


namespace find_p_max_area_of_triangle_l373_373510

def parabola (p : ℝ) : set (ℝ × ℝ) :=
  {xy | xy.1^2 = 2 * p * xy.2 ∧ p > 0}

def circle : set (ℝ × ℝ) :=
  {xy | xy.1^2 + (xy.2 + 4)^2 = 1}

def focus (p : ℝ) : ℝ × ℝ :=
  (0, p/2)

def min_distance (p : ℝ) : ℝ :=
  let f := focus p in
  let distance_fn := λ (x y : ℝ), real.sqrt ((x - f.1)^2 + (y - f.2)^2) in
  inf (distance_fn <$> (circle.image prod.fst) <*> (circle.image prod.snd))

theorem find_p: 
  ∃ p > 0, min_distance p = 4 :=
sorry

theorem max_area_of_triangle (P A B: ℝ × ℝ) (P_on_circle : P ∈ circle)
  (A_on_parabola : ∃ p > 0, A ∈ parabola p)
  (B_on_parabola : ∃ p > 0, B ∈ parabola p)
  (PA_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P A p)
  (PB_tangent_to_parabola : ∃ p > 0, tangent_to_parabola P B p) :
  ∃ max_area, max_area = 20 * real.sqrt 5 :=
sorry

end find_p_max_area_of_triangle_l373_373510


namespace tangent_line_through_origin_is_y_eq_x_l373_373310

open Real

-- Define the problem setup: circle equation, line passing through origin, tangent point in third quadrant.

def is_tangent_line (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ P : ℝ × ℝ, C P.1 P.2 ∧ l P.1 P.2 ∧ is_tangent l P

def circle (x y : ℝ) : Prop := x^2 + y^2 + 4 * x + 3 = 0

def line_passing_through_origin (x y : ℝ) : Prop := y = x

-- Proving that the line_passing_through_origin is tangent to circle
theorem tangent_line_through_origin_is_y_eq_x :
  is_tangent_line line_passing_through_origin circle :=
sorry

end tangent_line_through_origin_is_y_eq_x_l373_373310


namespace part_i_part_ii_l373_373420

-- Proof Problem part (i)
theorem part_i (n : ℕ) (h : n ≥ 3) (x : Fin n → ℝ) 
(hp : p = ∑ i, x i) (hq : q = ∑ i j, if i < j then x i * x j else 0) :
  (n - 1) * p^2 / n - 2 * q ≥ 0 := 
sorry

-- Proof Problem part (ii)
theorem part_ii (n : ℕ) (h : n ≥ 3) (x : Fin n → ℝ) (i : Fin n)
(hp : p = ∑ i, x i) (hq : q = ∑ i j, if i < j then x i * x j else 0) :
  | x i - p / n | ≤ (n - 1) / n * ((p^2 - 2 * n / (n - 1) * q)^(1/2)) := 
sorry

end part_i_part_ii_l373_373420


namespace quadratic_roots_property_l373_373165

theorem quadratic_roots_property (m n : ℝ) 
  (h1 : ∀ x, x^2 - 2 * x - 2025 = (x - m) * (x - n))
  (h2 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 := 
by 
  sorry

end quadratic_roots_property_l373_373165


namespace minyoung_position_from_front_l373_373131

theorem minyoung_position_from_front : 
  ∀ (n m : ℕ), n = 27 → m = 13 → (m + (27 - m + 1) = 28) → (27 - m + 1 = 15) :=
by
  intros n m h1 h2 h3
  rwa [h1, h2] at h3

end minyoung_position_from_front_l373_373131


namespace maximal_length_sequence_l373_373869

theorem maximal_length_sequence :
  ∃ (a : ℕ → ℤ) (n : ℕ), (∀ i, 1 ≤ i → i + 6 ≤ n → (a i + a (i + 1) + a (i + 2) + a (i + 3) + a (i + 4) + a (i + 5) + a (i + 6) > 0)) ∧ 
                          (∀ j, 1 ≤ j → j + 10 ≤ n → (a j + a (j + 1) + a (j + 2) + a (j + 3) + a (j + 4) + a (j + 5) + a (j + 6) + a (j + 7) + a (j + 8) + a (j + 9) + a (j + 10) < 0)) ∧ 
                          n = 16 :=
sorry

end maximal_length_sequence_l373_373869


namespace evaluate_i_powers_l373_373862

theorem evaluate_i_powers : (i : ℂ) (H : i^4 = 1) : i^12 + i^17 + i^22 + i^27 + i^32 = 1 :=
by
  sorry

end evaluate_i_powers_l373_373862


namespace polynomial_expression_l373_373759

theorem polynomial_expression :
  (λ x : ℝ, (3 * x^2 + 4 * x + 8) * (x - 1) - (x - 1) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 1) * (x + 6))
  = λ x, 6 * x^3 + 2 * x^2 - 18 * x + 10 :=
by
  sorry

end polynomial_expression_l373_373759


namespace count_integers_with_at_most_three_different_digits_l373_373097

theorem count_integers_with_at_most_three_different_digits :
  let count : ℕ := 
    let single_digit := 9
    let two_different_digits := 
      let combinations_no_zero := 36
      let non_repetitive_arrangements := 2 + 6 + 14 + 30
      let including_zero := 9
      let arrangements_including_zero := 1 + 3 + 7 + 15
      (combinations_no_zero * non_repetitive_arrangements) + (including_zero * arrangements_including_zero)
    let three_different_digits := 
      let combinations_three_digits_no_zero := 84
      let arrangements_no_zero := 15 + 66 + 222
      let including_zero_combinations := 36
      let arrangements_including_zero := 6 + 20 + 56
      (combinations_three_digits_no_zero * arrangements_no_zero) + (including_zero_combinations * arrangements_including_zero)

    single_digit + two_different_digits + three_different_digits
  in count = 29555 :=
by
  sorry

end count_integers_with_at_most_three_different_digits_l373_373097


namespace cannot_determine_type_of_quadrilateral_l373_373318

-- Definition of a quadrilateral with equal diagonals
def quadrilateral_with_equal_diagonals : Prop := 
  ∃ (quad : Type) (d1 d2 : quad → quad), ∀ (p : quad), d1 p = d2 p

-- Theorem stating that it cannot be determined whether the quadrilateral is a rectangle, an isosceles trapezoid, or a square
theorem cannot_determine_type_of_quadrilateral (h : quadrilateral_with_equal_diagonals) :
  ¬(∃ (quad : Type), (is_rectangle quad ∨ is_isosceles_trapezoid quad ∨ is_square quad) ↔ h) := 
sorry

end cannot_determine_type_of_quadrilateral_l373_373318


namespace find_counterfeit_coins_l373_373716

def Coin := ℕ

noncomputable def num_coins : ℕ := 23
noncomputable def num_counterfeits : ℕ := 6

axiom consecutive_counterfeits (coins : List Coin) : 
  ∃ s t, s > 0 ∧ t > 0 ∧ length coins = num_coins ∧ length (drop s (take (s + num_counterfeits) coins)) = num_counterfeits

axiom differs_in_weight (c1 c2 : Coin) : c1 ≠ c2 → c1.weight ≠ c2.weight

theorem find_counterfeit_coins (coins : List Coin) : 
  consecutive_counterfeits coins → 
  (∃ c, c ∈ coins ∧ is_counterfeit c) :=
begin
  sorry
end

end find_counterfeit_coins_l373_373716


namespace B_profit_solution_l373_373296

-- Define the conditions as hypotheses
variables (B_investment A_investment B_period A_period : ℝ)
variables (B_proft Total_profit : ℝ)

-- Given conditions
def three_times_investment := A_investment = 3 * B_investment
def two_times_period := A_period = 2 * B_period
def total_profit := Total_profit = 42000

-- The profit sharing ratio
def ratio := 6 / (6 + 1)

-- The final statement to prove
theorem B_profit_solution (h1 : three_times_investment) (h2 : two_times_period) (h3 : total_profit) : 
  B_proft = (1 / 7) * Total_profit :=
by
  -- Proof goes here
  sorry

end B_profit_solution_l373_373296


namespace circle_area_l373_373580

theorem circle_area (r : ℝ) (h : 6 / (2 * π * r) = r / 2) : π * r^2 = 3 :=
by
  sorry

end circle_area_l373_373580


namespace number_of_correct_statements_l373_373446

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then exp(x) * (x + 1) else -exp(-x) * (1 - x)

theorem number_of_correct_statements :
  (is_odd f) ∧ 
  (∀ x < 0, f x = real.exp x * (x + 1)) →
  (¬(∀ x > 0, f x = real.exp x * (1 - x)) ∧ 
  ¬(∃ x1 x2 x3, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) ∧ 
  (¬((∀ x, f x > 0) ∨ f x > 0 = set.Ioo (-1) 0 ∪ set.Ioi 1)) ∧ 
  (¬(∀ x1 x2 : ℝ, abs (f x1 - f x2) < 2))) → 
  2 :=
by
  sorry

end number_of_correct_statements_l373_373446


namespace remainder_2468135792_div_101_l373_373734

theorem remainder_2468135792_div_101 : (2468135792 % 101) = 52 := 
by 
  -- Conditions provided in the problem
  have decompose_num : 2468135792 = 24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92, 
  from sorry,
  
  -- Assert large powers of 10 modulo properties
  have ten_to_pow2 : (10^2 - 1) % 101 = 0, from sorry,
  have ten_to_pow4 : (10^4 - 1) % 101 = 0, from sorry,
  have ten_to_pow6 : (10^6 - 1) % 101 = 0, from sorry,
  have ten_to_pow8 : (10^8 - 1) % 101 = 0, from sorry,
  
  -- Summing coefficients
  have coefficients_sum : 24 + 68 + 13 + 57 + 92 = 254, from
  by linarith,
  
  -- Calculating modulus
  calc 
    2468135792 % 101
        = (24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92) % 101 : by rw decompose_num
    ... = (24 + 68 + 13 + 57 + 92) % 101 : by sorry
    ... = 254 % 101 : by rw coefficients_sum
    ... = 52 : by norm_num,

  sorry

end remainder_2468135792_div_101_l373_373734


namespace ratio_total_surface_area_lateral_surface_area_l373_373586

theorem ratio_total_surface_area_lateral_surface_area (central_angle : ℝ) (l : ℝ) (h_angle : central_angle = 120) : 
  let r := l / 3,
      S_lateral := π * r * l,
      S_total := S_lateral + π * r^2
  in (S_total / S_lateral) = (4 / 3) :=
by
  -- Definitions based on given conditions
  let r := l / 3
  let S_lateral := π * r * l
  let S_total := S_lateral + π * r^2
  
  -- Lateral surface area calculation
  have h_lateral : S_lateral = π * (l / 3) * l
    := sorry

  have h_lateral_simp : S_lateral = (π * l^2 / 3)
    := sorry

  -- Base area calculation
  have h_base : π * r^2 = π * (l / 3) ^ 2
    := sorry

  have h_base_simp : π * r^2 = π * l^2 / 9
    := sorry

  -- Total surface area calculation
  have h_total : S_total = (π * l^2 / 3) + (π * l^2 / 9)
    := sorry

  have h_total_simp : S_total = (4 * π * l^2 / 9)
    := sorry

  -- Ratio of total to lateral surface area
  have h_ratio : (S_total / S_lateral) = (4 * π * l^2 / 9) / (π * l^2 / 3)
    := sorry

  have h_ratio_simp : (S_total / S_lateral) = 4 / 3
    := sorry

  show (S_total / S_lateral) = 4 / 3
    from h_ratio_simp

end ratio_total_surface_area_lateral_surface_area_l373_373586


namespace combined_cost_of_items_l373_373843

theorem combined_cost_of_items (wallet_cost : ℕ) 
  (purse_cost : ℕ) (combined_cost : ℕ) :
  wallet_cost = 22 →
  purse_cost = 4 * wallet_cost - 3 →
  combined_cost = wallet_cost + purse_cost →
  combined_cost = 107 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end combined_cost_of_items_l373_373843


namespace train_cross_pole_time_l373_373328

theorem train_cross_pole_time (train_length : ℝ) (train_speed_kmh : ℝ) (conversion_factor : ℝ) 
  (h1 : train_length = 360) 
  (h2 : train_speed_kmh = 43.2) 
  (h3 : conversion_factor = 1000 / 3600) : 
  ((train_speed_kmh * conversion_factor) ≠ 0 → train_length / (train_speed_kmh * conversion_factor) = 30) :=
by 
  intros h
  rw [h1, h2, h3]
  have speed_ms : ℝ := 43.2 * (1000 / 3600)
  have h_speed : speed_ms = 12 := by sorry
  rw [←h_speed]
  norm_num
  sorry

end train_cross_pole_time_l373_373328


namespace sqrt_50_value_l373_373179

def f (x : ℝ) : ℝ :=
  if x ∈ Int then 7 * x + 3 else Real.floor x + 6

theorem sqrt_50_value : f (Real.sqrt 50) = 13 :=
by
  sorry

end sqrt_50_value_l373_373179


namespace ABCD_equals_one_l373_373050

-- Define the constants A, B, C, and D
def A : ℝ := real.sqrt 2008 + real.sqrt 2009
def B : ℝ := - real.sqrt 2008 - real.sqrt 2009
def C : ℝ := real.sqrt 2008 - real.sqrt 2009
def D : ℝ := real.sqrt 2009 - real.sqrt 2008

-- The theorem to prove
theorem ABCD_equals_one : A * B * C * D = 1 := by
  sorry

end ABCD_equals_one_l373_373050


namespace range_of_positive_integers_in_set_J_l373_373677

theorem range_of_positive_integers_in_set_J (J : Finset ℤ) (h1 : J.card = 22)
  (h2 : ∃ k : ℤ, (∀ n : ℤ, (n ∈ J) ↔ (∃ i : ℤ, 0 ≤ i ∧ i < 22 ∧ n = k + 2 * i)) ∧ (k + 8 = -10)) :
  (Finset.filter (λ x, x > 0) J).range = 22 :=
by
  sorry

end range_of_positive_integers_in_set_J_l373_373677


namespace circle_tangent_to_axes_and_line_l373_373303

theorem circle_tangent_to_axes_and_line (a b : ℝ)
  (h1 : 2 * a - b + 6 = 0)
  (h2 : abs a = abs b) :
  (eq ((λ (x y : ℝ), x = -2 ∧ y = 2) a b) ∨ eq ((λ (x y : ℝ), x = -6 ∧ y = -6) a b)) :=
by
  sorry

end circle_tangent_to_axes_and_line_l373_373303


namespace find_k_l373_373082

variables (a b : ℝ × ℝ)
variables (k : ℝ)

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (2, -1)

def k_a_plus_b (k : ℝ) : ℝ × ℝ := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2)
def a_minus_2b : ℝ × ℝ := (vector_a.1 - 2 * vector_b.1, vector_a.2 - 2 * vector_b.2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_product (k_a_plus_b k) a_minus_2b = 0 ↔ k = 2 :=
by
  sorry

end find_k_l373_373082


namespace g_of_900_eq_34_l373_373159

theorem g_of_900_eq_34 (g : ℕ+ → ℝ) 
  (h_mul : ∀ x y : ℕ+, g (x * y) = g x + g y)
  (h_30 : g 30 = 17)
  (h_60 : g 60 = 21) :
  g 900 = 34 :=
sorry

end g_of_900_eq_34_l373_373159


namespace find_sin_combined_angle_l373_373061

-- Define the sine and cosine of alpha based on the given point P (-1, 2)
noncomputable def sin_alpha : ℝ := 2 / real.sqrt 5
noncomputable def cos_alpha : ℝ := -1 / real.sqrt 5

-- Define the double-angle sine and cosine
noncomputable def sin_2alpha : ℝ := 2 * sin_alpha * cos_alpha
noncomputable def cos_2alpha : ℝ := cos_alpha^2 - sin_alpha^2

-- Define the value to prove
noncomputable def result : ℝ := (4 - 3 * real.sqrt 3) / 10

-- Statement of the problem
theorem find_sin_combined_angle :
  sin (2 * asin(sin_alpha) + 2/3 * real.pi) = result :=
sorry

end find_sin_combined_angle_l373_373061


namespace integral_x_pow_probability_l373_373610

theorem integral_x_pow_probability :
  let p := 1 / 6 in
  ∫ (x : ℝ) in 0..1, x ^ p = 6 / 7 :=
by
  sorry

end integral_x_pow_probability_l373_373610


namespace parabola_condition_max_area_triangle_l373_373491

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

theorem parabola_condition (p : ℝ) (h₀ : 0 < p) : 
  ((p / 2 + 4 - 1 = 4) → (p = 2)) :=
by sorry

theorem max_area_triangle (P : ℝ × ℝ) (k b : ℝ) 
  (h₀ : P.1 ^ 2 + (P.2 + 4) ^ 2 = 1) 
  (h₁ : P.1 = 2 * k) 
  (h₂ : -P.2 = b) 
  (h₃ : k ^ 2 + (b - 4) ^ 2 < 1) :
  4 * ((k ^ 2 + b) ^ (3 / 2)) = 20 * Real.sqrt 5 :=
by sorry

end parabola_condition_max_area_triangle_l373_373491


namespace find_fx_sum_roots_l373_373222

noncomputable def f : ℝ → ℝ
| x => if x = 2 then 1 else Real.log (abs (x - 2))

theorem find_fx_sum_roots
  (b c : ℝ)
  (x1 x2 x3 x4 x5 : ℝ)
  (h : ∀ x, (f x) ^ 2 + b * (f x) + c = 0)
  (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x3 ≠ x4 ∧ x3 ≠ x5 ∧ x4 ≠ x5 ) :
  f (x1 + x2 + x3 + x4 + x5) = Real.log 8 :=
sorry

end find_fx_sum_roots_l373_373222


namespace orange_sacks_after_95_days_l373_373572

-- Define the conditions as functions or constants
def harvest_per_day : ℕ := 150
def discard_per_day : ℕ := 135
def days_of_harvest : ℕ := 95

-- State the problem formally
theorem orange_sacks_after_95_days :
  (harvest_per_day - discard_per_day) * days_of_harvest = 1425 := 
by 
  sorry

end orange_sacks_after_95_days_l373_373572


namespace f_domain_and_period_f_monotonicity_and_extremum_l373_373066

def f (x : ℝ) : ℝ := 4 * tan x * sin (π / 2 - x) * cos (x - π / 3) - sqrt 3

theorem f_domain_and_period : 
    (∀ k : ℤ, x ≠ (k : ℝ) * π + π / 2) 
    ∧ (∀ T > 0, (∀ (x : ℝ), f (x + T) = f x) → T = π) :=
sorry

theorem f_monotonicity_and_extremum :
    (∀ x ∈ Icc (-π/12 : ℝ) (π/4 : ℝ), ∀ y ∈ Icc f x f y, f x <= f y) 
    ∧ (∀ x ∈ Icc (-π/4 : ℝ) (-π/12 : ℝ), ∀ y ∈ Icc f y f x, f x >= f y) 
    ∧ (∃ min_x ∈ Icc (-π/12) (π/4), f min_x = -2) 
    ∧ (∃ max_x ∈ Icc (-π/4) (π/12), f max_x = 1) :=
sorry

end f_domain_and_period_f_monotonicity_and_extremum_l373_373066


namespace product_of_tangents_l373_373360

theorem product_of_tangents : 
  (Real.tan (Real.pi / 8) * Real.tan (3 * Real.pi / 8) * 
   Real.tan (5 * Real.pi / 8) * Real.tan (7 * Real.pi / 8) = -2 * Real.sqrt 2) :=
sorry

end product_of_tangents_l373_373360


namespace trig_identity_l373_373290

theorem trig_identity : 
  (cos (real.pi / 15) - (cos (real.pi / 10) * sin (real.pi / 3))) / sin (real.pi / 10) = 1 / 2 :=
by
  sorry

end trig_identity_l373_373290


namespace monotonic_intervals_a_eq_1_minimum_a_no_zeros_l373_373955

noncomputable def f (a x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- Problem 1
theorem monotonic_intervals_a_eq_1 (x : ℝ) (hx : 0 < x) : 
    ((0 < x ∧ x ≤ 2) → (f 1 x) is_decreasing_on (0, 2]) ∧
    ((2 < x ∧ x < +∞) → (f 1 x) is_increasing_on [2, +∞)) :=
sorry

-- Problem 2
theorem minimum_a_no_zeros (x : ℝ) (hx : 0 < x ∧ x < 1 / 2) :
    (∀ x ∈ (0, 1 / 2), f a x > 0) ↔ a ≥ 2 - 4 * Real.log 2 :=
sorry

end monotonic_intervals_a_eq_1_minimum_a_no_zeros_l373_373955


namespace proof_part1_proof_part2_l373_373480

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) := (0, p / 2)

def min_distance_condition (p : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus p).snd + 4 - 1 = 4

def parabola (p : ℝ) : Prop := x^2 = 2 * p * y

axiom parabolic_eq : ∀ x y : ℝ, parabola 2

theorem proof_part1 : ∀ x y : ℝ, x^2 + (y + 4)^2 = 1 → (parabola_focus 2).snd + 4 - 1 = 4 :=
by sorry

noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  (1 / 2) * (abs (x1 * y2 - x2 * y1))

def max_triangle_area_condition (x y : ℝ) (A B : ℝ → ℝ) (tangent : ℝ → ℝ → ℝ) (P : (ℝ × ℝ)) : Prop :=
  ∀ P ∈ circle 1 (0, -4), P ∈ M_tangent_1 (A x) (B x) → triangle_area (A x) (B x) (P) = 20 * sqrt 5

theorem proof_part2 : ∀ P ∈ circle 1 (0, -4), 
  ∀ A B : (ℝ → ℝ),
  ∀ tangent : ℝ → ℝ → ℝ,
  max_triangle_area_condition P A B tangent P :=
by sorry

end proof_part1_proof_part2_l373_373480


namespace calculate_abs_mul_l373_373832

theorem calculate_abs_mul : |(-3 : ℤ)| * 2 = 6 := 
by 
  -- |(-3)| equals 3 and 3 * 2 equals 6.
  -- The "sorry" is used to complete the statement without proof.
  sorry

end calculate_abs_mul_l373_373832


namespace find_salary_l373_373765

variable (S : ℝ)
variable (house_rent_percentage : ℝ) (education_percentage : ℝ) (clothes_percentage : ℝ)
variable (remaining_amount : ℝ)

theorem find_salary (h1 : house_rent_percentage = 0.20)
                    (h2 : education_percentage = 0.10)
                    (h3 : clothes_percentage = 0.10)
                    (h4 : remaining_amount = 1377)
                    (h5 : (1 - clothes_percentage) * (1 - education_percentage) * (1 - house_rent_percentage) * S = remaining_amount) :
                    S = 2125 := 
sorry

end find_salary_l373_373765


namespace tan_alpha_mul_tan_beta_l373_373040

variables (a b : ℝ)

theorem tan_alpha_mul_tan_beta 
  (α β : ℝ)
  (P : (ℝ × ℝ × ℝ) ) -- point P in 3D space
  (h1 : true)        -- This needs to formalize point P on AC_1, consider true since the specific geometry is not defined
  (h2 : α = real.arctan ((P.1 - a) / P.3))
  (h3 : β = real.arctan ((P.2 - b) / P.1)) :
  real.tan α * real.tan β = (real.sqrt 2 * b * real.sqrt (a^2 + b^2)) / (2 * (a^2 + b^2)) := sorry

end tan_alpha_mul_tan_beta_l373_373040


namespace triangle_bc_length_l373_373059

theorem triangle_bc_length (A B C : Point) -- A, B, C are points representing vertices of the triangle
    (r : ℝ) -- r represents the radius of the circumcircle
    (h_r_eq : r = (Real.sqrt 3 / 3) * (dist B C)) -- Radius condition
    (h_AB : dist A B = 3) -- AB = 3
    (h_AC : dist A C = 4) -- AC = 4
    (h_acute : ∀ (x y z : Point), IsAcuteAngle x y z) -- Acute angle condition
    : dist B C = Real.sqrt 13 := -- Prove BC = sqrt(13)
sorry -- Proof is omitted

end triangle_bc_length_l373_373059


namespace trigonometric_identity_l373_373979

-- Define vectors a and b as functions of theta
def vec_a (θ : ℝ) : ℝ × ℝ := (Real.sin θ, -2)
def vec_b (θ : ℝ) : ℝ × ℝ := (1, Real.cos θ)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (θ : ℝ) : Prop :=
  let (a1, a2) := vec_a θ
  let (b1, b2) := vec_b θ
  a1 * b1 + a2 * b2 = 0

-- The main theorem that proves the value of sin(2θ) + cos²(θ) is 1 when the vectors are perpendicular
theorem trigonometric_identity (θ : ℝ) (h : perpendicular θ) : Real.sin (2 * θ) + (Real.cos θ)^2 = 1 :=
sorry

end trigonometric_identity_l373_373979


namespace sqrt_four_eq_two_l373_373830

theorem sqrt_four_eq_two : Real.sqrt 4 = 2 :=
by
  sorry

end sqrt_four_eq_two_l373_373830


namespace eq_sum_disjoint_subsets_of_10_twodigits_l373_373667

theorem eq_sum_disjoint_subsets_of_10_twodigits (E : Finset ℕ)
  (h_card : E.card = 10)
  (h_range : ∀ x ∈ E, 10 ≤ x ∧ x ≤ 99) :
  ∃ (A B : Finset ℕ), A ⊆ E ∧ B ⊆ E ∧ A ∩ B = ∅ ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ A.sum id = B.sum id := by
  sorry

end eq_sum_disjoint_subsets_of_10_twodigits_l373_373667


namespace find_p_max_area_triangle_l373_373512

-- Define given conditions in lean
structure Parabola (p : ℝ) :=
(h_p : p > 0)
(eq : ∀ x y, x^2 = 2 * p * y)

structure Circle :=
(eq : ∀ x y, x^2 + (y + 4)^2 = 1)

-- Define the key distance condition
def min_distance (F P : ℝ × ℝ) (d : ℝ) : Prop :=
dist F P - 1 = d

-- Define problems to prove
theorem find_p (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle) (F : ℝ × ℝ) (d : ℝ)
  (hd : min_distance F (0, -4) 4) : p = 2 :=
sorry

theorem max_area_triangle (p : ℝ) (C_parabola : Parabola p) (M_circle : Circle)
  (P A B : ℝ × ℝ) (PA PB : ℝ) : p = 2 → PA = P → PB = B → 
  ∃ A B : ℝ × ℝ, max_area (P A B) = 20 * sqrt 5 :=
sorry

end find_p_max_area_triangle_l373_373512


namespace roots_eq_solution_l373_373162

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end roots_eq_solution_l373_373162


namespace B_pow_2048_l373_373632

open Real Matrix

noncomputable def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![cos (π / 4), 0, -sin (π / 4)],
    ![0, 1, 0],
    ![sin (π / 4), 0, cos (π / 4)]]

theorem B_pow_2048 :
  B ^ 2048 = (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by
  sorry

end B_pow_2048_l373_373632


namespace largest_term_is_181_l373_373864

noncomputable def A_k (k : ℕ) : ℝ := (nat.binom 2000 k) * (0.1 ^ k)

theorem largest_term_is_181 :
    (argmax k in Finset.range 2001, A_k k) = 181 := 
    sorry

end largest_term_is_181_l373_373864


namespace proj_magnitude_l373_373081

variables (u z : E) [InnerProductSpace ℝ E]
variables (h₁ : ∥u∥ = 5)
variables (h₂ : ∥z∥ = 8)
variables (h₃ : ⟪u, z⟫ = 20)

theorem proj_magnitude (h₁ h₂ h₃) : ∥(⟪u, z⟫ / ∥z∥^2) • z∥ = 2.5 := by
  sorry

end proj_magnitude_l373_373081


namespace gcd_max_possible_value_l373_373403

theorem gcd_max_possible_value (x y : ℤ) (h_coprime : Int.gcd x y = 1) : 
  ∃ d, d = Int.gcd (x + 2015 * y) (y + 2015 * x) ∧ d = 4060224 :=
by
  sorry

end gcd_max_possible_value_l373_373403


namespace orange_cost_l373_373792

theorem orange_cost
    (family_size : ℕ) 
    (planned_spending : ℝ) 
    (saved_percentage : ℝ)
    (saved_amount : ℝ)
    (price_per_orange : ℝ) : 
    family_size = 4 → 
    planned_spending = 15 → 
    saved_percentage = 0.40 → 
    saved_amount = planned_spending * saved_percentage → 
    price_per_orange = saved_amount / family_size → 
    price_per_orange = 1.50 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h4] at h5
  sorry

end orange_cost_l373_373792


namespace problem_solution_l373_373141
noncomputable theory
open classical

def is_arithmetic_sequence (a : ℕ → ℚ) :=
∃ a₁ d : ℚ, ∀ n, a n = a₁ + n * d

def satisfies_equation (x : ℚ) := x^2 - 3*x - 5 = 0

theorem problem_solution :
  ∀ (a : ℕ → ℚ),
  is_arithmetic_sequence a →
  satisfies_equation (a 3) →
  satisfies_equation (a 11) →
  a 5 + a 6 + a 10 = 9 / 2 :=
by
  intros a h_arith h_eq1 h_eq2
  sorry

end problem_solution_l373_373141


namespace max_value_f_value_of_f_at_alpha_l373_373067

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * Real.sin x

theorem max_value_f :
  (∀ x, f x ≤ 3)
  ∧ (∃ x, f x = 3)
  ∧ {x : ℝ | ∃ k : ℤ, x = (π / 3) + 2 * k * π} = {x : ℝ | ∃ k : ℤ, x = (π / 3) + 2 * k * π} :=
sorry

theorem value_of_f_at_alpha {α : ℝ} (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end max_value_f_value_of_f_at_alpha_l373_373067


namespace students_scoring_no_less_than_120_l373_373127

/-- Normal distribution with mean 100 and variance σ^2,
    1600 students in total, and three-quarters scored between 80 and 120 points.
    Prove approximately how many students scored no less than 120 points. -/
theorem students_scoring_no_less_than_120 (σ : ℝ) :
  let total_students := 1600 in
  let mean := 100 in
  let variance := σ^2 in
  let scores_are_normal := true in -- placeholder for normal distribution assumption
  let students_between_80_120 := 3 / 4 * total_students in
  let students_no_less_than_120 := 1 / 8 * total_students in
  students_no_less_than_120 = 200 :=
by
  sorry

end students_scoring_no_less_than_120_l373_373127


namespace minimum_K_l373_373642

noncomputable def f (x : ℝ) : ℝ := 2 - x - 1 / Real.exp(x)

def f_k (x : ℝ) (K : ℝ) : ℝ :=
  if f(x) ≤ K then f(x) else K

def condition (K : ℝ) : Prop :=
  ∀ x : ℝ, f_k x K = f x

theorem minimum_K : ∃ K : ℝ, K = 1 ∧ condition K :=
begin
  use 1,
  split,
  { refl },
  {
    intro x,
    suffices H : f(x) ≤ 1,
    { dsimp [f_k, H] },
    sorry
  }
end

end minimum_K_l373_373642


namespace min_value_expression_l373_373436

theorem min_value_expression (x y a : ℝ) 
  (h1 : (x-3)^3 + 2016*(x-3) = a)
  (h2 : (2*y-3)^3 + 2016*(2*y-3) = -a) :
  x^2 + 4*y^2 + 4*x ≥ 28 :=
sorry

# Test instance showing an example of the expressions
example : min_value_expression x y a h1 h2 :=
begin
  sorry
end

end min_value_expression_l373_373436


namespace tangent_line_at_point_l373_373696

noncomputable def curve (x : ℝ) := Real.exp x + 2 * x
noncomputable def tangent_equation (x : ℝ) := 3 * x + 1
noncomputable def tangent_point : ℝ × ℝ := (0, 1)

theorem tangent_line_at_point :
  ∃ m b : ℝ, (∀ x : ℝ, tangent_equation x = m * x + b) ∧ m = 3 ∧ b = 1 ∧
  (∀ (x y : ℝ), y = curve x → y - 1 = m * (x - 0)) :=
begin
  sorry
end

end tangent_line_at_point_l373_373696


namespace range_of_m_l373_373071

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 
  if x > 1 then log x + x 
  else 2 * x^2 - m * x + m / 2

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 
  f x m - m

-- Define the conditions for having three zeros
def has_three_zeros (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), g x₁ m = 0 ∧ g x₂ m = 0 ∧ g x₃ m = 0 ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃

theorem range_of_m (m : ℝ) : has_three_zeros m ↔ (1 < m ∧ m ≤ 4 / 3) :=
sorry

end range_of_m_l373_373071


namespace graph_shift_l373_373723

-- Definitions from conditions
def f (x : ℝ) : ℝ := 3 * sin (2 * x)
def g (x : ℝ) : ℝ := 3 * sin (2 * x - π / 8)

-- The statement that asserts the transformation is correct
theorem graph_shift :
  ∀ x, g x = f (x - π / 16) :=
by
  sorry

end graph_shift_l373_373723


namespace parallel_midpoints_l373_373726

theorem parallel_midpoints (A B C M N P Q : Type*)
  [Geometry A B C M N P Q] -- This should specify the geometric relationships and properties
  (h1 : right_angle C) -- Triangle ABC is right-angled at C
  (h2 : angle_bisectors A M B N) -- AM and BN are internal angle bisectors
  (h3 : altitude_intersects CH P Q) -- AM and BN intersect altitude CH at points P and Q
  : parallel (midpoint Q N) (midpoint P M) (A B) := sorry

end parallel_midpoints_l373_373726


namespace remainder_2468135792_div_101_l373_373736

theorem remainder_2468135792_div_101 : (2468135792 % 101) = 52 := 
by 
  -- Conditions provided in the problem
  have decompose_num : 2468135792 = 24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92, 
  from sorry,
  
  -- Assert large powers of 10 modulo properties
  have ten_to_pow2 : (10^2 - 1) % 101 = 0, from sorry,
  have ten_to_pow4 : (10^4 - 1) % 101 = 0, from sorry,
  have ten_to_pow6 : (10^6 - 1) % 101 = 0, from sorry,
  have ten_to_pow8 : (10^8 - 1) % 101 = 0, from sorry,
  
  -- Summing coefficients
  have coefficients_sum : 24 + 68 + 13 + 57 + 92 = 254, from
  by linarith,
  
  -- Calculating modulus
  calc 
    2468135792 % 101
        = (24 * 10^8 + 68 * 10^6 + 13 * 10^4 + 57 * 10^2 + 92) % 101 : by rw decompose_num
    ... = (24 + 68 + 13 + 57 + 92) % 101 : by sorry
    ... = 254 % 101 : by rw coefficients_sum
    ... = 52 : by norm_num,

  sorry

end remainder_2468135792_div_101_l373_373736


namespace rectangle_area_increase_l373_373320

theorem rectangle_area_increase (l w : ℝ) :
    let l_new := 1.3 * l
    let w_new := 1.15 * w
    let A_old := l * w
    let A_new := l_new * w_new
    (A_new = 1.495 * A_old) → (A_new / A_old - 1) * 100 = 49.5 := by
  intros
  sorry

end rectangle_area_increase_l373_373320


namespace people_going_to_movie_l373_373301

variable (people_per_car : ℕ) (number_of_cars : ℕ)

theorem people_going_to_movie (h1 : people_per_car = 6) (h2 : number_of_cars = 18) : 
    (people_per_car * number_of_cars) = 108 := 
by
  sorry

end people_going_to_movie_l373_373301


namespace sum_first_99_terms_l373_373427

noncomputable def sequence (a : ℕ → ℚ) (a_n : ℚ): Prop :=
  ∀ n, a (n + 1) = (a n - 1) / (a n + 1)

theorem sum_first_99_terms (a : ℕ → ℚ) (ha : sequence a) 
  (h_initial : a 10 = 1/3) :
  (∑ i in finset.range 99, a i) = -26 :=
sorry

end sum_first_99_terms_l373_373427


namespace radius_of_incircle_proof_l373_373774

noncomputable def radius_of_incircle {A B C D : Type*} 
  [convex_quadrilateral A B C D] 
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (AD : ℝ) 
  (angle_ABC_right : angle ABC = 90) 
  (has_incircle : ∃ (r : ℝ), r > 0 ∧ is_incircle_of_quadrilateral r A B C D) : ℝ :=
  1

theorem radius_of_incircle_proof (A B C D : Type*) 
  [convex_quadrilateral A B C D] 
  (AB BC CD AD : ℝ) 
  (angle_ABC_right : angle ABC = 90)
  (has_incircle : ∃ (r : ℝ), r > 0 ∧ is_incircle_of_quadrilateral r A B C D) :
  radius_of_incircle AB BC CD AD angle_ABC_right has_incircle = 1 :=
by sorry

end radius_of_incircle_proof_l373_373774


namespace median_eq_BC_altitude_eq_BC_l373_373042

def point := (ℝ × ℝ)
def A : point := (-5, 0)
def B : point := (4, -4)
def C : point := (0, 2)

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def slope (p1 p2 : point) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

noncomputable def line_through (p : point) (m : ℝ) : (ℝ × ℝ × ℝ) :=
  let b := p.2 - m * p.1 in
  (-m, 1, b)

theorem median_eq_BC :
  let D : point := midpoint B C in
  let m_AD := slope A D in
  let (a, b, c) := line_through A m_AD in
  a * -5 + b * 0 + c = 0 → a * 4 + b * -4 + c = 0 →
  a = 1 ∧ b = 7 ∧ c = 5 := sorry

theorem altitude_eq_BC :
  let m_BC := slope B C in
  let m_AE := -1 / m_BC in
  let (a, b, c) := line_through A m_AE in
  a * -5 + b * 0 + c = 0 → a * 0 + b * 2 + c = 0 →
  a = 2 ∧ b = -3 ∧ c = 10 := sorry

end median_eq_BC_altitude_eq_BC_l373_373042
