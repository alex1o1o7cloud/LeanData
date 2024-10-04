import Mathlib

namespace find_positive_value_of_X_l464_464179

-- define the relation X # Y
def rel (X Y : ℝ) : ℝ := X^2 + Y^2

theorem find_positive_value_of_X (X : ℝ) (h : rel X 7 = 250) : X = Real.sqrt 201 :=
by
  sorry

end find_positive_value_of_X_l464_464179


namespace cos_of_angle_through_point_P_l464_464927

noncomputable def cos_alpha_through_point : Prop :=
  let P := (-1: ℝ, 2: ℝ) in
  let r := real.sqrt (P.1^2 + P.2^2) in
  ∃ α : ℝ, cos α = P.1 / r

theorem cos_of_angle_through_point_P : cos_alpha_through_point :=
  by
    let P := (-1: ℝ, 2: ℝ)
    let r := real.sqrt (P.1^2 + P.2^2)
    use P.1 / r
    sorry

end cos_of_angle_through_point_P_l464_464927


namespace average_donation_is_integer_l464_464792

variable (num_classes : ℕ) (students_per_class : ℕ) (num_teachers : ℕ) (total_donation : ℕ)

def valid_students (n : ℕ) : Prop := 30 < n ∧ n ≤ 45

theorem average_donation_is_integer (h_classes : num_classes = 14)
                                    (h_teachers : num_teachers = 35)
                                    (h_donation : total_donation = 1995)
                                    (h_students_per_class : valid_students students_per_class)
                                    (h_total_people : ∃ n, 
                                      n = num_teachers + num_classes * students_per_class ∧ 30 < students_per_class ∧ students_per_class ≤ 45) :
  total_donation % (num_teachers + num_classes * students_per_class) = 0 ∧ 
  total_donation / (num_teachers + num_classes * students_per_class) = 3 := 
sorry

end average_donation_is_integer_l464_464792


namespace plane_through_points_l464_464879

-- Define the points in three-dimensional space
structure Point3D where
  x : ℤ
  y : ℤ
  z : ℤ

noncomputable def a : Point3D := ⟨0, 1, 2⟩
noncomputable def b : Point3D := ⟨1, 1, 3⟩
noncomputable def c : Point3D := ⟨2, 0, 4⟩

-- Function to define the equation of a plane
def plane_eqn (A B C D : ℤ) (P : Point3D) : Prop :=
  A * P.x + B * P.y + C * P.z + D = 0

-- Lean statement proving the plane equation
theorem plane_through_points :
  ∃ (A B C D : ℤ), 
  A > 0 ∧ 
  Int.gcd (A.natAbs) (B.natAbs) (C.natAbs) (D.natAbs) = 1 ∧ 
  plane_eqn A B C D a ∧
  plane_eqn A B C D b ∧ 
  plane_eqn A B C D c ∧ 
  (A, B, C, D) = (2, 0, 1, -2) :=
by
  sorry

end plane_through_points_l464_464879


namespace circle_equation_max_area_l464_464882

theorem circle_equation_max_area {m : ℝ} : 
  let center := (2 : ℝ, -3 : ℝ)
  let line := λ m x y, 2 * m * x - y - 2 * m - 1 = 0 
  in
  ∃ r, r^2 = 5 ∧ (∀ ⦃x y : ℝ⦄, line m x y → (x - 2)^2 + (y + 3)^2 = r^2) :=
by
  let center := (2 : ℝ, -3 : ℝ)
  have line := λ m x y, 2 * m * x - y - 2 * m - 1 = 0
  existsi 5
  sorry

end circle_equation_max_area_l464_464882


namespace ls_parallel_pq_l464_464603

-- Declare the points and lines
variables {A P C M N Q L S : Type*}

-- Assume given conditions as definitions
def intersection (l1 l2 : set (Type*)) : Type* := sorry -- Define intersection point of l1 and l2 (details omitted for simplicity)

-- Define L as intersection of lines AP and CM
def L : Type* := intersection (set_of (λ x, ∃ L, ∃ P, ∃ C, ∃ M, L = x ∧ x ∈ AP ∧ x ∈ CM)) (λ x, x ∈ AP ∧ x ∈ CM)

-- Define S as intersection of lines AN and CQ
def S : Type* := intersection (set_of (λ x, ∃ S, ∃ N, ∃ C, ∃ Q, S = x ∧ x ∈ AN ∧ x ∈ CQ)) (λ x, x ∈ AN ∧ x ∈ CQ)

-- Statement to prove LS is parallel to PQ
theorem ls_parallel_pq (AP CM AN CQ PQ : set (Type*)) (L : Type*) (S : Type*) :
    L = intersection (AP) (CM) ∧ S = intersection (AN) (CQ) → is_parallel LS PQ :=
by
  sorry -- Proof omitted

end ls_parallel_pq_l464_464603


namespace solve_for_x_l464_464718

theorem solve_for_x (x : ℝ) : 5^(3 * x) = sqrt 125 → x = 1 / 2 := by
  sorry

end solve_for_x_l464_464718


namespace eval_expr_l464_464421

noncomputable def expr1 : ℝ := (4/9)^(1/2)
noncomputable def expr2 : ℝ := (sqrt 2 / 2)^0
noncomputable def expr3 : ℝ := (27/64)^(-1/3)

theorem eval_expr :
  expr1 - expr2 + expr3 = 1 := by
  sorry

end eval_expr_l464_464421


namespace ls_parallel_pq_l464_464602

-- Declare the points and lines
variables {A P C M N Q L S : Type*}

-- Assume given conditions as definitions
def intersection (l1 l2 : set (Type*)) : Type* := sorry -- Define intersection point of l1 and l2 (details omitted for simplicity)

-- Define L as intersection of lines AP and CM
def L : Type* := intersection (set_of (λ x, ∃ L, ∃ P, ∃ C, ∃ M, L = x ∧ x ∈ AP ∧ x ∈ CM)) (λ x, x ∈ AP ∧ x ∈ CM)

-- Define S as intersection of lines AN and CQ
def S : Type* := intersection (set_of (λ x, ∃ S, ∃ N, ∃ C, ∃ Q, S = x ∧ x ∈ AN ∧ x ∈ CQ)) (λ x, x ∈ AN ∧ x ∈ CQ)

-- Statement to prove LS is parallel to PQ
theorem ls_parallel_pq (AP CM AN CQ PQ : set (Type*)) (L : Type*) (S : Type*) :
    L = intersection (AP) (CM) ∧ S = intersection (AN) (CQ) → is_parallel LS PQ :=
by
  sorry -- Proof omitted

end ls_parallel_pq_l464_464602


namespace max_distance_mn_solved_a_l464_464926

-- Define the equations and conditions
def circle (a : ℝ) : set (ℝ × ℝ) :=
  {p | let ⟨x, y⟩ := p in x^2 + y^2 = a * y}

def line (t : ℝ) : ℝ × ℝ :=
  (-3/5 * t + 2, 4/5 * t)

def chord_length_eq_radius (a : ℝ) : Prop :=
  ∃ (t1 t2 : ℝ), line t1 ∈ circle a ∧ line t2 ∈ circle a ∧ 
  (real.sqrt ( (line t1).1 - (line t2).1)^2 + ( (line t1).2 - (line t2).2)^2) = real.sqrt 3 * (abs (a / 2))

-- Proof statements
theorem max_distance_mn (a : ℝ) (t : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) :
  a = 2 → ∃ M, M = (2, 0) ∧ ∀ N ∈ circle a, |dist M N| ≤ real.sqrt 8 + 2 := sorry

theorem solved_a (a : ℝ) :
  chord_length_eq_radius a → a = 32 ∨ a = 32 / 11 := sorry

end max_distance_mn_solved_a_l464_464926


namespace valid_pairs_l464_464671

noncomputable def S (r : ℕ) (x y z : ℝ) : ℝ := x^r + y^r + z^r

theorem valid_pairs (x y z : ℝ) (h : x + y + z = 0):
  (∀ m n, 
    (m, n) ∈ [(2, 3), (3, 2), (2, 5), (5, 2)] ↔
    (S (m + n) x y z / (m + n)) = (S m x y z / m) * (S n x y z / n)) := 
begin
  sorry
end

end valid_pairs_l464_464671


namespace find_locus_of_H_l464_464189

noncomputable def locus_of_H (O A B : Point) (p : Plane) (hO : O ∉ p) (hA : A ∈ p) 
(hB : B ∈ p) (hOB : B = foot_of_perpendicular O p) : Set Point :=
{ H : Point | ∃ ℓ : Line, ℓ ∈ p ∧ A ∈ ℓ ∧ (H = foot_of_perpendicular O ℓ) }

theorem find_locus_of_H
  (O A B : Point) (p : Plane) (hO : O ∉ p) (hA : A ∈ p) (hB : B ∈ p)
  (hOB : B = foot_of_perpendicular O p) :
  locus_of_H O A B p hO hA hB hOB = { K : Point | K ∈ circle_with_diameter A B } :=
sorry -- proof to be filled in

end find_locus_of_H_l464_464189


namespace probability_reverse_order_probability_second_to_bottom_l464_464381

noncomputable def prob_reverse_order : ℚ := 1/8
noncomputable def prob_second_to_bottom : ℚ := 1/8

theorem probability_reverse_order (n: ℕ) (h : n = 4) 
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_reverse_order = 1/8 :=
by sorry

theorem probability_second_to_bottom (n: ℕ) (h : n = 4)
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_second_to_bottom = 1/8 :=
by sorry

end probability_reverse_order_probability_second_to_bottom_l464_464381


namespace polynomial_is_even_l464_464585

variable {R : Type*} [OrderedRing R] [CharZero R]

def polynomial_symmetric_about_origin (P : R[X]) (a b : R) (h_deg : degree P = 6) (h_pos : 0 < a ∧ a < b)
                                        (h_pa : P.eval a = P.eval (-a)) (h_pb : P.eval b = P.eval (-b)) 
                                        (h_prime_zero : P.derivative.eval 0 = 0) : Prop :=
  ∀ x : R, P.eval x = P.eval (-x)

theorem polynomial_is_even (P : R[X]) (a b : R) (h_deg : degree P = 6) (h_pos : 0 < a ∧ a < b)
                           (h_pa : P.eval a = P.eval (-a)) (h_pb : P.eval b = P.eval (-b)) 
                           (h_prime_zero : P.derivative.eval 0 = 0) :
  polynomial_symmetric_about_origin P a b h_deg h_pos h_pa h_pb h_prime_zero := sorry

end polynomial_is_even_l464_464585


namespace min_L_shaped_tiles_l464_464767

theorem min_L_shaped_tiles (n : ℕ) (h₁ : n = 8) :
  ∃ k : ℕ, k ≥ 11 ∧ (∀ (grid : fin n × fin n),
        (∑ (i : fin 16), (coverage grid i) ≥ 32) → coverage_by_L_tiles grid k) :=
sorry

end min_L_shaped_tiles_l464_464767


namespace cylinder_surface_area_l464_464804

theorem cylinder_surface_area (side : ℝ) (h : ℝ) (r : ℝ) : 
  side = 2 ∧ h = side ∧ r = side → 
  (2 * Real.pi * r^2 + 2 * Real.pi * r * h) = 16 * Real.pi := 
by
  intro h
  sorry

end cylinder_surface_area_l464_464804


namespace base_conversion_sum_l464_464258

theorem base_conversion_sum :
  let n1 := 29
  let n2 := 45
  let n3 := 131 -- 29 in base 4 is 131
  let n4 := 140 -- 45 in base 5 is 140
  nat.ofDigits 5 [2, 4, 4] = (nat.ofDigits 4 [1, 3, 1] + nat.ofDigits 5 [1, 4, 0])
:=
by {
  sorry,
}

end base_conversion_sum_l464_464258


namespace number_of_integers_l464_464513

theorem number_of_integers (n : ℤ) : 150 < n ∧ n < 250 ∧ n % 7 = n % 9 → (finset.range 251).filter (λ n, 150 < n ∧ n % 7 = n % 9).card = 7 := 
by
  sorry

end number_of_integers_l464_464513


namespace avg_combined_is_2a_plus_3b_l464_464062

variables {x1 x2 x3 y1 y2 y3 a b : ℝ}

-- Given conditions
def avg_x_is_a (x1 x2 x3 a : ℝ) : Prop := (x1 + x2 + x3) / 3 = a
def avg_y_is_b (y1 y2 y3 b : ℝ) : Prop := (y1 + y2 + y3) / 3 = b

-- The statement to be proved
theorem avg_combined_is_2a_plus_3b
    (hx : avg_x_is_a x1 x2 x3 a) 
    (hy : avg_y_is_b y1 y2 y3 b) :
    ((2 * x1 + 3 * y1) + (2 * x2 + 3 * y2) + (2 * x3 + 3 * y3)) / 3 = 2 * a + 3 * b := 
by
  sorry

end avg_combined_is_2a_plus_3b_l464_464062


namespace area_enclosed_by_curves_l464_464260

noncomputable def areaBetweenCurves : ℝ :=
  ∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))

theorem area_enclosed_by_curves :
  (∫ x in (0 : ℝ)..(4 : ℝ), (x - (x^2 - 3*x))) = (32 / 3 : ℝ) :=
by
  sorry

end area_enclosed_by_curves_l464_464260


namespace cos_diff_formula_example_l464_464294

theorem cos_diff_formula_example : 
  cos (43 * real.pi / 180) * cos (13 * real.pi / 180)
  + sin (43 * real.pi / 180) * sin (13 * real.pi / 180) 
  = cos (30 * real.pi / 180) :=
sorry

end cos_diff_formula_example_l464_464294


namespace find_side_b_in_triangle_l464_464977

theorem find_side_b_in_triangle 
  (A B : ℝ) (a : ℝ)
  (h_cosA : Real.cos A = -1/2)
  (h_B : B = Real.pi / 4)
  (h_a : a = 3) :
  ∃ b, b = Real.sqrt 6 :=
by
  sorry

end find_side_b_in_triangle_l464_464977


namespace average_points_per_player_l464_464584

theorem average_points_per_player (Lefty_points Righty_points OtherTeammate_points : ℕ)
  (hL : Lefty_points = 20)
  (hR : Righty_points = Lefty_points / 2)
  (hO : OtherTeammate_points = 6 * Righty_points) :
  (Lefty_points + Righty_points + OtherTeammate_points) / 3 = 30 :=
by
  sorry

end average_points_per_player_l464_464584


namespace ls_parallel_pq_l464_464649

-- Definitions for points L and S, and their parallelism
variables {A P C M N Q L S : Type}

-- Assume points and lines for the geometric setup
variables (A P C M N Q L S : Type)

-- Assume the intersections and the required parallelism conditions
variables (hL : ∃ (AP CM : Set (Set L)), L ∈ AP ∩ CM)
variables (hS : ∃ (AN CQ : Set (Set S)), S ∈ AN ∩ CQ)

-- The goal: LS is parallel to PQ
theorem ls_parallel_pq : LS ∥ PQ :=
sorry

end ls_parallel_pq_l464_464649


namespace sum_bn_first_10_terms_l464_464904

-- Define the geometric sequence and related conditions
variable (a : ℕ → ℝ) (b : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Assume conditions involving a and S
axiom a_geom_seq : ∀ n, a (n + 1) / a n = a 2 / a 1
axiom S_geom_seq : ∀ n, S n = (a 1 * (1 - (a (2) / a 1) ^ n)) / (1 - a 2 / a 1)

-- Given conditions
axiom a4_minus_a1 : a 4 - a 1 = 78
axiom S3 : S 3 = 39

-- Definition of bn
def bn (n : ℕ) := log 3 (a n)

-- Problem statement: sum of the first 10 terms of sequence bn
theorem sum_bn_first_10_terms : (Finset.range 10).sum b = 55 := sorry

end sum_bn_first_10_terms_l464_464904


namespace find_m_n_range_of_k_l464_464903

theorem find_m_n (f : ℝ → ℝ) (n m : ℝ) (h₁ : ∀ x, f x = (-(-2: ℝ)^x + n) / ((2: ℝ)^(x + 1) + m)) (h₂ : ∀ x, f (-x) = -f x) : 
  n = 1 ∧ m = 2 :=
sorry

theorem range_of_k (f : ℝ → ℝ) (k : ℝ) 
  (h₁ : ∀ x, f x = (-(-2: ℝ)^x + 1) / ((2: ℝ)^(x + 1) + 2))
  (h₂ : ∀ t ∈ (1: ℝ, 2: ℝ), f (t^2 - 2*t) + f (2*t^2 - k) < 0) : 
  k ≤ 1 :=
sorry

end find_m_n_range_of_k_l464_464903


namespace cosine_between_vectors_l464_464947

noncomputable theory

open Real

variables {a b : ℝ^3}

def vector_norm (v : ℝ^3) : ℝ := sqrt (v.dot v)

def cosine_angle (u v : ℝ^3) : ℝ := (u.dot v) / (vector_norm u * vector_norm v)

theorem cosine_between_vectors :
  a ≠ 0 ∧ b ≠ 0 ∧ vector_norm a = vector_norm b ∧ vector_norm a = vector_norm (a + b) →
  cosine_angle a (2 * a - b) = (5 * sqrt 7) / 14 :=
by
  intros h,
  sorry

end cosine_between_vectors_l464_464947


namespace jennifer_score_l464_464988

theorem jennifer_score 
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (unanswered_questions : ℕ)
  (points_per_correct : ℤ)
  (points_deduction_incorrect : ℤ)
  (points_per_unanswered : ℤ)
  (h_total : total_questions = 30)
  (h_correct : correct_answers = 15)
  (h_incorrect : incorrect_answers = 10)
  (h_unanswered : unanswered_questions = 5)
  (h_points_correct : points_per_correct = 2)
  (h_deduction_incorrect : points_deduction_incorrect = -1)
  (h_points_unanswered : points_per_unanswered = 0) : 
  ∃ (score : ℤ), score = (correct_answers * points_per_correct 
                          + incorrect_answers * points_deduction_incorrect 
                          + unanswered_questions * points_per_unanswered) 
                        ∧ score = 20 := 
by
  sorry

end jennifer_score_l464_464988


namespace yellow_balls_calculation_l464_464545

variable (total_balls : ℕ)
variable (red_fraction : ℚ)
variable (blue_fraction : ℚ)
variable (green_fraction : ℚ)
variable (orange_fraction : ℚ)

def red_balls := (red_fraction * total_balls).toNat
def remaining_after_red := total_balls - red_balls
def blue_balls := (blue_fraction * remaining_after_red).toNat
def remaining_after_blue := remaining_after_red - blue_balls
def green_balls := (green_fraction * remaining_after_blue).toNat
def remaining_after_green := remaining_after_blue - green_balls
def orange_balls := (orange_fraction * remaining_after_green).toNat
def remaining_after_orange := remaining_after_green - orange_balls

theorem yellow_balls_calculation :
  yellow_balls total_balls red_fraction blue_fraction green_fraction orange_fraction = 546 :=
by
  have total_balls := 1500
  have red_fraction := 2/7
  have blue_fraction := 3/11
  have green_fraction := 1/5
  have orange_fraction := 1/8
  let red_balls := (red_fraction * total_balls).toNat
  let remaining_after_red := total_balls - red_balls
  let blue_balls := (blue_fraction * remaining_after_red).toNat
  let remaining_after_blue := remaining_after_red - blue_balls
  let green_balls := (green_fraction * remaining_after_blue).toNat
  let remaining_after_green := remaining_after_blue - green_balls
  let orange_balls := (orange_fraction * remaining_after_green).toNat
  let remaining_after_orange := remaining_after_green - orange_balls
  exact Eq.refl 546

end yellow_balls_calculation_l464_464545


namespace find_q_l464_464533

theorem find_q (A : ℝ) (p q : ℝ) (h1 : sin A = p / 5) (h2 : (cos A) / (tan A) = q / 15) (h3 : p = 3) : q = 16 :=
by
  sorry

end find_q_l464_464533


namespace three_digit_numbers_divisible_by_17_l464_464097

theorem three_digit_numbers_divisible_by_17 :
  ∃ n : ℕ, n = 53 ∧ ∀ k : ℕ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ↔ k ∈ finset.Icc 6 58 :=
by
  sorry

end three_digit_numbers_divisible_by_17_l464_464097


namespace wheel_total_distance_l464_464283

theorem wheel_total_distance (r : ℝ) (revs : ℕ) (decrease : ℝ) :
  r = 14.6 → revs = 100 → decrease = 0.12 →
  let C := 2 * Real.pi * r in
  let EC := C * (1 - decrease) in
  let TD := EC * revs in
  TD = 8076.8512 :=
by
  intros hr hrev hdec
  simp only [hr, hrev, hdec]
  let C := 2 * Real.pi * 14.6
  let EC := C * 0.88
  let TD := EC * 100
  norm_num at C EC TD
  sorry

end wheel_total_distance_l464_464283


namespace probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464398

-- Definitions based on conditions
def bags_in_truck : ℕ := 4

-- Assume random_choice has probability 0.5 for selection
def random_choice : ℕ → ℕ → ℝ := λ a b, 0.5

-- Bag positions and movements
def initial_position (i : ℕ) : ℕ := i

def final_position (i : ℕ) : ℕ := 4 - i

-- Events
def end_up_in_reverse_order (n : ℕ) : Prop := 
  ∀ i < n, final_position i = n - 1 - i

def second_from_bottom_ends_up_bottom (n : ℕ) : Prop :=
  final_position 1 = 0

-- Probabilities calculation
noncomputable def probability_of_reverse_order (n : ℕ) : ℝ := (1/2)^(n - 1)
noncomputable def probability_of_second_from_bottom_as_bottom (n : ℕ) : ℝ := (1/2)^(n - 1)

-- Theorem to prove the statement
theorem probability_reverse_order :
  probability_of_reverse_order bags_in_truck = 1 / 8 :=
by sorry

theorem probability_second_from_bottom_as_bottom :
  probability_of_second_from_bottom_as_bottom bags_in_truck = 1 / 8 :=
by sorry

end probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464398


namespace probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464396

-- Definitions based on conditions
def bags_in_truck : ℕ := 4

-- Assume random_choice has probability 0.5 for selection
def random_choice : ℕ → ℕ → ℝ := λ a b, 0.5

-- Bag positions and movements
def initial_position (i : ℕ) : ℕ := i

def final_position (i : ℕ) : ℕ := 4 - i

-- Events
def end_up_in_reverse_order (n : ℕ) : Prop := 
  ∀ i < n, final_position i = n - 1 - i

def second_from_bottom_ends_up_bottom (n : ℕ) : Prop :=
  final_position 1 = 0

-- Probabilities calculation
noncomputable def probability_of_reverse_order (n : ℕ) : ℝ := (1/2)^(n - 1)
noncomputable def probability_of_second_from_bottom_as_bottom (n : ℕ) : ℝ := (1/2)^(n - 1)

-- Theorem to prove the statement
theorem probability_reverse_order :
  probability_of_reverse_order bags_in_truck = 1 / 8 :=
by sorry

theorem probability_second_from_bottom_as_bottom :
  probability_of_second_from_bottom_as_bottom bags_in_truck = 1 / 8 :=
by sorry

end probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464396


namespace minimum_value_l464_464060

theorem minimum_value (x y z : ℝ) (h : 2 * x - 3 * y + z = 3) :
  ∃ y_min, y_min = -2 / 7 ∧ x = 6 / 7 ∧ (x^2 + (y - 1)^2 + z^2) = 18 / 7 :=
by
  sorry

end minimum_value_l464_464060


namespace distribute_indistinguishable_balls_l464_464105

theorem distribute_indistinguishable_balls : 
  ∃ n : ℕ, (n = 104 ∧ 
  ∀ (b : fin 7 → fin 4), 
  ∃ (c : fin 4 → ℕ), 
  (∀ (i : fin 4), ∃ (m : list ℕ), 
    list.length m = 7 ∧ 
    ∀ (j : fin 7), (list.nthLe m j sorry) = (ite (b j = i) 1 0) ∧ 
    list.sum m = 7 ∧ 
    list.count 1 m = 4))) := 
  sorry

end distribute_indistinguishable_balls_l464_464105


namespace paths_from_A_to_B_l464_464799

theorem paths_from_A_to_B :
  let width := 7
  let height := 6
  ∃ n : ℕ, n = Nat.choose (width + height) height ∧ n = 1716 :=
by
  let width := 7
  let height := 6
  let total_steps := width + height
  let paths := Nat.choose total_steps height
  existsi paths
  split
  {
    refl
  }
  {
    have : paths = 1716 := by
    {
      calc
        paths = Nat.choose 13 6 : by simp only [width, height, total_steps]
        ... = 1716 : by sorry
    }
    exact this
  }

end paths_from_A_to_B_l464_464799


namespace probability_point_on_hyperbola_l464_464230

theorem probability_point_on_hyperbola :
  let S := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) },
      Hyperbola := { (x, y) | y = 6 / x },
      Favourable := S ∩ Hyperbola
  in
    Favourable.card / S.card = 1 / 3 :=
by
  sorry

end probability_point_on_hyperbola_l464_464230


namespace probability_point_on_hyperbola_l464_464218

-- Define the problem conditions
def number_set := {1, 2, 3}
def point_on_hyperbola (x y : ℝ) : Prop := y = 6 / x

-- Formalize the problem statement
theorem probability_point_on_hyperbola :
  let combinations := ({(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} : set (ℝ × ℝ)) in
  let on_hyperbola := set.filter (λ p : ℝ × ℝ, point_on_hyperbola p.1 p.2) combinations in
  fintype.card on_hyperbola / fintype.card combinations = 1 / 3 :=
by sorry

end probability_point_on_hyperbola_l464_464218


namespace general_formula_T_2017_l464_464047

noncomputable def sequence_formula : ℕ → ℝ
  | n => (1 / 3) ^ n

theorem general_formula (a : ℕ → ℝ) (h : ∀ n, (a (n + 1))^2 = a n * a (n + 2))
    (h1 : a 1 = 1 / 3) (h2 : a 4 = 1 / 81) : ∀ n, a n = sequence_formula n := 
sorry

noncomputable def T (n : ℕ) : ℝ :=
∑ k in finset.range n, 1 / (-(k * (k + 1)) / 2)

theorem T_2017 : T 2017 = -2017 / 1009 := 
sorry

end general_formula_T_2017_l464_464047


namespace ratio_DE_AC_l464_464568

theorem ratio_DE_AC (A B C D E : Type) 
  [has_area_ABC : has_area A B C 8] 
  [has_area_DEC : has_area D E C 2] 
  [line_DE_parallel_AC : parallel D E A C] :
  DE / AC = 1 / 2 := 
sorry

end ratio_DE_AC_l464_464568


namespace median_of_list_l464_464325

theorem median_of_list : 
  let list := (list.range 1500).map (λ n, n + 2) ++ (list.range 1500).map (λ n, (n + 2) ^ 2) in
  list.length = 3000 →
  (list.nth (1500 - 1)).get_or_else 0 = 1501 →
  (list.nth 1500).get_or_else 0 = 1521 →
  (list.sort.nth (list.length / 2 - 1)).get_or_else 0 + (list.sort.nth (list.length / 2)).get_or_else 0) / 2 = 1511 :=
by
  sorry

end median_of_list_l464_464325


namespace range_of_x0_l464_464915

theorem range_of_x0 (x_0 : ℝ) : 
  (∃ (N : ℝ × ℝ), N.1 ^ 2 + N.2 ^ 2 = 1 ∧ ∃ (M : ℝ × ℝ) (O : ℝ × ℝ), M = (x_0, 1) ∧ O = (0, 0) ∧ 
  ∃ angle_OMN : ℝ, angle_OMN = 30 ∧ ∃ angle_OMT : ℝ, angle_OMT = atan (1 / x_0.abs) ∧ 
  angle_OMN ≤ angle_OMT) ↔ x_0 ∈ set.Icc (-real.sqrt 3) (real.sqrt 3) := by
  sorry

end range_of_x0_l464_464915


namespace perimeter_of_square_l464_464728

theorem perimeter_of_square (s : ℝ) (area : ℝ) (h : area = s * s) (h_area : area = 200) : 
  4 * s = 40 * Real.sqrt 2 :=
by
  have h_s : s = Real.sqrt 200 := by
    rw [←Real.sqrt_mul_self_eq_iff s, h_area]
    norm_num
  rw [h_s]
  norm_num
  sorry

end perimeter_of_square_l464_464728


namespace length_of_major_axis_is_five_l464_464359

-- Problem conditions as Lean definitions:

def cylinder_radius : ℝ := 2

-- The minor axis of the ellipse is twice the cylinder radius.
def minor_axis (r : ℝ) : ℝ := 2 * r

-- The major axis is 25% longer than the minor axis.
def major_axis (minor : ℝ) : ℝ := 1.25 * minor

-- Proof problem statement:
theorem length_of_major_axis_is_five :
  let r := cylinder_radius in
  let minor := minor_axis r in
  let major := major_axis minor in
  major = 5 :=
by
  sorry

end length_of_major_axis_is_five_l464_464359


namespace locus_of_intersection_point_l464_464460

theorem locus_of_intersection_point 
  {a b : ℝ} (h1 : 0 < a) (h2 : a < b) 
  (l m : Line) (A : Point) (B : Point)
  (hA : A = (a, 0)) (hB : B = (b, 0))
  (intersects_parabola : ∃ p1 p2 p3 p4 : Point, 
    p1.on l ∧ p2.on l ∧ p3.on l ∧ p4.on l ∧ 
    p1.on_parabola ∧ p2.on_parabola ∧ 
    p3.on_parabola ∧ p4.on_parabola ∧ 
    concyclic {p1, p2, p3, p4}) :
  ∀ P : Point, P.on l ∧ P.on m → P.x = (a + b) / 2 := 
by sorry

structure Line : Type :=
(origin : Point)
(direction : Vector)

structure Point : Type :=
(x : ℝ)
(y : ℝ)

structure Vector : Type :=
(dx : ℝ)
(dy : ℝ)

def Point.on (p : Point) (l : Line) : Prop := sorry
def Point.on_parabola (p : Point) : Prop := p.y^2 = p.x
def concyclic (S : set Point) : Prop := sorry

end locus_of_intersection_point_l464_464460


namespace problem_statement_l464_464672

variable {a b c x y z : ℝ}
variable (h1 : 17 * x + b * y + c * z = 0)
variable (h2 : a * x + 29 * y + c * z = 0)
variable (h3 : a * x + b * y + 53 * z = 0)
variable (ha : a ≠ 17)
variable (hx : x ≠ 0)

theorem problem_statement : 
  (a / (a - 17)) + (b / (b - 29)) + (c / (c - 53)) = 1 :=
sorry

end problem_statement_l464_464672


namespace tangent_line_equation_l464_464014

theorem tangent_line_equation (b : ℝ) :
  (∃ (l : ℝ) (m : ℝ) (d : ℝ), l*x + m*y + d = 0 ∧ l = 2 ∧ m = 1 ∧
   circle="(x-1)^2 + y^2 = 5" ∧
   par_line="2x + y + 1 = 0" ∧
   tangent_line="2x + y + b = 0" ∧ 
   (center(1, 0) and radius ∞=(sqrt 5)) ∧
   parallel (line=l=2, par_line) ∧
   tangent (circle, tangent_line))
  → b = 3 ∨ b = -7 :=
begin
  sorry
end

end tangent_line_equation_l464_464014


namespace num_integers_with_properties_l464_464435

theorem num_integers_with_properties : 
  ∃ (count : ℕ), count = 6 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 3400 ∧ (n % 34 = 0) ∧ 
             ((∃ (odd_divisors : ℕ), odd_divisors = (filter (λ d, d % 2 = 1) (n.divisors)) ∧ odd_divisors.length = 2) →
             (count = 6)) :=
begin
  sorry
end

end num_integers_with_properties_l464_464435


namespace find_a_l464_464506

noncomputable def circle1 (x y : ℝ) := x^2 + y^2 + 4 * y = 0

noncomputable def circle2 (x y a : ℝ) := x^2 + y^2 + 2 * (a - 1) * x + 2 * y + a^2 = 0

theorem find_a (a : ℝ) :
  (∀ x y, circle1 x y → circle2 x y a → false) → a = -2 :=
by sorry

end find_a_l464_464506


namespace ls_parallel_pq_l464_464650

-- Definitions for points L and S, and their parallelism
variables {A P C M N Q L S : Type}

-- Assume points and lines for the geometric setup
variables (A P C M N Q L S : Type)

-- Assume the intersections and the required parallelism conditions
variables (hL : ∃ (AP CM : Set (Set L)), L ∈ AP ∩ CM)
variables (hS : ∃ (AN CQ : Set (Set S)), S ∈ AN ∩ CQ)

-- The goal: LS is parallel to PQ
theorem ls_parallel_pq : LS ∥ PQ :=
sorry

end ls_parallel_pq_l464_464650


namespace remainder_sum_from_1_to_12_div_9_l464_464329

-- Define the sum of integers from 1 to 12
def sum_from_1_to_12 : ℕ := (12 * (1 + 12)) / 2

-- Prove that the remainder when sum_from_1_to_12 is divided by 9 is 6
theorem remainder_sum_from_1_to_12_div_9 : (78 % 9) = 6 :=
by {
  -- Explicitly stating our sum calculated earlier
  have sum_eq : sum_from_1_to_12 = 78 := by { unfold sum_from_1_to_12, norm_num },
  -- Therefore statement
  rw ←sum_eq,
  -- Apply remainder calculation
  norm_num,
}

end remainder_sum_from_1_to_12_div_9_l464_464329


namespace total_people_waiting_l464_464837

theorem total_people_waiting 
  (initial_first_line : ℕ := 7)
  (left_first_line : ℕ := 4)
  (joined_first_line : ℕ := 8)
  (initial_second_line : ℕ := 12)
  (left_second_line : ℕ := 3)
  (joined_second_line : ℕ := 10)
  (initial_third_line : ℕ := 15)
  (left_third_line : ℕ := 5)
  (joined_third_line : ℕ := 7) :
  (initial_first_line - left_first_line + joined_first_line) +
  (initial_second_line - left_second_line + joined_second_line) +
  (initial_third_line - left_third_line + joined_third_line) = 47 :=
by
  sorry

end total_people_waiting_l464_464837


namespace parallel_ls_pq_l464_464614

variables {A P C M N Q L S : Type*}
variables [affine_space ℝ (A P C M N Q)]

/- Definitions of the intersection points -/
def is_intersection_point (L : Type*) (AP : set (A P))
  (CM : set (C M)) : Prop := L ∈ AP ∧ L ∈ CM

def is_intersection_point' (S : Type*) (AN : set (A N))
  (CQ : set (C Q)) : Prop := S ∈ AN ∧ S ∈ CQ

/- Main Theorem -/
theorem parallel_ls_pq (AP CM AN CQ : set (A P))
  (PQ : set (P Q)) (L S : Type*)
  (hL : is_intersection_point L AP CM)
  (hS : is_intersection_point' S AN CQ) : L S ∥ P Q :=
sorry

end parallel_ls_pq_l464_464614


namespace parallel_LS_PQ_l464_464630

open_locale big_operators

variables (A P M N C Q L S : Type*)

-- Assume that L is the intersection of lines AP and CM
def is_intersection_L (AP CM : Set (Set Type*)) : Prop :=
  L ∈ AP ∩ CM

-- Assume that S is the intersection of lines AN and CQ
def is_intersection_S (AN CQ : Set (Set Type*)) : Prop :=
  S ∈ AN ∩ CQ

-- Prove that LS ∥ PQ
theorem parallel_LS_PQ 
  (hL : is_intersection_L A P M C L) 
  (hS : is_intersection_S A N C Q S) : 
  parallel (line_through L S) (line_through P Q) :=
sorry

end parallel_LS_PQ_l464_464630


namespace num_bijective_selfmaps_7fixedpoints_l464_464657

open Finset

-- Define the set M
def M : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the condition that a function is bijective
def bijective (f : ℕ → ℕ) : Prop := Function.Bijective f

-- Define the main theorem
theorem num_bijective_selfmaps_7fixedpoints : 
  ∃ f : M → M, bijective f ∧ (card (filter (λ x, f x = x) M) = 7) = 240 :=
by sorry

end num_bijective_selfmaps_7fixedpoints_l464_464657


namespace proof_problem_l464_464078

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 2 * x

def line_l_through_focus (m y : ℝ) : ℝ := m * y + 1/2

def focus : (ℝ × ℝ) := (1/2, 0)

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

variable (A B : ℝ × ℝ)

theorem proof_problem :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  (∃ m : ℝ, line_l_through_focus m A.2 = A.1 ∧ line_l_through_focus m B.2 = B.1) ∧
  dist A focus > dist B focus ∧
  (dist A B = 9) →
  ((dist A focus - dist B focus = 3) ∧
   (dist A focus * dist B focus = 9/4) ∧
   (dist A focus / dist B focus = 2 + sqrt(3))) :=
by
  sorry

end proof_problem_l464_464078


namespace unique_handshakes_l464_464835

theorem unique_handshakes :
  let twins_sets := 12
  let triplets_sets := 3
  let twins := twins_sets * 2
  let triplets := triplets_sets * 3
  let twin_shakes_twins := twins * (twins - 2)
  let triplet_shakes_triplets := triplets * (triplets - 3)
  let twin_shakes_triplets := twins * (triplets / 3)
  (twin_shakes_twins + triplet_shakes_triplets + twin_shakes_triplets) / 2 = 327 := by
  sorry

end unique_handshakes_l464_464835


namespace rationality_of_expression_l464_464416

def is_perfect_square (n : ℝ) : Prop :=
∃ (m : ℤ), n = m^2

theorem rationality_of_expression (x : ℝ) :
  (∃ (r : ℚ), x + sqrt(x^2 + 9) - 1 / (x + sqrt(x^2 + 9)) = r) ↔ is_perfect_square(x^2 + 9) :=
by
  sorry

end rationality_of_expression_l464_464416


namespace buttons_probability_l464_464573

theorem buttons_probability :
  ∃ (C D : Type) (n_red_C n_blue_C n_red_C' n_blue_C' n_red_D n_blue_D : ℕ),
  n_red_C = 5 ∧
  n_blue_C = 10 ∧
  n_red_C' = n_red_C - 2 ∧
  n_blue_C' = n_blue_C - 4 ∧
  n_red_D = 2 ∧
  n_blue_D = 4 ∧
  (n_red_C' + n_blue_C') = (5 + 10) * 3 / 5 ∧
  n_red_C' + n_blue_C' = 9 ∧
  (n_red_D + n_blue_D) = 6 ∧
  (n_red_C + n_red_D = n_red_C' + n_red_D) ∧
  (n_blue_C + n_blue_D = n_blue_C' + n_blue_D) ∧
  ( ∃ p_bl_C p_bl_D : ℚ,
    p_bl_C = (n_blue_C' : ℚ) / (n_red_C' + n_blue_C' : ℚ) ∧
    p_bl_D = (n_blue_D : ℚ) / (n_red_D + n_blue_D : ℚ) ∧
    p_bl_C = 2 / 3 ∧
    p_bl_D = 2 / 3 ∧
    p_bl_C * p_bl_D = 4 / 9
  ) :=
by
  have h1: 5 + 10 = 15 := rfl
  have h2: (5 + 10) * 3 / 5 = 9 := rfl
  have h3: n_red_C' = 3 := rfl
  have h4: n_blue_C' = 6 := rfl
  have h5: n_red_D = 2 := rfl
  have h6: n_blue_D = 4 := rfl
  use Unit, Unit, 5, 10, 3, 6, 2, 4 -- Types and values
  split,
  all_goals { try { assumption } }, -- Assume already proven values.
  existsi (6 : ℚ) / 9, (4 : ℚ) / 6, -- Probabilities in C and D
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  split,
  exact rfl,
  sorry -- We skip the proof here.

end buttons_probability_l464_464573


namespace smallest_n_Sn_l464_464943

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n ≥ 1 → 3 * (a (n + 1)) + a n = 4

def a1 : ℝ := 9

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
finset.sum (finset.range n) (λ i, a (i + 1))

theorem smallest_n_Sn (a : ℕ → ℝ) (S_n : ℕ → ℝ)
  (seq_cond : sequence a)
  (init_cond : a 1 = a1)
  (sum_cond : ∀ n, S_n n = S a n)
  : (∃ n : ℕ, (|S_n n - n - 6| < 1 / 125) ∧ 
      (∀ m : ℕ, (|S_n m - m - 6| < 1 / 125) → n ≤ m)) :=
sorry

end smallest_n_Sn_l464_464943


namespace tangent_line_eq_l464_464731

theorem tangent_line_eq (f : ℝ → ℝ) (f' : ℝ → ℝ) (x y : ℝ) :
  (∀ x, f x = Real.exp x) →
  (∀ x, f' x = Real.exp x) →
  f 0 = 1 →
  f' 0 = 1 →
  x = 0 →
  y = 1 →
  x - y + 1 = 0 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end tangent_line_eq_l464_464731


namespace find_angle_A_find_length_AD_l464_464681

-- Statement for part (1): Proving angle A
theorem find_angle_A 
  (a b c A B C : ℝ) 
  (h_triangle : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_equation : 2 * sin A * sin (C + π / 6) = sin (A + C) + sin C) :
  A = π / 3 := by
  sorry

-- Statement for part (2): Proving the length of AD
theorem find_length_AD 
  (a b c A B C D AD : ℝ) 
  (h_b : b = 2)
  (h_c : c = 1)
  (h_isosceles : A = π / 3 ∧ B = π / 2 ∧ C + π / 3 = π)
  (h_midpoint_D : D = (b / 2) * sqrt (1 - (2 / b) ^ 2)) : 
  AD = sqrt (c^2 + (D^2)) := by
  sorry

end find_angle_A_find_length_AD_l464_464681


namespace compare_a_b_c_l464_464662

noncomputable def a : ℝ := 4^0.9
noncomputable def b : ℝ := 8^0.48
noncomputable def c : ℝ := (1/2)^(-1.5)

theorem compare_a_b_c : a > c ∧ c > b := by
  sorry

end compare_a_b_c_l464_464662


namespace equivalent_proof_problem_l464_464955

noncomputable def perimeter_inner_polygon (pentagon_perimeter : ℕ) : ℕ :=
  let side_length := pentagon_perimeter / 5
  let inner_polygon_sides := 10
  inner_polygon_sides * side_length

theorem equivalent_proof_problem :
  perimeter_inner_polygon 65 = 130 :=
by
  sorry

end equivalent_proof_problem_l464_464955


namespace path_count_in_grid_l464_464418

theorem path_count_in_grid :
  let grid_width := 6
  let grid_height := 5
  let total_steps := 8
  let right_steps := 5
  let up_steps := 3
  ∃ (C : Nat), C = Nat.choose total_steps up_steps ∧ C = 56 :=
by
  sorry

end path_count_in_grid_l464_464418


namespace new_hens_laying_pattern_l464_464206

theorem new_hens_laying_pattern (x y : ℕ) (h1 : x = 4) (h2 : y = 2) :
  (∀ d : ℕ, (d = 1 → 60 / d = 60) ∧ (d = 2 → 60 / d = 30) ∧ (d = 3 → 60 / d = 20)) ∧
  (155 = 60 + 30 + 20 + 60 / x + 60 / y) ∧ (x = 2 * y) :=
begin
  sorry
end

end new_hens_laying_pattern_l464_464206


namespace possible_values_of_a_l464_464957

theorem possible_values_of_a (a : ℝ) (h : 2 ∈ ({1, a^2 - 3 * a + 2, a + 1} : Set ℝ)) : a = 1 ∨ a = 3 :=
by
  sorry

end possible_values_of_a_l464_464957


namespace initial_percentage_increase_is_10_l464_464742

-- Define the constants for the original and final salary
def original_salary : ℝ := 6000
def final_salary : ℝ := 6270

-- Define the function representing the final salary after increase and decrease
def modified_salary (x : ℝ) : ℝ :=
  (original_salary * (1 + x / 100)) * 0.95

-- Prove that the initial percentage increase x is 10%
theorem initial_percentage_increase_is_10 :
  ∃ x : ℝ, modified_salary x = final_salary ∧ x = 10 :=
by
  sorry

end initial_percentage_increase_is_10_l464_464742


namespace not_possible_to_paint_cells_l464_464159

theorem not_possible_to_paint_cells
  (m n : ℕ)
  (h1 : ∃ (P : Fin m × Fin n → Prop), (∀ i j, P i j ∨ ¬P i j) ∧ (∑ i j, if P i j then 1 else 0 = m * n / 2))
  (h2 : ∀ i, (∑ j, if P i j then 1 else 0 > 3 / 4 * n) ∨ (∑ j, if ¬P i j then 1 else 0 > 3 / 4 * n))
  (h3 : ∀ j, (∑ i, if P i j then 1 else 0 > 3 / 4 * m) ∨ (∑ i, if ¬P i j then 1 else 0 > 3 / 4 * m)) :
  False :=
by
sorrry

end not_possible_to_paint_cells_l464_464159


namespace max_values_greater_than_half_l464_464056

theorem max_values_greater_than_half (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (hdistinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (if sin α * cos β > 1 / 2 then 1 else 0) +
  (if sin β * cos γ > 1 / 2 then 1 else 0) +
  (if sin γ * cos α > 1 / 2 then 1 else 0) ≤ 2 :=
sorry

end max_values_greater_than_half_l464_464056


namespace variance_shifted_data_l464_464082

theorem variance_shifted_data {n : ℕ} (a : ℕ → ℝ) (h : n = 2018)
  (var_a : (1 / n) * (finset.sum (finset.range n) (λ i, (a i - (finset.sum (finset.range n) a / n)) ^ 2)) = 4) :
  let b := λ i, a i - 2 in
  (1 / n) * (finset.sum (finset.range n) (λ i, (b i - (finset.sum (finset.range n) b / n)) ^ 2)) = 4 := by
  sorry

end variance_shifted_data_l464_464082


namespace positive_integer_solutions_count_positive_integer_solutions_l464_464515

theorem positive_integer_solutions (x : ℕ) : 8 < -2 * ↑x + 20 → x < 6 :=
by sorry

theorem count_positive_integer_solutions : 
  {x : ℕ | 8 < -2 * ↑x + 20 ∧ x > 0 }.to_finset.card = 5 :=
by sorry

end positive_integer_solutions_count_positive_integer_solutions_l464_464515


namespace prove_AG_AH_eq_AD_sq_l464_464991

noncomputable def acute_triangle (A B C D E F G H : Type*)
  [linear_ordered_field A]
  [linear_ordered_field B]
  [linear_ordered_field C]
  [linear_ordered_field D]
  [linear_ordered_field E]
  [linear_ordered_field F]
  [linear_ordered_field G]
  [linear_ordered_field H] :=
let AD := A → D,
    EF := E → F,
    circumcircle_ABC := sorry in

Prop := 
AC > AB ∧ 
(∀ (D : AD), D ⟂ BC) ∧
(∀ (E : AD), E ⟂ AB) ∧
(∀ (F : AD), F ⟂ AC) ∧
(∀ (G : AD), G = AD ∩ EF) ∧
(∀ (H : AD), H = AD ∩ circumcircle_ABC \ {A}) ∧
AG * AH = AD * AD

theorem prove_AG_AH_eq_AD_sq {A B C D E F G H : Type*} [acute_triangle A B C D E F G H] :
  AG * AH = AD * AD := sorry

end prove_AG_AH_eq_AD_sq_l464_464991


namespace volume_of_given_surface_area_is_correct_l464_464745

-- Define the condition that the surface area S of a sphere with radius r is given by S = 4π r^2.
def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Define the known surface area.
def known_surface_area : ℝ := 400 * Real.pi

-- Define the volume of the sphere given the radius.
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the problem statement to prove.
theorem volume_of_given_surface_area_is_correct :
  ∃ r : ℝ, surface_area_of_sphere r = known_surface_area ∧ volume_of_sphere r = (4000 / 3) * Real.pi :=
by
  sorry

end volume_of_given_surface_area_is_correct_l464_464745


namespace salt_added_correctly_l464_464368

-- Define the problem's conditions and the correct answer in Lean
variable (x : ℝ) (y : ℝ)
variable (S : ℝ := 0.2 * x) -- original salt
variable (E : ℝ := (1 / 4) * x) -- evaporated water
variable (New_volume : ℝ := x - E + 10) -- new volume after adding water

theorem salt_added_correctly :
  x = 150 → y = (1 / 3) * New_volume - S :=
by
  sorry

end salt_added_correctly_l464_464368


namespace acute_angle_condition_projection_vector_l464_464507

variables (a b : ℝ × ℝ) (λ : ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ :=
real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem acute_angle_condition :
  a = (1, 2) → b = (1, 1) →
  (dot_product a (a + (λ * b)) > 0 ∧ λ ≠ 0 →
  λ > -5 / 3 ∧ (λ > 0 ∨ λ < 0)) := by
  intros h1 h2 h3
  sorry

theorem projection_vector :
  a = (1, 2) → b = (1, 1) →
  let sum := (a.1 + b.1, a.2 + b.2) in
  (dot_product sum a / (magnitude a * magnitude a)) *⟦a⟧ =
    (8 / 5, 16 / 5) := by
  intros h1 h2
  sorry

end acute_angle_condition_projection_vector_l464_464507


namespace more_valleyed_than_humped_l464_464795

-- Defining what it means for a five-digit number to be humped or valleyed
def is_humped (n : Nat) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  (n >= 10000) ∧ (n < 100000) ∧ (d3 > d1) ∧ (d3 > d2) ∧ (d3 > d4) ∧ (d3 > d5)

def is_valleyed (n : Nat) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  (n >= 10000) ∧ (n < 100000) ∧ (d3 < d1) ∧ (d3 < d2) ∧ (d3 < d4) ∧ (d3 < d5)

-- Proving that there are more valleyed numbers than humped numbers
theorem more_valleyed_than_humped :
  (∃ n : Nat, is_valleyed n) ∧ (∃ m : Nat, is_humped m) → 
  (count {n : Nat | is_valleyed n} > count {m : Nat | is_humped m}) :=
sorry

end more_valleyed_than_humped_l464_464795


namespace harkamal_payment_l464_464951

noncomputable def total_amount_paid : ℝ :=
  let cost_grapes := 3 * 70 in
  let discount_grapes := 0.05 * cost_grapes in
  let discounted_grapes := cost_grapes - discount_grapes in
  
  let cost_mangoes := 9 * 55 in
  let discount_mangoes := 0.10 * cost_mangoes in
  let discounted_mangoes := cost_mangoes - discount_mangoes in

  let cost_oranges := 5 * 40 in
  let discount_oranges := 0.08 * cost_oranges in
  let discounted_oranges := cost_oranges - discount_oranges in

  let cost_bananas := 7 * 20 in
  let discounted_bananas := cost_bananas in
  
  let total_cost_before_tax := discounted_grapes + discounted_mangoes + discounted_oranges + discounted_bananas in
  let sales_tax := 0.05 * total_cost_before_tax in
  total_cost_before_tax + sales_tax

theorem harkamal_payment : total_amount_paid = 1017.45 :=
by
  -- proof omitted
  sorry

end harkamal_payment_l464_464951


namespace polynomial_root_abs_sum_eq_l464_464031

theorem polynomial_root_abs_sum_eq (n p q r : ℤ) (h1 : p + q + r = 0) 
  (h2 : pq + qr + rp = -3001) (h3 : pqr = -n) (h4 : ∃ n, ∀ p q r, 
  p^3 + q^3 + r^3 - 3 * pqr = 0) : |p| + |q| + |r| = 118 :=
sorry

end polynomial_root_abs_sum_eq_l464_464031


namespace prob_6_higher_than_3_after_10_shuffles_l464_464709

def p_k (k : Nat) : ℚ := (3^k - 2^k) / (2 * 3^k)

theorem prob_6_higher_than_3_after_10_shuffles :
  p_k 10 = (3^10 - 2^10) / (2 * 3^10) :=
by
  sorry

end prob_6_higher_than_3_after_10_shuffles_l464_464709


namespace reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464394

-- Probability that the bags end up in the shed in the reverse order
def probability_reverse_order : ℕ → ℚ
| 0 := 1
| (n + 1) := (1 / 2) * probability_reverse_order n

theorem reverse_order_probability_is_1_over_8 :
  probability_reverse_order 3 = 1 / 8 :=
sorry

-- Probability that the second bag from the bottom ends up as the bottom bag in the shed
def probability_second_from_bottom_bottom : ℚ := 
(1 / 2) * (1 / 2) * (1 / 2)

theorem second_from_bottom_as_bottom_is_1_over_8 :
  probability_second_from_bottom_bottom = 1 / 8 :=
sorry

end reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464394


namespace linear_dependency_vectors_l464_464025

theorem linear_dependency_vectors :
  ∃ (k1 k2 k3 : ℝ), k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0 ∧
  k1 * (1, 0) + k2 * (1, -1) + k3 * (2, 2) = (0 : ℝ × ℝ) :=
begin
  use [-4, 2, 1],
  split; norm_num, 
  split; norm_num, 
  split; norm_num,
  simp, 
  apply and.intro,
  calc
    -4 * 1 + 2 * 1 + 2 * 2 = 0 : by norm_num,
  calc
    -4 * 0 + 2 * (-1) + 1 * 2 = 0 : by norm_num,
end 

end linear_dependency_vectors_l464_464025


namespace regular_octagon_side_length_sum_l464_464509

theorem regular_octagon_side_length_sum (s : ℝ) (h₁ : s = 2.3) (h₂ : 1 = 100) : 
  8 * (s * 100) = 1840 :=
by
  sorry

end regular_octagon_side_length_sum_l464_464509


namespace ls_parallel_pq_l464_464652

-- Definitions for points L and S, and their parallelism
variables {A P C M N Q L S : Type}

-- Assume points and lines for the geometric setup
variables (A P C M N Q L S : Type)

-- Assume the intersections and the required parallelism conditions
variables (hL : ∃ (AP CM : Set (Set L)), L ∈ AP ∩ CM)
variables (hS : ∃ (AN CQ : Set (Set S)), S ∈ AN ∩ CQ)

-- The goal: LS is parallel to PQ
theorem ls_parallel_pq : LS ∥ PQ :=
sorry

end ls_parallel_pq_l464_464652


namespace odd_periodic_function_l464_464685

noncomputable def f : ℝ → ℝ :=
  sorry

theorem odd_periodic_function (f : ℝ → ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 4) = f x)
  (h_domain : ∀ x : ℝ, x ∈ set.Ici 4 → x < 6 → f x = 2 - x^2) :
  f (-1) = 23 :=
begin
  sorry
end

end odd_periodic_function_l464_464685


namespace angle_C_60_degrees_l464_464975

theorem angle_C_60_degrees {A B C : ℝ} (h : sin A ^ 2 - sin C ^ 2 + sin B ^ 2 = sin A * sin B) : 
  C = 60 := 
sorry

end angle_C_60_degrees_l464_464975


namespace arithmetic_sequence_value_l464_464998

variable (a : ℕ → ℝ)
variable (a₁ d a₇ a₅ : ℝ)
variable (h_seq : ∀ n, a n = a₁ + (n - 1) * d)
variable (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120)

theorem arithmetic_sequence_value :
  a 7 - 1/3 * a 5 = 16 :=
sorry

end arithmetic_sequence_value_l464_464998


namespace alice_study_time_l464_464826

noncomputable def study_time_for_physics (k : ℕ) (score : ℕ) : ℚ :=
k / score

theorem alice_study_time :
  (∃ k, 
    (∀ t1 s1, s1 = 6 ∧ t1 = 80 → t1 * s1 = k) ∧
    (∃ s2, s2 = 90 ∧ (k / s2 = 16 / 3))) :=
begin
  use 480,
  split,
  {
    intros t1 s1 h,
    cases h with h1 h2,
    rw [h1, h2],
    norm_num,
  },
  {
    use 90,
    split,
    norm_num,
    norm_num,
  }
end

end alice_study_time_l464_464826


namespace intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l464_464945

-- Define the solution sets A and B given conditions
def solution_set_A (a : ℝ) : Set ℝ :=
  { x | |x - 1| ≤ a }

def solution_set_B : Set ℝ :=
  { x | (x - 2) * (x + 2) > 0 }

theorem intersection_A_B_when_a_eq_2 :
  solution_set_A 2 ∩ solution_set_B = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

theorem range_of_a_when_intersection_is_empty :
  ∀ (a : ℝ), solution_set_A a ∩ solution_set_B = ∅ → 0 < a ∧ a ≤ 1 :=
by
  sorry

end intersection_A_B_when_a_eq_2_range_of_a_when_intersection_is_empty_l464_464945


namespace pairs_of_positive_integers_l464_464425

def number_of_divisors (n : ℕ) : ℕ :=
∏ i in (unique_factors n), (i + 1)

theorem pairs_of_positive_integers (n k : ℕ) (h : ∃s : ℕ, number_of_divisors (s * n) = number_of_divisors (s * k)) :
  ¬ (n ∣ k) ∧ ¬ (k ∣ n) := 
begin
  sorry
end

end pairs_of_positive_integers_l464_464425


namespace sqrt_sum_inequality_l464_464961

theorem sqrt_sum_inequality 
  (x y z : ℝ)
  (hx : 1 < x)
  (hy : 1 < y)
  (hz : 1 < z)
  (hxyz_eq : 1 / x + 1 / y + 1 / z = 2) :
  sqrt (x + y + z) >= sqrt (x - 1) + sqrt (y - 1) + sqrt (z - 1) := by 
  sorry

end sqrt_sum_inequality_l464_464961


namespace sin_double_angle_l464_464958

theorem sin_double_angle (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 := by
  sorry

end sin_double_angle_l464_464958


namespace angle_equality_l464_464315

noncomputable def S1 : Type := sorry  -- Definition for circle S1
noncomputable def S2 : Type := sorry  -- Definition for circle S2
noncomputable def O1 : S1 := sorry     -- Center of circle S1
noncomputable def O2 : S2 := sorry     -- Center of circle S2
noncomputable def A : Type := sorry    -- Intersection point of S1 and S2
noncomputable def K1 : Type := sorry   -- Intersection point of O2A with S1
noncomputable def K2 : Type := sorry   -- Intersection point of O1A with S2

theorem angle_equality 
  (h1 : intersect S1 S2 A)
  (h2 : intersects_line_circle O1 A S2 K2)
  (h3 : intersects_line_circle O2 A S1 K1)
  : angle O1 O2 A = angle K1 K2 A := 
  sorry

end angle_equality_l464_464315


namespace max_size_of_S_l464_464190

def maxSubsetDivisibleBySeven (S : Set ℕ) : Prop :=
  S ⊆ Finₓ 51 ∧ (∀ a b ∈ S, a ≠ b → (a + b) % 7 ≠ 0)

theorem max_size_of_S (S : Set ℕ) (h : maxSubsetDivisibleBySeven S) : 
  ∃ t : Finset ℕ, (t.card = 23 ∧ (∀ s ∈ t, s ∈ S) ∧ maxSubsetDivisibleBySeven (↑t)) :=
sorry

end max_size_of_S_l464_464190


namespace minimum_value_of_function_l464_464684

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_function :
  ∀ x ≥ 0, x ≠ 0 → ∃ c, 0 < c ∧ c < ∞ ∧ x = 1 / Real.exp 1 → f x = -1 / Real.exp 1 :=
begin
  sorry
end

end minimum_value_of_function_l464_464684


namespace parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l464_464441

-- Define the first parabola proof problem
theorem parabola_vertex_at_origin_axis_x_passing_point :
  (∃ (m : ℝ), ∀ (x y : ℝ), y^2 = m * x ↔ (y, x) = (0, 0) ∨ (x = 6 ∧ y = -3)) → 
  ∃ m : ℝ, m = 1.5 ∧ (y^2 = m * x) :=
sorry

-- Define the second parabola proof problem
theorem parabola_vertex_at_origin_axis_y_distance_focus :
  (∃ (p : ℝ), ∀ (x y : ℝ), x^2 = 4 * p * y ↔ (y, x) = (0, 0) ∨ (p = 3)) → 
  ∃ q : ℝ, q = 12 ∧ (x^2 = q * y ∨ x^2 = -q * y) :=
sorry

end parabola_vertex_at_origin_axis_x_passing_point_parabola_vertex_at_origin_axis_y_distance_focus_l464_464441


namespace ceiling_floor_expression_l464_464846

theorem ceiling_floor_expression :
  (Int.ceil ((12:ℚ) / 5 * ((-19:ℚ) / 4 - 3)) - Int.floor (((12:ℚ) / 5) * Int.floor ((-19:ℚ) / 4)) = -6) :=
by 
  sorry

end ceiling_floor_expression_l464_464846


namespace unique_alpha_exists_l464_464913

noncomputable def f (x a b : ℝ) := -x + Real.sqrt ((x + a) * (x + b))

theorem unique_alpha_exists (a b s : ℝ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_distinct : a ≠ b) (h_s_range : 0 < s ∧ s < 1) :
  ∃! α : ℝ, 0 < α ∧ f α a b = Real.pow ((a ^ s + b ^ s) / 2) (1 / s) :=
sorry

end unique_alpha_exists_l464_464913


namespace real_roots_of_f_l464_464439

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem real_roots_of_f :
  {x | f x = 0} = {-1, 1, 2, 3} :=
sorry

end real_roots_of_f_l464_464439


namespace product_of_roots_cubic_l464_464405

theorem product_of_roots_cubic:
  (∀ x : ℝ, x^3 - 15 * x^2 + 60 * x - 45 = 0 → x = r_1 ∨ x = r_2 ∨ x = r_3) →
  r_1 * r_2 * r_3 = 45 :=
by
  intro h
  -- the proof should be filled in here
  sorry

end product_of_roots_cubic_l464_464405


namespace monotonic_decreasing_interval_l464_464738

noncomputable def f (x : ℝ) : ℝ := (4 - x) * Real.exp x

theorem monotonic_decreasing_interval :
  { x : ℝ | f x is decreasing } = { x : ℝ | x > 3 } :=
by
  sorry

end monotonic_decreasing_interval_l464_464738


namespace largest_m_l464_464030

def f (n : Nat) : Nat := 
  -- function to compute the product of the digits of n
  sorry

theorem largest_m (m : Nat) : 
  (∀ (n : Nat), n > 0 → ∃ k : Nat, n = 10^k + a (a < 10^k) → 
      (∞ : ℝ) → (∑ (n from 1 to ∞), (f n) / (m ^ (⌊ log₁₀ n ⌋))) ∈ ℤ) → 
  m = 2070 :=
sorry

end largest_m_l464_464030


namespace gum_distribution_l464_464168

theorem gum_distribution : 
  ∀ (John Cole Aubrey: ℕ), 
    John = 54 → 
    Cole = 45 → 
    Aubrey = 0 → 
    ((John + Cole + Aubrey) / 3) = 33 := 
by
  intros John Cole Aubrey hJohn hCole hAubrey
  sorry

end gum_distribution_l464_464168


namespace convert_to_rectangular_form_l464_464860

theorem convert_to_rectangular_form :
  (Complex.exp (13 * Real.pi * Complex.I / 2)) = Complex.I :=
by
  sorry

end convert_to_rectangular_form_l464_464860


namespace hyperbola_equation_l464_464075

-- Define the conditions
variables (a b : ℝ) (x y : ℝ)
variables (a_pos : a > 0) (b_pos : b > 0)

-- Define the equation of the hyperbola
def hyperbola := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the asymptotes
def asymptotes := ∀ x, y = (sqrt 3 / 3) * x ∨ y = -((sqrt 3 / 3) * x )

-- Define the distance from the vertex to the asymptote
def vertex_asymptote_distance := a / 2 = 1

-- Theorem statement
theorem hyperbola_equation :
  hyperbola 2 (2 * sqrt 3 / 3) :=
begin
  -- We state the required proof for the given equation
  sorry
end

end hyperbola_equation_l464_464075


namespace LS_parallel_PQ_l464_464637

universe u
variables {Point : Type u} [Geometry Point]

-- Defining points A, P, C, M, N, Q as elements of type Point
variables (A P C M N Q : Point)

-- Let L be the intersection point of the lines AP and CM
def L := Geometry.intersection (Geometry.line_through A P) (Geometry.line_through C M)

-- Let S be the intersection point of the lines AN and CQ
def S := Geometry.intersection (Geometry.line_through A N) (Geometry.line_through C Q)

-- Proof: LS is parallel to PQ
theorem LS_parallel_PQ : Geometry.parallel (Geometry.line_through L S) (Geometry.line_through P Q) :=
sorry

end LS_parallel_PQ_l464_464637


namespace exp_thirteen_pi_over_two_eq_i_l464_464858

theorem exp_thirteen_pi_over_two_eq_i : exp (13 * real.pi * complex.I / 2) = complex.I := 
by
  sorry

end exp_thirteen_pi_over_two_eq_i_l464_464858


namespace cost_per_kg_mixture_l464_464822

variables (C1 C2 R Cm : ℝ)

-- Statement of the proof problem
theorem cost_per_kg_mixture :
  C1 = 6 → C2 = 8.75 → R = 5 / 6 → Cm = C1 * R + C2 * (1 - R) → Cm = 6.458333333333333 :=
by intros hC1 hC2 hR hCm; sorry

end cost_per_kg_mixture_l464_464822


namespace greatest_n_factorial_l464_464964

theorem greatest_n_factorial (n : ℕ) :
  (3^9 : ℕ) ∣ (22.factorial) :=
sorry

end greatest_n_factorial_l464_464964


namespace largest_n_for_unique_solution_l464_464360

def base_n_polynomial (n : ℕ) (d : ℕ) : Type :=
{ p : Π (i : ℕ), ℤ // (∀ i, 0 ≤ p i ∧ p i < n) ∧ p d > 0 }

def eval_base_n_polynomial (n : ℕ) (d : ℕ) (p : base_n_polynomial n d) (x : ℝ) : ℝ :=
  ∑ i in range (d + 1), (p.1 i) * (x ^ i)

theorem largest_n_for_unique_solution :
  ∃ n : ℕ, 0 < n ∧ (∀ c : ℝ, ∀ (p1 p2 : base_n_polynomial n (degree p1)),
    eval_base_n_polynomial n (degree p1) p1 (sqrt 2 + sqrt 3) = c ∧
    eval_base_n_polynomial n (degree p2) p2 (sqrt 2 + sqrt 3) = c →
    p1 = p2) ∧ n = 9 :=
begin
  sorry
end

end largest_n_for_unique_solution_l464_464360


namespace max_value_ab_l464_464996

noncomputable def tangency_condition (a b : ℝ) := 
(a + 2 * b) / √5 = √5

noncomputable def circle_center_above_line (a b : ℝ) :=
a + 2 * b > 0

theorem max_value_ab (a b : ℝ) 
    (h1 : tangency_condition a b) 
    (h2 : circle_center_above_line a b) : 
    a * b ≤ 25 / 8 :=
sorry

end max_value_ab_l464_464996


namespace total_hours_l464_464696

variable (K : ℕ) (P : ℕ) (M : ℕ)

-- Conditions:
axiom h1 : P = 2 * K
axiom h2 : P = (1 / 3 : ℝ) * M
axiom h3 : M = K + 105

-- Goal: Proving the total number of hours is 189
theorem total_hours : K + P + M = 189 := by
  sorry

end total_hours_l464_464696


namespace vector_addition_l464_464043

def c : ℝ × ℝ × ℝ := (3, 5, -2)
def d : ℝ × ℝ × ℝ := (-1, 4, 3)

theorem vector_addition : 
  (2 * c.1 + 1/2 * d.1, 2 * c.2 + 1/2 * d.2, 2 * c.3 + 1/2 * d.3) = (5.5, 12, -2.5) :=
by sorry

end vector_addition_l464_464043


namespace chloe_next_multiple_age_sum_digits_l464_464165

noncomputable def chloe_age_next_multiple_digit_sum : ℕ → ℕ 
| C := if h : (C = 10) then (1 + 3) else (if h2 : (C = 0) then 0 else 0) -- some determination based on proof structure, real determination in proof steps

theorem chloe_next_multiple_age_sum_digits :
  ∃ C : ℕ, ∀ J L : ℕ, J = C + 2 → L = 1 → (∀ n ∈ range(6), J = (L + n)) → (C = 10) → (chloe_age_next_multiple_digit_sum C = 4) :=
begin
  sorry
end

end chloe_next_multiple_age_sum_digits_l464_464165


namespace price_per_jar_of_jam_l464_464840

def Betty_strawberries : ℕ := 16
def Matthew_strawberries : ℕ := Betty_strawberries + 20
def Natalie_strawberries : ℕ := Matthew_strawberries / 2
def strawberries_per_jar : ℕ := 7
def total_money : ℕ := 40

theorem price_per_jar_of_jam :
  let total_strawberries := Betty_strawberries + Matthew_strawberries + Natalie_strawberries in
  let jars_of_jam := total_strawberries / strawberries_per_jar in
  total_money / jars_of_jam = 4 :=
by
  sorry

end price_per_jar_of_jam_l464_464840


namespace problem_solution_l464_464997

-- Definitions of the involved geometric elements
def triangle (A B C : Point) : Prop := 
  acute ∠ABC ∧ acute ∠BCA ∧ acute ∠CAB

def circumcircle (ABC : triangle) : Circle := sorry

def orthocenter (ABC : triangle) : Point := sorry
def incenter (ABC : triangle) : Point := sorry

def midpoint (A H : Point) : Point := sorry

noncomputable def line (A B : Point) : Line := sorry
def parallel (l m : Line) : Prop := sorry
def perpendicular (l m : Line) : Prop := sorry

-- Proof goals - given conditions and proof properties
theorem problem_solution (ABC : triangle A B C)
  (O : circumcircle ABC)
  (H : orthocenter ABC)
  (I : incenter ABC)
  (M : midpoint A H)
  (cond1 : parallel (line A O) (line M I))
  (D : Point)
  (cond2 : D = intersection (extended_line A H) O)
  (P Q : Point)
  (cond3 : P = intersection (line A O) (line B C))
  (cond4 : Q = intersection (line O D) (line B C)) :
  isosceles_triangle O P Q ∧ (perpendicular (line I Q) (line B C) → cos ♯B + cos ♯C = 1) :=
by
  sorry

end problem_solution_l464_464997


namespace find_g1_l464_464272

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x : ℝ) (hx : x ≠ 1 / 2) : g x + g ((x + 2) / (2 - 4 * x)) = 2 * x + 1

theorem find_g1 : g 1 = 39 / 11 :=
by
  sorry

end find_g1_l464_464272


namespace correct_inequality_l464_464483

variable {f : ℝ → ℝ}

-- Conditions
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Icc (-5 : ℝ) (5 : ℝ), f (-x) = f x

def is_monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ Icc (0 : ℝ) (5 : ℝ), x ≤ y → f x ≤ f y

def condition3 (f : ℝ → ℝ) : Prop :=
  f (-3) < f 1

-- Theorem to be proven
theorem correct_inequality
  (h1 : is_even f)
  (h2 : is_monotonic f)
  (h3 : condition3 f) : f 1 < f 0 :=
sorry

end correct_inequality_l464_464483


namespace problem_solution_product_of_distances_ratio_of_distances_l464_464081

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ :=
  ((1 / 2), 0)

noncomputable def line_through_focus (m : ℝ) (y : ℝ) : ℝ :=
  m * y + (1 / 2)

noncomputable def parabola_intersection_points (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let d := real.sqrt (1 + m^2)
  let y₁ := m * d + 1
  let y₂ := m * (-d) + 1
  (((line_through_focus m y₁), y₁), ((line_through_focus m y₂), y₂))

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem problem_solution (m : ℝ) (A B : ℝ × ℝ) (F : ℝ × ℝ)
  (h₁ : B = parabola_focus 2)
  (h₂ : parabola_intersection_points m = (A, F))
  (h₃ : distance A.1 A.2 B.1 B.2 = 3)
  (h4 : abs (F.fst) > abs (A.fst)) :
  abs (distance F.1 F.2 A.1 A.2 - distance F.1 F.2 B.1 B.2) = real.sqrt 3 :=
begin
  sorry
end

theorem product_of_distances (A B F : ℝ × ℝ)
  (h₁ : distance A.1 A.2 B.1 B.2 = 3)
  (h₂ : parabola_focus 2 = F) :
  (distance A.1 A.2 F.1 F.2) * (distance F.1 F.2 B.1 B.2) = 3 / 2 :=
begin
  sorry
end

theorem ratio_of_distances (A B F : ℝ × ℝ) 
  (h₁ : distance F.fst F.snd A.fst A.snd = (3 + abs (real.sqrt 3)) / 2)
  (h₂ : distance F.fst F.snd B.fst B.snd = (3 - abs (real.sqrt 3)) / 2):
  distance F.fst F.snd A.fst A.snd / distance F.fst F.snd B.fst B.snd = 2 + real.sqrt 3 :=
begin
  sorry
end

end problem_solution_product_of_distances_ratio_of_distances_l464_464081


namespace semicircle_PQ_parallel_PM_PS_eq_l464_464366

open EuclideanGeometry -- Open the necessary module for Euclidean Geometry

theorem semicircle_PQ_parallel_PM_PS_eq
    {A B P Q M S : Point}
    (hAB : M = midpoint A B)
    (hP : P ∉ {A, B})
    (hQ : Q = midpoint_circle_arc A P)
    (hS : ∃ l : Line, parallel l (line_from_points P Q) ∧ S = intersection (line_from_points B P) l)
    : dist M P = dist P S := 
sorry

end semicircle_PQ_parallel_PM_PS_eq_l464_464366


namespace cartesian_to_polar_l464_464502

-- Definition for the Cartesian coordinates of point P
def point_P := (1 : ℝ, -Real.sqrt 3)

-- Definition of expected polar coordinates in tuple format
def polar_P := (2 : ℝ, -(Real.pi / 3))

-- The actual proof statement (skeleton) asserting the equivalence
theorem cartesian_to_polar (P : ℝ × ℝ) (polar : ℝ × ℝ) : P = point_P → polar = polar_P → 
  ∃ ρ θ, P = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ polar = (ρ, θ) :=
by
  intros h₁ h₂
  sorry

end cartesian_to_polar_l464_464502


namespace determine_omega_l464_464494

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

-- Conditions
variables (ω : ℝ) (ϕ : ℝ)
axiom omega_pos : ω > 0
axiom phi_bound : abs ϕ < Real.pi / 2
axiom symm_condition1 : ∀ x, f ω ϕ (Real.pi / 4 - x) = -f ω ϕ (Real.pi / 4 + x)
axiom symm_condition2 : ∀ x, f ω ϕ (-Real.pi / 2 - x) = f ω ϕ x
axiom monotonic_condition : ∀ x1 x2, 0 < x1 → x1 < x2 → x2 < Real.pi / 8 → f ω ϕ x1 < f ω ϕ x2

theorem determine_omega : ω = 1 ∨ ω = 5 :=
sorry

end determine_omega_l464_464494


namespace parallel_LS_PQ_l464_464632

open_locale big_operators

variables (A P M N C Q L S : Type*)

-- Assume that L is the intersection of lines AP and CM
def is_intersection_L (AP CM : Set (Set Type*)) : Prop :=
  L ∈ AP ∩ CM

-- Assume that S is the intersection of lines AN and CQ
def is_intersection_S (AN CQ : Set (Set Type*)) : Prop :=
  S ∈ AN ∩ CQ

-- Prove that LS ∥ PQ
theorem parallel_LS_PQ 
  (hL : is_intersection_L A P M C L) 
  (hS : is_intersection_S A N C Q S) : 
  parallel (line_through L S) (line_through P Q) :=
sorry

end parallel_LS_PQ_l464_464632


namespace find_f_difference_l464_464058

variable {α : Type*}
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_period : ∀ x, f (x + 5) = f x)
variable (h_value : f (-2) = 2)

theorem find_f_difference : f 2012 - f 2010 = -2 :=
by {
  sorry
}

end find_f_difference_l464_464058


namespace negate_exists_implies_forall_l464_464279

-- Define the original proposition
def prop1 (x : ℝ) : Prop := x^2 + 2 * x + 2 < 0

-- The negation of the proposition
def neg_prop1 := ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0

-- Statement of the equivalence
theorem negate_exists_implies_forall :
  ¬(∃ x : ℝ, prop1 x) ↔ neg_prop1 := by
  sorry

end negate_exists_implies_forall_l464_464279


namespace probability_point_A_on_hyperbola_l464_464221

-- Define the set of numbers
def numbers : List ℕ := [1, 2, 3]

-- Define the coordinates of point A taken from the set, where both numbers are different
def point_A_pairs : List (ℕ × ℕ) :=
  [ (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) ]

-- Define the function indicating if a point (m, n) lies on the hyperbola y = 6/x
def lies_on_hyperbola (m n : ℕ) : Prop :=
  n = 6 / m

-- Calculate the probability of a point lying on the hyperbola
theorem probability_point_A_on_hyperbola : 
  (point_A_pairs.countp (λ (p : ℕ × ℕ), lies_on_hyperbola p.1 p.2)).toRat / (point_A_pairs.length).toRat = 1 / 3 := 
sorry

end probability_point_A_on_hyperbola_l464_464221


namespace sqrt_sum_inequality_l464_464479

theorem sqrt_sum_inequality (a b : ℝ) (h1 : a + b = 1) (h2 : (a + 1/2) * (b + 1/2) ≥ 0) : 
  sqrt(a + 1/2) + sqrt(b + 1/2) ≤ 2 := 
sorry

end sqrt_sum_inequality_l464_464479


namespace july_15th_conditions_l464_464403

def temperature_at_least_70 : Prop := 
  ∃ T : ℝ, T ≥ 70

def no_wind : Prop := 
  ¬windy

def ideal_for_picnicking := 
  temperature_at_least_70 ∧ no_wind

theorem july_15th_conditions 
  (h1 : (temperature_at_least_70 ∧ no_wind) → ideal_for_picnicking) 
  (h2 : ¬ideal_for_picnicking) :
  T < 70 ∨ windy := 
by
  sorry

end july_15th_conditions_l464_464403


namespace billy_crayons_l464_464841

theorem billy_crayons (total_crayons: ℕ) (monkey_ate: ℕ) (hippo_ate: ℕ) (crayons_left: ℕ)
                      (h1: total_crayons = 200)
                      (h2: monkey_ate = 64)
                      (h3: hippo_ate = 2 * monkey_ate)
                      (h4: crayons_left = total_crayons - (monkey_ate + hippo_ate)):
                      crayons_left = 8 :=
by
  have h_hippo: hippo_ate = 128, from calc
    hippo_ate = 2 * monkey_ate : by rw [h3]
            ... = 2 * 64       : by rw [h2]
            ... = 128          : by norm_num,
  
  have h_total_eaten: monkey_ate + hippo_ate = 192, from calc
    monkey_ate + hippo_ate = 64 + 128 : by rw [h_hippo, h2]
                          ... = 192   : by norm_num,

  have h_crayons_left: total_crayons - (monkey_ate + hippo_ate) = 8, from calc
    total_crayons - (monkey_ate + hippo_ate) = 200 - 192 : by rw [h1, h_total_eaten]
                                              ... = 8     : by norm_num,
  
  show crayons_left = 8, from calc
    crayons_left = total_crayons - (monkey_ate + hippo_ate) : by rw [h4]
               ... = 8 : by rw [h_crayons_left]

end billy_crayons_l464_464841


namespace find_abc_l464_464404

theorem find_abc :
  ∃ abc : ℕ, 100 ≤ abc ∧ abc ≤ 999 ∧ (594000 + abc) % 651 = 0 ∧ abc = 112 :=
begin
  use 112,
  split,
  { exact dec_trivial }, -- 100 ≤ 112
  split,
  { exact dec_trivial }, -- 112 ≤ 999
  split,
  { norm_num }, -- (594000 + 112) % 651 = 0
  { refl },
end

end find_abc_l464_464404


namespace three_digit_numbers_div_by_17_l464_464104

theorem three_digit_numbers_div_by_17 : 
  let k_min := 6
  let k_max := 58
  (k_max - k_min + 1) = 53 := by
  let k_min := 6
  let k_max := 58
  show (k_max - k_min + 1) = 53 from sorry

end three_digit_numbers_div_by_17_l464_464104


namespace arithmetic_sequence_sum_l464_464930

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : a 8 + a 10 = 2) : 
  (17 * (a 1 + a 17) / 2) = 17 := by
sorry

end arithmetic_sequence_sum_l464_464930


namespace problem_statement_l464_464480

theorem problem_statement (m : ℝ) (h : m^2 - m - 2 = 0) : m^2 - m + 2023 = 2025 :=
sorry

end problem_statement_l464_464480


namespace four_student_committees_from_six_l464_464033

theorem four_student_committees_from_six : (Nat.choose 6 4) = 15 := 
by 
  sorry

end four_student_committees_from_six_l464_464033


namespace average_goals_is_approximately_l464_464262

def number_of_players_scoring (goals : ℕ) : ℕ :=
  match goals with
  | 5 => 3
  | 6 => 6
  | 7 => 2
  | 9 => 1
  | 12 => 1
  | _ => 0

def total_goals : ℕ :=
  5 * number_of_players_scoring 5 +
  6 * number_of_players_scoring 6 +
  7 * number_of_players_scoring 7 +
  9 * number_of_players_scoring 9 +
  12 * number_of_players_scoring 12

def total_players : ℕ :=
  number_of_players_scoring 5 +
  number_of_players_scoring 6 +
  number_of_players_scoring 7 +
  number_of_players_scoring 9 +
  number_of_players_scoring 12

def average_goals : ℚ :=
  total_goals / total_players

theorem average_goals_is_approximately :
  average_goals ≈ 6.62 := sorry

end average_goals_is_approximately_l464_464262


namespace correct_system_of_eqns_l464_464343

-- Define the conditions and the problem statement
theorem correct_system_of_eqns (x y : ℝ) : 
    (20 / 60) * x + (20 / 60) * y = 3 ∧ 3 - (30 / 60) * x = 2 * (3 - (30 / 60) * y) :=
begin
    sorry
end

end correct_system_of_eqns_l464_464343


namespace part_a_part_b_l464_464380

-- Given conditions for both parts of the problem
def choices (n : ℕ) : list (fin n) := list.fin_range n

-- Part (a)
-- The probability that the bags end up in reverse order in the shed
theorem part_a (n : ℕ) (hn : n = 4) : 
  probability_reverse_order hn = 1 / 8 :=
sorry

-- Part (b)
-- The probability that the second-from-bottom bag in the truck ends up as the bottom bag in the shed
theorem part_b (n : ℕ) (hn : n = 4) : 
  probability_second_from_bottom hn = 1 / 8 :=
sorry

-- Definitions of probability calculations used in the conditions
noncomputable def probability_reverse_order (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

noncomputable def probability_second_from_bottom (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

end part_a_part_b_l464_464380


namespace swim_time_ratio_l464_464355

theorem swim_time_ratio (swim_speed_still : ℝ) (stream_speed : ℝ) (x : ℝ) (h1 : swim_speed_still = 7.5) (h2 : stream_speed = 2.5) :
  let U := swim_speed_still - stream_speed,
      D := swim_speed_still + stream_speed,
      T_upstream := x / U,
      T_downstream := x / D
  in T_upstream / T_downstream = 2 :=
by
  intros
  rw [h1, h2]
  let U := 7.5 - 2.5
  let D := 7.5 + 2.5
  let T_upstream := x / U
  let T_downstream := x / D
  have hU : U = 5 := by norm_num
  have hD : D = 10 := by norm_num
  rw [hU, hD]
  let T_upstream := x / 5
  let T_downstream := x / 10
  have hT : (x / 5) / (x / 10) = 2 := by 
    field_simp
    ring
  exact hT
  sorry

end swim_time_ratio_l464_464355


namespace area_three_layers_is_nine_l464_464305

-- Define the areas as natural numbers
variable (P Q R S T U V : ℕ)

-- Define the combined area of the rugs
def combined_area_rugs := P + Q + R + 2 * (S + T + U) + 3 * V = 90

-- Define the total area covered by the floor
def total_area_floor := P + Q + R + S + T + U + V = 60

-- Define the area covered by exactly two layers of rug
def area_two_layers := S + T + U = 12

-- Define the area covered by exactly three layers of rug
def area_three_layers := V

-- Prove the area covered by exactly three layers of rug is 9
theorem area_three_layers_is_nine
  (h1 : combined_area_rugs P Q R S T U V)
  (h2 : total_area_floor P Q R S T U V)
  (h3 : area_two_layers S T U) :
  area_three_layers V = 9 := by
  sorry

end area_three_layers_is_nine_l464_464305


namespace percentage_of_women_picnic_l464_464983

theorem percentage_of_women_picnic (E : ℝ) (h1 : 0.20 * 0.55 * E + W * 0.45 * E = 0.29 * E) : 
  W = 0.4 := 
  sorry

end percentage_of_women_picnic_l464_464983


namespace find_positive_value_of_X_l464_464180

-- define the relation X # Y
def rel (X Y : ℝ) : ℝ := X^2 + Y^2

theorem find_positive_value_of_X (X : ℝ) (h : rel X 7 = 250) : X = Real.sqrt 201 :=
by
  sorry

end find_positive_value_of_X_l464_464180


namespace keiko_speed_l464_464170

-- Define the speed of Keiko
def KeikoSpeed : ℝ := π / 3

-- Define the lengths of the inner and outer tracks
def L_inner (a b : ℝ) : ℝ := 2 * a + 2 * π * b
def L_outer (a b : ℝ) : ℝ := 2 * a + 2 * π * (b + 8)

-- Define the conditions and problem
theorem keiko_speed (a b s : ℝ) (hb : s = π / 3) : 
  ∀ a b, 
   (48 / (π / 3) ) = (L_outer a b) / s - (L_inner a b) / s :=
by sorry

end keiko_speed_l464_464170


namespace job_completion_people_needed_l464_464857

theorem job_completion_people_needed
(Job : ℕ) (InitialDays : ℕ) (TotalDays : ℕ) (InitialPeople : ℕ) (JobCompletedPortion : ℝ)
(InitialDays = 5) (TotalDays = 25) (InitialPeople = 10) (JobCompletedPortion = 1/5) 
: 10 := 
begin
  -- define variables suitable for the problem
  let RemainingJob := 1 - JobCompletedPortion,
  let RemainingDays := TotalDays - InitialDays,
  let WorkDonePerPersonIn5Days : ℝ := JobCompletedPortion / InitialPeople,
  let WorkDonePerPersonPerDay : ℝ := WorkDonePerPersonIn5Days / 5,
  let TotalPersonDaysNeededToCompleteRemainingJob : ℝ := RemainingJob / WorkDonePerPersonPerDay,
  let NumberOfPeopleNeeded : ℝ := TotalPersonDaysNeededToCompleteRemainingJob / RemainingDays,
  have H : NumberOfPeopleNeeded = 10, by {
    -- From the calculations above, we can show that the necessary number 
    -- of people to complete the job on time, given these rates, is 10.
    sorry
  },
  exact H
end

end job_completion_people_needed_l464_464857


namespace ls_parallel_pq_l464_464653

-- Definitions for points L and S, and their parallelism
variables {A P C M N Q L S : Type}

-- Assume points and lines for the geometric setup
variables (A P C M N Q L S : Type)

-- Assume the intersections and the required parallelism conditions
variables (hL : ∃ (AP CM : Set (Set L)), L ∈ AP ∩ CM)
variables (hS : ∃ (AN CQ : Set (Set S)), S ∈ AN ∩ CQ)

-- The goal: LS is parallel to PQ
theorem ls_parallel_pq : LS ∥ PQ :=
sorry

end ls_parallel_pq_l464_464653


namespace find_a9_l464_464972

noncomputable def poly_expansion (x : ℝ) (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 * (x + 1) + a 2 * (x + 1)^2 + a 3 * (x + 1)^3 + a 4 * (x + 1)^4 +
  a 5 * (x + 1)^5 + a 6 * (x + 1)^6 + a 7 * (x + 1)^7 + 
  a 8 * (x + 1)^8 + a 9 * (x + 1)^9 + a 10 * (x + 1)^(10)

theorem find_a9 (a : ℕ → ℝ) : 
  (∀ (x : ℝ), x^2 + x^{10} = poly_expansion x a) → a 10 = 1 → a 9 = -10 := by
  sorry

end find_a9_l464_464972


namespace book_arrangement_l464_464522

theorem book_arrangement : 
  let total_books := 7
  let identical_math := 3
  let identical_science := 2
  let unique_books := 2
  in finset.card (finset.perm (finset.range total_books)) // (finset.factorial identical_math * finset.factorial identical_science) = 420 := 
by 
  let total_books := 7
  let identical_math := 3
  let identical_science := 2
  let unique_books := 2
  have h1 : finset.factorial total_books = 5040 := by sorry
  have h2 : finset.factorial identical_math = 6 := by sorry
  have h3 : finset.factorial identical_science = 2 := by sorry
  calc
    finset.card (finset.perm (finset.range total_books)) // (finset.factorial identical_math * finset.factorial identical_science) 
      = 5040 // (6 * 2) : by rw [h1, h2, h3]
      ... = 420 : by sorry

end book_arrangement_l464_464522


namespace number_of_subsets_of_3_element_set_l464_464293

theorem number_of_subsets_of_3_element_set :
  ∀ (A : Set (Fin 3)), finset.card (finset.powerset (finset.univ : finset (Fin 3))) = 8 :=
begin
  intro A,
  sorry
end

end number_of_subsets_of_3_element_set_l464_464293


namespace geometric_sequence_max_T_n_l464_464175

noncomputable def geometric_sequence_a_n (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

noncomputable def T_n (n : ℕ) (a : ℝ) (r : ℝ) : ℝ :=
  ∏ i in range n, geometric_sequence_a_n a r (i + 1)

theorem geometric_sequence_max_T_n :
  let a := -6
  let a4 := -3/4
  let r := 1 / 2 in
  ∃ n, 
    a4 = a * r ^ 3 ∧
    r = 1 / 2 ∧
    T_n n a r = (36 : ℝ) ^ (n / 2) * (1 / 2) ^ (n * (n - 1) / 2) ∧
    T_n 4 a r = (36 : ℝ) ^ (4 / 2) * (1 / 2) ^ (4 * (4 - 1) / 2)
:=
begin
  sorry
end

end geometric_sequence_max_T_n_l464_464175


namespace xiao_ming_math_score_l464_464285

theorem xiao_ming_math_score :
  ∀ (max_score routine_weight midterm_weight final_weight routine_score midterm_score final_score : ℝ),
    max_score = 100 →
    routine_weight = 0.3 →
    midterm_weight = 0.3 →
    final_weight = 0.4 →
    routine_score = 90 →
    midterm_score = 90 →
    final_score = 96 →
    (routine_score * routine_weight + midterm_score * midterm_weight + final_score * final_weight) = 92.4 :=
by
  intros max_score routine_weight midterm_weight final_weight routine_score midterm_score final_score,
  intros h_max_score h_routine_weight h_midterm_weight h_final_weight h_routine_score h_midterm_score h_final_score,
  calc
    routine_score * routine_weight + midterm_score * midterm_weight + final_score * final_weight
        = 90 * 0.3 + 90 * 0.3 + 96 * 0.4 : by rw [h_routine_score, h_midterm_score, h_final_score, h_routine_weight, h_midterm_weight, h_final_weight]
    ... = 27 + 27 + 38.4 : by norm_num
    ... = 92.4 : by norm_num

end xiao_ming_math_score_l464_464285


namespace max_values_greater_than_half_l464_464054

theorem max_values_greater_than_half {α β γ : ℝ} (hα : 0 < α ∧ α < π/2)
  (hβ : 0 < β ∧ β < π/2) (hγ : 0 < γ ∧ γ < π/2) (h_distinct : α ≠ β ∧ β ≠ γ ∧ γ ≠ α) :
  (count_fun >(1/2) [sin α * cos β, sin β * cos γ, sin γ * cos α]) ≤ 2 :=
sorry

end max_values_greater_than_half_l464_464054


namespace find_a_l464_464197

variable (a : ℤ)
def U (a : ℤ) := {1, 3*a + 5, a^2 + 1}
def A (a : ℤ) := {1, a + 1}
def complement (s1 s2 : Set ℤ) := {x | x ∈ s1 ∧ x ∉ s2}

theorem find_a :
  complement (U a) (A a) = {5} → U a = {1, 3*a + 5, a^2 + 1} → a = -2 :=
by
  intros h1 h2
  sorry

end find_a_l464_464197


namespace phil_pages_left_is_correct_l464_464208

def pages_in_books : List Nat := 
  [120, 150, 80, 200, 90, 180, 75, 190, 110, 160, 130, 170, 100, 140, 210, 185, 220, 135, 145, 205]

def missing_book_indices : List Nat := [1, 6, 10, 13, 15, 19]

def pages_of_books (indices : List Nat) (all_books : List Nat) : Nat :=
  indices.foldl (λ acc i => acc + List.get all_books i) 0

noncomputable def total_pages_left : Nat :=
  pages_of_books (List.range 20) pages_in_books - pages_of_books missing_book_indices pages_in_books

theorem phil_pages_left_is_correct : total_pages_left = 2110 := sorry

end phil_pages_left_is_correct_l464_464208


namespace tangent_lines_tangent_length_line_AB_l464_464487

noncomputable def circle_C := λ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 2
noncomputable def point_P := (2 : ℝ, -1 : ℝ)

theorem tangent_lines (x y : ℝ) :
  circle_C x y → ∃ k : ℝ, k = 7 ∨ k = -1 :=
sorry

theorem tangent_length :
  ∃ r : ℝ, r = 2 * real.sqrt 2 :=
sorry

theorem line_AB :
  ∀ x y : ℝ, x - 3*y + 3 = 0 :=
sorry

end tangent_lines_tangent_length_line_AB_l464_464487


namespace parabola_focus_equals_hyperbola_focus_l464_464063

noncomputable def hyperbola_right_focus : (Float × Float) := (2, 0)

noncomputable def parabola_focus (p : Float) : (Float × Float) := (p / 2, 0)

theorem parabola_focus_equals_hyperbola_focus (p : Float) :
  parabola_focus p = hyperbola_right_focus → p = 4 := by
  intro h
  sorry

end parabola_focus_equals_hyperbola_focus_l464_464063


namespace parallel_LS_pQ_l464_464617

universe u

variables {P Q A C M N L S T : Type u} [incidence_geometry P Q A C M N L S T] -- Customize as per specific needs in Lean

-- Assume the required points and their intersections
def intersection_L (A P C M : Type u) [incidence_geometry A P C M] : Type u := sorry
def intersection_S (A N C Q : Type u) [incidence_geometry A N C Q] : Type u := sorry

-- Defining the variables as intersections
variables (L : intersection_L A P C M)
variables (S : intersection_S A N C Q)

theorem parallel_LS_pQ (hL : intersection_L A P C M) (hS : intersection_S A N C Q) :
  parallel L S P Q :=
sorry

end parallel_LS_pQ_l464_464617


namespace gcd_1020_multiple_38962_l464_464897

-- Define that x is a multiple of 38962
def multiple_of (x n : ℤ) : Prop := ∃ k : ℤ, x = k * n

-- The main theorem statement
theorem gcd_1020_multiple_38962 (x : ℤ) (h : multiple_of x 38962) : Int.gcd 1020 x = 6 := 
sorry

end gcd_1020_multiple_38962_l464_464897


namespace probability_correct_number_l464_464200

def count_permutations_with_repetition (n: ℕ) (repeated_count: ℕ): ℕ :=
  Nat.factorial n / Nat.factorial repeated_count

theorem probability_correct_number :
  let possible_first_three := 3
  let possible_arrangements := count_permutations_with_repetition 5 2
  let total_combinations := possible_first_three * possible_arrangements
  total_combinations = 180 → 1 / total_combinations = (1 / 180: ℚ) :=
by
  intros
  rw h 
  simp
  sorry

end probability_correct_number_l464_464200


namespace DF_not_bisects_angle_CDA_l464_464592

noncomputable def right_triangle_and_circle (ABC : Triangle) (B C : Point) :=
BC_is_diameter : (diameter BC) ∈ circle ABC

noncomputable def tangent_intersect (D F : Point) :=
(intersect CA_f_leg D_leg) ∧ (is_tangent_at D)

theorem DF_not_bisects_angle_CDA (ABC : Triangle) (B C : Point) (D F : Point) (DF_bisects : bisects DF (∠ CDA)) :
  ∀ (right_triangle_and_circle ABC B C) (tangent_intersect D F), 
    bisects DF (∠ CDA) = false :=
sorry

end DF_not_bisects_angle_CDA_l464_464592


namespace correct_options_l464_464931

variable (α : ℝ)

lemma option_A : 
  sin (π / 3 + α) = sin (2 * π / 3 - α) := 
by sorry

lemma option_B : 
  sin (π / 4 + α) = -cos (5 * π / 4 - α) := 
by sorry

lemma option_C :
  ¬ (tan (π / 3 - α) = tan (2 * π / 3 + α)) := 
by sorry

lemma option_D :
  tan(α) ^ 2 * sin(α) ^ 2 = tan(α) ^ 2 - sin(α) ^ 2 := 
by sorry

theorem correct_options : 
  (sin (π / 3 + α) = sin (2 * π / 3 - α)) ∧ 
  (sin (π / 4 + α) = -cos (5 * π / 4 - α)) ∧
  (¬ (tan (π / 3 - α) = tan (2 * π / 3 + α))) ∧ 
  (tan(α) ^ 2 * sin(α) ^ 2 = tan(α) ^ 2 - sin(α) ^ 2) :=
by
  apply And.intro
  . exact option_A α
  apply And.intro
  . exact option_B α
  apply And.intro
  . exact option_C α
  . exact option_D α

end correct_options_l464_464931


namespace boys_who_did_not_bring_laptops_l464_464691

-- Definitions based on the conditions.
def total_boys : ℕ := 20
def students_who_brought_laptops : ℕ := 25
def girls_who_brought_laptops : ℕ := 16

-- Main theorem statement.
theorem boys_who_did_not_bring_laptops : total_boys - (students_who_brought_laptops - girls_who_brought_laptops) = 11 := by
  sorry

end boys_who_did_not_bring_laptops_l464_464691


namespace probability_of_red_ball_is_correct_l464_464549

noncomputable def probability_of_drawing_red_ball (white_balls : ℕ) (red_balls : ℕ) :=
  let total_balls := white_balls + red_balls
  let favorable_outcomes := red_balls
  (favorable_outcomes : ℚ) / total_balls

theorem probability_of_red_ball_is_correct :
  probability_of_drawing_red_ball 5 2 = 2 / 7 :=
by
  sorry

end probability_of_red_ball_is_correct_l464_464549


namespace bacteria_colony_exceeds_500_l464_464990

theorem bacteria_colony_exceeds_500 :
  ∃ (n : ℕ), (∀ m : ℕ, m < n → 4 * 3^m ≤ 500) ∧ 4 * 3^n > 500 :=
sorry

end bacteria_colony_exceeds_500_l464_464990


namespace ab_eq_neg2_of_fraction_eq_l464_464455

theorem ab_eq_neg2_of_fraction_eq (a b : ℝ) (i : ℂ) (hi : i = complex.I) 
  (h : (a + 2 * i) / i = b + i) : a * b = -2 :=
by
  sorry

end ab_eq_neg2_of_fraction_eq_l464_464455


namespace time_and_distance_to_meet_l464_464316

/-- Define the initial speeds and the increased speeds for Misha and Vasya. -/
structure Speeds where
  misha_initial : ℝ := 8  -- Misha's initial speed (km/h)
  vasya_initial : ℝ := misha_initial / 2  -- Vasya's initial speed is half of Misha's initial speed (km/h)
  misha_increased : ℝ := misha_initial * 1.5  -- Misha's increased speed (km/h)
  vasya_increased : ℝ := vasya_initial * 1.5  -- Vasya's increased speed (km/h)

/-- Calculate the time for Misha and Vasya to meet and the total distance covered before meeting. -/
theorem time_and_distance_to_meet : 
  let initial_separation_rate := Speeds.misha_initial + Speeds.vasya_initial,
      distance_apart := (initial_separation_rate * 1000 / 60) * 45,
      closing_speed := Speeds.misha_increased + Speeds.vasya_increased,
      time_to_meet := distance_apart / (closing_speed * 1000 / 60),
      total_distance := distance_apart + distance_apart in
  distance_apart = 9000 ∧ time_to_meet = 30 ∧ total_distance = 18000 := 
by
  -- Definitions and calculations for the theorem
  let speeds := Speeds.mk
  let initial_separation_rate := speeds.misha_initial + speeds.vasya_initial
  let distance_apart := (initial_separation_rate * 1000 / 60) * 45
  let closing_speed := speeds.misha_increased + speeds.vasya_increased
  let time_to_meet := distance_apart / (closing_speed * 1000 / 60)
  let total_distance := distance_apart + distance_apart
  -- Explicit assertions
  have distance_eq : distance_apart = 9000 := by sorry
  have time_eq : time_to_meet = 30 := by sorry
  have total_dist_eq : total_distance = 18000 := by sorry
  -- Final proof
  exact ⟨distance_eq, time_eq, total_dist_eq⟩

end time_and_distance_to_meet_l464_464316


namespace find_positive_X_l464_464178

variable (X : ℝ) (Y : ℝ)

def hash_rel (X Y : ℝ) : ℝ :=
  X^2 + Y^2

theorem find_positive_X :
  hash_rel X 7 = 250 → X = Real.sqrt 201 :=
by
  sorry

end find_positive_X_l464_464178


namespace prove_parallel_l464_464648

-- Define points A, P, C, M, N, Q, and functions L, S.
variables {A P C M N Q L S : Point}
variables [incidence_geometry]

-- Define L and S based on the given conditions.
def L := intersection (line_through A P) (line_through C M)
def S := intersection (line_through A N) (line_through C Q)

-- Define the condition to prove LS is parallel to PQ.
theorem prove_parallel : parallel (line_through L S) (line_through P Q) :=
sorry

end prove_parallel_l464_464648


namespace magnitude_sum_is_sqrt_10_l464_464085

open Real

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (2, -m)

def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.fst * v2.fst + v1.snd * v2.snd = 0
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.fst^2 + v.snd^2)

theorem magnitude_sum_is_sqrt_10 {m : ℝ} (h : orthogonal a (b m)) : magnitude (a.1 + b m.1, a.2 + b m.2) = sqrt 10 := by
  sorry

end magnitude_sum_is_sqrt_10_l464_464085


namespace ratio_of_boys_to_total_l464_464207

theorem ratio_of_boys_to_total (p_b p_g : ℝ) (h1 : p_b + p_g = 1) (h2 : p_b = (2 / 3) * p_g) :
  p_b = 2 / 5 :=
by
  sorry

end ratio_of_boys_to_total_l464_464207


namespace edward_can_buy_candies_l464_464334

theorem edward_can_buy_candies (whack_a_mole_tickets skee_ball_tickets candy_cost : ℕ)
  (h1 : whack_a_mole_tickets = 3) (h2 : skee_ball_tickets = 5) (h3 : candy_cost = 4) :
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost = 2 :=
by
  sorry

end edward_can_buy_candies_l464_464334


namespace rice_price_decrease_l464_464739

theorem rice_price_decrease (P : ℝ) (h1 : ∀ P : ℝ, P > 0) :
    let P_new := 20 * P / 25 in
    ((P - P_new) / P) * 100 = 20 :=
by
  sorry

end rice_price_decrease_l464_464739


namespace sum_of_digits_inequality_l464_464909

-- Assume that S(x) represents the sum of the digits of x in its decimal representation.
axiom sum_of_digits (x : ℕ) : ℕ

-- Given condition: for any natural numbers a and b, the sum of digits function satisfies the inequality
axiom sum_of_digits_add (a b : ℕ) : sum_of_digits (a + b) ≤ sum_of_digits a + sum_of_digits b

-- Theorem statement we want to prove
theorem sum_of_digits_inequality (k : ℕ) : sum_of_digits k ≤ 8 * sum_of_digits (8 * k) := 
  sorry

end sum_of_digits_inequality_l464_464909


namespace angle_bisectors_lemma_l464_464188

theorem angle_bisectors_lemma (A B C D E : Point) 
  (hABC : Ab = Bc)
  (hIsos_relations : A ≠ B) (C ≠ B) 
  (h_AE_bisector : IsAngleBisector A B E) 
  (h_CD_bisector : IsAngleBisector C B D) : 
  ∠BED E D B = 2 * ∠AED E D A :=
by
  sorry

end angle_bisectors_lemma_l464_464188


namespace delta_discount_percentage_l464_464850

theorem delta_discount_percentage (original_delta : ℝ) (original_united : ℝ)
  (united_discount_percent : ℝ) (savings : ℝ) (delta_discounted : ℝ) : 
  original_delta - delta_discounted = 0.2 * original_delta := by
  -- Given conditions
  let discounted_united := original_united * (1 - united_discount_percent / 100)
  have : delta_discounted = discounted_united - savings := sorry
  let delta_discount_amount := original_delta - delta_discounted
  have : delta_discount_amount = 0.2 * original_delta := sorry
  exact this

end delta_discount_percentage_l464_464850


namespace exchangeable_events_theorems_l464_464816

noncomputable def exchangeable_events (A : ℕ → set ℕ) (P : set ℕ → ℝ): Prop :=
  ∀ (n : ℕ) (i₁ i₂ : ℕ) (h₁ : i₁ ≤ i₂), 
    P (⋂ i < n, A i₁) = P (⋂ i < n, A i₂)

def p (A : ℕ → set ℕ) (P : set ℕ → ℝ) (n : ℕ) : ℝ :=
  P (⋂ i < n, A i)

noncomputable def delta (p : ℕ → ℝ) : ℕ → ℝ
| 0 := 1
| (n + 1) := p (n+1) - p n

theorem exchangeable_events_theorems
  (A : ℕ → set ℕ)
  (P : set ℕ → ℝ)
  (h1 : exchangeable_events A P) 
  (h2 : ∀ (n : ℕ), P (⋂ i < n, A i) = p A P n)
: (P (⋃ i < n, A i) = ∑ k in finset.range n, (-1)^k * (∫ _ in finset.range k, p A P k)) ∧
  (P (⋂ n, A n) = ∑ i in finset.range k, p A P k) ∧
  (P (⋃ n, A n) = 1 - ∑ i in finset.range k, delta (λ n => p A P n) n)
:= sorry

end exchangeable_events_theorems_l464_464816


namespace can_cut_into_equal_parts_l464_464160

-- We assume the existence of a shape S and some grid G along with a function cut
-- that cuts the shape S along grid G lines and returns two parts.
noncomputable def Shape := Type
noncomputable def Grid := Type
noncomputable def cut (S : Shape) (G : Grid) : Shape × Shape := sorry

-- We assume a function superimpose that checks whether two shapes can be superimposed
noncomputable def superimpose (S1 S2 : Shape) : Prop := sorry

-- Assume the given shape S and grid G
variable (S : Shape) (G : Grid)

-- The question rewritten as a Lean statement
theorem can_cut_into_equal_parts : ∃ (S₁ S₂ : Shape), cut S G = (S₁, S₂) ∧ superimpose S₁ S₂ := sorry

end can_cut_into_equal_parts_l464_464160


namespace cost_of_white_washing_l464_464820

-- Definitions for room dimensions, doors, windows, and cost per square foot
def length : ℕ := 25
def width : ℕ := 15
def height1 : ℕ := 12
def height2 : ℕ := 8
def door_height : ℕ := 6
def door_width : ℕ := 3
def window_height : ℕ := 4
def window_width : ℕ := 3
def cost_per_sq_ft : ℕ := 10
def ceiling_decoration_area : ℕ := 10

-- Definitions for the areas calculation
def area_walls_height1 : ℕ := 2 * (length * height1)
def area_walls_height2 : ℕ := 2 * (width * height2)
def total_wall_area : ℕ := area_walls_height1 + area_walls_height2

def area_one_door : ℕ := door_height * door_width
def total_doors_area : ℕ := 2 * area_one_door

def area_one_window : ℕ := window_height * window_width
def total_windows_area : ℕ := 3 * area_one_window

def adjusted_wall_area : ℕ := total_wall_area - total_doors_area - total_windows_area - ceiling_decoration_area

def total_cost : ℕ := adjusted_wall_area * cost_per_sq_ft

-- The theorem we want to prove
theorem cost_of_white_washing : total_cost = 7580 := by
  sorry

end cost_of_white_washing_l464_464820


namespace count_two_digit_primes_with_given_conditions_l464_464884

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def sum_of_digits_is_nine (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens + units = 9

def tens_greater_than_units (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens > units

theorem count_two_digit_primes_with_given_conditions :
  ∃ count : ℕ, count = 0 ∧ ∀ n, is_two_digit_prime n ∧ sum_of_digits_is_nine n ∧ tens_greater_than_units n → false :=
by
  -- proof goes here
  sorry

end count_two_digit_primes_with_given_conditions_l464_464884


namespace number_of_10_yuan_coins_is_1_l464_464751

theorem number_of_10_yuan_coins_is_1
  (n : ℕ) -- number of coins
  (v : ℕ) -- total value of coins
  (c1 c5 c10 c50 : ℕ) -- number of 1, 5, 10, and 50 yuan coins
  (h1 : n = 9) -- there are nine coins in total
  (h2 : v = 177) -- the total value of these coins is 177 yuan
  (h3 : c1 ≥ 1 ∧ c5 ≥ 1 ∧ c10 ≥ 1 ∧ c50 ≥ 1) -- at least one coin of each denomination
  (h4 : c1 + c5 + c10 + c50 = n) -- sum of all coins number is n
  (h5 : c1 * 1 + c5 * 5 + c10 * 10 + c50 * 50 = v) -- total value of all coins is v
  : c10 = 1 := 
sorry

end number_of_10_yuan_coins_is_1_l464_464751


namespace inequality_solution_l464_464721

theorem inequality_solution :
  {x : ℝ | (x + 2) / (x + 1)^2 < 0} = set.Iio (-2) :=
by
  sorry

end inequality_solution_l464_464721


namespace range_of_a_l464_464498

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 2) → (a ≤ -1 ∨ a ≥ 3) :=
by 
  sorry

end range_of_a_l464_464498


namespace trees_died_more_than_survived_l464_464508

theorem trees_died_more_than_survived :
  ∀ (initial_trees survived_percent : ℕ),
    initial_trees = 25 →
    survived_percent = 40 →
    (initial_trees * survived_percent / 100) + (initial_trees - initial_trees * survived_percent / 100) -
    (initial_trees * survived_percent / 100) = 5 :=
by
  intro initial_trees survived_percent initial_trees_eq survived_percent_eq
  sorry

end trees_died_more_than_survived_l464_464508


namespace pairs_of_integers_satisfying_equation_l464_464091

theorem pairs_of_integers_satisfying_equation :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 40} = { ⟨11, 9⟩, ⟨7, 3⟩ } :=
sorry

end pairs_of_integers_satisfying_equation_l464_464091


namespace find_lambda_l464_464310

-- Definitions of the lines and the circle
def original_line (λ : ℝ) (x y : ℝ) : Prop :=
  2 * x - y + λ = 0

def translated_line (λ : ℝ) (x y : ℝ) : Prop :=
  2 * (x + 1) - y + λ = 0

def circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y = 0

-- Center and radius after completing the square
def center : ℝ × ℝ := (-1, 2)
def radius : ℝ := real.sqrt 5

-- Distance from a point to a line calculation
def distance_to_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  |A * p.1 + B * p.2 + C| / real.sqrt (A^2 + B^2)

-- Prove that λ is -3 or 7
theorem find_lambda (λ : ℝ) :
  (∃ x y, original_line λ x y) ∧ (∃ x y, translated_line λ x y) ∧ (∃ x y, circle x y) ∧ 
  (∃ x y, distance_to_line center (2:ℝ) (-1:ℝ) (λ + 2) = radius) →
  λ = -3 ∨ λ = 7 :=
by
  sorry

end find_lambda_l464_464310


namespace sin_alpha_value_l464_464928

noncomputable def alpha : ℝ := 30 * Real.pi / 180  -- 30 degrees

def point : ℝ × ℝ := (2 * Real.sin alpha, 2 * Real.cos alpha)

def radius : ℝ := 2

theorem sin_alpha_value :
  ∃ (α : ℝ), point = (2 * Real.sin α, 2 * Real.cos α) ∧ Real.sin α = Real.cos alpha :=
by {
  use α,
  split,
  { -- Prove that the point (2 * sin 30°, 2 * cos 30°) is on the terminal side
    exact rfl,
  },
  { -- Prove that sin α = cos 30°
    exact (by norm_num : Real.sin α = Real.cos (30 * Real.pi / 180)).symm,
    sorry,
  },
}

end sin_alpha_value_l464_464928


namespace LS_parallel_PQ_l464_464638

universe u
variables {Point : Type u} [Geometry Point]

-- Defining points A, P, C, M, N, Q as elements of type Point
variables (A P C M N Q : Point)

-- Let L be the intersection point of the lines AP and CM
def L := Geometry.intersection (Geometry.line_through A P) (Geometry.line_through C M)

-- Let S be the intersection point of the lines AN and CQ
def S := Geometry.intersection (Geometry.line_through A N) (Geometry.line_through C Q)

-- Proof: LS is parallel to PQ
theorem LS_parallel_PQ : Geometry.parallel (Geometry.line_through L S) (Geometry.line_through P Q) :=
sorry

end LS_parallel_PQ_l464_464638


namespace f_at_47_l464_464044

noncomputable def f : ℝ → ℝ := sorry

axiom f_functional_equation : ∀ x : ℝ, f (x - 1) + f (x + 1) = 0
axiom f_interval_definition : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem f_at_47 : f 47 = -1 := by
  sorry

end f_at_47_l464_464044


namespace ratio_of_rises_in_cones_l464_464759

theorem ratio_of_rises_in_cones 
  (h₁ h₂ : ℝ) 
  (r₁ r₂ : ℝ) 
  (marble₁_volume marble₂_volume : ℝ)
  (h₁_initial h₂_initial : ℝ)
  (h₁_final h₂_final : ℝ)
  (initial_volume₁ : ℝ)
  (initial_volume₂ : ℝ)
  (final_volume₁ : ℝ)
  (final_volume₂ : ℝ) 
  (condition₁ : r₁ = 5)
  (condition₂ : r₂ = 10)
  (marble₁_radius : ℝ)
  (marble₂_radius : ℝ)
  (condition3 : marble₁_radius = 2)
  (condition4 : marble₂_radius = 3)
  (initial_volume_eq : initial_volume₁ = initial_volume₂)
  (height_ratio : h₁ = 4 * h₂)
  (marble₁_vol : marble₁_volume = (4/3) * Math.pi * marble₁_radius^3)
  (marble₂_vol : marble₂_volume = (4/3) * Math.pi * marble₂_radius^3)
  (initial_vol_1_def : initial_volume₁ = (1/3) * Math.pi * r₁^2 * h₁)
  (initial_vol_2_def : initial_volume₂ = (1/3) * Math.pi * r₂^2 * h₂)
  (final_vol_1_def : final_volume₁ = initial_volume₁ + marble₁_volume)
  (final_vol_2_def : final_volume₂ = initial_volume₂ + marble₂_volume)
  (h₁_final_def : final_volume₁ = (1/3) * Math.pi * r₁^2 * h₁_final)
  (h₂_final_def : final_volume₂ = (1/3) * Math.pi * r₂^2 * h₂_final)
  (delta_h₁ : h₁_final = h₁ + marble₁_volume / ( (1/3) * Math.pi * r₁^2))
  (delta_h₂ : h₂_final = h₂ + marble₂_volume / ( (1/3) * Math.pi * r₂^2))
  :
  marble₁_volume / ( (1/3) * Math.pi * r₁^2) / (marble₂_volume / ( (1/3) * Math.pi * r₂^2)) = 32 / 9 := 
by 
  sorry

end ratio_of_rises_in_cones_l464_464759


namespace problem_a_problem_b_l464_464183

-- Part (a)
theorem problem_a (n: Nat) : ∃ k: ℤ, (32^ (3 * n) - 1312^ n) = 1966 * k := sorry

-- Part (b)
theorem problem_b (n: Nat) : ∃ m: ℤ, (843^ (2 * n + 1) - 1099^ (2 * n + 1) + 16^ (4 * n + 2)) = 1967 * m := sorry

end problem_a_problem_b_l464_464183


namespace problem_statement_l464_464412

namespace ProofProblem

variable {x : ℝ}

def p : Prop := ∃ x0 : ℝ, 0 < x0 ∧ 9 * x0 = 6 - 1 / x0
def q : Prop := ∀ x : ℕ+, (x - 1) ^ 2 > 0

theorem problem_statement : ¬ p ∨ ¬ q :=
by
  have xp : ∃ (x0 : ℝ), 0 < x0 ∧ 9 * x0 = 6 - 1 / x0 := by
    use 1 / 3
    split
    · linarith 
    · field_simp [(ne_of_gt zero_lt_three).symm]
      ring
  have nq : ¬ ∀ (x : ℕ+), (x - 1) ^ 2 > 0 := by
    intro h
    specialize h 1
    norm_num at h
  exact Or.inr nq

end ProofProblem

end problem_statement_l464_464412


namespace LS_parallel_PQ_l464_464635

universe u
variables {Point : Type u} [Geometry Point]

-- Defining points A, P, C, M, N, Q as elements of type Point
variables (A P C M N Q : Point)

-- Let L be the intersection point of the lines AP and CM
def L := Geometry.intersection (Geometry.line_through A P) (Geometry.line_through C M)

-- Let S be the intersection point of the lines AN and CQ
def S := Geometry.intersection (Geometry.line_through A N) (Geometry.line_through C Q)

-- Proof: LS is parallel to PQ
theorem LS_parallel_PQ : Geometry.parallel (Geometry.line_through L S) (Geometry.line_through P Q) :=
sorry

end LS_parallel_PQ_l464_464635


namespace area_midpoint_quadrilateral_half_l464_464174

variable {A B C D : Type}
variables (AB BC CD DA : A)
variable [convex AB BC CD DA]
variable {M_AB M_BC M_CD M_DA : A}
variables (mid_AB : M_AB = (1/2) * (AB + BC))
variables (mid_BC : M_BC = (1/2) * (BC + CD))
variables (mid_CD : M_CD = (1/2) * (CD + DA))
variables (mid_DA : M_DA = (1/2) * (DA + AB))

theorem area_midpoint_quadrilateral_half (A B C D : Point) :
  area (quadrilateral (midpoint A B) (midpoint B C) (midpoint C D) (midpoint D A)) = 
  (1/2) * area (quadrilateral A B C D) :=
by
  sorry

end area_midpoint_quadrilateral_half_l464_464174


namespace parallel_LS_PQ_l464_464629

open_locale big_operators

variables (A P M N C Q L S : Type*)

-- Assume that L is the intersection of lines AP and CM
def is_intersection_L (AP CM : Set (Set Type*)) : Prop :=
  L ∈ AP ∩ CM

-- Assume that S is the intersection of lines AN and CQ
def is_intersection_S (AN CQ : Set (Set Type*)) : Prop :=
  S ∈ AN ∩ CQ

-- Prove that LS ∥ PQ
theorem parallel_LS_PQ 
  (hL : is_intersection_L A P M C L) 
  (hS : is_intersection_S A N C Q S) : 
  parallel (line_through L S) (line_through P Q) :=
sorry

end parallel_LS_PQ_l464_464629


namespace tan_pi_div4_plus_2a_cos_5pi_div6_minus_2a_l464_464457

variable (a : ℝ)
variable (ha : a ∈ (Real.pi / 2, Real.pi))
variable (hsa : Real.sin a = Real.sqrt 5 / 5)

theorem tan_pi_div4_plus_2a : 
  Real.tan (Real.pi / 4 + 2 * a) = -1 / 7 := by
  sorry

theorem cos_5pi_div6_minus_2a : 
  Real.cos (5 * Real.pi / 6 - 2 * a) = -((3 * Real.sqrt 3 + 4) / 10) := by
  sorry

end tan_pi_div4_plus_2a_cos_5pi_div6_minus_2a_l464_464457


namespace permutation_partial_sums_modulo_l464_464032

open Nat

theorem permutation_partial_sums_modulo (m : ℕ) (h : m > 1) :
  (∃ (a : Fin m → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ m) ∧ (∀ i j, i ≠ j → a i ≠ a j) ∧ 
  (∃ (b : Fin m → ℕ), ∀ k, b k = (∑ i in finRange (k+1), a (i : Fin m)) % m ∧
  ∀ i j, i ≠ j → b i ≠ b j)) ↔ Even m :=
by
  sorry

end permutation_partial_sums_modulo_l464_464032


namespace anna_coaching_days_l464_464830

-- Define the number of days in each month for a non-leap year.
def daysInMonth : ℕ → ℕ
| 1  := 31  -- January
| 2  := 28  -- February
| 3  := 31  -- March
| 4  := 30  -- April
| 5  := 31  -- May
| 6  := 30  -- June
| 7  := 31  -- July
| 8  := 31  -- August
| 9  := 4   -- Up to September 4th
| _  := 0   -- All other months set to 0 to be safe

-- Define a function to sum the days from January 1st to September 4th.
def totalDaysCoaching : ℕ :=
  (List.range 9).map (λ m => daysInMonth (m + 1)).sum

-- The theorem to prove that Anna took coaching for 247 days.
theorem anna_coaching_days : totalDaysCoaching = 247 := by
  sorry

end anna_coaching_days_l464_464830


namespace not_possible_100_odd_sequence_l464_464161

def is_square_mod_8 (n : ℤ) : Prop :=
  n % 8 = 0 ∨ n % 8 = 1 ∨ n % 8 = 4

def sum_consecutive_is_square_mod_8 (seq : List ℤ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i + k ≤ seq.length →
  is_square_mod_8 (seq.drop i |>.take k |>.sum)

def valid_odd_sequence (seq : List ℤ) : Prop :=
  seq.length = 100 ∧
  (∀ n ∈ seq, n % 2 = 1) ∧
  sum_consecutive_is_square_mod_8 seq 5 ∧
  sum_consecutive_is_square_mod_8 seq 9

theorem not_possible_100_odd_sequence :
  ¬∃ seq : List ℤ, valid_odd_sequence seq :=
by
  sorry

end not_possible_100_odd_sequence_l464_464161


namespace kids_meeting_restrictions_l464_464872

-- Define the number of kids in each school
def Riverside_kids : ℕ := 150
def WestSide_kids : ℕ := 100
def Mountaintop_kids : ℕ := 80
def OakGrove_kids : ℕ := 60
def Lakeview_kids : ℕ := 110

-- Define the percentages of kids who meet the age restrictions
def Riverside_percent : ℝ := 0.20
def WestSide_percent : ℝ := 0.40
def Mountaintop_percent : ℝ := 0.30
def OakGrove_percent : ℝ := 0.50
def Lakeview_percent : ℝ := 0.35

-- Calculate the number of kids meeting the age restriction
def Riverside_age : ℕ := Real.to_nat (Riverside_percent * Riverside_kids)
def WestSide_age : ℕ := Real.to_nat (WestSide_percent * WestSide_kids)
def Mountaintop_age : ℕ := Real.to_nat (Mountaintop_percent * Mountaintop_kids)
def OakGrove_age : ℕ := Real.to_nat (OakGrove_percent * OakGrove_kids)
def Lakeview_age : ℕ := Real.to_nat (Lakeview_percent * Lakeview_kids)

-- Add up all the results
def total_kids_meeting_restrictions : ℕ :=
  Riverside_age + WestSide_age + Mountaintop_age + OakGrove_age + Lakeview_age

-- Theorem to prove the total number of kids meeting the restrictions is 162
theorem kids_meeting_restrictions : total_kids_meeting_restrictions = 162 := by
  sorry

end kids_meeting_restrictions_l464_464872


namespace sqrt_x_plus_4_eq_3_then_squared_result_l464_464531

theorem sqrt_x_plus_4_eq_3_then_squared_result {x : ℝ} (h : sqrt (x + 4) = 3) : (x + 4) ^ 2 = 81 :=
by
  -- We would write the proof steps here, which is optional as per the instruction
  sorry

end sqrt_x_plus_4_eq_3_then_squared_result_l464_464531


namespace parallel_ls_pq_l464_464611

variables {A P C M N Q L S : Type*}
variables [affine_space ℝ (A P C M N Q)]

/- Definitions of the intersection points -/
def is_intersection_point (L : Type*) (AP : set (A P))
  (CM : set (C M)) : Prop := L ∈ AP ∧ L ∈ CM

def is_intersection_point' (S : Type*) (AN : set (A N))
  (CQ : set (C Q)) : Prop := S ∈ AN ∧ S ∈ CQ

/- Main Theorem -/
theorem parallel_ls_pq (AP CM AN CQ : set (A P))
  (PQ : set (P Q)) (L S : Type*)
  (hL : is_intersection_point L AP CM)
  (hS : is_intersection_point' S AN CQ) : L S ∥ P Q :=
sorry

end parallel_ls_pq_l464_464611


namespace correct_statement_about_residuals_l464_464773

-- Define the properties and characteristics of residuals as per the definition
axiom residuals_definition : Prop
axiom residuals_usefulness : residuals_definition → Prop

-- The theorem to prove that the correct statement about residuals is that they can be used to assess the effectiveness of model fitting
theorem correct_statement_about_residuals (h : residuals_definition) : residuals_usefulness h :=
sorry

end correct_statement_about_residuals_l464_464773


namespace find_phi_if_odd_function_and_phi_in_interval_l464_464917

noncomputable def phi : ℝ := sorry

theorem find_phi_if_odd_function_and_phi_in_interval
  (h1 : φ ∈ Ioo 0 real.pi)
  (h2 : ∀ x : ℝ, cos (2 * -x + φ) = - cos (2 * x + φ)) :
  φ = real.pi / 2 :=
sorry

end find_phi_if_odd_function_and_phi_in_interval_l464_464917


namespace inequality_has_exactly_two_integer_solutions_l464_464885

theorem inequality_has_exactly_two_integer_solutions (a : ℝ) :
  ((ax - 1)^2 < x^2) has exactly 2 integer solutions for x ↔ a ∈ set.Icc (4/3) (3/2) ∨ a ∈ set.Ico (-3/2) (-4/3) :=
sorry

end inequality_has_exactly_two_integer_solutions_l464_464885


namespace probability_point_on_hyperbola_l464_464233

theorem probability_point_on_hyperbola :
  let S := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) },
      Hyperbola := { (x, y) | y = 6 / x },
      Favourable := S ∩ Hyperbola
  in
    Favourable.card / S.card = 1 / 3 :=
by
  sorry

end probability_point_on_hyperbola_l464_464233


namespace incorrect_judgment_l464_464896

theorem incorrect_judgment : (∀ x : ℝ, x^2 - 1 ≥ -1) ∧ (4 + 2 ≠ 7) :=
by 
  sorry

end incorrect_judgment_l464_464896


namespace card_at_52_is_10_l464_464419

theorem card_at_52_is_10 :
  let sequence := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "Joker"] in
  sequence[(52 % sequence.length)] = "10" :=
by
  sorry

end card_at_52_is_10_l464_464419


namespace ls_parallel_pq_l464_464608

-- Declare the points and lines
variables {A P C M N Q L S : Type*}

-- Assume given conditions as definitions
def intersection (l1 l2 : set (Type*)) : Type* := sorry -- Define intersection point of l1 and l2 (details omitted for simplicity)

-- Define L as intersection of lines AP and CM
def L : Type* := intersection (set_of (λ x, ∃ L, ∃ P, ∃ C, ∃ M, L = x ∧ x ∈ AP ∧ x ∈ CM)) (λ x, x ∈ AP ∧ x ∈ CM)

-- Define S as intersection of lines AN and CQ
def S : Type* := intersection (set_of (λ x, ∃ S, ∃ N, ∃ C, ∃ Q, S = x ∧ x ∈ AN ∧ x ∈ CQ)) (λ x, x ∈ AN ∧ x ∈ CQ)

-- Statement to prove LS is parallel to PQ
theorem ls_parallel_pq (AP CM AN CQ PQ : set (Type*)) (L : Type*) (S : Type*) :
    L = intersection (AP) (CM) ∧ S = intersection (AN) (CQ) → is_parallel LS PQ :=
by
  sorry -- Proof omitted

end ls_parallel_pq_l464_464608


namespace three_pairs_with_same_difference_l464_464775

theorem three_pairs_with_same_difference (X : Finset ℕ) (hX₁ : X ⊆ {n | n ∈ Finset.range 18}) (hX₂ : X.card = 8) : 
  ∃ a b c d e f : ℕ, a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ a - b = c - d ∧ c - d = e - f :=
sorry

end three_pairs_with_same_difference_l464_464775


namespace trigonometry_problem_l464_464979

theorem trigonometry_problem
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : a = 5)
  (h2 : b = 4)
  (h3 : cos (A - B) = 31 / 32)
  (h4 : a = b * sin A / sin B )  -- Law of Sines
  (h5 : A + B + C = pi)          -- Sum of angles in a triangle

  : sin B = (sqrt 7) / 4 ∧ cos C = 1 / 8 :=
sorry

end trigonometry_problem_l464_464979


namespace tan_theta_value_l464_464057

variable (θ : ℝ)

def sin_theta := 4 / 5
def angle_in_second_quadrant : Prop := (π / 2 < θ) ∧ (θ < π)

theorem tan_theta_value (h1 : real.sin θ = sin_theta) (h2 : angle_in_second_quadrant):
  real.tan θ = -4 / 3 := 
sorry

end tan_theta_value_l464_464057


namespace kenny_played_basketball_last_week_l464_464583

def time_practicing_trumpet : ℕ := 40
def time_running : ℕ := time_practicing_trumpet / 2
def time_playing_basketball : ℕ := time_running / 2
def answer : ℕ := 10

theorem kenny_played_basketball_last_week :
  time_playing_basketball = answer :=
by
  -- sorry to skip the proof
  sorry

end kenny_played_basketball_last_week_l464_464583


namespace gold_hammer_weight_l464_464558

theorem gold_hammer_weight (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m)
  (h_a1 : a 1 = 4) 
  (h_a5 : a 5 = 2) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := 
sorry

end gold_hammer_weight_l464_464558


namespace three_digit_special_count_l464_464521

theorem three_digit_special_count : 
  let count := ∑ A in finset.range 9 \ finset.singleton 0, (finset.range (10 - A)).filter (λ B, B ≠ A).card
  count = 36 := 
by
  sorry

end three_digit_special_count_l464_464521


namespace dog_travel_distance_l464_464337

-- Definitions based on the conditions
-- Total travel time (hours)
def total_time : ℝ := 2

-- Speeds (km/h)
def speed_first_half : ℝ := 10
def speed_second_half : ℝ := 5

-- Distance covered in each half
def distance_first_half (D : ℝ) : ℝ := D / 2
def distance_second_half (D : ℝ) : ℝ := D / 2

-- Time taken for each half
def time_first_half (D : ℝ) : ℝ := (distance_first_half D) / speed_first_half
def time_second_half (D : ℝ) : ℝ := (distance_second_half D) / speed_second_half

-- Proof problem statement
theorem dog_travel_distance : 
  ∃ (D : ℝ), time_first_half D + time_second_half D = total_time ∧ D = 40 / 3 :=
by
  sorry

end dog_travel_distance_l464_464337


namespace circle_x_intersect_l464_464263

theorem circle_x_intersect (x y : ℝ) : 
  (x, y) = (0, 0) ∨ (x, y) = (10, 0) → (x = 10) :=
by
  -- conditions:
  -- The endpoints of the diameter are (0,0) and (10,10)
  -- (proving that the second intersect on x-axis has x-coordinate 10)
  sorry

end circle_x_intersect_l464_464263


namespace mary_has_82_cards_l464_464688

def initial_cards : Nat := 18
def torn_cards : Nat := 8
def repair_percent : Rat := 0.75
def repaired_cards : Nat := (repair_percent * torn_card).toNat
def freds_cards : Nat := 26
def bought_cards : Nat := 40

def total_cards : Nat :=
  let cards_after_repair := initial_cards - torn_cards + repaired_cards
  let cards_after_freds := cards_after_repair + freds_cards
  cards_after_freds + bought_cards

theorem mary_has_82_cards : total_cards = 82 := sorry

end mary_has_82_cards_l464_464688


namespace shape_of_constant_phi_is_cone_l464_464143

-- Define spherical coordinates angles
variable (ρ θ φ : ℝ)

-- Define a constant angle c
variable (c : ℝ)

-- Mathematical problem to prove that the shape described by φ = c is a cone
theorem shape_of_constant_phi_is_cone (ρ θ : ℝ) (h: φ = c) : shape ρ θ φ = cone :=
sorry

end shape_of_constant_phi_is_cone_l464_464143


namespace probability_on_hyperbola_l464_464237

open Finset

-- Define the function for the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the set of different number pairs from {1, 2, 3}
def pairs : Finset (ℕ × ℕ) := 
  {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}.to_finset

-- Define the set of pairs that lie on the hyperbola
def hyperbola_pairs : Finset (ℕ × ℕ) :=
  pairs.filter (λ mn, on_hyperbola mn.1 mn.2)

-- The theorem to prove the probability
theorem probability_on_hyperbola : 
  (hyperbola_pairs.card : ℝ) / (pairs.card : ℝ) = 1 / 3 :=
by
  -- Placeholder for the proof
  sorry

end probability_on_hyperbola_l464_464237


namespace number_of_trivial_subsets_l464_464414

open Set

def A : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2017}

def is_trivial_set (s: Set ℕ) : Prop :=
  (∑ x in s, x ^ 2) % 2 = 1

theorem number_of_trivial_subsets :
  ∃ n : ℕ, n = 2^2016 - 1 ∧ (finset.image Set.toFinset (powerset A)).filter is_trivial_set = n := 
by {
  sorry
}

end number_of_trivial_subsets_l464_464414


namespace bat_pattern_area_l464_464725

-- Define the areas of the individual components
def area_large_square : ℕ := 8
def num_large_squares : ℕ := 2

def area_medium_square : ℕ := 4
def num_medium_squares : ℕ := 2

def area_triangle : ℕ := 1
def num_triangles : ℕ := 3

-- Define the total area calculation
def total_area : ℕ :=
  (num_large_squares * area_large_square) +
  (num_medium_squares * area_medium_square) +
  (num_triangles * area_triangle)

-- The theorem statement
theorem bat_pattern_area : total_area = 27 := by
  sorry

end bat_pattern_area_l464_464725


namespace pen_cost_num_pens_l464_464870

variable (n_price : ℝ) (p_price : ℝ)
variable (initial_amount : ℝ)
variable (leftover_after_three_notebooks : ℝ)
variable (additional_money : ℝ)

noncomputable def calc_initial_amt (n_price leftover : ℝ) : ℝ :=
  3 * n_price + leftover

noncomputable def calc_pen_cost (n_price additional expense_added amount_added pens : ℝ) : ℝ :=
  (calc_initial_amt n_price leftover_after_three_notebooks + additional - expense * amount) / num_pens

noncomputable def calc_num_pens (n_price p_price initial_amount : ℝ) : ℝ :=
  (initial_amount - 2 * n_price) / p_price

theorem pen_cost (h_initial: initial_amount = 22)
  (h_leftover: leftover_after_three_notebooks = 4)
  (h_n_price: n_price = 6)
  (h_additional: additional_money = 4)
  (h_pens_7_cost: calc_pen_cost n_price additional_money 2 initial_amount 7 = p_price)
  : p_price = 2 := sorry

theorem num_pens (h_initial: initial_amount = 22)
  (h_n_price: n_price = 6)
  (h_p_price: p_price = 2)
  : calc_num_pens n_price p_price initial_amount = 5 := sorry

end pen_cost_num_pens_l464_464870


namespace midpoints_not_collinear_l464_464296

variable {α : Type*} [LinearOrderedField α]
variables {A B C A1 B1 C1 M_A M_B M_C : EuclideanGeometry.Point α}

def is_midpoint (P Q M : EuclideanGeometry.Point α) : Prop :=
  ∃ (t : α), 0 < t ∧ t < 1 ∧ t • (Q - P) + P = M

theorem midpoints_not_collinear
  (hA1 : EuclideanGeometry.Between A B C)
  (hB1 : EuclideanGeometry.Between B A C)
  (hC1 : EuclideanGeometry.Between C A B)
  (hMid_A : is_midpoint A A1 M_A)
  (hMid_B : is_midpoint B B1 M_B)
  (hMid_C : is_midpoint C C1 M_C) :
  ¬ EuclideanGeometry.Collinear ({M_A, M_B, M_C} : set (EuclideanGeometry.Point α)) :=
sorry

end midpoints_not_collinear_l464_464296


namespace max_primary_school_students_served_l464_464788

theorem max_primary_school_students_served :
  ∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ 3 * x + 5 * y ≤ 37 ∧ y ≥ x + 1 ∧ 5 * x + 3 * y = 35 :=
begin
  use [4, 5],
  sorry, -- Proof omitted
end

end max_primary_school_students_served_l464_464788


namespace simplify_expression_l464_464847

variable (x : ℝ)

theorem simplify_expression : (x + 2)^2 - (x + 1) * (x + 3) = 1 := 
by 
  sorry

end simplify_expression_l464_464847


namespace correct_answer_l464_464345

-- Statement of the problem
theorem correct_answer :
  ∃ (answer : String),
    (answer = "long before" ∨ answer = "before long" ∨ answer = "soon after" ∨ answer = "shortly after") ∧
    answer = "long before" :=
by
  sorry

end correct_answer_l464_464345


namespace consecutive_sum_ways_l464_464551

theorem consecutive_sum_ways : 
  (∃ k (k ≥ 2) (k ∣ 420) (2 * 420 = k * (2 * n + k - 1)), nat), 
  n = (420 / k) - ((k - 1) / 2) ∧ k > 1 ∧ nat.succ_pred k = 7 
  := sorry

end consecutive_sum_ways_l464_464551


namespace number_of_distinguishable_large_triangles_is_960_l464_464302

-- Define the number of different colors available.
def num_colors : ℕ := 8

-- Define the number of distinguishable large triangles using conditions provided.
noncomputable def num_distinguishable_large_triangles : ℕ :=
  let num_combinations := 8 + 8 * 7 + nat.choose 8 3
  8 * num_combinations

-- State the theorem to be proved.
theorem number_of_distinguishable_large_triangles_is_960 :
  num_distinguishable_large_triangles = 960 :=
by sorry

end number_of_distinguishable_large_triangles_is_960_l464_464302


namespace LS_parallel_PQ_l464_464639

universe u
variables {Point : Type u} [Geometry Point]

-- Defining points A, P, C, M, N, Q as elements of type Point
variables (A P C M N Q : Point)

-- Let L be the intersection point of the lines AP and CM
def L := Geometry.intersection (Geometry.line_through A P) (Geometry.line_through C M)

-- Let S be the intersection point of the lines AN and CQ
def S := Geometry.intersection (Geometry.line_through A N) (Geometry.line_through C Q)

-- Proof: LS is parallel to PQ
theorem LS_parallel_PQ : Geometry.parallel (Geometry.line_through L S) (Geometry.line_through P Q) :=
sorry

end LS_parallel_PQ_l464_464639


namespace shaded_region_area_correct_l464_464319

noncomputable def radius : ℝ := 15
noncomputable def theta : ℝ := 45 * (Real.pi / 180)
noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ := (θ / (2 * Real.pi)) * Real.pi * r^2
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Definitions from problem conditions
def area_of_sector := sector_area radius theta
def area_of_triangle := equilateral_triangle_area radius

-- Result to prove
def shaded_region_area := 2 * (area_of_sector - area_of_triangle)

-- The main theorem
theorem shaded_region_area_correct : shaded_region_area = 56.25 * Real.pi - 112.5 * Real.sqrt 3 :=
by
  sorry

end shaded_region_area_correct_l464_464319


namespace rational_roots_count_l464_464361

theorem rational_roots_count (b₃ b₂ b₁ : ℤ) :
  (∀ x : ℚ, (x ≠ 0) → (4 * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 21 = 0 → 
                     (∃ p q : ℤ, p ∣ 21 ∧ q ∣ 4 ∧ x = (p : ℚ) / q))) →
  ((finset.univ.filter (λ x : ℚ, 4 * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 21 = 0)).card = 24) :=
sorry

end rational_roots_count_l464_464361


namespace point_on_hyperbola_probability_l464_464226

theorem point_on_hyperbola_probability :
  let s := ({1, 2, 3} : Finset ℕ) in
  let p := ∑ x in s.sigma (λ x, s.filter (λ y, y ≠ x)),
             if (∃ m n, x = (m, n) ∧ n = (6 / m)) then 1 else 0 in
  p / (s.card * (s.card - 1)) = (1 / 3) :=
by
  -- Conditions and setup
  let s := ({1, 2, 3} : Finset ℕ)
  let t := s.sigma (λ x, s.filter (λ y, y ≠ x))
  let p := t.filter (λ (xy : ℕ × ℕ), xy.snd = 6 / xy.fst)
  have h_total : t.card = 6, by sorry
  have h_count : p.card = 2, by sorry

  -- Calculate probability
  calc
    ↑(p.card) / ↑(t.card) = 2 / 6 : by sorry
    ... = 1 / 3 : by norm_num

end point_on_hyperbola_probability_l464_464226


namespace h_inverse_is_correct_l464_464193

def f (x : ℝ) : ℝ := 4 * x + 5
def g (x : ℝ) : ℝ := 3 * x - 4
def h (x : ℝ) : ℝ := f (g x)

def h_inv (x : ℝ) : ℝ := (x + 11) / 12

theorem h_inverse_is_correct : ∀ x, h (h_inv x) = x ∧ h_inv (h x) = x :=
by
  sorry

end h_inverse_is_correct_l464_464193


namespace LS_parallel_PQ_l464_464594

noncomputable def L (A P C M : Point) : Point := 
  -- Intersection of AP and CM
  sorry

noncomputable def S (A N C Q : Point) : Point := 
  -- Intersection of AN and CQ
  sorry

theorem LS_parallel_PQ (A P C M N Q : Point) 
  (hL : L A P C M) (hS : S A N C Q) : 
  Parallel (Line (L A P C M)) (Line (S A N C Q)) :=
sorry

end LS_parallel_PQ_l464_464594


namespace quadratic_real_roots_count_l464_464853

theorem quadratic_real_roots_count :
  (finset.filter (λ bc : ℕ × ℕ, bc.fst^2 - 4 * (bc.fst - 1) * bc.snd ≥ 0)
    ((fin_range 7).product (fin_range 7))).card = 28 :=
sorry

end quadratic_real_roots_count_l464_464853


namespace sin_alpha_plus_2beta_l464_464111

variable {α β : ℝ}
variables (h1 : α < π / 2) (h2 : β < π / 2)
variables (h3 : cos (α + β) = -5 / 13) (h4 : sin β = 3 / 5)

theorem sin_alpha_plus_2beta :
  sin (α + 2 * β) = 33 / 65 :=
by
  sorry

end sin_alpha_plus_2beta_l464_464111


namespace find_acute_angles_of_triangle_l464_464138

-- Given a right triangle ABC with ∠ACB = 90°
variables (A B C H N M : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty H] [Nonempty N] [Nonempty M]
variables (r s t : ℝ)

-- Assume ∠BAC + ∠ABC = 90° (not explicitly given, but true for any right triangle and is derived) 
-- and altitude CH from C to AB
def right_triangle (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C] : Prop :=
  let ∠ACB := 90 in
  r + s = 90

-- Altitude CH
def altitude_CH (A B C H : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty H] : Prop :=
  true  -- this needs refinement to express CH's properties, let's assume it's correct.

-- From point N on leg BC, a perpendicular NM is dropped to the hypotenuse AB
def perpendicular_NM (B C N M : Type) [Nonempty B] [Nonempty C] [Nonempty N] [Nonempty M] : Prop := 
  true  -- assume construction correctness for simplicity

-- The line NA is perpendicular to CM:
def perpendicular_NA_CM (A C N M : Type) [Nonempty A] [Nonempty C] [Nonempty N] [Nonempty M] : Prop := 
  true  -- assume construction correctness for simplicity

-- Ratio MH:CH = 1:√3
def ratio_MH_CH (M H : Type) [Nonempty M] [Nonempty H] (x : ℝ) : Prop :=
  x / (x * real.sqrt 3) = 1 / real.sqrt 3

-- Lean statement to conclude acute angles of triangle ABC
theorem find_acute_angles_of_triangle (A B C H N M : Type) [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty H] [Nonempty N] [Nonempty M] 
  (r s t : ℝ) (x : ℝ) 
  (ht1 : right_triangle A B C)
  (ht2 : altitude_CH A B C H)
  (ht3 : perpendicular_NM B C N M)
  (ht4 : perpendicular_NA_CM A C N M)
  (ht5 : ratio_MH_CH M H x) 
: r = 30 ∧ s = 60 := 
by 
  sorry  -- skipping the proof for now

end find_acute_angles_of_triangle_l464_464138


namespace parallel_ls_pq_l464_464610

variables {A P C M N Q L S : Type*}
variables [affine_space ℝ (A P C M N Q)]

/- Definitions of the intersection points -/
def is_intersection_point (L : Type*) (AP : set (A P))
  (CM : set (C M)) : Prop := L ∈ AP ∧ L ∈ CM

def is_intersection_point' (S : Type*) (AN : set (A N))
  (CQ : set (C Q)) : Prop := S ∈ AN ∧ S ∈ CQ

/- Main Theorem -/
theorem parallel_ls_pq (AP CM AN CQ : set (A P))
  (PQ : set (P Q)) (L S : Type*)
  (hL : is_intersection_point L AP CM)
  (hS : is_intersection_point' S AN CQ) : L S ∥ P Q :=
sorry

end parallel_ls_pq_l464_464610


namespace negate_exists_implies_forall_l464_464278

-- Define the original proposition
def prop1 (x : ℝ) : Prop := x^2 + 2 * x + 2 < 0

-- The negation of the proposition
def neg_prop1 := ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0

-- Statement of the equivalence
theorem negate_exists_implies_forall :
  ¬(∃ x : ℝ, prop1 x) ↔ neg_prop1 := by
  sorry

end negate_exists_implies_forall_l464_464278


namespace find_angle_B_l464_464132

def is_triangle (A B C : ℝ) : Prop :=
A + B > C ∧ B + C > A ∧ C + A > B

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Defining the problem conditions
lemma given_condition : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a := sorry
-- A triangle with sides a, b, c
lemma triangle_property : is_triangle a b c := sorry

-- The equivalent proof problem
theorem find_angle_B (h_triangle : is_triangle a b c) (h_cond : 2 * b * Real.cos A = 2 * c - Real.sqrt 3 * a) : 
    B = π / 6 := sorry

end find_angle_B_l464_464132


namespace share_of_y_is_63_l464_464813

theorem share_of_y_is_63 (x y z : ℝ) (h1 : y = 0.45 * x) (h2 : z = 0.50 * x) (h3 : x + y + z = 273) : y = 63 :=
by
  -- The proof will go here
  sorry

end share_of_y_is_63_l464_464813


namespace tangent_line_at_1_tangent_line_through_2_3_l464_464936

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Derivative of the function
def f' (x : ℝ) : ℝ := 3 * x^2

-- Problem 1: Prove that the tangent line at point (1, 1) is y = 3x - 2
theorem tangent_line_at_1 (x y : ℝ) (h : y = f 1 + f' 1 * (x - 1)) : y = 3 * x - 2 := 
sorry

-- Problem 2: Prove that the tangent line passing through (2/3, 0) is either y = 0 or y = 3x - 2
theorem tangent_line_through_2_3 (x y x0 : ℝ) 
  (hx0 : y = f x0 + f' x0 * (x - x0))
  (hp : 0 = f' x0 * (2/3 - x0)) :
  y = 0 ∨ y = 3 * x - 2 := 
sorry

end tangent_line_at_1_tangent_line_through_2_3_l464_464936


namespace combined_diameter_l464_464330

theorem combined_diameter (magnification : ℝ) 
  (d₁ d₂ d₃ : ℝ) 
  (h₁ : magnification = 1000) 
  (h₂ : d₁ = 1) 
  (h₃ : d₂ = 1.5) 
  (h₄ : d₃ = 2) :
  (d₁ / magnification + d₂ / magnification + d₃ / magnification) = 0.0045 :=
by 
  have h₅ : d₁ / magnification = 0.001 := by sorry
  have h₆ : d₂ / magnification = 0.0015 := by sorry
  have h₇ : d₃ / magnification = 0.002 := by sorry
  calc
    (d₁ / magnification + d₂ / magnification + d₃ / magnification)
        = 0.001 + 0.0015 + 0.002 : by rw [h₅, h₆, h₇]
    ... = 0.0045 : by norm_num

end combined_diameter_l464_464330


namespace michael_current_chickens_l464_464689

-- Defining variables and constants
variable (initial_chickens final_chickens annual_increase : ℕ)

-- Given conditions
def chicken_increase_condition : Prop :=
  final_chickens = initial_chickens + annual_increase * 9

-- Question to answer
def current_chickens (final_chickens annual_increase : ℕ) : ℕ :=
  final_chickens - annual_increase * 9

-- Proof problem
theorem michael_current_chickens
  (initial_chickens : ℕ)
  (final_chickens : ℕ)
  (annual_increase : ℕ)
  (h1 : chicken_increase_condition final_chickens initial_chickens annual_increase) :
  initial_chickens = 550 :=
by
  -- Formal proof would go here.
  sorry

end michael_current_chickens_l464_464689


namespace probability_correct_match_unlabeled_photos_l464_464824

/-- Given five celebrities and their photos from when they were teenagers, with exactly three photos correctly labeled and two unlabeled, prove that the probability of randomly guessing the correct match for both unlabeled photos is 1/20. -/
theorem probability_correct_match_unlabeled_photos :
  let celebrities : Fin 5 := sorry,
      photos : Fin 5 := sorry,
      correctly_labeled : Fin 3 := sorry,
      unlabeled : Fin 2 := sorry,
      total_permutations := Nat.choose 5 2 * 2! in
  1 / total_permutations = 1 / 20 := by
  sorry

end probability_correct_match_unlabeled_photos_l464_464824


namespace point_location_l464_464898

noncomputable def z1 : ℂ := 3 - 4 * complex.i
noncomputable def z2 : ℂ := -2 + 3 * complex.i

def complex_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def quadrant (p : ℝ × ℝ) : string :=
  if p.1 > 0 ∧ p.2 > 0 then "first quadrant"
  else if p.1 < 0 ∧ p.2 > 0 then "second quadrant"
  else if p.1 < 0 ∧ p.2 < 0 then "third quadrant"
  else if p.1 > 0 ∧ p.2 < 0 then "fourth quadrant"
  else "on an axis"

theorem point_location :
  quadrant (complex_point (z1 - z2)) = "fourth quadrant" :=
by
  sorry

end point_location_l464_464898


namespace roots_sum_of_squares_of_pairs_l464_464184

noncomputable def roots (p q r : ℝ) : Prop := 
  ∃ (p q r : ℝ), (p, q, r ∈ Roots (x^3 - 15*x^2 + 25*x - 10))

theorem roots_sum_of_squares_of_pairs (p q r : ℝ) (h : roots p q r) : 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by
  sorry

end roots_sum_of_squares_of_pairs_l464_464184


namespace find_intersection_l464_464083

noncomputable def setM : Set ℝ := {x : ℝ | x^2 ≤ 9}
noncomputable def setN : Set ℝ := {x : ℝ | x ≤ 1}
noncomputable def intersection : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 1}

theorem find_intersection (x : ℝ) : (x ∈ setM ∧ x ∈ setN) ↔ (x ∈ intersection) := 
by sorry

end find_intersection_l464_464083


namespace polynomial_integer_condition_l464_464006

theorem polynomial_integer_condition (P : ℝ → ℝ) (hP : ∀ x, is_polynomial_with_integer_coefficients P x) :
  (∀ s t : ℝ, (P s ∈ ℤ) → (P t ∈ ℤ) → (P (s * t) ∈ ℤ)) →
  (∃ (n : ℕ) (k : ℤ), P = λ x, x^n + k ∨ P = λ x, -x^n + k) :=
by
  sorry

end polynomial_integer_condition_l464_464006


namespace angle_bisector_of_tangent_circles_l464_464754

/-
We define the given conditions:
- Two circles are internally tangent at a point A.
- A tangent line to the inner circle at B intersects the outer circle at points C and D.
Then we state that AB bisects the angle CAD.
-/

-- Definitions of circles and tangency
variable (ω1 ω2 : Circle) (A B C D : Point)

-- Conditions
variable (h1 : ω1.TangentAtOverlap ω2 A)
variable (h2 : TangentLine ω1 B (Line.mk B (Line.tangentAt B ω1)))
variable (h3 : IntersectOuterCircle ω2 (Line.mk B (Line.tangentAt B ω1)) C D)

-- Theorem statement
theorem angle_bisector_of_tangent_circles
  (h1 : ω1.TangentAtOverlap ω2 A)
  (h2 : TangentLine ω1 B (Line.mk B (Line.tangentAt B ω1)))
  (h3 : IntersectOuterCircle ω2 (Line.mk B (Line.tangentAt B ω1)) C D) :
  AngleBisector (Segment.mk A B) (Angle.mk C A D) :=
sorry

end angle_bisector_of_tangent_circles_l464_464754


namespace roots_sum_of_squares_of_pairs_l464_464185

noncomputable def roots (p q r : ℝ) : Prop := 
  ∃ (p q r : ℝ), (p, q, r ∈ Roots (x^3 - 15*x^2 + 25*x - 10))

theorem roots_sum_of_squares_of_pairs (p q r : ℝ) (h : roots p q r) : 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 :=
by
  sorry

end roots_sum_of_squares_of_pairs_l464_464185


namespace range_of_m_length_of_chord_l464_464490

-- Part 1: Range of values of m for the circle to exist
theorem range_of_m (m : ℝ) : (∀ x y : ℝ, x^2 + y^2 - 2*x + 2*y + m - 3 = 0 → m < 7) :=
sorry

-- Part 2: Length of the chord for m = 1
theorem length_of_chord (x y : ℝ) : (x^2 + y^2 - 2*x + 2*y - 2 = 0) → 
  (∀ x y : ℝ, 4*x - 4*y - 4^2 = 0 → 2 * (2 ((2 * 2 - 2) * 2))^2 = 2 * sqrt(2).to_nat) :=
sorry

end range_of_m_length_of_chord_l464_464490


namespace problem_solution_l464_464680

-- Define the double factorial for odd and even cases.
def double_factorial : ℕ → ℕ 
| 0 => 1
| 1 => 1
| k => k * double_factorial (k - 2)

def S : ℚ :=
  ∑ i in (Finset.range (1005 + 1)).map (Function.embedding.subtype _) 
    (λ i, (double_factorial (2 * i)) / (double_factorial (2 * i + 1)))

-- Statement of the problem
theorem problem_solution :
  let c := 0
  let d := (denom S)
  (c * d) / 10 = 0 :=
by
  sorry

end problem_solution_l464_464680


namespace g_eq_g_inv_solution_l464_464863

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem g_eq_g_inv_solution (x : ℝ) : g x = g_inv x ↔ x = 5 / 3 :=
by
  sorry

end g_eq_g_inv_solution_l464_464863


namespace find_y_l464_464118

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end find_y_l464_464118


namespace determine_function_l464_464865

theorem determine_function (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (x + 1)) :=
by
  sorry

end determine_function_l464_464865


namespace odd_prime_power_condition_l464_464336

noncomputable def is_power_of (a b : ℕ) : Prop :=
  ∃ t : ℕ, b = a ^ t

theorem odd_prime_power_condition (n p x y k : ℕ) (hn : 1 < n) (hp_prime : Prime p) 
  (hp_odd : p % 2 = 1) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (hx_odd : x % 2 ≠ 0) 
  (hy_odd : y % 2 ≠ 0) (h_eq : x^n + y^n = p^k) :
  is_power_of p n :=
sorry

end odd_prime_power_condition_l464_464336


namespace complex_magnitude_l464_464463

theorem complex_magnitude (z : ℂ) (i_unit : ℂ := Complex.I) 
  (h : (z - i_unit) * i_unit = 2 + i_unit) : Complex.abs z = Real.sqrt 5 := 
by
  sorry

end complex_magnitude_l464_464463


namespace product_of_possible_values_l464_464734

theorem product_of_possible_values (b : ℝ) (side_length : ℝ) (square_condition : (b - 2) = side_length ∨ (2 - b) = side_length) : 
  (b = -3 ∨ b = 7) → (-3 * 7 = -21) :=
by
  intro h
  sorry

end product_of_possible_values_l464_464734


namespace harlequin_first_and_wins_l464_464843

theorem harlequin_first_and_wins 
  (initial_matches : ℕ) 
  (final_matches : ℕ) 
  (harlequin_moves : ℕ → ℕ → Prop)
  (pierrot_moves : ℕ → ℕ → Prop)
  (divisible_by_seven : ∀ n ∈ {5, 26}, ∃ k : ℕ, n = 7 * k)
  (divisible_by_seven_pierrot : ∀ n ∈ {9, 23}, ∃ k : ℕ, n = 2 * 7 * k + 2)
  (initial_mod_seven : initial_matches % 7 = 0)
  (final_mod_seven : final_matches % 7 = 2)
  : final_matches = 2 ∧ initial_matches = 2016 ∧ ∃ move_sequence, harlequin_moves 2016 final_matches ∧ ¬ ∃ move_sequence_pierrot, pierrot_moves initial_matches final_matches :=
by
  sorry

end harlequin_first_and_wins_l464_464843


namespace harmonic_quad_l464_464889

noncomputable theory

-- Definitions based on the conditions
def point := ℝ × ℝ

inductive Circle
| mk : point → ℝ → Circle -- center and radius

structure Tangent (c : Circle) (p : point) :=
(tangent_point : point)
(is_tangent : ∃ t, tangent_point = t ∧ dist p t = dist c.center t * dist c.center p)

structure Secant (c : Circle) (p1 p2: point) :=
(intersections: p1 ≠ p2 ∧
  (c.mk c.center c.radius).contains p1 ∧
  (c.mk c.center c.radius).contains p2)

-- harmonic_quadratic checks if four points form a harmonic quadruple
def harmonic_quadratic (a b c d : point) : Prop :=
  dist a b / dist b c = dist a d / dist d c

-- Given conditions in proto-proof form
variables (A B C D T1 T2 : point) (circle : Circle)
(h1 : Tangent circle A)
(h2 : Tangent circle A)
(h3 : Secant circle B D)
(h4 : C = chord_intersection T1 T2 B D)

-- Final statement reflecting the problem's conditions and required proof
theorem harmonic_quad (circle : Circle) (A B C D T1 T2 : point)
  (h1 : Tangent circle A)
  (h2 : Tangent circle A)
  (h3 : Secant circle B D)
  (h4 : C = chord_intersection T1 T2 B D) : 
  harmonic_quadratic A B C D := sorry

end harmonic_quad_l464_464889


namespace weight_of_new_person_l464_464261

-- Define the parameters
def avg_weight_increase := 5 -- The average weight increases by 5 kg

def original_weight (n : ℕ) (weight : ℕ) : Prop :=
  weight = 35 ∧ n = 8 -- The replaced person weighs 35 kg and there are 8 persons

-- Define the new weight to be proved which is 75 kg
def new_weight (original_weight : ℕ) (increase : ℕ) : ℕ :=
  original_weight + (8 * increase)

-- The proof goal in Lean
theorem weight_of_new_person (n : ℕ) (weight increase : ℕ) (h : original_weight n weight) :
  new_weight weight avg_weight_increase = 75 :=
by
  unfold original_weight at h
  cases h with hw hn 
  rw [← hw, ← hn]
  unfold new_weight
  norm_num
  sorry -- Proof logic to be provided

end weight_of_new_person_l464_464261


namespace pounds_per_point_l464_464950

def pounds_recycled_gwen := 5
def pounds_recycled_friends := 13
def total_points_earned := 6
def total_pounds_recycled := pounds_recycled_gwen + pounds_recycled_friends

theorem pounds_per_point : total_pounds_recycled / total_points_earned = 3 := by
  have h := total_pounds_recycled
  have h2 := total_points_earned
  rw [←nat.div_eq_of_eq_mul (18 / 6)]
  sorry

end pounds_per_point_l464_464950


namespace roots_polynomial_sum_squares_l464_464187

theorem roots_polynomial_sum_squares (p q r : ℝ) 
  (h_roots : ∀ x : ℝ, x^3 - 15 * x^2 + 25 * x - 10 = 0 → x = p ∨ x = q ∨ x = r) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := 
by {
  sorry
}

end roots_polynomial_sum_squares_l464_464187


namespace odd_function_inequality_l464_464028

variable {R : Type} [LinearOrderedField R]

def is_odd_function (f : R → R) : Prop :=
  ∀ x : R, f (-x) = -f x

theorem odd_function_inequality {f : R → R} (h_odd : is_odd_function f) :
  ∀ x : R, f x * f (-x) ≤ 0 :=
begin
  intros x,
  have h := h_odd x,
  rw h,
  rw mul_neg_eq_neg_mul_symm,
  exact neg_nonpos_of_nonneg (mul_self_nonneg (f x)),
end

end odd_function_inequality_l464_464028


namespace prove_parallel_l464_464644

-- Define points A, P, C, M, N, Q, and functions L, S.
variables {A P C M N Q L S : Point}
variables [incidence_geometry]

-- Define L and S based on the given conditions.
def L := intersection (line_through A P) (line_through C M)
def S := intersection (line_through A N) (line_through C Q)

-- Define the condition to prove LS is parallel to PQ.
theorem prove_parallel : parallel (line_through L S) (line_through P Q) :=
sorry

end prove_parallel_l464_464644


namespace solve_factorial_equation_in_natural_numbers_l464_464250

theorem solve_factorial_equation_in_natural_numbers :
  ∃ n k : ℕ, n! + 3 * n + 8 = k^2 ↔ n = 2 ∧ k = 4 := by
sorry

end solve_factorial_equation_in_natural_numbers_l464_464250


namespace product_of_possible_values_l464_464735

theorem product_of_possible_values (b : ℝ) (side_length : ℝ) (square_condition : (b - 2) = side_length ∨ (2 - b) = side_length) : 
  (b = -3 ∨ b = 7) → (-3 * 7 = -21) :=
by
  intro h
  sorry

end product_of_possible_values_l464_464735


namespace perimeter_of_rearranged_rectangles_l464_464808

theorem perimeter_of_rearranged_rectangles 
  (side_length : ℕ) 
  (h_side_length : side_length = 100) 
  (h_split : ∀ r : ℕ × ℕ, r = (100, 50) ∨ r = (50, 100)) :
  let perimeter := 3 * 100 + 4 * 50 in
  perimeter = 500 :=
by
  let perimeter := 3 * 100 + 4 * 50
  sorry

end perimeter_of_rearranged_rectangles_l464_464808


namespace num_two_digit_factors_of_3_18_minus_1_l464_464954

theorem num_two_digit_factors_of_3_18_minus_1 : 
  (∃ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n ∣ (3^18 - 1)) → (Finset.card (Finset.filter (λ n, 10 ≤ n ∧ n < 100) (nat.divisors (3^18 - 1))) = 11) :=
begin
  sorry
end

end num_two_digit_factors_of_3_18_minus_1_l464_464954


namespace student_attendance_each_day_l464_464714

def total_students := 1000
def percent_learning_from_home := 60
def students_learning_from_home := (percent_learning_from_home * total_students) / 100
def students_attending_school := total_students - students_learning_from_home
def group_size_A_and_B := students_attending_school / 3
def group_size_C := students_attending_school - 2 * group_size_A_and_B

theorem student_attendance_each_day :
  (group_size_A_and_B = 133) →
  (group_size_C = 134) →
  ∀ day,
    (day = "Monday" ∨ day = "Wednesday" → group_size_A_and_B = 133) ∧ 
    (day = "Tuesday" ∨ day = "Thursday" → group_size_A_and_B = 133) ∧ 
    (day = "Friday" → group_size_C = 134) := 
begin
  sorry
end

end student_attendance_each_day_l464_464714


namespace reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464392

-- Probability that the bags end up in the shed in the reverse order
def probability_reverse_order : ℕ → ℚ
| 0 := 1
| (n + 1) := (1 / 2) * probability_reverse_order n

theorem reverse_order_probability_is_1_over_8 :
  probability_reverse_order 3 = 1 / 8 :=
sorry

-- Probability that the second bag from the bottom ends up as the bottom bag in the shed
def probability_second_from_bottom_bottom : ℚ := 
(1 / 2) * (1 / 2) * (1 / 2)

theorem second_from_bottom_as_bottom_is_1_over_8 :
  probability_second_from_bottom_bottom = 1 / 8 :=
sorry

end reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464392


namespace ls_parallel_pq_l464_464655

-- Definitions for points L and S, and their parallelism
variables {A P C M N Q L S : Type}

-- Assume points and lines for the geometric setup
variables (A P C M N Q L S : Type)

-- Assume the intersections and the required parallelism conditions
variables (hL : ∃ (AP CM : Set (Set L)), L ∈ AP ∩ CM)
variables (hS : ∃ (AN CQ : Set (Set S)), S ∈ AN ∩ CQ)

-- The goal: LS is parallel to PQ
theorem ls_parallel_pq : LS ∥ PQ :=
sorry

end ls_parallel_pq_l464_464655


namespace least_n_for_A0An_geq_50_l464_464591

theorem least_n_for_A0An_geq_50 :
  (∀ n : ℕ, A (n : ℕ) = (n, 0) ∧ B (n : ℕ) = ((n + 1) / 2, ((n + 1) / 2)^2) ∧ is_equilateral A (n - 1) B (n) A (n) ) →
  ∀ x, x ∈ (range 10) →
  dist (A 0) (A x) < 50 :=
begin
  sorry
end

end least_n_for_A0An_geq_50_l464_464591


namespace volume_calc_l464_464777

noncomputable
def volume_of_open_box {l w : ℕ} (sheet_length : l = 48) (sheet_width : w = 38) (cut_length : ℕ) (cut_length_eq : cut_length = 8) : ℕ :=
  let new_length := l - 2 * cut_length
  let new_width := w - 2 * cut_length
  let height := cut_length
  new_length * new_width * height

theorem volume_calc : volume_of_open_box (sheet_length := rfl) (sheet_width := rfl) (cut_length := 8) (cut_length_eq := rfl) = 5632 :=
sorry

end volume_calc_l464_464777


namespace square_rotation_resulting_body_l464_464367

theorem square_rotation_resulting_body : 
  (∃ sq : ℝ × ℝ → Prop, 
    (∀ (x y : ℝ), sq (x, y) ↔ x^2 + y^2 = 1 ∨ (-x, -y) ∧ (-x)^2 + (-y)^2 = 1)
    ∧ ∀ diag : ℝ × ℝ → Prop, ∀ (x y : ℝ), diag (x, y) ↔ x = y) →
    (∀ rot : ℝ × ℝ × ℝ → Prop, 
    (∀ (x y θ : ℝ), rot (x, y, θ) ↔ 
      (∃ (r : ℝ), (r^2 = x^2 + y^2) 
      ∧ ( ∃ (φ : ℝ), (sin φ = x / r) ∧ (cos φ = y / r) 
      ∧ θ = φ ∨ θ = φ + π )))) →
     resulting_body = "Two Cones" := sorry

end square_rotation_resulting_body_l464_464367


namespace sum_reaches_max_at_7_or_8_l464_464146

def arithmetic_seq (a_n : ℕ → ℤ) (d a_1 a_2 a_6 : ℤ) : Prop :=
  a_2 = a_1 + d ∧
  a_6 = a_1 + 5 * d ∧ 
  ∀ n, a_n n = a_1 + (n - 1) * d

theorem sum_reaches_max_at_7_or_8 
  (a_n : ℕ → ℤ) (a_2 a_6 d a_1 S : ℤ) (Sn : ℕ → ℤ) :
  a_2 = 6 →
  a_6 = 2 →
  arithmetic_seq a_n d a_1 a_2 a_6 →
  (∀ n, Sn n = n * (2 * a_n 1 + (n - 1) * d) / 2) →
  (∀ n, Sn n ≤ S) →
  (S = Sn 7 ∨ S = Sn 8) :=
begin
  sorry
end

end sum_reaches_max_at_7_or_8_l464_464146


namespace k_equals_10_l464_464291

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) : ℕ → α
  | 0     => a
  | (n+1) => a + (n+1) * d

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem k_equals_10
  (a d : α)
  (h1 : sum_of_first_n_terms a d 9 = sum_of_first_n_terms a d 4)
  (h2 : arithmetic_sequence a d 4 + arithmetic_sequence a d 10 = 0) :
  k = 10 :=
sorry

end k_equals_10_l464_464291


namespace abscissa_range_of_point_P_l464_464700

-- Definitions based on the conditions from the problem
def y_function (x : ℝ) : ℝ := 4 - 3 * x
def point_P (x y : ℝ) : Prop := y = y_function x
def ordinate_greater_than_negative_five (y : ℝ) : Prop := y > -5

-- Theorem statement combining the above definitions
theorem abscissa_range_of_point_P (x y : ℝ) :
  point_P x y →
  ordinate_greater_than_negative_five y →
  x < 3 :=
sorry

end abscissa_range_of_point_P_l464_464700


namespace cube_surface_area_l464_464780

noncomputable def volume (x : ℝ) : ℝ := x ^ 3

noncomputable def surface_area (x : ℝ) : ℝ := 6 * x ^ 2

theorem cube_surface_area (x : ℝ) :
  surface_area x = 6 * x ^ 2 :=
by sorry

end cube_surface_area_l464_464780


namespace prove_parallel_l464_464645

-- Define points A, P, C, M, N, Q, and functions L, S.
variables {A P C M N Q L S : Point}
variables [incidence_geometry]

-- Define L and S based on the given conditions.
def L := intersection (line_through A P) (line_through C M)
def S := intersection (line_through A N) (line_through C Q)

-- Define the condition to prove LS is parallel to PQ.
theorem prove_parallel : parallel (line_through L S) (line_through P Q) :=
sorry

end prove_parallel_l464_464645


namespace point_on_hyperbola_probability_l464_464227

theorem point_on_hyperbola_probability :
  let s := ({1, 2, 3} : Finset ℕ) in
  let p := ∑ x in s.sigma (λ x, s.filter (λ y, y ≠ x)),
             if (∃ m n, x = (m, n) ∧ n = (6 / m)) then 1 else 0 in
  p / (s.card * (s.card - 1)) = (1 / 3) :=
by
  -- Conditions and setup
  let s := ({1, 2, 3} : Finset ℕ)
  let t := s.sigma (λ x, s.filter (λ y, y ≠ x))
  let p := t.filter (λ (xy : ℕ × ℕ), xy.snd = 6 / xy.fst)
  have h_total : t.card = 6, by sorry
  have h_count : p.card = 2, by sorry

  -- Calculate probability
  calc
    ↑(p.card) / ↑(t.card) = 2 / 6 : by sorry
    ... = 1 / 3 : by norm_num

end point_on_hyperbola_probability_l464_464227


namespace num_rectangles_on_grid_l464_464520

theorem num_rectangles_on_grid :
  let points := {(0,0), (0,5), (0,10), (0,15), (5,0), (5,5), (5,10), (5,15), (10,0), (10,5), (10,10), (10,15), (15,0), (15,5), (15,10), (15,15)} in
  let vertical_lines := {0, 5, 10, 15} in
  let horizontal_lines := {0, 5, 10, 15} in
  (vertical_lines.card.choose 2) * (horizontal_lines.card.choose 2) = 36 := 
by
  let points := {(0,0), (0,5), (0,10), (0,15), (5,0), (5,5), (5,10), (5,15), (10,0), (10,5), (10,10), (10,15), (15,0), (15,5), (15,10), (15,15)}
  let vertical_lines := {0, 5, 10, 15}
  let horizontal_lines := {0, 5, 10, 15}
  have h_vertical := vertical_lines.card_choose_2
  have h_horizontal := horizontal_lines.card_choose_2
  exact mul_eq_one

end num_rectangles_on_grid_l464_464520


namespace acute_angle_OP_MI_l464_464133

variable (M I T X Y O P : Type)
variable (MI_length : ℝ) (MX_length YI_length : ℝ)
variable [IsTriangle M I T] [HasAngle M 30] [HasAngle I 60] 
variable [HasLength MI MI_length] [HasLength MX MX_length] [HasLength YI YI_length]
variable [MidpointOf O MI] [MidpointOf P XY]
variable (angle_OP_MI : ℝ)

noncomputable def angle_OP_MI_correct : Prop :=
  angle_OP_MI = 15

theorem acute_angle_OP_MI :
  angle_OP_MI_correct M I T X Y O P MI_length MX_length YI_length angle_OP_MI :=
sorry

end acute_angle_OP_MI_l464_464133


namespace domain_of_function_l464_464878

noncomputable def function_well_defined_domain : Prop :=
  ∀ (x : ℝ), (∃ (k : ℤ), 2 * k * Real.pi < x ∧ x < (2 * k + 1) * Real.pi) ↔
  (∃ (y : ℝ), y = Real.sqrt (Real.logBase (1/2) (Real.sin x)))

theorem domain_of_function : function_well_defined_domain :=
by
  sorry

end domain_of_function_l464_464878


namespace polynomial_sum_eq_zero_l464_464724

theorem polynomial_sum_eq_zero {x : ℂ} (h1 : x ≠ 1) (h2 : x^2018 - 2*x + 3 = 0) :
  x^2017 + x^2016 + ⋯ + x + 1 = 0 :=
by sorry

end polynomial_sum_eq_zero_l464_464724


namespace determine_g50_l464_464273

noncomputable def g : ℝ → ℝ := sorry
parameter (k : ℝ)
axiom functional_equation (x y : ℝ) : 0 < x → 0 < y → x * g y - y * g x = k * g (x / y)

-- The task: Prove that g(50) = 0 if k = -1 and g(50) = 50C for some constant C if k ≠ -1
theorem determine_g50 (C : ℝ) (hk_ne_neg1 : k ≠ -1) :
  ∃ C, (∀ x : ℝ, 0 < x → g x = C * x) → g 50 = 0 ∨ g 50 = 50 * C :=
sorry

end determine_g50_l464_464273


namespace sum_of_common_ratios_l464_464677

theorem sum_of_common_ratios (k p r : ℝ) (h₁ : k ≠ 0) (h₂ : p ≠ r) (h₃ : (k * (p ^ 2)) - (k * (r ^ 2)) = 4 * (k * p - k * r)) : 
  p + r = 4 :=
by
  -- Using the conditions provided, we can prove the sum of the common ratios is 4.
  sorry

end sum_of_common_ratios_l464_464677


namespace max_points_distance_at_least_sqrt2_max_points_distance_more_than_sqrt2_l464_464429

open Classical

-- Part (a)
theorem max_points_distance_at_least_sqrt2 (n : ℕ) :
  (∀ (i j : ℕ) (hi : i ≠ j), dist (pts i) (pts j) ≥ sqrt 2) →
  n ≤ 6 :=
by sorry

-- Part (b)
theorem max_points_distance_more_than_sqrt2 (n : ℕ) :
  (∀ (i j : ℕ) (hi : i ≠ j), dist (pts i) (pts j) > sqrt 2) →
  n ≤ 4 :=
by sorry

end max_points_distance_at_least_sqrt2_max_points_distance_more_than_sqrt2_l464_464429


namespace probability_reverse_order_probability_second_to_bottom_l464_464385

noncomputable def prob_reverse_order : ℚ := 1/8
noncomputable def prob_second_to_bottom : ℚ := 1/8

theorem probability_reverse_order (n: ℕ) (h : n = 4) 
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_reverse_order = 1/8 :=
by sorry

theorem probability_second_to_bottom (n: ℕ) (h : n = 4)
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_second_to_bottom = 1/8 :=
by sorry

end probability_reverse_order_probability_second_to_bottom_l464_464385


namespace sum_of_cubes_of_unique_integers_l464_464409

theorem sum_of_cubes_of_unique_integers (n : ℤ) :
  (n + 7)^3 - (n + 6)^3 - (n + 5)^3 + (n + 4)^3 - (n + 3)^3 + (n + 2)^3 + (n + 1)^3 - n^3 = 48 →
  ∃ k : ℤ, ∃ (a : Finset ℤ), (∀ x ∈ a, x ≠ y ∧ x ≠ z ∀ y, z ∈ a with y ≠ z) ∧ ∑ x in a, x^3 = n :=
begin
  sorry
end

end sum_of_cubes_of_unique_integers_l464_464409


namespace conditional_probability_l464_464888

noncomputable def P (E : Prop) : ℝ := sorry -- Probability of an event E
noncomputable def P_cond (E F : Prop) : ℝ := P(E ∧ F) / P(F) -- Conditional probability

/-
  Four students (A, B, C, and D) have signed up for a community service activity 
  during the holidays. The activity consists of four projects: caring for the elderly, 
  environmental monitoring, educational consultation, and traffic promotion.
  Each student can only sign up for one project.
  Let event A be "The projects chosen by the four students are all different",
  and event B be "Only student A signed up for the caring for the elderly project".
  We need to prove that P(A | B) = 2/9.
-/

def event_A : Prop := sorry -- "The projects chosen by the four students are all different"
def event_B : Prop := sorry -- "Only student A signed up for the caring for the elderly project"

theorem conditional_probability :
  P_cond event_A event_B = 2/9 :=
by
  sorry  -- Proof not required

end conditional_probability_l464_464888


namespace probability_point_A_on_hyperbola_l464_464220

-- Define the set of numbers
def numbers : List ℕ := [1, 2, 3]

-- Define the coordinates of point A taken from the set, where both numbers are different
def point_A_pairs : List (ℕ × ℕ) :=
  [ (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) ]

-- Define the function indicating if a point (m, n) lies on the hyperbola y = 6/x
def lies_on_hyperbola (m n : ℕ) : Prop :=
  n = 6 / m

-- Calculate the probability of a point lying on the hyperbola
theorem probability_point_A_on_hyperbola : 
  (point_A_pairs.countp (λ (p : ℕ × ℕ), lies_on_hyperbola p.1 p.2)).toRat / (point_A_pairs.length).toRat = 1 / 3 := 
sorry

end probability_point_A_on_hyperbola_l464_464220


namespace train_length_is_500_meters_l464_464371

noncomputable def train_speed_km_hr : ℝ := 120
noncomputable def time_to_cross_pole_sec : ℝ := 15

noncomputable def km_hr_to_m_s (speed_km_hr : ℝ) : ℝ := speed_km_hr * 1000 / 3600

noncomputable def train_length_in_meters : ℝ :=
  km_hr_to_m_s train_speed_km_hr * time_to_cross_pole_sec

theorem train_length_is_500_meters : train_length_in_meters ≈ 500 :=
by
  sorry

end train_length_is_500_meters_l464_464371


namespace area_triangle_QMN_is_54_l464_464554

-- Definitions for the problem
variables (P Q R S M N : Type) -- Points on the plane
variables (a : ℝ) -- side length of the square PQRS
variables (area_triangle_SMN : ℝ) -- Area of triangle SMN

-- Conditions
variable (h_square_PQRS : PQRS_is_square P Q R S a)
variable (h_midpoint_M : midpoint P S M)
variable (h_midpoint_N : midpoint S R N)
variable (h_area_SMN : area_triangle_SMN = 18)

-- Proposition to prove
theorem area_triangle_QMN_is_54 : area_triangle Q M N = 54 :=
by
  sorry

end area_triangle_QMN_is_54_l464_464554


namespace Annika_multiple_of_Hans_in_four_years_l464_464142

theorem Annika_multiple_of_Hans_in_four_years 
  (Hans_age_now : ℕ) (Annika_age_now : ℕ) :
  Hans_age_now = 8 -> Annika_age_now = 32 -> 
  let Hans_age_in_four_years := Hans_age_now + 4 in
  let Annika_age_in_four_years := Annika_age_now + 4 in
  Annika_age_in_four_years = 3 * Hans_age_in_four_years :=
by {
  intros h1 h2,
  simp [h1, h2],
  sorry
}

end Annika_multiple_of_Hans_in_four_years_l464_464142


namespace prob_reverse_order_prob_second_from_bottom_l464_464390

section CementBags

-- Define the basic setup for the bags and worker choices
variables (bags : list ℕ) (choices : list ℕ) (shed gate : list ℕ)

-- This ensures we are indeed considering 4 bags in a sequence
axiom bags_is_4 : bags.length = 4

-- This captures that the worker makes 4 sequential choices with equal probability for each, where 0 represents shed and 1 represents gate
axiom choices_is_4_and_prob : (choices.length = 4) ∧ (∀ i in choices, i ∈ [0, 1])

-- This condition captures that eventually, all bags must end up in the shed
axiom all_bags_in_shed : ∀ b ∈ bags, b ∈ shed

-- Prove for part (a): Probability of bags in reverse order in shed is 1/8
theorem prob_reverse_order : (choices = [0, 0, 0, 0] → (reverse bags = shed)) → 
  (probability (choices = [0, 0, 0, 0]) = 1 / 8) :=
 sorry

-- Prove for part (b): Probability of the second-from-bottom bag in the bottom is 1/8
theorem prob_second_from_bottom : 
  (choices = [1, 1, 0, 0] → (shed.head = bags.nth 1.get_or_else 0)) → 
  (probability (choices = [1, 1, 0, 0]) = 1 / 8) :=
 sorry

end CementBags

end prob_reverse_order_prob_second_from_bottom_l464_464390


namespace variables_and_unknowns_l464_464992

theorem variables_and_unknowns (f_1 f_2: ℝ → ℝ → ℝ) (f: ℝ → ℝ → ℝ) :
  (∀ x y, f_1 x y = 0 ∧ f_2 x y = 0 → (x ≠ 0 ∨ y ≠ 0)) ∧
  (∀ x y, f x y = 0 → (∃ a b, x = a ∧ y = b)) :=
by sorry

end variables_and_unknowns_l464_464992


namespace tan_theta_pure_imaginary_l464_464458

variable {θ : ℝ} (z1 z2 : ℂ)
variable (pure_imaginary : ℂ → Prop)

noncomputable def is_pure_imaginary (z : ℂ) : Prop :=
  ∃ y : ℂ, z = y * complex.I

theorem tan_theta_pure_imaginary (h1 : z1 = sin θ - (4 / 5) * complex.I)
  (h2 : z2 = (3 / 5) - cos θ * complex.I)
  (h3 : is_pure_imaginary (z1 - z2)) :
  tan θ = - (3 / 4) :=
sorry

end tan_theta_pure_imaginary_l464_464458


namespace derivative_f_at_pi_l464_464450

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem derivative_f_at_pi : (deriv f π) = -1 := 
by
  sorry

end derivative_f_at_pi_l464_464450


namespace value_of_a_standard_equation_of_curve_l464_464501

variable (a : ℝ) (x y t : ℝ)

/- Parametric equations conditions -/
def parametric_x := 1 + 2 * t
def parametric_y := a * t^2

/- Given point M(5, 4) lies on the curve C -/
def point_on_curve := (x = 5) ∧ (y = 4)

/- Proof part (1): Value of a -/
theorem value_of_a :
  point_on_curve → parametric_x = 5 → parametric_y = 4 → a = 1 :=
by
  intros h_point_on_curve h_parametric_x h_parametric_y
  sorry

/- Proof part (2): Standard equation of the curve C -/
theorem standard_equation_of_curve :
  a = 1 → (y = a * (x - 1)^2 / 4) → 4 * y = (x - 1)^2 :=
by
  intros h_a h_cartesian
  sorry

end value_of_a_standard_equation_of_curve_l464_464501


namespace smallest_integer_of_three_l464_464752

theorem smallest_integer_of_three (a b c : ℕ) (h_mean : (a + b + c) / 3 = 29)
  (h_median : median a b c = 30)
  (h_largest : largest a b c = 30 + 5) : 
  smallest a b c = 22 := by
  sorry

end smallest_integer_of_three_l464_464752


namespace length_CF_correct_l464_464668

-- Define the given conditions in the problem
variables {A B C D E F : Type} [InABC : triangle A B C]
variable (D : Point3 ℝ) (hD : is_midpoint D (segment B C))
variable (E : Point3 ℝ) (hE : is_midpoint E (segment A C))
variable (F : Point3 ℝ) (hF : is_midpoint F (segment A B))
variable (hAD_perp_BE : perp (median A D) (median B E))
variable (hAD_length : length (median A D) = 18)
variable (hBE_length : length (median B E) = 13.5)

-- Define the goal: length of the median CF
noncomputable def median_CF_length : ℝ :=
  length (median C F)

-- The theorem that we want to prove
theorem length_CF_correct :
  median_CF_length = 45 / 2 :=
sorry

end length_CF_correct_l464_464668


namespace probability_point_on_hyperbola_l464_464217

-- Define the problem conditions
def number_set := {1, 2, 3}
def point_on_hyperbola (x y : ℝ) : Prop := y = 6 / x

-- Formalize the problem statement
theorem probability_point_on_hyperbola :
  let combinations := ({(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} : set (ℝ × ℝ)) in
  let on_hyperbola := set.filter (λ p : ℝ × ℝ, point_on_hyperbola p.1 p.2) combinations in
  fintype.card on_hyperbola / fintype.card combinations = 1 / 3 :=
by sorry

end probability_point_on_hyperbola_l464_464217


namespace ls_parallel_pq_l464_464654

-- Definitions for points L and S, and their parallelism
variables {A P C M N Q L S : Type}

-- Assume points and lines for the geometric setup
variables (A P C M N Q L S : Type)

-- Assume the intersections and the required parallelism conditions
variables (hL : ∃ (AP CM : Set (Set L)), L ∈ AP ∩ CM)
variables (hS : ∃ (AN CQ : Set (Set S)), S ∈ AN ∩ CQ)

-- The goal: LS is parallel to PQ
theorem ls_parallel_pq : LS ∥ PQ :=
sorry

end ls_parallel_pq_l464_464654


namespace find_value_of_M_l464_464288

variable {C y M A : ℕ}

theorem find_value_of_M (h1 : C + y + 2 * M + A = 11)
                        (h2 : C ≠ y)
                        (h3 : C ≠ M)
                        (h4 : C ≠ A)
                        (h5 : y ≠ M)
                        (h6 : y ≠ A)
                        (h7 : M ≠ A)
                        (h8 : 0 < C)
                        (h9 : 0 < y)
                        (h10 : 0 < M)
                        (h11 : 0 < A) : M = 1 :=
by
  sorry

end find_value_of_M_l464_464288


namespace area_triangle_DBC_l464_464985

-- Definitions of points A, B, and C.
def A := (0, 10)
def B := (0, 0)
def C := (12, 0)

-- Functions to calculate midpoints
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Definitions of points D and E as midpoints
def D := midpoint A B
def E := midpoint B C

-- Function to calculate the area of a triangle given coordinates of vertices.
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2

-- Statement that the area of triangle DBC is 30
theorem area_triangle_DBC : triangle_area D B C = 30 := by
  -- Sorry used to skip the proof.
  sorry

end area_triangle_DBC_l464_464985


namespace find_a_b_l464_464464

def z := Complex.ofReal 3 + Complex.I * 4
def z_conj := Complex.ofReal 3 - Complex.I * 4

theorem find_a_b 
  (a b : ℝ) 
  (h : z + Complex.ofReal a * z_conj + Complex.I * b = Complex.ofReal 9) : 
  a = 2 ∧ b = 4 := 
by 
  sorry

end find_a_b_l464_464464


namespace reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464393

-- Probability that the bags end up in the shed in the reverse order
def probability_reverse_order : ℕ → ℚ
| 0 := 1
| (n + 1) := (1 / 2) * probability_reverse_order n

theorem reverse_order_probability_is_1_over_8 :
  probability_reverse_order 3 = 1 / 8 :=
sorry

-- Probability that the second bag from the bottom ends up as the bottom bag in the shed
def probability_second_from_bottom_bottom : ℚ := 
(1 / 2) * (1 / 2) * (1 / 2)

theorem second_from_bottom_as_bottom_is_1_over_8 :
  probability_second_from_bottom_bottom = 1 / 8 :=
sorry

end reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464393


namespace parallel_LS_pQ_l464_464624

universe u

variables {P Q A C M N L S T : Type u} [incidence_geometry P Q A C M N L S T] -- Customize as per specific needs in Lean

-- Assume the required points and their intersections
def intersection_L (A P C M : Type u) [incidence_geometry A P C M] : Type u := sorry
def intersection_S (A N C Q : Type u) [incidence_geometry A N C Q] : Type u := sorry

-- Defining the variables as intersections
variables (L : intersection_L A P C M)
variables (S : intersection_S A N C Q)

theorem parallel_LS_pQ (hL : intersection_L A P C M) (hS : intersection_S A N C Q) :
  parallel L S P Q :=
sorry

end parallel_LS_pQ_l464_464624


namespace original_survey_customers_l464_464815

theorem original_survey_customers : ∃ (x : ℕ), (7 / x : ℚ) + 0.06 = 1 / 7 ∧ x ≈ 84 :=
by
  sorry

end original_survey_customers_l464_464815


namespace position_of_2023rd_square_l464_464466

theorem position_of_2023rd_square :
  let initial_position := "ABCD"
  let first_transformation := "DABC" -- 90° CW
  let second_transformation := "CBAD" -- Reflect vertically
  let third_transformation := "BADC" -- 180° around center
  let fourth_transformation := "DCBA" -- 90° CW
  let fifth_transformation := "ABCD" -- Reflect vertically
  let sixth_transformation := "ABCD" -- 180° returns to original
  let pattern := [initial_position, first_transformation, second_transformation, third_transformation, fourth_transformation, fifth_transformation, sixth_transformation]
  initial_position = "ABCD" →
  first_transformation = "DABC" →
  second_transformation = "CBAD" →
  third_transformation = "BADC" →
  fourth_transformation = "DCBA" →
  fifth_transformation = "ABCD" →
  sixth_transformation = "ABCD" →
  pattern[(2023 % 6)] = "CBAD" :=
by
  intros
  sorry

end position_of_2023rd_square_l464_464466


namespace prove_parallel_l464_464643

-- Define points A, P, C, M, N, Q, and functions L, S.
variables {A P C M N Q L S : Point}
variables [incidence_geometry]

-- Define L and S based on the given conditions.
def L := intersection (line_through A P) (line_through C M)
def S := intersection (line_through A N) (line_through C Q)

-- Define the condition to prove LS is parallel to PQ.
theorem prove_parallel : parallel (line_through L S) (line_through P Q) :=
sorry

end prove_parallel_l464_464643


namespace count_3digit_multiples_of_30_not_75_l464_464095

-- Define the conditions
def is_positive_3_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000
def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

-- Define the main theorem
theorem count_3digit_multiples_of_30_not_75 : 
  { n : ℕ // is_positive_3_digit n ∧ is_multiple_of n 30 ∧ ¬ is_multiple_of n 75 }.to_finset.card = 24 :=
by
  sorry

end count_3digit_multiples_of_30_not_75_l464_464095


namespace least_n_for_A0An_geq_50_l464_464590

theorem least_n_for_A0An_geq_50 :
  (∀ n : ℕ, A (n : ℕ) = (n, 0) ∧ B (n : ℕ) = ((n + 1) / 2, ((n + 1) / 2)^2) ∧ is_equilateral A (n - 1) B (n) A (n) ) →
  ∀ x, x ∈ (range 10) →
  dist (A 0) (A x) < 50 :=
begin
  sorry
end

end least_n_for_A0An_geq_50_l464_464590


namespace number_of_valid_integers_l464_464436

def has_two_odd_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).filter (λ x, x % 2 = 1).length = 2

def is_multiple_of_34 (n : ℕ) : Prop :=
  n % 34 = 0

def count_valid_numbers : ℕ :=
  (Finset.range (3400 + 1)).filter (λ n, is_multiple_of_34 n ∧ has_two_odd_divisors n).card

theorem number_of_valid_integers : count_valid_numbers = 6 :=
by
  sorry

end number_of_valid_integers_l464_464436


namespace smaller_angle_is_70_l464_464135

def measure_of_smaller_angle (x : ℕ) : Prop :=
  (x + (x + 40) = 180) ∧ (2 * x - 60 = 80)

theorem smaller_angle_is_70 {x : ℕ} : measure_of_smaller_angle x → x = 70 :=
by
  sorry

end smaller_angle_is_70_l464_464135


namespace garlic_cloves_remaining_l464_464198

theorem garlic_cloves_remaining (initial used : ℕ) (h_initial : initial = 93) (h_used : used = 86) : (initial - used = 7) :=
by 
  rw [h_initial, h_used]
  exact rfl

end garlic_cloves_remaining_l464_464198


namespace max_values_greater_than_half_l464_464055

theorem max_values_greater_than_half (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (hdistinct : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) :
  (if sin α * cos β > 1 / 2 then 1 else 0) +
  (if sin β * cos γ > 1 / 2 then 1 else 0) +
  (if sin γ * cos α > 1 / 2 then 1 else 0) ≤ 2 :=
sorry

end max_values_greater_than_half_l464_464055


namespace question_1_question_2_l464_464902

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem question_1 :
  f 1 * f 2 * f 3 = 36 * 108 * 360 := by
  sorry

theorem question_2 :
  ∃ m ≥ 2, ∀ n : ℕ, n > 0 → f n % m = 0 ∧ m = 36 := by
  sorry

end question_1_question_2_l464_464902


namespace find_integers_l464_464005

theorem find_integers (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h1 : 7 ∣ (a + b)^7 - a^7 - b^7) (h2 : ¬ 7 ∣ a * b * (a + b)) : 
  a = 18 ∧ b = 1 := 
sorry

def specific_example : Prop :=
find_integers 18 1 (by dec_trivial) (by dec_trivial)
  (by norm_num [pow_succ, pow_add, pow_bit0, pow_bit1, nat.sub_eq_iff_eq_add, nat.add_comm, nat.add_left_comm]; ring; norm_cast)
  (by norm_num [nat.mod_eq_zero, ← nat.dvd_iff_mod_eq_zero]; ring; norm_cast)

end find_integers_l464_464005


namespace three_digit_count_divisible_by_five_l464_464518

theorem three_digit_count_divisible_by_five : 
  let valid_digits := {x : ℕ | x > 5 ∧ x < 10} in
  let hundreds_digit_choices := valid_digits in
  let tens_digit_choices := valid_digits in
  let ones_digit_choice := {5} in
  (∃ (h t u : ℕ), h ∈ hundreds_digit_choices ∧ t ∈ tens_digit_choices ∧ u ∈ ones_digit_choice ∧ 
    100 * h + 10 * t + u > 99 ∧ 100 * h + 10 * t + u < 1000) ∧
  (100 * h + 10 * t + u) % 5 = 0 → 
  (Set.card hundreds_digit_choices) * 
  (Set.card tens_digit_choices) * 
  (Set.card ones_digit_choice) = 16 :=
by
  sorry

end three_digit_count_divisible_by_five_l464_464518


namespace trapezoid_area_l464_464150

theorem trapezoid_area (A_outer A_inner : ℝ) (n : ℕ)
  (h_outer : A_outer = 36)
  (h_inner : A_inner = 4)
  (h_n : n = 4) :
  (A_outer - A_inner) / n = 8 := by
  sorry

end trapezoid_area_l464_464150


namespace equivalent_polar_coordinates_l464_464995

theorem equivalent_polar_coordinates (r θ : ℝ) (h_r : r = -3) (h_θ : θ = (7 * Real.pi) / 6) :
  ∃ r' θ', r' = 3 ∧ θ' = Real.pi / 6 :=
by
  have h₀ : r < 0 := by sorry
  have h₁ : r' = -r := by sorry
  have h₂ : θ' = (θ + Real.pi) % (2 * Real.pi) := by sorry
  exists 3, Real.pi / 6
  split
  · exact eq.symm (abs_neg_of_neg h_r)
  · exact congr_arg (λ x => x % (2 * Real.pi)) (by congr; field_simp; linarith)

end equivalent_polar_coordinates_l464_464995


namespace probability_correct_l464_464550

open Finset

def boxA : Finset Nat := {1, 2}
def boxB : Finset Nat := {3, 4, 5, 6}

def favorable_pairs : Finset (Nat × Nat) := 
  {pair ∈ (boxA.product boxB) | (pair.fst + pair.snd) > 6}

def total_pairs : Finset (Nat × Nat) := boxA.product boxB

def probability_of_sum_greater_than_6 : ℚ := 
  (favorable_pairs.card : ℚ) / (total_pairs.card : ℚ)

theorem probability_correct : probability_of_sum_greater_than_6 = 3 / 8 := by
  sorry

end probability_correct_l464_464550


namespace range_of_x_in_function_l464_464741

/--
Given the function y = sqrt(x-1) / (x-2),
prove that the range of the independent variable x is 
x ≥ 1 and x ≠ 2.
-/
theorem range_of_x_in_function :
  ∀ x : ℝ, (x ≥ 1) ∧ (x ≠ 2) ↔ ∃ y : ℝ, y = (sqrt(x-1) / (x-2)) := 
sorry

end range_of_x_in_function_l464_464741


namespace parallel_LS_pQ_l464_464620

universe u

variables {P Q A C M N L S T : Type u} [incidence_geometry P Q A C M N L S T] -- Customize as per specific needs in Lean

-- Assume the required points and their intersections
def intersection_L (A P C M : Type u) [incidence_geometry A P C M] : Type u := sorry
def intersection_S (A N C Q : Type u) [incidence_geometry A N C Q] : Type u := sorry

-- Defining the variables as intersections
variables (L : intersection_L A P C M)
variables (S : intersection_S A N C Q)

theorem parallel_LS_pQ (hL : intersection_L A P C M) (hS : intersection_S A N C Q) :
  parallel L S P Q :=
sorry

end parallel_LS_pQ_l464_464620


namespace distinguishable_large_triangles_l464_464304

/--
There is an unlimited supply of congruent equilateral triangles made of colored paper. 
Each triangle is a solid color with the same color on both sides of the paper. 
A large equilateral triangle is constructed from four of these paper triangles. 
Two large triangles are considered distinguishable if it is not possible to place one 
on the other, using translations, rotations, and/or reflections, so that their 
corresponding small triangles are of the same color. Given that there are eight different 
colors of triangles from which to choose, how many distinguishable large equilateral 
triangles can be constructed?
-/
theorem distinguishable_large_triangles : 
  let colors := 8 in
  let all_same := colors in
  let two_same_one_diff := colors * (colors - 1) in
  let all_diff := Nat.choose colors 3 in
  let combinations := all_same + two_same_one_diff + all_diff in
  combinations * colors = 960 :=
by 
  sorry

end distinguishable_large_triangles_l464_464304


namespace sign_pyramid_top_plus_l464_464139
noncomputable theory

def valid_sign_pyramid_configurations : ℕ :=
  (finset.card (finset.filter (λ v: vector (ℤ) 5, 
    v.to_list.prod = 1) (finset.univ : finset (vector (ℤ) 5))))

theorem sign_pyramid_top_plus : valid_sign_pyramid_configurations = 16 :=
by
  sorry

end sign_pyramid_top_plus_l464_464139


namespace drivers_distance_difference_l464_464758

noncomputable def total_distance_driven (initial_distance : ℕ) (speed_A : ℕ) (speed_B : ℕ) (start_delay : ℕ) : ℕ := sorry

theorem drivers_distance_difference
  (initial_distance : ℕ)
  (speed_A : ℕ)
  (speed_B : ℕ)
  (start_delay : ℕ)
  (correct_difference : ℕ)
  (h_initial : initial_distance = 1025)
  (h_speed_A : speed_A = 90)
  (h_speed_B : speed_B = 80)
  (h_start_delay : start_delay = 1)
  (h_correct_difference : correct_difference = 145) :
  total_distance_driven initial_distance speed_A speed_B start_delay = correct_difference :=
sorry

end drivers_distance_difference_l464_464758


namespace solve_for_x_l464_464717

theorem solve_for_x (x : ℝ) : 5^(3 * x) = sqrt 125 → x = 1 / 2 := by
  sorry

end solve_for_x_l464_464717


namespace intersection_M_N_l464_464658

open Set Real

def M : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def N : Set ℝ := {x | log x / log 2 ≤ 1}

theorem intersection_M_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_M_N_l464_464658


namespace limit_value_l464_464530

noncomputable def f (x : ℝ) : ℝ := x^2

theorem limit_value :
  (filter.tendsto (λ (Δx : ℝ), (f (-1 + Δx) - f (-1)) / Δx) (nhds 0) (nhds (-2))) :=
sorry

end limit_value_l464_464530


namespace joe_height_l464_464244

theorem joe_height (S J A : ℝ) (h1 : S + J + A = 180) (h2 : J = 2 * S + 6) (h3 : A = S - 3) : J = 94.5 :=
by 
  -- Lean proof goes here
  sorry

end joe_height_l464_464244


namespace rational_root_of_polynomial_l464_464011

/-- Prove that 1/2 is a rational root of the polynomial 6x^5 - 4x^4 - 16x^3 + 8x^2 + 4x - 3 -/
theorem rational_root_of_polynomial : 
  IsRoot (Polynomial.C 6 * Polynomial.X^5 - Polynomial.C 4 * Polynomial.X^4 
            - Polynomial.C 16 * Polynomial.X^3 + Polynomial.C 8 * Polynomial.X^2 
            + Polynomial.C 4 * Polynomial.X - Polynomial.C 3) (1 / 2) := 
by 
  sorry

end rational_root_of_polynomial_l464_464011


namespace circle_intersection_parallels_l464_464851

theorem circle_intersection_parallels (
  (C1 C2 : Type) 
  (O1 O2 A B F E M N : Point)
  (h1 : circle C1 O1) (h2 : circle C2 O2)
  (h3 : C1 ∩ C2 = {A, B})
  (h4 : O1B ∈ radii C1 ∧ O2B ∈ radii C2)
  (h5 : ∃ (EF : Line), EF ⊥ tangent_line_at B C1 ∧ EF ⊥ tangent_line_at B C2)
  (h6 : line_through B ∥ EF)
  (h7 : M ∈ C1 ∧ N ∈ C2)
  (h8 : line_through B ∥ line_through E F)
  (h9 : M ∈ C1 \ {B} ∧ N ∈ C2 \ {B})
): MN = AE + AF := by
  sorry

end circle_intersection_parallels_l464_464851


namespace power_of_thousand_div_ten_seventeen_l464_464284

theorem power_of_thousand_div_ten_seventeen (x : ℤ) :
  (1000^x) / (10^17) = 10000 → 1000 = 10^3 → x = 7 :=
by
  intro h1 h2
  sorry

end power_of_thousand_div_ten_seventeen_l464_464284


namespace parallel_ls_pq_l464_464615

variables {A P C M N Q L S : Type*}
variables [affine_space ℝ (A P C M N Q)]

/- Definitions of the intersection points -/
def is_intersection_point (L : Type*) (AP : set (A P))
  (CM : set (C M)) : Prop := L ∈ AP ∧ L ∈ CM

def is_intersection_point' (S : Type*) (AN : set (A N))
  (CQ : set (C Q)) : Prop := S ∈ AN ∧ S ∈ CQ

/- Main Theorem -/
theorem parallel_ls_pq (AP CM AN CQ : set (A P))
  (PQ : set (P Q)) (L S : Type*)
  (hL : is_intersection_point L AP CM)
  (hS : is_intersection_point' S AN CQ) : L S ∥ P Q :=
sorry

end parallel_ls_pq_l464_464615


namespace exists_point_X_on_line_such_that_OX_eq_AB_l464_464906

-- Definitions for the conditions
variable {Point : Type} [AffineSpace Point]
variable (O : Point) (A B : Point) (l : AffineSubspace ℝ Point)

-- Functions to define the segments and construction details
def segment (A B : Point) : set Point := lineMap A B '' set.univ
noncomputable def length (A B : Point) : ℝ := (B -ᵥ A).norm
noncomputable def is_right_angle (P Q R : Point) : Prop := 
  ((R -ᵥ Q) ⬝ (P -ᵥ Q)) = 0

-- Theorem stating the problem's requirement
theorem exists_point_X_on_line_such_that_OX_eq_AB :
  ∃ X : Point, X ∈ l ∧ length O X = length A B :=
by
  sorry

end exists_point_X_on_line_such_that_OX_eq_AB_l464_464906


namespace warehouse_box_storage_l464_464842

theorem warehouse_box_storage (S : ℝ) (h1 : (3 - 1/4) * S = 55000) : (1/4) * S = 5000 :=
by
  sorry

end warehouse_box_storage_l464_464842


namespace at_least_six_students_solve_same_exercise_l464_464713

theorem at_least_six_students_solve_same_exercise (S : Finset ℕ) (E : Finset ℕ)
  (solves : ℕ → ℕ) (hS : S.card = 16) (hE : E.card = 3)
  (hsolve : ∀ s ∈ S, solves s ∈ E) :
  ∃ e ∈ E, (S.filter (λ s, solves s = e)).card ≥ 6 :=
by
  sorry

end at_least_six_students_solve_same_exercise_l464_464713


namespace largest_prime_factor_of_9870_is_47_l464_464015

theorem largest_prime_factor_of_9870_is_47 (n : ℕ) (h : n = 9870) : 
  ∃ p : ℕ, nat.prime p ∧ p ∈ n.factors ∧ p = 47 ∧ ∀ q ∈ n.factors, nat.prime q → q ≤ p :=
by sorry

end largest_prime_factor_of_9870_is_47_l464_464015


namespace jake_peaches_is_7_l464_464254

variable (Steven_peaches Jake_peaches Jill_peaches : ℕ)

-- Conditions:
def Steven_has_19_peaches : Steven_peaches = 19 := by sorry

def Jake_has_12_fewer_peaches_than_Steven : Jake_peaches = Steven_peaches - 12 := by sorry

def Jake_has_72_more_peaches_than_Jill : Jake_peaches = Jill_peaches + 72 := by sorry

-- Proof problem:
theorem jake_peaches_is_7 
    (Steven_peaches Jake_peaches Jill_peaches : ℕ)
    (h1 : Steven_peaches = 19)
    (h2 : Jake_peaches = Steven_peaches - 12)
    (h3 : Jake_peaches = Jill_peaches + 72) :
    Jake_peaches = 7 := by sorry

end jake_peaches_is_7_l464_464254


namespace max_area_PAB_l464_464153

noncomputable def curve_C1_parametric (α : ℝ) : ℝ × ℝ := 
  (2 + sqrt 7 * Real.cos α, sqrt 7 * Real.sin α)

noncomputable def curve_C1_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * Real.cos θ = 3

noncomputable def curve_C2_polar (θ : ℝ) : ℝ :=
  8 * Real.cos θ

def line_l_polar (θ : ℝ) : Prop :=
  θ = Real.pi / 3

def line_l_rect (x : ℝ) : ℝ :=
  Real.sqrt 3 * x

theorem max_area_PAB (ρ1 ρ2 : ℝ) (α : ℝ):
  let A := (ρ1, Real.pi / 3)
  let B := (ρ2, Real.pi / 3)
  let P := curve_C2_polar α
  (ρ1^2 - 4 * ρ1 * Real.cos (Real.pi / 3) = 3) →
  (ρ1 = 3) → 
  (A.1 = 3) → 
  (ρ2 = 4) → 
  |A.1 - B.1| = 1 →
  (1 / 2) * 1 * (4 + 2 * Real.sqrt 3) = 2 + Real.sqrt 3 :=
sorry

end max_area_PAB_l464_464153


namespace perimeter_of_rearranged_rectangles_l464_464809

theorem perimeter_of_rearranged_rectangles 
  (side_length : ℕ) 
  (h_side_length : side_length = 100) 
  (h_split : ∀ r : ℕ × ℕ, r = (100, 50) ∨ r = (50, 100)) :
  let perimeter := 3 * 100 + 4 * 50 in
  perimeter = 500 :=
by
  let perimeter := 3 * 100 + 4 * 50
  sorry

end perimeter_of_rearranged_rectangles_l464_464809


namespace least_n_for_length_l464_464589

/-- We start with the origin --/
def A0 : ℕ × ℕ := (0, 0)

/-- Distinct points Aj and Bj where Aj lies on x-axis and Bj on y = x^2 --/
variable A : ℕ → ℝ
variable B : ℕ → ℝ

/-- Equilateral triangle property --/
variable (a : ℕ → ℝ)
variable (x_squared : ℝ → ℝ := λ x, x * x)

axiom a1 : ∀ (n : ℕ), A n ≠ A (n + 1)
axiom a2 : ∀ (n : ℕ), B n = x_squared (A n)
axiom a3 : ∀ (n : ℕ) (m : ℕ), (m ≠ n) → (B m ≠ B n)
axiom eq_triangle : ∀ (n : ℕ), (n > 0) → a n = (2* A n - A (n - 1)) / √3

-- Initial conditions and recursion
def a_n_recurrence (n : ℕ) : ℝ :=
  match n with
  | 0     => 0
  | 1     => 2 / 3
  | (n+1) => 2 / 3 + a n

theorem least_n_for_length (n : ℕ) :
  ((n * (n + 1)) / 3) ≥ 50 → n ≥ 12 := by
  sorry

end least_n_for_length_l464_464589


namespace allowance_calculation_l464_464952

theorem allowance_calculation (A : ℝ)
  (h1 : (3 / 5) * A + (1 / 3) * (2 / 5) * A + 0.40 = A)
  : A = 1.50 :=
sorry

end allowance_calculation_l464_464952


namespace num_positive_integers_in_T_l464_464659

theorem num_positive_integers_in_T (T : Set ℕ) 
  (hT : ∀ n, n ∈ T ↔ n > 1 ∧ ∃ e_1 e_2 e_3 e_4 e_5 e_6,
   (1 : ℚ) / n = 0.e_1e_2e_3e_4e_5e_6e_1e_2e_3e_4e_5e_6...)
  (h1001: ∃ p1 p2, 1001 = p1 * p2 ∧ Prime p1 ∧ Prime p2) :
  Finset.card (Finset.filter (> 1) (Finset.divisors 999999)) = 47 :=
by sorry

end num_positive_integers_in_T_l464_464659


namespace sum_planar_angles_lt_360_l464_464704

theorem sum_planar_angles_lt_360 (n : ℕ) (α : Fin n → ℝ) (h : ∀ i, α i < 180) :
  ∑ i in Finset.range n, α i < 360 := 
sorry

end sum_planar_angles_lt_360_l464_464704


namespace truncated_cone_lateral_surface_area_truncated_cone_volume_l464_464866

def circumscribed_truncated_cone (r l : ℝ) :=
  ∃ r₁ r₂ : ℝ, r₁ + r₂ = l ∧ r₁ * r₂ = r^2

theorem truncated_cone_lateral_surface_area (r l : ℝ) :
  (circumscribed_truncated_cone r l) → 
  (π * l^2 = S_lateral) :=
begin
  sorry
end

theorem truncated_cone_volume (r l : ℝ) :
  (circumscribed_truncated_cone r l) → 
  (∃ H : ℝ, H = 2 * r) → 
  (∃ V : ℝ, (V = (2 * π * r * (l^2 - r^2)) / 3)) :=
begin
  sorry
end

end truncated_cone_lateral_surface_area_truncated_cone_volume_l464_464866


namespace expected_number_of_socks_taken_l464_464341

def expected_socks_taken (n : ℕ) : ℚ :=
  2 * (n + 1) / 3

theorem expected_number_of_socks_taken (n : ℕ) :
  let ξ := expected_socks_taken n in
  ξ = 2 * (n + 1) / 3 :=
by
  sorry

end expected_number_of_socks_taken_l464_464341


namespace square_area_ratio_l464_464327

theorem square_area_ratio (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    let s1 := 4 * a^2 * b^2 / (a^2 + b^2), s2 := 2 * b^2 in
    s1 / s2 = 2 * a^2 / (a^2 + b^2) :=
by
  sorry

end square_area_ratio_l464_464327


namespace number_of_cows_l464_464348

def each_cow_milk_per_day : ℕ := 1000
def total_milk_per_week : ℕ := 364000
def days_in_week : ℕ := 7

theorem number_of_cows : 
  (total_milk_per_week = 364000) →
  (each_cow_milk_per_day = 1000) →
  (days_in_week = 7) →
  (total_milk_per_week / (each_cow_milk_per_day * days_in_week)) = 52 :=
by
  sorry

end number_of_cows_l464_464348


namespace coordinates_of_F_double_prime_l464_464311

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

theorem coordinates_of_F_double_prime :
  let F : ℝ × ℝ := (3, 3)
  let F' := reflect_over_y_axis F
  let F'' := reflect_over_x_axis F'
  F'' = (-3, -3) :=
by
  sorry

end coordinates_of_F_double_prime_l464_464311


namespace negation_prop_l464_464281

open Classical

variable (x : ℝ)

theorem negation_prop :
    (∃ x : ℝ, x^2 + 2*x + 2 < 0) = False ↔
    (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by
    sorry

end negation_prop_l464_464281


namespace root_property_l464_464922

noncomputable def g (x : ℝ) : ℝ := 2 * x^2 * exp (2 * x) + log x

theorem root_property (x0 : ℝ) (hx0 : 0 < x0) (hx0_root : g x0 = 0) : 
  2 * x0 + log x0 = 0 := sorry

end root_property_l464_464922


namespace curve_crosses_itself_l464_464744

theorem curve_crosses_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ (t1^2 - 3 = t2^2 - 3) ∧ (t1^3 - 6*t1 + 2 = t2^3 - 6*t2 + 2) ∧
  ((t1^2 - 3 = 3) ∧ (t1^3 - 6*t1 + 2 = 2)) :=
by
  sorry

end curve_crosses_itself_l464_464744


namespace student_rank_from_right_l464_464812

theorem student_rank_from_right (total_students : ℕ) (rank_from_left : ℕ) (rank_from_right : ℕ) : 
  total_students = 21 → rank_from_left = 6 → rank_from_right = total_students - rank_from_left + 1 → rank_from_right = 17 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end student_rank_from_right_l464_464812


namespace total_weight_is_438_l464_464253

/-- Stan weighs 5 more pounds than Steve -/
def stan_weight (steve : ℤ) : ℤ := steve + 5

/-- Steve is 8 pounds lighter than Jim -/
def steve_weight (jim : ℤ) : ℤ := jim - 8

/-- Jim weighs 110 pounds -/
def jim_weight : ℤ := 110

/-- Tim weighs 12 pounds more than Stan -/
def tim_weight (stan : ℤ) : ℤ := stan + 12

theorem total_weight_is_438 :
  let jim := jim_weight;
      steve := steve_weight jim;
      stan := stan_weight steve;
      tim := tim_weight stan
  in (jim + steve + stan + tim) = 438 := by
  sorry

end total_weight_is_438_l464_464253


namespace find_c_l464_464537

-- Define the function
def f (c x : ℝ) : ℝ := x^4 - 8 * x^2 + c

-- Condition: The function has a minimum value of -14 on the interval [-1, 3]
def condition (c : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1 : ℝ) 3, ∀ y ∈ Set.Icc (-1 : ℝ) 3, f c x ≤ f c y ∧ f c x = -14

-- The theorem to be proved
theorem find_c : ∃ c : ℝ, condition c ∧ c = 2 :=
sorry

end find_c_l464_464537


namespace probability_bounds_l464_464666

noncomputable def roots_of_unity (n : ℕ) : set ℂ :=
  {z : ℂ | z ^ n = 1}

noncomputable def probability_satisfying (n : ℕ) (k : ℝ) (roots : set ℂ) : ℝ :=
  let pairs := {(v, w) | v ∈ roots ∧ w ∈ roots ∧ v ≠ w} in
  let eligible_pairs := {(v, w) | (v, w) ∈ pairs ∧ (2 + 2 * real.cos (2 * real.pi * ℝ.atan2 w.im w.re / n) ≥ k)} in
  eligible_pairs.card / pairs.card

theorem probability_bounds (n : ℕ) (k : ℝ) (h : k = 3 + real.sqrt 5) :
  probability_satisfying n k (roots_of_unity n) = 101 / 506 :=
by
  sorry

end probability_bounds_l464_464666


namespace min_a_plus_5b_l464_464471

theorem min_a_plus_5b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a * b + b^2 = b + 1) : 
  a + 5 * b ≥ 7 / 2 :=
by
  sorry

end min_a_plus_5b_l464_464471


namespace quadratic_trinomial_Inequality_l464_464907

theorem quadratic_trinomial_Inequality
  (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a + b + c = 1)
  (n : ℕ) (x : Fin n → ℝ)
  (h_prod : (∏ i, x i) = 1)
  (h_pos_seq : ∀ i, 0 < x i) :
  (∏ i, (a * (x i) ^ 2 + b * x i + c)) ≥ 1 :=
sorry

end quadratic_trinomial_Inequality_l464_464907


namespace animal_counts_l464_464259

-- Definitions based on given conditions
def ReptileHouse (R : ℕ) : ℕ := 3 * R - 5
def Aquarium (ReptileHouse : ℕ) : ℕ := 2 * ReptileHouse
def Aviary (Aquarium RainForest : ℕ) : ℕ := (Aquarium - RainForest) + 3

-- The main theorem statement
theorem animal_counts
  (R : ℕ)
  (ReptileHouse_eq : ReptileHouse R = 16)
  (A : ℕ := Aquarium 16)
  (V : ℕ := Aviary A R) :
  (R = 7) ∧ (A = 32) ∧ (V = 28) :=
by
  sorry

end animal_counts_l464_464259


namespace smallest_value_l464_464532

theorem smallest_value (x : ℝ) (y : ℝ) (h1 : y = 3 * x) (h2 : 0 < x) (h3 : x < 1) :
  (min (y) (min (y^2) (min (real.sqrt y) (1 / y))) = (real.sqrt y)) :=
by
  sorry

end smallest_value_l464_464532


namespace mismatching_socks_count_l464_464723

-- Define the conditions given in the problem
def total_socks : ℕ := 65
def pairs_matching_ankle_socks : ℕ := 13
def pairs_matching_crew_socks : ℕ := 10

-- Define the calculated counts as per the conditions
def matching_ankle_socks : ℕ := pairs_matching_ankle_socks * 2
def matching_crew_socks : ℕ := pairs_matching_crew_socks * 2
def total_matching_socks : ℕ := matching_ankle_socks + matching_crew_socks

-- The statement to prove
theorem mismatching_socks_count : total_socks - total_matching_socks = 19 := by
  sorry

end mismatching_socks_count_l464_464723


namespace min_value_of_f_is_5_l464_464737

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 + 4 * x + 5) + real.sqrt (x^2 - 2 * x + 10)

def pointA : ℝ × ℝ := (-2, -1)
def pointB : ℝ × ℝ := (1, 3)

theorem min_value_of_f_is_5 : ∃ x : ℝ, f(x) = 5 :=
sorry

end min_value_of_f_is_5_l464_464737


namespace ratio_of_surface_areas_l464_464363

theorem ratio_of_surface_areas 
  (r R : ℝ) 
  (h_side_length : ∃ r, (side_length : ℝ) = 3 * r)
  (h_R_def : R = sqrt 5 * r) :
  let S_inscribed := 4 * Real.pi * r^2 in
  let S_circumscribed := 4 * Real.pi * R^2 in
  S_circumscribed / S_inscribed = 5 :=
by
  sorry

end ratio_of_surface_areas_l464_464363


namespace find_positive_X_l464_464177

variable (X : ℝ) (Y : ℝ)

def hash_rel (X Y : ℝ) : ℝ :=
  X^2 + Y^2

theorem find_positive_X :
  hash_rel X 7 = 250 → X = Real.sqrt 201 :=
by
  sorry

end find_positive_X_l464_464177


namespace parallel_LS_PQ_l464_464625

open_locale big_operators

variables (A P M N C Q L S : Type*)

-- Assume that L is the intersection of lines AP and CM
def is_intersection_L (AP CM : Set (Set Type*)) : Prop :=
  L ∈ AP ∩ CM

-- Assume that S is the intersection of lines AN and CQ
def is_intersection_S (AN CQ : Set (Set Type*)) : Prop :=
  S ∈ AN ∩ CQ

-- Prove that LS ∥ PQ
theorem parallel_LS_PQ 
  (hL : is_intersection_L A P M C L) 
  (hS : is_intersection_S A N C Q S) : 
  parallel (line_through L S) (line_through P Q) :=
sorry

end parallel_LS_PQ_l464_464625


namespace q_investment_l464_464781

theorem q_investment (p_investment : ℕ) (ratio_pq : ℕ × ℕ) (profit_ratio : ℕ × ℕ) (hp : p_investment = 12000) (hpr : ratio_pq = (3, 5)) : 
  (∃ q_investment, q_investment = 20000) :=
  sorry

end q_investment_l464_464781


namespace general_term_formula_l464_464131

theorem general_term_formula (a S : ℕ → ℝ) (h : ∀ n, S n = (2 / 3) * a n + (1 / 3)) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = -2 * a (n - 1)) →
  ∀ n, a n = (-2)^(n - 1) :=
by
  sorry

end general_term_formula_l464_464131


namespace total_cost_of_dresses_l464_464121

-- Define the costs of each dress
variables (patty_cost ida_cost jean_cost pauline_cost total_cost : ℕ)

-- Given conditions
axiom pauline_cost_is_30 : pauline_cost = 30
axiom jean_cost_is_10_less_than_pauline : jean_cost = pauline_cost - 10
axiom ida_cost_is_30_more_than_jean : ida_cost = jean_cost + 30
axiom patty_cost_is_10_more_than_ida : patty_cost = ida_cost + 10

-- Statement to prove total cost
theorem total_cost_of_dresses : total_cost = pauline_cost + jean_cost + ida_cost + patty_cost 
                                 → total_cost = 160 :=
by {
  -- Proof is left as an exercise
  sorry
}

end total_cost_of_dresses_l464_464121


namespace problem_statement_l464_464061

noncomputable theory

variables (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (c_n : ℕ → ℝ) (T_n : ℕ → ℝ)

-- Conditions
def is_geometric (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_n n = 3^(n - 1)

def sum_first_n_terms_geometric (a_n : ℕ → ℝ) : Prop :=
  a_n 2 = 3 ∧ (a_n 0 + a_n 1 + a_n 2) = 13

def on_line (b_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (λ P : ℕ × ℕ, P.2 - P.1 = 2) (b_n n, b_n (n + 1))

-- c_n and T_n definitions
def cn_def (c_n : ℕ → ℝ) (a_n b_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c_n n = (b_n n) / (a_n n)

def Tn_def (T_n : ℕ → ℝ) (c_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, T_n n = (finset.range n).sum (λ k, c_n k)

-- Theorem to prove
theorem problem_statement (a_n b_n c_n T_n : ℕ → ℝ) :
  is_geometric a_n ∧ sum_first_n_terms_geometric a_n ∧ on_line b_n ∧ cn_def c_n a_n b_n ∧ Tn_def T_n c_n →
  (∀ n : ℕ, T_n n > 2 * a - 1) → a < 1 :=
by sorry

end problem_statement_l464_464061


namespace part_I_solution_set_part_II_range_a_l464_464496

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part_I_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} :=
by
  sorry

theorem part_II_range_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end part_I_solution_set_part_II_range_a_l464_464496


namespace determine_number_l464_464286

def S(n : ℕ) : ℕ := n.digits (λ a : List ℕ => List.sum a)

theorem determine_number :
  ∃ n : ℕ, (n < 10000 ∧ n ≥ 1000) ∧ n + S(n) = 2009 ∧ n = 1990 :=
by
  sorry

end determine_number_l464_464286


namespace satisfy_third_eq_l464_464703

theorem satisfy_third_eq 
  (x y : ℝ) 
  (h1 : x^2 - 3 * x * y + 2 * y^2 + x - y = 0)
  (h2 : x^2 - 2 * x * y + y^2 - 5 * x + 7 * y = 0) 
  : x * y - 12 * x + 15 * y = 0 :=
by
  sorry

end satisfy_third_eq_l464_464703


namespace decompose_fraction1_decompose_fraction2_l464_464862

-- Define the first problem as a theorem
theorem decompose_fraction1 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x^2 - 1)) = (1 / (x - 1)) - (1 / (x + 1)) :=
sorry  -- Proof required

-- Define the second problem as a theorem
theorem decompose_fraction2 (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 * x / (x^2 - 1)) = (1 / (x - 1)) + (1 / (x + 1)) :=
sorry  -- Proof required

end decompose_fraction1_decompose_fraction2_l464_464862


namespace not_otimes_square_l464_464076

def otimes (a b : ℝ) : ℝ :=
  if a ≥ b then a else b

theorem not_otimes_square (a b: ℝ) : (otimes a b)^2 ≠ otimes (a^2) (b^2) := 
sorry

end not_otimes_square_l464_464076


namespace chronological_order_correct_l464_464695

inductive Event
| I | A | J | C | B | M | H | O | D | E | F | L | N | G | K
  deriving DecidableEq, Repr

open Event

def eventYear : Event → ℕ
| I => 1900
| A => 1908
| J => 1931
| C => 1937
| B => 1942 -- assume the earliest year in range
| M => 1950
| H => 1956
| O => 1960
| D => 1969
| E => 1974
| F => 1976
| L => 1992
| N => 1994
| G => 1998
| K => 2000

def sortedEvents := [I, A, J, C, B, M, H, O, D, E, F, L, N, G, K]

theorem chronological_order_correct :
  list.map eventYear sortedEvents = list.range' 1900 101 [1900, 1908, 1931, 1937, 1942, 1950, 1956, 1960, 1969, 1974, 1976, 1992, 1994, 1998, 2000] :=
by
  sorry

end chronological_order_correct_l464_464695


namespace reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464391

-- Probability that the bags end up in the shed in the reverse order
def probability_reverse_order : ℕ → ℚ
| 0 := 1
| (n + 1) := (1 / 2) * probability_reverse_order n

theorem reverse_order_probability_is_1_over_8 :
  probability_reverse_order 3 = 1 / 8 :=
sorry

-- Probability that the second bag from the bottom ends up as the bottom bag in the shed
def probability_second_from_bottom_bottom : ℚ := 
(1 / 2) * (1 / 2) * (1 / 2)

theorem second_from_bottom_as_bottom_is_1_over_8 :
  probability_second_from_bottom_bottom = 1 / 8 :=
sorry

end reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464391


namespace f_g_x_ne_zero_l464_464963

theorem f_g_x_ne_zero {x : ℕ} (h : x > 0) : 
  let f (y : ℝ) := 1 / y 
  let g (n : ℕ) := n^2 + 3 * n - 2 
  f (g x) ≠ 0 := 
by
  sorry

end f_g_x_ne_zero_l464_464963


namespace kenny_played_basketball_last_week_l464_464582

def time_practicing_trumpet : ℕ := 40
def time_running : ℕ := time_practicing_trumpet / 2
def time_playing_basketball : ℕ := time_running / 2
def answer : ℕ := 10

theorem kenny_played_basketball_last_week :
  time_playing_basketball = answer :=
by
  -- sorry to skip the proof
  sorry

end kenny_played_basketball_last_week_l464_464582


namespace pq_implications_l464_464973

theorem pq_implications (p q : Prop) (hpq_or : p ∨ q) (hpq_and : p ∧ q) : p ∧ q :=
by
  sorry

end pq_implications_l464_464973


namespace probability_square_tile_die_l464_464004

/-
  Define the event of getting a product that is a perfect square
  when choosing a tile from 1 to 15 and a number from a die (1 to 10).
-/

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def possibleTiles := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
def dieFaces := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def favorableOutcomes :=
  { (t, d) ∈ possibleTiles × dieFaces | isPerfectSquare (t * d)}

def totalOutcomes := possibleTiles × dieFaces

noncomputable def probabilityPerfectSquare : ℚ :=
  (favorableOutcomes.toFinset.card : ℚ) / (totalOutcomes.toFinset.card : ℚ)

theorem probability_square_tile_die : probabilityPerfectSquare = 19 / 150 :=
by
  sorry

end probability_square_tile_die_l464_464004


namespace train_speed_is_approx_36_003_kmph_l464_464801

theorem train_speed_is_approx_36_003_kmph :
  let length := 30 -- in meters
  let time := 2.9997600191984644 -- in seconds
  let speed_mps := length / time -- speed in meters per second
  let conversion_factor := 3.6
  let speed_kmph := speed_mps * conversion_factor -- speed in kilometers per hour
  abs (speed_kmph - 36.003) < 0.001 :=
by
  sorry

end train_speed_is_approx_36_003_kmph_l464_464801


namespace smallest_positive_period_f_range_and_increasing_interval_of_f_l464_464071

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * (sin x ^ 2 - cos x ^ 2) + 2 * sin x * cos x

-- Smallest positive period of f(x) is π
theorem smallest_positive_period_f : smallest_positive_period f = π := 
sorry

-- Given the interval and required properties.
theorem range_and_increasing_interval_of_f :
  ∀ x, x ∈ Icc (-π / 3) (π / 3) → 
       (f x ∈ Icc (-2) (sqrt 3)) ∧ 
       (strictly_increasing_on f (Icc (-π / 12) (π / 3))) := 
sorry

end smallest_positive_period_f_range_and_increasing_interval_of_f_l464_464071


namespace parallel_ls_pq_l464_464609

variables {A P C M N Q L S : Type*}
variables [affine_space ℝ (A P C M N Q)]

/- Definitions of the intersection points -/
def is_intersection_point (L : Type*) (AP : set (A P))
  (CM : set (C M)) : Prop := L ∈ AP ∧ L ∈ CM

def is_intersection_point' (S : Type*) (AN : set (A N))
  (CQ : set (C Q)) : Prop := S ∈ AN ∧ S ∈ CQ

/- Main Theorem -/
theorem parallel_ls_pq (AP CM AN CQ : set (A P))
  (PQ : set (P Q)) (L S : Type*)
  (hL : is_intersection_point L AP CM)
  (hS : is_intersection_point' S AN CQ) : L S ∥ P Q :=
sorry

end parallel_ls_pq_l464_464609


namespace local_minimum_condition_l464_464968

-- Define the function f(x)
def f (x b : ℝ) : ℝ := x ^ 3 - 3 * b * x + 3 * b

-- Define the first derivative of f(x)
def f_prime (x b : ℝ) : ℝ := 3 * x ^ 2 - 3 * b

-- Define the second derivative of f(x)
def f_double_prime (x b : ℝ) : ℝ := 6 * x

-- Theorem stating that f(x) has a local minimum if and only if b > 0
theorem local_minimum_condition (b : ℝ) (x : ℝ) (h : f_prime x b = 0) : f_double_prime x b > 0 ↔ b > 0 :=
by sorry

end local_minimum_condition_l464_464968


namespace LS_parallel_PQ_l464_464599

noncomputable def L (A P C M : Point) : Point := 
  -- Intersection of AP and CM
  sorry

noncomputable def S (A N C Q : Point) : Point := 
  -- Intersection of AN and CQ
  sorry

theorem LS_parallel_PQ (A P C M N Q : Point) 
  (hL : L A P C M) (hS : S A N C Q) : 
  Parallel (Line (L A P C M)) (Line (S A N C Q)) :=
sorry

end LS_parallel_PQ_l464_464599


namespace more_oil_l464_464362

noncomputable def original_price (P : ℝ) :=
  P - 0.3 * P = 70

noncomputable def amount_of_oil_before (P : ℝ) :=
  700 / P

noncomputable def amount_of_oil_after :=
  700 / 70

theorem more_oil (P : ℝ) (h1 : original_price P) :
  (amount_of_oil_after - amount_of_oil_before P) = 3 :=
  sorry

end more_oil_l464_464362


namespace problem_solution_product_of_distances_ratio_of_distances_l464_464080

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ :=
  ((1 / 2), 0)

noncomputable def line_through_focus (m : ℝ) (y : ℝ) : ℝ :=
  m * y + (1 / 2)

noncomputable def parabola_intersection_points (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let d := real.sqrt (1 + m^2)
  let y₁ := m * d + 1
  let y₂ := m * (-d) + 1
  (((line_through_focus m y₁), y₁), ((line_through_focus m y₂), y₂))

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem problem_solution (m : ℝ) (A B : ℝ × ℝ) (F : ℝ × ℝ)
  (h₁ : B = parabola_focus 2)
  (h₂ : parabola_intersection_points m = (A, F))
  (h₃ : distance A.1 A.2 B.1 B.2 = 3)
  (h4 : abs (F.fst) > abs (A.fst)) :
  abs (distance F.1 F.2 A.1 A.2 - distance F.1 F.2 B.1 B.2) = real.sqrt 3 :=
begin
  sorry
end

theorem product_of_distances (A B F : ℝ × ℝ)
  (h₁ : distance A.1 A.2 B.1 B.2 = 3)
  (h₂ : parabola_focus 2 = F) :
  (distance A.1 A.2 F.1 F.2) * (distance F.1 F.2 B.1 B.2) = 3 / 2 :=
begin
  sorry
end

theorem ratio_of_distances (A B F : ℝ × ℝ) 
  (h₁ : distance F.fst F.snd A.fst A.snd = (3 + abs (real.sqrt 3)) / 2)
  (h₂ : distance F.fst F.snd B.fst B.snd = (3 - abs (real.sqrt 3)) / 2):
  distance F.fst F.snd A.fst A.snd / distance F.fst F.snd B.fst B.snd = 2 + real.sqrt 3 :=
begin
  sorry
end

end problem_solution_product_of_distances_ratio_of_distances_l464_464080


namespace smallest_real_number_l464_464018

def satisfies_conditions (x : ℝ) : Prop :=
  x > 1 ∧ sin x = sin (x^2 + 30) ∧ cos x > 0.5

theorem smallest_real_number :
  (∀ x : ℝ, satisfies_conditions x → x ≥ 14) ∧ satisfies_conditions 14 :=
by
  sorry

end smallest_real_number_l464_464018


namespace variance_of_binomial_l464_464982

-- Definitions and problem setup
def n : ℕ := 7
def p : ℚ := 3 / 7

-- Problem statement
theorem variance_of_binomial :
  let X := binomial_var n p in
  variance X = 12 / 7 :=
by sorry

end variance_of_binomial_l464_464982


namespace solve_z4_plus_16_eq_0_l464_464019

noncomputable def solutions (z : ℂ) : Prop :=
  z^4 + 16 = 0

theorem solve_z4_plus_16_eq_0 :
  { z : ℂ | solutions z } = { √2 + √2 * complex.I, -√2 + √2 * complex.I, -√2 - √2 * complex.I, √2 - √2 * complex.I } :=
by
  sorry

end solve_z4_plus_16_eq_0_l464_464019


namespace Linda_total_amount_at_21_years_l464_464686

theorem Linda_total_amount_at_21_years (P : ℝ) (r : ℝ) (n : ℕ) (initial_principal : P = 1500) (annual_rate : r = 0.03) (years : n = 21):
    P * (1 + r)^n = 2709.17 :=
by
  sorry

end Linda_total_amount_at_21_years_l464_464686


namespace triangle_CE_eq_BF_l464_464978

theorem triangle_CE_eq_BF
  (A B C : ℝ) [hA2 : has_rat_cast ℝ (2 : ℚ)] -- To use fractional values in Lean
  {α β γ : ℝ} (α_pos : 0 < α) (β_pos : 0 < β) (γ_pos : 0 < γ)
  (sum_angles : α + β + γ = 180)
  (E F : ℝ)
  (BC : ℝ) {angle_BCF angle_CBE : ℝ}
  (angle_BCF_eq : angle_BCF = α / 2)
  (angle_CBE_eq : angle_CBE = α / 2) :
  CE = BF :=
by
  sorry

end triangle_CE_eq_BF_l464_464978


namespace prove_parallel_l464_464646

-- Define points A, P, C, M, N, Q, and functions L, S.
variables {A P C M N Q L S : Point}
variables [incidence_geometry]

-- Define L and S based on the given conditions.
def L := intersection (line_through A P) (line_through C M)
def S := intersection (line_through A N) (line_through C Q)

-- Define the condition to prove LS is parallel to PQ.
theorem prove_parallel : parallel (line_through L S) (line_through P Q) :=
sorry

end prove_parallel_l464_464646


namespace three_digit_numbers_divisible_by_17_l464_464100

theorem three_digit_numbers_divisible_by_17 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_17 := {n : ℕ | n % 17 = 0}
  ∃! count : ℕ, count = fintype.card {n : ℕ // n ∈ three_digit_numbers ∧ n ∈ divisible_by_17} ∧ count = 53 :=
begin
  sorry
end

end three_digit_numbers_divisible_by_17_l464_464100


namespace pyramid_height_correct_l464_464364

noncomputable def pyramid_height :=
  let w := 20 / 3 in
  let l := 40 / 3 in
  let d := Real.sqrt (w^2 + l^2) in
  let fb := d / 2 in
  let pb := 10 in
  Real.sqrt (pb^2 - fb^2)

theorem pyramid_height_correct :
  let w := 20 / 3 in
  let l := 40 / 3 in
  let d := Real.sqrt (w^2 + l^2) in
  let fb := d / 2 in
  let pb := 10 in
  pyramid_height = 20 / 3 :=
by
  sorry

end pyramid_height_correct_l464_464364


namespace remainder_polynomial_find_positive_integers_l464_464708

noncomputable def p (n : ℕ) : (ℕ → ℕ) := λ x, (Finset.range n).sum (λ k, x^k)

theorem remainder_polynomial (m n i : ℕ) : ∃ (p_i : ℕ → ℕ), 
  p_i = p (n % m) ∧ (∀ x, ∃ q r, p (n) x = q * p (m) x + r ∧ r = p_i x) :=
  sorry

theorem find_positive_integers (i j k : ℕ) : 
  i + 2 * j + 4 * k = 100 ∧ i > 0 ∧ j > 0 ∧ k > 0→ 
  p (100) = p (i) + p (j) ∘ (λ x, x^2) + p (k) ∘ (λ x, x^4) :=
  sorry

end remainder_polynomial_find_positive_integers_l464_464708


namespace general_term_a_seq_sum_b_seq_terms_l464_464046

noncomputable def a_seq (n : ℕ) : ℝ := (1/2)^n

theorem general_term_a_seq (n : ℕ) : 
  let S_n := (∑ i in Finset.range (n+1), a_seq i) in
  S_n = 1 - a_seq n :=
by
  sorry

noncomputable def b_seq (n : ℕ) : ℝ := Real.logb 2 (a_seq n)

theorem sum_b_seq_terms (n : ℕ) : 
  let T_n := (∑ k in Finset.range n, 1 / (b_seq k * b_seq (k+1))) in 
  T_n = n / (n + 1) :=
by
  sorry

end general_term_a_seq_sum_b_seq_terms_l464_464046


namespace tan_B_value_side_c_value_l464_464140

-- Define the conditions as given in the problem
variables (A B C : ℝ) (a b c : ℝ) (triangle_abc : ∀ A B C : ℝ, 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧ A + B + C = π)
variables (sin_A : ℝ) (tan_AB_mB : ℝ)

-- Given conditions
def conditions : Prop :=
  sin_A = 3 / 5 ∧ tan_AB_mB = -1 / 2

-- Prove the value of tan B
theorem tan_B_value (h : conditions) : tan B = 2 :=
sorry

-- Prove the value of c given additional condition b = 5
theorem side_c_value (h : conditions) (h_tanB : tan B = 2) (h_b : b = 5) : c = 11 / 2 :=
sorry

end tan_B_value_side_c_value_l464_464140


namespace prove_parallel_l464_464642

-- Define points A, P, C, M, N, Q, and functions L, S.
variables {A P C M N Q L S : Point}
variables [incidence_geometry]

-- Define L and S based on the given conditions.
def L := intersection (line_through A P) (line_through C M)
def S := intersection (line_through A N) (line_through C Q)

-- Define the condition to prove LS is parallel to PQ.
theorem prove_parallel : parallel (line_through L S) (line_through P Q) :=
sorry

end prove_parallel_l464_464642


namespace three_digit_numbers_div_by_17_l464_464102

theorem three_digit_numbers_div_by_17 : 
  let k_min := 6
  let k_max := 58
  (k_max - k_min + 1) = 53 := by
  let k_min := 6
  let k_max := 58
  show (k_max - k_min + 1) = 53 from sorry

end three_digit_numbers_div_by_17_l464_464102


namespace trigonometric_relationship_l464_464891

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < (π / 2))
variable (hβ : 0 < β ∧ β < π)
variable (h : Real.tan α = Real.cos β / (1 - Real.sin β))

theorem trigonometric_relationship : 
    2 * α - β = π / 2 :=
sorry

end trigonometric_relationship_l464_464891


namespace average_minutes_per_person_l464_464798

variables (num_audience : ℕ) (lecture_length : ℕ)
variables (listened_entire : ℕ) (listened_none : ℕ) 
variables (listened_half : ℕ) (listened_quarter : ℕ) (listened_three_quarters : ℕ)

-- Definition of percentages
def percent_full (n : ℕ) : ℕ := n * num_audience / 100
def percent_listened_entire := percent_full 30
def percent_listened_none := percent_full 15
def remaining_audience := num_audience - percent_listened_entire - percent_listened_none

-- Definition of fractions of remaining audience
def fraction_remaining (fraction : ℕ) : ℕ := fraction * remaining_audience / 3
def listened_half_remaining := fraction_remaining 1
def listened_quarter_remaining := fraction_remaining 1
def listened_three_quarters_remaining := remaining_audience - listened_half_remaining - listened_quarter_remaining

-- Calculation of total minutes heard
noncomputable def total_minutes_heard : ℝ :=
  percent_listened_entire * lecture_length +
  0 +
  listened_half_remaining * (lecture_length / 2) +
  listened_quarter_remaining * (lecture_length / 4) +
  listened_three_quarters_remaining * (3 * lecture_length / 4)

def average_minutes_heard : ℝ := total_minutes_heard / num_audience

-- Proof Statement
theorem average_minutes_per_person (h1 : num_audience = 100) (h2 : lecture_length = 90) :
  average_minutes_heard num_audience lecture_length
                        percent_listened_entire percent_listened_none
                        listened_half_remaining listened_quarter_remaining
                        listened_three_quarters_remaining = 52 :=
by sorry

end average_minutes_per_person_l464_464798


namespace ian_riding_time_l464_464523

variables {r s d t : ℝ}

-- Given conditions as assumptions
def condition_1 : d = 120 * r := sorry
def condition_2 : d = 60 * (r + s) := sorry

-- Lean statement to prove the required time
theorem ian_riding_time : s = r → t = 120 :=
begin
  intro h,
  have h1 : 120 * r = 60 * (r + s), from sorry,
  have h2 : s = r, by linarith,
  rw [h2] at h1,
  have h3 : t = d / s, from sorry,
  rw [condition_1, h2] at h3,
  have h4 : t = 120, by linarith,
  exact h4,
end

end ian_riding_time_l464_464523


namespace monotonic_intervals_a0_extreme_points_tangency_x_axis_l464_464491

open Real

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x * log x + a * x + a ^ 2 - a - 1) * exp x 

theorem monotonic_intervals_a0 :
  let f0 (x : ℝ) := (x * log x - 1) * exp x in
  ∀ x, (x > 0) → 
    if x < 1 then f0' x < 0 ∧ (∀ y, (y < 1) → f0' y < 0) else 
    f0' x > 0 ∧ (∀ y, (y > 1) → f0' y > 0) := 
begin
  sorry
end

theorem extreme_points (a : ℝ) (h : a ≥ -2) :
  if -2 ≤ a ∧ a ≤ -1 - 1 / exp 1 ∨ a ≥ 1 then 
    ∀ x, x ∈ Ioi (1 / exp 1) → f' x a ≠ 0 else
    ∃ x₀, x₀ ∈ Ioi (1 / exp 1) ∧ f' x₀ a = 0 :=
begin
  sorry
end

theorem tangency_x_axis (a : ℝ) : 
  (∃ x₀, x₀ ∈ Ioi (1 / exp 1) ∧ (f x₀ a = 0 ∧ f' x₀ a = 0)) ↔ (a = -1) := 
begin
  sorry
end

end monotonic_intervals_a0_extreme_points_tangency_x_axis_l464_464491


namespace triangle_perimeter_l464_464277

theorem triangle_perimeter 
  (a b c : ℝ)
  (r : ℝ)
  (h1 : b = 16 * r)
  (h2 : c = 16 * r^2)
  (h3 : a = 16)
  (trig_identity : (sin a - 2 * sin b + 3 * sin c) / (sin c - 2 * sin b + 3 * sin a) = 19/9)
  (law_of_sines : a / sin a = b / sin b ∧ b / sin b = c / sin c ∧ a / sin a = 2 * circumradius) :
  a + b + c = 76 :=
sorry

end triangle_perimeter_l464_464277


namespace sum_f_eq_96_l464_464422

def f (x : ℝ) : ℝ := 
  2 * Real.sin x * Real.sin 2 * (1 + Real.sec (x - 2) * Real.sec (x + 2))

theorem sum_f_eq_96 : 
  ∑ x in Finset.range (46 - 2 + 1), f (x + 2) = 96 :=
sorry

end sum_f_eq_96_l464_464422


namespace next_perfect_number_after_6_l464_464534

def is_prime (n : ℕ) : Prop :=
  2 <= n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_perfect_number (n : ℕ) : Prop :=
  n = ∑ m in (finset.range n).filter (∣ n), m

theorem next_perfect_number_after_6 : ∃ n : ℕ, n > 6 ∧ is_perfect_number n :=
begin
  let n := 3,
  have h1 : is_prime (2^n - 1), by sorry,  -- Prove that 7 is prime.
  let candidate := 2^(n-1) * (2^n - 1),
  have h2 : candidate = 28, by norm_num,
  have h3 : candidate > 6, by norm_num,
  use candidate,
  split,
  { exact h3, },
  { unfold is_perfect_number,
    rw h2,
    -- Prove that 28 is a perfect number.
    sorry,
  },
end

end next_perfect_number_after_6_l464_464534


namespace three_digit_multiples_of_30_not_75_count_l464_464092

theorem three_digit_multiples_of_30_not_75_count : 
  let nums := {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 30 = 0}
  let multiples_of_75 := {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 75 = 0}
  let valid_nums := nums \ multiples_of_75
  24 = valid_nums.size :=
by
  sorry

end three_digit_multiples_of_30_not_75_count_l464_464092


namespace distinct_solution_count_eq_2_l464_464867

theorem distinct_solution_count_eq_2 :
  ∃! (pairs : Set (ℝ × ℝ)), 
    (∀ (x y : ℝ), ((x, y) ∈ pairs ↔ (x = x^2 + 2*y^2 ∧ y = 2*x^2*y)))
    ∧ (pairs.card = 2) := 
sorry

end distinct_solution_count_eq_2_l464_464867


namespace roots_polynomial_sum_squares_l464_464186

theorem roots_polynomial_sum_squares (p q r : ℝ) 
  (h_roots : ∀ x : ℝ, x^3 - 15 * x^2 + 25 * x - 10 = 0 → x = p ∨ x = q ∨ x = r) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 350 := 
by {
  sorry
}

end roots_polynomial_sum_squares_l464_464186


namespace find_divisor_l464_464013

theorem find_divisor (D Q: ℕ) (hD : D = 62976) (hQ : Q = 123) : D / Q = 512 :=
by
  rw [hD, hQ]
  norm_num
  exact 512

end find_divisor_l464_464013


namespace prob_reverse_order_prob_second_from_bottom_l464_464387

section CementBags

-- Define the basic setup for the bags and worker choices
variables (bags : list ℕ) (choices : list ℕ) (shed gate : list ℕ)

-- This ensures we are indeed considering 4 bags in a sequence
axiom bags_is_4 : bags.length = 4

-- This captures that the worker makes 4 sequential choices with equal probability for each, where 0 represents shed and 1 represents gate
axiom choices_is_4_and_prob : (choices.length = 4) ∧ (∀ i in choices, i ∈ [0, 1])

-- This condition captures that eventually, all bags must end up in the shed
axiom all_bags_in_shed : ∀ b ∈ bags, b ∈ shed

-- Prove for part (a): Probability of bags in reverse order in shed is 1/8
theorem prob_reverse_order : (choices = [0, 0, 0, 0] → (reverse bags = shed)) → 
  (probability (choices = [0, 0, 0, 0]) = 1 / 8) :=
 sorry

-- Prove for part (b): Probability of the second-from-bottom bag in the bottom is 1/8
theorem prob_second_from_bottom : 
  (choices = [1, 1, 0, 0] → (shed.head = bags.nth 1.get_or_else 0)) → 
  (probability (choices = [1, 1, 0, 0]) = 1 / 8) :=
 sorry

end CementBags

end prob_reverse_order_prob_second_from_bottom_l464_464387


namespace determine_k_for_quadratic_eq_l464_464854

theorem determine_k_for_quadratic_eq :
  ∃ k : ℝ, (∀ x : ℝ, x^2 + 4 * x * real.sqrt 2 + k = 0 → (4 * real.sqrt 2)^2 - 4 * 1 * k = 0) ∧ k = 8 :=
by
  sorry

end determine_k_for_quadratic_eq_l464_464854


namespace LS_parallel_PQ_l464_464640

universe u
variables {Point : Type u} [Geometry Point]

-- Defining points A, P, C, M, N, Q as elements of type Point
variables (A P C M N Q : Point)

-- Let L be the intersection point of the lines AP and CM
def L := Geometry.intersection (Geometry.line_through A P) (Geometry.line_through C M)

-- Let S be the intersection point of the lines AN and CQ
def S := Geometry.intersection (Geometry.line_through A N) (Geometry.line_through C Q)

-- Proof: LS is parallel to PQ
theorem LS_parallel_PQ : Geometry.parallel (Geometry.line_through L S) (Geometry.line_through P Q) :=
sorry

end LS_parallel_PQ_l464_464640


namespace find_a_find_b_employees_receiving_rewards_l464_464994

def sales_data : List ℝ := [5.0, 9.9, 6.0, 5.2, 8.2, 6.2, 7.6, 9.4, 8.2, 7.8,
                            5.1, 7.5, 6.1, 6.3, 6.7, 7.9, 8.2, 8.5, 9.2, 9.8]

def frequency_distribution : List ℕ := [3, 5, 4, 4, 4]

def mean_sales : ℝ := 7.44
def mode_sales : ℝ := 8.2
def employee_a_sales : ℝ := 7.5

theorem find_a : frequency_distribution.nth 2 = some 4 :=
by sorry

theorem find_b : (sales_data ≠ []) ∧ median sales_data = 7.7 :=
by sorry

theorem employees_receiving_rewards : sales_data.count (λ x, x ≥ 7) = 12 :=
by sorry

end find_a_find_b_employees_receiving_rewards_l464_464994


namespace mean_temperature_equal_20_l464_464443

theorem mean_temperature_equal_20 :
  (let temp1 := 21
       temp2 := 15
       days1 := 25
       days2 := 5
       total_days := days1 + days2
       total_temp := (days1 * temp1) + (days2 * temp2)
    in total_temp / total_days = 20) :=
by
  sorry

end mean_temperature_equal_20_l464_464443


namespace negation_of_proposition_l464_464282

-- Given condition
def original_statement (a : ℝ) : Prop :=
  ∃ x : ℝ, a*x^2 - 2*a*x + 1 ≤ 0

-- Correct answer (negation statement)
def negated_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - 2*a*x + 1 > 0

-- Statement to prove
theorem negation_of_proposition (a : ℝ) :
  ¬ (original_statement a) ↔ (negated_statement a) :=
by 
  sorry

end negation_of_proposition_l464_464282


namespace no_x_for_arccos_arcsin_eq_l464_464489

theorem no_x_for_arccos_arcsin_eq :
  ¬ ∃ x : ℝ, arccos (4 / 5) - arccos (-4 / 5) = arcsin x :=
by
  sorry

end no_x_for_arccos_arcsin_eq_l464_464489


namespace LS_parallel_PQ_l464_464633

universe u
variables {Point : Type u} [Geometry Point]

-- Defining points A, P, C, M, N, Q as elements of type Point
variables (A P C M N Q : Point)

-- Let L be the intersection point of the lines AP and CM
def L := Geometry.intersection (Geometry.line_through A P) (Geometry.line_through C M)

-- Let S be the intersection point of the lines AN and CQ
def S := Geometry.intersection (Geometry.line_through A N) (Geometry.line_through C Q)

-- Proof: LS is parallel to PQ
theorem LS_parallel_PQ : Geometry.parallel (Geometry.line_through L S) (Geometry.line_through P Q) :=
sorry

end LS_parallel_PQ_l464_464633


namespace probability_sum_leq_12_l464_464757

theorem probability_sum_leq_12 (dice1 dice2 : ℕ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 8) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 8) :
  (∃ outcomes : ℕ, (outcomes = 64) ∧ 
   (∃ favorable : ℕ, (favorable = 54) ∧ 
   (favorable / outcomes = 27 / 32))) :=
sorry

end probability_sum_leq_12_l464_464757


namespace perimeter_of_triangle_l464_464566

theorem perimeter_of_triangle 
  (A : ℝ) (A_eq : A = 2 * real.pi / 3)
  (a : ℝ) (a_eq : a = 4)
  (D_is_midpoint : ∀ B C D : Type, is_midpoint D B C)
  (AD : ℝ) (AD_eq : AD = sqrt 2)
  (b : ℝ) (c : ℝ)
  (eq1 : a^2 = b^2 + c^2 + b * c)
  (eq2 : (sqrt 2)^2 = (1 / 4) * (b^2 + c^2 + 2 * b * c)) :
  a + (b + c) = 4 + 2 * sqrt 5 := 
by 
  sorry

end perimeter_of_triangle_l464_464566


namespace distance_between_cities_l464_464314

-- Define the parameters from the problem
variables (T : ℝ) (D : ℝ)

-- Define constants for speeds and time difference
def speed_faster := 78   -- Speed of the faster car in km/h
def speed_slower := 72   -- Speed of the slower car in km/h
def time_difference := 1 / 3  -- Time difference in hours

-- Define the conditions as equations
def eq_faster_car : Prop := D = speed_faster * T
def eq_slower_car : Prop := D = speed_slower * (T + time_difference)

-- Statement of the problem
theorem distance_between_cities
  (h_faster : eq_faster_car)
  (h_slower : eq_slower_car) :
  D = 312 :=
sorry

end distance_between_cities_l464_464314


namespace sqrt_equation_solution_l464_464021

theorem sqrt_equation_solution (x : ℝ) (h : sqrt (4 - 5 * x) = 3) : x = -1 := 
by
  sorry

end sqrt_equation_solution_l464_464021


namespace solve_for_x_l464_464716

theorem solve_for_x (x : ℝ) (h1 : 5^(3 * x) = Real.sqrt 125) :
  x = 1/2 :=
by
  -- Assuming necessary equivalences for simplifications
  have h2 : Real.sqrt 125 = 125^(1/2) := sorry,
  have h3 : 125 = 5^3 := sorry,
  have h4 : 125^(1/2) = (5^3)^(1/2) := by rw [h3],
  have h5 : (5^3)^(1/2) = 5^(3/2) := Real.rpow_mul 5 3 (1/2),
  rw [h2, h4, h5] at h1,
  -- Solve exponents after simplifying equal bases
  have h6 : 5^(3 * x) = 5^(3/2) := h1,
  have h7 : 3 * x = 3 / 2 := sorry,
  field_simp at h7,
  exact h7,
  sorry

end solve_for_x_l464_464716


namespace probability_one_first_class_product_l464_464317

-- Define the probabilities for the interns processing first-class products
def P_first_intern_first_class : ℚ := 2 / 3
def P_second_intern_first_class : ℚ := 3 / 4

-- Define the events 
def P_A1 : ℚ := P_first_intern_first_class * (1 - P_second_intern_first_class)
def P_A2 : ℚ := (1 - P_first_intern_first_class) * P_second_intern_first_class

-- Probability of exactly one of the two parts being first-class product
def P_one_first_class_product : ℚ := P_A1 + P_A2

-- Theorem to be proven: the probability is 5/12
theorem probability_one_first_class_product : 
    P_one_first_class_product = 5 / 12 :=
by
  -- Proof goes here
  sorry

end probability_one_first_class_product_l464_464317


namespace abc_sum_is_51_l464_464529

noncomputable def find_abc_sum : ℕ → ℕ → ℕ → Prop :=
  λ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ ab + c = 50 ∧ ac + b = 50 ∧ bc + a = 50 → a + b + c = 51

theorem abc_sum_is_51 (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) : (ab + c = 50) → (ac + b = 50) → (bc + a = 50) → a + b + c = 51 :=
  begin
    sorry -- Proof required
  end

end abc_sum_is_51_l464_464529


namespace jemma_sold_400_frames_l464_464417

-- Let J and D represent the number of frames Jemma and Dorothy sell, respectively.
def Jemma_frames (J D : ℕ) : Prop :=
  -- 1. Dorothy sells glass frames at half the price that Jemma sells them.
  -- Implicit in the prices: Dorothy's price is 2.5 dollars, Jemma's price is 5 dollars.
  -- 2. Jemma sells the glass frames at $5 each.
  -- 3. Jemma sells twice as many frames as Dorothy does.
  (J = 2 * D) ∧
  -- 4. They made $2500 together in total from the sale of the glass frames.
  (5 * J + 2.5 * D = 2500)

theorem jemma_sold_400_frames (J D : ℕ) (h : Jemma_frames J D) : J = 400 :=
by
  sorry

end jemma_sold_400_frames_l464_464417


namespace collinearity_condition_minimum_distance_l464_464084

-- Definitions and Problem 1
theorem collinearity_condition (a b : ℝ^3) (t : ℝ)
    (h_a_nonzero : a ≠ 0) 
    (h_b_nonzero : b ≠ 0) 
    (h_a_b_not_collinear : ¬ (∃ k : ℝ, a = k • b) ) :
    (∃ λ : ℝ, t • b - a = λ • (1/3 • b - 2/3 • a)) ↔ t = 1/2 :=
sorry

-- Definitions and Problem 2
theorem minimum_distance (a b : ℝ^3) (t : ℝ)
    (h_a_eq_b_magnitude : ∥a∥ = ∥b∥) 
    (h_angle_60 : inner a b = ∥a∥ * ∥b∥ * (1/2)) :
    (∀ t' : ℝ, ∥a - t' • b∥ ≥ ∥a - 1/2 • b∥) :=
sorry

end collinearity_condition_minimum_distance_l464_464084


namespace fraction_sum_eq_one_l464_464966

noncomputable def log10 (x : ℝ) := real.log x / real.log 10

theorem fraction_sum_eq_one (a b : ℝ) (h0 : 0 < a) (h1 : 0 < b)
  (h2 : real.log a / real.log 2 = real.log b / real.log 5)
  (h3 : real.log a / real.log 2 = log10 (a + b)) :
  (1 / a) + (1 / b) = 1 :=
by sorry

end fraction_sum_eq_one_l464_464966


namespace negation_all_children_careful_equiv_at_least_one_child_reckless_l464_464070

-- Definitions based on the problem conditions
def all_adults_careful := ∀ x, adult x → careful_investor x
def some_adults_careful := ∃ x, adult x ∧ careful_investor x
def no_children_careful := ∀ x, child x → ¬careful_investor x
def all_children_reckless := ∀ x, child x → reckless_investor x
def at_least_one_child_reckless := ∃ x, child x ∧ reckless_investor x
def all_children_careful := ∀ x, child x → careful_investor x

-- Theorems to be proven
theorem negation_all_children_careful_equiv_at_least_one_child_reckless :
  (¬ all_children_careful) ↔ at_least_one_child_reckless :=
sorry

end negation_all_children_careful_equiv_at_least_one_child_reckless_l464_464070


namespace three_digit_numbers_divisible_by_17_l464_464098

theorem three_digit_numbers_divisible_by_17 :
  ∃ n : ℕ, n = 53 ∧ ∀ k : ℕ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ↔ k ∈ finset.Icc 6 58 :=
by
  sorry

end three_digit_numbers_divisible_by_17_l464_464098


namespace odd_function_find_range_l464_464027

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x - 1 else -(x + 1)

theorem odd_function (x : ℝ) (h : x ≠ 0) : f (-x) = -f x :=
begin
  rw f,
  rw f,
  split_ifs,
  case h_1 : hx_neg hx_pos {
    exact rfl,
  },
  case h_2 : hx_pos hx_pos {
    exact rfl,
  },
  case h_3 : hx_neg' hx_neg {
    exact rfl,
  },
  case h_4 : hx_non_pos hx_non_pos {
    exact rfl,
  },
end

theorem find_range (f : ℝ → ℝ) (h : ∀ x, f (-x) = -f x) :
  (∀ x ∈ (-∞, 0) ∪ (1, 2), f (x - 1) < 0) :=
begin
  sorry
end

end odd_function_find_range_l464_464027


namespace LS_parallel_PQ_l464_464595

noncomputable def L (A P C M : Point) : Point := 
  -- Intersection of AP and CM
  sorry

noncomputable def S (A N C Q : Point) : Point := 
  -- Intersection of AN and CQ
  sorry

theorem LS_parallel_PQ (A P C M N Q : Point) 
  (hL : L A P C M) (hS : S A N C Q) : 
  Parallel (Line (L A P C M)) (Line (S A N C Q)) :=
sorry

end LS_parallel_PQ_l464_464595


namespace P_eq_3Q_l464_464024

noncomputable def M(n : ℕ) : ℕ := nat.choose n 3

def p_i (n : ℕ) (i : ℕ) : ℕ := 
  let M := finset.range (n + 1) in
  let subsets := finset.powersetLen 3 M in
  subsets.toList.get_or_else i {0}.max

def q_i (n : ℕ) (i : ℕ) : ℕ := 
  let M := finset.range (n + 1) in
  let subsets := finset.powersetLen 3 M in
  subsets.toList.get_or_else i {nat.succ 0}.min

def P (n : ℕ) : ℕ :=
  finset.range (M n).sum (λ i, p_i n i)

def Q (n : ℕ) : ℕ :=
  finset.range (M n).sum (λ i, q_i n i)

theorem P_eq_3Q (n : ℕ) (hn : n ≥ 3) : P n = 3 * Q n :=
by sorry

end P_eq_3Q_l464_464024


namespace minimize_area_of_quadrilateral_l464_464503

noncomputable def minimize_quad_area (AB BC CD DA A1 B1 C1 D1 : ℝ) (k : ℝ) : Prop :=
  -- Conditions
  AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0 ∧ k > 0 ∧
  A1 = k * AB ∧ B1 = k * BC ∧ C1 = k * CD ∧ D1 = k * DA →
  -- Conclusion
  k = 1 / 2

-- Statement without proof
theorem minimize_area_of_quadrilateral (AB BC CD DA : ℝ) : ∃ k : ℝ, minimize_quad_area AB BC CD DA (k * AB) (k * BC) (k * CD) (k * DA) k :=
sorry

end minimize_area_of_quadrilateral_l464_464503


namespace exam_exercise_distribution_l464_464710

theorem exam_exercise_distribution
  (students : Fin 16 → Fin 3) :
  ∃ e : Fin 3, 6 ≤ (finset.univ.filter (λ i, students i = e)).card :=
by
  sorry

end exam_exercise_distribution_l464_464710


namespace parallel_line_plane_perp_line_plane_l464_464925

-- Definitions of the vectors
variables {m n : ℝ}

-- If the line is parallel to the plane, then the dot product of direction 
-- vector of the line and normal vector of the plane must be zero.
theorem parallel_line_plane (h : m*(-2) + 1*n + 3*1 = 0) : 2*m - n = 3 :=
by sorry

-- If the line is perpendicular to the plane, then the normal vector and 
-- direction vector of the line are scalar multiples of each other,
-- giving the condition mn + 2 = 0.
theorem perp_line_plane (h : ∃ λ : ℝ, (-2) = λ*m ∧ n = λ*1 ∧ 1 = λ*3) : m*n + 2 = 0 :=
by sorry

end parallel_line_plane_perp_line_plane_l464_464925


namespace arrangement_count_l464_464748

def number_of_arrangements (n : ℕ) : ℕ :=
  if n = 6 then 5 * (Nat.factorial 5) else 0

theorem arrangement_count : number_of_arrangements 6 = 600 :=
by
  sorry

end arrangement_count_l464_464748


namespace find_probability_l464_464130

open Probability

noncomputable def X : NormalDist := NormalDist.mk 3 1

theorem find_probability : 
  P(0 < fun x : ℝ => X.density x ∘ 1) = 0.0215 :=
by 
  sorry

end find_probability_l464_464130


namespace jeans_price_increase_l464_464351

theorem jeans_price_increase 
  (C : ℝ) 
  (R : ℝ) 
  (F : ℝ) 
  (H1 : R = 1.40 * C)
  (H2 : F = 1.82 * C) 
  : (F - C) / C * 100 = 82 := 
sorry

end jeans_price_increase_l464_464351


namespace total_time_spent_l464_464577

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2
def total_time : ℕ := outlining_time + writing_time + practicing_time

theorem total_time_spent : total_time = 117 := by
  sorry

end total_time_spent_l464_464577


namespace polynomial_nonzero_terms_count_l464_464875

theorem polynomial_nonzero_terms_count :
  let p := (x + 5) * (3 * x^2 + 2 * x + 4) - 4 * (x^3 - x^2 + 3 * x)
  (p.coeff 3 ≠ 0 ∧ p.coeff 2 ≠ 0 ∧ p.coeff 1 ≠ 0 ∧ p.coeff 0 ≠ 0) ∧
  (p.degree = 3 ∨ p.degree = 2 ∨ p.degree = 1 ∨ p.degree = 0) →
  ∃ t : ℕ, t = 4 := 
sorry

end polynomial_nonzero_terms_count_l464_464875


namespace polar_coordinates_of_point_l464_464413

theorem polar_coordinates_of_point {x y : ℝ} (hx : x = -3) (hy : y = 1) :
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.pi - Real.arctan (y / abs x)
  r = Real.sqrt 10 ∧ θ = Real.pi - Real.arctan (1 / 3) := 
by
  rw [hx, hy]
  sorry

end polar_coordinates_of_point_l464_464413


namespace set_intersection_l464_464541

noncomputable def A : set ℝ := {x | log 2 x < 2}
def B : set ℝ := {x | x ≥ 2}
def intersection : set ℝ := {x | 2 ≤ x ∧ x < 4}

theorem set_intersection (x : ℝ) : x ∈ A ∩ B ↔ x ∈ intersection := 
by sorry

end set_intersection_l464_464541


namespace intersect_at_single_point_l464_464948

theorem intersect_at_single_point
  (P1 P2 : Fin 5 → EuclideanSpace ℝ (Fin 2))
  (common_vertex : ∃ v, P1 0 = v ∧ P2 0 = v)
  (regular_pentagon : ∀ (i : Fin 5), dist (P1 i) (P1 (i + 1) % 5) = dist (P2 i) (P2 (i + 1) % 5)) :
  ∃ C : EuclideanSpace ℝ (Fin 2), ∀ i : Fin 5, ∃ t : ℝ, (P1 i + t • (P2 i - P1 i) = C) := 
sorry

end intersect_at_single_point_l464_464948


namespace solve_problem_l464_464050

noncomputable def ellipse_equation {a b c : ℝ} (h1 : a = 2 * b) (h2 : a * b = 2) (h3 : c = (a^2 - b^2) ^ (1/2)) (h4 : (c/a) = (sqrt 3 / 2)) :
    Prop :=
((a = 2) ∧ (b = 1) ∧ (c = sqrt 3) ∧ (∀ x y : ℝ, (x^2 / 4 + y^2 = 1)))

noncomputable def slope_of_line_l {a : ℝ} (h1 : a = 2) (h2 : |(4 * sqrt 2)/5| = (sqrt (1 + k^2))/(1 + 4 * k^2)) :
    Prop :=
    (k = 1 ∨ k = -1)

theorem solve_problem :
    (∀ a b c : ℝ, (a = 2 * b) ∧ (a * b = 2) ∧ (c = (a^2 - b^2) ^ (1/2)) ∧ ((c/a) = (sqrt 3 / 2)) → ellipse_equation h₁ h₂ h₃ h₄) ∧ 
    ((∀ a : ℝ, a = 2 → |(4 * sqrt 2)/5| = (sqrt (1 + k^2))/(1 + 4 * k^2)) → slope_of_line_l a) :=
sorry

end solve_problem_l464_464050


namespace LS_parallel_PQ_l464_464597

noncomputable def L (A P C M : Point) : Point := 
  -- Intersection of AP and CM
  sorry

noncomputable def S (A N C Q : Point) : Point := 
  -- Intersection of AN and CQ
  sorry

theorem LS_parallel_PQ (A P C M N Q : Point) 
  (hL : L A P C M) (hS : S A N C Q) : 
  Parallel (Line (L A P C M)) (Line (S A N C Q)) :=
sorry

end LS_parallel_PQ_l464_464597


namespace find_closest_point_l464_464881

noncomputable def closest_point_to_line (line_point : ℝ × ℝ) (direction : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
let (x1, y1) := line_point in
let (dx, dy) := direction in
let (x2, y2) := point in
let t := ((x2 - x1) * dx + (y2 - y1) * dy) / (dx * dx + dy * dy) in
(x1 + t * dx, y1 + t * dy)
  
statement : Prop :=
  let line_point := (0 : ℝ, 6 : ℝ) in
  let direction := (1 : ℝ, -2 : ℝ) in
  let point := (1 : ℝ, -3 : ℝ) in
  closest_point_to_line line_point direction point = (19 / 5, 2 / 5)
  
theorem find_closest_point :
  statement := by
  sorry

end find_closest_point_l464_464881


namespace cost_to_paint_cube_l464_464965

def paint_cost (cost_per_quart : ℝ) (coverage_per_quart : ℝ) (edge_length : ℝ) : ℝ :=
  let area_per_face := edge_length * edge_length
  let total_surface_area := 6 * area_per_face
  let quarts_needed := total_surface_area / coverage_per_quart
  quarts_needed * cost_per_quart

theorem cost_to_paint_cube :
  paint_cost 3.20 10 10 = 192 := by
  sorry

end cost_to_paint_cube_l464_464965


namespace minimize_total_cost_l464_464255

-- Define the constants and fuel consumption function.
def distance : ℝ := 1300
def gas_price : ℝ := 7
def driver_wage : ℝ := 30

def fuel_consumption (x : ℝ) : ℝ := 2 + (x ^ 2) / 360

-- Define the total cost function.
def total_cost (x : ℝ) : ℝ := (distance / x) * gas_price * fuel_consumption(x) + (driver_wage * distance / x)

-- Specify the constraints on the speed.
def is_within_speed_limit (x : ℝ) : Prop := 40 ≤ x ∧ x ≤ 100

-- Ensure the final statement proves the correct expression and minimum cost.
theorem minimize_total_cost :
  is_within_speed_limit 40 →
  (∀ x, is_within_speed_limit x → total_cost(x) ≥ total_cost(40)) ∧
  abs (total_cost(40) - 2441.11) < 0.01 :=
by
  sorry

end minimize_total_cost_l464_464255


namespace three_digit_numbers_divisible_by_17_l464_464096

theorem three_digit_numbers_divisible_by_17 :
  ∃ n : ℕ, n = 53 ∧ ∀ k : ℕ, 100 ≤ 17 * k ∧ 17 * k ≤ 999 ↔ k ∈ finset.Icc 6 58 :=
by
  sorry

end three_digit_numbers_divisible_by_17_l464_464096


namespace candy_seller_initial_candies_l464_464423

-- Given conditions
def num_clowns : ℕ := 4
def num_children : ℕ := 30
def candies_per_person : ℕ := 20
def candies_left : ℕ := 20

-- Question: What was the initial number of candies?
def total_people : ℕ := num_clowns + num_children
def total_candies_given_out : ℕ := total_people * candies_per_person
def initial_candies : ℕ := total_candies_given_out + candies_left

theorem candy_seller_initial_candies : initial_candies = 700 :=
by
  sorry

end candy_seller_initial_candies_l464_464423


namespace remainder_of_sum_l464_464331

theorem remainder_of_sum (a b c : ℕ) (h1 : a % 15 = 8) (h2 : b % 15 = 12) (h3 : c % 15 = 13) : (a + b + c) % 15 = 3 := 
by
  sorry

end remainder_of_sum_l464_464331


namespace T_shaped_figure_perimeter_l464_464722

-- Define the T-shaped figure's properties
def T_shaped_figure (a : ℕ) : Prop :=
  a = 1 ∧ 
  let top_row := 3 * a in
  let bottom_row := 2 * a in
  let vertical_segments := 2 * a + 2 * a + 2 * a + 2 * a in
  let horizontal_segments := 4 * a + 2 * a in
  2 * (2 * vertical_segments + 2 * horizontal_segments) - 4 * a = 14

-- Prove the perimeter of the T-shaped figure is 14 units
theorem T_shaped_figure_perimeter : T_shaped_figure 1 :=
by
  simp [T_shaped_figure]
  sorry

end T_shaped_figure_perimeter_l464_464722


namespace parabola_tangent_to_line_l464_464538

theorem parabola_tangent_to_line (a : ℝ) : 
  (∀ x : ℝ, y = ax^2 + 6) ∧ (∀ x : ℝ, y = 2x + 4) ∧ (discriminant (a*x^2 - 2*x + 2) = 0) → a = 1/2 := 
by 
  sorry

end parabola_tangent_to_line_l464_464538


namespace induction_step_term_added_l464_464761

open Nat

theorem induction_step_term_added (k : ℕ) :
  (∑ i in range (k+1), 1 / (i + (k + 1))) - (∑ i in range k, 1 / (i + k)) = (1 / (2 * k + 1) - 1 / (2 * (k + 1))) :=
by
  sorry

end induction_step_term_added_l464_464761


namespace part_a_part_b_l464_464378

-- Given conditions for both parts of the problem
def choices (n : ℕ) : list (fin n) := list.fin_range n

-- Part (a)
-- The probability that the bags end up in reverse order in the shed
theorem part_a (n : ℕ) (hn : n = 4) : 
  probability_reverse_order hn = 1 / 8 :=
sorry

-- Part (b)
-- The probability that the second-from-bottom bag in the truck ends up as the bottom bag in the shed
theorem part_b (n : ℕ) (hn : n = 4) : 
  probability_second_from_bottom hn = 1 / 8 :=
sorry

-- Definitions of probability calculations used in the conditions
noncomputable def probability_reverse_order (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

noncomputable def probability_second_from_bottom (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

end part_a_part_b_l464_464378


namespace LS_parallel_PQ_l464_464600

noncomputable def L (A P C M : Point) : Point := 
  -- Intersection of AP and CM
  sorry

noncomputable def S (A N C Q : Point) : Point := 
  -- Intersection of AN and CQ
  sorry

theorem LS_parallel_PQ (A P C M N Q : Point) 
  (hL : L A P C M) (hS : S A N C Q) : 
  Parallel (Line (L A P C M)) (Line (S A N C Q)) :=
sorry

end LS_parallel_PQ_l464_464600


namespace simplify_trig_expression_l464_464248

theorem simplify_trig_expression (x y : ℝ) :
  sin x * sin x + sin (x + 2 * y) * sin (x + 2 * y) - 2 * sin x * sin (2 * y) * cos (x + 2 * y) = 
  2 * sin x * sin x - sin x * sin x * cos y * cos y := 
sorry

end simplify_trig_expression_l464_464248


namespace probability_of_at_least_two_same_rank_approx_l464_464707

noncomputable def probability_at_least_two_same_rank (cards_drawn : ℕ) (total_cards : ℕ) : ℝ :=
  let ranks := 13
  let different_ranks_comb := Nat.choose ranks cards_drawn
  let rank_suit_combinations := different_ranks_comb * (4 ^ cards_drawn)
  let total_combinations := Nat.choose total_cards cards_drawn
  let p_complement := rank_suit_combinations / total_combinations
  1 - p_complement

theorem probability_of_at_least_two_same_rank_approx (h : 5 ≤ 52) : 
  abs (probability_at_least_two_same_rank 5 52 - 0.49) < 0.01 := 
by
  sorry

end probability_of_at_least_two_same_rank_approx_l464_464707


namespace range_of_a_for_monotonically_increasing_function_l464_464128

theorem range_of_a_for_monotonically_increasing_function :
  (∀ x y : ℝ, x < y → f x ≤ f y) → (a ∈ Icc (-Real.sqrt 3) (Real.sqrt 3)) := by
-- Define the function f(x) = x^3 + ax^2 + x - 7
let f : ℝ → ℝ := λ x, x^3 + a * x^2 + x - 7
-- Define the first derivative of f(x)
let f_prime : ℝ → ℝ := λ x, 3 * x^2 + 2 * a * x + 1
-- Monotonic increasing condition: f'(x) ≥ 0 for all x in ℝ
have h : ∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≥ 0 := sorry
-- Solve the inequality to find the range of a
sorry

end range_of_a_for_monotonically_increasing_function_l464_464128


namespace three_digit_numbers_divisible_by_17_l464_464101

theorem three_digit_numbers_divisible_by_17 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_17 := {n : ℕ | n % 17 = 0}
  ∃! count : ℕ, count = fintype.card {n : ℕ // n ∈ three_digit_numbers ∧ n ∈ divisible_by_17} ∧ count = 53 :=
begin
  sorry
end

end three_digit_numbers_divisible_by_17_l464_464101


namespace completing_square_solution_l464_464719

theorem completing_square_solution (x : ℝ) :
  2 * x^2 + 4 * x - 3 = 0 →
  (x + 1)^2 = 5 / 2 :=
by
  sorry

end completing_square_solution_l464_464719


namespace division_by_fraction_l464_464844

theorem division_by_fraction (a b : ℝ) (h : b ≠ 0) : (a / (1 / b)) = a * b :=
  sorry

example : (12 / (1 / 12)) = 144 :=
  by
  apply division_by_fraction
  linarith

end division_by_fraction_l464_464844


namespace axis_of_symmetry_find_phi_l464_464074

noncomputable def interval := Set.Ioo (Real.pi/6) (Real.pi/2)

theorem axis_of_symmetry (ω : ℤ) (h1 : 0 < ω) (φ : ℝ) 
  (h2 : |φ| < Real.pi / 2) (h3 : (sin (ω * (Real.pi / 2) + φ)) = 
  (sin (ω * (2 * Real.pi / 3) + φ))) : 
  ∃ x : ℝ, x = 7 * Real.pi / 12 :=
sorry

theorem find_phi (ω : ℤ) (h1 : 0 < ω) 
  (h2 : f = λ x : ℝ, sin(ω * x + phi)) 
  (h3 : ((λ x, sin(ω * x + φ)) 
  (Real.pi / 6)) = sqrt 3 / 2) 
  (h4 : |φ| < Real.pi / 2) 
  (h5 : f (Real.pi / 2) = f (2 * Real.pi / 3)) 
  (h6 : ω = 2): 
  φ = Real.pi / 3 :=
sorry

end axis_of_symmetry_find_phi_l464_464074


namespace minimum_shapes_to_form_square_proof_l464_464326

def shape_area : ℕ := 3

def minimum_shapes_to_form_square (n : ℕ) : Prop :=
  ∃ (side_length : ℕ), side_length * side_length = n * shape_area

theorem minimum_shapes_to_form_square_proof :
  minimum_shapes_to_form_square 12 :=
begin
  -- Proof goes here
  sorry
end

end minimum_shapes_to_form_square_proof_l464_464326


namespace simplify_complex_expression_l464_464247

theorem simplify_complex_expression : 
  (7 * (2 - 2 * complex.I) + 4 * complex.I * (7 - 3 * complex.I)) = (26 + 14 * complex.I) := by
  sorry

end simplify_complex_expression_l464_464247


namespace three_digit_count_divisible_by_five_l464_464519

theorem three_digit_count_divisible_by_five : 
  let valid_digits := {x : ℕ | x > 5 ∧ x < 10} in
  let hundreds_digit_choices := valid_digits in
  let tens_digit_choices := valid_digits in
  let ones_digit_choice := {5} in
  (∃ (h t u : ℕ), h ∈ hundreds_digit_choices ∧ t ∈ tens_digit_choices ∧ u ∈ ones_digit_choice ∧ 
    100 * h + 10 * t + u > 99 ∧ 100 * h + 10 * t + u < 1000) ∧
  (100 * h + 10 * t + u) % 5 = 0 → 
  (Set.card hundreds_digit_choices) * 
  (Set.card tens_digit_choices) * 
  (Set.card ones_digit_choice) = 16 :=
by
  sorry

end three_digit_count_divisible_by_five_l464_464519


namespace common_difference_l464_464141

-- Define the arithmetic sequence with general term
def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem common_difference (a₁ a₅ a₄ d : ℕ) 
  (h₁ : a₁ + a₅ = 10)
  (h₂ : a₄ = 7)
  (h₅ : a₅ = a₁ + 4 * d)
  (h₄ : a₄ = a₁ + 3 * d) :
  d = 2 :=
by
  sorry

end common_difference_l464_464141


namespace angle_BIM_right_l464_464158

theorem angle_BIM_right (A B C I M N : Point)
  (hI_incenter : is_incenter I A B C)
  (hM_midpoint : is_midpoint M B C)
  (hN_midpoint : is_midpoint N A C)
  (hAIN_right : angle A I N = 90) :
  angle B I M = 90 :=
sorry

end angle_BIM_right_l464_464158


namespace part_one_part_two_l464_464039

-- Part 1: proving the area of the triangle
theorem part_one (f : ℝ → ℝ) (f_eq : ∀ x, f x = Real.sin x) :
  let point := (Real.pi / 3, Real.sqrt 3 / 2)
  let tangent_line := λ x, (Real.sqrt 3 / 2) - (1 / 2) * (x - Real.pi / 3)
  let x_intercept := Real.pi / 3 - Real.sqrt 3
  let y_intercept := Real.sqrt 3 / 2 - Real.pi / 6
  (1 / 2 * abs (x_intercept * y_intercept)) = (1 / 4 * (Real.sqrt 3 - Real.pi / 3)^2) :=
by
  sorry

-- Part 2: proving the value of m
theorem part_two (f g h : ℝ → ℝ) (f_eq : ∀ x, f x = Real.sin x)
  (g_eq : ∀ x m, g x = x - m) (h_eq : ∀ x m, h x = f x * g x)
  (m : ℝ) (x1 x2 : ℝ) (cond : h x1 + h x2 = 0) :
  m = Real.pi / 2 :=
by
  sorry

end part_one_part_two_l464_464039


namespace min_distance_PQ_l464_464555

-- Define the line l in Cartesian coordinates
def line_l (x y : ℝ) : Prop :=
  4 * x - y - 25 = 0

-- Define the curve W using its general equation
def curve_W (x y : ℝ) : Prop :=
  y = (1 / 4) * x^2 - 1

-- Define the distance formula from a point to the line
def distance_to_line (x y : ℝ) : ℝ :=
  abs (4 * x - y - 25) / real.sqrt(4^2 + (-1)^2)

-- Define the parametric point Q on curve W
def parametric_Q (t : ℝ) : ℝ × ℝ :=
  (2 * t, t^2 - 1)

-- Define the minimum distance of |PQ|
def min_distance : ℝ :=
  (8 * real.sqrt 17) / 17

-- Main theorem statement
theorem min_distance_PQ :
  ∀ (t : ℝ), distance_to_line (2 * t) (t^2 - 1) ≥ min_distance :=
by
  sorry

end min_distance_PQ_l464_464555


namespace polynomial_is_x_l464_464864

-- Given conditions definitions
def polynomial_condition (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, p(x^2 + 1) = (p(x))^2 + 1

def p_zero_condition (p : Polynomial ℝ) : Prop :=
  p 0 = 0

-- The theorem to prove that p(x) = x given the conditions
theorem polynomial_is_x (p : Polynomial ℝ) (h1 : polynomial_condition p) (h2 : p_zero_condition p) : p = Polynomial.X :=
sorry

end polynomial_is_x_l464_464864


namespace same_school_probability_l464_464245

/-- 
Given:
- School A has 2 male and 1 female teachers.
- School B has 1 male and 2 female teachers.
- There are a total of 6 teachers (3 from each school).
  
Prove that the probability of selecting 2 teachers from the 6 such that both teachers are from the same school is 2/5.
-/
theorem same_school_probability :
  let teachers := ({2, 1} ++ {1, 2} : List ℕ) in
  let total_pairs := (Nat.choose teachers.length 2) in
  let same_school_pairs := List.length ([{a, b, c}, {A, B, C}].bind (λ l, List.subsets l 2)) in
  (same_school_pairs / total_pairs : ℚ) = 2 / 5 :=
by {
  sorry
}

end same_school_probability_l464_464245


namespace prove_b_is_neg_two_l464_464540

-- Define the conditions
variables (b : ℝ)

-- Hypothesis: The real and imaginary parts of the complex number (2 - b * I) * I are opposites
def complex_opposite_parts (b : ℝ) : Prop :=
  b = -2

-- The theorem statement
theorem prove_b_is_neg_two : complex_opposite_parts b :=
sorry

end prove_b_is_neg_two_l464_464540


namespace parallel_ls_pq_l464_464613

variables {A P C M N Q L S : Type*}
variables [affine_space ℝ (A P C M N Q)]

/- Definitions of the intersection points -/
def is_intersection_point (L : Type*) (AP : set (A P))
  (CM : set (C M)) : Prop := L ∈ AP ∧ L ∈ CM

def is_intersection_point' (S : Type*) (AN : set (A N))
  (CQ : set (C Q)) : Prop := S ∈ AN ∧ S ∈ CQ

/- Main Theorem -/
theorem parallel_ls_pq (AP CM AN CQ : set (A P))
  (PQ : set (P Q)) (L S : Type*)
  (hL : is_intersection_point L AP CM)
  (hS : is_intersection_point' S AN CQ) : L S ∥ P Q :=
sorry

end parallel_ls_pq_l464_464613


namespace coefficient_of_x3_in_expansion_l464_464264

def binomial_coeff (n k : ℕ) : ℤ := nat.choose n k

noncomputable def binomial_expansion : ℕ → ℤ := sorry  -- This is expanded in terms of binomial coefficients

theorem coefficient_of_x3_in_expansion :
  let x : ℤ := 3,
      a : ℤ := x^2 - 4,
      b : ℤ := x + (1 / x : ℤ)^9,
      target_coefficient : ℤ := -210 in
  (a * b).coeff x = target_coefficient :=
sorry

end coefficient_of_x3_in_expansion_l464_464264


namespace part_one_part_two_l464_464482

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) := 
  (A + B + C = π ∧
  b + c = 7 ∧
  b > c ∧
  ∆ABC_area = 3 * sqrt 3 ∧
  cos ((B + C) / 2) = 1 - cos A)

theorem part_one (a b c A B C : ℝ) (h : triangle_problem a b c A B C) : 
  A = π / 3 :=
sorry

theorem part_two (a b c A B C : ℝ) (h : triangle_problem a b c A B C) : 
  a = sqrt 13 :=
sorry

end part_one_part_two_l464_464482


namespace parallel_LS_pQ_l464_464622

universe u

variables {P Q A C M N L S T : Type u} [incidence_geometry P Q A C M N L S T] -- Customize as per specific needs in Lean

-- Assume the required points and their intersections
def intersection_L (A P C M : Type u) [incidence_geometry A P C M] : Type u := sorry
def intersection_S (A N C Q : Type u) [incidence_geometry A N C Q] : Type u := sorry

-- Defining the variables as intersections
variables (L : intersection_L A P C M)
variables (S : intersection_S A N C Q)

theorem parallel_LS_pQ (hL : intersection_L A P C M) (hS : intersection_S A N C Q) :
  parallel L S P Q :=
sorry

end parallel_LS_pQ_l464_464622


namespace compute_f_in_terms_of_fx_l464_464182

def f (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

theorem compute_f_in_terms_of_fx (x : ℝ) (h : -1 < x ∧ x < 1) :
  f ((2 * x + x ^ 2) / (1 + 2 * x)) = 2 * f x :=
sorry

end compute_f_in_terms_of_fx_l464_464182


namespace problem_180_180_minus_12_l464_464322

namespace MathProof

theorem problem_180_180_minus_12 :
  180 * (180 - 12) - (180 * 180 - 12) = -2148 := 
by
  -- Placeholders for computation steps
  sorry

end MathProof

end problem_180_180_minus_12_l464_464322


namespace LS_parallel_PQ_l464_464636

universe u
variables {Point : Type u} [Geometry Point]

-- Defining points A, P, C, M, N, Q as elements of type Point
variables (A P C M N Q : Point)

-- Let L be the intersection point of the lines AP and CM
def L := Geometry.intersection (Geometry.line_through A P) (Geometry.line_through C M)

-- Let S be the intersection point of the lines AN and CQ
def S := Geometry.intersection (Geometry.line_through A N) (Geometry.line_through C Q)

-- Proof: LS is parallel to PQ
theorem LS_parallel_PQ : Geometry.parallel (Geometry.line_through L S) (Geometry.line_through P Q) :=
sorry

end LS_parallel_PQ_l464_464636


namespace maximum_elements_in_S_l464_464962

-- Definition of "short" rational number
def is_short (x : ℚ) : Prop :=
  ∃ (a b : ℕ), (2^a) * (5^b) * x.denom ≤ x.num

-- Definition of "m-splendid" number
def is_m_splendid (m t : ℕ) : Prop :=
  ∃ (c : ℕ), (1 ≤ c ∧ c ≤ 2017 ∧ is_short (10 ^ t - 1 / c / m)) ∧
    (∀ k, 1 ≤ k ∧ k < t → ¬ is_short (10 ^ k - 1 / c / m))

-- Definition of the set S(m)
def S (m : ℕ) : set ℕ :=
  { t | is_m_splendid m t }

-- The main theorem statement
theorem maximum_elements_in_S (m : ℕ) : set.card (S m) ≤ 807 := sorry

end maximum_elements_in_S_l464_464962


namespace find_value_of_M_l464_464287

variable {C y M A : ℕ}

theorem find_value_of_M (h1 : C + y + 2 * M + A = 11)
                        (h2 : C ≠ y)
                        (h3 : C ≠ M)
                        (h4 : C ≠ A)
                        (h5 : y ≠ M)
                        (h6 : y ≠ A)
                        (h7 : M ≠ A)
                        (h8 : 0 < C)
                        (h9 : 0 < y)
                        (h10 : 0 < M)
                        (h11 : 0 < A) : M = 1 :=
by
  sorry

end find_value_of_M_l464_464287


namespace part_a_part_b_l464_464379

-- Given conditions for both parts of the problem
def choices (n : ℕ) : list (fin n) := list.fin_range n

-- Part (a)
-- The probability that the bags end up in reverse order in the shed
theorem part_a (n : ℕ) (hn : n = 4) : 
  probability_reverse_order hn = 1 / 8 :=
sorry

-- Part (b)
-- The probability that the second-from-bottom bag in the truck ends up as the bottom bag in the shed
theorem part_b (n : ℕ) (hn : n = 4) : 
  probability_second_from_bottom hn = 1 / 8 :=
sorry

-- Definitions of probability calculations used in the conditions
noncomputable def probability_reverse_order (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

noncomputable def probability_second_from_bottom (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

end part_a_part_b_l464_464379


namespace modulus_complex_expression_l464_464956

theorem modulus_complex_expression (a b : ℝ) (h : (1 + 2*a*complex.I) * complex.I = 1 - b*complex.I) :
  complex.abs (a + b*complex.I) = real.sqrt 5 / 2 :=
sorry

end modulus_complex_expression_l464_464956


namespace find_a_value_l464_464970

noncomputable def perpendicular_line (a : ℝ) : Prop :=
  let slope_of_line := (a - 0) / (3 - (-2)) in
  slope_of_line * (1/2) = -1

theorem find_a_value :
  ∃ a : ℝ, perpendicular_line a ∧ a = -10 :=
by
  use -10
  unfold perpendicular_line
  sorry

end find_a_value_l464_464970


namespace range_of_f_l464_464493

-- Define the function f
def f (x : ℕ) : ℤ := 2 * (x : ℤ) - 3

-- Define the domain
def domain : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 1, 3, 5, 7}

-- Prove the range of f given the domain
theorem range_of_f : domain.image f = expected_range :=
  sorry

end range_of_f_l464_464493


namespace henrys_distance_from_start_l464_464089

noncomputable def meters_to_feet (x : ℝ) : ℝ := x * 3.281
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem henrys_distance_from_start :
  let west_walk_feet := meters_to_feet 15
  let north_walk_feet := 60
  let east_walk_feet := 156
  let south_walk_meter_backwards := 30
  let south_walk_feet_backwards := 12
  let total_south_feet := meters_to_feet south_walk_meter_backwards + south_walk_feet_backwards
  let net_south_feet := total_south_feet - north_walk_feet
  let net_east_feet := east_walk_feet - west_walk_feet
  distance 0 0 net_east_feet (-net_south_feet) = 118 := 
by
  sorry

end henrys_distance_from_start_l464_464089


namespace isosceles_triangulation_l464_464794

variables {A B C D M : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace M]
variables (dABC : convex_hull ℝ {A, B, C, D}) (M_inside : M ∈ convex_hull ℝ {A, B, C, D})

theorem isosceles_triangulation (h1 : isosceles_triangle A B M) 
  (h2 : isosceles_triangle B C M) 
  (h3 : isosceles_triangle C D M) 
  (h4 : isosceles_triangle D A M) : 
  ∃ a b c d : ℝ, (a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d) :=
by sorry

end isosceles_triangulation_l464_464794


namespace sum_of_common_ratios_l464_464676

variables {k p r : ℝ}
variables {a_2 a_3 b_2 b_3 : ℝ}

def geometric_seq1 (k p : ℝ) := a_2 = k * p ∧ a_3 = k * p^2
def geometric_seq2 (k r : ℝ) := b_2 = k * r ∧ b_3 = k * r^2

theorem sum_of_common_ratios (h1 : geometric_seq1 k p) (h2 : geometric_seq2 k r)
  (h3 : p ≠ r) (h4 : a_3 - b_3 = 4 * (a_2 - b_2)) : p + r = 4 :=
by sorry

end sum_of_common_ratios_l464_464676


namespace ls_parallel_pq_l464_464601

-- Declare the points and lines
variables {A P C M N Q L S : Type*}

-- Assume given conditions as definitions
def intersection (l1 l2 : set (Type*)) : Type* := sorry -- Define intersection point of l1 and l2 (details omitted for simplicity)

-- Define L as intersection of lines AP and CM
def L : Type* := intersection (set_of (λ x, ∃ L, ∃ P, ∃ C, ∃ M, L = x ∧ x ∈ AP ∧ x ∈ CM)) (λ x, x ∈ AP ∧ x ∈ CM)

-- Define S as intersection of lines AN and CQ
def S : Type* := intersection (set_of (λ x, ∃ S, ∃ N, ∃ C, ∃ Q, S = x ∧ x ∈ AN ∧ x ∈ CQ)) (λ x, x ∈ AN ∧ x ∈ CQ)

-- Statement to prove LS is parallel to PQ
theorem ls_parallel_pq (AP CM AN CQ PQ : set (Type*)) (L : Type*) (S : Type*) :
    L = intersection (AP) (CM) ∧ S = intersection (AN) (CQ) → is_parallel LS PQ :=
by
  sorry -- Proof omitted

end ls_parallel_pq_l464_464601


namespace rational_sum_square_implies_integers_l464_464240

theorem rational_sum_square_implies_integers
  (a b : ℚ)
  (h1 : (a + b) ∈ ℤ)
  (h2 : (a^2 + b^2) ∈ ℤ) :
  a ∈ ℤ ∧ b ∈ ℤ :=
sorry

end rational_sum_square_implies_integers_l464_464240


namespace largest_integer_solution_l464_464765

theorem largest_integer_solution (x : ℤ) (h : (x : ℚ) / 4 + 3 / 7 < 4 / 3) : x ≤ 3 :=
by
  sorry

example : ∃ x : ℤ, (x : ℚ) / 4 + 3 / 7 < 4 / 3 ∧ x = 3 :=
by
  use 3
  constructor
  . norm_cast
    linarith
  . rfl

end largest_integer_solution_l464_464765


namespace probability_point_A_on_hyperbola_l464_464222

-- Define the set of numbers
def numbers : List ℕ := [1, 2, 3]

-- Define the coordinates of point A taken from the set, where both numbers are different
def point_A_pairs : List (ℕ × ℕ) :=
  [ (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) ]

-- Define the function indicating if a point (m, n) lies on the hyperbola y = 6/x
def lies_on_hyperbola (m n : ℕ) : Prop :=
  n = 6 / m

-- Calculate the probability of a point lying on the hyperbola
theorem probability_point_A_on_hyperbola : 
  (point_A_pairs.countp (λ (p : ℕ × ℕ), lies_on_hyperbola p.1 p.2)).toRat / (point_A_pairs.length).toRat = 1 / 3 := 
sorry

end probability_point_A_on_hyperbola_l464_464222


namespace parallel_LS_pQ_l464_464619

universe u

variables {P Q A C M N L S T : Type u} [incidence_geometry P Q A C M N L S T] -- Customize as per specific needs in Lean

-- Assume the required points and their intersections
def intersection_L (A P C M : Type u) [incidence_geometry A P C M] : Type u := sorry
def intersection_S (A N C Q : Type u) [incidence_geometry A N C Q] : Type u := sorry

-- Defining the variables as intersections
variables (L : intersection_L A P C M)
variables (S : intersection_S A N C Q)

theorem parallel_LS_pQ (hL : intersection_L A P C M) (hS : intersection_S A N C Q) :
  parallel L S P Q :=
sorry

end parallel_LS_pQ_l464_464619


namespace least_n_factorial_multiple_840_exists_least_n_factorial_multiple_840_l464_464779

theorem least_n_factorial_multiple_840 (n : ℕ) (h_pos : 0 < n) (h_mult : 840 ∣ n!) : n ≥ 8 :=
by sorry

theorem exists_least_n_factorial_multiple_840 : ∃ n : ℕ, 0 < n ∧ 840 ∣ n! ∧ n = 8 :=
by {
  use 8,
  split,
  { exact nat.succ_pos 7, },
  split,
  { norm_num, },
  { rfl, }
}

end least_n_factorial_multiple_840_exists_least_n_factorial_multiple_840_l464_464779


namespace constant_term_of_binomial_expansion_is_60_l464_464974

theorem constant_term_of_binomial_expansion_is_60 :
  ∀ (n : ℕ), 2^n = 64 → ∃ (c : ℕ), c = 60 ∧ (constant_term (λ x, (2*x^2 - x⁻¹)^n) = c) := 
by sorry

end constant_term_of_binomial_expansion_is_60_l464_464974


namespace point_on_hyperbola_probability_l464_464225

theorem point_on_hyperbola_probability :
  let s := ({1, 2, 3} : Finset ℕ) in
  let p := ∑ x in s.sigma (λ x, s.filter (λ y, y ≠ x)),
             if (∃ m n, x = (m, n) ∧ n = (6 / m)) then 1 else 0 in
  p / (s.card * (s.card - 1)) = (1 / 3) :=
by
  -- Conditions and setup
  let s := ({1, 2, 3} : Finset ℕ)
  let t := s.sigma (λ x, s.filter (λ y, y ≠ x))
  let p := t.filter (λ (xy : ℕ × ℕ), xy.snd = 6 / xy.fst)
  have h_total : t.card = 6, by sorry
  have h_count : p.card = 2, by sorry

  -- Calculate probability
  calc
    ↑(p.card) / ↑(t.card) = 2 / 6 : by sorry
    ... = 1 / 3 : by norm_num

end point_on_hyperbola_probability_l464_464225


namespace superinvariant_sets_l464_464320

def is_stretching (A : ℝ → ℝ) (x0 a : ℝ) : Prop :=
  ∀ x, A(x) = x0 + a * (x - x0)

def is_translation (B : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, B(x) = x + b

def superinvariant (S : Set ℝ) : Prop :=
  ∀ (x0 a b x t : ℝ),
    a > 0 →
    (∀ x, x ∈ S → (∃ y ∈ S, x0 + a * (x - x0) = y + b) ∧
           (∃ u ∈ S, t + b = x0 + a * (u - x0)))

theorem superinvariant_sets (S : Set ℝ) :
  superinvariant S ↔ 
  (∃ Γ : ℝ, S = ∅ ∨ S = {Γ} ∨ S = Iio Γ ∨ S = Iic Γ ∨ S = Ioi Γ ∨ S = Ici Γ ∨ S = (Iio Γ ∪ Ioi Γ) ∨ S = univ) :=
sorry

end superinvariant_sets_l464_464320


namespace equation_of_line_l464_464470

-- Definition of the point P(3, -2)
def P := (3 : ℝ, -2 : ℝ)

-- Definition of a line passing through a point and having intercepts of opposite sign
def line_passing_through_point_with_opposite_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  l 3 (-2) ∧ ∃ a : ℝ, (a ≠ 0) ∧ (l a 0 ∧ l 0 (-a))

-- Statement that expresses the conclusion
theorem equation_of_line (l : ℝ → ℝ → Prop) :
  line_passing_through_point_with_opposite_intercepts l →
  (∀ x y : ℝ, l x y ↔ 2 * x + 3 * y = 0) ∨ (∀ x y : ℝ, l x y ↔ x - y - 5 = 0) :=
  sorry

end equation_of_line_l464_464470


namespace bob_has_winning_strategy_l464_464774

theorem bob_has_winning_strategy (grid : Π (r : ℕ) (c : ℕ), ℚ)
  (distinct_rat : ∀ r1 r2 c1 c2, (r1 = r2 ∧ c1 = c2) ∨ (grid r1 c1 ≠ grid r2 c2))
  (largest_black : ∀ r, ∃ c, ∀ c', grid r c' ≤ grid r c)
  (alice_path : (Π r, Σ c, grid r c ≥ grid r c) → Prop)
  (bob_block : ∀ (alice_attempt : Π r, Σ c, grid r c ≥ grid r c), ¬ alice_path alice_attempt) :
  ∃ (bob_strategy : (Π r, Σ c, grid r c ≥ grid r c) → Prop), bob_block bob_strategy
  := sorry

end bob_has_winning_strategy_l464_464774


namespace find_sum_l464_464125

variable (a b : ℚ)

theorem find_sum :
  2 * a + 5 * b = 31 ∧ 4 * a + 3 * b = 35 → a + b = 68 / 7 := by
  sorry

end find_sum_l464_464125


namespace length_of_train_approx_l464_464374

-- Define the given conditions
def speed_km_hr := 120 -- Speed in km/hr
def time_seconds := 15 -- Time in seconds

-- Define the conversion factor from km/hr to m/s
def km_to_m := 1000
def hr_to_s := 3600
def conversion_factor := (km_to_m : ℝ) / (hr_to_s : ℝ)

-- Define the speed in m/s
def speed_m_s := speed_km_hr * conversion_factor

-- Define the length of the train
def length_of_train := speed_m_s * time_seconds

-- The proof problem statement
theorem length_of_train_approx : length_of_train ≈ 500 := by
  sorry

end length_of_train_approx_l464_464374


namespace hyperbola_eccentricity_sqrt2_l464_464905

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_eccentricity_sqrt2 (a : ℝ) (b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hyp : ∀ (x : ℝ), ⟦∃ y : ℝ, y = x ∨ y = -x⟧) :
  hyperbola_eccentricity a b = Real.sqrt 2 :=
by
  have hypAsymptotes : b = a := sorry
  have ecc : hyperbola_eccentricity a b = Real.sqrt 2 := sorry
  exact ecc

end hyperbola_eccentricity_sqrt2_l464_464905


namespace students_taking_either_not_both_l464_464313

theorem students_taking_either_not_both (students_both : ℕ) (students_physics : ℕ) (students_only_chemistry : ℕ) :
  students_both = 12 →
  students_physics = 22 →
  students_only_chemistry = 9 →
  students_physics - students_both + students_only_chemistry = 19 :=
by
  intros h_both h_physics h_chemistry
  rw [h_both, h_physics, h_chemistry]
  repeat { sorry }

end students_taking_either_not_both_l464_464313


namespace find_number_of_pens_l464_464571

-- Definitions based on the conditions in the problem
def total_utensils (P L : ℕ) : Prop := P + L = 108
def pencils_formula (P L : ℕ) : Prop := L = 5 * P + 12

-- The theorem we need to prove
theorem find_number_of_pens (P L : ℕ) (h1 : total_utensils P L) (h2 : pencils_formula P L) : P = 16 :=
by sorry

end find_number_of_pens_l464_464571


namespace maximum_interesting_number_with_distinct_digits_l464_464408

def is_divisible_by_suffixes (A : ℕ) : Prop :=
  ∀ (suffix : ℕ), (suffix ∣ A) → (∃ d : ℕ, suffix = A / (10 ^ d) ∧ d > 0)

def has_distinct_digits (A : ℕ) : Prop :=
  let digits := A.digits 10
  list.nodup digits

def is_interesting (A : ℕ) : Prop :=
  is_divisible_by_suffixes A ∧ has_distinct_digits A

theorem maximum_interesting_number_with_distinct_digits :
  ∃ A : ℕ, is_interesting A ∧ (∀ B : ℕ, is_interesting B → B ≤ A) ∧ A = 3570 :=
begin
  sorry
end

end maximum_interesting_number_with_distinct_digits_l464_464408


namespace intersection_correct_l464_464787

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := { y | ∃ x ∈ A, y = 2 * x - 1 }

def intersection : Set ℕ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_correct : intersection = {1, 3} := by
  sorry

end intersection_correct_l464_464787


namespace tangent_lines_to_circle_are_x_equals_1_and_4x_plus_3y_minus_7_equals_0_l464_464077

noncomputable def circle : Set (ℝ × ℝ) := {p | (p.1)^2 + 4*p.1 + (p.2)^2 - 5 = 0}
def point_M := (1, 1)

theorem tangent_lines_to_circle_are_x_equals_1_and_4x_plus_3y_minus_7_equals_0 :
  (∀ p ∈ circle, p = point_M ∨ 
                  ∃ k : ℝ, (p.2 = p.1*k + b) ∧ ∀ q ∈ circle, (dist p q = 3) → (4*q.1 + 3*q.2 - 7 = 0)) :=
sorry

end tangent_lines_to_circle_are_x_equals_1_and_4x_plus_3y_minus_7_equals_0_l464_464077


namespace probability_on_hyperbola_l464_464239

open Finset

-- Define the function for the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the set of different number pairs from {1, 2, 3}
def pairs : Finset (ℕ × ℕ) := 
  {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}.to_finset

-- Define the set of pairs that lie on the hyperbola
def hyperbola_pairs : Finset (ℕ × ℕ) :=
  pairs.filter (λ mn, on_hyperbola mn.1 mn.2)

-- The theorem to prove the probability
theorem probability_on_hyperbola : 
  (hyperbola_pairs.card : ℝ) / (pairs.card : ℝ) = 1 / 3 :=
by
  -- Placeholder for the proof
  sorry

end probability_on_hyperbola_l464_464239


namespace equilateral_classification_l464_464828

-- Conditions as definitions
def two_angles_sixty (T : Triangle) : Prop :=
  T.angle_a = 60 ∧ T.angle_b = 60

def isosceles_one_angle_sixty (T : Triangle) : Prop :=
  T.is_isosceles ∧ (T.angle_a = 60 ∨ T.angle_b = 60 ∨ T.angle_c = 60)

def all_angles_equal (T : Triangle) : Prop :=
  T.angle_a = T.angle_b ∧ T.angle_b = T.angle_c

def all_sides_equal (T : Triangle) : Prop :=
  T.side_a = T.side_b ∧ T.side_b = T.side_c

-- Define what it means to be an equilateral triangle
def is_equilateral (T : Triangle) : Prop :=
  T.side_a = T.side_b ∧ T.side_b = T.side_c

-- The Lean theorem statement
theorem equilateral_classification (T : Triangle) :
  (two_angles_sixty T ∨ all_angles_equal T ∨ all_sides_equal T) ↔ is_equilateral T := 
by 
s o r r y

end equilateral_classification_l464_464828


namespace us_more_than_canada_l464_464750

/-- Define the total number of supermarkets -/
def total_supermarkets : ℕ := 84

/-- Define the number of supermarkets in the US -/
def us_supermarkets : ℕ := 49

/-- Define the number of supermarkets in Canada -/
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

/-- The proof problem: Prove that there are 14 more supermarkets in the US than in Canada -/
theorem us_more_than_canada : us_supermarkets - canada_supermarkets = 14 := by
  sorry

end us_more_than_canada_l464_464750


namespace Varya_used_discount_l464_464306

variables {P K L : ℝ} -- Prices of pen, pencil, notebook

def Anya_cost := 2 * P + 7 * K + 1 * L
def Varya_cost := 5 * P + 6 * K + 5 * L
def Sasha_cost := 8 * P + 4 * K + 9 * L

theorem Varya_used_discount (h : Anya_cost + Sasha_cost = 2 * Varya_cost): Anya_cost + Sasha_cost < 2 * Varya_cost :=
by
  sorry

end Varya_used_discount_l464_464306


namespace exam_exercise_distribution_l464_464711

theorem exam_exercise_distribution
  (students : Fin 16 → Fin 3) :
  ∃ e : Fin 3, 6 ≤ (finset.univ.filter (λ i, students i = e)).card :=
by
  sorry

end exam_exercise_distribution_l464_464711


namespace angle_B_find_c_l464_464565

variables {α : Type*}

def m (B : ℝ) : ℝ × ℝ := (2 * sin B, 2 - cos (2 * B))

def n (B : ℝ) : ℝ × ℝ := (2 * sin^2 ((B / 2) + (π / 4)), -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem angle_B :
  ∀ (B : ℝ), dot_product (m B) (n B) = 0 → (B = π / 6 ∨ B = 5 * π / 6) :=
begin
  intros B h,
  sorry -- Proof to be provided
end

theorem find_c 
  (a b : ℝ) (A B c : ℝ)  
  (H : a = real.sqrt 3) (H1 : b = 1) (H2 : B = π / 6) :
  b^2 = a^2 + c^2 - 2 * a * c * cos B → (c = 1 ∨ c = 2) :=
begin
  intros h,
  sorry -- Proof to be provided
end

end angle_B_find_c_l464_464565


namespace find_p_value_l464_464914

noncomputable def parabola_finding_p (p : ℝ) (y1 : ℝ) : Prop :=
  (∃ x y1, x = y1^2 / (2 * p) ∧ (y1 = 4) ∧ (((p / 2) + (y1^2 / (2 * p))) = 4))

theorem find_p_value :
  (∃ (p : ℝ), parabola_finding_p p 4 ∧ p > 0) → ∃ (p : ℝ), p = 4 :=
begin
  sorry
end

end find_p_value_l464_464914


namespace range_of_function_l464_464740

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem range_of_function : Set.Icc (-2 : ℝ) 7 = Set.image f (Set.Icc (-3 : ℝ) 2) :=
by
  sorry

end range_of_function_l464_464740


namespace comparison_of_a_b_c_l464_464064

noncomputable def f : ℝ → ℝ := sorry
def a := f (Real.logb 0.5 2)
def b := f (Real.logb 2 4)
def c := f (Real.logb 2^(0.5))

lemma problem_conditions :
  (∀ x : ℝ, f x = f (-x)) ∧ -- even function
  (∀ x : ℝ, f (x + 2) = f x) ∧ -- periodic function with period 2
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f y < f x) ∧ -- decreasing on [-1,0]
  a = f 1 ∧ -- a definition
  b = f 0 ∧ -- b definition
  c = f (2 - Real.logb 2 0.5) ∧ -- c definition
  0 < 2 - Real.logb 2 0.5 ∧ 2 - Real.logb 2 0.5 < 1 -- 0 < 2 - 2^(0.5) < 1
  :=
  by {
    sorry -- provide the proof for the conditions here
  }

theorem comparison_of_a_b_c :
  a > c ∧ c > b :=
by {
  have h_conditions := problem_conditions,
  sorry -- logic to derive a > c > b
}

end comparison_of_a_b_c_l464_464064


namespace range_of_theta_l464_464526

theorem range_of_theta
  (θ : ℝ)
  (α β : ℝ)
  (h1 : α^2 + 2 * (cos θ + 1) * α + cos θ^2 = 0)
  (h2 : β^2 + 2 * (cos θ + 1) * β + cos θ^2 = 0)
  (h3 : |α - β| ≤ 2 * sqrt 2) :
  -1/2 ≤ cos θ ∧ cos θ ≤ 1/2 := sorry

end range_of_theta_l464_464526


namespace javier_total_time_spent_l464_464574

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2

theorem javier_total_time_spent : outlining_time + writing_time + practicing_time = 117 := by
  sorry

end javier_total_time_spent_l464_464574


namespace range_of_m_l464_464495

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (m * x ^ 2 + (m - 3) * x + 1)

theorem range_of_m (m : ℝ) :
  (∀ (x : ℝ), f m x ≥ 0) →
  ∃ (S : Set ℝ), S = {[ m | 0 ≤ m ∧ m ≤ 1 ] ∪ [ m | m ≥ 9 ]} :=
by
  sorry

end range_of_m_l464_464495


namespace sequence_for_an_l464_464048

open Nat

noncomputable def a (n : ℕ+) : ℚ :=
  match n with
  | 1 => 1
  | n+1 => 2 - 1 / (a n - 4)

theorem sequence_for_an (n : ℕ+) : a n = (6 * n - 5) / (2 * n - 1) :=
by
  sorry

end sequence_for_an_l464_464048


namespace find_x_in_sequence_l464_464155

theorem find_x_in_sequence :
  ∃ x y z : Int, (z + 3 = 5) ∧ (y + z = 5) ∧ (x + y = 2) ∧ (x = -1) :=
by
  use -1, 3, 2
  sorry

end find_x_in_sequence_l464_464155


namespace amount_spent_on_first_shop_l464_464214

-- Define the conditions
def booksFromFirstShop : ℕ := 65
def costFromSecondShop : ℕ := 2000
def booksFromSecondShop : ℕ := 35
def avgPricePerBook : ℕ := 85

-- Calculate the total books and the total amount spent
def totalBooks : ℕ := booksFromFirstShop + booksFromSecondShop
def totalAmountSpent : ℕ := totalBooks * avgPricePerBook

-- Prove the amount spent on the books from the first shop is Rs. 6500
theorem amount_spent_on_first_shop : 
  (totalAmountSpent - costFromSecondShop) = 6500 :=
by
  sorry

end amount_spent_on_first_shop_l464_464214


namespace count_two_digit_numbers_l464_464012

theorem count_two_digit_numbers (low high : ℕ) (h1 : low = 29) (h2 : high = 36) : 
  {x : ℕ | x > low ∧ x < high ∧ 10 ≤ x ∧ x < 100}.card = 6 :=
by
  sorry

end count_two_digit_numbers_l464_464012


namespace number_of_valid_integers_l464_464438

def has_two_odd_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).filter (λ x, x % 2 = 1).length = 2

def is_multiple_of_34 (n : ℕ) : Prop :=
  n % 34 = 0

def count_valid_numbers : ℕ :=
  (Finset.range (3400 + 1)).filter (λ n, is_multiple_of_34 n ∧ has_two_odd_divisors n).card

theorem number_of_valid_integers : count_valid_numbers = 6 :=
by
  sorry

end number_of_valid_integers_l464_464438


namespace range_of_m_range_of_a_l464_464935

-- Define the function \( f(x) \)
def f (a : ℝ) (x : ℝ) : ℝ := (a - 0.5) * x^2 + Real.log x

-- Define the interval for part (1)
def interval1 := Icc 1 Real.exp 1

-- Statement for part (1):
theorem range_of_m (m : ℝ) : 
  (∃ x0 ∈ interval1, f 1 x0 ≤ m) ↔ m ∈ Icc 0.5 ⊤ := sorry

-- Define the condition for part (2)
def below_line (a : ℝ) : Prop :=
  ∀ x > 1, f a x < 2 * a * x

-- Statement for part (2):
theorem range_of_a : 
  (∀ x, 1 < x → f a x < 2 * a * x) ↔ (a ∈ Icc (-0.5) 0.5) := sorry

end range_of_m_range_of_a_l464_464935


namespace line_of_intersection_l464_464856

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 2 * x - y - 3 * z + 5 = 0
def plane2 (x y z : ℝ) : Prop := x + y - 2 = 0

-- Define the parametric equations
def parametric_x (t : ℝ) : ℝ := t
def parametric_y (t : ℝ) : ℝ := 2 - t
def parametric_z (t : ℝ) : ℝ := t + 1

-- The theorem that we want to prove
theorem line_of_intersection :
  ∀ t : ℝ, plane1 (parametric_x t) (parametric_y t) (parametric_z t) ∧ plane2 (parametric_x t) (parametric_y t) (parametric_z t) :=
by
  intro t
  apply And.intro
  -- Proof for plane1
  unfold plane1 parametric_x parametric_y parametric_z
  simp
  rw [←add_assoc]
  exact (eq.symm (sub_eq_add_neg _ _))
  -- Proof for plane2
  unfold plane2 parametric_x parametric_y parametric_z
  simp
  rw [add_sub_cancel]
  exact (eq.symm (sub_self _))

end line_of_intersection_l464_464856


namespace period_of_transformed_sine_curve_l464_464309

theorem period_of_transformed_sine_curve :
  let x' := λ (x : ℝ), 1 / 2 * x
  let y' := λ (y : ℝ), 3 * y
  let x := λ (x' : ℝ), 2 * x'
  let y := λ (y' : ℝ), 1 / 3 * y'
  ∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, y' (sin (x x')) = y' (sin (x' x + T)) -> T = π := by
{
  sorry
}

end period_of_transformed_sine_curve_l464_464309


namespace area_of_triangle_from_lines_l464_464318

theorem area_of_triangle_from_lines :
  ∀ (m1 m2 : ℝ) (x1 x2 y1 y2 x3 y3 : ℝ),
  m1 = 3 → m2 = -1 →
  (x1, y1) = (5, 3) →
  (x2 + y2 = 4) →
  (x3, y3) = (4, 0) ∨ (x3, y3) = (2, 2) ∨ (x3, y3) = (5, 3) →
  ∃ (A B C : ℝ × ℝ), 
    (A = (4, 0) ∨ A = (2, 2) ∨ A = (5, 3)) ∧ 
    (B = (4, 0) ∨ B = (2, 2) ∨ B = (5, 3)) ∧
    (C = (4, 0) ∨ C = (2, 2) ∨ C = (5, 3)) ∧
    abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2 = 4 :=
by
  intros m1 m2 x1 x2 y1 y2 x3 y3
  assume h1 h2 h3 h4 h5
  sorry -- skip the proof

end area_of_triangle_from_lines_l464_464318


namespace sin_squared_value_l464_464916

theorem sin_squared_value (α : ℝ) (h : tan (α + π / 4) = 3 / 4) : 
  sin^2 (π / 4 - α) = 16 / 25 := 
  sorry

end sin_squared_value_l464_464916


namespace log_base_2_div_3_a_l464_464478

theorem log_base_2_div_3_a (a : ℝ) (h1 : a > 0) (h2 : a^(2/3) = 4/9) : log (2 / 3) a = 3 :=
sorry

end log_base_2_div_3_a_l464_464478


namespace no_such_base_exists_l464_464886

theorem no_such_base_exists : ¬ ∃ b : ℕ, (b^3 ≤ 630 ∧ 630 < b^4) ∧ (630 % b) % 2 = 1 := by
  sorry

end no_such_base_exists_l464_464886


namespace cube_root_sum_lt_sqrt_sum_l464_464894

theorem cube_root_sum_lt_sqrt_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a^3 + b^3)^(1/3) < (a^2 + b^2)^(1/2) := by
    sorry

end cube_root_sum_lt_sqrt_sum_l464_464894


namespace exp_thirteen_pi_over_two_eq_i_l464_464859

theorem exp_thirteen_pi_over_two_eq_i : exp (13 * real.pi * complex.I / 2) = complex.I := 
by
  sorry

end exp_thirteen_pi_over_two_eq_i_l464_464859


namespace probability_on_hyperbola_l464_464235

open Finset

-- Define the function for the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the set of different number pairs from {1, 2, 3}
def pairs : Finset (ℕ × ℕ) := 
  {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}.to_finset

-- Define the set of pairs that lie on the hyperbola
def hyperbola_pairs : Finset (ℕ × ℕ) :=
  pairs.filter (λ mn, on_hyperbola mn.1 mn.2)

-- The theorem to prove the probability
theorem probability_on_hyperbola : 
  (hyperbola_pairs.card : ℝ) / (pairs.card : ℝ) = 1 / 3 :=
by
  -- Placeholder for the proof
  sorry

end probability_on_hyperbola_l464_464235


namespace sine_double_angle_identity_l464_464475

variable (α : ℝ)

theorem sine_double_angle_identity (h : real.sin (real.pi / 4 + α) = (real.sqrt 5) / 5) : 
  real.sin (2 * α) = -3 / 5 :=
sorry

end sine_double_angle_identity_l464_464475


namespace parallel_LS_PQ_l464_464626

open_locale big_operators

variables (A P M N C Q L S : Type*)

-- Assume that L is the intersection of lines AP and CM
def is_intersection_L (AP CM : Set (Set Type*)) : Prop :=
  L ∈ AP ∩ CM

-- Assume that S is the intersection of lines AN and CQ
def is_intersection_S (AN CQ : Set (Set Type*)) : Prop :=
  S ∈ AN ∩ CQ

-- Prove that LS ∥ PQ
theorem parallel_LS_PQ 
  (hL : is_intersection_L A P M C L) 
  (hS : is_intersection_S A N C Q S) : 
  parallel (line_through L S) (line_through P Q) :=
sorry

end parallel_LS_PQ_l464_464626


namespace choose_30_4_eq_27405_l464_464771

theorem choose_30_4_eq_27405 :
  nat.choose 30 4 = 27405 :=
by
  sorry

end choose_30_4_eq_27405_l464_464771


namespace second_smallest_5_8_9_7_l464_464763

theorem second_smallest_5_8_9_7 :
  ∃! n, n = 7 ∧
  (5 = second_smallest ([5, 8, 9, 7] : list ℕ) ∨
  8 = second_smallest ([5, 8, 9, 7] : list ℕ) ∨
  9 = second_smallest ([5, 8, 9, 7] : list ℕ) ∨
  7 = second_smallest ([5, 8, 9, 7] : list ℕ)) :=
sorry

noncomputable def second_smallest (l : list ℕ) := list.nth_le (l.qsort (≤)) 1 sorry

end second_smallest_5_8_9_7_l464_464763


namespace intersection_of_A_and_B_l464_464682

def set_A : Set ℝ := {x | x^2 ≤ 4 * x}
def set_B : Set ℝ := {x | |x| ≥ 2}

theorem intersection_of_A_and_B :
  {x | x ∈ set_A ∧ x ∈ set_B} = {x | 2 ≤ x ∧ x ≤ 4} :=
sorry

end intersection_of_A_and_B_l464_464682


namespace find_y_l464_464119

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end find_y_l464_464119


namespace factorial_floor_value_l464_464852

noncomputable def compute_expression : ℝ :=
  (Nat.factorial 2010 + Nat.factorial 2006) / (Nat.factorial 2009 + Nat.factorial 2008)

theorem factorial_floor_value : 
  ⌊compute_expression⌋ = 2009 := by
  sorry

end factorial_floor_value_l464_464852


namespace Norm_photo_count_l464_464789

variables (L M N : ℕ)

-- Conditions from the problem
def cond1 : Prop := L = N - 60
def cond2 : Prop := N = 2 * L + 10

-- Given the conditions, prove N = 110
theorem Norm_photo_count (h1 : cond1 L N) (h2 : cond2 L N) : N = 110 :=
by
  sorry

end Norm_photo_count_l464_464789


namespace grocer_sales_l464_464352

theorem grocer_sales 
  (s1 s2 s3 s4 s5 s6 s7 s8 sales : ℝ)
  (h_sales_1 : s1 = 5420)
  (h_sales_2 : s2 = 5660)
  (h_sales_3 : s3 = 6200)
  (h_sales_4 : s4 = 6350)
  (h_sales_5 : s5 = 6500)
  (h_sales_6 : s6 = 6780)
  (h_sales_7 : s7 = 7000)
  (h_sales_8 : s8 = 7200)
  (h_avg : (5420 + 5660 + 6200 + 6350 + 6500 + 6780 + 7000 + 7200 + 2 * sales) / 10 = 6600) :
  sales = 9445 := 
  by 
  sorry

end grocer_sales_l464_464352


namespace number_of_distinguishable_large_triangles_is_960_l464_464301

-- Define the number of different colors available.
def num_colors : ℕ := 8

-- Define the number of distinguishable large triangles using conditions provided.
noncomputable def num_distinguishable_large_triangles : ℕ :=
  let num_combinations := 8 + 8 * 7 + nat.choose 8 3
  8 * num_combinations

-- State the theorem to be proved.
theorem number_of_distinguishable_large_triangles_is_960 :
  num_distinguishable_large_triangles = 960 :=
by sorry

end number_of_distinguishable_large_triangles_is_960_l464_464301


namespace integer_solutions_of_system_l464_464424

theorem integer_solutions_of_system (x y z : ℤ) :
  x + y + z = 2 ∧ x^3 + y^3 + z^3 = -10 ↔ 
  (x = 3 ∧ y = 3 ∧ z = -4) ∨ 
  (x = 3 ∧ y = -4 ∧ z = 3) ∨ 
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end integer_solutions_of_system_l464_464424


namespace unique_pentagon_construction_l464_464855

structure Point := (x : ℝ) (y : ℝ)

structure Pentagon :=
(A B C D E : Point)
(midpoint_AB : Point)
(midpoint_BC : Point)
(midpoint_CD : Point)
(midpoint_DE : Point)
(midpoint_EA : Point)

-- Given midpoints F, G, H, J, K
def given_midpoints (F G H J K : Point) := True

-- Definition of a function that finds a pentagon from midpoints
noncomputable def find_pentagon : Point → Point → Point → Point → Point → Pentagon
| F, G, H, J, K => sorry

-- The proof problem statement
theorem unique_pentagon_construction (F G H J K : Point) (h : given_midpoints F G H J K) :
  ∃ (pent : Pentagon), pent.midpoint_AB = F ∧ pent.midpoint_BC = G ∧ 
                      pent.midpoint_CD = H ∧ pent.midpoint_DE = J ∧ 
                      pent.midpoint_EA = K :=
begin
  obtain pent := find_pentagon F G H J K,
  use pent,
  sorry
end

end unique_pentagon_construction_l464_464855


namespace rounding_proof_l464_464347

def rounding_question : Prop :=
  let num := 9.996
  let rounded_value := ((num * 100).round / 100)
  rounded_value ≠ 10.00

theorem rounding_proof : rounding_question :=
by
  sorry

end rounding_proof_l464_464347


namespace parallel_ls_pq_l464_464616

variables {A P C M N Q L S : Type*}
variables [affine_space ℝ (A P C M N Q)]

/- Definitions of the intersection points -/
def is_intersection_point (L : Type*) (AP : set (A P))
  (CM : set (C M)) : Prop := L ∈ AP ∧ L ∈ CM

def is_intersection_point' (S : Type*) (AN : set (A N))
  (CQ : set (C Q)) : Prop := S ∈ AN ∧ S ∈ CQ

/- Main Theorem -/
theorem parallel_ls_pq (AP CM AN CQ : set (A P))
  (PQ : set (P Q)) (L S : Type*)
  (hL : is_intersection_point L AP CM)
  (hS : is_intersection_point' S AN CQ) : L S ∥ P Q :=
sorry

end parallel_ls_pq_l464_464616


namespace length_of_train_approx_l464_464373

-- Define the given conditions
def speed_km_hr := 120 -- Speed in km/hr
def time_seconds := 15 -- Time in seconds

-- Define the conversion factor from km/hr to m/s
def km_to_m := 1000
def hr_to_s := 3600
def conversion_factor := (km_to_m : ℝ) / (hr_to_s : ℝ)

-- Define the speed in m/s
def speed_m_s := speed_km_hr * conversion_factor

-- Define the length of the train
def length_of_train := speed_m_s * time_seconds

-- The proof problem statement
theorem length_of_train_approx : length_of_train ≈ 500 := by
  sorry

end length_of_train_approx_l464_464373


namespace sin_sum_l464_464454

open Real

noncomputable def sin_angle_sum (α β : ℝ) : ℝ :=
  sin α * cos β + cos α * sin β

theorem sin_sum (α β : ℝ) (h1 : cos α = 4 / 5) (h2 : cos β = 3 / 5)
  (h3 : β ∈ set.Ioo (3 * π / 2) (2 * π)) (h4 : 0 < α) (h5 : α < β) :
  sin_angle_sum α β = -7 / 25 := by
  sorry

end sin_sum_l464_464454


namespace triangle_angle_side_inequality_l464_464772

variable {A B C : Type} -- Variables for points in the triangle
variable {a b : ℝ} -- Variables for the lengths of sides opposite to angles A and B
variable {A_angle B_angle : ℝ} -- Variables for the angles at A and B in triangle ABC

-- Define that we are in a triangle setting
def is_triangle (A B C : Type) := True

-- Define the assumption for the proof by contradiction
def assumption (a b : ℝ) := a ≤ b

theorem triangle_angle_side_inequality (h_triangle : is_triangle A B C)
(h_angle : A_angle > B_angle) 
(h_assumption : assumption a b) : a > b := 
sorry

end triangle_angle_side_inequality_l464_464772


namespace radius_of_sphere_in_truncated_cone_l464_464821

def truncated_cone_sphere_radius (r1 r2 h : ℝ) : ℝ :=
  1 / 2 * real.sqrt (h ^ 2 - (r1 - r2) ^ 2)

theorem radius_of_sphere_in_truncated_cone :
  let r1 := 12
  let r2 := 4
  let h := 15
  truncated_cone_sphere_radius r1 r2 h = real.sqrt 161 / 2 :=
sorry

end radius_of_sphere_in_truncated_cone_l464_464821


namespace find_M_value_l464_464289

def distinct_positive_integers (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_M_value (C y M A : ℕ) 
  (h1 : distinct_positive_integers C y M A) 
  (h2 : C + y + 2 * M + A = 11) : M = 1 :=
sorry

end find_M_value_l464_464289


namespace num_integers_with_properties_l464_464434

theorem num_integers_with_properties : 
  ∃ (count : ℕ), count = 6 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 3400 ∧ (n % 34 = 0) ∧ 
             ((∃ (odd_divisors : ℕ), odd_divisors = (filter (λ d, d % 2 = 1) (n.divisors)) ∧ odd_divisors.length = 2) →
             (count = 6)) :=
begin
  sorry
end

end num_integers_with_properties_l464_464434


namespace problem_statement_l464_464527

-- Define the binary operation "*"
def custom_mul (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the problem with the conditions
theorem problem_statement : custom_mul 5 (-3) = 1 := by
  sorry

end problem_statement_l464_464527


namespace point_corresponding_to_conjugate_in_first_quadrant_l464_464900

theorem point_corresponding_to_conjugate_in_first_quadrant
  (z : ℂ)
  (h : z * (2 + complex.I^7) = 3 * (complex.I^27) + 4 * (complex.I^28)) :
  complex.re (conj z) > 0 ∧ complex.im (conj z) > 0 :=
sorry

end point_corresponding_to_conjugate_in_first_quadrant_l464_464900


namespace rebecca_income_increase_l464_464241

-- Definitions for the problem
def rebecca_income : ℝ := 15000
def jimmy_initial_income : ℝ := 18000
variables (x : ℝ) -- the raise percentage

-- Jimmy's new income after the raise
def jimmy_new_income (x : ℝ) : ℝ :=
  jimmy_initial_income * (1 + x / 100)

-- Rebecca's required new income to constitute 60% of combined income
theorem rebecca_income_increase (x : ℝ) :
  let y := jimmy_new_income x,
      combined_income := rebecca_income + y,
      required_rebecca_income := 0.60 * combined_income 
  in required_rebecca_income - rebecca_income = 12000 + 270 * x :=
by
  sorry

end rebecca_income_increase_l464_464241


namespace total_cost_of_dresses_l464_464122

-- Define the costs of each dress
variables (patty_cost ida_cost jean_cost pauline_cost total_cost : ℕ)

-- Given conditions
axiom pauline_cost_is_30 : pauline_cost = 30
axiom jean_cost_is_10_less_than_pauline : jean_cost = pauline_cost - 10
axiom ida_cost_is_30_more_than_jean : ida_cost = jean_cost + 30
axiom patty_cost_is_10_more_than_ida : patty_cost = ida_cost + 10

-- Statement to prove total cost
theorem total_cost_of_dresses : total_cost = pauline_cost + jean_cost + ida_cost + patty_cost 
                                 → total_cost = 160 :=
by {
  -- Proof is left as an exercise
  sorry
}

end total_cost_of_dresses_l464_464122


namespace total_time_spent_l464_464576

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2
def total_time : ℕ := outlining_time + writing_time + practicing_time

theorem total_time_spent : total_time = 117 := by
  sorry

end total_time_spent_l464_464576


namespace find_a_plus_b_l464_464486

noncomputable def line_l_slope : ℝ :=
  Real.tan (Real.pi / 4)

def line_l1_slope (a : ℝ) : ℝ :=
  (2 - (-1)) / (3 - a)

def line_l2_slope (b : ℝ) : ℝ :=
  -2 / b

theorem find_a_plus_b (a b : ℝ) (h_l_slope : line_l_slope = 1)
    (h_perpendicular : line_l1_slope a = -1)
    (h_parallel : line_l2_slope b = line_l1_slope a) :
    a + b = 8 :=
sorry

end find_a_plus_b_l464_464486


namespace find_f_2011_l464_464921

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 2 then 2 * x^2
  else sorry  -- Placeholder, since f is only defined in (0, 2)

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 2) = -f x

theorem find_f_2011 : f 2011 = -2 :=
by
  -- Use properties of f to reduce and eventually find f(2011)
  sorry

end find_f_2011_l464_464921


namespace probability_point_A_on_hyperbola_l464_464223

-- Define the set of numbers
def numbers : List ℕ := [1, 2, 3]

-- Define the coordinates of point A taken from the set, where both numbers are different
def point_A_pairs : List (ℕ × ℕ) :=
  [ (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) ]

-- Define the function indicating if a point (m, n) lies on the hyperbola y = 6/x
def lies_on_hyperbola (m n : ℕ) : Prop :=
  n = 6 / m

-- Calculate the probability of a point lying on the hyperbola
theorem probability_point_A_on_hyperbola : 
  (point_A_pairs.countp (λ (p : ℕ × ℕ), lies_on_hyperbola p.1 p.2)).toRat / (point_A_pairs.length).toRat = 1 / 3 := 
sorry

end probability_point_A_on_hyperbola_l464_464223


namespace number_of_extreme_points_l464_464868

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - log x - x

-- Define the domain condition
def f_domain (x : ℝ) : Prop := x > 0

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 6 * x - (1 / x) - 1

-- The main theorem statement
theorem number_of_extreme_points : (∃ x, f_domain x ∧ f' x = 0) ∧ ∀ x, f_domain x → f' x < 0 ∨ f' x > 0 → 1 :=
begin
  sorry
end

end number_of_extreme_points_l464_464868


namespace div_sum_eq_not_odd_prime_l464_464246

theorem div_sum_eq_not_odd_prime (n : ℕ) :
  (n! % (∑ k in finset.range (n+1), k) = 0) ↔ ¬ nat.prime (n + 1) ∨ (n + 1) % 2 = 0 :=
sorry

end div_sum_eq_not_odd_prime_l464_464246


namespace abs_fraction_eq_sqrt_77_div_7_l464_464528

theorem abs_fraction_eq_sqrt_77_div_7 {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 9 * a * b) :
  abs ((a + b) / (a - b)) = real.sqrt 77 / 7 :=
sorry

end abs_fraction_eq_sqrt_77_div_7_l464_464528


namespace imaginary_part_of_z_l464_464069

-- Define the problem statement
theorem imaginary_part_of_z (z : ℂ) (h : (1 + z) / (1 - z) = complex.I) : z.im = 1 :=
sorry

end imaginary_part_of_z_l464_464069


namespace javier_total_time_spent_l464_464575

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2

theorem javier_total_time_spent : outlining_time + writing_time + practicing_time = 117 := by
  sorry

end javier_total_time_spent_l464_464575


namespace ratio_a_b_equals_sqrt2_l464_464980

variable (A B C a b c : ℝ) -- Define the variables representing the angles and sides.

-- Assuming the sides a, b, c are positive and a triangle is formed (non-degenerate)
axiom triangle_ABC : 0 < a ∧ 0 < b ∧ 0 < c

-- Assuming the sum of the angles in a triangle equals 180 degrees (π radians)
axiom sum_angles_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : b * Real.cos C + c * Real.cos B = Real.sqrt 2 * b

-- Problem statement to be proven
theorem ratio_a_b_equals_sqrt2 : (a / b) = Real.sqrt 2 :=
by
  -- Assume the problem statement is correct
  sorry

end ratio_a_b_equals_sqrt2_l464_464980


namespace area_of_sector_l464_464127

theorem area_of_sector (s θ : ℝ) (r : ℝ) (h_s : s = 4) (h_θ : θ = 2) (h_r : r = s / θ) :
  (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end area_of_sector_l464_464127


namespace travel_time_l464_464692

namespace NatashaSpeedProblem

def distance : ℝ := 60
def speed_limit : ℝ := 50
def speed_over_limit : ℝ := 10
def actual_speed : ℝ := speed_limit + speed_over_limit

theorem travel_time : (distance / actual_speed) = 1 := by
  sorry

end NatashaSpeedProblem

end travel_time_l464_464692


namespace probability_reverse_order_probability_second_to_bottom_l464_464384

noncomputable def prob_reverse_order : ℚ := 1/8
noncomputable def prob_second_to_bottom : ℚ := 1/8

theorem probability_reverse_order (n: ℕ) (h : n = 4) 
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_reverse_order = 1/8 :=
by sorry

theorem probability_second_to_bottom (n: ℕ) (h : n = 4)
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_second_to_bottom = 1/8 :=
by sorry

end probability_reverse_order_probability_second_to_bottom_l464_464384


namespace green_dots_third_row_l464_464204

noncomputable def row_difference (a b : Nat) : Nat := b - a

theorem green_dots_third_row (a1 a2 a4 a5 a3 d : Nat)
  (h_a1 : a1 = 3)
  (h_a2 : a2 = 6)
  (h_a4 : a4 = 12)
  (h_a5 : a5 = 15)
  (h_d : row_difference a2 a1 = d)
  (h_d_consistent : row_difference a2 a1 = row_difference a4 a3) :
  a3 = 9 :=
sorry

end green_dots_third_row_l464_464204


namespace inner_square_area_l464_464999

theorem inner_square_area :
  ∀ (AB BE : ℝ), 
  AB = 10 ∧ BE = 2 →
  ∃ (x : ℝ), (4 * real.sqrt 6 - 2) = x ∧ (x ^ 2 = 92 - 16 * real.sqrt 6) :=
by
  intros AB BE h
  have h₁ : AB = 10 := h.1
  have h₂ : BE = 2 := h.2
  -- The rest of the proof goes here, skipped with sorry
  sorry

end inner_square_area_l464_464999


namespace side_length_after_transformation_is_8_l464_464805

-- Define the points and the translation vector
def initial_vertices : List (ℝ × ℝ) := [(5, -3), (9, 1), (5, 5), (1, 1)]
def translation_vector : ℝ × ℝ := (-2, 3)

-- Define the translation of each point
def translate (p : ℝ × ℝ) (t : ℝ × ℝ) : ℝ × ℝ := (p.1 + t.1, p.2 + t.2)
def translated_vertices : List (ℝ × ℝ) := initial_vertices.map (λ p, translate p translation_vector)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove that the side length after transformation is 8 units
theorem side_length_after_transformation_is_8 :
  (distance (3, 0) (3, 8) = 8) :=
  by sorry

end side_length_after_transformation_is_8_l464_464805


namespace probability_on_hyperbola_l464_464238

open Finset

-- Define the function for the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the set of different number pairs from {1, 2, 3}
def pairs : Finset (ℕ × ℕ) := 
  {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}.to_finset

-- Define the set of pairs that lie on the hyperbola
def hyperbola_pairs : Finset (ℕ × ℕ) :=
  pairs.filter (λ mn, on_hyperbola mn.1 mn.2)

-- The theorem to prove the probability
theorem probability_on_hyperbola : 
  (hyperbola_pairs.card : ℝ) / (pairs.card : ℝ) = 1 / 3 :=
by
  -- Placeholder for the proof
  sorry

end probability_on_hyperbola_l464_464238


namespace number_of_sheets_in_stack_l464_464357

theorem number_of_sheets_in_stack (
  thickness_per_packet : ℕ,
  height_of_packet : ℝ,
  height_of_stack : ℝ
) (h1 : thickness_per_packet = 400)
  (h2 : height_of_packet = 4)
  (h3 : height_of_stack = 6) :
  Nat.floor (height_of_stack / (height_of_packet / thickness_per_packet)) = 600 :=
by
  sorry

end number_of_sheets_in_stack_l464_464357


namespace regular_polygon_sides_l464_464803

theorem regular_polygon_sides (P S : ℕ) (hP: P = 160) (hS: S = 10) : 
  ∃ n : ℕ, P = n * S ∧ n = 16 :=
by {
  let n := P / S,
  have hn : n = 16 := by norm_num [P, S, hP, hS, -add_comm],
  exact ⟨n, by simp [P, S, hP, hS, hn]⟩,
}

end regular_polygon_sides_l464_464803


namespace no_three_real_numbers_satisfy_inequalities_l464_464702

theorem no_three_real_numbers_satisfy_inequalities (a b c : ℝ) :
  ¬ (|a| < |b - c| ∧ |b| < |c - a| ∧ |c| < |a - b| ) :=
by
  sorry

end no_three_real_numbers_satisfy_inequalities_l464_464702


namespace polynomial_equality_of_values_l464_464674

theorem polynomial_equality_of_values 
  (k : ℕ) (hk : k ≥ 6) 
  (P : Polynomial ℤ) 
  (x : Fin k → ℤ) 
  (hx : Function.Injective x) 
  (hp : ∀ i : Fin k, P.eval (x i) ∈ Finset.range (k - 1) + 1) 
  : ∀ i j : Fin k, P.eval (x i) = P.eval (x j) := 
by
  sorry

end polynomial_equality_of_values_l464_464674


namespace reciprocal_sum_eq_inv_S_l464_464195

variable (q : ℝ) (n : ℕ)

noncomputable def S (q : ℝ) (n : ℕ) : ℝ := 3 * (1 - q^(4*n)) / (1 - q^2)

theorem reciprocal_sum_eq_inv_S (hq : q ≠ 1) :
  (∑ i in finset.range (2 * n), (1 / (3 * q^(2 * i)))) 
  = 1 / S q n := 
by 
  sorry

end reciprocal_sum_eq_inv_S_l464_464195


namespace zeros_of_f_l464_464022

noncomputable def f (z : ℂ) : ℂ := 1 + Complex.cos z

-- Prove that zeros of the function f(z) = 1 + cos(z) are (2n + 1)π with order 1
theorem zeros_of_f (n : ℤ) : 
  let z_n := (2 * n + 1) * Real.pi in
  f z_n = 0 ∧ 
  ∃ δ > 0, ∀ ε (ε > 0), ∃ z, 0 < Complex.abs (z - z_n) < δ ∧ f z = 0 :=
by
  sorry

end zeros_of_f_l464_464022


namespace find_x_plus_y_l464_464173

theorem find_x_plus_y (x y : ℕ) 
  (h1 : 4^x = 16^(y + 1)) 
  (h2 : 5^(2 * y) = 25^(x - 2)) : 
  x + y = 2 := 
sorry

end find_x_plus_y_l464_464173


namespace sum_sequence_l464_464067

noncomputable def seq_a (n : ℕ) : ℤ :=
  2 * n + 1

noncomputable def sum_S (n : ℕ) : ℤ :=
  2 * n + n^2 + 1

noncomputable def c_n (n : ℕ) : ℝ :=
  let an := seq_a n in
  let an1 := seq_a (n + 1) in
  let Sn := sum_S n in
  (an * an1) / Sn - ⌊(an * an1) / Sn⌋

noncomputable def T_n (n : ℕ) : ℝ :=
  (5 * n^2 + 3 * n - 8) / (4 * n^2 + 12 * n + 8)

theorem sum_sequence (n : ℕ) (hn : n > 0) : 
  let Sn := sum_S n in
  let an := seq_a n in
  let cn (m : ℕ) := c_n m in
  (∑ i in finset.range n, cn i) = T_n n := sorry

end sum_sequence_l464_464067


namespace min_value_expression_product_identity_l464_464786

-- Proof Problem 1
theorem min_value_expression (a b c : ℝ) (h : 2 * a + 2 * b + c = 8) : 
  ∃ (x : ℝ), x = (a - 1)^2 + (b + 2)^2 + (c - 3)^2 ∧ x = 49 / 9 :=
sorry

-- Proof Problem 2
theorem product_identity (n : ℕ) (h : n ≥ 2) : 
  (∏ i in Finset.range n, (1 - 1/(i+1)^2 : ℚ)) = (n + 1) / (2 * n) :=
sorry

end min_value_expression_product_identity_l464_464786


namespace rectangle_perimeter_divisible_by_4_l464_464810

theorem rectangle_perimeter_divisible_by_4 :
  ∃ (rect : ℕ × ℕ), ((rect.1 + rect.2) % 2 = 0) ∧
  (rect.1 ≤ 2009) ∧ (rect.2 ≤ 2009) := 
begin
  -- Introducing our main square aligned with grid points
  let n := 2009,

  -- Array of pairs of sides (length, width) of all rectangles formed by subdividing the square
  let subdivisions := (list.range (n + 1)).product (list.range (n + 1)),
  
  -- Filter out invalid subdivisions (where one or more sides are zero), and select valid rectangles
  let valid_rectangles := subdivisions.filter (λ rect, rect.1 ≠ 0 ∧ rect.2 ≠ 0),

  -- Define a predicate to check if their perimeter is divisible by 4
  let has_perimeter_divisible_by_4 := valid_rectangles.any (λ rect, (rect.1 + rect.2) % 2 = 0),

  -- If the predicate is true, then such a rectangle exists
  have h : ∃ (rect : ℕ × ℕ), (rect ∈ valid_rectangles) ∧ ((rect.1 + rect.2) % 2 = 0),
  {
    exact valid_rectangles.exists (λ rect, (rect.1 + rect.2) % 2 = 0)
  },

  -- This confirms our theorem
  exact h,
end

end rectangle_perimeter_divisible_by_4_l464_464810


namespace triangle_area_ratios_l464_464559

theorem triangle_area_ratios (A B C D E F : Point)
  (hAB: dist A B = 130) (hAC: dist A C = 130)
  (hAD: dist A D = 50) (hCF: dist C F = 50) :
  area (triangle C E F) / area (triangle D B E) = 8 / 5 :=
by
  sorry

end triangle_area_ratios_l464_464559


namespace total_amount_spent_l464_464124

variables (P J I T : ℕ)

-- Given conditions
def Pauline_dress : P = 30 := sorry
def Jean_dress : J = P - 10 := sorry
def Ida_dress : I = J + 30 := sorry
def Patty_dress : T = I + 10 := sorry

-- Theorem to prove the total amount spent
theorem total_amount_spent :
  P + J + I + T = 160 :=
by
  -- Placeholder for proof
  sorry

end total_amount_spent_l464_464124


namespace parallel_LS_PQ_l464_464628

open_locale big_operators

variables (A P M N C Q L S : Type*)

-- Assume that L is the intersection of lines AP and CM
def is_intersection_L (AP CM : Set (Set Type*)) : Prop :=
  L ∈ AP ∩ CM

-- Assume that S is the intersection of lines AN and CQ
def is_intersection_S (AN CQ : Set (Set Type*)) : Prop :=
  S ∈ AN ∩ CQ

-- Prove that LS ∥ PQ
theorem parallel_LS_PQ 
  (hL : is_intersection_L A P M C L) 
  (hS : is_intersection_S A N C Q S) : 
  parallel (line_through L S) (line_through P Q) :=
sorry

end parallel_LS_PQ_l464_464628


namespace train_length_l464_464818

  noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℕ) : ℝ :=
    let speed_ms := speed_kmh * 1000 / 3600
    let distance := speed_ms * time_s
    distance / 2

  theorem train_length (speed_kmh : ℝ) (time_s : ℕ) (length : ℝ) :
    speed_kmh = 64 → time_s = 27 → length_of_train speed_kmh time_s ≈ 239.97 :=
  by
    intros hs ht
    rw [hs, ht]
    unfold length_of_train
    sorry
  
end train_length_l464_464818


namespace arithmetic_sequence_a5_l464_464147

theorem arithmetic_sequence_a5 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : S 4 = 16)
  (h_sum : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2 - a 1))) / 2)
  (h_a : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  a 5 = 9 :=
by 
  sorry

end arithmetic_sequence_a5_l464_464147


namespace highest_coefficient_in_expansion_l464_464266

theorem highest_coefficient_in_expansion (x y : ℝ) :
  (∃ k : ℕ, (k ≤ 9) ∧ 
  (∃ c : ℝ, 
    coefficient (binomial_expansion x y 9) k = c 
  )) → coefficient (binomial_expansion x y 9) 4 = 126 := sorry

end highest_coefficient_in_expansion_l464_464266


namespace find_teaspoons_of_salt_l464_464163

def sodium_in_salt (S : ℕ) : ℕ := 50 * S
def sodium_in_parmesan (P : ℕ) : ℕ := 25 * P

-- Initial total sodium amount with 8 ounces of parmesan
def initial_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 8

-- Reduced sodium after removing 4 ounces of parmesan
def reduced_sodium (S : ℕ) : ℕ := initial_total_sodium S * 2 / 3

-- Reduced sodium with 4 fewer ounces of parmesan cheese
def new_total_sodium (S : ℕ) : ℕ := sodium_in_salt S + sodium_in_parmesan 4

theorem find_teaspoons_of_salt : ∃ (S : ℕ), reduced_sodium S = new_total_sodium S ∧ S = 2 :=
by
  sorry

end find_teaspoons_of_salt_l464_464163


namespace proof_problem1_proof_problem2_l464_464411

namespace MathProof

-- Proof problem (1)
theorem proof_problem1 : 
  sqrt 27 / sqrt 3 - 16 * 4⁻¹ + abs (-5) - (3 - sqrt 3) ^ 0 = 3 := 
by
  sorry

-- Proof problem (2)
theorem proof_problem2 : 
  2 * tan (pi / 6) - abs (1 - sqrt 3) + (2014 - sqrt 2) ^ 0 + sqrt (1 / 3) = 2 := 
by 
  sorry

end MathProof

end proof_problem1_proof_problem2_l464_464411


namespace find_f_expression_g_max_min_l464_464797

noncomputable def f (x : ℝ) : ℝ := sin (2 * x - π / 3)
noncomputable def g (x : ℝ) : ℝ := sin (4 * x - 2 * π / 3)

theorem find_f_expression :
  let f : ℝ → ℝ := sin ∘ (λ x, 2 * x - π / 3)
  ∃ (ω : ℝ) (φ : ℝ), (0 < ω) ∧ (abs φ < π / 2) ∧ 
  f = (λ x, sin (ω * x + φ)) ∧ 
  (∀ x ∈ Icc (5 * π / 12) (11 * π / 12), deriv f x < 0) :=
by
  sorry

theorem g_max_min :
  let g : ℝ → ℝ := sin ∘ (λ x, 4 * x - 2 * π / 3)
  ∃ (max_value min_value : ℝ), 
  (max_value = 1) ∧ (min_value = -1/2) ∧ 
  g (π / 8) = min_value ∧ g (3 * π / 8) = max_value :=
by
  sorry

end find_f_expression_g_max_min_l464_464797


namespace abs_five_is_five_l464_464727

theorem abs_five_is_five : abs 5 = 5 := by
  sorry

end abs_five_is_five_l464_464727


namespace polynomial_int_coeff_property_l464_464008

open Polynomial

theorem polynomial_int_coeff_property 
  (P : Polynomial ℤ) : 
  (∀ s t : ℝ, P.eval s ∈ ℤ ∧ P.eval t ∈ ℤ → P.eval (s * t) ∈ ℤ) ↔ 
  ∃ (n : ℕ) (k : ℤ), P = X^n + C k ∨ P = -X^n + C k :=
by
  sorry

end polynomial_int_coeff_property_l464_464008


namespace prob_reverse_order_prob_second_from_bottom_l464_464386

section CementBags

-- Define the basic setup for the bags and worker choices
variables (bags : list ℕ) (choices : list ℕ) (shed gate : list ℕ)

-- This ensures we are indeed considering 4 bags in a sequence
axiom bags_is_4 : bags.length = 4

-- This captures that the worker makes 4 sequential choices with equal probability for each, where 0 represents shed and 1 represents gate
axiom choices_is_4_and_prob : (choices.length = 4) ∧ (∀ i in choices, i ∈ [0, 1])

-- This condition captures that eventually, all bags must end up in the shed
axiom all_bags_in_shed : ∀ b ∈ bags, b ∈ shed

-- Prove for part (a): Probability of bags in reverse order in shed is 1/8
theorem prob_reverse_order : (choices = [0, 0, 0, 0] → (reverse bags = shed)) → 
  (probability (choices = [0, 0, 0, 0]) = 1 / 8) :=
 sorry

-- Prove for part (b): Probability of the second-from-bottom bag in the bottom is 1/8
theorem prob_second_from_bottom : 
  (choices = [1, 1, 0, 0] → (shed.head = bags.nth 1.get_or_else 0)) → 
  (probability (choices = [1, 1, 0, 0]) = 1 / 8) :=
 sorry

end CementBags

end prob_reverse_order_prob_second_from_bottom_l464_464386


namespace solution_system_of_equations_l464_464252

theorem solution_system_of_equations (n : ℕ) (a : Fin n → ℝ) (x : Fin n → ℝ) (A : ℝ) :
  (∀ i j : Fin n, x i / a i = x j / a j) ∧ (∑ i, x i = A) →
  ((∑ i, a i ≠ 0) ∧ (∀ i, x i = (A * a i) / (∑ i, a i))) ∨
  ((∑ i, a i = 0) ∧ (A = 0) ∧ (∃ t : ℝ, ∀ i, x i = a i * t)) ∨
  ((∑ i, a i = 0) ∧ (A ≠ 0) ∧ False) :=
sorry

end solution_system_of_equations_l464_464252


namespace geometric_mean_fraction_l464_464323

noncomputable def geometric_mean (a b : ℝ) : ℝ :=
  real.sqrt (a * b)

theorem geometric_mean_fraction (a b : ℝ) (h1 : a = 3/7) (h2 : b = 5/9) : 
  geometric_mean a b = real.sqrt (5 / 21) := by
  rw [geometric_mean, h1, h2]
  norm_num
  sorry

end geometric_mean_fraction_l464_464323


namespace hh_two_eq_902_l464_464664

def h (x : ℝ) : ℝ := 3 * x^2 + 2 * x + 1

theorem hh_two_eq_902 : h (h 2) = 902 := 
by
  sorry

end hh_two_eq_902_l464_464664


namespace intersection_P_Q_eq_1_2_l464_464504

noncomputable def P := {x : ℕ | -x^2 + x + 2 ≥ 0}
noncomputable def Q := {x : ℝ | 0 < x ∧ x < Real.exp 1}

theorem intersection_P_Q_eq_1_2 : (P ∩ Q : Set ℝ) = {1, 2} :=
by
  sorry

end intersection_P_Q_eq_1_2_l464_464504


namespace find_m_l464_464499

-- Definitions of the conditions
def is_linear (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

def y (m : ℝ) : ℝ → ℝ := λ x, (m-2) * x^(m^2-3) + m

-- Statement of the theorem
theorem find_m : ∀ m : ℝ, is_linear (y m) → m = -2 :=
by
  assume m : ℝ,
  assume h : is_linear (y m),
  sorry -- proof not required

end find_m_l464_464499


namespace ladder_slides_out_l464_464790

-- Define the conditions and the theorem
theorem ladder_slides_out :
  ∀ (h l x : ℝ), h = 30 ∧ l = 8 ∧ x = 5 →
  ∃ y : ℝ, y = 10.1 :=
by
  intro h l x
  intro h_l_x
  exists 10.1
  sorry

end ladder_slides_out_l464_464790


namespace triangle_angle_C_maximum_triangle_area_l464_464543

-- Part (I)
theorem triangle_angle_C (A B C a b c : ℝ) (h₁ : 2 * a + b = c * cos(A + C) / cos C) :
  C = 2 * π / 3 := 
sorry

-- Part (II)
theorem maximum_triangle_area (a b : ℝ) (h₁ : C = 2 * π / 3) (h₂ : c = 2) :
  ∃ (S : ℝ), S = sqrt 3 / 3 :=
sorry

end triangle_angle_C_maximum_triangle_area_l464_464543


namespace sum_of_side_lengths_in_cm_l464_464512

-- Definitions for the given conditions
def side_length_meters : ℝ := 2.3
def meters_to_centimeters : ℝ := 100
def num_sides : ℕ := 8

-- The statement to prove
theorem sum_of_side_lengths_in_cm :
  let side_length_cm := side_length_meters * meters_to_centimeters in
  let total_length_cm := side_length_cm * (num_sides : ℝ) in
  total_length_cm = 1840 :=
by
  sorry

end sum_of_side_lengths_in_cm_l464_464512


namespace count_multiples_of_34_with_two_odd_divisors_l464_464431

-- Define a predicate to check if a number has exactly 2 odd natural divisors
def has_exactly_two_odd_divisors (n : ℕ) : Prop :=
  (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ d1 * d2 = n ∧ d1 % 2 = 1 ∧ d2 % 2 = 1) ∧
  (∀ d : ℕ, d ∣ n → (d % 2 = 0 ∨ d = d1 ∨ d = d2))

-- Main theorem to prove the number of integers that satisfy the given conditions
theorem count_multiples_of_34_with_two_odd_divisors : 
  let valid_numbers := {n : ℕ | n ≤ 3400 ∧ n % 34 = 0 ∧ has_exactly_two_odd_divisors n} in
  valid_numbers.to_finset.card = 6 :=
by
  sorry

end count_multiples_of_34_with_two_odd_divisors_l464_464431


namespace move_factor_inside_sqrt_l464_464690

theorem move_factor_inside_sqrt (a : ℝ) (h : a < 0) : a * real.sqrt (-1 / a) = - real.sqrt (-a) :=
sorry

end move_factor_inside_sqrt_l464_464690


namespace find_positive_integers_l464_464010

theorem find_positive_integers (n : ℕ) : 
  (∀ a : ℕ, a.gcd n = 1 → 2 * n * n ∣ a ^ n - 1) ↔ (n = 2 ∨ n = 6 ∨ n = 42 ∨ n = 1806) :=
sorry

end find_positive_integers_l464_464010


namespace num_of_subsets_l464_464525

open Set

theorem num_of_subsets : {A : Set ℤ // A ∪ {-1, 1} = {-1, 1}}.card = 4 :=
by sorry

end num_of_subsets_l464_464525


namespace at_least_six_students_solve_same_exercise_l464_464712

theorem at_least_six_students_solve_same_exercise (S : Finset ℕ) (E : Finset ℕ)
  (solves : ℕ → ℕ) (hS : S.card = 16) (hE : E.card = 3)
  (hsolve : ∀ s ∈ S, solves s ∈ E) :
  ∃ e ∈ E, (S.filter (λ s, solves s = e)).card ≥ 6 :=
by
  sorry

end at_least_six_students_solve_same_exercise_l464_464712


namespace complex_roots_properties_l464_464959

theorem complex_roots_properties (a : ℝ) (h_a : -2 < a ∧ a < 2) (z1 z2 : ℂ)
    (h_roots : ∀ (x : ℂ), x^2 + (↑a : ℂ)*x + 1 = 0 ↔ x = z1 ∨ x = z2) :
    (z1.conj = z2) ∧ (z1 * z2 = 1) ∧ (z1 + (a / 2 : ℂ)).im ≠ 0 := 
sorry

end complex_roots_properties_l464_464959


namespace winning_ratio_at_first_quarter_l464_464836

variable (W1 W2 W3 L1 : ℕ)

theorem winning_ratio_at_first_quarter :
  L1 = 10 →
  W2 = W1 + 10 →
  W3 = W1 + 30 →
  W1 + W2 + W3 = 80 →
  W1 = 10 →
  W1 / L1 = 1 :=
by
  intros hL1 hW2 hW3 hTotal hW1
  rw [hL1, hW1]
  exact nat.div_self _ sorry

end winning_ratio_at_first_quarter_l464_464836


namespace day_of_week_150th_day_of_year_N_minus_1_l464_464569

/-- Given that the 250th day of year N is a Friday and year N is a leap year,
    prove that the 150th day of year N-1 is a Friday. -/
theorem day_of_week_150th_day_of_year_N_minus_1
  (N : ℕ) 
  (H1 : (250 % 7 = 5) → true)  -- Condition that 250th day is five days after Sunday (Friday).
  (H2 : 366 % 7 = 2)           -- Condition that year N is a leap year with 366 days.
  (H3 : (N - 1) % 7 = (N - 1) % 7) -- Used for year transition check.
  : 150 % 7 = 5 := sorry       -- Proving that the 150th of year N-1 is Friday.

end day_of_week_150th_day_of_year_N_minus_1_l464_464569


namespace proving_real_and_imaginary_parts_l464_464665

noncomputable def root_equation (z : ℂ) := z^2 + 2*z + (4 - 8*ℂ.I) = 0

def product_real_parts (z1 z2 : ℂ) : ℂ :=
  (z1.re) * (z2.re)

def sum_imaginary_parts (z1 z2 : ℂ) : ℂ :=
  (z1.im) + (z2.im)

theorem proving_real_and_imaginary_parts :
  ∃ (z1 z2 : ℂ), root_equation z1 ∧ root_equation z2 ∧ product_real_parts z1 z2 = -7 ∧ sum_imaginary_parts z1 z2 = 0 :=
sorry

end proving_real_and_imaginary_parts_l464_464665


namespace find_ABC_base10_l464_464256

theorem find_ABC_base10
  (A B C : ℕ)
  (h1 : 0 < A ∧ A < 6)
  (h2 : 0 < B ∧ B < 6)
  (h3 : 0 < C ∧ C < 6)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : B + C = 6)
  (h6 : A + 1 = C)
  (h7 : A + B = C) :
  100 * A + 10 * B + C = 415 :=
by
  sorry

end find_ABC_base10_l464_464256


namespace gcd_2023_1991_l464_464428

theorem gcd_2023_1991 : Nat.gcd 2023 1991 = 1 :=
by
  sorry

end gcd_2023_1991_l464_464428


namespace find_tan_of_angle_l464_464275

def initial_side α : Prop :=
α = 0

def terminal_side_through_point (α : ℝ) : Prop :=
∃ (P : ℝ × ℝ), P = (-2, 1) ∧ tan α = P.2 / P.1

theorem find_tan_of_angle (α : ℝ) (h1 : initial_side α) (h2 : terminal_side_through_point α) : tan α = -1 / 2 :=
by
  rcases h2 with ⟨P, hP1, hP2⟩
  simp at hP1
  simp [hP1] at hP2
  exact hP2
sorry

end find_tan_of_angle_l464_464275


namespace distance_walked_l464_464164

-- Definition of the walking rate
def walking_rate : ℝ := 1 / 18

-- Definition of time in minutes
def time_in_minutes : ℕ := 45

-- The correct distance to be proven
theorem distance_walked :
  walking_rate * (time_in_minutes : ℝ) = 2.50 :=
sorry

end distance_walked_l464_464164


namespace percent_counties_fewer_than_100000_l464_464267

def P (A : Type) (f: A -> Prop) (p: A -> ℝ) : ℝ :=
  Σ' (a : A), if f a then p a else 0

variables 
  (P_less_than_10k : ℝ := 0.25)
  (P_10k_to_99k : ℝ := 0.59)
  (P_100k_or_more : ℝ := 0.16)

theorem percent_counties_fewer_than_100000 :
  P_less_than_10k + P_10k_to_99k = 0.84 :=
by
  sorry

end percent_counties_fewer_than_100000_l464_464267


namespace translated_line_eqn_l464_464557

theorem translated_line_eqn
  (c : ℝ) :
  ∀ (y_eqn : ℝ → ℝ), 
    (∀ x, y_eqn x = 2 * x + 1) →
    (∀ x, (y_eqn (x - 2) - 3) = (2 * x - 6)) :=
by
  sorry

end translated_line_eqn_l464_464557


namespace units_digit_base7_product_l464_464202

theorem units_digit_base7_product (a b : ℕ) (ha : a = 354) (hb : b = 78) : (a * b) % 7 = 4 := by
  sorry

end units_digit_base7_product_l464_464202


namespace count_three_digit_integers_divisible_by_5_l464_464517

theorem count_three_digit_integers_divisible_by_5 :
  let digits := {d | d ∈ {6, 7, 8, 9}};
  (0 < ∀ digit ∈ digits) →
  (∀ x : ℕ, 100 ≤ x ∧ x < 1000 →
    (∀ d ∈ nat.digits 10 x, d > 5) →
    (x % 5 = 0) → (card {x | (∀ d ∈ nat.digits 10 x, d > 5) ∧ x % 5 = 0}) = 32) :=
by
  intros digits h_digit_pos h_divisibility h_ranges
  sorry

end count_three_digit_integers_divisible_by_5_l464_464517


namespace tan_expression_l464_464112

variables {γ δ : ℝ}

theorem tan_expression 
  (hγ : Real.tan γ = 3) 
  (hδ : Real.tan δ = 2) : 
  Real.tan (2 * γ - δ) = 11 / 2 := 
sorry

end tan_expression_l464_464112


namespace average_age_youngest_and_oldest_new_employees_l464_464546

-- Given conditions
variables (company_employees : ℕ) (avg_age_20_1 avg_age_20_2 avg_age_10 : ℕ)
variables (new_hires : ℕ) (combined_age_new_hires : ℕ)
variables (age_diff : ℕ)

-- Assuming the specific values from the problem
def fifty_employees : Prop := company_employees = 50
def ages_avg_20_1 : Prop := avg_age_20_1 = 30
def ages_avg_20_2 : Prop := avg_age_20_2 = 40
def ages_avg_10 : Prop := avg_age_10 = 50
def five_new_hires : Prop := new_hires = 5
def combined_age_five_new_hires : Prop := combined_age_new_hires = 150
def difference_young_old : Prop := age_diff = 20

-- Lean statement for proof problem
theorem average_age_youngest_and_oldest_new_employees
  (h1 : fifty_employees)
  (h2 : ages_avg_20_1)
  (h3 : ages_avg_20_2)
  (h4 : ages_avg_10)
  (h5 : five_new_hires)
  (h6 : combined_age_five_new_hires)
  (h7 : difference_young_old)
  : (20 + 40) / 2 = 30 := by
  sorry

end average_age_youngest_and_oldest_new_employees_l464_464546


namespace solve_for_y_l464_464115

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end solve_for_y_l464_464115


namespace distance_between_A_and_B_l464_464698

-- Definitions and conditions based on problem statement
def speed_A : ℕ := 70
def speed_B : ℕ := 60
def extra_distance_from_midpoint : ℕ := 80

-- Lean statement to prove the total distance between points A and B
theorem distance_between_A_and_B :
  let relative_distance_covered := 2 * extra_distance_from_midpoint in
  let speed_difference := speed_A - speed_B in
  let meeting_time := relative_distance_covered / speed_difference in
  let combined_speed := speed_A + speed_B in
  let total_distance := meeting_time * combined_speed in
  total_distance = 2080 := by
  sorry

end distance_between_A_and_B_l464_464698


namespace probability_point_on_hyperbola_l464_464219

-- Define the problem conditions
def number_set := {1, 2, 3}
def point_on_hyperbola (x y : ℝ) : Prop := y = 6 / x

-- Formalize the problem statement
theorem probability_point_on_hyperbola :
  let combinations := ({(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} : set (ℝ × ℝ)) in
  let on_hyperbola := set.filter (λ p : ℝ × ℝ, point_on_hyperbola p.1 p.2) combinations in
  fintype.card on_hyperbola / fintype.card combinations = 1 / 3 :=
by sorry

end probability_point_on_hyperbola_l464_464219


namespace complex_imaginary_part_l464_464880

theorem complex_imaginary_part (i : ℂ) (h1 : i = complex.I) :
  complex.imag (1 / (1 + i)^2) = -1 / 2 :=
by
  sorry

end complex_imaginary_part_l464_464880


namespace total_gas_cost_l464_464410

def car_city_mpg : ℝ := 30
def car_highway_mpg : ℝ := 40
def city_miles : ℝ := 60 + 40 + 25
def highway_miles : ℝ := 200 + 150 + 180
def gas_price_per_gallon : ℝ := 3.00

theorem total_gas_cost : 
  (city_miles / car_city_mpg + highway_miles / car_highway_mpg) * gas_price_per_gallon = 52.25 := 
by
  sorry

end total_gas_cost_l464_464410


namespace rational_root_polynomial_factoring_l464_464209

theorem rational_root_polynomial_factoring (p q : ℚ) (P : Polynomial ℤ) (h_irreducible : p.denom = q) (h_root : P.eval (p / q) = 0) :
  ∃ Q : Polynomial ℤ, P = Polynomial.C q * (Polynomial.X - Polynomial.C p) * Q :=
sorry

end rational_root_polynomial_factoring_l464_464209


namespace cube_root_5832_simplified_l464_464769

theorem cube_root_5832_simplified : 
  ∃ a b : ℕ, 0 < b ∧ 5832 = a^3 * b ∧ a + b = 19 :=
by {
  let a := 18;
  let b := 1;
  have h1 : 0 < b := by norm_num;
  have h2 : 5832 = a^3 * b := by norm_num;
  existsi [a, b],
  exact ⟨h1, h2, by norm_num⟩,
  sorry
}

end cube_root_5832_simplified_l464_464769


namespace circum_inradius_ratio_l464_464784

variable (R r : ℝ) (n : ℕ)

theorem circum_inradius_ratio (hR : R ≥ 0) (hr : r ≥ 0) (hn : n ≥ 3) :
    R / r ≥ Real.sec (Real.pi / n) ↔ is_regular_polygon n :=
sorry

end circum_inradius_ratio_l464_464784


namespace number_of_valid_integers_l464_464437

def has_two_odd_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).filter (λ x, x % 2 = 1).length = 2

def is_multiple_of_34 (n : ℕ) : Prop :=
  n % 34 = 0

def count_valid_numbers : ℕ :=
  (Finset.range (3400 + 1)).filter (λ n, is_multiple_of_34 n ∧ has_two_odd_divisors n).card

theorem number_of_valid_integers : count_valid_numbers = 6 :=
by
  sorry

end number_of_valid_integers_l464_464437


namespace problem1_problem2_l464_464933

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ -1 then (1/2)^x - 2 else (x - 2) * (|x| - 1)

theorem problem1 : f (f (-2)) = 0 := by 
  sorry

theorem problem2 (x : ℝ) (h : f x ≥ 2) : x ≥ 3 ∨ x = 0 := by
  sorry

end problem1_problem2_l464_464933


namespace common_ratio_of_increasing_geometric_sequence_l464_464919

theorem common_ratio_of_increasing_geometric_sequence 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_inc : ∀ n, a n < a (n + 1))
  (h_a2 : a 2 = 2)
  (h_a4_a3 : a 4 - a 3 = 4) : 
  q = 2 :=
by
  -- sorry - placeholder for proof
  sorry

end common_ratio_of_increasing_geometric_sequence_l464_464919


namespace geometric_seq_a5_value_l464_464068

theorem geometric_seq_a5_value 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n : ℕ, a (n+1) = a n * q)
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h1 : a 1 * a 8 = 4 * a 5)
  (h2 : (a 4 + 2 * a 6) / 2 = 18) 
  : a 5 = 16 := 
sorry

end geometric_seq_a5_value_l464_464068


namespace proof_problem_l464_464079

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 2 * x

def line_l_through_focus (m y : ℝ) : ℝ := m * y + 1/2

def focus : (ℝ × ℝ) := (1/2, 0)

def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

variable (A B : ℝ × ℝ)

theorem proof_problem :
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  (∃ m : ℝ, line_l_through_focus m A.2 = A.1 ∧ line_l_through_focus m B.2 = B.1) ∧
  dist A focus > dist B focus ∧
  (dist A B = 9) →
  ((dist A focus - dist B focus = 3) ∧
   (dist A focus * dist B focus = 9/4) ∧
   (dist A focus / dist B focus = 2 + sqrt(3))) :=
by
  sorry

end proof_problem_l464_464079


namespace LS_parallel_PQ_l464_464634

universe u
variables {Point : Type u} [Geometry Point]

-- Defining points A, P, C, M, N, Q as elements of type Point
variables (A P C M N Q : Point)

-- Let L be the intersection point of the lines AP and CM
def L := Geometry.intersection (Geometry.line_through A P) (Geometry.line_through C M)

-- Let S be the intersection point of the lines AN and CQ
def S := Geometry.intersection (Geometry.line_through A N) (Geometry.line_through C Q)

-- Proof: LS is parallel to PQ
theorem LS_parallel_PQ : Geometry.parallel (Geometry.line_through L S) (Geometry.line_through P Q) :=
sorry

end LS_parallel_PQ_l464_464634


namespace initial_gasoline_percentage_calculation_l464_464350

variable (initial_volume : ℝ)
variable (initial_ethanol_percentage : ℝ)
variable (additional_ethanol : ℝ)
variable (final_ethanol_percentage : ℝ)

theorem initial_gasoline_percentage_calculation
  (h1: initial_ethanol_percentage = 5)
  (h2: initial_volume = 45)
  (h3: additional_ethanol = 2.5)
  (h4: final_ethanol_percentage = 10) :
  100 - initial_ethanol_percentage = 95 :=
by
  sorry

end initial_gasoline_percentage_calculation_l464_464350


namespace dihedral_angle_trapezoid_l464_464833

/-- The problem statement, given the conditions, is to prove the dihedral angle measure for the given trapezoid setup. -/
theorem dihedral_angle_trapezoid 
  (AB : ℝ)
  (CD : ℝ)
  (height : ℝ)
  (h_AB : AB = 6)
  (h_CD : CD = 2)
  (h_height : height = sqrt 3)
  (A B C : ℝ × ℝ × ℝ)
  (O O₁ : ℝ × ℝ × ℝ)
  (h_A : A = (3, 0, 0))
  (h_B : B = (0, 3, 0))
  (h_O₁ : O₁ = (0, 0, sqrt 3))
  (h_C : C = (0, 1, sqrt 3)) :
  ∃ θ : ℝ, θ = arccos (sqrt 3 / 4) :=
sorry

end dihedral_angle_trapezoid_l464_464833


namespace chocolates_sold_at_selling_price_l464_464536
noncomputable def chocolates_sold (C S : ℝ) (n : ℕ) : Prop :=
  (35 * C = n * S) ∧ ((S - C) / C * 100) = 66.67

theorem chocolates_sold_at_selling_price : ∃ n : ℕ, ∀ C S : ℝ,
  chocolates_sold C S n → n = 21 :=
by
  sorry

end chocolates_sold_at_selling_price_l464_464536


namespace rectangle_segment_sum_l464_464242

theorem rectangle_segment_sum :
  let len_diagonal := Real.sqrt (6^2 + 5^2), 
      a_k := λ k : ℕ, len_diagonal * (200 - k) / 200,
      sum_segments := 2 * (∑ k in Finset.range 200 \ {0}, a_k k) - len_diagonal in
  sum_segments = 199 * Real.sqrt 61 :=
by
  let len_diagonal := Real.sqrt (6^2 + 5^2)
  let a_k := λ k : ℕ, len_diagonal * (200 - k) / 200
  let sum_segments := 2 * (∑ k in Finset.range 200 \ {0}, a_k k) - len_diagonal
  have h : len_diagonal = Real.sqrt 61 := by sorry
  have t : ∑ k in Finset.range 200 \ {0}, a_k k = (∑ k in Finset.range 199, Real.sqrt 61 * (199 - k) / 200) := by sorry
  have s : sum_segments = 199 * Real.sqrt 61 := by sorry
  exact s

end rectangle_segment_sum_l464_464242


namespace prob_reverse_order_prob_second_from_bottom_l464_464389

section CementBags

-- Define the basic setup for the bags and worker choices
variables (bags : list ℕ) (choices : list ℕ) (shed gate : list ℕ)

-- This ensures we are indeed considering 4 bags in a sequence
axiom bags_is_4 : bags.length = 4

-- This captures that the worker makes 4 sequential choices with equal probability for each, where 0 represents shed and 1 represents gate
axiom choices_is_4_and_prob : (choices.length = 4) ∧ (∀ i in choices, i ∈ [0, 1])

-- This condition captures that eventually, all bags must end up in the shed
axiom all_bags_in_shed : ∀ b ∈ bags, b ∈ shed

-- Prove for part (a): Probability of bags in reverse order in shed is 1/8
theorem prob_reverse_order : (choices = [0, 0, 0, 0] → (reverse bags = shed)) → 
  (probability (choices = [0, 0, 0, 0]) = 1 / 8) :=
 sorry

-- Prove for part (b): Probability of the second-from-bottom bag in the bottom is 1/8
theorem prob_second_from_bottom : 
  (choices = [1, 1, 0, 0] → (shed.head = bags.nth 1.get_or_else 0)) → 
  (probability (choices = [1, 1, 0, 0]) = 1 / 8) :=
 sorry

end CementBags

end prob_reverse_order_prob_second_from_bottom_l464_464389


namespace tan_angle_sum_identity_l464_464923

theorem tan_angle_sum_identity
  (θ : ℝ)
  (h1 : θ > π / 2 ∧ θ < π)
  (h2 : Real.cos θ = -3 / 5) :
  Real.tan (θ + π / 4) = -1 / 7 := by
  sorry

end tan_angle_sum_identity_l464_464923


namespace solve_for_m_monotonic_decreasing_l464_464932

noncomputable def f (m : ℝ) (x : ℝ) := 1 + m / x

theorem solve_for_m {m : ℝ} : f m 1 = 2 → m = 1 :=
by
  intros h,
  have h1 : f m 1 = 1 + m := by sorry, -- as per the condition f(x) = 1 + m/x
  rw h1 at h,
  exact sorry -- statement needs to show m = 1

theorem monotonic_decreasing (x₁ x₂ : ℝ) (h₀ : 0 < x₁) (h₁ : 0 < x₂) (h₂ : x₁ < x₂) :
  f 1 x₁ > f 1 x₂ :=
by 
  -- Proof steps skipped, just the statement
  sorry

end solve_for_m_monotonic_decreasing_l464_464932


namespace intersection_points_count_l464_464969

noncomputable def f : ℝ → ℝ
| x := if -1 ≤ x ∧ x < 1 then |x| else f (x - 2)

theorem intersection_points_count :
  (function.intersections f (λ x, real.logb 4 (|x|))).count = 6 :=
sorry

end intersection_points_count_l464_464969


namespace monotone_increasing_interval_l464_464492

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 3)

theorem monotone_increasing_interval :
  ∃ (a b : ℝ), -Real.pi / 12 = a ∧ 5 * Real.pi / 12 = b ∧ ∀ x : ℝ, a ≤ x ∧ x ≤ b → f x = 2 * Real.sin (2 * x - Real.pi / 3) 
sorry

end monotone_increasing_interval_l464_464492


namespace probability_reverse_order_probability_second_to_bottom_l464_464383

noncomputable def prob_reverse_order : ℚ := 1/8
noncomputable def prob_second_to_bottom : ℚ := 1/8

theorem probability_reverse_order (n: ℕ) (h : n = 4) 
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_reverse_order = 1/8 :=
by sorry

theorem probability_second_to_bottom (n: ℕ) (h : n = 4)
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_second_to_bottom = 1/8 :=
by sorry

end probability_reverse_order_probability_second_to_bottom_l464_464383


namespace parallel_ls_pq_l464_464612

variables {A P C M N Q L S : Type*}
variables [affine_space ℝ (A P C M N Q)]

/- Definitions of the intersection points -/
def is_intersection_point (L : Type*) (AP : set (A P))
  (CM : set (C M)) : Prop := L ∈ AP ∧ L ∈ CM

def is_intersection_point' (S : Type*) (AN : set (A N))
  (CQ : set (C Q)) : Prop := S ∈ AN ∧ S ∈ CQ

/- Main Theorem -/
theorem parallel_ls_pq (AP CM AN CQ : set (A P))
  (PQ : set (P Q)) (L S : Type*)
  (hL : is_intersection_point L AP CM)
  (hS : is_intersection_point' S AN CQ) : L S ∥ P Q :=
sorry

end parallel_ls_pq_l464_464612


namespace find_H_value_l464_464156

def distinct_digits (l : List ℕ) : Prop :=
  l.nodup ∧ l.all (λ x, 0 ≤ x ∧ x ≤ 9)

def main_problem (F I V T H R E G : ℕ) : Prop :=
  F = 9 ∧ odd I ∧ distinct_digits [F, I, V, T, H, R, E, G] ∧
  (F * 100 + I * 10 + V + T * 100 + H * 10 + R = E * 1000 + I * 100 + G * 10 + H) ∧
  H = 4

-- The final proof statement:
theorem find_H_value (F I V T H R E G : ℕ) :
  F = 9 →
  odd I →
  distinct_digits [F, I, V, T, H, R, E, G] →
  (F * 100 + I * 10 + V + T * 100 + H * 10 + R = E * 1000 + I * 100 + G * 10 + H) →
  H = 4 :=
by
  intros
  sorry

end find_H_value_l464_464156


namespace part1_part2_part3_l464_464203

open Real

-- Definition of the first problem statement
theorem part1 (x : ℝ) : 
  (x^5 - 1) / (x - 1) = x^4 + x^3 + x^2 + x + 1 :=
by sorry

-- Definition of the second problem statement with n as a positive integer ≥ 2
theorem part2 (x : ℝ) (n : ℕ) (h : 2 ≤ n) : 
  (x^n - 1) / (x - 1) = ∑ i in finset.range n, x^i :=
by sorry

-- Definition of the third problem statement 
theorem part3 : 
  (∑ i in finset.range (2023), 2^i) = 2^2023 - 1 :=
by sorry

end part1_part2_part3_l464_464203


namespace point_on_x_axis_m_eq_2_l464_464699

theorem point_on_x_axis_m_eq_2 (m : ℝ) (h : (m + 5, m - 2).2 = 0) : m = 2 :=
sorry

end point_on_x_axis_m_eq_2_l464_464699


namespace expected_weekly_rainfall_l464_464823

noncomputable def daily_expected_rainfall : ℝ :=
  (0.30 * 0) + (0.40 * 5) + (0.30 * 12)

noncomputable def weekly_expected_rainfall : ℝ :=
  seven_days * daily_expected_rainfall

theorem expected_weekly_rainfall :
  weekly_expected_rainfall = 39.2 :=
by
  sorry

end expected_weekly_rainfall_l464_464823


namespace combination_6_3_l464_464811

theorem combination_6_3 : nat.choose 6 3 = 20 := by
  sorry

end combination_6_3_l464_464811


namespace gum_sharing_l464_464167

theorem gum_sharing (john cole aubrey : ℕ) (sharing_people : ℕ) 
  (hj : john = 54) (hc : cole = 45) (ha : aubrey = 0) 
  (hs : sharing_people = 3) : 
  john + cole + aubrey = 99 ∧ (john + cole + aubrey) / sharing_people = 33 := 
by
  sorry

end gum_sharing_l464_464167


namespace angle_DHQ_right_l464_464049

-- Define the geometrical setup
variables {A B C D P Q H : Type}
variables [square A B C D]
variables [on_side P A B]
variables [on_side Q B C]
variables [foot_perpendicular H B P C]
variables (BP_BQ : dist B P = dist B Q)

-- The theorem to be proven
theorem angle_DHQ_right : angle D H Q = pi / 2 := by
  sorry

end angle_DHQ_right_l464_464049


namespace log_point_through_A_l464_464938

noncomputable theory

variables {a : ℝ} {b : ℝ}

def f (x : ℝ) := 3^x + b

theorem log_point_through_A
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (A : ℝ × ℝ)
  (hA1 : A = (-1, -1))
  (hA2 : ∀ x, A = (x, log a (x + 2) - 1))
  (hA3 : ∀ x, A = (x, 3^x + b)) :
  f (log 3 2) = 2 / 3 :=
by {
  sorry
}

end log_point_through_A_l464_464938


namespace intersection_of_solution_sets_solution_set_of_modified_inequality_l464_464505

open Set Real

theorem intersection_of_solution_sets :
  let A := {x | x ^ 2 - 2 * x - 3 < 0}
  let B := {x | x ^ 2 + x - 6 < 0}
  A ∩ B = {x | -1 < x ∧ x < 2} := by {
  sorry
}

theorem solution_set_of_modified_inequality :
  let A := {x | x ^ 2 + (-1) * x + (-2) < 0}
  A = {x | true} := by {
  sorry
}

end intersection_of_solution_sets_solution_set_of_modified_inequality_l464_464505


namespace burger_cost_l464_464760

theorem burger_cost (b s : ℕ) (h1 : 3 * b + 2 * s = 385) (h2 : 2 * b + 3 * s = 360) : b = 87 :=
sorry

end burger_cost_l464_464760


namespace complex_division_l464_464113

theorem complex_division (i : ℂ) (h : i^2 = -1) : (2 + i) / (1 - 2 * i) = i := 
by
  sorry

end complex_division_l464_464113


namespace negation_prop_l464_464280

open Classical

variable (x : ℝ)

theorem negation_prop :
    (∃ x : ℝ, x^2 + 2*x + 2 < 0) = False ↔
    (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by
    sorry

end negation_prop_l464_464280


namespace probability_reverse_order_probability_second_to_bottom_l464_464382

noncomputable def prob_reverse_order : ℚ := 1/8
noncomputable def prob_second_to_bottom : ℚ := 1/8

theorem probability_reverse_order (n: ℕ) (h : n = 4) 
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_reverse_order = 1/8 :=
by sorry

theorem probability_second_to_bottom (n: ℕ) (h : n = 4)
  (prob_truck_to_gate_or_shed : ∀ i, i < n → ℚ) :
  prob_truck_to_gate_or_shed 0 = 1/2 ∧
  prob_truck_to_gate_or_shed 1 = 1/2 ∧
  prob_truck_to_gate_or_shed 2 = 1/2 →
  prob_second_to_bottom = 1/8 :=
by sorry

end probability_reverse_order_probability_second_to_bottom_l464_464382


namespace number_of_possible_values_l464_464706
-- Import the necessary library to ensure all definitions are available.

-- Define a structure to capture the problem's conditions.
structure ResistorProblem where
  initial_resistors : List ℕ      -- List representing the resistors, each initially has resistance of 1 ohm
  combine_series : ℕ → ℕ → ℕ    -- Function to combine two resistors in series
  combine_parallel : ℕ → ℕ → ℕ  -- Function to combine two resistors in parallel
  short_circuit : ℕ → ℕ → ℕ    -- Function to short-circuit one of the two resistors

-- Instantiate the problem with the given conditions.
def rebecca_resistor_problem : ResistorProblem :=
  { initial_resistors := List.replicate 24 1
  , combine_series := λ a b, a + b
  , combine_parallel := λ a b, a * b / (a + b)
  , short_circuit := λ a b, a  -- (This captures the essence but ignores the choice aspect for simplicity)
  }

-- State the theorem which is equivalent to the given problem.
theorem number_of_possible_values (r_problem : ResistorProblem) (total_minutes : ℕ)
  (resistance_after_combination : ℕ) : 
  total_minutes = 23 ∧ resistance_after_combination = 1 -> 
  ∃ R : ℤ, (total_minutes = 23 ∧ resistance_after_combination = 1) ∧ 
  R = 1015080877 := 
by 
  sorry

end number_of_possible_values_l464_464706


namespace find_AC_l464_464038

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem find_AC :
  let AB := (2, 3)
  let BC := (1, -4)
  vector_add AB BC = (3, -1) :=
by 
  sorry

end find_AC_l464_464038


namespace max_value_of_expression_l464_464473

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 3) : 
  2 * a * b + 3 * c ≤ 21 / 4 :=
begin
  -- Proof omitted
  sorry
end

end max_value_of_expression_l464_464473


namespace convert_to_rectangular_form_l464_464861

theorem convert_to_rectangular_form :
  (Complex.exp (13 * Real.pi * Complex.I / 2)) = Complex.I :=
by
  sorry

end convert_to_rectangular_form_l464_464861


namespace real_solution_for_any_y_l464_464873

theorem real_solution_for_any_y (x : ℝ) :
  (∀ y z : ℝ, x^2 + y^2 + z^2 + 2 * x * y * z = 1 → ∃ z : ℝ,  x^2 + y^2 + z^2 + 2 * x * y * z = 1) ↔ (x = 1 ∨ x = -1) :=
by sorry

end real_solution_for_any_y_l464_464873


namespace linear_function_increasing_l464_464144

variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)
variable (hx : x1 < x2)
variable (P1_eq : y1 = 2 * x1 + 1)
variable (P2_eq : y2 = 2 * x2 + 1)

theorem linear_function_increasing (hx : x1 < x2) (P1_eq : y1 = 2 * x1 + 1) (P2_eq : y2 = 2 * x2 + 1) 
    : y1 < y2 := sorry

end linear_function_increasing_l464_464144


namespace function_properties_l464_464073

def f (x : ℝ) (a : ℝ) : ℝ := 1 / (2^x + 1) + a

theorem function_properties (a : ℝ) :
  (∃ a' ∈ ℝ, ∀ x : ℝ, f x a' = -f (-x) a')
  ∧ ¬ (∀ x : ℝ, f x a = f (-x) a)
  ∧ (∀ x : ℝ, f x a + f (-x) a = 1 + 2 * a)
  ∧ (∀ x : ℝ, f x a - f (-x) a > f (x + 1) a - f (- (x + 1)) a) :=
sorry

end function_properties_l464_464073


namespace janet_earnings_eur_l464_464572

noncomputable def usd_to_eur (usd : ℚ) : ℚ :=
  usd * 0.85

def janet_earnings_usd : ℚ :=
  (130 * 0.25) + (90 * 0.30) + (30 * 0.40)

theorem janet_earnings_eur : usd_to_eur janet_earnings_usd = 60.78 :=
  by
    sorry

end janet_earnings_eur_l464_464572


namespace cyclic_quadrilateral_area_l464_464562

open EuclideanGeometry

/-- Given an inscribed quadrilateral ABCD with BC = CD, prove that the area of ABCD is 
    (1/2) * (AC ^ 2) * sin(∠A). -/
theorem cyclic_quadrilateral_area {A B C D : Point} (h1 : CyclicQuad A B C D) (h2 : dist B C = dist C D) :
  area A B C D = (1 / 2) * (dist A C) ^ 2 * Real.sin (angle A) := 
sorry

end cyclic_quadrilateral_area_l464_464562


namespace max_m_chords_in_k_consecutive_points_l464_464783

theorem max_m_chords_in_k_consecutive_points
  (n k : ℕ)
  (n_even : Even n)
  (k_ge_2 : k ≥ 2)
  (n_gt_4k : n > 4 * k)
  (matching_exists : ∃ (chords : Finset (Fin n × Fin n)), 
                     ∀ (c ∈ chords), (c.1 ≠ c.2) ∧ 
                     (∀ (c₁ c₂ : (Fin n × Fin n)), c₁ ∈ chords → c₂ ∈ chords → c₁ ≠ c₂ → ¬intersecting c₁ c₂) ∧
                     (chords.card = n / 2)) : 
  ∃ m : ℕ, m = Nat.ceil (k/3 : ℚ) ∧
  ∀ (chords : Finset (Fin n × Fin n)) (_ : ∀ (c ∈ chords), (c.1 ≠ c.2) ∧ 
                     (∀ (c₁ c₂ : (Fin n × Fin n)), c₁ ∈ chords → c₂ ∈ chords → c₁ ≠ c₂ → ¬intersecting c₁ c₂) ∧
                     (chords.card = n / 2)),
  ∃ (consec_points : Finset (Fin n)), (consec_points.card = k) ∧
                                        (count_chords_within_k_points chords consec_points ≥ m) := by
  sorry

-- To define helper predicates/functions if needed
def intersecting (c1 c2 : Fin n × Fin n) : Prop :=
  sorry

def count_chords_within_k_points (chords : Finset (Fin n × Fin n)) (points : Finset (Fin n)) : ℕ :=
  sorry

end max_m_chords_in_k_consecutive_points_l464_464783


namespace infinite_product_e_l464_464191

noncomputable def a : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * (a n + 1)

theorem infinite_product_e : 
  (∏ n in (range ∞), (1 + 1 / (a n))) = real.exp 1 :=
sorry

end infinite_product_e_l464_464191


namespace identify_longest_segment_l464_464732

-- Define the properties and angles in triangles ABD and BCD
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables {AB AD BD BC CD : ℝ}
variables {angle_BAD angle_ABD angle_ADB angle_BCD angle_CBD angle_BDC: ℝ}

-- Given angles in triangles ABD and BCD
def angles_ABD_triangle : Prop :=
  angle_ABD = 50 ∧ angle_ADB = 45 ∧ angle_BAD = 180 - angle_ABD - angle_ADB

def angles_BCD_triangle : Prop :=
  angle_CBD = 70 ∧ angle_BDC = 65 ∧ angle_BCD = 180 - angle_CBD - angle_BDC

-- Conditions for the longest segment
def longest_segment (AD AB BD BC CD : ℝ) : Prop :=
  AD < AB ∧ AB < BC ∧ BC < CD ∧ CD < BD

theorem identify_longest_segment
  (h1: angles_ABD_triangle)
  (h2: angles_BCD_triangle)
  (h3: longest_segment AD AB BD BC CD) :
  BD = max (max (max (max AD AB) BC) CD) :=
sorry

end identify_longest_segment_l464_464732


namespace perpendicular_lines_m_eq_parallel_lines_distance_l464_464940

-- Definitions of the lines l1 and l2
def line1 (m : ℝ) : ℝ × ℝ → Prop :=
  λ p, p.1 + (m - 3) * p.2 + m = 0

def line2 (m : ℝ) : ℝ × ℝ → Prop :=
  λ p, m * p.1 - 2 * p.2 + 4 = 0

-- Part 1: Proving the value of m when lines are perpendicular
theorem perpendicular_lines_m_eq (m : ℝ) :
  (∃ (p1 p2 : ℝ × ℝ), line1 m p1 ∧ line2 m p2 ∧
   (-1 / (m - 3)) * (m / 2) = -1) → m = 6 := 
sorry

-- Part 2: Proving the distance between lines when they are parallel
theorem parallel_lines_distance (m : ℝ) (p1 p2 : ℝ × ℝ) :
  (m = 1 ∧ line1 m p1 ∧ line2 m p2 ∧
   (1 / (m - 3) = 2 / m)) →
  let d := abs (4 - 1) / real.sqrt ((1)^2 + (-2)^2)
  in d = 3 * real.sqrt 5 / 5 :=
sorry

end perpendicular_lines_m_eq_parallel_lines_distance_l464_464940


namespace polynomial_int_coeff_property_l464_464009

open Polynomial

theorem polynomial_int_coeff_property 
  (P : Polynomial ℤ) : 
  (∀ s t : ℝ, P.eval s ∈ ℤ ∧ P.eval t ∈ ℤ → P.eval (s * t) ∈ ℤ) ↔ 
  ∃ (n : ℕ) (k : ℤ), P = X^n + C k ∨ P = -X^n + C k :=
by
  sorry

end polynomial_int_coeff_property_l464_464009


namespace angle_in_third_quadrant_l464_464459

theorem angle_in_third_quadrant (α : ℝ) (h : α = -89 * Real.pi / 6) : 
(∃ k : ℤ, -Real.pi + k * 2 * Real.pi ≤ α + 89 * Real.pi / 6 ∧ α + 89 * Real.pi / 6 ≤ 0) -> 
(4 * k * Real.pi / 2 + Real.pi / 2 ≤ α - 2 * k * Real.pi / 2 + α + 89 * Real.pi / 6 ∧ α - 2 * k * Real.pi / 2 + α + 89 * Real.pi / 6 ≤ 0): 
(∃ n : ℤ, n = 3 * Real.pi) ∧ (-16 * Real.pi + -89 * Real.pi / 6 + 2 * k 2 * Real.pi)↔(0 ≤ α ∧ α < 3 *Real.pi / 2)  ∧ -Real.pi +  ((2 / 6 * Real.pi) - α =  Real.pi ) :=
sorry

end angle_in_third_quadrant_l464_464459


namespace regular_octagon_side_length_sum_l464_464510

theorem regular_octagon_side_length_sum (s : ℝ) (h₁ : s = 2.3) (h₂ : 1 = 100) : 
  8 * (s * 100) = 1840 :=
by
  sorry

end regular_octagon_side_length_sum_l464_464510


namespace find_a_l464_464488

noncomputable def parametric_curve (θ : ℝ) : ℝ × ℝ :=
(2 * Real.cos θ, Real.sin θ)

noncomputable def distance_squared (a θ : ℝ) : ℝ :=
let p := parametric_curve θ
in (p.1 - a)^2 + p.2^2

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : ∃ θ ∈ Icc (0 : ℝ) (2 * Real.pi), distance_squared a θ = 9 / 16) :
a = Real.sqrt 21 / 4 ∨ a = 11 / 4 :=
sorry

end find_a_l464_464488


namespace sum_arithmetic_series_85_to_100_l464_464768

theorem sum_arithmetic_series_85_to_100 : 
  let a := 85 in
  let l := 100 in
  let n := l - a + 1 in
  let S := n / 2 * (a + l) in
  S = 1480 :=
by
  let a := 85
  let l := 100
  let n := l - a + 1
  let S := n / 2 * (a + l)
  calc S = 8 * 185 : sorry
    ... = 1480 : sorry

end sum_arithmetic_series_85_to_100_l464_464768


namespace probability_point_on_hyperbola_l464_464215

-- Define the problem conditions
def number_set := {1, 2, 3}
def point_on_hyperbola (x y : ℝ) : Prop := y = 6 / x

-- Formalize the problem statement
theorem probability_point_on_hyperbola :
  let combinations := ({(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} : set (ℝ × ℝ)) in
  let on_hyperbola := set.filter (λ p : ℝ × ℝ, point_on_hyperbola p.1 p.2) combinations in
  fintype.card on_hyperbola / fintype.card combinations = 1 / 3 :=
by sorry

end probability_point_on_hyperbola_l464_464215


namespace count_multiples_of_34_with_two_odd_divisors_l464_464432

-- Define a predicate to check if a number has exactly 2 odd natural divisors
def has_exactly_two_odd_divisors (n : ℕ) : Prop :=
  (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ d1 * d2 = n ∧ d1 % 2 = 1 ∧ d2 % 2 = 1) ∧
  (∀ d : ℕ, d ∣ n → (d % 2 = 0 ∨ d = d1 ∨ d = d2))

-- Main theorem to prove the number of integers that satisfy the given conditions
theorem count_multiples_of_34_with_two_odd_divisors : 
  let valid_numbers := {n : ℕ | n ≤ 3400 ∧ n % 34 = 0 ∧ has_exactly_two_odd_divisors n} in
  valid_numbers.to_finset.card = 6 :=
by
  sorry

end count_multiples_of_34_with_two_odd_divisors_l464_464432


namespace marked_points_rational_l464_464342

theorem marked_points_rational {n : ℕ} (x : Fin (n + 2) → ℝ) :
  x 0 = 0 ∧ x (Fin.last (n + 1)) = 1 ∧
  (∀ i : Fin (n + 2), ∃ a b : Fin (n + 2),
    (x i = (x a + x b) / 2 ∧ x a < x i ∧ x i < x b) ∨
    (x i = (x a + 0) / 2 ∧ x a = x 0 ∨ (x i = (x b + 1) / 2 ∧ x b = 1))) →
  (∀ i : Fin (n + 2), ∃ q : ℚ, x i = q) :=
begin
  sorry
end

end marked_points_rational_l464_464342


namespace ali_less_nada_l464_464825

variable (Ali Nada John : ℕ)

theorem ali_less_nada
  (h_total : Ali + Nada + John = 67)
  (h_john_nada : John = 4 * Nada)
  (h_john : John = 48) :
  Nada - Ali = 5 :=
by
  sorry

end ali_less_nada_l464_464825


namespace sum_A_eq_sum_B_l464_464444

-- Definition of a partition of n
def is_partition (π : List ℕ) (n : ℕ) : Prop :=
  π.sum = n ∧ π.all (λ x, x > 0) ∧ List.sorted (≤) π

-- Function A that counts the number of times 1 appears in the partition
def A (π : List ℕ) : ℕ :=
  π.count 1

-- Function B that counts the number of distinct elements in the partition
def B (π : List ℕ) : ℕ :=
  (π.eraseDuplicates).length

-- The set of all partitions of n
def all_partitions (n : ℕ) : List (List ℕ) :=
  -- This would be implemented to generate all partitions, but left abstract here
  sorry

-- The theorem stating the required equality
theorem sum_A_eq_sum_B (n : ℕ) (h : n > 0) : 
  (∑ π in all_partitions n, A π) = (∑ π in all_partitions n, B π) :=
sorry

end sum_A_eq_sum_B_l464_464444


namespace train_length_is_500_meters_l464_464369

noncomputable def train_speed_km_hr : ℝ := 120
noncomputable def time_to_cross_pole_sec : ℝ := 15

noncomputable def km_hr_to_m_s (speed_km_hr : ℝ) : ℝ := speed_km_hr * 1000 / 3600

noncomputable def train_length_in_meters : ℝ :=
  km_hr_to_m_s train_speed_km_hr * time_to_cross_pole_sec

theorem train_length_is_500_meters : train_length_in_meters ≈ 500 :=
by
  sorry

end train_length_is_500_meters_l464_464369


namespace count_3digit_multiples_of_30_not_75_l464_464094

-- Define the conditions
def is_positive_3_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000
def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

-- Define the main theorem
theorem count_3digit_multiples_of_30_not_75 : 
  { n : ℕ // is_positive_3_digit n ∧ is_multiple_of n 30 ∧ ¬ is_multiple_of n 75 }.to_finset.card = 24 :=
by
  sorry

end count_3digit_multiples_of_30_not_75_l464_464094


namespace second_intersections_on_circle_l464_464910

variables {A B C D A1 B1 C1 D1 : Point}

-- Definitions for points and conditions
def cyclic_points (A B C D : Point) : Prop := ∃ (O : Circle), 
  A ∈ O ∧ B ∈ O ∧ C ∈ O ∧ D ∈ O

def second_intersection (A B C D A1 B1 C1 D1 : Point) (O1 O2 O3 O4 : Circle) : Prop :=
  (A ∈ O1 ∧ B ∈ O1 ∧ A1 ∈ O1 ∧ 
   B ∈ O2 ∧ C ∈ O2 ∧ B1 ∈ O2 ∧
   C ∈ O3 ∧ D ∈ O3 ∧ C1 ∈ O3 ∧
   D ∈ O4 ∧ A ∈ O4 ∧ D1 ∈ O4)

-- Lean 4 theorem statement
theorem second_intersections_on_circle 
  (A B C D A1 B1 C1 D1 : Point)
  (h1 : cyclic_points A B C D)
  (h2 : second_intersection A B C D A1 B1 C1 D1 O1 O2 O3 O4) : 
  ∃ (O5 : Circle), A1 ∈ O5 ∧ B1 ∈ O5 ∧ C1 ∈ O5 ∧ D1 ∈ O5 :=
sorry

end second_intersections_on_circle_l464_464910


namespace proof_equiv_l464_464453

theorem proof_equiv
  (x : ℝ)
  (h : x^(1/2) + x^(-1/2) = 3) :
  (x^(3/2) + x^(-3/2) - 3) / (x^2 + x^(-2) - 2) = 1 / 3 :=
by sorry

end proof_equiv_l464_464453


namespace proof_problem_l464_464162

noncomputable def problem_equivalent_proof (x y z : ℝ) : Prop :=
  (x + y + z = 3) ∧
  (z + 6 = 2 * y - z) ∧
  (x + 8 * z = y + 2) →
  (x^2 + y^2 + z^2 = 21)

theorem proof_problem (x y z : ℝ) : problem_equivalent_proof x y z :=
by
  sorry

end proof_problem_l464_464162


namespace probability_one_from_harold_and_one_from_marilyn_l464_464087

-- Define the names and the number of letters in each name
def harold_name_length := 6
def marilyn_name_length := 7

-- Total cards
def total_cards := harold_name_length + marilyn_name_length

-- Probability of drawing one card from Harold's name and one from Marilyn's name
theorem probability_one_from_harold_and_one_from_marilyn :
    (harold_name_length : ℚ) / total_cards * marilyn_name_length / (total_cards - 1) +
    marilyn_name_length / total_cards * harold_name_length / (total_cards - 1) 
    = 7 / 13 := 
by
  sorry

end probability_one_from_harold_and_one_from_marilyn_l464_464087


namespace count_elements_in_set_l464_464176

noncomputable def floor_sum (x : ℝ) : ℤ :=
  Int.floor x + Int.floor (2 * x) + Int.floor (3 * x)

theorem count_elements_in_set : 
  (finset.range 100).filter (λ n, ∃ x : ℝ, floor_sum x = n + 1).card = 67 :=
  sorry

end count_elements_in_set_l464_464176


namespace average_marks_class_l464_464984

theorem average_marks_class (total_students : ℕ)
  (students_98 : ℕ) (score_98 : ℕ)
  (students_0 : ℕ) (score_0 : ℕ)
  (remaining_avg : ℝ)
  (h1 : total_students = 40)
  (h2 : students_98 = 6)
  (h3 : score_98 = 98)
  (h4 : students_0 = 9)
  (h5 : score_0 = 0)
  (h6 : remaining_avg = 57) :
  ( (( students_98 * score_98) + (students_0 * score_0) + ((total_students - students_98 - students_0) * remaining_avg)) / total_students ) = 50.325 :=
by 
  -- This is where the proof steps would go
  sorry

end average_marks_class_l464_464984


namespace num_real_solutions_system_l464_464869

theorem num_real_solutions_system :
  ∃! (num_solutions : ℕ), 
  num_solutions = 5 ∧
  ∃ x y z w : ℝ, 
    (x = z + w + x * z) ∧ 
    (y = w + x + y * w) ∧ 
    (z = x + y + z * x) ∧ 
    (w = y + z + w * z) :=
sorry

end num_real_solutions_system_l464_464869


namespace num_integers_with_properties_l464_464433

theorem num_integers_with_properties : 
  ∃ (count : ℕ), count = 6 ∧
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 3400 ∧ (n % 34 = 0) ∧ 
             ((∃ (odd_divisors : ℕ), odd_divisors = (filter (λ d, d % 2 = 1) (n.divisors)) ∧ odd_divisors.length = 2) →
             (count = 6)) :=
begin
  sorry
end

end num_integers_with_properties_l464_464433


namespace log_inequality_implies_order_l464_464126

theorem log_inequality_implies_order (a b : ℝ) (h : log a 3 > log b 3 ∧ log b 3 > 0) : b > a ∧ a > 1 ∧ b > 1 :=
by
  sorry

end log_inequality_implies_order_l464_464126


namespace parallel_LS_pQ_l464_464623

universe u

variables {P Q A C M N L S T : Type u} [incidence_geometry P Q A C M N L S T] -- Customize as per specific needs in Lean

-- Assume the required points and their intersections
def intersection_L (A P C M : Type u) [incidence_geometry A P C M] : Type u := sorry
def intersection_S (A N C Q : Type u) [incidence_geometry A N C Q] : Type u := sorry

-- Defining the variables as intersections
variables (L : intersection_L A P C M)
variables (S : intersection_S A N C Q)

theorem parallel_LS_pQ (hL : intersection_L A P C M) (hS : intersection_S A N C Q) :
  parallel L S P Q :=
sorry

end parallel_LS_pQ_l464_464623


namespace sum_of_center_coordinates_l464_464426

theorem sum_of_center_coordinates (x y : ℝ) 
  (h : x^2 + y^2 = 10 * x - 8 * y + 4) : 
  let center := (5 : ℝ, -4 : ℝ) in 
  (center.1 + center.2) = 1 := 
by
  sorry

end sum_of_center_coordinates_l464_464426


namespace range_of_a_subset_B_range_of_a_disjoint_B_l464_464949

-- Defining the necessary conditions and the domain of the function f(x)
def f (x : ℝ) : ℝ := real.sqrt (4 - x) + real.log (3^x - 9)

def A : set ℝ := {x | 2 < x ∧ x ≤ 4}
def B (a : ℝ) : set ℝ := {x | (x - a) * (x - (a + 3)) < 0}

-- Part (1) A ⊆ B, find the range of a
theorem range_of_a_subset_B : ∀ a : ℝ, (A ⊆ B a) ↔ (1 < a ∧ a ≤ 2) :=
by sorry

-- Part (2) A ∩ B = ∅, find the range of a
theorem range_of_a_disjoint_B : ∀ a : ℝ, (A ∩ B a = ∅) ↔ (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end range_of_a_subset_B_range_of_a_disjoint_B_l464_464949


namespace probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464400

-- Definitions based on conditions
def bags_in_truck : ℕ := 4

-- Assume random_choice has probability 0.5 for selection
def random_choice : ℕ → ℕ → ℝ := λ a b, 0.5

-- Bag positions and movements
def initial_position (i : ℕ) : ℕ := i

def final_position (i : ℕ) : ℕ := 4 - i

-- Events
def end_up_in_reverse_order (n : ℕ) : Prop := 
  ∀ i < n, final_position i = n - 1 - i

def second_from_bottom_ends_up_bottom (n : ℕ) : Prop :=
  final_position 1 = 0

-- Probabilities calculation
noncomputable def probability_of_reverse_order (n : ℕ) : ℝ := (1/2)^(n - 1)
noncomputable def probability_of_second_from_bottom_as_bottom (n : ℕ) : ℝ := (1/2)^(n - 1)

-- Theorem to prove the statement
theorem probability_reverse_order :
  probability_of_reverse_order bags_in_truck = 1 / 8 :=
by sorry

theorem probability_second_from_bottom_as_bottom :
  probability_of_second_from_bottom_as_bottom bags_in_truck = 1 / 8 :=
by sorry

end probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464400


namespace max_y_when_a_eq_1_range_of_k_when_a_eq_neg_1_range_of_a_for_inequality_l464_464497

noncomputable def f (x a : ℝ) : ℝ := x^2 + a*x + 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem max_y_when_a_eq_1 :
  ∃ x ∈ Set.Icc (-2:ℝ) (0:ℝ), 
  (∀ y ∈ Set.Icc (-2:ℝ) (0:ℝ), f y 1 * g y ≤ f x 1 * g x) ∧ f x 1 * g x = 1 := sorry

theorem range_of_k_when_a_eq_neg_1 :
  ∀ k : ℝ, (∃ x, f x (-1) = k * g x) ↔ (k > 3 / Real.exp 2 ∨ (0 < k ∧ k < 1 / Real.exp 1)) := sorry

theorem range_of_a_for_inequality :
  ∀ a : ℝ, (∀ x1 x2 ∈ Set.Icc (0:ℝ) (2:ℝ), x1 ≠ x2 → 
    abs (f x1 a - f x2 a) < abs (g x1 - g x2)) ↔ -1 ≤ a ∧ a ≤ 2 - 2 * Real.log 2 := sorry

end max_y_when_a_eq_1_range_of_k_when_a_eq_neg_1_range_of_a_for_inequality_l464_464497


namespace optimal_position_station_l464_464693

-- Definitions for the conditions
def num_buildings := 5
def building_workers (k : ℕ) : ℕ := if k ≤ 5 then k else 0
def distance_between_buildings := 50

-- Function to calculate the total walking distance
noncomputable def total_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

-- Theorem statement
theorem optimal_position_station :
  ∃ x : ℝ, (∀ y : ℝ, total_distance x ≤ total_distance y) ∧ x = 150 :=
by
  sorry

end optimal_position_station_l464_464693


namespace overall_average_length_l464_464749

theorem overall_average_length
    (N : ℕ)
    (hN : N = 6)
    (average_length_third : ℕ -> ℤ)
    (h_average_length_third : average_length_third 2 = 70)
    (average_length_rest : ℕ -> ℤ)
    (h_average_length_rest : average_length_rest 4 = 85) :
    (140 + 340) / 6 = 80 :=
by
  have h1 : (140 + 340 : ℤ) = 480 := rfl
  have h2 : (480 : ℤ) / 6 = 80 := by norm_num
  exact h2

end overall_average_length_l464_464749


namespace perimeter_of_adjacent_rectangles_l464_464807

theorem perimeter_of_adjacent_rectangles (s : ℕ) (h : s = 100) :
  let width := s,
      height := s / 2 in
  3 * width + 4 * height = 500 :=
by
  have width := s
  have height := s / 2
  rw [h]
  unfold width height
  sorry

end perimeter_of_adjacent_rectangles_l464_464807


namespace total_time_is_correct_l464_464705

-- Defining the number of items
def chairs : ℕ := 7
def tables : ℕ := 3
def bookshelves : ℕ := 2
def lamps : ℕ := 4

-- Defining the time spent on each type of furniture
def time_per_chair : ℕ := 4
def time_per_table : ℕ := 8
def time_per_bookshelf : ℕ := 12
def time_per_lamp : ℕ := 2

-- Defining the total time calculation
def total_time : ℕ :=
  (chairs * time_per_chair) + 
  (tables * time_per_table) +
  (bookshelves * time_per_bookshelf) +
  (lamps * time_per_lamp)

-- Theorem stating the total time
theorem total_time_is_correct : total_time = 84 :=
by
  -- Skipping the proof details
  sorry

end total_time_is_correct_l464_464705


namespace magnitude_of_complex_number_l464_464420

theorem magnitude_of_complex_number : 
  abs (3/4 + (5/12 : ℂ) * complex.i) = real.sqrt 106 / 12 := 
by
  sorry

end magnitude_of_complex_number_l464_464420


namespace sum_of_common_ratios_l464_464678

theorem sum_of_common_ratios (k p r : ℝ) (h₁ : k ≠ 0) (h₂ : p ≠ r) (h₃ : (k * (p ^ 2)) - (k * (r ^ 2)) = 4 * (k * p - k * r)) : 
  p + r = 4 :=
by
  -- Using the conditions provided, we can prove the sum of the common ratios is 4.
  sorry

end sum_of_common_ratios_l464_464678


namespace ellipse_eq_find_k_l464_464924

-- Given conditions:
def focus_parabola := (sqrt 3, 0)
def point_on_ellipse := (sqrt 3, 1 / 2)

-- a) Proving the equation of the ellipse C
theorem ellipse_eq (a b : ℝ) (ha : a > b) (hb : b > 0) 
  (hfocus : focus_parabola = (sqrt 3, 0)) 
  (hpoint : point_on_ellipse = (sqrt 3, 1 / 2)) 
  (hellipse : ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1) :
  (a^2 = 4 ∧ b^2 = 1 ∧ ∀ x y, (x^2 / 4) + y^2 = 1) :=
sorry

-- b) Proving the possible values of k
theorem find_k (k m : ℝ) (hk : k ≠ 0) 
  (PQ_length : ∃ P Q : ℚ × ℚ, (P ≠ Q ∧ dist P Q = (2 * sqrt 6) / 3))
  (vertex_rhombus : ∃ (R : ℚ × ℚ), R = (-1, 0) ∧ 
    ∀ x y, (x^2 / 4) + y^2 = 1 → dist (midpoint P Q) R = (1 + (P.1 + Q.1)^2 / 4 + (P.2 + Q.2)^2 / 4)) :
  (k = sqrt 2 ∨ k = -sqrt 2 ∨ k = sqrt 2 / 2 ∨ k = -sqrt 2 / 2) :=
sorry

end ellipse_eq_find_k_l464_464924


namespace thirteen_pow_2011_mod_100_l464_464766

theorem thirteen_pow_2011_mod_100 : (13^2011) % 100 = 37 := by
  sorry

end thirteen_pow_2011_mod_100_l464_464766


namespace largest_prime_factor_989_largest_prime_factor_989_is_43_l464_464324

-- Define the largest prime factor of a number using Lean
theorem largest_prime_factor_989 (p : ℕ) (hp : Nat.Prime p) (pf : p ∣ 989) : p ≤ 43 := sorry

-- Theorem that states 43 is the largest prime factor of 989
theorem largest_prime_factor_989_is_43 : ∃ p, Nat.Prime p ∧ p ∣ 989 ∧ ∀ q, Nat.Prime q ∧ q ∣ 989 → q ≤ p := 
begin
  use 43,
  repeat { split },
  { exact Nat.Prime.intro (show (43 = 43)), sorry }, -- Proof that 43 is prime
  { exact dvd_refl 989 },                            -- Proof that 43 divides 989
  { intros q hq1 hq2,                                -- Proof that 43 is the largest
    exact largest_prime_factor_989 q hq1 hq2 },
end

end largest_prime_factor_989_largest_prime_factor_989_is_43_l464_464324


namespace rectangular_equation_distance_l464_464485

-- Definition of the polar equation and conditions
def polar_curve (ρ θ : ℝ) : Prop := ρ = 4 * real.cos θ

-- Definition of the parameterized line
def parametric_line (t : ℝ) : ℝ × ℝ := (t + 1, t)

-- The main theorem to prove
theorem rectangular_equation_distance :
  (∀ ρ θ, polar_curve ρ θ → (∀ x y, (x = ρ * real.cos θ) ∧ (y = ρ * real.sin θ) → (x^2 + y^2 = 4 * x))) ∧
  (∀ t, let A := parametric_line t, B := parametric_line (t + 1) in
           let d := abs (2 - 1) / real.sqrt 2 in
           (2 * real.sqrt (4 - (d^2))) = real.sqrt 14) :=
  sorry

end rectangular_equation_distance_l464_464485


namespace triple_sum_of_divisors_of_8_eq_0_l464_464845

def sum_of_divisors_excluding_self (n : ℕ) : ℕ :=
  (Finset.filter (λ x, x ≠ n) (Finset.range (n + 1))).sum

theorem triple_sum_of_divisors_of_8_eq_0 :
  sum_of_divisors_excluding_self (sum_of_divisors_excluding_self (sum_of_divisors_excluding_self 8)) = 0 := 
  sorry

end triple_sum_of_divisors_of_8_eq_0_l464_464845


namespace chairs_per_row_l464_464449

theorem chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) 
  (h_total_chairs : total_chairs = 432) (h_num_rows : num_rows = 27) : 
  total_chairs / num_rows = 16 :=
by
  sorry

end chairs_per_row_l464_464449


namespace max_value_fourth_power_l464_464181

theorem max_value_fourth_power (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) : 
  a^4 + b^4 + c^4 + d^4 ≤ 4^(4/3) :=
sorry

end max_value_fourth_power_l464_464181


namespace exist_2020_consecutive_integers_with_1812_signatures_l464_464445

-- Define the signature function
def signature (n : ℕ) : ℕ :=
  n.factors.length

-- Define the function f(n) that counts the number of integers 
-- with signature less than 11 in the range [n, n+2019]
def f (n : ℕ) : ℕ :=
  (List.range 2020).count (λ k, (signature (n + k) < 11))

theorem exist_2020_consecutive_integers_with_1812_signatures : 
  ∃ n : ℕ, f n = 1812 :=
sorry

end exist_2020_consecutive_integers_with_1812_signatures_l464_464445


namespace relay_race_distance_l464_464300

theorem relay_race_distance
  (n : ℤ) (hn : n = 9)
  (total_distance : ℚ) (hd : total_distance = 425) :
  let each_member_run := total_distance / n in
  each_member_run ≈ 47.22 :=
by
  sorry

end relay_race_distance_l464_464300


namespace prob_reverse_order_prob_second_from_bottom_l464_464388

section CementBags

-- Define the basic setup for the bags and worker choices
variables (bags : list ℕ) (choices : list ℕ) (shed gate : list ℕ)

-- This ensures we are indeed considering 4 bags in a sequence
axiom bags_is_4 : bags.length = 4

-- This captures that the worker makes 4 sequential choices with equal probability for each, where 0 represents shed and 1 represents gate
axiom choices_is_4_and_prob : (choices.length = 4) ∧ (∀ i in choices, i ∈ [0, 1])

-- This condition captures that eventually, all bags must end up in the shed
axiom all_bags_in_shed : ∀ b ∈ bags, b ∈ shed

-- Prove for part (a): Probability of bags in reverse order in shed is 1/8
theorem prob_reverse_order : (choices = [0, 0, 0, 0] → (reverse bags = shed)) → 
  (probability (choices = [0, 0, 0, 0]) = 1 / 8) :=
 sorry

-- Prove for part (b): Probability of the second-from-bottom bag in the bottom is 1/8
theorem prob_second_from_bottom : 
  (choices = [1, 1, 0, 0] → (shed.head = bags.nth 1.get_or_else 0)) → 
  (probability (choices = [1, 1, 0, 0]) = 1 / 8) :=
 sorry

end CementBags

end prob_reverse_order_prob_second_from_bottom_l464_464388


namespace probability_point_A_on_hyperbola_l464_464224

-- Define the set of numbers
def numbers : List ℕ := [1, 2, 3]

-- Define the coordinates of point A taken from the set, where both numbers are different
def point_A_pairs : List (ℕ × ℕ) :=
  [ (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) ]

-- Define the function indicating if a point (m, n) lies on the hyperbola y = 6/x
def lies_on_hyperbola (m n : ℕ) : Prop :=
  n = 6 / m

-- Calculate the probability of a point lying on the hyperbola
theorem probability_point_A_on_hyperbola : 
  (point_A_pairs.countp (λ (p : ℕ × ℕ), lies_on_hyperbola p.1 p.2)).toRat / (point_A_pairs.length).toRat = 1 / 3 := 
sorry

end probability_point_A_on_hyperbola_l464_464224


namespace find_f_2022_l464_464271

noncomputable def f : ℕ → ℕ := sorry

axiom recurrence_relation (n : ℕ) : f (n + 2) - 2022 * f (n + 1) + 2021 * f n = 0

axiom functional_equality : f (20^22) = f (22^20)

axiom initial_condition : f 2021 = 2022

theorem find_f_2022 : f 2022 = 2022 :=
begin
  sorry
end

end find_f_2022_l464_464271


namespace probability_point_on_hyperbola_l464_464234

theorem probability_point_on_hyperbola :
  let S := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) },
      Hyperbola := { (x, y) | y = 6 / x },
      Favourable := S ∩ Hyperbola
  in
    Favourable.card / S.card = 1 / 3 :=
by
  sorry

end probability_point_on_hyperbola_l464_464234


namespace max_size_of_M_l464_464669

open Finset

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def valid_subset (M : Finset ℕ) : Prop :=
  ∀ x y z ∈ M, x ≠ y → y ≠ z → z ≠ x → ¬ is_perfect_square (x * y * z)

theorem max_size_of_M (M : Finset ℕ) (h1 : M ⊆ (range 16).filter (λ x, x > 0)) (h2 : valid_subset M) : M.card ≤ 10 := 
  sorry

end max_size_of_M_l464_464669


namespace remaining_days_to_finish_coke_l464_464106

def initial_coke_in_ml : ℕ := 2000
def daily_consumption_in_ml : ℕ := 200
def days_already_drunk : ℕ := 3

theorem remaining_days_to_finish_coke : 
  (initial_coke_in_ml / daily_consumption_in_ml) - days_already_drunk = 7 := 
by
  sorry -- Proof placeholder

end remaining_days_to_finish_coke_l464_464106


namespace LS_parallel_PQ_l464_464593

noncomputable def L (A P C M : Point) : Point := 
  -- Intersection of AP and CM
  sorry

noncomputable def S (A N C Q : Point) : Point := 
  -- Intersection of AN and CQ
  sorry

theorem LS_parallel_PQ (A P C M N Q : Point) 
  (hL : L A P C M) (hS : S A N C Q) : 
  Parallel (Line (L A P C M)) (Line (S A N C Q)) :=
sorry

end LS_parallel_PQ_l464_464593


namespace probability_point_on_hyperbola_l464_464231

theorem probability_point_on_hyperbola :
  let S := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) },
      Hyperbola := { (x, y) | y = 6 / x },
      Favourable := S ∩ Hyperbola
  in
    Favourable.card / S.card = 1 / 3 :=
by
  sorry

end probability_point_on_hyperbola_l464_464231


namespace inequality_holds_l464_464257

theorem inequality_holds
  (n : ℕ) (x : Fin n → ℝ)
  (h1 : 2 ≤ n)
  (h2 : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ (i : Fin (n-1)), x 0 * (1 - x (i + 1)) ≥ (1/4) * x 0 * (1 - x (n-1)) := 
by 
  sorry

end inequality_holds_l464_464257


namespace division_and_multiplication_l464_464730

theorem division_and_multiplication (dividend divisor quotient factor product : ℕ) 
  (h₁ : dividend = 24) 
  (h₂ : divisor = 3) 
  (h₃ : quotient = dividend / divisor) 
  (h₄ : factor = 5) 
  (h₅ : product = quotient * factor) : 
  quotient = 8 ∧ product = 40 := 
by 
  sorry

end division_and_multiplication_l464_464730


namespace min_people_in_group_l464_464561

theorem min_people_in_group (B G : ℕ) (h : B / (B + G : ℝ) > 0.94) : B + G ≥ 17 :=
sorry

end min_people_in_group_l464_464561


namespace compute_a_plus_b_l464_464448

theorem compute_a_plus_b
  (a b : ℕ)
  (hpos_a : 0 < a)
  (hpos_b : 0 < b)
  (hproduct : (∏ i in finset.range (b - a), real.log ((a : ℝ) + (i : ℕ) + 1) / real.log ((a : ℝ) + (i : ℕ))) = 3)
  (hterms : b - a = 1024)
  (ha_eq : b = a ^ 3) : 
  a + b = 4112 :=
by {
  sorry
}

end compute_a_plus_b_l464_464448


namespace select_students_l464_464344

theorem select_students :
  ∃ n : ℕ, n = choose 7 4 - choose 4 4 ∧ n = 34 :=
by
  use 34
  calc
    choose 7 4 - choose 4 4
    = 35 - 1 : by norm_num
    = 34 : by norm_num

end select_students_l464_464344


namespace factorize_difference_of_squares_l464_464003

variable {R : Type} [CommRing R]

theorem factorize_difference_of_squares (a x y : R) : a * x^2 - a * y^2 = a * (x + y) * (x - y) :=
by
  sorry

end factorize_difference_of_squares_l464_464003


namespace a_10_equals_19_l464_464743

noncomputable def a_sequence : ℕ → ℕ
| 0     := 1
| (n+1) := a_sequence n + 2

theorem a_10_equals_19 : a_sequence 9 = 19 :=
by {
  sorry
}

end a_10_equals_19_l464_464743


namespace equal_sum_partition_l464_464451

theorem equal_sum_partition (n : ℕ) (a : fin 2n → ℕ) (h_mean : (∑ i, a i = 4 * n)) (h_bound : ∀ i, a i ≤ 2 * n) :
    ∃ S1 S2 : finset (fin 2n), S1 ∩ S2 = ∅ ∧ S1 ∪ S2 = finset.univ ∧ S1.card = n ∧ S2.card = n ∧ (∑ i in S1, a i) = (∑ i in S2, a i) := 
sorry

end equal_sum_partition_l464_464451


namespace pythagorean_theorem_l464_464213

theorem pythagorean_theorem (a b c : ℕ) (h : a^2 + b^2 = c^2) : a^2 + b^2 = c^2 :=
by
  sorry

end pythagorean_theorem_l464_464213


namespace special_points_max_l464_464205

/- Definitions based on conditions -/

-- Define a spherical planet with continents and ocean
structure SphericalPlanet :=
(continents : ℕ)
(ocean : Type)

-- A condition where the planet has 4 continents
def hasFourContinents (p : SphericalPlanet) : Prop :=
p.continents = 4

-- Definition of a special point in the ocean
structure SpecialPoint (p : SphericalPlanet) :=
(point : p.ocean)
(is_special : ∃ C₁ C₂ C₃ : Type, C₁ ≠ C₂ ∧ C₂ ≠ C₃ ∧ C₃ ≠ C₁)

-- Predicate stating that a point is a special point in the ocean.
def isSpecialPoint (p : SphericalPlanet) (sp : SpecialPoint p) : Prop :=
∃ pts : list p.ocean, pts.length = 3 ∧ ∀ pt in pts, pt ≠ sp.point

-- The proposition that states the maximum number of special points is 4.
def maxSpecialPointsIsFour (p : SphericalPlanet) : Prop :=
∀ (sps : list (SpecialPoint p)), (∀ sp in sps, isSpecialPoint p sp) → sps.length ≤ 4

/- The main proof statement -/
theorem special_points_max (p : SphericalPlanet) (h : hasFourContinents p) :
  maxSpecialPointsIsFour p :=
sorry

end special_points_max_l464_464205


namespace find_n_l464_464560

theorem find_n (n k : ℕ) (b : ℝ) (h_n2 : n ≥ 2) (h_ab : b ≠ 0 ∧ k > 0) (h_a_eq : ∀ (a : ℝ), a = k^2 * b) :
  (∀ (S : ℕ → ℝ → ℝ), S 1 b + S 2 b = 0) →
  n = 2 * k + 1 := 
sorry

end find_n_l464_464560


namespace number_of_negatives_l464_464401

def is_negative (a : ℤ) := a < 0

def num_negatives := (list.map is_negative [-5, -1 ^ 2, (-1 : ℤ) ^ 2021, (-(1/3 : ℚ))^2, | -3 |]).countp id

theorem number_of_negatives :
  num_negatives = 3 :=
sorry

end number_of_negatives_l464_464401


namespace find_theta_find_AB_distance_l464_464556

-- Curve C parametric equations
def C (α : ℝ) := (2 * Real.cos α, 2 + 2 * Real.sin α)

-- Line l parametric equations
def l (t : ℝ) := (√3 - (√3 / 2) * t, 3 + (1 / 2) * t)

-- Point A in polar coordinates
def A := (2 * √3, 2 * Real.pi / 3)

-- The angle θ for point A
theorem find_theta (θ : ℝ) 
  (h1 : θ ∈ Set.Ioo (Real.pi / 2) Real.pi) 
  (h2 : ∃ (ρ : ℝ), A = (ρ, θ)) :
  θ = 2 * Real.pi / 3 :=
sorry

-- Point B where the ray OA intersects line l
def B (ρB : ℝ) := (ρB, 2 * Real.pi / 3)

-- The distance |AB|
theorem find_AB_distance (ρA ρB : ℝ)
  (hA : A = (ρA, 2 * Real.pi / 3))
  (hB : B ρB = (4 * √3, 2 * Real.pi / 3)) :
  |ρB - ρA| = 2 * √3 :=
sorry

end find_theta_find_AB_distance_l464_464556


namespace angle_C_triangle_area_l464_464542

-- Given triangle ABC and known relations
variables {A B C a b c : ℝ}
hypothesis (h_1 : 2 * c * real.cos A + a = 2 * b)
hypothesis (h_2 : a + b = 4)
hypothesis (h_3 : C = real.pi / 3)

-- Part Ⅰ: Proof that C = π/3 given 2c cos A + a = 2b.
theorem angle_C : C = real.pi / 3 := by apply h_3

-- Part Ⅱ: Proof for the area of triangle ABC given above conditions
theorem triangle_area : 
  c = 2 → 
  A + B + C = real.pi → 
  1/2 * a * b * real.sin C = real.sqrt 3 := sorry

end angle_C_triangle_area_l464_464542


namespace investment_inequality_l464_464697

-- Defining the initial investment
def initial_investment : ℝ := 200

-- Year 1 changes
def alpha_year1 := initial_investment * 1.30
def beta_year1 := initial_investment * 0.80
def gamma_year1 := initial_investment * 1.10
def delta_year1 := initial_investment * 0.90

-- Year 2 changes
def alpha_final := alpha_year1 * 0.85
def beta_final := beta_year1 * 1.30
def gamma_final := gamma_year1 * 0.95
def delta_final := delta_year1 * 1.20

-- Prove the final inequality
theorem investment_inequality : beta_final < gamma_final ∧ gamma_final < delta_final ∧ delta_final < alpha_final :=
by {
  sorry
}

end investment_inequality_l464_464697


namespace find_y_l464_464120

theorem find_y (x y : ℝ) (h1 : x ^ (3 * y) = 8) (h2 : x = 2) : y = 1 :=
by {
  sorry
}

end find_y_l464_464120


namespace andy_wrong_questions_l464_464402

axiom a b c d : ℕ
axiom h1 : a + b = c + d
axiom h2 : a + d = b + c + 6
axiom h3 : c = 6

theorem andy_wrong_questions : a = 12 := by
  sorry

end andy_wrong_questions_l464_464402


namespace max_area_trapezoid_l464_464756

theorem max_area_trapezoid :
  ∀ {AB CD : ℝ}, 
    AB = 6 → CD = 14 → 
    (∃ (r1 r2 : ℝ), r1 = AB / 2 ∧ r2 = CD / 2 ∧ r1 + r2 = 10) → 
    (1 / 2 * (AB + CD) * 10 = 100) :=
by
  intros AB CD hAB hCD hExist
  sorry

end max_area_trapezoid_l464_464756


namespace length_of_intervals_l464_464912

theorem length_of_intervals (n : ℕ) (hn : 0 < n) :
  ∃ A : set ℝ, (∀ x ∈ A, 0 < x ∧ x < 1) ∧ (∀ (p q : ℕ), q ≤ n^2 → (x ∈ A → |x - p/q| > 1/n^3)) ∧
  (∀ I ∈ A, I.is_interval) ∧ measurable_set A ∧ measure A ≤ 100 / n := sorry

end length_of_intervals_l464_464912


namespace find_x_l464_464452

def vector_a := (4, 8)
def vector_b (x : ℝ) := (x, 4)

def vectors_parallel {α : Type} [LinearOrderedField α] (v1 v2 : α × α) : Prop :=
  ∃ k : α, k ≠ 0 ∧ v1 = (k * (v2.1), k * (v2.2))

theorem find_x (x : ℝ) (h : vectors_parallel vector_a (vector_b x)) : x = 2 :=
by sorry

end find_x_l464_464452


namespace smallest_period_l464_464440

noncomputable def f (x : ℝ) : ℝ := 3 * sin (2 * x) + 2^|sin (2 * x)^2| + 5 * |sin (2 * x)|

theorem smallest_period (x : ℝ) : f (x + π / 2) = f x :=
by
  sorry

end smallest_period_l464_464440


namespace range_of_k_l464_464017

-- Definitions of the conditions in the original problem
def a : ℝ := 2
def hyperbola (x y k : ℝ) : Prop := x^2 / 4 + y^2 / k = 1

-- The key property for the problem: eccentricity in (1, 2)
def eccentricity (k : ℝ) : ℝ := (Real.sqrt(4 - k)) / a
def e_in_interval (e : ℝ) : Prop := 1 < e ∧ e < 2

-- The statement we want to prove
theorem range_of_k {k : ℝ} (h : ∀ x y, hyperbola x y k) (he : e_in_interval (eccentricity k)) : -12 < k ∧ k < 0 :=
sorry

end range_of_k_l464_464017


namespace binomial_expansion_terms_equal_iff_coefficient_of_x_term_l464_464890

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_expansion_terms_equal_iff (n : ℕ) :
  binomial_coefficient n 3 = binomial_coefficient n 8 ↔ n = 11 :=
by sorry

theorem coefficient_of_x_term (x : ℤ) (n : ℕ) (h : n = 11) :
  -- The coefficient of the x term in the expansion of (sqrt(x) - 2/x)^n when n = 11
  let expansion_term := λ r, (-2 : ℤ)^r * binomial_coefficient n r * (x ^ ((11 - 3 * r) / 2)) in
  expansion_term 3 = -1320 * x :=
by sorry

end binomial_expansion_terms_equal_iff_coefficient_of_x_term_l464_464890


namespace yuna_solved_problems_l464_464335

def yuna_problems_per_day : ℕ := 8
def days_per_week : ℕ := 7
def yuna_weekly_problems : ℕ := 56

theorem yuna_solved_problems :
  yuna_problems_per_day * days_per_week = yuna_weekly_problems := by
  sorry

end yuna_solved_problems_l464_464335


namespace log_equation_solution_l464_464720

theorem log_equation_solution (a : ℝ) (h₁ : a ≠ 1) : 
  ∀ x : ℝ, 2 * log x a + log (a * x) a + 3 * log (a^2 * x) a = 0 ↔ x = a^(-1 / 2) ∨ x = a^(-4 / 3) :=
by
  sorry

end log_equation_solution_l464_464720


namespace sine_theorem_trihedral_angle_first_cosine_theorem_trihedral_angle_second_cosine_theorem_trihedral_angle_l464_464683

-- Given conditions
variables (α β γ A B C : ℝ)

-- Main theorem to prove each part
theorem sine_theorem_trihedral_angle :
  (sin α / sin A = sin β / sin B) ∧
  (sin β / sin B = sin γ / sin C) :=
sorry

theorem first_cosine_theorem_trihedral_angle :
  cos α = cos β * cos γ + sin β * sin γ * cos A :=
sorry

theorem second_cosine_theorem_trihedral_angle :
  cos A = - cos B * cos C + sin B * sin C * cos α :=
sorry

end sine_theorem_trihedral_angle_first_cosine_theorem_trihedral_angle_second_cosine_theorem_trihedral_angle_l464_464683


namespace four_letters_three_mailboxes_l464_464297

theorem four_letters_three_mailboxes : (3 ^ 4) = 81 :=
  by sorry

end four_letters_three_mailboxes_l464_464297


namespace triangle_angles_l464_464157

variable (A B C a b c : ℝ)

-- Conditions
def condition1 : Prop := (a + b + c) * (a + b - c) = 3 * a * b
def condition2 : Prop := Real.sin A ^ 2 = Real.sin B ^ 2 + Real.sin C ^ 2

-- Angles are 30°, 60°, 90°
def angles_30_60_90 : Prop :=
    (A = π / 6 ∧ B = π / 3 ∧ C = π / 2) ∨
    (A = π / 3 ∧ B = π / 6 ∧ C = π / 2) ∨
    (A = π / 2 ∧ B = π / 6 ∧ C = π / 3) ∨
    (A = π / 2 ∧ B = π / 3 ∧ C = π / 6)

-- The theorem statement
theorem triangle_angles (h1 : condition1 a b c) (h2 : condition2 A B C) : angles_30_60_90 A B C :=
by sorry

end triangle_angles_l464_464157


namespace ten_faucets_fill_time_l464_464883

theorem ten_faucets_fill_time (rate : ℕ → ℕ → ℝ) (gallons : ℕ) (minutes : ℝ) :
  rate 5 9 = 150 / 5 ∧
  rate 10 135 = 75 / 30 * rate 10 9 / 0.9 * 60 →
  9 * 60 / 30 * 75 / 10 * 60 = 135 :=
sorry

end ten_faucets_fill_time_l464_464883


namespace bill_due_in_9_months_l464_464746

-- Define the conditions
def true_discount : ℝ := 240
def face_value : ℝ := 2240
def interest_rate : ℝ := 0.16

-- Define the present value calculated from the true discount and face value
def present_value := face_value - true_discount

-- Define the time in months required to match the conditions
noncomputable def time_in_months : ℝ := 12 * ((face_value / present_value - 1) / interest_rate)

-- State the theorem that the bill is due in 9 months
theorem bill_due_in_9_months : time_in_months = 9 :=
by
  sorry

end bill_due_in_9_months_l464_464746


namespace fill_in_square_l464_464109

theorem fill_in_square (x y : ℝ) (h : 4 * x^2 * (81 / 4 * x * y) = 81 * x^3 * y) : (81 / 4 * x * y) = (81 / 4 * x * y) :=
by
  sorry

end fill_in_square_l464_464109


namespace ratio_of_supply_to_demand_l464_464134

theorem ratio_of_supply_to_demand (supply demand : ℕ)
  (hs : supply = 1800000)
  (hd : demand = 2400000) :
  supply / (Nat.gcd supply demand) = 3 ∧ demand / (Nat.gcd supply demand) = 4 :=
by
  sorry

end ratio_of_supply_to_demand_l464_464134


namespace light_gray_area_correct_dark_gray_area_correct_l464_464151

def side_of_square (k : ℝ) : ℝ := 2 * Real.sqrt k

def congruent_triangle_area (area_square : ℝ) : ℝ := area_square / 2

def right_triangle_area (s : ℝ) : ℝ := (s * (s / 2)) / 2

def light_gray_area (k : ℝ) (area_square : ℝ) : ℝ :=
  congruent_triangle_area (area_square k) - (right_triangle_area (side_of_square k))

def dark_gray_area (k : ℝ) (area_square : ℝ) : ℝ :=
  area_square (area_square k) / 5

theorem light_gray_area_correct (k : ℝ) (h : ∃ area_square = 4 * k) :
  light_gray_area k area_square = k := 
sorry

theorem dark_gray_area_correct (k : ℝ) (h : ∃ area_square = 4 * k) :
  dark_gray_area k area_square = 4 * k / 5 := 
sorry

end light_gray_area_correct_dark_gray_area_correct_l464_464151


namespace coefficient_of_neg2ab_is_neg2_l464_464265

-- Define the term -2ab
def term : ℤ := -2

-- Define the function to get the coefficient from term -2ab
def coefficient (t : ℤ) : ℤ := t

-- The theorem stating the coefficient of -2ab is -2
theorem coefficient_of_neg2ab_is_neg2 : coefficient term = -2 :=
by
  -- Proof can be filled later
  sorry

end coefficient_of_neg2ab_is_neg2_l464_464265


namespace max_values_greater_than_half_l464_464053

theorem max_values_greater_than_half {α β γ : ℝ} (hα : 0 < α ∧ α < π/2)
  (hβ : 0 < β ∧ β < π/2) (hγ : 0 < γ ∧ γ < π/2) (h_distinct : α ≠ β ∧ β ≠ γ ∧ γ ≠ α) :
  (count_fun >(1/2) [sin α * cos β, sin β * cos γ, sin γ * cos α]) ≤ 2 :=
sorry

end max_values_greater_than_half_l464_464053


namespace prove_parallel_l464_464647

-- Define points A, P, C, M, N, Q, and functions L, S.
variables {A P C M N Q L S : Point}
variables [incidence_geometry]

-- Define L and S based on the given conditions.
def L := intersection (line_through A P) (line_through C M)
def S := intersection (line_through A N) (line_through C Q)

-- Define the condition to prove LS is parallel to PQ.
theorem prove_parallel : parallel (line_through L S) (line_through P Q) :=
sorry

end prove_parallel_l464_464647


namespace least_n_for_length_l464_464588

/-- We start with the origin --/
def A0 : ℕ × ℕ := (0, 0)

/-- Distinct points Aj and Bj where Aj lies on x-axis and Bj on y = x^2 --/
variable A : ℕ → ℝ
variable B : ℕ → ℝ

/-- Equilateral triangle property --/
variable (a : ℕ → ℝ)
variable (x_squared : ℝ → ℝ := λ x, x * x)

axiom a1 : ∀ (n : ℕ), A n ≠ A (n + 1)
axiom a2 : ∀ (n : ℕ), B n = x_squared (A n)
axiom a3 : ∀ (n : ℕ) (m : ℕ), (m ≠ n) → (B m ≠ B n)
axiom eq_triangle : ∀ (n : ℕ), (n > 0) → a n = (2* A n - A (n - 1)) / √3

-- Initial conditions and recursion
def a_n_recurrence (n : ℕ) : ℝ :=
  match n with
  | 0     => 0
  | 1     => 2 / 3
  | (n+1) => 2 / 3 + a n

theorem least_n_for_length (n : ℕ) :
  ((n * (n + 1)) / 3) ≥ 50 → n ≥ 12 := by
  sorry

end least_n_for_length_l464_464588


namespace probability_point_on_hyperbola_l464_464232

theorem probability_point_on_hyperbola :
  let S := { (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2) },
      Hyperbola := { (x, y) | y = 6 / x },
      Favourable := S ∩ Hyperbola
  in
    Favourable.card / S.card = 1 / 3 :=
by
  sorry

end probability_point_on_hyperbola_l464_464232


namespace power_function_value_l464_464066

theorem power_function_value (f : ℝ → ℝ) (h : ∀ x, f x = x ^ (-1 / 2))
    (hf : f 2 = (2:ℝ) ^ (-1 / 2)) : f 4 = 1 / 2 :=
by
  sorry

end power_function_value_l464_464066


namespace probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464397

-- Definitions based on conditions
def bags_in_truck : ℕ := 4

-- Assume random_choice has probability 0.5 for selection
def random_choice : ℕ → ℕ → ℝ := λ a b, 0.5

-- Bag positions and movements
def initial_position (i : ℕ) : ℕ := i

def final_position (i : ℕ) : ℕ := 4 - i

-- Events
def end_up_in_reverse_order (n : ℕ) : Prop := 
  ∀ i < n, final_position i = n - 1 - i

def second_from_bottom_ends_up_bottom (n : ℕ) : Prop :=
  final_position 1 = 0

-- Probabilities calculation
noncomputable def probability_of_reverse_order (n : ℕ) : ℝ := (1/2)^(n - 1)
noncomputable def probability_of_second_from_bottom_as_bottom (n : ℕ) : ℝ := (1/2)^(n - 1)

-- Theorem to prove the statement
theorem probability_reverse_order :
  probability_of_reverse_order bags_in_truck = 1 / 8 :=
by sorry

theorem probability_second_from_bottom_as_bottom :
  probability_of_second_from_bottom_as_bottom bags_in_truck = 1 / 8 :=
by sorry

end probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464397


namespace relationship_between_m_and_n_l464_464971

theorem relationship_between_m_and_n
  (b m n : ℝ)
  (h₁ : m = 2 * (-1 / 2) + b)
  (h₂ : n = 2 * 2 + b) :
  m < n :=
by
  sorry

end relationship_between_m_and_n_l464_464971


namespace one_common_tangent_has_a_value_common_tangent_bisect_each_other_l464_464941

noncomputable def parabola1 (x : ℝ) : ℝ := x^2 + 2 * x

noncomputable def parabola2 (x a : ℝ) : ℝ := -x^2 + a

theorem one_common_tangent_has_a_value
    (a : ℝ)
    (C1 : ℝ → ℝ := parabola1)
    (C2 : ℝ → ℝ := parabola2) :
  (∃ k b : ℝ, ∀ x, C1 x = k * x + b → C2 x a = k * x + b) → a = -1/2 :=
by
  sorry

theorem common_tangent_bisect_each_other
    (a k1 k2 b1 b2 : ℝ)
    (C1 : ℝ → ℝ := parabola1)
    (C2 : ℝ → ℝ := parabola2)
    (tangent1 tangent2 : ℝ → ℝ := λ x, k1 * x + b1 → λ x, k2 * x + b2)
    (x1 x2 y1 y2 : ℝ) :
  tangent1 x1 = C1 x1 → tangent2 x2 = C2 x2 a →
  (x1 + x2) / 2 = (x1 + (-(x1 + 1))) / 2 ∧
  (y1 + y2) / 2 = (C1 x1 + C2 ( - (x1 + 1)) a) / 2 :=
by
  sorry

end one_common_tangent_has_a_value_common_tangent_bisect_each_other_l464_464941


namespace real_if_and_only_if_m_eq_neg3_complex_if_and_only_if_m_not_eq_purely_imaginary_if_and_only_if_m_eq_2_in_second_quadrant_if_and_only_if_l464_464887

section ComplexConditions

variables (m : ℝ)

def z : ℂ := m^2 * (1 / (m + 5) + complex.I)
             + (8 * m + 15) * complex.I
             + (m - 6) / (m + 5)

-- 1. z is a real number if and only if m = -3
theorem real_if_and_only_if_m_eq_neg3 : z m ∈ ℝ ↔ m = -3 := sorry

-- 2. z is a complex number if and only if m ≠ -3 and m ≠ -5
theorem complex_if_and_only_if_m_not_eq : z m ∈ ℂ ↔ (m ≠ -3 ∧ m ≠ -5) := sorry

-- 3. z is a purely imaginary number if and only if m = 2
theorem purely_imaginary_if_and_only_if_m_eq_2 : (z m).im ≠ 0 ∧ (z m).re = 0 ↔ m = 2 := sorry

-- 4. z is in the second quadrant if and only if m < -5 or -3 < m < 2
theorem in_second_quadrant_if_and_only_if : (z m).re < 0 ∧ (z m).im > 0 ↔ (m < -5 ∨ (-3 < m ∧ m < 2)) := sorry

end ComplexConditions

end real_if_and_only_if_m_eq_neg3_complex_if_and_only_if_m_not_eq_purely_imaginary_if_and_only_if_m_eq_2_in_second_quadrant_if_and_only_if_l464_464887


namespace find_integer_K_l464_464524

-- Definitions based on the conditions
def is_valid_K (K Z : ℤ) : Prop :=
  Z = K^4 ∧ 3000 < Z ∧ Z < 4000 ∧ K > 1 ∧ ∃ (z : ℤ), K^4 = z^3

theorem find_integer_K :
  ∃ (K : ℤ), is_valid_K K 2401 :=
by
  sorry

end find_integer_K_l464_464524


namespace irreducible_root_decomposition_l464_464211

theorem irreducible_root_decomposition {p q : ℤ} (hpq : Int.gcd p q = 1) 
  (P : ℤ[X]) (hroot : eval (rat.mk p q) P = 0) :
  ∃ Q : ℤ[X], P = (C q * X - C p) * Q := by
  sorry

end irreducible_root_decomposition_l464_464211


namespace solve_for_y_l464_464116

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end solve_for_y_l464_464116


namespace license_plate_count_l464_464107

theorem license_plate_count : 
  let letters := 26 in 
  let digit_choices := 10 in
  let unique_digit_choices := 9 in
  letters * digit_choices * unique_digit_choices = 2340 :=
by sorry

end license_plate_count_l464_464107


namespace necessary_but_not_sufficient_l464_464114

-- Definitions based on conditions from part a)
def real_numbers := Set ℝ
def A := {x : ℝ | -3 < x ∧ x < 2}
def B := {x : ℝ | 0 < x ∧ x < 2}

-- The theorem that captures the necessary but not sufficient relationship
theorem necessary_but_not_sufficient (x : ℝ) :
  (x ∈ A) → ¬(x ∈ B) :=
begin
  assume h1 : x ∈ A,
  sorry
end

end necessary_but_not_sufficient_l464_464114


namespace infinite_solutions_of_quadratic_l464_464586

theorem infinite_solutions_of_quadratic :
  ∀ m n i j k l : ℕ,
  ∃ a b c : ℕ,
  (a = 2^m * 5^n) ∧ (b = 2^i * 5^j) ∧ (c = 2^k * 5^l) ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (b^2 = a * c + a * c) →
  ∃ infinitely_many_solutions : ℕ,
      ∃ (ax^2 - 2bx + c = 0), have a_property that(ax^2 - 2bx + c = 0),
      sorry.

end infinite_solutions_of_quadratic_l464_464586


namespace perimeter_of_adjacent_rectangles_l464_464806

theorem perimeter_of_adjacent_rectangles (s : ℕ) (h : s = 100) :
  let width := s,
      height := s / 2 in
  3 * width + 4 * height = 500 :=
by
  have width := s
  have height := s / 2
  rw [h]
  unfold width height
  sorry

end perimeter_of_adjacent_rectangles_l464_464806


namespace max_product_of_two_digit_numbers_l464_464090

theorem max_product_of_two_digit_numbers :  
  ∃ (a b c d e : ℤ), 
  (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (10 ≤ c ∧ c < 100) ∧ (10 ≤ d ∧ d < 100) ∧ (10 ≤ e ∧ e < 100) ∧
  (a * b * c * d * e = 1785641760) ∧ 
  (∀ x y : ℤ, (x ∈ digits (a * b * c * d * e)) ↔ (y ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) → (x, y) ∈ digits a ∨ (x, y) ∈ digits b ∨ (x, y) ∈ digits c ∨ (x, y) ∈ digits d ∨ (x, y) ∈ digits e) := 
sorry

end max_product_of_two_digit_numbers_l464_464090


namespace total_sample_size_l464_464793

noncomputable def sample_size (x : ℕ) : ℕ :=
  10 * x

theorem total_sample_size : ∀ (x : ℕ), 
  (3 * x = 150) → sample_size x = 500 :=
by
  intros x h
  have h1 : x = 50 := by linarith
  rw [h1, sample_size]
  sorry

end total_sample_size_l464_464793


namespace irreducible_root_decomposition_l464_464212

theorem irreducible_root_decomposition {p q : ℤ} (hpq : Int.gcd p q = 1) 
  (P : ℤ[X]) (hroot : eval (rat.mk p q) P = 0) :
  ∃ Q : ℤ[X], P = (C q * X - C p) * Q := by
  sorry

end irreducible_root_decomposition_l464_464212


namespace joan_football_games_l464_464579

/-- 
  Joan went to 4 football games this year,
  Joan went to 13 football games in all,
  Prove that Joan went to 9 football games last year.
-/
theorem joan_football_games (games_this_year games_total : ℕ) (h1 : games_this_year = 4) (h2 : games_total = 13) : games_total - games_this_year = 9 := 
by 
  rw [h1, h2]
  rfl

end joan_football_games_l464_464579


namespace range_of_a_l464_464895

noncomputable def f (x a : ℝ) := Real.log x + a / x

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 0, x * (2 * Real.log a - Real.log x) ≤ a) : 
  0 < a ∧ a ≤ 1 / Real.exp 1 :=
by
  sorry

end range_of_a_l464_464895


namespace exists_non_integer_solution_l464_464171

def q (x y : ℝ) (b : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) : ℝ :=
  b.1 + b.2 * x + b.3 * y + b.4 * x^2 + b.5 * x * y + b.6 * y^2 + b.7 * x^3 + b.8 * x^2 * y + b.9 * x * y^2 + b.10 * y^3

lemma q_conditions (b : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) :
  q 0 0 b = 0 ∧
  q 1 0 b = 0 ∧
  q (-1) 0 b = 0 ∧
  q 0 1 b = 0 ∧
  q 0 (-1) b = 0 ∧
  q 1 1 b = 0 ∧
  q (-1) (-1) b = 0 ∧
  q 2 2 b = 0 ∧
  (∂ x, q x 1 b) 1 = 0 ∧
  (∂ y, q 1 y b) 1 = 0 :=
sorry

theorem exists_non_integer_solution (b : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ) 
  (h : q_conditions b) : 
  ∃ u v : ℝ, u ≠ ⌊u⌋ ∧ v ≠ ⌊v⌋ ∧ q u v b = 0 :=
sorry

end exists_non_integer_solution_l464_464171


namespace area_ratio_of_octagon_l464_464736

theorem area_ratio_of_octagon (P : Type) [parallelogram P]
  (midpoints_connected_to_opposite_vertices : ∀ (M N : P), midpoint M N → opposite_vertex N)
  (octagon_area : ℝ) (parallelogram_area : ℝ) :
  octagon_area / parallelogram_area = 1 / 6 :=
sorry

end area_ratio_of_octagon_l464_464736


namespace possible_m_value_l464_464072

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3 - (1/2)*x - 1
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := x^3 + m / x

theorem possible_m_value :
  ∃ m : ℝ, (m = (1/2) - (1/Real.exp 1)) ∧
    (∀ x1 x2 : ℝ, 
      (f x1 = g (-x1) m) →
      (f x2 = g (-x2) m) →
      x1 ≠ 0 ∧ x2 ≠ 0 ∧
      m = x1 * Real.exp x1 - (1/2) * x1^2 - x1 ∧
      m = x2 * Real.exp x2 - (1/2) * x2^2 - x2) :=
by
  sorry

end possible_m_value_l464_464072


namespace correct_statements_of_the_following_l464_464415

theorem correct_statements_of_the_following (statement1 statement2 statement3 statement4 statement5 statement6 statement7 : Prop)
  (h1 : statement1 = (∀ (x̄ ȳ : ℝ) (b a : ℝ),
    ∃ (x₁ y₁ : ℝ), y₁ = b * x₁ + a ∧ (x₁, y₁) = (x̄, ȳ)))
  (h2 : statement2 = (∀ (c : ℝ) (data : list ℝ),
    variance (map (+ c) data) = variance data))
  (h3 : statement3 = (∀ (R2 : ℝ), R2 = 1 → "better fit"))
  (h4 : statement4 = (∀ (K : ℝ) (X Y : set ℝ),
    K = rvs.correlation X Y → "larger K means weaker relationship"))
  (h5 : statement5 = (∀ (x y : ℝ),
    correlation x y → functional relationship x y))
  (h6 : statement6 = (∀ (residuals : list ℝ),
    (evenly_distributed residuals) → "better model"))
  (h7 : statement7 = (∀ (rss1 rss2 : ℝ), rss1 < rss2 → "better fit")) :
  statement2 ∧ statement3 ∧ statement6 ∧ statement7 := 
by { sorry }

end correct_statements_of_the_following_l464_464415


namespace sin_double_beta_l464_464036

theorem sin_double_beta (α β : ℝ) (h₀ : 0 < α) (h₁ : α < real.pi) (h₂ : 0 < β) (h₃ : β < real.pi)
  (h : real.cos (2 * α + β) - 2 * real.cos (α + β) * real.cos α = 3 / 5) :
  real.sin (2 * β) = -24 / 25 := sorry

end sin_double_beta_l464_464036


namespace trapezoid_perimeter_l464_464563

-- Defining the conditions of the trapezoid ABCD
variables (A B C D : Type) [metric_space A]
variables (dist : A → A → ℝ)

-- Given conditions on lengths and right angles
def condition1 : dist A B = 6 := sorry
def condition2 : dist B C = 4 := sorry
def condition3 : dist C D = 2 * dist A B := sorry
def condition4 : angle (line A B) (line B C) = 90 := sorry
def condition5 : angle (line B C) (line C D) = 90 := sorry

-- The theorem statement
theorem trapezoid_perimeter (h1 : condition1) (h2 : condition2)
  (h3 : condition3) (h4 : condition4) (h5 : condition5) :
  dist A B + dist B C + dist C D + (dist A D) = 22 + 3 * sqrt 5 := 
sorry

end trapezoid_perimeter_l464_464563


namespace probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464399

-- Definitions based on conditions
def bags_in_truck : ℕ := 4

-- Assume random_choice has probability 0.5 for selection
def random_choice : ℕ → ℕ → ℝ := λ a b, 0.5

-- Bag positions and movements
def initial_position (i : ℕ) : ℕ := i

def final_position (i : ℕ) : ℕ := 4 - i

-- Events
def end_up_in_reverse_order (n : ℕ) : Prop := 
  ∀ i < n, final_position i = n - 1 - i

def second_from_bottom_ends_up_bottom (n : ℕ) : Prop :=
  final_position 1 = 0

-- Probabilities calculation
noncomputable def probability_of_reverse_order (n : ℕ) : ℝ := (1/2)^(n - 1)
noncomputable def probability_of_second_from_bottom_as_bottom (n : ℕ) : ℝ := (1/2)^(n - 1)

-- Theorem to prove the statement
theorem probability_reverse_order :
  probability_of_reverse_order bags_in_truck = 1 / 8 :=
by sorry

theorem probability_second_from_bottom_as_bottom :
  probability_of_second_from_bottom_as_bottom bags_in_truck = 1 / 8 :=
by sorry

end probability_reverse_order_probability_second_from_bottom_as_bottom_l464_464399


namespace imaginary_part_of_z_is_zero_l464_464041

-- Given definition of complex numbers and the imaginary unit i
def i : ℂ := Complex.I
def z : ℂ := (1 - i) / (i * (1 + i))

-- Statement to prove: the imaginary part of z is 0.
theorem imaginary_part_of_z_is_zero : Complex.im z = 0 := sorry

end imaginary_part_of_z_is_zero_l464_464041


namespace parabola_directrix_l464_464045

variable {F P1 P2 : Point}

def is_on_parabola (F : Point) (P1 : Point) : Prop := 
  -- Definition of a point being on the parabola with focus F and a directrix (to be determined).
  sorry

def construct_circles (F P1 P2 : Point) : Circle × Circle :=
  -- Construct circles centered at P1 and P2 passing through F.
  sorry

def common_external_tangents (k1 k2 : Circle) : Nat :=
  -- Function to find the number of common external tangents between two circles.
  sorry

theorem parabola_directrix (F P1 P2 : Point) (h1 : is_on_parabola F P1) (h2 : is_on_parabola F P2) :
  ∃ (k1 k2 : Circle), construct_circles F P1 P2 = (k1, k2) → 
    common_external_tangents k1 k2 = 2 :=
by
  -- Proof that under these conditions, there are exactly 2 common external tangents.
  sorry

end parabola_directrix_l464_464045


namespace max_positive_integer_n_l464_464918

-- Declare the sequence and necessary conditions
variables {a : ℕ → ℤ} -- Arithmetic sequence of integers
variables {d : ℤ} -- Common difference

-- Conditions for the sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def cond_a1 (a : ℕ → ℤ) : Prop :=
  a 1 > 0

def cond_sum_23_and_24 (a : ℕ → ℤ) : Prop :=
  a 23 + a 24 > 0

def cond_product_23_and_24 (a : ℕ → ℤ) : Prop :=
  a 23 * a 24 < 0

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a i

-- The statement to be proved
theorem max_positive_integer_n (a : ℕ → ℤ) (d : ℤ) 
  (h_arith : is_arithmetic_sequence a d) 
  (h_a1 : cond_a1 a) 
  (h_sum_23_24 : cond_sum_23_and_24 a) 
  (h_prod_23_24 : cond_product_23_and_24 a) : 
  ∃ n : ℕ, n = 46 ∧ sum_first_n_terms a n > 0 := 
sorry

end max_positive_integer_n_l464_464918


namespace hyperbola_vertex_distance_l464_464427

open Real

/-- The distance between the vertices of the hyperbola represented by the equation
    (y-4)^2 / 32 - (x+3)^2 / 18 = 1 is 8√2. -/
theorem hyperbola_vertex_distance :
  let a := sqrt 32
  2 * a = 8 * sqrt 2 :=
by
  sorry

end hyperbola_vertex_distance_l464_464427


namespace ls_parallel_pq_l464_464651

-- Definitions for points L and S, and their parallelism
variables {A P C M N Q L S : Type}

-- Assume points and lines for the geometric setup
variables (A P C M N Q L S : Type)

-- Assume the intersections and the required parallelism conditions
variables (hL : ∃ (AP CM : Set (Set L)), L ∈ AP ∩ CM)
variables (hS : ∃ (AN CQ : Set (Set S)), S ∈ AN ∩ CQ)

-- The goal: LS is parallel to PQ
theorem ls_parallel_pq : LS ∥ PQ :=
sorry

end ls_parallel_pq_l464_464651


namespace selection_arrangement_count_l464_464747

theorem selection_arrangement_count (boys girls : ℕ) (total selection : ℕ) :
  boys = 5 → girls = 4 → total = boys + girls → selection = 3 →
  nat.choose total selection * nat.factorial selection = 504 :=
by
  intros h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  subst h4
  sorry

end selection_arrangement_count_l464_464747


namespace triangle_A_range_l464_464976

noncomputable def angle_range (a b : ℝ) (has_solution : Prop) : set (ℝ) :=
  {A | 0 < A ∧ A ≤ (real.pi / 4)}

theorem triangle_A_range :
  ∀ (A : ℝ),
  ∀ (b : ℝ), b = 2 * real.sqrt 2 → 
  ∀ (a : ℝ), a = 2 → 
  (∃ C : ℝ, ∃ c : ℝ, A + C < real.pi ∧ A + C > 0 ∧ C > 0 ∧ c > 0) →
  (A ∈ angle_range 2 (2 * real.sqrt 2) true) :=
by
  intros A b hb a ha has_solution
  rw [ha, hb]
  sorry

end triangle_A_range_l464_464976


namespace cannot_form_tetrahedron_l464_464548

/-- 
If \(ABC\) is an equilateral triangle with \(AM, BM, CM \in (\frac{1}{2}, \frac{\sqrt{3}}{2})\), 
\(\varepsilon = \min \{|AM - BM|, |BM - CM|, |CM - AM|, \frac{1}{10}\}\), 
\(AN < \varepsilon\), 
\(BN > \frac{\sqrt{3}}{2} + \varepsilon\), 
and \(CN > \frac{\sqrt{3}}{2} + \varepsilon\), 
then the six segments \(AM, BM, CM, AN, BN, CN\) cannot form a tetrahedron.
-/
theorem cannot_form_tetrahedron
  (ABC : Triangle) 
  (AM BM CM AN BN CN : ℝ)
  (h_AM : AM ∈ (1 / 2, real.sqrt 3 / 2))
  (h_BM : BM ∈ (1 / 2, real.sqrt 3 / 2))
  (h_CM : CM ∈ (1 / 2, real.sqrt 3 / 2))
  (ε : ℝ := min {abs (AM - BM), abs (BM - CM), abs (CM - AM), 1 / 10})
  (h_AN : AN < ε)
  (h_BN : BN > real.sqrt 3 / 2 + ε)
  (h_CN : CN > real.sqrt 3 / 2 + ε) :
  ¬ (tetrahedron AM BM CM AN BN CN) :=
by
  sorry

end cannot_form_tetrahedron_l464_464548


namespace distinguishable_large_triangles_l464_464303

/--
There is an unlimited supply of congruent equilateral triangles made of colored paper. 
Each triangle is a solid color with the same color on both sides of the paper. 
A large equilateral triangle is constructed from four of these paper triangles. 
Two large triangles are considered distinguishable if it is not possible to place one 
on the other, using translations, rotations, and/or reflections, so that their 
corresponding small triangles are of the same color. Given that there are eight different 
colors of triangles from which to choose, how many distinguishable large equilateral 
triangles can be constructed?
-/
theorem distinguishable_large_triangles : 
  let colors := 8 in
  let all_same := colors in
  let two_same_one_diff := colors * (colors - 1) in
  let all_diff := Nat.choose colors 3 in
  let combinations := all_same + two_same_one_diff + all_diff in
  combinations * colors = 960 :=
by 
  sorry

end distinguishable_large_triangles_l464_464303


namespace descending_number_count_correct_l464_464776

def is_descending (n : ℕ) : Prop :=
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d1 > d2 ∧ d2 > d3 ∧ d3 > d4

def four_digit_numbers : finset ℕ :=
  {n | n ∈ finset.range 10000 ∧ 
       (n / 1000 % 10 ∈ {0, 1, 2, 3, 4, 5}) ∧ 
       (n / 100 % 10 ∈ {0, 1, 2, 3, 4, 5}) ∧ 
       (n / 10 % 10 ∈ {0, 1, 2, 3, 4, 5}) ∧ 
       (n % 10 ∈ {0, 1, 2, 3, 4, 5}) ∧ 
       ((n / 1000 % 10) ≠ (n / 100 % 10)) ∧ 
       ((n / 1000 % 10) ≠ (n / 10 % 10)) ∧ 
       ((n / 1000 % 10) ≠ (n % 10)) ∧ 
       ((n / 100 % 10) ≠ (n / 10 % 10)) ∧ 
       ((n / 100 % 10) ≠ (n % 10)) ∧ 
       ((n / 10 % 10) ≠ (n % 10))}

theorem descending_number_count_correct : 
  (four_digit_numbers.filter is_descending).card = 15 := 
sorry

end descending_number_count_correct_l464_464776


namespace fraction_evaluation_l464_464002

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem fraction_evaluation :
  (sqrt 2 * (sqrt 3 - sqrt 7)) / (2 * sqrt (3 + sqrt 5)) =
  (30 - 10 * sqrt 5 - 6 * sqrt 21 + 2 * sqrt 105) / 8 :=
by
  sorry

end fraction_evaluation_l464_464002


namespace find_number_of_pens_l464_464570

-- Definitions based on the conditions in the problem
def total_utensils (P L : ℕ) : Prop := P + L = 108
def pencils_formula (P L : ℕ) : Prop := L = 5 * P + 12

-- The theorem we need to prove
theorem find_number_of_pens (P L : ℕ) (h1 : total_utensils P L) (h2 : pencils_formula P L) : P = 16 :=
by sorry

end find_number_of_pens_l464_464570


namespace circle_periodicity_l464_464729

-- Define the setup for the circles inscribed in the triangles
def triangle (A B C : Point) : Type :=
  { p1 : Point // p1 ≠ A ∧ p1 ≠ B ∧ p1 ≠ C}

def Circle : Type := { center : Point // center ≠ ∅ ∧ radius : Real }

-- Functions to retrieve the radius and assign height to each step.
def radius (S : Circle) : Real := S.radius
def height (tri : triangle A B C) (i : Nat) : Real := sorry  -- To be defined properly

-- Relationship from the problem 5.9a
axiom radius_height_relation (r : Real) (ri : Real) (hi_plus_2 : Real) :
  r / ri - 1 = 1 - (2 * r / hi_plus_2)

-- Goal: Prove the periodicity of the radius such that S_1 coincides with S_7
theorem circle_periodicity (S1 : Circle) (h : triangle A B C) (r : Real) :
  ∃ S7 : Circle, ∀ i : Nat, (radius S1 = radius S7) :=
by
  sorry

end circle_periodicity_l464_464729


namespace card_picking_l464_464356

/-
Statement of the problem:
- A modified deck of cards has 65 cards.
- The deck is divided into 5 suits, each of which has 13 cards.
- The cards are placed in random order.
- Prove that the number of ways to pick two different cards from this deck with the order of picking being significant is 4160.
-/
theorem card_picking : (65 * 64) = 4160 := by
  sorry

end card_picking_l464_464356


namespace three_digit_numbers_subtract_297_l464_464447

theorem three_digit_numbers_subtract_297:
  (∃ (p q r : ℕ), 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ 0 ≤ r ∧ r ≤ 9 ∧ (100 * p + 10 * q + r - 297 = 100 * r + 10 * q + p)) →
  (num_valid_three_digit_numbers = 60) :=
by
  sorry

end three_digit_numbers_subtract_297_l464_464447


namespace valid_range_m_l464_464472

/-- Proposition stating that vector (1, 1, m) is parallel to vector (-1, -1, |m|) implies m ≤ 0. -/
def proposition_p (m : ℝ) : Prop := (∃ k : ℝ, k ≠ 0 ∧ (1, 1, m) = k • (-1, -1, |m|)) = false

/-- Proposition stating that the equation x²/(2m+1) + y²/(m-3) = 1 represents a hyperbola implies
    -1/2 < m < 3. -/
def proposition_q (m : ℝ) : Prop := (2 * m + 1) * (m - 3) < 0

/-- Final condition stating the logical combination of ¬p and p ∨ q, leading to the range of m. -/
theorem valid_range_m (m : ℝ) :
  (¬ proposition_p m) ∧ (proposition_p m ∨ proposition_q m) → 0 < m ∧ m < 3 :=
by
  intros h
  sorry

end valid_range_m_l464_464472


namespace mary_money_left_l464_464199

variable (p : ℝ)

theorem mary_money_left :
  have cost_drinks := 3 * p
  have cost_medium_pizza := 2 * p
  have cost_large_pizza := 3 * p
  let total_cost := cost_drinks + cost_medium_pizza + cost_large_pizza
  30 - total_cost = 30 - 8 * p := by {
    sorry
  }

end mary_money_left_l464_464199


namespace austin_starting_amount_l464_464838

theorem austin_starting_amount :
  let robots := 7 in
  let cost_per_robot := 8.75 in
  let tax := 7.22 in
  let change := 11.53 in
  let total_cost := robots * cost_per_robot in
  let total_paid := total_cost + tax in
  let starting_amount := total_paid + change in
  starting_amount = 80.00 :=
by
  sorry

end austin_starting_amount_l464_464838


namespace odds_burning_out_during_second_period_l464_464029

def odds_burning_out_during_first_period := 1 / 3
def odds_not_burning_out_first_period := 1 - odds_burning_out_during_first_period
def odds_not_burning_out_next_period := odds_not_burning_out_first_period / 2

theorem odds_burning_out_during_second_period :
  (1 - odds_not_burning_out_next_period) = 2 / 3 := by
  sorry

end odds_burning_out_during_second_period_l464_464029


namespace arithmetic_and_geometric_sequence_l464_464468

-- Definitions for the problem
variables {a : ℕ → ℤ} {b : ℕ → ℤ}
variable {d : ℤ}

-- Conditions in the problem
def arith_seq (d : ℤ) : Prop := ∀ n, a (n + 1) = a n + d
def geom_cond (a : ℕ → ℤ) : Prop := a 7 * a 1 = (a 3) ^ 2
def a1_eq_4 : Prop := a 1 = 4

-- Questions to prove
def S10_eq_130 : Prop := (10 * (a 1 + a 10)) / 2 = 130
def T100_eq_150 (b : ℕ → ℤ) : Prop := 
  (Σ i in finset.range 100, if i % 2 = 0 then b (i + 1) else 0) = 150

def b246_to_100_eq_50 (b : ℕ → ℤ) : Prop :=
  (Σ i in finset.range 50, b (2 * (i + 1))) = 50

-- Lean statement
theorem arithmetic_and_geometric_sequence
  (d_ne_zero : d ≠ 0) 
  (a_cond : arith_seq d) 
  (geom_seq: geom_cond a)
  (ha1: a1_eq_4)
  (T100_def : T100_eq_150 b):
  S10_eq_130 ∧ b246_to_100_eq_50 b :=
by
  sorry

end arithmetic_and_geometric_sequence_l464_464468


namespace right_angled_trapezoid_distinct_configurations_l464_464154

theorem right_angled_trapezoid_distinct_configurations
  (ABCD : Type)
  [is_trapezoid : is_right_angled_trapezoid ABCD]
  [diagonals_intersect_perpendicularly : diagonals_intersect_perpendicularly ABCD]
  (M : Point)
  (A B C D : Point)
  (dist_MA : dist M A = 8)
  (dist_MB : dist M B = 27) :
  number_of_distinct_trapezoids ABCD = 12 :=
sorry

end right_angled_trapezoid_distinct_configurations_l464_464154


namespace sin_HIC_is_correct_l464_464785

/-- Definition of a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Definition of a cube given its vertices -/
structure Cube where
  A B C D E F G H : Point3D
  side_length : ℝ

/-- Definition of the midpoint calculation between two points -/
def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2, y := (p1.y + p2.y) / 2, z := (p1.z + p2.z) / 2 }

/-- Sin of the angle HIC in a given cube with point I as the midpoint of edge AB -/
def sin_HIC (cube : Cube) : ℝ :=
  let I := midpoint cube.A cube.B
  let H := cube.H
  let C := cube.C
  -- Calculate lengths HI, IC, and HC
  let HI := Math.sqrt ((H.x - I.x) ^ 2 + (H.y - I.y) ^ 2 + (H.z - I.z) ^ 2)
  let IC := Math.sqrt ((I.x - C.x) ^ 2 + (I.y - C.y) ^ 2 + (I.z - C.z) ^ 2)
  let HC := Math.sqrt ((H.x - C.x) ^ 2 + (H.y - C.y) ^ 2 + (H.z - C.z) ^ 2)
  -- Calculate cos(angle HIC) using the cosine rule
  let cos_HIC := (HI ^ 2 + IC ^ 2 - HC ^ 2) / (2 * HI * IC)
  -- Calculate sin(angle HIC) using the identity sin^2 + cos^2 = 1
  Math.sqrt (1 - cos_HIC ^ 2)

theorem sin_HIC_is_correct (cube : Cube) (I_is_midpoint : midpoint cube.A cube.B = {x := 0.5 * cube.side_length, y := 0, z := 0}) :
  sin_HIC cube = 0.613 :=
by
  sorry

end sin_HIC_is_correct_l464_464785


namespace point_on_hyperbola_probability_l464_464228

theorem point_on_hyperbola_probability :
  let s := ({1, 2, 3} : Finset ℕ) in
  let p := ∑ x in s.sigma (λ x, s.filter (λ y, y ≠ x)),
             if (∃ m n, x = (m, n) ∧ n = (6 / m)) then 1 else 0 in
  p / (s.card * (s.card - 1)) = (1 / 3) :=
by
  -- Conditions and setup
  let s := ({1, 2, 3} : Finset ℕ)
  let t := s.sigma (λ x, s.filter (λ y, y ≠ x))
  let p := t.filter (λ (xy : ℕ × ℕ), xy.snd = 6 / xy.fst)
  have h_total : t.card = 6, by sorry
  have h_count : p.card = 2, by sorry

  -- Calculate probability
  calc
    ↑(p.card) / ↑(t.card) = 2 / 6 : by sorry
    ... = 1 / 3 : by norm_num

end point_on_hyperbola_probability_l464_464228


namespace roots_opposite_signs_l464_464800

theorem roots_opposite_signs (a b c: ℝ) 
  (h1 : (b^2 - a * c) > 0)
  (h2 : (b^4 - a^2 * c^2) < 0) :
  a * c < 0 :=
sorry

end roots_opposite_signs_l464_464800


namespace lucy_withdrawal_l464_464687

-- Given conditions
def initial_balance : ℕ := 65
def deposit : ℕ := 15
def final_balance : ℕ := 76

-- Define balance before withdrawal
def balance_before_withdrawal := initial_balance + deposit

-- Theorem to prove
theorem lucy_withdrawal : balance_before_withdrawal - final_balance = 4 :=
by sorry

end lucy_withdrawal_l464_464687


namespace radius_of_tangent_circles_l464_464755

theorem radius_of_tangent_circles
  (r : ℝ)
  (h1 : ∀ (x y : ℝ), (x-r)^2 + y^2 = r^2 → 3x^2 + 4y^2 = 12) :
  r = (sqrt 3) / 2 :=
by
  sorry

end radius_of_tangent_circles_l464_464755


namespace isosceles_triangle_area_percentage_l464_464358

-- Definitions based on problem conditions
structure Pentagon :=
  (x : ℝ) -- length of the short side of the rectangle and the leg of the isosceles triangle

def rectangle_area (p : Pentagon) := 2 * p.x ^ 2

def triangle_height (p : Pentagon) := p.x * (sqrt 3) / 2

def triangle_area (p : Pentagon) := 1 / 2 * p.x * triangle_height p

def pentagon_area (p : Pentagon) := 
  rectangle_area p + triangle_area p

def triangle_fraction (p : Pentagon) :=
  triangle_area p / pentagon_area p

def triangle_percentage (p : Pentagon) := 
  triangle_fraction p * 100

-- Theorem to prove
theorem isosceles_triangle_area_percentage (p : Pentagon) : 
  abs (triangle_percentage p - 11.83) < 1e-2 :=
by
  sorry

end isosceles_triangle_area_percentage_l464_464358


namespace probability_point_on_hyperbola_l464_464216

-- Define the problem conditions
def number_set := {1, 2, 3}
def point_on_hyperbola (x y : ℝ) : Prop := y = 6 / x

-- Formalize the problem statement
theorem probability_point_on_hyperbola :
  let combinations := ({(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)} : set (ℝ × ℝ)) in
  let on_hyperbola := set.filter (λ p : ℝ × ℝ, point_on_hyperbola p.1 p.2) combinations in
  fintype.card on_hyperbola / fintype.card combinations = 1 / 3 :=
by sorry

end probability_point_on_hyperbola_l464_464216


namespace medal_award_ways_l464_464299

-- Conditions
def sprinters : ℕ := 12
def americans : ℕ := 5
def medals : ℕ := 3 -- gold, silver, bronze

-- Question: Prove the number of ways to award medals with no more than two Americans winning is 1260.
theorem medal_award_ways :
  ∃ ways : ℕ,
    (ways = 210 + 630 + 420) ∧
    ways = 1260 :=
begin
  use 1260,
  split,
  { -- We are provided that the correct way count is the sum of 210, 630, and 420
    norm_num, 
  },
  { -- The total ways should be 1260
    refl
  }
end

end medal_award_ways_l464_464299


namespace sin_shift_l464_464308

theorem sin_shift (x : ℝ) : 
  ∃ a : ℝ, (∀ x : ℝ, sin (2 * x - π / 3) = sin (2 * (x - a))) ∧ a = π / 6 :=
by 
  use π / 6
  intro x
  have h : 2 * x - π / 3 = 2 * (x - π / 6),
    by linarith
  rw h
  exact (eq.refl (sin (2 * (x - π / 6))))

end sin_shift_l464_464308


namespace triangle_perimeter_is_345_l464_464276

/-- Define the elements and given conditions -/
noncomputable def tangent_points (XT TY : ℝ) : Prop :=
XT = 26 ∧ TY = 31

noncomputable def radius (r : ℝ) : Prop :=
r = 24

/-- Main statement to prove the perimeter is 345 -/
theorem triangle_perimeter_is_345 (XT TY r : ℝ) 
  (hXT : XT = 26) 
  (hTY : TY = 31) 
  (hr : r = 24) : 
  (XT + TY + 2 * ((576 * 57 / 230)) = 345) :=
begin
  sorry
end

end triangle_perimeter_is_345_l464_464276


namespace wax_current_eq_l464_464088

-- Define the constants for the wax required and additional wax needed
def w_required : ℕ := 166
def w_more : ℕ := 146

-- Define the term to represent the current wax he has
def w_current : ℕ := w_required - w_more

-- Theorem statement to prove the current wax quantity
theorem wax_current_eq : w_current = 20 := by
  -- Proof outline would go here, but per instructions, we skip with sorry
  sorry

end wax_current_eq_l464_464088


namespace number_of_non_3_divisors_of_300_l464_464514

/--
  The statement of the problem:
  Prove that the number of positive divisors of 300 that are not divisible by 3 is 9.
-/
theorem number_of_non_3_divisors_of_300 :
  let count_non_3_divisors (n : ℕ) : ℕ :=
    if n = 300 then 
      let valid_a := {0, 1, 2}
      let valid_c := {0, 1, 2}
      finset.card (finset.product valid_a valid_c)
    else 0
  in count_non_3_divisors 300 = 9 :=
by
  sorry

end number_of_non_3_divisors_of_300_l464_464514


namespace rational_root_polynomial_factoring_l464_464210

theorem rational_root_polynomial_factoring (p q : ℚ) (P : Polynomial ℤ) (h_irreducible : p.denom = q) (h_root : P.eval (p / q) = 0) :
  ∃ Q : Polynomial ℤ, P = Polynomial.C q * (Polynomial.X - Polynomial.C p) * Q :=
sorry

end rational_root_polynomial_factoring_l464_464210


namespace sufficient_but_not_necessary_l464_464462

theorem sufficient_but_not_necessary (x : ℝ) :
  (|x - 3| - |x - 1| < 2) → x ≠ 1 ∧ ¬ (∀ x : ℝ, x ≠ 1 → |x - 3| - |x - 1| < 2) :=
by
  sorry

end sufficient_but_not_necessary_l464_464462


namespace sin_cos_product_l464_464892

theorem sin_cos_product (α : ℝ) (h : sin α + cos α = 1 / 2) : sin α * cos α = -3 / 8 :=
sorry

end sin_cos_product_l464_464892


namespace value_of_f_3_and_f_neg_7_point_5_l464_464051

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 1) = -f x
axiom definition_f : ∀ x : ℝ, -1 < x → x < 1 → f x = x

theorem value_of_f_3_and_f_neg_7_point_5 :
  f 3 + f (-7.5) = 0.5 :=
sorry

end value_of_f_3_and_f_neg_7_point_5_l464_464051


namespace smallest_positive_period_f_l464_464934

def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x - π / 6) + 2 * sin^2 (x - π / 12)

theorem smallest_positive_period_f : ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ ε > 0, ε < T → ∀ x, f (x + ε) ≠ f x) :=
sorry

end smallest_positive_period_f_l464_464934


namespace part1_monotonicity_part2_intersection_l464_464937

noncomputable def f (a x : ℝ) : ℝ := -x * Real.exp (a * x + 1)

theorem part1_monotonicity (a : ℝ) : 
  ∃ interval : Set ℝ, 
    (∀ x ∈ interval, ∃ interval' : Set ℝ, 
      (∀ x' ∈ interval', f a x' ≤ f a x) ∧ 
      (∀ x' ∈ Set.univ \ interval', f a x' > f a x)) :=
sorry

theorem part2_intersection (a b x_1 x_2 : ℝ) (h1 : a > 0) (h2 : b ≠ 0)
  (h3 : f a x_1 = -b * Real.exp 1) (h4 : f a x_2 = -b * Real.exp 1)
  (h5 : x_1 ≠ x_2) : 
  - (1 / Real.exp 1) < a * b ∧ a * b < 0 ∧ a * (x_1 + x_2) < -2 :=
sorry

end part1_monotonicity_part2_intersection_l464_464937


namespace find_diameter_l464_464152

noncomputable def diameter_of_circle (AE EB ED : ℝ) : ℝ :=
  let CE := (AE * EB) / ED in
  let radius := real.sqrt (CE^2 + (AE + EB)^2 / 4) in
  2 * radius

theorem find_diameter :
  diameter_of_circle 2 6 3 = real.sqrt 65 :=
begin
  sorry
end

end find_diameter_l464_464152


namespace total_amount_spent_l464_464123

variables (P J I T : ℕ)

-- Given conditions
def Pauline_dress : P = 30 := sorry
def Jean_dress : J = P - 10 := sorry
def Ida_dress : I = J + 30 := sorry
def Patty_dress : T = I + 10 := sorry

-- Theorem to prove the total amount spent
theorem total_amount_spent :
  P + J + I + T = 160 :=
by
  -- Placeholder for proof
  sorry

end total_amount_spent_l464_464123


namespace joneal_stops_in_quarter_A_l464_464694

theorem joneal_stops_in_quarter_A (circumference run_distance : ℕ) 
    (S A_start : ℕ) (h_circ : circumference = 40) (h_run : run_distance = 8000) 
    (h_start : S = 0) (h_A : A_start = 0) : 
    (run_distance % circumference = 0) → (S + (run_distance % circumference)) % circumference = A_start :=
by
  intros h_mod
  have h1 : run_distance % circumference = 0 := h_mod
  have h2 : S + (run_distance % circumference) = 0 := by
    rw [h1, add_zero]
  exact eq.trans (mod_eq_of_lt 0 (by norm_num)) (eq.symm h_A)
  sorry

end joneal_stops_in_quarter_A_l464_464694


namespace average_probable_weight_l464_464544

theorem average_probable_weight (weight : ℝ) (h1 : 61 < weight) (h2 : weight ≤ 64) : 
  (61 + 64) / 2 = 62.5 := 
by
  sorry

end average_probable_weight_l464_464544


namespace max_product_S1_S2_l464_464149

-- Define necessary points and objects
variables {A B C E F D : Type} 
variable {area : ∀ {X Y Z : Type}, (X × Y × Z) → ℝ}
variable {S1 S2 : ℝ} -- Defining S1 and S2 as real numbers

-- Given conditions
variables (hABC_area : area (A, B, C) = 60)
           (hAE_AB : ∀ {x : ℝ}, x = 3)
           (hAF_AC : ∀ {x : ℝ}, x = 3)
           (hD_on_BC : ∀ {x : ℝ}, x ∈ (B, C))

-- Variables for the subareas S1 and S2
variables (S1 : ℝ) (S2 : ℝ)

-- Theorem statement that translates the problem to Lean
theorem max_product_S1_S2 (hS1 : S1 = (1 / 3) * area (A, B, C))
                          (hS2 : S2 = (2 / 3) * area (A, B, C))
                          (hS1_S2_sum : S1 + S2 = 40)
                          : S1 * S2 = 400 := 
by
  sorry

end max_product_S1_S2_l464_464149


namespace arithmetic_sequence_problem_l464_464469

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h1 : (a 1 - 3) ^ 3 + 3 * (a 1 - 3) = -3)
  (h12 : (a 12 - 3) ^ 3 + 3 * (a 12 - 3) = 3) :
  a 1 < a 12 ∧ (12 * (a 1 + a 12)) / 2 = 36 :=
by
  sorry

end arithmetic_sequence_problem_l464_464469


namespace three_digit_numbers_divisible_by_17_l464_464099

theorem three_digit_numbers_divisible_by_17 : 
  let three_digit_numbers := {n : ℕ | 100 ≤ n ∧ n ≤ 999}
  let divisible_by_17 := {n : ℕ | n % 17 = 0}
  ∃! count : ℕ, count = fintype.card {n : ℕ // n ∈ three_digit_numbers ∧ n ∈ divisible_by_17} ∧ count = 53 :=
begin
  sorry
end

end three_digit_numbers_divisible_by_17_l464_464099


namespace find_side_length_l464_464908

theorem find_side_length (a : ℤ) (h1 : odd a) (h2 : 1 < a) (h3 : a < 5) : a = 3 := by
  sorry

end find_side_length_l464_464908


namespace calculate_expression_l464_464407

theorem calculate_expression (a : ℝ) : 3 * a * (2 * a^2 - 4 * a) - 2 * a^2 * (3 * a + 4) = -20 * a^2 :=
by
  sorry

end calculate_expression_l464_464407


namespace LS_parallel_PQ_l464_464598

noncomputable def L (A P C M : Point) : Point := 
  -- Intersection of AP and CM
  sorry

noncomputable def S (A N C Q : Point) : Point := 
  -- Intersection of AN and CQ
  sorry

theorem LS_parallel_PQ (A P C M N Q : Point) 
  (hL : L A P C M) (hS : S A N C Q) : 
  Parallel (Line (L A P C M)) (Line (S A N C Q)) :=
sorry

end LS_parallel_PQ_l464_464598


namespace ls_parallel_pq_l464_464656

-- Definitions for points L and S, and their parallelism
variables {A P C M N Q L S : Type}

-- Assume points and lines for the geometric setup
variables (A P C M N Q L S : Type)

-- Assume the intersections and the required parallelism conditions
variables (hL : ∃ (AP CM : Set (Set L)), L ∈ AP ∩ CM)
variables (hS : ∃ (AN CQ : Set (Set S)), S ∈ AN ∩ CQ)

-- The goal: LS is parallel to PQ
theorem ls_parallel_pq : LS ∥ PQ :=
sorry

end ls_parallel_pq_l464_464656


namespace jenna_owes_amount_l464_464578

theorem jenna_owes_amount (initial_bill : ℝ) (rate : ℝ) (times : ℕ) : 
  initial_bill = 400 → rate = 0.02 → times = 3 → 
  owed_amount = (400 * (1 + 0.02)^3) := 
by
  intros
  sorry

end jenna_owes_amount_l464_464578


namespace spanish_not_german_l464_464986

theorem spanish_not_german :
  ∀ (total_students : ℕ) (both_languages : ℕ) (total_spanish : ℕ) (total_german : ℕ)
  (only_spanish : ℕ),
  total_students = 30 →
  both_languages = 2 →
  total_spanish = 3 * total_german →
  only_spanish = (total_spanish - both_languages) →
  only_spanish = 20 :=
by {
  intros,
  sorry
}

end spanish_not_german_l464_464986


namespace runner_time_difference_l464_464365

def run_speeds_problem (v : ℝ) : Prop :=
  let t1 := 20 / v in
  let t2 := 24 in
  t2 - t1 = 12

theorem runner_time_difference :
  ∀ (v : ℝ), 40 / v = 24 -> run_speeds_problem v := 
by
  intros v hv_eq
  -- The proof goes here
  sorry

end runner_time_difference_l464_464365


namespace right_triangle_third_side_square_l464_464547

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) :
  c^2 = 28 ∨ c^2 = 100 :=
by { sorry }

end right_triangle_third_side_square_l464_464547


namespace unique_homomorphism_is_identity_l464_464552

-- Defining the graph structure
structure Graph :=
  (V : Type) -- Vertices
  (E : V → V → Prop) -- Edges (undirected)

-- Given graph with vertices A, B, C, and D
def G : Graph :=
  { V := { A, B, C, D },
    E := λ x y, (x = A ∧ y = B) ∨ (x = B ∧ y = A)
             ∨ (x = A ∧ y = C) ∨ (x = C ∧ y = A)
             ∨ (x = A ∧ y = D) ∨ (x = D ∧ y = A)
             ∨ (x = B ∧ y = C) ∨ (x = C ∧ y = B)
             ∨ (x = B ∧ y = D) ∨ (x = D ∧ y = B)
             ∨ (x = C ∧ y = D) ∨ (x = D ∧ y = C) }

-- Defining a graph homomorphism
def graph_homomorphism (G H : Graph) :=
  { f : G.V → H.V // ∀ (x y : G.V), G.E x y → H.E (f x) (f y) }

-- The identity map on a graph
def id_homomorphism (G : Graph) : graph_homomorphism G G :=
  ⟨id, by { intros x y hxy, exact hxy }⟩

-- Stating the theorem: The only graph homomorphism from G to itself is the identity map
theorem unique_homomorphism_is_identity : ∀ f : graph_homomorphism G G, f = id_homomorphism G :=
sorry

end unique_homomorphism_is_identity_l464_464552


namespace count_three_digit_integers_divisible_by_5_l464_464516

theorem count_three_digit_integers_divisible_by_5 :
  let digits := {d | d ∈ {6, 7, 8, 9}};
  (0 < ∀ digit ∈ digits) →
  (∀ x : ℕ, 100 ≤ x ∧ x < 1000 →
    (∀ d ∈ nat.digits 10 x, d > 5) →
    (x % 5 = 0) → (card {x | (∀ d ∈ nat.digits 10 x, d > 5) ∧ x % 5 = 0}) = 32) :=
by
  intros digits h_digit_pos h_divisibility h_ranges
  sorry

end count_three_digit_integers_divisible_by_5_l464_464516


namespace gcd_of_powers_l464_464764

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2016 - 1) (h2 : n = 2^2008 - 1) : 
  Nat.gcd m n = 255 :=
by
  -- (Definitions and steps are omitted as only the statement is required)
  sorry

end gcd_of_powers_l464_464764


namespace range_of_a_l464_464939

theorem range_of_a 
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (hf : ∀ x, f x = x + 4 / x)
  (hg : ∀ x, g x = 2 ^ x + a)
  (cond : ∀ x₁ ∈ set.Icc (1/2) 3, ∃ x₂ ∈ set.Icc 2 3, f x₁ ≥ g x₂) : a ≤ 0 :=
by sorry

end range_of_a_l464_464939


namespace number_of_girls_l464_464989

open Rat

theorem number_of_girls 
  (G B : ℕ) 
  (h1 : G / B = 5 / 8)
  (h2 : G + B = 300) 
  : G = 116 := 
by
  sorry

end number_of_girls_l464_464989


namespace flora_needs_more_daily_l464_464023

-- Definitions based on conditions
def totalMilk : ℕ := 105   -- Total milk requirement in gallons
def weeks : ℕ := 3         -- Total weeks
def daysInWeek : ℕ := 7    -- Days per week
def floraPlan : ℕ := 3     -- Flora's planned gallons per day

-- Proof statement
theorem flora_needs_more_daily : (totalMilk / (weeks * daysInWeek)) - floraPlan = 2 := 
by
  sorry

end flora_needs_more_daily_l464_464023


namespace correctness_statements_l464_464270

-- Defining the predicate "Opposite"
def Opposite (a b : Rat) : Prop := a = -b

-- The theorem we want to prove
theorem correctness_statements :
  (∀ a b : Rat, Opposite a b → a + b = 0) ∧
  (∀ a b : Rat, a + b = 0 → Opposite a b) ∧
  (∀ a b : Rat, (a / b = -1) → Opposite a b) :=
by
  sorry

end correctness_statements_l464_464270


namespace min_norm_sum_l464_464796

def complex_function (z : ℂ) (α γ : ℂ) : ℂ :=
  (4 + complex.i) * z^2 + α * z + γ

def α_is_real (α: ℂ) (real_imag: ℝ × ℝ) : Prop :=
  α = real_imag.1 + complex.i * real_imag.2

def γ_is_real (γ: ℂ) (real_imag: ℝ × ℝ) : Prop :=
  γ = real_imag.1 + complex.i * real_imag.2

theorem min_norm_sum (f : ℂ → ℂ) (α γ : ℂ) (real1 : ℝ) (real2 : ℝ) :
  f = complex_function ∧
  (∀ z, f(z) = (4 + complex.i) * z^2 + α * z + γ) ∧ 
  f 1 ∈ ℝ ∧
  f complex.i ∈ ℝ ∧
  ∃ (rα : ℝ × ℝ) (rγ : ℝ × ℝ), α_is_real α rα ∧ γ_is_real γ rγ ∧ 
  re α + im α = 2 ∧ re γ + im γ = 1 → 
  (∃ (res : ℝ), res = complex.abs α + complex.abs γ ∧ res = complex.abs (rα.1 + rα.2) + complex.abs (rγ.1 + rγ.2) ∧ res = sqrt 2 ) :=
by {
  -- proof is required here
  sorry
}

end min_norm_sum_l464_464796


namespace parallel_LS_pQ_l464_464618

universe u

variables {P Q A C M N L S T : Type u} [incidence_geometry P Q A C M N L S T] -- Customize as per specific needs in Lean

-- Assume the required points and their intersections
def intersection_L (A P C M : Type u) [incidence_geometry A P C M] : Type u := sorry
def intersection_S (A N C Q : Type u) [incidence_geometry A N C Q] : Type u := sorry

-- Defining the variables as intersections
variables (L : intersection_L A P C M)
variables (S : intersection_S A N C Q)

theorem parallel_LS_pQ (hL : intersection_L A P C M) (hS : intersection_S A N C Q) :
  parallel L S P Q :=
sorry

end parallel_LS_pQ_l464_464618


namespace probability_on_hyperbola_l464_464236

open Finset

-- Define the function for the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the set of different number pairs from {1, 2, 3}
def pairs : Finset (ℕ × ℕ) := 
  {(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)}.to_finset

-- Define the set of pairs that lie on the hyperbola
def hyperbola_pairs : Finset (ℕ × ℕ) :=
  pairs.filter (λ mn, on_hyperbola mn.1 mn.2)

-- The theorem to prove the probability
theorem probability_on_hyperbola : 
  (hyperbola_pairs.card : ℝ) / (pairs.card : ℝ) = 1 / 3 :=
by
  -- Placeholder for the proof
  sorry

end probability_on_hyperbola_l464_464236


namespace solve_for_y_l464_464117

theorem solve_for_y (x y : ℝ) (h1 : x = 2) (h2 : x^(3*y) = 8) : y = 1 :=
by {
  -- Apply the conditions and derive the proof
  sorry
}

end solve_for_y_l464_464117


namespace min_f_value_l464_464911

theorem min_f_value : 
  ∀ (a : Fin 2019 → ℤ), 
  a 1 = 1 ∧ 
  a 2019 = 99 ∧ 
  (∀ i j, 1 ≤ i → i < j → j ≤ 2019 → a i ≤ a j) → 
  (∃ f₀ : ℤ, f₀ = 7400 ∧
    ∀ (a : Fin 2019 → ℤ), 
    a 1 = 1 ∧ 
    a 2019 = 99 ∧ 
    (∀ i j, 1 ≤ i → i < j → j ≤ 2019 → a i ≤ a j) → 
    (f (λ i : Fin 2019, a i) = f₀))
where
f (a : Fin 2019 → ℤ) : ℤ :=
  (Σ i, (a i) ^ 2) -
  Σ i in finset.range 1 2017, (a i) * (a (i + 2))

end min_f_value_l464_464911


namespace G8_1_l464_464340

theorem G8_1 (A : Real) (h : (√3 / 4) * A^2 = √3) : A = 2 :=
sorry

end G8_1_l464_464340


namespace externally_tangent_circles_l464_464535

variable (m : ℝ)

def circle1 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 - 4 * p.2 + m = 0}
def circle2 := {p : ℝ × ℝ | p.1^2 + p.2^2 - 8 * p.1 - 12 * p.2 + 36 = 0}

theorem externally_tangent_circles :
  (∀ p : ℝ × ℝ, p ∈ circle1 ↔ p ∈ circle2) →
  m = 4 := by
  sorry

end externally_tangent_circles_l464_464535


namespace find_a_l464_464456

noncomputable def general_term (n r : ℕ) (a : ℝ) (x : ℝ) : ℝ :=
  Mathlib.choose n r * a^r * x^(r / 2)

theorem find_a (a : ℝ) (n : ℕ) (h_pos : a > 0)
  (h_coeff : general_term n 4 a 1 = 9 * general_term n 2 a 1)
  (h_third : general_term n 2 a 1 = 135) :
  a = 3 :=
sorry

end find_a_l464_464456


namespace part1_part2_l464_464465

open Finset
open BigOperators

/-- Given a sequence {a_n} with initial conditions and relationships, prove that a_n = n for all natural numbers n. --/
theorem part1 (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, (a n) = (2 * S n) / (n + 1)) :
  ∀ n, a n = n := 
begin
  sorry
end

/-- Given that a_n equals n for all natural numbers n, prove the sum of the sequence {a_n / 2^n} is T_n = 2 - (n + 2) * (1 / 2)^n. --/
theorem part2 (a : ℕ → ℕ) (T : ℕ → ℝ)
  (h : ∀ n, a n = n) :
  ∀ n, T n = ↑2 - ((n + 2) * (1 / 2)^n) := 
begin
  sorry
end

end part1_part2_l464_464465


namespace compound_interest_rate_is_approx_15_percent_l464_464877

noncomputable def interest_rate (P A t : ℝ) : ℝ :=
  ((A / P) ^ (1 / t)) - 1

theorem compound_interest_rate_is_approx_15_percent :
  let P := 4000
  let CI := 1554.5
  let t := 2 + 4 / 12
  let A := P + CI
  interest_rate P A t ≈ 0.15 :=
by
  let P := 4000
  let CI := 1554.5
  let t := 2 + 4 / 12
  let A := P + CI
  -- Here you would proceed to prove interest_rate P A t ≈ 0.15
  sorry

end compound_interest_rate_is_approx_15_percent_l464_464877


namespace multiple_configurations_possible_l464_464052

/-- Given distinct points P(x1, y1), Q(x2, y2), and R = k(x1 + x2, y1 + y2) where k is a real nonzero scalar,
    the possible shape formed by connecting these points and the origin O can be multiple configurations. -/
theorem multiple_configurations_possible (x1 y1 x2 y2 k : ℝ) (h_distinct : (x1, y1) ≠ (x2, y2)) (h_nonzero : k ≠ 0) :
  ∃ (shape : string), shape ∈ ["triangle", "parallelogram", "straight line", "irregular quadrilateral", "multiple configurations possible"] :=
begin
  sorry
end

end multiple_configurations_possible_l464_464052


namespace range_of_f_eq_2_plus_infty_range_of_a_for_inequality_l464_464663

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (log x / log 3) ^ 2 + (a - 1) * (log x / log 3) + 3 * a - 2

theorem range_of_f_eq_2_plus_infty (a : ℝ) (h : ∀ x, x > 0 → 2 ≤ f x a) : a = 7 + 4 * real.sqrt 2 ∨ a = 7 - 4 * real.sqrt 2 :=
sorry

theorem range_of_a_for_inequality (a : ℝ) (h : ∀ x, 3 ≤ x ∧ x ≤ 9 → f (3 * x) a + log (9 * x) / log 3 ≤ 0) : a ≤ -4 / 3 :=
sorry

end range_of_f_eq_2_plus_infty_range_of_a_for_inequality_l464_464663


namespace eggs_in_each_basket_l464_464580

theorem eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 42 % n = 0) (h3 : n ≥ 5) :
  n = 6 :=
by sorry

end eggs_in_each_basket_l464_464580


namespace monotonically_decreasing_interval_l464_464733

noncomputable def f (x : ℝ) : ℝ := log (x^2)

theorem monotonically_decreasing_interval :
  ∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → x < y ∧ y < 0 → f y < f x := 
begin 
  sorry 
end

end monotonically_decreasing_interval_l464_464733


namespace smaller_mold_radius_l464_464353

noncomputable def large_bowl_volume (r : ℝ) : ℝ :=
  (2 / 3) * real.pi * r^3

noncomputable def smaller_molds_volume (r : ℝ) (n : ℕ) : ℝ :=
  n * (2 / 3) * real.pi * r^3

theorem smaller_mold_radius {r_big r_small : ℝ} (n : ℕ) (h_big : large_bowl_volume r_big = (16 / 3) * real.pi)
  (h_eq : smaller_molds_volume r_small n = large_bowl_volume r_big) :
  r_small = 1 / 2 :=
by
  sorry

end smaller_mold_radius_l464_464353


namespace rectangle_area_l464_464802

theorem rectangle_area (d : ℝ) (x : ℝ) (h1 : 5 * x > 0) (h2 : 2 * x > 0) 
(h3 : d = real.sqrt (26 * x^2)) : ∃ k : ℝ, k = 5 / 13 ∧ (5 * x) * (2 * x) = k * d^2 :=
by
  sorry

end rectangle_area_l464_464802


namespace set_intersection_complement_l464_464946

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {1, 3, 4, 6, 7}

theorem set_intersection_complement :
  A ∩ (U \ B) = {2, 5} := 
by
  sorry

end set_intersection_complement_l464_464946


namespace floor_sqrt_18_squared_eq_16_l464_464000

theorem floor_sqrt_18_squared_eq_16 : (Int.floor (Real.sqrt 18)) ^ 2 = 16 := 
by 
  sorry

end floor_sqrt_18_squared_eq_16_l464_464000


namespace num_4_digit_numbers_l464_464953

theorem num_4_digit_numbers : 
  ∃ n : ℕ, n = 6 ∧ (∀ d1 d2 d3 d4 : ℕ, (d1 = 2 ∨ d1 = 0 ∨ d1 = 3) ∧ (d2 = 2 ∨ d2 = 0 ∨ d2 = 3) ∧ 
  (d3 = 2 ∨ d3 = 0 ∨ d3 = 3) ∧ (d4 = 2 ∨ d4 = 0 ∨ d4 = 3) ∧ (d1 * 1000 + d2 * 100 + d3 * 10 + d4 > 3000) →
  let digits := multiset.of_list [d1, d2, d3, d4] in
  digits = multiset.of_list [2, 0, 3, 3]) :=
by {
  -- Proof can be done correctly in Lean
  sorry
}

end num_4_digit_numbers_l464_464953


namespace train_length_is_500_meters_l464_464370

noncomputable def train_speed_km_hr : ℝ := 120
noncomputable def time_to_cross_pole_sec : ℝ := 15

noncomputable def km_hr_to_m_s (speed_km_hr : ℝ) : ℝ := speed_km_hr * 1000 / 3600

noncomputable def train_length_in_meters : ℝ :=
  km_hr_to_m_s train_speed_km_hr * time_to_cross_pole_sec

theorem train_length_is_500_meters : train_length_in_meters ≈ 500 :=
by
  sorry

end train_length_is_500_meters_l464_464370


namespace binomial_problem_solution_l464_464035

theorem binomial_problem_solution
  (a : Fin 10 → ℚ)
  (h1 : ∑ i in Finset.range 10, a i * (2^i) = 3^10)
  (h2 : ∑ i in Finset.range 10, a i * ((-2)^i) = 9) :
  (∑ i in Finset.range 10 | odd i, a i * i) ^ 2 - (∑ i in Finset.range 10 | even i, a i * i) ^ 2 = 3^12 := 
sorry

end binomial_problem_solution_l464_464035


namespace calculation_correct_l464_464406

def calculation : ℝ := 1.23 * 67 + 8.2 * 12.3 - 90 * 0.123

theorem calculation_correct : calculation = 172.20 := by
  sorry

end calculation_correct_l464_464406


namespace triangle_similarity_and_reflections_l464_464042

noncomputable def acute_angle_triangle (A B C H HA HB HC : Point) : Prop :=
  acute_triangle A B C ∧
  orthocenter A B C = H ∧
  foot HA A B C ∧
  foot HB B A C ∧
  foot HC C A B

theorem triangle_similarity_and_reflections (
  A B C H HA HB HC : Point)
  (h: acute_angle_triangle A B C H HA HB HC) :
  (similar (triangle A HB HC) (triangle HA B HC) ∧
  similar (triangle A HB HC) (triangle HA HB C) ∧
  similar (triangle HA B HC) (triangle HA HB C) ∧
  reflection_circumcircle H A B C) :=
by
  sorry

end triangle_similarity_and_reflections_l464_464042


namespace ls_parallel_pq_l464_464606

-- Declare the points and lines
variables {A P C M N Q L S : Type*}

-- Assume given conditions as definitions
def intersection (l1 l2 : set (Type*)) : Type* := sorry -- Define intersection point of l1 and l2 (details omitted for simplicity)

-- Define L as intersection of lines AP and CM
def L : Type* := intersection (set_of (λ x, ∃ L, ∃ P, ∃ C, ∃ M, L = x ∧ x ∈ AP ∧ x ∈ CM)) (λ x, x ∈ AP ∧ x ∈ CM)

-- Define S as intersection of lines AN and CQ
def S : Type* := intersection (set_of (λ x, ∃ S, ∃ N, ∃ C, ∃ Q, S = x ∧ x ∈ AN ∧ x ∈ CQ)) (λ x, x ∈ AN ∧ x ∈ CQ)

-- Statement to prove LS is parallel to PQ
theorem ls_parallel_pq (AP CM AN CQ PQ : set (Type*)) (L : Type*) (S : Type*) :
    L = intersection (AP) (CM) ∧ S = intersection (AN) (CQ) → is_parallel LS PQ :=
by
  sorry -- Proof omitted

end ls_parallel_pq_l464_464606


namespace complex_number_fourth_quadrant_range_l464_464059

theorem complex_number_fourth_quadrant_range (a : ℝ) : 
  let z := (1 : ℂ) + 3 * I
  let w := (a : ℂ) - I
  let product := z * w
  (product.re + 3 > 0 ∧ 3 * a - 1 < 0) → (-3 < a ∧ a < 1/3) :=
begin
  intros h,
  sorry
end

end complex_number_fourth_quadrant_range_l464_464059


namespace equation_of_line_l464_464354

open Real

noncomputable def line_intersects_circle (A B : Point) (l : Line) : Prop :=
  l.passes_through A ∧ l.passes_through B ∧
  distance A B = 8

noncomputable def circle (C : Point) (r : ℝ) (P : Point) : Prop :=
  distance C P = r

def pointA : Point := ⟨-4, 0⟩
def centerC : Point := ⟨-1, 2⟩
def radius : ℝ := 5

theorem equation_of_line : 
  ∃ l : Line, 
    line_intersects_circle A B l ∧ 
    (l.equation = "5x + 12y + 20 = 0" ∨ l.equation = "x + 4 = 0") :=
sorry

end equation_of_line_l464_464354


namespace inequality_holds_for_theta_l464_464484

noncomputable def theta_range (θ : ℝ) : Prop :=
  ∀ (k : ℤ), (2 * k : ℝ) * π + π / 12 < θ ∧ θ < (2 * k : ℝ) * π + 5 * π / 12

theorem inequality_holds_for_theta (θ : ℝ) :
  (∀ x ∈ Icc (0 : ℝ) 1, x^2 * cos θ - x * (1 - x) + (1 - x)^2 * sin θ > 0) →
  theta_range θ :=
sorry

end inequality_holds_for_theta_l464_464484


namespace three_digit_multiples_of_30_not_75_count_l464_464093

theorem three_digit_multiples_of_30_not_75_count : 
  let nums := {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 30 = 0}
  let multiples_of_75 := {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n % 75 = 0}
  let valid_nums := nums \ multiples_of_75
  24 = valid_nums.size :=
by
  sorry

end three_digit_multiples_of_30_not_75_count_l464_464093


namespace fold_line_length_l464_464375

theorem fold_line_length (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 13)
  (triangle : a^2 + b^2 = c^2) : 
  ∃ l : ℝ, l = Real.sqrt 7.33475 :=
by
  use Real.sqrt 7.33475
  sorry

end fold_line_length_l464_464375


namespace minimum_distance_is_2_5_l464_464137

noncomputable def minimum_distance {α : Type*} (rect : α) (points : list (ℝ × ℝ)) : ℝ :=
  let d := λ p1 p2 : ℝ × ℝ, real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)
  let distances := list.map (λ (pair : (ℝ × ℝ) × (ℝ × ℝ)), d pair.1 pair.2) (list.sublists_len 2 points)
  list.foldr min 5 distances

theorem minimum_distance_is_2_5 :
  ∀ (points : list (ℝ × ℝ)),
    list.length points = 4 →
    ∀ (x y : ℝ × ℝ),
      x ∈ list.product (list.range 4) (list.range 3) →
      y ∈ list.product (list.range 4) (list.range 3) →
      minimum_distance (3, 4) points ≤ 2.5 := 
by 
  intro points
  intro h_points_len
  intro x y
  intro h_x_in_range
  intro h_y_in_range
  sorry

end minimum_distance_is_2_5_l464_464137


namespace probability_at_least_one_turns_left_l464_464753

/-
Given:
1. Two cars crossing an intersection from south to north.
2. Each car has an equal probability (1/3) of turning left, going straight, or turning right.
3. The choices of the two cars are independent.

Prove:
The probability that at least one car turns left is 5/9.
-/

theorem probability_at_least_one_turns_left :
  let events := [("left", "left"), ("left", "straight"), ("left", "right"),
                 ("straight", "right"), ("straight", "left"), ("straight", "straight"),
                 ("right", "right"), ("right", "straight"), ("right", "left")] in
  let at_least_one_left := list.filter (fun (x : String × String) => x.fst = "left" ∨ x.snd = "left") events in
  (list.length at_least_one_left) / (list.length events) = 5 / 9 :=
by
  sorry

end probability_at_least_one_turns_left_l464_464753


namespace M_minus_N_positive_l464_464110

variable (a b : ℝ)

def M : ℝ := 10 * a^2 + b^2 - 7 * a + 8
def N : ℝ := a^2 + b^2 + 5 * a + 1

theorem M_minus_N_positive : M a b - N a b ≥ 3 := by
  sorry

end M_minus_N_positive_l464_464110


namespace trigonometric_identity_l464_464476

theorem trigonometric_identity (x : ℝ) (h : sqrt 3 * sin x + cos x = 2 / 3) : 
  tan (x + 7 * π / 6) = sqrt 2 / 4 ∨ tan (x + 7 * π / 6) = -sqrt 2 / 4 :=
by
  sorry

end trigonometric_identity_l464_464476


namespace height_of_water_tower_l464_464307

theorem height_of_water_tower (height_of_bamboo : ℝ) (shadow_of_bamboo : ℝ) (shadow_of_tower : ℝ) :
  height_of_bamboo = 2 ∧ shadow_of_bamboo = 1.5 ∧ shadow_of_tower = 24 → ∃ h : ℝ, h = 32 :=
by
  intros,
  cases a,
  use (32 : ℝ),
  sorry

end height_of_water_tower_l464_464307


namespace ls_parallel_pq_l464_464605

-- Declare the points and lines
variables {A P C M N Q L S : Type*}

-- Assume given conditions as definitions
def intersection (l1 l2 : set (Type*)) : Type* := sorry -- Define intersection point of l1 and l2 (details omitted for simplicity)

-- Define L as intersection of lines AP and CM
def L : Type* := intersection (set_of (λ x, ∃ L, ∃ P, ∃ C, ∃ M, L = x ∧ x ∈ AP ∧ x ∈ CM)) (λ x, x ∈ AP ∧ x ∈ CM)

-- Define S as intersection of lines AN and CQ
def S : Type* := intersection (set_of (λ x, ∃ S, ∃ N, ∃ C, ∃ Q, S = x ∧ x ∈ AN ∧ x ∈ CQ)) (λ x, x ∈ AN ∧ x ∈ CQ)

-- Statement to prove LS is parallel to PQ
theorem ls_parallel_pq (AP CM AN CQ PQ : set (Type*)) (L : Type*) (S : Type*) :
    L = intersection (AP) (CM) ∧ S = intersection (AN) (CQ) → is_parallel LS PQ :=
by
  sorry -- Proof omitted

end ls_parallel_pq_l464_464605


namespace y_share_per_rupee_l464_464814

theorem y_share_per_rupee (a p : ℝ) (h1 : a * p = 18)
                            (h2 : p + a * p + 0.30 * p = 70) :
    a = 0.45 :=
by 
  sorry

end y_share_per_rupee_l464_464814


namespace range_of_a_for_real_roots_l464_464129

theorem range_of_a_for_real_roots (a : ℝ) (h : a ≠ 0) :
  (∃ (x : ℝ), a*x^2 + 2*x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_for_real_roots_l464_464129


namespace large_ball_radius_l464_464312

noncomputable def r_large_ball : ℝ :=
  (12 * (4 / 3) * Real.pi * (2^3) / ((4 / 3) * Real.pi))^(1 / 3)

theorem large_ball_radius :
  r_large_ball = Real.cbrt 96 :=
by
  sorry

end large_ball_radius_l464_464312


namespace prob_of_yellow_second_l464_464839

-- Defining the probabilities based on the given conditions
def prob_white_from_X : ℚ := 5 / 8
def prob_black_from_X : ℚ := 3 / 8
def prob_yellow_from_Y : ℚ := 8 / 10
def prob_yellow_from_Z : ℚ := 3 / 7

-- Combining probabilities
def combined_prob_white_Y : ℚ := prob_white_from_X * prob_yellow_from_Y
def combined_prob_black_Z : ℚ := prob_black_from_X * prob_yellow_from_Z

-- Total probability of drawing a yellow marble in the second draw
def total_prob_yellow_second : ℚ := combined_prob_white_Y + combined_prob_black_Z

-- Proof statement
theorem prob_of_yellow_second :
  total_prob_yellow_second = 37 / 56 := 
sorry

end prob_of_yellow_second_l464_464839


namespace determine_value_of_a_l464_464899

theorem determine_value_of_a (a : ℝ) (h : 1 < a) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 
  1 ≤ (1 / 2 * x^2 - x + 3 / 2) ∧ (1 / 2 * x^2 - x + 3 / 2) ≤ a) →
  a = 3 :=
by
  sorry

end determine_value_of_a_l464_464899


namespace subsets_1000_subsets_1001_l464_464670

-- Definitions for the number of subsets whose sizes are congruent to 0, 1, and 2 modulo 3 respectively
def f_0 (n : ℕ) : ℕ := sorry
def f_1 (n : ℕ) : ℕ := sorry
def f_2 (n : ℕ) : ℕ := sorry

-- These are noncomputable because of the large numbers involved
noncomputable def value1000 : ℕ := (2^1000 / 3).floor
noncomputable def value1001 : ℕ := (2^1001 / 3).floor

-- Prove the results for n = 1000
theorem subsets_1000 :
  f_0 1000 = value1000 ∧
  f_1 1000 = value1000 ∧
  f_2 1000 = value1000 := by
  sorry

-- Prove the results for n = 1001
theorem subsets_1001 :
  f_0 1001 = value1001 + 1 ∧
  f_1 1001 = value1001 ∧
  f_2 1001 = value1001 + 1 := by
  sorry

end subsets_1000_subsets_1001_l464_464670


namespace range_of_m_for_false_proposition_l464_464942

theorem range_of_m_for_false_proposition :
  (∀ x ∈ (Set.Icc 0 (Real.pi / 4)), Real.tan x < m) → False ↔ m ≤ 1 :=
by
  sorry

end range_of_m_for_false_proposition_l464_464942


namespace floor_sqrt_18_squared_eq_16_l464_464001

theorem floor_sqrt_18_squared_eq_16 : (Int.floor (Real.sqrt 18)) ^ 2 = 16 := 
by 
  sorry

end floor_sqrt_18_squared_eq_16_l464_464001


namespace base_b_addition_correct_base_b_l464_464020

theorem base_b_addition (b : ℕ) (hb : b > 5) :
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 :=
  by
    sorry

theorem correct_base_b : ∃ (b : ℕ), b > 5 ∧ 
  (2 * b^2 + 4 * b + 3) + (1 * b^2 + 5 * b + 2) = 4 * b^2 + 1 * b + 5 ∧
  (4 + 5 = b + 1) ∧
  (2 + 1 + 1 = 4) :=
  ⟨8, 
   by decide,
   base_b_addition 8 (by decide),
   by decide,
   by decide⟩ 

end base_b_addition_correct_base_b_l464_464020


namespace Golden_State_Total_is_94_l464_464145

def points_Draymond := 12
def points_Curry := 2 * points_Draymond
def points_Kelly := 9
def points_Durant := 2 * points_Kelly
def points_Klay := points_Draymond / 2
def points_Jordan := points_Kelly * 1.2
def points_Migel := max (0) (sqrt points_Curry - 5)
def points_Green := 3 / 4 * points_Durant

def Golden_State_Total := points_Draymond + points_Curry + points_Kelly + points_Durant +
                          points_Klay + points_Jordan + points_Migel + points_Green

theorem Golden_State_Total_is_94 : Golden_State_Total = 94 := by
  sorry

end Golden_State_Total_is_94_l464_464145


namespace find_real_numbers_l464_464876

noncomputable def real_constants (a b p q : ℝ) : Prop :=
  (2 * x - 1)^20 - (a * x + b)^20 = (x^2 + p * x + q)^10

theorem find_real_numbers: 
  ∃ a b p q : ℝ, (∀ (x : ℝ), real_constants a b p q) ∧ 
  a = - (2 * (1/2) * a + b)^20 ∧ 
  b =  1/2 * sqrt[20]{2^20 - 1} ∧ 
  p = -1 ∧ 
  q = 1/4 :=
sorry

end find_real_numbers_l464_464876


namespace winning_strategy_l464_464871

-- Conditions:
-- 1. Eva and Camille take turns placing 2x1 dominos on a 2x(n) grid.
-- 2. Eva starts first.
-- 3. A player loses if they cannot place a domino.

-- Definitions:
def even (n : ℕ) := n % 2 = 0
def odd (n : ℕ) := n % 2 = 1

-- Theorem statement:
theorem winning_strategy (n : ℕ) :
  (odd n → Eva_wins n) ∧ (even n → Camille_wins n) :=
sorry

end winning_strategy_l464_464871


namespace kenny_played_basketball_last_week_l464_464581

def time_practicing_trumpet : ℕ := 40
def time_running : ℕ := time_practicing_trumpet / 2
def time_playing_basketball : ℕ := time_running / 2
def answer : ℕ := 10

theorem kenny_played_basketball_last_week :
  time_playing_basketball = answer :=
by
  -- sorry to skip the proof
  sorry

end kenny_played_basketball_last_week_l464_464581


namespace parallel_LS_PQ_l464_464627

open_locale big_operators

variables (A P M N C Q L S : Type*)

-- Assume that L is the intersection of lines AP and CM
def is_intersection_L (AP CM : Set (Set Type*)) : Prop :=
  L ∈ AP ∩ CM

-- Assume that S is the intersection of lines AN and CQ
def is_intersection_S (AN CQ : Set (Set Type*)) : Prop :=
  S ∈ AN ∩ CQ

-- Prove that LS ∥ PQ
theorem parallel_LS_PQ 
  (hL : is_intersection_L A P M C L) 
  (hS : is_intersection_S A N C Q S) : 
  parallel (line_through L S) (line_through P Q) :=
sorry

end parallel_LS_PQ_l464_464627


namespace hyperbola_eq_standard_l464_464481

-- Defining the conditions
def ellipse_foci : Prop := ∀ x y, (x^2 / 16 + y^2 / 3 = 1) → (x, y) ∉ {p : ℝ × ℝ | p = (±sqrt(13), 0)}

def asymptote_hyperbola : Prop := ∀ x y, y = (3/2) * x

noncomputable def hyperbola_standard_eq : Prop := ∃ a b, (b / a = 3/2) ∧ (a^2 + b^2 = 13) ∧ (∀ x y, x^2 / a^2 - y^2 / b^2 = 1)

-- The statement to be proven
theorem hyperbola_eq_standard : ellipse_foci ∧ asymptote_hyperbola → hyperbola_standard_eq :=
by
  sorry

end hyperbola_eq_standard_l464_464481


namespace seating_arrangements_l464_464993

theorem seating_arrangements (total_people : ℕ) (refusing_people_list : list ℕ) (total_possible_arrangements : ℕ) (consecutive_group_arrangements : ℕ) : (refusing_people_list.length = 4) ∧ (total_people = 10) → (total_possible_arrangements = 10!) ∧ (consecutive_group_arrangements = 7! * 4!) → total_possible_arrangements - consecutive_group_arrangements = 3507840 :=
by 
  intro h1 h2
  cases h1
  cases h2
  sorry

end seating_arrangements_l464_464993


namespace max_leopard_moves_l464_464442

theorem max_leopard_moves (n : ℕ) (hn : 2 ≤ n) :
  let leopard_moves := λ (move : ℤ × ℤ), move = (1, 0) ∨ move = (0, 1) ∨ move = (-1, -1)
  in
  ∃ (path : list (ℤ × ℤ)), path.length = 9 * n^2 - 3 ∧
  (∀ move ∈ path, leopard_moves move) ∧
  (∀ pos ∈ path, 0 ≤ fst pos < 3 * n ∧ 0 ≤ snd pos < 3 * n) ∧
  (list.nodup path) ∧
  (path.head = path.last ∧ path ≠ []) :=
sorry

end max_leopard_moves_l464_464442


namespace sum_of_side_lengths_in_cm_l464_464511

-- Definitions for the given conditions
def side_length_meters : ℝ := 2.3
def meters_to_centimeters : ℝ := 100
def num_sides : ℕ := 8

-- The statement to prove
theorem sum_of_side_lengths_in_cm :
  let side_length_cm := side_length_meters * meters_to_centimeters in
  let total_length_cm := side_length_cm * (num_sides : ℝ) in
  total_length_cm = 1840 :=
by
  sorry

end sum_of_side_lengths_in_cm_l464_464511


namespace length_of_train_approx_l464_464372

-- Define the given conditions
def speed_km_hr := 120 -- Speed in km/hr
def time_seconds := 15 -- Time in seconds

-- Define the conversion factor from km/hr to m/s
def km_to_m := 1000
def hr_to_s := 3600
def conversion_factor := (km_to_m : ℝ) / (hr_to_s : ℝ)

-- Define the speed in m/s
def speed_m_s := speed_km_hr * conversion_factor

-- Define the length of the train
def length_of_train := speed_m_s * time_seconds

-- The proof problem statement
theorem length_of_train_approx : length_of_train ≈ 500 := by
  sorry

end length_of_train_approx_l464_464372


namespace polynomial_integer_condition_l464_464007

theorem polynomial_integer_condition (P : ℝ → ℝ) (hP : ∀ x, is_polynomial_with_integer_coefficients P x) :
  (∀ s t : ℝ, (P s ∈ ℤ) → (P t ∈ ℤ) → (P (s * t) ∈ ℤ)) →
  (∃ (n : ℕ) (k : ℤ), P = λ x, x^n + k ∨ P = λ x, -x^n + k) :=
by
  sorry

end polynomial_integer_condition_l464_464007


namespace tangent_line_at_origin_l464_464268

noncomputable def f (x : ℝ) : ℝ := Real.exp x

theorem tangent_line_at_origin : ∀ x y : ℝ, (x = 0) → (y = f 0) → (y = 1) →
  (∃ m : ℝ, m = (f' 0) ∧ (y - f 0 = m * (x - 0)) → 
  (x - y + 1 = 0)) :=
by
  sorry

end tangent_line_at_origin_l464_464268


namespace parallel_LS_PQ_l464_464631

open_locale big_operators

variables (A P M N C Q L S : Type*)

-- Assume that L is the intersection of lines AP and CM
def is_intersection_L (AP CM : Set (Set Type*)) : Prop :=
  L ∈ AP ∩ CM

-- Assume that S is the intersection of lines AN and CQ
def is_intersection_S (AN CQ : Set (Set Type*)) : Prop :=
  S ∈ AN ∩ CQ

-- Prove that LS ∥ PQ
theorem parallel_LS_PQ 
  (hL : is_intersection_L A P M C L) 
  (hS : is_intersection_S A N C Q S) : 
  parallel (line_through L S) (line_through P Q) :=
sorry

end parallel_LS_PQ_l464_464631


namespace LS_parallel_PQ_l464_464596

noncomputable def L (A P C M : Point) : Point := 
  -- Intersection of AP and CM
  sorry

noncomputable def S (A N C Q : Point) : Point := 
  -- Intersection of AN and CQ
  sorry

theorem LS_parallel_PQ (A P C M N Q : Point) 
  (hL : L A P C M) (hS : S A N C Q) : 
  Parallel (Line (L A P C M)) (Line (S A N C Q)) :=
sorry

end LS_parallel_PQ_l464_464596


namespace ratio_min_squared_sum_ge_l464_464673

theorem ratio_min_squared_sum_ge {n : ℕ} (h : 2 ≤ n) (a : Fin n → ℝ) :
  (∑ i, a i ^ 2) / (Finset.min' (Finset.image (λ (p : Fin n × Fin n), if p.1 < p.2 then (a p.1 - a p.2) ^ 2 else (a p.2 - a p.1) ^ 2) (Finset.univ ×ˢ Finset.univ))
  sorry) ≥ n * (n ^ 2 - 1) / 12 :=
sorry

end ratio_min_squared_sum_ge_l464_464673


namespace total_cost_production_l464_464269

-- Define the fixed cost and marginal cost per product as constants
def fixedCost : ℤ := 12000
def marginalCostPerProduct : ℤ := 200
def numberOfProducts : ℤ := 20

-- Define the total cost as the sum of fixed cost and total variable cost
def totalCost : ℤ := fixedCost + (marginalCostPerProduct * numberOfProducts)

-- Prove that the total cost is equal to 16000
theorem total_cost_production : totalCost = 16000 :=
by
  sorry

end total_cost_production_l464_464269


namespace find_M_value_l464_464290

def distinct_positive_integers (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_M_value (C y M A : ℕ) 
  (h1 : distinct_positive_integers C y M A) 
  (h2 : C + y + 2 * M + A = 11) : M = 1 :=
sorry

end find_M_value_l464_464290


namespace train_speed_l464_464817

noncomputable def train_speed_kmph (L_t L_b : ℝ) (T : ℝ) : ℝ :=
  (L_t + L_b) / T * 3.6

theorem train_speed (L_t L_b : ℝ) (T : ℝ) :
  L_t = 110 ∧ L_b = 190 ∧ T = 17.998560115190784 → train_speed_kmph L_t L_b T = 60 :=
by
  intro h
  sorry

end train_speed_l464_464817


namespace find_x_l464_464346

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
by
  sorry

end find_x_l464_464346


namespace triangle_coloring_l464_464461

theorem triangle_coloring 
  (n : ℕ)
  (points : Fin n → ℝ × ℝ)
  (no_three_collinear : ∀ i j k : Fin n, i ≠ j → i ≠ k → j ≠ k → ¬Collinear ({points i, points j, points k} : Set (ℝ × ℝ)))
  (edges : Set (Fin n × Fin n))
  (colored_edges : Set (Fin n × Fin n) → Prop)
  (unique_path : ∀ i j : Fin n, i ≠ j → ∃! p : List (Fin n × Fin n), path_from_to p (points i) (points j) ∧ colored_edges p) :
  ∃ color_scheme : Set (Fin n × Fin n) → Prop, ∀ i j k : Fin n, i ≠ j → i ≠ k → j ≠ k → odd (count_red_edges (triangle_edges i j k color_scheme)) :=
sorry

end triangle_coloring_l464_464461


namespace no_solution_condition_l464_464960

theorem no_solution_condition (n : ℝ) : ¬(∃ x y z : ℝ, n^2 * x + y = 1 ∧ n * y + z = 1 ∧ x + n^2 * z = 1) ↔ n = -1 := 
by {
    sorry
}

end no_solution_condition_l464_464960


namespace monomial_degree_and_coefficient_l464_464108

theorem monomial_degree_and_coefficient (a b : ℤ) (h1 : -a = 7) (h2 : 1 + b = 4) : a + b = -4 :=
by
  sorry

end monomial_degree_and_coefficient_l464_464108


namespace angle_of_vectors_l464_464967

theorem angle_of_vectors (a b : ℝ) (angle_ab : ℝ) (h : angle_ab = 57) :
  angle (2 * -a + b) (3 * b) = 123 := by
  -- proof goes here
  sorry

end angle_of_vectors_l464_464967


namespace valid_digits_for_divisibility_by_8_and_5_l464_464782

theorem valid_digits_for_divisibility_by_8_and_5 :
  ∃ (at hash : ℕ), (at < 10) ∧ (hash < 10) ∧ (hash = 0 ∨ hash = 5) ∧
  (84 * 10 + at) * 10 + hash % 8 = 0 ∧ (84 * 10 + at) * 10 + hash % 5 = 0 ∧
  (at = 0 ∨ at = 8) ∧ hash = 0 :=
by
  sorry

end valid_digits_for_divisibility_by_8_and_5_l464_464782


namespace invertible_product_l464_464274

-- Define the conditions
def domain_f3 : Set ℤ := {-6, -5, -4, -3, -2, -1, 0, 1, 2, 3}
def f2 (x : ℝ) : ℝ := x^3 - 3*x + 1  -- example cubic polynomial with specified points
def f3 (x : ℤ) : ℤ := x + 8 -- continues linearly from the given points
def f4 (x : ℝ) : ℝ := Math.tan x  -- tangent function
def f5 (x : ℝ) : ℝ := 3 / x  -- hyperbola function

-- Definitions of invertibility
def invertible (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x
def invertible_domain (f : ℤ → ℤ) (domain : Set ℤ) : Prop := ∀ y ∈ domain, ∃ x ∈ domain, f x = y

-- Invertibility of each function
def f2_invertible : Prop := invertible f2
def f3_invertible : Prop := invertible_domain f3 domain_f3
def f4_invertible : Prop := invertible f4
def f5_invertible : Prop := invertible f5

-- The main theorem
theorem invertible_product : (¬ f2_invertible) ∧ f3_invertible ∧ f4_invertible ∧ f5_invertible → 
  3 * 4 * 5 = 60 :=
sorry

end invertible_product_l464_464274


namespace three_digit_numbers_div_by_17_l464_464103

theorem three_digit_numbers_div_by_17 : 
  let k_min := 6
  let k_max := 58
  (k_max - k_min + 1) = 53 := by
  let k_min := 6
  let k_max := 58
  show (k_max - k_min + 1) = 53 from sorry

end three_digit_numbers_div_by_17_l464_464103


namespace parallel_LS_pQ_l464_464621

universe u

variables {P Q A C M N L S T : Type u} [incidence_geometry P Q A C M N L S T] -- Customize as per specific needs in Lean

-- Assume the required points and their intersections
def intersection_L (A P C M : Type u) [incidence_geometry A P C M] : Type u := sorry
def intersection_S (A N C Q : Type u) [incidence_geometry A N C Q] : Type u := sorry

-- Defining the variables as intersections
variables (L : intersection_L A P C M)
variables (S : intersection_S A N C Q)

theorem parallel_LS_pQ (hL : intersection_L A P C M) (hS : intersection_S A N C Q) :
  parallel L S P Q :=
sorry

end parallel_LS_pQ_l464_464621


namespace triangle_inequalities_l464_464467

open Real

-- Define a structure for a triangle with its properties
structure Triangle :=
(a b c R ra rb rc : ℝ)

-- Main statement to be proved
theorem triangle_inequalities (Δ : Triangle) (h : 2 * Δ.R ≤ Δ.ra) :
  Δ.a > Δ.b ∧ Δ.a > Δ.c ∧ 2 * Δ.R > Δ.rb ∧ 2 * Δ.R > Δ.rc :=
sorry

end triangle_inequalities_l464_464467


namespace vector_length_and_cosine_l464_464086

variables (α : ℝ) (a b : ℝ × ℝ)
def a_vec := (4, 5 * Real.cos α)
def b_vec := (3, -4 * Real.tan α)
def orthogonal (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem vector_length_and_cosine
  (hα : 0 < α ∧ α < Real.pi / 2)
  (h_orth : orthogonal a_vec b_vec) :
  (Real.sqrt ((a_vec.1 + b_vec.1)^2 + (a_vec.2 + b_vec.2)^2) = 5 * Real.sqrt 2) ∧ 
  (Real.cos (α + Real.pi / 4) = Real.sqrt 2 / 10) :=
  sorry

end vector_length_and_cosine_l464_464086


namespace sin_val_l464_464893

theorem sin_val (θ : ℝ) (h : Real.tan θ = 2) : 
    Real.sin (2 * θ + π / 4) = sqrt 2 / 10 :=
by 
  sorry

end sin_val_l464_464893


namespace correct_password_probability_l464_464770

def passwordProbabilityCorrect : ℚ :=
  1 / 15

theorem correct_password_probability :
  ∀ c1 c2,
    c1 ∈ {'M', 'I', 'N'} ∧ c2 ∈ {1, 2, 3, 4, 5} →
    (∃! p : ℚ, p = passwordProbabilityCorrect) :=
by
  intros c1 c2 h
  sorry

end correct_password_probability_l464_464770


namespace isosceles_triangle_sides_l464_464016

theorem isosceles_triangle_sides (a b : ℝ) (h1 : 2 * a + a = 14 ∨ 2 * a + a = 18)
  (h2 : a + b = 18 ∨ a + b = 14) : 
  (a = 14/3 ∧ b = 40/3 ∨ a = 6 ∧ b = 8) :=
by
  sorry

end isosceles_triangle_sides_l464_464016


namespace exists_x_sum_leq_8n_sum_l464_464192

theorem exists_x_sum_leq_8n_sum (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, a i ∈ Set.Icc 0 1) :
  ∃ x ∈ Set.Ioo 0 1, ∑ i, (1 / |x - a i|) ≤ 8 * n * ∑ i in Finset.range n + 1, (1 / (2 * i - 1)) :=
sorry

end exists_x_sum_leq_8n_sum_l464_464192


namespace multiply_negatives_l464_464295

theorem multiply_negatives : (-3) * (-4) * (-1) = -12 := 
by sorry

end multiply_negatives_l464_464295


namespace triangle_congruence_l464_464172

variables {A B C D M N E F : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F]

def quadrilateral (ABCD : Prop) (AD BC : Type*) (M N : Type*) (MCBF ADME : Prop) (mid_M: M = (C + D) / 2)  (mid_N: N = (A + B) / 2) : Prop :=
  AD = BC ∧ ¬ (AD ∥ BC) ∧ ¬ (CD ∥ AB) ∧ parallelogram MCBF ∧ parallelogram ADME

theorem triangle_congruence {A B C D M N E F : Type*} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F] 
(quadrilateral : Prop) 
(AD_eq_BC : AD = BC)
(not_parallel_CD_AB : ¬ (CD ∥ AB))
(not_parallel_AD_BC : ¬ (AD ∥ BC))
(midpoint_M : M = (C + D) / 2) 
(midpoint_N : N = (A + B) / 2)
(parallelogram_MCBF : parallelogram MCBF)
(parallelogram_ADME : parallelogram ADME) : 
triangle_congruent (B F N) (A E N) :=
sorry

end triangle_congruence_l464_464172


namespace inequality_no_solution_l464_464500

-- Define the quadratic inequality.
def quadratic_ineq (m x : ℝ) : Prop :=
  (m + 1) * x^2 - m * x + (m - 1) > 0

-- Define the condition for m.
def range_of_m (m : ℝ) : Prop :=
  m ≤ - (2 * Real.sqrt 3) / 3

-- Theorem stating that if the inequality has no solution, m gets restricted.
theorem inequality_no_solution (m : ℝ) :
  (∀ x : ℝ, ¬ quadratic_ineq m x) ↔ range_of_m m :=
by sorry

end inequality_no_solution_l464_464500


namespace sum_of_log_sequence_l464_464944

noncomputable def sequence (n : ℕ) : ℝ := if n = 1 then 1/2 else 2 * sequence (n - 1)

theorem sum_of_log_sequence :
  ∑ n in finset.range 100, (1 / (Real.log 2 (sequence (n + 3)) * Real.log 2 (sequence (n + 2)))) = 100 / 101 :=
by
  sorry

end sum_of_log_sequence_l464_464944


namespace dealer_overall_gain_l464_464349

noncomputable def dealer_gain_percentage (weight1 weight2 : ℕ) (cost_price : ℕ) : ℚ :=
  let actual_weight_sold := weight1 + weight2
  let supposed_weight_sold := 1000 + 1000
  let gain_item1 := cost_price - (weight1 / 1000) * cost_price
  let gain_item2 := cost_price - (weight2 / 1000) * cost_price
  let total_gain := gain_item1 + gain_item2
  let total_actual_cost := (actual_weight_sold / 1000) * cost_price
  (total_gain / total_actual_cost) * 100

theorem dealer_overall_gain :
  dealer_gain_percentage 900 850 100 = 14.29 := 
sorry

end dealer_overall_gain_l464_464349


namespace male_to_female_cat_weight_ratio_l464_464831

variable (w_f w_m w_t : ℕ)

def female_cat_weight : Prop := w_f = 2
def total_weight : Prop := w_t = 6
def male_cat_heavier : Prop := w_m > w_f

theorem male_to_female_cat_weight_ratio
  (h_female_cat_weight : female_cat_weight w_f)
  (h_total_weight : total_weight w_t)
  (h_male_cat_heavier : male_cat_heavier w_m w_f) :
  w_m = 4 ∧ w_t = w_f + w_m ∧ (w_m / w_f) = 2 :=
by
  sorry

end male_to_female_cat_weight_ratio_l464_464831


namespace AD_tangent_to_circumcircle_of_D_B1_C1_l464_464667

-- Given Definitions and Conditions
variables {A B C I D A1 B1 C1 : Type}
variable [Incenter I]
variables (triangle_ABC : Triangle A B C) (excircle_A : Excircle A)
variables (line_BC : Line B C) (line_CA : Line C A) (line_AB : Line A B)
variables [PointsOnLine D line_BC] [PointsOnExcircleTangent A1 excircle_A line_BC]
variables [PointsOnExcircleTangent B1 (Excircle B) line_CA]
variables [PointsOnExcircleTangent C1 (Excircle C) line_AB]
variable [QuadrilateralCyclic A B1 A1 C1]
variable (angle_AID_right : Angle A I D = 90)

-- Theorem Statement
theorem AD_tangent_to_circumcircle_of_D_B1_C1 :
    Tangent (LineSegment D A) (Circumcircle (Triangle D B1 C1)) := sorry

end AD_tangent_to_circumcircle_of_D_B1_C1_l464_464667


namespace total_animal_eyes_l464_464136

theorem total_animal_eyes : 
  let frog_eyes := 20 * 2 in
  let crocodile_eyes := 6 * 2 in
  let spider_eyes := 10 * 8 in
  let cyclops_fish_eyes := 4 * 1 in
  frog_eyes + crocodile_eyes + spider_eyes + cyclops_fish_eyes = 136 := by
  sorry

end total_animal_eyes_l464_464136


namespace arithmetic_sequences_diff_l464_464477

theorem arithmetic_sequences_diff
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (d_a d_b : ℤ)
  (ha : ∀ n, a n = 3 + n * d_a)
  (hb : ∀ n, b n = -3 + n * d_b)
  (h : a 19 - b 19 = 16) :
  a 10 - b 10 = 11 := by
    sorry

end arithmetic_sequences_diff_l464_464477


namespace robin_brother_contribution_l464_464243

variable (initial_pieces : ℕ) (final_pieces : ℕ) (brother_contribution : ℕ)

theorem robin_brother_contribution :
  initial_pieces = 63 → final_pieces = 159 → brother_contribution = final_pieces - initial_pieces → brother_contribution = 96 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3
  sorry

end robin_brother_contribution_l464_464243


namespace sin_alpha_trig_expression_l464_464929

variable (x y r : ℝ)

theorem sin_alpha (h : x = 4/5) (hy : y = -3/5) (hr : r = 1) : sin α = -3/5 :=
by sorry

theorem trig_expression (h : x = 4/5) (hy : y = -3/5) (hr : r = 1) (h_sin : sin α = -3/5)
  : (sin (π / 2 - α)) / (sin (α + π)) - (tan (α - π)) / (cos (3 * π - α)) = 19 / 48 :=
by sorry

end sin_alpha_trig_expression_l464_464929


namespace triangle_inequality_l464_464567

variable {x y z : ℝ}
variable {A B C : ℝ}

theorem triangle_inequality (hA: A > 0) (hB : B > 0) (hC : C > 0) (h_sum : A + B + C = π):
  x^2 + y^2 + z^2 ≥ 2 * y * z * Real.sin A + 2 * z * x * Real.sin B - 2 * x * y * Real.cos C := by
  sorry

end triangle_inequality_l464_464567


namespace gum_distribution_l464_464169

theorem gum_distribution : 
  ∀ (John Cole Aubrey: ℕ), 
    John = 54 → 
    Cole = 45 → 
    Aubrey = 0 → 
    ((John + Cole + Aubrey) / 3) = 33 := 
by
  intros John Cole Aubrey hJohn hCole hAubrey
  sorry

end gum_distribution_l464_464169


namespace cone_sector_central_angle_l464_464901

noncomputable def base_radius := 1
noncomputable def slant_height := 2
noncomputable def circumference (r : ℝ) := 2 * Real.pi * r
noncomputable def arc_length (r : ℝ) := circumference r
noncomputable def central_angle (l : ℝ) (s : ℝ) := l / s

theorem cone_sector_central_angle : central_angle (arc_length base_radius) slant_height = Real.pi := 
by 
  -- Here we acknowledge that the proof would go, but it is left out as per instructions.
  sorry

end cone_sector_central_angle_l464_464901


namespace remainder_pow_5_mod_11_l464_464328

theorem remainder_pow_5_mod_11 :
  ((5^120 + 4) % 11) = 5 :=
by
  have h₁: 5^5 % 11 = 1 := by sorry
  have h₂: 5^120 % 11 = ((5^5)^24) % 11 := by sorry
  have h₃: (5^120 % 11) = 1 := by sorry
  calc (5^120 + 4) % 11 = (1 + 4) % 11 : by sorry
                      ... = 5 : by sorry

end remainder_pow_5_mod_11_l464_464328


namespace person_speed_in_kmph_l464_464778

noncomputable def speed_calculation (distance_meters : ℕ) (time_minutes : ℕ) : ℝ :=
  let distance_km := (distance_meters : ℝ) / 1000
  let time_hours := (time_minutes : ℝ) / 60
  distance_km / time_hours

theorem person_speed_in_kmph :
  speed_calculation 1080 12 = 5.4 :=
by
  sorry

end person_speed_in_kmph_l464_464778


namespace prove_parallel_l464_464641

-- Define points A, P, C, M, N, Q, and functions L, S.
variables {A P C M N Q L S : Point}
variables [incidence_geometry]

-- Define L and S based on the given conditions.
def L := intersection (line_through A P) (line_through C M)
def S := intersection (line_through A N) (line_through C Q)

-- Define the condition to prove LS is parallel to PQ.
theorem prove_parallel : parallel (line_through L S) (line_through P Q) :=
sorry

end prove_parallel_l464_464641


namespace car_travel_speed_l464_464791

theorem car_travel_speed (v : ℝ) : 
  (1 / 60) * 3600 + 5 = (1 / v) * 3600 → v = 65 := 
by
  intros h
  sorry

end car_travel_speed_l464_464791


namespace total_seats_l464_464829

theorem total_seats (s : ℕ) 
  (h1 : 30 + (0.20 * s : ℝ) + (0.60 * s : ℝ) = s) : s = 150 :=
  sorry

end total_seats_l464_464829


namespace distance_between_trees_l464_464981

theorem distance_between_trees
  (L : ℕ) (T : ℕ)
  (hL : L = 300)
  (hT : T = 26) :
  (L / (T - 1) = 12) :=
by {
  rw [hL, hT],
  norm_num,
  sorry
}

end distance_between_trees_l464_464981


namespace count_ordered_pairs_l464_464446

noncomputable def harmonic_mean (x y : ℕ) : ℝ :=
  (2 * x * y) / (x + y)

theorem count_ordered_pairs :
  (nat.card {p : ℕ × ℕ // let x := p.fst in
             let y := p.snd in
             x < y ∧ harmonic_mean x y = 6^10}) = 199 :=
sorry

end count_ordered_pairs_l464_464446


namespace ls_parallel_pq_l464_464607

-- Declare the points and lines
variables {A P C M N Q L S : Type*}

-- Assume given conditions as definitions
def intersection (l1 l2 : set (Type*)) : Type* := sorry -- Define intersection point of l1 and l2 (details omitted for simplicity)

-- Define L as intersection of lines AP and CM
def L : Type* := intersection (set_of (λ x, ∃ L, ∃ P, ∃ C, ∃ M, L = x ∧ x ∈ AP ∧ x ∈ CM)) (λ x, x ∈ AP ∧ x ∈ CM)

-- Define S as intersection of lines AN and CQ
def S : Type* := intersection (set_of (λ x, ∃ S, ∃ N, ∃ C, ∃ Q, S = x ∧ x ∈ AN ∧ x ∈ CQ)) (λ x, x ∈ AN ∧ x ∈ CQ)

-- Statement to prove LS is parallel to PQ
theorem ls_parallel_pq (AP CM AN CQ PQ : set (Type*)) (L : Type*) (S : Type*) :
    L = intersection (AP) (CM) ∧ S = intersection (AN) (CQ) → is_parallel LS PQ :=
by
  sorry -- Proof omitted

end ls_parallel_pq_l464_464607


namespace proof_problem_l464_464661

-- Given conditions
variables {a b : Type}  -- Two non-coincident lines
variables {α β : Type}  -- Two non-coincident planes

-- Definitions of the relationships
def is_parallel_to (x y : Type) : Prop := sorry  -- Parallel relationship
def is_perpendicular_to (x y : Type) : Prop := sorry  -- Perpendicular relationship

-- Statements to verify
def statement1 (a α b : Type) : Prop := 
  (is_parallel_to a α ∧ is_parallel_to b α) → is_parallel_to a b

def statement2 (a α β : Type) : Prop :=
  (is_perpendicular_to a α ∧ is_perpendicular_to a β) → is_parallel_to α β

def statement3 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ l : Type, is_perpendicular_to l α ∧ is_parallel_to l β

def statement4 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ γ : Type, is_perpendicular_to γ α ∧ is_perpendicular_to γ β

-- Proof problem: verifying which statements are true.
theorem proof_problem :
  ¬ (statement1 a α b) ∧ statement2 a α β ∧ statement3 α β ∧ statement4 α β :=
by
  sorry

end proof_problem_l464_464661


namespace oranges_total_revenue_l464_464298

/--
There are 10 bags on a truck, each containing 30 oranges. Four bags contain 10% rotten oranges,
three bags contain 20% rotten oranges, and the remaining three bags contain 5% rotten oranges.
Out of the good oranges, 70 will be kept for making orange juice, 15 will be used for making jams,
and the rest will be sold. Each bag of oranges costs $10 and the transportation cost for each bag
is $2, while the selling price for each good orange is $0.50. Prove that the total revenue from 
selling the rest of the good oranges is $90.
-/
theorem oranges_total_revenue :
  let num_bags := 10
  let oranges_per_bag := 30
  let rotten_10_percent_bags := 4
  let rotten_20_percent_bags := 3
  let rotten_5_percent_bags := 3
  let juice_oranges := 70
  let jam_oranges := 15
  let selling_price := 0.5
  let total_oranges := num_bags * oranges_per_bag
  let rotten_oranges := 
        (rotten_10_percent_bags * oranges_per_bag * 0.1) + 
        (rotten_20_percent_bags * oranges_per_bag * 0.2) +
        (rotten_5_percent_bags * oranges_per_bag * 0.05)
  let good_oranges := total_oranges - rotten_oranges
  let good_oranges_for_sale := good_oranges - juice_oranges - jam_oranges
  let total_revenue := good_oranges_for_sale * selling_price
  in total_revenue = 90 := 
by
  sorry

end oranges_total_revenue_l464_464298


namespace solve_for_x_l464_464715

theorem solve_for_x (x : ℝ) (h1 : 5^(3 * x) = Real.sqrt 125) :
  x = 1/2 :=
by
  -- Assuming necessary equivalences for simplifications
  have h2 : Real.sqrt 125 = 125^(1/2) := sorry,
  have h3 : 125 = 5^3 := sorry,
  have h4 : 125^(1/2) = (5^3)^(1/2) := by rw [h3],
  have h5 : (5^3)^(1/2) = 5^(3/2) := Real.rpow_mul 5 3 (1/2),
  rw [h2, h4, h5] at h1,
  -- Solve exponents after simplifying equal bases
  have h6 : 5^(3 * x) = 5^(3/2) := h1,
  have h7 : 3 * x = 3 / 2 := sorry,
  field_simp at h7,
  exact h7,
  sorry

end solve_for_x_l464_464715


namespace mix_ratios_l464_464338

theorem mix_ratios (milk1 water1 milk2 water2 : ℕ) 
  (h1 : milk1 = 7) (h2 : water1 = 2)
  (h3 : milk2 = 8) (h4 : water2 = 1) :
  (milk1 + milk2) / (water1 + water2) = 5 :=
by
  -- Proof required here
  sorry

end mix_ratios_l464_464338


namespace sum_bn_2999_l464_464026

def b_n (n : ℕ) : ℕ :=
  if n % 17 = 0 ∧ n % 19 = 0 then 15
  else if n % 19 = 0 ∧ n % 13 = 0 then 18
  else if n % 13 = 0 ∧ n % 17 = 0 then 17
  else 0

theorem sum_bn_2999 : (Finset.range 3000).sum b_n = 572 := by
  sorry

end sum_bn_2999_l464_464026


namespace sum_of_subsets_with_3_elements_l464_464587

open Finset

def P : Finset ℕ := {1, 3, 5, 7}

def subsets_of_P_with_3_elements : Finset (Finset ℕ) := P.powerset.filter (λ s, s.card = 3)

def sum_of_elements (s : Finset ℕ) : ℕ := s.sum id

def sum_of_sums_of_subsets (s : Finset (Finset ℕ)) : ℕ := s.sum sum_of_elements

theorem sum_of_subsets_with_3_elements :
  sum_of_sums_of_subsets subsets_of_P_with_3_elements = 48 :=
by
  sorry

end sum_of_subsets_with_3_elements_l464_464587


namespace gum_sharing_l464_464166

theorem gum_sharing (john cole aubrey : ℕ) (sharing_people : ℕ) 
  (hj : john = 54) (hc : cole = 45) (ha : aubrey = 0) 
  (hs : sharing_people = 3) : 
  john + cole + aubrey = 99 ∧ (john + cole + aubrey) / sharing_people = 33 := 
by
  sorry

end gum_sharing_l464_464166


namespace cube_cross_sectional_area_range_l464_464834

def edge_length : ℝ := 1

def BD₁_dist : ℝ := Real.sqrt 3

def area_range (α : Real → Real → Prop) : Prop :=
  let lower_bound := Real.sqrt 6 / 4
  let upper_bound := Real.sqrt 2
  ∀ (a : ℝ), α a → a ∈ Set.Icc lower_bound upper_bound

theorem cube_cross_sectional_area_range :
  let α := λ (a b : ℝ), b = edge_length ∧ a = BD₁_dist in
  area_range α :=
sorry

end cube_cross_sectional_area_range_l464_464834


namespace find_a_l464_464920

theorem find_a (a : ℝ) (h : a > 0) (h_range : ∀ x : ℝ, x ∈ Icc a (2 * a) → (8 / x) ∈ Icc (a / 4) 2) : a = 4 :=
sorry

end find_a_l464_464920


namespace chocolate_bars_cases_l464_464726

theorem chocolate_bars_cases :
  ∃ C : ℕ, C + 55 = 80 ∧ C = 25 :=
by
  use 25
  split
  · exact Nat.add_sub_cancel_left 55 25
  · rfl
  sorry

end chocolate_bars_cases_l464_464726


namespace solve_equation1_solve_equation2_l464_464251

def equation1 (x : ℝ) : Prop := 3 * x^2 + 2 * x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 2) * (x - 1) = 2 - 2 * x

theorem solve_equation1 :
  (equation1 (-1) ∨ equation1 (1 / 3)) ∧ 
  (∀ x, equation1 x → x = -1 ∨ x = 1 / 3) :=
sorry

theorem solve_equation2 :
  (equation2 1 ∨ equation2 (-4)) ∧ 
  (∀ x, equation2 x → x = 1 ∨ x = -4) :=
sorry

end solve_equation1_solve_equation2_l464_464251


namespace central_cell_value_l464_464148

-- Define the grid and corner cell sum condition
variables (a b c d e f g h i : ℕ)

-- Hypotheses based on the problem conditions
axiom all_numbers : {a, b, c, d, e, f, g, h, i} = {1, 2, 3, 4, 5, 6, 7, 8, 9}
axiom consecutive_adjacent (x y : ℕ) : x + 1 = y → (x, y) ∈ { (a, b), (b, c), (d, e), (e, f), (g, h), (h, i), (a, d), (d, g), (b, e), (e, h), (c, f), (f, i) }
axiom sum_corners : a + c + g + i = 18

-- Conclusion to be proven
theorem central_cell_value : e = 7 :=
sorry

end central_cell_value_l464_464148


namespace find_AE_length_l464_464553

-- Define the problem in Lean terms
noncomputable def AE_length_in_rectangle (AB BC : ℝ) (θ : ℝ) (E_on_CD : Prop) : Prop := 
  let AE := 6 * Real.sqrt 6 in
  rectangle ABCDE AB BC θ E_on_CD -> AE = 6 * Real.sqrt 6

-- Define what rectangle means with the given conditions (using implicit assumptions)
structure rectangle (ABCD : Type) (AB BC : ℝ) (θ : ℝ) (E_on_CD : Prop) :=
  (AB_length : AB = 24)
  (BC_length : BC = 12)
  (theta_angle : θ = Real.pi / 6) -- π / 6 radians = 30 degrees
  (E_on_CD_condition : E_on_CD)

-- The main theorem to state
theorem find_AE_length :
  ∀ (AB BC : ℝ) (θ : ℝ) (E_on_CD : Prop),
    rectangle ABCD AB BC θ E_on_CD → (6 * Real.sqrt 6 = AE_length_in_rectangle AB BC θ E_on_CD) :=
by
  intros AB BC θ E_on_CD hrectangle
  -- Proof to be constructed later
  sorry

end find_AE_length_l464_464553


namespace solution_set_for_f_gt_0_l464_464065

noncomputable def f (x : ℝ) : ℝ := sorry

theorem solution_set_for_f_gt_0
  (odd_f : ∀ x : ℝ, f (-x) = -f x)
  (f_one_eq_zero : f 1 = 0)
  (ineq_f : ∀ x : ℝ, x > 0 → (x * (deriv^[2] f x) - f x) / x^2 > 0) :
  { x : ℝ | f x > 0 } = { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
sorry

end solution_set_for_f_gt_0_l464_464065


namespace trigonometric_quadrant_l464_464037

theorem trigonometric_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.sin α > 0) : 
  (π / 2 < α) ∧ (α < π) :=
by
  sorry

end trigonometric_quadrant_l464_464037


namespace exists_sin_cos_gt_m_l464_464539

theorem exists_sin_cos_gt_m (m : ℝ) : (∃ x : ℝ, sin x * cos x > m) → m < (1/2) := by
  sorry

end exists_sin_cos_gt_m_l464_464539


namespace tokens_never_equal_l464_464987

theorem tokens_never_equal (G R : ℤ) (k : ℕ) :
  (G = 1) → (R = 0) →
  (∀ n, (Gₙ : ℤ) = G + 5 * n - n) →
  (∀ n, (Rₙ : ℤ) = R - 5 * n + 5 * n) →
  ¬ ∃ n, Gₙ = Rₙ :=
by
  intros hG hR hGn hRn
  sorry

end tokens_never_equal_l464_464987


namespace count_multiples_of_34_with_two_odd_divisors_l464_464430

-- Define a predicate to check if a number has exactly 2 odd natural divisors
def has_exactly_two_odd_divisors (n : ℕ) : Prop :=
  (∃ d1 d2 : ℕ, d1 ≠ d2 ∧ d1 * d2 = n ∧ d1 % 2 = 1 ∧ d2 % 2 = 1) ∧
  (∀ d : ℕ, d ∣ n → (d % 2 = 0 ∨ d = d1 ∨ d = d2))

-- Main theorem to prove the number of integers that satisfy the given conditions
theorem count_multiples_of_34_with_two_odd_divisors : 
  let valid_numbers := {n : ℕ | n ≤ 3400 ∧ n % 34 = 0 ∧ has_exactly_two_odd_divisors n} in
  valid_numbers.to_finset.card = 6 :=
by
  sorry

end count_multiples_of_34_with_two_odd_divisors_l464_464430


namespace determine_d_l464_464249

theorem determine_d :
  ∃ d : ℝ, 
    let total_area := 6 in 
    let half_area := total_area / 2 in
    let triangle_area := 4 - d in
    (triangle_area = half_area) ↔ (d = 1) :=
by
  sorry

end determine_d_l464_464249


namespace point_on_hyperbola_probability_l464_464229

theorem point_on_hyperbola_probability :
  let s := ({1, 2, 3} : Finset ℕ) in
  let p := ∑ x in s.sigma (λ x, s.filter (λ y, y ≠ x)),
             if (∃ m n, x = (m, n) ∧ n = (6 / m)) then 1 else 0 in
  p / (s.card * (s.card - 1)) = (1 / 3) :=
by
  -- Conditions and setup
  let s := ({1, 2, 3} : Finset ℕ)
  let t := s.sigma (λ x, s.filter (λ y, y ≠ x))
  let p := t.filter (λ (xy : ℕ × ℕ), xy.snd = 6 / xy.fst)
  have h_total : t.card = 6, by sorry
  have h_count : p.card = 2, by sorry

  -- Calculate probability
  calc
    ↑(p.card) / ↑(t.card) = 2 / 6 : by sorry
    ... = 1 / 3 : by norm_num

end point_on_hyperbola_probability_l464_464229


namespace irrationals_only_C_l464_464332

-- Define the rationality of numbers
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define the given numbers
def A := 1 / 2
def B := 0
def C := (2 / 3 : ℝ) * Real.pi
def D := -3

-- Prove that the only irrational number among A, B, C, D is C.
theorem irrationals_only_C (hA : ¬ is_rational A) (hB : ¬ is_rational B) (hD : ¬ is_rational D) :
  ¬ is_rational C := sorry

end irrationals_only_C_l464_464332


namespace BT_perpendicular_IE_l464_464660

-- Define the geometric elements and setup the conditions
variables (A B C D E M O T I : Type) [plane A B C]
variables [is_acute_angled_triangle A B C]
variables [is_circumcenter O A B C]
variables [is_circumcenter T A C]
variables [is_midpoint M A C]
variables [on_segment D A B]
variables [on_segment E B C]
variables [angle_diff_eq BD M BE M ABC]

-- Prove the desired perpendicularity property
theorem BT_perpendicular_IE : is_perpendicular BT IE :=
sorry

end BT_perpendicular_IE_l464_464660


namespace construct_transformed_graphs_l464_464339

def f(x : ℝ) := x^2
def g(x : ℝ) := x^3
def h(x : ℝ) := abs x

-- Using these definitions to prove transformed functions
theorem construct_transformed_graphs (x : ℝ) :
  (f(x) - 2 = (x^2 - 2)) ∧
  (g(x) - 2 = (x^3 - 2)) ∧
  (h(x) - 2 = (abs(x) - 2)) ∧
  (-f(x) = -x^2) ∧
  (-g(x) = -x^3) ∧
  (-h(x) = -abs(x)) ∧
  (1 - f(x) = 1 - x^2) ∧
  (1 - g(x) = 1 - x^3) ∧
  (abs(1 - f(x)) = abs(1 - x^2)) 
  := by
  sorry

end construct_transformed_graphs_l464_464339


namespace intersection_eq_l464_464474

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x ∈ ℕ | (x + 1 : ℤ) / (x - 2 : ℤ) ≤ 0 }

theorem intersection_eq {A B : Set ℕ} (hA : A = {0, 1, 2}) 
  (hB : B = {x ∈ ℕ | (x + 1 : ℤ) / (x - 2 : ℤ) ≤ 0 }) : 
  A ∩ B = {0, 1} := 
sorry

end intersection_eq_l464_464474


namespace perimeter_inequality_l464_464321

-- Define the points A, B, C, D, E, and E'
variables A B C D E E' : Type

-- Assume distances DE' = DE and DB = DC
variables (dE' dE dB dC : ℝ)
variables (θ1 θ2 : ℝ) 

-- The given conditions
axiom h1 : dE' = dE
axiom h2 : dB = dC
axiom h3 : θ1 = θ2

-- The distances AE' and E'B such that AE' + E'B > AB
variables (aE' e'B aB : ℝ)

-- Declare the main theorem we want to prove
theorem perimeter_inequality
  (h1 : dE' = dE)
  (h2 : dB = dC)
  (h3 : θ1 = θ2)
  : aE' + e'B > aB := 
sorry

end perimeter_inequality_l464_464321


namespace ls_parallel_pq_l464_464604

-- Declare the points and lines
variables {A P C M N Q L S : Type*}

-- Assume given conditions as definitions
def intersection (l1 l2 : set (Type*)) : Type* := sorry -- Define intersection point of l1 and l2 (details omitted for simplicity)

-- Define L as intersection of lines AP and CM
def L : Type* := intersection (set_of (λ x, ∃ L, ∃ P, ∃ C, ∃ M, L = x ∧ x ∈ AP ∧ x ∈ CM)) (λ x, x ∈ AP ∧ x ∈ CM)

-- Define S as intersection of lines AN and CQ
def S : Type* := intersection (set_of (λ x, ∃ S, ∃ N, ∃ C, ∃ Q, S = x ∧ x ∈ AN ∧ x ∈ CQ)) (λ x, x ∈ AN ∧ x ∈ CQ)

-- Statement to prove LS is parallel to PQ
theorem ls_parallel_pq (AP CM AN CQ PQ : set (Type*)) (L : Type*) (S : Type*) :
    L = intersection (AP) (CM) ∧ S = intersection (AN) (CQ) → is_parallel LS PQ :=
by
  sorry -- Proof omitted

end ls_parallel_pq_l464_464604


namespace length_Carol_is_12_l464_464849

-- Define the conditions
variable (width_Carol : ℝ) -- Width of Carol's rectangle
variable (length_Jordan : ℝ) -- Length of Jordan's rectangle
variable (width_Jordan : ℝ) -- Width of Jordan's rectangle
variable (equal_areas : Prop) -- The rectangles have equal areas

-- Provide the given values
def carol_width_value : width_Carol = 15 := by sorry
def jordan_length_value : length_Jordan = 9 := by sorry
def jordan_width_value : width_Jordan = 20 := by sorry
def equal_areas_def : equal_areas = (width_Carol * (180 / width_Carol) = length_Jordan * width_Jordan) := 
begin
  sorry
end

theorem length_Carol_is_12 :
  equal_areas →
  width_Carol = 15 →
  length_Jordan = 9 →
  width_Jordan = 20 →
  (180 / width_Carol) = 12 :=
begin
  intros,
  sorry
end

end length_Carol_is_12_l464_464849


namespace monomial_sum_l464_464292

theorem monomial_sum (m n : ℤ) (h1 : n - 1 = 4) (h2 : m - 1 = 2) : m - 2 * n = -7 := by
  sorry

end monomial_sum_l464_464292


namespace arthur_pages_to_read_l464_464832

theorem arthur_pages_to_read (P Q : ℕ) (hP : P = 500) (hQ : Q = 1000)
  (h1 : 0.80 * P = 400) (h2 : (1/5:ℚ) * Q = 200) (h3 : 200 = 200) : 
  (400 + 200 + 200 = 800) :=
by
  sorry

end arthur_pages_to_read_l464_464832


namespace lenny_jump_steps_difference_l464_464034

theorem lenny_jump_steps_difference : 
  (let
    number_of_markers  := 51,
    total_distance_ft  := 10560,
    steps_between_markers := 70,
    jumps_between_markers := 22,
    number_of_gaps := number_of_markers - 1,
    total_steps := steps_between_markers * number_of_gaps,
    total_jumps := jumps_between_markers * number_of_gaps,
    ginny_step_length := total_distance_ft / total_steps,
    lenny_jump_length := total_distance_ft / total_jumps
  in 
    int ((lenny_jump_length - ginny_step_length).round) = 7) := 
by
  sorry

end lenny_jump_steps_difference_l464_464034


namespace max_value_sqrt_expr_l464_464194

theorem max_value_sqrt_expr (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) (hc : 0 ≤ c) (hc1 : c ≤ 1) :
  (Real.cbrt (a^2 * b^2 * c^2) + Real.cbrt ((1 - a^2) * (1 - b^2) * (1 - c^2)) ≤ 1) :=
sorry

end max_value_sqrt_expr_l464_464194


namespace sum_of_common_ratios_l464_464675

variables {k p r : ℝ}
variables {a_2 a_3 b_2 b_3 : ℝ}

def geometric_seq1 (k p : ℝ) := a_2 = k * p ∧ a_3 = k * p^2
def geometric_seq2 (k r : ℝ) := b_2 = k * r ∧ b_3 = k * r^2

theorem sum_of_common_ratios (h1 : geometric_seq1 k p) (h2 : geometric_seq2 k r)
  (h3 : p ≠ r) (h4 : a_3 - b_3 = 4 * (a_2 - b_2)) : p + r = 4 :=
by sorry

end sum_of_common_ratios_l464_464675


namespace initial_value_exists_l464_464762

theorem initial_value_exists (x : ℕ) (h : ∃ k : ℕ, x + 7 = k * 456) : x = 449 :=
sorry

end initial_value_exists_l464_464762


namespace prism_properties_l464_464333

-- Assume we have a prism with the following properties:
variables (Prism : Type)
variables (Base1 Base2 : Set Prism)
variables (LateralFaces : Set (Set Prism))
variables (LateralEdges : Set Prism)

-- Conditions:
axiom lateral_edges_equal : ∀ (e1 e2 ∈ LateralEdges), e1 = e2
axiom lateral_faces_parallelograms : ∀ (f ∈ LateralFaces), ∃ (parallelogram : Set Prism), f = parallelogram
axiom base_faces_congruent : Base1 ≅ Base2
axiom cross_section_congruent_to_base : ∀ (cross_section : Set Prism), (cross_section || Base1) → cross_section ≅ Base1
axiom cross_section_through_non_adjacent_edges :
  ∀ (e1 e2 ∈ LateralEdges) (e1 ≠ e2), (cross_section_with_edges e1 e2) → ∃ (parallelogram : Set Prism), cross_section_with_edges e1 e2 = parallelogram

-- Definition for the cross-section function:
def cross_section_with_edges : Prism → Prism → Set Prism := sorry

-- Theorem to prove all conditions are satisfied
theorem prism_properties :
  (∀ (e1 e2 ∈ LateralEdges), e1 = e2) ∧ 
  (∀ (f ∈ LateralFaces), ∃ (parallelogram : Set Prism), f = parallelogram) ∧ 
  (Base1 ≅ Base2) ∧ 
  (∀ (cross_section : Set Prism), (cross_section || Base1) → cross_section ≅ Base1) ∧
  (∀ (e1 e2 ∈ LateralEdges) (e1 ≠ e2), (cross_section_with_edges e1 e2) → ∃ (parallelogram : Set Prism), cross_section_with_edges e1 e2 = parallelogram) := sorry

end prism_properties_l464_464333


namespace find_theta_l464_464874

theorem find_theta (θ : ℝ) :
  (0 : ℝ) ≤ θ ∧ θ ≤ 2 * Real.pi →
  (∀ x, (0 : ℝ) ≤ x ∧ x ≤ 2 →
    x^2 * Real.cos θ - 2 * x * (1 - x) + (2 - x)^2 * Real.sin θ > 0) →
  (Real.pi / 12 < θ ∧ θ < 5 * Real.pi / 12) :=
by
  intros hθ hx
  sorry

end find_theta_l464_464874


namespace Triangle_ABC_90_degree_l464_464564

theorem Triangle_ABC_90_degree {A B C D E F : Type*} [triangle A B C]
  (h1 : length AB = 3 * (length AC))
  (h2 : ∠BAE = ∠ACD)
  (h3 : ∠CFE = 90°) :
  ∠ACB = 90° :=
sorry

end Triangle_ABC_90_degree_l464_464564


namespace problem1_problem2_l464_464848

theorem problem1 : 12 - (-18) + (-7) + (-15) = 8 :=
by sorry

theorem problem2 : (-1)^7 * 2 + (-3)^2 / 9 = -1 :=
by sorry

end problem1_problem2_l464_464848


namespace train_pass_time_l464_464819

theorem train_pass_time (length_train : ℕ) (length_bridge : ℕ) (speed_kmh : ℕ) 
  (length_train = 592) (length_bridge = 253) (speed_kmh = 36) : 
  (length_train + length_bridge) / (speed_kmh * 1000 / 3600) = 84.5 := 
by
  sorry

end train_pass_time_l464_464819


namespace mork_tax_rate_l464_464201

variables (M : ℝ) (r : ℝ) (combined_tax_rate : ℝ)

-- Given Conditions
def tax_rate_mindy : ℝ := 0.20
def income_mindy : ℝ := 4 * M
def combined_tax_paid := combined_tax_rate * (M + income_mindy)
def tax_paid_mork := r * M
def tax_paid_mindy := tax_rate_mindy * income_mindy

-- Theorem statement to prove Mork's tax rate is 45%
theorem mork_tax_rate :
  (combined_tax_rate = 0.25) →
  (tax_paid_mork + tax_paid_mindy = combined_tax_paid) →
  r = 0.45 :=
by
  intro h_combined_tax_rate h_tax_equality
  sorry

end mork_tax_rate_l464_464201


namespace part_a_part_b_l464_464377

-- Given conditions for both parts of the problem
def choices (n : ℕ) : list (fin n) := list.fin_range n

-- Part (a)
-- The probability that the bags end up in reverse order in the shed
theorem part_a (n : ℕ) (hn : n = 4) : 
  probability_reverse_order hn = 1 / 8 :=
sorry

-- Part (b)
-- The probability that the second-from-bottom bag in the truck ends up as the bottom bag in the shed
theorem part_b (n : ℕ) (hn : n = 4) : 
  probability_second_from_bottom hn = 1 / 8 :=
sorry

-- Definitions of probability calculations used in the conditions
noncomputable def probability_reverse_order (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

noncomputable def probability_second_from_bottom (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

end part_a_part_b_l464_464377


namespace complex_conjugate_power_l464_464679

-- Given a complex number w such that its magnitude is 7
variable (w : ℂ) (hw : complex.abs w = 7)

-- We need to show that (w * complex.conj w)^3 = 117649
theorem complex_conjugate_power :
  (w * complex.conj w)^3 = 117649 :=
by
  sorry

end complex_conjugate_power_l464_464679


namespace reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464395

-- Probability that the bags end up in the shed in the reverse order
def probability_reverse_order : ℕ → ℚ
| 0 := 1
| (n + 1) := (1 / 2) * probability_reverse_order n

theorem reverse_order_probability_is_1_over_8 :
  probability_reverse_order 3 = 1 / 8 :=
sorry

-- Probability that the second bag from the bottom ends up as the bottom bag in the shed
def probability_second_from_bottom_bottom : ℚ := 
(1 / 2) * (1 / 2) * (1 / 2)

theorem second_from_bottom_as_bottom_is_1_over_8 :
  probability_second_from_bottom_bottom = 1 / 8 :=
sorry

end reverse_order_probability_is_1_over_8_second_from_bottom_as_bottom_is_1_over_8_l464_464395


namespace part_a_part_b_l464_464376

-- Given conditions for both parts of the problem
def choices (n : ℕ) : list (fin n) := list.fin_range n

-- Part (a)
-- The probability that the bags end up in reverse order in the shed
theorem part_a (n : ℕ) (hn : n = 4) : 
  probability_reverse_order hn = 1 / 8 :=
sorry

-- Part (b)
-- The probability that the second-from-bottom bag in the truck ends up as the bottom bag in the shed
theorem part_b (n : ℕ) (hn : n = 4) : 
  probability_second_from_bottom hn = 1 / 8 :=
sorry

-- Definitions of probability calculations used in the conditions
noncomputable def probability_reverse_order (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

noncomputable def probability_second_from_bottom (n : ℕ) : ℝ :=
1 / (2 ^ (n - 1))

end part_a_part_b_l464_464376


namespace imaginary_part_of_z_is_zero_l464_464040

-- Given definition of complex numbers and the imaginary unit i
def i : ℂ := Complex.I
def z : ℂ := (1 - i) / (i * (1 + i))

-- Statement to prove: the imaginary part of z is 0.
theorem imaginary_part_of_z_is_zero : Complex.im z = 0 := sorry

end imaginary_part_of_z_is_zero_l464_464040


namespace at_least_one_gt_one_l464_464701

theorem at_least_one_gt_one (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : x > 1 ∨ y > 1 :=
sorry

end at_least_one_gt_one_l464_464701


namespace no_zeros_in_intervals_l464_464196

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x * Real.log x

theorem no_zeros_in_intervals :
  (∀ x : ℝ, x ∈ Ioo (1 / Real.exp 1) 1 → f x ≠ 0) ∧
  (∀ x : ℝ, x ∈ Ioo 1 (Real.exp 1) → f x ≠ 0) :=
by
  split
  { intros x hx
    dsimp [f]
    have H : Real.log x ≠ 0
    from sorry
    sorry }
  { intros x hx
    dsimp [f]
    have H : Real.log x ≠ 0
    from sorry
    sorry }

end no_zeros_in_intervals_l464_464196


namespace probability_three_odd_numbers_l464_464827

theorem probability_three_odd_numbers (n : ℕ) (p : ℕ → ℕ)
  (h_n : n = 8)
  (h_p : p odd = 4) (h_p even = 4) :
  let prob_odd := 4 / 8, prob_even := 4 / 8 in
  let comb := @Nat.choose 5 3 in
  let prob_exactly_three_odds := comb * prob_odd^3 * prob_even^2 in
  prob_exactly_three_odds = 5 / 16 :=
by
  sorry

end probability_three_odd_numbers_l464_464827
