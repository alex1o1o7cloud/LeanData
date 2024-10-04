import Mathlib

namespace car_speed_second_hour_l594_594067

theorem car_speed_second_hour
  (v1 : ℕ) (avg_speed : ℕ) (time : ℕ) (v2 : ℕ)
  (h1 : v1 = 90)
  (h2 : avg_speed = 70)
  (h3 : time = 2) :
  v2 = 50 :=
by
  sorry

end car_speed_second_hour_l594_594067


namespace coefficient_of_x10_in_expansion_l594_594865

theorem coefficient_of_x10_in_expansion : 
  let a := (λ x : ℝ, x^3 / 3)
  let b := (λ x : ℝ, -3 / x^2)
  let expr := (λ x : ℝ, (a x + b x)^10)
  (∃ c, c = 17010 / 729 ∧ ∀ x, (x^10) ∣ expr x) :=
sorry

end coefficient_of_x10_in_expansion_l594_594865


namespace expected_survivors_l594_594611

noncomputable def probability_of_survival : ℕ → ℚ
| n := if n ≤ 10 then (10 - n) / 10 else 0

def survival_probability_for_year : ℚ :=
(probability_of_survival 1) * (probability_of_survival 2) * (probability_of_survival 3) *
(probability_of_survival 4) * (probability_of_survival 5) * (probability_of_survival 6) *
(probability_of_survival 7) * (probability_of_survival 8) * (probability_of_survival 9) *
(probability_of_survival 10)

theorem expected_survivors (newborns : ℕ) : newborns = 300 → newborns * survival_probability_for_year = 0 :=
by
  intro h
  simp [h, survival_probability_for_year]
  norm_num
  sorry

end expected_survivors_l594_594611


namespace tangent_line_correct_l594_594110

noncomputable def tangent_line_equation (x_0 : ℝ) : ℝ → ℝ :=
  let y := λ x : ℝ, x^2 / 10 + 3
  let y' := λ x : ℝ, x / 5
  let y0 := y x_0
  let m := y' x_0
  λ x : ℝ, m * (x - x_0) + y0

theorem tangent_line_correct :
  ∀ x : ℝ, tangent_line_equation 2 x = (2/5) * x + (13/5) := by
  sorry

end tangent_line_correct_l594_594110


namespace Karlsson_eats_more_than_half_l594_594655

open Real

theorem Karlsson_eats_more_than_half
  (D : ℝ) (S : ℕ → ℝ)
  (a b : ℕ → ℝ)
  (cut_and_eat : ∀ n, S (n + 1) = S n - (S n * a n) / (a n + b n))
  (side_conditions : ∀ n, max (a n) (b n) ≤ D) :
  ∃ n, S n < (S 0) / 2 := sorry

end Karlsson_eats_more_than_half_l594_594655


namespace intersection_exactly_one_l594_594859

-- Definitions of the functions
def f (x : ℝ) : ℝ := 2 * log x + x^2
def g (x : ℝ) : ℝ := log (2 * x) + x

-- The theorem to prove that f and g intersect exactly at one point
theorem intersection_exactly_one : (∃! x : ℝ, f x = g x) :=
sorry

end intersection_exactly_one_l594_594859


namespace sqrt_expression_l594_594466

theorem sqrt_expression 
  (a b c : ℤ) 
  (h1 : 2 * a * b * Int.sqrt c = 12 * Int.sqrt 6) 
  (h2 : a^2 + b^2 * c = 35) 
  (h3 : sqrt 6 * sqrt 6 = 6): 
  a + b + c = 11 := 
by
  sorry

end sqrt_expression_l594_594466


namespace percentage_of_trees_cut_l594_594503

-- Define the initial and final number of trees
def initial_trees : ℕ := 400
def final_trees : ℕ := 720

-- Define the function that models the total trees after cutting and planting
def total_trees_after_cutting (P : ℝ) : ℝ :=
  initial_trees - (P / 100) * initial_trees + 5 * (P / 100) * initial_trees

-- The theorem to prove
theorem percentage_of_trees_cut (P : ℝ) (h : total_trees_after_cutting P = final_trees) : P = 20 :=
by 
  -- We need to parse the proof problem statement
  sorry

end percentage_of_trees_cut_l594_594503


namespace distance_between_trains_l594_594428

def speed_train1 : ℝ := 11 -- Speed of the first train in mph
def speed_train2 : ℝ := 31 -- Speed of the second train in mph
def time_travelled : ℝ := 8 -- Time in hours

theorem distance_between_trains : 
  (speed_train2 * time_travelled) - (speed_train1 * time_travelled) = 160 := by
  sorry

end distance_between_trains_l594_594428


namespace math_problem_l594_594556

-- Conditions and definitions
def ellipse (a b x y : ℝ) : Prop := (a > b) ∧ (b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def hyperbola (x y : ℝ) : Prop := (x^2 / 3 - y^2 = 1)
def reciprocal_eccentricity (ae he : ℝ) : Prop := ae * he = 1
def line_through_right_vertex (x y : ℝ) : Prop := (x - y - 2 = 0)

-- Lean Proof Problem
theorem math_problem (a b x y : ℝ) (ae he : ℝ) :
  ellipse a b x y ∧ hyperbola x y ∧ reciprocal_eccentricity ae he ∧ line_through_right_vertex x y →
  -- (Ⅰ) Prove the standard equation of ellipse
  (∃ (a' b' : ℝ), a' = 2 ∧ b' = 1 ∧ x^2 / 4 + y^2 = 1) ∧
  -- (Ⅱ) Prove the range of the area of △OMN
  (∀ (k m x1 x2 y1 y2 : ℝ),
    (k ≠ 0) ∧ (m ≠ 0) ∧
    (y = k * x + m) ∧ 
    (x1 + x2 = -8 * k * m / (1 + 4 * k^2)) ∧ 
    (x1 * x2 = 4 * (m^2 - 1) / (1 + 4 * k^2)) → 
    (∃ (range : set ℝ), range = Ioo 0 1)) :=
begin
  sorry
end

end math_problem_l594_594556


namespace probability_all_red_before_blue_and_green_l594_594813

theorem probability_all_red_before_blue_and_green :
  ∃ (num_red : ℕ) (num_green : ℕ) (num_blue : ℕ) (total_chips : ℕ),
    num_red = 4 ∧ num_green = 3 ∧ num_blue = 1 ∧ total_chips = num_red + num_green + num_blue ∧
    (∀ arrangement : list (fin total_chips),
      (∀ i < total_chips, arrangement.nth i ∈ [0, 1, 2]) →
      ∃ (favorable_count : ℕ),
        favorable_count = 
          (nat.choose 5 4) * (nat.factorial 3) ∧
        favorable_count / (nat.factorial total_chips) = (5 / 6720)) :=
begin
  sorry
end

end probability_all_red_before_blue_and_green_l594_594813


namespace intersect_xz_plane_at_l594_594540

noncomputable def direction_vector (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

noncomputable def parametric_line (p : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ) (t : ℝ) : ℝ × ℝ × ℝ :=
  (p.1 + d.1 * t, p.2 + d.2 * t, p.3 + d.3 * t)

theorem intersect_xz_plane_at {p1 p2 : ℝ × ℝ × ℝ} (h1 : p1 = (2, 7, 4)) (h2 : p2 = (8, 3, 9)) :
  ∃ (t : ℝ), (parametric_line p1 (direction_vector p1 p2) t).2 = 0 ∧ 
             (parametric_line p1 (direction_vector p1 p2) t) = (59/4, 0, 51/4) :=
by
  sorry

end intersect_xz_plane_at_l594_594540


namespace largest_possible_s_l594_594662

theorem largest_possible_s :
  ∃ s r : ℕ, (r ≥ s) ∧ (s ≥ 5) ∧ (122 * r - 120 * s = r * s) ∧ (s = 121) :=
by sorry

end largest_possible_s_l594_594662


namespace central_angle_chord_AB_is_90_degrees_l594_594164

-- Define the circle with center (2, 2) and radius sqrt(8)
def circleC (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 2) ^ 2 = 8

-- Define point A coordinates: (0, 0)
def pointA : ℝ × ℝ := (0, 0)

-- Define point B coordinates: (0, 4)
def pointB : ℝ × ℝ := (0, 4)

-- Define the length of chord AB
def lengthAB : ℝ := 4

-- Prove the measure of the central angle subtended by chord AB
theorem central_angle_chord_AB_is_90_degrees : ∀ (O A B : ℝ × ℝ),
  circleC O.1 O.2 ∧ A = pointA ∧ B = pointB ->
  ∠ A O B = 90 :=
by
  sorry

end central_angle_chord_AB_is_90_degrees_l594_594164


namespace solution_set_of_inequality_l594_594065

theorem solution_set_of_inequality (x : ℝ) :
  2^(x^2 - x) < 4 ↔ x ∈ set.Ioo (-1 : ℝ) (2 : ℝ) := by
  sorry

end solution_set_of_inequality_l594_594065


namespace find_side_a_l594_594308

def sides (A B C : ℝ) (a b c : ℝ) : Prop :=
  cos A = sqrt 6 / 3 ∧ b = 2 * sqrt 2 ∧ c = sqrt 3

theorem find_side_a (A B C a b c : ℝ) (h : sides A B C a b c) : a = sqrt 3 :=
by
  sorry

end find_side_a_l594_594308


namespace integer_solutions_count_l594_594959

theorem integer_solutions_count : ∃ (s : Finset ℤ), (∀ x ∈ s, x^2 - x - 2 ≤ 0) ∧ (Finset.card s = 4) :=
by
  sorry

end integer_solutions_count_l594_594959


namespace circle_radius_proof_l594_594113

theorem circle_radius_proof
  (N : ℕ)
  (n : ℕ)
  (k : ℝ)
  (r : ℕ → ℝ)
  (h1 : ∀ i, r (i + 2) - k * r (i + 1) + r i = 0)
  (λ1 λ2 : ℝ)
  (h2 : λ1 * λ2 = 1)
  (h3 : λ1^2 - k * λ1 + 1 = 0)
  (h4 : λ2^2 - k * λ2 + 1 = 0)
  (h5 : 3 * n - 2 > N) :
  r (2 * n - 1) * (r 1 + r (2 * n - 1)) = r n * (r n + r (3 * n - 2)) :=
by
  sorry

end circle_radius_proof_l594_594113


namespace correct_option_l594_594102

theorem correct_option :
  (3 * a^2 - a^2 = 2 * a^2) ∧
  (¬ (a^2 * a^3 = a^6)) ∧
  (¬ ((3 * a)^2 = 6 * a^2)) ∧
  (¬ (a^6 / a^3 = a^2)) :=
by
  -- We only need to state the theorem; the proof details are omitted per the instructions.
  sorry

end correct_option_l594_594102


namespace work_done_in_one_day_by_A_and_B_l594_594796

noncomputable def A_days : ℕ := 12
noncomputable def B_days : ℕ := A_days / 2

theorem work_done_in_one_day_by_A_and_B : 1 / (A_days : ℚ) + 1 / (B_days : ℚ) = 1 / 4 := by
  sorry

end work_done_in_one_day_by_A_and_B_l594_594796


namespace number_of_zeros_in_interval_l594_594733

-- Define the function and interval
def f (x : ℝ) : ℝ := Real.cos (3 * x + Real.pi / 6)
def interval := set.Icc (0 : ℝ) Real.pi

-- Statement of the proof problem
theorem number_of_zeros_in_interval :
  ∃ (cnt : ℕ), cnt = 3 ∧
  ∀ (x : ℝ), x ∈ interval → f x = 0 → x = Real.pi / 9 ∨ x = 4 * Real.pi / 9 ∨ x = 7 * Real.pi / 9 :=
begin
  sorry
end

end number_of_zeros_in_interval_l594_594733


namespace multiple_of_q_l594_594368

-- Definitions of the primes p and q and the subset S
variables {p q : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) (h_distinct : p ≠ q)
variable (S : Finset ℕ)

-- Defining subsets and the function serving N(S)
def is_valid_subset : Prop := S ⊆ (Finset.range p).erase 0

noncomputable def N (S : Finset ℕ) : ℕ :=
  Finset.card { x : Fin (p ^ q) // x.val % p = 0 ∧ ∀ i, i < q → (x.val / p ^ i % p) ∈ S }

-- The final theorem statement
theorem multiple_of_q (hS : is_valid_subset S) : N hS % q = 0 :=
sorry

end multiple_of_q_l594_594368


namespace cover_9_points_with_semicircle_l594_594993

theorem cover_9_points_with_semicircle:
  ∀ (T : Type) (triangle : T) (hypotenuse one unit : ℝ) (angles : set ℝ) (points : set (ℝ × ℝ)),
    (hypotenuse = 1) →
    (angles = {30 * π / 180, 60 * π / 180, π / 2}) →
    (points ⊆ triangle) →
    (|points| = 25) →
    ∃ (semicircle : set ℝ),
      (radius semicircle = 3 / 10) ∧
      (∀ (x y : ℝ), (x, y) ∈ points → dist (x, y) center semicircle ≤ radius semicircle) ∧
      (|{p ∈ points | dist (fst p, snd p) (center semicircle) ≤ 3 / 10}| ≥ 9) :=
by sorry

end cover_9_points_with_semicircle_l594_594993


namespace students_play_cricket_l594_594693

theorem students_play_cricket 
  (total_students : ℕ) (neither_play : ℕ) (play_football : ℕ) (play_both : ℕ)
  (h1 : total_students = 460) 
  (h2 : neither_play = 50) 
  (h3 : play_football = 325) 
  (h4 : play_both = 90) : 
  ∃ play_cricket : ℕ, play_cricket = 175 :=
by
  let students_at_least_one := total_students - neither_play
  let only_football := play_football - play_both
  let only_cricket := students_at_least_one - only_football - play_both
  have h5 : students_at_least_one = 410 := by { rw [h1, h2], norm_num }
  have h6 : only_football = 235 := by { rw [h3, h4], norm_num }
  have h7 : only_cricket = 85 := by { rw [h5, h6], norm_num }
  have play_cricket := only_cricket + play_both
  use play_cricket
  rw h7
  rw h4
  norm_num
  done

end students_play_cricket_l594_594693


namespace halfway_between_two_sevenths_and_four_ninths_l594_594535

theorem halfway_between_two_sevenths_and_four_ninths : (2 : ℚ) / 7 <|> (4 : ℚ) / 9 = (23 : ℚ) / 63 :=
by
-- Proofs to be provided
sorry

end halfway_between_two_sevenths_and_four_ninths_l594_594535


namespace scientific_notation_l594_594144

theorem scientific_notation (n : ℕ) (h : n = 3100000) : n = 3.1 * 10^6 := 
by {
  sorry
}

end scientific_notation_l594_594144


namespace segment_MP_PN_indeterminate_l594_594624

theorem segment_MP_PN_indeterminate 
  (A B C D M N P : Type)
  (triangle_ABC : is_acute_triangle A B C)
  (angle_bisector_A : angle_bisector A B C D)
  (circle_B : is_circle_center_radius B D M)
  (circle_C : is_circle_center_radius C D N) 
  (MN_intersects_AD_at_P : segment_intersection M N A D P) :
  ¬ (MP = PN) ∧ ¬ (MP ≠ PN) :=
sorry

end segment_MP_PN_indeterminate_l594_594624


namespace profit_percentage_l594_594601

/-- If the cost price is 81% of the selling price, then the profit percentage is approximately 23.46%. -/
theorem profit_percentage (SP CP: ℝ) (h : CP = 0.81 * SP) : 
  (SP - CP) / CP * 100 = 23.46 := 
sorry

end profit_percentage_l594_594601


namespace tracy_initial_candies_l594_594420

theorem tracy_initial_candies (x : ℕ) :
  ((x % 4 = 0) ∧ (x % 2 = 0) ∧ (x / 2 - 29 = 10)) → x = 78 :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  sorry

end tracy_initial_candies_l594_594420


namespace lattice_points_condition_l594_594345

/-- A lattice point is a point on the plane with integer coordinates. -/
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

/-- A triangle in the plane with three vertices and at least two lattice points inside. -/
structure Triangle :=
  (A B C : LatticePoint)
  (lattice_points_inside : List LatticePoint)
  (lattice_points_nonempty : lattice_points_inside.length ≥ 2)

noncomputable def exists_lattice_points (T : Triangle) : Prop :=
∃ (X Y : LatticePoint) (hX : X ∈ T.lattice_points_inside) (hY : Y ∈ T.lattice_points_inside), 
  ((∃ (V : LatticePoint), V = T.A ∨ V = T.B ∨ V = T.C ∧ ∃ (k : ℤ), (k : ℝ) * (Y.x - X.x) = (V.x - X.x) ∧ (k : ℝ) * (Y.y - X.y) = (V.y - X.y)) ∨
  (∃ (l m n : ℝ), l * (Y.x - X.x) = m * (T.A.x - T.B.x) ∧ l * (Y.y - X.y) = m * (T.A.y - T.B.y) ∨ l * (Y.x - X.x) = n * (T.B.x - T.C.x) ∧ l * (Y.y - X.y) = n * (T.B.y - T.C.y) ∨ l * (Y.x - X.x) = m * (T.C.x - T.A.x) ∧ l * (Y.y - X.y) = m * (T.C.y - T.A.y)))

theorem lattice_points_condition (T : Triangle) : exists_lattice_points T :=
sorry

end lattice_points_condition_l594_594345


namespace range_of_x_for_sqrt_l594_594053

theorem range_of_x_for_sqrt (x : ℝ) (y : ℝ) (h : y = real.sqrt (x - 2)) :
  x ≥ 2 :=
by sorry

end range_of_x_for_sqrt_l594_594053


namespace isosceles_trapezoid_inscribed_circle_radius_l594_594150

theorem isosceles_trapezoid_inscribed_circle_radius
  (a b c d m : ℝ)
  (h_isosceles : a = c ∧ b = d)
  (angle_30 : angle a b c = 30)
  (midline_eq_10 : m = 10)
  (h_leg_sum : b + d = 20)
  (height : height a b c = 5)
  (sum_of_bases : 2 * m = 20)
  (area : trapezoid_area a b c height = 50)
  (semiperimeter : (a + b + c + d) / 2 = 20) :
  (radius := (trapezoid_area a b c height) / (semiperimeter)) :
  radius = 2.5 := 
sorry

end isosceles_trapezoid_inscribed_circle_radius_l594_594150


namespace danielle_vs_charles_wage_l594_594744

-- Define variables
variables (E : ℝ)
variables (Robin Charles Danielle : ℝ)

-- Define conditions
def robin_wage : Prop := Robin = 1.30 * E
def charles_wage : Prop := Charles = 1.70 * E
def danielle_wage : Prop := Danielle = 0.715 * E

-- Define the final proof goal
theorem danielle_vs_charles_wage {E : ℝ} (h1 : robin_wage E Robin) (h2 : charles_wage E Charles) (h3 : danielle_wage E Danielle) : 
  ((Charles - Danielle) / Charles) * 100 ≈ 57.94 :=
by sorry

end danielle_vs_charles_wage_l594_594744


namespace distance_between_points_l594_594967

-- Define the line equation
def line (x : ℝ) (b : ℝ) : ℝ := 2 * x + b

-- Define the distance formula
def distance (p q r s : ℝ) : ℝ := real.sqrt ((r - p) ^ 2 + (s - q) ^ 2)

-- Define the given conditions
lemma point_q (p b : ℝ) : line p b = 2 * p + b := rfl
lemma point_s (r b : ℝ) : line r b = 2 * r + b := rfl

-- Statement of the proof problem
theorem distance_between_points (p q r s b : ℝ) (hq : q = line p b) (hs : s = line r b) : 
  distance p q r s = real.sqrt 5 * |r - p| := by
  sorry

end distance_between_points_l594_594967


namespace part1_part2_l594_594575

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin ((1 / 3) * x - (Real.pi / 6))

theorem part1 : f (5 * Real.pi / 4) = Real.sqrt 2 :=
by sorry

theorem part2 (α β : ℝ) (hαβ : 0 ≤ α ∧ α ≤ Real.pi / 2 ∧ 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h1: f (3 * α + Real.pi / 2) = 10 / 13) (h2: f (3 * β + 2 * Real.pi) = 6 / 5) :
  Real.cos (α + β) = 16 / 65 :=
by sorry

end part1_part2_l594_594575


namespace max_value_of_expression_l594_594112

theorem max_value_of_expression (x y : ℝ) 
  (h : Real.sqrt (x * y) + Real.sqrt ((1 - x) * (1 - y)) = Real.sqrt (7 * x * (1 - y)) + (Real.sqrt (y * (1 - x)) / Real.sqrt 7)) :
  x + 7 * y ≤ 57 / 8 :=
sorry

end max_value_of_expression_l594_594112


namespace sum_of_prime_factors_of_2_to_10_minus_1_equals_45_l594_594185

noncomputable def sum_of_prime_factors_of_2_to_10_minus_1 : ℕ := 2^10 - 1

theorem sum_of_prime_factors_of_2_to_10_minus_1_equals_45 :
  ∃ (factors : List ℕ), factors = [31, 3, 11] ∧ factors.all Prime ∧ factors.sum = 45 :=
by
  -- Define the prime factors
  let factors := [31, 3, 11]
  -- Assert that these are the factors of 2^10 - 1
  have : (2^10 - 1).factors = factors :=
    sorry -- Manual factorization is assumed here
  -- Assert that all elements in the list are prime
  have : factors.all Prime :=
    by simp [factors, Prime]
  -- Sum the factors
  have : factors.sum = 31 + 3 + 11 :=
    by simp [factors]
  exact Exists.intro factors ⟨rfl, this, by norm_num⟩

end sum_of_prime_factors_of_2_to_10_minus_1_equals_45_l594_594185


namespace not_lee_soccer_game_probability_l594_594050

theorem not_lee_soccer_game_probability (p : ℚ) (h : p = 5/9) :
  1 - p = 4/9 :=
by
  rw [h]
  norm_num
  sorry

end not_lee_soccer_game_probability_l594_594050


namespace arrangement_count_l594_594788

def arrangements_with_conditions 
  (boys girls : Nat) 
  (cannot_be_next_to_each_other : Bool) : Nat :=
if cannot_be_next_to_each_other then
  sorry -- The proof will go here
else
  sorry

theorem arrangement_count :
  arrangements_with_conditions 3 2 true = 72 :=
sorry

end arrangement_count_l594_594788


namespace original_purchase_price_l594_594479

theorem original_purchase_price (P S : ℝ) (hs1 : S = P + 0.25 * S) (hs2 : 5.40 = 0.80 * S - P) : P = 81 := by
  have h1 : P = 0.75 * S := by
    rw [hs1]
    linarith
  have h2 : 5.40 = 0.80 * S - 0.75 * S := by
    rw [h1] at hs2
    exact hs2
  have h3 : S = 108 := by
    linarith
  have h4 : P = 0.75 * 108 := by
    rw [h3] at h1
    exact h1
  norm_num at h4
  exact h4

end original_purchase_price_l594_594479


namespace problem_integer_pairs_l594_594563

theorem problem_integer_pairs (a b q r : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : q^2 + r = 1977) :
    (a, b) = (50, 7) ∨ (a, b) = (50, 37) ∨ (a, b) = (7, 50) ∨ (a, b) = (37, 50) :=
sorry

end problem_integer_pairs_l594_594563


namespace calculate_expression_l594_594510

-- Defining the main theorem to prove
theorem calculate_expression (a b : ℝ) : 
  3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by 
  sorry

end calculate_expression_l594_594510


namespace max_angle_ACB_l594_594422

noncomputable def max_angle_condition (Ω : set Point) (A B : Point) : set Point :=
  {C : Point | C ∈ Ω ∧ (∃ ω1 ω2 : Circle, A ∈ ω1 ∧ B ∈ ω1 ∧ ω1 ⊆ Ω ∧ C = tangency_point ω1 Ω ∨
                             A ∈ ω2 ∧ B ∈ ω2 ∧ ω2 ⊆ Ω ∧ C = tangency_point ω2 Ω)}

theorem max_angle_ACB (Ω : set Point) (A B : Point) (hA : A ∈ interior Ω) (hB : B ∈ interior Ω)
  : (∀ C ∈ Ω, ∠ A C B ≤ ∠ A (tangency_point (circle_through A B) Ω) B) :=
sorry

end max_angle_ACB_l594_594422


namespace function_condition_l594_594895

variable (f : ℝ → ℝ)
variable (C : ℝ)

theorem function_condition (x y : ℝ) : 
  (∀ x y : ℝ, f(x + y) + y ≤ f(f(f(x)))) → (∀ x : ℝ, f(x) = C - x) :=
by 
  sorry

end function_condition_l594_594895


namespace smallest_number_of_students_l594_594988

theorem smallest_number_of_students (n : ℕ) : 
  (6 * n + 2 > 40) → (∃ n, 4 * n + 2 * (n + 1) = 44) :=
 by
  intro h
  exact sorry

end smallest_number_of_students_l594_594988


namespace num_integers_abs_le_3_l594_594035

theorem num_integers_abs_le_3 : 
  ∃ (s : Finset ℤ), (∀ x ∈ s, |x| ≤ 3) ∧ s.card = 7 :=
by
  let s := { -3, -2, -1, 0, 1, 2, 3 }.to_finset
  have h1 : ∀ x ∈ s, |x| ≤ 3 := by simp
  have h2 : s.card = 7 := by simp
  exact ⟨s, h1, h2⟩

end num_integers_abs_le_3_l594_594035


namespace floor_e_eq_two_l594_594886

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end floor_e_eq_two_l594_594886


namespace largest_possible_value_of_m_l594_594040

theorem largest_possible_value_of_m : 
  ∃ (qs : List (Polynomial ℝ)), 
    (∀ q ∈ qs, ¬q.is_constant ∧ q.splits (ring_hom.id _)) ∧ 
    (Polynomial.of_real 1).factors = qs ∧ 
    qs.length = 4 :=
by
  sorry

end largest_possible_value_of_m_l594_594040


namespace complex_modulus_product_l594_594191

noncomputable def z1 : ℂ := 5 * Real.sqrt 3 - 5 * Complex.i
noncomputable def z2 : ℂ := 2 * Real.sqrt 2 + 4 * Complex.i

theorem complex_modulus_product :
  Complex.abs (z1 * z2) = 20 * Real.sqrt 6 :=
  sorry

end complex_modulus_product_l594_594191


namespace polygon_sides_l594_594489

theorem polygon_sides (n : ℕ) (h : 144 * n = 180 * (n - 2)) : n = 10 :=
by { sorry }

end polygon_sides_l594_594489


namespace problem_statement_l594_594331

-- Definitions based on conditions in the problem.
def ω : ℂ := sorry
def α : ℂ := ω + ω^3 + ω^5
def β : ℂ := ω^2 + ω^4 + ω^6 + ω^7

-- Main statement
theorem problem_statement 
  (h₁ : ω^8 = 1)
  (h₂ : ω ≠ 1) 
  (α_def : α = ω + ω^3 + ω^5)
  (β_def : β = ω^2 + ω^4 + ω^6 + ω^7)
  : ∃ a b : ℝ, a = 1 ∧ b = 2 ∧ ∀ x : ℂ, x^2 + a * x + b = 0 := 
begin
  sorry
end

end problem_statement_l594_594331


namespace students_answered_both_correctly_l594_594453

theorem students_answered_both_correctly
    (total_students : ℕ)
    (did_not_take_test : ℕ)
    (answered_q1_correctly : ℕ)
    (answered_q2_correctly : ℕ)
    (took_test := total_students - did_not_take_test)
    (correct_both_questions : ℕ := answered_q1_correctly + answered_q2_correctly - took_test) :
  total_students = 30 →
  did_not_take_test = 5 →
  answered_q1_correctly = 25 →
  answered_q2_correctly = 22 →
  correct_both_questions = 22 :=
by
  intros h_total h_did_not_take h_answered_q1 h_answered_q2
  have ht : took_test = total_students - did_not_take_test := by rw h_total; rw h_did_not_take; rfl
  rw ht
  sorry

end students_answered_both_correctly_l594_594453


namespace total_stairs_climbed_l594_594325

theorem total_stairs_climbed (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 1269 → 
  julia_stairs = 1269 / 3 - 7 → 
  jonny_stairs + julia_stairs = 1685 :=
begin
  intros hjonny hjulia,
  rw hjonny at *,
  rw hjulia,
  norm_num,
end

end total_stairs_climbed_l594_594325


namespace circle_intersecting_segments_l594_594787

theorem circle_intersecting_segments (A B C D P : Type)
  [point_circle : ∀ (X : Type), X ∈ {A, B, C, D}]
  (h_int : ∀ (X Y : Type), (X, Y) ∈ {(A, C), (B, D)} → P ∈ X ∩ Y)
  (h_AP : (dist A P = 5)) 
  (h_PC : (dist P C = 3)) 
  (h_BD : (dist B D = 10)) 
  (h_BP_less_DP : (∀ (BP DP : ℝ), BP < DP)) :
  (dist P D = 5 + Real.sqrt 10) :=
by
  sorry

end circle_intersecting_segments_l594_594787


namespace find_m_if_lines_perpendicular_l594_594560

-- Definitions for the slopes and perpendicular condition
def slope_l1 (m : ℝ) : ℝ := -(3 + m) / 4
def slope_l2 (m : ℝ) : ℝ := -(2 / (5 + m))

def perpendicular_slopes (k1 k2 : ℝ) : Prop := k1 * k2 = -1

theorem find_m_if_lines_perpendicular (m : ℝ) :
  perpendicular_slopes (slope_l1 m) (slope_l2 m) → m = -13 / 3 :=
by
  sorry

end find_m_if_lines_perpendicular_l594_594560


namespace mode_of_scores_l594_594736

def stem_leaf_plot : List (ℕ × List ℕ) :=
  [(9, [5, 5, 6]),
   (10, [4, 8]),
   (11, [2, 2, 2, 6, 6, 7]),
   (12, [0, 0, 3, 7, 7, 7]),
   (13, [1, 1, 1, 1]),
   (14, [5, 9])]

theorem mode_of_scores : ∃! n, n = 131 ∧ ∀ (s : ℕ × List ℕ) ∈ stem_leaf_plot, 
  let frequencies := s.2.foldr (fun x counts => counts.cons (s.2.count x)) []
  in max frequencies = frequencies.head! :=
sorry

end mode_of_scores_l594_594736


namespace area_larger_sphere_l594_594825

variables {r1 r2 r : ℝ}
variables {A1 A2 : ℝ}

-- Declare constants for the problem
def radius_smaller_sphere : ℝ := 4 -- r1
def radius_larger_sphere : ℝ := 6  -- r2
def radius_ball : ℝ := 1           -- r
def area_smaller_sphere : ℝ := 27  -- A1

-- Given conditions
axiom radius_smaller_sphere_condition : r1 = radius_smaller_sphere
axiom radius_larger_sphere_condition : r2 = radius_larger_sphere
axiom radius_ball_condition : r = radius_ball
axiom area_smaller_sphere_condition : A1 = area_smaller_sphere

-- Statement to be proved
theorem area_larger_sphere :
  r1 = radius_smaller_sphere → r2 = radius_larger_sphere → r = radius_ball → A1 = area_smaller_sphere → A2 = 60.75 :=
by
  intros
  sorry

end area_larger_sphere_l594_594825


namespace probability_point_outside_circle_l594_594815

open Classical

noncomputable def prob_point_outside_circle : ℚ :=
  let outcomes := (Σ m : fin 6, fin 6)
  let P := {p : outcomes × outcomes | p.1.1.val + 1 = p.1.2.val + 1 ∧ 
                          p.2.1.val + 1 = p.2.2.val + 1 ∧
                          (p.1.1.val + p.2.1.val + 2)^2 + (p.1.2.val + p.2.2.val + 2)^2 > 17}
  (P.to_finset.card : ℚ) / (6 * 6 * 6 * 6)

theorem probability_point_outside_circle :
  prob_point_outside_circle = 13 / 18 := by
  sorry

end probability_point_outside_circle_l594_594815


namespace Erica_net_income_l594_594873

theorem Erica_net_income:
  let price_per_kg := 20 in
  let fish_past_four_months := 80 in
  let fish_today := 2 * fish_past_four_months in
  let total_fish := fish_past_four_months + fish_today in
  let total_income := total_fish * price_per_kg in
  let month_maintenance_cost := 50 in
  let fuel_cost_per_kg := 2 in
  let total_maintenance_cost := 5 * month_maintenance_cost in
  let total_fuel_cost := fuel_cost_per_kg * total_fish in
  let total_cost := total_maintenance_cost + total_fuel_cost in
  let net_income := total_income - total_cost in
  net_income = 4070 :=
by
  sorry

end Erica_net_income_l594_594873


namespace magnitude_a_minus_2b_l594_594957

noncomputable def magnitude_of_vector_difference : ℝ :=
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)

theorem magnitude_a_minus_2b :
  let a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
  let b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))
  Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2) = Real.sqrt 3 :=
by
  sorry

end magnitude_a_minus_2b_l594_594957


namespace rotation_matrix_120_degrees_l594_594534

-- Define the angle in radians
def angle : ℝ := 2 * Real.pi / 3

-- Define the expected rotation matrix
def expected_rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -1 / 2, - (Real.sqrt 3) / 2 ],
    ![  (Real.sqrt 3) / 2, -1 / 2 ]]

-- Define the rotation matrix function for an angle theta
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

-- The theorem stating that the rotation matrix for 120 degrees (2π/3 radians) 
-- is the expected rotation matrix
theorem rotation_matrix_120_degrees :
  rotation_matrix angle = expected_rotation_matrix := 
sorry

end rotation_matrix_120_degrees_l594_594534


namespace height_of_triangular_pyramid_perpendicular_l594_594532

theorem height_of_triangular_pyramid_perpendicular (DA DB DC : ℝ) (DK : ℝ) 
  (hDA : DA = 2) (hDB : DB = 3) (hDC : DC = 4) 
  (h1 : DA ⊥ DB) (h2 : DB ⊥ DC) (h3 : DC ⊥ DA) : 
  DK = 12 / Real.sqrt 13 := 
sorry

end height_of_triangular_pyramid_perpendicular_l594_594532


namespace value_to_subtract_l594_594980

theorem value_to_subtract (N x : ℕ) (h1 : (N - x) / 7 = 7) (h2 : (N - 34) / 10 = 2) : x = 5 :=
by 
  sorry

end value_to_subtract_l594_594980


namespace vertical_shirts_count_l594_594790

-- Definitions from conditions
def total_people : ℕ := 40
def checkered_shirts : ℕ := 7
def horizontal_shirts := 4 * checkered_shirts

-- Proof goal
theorem vertical_shirts_count :
  ∃ vertical_shirts : ℕ, vertical_shirts = total_people - (checkered_shirts + horizontal_shirts) ∧ vertical_shirts = 5 :=
sorry

end vertical_shirts_count_l594_594790


namespace solution_set_of_inequality_l594_594066

theorem solution_set_of_inequality :
  {x : ℝ | x^2 - 5 * x - 14 < 0} = {x : ℝ | -2 < x ∧ x < 7} :=
by
  sorry

end solution_set_of_inequality_l594_594066


namespace BD_length_l594_594289

theorem BD_length (A B C D : Point)
    (h_triangle : Triangle A B C)
    (h_AC : dist A C = 10)
    (h_BC : dist B C = 10)
    (h_AB : dist A B = 3)
    (h_D_on_AB : lies_on D (line_through A B))
    (h_B_between_A_D : B between A and D)
    (h_CD : dist C D = 11) :
    dist B D ≈ 3.32 :=
sorry

end BD_length_l594_594289


namespace scheduling_exists_l594_594009

theorem scheduling_exists
  (n : ℕ) (n_pos : 0 < n) :
  ∃ (schedule : Finset (Σ i j, ℕ)), 
  (∀ (i j : ℕ) (h₁ : 1 ≤ i ∧ i ≤ 2 * n) (h₂ : 1 ≤ j ∧ j ≤ 2 * n), i ≠ j → 
    ∃! d, 1 ≤ d ∧ d ≤ 2 * n - 1 ∧ (i, j, d) ∈ schedule ∨ (j, i, d) ∈ schedule) ∧ 
  (∀ (k : ℕ) (1 ≤ k ∧ k ≤ 2 * n),
    Finset.filter (λ triple, triple.2.2 = k) schedule ∈ Finset.range 1
  ) :=
begin
  sorry
end

end scheduling_exists_l594_594009


namespace smallest_angle_in_convex_polygon_l594_594717

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
a + n * d

def sum_of_interior_angles (n : ℕ) : ℤ :=
(n - 2) * 180

def average_angle (n : ℕ) : ℤ :=
(sum_of_interior_angles n) / n

theorem smallest_angle_in_convex_polygon :
  ∀ (n : ℕ) (angles : Finset ℤ) (d : ℤ),
    n = 18 →
    (angles.card = n) →
    (∀ i : ℕ, i ∈ Finset.range n →
      ∃ a, angles = Finset.image (arithmetic_sequence a d) (Finset.range n)) →
    (∀ θ ∈ angles, θ < 180) →
    ∃ a : ℤ, Finset.min' angles (by
        { rw Finset.nonempty_iff_ne_empty, intro h,
          rw h at h_1, contradiction }) = 143 :=
by {
  sorry
}

end smallest_angle_in_convex_polygon_l594_594717


namespace cone_volume_l594_594543

theorem cone_volume
  (a α β : ℝ)
  (hα : 0 < α ∧ α < 180)
  (hβ : 0 < β ∧ β < π / 2) :
  let R := a / (2 * Real.sin (α / 2))
      h := (a / (2 * Real.sin (α / 2))) / Real.tan β in
  (1 / 3) * Real.pi * R^2 * h = 
  Real.pi * a^3 * Real.cot β / (24 * (Real.sin (α / 2))^3) :=
by
  sorry

end cone_volume_l594_594543


namespace expression_value_l594_594607

-- Define the given condition as an assumption
variable (x : ℝ)
variable (h : 2 * x^2 + 3 * x - 1 = 7)

-- Define the target expression and the required result
theorem expression_value :
  4 * x^2 + 6 * x + 9 = 25 :=
by
  sorry

end expression_value_l594_594607


namespace cone_slant_height_l594_594285

theorem cone_slant_height (r : ℝ) (θ : ℝ) (x : ℝ) (h1 : r = 2) (h2 : θ = 120) :
  x = 6 :=
by
  have hc : 2 * Real.pi * r = 4 * Real.pi,
  { rw [h1, two_mul, show 2 * Real.pi = Real.pi * 2, by ring] }
  have ha : (θ / 360) * 2 * Real.pi * x = (1 / 3) * 2 * Real.pi * x,
  { rw [h2, div_eq_mul_inv, show 120 / 360 = 1 / 3, by norm_num] }
  have h_eq : (1 / 3) * 2 * Real.pi * x = 4 * Real.pi,
  { rw [mul_assoc, ←ha, hc] }
  have h_simpl : 2 * Real.pi * x = 12 * Real.pi,
  { field_simp [h_eq] }
  have h_final : x = 6,
  { apply mul_right_cancel₀ (ne_of_gt Real.pi_pos),
    field_simp [h_simpl] }
  exact h_final

end cone_slant_height_l594_594285


namespace cookie_milk_ratio_l594_594367

def price_milk : ℝ := 3 
def price_cereal : ℝ := 3.5
def price_banana : ℝ := 0.25
def price_apple : ℝ := 0.5
def num_cereal : ℕ := 2 
def num_banana : ℕ := 4
def num_apple : ℕ := 4 
def num_cookies : ℕ := 2 
def total_cost : ℝ := 25

theorem cookie_milk_ratio :
  let total_non_cookie_cost := price_milk + num_cereal * price_cereal + num_banana * price_banana + num_apple * price_apple in
  let cookie_cost := total_cost - total_non_cookie_cost in
  let price_cookie := cookie_cost / num_cookies in
  let ratio := price_cookie / price_milk in
  ratio = 2 :=
by
  sorry

end cookie_milk_ratio_l594_594367


namespace intersection_P_Q_l594_594258

open Set

noncomputable def P : Set ℝ := { x : ℝ | 1 < 2^x ∧ 2^x < 2}
noncomputable def Q : Set ℝ := { x : ℝ | log (1/2) x > 1 }

theorem intersection_P_Q : P ∩ Q = { x : ℝ | 0 < x ∧ x < 1/2 } := by
  sorry

end intersection_P_Q_l594_594258


namespace regular_tetrahedron_net_is_triangle_l594_594354

-- Define a tetrahedron structure
structure Tetrahedron :=
  (vertices : Fin 4 → ℝ × ℝ × ℝ)
  (congruent_faces : ∀ (i j : Fin 4), congruent (face i) (face j))

-- Define congruence for faces as a proof
def congruent (face1 face2 : Fin 3 → ℝ × ℝ) : Prop := sorry

-- Extract a face function
def Tetrahedron.face {T : Tetrahedron} (i : Fin 4) : Fin 3 → ℝ × ℝ :=
  sorry

-- Define the net of a tetrahedron unfolded onto a plane
def net (T : Tetrahedron) : Fin 3 → ℝ × ℝ :=
  sorry

theorem regular_tetrahedron_net_is_triangle (T : Tetrahedron) (h1 : T.congruent_faces) :
  ∃ (triangle_vertices : Fin 3 → Fin 3 → ℝ × ℝ), 
    (net T) = (λ i, triangle_vertices i) :=
sorry

end regular_tetrahedron_net_is_triangle_l594_594354


namespace count_integers_with_digit_sum_9_l594_594897

theorem count_integers_with_digit_sum_9 :
  {n : ℕ | 1 ≤ n ∧ n ≤ 2013 ∧ (nat.digits 10 n).sum = 9}.card = 101 := by
  sorry

end count_integers_with_digit_sum_9_l594_594897


namespace tire_radius_increase_l594_594004

noncomputable def radius_increase (initial_radius : ℝ) (odometer_initial : ℝ) (odometer_winter : ℝ) : ℝ :=
  let rotations := odometer_initial / ((2 * Real.pi * initial_radius) / 63360)
  let winter_circumference := (odometer_winter / rotations) * 63360
  let new_radius := winter_circumference / (2 * Real.pi)
  new_radius - initial_radius

theorem tire_radius_increase : radius_increase 16 520 505 = 0.32 := by
  sorry

end tire_radius_increase_l594_594004


namespace kant_correct_time_l594_594417

-- Definitions of the variables involved
variables (T_1 T_F T_S T_2 : ℕ)

-- Defining what needs to be proven
theorem kant_correct_time : 
  let T_total := T_2 in
  let T_T := (T_total - T_S) / 2 in
  T = T_F + T_T :=
by
  sorry

end kant_correct_time_l594_594417


namespace work_completion_l594_594449

-- Given conditions
def workRateA (days : ℕ) : ℝ := 1 / (days : ℝ)  -- A's work rate
def workRateTogether (days : ℕ) : ℝ := 1 / (days : ℝ)  -- A and B's combined work rate

-- Proof problem
theorem work_completion (A_days B_days together_days : ℕ) 
  (h1 : together_days = 8) 
  (h2 : A_days = 12) 
  (h3 : workRateTogether together_days = workRateA A_days + workRateA B_days) :
  together_days = 8 := by
  -- the actual proof would go here
  sorry

end work_completion_l594_594449


namespace total_travel_distance_correct_l594_594006

-- Definitions for the conditions
def XZ := 5000 -- km
def XY := 5200 -- km

-- Using Pythagorean theorem to define YZ
def YZ : ℝ := Real.sqrt (XY^2 - XZ^2)

-- Travel distances
def total_distance := XZ + XY + YZ

-- The proof problem statement
theorem total_travel_distance_correct : total_distance = 11628 := by
  sorry

end total_travel_distance_correct_l594_594006


namespace factorize_expression_l594_594890

theorem factorize_expression (x y : ℝ) : x^3 * y - 4 * x * y = x * y * (x - 2) * (x + 2) :=
sorry

end factorize_expression_l594_594890


namespace linear_search_average_comparisons_l594_594234

noncomputable def average_comparisons_linear_search (n : ℕ) : ℝ :=
  (1 + n) / 2

-- Let n be 10,000
def n : ℕ := 10000

theorem linear_search_average_comparisons :
  average_comparisons_linear_search n = 5000.5 := 
by
  rw [average_comparisons_linear_search, n]
  norm_num
  sorry  -- Skipping the detailed proof

end linear_search_average_comparisons_l594_594234


namespace exists_fixed_point_in_interval_l594_594253

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 + (x / 2) + 1 / 4

theorem exists_fixed_point_in_interval : 
  ∃ x₀ ∈ set.Ioo (0 : ℝ) (1 / 2), f x₀ = x₀ :=
sorry

end exists_fixed_point_in_interval_l594_594253


namespace solution_set_for_inequality_l594_594569

-- Definitions from conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def in_domain (x : ℝ) : Prop := x ∈ Ioo (-π / 2) (π / 2)

-- Proving the main statement
theorem solution_set_for_inequality (f : ℝ → ℝ) (x : ℝ) 
  (h_domain : in_domain x) 
  (h_odd : is_odd_function (λ x, f x - 1)) 
  (h_inequality : ∀ x, f''' x + f x * tan x > 0) :
  (f x > cos x) ↔ (0 < x ∧ x < (π / 2)) :=
sorry

end solution_set_for_inequality_l594_594569


namespace area_square_inscribed_in_ellipse_l594_594492

-- Define the ellipse equation as a function
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 5) + (y^2 / 10) = 1

-- Define the condition that the vertices of the inscribed square are (±t, ±t)
def vertices_inscribed_square (t : ℝ) : Prop :=
  ellipse t t

-- Define the area of the inscribed square
def area_inscribed_square (t : ℝ) : ℝ :=
  let side_length := 2 * t
  in side_length^2

-- State the theorem with the given conditions and expected result
theorem area_square_inscribed_in_ellipse :
  ∃ (t : ℝ), area_inscribed_square t = 40 / 3 ∧ vertices_inscribed_square t := 
sorry

end area_square_inscribed_in_ellipse_l594_594492


namespace find_angle_C_find_range_a_b_l594_594288

-- Problem statement definition
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Sides

-- Given conditions
axiom condition1 : in_triangle A B C a b c
axiom condition2 : 2 * c * sin C = (2 * b + a) * sin B + (2 * a - 3 * b) * sin A

-- Needed to prove
theorem find_angle_C : C = π / 3 :=
by
  -- We would prove here normally, but the proof is intentionally omitted.
  sorry

theorem find_range_a_b (h : c = 4) : 4 < a + b ∧ a + b ≤ 8 :=
by
  -- We would prove here normally, but the proof is intentionally omitted.
  sorry

end find_angle_C_find_range_a_b_l594_594288


namespace main_l594_594332

open EuclideanGeometry

noncomputable def XY_squared_proof (A B C T X Y : Point) (ω : Circle) : ℝ :=
  let BT := 20
  let CT := 20
  let BC := 26
  let equation := TX^2 + TY^2 + 2 * XY^2 = 1609
  
  have h1 : 0
  {
    -- Proof steps would go here.
    -- However, for demonstration purposes and according to the instruction, we only state the problem.
  }
  
  -- This is the theorem we need to prove
  (XY^2 = 108)

-- Apply the hypothesis conditions and goals
theorem main : ∀ {A B C T X Y : Point} {ω : Circle}
  (h1 : acuteTriangle A B C)
  (h2 : tangentTo ω B ∧ tangentTo ω C)
  (h3 : intersectionAt B C T)
  (h4 : projection T A B X)
  (h5 : projection T A C Y)
  (h6 : BT = 20)
  (h7 : CT = 20)
  (h8 : BC = 26)
  (h9 : (TX ^ 2 + TY ^ 2 + 2 * XY ^ 2 = 1609))
  : XY_squared_proof A B C T X Y ω = 108 := sorry

end main_l594_594332


namespace cloth_meters_sold_l594_594831

-- Conditions as definitions
def total_selling_price : ℝ := 4500
def profit_per_meter : ℝ := 14
def cost_price_per_meter : ℝ := 86

-- The statement of the problem
theorem cloth_meters_sold (SP : ℝ := cost_price_per_meter + profit_per_meter) :
  total_selling_price / SP = 45 := by
  sorry

end cloth_meters_sold_l594_594831


namespace trigonometric_expression_evaluation_l594_594523

theorem trigonometric_expression_evaluation:
  (\frac{\sin (20 * Real.pi / 180) * \cos (15 * Real.pi / 180) + 
          \cos (160 * Real.pi / 180) * \cos (105 * Real.pi / 180)}
         { \sin (25 * Real.pi / 180) * \cos (10 * Real.pi / 180) + 
           \cos (155 * Real.pi / 180) * \cos (95 * Real.pi / 180)}
  ) = \frac{1}{3} :=
by
  sorry

end trigonometric_expression_evaluation_l594_594523


namespace solve_geometric_lines_l594_594145

noncomputable def geometric_solution_lines {O P : Point} (r1 r2 : ℝ) (h_r1_r2 : r1 > r2) (angle : ℝ) (h_angle : angle = 30) : ℕ :=
  let solutions := {g : Line | 
                      g.passes_through P ∧
                      chord_ratio g O r1 r2 2 ∧
                      angle_with_horizontal g angle
                   } 
  in solutions.card

theorem solve_geometric_lines (O P : Point) (r1 r2 : ℝ) (h_r1_r2 : r1 > r2) (angle : ℝ) (h_angle : angle = 30) :
  geometric_solution_lines O P r1 r2 h_r1_r2 angle h_angle = 4 :=
sorry

end solve_geometric_lines_l594_594145


namespace evaluate_g_at_xplus3_l594_594276

def g (x : ℝ) : ℝ := (x^2 + 3 * x) / 2

theorem evaluate_g_at_xplus3 (x : ℝ) : g (x + 3) = (x^2 + 9 * x + 18) / 2 :=
by
  sorry

end evaluate_g_at_xplus3_l594_594276


namespace apex_angle_of_third_cone_l594_594079

def vertex : Type := ℝ

-- Definition for the apex angles of the first two cones
def angle1 : ℝ := π / 6
def angle2 : ℝ := π / 6

-- Definition for the apex angle of the fourth cone
def angle4 : ℝ := π / 3

-- Definition for the angles in terms of cotangent
def beta := Real.arccot (sqrt 3 + 4)

-- Theorem stating that the apex angle of the third cone equals the derived value
theorem apex_angle_of_third_cone (A : vertex) 
  (angle1_eq : angle1 = π / 6)
  (angle2_eq : angle2 = π / 6)
  (angle4_eq : angle4 = π / 3) :
  2 * beta = 2 * Real.arccot (sqrt 3 + 4) := by
  sorry

end apex_angle_of_third_cone_l594_594079


namespace sum_of_triangle_areas_l594_594312

theorem sum_of_triangle_areas (r : ℝ) (n : ℕ) (s : ℝ) (area_of_eq_triangle : ℝ) :
  r = 1 →
  n = 51 →
  s = real.sqrt 3 →
  area_of_eq_triangle = (3.sqrt) / 4 →
  let polygon := closed_polygon r n s in
  let triangles_total_area := sum_of_triangle_areas polygon in
  triangles_total_area ≥ 3 * area_of_eq_triangle :=
by
  sorry

end sum_of_triangle_areas_l594_594312


namespace annual_increase_in_living_space_l594_594153

-- Definitions based on conditions
def population_2000 : ℕ := 200000
def living_space_2000_per_person : ℝ := 8
def target_living_space_2004_per_person : ℝ := 10
def annual_growth_rate : ℝ := 0.01
def years : ℕ := 4

-- Goal stated as a theorem
theorem annual_increase_in_living_space :
  let final_population := population_2000 * (1 + annual_growth_rate)^years
  let total_living_space_2004 := target_living_space_2004_per_person * final_population
  let initial_living_space := living_space_2000_per_person * population_2000
  let total_additional_space := total_living_space_2004 - initial_living_space
  let average_annual_increase := total_additional_space / years
  average_annual_increase = 120500.0 :=
sorry

end annual_increase_in_living_space_l594_594153


namespace percentage_increase_l594_594369

theorem percentage_increase (A B : ℝ) (y : ℝ) (h : A > B) (h1 : B > 0) (h2 : C = A + B) (h3 : C = (1 + y / 100) * B) : y = 100 * (A / B) := 
sorry

end percentage_increase_l594_594369


namespace find_a11_l594_594926

-- Conditions
variables {a : ℕ → ℚ}
def a_3_condition : Prop := a 3 = 2
def a_7_condition : Prop := a 7 = 1
def arithmetic_progression_condition : Prop :=
  ∀ m n : ℕ, ∃ d : ℚ, (1 / (a m + 1) - 1 / (a n + 1)) = d * (m - n)

-- Target
theorem find_a11 (h1 : a_3_condition) (h2 : a_7_condition) (h3 : arithmetic_progression_condition) :
  a 11 = -2 / 5 := sorry

end find_a11_l594_594926


namespace train_passes_man_in_12_seconds_l594_594775

noncomputable def time_to_pass_man (train_length: ℝ) (train_speed_kmph: ℝ) (man_speed_kmph: ℝ) : ℝ :=
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18)
  train_length / relative_speed_mps

theorem train_passes_man_in_12_seconds :
  time_to_pass_man 220 60 6 = 12 := by
 sorry

end train_passes_man_in_12_seconds_l594_594775


namespace cos_shifted_overlaps_sine_l594_594384

theorem cos_shifted_overlaps_sine
  (φ : ℝ)
  (hφ1 : -π ≤ φ)
  (hφ2 : φ < π)
  (h_overlap : ∀ x : ℝ, cos (2 * (x - π / 2) + φ) = sin (2 * x + π / 3)) :
  |φ| = 5 * π / 6 :=
by
  sorry

end cos_shifted_overlaps_sine_l594_594384


namespace minimum_distance_AO_l594_594389

variable (h s : ℝ)

theorem minimum_distance_AO (h s : ℝ) (h_pos : h > 0) (s_pos : s > 0) (h_gt_s : h > s) :
    ∃ AO : ℝ, AO = Real.sqrt (h^2 - s^2) :=
by
  use Real.sqrt (h^2 - s^2)
  sorry

end minimum_distance_AO_l594_594389


namespace sufficient_condition_sufficient_but_not_necessary_condition_l594_594955

def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem sufficient_condition (m : ℝ) : collinear (1, m - 1) (m, 2) ↔ m = 2 ∨ m = -1 :=
by {
  dsimp [collinear],
  calc  1 * 2 = (m - 1) * m ↔ 2 = m^2 - m
    : by linarith, -- perform the algebra
  { exact mul_eq_zero.mp (eq_zero_of_sub_eq_zero (by linarith)) }
}

theorem sufficient_but_not_necessary_condition :
  (m = 2 → collinear (1, 1) (m, 2)) ∧ ¬ (m = 2 → ∀ m, collinear (1, m-1) (m, 2) → m = 2) :=
by { sorry } -- Skipping the proof as per the instructions.

end sufficient_condition_sufficient_but_not_necessary_condition_l594_594955


namespace floor_e_eq_two_l594_594887

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end floor_e_eq_two_l594_594887


namespace maximize_product_numbers_l594_594521

theorem maximize_product_numbers (a b : ℕ) (ha : a = 96420) (hb : b = 87531) (cond: a * b = 96420 * 87531):
  b = 87531 := 
by sorry

end maximize_product_numbers_l594_594521


namespace isosceles_trapezoid_right_triangle_l594_594700
noncomputable theory
open_locale classical

theorem isosceles_trapezoid_right_triangle
  (a c b e : ℝ)
  (h_diag : e^2 = b^2 + a * c) :
  ∃ (right_triangle : Type), (right_triangle = {diagonal := e, leg := b, geomean := real.sqrt (a * c)}) :=
by
  have h1 : e^2 = b^2 + a * c := h_diag,
  sorry,

end isosceles_trapezoid_right_triangle_l594_594700


namespace quadrilateral_tangent_circle_condition_l594_594475

theorem quadrilateral_tangent_circle_condition {A B C D G H E F J K : Point} : 
    circle_tangent_to_sides A B B C G H → 
    circle_intersects_diagonal A C E F → 
    (∃ J K, other_circle_passing_through E F_and_tangent_to_extensions_of_sides D A D C J K) ↔
    (segment_length A B + segment_length A D = segment_length B C + segment_length C D) :=
sorry

end quadrilateral_tangent_circle_condition_l594_594475


namespace integer_pairs_satisfying_equation_and_nonnegative_product_l594_594202

theorem integer_pairs_satisfying_equation_and_nonnegative_product :
  ∃ (pairs : List (ℤ × ℤ)), 
    (∀ p ∈ pairs, p.1 * p.2 ≥ 0 ∧ p.1^3 + p.2^3 + 99 * p.1 * p.2 = 33^3) ∧ 
    pairs.length = 35 :=
by sorry

end integer_pairs_satisfying_equation_and_nonnegative_product_l594_594202


namespace part1_part2_l594_594580

def f (x : ℝ) : ℝ := abs (x + 2) - 2 * abs (x - 1)

theorem part1 : { x : ℝ | f x ≥ -2 } = { x : ℝ | -2/3 ≤ x ∧ x ≤ 6 } :=
by
  sorry

theorem part2 (a : ℝ) :
  (∀ x ≥ a, f x ≤ x - a) ↔ a ≤ -2 ∨ a ≥ 4 :=
by
  sorry

end part1_part2_l594_594580


namespace sum_of_roots_l594_594591

theorem sum_of_roots : ∀ x : ℝ, ((x + 3) * (x - 4) = 20) → (∃ a b : ℝ, a = x + 3 ∧ b = x - 4 ∧ a * b = 20 ∧ (∑ x in {a, b}, x) = 1) :=
begin
  intros x h,
  sorry
end

end sum_of_roots_l594_594591


namespace bulb_works_longer_than_4000_hours_l594_594527

noncomputable def P_X := 0.5
noncomputable def P_Y := 0.3
noncomputable def P_Z := 0.2

noncomputable def P_4000_given_X := 0.59
noncomputable def P_4000_given_Y := 0.65
noncomputable def P_4000_given_Z := 0.70

noncomputable def P_4000 := 
  P_X * P_4000_given_X + P_Y * P_4000_given_Y + P_Z * P_4000_given_Z

theorem bulb_works_longer_than_4000_hours : P_4000 = 0.63 :=
by
  sorry

end bulb_works_longer_than_4000_hours_l594_594527


namespace simplify_expression_l594_594017

theorem simplify_expression :
  ( ( (sqrt 2 + 1) ^ (1 - sqrt 3) ) / ( (sqrt 2 - 1) ^ (1 + sqrt 3) ) ) = 3 + 2 * sqrt 2 := 
by
  sorry

end simplify_expression_l594_594017


namespace cone_volume_proof_l594_594206

noncomputable def slant_height := 21
noncomputable def horizontal_semi_axis := 10
noncomputable def vertical_semi_axis := 12
noncomputable def equivalent_radius :=
  Real.sqrt (horizontal_semi_axis * vertical_semi_axis)
noncomputable def cone_height :=
  Real.sqrt (slant_height ^ 2 - equivalent_radius ^ 2)

noncomputable def cone_volume :=
  (1 / 3) * Real.pi * horizontal_semi_axis * vertical_semi_axis * cone_height

theorem cone_volume_proof :
  cone_volume = 2250.24 * Real.pi := sorry

end cone_volume_proof_l594_594206


namespace agent_commission_l594_594842

-- Define the conditions
def total_sales : ℝ := 600
def commission_rate : ℝ := 2.5 / 100

-- Define the target commission
def expected_commission : ℝ := 15

-- The theorem to be proven
theorem agent_commission : total_sales * commission_rate = expected_commission := by
  sorry

end agent_commission_l594_594842


namespace emily_distance_l594_594522

noncomputable def distance_from_start (start final: ℝ × ℝ): ℝ :=
  real.sqrt ((final.1 - start.1) ^ 2 + (final.2 - start.2) ^ 2)

theorem emily_distance :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (7, 0)
  let C : ℝ × ℝ := (7 + 5 * real.cos (real.pi / 4), 5 * real.sin (real.pi / 4))
  distance_from_start A C = real.sqrt 193 / 2 :=
by
  sorry

end emily_distance_l594_594522


namespace real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l594_594906

variables {m : ℝ}

-- (1) For z to be a real number
theorem real_number_condition : (m^2 - 3 * m = 0) ↔ (m = 0 ∨ m = 3) :=
by sorry

-- (2) For z to be an imaginary number
theorem imaginary_number_condition : (m^2 - 3 * m ≠ 0) ↔ (m ≠ 0 ∧ m ≠ 3) :=
by sorry

-- (3) For z to be a purely imaginary number
theorem pure_imaginary_number_condition : (m^2 - 5 * m + 6 = 0 ∧ m^2 - 3 * m ≠ 0) ↔ (m = 2) :=
by sorry

end real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l594_594906


namespace garden_perimeter_l594_594781

theorem garden_perimeter (width_garden length_playground width_playground : ℕ) 
  (h1 : width_garden = 12) 
  (h2 : length_playground = 16) 
  (h3 : width_playground = 12) 
  (area_playground : ℕ)
  (h4 : area_playground = length_playground * width_playground) 
  (area_garden : ℕ) 
  (h5 : area_garden = area_playground) 
  (length_garden : ℕ) 
  (h6 : area_garden = length_garden * width_garden) :
  2 * length_garden + 2 * width_garden = 56 := by
  sorry

end garden_perimeter_l594_594781


namespace pyramid_lateral_edge_ratio_l594_594821

variable (h x : ℝ)

-- We state the conditions as hypotheses
axiom pyramid_intersected_by_plane_parallel_to_base (h : ℝ) (S S' : ℝ) :
  S' = S / 2 → (S' / S = (x / h) ^ 2) → (x = h / Real.sqrt 2)

-- The theorem we need to prove
theorem pyramid_lateral_edge_ratio (h x : ℝ) (S S' : ℝ)
  (cond1 : S' = S / 2)
  (cond2 : S' / S = (x / h) ^ 2) :
  x / h = 1 / Real.sqrt 2 :=
by
  -- skip the proof
  sorry

end pyramid_lateral_edge_ratio_l594_594821


namespace tan_of_trig_eq_l594_594913

theorem tan_of_trig_eq (x : Real) (h : (1 - Real.cos x + Real.sin x) / (1 + Real.cos x + Real.sin x) = -2) : Real.tan x = 4 / 3 :=
by sorry

end tan_of_trig_eq_l594_594913


namespace complement_union_eq_l594_594688

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3}
def N : Set ℕ := {3, 5}

theorem complement_union_eq :
  (U \ (M ∪ N)) = {2, 4} := by
  sorry

end complement_union_eq_l594_594688


namespace sum_of_valid_k_values_l594_594380

theorem sum_of_valid_k_values : 
  let k_values := {k : ℕ | ∃ (α β : ℤ), α * β = 15 ∧ α + β = k} in
  k_values.sum = 48 :=
by
  sorry

end sum_of_valid_k_values_l594_594380


namespace complex_product_polar_form_l594_594169

def cis (theta : ℝ) : ℂ := complex.of_real (cos theta) + complex.I * (complex.of_real (sin theta))

theorem complex_product_polar_form :
  let z1 := 4 * cis (real.pi / 6)
  let z2 := -3 * cis (real.pi / 4)
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 2*real.pi ∧ (z1 * z2 = r * cis θ ∧ r = 12 ∧ θ = 17 * real.pi / 12) :=
by
  sorry

end complex_product_polar_form_l594_594169


namespace decagon_exterior_angle_sum_l594_594407

-- Define what it means to be a regular decagon
def regular_decagon (P : Type) := ∃ (sides : P → Prop), (∀ x : P, sides x)

-- Define the sum of exterior angles
def sum_exterior_angles (P : Type) [regular_decagon P] : ℝ :=
  360

-- The theorem to be proved
theorem decagon_exterior_angle_sum (P : Type) [regular_decagon P] : sum_exterior_angles P = 360 := by
  sorry

end decagon_exterior_angle_sum_l594_594407


namespace village_population_rate_l594_594089

theorem village_population_rate
    (population_X : ℕ := 68000)
    (population_Y : ℕ := 42000)
    (increase_Y : ℕ := 800)
    (years : ℕ := 13) :
  ∃ R : ℕ, population_X - years * R = population_Y + years * increase_Y ∧ R = 1200 :=
by
  exists 1200
  sorry

end village_population_rate_l594_594089


namespace blue_snakes_can_multiply_l594_594753

-- Defining the properties and conditions
def snake := ℕ

def blue_snakes : set snake := {1, 2, 3, 4, 5}
def happy_snakes : set snake := {1, 2, 6, 7, 8, 9}
def can_add : snake → Prop := λ s, s ∈ happy_snakes
def can_subtract : snake → Prop := λ s, ¬(s ∈ blue_snakes)
def can_multiply : snake → Prop := λ s, s ∈ happy_snakes

axiom all_happy_can_add_and_multiply : ∀ s, s ∈ happy_snakes → can_add s ∧ can_multiply s
axiom none_blue_can_subtract : ∀ s, s ∈ blue_snakes → ¬ can_subtract s
axiom none_can_subtract_and_add : ∀ s, ¬ can_subtract s → ¬ can_add s
axiom at_least_one_happy_is_blue : ∃ s, s ∈ happy_snakes ∧ s ∈ blue_snakes

theorem blue_snakes_can_multiply : ∀ s, s ∈ blue_snakes → can_multiply s :=
by {
  sorry
}

end blue_snakes_can_multiply_l594_594753


namespace count_integers_with_seven_or_eight_as_digit_l594_594588

def uses_digit_seven_or_eight (n : ℕ) : Prop :=
  let digits := List.ofArray (n.digits 9)
  List.any digits (λ d, d = 7 ∨ d = 8)

theorem count_integers_with_seven_or_eight_as_digit :
  (Finset.range 729).filter uses_digit_seven_or_eight .card = 386 :=
by
  sorry

end count_integers_with_seven_or_eight_as_digit_l594_594588


namespace value_of_a_l594_594257

theorem value_of_a (a : ℝ) : 
  let M := {5, a^2 - 3 * a + 5}
  let N := {1, 3}
  M ∩ N ≠ ∅ → (a = 1 ∨ a = 2) :=
by
  sorry

end value_of_a_l594_594257


namespace find_vertex_P_l594_594638

noncomputable def midpoint (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
((a.1 + b.1) / 2, (a.2 + b.2) / 2, (a.3 + b.3) / 2)

theorem find_vertex_P
  (midQR : ℝ × ℝ × ℝ)
  (midPR : ℝ × ℝ × ℝ)
  (midPQ : ℝ × ℝ × ℝ)
  (hQR : midQR = (2, 3, -1))
  (hPR : midPR = (1, 2, -2))
  (hPQ : midPQ = (3, 1, 4))
  : ∃ (P : ℝ × ℝ × ℝ), P = (2, 3, -1) :=
sorry

end find_vertex_P_l594_594638


namespace largest_possible_factors_l594_594047

theorem largest_possible_factors :
  ∃ (m : ℕ) (q : Fin m → Polynomial ℝ),
    (x : ℝ) → x^10 - 1 = ∏ i, q i ∧ ∀ i,  degree (q i) > 0 ∧ m = 3 :=
by
  sorry

end largest_possible_factors_l594_594047


namespace reduction_for_same_profit_cannot_reach_460_profit_l594_594270

-- Defining the original conditions
noncomputable def cost_price_per_kg : ℝ := 20
noncomputable def original_selling_price_per_kg : ℝ := 40
noncomputable def daily_sales_volume : ℝ := 20

-- Reduction in selling price required for same profit
def reduction_to_same_profit (x : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - x
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * x
  new_profit_per_kg * new_sales_volume = (original_selling_price_per_kg - cost_price_per_kg) * daily_sales_volume

-- Check if it's impossible to reach a daily profit of 460 yuan
def reach_460_yuan_profit (y : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - y
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * y
  new_profit_per_kg * new_sales_volume = 460

theorem reduction_for_same_profit : reduction_to_same_profit 10 :=
by
  sorry

theorem cannot_reach_460_profit : ∀ y, ¬ reach_460_yuan_profit y :=
by
  sorry

end reduction_for_same_profit_cannot_reach_460_profit_l594_594270


namespace city_population_inference_l594_594411

-- Definitions based on conditions
variables {Population : Type} [has_add Population] [has_sub Population] [has_zero Population]
variables (PopA PopB PopC PopD : Population)

-- Given conditions
def condition1 : Prop := PopA + PopB = PopC + PopD + 5000
def condition2 : Prop := PopC = PopA - 5000

-- The theorem we need to prove
theorem city_population_inference (h1 : condition1) (h2 : condition2) : 
    PopA = PopC + 5000 ∨ PopD = PopC + 5000 :=
by
  sorry

end city_population_inference_l594_594411


namespace max_distance_from_circle_to_line_l594_594031

noncomputable def max_distance_point_to_line_circle : Real :=
let center := (1, 1)
let radius := Real.sqrt 2
let distance_from_center_to_line := (Real.abs (1 + 1 + 2)) / (Real.sqrt 2)
let max_distance := distance_from_center_to_line + radius
max_distance

theorem max_distance_from_circle_to_line : 
  max_distance_point_to_line_circle = 3 * Real.sqrt 2 := 
by 
  sorry

end max_distance_from_circle_to_line_l594_594031


namespace polar_equation_of_C_distance_AB_l594_594628

-- Definitions based on given conditions
def cartesian_eq_curve (a : ℝ) : ℝ × ℝ := (4 * Real.cos a + 2, 4 * Real.sin a)

def polar_eq_line (θ : ℝ) : Prop := θ = Real.pi / 6

-- Theorems to prove
theorem polar_equation_of_C :
  ∀ (ρ θ : ℝ),
  (∃ a : ℝ, (ρ * Real.cos θ, ρ * Real.sin θ) = (4 * Real.cos a + 2, 4 * Real.sin a)) →
  ρ^2 - 4 * ρ * Real.cos θ = 12 :=
sorry

theorem distance_AB :
  ∀ (ρ1 ρ2 : ℝ),
  polar_eq_line (Real.pi / 6) →
  (ρ1^2 - 4 * ρ1 * Real.cos (Real.pi / 6) = 12) →
  (ρ2^2 - 4 * ρ2 * Real.cos (Real.pi / 6) = 12) →
  |ρ1 - ρ2| = 2 * Real.sqrt 15 :=
sorry

end polar_equation_of_C_distance_AB_l594_594628


namespace triangle_geometry_property_proof_l594_594306

-- Definitions of geometry conditions
variables (A B C M : Type) [Euc_space : EuclideanSpace ℝ ℝ³]
(h_triangle : triangle A B C)
(h_M_midpoint : midpoint M B C)
(h_angle_B_45 : angle A B C = π / 4)
(h_AB_length : dist A B = 2 * √2)
(h_AM_length : dist A M = 2 * √2)

-- The proof goals
theorem triangle_geometry_property_proof :
  dist A C = 2 * √10 ∧ cos (angle M A C) = (2 * √5) / 5 :=
by sorry

end triangle_geometry_property_proof_l594_594306


namespace Amy_crumbs_l594_594109

variable (z : ℕ)

theorem Amy_crumbs (T C : ℕ) (h1 : T * C = z)
  (h2 : ∃ T_A : ℕ, T_A = 2 * T)
  (h3 : ∃ C_A : ℕ, C_A = (3 * C) / 2) :
  ∃ z_A : ℕ, z_A = 3 * z :=
by
  sorry

end Amy_crumbs_l594_594109


namespace income_comparison_l594_594497

variables (C A B D E : ℝ)

-- Define the conditions
def cond1 : Prop := A = 1.2 * C
def cond2 : Prop := B = 1.25 * A
def cond3 : Prop := D = 0.85 * B
def cond4 : Prop := E = 1.1 * C

-- Define average income of A, C, D, and E
def avg_income : ℝ := (A + C + D + E) / 4

-- Calculate the percentage difference
def pct_difference : ℝ := (B - avg_income) / avg_income * 100

-- Lean statement to prove the main question
theorem income_comparison :
  cond1 →
  cond2 →
  cond3 →
  cond4 →
  pct_difference = 31.15 :=
by {
  intros,
  sorry
}

end income_comparison_l594_594497


namespace area_triangle_LEF_l594_594631

noncomputable def circle_radius : ℝ := 5
noncomputable def chord_length : ℝ := 6
noncomputable def parallel_segments (L M : ℝ) : Prop := true
noncomputable def LN : ℝ := 15
noncomputable def collinear (L N P M : ℝ) : Prop := true

theorem area_triangle_LEF {L N P M : ℝ} (h1 : collinear L N P M) (h2 : parallel_segments L M) : 
  let R := (LN - circle_radius) + sqrt (circle_radius^2 - (chord_length/2)^2)
  ∆ := (1/2) * chord_length * R
  ∆ = 42 :=
sorry

end area_triangle_LEF_l594_594631


namespace directrix_of_parabola_l594_594024

noncomputable def parabola_directrix : Prop :=
  let a : ℝ := 1 in
  let k : ℝ := 0 in
  let p : ℝ := 1 / (4 * a) in
  ∀ y : ℝ, (y = -p) ↔ (y = -1 / 4)

theorem directrix_of_parabola :
  parabola_directrix :=
by
  sorry

end directrix_of_parabola_l594_594024


namespace count_blocks_differ_in_three_ways_l594_594478

def blocks : ℕ := 64

def material : Type := {x // x = "plastic" ∨ x = "wood"}
def size : Type := {x // x = "small" ∨ x = "large"}
def color : Type := {x // x = "blue" ∨ x = "green" ∨ x = "red" ∨ x = "yellow"}
def shape : Type := {x // x = "circle" ∨ x = "hexagon" ∨ x = "square" ∨ x = "triangle"}
def pattern : Type := {x // x = "striped" ∨ x = "dotted"}

def plastic_large_red_circle_striped : material × size × color × shape × pattern :=
 (⟨"plastic", by simp⟩, ⟨"large", by simp⟩, ⟨"red", by simp⟩, ⟨"circle", by simp⟩, ⟨"striped", by simp⟩)

def difference_ways (a b : material × size × color × shape × pattern) : ℕ :=
(match a.1, b.1 with
| ⟨x, _⟩, ⟨y, _⟩ => if x ≠ y then 1 else 0) +
(match a.2, b.2 with
| ⟨x, _⟩, ⟨y, _⟩ => if x ≠ y then 1 else 0) +
(match a.3, b.3 with
| ⟨x, _⟩, ⟨y, _⟩ => if x ≠ y then 1 else 0) +
(match a.4, b.4 with
| ⟨x, _⟩, ⟨y, _⟩ => if x ≠ y then 1 else 0) +
(match a.5, b.5 with
| ⟨x, _⟩, ⟨y, _⟩ => if x ≠ y then 1 else 0)

def count_exactly_diff_three_ways : ℕ := 
  (List.filter (λ b => difference_ways plastic_large_red_circle_striped b = 3)
  ((List.product
      (List.map subtype.val [⟨"plastic", by simp⟩, ⟨"wood", by simp⟩])
      (List.product
        (List.map subtype.val [⟨"small", by simp⟩, ⟨"large", by simp⟩])
        (List.product
          (List.map subtype.val [⟨"blue", by simp⟩, ⟨"green", by simp⟩, ⟨"red", by simp⟩, ⟨"yellow", by simp⟩])
          (List.product
            (List.map subtype.val [⟨"circle", by simp⟩, ⟨"hexagon", by simp⟩, ⟨"square", by simp⟩, ⟨"triangle", by simp⟩])
            (List.map subtype.val [⟨"striped", by simp⟩, ⟨"dotted", by simp⟩] )))))))
  .length

theorem count_blocks_differ_in_three_ways :
  count_exactly_diff_three_ways = 21 :=
sorry

end count_blocks_differ_in_three_ways_l594_594478


namespace proposition_1_proposition_2_proposition_3_correct_answer_l594_594900

variables {x1 y1 x2 y2 x y : ℝ}

def new_distance (A B: ℝ × ℝ) : ℝ :=
  |B.1 - A.1| + |B.2 - A.2|

theorem proposition_1 (A B C: ℝ × ℝ) (hA : A.1 = x1 ∧ A.2 = y1) (hB : B.1 = x2 ∧ B.2 = y2) (hC : (C.1 = x ∧ C.2 = y) ∧ (min x1 x2 ≤ x ∧ x ≤ max x1 x2) ∧ (min y1 y2 ≤ y ∧ y ≤ max y1 y2)) :
  new_distance A C + new_distance C B = new_distance A B :=
sorry

theorem proposition_2 (A B C: ℝ × ℝ) (hA : A.1 = x1 ∧ A.2 = y1) (hB : B.1 = x2 ∧ B.2 = y2) (hC : C.1 = x ∧ C.2 = y) (h_angle : ∠ABC = 90) :
  new_distance A C ^ 2 + new_distance C B ^ 2 = new_distance A B ^ 2 → False :=
sorry

theorem proposition_3 (A B C: ℝ × ℝ) (hA : A.1 = x1 ∧ A.2 = y1) (hB : B.1 = x2 ∧ B.2 = y2) (hC : C.1 = x ∧ C.2 = y) :
  new_distance A C + new_distance C B > new_distance A B → False :=
sorry

theorem correct_answer : (proposition_1 ∧ ¬ proposition_2 ∧ ¬ proposition_3) :=
by {
  split,
  exact proposition_1 /* with the necessary hypotheses */,
  split,
  exact proposition_2 /* with the necessary hypotheses */,
  exact proposition_3 /* with the necessary hypotheses */
}

end proposition_1_proposition_2_proposition_3_correct_answer_l594_594900


namespace even_number_of_odd_faces_even_number_of_odd_vertices_l594_594778

-- Definition of a polyhedron and its properties
structure Polyhedron where
  faces : Set (Set Nat)  -- faces represented as sets of sides (each number of sides is Nat)
  vertices : Set Nat  -- vertices represented as a set of natural numbers

-- Define the property of having an odd number of sides for faces
def odd_faces_count (P : Polyhedron) : Nat :=
  P.faces.toList.count (λ face => face.toList.length % 2 = 1)

-- Theorem that states the number of faces with an odd number of sides is even
theorem even_number_of_odd_faces (P : Polyhedron) : (odd_faces_count P) % 2 = 0 := sorry

-- Define the property of vertices having an odd number of edges
def vertex_degree (P : Polyhedron) (v : Nat) : Nat :=
  (P.faces.toList.flatMap (λ face => if v ∈ face then face.toList else [])).length

def odd_vertices_count (P : Polyhedron) : Nat :=
  P.vertices.toList.count (λ v => vertex_degree P v % 2 = 1)

-- Theorem that states the number of vertices with an odd number of incident edges is even
theorem even_number_of_odd_vertices (P : Polyhedron) : (odd_vertices_count P) % 2 = 0 := sorry

end even_number_of_odd_faces_even_number_of_odd_vertices_l594_594778


namespace fraction_multiplication_validity_l594_594214

theorem fraction_multiplication_validity (a b m x : ℝ) (hb : b ≠ 0) : 
  (x ≠ m) ↔ (b * (x - m) ≠ 0) :=
by
  sorry

end fraction_multiplication_validity_l594_594214


namespace problem_statement_l594_594675

theorem problem_statement (w x y z : ℕ) (h : 2^w * 3^x * 5^y * 7^z = 882) : 2 * w + 3 * x + 5 * y + 7 * z = 22 :=
sorry

end problem_statement_l594_594675


namespace amount_paid_correct_l594_594318

def initial_debt : ℕ := 100
def hourly_wage : ℕ := 15
def hours_worked : ℕ := 4
def amount_paid_before_work : ℕ := initial_debt - (hourly_wage * hours_worked)

theorem amount_paid_correct : amount_paid_before_work = 40 := by
  sorry

end amount_paid_correct_l594_594318


namespace monotonic_function_range_l594_594728

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * exp (a * x)

theorem monotonic_function_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y ∨ f a y ≤ f a x) ↔
  a ∈ set.Iic (-real.sqrt 2) ∪ set.Ioc 1 (real.sqrt 2) := 
sorry

end monotonic_function_range_l594_594728


namespace part1_intersection_1_part1_union_1_part2_range_a_l594_594689

open Set

def U := ℝ
def A (x : ℝ) := -1 < x ∧ x < 3
def B (a x : ℝ) := a - 1 ≤ x ∧ x ≤ a + 6

noncomputable def part1_a : ℝ → Prop := sorry
noncomputable def part1_b : ℝ → Prop := sorry

-- part (1)
theorem part1_intersection_1 (a : ℝ) : A x ∧ B a x := sorry

theorem part1_union_1 (a : ℝ) : A x ∨ B a x := sorry

-- part (2)
theorem part2_range_a : {a : ℝ | -3 ≤ a ∧ a ≤ 0} := sorry

end part1_intersection_1_part1_union_1_part2_range_a_l594_594689


namespace will_initial_money_l594_594771

theorem will_initial_money (spent_game : ℕ) (number_of_toys : ℕ) (cost_per_toy : ℕ) (initial_money : ℕ) :
  spent_game = 27 →
  number_of_toys = 5 →
  cost_per_toy = 6 →
  initial_money = spent_game + number_of_toys * cost_per_toy →
  initial_money = 57 :=
by
  intros
  sorry

end will_initial_money_l594_594771


namespace largest_solution_sqrt_2x_eq_4x_l594_594759

theorem largest_solution_sqrt_2x_eq_4x (x : ℝ) (hx : sqrt (2 * x) = 4 * x) : x ≤ 1 / 8 ∧ (∃ (y : ℝ), y = 1 / 8) :=
by
  sorry

end largest_solution_sqrt_2x_eq_4x_l594_594759


namespace monotonicity_of_f_range_of_k_l594_594577

def f (x : ℝ) (k : ℝ) : ℝ := (1/2) * x^2 + (1 - k) * x - k * Real.log x

theorem monotonicity_of_f (k : ℝ) : 
  (∀ x > 0, k ≤ 0 → (1/2 * x^2 + (1 - k) * x - k * Real.log x) > 0) ∧ 
  (k > 0 → (∀ x ∈ Icc (0:ℝ) k, f x k < 0 ↔ x < k) 
  ∧ (∀ x ∈ Ioc k +∞, f x k > 0 ↔ k < x)) := 
sorry

theorem range_of_k (k : ℝ) (x_0 : ℝ) (h : 0 < k ∧ f x_0 k < 3/2 - k^2) :
  k ∈ set.Ioo (0 : ℝ) 1 :=
sorry

end monotonicity_of_f_range_of_k_l594_594577


namespace find_sin_value_l594_594935

variable (x : ℝ)

theorem find_sin_value (h : Real.sin (x + Real.pi / 3) = Real.sqrt 3 / 3) : 
  Real.sin (2 * Real.pi / 3 - x) = Real.sqrt 3 / 3 :=
by 
  sorry

end find_sin_value_l594_594935


namespace composite_sequence_count_l594_594558

theorem composite_sequence_count {a1 : ℕ} (h_a1_digits : 10 ≤ a1 ∧ a1 < 100)
  (sequence : ℕ → ℕ) (h_sequence : ∀ n, ∃ d, d ≠ 9 ∧ sequence (n + 1) = sequence n * 10 + d) :
  ∃ n m, n ≠ m ∧ ¬nat.prime (sequence n) ∧ ¬nat.prime (sequence m) := 
sorry

end composite_sequence_count_l594_594558


namespace range_of_a_l594_594952

noncomputable def translated_problem (a : ℝ) : Prop :=
  ∀ (n : ℕ+), (-1 : ℝ) ^ (n : ℕ) * a < 3 + (-1 : ℝ) ^ ((n : ℕ) + 1) / (n : ℝ + 1)

theorem range_of_a (a : ℝ) (h : translated_problem a) : -3 ≤ a ∧ a < 2 :=
sorry

end range_of_a_l594_594952


namespace least_value_of_b_l594_594370

-- Statement of the problem in Lean 4
theorem least_value_of_b (a b : ℕ) (h1 : Nat.prime (Nat.sqrt a)) (h2 : a = (Nat.sqrt a) ^ 2) 
(h3 : b % a = 0) 
(h4 : Nat.factors_count b = a + 1) : 
b = 162 := 
sorry

end least_value_of_b_l594_594370


namespace slope_of_line_l594_594096

theorem slope_of_line (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (1, 2)) (h2 : (x2, y2) = (4, 8)) :
  (y2 - y1) / (x2 - x1) = 2 := 
by
  sorry

end slope_of_line_l594_594096


namespace probability_shaded_region_l594_594805

theorem probability_shaded_region :
  ∀ (T : Triangle) (subdivided : Subdivision T 6) (shaded_regions : Finset (Subdivision.Region subdivided)),
  shaded_regions.card = 2 → (∀ p : Point, p ∈ shaded_regions → probability p = 1 / 3) := 
sorry

end probability_shaded_region_l594_594805


namespace sin_double_alpha_sin_beta_l594_594225

-- Part 1
theorem sin_double_alpha (α : ℝ) (h₁ : α ∈ Ioo (π / 2) π) (h₂ : sin α = 1 / 3) : 
  sin (2 * α) = -4 * sqrt 2 / 9 := by
  sorry
  
-- Part 2
theorem sin_beta (α β : ℝ) 
  (h₁ : α ∈ Ioo (π / 2) π) 
  (h₂ : β ∈ Ioo 0 (π / 2)) 
  (h₃ : sin α = 1 / 3) 
  (h₄ : sin (α + β) = -3 / 5) : 
  sin β = (6 * sqrt 2 + 4) / 15 := by
  sorry

end sin_double_alpha_sin_beta_l594_594225


namespace count_valid_scalene_triangles_l594_594586

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_scalene (a b c : ℕ) : Prop :=
  a < b ∧ b < c

def is_valid_triangle (a b c : ℕ) : Prop :=
  is_triangle a b c ∧ is_scalene a b c ∧ a + b + c = 24

theorem count_valid_scalene_triangles : 
  (finset.univ.filter (λ (abc : ℕ × ℕ × ℕ), is_valid_triangle abc.1 abc.2.1 abc.2.2)).card = 6 :=
by {
  sorry
}

end count_valid_scalene_triangles_l594_594586


namespace first_percentage_increase_l594_594394

theorem first_percentage_increase (x : ℝ) :
  (1 + x / 100) * 1.4 = 1.82 → x = 30 := 
by 
  intro h
  -- start your proof here
  sorry

end first_percentage_increase_l594_594394


namespace angle_between_planes_l594_594461

theorem angle_between_planes :
  let n1 : ℝ × ℝ × ℝ := (2, 2, 1)
  let n2 : ℝ × ℝ × ℝ := (1, -1, 3)
  let dot_product := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3
  let magnitude_n1 := Real.sqrt (n1.1^2 + n1.2^2 + n1.3^2)
  let magnitude_n2 := Real.sqrt (n2.1^2 + n2.2^2 + n2.3^2)
  let cos_phi := dot_product / (magnitude_n1 * magnitude_n2)
in Real.arccos cos_phi = Real.arccos (1 / Real.sqrt 11) :=
by
  sorry

end angle_between_planes_l594_594461


namespace number_of_diagonal_intersections_of_convex_n_gon_l594_594203

theorem number_of_diagonal_intersections_of_convex_n_gon (n : ℕ) (h : 4 ≤ n) :
  (∀ P : Π m, m = n ↔ m ≥ 4, ∃ i : ℕ, i = n * (n - 1) * (n - 2) * (n - 3) / 24) := 
by
  sorry

end number_of_diagonal_intersections_of_convex_n_gon_l594_594203


namespace percentage_of_remainder_left_l594_594994

theorem percentage_of_remainder_left (initial_population died_population current_population: ℕ) 
(h1 : initial_population = 3800) 
(h2 : died_population = 10 * initial_population / 100) 
(h3 : current_population = 2907):
  (current_population = initial_population - died_population - 0.15 * (initial_population - died_population)) :=
by 
  sorry

end percentage_of_remainder_left_l594_594994


namespace selena_left_with_l594_594358

/-- Selena got a tip of $99 and spent money on various foods whose individual costs are provided. 
Prove that she will be left with $38. -/
theorem selena_left_with : 
  let tip := 99
  let steak_cost := 24
  let num_steaks := 2
  let burger_cost := 3.5
  let num_burgers := 2
  let ice_cream_cost := 2
  let num_ice_cream := 3
  let total_spent := (steak_cost * num_steaks) + (burger_cost * num_burgers) + (ice_cream_cost * num_ice_cream)
  tip - total_spent = 38 := 
by 
  sorry

end selena_left_with_l594_594358


namespace eccentricity_of_hyperbola_l594_594944

def hyperbola_eccentricity (a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : c^2 = a^2 + b^2)
    (h3 : real.dist (0, 0) (a, b) = (√3 / 4) * c) : Prop :=
  let e := c / a in e = 2

theorem eccentricity_of_hyperbola :
  ∃ (a b c : ℝ), 0 < a ∧ a < b ∧ c^2 = a^2 + b^2 ∧ real.dist (0, 0) (a, b) = (√3 / 4) * c ∧ hyperbola_eccentricity a b c 0 < a a < b c^2 = a^2 + b^2 = (√3 / 4) * c :=
sorry

end eccentricity_of_hyperbola_l594_594944


namespace company_packages_C_purchased_l594_594500

variables (a b c d : ℝ)
variables (m n : ℕ)
variables (A_cost B_cost C_cost : ℝ)
variables (A_profit B_profit C_profit total_margin : ℝ)

-- Defining the relations given in the problem
def condition_1 : Prop := a = 2 * b
def condition_2 : Prop := c + d = 3 * a
def condition_3 : Prop := c - d = 2 * a

-- Package costs (based on relationships established)
def package_A_cost (m : ℕ) : ℝ := m * b + a + 3 * c + 4 * d
def package_B_cost : ℝ := 5 * b + 2 * a + 4 * c + 6 * d
def package_C_cost : ℝ := 4 * b + 3 * a + 4 * c + 2 * d

-- Profit margins
def package_A_profit (m : ℕ) : ℝ := 0.075 * (package_A_cost m)
def package_B_profit : ℝ := 0.1 * package_B_cost
def package_C_profit : ℝ := 0.125 * package_C_cost

-- Total profit margin equating
def total_margin_equation (m n : ℕ) : Prop := 
  20 * package_A_profit m + package_B_profit * (92 - 20 - n) + package_C_profit * n = 92 * package_B_profit

-- The final assertion
theorem company_packages_C_purchased (m : ℕ) (h1 : condition_1) (h2 : condition_2) (h3 : condition_3) : 
  total_margin_equation m 17 → ∃ n, n = 17 := 
sorry

end company_packages_C_purchased_l594_594500


namespace triangle_perimeter_l594_594755

/-- Given a triangle DEF with side lengths DE = 150, EF = 250, FD = 200, and lines 
ℓ_D, ℓ_E, and ℓ_F are drawn parallel to the sides EF, FD, and DE respectively. 
The segments formed by the intersection of these lines have lengths 65, 55, and 25 respectively.
We need to prove that the perimeter of the triangle formed by these intersecting segments is 990. -/
theorem triangle_perimeter
  (DE EF FD: ℝ)
  (ℓ_D_segment ℓ_E_segment ℓ_F_segment: ℝ)
  (hDE: DE = 150)
  (hEF: EF = 250)
  (hFD: FD = 200)
  (hℓ_D_segment: ℓ_D_segment = 65)
  (hℓ_E_segment: ℓ_E_segment = 55)
  (hℓ_F_segment: ℓ_F_segment = 25):
  let perimeter := 990 in
  perimeter = 990 :=
by
  -- proof goes here
  sorry

end triangle_perimeter_l594_594755


namespace area_triangle_KDC_l594_594300

-- Define the geometric entities and given conditions
variables (O K A B C D : Point)
variables (r : ℝ) (l_KA l_CD : ℝ)
variable (are_collinear: Collinear {K, A, O, B})
variable (is_parallel: Parallel CD KA)

-- Basic geometric structure
axiom O_circle_center : Center O.Circle = O
axiom radius_O : Radius O.Circle = r
axiom length_CD : length CD = l_CD
axiom length_KA : length KA = l_KA

-- Specific provided measurements
axiom radius_10 : r = 10
axiom length_CD_12 : l_CD = 12
axiom length_KA_20 : l_KA = 20

-- Proven goal to establish
theorem area_triangle_KDC : area (triangle K D C) = 48 := by
  sorry

end area_triangle_KDC_l594_594300


namespace inequality_solution_range_of_m_l594_594254

-- Definition of the function
def f (x : ℝ) : ℝ := |x + 1|

-- First statement: solving the inequality f(x) ≥ 2x + 1
theorem inequality_solution (x : ℝ) : f(x) ≥ 2 * x + 1 ↔ x ≤ 0 := by
  sorry

-- Second statement: finding the range of m such that an inequality holds
theorem range_of_m (m : ℝ) : (∃ x : ℝ, f(x - 2) - f(x + 6) < m) ↔ m > -8 := by
  sorry

end inequality_solution_range_of_m_l594_594254


namespace photos_in_each_album_l594_594841

theorem photos_in_each_album (total_photos : ℕ) (number_of_albums : ℕ) (photos_per_album : ℕ) 
    (h1 : total_photos = 2560) 
    (h2 : number_of_albums = 32) 
    (h3 : total_photos = number_of_albums * photos_per_album) : 
    photos_per_album = 80 := 
by 
    sorry

end photos_in_each_album_l594_594841


namespace possible_denominators_count_l594_594710

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def valid_ab (a b : ℕ) : Prop :=
  is_digit a ∧ is_digit b ∧ a ≠ b ∧ a ≠ 1 ∧ b ≠ 1 

def num_possible_denominators : ℕ := 5

theorem possible_denominators_count (a b : ℕ) (h : valid_ab a b) :
  ∃ d ∈ {3, 9, 11, 33, 99}, d ∈ {k | ∃ n, 10 * a + b = 99 * n / k} → num_possible_denominators = 5 :=
sorry

end possible_denominators_count_l594_594710


namespace sufficient_not_necessary_condition_l594_594786

theorem sufficient_not_necessary_condition (x : ℝ) : (x ≥ 3 → (x - 2) ≥ 0) ∧ ((x - 2) ≥ 0 → x ≥ 3) = false :=
by
  sorry

end sufficient_not_necessary_condition_l594_594786


namespace part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l594_594269

-- Define initial conditions
def cost_price : ℝ := 20
def initial_selling_price : ℝ := 40
def initial_sales_volume : ℝ := 20
def price_decrease_per_kg : ℝ := 1
def sales_increase_per_kg : ℝ := 2
def original_profit : ℝ := 400

-- Part (1) statement
theorem part1_price_reduction_maintains_profit :
  ∃ x : ℝ, (initial_selling_price - x - cost_price) * (initial_sales_volume + sales_increase_per_kg * x) = original_profit ∧ x = 20 := 
sorry

-- Part (2) statement
theorem part2_profit_reach_460_impossible :
  ¬∃ y : ℝ, (initial_selling_price - y - cost_price) * (initial_sales_volume + sales_increase_per_kg * y) = 460 :=
sorry

end part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l594_594269


namespace cost_of_second_batch_l594_594749

theorem cost_of_second_batch
  (C_1 C_2 : ℕ)
  (quantity_ratio cost_increase: ℕ) 
  (H1 : C_1 = 3000) 
  (H2 : C_2 = 9600) 
  (H3 : quantity_ratio = 3) 
  (H4 : cost_increase = 1)
  : (∃ x : ℕ, C_1 / x = C_2 / (x + cost_increase) / quantity_ratio) ∧ 
    (C_2 / (C_1 / 15 + cost_increase) / 3 = 16) :=
by
  sorry

end cost_of_second_batch_l594_594749


namespace evaluate_complex_expr_l594_594108

-- Define the expression as described in the condition.
def complex_expr : ℝ :=
  2 * (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)

-- State the theorem that we want to prove.
theorem evaluate_complex_expr : complex_expr = 4000 :=
by
  -- Proof would go here
  sorry

end evaluate_complex_expr_l594_594108


namespace parabola_properties_l594_594019

noncomputable def sum_of_FV_lengths (d : ℝ) 
  (BF BV CV FV : ℝ) 
  (hBF : BF = 25) 
  (hBV : BV = 24) 
  (hCV : CV = 20) 
  (hFV : sum_of_all_values FV d) : 
  ℝ :=
let roots_sum := (50 : ℝ) / 3 in
roots_sum

theorem parabola_properties :
  ∀ d BF BV CV FV : ℝ,
  BF = 25 →
  BV = 24 →
  CV = 20 →
  (sum_of_FV_lengths d BF BV CV FV = 50 / 3) :=
by
  intros d BF BV CV FV hBF hBV hCV hFV
  apply hFV
  -- Proof here
  sorry

end parabola_properties_l594_594019


namespace cube_triangles_sum_areas_l594_594740

theorem cube_triangles_sum_areas (q r s : ℤ) :
  let A := 2 * 2 / 2
  let A1 := 6 * 4 * A
  let A2 := 2 * 2 * (2 * Real.sqrt 2) / 2
  let A3 := (Real.sqrt 3 / 4) * (2 * Real.sqrt 2) ^ 2
  let total_area := A1 + 12 * 2 * A2 + 8 * A3
  total_area = 48 + Real.sqrt (4608 : ℝ) + Real.sqrt (3072 : ℝ) ∧ q + r + s = 48 + 4608 + 3072 :=
begin
  sorry
end

end cube_triangles_sum_areas_l594_594740


namespace calculate_percentage_error_l594_594505

noncomputable def percentage_error_in_area
  (a : ℝ) -- original side length 'a'
  (measured_a : ℝ) -- measured side length '1.06 * a'
  (A : ℝ) -- actual area
  (A' : ℝ) -- area calculated with error
  (percentage_error : ℝ) -- percentage error
  : Prop :=
  measured_a = 1.06 * a ∧
  A = (sqrt 3 / 4) * a^2 ∧
  A' = (sqrt 3 / 4) * (measured_a)^2 ∧
  percentage_error = ((A' - A) / A) * 100

theorem calculate_percentage_error (a : ℝ) :
  percentage_error_in_area a (1.06 * a) ((sqrt 3 / 4) * a^2) ((sqrt 3 / 4) * (1.06 * a)^2) 12.36 :=
  by
    sorry

end calculate_percentage_error_l594_594505


namespace annual_population_decrease_due_to_migration_l594_594393

theorem annual_population_decrease_due_to_migration
  (r : ℝ) (x : ℝ) (R : ℝ) 
  (h_r : r = 0.06) 
  (h_R : R = 1.157625) :
  (1 + r) * (1 - x / 100) ^ 3 = R → x ≈ 0.9434 := 
sorry

end annual_population_decrease_due_to_migration_l594_594393


namespace limit_of_sequence_l594_594723

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := 1 + n / a n

theorem limit_of_sequence : 
  filter.tendsto (λ n, a n - real.sqrt n) filter.at_top (𝓝 (1 / 2)) :=
sorry

end limit_of_sequence_l594_594723


namespace quadratic_equation_of_list_l594_594765

def is_quadratic (eq : Polynomial ℝ) : Prop :=
  eq.degree = 2

def equations : List (Polynomial ℝ) :=
  [3 * Polynomial.x + Polynomial.C 1,
   Polynomial.x - 2 * Polynomial.x ^ 3 - Polynomial.C 3,
   Polynomial.x ^ 2 - Polynomial.C 5,
   2 * Polynomial.x + Polynomial.C 1 / Polynomial.x - Polynomial.C 3]

theorem quadratic_equation_of_list : 
  ∃ (eq : Polynomial ℝ), 
    eq ∈ equations ∧ is_quadratic eq ∧ 
    ∀ eq' ∈ equations, eq' ≠ eq → ¬ is_quadratic eq' :=
by
  sorry

end quadratic_equation_of_list_l594_594765


namespace fred_baseball_cards_l594_594908

theorem fred_baseball_cards :
  ∀ (fred_cards_initial melanie_bought : ℕ), fred_cards_initial = 5 → melanie_bought = 3 → fred_cards_initial - melanie_bought = 2 :=
by
  intros fred_cards_initial melanie_bought h1 h2
  sorry

end fred_baseball_cards_l594_594908


namespace inequality_for_positive_n_and_x_l594_594698

theorem inequality_for_positive_n_and_x (n : ℕ) (x : ℝ) (hn : n > 0) (hx : x > 0) :
  (x^(2 * n - 1) - 1) / (2 * n - 1) ≤ (x^(2 * n) - 1) / (2 * n) :=
by sorry

end inequality_for_positive_n_and_x_l594_594698


namespace correct_props_l594_594013

variable (k : ℝ)
variable (a b c : ℝ × ℝ)
variable (G A B C : Type)
variable [InnerProductSpace ℝ (ℝ × ℝ)]
variable [AddCommGroup G] [Module ℝ G] [Fin (3 : ℵ)]
variable [Module ℝ (ℝ × ℝ)]

open Finset

def proposition_A := ((3 / 2, k) : ℝ × ℝ) ∥ (k, 8) → k = 6

def proposition_B := (a • c = b • c ∧ c ≠ (0, 0)) → a = b

def proposition_C := is_centroid G A B C → (GA + GB + GC = (0 : ℝ × ℝ))

def proposition_D := ((-1, 1) : ℝ × ℝ)∥ ((2, 3) : ℝ × ℝ) → projection (Submodule.span ℝ {(-1, 1)}) (2, 3) = 1 / 2 • (-1, 1)

theorem correct_props : (proposition_C ∧ proposition_D) ∧ ¬proposition_A ∧ ¬proposition_B :=
proof sorry

end correct_props_l594_594013


namespace proof_problem_l594_594555

def a_n (n : ℕ) : ℕ := 2 * n + 1

def S_n (n : ℕ) : ℕ := n^2 + 2 * n

def b_n (n : ℕ) : ℚ := 1 / ((a_n n)^2 - 1 : ℚ)

def T_n (n : ℕ) : ℚ := (n : ℚ) / (4 * ((n + 1) : ℚ))

theorem proof_problem :
  (a_n 3 = 7) ∧
  (a_n 5 + a_n 7 = 26) ∧
  (S_n n = Σ i in finset.range (n + 1), a_n i) ∧
  (T_n n = Σ i in finset.range (n + 1), b_n i) :=
begin
  sorry
end

end proof_problem_l594_594555


namespace probability_digit_six_in_11_over_13_l594_594692

theorem probability_digit_six_in_11_over_13 :
  (0.\overline{846153} : Real) = 11/13 → 
  let digits := [8, 4, 6, 1, 5, 3] in 
  let count_6 := (digits.count 6) in
  (count_6 / digits.length : Real) = 1/6 := 
by 
  intro h_dec_rep
  let digits := [8, 4, 6, 1, 5, 3]
  let count_6 := digits.count 6
  have h_count : count_6 = 1 := sorry
  have h_length : digits.length = 6 := sorry
  rw [h_count, h_length]
  norm_num

end probability_digit_six_in_11_over_13_l594_594692


namespace line_intersects_circle_l594_594629

noncomputable theory
open_locale classical

-- Define the point P and the circle with center P and radius 5
def point_P : ℝ × ℝ := (4, 0)
def radius : ℝ := 5

-- Define the equation of the circle being centered at point P
def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + y^2 = radius^2

-- Define the line y = kx + 2, with the condition k ≠ 0
def line_eq (k x y: ℝ) : Prop := y = k * x + 2

theorem line_intersects_circle (k : ℝ) (h : k ≠ 0) : 
  ∃ x y : ℝ, circle_eq x y ∧ line_eq k x y :=
sorry

end line_intersects_circle_l594_594629


namespace negation_of_existential_statement_l594_594034

theorem negation_of_existential_statement (x : ℚ) :
  ¬ (∃ x : ℚ, x^2 = 3) ↔ ∀ x : ℚ, x^2 ≠ 3 :=
by sorry

end negation_of_existential_statement_l594_594034


namespace value_a_maximum_period_intervals_monotonic_increase_l594_594576

def f (a : ℝ) (x : ℝ) := 4 * Real.cos x * Real.sin (x + Real.pi / 6) + a

theorem value_a_maximum_period :
  (∀ x : ℝ, f a x ≤ 2) → a = -1 ∧ (∀ x, f (-1) x = f (-1) (x + π)) := by
  sorry

theorem intervals_monotonic_increase :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ π → 
  0 ≤ 2 * x + Real.pi / 6 ∧ 2 * x + Real.pi / 6 ≤ Real.pi / 2 → 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 6 ∨ (2 * Real.pi) / 3 ≤ x ∧ x ≤ Real.pi) := by
  sorry

end value_a_maximum_period_intervals_monotonic_increase_l594_594576


namespace f_g_comparison_l594_594917

def f (x : ℝ) : ℝ := 3 * x^2 - x + 1
def g (x : ℝ) : ℝ := 2 * x^2 + x - 1

theorem f_g_comparison (x : ℝ) : f(x) > g(x) := by
  sorry

end f_g_comparison_l594_594917


namespace octal_to_binary_l594_594518

theorem octal_to_binary (n : ℕ) (h : n = 135) : 
  -- Define octal to binary conversion for single digits
  let oct_to_bin : ℕ → string := λ x, 
    match x with
    | 0 => "000"
    | 1 => "001"
    | 2 => "010"
    | 3 => "011"
    | 4 => "100"
    | 5 => "101"
    | 6 => "110"
    | 7 => "111"
    | _ => ""
    end in

  -- Apply conversion to the octal digits of 135
  let bin_rep := oct_to_bin 1 ++ oct_to_bin 3 ++ oct_to_bin 5 in
  
  -- Remove leading zeros
  let bin_result := bin_rep.trim_left in

  bin_result = "1011101" :=
by
  -- Replace with actual proof steps
  sorry

end octal_to_binary_l594_594518


namespace students_just_passed_l594_594625

-- Define the total number of students
def total_students : ℕ := 300

-- Define the percentage of students in first division
def first_division_percentage : ℝ := 0.26

-- Define the percentage of students in second division
def second_division_percentage : ℝ := 0.54

-- Calculate the number of students in first division
def first_division_students : ℕ := (first_division_percentage * total_students).to_nat

-- Calculate the number of students in second division
def second_division_students : ℕ := (second_division_percentage * total_students).to_nat

-- State the main theorem
theorem students_just_passed : (total_students - (first_division_students + second_division_students)) = 60 := sorry

end students_just_passed_l594_594625


namespace collinear_points_l594_594912

structure Triangle :=
(A B C : Point)

structure Circle :=
(center : Point) (radius : ℝ)

structure Line :=
(point1 point2 : Point)

def incenter (T : Triangle) : Point := sorry
def circumcircle (T : Triangle) : Circle := sorry
def extend_to_circumcircle (L : Line) (C : Circle) : Point := sorry
def intersection (L1 L2 : Line) : Point := sorry

theorem collinear_points (T : Triangle) :
  let O := incenter T,
      circ := circumcircle T,
      D := extend_to_circumcircle (Line.mk O T.A) circ,
      E := extend_to_circumcircle (Line.mk O T.B) circ,
      F := extend_to_circumcircle (Line.mk O T.C) circ,
      G := intersection (Line.mk D E) (Line.mk T.A T.C),
      H := intersection (Line.mk D F) (Line.mk T.A T.B) in
  collinear H O G := 
sorry

end collinear_points_l594_594912


namespace exists_sequences_l594_594010

theorem exists_sequences (m n : Nat → Nat) (h₁ : ∀ k, m k = 2 * k) (h₂ : ∀ k, n k = 5 * k * k)
  (h₃ : ∀ (i j : Nat), (i ≠ j) → (m i ≠ m j) ∧ (n i ≠ n j)) :
  (∀ k, Nat.sqrt (n k + (m k) * (m k)) = 3 * k) ∧
  (∀ k, Nat.sqrt (n k - (m k) * (m k)) = k) :=
by 
  sorry

end exists_sequences_l594_594010


namespace instantaneous_velocity_at_1_sec_l594_594603

noncomputable def equation_of_motion (t : ℝ) : ℝ := (t + 1)^2 * (t - 1)

theorem instantaneous_velocity_at_1_sec :
  (derivative equation_of_motion 1) = 4 :=
sorry

end instantaneous_velocity_at_1_sec_l594_594603


namespace probability_is_one_fifth_l594_594484

variable (x : ℝ)
def equation : ℝ := (x^3) * (x + 14) * (2 * x + 5) * (x - 3) * (x + 7) * (3 * x - 4) * (x - 17) * (x + 25) * (4 * x - 18)

def set_of_numbers : Finset ℝ := {-25, -17, -10, -6, -5, -4, -2.5, -1, 0, 2.5, 4, 6, 7, 10, 18}

def solutions_in_set : Finset ℝ := set_of_numbers.filter (λ x, equation x = 0)

def probability_of_solution : ℝ := (solutions_in_set.card.to_real / set_of_numbers.card.to_real)

theorem probability_is_one_fifth : probability_of_solution = 1 / 5 :=
by
  sorry

end probability_is_one_fifth_l594_594484


namespace vector_BC_is_correct_l594_594562

-- Given points B(1,2) and C(4,5)
def point_B := (1, 2)
def point_C := (4, 5)

-- Define the vector BC
def vector_BC (B C : ℕ × ℕ) : ℕ × ℕ :=
  (C.1 - B.1, C.2 - B.2)

-- Prove that the vector BC is (3, 3)
theorem vector_BC_is_correct : vector_BC point_B point_C = (3, 3) :=
  sorry

end vector_BC_is_correct_l594_594562


namespace polynomial_sequence_symmetric_l594_594491

def P : ℕ → ℝ → ℝ → ℝ → ℝ 
| 0, x, y, z => 1
| (m + 1), x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem polynomial_sequence_symmetric (m : ℕ) (x y z : ℝ) (σ : ℝ × ℝ × ℝ): 
  P m x y z = P m σ.1 σ.2.1 σ.2.2 :=
sorry

end polynomial_sequence_symmetric_l594_594491


namespace sum_of_first_100_terms_l594_594398

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = (2 * |Real.sin (n * Real.pi / 2)| - 1) * a n + 2 * n

theorem sum_of_first_100_terms (a : ℕ → ℝ) (h : sequence a) :
  (Finset.range 100).sum a = 5100 :=
sorry

#check sum_of_first_100_terms

end sum_of_first_100_terms_l594_594398


namespace sequences_zero_at_2_l594_594930

theorem sequences_zero_at_2
  (a b c d : ℕ → ℝ)
  (h1 : ∀ n, a (n+1) = a n + b n)
  (h2 : ∀ n, b (n+1) = b n + c n)
  (h3 : ∀ n, c (n+1) = c n + d n)
  (h4 : ∀ n, d (n+1) = d n + a n)
  (k m : ℕ)
  (hk : 1 ≤ k)
  (hm : 1 ≤ m)
  (h5 : a (k + m) = a m)
  (h6 : b (k + m) = b m)
  (h7 : c (k + m) = c m)
  (h8 : d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 :=
by sorry

end sequences_zero_at_2_l594_594930


namespace smallest_interval_l594_594860

-- Define the probability measures P(C) and P(D | C)
variables (C D : Prop)
def P (A : Prop) [decidable A] : ℝ := sorry 

-- Given conditions
axiom hP_C : P C = 5 / 6
axiom hP_D_given_C : P (D ∧ C) / P C = 3 / 5

-- Define the probability of both C and D occurring
def q := P (D ∧ C)

-- The goal is to prove the interval containing q
theorem smallest_interval (hP_C : P C = 5 / 6)
  (hP_D_given_C : P (D ∧ C) / P C = 3 / 5) 
  : 1 / 2 ≤ q ∧ q ≤ 5 / 6 :=
sorry

end smallest_interval_l594_594860


namespace number_of_subsets_of_set_l594_594399

theorem number_of_subsets_of_set : (∀ (A : Set Int), A = { -1, 0, 1 } → A.Powerset.card = 8) :=
by
  intro A
  intro hA
  rw [hA]
  dsimp
  have h : A.card = 3 := sorry
  rw [←Set.card_powerset_eq_two_pow _ h]
  show 2 ^ 3 = 8
  norm_num

end number_of_subsets_of_set_l594_594399


namespace f_eq_n_l594_594667

noncomputable def f : ℕ → ℕ := sorry

axiom strictly_increasing (f : ℕ → ℕ) : ∀ m n, m < n → f(m) < f(n)
axiom f_2_eq_2 : f(2) = 2
axiom coprime_property (f : ℕ → ℕ) : ∀ m n, Nat.coprime m n → f(m * n) = f(m) * f(n)

theorem f_eq_n (f : ℕ → ℕ) (h1 : ∀ m n, m < n → f(m) < f(n)) (h2 : f(2) = 2) (h3 : ∀ m n, Nat.coprime m n → f(m * n) = f(m) * f(n)) : ∀ n, f(n) = n := 
by
  sorry

end f_eq_n_l594_594667


namespace length_of_bridge_eq_l594_594118

-- Definitions of the given conditions
def train_length : ℝ := 300
def crossing_time : ℝ := 45
def train_speed : ℝ := 47.99999999999999

-- Theorem stating that the length of the bridge 
-- is as calculated given the conditions above.
theorem length_of_bridge_eq :
  let distance := train_speed * crossing_time in
  let length_of_bridge := distance - train_length in
  length_of_bridge = 1859.9999999999995 :=
by
  sorry

end length_of_bridge_eq_l594_594118


namespace part_a_part_b_part_c_l594_594281

def quadradois (n : ℕ) : Prop :=
  ∃ (S1 S2 : ℕ), S1 ≠ S2 ∧ (S1 * S1 + S2 * S2 ≤ S1 * S1 + S2 * S2 + (n - 2))

theorem part_a : quadradois 6 := 
sorry

theorem part_b : quadradois 2015 := 
sorry

theorem part_c : ∀ (n : ℕ), n > 5 → quadradois n := 
sorry

end part_a_part_b_part_c_l594_594281


namespace const_product_l594_594661

def P (n m : ℕ) : ℚ := ∑ k in Finset.range(n + 1), (-1:ℚ)^k * Nat.choose n k * (m / (m + k))
def Q (n m : ℕ) : ℚ := Nat.choose (n + m) m

theorem const_product (n m : ℕ) (hm : 0 < m) (hn : 0 < n) : P n m * Q n m = 1 :=
by
  sorry

end const_product_l594_594661


namespace desired_avg_price_l594_594132

-- Definitions based on conditions
def num_shirts := 6
def first_two_shirts_prices := [40, 50]  -- List of prices for the first 2 shirts
def min_avg_price_remaining_shirts := 52.5  -- Minimum average price for the remaining 4 shirts

-- Theorem statement to prove the desired overall average price
theorem desired_avg_price :
  ∃ X : ℝ, (X = 50) ∧ 
      (∑ price in first_two_shirts_prices, price + min_avg_price_remaining_shirts * (num_shirts - first_two_shirts_prices.length) = num_shirts * X) :=
by
  sorry

end desired_avg_price_l594_594132


namespace cyclic_polygon_l594_594338

-- Definitions
variable (n : ℕ) (n_ge_4 : n ≥ 4)
variable (A : Fin n → ℝ) -- Represents the vertices of the polygon
variable (b c : Fin n → ℝ) -- Represents the pairs of real numbers

-- Conditions
variable h : ∀ (i j : Fin n), (i.val < j.val) → (A i - A (Fin.castSucc i)) = b j * c i - b i * c j

-- The Lean 4 statement of the problem
theorem cyclic_polygon (A : Fin n → ℝ) (b c : Fin n → ℝ) (h : ∀ (i j : Fin n), (i.val < j.val) → (A i - A (Fin.castSucc i)) = b j * c i - b i * c j) : 
  ∃ (O : ℝ) (R : ℝ), ∀ j, (A j - O) ^ 2 = R^2 :=
sorry

end cyclic_polygon_l594_594338


namespace problem1_problem2_l594_594570

open Real

-- Conditions
variable {f : ℝ → ℝ}
variable {x : ℝ}
variable {m : ℝ}

-- Proof for the monotonicity of g(x)
theorem problem1 (h : ∀ x > 1, (x + x * ln x) * deriv f x > f x) :
  ∀ x > 1, deriv (λ x, f x / (1 + ln x)) x > 0 := sorry

-- Proof for the range of m
theorem problem2 (h : ∀ x > 1, (x + x * ln x) * (exp x + m) > exp x + m * x) :
  m ≥ -2 * exp 1 := sorry

end problem1_problem2_l594_594570


namespace reflected_light_eq_l594_594471

theorem reflected_light_eq
  (incident_light : ∀ x y : ℝ, 2 * x - y + 6 = 0)
  (reflection_line : ∀ x y : ℝ, y = x) :
  ∃ x y : ℝ, x + 2 * y + 18 = 0 :=
sorry

end reflected_light_eq_l594_594471


namespace largest_possible_factors_l594_594045

theorem largest_possible_factors (x : ℝ) :
  ∃ m q1 q2 q3 q4 : polynomial ℝ,
    m = 4 ∧
    x^10 - 1 = q1 * q2 * q3 * q4 ∧
    ¬(q1.degree = 0) ∧ ¬(q2.degree = 0) ∧ ¬(q3.degree = 0) ∧ ¬(q4.degree = 0) :=
sorry

end largest_possible_factors_l594_594045


namespace count_even_3_digit_numbers_with_digit_sum_25_l594_594197

theorem count_even_3_digit_numbers_with_digit_sum_25 : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (nat.digits 10 n).sum = 25 ∧ n % 2 = 0 := 
begin 
  sorry 
end

end count_even_3_digit_numbers_with_digit_sum_25_l594_594197


namespace S_on_circumcircle_circle_S_through_C_and_I_l594_594359

open EuclideanGeometry

variables {A B C S I : Point}
variables (triangle_ABC : Triangle A B C)

-- Assume that S is the intersection of the angle bisector of ∠BAC and the perpendicular bisector of segment BC
axiom h₁ : IsIntersection S (AngleBisector (angle BAC A B C)) (PerpendicularBisector (segment B C))

-- Assume that the circle centered at S passes through B
axiom h₂ : CircleCenteredAt S (distance S B)

-- Prove that S lies on the circumcircle of triangle ABC
theorem S_on_circumcircle :
  On (Circumcircle triangle_ABC) S :=
sorry

-- Prove that the circle centered at S passing through B also passes through C and the incenter I of triangle ABC
theorem circle_S_through_C_and_I :
  (distance S C) = (distance S B) ∧ (distance S I) = (distance S B) :=
sorry

end S_on_circumcircle_circle_S_through_C_and_I_l594_594359


namespace bingo_first_column_possibilities_l594_594990

theorem bingo_first_column_possibilities :
  (∏ i in (Finset.range 5), (15 - i)) = 360360 :=
by
  sorry

end bingo_first_column_possibilities_l594_594990


namespace total_games_played_l594_594119

-- Definition of the conditions
def teams : Nat := 10
def games_per_pair : Nat := 4

-- Statement of the problem
theorem total_games_played (teams games_per_pair : Nat) : 
  teams = 10 → 
  games_per_pair = 4 → 
  ∃ total_games, total_games = 180 :=
by
  intro h1 h2
  sorry

end total_games_played_l594_594119


namespace sin_C_in_right_triangle_l594_594997

theorem sin_C_in_right_triangle 
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : B = π / 2)
  (h3 : sin A = 8 / 17) :
  sin C = 15 / 17 :=
by
  sorry

end sin_C_in_right_triangle_l594_594997


namespace circumcenter_signed_distance_sum_l594_594111

theorem circumcenter_signed_distance_sum
  (ABC : Triangle)          -- Triangle ∆ABC
  (O : Point)               -- Circumcenter O of ∆ABC
  (R r : ℝ)                 -- R is the circumradius, r is the inradius
  (d1 d2 d3 : ℝ)            -- Signed distances
  (h1 : is_circumcenter O ABC) -- O is the circumcenter of ∆ABC
  (h2 : distance_to_side O (side ABC BC) = d1) -- Distance from O to side BC
  (h3 : distance_to_side O (side ABC AC) = d2) -- Distance from O to side AC
  (h4 : distance_to_side O (side ABC AB) = d3) -- Distance from O to side AB
  : d1 + d2 + d3 = R + r := 
sorry

end circumcenter_signed_distance_sum_l594_594111


namespace find_the_number_l594_594021

theorem find_the_number : ∃ x : ℝ, (10 + x + 50) / 3 = (20 + 40 + 6) / 3 + 8 ∧ x = 30 := 
by
  sorry

end find_the_number_l594_594021


namespace total_medals_and_days_l594_594294

def medals_on_day (n m : ℕ) (day : ℕ) : ℕ :=
  day + (if day ≤ n then (m - (sum (range day))) / 7 else 0)

theorem total_medals_and_days (m n : ℕ) :
  (∀ day : ℕ, day ≤ n → medals_on_day n m day = day + (m - (sum (range day))) / 7) →
  (medals_on_day n m n = n) →
  m = 36 ∧ n = 6 :=
by
  sorry

end total_medals_and_days_l594_594294


namespace maximum_m_l594_594553

def is_sequence (A : List ℕ) (n : ℕ) : Prop :=
  A.length = n ∧ ∀ i, i < n → 0 < A.nth_le i (by linarith)

def m (A : List ℕ) : ℕ :=
  (List.range (A.length - 2)).count (λ i, A.nth_le i (by linarith) + 1 = A.nth_le (i + 1) (by linarith) ∧ A.nth_le (i + 1) (by linarith) + 1 = A.nth_le (i + 2) (by linarith))

theorem maximum_m :
  ∃ (A : List ℕ), is_sequence A 2005 ∧ m A = 668^2 * 669 := sorry

end maximum_m_l594_594553


namespace fence_cost_correct_l594_594098

def side_lengths : ℕ × ℕ × ℕ × ℕ × ℕ := (9, 12, 15, 11, 13)
def costs_per_foot : ℕ × ℕ × ℕ × ℕ × ℕ := (45, 55, 60, 50, 65)

def total_cost (sides costs : ℕ × ℕ × ℕ × ℕ × ℕ) : ℕ :=
  sides.1 * costs.1 + sides.2 * costs.2 + sides.3 * costs.3 + sides.4 * costs.4 + sides.5 * costs.5

theorem fence_cost_correct :
  total_cost side_lengths costs_per_foot = 3360 :=
  by sorry

end fence_cost_correct_l594_594098


namespace cross_section_area_of_inscribed_sphere_l594_594064

theorem cross_section_area_of_inscribed_sphere
  (a α : ℝ)
  (ha : 0 < a)
  (hα : 0 < α ∧ α < π)
  (H : True) -- Placeholder for "A sphere is inscribed in the pyramid" condition
  : 
  ∃ LP : ℝ, 
  LP^2 = (a^2 / 9) * (sin (α / 2)^2) * (sin ((π / 3) - α / 2) / sin ((π / 3) + α / 2)) →
  let cross_section_area := π * LP^2,
  cross_section_area = π * (a^2 / 9) * (sin (α / 2)^2) * (sin ((π / 3) - α / 2) / sin ((π / 3) + α / 2)) :=
sorry

end cross_section_area_of_inscribed_sphere_l594_594064


namespace triangle_side_length_l594_594311

variable {A B C a b c : ℝ}

theorem triangle_side_length (hcosA : cos A = √6 / 3) (hb : b = 2 * √2) (hc : c = √3)
  (h_cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * cos A) : a = √3 := by
  sorry

end triangle_side_length_l594_594311


namespace percentage_reduction_is_20_l594_594774

def original_sheets := 20
def original_lines_per_sheet := 55
def original_characters_per_line := 65
def retyped_lines_per_sheet := 65
def retyped_characters_per_line := 70

def total_characters_original : ℕ := 
  original_sheets * original_lines_per_sheet * original_characters_per_line

def total_characters_per_retyped_sheet : ℕ := 
  retyped_lines_per_sheet * retyped_characters_per_line

def retyped_sheets : ℕ :=
  (total_characters_original + total_characters_per_retyped_sheet - 1) / 
  total_characters_per_retyped_sheet

def percentage_reduction : ℚ := 
  ((original_sheets : ℚ) - (retyped_sheets : ℚ)) / (original_sheets : ℚ) * 100

theorem percentage_reduction_is_20 :
  percentage_reduction ≈ 20 := sorry

end percentage_reduction_is_20_l594_594774


namespace closest_ratio_adults_children_l594_594375

theorem closest_ratio_adults_children (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 2) (h3 : c ≥ 2) : 
  (a : ℚ) / (c : ℚ) = 1 :=
  sorry

end closest_ratio_adults_children_l594_594375


namespace eccentricity_of_hyperbola_distance_from_focus_to_asymptote_l594_594719

-- Given conditions
def a : ℝ := 4
def b : ℝ := 3
def c : ℝ := real.sqrt (a^2 + b^2) -- equivalent to sqrt(16 + 9) = 5
def hyperbola : Prop := ( ∀ x y : ℝ, (x^2)/16 - (y^2)/9 = 1 )

-- Statements to prove
theorem eccentricity_of_hyperbola (h : hyperbola) : c / a = 5 / 4 := sorry

theorem distance_from_focus_to_asymptote (focus : ℝ × ℝ) (asymptote_slope : ℝ → ℝ)
  (h : hyperbola) (hf : focus = (5, 0)) (ha : ∀ x : ℝ, asymptote_slope x = - (3/4) * x) :
  let d := abs ((3 * 5) + (4 * 0)) / real.sqrt ((3^2) + (4^2)) in d = 3 := sorry

end eccentricity_of_hyperbola_distance_from_focus_to_asymptote_l594_594719


namespace selection_of_individuals_l594_594357

theorem selection_of_individuals : 
  let boys := 4,
      girls := 3,
      total_individuals := boys + girls,
      select := 4 in
  (choose total_individuals select) - (choose boys select) = 34 :=
by sorry

end selection_of_individuals_l594_594357


namespace max_area_of_triangle_l594_594421

noncomputable def max_area_triangle (PQ QR PR : ℝ) : ℝ :=
  let x := (QR / 50) in
  let s := (PQ + QR + PR) / 2 in
  let area := (s * (s - PQ) * (s - QR) * (s - PR)).sqrt in
  area

theorem max_area_of_triangle (PQ QR PR : ℝ) (hPQ : PQ = 13) (hQR_PR_ratio : QR / PR = 50 / 51) :
  max_area_triangle PQ QR PR = 3565 :=
by {
  sorry
}

end max_area_of_triangle_l594_594421


namespace coins_evenly_distributed_l594_594782

theorem coins_evenly_distributed (
  chests : Fin 77 → ℕ
  (h2_76 : ∀ (k : Fin 76) (sub_k : Finset (Fin 77)), sub_k.card = k.succ → ∃ (x : ℕ), ∀ i ∈ sub_k, chests i = x
) : ∃ x : ℕ, ∀ i : Fin 77, chests i = x :=
by 
  sorry

end coins_evenly_distributed_l594_594782


namespace triangle_angleC_l594_594287

noncomputable def angleC_possible_values
  (a b : ℝ) (A : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hA : A = Real.pi / 4) : Prop :=
  ∃ C : ℝ, (C = Real.pi * 5 / 12 ∨ C = Real.pi / 12)

theorem triangle_angleC
  (a b : ℝ) (A C : ℝ) 
  (ha : a = Real.sqrt 2) 
  (hb : b = Real.sqrt 3) 
  (hA : A = Real.pi / 4) 
  (hC: C = Real.pi * 5 / 12 ∨ C = Real.pi / 12) :
  angleC_possible_values a b A :=
begin
  sorry
end

end triangle_angleC_l594_594287


namespace two_linear_equations_with_two_variables_l594_594838

def isLinearEquationWithTwoVariables (eq : Prop) : Prop :=
  ∃ (a b c : ℚ), eq = (a * (x : ℚ) + b * (y : ℚ) = c)

def eq1 : Prop := 4 * (x : ℚ) - 7 = 0
def eq2 : Prop := 3 * (x : ℚ) + (y : ℚ) = z
def eq3 : Prop := (x : ℚ) - 7 = (x : ℚ)^2
def eq4 : Prop := 4 * (x : ℚ) * (y : ℚ) = 3
def eq5 : Prop := (x : ℚ + y : ℚ) / 2 = y : ℚ / 3
def eq6 : Prop := 3 / (x : ℚ) = 1
def eq7 : Prop := (y : ℚ) * ((y : ℚ) - 1) = (y : ℚ)^2 - (x : ℚ)

theorem two_linear_equations_with_two_variables : 
  (isLinearEquationWithTwoVariables eq5) ∧ (isLinearEquationWithTwoVariables eq7) :=
  begin
    sorry
  end

end two_linear_equations_with_two_variables_l594_594838


namespace four_digit_palindromes_are_multiple_of_11_l594_594599

-- Define what it means to be a four-digit palindrome
def is_four_digit_palindrome (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

-- Prove that a four-digit palindrome is a multiple of 11
theorem four_digit_palindromes_are_multiple_of_11 (n : ℕ) (h : is_four_digit_palindrome n) : 11 ∣ n :=
by
  rcases h with ⟨a, b, ha1, ha9, hb0, hb9, hn⟩
  have h1 : n = 1001 * a + 110 * b := by
    rw [hn]
    simp
  use (91 * a + 10 * b)
  rw [h1]
  ring

end four_digit_palindromes_are_multiple_of_11_l594_594599


namespace minute_hand_distance_traveled_l594_594390

noncomputable def radius : ℝ := 8
noncomputable def minutes_in_one_revolution : ℝ := 60
noncomputable def total_minutes : ℝ := 45

theorem minute_hand_distance_traveled :
  (total_minutes / minutes_in_one_revolution) * (2 * Real.pi * radius) = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_traveled_l594_594390


namespace upper_limit_b_l594_594450

theorem upper_limit_b (a b : ℕ) (h1 : 39 < a) (h2 : a < 51) (h3 : 49 < b) (h4 : 0.33333333333333337 ≤ a / b) (h5 : a / b ≤ 0.33333333333333337) :
  b ≤ 120 :=
by
  sorry

end upper_limit_b_l594_594450


namespace students_taking_neither_l594_594476

def total_students : ℕ := 1200
def music_students : ℕ := 60
def art_students : ℕ := 80
def sports_students : ℕ := 30
def music_and_art_students : ℕ := 25
def music_and_sports_students : ℕ := 15
def art_and_sports_students : ℕ := 20
def all_three_students : ℕ := 10

theorem students_taking_neither :
  total_students - (music_students + art_students + sports_students 
  - music_and_art_students - music_and_sports_students - art_and_sports_students 
  + all_three_students) = 1080 := sorry

end students_taking_neither_l594_594476


namespace paired_divisors_prime_properties_l594_594336

theorem paired_divisors_prime_properties (n : ℕ) (h : n > 0) (h_pairing : ∃ (pairing : (ℕ × ℕ) → Prop), 
  (∀ d1 d2 : ℕ, 
    pairing (d1, d2) → d1 * d2 = n ∧ Prime (d1 + d2))) : 
  (∀ (d1 d2 : ℕ), d1 ≠ d2 → d1 + d2 ≠ d3 + d4) ∧ (∀ p : ℕ, Prime p → ¬ p ∣ n) :=
by
  sorry

end paired_divisors_prime_properties_l594_594336


namespace problem_statement_l594_594937

-- Define the sequence a_n
def a (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), (Real.sin (i : ℝ)) / 2 ^ (i + 1)

-- The theorem to be proved
theorem problem_statement (m n : ℕ) (h : m > n) : |a n - a m| < 1 / 2 ^ n :=
  sorry

end problem_statement_l594_594937


namespace log_equals_third_term_l594_594027

def a_n (n : ℕ) : ℝ := real.logb 2 (n^2 + 3) - 2

theorem log_equals_third_term (n : ℕ) : 
  a_n n = real.logb 2 3 ↔ n = 3 :=
by
  sorry

end log_equals_third_term_l594_594027


namespace sufficient_but_not_necessary_condition_l594_594939

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|)
  ∧ (∃ y : ℝ, ¬ (y ≥ 1) ∧ |y + 1| + |y - 1| = 2 * |y|) :=
by
  sorry

end sufficient_but_not_necessary_condition_l594_594939


namespace find_triples_l594_594196

theorem find_triples (x y z : ℕ) :
  (x + 1)^(y + 1) + 1 = (x + 2)^(z + 1) ↔ (x = 1 ∧ y = 2 ∧ z = 1) :=
sorry

end find_triples_l594_594196


namespace rectangle_area_l594_594377

theorem rectangle_area
  (h_circle : ∀ x y, 2 * x^2 = -2 * y^2 + 16 * x - 8 * y + 40)
  (h_inscribed : ∃ cx cy r, ∀ x y, (x - cx)^2 + (y - cy)^2 = r^2)
  (h_rectangle_parallel : ∃ l w, ∀ p1 p2, p1.y = p2.y ↔ p1.y // sides parallel to x-axis)
  (h_length_twice_width : ∃ l w, l = 2 * w) :
  ∃ A, A = 96 :=
by
  sorry

end rectangle_area_l594_594377


namespace count_five_digit_numbers_not_divisible_by_1000_l594_594958

theorem count_five_digit_numbers_not_divisible_by_1000 (a b c d e : ℕ) 
  (h1 : 10000 ≤ a * 10000 + b * 1000 + c * 100 + d * 10 + e)
  (h2 : a ≠ 0)
  (h3 : a % 2 = 0)
  (h4 : c % 2 = 0)
  (h5 : e % 2 = 0) 
  (h6 : (a * 10000 + b * 1000 + c * 100 + d * 10 + e) % 1000 ≠ 0) :
  ∑ a in finset.range 9, ∑ b in finset.range 10, ∑ c in finset.range 10, finset.card {e | e % 2 = 0} * ∑ d in finset.range 10, 1 = 9960 := 
sorry

end count_five_digit_numbers_not_divisible_by_1000_l594_594958


namespace find_equations_and_distance_l594_594634

-- Parametric equations of curve C1
def parametric_eq_C1 (θ : ℝ) : ℝ × ℝ :=
  (1 + sqrt 3 * cos θ, sqrt 3 * sin θ)

-- Point P on curve C2 such that OP = 2OM
def point_P_on_C2 (OM OP : ℝ × ℝ) : Prop :=
  OP = (2 * OM.1, 2 * OM.2)

-- Cartesian equation of curve C2
def cartesian_eq_C2 (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 12

-- Polar equation of curve C1
def polar_eq_C1 (ρ θ : ℝ) : Prop :=
  ρ^2 - 2 * ρ * cos θ - 2 = 0

-- Polar equation of curve C2
def polar_eq_C2 (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * cos θ - 8 = 0

theorem find_equations_and_distance :
  (∀ (θ : ℝ), ∃ (ρ : ℝ), polar_eq_C1 ρ (π / 3) → ρ = 2) →
  (∀ (θ : ℝ), ∃ (ρ : ℝ), polar_eq_C2 ρ (π / 3) → ρ = 4) →
  (c ×) ⟨(x, y) ⟩
  (∀ (M P : ℝ × ℝ), point_P_on_C2 (parametric_eq_C1 (π / 3)) P → (cartesian_eq_C2 P.1 P.2)) → 2 :=
sorry

end find_equations_and_distance_l594_594634


namespace margaret_speed_on_time_l594_594690
-- Import the necessary libraries from Mathlib

-- Define the problem conditions and state the theorem
theorem margaret_speed_on_time :
  ∃ r : ℝ, (∀ d t : ℝ,
    d = 50 * (t - 1/12) ∧
    d = 30 * (t + 1/12) →
    r = d / t) ∧
  r = 37.5 := 
sorry

end margaret_speed_on_time_l594_594690


namespace triangle_is_isosceles_if_median_bisects_perimeter_l594_594032

-- Defining the sides of the triangle
variables {a b c : ℝ}

-- Defining the median condition
def median_bisects_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 2 * (a/2 + b)

-- The main theorem stating that the triangle is isosceles if the median bisects the perimeter
theorem triangle_is_isosceles_if_median_bisects_perimeter (a b c : ℝ) 
  (h : median_bisects_perimeter a b c) : b = c :=
by
  sorry

end triangle_is_isosceles_if_median_bisects_perimeter_l594_594032


namespace function_properties_l594_594726
noncomputable section

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x - sin x

-- State the theorem
theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, 1 - cos x ≥ 0) :=
by
  split
  { -- Prove that f is odd
    intro x
    simp [f, sin, add_comm] }

  { -- Prove that f'(x) ≥ 0
    intro x
    have h : deriv f x = 1 - cos x := deriv_sub (deriv_id x) (deriv_sin x)
    rw h
    exact sub_nonneg_of_le (cos_le_one x) }

end function_properties_l594_594726


namespace new_average_after_deductions_l594_594454

noncomputable def deducted_avg (xs : List ℤ) (n : ℤ) : ℤ :=
(xs.take n.toNat).sum

theorem new_average_after_deductions :
  ∀ (xs : list ℤ), (∀ i < 10, xs[i] = xs.head + i) →
  list.average xs = 11 →
  list.average (list.map_with_index (λ i x, x - (9 - i)) xs) = 7 :=
by sorry

end new_average_after_deductions_l594_594454


namespace Jill_age_l594_594701

variable (J R : ℕ) -- representing Jill's current age and Roger's current age

theorem Jill_age :
  (R = 2 * J + 5) →
  (R - J = 25) →
  J = 20 :=
by
  intros h1 h2
  sorry

end Jill_age_l594_594701


namespace solve_x_1_solve_x_2_solve_x_3_l594_594205

-- Proof 1: Given 356 * x = 2492, prove that x = 7
theorem solve_x_1 (x : ℕ) (h : 356 * x = 2492) : x = 7 :=
sorry

-- Proof 2: Given x / 39 = 235, prove that x = 9165
theorem solve_x_2 (x : ℕ) (h : x / 39 = 235) : x = 9165 :=
sorry

-- Proof 3: Given 1908 - x = 529, prove that x = 1379
theorem solve_x_3 (x : ℕ) (h : 1908 - x = 529) : x = 1379 :=
sorry

end solve_x_1_solve_x_2_solve_x_3_l594_594205


namespace sum_of_valid_primes_eq_431_l594_594658

open Nat

-- Define n as the product of the first 2013 primes
noncomputable def n : Nat := Nat.prod (Finset.filter Nat.prime (Finset.range (Prime.prime 2013 + 1)))

-- Define a helper function to check if a number is even but not a power of 2
def even_but_not_power_of_2 (x : Nat) : Bool :=
  x % 2 = 0 && ¬ (∃ k, 2^k = x)

-- Define the condition that needs to be checked for each prime in the range
def condition (p : Nat) :=
  p ≥ 20 ∧ p ≤ 150 ∧
  (even_but_not_power_of_2 ((p + 1) / 2)) ∧
  (∃ a b c, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (a^n * (a - b) * (a - c) + b^n * (b - c) * (b - a) + c^n * (c - a) * (c - b)) % p = 0 ∧
    (a^n * (a - b) * (a - c) + b^n * (b - c) * (b - a) + c^n * (c - a) * (c - b)) % (p^2) ≠ 0)

-- Find all primes satisfying the condition
def valid_primes : List Nat :=
  List.filter (fun p => condition p)
    (List.filter Nat.prime (List.range' 20 131))

-- Sum of valid primes
theorem sum_of_valid_primes_eq_431 : (List.sum valid_primes) = 431 :=
by
  sorry  -- The proof is omitted

end sum_of_valid_primes_eq_431_l594_594658


namespace orthonormal_basis_constructed_l594_594430

noncomputable def g1 : ℝ × ℝ × ℝ := (1, 1, 0)
noncomputable def g2 : ℝ × ℝ × ℝ := (2, 0, 1)
noncomputable def g3 : ℝ × ℝ × ℝ := (0, 1, -2)

noncomputable def e1 : ℝ × ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2, 0)
noncomputable def e2 : ℝ × ℝ × ℝ := (Real.sqrt 3 / 3, -Real.sqrt 3 / 3, Real.sqrt 3 / 3)
noncomputable def e3 : ℝ × ℝ × ℝ := (Real.sqrt 6 / 6, -Real.sqrt 6 / 6, -Real.sqrt 6 / 3)

theorem orthonormal_basis_constructed : 
  let g1 := g1
  let g2 := g2
  let g3 := g3
in OrthonormalBasisUsingSchmidt g1 g2 g3 = {e1, e2, e3} := sorry

end orthonormal_basis_constructed_l594_594430


namespace count_multiples_5_or_10_l594_594590

theorem count_multiples_5_or_10 (n : ℕ) (hn : n = 999) : 
  ∃ k : ℕ, k = 199 ∧ (∀ i : ℕ, i < 1000 → (i % 5 = 0 ∨ i % 10 = 0) → i = k) := 
by {
  sorry
}

end count_multiples_5_or_10_l594_594590


namespace sum_max_min_values_l594_594549

open Real

def fx (x : ℝ) : ℝ := max (sin x) (max (cos x) ((sin x + cos x) / sqrt 2))

theorem sum_max_min_values :
  ∃ x_max x_min : ℝ, (∀ x : ℝ, fx x ≤ x_max) ∧ (∀ x : ℝ, x_min ≤ fx x) ∧ (x_max + x_min = 1 - sqrt 2 / 2) :=
by
  sorry

end sum_max_min_values_l594_594549


namespace min_values_and_roots_l594_594333

theorem min_values_and_roots
    (a b : ℕ) (ha : a > 0) (hb : b > 0)
    (x1 x2 : ℝ)
    (hx : -1 < x1 ∧ x1 < x2 ∧ x2 < 0)
    (roots : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + 1 = 0 ∧ a * x2^2 + b * x2 + 1 = 0) :
  a ≥ 5 ∧ b ≥ 5 ∧
  (a = 5 → b = 5 → x1 = (real.sqrt 5 - 5) / 10 → x2 = (real.sqrt 5 + 5) / 10) := by
  sorry

end min_values_and_roots_l594_594333


namespace line_intersects_circle_l594_594734

theorem line_intersects_circle (k : ℝ) : ∀ (x y : ℝ),
  (x + y) ^ 2 = x ^ 2 + y ^ 2 →
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2)) ∧ 
  ((-1/2)^2 + (0)^2 < 1) →
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = k * (x + 1 / 2) := 
by
  intro x y h₁ h₂
  sorry

end line_intersects_circle_l594_594734


namespace bulldozer_cannot_collide_both_sides_l594_594630

-- Define a wall as a non-intersecting line segment
structure Wall (α : Type*) :=
(start_point : α) (end_point : α)
(non_intersecting : Prop) -- This property signifies no two walls intersect
(not_parallel_to_axes : Prop) -- This property signifies that the wall is not parallel to the coordinate axes

-- Main theorem statement
theorem bulldozer_cannot_collide_both_sides
  (walls : list (Wall ℝ)) -- Assume finite walls
  (non_intersecting_walls : ∀ (w1 w2 : Wall ℝ), w1 ≠ w2 → ¬(w1.non_intersecting ∧ w2.non_intersecting))
  (not_parallel : ∀ (w : Wall ℝ), w.not_parallel_to_axes) :
  ∀ (start : ℝ × ℝ), ¬ (∃ (w : Wall ℝ), BulldozerCollidesBothSides start w) :=
by
  sorry

end bulldozer_cannot_collide_both_sides_l594_594630


namespace find_specific_n_l594_594530

theorem find_specific_n :
  ∀ (n : ℕ), (∃ (a b : ℤ), n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by {
  sorry
}

end find_specific_n_l594_594530


namespace largest_possible_factors_l594_594046

theorem largest_possible_factors :
  ∃ (m : ℕ) (q : Fin m → Polynomial ℝ),
    (x : ℝ) → x^10 - 1 = ∏ i, q i ∧ ∀ i,  degree (q i) > 0 ∧ m = 3 :=
by
  sorry

end largest_possible_factors_l594_594046


namespace partner_Q_investment_duration_l594_594081

theorem partner_Q_investment_duration
  (investment_ratio_PQ_R : ℚ := 3 / 4)
  (investment_ratio_QR_R : ℚ := 4 / 5)
  (profit_ratio_PQ_R : ℚ := 9 / 16)
  (profit_ratio_QR_R : ℚ := 16 / 25)
  (time_P : ℚ := 4)
  (time_R : ℚ := 10) :
  (∃ T : ℚ, 4 * T / 50 = 16 / 25 ∧ 12 / (4 * T) = 9 / 16) → T = 8 :=
begin
  intro h,
  cases h with T hT,
  have h1 : 4 * T / 50 = 16 / 25 := hT.left,
  have h2 : 12 / (4 * T) = 9 / 16 := hT.right,
  sorry
end

end partner_Q_investment_duration_l594_594081


namespace magnitude_of_z_l594_594919

theorem magnitude_of_z (z : ℂ) (h : (1 + complex.i) * z = 2 * complex.i) : complex.abs z = real.sqrt 2 :=
sorry

end magnitude_of_z_l594_594919


namespace best_statistical_measure_for_common_prosperity_l594_594397

def common_prosperity (mean : ℝ) (variance : ℝ) : Prop :=
  (∃ (high_income_threshold : ℝ) (low_variance_threshold : ℝ), mean ≥ high_income_threshold ∧ variance ≤ low_variance_threshold)

theorem best_statistical_measure_for_common_prosperity :
  ∀ (mean variance : ℝ), common_prosperity mean variance ↔ 
  mean > 0 ∧ variance < Inf {v | v > 0} :=
sorry

end best_statistical_measure_for_common_prosperity_l594_594397


namespace correct_statements_l594_594946

-- Defining the binomial expansion and related properties
def a : ℝ := 1
def expr (x : ℝ) := (a * Real.sqrt x + 1 / (x^2))^10

-- Stating the properties to prove
theorem correct_statements :
  let x_val : ℝ := 1 in
  expr x_val = 1024 ∧
  (∃ T_r : ℕ → ℝ, ∀ r ≠ 5, T_r 5 > T_r r) ∧
  (∃ r_const : ℕ, r_const = 2) ∧
  (∃ n_rat_terms : ℕ, n_rat_terms = 6) :=
by
  sorry

end correct_statements_l594_594946


namespace dice_probability_prime_or_even_l594_594085

/-- Two 8-sided dice are tossed. The probability that the sum of the numbers shown on the dice is a prime number or an even number is 27/32. -/
theorem dice_probability_prime_or_even : 
  let outcomes := [(i, j) | i in (List.range 8).map (λ x => x + 1), j in (List.range 8).map (λ x => x + 1)],
      is_prime := λ n => n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13,
      is_even := λ n => n % 2 = 0,
      favorable_outcomes := List.countp (λ (x : ℕ × ℕ) => is_prime (x.1 + x.2) ∨ is_even (x.1 + x.2)) outcomes
  in favorable_outcomes / 64 = 27 / 32 :=
by
  sorry

end dice_probability_prime_or_even_l594_594085


namespace adam_played_rounds_l594_594106

theorem adam_played_rounds (total_points points_per_round : ℕ) (h_total : total_points = 283) (h_per_round : points_per_round = 71) : total_points / points_per_round = 4 := by
  -- sorry is a placeholder for the actual proof
  sorry

end adam_played_rounds_l594_594106


namespace parity_equiv_l594_594680

open Nat

theorem parity_equiv (p q : ℕ) : (Even (p^3 - q^3) ↔ Even (p + q)) :=
by
  sorry

end parity_equiv_l594_594680


namespace whiskers_ratio_l594_594215

/-- Four cats live in the old grey house at the end of the road. Their names are Puffy, Scruffy, Buffy, and Juniper.
Puffy has three times more whiskers than Juniper, but a certain ratio as many as Scruffy. Buffy has the same number of whiskers
as the average number of whiskers on the three other cats. Prove that the ratio of Puffy's whiskers to Scruffy's whiskers is 1:2
given Juniper has 12 whiskers and Buffy has 40 whiskers. -/
theorem whiskers_ratio (J B P S : ℕ) (hJ : J = 12) (hB : B = 40) (hP : P = 3 * J) (hAvg : B = (P + S + J) / 3) :
  P / gcd P S = 1 ∧ S / gcd P S = 2 := by
  sorry

end whiskers_ratio_l594_594215


namespace sum_mod_500_l594_594678

theorem sum_mod_500 :
  (∑ n in finset.range 335, (-1) ^ n * nat.choose 1002 (3 * n)) % 500 = 6 :=
by
  sorry

end sum_mod_500_l594_594678


namespace cyclic_quad_inequality_l594_594229

theorem cyclic_quad_inequality (AB CD AD BC AC BD : ℝ) : 
  ∀ (ABCD_cyclic : IsCyclicQuadrilateral AB CD AD BC AC BD), |AB - CD| + |AD - BC| ≥ 2 * |AC - BD| :=
by
  sorry

end cyclic_quad_inequality_l594_594229


namespace largest_base_condition_l594_594099

-- Define sum of digits in a given base
def sum_of_digits_in_base (n : ℕ) (b : ℕ) : ℕ :=
  (nat.digits b n).sum

theorem largest_base_condition (y : ℕ) (h1 : y = 12^4) :
  ∀ b > 10, sum_of_digits_in_base y b ≠ 32 :=
by
  sorry

end largest_base_condition_l594_594099


namespace Amit_left_after_3_days_l594_594502

-- Definitions based on the problem conditions.
def Amit_work_rate (W : ℝ) : ℝ := W / 15
def Ananthu_work_rate (W : ℝ) : ℝ := W / 30
def total_days : ℝ := 27

-- Proposition statement.
theorem Amit_left_after_3_days (W : ℝ) (x : ℝ) (h1 : Amit_work_rate W = W / 15)
    (h2 : Ananthu_work_rate W = W / 30)
    (h3 : x * (W / 15) + (total_days - x) * (W / 30) = W) : x = 3 :=
by
  sorry

end Amit_left_after_3_days_l594_594502


namespace color_uniqueness_l594_594745

-- Definitions that correspond to the conditions of the problem
def total_pieces : ℕ := 1987

def is_red (x : ℕ) : Prop := x ≠ 0
def is_yellow (y : ℕ) : Prop := y ≠ 0
def is_blue (z : ℕ) : Prop := z ≠ 0

-- Hypothesis: sum of the pieces
axiom total_color_condition (x y z : ℕ) : x + y + z = total_pieces

-- Theorem: It is possible to make all pieces the same color, and the final color is the same regardless of the order
theorem color_uniqueness (x y z : ℕ) (hx : is_red x) (hy : is_yellow y) (hz : is_blue z) 
  (h : x + y + z = total_pieces) : 
  ∃ (c : ℕ), (∀ x', x' = c) ∧ (∀ x' y', (x', y' ≠ c) -> (oper x' y' = c))
sorry

end color_uniqueness_l594_594745


namespace closest_distance_is_90_l594_594770

-- Definitions according to conditions provided in the problem.
def Sally_speed_kmh : ℝ := 80
def seconds_per_pole : ℝ := 4

-- Convert speed from km/h to m/s.
def Sally_speed_ms : ℝ := (Sally_speed_kmh * 1000) / 3600

-- Calculate the distance Sally's car travels in the given time.
def distance_between_poles : ℝ := Sally_speed_ms * seconds_per_pole

-- The theorem we need to prove: The closest distance between two poles is 90 meters.
theorem closest_distance_is_90 : distance_between_poles ≈ 90 :=
sorry

end closest_distance_is_90_l594_594770


namespace number_of_paths_l594_594473

-- Assume number of ways for each segment
axiom num_ways_A_C : ℕ
axiom num_ways_C_D : ℕ
axiom num_ways_D_B : ℕ

-- Let the conditions be defined
def conditions (A B C D : Type) (path_AC path_CD path_DB : ℕ) : Prop :=
  path_AC = num_ways_A_C ∧ path_CD = num_ways_C_D ∧ path_DB = num_ways_D_B

-- The main theorem to prove
theorem number_of_paths (A B C D : Type) (path_AC path_CD path_DB : ℕ) 
    (h : conditions A B C D path_AC path_CD path_DB) : 
        path_AC * path_CD * path_DB = 54 :=
by 
  -- Here we list the conditions
  have h1: path_AC = 3 := by sorry
  have h2: path_CD = 6 := by sorry
  have h3: path_DB = 3 := by sorry
  -- Calculate the total number of paths
  calc 
    path_AC * path_CD * path_DB
        = 3 * 6 * 3 : by rw [h1, h2, h3]
    ... = 54 : by norm_num

#check number_of_paths

end number_of_paths_l594_594473


namespace simplify_expr_l594_594279

-- Define the condition
def y : ℕ := 77

-- Define the expression and the expected result
def expr := (7 * y + 77) / 77

-- The theorem statement
theorem simplify_expr : expr = 8 :=
by
  sorry

end simplify_expr_l594_594279


namespace percentage_to_add_for_30_l594_594498

-- Definitions based on the conditions
def percentageOf (x : ℝ) (y : ℝ) : ℝ := (x / 100) * y

-- Condition: 15% of 50
def fifteenPercentOfFifty : ℝ := percentageOf 15 50

-- Condition: sum of x% of 30 and 15% of 50 should equal 10.5
def equation (x : ℝ) : Prop := percentageOf x 30 + fifteenPercentOfFifty = 10.5

-- Main theorem statement
theorem percentage_to_add_for_30 :
  ∃ x : ℝ, equation x ∧ x = 10 := by
  sorry

end percentage_to_add_for_30_l594_594498


namespace distinct_numbers_count_l594_594392

/-- Definition of possible numbers obtained from manipulating "日". -/
def possible_numbers : Set ℕ :=
  {0, 1, 3, 4, 5}

/-- Theorem: The number of distinct numbers that can be obtained by removing matchsticks from "日". -/
theorem distinct_numbers_count (S : Set ℕ) : S = possible_numbers → S.card = 5 :=
by
  intros
  sorry

end distinct_numbers_count_l594_594392


namespace not_R2_seq_first_4_terms_R0_seq_a5_eq_1_exists_p_Sn_geq_S10_l594_594179

-- Define the sequence and conditions
def R_seq (a : ℕ → ℤ) (p : ℤ) : Prop :=
  (a 1 + p >= 0) ∧ (a 2 + p = 0) ∧
  (∀ n : ℕ, n > 0 → a (4 * n - 1) < a (4 * n)) ∧
  (∀ m n : ℕ, m > 0 → n > 0 → a (m + n) ∈ {a m + a n + p, a m + a n + p + 1})

-- Part (1): Sequence {2, -2, 0, 1} cannot be an R_2 sequence
theorem not_R2_seq_first_4_terms : ¬ R_seq (λ n, if n = 1 then 2 else if n = 2 then -2 else if n = 3 then 0 else if n = 4 then 1 else 0) 2 := sorry

-- Part (2): For an R_0 sequence, a_5 = 1
theorem R0_seq_a5_eq_1 (a : ℕ → ℤ) (h : R_seq a 0) : a 5 = 1 := sorry

-- Part (3): There exists p = 2 such that for an R_p sequence, S_n >= S_10
theorem exists_p_Sn_geq_S10 (a : ℕ → ℤ) (p : ℤ) (h : R_seq a p) (S : ℕ → ℤ) 
  (hS : ∀ n, S n = ∑ i in range n, a i)
  : (∃ (p : ℤ), p = 2 ∧ ∀ n : ℕ, n > 0 → S n >= S 10) := sorry

end not_R2_seq_first_4_terms_R0_seq_a5_eq_1_exists_p_Sn_geq_S10_l594_594179


namespace point_quadrant_l594_594240

theorem point_quadrant (θ : ℝ) (h1 : π < θ ∧ θ < 2 * π) :
    (∃ t : ℝ, t = sin θ ∧ -1 < t ∧ t < 0 ∧ sin t < 0 ∧ cos t > 0 ∧
    -1 < sin (sin θ) ∧ sin (sin θ) < 0 ∧ 0 < cos (sin θ) ∧ cos (sin θ) ≤ 1 ∧
    -1 < sin (sin θ) ∧ 0 < cos (sin θ) ∧ cos (sin θ) ≤ 1
    ∧ ∀ x y : ℝ, (x = sin (sin θ) ∧ y = cos (sin θ) →
    ((x > 0 ∧ x < 1 ∧ y > 0 ∧ y < 1) → false ∧ (x < 0 ∧ y > 0) → true))) := sorry

end point_quadrant_l594_594240


namespace speed_of_stream_l594_594457

theorem speed_of_stream
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (downstream_time_eq_upstream_time : downstream_distance / (boat_speed + v) = upstream_distance / (boat_speed - v)) :
  v = 8 :=
by
  let v := 8
  sorry

end speed_of_stream_l594_594457


namespace intersection_points_form_circle_l594_594210

theorem intersection_points_form_circle :
  ∃ c : ℝ, ∀ s : ℝ, let (x, y) := ( 
    (-(7 * s^2 + 4) / (2 - 3 * s^2)), 
    (-(21 * s^3 + 19 * s) / (5 * (2 - 3 * s^2)))
  ) in x^2 + y^2 = c := 
by
  sorry

end intersection_points_form_circle_l594_594210


namespace polynomial_in_integer_ring_l594_594656

-- Define the content of a polynomial
def content (g : polynomial ℚ) : ℤ := g.coeffs.map rat.num.gcd

-- Define a polynomial is primitive if its content is 1
def is_primitive (p : polynomial ℚ) : Prop := content p = 1

-- Prove that if P(P(x)) ∈ ℤ[x] and P(P(P(x))) ∈ ℤ[x], then P(x) ∈ ℤ[x]
theorem polynomial_in_integer_ring
  (P : polynomial ℚ)
  (hPP : polynomial ℤ)
  (hPPP : polynomial ℤ) :
  P ∈ polynomial ℤ :=
begin
  sorry -- proof goes here
end

end polynomial_in_integer_ring_l594_594656


namespace selection_including_both_boys_and_girls_l594_594217

-- Definition: Boys and Girls count
def boys : ℕ := 4
def girls : ℕ := 3
def total_selection : ℕ := 3

-- Theorem statement: Number of ways to select three people such that the group includes both boys and girls
theorem selection_including_both_boys_and_girls : 
  (nat.choose boys 2 * nat.choose girls 1) + (nat.choose girls 2 * nat.choose boys 1) = 30 := 
by
  sorry

end selection_including_both_boys_and_girls_l594_594217


namespace trajectory_of_Q_l594_594316

-- Define the conditions
variables (x y : ℝ)
def point_on_line (P : ℝ × ℝ) : Prop := ∃ x y, (P = (x, y) ∧ 2 * x - y + 3 = 0) -- P is on the line
def midpoint (P M Q : ℝ × ℝ) : Prop := ∃ x y, (M = (-1, 2) ∧ Q = (x, y) ∧ P = (-2 - x, 4 - y)) -- M is midpoint

-- Prove the question == answer given conditions
theorem trajectory_of_Q (P Q : ℝ × ℝ) (hP : point_on_line P) (hM : midpoint P (-1, 2) Q) : 2 * (Q.1) - (Q.2) + 5 = 0 :=
by {
  sorry
}

end trajectory_of_Q_l594_594316


namespace count_three_digit_primes_l594_594266

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ (∀ m, m ∣ n → m = 1 ∨ m = n)

def valid_digits (n : ℕ) : Prop :=
  let digits := [1, 3, 7, 9] in
  ∀ k, k ∈ List.digits n → k ∈ digits

def three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ valid_digits n ∧ List.nodup (List.digits n)

theorem count_three_digit_primes : finset.card 
  (finset.filter is_prime (finset.filter three_digit_number (finset.Ico 100 1000))) = 22 := 
by sorry

end count_three_digit_primes_l594_594266


namespace max_value_of_function_l594_594435

open Real 

theorem max_value_of_function : ∀ x : ℝ, 
  cos (2 * x) + 6 * cos (π / 2 - x) ≤ 5 ∧ 
  ∃ x' : ℝ, cos (2 * x') + 6 * cos (π / 2 - x') = 5 :=
by 
  sorry

end max_value_of_function_l594_594435


namespace find_x_l594_594194

theorem find_x (x : ℝ) : log 8 (3 * x - 1) = 2 → x = 65 / 3 :=
by
  sorry

end find_x_l594_594194


namespace trapezoid_area_correct_l594_594124

-- Define the basic setup for the trapezoid problem
variables (a m n : ℝ)
-- Hypotheses to state the conditions given in the problem
variables (h_pos_a : 0 < a) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_ineq : m + n < a)

-- Definition of area calculation for the given inscribed trapezoid 
def trapezoid_area (a m n : ℝ) : ℝ :=
  a * sqrt (m * n) * (a - n + m) / (a - n)

-- The main statement to prove
theorem trapezoid_area_correct (h_pos_a : 0 < a) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_ineq : m + n < a) : 
  trapezoid_area a m n = a * sqrt (m * n) * (a - n + m) / (a - n) :=
by
  sorry

end trapezoid_area_correct_l594_594124


namespace convex_polygon_diagonal_acute_angles_l594_594314

theorem convex_polygon_diagonal_acute_angles (n : ℕ) (hn : n > 3) : 
  ∃ (v : ℝ × ℝ) (d : ℝ × ℝ), 
    is_convex_polygon n ∧ vertex_of_polygon v ∧ diagonal_of_polygon d v ∧
    acute_angle_with_sides d v :=
sorry

end convex_polygon_diagonal_acute_angles_l594_594314


namespace friends_professions_l594_594296

def Person := {Ivanov, Petrenko, Sidorchuk, Grishin, Altman}
def Profession := {painter, miller, carpenter, postman, barber}

variables (assignment : Person → Profession)

theorem friends_professions :
  (assignment Petrenko ≠ painter ∧ assignment Grishin ≠ painter) ∧
  (assignment Ivanov ≠ miller ∧ assignment Grishin ≠ miller) ∧
  (assignment Petrenko ≠ postman ∧ assignment Altman ≠ postman) ∧
  (assignment Sidorchuk ≠ barber ∧ assignment Petrenko ≠ barber) ∧
  (assignment Ivanov ≠ carpenter ∧ assignment Petrenko ≠ carpenter) ∧
  (assignment Grishin ≠ barber ∧ assignment Altman ≠ barber) ∧ 
  (assignment Altman = painter) ∧
  (assignment Petrenko = miller) ∧
  (assignment Grishin = carpenter) ∧
  (assignment Sidorchuk = postman) ∧
  (assignment Ivanov = barber) := 
sorry

end friends_professions_l594_594296


namespace expand_product_polynomials_l594_594889

noncomputable def poly1 : Polynomial ℤ := 5 * Polynomial.X + 3
noncomputable def poly2 : Polynomial ℤ := 7 * Polynomial.X^2 + 2 * Polynomial.X + 4
noncomputable def expanded_form : Polynomial ℤ := 35 * Polynomial.X^3 + 31 * Polynomial.X^2 + 26 * Polynomial.X + 12

theorem expand_product_polynomials :
  poly1 * poly2 = expanded_form := 
by
  sorry

end expand_product_polynomials_l594_594889


namespace stairs_climbed_together_l594_594322

-- Definitions from conditions
def Jonny_stairs : ℕ := 1269
def Julia_stairs : ℕ := (Jonny_stairs / 3) - 7

-- Proof that their combined climbed stairs is 1685
theorem stairs_climbed_together :
  Jonny_stairs + Julia_stairs = 1685 := by
  have Jonny_stairs_eq : Jonny_stairs = 1269 := rfl
  have Julia_stairs_eq : Julia_stairs = (1269 / 3) - 7 := rfl
  calc
    Jonny_stairs + Julia_stairs
        = 1269 + ((1269 / 3) - 7) : by rw [Jonny_stairs_eq, Julia_stairs_eq]
    ... = 1269 + (423 - 7)         : by norm_num
    ... = 1269 + 416               : by norm_num
    ... = 1685                     : by norm_num

end stairs_climbed_together_l594_594322


namespace diagonal_constant_l594_594822

-- Definitions to establish the setup of the problem
def ellipse (center : ℝ × ℝ) (f1 f2 : ℝ × ℝ) := ℕ -- representing an ellipse by its center and two foci

-- A rectangle circumscribed around an ellipse
def circumscribed_rectangle_diagonal (ellipse_center : ℝ × ℝ) (f1 f2 : ℝ × ℝ) : ℝ :=
  let dist_squared (a b : ℝ × ℝ) : ℝ := ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let l1 := dist_squared ellipse_center f1
  let l2 := dist_squared ellipse_center f2
  let f_dist := dist_squared f1 f2
  (l1 + l2 - f_dist) / 4

-- The final theorem statement
theorem diagonal_constant (ellipse_center : ℝ × ℝ) (f1 f2 : ℝ × ℝ) :
  ∀ (rect : ℕ), (∃ (c1 c2 : ℝ × ℝ), ellipse ellipse_center f1 f2 ∧ 
    circumscribed_rectangle_diagonal ellipse_center f1 f2 = (l1 + l2 - f_dist) / 4) :=
sorry

end diagonal_constant_l594_594822


namespace john_correct_questions_l594_594654

theorem john_correct_questions:
  ∀ (q1 q2: ℕ) (p1 p2: ℝ),
  q1 = 40 → p1 = 0.90 → q2 = 40 → p2 = 0.95 →
  (p1 * q1 + p2 * q2).toNat = 74 := 
by
  intros q1 q2 p1 p2 hq1 hp1 hq2 hp2
  sorry

end john_correct_questions_l594_594654


namespace min_value_function_l594_594033

open Real

theorem min_value_function : 
  ∃ x ∈ ℝ, (5 / 4 - sin x ^ 2 - 3 * cos x) = -7 / 4 :=
by sorry

end min_value_function_l594_594033


namespace boat_speed_determination_l594_594403

theorem boat_speed_determination :
  ∃ x : ℝ, 
    (∀ u d : ℝ, u = 170 / (x + 6) ∧ d = 170 / (x - 6))
    ∧ (u + d = 68)
    ∧ (x = 9) := 
by
  sorry

end boat_speed_determination_l594_594403


namespace menelaus_theorem_collinear_ceva_theorem_concurrent_l594_594677

variables {A B C M N P : Type} [ordered_field A] 
variables {AM BM BN CN CP AP : A}

-- Menelaus' Theorem
theorem menelaus_theorem_collinear:
  (M, N, P are collinear) ↔ 
  (AM / BM) * (BN / CN) * (CP / AP) = 1 := sorry

-- Ceva's Theorem
theorem ceva_theorem_concurrent:
  (lines CM, AN, BP are concurrent or parallel) ↔ 
  (AM / BM) * (BN / CN) * (CP / AP) = -1 := sorry

end menelaus_theorem_collinear_ceva_theorem_concurrent_l594_594677


namespace battery_current_max_min_l594_594792

variable {R : ℝ} {r_b : ℝ} {U_0 : ℝ} {n : ℝ}

def current (p s : ℝ) : ℝ :=
  (p * s * U_0) / (p * R + s * r_b)

def max_current (n : ℝ) (R : ℝ) (r_b : ℝ) (U_0 : ℝ) : ℝ :=
  current 2 (n / 2)

def min_current (n : ℝ) (R : ℝ) (r_b : ℝ) (U_0 : ℝ) : ℝ :=
  current n 1

theorem battery_current_max_min (R : ℝ) (r_b : ℝ) (U_0 : ℝ) :
  R = 36 → r_b = 2 → U_0 = 4.5 → n = 72 →
  max_current n R r_b U_0 = 2.25 ∧ min_current n R r_b U_0 = 0.125 :=
by
  intros
  simp [max_current, min_current, current]
  sorry

end battery_current_max_min_l594_594792


namespace power_sum_l594_594848

theorem power_sum : 2^4 + 2^4 + 2^5 + 2^5 = 96 := 
by
  sorry

end power_sum_l594_594848


namespace bill_miles_difference_l594_594351

theorem bill_miles_difference (B_Sun : ℕ) (B_Sat : ℕ) (J_Sat : ℕ) (J_Sun : ℕ) (h1 : B_Sun = 10) (h2 : J_Sat = 0) (h3 : J_Sun = 2 * B_Sun) (h4 : B_Sat + B_Sun + J_Sun = 36) : B_Sun - B_Sat = 4 :=
by
  subst h1
  subst h2
  subst h3
  subst h4
  simp only [Nat.add_sub_cancel]
  sorry

end bill_miles_difference_l594_594351


namespace arithmetic_sequence_l594_594642

theorem arithmetic_sequence {a b : ℤ} :
  (-1 < a ∧ a < b ∧ b < 8) ∧
  (8 - (-1) = 9) ∧
  (a + b = 7) →
  (a = 2 ∧ b = 5) :=
by
  sorry

end arithmetic_sequence_l594_594642


namespace A_subset_B_range_of_a_l594_594211

-- Definitions
def A (f : ℝ → ℝ) := {x : ℝ | f x = x}
def B (f : ℝ → ℝ) := {x : ℝ | f (f x) = x}

-- Problems
theorem A_subset_B {f : ℝ → ℝ} : A f ⊆ B f := sorry

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, f x = x) :
  - (1 / 4 : ℝ) ≤ a ∧ a ≤ 3 / 4 := sorry

end A_subset_B_range_of_a_l594_594211


namespace quitters_same_tribe_probability_l594_594056

theorem quitters_same_tribe_probability (total_participants : ℕ) (tribe1 : ℕ) (tribe2 : ℕ) (quitters : ℕ)
  (h1 : total_participants = 18) (h2 : tribe1 = 10) (h3 : tribe2 = 8) (h4 : quitters = 3) :
  (↑(nat.choose tribe1 quitters) + ↑(nat.choose tribe2 quitters)) / (↑(nat.choose total_participants quitters)) = 11 / 51 :=
by
  sorry

end quitters_same_tribe_probability_l594_594056


namespace numbers_divisible_by_three_l594_594964

theorem numbers_divisible_by_three (a b : ℕ) (h1 : a = 150) (h2 : b = 450) :
  ∃ n : ℕ, ∀ x : ℕ, (a < x) → (x < b) → (x % 3 = 0) → (x = 153 + 3 * (n - 1)) :=
by
  sorry

end numbers_divisible_by_three_l594_594964


namespace no_sum_of_squares_of_rationals_l594_594360

theorem no_sum_of_squares_of_rationals (p q r s : ℕ) (hq : q ≠ 0) (hs : s ≠ 0)
    (hpq : Nat.gcd p q = 1) (hrs : Nat.gcd r s = 1) :
    (↑p / q : ℚ) ^ 2 + (↑r / s : ℚ) ^ 2 ≠ 168 := by 
    sorry

end no_sum_of_squares_of_rationals_l594_594360


namespace gcd_factorials_l594_594200

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := by
  sorry

end gcd_factorials_l594_594200


namespace ratio_of_wealth_l594_594861

-- Definitions of the percentages as variables
variables (c d e f : ℝ)

-- Assumptions: percentages are in the range 0 < c, d, e, f ≤ 100
axiom (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) (h_f : 0 < f)
axiom (h_c_le_100 : c ≤ 100) (h_d_le_100 : d ≤ 100) (h_e_le_100 : e ≤ 100) (h_f_le_100 : f ≤ 100)

-- Problem Statement: The ratio of the wealth of a citizen of A to the wealth of a citizen of B is de/cf
theorem ratio_of_wealth (P W : ℝ) (hP : P > 0) (hW : W > 0) : 
  (d * e) / (c * f) = ( (d * W / (c * P)) / ((f * W) / (e * P)) ) := 
sorry

end ratio_of_wealth_l594_594861


namespace necessary_but_not_sufficient_for_gt_l594_594222

variable {a b : ℝ}

theorem necessary_but_not_sufficient_for_gt : a > b → a > b - 1 :=
by sorry

end necessary_but_not_sufficient_for_gt_l594_594222


namespace floor_e_is_two_l594_594883

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end floor_e_is_two_l594_594883


namespace find_ab_average_l594_594623

variable (a b c k : ℝ)

-- Conditions
def sum_condition : Prop := (4 + 6 + 8 + 12 + a + b + c) / 7 = 20
def abc_condition : Prop := a + b + c = 3 * ((4 + 6 + 8) / 3)

-- Theorem
theorem find_ab_average 
  (sum_cond : sum_condition a b c) 
  (abc_cond : abc_condition a b c) 
  (c_eq_k : c = k) : 
  (a + b) / 2 = (18 - k) / 2 :=
sorry  -- Proof is omitted


end find_ab_average_l594_594623


namespace proportion_equiv_l594_594116

theorem proportion_equiv (X : ℕ) (h : 8 / 4 = X / 240) : X = 480 :=
by
  sorry

end proportion_equiv_l594_594116


namespace pentagon_segment_condition_l594_594136

-- Define the problem context and hypothesis
variable (a b c d e : ℝ)

theorem pentagon_segment_condition 
  (h₁ : a + b + c + d + e = 3)
  (h₂ : a ≤ b)
  (h₃ : b ≤ c)
  (h₄ : c ≤ d)
  (h₅ : d ≤ e) : 
  a < 3 / 2 ∧ b < 3 / 2 ∧ c < 3 / 2 ∧ d < 3 / 2 ∧ e < 3 / 2 := 
sorry

end pentagon_segment_condition_l594_594136


namespace integer_representation_l594_594696

theorem integer_representation (n : ℤ) : ∃ x y z : ℤ, n = x^2 + y^2 - z^2 :=
by sorry

end integer_representation_l594_594696


namespace probability_not_forming_triangle_l594_594051

theorem probability_not_forming_triangle :
  let segments := [2, 3, 5, 7, 11]
  let total_combinations := (segments.combination 3).length
  let non_triangle_combinations := (segments.combination 3).filter (λ s, 
    ¬ (s[0] + s[1] > s[2] ∧ s[0] + s[2] > s[1] ∧ s[1] + s[2] > s[0])
  ).length
  (non_triangle_combinations : ℚ) / total_combinations = 4 / 5 :=
by
  sorry

end probability_not_forming_triangle_l594_594051


namespace tom_ticket_count_l594_594751

theorem tom_ticket_count : 
  ∃ T : ℕ, T = 28 + 3 * 4 :=
by {
  let T := 28 + 3 * 4,
  use T,
  refl
}

end tom_ticket_count_l594_594751


namespace T_size_l594_594862

-- Define the set T and its conditions
def T (n : ℕ) : Prop := 
  n > 1 ∧ ∃ e : ℕ → ℕ, (∀ i, e i = e (i + 18)) ∧ (n.digitsBase 10 = λ k, e k)

-- The number we are dealing with
def number := 10^18 - 1

-- Assuming 10^18 - 1 has 360 divisors
axiom known_prime_factors : (360 : ℕ) = ∏ (p : ℕ) in (factors number), p

-- Prove the desired number of integers in T
theorem T_size : (∑ n in (divisors number) ∖ {1}, 1) = 359 := sorry

end T_size_l594_594862


namespace transformed_polynomial_l594_594971

theorem transformed_polynomial (x y : ℝ) (h : y = x + 1 / x) :
  (x^4 - 2*x^3 - 3*x^2 + 2*x + 1 = 0) → (x^2 * (y^2 - y - 3) = 0) :=
by
  sorry

end transformed_polynomial_l594_594971


namespace find_value_of_fraction_of_x_six_l594_594173

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := (Real.log x) / (Real.log b)

theorem find_value_of_fraction_of_x_six (x : ℝ) (h : log_base (10 * x) 10 + log_base (100 * x ^ 2) 10 = -1) : 
    1 / x ^ 6 = 31622.7766 :=
by
  sorry

end find_value_of_fraction_of_x_six_l594_594173


namespace sin_alpha_of_terminal_side_l594_594982

theorem sin_alpha_of_terminal_side (α : ℝ) (h : ∃ (θ : ℝ), θ = 30 * (2 * π / 360) ∧ (sin θ, -cos θ) = (sin 30 * (2 * π / 360), -cos 30 * (2 * π / 360))) : 
  sin α = -√3/2 := 
sorry

end sin_alpha_of_terminal_side_l594_594982


namespace area_enclosed_by_curves_l594_594864

-- Define the functions for the curves
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := sqrt x

-- Prove that the area enclosed by the curves y = x^2 and y = sqrt(x) from x = 0 to x = 1 is 1/3
theorem area_enclosed_by_curves : (∫ x in 0..1, g x - f x) = 1 / 3 :=
by
  -- Here should be the proof, but it's omitted
  sorry

end area_enclosed_by_curves_l594_594864


namespace group_size_correct_l594_594218

noncomputable def group_size : ℕ :=
  let x := 5 in
  let y := 144 in
  let f := λ n, factorial (n - 1) in
  nat.find (λ n, f n = y + (factorial (x - 1)))

theorem group_size_correct : group_size = 7 := by
  sorry

end group_size_correct_l594_594218


namespace correct_formula_for_given_table_l594_594256

theorem correct_formula_for_given_table :
  ∀ x y, (x, y) ∈ [(0, 200), (1, 160), (2, 80), (3, 0), (4, -120)] → y = 200 - 40 * x - 10 * x^2 :=
by
  intros x y h
  cases h with
  | or.inl hl => sorry
  | or.inr hr => sorry

end correct_formula_for_given_table_l594_594256


namespace train_speed_incl_stoppages_l594_594526

theorem train_speed_incl_stoppages
  (speed_excl_stoppages : ℝ)
  (stoppage_time_minutes : ℝ)
  (h1 : speed_excl_stoppages = 42)
  (h2 : stoppage_time_minutes = 21.428571428571423)
  : ∃ speed_incl_stoppages, speed_incl_stoppages = 27 := 
sorry

end train_speed_incl_stoppages_l594_594526


namespace number_of_liars_is_1001_l594_594352

-- Define the conditions of the problem
def num_inhabitants : ℕ := 2001
def liars_and_knights := Σ n : ℕ, Σ k : ℕ, n + k = num_inhabitants

-- Given the conditions and the correct answer, articulate the proof problem
theorem number_of_liars_is_1001 (n k : ℕ) (h : n + k = num_inhabitants)
  (hi : ∀ (i : ℕ), i < num_inhabitants → (if i < n then "Among the remaining inhabitants, more than half are liars" else true))
  : n = 1001 := 
sorry

end number_of_liars_is_1001_l594_594352


namespace total_selections_l594_594292

theorem total_selections (males females : ℕ)
  (h_males : males = 5) (h_females : females = 4) :
  ∑ k in finset.Icc 2 4, (nat.choose females k) * (nat.choose males (4 - k)) = 81 :=
by
  rw [h_males, h_females]
  sorry

end total_selections_l594_594292


namespace find_ellipse_equation_l594_594720

theorem find_ellipse_equation (a b c : ℝ) (ε : ℝ) :
    (a > 0 ∧ b > 0 ∧ c > 0 ∧ a > b) →
    (ε = √3 / 2) →
    (c / a = ε) →
    (a - c = 2 - √3) →
    b^2 = a^2 - c^2 →
    (∀ x y : ℝ, y^2 / a^2 + x^2 / b^2 = 1 ↔ y^2 / 4 + x^2 = 1) :=
by
  intros hpos hecc hca hac hbsq
  sorry

end find_ellipse_equation_l594_594720


namespace area_triangle_BCD_l594_594617

-- Definition of the quadrilateral and its properties
variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variable (convex : ∀ P Q R S, ∃ (A : Prop), Prop) -- Define the convexity condition as a placeholder

-- Side lengths
variable (AB BC CD AD : ℝ)
variable (hAB : AB = 6)
variable (hBC : BC = 9)
variable (hCD : CD = 4)
variable (hAD : AD = 12)

-- Inradius of triangle ABC
variable (inradius_ABC : ℝ)
variable (hinradius : inradius_ABC = 3)

-- Right angle condition at BCD
variable (angle_BCD : Prop)
variable (hangle_BCD : angle_BCD = true)

-- Proof statement for the area of triangle BCD
theorem area_triangle_BCD : 
  1/2 * BC * CD = 18 := by
sorry

end area_triangle_BCD_l594_594617


namespace sum_of_reciprocals_of_roots_l594_594343

theorem sum_of_reciprocals_of_roots :
  let p q : ℚ
  let α β : ℚ
  assume h1 : 6 * p ^ 2 + 5 * p + 7 = 0,
  assume h2 : 6 * q ^ 2 + 5 * q + 7 = 0,
  assume hα : α = 1 / p,
  assume hβ : β = 1 / q,
  α + β = -5 / 7 :=
by sorry

end sum_of_reciprocals_of_roots_l594_594343


namespace lambda_max_inequality_l594_594552

theorem lambda_max_inequality (n : ℕ) (hn : 2 ≤ n) (z : ℕ → ℂ) (h_sum_zero : ∑ k in finset.range n, z k = 0) (h_cycle : z n = z 0) :
  (12 * n / (n^2 - 1)) * (finset.range n).sup (λ k, ∥z k∥^2) ≤ ∑ k in finset.range n, ∥z (k + 1) - z k∥^2 :=
sorry

end lambda_max_inequality_l594_594552


namespace triangular_pyramid_circumscribed_sphere_surface_area_l594_594832

noncomputable def surface_area_of_circumscribed_sphere (a : ℝ) : ℝ :=
  4 * Real.pi * ((Real.sqrt 6 / 4 * a)^2)

theorem triangular_pyramid_circumscribed_sphere_surface_area (a : ℝ) :
  (∃ R, R = Real.sqrt 6 / 4 * a) →
  surface_area_of_circumscribed_sphere a = 3 / 2 * Real.pi * a^2 :=
by
  intro hR
  cases hR with R hR_eq
  rw [surface_area_of_circumscribed_sphere, hR_eq]
  sorry

end triangular_pyramid_circumscribed_sphere_surface_area_l594_594832


namespace probability_after_6_passes_l594_594418

noncomputable section

-- We define people
inductive Person
| A | B | C

-- Probability that person A has the ball after n passes
def P : ℕ → Person → ℚ
| 0, Person.A => 1
| 0, _ => 0
| n+1, Person.A => (P n Person.B + P n Person.C) / 2
| n+1, Person.B => (P n Person.A + P n Person.C) / 2
| n+1, Person.C => (P n Person.A + P n Person.B) / 2

theorem probability_after_6_passes :
  P 6 Person.A = 11 / 32 := by
  sorry

end probability_after_6_passes_l594_594418


namespace triangle_median_pq_len_l594_594608

-- Definitions for the provided conditions
theorem triangle_median_pq_len
  (P Q R M N : Type)
  [metric_space P] [metric_space Q] [metric_space R] [metric_space M] [metric_space N]
  (PQ QR PR : ℝ) 
  (hQR : QR = 8)
  (hPR : PR = 5)
  (median_from_P_perpendicular_to_median_from_Q : ∀ (M N : Type) [metric_space M] [metric_space N],
    M.is_median_for (P, QR) ∧ N.is_median_for (Q, PR) → perpendicular M N)
  -- Conclude that the length of PQ is √(281/15)
  : PQ = real.sqrt (281 / 15) := by 
  -- Realize that we need the proof here, so use sorry
  sorry

end triangle_median_pq_len_l594_594608


namespace inequality_solution_l594_594055

open Set

theorem inequality_solution (x : ℝ) : (x ≠ -2) → (abs (x + 1) / abs (x + 2) ≥ 1) ↔ (x ∈ Iio (-2) ∪ Ioc (-2, -3 / 2)) :=
by
  sorry

end inequality_solution_l594_594055


namespace coefficient_x2_expansion_eq_5_l594_594941

theorem coefficient_x2_expansion_eq_5 (a : ℝ) (h : (binom 5 2) + a * (binom 5 1) = 5) : a = -1 :=
by
  sorry

end coefficient_x2_expansion_eq_5_l594_594941


namespace small_cone_altitude_correct_l594_594482

-- Define conditions for the problem
def area_lower_base : ℝ := 324 * Real.pi
def area_upper_base : ℝ := 36 * Real.pi
def frustum_altitude : ℝ := 36

-- Define the resulting altitude of the small cone that was cut off
def small_cone_altitude : ℝ := 18

-- State the theorem to be proved
theorem small_cone_altitude_correct :
  (∃ r_u r_l H h, area_upper_base = Real.pi * r_u^2 ∧
                  area_lower_base = Real.pi * r_l^2 ∧
                  r_u = r_l / 3 ∧
                  h = H / 3 ∧
                  frustum_altitude = (2 / 3) * H ∧
                  h = small_cone_altitude) :=
by
  sorry

end small_cone_altitude_correct_l594_594482


namespace license_plate_combinations_l594_594508

theorem license_plate_combinations :
  let letters := 26
  let choose (n k : ℕ) := nat.choose n k
  let factorial := nat.factorial
  let digits := 10
  26 * choose 25 3 * choose 5 2 * factorial 3 * 10 * 9 * 8 = 44,400,000 :=
by
  let letters := 26
  let choose (n k : ℕ) := nat.choose
  let factorial := nat.factorial
  let digits := 10
  calc
    26 * choose 25 3 * choose 5 2 * factorial 3 * 10 * 9 * 8 = 26 * 2300 * 10 * 6 * 720 : by
      -- Expand and verify intermediate steps if needed
      sorry
    ... = 26 * 25440000 : by
      -- Intermediate calculation
      sorry
    ... = 44400000 : by
      -- Final multiplication
      sorry

end license_plate_combinations_l594_594508


namespace series_sum_solution_l594_594669

noncomputable def compute_series_sum (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hgt : a > 2 * b) : ℝ :=
∑' n, 1 / ([(n - 1) * a - 2 * (n - 2) * b] * [n * a - 2 * (n - 1) * b])

theorem series_sum_solution (a b : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hgt : a > 2 * b) :
  compute_series_sum a b hpos_a hpos_b hgt = 1 / ((a - 2 * b) * b) :=
sorry

end series_sum_solution_l594_594669


namespace shuai_fen_ratio_l594_594069

theorem shuai_fen_ratio 
  (C : ℕ) (B_and_D : ℕ) (a : ℕ) (x : ℚ) 
  (hC : C = 36) (hB_and_D : B_and_D = 75) :
  (x = 0.25) ∧ (a = 175) := 
by {
  -- This is where the proof steps would go
  sorry
}

end shuai_fen_ratio_l594_594069


namespace cafeteria_apples_end_of_friday_l594_594061

variable (apples_initial : ℕ)
variable (used_monday : ℕ)
variable (used_tuesday : ℕ)
variable (used_wednesday : ℕ)
variable (used_thursday : ℕ)
variable (used_friday : ℕ)
variable (purchased_rate : ℕ)

theorem cafeteria_apples_end_of_friday :
  apples_initial = 17 →
  used_monday = 2 →
  used_tuesday = 2 * used_monday →
  used_wednesday = 2 * used_tuesday →
  used_thursday = 2 * used_wednesday →
  used_friday = 2 * used_thursday →
  purchased_rate = 2 →
  let remaining_monday := apples_initial - used_monday + purchased_rate * used_monday in
  let remaining_tuesday := remaining_monday - used_tuesday + purchased_rate * used_tuesday in
  let remaining_wednesday := remaining_tuesday - used_wednesday + purchased_rate * used_wednesday in
  let remaining_thursday := remaining_wednesday - used_thursday + purchased_rate * used_thursday in
  let remaining_friday := remaining_thursday - used_friday + purchased_rate * used_friday in
  remaining_friday = 79 :=
by
  intros
  sorry

end cafeteria_apples_end_of_friday_l594_594061


namespace prob_XYZ_wins_l594_594620

-- Define probabilities as given in the conditions
def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_Z : ℚ := 1 / 12

-- Define the probability that one of X, Y, or Z wins, assuming events are mutually exclusive
def P_XYZ_wins : ℚ := P_X + P_Y + P_Z

theorem prob_XYZ_wins : P_XYZ_wins = 11 / 24 := by
  -- sorry is used to skip the proof
  sorry

end prob_XYZ_wins_l594_594620


namespace sum_and_divide_repeating_decimals_l594_594850

noncomputable def repeating_decimal_83 : ℚ := 83 / 99
noncomputable def repeating_decimal_18 : ℚ := 18 / 99

theorem sum_and_divide_repeating_decimals :
  (repeating_decimal_83 + repeating_decimal_18) / (1 / 5) = 505 / 99 :=
by
  sorry

end sum_and_divide_repeating_decimals_l594_594850


namespace problem_statement_l594_594666
  
noncomputable def isosceles_triangle_area_prob : ℝ :=
  let DEF := {(x, y) : ℝ × ℝ | 0 ≤ x ∧ 0 ≤ y ∧ x + y ≤ 1} in
  let region := {(x, y) : ℝ × ℝ | x ∈ Icc 0 1 ∧ y ∈ Icc 0 1 ∧ y > x ∧ y > (x + y) / 2} in
  (set.finite.to_finset DEF).card⁻¹ * (set.finite.to_finset region).card

theorem problem_statement :
  isosceles_triangle_area_prob = 1 / 4 :=
sorry

end problem_statement_l594_594666


namespace correct_option_l594_594101

theorem correct_option :
  (3 * a^2 - a^2 = 2 * a^2) ∧
  (¬ (a^2 * a^3 = a^6)) ∧
  (¬ ((3 * a)^2 = 6 * a^2)) ∧
  (¬ (a^6 / a^3 = a^2)) :=
by
  -- We only need to state the theorem; the proof details are omitted per the instructions.
  sorry

end correct_option_l594_594101


namespace propositions_correct_l594_594931

variables {V : Type*} [inner_product_space ℝ V] (a b : V) (hb : b ≠ 0)

theorem propositions_correct (a b : V) (hb : b ≠ 0) :
  ((∥a + b∥ = ∥a - b∥) → (inner a b = 0)) ∧ -- Proposition A
  ((inner b (1:ℝ) = inner b (1:ℝ)) → b = b) ∧ -- Proposition B (shown to be false, so omitted)
  (∃ (c : V) (hc: c ≠ 0), (proj a c = (inner a c / ∥a∥^2) • a)) ∧ -- Proposition C
  (∃ (λ : ℝ), a = λ • b) -- Proposition D
:= by sorry

end propositions_correct_l594_594931


namespace slope_of_line_l594_594436

theorem slope_of_line (x y : ℝ) (h : 2 * y = -3 * x + 6) : (∃ m b : ℝ, y = m * x + b) ∧  (m = -3 / 2) :=
by 
  sorry

end slope_of_line_l594_594436


namespace count_n_leq_1000_l594_594539

theorem count_n_leq_1000 :
  let count (P : ℕ → Prop) := (Finset.range 1001).filter P |>.card
  ∃ (x : ℝ), ∀ n, (n ≤ 1000 ∧ ∃ x, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n) →
    count (λ n, ∃ x, ⌊x⌋ + ⌊2 * x⌋ + ⌊4 * x⌋ = n) = 568 :=
sorry

end count_n_leq_1000_l594_594539


namespace fraction_of_even_integers_divisible_by_4_l594_594866

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem fraction_of_even_integers_divisible_by_4 :
  let candidates := (multiset.range (151 - 50)).map (λ x, x + 50)
  let evens := candidates.filter (λ n, n % 2 = 0 ∧ sum_of_digits n = 12)
  let divisible_by_4 := evens.filter (λ n, n % 4 = 0)
  (divisible_by_4.card : ℚ) / (evens.card : ℚ) = 2 / 5 :=
by
  sorry

end fraction_of_even_integers_divisible_by_4_l594_594866


namespace copies_made_in_half_hour_l594_594772

theorem copies_made_in_half_hour :
  let copies_per_minute_machine1 := 40
  let copies_per_minute_machine2 := 55
  let time_minutes := 30
  (copies_per_minute_machine1 * time_minutes) + (copies_per_minute_machine2 * time_minutes) = 2850 := by
    sorry

end copies_made_in_half_hour_l594_594772


namespace odd_sum_pairs_count_l594_594219

open Finset

-- Define the set of numbers from 1 to 10
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define what it means for a sum to be odd
def is_odd_sum (a b : ℕ) : Prop := (a + b) % 2 = 1

-- Define the set of pairs whose sum is odd
def odd_sum_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  S ×ˢ S -- Cartesian product of S with itself
  |> filter (λ pair, is_odd_sum pair.1 pair.2)

-- The theorem we need to prove: There are 25 such pairs
theorem odd_sum_pairs_count : (odd_sum_pairs S).card = 25 :=
by sorry

end odd_sum_pairs_count_l594_594219


namespace find_f_neg_log3_5_l594_594567

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  if x >= 0 then 3^x + m else -f (-x)

theorem find_f_neg_log3_5 (m : ℝ) (h_odd : ∀ x : ℝ, f x m = -f (-x) m)
  (hx_geq_0 : ∀ x : ℝ, x ≥ 0 → f x m = 3^x + m) :
  ∃ m : ℝ, f (-Real.logb 5 3) m = -4 :=
sorry

end find_f_neg_log3_5_l594_594567


namespace ratio_minutes_0_2_l594_594760

/-- 
This statement proves that the number of minutes that corresponds to a ratio of 0.2 to 1 hour is 12 minutes.
-/
theorem ratio_minutes_0_2 (m : ℕ) (h : m / 60 = 1 / 5) : m = 12 :=
by 
  let h60 : ℕ := 60
  let h05 : ℚ := 1 / 5
  have h_mul : 5 * m = 60 := by 
    rw [← h05, ← ne_of_gt (by norm_num : (0 : ℚ) < 5)]
    ring_nf
  sorry

end ratio_minutes_0_2_l594_594760


namespace floor_e_eq_two_l594_594885

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 := 
sorry

end floor_e_eq_two_l594_594885


namespace isosceles_right_triangle_circles_ratio_l594_594800

theorem isosceles_right_triangle_circles_ratio
    (x : ℝ) : ∃ (R r : ℝ), R = x * (Real.sqrt 2) / 2 ∧ r = x * (Real.sqrt 2 - 1) / 2 ∧ R / r = 1 + Real.sqrt 2 := 
begin
    sorry
end

end isosceles_right_triangle_circles_ratio_l594_594800


namespace supplementary_angles_difference_l594_594388
-- Import necessary libraries

-- Define the conditions
def are_supplementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 180

def ratio_7_2 (θ₁ θ₂ : ℝ) : Prop := θ₁ / θ₂ = 7 / 2

-- State the theorem
theorem supplementary_angles_difference (θ₁ θ₂ : ℝ) 
  (h_supp : are_supplementary θ₁ θ₂) 
  (h_ratio : ratio_7_2 θ₁ θ₂) :
  |θ₁ - θ₂| = 100 :=
by
  sorry

end supplementary_angles_difference_l594_594388


namespace train_speed_l594_594141

theorem train_speed (train_length bridge_length : ℕ) (crossing_time : ℕ) (h1 : train_length = 120) (h2 : bridge_length = 255) (h3 : crossing_time = 30) : 
    let total_distance := train_length + bridge_length in
    let speed_mps := total_distance / crossing_time in
    let speed_kmph := speed_mps * 36 / 10 in
    speed_kmph = 45 := by
    -- Definitions and proof steps will go here
    sorry

end train_speed_l594_594141


namespace only_correct_statements_l594_594573

variables (a b : Vec₁)

-- Collinear vectors in the same direction have their dot product equal to the product of their magnitudes.
axiom collinear_in_same_direction (h : collinear_same_direction a b) : 
  (a • b) = ∥a∥ * ∥b∥

-- Unit vectors have magnitudes of 1, hence their squares are equal.
axiom unit_vectors_square (h : unit_vector a ∧ unit_vector b) : 
  (a^2) = (b^2)

theorem only_correct_statements :
  (∃ h : collinear_same_direction a b, (a • b) = ∥a∥ * ∥b∥) ∧
  (∃ h : unit_vector a ∧ unit_vector b, (a^2) = (b^2)) :=
by {
  sorry
}

end only_correct_statements_l594_594573


namespace correct_multiplicand_l594_594493

theorem correct_multiplicand (x : ℕ) (h1 : x * 467 = 1925817) : 
  ∃ n : ℕ, n * 467 = 1325813 :=
by
  sorry

end correct_multiplicand_l594_594493


namespace no_eight_roots_for_nested_quadratics_l594_594241

theorem no_eight_roots_for_nested_quadratics
  (f g h : ℝ → ℝ)
  (hf : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e k : ℝ, ∀ x, g x = d * x^2 + e * x + k)
  (hh : ∃ p q r : ℝ, ∀ x, h x = p * x^2 + q * x + r)
  (hroots : ∀ x, f (g (h x)) = 0 → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8)) :
  false :=
by
  sorry

end no_eight_roots_for_nested_quadratics_l594_594241


namespace angle_between_vectors_is_3pi_over_4_l594_594584

def vector_a : ℝ × ℝ := (1, 1)

def vector_b : ℝ × ℝ := (0, -2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

noncomputable def cos_angle_between_vectors (v1 v2 : ℝ × ℝ) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

theorem angle_between_vectors_is_3pi_over_4 :
  cos_angle_between_vectors vector_a vector_b = -Real.sqrt 2 / 2 → 
  Real.arccos (cos_angle_between_vectors vector_a vector_b) = 3 * Real.pi / 4 :=
by 
  sorry

end angle_between_vectors_is_3pi_over_4_l594_594584


namespace spending_limit_l594_594353

variable (n b total_spent limit: ℕ)

theorem spending_limit (hne: n = 34) (hbe: b = n + 5) (hts: total_spent = n + b) (hlo: total_spent = limit + 3) : limit = 70 := by
  sorry

end spending_limit_l594_594353


namespace central_angle_of_chord_l594_594757

theorem central_angle_of_chord {O A B : Point} (hO : center O) (hAB : chord O A B)
  (hAngle : angle O B A = 40) : angle A O B = 100 := 
sorry

end central_angle_of_chord_l594_594757


namespace books_not_adjacent_probability_l594_594414

-- Given conditions
def num_books : ℕ := 5
def num_chinese_books : ℕ := 2
def num_math_books : ℕ := 2
def num_physics_books : ℕ := 1

-- Problem statement
-- We are calculating the probability that books of the same subject are 
-- not adjacent to each other.
theorem books_not_adjacent_probability : 
  let total_arrangements := 120 in
  let valid_arrangements := 48 in
  valid_arrangements / total_arrangements = (2 / 5) :=
by
  sorry

end books_not_adjacent_probability_l594_594414


namespace contradiction_proof_l594_594442

theorem contradiction_proof (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end contradiction_proof_l594_594442


namespace totalSandwiches_l594_594467

def numberOfPeople : ℝ := 219.0
def sandwichesPerPerson : ℝ := 3.0

theorem totalSandwiches : numberOfPeople * sandwichesPerPerson = 657.0 := by
  -- Proof goes here
  sorry

end totalSandwiches_l594_594467


namespace area_larger_sphere_l594_594826

variables {r1 r2 r : ℝ}
variables {A1 A2 : ℝ}

-- Declare constants for the problem
def radius_smaller_sphere : ℝ := 4 -- r1
def radius_larger_sphere : ℝ := 6  -- r2
def radius_ball : ℝ := 1           -- r
def area_smaller_sphere : ℝ := 27  -- A1

-- Given conditions
axiom radius_smaller_sphere_condition : r1 = radius_smaller_sphere
axiom radius_larger_sphere_condition : r2 = radius_larger_sphere
axiom radius_ball_condition : r = radius_ball
axiom area_smaller_sphere_condition : A1 = area_smaller_sphere

-- Statement to be proved
theorem area_larger_sphere :
  r1 = radius_smaller_sphere → r2 = radius_larger_sphere → r = radius_ball → A1 = area_smaller_sphere → A2 = 60.75 :=
by
  intros
  sorry

end area_larger_sphere_l594_594826


namespace singing_competition_probability_l594_594122

theorem singing_competition_probability :
  let songs := ["Difficult to Avoid", "Orchid Pavilion Preface", "Make a Wish", "The First Dream"] in
  let A_choices := [["Difficult to Avoid", "Orchid Pavilion Preface"], 
                    ["Difficult to Avoid", "Make a Wish"], 
                    ["Difficult to Avoid", "The First Dream"]] in
  let B_choices := [["Difficult to Avoid", "Orchid Pavilion Preface"], 
                    ["Difficult to Avoid", "The First Dream"], 
                    ["Orchid Pavilion Preface", "The First Dream"]] in
  let total_scenarios := 9 in
  let matching_scenarios := 8 in
  (matching_scenarios / total_scenarios : ℚ) = 8 / 9 := 
by
  sorry

end singing_competition_probability_l594_594122


namespace maria_towels_l594_594107

theorem maria_towels (green_towels white_towels given_towels : ℕ) (h1 : green_towels = 35) (h2 : white_towels = 21) (h3 : given_towels = 34) :
  green_towels + white_towels - given_towels = 22 :=
by
  sorry

end maria_towels_l594_594107


namespace work_completion_time_l594_594474

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 6
def work_rate_B : ℚ := 1 / 6
def work_rate_C : ℚ := 1 / 6

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Define the total work to be done (1 represents the whole job)
def total_work : ℚ := 1

-- Calculate the number of days to complete the work together
def days_to_complete_work : ℚ := total_work / combined_work_rate

theorem work_completion_time :
  work_rate_A = 1 / 6 ∧
  work_rate_B = 1 / 6 ∧
  work_rate_C = 1 / 6 →
  combined_work_rate = (work_rate_A + work_rate_B + work_rate_C) →
  days_to_complete_work = 2 :=
by
  intros
  sorry

end work_completion_time_l594_594474


namespace maximal_lines_l594_594003

-- Definitions
def lines (N : ℕ) := ℕ -- A placeholder definition for lines

-- Conditions
-- Given N lines such that:
-- 1. Any two lines intersect
-- 2. Among any 15 lines, there are necessarily two whose angle between them is 60 degrees

-- Theorem statement
theorem maximal_lines (N : ℕ) (h1 : ∀ (l₁ l₂ : lines N), l₁ ≠ l₂ → exists (p : ℝ × ℝ), True)
                     (h2 : ∀ (L : finset (lines N)), L.card = 15 → ∃ (l₁ l₂ ∈ L), angle_between l₁ l₂ = 60) :
  N ≤ 42 :=
  sorry

-- Auxiliary Definitions
def angle_between (l₁ l₂ : lines ℕ) := 60 -- Placeholder for actual definition

end maximal_lines_l594_594003


namespace positive_integer_x_l594_594115

theorem positive_integer_x (x : ℕ) (hx : 15 * x = x^2 + 56) : x = 8 := by
  sorry

end positive_integer_x_l594_594115


namespace max_elements_in_A_l594_594934

open Set

theorem max_elements_in_A {A B C : Set ℕ}
  (h1 : A ⊆ B) (h2 : A ⊆ C) (hB : B = {0, 1, 2, 3, 4}) (hC : C = {0, 2, 4, 8}) :
  cardinal A ≤ 2 := 
sorry

end max_elements_in_A_l594_594934


namespace domain_of_f_l594_594604

open Set

noncomputable def f (x : ℝ) : ℝ := (sqrt (4 - x^2)) / x

theorem domain_of_f :
  {x : ℝ | 4 - x^2 ≥ 0 ∧ x ≠ 0} = {x : ℝ | (-2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end domain_of_f_l594_594604


namespace integral_evaluation_correct_l594_594147

def eval_integral_option_C : Prop :=
  ∫ x in 0..1, (1:ℝ) = 1

theorem integral_evaluation_correct : eval_integral_option_C :=
by
  sorry

end integral_evaluation_correct_l594_594147


namespace total_floor_area_covered_l594_594082

-- Definitions for the given problem
def combined_area : ℕ := 204
def overlap_two_layers : ℕ := 24
def overlap_three_layers : ℕ := 20
def total_floor_area : ℕ := 140

-- Theorem to prove the total floor area covered by the rugs
theorem total_floor_area_covered :
  combined_area - overlap_two_layers - 2 * overlap_three_layers = total_floor_area := by
  sorry

end total_floor_area_covered_l594_594082


namespace sqrt_condition_l594_594606

theorem sqrt_condition (x : ℝ) : (3 * x - 5 ≥ 0) → (x ≥ 5 / 3) :=
by
  intros h
  have h1 : 3 * x ≥ 5 := by linarith
  have h2 : x ≥ 5 / 3 := by linarith
  exact h2

end sqrt_condition_l594_594606


namespace correct_least_squares_method_def_l594_594445

def least_squares_method_def (s : string) := 
  (s = "The method of taking the sum of the squares of each deviation as the total deviation and minimizing it")

theorem correct_least_squares_method_def :
  least_squares_method_def "The method of taking the sum of the squares of each deviation as the total deviation and minimizing it" = true :=
sorry

end correct_least_squares_method_def_l594_594445


namespace find_radius_of_sphere_l594_594743

def radius_of_sphere_equal_to_cylinder_area (r : ℝ) (h : ℝ) (d : ℝ) : Prop :=
  (4 * Real.pi * r^2 = 2 * Real.pi * ((d / 2) * h))

theorem find_radius_of_sphere : ∃ r : ℝ, radius_of_sphere_equal_to_cylinder_area r 6 6 ∧ r = 3 :=
by
  sorry

end find_radius_of_sphere_l594_594743


namespace find_n_such_that_a_n_is_1995_l594_594208

def smallest_prime_not_dividing (n : ℕ) : ℕ :=
  if n = 1 then 2 else if n % 2 ≠ 0 then 2
    else if n % 3 ≠ 0 then 3
    else if n % 5 ≠ 0 then 5
    else if n % 7 ≠ 0 then 7
    else if n % 11 ≠ 0 then 11
    else if n % 13 ≠ 0 then 13
    else if n % 17 ≠ 0 then 17
    else if n % 19 ≠ 0 then 19
    else sorry -- Continue this logic for all required primes.

def product_of_all_primes_less_than (p : ℕ) : ℕ :=
  if p = 2 then 1 else if p = 3 then 2 else if p = 5 then 6
  else if p = 7 then 30
  else sorry -- Continue this logic accurately for primes as needed.

def a : ℕ → ℕ
| 0       := 1
| (n + 1) := let an := a n in
             an * smallest_prime_not_dividing an / product_of_all_primes_less_than (smallest_prime_not_dividing an)

theorem find_n_such_that_a_n_is_1995 : ∃ n, a n = 1995 ∧ n = 142 :=
by
  existsi 142
  split
  sorry -- This part proves that a 142 = 1995
  refl -- This concludes the definition of n as 142

end find_n_such_that_a_n_is_1995_l594_594208


namespace line_intersects_ellipse_l594_594246
open Real

variables {x y : ℝ}
noncomputable def ellipse := {p : ℝ × ℝ | (p.2^2 / 9) + (p.1^2) = 1}
def midpoint (a b : ℝ × ℝ) := (a.1 + b.1) / 2 = 1/2 ∧ (a.2 + b.2) / 2 = 1/2
def lineAB (a b : ℝ × ℝ) := 9 * (a.1 + b.1) + (a.2 + b.2) - 5 = 0

theorem line_intersects_ellipse (a b : ℝ × ℝ) :
  a ∈ ellipse ∧ b ∈ ellipse ∧ midpoint a b → lineAB a b :=
by sorry

end line_intersects_ellipse_l594_594246


namespace trigonometric_comparison_l594_594671

theorem trigonometric_comparison:
  let a := Real.sin (13 * Real.pi / 5)
  let b := Real.cos (- 2 * Real.pi / 5)
  let c := Real.tan (7 * Real.pi / 5) in
  b < a ∧ a < c :=
by
  have h1: a = Real.sin (2 * Real.pi / 5), from sorry,
  have h2: b = Real.cos (2 * Real.pi / 5), from sorry,
  have h3: c = Real.tan (2 * Real.pi / 5), from sorry,
  have h4: (2 * Real.pi / 5) ∈ Set.Ioc (Real.pi / 4) (Real.pi / 2), from sorry,
  have comparison: Real.cos (2 * Real.pi / 5) < Real.sin (2 * Real.pi / 5) ∧ Real.sin (2 * Real.pi / 5) < Real.tan (2 * Real.pi / 5), from sorry,
  exact ⟨comparison.left, comparison.right⟩

end trigonometric_comparison_l594_594671


namespace count_odd_tens_digit_squares_is_20_l594_594636

def tens_digit (n : ℕ) : ℕ :=
(n / 10) % 10

def is_odd (n : ℕ) : Prop :=
n % 2 = 1

def count_odd_tens_digit_squares : ℕ :=
(finset.range 101).filter (λ n, is_odd (tens_digit (n^2))).card

theorem count_odd_tens_digit_squares_is_20 : count_odd_tens_digit_squares = 20 := 
sorry

end count_odd_tens_digit_squares_is_20_l594_594636


namespace min_alterations_for_unique_sums_l594_594513

theorem min_alterations_for_unique_sums :
  let initial_matrix : Matrix (Fin 3) (Fin 3) ℕ := ![ ![5, 10, 3], ![7, 2, 8], ![6, 3, 9] ],
  ∃ (altered_matrix : Matrix (Fin 3) (Fin 3) ℕ), 
    (∃ (a b c : Fin 3), 
      altered_matrix a 0 ≠ initial_matrix a 0 ∨ 
      altered_matrix b 1 ≠ initial_matrix b 1 ∨
      altered_matrix c 2 ≠ initial_matrix c 2) ∧ 
    (altered_matrix.rowSums.toFinVec ≠ initial_matrix.rowSums.toFinVec ∧ 
    altered_matrix.colSums.toFinVec ≠ initial_matrix.colSums.toFinVec) ∧ 
    (∀ i j : Fin 3, ∀ k l : Fin 3, 
      (i ≠ k ∨ j ≠ l) → 
      (altered_matrix.rowSums i ≠ altered_matrix.rowSums k ∧ 
      altered_matrix.colSums j ≠ altered_matrix.colSums l)) ∧ 
      3 = 3 :=
by 
  sorry

end min_alterations_for_unique_sums_l594_594513


namespace find_x_l594_594932

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 3)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

-- Define the parallel condition between (b - a) and b
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u  = (k * v.1, k * v.2)

-- The problem statement in Lean 4
theorem find_x (x : ℝ) (h : parallel (b x - a) (b x)) : x = 2 := 
  sorry

end find_x_l594_594932


namespace problem_solution_l594_594297

-- Parametric equations of line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (⟨(1 / √2) * t, -1 + (1 / √2) * t⟩)

-- Standard equation of line l
def line_l_standard (x y : ℝ) : Prop :=
  x - y - 1 = 0

-- Polar equation of curve C
def curve_C_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 6 * ρ * Real.cos θ + 1 = 0

-- Cartesian equation of curve C
def curve_C_cartesian (x y : ℝ) : Prop :=
  x^2 + y^2 - 6 * x + 1 = 0

-- Given M(0, -1)
def M : ℝ × ℝ := (0, -1)

-- Proof statement
theorem problem_solution :
  (∀ t : ℝ, let (x, y) := line_l t in line_l_standard x y) ∧
  (∀ ρ θ : ℝ, curve_C_polar ρ θ ↔ let x := ρ * Real.cos θ, y := ρ * Real.sin θ in curve_C_cartesian x y) ∧
  (∀ t₁ t₂ : ℝ, (t₁^2 - 4 * √2 * t₁ + 2 = 0) ∧ (t₂^2 - 4 * √2 * t₂ + 2 = 0) → (M.fst - (1 / √2) * t₁)^2 + (M.snd - (-1 + (1 / √2) * t₁))^2 * (M.fst - (1 / √2) * t₂)^2 + (M.snd - (-1 + (1 / √2) * t₂))^2 = 2) := 
by sorry

end problem_solution_l594_594297


namespace tangent_line_through_point_l594_594947

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def f' (x : ℝ) := 1 + Real.log x
noncomputable def g (x : ℝ) := Real.log x / x

theorem tangent_line_through_point (a e : ℝ) (h1 : e = Real.exp 1)
  (h2 : ∀ (m : ℝ), m > 0 → 1 + Real.log m = (m * Real.log m - a) / (m - a))
  (h3 : ∀ (m : ℝ), m > 0 → Deriv.g (m) = (1 - Real.log m) / m^2) :
  (a > e) :=
sorry

end tangent_line_through_point_l594_594947


namespace quadratic_roots_product_sum_l594_594681

theorem quadratic_roots_product_sum :
  (∀ d e : ℝ, 3 * d^2 + 4 * d - 7 = 0 ∧ 3 * e^2 + 4 * e - 7 = 0 →
   (d + 1) * (e + 1) = - 8 / 3) := by
sorry

end quadratic_roots_product_sum_l594_594681


namespace other_sides_of_square_l594_594381

open Real EuclideanGeometry

noncomputable def square_AB_side_eqns (P : Point ℝ) (AB_eqn : ℝ → ℝ → Prop) :=
  let A := (-1, 0)
  let B_eqn1 := 3*x - y - 3 = 0
  let B_eqn2 := x + 3*y - 5 = 0
  let B_eqn3 := x - 3 / 4 * y / (4 * x * y) - 9 = 0
  (P = A) ∧ (AB_eqn = λ x y, x + 3*y - 5 = 0) ∧ 
  (B_eqn1 ∨ B_eqn2 ∨ B_eqn3)

theorem other_sides_of_square (P : Point ℝ) (AB_eqn : ℝ → ℝ → Prop) (cond1 : P = (-1,0)) 
  (cond2 : AB_eqn = (λ x y, x + 3y - 5 = 0)) : 
  ∃ C D : ℝ × ℝ,  square_AB_side_eqns P AB_eqn :=
sorry

end other_sides_of_square_l594_594381


namespace find_interest_rate_l594_594718

-- Given conditions
def P : ℝ := 4099.999999999999
def t : ℕ := 2
def CI_minus_SI : ℝ := 41

-- Formulas for Simple Interest and Compound Interest
def SI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * (t : ℝ)
def CI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * ((1 + r) ^ t) - P

-- Main theorem to prove: the interest rate r is 0.1 (i.e., 10%)
theorem find_interest_rate (r : ℝ) : 
  (CI P r t - SI P r t = CI_minus_SI) → r = 0.1 :=
by
  sorry

end find_interest_rate_l594_594718


namespace length_of_side_c_proof_l594_594984

def length_of_side_c (a b A : ℝ) (hA : A = 120) (ha : a = 2 * sqrt 3) (hb : b = 2) : ℝ :=
  -- The length of side c given the values of a, b, and angle A
  let c_squared := b^2 + (2 * sqrt 3)^2 - 2 * b * (2 * sqrt 3) * cos (120 * real.pi / 180)
  sqrt c_squared

theorem length_of_side_c_proof :
  ∀ (a b A : ℝ), A = 120 → a = 2 * sqrt 3 → b = 2 → (length_of_side_c a b A = 2) :=
by
  intros a b A hA ha hb
  -- Calculation to prove that the length of c should be 2
  -- By using the Law of Cosines and the given values
  sorry

end length_of_side_c_proof_l594_594984


namespace min_G_of_p_is_8_div_17_l594_594594

def F (p q : ℝ) : ℝ := -4 * p * q + 5 * p * (1 - q) + 2 * (1 - p) * q - 6 * (1 - p) * (1 - q)

def G (p : ℝ) : ℝ := max (F p 0) (F p 1)

theorem min_G_of_p_is_8_div_17 : ∀ (p : ℝ), 0 ≤ p ∧ p ≤ 1 → (∃ p, p = 8/17 ∧ ∀ x ∈ set.Icc (0 : ℝ) 1, G p ≤ G x) :=
by
  sorry

end min_G_of_p_is_8_div_17_l594_594594


namespace sin_cos_symmetric_l594_594446

open Real

noncomputable def sin_func : ℝ → ℝ := sin
noncomputable def cos_func : ℝ → ℝ := cos

def symmetric_axes (k : ℤ) : ℝ := k * π + (π / 4)

theorem sin_cos_symmetric (k : ℤ) : 
  ∀ h : ℝ, sin_func (symmetric_axes k - h) = cos_func (symmetric_axes k + h) :=
sorry

end sin_cos_symmetric_l594_594446


namespace count_15_tuples_l594_594538

-- Define the condition for b_i in the 15-tuple
def condition (b : Fin 15 → ℤ) : Prop :=
  ∀ i, b i ^ 2 = (Finset.univ.sum (λ j, b j)) - (b i)

-- Define the theorem
theorem count_15_tuples : ∃ n : ℕ, n = 2730 ∧ ∃ b : ℕ, (Sum (b $ Fin.mk 0 sorry, b $ Fin.mk 1 sorry, ..., b $ Fin.mk 14 sorry)) ∧ enat.to_nat ( (/ finset.card (Finset.image _ (ordered_tuples_of_condition b))) )  = n := sorry

end count_15_tuples_l594_594538


namespace untouched_diameter_length_l594_594834

theorem untouched_diameter_length (wheel_radius semicircle_radius : ℝ) (h_wheel : wheel_radius = 8) (h_semicircle : semicircle_radius = 25) :
  let length_untouched := 2 * (semicircle_radius - real.sqrt (semicircle_radius ^ 2 - wheel_radius ^ 2))
  length_untouched = 20 :=
by
  have hc := by linarith [h_semicircle, h_wheel]; exact sorry

end untouched_diameter_length_l594_594834


namespace collinear_A_M_X_l594_594922

-- Define the basic setup
variables {A B C D E M T Q X : Type}
variables [triangle ABC : Triangle]
variables {O : Point}
variables (circumcircle_ABC : Circle ABC O)
variables (M : midpoint B C)
variables (D : Point) (H : Point)
variables (T : Point) (Q : Point)
variables (circumcircle_ETQ : Circle E T Q)
variables {X : Point}

-- Define the conditions
variables (H_perpendicular_AD : perpendicular AD BC)
variables (BDCT_parallelogram : parallelogram BD C T)
variables (Q_properties : (Angle BQM = Angle BCA) ∧ (Angle CQM = Angle CBA))
variables (AO_intersect : intersects AO circumcircle_ABC)
variables (E_non_equal_A : E ≠ A)
variables (circumcircle_intersection : intersects (circumcircle_ETQ) (circumcircle_ABC))
variables (X_non_equal_E : X ≠ E)

-- State the theorem
theorem collinear_A_M_X :
  collinear A M X := 
begin
  sorry
end

end collinear_A_M_X_l594_594922


namespace tiles_are_not_interchangeable_l594_594725

/-- The floor is tiled with 2x2 and 1x4 tiles following a specific coloring pattern:
    Row 1: blue, red, blue, red, ...
    Row 2: black, white, black, white, ...
    Row 3: blue, red, blue, red, ...
    Row 4: black, white, black, white, ...
    One tile is broken.
    Prove that you cannot replace a broken tile with a tile of the other type while maintaining the color pattern.
-/
theorem tiles_are_not_interchangeable (floor : list (list char)) (broken_tile : (nat × nat)) 
  (color_pattern : list (list char)) :
  (∀x y, 
    (floor[x][y] = '2' ∧ 
      (color_pattern[x][y] ∈ ['b', 'r', 'b', 'r'] ∨ 
       color_pattern[x][y] ∈ ['k', 'w', 'k', 'w'])) ∨
    (floor[x][y] = '4' ∧ 
      (color_pattern[x][y] ∈ ['b', 'r', 'b', 'r'] ∨ 
       color_pattern[x][y] ∈ ['k', 'w', 'k', 'w']))) →
    ¬(∃ new_tile, 
       (new_tile ≠ '2' ∧ new_tile ≠ '4') ∧ 
       floor[broken_tile.fst][broken_tile.snd] = new_tile ∧
       (∀x y, 
         floor[x][y] = new_tile → 
         (color_pattern[x][y] ∈ ['b', 'r', 'b', 'r'] ∨ 
         color_pattern[x][y] ∈ ['k', 'w', 'k', 'w'])))) :=
sorry

end tiles_are_not_interchangeable_l594_594725


namespace tom_bonus_points_l594_594613

theorem tom_bonus_points (customers_per_hour : ℕ) (hours_worked : ℕ) : 
  customers_per_hour = 10 → 
  hours_worked = 8 → 
  (customers_per_hour * hours_worked * 20) / 100 = 16 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  norm_num
  sorry

end tom_bonus_points_l594_594613


namespace correct_option_d_l594_594104

-- Define variables and constants.
variable (a : ℝ)

-- State the conditions as definitions.
def optionA := a^2 * a^3 = a^5
def optionB := (3 * a)^2 = 9 * a^2
def optionC := a^6 / a^3 = a^3
def optionD := 3 * a^2 - a^2 = 2 * a^2

-- The theorem states that the correct option is D.
theorem correct_option_d : optionD := by
  sorry

end correct_option_d_l594_594104


namespace new_percentage_female_workers_l594_594412

theorem new_percentage_female_workers (E : ℕ) (H : 0.60 * E = 0.60 * 308) :
  let original_female_workers := 0.60 * E
  let new_total_employees := E + 28
  336 = new_total_employees →
  let new_percentage_female := (original_female_workers / 336) * 100
  new_percentage_female ≈ 55.06 := 
sorry

end new_percentage_female_workers_l594_594412


namespace rotating_pentagon_does_not_change_sum_l594_594488

-- Definitions of regular pentagon and the sum of interior angles
noncomputable def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem rotating_pentagon_does_not_change_sum :
  ∀ (θ : ℝ) (n : ℕ) (h_n : n = 5) (h_θ : θ = 72), sum_interior_angles n = 540 :=
by
  intros θ n h_n h_θ
  rw [h_n]
  show sum_interior_angles 5 = 540
  dsimp [sum_interior_angles]
  norm_num
  sorry

end rotating_pentagon_does_not_change_sum_l594_594488


namespace abs_ac_bd_leq_one_l594_594188

theorem abs_ac_bd_leq_one {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) : |a * c + b * d| ≤ 1 :=
by
  sorry

end abs_ac_bd_leq_one_l594_594188


namespace xiaoming_grade_is_89_l594_594477

noncomputable def xiaoming_physical_education_grade
  (extra_activity_score : ℕ) (midterm_score : ℕ) (final_exam_score : ℕ)
  (ratio_extra : ℕ) (ratio_mid : ℕ) (ratio_final : ℕ) : ℝ :=
  (extra_activity_score * ratio_extra + midterm_score * ratio_mid + final_exam_score * ratio_final) / (ratio_extra + ratio_mid + ratio_final)

theorem xiaoming_grade_is_89 :
  xiaoming_physical_education_grade 95 90 85 2 4 4 = 89 := by
    sorry

end xiaoming_grade_is_89_l594_594477


namespace number_of_odd_four_digit_numbers_l594_594537

-- Defining the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def no_repeated_digits (digits : List ℕ) : Prop := digits.nodup
def digits_adjacent (d1 d2 : ℕ) (digits : List ℕ) : Prop :=
  ∃ i, digits.nth i = some d1 ∧ digits.nth (i+1) = some d2 ∨
  digits.nth i = some d2 ∧ digits.nth (i+1) = some d1

-- The set of allowed digits
def allowed_digits := [2, 3, 4, 5, 6]

-- Main statement: Proving the number of satisfying four-digit numbers is 14
theorem number_of_odd_four_digit_numbers : ∃ l : List ℕ,
  l.length = 4 ∧
  no_repeated_digits l ∧
  is_odd (l.nth 3).get_or_else 0 ∧
  digits_adjacent 5 6 l ∧
  l.all (λ d, d ∈ allowed_digits) ∧
  l.dedup.length = l.length ∧
  (sorry : (number_of_such_l_of_length 4 = 14)) :=
sorry

end number_of_odd_four_digit_numbers_l594_594537


namespace cost_of_each_soda_l594_594650

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l594_594650


namespace find_fn_zero_l594_594226

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x^2)

def f_derivative : ℕ → ℝ → ℝ
| 1    := f
| (n + 1) := λ x, (f_derivative n)' x

theorem find_fn_zero (n : ℕ) : f_derivative n 0 = (1 + (-1 : ℤ)^n) / 2 * n.factorial := by
  sorry

end find_fn_zero_l594_594226


namespace reciprocal_neg4_l594_594059

def reciprocal (x : ℝ) : ℝ :=
  1 / x

theorem reciprocal_neg4 : reciprocal (-4) = -1 / 4 := by
  sorry

end reciprocal_neg4_l594_594059


namespace range_of_omega_l594_594547

noncomputable def g (ω : ℝ) (x : ℝ) := cos (ω * x + (2 * real.pi / 3))

theorem range_of_omega (ω : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) (2 * real.pi / 3), g ω x = 0 → g ω x > 0) ∧
  (∀ x ∈ set.Icc ((-real.pi) / 12) (real.pi / 12), 
    (differentiable_at ℝ (g ω) x) ∧ 
    (deriv (g ω) x < 0)) →
  ω ∈ set.Icc ((11 : ℝ) / 4) 4 :=
sorry

end range_of_omega_l594_594547


namespace solution_to_equation_l594_594364

theorem solution_to_equation :
  (∀ x : ℝ, (|2 * x + 3| - |x - 1| = 4 * x - 3) ↔ x = 7 / 3) := by
sorr

end solution_to_equation_l594_594364


namespace max_value_of_f_l594_594664

noncomputable def f (x : ℝ) : ℝ := min (2^x) (min (x + 2) (10 - x))

theorem max_value_of_f : ∃ M, (∀ x ≥ 0, f x ≤ M) ∧ (∃ x ≥ 0, f x = M) ∧ M = 6 :=
by
  sorry

end max_value_of_f_l594_594664


namespace erasers_per_friend_l594_594186

variable (erasers friends : ℕ)

theorem erasers_per_friend (h1 : erasers = 3840) (h2 : friends = 48) :
  erasers / friends = 80 :=
by sorry

end erasers_per_friend_l594_594186


namespace partition_exists_min_n_in_A_l594_594114

-- Definition of subsets and their algebraic properties
variable (A B C : Set ℕ)

-- The Initial conditions
axiom A_squared_eq_A : ∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)
axiom B_squared_eq_C : ∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)
axiom C_squared_eq_B : ∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)
axiom AB_eq_B : ∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)
axiom AC_eq_C : ∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)
axiom BC_eq_A : ∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)

-- Statement for the partition existence with given conditions
theorem partition_exists :
  ∃ A B C : Set ℕ, (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
               (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
               (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
               (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
               (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
               (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) :=
sorry

-- Statement for the minimum n in A such that n and n+1 are both in A is at most 77
theorem min_n_in_A :
  ∀ A B C : Set ℕ,
    (∀ a b : ℕ, (a ∈ A) → (b ∈ A) → (a * b ∈ A)) ∧
    (∀ a b : ℕ, (a ∈ B) → (b ∈ B) → (a * b ∈ C)) ∧
    (∀ a b : ℕ, (a ∈ C) → (b ∈ C) → (a * b ∈ B)) ∧
    (∀ a b : ℕ, (a ∈ A) → (b ∈ B) → (a * b ∈ B)) ∧
    (∀ a c : ℕ, (a ∈ A) → (c ∈ C) → (a * c ∈ C)) ∧
    (∀ b c : ℕ, (b ∈ B) → (c ∈ C) → (b * c ∈ A)) →
    ∃ n : ℕ, (n ∈ A) ∧ (n + 1 ∈ A) ∧ n ≤ 77 :=
sorry

end partition_exists_min_n_in_A_l594_594114


namespace intersect_area_is_correct_l594_594605

theorem intersect_area_is_correct :
  ∃ (x₁ x₂ : ℝ), (x₁ - x₂).abs = 2 * Real.pi / 3 ∧ 
  (∃ y : ℝ, y = 1 ∧ ∃ a b : ℝ, a = x₁ ∧ b = x₂ ∧ 
  (a < b) ∧ 
  ∫ x in a..b, 2 * Real.sin (2 * x) * 2 = 
  (2 * Real.pi / 3 + Real.sqrt 3) :=
by
  sorry

end intersect_area_is_correct_l594_594605


namespace largest_possible_factors_l594_594043

theorem largest_possible_factors (x : ℝ) :
  ∃ m q1 q2 q3 q4 : polynomial ℝ,
    m = 4 ∧
    x^10 - 1 = q1 * q2 * q3 * q4 ∧
    ¬(q1.degree = 0) ∧ ¬(q2.degree = 0) ∧ ¬(q3.degree = 0) ∧ ¬(q4.degree = 0) :=
sorry

end largest_possible_factors_l594_594043


namespace find_x_plus_y_div_3_l594_594452

theorem find_x_plus_y_div_3 (x y : ℝ) 
  (h1 : 2 * x + y = 7) 
  (h2 : x + 2 * y = 10) : 
  (x + y) / 3 = 1.888... :=
sorry

end find_x_plus_y_div_3_l594_594452


namespace proof_by_contradiction_example_l594_594441

theorem proof_by_contradiction_example (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end proof_by_contradiction_example_l594_594441


namespace solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l594_594977

-- Definitions as conditions
def is_cone (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_cylinder (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_triangular_pyramid (solid : Type) : Prop := -- Definition placeholder
sorry 

def is_rectangular_prism (solid : Type) : Prop := -- Definition placeholder
sorry 

-- Predicate to check if the front view of a solid is a quadrilateral
def front_view_is_quadrilateral (solid : Type) : Prop :=
  (is_cylinder solid ∨ is_rectangular_prism solid)

-- Theorem stating the problem
theorem solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism
    (s : Type) :
  front_view_is_quadrilateral s ↔ is_cylinder s ∨ is_rectangular_prism s :=
by
  sorry

end solids_with_quadrilateral_front_view_are_cylinder_and_rectangular_prism_l594_594977


namespace monomials_exponents_l594_594981

theorem monomials_exponents (m n : ℕ) 
  (h₁ : 3 * x ^ 5 * y ^ m + -2 * x ^ n * y ^ 7 = 0) : m - n = 2 := 
by
  sorry

end monomials_exponents_l594_594981


namespace centroid_distance_to_line_RS_l594_594704

-- Definitions of given lengths
def AD : ℝ := 12
def BE : ℝ := 8
def CF : ℝ := 20

-- Definition of y-coordinates
def y_A : ℝ := AD
def y_B : ℝ := BE
def y_C : ℝ := CF

-- Definition of centroid's y-coordinate
def y_G : ℝ := (y_A + y_B + y_C) / 3

-- Goal: Prove that the perpendicular distance, GH, is 40/3
theorem centroid_distance_to_line_RS : y_G = 40 / 3 := by
  sorry

end centroid_distance_to_line_RS_l594_594704


namespace necessary_but_not_sufficient_condition_l594_594245

-- Conditions from the problem
def p (x : ℝ) : Prop := (x - 1) * (x - 3) ≤ 0
def q (x : ℝ) : Prop := 2 / (x - 1) ≥ 1

-- Theorem statement to prove the correct answer
theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, p x → (¬p x → ¬q x) ∧ (q x → p x) ∧ (∃ x, p x ∧ ¬q x) :=
by
  -- Solution is implied by given answer in problem description
  sorry

end necessary_but_not_sufficient_condition_l594_594245


namespace number_of_correct_inequalities_l594_594548

variable {a b : ℝ}

theorem number_of_correct_inequalities (h₁ : a > 0) (h₂ : 0 > b) (h₃ : a + b > 0) :
  (ite (a^2 > b^2) 1 0) + (ite (1/a > 1/b) 1 0) + (ite (a^3 < ab^2) 1 0) + (ite (a^2 * b < b^3) 1 0) = 3 := 
sorry

end number_of_correct_inequalities_l594_594548


namespace number_of_girls_l594_594054

variable (g b : ℕ) -- Number of girls (g) and boys (b) in the class
variable (h_ratio : g / b = 4 / 3) -- The ratio condition
variable (h_total : g + b = 63) -- The total number of students condition

theorem number_of_girls (g b : ℕ) (h_ratio : g / b = 4 / 3) (h_total : g + b = 63) :
    g = 36 :=
sorry

end number_of_girls_l594_594054


namespace determine_bisecting_line_l594_594811

theorem determine_bisecting_line (x y : ℝ) (b : ℝ) (h_circle : x^2 + y^2 - 2*x - 4*y = 0) (h_parallel : y = -0.5 * x) : 
 x + 2 * y - 5 = 0 :=
by
  -- We established that the line is parallel to x + 2y = 0
  calc
  1 + 4 + b = 0 : sorry
  b = -5 : sorry
  
  -- To conclude the equation of the bisecting line
  x + 2*y + b = 0 := sorry

end determine_bisecting_line_l594_594811


namespace lattice_points_on_diagonal_30_45_l594_594468

theorem lattice_points_on_diagonal_30_45 : 
  let rect_width := 30
  let rect_height := 45 
  let lattice_points := rect_width / (nat.gcd rect_width rect_height) + rect_height / (nat.gcd rect_width rect_height) - 1 
  in lattice_points = 16 := by
  sorry

end lattice_points_on_diagonal_30_45_l594_594468


namespace find_discount4_l594_594731

noncomputable def list_price : ℝ := 35000
noncomputable def paid_price : ℝ := 27922.95
noncomputable def sales_tax_rate : ℝ := 0.05
noncomputable def discount1 : ℝ := 0.20
noncomputable def discount2 : ℝ := 0.125
noncomputable def discount3 : ℝ := 0.05
noncomputable def discounted_price (p : ℝ) (d : ℝ) : ℝ := p * (1 - d)
noncomputable def pre_tax_price (paid : ℝ) : ℝ := paid / (1 + sales_tax_rate)

theorem find_discount4 :
  let p1 := discounted_price list_price discount1,
      p2 := discounted_price p1 discount2,
      p3 := discounted_price p2 discount3,
      pBT := pre_tax_price paid_price,
      D4_percent := 100 * (1 - pBT / p3)
  in D4_percent = 14.46 := sorry

end find_discount4_l594_594731


namespace balance_ways_l594_594746

/- Define the conditions as specified in the problem -/
def valid_weights (x y z : ℤ) : Prop :=
  abs x ≤ 4 ∧ abs y ≤ 3 ∧ abs z ≤ 2

/- Define the equation to balance the weights -/
def balance (x y z : ℤ) : Prop :=
  abs (2 * x + 3 * y + 5 * z) = 12

/- Prove the number of valid (x, y, z) combinations satisfying the conditions -/
theorem balance_ways : {p : ℤ × ℤ × ℤ // valid_weights p.1 p.2 p.3 ∧ balance p.1 p.2 p.3}.to_finset.card = 7 :=
by
  sorry

end balance_ways_l594_594746


namespace sum_f_values_l594_594252

noncomputable def f (x : ℚ) : ℚ := x^3 - (3/2)*x^2 + (3/4)*x + (1/8)

theorem sum_f_values : 
  ∑ k in Finset.range 2016 | (1 : ℚ) ≤ k + 1 ∧ k + 1 ≤ 2016, f ((k + 1 : ℚ) / 2017) = 504 :=
  sorry

end sum_f_values_l594_594252


namespace find_b_l594_594571

noncomputable def tangent_condition (b : ℝ) : Prop :=
∀ (x : ℝ), (x > 0) → 
(∃ (y : ℝ), y = log x ∧ y = (1/2) * x + b ∧ (1/x) = (1/2))

theorem find_b : ∃ b : ℝ, tangent_condition b ∧ b = log 2 - 1 := 
by
  -- Proof goes here (this statement shows that such b exists and is indeed log 2 - 1)
  sorry

end find_b_l594_594571


namespace jennifer_marbles_l594_594646

noncomputable def choose_ways (total special non_special choose_total choose_special choose_non_special : ℕ) : ℕ :=
  let ways_special := choose_special * choose_non_special
  ways_special

theorem jennifer_marbles :
  let total := 20
  let red := 3
  let green := 3
  let blue := 2
  let special := red + green + blue
  let non_special := total - special
  let choose_total := 5
  let choose_special := 2
  let choose_non_special := choose_total - choose_special
  let ways_special :=
    (Nat.choose red 2) + (Nat.choose green 2) + (Nat.choose blue 2) +
    ((Nat.choose red 1) * (Nat.choose green 1)) +
    ((Nat.choose red 1) * (Nat.choose blue 1)) +
    ((Nat.choose green 1) * (Nat.choose blue 1))
  let ways_non_special := Nat.choose non_special 3
  choose_ways total special non_special choose_total choose_special choose_non_special = 6160 :=
by
  simp only [choose_ways]
  exact sorry

end jennifer_marbles_l594_594646


namespace magnitude_relation_l594_594727

noncomputable def f : ℝ → ℝ := sorry  -- Define f as a noncomputable function since details are not given

variable (a b c : ℝ)

-- Defining conditions
axiom inc_f : ∀ x y : ℝ, x < y → f(x) < f(y)   -- f is an increasing function
axiom a_def : a = f(2) < 0                     -- a is f(2) which is less than 0
axiom b_def : b = classical.some (exists_inverse (f(2))) -- b is the inverse value f(b) = 2
axiom c_def : c = classical.some (exists_inverse (f(0))) -- c is the inverse value f(c) = 0

-- Prove the required relationship
theorem magnitude_relation : b > c ∧ c > a :=
sorry

end magnitude_relation_l594_594727


namespace paint_for_smaller_statues_l594_594140

open Real

theorem paint_for_smaller_statues :
  ∀ (paint_needed : ℝ) (height_big_statue height_small_statue : ℝ) (num_small_statues : ℝ),
  height_big_statue = 10 → height_small_statue = 2 → paint_needed = 5 → num_small_statues = 200 →
  (paint_needed / (height_big_statue / height_small_statue) ^ 2) * num_small_statues = 40 :=
by
  intros paint_needed height_big_statue height_small_statue num_small_statues
  intros h_big_height h_small_height h_paint_needed h_num_small
  rw [h_big_height, h_small_height, h_paint_needed, h_num_small]
  sorry

end paint_for_smaller_statues_l594_594140


namespace all_points_on_same_circle_l594_594660

-- Define the given conditions
variable (P : Set Point)  -- P is a set of points in the plane
variable (h : ∀ (T : Set Point), T ⊆ P → T.card = 4 → ∃ (c : Circle), ∀ (p ∈ T), p ∈ c)

-- The statement to prove
theorem all_points_on_same_circle (P : Set Point) (h : ∀ (T : Set Point), T ⊆ P → T.card = 4 → ∃ (c : Circle), ∀ (p ∈ T), p ∈ c) : 
  ∃ (c : Circle), ∀ (p ∈ P), p ∈ c :=
sorry

end all_points_on_same_circle_l594_594660


namespace circumscribed_sphere_radius_is_3_l594_594231

noncomputable def radius_of_circumscribed_sphere (SA SB SC : ℝ) : ℝ :=
  let space_diagonal := Real.sqrt (SA^2 + SB^2 + SC^2)
  space_diagonal / 2

theorem circumscribed_sphere_radius_is_3 : radius_of_circumscribed_sphere 2 4 4 = 3 :=
by
  unfold radius_of_circumscribed_sphere
  simp
  apply sorry

end circumscribed_sphere_radius_is_3_l594_594231


namespace mass_of_cork_l594_594820

theorem mass_of_cork (ρ_p ρ_w ρ_s : ℝ) (m_p x : ℝ) :
  ρ_p = 2.15 * 10^4 → 
  ρ_w = 2.4 * 10^2 →
  ρ_s = 4.8 * 10^2 →
  m_p = 86.94 →
  x = 2.4 * 10^2 * (m_p / ρ_p) →
  x = 85 :=
by
  intros
  sorry

end mass_of_cork_l594_594820


namespace area_of_triangle_is_correct_l594_594307

noncomputable def area_of_triangle_ABC
  (A B C : ℝ) (a b c : ℝ)
  (h1 : B = 2 * C)
  (h2 : b = 6)
  (h3 : c = 5)
  : ℝ :=
  have h4 : a = 11 / 5
    := by sorry,
  let sin_C := (4 : ℝ) / 5 in
  let result := (1 / 2) * a * b * sin_C in
  result

theorem area_of_triangle_is_correct (A B C : ℝ) (a b c : ℝ)
  (h1 : B = 2 * C)
  (h2 : b = 6)
  (h3 : c = 5) :
  area_of_triangle_ABC A B C a b c h1 h2 h3 = 132 / 25 :=
  by sorry

end area_of_triangle_is_correct_l594_594307


namespace number_of_integers_l594_594587

theorem number_of_integers (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 2020) (h3 : ∃ k : ℕ, n^n = k^2) : n = 1032 :=
sorry

end number_of_integers_l594_594587


namespace num_positive_integers_condition_l594_594901

theorem num_positive_integers_condition : 
  ∃! n : ℤ, 0 < n ∧ n < 50 ∧ (n + 2) % (50 - n) = 0 :=
by
  sorry

end num_positive_integers_condition_l594_594901


namespace transform_log_graph_l594_594083

noncomputable def f (x : ℝ) : ℝ := log x
noncomputable def g (x : ℝ) : ℝ := log (x + 2) - 1

theorem transform_log_graph :
  (∀ x, g x = f (x + 2) - 1) :=
by sorry

end transform_log_graph_l594_594083


namespace inverse_of_f_at_2_l594_594386

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem inverse_of_f_at_2 : ∀ x, x ≥ 0 → f x = 2 → x = Real.sqrt 3 :=
by
  intro x hx heq
  sorry

end inverse_of_f_at_2_l594_594386


namespace jackie_push_ups_proof_l594_594645

-- Define the conditions given in the problem
def jackie_push_ups (total_time : ℕ) (break_time : ℕ) (push_ups_per_time : ℕ) (push_up_duration : ℕ) : ℕ :=
  let effective_time := total_time - break_time in
  effective_time / push_up_duration * push_ups_per_time

theorem jackie_push_ups_proof :
  let total_time := 60 in
  let break_time := 2 * 8 in
  let push_ups_per_time := 5 in
  let push_up_duration := 10 / 5 in
  jackie_push_ups total_time break_time push_ups_per_time push_up_duration = 22 :=
by
  -- Sorry keyword to skip proof
  sorry

end jackie_push_ups_proof_l594_594645


namespace min_value_of_expression_l594_594784

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

theorem min_value_of_expression (a b c : ℝ × ℝ) (x y : ℝ) 
  (h1 : vector_magnitude a = 2) 
  (h2 : vector_magnitude b = 2) 
  (h3 : vector_magnitude c = 2) 
  (h4 : a.1 + b.1 + c.1 = 0 ∧ a.2 + b.2 + c.2 = 0)
  (h5 : 0 ≤ x ∧ x ≤ 1/2) 
  (h6 : 1/2 ≤ y ∧ y ≤ 1) : 
  ∃ m, m = |x * (a.1 - c.1, a.2 - c.2) + y * (b.1 - c.1, b.2 - c.2) + c| ∧ m = 1/2 :=
sorry

end min_value_of_expression_l594_594784


namespace area_of_triangle_l594_594494

theorem area_of_triangle : 
  let x1 := 1
  let y1 := 3
  let x2 := -2
  let y2 := 4
  let x3 := 4
  let y3 := -1
  let area := (1 / 2 : ℝ) * | x1 * y2 + x2 * y3 + x3 * y1 - y1 * x2 - y2 * x3 - y3 * x1 |
  area = 4.5 :=
by 
  sorry

end area_of_triangle_l594_594494


namespace part_a_l594_594464

def increment (p : ℝ → ℝ) (a b : ℝ) := p b - p a

def valid_partition (black_intervals white_intervals : List (ℝ × ℝ)) : Prop :=
  ∀ (p : ℝ → ℝ), degree_le p 2 → 
  (∑ bi in black_intervals, increment p bi.fst bi.snd) = 
  (∑ wi in white_intervals, increment p wi.fst wi.snd)

theorem part_a :
  valid_partition [(0, 0.25), (0.75, 1)] [(0.25, 0.75)] :=
sorry

end part_a_l594_594464


namespace min_value_expression_l594_594596

theorem min_value_expression (x : ℝ) (h : x > 1) : x + 9 / x - 2 ≥ 4 :=
sorry

end min_value_expression_l594_594596


namespace ivy_collectors_edition_dolls_l594_594868

-- Definitions from the conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def collectors_edition_dolls : ℕ := (2 * ivy_dolls) / 3

-- Assertion
theorem ivy_collectors_edition_dolls : collectors_edition_dolls = 20 := by
  sorry

end ivy_collectors_edition_dolls_l594_594868


namespace total_money_is_2800_l594_594161

-- Define variables for money
def Cecil_money : ℕ := 600
def Catherine_money : ℕ := 2 * Cecil_money - 250
def Carmela_money : ℕ := 2 * Cecil_money + 50

-- Assertion to prove the total money 
theorem total_money_is_2800 : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- placeholder proof
  sorry

end total_money_is_2800_l594_594161


namespace fraction_of_value_l594_594072

def value_this_year : ℝ := 16000
def value_last_year : ℝ := 20000

theorem fraction_of_value : (value_this_year / value_last_year) = 4 / 5 := by
  sorry

end fraction_of_value_l594_594072


namespace floor_e_eq_two_l594_594879

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end floor_e_eq_two_l594_594879


namespace adult_ticket_cost_l594_594372

-- Definitions based on the conditions
def num_adults : ℕ := 10
def num_children : ℕ := 11
def total_bill : ℝ := 124
def child_ticket_cost : ℝ := 4

-- The proof which determines the cost of one adult ticket
theorem adult_ticket_cost : ∃ (A : ℝ), A * num_adults = total_bill - (num_children * child_ticket_cost) ∧ A = 8 := 
by
  sorry

end adult_ticket_cost_l594_594372


namespace decorations_count_l594_594178

/-
Danai is decorating her house for Halloween. She puts 12 plastic skulls all around the house.
She has 4 broomsticks, 1 for each side of the front and back doors to the house.
She puts up 12 spiderwebs around various areas of the house.
Danai puts twice as many pumpkins around the house as she put spiderwebs.
She also places a large cauldron on the dining room table.
Danai has the budget left to buy 20 more decorations and has 10 left to put up.
-/

def plastic_skulls := 12
def broomsticks := 4
def spiderwebs := 12
def pumpkins := 2 * spiderwebs
def cauldron := 1
def budget_remaining := 20
def undecorated_items := 10

def initial_decorations := plastic_skulls + broomsticks + spiderwebs + pumpkins + cauldron
def additional_decorations := budget_remaining + undecorated_items
def total_decorations := initial_decorations + additional_decorations

theorem decorations_count : total_decorations = 83 := by
  /- Detailed proof steps -/
  sorry

end decorations_count_l594_594178


namespace sqrt_expression_l594_594166

theorem sqrt_expression (x : ℕ) (h : x = 19) : 
  Int.sqrt ((x + 2) * (x + 1) * x * (x - 1) + 1) = 379 :=
by
  rw h
  -- Placeholder for proof
  sorry

end sqrt_expression_l594_594166


namespace find_scalar_r_l594_594075

open Matrix

noncomputable def vec_a : Vector3 ℝ := ![3, 1, -2]
noncomputable def vec_b : Vector3 ℝ := ![0, 2, -1]
noncomputable def vec_c : Vector3 ℝ := ![5, 0, -3]
noncomputable def cross_product (u v : Vector3 ℝ) : Vector3 ℝ :=
  ![(u 1 * v 2 - u 2 * v 1), (u 2 * v 0 - u 0 * v 2), (u 0 * v 1 - u 1 * v 0)]
noncomputable def dot_product (u v : Vector3 ℝ) : ℝ :=
  u 0 * v 0 + u 1 * v 1 + u 2 * v 2

def scalar_r : ℝ := -1 / 18

theorem find_scalar_r :
  ∃ p q r, vec_c = p • vec_a + q • vec_b + r • cross_product vec_a vec_b ∧ r = scalar_r :=
by 
  use [0, 0, scalar_r]
  split
  case left =>
    sorry
  case right =>
    refl

end find_scalar_r_l594_594075


namespace merchant_articles_l594_594814

theorem merchant_articles 
   (CP SP : ℝ)
   (N : ℝ)
   (h1 : SP = 1.25 * CP)
   (h2 : N * CP = 16 * SP) : 
   N = 20 := by
   sorry

end merchant_articles_l594_594814


namespace computer_table_cost_price_l594_594455

theorem computer_table_cost_price (CP SP : ℝ) (h1 : SP = CP * (124 / 100)) (h2 : SP = 8091) :
  CP = 6525 :=
by
  sorry

end computer_table_cost_price_l594_594455


namespace maximum_value_of_a_l594_594236

theorem maximum_value_of_a
  (a b c d : ℝ)
  (h1 : b + c + d = 3 - a)
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) :
  a ≤ 2 := by
  sorry

end maximum_value_of_a_l594_594236


namespace coordinates_of_N_l594_594561

-- Define the given conditions
def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)
def minusThreeA : ℝ × ℝ := (-3, 6)
def vectorMN (N : ℝ × ℝ) : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define the required goal
theorem coordinates_of_N (N : ℝ × ℝ) : vectorMN N = minusThreeA → N = (2, 0) :=
by
  sorry

end coordinates_of_N_l594_594561


namespace area_AOB_l594_594633

def A : ℝ × ℝ := (3, Real.pi / 3)
def B_initial : ℝ × ℝ := (-4, 7 * Real.pi / 6)

-- Convert B from (-4, 7π/6) to (4, π/6) due to polar coordinates properties 
def B : ℝ × ℝ := (4, Real.pi / 6)

def area_of_triangle_OAOB (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * A.1 * B.1 * Real.sin (A.2 - B.2)

theorem area_AOB : area_of_triangle_OAOB A B = 3 := 
  sorry

end area_AOB_l594_594633


namespace log_expansion_l594_594158

/-- 
Prove that 21 * log 2 + log 25 = 2 where log a is the natural logarithm of a.
-/
theorem log_expansion : 21 * log 2 + log 25 = 2 :=
sorry

end log_expansion_l594_594158


namespace cycles_combined_final_selling_price_l594_594127

def cost_A := 1800
def repairs_A := 200
def discount_rate_A := 0.10
def sales_tax_rate_A := 0.05

def cost_B := 2400
def repairs_B := 300
def discount_rate_B := 0.12
def sales_tax_rate_B := 0.06

def cost_C := 3200
def repairs_C := 400
def discount_rate_C := 0.15
def sales_tax_rate_C := 0.07

def total_cost (initial_cost : ℕ) (repairs_cost : ℕ) : ℕ := 
  initial_cost + repairs_cost

def discount_amount (total_cost : ℕ) (discount_rate : ℝ) : ℝ := 
  total_cost * discount_rate

def discounted_price (total_cost : ℕ) (discount_rate : ℝ) : ℝ := 
  total_cost - total_cost * discount_rate

def sales_tax (discounted_price : ℝ) (sales_tax_rate : ℝ) : ℝ := 
  discounted_price * sales_tax_rate

def final_selling_price (initial_cost : ℕ) (repairs_cost : ℕ) (discount_rate : ℝ) (sales_tax_rate : ℝ) : ℝ := 
  let total := total_cost initial_cost repairs_cost
  let discounted := discounted_price total discount_rate
  discounted + sales_tax discounted sales_tax_rate

def combined_final_selling_price (pA pB pC : ℝ) : ℝ := 
  pA + pB + pC

theorem cycles_combined_final_selling_price :
  combined_final_selling_price 
    (final_selling_price cost_A repairs_A discount_rate_A sales_tax_rate_A) 
    (final_selling_price cost_B repairs_B discount_rate_B sales_tax_rate_B)
    (final_selling_price cost_C repairs_C discount_rate_C sales_tax_rate_C) = 7682.76 
  := by
  sorry

end cycles_combined_final_selling_price_l594_594127


namespace part1_part2_l594_594949

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - a - Real.log x

theorem part1 (a : ℝ) :
  (∀ x > 0, f x a ≥ 0) → a ≤ 1 := sorry

theorem part2 (a : ℝ) (x₁ x₂ : ℝ) (hx : 0 < x₁ ∧ x₁ < x₂) :
  (f x₁ a - f x₂ a) / (x₂ - x₁) < 1 / (x₁ * (x₁ + 1)) := sorry

end part1_part2_l594_594949


namespace squarefree_juicy_integers_complete_l594_594131

-- Define the concept of a juicy integer
def is_juicy (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d1 d2 : ℕ, (d1 ∣ n) ∧ (d2 ∣ n) ∧ (d1 < d2) → ((d2 - d1) ∣ n)

-- Define the concept of a squarefree integer
def is_squarefree (n : ℕ) : Prop :=
  ∀ p : ℕ, nat.prime p → p^2 ∣ n → false

-- Define the set of all squarefree juicy integers
noncomputable def squarefree_juicy_integers :=
  { n : ℕ | is_juicy n ∧ is_squarefree n }

-- The set of all squarefree juicy integers is exactly {2, 6, 42, 1806}
theorem squarefree_juicy_integers_complete :
  squarefree_juicy_integers = {2, 6, 42, 1806} :=
by
  -- Proof omitted
  sorry

end squarefree_juicy_integers_complete_l594_594131


namespace implied_statements_l594_594619

variable (Zelms Xenos Yarns Wurms : Set _) -- Assuming a universe of discourse

axiom Zelms_are_Xenos : Zelms ⊆ Xenos
axiom Yarns_are_Zelms : Yarns ⊆ Zelms
axiom Wurms_are_Yarns : Wurms ⊆ Yarns
axiom Xenos_are_Wurms : Xenos ⊆ Wurms

theorem implied_statements :
  (Yarns ⊆ Wurms ∧ Yarns ⊆ Zelms) ∧ (Wurms ⊆ Zelms ∧ Wurms ⊆ Xenos) :=
by
  sorry

end implied_statements_l594_594619


namespace gen_term_an_sum_Cn_l594_594243

-- Define the sequences and initial conditions
def seq_an (a : Nat → ℝ) := ∀ n : Nat, 0 < a n
def sum_Sn (a : Nat → ℝ) (S : Nat → ℝ) := ∀ n : Nat, S n = ∑ i in finset.range n, a i
def arithmetic_seq (S a : Nat → ℝ) := ∀ n : Nat, (2 : ℝ) * a n = S n + (1 / 2)

-- The first proof: general term formula for the sequence {a_n}
theorem gen_term_an {a S : Nat → ℝ} (h1 : seq_an a) (h2 : sum_Sn a S) 
    (h3 : arithmetic_seq S a) : ∀ n, a n = 2 ^ (n - 2) :=
by
  sorry

-- The second proof: sum of first n terms for the sequence {C_n}
theorem sum_Cn {a b C : Nat → ℝ} (h1 : ∀ n, a n = 2 ^ (n - 2))
    (h2 : ∀ n, a n ^ 2 = 2 ^ (- (b n))) 
    (h3 : ∀ n, C n = b n / a n) : ∀ n, ∑ i in finset.range n, C i = (8 * n) / (2 ^ n) :=
by 
  sorry

end gen_term_an_sum_Cn_l594_594243


namespace sum_of_digits_of_d_l594_594872

theorem sum_of_digits_of_d (d : ℕ) (h₁ : ∃ d_ca : ℕ, d_ca = (8 * d) / 5) (h₂ : d_ca - 75 = d) :
  (1 + 2 + 5 = 8) :=
by
  sorry

end sum_of_digits_of_d_l594_594872


namespace cost_of_each_soda_l594_594649

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l594_594649


namespace part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l594_594579

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem part1_min_value : ∀ (x : ℝ), x > 0 → f x ≥ -1 / Real.exp 1 := 
by sorry

noncomputable def g (x k : ℝ) : ℝ := f x - k * (x - 1)

theorem part2_max_value_k_lt : ∀ (k : ℝ), k < Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ Real.exp 1 - k * Real.exp 1 + k :=
by sorry

theorem part2_max_value_k_geq : ∀ (k : ℝ), k ≥ Real.exp 1 / (Real.exp 1 - 1) → 
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → g x k ≤ 0 :=
by sorry

end part1_min_value_part2_max_value_k_lt_part2_max_value_k_geq_l594_594579


namespace minimal_odd_sum_is_1683_l594_594088

/-!
# Proof Problem:
Prove that the minimal odd sum of two three-digit numbers and one four-digit number 
formed using the digits 0 through 9 exactly once is 1683.
-/
theorem minimal_odd_sum_is_1683 :
  ∃ (a b : ℕ) (c : ℕ), 
    100 ≤ a ∧ a < 1000 ∧ 
    100 ≤ b ∧ b < 1000 ∧ 
    1000 ≤ c ∧ c < 10000 ∧ 
    a + b + c % 2 = 1 ∧ 
    (∀ d e f : ℕ, 
      100 ≤ d ∧ d < 1000 ∧ 
      100 ≤ e ∧ e < 1000 ∧ 
      1000 ≤ f ∧ f < 10000 ∧ 
      d + e + f % 2 = 1 → a + b + c ≤ d + e + f) ∧ 
    a + b + c = 1683 := 
sorry

end minimal_odd_sum_is_1683_l594_594088


namespace power_of_m_divisible_by_33_l594_594282

theorem power_of_m_divisible_by_33 (m : ℕ) (h : m > 0) (k : ℕ) (h_pow : (m ^ k) % 33 = 0) :
  ∃ n, n > 0 ∧ 11 ∣ m ^ n :=
by
  sorry

end power_of_m_divisible_by_33_l594_594282


namespace fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l594_594212

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

-- Define what it means to be a fixed point
def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

-- Condition 1: a = 1, b = 5; the fixed points are x = -1 or x = -4
theorem fixed_points_a_one_b_five : 
  ∀ x : ℝ, is_fixed_point (f 1 5) x ↔ x = -1 ∨ x = -4 := by
  -- Proof goes here
  sorry

-- Condition 2: For any real b, f(x) always having two distinct fixed points implies 0 < a < 1
theorem range_of_a_two_distinct_fixed_points : 
  (∀ b : ℝ, ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point (f a b) x1 ∧ is_fixed_point (f a b) x2) ↔ 0 < a ∧ a < 1 := by
  -- Proof goes here
  sorry

end fixed_points_a_one_b_five_range_of_a_two_distinct_fixed_points_l594_594212


namespace value_of_expression_l594_594224

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 - 2005 = -2004 :=
by
  sorry

end value_of_expression_l594_594224


namespace subset_fourth_power_l594_594339

theorem subset_fourth_power (M : Finset ℕ) (hM1 : M.card = 1985) 
  (hM2 : ∀ n ∈ M, ∀ p : ℕ, p.prime → (p ∣ n → p ≤ 26)) :
  ∃ A : Finset ℕ, A ⊆ M ∧ A.card = 4 ∧ ∃ k : ℕ, ∏ a in A, a = k ^ 4 :=
begin
  sorry
end

end subset_fourth_power_l594_594339


namespace problem_a_b_squared_l594_594668

theorem problem_a_b_squared {a b : ℝ} (h1 : a + 3 = (b-1)^2) (h2 : b + 3 = (a-1)^2) (h3 : a ≠ b) : a^2 + b^2 = 10 :=
by
  sorry

end problem_a_b_squared_l594_594668


namespace problem_I_problem_II_l594_594255

noncomputable def f (x a : ℝ) : ℝ := 2 / x + a * Real.log x

theorem problem_I (a : ℝ) (h : a > 0) (h' : (2:ℝ) = (1 / (4 / a)) * (a^2) / 8):
  ∃ x : ℝ, f x a = f (1 / 2) a := sorry

theorem problem_II (a : ℝ) (h : a > 0) (h' : ∃ x : ℝ, f x a < 2) :
  (True : Prop) := sorry

end problem_I_problem_II_l594_594255


namespace M_diff_N_eq_l594_594659

noncomputable def A_diff_B (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∉ B }

noncomputable def M : Set ℝ := { x | -3 ≤ x ∧ x ≤ 1 }

noncomputable def N : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem M_diff_N_eq : A_diff_B M N = { x | -3 ≤ x ∧ x < 0 } :=
by
  sorry

end M_diff_N_eq_l594_594659


namespace orange_juice_fraction_correct_l594_594427

-- Define the total capacity of each pitcher
def total_capacity : ℕ := 800

-- Define the fraction of the orange juice in the first pitcher
def fraction_orange_juice : ℚ := 1 / 4

-- Define the fraction of the apple juice in the second pitcher
def fraction_apple_juice : ℚ := 3 / 8

-- Define the amount of orange juice in the first pitcher
def amount_orange_juice : ℕ := total_capacity * (fraction_orange_juice).toNat

-- Define the amount of apple juice in the second pitcher
def amount_apple_juice : ℕ := total_capacity * (fraction_apple_juice).toNat

-- Define the total volume in the large container
def total_volume : ℕ := 2 * total_capacity

-- Define the fraction of orange juice in the mixture
def fraction_of_orange_juice : ℚ := (amount_orange_juice : ℚ) / (total_volume)

-- Lean theorem stating the problem and the expected solution
theorem orange_juice_fraction_correct :
  fraction_of_orange_juice = 1 / 8 :=
begin
  sorry
end

end orange_juice_fraction_correct_l594_594427


namespace sale_record_is_negative_five_l594_594833

-- Given that a purchase of 10 items is recorded as +10
def purchase_record (items : Int) : Int := items

-- Prove that the sale of 5 items should be recorded as -5
theorem sale_record_is_negative_five : purchase_record 10 = 10 → purchase_record (-5) = -5 :=
by
  intro h
  sorry

end sale_record_is_negative_five_l594_594833


namespace f_of_1_eq_zero_l594_594382

-- Conditions
variables (f : ℝ → ℝ)
-- f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x
-- f is a periodic function with a period of 2
def periodic_function (f : ℝ → ℝ) := ∀ x : ℝ, f (x + 2) = f x

-- Theorem statement
theorem f_of_1_eq_zero {f : ℝ → ℝ} (h1 : odd_function f) (h2 : periodic_function f) : f 1 = 0 :=
by { sorry }

end f_of_1_eq_zero_l594_594382


namespace circular_seat_coloring_l594_594362

def count_colorings (n : ℕ) : ℕ :=
  sorry

theorem circular_seat_coloring :
  count_colorings 6 = 66 :=
by
  sorry

end circular_seat_coloring_l594_594362


namespace A2023_coords_l594_594999

-- Define a structure for a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a function to compute the conjugate point
def conjugate (p : Point) : Point :=
  Point.mk (-p.y + 1) (p.x + 1)

-- Define the initial point A1
def A1 : Point := Point.mk 2 2

-- Define a recursive function to compute An
def An (n : ℕ) : Point :=
  match n with
  | 0     => A1
  | (n+1) => conjugate (An n)

-- The goal is to prove that A2023 = (-2, 0)
theorem A2023_coords : An 2022 = Point.mk -2 0 := sorry

end A2023_coords_l594_594999


namespace area_of_triangle_tangent_line_l594_594155

-- Define the curve
def curve (x : ℝ) : ℝ := (1/2) * x^2 + x

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (2, 4)

-- Define the area of the triangle formed by the tangent line and the coordinate axes
def triangle_area : ℝ := (2 / 3)

-- Prove that the area of the triangle formed by the tangent to the curve at given point and the coordinate axes is 2/3
theorem area_of_triangle_tangent_line :
  let tangent_slope := 3
  let tangent_eq : ℝ → ℝ := λ x, 3*x - 2  -- y=3x-2
  let x_intercept := 2 / 3
  let y_intercept := -2
  let height := abs y_intercept
  let base := x_intercept
  let area := (1 / 2) * height * base
  in area = triangle_area := by
  sorry

end area_of_triangle_tangent_line_l594_594155


namespace first_term_is_3629_l594_594063

-- Definitions based on the conditions
def a : ℝ := 3629
def r : ℝ := real.cbrt 10

-- Conditions from the problem
def cond1 : Prop := a * r ^ 6 = real.fact 9
def cond2 : Prop := a * r ^ 9 = real.fact 10

theorem first_term_is_3629 (h1 : cond1) (h2 : cond2) : a = 3629 :=
by {
  -- Proof would go here
  sorry
}

end first_term_is_3629_l594_594063


namespace find_ab_find_extremum_point_g_num_zeros_h_l594_594284

section
variables {a b c : ℝ} (f g h : ℝ → ℝ)

-- condition 1: defining f and extremum points
def f := λ x : ℝ, x^3 + a * x^2 + b * x

-- Question 1: Proving values of a and b
theorem find_ab (h1 : f'(1) = 0) (h2 : f'(-1) = 0) :
a = 0 ∧ b = -3 :=
sorry

-- condition 2: derivative of g and solution
def g' := λ x : ℝ, f x + 2

-- Question 2: Extremum points of g
theorem find_extremum_point_g {x : ℝ} (h3 : g'(x) = 0) :
x = -2 :=
sorry

-- condition 3: definition of h and range of c
def h := λ x : ℝ, f (f x) - c

-- Question 3: Number of zeros of h
theorem num_zeros_h (hc : c ∈ Icc (-2 : ℝ) 2) :
(|c| = 2 → (∃ z : ℝ, h z = 0 ∧ (∃ t1 t2 t3 t4 t5 : ℝ, t1 ≠ t2 ∧ t3 ≠ t4 ∧ t4 ≠ t5 ∧ h t1 = 0 ∧ h t2 = 0 ∧ h t3 = 0 ∧ h t4 = 0 ∧ h t5 = 0))) ∧
(|c| < 2 → (∃ z1 z2 z3 z4 z5 z6 z7 z8 z9 : ℝ, h z1 = 0 ∧ h z2 = 0 ∧ h z3 = 0 ∧ h z4 = 0 ∧ h z5 = 0 ∧ h z6 = 0 ∧ h z7 = 0 ∧ h z8 = 0 ∧ h z9 = 0)) :=
sorry

end

end find_ab_find_extremum_point_g_num_zeros_h_l594_594284


namespace cos_double_angle_l594_594239

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 4 / 5) : Real.cos (2 * α) = 7 / 25 := 
by
  sorry

end cos_double_angle_l594_594239


namespace max_fraction_sum_l594_594670

theorem max_fraction_sum (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 2) :
  (\frac{ab}{a + b + 1} + \frac{bc}{b + c + 1} + \frac{ca}{c + a + 1}) ≤ (\frac{2}{3}) :=
sorry

end max_fraction_sum_l594_594670


namespace sum_of_valid_digits_l594_594121

theorem sum_of_valid_digits (A : ℕ) (h₁ : A ∈ set.range (λ i, i) (finset.range 10))
  (h₂ : let N := 62 * 10000 + A * 100 + 94 in (N / 1000) * 1000 + (if (N % 1000) / 100 ≥ 5 then 1000 else 0) = 63000) :
  (finset.sum (finset.filter (λ a, 5 ≤ a ∧ a < 10) (finset.range 10))) = 35 :=
by
  simp only [finset.sum_filter, finset.sum, finset.range, set.mem_range, nat.mem_range]
  sorry

end sum_of_valid_digits_l594_594121


namespace cost_of_each_soda_l594_594651

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l594_594651


namespace total_money_l594_594162

-- Definitions for the conditions
def Cecil_money : ℕ := 600
def twice_Cecil_money : ℕ := 2 * Cecil_money
def Catherine_money : ℕ := twice_Cecil_money - 250
def Carmela_money : ℕ := twice_Cecil_money + 50

-- Theorem statement to prove
theorem total_money : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- sorry is used since no proof is required.
  sorry

end total_money_l594_594162


namespace number_of_true_propositions_l594_594248

theorem number_of_true_propositions :
  let p1 := ∀ (l1 l2 l3 : Line), (Perpendicular l1 l3) ∧ (Perpendicular l2 l3) → Parallel l1 l2
  let p2 := ∀ (l1 l2 : Line) (p : Plane), (Parallel l1 p) ∧ (Parallel l2 p) → Parallel l1 l2
  let p3 := ∀ (l1 l2 : Line), (Parallel l1 l3) ∧ (Parallel l2 l3) → Parallel l1 l2
  let p4 := ∀ (l1 l2 : Line) (pl : Plane), In_plane l1 pl ∧ In_plane l2 pl ∧ ¬Intersect l1 l2 → Parallel l1 l2
  (¬p1) ∧ (¬p2) ∧ p3 ∧ p4 → (number_of_correct_propositions = 2) 
:= by
  sorry

end number_of_true_propositions_l594_594248


namespace sequence_nth_term_l594_594581

theorem sequence_nth_term (n : ℕ) (a_n : ℕ → ℝ) (h : ∀ n, a_n n = real.sqrt (2 * n - 1)) :
  a_n n = real.sqrt 21 → n = 11 := by
  intro h1
  sorry

end sequence_nth_term_l594_594581


namespace total_money_l594_594163

-- Definitions for the conditions
def Cecil_money : ℕ := 600
def twice_Cecil_money : ℕ := 2 * Cecil_money
def Catherine_money : ℕ := twice_Cecil_money - 250
def Carmela_money : ℕ := twice_Cecil_money + 50

-- Theorem statement to prove
theorem total_money : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- sorry is used since no proof is required.
  sorry

end total_money_l594_594163


namespace linda_winning_probability_l594_594291

noncomputable def probability_linda_wins : ℝ :=
  (1 / 16 : ℝ) / (1 - (1 / 32 : ℝ))

theorem linda_winning_probability :
  probability_linda_wins = 2 / 31 :=
sorry

end linda_winning_probability_l594_594291


namespace solve_y_l594_594363

theorem solve_y : ∃ y : ℚ, (1 / 3 + 1 / y = 4 / 5) ∧ (y = 15 / 7) :=
begin
  sorry
end

end solve_y_l594_594363


namespace floor_e_is_two_l594_594882

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end floor_e_is_two_l594_594882


namespace prime_square_minus_five_not_div_by_eight_l594_594699

theorem prime_square_minus_five_not_div_by_eight (p : ℕ) (prime_p : Prime p) (p_gt_two : p > 2) : ¬ (8 ∣ (p^2 - 5)) :=
sorry

end prime_square_minus_five_not_div_by_eight_l594_594699


namespace area_quadrilateral_ABCD_l594_594401

variable (A B C D : Type*) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D]
variables (a b c d : A)
variables (AB BC CD DA: ℝ)
variables (angle_CBA: Real.Angle)

-- Given conditions
def is_quadrilateral_ABCD (A B C D : ℝ) : Prop :=
(AB = 3) ∧ (BC = 4) ∧ (CD = 12) ∧ (DA = 13) ∧ (angle_CBA = Real.pi / 2)

-- Goal
theorem area_quadrilateral_ABCD : is_quadrilateral_ABCD A B C D → (Area ABCD = 36) :=
sorry

end area_quadrilateral_ABCD_l594_594401


namespace fourth_term_correct_l594_594724

noncomputable def fourth_term_arithmetic (x y : ℝ) : ℝ := 
  let a1 := x - y
  let a2 := x + y
  let a3 := x / y
  let common_difference := 2 * y
  a3 + common_difference

theorem fourth_term_correct (x y : ℝ) (h : x ≠ 0) (h_y_pos : y = (√13 - 1) * x / 6) 
  : fourth_term_arithmetic x y = (10 - (2 * √13)/3) / (√13 - 1) := 
by 
  sorry

end fourth_term_correct_l594_594724


namespace angle_between_generator_and_base_plane_l594_594829

theorem angle_between_generator_and_base_plane (R : ℝ) (V_sphere V_cone_segment : ℝ)
  (h : ℝ) (V_c : ℝ) (x : ℝ) (H1 : V_sphere = (4/3) * π * R^3)
  (H2 : h = R * (1 - Math.cos x))
  (H3 : V_c = (π * h^2 * (R - (1/3) * h)))
  (H4 : ((4/3) * π * R^3 - V_c) / V_c = 27 / 5)
  : x = π / 3 := sorry

end angle_between_generator_and_base_plane_l594_594829


namespace sum_of_diagonals_correct_l594_594807

noncomputable def sum_of_diagonals_hexagon (a b c d e f : ℕ) (AB BC CD DE EF FA : ℕ) (x y z : ℕ) : ℕ :=
  x + y + z

theorem sum_of_diagonals_correct :
  ∀(A B C D E F : ℕ) (AB BC CD DE EF FA : ℕ),
  AB = 26 ∧ BC = 73 ∧ CD = 73 ∧ DE = 73 ∧ EF = 58 ∧ FA = 58 →
  ∃ x y z,
  (73 * y + 26 * 73 = x * z) ∧ (x * z + 73^2 = y^2) ∧ (73 * y + 73^2 = z^2) ∧ sum_of_diagonals_hexagon A B C D E F AB BC CD DE EF FA x y z = 352 :=
begin
  intros A B C D E F AB BC CD DE EF FA h,
  sorry
end

end sum_of_diagonals_correct_l594_594807


namespace simplest_form_l594_594768

theorem simplest_form:
  (∀ x y : ℕ, (x > 1 ∧ y > 1 → ¬(x * x | 15 ∧ y * y | 15))) ∧
  (∀ a b : ℕ, (a > 1 ∧ b > 1 → (a * a | 24 ∨ b * b | 24))) ∧
  (∀ p q : ℕ, (p > 1 ∧ q > 1 → (p * p | 7 ∨ q * q | 3))) ∧
  (∀ m n : ℕ, (m * m | 9 ∧ ¬(n * n | 10))) :=
by
  -- Prove that sqrt(15) is already in its simplest form.
  have h_sqrt_15 : ∀ x y : ℕ, (x > 1 ∧ y > 1 → ¬(x * x | 15 ∧ y * y | 15)),
    sorry,
  -- Prove that sqrt(24) can be simplified further.
  have h_sqrt_24 : ∀ a b : ℕ, (a > 1 ∧ b > 1 → (a * a | 24 ∨ b * b | 24)),
    sorry,
  -- Prove that sqrt(7/3) can be simplified further.
  have h_sqrt_7_3 : ∀ p q : ℕ, (p > 1 ∧ q > 1 → (p * p | 7 ∨ q * q | 3)),
    sorry,
  -- Prove that sqrt(0.9) can be simplified further.
  have h_sqrt_0_9 : ∀ m n : ℕ, (m * m | 9 ∧ ¬(n * n | 10)),
    sorry,
  exact ⟨h_sqrt_15, h_sqrt_24, h_sqrt_7_3, h_sqrt_0_9⟩

end simplest_form_l594_594768


namespace inequality_proof_l594_594966

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9 * x * y * z) :
    x / Real.sqrt (x^2 + 2 * y * z + 2) + y / Real.sqrt (y^2 + 2 * z * x + 2) + z / Real.sqrt (z^2 + 2 * x * y + 2) ≥ 1 :=
by
  sorry

end inequality_proof_l594_594966


namespace plane_divided_into_regions_l594_594174

theorem plane_divided_into_regions :
  let line1 (x : ℝ) := 2 * x
  let line2 (x : ℝ) := (1 / 2) * x
  (region_count : ℝ) :=
  4 := by
  sorry

end plane_divided_into_regions_l594_594174


namespace proof_question_1_proof_question_2_l594_594945

-- Conditions 
def sum_of_coeffs_eq_512 (n : ℕ) : Prop := (2^n = 512)

def expansion_term_coeff_x3 (n : ℕ) (r : ℕ) : Prop := 
  (binom n r * 3^(n-r) * (-1)^r * (3*r / 2 - n = 3))

-- Questions
def coefficient_of_term_x3 (n : ℕ) : ℕ := 
  if sum_of_coeffs_eq_512 n then binom n 8 * 3 else 0

def constant_term_expansion (n : ℕ) : ℤ := 
  if n = 9 then 
    (-1)^9 + binom 9 8 * 2
  else 
    0

theorem proof_question_1 (n : ℕ) :
  sum_of_coeffs_eq_512 n -> coefficient_of_term_x3 n = 27 := 
by 
  sorry

theorem proof_question_2 :
  constant_term_expansion 9 = 17 := 
by 
  sorry

end proof_question_1_proof_question_2_l594_594945


namespace quadratic_vertex_transform_l594_594735

theorem quadratic_vertex_transform {p q r m k : ℝ} (h : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 5 * (x + 3)^2 - 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = m * (x - h)^2 + k) →
  h = -3 :=
by
  intros h1 h2
  -- The actual proof goes here
  sorry

end quadratic_vertex_transform_l594_594735


namespace expand_product_l594_594192

theorem expand_product (y : ℝ) : (y + 3) * (y + 7) = y^2 + 10 * y + 21 := by
  sorry

end expand_product_l594_594192


namespace annie_extracurricular_hours_l594_594152

def total_hours_chess_club (total_weeks : Nat) (missed_weeks : Nat) := 2 * (total_weeks - missed_weeks)
def total_hours_drama_club (total_weeks : Nat) (missed_weeks : Nat) (canceled_week : Nat) := 8 * (total_weeks - missed_weeks - 1)
def total_hours_glee_club (odd_weeks : Finset Nat) := 3 * odd_weeks.card
def total_hours_robotics_club (even_weeks : Finset Nat) := 4 * even_weeks.card
def total_hours_soccer_club (odd_weeks : Finset Nat) (even_weeks : Finset Nat) := 1 * odd_weeks.card + 2 * (even_weeks.card - 1) + 1

def odd_weeks := {3, 5, 7} : Finset Nat
def even_weeks := {4, 6, 8} : Finset Nat

theorem annie_extracurricular_hours :
  total_hours_chess_club 8 2 +
  total_hours_drama_club 8 2 1 +
  total_hours_glee_club odd_weeks +
  total_hours_robotics_club even_weeks +
  total_hours_soccer_club odd_weeks even_weeks = 81 := 
by 
  unfold total_hours_chess_club total_hours_drama_club total_hours_glee_club total_hours_robotics_club total_hours_soccer_club odd_weeks even_weeks
  simp
  sorry

end annie_extracurricular_hours_l594_594152


namespace sum_of_integers_abs_lt_2005_l594_594738

theorem sum_of_integers_abs_lt_2005 : 
  (Finset.sum (Finset.filter (fun n => Int.abs n < 2005) (Finset.range (2 * 2005)))) = 0 :=
sorry

end sum_of_integers_abs_lt_2005_l594_594738


namespace survived_trees_difference_l594_594585

theorem survived_trees_difference {original_trees died_trees survived_trees: ℕ} 
  (h1 : original_trees = 13) 
  (h2 : died_trees = 6)
  (h3 : survived_trees = original_trees - died_trees) :
  survived_trees - died_trees = 1 :=
by
  sorry

end survived_trees_difference_l594_594585


namespace mixed_number_power_decimal_equivalent_l594_594432

theorem mixed_number_power_decimal_equivalent (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ)
  (h : a = 3) (h1 : b = 1) (h2 : c = 5) (h3 : d = 3) : 
  (a + b / c)^d = 32.768 := by
  sorry

end mixed_number_power_decimal_equivalent_l594_594432


namespace bisection_method_termination_condition_l594_594087

theorem bisection_method_termination_condition (x1 x2 e : ℝ) (h : e > 0) :
  |x1 - x2| < e → true :=
sorry

end bisection_method_termination_condition_l594_594087


namespace R_ne_S_at_negative_one_l594_594170

def R (x : ℝ) : ℝ := 3 * x ^ 3 - 5 * x ^ 2 + 7 * x - 6
def mean_coeff : ℝ := (-1) / 4 -- -0.25 calculated mean of nonzero coefficients.
def S (x : ℝ) : ℝ := mean_coeff * x ^ 3 + mean_coeff * x ^ 2 + mean_coeff * x + mean_coeff

theorem R_ne_S_at_negative_one : R (-1) ≠ S (-1) := by
  sorry

end R_ne_S_at_negative_one_l594_594170


namespace num_integers_between_700_and_900_with_sum_of_digits_18_l594_594961

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem num_integers_between_700_and_900_with_sum_of_digits_18 : 
  ∃ k, k = 17 ∧ ∀ n, 700 ≤ n ∧ n ≤ 900 ∧ sum_of_digits n = 18 ↔ (1 ≤ k) := 
sorry

end num_integers_between_700_and_900_with_sum_of_digits_18_l594_594961


namespace parabola_difference_eq_l594_594816

variable (a b c : ℝ)

def original_parabola (x : ℝ) : ℝ := a * x^2 + b * x + c
def reflected_parabola (x : ℝ) : ℝ := -(a * x^2 + b * x + c)
def translated_original (x : ℝ) : ℝ := a * x^2 + b * x + c + 3
def translated_reflection (x : ℝ) : ℝ := -(a * x^2 + b * x + c) - 3

theorem parabola_difference_eq (x : ℝ) :
  (translated_original a b c x) - (translated_reflection a b c x) = 2 * a * x^2 + 2 * b * x + 2 * c + 6 :=
by 
  sorry

end parabola_difference_eq_l594_594816


namespace max_intersection_points_l594_594747

theorem max_intersection_points (l1 l2 l3 : set (ℝ × ℝ))
  (h_distinct : l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3)
  (C : set (ℝ × ℝ)) : 
  (∀ l ∈ {l1, l2, l3}, ∃ p : ℝ × ℝ, p ∈ l ∩ C → l ∩ C ≠ ∅ ∧ ∀ (x y : ℝ × ℝ), x ≠ y → {x, y} ∉ l ∩ C) →
  (∀ (l_a l_b : set (ℝ × ℝ)), l_a ≠ l_b → ∃ p : ℝ × ℝ, p ∈ l_a ∩ l_b → l_a ∩ l_b ≠ ∅ ∧ ∀ (x y : ℝ × ℝ), x ≠ y → {x, y} ∉ l_a ∩ l_b) →
  9 :=
by
  sorry

end max_intersection_points_l594_594747


namespace sin_alpha_value_l594_594230

theorem sin_alpha_value (x y : ℝ) (r : ℝ) (h1 : x = 1) (h2 : y = -2) (h3 : r = Real.sqrt 5) :
  Real.sin (Real.atan2 y x) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end sin_alpha_value_l594_594230


namespace floor_abs_of_neg_58_point_7_l594_594876

theorem floor_abs_of_neg_58_point_7 : 
  let x := -58.7 in ⌊|x|⌋ = 58 :=
by
  -- defining x
  let x : ℝ := -58.7
  -- Evaluating
  have h1 : |x| = 58.7 := by
    sorry
  have h2 : ⌊58.7⌋ = 58 := by
    sorry
  rw [h1, h2]

end floor_abs_of_neg_58_point_7_l594_594876


namespace platform_length_is_500_l594_594470

-- Define the length of the train, the time to cross a tree, and the time to cross a platform as given conditions
def train_length := 1500 -- in meters
def time_to_cross_tree := 120 -- in seconds
def time_to_cross_platform := 160 -- in seconds

-- Define the speed based on the train crossing the tree
def train_speed := train_length / time_to_cross_tree -- in meters/second

-- Define the total distance covered when crossing the platform
def total_distance_crossing_platform (platform_length : ℝ) := train_length + platform_length

-- State the main theorem to prove the platform length is 500 meters
theorem platform_length_is_500 (platform_length : ℝ) :
  (train_speed * time_to_cross_platform = total_distance_crossing_platform platform_length) → platform_length = 500 :=
by
  sorry

end platform_length_is_500_l594_594470


namespace a_sufficient_not_necessary_for_a_squared_eq_b_squared_l594_594969

theorem a_sufficient_not_necessary_for_a_squared_eq_b_squared
  (a b : ℝ) :
  (a = b) → (a^2 = b^2) ∧ ¬ ((a^2 = b^2) → (a = b)) :=
  sorry

end a_sufficient_not_necessary_for_a_squared_eq_b_squared_l594_594969


namespace best_fit_model_l594_594992

theorem best_fit_model (R2_1 R2_2 R2_3 R2_4 : ℝ) (h1 : R2_1 = 0.98) (h2 : R2_2 = 0.80) (h3 : R2_3 = 0.50) (h4 : R2_4 = 0.25) :
  R2_1 > R2_2 ∧ R2_1 > R2_3 ∧ R2_1 > R2_4 :=
by {
  rw [h1, h2, h3, h4],
  repeat {split};
  norm_num,
  sorry
}

end best_fit_model_l594_594992


namespace moles_h2o_l594_594536

-- Define the conditions
def nh4no3_moles : ℕ
def naoh_moles : ℕ := 1
def h2o_formed : ℕ := 1

-- Define the theorem we need to prove
theorem moles_h2o (h : naoh_moles = 1) (h_h2o : h2o_formed = 1) : h2o_formed = 1 :=
sorry

end moles_h2o_l594_594536


namespace combined_age_of_staff_l594_594622

/--
In a school, the average age of a class of 50 students is 25 years. 
The average age increased by 2 years when the ages of 5 additional 
staff members, including the teacher, are also taken into account. 
Prove that the combined age of these 5 staff members is 235 years.
-/
theorem combined_age_of_staff 
    (n_students : ℕ) (avg_age_students : ℕ) (n_staff : ℕ) (avg_age_total : ℕ)
    (h1 : n_students = 50) 
    (h2 : avg_age_students = 25) 
    (h3 : n_staff = 5) 
    (h4 : avg_age_total = 27) :
  n_students * avg_age_students + (n_students + n_staff) * avg_age_total - 
  n_students * avg_age_students = 235 :=
by
  sorry

end combined_age_of_staff_l594_594622


namespace directrix_eq_l594_594531

-- Defining the given condition which is the equation of the parabola
def parabola_eq (y : ℝ) : ℝ := -1/8 * y^2

-- Stating the proof problem: The directrix of the parabola defined by the equation parabola_eq is x = 1/2
theorem directrix_eq : ∀ y : ℝ, (∃ x : ℝ, x = 1/2) :=
by
  sorry

end directrix_eq_l594_594531


namespace inequality_true_l594_594275

theorem inequality_true {a b c : ℝ} (h1 : a > b) (h2 : b > 0) : a + c > b + c :=
begin
  calc
    a + c > b + c : add_lt_add_right h1 c,
end

end inequality_true_l594_594275


namespace revenue_percentage_change_l594_594049

theorem revenue_percentage_change (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  let P_new := 1.30 * P
  let S_new := 0.80 * S
  let R := P * S
  let R_new := P_new * S_new
  (R_new - R) / R * 100 = 4 := by
  sorry

end revenue_percentage_change_l594_594049


namespace caffeine_mass_percentage_C_l594_594156

-- conditions
def molar_mass_C : ℝ := 12.01 
def molar_mass_H : ℝ := 1.008
def molar_mass_N : ℝ := 14.01
def molar_mass_O : ℝ := 16.00

def C_atoms : ℕ := 8
def H_atoms : ℕ := 10
def N_atoms : ℕ := 4
def O_atoms : ℕ := 2

-- Computed values
def molar_mass_caffeine : ℝ := 
  (C_atoms * molar_mass_C) + 
  (H_atoms * molar_mass_H) + 
  (N_atoms * molar_mass_N) + 
  (O_atoms * molar_mass_O)

def mass_percentage_C : ℝ :=
  (C_atoms * molar_mass_C) / molar_mass_caffeine * 100

-- proof statement
theorem caffeine_mass_percentage_C : mass_percentage_C = 49.47 :=
  by sorry

end caffeine_mass_percentage_C_l594_594156


namespace intersection_result_l594_594954

def A : Set ℝ := {x | |x - 2| ≤ 2}

def B : Set ℝ := {y | ∃ x ∈ A, y = -2 * x + 2}

def intersection : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_result : (A ∩ B) = intersection :=
by
  sorry

end intersection_result_l594_594954


namespace length_of_EF_l594_594627

noncomputable def EF := 6 -- EF in cm

theorem length_of_EF {A B C D E F : Type*} 
  [CyclicQuadrilateral A B C D] 
  (parallel_AB_CD : Parallel AB CD) 
  (AB_eq : length AB = 6) 
  (CD_eq : length CD = 10)
  (intersect_AC_BD_at_E : Intersect AC BD E)
  (BE_eq : length BE = 3) 
  (EC_eq : length EC = 9) 
  (perp_EF_CD : Perpendicular EF CD) : 
  length EF = 6 :=
sorry

end length_of_EF_l594_594627


namespace solve_inequality_f_range_of_a_l594_594250

def f (x : ℝ) : ℝ := |1 - 2 * x| - |1 + x|

theorem solve_inequality_f (x : ℝ) : f(x) ≥ 4 → (x ≤ -2 ∨ x ≥ 6) :=
sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a^2 + 2 * a + |1 + x| > f(x)) : a < -3 ∨ a > 1 :=
sorry

end solve_inequality_f_range_of_a_l594_594250


namespace find_m_l594_594674

open Nat

theorem find_m (m : ℕ) (h_pos : 0 < m) 
  (a : ℕ := Nat.choose (2 * m) m) 
  (b : ℕ := Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 :=
by
  sorry

end find_m_l594_594674


namespace triangle_ineq_sqrt_triangle_l594_594090

open Real

theorem triangle_ineq_sqrt_triangle (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a):
  (∃ u v w : ℝ, u > 0 ∧ v > 0 ∧ w > 0 ∧ a = v + w ∧ b = u + w ∧ c = u + v) ∧ 
  (sqrt (a * b) + sqrt (b * c) + sqrt (c * a) ≤ a + b + c ∧ a + b + c ≤ 2 * sqrt (a * b) + 2 * sqrt (b * c) + 2 * sqrt (c * a)) :=
  sorry

end triangle_ineq_sqrt_triangle_l594_594090


namespace marble_selection_l594_594483

theorem marble_selection (green yellow purple : ℕ) 
  (h_green : green = 10) (h_yellow : yellow = 8) (h_purple : purple = 12) : 
  ∃ n, n = 15 ∧ ∀ picks : fin n → ℕ, ∃ c, (c = 5 → 
  (multiset.count c (multiset.map picks.val (multiset.replicate 15)) >= 5)) := 
begin
  sorry
end

end marble_selection_l594_594483


namespace first_ellipse_standard_equation_second_ellipse_standard_equation_l594_594898

-- First ellipse proof statement
theorem first_ellipse_standard_equation :
  ∃ (x y : ℝ), x^2 / 9 + y^2 / 5 = 1 :=
begin
  sorry,
end

-- Second ellipse proof statement
theorem second_ellipse_standard_equation :
  ∃ (y x : ℝ), y^2 / 40 + x^2 / 15 = 1 :=
begin
  sorry,
end

end first_ellipse_standard_equation_second_ellipse_standard_equation_l594_594898


namespace second_ball_red_probability_l594_594146

-- Definitions based on given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4
def first_ball_is_red : Prop := true

-- The probability that the second ball drawn is red given the first ball drawn is red
def prob_second_red_given_first_red : ℚ :=
  (red_balls - 1) / (total_balls - 1)

theorem second_ball_red_probability :
  first_ball_is_red → prob_second_red_given_first_red = 5 / 9 :=
by
  intro _
  -- proof goes here
  sorry

end second_ball_red_probability_l594_594146


namespace avg_cost_is_7000_l594_594643

open ProbabilityTheory

noncomputable def avg_cost_of_testing (n m : ℕ) (cost: ℕ) : ℝ := 
  let prob_X_4000 := (2/5) * (1/4)
  let prob_X_6000 := (2/5) * (3/4) * (1/3) + (3/5) * (2/4) * (1/3) + (3/5) * (2/4) * (1/3)
  let prob_X_8000 := 1 - prob_X_4000 - prob_X_6000
  4000 * prob_X_4000 + 6000 * prob_X_6000 + 8000 * prob_X_8000

theorem avg_cost_is_7000 : 
  avg_cost_of_testing 5 2 2000 = 7000 :=
sorry

end avg_cost_is_7000_l594_594643


namespace monomial_coefficient_degree_l594_594148

noncomputable def monomial := -((7: ℝ) * real.pi * (λ x : ℝ, x^3) * (λ y : ℝ, y)) / 6

def coefficient (m : ℝ) := 
  match m with
  | -((a:ℝ) * b * (λ x : ℝ, x^n) * (λ y : ℝ, y^m)) / c => -(a / c) * b
  | _ => 0

def degree (m : ℝ) :=
  match m with
  | -((a:ℝ) * b * (λ x : ℝ, x^n) * (λ y : ℝ, y^m)) / c => n + m
  | _ => 0

theorem monomial_coefficient_degree :
  coefficient monomial = -((7:ℝ) / 6) * real.pi ∧ degree monomial = 4 :=
by
  sorry

end monomial_coefficient_degree_l594_594148


namespace cost_of_each_soda_l594_594648

theorem cost_of_each_soda (total_paid : ℕ) (number_of_sodas : ℕ) (change_received : ℕ) 
  (h1 : total_paid = 20) 
  (h2 : number_of_sodas = 3) 
  (h3 : change_received = 14) : 
  (total_paid - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l594_594648


namespace orthocenters_parallelogram_l594_594328

-- Define the cyclic quadrilateral and the midpoints of its sides
variables {A B C D K L M N H1 H2 H3 H4 : ℝ → ℝ}

-- Conditions for the midpoints
def midpoint_AB (a b k : ℝ → ℝ) : Prop := k = (a + b) / 2
def midpoint_BC (b c l : ℝ → ℝ) : Prop := l = (b + c) / 2
def midpoint_CD (c d m : ℝ → ℝ) : Prop := m = (c + d) / 2
def midpoint_DA (d a n : ℝ → ℝ) : Prop := n = (d + a) / 2

-- Define the orthocenters
def orthocenter_ANK (a b d h1 : ℝ → ℝ) : Prop := h1 = 2 * a + (b + d) / 2
def orthocenter_BKL (b a c h2 : ℝ → ℝ) : Prop := h2 = 2 * b + (a + c) / 2
def orthocenter_CLM (c b d h3 : ℝ → ℝ) : Prop := h3 = 2 * c + (b + d) / 2
def orthocenter_DMN (d a c h4 : ℝ → ℝ) : Prop := h4 = 2 * d + (a + c) / 2

-- Prove the given condition
theorem orthocenters_parallelogram (a b c d : ℝ → ℝ)
[midpoint_AB a b K] [midpoint_BC b c L] [midpoint_CD c d M] [midpoint_DA d a N]
[orthocenter_ANK a b d H1] [orthocenter_BKL b a c H2] [orthocenter_CLM c b d H3] [orthocenter_DMN d a c H4] :
H1 + H3 = H2 + H4 :=
sorry

end orthocenters_parallelogram_l594_594328


namespace number_of_moles_NaCl_l594_594204

def moles_NaOH := 3
def moles_HCl := 3

theorem number_of_moles_NaCl (h_equal : moles_NaOH = moles_HCl) : moles_NaOH = 3 :=
by
  have h1 : moles_HCl = 3 := rfl
  have h2 : moles_NaOH = 3 := by rw [h1, h_equal]
  exact h2

end number_of_moles_NaCl_l594_594204


namespace ivy_collectors_edition_dolls_l594_594869

-- Definitions from the conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def collectors_edition_dolls : ℕ := (2 * ivy_dolls) / 3

-- Assertion
theorem ivy_collectors_edition_dolls : collectors_edition_dolls = 20 := by
  sorry

end ivy_collectors_edition_dolls_l594_594869


namespace sum_of_squares_series_l594_594157

theorem sum_of_squares_series :
  (∑ k in Finset.range 50, (k + 1) ^ 2) + (∑ k in Finset.range 50, (- (k + 1)) ^ 2) = 85850 :=
by
  sorry

end sum_of_squares_series_l594_594157


namespace constant_term_expansion_l594_594068

theorem constant_term_expansion (a : ℝ) (x : ℝ) (sum_of_coeffs : (x + a) * (2 * x - 1 / x) ^ 5 = 2) : 
    constant_term ((x + a) * (2 * x - 1 / x) ^ 5) = 0 :=
by
  sorry

end constant_term_expansion_l594_594068


namespace a_k_bounds_l594_594940

theorem a_k_bounds (n : ℕ) (h_n : 2 ≤ n) (A : ℝ) (hA : 0 < A) (a : Fin n → ℝ)
  (sum_a : ∑ i, a i = A) (sum_sq_a : ∑ i, (a i)^2 = A^2 / (n - 1)) :
  ∀ k, 0 ≤ a k ∧ a k ≤ 2 * A / n := 
begin 
  sorry 
end

end a_k_bounds_l594_594940


namespace odd_primes_mod_32_l594_594329

-- Define the set of odd primes less than 2^5
def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Define the product of all elements in the list
def N : ℕ := odd_primes_less_than_32.foldl (·*·) 1

-- State the theorem
theorem odd_primes_mod_32 :
  N % 32 = 9 :=
sorry

end odd_primes_mod_32_l594_594329


namespace find_k_such_that_distance_AB_l594_594030

theorem find_k_such_that_distance_AB (
  k : ℝ
  (h1 : ∃ A B : ℝ × ℝ, A ≠ B ∧ 
        A.2 = k * (A.1 - 1) ∧ B.2 = k * (B.1 - 1) ∧ 
        A.2 ^ 2 = 4 * A.1 ∧ B.2 ^ 2 = 4 * B.1) 
  (h2 : (A, B : ℝ × ℝ) ∃ (AB_dist : (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = (16 / 3) ^ 2
  ) :
  k = sqrt(3) ∨ k = -sqrt(3) := 
sorry

end find_k_such_that_distance_AB_l594_594030


namespace radius_of_second_circle_l594_594295

theorem radius_of_second_circle
  (a α : ℝ)
  (hα : 0 < α ∧ α < π) :
  let r := (a / 2) * Real.tan (α / 2) in
  let x := (r * (1 - Real.cos α)) / (1 + Real.cos α) in
  x = (a / 2) * (Real.tan (α / 2))^3 :=
by
  sorry

end radius_of_second_circle_l594_594295


namespace floor_e_is_two_l594_594884

noncomputable def e : ℝ := Real.exp 1

theorem floor_e_is_two : ⌊e⌋ = 2 := by
  sorry

end floor_e_is_two_l594_594884


namespace maximum_area_PDE_l594_594554

-- Given the conditions of the problem, we will define the necessary elements in Lean.
variables {A B C D E P : Type}

-- Areas of respective triangles
variables [linear_ordered_field ℝ]
variables [decidable_eq ℝ]

-- Definitions for the areas and related variables
def area_triangle_ABC : ℝ := 1
def area_quadrilateral_BCED := 2 * area_triangle_PBC
def area_triangle_PDE : ℝ := 5 * real.sqrt 2 - 7

-- Proof statement to show maximum area of triangle PDE given conditions
theorem maximum_area_PDE (hABC : area_triangle_ABC = 1)
  (hBCED_PBC : area_quadrilateral_BCED = 2 * area_triangle_PBC)
  (hPDE_max : ∃ λ1 λ2 : ℝ, (λ1 * λ2) ∈ Icc 0 (3 - 2 * real.sqrt 2)
              ∧ 5 * real.sqrt 2 - 7 = - (λ1 * λ2 - 1/2)^2 / 2 + 1/8) : 
  area_triangle_PDE = 5 * real.sqrt 2 - 7 :=
sorry

end maximum_area_PDE_l594_594554


namespace driving_time_constraint_l594_594419

variable (x y z : ℝ)

theorem driving_time_constraint (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) :
  3 + (60 / x) + (90 / y) + (200 / z) ≤ 10 :=
sorry

end driving_time_constraint_l594_594419


namespace evaluate_expression_l594_594524

theorem evaluate_expression (a b c : ℤ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  ((a^2 + b + c)^2 - (a^2 - b - c)^2) = 108 :=
by
  sorry

end evaluate_expression_l594_594524


namespace binomial_product_l594_594888

theorem binomial_product (x : ℝ) : 
  (2 - x^4) * (3 + x^5) = -x^9 - 3 * x^4 + 2 * x^5 + 6 :=
by 
  sorry

end binomial_product_l594_594888


namespace y_decreases_as_x_increases_l594_594632

-- Define the function y = 7 - x
def my_function (x : ℝ) : ℝ := 7 - x

-- Prove that y decreases as x increases
theorem y_decreases_as_x_increases : ∀ x1 x2 : ℝ, x1 < x2 → my_function x1 > my_function x2 := by
  intro x1 x2 h
  unfold my_function
  sorry

end y_decreases_as_x_increases_l594_594632


namespace incorrect_statement_C_l594_594769

theorem incorrect_statement_C :
  ¬ (percentile 60 (sort [4, 3, 2, 6, 5, 8]) = 6) := 
by {
  sorry
}

end incorrect_statement_C_l594_594769


namespace max_min_values_of_f_area_of_triangle_ABC_l594_594244

-- Define the function f(x)
def f (ω x : ℝ) : ℝ := √3 * sin (ω * x) - 2 * (sin (ω * x / 2)) ^ 2

-- Define the interval for part (1)
def interval := set.Icc (-3 * π / 4) π

theorem max_min_values_of_f {ω : ℝ} (hω : ω > 0) (h_per : 2 * π / ω = 3 * π) :
  ∃ x_max x_min : ℝ, x_max ∈ interval ∧ x_min ∈ interval ∧ f ω x_max = 1 ∧ f ω x_min = -√3 - 1 := sorry

-- Define the triangle ABC conditions for part (2)
variables {a b c A B C : ℝ} (h_triangle : 0 < A ∧ A < π / 2)
(h_triangle_acute : 0 < B ∧ B < π / 2)
(h_b : b = 2) (h_fA : f (2 / 3) A = √3 - 1) (h_a_sin : √3 * a = 2 * b * sin A)

theorem area_of_triangle_ABC :
  ∃ S : ℝ, S = (1 / 2) * a * b * sin C ∧ S = (3 + √3) / 3 := sorry

end max_min_values_of_f_area_of_triangle_ABC_l594_594244


namespace sqrt_expr_simplification_l594_594159

theorem sqrt_expr_simplification : 
  sqrt 45 - (sqrt 20 / 2) = 2 * sqrt 5 := 
by
  sorry

end sqrt_expr_simplification_l594_594159


namespace length_fraction_of_radius_l594_594730

noncomputable def side_of_square_area (A : ℕ) : ℕ := Nat.sqrt A
noncomputable def radius_of_circle_from_square_area (A : ℕ) : ℕ := side_of_square_area A

noncomputable def length_of_rectangle_from_area_breadth (A b : ℕ) : ℕ := A / b
noncomputable def fraction_of_radius (len rad : ℕ) : ℚ := len / rad

theorem length_fraction_of_radius 
  (A_square A_rect breadth : ℕ) 
  (h_square_area : A_square = 1296)
  (h_rect_area : A_rect = 360)
  (h_breadth : breadth = 10) : 
  fraction_of_radius 
    (length_of_rectangle_from_area_breadth A_rect breadth)
    (radius_of_circle_from_square_area A_square) = 1 := 
by
  sorry

end length_fraction_of_radius_l594_594730


namespace problem_1_problem_2_l594_594853

theorem problem_1 : 1 - 1^(2023) + Real.sqrt (81) - Real.cbrt (64) = 4 := 
by
  -- conditions
  have h1 : 1^(2023) = 1 := by sorry
  have h2 : Real.sqrt 81 = 9 := by sorry
  have h3 : Real.cbrt 64 = 4 := by sorry
  -- show
  show 1 - 1 + 9 - 4 = 4 from by sorry

theorem problem_2 : Real.sqrt ((-2)^2) + abs (Real.sqrt 2 - Real.sqrt 3) - abs (Real.sqrt 3 - 1) = 3 - Real.sqrt 2 := 
by
  -- conditions
  have h1 : Real.sqrt ((-2)^2) = 2 := by sorry
  have h2 : abs (Real.sqrt 2 - Real.sqrt 3) = Real.sqrt 3 - Real.sqrt 2 := by sorry
  have h3 : abs (Real.sqrt 3 - 1) = Real.sqrt 3 - 1 := by sorry
  -- show
  show 2 + (Real.sqrt 3 - Real.sqrt 2) - (Real.sqrt 3 - 1) = 3 - Real.sqrt 2 from by sorry

end problem_1_problem_2_l594_594853


namespace general_formula_sum_of_sequence_l594_594687

section
variable (a : ℕ → ℝ)

-- Definition: Given condition for the sequence {a_n}
def sequence_condition := ∀ n : ℕ, n ≥ 1 → a 1 + (∑ k in finset.range(n), (2 * k.succ + 1) * a k.succ) = 2 * n

-- Problem 1: General formula for {a_n}
theorem general_formula (h : sequence_condition a):
  ∀ n : ℕ, n ≥ 1 → a n = 2 / (2 * n - 1) :=
sorry

-- Problem 2: Sum of the first n terms of {a_n / (2n + 1)}
theorem sum_of_sequence (h : sequence_condition a):
  ∀ n : ℕ, n ≥ 1 → (∑ k in finset.range(n), (a k.succ / (2 * k.succ + 1))) = (2 * n) / (2 * n + 1) :=
sorry
end

end general_formula_sum_of_sequence_l594_594687


namespace somu_present_age_l594_594712

variable (S F : ℕ)

-- Conditions from the problem
def condition1 : Prop := S = F / 3
def condition2 : Prop := S - 10 = (F - 10) / 5

-- The statement we need to prove
theorem somu_present_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 20 := 
by sorry

end somu_present_age_l594_594712


namespace profit_percentage_l594_594819

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 550) (hSP : SP = 715) : 
  ((SP - CP) / CP) * 100 = 30 := sorry

end profit_percentage_l594_594819


namespace convert_C_to_F_l594_594972

theorem convert_C_to_F (C F : ℝ) (h1 : C = 40) (h2 : C = 5 / 9 * (F - 32)) : F = 104 := 
by
  -- Proof goes here
  sorry

end convert_C_to_F_l594_594972


namespace decaf_percentage_l594_594125

theorem decaf_percentage (initial_stock total_stock : ℕ)
                         (a_percentage_initial a_decaf_initial
                          b_percentage_initial b_decaf_initial
                          c_percentage_initial c_decaf_initial
                          a_percentage_additional a_decaf_additional
                          b_percentage_additional b_decaf_additional
                          c_percentage_additional c_decaf_additional
                          a_weight_initial b_weight_initial c_weight_initial
                          a_weight_additional b_weight_additional c_weight_additional
                          total_coffee_after total_decaf_after : ℝ) :
  initial_stock = 800 →
  total_stock = 1100 →
  a_percentage_initial = 40 → a_decaf_initial = 20 →
  b_percentage_initial = 35 → b_decaf_initial = 50 →
  c_percentage_initial = 25 → c_decaf_initial = 0 →
  a_percentage_additional = 50 → a_decaf_additional = 25 →
  b_percentage_additional = 30 → b_decaf_additional = 45 →
  c_percentage_additional = 20 → c_decaf_additional = 10 →
  a_weight_initial = 320 → b_weight_initial = 280 → c_weight_initial = 200 →
  a_weight_additional = 150 → b_weight_additional = 90 → c_weight_additional = 60 →
  total_coffee_after = initial_stock + 300 →
  total_decaf_after = (a_weight_initial * a_decaf_initial / 100) + 
                      (b_weight_initial * b_decaf_initial / 100) + 
                      (c_weight_initial * c_decaf_initial / 100) + 
                      (a_weight_additional * a_decaf_additional / 100) + 
                      (b_weight_additional * b_decaf_additional / 100) + 
                      (c_weight_additional * c_decaf_additional / 100) →
  (total_decaf_after / total_stock) * 100 ≈ 26.18 := 
sorry

end decaf_percentage_l594_594125


namespace liquid_X_percentage_in_B_l594_594346

noncomputable def percentage_of_solution_B (X_A : ℝ) (w_A w_B total_X : ℝ) : ℝ :=
  let X_B := (total_X - (w_A * (X_A / 100))) / w_B 
  X_B * 100

theorem liquid_X_percentage_in_B :
  percentage_of_solution_B 0.8 500 700 19.92 = 2.274 := by
  sorry

end liquid_X_percentage_in_B_l594_594346


namespace face_value_of_shares_l594_594128

def investment := 4455
def quotedPrice := 8.25
def rateOfDividend := 0.12
def annualIncome := 648

theorem face_value_of_shares :
  let numberOfShares := investment / quotedPrice in
  let dividendPerShare (F : ℝ) := rateOfDividend * F in
  let totalDividend (F : ℝ) := numberOfShares * dividendPerShare F in
  totalDividend 10 = annualIncome :=
by
  sorry

end face_value_of_shares_l594_594128


namespace dave_time_correct_l594_594514

-- Definitions for the given conditions
def chuck_time (dave_time : ℕ) := 5 * dave_time
def erica_time (chuck_time : ℕ) := chuck_time + (3 * chuck_time / 10)
def erica_fixed_time := 65

-- Statement to prove
theorem dave_time_correct : ∃ (dave_time : ℕ), erica_time (chuck_time dave_time) = erica_fixed_time ∧ dave_time = 10 := by
  sorry

end dave_time_correct_l594_594514


namespace problem_correctness_l594_594259

variables {a b c : ℝ}

theorem problem_correctness (h1 : a + b + c = 0) (h2 : ab + c + 1 = 0) (h3 : a = 1) : 
  b^2 - 4c ≥ 0 :=
by
  sorry

end problem_correctness_l594_594259


namespace triangle_side_length_l594_594310

variable {A B C a b c : ℝ}

theorem triangle_side_length (hcosA : cos A = √6 / 3) (hb : b = 2 * √2) (hc : c = √3)
  (h_cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * cos A) : a = √3 := by
  sorry

end triangle_side_length_l594_594310


namespace blue_bordered_area_on_outer_sphere_l594_594827

theorem blue_bordered_area_on_outer_sphere :
  let r := 1 -- cm
  let r1 := 4 -- cm
  let r2 := 6 -- cm
  let A_inner := 27 -- cm^2
  let h := A_inner / (2 * π * r1)
  let A_outer := 2 * π * r2 * h
  A_outer = 60.75 := sorry

end blue_bordered_area_on_outer_sphere_l594_594827


namespace cost_of_plastering_the_tank_is_correct_l594_594830

/-- Definitions -/
def length_of_tank : ℝ := 25
def width_of_tank : ℝ := 12
def depth_of_tank : ℝ := 6
def cost_per_square_meter_in_paise : ℝ := 30
def cost_per_square_meter_in_rupees : ℝ := cost_per_square_meter_in_paise / 100

def surface_area_of_long_walls (length : ℝ) (depth : ℝ) : ℝ := 2 * (length * depth)
def surface_area_of_wide_walls (width : ℝ) (depth : ℝ) : ℝ := 2 * (width * depth)
def surface_area_of_bottom (length : ℝ) (width : ℝ) : ℝ := length * width
def total_surface_area_of_tank (length : ℝ) (width : ℝ) (depth : ℝ) : ℝ :=
  surface_area_of_long_walls length depth + surface_area_of_wide_walls width depth + surface_area_of_bottom length width

noncomputable def total_cost_of_plastering (length : ℝ) (width : ℝ) (depth : ℝ) (cost_per_sq_meter : ℝ) : ℝ :=
  total_surface_area_of_tank length width depth * cost_per_sq_meter

/-- Proof statement -/
theorem cost_of_plastering_the_tank_is_correct :
  total_cost_of_plastering length_of_tank width_of_tank depth_of_tank cost_per_square_meter_in_rupees = 223.2 :=
by
  sorry

end cost_of_plastering_the_tank_is_correct_l594_594830


namespace cos_double_angle_l594_594974

theorem cos_double_angle (θ : ℝ) (h : ∑' n : ℕ, (Real.cos θ)^(2*n) = 8) :
  Real.cos (2 * θ) = 3 / 4 :=
sorry

end cos_double_angle_l594_594974


namespace limit_e2x_ex_tanx2_l594_594849

theorem limit_e2x_ex_tanx2 (f : ℝ → ℝ) (g : ℝ → ℝ) (h : ℝ → ℝ) :
  (∀ x, f x = exp (2 * x)) →
  (∀ x, g x = exp x) →
  (∀ x, h x = tan (x^2)) →
  (filter.tendsto f (nhds 0) (nhds 0)) →
  (filter.tendsto g (nhds 0) (nhds 0)) →
  (filter.tendsto h (nhds 0) (nhds 0)) →
  (filter.tendsto (λ x, (f x - g x) / (x + h x)) (nhds 0) (nhds 1)) := by
  sorry

end limit_e2x_ex_tanx2_l594_594849


namespace kids_staying_home_l594_594327

theorem kids_staying_home (total_kids : ℕ) (kids_camp : ℕ) : 
  total_kids = 898051 → kids_camp = 629424 → total_kids - kids_camp = 268627 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  exact dec_trivial

end kids_staying_home_l594_594327


namespace real_value_iff_imaginary_value_iff_bisector_value_iff_l594_594335

noncomputable def real_value_conditions (m : ℝ) : Prop :=
  let Z := (2 + complex.i) * m^2 - 3 * (1 + complex.i) * m - 2 * (1 - complex.i)
  Z.im = 0

noncomputable def imaginary_value_conditions (m : ℝ) : Prop :=
  let Z := (2 + complex.i) * m^2 - 3 * (1 + complex.i) * m - 2 * (1 - complex.i)
  Z.re = 0 ∧ Z.im ≠ 0

noncomputable def bisector_conditions (m : ℝ) : Prop :=
  let Z := (2 + complex.i) * m^2 - 3 * (1 + complex.i) * m - 2 * (1 - complex.i)
  2 * Z.im = Z.re

theorem real_value_iff (m : ℝ) : real_value_conditions m ↔ m = 1 ∨ m = 2 :=
sorry

theorem imaginary_value_iff (m : ℝ) : imaginary_value_conditions m ↔ m = -1 / 2 :=
sorry

theorem bisector_value_iff (m : ℝ) : bisector_conditions m ↔ m = 2 ∨ m = -2 :=
sorry

end real_value_iff_imaginary_value_iff_bisector_value_iff_l594_594335


namespace n_equals_six_max_value_f_sum_x1_x2_gt_2_l594_594251

-- Define the function f(x)
def f (x : ℝ) (m n : ℝ) : ℝ := (m * x - n) / x - log x

-- (1) Prove n = 6 for the tangent line condition
theorem n_equals_six (m n : ℝ) (h : f (2 : ℝ) m n = 0) 
  (h_tangent : (∂ / ∂ x, f x m n)|_{x = 2} = 1) : n = 6 := 
sorry

-- (2) Prove the maximum value of f(x) on [1, +∞)
theorem max_value_f (m n : ℝ) : 
  ∃ max_value : ℝ, ∀ x ∈ set.Ici 1, f x m n ≤ max_value := 
sorry

-- (3) Prove x1 + x2 > 2 for n = 1 and exactly two zeros
theorem sum_x1_x2_gt_2 (m x1 x2 : ℝ) (h_n : n = 1)
  (h_zeros : f x1 m 1 = 0 ∧ f x2 m 1 = 0) (h_order : 0 < x1 ∧ x1 < x2) : 
  x1 + x2 > 2 := 
sorry

end n_equals_six_max_value_f_sum_x1_x2_gt_2_l594_594251


namespace total_peaches_l594_594077

variable (numberOfBaskets : ℕ)
variable (redPeachesPerBasket : ℕ)
variable (greenPeachesPerBasket : ℕ)

theorem total_peaches (h1 : numberOfBaskets = 1) 
                      (h2 : redPeachesPerBasket = 4)
                      (h3 : greenPeachesPerBasket = 3) :
  numberOfBaskets * (redPeachesPerBasket + greenPeachesPerBasket) = 7 := 
by
  sorry

end total_peaches_l594_594077


namespace find_a_l594_594858

theorem find_a (a : ℚ) :
  let p1 := (3, 4)
  let p2 := (-4, 1)
  let direction_vector := (a, -2)
  let vector_between_points := (p2.1 - p1.1, p2.2 - p1.2)
  ∃ k : ℚ, direction_vector = (k * vector_between_points.1, k * vector_between_points.2) →
  a = -14 / 3 := by
    sorry

end find_a_l594_594858


namespace eleven_y_minus_x_equals_two_l594_594459

-- Defining the conditions as premises
variables {x y : ℕ}

-- Condition: When x is divided by 10, the quotient is y and the remainder is 3
def condition1 : Prop := x = 10 * y + 3

-- Condition: When 2x is divided by 7, the quotient is 3y and the remainder is 1
def condition2 : Prop := 2 * x = 7 * (3 * y) + 1

-- The theorem we need to prove
theorem eleven_y_minus_x_equals_two (h1 : condition1) (h2 : condition2) : 11 * y - x = 2 :=
by
  sorry

end eleven_y_minus_x_equals_two_l594_594459


namespace function_properties_l594_594804

variable {α : Type*} [LinearOrder α]

def f (x : α) : α := sorry

theorem function_properties (h1 : ∀ m n : α, -1 < m → m < 1 → -1 < n → n < 1 → f (m) - f (n) = f ((m - n) / (1 - m * n)))
                         (h2 : ∀ x : α, -1 < x → x < 0 → f (x) < 0) :
                         (∀ x : α, f(x) = - f(-x)) ∧ (f(1/3) + f(1/5) = f(1/2)) :=
by
  sorry

end function_properties_l594_594804


namespace median_of_number_list_3027_5_l594_594093

noncomputable def number_list (n : ℕ) : list ℕ :=
  (list.range (n + 1)).tail ++ (list.range (n + 1)).tail.map (λ x, x * x)

noncomputable def median_of_list (l : list ℕ) : ℝ :=
  let len := l.length in
  if len % 2 = 0 then
    let mid1 := l.nth_le (len / 2 - 1) sorry in
    let mid2 := l.nth_le (len / 2) sorry in
    (mid1 + mid2) / 2
  else
    l.nth_le (len / 2) sorry

theorem median_of_number_list_3027_5 :
  median_of_list (number_list 3030) = 3027.5 :=
sorry

end median_of_number_list_3027_5_l594_594093


namespace quadratic_eq_is_general_form_l594_594729

def quadratic_eq_general_form (x : ℝ) : Prop :=
  x^2 - 2 * (3 * x - 2) + (x + 1) = x^2 - 5 * x + 5

theorem quadratic_eq_is_general_form :
  quadratic_eq_general_form x :=
sorry

end quadratic_eq_is_general_form_l594_594729


namespace a_value_condition1_sinC_value_condition1_area_condition1_a_value_condition2_sinC_value_condition2_area_condition2_l594_594639

-- Definitions under Condition 1
def condition1 (a b c : ℝ) : Prop :=
  c = 7 ∧ cos A = -1/7 ∧ a + b = 11

-- Proving value of a
theorem a_value_condition1 (a b : ℝ) (h : condition1 a b 7) : a = 8 :=
  sorry

-- Proving sin C
theorem sinC_value_condition1 (a b : ℝ) (h : condition1 a b 7) : sin C = sqrt 3 / 2 :=
  sorry

-- Proving area
theorem area_condition1 (a b : ℝ) (h : condition1 a b 7) : area a b 7 = 6 * sqrt 3 :=
  sorry

-- Definitions under Condition 2
def condition2 (a b : ℝ) : Prop :=
  cos A = 1/8 ∧ cos B = 9/16 ∧ a + b = 11

-- Proving value of a
theorem a_value_condition2 (a b : ℝ) (h : condition2 a b) : a = 6 :=
  sorry

-- Proving sin C
theorem sinC_value_condition2 (a b : ℝ) (h : condition2 a b) : sin C = sqrt 7 / 4 :=
  sorry

-- Proving area
theorem area_condition2 (a b : ℝ) (h : condition2 a b) : area a b (a + 5) = 15 * sqrt 7 / 4 :=
  sorry

end a_value_condition1_sinC_value_condition1_area_condition1_a_value_condition2_sinC_value_condition2_area_condition2_l594_594639


namespace part1_l594_594933

theorem part1 (x : ℝ) (P : ℝ × ℝ) (A : ℝ × ℝ) (hA : A = (-3, 4)) (hLine : A.2 = P.2) 
  (hP : P = (2 * x - 3, 3 - x)) : P = (-5, 4) :=
by 
  have hY : 3 - x = 4 := hLine
  have hX : x = -1 := by linarith
  rw [hX, hP]
  linarith

end part1_l594_594933


namespace Sony_caught_4_times_more_l594_594366

/--
Sony and Johnny caught 40 fishes together. Johnny caught 8 fishes, and Sony caught some times as many as Johnny. 
Prove that Sony caught 4 times more fishes than Johnny.
-/
theorem Sony_caught_4_times_more : 
  ∃ (sony_fishes : ℕ), 
    (Johnny_fishes : ℕ) 
    (Total_fishes : ℕ), 
    Johnny_fishes = 8 ∧ 
    Total_fishes = 40 ∧ 
    Total_fishes = sony_fishes + Johnny_fishes ∧ 
    sony_fishes = 4 * Johnny_fishes := 
  by 
    -- No proof required
    sorry

end Sony_caught_4_times_more_l594_594366


namespace no_cubic_integer_coeff_takes_value_pm3_at_4_distinct_primes_l594_594016

theorem no_cubic_integer_coeff_takes_value_pm3_at_4_distinct_primes :
  ∀ (f : ℚ[X]) (p q r s : ℕ), 
    f.degree = 3 ∧ 
    (∀ x, f.coeff x ∈ ℤ) ∧ 
    (prime p ∧ prime q ∧ prime r ∧ prime s) ∧ 
    (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) 
    → ¬(f.eval p = 3 ∧ f.eval q = 3 ∧ f.eval r = 3 ∧ f.eval s = 3) 
      ∧ ¬(f.eval p = -3 ∧ f.eval q = -3 ∧ f.eval r = -3 ∧ f.eval s = -3) 
      ∧ ¬(f.eval p = 3 ∧ f.eval q = 3 ∧ f.eval r = 3 ∧ f.eval s = -3) 
      ∧ ¬(f.eval p = -3 ∧ f.eval q = -3 ∧ f.eval r = -3 ∧ f.eval s = 3)
      ∧ ¬(f.eval p = 3 ∧ f.eval q = 3 ∧ f.eval r = -3 ∧ f.eval s = -3)
      ∧ ¬(f.eval p = -3 ∧ f.eval q = -3 ∧ f.eval r = 3 ∧ f.eval s = 3) := 
by
  sorry

end no_cubic_integer_coeff_takes_value_pm3_at_4_distinct_primes_l594_594016


namespace calc_molecular_weight_8_moles_Al2O3_l594_594094

variables (Al_weight : ℝ) (O_weight : ℝ) (n_Al : ℝ) (n_O : ℝ) (moles : ℝ)
noncomputable def molecular_weight_one_mole_Al2O3 : ℝ :=
  (2 * Al_weight) + (3 * O_weight)

noncomputable def molecular_weight_8_moles_Al2O3 : ℝ :=
  molecular_weight_one_mole_Al2O3 Al_weight O_weight * 8

theorem calc_molecular_weight_8_moles_Al2O3
  (h_Al_weight : Al_weight = 26.98)
  (h_O_weight : O_weight = 16.00)
  : molecular_weight_8_moles_Al2O3 Al_weight O_weight = 815.68 :=
by
  unfold molecular_weight_one_mole_Al2O3 molecular_weight_8_moles_Al2O3
  rw [h_Al_weight, h_O_weight]
  norm_num
  sorry

end calc_molecular_weight_8_moles_Al2O3_l594_594094


namespace spadesuit_calculation_l594_594902

def spadesuit (x y : ℝ) : ℝ := (x + 2 * y) ^ 2 * (x - y)

theorem spadesuit_calculation :
  spadesuit 3 (spadesuit 2 3) = 1046875 :=
by
  sorry

end spadesuit_calculation_l594_594902


namespace race_distance_l594_594991

/-
In a race, the ratio of the speeds of two contestants A and B is 3 : 4.
A has a start of 140 m.
A wins by 20 m.
Prove that the total distance of the race is 360 times the common speed factor.
-/
theorem race_distance (x D : ℕ)
  (ratio_A_B : ∀ (speed_A speed_B : ℕ), speed_A / speed_B = 3 / 4)
  (start_A : ∀ (start : ℕ), start = 140) 
  (win_A : ∀ (margin : ℕ), margin = 20) :
  D = 360 * x := 
sorry

end race_distance_l594_594991


namespace range_of_x_eq_l594_594950

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.log2 (-x) else 2 * x - 1

def range_of_x (x : ℝ) : Prop :=
  f x ≥ 1

theorem range_of_x_eq : { x : ℝ | range_of_x x } = { x : ℝ | x ≤ -2 } ∪ { x : ℝ | x ≥ 1 } :=
by
  sorry

end range_of_x_eq_l594_594950


namespace maximize_angle_ACB_l594_594424

variables {A B C C1 C2 : Type*} {Ω : Type*}
variable [circle Ω]
variables [point A] [point B] [point C] [point C1] [point C2]
variables [incircle Ω A B]

-- Definitions based on the problem conditions
def is_tangency_point (p : point) (ω : circle) : Prop :=
  tangent_to p ω

def on_circle (p : point) (ω : circle) : Prop :=
  on p ω

-- The theorem statement
theorem maximize_angle_ACB (A B : point) (Ω : circle) :
  ∃ (C1 C2 : point), is_tangency_point C1 Ω ∧ is_tangency_point C2 Ω ∧
  (∀ (C : point), on_circle C Ω → (angle A C B ≤ angle A C1 B ∧ angle A C B ≤ angle A C2 B)) := 
sorry

end maximize_angle_ACB_l594_594424


namespace minimal_digits_l594_594506

theorem minimal_digits : 
  ∃ (a b c : ℕ), (a < 10) ∧ (b < 10) ∧ (c < 10) ∧ 
    (∃ k : ℕ, 2007000 + 100 * a + 10 * b + c = 105 * k) ∧ 
    (∀ (a' b' c' : ℕ), (a' < 10) ∧ (b' < 10) ∧ (c' < 10) ∧ 
      (∃ k' : ℕ, 2007000 + 100 * a' + 10 * b' + c' = 105 * k') → 
      2007000 + 100 * a + 10 * b + c ≤ 2007000 + 100 * a' + 10 * b' + c') :=
begin
  -- We know the minimum suffix is 075.
  use [0, 7, 5],
  split,
  { exact nat.lt_succ_self 0 },
  split,
  { exact nat.succ_lt_succ (nat.lt_succ_self 6) },
  split,
  { exact nat.succ_lt_succ (nat.lt_of_succ_eq 4) },
  split,
  { use 19115,
    exact eq.refl _ },
  { intros a' b' c',
    intros ha hb hc h,
    cases h with k' hk',
    suffices : 2007075 ≤ 2007000 + 100 * a' + 10 * b' + c', by assumption,
    rw ← hi,
    sorry, -- We'll skip the proof of the inequality.
  }
end

end minimal_digits_l594_594506


namespace reciprocal_neg4_l594_594058

def reciprocal (x : ℝ) : ℝ :=
  1 / x

theorem reciprocal_neg4 : reciprocal (-4) = -1 / 4 := by
  sorry

end reciprocal_neg4_l594_594058


namespace area_triangle_YVW_l594_594490

-- Define the semicircle with center V and radius r
variables (r : ℝ)

def semicircle_center := (0 : ℝ, 0 : ℝ)   -- V = (0, 0)
def radius := r

-- Define points U, W, and X based on the given conditions
def U := (-r, 0)
def W := (r, 0)
def X := (3 * r, 0)

-- Define point Z where XY is tangent to the semicircle
def Z := (r, r)

-- Triangle Y V W, with Y on the arc centered at X with radius 4r
def Y := (r, 4 * r)

-- Theorem stating the area of triangle Y V W
theorem area_triangle_YVW : 
  area (triangle (Y (r)) (semicircle_center) (W (r))) = (2 / 3) * r^2 :=
sorry

end area_triangle_YVW_l594_594490


namespace sin_cos_alpha_frac_l594_594566

theorem sin_cos_alpha_frac (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 := 
by
  sorry

end sin_cos_alpha_frac_l594_594566


namespace tom_earned_bonus_points_l594_594614

theorem tom_earned_bonus_points 
  (customers_per_hour: ℕ) (hours_worked: ℕ) (bonus_percentage: ℝ)
  (h1: customers_per_hour = 10)
  (h2: hours_worked = 8)
  (h3: bonus_percentage = 0.20) : 
  let total_customers_served := customers_per_hour * hours_worked in
  let bonus_points := bonus_percentage * total_customers_served in
  bonus_points = 16 := 
by
  sorry

end tom_earned_bonus_points_l594_594614


namespace angle_X_I_Y_eq_90_l594_594920

open EuclideanGeometry

variables {A B C D I I_a I_b I_c I_d X Y : Point}

/-- Let A, B, C, D be points of a convex quadrilateral with an inscribed circle centered at I. 
    I_a, I_b, I_c, and I_d are the incenters of triangles DAB, ABC, BCD, and CDA respectively.
    The external common tangents to the circumcircles of ΔA I_b I_d and ΔC I_b I_d intersect at X, 
    and the external common tangents to the circumcircles of ΔB I_a I_c and ΔD I_a I_c intersect at Y.
    Prove that ∠X I Y = 90°. -/
theorem angle_X_I_Y_eq_90 (convex_quad : ConvexQuadrilateral A B C D) 
    (incenter_I : InscribedCircle I A B C D)
    (incenters : incenters I_a I_b I_c I_d A B C D) 
    (tangents_X : external_tangents_circum I_b I_d X A C) 
    (tangents_Y : external_tangents_circum I_a I_c Y B D) : 
  ∠ X I Y = 90° := 
sorry

end angle_X_I_Y_eq_90_l594_594920


namespace tangent_directrix_circle_parabola_l594_594227

theorem tangent_directrix_circle_parabola (p : ℝ) (h : p > 0) :
  let C : ℝ := 3; let r : ℝ := 4;
  circle (x y : ℝ) := x^2 + y^2 - 6*y - 7;
  parabola (x y : ℝ) := x^2 = 2*p*y;
  ∃ (tangent_distance : ℝ), ∀ t : ℝ, abs (t - p / 2) = 4 → tangent_distance = 4 → p = 2 :=
sorry

end tangent_directrix_circle_parabola_l594_594227


namespace dividend_percentage_correct_l594_594480

def face_value : ℝ := 50
def investment_return_percentage : ℝ := 25
def purchase_price : ℝ := 31

def dividend_percentage (face_value : ℝ) (investment_return_percentage : ℝ) (purchase_price : ℝ) : ℝ :=
  let dividend_per_share := (investment_return_percentage / 100) * purchase_price
  (dividend_per_share / face_value) * 100

theorem dividend_percentage_correct :
  dividend_percentage face_value investment_return_percentage purchase_price = 15.5 :=
by
  sorry

end dividend_percentage_correct_l594_594480


namespace solve_sqrt_equation_l594_594181

-- Define the condition as a function
def equation (x : ℝ) : Prop := (Real.sqrt (6 * x - 5)) + (12 / Real.sqrt (6 * x - 5)) = 8

-- Define the solutions to check
def solution1 : ℝ := 41 / 6
def solution2 : ℝ := 3 / 2

-- The main statement to prove
theorem solve_sqrt_equation : (equation solution1) ∨ (equation solution2) := by
  sorry

end solve_sqrt_equation_l594_594181


namespace sum_of_solutions_eq_l594_594437

theorem sum_of_solutions_eq :
  let eqn := (3 * x + 5) * (2 * x - 9) = 0 in
  (∃ x₁ x₂ : ℝ, eqn x₁ ∧ eqn x₂ ∧ x₁ ≠ x₂) →
  ((∃ a b c : ℝ, a * x^2 + b * x + c = 0 ∧ a ≠ 0 ∧ a = 6 ∧ b = -17 ∧ c = -45) →
   (∑ x₀ in { x₁, x₂ }, x₀) = 17 / 6) :=
by
  sorry

end sum_of_solutions_eq_l594_594437


namespace find_a1_l594_594928

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_a1 (d : ℝ) (h1 : a 13 = 13) (h2 : S 13 = 13) : a 0 = -11 :=
by
  sorry

end find_a1_l594_594928


namespace general_position_circles_l594_594644

-- Define the general position concept and the existence of circumcircles passing through a common point
theorem general_position_circles (
  lines: Fin 45 → Line
) (h_general: ∀ i j k : Fin 45, i ≠ j → j ≠ k → i ≠ k → 
  ¬lines i ∥ lines j ∧ ¬(∃ p, ∀ q r, 
      (p = lines q ∩ lines r) → (q ≠ r))) :
  ∃ (O : Point), ∀ (i j k : Fin 45),
    i ≠ j → j ≠ k → i ≠ k →
    (∃ (circ : Circle), 
    has_point O circ ∧ 
    passes_through (lines i ∩ lines j) circ ∧ 
    passes_through (lines j ∩ lines k) circ ∧ 
    passes_through (lines k ∩ lines i) circ) := sorry

end general_position_circles_l594_594644


namespace product_of_consecutive_numbers_with_25_is_perfect_square_l594_594313

theorem product_of_consecutive_numbers_with_25_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n * (n + 1)) + 25 = k^2 := 
by
  -- Proof body omitted
  sorry

end product_of_consecutive_numbers_with_25_is_perfect_square_l594_594313


namespace quadrilateral_condition_l594_594135

variable (a b c d : ℝ)

theorem quadrilateral_condition (h1 : a + b + c + d = 2) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ a + b + c > 1 :=
by
  sorry

end quadrilateral_condition_l594_594135


namespace find_family_ages_l594_594609

theorem find_family_ages :
  ∃ (a b father_age mother_age : ℕ), 
    (a < 21) ∧
    (b < 21) ∧
    (a^3 + b^2 > 1900) ∧
    (a^3 + b^2 < 1978) ∧
    (father_age = 1978 - (a^3 + b^2)) ∧
    (mother_age = father_age - 8) ∧
    (a = 12) ∧
    (b = 14) ∧
    (father_age = 54) ∧
    (mother_age = 46) := 
by 
  use 12, 14, 54, 46
  sorry

end find_family_ages_l594_594609


namespace correct_set_of_equations_l594_594742

-- Define the digits x and y as integers
def digits (x y : ℕ) := x + y = 8

-- Conditions
def condition_1 (x y : ℕ) := 10*y + x + 18 = 10*x + y

theorem correct_set_of_equations : 
  ∃ (x y : ℕ), digits x y ∧ condition_1 x y :=
sorry

end correct_set_of_equations_l594_594742


namespace volume_of_circumscribed_sphere_l594_594232

-- Define the conditions
variables {P A B C : Point}
variables (V_PABC : ℝ) (angle_APC angle_BPC : ℝ)
variables (perpendicular_PA_AC perpendicular_PB_BC perpendicular_PAC_PBC : Prop)

noncomputable def volume_circumsphere (V_PABC : ℝ) (angle_APC angle_BPC : ℝ)
(perpendicular_PA_AC perpendicular_PB_BC perpendicular_PAC_PBC : Prop) : ℝ :=
if (V_PABC = 4 * real.sqrt 3 / 3) ∧ (angle_APC = real.pi / 4) ∧ (angle_BPC = real.pi / 3) 
  ∧ (perpendicular_PA_AC ∧ perpendicular_PB_BC ∧ perpendicular_PAC_PBC) 
then 32 * real.pi / 3 
else 0

theorem volume_of_circumscribed_sphere (h1 : V_PABC = 4 * real.sqrt 3 / 3)
(h2 : angle_APC = real.pi / 4) (h3 : angle_BPC = real.pi / 3)
(h4 : perpendicular_PA_AC) (h5 : perpendicular_PB_BC) (h6 : perpendicular_PAC_PBC) :
volume_circumsphere V_PABC angle_APC angle_BPC perpendicular_PA_AC perpendicular_PB_BC perpendicular_PAC_PBC 
= 32 * real.pi / 3 :=
sorry

end volume_of_circumscribed_sphere_l594_594232


namespace total_tbs_of_coffee_l594_594020

theorem total_tbs_of_coffee (guests : ℕ) (weak_drinkers : ℕ) (medium_drinkers : ℕ) (strong_drinkers : ℕ) 
                           (cups_per_weak_drinker : ℕ) (cups_per_medium_drinker : ℕ) (cups_per_strong_drinker : ℕ) 
                           (tbsp_per_cup_weak : ℕ) (tbsp_per_cup_medium : ℝ) (tbsp_per_cup_strong : ℕ) :
  guests = 18 ∧ 
  weak_drinkers = 6 ∧ 
  medium_drinkers = 6 ∧ 
  strong_drinkers = 6 ∧ 
  cups_per_weak_drinker = 2 ∧ 
  cups_per_medium_drinker = 3 ∧ 
  cups_per_strong_drinker = 1 ∧ 
  tbsp_per_cup_weak = 1 ∧ 
  tbsp_per_cup_medium = 1.5 ∧ 
  tbsp_per_cup_strong = 2 →
  (weak_drinkers * cups_per_weak_drinker * tbsp_per_cup_weak + 
   medium_drinkers * cups_per_medium_drinker * tbsp_per_cup_medium + 
   strong_drinkers * cups_per_strong_drinker * tbsp_per_cup_strong) = 51 :=
by
  sorry

end total_tbs_of_coffee_l594_594020


namespace area_FPG_is_90_l594_594084

-- Given conditions
variable (EF GH: ℝ) (area_trapezoid : ℝ)
variable (isosceles : Prop)

-- Define variables with specific conditions
def trapezoid := EF = 24 ∧ GH = 40 ∧ area_trapezoid = 480 ∧ isosceles

-- Define the problem statement
theorem area_FPG_is_90 (hp : trapezoid EF GH area_trapezoid isosceles):
  ∃ (F P G: Type), area_FPG F P G = 90 := 
sorry

end area_FPG_is_90_l594_594084


namespace height_of_larger_triangle_l594_594515

def side_length_larger_triangle : ℝ := 4

def height_of_equilateral_triangle (s : ℝ) : ℝ :=
  (sqrt 3 / 2) * s

theorem height_of_larger_triangle :
  height_of_equilateral_triangle side_length_larger_triangle = 2 * sqrt 3 :=
by
  -- Proof goes here, but we just add sorry placeholder
  sorry

end height_of_larger_triangle_l594_594515


namespace isomorphic_f2_f4_l594_594426

def f1 (x : ℝ) : ℝ := 2 * real.logb 2 (x + 2)
def f2 (x : ℝ) : ℝ := real.logb 2 (x + 2)
def f3 (x : ℝ) : ℝ := (real.logb 2 (x + 2))^2
def f4 (x : ℝ) : ℝ := real.logb 2 (2 * x)

def is_isomorphic (f g : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f (x - a) = g (x) + b

theorem isomorphic_f2_f4 : is_isomorphic f2 f4 :=
sorry

end isomorphic_f2_f4_l594_594426


namespace sum_puzzles_values_eq_five_l594_594029

-- Define the pattern as a list
def pattern : List Int := [-1, 2, -1, 0, 1, -2, 1, 0]

-- Function to determine the numeric value of a letter given its position in the alphabet
def numericValue (pos : Nat) : Int :=
  pattern.get! (pos % 8)

-- Function to convert a letter to its position in the alphabet (1-indexed)
def letterToPos (letter : Char) : Nat :=
  (letter.toNat - 'a'.toNat + 1)

-- Define the word
def word : List Char := ['p', 'u', 'z', 'z', 'l', 'e', 's']

-- Calculate the sum of the numeric values of the letters in the word "puzzles"
def sumValues : Int :=
  word.map (λ c => numericValue (letterToPos c)).sum
 
theorem sum_puzzles_values_eq_five : sumValues = 5 := by
  -- Implement the proof here
  sorry

end sum_puzzles_values_eq_five_l594_594029


namespace true_propositions_count_l594_594391

theorem true_propositions_count (a : ℝ) :
  (if (a > 2) then (a > 1) else true) ∧ ¬(if (a > 1) then (a > 2) else true) ∧ 
  ¬(if (a ≤ 2) then (a ≤ 1) else true) ∧ (if (a ≤ 1) then (a ≤ 2) else true) → 
  2 :=
sorry

end true_propositions_count_l594_594391


namespace maximize_angle_ACB_l594_594425

variables {A B C C1 C2 : Type*} {Ω : Type*}
variable [circle Ω]
variables [point A] [point B] [point C] [point C1] [point C2]
variables [incircle Ω A B]

-- Definitions based on the problem conditions
def is_tangency_point (p : point) (ω : circle) : Prop :=
  tangent_to p ω

def on_circle (p : point) (ω : circle) : Prop :=
  on p ω

-- The theorem statement
theorem maximize_angle_ACB (A B : point) (Ω : circle) :
  ∃ (C1 C2 : point), is_tangency_point C1 Ω ∧ is_tangency_point C2 Ω ∧
  (∀ (C : point), on_circle C Ω → (angle A C B ≤ angle A C1 B ∧ angle A C B ≤ angle A C2 B)) := 
sorry

end maximize_angle_ACB_l594_594425


namespace euler_totient_divisor_l594_594546

def euler_totient (m : ℕ) : ℕ :=
  (Finset.range m).filter (λ k, Nat.coprime k m).card

theorem euler_totient_divisor (n : ℕ) (hn : 0 < n) :
  (∀ k : ℕ, k > n^2 → n ∣ euler_totient (n * k + 1)) ↔ n = 1 ∨ n = 2 :=
by
  sorry

end euler_totient_divisor_l594_594546


namespace rotated_logarithm_is_exponential_l594_594836

theorem rotated_logarithm_is_exponential (x : ℝ) (hx : x > 0) :
  (∃ y, y = 10^x) ↔ (∃ z, z = log10 x) :=
by
  sorry

end rotated_logarithm_is_exponential_l594_594836


namespace correct_operation_l594_594105

/-- Proving that among the given mathematical operations, only the second option is correct. -/
theorem correct_operation (m : ℝ) : ¬ (m^3 - m^2 = m) ∧ (3 * m^2 * 2 * m^3 = 6 * m^5) ∧ ¬ (3 * m^2 + 2 * m^3 = 5 * m^5) ∧ ¬ ((2 * m^2)^3 = 8 * m^5) :=
by
  -- These are the conditions, proof is omitted using sorry
  sorry

end correct_operation_l594_594105


namespace even_function_zero_coefficient_l594_594978

theorem even_function_zero_coefficient: ∀ a : ℝ, (∀ x : ℝ, (x^2 + a * x + 1) = ((-x)^2 + a * (-x) + 1)) → a = 0 :=
by
  intros a h
  sorry

end even_function_zero_coefficient_l594_594978


namespace drummer_difference_l594_594060

def flute_players : Nat := 5
def trumpet_players : Nat := 3 * flute_players
def trombone_players : Nat := trumpet_players - 8
def clarinet_players : Nat := 2 * flute_players
def french_horn_players : Nat := trombone_players + 3
def total_seats_needed : Nat := 65
def total_seats_taken : Nat := flute_players + trumpet_players + trombone_players + clarinet_players + french_horn_players
def drummers : Nat := total_seats_needed - total_seats_taken

theorem drummer_difference : drummers - trombone_players = 11 := by
  sorry

end drummer_difference_l594_594060


namespace largest_possible_value_of_m_l594_594042

theorem largest_possible_value_of_m : 
  ∃ (qs : List (Polynomial ℝ)), 
    (∀ q ∈ qs, ¬q.is_constant ∧ q.splits (ring_hom.id _)) ∧ 
    (Polynomial.of_real 1).factors = qs ∧ 
    qs.length = 4 :=
by
  sorry

end largest_possible_value_of_m_l594_594042


namespace sin_C_in_right_triangle_l594_594996

theorem sin_C_in_right_triangle 
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : B = π / 2)
  (h3 : sin A = 8 / 17) :
  sin C = 15 / 17 :=
by
  sorry

end sin_C_in_right_triangle_l594_594996


namespace measure_angle_BAC_l594_594797

open Real

-- Definitions of the given angles
noncomputable def angle_AOC : ℝ := 110
noncomputable def angle_AOB : ℝ := 100

-- The theorem to prove
theorem measure_angle_BAC (A B C O : Type) [metric_space O]
    {aoc : angle_AOC = 110} {aob : angle_AOB = 100}
    {circumcircle : circumcircle ABC = O} :
    let angle_BAC := 50 in angle_BAC = 50 :=
by sorry

end measure_angle_BAC_l594_594797


namespace count_integers_with_sum_of_digits_18_l594_594963

def sum_of_digits (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_integer_count : ℕ :=
  let range := List.range' 700 (900 - 700 + 1)
  List.length $ List.filter (λ n => sum_of_digits n = 18) range

theorem count_integers_with_sum_of_digits_18 :
  valid_integer_count = 17 :=
sorry

end count_integers_with_sum_of_digits_18_l594_594963


namespace CDs_fit_per_shelf_l594_594283

def total_shelves : ℕ := 2
def total_CDs : ℝ := 8.0
def CDs_per_shelf (total_CDs : ℝ) (total_shelves : ℕ) : ℝ := total_CDs / total_shelves

theorem CDs_fit_per_shelf : CDs_per_shelf total_CDs total_shelves = 4.0 := by
  sorry

end CDs_fit_per_shelf_l594_594283


namespace minimum_number_of_hexagons_l594_594469

structure Hexagon (a b c r : ℕ) :=
(points : set (ℕ × ℕ × ℕ))
(condition : (∀ x y z, (x, y, z) ∈ points ↔ 
  x + y + z = a + b + c ∧ 
  max (abs (x - a)) (abs (y - b)) (abs (z - c)) ≤ r))

structure S (n : ℕ) :=
(points : set (ℕ × ℕ × ℕ))
(condition : ∀ x y z, (x, y, z) ∈ points ↔ x + y + z = n)

theorem minimum_number_of_hexagons (n : ℕ) (H : ℕ → Hexagon) (s : S n) :
  (∃ k, (∑ i in range k, ((H i).points ∩ s.points).card) = (s.points).card) ∧ k ≥ n + 1 :=
sorry

end minimum_number_of_hexagons_l594_594469


namespace diff_between_roots_l594_594199

theorem diff_between_roots (p : ℝ) (r s : ℝ)
  (h_eq : ∀ x : ℝ, x^2 - (p+1)*x + (p^2 + 2*p - 3)/4 = 0 → x = r ∨ x = s)
  (h_ge : r ≥ s) :
  r - s = Real.sqrt (2*p + 1 - p^2) := by
  sorry

end diff_between_roots_l594_594199


namespace composite_integer_s_l594_594519

theorem composite_integer_s (s : ℕ) (a b c d : ℕ) 
  (h1 : s ≥ 4) (h2 : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h3 : s = a + b + c + d) 
  (h4 : s ∣ (a * b * c + a * b * d + a * c * d + b * c * d)) 
  : ¬ nat.prime s := 
sorry

end composite_integer_s_l594_594519


namespace arc_radius_l594_594460

-- Definitions for conditions
def small_sphere_radius := r : ℝ
def height_difference := h : ℝ
def radius_arc (R r h : ℝ) := R = r + 2 * r^2 / h

-- Theorem statement: 
theorem arc_radius (r h : ℝ) : 
    radius_arc (r + 2 * r^2 / h) r h := 
sorry

end arc_radius_l594_594460


namespace binary_operation_l594_594165

def binary_str_to_nat (s : String) : ℕ :=
  s.foldl (λ acc c, acc * 2 + (if c = '1' then 1 else 0)) 0

def bin_add (a b : String) : String := 
  Nat.toDigits 2 (binary_str_to_nat a + binary_str_to_nat b) |>.reverse |>.asString

def bin_sub (a b : String) : String := 
  Nat.toDigits 2 (binary_str_to_nat a - binary_str_to_nat b) |>.reverse |>.asString

theorem binary_operation :
  bin_sub (bin_add (bin_add "11011" "1010") "1001") "11100" = "100010" :=
by { sorry }

end binary_operation_l594_594165


namespace simplify_and_evaluate_expression_l594_594361

-- Definitions of the variables and their values
def x : ℤ := -2
def y : ℚ := 1 / 2

-- Theorem statement
theorem simplify_and_evaluate_expression : 
  2 * (x^2 * y + x * y^2) - 2 * (x^2 * y - 1) - 3 * x * y^2 - 2 = 
  (1 : ℚ) / 2 :=
by
  -- Proof goes here
  sorry

end simplify_and_evaluate_expression_l594_594361


namespace log8_2000_rounded_l594_594756

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

example : (8 : ℝ) ^ (3 : ℝ) = 512 := by 
  norm_num

example : (8 : ℝ) ^ (4 : ℝ) = 4096 := by 
  norm_num

theorem log8_2000_rounded : 
  3 < log_base 8 2000 ∧ log_base 8 2000 < 4 ∧ Real.abs (2000 - 4096) < Real.abs (2000 - 512) → 
  Int.round (log_base 8 2000) = 4 := by 
  sorry

end log8_2000_rounded_l594_594756


namespace max_books_borrowed_l594_594779

noncomputable def max_books_per_student : ℕ := 14

theorem max_books_borrowed (students_borrowed_0 : ℕ)
                           (students_borrowed_1 : ℕ)
                           (students_borrowed_2 : ℕ)
                           (total_students : ℕ)
                           (average_books : ℕ)
                           (remaining_students_borrowed_at_least_3 : ℕ)
                           (total_books : ℕ)
                           (max_books : ℕ) 
  (h1 : students_borrowed_0 = 2)
  (h2 : students_borrowed_1 = 10)
  (h3 : students_borrowed_2 = 5)
  (h4 : total_students = 20)
  (h5 : average_books = 2)
  (h6 : remaining_students_borrowed_at_least_3 = total_students - students_borrowed_0 - students_borrowed_1 - students_borrowed_2)
  (h7 : total_books = total_students * average_books)
  (h8 : total_books = (students_borrowed_1 * 1 + students_borrowed_2 * 2) + remaining_students_borrowed_at_least_3 * 3 + (max_books - 6))
  (h_max : max_books = max_books_per_student) :
  max_books ≤ max_books_per_student := 
sorry

end max_books_borrowed_l594_594779


namespace problem_statement_l594_594334

noncomputable def AB2_AC2_BC2_eq_4 (l m n k : ℝ) : Prop :=
  let D := (l+k, 0, 0)
  let E := (0, m+k, 0)
  let F := (0, 0, n+k)
  let AB_sq := 4 * (n+k)^2
  let AC_sq := 4 * (m+k)^2
  let BC_sq := 4 * (l+k)^2
  AB_sq + AC_sq + BC_sq = 4 * ((l+k)^2 + (m+k)^2 + (n+k)^2)

theorem problem_statement (l m n k : ℝ) : 
  AB2_AC2_BC2_eq_4 l m n k :=
by
  sorry

end problem_statement_l594_594334


namespace petya_coins_l594_594694

theorem petya_coins
  (coins : List (Fin 2 → ℕ))
  (h1 : ∀ (c : List (Fin 2 → ℕ)), c.length = 3 → ∃ i, c[i] 0 = 1)
  (h2 : ∀ (c : List (Fin 2 → ℕ)), c.length = 4 → ∃ i, c[i] 1 = 2) :
  ∃ a b : ℕ, (a, b).fst = 3 ∧ (a, b).snd = 2 ∧ (coins = List.replicate a (λ _ => 1) ++ List.replicate b (λ _ => 2)) :=
by
  sorry

end petya_coins_l594_594694


namespace shaded_regions_area_l594_594507

/-- Given a grid of 1x1 squares with 2015 shaded regions where boundaries are either:
    - Horizontal line segments
    - Vertical line segments
    - Segments connecting the midpoints of adjacent sides of 1x1 squares
    - Diagonals of 1x1 squares

    Prove that the total area of these 2015 shaded regions is 47.5.
-/
theorem shaded_regions_area (n : ℕ) (h1 : n = 2015) : 
  ∃ (area : ℝ), area = 47.5 :=
by sorry

end shaded_regions_area_l594_594507


namespace evaluate_expression_l594_594525

def numerator : ℤ :=
  (12 - 11) + (10 - 9) + (8 - 7) + (6 - 5) + (4 - 3) + (2 - 1)

def denominator : ℤ :=
  (2 - 3) + (4 - 5) + (6 - 7) + (8 - 9) + (10 - 11) + 12

theorem evaluate_expression : numerator / denominator = 6 / 7 := by
  sorry

end evaluate_expression_l594_594525


namespace find_a₁_find_a_l594_594582

noncomputable def S (n : ℕ) : ℕ :=
n^2 - 2*n + 2

def a (n : ℕ) : ℕ :=
if n = 1 then
  1
else if n ≥ 2 then
  S n - S (n - 1)
else
  0

theorem find_a₁ : a 1 = 1 :=
by
  -- This is where the detailed proof would go
  sorry

theorem find_a (n : ℕ) (h : n ≥ 2) : a n = 2 * n - 3 :=
by
  -- This is where the detailed proof would go
  sorry

end find_a₁_find_a_l594_594582


namespace wheel_revolutions_l594_594086

theorem wheel_revolutions (x y : ℕ) (h1 : y = x + 300)
  (h2 : 10 / (x : ℝ) = 10 / (y : ℝ) + 1 / 60) : 
  x = 300 ∧ y = 600 := 
by sorry

end wheel_revolutions_l594_594086


namespace polar_coordinates_of_point_l594_594177

theorem polar_coordinates_of_point :
  ∀ (x y : ℝ) (r θ : ℝ), x = -1 ∧ y = 1 ∧ r = Real.sqrt (x^2 + y^2) ∧ θ = Real.arctan (y / x) ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi
  → r = Real.sqrt 2 ∧ θ = 3 * Real.pi / 4 := 
by
  intros x y r θ h
  sorry

end polar_coordinates_of_point_l594_594177


namespace expected_value_dodecahedral_die_l594_594758

theorem expected_value_dodecahedral_die : 
  let outcomes := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] in
  let probability := 1 / 12 in
  (∑ i in outcomes, i * probability) = 6.5 :=
by
  sorry

end expected_value_dodecahedral_die_l594_594758


namespace solve_sqrt_eq_l594_594572

theorem solve_sqrt_eq:
  (x : ℝ) (h : sqrt (x - 14) = 2) : x = 18 :=
by
  sorry

end solve_sqrt_eq_l594_594572


namespace greater_number_is_84_l594_594458

theorem greater_number_is_84
  (x y : ℕ)
  (h1 : x * y = 2688)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) : x = 84 :=
sorry

end greater_number_is_84_l594_594458


namespace eval_nabla_l594_594863

def nabla (a b : ℕ) : ℕ := 3 + b^(a-1)

theorem eval_nabla : nabla (nabla 2 3) 4 = 1027 := by
  -- proof goes here
  sorry

end eval_nabla_l594_594863


namespace max_length_segment_l594_594092

theorem max_length_segment (p b : ℝ) (h : b = p / 2) : (b * (p - b)) / p = p / 4 :=
by
  sorry

end max_length_segment_l594_594092


namespace max_angle_ACB_l594_594423

noncomputable def max_angle_condition (Ω : set Point) (A B : Point) : set Point :=
  {C : Point | C ∈ Ω ∧ (∃ ω1 ω2 : Circle, A ∈ ω1 ∧ B ∈ ω1 ∧ ω1 ⊆ Ω ∧ C = tangency_point ω1 Ω ∨
                             A ∈ ω2 ∧ B ∈ ω2 ∧ ω2 ⊆ Ω ∧ C = tangency_point ω2 Ω)}

theorem max_angle_ACB (Ω : set Point) (A B : Point) (hA : A ∈ interior Ω) (hB : B ∈ interior Ω)
  : (∀ C ∈ Ω, ∠ A C B ≤ ∠ A (tangency_point (circle_through A B) Ω) B) :=
sorry

end max_angle_ACB_l594_594423


namespace workman_problem_l594_594808

theorem workman_problem : 
  ∃ x : ℝ, 
    (∀ A B : Type, 
      ((∀ y : ℝ, A = 1/(2*y) → B = 1/y)) ∧
      ((1/(2*x) + 1/x) = 1/13)) → 
    x = 19.5 :=
by 
  sorry

end workman_problem_l594_594808


namespace quad_angle_sum_l594_594301

theorem quad_angle_sum (a b c d x y z : ℝ) (h1 : a + b + c + d = 360) (h2 : d = 360 - (x + y + z)) : 
  a = x → b = y → c = z → d = 360 - x - y - z :=
by
  intro h3 h4 h5
  rw [h3, h4, h5, h2]
  sorry

end quad_angle_sum_l594_594301


namespace area_tripled_radius_l594_594123

noncomputable def original_radius : ℝ := 5

noncomputable def area_ratio (r : ℝ) : ℝ := (Math.pi * (3 * r) ^ 2) / (Math.pi * r ^ 2)

theorem area_tripled_radius (r : ℝ) (h : r = 5) : area_ratio r = 9 := by
  sorry

end area_tripled_radius_l594_594123


namespace expected_rounds_4_l594_594989

def rounds (n : ℕ) : ℝ
| 1 := 0
| 2 := (1 / 3) * (rounds 2 + 1) + (2 / 3) * 1
| 3 := (1 / 3) * (rounds 3 + 1) + (1 / 3) * (rounds 2 + 1) + (1 / 3) * (rounds 1 + 1)
| 4 := (13 / 27) * (rounds 4 + 1) + (4 / 27) * (rounds 3 + 1) + (2 / 9) * (rounds 2 + 1) + (4 / 27) * (rounds 1 + 1)
| _ := sorry

theorem expected_rounds_4 : rounds 4 = 45 / 14 := sorry

end expected_rounds_4_l594_594989


namespace reciprocal_neg4_l594_594057

def reciprocal (x : ℝ) : ℝ :=
  1 / x

theorem reciprocal_neg4 : reciprocal (-4) = -1 / 4 := by
  sorry

end reciprocal_neg4_l594_594057


namespace triangle_geom_seq_l594_594304

variables (A B C : Type) [Field A] [Field B] [Field C]
variables (a b c k : A) (h_geom_seq : a * 4 = b * 2) (cosB : B) (h_cosB : cosB = 1 / 3)
variable (h_frac : a / c = 1 / 2)

theorem triangle_geom_seq (h_geom_seq : a * 4 = b * 2) (h_cosB : cosB = 1 / 3) (h_frac : a / c = 1 / 2) : a + c = 5 * (c / 4) :=
begin
  -- steps of proof
  sorry
end

end triangle_geom_seq_l594_594304


namespace contradiction_proof_l594_594443

theorem contradiction_proof (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end contradiction_proof_l594_594443


namespace range_of_x_l594_594979

theorem range_of_x (x : ℝ) :
  (∀ y : ℝ, 0 < y → y^2 + (2*x - 5)*y - x^2 * (Real.log x - Real.log y) ≤ 0) ↔ x = 5 / 2 :=
by 
  sorry

end range_of_x_l594_594979


namespace work_completion_l594_594448

theorem work_completion (a b : ℕ) (h1 : a + b = 5) (h2 : a = 10) : b = 10 := by
  sorry

end work_completion_l594_594448


namespace identity_sum_l594_594012

open Nat

theorem identity_sum (n : ℕ) (h : n > 0) :
  (∑ k in range n, 1 / ((2 * k + 1) * (2 * k + 2))) = 
  (∑ k in range n, 1 / (n + k + 1)) := 
sorry

end identity_sum_l594_594012


namespace min_value_l594_594936

theorem min_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : 
  ∃ min_val, min_val = 5 + 2 * Real.sqrt 6 ∧ (∀ x, (x = 5 + 2 * Real.sqrt 6) → x ≥ min_val) :=
by
  sorry

end min_value_l594_594936


namespace pyramid_surface_area_l594_594133

noncomputable def total_surface_area_of_pyramid (s h : ℝ) (side_length : ℝ := 8) (height : ℝ := 15) : ℝ :=
  let r := s / (real.sqrt (3 - real.sqrt 3)) -- distance from center of hexagon to any vertex
  let slant_height := real.sqrt (height^2 + r^2) -- slant height of the pyramid
  let triangular_face_area := (1 / 2) * side_length * slant_height
  let total_triangular_area := 6 * triangular_face_area
  let hexagon_area := (3 * real.sqrt 3 / 2) * side_length^2 -- area of the hexagon base
  total_triangular_area + hexagon_area -- total surface area of the pyramid

theorem pyramid_surface_area :
    total_surface_area_of_pyramid 8 15 = 586.52 :=
by
  sorry

end pyramid_surface_area_l594_594133


namespace interest_rate_proof_l594_594126
noncomputable def interest_rate_B (P : ℝ) (rA : ℝ) (t : ℝ) (gain_B : ℝ) : ℝ := 
  (P * rA * t + gain_B) / (P * t)

theorem interest_rate_proof
  (P : ℝ := 3500)
  (rA : ℝ := 0.10)
  (t : ℝ := 3)
  (gain_B : ℝ := 210) :
  interest_rate_B P rA t gain_B = 0.12 :=
sorry

end interest_rate_proof_l594_594126


namespace center_element_arith_seq_matrix_l594_594610

theorem center_element_arith_seq_matrix :
  ∀ (M : matrix (fin 4) (fin 4) ℕ),
  (∀ i, is_arith_seq (fun j => M i j)) ∧
  (∀ j, is_arith_seq (fun i => M i j)) ∧
  M 0 0 = 12 ∧
  M 3 3 = 72 →
  M 1 2 = 52 :=
by
  intros M h
  sorry

-- Helper definition to define arithmetic sequence
def is_arith_seq (f : ℕ → ℕ) : Prop :=
  ∃ d a, ∀ n, f n = a + d * n

end center_element_arith_seq_matrix_l594_594610


namespace calculate_fff2_l594_594679

def f (x : ℤ) : ℤ := x^2 + x - 6

theorem calculate_fff2 : f(f(f(2))) = 24 := 
by 
  sorry

end calculate_fff2_l594_594679


namespace tan_alpha_eq_one_then_expr_value_l594_594273

theorem tan_alpha_eq_one_then_expr_value (α : ℝ) (h : Real.tan α = 1) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 2 / 3 :=
by
  sorry

end tan_alpha_eq_one_then_expr_value_l594_594273


namespace Toms_out_of_pocket_cost_l594_594752

theorem Toms_out_of_pocket_cost (visit_cost cast_cost insurance_percent : ℝ) 
  (h1 : visit_cost = 300) 
  (h2 : cast_cost = 200) 
  (h3 : insurance_percent = 0.6) : 
  (visit_cost + cast_cost) - ((visit_cost + cast_cost) * insurance_percent) = 200 :=
by
  sorry

end Toms_out_of_pocket_cost_l594_594752


namespace exist_sequence_with_strictly_decreasing_gcds_l594_594187

theorem exist_sequence_with_strictly_decreasing_gcds :
  ∃ (a : Fin 100 → ℕ),
    (∀ i j, i < j → a i < a j) ∧
    (∀ k : Fin 99, gcd (a k) (a ⟨k + 1, sorry⟩) > gcd (a ⟨k + 1, sorry⟩) (a ⟨k + 2, sorry⟩)) :=
begin
  sorry
end

end exist_sequence_with_strictly_decreasing_gcds_l594_594187


namespace area_of_triangle_formed_by_tangent_line_l594_594713

-- Definitions and conditions from the problem
def curve (x : ℝ) : ℝ := (1 / 4) * x^2
def tangent_slope_at_point (x : ℝ) : ℝ := (1 / 2) * x

def tangent_line (x : ℝ) : ℝ := x - 2

theorem area_of_triangle_formed_by_tangent_line :
  let intercept_x := 1 -- x-intercept
  let intercept_y := -1 -- y-intercept
  let area := (1 / 2) * (intercept_x * intercept_y).abs
  area = 1 / 2 :=
by 
  let intercept_x := 1
  let intercept_y := -1
  let area := (1 / 2) * (intercept_x * intercept_y).abs
  have h1: (intercept_x * intercept_y).abs = 1 := by sorry
  rw [h1, mul_one, mul_div_cancel'] 
  exact one_ne_zero
  sorry

end area_of_triangle_formed_by_tangent_line_l594_594713


namespace extreme_points_range_l594_594574

noncomputable def f (a : ℝ) :=
λ x : ℝ, if x > 0 then x * Real.log x - a * x^2 else x^2 + a * x

theorem extreme_points_range (a : ℝ) :
  (∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    (∀ x, ∃ (d : ℝ), Deriv f a x = 0 ∧ 
      (Deriv f a d > 0 → d < x) ∧ 
      (Deriv f a d < 0 → d > x))) ↔ (0 < a ∧ a < 1 / 2) := 
sorry

end extreme_points_range_l594_594574


namespace xy_plus_one_ge_four_l594_594340

theorem xy_plus_one_ge_four {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 1) : 
  (x + 1) * (y + 1) >= 4 ∧ ((x + 1) * (y + 1) = 4 ↔ x = 1 ∧ y = 1) :=
by
  sorry

end xy_plus_one_ge_four_l594_594340


namespace hyperbola_vertex_distance_l594_594894

theorem hyperbola_vertex_distance :
  (∀ a b x y : ℝ, a^2 = 121 ∧ b^2 = 49 ∧ (x^2 / a^2) - (y^2 / b^2) = 1) →
  (distance = 2 * real.sqrt 121) :=
by
  sorry

end hyperbola_vertex_distance_l594_594894


namespace union_of_A_and_B_l594_594564

-- Definitions of sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 1}

-- The theorem we aim to prove
theorem union_of_A_and_B : A ∪ B = { x | -3 ≤ x ∧ x ≤ 4 } :=
sorry

end union_of_A_and_B_l594_594564


namespace rectangle_shaded_fraction_l594_594695

theorem rectangle_shaded_fraction {l w : ℝ} (hl : 0 < l) (hw : 0 < w) :
  let P := (0, w / 2)
  let Q := (l / 2, w)
  let A_rect := l * w
  let A_unshaded := (1 / 2) * (l / 2) * (w / 2)
  let A_shaded := A_rect - A_unshaded
  let fraction_shaded := A_shaded / A_rect
  in fraction_shaded = (7 / 8) :=
by
  sorry

end rectangle_shaded_fraction_l594_594695


namespace sum_of_last_two_greatest_two_digit_multiples_of_17_l594_594762

theorem sum_of_last_two_greatest_two_digit_multiples_of_17: 
  (∃ a b : ℕ, (a = 4 ∧ b = 5 ∧ a * 17 < 100 ∧ b * 17 < 100 ∧ 100 ≤ (a+1) * 17 ∧ 100 ≤ (b+1) * 17) ∧ (a * 17 + b * 17 = 153)) := 
begin 
  use [4, 5], 
  split; 
  try {split}; 
  try {norm_num},
  sorry
end

end sum_of_last_two_greatest_two_digit_multiples_of_17_l594_594762


namespace tom_bonus_points_l594_594612

theorem tom_bonus_points (customers_per_hour : ℕ) (hours_worked : ℕ) : 
  customers_per_hour = 10 → 
  hours_worked = 8 → 
  (customers_per_hour * hours_worked * 20) / 100 = 16 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  norm_num
  sorry

end tom_bonus_points_l594_594612


namespace who_drank_most_l594_594874

theorem who_drank_most (eunji yujeong yuna : ℝ) 
    (h1 : eunji = 0.5) 
    (h2 : yujeong = 7 / 10) 
    (h3 : yuna = 6 / 10) :
    max (max eunji yujeong) yuna = yujeong :=
by {
    sorry
}

end who_drank_most_l594_594874


namespace evaluate_expression_l594_594190

def a := 150
def b := a - 4
def c := a * a
def d := 2

theorem evaluate_expression : a * b - d * (c - 4) = -23092 :=
by
  sorry

end evaluate_expression_l594_594190


namespace sqrt_2_plus_x_nonnegative_l594_594025

theorem sqrt_2_plus_x_nonnegative (x : ℝ) : (2 + x ≥ 0) → (x ≥ -2) :=
by
  sorry

end sqrt_2_plus_x_nonnegative_l594_594025


namespace parabola_c_value_l594_594715

theorem parabola_c_value :
  ∃ a b c : ℝ, (∀ y : ℝ, 4 = a * (3 : ℝ)^2 + b * 3 + c ∧ 2 = a * 5^2 + b * 5 + c ∧ c = -1 / 2) :=
by
  sorry

end parabola_c_value_l594_594715


namespace regular_hexagon_perimeter_is_30_l594_594349

-- Define a regular hexagon with each side length 5 cm
def regular_hexagon_side_length : ℝ := 5

-- Define the perimeter of a regular hexagon
def regular_hexagon_perimeter (side_length : ℝ) : ℝ := 6 * side_length

-- State the theorem about the perimeter of a regular hexagon with side length 5 cm
theorem regular_hexagon_perimeter_is_30 : regular_hexagon_perimeter regular_hexagon_side_length = 30 := 
by 
  sorry

end regular_hexagon_perimeter_is_30_l594_594349


namespace number_of_valid_arrangements_l594_594802

-- Definitions based on the conditions
/-- Crops are represented by an enumerated type. -/
inductive Crop
| Lettuce | Carrot | Tomato | Radish

/-- The field is represented as a 2x2 grid of Crops. -/
structure Field :=
(c00 c01 c10 c11 : Crop)

def adjacent (f : Field) (c1 c2 : Crop) : Prop :=
  match f with
  | ⟨c00, c01, c10, c11⟩ => 
    (c00 = c1 ∧ (c01 = c2 ∨ c10 = c2)) ∨ 
    (c01 = c1 ∧ (c00 = c2 ∨ c11 = c2)) ∨ 
    (c10 = c1 ∧ (c00 = c2 ∨ c11 = c2)) ∨ 
    (c11 = c1 ∧ (c01 = c2 ∨ c10 = c2))

def validField (f : Field) : Prop :=
  ¬ adjacent f Crop.Lettuce Crop.Carrot ∧
  ¬ adjacent f Crop.Tomato Crop.Radish ∧
  (f.c00 = f.c11 ∨ f.c01 = f.c10)

theorem number_of_valid_arrangements : 
  ∃ (n : ℕ), validField ∧ n = 8 :=
sorry

end number_of_valid_arrangements_l594_594802


namespace truck_distance_l594_594495

noncomputable def distance_from_starting_point : ℝ :=
let north1 : ℝ := 20
let east : ℝ := 30
let north2 : ℝ := 20
let southwest : ℝ := 10
let west : ℝ := 15
let southeast : ℝ := 25 in
let north_total : ℝ := north1 + north2 - (southwest * real.sin (real.pi / 4))
let east_total : ℝ := east - (southwest * real.sin (real.pi / 4)) - west + (southeast * real.cos (real.pi / 4)) in
real.sqrt (north_total^2 + east_total^2)

theorem truck_distance : distance_from_starting_point ≈ 41.72 :=
by
-- Proof would be placed here
sorry

end truck_distance_l594_594495


namespace sum_of_roots_l594_594592

theorem sum_of_roots : ∀ x : ℝ, ((x + 3) * (x - 4) = 20) → (∃ a b : ℝ, a = x + 3 ∧ b = x - 4 ∧ a * b = 20 ∧ (∑ x in {a, b}, x) = 1) :=
begin
  intros x h,
  sorry
end

end sum_of_roots_l594_594592


namespace total_money_is_2800_l594_594160

-- Define variables for money
def Cecil_money : ℕ := 600
def Catherine_money : ℕ := 2 * Cecil_money - 250
def Carmela_money : ℕ := 2 * Cecil_money + 50

-- Assertion to prove the total money 
theorem total_money_is_2800 : Cecil_money + Catherine_money + Carmela_money = 2800 :=
by
  -- placeholder proof
  sorry

end total_money_is_2800_l594_594160


namespace range_of_m_l594_594213

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (mx-1)*(x-2) > 0 ↔ (1/m < x ∧ x < 2)) → m < 0 :=
by
  sorry

end range_of_m_l594_594213


namespace average_score_10_students_l594_594598

theorem average_score_10_students (x : ℝ)
  (h1 : 15 * 70 = 1050)
  (h2 : 25 * 78 = 1950)
  (h3 : 15 * 70 + 10 * x = 25 * 78) :
  x = 90 :=
sorry

end average_score_10_students_l594_594598


namespace number_of_ways_to_pick_book_l594_594794

theorem number_of_ways_to_pick_book (E M : ℕ) (hE : E = 6) (hM : M = 2) : E + M = 8 :=
by
  rw [hE, hM]
  exact rfl

end number_of_ways_to_pick_book_l594_594794


namespace evaluate_floor_abs_l594_594878

theorem evaluate_floor_abs : ⌊|(-58.7 : ℝ)|⌋ = 58 := 
begin
  sorry
end

end evaluate_floor_abs_l594_594878


namespace trinomial_square_value_l594_594851

theorem trinomial_square_value (a b c : ℕ) 
  (a_val : a = 13) (b_val : b = 5) (c_val : c = 3) :
  a^2 + b^2 + c^2 + 2 * a * b + 2 * a * c + 2 * b * c = 441 :=
by
  rw [a_val, b_val, c_val]
  simp
  norm_num
  sorry

end trinomial_square_value_l594_594851


namespace ratio_AC_RS_l594_594923

open Real

variables {A B C D M N R S : Point} {m : ℝ}

-- Definitions for the conditions in the problem
def is_parallelogram (A B C D : Point) : Prop := 
  parallel (line A B) (line C D) ∧ parallel (line A D) (line B C) ∧ 
  ¬ collinear A B C ∧ ¬ collinear A D C

def point_on_line_segment (P Q : Point) (t : ℝ) (R : Point) : Prop :=
  0 ≤ t ∧ t ≤ 1 ∧ R = (1 - t) • P + t • Q

def intersects (A B C D : Point) (M : Point) : Prop :=
  ∃ t : ℝ, point_on_line_segment A B t M

def ratio_side (A B M : Point) (m : ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ m = t / (1 - t)

-- Stating the proof problem in Lean
theorem ratio_AC_RS (parallelogram : is_parallelogram A B C D)
  (hM : intersects A B M) (hN : intersects C D N)
  (ratioM : ratio_side A B M m) (ratioN : ratio_side C D N m)
  (intersection_R : R = line_intersection (line A C) (line D M))
  (intersection_S : S = line_intersection (line A C) (line B N)) :
  AC / RS = 2 * m + 1 :=
sorry

end ratio_AC_RS_l594_594923


namespace correct_option_d_l594_594103

-- Define variables and constants.
variable (a : ℝ)

-- State the conditions as definitions.
def optionA := a^2 * a^3 = a^5
def optionB := (3 * a)^2 = 9 * a^2
def optionC := a^6 / a^3 = a^3
def optionD := 3 * a^2 - a^2 = 2 * a^2

-- The theorem states that the correct option is D.
theorem correct_option_d : optionD := by
  sorry

end correct_option_d_l594_594103


namespace four_isosceles_not_congruent_from_square_four_isosceles_not_congruent_from_triangle_l594_594777

-- Part (a): Prove that the triangles formed by cutting a square into 4 isosceles triangles are not congruent
theorem four_isosceles_not_congruent_from_square (A B C D K E : Type) 
  (h_square: is_square A B C D)
  (h_diagonal: is_diagonal A C)
  (h_midpoint: is_midpoint K A C)
  (h_reflection: reflects B K E) :
  (is_isosceles ABK) ∧ (is_isosceles BKC) ∧ (is_isosceles DAE) ∧ (is_isosceles EDC) ∧
  (¬congruent ABK BKC) ∧ (¬congruent ABK DAE) ∧ (¬congruent ABK EDC) ∧ 
  (¬congruent BKC DAE) ∧ (¬congruent BKC EDC) ∧ (¬congruent DAE EDC) := sorry

-- Part (b): Prove that the triangles formed by cutting an equilateral triangle into 4 isosceles triangles are not congruent
theorem four_isosceles_not_congruent_from_triangle (A B C D E F : Type)
  (h_equilateral: is_equilateral A B C)
  (h_on_sideBC: on_side D B C)
  (h_on_sideAC: on_side E A C)
  (h_on_sideAB: on_side F A B) :
  (is_isosceles ADF) ∧ (is_isosceles DEF) ∧ (is_isosceles EFB) ∧ (is_isosceles FCD) ∧
  (¬congruent ADF DEF) ∧ (¬congruent ADF EFB) ∧ (¬congruent ADF FCD) ∧
  (¬congruent DEF EFB) ∧ (¬congruent DEF FCD) ∧ (¬congruent EFB FCD) := sorry

end four_isosceles_not_congruent_from_square_four_isosceles_not_congruent_from_triangle_l594_594777


namespace function_range_l594_594915

theorem function_range (f : ℝ → ℝ) (h_f : ∀ x ∈ ℝ, f(a - x) + f(a*x^2 - 1) < 0) : 
  (a < (1 - real.sqrt 2) / 2) :=
by
  sorry

end function_range_l594_594915


namespace probability_blue_tile_l594_594472

-- Define the set of tiles and the property of being blue
def tiles : Finset ℕ := (Finset.range 50).map (Finset.embedding (Fin.val ∘ (λ x, x + 1)))

def is_blue (n : ℕ) : Prop := n % 4 = 1

-- Define a proof goal for the probability that a randomly chosen tile is blue
theorem probability_blue_tile :
  (Finset.filter is_blue tiles).card / tiles.card = 13 / 50 :=
by
  -- filtered set should be constructed from the tiles matching the blue tile property
  let blue_tiles := Finset.filter is_blue tiles
  have cards_match : blue_tiles.card = 13 := by sorry
  have total_tiles : tiles.card = 50 := by sorry
  -- Given these two facts, the ratio of blue tiles to total tiles is 13 / 50
  rw [cards_match, total_tiles]
  norm_num
  sorry

end probability_blue_tile_l594_594472


namespace magnitude_sum_of_vectors_l594_594956

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def O : Point3D := { x := 0, y := 0, z := 0 }
def P1 : Point3D := { x := 1, y := 1, z := 0 }
def P2 : Point3D := { x := 0, y := 1, z := 1 }
def P3 : Point3D := { x := 1, y := 0, z := 1 }

def vector_add (u v : Point3D) : Point3D :=
  { x := u.x + v.x, y := u.y + v.y, z := u.z + v.z }

def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x * v.x + v.y * v.y + v.z * v.z)

theorem magnitude_sum_of_vectors : magnitude (vector_add (vector_add (P1) (P2)) (P3)) = 2 * Real.sqrt 3 := 
  by
    -- proof goes here
    sorry

end magnitude_sum_of_vectors_l594_594956


namespace associates_hired_l594_594803

variable (partners : ℕ) (associates initial_associates hired_associates : ℕ)
variable (initial_ratio : partners / initial_associates = 2 / 63)
variable (final_ratio : partners / (initial_associates + hired_associates) = 1 / 34)
variable (partners_count : partners = 18)

theorem associates_hired : hired_associates = 45 :=
by
  -- Insert solution steps here...
  sorry

end associates_hired_l594_594803


namespace min_regions_l594_594705

namespace CircleDivision

def k := 12

-- Theorem statement: Given exactly 12 points where at least two circles intersect,
-- the minimum number of regions into which these circles divide the plane is 14.
theorem min_regions (k := 12) : ∃ R, R = 14 :=
by
  let R := 14
  existsi R
  exact rfl

end min_regions_l594_594705


namespace perfect_square_expression_l594_594722
open Real

theorem perfect_square_expression (x : ℝ) :
  (12.86 * 12.86 + 12.86 * x + 0.14 * 0.14 = (12.86 + 0.14)^2) → x = 0.28 :=
by
  sorry

end perfect_square_expression_l594_594722


namespace solve_for_y_l594_594763

theorem solve_for_y (y : ℝ) (h : 3 / y + 4 / y / (6 / y) = 1.5) : y = 3.6 :=
sorry

end solve_for_y_l594_594763


namespace dot_product_a_b_norm_a_sub_2b_l594_594262

variables {a b : ℝ} {a_vec b_vec : ℝ^2}

-- Given conditions
def condition_1 : norm a_vec = 4 := sorry
def condition_2 : norm b_vec = 3 := sorry
def condition_3 : (2 • a_vec - 3 • b_vec) ⬝ (2 • a_vec + b_vec) = 61 := sorry

-- Question 1: Prove a · b = -6
theorem dot_product_a_b : a_vec ⬝ b_vec = -6 :=
by {
  sorry
}

-- Question 2: Prove |a - 2b| = 2\sqrt{19}
theorem norm_a_sub_2b : norm (a_vec - 2 • b_vec) = 2 * real.sqrt 19 :=
by {
  sorry
}

end dot_product_a_b_norm_a_sub_2b_l594_594262


namespace find_min_value_l594_594201

noncomputable def f (a x : ℝ) := x^2 + 2*a*x + 3

theorem find_min_value (a : ℝ) :
  (1 ≤ 2) →
  (min_value a (λ x, f a x) 1 2) = 
    if a ≥ -1 then 2*a + 4
    else if (-2 < a ∧ a < -1) then 3 - a^2
    else 4*a + 7 :=
by
  intros h
  sorry

end find_min_value_l594_594201


namespace find_pairs_l594_594892

variable {Nat : Type}
variable (a n m x y : Nat)

theorem find_pairs (a n m x y : Nat) (h1 : x + y = a^n) (h2 : x^2 + y^2 = a^m) :
  ∃ k : Nat, x = 2^k ∧ y = 2^k := by
  sorry

end find_pairs_l594_594892


namespace people_waiting_l594_594439

theorem people_waiting (people_per_entrance : ℕ) (num_entrances : ℕ) (total_people : ℕ) :
  people_per_entrance = 283 → num_entrances = 5 → total_people = people_per_entrance * num_entrances → total_people = 1415 :=
by
  intros hpe hne htp
  rw [hpe, hne] at htp
  exact htp
  sorry

end people_waiting_l594_594439


namespace proportion_neq_ad_bc_l594_594504

variable (a b c d : ℚ) 

-- Given the condition that all variables are non-zero
noncomputable def non_zero (x : ℚ) : Prop := x ≠ 0

-- Theorem statement
theorem proportion_neq_ad_bc 
  (h_nz_a : non_zero a)
  (h_nz_b : non_zero b)
  (h_nz_c : non_zero c)
  (h_nz_d : non_zero d) 
  (h_prop : b / d = c / a) : ¬ (a * d = b * c) :=
begin
  sorry
end

end proportion_neq_ad_bc_l594_594504


namespace tan_X_in_triangle_XYZ_l594_594305

noncomputable def triangle_XYZ (X Y Z : Type) :=
  (angle_Z: ℝ) (XY YZ: ℝ) : Prop :=
  angle_Z = 90 ∧ XY = 20 ∧ YZ = Real.sqrt 51

theorem tan_X_in_triangle_XYZ
  {X Y Z : Type}
  (h : triangle_XYZ X Y Z)
  : Real.tan X = Real.sqrt 51 / Real.sqrt 349 :=
by
  sorry

end tan_X_in_triangle_XYZ_l594_594305


namespace harmonic_mean_pairs_count_l594_594209

open Nat

theorem harmonic_mean_pairs_count :
  ∃! n : ℕ, (∀ x y : ℕ, x < y ∧ x > 0 ∧ y > 0 ∧ (2 * x * y) / (x + y) = 4^15 → n = 29) :=
sorry

end harmonic_mean_pairs_count_l594_594209


namespace prize_money_distribution_l594_594487

theorem prize_money_distribution :
  ∀ (total_prize : ℕ) (total_winners : ℕ) (first_prize : ℕ) (second_prize : ℕ) (third_prize : ℕ),
  total_prize = 800 →
  total_winners = 18 →
  first_prize = 200 →
  second_prize = 150 →
  third_prize = 120 →
  let remaining_prize := total_prize - (first_prize + second_prize + third_prize) in
  let remaining_winners := total_winners - 3 in
  remaining_prize / remaining_winners = 22 :=
begin
  intros _ _ _ _ _ _ _ _ _ _,
  sorry
end

end prize_money_distribution_l594_594487


namespace jill_total_tax_is_5_3495_percent_of_T_l594_594005

variables (T : ℝ) (clothing_pct food_pct electronics_pct other_items_pct : ℝ)
variables (clothing_tax food_tax electronics_tax other_items_tax : ℝ)

-- Conditions
def spending_clothing := 0.585 * T
def spending_food := 0.12 * T
def spending_electronics := 0.225 * T
def spending_other_items := 0.07 * T

def tax_clothing := 0.052 * spending_clothing
def tax_food := 0 * spending_food 
def tax_electronics := 0.073 * spending_electronics
def tax_other_items := 0.095 * spending_other_items

def total_tax_paid := tax_clothing + tax_food + tax_electronics + tax_other_items
def expected_tax_rate := 0.053495

theorem jill_total_tax_is_5_3495_percent_of_T : total_tax_paid / T = expected_tax_rate := sorry

end jill_total_tax_is_5_3495_percent_of_T_l594_594005


namespace hyperbola_eq_l594_594551

theorem hyperbola_eq (a : ℝ) (h : 3 = a ∧ ∀ F : ℝ×ℝ, F = (3,0) ∧ ∃ √5 : ℝ, distance F (\asymptote) = √5): 
  ∃ a, (EquationOfHyperbola : x^2/4 - y^2/5 = 1) :=
by
  sorry

end hyperbola_eq_l594_594551


namespace constructible_triangle_exists_l594_594516

theorem constructible_triangle_exists 
  (k : Set Point) (R r c varrho_c : ℝ)
  (is_circumcircle : IsCircumcircle k R)
  (is_incircle : IsIncircle r)
  (side_AB_eq : Side AB = c)
  (excircle_radius : ExcircleRadius opposite C varrho_c)
  (centre_excircle_distance : CentreExcircleDistance varrho_c from AB):
  ∃ (Δ : Triangle), IsInscribedCircle Δ k ∧ Side Δ.AB = c ∧ ExcircleRadius opposite C varrho_c := 
sorry

end constructible_triangle_exists_l594_594516


namespace complement_of_A_l594_594237

def R : set ℝ := set.univ

def A : set ℝ := {y | ∃ x ∈ R, y = x ^ 2}

theorem complement_of_A :
  R \ A = {y | y < 0} :=
by
  sorry

end complement_of_A_l594_594237


namespace matrix_exponentiation_l594_594180

theorem matrix_exponentiation (a n : ℕ) (M : Matrix (Fin 3) (Fin 3) ℕ) (N : Matrix (Fin 3) (Fin 3) ℕ) :
  (M^n = N) →
  M = ![
    ![1, 3, a],
    ![0, 1, 5],
    ![0, 0, 1]
  ] →
  N = ![
    ![1, 27, 3060],
    ![0, 1, 45],
    ![0, 0, 1]
  ] →
  a + n = 289 :=
by
  intros h1 h2 h3
  sorry

end matrix_exponentiation_l594_594180


namespace interest_rate_simple_and_compound_l594_594149

theorem interest_rate_simple_and_compound (P T: ℝ) (SI CI R: ℝ) 
  (simple_interest_eq: SI = (P * R * T) / 100)
  (compound_interest_eq: CI = P * ((1 + R / 100) ^ T - 1)) 
  (hP : P = 3000) (hT : T = 2) (hSI : SI = 300) (hCI : CI = 307.50) :
  R = 5 :=
by
  sorry

end interest_rate_simple_and_compound_l594_594149


namespace calc_problem1_calc_problem2_calc_problem3_calc_problem4_l594_594852

theorem calc_problem1 : (-3 + 8 - 15 - 6 = -16) :=
by
  sorry

theorem calc_problem2 : (-4/13 - (-4/17) + 4/13 + (-13/17) = -9/17) :=
by
  sorry

theorem calc_problem3 : (-25 - (5/4 * 4/5) - (-16) = -10) :=
by
  sorry

theorem calc_problem4 : (-2^4 - (1/2 * (5 - (-3)^2)) = -14) :=
by
  sorry

end calc_problem1_calc_problem2_calc_problem3_calc_problem4_l594_594852


namespace x_equals_y_squared_plus_2y_minus_1_l594_594272

theorem x_equals_y_squared_plus_2y_minus_1 (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 :=
sorry

end x_equals_y_squared_plus_2y_minus_1_l594_594272


namespace vector_sum_zero_l594_594263

-- Define the board and required properties
structure Board (m n : ℕ) :=
  (cells : Matrix (Fin (2 * m)) (Fin n) (Fin 2)) -- 1 for black, 0 for white
  (opposite_ends_different_colors : cells 0 0 ≠ cells (Fin.mk (2*m-1) (by decide)) (Fin.mk (n-1) (by decide)))
  (half_black_half_white : (∑ i j, cells i j) = (m * n))

-- Define the proof problem statement
theorem vector_sum_zero (m n : ℕ) (B : Board m n) :
  ∃ (v : Fin (2 * m) × Fin n → Fin (2 * m) × Fin n → ℝ × ℝ), 
    (∀ i j, ¬ B.cells i.fst i.snd = 1 ∨ ¬ B.cells j.fst j.snd = 1 → v (i, j) = (0, 0)) ∧
    (∑ i j, v (i, j)) = (0, 0) :=
sorry

end vector_sum_zero_l594_594263


namespace find_natural_number_n_l594_594529

theorem find_natural_number_n (n : ℕ) (x y k : ℤ) (h1 : gcd x y = 1) (h2 : k > 1) (h3 : 3^n = x^k + y^k) : n = 2 :=
by
  sorry

end find_natural_number_n_l594_594529


namespace platform_length_l594_594117

-- Define the main function that encapsulates the conditions and statement.
theorem platform_length 
(train_length : ℝ) 
(crosses_platform : ℝ) 
(crosses_pole : ℝ) 
(platform_cross_time : ℝ) 
(pole_cross_time : ℝ) : 
  train_length = 300 → crosses_platform = 36 → crosses_pole = 18 → 
  platform_cross_time = platform_cross_time → 
  pole_cross_time = pole_cross_time → 
  platform_cross_time / pole_cross_time = (train_length + 300) / train_length → 
  300 ≤ (train_length + 300) / 36 := 
  begin
    -- sorry marks the proof section to be completed.
    sorry
  end

end platform_length_l594_594117


namespace car_cleaning_ratio_l594_594348

theorem car_cleaning_ratio
    (outside_cleaning_time : ℕ)
    (total_cleaning_time : ℕ)
    (h1 : outside_cleaning_time = 80)
    (h2 : total_cleaning_time = 100) :
    (total_cleaning_time - outside_cleaning_time) / outside_cleaning_time = 1 / 4 :=
by
  sorry

end car_cleaning_ratio_l594_594348


namespace solve_for_a_l594_594278

theorem solve_for_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
(h_eq_exponents : a ^ b = b ^ a) (h_b_equals_3a : b = 3 * a) : a = Real.sqrt 3 :=
sorry

end solve_for_a_l594_594278


namespace proof1_l594_594511

def prob1 : Prop :=
  (1 : ℝ) * (Real.sqrt 45 + Real.sqrt 18) - (Real.sqrt 8 - Real.sqrt 125) = 8 * Real.sqrt 5 + Real.sqrt 2

theorem proof1 : prob1 :=
by
  sorry

end proof1_l594_594511


namespace tetrahedron_area_relationship_l594_594845

theorem tetrahedron_area_relationship
  (O A B C : Point) -- Points representing the vertices
  (S S1 S2 S3 : ℝ) -- Real numbers representing the areas
  (h1 : ∠AOB = 90) (h2 : ∠BOC = 90) (h3 : ∠COA = 90)
  (h4 : area_face ABC = S) -- Area of face ABC opposite vertex O
  (h5 : area_face OAB = S1) -- Area of face OAB
  (h6 : area_face OAC = S2) -- Area of face OAC
  (h7 : area_face OBC = S3) -- Area of face OBC
  : S^2 = S1^2 + S2^2 + S3^2 := 
  sorry

end tetrahedron_area_relationship_l594_594845


namespace weight_of_new_student_l594_594714

-- Assuming that the conditions about weight decrease and replacement are given:
-- Let W_new be the weight of the new student.

theorem weight_of_new_student (d: ℕ) (w: ℕ) (W_new: ℕ) (h₁: d = 12) (h₂: w = 72) (h₃: 5 * d = w - W_new) : W_new = 12 := by
  rw [h₁, h₂]
  sorry

end weight_of_new_student_l594_594714


namespace sequence_problem_l594_594302

-- Define the sequence aₙ recursively
def a : ℕ → ℤ
| 0       := 2
| (n + 1) := 1 - (a n)

-- Define the partial sum Sₙ
def S (n : ℕ) : ℤ :=
  (Finset.range (n + 1)).sum (λ i, a i)

theorem sequence_problem :
  S 2015 - 2 * S 2016 + S 2017 = 3 :=
by
  sorry

end sequence_problem_l594_594302


namespace area_midpoints_l594_594676

variable (A B C D E F G H : Point)
variable [ConvexQuadrilateral A B C D]
variable (midpoint_AB : Midpoint A B E)
variable (midpoint_BC : Midpoint B C F)
variable (midpoint_CD : Midpoint C D G)
variable (midpoint_DA : Midpoint D A H)

theorem area_midpoints (h : ConvexQuadrilateral A B C D) (h1 : Midpoint A B E) 
  (h2 : Midpoint B C F) (h3 : Midpoint C D G) (h4 : Midpoint D A H) :
  Area A B C D ≤ distance E G * distance H F := 
sorry

end area_midpoints_l594_594676


namespace smallest_natural_number_l594_594299

theorem smallest_natural_number :
  ∃ N : ℕ, ∃ f : ℕ → ℕ → ℕ, 
  f (f (f 9 8 - f 7 6) 5 + 4 - f 3 2) 1 = N ∧
  N = 1 := 
by sorry

end smallest_natural_number_l594_594299


namespace exists_eulerian_circuit_without_small_cycles_l594_594015

open Nat

noncomputable def has_eulerian_circuit_without_small_cycles (p : ℕ) (hp : Nat.Prime p) (h_large : p > 2023) : Prop :=
  ∃ (g : ℕ), (∀ (a b : ℕ), a ≤ 2023 → b ≤ 2023 → g ≠ (-a) / b) ∧
    ∃ (circuit : List (Fin p)), 
    (∀ (v : Fin p), v ∈ circuit) ∧
    (∃ (seqs : List (List (Fin p))), 
      (∀ seq ∈ seqs, ∀ i j, i ≠ j → seq[i] ≠ seq[j]) ∧
      seqs.chain' (λ s t, t.head = s.last) ∧ 
      (circuit = seqs.join) ∧
      (∀ k ≤ 2023, ¬ list_cycle_length circuit k))

theorem exists_eulerian_circuit_without_small_cycles (p : ℕ) (hp : Nat.Prime p) (h_large : p > 2023) :
  has_eulerian_circuit_without_small_cycles p hp h_large :=
sorry

end exists_eulerian_circuit_without_small_cycles_l594_594015


namespace distance_MN_l594_594635

-- Definitions based on the conditions
def ray_l_polar (ρ : ℝ) : Prop := 
  ρ ≥ 0 ∧ ∃ θ : ℝ, θ = π / 3

def curve_C1_cartesian (x y : ℝ) : Prop := 
  (x / 3) ^ 2 + (y / 2) ^ 2 = 1

def curve_C2_cartesian (x y : ℝ) : Prop := 
  (x^2 + (y - 2)^2 = 4)

def curve_C3_polar (θ : ℝ) : ℝ := 
  8 * Real.sin θ

-- The proof problem statement: Given the curves and ray, prove |MN| = 2√3
theorem distance_MN : 
  ∀ (ρ₁ ρ₂ : ℝ), 
    (ray_l_polar ρ₁ ∧ curve_C2_cartesian ρ₁ 2) →
    (ray_l_polar ρ₂ ∧ ∃ θ : ℝ, curve_C3_polar θ = ρ₂) →
    abs (ρ₂ - ρ₁) = 2 * Real.sqrt 3 :=
by
  sorry

end distance_MN_l594_594635


namespace math_problem_l594_594410

noncomputable def sequence_satisfies : Prop :=
  ∀ n : ℕ, n > 0 → (∃ S_n : ℝ, ∃ a_n : ℝ, a_n > 0 ∧ (a_n^2 + 2 * a_n = 4 * S_n + 3))

noncomputable def general_term_formula : Prop :=
  ∀ n : ℕ, n > 0 → (∃ a_n : ℝ, a_n = 2 * n + 1)

noncomputable def product_inequality (n : ℕ) (a : ℕ → ℝ) : Prop :=
  (∀ i, 0 < i ∧ i < n → a i > 0) →
  (∀ i, 0 < i ∧ i ≤ n → a i = 2 * i + 1) →
  (∏ i in Finset.range (n-1), (1 + 1 / (a (i + 1)))) > (real.sqrt (a n)) / 2

theorem math_problem :
  sequence_satisfies →
  general_term_formula →
  (∀ (n : ℕ), n > 1 → product_inequality n (λ i, 2 * i + 1)) :=
begin
  sorry
end

end math_problem_l594_594410


namespace common_sum_of_magic_square_l594_594373

theorem common_sum_of_magic_square : 
  let nums := list.range’ (-12) 25 in
  let total_sum := nums.sum in
  let common_sum := total_sum / 3 in
  (forall (row : list ℤ), row.length = 3 → row.sum = common_sum) ∧
  (forall (col : list ℤ), col.length = 3 → col.sum = common_sum) ∧
  (list.nth_le nums 0 0 + list.nth_le nums 4 0 + list.nth_le nums 8 0 = common_sum) ∧
  (list.nth_le nums 2 0 + list.nth_le nums 4 0 + list.nth_le nums 6 0 = common_sum) →
  common_sum = 0 := 
by
  sorry

end common_sum_of_magic_square_l594_594373


namespace shaded_square_144_covers_all_rows_l594_594139

def shaded_squares : ℕ → ℕ
| 1 => 1
| n+1 => shaded_squares n + (2 * n + 1)

theorem shaded_square_144_covers_all_rows :
  ∀ n < 12, ∃ k, k < n ↔ ∃ m, shaded_squares m % 12 = k :=
begin
  sorry
end

end shaded_square_144_covers_all_rows_l594_594139


namespace midpoint_and_distance_l594_594298

-- Define the given complex numbers
def z1 : ℂ := 5 - 2 * Complex.I
def z2 : ℂ := -1 + 6 * Complex.I

-- Define the midpoint calculation
def midpoint (a b : ℂ) : ℂ := (a + b) / 2

-- Theorem statement proving the midpoint and the distance
theorem midpoint_and_distance : 
  (midpoint z1 z2 = 2 + 2 * Complex.I) ∧ (Complex.abs ((midpoint z1 z2) - z1) = 5) :=
by
  sorry

end midpoint_and_distance_l594_594298


namespace f_periodic_with_period_4a_l594_594683

-- Definitions 'f' and 'g' (functions on real numbers), and the given conditions:
variables {a : ℝ} (f g : ℝ → ℝ)
-- Condition on a: a ≠ 0
variable (ha : a ≠ 0)

-- Given conditions
variable (hf0 : f 0 = 1) (hga : g a = 1) (h_odd_g : ∀ x : ℝ, g x = -g (-x))

-- Functional equation
variable (h_func_eq : ∀ x y : ℝ, f (x - y) = f x * f y + g x * g y)

-- The theorem stating that f is periodic with period 4a
theorem f_periodic_with_period_4a : ∀ x : ℝ, f (x + 4 * a) = f x :=
by
  sorry

end f_periodic_with_period_4a_l594_594683


namespace part_i_part_ii_l594_594916

-- Part (i)
theorem part_i (m : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) 1, -x^2 + 3 * m - 1 ≤ 0) ∧ 
  (∃ x ∈ Icc (-1 : ℝ) 1, m - x ≤ 0) = False ∧ 
  (∀ x ∈ Icc (-1 : ℝ) 1, -x^2 + 3 * m - 1 ≤ 0) ∨ 
  (∃ x ∈ Icc (-1 : ℝ) 1, m - x ≤ 0)
  → (1 / 3 < m) ∧ (m ≤ 1) :=
sorry

-- Part (ii)
theorem part_ii (a : ℝ) :
  (∀ m, (∀ x ∈ Icc (-1 : ℝ) 1, -x^2 + 3 * m - 1 ≤ 0) → 
        (∃ x ∈ Icc (-1 : ℝ) 1, m - a * x ≤ 0)) ∧
  (∃ m, (∃ x ∈ Icc (-1 : ℝ) 1, m - a * x ≤ 0) ∧ 
        ¬ (∀ x ∈ Icc (-1 : ℝ) 1, -x^2 + 3 * m - 1 ≤ 0))
  → (a ≥ 1 / 3 ∨ a ≤ -1 / 3) :=
sorry

end part_i_part_ii_l594_594916


namespace sqrt_expression_l594_594167

theorem sqrt_expression (x : ℕ) (h : x = 19) : 
  Int.sqrt ((x + 2) * (x + 1) * x * (x - 1) + 1) = 379 :=
by
  rw h
  -- Placeholder for proof
  sorry

end sqrt_expression_l594_594167


namespace total_stairs_climbed_l594_594324

theorem total_stairs_climbed (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 1269 → 
  julia_stairs = 1269 / 3 - 7 → 
  jonny_stairs + julia_stairs = 1685 :=
begin
  intros hjonny hjulia,
  rw hjonny at *,
  rw hjulia,
  norm_num,
end

end total_stairs_climbed_l594_594324


namespace cyclone_pump_30_minutes_l594_594711

theorem cyclone_pump_30_minutes (rate_per_hour : ℕ) (time_minutes : ℕ) (pump_in_30_min : ℕ) :
  rate_per_hour = 500 → time_minutes = 30 → pump_in_30_min = (rate_per_hour * (time_minutes / 60)) → pump_in_30_min = 250 :=
by
  intros h_rate h_time h_pump
  rw [h_rate, h_time]
  dsimp at h_pump
  norm_num at h_pump
  exact h_pump.symm

end cyclone_pump_30_minutes_l594_594711


namespace paint_faces_not_sum_to_9_l594_594844

theorem paint_faces_not_sum_to_9 :
  let faces := Finset.univ : Finset (Fin 8)
  let forbidden_pairs := {(1, 8), (2, 7), (3, 6), (4, 5), (8, 1), (7, 2), (6, 3), (5, 4)}
  let valid_pairs := (faces.product faces).filter (λ p, p.1 ≠ p.2 ∧ (p.1 + p.2).val ≠ 9)
  valid_pairs.card / 2 = 20 := 
by
  sorry

end paint_faces_not_sum_to_9_l594_594844


namespace two_f_eq_eight_over_four_plus_x_l594_594976

noncomputable def f : ℝ → ℝ := sorry

theorem two_f_eq_eight_over_four_plus_x (f_def : ∀ x > 0, f (2 * x) = 2 / (2 + x)) :
  ∀ x > 0, 2 * f x = 8 / (4 + x) :=
by
  sorry

end two_f_eq_eight_over_four_plus_x_l594_594976


namespace compound_interest_l594_594709

-- Definitions and conditions
variables (t p1 p2 : ℝ) (n : ℕ)
-- Ensure the conditions (positivity)
variables (ht : 0 < t) (hp1 : 0 < p1) (hp2 : 0 < p2)

-- Main statement to prove
theorem compound_interest (t p1 p2 : ℝ) (n : ℕ) (ht : 0 < t) (hp1 : 0 < p1) (hp2 : 0 < p2) :
  let r := 1 + p2 / 100 in
  T = t * (1 + p1 / p2 * (r ^ n - 1)) :=
sorry

end compound_interest_l594_594709


namespace percentage_of_stock_l594_594732

noncomputable def investment_amount : ℝ := 6000
noncomputable def income_derived : ℝ := 756
noncomputable def brokerage_percentage : ℝ := 0.25
noncomputable def brokerage_fee : ℝ := investment_amount * (brokerage_percentage / 100)
noncomputable def net_investment_amount : ℝ := investment_amount - brokerage_fee
noncomputable def dividend_yield : ℝ := (income_derived / net_investment_amount) * 100

theorem percentage_of_stock :
  ∃ (percentage_of_stock : ℝ), percentage_of_stock = dividend_yield := by
  sorry

end percentage_of_stock_l594_594732


namespace pi_times_volume_difference_eq_31_25_l594_594319

-- Definitions according to the given conditions
def john_height : ℝ := 10
def john_base_circumference : ℝ := 5
def chris_height : ℝ := 5
def chris_base_circumference : ℝ := 10

-- Calculations following the provided conditions
def john_radius : ℝ := john_base_circumference / (2 * Real.pi)
def chris_radius : ℝ := chris_base_circumference / (2 * Real.pi)

def john_volume : ℝ := Real.pi * (john_radius^2) * john_height
def chris_volume : ℝ := Real.pi * (chris_radius^2) * chris_height

-- Positive difference of the volumes
def volume_difference : ℝ := chris_volume - john_volume

-- Final proof statement
theorem pi_times_volume_difference_eq_31_25 : Real.pi * volume_difference = 31.25 := by
  sorry

end pi_times_volume_difference_eq_31_25_l594_594319


namespace digit_B_divisibility_l594_594665

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧
    (∃ n : ℕ, 658274 * 10 + B = 2 * n) ∧
    (∃ m : ℕ, 6582740 + B = 4 * m) ∧
    (B = 0 ∨ B = 5) ∧
    (∃ k : ℕ, 658274 * 10 + B = 7 * k) ∧
    (∃ p : ℕ, 6582740 + B = 8 * p) :=
sorry

end digit_B_divisibility_l594_594665


namespace problem_part_one_problem_part_two_l594_594557

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  (a > b ∧ b > 0 ∧ 
   (∃ e : ℝ, e = (Real.sqrt 2) / 2 ∧ e = (Real.sqrt (1 - (b^2) / (a^2))))) →
  (a = Real.sqrt 2 * b) →
  (b = 1) →
  (a = Real.sqrt 2) →
  (∀ x y : ℝ, ((x^2) / 2 + y^2 = 1))

noncomputable def trajectory_eq : Prop :=
  (∀ P Q : ℝ × ℝ, 
    (P.1^2 / 2 + P.2^2 = 1 ∧ Q.1^2 / 2 + Q.2^2 = 1 ∧ 
     P.1 * Q.1 + P.2 * Q.2 = 0 ∧ 
     (O : ℝ × ℝ) (R : ℝ × ℝ), R.1^2 + R.2^2 = P.1^2 + P.2^2 ∧ P.1^2 + P.2^2 = Q.1^2 + Q.2^2)) →
  (∀ R : ℝ × ℝ, (R.1^2 + R.2^2 = 2 / 3))

theorem problem_part_one : ∀ (a b : ℝ), ellipse_eq a b :=
by sorry

theorem problem_part_two : trajectory_eq :=
by sorry

end problem_part_one_problem_part_two_l594_594557


namespace transform_log2_l594_594754

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 x

noncomputable def g (x : ℝ) : ℝ := Real.logb 2 (1 - x)

theorem transform_log2 (x : ℝ) : g(x) = Real.logb 2 (1 - x) :=
by
  sorry

end transform_log2_l594_594754


namespace D_base_cases_D_recursive_D_2n_odd_l594_594303

-- Definitions of derangements and the number of derangements D_n
def A (n : ℕ) : Set (Fin (n + 1) → Fin (n + 1)) := 
  { f | ∀ i, f i ≠ i }

def D : ℕ → ℕ
| 0     := 1   -- Since there's just 1 way to derange an empty set
| 1     := 0
| 2     := 1
| 3     := 2
| 4     := 9
| (n+1) := (n) * (D n + D (n-1))

-- Proofs of the specific properties of derangements
theorem D_base_cases :
  D 1 = 0 ∧ D 2 = 1 ∧ D 3 = 2 ∧ D 4 = 9 := by
  repeat { split }
  exact rfl
  repeat { exact rfl }

theorem D_recursive (n : ℕ) (h : n ≥ 3) : 
  D n = (n-1) * (D (n-1) + D (n-2)) := by
  cases n
  . exfalso; linarith
  cases n
  . exfalso; linarith
  cases n
  . exfalso; linarith
  unfold D
  sorry -- detailed proof steps are skipped

theorem D_2n_odd (n : ℕ) (h : 0 < n) : 
  D (2 * n) % 2 = 1 := by
  induction n with k hk
  . exfalso; linarith
  . sorry -- inductive step proof is skipped

end D_base_cases_D_recursive_D_2n_odd_l594_594303


namespace find_integer_N_l594_594195

theorem find_integer_N :
  ∃ N : ℕ, (N % 10 = 2) ∧ (let d := Nat.digits 10 N in (Nat.ofDigits 10 (2 :: d.dropLast) = 2 * N)) 
:= sorry

end find_integer_N_l594_594195


namespace finite_values_initial_values_l594_594672

def f (x : ℝ) : ℝ := 6 * x - x^2

def sequence (x₀ : ℝ) (n : ℕ) : ℝ :=
  (nat.rec_on n x₀ (λ n xn, f xn))

theorem finite_values_initial_values : {x₀ : ℝ | ∃ N, ∀ n ≥ N, sequence x₀ n = sequence x₀ N}.finite.card = 2 := 
sorry

end finite_values_initial_values_l594_594672


namespace evaluate_floor_abs_l594_594877

theorem evaluate_floor_abs : ⌊|(-58.7 : ℝ)|⌋ = 58 := 
begin
  sorry
end

end evaluate_floor_abs_l594_594877


namespace max_blocks_fit_in_box_l594_594434

def box_dimensions : ℕ × ℕ × ℕ := (4, 6, 2)
def block_dimensions : ℕ × ℕ × ℕ := (3, 2, 1)
def block_volume := 6
def box_volume := 48

theorem max_blocks_fit_in_box (box_dimensions : ℕ × ℕ × ℕ)
    (block_dimensions : ℕ × ℕ × ℕ) : 
  (box_volume / block_volume = 8) := 
by
  sorry

end max_blocks_fit_in_box_l594_594434


namespace sum_of_a_for_single_solution_l594_594867

theorem sum_of_a_for_single_solution :
  let discriminant (a : ℝ) := (a + 12) ^ 2 - 4 * 5 * 4 in
  ∀ a : ℝ, discriminant a = 0 →
  a = -8 ∨ a = -16 →
  (-8 + -16) = -24 :=
by
  intro discriminant
  intro ha
  intro hasingle
  sorry

end sum_of_a_for_single_solution_l594_594867


namespace midpoints_distance_l594_594462

theorem midpoints_distance {A B C D M : Type} {AB BC CD DE AE : ℝ}
  (h1 : ∠A B D = ∠B C D)
  (h2 : ∠B D C = 90)
  (h3 : AB = 5)
  (h4 : BC = 6)
  (h5 : M.is_midpoint A C)
  (h6 : D.is_midpoint C E)
  (h7 : AE = sqrt 11) :
  let DM := dist D M in
  8 * (DM ^ 2) = 22 :=
by
  sorry

end midpoints_distance_l594_594462


namespace absolute_value_equality_l594_594764

theorem absolute_value_equality (x : ℝ) : abs (-2) * (abs (-25) - abs x) = 40 → abs x = 5 :=
by
  have h₁ : abs (-2) = 2 := abs_neg 2
  have h₂ : abs (-25) = 25 := abs_neg 25
  sorry

end absolute_value_equality_l594_594764


namespace binom_sum_identity_series_sum_l594_594706

-- Proof of the binomial identity.
theorem binom_sum_identity (n k m : ℕ) :
  (finset.range (m+1)).sum (λ i, nat.choose (n+i) k) = 
  nat.choose (n+m+1) (k+1) - nat.choose n (k+1) :=
sorry

-- Summation of the series based on the binomial sum result.
theorem series_sum (n m : ℕ) :
  (finset.range (m+1)).sum (λ j, (finset.range n).prod (λ i, (j+1) + i)) = 
  (nat.factorial n) * (nat.choose (n+m+1) (n+1)) :=
sorry

end binom_sum_identity_series_sum_l594_594706


namespace asymptotes_of_hyperbola_l594_594721

def hyperbola_eq_asymptotes (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 3 = 1)

def asymptotes_eq (x y : ℝ) : Prop :=
  (sqrt 3 * x + 2 * y = 0) ∨ (sqrt 3 * x - 2 * y = 0)

theorem asymptotes_of_hyperbola (x y : ℝ) :
  hyperbola_eq_asymptotes x y → asymptotes_eq x y :=
sorry

end asymptotes_of_hyperbola_l594_594721


namespace find_m_of_quad_roots_l594_594371

theorem find_m_of_quad_roots
  (a b : ℝ) (m : ℝ)
  (ha : a = 5)
  (hb : b = -4)
  (h_roots : ∀ x : ℂ, (x = (2 + Complex.I * Real.sqrt 143) / 5 ∨ x = (2 - Complex.I * Real.sqrt 143) / 5) →
                     (a * x^2 + b * x + m = 0)) :
  m = 7.95 :=
by
  -- Proof goes here
  sorry

end find_m_of_quad_roots_l594_594371


namespace compare_l594_594447

-- Define the quadratic equation condition for x
def x : ℝ := (1 + Real.sqrt 5) / 2

-- x satisfies the quadratic equation x^2 = x + 1
axiom x_sq : x^2 = x + 1

-- Compute x^6 using the relation x^6 = 8x + 5
def x_six : ℝ := 8 * x + 5

-- Define the sixth root of 18
def root6_18 : ℝ := Real.root 6 18

-- The theorem to prove: root6_18 > x
theorem compare : root6_18 > x := 
sorry

end compare_l594_594447


namespace vehicle_value_fraction_l594_594073

theorem vehicle_value_fraction (V_this_year V_last_year : ℕ)
  (h_this_year : V_this_year = 16000)
  (h_last_year : V_last_year = 20000) :
  (V_this_year : ℚ) / V_last_year = 4 / 5 := by 
  rw [h_this_year, h_last_year]
  norm_num 
  sorry

end vehicle_value_fraction_l594_594073


namespace find_depth_of_water_l594_594801

-- Define the conditions
def tank_length : ℝ := 12
def tank_radius : ℝ := 2
def water_area : ℝ := 24

-- Define the depth of water
def depth_of_water (h : ℝ) : Prop :=
  2 * sqrt(2 * tank_radius * h - h ^ 2) = water_area / tank_length

-- The theorem to be proved
theorem find_depth_of_water (h : ℝ) (h_eq : depth_of_water h) : h = 2 - sqrt 3 :=
sorry

end find_depth_of_water_l594_594801


namespace cosine_angle_AB_CD_l594_594405

-- Condition: The square ABCD is constructed from six equilateral triangles.
def construct_square_eq_tri := ∃ (AB BC CD DA: ℝ), 
  AB = BC ∧ BC = CD ∧ CD = DA ∧ 
  let equilateral_triangle := Triangle.mk AB BC CD in
  (6 : ℕ) • equilateral_triangle

-- Condition: By folding, we get a six-faced polyhedron shaped like a pyramid.
def fold_square_to_pyramid := ∀ (AB CD : ℝ),
  construct_square_eq_tri → 
  let folded_pyramid := Pyramid.mk AB CD in
  folded_pyramid

-- Proof problem statement: Prove the cosine of the angle between AB and CD is 1/2
theorem cosine_angle_AB_CD (AB CD : ℝ) 
  (h1 : construct_square_eq_tri)
  (h2 : fold_square_to_pyramid AB CD h1) :
  real.cos (angle_between_edges AB CD) = 1/2 := 
  sorry

end cosine_angle_AB_CD_l594_594405


namespace a_plus_b_equals_4_l594_594686

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := log a (x + b)

theorem a_plus_b_equals_4 (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : f a b 2 = 1) (h4 : ∃ y, y = f a b 8 ∧ y = 2):
  a + b = 4 :=
sorry

end a_plus_b_equals_4_l594_594686


namespace reduction_for_same_profit_cannot_reach_460_profit_l594_594271

-- Defining the original conditions
noncomputable def cost_price_per_kg : ℝ := 20
noncomputable def original_selling_price_per_kg : ℝ := 40
noncomputable def daily_sales_volume : ℝ := 20

-- Reduction in selling price required for same profit
def reduction_to_same_profit (x : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - x
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * x
  new_profit_per_kg * new_sales_volume = (original_selling_price_per_kg - cost_price_per_kg) * daily_sales_volume

-- Check if it's impossible to reach a daily profit of 460 yuan
def reach_460_yuan_profit (y : ℝ) : Prop :=
  let new_selling_price := original_selling_price_per_kg - y
  let new_profit_per_kg := new_selling_price - cost_price_per_kg
  let new_sales_volume := daily_sales_volume + 2 * y
  new_profit_per_kg * new_sales_volume = 460

theorem reduction_for_same_profit : reduction_to_same_profit 10 :=
by
  sorry

theorem cannot_reach_460_profit : ∀ y, ¬ reach_460_yuan_profit y :=
by
  sorry

end reduction_for_same_profit_cannot_reach_460_profit_l594_594271


namespace min_diagonal_distance_cube_l594_594921

noncomputable theory

def diagonal_distance_min (a b c d e f : ℝ) (h_cube : ∀ i, i^2 = 1)
  (M N : ℝ) (h_parallel : (a - M) * (d - N) = 0) : ℝ :=
  min (4 * M^2 - 2 * M + 1) (4 * N^2 - 2 * N + 1)

theorem min_diagonal_distance_cube : 
  ∀ (a b c d e f M N : ℝ) (h : ∀ i, i^2 = 1) (h_parallel : (a - M) * (d - N) = 0), 
    diagonal_distance_min a b c d e f h M N h_parallel = 4 * ((1 / 3) - (1 / 4))^2 + 1 :=
begin
  sorry
end

end min_diagonal_distance_cube_l594_594921


namespace unique_solution_for_quadratic_l594_594942

theorem unique_solution_for_quadratic (a : ℝ) : 
  ∃! (x : ℝ), x^2 - 2 * a * x + a^2 = 0 := 
by
  sorry

end unique_solution_for_quadratic_l594_594942


namespace geometric_seq_solution_l594_594593

-- Define necessary variables and hypotheses
variables (a b c : ℝ)

-- Hypothesis: The sequence -1, a, b, c, -9 is a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  (b^2 = (-1) * (-9)) ∧ (b^2 = a * c)

-- Prove the value of b and a * c
theorem geometric_seq_solution (h : geometric_sequence a b c) : b = -3 ∧ a * c = 9 := 
by {
  -- Using the hypothesis of geometric sequence
  have h1 : b^2 = 9 := h.1,
  have h2 : b^2 = a * c := h.2,

  -- Solve for b
  have h3 : b = 3 ∨ b = -3 := by nlinarith,
  have h4 : b ≠ 3 := by nlinarith, -- Applying the logic that if b = 3 then a^2 = -3 which is impossible

  -- Conclude b = -3
  have hb : b = -3 := by nlinarith,

  -- Conclude ac = 9
  have hac : a * c = 9 := by nlinarith,
  
  exact ⟨hb, hac⟩
}

end geometric_seq_solution_l594_594593


namespace cos_theta_value_l594_594817

def vector_a : ℝ × ℝ × ℝ := (2, 1, 1)
def vector_b : ℝ × ℝ × ℝ := (1, -1, -1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def norm (u : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

noncomputable def cos_theta : ℝ :=
  let u := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2, vector_a.3 + vector_b.3)
  let v := (vector_b.1 - vector_a.1, vector_b.2 - vector_a.2, vector_b.3 - vector_a.3)
  dot_product u v / (norm u * norm v)

theorem cos_theta_value : cos_theta = 1 / 3 :=
  sorry

end cos_theta_value_l594_594817


namespace tangent_parabola_line_l594_594095

variable {R : Type*} [Field R]

def parabola (p q x : R) : R := x^2 + p*x + q
def line (p q x : R) : R := p*x + q

theorem tangent_parabola_line {p q : R} :
  ∃ x y : R, parabola p q x = y ∧ line p q x = y ∧ (parabola p q) x = (line p q) x ∧ x = 0 ∧ y = q ∧ 
  (derivative (parabola p q)) 0 = (derivative (line p q)) 0 :=
sorry

end tangent_parabola_line_l594_594095


namespace kaleb_earnings_and_boxes_l594_594326

-- Conditions
def initial_games : ℕ := 76
def games_sold : ℕ := 46
def price_15_dollar : ℕ := 20
def price_10_dollar : ℕ := 15
def price_8_dollar : ℕ := 11
def games_per_box : ℕ := 5

-- Definitions and proof problem
theorem kaleb_earnings_and_boxes (initial_games games_sold price_15_dollar price_10_dollar price_8_dollar games_per_box : ℕ) :
  let earnings := (price_15_dollar * 15) + (price_10_dollar * 10) + (price_8_dollar * 8)
  let remaining_games := initial_games - games_sold
  let boxes_needed := remaining_games / games_per_box
  earnings = 538 ∧ boxes_needed = 6 :=
by
  sorry

end kaleb_earnings_and_boxes_l594_594326


namespace intersection_M_N_l594_594341

noncomputable def M : set ℝ := {x | x^2 - 3*x + 2 > 0}
noncomputable def N : set ℝ := {x | (1/2:ℝ)^x ≥ 4}

theorem intersection_M_N :
  M ∩ N = {x | x ≤ -2} :=
sorry

end intersection_M_N_l594_594341


namespace problem1_problem2_problem3_problem4_l594_594512

theorem problem1 : 
  (4 + 1/4) - 3.8 + 4/5 - (-2.75) = 4 := by
  sorry

theorem problem2 : 
  | -1/10 | * (-5) + | -3 - 1/2 | = 3 := by
  sorry

theorem problem3 : 
  10 - 1 / ((1/3) - (1/6)) / (1/12) = -62 := by
  sorry

theorem problem4 : 
  -1^2024 - (1 - 0.5) * 1/3 * (2 - (-3)^2) = 1/6 := by
  sorry

end problem1_problem2_problem3_problem4_l594_594512


namespace ivy_has_20_collectors_dolls_l594_594870

theorem ivy_has_20_collectors_dolls
  (D : ℕ) (I : ℕ) (C : ℕ)
  (h1 : D = 60)
  (h2 : D = 2 * I)
  (h3 : C = 2 * I / 3) 
  : C = 20 :=
by sorry

end ivy_has_20_collectors_dolls_l594_594870


namespace parking_methods_count_l594_594496

theorem parking_methods_count (total_spaces : ℕ) (car_models : ℕ) (consecutive_empty_spaces : ℕ) 
  (h1 : total_spaces = 7) (h2 : car_models = 3) (h3 : consecutive_empty_spaces = 4) : 
  ∃ n : ℕ, n = 24 :=
by {
  use 24,
  sorry
}

end parking_methods_count_l594_594496


namespace find_a5_l594_594927

theorem find_a5 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = 2 * n * (n + 1))
  (ha : ∀ n ≥ 2, a n = S n - S (n - 1)) : 
  a 5 = 20 := 
sorry

end find_a5_l594_594927


namespace polygon_area_correct_l594_594520

def polygon_vertices : list (ℝ × ℝ) := 
  [(0, 0), (20, 0), (30, 10), (20, 20), (10, 10), (0, 20)]

def triangle_area (v1 v2 v3 : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2))

def polygon_area : ℝ :=
  triangle_area (0,0) (20,0) (10,10) + 
  triangle_area (20,0) (30,10) (10,10) + 
  triangle_area (30,10) (20,20) (10,10) + 
  triangle_area (30,10) (20,20) (0,20)

theorem polygon_area_correct : polygon_area = 400 :=
  by
    sorry

end polygon_area_correct_l594_594520


namespace opposite_numbers_A_l594_594444

theorem opposite_numbers_A :
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1
  
  (A1 = -A2 ∧ A2 = 1) ∧ ¬(B1 = -B2) ∧ ¬(C1 = -C2) ∧ ¬(D1 = -D2)
:= by
  let A1 := -((-1)^2)
  let A2 := abs (-1)

  let B1 := (-2)^3
  let B2 := -(2^3)
  
  let C1 := 2
  let C2 := 1 / 2
  
  let D1 := -(-1)
  let D2 := 1

  sorry

end opposite_numbers_A_l594_594444


namespace problem1_problem2_l594_594544

noncomputable def f1 (x : ℝ) := (x^2 + 2*x + 3) / (x + 1)
noncomputable def g1 (x : ℝ) := x + 1
noncomputable def t1 (x : ℝ) := f1(x) - g1(x)

noncomputable def f2 (x : ℝ) := sqrt(x^2 + 1)
noncomputable def g2 (a : ℝ) (x : ℝ) := a * x
noncomputable def t2 (a : ℝ) (x : ℝ) := f2(x) - g2(a, x)

theorem problem1 : (∀ x ≥ 0, t1(x) > 0 ∧ t1(x) ≤ 2) ∧ (∀ x ≥ 0, derivative t1 x < 0) ∧ (∀ x ≥ 0, lim (x : ℝ) at_top t1 = 0) → (∃ p > 0, t1(x) ≤ p):
  sorry

theorem problem2 (a : ℝ) : (∀ x ≥ 0, derivative (t2 a) x < 0) → (∀ x ≥ 0, t2(a, x) > 0) → (∀ x ≥ 0, lim (x : ℝ) at_top (t2 a) = 0) → a ≥ 1:
  sorry

end problem1_problem2_l594_594544


namespace exists_triangle_of_area_one_l594_594039

theorem exists_triangle_of_area_one (colors : Set (ℝ × ℝ)) (h_colored : ∃ n : ℕ, n = 100 ∧ ∀ p ∈ colors, p = n) : 
  ∃ (a b c : ℝ × ℝ), (∃ color : ℕ, (a ∈ color ∧ b ∈ color ∧ c ∈ color) ∧ 
  ∃ (a1 a2 : ℝ) (b1 b2 : ℝ) (c1 c2 : ℝ), a = (a1, a2) ∧ b = (b1, b2) ∧ c = (c1, c2) ∧ 
  (1 / 2) * abs ((b1 - a1) * (c2 - a2) - (b2 - a2) * (c1 - a1)) = 1) :=
sorry

end exists_triangle_of_area_one_l594_594039


namespace curves_intersect_four_points_l594_594761

theorem curves_intersect_four_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = 4 * a^2 ∧ y = x^2 - 2 * a) → (a > 1/3)) :=
sorry

end curves_intersect_four_points_l594_594761


namespace tangent_line_equation_exists_l594_594542

noncomputable def a : ℝ := 1 / 10

def curve1 (x y : ℝ) : Prop := x = sqrt(2 * y^2 + 25 / 2)
def curve2 (x y : ℝ) : Prop := y = a * x^2

def tangent_point (x y : ℝ) : Prop :=
curve1 x y ∧ curve2 x y

def tangent_line (x y : ℝ) : Prop :=
2 * x - 2 * y - 5 = 0

theorem tangent_line_equation_exists :
  ∃ x y : ℝ, tangent_point x y ∧ tangent_line x y :=
sorry

end tangent_line_equation_exists_l594_594542


namespace smallest_T_6_horses_return_l594_594171

def first_12_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def lcm_of_list (l : List Nat) : Nat := l.foldr Nat.lcm 1

theorem smallest_T_6_horses_return :
  let T := lcm_of_list (first_12_primes.take 6)
  T = 30030 ∧ (T.digits.sum = 6) :=
by
  sorry

end smallest_T_6_horses_return_l594_594171


namespace permutations_mississippi_correct_l594_594183

def permutations_mississippi : ℕ :=
  Nat.factorial 11 / (Nat.factorial 1 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 2)

theorem permutations_mississippi_correct : permutations_mississippi = 34650 :=
  by {
    unfold permutations_mississippi,
    -- Calculation steps omitted
    sorry
  }

end permutations_mississippi_correct_l594_594183


namespace point_on_line_l594_594193

theorem point_on_line (t : ℝ) :
  (∃ t, (∀ x y : ℝ, ((x, y) = (2, 4) ∨ (x, y) = (10, 1) ∨ (x, y) = (t, 7)) → 
      (y - 4) * (10 - 2) = (1 - 4) * (x - 2)) → t = -6) :=
begin
  sorry
end

end point_on_line_l594_594193


namespace find_valid_n_l594_594891

-- Definitions and conditions based on the problem statement
def lcm (p q : ℕ) : ℕ := Nat.lcm p q

def valid_n (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ n = lcm a b + lcm b c + lcm c a

def not_power_of_2 (n : ℕ) : Prop :=
  ∀ (k : ℕ), 2^k ≠ n

-- Main theorem statement
theorem find_valid_n (n : ℕ) : valid_n n ↔ not_power_of_2 n := sorry

end find_valid_n_l594_594891


namespace factor_expression_l594_594975

-- Define variables s and m
variables (s m : ℤ)

-- State the theorem to be proven: If s = 5, then m^2 - sm - 24 can be factored as (m - 8)(m + 3)
theorem factor_expression (hs : s = 5) : m^2 - s * m - 24 = (m - 8) * (m + 3) :=
by {
  sorry
}

end factor_expression_l594_594975


namespace cost_of_each_soda_l594_594653

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l594_594653


namespace pentagon_area_l594_594290

theorem pentagon_area
  (A B C D E : Point)
  (AB BC CD DE : ℝ)
  (h_AB_BC : AB = BC)
  (h_CD_DE : CD = DE)
  (angle_ABC : Real.angle)
  (angle_CDE : Real.angle)
  (BD_sq : ℝ)
  (h_angle_ABC : angle_ABC = 100)
  (h_angle_CDE : angle_CDE = 80)
  (h_BD_sq : BD_sq = 100 / Real.sin 100) :
  polygon_area ⟨[A, B, C, D, E], by sorry⟩ = 50 :=
sorry

end pentagon_area_l594_594290


namespace composite_evaluation_at_two_l594_594657

-- Define that P(x) is a polynomial with coefficients in {0, 1}
def is_binary_coefficient_polynomial (P : Polynomial ℤ) : Prop :=
  ∀ (n : ℕ), P.coeff n = 0 ∨ P.coeff n = 1

-- Define that P(x) can be factored into two nonconstant polynomials with integer coefficients
def is_reducible_to_nonconstant_polynomials (P : Polynomial ℤ) : Prop :=
  ∃ (f g : Polynomial ℤ), f.degree > 0 ∧ g.degree > 0 ∧ P = f * g

theorem composite_evaluation_at_two {P : Polynomial ℤ}
  (h1 : is_binary_coefficient_polynomial P)
  (h2 : is_reducible_to_nonconstant_polynomials P) :
  ∃ (m n : ℤ), m > 1 ∧ n > 1 ∧ P.eval 2 = m * n := sorry

end composite_evaluation_at_two_l594_594657


namespace parallelogram_property_area_l594_594129

open Real

noncomputable def calculate_property_area
  (scale : ℝ) -- scale in miles per inch
  (longer_diagonal : ℝ) -- longer diagonal in inches
  (shorter_diagonal : ℝ) -- shorter diagonal in inches
  (angle : ℝ) -- angle between diagonals in degrees
  : ℝ :=
  0.5 * (scale * longer_diagonal) * (scale * shorter_diagonal) * (sin (angle * pi / 180))

theorem parallelogram_property_area :
  calculate_property_area 200 10 6 90 = 1200000 :=
by
  unfold calculate_property_area
  have h1 : sin (90 * pi / 180) = 1 := by simp [sin_pi_div_two]
  simp [h1, mul_assoc]
  norm_num
  sorry

end parallelogram_property_area_l594_594129


namespace solve_quadratic_eq_l594_594402

theorem solve_quadratic_eq : (x : ℝ) → (x^2 - 4 = 0) → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_quadratic_eq_l594_594402


namespace concyclic_points_l594_594385

variables {A B C P D E F Q R : Type}
variables (AB BC CA : line_segment A B) (AB : line_segment A B)
variables (AB_P_AP BP R_P_AP : line_segment P)

def inscribed_circle_touches_sides (A B C P D E F Q R : Type) 
  (AB BC CA : line_segment A B) : Prop :=
  ∃ (I : Type), tangent_line A D I ∧ tangent_line B E I ∧ tangent_line C F I

def point_P_interior_of_triangle (A B C P D E F Q R : Type) 
  (AB_P_AP BP R_P_AP : line_segment P) : Prop :=
  ∃ (J : Type),
  tangent_line A D J ∧ tangent_line P Q J ∧ tangent_line P R J

theorem concyclic_points (A B C P D E F Q R : Type)
  (h1 : inscribed_circle_touches_sides A B C P D E F Q R AB BC CA)
  (h2 : point_P_interior_of_triangle A B C P D E F Q R AB_P_AP BP R_P_AP) :
  circle_through_points E F Q R := sorry

end concyclic_points_l594_594385


namespace ellipse_is_specific_ellipse_slope_range_l594_594235

noncomputable def ellipse_equation (x y: ℝ) (a b: ℝ) (H : a > b > 0) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

theorem ellipse_is_specific_ellipse (a b: ℝ) (H : a > b > 0):
  let e := real.sqrt 3 / 2 in
  e = real.sqrt (1 - b^2 / a^2) →
  (a^2 = 4 * b^2) →
  (x^2 / 4 + y^2 = 1) :=
by
  intros e Heq H1 H2,
  sorry

theorem slope_range (k1 k2 k: ℝ) (H: k1 * k2 - 1 = k1 + k2) :
  (1 - real.sqrt 2 ≤ k ∧ k < (1 - real.sqrt 2) / 4) ∨
  ((1 + real.sqrt 2) / 4 < k ∧ k ≤ 1 + real.sqrt 2) :=
by
  intros H,
  sorry

end ellipse_is_specific_ellipse_slope_range_l594_594235


namespace length_of_platform_l594_594793

theorem length_of_platform {train_length platform_crossing_time signal_pole_crossing_time : ℚ}
  (h_train_length : train_length = 300)
  (h_platform_crossing_time : platform_crossing_time = 40)
  (h_signal_pole_crossing_time : signal_pole_crossing_time = 18) :
  ∃ L : ℚ, L = 1100 / 3 :=
by
  sorry

end length_of_platform_l594_594793


namespace matrix_multiplication_correct_l594_594663

noncomputable def matrixN : Matrix (Fin 3) (Fin 3) ℝ := sorry

theorem matrix_multiplication_correct :
  matrixN ⬝ (λ i => if i = 0 then 3 else if i = 1 then -2 else 1) = (λ i => if i = 0 then 4 else if i = 1 then 1 else -1) ∧
  matrixN ⬝ (λ i => if i = 0 then 4 else if i = 1 then 1 else -4) = (λ i => if i = 0 then 0 else if i = 1 then 2 else 1) →
  matrixN ⬝ (λ i => if i = 0 then 7 else if i = 1 then -1 else -2) = (λ i => if i = 0 then 16 else if i = 1 then 7 else -2.5) :=
sorry

end matrix_multiplication_correct_l594_594663


namespace subcommittee_has_teacher_l594_594038

def total_combinations (n k : ℕ) : ℕ := Nat.choose n k

def teacher_subcommittee_count : ℕ := total_combinations 12 5 - total_combinations 7 5

theorem subcommittee_has_teacher : teacher_subcommittee_count = 771 := 
by
  sorry

end subcommittee_has_teacher_l594_594038


namespace parallelogram_CD_length_l594_594995

theorem parallelogram_CD_length (ABCD : Type) [parallelogram ABCD]
  (A B C D : Point)
  (hAB : AB = 10) (hBC : BC = 12) (hBD : BD = 15)
  (hAngle : ∠BAD ≅ ∠BCD) :
  CD = 18 ∧ (∃ (m n: ℕ), m + n = 19 ∧ gcd m n = 1) := 
sorry

end parallelogram_CD_length_l594_594995


namespace check_number_of_correct_statements_l594_594416

def statement_1 (α : Type) [LinearOrder α] (seq : ℕ → α) (formula : ℕ → α) : Prop :=
∀ n, seq n = formula n ∧ formula = λ n, seq n

def statement_2 (a : ℕ → ℚ) : Prop :=
∀ n, a n = (n+1)/(n+2)

def statement_3 (a : ℕ → ℚ) : Prop :=
∀ x : ℚ, ¬ ∃ n: ℕ, a n = x ∧ a n = a (n+1)

def statement_4 (a b : ℕ → ℚ): Prop :=
∀ n, a n = (-1) ^ n ∧ b n = (-1) ^ (n+1)

theorem check_number_of_correct_statements :
  (statement_1 real (λ n, (n:ℚ) / ((n + 1):ℚ)) → false) ∧
  (statement_2 (λ n, (n:ℚ) / ((n + 1):ℚ)) → false) ∧
  statement_3 (λ n, (n:ℚ) / ((n + 1):ℚ)) ∧
  (statement_4 (λ n, (1:ℚ) * (-1)^n) (λ n, (1:ℚ) * (-1)^(n+1)) → false) →
  true
:= sorry

end check_number_of_correct_statements_l594_594416


namespace path_count_equilateral_triangle_l594_594843

def f (n : ℕ) : ℕ :=
n.factorial

theorem path_count_equilateral_triangle (n : ℕ) : f(n) = n! := by
  sorry

end path_count_equilateral_triangle_l594_594843


namespace largest_lambda_l594_594533

-- Define the conditions
variables {A B C P A₁ B₁ C₁ : Type} [real] [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space A₁] [metric_space B₁] [metric_space C₁]

-- Define the angles
variables (ω : ℝ) (angle_PAB angle_PBC angle_PCA : ℝ)
variables (h1 : angle_PAB = ω) (h2 : angle_PBC = ω) (h3 : angle_PCA = ω)

-- Define the intersection points
variables (circum_PBC circum_PCA circum_PAB : Type) [metric_space circum_PBC] [metric_space circum_PCA] [metric_space circum_PAB]
variables (HA : ∀ (A : Type), A ∈ circum_PBC) (HB : ∀ (B : Type), B ∈ circum_PCA) (HC : ∀ (C : Type), C ∈ circum_PAB)

-- Define the function S to represent area
noncomputable def S (x y z : Type) [metric_space x] [metric_space y] [metric_space z] : ℝ := sorry
variables (S_A1BC S_B1CA S_C1AB S_ABC : ℝ)

-- The inequality to be proved
theorem largest_lambda (h_condition : S_A1BC + S_B1CA + S_C1AB ≥ 3 * S_ABC) : 
  ∃ (λ : ℝ), λ = 3 :=
begin
  use 3,
  sorry
end

end largest_lambda_l594_594533


namespace largest_possible_value_of_m_l594_594041

theorem largest_possible_value_of_m : 
  ∃ (qs : List (Polynomial ℝ)), 
    (∀ q ∈ qs, ¬q.is_constant ∧ q.splits (ring_hom.id _)) ∧ 
    (Polynomial.of_real 1).factors = qs ∧ 
    qs.length = 4 :=
by
  sorry

end largest_possible_value_of_m_l594_594041


namespace distance_to_school_l594_594321

theorem distance_to_school :
  ∃ d : ℝ, (d / 5 + d / 25 = 1) ∧ d = 25 / 6 :=
by
  use 25 / 6
  split
  sorry

end distance_to_school_l594_594321


namespace num_integers_between_700_and_900_with_sum_of_digits_18_l594_594960

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem num_integers_between_700_and_900_with_sum_of_digits_18 : 
  ∃ k, k = 17 ∧ ∀ n, 700 ≤ n ∧ n ≤ 900 ∧ sum_of_digits n = 18 ↔ (1 ≤ k) := 
sorry

end num_integers_between_700_and_900_with_sum_of_digits_18_l594_594960


namespace find_g_at_3_l594_594595

theorem find_g_at_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 2) = 4 * x + 1) : g 3 = 23 / 3 :=
by
  sorry

end find_g_at_3_l594_594595


namespace system_of_equations_sum_x_values_l594_594365

theorem system_of_equations_sum_x_values :
  ∀ (x y : ℝ), (y = 10) → (x^2 + y^2 = 169) → (x + -x = 0) :=
by
  intros x y hy heq
  have hx : x = real.sqrt 69 ∨ x = -real.sqrt 69 :=
    by sorry  -- this part needs proof
  cases hx
  case inl =>
    rw [hx, neg_sqrt_eq_neg] sorry  -- for example: complex.conj_sqrt_eq_neg
  case inr =>
    rw [hx] sorry  -- for example: complex.conj_neg_eq_neg

end system_of_equations_sum_x_values_l594_594365


namespace max_content_mps_l594_594737

universe u

-- Condition definitions
def UniqueSalaries (salaries : Matrix (Fin 10) (Fin 10) ℕ) : Prop :=
  Function.Injective salaries.flatten

def NeighbouringMPs (i j : Fin 10) : List (Fin 10 × Fin 10) :=
  if i = 0 then
    if j = 0 then [(0, 1), (1, 0)]
    else if j = 9 then [(0, 8), (1, 9)]
    else [(0, j-1), (0, j+1), (1, j)]
  else if i = 9 then
    if j = 0 then [(8, 0), (9, 1)]
    else if j = 9 then [(9, 8), (8, 9)]
    else [(9, j-1), (9, j+1), (8, j)]
  else if j = 0 then [(i-1, 0), (i+1, 0), (i, 1)]
  else if j = 9 then [(i-1, 9), (i+1, 9), (i, 8)]
  else [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

def ContentMP (salaries : Matrix (Fin 10) (Fin 10) ℕ) (i j : Fin 10) : Prop :=
  List.length (List.filter (fun (p : Fin 10 × Fin 10) => salaries p.fst p.snd > salaries i j) (NeighbouringMPs i j)) ≤ 1

-- Lean statement for the proof problem
theorem max_content_mps (salaries : Matrix (Fin 10) (Fin 10) ℕ) :
  UniqueSalaries salaries →
  ∃ M : ℕ, M = 72 ∧ ∀ i j, ContentMP salaries i j → True := sorry

end max_content_mps_l594_594737


namespace part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l594_594268

-- Define initial conditions
def cost_price : ℝ := 20
def initial_selling_price : ℝ := 40
def initial_sales_volume : ℝ := 20
def price_decrease_per_kg : ℝ := 1
def sales_increase_per_kg : ℝ := 2
def original_profit : ℝ := 400

-- Part (1) statement
theorem part1_price_reduction_maintains_profit :
  ∃ x : ℝ, (initial_selling_price - x - cost_price) * (initial_sales_volume + sales_increase_per_kg * x) = original_profit ∧ x = 20 := 
sorry

-- Part (2) statement
theorem part2_profit_reach_460_impossible :
  ¬∃ y : ℝ, (initial_selling_price - y - cost_price) * (initial_sales_volume + sales_increase_per_kg * y) = 460 :=
sorry

end part1_price_reduction_maintains_profit_part2_profit_reach_460_impossible_l594_594268


namespace sum_exterior_angles_regular_decagon_l594_594408

theorem sum_exterior_angles_regular_decagon : 
  ∀ (P : Type) [add_comm_group P] [modular_lattice P] [h : ∀ (p q : P), p ≤ q ∨ q ≤ p], 
  (∀ (n : ℕ), n = 10 → ∀ (p : P), ∃ (exterior_angle_sum : P), exterior_angle_sum = 360) := 
begin
  sorry
end

end sum_exterior_angles_regular_decagon_l594_594408


namespace expected_value_base_3_l594_594703

theorem expected_value_base_3 :
  (∀ (n : ℕ), n > 0 → ∃ (a_n : ℕ), a_n = expected_value (erase_digits n)) →
  (∏ k in (finset.range 2021).erase(0), (k + 2) / k) * 1 = 681751 :=
by
  have h_base := sorry
  exact h_base

end expected_value_base_3_l594_594703


namespace angle_of_inclination_range_l594_594052

theorem angle_of_inclination_range (a : ℝ) :
  let α := real.atan (-1 / (a^2 + 1)) in
  3 * real.pi / 4 ≤ α ∧ α < real.pi :=
by
  sorry

end angle_of_inclination_range_l594_594052


namespace numberOfBigBoats_l594_594789

-- Conditions
variable (students : Nat) (bigBoatCapacity : Nat) (smallBoatCapacity : Nat) (totalBoats : Nat)
variable (students_eq : students = 52)
variable (bigBoatCapacity_eq : bigBoatCapacity = 8)
variable (smallBoatCapacity_eq : smallBoatCapacity = 4)
variable (totalBoats_eq : totalBoats = 9)

theorem numberOfBigBoats : bigBoats + smallBoats = totalBoats → 
                         bigBoatCapacity * bigBoats + smallBoatCapacity * smallBoats = students → 
                         bigBoats = 4 := 
by
  intros h1 h2
  -- Proof steps
  sorry


end numberOfBigBoats_l594_594789


namespace integral_value_eq_one_third_l594_594247

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem integral_value_eq_one_third : 
  (∃ (a : ℝ), (binomial_coeff 6 3) * (3^3) * (a^3) = 40 ∧ (∫ x in 0..1, x^a) = (1/3)) :=
by
  sorry

end integral_value_eq_one_third_l594_594247


namespace increasing_intervals_sine_transformed_l594_594026

open Real Function

theorem increasing_intervals_sine_transformed (k : ℤ) :
  let g := fun (x : ℝ) => sin (2 * x - π / 3)
  let I := {k : ℤ} × set.Icc (k * π - π / 12) (k * π + 5 * π / 12)
  ∀ x, x ∈ I → ∃ δ > 0, ∀ ε (hε : ε > 0), abs (ε) < δ → g (x + ε) > g x := sorry

end increasing_intervals_sine_transformed_l594_594026


namespace ratio_to_percent_l594_594396

theorem ratio_to_percent (a b : ℕ) (h : a = 2 ∧ b = 3) : ((a.toRat / (a + b).toRat) * 100) = 40 := by
  have h_frac : (a.toRat / (a + b).toRat) = (2.toRat / 5.toRat) := by
    rw [h.left, h.right]
    norm_cast
    simp
    
  have h_percent : ((2.toRat / 5.toRat) * 100) = 40 := by
    norm_cast
    exact rat.mul_div_cancel (2 * 100) (by norm_num : 5 ≠ 0)
    
  rw h_frac
  exact h_percent

end ratio_to_percent_l594_594396


namespace find_value_of_f_log_l594_594242

open Real

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

noncomputable def given_function : ℝ → ℝ := 
  λ x, if x > 0 then 1 + 2^x else - (1 + 2^(-x))

theorem find_value_of_f_log:
  is_odd given_function →
  f = given_function →
  f (logb (1/2) 8) = -9 :=
begin
  intros h_odd h_f,
  sorry
end

end find_value_of_f_log_l594_594242


namespace car_speed_first_hour_l594_594404

theorem car_speed_first_hour (speed1 speed2 avg_speed : ℕ) (h1 : speed2 = 70) (h2 : avg_speed = 95) :
  (2 * avg_speed) = speed1 + speed2 → speed1 = 120 :=
by
  sorry

end car_speed_first_hour_l594_594404


namespace part1_part2_l594_594951

noncomputable def f (x a : ℝ) : ℝ := exp x - sin x - cos x - (1 / 2) * a * x^2
noncomputable def f' (x a : ℝ) : ℝ := deriv (λ x, f x a) x

theorem part1 (a : ℝ) :
  let x := π / 4
  let slope := exp (π / 4) - π
  f' x a = slope ↔ a = 4 :=
begin
  sorry
end

theorem part2 (a : ℝ) :
  (∀ x ∈ Iio 1, f' x a ≥ log (1 - x)) ↔ a = 3 :=
begin
  sorry
end

end part1_part2_l594_594951


namespace stratified_sampling_group_D_l594_594823

theorem stratified_sampling_group_D :
    let total_districts := 38
    let group_D_districts := 8
    let total_sampled := 9
    (group_D_districts * total_sampled) / total_districts = 2 :=
by
  let total_districts := 38
  let group_D_districts := 8
  let total_sampled := 9
  have h : (group_D_districts * total_sampled) / total_districts = 2
  from sorry
  exact h

end stratified_sampling_group_D_l594_594823


namespace distribution_of_learning_machines_l594_594641

theorem distribution_of_learning_machines :
  let machines := 6
  let people := 4
  let total_arrangements := 1560
  (∃ distrib : Vector Nat people, 
    sum distrib = machines ∧
    ∀ n, n ∈ distrib → n ≥ 1) → 
  num_possible_arrangements machines people = total_arrangements := 
by
  sorry

end distribution_of_learning_machines_l594_594641


namespace beetles_day4_l594_594846

/-
  Anton has two types of beetles, Type X and Type Y, in his enclosure. The beetles are indistinguishable by appearance, but Anton knows that the population of Type X triples every day, while the population of Type Y quadruples every day. On Day 0, there are 40 beetles in total. On Day 4, Anton counts 4160 beetles. How many of these beetles are of Type X?
-/

theorem beetles_day4 (x y : ℝ) (hx : 3^4 * x = 81 * x) (hy : 4^4 * y = 256 * y) 
  (h1 : x + y = 40) (h2 : 81 * x + 256 * y = 4160) : x = 35 :=
by {
  have h3 : y = 40-x, sorry,
  rw h3 at h2,
  linarith,
}

end beetles_day4_l594_594846


namespace cost_of_each_soda_l594_594652

def initial_money := 20
def change_received := 14
def number_of_sodas := 3

theorem cost_of_each_soda :
  (initial_money - change_received) / number_of_sodas = 2 :=
by
  sorry

end cost_of_each_soda_l594_594652


namespace quadratic_equation_of_list_l594_594766

def is_quadratic (eq : Polynomial ℝ) : Prop :=
  eq.degree = 2

def equations : List (Polynomial ℝ) :=
  [3 * Polynomial.x + Polynomial.C 1,
   Polynomial.x - 2 * Polynomial.x ^ 3 - Polynomial.C 3,
   Polynomial.x ^ 2 - Polynomial.C 5,
   2 * Polynomial.x + Polynomial.C 1 / Polynomial.x - Polynomial.C 3]

theorem quadratic_equation_of_list : 
  ∃ (eq : Polynomial ℝ), 
    eq ∈ equations ∧ is_quadratic eq ∧ 
    ∀ eq' ∈ equations, eq' ≠ eq → ¬ is_quadratic eq' :=
by
  sorry

end quadratic_equation_of_list_l594_594766


namespace sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l594_594023

theorem sufficient_but_not_necessary_condition 
  (x : ℝ) : (x + 1) * (x - 2) > 0 → x > 2 :=
by sorry

theorem converse_not_true 
  (x : ℝ) : x > 2 → (x + 1) * (x - 2) > 0 :=
by sorry

theorem cond_x_gt_2_iff_sufficient_not_necessary 
  (x : ℝ) : (x > 2 → (x + 1) * (x - 2) > 0) ∧ 
            ((x + 1) * (x - 2) > 0 → x > 2) :=
by sorry

end sufficient_but_not_necessary_condition_converse_not_true_cond_x_gt_2_iff_sufficient_not_necessary_l594_594023


namespace trapezoid_area_ratio_eq_one_l594_594080

theorem trapezoid_area_ratio_eq_one
  (AD AO OB BC AB DO OC : ℝ)
  (h1 : AD = 13) (h2 : AO = 13) (h3 : OB = 13) (h4 : BC = 13)
  (h5 : AB = 15) (h6 : DO = 15) (h7 : OC = 15)
  (X : ℝ × ℝ) (Y : ℝ × ℝ)
  (midX : X = ((AD / 2), 0))  -- X is midpoint of AD
  (midY : Y = ((BC / 2), 0))  -- Y is midpoint of BC
  (angle_OPA_deg : ℝ)
  (h8 : angle_OPA_deg = 30) :
  (area (trapezoid AB Y X X) / area (trapezoid X Y CD X)) = 1 := 
begin
  sorry
end

end trapezoid_area_ratio_eq_one_l594_594080


namespace speed_conversion_l594_594175

theorem speed_conversion (s : ℝ) (h1 : s = 1 / 3) : s * 3.6 = 1.2 := by
  -- Proof follows from the conditions given
  sorry

end speed_conversion_l594_594175


namespace find_a_modulus_z2_l594_594228

-- Define the conditions
noncomputable def z1 (a : ℝ) : ℂ := 1 + a * complex.I
-- a < 0
def a_condition (a : ℝ) : Prop := a < 0
-- z1^2 is purely imaginary (i.e., the real part of z1^2 is zero)
def purely_imaginary (z : ℂ) : Prop := z.re = 0
-- define the second complex number z2
noncomputable def z2 (a : ℝ) : ℂ := (z1 a) / (1 + complex.I) + 2

-- Prove that a = -1
theorem find_a (a : ℝ) : a_condition a → purely_imaginary ((z1 a)^2) → a = -1 := 
  by sorry

-- Prove that |z2| = sqrt 5
theorem modulus_z2 (a : ℝ) : a_condition a → purely_imaginary ((z1 a)^2) → complex.abs (z2 a) = Real.sqrt 5 := 
  by sorry

end find_a_modulus_z2_l594_594228


namespace sum_of_valid_a_l594_594528

theorem sum_of_valid_a : 
  ∑ k in ({k : ℤ | |k| ≤ 15 ∧ ∀ x ∈ (Icc 2 3), (4 * x - k - 4) / (6 * x + k - 12) ≤ 0}).toFinset = -7 :=
begin
  sorry
end

end sum_of_valid_a_l594_594528


namespace not_function_from_A_to_B_l594_594910

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {y | 1 ≤ y ∧ y ≤ 4}
def f (x : ℝ) : ℝ := 1 / x

theorem not_function_from_A_to_B :
  ¬(∀ x ∈ A, f x ∈ B) :=
by
  let ⟨x_min, x_max⟩ := ⟨1, 2⟩
  let ⟨f_min, f_max⟩ := ⟨f x_max, f x_min⟩
  have h1 : f_min = f 2 := rfl
  have h2 : f_max = f 1 := rfl
  have h3 : f_min = 1 / 2 := h1
  have h4 : f_max = 1 := h2
  have not_in_B_min : ¬(1 / 2 ∈ B) := by
    let h1 : (1 / 2) < 1 := by norm_num
    intro h2
    linarith
  exact ⟨x_max, ⟨by norm_num, by norm_num⟩, by simp [f, not_in_B_min]⟩

end not_function_from_A_to_B_l594_594910


namespace operations_to_equal_numbers_l594_594078

theorem operations_to_equal_numbers :
  ∃ n : ℕ, (515 - 11 * n) = (53 + 11 * n) :=
by {
  use 21,
  sorry
}

end operations_to_equal_numbers_l594_594078


namespace train_length_l594_594776

theorem train_length (S L : ℝ)
  (h1 : L = S * 11)
  (h2 : L + 120 = S * 22) : 
  L = 120 := 
by
  -- proof goes here
  sorry

end train_length_l594_594776


namespace arithmetic_expression_equality_l594_594154

theorem arithmetic_expression_equality :
  ( ( (4 + 6 + 5) * 2 ) / 4 - ( (3 * 2) / 4 ) ) = 6 :=
by sorry

end arithmetic_expression_equality_l594_594154


namespace cos_A_eq_a_eq_l594_594640

-- Defining the problem conditions:
variables {A B C a b c : ℝ}
variable (sin_eq : Real.sin (B + C) = 3 * Real.sin (A / 2) ^ 2)
variable (area_eq : 1 / 2 * b * c * Real.sin A = 6)
variable (sum_eq : b + c = 8)
variable (bc_prod_eq : b * c = 13)
variable (cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)

-- Proving the statements:
theorem cos_A_eq : Real.cos A = 5 / 13 :=
sorry

theorem a_eq : a = 3 * Real.sqrt 2 :=
sorry

end cos_A_eq_a_eq_l594_594640


namespace sin_double_angle_l594_594568

theorem sin_double_angle (x : ℝ) (h : sin (x - π / 4) = 2 / 3) : sin (2 * x) = 1 / 9 :=
sorry

end sin_double_angle_l594_594568


namespace duplicate_vertex_degrees_in_graph_l594_594413

theorem duplicate_vertex_degrees_in_graph (G : SimpleGraph (Fin 2000))
  (deg : Fin 2000 → Nat)
  (h_deg : ∀ v, deg v = G.degree v) 
  (two_equal_degrees : ∃ v1 v2 : Fin 2000, v1 ≠ v2 ∧ deg v1 = deg v2) :
  ∃ k, (k = 999 ∨ k = 1000) ∧ ∃ v1 v2 : Fin 2000, v1 ≠ v2 ∧ deg v1 = k ∧ deg v2 = k :=
by sorry

end duplicate_vertex_degrees_in_graph_l594_594413


namespace find_subtracted_value_l594_594485

theorem find_subtracted_value (N : ℕ) (V : ℕ) (hN : N = 2976) (h : (N / 12) - V = 8) : V = 240 := by
  sorry

end find_subtracted_value_l594_594485


namespace round_to_nearest_hundredth_l594_594702

-- Definitions
def repeating_decimal := "37.373737..."
def third_digit := 7
def second_digit := 3

-- Rounding rule for third digit
theorem round_to_nearest_hundredth (x : ℝ) (h : x = 37.373737...) :
  37.373737... = 37.38 := by
  have : third_digit > 5 := by norm_num
  have : second_digit = 4 := by norm_num
  sorry

end round_to_nearest_hundredth_l594_594702


namespace floor_abs_sum_eq_500_l594_594172

theorem floor_abs_sum_eq_500 (y : Fin 3012 → ℝ)
  (h : ∀ i : Fin 3012, y i + (i : ℝ) + 1 = 3 * (∑ i, y i) + 3013) :
  (Real.floor (| (∑ i, y i) |) = 500) := by
  sorry

end floor_abs_sum_eq_500_l594_594172


namespace percentage_of_discount_l594_594137

variable (C : ℝ) -- Cost Price of the Book

-- Conditions
axiom profit_with_discount (C : ℝ) : ∃ S_d : ℝ, S_d = C * 1.235
axiom profit_without_discount (C : ℝ) : ∃ S_nd : ℝ, S_nd = C * 2.30

-- Theorem to prove
theorem percentage_of_discount (C : ℝ) : 
  ∃ discount_percentage : ℝ, discount_percentage = 46.304 := by
  sorry

end percentage_of_discount_l594_594137


namespace ratio_of_combined_areas_l594_594856

theorem ratio_of_combined_areas (r1 r2 R : ℝ) (h1: r1 = 8) (h2: r2 = 6) (hR: R = 10) :
  let area_large_circle := π * (R^2)
  let area_semicircle1 := (1/2) * π * (r1^2)
  let area_semicircle2 := (1/2) * π * (r2^2)
  (area_semicircle1 + area_semicircle2) / area_large_circle = 1 / 2 := 
by
  rw [h1, h2, hR]
  let area_large_circle := π * (10^2)
  let area_semicircle1 := (1/2) * π * (8^2)
  let area_semicircle2 := (1/2) * π * (6^2)
  have h_area_semicircle1 : area_semicircle1 = 32 * π := by sorry
  have h_area_semicircle2 : area_semicircle2 = 18 * π := by sorry
  have h_combined_area : area_semicircle1 + area_semicircle2 = 50 * π := by sorry
  have h_area_large_circle : area_large_circle = 100 * π := by sorry
  rw [h_area_semicircle1, h_area_semicircle2, h_combined_area, h_area_large_circle]
  exact sorry

end ratio_of_combined_areas_l594_594856


namespace max_value_of_s_l594_594337

theorem max_value_of_s (p q r s : ℝ) (h1 : p + q + r + s = 10)
  (h2 : p * q + p * r + p * s + q * r + q * s + r * s = 22) :
  s ≤ (5 + Real.sqrt 93) / 2 :=
sorry

end max_value_of_s_l594_594337


namespace minimal_travel_path_l594_594138

noncomputable def euclidean_distance (p1 p2: ℝ × ℝ): ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

structure Triangle :=
  (A B C: ℝ × ℝ)
  (equilateral: euclidean_distance A B = euclidean_distance B C ∧ euclidean_distance B C = euclidean_distance C A)
  (side_length: euclidean_distance A B = 1)

def altitude (t: Triangle): ℝ :=
  (real.sqrt 3) / 2

def radius (t: Triangle): ℝ :=
  altitude t / 2

def covers_whole_region (t: Triangle) (P Q R: ℝ × ℝ): Prop :=
  true -- Placeholder for the actual condition ensuring whole region coverage.

theorem minimal_travel_path (t: Triangle) (A: ℝ × ℝ) (r: ℝ) (X Y: ℝ × ℝ):
  t.A = A →
  radius t = r →
  covers_whole_region t A X Y →
  euclidean_distance A X + euclidean_distance X Y + euclidean_distance Y A = minimal_travel_distance :=
sorry

end minimal_travel_path_l594_594138


namespace select_three_half_planes_cover_whole_plane_l594_594037

theorem select_three_half_planes_cover_whole_plane (n : ℕ) (half_planes : Fin n → Set (Set ℝ)) :
  (∀ p : Set ℝ, (∃ i : Fin n, p ∈ half_planes i) → (p = ℝ)) →
  ∃ a b c : Fin n, (∃ (cover : Set ℝ), cover = (half_planes a ∪ half_planes b ∪ half_planes c) ∧ cover = ℝ) :=
sorry

end select_three_half_planes_cover_whole_plane_l594_594037


namespace coffee_cooling_time_l594_594691

theorem coffee_cooling_time :
  ∀ (θ₀ θ₁ θ : ℝ) (t : ℝ), 
  θ₀ = 25 → θ₁ = 85 → θ = 65 → 
  (θ = θ₀ + (θ₁ - θ₀) * real.exp (-0.08 * t)) → 
  t = 5 :=
by
  intros θ₀ θ₁ θ t hθ₀ hθ₁ hθ heq
  -- Proof will be filled out here
  sorry

end coffee_cooling_time_l594_594691


namespace median_of_dataset_l594_594750

def dataset : List ℝ := [96, 112, 97, 108, 99, 104, 86, 98]

theorem median_of_dataset : (dataset.sort.nthLe 3 sorry + dataset.sort.nthLe 4 sorry) / 2 = 98.5 := by
  sorry

end median_of_dataset_l594_594750


namespace find_principal_l594_594378

theorem find_principal (CI SI : ℝ) (hCI : CI = 11730) (hSI : SI = 10200)
  (P R : ℝ)
  (hSI_form : SI = P * R * 2 / 100)
  (hCI_form : CI = P * (1 + R / 100)^2 - P) :
  P = 34000 := by
  sorry

end find_principal_l594_594378


namespace not_exist_k_chords_l594_594799

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 10 * x

def point_inside_circle (x y : ℝ) (hx : circle_eq x y) (px py : ℝ) : Prop := (px, py) = (5, 3)

def chord_lengths_eq (a_1 a_k : ℝ) (d : ℝ) (k : ℕ) : Prop := 
  a_1 = 8 ∧ a_k = 10 ∧ (1 / 3 <= d ∧ d <= 1 / 2) ∧ d = 2 / (k - 1)

theorem not_exist_k_chords (k : ℕ) :
  ∃ (x y : ℝ), circle_eq x y →
  ∃ (px py : ℝ), point_inside_circle x y (λ x y, rfl) px py →
  ∃ (a_1 a_k : ℝ), ∃ (d : ℝ), chord_lengths_eq a_1 a_k d k  →
  k ≠ 4 :=
by
  sorry

end not_exist_k_chords_l594_594799


namespace cat_replacement_ratio_l594_594076

-- Defining initial conditions and their relationships
theorem cat_replacement_ratio (cats_initial : ℕ) (adopted_fraction : ℚ) (dogs_mult_cats : ℕ) 
  (total_animals : ℕ) (cats_replaced : ℕ) :
  cats_initial = 15 →
  adopted_fraction = 1/3 →
  dogs_mult_cats = 2 →
  total_animals = 60 →
  cats_replaced = 15 →
  (cats_initial - (adopted_fraction * cats_initial).natAbs + cats_replaced) = cats_initial →
  dogs_mult_cats * cats_initial = 2 * cats_initial →
  ∀ adopted_cats : ℕ, adopted_cats = (adopted_fraction * cats_initial).natAbs →
  (cats_replaced : adopted_cats) = 3 : 1 :=
sorry

end cat_replacement_ratio_l594_594076


namespace eval_expression_l594_594395

theorem eval_expression :
  ((-2 : ℤ) ^ 3 : ℝ) ^ (1/3 : ℝ) - (-1 : ℤ) ^ 0 = -3 := by
  sorry

end eval_expression_l594_594395


namespace tan_half_angle_l594_594565

theorem tan_half_angle (α : ℝ) (h1 : Real.sin α + Real.cos α = 1 / 5)
  (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -1 / 3 :=
sorry

end tan_half_angle_l594_594565


namespace linear_combination_value_l594_594857

theorem linear_combination_value (x y : ℝ) (h₁ : 2 * x + y = 8) (h₂ : x + 2 * y = 10) :
  8 * x ^ 2 + 10 * x * y + 8 * y ^ 2 = 164 :=
sorry

end linear_combination_value_l594_594857


namespace find_x_l594_594070

-- Define the conditions
def condition : Prop :=
  (4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470)

-- State the theorem to prove the value of x
theorem find_x (x : ℝ) (h : condition) : x = 13.26 :=
sorry

end find_x_l594_594070


namespace sum_exterior_angles_regular_decagon_l594_594409

theorem sum_exterior_angles_regular_decagon : 
  ∀ (P : Type) [add_comm_group P] [modular_lattice P] [h : ∀ (p q : P), p ≤ q ∨ q ≤ p], 
  (∀ (n : ℕ), n = 10 → ∀ (p : P), ∃ (exterior_angle_sum : P), exterior_angle_sum = 360) := 
begin
  sorry
end

end sum_exterior_angles_regular_decagon_l594_594409


namespace range_of_m_l594_594914

theorem range_of_m (α m : ℝ) (hα : 0 < α ∧ α < π / 2) (h : sqrt 3 * sin α + cos α = m) : 
  1 < m ∧ m ≤ 2 :=
begin
  sorry
end

end range_of_m_l594_594914


namespace last_number_nth_row_sum_of_nth_row_position_of_2008_l594_594002

theorem last_number_nth_row (n : ℕ) : 
  ∃ last_number, last_number = 2^n - 1 := by
  sorry

theorem sum_of_nth_row (n : ℕ) : 
  ∃ sum_nth_row, sum_nth_row = 2^(2*n-2) + 2^(2*n-3) - 2^(n-2) := by
  sorry

theorem position_of_2008 : 
  ∃ (row : ℕ) (position : ℕ), row = 11 ∧ position = 2008 - 2^10 + 1 :=
  by sorry

end last_number_nth_row_sum_of_nth_row_position_of_2008_l594_594002


namespace remainder_polynomial_l594_594486

theorem remainder_polynomial (p : ℚ → ℚ) (s : ℚ → ℚ) (h1 : p 1 = 2) (h2 : p 3 = -4) (h3 : p (-2) = 5)
  (h4 : ∀ x, ∃ q : ℚ → ℚ, p x = (x - 1) * (x - 3) * (x + 2) * q x + s x)
  (h_s : s = λ x, - 0.16 * x^2 - 0.76 * x + 2.92) :
  s (-1) = 3.52 := 
by {
    sorry
}

end remainder_polynomial_l594_594486


namespace similar_triangles_iff_l594_594011

variables {a b c a' b' c' : ℂ}

theorem similar_triangles_iff :
  (∃ (z w : ℂ), a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔
  a' * (b - c) + b' * (c - a) + c' * (a - b) = 0 :=
sorry

end similar_triangles_iff_l594_594011


namespace math_problem_l594_594929

def arithmetic_sequence (a d : ℕ → ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d

def geometric_sequence (b q : ℕ → ℕ) (n : ℕ) : ℕ := q^(n - 1)

def sum_sequence (S : ℕ → ℕ) (n : ℕ) (f : ℕ → ℕ) : ℕ := 
  (n * (2 * f(1) + (n - 1) * (f(2) - f(1)))) / 2

theorem math_problem (a b : ℕ → ℕ) (T : ℕ → ℕ) 
  (d q : ℕ) 
  (h_arith : ∀ n, a n = 3*n - 2)
  (h_geom : ∀ n, b n = 2^(n - 1)) 
  (Sn_cond : sum_sequence 35 5 a) 
  (an_bn_cond : a 6 = b 5) :
  (∀ n, a n = 3*n - 2) ∧ (∀ n, b n = 2^(n - 1)) ∧ 
  (∀ n, T n = (3*n^2 - n) / 2 + 2^(n + 1) - 2) :=
  sorry

end math_problem_l594_594929


namespace total_area_of_hexagon_and_triangle_l594_594806

-- Define the regular hexagon side length
def side_length : ℝ := 3

-- Define the length of the sides of hexagon ABCDEF and midpoints G, H, I
def hexagon_area (s : ℝ) : ℝ := 3 * s^2 * Real.sqrt 3 / 2

-- Area of equilateral triangle with side length 3/2
def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Total area calculation
def total_area (s : ℝ) : ℝ := hexagon_area side_length + equilateral_triangle_area (s / 2)

-- Theorem stating the total area is as calculated
theorem total_area_of_hexagon_and_triangle :
  total_area side_length = (225 * Real.sqrt 3) / 16 :=
by
  sorry

end total_area_of_hexagon_and_triangle_l594_594806


namespace chord_length_alpha_135_chord_equation_bisected_by_P0_l594_594918

-- Definitions for the conditions given in the problem
def circle (x y : ℝ) : Prop := x^2 + y^2 = 8
def point_P0 : ℝ × ℝ := (-1, 2)
def alpha := 135

-- Prove length of chord AB when alpha = 135 degrees
theorem chord_length_alpha_135 (A B : ℝ × ℝ) :
  (A = (-7/5, -8/5)) →
  (B = (5/5, 14/5)) →
  ∃ (len : ℝ), len = sqrt 30 := by
    sorry

-- Prove equation of line AB when chord AB is bisected by point P_0
theorem chord_equation_bisected_by_P0 (A B : ℝ × ℝ) :
  ∃ (m : ℝ), m = point_P0 ∧
  ∀ (x y : ℝ), circle x y →
  (y = (1/2) * x - 5/2) := by
    sorry

end chord_length_alpha_135_chord_equation_bisected_by_P0_l594_594918


namespace hyperbola_vertex_distance_l594_594893

theorem hyperbola_vertex_distance :
  (∀ a b x y : ℝ, a^2 = 121 ∧ b^2 = 49 ∧ (x^2 / a^2) - (y^2 / b^2) = 1) →
  (distance = 2 * real.sqrt 121) :=
by
  sorry

end hyperbola_vertex_distance_l594_594893


namespace sin_diff_acute_angles_l594_594238

theorem sin_diff_acute_angles (α β : ℝ) 
  (hα1 : 0 < α ∧ α < π / 2) 
  (hβ1 : 0 < β ∧ β < π / 2) 
  (h1 : sin α = 4 / 5) 
  (h2 : cos β = 5 / 13) : 
  sin (β - α) = 16 / 65 := 
by 
  sorry

end sin_diff_acute_angles_l594_594238


namespace find_y_at_x8_l594_594973

-- Conditions
def y_at_x (k : ℝ) (x : ℝ) : ℝ := k * x^(1/3 : ℝ)

-- Given that y = 4sqrt(3) at x = 125, we find the value of k
def k_value : ℝ :=
  let y := 4 * Real.sqrt 3
  let x := 125
  y / (x^(1/3 : ℝ))

-- Prove that y = 8sqrt(3)/5 when x = 8
theorem find_y_at_x8 : y_at_x k_value 8 = 8 * Real.sqrt 3 / 5 := by
  sorry

end find_y_at_x8_l594_594973


namespace integral_sequence_and_divisibility_l594_594062

theorem integral_sequence_and_divisibility (k : ℕ) (h_pos : 0 < k) :
  (∀ n : ℕ, ∃ (a : ℕ), a_n = a) ∧ (∀ n : ℕ, 2 * k ∣ a_{2 * n}) :=
by
  -- Definitions of sequence and conditions
  let a : ℕ → ℕ := λ n, sorry 
  let a_0 := 0
  let recurrence_relation := λ n, a (n + 1) = k * a n + Math.sqrt (k^2 - 1 * a n^2 + 1)
  -- Initial conditions and the proof will be completed
  sorry

end integral_sequence_and_divisibility_l594_594062


namespace color_of_last_bead_l594_594317

-- Define the pattern of beads
inductive Bead
| red
| orange
| yellow
| green
| blue

open Bead

-- Define the function for the color sequence pattern
def beadPattern : ℕ → Bead
| 0 := red
| 1 := orange
| 2 := yellow
| 3 := yellow
| 4 := yellow
| 5 := green
| 6 := blue
| (n + 7) := beadPattern n

-- The problem is to show that the 85th bead is red
theorem color_of_last_bead :
  beadPattern 84 = red := -- since indices are zero-based, the 85th bead is at index 84
sorry

end color_of_last_bead_l594_594317


namespace blue_bordered_area_on_outer_sphere_l594_594828

theorem blue_bordered_area_on_outer_sphere :
  let r := 1 -- cm
  let r1 := 4 -- cm
  let r2 := 6 -- cm
  let A_inner := 27 -- cm^2
  let h := A_inner / (2 * π * r1)
  let A_outer := 2 * π * r2 * h
  A_outer = 60.75 := sorry

end blue_bordered_area_on_outer_sphere_l594_594828


namespace problem_solution_l594_594839

theorem problem_solution :
  ∃ f : ℝ → ℝ, (∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 < f x2) ∧ f = fun x ↦ 2^x :=
by
  use (fun x ↦ 2^x)
  sorry

end problem_solution_l594_594839


namespace pen_ratio_l594_594818

theorem pen_ratio (x y : ℕ) (h : 2 * x + y = 3 * y + 1.5 * x) : x = y / 4 :=
by
  sorry

end pen_ratio_l594_594818


namespace ellipse_properties_l594_594682

theorem ellipse_properties :
  let a := 2 in
  let b := √3 in
  let e := 1 / 2 in
  let major_axis_length := 4 in
  let O := (0, 0) in
  let x : ℝ := sorry in
  let y : ℝ := sorry in
  let M : ℝ × ℝ := sorry in
  let N : ℝ × ℝ := sorry in
  let AB : ℝ × ℝ := sorry in
  let MN : ℝ × ℝ := sorry in
  let k := √2 in
  a > b ∧ b > 0 ∧ (c := a / 2) ∧ b^2 = a^2 - c^2 ∧ 
  ( ∃ M N, ∀ x y, 
      (x^2 / a^2 + y^2 / b^2 = 1) → 
      (|O - M|^2 + |O - N|^2 = 4) → 
      slope_l = k ∧ 
      (∀ M N AB, |AB|^2 / |MN| = 4) ) :=
sorry

end ellipse_properties_l594_594682


namespace carpenters_needed_l594_594277

statement : Prop :=
  let original_carpenters := 8
  let original_chairs := 50
  let original_days := 10
  let new_chairs := 75
  let new_days := 10
  let ratio := new_chairs / original_chairs
  (new_days = original_days) → (new_chairs / original_chairs) * original_carpenters = 12

theorem carpenters_needed (h: statement) : ∃ carpenters : ℕ, carpenters = 12 :=
by { exact ⟨12, sorry⟩ }

end carpenters_needed_l594_594277


namespace sammy_remaining_problems_l594_594356

def initial_fraction_problems : ℕ := 20
def initial_decimal_problems : ℕ := 55
def completed_fraction_problems : ℕ := 8
def completed_decimal_problems : ℕ := 12

def remaining_fraction_problems : ℕ := initial_fraction_problems - completed_fraction_problems
def remaining_decimal_problems : ℕ := initial_decimal_problems - completed_decimal_problems
def total_remaining_problems : ℕ := remaining_fraction_problems + remaining_decimal_problems

theorem sammy_remaining_problems :
  total_remaining_problems = 55 :=
by
  rw [remaining_fraction_problems, remaining_decimal_problems, total_remaining_problems]
  simp
  sorry

end sammy_remaining_problems_l594_594356


namespace dihedral_angle_at_apex_of_pyramid_l594_594621

-- Definitions and conditions according to part a)
def is_regular_quadrilateral_pyramid (P : Type) [plane_geometry P] (λ : Length P) : Prop :=
  ∃ a b l : P, a ≠ b ∧ l ≠ a ∧ l ≠ b ∧ λ (distance a b) ∧ λ (distance b l) ∧ λ (distance a l)

def center_circumscribed_sphere_on_surface_of_inscribed_sphere (P : Type) [plane_geometry P] (C I : P) : Prop :=
  ∃ r R : real, r < R ∧ distance C I = r ∧ distance C I = R

-- Theorem statement according to part c)
theorem dihedral_angle_at_apex_of_pyramid {P : Type} [plane_geometry P]
  (is_regular_quadrilateral_pyramid P) 
  (center_circumscribed_sphere_on_surface_of_inscribed_sphere P) :
  angle_apex_of_pyramid = π / 2 ∨ angle_apex_of_pyramid = π / 6 := 
sorry

end dihedral_angle_at_apex_of_pyramid_l594_594621


namespace length_of_major_axis_l594_594028

def ellipse_length_major_axis (a b : ℝ) : ℝ := 2 * a

theorem length_of_major_axis : ellipse_length_major_axis 4 1 = 8 :=
by
  unfold ellipse_length_major_axis
  norm_num

end length_of_major_axis_l594_594028


namespace find_wrong_observation_value_l594_594387

-- Defining the given conditions
def original_mean : ℝ := 36
def corrected_mean : ℝ := 36.5
def num_observations : ℕ := 50
def correct_value : ℝ := 30

-- Defining the given sums based on means
def original_sum : ℝ := num_observations * original_mean
def corrected_sum : ℝ := num_observations * corrected_mean

-- The wrong value can be calculated based on the difference
def wrong_value : ℝ := correct_value + (corrected_sum - original_sum)

-- The theorem to prove
theorem find_wrong_observation_value (h : original_sum = 1800) (h' : corrected_sum = 1825) :
  wrong_value = 55 :=
sorry

end find_wrong_observation_value_l594_594387


namespace vasya_made_mistake_l594_594835

theorem vasya_made_mistake : 
  ∀ (total_digits : ℕ), 
    total_digits = 301 → 
    ¬∃ (n : ℕ), 
      (n ≤ 9 ∧ total_digits = (n * 1)) ∨ 
      (10 ≤ n ∧ n ≤ 99 ∧ total_digits = (9 * 1) + ((n - 9) * 2)) ∨ 
      (100 ≤ n ∧ total_digits = (9 * 1) + (90 * 2) + ((n - 99) * 3)) := 
by 
  sorry

end vasya_made_mistake_l594_594835


namespace triangle_incircle_concurrence_proof_l594_594785

open Euclidean_geometry

noncomputable theory

def concurrent_lines_intriangle (A B C C' C1 B1 C2 A2 : Point) : Prop :=
  let I₁ := incircle (triangle.mk A B C')
  let I₂ := incircle (triangle.mk A C C')
  let I₃ := incircle (triangle.mk B C C') in
  (touches I₁ (line.mk A B)) ∧
  (touches I₂ (line.mk A B)) ∧ (touches I₂ (line.mk A C)) ∧
  (touches I₃ (line.mk A B)) ∧ (touches I₃ (line.mk B C)) ∧
  (collinear_points (line.mk B1 C1)) ∧ (collinear_points (line.mk A2 C2)) ∧
  (intersect_at_point (line.mk B1 C1) (line.mk A2 C2) (line.mk C C'))

theorem triangle_incircle_concurrence_proof (A B C C' C1 B1 C2 A2 : Point) 
  (h₁ : touches (incircle (triangle.mk A B C')) (line.mk A B))
  (h₂ : touches (incircle (triangle.mk A C C')) (line.mk A B))
  (h₃ : touches (incircle (triangle.mk A C C')) (line.mk A C))
  (h₄ : touches (incircle (triangle.mk B C C')) (line.mk A B))
  (h₅ : touches (incircle (triangle.mk B C C')) (line.mk B C))
  (h₆ : collinear_points (line.mk B1 C1))
  (h₇ : collinear_points (line.mk A2 C2))
  (h₈ : intersect_at_point (line.mk B1 C1) (line.mk A2 C2) (line.mk C C')) :
  concurrent_lines_intriangle A B C C' C1 B1 C2 A2 :=
sorry

end triangle_incircle_concurrence_proof_l594_594785


namespace tyler_meal_combinations_is_720_l594_594429

-- Required imports for permutations and combinations
open Nat
open BigOperators

-- Assumptions based on the problem conditions
def meat_options  := 4
def veg_options := 4
def dessert_options := 5
def bread_options := 3

-- Using combinations and permutations for calculations
def comb(n k : ℕ) := Nat.choose n k
def perm(n k : ℕ) := n.factorial / (n - k).factorial

-- Number of ways to choose meals
def meal_combinations : ℕ :=
  meat_options * (comb veg_options 2) * dessert_options * (perm bread_options 2)

theorem tyler_meal_combinations_is_720 : meal_combinations = 720 := by
  -- We provide proof later; for now, put sorry to skip
  sorry

end tyler_meal_combinations_is_720_l594_594429


namespace binary_to_octal_l594_594176

theorem binary_to_octal (b : string) (h : b = "101110") : convert_to_octal b = "56" :=
sorry

end binary_to_octal_l594_594176


namespace parallel_vectors_x_value_l594_594220

theorem parallel_vectors_x_value (x : ℝ) : 
  (2, 3) = λ x, (2 / 3 = x / -6) → x = -4 := 
by
    intro x
    sorry

end parallel_vectors_x_value_l594_594220


namespace cos_identity_l594_594274

variable {θ : ℝ} {x : ℂ}

theorem cos_identity (h_theta : 0 < θ ∧ θ < Real.pi)
  (h_x : x + x⁻¹ = 2 * Real.cos θ) (n : ℕ) (hn: 0 < n) : 
  x^n + x^(-n) = 2 * Complex.cos (n * θ) := 
sorry

end cos_identity_l594_594274


namespace floor_e_eq_two_l594_594880

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end floor_e_eq_two_l594_594880


namespace killing_two_birds_is_random_event_l594_594767

-- Define the idioms
inductive Idioms
| catching_a_turtle_in_a_jar
| killing_two_birds_with_one_stone
| fishing_for_the_moon_in_the_water
| dripping_water_wears_away_a_stone

-- Definition of random event characteristic
def is_random_event : Idioms → Prop
| Idioms.catching_a_turtle_in_a_jar := false
| Idioms.killing_two_birds_with_one_stone := true
| Idioms.fishing_for_the_moon_in_the_water := false
| Idioms.dripping_water_wears_away_a_stone := false

-- The proof statement
theorem killing_two_birds_is_random_event : is_random_event Idioms.killing_two_birds_with_one_stone = true :=
by
  sorry

end killing_two_birds_is_random_event_l594_594767


namespace center_circle_tangent_to_angle_sides_l594_594233

-- Define the setting of the problem: a given angle and the condition of being tangent to both sides
structure Angle (α β γ : ℝ) :=
(angle_eq_sum : α + γ = β)

-- Define the problem: For a given angle, the center of a circle tangent to both sides lies on the bisector
theorem center_circle_tangent_to_angle_sides (α β γ : ℝ) (a : Angle α β γ) :
  ∃ (C : ℝ × ℝ), -- Assume C is the center of the circle being (x, y) coordinates
  ∀ (r : ℝ), -- r is the radius of the circle
  is_tangent (C, r) α ∧ is_tangent (C, r) γ →
  is_on_angle_bisector C α γ :=
sorry

end center_circle_tangent_to_angle_sides_l594_594233


namespace ratio_of_heights_of_cones_l594_594134

-- Given conditions
variables (R : ℝ) -- Radius of the original circle
-- r1 is the radius of the base of the first cone
def r1 := R / 4
-- r2 is the radius of the base of the second cone
def r2 := 3 * R / 4
-- h1 is the height of the first cone using Pythagorean theorem
def h1 := sqrt (R^2 - (r1)^2)
-- h2 is the height of the second cone using Pythagorean theorem
def h2 := sqrt (R^2 - (r2)^2)

-- Theorem to prove the ratio of the heights of the cones
theorem ratio_of_heights_of_cones : h1 / h2 = sqrt (15) / sqrt (7) :=
by
  -- Use the definitions of h1 and h2
  sorry

end ratio_of_heights_of_cones_l594_594134


namespace sum_sin_squared_l594_594855

theorem sum_sin_squared (k : ℕ) (h : k ∈ Finset.range 9) :
  ∑ k in Finset.range 9, (Real.sin (k * Real.pi / 4)) ^ 4 = 4 :=
by
  sorry

end sum_sin_squared_l594_594855


namespace fraction_revenue_large_is_1_10_l594_594810

-- Definition of conditions
variables {x p : ℝ} (hx : x > 0) (hp : p > 0)
def small_cups (x : ℝ) := (3 / 5) * x
def large_cups (x : ℝ) := x - small_cups x
def price_small (p : ℝ) := p
def price_large (p : ℝ) := (1 / 6) * p

-- Total revenue from small cups
def revenue_small (x p : ℝ) := small_cups x * p

-- Total revenue from large cups
def revenue_large (x p : ℝ) := large_cups x * (1 / 6) * p

-- Total revenue
def total_revenue (x p : ℝ) := revenue_small x p + revenue_large x p

-- The fraction of total revenue from large cups
def fraction_large_revenue (x p : ℝ) := revenue_large x p / total_revenue x p

theorem fraction_revenue_large_is_1_10 :
  fraction_large_revenue x p = 1 / 10 := by
  sorry

end fraction_revenue_large_is_1_10_l594_594810


namespace courtyard_length_l594_594965

theorem courtyard_length (area_paving_stone : ℝ) (num_paving_stones : ℕ) (width_courtyard : ℝ) (total_area : ℝ) (length_courtyard : ℝ) :
  (area_paving_stone = 2) → 
  (num_paving_stones = 240) → 
  (width_courtyard = 16) → 
  (total_area = num_paving_stones * area_paving_stone) → 
  (length_courtyard = total_area / width_courtyard) → 
  length_courtyard = 30 := 
by 
  intros h1 h2 h3 h4 h5 
  rw [h1, h2, h3] at h4 
  simp at h4
  rw h4 at h5
  simp at h5
  exact h5


end courtyard_length_l594_594965


namespace each_serving_requires_1_5_apples_l594_594647

theorem each_serving_requires_1_5_apples 
  (guest_count : ℕ) (pie_count : ℕ) (servings_per_pie : ℕ) (apples_per_guest : ℝ) 
  (h_guest_count : guest_count = 12)
  (h_pie_count : pie_count = 3)
  (h_servings_per_pie : servings_per_pie = 8)
  (h_apples_per_guest : apples_per_guest = 3) :
  (apples_per_guest * guest_count) / (pie_count * servings_per_pie) = 1.5 :=
by
  sorry

end each_serving_requires_1_5_apples_l594_594647


namespace positive_solution_of_equation_l594_594541

theorem positive_solution_of_equation :
  ∃ y : ℝ, y > 0 ∧ (sqrt(y + sqrt(y + sqrt(y + ...))) = sqrt(y * sqrt(y * sqrt(y * ...)))) ∧ y^2 = 4 :=
by
  sorry

end positive_solution_of_equation_l594_594541


namespace original_length_in_meters_l594_594355

-- Conditions
def erased_length : ℝ := 10 -- 10 cm
def remaining_length : ℝ := 90 -- 90 cm

-- Question: What is the original length of the line in meters?
theorem original_length_in_meters : (remaining_length + erased_length) / 100 = 1 := 
by 
  -- The proof is omitted
  sorry

end original_length_in_meters_l594_594355


namespace basketball_team_initial_games_l594_594120

theorem basketball_team_initial_games (G W : ℝ) 
  (h1 : W = 0.70 * G) 
  (h2 : W + 2 = 0.60 * (G + 10)) : 
  G = 40 :=
by
  sorry

end basketball_team_initial_games_l594_594120


namespace range_f_l594_594578

def f (ω x : ℝ) := sin (ω * x) ^ 2 + sqrt 3 * sin (ω * x) * sin (ω * x + π / 2)

theorem range_f (ω : ℝ) (hω : ω > 0) (hT : ∀ x, f ω (x + π / ω) = f ω x) : 
  set.range (λ x, f ω x) = set.Icc (0 : ℝ) (3 / 2) :=
by
  sorry

end range_f_l594_594578


namespace area_of_triangle_calc_l594_594330

noncomputable def area_of_triangle {P F1 F2 : ℝ × ℝ} 
  (hP : P = (x, y) ∧ (x^2 / 5 + y^2 / 4 = 1)) 
  (hFoci : F1 = (c, d) ∧ F2 = (-c, -d)) 
  (hAngle : ∠ (F1, P) (F2, P) = 30) : ℝ := 
8 - 4 * Real.sqrt 3

theorem area_of_triangle_calc (P F1 F2 : ℝ × ℝ) 
  (hP : P = (x, y) ∧ (x^2 / 5 + y^2 / 4 = 1)) 
  (hFoci : F1 = (c, d) ∧ F2 = (-c, -d)) 
  (hAngle : ∠ (F1, P) (F2, P) = 30) : 
  area_of_triangle hP hFoci hAngle = 8 - 4 * Real.sqrt 3 := 
sorry

end area_of_triangle_calc_l594_594330


namespace symmetric_points_sum_l594_594600

theorem symmetric_points_sum (n m : ℤ) 
  (h₁ : (3 : ℤ) = m)
  (h₂ : n = (-5 : ℤ)) : 
  m + n = (-2 : ℤ) := 
by 
  sorry

end symmetric_points_sum_l594_594600


namespace manager_salary_l594_594376

variable {M : ℕ} -- manager's salary

def avg_salary_24_employees (s : ℕ) := s / 24 = 1500
def new_avg_salary_with_manager (s : ℕ) := (s + M) / 25 = 1900

theorem manager_salary (s : ℕ) (h1 : avg_salary_24_employees s) (h2 : new_avg_salary_with_manager s) : M = 11500 := by
  sorry

end manager_salary_l594_594376


namespace tangent_intersects_on_segment_l594_594007

noncomputable def midpoint {A B : Point} : Point := sorry
noncomputable def tangent_line (S : Circle) (P : Point) : Line := sorry
noncomputable def intersection (l1 l2 : Line) : Point := sorry

theorem tangent_intersects_on_segment
    (A B : Point) (S S1 : Circle) (M C D K L : Point)
    (tangent_A : tangent_line S A)
    (tangent_B : tangent_line S B)
    (midpoint_M : M = midpoint A B)
    (AB_intersects_D : intersects (line_through_points A B) S1 = D)
    (S1_intersects_S : ∃ K L, K ≠ L ∧ K ∈ S ∧ K ∈ S1 ∧ L ∈ S ∧ L ∈ S1)
    (tangent_K : tangent_line S K)
    (tangent_L : tangent_line S L)
    (intersection_C : intersection tangent_A tangent_B = C)
    (D_on_AB : D ∈ segment A B)
    (segment_intersects_CD : ∀ P : Point, intersects P (line_through_points C D)) :
    ∃ P : Point, (intersection (tangent_line S K) (tangent_line S L) = P ∧ P ∈ segment C D) :=
begin
  sorry
end

end tangent_intersects_on_segment_l594_594007


namespace grain_output_l594_594350

-- Define the condition regarding grain output.
def premier_goal (x : ℝ) : Prop :=
  x > 1.3

-- The mathematical statement that needs to be proved, given the condition.
theorem grain_output (x : ℝ) (h : premier_goal x) : x > 1.3 :=
by
  sorry

end grain_output_l594_594350


namespace smallest_third_term_of_arithmetic_sequence_l594_594739

theorem smallest_third_term_of_arithmetic_sequence :
  ∃ (a d : ℕ) (a_pos : 0 < a) (d_pos : 0 ≤ d), 
  (4 * a + 6 * d = 50) ∧ (a + 2 * d = 16) :=
begin
  sorry
end

end smallest_third_term_of_arithmetic_sequence_l594_594739


namespace parameter_values_exactly_three_solutions_l594_594907

theorem parameter_values_exactly_three_solutions :
  ∃ a : ℝ, (a = 1 / 2) ∨ (a = 1) ∨ (a = 3 / 2) ∧
    (∀ x : ℝ, 4^|x - a| * log 1/3 (x^2 - 2*x + 4) + 2^(x^2 - 2*x) * log (sqrt 3) (2*|x - a| + 3) = 0 → 
    -- exactly three solutions) := 
  sorry

end parameter_values_exactly_three_solutions_l594_594907


namespace equivalent_proof_problem_l594_594943

-- Define the context
variable (x y m n k : ℝ)

-- Define the curve equation and conditions
def curve_equation (x y : ℝ) (m n : ℕ) : Prop := m * x ^ 2 + n * y ^ 2 = 1

-- Points A and B
def point_A := (x = sqrt 2 / 4 ∧ y = sqrt 2 / 2)
def point_B := (x = sqrt 6 / 6 ∧ y = sqrt 3 / 3)

-- The line equation and its passing through the point
def line_equation (x y k : ℝ) : Prop := y = k * x + sqrt 3 / 2

-- The expression for the dot product to be zero
def dot_product_zero (x₁ y₁ x₂ y₂ m n : ℝ) : Prop :=
  (sqrt m * x₁) * (sqrt m * x₂) + (sqrt n * y₁) * (sqrt n * y₂) = 0

-- The equivalent proof problem
theorem equivalent_proof_problem :
  (curve_equation (sqrt 2 / 4) (sqrt 2 / 2) 4 1 ∧ curve_equation (sqrt 6 / 6) (sqrt 3 / 3) 4 1) ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ), (line_equation x y k ∧ dot_product_zero x₁ y₁ x₂ y₂ 4 1) →
  k = sqrt 2 ∨ k = -sqrt 2 := sorry

end equivalent_proof_problem_l594_594943


namespace distance_equidistant_point_l594_594189

noncomputable def distance_from_O_to_all_points (A B C P Q O : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space Q] [metric_space O] : ℝ :=
if h : (∀ pt ∈ [A, B, C, P, Q], dist pt O = d) then d else 0

theorem distance_equidistant_point 
  (A B C P Q O : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space Q] [metric_space O] 
  (d₁ d₂ : ℝ)
  (h₁ : equilateral A B C) 
  (h₂ : side_length A B C = 300) 
  (h₃ : dist P A = 350) 
  (h₄ : dist P B = 350) 
  (h₅ : dist P C = 350) 
  (h₆ : dist Q A = 350) 
  (h₇ : dist Q B = 350) 
  (h₈ : dist Q C = 350) 
  (h₉ : dihedral_angle (triangle_plane P A B) (triangle_plane Q A B) = 60)
  (h₀ : ∀ pt ∈ [A, B, C, P, Q], dist pt O = d₁) :
  d₁ = 150 :=
sorry

end distance_equidistant_point_l594_594189


namespace basketball_not_table_tennis_l594_594986

-- Definitions and conditions
def total_students := 30
def like_basketball := 15
def like_table_tennis := 10
def do_not_like_either := 8
def like_both (x : ℕ) := x

-- Theorem statement
theorem basketball_not_table_tennis (x : ℕ) (H : (like_basketball - x) + (like_table_tennis - x) + x + do_not_like_either = total_students) : (like_basketball - x) = 12 :=
by
  sorry

end basketball_not_table_tennis_l594_594986


namespace c_in_terms_of_t_l594_594847

theorem c_in_terms_of_t (t a b c : ℝ) (h_t_ne_zero : t ≠ 0)
    (h1 : t^3 + a * t = 0)
    (h2 : b * t^2 + c = 0)
    (h3 : 3 * t^2 + a = 2 * b * t) :
    c = -t^3 :=
by
sorry

end c_in_terms_of_t_l594_594847


namespace area_of_ABF_is_correct_l594_594431

-- Given points on the coordinate plane
def A := (0 : ℝ, 0 : ℝ)
def B := (2 : ℝ, 0 : ℝ)
def C := (2 : ℝ, 2 : ℝ)
def D := (0 : ℝ, 2 : ℝ)
def E := (2 : ℝ, 1 : ℝ)

-- Define the lines BD and AE
def line_BD (x : ℝ) := -x + 2
def line_AE (x : ℝ) := (1 / 2) * x

-- Define the intersection point F of BD and AE
def F : ℝ × ℝ := 
  let x := (4 : ℝ) / 3
  let y := (1 : ℝ) / 2 * x
  (x, y)

-- Definition of the base of triangle AB
def base_AB := (B.1 - A.1 : ℝ) -- 2

-- The y-coordinate of point F gives the height of the triangle
def height_AF := F.2 -- y-coordinate of F

-- The area formula for triangle ABF
def area_triangle_ABF := (1 : ℝ) / 2 * base_AB * height_AF

-- Proof problem statement
theorem area_of_ABF_is_correct : 
  area_triangle_ABF = (2 : ℝ) / 3 :=
by
  sorry

end area_of_ABF_is_correct_l594_594431


namespace students_in_group_l594_594618

theorem students_in_group (H S H_union_S H_sub_B : ℕ) (h1 : H = 36) (h2 : S = 32) (h3 : H_union_S = 57) (h4 : H_sub_B = 25) :
  let B := H + S - H_union_S in
  let total_students := (H - B) + S in
  total_students = 57 :=
by 
  unfold total_students B
  rw [h1, h2, h3, h4]
  have B_val : B = 36 + 32 - 57 := rfl
  rw B_val
  have total : 57 - 11 + 32 = 57 := rfl
  exact total

end students_in_group_l594_594618


namespace monotonically_increasing_f_g_has_two_zeros_l594_594249

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * exp (-x) + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f a x) - a - 3

theorem monotonically_increasing_f (a : ℝ) : (∀ x, 1 < x → f' a x ≥ 0) → a ≥ 2 / exp 1 :=
sorry

theorem g_has_two_zeros (a : ℝ) : (a ≠ 0 ∧ ∃ x1 x2, x1 ≠ x2 ∧ g a x1 = 0 ∧ g a x2 = 0) → a > 0 :=
sorry

end monotonically_increasing_f_g_has_two_zeros_l594_594249


namespace proof_by_contradiction_example_l594_594440

theorem proof_by_contradiction_example (a b : ℝ) (h : a + b ≥ 0) : ¬ (a < 0 ∧ b < 0) :=
by
  sorry

end proof_by_contradiction_example_l594_594440


namespace man_total_distance_l594_594773

def total_distance_travelled (Vm Vr T: ℝ) : ℝ :=
  let upstream_speed := Vm - Vr
  let downstream_speed := Vm + Vr
  let D := (upstream_speed * downstream_speed * T) / 
           (upstream_speed + downstream_speed)
  2 * D

theorem man_total_distance : 
  total_distance_travelled 8 1.2 1 = 7.82 :=
by
  sorry

end man_total_distance_l594_594773


namespace geometric_solid_triangle_intersection_l594_594280

-- Define the geometric solid type
inductive GeometricSolid
  | cone
  | cylinder
  | pyramid
  | cube

-- Define the condition predicate for intersection being a triangle
def canIntersectAsTriangle (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.cone := True
  | GeometricSolid.cylinder := False
  | GeometricSolid.pyramid := True
  | GeometricSolid.cube := True

-- Theorem statement
theorem geometric_solid_triangle_intersection :
  canIntersectAsTriangle GeometricSolid.cone ∧
  ¬canIntersectAsTriangle GeometricSolid.cylinder ∧
  canIntersectAsTriangle GeometricSolid.pyramid ∧
  canIntersectAsTriangle GeometricSolid.cube :=
by
  sorry

end geometric_solid_triangle_intersection_l594_594280


namespace fib_mod_139_formula_l594_594091

open Function

noncomputable def fib_mod_139 : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => (fib_mod_139 (n + 1) + fib_mod_139 n) % 139

theorem fib_mod_139_formula (F : ℕ → ℕ) (hF0 : F 0 = 0) (hF1 : F 1 = 1) :
  (∀ n, F (n + 2) = (F (n + 1) + F n) % 139) →
  (∀ n, F n ≡ 58 * (76^n - 64^n) \mod 139) :=
begin
  intro h,
  have h₁ : 12^2 ≡ 5 [MOD 139], from by norm_num,
  have h₂ : (∃ (x1 x2 : ℕ), x1^2 - x1 - 1 ≡ 0 [MOD 139] ∧ x2^2 - x2 - 1 ≡ 0 [MOD 139] ∧ x1 ≠ x2), 
  from by {
    existsi [64, 76],
    split; norm_num,
    split; norm_num, 
    norm_num 
  },
  have h₃ : 12 * 58 ≡ 1 [MOD 139], from by norm_num,
  sorry,
end

end fib_mod_139_formula_l594_594091


namespace range_m_l594_594602

noncomputable def quadratic_discriminant_pos (m : ℝ) : Prop :=
  let Δ := 4 * m^2 - 4 * (2 - m)
  Δ > 0

noncomputable def quadratic_positive_roots (m : ℝ) : Prop :=
  2 * m > 0 ∧ 2 - m > 0

theorem range_m (m : ℝ) : 
  (∃ x : ℝ, 4^x - m * 2^(x+1) + 2 - m = 0 ∧ ∃ y : ℝ, 4^y - m * 2^(y+1) + 2 - m = 0 ∧ x ≠ y) ↔ 
  (1 < m ∧ m < 2) :=
begin
  sorry
end

end range_m_l594_594602


namespace relationship_between_M_and_N_l594_594911

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 4
def N : ℝ := (a - 1) * (a - 3)

theorem relationship_between_M_and_N : M a > N a :=
by sorry

end relationship_between_M_and_N_l594_594911


namespace prop1_prop2_l594_594465

variable {Plane Line : Type}
variable (Perpendicular : Plane → Plane → Prop) (PerpendicularL : Line → Line → Prop)
variable (m n : Line) (α β : Plane)

-- Proposition 1
theorem prop1 (h1 : PerpendicularL m n) (h2 : PerpendicularL n β) (h3 : PerpendicularL m α) : Perpendicular α β := sorry

-- Proposition 2
theorem prop2 (h1 : Perpendicular α β) (h2 : PerpendicularL n β) (h3 : PerpendicularL m α) : PerpendicularL m n := sorry

end prop1_prop2_l594_594465


namespace sqrt_144_times_3_squared_l594_594168

theorem sqrt_144_times_3_squared :
  ( (Real.sqrt 144) * 3 ) ^ 2 = 1296 := by
  sorry

end sqrt_144_times_3_squared_l594_594168


namespace turnip_growth_proof_l594_594293

/-- Melanie, Benny, and Carol's turnip growth over two weeks -/
def total_turnips_two_weeks (melanie1 benny1 carol1 : ℕ) (melanie2_factor benny_factor carol_factor : ℚ) : ℕ :=
  let melanie2 := (melanie1 * melanie2_factor.toNat) in
  let benny2 := (benny1 * benny_factor).toNat in
  let carol2 := (carol1 * carol_factor).toNat in
  melanie1 + melanie2 + benny1 + benny2 + carol1 + carol2

theorem turnip_growth_proof :
  total_turnips_two_weeks 139 113 195 2 1.3 1.3 = 1126 :=
by
  sorry

end turnip_growth_proof_l594_594293


namespace largest_possible_factors_l594_594044

theorem largest_possible_factors (x : ℝ) :
  ∃ m q1 q2 q3 q4 : polynomial ℝ,
    m = 4 ∧
    x^10 - 1 = q1 * q2 * q3 * q4 ∧
    ¬(q1.degree = 0) ∧ ¬(q2.degree = 0) ∧ ¬(q3.degree = 0) ∧ ¬(q4.degree = 0) :=
sorry

end largest_possible_factors_l594_594044


namespace fish_population_estimate_l594_594616

/-- Definition of the capture-recapture method for estimating fish population in a pond. -/
theorem fish_population_estimate
  (C1 : ℕ) (C2 : ℕ) (R : ℕ)
  (C2' : ℕ) (R' : ℕ) (C1' : ℕ)
  (hC1 : C1 = 50)
  (hC2 : C2 = 100)
  (hR : R = 30)
  (hC2' : C2' = 80)
  (hR' : R' = 12)
  (hC1' : C1' = 120) :
  let N2 := (C1' * C2') / R' in
  N2 = 800 :=
by
  sorry

end fish_population_estimate_l594_594616


namespace isosceles_triangle_base_l594_594716

theorem isosceles_triangle_base (b : ℝ) (h1 : 7 + 7 + b = 20) : b = 6 :=
by {
    sorry
}

end isosceles_triangle_base_l594_594716


namespace unoccupied_chairs_after_event_l594_594904

theorem unoccupied_chairs_after_event :
  ∃ unoccupied_chairs : ℕ, unoccupied_chairs = 86 :=
by
  -- Define the problem conditions
  let rows := 45
  let chairs_per_row := 26
  let attendees := 1500
  let confirmed_attendees := 1422
  let buffer_percentage := 0.05

  -- Calculate the current number of chairs
  let total_chairs_current := rows * chairs_per_row

  -- Calculate the 5% buffer for confirmed attendees
  let buffer := (buffer_percentage * confirmed_attendees).ceil.toNat

  -- Calculate the total number of chairs needed
  let total_chairs_needed := confirmed_attendees + buffer

  -- Calculate how many more chairs need to be added
  let additional_chairs_needed := total_chairs_needed - total_chairs_current

  -- Calculate additional rows needed
  let additional_rows_needed := (additional_chairs_needed / chairs_per_row.toFloat).ceil.toNat

  -- Calculate the total number of chairs after adding additional rows
  let total_chairs_after_additional_rows := (rows + additional_rows_needed) * chairs_per_row

  -- Calculate the unoccupied chairs after the event
  let unoccupied_chairs := total_chairs_after_additional_rows - confirmed_attendees

  -- Assert the calculated unoccupied chairs equal to 86
  have : unoccupied_chairs = 86 := sorry
  exact ⟨unoccupied_chairs, this⟩

end unoccupied_chairs_after_event_l594_594904


namespace derivative_of_func_l594_594783

noncomputable def func (a b x : ℝ) : ℝ :=
  (a^2 + b^2)⁻¹/2 * asin ((sqrt (a^2 + b^2) * sin x) / b)

theorem derivative_of_func (a b x : ℝ) (h1 : a^2 + b^2 ≠ 0) (h2 : b ≠ 0) :
  deriv (func a b) x = cos x / sqrt (b^2 * cos x^2 - a^2 * sin x^2) :=
by
  sorry

end derivative_of_func_l594_594783


namespace conjugate_of_fraction_l594_594379

noncomputable def compute_conjugate : ℂ → ℂ
| z := z.conj

theorem conjugate_of_fraction : compute_conjugate ((3 + complex.I) / (1 - complex.I)) = 1 - 2 * complex.I := by 
  sorry

end conjugate_of_fraction_l594_594379


namespace sum_of_consecutive_2022_l594_594909

theorem sum_of_consecutive_2022 (m n : ℕ) (h : m ≤ n - 1) (sum_eq : (n - m + 1) * (m + n) = 4044) :
  (m = 163 ∧ n = 174) ∨ (m = 504 ∧ n = 507) ∨ (m = 673 ∧ n = 675) :=
sorry

end sum_of_consecutive_2022_l594_594909


namespace calc_sum_of_digits_l594_594597

theorem calc_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10) 
(hm : 10 * 3 + x = 34) (hmy : 34 * (10 * y + 4) = 136) : x + y = 7 :=
sorry

end calc_sum_of_digits_l594_594597


namespace caleb_dandelion_puffs_l594_594854

-- Definitions corresponding to the problem conditions
def puffs_given_away : Nat := 3 + 3 + 5 + 2
def puffs_per_friend : Nat := 9
def num_friends : Nat := 3

-- Theorem statement corresponding to the mathematically equivalent proof problem
theorem caleb_dandelion_puffs (puffs_given_away = 13) (puffs_per_friend = 9) (num_friends = 3) : 
  let original_puffs := puffs_given_away + (puffs_per_friend * num_friends)
  original_puffs = 40 :=
by
  sorry

end caleb_dandelion_puffs_l594_594854


namespace weight_of_new_person_l594_594022

-- Define the problem conditions
variables (W : ℝ) -- Weight of the new person
variable (initial_weight : ℝ := 65) -- Weight of the person being replaced
variable (increase_in_avg : ℝ := 4) -- Increase in average weight
variable (num_persons : ℕ := 8) -- Number of persons

-- Define the total increase in weight due to the new person
def total_increase : ℝ := num_persons * increase_in_avg

-- The Lean statement to prove
theorem weight_of_new_person (W : ℝ) (h : total_increase = W - initial_weight) : W = 97 := sorry

end weight_of_new_person_l594_594022


namespace greatest_integer_less_than_or_equal_to_expression_l594_594182

theorem greatest_integer_less_than_or_equal_to_expression : 
  let expr := (5^50 + 3^50) / (5^45 + 3^45) in 
  ⌊expr⌋ = 3124 := 
by
  -- Here we would provide the proof steps
  sorry

end greatest_integer_less_than_or_equal_to_expression_l594_594182


namespace point_in_first_quadrant_l594_594948

noncomputable def complex_location (z : ℂ) : String :=
if (z.re > 0) ∧ (z.im > 0) then "first quadrant" else 
if (z.re < 0) ∧ (z.im > 0) then "second quadrant" else 
if (z.re < 0) ∧ (z.im < 0) then "third quadrant" else 
if (z.re > 0) ∧ (z.im < 0) then "fourth quadrant" else "axis"

theorem point_in_first_quadrant : ∀ (z : ℂ), (2 - complex.i) * z = 5 → complex_location z = "first quadrant" :=
begin
  intros z hz,
  sorry
end

end point_in_first_quadrant_l594_594948


namespace meaningful_fraction_l594_594748

theorem meaningful_fraction (x : ℝ) : (∃ y : ℝ, y = 5 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end meaningful_fraction_l594_594748


namespace cosine_angle_AKB_correct_l594_594987

noncomputable def cosine_angle_AKB (AB BC AD DC BK KD : ℝ) : ℝ :=
if AB = BC ∧ DB_bisects_ADC ∧ AD / DC = 4 / 3 ∧ BK / KD = 1 / 3 then
  1 / 4 
else 
  0 -- or any default value if conditions are not met

theorem cosine_angle_AKB_correct (AB BC AD DC BK KD : ℝ) (h₁ : AB = BC) (h₂ : DB_bisects_ADC) (h₃ : AD / DC = 4 / 3) (h₄ : BK / KD = 1 / 3) : 
  cosine_angle_AKB AB BC AD DC BK KD = 1 / 4 :=
by sorry

-- Definitions of predicates used in the theorem (these can be expanded as needed)
def DB_bisects_ADC : Prop := sorry

end cosine_angle_AKB_correct_l594_594987


namespace find_P_X_lt_0_l594_594925

noncomputable def X : ℝ → ℝ := sorry
def μ : ℝ := 2
def σ2 : ℝ := sorry
def N : ℝ → ℝ := sorry

axiom cond1 : ∀ x, X(x) ∼ N(μ, σ2)
axiom cond2 : P(0 < X < 4) = 0.4

theorem find_P_X_lt_0 : P(X < 0) = 0.3 :=
by
  sorry

end find_P_X_lt_0_l594_594925


namespace vertical_shirts_count_l594_594791

-- Definitions from conditions
def total_people : ℕ := 40
def checkered_shirts : ℕ := 7
def horizontal_shirts := 4 * checkered_shirts

-- Proof goal
theorem vertical_shirts_count :
  ∃ vertical_shirts : ℕ, vertical_shirts = total_people - (checkered_shirts + horizontal_shirts) ∧ vertical_shirts = 5 :=
sorry

end vertical_shirts_count_l594_594791


namespace find_two_digit_divisors_l594_594143

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_remainder (a b r : ℕ) : Prop := a = b * (a / b) + r

theorem find_two_digit_divisors (n : ℕ) (h1 : is_two_digit n) (h2 : has_remainder 723 n 30) :
  n = 33 ∨ n = 63 ∨ n = 77 ∨ n = 99 :=
sorry

end find_two_digit_divisors_l594_594143


namespace trigonometric_identity_l594_594315

theorem trigonometric_identity (x : ℝ) :
  (1 / real.cos (2022 * x) + real.tan (2022 * x) = 1 / 2022) →
  (1 / real.cos (2022 * x) - real.tan (2022 * x) = 2022) :=
by
  intro h
  sorry

end trigonometric_identity_l594_594315


namespace max_plus_min_value_l594_594223

-- Define the function f as given in the problem
def f (x : ℝ) : ℝ := (sqrt (12 - x^4) + x^2) / x^3 + 4

-- Define the domain of x
def domain (x : ℝ) : Prop := x ∈ [-1, 0) ∨ x ∈ (0, 1]

-- The theorem stating that the maximum value of f(x) plus the minimum value of f(x) is 8
theorem max_plus_min_value (A B : ℝ) (hA : ∀ x, domain x → f(x) ≤ A) (hB : ∀ x, domain x → B ≤ f(x)) : A + B = 8 :=
sorry

end max_plus_min_value_l594_594223


namespace eval_ff_neg1_l594_594685

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then Real.log x / Real.log 2 else 4^x

theorem eval_ff_neg1 : f(f(-1)) = -2 :=
by
  sorry

end eval_ff_neg1_l594_594685


namespace MN_intersection_correct_l594_594953

-- Define the sets M and N
def setM : Set ℝ := {y | ∃ x ∈ (Set.univ : Set ℝ), y = x^2 + 2*x - 3}
def setN : Set ℝ := {x | |x - 2| ≤ 3}

-- Reformulated sets
def setM_reformulated : Set ℝ := {y | y ≥ -4}
def setN_reformulated : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- The intersection set
def MN_intersection : Set ℝ := {y | -1 ≤ y ∧ y ≤ 5}

-- The theorem stating the intersection of M and N equals MN_intersection
theorem MN_intersection_correct :
  {y | ∃ x ∈ setN_reformulated, y = x^2 + 2*x - 3} = MN_intersection :=
sorry  -- Proof not required as per instruction

end MN_intersection_correct_l594_594953


namespace find_f7_l594_594673

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x^7 + b * x^3 + c * x - 5

theorem find_f7 (a b c : ℝ) (h : f (-7) a b c = 7) : f 7 a b c = -17 :=
by
  sorry

end find_f7_l594_594673


namespace num_sets_A_eq_5_l594_594583

noncomputable def num_valid_sets_A (A : Set ℕ) (B : Set ℕ) : ℕ :=
if (A ∪ B = {1, 2, 3, 4, 5, 6, 7}) ∧
    (A ∩ B = ∅) ∧
    (A.card = 2) ∧
    (2 ∉ A) ∧
    (B.card = 5) ∧
    (5 ∉ B) then 1 else 0

theorem num_sets_A_eq_5 : ∑ (A B : Set ℕ in { s | (s.card = 2)}, num_valid_sets_A A B) = 5 := 
sorry

end num_sets_A_eq_5_l594_594583


namespace decagon_exterior_angle_sum_l594_594406

-- Define what it means to be a regular decagon
def regular_decagon (P : Type) := ∃ (sides : P → Prop), (∀ x : P, sides x)

-- Define the sum of exterior angles
def sum_exterior_angles (P : Type) [regular_decagon P] : ℝ :=
  360

-- The theorem to be proved
theorem decagon_exterior_angle_sum (P : Type) [regular_decagon P] : sum_exterior_angles P = 360 := by
  sorry

end decagon_exterior_angle_sum_l594_594406


namespace tan_2x_solution_l594_594938

theorem tan_2x_solution (x : ℝ) (h1 : sin x - 3 * cos x = sqrt 5) : tan (2 * x) = 4 / 3 :=
sorry

end tan_2x_solution_l594_594938


namespace circle_diameter_l594_594798

theorem circle_diameter (r : ℝ) (h : π * r^2 = 16 * π) : 2 * r = 8 :=
by
  sorry

end circle_diameter_l594_594798


namespace equal_stickers_received_l594_594374

theorem equal_stickers_received (S : Type) (f g : S → ℕ) 
  (h : ∀ (s : S), f s = (∑ t in S, g t) - g s) :
  ∃ N, ∀ s, f s + g s = N :=
sorry

end equal_stickers_received_l594_594374


namespace sum_of_ages_l594_594741

-- Define Henry's and Jill's present ages
def Henry_age : ℕ := 23
def Jill_age : ℕ := 17

-- Define the condition that 11 years ago, Henry was twice the age of Jill
def condition_11_years_ago : Prop := (Henry_age - 11) = 2 * (Jill_age - 11)

-- Theorem statement: sum of Henry's and Jill's present ages is 40
theorem sum_of_ages : Henry_age + Jill_age = 40 :=
by
  -- Placeholder for proof
  sorry

end sum_of_ages_l594_594741


namespace tom_earned_bonus_points_l594_594615

theorem tom_earned_bonus_points 
  (customers_per_hour: ℕ) (hours_worked: ℕ) (bonus_percentage: ℝ)
  (h1: customers_per_hour = 10)
  (h2: hours_worked = 8)
  (h3: bonus_percentage = 0.20) : 
  let total_customers_served := customers_per_hour * hours_worked in
  let bonus_points := bonus_percentage * total_customers_served in
  bonus_points = 16 := 
by
  sorry

end tom_earned_bonus_points_l594_594615


namespace count_integers_with_34_l594_594264

theorem count_integers_with_34 (h : ∀ n, 600 ≤ n ∧ n < 1100) : 
  (count (λ n, contains_digits n 3 4) (range 600 1100)) = 8 :=
  sorry

def contains_digits (n : Nat) (d1 d2 : Nat) : Prop :=
  let digits := (n / 100) % 10 :: (n / 10) % 10 :: n % 10 :: []
  d1 ∈ digits ∧ d2 ∈ digits

def count {α : Type} (p : α → Prop) [DecidablePred p] (xs : List α) : Nat :=
  (xs.filter p).length

end count_integers_with_34_l594_594264


namespace probability_A_given_conditions_l594_594968

theorem probability_A_given_conditions (P : Type) [ProbabilitySpace P] 
  {A B : Event P} :
  P(AB) = 3 / 10 → P(B|A) = 1 / 2 → P(A) = 3 / 5 :=
by
  intros h1 h2
  sorry

end probability_A_given_conditions_l594_594968


namespace number_of_perfect_squares_criteria_l594_594265

noncomputable def number_of_multiples_of_40_squares_lt_4e6 : ℕ :=
  let upper_limit := 2000
  let multiple := 40
  let largest_multiple := upper_limit - (upper_limit % multiple)
  largest_multiple / multiple

theorem number_of_perfect_squares_criteria :
  number_of_multiples_of_40_squares_lt_4e6 = 49 :=
sorry

end number_of_perfect_squares_criteria_l594_594265


namespace total_cost_is_18_l594_594320

def round_to_nearest_dollar (x : ℝ) : ℤ :=
  Int.ofNat (Float.ceil $ x + 0.5)

def total_cost_rounded_to_nearest_dollar :=
  round_to_nearest_dollar 2.47 +
  round_to_nearest_dollar 6.25 +
  round_to_nearest_dollar 8.76 +
  round_to_nearest_dollar 1.49

theorem total_cost_is_18 :
  total_cost_rounded_to_nearest_dollar = 18 :=
by
  unfold total_cost_rounded_to_nearest_dollar
  unfold round_to_nearest_dollar
  sorry

end total_cost_is_18_l594_594320


namespace prime_square_mod_24_l594_594684

theorem prime_square_mod_24 (p q : ℕ) (k : ℤ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : p > 5) (hq_gt_5 : q > 5) 
  (h_diff : p ≠ q)
  (h_eq : p^2 - q^2 = 6 * k) : (p^2 - q^2) % 24 = 0 := by
sorry

end prime_square_mod_24_l594_594684


namespace percentage_water_in_fresh_grapes_is_65_l594_594216

noncomputable def percentage_water_in_fresh_grapes 
  (weight_fresh : ℝ) (weight_dried : ℝ) (percentage_water_dried : ℝ) : ℝ :=
  100 - ((weight_dried / weight_fresh) - percentage_water_dried / 100 * weight_dried / weight_fresh) * 100

theorem percentage_water_in_fresh_grapes_is_65 :
  percentage_water_in_fresh_grapes 400 155.56 10 = 65 := 
by
  sorry

end percentage_water_in_fresh_grapes_is_65_l594_594216


namespace find_initial_markup_l594_594130

variable (CP : ℕ) (x SP MP : ℝ)
variable hCP : CP = 100
variable hMP : MP = CP + x
variable hSP_discount : SP = MP - 0.1 * MP
variable hSP_profit : SP = CP + 0.17 * CP

theorem find_initial_markup : x = 30 :=
  by
    have h1 : MP = 100 + x := by rw [hCP, hMP]
    have h2 : SP = (100 + x) - 0.1 * (100 + x) := by rw [hMP, hSP_discount]
    have h3 : SP = 90 + 0.9 * x := 
      by 
        rw [h1]
        linarith
    have h4 : SP = 117 := by rw [hCP, hSP_profit]
    have h5 : 90 + 0.9 * x = 117 := 
      by 
        rw [h4, h3]
    have hx : 0.9 * x = 27 := 
      by 
        linarith
    have h_result : x = 30 := 
      by 
        ring_nf in hx
        linarith
    exact h_result

end find_initial_markup_l594_594130


namespace stairs_climbed_together_l594_594323

-- Definitions from conditions
def Jonny_stairs : ℕ := 1269
def Julia_stairs : ℕ := (Jonny_stairs / 3) - 7

-- Proof that their combined climbed stairs is 1685
theorem stairs_climbed_together :
  Jonny_stairs + Julia_stairs = 1685 := by
  have Jonny_stairs_eq : Jonny_stairs = 1269 := rfl
  have Julia_stairs_eq : Julia_stairs = (1269 / 3) - 7 := rfl
  calc
    Jonny_stairs + Julia_stairs
        = 1269 + ((1269 / 3) - 7) : by rw [Jonny_stairs_eq, Julia_stairs_eq]
    ... = 1269 + (423 - 7)         : by norm_num
    ... = 1269 + 416               : by norm_num
    ... = 1685                     : by norm_num

end stairs_climbed_together_l594_594323


namespace area_of_bounded_shape_l594_594509

def parametric_curve_x (t : ℝ) : ℝ := t - Real.sin t
def parametric_curve_y (t : ℝ) : ℝ := 1 - Real.cos t
def boundary_y : ℝ := 1

def integrand (t : ℝ) : ℝ := (1 - Real.cos t) * (1 - Real.cos t)

noncomputable def area_bounded_by_parametric_curves : ℝ :=
  Real.integral (set.Icc (Real.pi / 2) (3 * Real.pi / 2)) integrand

noncomputable def area_under_line : ℝ :=
  let x1 := parametric_curve_x (Real.pi / 2)
  let x2 := parametric_curve_x (3 * Real.pi / 2)
  boundary_y * (x2 - x1)

theorem area_of_bounded_shape : 
  (area_bounded_by_parametric_curves - area_under_line) = (Real.pi / 2) + 2 :=
by
  sorry

end area_of_bounded_shape_l594_594509


namespace fraction_inequality_solution_l594_594036

theorem fraction_inequality_solution (x : ℝ) :
  -1 ≤ x ∧ x ≤ 3 ∧ (4 * x + 3 > 2 * (8 - 3 * x)) → (13 / 10) < x ∧ x ≤ 3 :=
by
  sorry

end fraction_inequality_solution_l594_594036


namespace rectangle_perimeter_is_22_l594_594142

-- Definition of sides of the triangle DEF
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Helper function to compute the area of a right triangle
def triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Ensure the triangle is a right triangle and calculate its area
def area_of_triangle : ℕ :=
  if (side1 * side1 + side2 * side2 = hypotenuse * hypotenuse) then
    triangle_area side1 side2
  else
    0

-- Definition of rectangle's width and equation to find its perimeter
def width : ℕ := 5
def rectangle_length : ℕ := area_of_triangle / width
def perimeter_of_rectangle : ℕ := 2 * (width + rectangle_length)

theorem rectangle_perimeter_is_22 : perimeter_of_rectangle = 22 :=
by
  -- Proof content goes here
  sorry

end rectangle_perimeter_is_22_l594_594142


namespace min_value_expression_l594_594905

theorem min_value_expression (x : ℝ) (h : 0 ≤ x ∧ x < 4) : ∃ m : ℝ, m = sqrt 5 ∧ ∀ y : ℝ, y = (x^2 + 2 * x + 6) / (2 * x + 2) → y ≥ m := by
  sorry

end min_value_expression_l594_594905


namespace min_blocks_for_wall_l594_594795

noncomputable def min_blocks_needed (length height : ℕ) (block_sizes : List (ℕ × ℕ)) : ℕ :=
  sorry

theorem min_blocks_for_wall :
  min_blocks_needed 120 8 [(1, 3), (1, 2), (1, 1)] = 404 := by
  sorry

end min_blocks_for_wall_l594_594795


namespace fg_eq_gf_solution_condition_l594_594970

-- Define the problem with given conditions
variables {R : Type*} [CommRing R]
variables (m n p q : R) -- where m, n, p, q are real numbers
noncomputable def f (x : R) : R := m * x + n
noncomputable def g (x : R) : R := p * x + q

-- Define the proof problem statement
theorem fg_eq_gf_solution_condition :
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) - q * (1 - m) = 0 :=
sorry

end fg_eq_gf_solution_condition_l594_594970


namespace power_function_passes_through_one_one_l594_594501

theorem power_function_passes_through_one_one (α : ℝ) (h : α ≠ 0) : (∀ x, x = 1 → x^α = 1) :=
by
  intro x hx
  rw hx
  norm_num

end power_function_passes_through_one_one_l594_594501


namespace five_times_number_equals_hundred_l594_594899

theorem five_times_number_equals_hundred (x : ℝ) (h : 5 * x = 100) : x = 20 :=
sorry

end five_times_number_equals_hundred_l594_594899


namespace student_mark_for_correct_answer_l594_594626

theorem student_mark_for_correct_answer
  (total_questions : ℕ) (correct_questions : ℕ) (total_marks : ℤ)
  (wrong_mark_penalty : ℤ) (correct_mark : ℤ) :
  total_questions = 50 →
  correct_questions = 36 →
  total_marks = 130 →
  wrong_mark_penalty = -1 →
  let wrong_questions := total_questions - correct_questions in
  let marks_from_correct := correct_questions * correct_mark in
  let marks_from_wrong := wrong_questions * wrong_mark_penalty in
  total_marks = marks_from_correct + marks_from_wrong →
  correct_mark = 4 := 
by 
  intros h1 h2 h3 h4 hw mq mw htot;
  sorry

end student_mark_for_correct_answer_l594_594626


namespace downstream_distance_l594_594456

noncomputable def effective_speed_downstream (boat_speed current_speed : ℝ) : ℝ :=
  boat_speed + current_speed

noncomputable def time_in_hour (time_in_minutes : ℝ) : ℝ :=
  time_in_minutes / 60

noncomputable def distance (speed time : ℝ) : ℝ :=
  speed * time

theorem downstream_distance 
  (boat_speed : ℝ) (current_speed : ℝ) (time_in_minutes : ℝ) 
  (h1 : boat_speed = 15) (h2 : current_speed = 3) (h3 : time_in_minutes = 24) : 
  distance (effective_speed_downstream boat_speed current_speed) (time_in_hour time_in_minutes) = 7.2 :=
by {
  sorry,
}

end downstream_distance_l594_594456


namespace count_ways_to_distribute_balls_l594_594267

theorem count_ways_to_distribute_balls : 
  ∀ (balls boxes : ℕ), balls = 6 → boxes = 2 → 
  (let total_ways := pow boxes balls,
       unique_ways := total_ways / 2 - 1
  in unique_ways = 31) :=
by
  intros balls boxes h_balls h_boxes
  let total_ways := boxes ^ balls
  let unique_ways := total_ways / 2 - 1
  rw [h_balls, h_boxes]
  have h1 : total_ways = 2 ^ 6 := by sorry
  have h2 : unique_ways = (2 ^ 6) / 2 - 1 := by sorry
  rw [h1, h2]
  exact sorry

end count_ways_to_distribute_balls_l594_594267


namespace houses_without_features_l594_594780

-- Definitions for the given conditions
def N : ℕ := 70
def G : ℕ := 50
def P : ℕ := 40
def GP : ℕ := 35

-- The statement of the proof problem
theorem houses_without_features : N - (G + P - GP) = 15 := by
  sorry

end houses_without_features_l594_594780


namespace board_game_k_l594_594637

theorem board_game_k (s1 s2 : ℕ) (k : ℕ) (config : ℕ → set ℕ) :
  s1 = 3 → s2 = 2 →
  (∀ n < k, ∀ b ∈ config n, (player1_wins_probability b > 1 / 2)) →
  (∃ b ∈ config k, (player2_wins_probability b > 1 / 2)) →
  k = 3 :=
by
  intros h1 h2 hk h3
  sorry

end board_game_k_l594_594637


namespace floor_abs_of_neg_58_point_7_l594_594875

theorem floor_abs_of_neg_58_point_7 : 
  let x := -58.7 in ⌊|x|⌋ = 58 :=
by
  -- defining x
  let x : ℝ := -58.7
  -- Evaluating
  have h1 : |x| = 58.7 := by
    sorry
  have h2 : ⌊58.7⌋ = 58 := by
    sorry
  rw [h1, h2]

end floor_abs_of_neg_58_point_7_l594_594875


namespace trajectory_of_point_l594_594924

theorem trajectory_of_point (x y : ℝ) : 
  (√((x - 1)^2 + (y - 1)^2) = |x + y - 2| / √2) 
  → (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a * x + b * y + c = 0) :=
by 
  sorry

end trajectory_of_point_l594_594924


namespace cannot_form_right_triangle_A_l594_594840

-- Define the conditions as given in the problem
def sidesA := (Real.sqrt 3, Real.sqrt 4, Real.sqrt 5)
def sidesB := (1, Real.sqrt 2, Real.sqrt 3)
def sidesC := (6, 8, 10)
def sidesD := (3, 4, 5)

-- Define a function that checks if three sides can form a right triangle
def can_form_right_triangle (a b c : ℝ) : Bool :=
  (a^2 + b^2 = c^2) || (a^2 + c^2 = b^2) || (b^2 + c^2 = a^2)

-- Problem statement in Lean 4
theorem cannot_form_right_triangle_A :
  ¬ can_form_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5) :=
sorry

end cannot_form_right_triangle_A_l594_594840


namespace incorrect_statement_d_l594_594499

noncomputable def kw : ℝ := 1e-14 -- Assuming a standard value for the ion product constant of water at a given temperature

-- Definitions based on conditions
def constant_temperature : Prop := true -- Placeholder for constant temperature condition
def increase_hydrogen_ions (ch : ℝ) (hcl : ℝ) : Prop := ch > 0 ∧ hcl > 0

-- Main statement to prove
theorem incorrect_statement_d (ch : ℝ) (hcl : ℝ) (kw : ℝ) :
  (increase_hydrogen_ions ch hcl) ∧ (constant_temperature) → 
  ¬(c(H⁺ from water) increases) :=
by {
  sorry
}

end incorrect_statement_d_l594_594499


namespace find_side_a_l594_594309

def sides (A B C : ℝ) (a b c : ℝ) : Prop :=
  cos A = sqrt 6 / 3 ∧ b = 2 * sqrt 2 ∧ c = sqrt 3

theorem find_side_a (A B C a b c : ℝ) (h : sides A B C a b c) : a = sqrt 3 :=
by
  sorry

end find_side_a_l594_594309


namespace pairwise_sum_product_l594_594415

theorem pairwise_sum_product (a b c d e : ℕ)
  (h : multiset.pairwise_sum {a, b, c, d, e} = {5, 9, 10, 11, 12, 16, 16, 17, 21, 23}) :
  a * b * c * d * e = 5292 := by
  sorry

end pairwise_sum_product_l594_594415


namespace number_of_stacks_l594_594347

theorem number_of_stacks (total_coins stacks coins_per_stack : ℕ) (h1 : coins_per_stack = 3) (h2 : total_coins = 15) (h3 : total_coins = stacks * coins_per_stack) : stacks = 5 :=
by
  sorry

end number_of_stacks_l594_594347


namespace circle_properties_l594_594463

variables {α : Type*} [linear_ordered_field α]

def ModHomothety (s : Circle α) (k : α) (P : Point α) : Circle α := sorry

def NagelPoint (Δ : Triangle α) : Point α := sorry
def Centroid (Δ : Triangle α) : Point α := sorry
def Incircle (Δ : Triangle α) : Circle α := sorry
def MidlineTriangle (Δ : Triangle α) : Set (Line α) := sorry
def MidpointsToNagelLines (Δ : Triangle α) (N : Point α) : Set (Line α) := sorry

theorem circle_properties (Δ : Triangle α) :
  let s := Incircle Δ,
      N := NagelPoint Δ,
      M := Centroid Δ,
      S' := ModHomothety s (1 / 2) N,
      S := ModHomothety s (-1 / 2) M in
  ∃ S₁ : Circle α, S = S₁ ∧
      S.TouchesMidlines (MidlineTriangle Δ) ∧ 
      S'.TouchesLines (MidpointsToNagelLines Δ N) :=
begin
  sorry,
end

end circle_properties_l594_594463


namespace area_of_parallelogram_l594_594198

def base := 12
def height := 18

theorem area_of_parallelogram : (base * height = 216) := by
  sorry

end area_of_parallelogram_l594_594198


namespace smallest_positive_varphi_l594_594014

theorem smallest_positive_varphi {k : ℤ} :
  ∃ (φ : ℝ), (∀ x : ℝ, real.sin (2 * (x - real.pi / 4) + φ) = real.sin (2 * x - real.pi / 2 + φ)) ∧
  (∀ k : ℤ, φ = k * real.pi + real.pi → φ > 0) → φ = real.pi :=
sorry

end smallest_positive_varphi_l594_594014


namespace solve_for_x_l594_594708

noncomputable def proof (x : ℚ) : Prop :=
  (x + 6) / (x - 4) = (x - 7) / (x + 2)

theorem solve_for_x (x : ℚ) (h : proof x) : x = 16 / 19 :=
by
  sorry

end solve_for_x_l594_594708


namespace integer_root_sum_abs_l594_594903

theorem integer_root_sum_abs :
  ∃ a b c m : ℤ, 
    (a + b + c = 0 ∧ ab + bc + ca = -2023 ∧ |a| + |b| + |c| = 94) := sorry

end integer_root_sum_abs_l594_594903


namespace oranges_to_put_back_l594_594451

theorem oranges_to_put_back
  (p_A p_O : ℕ)
  (A O : ℕ)
  (total_fruits : ℕ)
  (initial_avg_price new_avg_price : ℕ)
  (x : ℕ)
  (h1 : p_A = 40)
  (h2 : p_O = 60)
  (h3 : total_fruits = 15)
  (h4 : initial_avg_price = 48)
  (h5 : new_avg_price = 45)
  (h6 : A + O = total_fruits)
  (h7 : (p_A * A + p_O * O) / total_fruits = initial_avg_price)
  (h8 : (720 - 60 * x) / (15 - x) = 45) :
  x = 3 :=
by
  sorry

end oranges_to_put_back_l594_594451


namespace speech_competition_average_and_variance_l594_594985

def scores : List ℝ := [90, 86, 90, 97, 93, 94, 93]

def remove_extreme_values (l : List ℝ) : List ℝ := 
  List.erase (List.erase l (List.maximum l)) (List.minimum l)

def average (l : List ℝ) : ℝ := l.sum / l.length

def variance (l : List ℝ) : ℝ := 
  let avg := average l
  (l.map (λ x => (x - avg) ^ 2)).sum / l.length

theorem speech_competition_average_and_variance :
  let cleaned_data := remove_extreme_values scores in
  average cleaned_data = 92 ∧ variance cleaned_data = 2.8 := by
  sorry

end speech_competition_average_and_variance_l594_594985


namespace magnitude_of_b_add_c_l594_594260

open real

variables (x : ℝ)
def a := (x, 2 : ℝ × ℝ)
def b := (2, 1 : ℝ × ℝ)
def c := (3, 2 * x : ℝ × ℝ)

-- Define the perpendicular condition
axiom a_perpendicular_b : (a x).fst * (b).fst + (a x).snd * (b).snd = 0

-- Define the magnitude function
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.fst ^ 2 + v.snd ^ 2)

theorem magnitude_of_b_add_c : magnitude (b.fst + c.fst, b.snd + c.snd) = real.sqrt 26 :=
by {
  sorry -- Proof goes here
}

end magnitude_of_b_add_c_l594_594260


namespace sum_of_sequences_equals_n_l594_594697

theorem sum_of_sequences_equals_n (n : ℕ) :
  (∑ s in finset.powerset_len n (finset.range n).powerset, (1 : ℚ) / (s.prod id)) = n := 
sorry

end sum_of_sequences_equals_n_l594_594697


namespace mrs_generous_jelly_beans_l594_594001

-- Define necessary terms and state the problem
def total_children (x : ℤ) : ℤ := x + (x + 3)

theorem mrs_generous_jelly_beans :
  ∃ x : ℤ, x^2 + (x + 3)^2 = 490 ∧ total_children x = 31 :=
by {
  sorry
}

end mrs_generous_jelly_beans_l594_594001


namespace area_excluding_hole_l594_594809

theorem area_excluding_hole (x : ℝ) : 
  let large_rectangle_area := (x + 8) * (x + 6)
  let hole_area := (2 * x - 4) * (x - 3)
  large_rectangle_area - hole_area = -x^2 + 24 * x + 36 := 
by 
  -- Define areas
  let large_rectangle_area := (x + 8) * (x + 6)
  let hole_area := (2 * x - 4) * (x - 3)
  -- Perform the calculation
  calc
    large_rectangle_area - hole_area
      = (x + 8) * (x + 6) - (2 * x - 4) * (x - 3) : rfl
  ... = x^2 + 14 * x + 48 - (2 * x^2 - 10 * x + 12) : by 
    {
      calc
        (x + 8) * (x + 6)
          = x^2 + 6 * x + 8 * x + 48 : by ring
      ... = x^2 + 14 * x + 48 : by ring,
  
      calc
        (2 * x - 4) * (x - 3)
          = 2 * x * x - 2 * x * 3 - 4 * x + 12 : by ring
      ... = 2 * x^2 - 6 * x - 4 * x + 12 : by ring
      ... = 2 * x^2 - 10 * x + 12 : by ring
    }
  ... = - x^2 + 24 * x + 36 : by ring

end area_excluding_hole_l594_594809


namespace best_player_is_daughter_l594_594000

-- Defining players and their attributes for clarity:
inductive Player
| MrNiu : Player
| Sister : Player
| Son : Player
| Daughter : Player

def gender : Player → Prop
| Player.MrNiu   := true  -- Assume true represents male
| Player.Sister  := false -- Assume false represents female
| Player.Son     := true
| Player.Daughter:= false

def age : Player → nat
| Player.MrNiu   := 50     -- Assume arbitrary higher age
| Player.Sister  := 20     -- Assume arbitrary same age for twins
| Player.Son     := 20
| Player.Daughter:= 20

-- Definitions from conditions
def same_age (p1 p2 : Player) : Prop :=
  age p1 = age p2

def different_gender (p1 p2 : Player) : Prop :=
  gender p1 ≠ gender p2

-- Given conditions
axiom cond1 : ∀ (best worst : Player), different_gender best worst → same_age best worst → (best = Player.Son ∧ worst = Player.Daughter) ∨ (best = Player.Daughter ∧ worst = Player.Son)

-- Proving the statement
theorem best_player_is_daughter : ∃ best : Player, best = Player.Daughter :=
begin
  use Player.Daughter,
  sorry
end

end best_player_is_daughter_l594_594000


namespace projection_vecAB_onto_vecAC_is_correct_l594_594221

-- Define the points A, B, and C
def A : ℝ × ℝ × ℝ := (1, 1, 0)
def B : ℝ × ℝ × ℝ := (0, 3, 0)
def C : ℝ × ℝ × ℝ := (2, 2, 2)

-- Define the vector AB and AC
def vecAB : ℝ × ℝ × ℝ := (0 - 1, 3 - 1, 0 - 0)
def vecAC : ℝ × ℝ × ℝ := (2 - 1, 2 - 1, 2 - 0)

-- Calculate the dot product of vecAB and vecAC
def dotProduct (u v: ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Calculate the length squared of a vector
def lengthSquared (v: ℝ × ℝ × ℝ) : ℝ := v.1^2 + v.2^2 + v.3^2

-- The projection of vecAB onto vecAC
def projection (u v: ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let k := dotProduct u v / lengthSquared v in (k * v.1, k * v.2, k * v.3)

-- Assertion to be proved
theorem projection_vecAB_onto_vecAC_is_correct : projection vecAB vecAC = (1/6, 1/6, 1/3) :=
  sorry

end projection_vecAB_onto_vecAC_is_correct_l594_594221


namespace complement_of_A_in_U_l594_594344

-- Define the universal set U and the subset A
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {1, 2, 5, 7}

-- Define the complement of A with respect to U
def complementU_A : Set Nat := {x ∈ U | x ∉ A}

-- Prove the complement of A in U is {3, 4, 6}
theorem complement_of_A_in_U :
  complementU_A = {3, 4, 6} :=
by
  sorry

end complement_of_A_in_U_l594_594344


namespace ivy_has_20_collectors_dolls_l594_594871

theorem ivy_has_20_collectors_dolls
  (D : ℕ) (I : ℕ) (C : ℕ)
  (h1 : D = 60)
  (h2 : D = 2 * I)
  (h3 : C = 2 * I / 3) 
  : C = 20 :=
by sorry

end ivy_has_20_collectors_dolls_l594_594871


namespace triangle_sides_and_angles_l594_594400

def arithmetic_progression_sides {a d : ℕ} (h : a > d) : Prop := 
  (a - d, a, a + d)

def right_angled_triangle (a b c : ℝ) : Prop := 
  a * a + b * b = c * c

theorem triangle_sides_and_angles (a d : ℝ) (h_ad : a > d) (h_area : (1 / 2) * a * (a - d) = 216) 
    (h_right : right_angled_triangle (a - d) a (a + d)) :
    a = 24 ∧ d = 6 ∧ 
    arithmetic_progression_sides (by exact_mod_cast 24 : ℕ) (by exact_mod_cast 6 : ℕ) = (18,24,30) ∧
    real.sin (real.arcsin (4 / 5)) ≈ 4 / 5 ∧ 
    real.arcsin (4 / 5) * (180 / real.pi) ≈ 53.13 ∧
    (90 - real.arcsin (4 / 5) * (180 / real.pi)) ≈ 36.87 := sorry

end triangle_sides_and_angles_l594_594400


namespace dave_first_to_six_prob_l594_594837

/--
  Consider a game where four players—Alice, Bob, Carol, and Dave—take turns rolling a die in sequence. Each player rolls the die independently with 
  the probability of rolling a six being 1/6. The sequence continues indefinitely until one player rolls a six.

  We wish to prove that the probability that Dave is the first player to roll a six is 125/671.
-/
theorem dave_first_to_six_prob :
  let p := (5 / 6) * (5 / 6) * (5 / 6) * (1 / 6) in
  let q := (5 / 6) ^ 4 in
  ∑' n : ℕ, p * q ^ n = 125 / 671 :=
by
  sorry  -- Proof is omitted

end dave_first_to_six_prob_l594_594837


namespace line_and_circle_separated_l594_594184

theorem line_and_circle_separated (θ : ℝ) :
  let A := Real.sin θ
  let B := Real.cos θ
  let C := 1 + Real.cos θ
  let center_distance := (|B - C|) / Real.sqrt (A^2 + B^2)
  let radius := Real.sqrt (1 / 2)
  center_distance > radius :=
by
  let A := Real.sin θ
  let B := Real.cos θ
  let C := 1 + Real.cos θ
  let center_distance := |B - C| / Real.sqrt (A^2 + B^2)
  let radius := Real.sqrt (1 / 2)
  have h1 : center_distance = 1 := by sorry
  have h2 : radius = Real.sqrt (1 / 2) := by sorry
  have h3 : 1 > Real.sqrt (1 / 2) := by sorry
  exact h3

end line_and_circle_separated_l594_594184


namespace ratio_a5_b5_l594_594286

-- Definitions of arithmetic sequences
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a : ℕ → ℚ) : ℕ → ℚ
| 0 := 0
| (n + 1) := sum_arithmetic_seq a n + a (n + 1)

theorem ratio_a5_b5
  {a b : ℕ → ℚ}
  {d_a d_b : ℚ}
  (ha : arithmetic_seq a d_a)
  (hb : arithmetic_seq b d_b)
  (h : ∀ n, (sum_arithmetic_seq a n) / (sum_arithmetic_seq b n) = (7 * n + 5) / (n + 3))
  : (a 5) / (b 5) = 17 / 3 :=
sorry

end ratio_a5_b5_l594_594286


namespace triangle_area_is_3_over_8_l594_594983

noncomputable def area_of_triangle_ABC (AB BC : ℝ × ℝ) : ℝ :=
  let ⟨xA, yA⟩ := AB
  let ⟨xB, yB⟩ := BC
  let a := xA * yB - yA * xB
  (1 / 2) * abs a

theorem triangle_area_is_3_over_8 :
  let AB := (Real.cos (32 * Real.pi / 180), Real.cos (58 * Real.pi / 180))
      BC := ((Real.sin (60 * Real.pi / 180)) * (Real.sin (118 * Real.pi / 180)), 
             (Real.sin (120 * Real.pi / 180)) * (Real.sin (208 *Real.pi / 180)))
  area_of_triangle_ABC AB BC = 3 / 8 :=
by
  -- The proof is skipped
  sorry

end triangle_area_is_3_over_8_l594_594983


namespace lcm_of_1_to_20_equals_232792560_l594_594433

theorem lcm_of_1_to_20_equals_232792560 : (Nat.lcmList (List.range 21).tail) = 232792560 := by
  sorry

end lcm_of_1_to_20_equals_232792560_l594_594433


namespace part1_part2_part3_l594_594559

-- Define the sequence and conditions
variable {a : ℕ → ℕ}
axiom sequence_def (n : ℕ) : a n = max (a (n + 1)) (a (n + 2)) - min (a (n + 1)) (a (n + 2))

-- Part (1)
axiom a1_def : a 1 = 1
axiom a2_def : a 2 = 2
theorem part1 : a 4 = 1 ∨ a 4 = 3 ∨ a 4 = 5 :=
  sorry

-- Part (2)
axiom has_max (M : ℕ) : ∀ n, a n ≤ M
theorem part2 : ∃ n, a n = 0 :=
  sorry

-- Part (3)
axiom positive_seq : ∀ n, a n > 0
theorem part3 : ¬∃ M : ℝ, ∀ n, a n ≤ M :=
  sorry

end part1_part2_part3_l594_594559


namespace smallest_value_at_x_5_l594_594207

-- Define the variable x
def x : ℕ := 5

-- Define each expression
def exprA := 8 / x
def exprB := 8 / (x + 2)
def exprC := 8 / (x - 2)
def exprD := x / 8
def exprE := (x + 2) / 8

-- The goal is to prove that exprD yields the smallest value
theorem smallest_value_at_x_5 : exprD = min (min (min exprA exprB) (min exprC exprE)) :=
sorry

end smallest_value_at_x_5_l594_594207


namespace num_ordered_pairs_l594_594589

theorem num_ordered_pairs:
  ({p : ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ (5 * p.2 + 3 * p.1 = p.1 * p.2)}.to_finset.card = 4) :=
by
  sorry

end num_ordered_pairs_l594_594589


namespace six_card_arrangements_l594_594707

theorem six_card_arrangements :
  let cards := {1, 2, 3, 4, 5, 6}
  let even_cards := {2, 4, 6}
  (∃ card ∈ even_cards, ∃ seq ∈ (cards \ {card}), (is_strictly_ascending seq ∨ is_strictly_descending seq)) = 6 :=
sorry

-- Note: is_strictly_ascending and is_strictly_descending are placeholders for the actual proofs or definitions of these properties.

end six_card_arrangements_l594_594707


namespace derivative_function_f_l594_594383

noncomputable def function_f : ℝ → ℝ := λ x, 2 * x + 3

theorem derivative_function_f : (derivative function_f) = (λ x, 2) :=
by
  sorry

end derivative_function_f_l594_594383


namespace cost_effective_bus_choice_l594_594824

theorem cost_effective_bus_choice (x y : ℕ) (h1 : y = x - 1) (h2 : 32 < 48 * x - 64 * y ∧ 48 * x - 64 * y < 64) : 
  64 * 300 < x * 2600 → True :=
by {
  sorry
}

end cost_effective_bus_choice_l594_594824


namespace perpendicular_lines_l594_594097

theorem perpendicular_lines {a : ℝ} (h : ∀ (x y : ℝ), ax + 2 * y - 1 = 0 ↔ 2 * x - 3 * y - 1 = 0 ∧ (ax + 2 * y - 1).is_perpendicular (2 * x - 3 * y - 1)) :
  a = 3 :=
by sorry

end perpendicular_lines_l594_594097


namespace center_DEP_on_perp_bisector_AB_l594_594812

-- Definitions related to the problem

variable {A B C D E M P : Type} -- The points involved in the problem
variable [EuclideanGeometry A B C D E M P] -- Euclidean geometry context

-- Conditions
variables (hABC_equilateral : EquilateralTriangle A B C)
          (hM_centroid : Centroid A B C M)
          (hD_on_BC : OnLineThroughPoint M D (LineSegment B C))
          (hE_on_CA : OnLineThroughPoint M E (LineSegment C A))
          (hP_circumcircle_intersection : Intersects (Circumcircle A E M) (Circumcircle B D M) P)

-- Conclusion to be proved
theorem center_DEP_on_perp_bisector_AB :
  OnPerpendicularBisector (Circumcircle D E P).center (LineSegment A B) :=
sorry

end center_DEP_on_perp_bisector_AB_l594_594812


namespace fraction_of_value_l594_594071

def value_this_year : ℝ := 16000
def value_last_year : ℝ := 20000

theorem fraction_of_value : (value_this_year / value_last_year) = 4 / 5 := by
  sorry

end fraction_of_value_l594_594071


namespace factor_count_eq_six_l594_594100

theorem factor_count_eq_six : 
  ∃ (factors : List (Polynomial ℤ)), 
    (∏ x in factors, x) = Polynomial.CoeffRing.mk (⟨some 1, some 2⟩) 
    ∧ factors.length = 6 
    ∧ ∀ f ∈ factors, Polynomial.IsMonomialWithIntegralCoefficients f :=
by
  sorry

end factor_count_eq_six_l594_594100


namespace floor_e_eq_two_l594_594881

theorem floor_e_eq_two : ⌊Real.exp 1⌋ = 2 :=
by
  sorry

end floor_e_eq_two_l594_594881


namespace main_theorem_l594_594998

noncomputable def line_passing_through : Prop :=
  ∀ (M : ℝ × ℝ) (m : ℝ), M = (-1, m) → 
    (∃ t : ℝ, (∀ (x y : ℝ), x = -1 + (Real.sqrt 2 / 2) * t ∧ y = m + (Real.sqrt 2 / 2) * t))

noncomputable def cartesian_equation_curve : Prop :=
  ∀ (p : ℝ), p > 0 → ∀ (x y : ℝ), y^2 = 2 * p * x

noncomputable def value_of_m (m : ℝ) : Prop :=
  ∃ (A B M : ℝ × ℝ), M = (-1, m) ∧ (∃ (p : ℝ), 
    p = 2 ∧ ∀ (x y : ℝ), y^2 = 4 * x → 
    (line_passing_through M m) ∧ 
    (A ≠ B ∧ A, B ∈ curve y^2 = 4 * x) ∧
    let dist_MA := dist M A,
        dist_MB := dist M B,
        dist_AB := dist A B 
    in  ∣dist_MA∣, (1/2)*∣dist_AB∣, ∣dist_MB∣ form_geometric_sequence → 
    m = -2)

-- We construct our main theorem here
theorem main_theorem : ∃ m : ℝ, value_of_m m := 
sorry

end main_theorem_l594_594998


namespace cost_of_8_dozen_oranges_l594_594151

noncomputable def cost_per_dozen (cost_5_dozen : ℝ) : ℝ :=
  cost_5_dozen / 5

noncomputable def cost_8_dozen (cost_5_dozen : ℝ) : ℝ :=
  8 * cost_per_dozen cost_5_dozen

theorem cost_of_8_dozen_oranges (cost_5_dozen : ℝ) (h : cost_5_dozen = 39) : cost_8_dozen cost_5_dozen = 62.4 :=
by
  sorry

end cost_of_8_dozen_oranges_l594_594151


namespace greatest_divisor_same_remainder_l594_594896

theorem greatest_divisor_same_remainder (a b c : ℕ) (h₁ : a = 54) (h₂ : b = 87) (h₃ : c = 172) : 
  ∃ d, (d ∣ (b - a)) ∧ (d ∣ (c - b)) ∧ (d ∣ (c - a)) ∧ (∀ e, (e ∣ (b - a)) ∧ (e ∣ (c - b)) ∧ (e ∣ (c - a)) → e ≤ d) ∧ d = 1 := 
by 
  sorry

end greatest_divisor_same_remainder_l594_594896


namespace integer_part_sum_l594_594545

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := (1 / 2) * (a n + 1 / (a n))

theorem integer_part_sum :
      (a 1 = 1) →
      (∀ n, a n > 0) →
      (∀ n, S n = (1 / 2) * (a n + 1 / (a n))) →
      (⌊∑ i in Finset.range 100 + 1, 1 / S i⌋ = 18) :=
by
  intros
  sorry

end integer_part_sum_l594_594545


namespace angle_range_l594_594261

variable {a b : ℝ^3}

-- Define the conditions 
noncomputable def cond1 := ∥a∥ = 3 * ∥b∥ ∧ ∥a∥ ≠ 0

noncomputable def f (x : ℝ) := (1/2) * x^3 + (1/2) * ∥a∥ * x^2 + (a • b) * x

noncomputable def monotonicallyIncreasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

-- Proof statement
theorem angle_range (h₁ : cond1) (h₂ : monotonicallyIncreasing f) :
  let θ := real.acos ((a • b) / (∥a∥ * ∥b∥))
  0 ≤ θ ∧ θ ≤ π / 3 :=
sorry

end angle_range_l594_594261


namespace right_triangle_legs_l594_594517

-- Right triangle with hypotenuse AB and angle bisector of the right angle dividing AB in a 4:1 ratio
theorem right_triangle_legs (AB : Real) (AC BC : Real) 
  (h1 : AC = 4 * BC)
  (h2 : AC^2 + BC^2 = AB^2) :
  BC = AB / Real.sqrt 17 ∧ AC = 4 * AB / Real.sqrt 17 :=
begin
  -- Proof will be added here
  sorry
end

end right_triangle_legs_l594_594517


namespace largest_possible_factors_l594_594048

theorem largest_possible_factors :
  ∃ (m : ℕ) (q : Fin m → Polynomial ℝ),
    (x : ℝ) → x^10 - 1 = ∏ i, q i ∧ ∀ i,  degree (q i) > 0 ∧ m = 3 :=
by
  sorry

end largest_possible_factors_l594_594048


namespace vehicle_value_fraction_l594_594074

theorem vehicle_value_fraction (V_this_year V_last_year : ℕ)
  (h_this_year : V_this_year = 16000)
  (h_last_year : V_last_year = 20000) :
  (V_this_year : ℚ) / V_last_year = 4 / 5 := by 
  rw [h_this_year, h_last_year]
  norm_num 
  sorry

end vehicle_value_fraction_l594_594074


namespace count_integers_with_sum_of_digits_18_l594_594962

def sum_of_digits (n : ℕ) : ℕ := (n / 100) + (n / 10 % 10) + (n % 10)

def valid_integer_count : ℕ :=
  let range := List.range' 700 (900 - 700 + 1)
  List.length $ List.filter (λ n => sum_of_digits n = 18) range

theorem count_integers_with_sum_of_digits_18 :
  valid_integer_count = 17 :=
sorry

end count_integers_with_sum_of_digits_18_l594_594962


namespace kabadi_players_l594_594018

theorem kabadi_players 
  (Kho_only : ℕ)
  (Both : ℕ)
  (Total : ℕ)
  (h1 : Kho_only = 25)
  (h2 : Both = 5)
  (h3 : Total = 35) : 
  ∃ K : ℕ, Total = K + Kho_only - Both ∧ K = 15 := 
by 
  use 15
  rw [h1, h2, h3]
  simp
  sorry

end kabadi_players_l594_594018


namespace cos_theta_value_l594_594438

variable (θ : ℝ)
noncomputable def f (x : ℝ) : ℝ := sin x - 2 * cos x

theorem cos_theta_value (h : ∀ x, f θ ≥ f x) : cos θ = - (real.sqrt 5 / 5) := sorry

end cos_theta_value_l594_594438


namespace factorial_multiple_l594_594008

theorem factorial_multiple (m n : ℕ) : 
  ∃ k : ℕ, k * (m! * n! * (m + n)!) = (2 * m)! * (2 * n)! :=
sorry

end factorial_multiple_l594_594008


namespace largest_angle_is_158_5_l594_594481

/- Let the interior angles of the pentagon be defined as given measures in terms of x -/

def pentagon_angles (x : ℝ) : List ℝ := [2 * x + 2, 3 * x - 3, 4 * x + 1, 5 * x, 6 * x - 5]

/- Define the total sum of the angles of the pentagon -/

def sum_of_pentagon_angles (x : ℝ) : ℝ := List.sum (pentagon_angles x)

/- Define the largest angle in the pentagon given x -/

def largest_angle (x : ℝ) : ℝ := List.maximum (pentagon_angles x) 

/- Formalize the problem statement in Lean 4 where we need to prove the largest angle equals 158.5 degrees given the total sum of the angles equals 540 degrees -/

theorem largest_angle_is_158_5 (x : ℝ) 
  (h: sum_of_pentagon_angles x = 540) : 
  largest_angle x = 158.5 :=
by
  sorry

end largest_angle_is_158_5_l594_594481


namespace even_function_a_eq_neg1_l594_594342

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * (Real.exp x + a * Real.exp (-x))

/-- Given that the function f(x) = x(e^x + a e^{-x}) is an even function, prove that a = -1. -/
theorem even_function_a_eq_neg1 (a : ℝ) (h : ∀ x : ℝ, f a x = f a (-x)) : a = -1 :=
sorry

end even_function_a_eq_neg1_l594_594342


namespace sum_greater_or_equal_l594_594550

variables {n : ℕ} {a b : ℕ → ℝ} -- defining variables for our sequences

-- defining the conditions
def conditions (a b : ℕ → ℝ) : Prop :=
(∀ i, 1 ≤ i → i ≤ n → a i > 0) ∧
(∀ i, 1 < i → i ≤ n → a (i - 1) ≥ a i) ∧
(b 1 ≥ a 1) ∧
(∀ i, 2 ≤ i → i ≤ n → b 1 * b i ≥ a 1 * a i) ∧
(∀ i, 3 ≤ i → i ≤ n → b 1 * b (i - 1) * b i ≥ a 1 * a (i - 1) * a i) ∧
(b 1 * b 2 * ⋯ * b n ≥ a 1 * a 2 * ⋯ * a n)

-- the goal/proof to be established
theorem sum_greater_or_equal (a b : ℕ → ℝ) (h : conditions a b) : 
  (finset.range n).sum b ≥ (finset.range n).sum a := sorry

end sum_greater_or_equal_l594_594550
