import Mathlib

namespace radius_of_Q3_l92_92040

theorem radius_of_Q3 
  (A B C : Type) 
  (AB BC AC : ℕ) 
  (h₁ : AB = 80)
  (h₂ : BC = 80)
  (h₃ : AC = 96)
  (Q1 Q2 Q3 : Type)
  (Q1_tangent_to_ABC : Q1 → Prop)
  (Q2_tangent_to_Q1_AB_BC : Q2 → Prop)
  (Q3_tangent_to_Q2_AB_BC : Q3 → Prop)
  (r1 r2 r3 : ℕ)
  (r1_correct : r1 = 24)
  (r2_correct : r2 = 6) : 
  r3 = 1.5 :=
  sorry

end radius_of_Q3_l92_92040


namespace max_value_of_z_l92_92058

variable (x y a : ℝ)

def condition1 : Prop := cos x + sin y = 1/2
def condition2 : Prop := -1 ≤ cos x ∧ cos x ≤ 1
def condition3 : Prop := a ∈ Set.univ

theorem max_value_of_z (h1 : condition1 x y) (h2 : condition2 x) (h3 : condition3 a) : 
  let z := λ (x y a : ℝ), a * sin y + cos x ^ 2
  (z x y a).sup =
    if a ≤ 1/2 then 1 - a/2 else a + 1/4 := 
sorry

end max_value_of_z_l92_92058


namespace eccentricity_of_ellipse_l92_92731

open Real

-- Definitions of the foci, point, and ellipse conditions.
def is_foci (a b : ℝ) (F1 F2 : ℝ × ℝ) :=
  F1 = (-sqrt (a^2 - b^2), 0) ∧ F2 = (sqrt (a^2 - b^2), 0)

def on_line_x (P : ℝ × ℝ) (a : ℝ) :=
  P.1 = a

def is_isosceles_triangle (F1 F2 P : ℝ × ℝ) :=
  ∠ (P - F1) (F2 - F1) = π / 6 ∧ ∠ (P - F2) (F1 - F2) = π / 6

def eccentricity (a c : ℝ) :=
  c / a

-- Hypotheses from the problem
variables {a b : ℝ} (F1 F2 P : ℝ × ℝ)
variables (h_foci : is_foci a b F1 F2)
variables (h_a_gt_b : a > b)
variables (hP_on_line : on_line_x P a)
variables (h_isosceles : is_isosceles_triangle F1 F2 P)

-- Goal
theorem eccentricity_of_ellipse : eccentricity a (sqrt (a^2 - b^2)) = sqrt (3) / 2 :=
sorry

end eccentricity_of_ellipse_l92_92731


namespace equilateral_triangle_side_length_l92_92295

theorem equilateral_triangle_side_length 
    (D A B C : ℝ × ℝ)
    (h_distances : dist D A = 2 ∧ dist D B = 3 ∧ dist D C = 5)
    (h_equilateral : dist A B = dist B C ∧ dist B C = dist C A) :
    dist A B = Real.sqrt 19 :=
by
    sorry -- Proof to be filled

end equilateral_triangle_side_length_l92_92295


namespace quoted_value_correct_l92_92381

-- Define the context and conditions
def investment : ℝ := 1620
def dividendEarned : ℝ := 135
def dividendRate : ℝ := 8

-- Define the statement we want to prove
theorem quoted_value_correct :
  let faceValue := (dividendEarned * 100) / dividendRate in
  let quotedValue := (investment / faceValue) * 100 in
  quotedValue = 96 :=
by
  sorry

end quoted_value_correct_l92_92381


namespace area_of_quadrilateral_DPFQ_l92_92800

-- Definitions of points
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨1, 0, 0⟩
def C : Point3D := ⟨1, 1, 0⟩
def D : Point3D := ⟨0, 1, 0⟩
def E : Point3D := ⟨0, 0, 1⟩
def F : Point3D := ⟨1, 0, 1⟩
def G : Point3D := ⟨1, 1, 1⟩
def H : Point3D := ⟨0, 1, 1⟩

-- Definitions of midpoints
def P : Point3D := ⟨0, 0, 1/2⟩
def Q : Point3D := ⟨1, 1, 1/2⟩

-- Euclidean distance function for calculating length of diagonals
def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2)

-- Diagonal lengths
def DF := distance D F
def PQ := distance P Q

-- Area of quadrilateral DPFQ
def area_DPFQ := (1/2) * DF * PQ

-- Theorem statement
theorem area_of_quadrilateral_DPFQ : area_DPFQ = real.sqrt 6 / 2 := by
  sorry

end area_of_quadrilateral_DPFQ_l92_92800


namespace sin_90_eq_1_l92_92439

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92439


namespace greatest_integer_100x_l92_92178

-- Define x based on the given sums of cosines and sines
def x : ℝ := (∑ n in range 44, real.cos (n * real.pi / 180)) / (∑ n in range 44, real.sin (n * real.pi / 180))

-- Define the goal to prove the greatest integer that does not exceed 100x is 241
theorem greatest_integer_100x : (⌊100 * x⌋₊ : ℤ) = 241 := 
by
  -- The proof goes here but is omitted
  sorry

end greatest_integer_100x_l92_92178


namespace intersection_points_count_l92_92104

noncomputable def line1 : Set (ℝ × ℝ) := { p | 3 * p.2 - 2 * p.1 = 1 }
noncomputable def line2 : Set (ℝ × ℝ) := { p | p.1 + 2 * p.2 = 2 }
noncomputable def line3 : Set (ℝ × ℝ) := { p | 4 * p.1 - 6 * p.2 = 5 }

def countIntersections : ℕ :=
  let points := (line1 ∩ line2) ∪ (line1 ∩ line3) ∪ (line2 ∩ line3)
  Set.card points

theorem intersection_points_count : countIntersections = 2 :=
  sorry

end intersection_points_count_l92_92104


namespace smallest_positive_angle_equivalent_neg_1990_l92_92248

theorem smallest_positive_angle_equivalent_neg_1990:
  ∃ k : ℤ, 0 ≤ (θ : ℤ) ∧ θ < 360 ∧ -1990 + 360 * k = θ := by
  use 6
  sorry

end smallest_positive_angle_equivalent_neg_1990_l92_92248


namespace stratified_sampling_third_grade_l92_92804

theorem stratified_sampling_third_grade (r1 r2 r3 total_sample : ℝ) 
  (hr1 : r1 = 2) (hr2 : r2 = 3) (hr3 : r3 = 5) (htotal_sample : total_sample = 200) :
  (r3 / (r1 + r2 + r3)) * total_sample = 100 := 
by
  sorry

end stratified_sampling_third_grade_l92_92804


namespace green_center_area_l92_92918

variable {s : ℝ} (h : 0 < s)

-- The entire cross occupies 40% of the flag's area
def entire_cross_area := 0.4 * s^2

-- The red arms alone occupy 36% of the flag's area
def red_arms_area := 0.36 * s^2

-- Prove that the green center occupies 4% of the area of the flag
theorem green_center_area : (entire_cross_area h - red_arms_area h) / (s^2) = 0.04 :=
by
  sorry

end green_center_area_l92_92918


namespace complex_purely_imaginary_l92_92095

theorem complex_purely_imaginary (a : ℝ) :
  (a^2 + a - 2 = 0) ∧ (a^2 - 3a + 2 ≠ 0) → a = -2 :=
by
  sorry

end complex_purely_imaginary_l92_92095


namespace compute_sin_90_l92_92507

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92507


namespace fourth_number_is_two_eighth_number_is_two_l92_92853

-- Conditions:
-- 1. Initial number on the board is 1
-- 2. Sequence of medians observed by Mitya

def initial_number : ℕ := 1
def medians : list ℚ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- Required proof statements:

-- a) The fourth number written on the board is 2
theorem fourth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 3 = some 2 :=
sorry

-- b) The eighth number written on the board is 2
theorem eighth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 7 = some 2 :=
sorry

end fourth_number_is_two_eighth_number_is_two_l92_92853


namespace three_equilateral_triangles_union_area_l92_92824

theorem three_equilateral_triangles_union_area :
  let s := Real.sqrt 3
  let area_one_triangle := (Real.sqrt 3 / 4) * s * s
  let total_area_without_overlaps := 3 * area_one_triangle
  let side_small_triangle := s / 2
  let area_small_triangle := (Real.sqrt 3 / 4) * side_small_triangle * side_small_triangle
  let overlap_area := 2 * area_small_triangle
  let net_area := total_area_without_overlaps - overlap_area
  in net_area = (15 * Real.sqrt 3) / 8 :=
by
  sorry

end three_equilateral_triangles_union_area_l92_92824


namespace hexagonal_bipyramid_probability_l92_92517

theorem hexagonal_bipyramid_probability :
  ∀ (hexagonal_bipyramid : Type)
    (top_vertex bottom_vertex : hexagonal_bipyramid)
    (middle_ring_vertices : Fin 6 → hexagonal_bipyramid),
  (∀ i : Fin 6, ∃ j : Fin 6, middle_ring_vertices i ≠ middle_ring_vertices j) →
  (∀ i : Fin 6, ∃! j : Fin 6, middle_ring_vertices j = bottom_vertex) →
  (∃ A : hexagonal_bipyramid, ∃ B : hexagonal_bipyramid,
    A ∈ (Set.range middle_ring_vertices) ∧
    B ∈ (Set.range middle_ring_vertices) ∧
    B = bottom_vertex) →
  (1 / 6 : ℝ) :=
by
  sorry

end hexagonal_bipyramid_probability_l92_92517


namespace smallest_composite_no_prime_factors_less_than_20_l92_92994

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92994


namespace optimal_room_rate_l92_92232

def max_revenue_room_rate (rooms : ℕ) (current_rate : ℝ) (occupancy_rate : ℝ)
                           (reduction_step : ℝ) (increase_occupancy_per_step : ℕ) : ℝ :=
(current_rate - reduction_step * (rooms - increase_occupancy_per_step)) *
(occupancy_rate * rooms + increase_occupancy_per_step)

theorem optimal_room_rate :
  ∃ rate : ℝ, rate = 300 ∧ ∀ rooms occupancy_rate reduction_step increase_occupancy_per_step,
  rooms = 100 ∧ 
  current_rate = 400 ∧ 
  occupancy_rate = 0.5 ∧ 
  reduction_step = 20 ∧ 
  increase_occupancy_per_step = 5 →
  max_revenue_room_rate rooms current_rate occupancy_rate reduction_step increase_occupancy_per_step = 22500 :=
begin
  sorry
end

end optimal_room_rate_l92_92232


namespace number_of_girls_l92_92765

theorem number_of_girls {total_children boys girls : ℕ} 
  (h_total : total_children = 60) 
  (h_boys : boys = 18) 
  (h_girls : girls = total_children - boys) : 
  girls = 42 := by 
  sorry

end number_of_girls_l92_92765


namespace expectation_X_expectation_Y_l92_92892

-- Definitions of initial conditions
def initial_red_balls : ℕ := 3
def initial_white_balls : ℕ := 3

-- Definitions of the Options
def option1_draw_once (bag : ℕ × ℕ) : punit → Prob (ℕ × ℕ)
| punit.unit :=
  if bag.1 = initial_red_balls then
    uniform [(bag.1, bag.2), (bag.1+1, bag.2-1)]
  else
    uniform [(bag.1+1, bag.2-1)]

def option2_draw_once : ν {bag : set (pair ℕ ℕ)} := 
  do b ← uniform [(1,1)]
  -- Define the draw twice more according to similar logic as above (abstracted for simplicity)

-- Setting up the Probability Space and Conditions for Normal Distribution
def normal_dist (μ σ : ℝ) : dist := dist.normal μ σ

-- Main statement (skip implementation as requested)
theorem expectation_X :
  -- Assuming other required definitions and theorems.
  E[X] = 307 / 72 :=
sorry

theorem expectation_Y :
  -- Assuming other required definitions and theorems.
  E[Y] = 9 / 2 :=
sorry

end expectation_X_expectation_Y_l92_92892


namespace intersection_condition_l92_92077

noncomputable def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
noncomputable def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem intersection_condition (k : ℝ) (h : M ⊆ N k) : k ≥ 2 :=
  sorry

end intersection_condition_l92_92077


namespace find_m_eq_2_l92_92617

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l92_92617


namespace trapezoid_circumscribed_l92_92788

noncomputable def is_circumscribed (t: trapezoid) := sorry -- Definition needed

variables (α β : Real)
variables (ABCD : trapezoid)
variables (AD BC : Real)

-- Angles at the base AD
def angles_at_base (t: trapezoid) (α β : Real) : Prop :=
  -- The assertion here is symbolic, real proof would need concrete definitions
  sorry

-- The main theorem to prove
theorem trapezoid_circumscribed (ABCD : trapezoid) (α β : Real) (AD BC : ℝ)
  (h_angles : angles_at_base ABCD α β) :
  (is_circumscribed ABCD) ↔ (BC / AD = tan(α) * tan(β)) :=
sorry

end trapezoid_circumscribed_l92_92788


namespace coefficient_x4_in_expression_l92_92542

theorem coefficient_x4_in_expression : 
  let expr := 4 * (Polynomial.C 1 * Polynomial.X ^ 4 - Polynomial.C 2 * Polynomial.X ^ 5) + 
              3 * (Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 1 * Polynomial.X ^ 4 - Polynomial.C 2 * Polynomial.X ^ 6) - 
              (Polynomial.C 5 * Polynomial.X ^ 5 - Polynomial.C 2 * Polynomial.X ^ 4)
  in Polynomial.coeff expr 4 = 3 := 
by 
  sorry

end coefficient_x4_in_expression_l92_92542


namespace oranges_left_to_sell_today_l92_92191

theorem oranges_left_to_sell_today (initial_dozen : Nat)
    (reserved_fraction1 reserved_fraction2 sold_fraction eaten_fraction : ℚ)
    (rotten_oranges : Nat) 
    (h1 : initial_dozen = 7)
    (h2 : reserved_fraction1 = 1/4)
    (h3 : reserved_fraction2 = 1/6)
    (h4 : sold_fraction = 3/7)
    (h5 : eaten_fraction = 1/10)
    (h6 : rotten_oranges = 4) : 
    let total_oranges := initial_dozen * 12
    let reserved1 := total_oranges * reserved_fraction1
    let reserved2 := total_oranges * reserved_fraction2
    let remaining_after_reservation := total_oranges - reserved1 - reserved2
    let sold_yesterday := remaining_after_reservation * sold_fraction
    let remaining_after_sale := remaining_after_reservation - sold_yesterday
    let eaten_by_birds := remaining_after_sale * eaten_fraction
    let remaining_after_birds := remaining_after_sale - eaten_by_birds
    let final_remaining := remaining_after_birds - rotten_oranges
    final_remaining = 22 :=
by
    sorry

end oranges_left_to_sell_today_l92_92191


namespace philip_oranges_count_l92_92023

def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def betty_bill_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * betty_bill_oranges
def seeds_planted := frank_oranges * 2
def orange_trees := seeds_planted
def oranges_per_tree : ℕ := 5
def oranges_for_philip := orange_trees * oranges_per_tree

theorem philip_oranges_count : oranges_for_philip = 810 := by sorry

end philip_oranges_count_l92_92023


namespace best_play_wins_majority_l92_92139

variables (n : ℕ)

-- Conditions
def students_in_play_A : ℕ := n
def students_in_play_B : ℕ := n
def mothers : ℕ := 2 * n

-- Question
theorem best_play_wins_majority : 
  (probability_fin_votes_wins_majority (students_in_play_A n) (students_in_play_B n) (mothers n)) = 1 - (1/2)^n :=
sorry

end best_play_wins_majority_l92_92139


namespace integer_values_count_l92_92249

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l92_92249


namespace range_of_m_for_inequality_l92_92661

-- Define the condition
def condition (x : ℝ) := x ∈ Set.Iic (-1)

-- Define the inequality for proving the range of m
def inequality_holds (m x : ℝ) : Prop := (m - m^2) * 4^x + 2^x + 1 > 0

-- Prove the range of m for the given conditions such that the inequality holds
theorem range_of_m_for_inequality :
  (∀ (x : ℝ), condition x → inequality_holds m x) ↔ (-2 < m ∧ m < 3) :=
sorry

end range_of_m_for_inequality_l92_92661


namespace find_x_l92_92884

theorem find_x :
  ∃ x : ℝ, 1 / x = 16.666666666666668 ∧ x = 1 / 60 := 
by
  use 1 / 60
  split
  . simp
  . norm_num
  sorry

end find_x_l92_92884


namespace compute_sin_90_l92_92506

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92506


namespace best_play_majority_two_classes_l92_92144

theorem best_play_majority_two_classes (n : ℕ) :
  let prob_win := 1 - (1/2) ^ n
  in prob_win = 1 - (1/2) ^ n :=
by
  sorry

end best_play_majority_two_classes_l92_92144


namespace sin_90_eq_1_l92_92465

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92465


namespace area_difference_nearest_tenth_l92_92226

theorem area_difference_nearest_tenth
  (s : ℝ) (r : ℝ)
  (h1 : s * s + s * s = 8 * 8)
  (h2 : r = 8 / 2)
  (pi := Real.pi) :
  (round ((pi * r * r - s * s) * 10) / 10 = 18.3) :=
sorry

end area_difference_nearest_tenth_l92_92226


namespace slope_angle_of_line_l92_92808

theorem slope_angle_of_line :
  ∃ θ : ℝ, (x + real.sqrt 3 * y + 5 = 0) → θ = 150 :=
sorry

end slope_angle_of_line_l92_92808


namespace sin_90_eq_1_l92_92479

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92479


namespace smallest_k_l92_92033

theorem smallest_k (a b c d e k : ℕ) (h1 : a + 2 * b + 3 * c + 4 * d + 5 * e = k)
  (h2 : 5 * a = 4 * b) (h3 : 4 * b = 3 * c) (h4 : 3 * c = 2 * d) (h5 : 2 * d = e) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) : k = 522 :=
sorry

end smallest_k_l92_92033


namespace sin_90_degree_l92_92496

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92496


namespace how_many_statements_correct_l92_92069

-- Definitions based on the conditions
def contrapositive_sin (x y : ℝ) : Prop := (sin x ≠ sin y) → (x ≠ y)
def octal_conversion_correct (n : ℕ) : Prop := nat.to_digits 8 n = [3, 7, 3, 7]
def negation_correct (x : ℝ) : Prop := (x^2 = 1) → (x = 1)
def plane_condition (α β : Type) : Prop := ∃ (pts : set α), pts.noncollinear ∧ (∀ p ∈ pts, p ∈ α ∧ dist p β = dist (classical.some spec) β)

-- Main statement to prove
theorem how_many_statements_correct : (∃ x y : ℝ, contrapositive_sin x y) ∧ 
  ¬octal_conversion_correct 2015 ∧ 
  ¬(∀ x : ℝ, negation_correct x) ∧ 
  ¬(∃ (α β : Type), plane_condition α β) → 
  ([1].length = 1) := sorry

end how_many_statements_correct_l92_92069


namespace best_play_majority_two_classes_l92_92143

theorem best_play_majority_two_classes (n : ℕ) :
  let prob_win := 1 - (1/2) ^ n
  in prob_win = 1 - (1/2) ^ n :=
by
  sorry

end best_play_majority_two_classes_l92_92143


namespace exponential_form_theta_l92_92869

noncomputable def complex_number : ℂ := 1 + complex.I * real.sqrt 3

theorem exponential_form_theta :
  ∃ θ : ℝ, complex.exp (complex.I * θ) = complex_number / complex.abs complex_number ∧ θ = real.pi / 3 :=
begin
  use real.pi / 3,
  split,
  { sorry }, -- Proof of the exponential form
  { refl },  -- Proof that θ = π/3
end

end exponential_form_theta_l92_92869


namespace ratio_twice_width_to_length_l92_92234

-- Given conditions:
def length_of_field : ℚ := 24
def width_of_field : ℚ := 13.5

-- The problem is to prove the ratio of twice the width to the length of the field is 9/8
theorem ratio_twice_width_to_length : 2 * width_of_field / length_of_field = 9 / 8 :=
by sorry

end ratio_twice_width_to_length_l92_92234


namespace vector_perpendicular_l92_92051

theorem vector_perpendicular (x : ℝ) :
  let CA := (3, -9)
  let CB := (-3, x)
  (3 * (-3) + (-9) * x = 0) → x = -1 := 
by
  intros CA CB h
  have h1 : 3 * (-3) = -9 := by norm_num
  have h2 : -9 + (-9) * x = 0 := by rw [← h, h1]
  have h3 : -9 * (1 + x) = 0 := by ring
  exact eq_of_mul_eq_zero h3 (by norm_num)

#print axioms vector_perpendicular

end vector_perpendicular_l92_92051


namespace prob_Alex_Mel_Chelsea_wins_l92_92115

/- 
  Alex wins a round with a probability of 3/5
  Mel wins a round with a probability of 3c
  Chelsea wins a round with a probability of c
  Total rounds played = 8
  The outcomes of each round are independent
  The total probability for one round must sum to 1
  Chelsea wins one round
  We need to prove the probability that Alex wins four rounds, Mel wins three 
  rounds, and Chelsea wins one round is 61242/625000.
-/
theorem prob_Alex_Mel_Chelsea_wins :
  let pAlex := (3/5 : ℚ);            -- Alex's win probability
  let c := (1/10 : ℚ);                -- Chelsea's win probability
  let pMel := 3 * c;                -- Mel's win probability
  let total_rounds := 8;
  let sequence_probability := (pAlex^4) * (pMel^3) * (c);
  let sequence_count := (nat.choose 8 4) * (nat.choose 4 3) * (nat.choose 1 1);
  sequence_probability * sequence_count = (61242 / 625000 : ℚ)
:= by
  let pAlex := (3/5 : ℚ)
  let c := (1/10 : ℚ)
  let pMel := 3 * c
  let total_rounds := 8
  let sequence_probability := (pAlex^4) * (pMel^3) * (c)
  let sequence_count := (nat.choose 8 4) * (nat.choose 4 3) * (nat.choose 1 1)
  have : sequence_probability * sequence_count = 61242 / 625000 := sorry
  exact this

end prob_Alex_Mel_Chelsea_wins_l92_92115


namespace binomial_expansion_coefficient_l92_92567

theorem binomial_expansion_coefficient (a : ℝ)
  (h : a = ∫ x in 0..π, sin x) :
  coeff (λ x : ℝ, (1 - a / x) ^ 5) (-3) = -80 :=
by
  have ha : a = 2 := by
    sorry
  rw ha
  sorry

end binomial_expansion_coefficient_l92_92567


namespace triangle_side_difference_l92_92943

-- Define the given conditions
variables (a b c d k : ℝ)
-- Difference of squares condition
hypothesis (h1 : b^2 - c^2 = d^2)

-- Statement to prove
theorem triangle_side_difference (H: AC^2 - AB^2 = b^2 - c^2) : 
    ∃ (ABC : Triangle), (BC = a ∧ median_ak = k) ∧ (h1 : b^2 - c^2 = d^2) ∧ (AC^2 - AB^2 = d^2) := 
sorry

end triangle_side_difference_l92_92943


namespace table_tennis_expected_games_l92_92685

open ProbabilityTheory
open MeasureTheory
open Localization
open ENNReal

noncomputable def expected_games_stop : ℚ :=
  97 / 32

theorem table_tennis_expected_games :
  ∀ (E : Type) [Fintype E],
  let prob_A := (3 / 4 : ℚ)
  let prob_B := (1 / 4 : ℚ)
  let max_games := 6
  let xi : ℕ := expected_value_of_games (prob_A, prob_B) max_games
  (∀ i ∈ (finset.range max_games), i % 2 = 0 → xi <= max_games → xi ≤ max_games) →
  (∑ i in (finset.range max_games), i * (prob_A ^ (i / 2) * prob_B ^ (i / 2)) = expected_games_stop) := 
  sorry

end table_tennis_expected_games_l92_92685


namespace integer_values_count_l92_92251

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l92_92251


namespace determine_M_l92_92639

noncomputable def M : Set ℤ :=
  {a | ∃ k : ℕ, k > 0 ∧ 6 = k * (5 - a)}

theorem determine_M : M = {-1, 2, 3, 4} :=
  sorry

end determine_M_l92_92639


namespace five_digit_integers_count_l92_92651

theorem five_digit_integers_count :
  let choices_first_three := 3 * 3 * 3,
      choices_last_two := 3 * 2 in
  choices_first_three * choices_last_two = 162 :=
by
  let choices_first_three := 3 * 3 * 3
  let choices_last_two := 3 * 2
  calc
    choices_first_three * choices_last_two = 27 * 6 : by sorry
    ... = 162 : by sorry

end five_digit_integers_count_l92_92651


namespace smallest_composite_no_prime_factors_less_than_20_l92_92999

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92999


namespace sin_90_degrees_l92_92488

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92488


namespace probability_point_inside_circle_l92_92105

theorem probability_point_inside_circle : 
  (∑ (m n : ℕ) in (range 1 (6+1)).product (range 1 (6+1)), 
     if m^2 + n^2 < 16 then 1 else 0) / 
  ((6:ℕ) * (6:ℕ)) = 2 / 9 :=
by sorry

end probability_point_inside_circle_l92_92105


namespace sqrt_of_25_l92_92386

theorem sqrt_of_25 : ∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end sqrt_of_25_l92_92386


namespace exists_consecutive_integers_divisible_by_l92_92741

theorem exists_consecutive_integers_divisible_by
  (k : ℕ) (a : ℕ → ℕ)
  (h1 : k > 0)
  (h2 : ∀ i : fin k, a i > 0)
  (h3 : ∀ (i j : fin k), i ≠ j → coprime (a i) (a j)):
  ∃ (n : ℤ), ∀ (j : fin k), (n + j : ℤ) % (a j) = 0 :=
sorry

end exists_consecutive_integers_divisible_by_l92_92741


namespace property_P_M_property_P_N_neg_set_A_properties_arithmetic_sequence_n_eq_3_arithmetic_sequence_n_eq_4_l92_92076

-- Define the properties and the sets M and N
def has_property_P (A : Set ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i ≤ j → j ≤ A.size →
    (A.mem (A.elem j + A.elem i) ∨ A.mem (A.elem j - A.elem i))

def M := {0, 2, 4}
def N := {1, 2, 3}

theorem property_P_M : has_property_P M := sorry

theorem property_P_N_neg : ¬ has_property_P N := sorry

-- Define set A with the given properties and prove the related properties
theorem set_A_properties {A : Set ℕ} (hA : ∀ i j, 1 ≤ i → i ≤ j → j ≤ A.size →
  (A.mem (A.elem j + A.elem i) ∨ A.mem (A.elem j - A.elem i))) 
  (h_ord: ∀ i j, i < j → A.elem i < A.elem j) (n_ge_3 : 3 ≤ A.size) :
  A.elem 0 = 0 ∧ A.sum = (A.size / 2 : ℚ) * (A.elem A.size) := sorry

-- Prove whether sequence is arithmetic for n = 3 or n = 4
theorem arithmetic_sequence_n_eq_3 {A : Set ℕ} (hA : ∀ i j, 1 ≤ i → i ≤ j → j ≤ 3 →
  (A.mem (A.elem j + A.elem i) ∨ A.mem (A.elem j - A.elem i))) 
  (h_ord: ∀ i j, i < j → A.elem i < A.elem j) : 
  ∀ (a1 a2 a3 ∈ A), a3 - a2 = a2 - a1 := sorry

theorem arithmetic_sequence_n_eq_4 {A : Set ℕ} (hA : ∀ i j, 1 ≤ i → i ≤ j → j ≤ 4 →
  (A.mem (A.elem j + A.elem i) ∨ A.mem (A.elem j - A.elem i))) 
  (h_ord: ∀ i j, i < j → A.elem i < A.elem j) : 
  ¬ ∀ (a1 a2 a3 a4 ∈ A), (a4 - a3 = a3 - a2 ∧ a3 - a2 = a2 - a1) := sorry

end property_P_M_property_P_N_neg_set_A_properties_arithmetic_sequence_n_eq_3_arithmetic_sequence_n_eq_4_l92_92076


namespace sum_T_a_b_c_l92_92961

noncomputable def T : ℝ :=
  ∑ n in finset.range 9900,
    1 / real.sqrt (n + 1 + real.sqrt ((n + 1)^2 - 4))

theorem sum_T_a_b_c : 
  ∃ a b c : ℤ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  ¬ ∃ p : ℤ, nat.prime p ∧ p^2 ∣ c ∧ 
  (T = a + b * real.sqrt c) ∧ (a + b + c = 239) :=
begin
  sorry
end

end sum_T_a_b_c_l92_92961


namespace A_beats_B_by_7_seconds_l92_92109

noncomputable def speed_A : ℝ := 200 / 33
noncomputable def distance_A : ℝ := 200
noncomputable def time_A : ℝ := 33

noncomputable def distance_B : ℝ := 200
noncomputable def distance_B_at_time_A : ℝ := 165

-- B's speed is calculated at the moment A finishes the race
noncomputable def speed_B : ℝ := distance_B_at_time_A / time_A
noncomputable def time_B : ℝ := distance_B / speed_B

-- Prove that A beats B by 7 seconds
theorem A_beats_B_by_7_seconds : time_B - time_A = 7 := 
by 
  -- Proof goes here, assume all definitions and variables are correct.
  sorry

end A_beats_B_by_7_seconds_l92_92109


namespace sin_90_eq_1_l92_92440

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92440


namespace clock_shows_one_hour_ahead_l92_92196

theorem clock_shows_one_hour_ahead :
  ∃ t : ℕ, t = 92 ∧ (12 + t) % 24 = 8 ∧ (24 * ((12 + t) / 24)) / 24 + days 12 = friday  
  where 12 == 12_on_monday -- clock shows correct time at 12:00 PM on Monday
  clock_loses_hour_per_4_hours : ∀ t : ℕ, (92 * 4) / 4 = 23  -- clock loses 1 hour every 4 actual hours
  sorry

end clock_shows_one_hour_ahead_l92_92196


namespace function_classification_l92_92003

theorem function_classification {f : ℝ → ℝ} 
    (h : ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) : 
    ∀ x : ℝ, f x = 0 ∨ f x = 1 :=
by
  sorry

end function_classification_l92_92003


namespace rationalize_denominator_eq_sum_l92_92776

theorem rationalize_denominator_eq_sum :
  let A := 1
  let B := 1
  let C := 1
  let D := −1
  let E := 70
  let F := 12
  (A + B + C + D + E + F) = 84 :=
by
  -- The proof would go here
  sorry

end rationalize_denominator_eq_sum_l92_92776


namespace min_S_l92_92643

variables {V : Type*} [InnerProductSpace ℝ V]
open InnerProductSpace

theorem min_S (a b : V) (h₀ : a ≠ 0) (h₁ : b ≠ 0) :
  ∃ (S : ℝ), (∀ (x y : V), is_permutation [x, y] [a, a, b, b] →
    S = inner x.1 y.1 + inner x.2 y.2 + inner x.3 y.3 + inner x.4 y.4) ∧ S = 4 * inner a b :=
by {
  sorry
}

end min_S_l92_92643


namespace rectangular_box_diagonals_sum_l92_92518

theorem rectangular_box_diagonals_sum
    (x y z : ℝ) 
    (h1 : 2 * (x * y + y * z + z * x) = 118)
    (h2 : 4 * (x + y + z) = 60)
    (h3 : z = 2 * y) :
    4 * real.sqrt ((x ^ 2) + (y ^ 2) + (z ^ 2)) = 20 * real.sqrt 3 := by
  sorry

end rectangular_box_diagonals_sum_l92_92518


namespace sin_90_deg_l92_92450

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92450


namespace length_of_perpendicular_segment_l92_92209

theorem length_of_perpendicular_segment
  (A B C D E F G X : Type)
  (RS : Set (Type))
  (AD BE CF CX : ℝ)
  (h1 : AD = 12)
  (h2 : BE = 8)
  (h3 : CF = 30)
  (h4 : CX = 15)
  (h5 : ∀ A D, A ∈ RS ∧ D ∈ RS → AD ⊥ RS)
  (h6 : ∀ B E, B ∈ RS ∧ E ∈ RS → BE ⊥ RS)
  (h7 : ∀ C F, C ∈ RS ∧ F ∈ RS → CF ⊥ RS)
  (h8 : C ∈ RS ∧ X ∈ RS → CX = 15) :
  ∃ x, x = 35 / 3 := 
sorry

end length_of_perpendicular_segment_l92_92209


namespace exists_equal_segments_l92_92738

theorem exists_equal_segments (a : ℕ → ℕ) (h : ∀ i, a (i + 1000) = a i) : 
  ∃ k (hk : 100 ≤ k ∧ k ≤ 300), ∃ i, (∑ j in finset.range k, a (i + j)) = (∑ j in finset.range k, a (i + k + j)) :=
by
  -- Definitions
  sorry

end exists_equal_segments_l92_92738


namespace domain_of_function_l92_92228

theorem domain_of_function :
  (∀ x : ℝ, (2 * Real.sin x - 1 > 0) ∧ (1 - 2 * Real.cos x ≥ 0) ↔
    ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 3 ≤ x ∧ x < 2 * k * Real.pi + 5 * Real.pi / 6) :=
sorry

end domain_of_function_l92_92228


namespace abs_sum_eq_n_squared_l92_92799

theorem abs_sum_eq_n_squared (n : ℕ) (a b : Fin n → ℕ)
  (h_disjoint : ∀ i j, i ≠ j → a i ≠ a j ∧ b i ≠ b j ∧ a i ≠ b j)
  (h_union : ∀ k : ℕ, k ∈ (Finset.range (2 * n + 1)).erase 0 → 
    ∃ i, k = a i ∨ k = b i)
  (h_ascend : ∀ i j : Fin n, i < j → a i < a j)
  (h_descend : ∀ i j : Fin n, i < j → b i > b j) :
  ∑ i : Fin n, abs (a i - b i) = n^2 := 
begin
  sorry
end

end abs_sum_eq_n_squared_l92_92799


namespace sin_ninety_degrees_l92_92401

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92401


namespace Yvonne_probability_of_success_l92_92876

theorem Yvonne_probability_of_success
  (P_X : ℝ) (P_Z : ℝ) (P_XY_notZ : ℝ) :
  P_X = 1 / 3 →
  P_Z = 5 / 8 →
  P_XY_notZ = 0.0625 →
  ∃ P_Y : ℝ, P_Y = 0.5 :=
by
  intros hX hZ hXY_notZ
  existsi (0.5 : ℝ)
  sorry

end Yvonne_probability_of_success_l92_92876


namespace minimize_total_distance_l92_92580

open Real

-- Define the points on the line
variables {Q Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 : ℝ}

-- Define the distance function
def distance (x y : ℝ) : ℝ := abs (x - y)

-- Define the total distance sum function
def total_distance_sum (Q : ℝ) :=
  distance Q Q1 + distance Q Q2 + distance Q Q3 + distance Q Q4 + 
  distance Q Q5 + distance Q Q6 + distance Q Q7 + distance Q Q8 + distance Q Q9

-- Theorem statement: The point Q that minimizes the total distance sum is Q5
theorem minimize_total_distance : total_distance_sum Q = ∑ i in {Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9}, distance Q5 Q :=
begin
  sorry
end

end minimize_total_distance_l92_92580


namespace eighth_group_number_correct_stratified_sampling_below_30_correct_l92_92680

noncomputable def systematic_sampling_eighth_group_number 
  (total_employees : ℕ) (sample_size : ℕ) (groups : ℕ) (fifth_group_number : ℕ) : ℕ :=
  let interval := total_employees / groups
  let initial_number := fifth_group_number - 4 * interval
  initial_number + 7 * interval

theorem eighth_group_number_correct :
  systematic_sampling_eighth_group_number 200 40 40 22 = 37 :=
  sorry

noncomputable def stratified_sampling_below_30_persons 
  (total_employees : ℕ) (sample_size : ℕ) (percent_below_30 : ℕ) : ℕ :=
  (percent_below_30 * sample_size) / 100

theorem stratified_sampling_below_30_correct :
  stratified_sampling_below_30_persons 200 40 40 = 16 :=
  sorry

end eighth_group_number_correct_stratified_sampling_below_30_correct_l92_92680


namespace a_n_formula_b_n_formula_l92_92578

noncomputable def a_n (n : ℕ) : ℕ :=
  2^(n-1)

noncomputable def b_n (n : ℕ) : ℚ :=
  if n = 1 then 1 else 2^(n-2) / n

theorem a_n_formula (n : ℕ) : a_n n = 2^(n-1) := 
  sorry

theorem b_n_formula (n : ℕ) : 
  (∑ i in Finset.range (n+1), (i + 1) * b_n (i + 1)) = a_n n := 
  sorry

end a_n_formula_b_n_formula_l92_92578


namespace f_six_l92_92062

noncomputable def f : ℝ → ℝ
| x := if x < 0 then x ^ 3 - 1 else
       if -1 ≤ x ∧ x ≤ 1 then if x ≥ 0 then Classical.choose sorry else -Classical.choose sorry else
       if x > 1/2 then Classical.choose sorry else 0

axiom f_periodic {x : ℝ} (hx : x > 1/2) : f (x + 1/2) = f (x - 1/2)
axiom f_odd {x : ℝ} (hx : -1 ≤ x ∧ x ≤ 1) : f (-x) = -f (x)
axiom f_def_neg {x : ℝ} (hx : x < 0) : f x = x ^ 3 - 1

theorem f_six : f 6 = 2 := by
  sorry

end f_six_l92_92062


namespace sin_90_deg_l92_92453

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92453


namespace possible_red_ball_draws_l92_92307

/-- 
Given two balls in a bag where one is white and the other is red, 
if a ball is drawn and returned, and then another ball is drawn, 
prove that the possible number of times a red ball is drawn is 0, 1, or 2.
-/
theorem possible_red_ball_draws : 
  (∀ balls : Finset (ℕ × ℕ), 
    balls = {(0, 1), (1, 0)} →
    ∀ draw1 draw2 : ℕ × ℕ, 
    draw1 ∈ balls →
    draw2 ∈ balls →
    ∃ n : ℕ, (n = 0 ∨ n = 1 ∨ n = 2) ∧ 
    n = (if draw1 = (1, 0) then 1 else 0) + 
        (if draw2 = (1, 0) then 1 else 0)) → 
    True := sorry

end possible_red_ball_draws_l92_92307


namespace sin_90_degree_l92_92493

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92493


namespace compare_abc_l92_92028

open Real

noncomputable def a := log 0.8 / log 0.7
noncomputable def b := log 0.8 / log 1.2
noncomputable def c := 1.2 ^ 0.7

theorem compare_abc (a_def : a = log 0.8 / log 0.7) (b_def : b = log 0.8 / log 1.2) (c_def : c = 1.2 ^ 0.7) : 
  c > a ∧ a > b :=
sorry

end compare_abc_l92_92028


namespace thirty_ml_of_one_liter_is_decimal_fraction_l92_92314

-- We define the known conversion rule between liters and milliliters.
def liter_to_ml := 1000

-- We define the volume in milliliters that we are considering.
def volume_ml := 30

-- We state the main theorem which asserts that 30 ml of a liter is equal to the decimal fraction 0.03.
theorem thirty_ml_of_one_liter_is_decimal_fraction : (volume_ml / (liter_to_ml : ℝ)) = 0.03 := by
  -- insert proof here
  sorry

end thirty_ml_of_one_liter_is_decimal_fraction_l92_92314


namespace length_of_PQ_l92_92528

-- Definitions based on the conditions
def isEquilateralTriangle (A B C : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

-- The important points and their coordinates
variables {A B C Ap P Q : Point}

-- Given conditions
def conditions (A B C Ap : Point) : Prop :=
  isEquilateralTriangle A B C ∧
  dist B C = 5 ∧
  dist B Ap = 2 ∧
  dist Ap C = 3

-- The proof problem statement
theorem length_of_PQ (A B C Ap P Q : Point) (h : conditions A B C Ap) : dist P Q = 3 * Real.sqrt 3 :=
sorry

end length_of_PQ_l92_92528


namespace messages_in_February_l92_92159

theorem messages_in_February :
  let messages : ℕ → ℕ :=
    λ n, match n with
    | 0 => 1  -- November: 1 text message 
    | 1 => 2  -- December: 2 text messages
    | 2 => 4  -- January: 4 text messages
    | 3 => 8  -- February: To prove
    | 4 => 16 -- March: 16 text messages
    | _ => 0
  in
  messages 3 = 8 :=
by {
  -- Given the sequence follows doubling each month
  -- November: messages 0 = 1
  -- December: messages 1 = 2
  -- January: messages 2 = 4
  -- February: messages 3 = 8
  sorry
}

end messages_in_February_l92_92159


namespace max_dot_and_area_of_triangle_l92_92155

noncomputable def triangle_data (A B C : ℝ) (m n : ℝ × ℝ) : Prop :=
  A + B + C = Real.pi ∧
  (m = (2, 2 * (Real.cos ((B + C) / 2))^2 - 1)) ∧
  (n = (Real.sin (A / 2), -1))

noncomputable def is_max_dot_product (A : ℝ) (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = (if A = Real.pi / 3 then 3 / 2 else 0)

noncomputable def max_area (A B C : ℝ) : ℝ :=
  let a : ℝ := 2
  let b : ℝ := 2
  let c : ℝ := 2
  if A = Real.pi / 3 then (Real.sqrt 3) else 0

theorem max_dot_and_area_of_triangle {A B C : ℝ} {m n : ℝ × ℝ}
  (h_triangle : triangle_data A B C m n) :
  is_max_dot_product (Real.pi / 3) m n ∧ max_area A B C = Real.sqrt 3 := by sorry

end max_dot_and_area_of_triangle_l92_92155


namespace sin_90_eq_1_l92_92477

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92477


namespace distance_between_foci_of_ellipse_l92_92969

theorem distance_between_foci_of_ellipse (x y : ℝ) :
  9 * x^2 + y^2 = 36 → 2 * real.sqrt (36 - 4) = 8 * real.sqrt 2 :=
by
  intro h
  calc
    2 * real.sqrt (36 - 4) = 2 * real.sqrt (32) : sorry
    ...                   = 2 * 4 * real.sqrt 2  : sorry
    ...                   = 8 * real.sqrt 2      : sorry

end distance_between_foci_of_ellipse_l92_92969


namespace sin_90_eq_1_l92_92444

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92444


namespace distance_between_foci_of_ellipse_l92_92973

theorem distance_between_foci_of_ellipse (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 36 → 2 * Real.sqrt (b^2 - a^2) = 8 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end distance_between_foci_of_ellipse_l92_92973


namespace find_m_eq_2_l92_92616

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l92_92616


namespace find_cube_edge_length_l92_92227

-- Define the conditions of the problem
def parallelepiped_volume (l w h : ℕ) : ℕ := l * w * h

def parallelepiped_surface_area (l w h : ℕ) : ℕ := 
  2 * (l * w + w * h + h * l)

def cube_volume (a : ℕ) : ℕ := a^3

def cube_surface_area (a : ℕ) : ℕ := 6 * a^2

theorem find_cube_edge_length :
  let l := 2 in
  let w := 3 in
  let h := 6 in
  let V_par := parallelepiped_volume l w h in
  let S_par := parallelepiped_surface_area l w h in
  ∀ a : ℕ, 
    V_par / cube_volume a = S_par / cube_surface_area a → 
    a = 3 :=
by 
  sorry

end find_cube_edge_length_l92_92227


namespace int_values_satisfy_condition_l92_92259

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l92_92259


namespace tangent_line_eq_no_min_value_in_interval_min_value_ln_a_min_value_a_div_e_l92_92587

noncomputable def f (a x : ℝ) : ℝ := a / x + Real.log x - 1

noncomputable def tangent_line_at (a x : ℝ) :=
  x - 4 * (f 1 x) + 4 * Real.log 2 - 4 = 0

theorem tangent_line_eq (a : ℝ) (h : a = 1) :
  tangent_line_at 1 2 := sorry

theorem no_min_value_in_interval (a : ℝ) (h : a ≤ 0) :
  ∀ x ∈ Set.Ioo (0 : ℝ) Real.exp,
  ∃ y ∈ Set.Ioo (0 : ℝ) Real.exp, f a y < f a x := sorry

theorem min_value_ln_a (a : ℝ) (h : 0 < a) (h' : a < Real.exp) :
  ∀ x, x ∈ Set.Ioc (0 : ℝ) Real.exp → f a x ≥ Real.log a := sorry

theorem min_value_a_div_e (a : ℝ) (h : a ≥ Real.exp) :
  ∀ x, x ∈ Set.Ioc (0 : ℝ) Real.exp → 
  f a x ≥ a / Real.exp := sorry

end tangent_line_eq_no_min_value_in_interval_min_value_ln_a_min_value_a_div_e_l92_92587


namespace fifth_term_sequence_l92_92941

theorem fifth_term_sequence : 
  (4 + 8 + 16 + 32 + 64) = 124 := 
by 
  sorry

end fifth_term_sequence_l92_92941


namespace triangle_sides_divisible_by_prime_powers_l92_92728

theorem triangle_sides_divisible_by_prime_powers 
  (p : ℕ) (hp : p.prime) (p_odd : p % 2 = 1)
  (n : ℕ) (hn : 0 < n)
  (points : fin 8 → ℤ × ℤ)
  (on_circle : ∀ i, (points i).fst ^ 2 + (points i).snd ^ 2 = (p^n/2)^2) :
  ∃ (i j k : fin 8), 
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
    p^(n+1) ∣ ((points i).fst - (points j).fst)^2 + ((points i).snd - (points j).snd)^2 ∧
    p^(n+1) ∣ ((points j).fst - (points k).fst)^2 + ((points j).snd - (points k).snd)^2 ∧
    p^(n+1) ∣ ((points k).fst - (points i).fst)^2 + ((points k).snd - (points i).snd)^2 :=
sorry

end triangle_sides_divisible_by_prime_powers_l92_92728


namespace sum_of_super_cool_areas_l92_92912

noncomputable def is_super_cool_right_triangle (a b : ℕ) : Prop :=
  (a * b / 2) = 3 * (a + b)

theorem sum_of_super_cool_areas :
  let pairs := [(7, 42), (8, 24), (9, 18), (10, 15), (12, 12)] in
  let areas := list.map (λ (ab : ℕ × ℕ), ab.1 * ab.2 / 2) pairs in
  list.sum areas = 471 :=
by
  sorry

end sum_of_super_cool_areas_l92_92912


namespace pelican_count_in_shark_bite_cove_l92_92021

theorem pelican_count_in_shark_bite_cove
  (num_sharks_pelican_bay : ℕ)
  (num_pelicans_shark_bite_cove : ℕ)
  (num_pelicans_moved : ℕ) :
  num_sharks_pelican_bay = 60 →
  num_sharks_pelican_bay = 2 * num_pelicans_shark_bite_cove →
  num_pelicans_moved = num_pelicans_shark_bite_cove / 3 →
  num_pelicans_shark_bite_cove - num_pelicans_moved = 20 :=
by
  sorry

end pelican_count_in_shark_bite_cove_l92_92021


namespace general_term_sequence_a_sum_sequence_b_l92_92043

-- Define the sequence a_n satisfying the given condition
def seq_a (n : ℕ) (k : ℝ) : ℝ := (2 * n ^ 2 + n + k) / (n + 1)

-- Define the general term formula
def general_term_a (n : ℕ) : ℝ := 2 * n - 1

-- Define the sequence b_n
def seq_b (n : ℕ) (a_n a_n1 : ℝ) : ℝ := 4 * n ^ 2 / (a_n * a_n1)

-- Define the sum of the first n terms of b_n
def sum_b (n : ℕ) : ℝ := (2 * n ^ 2 + 2 * n) / (2 * n + 1)

-- The Lean 4 statements for the two proofs
theorem general_term_sequence_a (n : ℕ) (k : ℝ) : 
  (∃ a_n : ℕ → ℝ, ∀ n, (n + 1) * a_n n = 2 * n ^ 2 + n + k ∧ a_n n = general_term_a n) :=
begin
  sorry
end

theorem sum_sequence_b (n : ℕ) (a_n : ℕ → ℝ) : 
  (∃ b_n S_n : ℕ → ℝ, 
    (∀ n, b_n n = seq_b n (a_n n) (a_n (n + 1))) 
    ∧ S_n n = ∑ i in finset.range n, b_n i
    ∧ S_n n = sum_b n) :=
begin
  sorry
end

end general_term_sequence_a_sum_sequence_b_l92_92043


namespace roxanne_bought_2_cups_l92_92777

variable (x : ℕ) -- Denote the number of cups of lemonade as a natural number

-- Conditions
def cost_cup_lemonade (x : ℕ) := 2 * x
def cost_sandwiches := 2 * 2.50
def total_payment := 20
def change_received := 11

-- Total amount spent
def total_spent (x : ℕ) := total_payment - change_received

-- Proof that the number of cups of lemonade Roxanne bought is 2
theorem roxanne_bought_2_cups :
  cost_cup_lemonade x + cost_sandwiches = total_spent x → x = 2 :=
by
  intros h
  sorry

end roxanne_bought_2_cups_l92_92777


namespace sin_90_eq_1_l92_92471

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92471


namespace calculation_l92_92385

theorem calculation : (1 / 2) ^ (-2 : ℤ) + (-1 : ℝ) ^ (2022 : ℤ) = 5 := by
  sorry

end calculation_l92_92385


namespace acute_triangle_condition_l92_92171

theorem acute_triangle_condition (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (|a^2 - b^2| < c^2 ∧ c^2 < a^2 + b^2) ↔ (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) :=
sorry

end acute_triangle_condition_l92_92171


namespace log_value_bounds_l92_92291

theorem log_value_bounds (h1 : log 10 10000 = 4) (h2 : log 10 100000 = 5) (h3 : 10000 < 52789 ∧ 52789 < 100000) : 
  ∃ c d, (c = 4 ∧ d = 5) ∧ 4 < log 10 52789 ∧ log 10 52789 < 5 ∧ c + d = 9 :=
by
  sorry

end log_value_bounds_l92_92291


namespace line_passes_fixed_point_and_min_area_l92_92607

theorem line_passes_fixed_point_and_min_area
  (k : ℝ) : 
  (∀ k, ∃ p : ℝ × ℝ, p = (-2, 1) ∧ (k * (fst p + 2) + (1 - snd p) = 0)) ∧
  ((∃ A B : ℝ × ℝ,
      (A = (-2 - 1/k, 0)) ∧
      (B = (0, 2*k + 1)) ∧
      (2*k + 1 > 0) ∧
      let S := (1/2) * abs ((-2 - 1/k)*(2*k + 1)) in
      ∀ k > 0, S ≥ 4 ∧ 
      (S = 4 → k = 1/2)) →
   (∃ k : ℝ, k = 1/2 ∧ l = 1/2 * x - y + 2))
  
  sorry

end line_passes_fixed_point_and_min_area_l92_92607


namespace minimum_fence_length_l92_92311

variable (x y : ℝ)

-- Given conditions
axiom area_eq : x * y = 100
axiom fence_length_eq : ∀ x y, l = 2 * (x + y)

-- Proof problem statement
theorem minimum_fence_length : ∃ l, l ≥ 40 ∧ (∀ x y, x * y = 100 → l = 2 * (x + y)) := by
  sorry

end minimum_fence_length_l92_92311


namespace order_of_numbers_l92_92949

theorem order_of_numbers :
  0.7 ^ 6 > 0 ∧ 0.7 ^ 6 < 1 ∧ 6 ^ 0.7 > 1 ∧ log 0.7 6 < 0 →
  log 0.7 6 < 0.7 ^ 6 ∧ 0.7 ^ 6 < 6 ^ 0.7 :=
by
  intros h
  sorry

end order_of_numbers_l92_92949


namespace intersection_points_count_l92_92103

noncomputable def line1 : Set (ℝ × ℝ) := { p | 3 * p.2 - 2 * p.1 = 1 }
noncomputable def line2 : Set (ℝ × ℝ) := { p | p.1 + 2 * p.2 = 2 }
noncomputable def line3 : Set (ℝ × ℝ) := { p | 4 * p.1 - 6 * p.2 = 5 }

def countIntersections : ℕ :=
  let points := (line1 ∩ line2) ∪ (line1 ∩ line3) ∪ (line2 ∩ line3)
  Set.card points

theorem intersection_points_count : countIntersections = 2 :=
  sorry

end intersection_points_count_l92_92103


namespace sin_ninety_deg_l92_92436

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92436


namespace shelves_used_l92_92369

-- Define the initial conditions
def initial_stock : Float := 40.0
def additional_stock : Float := 20.0
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_stock + additional_stock

-- Define the number of shelves
def number_of_shelves : Float := total_books / books_per_shelf

-- The proof statement that needs to be proven
theorem shelves_used : number_of_shelves = 15.0 :=
by
  -- The proof will go here
  sorry

end shelves_used_l92_92369


namespace inequality_holds_l92_92772

theorem inequality_holds (a b : ℝ) : (6 * a - 3 * b - 3) * (a ^ 2 + a ^ 2 * b - 2 * a ^ 3) ≤ 0 :=
sorry

end inequality_holds_l92_92772


namespace find_angle_B_max_area_triangle_l92_92156

-- Define the vectors m and n
def vec_m (B : ℝ) : ℝ × ℝ := (2 * Real.sin B, -Real.sqrt 3)
def vec_n (B : ℝ) : ℝ × ℝ := (Real.cos (2 * B), 2 * (Real.cos (B / 2)) ^ 2 - 1)

-- Define that vectors m and n are parallel
def vectors_parallel (B : ℝ) : Prop :=  ∃ k : ℝ, vec_m B = (k * (vec_fst (vec_n B)), k * (vec_snd (vec_n B)))

-- Define that side b equals 2
def side_b_eq_2 : Prop := b = 2

-- First proof problem: Given the vectors m and n are parallel, find the measure of angle B
theorem find_angle_B (B : ℝ) (h : vectors_parallel B) : B = Real.pi / 3 := sorry

-- Second proof problem: Given B = π/3 and b = 2, find the maximum area of triangle ABC
theorem max_area_triangle (a c : ℝ) (B : ℝ) (hB : B = Real.pi / 3) (hb : side_b_eq_2) : 
  ∃ S : ℝ, S = Real.sqrt 3 := sorry

end find_angle_B_max_area_triangle_l92_92156


namespace value_of_m_l92_92622

theorem value_of_m (a m : ℝ) (h : a > 0) (hm : m ≠ 0) :
  (P : ℝ × ℝ) (P = (m, 3))
  (H : ∀ x : ℝ, -a * x^2 + 2 * a * x + 3 = 3 → x = 0 ∨ x = 2) :
  m = 2 :=
by
  sorry

end value_of_m_l92_92622


namespace mouse_cannot_end_at_central_l92_92893

/-- A cube \(3 \times 3 \times 3\) is made up of 27 smaller cubes.
The mouse starts at one of the corners and moves to adjacent pieces (sharing a face).
Prove that it is not possible for the last piece the mouse has eaten to be the central one. -/
theorem mouse_cannot_end_at_central :
  let cube_size := 3
  ∧ let cubes := Fin₃ cube_size × Fin₃ cube_size × Fin₃ cube_size
  ∧ let start_positions := {(i, j, k) | i = 0 ∨ i = cube_size - 1 ∨ j = 0 ∨ j = cube_size - 1 ∨ k = 0 ∨ k = cube_size - 1}
  ∧ let adjacent (a b : Fin₃ cube_size × Fin₃ cube_size × Fin₃ cube_size) := 
      (abs (a.1 - b.1) + abs (a.2 - b.2) + abs (a.3 - b.3) = 1)
  ∧ let central_cube := (1, 1, 1 : Fin₃ cube_size × Fin₃ cube_size × Fin₃ cube_size)
  ∧ ∀ (path : list (Fin₃ cube_size × Fin₃ cube_size × Fin₃ cube_size)),
      (path.head ∈ start_positions)
      ∧ (∀ n < path.length - 1, adjacent (path.nth_le n (sorry)) (path.nth_le (n + 1) (sorry)))
      ∧ (path.nodup)
      ∧ (path.length = 27)
  ∧ (path.last (sorry) = central_cube) :=
  false :=
sorry

end mouse_cannot_end_at_central_l92_92893


namespace spot_roam_area_l92_92781

-- Definitions based on the conditions
def side_length : ℝ := 1
def rope_length : ℝ := 2
def half_circle_area : ℝ := (1/2) * Real.pi * (rope_length)^2
def sector_area : ℝ := (1/4) * Real.pi * (side_length)^2
def total_area : ℝ := half_circle_area + 2 * sector_area

-- Lean 4 statement of the problem
theorem spot_roam_area :
  total_area = (2.5 : ℝ) * Real.pi := sorry

end spot_roam_area_l92_92781


namespace find_m_l92_92630

theorem find_m (a m : ℝ) (h_pos : a > 0) (h_points : (m, 3) ∈ set_of (λ x : ℝ × ℝ, ∃ x_val : ℝ, x.snd = -a * (x_val)^2 + 2 * a * x_val + 3)) (h_non_zero : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92630


namespace rect_coord_line_l_rect_coord_curve_C_distance_AB_l92_92700

-- Define the conditions
def curve_C (p θ : ℝ) : Prop := p ^ 2 = 12 / (2 + cos θ)
def line_l (p θ : ℝ) : Prop := 2 * p * cos (θ - π / 6) = sqrt 3

-- Defining the equivalency statements
theorem rect_coord_line_l (x y : ℝ) (p θ : ℝ) 
    (h : line_l p θ) : sqrt 3 * x + y = sqrt 3 := sorry

theorem rect_coord_curve_C (x y : ℝ) (p θ : ℝ) 
    (h : curve_C p θ) : x ^ 2 / 4 + y ^ 2 / 6 = 1 := sorry

#check rect_coord_line_l
#check rect_coord_curve_C

-- The problem involving the intersection points and distance computation
theorem distance_AB (x1 y1 x2 y2 : ℝ) 
    (h1 : sqrt 3 * x1 + y1 = sqrt 3) (h2 : x2 ^ 2 / 4 + y2 ^ 2 / 6 = 1)
    (h3 : curve_C _ _) (h4 : line_l _ _) : abs (dist (x1, y1) (x2, y2)) = 4 * sqrt 10 / 3 := sorry

#check distance_AB

end rect_coord_line_l_rect_coord_curve_C_distance_AB_l92_92700


namespace sin_90_eq_1_l92_92481

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92481


namespace intersection_AB_l92_92668

variable {x : ℝ}

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_AB : A ∩ B = {x | 0 < x ∧ x < 2} :=
by sorry

end intersection_AB_l92_92668


namespace sin_ninety_deg_l92_92428

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92428


namespace P_is_circumcenter_l92_92053

variables {P A B C D E F IA IB IC : Point}
variables (h1 : InsideTriangle P A B C)
variables (h2 : Projection P D B C)
variables (h3 : Projection P E C A)
variables (h4 : Projection P F A B)
variables (h5 : AP^2 + PD^2 = BP^2 + PE^2)
variables (h6 : BP^2 + PE^2 = CP^2 + PF^2)
variables (h7 : Excenter IA A B C)
variables (h8 : Excenter IB B C A)
variables (h9 : Excenter IC C A B)

theorem P_is_circumcenter (P A B C D E F IA IB IC : Point)
  (h1 : InsideTriangle P A B C)
  (h2 : Projection P D B C)
  (h3 : Projection P E C A)
  (h4 : Projection P F A B)
  (h5 : AP^2 + PD^2 = BP^2 + PE^2)
  (h6 : BP^2 + PE^2 = CP^2 + PF^2)
  (h7 : Excenter IA A B C)
  (h8 : Excenter IB B C A)
  (h9 : Excenter IC C A B) : 
  IsCircumcenter P IA IB IC :=
sorry

end P_is_circumcenter_l92_92053


namespace find_w_l92_92784

theorem find_w :
  (∀ p q r : ℝ, (p + q + r = 2) ∧ (p * q + q * r + r * p = 5) ∧ (p * q * r = 8) →
   (∃ u v : ℝ, (u + (p+q) + (q+r) + (r+p) = 0) ∧ 
               (v + ((p+q) * (q+r) + (q+r) * (r+p) + (r+p) * (p+q)) = 0) ∧ 
               ((p+q) * (q+r) * (r+p) = -w) ∧ w = 34)) := 
begin
  intros p q r h,
  obtain ⟨h1, h2, h3⟩ := h,
  use (- (p + q + r)), use (p*q + q*r + r*p),
  split, { simp [h1] },
  split, { sorry },
  split, { sorry },
  exact rfl,
end

end find_w_l92_92784


namespace transform_point_l92_92239

variables (p : ℝ × ℝ × ℝ) (q : ℝ × ℝ × ℝ)
def rotate_90_about_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(p.1, -p.3, p.2)

def reflect_through_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(p.1, -p.2, p.3)

def reflect_through_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(-p.1, p.2, p.3)

theorem transform_point :
  let initial_point := (2, 3, -4)
  let step1 := rotate_90_about_x initial_point
  let step2 := reflect_through_xz step1
  let step3 := reflect_through_yz step2
  let step4 := rotate_90_about_x step3
  let step5 := reflect_through_yz step4
  step5 = (2, -3, -4) :=
by {
  sorry
}

end transform_point_l92_92239


namespace sqrt_10_bounds_l92_92817

theorem sqrt_10_bounds : 10 > 9 ∧ 10 < 16 → 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := 
by 
  sorry

end sqrt_10_bounds_l92_92817


namespace find_third_vertex_l92_92833

open Real

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (9, 3)
def vertex2 : ℝ × ℝ := (0, 0)

-- Define the conditions
def on_negative_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ p.1 < 0

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

-- Statement of the problem in Lean
theorem find_third_vertex :
  ∃ (vertex3 : ℝ × ℝ), 
    on_negative_x_axis vertex3 ∧ 
    area_of_triangle vertex1 vertex2 vertex3 = 45 ∧
    vertex3 = (-30, 0) :=
sorry

end find_third_vertex_l92_92833


namespace segment_division_l92_92679

-- Definitions of the conditions
def radius : ℝ := 6
def diameter : ℝ := 2 * radius
def chord_length : ℝ := 10

-- Theorem statement
theorem segment_division (m n : ℝ) (h_sum : m + n = diameter) (h_product : m * n = (chord_length / 2) ^ 2) :
  (m = 6 + sqrt 11 ∧ n = 6 - sqrt 11) ∨ (m = 6 - sqrt 11 ∧ n = 6 + sqrt 11) :=
by
  sorry

end segment_division_l92_92679


namespace value_of_b_plus_c_l92_92029

theorem value_of_b_plus_c 
  (b c : ℝ) 
  (f : ℝ → ℝ)
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_solution_set : ∀ x, f x ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) :
  b + c = -1 :=
sorry

end value_of_b_plus_c_l92_92029


namespace floor_u_n_eq_l92_92364

noncomputable def u : ℕ → ℝ
| 0       := 2
| 1       := 5 / 2
| (n + 2) := u (n + 1) * (u n ^ 2 - 2) - u 1

theorem floor_u_n_eq : ∀ n : ℕ, 0 < n →
  Int.floor (u n) = 2 ^ ((2 ^ n - (-1:ℤ) ^ n) / 3) :=
by 
  intros n hn
  sorry

end floor_u_n_eq_l92_92364


namespace students_tried_out_l92_92305

theorem students_tried_out (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ)
    (h1 : not_picked = 5)
    (h2 : groups = 3)
    (h3 : students_per_group = 4) :
    (not_picked + groups * students_per_group = 17) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end students_tried_out_l92_92305


namespace find_zero_function_l92_92541

noncomputable def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x ^ 714 + y) = f (x ^ 2019) + f (y ^ 122)

theorem find_zero_function (f : ℝ → ℝ) (h : satisfiesCondition f) :
  ∀ x : ℝ, f x = 0 :=
sorry

end find_zero_function_l92_92541


namespace best_play_majority_win_probability_l92_92133

theorem best_play_majority_win_probability (n : ℕ) :
  (1 - (1 / 2) ^ n) = probability_best_play_wins_majority n :=
sorry

end best_play_majority_win_probability_l92_92133


namespace circles_through_FM_tangent_l_l92_92593

-- Given definitions
def parabola_focus_directrix (p : ℝ) (M : ℝ × ℝ) : Prop :=
  ∃ F l, let FM := (M.1 - F.1)^2 + (M.2 - F.2)^2 in
  let dist_to_directrix := abs(l - M.2) in
  (∃ c : ℝ × ℝ, (c.1 - M.1)^2 + (c.2 - M.2)^2 = FM ∧
                dist_to_directrix = FM ∧ 
                c ∈ parabola)

-- Proving the number of circles is either 1 or 2
theorem circles_through_FM_tangent_l (F : ℝ × ℝ) (l : ℝ) (M : ℝ × ℝ) (p : ℝ) :
  parabola_focus_directrix p M → ∃ n, n = 1 ∨ n = 2 :=
by
  -- The proof will be developed here
  sorry

end circles_through_FM_tangent_l_l92_92593


namespace hexagon_ratio_correct_l92_92361

noncomputable def hexagon_ratio {a : ℝ} : ℝ :=
  let c := sqrt 3 * a in
  let f1 := (-c, 0) in
  let f2 := (c, 0) in
  let B := (0, a) in
  let A := (a * sqrt 2, a) in
  let F1F2 := 2 * c in
  let side_length := dist A B in
  let perimeter := 6 * side_length in
  perimeter / F1F2

theorem hexagon_ratio_correct (a : ℝ) (ha : a ≠ 0) :
  hexagon_ratio = sqrt 6 := by
  sorry

end hexagon_ratio_correct_l92_92361


namespace dodecahedron_projection_regular_decagon_icosahedron_projection_regular_hexagon_l92_92205

-- Definitions of the polyhedra
structure Dodecahedron :=
  (has_faces : ∀ f, f ∈ faces -> is_regular_pentagon f)

structure Icosahedron :=
  (has_faces : ∀ f, f ∈ faces -> is_equilateral_triangle f)

-- Projections of the polyhedra
def projection_parallel_to_face (P : Polyhedron) (f : Face) : Polygon := sorry

theorem dodecahedron_projection_regular_decagon (D : Dodecahedron) (f : Face) (h : f ∈ D.faces) :
  is_regular_decagon (projection_parallel_to_face D f) := sorry

theorem icosahedron_projection_regular_hexagon (I : Icosahedron) (f : Face) (h : f ∈ I.faces) :
  is_regular_hexagon (projection_parallel_to_face I f) := sorry

end dodecahedron_projection_regular_decagon_icosahedron_projection_regular_hexagon_l92_92205


namespace mean_median_changes_l92_92117

def original_attendees : List ℕ := [15, 30, 40, 35, 25]
def corrected_attendees : List ℕ := [15, 30, 50, 35, 25]

noncomputable def mean (l : List ℕ) : ℚ := l.sum / l.length

noncomputable def median (l : List ℕ) : ℚ :=
  let sorted := l.qsort (≤)
  if sorted.length % 2 = 1 then sorted.nth_le (sorted.length / 2) sorry
  else (sorted.nth_le (sorted.length / 2 - 1) sorry + sorted.nth_le (sorted.length / 2) sorry) / 2

theorem mean_median_changes :
  let original_mean := mean original_attendees
  let corrected_mean := mean corrected_attendees
  let original_median := median original_attendees
  let corrected_median := median corrected_attendees
  corrected_mean - original_mean = 2 ∧ corrected_median = original_median := by
  sorry

end mean_median_changes_l92_92117


namespace chessboard_zero_within_kn_operations_l92_92823

-- Defining the problem conditions
variable (n k : ℕ)

-- Given an n × n chessboard where ...
-- Each square holds a number between 0 and k
-- Each row and column has a button that increases the numbers in that row or column by 1 (mod k+1)

theorem chessboard_zero_within_kn_operations (chessboard: Array (Array ℕ)) :
  (∀ i j, chessboard[i][j] ∈ Fin (k + 1)) → 
  ∃ (buttons : Array ℕ), buttons.size ≤ n ∧
  (∀ i, buttons[0] = 0) → -- Initial condition: all squares are filled 0
  (∃ ops : ℕ, ops ≤ k * n ∧ -- ops represents total button presses
  (∀ i, (buttons[i] + chessboard[i]) % (k + 1) = 0)) := 
sorry

end chessboard_zero_within_kn_operations_l92_92823


namespace min_value_of_sum_squares_l92_92054

theorem min_value_of_sum_squares (a b : ℝ) (h : (9 / a^2) + (4 / b^2) = 1) : a^2 + b^2 ≥ 25 :=
sorry

end min_value_of_sum_squares_l92_92054


namespace infinitely_many_not_fibonatic_l92_92551

def F (a n : ℕ) : ℕ :=
  match n with
  | 0   => 0
  | 1   => 1
  | 2   => a
  | (n+2) => F a (n+1) + F a n

def is_fibonatic (n : ℕ) : Prop :=
  ∃ a k : ℕ, 3 < k ∧ F a k = n

theorem infinitely_many_not_fibonatic :
  ∃ infinitely_many (n : ℕ), ¬ is_fibonatic n :=
sorry

end infinitely_many_not_fibonatic_l92_92551


namespace avg_diff_l92_92370

theorem avg_diff (n : ℕ) (m : ℝ) (mistake : ℝ) (true_value : ℝ)
   (h_n : n = 30) (h_mistake : mistake = 15) (h_true_value : true_value = 105) 
   (h_m : m = true_value - mistake) : 
   (m / n) = 3 := 
by
  sorry

end avg_diff_l92_92370


namespace centers_pass_through_fixed_point_l92_92166

open EuclideanGeometry

noncomputable def nine_point_circle_center (A B C H : Point) : Point :=
  sorry

theorem centers_pass_through_fixed_point
  {A B C P H K L M : Point}
  (height_AH : height A H)
  (circle_midpoints : ∀ p ∈ [K, L, M], circle_through [B, P, C, p])
  (P_not_in_BC : P ∉ line_through_points B C)
  (center_PBH PCH : Point)
  (center_PBH_def : center_of_circle center_PBH (circle_through [P, B, H]))
  (center_PCH_def : center_of_circle center_PCH (circle_through [P, C, H])) :
  ∃ N : Point,
    N = nine_point_circle_center A B C H ∧
    line_through_points center_PBH center_PCH N :=
sorry

end centers_pass_through_fixed_point_l92_92166


namespace train_length_l92_92920

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) (conversion_factor : ℝ) :
  speed_kmh = 54 →
  time_s = 33333333333333336 / 1000000000000000 →
  bridge_length_m = 140 →
  conversion_factor = 1000 / 3600 →
  ∃ (train_length_m : ℝ), 
    speed_kmh * conversion_factor * time_s + bridge_length_m = train_length_m + bridge_length_m :=
by
  intros
  use 360
  sorry

end train_length_l92_92920


namespace digit_sum_is_14_l92_92154

theorem digit_sum_is_14 (P Q R S T : ℕ) 
  (h1 : P = 1)
  (h2 : Q = 0)
  (h3 : R = 2)
  (h4 : S = 5)
  (h5 : T = 6) :
  P + Q + R + S + T = 14 :=
by 
  sorry

end digit_sum_is_14_l92_92154


namespace sin_ninety_deg_l92_92432

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92432


namespace difference_highest_lowest_score_l92_92343

-- Definitions based on conditions
def total_innings : ℕ := 46
def avg_innings : ℕ := 61
def highest_score : ℕ := 202
def avg_excl_highest_lowest : ℕ := 58
def innings_excl_highest_lowest : ℕ := 44

-- Calculated total runs
def total_runs : ℕ := total_innings * avg_innings
def total_runs_excl_highest_lowest : ℕ := innings_excl_highest_lowest * avg_excl_highest_lowest
def sum_of_highest_lowest : ℕ := total_runs - total_runs_excl_highest_lowest
def lowest_score : ℕ := sum_of_highest_lowest - highest_score

theorem difference_highest_lowest_score 
  (h1: total_runs = total_innings * avg_innings)
  (h2: avg_excl_highest_lowest * innings_excl_highest_lowest = total_runs_excl_highest_lowest)
  (h3: sum_of_highest_lowest = total_runs - total_runs_excl_highest_lowest)
  (h4: highest_score = 202)
  (h5: lowest_score = sum_of_highest_lowest - highest_score)
  : highest_score - lowest_score = 150 :=
by
  -- We only need to state the theorem, so we can skip the proof.
  -- The exact statements of conditions and calculations imply the result.
  sorry

end difference_highest_lowest_score_l92_92343


namespace sin_90_degree_l92_92497

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92497


namespace AN_bisects_CL_l92_92703

-- Define the given conditions and statement in Lean 4.
variable {A B C N L : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited N] [Inhabited L]

-- Assume the geometric properties
variables (triangle_ABC : Triangle A B C)
variables (right_angle_C : angle A C B = 90)
variables (midpoint_N : midpoint (segment B C) N)
variables (angle_bisector_CL : is_angle_bisector (angle A C B) (segment C L))

-- Goal: Prove line AN bisects the angle bisector CL
theorem AN_bisects_CL (AN_bisects_CL : bisector_of_segment (line A N) (segment C L)) : 
  bisector_of_segment (line A N) (segment C L) := by
  sorry -- Proof is omitted

end AN_bisects_CL_l92_92703


namespace eugene_initial_pencils_l92_92529

theorem eugene_initial_pencils (P : ℕ) (h1 : P + 6 = 57) : P = 51 :=
by
  sorry

end eugene_initial_pencils_l92_92529


namespace problem_statement_l92_92378

-- Assume F is a function defined such that given the point (4,4) is on the graph y = F(x)
def F : ℝ → ℝ := sorry

-- Hypothesis: (4, 4) is on the graph of y = F(x)
axiom H : F 4 = 4

-- We need to prove that F(4) = 4
theorem problem_statement : F 4 = 4 :=
by exact H

end problem_statement_l92_92378


namespace num_of_integers_satisfying_sqrt_condition_l92_92275

theorem num_of_integers_satisfying_sqrt_condition : 
  let S := { x : ℤ | 5 < Real.sqrt x ∧ x < 36 }
  in (S.card = 10) :=
begin
  let S := { x : ℤ | 25 < x ∧ x < 36 },
  sorry
end

end num_of_integers_satisfying_sqrt_condition_l92_92275


namespace sin_90_degrees_l92_92482

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92482


namespace branches_and_ornaments_l92_92938

def numberOfBranchesAndOrnaments (b t : ℕ) : Prop :=
  (b = t - 1) ∧ (2 * b = t - 1)

theorem branches_and_ornaments : ∃ (b t : ℕ), numberOfBranchesAndOrnaments b t ∧ b = 3 ∧ t = 4 :=
by
  sorry

end branches_and_ornaments_l92_92938


namespace sin_90_eq_1_l92_92470

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92470


namespace correct_fourth_number_correct_eighth_number_l92_92844

-- Condition: Initial number on the board and sequence of medians
def initial_board : List ℝ := [1]
def medians : List ℝ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- The number written fourth is 2
def fourth_number_written (board : List ℝ) : ℝ := 2

-- The number written eighth is also 2
def eighth_number_written (board : List ℝ) : ℝ := 2

-- Formalizing the conditions and assertions
theorem correct_fourth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    fourth_number_written board = 2 := 
sorry

theorem correct_eighth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    eighth_number_written board = 2 := 
sorry

-- Function to calculate medians from the board (to be implemented)
noncomputable def calculate_medians_from_board (board : List ℝ) : List ℝ := sorry

end correct_fourth_number_correct_eighth_number_l92_92844


namespace count_integers_between_25_and_36_l92_92281

theorem count_integers_between_25_and_36 :
  {x : ℤ | 25 < x ∧ x < 36}.finite.card = 10 :=
by
  sorry

end count_integers_between_25_and_36_l92_92281


namespace sin_90_degree_l92_92495

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92495


namespace fourth_number_on_board_eighth_number_on_board_l92_92849

theorem fourth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 4 ∧ median numbers[3] = 2 :=
sorry

theorem eighth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 8 ∧ median numbers[7] = 2 :=
sorry

end fourth_number_on_board_eighth_number_on_board_l92_92849


namespace towels_folded_in_one_hour_l92_92713

theorem towels_folded_in_one_hour :
  let jane_rate := 12 * 5 -- Jane's rate in towels/hour
  let kyla_rate := 6 * 9  -- Kyla's rate in towels/hour
  let anthony_rate := 3 * 14 -- Anthony's rate in towels/hour
  let david_rate := 4 * 6 -- David's rate in towels/hour
  jane_rate + kyla_rate + anthony_rate + david_rate = 180 := 
by
  let jane_rate := 12 * 5
  let kyla_rate := 6 * 9
  let anthony_rate := 3 * 14
  let david_rate := 4 * 6
  show jane_rate + kyla_rate + anthony_rate + david_rate = 180
  sorry

end towels_folded_in_one_hour_l92_92713


namespace find_discount_percentage_l92_92353

noncomputable def discount_percentage (C : ℝ) : ℝ :=
  let SP1 := 1.20 * C
  let SP2 := 1.50 * C
  let Profit := 0.365 * C
  let SP3 := SP2 * (1 - D)
  let D := (SP2 - (C + Profit)) / SP2
  D * 100

theorem find_discount_percentage (C : ℝ) (D : ℝ)
  (h1 : SP1 = 1.20 * C)
  (h2 : SP2 = 1.50 * C)
  (h3 : Profit = 0.365 * C)
  (h4 : SP3 = SP2 * (1 - D))
  (h5 : Profit = SP3 - C)
  : D = 0.09 :=
by
  sorry

end find_discount_percentage_l92_92353


namespace fourth_number_is_two_eighth_number_is_two_l92_92854

-- Conditions:
-- 1. Initial number on the board is 1
-- 2. Sequence of medians observed by Mitya

def initial_number : ℕ := 1
def medians : list ℚ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- Required proof statements:

-- a) The fourth number written on the board is 2
theorem fourth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 3 = some 2 :=
sorry

-- b) The eighth number written on the board is 2
theorem eighth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 7 = some 2 :=
sorry

end fourth_number_is_two_eighth_number_is_two_l92_92854


namespace initial_population_l92_92240

theorem initial_population (P : ℝ) (h : P * (1.24 : ℝ)^2 = 18451.2) : P = 12000 :=
by
  sorry

end initial_population_l92_92240


namespace fourth_number_is_2_eighth_number_is_2_l92_92856

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end fourth_number_is_2_eighth_number_is_2_l92_92856


namespace inequality_solution_l92_92335

-- Declare the constants m and n
variables (m n : ℝ)

-- State the conditions
def condition1 (x : ℝ) := m < 0
def condition2 := n = -m / 2

-- State the theorem
theorem inequality_solution (x : ℝ) (h1 : condition1 m n) (h2 : condition2 m n) : 
  nx - m < 0 ↔ x < -2 :=
sorry

end inequality_solution_l92_92335


namespace locust_population_doubling_time_l92_92241

theorem locust_population_doubling_time 
  (h: ℕ)
  (initial_population : ℕ := 1000)
  (time_past : ℕ := 4)
  (future_time: ℕ := 10)
  (population_limit: ℕ := 128000) :
  1000 * 2 ^ ((10 + 4) / h) > 128000 → h = 2 :=
by
  sorry

end locust_population_doubling_time_l92_92241


namespace average_speed_is_approx_55_19_l92_92940

-- Definition of the problem's conditions
def speed_on_flat_sand : ℝ := 60
def speed_on_downhill : ℝ := speed_on_flat_sand + 12
def speed_on_uphill : ℝ := speed_on_flat_sand - 18

def harmonic_mean (a b c : ℝ) : ℝ := 3 / (1/a + 1/b + 1/c)

def average_speed : ℝ := harmonic_mean speed_on_flat_sand speed_on_downhill speed_on_uphill

-- Lean 4 statement to verify the average speed
theorem average_speed_is_approx_55_19 : average_speed ≈ 55.19 :=
by
  -- exact calculation not required, verification of approximate result
  sorry

end average_speed_is_approx_55_19_l92_92940


namespace Elaine_rent_percentage_l92_92718

variable (E : ℝ) (last_year_rent : ℝ) (this_year_rent : ℝ)

def Elaine_last_year_earnings (E : ℝ) : ℝ := E

def Elaine_last_year_rent (E : ℝ) : ℝ := 0.20 * E

def Elaine_this_year_earnings (E : ℝ) : ℝ := 1.25 * E

def Elaine_this_year_rent (E : ℝ) : ℝ := 0.30 * (1.25 * E)

theorem Elaine_rent_percentage 
  (E : ℝ) 
  (last_year_rent := Elaine_last_year_rent E)
  (this_year_rent := Elaine_this_year_rent E) :
  (this_year_rent / last_year_rent) * 100 = 187.5 := 
by sorry

end Elaine_rent_percentage_l92_92718


namespace complex_modulus_equivalence_l92_92958

theorem complex_modulus_equivalence :
  abs ((7 - 5 * complex.i) * (3 + 4 * complex.i) + (4 - 3 * complex.i) * (2 + 7 * complex.i)) = real.sqrt 6073 :=
by
  sorry

end complex_modulus_equivalence_l92_92958


namespace initial_tree_height_l92_92873

variable (H : ℝ) -- Let H be the initial height of the tree
def height_at_year (year : ℕ) : ℝ := H + year * 0.4

theorem initial_tree_height :
  (height_at_year H 6 = (height_at_year H 4) + (1 / 7) * (height_at_year H 4)) → 
  H = 4 :=
begin 
  sorry 
end

end initial_tree_height_l92_92873


namespace math_problem_l92_92048

noncomputable def problem1 (a b c : ℝ) (A B C : ℝ) : Prop :=
  let triangle_obtuse := true  -- given obtuse triangle, for simplicity
  let condition1 := (sqrt 2 * a - c) * (cos B) = b * (cos C)
  let solution1 := B = π / 4
  triangle_obtuse → condition1 → solution1

noncomputable def problem2 (A B : ℝ) : Prop :=
  let m := (cos (2 * A) + 1, cos A)
  let n := (1, - (8 / 5))
  let dot_product := m.1 * n.1 + m.2 * n.2 = 0  -- \overrightarrow{m}·\overrightarrow{n} = 0
  let solution2 := tan (π / 4 + A) = 7
  dot_product → solution2

theorem math_problem (a b c A B C : ℝ) : problem1 a b c A B C ∧ problem2 A B := 
by
sry

end math_problem_l92_92048


namespace smallest_composite_no_prime_factors_less_than_20_l92_92995

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92995


namespace sin_90_eq_one_l92_92423

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92423


namespace integral_result_l92_92960

noncomputable def integral_expr := ∫ (x : ℝ), (cos(2 * x) / (cos(x) ^ 2 * sin(x) ^ 2))

theorem integral_result : 
  ∃ C : ℝ, ∀ x : ℝ, integral_expr = cot x - tan x + C := 
sorry

end integral_result_l92_92960


namespace sin_90_eq_1_l92_92443

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92443


namespace people_showed_up_didnt_get_gift_l92_92711

/-- Jack sent out 200 invitations. -/
def invitations : ℕ := 200

/-- 90% of people RSVPed. -/
def rsvp_percentage : ℝ := 0.90

/-- 80% of people who RSVPed actually showed up. -/
def show_up_percentage : ℝ := 0.80

/-- Jack needs 134 thank you cards. -/
def thank_you_cards_needed : ℕ := 134

/-- Calculate the number of people who RSVPed. -/
def people_rsvped : ℕ := (rsvp_percentage * invitations).to_nat

/-- Calculate the number of people who showed up. -/
def people_showed_up : ℕ := (show_up_percentage * people_rsvped).to_nat

/-- The number of people who showed up but didn't get a gift is 10. -/
theorem people_showed_up_didnt_get_gift :
  people_showed_up - thank_you_cards_needed = 10 :=
sorry

end people_showed_up_didnt_get_gift_l92_92711


namespace line_relation_in_perpendicular_planes_l92_92591

-- Let's define the notions of planes and lines being perpendicular/parallel
variables {α β : Plane} {a : Line}

def plane_perpendicular (α β : Plane) : Prop := sorry -- definition of perpendicular planes
def line_perpendicular_plane (a : Line) (β : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line being parallel to a plane
def line_in_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line lying in a plane

-- The theorem stating the relationship given the conditions
theorem line_relation_in_perpendicular_planes 
  (h1 : plane_perpendicular α β) 
  (h2 : line_perpendicular_plane a β) : 
  line_parallel_plane a α ∨ line_in_plane a α :=
sorry

end line_relation_in_perpendicular_planes_l92_92591


namespace find_s_m_l92_92942

theorem find_s_m (s m : ℝ) (t : ℝ) :
  (∀ t : ℝ, let x := s + 3 * t in let y := -2 + m * t in y = 2 * x + 5) →
  s = -7/2 ∧ m = 6 :=
by
  intros h
  have hs := h 0
  have hm := h 1
  sorry

end find_s_m_l92_92942


namespace correct_fourth_number_correct_eighth_number_l92_92843

-- Condition: Initial number on the board and sequence of medians
def initial_board : List ℝ := [1]
def medians : List ℝ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- The number written fourth is 2
def fourth_number_written (board : List ℝ) : ℝ := 2

-- The number written eighth is also 2
def eighth_number_written (board : List ℝ) : ℝ := 2

-- Formalizing the conditions and assertions
theorem correct_fourth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    fourth_number_written board = 2 := 
sorry

theorem correct_eighth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    eighth_number_written board = 2 := 
sorry

-- Function to calculate medians from the board (to be implemented)
noncomputable def calculate_medians_from_board (board : List ℝ) : List ℝ := sorry

end correct_fourth_number_correct_eighth_number_l92_92843


namespace compute_sum_of_valid_a_l92_92516

open Nat

def is_distinct_prime (n : ℕ) : Prop :=
  n.prime ∧ (∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ p^2 + n = (p^2 + n).natAbs ∧ q^2 + n = (q^2 + n).natAbs)

def valid_a (a : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 10 ∧ ∃ p q : ℕ, p.prime ∧ q.prime ∧ p ≠ q ∧ (p^2 + a).prime ∧ (p^2 + a ≠ q^2 + a)

def sum_of_valid_as : ℕ :=
  ∑ a in (range 11).filter valid_a, id a

theorem compute_sum_of_valid_a : sum_of_valid_as = 20 := by
  sorry

end compute_sum_of_valid_a_l92_92516


namespace probability_positive_ball_drawn_is_half_l92_92302

-- Definition of the problem elements
def balls : List Int := [-1, 0, 2, 3]

-- Definition for the event of drawing a positive number
def is_positive (x : Int) : Bool := x > 0

-- The proof statement
theorem probability_positive_ball_drawn_is_half : 
  (List.filter is_positive balls).length / balls.length = 1 / 2 :=
by
  sorry

end probability_positive_ball_drawn_is_half_l92_92302


namespace simplify_and_evaluate_div_fraction_l92_92211

theorem simplify_and_evaluate_div_fraction (a : ℤ) (h : a = -3) : 
  (a - 2) / (1 + 2 * a + a^2) / (a - 3 * a / (a + 1)) = 1 / 6 := by
  sorry

end simplify_and_evaluate_div_fraction_l92_92211


namespace positive_difference_of_A_B_l92_92545

noncomputable def A : ℕ :=
  (finset.range 22).sum (λ n, (2 * n + 1) * (2 * n + 2)) + 43

noncomputable def B : ℕ :=
  (finset.range 21).sum (λ n, (2 * n + 1) * (2 * n + 2)) + (finset.range 22).sum (λ n, 1)

theorem positive_difference_of_A_B :
  |A - B| = 882 :=
by
  sorry

end positive_difference_of_A_B_l92_92545


namespace find_m_l92_92612

variable (a m x : ℝ)

noncomputable def quadratic_function : ℝ → ℝ := λ x, -a * x^2 + 2 * a * x + 3

theorem find_m (h1 : a > 0) (h2 : quadratic_function a m = 3) (h3 : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92612


namespace dress_total_selling_price_l92_92896

theorem dress_total_selling_price (original_price discount_rate tax_rate : ℝ) 
  (h1 : original_price = 100) (h2 : discount_rate = 0.30) (h3 : tax_rate = 0.15) : 
  (original_price * (1 - discount_rate) * (1 + tax_rate)) = 80.5 := by
  sorry

end dress_total_selling_price_l92_92896


namespace complex_triangle_problem_l92_92391

noncomputable def complex_numbers_forming_equilateral_triangle_side_length_24 (a b c : ℂ) :=
  dist a b = 24 ∧ dist b c = 24 ∧ dist c a = 24

theorem complex_triangle_problem (a b c : ℂ) 
  (h_triangle : complex_numbers_forming_equilateral_triangle_side_length_24 a b c)
  (h_sum : complex.abs (a + b + c) = 42) :
  complex.abs (a * b + a * c + b * c) = 588 :=
sorry

end complex_triangle_problem_l92_92391


namespace best_play_wins_majority_l92_92131

/-- Probability that the best play wins with a majority of the votes given the conditions -/
theorem best_play_wins_majority (n : ℕ) :
  let p := 1 - (1 / 2)^n
  in p > (1 - (1 / 2)^n) ∧ p ≤ 1 :=
sorry

end best_play_wins_majority_l92_92131


namespace line_equation_standard_form_l92_92352

variables (a b T : ℝ)
variables (triangle_area : T = (1/2) * a * b)
variables (vertex_1 : (0, b))
variables (vertex_2 : (a, 0))

theorem line_equation_standard_form :
  2 * T * x - a^2 * y + 2 * T * a = 0 :=
sorry

end line_equation_standard_form_l92_92352


namespace sin_90_deg_l92_92459

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92459


namespace product_of_radii_of_tangent_circles_l92_92890

theorem product_of_radii_of_tangent_circles :
  let C : ℝ × ℝ := (3, 4)
  in let p := (λ a : ℝ, (3 - a)^2 + (4 - a)^2 - a^2)
  in (∀ a : ℝ, p a = 0) →
     let roots : List ℝ := [ℝ]  -- placeholder to represent the roots of the equation
     in (roots.length = 2) →
        roots.prod = 25 :=
by
  sorry

end product_of_radii_of_tangent_circles_l92_92890


namespace positive_integer_count_l92_92948

theorem positive_integer_count (n : ℕ) :
  ∃ (count : ℕ), (count = 122) ∧ 
  (∀ (k : ℕ), 27 < k ∧ k < 150 → ((150 * k)^40 > k^80 ∧ k^80 > 3^240)) :=
sorry

end positive_integer_count_l92_92948


namespace tanya_dan_error_l92_92779

theorem tanya_dan_error 
  (a b c d e f g : ℤ)
  (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) (h₅ : e < f) (h₆ : f < g)
  (h₇ : a % 2 = 1) (h₈ : b % 2 = 1) (h₉ : c % 2 = 1) (h₁₀ : d % 2 = 1) 
  (h₁₁ : e % 2 = 1) (h₁₂ : f % 2 = 1) (h₁₃ : g % 2 = 1)
  (h₁₄ : (a + b + c + d + e + f + g) / 7 - d = 3 / 7) :
  false :=
by sorry

end tanya_dan_error_l92_92779


namespace range_of_a_l92_92570

-- Define the function f
def f (a : ℝ) (x : ℝ) := a * real.exp x - (1 / 2) * x^2

-- Express the condition for critical points
def is_critical_point (a x : ℝ) := f a x = x

-- Define the main theorem statement using given conditions
theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h1 : is_critical_point a x1) (h2 : is_critical_point a x2) (h3 : x2 / x1 ≥ 2) : 
  0 < a ∧ a ≤ real.ln 2 / 2 :=
sorry  -- Proof to be provided

end range_of_a_l92_92570


namespace find_angle_C_l92_92674

-- Define the problem parameters
variables (A B C : ℝ) -- Angles in the triangle
variables (a b c : ℝ) -- Sides opposite to angles A, B, and C respectively

-- Given conditions
def problem_conditions : Prop :=
  a = 2 * Real.sqrt 3 ∧
  c = 2 * Real.sqrt 2 ∧
  (1 + (Real.tan A / Real.tan B) = (2 * c / b))

-- Statement of the proof
theorem find_angle_C (h : problem_conditions A B C a b c) : C = Real.pi / 4 :=
sorry

end find_angle_C_l92_92674


namespace best_play_majority_two_classes_l92_92145

theorem best_play_majority_two_classes (n : ℕ) :
  let prob_win := 1 - (1/2) ^ n
  in prob_win = 1 - (1/2) ^ n :=
by
  sorry

end best_play_majority_two_classes_l92_92145


namespace puppies_adopted_per_day_l92_92905

theorem puppies_adopted_per_day 
    (initial_puppies : ℕ) 
    (additional_puppies : ℕ) 
    (total_days : ℕ) 
    (total_puppies : ℕ)
    (H1 : initial_puppies = 5) 
    (H2 : additional_puppies = 35) 
    (H3 : total_days = 5) 
    (H4 : total_puppies = initial_puppies + additional_puppies) : 
    total_puppies / total_days = 8 := by
  sorry

end puppies_adopted_per_day_l92_92905


namespace oblique_projection_correctness_l92_92834

structure ProjectionConditions where
  intuitive_diagram_of_triangle_is_triangle : Prop
  intuitive_diagram_of_parallelogram_is_parallelogram : Prop

theorem oblique_projection_correctness (c : ProjectionConditions)
  (h1 : c.intuitive_diagram_of_triangle_is_triangle)
  (h2 : c.intuitive_diagram_of_parallelogram_is_parallelogram) :
  c.intuitive_diagram_of_triangle_is_triangle ∧ c.intuitive_diagram_of_parallelogram_is_parallelogram :=
by
  sorry

end oblique_projection_correctness_l92_92834


namespace solution_set_inequality_one_range_m_nonempty_set_l92_92071

def f (x : ℝ) := abs (x + 1) - abs (x - 2)

theorem solution_set_inequality_one : 
  {x | f x ≥ 1} = {x | x ≥ 1} :=
by
  sorry

theorem range_m_nonempty_set (m : ℝ) : 
  (∃ x, f x ≥ x^2 - x + m) ↔ m ≤ 5 / 4 :=
by
  sorry

end solution_set_inequality_one_range_m_nonempty_set_l92_92071


namespace num_valid_complex_numbers_l92_92739

noncomputable def g (z : ℂ) : ℂ := z^2 - 2 * complex.I * z + 2

theorem num_valid_complex_numbers :
  let s := {z : ℂ | im z > 0 ∧ ∃ a b : ℤ, g(z) = a + b * complex.I ∧ |a| ≤ 5 ∧ |b| ≤ 5} in
  fintype.card s = 20 :=
by
  sorry

end num_valid_complex_numbers_l92_92739


namespace sin_90_degrees_l92_92490

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92490


namespace sin_90_eq_1_l92_92467

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92467


namespace fourth_number_is_two_eighth_number_is_two_l92_92851

-- Conditions:
-- 1. Initial number on the board is 1
-- 2. Sequence of medians observed by Mitya

def initial_number : ℕ := 1
def medians : list ℚ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- Required proof statements:

-- a) The fourth number written on the board is 2
theorem fourth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 3 = some 2 :=
sorry

-- b) The eighth number written on the board is 2
theorem eighth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 7 = some 2 :=
sorry

end fourth_number_is_two_eighth_number_is_two_l92_92851


namespace ratio_of_width_to_length_l92_92296

theorem ratio_of_width_to_length (w l : ℕ) (h1 : w * l = 800) (h2 : l - w = 20) : w / l = 1 / 2 :=
by sorry

end ratio_of_width_to_length_l92_92296


namespace curve_intersection_max_OB_OA_l92_92699

noncomputable def max_OB_OA : ℝ :=
  let C1_polar (θ : ℝ) : ℝ := sqrt (4 / (3 * (sin θ)^2 + 1))
  let C2_polar (θ : ℝ) : ℝ := 4 * cos θ
  if hα : α ≠ 0 
  then (max (4 * sqrt 3 / 3))
  else 0

theorem curve_intersection_max_OB_OA {α : ℝ} (hα : 0 ≤ α ∧ α ≤ π) :
  (∃ (A B : ℝ), θ = α ∧ (C1_polar α) = A ∧ (C2_polar α) = B) → 
  max_OB_OA = 4 * sqrt 3 / 3 := 
sorry

end curve_intersection_max_OB_OA_l92_92699


namespace handshakes_at_convention_l92_92303

theorem handshakes_at_convention (n participants_per_company : ℕ) 
  (total_participants : n * participants_per_company) 
  (other_participants_company : total_participants - 1 - participants_per_company) 
  (each_handshakes : total_participants / 2 * other_participants_company / participants_per_company) : 
  total_participants = 15 ∧ participants_per_company = 5 ∧ each_handshakes = 75 := 
by
  sorry

end handshakes_at_convention_l92_92303


namespace tan_sum_identity_l92_92041

noncomputable def θ : ℝ := real.atan (4 / 2)

theorem tan_sum_identity : 
  tan (θ + π / 4) = -3 := 
by
  let tan_t := tan θ
  have tan_π_4 : tan (π / 4) = 1 := real.tan_pi_div_four
  calc
    tan (θ + π / 4) = (tan θ + tan (π / 4)) / (1 - tan θ * tan (π / 4)) : real.tan_add
                ... = (2 + 1) / (1 - 2 * 1)                          : by rw [tan_t, tan_π_4]
                ... = 3 / -1                                        : rfl
                ... = -3                                            : rfl

end tan_sum_identity_l92_92041


namespace find_least_constant_c_l92_92018

-- Definitions for the graph and functions f and g
structure Graph :=
  (vertices : Finset ℕ)
  (edges : Finset (ℕ × ℕ))

-- Assuming the existence of f(G) and g(G) as functions of the graph
def f (G : Graph) : ℕ := sorry  -- Number of triangles in graph G
def g (G : Graph) : ℕ := sorry  -- Number of tetrahedra in graph G

theorem find_least_constant_c (G : Graph) :
  g(G)^3 ≤ (3/32 : ℚ) * f(G)^4 := 
sorry

end find_least_constant_c_l92_92018


namespace find_T_l92_92182

variables (h K T : ℝ)
variables (h_val : 4 * h * 7 + 2 = 58)
variables (K_val : K = 9)

theorem find_T : T = 74 :=
by
  sorry

end find_T_l92_92182


namespace sin_90_degree_l92_92500

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92500


namespace angle_properties_l92_92806

def same_terminal_side (beta alpha : ℤ) : Prop :=
  ∃ k : ℤ, beta = k * 360 - alpha

def quadrant (alpha : ℤ) : ℕ :=
  let alpha := alpha % 360
  if 0 < alpha ∧ alpha < 90 then 1
  else if 90 < alpha ∧ alpha < 180 then 2
  else if 180 < alpha ∧ alpha < 270 then 3
  else if 270 < alpha ∧ alpha < 360 then 4
  else 0

theorem angle_properties :
  ( ∀ β, same_terminal_side β (-457) ↔ β = (λ k : ℤ, k * 360 - 457) ) ∧
  quadrant (-457) = 3 :=
  sorry

end angle_properties_l92_92806


namespace range_of_a_l92_92123

def is_on_line (M : ℝ × ℝ) (a : ℝ) : Prop :=
  M.1 + M.2 + a = 0

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem range_of_a (a : ℝ) :
  (∃ M : ℝ × ℝ, is_on_line M a ∧ distance M (2, 0) = 2 * distance M (0, 0)) →
  (real.to_nnreal (2 - 4 * real.sqrt 2) / 3 : ℝ) ≤ a ∧
  a ≤ (real.to_nnreal (2 + 4 * real.sqrt 2) / 3 : ℝ) :=
by
  sorry 

end range_of_a_l92_92123


namespace first_day_of_month_is_wednesday_l92_92221

theorem first_day_of_month_is_wednesday (d28 : Nat) (h : d28 = 28) (day28 : Day) (hday28 : day28 = Day.tuesday) : ∃ day1 : Day, day1 = Day.wednesday :=
by
  -- Assume the 28th day is a Tuesday
  have step1 : Day.tuesday = day28 := hday28
  -- By working backwards, we can deduce the other days of the week
  -- and finally arrive at the first day of the month.
  sorry

end first_day_of_month_is_wednesday_l92_92221


namespace sin_ninety_degrees_l92_92403

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92403


namespace range_of_f_l92_92950

theorem range_of_f :
  (∀ x : ℝ, -3 ≤ (cos (2 * x) + 2 * sin x) ∧ (cos (2 * x) + 2 * sin x) ≤ 3 / 2) :=
sorry

end range_of_f_l92_92950


namespace sum_even_indices_angles_l92_92818

noncomputable def z_values (k : ℕ) : set ℂ := { z | (z^(2 * k + 10)) - (z^(k + 4)) - 1 = 0 ∧ complex.abs z = 1 }

noncomputable def theta_sum (k : ℕ) (z_set : set ℂ) : ℝ :=
  let angles := set.to_finset (z_set.image (λ z, (complex.arg z * 180 / real.pi))) 
  let sorted_angles := angles.sort (≤)
  finset.sum (finset.filter (λ i, i % 2 = 1) (finset.range sorted_angles.card)) (sorted_angles (λ i, sorted_angles.val i))

theorem sum_even_indices_angles :
  theta_sum 2 (z_values 2) = 1207.5 :=
sorry

end sum_even_indices_angles_l92_92818


namespace prob_one_AB_stuck_prob_at_least_two_stuck_l92_92898

-- Define the events and their probabilities.
def prob_traffic_I := 1 / 10
def prob_no_traffic_I := 9 / 10
def prob_traffic_II := 3 / 5
def prob_no_traffic_II := 2 / 5

-- Define the events
def event_A := prob_traffic_I
def not_event_A := prob_no_traffic_I
def event_B := prob_traffic_I
def not_event_B := prob_no_traffic_I
def event_C := prob_traffic_II
def not_event_C := prob_no_traffic_II

-- Define the probabilities as required in the problem
def prob_exactly_one_of_A_B_in_traffic :=
  event_A * not_event_B + not_event_A * event_B

def prob_at_least_two_in_traffic :=
  event_A * event_B * not_event_C +
  event_A * not_event_B * event_C +
  not_event_A * event_B * event_C +
  event_A * event_B * event_C

-- Proofs (statements only)
theorem prob_one_AB_stuck :
  prob_exactly_one_of_A_B_in_traffic = 9 / 50 := sorry

theorem prob_at_least_two_stuck :
  prob_at_least_two_in_traffic = 59 / 500 := sorry

end prob_one_AB_stuck_prob_at_least_two_stuck_l92_92898


namespace find_PQ_in_right_triangle_l92_92688

theorem find_PQ_in_right_triangle (P Q R : Type*) (tan_P : ℝ) (PR : ℝ) (R_right : R) :
  PQR is_right_triangle_at R ∧ tan_P = 3 / 5 ∧ PR = 12 → 
  PQ = 14 :=
by
  -- Proof elided; provided statement aligns with the mathematical proof problem
  sorry

end find_PQ_in_right_triangle_l92_92688


namespace largest_divisor_three_consecutive_even_integers_l92_92733

theorem largest_divisor_three_consecutive_even_integers :
  (∀ n : ℕ, 0 < n → let Q := (2 * n) * (2 * n + 2) * (2 * n + 4) in 12 ∣ Q) := by
  sorry

end largest_divisor_three_consecutive_even_integers_l92_92733


namespace tan_30_degrees_correct_l92_92292

noncomputable def tan_30_degrees : ℝ := Real.tan (Real.pi / 6)

theorem tan_30_degrees_correct : tan_30_degrees = Real.sqrt 3 / 3 :=
by
  sorry

end tan_30_degrees_correct_l92_92292


namespace least_value_expression_l92_92316

theorem least_value_expression (x y : ℝ) : 
  (x^2 * y + x * y^2 - 1)^2 + (x + y)^2 ≥ 1 :=
sorry

end least_value_expression_l92_92316


namespace sin_90_eq_1_l92_92463

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92463


namespace reach_one_from_any_non_zero_l92_92110

-- Define the game rules as functions
def remove_units_digit (n : ℕ) : ℕ :=
  n / 10

def multiply_by_two (n : ℕ) : ℕ :=
  n * 2

-- Lemma: Prove that starting from 45, we can reach 1 using the game rules.
lemma reach_one_from_45 : ∃ f : ℕ → ℕ, f 45 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Lemma: Prove that starting from 345, we can reach 1 using the game rules.
lemma reach_one_from_345 : ∃ f : ℕ → ℕ, f 345 = 1 := 
by {
  -- You can define the sequence explicitly or use the function definitions.
  sorry
}

-- Theorem: Prove that any non-zero natural number can be reduced to 1 using the game rules.
theorem reach_one_from_any_non_zero (n : ℕ) (h : n ≠ 0) : ∃ f : ℕ → ℕ, f n = 1 :=
by {
  sorry
}

end reach_one_from_any_non_zero_l92_92110


namespace rectangular_prism_diagonals_l92_92360

theorem rectangular_prism_diagonals (a b c : ℕ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a) :
  let total_diagonals := 4 + 4 + 4 + 4 in
  total_diagonals = 16 :=
by
  sorry

end rectangular_prism_diagonals_l92_92360


namespace find_B_and_perpendicular_bisector_l92_92026

variable (A B C : Point)
variable (hA : A = (5,1))
variable (hAltitude : ∀ H, Line H (C,A) ⊥ Line H (B,C) ∧ Line H (C,A) = Line.mk 1 2 5)
variable (hMedian : ∀ M, Midpoint M (A,C) ∧ Line M (B,M) = Line.mk 2 1 1)

noncomputable def coordinates_of_B : Point := (3, 5)

noncomputable def equation_of_perpendicular_bisector_BC :=
  Line.mk 21 24 43

theorem find_B_and_perpendicular_bisector :
  B = coordinates_of_B ∧
  equation_of_perpendicular_bisector_BC = Line.mk 21 24 43 := by
  sorry

end find_B_and_perpendicular_bisector_l92_92026


namespace f_eq_n_plus_2_l92_92167

open Nat

-- We define the injective function f : ℕ → ℕ with the given conditions
constant f : ℕ → ℕ
axiom injective_f : Function.Injective f
axiom f1 : f 1 = 2
axiom f2 : f 2 = 4
axiom functional_eq (m n : ℕ) : f (f (m) + f (n)) = f (f (m)) + f (n)

-- The theorem we are to prove
theorem f_eq_n_plus_2 (n : ℕ) (h : n ≥ 2) : f n = n + 2 := by
  sorry

end f_eq_n_plus_2_l92_92167


namespace integer_values_count_l92_92254

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l92_92254


namespace sin_90_eq_1_l92_92466

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92466


namespace percentage_of_students_who_own_cats_l92_92119

theorem percentage_of_students_who_own_cats (total_students cats_owned : ℕ) (h_total: total_students = 500) (h_cats: cats_owned = 75) :
  (cats_owned : ℚ) / total_students * 100 = 15 :=
by
  sorry

end percentage_of_students_who_own_cats_l92_92119


namespace polar_line_equation_intersection_points_l92_92698

-- Given parametric equations of the line l
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (t, 4 - t)

-- Given polar coordinate equation of the curve C
def curve_polar (θ : ℝ) : ℝ :=
  4 * Real.cos θ

-- Function to convert cartesian coordinates to polar coordinates
def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan2 y x
  (ρ, θ)

-- Theorem: Polar coordinate equation of the line l
theorem polar_line_equation (ρ θ : ℝ) : ρ * (Real.cos θ + Real.sin θ) = 4 ↔
  ∃ t : ℝ, (ρ, θ) = cartesian_to_polar (line_parametric t).1 (line_parametric t).2 := sorry

-- Theorem: Intersection points of line l and curve C in polar coordinates
theorem intersection_points (ρ θ : ℝ) : (ρ, θ) = (4, 0) ∨ (ρ, θ) = (2 * Real.sqrt 2, Real.pi / 4) ↔
     (ρ, θ) ∈ set_of (λ p : ℝ × ℝ, ∃ t : ℝ, p = cartesian_to_polar (line_parametric t).1 (line_parametric t).2 ∧ ρ = curve_polar θ) := sorry

end polar_line_equation_intersection_points_l92_92698


namespace prove_f_f_x_eq_4_prove_f_f_x_eq_5_l92_92571

variable (f : ℝ → ℝ)

-- Conditions
axiom f_of_4 : f (-2) = 4 ∧ f 2 = 4 ∧ f 6 = 4
axiom f_of_5 : f (-4) = 5 ∧ f 4 = 5

-- Intermediate Values
axiom f_inv_of_4 : f 0 = -2 ∧ f (-1) = 2 ∧ f 3 = 6
axiom f_inv_of_5 : f 2 = 4

theorem prove_f_f_x_eq_4 :
  {x : ℝ | f (f x) = 4} = {0, -1, 3} :=
by
  sorry

theorem prove_f_f_x_eq_5 :
  {x : ℝ | f (f x) = 5} = {2} :=
by
  sorry

end prove_f_f_x_eq_4_prove_f_f_x_eq_5_l92_92571


namespace convert_to_rectangular_form_l92_92944

theorem convert_to_rectangular_form : 
  (√2 * Complex.exp (11 * Real.pi * Complex.I / 4)) = -1 + Complex.I :=
by 
  sorry

end convert_to_rectangular_form_l92_92944


namespace compute_sin_90_l92_92511

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92511


namespace seniors_at_morse_high_l92_92822

theorem seniors_at_morse_high (S : ℕ) (h1 : 0.4 * S = 0.15 * (S + 1500) - 150) : S = 300 :=
by
  sorry

end seniors_at_morse_high_l92_92822


namespace sin_90_deg_l92_92456

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92456


namespace M1_M2_product_l92_92169

theorem M1_M2_product (M_1 M_2 : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 →
  (42 * x - 51) / (x^2 - 5 * x + 6) = (M_1 / (x - 2)) + (M_2 / (x - 3))) →
  M_1 * M_2 = -2981.25 :=
by
  intros h
  sorry

end M1_M2_product_l92_92169


namespace best_play_majority_two_classes_l92_92147

theorem best_play_majority_two_classes (n : ℕ) :
  let prob_win := 1 - (1/2) ^ n
  in prob_win = 1 - (1/2) ^ n :=
by
  sorry

end best_play_majority_two_classes_l92_92147


namespace positive_real_x_condition_l92_92538

-- We define the conditions:
variables (x : ℝ)
#check (1 - x^4)
#check (1 + x^4)

-- The main proof statement:
theorem positive_real_x_condition (h1 : x > 0) 
    (h2 : (Real.sqrt (Real.sqrt (1 - x^4)) + Real.sqrt (Real.sqrt (1 + x^4)) = 1)) :
    (x^8 = 35 / 36) :=
sorry

end positive_real_x_condition_l92_92538


namespace prove_solution_set_l92_92606

-- Step 1: Define the conditions
def condition1 (x : ℝ) (a : ℝ) : Prop := a * x^2 + 3 * x - 1 > 0
def condition1_valid_set : set ℝ := { x | 1/2 < x ∧ x < 1 }
def a_value : ℝ := -2

-- Step 2: Define the main problem statement
theorem prove_solution_set (a : ℝ) : 
  (a = a_value)
  → (∀ x : ℝ, (x ∈ condition1_valid_set) → (condition1 x a))
  → ∀ x : ℝ, - (5:ℝ)/2 < x ∧ x < 1 
  ↔ a * x^2 - 3 * x + a^2 + 1 > 0 :=
by
  intros ha hcondition1 x,
  split;
  intro hx;
  sorry

end prove_solution_set_l92_92606


namespace cube_root_product_l92_92206

theorem cube_root_product (a b c : ℕ) (ha : a = 5 ^ 9) (hb : b = 7 ^ 6) (hc : c = 13 ^ 3) :
  (∛(a * b * c) : ℝ) = 79625 :=
by
  sorry

end cube_root_product_l92_92206


namespace fourth_number_is_two_eighth_number_is_two_l92_92837

theorem fourth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 4 ∧ board !! 3 = some 2 :=
by
  sorry

theorem eighth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 8 ∧ board !! 7 = some 2 :=
by
  sorry

end fourth_number_is_two_eighth_number_is_two_l92_92837


namespace triangle_properties_l92_92078

variable (a b c A B C : ℝ)
variable (CD BD : ℝ)

-- triangle properties and given conditions
variable (b_squared_eq_ac : b ^ 2 = a * c)
variable (cos_A_minus_C : Real.cos (A - C) = Real.cos B + 1 / 2)

theorem triangle_properties :
  B = π / 3 ∧ 
  A = π / 3 ∧ 
  (CD = 6 → ∃ x, x > 0 ∧ x = 4 * Real.sqrt 3 + 6) ∧
  (BD = 6 → ∀ area, area ≠ 9 / 4) :=
  by
    sorry

end triangle_properties_l92_92078


namespace sin_90_eq_1_l92_92448

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92448


namespace find_length_of_AB_l92_92235

noncomputable def line_eq (x : ℝ) : ℝ := x + 1

noncomputable def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 + 2 * y - 3

theorem find_length_of_AB :
  let A := (0 : ℝ, 1 : ℝ) in
  let B := (-2 : ℝ, -1 : ℝ) in
  dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end find_length_of_AB_l92_92235


namespace find_nat_numbers_l92_92561

-- Natural numbers not exceeding \( 10^m \)
def NatNotExceeding (m : ℕ) : set ℕ := { n | n ≤ 10^m }

-- Defining the set of allowed digits
def AllowedDigits : set ℕ := {2, 0, 1, 5}

-- Define the set of natural numbers n that satisfy the conditions
def SatisfyingNat (m : ℕ) : set ℕ :=
  { n | n ∈ NatNotExceeding m ∧ (∀ d ∈ Int.to_digits 10 n, d ∈ AllowedDigits)}

-- Main theorem to be proved
theorem find_nat_numbers (m : ℕ) (k : ℕ) (h1 : m ∈ ℕ) (h2 : k ∈ ℕ) :
  let A := if k % 3 = 0 then (4 ^ k + 2) / 3 else (4 ^ k - 1) / 3
  in ∃ (n : ℕ), n ∈ SatisfyingNat m ∧ 3 ∣ n ∧ n ≤ 10^m ∧ A = 
  find_coeff_sum k :=
sorry

end find_nat_numbers_l92_92561


namespace sin_90_eq_1_l92_92476

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92476


namespace concylicity_of_four_points_l92_92696

-- Definitions for the problem conditions and proof target

variables {A B C D E F M N P Q : Type*} -- Points in the type universe

-- Assuming geometric properties as hypotheses.

hypothesis h1 : Collinear (A, B, C) (Side points on line BC)
hypothesis h2 : ∃ D : point, On_D_Side_BC (D is on BC)
hypothesis h3 : Parallel (DE, AB) (DE parallel to AB)
hypothesis h4 : Parallel (DF, AC) (DF parallel to AC)
hypothesis h5 : Exists_E : Exists E at A intersecting DE=EF
hypothesis h6 : Exists_F : Exists F at B intersecting DF=EF
hypothesis h7 : ME : Exists (EF intersects circle) as points (M, N)
hypothesis h8 : DP_A_parallel : DP drawn parallel to AM intersecting MN at P
hypothesis h9 : AP_Meets_BC : AP meets BC at Q

theorem concylicity_of_four_points:
  (concyclic D P N Q) := 
  sorry -- proof is omitted


end concylicity_of_four_points_l92_92696


namespace value_of_m_l92_92625

theorem value_of_m (a m : ℝ) (h : a > 0) (hm : m ≠ 0) :
  (P : ℝ × ℝ) (P = (m, 3))
  (H : ∀ x : ℝ, -a * x^2 + 2 * a * x + 3 = 3 → x = 0 ∨ x = 2) :
  m = 2 :=
by
  sorry

end value_of_m_l92_92625


namespace emerie_dimes_count_l92_92878

variables (zain_coins emerie_coins num_quarters num_nickels : ℕ)
variable (emerie_dimes : ℕ)

-- Conditions as per part a)
axiom zain_has_more_coins : ∀ (e z : ℕ), z = e + 10
axiom total_zain_coins : zain_coins = 48
axiom emerie_coins_from_quarters_and_nickels : num_quarters = 6 ∧ num_nickels = 5
axiom emerie_known_coins : ∀ q n : ℕ, emerie_coins = q + n + emerie_dimes

-- The statement to prove
theorem emerie_dimes_count : emerie_coins = 38 → emerie_dimes = 27 := 
by 
  sorry

end emerie_dimes_count_l92_92878


namespace sin_90_eq_one_l92_92405

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92405


namespace gratuities_correct_l92_92917

def cost_of_striploin : ℝ := 80
def cost_of_wine : ℝ := 10
def sales_tax_rate : ℝ := 0.10
def total_bill_with_gratuities : ℝ := 140

def total_bill_before_tax : ℝ := cost_of_striploin + cost_of_wine := by sorry

def sales_tax : ℝ := sales_tax_rate * total_bill_before_tax := by sorry

def total_bill_with_tax : ℝ := total_bill_before_tax + sales_tax := by sorry

def gratuities : ℝ := total_bill_with_gratuities - total_bill_with_tax := by sorry

theorem gratuities_correct : gratuities = 41 := by sorry

end gratuities_correct_l92_92917


namespace smallest_d_bound_l92_92952

theorem smallest_d_bound :
  ∃ d > 0, ∀ (x y : ℝ) (z : ℝ), (0 ≤ x) → (0 ≤ y) → (0 < z) → 
  (sqrt (x * y * z) + d * abs (x - y) * z) ≥ ((x * z + y * z) / 2) ∧
  ∀ d', (d' > 0 ∧ (∀ (x y : ℝ) (z : ℝ), (0 ≤ x) → (0 ≤ y) → (0 < z) →
  (sqrt (x * y * z) + d' * abs (x - y) * z) ≥ ((x * z + y * z) / 2))) → (d' ≥ d) :=
begin
  sorry
end

end smallest_d_bound_l92_92952


namespace max_xy_l92_92177

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x + 8 * y = 112) : xy ≤ 56 :=
sorry

end max_xy_l92_92177


namespace union_P_complement_Q_l92_92641

-- Define sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the complement of Q in ℝ
def C_RQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the main theorem
theorem union_P_complement_Q : (P ∪ C_RQ) = {x | -2 < x ∧ x ≤ 3} := 
by
  sorry

end union_P_complement_Q_l92_92641


namespace best_play_wins_majority_l92_92130

/-- Probability that the best play wins with a majority of the votes given the conditions -/
theorem best_play_wins_majority (n : ℕ) :
  let p := 1 - (1 / 2)^n
  in p > (1 - (1 / 2)^n) ∧ p ≤ 1 :=
sorry

end best_play_wins_majority_l92_92130


namespace period_of_complex_tone_l92_92535

-- Define the function for the complex tone
def complex_tone (x : ℝ) : ℝ := (1/4) * sin (4 * x) + (1/6) * sin (6 * x)

-- Conditions: Define the periods of the individual sine functions
def period_y1 := π / 2
def period_y2 := π / 3

-- Prove the period of the complex tone
theorem period_of_complex_tone : ∃ T, (∀ x : ℝ, complex_tone (x + T) = complex_tone x) ∧ T = π := 
sorry

end period_of_complex_tone_l92_92535


namespace sin_90_eq_one_l92_92409

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92409


namespace find_value_of_sum_of_squares_l92_92592

theorem find_value_of_sum_of_squares (x y : ℝ) (h : x^2 + y^2 + x^2 * y^2 - 4 * x * y + 1 = 0) :
  (x + y)^2 = 4 :=
sorry

end find_value_of_sum_of_squares_l92_92592


namespace one_third_of_5_4_is_9_over_5_l92_92965

-- Define the initial value
def initial_value : ℝ := 5.4

-- Define the correct answer as a fraction
def correct_answer : ℚ := 9 / 5

-- State the theorem with required proof
theorem one_third_of_5_4_is_9_over_5 :
  (initial_value / 3 : ℚ) = correct_answer :=
by sorry

end one_third_of_5_4_is_9_over_5_l92_92965


namespace first_group_work_done_l92_92340

-- Define work amounts with the conditions given
variable (W : ℕ) -- amount of work 3 people can do in 3 days
variable (work_rate : ℕ → ℕ → ℕ) -- work_rate(p, d) is work done by p people in d days

-- Conditions
axiom cond1 : work_rate 3 3 = W
axiom cond2 : work_rate 6 3 = 6 * W

-- The proof statement
theorem first_group_work_done : work_rate 3 3 = 2 * W :=
by
  sorry

end first_group_work_done_l92_92340


namespace pier_influence_duration_l92_92290

noncomputable def distance_affected_by_typhoon (AB AC: ℝ) : ℝ :=
  let AD := 350
  let DC := (AD ^ 2 - AC ^ 2).sqrt
  2 * DC

noncomputable def duration_under_influence (distance speed: ℝ) : ℝ :=
  distance / speed

theorem pier_influence_duration :
  let AB := 400
  let AC := AB * (1 / 2)
  let speed := 40
  duration_under_influence (distance_affected_by_typhoon AB AC) speed = 2.5 :=
by
  -- Proof would go here, but since it's omitted
  sorry

end pier_influence_duration_l92_92290


namespace probability_all_same_room_probability_at_least_two_same_room_l92_92306

/-- 
  Given that there are three people and each person is assigned to one of four rooms with equal probability,
  let P1 be the probability that all three people are assigned to the same room,
  and let P2 be the probability that at least two people are assigned to the same room.
  We need to prove:
  1. P1 = 1 / 16
  2. P2 = 5 / 8
-/
noncomputable def P1 : ℚ := sorry

noncomputable def P2 : ℚ := sorry

theorem probability_all_same_room :
  P1 = 1 / 16 :=
sorry

theorem probability_at_least_two_same_room :
  P2 = 5 / 8 :=
sorry

end probability_all_same_room_probability_at_least_two_same_room_l92_92306


namespace distance_between_foci_of_ellipse_l92_92970

theorem distance_between_foci_of_ellipse (x y : ℝ) :
  9 * x^2 + y^2 = 36 → 2 * real.sqrt (36 - 4) = 8 * real.sqrt 2 :=
by
  intro h
  calc
    2 * real.sqrt (36 - 4) = 2 * real.sqrt (32) : sorry
    ...                   = 2 * 4 * real.sqrt 2  : sorry
    ...                   = 8 * real.sqrt 2      : sorry

end distance_between_foci_of_ellipse_l92_92970


namespace triangle_B_side_length_range_l92_92708

-- Define the problem and conditions
def triangle_has_exactly_one_solution (a b A : ℝ) : Prop :=
  let sin_A := Real.sin A in
  (a = b * sin_A) ∨ (a > b * sin_A ∧ a ≥ b)

def triangle_B_side_length_valid (b : ℝ) : Prop :=
  0 < b ∧ b ≤ Real.sqrt 3 ∨ b = 2

-- Given the conditions
def triangle_ABC_satisfies_conditions : Prop :=
  let a : ℝ := Real.sqrt 3 in
  let A : ℝ := Real.pi / 3 in -- 60 degrees in radians
  triangle_has_exactly_one_solution a b A

-- The problem statement
theorem triangle_B_side_length_range (b : ℝ) :
  triangle_ABC_satisfies_conditions -> triangle_B_side_length_valid b :=
sorry -- Proof omitted

end triangle_B_side_length_range_l92_92708


namespace solution_set_l92_92008

theorem solution_set (x : ℝ) (h : x ≠ 5) :
  (∃ y, y = (x * (x + 3) / (x - 5) ^ 2) ∧ y ≥ 15) ↔
  x ≤ 52 / 14 ∨ x ≥ 101 / 14 :=
begin
  sorry
end

end solution_set_l92_92008


namespace num_of_integers_satisfying_sqrt_condition_l92_92274

theorem num_of_integers_satisfying_sqrt_condition : 
  let S := { x : ℤ | 5 < Real.sqrt x ∧ x < 36 }
  in (S.card = 10) :=
begin
  let S := { x : ℤ | 25 < x ∧ x < 36 },
  sorry
end

end num_of_integers_satisfying_sqrt_condition_l92_92274


namespace milk_needed_for_one_batch_l92_92791

-- Define cost of one batch given amount of milk M
def cost_of_one_batch (M : ℝ) : ℝ := 1.5 * M + 6

-- Define cost of three batches
def cost_of_three_batches (M : ℝ) : ℝ := 3 * cost_of_one_batch M

theorem milk_needed_for_one_batch : ∃ M : ℝ, cost_of_three_batches M = 63 ∧ M = 10 :=
by
  sorry

end milk_needed_for_one_batch_l92_92791


namespace no_solution_function_l92_92540

theorem no_solution_function : ¬∃ (f : ℕ → ℕ), ∀ x : ℕ, f(f(x)) = x + 1 := by
  sorry

end no_solution_function_l92_92540


namespace angie_total_taxes_l92_92375

theorem angie_total_taxes:
  ∀ (salary : ℕ) (N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over : ℕ),
  salary = 80 →
  N_1 = 12 → T_1 = 8 → U_1 = 5 →
  N_2 = 15 → T_2 = 6 → U_2 = 7 →
  N_3 = 10 → T_3 = 9 → U_3 = 6 →
  N_4 = 14 → T_4 = 7 → U_4 = 4 →
  left_over = 18 →
  T_1 + T_2 + T_3 + T_4 = 30 :=
by
  intros salary N_1 N_2 N_3 N_4 T_1 T_2 T_3 T_4 U_1 U_2 U_3 U_4 left_over
  sorry

end angie_total_taxes_l92_92375


namespace digit_problem_l92_92525

theorem digit_problem (A B C D E F : ℕ) (hABC : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ D ≠ E ∧ D ≠ F ∧ E ≠ F) 
    (h1 : 100 * A + 10 * B + C = D * 100000 + A * 10000 + E * 1000 + C * 100 + F * 10 + B)
    (h2 : 100 * C + 10 * B + A = E * 100000 + D * 10000 + C * 1000 + A * 100 + B * 10 + F) : 
    A = 3 ∧ B = 6 ∧ C = 4 ∧ D = 1 ∧ E = 2 ∧ F = 9 := 
sorry

end digit_problem_l92_92525


namespace discount_difference_l92_92929

theorem discount_difference (initial_amount : ℝ) (single_discount : ℝ) (first_successive_discount : ℝ) (second_successive_discount : ℝ) :
  initial_amount = 15000 →
  single_discount = 0.3 →
  first_successive_discount = 0.25 →
  second_successive_discount = 0.05 →
  let single_discount_amount := initial_amount * (1 - single_discount)
  let successive_discount_amount := 
    let amount_after_first_discount := initial_amount * (1 - first_successive_discount)
    in amount_after_first_discount * (1 - second_successive_discount)
  in (successive_discount_amount - single_discount_amount) = 187.5 := 
begin
  intros h1 h2 h3 h4,
  let single_discount_amount := initial_amount * (1 - single_discount),
  let amount_after_first_discount := initial_amount * (1 - first_successive_discount),
  let successive_discount_amount := amount_after_first_discount * (1 - second_successive_discount),
  sorry
end

end discount_difference_l92_92929


namespace monomial_sum_mn_l92_92098

-- Define the conditions as Lean definitions
def is_monomial_sum (x y : ℕ) (m n : ℕ) : Prop :=
  ∃ k : ℕ, (x ^ 2) * (y ^ m) + (x ^ n) * (y ^ 3) = x ^ k

-- State our main theorem
theorem monomial_sum_mn (x y : ℕ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
sorry  -- Completion of the proof is not required

end monomial_sum_mn_l92_92098


namespace inscribed_square_length_l92_92923

noncomputable theory

-- Define the pentagon with given properties
structure Pentagon :=
(side_length : ℝ)

-- Define points and angles
structure Points (p : Pentagon) (abcd : Fin 5 → ℝ × ℝ) :=
(F : ℝ × ℝ)
(G : ℝ × ℝ)
(H : ℝ × ℝ)
(angleDFG : ℝ)
(angleDFH : ℝ)

-- Define the regular pentagon
def ABCDE : Pentagon := ⟨1⟩

-- Define criteria for points F, G, and H
def conditions : Points ABCDE (fun i => (1, 1)) :=
{ F := (0.5, 0),
  G := (0.75, 0.65),
  H := (0.75, -0.65),
  angleDFG := 30,
  angleDFH := 30 }

-- Theorem statement
theorem inscribed_square_length (P : Points ABCDE (fun i => (1, 1))) 
  (h_eq1 : ∃ (eq1: ℝ → ℝ → Prop), eq1 (P.angleDFG) 30)
  (h_eq2 : ∃ (eq2: ℝ → ℝ → Prop), eq2 (P.angleDFH) 30) 
  : ∃ (FGH_eq : Prop), FGH_eq :=
begin
  -- Proof is omitted
  sorry
end

-- Example application
example : ∃ (res: ℝ), res = sorry := 
begin
  apply inscribed_square_length,
  existsi (λ x y, x = y),
  repeat { existsi 30 },
  existsi (λ _ _, true),
  exact trivial
end

end inscribed_square_length_l92_92923


namespace best_play_wins_majority_l92_92138

variables (n : ℕ)

-- Conditions
def students_in_play_A : ℕ := n
def students_in_play_B : ℕ := n
def mothers : ℕ := 2 * n

-- Question
theorem best_play_wins_majority : 
  (probability_fin_votes_wins_majority (students_in_play_A n) (students_in_play_B n) (mothers n)) = 1 - (1/2)^n :=
sorry

end best_play_wins_majority_l92_92138


namespace sin_90_deg_l92_92454

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92454


namespace value_of_m_l92_92624

theorem value_of_m (a m : ℝ) (h : a > 0) (hm : m ≠ 0) :
  (P : ℝ × ℝ) (P = (m, 3))
  (H : ∀ x : ℝ, -a * x^2 + 2 * a * x + 3 = 3 → x = 0 ∨ x = 2) :
  m = 2 :=
by
  sorry

end value_of_m_l92_92624


namespace pinecones_left_l92_92297

theorem pinecones_left 
  (total_pinecones : ℕ)
  (pct_eaten_by_reindeer pct_collected_for_fires : ℝ)
  (total_eaten_by_reindeer : ℕ := (pct_eaten_by_reindeer * total_pinecones).to_nat)
  (total_eaten_by_squirrels : ℕ := 2 * total_eaten_by_reindeer)
  (total_eaten : ℕ := total_eaten_by_reindeer + total_eaten_by_squirrels)
  (pinecones_after_eating : ℕ := total_pinecones - total_eaten)
  (total_collected_for_fires : ℕ := (pct_collected_for_fires * pinecones_after_eating).to_nat)
  (remaining_pinecones : ℕ := pinecones_after_eating - total_collected_for_fires)
  (2000_pinecones : total_pinecones = 2000)
  (20_percent_eaten_by_reindeer : pct_eaten_by_reindeer = 0.20)
  (25_percent_collected_for_fires : pct_collected_for_fires = 0.25) : 
  remaining_pinecones = 600 := 
by
  sorry

end pinecones_left_l92_92297


namespace intercept_sum_l92_92880

theorem intercept_sum (x0 y0 : ℕ) (h1 : x0 < 17) (h2 : y0 < 17)
  (hx : 7 * x0 ≡ 2 [MOD 17]) (hy : 3 * y0 ≡ 15 [MOD 17]) : x0 + y0 = 17 :=
sorry

end intercept_sum_l92_92880


namespace grid_sum_max_min_l92_92676

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def three_by_three_divisible (g : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i + 2 < 4 ∧ 0 ≤ j ∧ j + 2 < 4 →
  is_divisible_by_4 (g i j + g (i+1) j + g (i+2) j +
                     g i (j+1) + g (i+1) (j+1) + g (i+2) (j+1) +
                     g i (j+2) + g (i+1) (j+2) + g (i+2) (j+2))

def sum_not_divisible_by_4 (s : ℕ) : Prop := ¬ is_divisible_by_4 s

def is_valid_grid (g : ℕ → ℕ → ℕ) : Prop :=
  (∀ i j, g i j = 1 ∨ g i j = 2) ∧
  three_by_three_divisible g

theorem grid_sum_max_min :
  ∃ g : ℕ → ℕ → ℕ, is_valid_grid g ∧ 
  (let s := (finset.univ.product finset.univ).sum (λ (p : ℕ × ℕ), g p.1 p.2)
  in sum_not_divisible_by_4 s ∧ s = 30) ∧
  (∃ h : ℕ → ℕ → ℕ, is_valid_grid h ∧ 
  let m := (finset.univ.product finset.univ).sum (λ (p : ℕ × ℕ), h p.1 p.2)
  in sum_not_divisible_by_4 m ∧ m = 19) :=
sorry

end grid_sum_max_min_l92_92676


namespace g_10_equals_100_l92_92521

-- Define the function g and the conditions it must satisfy.
def g : ℕ → ℝ := sorry

axiom g_2 : g 2 = 4

axiom g_condition : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = (g (2 * m) + g (2 * n)) / 2

-- Prove the required statement.
theorem g_10_equals_100 : g 10 = 100 :=
by sorry

end g_10_equals_100_l92_92521


namespace total_students_l92_92347

theorem total_students (a : ℕ) (h1: (71 * ((3480 - 69 * a) / 2) + 69 * (a - (3480 - 69 * a) / 2)) = 3480) : a = 50 :=
by
  -- Proof to be provided here
  sorry

end total_students_l92_92347


namespace sin_ninety_degrees_l92_92395

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92395


namespace sum_of_possible_two_digit_values_l92_92172

theorem sum_of_possible_two_digit_values (d : ℕ) (h1 : 0 < d) (h2 : d < 100) (h3 : 137 % d = 6) : d = 131 :=
by
  sorry

end sum_of_possible_two_digit_values_l92_92172


namespace granola_bars_split_l92_92648

theorem granola_bars_split : 
  ∀ (initial_bars : ℕ) (days_of_week : ℕ) (traded_bars : ℕ) (sisters : ℕ),
  initial_bars = 20 →
  days_of_week = 7 →
  traded_bars = 3 →
  sisters = 2 →
  (initial_bars - days_of_week - traded_bars) / sisters = 5 :=
by
  intros initial_bars days_of_week traded_bars sisters
  intros h_initial h_days h_traded h_sisters
  rw [h_initial, h_days, h_traded, h_sisters]
  norm_num
  sorry

end granola_bars_split_l92_92648


namespace water_flow_per_minute_l92_92325

def water_flow_rate_kmph : ℕ := 6
def river_depth_m : ℕ := 2
def river_width_m : ℕ := 45

theorem water_flow_per_minute :
  (6 * 1000 / 60) * river_depth_m * river_width_m = 9000 :=
by { dsimp [water_flow_rate_kmph, river_depth_m, river_width_m], norm_num, }

#print axioms water_flow_per_minute

end water_flow_per_minute_l92_92325


namespace common_tangents_of_separate_circles_l92_92883

-- Define the radii of the two circles
def radius1 : ℝ := 3
def radius2 : ℝ := 5

-- Define the distance between the centers of the two circles
def centerDistance : ℝ := 10

-- The theorem stating the number of common tangents
theorem common_tangents_of_separate_circles :
  radius1 + radius2 < centerDistance → number_of_common_tangents (radius1, radius2, centerDistance) = 4 :=
by
  sorry

end common_tangents_of_separate_circles_l92_92883


namespace num_of_integers_satisfying_sqrt_condition_l92_92276

theorem num_of_integers_satisfying_sqrt_condition : 
  let S := { x : ℤ | 5 < Real.sqrt x ∧ x < 36 }
  in (S.card = 10) :=
begin
  let S := { x : ℤ | 25 < x ∧ x < 36 },
  sorry
end

end num_of_integers_satisfying_sqrt_condition_l92_92276


namespace probability_of_same_color_is_correct_l92_92342

-- Definitions from the problem conditions
def red_marbles := 6
def white_marbles := 7
def blue_marbles := 8
def total_marbles := red_marbles + white_marbles + blue_marbles -- 21

-- Calculate the probability of drawing 4 red marbles
def P_all_red := (red_marbles / total_marbles) * ((red_marbles - 1) / (total_marbles - 1)) * ((red_marbles - 2) / (total_marbles - 2)) * ((red_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 white marbles
def P_all_white := (white_marbles / total_marbles) * ((white_marbles - 1) / (total_marbles - 1)) * ((white_marbles - 2) / (total_marbles - 2)) * ((white_marbles - 3) / (total_marbles - 3))

-- Calculate the probability of drawing 4 blue marbles
def P_all_blue := (blue_marbles / total_marbles) * ((blue_marbles - 1) / (total_marbles - 1)) * ((blue_marbles - 2) / (total_marbles - 2)) * ((blue_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles of the same color
def P_all_same_color := P_all_red + P_all_white + P_all_blue

-- Proof that the total probability is equal to the given correct answer
theorem probability_of_same_color_is_correct : P_all_same_color = 240 / 11970 := by
  sorry

end probability_of_same_color_is_correct_l92_92342


namespace therapy_charge_3_hours_l92_92346

variables (A F : ℕ)
variable h1 : F = A + 20
variable h2 : F + 4 * A = 300

theorem therapy_charge_3_hours : F + 2 * A = 188 := 
by
  sorry

end therapy_charge_3_hours_l92_92346


namespace alpha_perpendicular_beta_l92_92660

variable {Line : Type} {Plane : Type}
variable (m n : Line) (α β γ : Plane)
variable [lines_are_distinct : m ≠ n]
variable [planes_are_distinct : α ≠ β ∧ β ≠ γ ∧ γ ≠ α]

-- Defining geometric properties as predicates
@[class] def is_perpendicular (m : Line) (α : Plane) : Prop := sorry
@[class] def is_parallel (m : Line) (α : Plane) : Prop := sorry

-- Hypotheses for the correct answer
axiom perpendicular_m_beta : is_perpendicular m β
axiom parallel_m_alpha : is_parallel m α

theorem alpha_perpendicular_beta :
  is_perpendicular α β := by
  sorry

end alpha_perpendicular_beta_l92_92660


namespace annulus_area_is_8pi_l92_92747

noncomputable def annulusArea (z : ℂ) (r1 r2 : ℝ) : ℝ :=
  if 1 ≤ |z| ∧ |z| ≤ 3 then π * (r2 ^ 2 - r1 ^ 2) else 0

theorem annulus_area_is_8pi (z : ℂ) (h : 1 ≤ |z| ∧ |z| ≤ 3) : annulusArea z 1 3 = 8 * π := by
  sorry

lemma complex_number_area (h : ∀ z : ℂ, 1 ≤ |z| ∧ |z| ≤ 3) : annulusArea 0 1 3 = 8 * π := by
  apply annulus_area_is_8pi
  exact h 0
  sorry

end annulus_area_is_8pi_l92_92747


namespace milk_distribution_l92_92705

theorem milk_distribution 
  (x y z : ℕ)
  (h_total : x + y + z = 780)
  (h_equiv : 3 * x / 4 = 4 * y / 5 ∧ 3 * x / 4 = 4 * z / 7) :
  x = 240 ∧ y = 225 ∧ z = 315 := 
sorry

end milk_distribution_l92_92705


namespace find_real_values_for_inequality_l92_92006

theorem find_real_values_for_inequality {x : ℝ} : 
  (x ≠ 5) → ( (x ∈ [3, 5) ∪ (Set.Icc (5 : ℝ) (125 / 14))) ↔ (x * (x + 3) / (x - 5) ^ 2 ≥ 15)) :=
begin
  sorry
end

end find_real_values_for_inequality_l92_92006


namespace sin_90_deg_l92_92452

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92452


namespace sin_90_eq_1_l92_92438

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92438


namespace integer_solutions_count_l92_92271

theorem integer_solutions_count :
  (finset.filter (λ (x : ℤ), 5 < real.sqrt (x : ℝ) ∧ real.sqrt (x : ℝ) < 6) 
  (finset.Icc 26 35)).card = 10 :=
by
  sorry

end integer_solutions_count_l92_92271


namespace sin_ninety_degrees_l92_92396

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92396


namespace find_m_l92_92626

theorem find_m (a m : ℝ) (h_pos : a > 0) (h_points : (m, 3) ∈ set_of (λ x : ℝ × ℝ, ∃ x_val : ℝ, x.snd = -a * (x_val)^2 + 2 * a * x_val + 3)) (h_non_zero : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92626


namespace sin_90_eq_1_l92_92472

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92472


namespace seq_a5_equals_31_l92_92575

def seq (n : ℕ) : ℕ := 
  match n with
  | 1 => 1
  | n+1 => 2 * seq n + 1

theorem seq_a5_equals_31 : seq 5 = 31 :=
sorry

end seq_a5_equals_31_l92_92575


namespace cos_A_value_triangle_area_l92_92586

noncomputable def triangle_inscribed_in_unit_circle (a b c A B C : ℝ) (u v w : ℝ) : Prop :=
a = u ∧ b = v ∧ c = w ∧ A + B + C = π ∧ u + v + w = 1 ∧ v^2 + w^2 = 4 
∧ 2 * u * real.cos A = w * real.cos B + v * real.cos C

theorem cos_A_value (a b c A B C : ℝ) (h1 : triangle_inscribed_in_unit_circle a b c A B C a) :
  real.cos A = 1 / 2 :=
sorry

theorem triangle_area (a b c A B C : ℝ) (h2 : triangle_inscribed_in_unit_circle a b c A B C a) :
  let bc := b * c in real.sin A = √3 / 2 → bc / 2 * real.sin A = √3 / 4 :=
sorry

end cos_A_value_triangle_area_l92_92586


namespace sine_of_smaller_angle_and_k_domain_l92_92796

theorem sine_of_smaller_angle_and_k_domain (α : ℝ) (k : ℝ) (AD : ℝ) (h0 : 1 < k) 
  (h1 : CD = AD * Real.tan (2 * α)) (h2 : BD = AD * Real.tan α) 
  (h3 : k = CD / BD) :
  k > 2 ∧ Real.sin (Real.pi / 2 - 2 * α) = 1 / (k - 1) := by
  sorry

end sine_of_smaller_angle_and_k_domain_l92_92796


namespace sin_90_eq_one_l92_92412

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92412


namespace range_of_f_l92_92244

noncomputable def f (x : ℝ) : ℝ := (2^x - 3) / (2^x + 1)

theorem range_of_f : set.Ioo (-3 : ℝ) 1 = set.range (f) :=
by { sorry }

end range_of_f_l92_92244


namespace Jessica_age_when_Justin_was_born_l92_92163

-- Definitions based on given conditions
variable (Justin_curr_age : ℕ) (years_after_which_James_will_be_44 : ℕ)
variable (James_diff_Jessica_age : ℕ) (James_future_age : ℕ)

-- Assigning the specific conditions provided in the problem
def justin_age := 26
def years_after := 5
def james_jessica_diff := 7
def james_future := 44

-- Theorem statement
theorem Jessica_age_when_Justin_was_born : 
  James_future = james_future →
  years_after_which_James_will_be_44 = years_after →
  James_diff_Jessica_age = james_jessica_diff →
  Justin_curr_age = justin_age →
  let James_curr_age := James_future - years_after_which_James_will_be_44 in
  let Jessica_curr_age := James_curr_age - James_diff_Jessica_age in
  let Jessica_age_when_Justin_was_born := Jessica_curr_age - Justin_curr_age in
  Jessica_age_when_Justin_was_born = 6 :=
by
  sorry

end Jessica_age_when_Justin_was_born_l92_92163


namespace best_play_wins_majority_l92_92142

variables (n : ℕ)

-- Conditions
def students_in_play_A : ℕ := n
def students_in_play_B : ℕ := n
def mothers : ℕ := 2 * n

-- Question
theorem best_play_wins_majority : 
  (probability_fin_votes_wins_majority (students_in_play_A n) (students_in_play_B n) (mothers n)) = 1 - (1/2)^n :=
sorry

end best_play_wins_majority_l92_92142


namespace car_mpg_20_l92_92763

noncomputable def miles_per_gallon (total_miles : ℕ) (total_gallons : ℕ) : ℝ :=
total_miles / total_gallons

theorem car_mpg_20 :
  (total_miles total_gallons : ℕ) (h1 : total_miles = 100) (h2 : total_gallons = 5) :
  miles_per_gallon total_miles total_gallons = 20 :=
by
  unfold miles_per_gallon
  rw [h1, h2]
  norm_num
  sorry

end car_mpg_20_l92_92763


namespace a_geq_1_of_inequality_l92_92074

open Real

def f (a : ℝ) (x : ℝ) := a / x + x * log x
def g (x : ℝ) := x^3 - x^2 - 5

theorem a_geq_1_of_inequality
  (a : ℝ)
  (h : ∀ x1 x2, x1 ∈ Icc (1/2 : ℝ) 2 → x2 ∈ Icc (1/2 : ℝ) 2 → f a x1 - g x2 ≥ 2) :
  a ≥ 1 :=
begin
  sorry
end

end a_geq_1_of_inequality_l92_92074


namespace best_play_majority_win_probability_l92_92134

theorem best_play_majority_win_probability (n : ℕ) :
  (1 - (1 / 2) ^ n) = probability_best_play_wins_majority n :=
sorry

end best_play_majority_win_probability_l92_92134


namespace find_2f_l92_92088

variable {x : ℝ}
variable (f : ℝ → ℝ)
variable (h : ∀ x > 0, f(x^2) = 2 / (2 + x^2))

theorem find_2f (hx : x > 0) : 2 * f(x) = 4 / (2 + x^2) :=
sorry

end find_2f_l92_92088


namespace exists_diag_matrix_mul_eq_three_times_l92_92013

theorem exists_diag_matrix_mul_eq_three_times (w : ℝ × ℝ × ℝ) :
  ∃ N : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ, N w = (3 * w.1, 3 * w.2, 3 * w.3) :=
by
  let N : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := λ w, (3 * w.1, 3 * w.2, 3 * w.3)
  use N
  intro w
  simp
  apply Prod.ext
  simp
  simp
  sorry

end exists_diag_matrix_mul_eq_three_times_l92_92013


namespace sin_ninety_degrees_l92_92399

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92399


namespace sin_90_degrees_l92_92484

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92484


namespace integer_solutions_count_l92_92267

theorem integer_solutions_count :
  (finset.filter (λ (x : ℤ), 5 < real.sqrt (x : ℝ) ∧ real.sqrt (x : ℝ) < 6) 
  (finset.Icc 26 35)).card = 10 :=
by
  sorry

end integer_solutions_count_l92_92267


namespace employees_at_picnic_l92_92677

def percentage_of_employees_at_picnic (total_employees men_attend_ratio women_attend_ratio men_ratio : ℚ) : ℚ :=
  let total_men := total_employees * men_ratio in
  let total_women := total_employees - total_men in
  let men_at_picnic := total_men * men_attend_ratio in
  let women_at_picnic := total_women * women_attend_ratio in
  (men_at_picnic + women_at_picnic) / total_employees * 100

theorem employees_at_picnic
  (total_employees : ℚ)
  (h1 : total_employees = 100)
  (men_attend_ratio : ℚ)
  (h2 : men_attend_ratio = 0.20)
  (women_attend_ratio : ℚ)
  (h3 : women_attend_ratio = 0.40)
  (men_ratio : ℚ)
  (h4 : men_ratio = 0.50) :
  percentage_of_employees_at_picnic total_employees men_attend_ratio women_attend_ratio men_ratio = 30 :=
  by {
    have h5 : percentage_of_employees_at_picnic 100 0.20 0.40 0.50 = 30, sorry,
    exact h5
  }

end employees_at_picnic_l92_92677


namespace integer_solutions_count_l92_92272

theorem integer_solutions_count :
  (finset.filter (λ (x : ℤ), 5 < real.sqrt (x : ℝ) ∧ real.sqrt (x : ℝ) < 6) 
  (finset.Icc 26 35)).card = 10 :=
by
  sorry

end integer_solutions_count_l92_92272


namespace area_of_gray_region_l92_92153

theorem area_of_gray_region (r : ℝ) (h : r > 0) :
  (∫ θ in 0..2*π, ∫ ρ in 0..r, ρ) / (π * r^2) = 1 / 2 :=
by sorry

end area_of_gray_region_l92_92153


namespace sin_90_eq_1_l92_92469

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92469


namespace sin_90_degrees_l92_92487

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92487


namespace problem_1_problem_2_problem_3_problem_4i_problem_4ii_problem_4iii_problem_4iv_l92_92337

-- Problem 1:
theorem problem_1 (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n > 1, a n = 1 + 1 / a (n - 1)) :
  a 3 = 3 / 2 := sorry

-- Problem 2:
theorem problem_2 (a q : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : q 1 = 1 / 2) (h₁ : ∀ n > 1, q n = q 1)
  (h₂ : ∀ n, S n = (a 1 * (1 - q 1 ^ n)) / (1 - q 1)) :
  (S 4 / a 4) = 15 := sorry

-- Problem 3:
theorem problem_3 (a b c S : ℝ) (h₀ : 2 * S = (a + b) ^ 2 - c ^ 2) :
  Real.tan (Real.atan2 a b c S) = -4 / 3 := sorry

-- Problem 4i:
theorem problem_4i (a : ℕ → ℝ) (h₀ : a 1 = 2) (h₁ : ∀ n ∈ ℕ*, a (n + 1) = 2 * a n - 1) :
  a 11 = 1025 := sorry

-- Problem 4ii:
theorem problem_4ii (a b : ℕ → ℝ) (h₀ : ∀ n ∈ ℕ*, a (n + 1) = 1 - 1 / (4 * a n))
  (h₁ : b n = 2 / (2 * a n - 1)) :
  ¬ (∀ n ∈ ℕ*, b (n + 2) - b (n + 1) = b (n + 1) - b n) := sorry

-- Problem 4iii:
theorem problem_4iii (a : ℕ → ℝ) (S : ℕ → ℝ) (h₀ : S n = n ^ 2 + 2 * n) :
  (∀ n ∈ ℕ*, 1 / a (n + 1) + 1 / a (n + 2) + ... + 1 / a (2 * n) ≥ 1 / 5) := sorry

-- Problem 4iv:
theorem problem_4iv (a : ℤ → ℕ → ℝ) (h₀ : (∀ n ∈ ℕ*, a (1 + 3 * n + 5 * n (n + 1) / 2 + ... + (2 * n - 1) * a n) = 2 ^ (n + 1)) :
  ¬ (∀ n ∈ ℕ*, a n = 2 ^ n / (2 * n - 1)) := sorry

end problem_1_problem_2_problem_3_problem_4i_problem_4ii_problem_4iii_problem_4iv_l92_92337


namespace solution_set_l92_92669

variables {f : ℝ → ℝ}

-- Given conditions
def f_domain : Set ℝ := Set.univ
def f_derivative (x : ℝ) : Prop := (deriv f x) > 2
def f_at_neg_one : f (-1) = 2

-- The proof problem
theorem solution_set (h1 : ∀ x, x ∈ f_domain) (h2 : ∀ x, f_derivative x)
  (h3 : f_at_neg_one) : {x : ℝ | f x > 2 * x + 4} = Set.Ioi (-1) :=
sorry

end solution_set_l92_92669


namespace value_of_m_l92_92623

theorem value_of_m (a m : ℝ) (h : a > 0) (hm : m ≠ 0) :
  (P : ℝ × ℝ) (P = (m, 3))
  (H : ∀ x : ℝ, -a * x^2 + 2 * a * x + 3 = 3 → x = 0 ∨ x = 2) :
  m = 2 :=
by
  sorry

end value_of_m_l92_92623


namespace int_values_satisfy_condition_l92_92256

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l92_92256


namespace vertex_of_parabola_l92_92294

-- Definition of the parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- The theorem stating the vertex of the parabola
theorem vertex_of_parabola : ∃ h k : ℝ, (h, k) = (2, -5) :=
by
  sorry

end vertex_of_parabola_l92_92294


namespace circumcenter_on_omega_l92_92889

-- Definitions for the given geometry setup
variable (ω : Circle)
variable (B C P A Q : Point)
variable (BP_eq_AC : BP = AC)
variable (CQ_eq_AB : CQ = AB)
variable (circum_eq : Circle.mk A P Q)

theorem circumcenter_on_omega
  (ω = Circle.pass_through B C)
  (P_on_AB : P ∈ Segment A B)
  (P_on_AC : P ∈ Segment A C)
  (Q_on_CK : Q ∈ Ray C)
  (BP_eq_AC : distance B P = distance A C)
  (CQ_eq_AB : distance C Q = distance A B) :
  circumcenter A P Q ∈ ω :=
sorry

end circumcenter_on_omega_l92_92889


namespace smallest_composite_no_prime_factors_less_than_20_l92_92979

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92979


namespace parallelogram_area_l92_92730

open Real EuclideanSpace

def vector3 := EuclideanSpace ℝ (fin 3)

noncomputable def a : vector3 := ![4, -6, 3]
noncomputable def b : vector3 := ![6, -10, 6]
noncomputable def c : vector3 := ![5, -1, 1]
noncomputable def d : vector3 := ![7, -5, 4]

noncomputable def ab := b - a
noncomputable def cd := d - c
noncomputable def ac := c - a

theorem parallelogram_area :
  ab = cd →
  ‖ab × ac‖ = 7 * sqrt 6 :=
by
  sorry

end parallelogram_area_l92_92730


namespace determine_m_l92_92633

theorem determine_m (a m : ℝ) (h : a > 0) (h2 : (m, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = -a * p.1 ^ 2 + 2 * a * p.1 + 3)) (h3 : m ≠ 0) : m = 2 :=
sorry

end determine_m_l92_92633


namespace cos_triple_angle_l92_92184

theorem cos_triple_angle (x θ : ℝ) (h : x = Real.cos θ) : Real.cos (3 * θ) = 4 * x^3 - 3 * x :=
by
  sorry

end cos_triple_angle_l92_92184


namespace sin_90_eq_one_l92_92419

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92419


namespace wrestling_match_student_count_l92_92809

theorem wrestling_match_student_count (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 := by
  sorry

end wrestling_match_student_count_l92_92809


namespace sin_90_eq_1_l92_92473

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92473


namespace i_pow_2023_l92_92861

-- Define i and the conditions
def i : ℂ := complex.I

theorem i_pow_2023 : i ^ 2023 = -i :=
by
  have h1 : i ^ 1 = i := by rw [pow_one]
  have h2 : i ^ 2 = -1 := by rw [complex.I_sq, pow_two]
  have h3 : i ^ 3 = -i := by rw [pow_succ, h2, mul_neg, one_mul, neg_one_mul i]
  have h4 : i ^ 4 = 1 := by rw [pow_succ, h3, mul_neg, neg_one_mul, one_mul]
  
  -- Proving that the cycle repeats every 4
  have cycle : ∀ n, n % 4 = 1 -> i ^ n = i :=
    λ n hn, by rw [← nat.mod_add_div n 4, nat.mod_eq_of_lt (lt_four_pow_of_pow_mod_eq_one hn), pow_add, pow_mul, h4, one_pow, mul_one]

  have cycle' : ∀ n, n % 4 = 2 -> i ^ n = -1 :=
    λ n hn, by rw [← nat.mod_add_div n 4, nat.mod_eq_of_lt (lt_four_pow_of_pow_mod_eq_two hn), pow_add, pow_mul, h4, one_pow, mul_one, pow_twos h4]

  have cycle'' : ∀ n, n % 4 = 3 -> i ^ n = -i :=
    λ n hn, by rw [← nat.mod_add_div n 4, nat.mod_eq_of_lt (lt_four_pow_of_pow_mod_eq_three hn), pow_add, pow_mul, h4, one_pow, mul_one, pow_three]

  -- Final proof that i ^ 2023 = -i
  have : 2023 % 4 = 3 := nat.mod_eq_of_lt (show 3 < 4, by norm_num)
  exact cycle'' 2023 this

end i_pow_2023_l92_92861


namespace degree_d_l92_92524

-- Definitions based on given conditions
def f : Polynomial ℝ := sorry  -- Polynomial of degree 17
def q : Polynomial ℝ := sorry  -- Quotient polynomial of degree 10
def r : Polynomial ℝ := 2*X^4 + 3*X^3 - 5*X + 7  -- Remainder polynomial with degree 4

-- Condition of the polynomial division
def main_condition := ∃ (d : Polynomial ℝ), f = d * q + r

-- Main theorem to prove
theorem degree_d (d : Polynomial ℝ) (h1 : Polynomial.degree f = 17) (h2 : Polynomial.degree q = 10) (h3 : Polynomial.degree r = 4) (h4 : main_condition) : Polynomial.degree d = 7 :=
sorry

end degree_d_l92_92524


namespace sum_of_alternating_series_l92_92862

theorem sum_of_alternating_series : (Finset.range 2007).sum (λ n, (-1)^(n+1)) = -1 := by
  sorry

end sum_of_alternating_series_l92_92862


namespace min_abs_sum_l92_92552

theorem min_abs_sum (x y : ℝ) : |x - 1| + |x| + |y - 1| + |y + 1| ≥ 3 :=
by sorry

end min_abs_sum_l92_92552


namespace right_triangle_l92_92065

-- Define the side lengths of the triangle
def a : Nat := 3
def b : Nat := 4
def c : Nat := 5

-- State that the triangle with sides a, b, and c forms a right triangle
theorem right_triangle : a^2 + b^2 = c^2 := by
  -- This theorem requires the Pythagorean theorem to be verified
  have h : 3^2 + 4^2 = 5^2 := by
    exact Nat.pow 3 2 + Nat.pow 4 2 = Nat.pow 5 2
  exact h -- Finish the proof using the verified equality

end right_triangle_l92_92065


namespace integer_solutions_count_l92_92268

theorem integer_solutions_count :
  (finset.filter (λ (x : ℤ), 5 < real.sqrt (x : ℝ) ∧ real.sqrt (x : ℝ) < 6) 
  (finset.Icc 26 35)).card = 10 :=
by
  sorry

end integer_solutions_count_l92_92268


namespace coefficient_of_x4_in_expansion_l92_92126

theorem coefficient_of_x4_in_expansion :
  (coeff (expand ((x - x^(-1/3))^8)) 4) = -56 :=
by sorry

end coefficient_of_x4_in_expansion_l92_92126


namespace bisect_angle_XMB_l92_92691

variables {A B C D M X N : Type*}
  [IsAcuteAngledTriangle A B C]  -- acute-angled triangle condition
  (hAD : is_altitude_point A D B C)  -- D is the altitude from A to BC
  (hM : is_midpoint M A C)  -- M is the midpoint of AC
  (hN : is_midpoint N A B)  -- N is the midpoint of AB
  (hOppSide : on_opposite_sides X C M B)  -- points X and C are on opposite sides of BM
  (hAngleAXB : angle_eq A X B (90 : ℝ))  -- ∠AXB = 90 degrees
  (hAngleDXM : angle_eq D X M (90 : ℝ))  -- ∠DXM = 90 degrees

theorem bisect_angle_XMB :
  bisects_angle MN XMB :=
sorry

end bisect_angle_XMB_l92_92691


namespace sum_of_super_cool_triangle_areas_l92_92911

def is_super_cool_triangle (a b : ℕ) : Prop :=
  let area := (a * b) / 2
  area = 3 * (a + b)

def area (a b : ℕ) : ℕ :=
  (a * b) / 2

theorem sum_of_super_cool_triangle_areas : 
  Σ' (a b : ℕ), is_super_cool_triangle a b → 
  sum (finset.filter (λ x, is_super_cool_triangle x.1 x.2) (finset.range 50).product (finset.range 50)).val = 471 :=
sorry

end sum_of_super_cool_triangle_areas_l92_92911


namespace min_people_to_sit_next_to_each_other_l92_92891

theorem min_people_to_sit_next_to_each_other (chairs : ℕ) (h : chairs = 72) : 
  ∃ N : ℕ, N = 18 ∧
           ∀ (place : ℕ → ℕ),
             (∀ i < N, place i < chairs) →
             (∀ i j < N, i ≠ j → place i ≠ place j) →
             (∃ i < N, ∀ j < N, (place i + 1) % chairs ≠ place j ∧ (place i - 1 + chairs) % chairs ≠ place j) →
             N ≥ 18 :=
by {
  sorry
}

end min_people_to_sit_next_to_each_other_l92_92891


namespace volume_of_tetrahedron_l92_92704

-- Define the tetrahedron ABCD with conditions
variable (A B C D : ℝ³)
variable (DC DB AD AC : ℝ)
variable (radius : ℝ) (median_point : ℝ³)
variable (touches_faces_and_edge : Prop)
variable (orthogonal : Prop)

-- Given conditions
def given_conditions :=
  DC = 9 ∧
  DB = AD ∧
  touches_faces_and_edge ∧ -- placeholder for geometric constraints with the orthogonalities and sphere touching
  orthogonal -- placeholder for orthogonality conditions of the apex and the base

-- The volume of tetrahedron function (this is usually provided in geometry libraries)
def tetrahedron_volume (A B C D : ℝ³) : ℝ := sorry

-- The main theorem statement
theorem volume_of_tetrahedron (h : given_conditions) : tetrahedron_volume A B C D = 36 :=
sorry

end volume_of_tetrahedron_l92_92704


namespace coin_tails_probability_l92_92092

theorem coin_tails_probability (p : ℝ) (h : p = 0.5) (n : ℕ) (h_n : n = 3) :
  ∃ k : ℕ, k ≤ n ∧ (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end coin_tails_probability_l92_92092


namespace best_play_majority_win_probability_l92_92137

theorem best_play_majority_win_probability (n : ℕ) :
  (1 - (1 / 2) ^ n) = probability_best_play_wins_majority n :=
sorry

end best_play_majority_win_probability_l92_92137


namespace f_monotonicity_f_has_two_zeros_l92_92030

noncomputable theory

def f (a x : ℝ) : ℝ :=
  (x^2 - 2*x) * Real.log x + (a - 1/2) * x^2 + 2 * (1 - a) * x

-- Conditions
axiom a_pos (a : ℝ) : a > 0

-- Part 1: Monotonicity of f(x)
theorem f_monotonicity (a : ℝ) (a_gt_zero : a > 0) :
  (∀ x : ℝ, 0 < x ∧ x < Real.exp (-a) → 0 < (2 * (x - 1) * (Real.log x + a))) ∧
  (∀ x : ℝ, Real.exp (-a) < x ∧ x < 1 → (2 * (x - 1) * (Real.log x + a)) < 0) ∧
  (∀ x : ℝ, 1 < x → 0 < (2 * (x - 1) * (Real.log x + a))) :=
sorry

-- Part 2: Range of values for a if f(x) has two zeros
theorem f_has_two_zeros (a : ℝ) (a_gt_zero : a > 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) ↔ a > (3 / 2) :=
sorry

end f_monotonicity_f_has_two_zeros_l92_92030


namespace members_count_l92_92827

theorem members_count
  (n : ℝ)
  (h1 : 191.25 = n / 4) :
  n = 765 :=
by
  sorry

end members_count_l92_92827


namespace product_permutation_l92_92242

theorem product_permutation :
  (finset.range 12).prod (λ i, 89 + i) = nat.perm 100 12 :=
by sorry

end product_permutation_l92_92242


namespace domain_of_f_when_m_eq_one_range_of_m_when_range_of_f_is_ℝ_range_of_m_when_f_is_increasing_on_neg_infty_to_one_l92_92338

-- (1) Given f(x) = log(x^2 - x - 1), prove the domain is (1, +∞) ∪ (-∞, -1).
theorem domain_of_f_when_m_eq_one :
  ∀ x : ℝ, f (log (x^2 - x - 1)) ↔ (1 < x ∨ x < -1) := sorry

-- (2) If the range of f(x) = log(x^2 - mx - m) is ℝ, prove the range of m is m ≥ 0 ∨ m ≤ -4.
theorem range_of_m_when_range_of_f_is_ℝ :
  (∀ y : ℝ, ∃ x : ℝ, f (log (x^2 - mx - m)) = y) ↔ (m ≥ 0 ∨ m ≤ -4) := sorry

-- (3) If f(x) = log(x^2 - mx - m) is increasing on (-∞, 1), prove the range of m is [2 - sqrt(2), 2).
theorem range_of_m_when_f_is_increasing_on_neg_infty_to_one :
  (∀ x1 x2 : ℝ, x1 < x2 ∧ x2 < 1 → f (log (x1^2 - m * x1 - m)) < f (log (x2^2 - m * x2 - m))) ↔
    (2 - sqrt 2 ≤ m ∧ m < 2) := sorry

end domain_of_f_when_m_eq_one_range_of_m_when_range_of_f_is_ℝ_range_of_m_when_f_is_increasing_on_neg_infty_to_one_l92_92338


namespace cos_A_gt_cos_B_then_a_lt_b_bcosC_c_cosB_equal_to_asinA_then_right_triangle_l92_92594

theorem cos_A_gt_cos_B_then_a_lt_b
  (A B C : ℝ)
  (a b c : ℝ)
  (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (sum_angles : A + B + C = π)
  (correspond_a : a = sqrt (b^2 + c^2 - 2 * b * c * cos A))
  (correspond_b : b = sqrt (a^2 + c^2 - 2 * a * c * cos B))
  (correspond_c : c = sqrt (a^2 + b^2 - 2 * a * b * cos C))
  (cosine_inequality : cos A > cos B) : a < b := by
sorry

theorem bcosC_c_cosB_equal_to_asinA_then_right_triangle
  (A B C : ℝ)
  (a b c : ℝ)
  (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (sum_angles : A + B + C = π)
  (law_sines : b * cos C + c * cos B = a * sin A) : A = π / 2 := by
sorry

end cos_A_gt_cos_B_then_a_lt_b_bcosC_c_cosB_equal_to_asinA_then_right_triangle_l92_92594


namespace compute_sin_90_l92_92509

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92509


namespace race_time_diff_l92_92189

-- Define the speeds and race distance
def Malcolm_speed : ℕ := 5  -- in minutes per mile
def Joshua_speed : ℕ := 7   -- in minutes per mile
def Alice_speed : ℕ := 6    -- in minutes per mile
def race_distance : ℕ := 12 -- in miles

-- Calculate times
def Malcolm_time : ℕ := Malcolm_speed * race_distance
def Joshua_time : ℕ := Joshua_speed * race_distance
def Alice_time : ℕ := Alice_speed * race_distance

-- Lean 4 statement to prove the time differences
theorem race_time_diff :
  Joshua_time - Malcolm_time = 24 ∧ Alice_time - Malcolm_time = 12 := by
  sorry

end race_time_diff_l92_92189


namespace find_ordered_set_l92_92081

theorem find_ordered_set :
  ∀ (a b c d e f : ℝ),
    (⟪ (3, a, c) ⟫ × ⟪ (6, b, d) ⟫ = 0) ∧
    (⟪ (4, b, f) ⟫ × ⟪ (8, e, d) ⟫ = 0) →
    (a, b, c, d, e, f) = (1, 2, 1, 2, 4, 1) :=
by
  intros a b c d e f h
  sorry

end find_ordered_set_l92_92081


namespace range_of_a_l92_92027

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * Real.log x - a * x

theorem range_of_a (a : ℝ) (h : a > 0) : 
  (∀ x : ℝ, 0 < x → 0 ≤ 2 * x^2 - a * x + a) ↔ 0 < a ∧ a ≤ 8 :=
by
  sorry

end range_of_a_l92_92027


namespace sin_90_eq_1_l92_92475

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92475


namespace cos_C_values_l92_92707

theorem cos_C_values (sinA : ℝ) (cosB : ℝ) (h1 : sinA = 12 / 13) (h2 : cosB = 3 / 5) :
  ∃ c : ℝ, (cos C = 33 / 65 ∨ cos C = 63 / 65) :=
by
  sorry

end cos_C_values_l92_92707


namespace cricket_run_target_l92_92697

/-- Assuming the run rate in the first 15 overs and the required run rate for the next 35 overs to
reach a target, prove that the target number of runs is 275. -/
theorem cricket_run_target
  (run_rate_first_15 : ℝ := 3.2)
  (overs_first_15 : ℝ := 15)
  (run_rate_remaining_35 : ℝ := 6.485714285714286)
  (overs_remaining_35 : ℝ := 35)
  (runs_first_15 := run_rate_first_15 * overs_first_15)
  (runs_remaining_35 := run_rate_remaining_35 * overs_remaining_35)
  (target_runs := runs_first_15 + runs_remaining_35) :
  target_runs = 275 := by
  sorry

end cricket_run_target_l92_92697


namespace solve_diamondsuit_l92_92344

-- Define the binary operation diamondsuit.
def diamondsuit (a b : ℝ) : ℝ := a / b

-- State the conditions as axioms.
axiom diamondsuit_assoc (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) : a ≠ 0 → b ≠ 0 → c ≠ 0 → diamondsuit a (diamondsuit b c) = (diamondsuit a b) * c
axiom diamondsuit_idem (a : ℝ) (h : a ≠ 0) : diamondsuit a a = 1

-- The main statement to be proved.
theorem solve_diamondsuit : ∃ x : ℝ, 2024 ≠ 0 ∧ 8 ≠ 0 ∧ 2024 ≠ 0 ∧ 8 ≠ 0 ∧ x ≠ 0 ∧ diamondsuit 2024 (diamondsuit 8 x) = 200 → x = 200 / 253 :=
by
  sorry -- Proof omitted

end solve_diamondsuit_l92_92344


namespace sin_90_eq_one_l92_92407

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92407


namespace integer_values_count_l92_92253

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l92_92253


namespace count_integers_between_25_and_36_l92_92280

theorem count_integers_between_25_and_36 :
  {x : ℤ | 25 < x ∧ x < 36}.finite.card = 10 :=
by
  sorry

end count_integers_between_25_and_36_l92_92280


namespace ratio_b_c_l92_92589

theorem ratio_b_c (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : a * b * c / (d * e * f) = 0.1875)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) : 
  b / c = 3 :=
sorry

end ratio_b_c_l92_92589


namespace no_such_matrix_exists_l92_92526

theorem no_such_matrix_exists (a b c d : ℝ) :
  ¬∃ (M : Matrix (Fin 2) (Fin 2) ℝ), M ⬝ (Matrix.of ![![a, b], ![c, d]]) = Matrix.of ![![2*a, 3*b], ![c, d]] :=
by sorry

end no_such_matrix_exists_l92_92526


namespace units_digit_L_L_15_l92_92785

def Lucas (n : ℕ) : ℕ :=
match n with
| 0 => 2
| 1 => 1
| n + 2 => Lucas n + Lucas (n + 1)

theorem units_digit_L_L_15 : (Lucas (Lucas 15)) % 10 = 7 := by
  sorry

end units_digit_L_L_15_l92_92785


namespace coefficient_x_pow_4_expansion_l92_92946

theorem coefficient_x_pow_4_expansion : 
  let f := x * (x - (2 / x)) ^ 7,
      expansion := Algebra.expand f
  in coefficient expansion 4 = 84 :=
sorry

end coefficient_x_pow_4_expansion_l92_92946


namespace sum_of_super_cool_triangle_areas_l92_92910

def is_super_cool_triangle (a b : ℕ) : Prop :=
  let area := (a * b) / 2
  area = 3 * (a + b)

def area (a b : ℕ) : ℕ :=
  (a * b) / 2

theorem sum_of_super_cool_triangle_areas : 
  Σ' (a b : ℕ), is_super_cool_triangle a b → 
  sum (finset.filter (λ x, is_super_cool_triangle x.1 x.2) (finset.range 50).product (finset.range 50)).val = 471 :=
sorry

end sum_of_super_cool_triangle_areas_l92_92910


namespace number_of_employees_excluding_manager_l92_92789

theorem number_of_employees_excluding_manager 
  (avg_salary : ℕ)
  (manager_salary : ℕ)
  (new_avg_salary : ℕ)
  (n : ℕ)
  (T : ℕ)
  (h1 : avg_salary = 1600)
  (h2 : manager_salary = 3700)
  (h3 : new_avg_salary = 1700)
  (h4 : T = n * avg_salary)
  (h5 : T + manager_salary = (n + 1) * new_avg_salary) :
  n = 20 :=
by
  sorry

end number_of_employees_excluding_manager_l92_92789


namespace complex_in_fourth_quadrant_l92_92334

def i : ℂ := complex.I

def z : ℂ := (5 - i) / (1 + i)

theorem complex_in_fourth_quadrant (z : ℂ) : 
  z = (5 - i) / (1 + i) → z.re > 0 ∧ z.im < 0 :=
by
  intro h
  sorry

end complex_in_fourth_quadrant_l92_92334


namespace people_per_column_in_second_arrangement_l92_92682
-- Lean 4 Statement

theorem people_per_column_in_second_arrangement :
  ∀ P X : ℕ, (P = 30 * 16) → (12 * X = P) → X = 40 :=
by
  intros P X h1 h2
  sorry

end people_per_column_in_second_arrangement_l92_92682


namespace sin_90_degree_l92_92502

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92502


namespace problem_proof_l92_92605

theorem problem_proof (m : ℝ) (h_pos : 0 < m) :
  (∀ x1 ∈ Icc (-1 : ℝ) 2, ∃ x2 ∈ Icc (-1 : ℝ) 2, (1/2)^x1 = m * x2 - 1) →
  m ≥ 3 / 2 :=
by
  sorry

end problem_proof_l92_92605


namespace locus_of_P_on_diagonals_l92_92068

-- Given a convex quadrilateral ABCD:
variables {A B C D P : Type} [convex_quadrilateral A B C D]

-- Define areas of triangles functions
def area (X Y Z : Type) : Type := sorry  -- (Implementation is skipped here)

-- Define the locus condition
def area_product_condition (P A B C D : Type) :=
  let area_PAB := area P A B in
  let area_PCD := area P C D in
  let area_PBC := area P B C in
  let area_PDA := area P D A in
  area_PAB * area_PCD = area_PBC * area_PDA

-- Define the diagonals
def on_diagonal_BD (P : Type) : Type := sorry  -- (Implementation is skipped here)
def on_diagonal_AC (P : Type) : Type := sorry  -- (Implementation is skipped here)

-- State the theorem
theorem locus_of_P_on_diagonals (h : area_product_condition P A B C D) :
  on_diagonal_BD P ∨ on_diagonal_AC P := sorry

end locus_of_P_on_diagonals_l92_92068


namespace distance_between_foci_of_ellipse_l92_92971

theorem distance_between_foci_of_ellipse (x y : ℝ) :
  9 * x^2 + y^2 = 36 → 2 * real.sqrt (36 - 4) = 8 * real.sqrt 2 :=
by
  intro h
  calc
    2 * real.sqrt (36 - 4) = 2 * real.sqrt (32) : sorry
    ...                   = 2 * 4 * real.sqrt 2  : sorry
    ...                   = 8 * real.sqrt 2      : sorry

end distance_between_foci_of_ellipse_l92_92971


namespace sin_90_eq_1_l92_92462

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92462


namespace expected_tied_scores_l92_92831

theorem expected_tied_scores (n : ℕ) (h : n = 5) :
  let E_X := ∑ k in Finset.range n, (Nat.choose (2 * k) k : ℚ) / (2 : ℚ)^(2 * k)
  E_X = 1.707 :=
by
  -- Define the necessary conditions and expected value calculation.
  let indicator_probability : ℕ → ℚ := λ k, (Nat.choose (2 * k) k : ℚ) / (2 : ℚ)^(2 * k)
  let E_X := ∑ k in Finset.range n, indicator_probability k
  
  -- Calculate the expected number of times the score was tied.
  have hE_X : E_X = 1.707 := by sorry
  
  exact hE_X

end expected_tied_scores_l92_92831


namespace smallest_value_of_x_l92_92865

theorem smallest_value_of_x (x : ℝ) (h : 6 * x ^ 2 - 37 * x + 48 = 0) : x = 13 / 6 :=
sorry

end smallest_value_of_x_l92_92865


namespace large_circle_diameter_l92_92955

theorem large_circle_diameter :
  ∃ d, d = 28.92 ∧ 
    ∀ (n r : ℝ) (large_circle small_circle : ℝ → Prop),
       (n = 8) ∧ (r = 4) ∧ 
       (∀ θ : ℝ, small_circle θ ↔ θ = 0 ∨ θ = 2 * r * sin(π / n)) ∧
       (∀ θ : ℝ, large_circle θ ↔ θ = 0 ∨ θ = r + (r / (2 * sin(π / n)))) →
             d = 2 * (r / (2 * sin(π / 8)) + r)
:= by
  sorry

end large_circle_diameter_l92_92955


namespace sum_of_prime_factors_of_91_l92_92866

theorem sum_of_prime_factors_of_91 : 
  (¬ (91 % 2 = 0)) ∧ 
  (¬ (91 % 3 = 0)) ∧ 
  (¬ (91 % 5 = 0)) ∧ 
  (91 = 7 * 13) →
  (7 + 13 = 20) := 
by 
  intros h
  sorry

end sum_of_prime_factors_of_91_l92_92866


namespace sin_ninety_deg_l92_92435

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92435


namespace compute_sin_90_l92_92508

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92508


namespace a2_is_3_a3_is_4_a4_is_5_general_formula_l92_92519

def sequence_a : ℕ → ℕ
| 0       := 2
| (n + 1) := (sequence_a n)^2 - n * (sequence_a n) + 1

theorem a2_is_3 : sequence_a 1 = 3 := sorry

theorem a3_is_4 : sequence_a 2 = 4 := sorry

theorem a4_is_5 : sequence_a 3 = 5 := sorry

theorem general_formula (n : ℕ) : sequence_a n = n + 1 := sorry

end a2_is_3_a3_is_4_a4_is_5_general_formula_l92_92519


namespace rabbit_weight_l92_92908

variables (p s l r : ℝ)

-- Definitions from conditions
def condition1 : Prop := p + s + l + r = 36
def condition2 : Prop := p + l = 3 * s
def condition3 : Prop := p + s = l + 10
def condition4 : Prop := r = s

-- Theorem stating the rabbit's weight
theorem rabbit_weight : condition1 ∧ condition2 ∧ condition3 ∧ condition4 → r = 9 := by
  sorry

end rabbit_weight_l92_92908


namespace sin_90_deg_l92_92451

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92451


namespace sum_of_super_cool_areas_l92_92913

noncomputable def is_super_cool_right_triangle (a b : ℕ) : Prop :=
  (a * b / 2) = 3 * (a + b)

theorem sum_of_super_cool_areas :
  let pairs := [(7, 42), (8, 24), (9, 18), (10, 15), (12, 12)] in
  let areas := list.map (λ (ab : ℕ × ℕ), ab.1 * ab.2 / 2) pairs in
  list.sum areas = 471 :=
by
  sorry

end sum_of_super_cool_areas_l92_92913


namespace Part_a_Part_b_l92_92724

open SimpleGraph

def switchable (G : SimpleGraph ℕ) : Prop :=
  ∃ (sequence : list ℕ), ∀ e : G.edge_finset, G.color_flip_sequence e sequence = color.blue

theorem Part_a (n k : ℕ) (h : n ≥ 3) :
  ∀ (G : SimpleGraph ℕ), (G.order = n) → (G.size = k) → ¬ switchable G :=
sorry

theorem Part_b (n k : ℕ) (h : n ≥ 3) (h_le : k ≤ (n*n / 4)) :
  ∃ (G : SimpleGraph ℕ), (G.order = n) ∧ (G.size = k) ∧ switchable G :=
sorry

end Part_a_Part_b_l92_92724


namespace ratio_of_sides_l92_92560

-- Conditions
variable (s y x : ℝ)
variable (h1 : ∀ rects, congruent rects)
variable (h2 : ∀ rects, ∀ adj, (perpendicular long_side short_side))
variable (h3 : (3 * s)^2 = 9 * s^2)

-- Definitions derived from conditions
def inner_square_area := s^2
def outer_square_area := (3 * s)^2
def y_eq_s : y = s := by sorry
def x_eq_2s : x = 2 * s := by sorry

-- Proof statement
theorem ratio_of_sides (s y x : ℝ) (h1 : ∀ rects, congruent rects)
  (h2 : ∀ rects, ∀ adj, (perpendicular long_side short_side))
  (h3 : (3 * s)^2 = 9 * s^2) 
  (y_eq_s : y = s)
  (x_eq_2s : x = 2 * s) :
  x / y = 2 :=
by
  calc
  x / y = (2 * s) / s : by rw [x_eq_2s, y_eq_s]
       ... = 2 : by sorry

end ratio_of_sides_l92_92560


namespace _l92_92829

noncomputable def isosceles_right_triangle : Type :=
  {ABC : Triangle // ABC.is_isosceles_right ∧ ABC.a = 5 ∧ ABC.b = 5}

noncomputable def midpoint_at_M (ABC : isosceles_right_triangle) : Point :=
  midpoint ABC.c ABC.d

noncomputable def cyclic_quadrilateral (I E : Point) (M : Point) : Prop :=
  ∃ (AI AE : ℝ), AI > AE ∧ quadrilateral_cyclic {A M I E}

noncomputable theorem find_CI_length {ABC : isosceles_right_triangle}
    (M : Point)
    (h_M : M = midpoint_at_M ABC)
    (I E : Point)
    (h_cyclic : cyclic_quadrilateral I E M)
    (h_area : (area (triangle I E M) = 4)) : 
    ∃ (a b c : ℕ), b ≤ a ∧ (b = 0 ∨ ∀ p, prime p → p^2 ∤ b) ∧ 1 - b = 1 ∧ c = 2 ∧ 1 + b + c = 3 := 
sorry

end _l92_92829


namespace arithmetic_progression_common_difference_zero_l92_92686

theorem arithmetic_progression_common_difference_zero {a d : ℤ} (h₁ : a = 12) 
  (h₂ : ∀ n : ℕ, a + n * d = (a + (n + 1) * d + a + (n + 2) * d) / 2) : d = 0 :=
  sorry

end arithmetic_progression_common_difference_zero_l92_92686


namespace sin_90_degrees_l92_92483

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92483


namespace exists_N_l92_92722

open Nat

def smallest_prime_divisor (n : ℕ) : ℕ := sorry 

noncomputable def a_sequence (a₁ : ℕ) (p : ℕ → ℕ) : ℕ → ℕ
| 0       => a₁
| (n + 1) => let a_n := a_sequence a₁ p n
             in a_n + a_n / p n

theorem exists_N (a₁ : ℕ) (h₁ : a₁ ≥ 2) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → let p : ℕ → ℕ := λ n, smallest_prime_divisor (a_sequence a₁ p n)
                              in a_sequence a₁ p (n + 3) = 3 * (a_sequence a₁ p n) :=
begin
  sorry
end

end exists_N_l92_92722


namespace dime_probability_l92_92356

def dime_value : ℝ := 0.10
def quarter_value : ℝ := 0.25
def half_dollar_value : ℝ := 0.50

def total_dimes_value : ℝ := 12.00
def total_quarters_value : ℝ := 15.00
def total_half_dollars_value : ℝ := 20.00

def num_dimes : ℕ := (total_dimes_value / dime_value).toNat
def num_quarters : ℕ := (total_quarters_value / quarter_value).toNat
def num_half_dollars : ℕ := (total_half_dollars_value / half_dollar_value).toNat

def total_coins : ℕ := num_dimes + num_quarters + num_half_dollars

def probability_of_dime : ℚ := num_dimes / total_coins

theorem dime_probability :
  probability_of_dime = 6 / 11 :=
by
  sorry

end dime_probability_l92_92356


namespace sin_90_degrees_l92_92492

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92492


namespace quiz_total_points_l92_92113

theorem quiz_total_points (points : ℕ → ℕ) 
  (h1 : ∀ n, points (n+1) = points n + 4)
  (h2 : points 2 = 39) : 
  (points 0 + points 1 + points 2 + points 3 + points 4 + points 5 + points 6 + points 7) = 360 :=
sorry

end quiz_total_points_l92_92113


namespace speed_of_first_32_miles_l92_92534

-- Define Daniel's daily driving distance
def daily_distance : ℝ := 96

-- Define Daniel's speed on Sunday (x) and unknown speed on Monday (y)
variables (x y : ℝ) (h_x_pos : 0 < x)

-- Condition: Daniel drives 96 miles at speed x on Sunday
def T_sunday := daily_distance / x

-- Conditions for Monday's drive
def first_distance : ℝ := 32
def remaining_distance : ℝ := daily_distance - first_distance

-- Time to drive parts on Monday
def T_first_part := first_distance / y
def T_second_part := remaining_distance / (x / 2)

-- Total time on Monday
def T_monday := T_first_part + T_second_part

-- Given condition: Time on Monday is 50% more than time on Sunday
def condition := T_monday = 1.5 * T_sunday

-- The theorem to prove
theorem speed_of_first_32_miles (h : condition):
  y = 2 * x :=
by
  sorry

end speed_of_first_32_miles_l92_92534


namespace proof_problem_l92_92590

variables {a b c : ℝ} {A C : ℝ}
variables (sin : ℝ → ℝ) 

-- Condition 1: Vector m
def m := (a - b, sin A + sin C)

-- Condition 2: Vector n
def n := (a - c, sin(A + C))

-- Condition 3: Vectors m and n are collinear
def collinear (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Condition 4: Dot product condition
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def AC := (a, sin A)
def CB := (b, sin C)
def collinear_m_n := collinear m n
def dot_product_condition := dot_product AC CB = -27

-- Question 1: Verify that angle C is π/3
def angle_C_eq_pi_over_3 : Prop := C = π / 3

-- Question 2: Verify minimum value of |AB|
def min_value_AB (AB : ℝ) : Prop := AB = 3 * sqrt 6

-- Combined statement
theorem proof_problem
(collinear_m_n : collinear m n)
(dot_product_condition : dot_product AC CB = -27)
(angle_C_eq_pi_over_3 : C = π / 3)
(min_value_AB : ∀ AB, AB = 3 * sqrt 6) : sorry

end proof_problem_l92_92590


namespace sin_ninety_degrees_l92_92404

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92404


namespace symmetry_origin_points_l92_92579

theorem symmetry_origin_points (x y : ℝ) (h₁ : (x, -2) = (-3, -y)) : x + y = -1 :=
sorry

end symmetry_origin_points_l92_92579


namespace find_m_l92_92628

theorem find_m (a m : ℝ) (h_pos : a > 0) (h_points : (m, 3) ∈ set_of (λ x : ℝ × ℝ, ∃ x_val : ℝ, x.snd = -a * (x_val)^2 + 2 * a * x_val + 3)) (h_non_zero : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92628


namespace shaded_triangle_probability_l92_92118

-- Define the points and triangles
variables (F G H I J : Type*)

-- Define the triangles
def FGJ := (F, G, J)
def FGH := (F, G, H)
def GHI := (G, H, I)
def HIJ := (H, I, J)
def FIJ := (F, I, J)

-- Define the set of all triangles
def all_triangles : set (F × G × H) := {FGJ, FGH, GHI, HIJ, FIJ}

-- Define the problem statement
theorem shaded_triangle_probability :
  (∃ (TG : set (F × G × H)), TG = {GHI} ∧ ⊆ all_triangles ∧ TG.card = 1 ∧ all_triangles.card = 5) →
  (probability TG / probability all_triangles = 1 / 5) :=
begin
  sorry
end

end shaded_triangle_probability_l92_92118


namespace cosine_inequality_solution_l92_92539

noncomputable def cosine_inequality_solution_set : set ℝ := {y | (0 ≤ y ∧ y ≤ π/4) ∨ (5*π/4 ≤ y ∧ y ≤ 2*π)}

theorem cosine_inequality_solution (y : ℝ) (h : 0 ≤ y ∧ y ≤ 2*π) :
  (cos(π/2 + y) ≥ cos(π/2) - cos(y)) ↔ y ∈ cosine_inequality_solution_set :=
sorry

end cosine_inequality_solution_l92_92539


namespace sum_even_coefficients_identity_l92_92729

theorem sum_even_coefficients_identity (n : ℕ) : 
  let f : ℝ → ℝ := λ x, (1 - x + x^2)^n
  let s := (∑ i in (finset.range (2*n+1)).filter (λ i, i%2 = 0), (f i) / (x^i)) 
  s = (1 + 3^n) / 2 := sorry

end sum_even_coefficients_identity_l92_92729


namespace sum_of_three_is_odd_implies_one_is_odd_l92_92775

theorem sum_of_three_is_odd_implies_one_is_odd 
  (a b c : ℤ) 
  (h : (a + b + c) % 2 = 1) : 
  a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1 := 
sorry

end sum_of_three_is_odd_implies_one_is_odd_l92_92775


namespace sum_of_first_seven_terms_l92_92811

noncomputable def sum_arithmetic_sequence (n : ℕ) (a d : ℚ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_seven_terms (a d S : ℚ) (h1 : a + d + a + 9 * d = 16) (h2 : a + 7 * d = 11)
  (hS : S = sum_arithmetic_sequence 7 a d) : S = 35 :=
by calc
  S = sum_arithmetic_sequence 7 a d : by rw hS
   ... = 7 * (2 * a + (7 - 1) * d) / 2 : by rw sum_arithmetic_sequence
   ... = 7 * (2 * a + 6 * d) / 2 : by norm_num
   ... = 7 * 2 * (a + 3 * d) / 2 : by ring
   ... = 7 * (a + 3 * d) : by ring
   ... = 7 * 5 : by { simp only [h2], ring }
   ... = 35 : by norm_num  

end sum_of_first_seven_terms_l92_92811


namespace sum_of_solutions_f_eq_0_l92_92746

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then -2 * x - 6 else x / 3 + 2

theorem sum_of_solutions_f_eq_0 : 
  (finset.univ.filter (λ x, f x = 0)).sum id = -3 :=
sorry

end sum_of_solutions_f_eq_0_l92_92746


namespace ball_bounce_height_l92_92887

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (3 / 4 : ℝ)^k < 2) ∧ ∀ n < k, ¬ (20 * (3 / 4 : ℝ)^n < 2) :=
sorry

end ball_bounce_height_l92_92887


namespace perimeter_formula_l92_92350

noncomputable def perimeter_hexagon : ℝ :=
  -- Define the points
  let p1 := (0, 0)
  let p2 := (1, 1)
  let p3 := (3, 1)
  let p4 := (4, 0)
  let p5 := (3, -1)
  let p6 := (1, -1)
  -- Calculate the distance between consecutive points
  let d12 := Real.sqrt ((1 - 0) ^ 2 + (1 - 0) ^ 2)
  let d23 := Real.sqrt ((3 - 1) ^ 2 + (1 - 1) ^ 2)
  let d34 := Real.sqrt ((4 - 3) ^ 2 + (0 - 1) ^ 2)
  let d45 := Real.sqrt ((3 - 4) ^ 2 + (-1 - 0) ^ 2)
  let d56 := Real.sqrt ((1 - 3) ^ 2 + (-1 + 1) ^ 2)
  let d61 := Real.sqrt ((0 - 1) ^ 2 + (0 + 1) ^ 2)
  -- Sum the distances to get the perimeter
  d12 + d23 + d34 + d45 + d56 + d61

theorem perimeter_formula : ∃ a b c : ℕ, a + b * Real.sqrt 2 + c * Real.sqrt 5 = perimeter_hexagon ∧ a + b + c = 8 :=
by
  use 4, 4, 0
  split
  · simp [perimeter_hexagon]
    done
  · done

end perimeter_formula_l92_92350


namespace determine_m_l92_92634

theorem determine_m (a m : ℝ) (h : a > 0) (h2 : (m, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = -a * p.1 ^ 2 + 2 * a * p.1 + 3)) (h3 : m ≠ 0) : m = 2 :=
sorry

end determine_m_l92_92634


namespace integer_solutions_count_l92_92269

theorem integer_solutions_count :
  (finset.filter (λ (x : ℤ), 5 < real.sqrt (x : ℝ) ∧ real.sqrt (x : ℝ) < 6) 
  (finset.Icc 26 35)).card = 10 :=
by
  sorry

end integer_solutions_count_l92_92269


namespace gratuities_charged_l92_92914

-- Define the conditions in the problem
def total_bill : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def ny_striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Calculate the total cost before tax and gratuities
def subtotal : ℝ := ny_striploin_cost + wine_cost

-- Calculate the taxes paid
def tax : ℝ := subtotal * sales_tax_rate

-- Calculate the total bill before gratuities
def total_before_gratuities : ℝ := subtotal + tax

-- Goal: Prove that gratuities charged is 41
theorem gratuities_charged : (total_bill - total_before_gratuities) = 41 := by sorry

end gratuities_charged_l92_92914


namespace min_value_expression_l92_92737

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x + y) * (1 / x + 1 / y) ≥ 6 := 
by
  sorry

end min_value_expression_l92_92737


namespace sin_ninety_degrees_l92_92397

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92397


namespace geometric_progression_product_l92_92034

theorem geometric_progression_product (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
  (h1 : a 3 = a1 * r^2)
  (h2 : a 10 = a1 * r^9)
  (h3 : a1 * r^2 + a1 * r^9 = 3)
  (h4 : a1^2 * r^11 = -5) :
  a 5 * a 8 = -5 :=
by
  sorry

end geometric_progression_product_l92_92034


namespace bedrooms_count_l92_92712

/-- Number of bedrooms calculation based on given conditions -/
theorem bedrooms_count (B : ℕ) (h1 : ∀ b, b = 20 * B)
  (h2 : ∀ lr, lr = 20 * B)
  (h3 : ∀ bath, bath = 2 * 20 * B)
  (h4 : ∀ out, out = 2 * (20 * B + 20 * B + 40 * B))
  (h5 : ∀ siblings, siblings = 3)
  (h6 : ∀ work_time, work_time = 4 * 60) : B = 3 :=
by
  -- proof will be provided here
  sorry

end bedrooms_count_l92_92712


namespace acute_angle_at_940_l92_92864

-- Conditions
def clock_angles := λ minute hour : ℝ, 
  (minute / 60 * 360, 
   (hour % 12 + minute / 60) * 30)

-- Question & Proof Statement
theorem acute_angle_at_940: 
  let t := clock_angles 40 9 in 
  (t.2 - t.1 = 50 ∨ t.1 - t.2 = 50) :=
by
  let min_angle := 40 / 60 * 360
  let hour_angle := (9 + 40 / 60) * 30
  have angle : hour_angle - min_angle = 50 := sorry
  exact Or.inl angle

end acute_angle_at_940_l92_92864


namespace solve_for_x_l92_92091

theorem solve_for_x (x : ℝ) (h: (6 / (x + 1) = 3 / 2)) : x = 3 :=
sorry

end solve_for_x_l92_92091


namespace find_p_and_q_l92_92596

theorem find_p_and_q :
  ∀ (p q : ℝ),
  (p - 3 = 0) ∧ (q - 3 * p + 8 = 0) → (p = 3 ∧ q = 1) :=
by
  intros p q ⟨h1, h2⟩
  sorry

end find_p_and_q_l92_92596


namespace cylindrical_tank_volume_l92_92384

theorem cylindrical_tank_volume (d h : ℝ) (d_def : d = 20) (h_def : h = 10) : 
  ∃ (V : ℝ), V = 1000 * Real.pi :=
by
  let r := d / 2
  have r_def : r = 10 := by linarith [d_def]
  have V_def : V = Real.pi * r^2 * h := by linarith [r_def, h_def]
  use V
  rw [V_def, r_def, h_def]
  sorry

end cylindrical_tank_volume_l92_92384


namespace find_area_of_square_EFGH_l92_92930

noncomputable def radius_of_semicircles (side_length : ℝ) : ℝ :=
  side_length / 2

noncomputable def side_length_EFGH (original_side_length : ℝ) (radius : ℝ) : ℝ :=
  original_side_length + 2 * radius

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length * side_length

theorem find_area_of_square_EFGH :
  let original_side_length := 6 in
  let radius := radius_of_semicircles original_side_length in
  let side_length_EFGH := side_length_EFGH original_side_length radius in
  area_of_square side_length_EFGH = 144 :=
by
  sorry

end find_area_of_square_EFGH_l92_92930


namespace probability_at_least_two_odd_given_one_odd_l92_92874

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def count_odd (numbers : Fin 3 → ℕ) : ℕ :=
  (Finset.univ.filter (λ i => is_odd (numbers i))).card

def at_least_one_odd (numbers : Fin 3 → ℕ) : Prop :=
  count_odd numbers ≥ 1

def at_least_two_odd (numbers : Fin 3 → ℕ) : Prop :=
  count_odd numbers ≥ 2

theorem probability_at_least_two_odd_given_one_odd :
  ∀ (numbers : Fin 3 → ℕ),
    at_least_one_odd numbers →
    (Probability (at_least_two_odd numbers | at_least_one_odd numbers) = 4 / 7) :=
by
  -- Formal proof goes here
  sorry

end probability_at_least_two_odd_given_one_odd_l92_92874


namespace star_comm_l92_92770

section SymmetricOperation

variable {S : Type*} 
variable (star : S → S → S)
variable (symm : ∀ a b : S, star a b = star (star b a) (star b a)) 

theorem star_comm (a b : S) : star a b = star b a := 
by 
  sorry

end SymmetricOperation

end star_comm_l92_92770


namespace number_of_integers_between_25_and_36_l92_92263

theorem number_of_integers_between_25_and_36 :
  {n : ℕ | 25 < n ∧ n < 36}.card = 10 :=
by
  sorry

end number_of_integers_between_25_and_36_l92_92263


namespace concentration_salt_solution_used_l92_92348

theorem concentration_salt_solution_used :
  ∀ (C : ℝ), 
  0.5 * (C / 100) = 1.5 * (20 / 100) -> 
  C = 60 :=
by
  intro C
  intro h
  linarith

end concentration_salt_solution_used_l92_92348


namespace sin_90_degree_l92_92501

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92501


namespace granola_bars_split_l92_92647

theorem granola_bars_split : 
  ∀ (initial_bars : ℕ) (days_of_week : ℕ) (traded_bars : ℕ) (sisters : ℕ),
  initial_bars = 20 →
  days_of_week = 7 →
  traded_bars = 3 →
  sisters = 2 →
  (initial_bars - days_of_week - traded_bars) / sisters = 5 :=
by
  intros initial_bars days_of_week traded_bars sisters
  intros h_initial h_days h_traded h_sisters
  rw [h_initial, h_days, h_traded, h_sisters]
  norm_num
  sorry

end granola_bars_split_l92_92647


namespace time_spent_washing_car_l92_92756

theorem time_spent_washing_car (x : ℝ) 
  (h1 : x + (1/4) * x = 100) : x = 80 := 
sorry  

end time_spent_washing_car_l92_92756


namespace domain_transform_l92_92063

theorem domain_transform (f : ℝ → ℝ) : 
  (∀ x, x ∈ set.Icc (-3 : ℝ) (1 : ℝ) → f x ∈ set.univ) →
  (∀ x, x ∈ set.Icc (-1 : ℝ) (1 : ℝ) → f (2 * x - 1) ∈ set.univ) :=
by
  intro h
  sorry

end domain_transform_l92_92063


namespace geometric_sequence_b_range_l92_92825

noncomputable def geometric_sequence_bounds (a b c m : ℝ) (h1: a > 0) (h2: a + b + c = m) (h3: a * c = b^2) : Prop :=
  b ∈ set.Icc (-m) 0 ∪ set.Ioc 0 (m/3)

theorem geometric_sequence_b_range (a b c m : ℝ) (h1 : a > 0) (h2 : a + b + c = m) (h3 : a * c = b^2) (h4 : m > 0) :
  geometric_sequence_bounds a b c m h1 h2 h3 :=
  sorry

end geometric_sequence_b_range_l92_92825


namespace percent_decrease_in_price_l92_92164

theorem percent_decrease_in_price (old_price_per_pack new_price_per_pack : ℝ)
    (old_total_price : ℝ := 7) (old_pack_count : ℕ := 3)
    (new_total_price : ℝ := 5) (new_pack_count : ℕ := 4)
    (h_old_price : old_price_per_pack = old_total_price / old_pack_count)
    (h_new_price : new_price_per_pack = new_total_price / new_pack_count) :
  ((old_price_per_pack - new_price_per_pack) / old_price_per_pack) * 100 ≈ 46 := 
by
  sorry

end percent_decrease_in_price_l92_92164


namespace compute_sin_90_l92_92505

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92505


namespace correct_operation_l92_92323

variable {a : ℝ}

theorem correct_operation : a^4 / (-a)^2 = a^2 := by
  sorry

end correct_operation_l92_92323


namespace greater_number_l92_92287

theorem greater_number (a b : ℕ) (h1 : a + b = 36) (h2 : a - b = 8) : a = 22 :=
by
  sorry

end greater_number_l92_92287


namespace solve_for_x_l92_92016

theorem solve_for_x (x : ℚ) (h : (sqrt (8 * x)) / (sqrt (2 * (x - 2))) = 3) : 
  x = 18 / 5 := 
by 
  sorry

end solve_for_x_l92_92016


namespace binom_13_11_eq_78_l92_92392

theorem binom_13_11_eq_78 : Nat.choose 13 11 = 78 := by
  sorry

end binom_13_11_eq_78_l92_92392


namespace smallest_composite_no_prime_factors_less_than_20_l92_92991

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92991


namespace same_cost_number_of_guests_l92_92216

theorem same_cost_number_of_guests (x : ℕ) : 
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by {
  sorry
}

end same_cost_number_of_guests_l92_92216


namespace sin_90_degree_l92_92499

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92499


namespace sin_ninety_deg_l92_92429

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92429


namespace parallelogram_area_l92_92901

noncomputable def base : ℝ := 20
noncomputable def height : ℝ := 4

theorem parallelogram_area :
  base * height = 80 :=
by
  sorry

end parallelogram_area_l92_92901


namespace trajectory_and_min_dot_product_l92_92036

-- Step 1: Define the conditions
def P_f_eq_distance (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in
  (x - 1) ^ 2 + y ^ 2 = abs (x + 1) ^ 2

def passing_through_F (F : ℝ × ℝ) (slope : ℝ) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in
  y = slope * (x - F.1)

-- Step 2: Define the main theorem
theorem trajectory_and_min_dot_product (P : ℝ × ℝ) (F : ℝ × ℝ) (l1_slope l2_slope : ℝ) :
  P_f_eq_distance P →
  F = (1, 0) →
  l1_slope ≠ 0 →
  l2_slope = -1 / l1_slope →
  (∃ (C : ℝ → ℝ), ∀ x, C x = 4 * x) ∧
  ∃ A B D E : ℝ × ℝ,
    -- Line l1 passing through F with slope l1_slope
    passing_through_F F l1_slope A ∧ passing_through_F F l1_slope B ∧
    -- Line l2 passing through F perpendicular to l1
    passing_through_F F l2_slope D ∧ passing_through_F F l2_slope E →
  ∀ (A B D E : ℝ × ℝ),
    -- Verify minimum dot product
    let AD := (D.1 - A.1, D.2 - A.2) in
    let EB := (B.1 - E.1, B.2 - E.2) in
    min ((AD.1 * EB.1) + (AD.2 * EB.2)) 16 = 16 :=
sorry

end trajectory_and_min_dot_product_l92_92036


namespace modulus_of_z_l92_92225

open Complex

theorem modulus_of_z (z : ℂ) (h : z * (3 - 4 * Complex.i) = 1) : Complex.abs z = 1 / 5 :=
by
  sorry

end modulus_of_z_l92_92225


namespace solve_hyperbola_parabola_l92_92096

noncomputable def hyperbola_focus_condition (p : ℝ) (hp : p > 0) : Prop :=
  let focus : ℝ := -Real.sqrt (3 + p^2 / 16) in
  focus = -p/2

theorem solve_hyperbola_parabola (p : ℝ) (hp : p > 0) : p = 4 :=
by
  have H : hyperbola_focus_condition p hp := sorry
  sorry

end solve_hyperbola_parabola_l92_92096


namespace fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l92_92075

theorem fixed_point_of_line (a : ℝ) (A : ℝ × ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> A = (1, 6)) :=
sorry

theorem range_of_a_to_avoid_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, (a - 1) * x + y - a - 5 = 0 -> x * y < 0 -> a ≤ -5) :=
sorry

end fixed_point_of_line_range_of_a_to_avoid_second_quadrant_l92_92075


namespace collinear_A_F_C_l92_92939

-- Definitions of the circles and tangency points
variables {F A B C : Point} {S1 S2 : Circle}

-- Conditions
axiom circles_touch_externally (S1 S2 : Circle) (F : Point) :
  touches_externally_at S1 S2 F

axiom line_tangent_to_S1_at_A (l : Line) (S1 : Circle) (A : Point) :
  tangent_to S1 l A

axiom line_tangent_to_S2_at_B (l : Line) (S2 : Circle) (B : Point) :
  tangent_to S2 l B

axiom parallel_line_tangent_to_S2_at_C (l' : Line) (l : Line) (S2 : Circle) (C : Point) :
  parallel l l' ∧ tangent_to S2 l' C

axiom parallel_line_intersects_S1 (l' : Line) (S1 : Circle) :
  intersects_in_two_points l' S1

-- The proof statement
theorem collinear_A_F_C :
  are_collinear A F C :=
sorry

end collinear_A_F_C_l92_92939


namespace curve_equation_perpendicular_line_max_triangle_area_l92_92124

section Problem1
variable {M : ℝ × ℝ}
variable {A : ℝ × ℝ := (1, 0)}
variable {B : ℝ × ℝ := (4, 0)}

theorem curve_equation (h: dist M B = 2 * dist M A):
  (M.1)^2 + (M.2)^2 = 4 := 
sorry
end Problem1

section Problem2Part1
variable {P Q : ℝ × ℝ}
variable {O : ℝ × ℝ := (0, 0)}
variable l : ℝ → ℝ := fun x => x * (x - 4)

theorem perpendicular_line (h: dist O P = dist O Q ∧ P.1 ≠ Q.1):
  l = (fun x => x * (x - 4) * (7.sqrt / 7)):
sorry
end Problem2Part1

section Problem2Part2
variable {P Q A : ℝ × ℝ}
variable {l : ℝ → ℝ := fun x => x * (x - 4)}

theorem max_triangle_area (hPQ: dist P Q = 2 * sqrt (4 - (dist O P) ** 2) ∧
                                P ∈ l ∧ Q ∈ l ∧ P.1 ≠ Q.1):
  (1/2) * dist P Q * (dist A {P.1, P.2}) = (3/2): 
sorry
end Problem2Part2

end curve_equation_perpendicular_line_max_triangle_area_l92_92124


namespace max_min_abs_m_l92_92243

theorem max_min_abs_m {z1 z2 m : ℂ} (h1 : z1 - 4 * z2 = 16 + 20 * Complex.I)
  (h2 : ∃ α β : ℂ, (α - β).abs = 2 ∧ ∀ x : ℂ, x^2 + z1 * x + z2 + m = 0 → (x = α ∨ x = β)) :
  (∃ a b : ℝ, m = a + b * Complex.I ∧ (a - 4)^2 + (b - 5)^2 = 7^2) ∧
  (Complex.abs m = 7 + ∨ Complex.abs m = 7 -) :=
by
  sorry

end max_min_abs_m_l92_92243


namespace sum_of_variables_l92_92881

noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_of_variables (x y z : ℝ) :
  log 2 (log 3 (log 4 x)) = 0 ∧ log 3 (log 4 (log 2 y)) = 0 ∧ log 4 (log 2 (log 3 z)) = 0 →
  x + y + z = 89 :=
by
  sorry

end sum_of_variables_l92_92881


namespace pq_passes_through_fixed_point_l92_92045

noncomputable def C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1 

def ellipse_vertex_A : Prop := (0, 1)

def ellipse_vertex_B : Prop := (0, -1)

def point_M (t : ℝ) : Prop := (t, 2)

def line_MA (t : ℝ) (x y : ℝ) : Prop :=
  y = 1/t*x + 1

def line_MB (t : ℝ) (x y : ℝ) : Prop :=
  y = 3/t*x - 1

def is_on_ellipse (x y : ℝ) : Prop :=
  C x y

def intersects_ellipse_MA (P : ℝ × ℝ) (x y : ℝ) (t : ℝ) : Prop :=
  C x y ∧ line_MA t x y

def intersects_ellipse_MB (Q : ℝ × ℝ) (x y : ℝ) (t : ℝ) : Prop :=
  C x y ∧ line_MB t x y

theorem pq_passes_through_fixed_point (t : ℝ) (h_t : t ≠ 0) :
  ∀ (P Q : ℝ × ℝ), 
    intersects_ellipse_MA P.fst P.snd t →
    intersects_ellipse_MB Q.fst Q.snd t →
    ∃ k : ℝ, 
      (k * P.fst + (1 - k) * Q.fst = 0) ∧
      (k * P.snd + (1 - k) * Q.snd = 1 / 2) :=
by 
  sorry

end pq_passes_through_fixed_point_l92_92045


namespace fourth_number_on_board_eighth_number_on_board_l92_92848

theorem fourth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 4 ∧ median numbers[3] = 2 :=
sorry

theorem eighth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 8 ∧ median numbers[7] = 2 :=
sorry

end fourth_number_on_board_eighth_number_on_board_l92_92848


namespace complex_multiplication_l92_92333

theorem complex_multiplication :
  ∀ (i : ℂ), i * i = -1 → i * (1 + i) = -1 + i :=
by
  intros i hi
  sorry

end complex_multiplication_l92_92333


namespace triangle_inequality_l92_92702

variable (a b c : ℝ)
variable (S_triangle_ABC : ℝ)
variable (h_triangle : (3 * a^2 + 2 * (b^2 + c^2)) / 2 = a^2 + b^2 + c^2) -- Condition representing the triangle inequality

theorem triangle_inequality :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt(3) * S_triangle_ABC := 
sorry

end triangle_inequality_l92_92702


namespace quadratic_least_value_l92_92555

variable (a b c : ℝ)

theorem quadratic_least_value (h_a_pos : a > 0)
  (h_c_eq : ∀ x : ℝ, a * x^2 + b * x + c ≥ 9) :
  c = 9 + b^2 / (4 * a) :=
by
  sorry

end quadratic_least_value_l92_92555


namespace instantaneous_velocity_time_to_fall_l92_92312

noncomputable def h (t : ℝ) : ℝ := -4.8 * t ^ 2 + 8 * t + 10

theorem instantaneous_velocity (t : ℝ) : deriv h 2 = -11.2 :=
by
  -- Define the function h(t)
  let h : ℝ → ℝ := λ t, -4.8 * t ^ 2 + 8 * t + 10
  -- Compute the derivative of h(t)
  have h_deriv : deriv h t = -9.6 * t + 8
  sorry
  -- Evaluate the derivative at t = 2
  show deriv h 2 = -11.2
  from calc
    deriv h 2 = -9.6 * 2 + 8 := by rw h_deriv
            ... = -11.2      := by norm_num

theorem time_to_fall : ∃ t : ℝ, h t = 0 ∧ t = 2.5 :=
by
  -- Define the function h(t)
  let h : ℝ → ℝ := λ t, -4.8 * t ^ 2 + 8 * t + 10
  -- Find the roots of h(t) = 0
  have h_zero : ∃ t : ℝ, h t = 0 := by
    use 2.5
    sorry
  -- Check that the root is t = 2.5
  exact h_zero

end instantaneous_velocity_time_to_fall_l92_92312


namespace recurring_decimal_as_fraction_l92_92868

theorem recurring_decimal_as_fraction :
  0.53 + (247 / 999) * 0.001 = 53171 / 99900 :=
by
  sorry

end recurring_decimal_as_fraction_l92_92868


namespace ellipse_foci_distance_l92_92968

theorem ellipse_foci_distance (x y : ℝ) (h : 9 * x^2 + y^2 = 36) : 
  let a := 6
      b := 2
      c := Real.sqrt (a^2 - b^2)
  in 2 * c = 8 * Real.sqrt 2 :=
by
  sorry

end ellipse_foci_distance_l92_92968


namespace suitable_survey_set_l92_92230

def Survey1 := "Investigate the lifespan of a batch of light bulbs"
def Survey2 := "Investigate the household income situation in a city"
def Survey3 := "Investigate the vision of students in a class"
def Survey4 := "Investigate the efficacy of a certain drug"

-- Define what it means for a survey to be suitable for sample surveys
def suitable_for_sample_survey (survey : String) : Prop :=
  survey = Survey1 ∨ survey = Survey2 ∨ survey = Survey4

-- The question is to prove that the surveys suitable for sample surveys include exactly (1), (2), and (4).
theorem suitable_survey_set :
  {Survey1, Survey2, Survey4} = {s : String | suitable_for_sample_survey s} :=
by
  sorry

end suitable_survey_set_l92_92230


namespace sin_90_degrees_l92_92491

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92491


namespace number_of_divisors_congruent_to_1_mod_3_l92_92185

theorem number_of_divisors_congruent_to_1_mod_3 (n : ℕ) :
  finset.card ((finset.range (n + 1)).filter (λ k, (2^k % 3 = 1))) = n + 1 :=
sorry

end number_of_divisors_congruent_to_1_mod_3_l92_92185


namespace angle_value_l92_92693

theorem angle_value (P Q R S T : Type) (x : ℝ) :
  ∀ {PQR_straight : PQR}, 
  ∀ {QS_eq_QT : QS = QT}, 
  ∀ {angle_PQS : ∠ PQS = x}, 
  ∀ {angle_TQR : ∠ TQR = 3 * x},
  ∀ {angle_QTS : ∠ QTS = 76}, 
  x = 38 := sorry

end angle_value_l92_92693


namespace compute_sin_90_l92_92504

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92504


namespace variance_ξ_l92_92357

variable (P : ℕ → ℝ) (ξ : ℕ)

-- conditions
axiom P_0 : P 0 = 1 / 5
axiom P_1 : P 1 + P 2 = 4 / 5
axiom E_ξ : (0 * P 0 + 1 * P 1 + 2 * P 2) = 1

-- proof statement
theorem variance_ξ : (0 - 1)^2 * P 0 + (1 - 1)^2 * P 1 + (2 - 1)^2 * P 2 = 2 / 5 :=
by sorry

end variance_ξ_l92_92357


namespace evaluate_expression_l92_92959

-- Define the given numbers as real numbers
def x : ℝ := 175.56
def y : ℝ := 54321
def z : ℝ := 36947
def w : ℝ := 1521

-- State the theorem to be proved
theorem evaluate_expression : (x / y) * (z / w) = 0.07845 :=
by 
  -- We skip the proof here
  sorry

end evaluate_expression_l92_92959


namespace f_inv_sum_l92_92719

def f (x : ℝ) : ℝ :=
if x < 15 then x + 4 else 3 * x - 5

noncomputable def f_inv (y : ℝ) : ℝ :=
if y = 8 then 4
else if y = 64 then 23
else 0 -- This is a placeholder: f_inv is noncomputable and not fully defined here.

theorem f_inv_sum :
  f_inv 8 + f_inv 64 = 27 :=
by
  -- The proof is omitted.
  sorry

end f_inv_sum_l92_92719


namespace min_m_for_derivative_range_of_a_l92_92598

-- First problem
theorem min_m_for_derivative (f : ℝ → ℝ) (a b : ℝ) (h : ∀ x, f x = (1/3) * x^3 - (1/2) * x^2 + 2 * x) :
  ∃ m, (∀ x ∈ set.Icc a b, (∀ x ∈ set.Icc a b, deriv f x ≤ m)) :=
sorry

-- Second problem
theorem range_of_a (f g : ℝ → ℝ) (a : ℝ) (h₁ : ∀ x, f x = (1/3) * x^3 - (1/2) * x^2 + 2 * x)
  (h₂ : ∀ x, g x = (1/2) * a * x^2 - (a - 2) * x) :
  (∃ r1 r2 r3 : ℝ, r1 < r2 ∧ r2 < r3 ∧ ∀ x, (x > -1) → (f x = g x)) ↔ ( (a > -5/9 ∧ a < 1/3) ∨ (a > 3) ∧ a ≠ 0) :=
sorry

end min_m_for_derivative_range_of_a_l92_92598


namespace domain_of_g_l92_92515

def quadratic (x : ℝ) : ℝ := x^2 - 8 * x + 18
def g (x : ℝ) : ℝ := 1 / (quadratic x).floor

theorem domain_of_g :
  {x : ℝ | quadratic x ≠ 0} = {x : ℝ | x ≤ 1} ∪ {x : ℝ | x ≥ 17} :=
by
  sorry

end domain_of_g_l92_92515


namespace correct_fourth_number_correct_eighth_number_l92_92842

-- Condition: Initial number on the board and sequence of medians
def initial_board : List ℝ := [1]
def medians : List ℝ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- The number written fourth is 2
def fourth_number_written (board : List ℝ) : ℝ := 2

-- The number written eighth is also 2
def eighth_number_written (board : List ℝ) : ℝ := 2

-- Formalizing the conditions and assertions
theorem correct_fourth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    fourth_number_written board = 2 := 
sorry

theorem correct_eighth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    eighth_number_written board = 2 := 
sorry

-- Function to calculate medians from the board (to be implemented)
noncomputable def calculate_medians_from_board (board : List ℝ) : List ℝ := sorry

end correct_fourth_number_correct_eighth_number_l92_92842


namespace arc_length_eq_20_l92_92331

theorem arc_length_eq_20 :
  (∫ t in 0..π, sqrt ((5 * (1 - cos t))^2 + (5 * sin t)^2)) = 20 :=
by
  sorry

end arc_length_eq_20_l92_92331


namespace minimum_value_of_f_l92_92544

open Real

def f (x : ℝ) : ℝ := cos (3 * x) + 4 * cos (2 * x) + 8 * cos x

theorem minimum_value_of_f :
  ∃ x : ℝ, ∀ y : ℝ, f(y) ≥ f(-acos(-1)) ∧ f(x) = -5 :=
by
  sorry

end minimum_value_of_f_l92_92544


namespace fourth_number_is_two_eighth_number_is_two_l92_92836

theorem fourth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 4 ∧ board !! 3 = some 2 :=
by
  sorry

theorem eighth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 8 ∧ board !! 7 = some 2 :=
by
  sorry

end fourth_number_is_two_eighth_number_is_two_l92_92836


namespace find_multiple_l92_92310

variable (M A X : ℤ)

-- Given conditions
def cond1 := M = 28
def cond2 := M - 3 = 3 * (A - 3) + 1
def cond3 := M + 4 = X * (A + 4) + 2

theorem find_multiple (h1 : cond1) (h2 : cond2) (h3 : cond3) : X = 2 := 
by
  sorry

end find_multiple_l92_92310


namespace y_intercept_of_tangent_line_l92_92017

-- Define the function of the curve
def curve (x : ℝ) : ℝ := x^3 + 11

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, 12)

-- State the theorem for the y-intercept of the tangent line
theorem y_intercept_of_tangent_line : 
  let slope := 3 * (point_of_tangency.1)^2 in
  let tangent_line (x : ℝ) := slope * (x - point_of_tangency.1) + point_of_tangency.2 in
  tangent_line 0 = 9 :=
by
  let slope := 3 * 1^2
  let tangent_line (x : ℝ) := slope * (x - 1) + 12
  show tangent_line 0 = 9
  sorry

end y_intercept_of_tangent_line_l92_92017


namespace inequality_of_powers_l92_92773

theorem inequality_of_powers (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  a^b * b^c * c^d * d^a ≥ b^a * c^b * d^c * a^d := 
sorry

end inequality_of_powers_l92_92773


namespace elevation_after_descend_l92_92162

theorem elevation_after_descend (initial_elevation : ℕ) (rate : ℕ) (time : ℕ) (final_elevation : ℕ) 
  (h_initial : initial_elevation = 400) 
  (h_rate : rate = 10) 
  (h_time : time = 5) 
  (h_final : final_elevation = initial_elevation - rate * time) : 
  final_elevation = 350 := 
by 
  sorry

end elevation_after_descend_l92_92162


namespace sin_value_l92_92563

open Real

-- Define the given conditions
variables (x : ℝ) (h1 : cos (π + x) = 3 / 5) (h2 : π < x) (h3 : x < 2 * π)

-- State the problem to be proved
theorem sin_value : sin x = - 4 / 5 :=
by
  sorry

end sin_value_l92_92563


namespace find_a7_l92_92701

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = -4/3 ∧ (∀ n, a (n + 2) = 1 / (a n + 1))

theorem find_a7 (a : ℕ → ℚ) (h : seq a) : a 7 = 2 :=
by
  sorry

end find_a7_l92_92701


namespace possible_values_of_k_l92_92766

theorem possible_values_of_k (n : ℕ) (h : n ≥ 3) :
  ∃ t : ℕ, k = 2 ^ t ∧ 2 ^ t ≥ n :=
sorry

end possible_values_of_k_l92_92766


namespace sum_is_220_l92_92286

def second_number := 60
def first_number := 2 * second_number
def third_number := first_number / 3
def sum_of_numbers := first_number + second_number + third_number

theorem sum_is_220 : sum_of_numbers = 220 :=
by
  sorry

end sum_is_220_l92_92286


namespace slope_angle_tangent_line_at_point_l92_92247

theorem slope_angle_tangent_line_at_point :
  let curve (x : ℝ) := (1/3) * x^3 - 2
  let point_of_tangency := (-1, -7/3 : ℝ)
  let slope (x : ℝ) := x^2
  curve point_of_tangency.1 = point_of_tangency.2 →
  ∃ θ : ℝ, θ = 45 ∧ slope point_of_tangency.1 = 1 :=
by
  intros curve point_of_tangency slope h_curve
  sorry

end slope_angle_tangent_line_at_point_l92_92247


namespace problem1_l92_92336

-- Define the given conditions
def tan_theta : ℝ := -3/4
def cos_sq_theta : ℝ := ((1 : ℝ) / (1 + tan_theta^2)) -- cos^2θ from tanθ
def sin_theta : ℝ := real.sqrt (1 - cos_sq_theta)  -- sinθ using cos^2θ + sin^2θ = 1

-- Statement to be proved
theorem problem1 (θ : ℝ) (h₁ : real.tan θ = tan_theta) : 
  1 + real.sin θ * real.cos θ - (real.cos θ ^ 2) = -3 / 25 :=
sorry

end problem1_l92_92336


namespace equal_costs_at_60_guests_l92_92218

def caesars_cost (x : ℕ) : ℕ := 800 + 30 * x
def venus_cost (x : ℕ) : ℕ := 500 + 35 * x

theorem equal_costs_at_60_guests : 
  ∃ x : ℕ, caesars_cost x = venus_cost x ∧ x = 60 := 
by
  existsi 60
  unfold caesars_cost venus_cost
  split
  . sorry
  . refl

end equal_costs_at_60_guests_l92_92218


namespace lines_perpendicular_l92_92527

variable (b : ℝ)

/-- Proof that if the given lines are perpendicular, then b must be 3 -/
theorem lines_perpendicular (h : b ≠ 0) :
    let l₁_slope := -3
    let l₂_slope := b / 9
    l₁_slope * l₂_slope = -1 → b = 3 :=
by
  intros slope_prod
  simp only [h]
  sorry

end lines_perpendicular_l92_92527


namespace mono_sum_eq_five_l92_92099

-- Conditions
def term1 (x y : ℝ) (m : ℕ) : ℝ := x^2 * y^m
def term2 (x y : ℝ) (n : ℕ) : ℝ := x^n * y^3

def is_monomial_sum (x y : ℝ) (m n : ℕ) : Prop :=
  term1 x y m + term2 x y n = x^(2:ℕ) * y^(3:ℕ)

-- Theorem stating the result
theorem mono_sum_eq_five (x y : ℝ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
by
  sorry

end mono_sum_eq_five_l92_92099


namespace sin_90_degree_l92_92494

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92494


namespace fourth_number_on_board_eighth_number_on_board_l92_92846

theorem fourth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 4 ∧ median numbers[3] = 2 :=
sorry

theorem eighth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 8 ∧ median numbers[7] = 2 :=
sorry

end fourth_number_on_board_eighth_number_on_board_l92_92846


namespace M_inter_N_eq_N_l92_92640

def M : set ℝ := {y : ℝ | ∃ x : ℝ, 0 < x ∧ y = x^2}
def N : set ℝ := {y : ℝ | ∃ x : ℝ, 0 < x ∧ y = x + 2}

theorem M_inter_N_eq_N : M ∩ N = N := 
  by sorry

end M_inter_N_eq_N_l92_92640


namespace problem_statement_l92_92170

-- Define the vectors
def u : ℝ × ℝ × ℝ := (4, -2, -1)
def v : ℝ × ℝ × ℝ := (-2, -3, 6)
def w : ℝ × ℝ × ℝ := (3, 7, 0)

-- Define vector subtraction
def vector_sub (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.1 - b.1, a.2 - b.2, a.3 - b.3)

-- Define cross product
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2 * b.3 - a.3 * b.2, a.3 * b.1 - a.1 * b.3, a.1 * b.2 - a.2 * b.1)

-- Define dot product
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Prove the statement
theorem problem_statement : 
  let u_v := vector_sub u v in
  let v_w := vector_sub v w in
  let w_u := vector_sub w u in
  dot_product u_v (cross_product v_w w_u) = 8 :=
by 
  let u_v := vector_sub u v
  let v_w := vector_sub v w
  let w_u := vector_sub w u
  have h1 : u_v = (6, 1, -7) := rfl
  have h2 : v_w = (-5, -10, 6) := rfl
  have h3 : w_u = (-1, 9, 1) := rfl
  have cross := cross_product v_w w_u
  have dot := dot_product u_v cross
  -- Just checking that the calculation holds
  show dot = 8 from sorry

end problem_statement_l92_92170


namespace compute_sin_90_l92_92514

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92514


namespace only_one_critical_point_interval_l92_92670

noncomputable def f (a : ℕ) (x : ℝ) : ℝ :=
  Real.log x + a / (x + 1)

noncomputable def f_deriv (a : ℕ) (x : ℝ) : ℝ :=
  1 / x - a / (x + 1)^2

theorem only_one_critical_point_interval (a : ℕ) :
  (∃ c ∈ set.Ioo (1:ℝ) 3, f_deriv a c = 0 ∧ 
  ∀ x ∈ set.Ioo (1:ℝ) 3, (x ≠ c → f_deriv a x ≠ 0)) ↔ a = 5 :=
by sorry

end only_one_critical_point_interval_l92_92670


namespace twodot_product_neg4_l92_92564

def vec3 := (ℝ × ℝ × ℝ)

def dot_product (v w : vec3) : ℝ :=
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

noncomputable def twodot {a b : vec3} (ha : a = (1, -2, 1)) (hb : b = (-1, 2, 3)) : ℝ :=
  2 * dot_product a b

theorem twodot_product_neg4 {a b : vec3} (ha : a = (1, -2, 1)) (hb : b = (-1, 2, 3)) :
  twodot ha hb = -4 :=
sorry

end twodot_product_neg4_l92_92564


namespace cars_return_to_starting_point_l92_92820

-- Definitions
def car := ℕ  -- Representing each car by a natural number
def n_cars (n : ℕ) : Type := fin n  -- n cars represented by fin n

-- The main theorem statement
theorem cars_return_to_starting_point (n : ℕ) :
  ∃ t : ℕ, ∀ (c : n_cars n), car_at_time t c = car_at_time 0 c := 
sorry

end cars_return_to_starting_point_l92_92820


namespace minimum_value_of_linear_combination_l92_92188

noncomputable def vector1 : ℝ × ℝ := (1, 5)
noncomputable def vector2 : ℝ × ℝ := (4, -1)
noncomputable def vector3 : ℝ × ℝ := (2, 1)

theorem minimum_value_of_linear_combination 
  (λ1 λ2 λ3 : ℝ) 
  (h_nonneg : λ1 ≥ 0 ∧ λ2 ≥ 0 ∧ λ3 ≥ 0) 
  (h_cond : λ1 + λ2 / 2 + λ3 / 3 = 1) : 
  ∥(λ1 • vector1 + λ2 • vector2 + λ3 • vector3)∥ = 3 * Real.sqrt 2 :=
sorry

end minimum_value_of_linear_combination_l92_92188


namespace value_of_m_l92_92620

theorem value_of_m (a m : ℝ) (h : a > 0) (hm : m ≠ 0) :
  (P : ℝ × ℝ) (P = (m, 3))
  (H : ∀ x : ℝ, -a * x^2 + 2 * a * x + 3 = 3 → x = 0 ∨ x = 2) :
  m = 2 :=
by
  sorry

end value_of_m_l92_92620


namespace garbage_bill_problem_l92_92755

theorem garbage_bill_problem
  (R : ℝ)
  (trash_bins : ℝ := 2)
  (recycling_bins : ℝ := 1)
  (weekly_trash_cost_per_bin : ℝ := 10)
  (weeks_per_month : ℝ := 4)
  (discount_rate : ℝ := 0.18)
  (fine : ℝ := 20)
  (final_bill : ℝ := 102) :
  (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  - discount_rate * (trash_bins * weekly_trash_cost_per_bin * weeks_per_month + recycling_bins * R * weeks_per_month)
  + fine = final_bill →
  R = 5 := 
by
  sorry

end garbage_bill_problem_l92_92755


namespace parabola_focus_vertex_ratio_l92_92745

theorem parabola_focus_vertex_ratio :
  let (V1 : ℝ × ℝ) := (0, 0) in
  let (F1 : ℝ × ℝ) := (0, 1/4) in
  ∀ (a b : ℝ),
  a ≠ b ∧ a * b = -1 →
  let Mid : ℝ × ℝ := ((a + b) / 2, ((a + b)^2 + 2) / 2) in
  let (V2 : ℝ × ℝ) := (0, 1) in
  let (F2 : ℝ × ℝ) := (0, 9 / 8) in
  let dV := (1 : ℝ) in
  let dF := (7 / 8 : ℝ) in
  dF / dV = 7 / 8 :=
begin
  intros V1 F1 a b H Mid V2 F2 dV dF,
  sorry,  -- proof would go here
end

end parabola_focus_vertex_ratio_l92_92745


namespace no_values_of_a_divisible_by_3_l92_92173

theorem no_values_of_a_divisible_by_3 (a : ℤ) (h1 : 1 ≤ a) (h2 : a ≤ 251) :
    let d1 := a^3 + 3^a + a * 3^((a + 1) / 2)
    let d2 := a^3 + 3^a - a * 3^((a + 1) / 2)
    nat.gcd (d1 * d2) 3 ≠ 0 := by
  sorry

end no_values_of_a_divisible_by_3_l92_92173


namespace find_m_l92_92611

variable (a m x : ℝ)

noncomputable def quadratic_function : ℝ → ℝ := λ x, -a * x^2 + 2 * a * x + 3

theorem find_m (h1 : a > 0) (h2 : quadratic_function a m = 3) (h3 : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92611


namespace trigonometric_identity_l92_92584

theorem trigonometric_identity
  (h1 : cos (x - π / 4) = sqrt 2 / 10)
  (h2 : π / 2 < x ∧ x < 3 * π / 4) :
  sin x = 4 / 5 ∧ sin (2 * x - π / 6) = (7 - 24 * sqrt 3) / 50 := by
  sorry

end trigonometric_identity_l92_92584


namespace sqrt_sum_eq_two_l92_92531

theorem sqrt_sum_eq_two : 
  sqrt (16 - 8 * sqrt 3) + sqrt (16 + 8 * sqrt 3) = 2 :=
sorry

end sqrt_sum_eq_two_l92_92531


namespace shorter_piece_length_l92_92886

/-- A 69-inch board is cut into 2 pieces. One piece is 2 times the length of the other.
    Prove that the length of the shorter piece is 23 inches. -/
theorem shorter_piece_length (x : ℝ) :
  let shorter := x
  let longer := 2 * x
  (shorter + longer = 69) → shorter = 23 :=
by
  intro h
  sorry

end shorter_piece_length_l92_92886


namespace problem_statement_l92_92673

noncomputable def range_of_a2_b2 (a b c : ℝ) (C : ℝ) : Set ℝ :=
  {x : ℝ | c = Real.sqrt 3 ∧ C = Real.pi / 3 ∧ x = a^2 + b^2 ∧ 3 < x ∧ x ≤ 6}

theorem problem_statement (a b c A B C : ℝ) 
  (h1 : c = Real.sqrt 3)
  (h2 : tan C = (Real.sin A + Real.sin B) / (Real.cos A + Real.cos B)) :
  C = Real.pi / 3 ∧ (range_of_a2_b2 a b c C).nonempty := by
  sorry

end problem_statement_l92_92673


namespace interpretation_of_k5_3_l92_92659

theorem interpretation_of_k5_3 (k : ℕ) (hk : 0 < k) : (k^5)^3 = k^5 * k^5 * k^5 :=
by sorry

end interpretation_of_k5_3_l92_92659


namespace best_play_majority_two_classes_l92_92146

theorem best_play_majority_two_classes (n : ℕ) :
  let prob_win := 1 - (1/2) ^ n
  in prob_win = 1 - (1/2) ^ n :=
by
  sorry

end best_play_majority_two_classes_l92_92146


namespace expand_binomials_l92_92537

variable {x y : ℝ}

theorem expand_binomials (x y : ℝ) : 
  (x + 5) * (3 * y + 15) = 3 * x * y + 15 * x + 15 * y + 75 := 
by
  sorry

end expand_binomials_l92_92537


namespace int_values_satisfy_condition_l92_92255

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l92_92255


namespace sin_ninety_deg_l92_92433

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92433


namespace find_m_eq_2_l92_92619

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l92_92619


namespace best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92149

-- Part (a)
theorem best_play_wins_majority_two_classes (n : ℕ) :
  let prob_tie := (1 / 2) ^ n in
  1 - prob_tie = 1 - (1 / 2) ^ n :=
sorry

-- Part (b)
theorem best_play_wins_majority_multiple_classes (n s : ℕ) :
  let prob_tie := (1 / 2) ^ ((s - 1) * n) in
  1 - prob_tie = 1 - (1 / 2) ^ ((s - 1) * n) :=
sorry

end best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92149


namespace num_green_cards_l92_92301

theorem num_green_cards (total_cards : ℕ) 
  (red_fraction : ℚ) (black_fraction : ℚ)
  (h1 : total_cards = 2160)
  (h2 : red_fraction = 7/12)
  (h3 : black_fraction = 11/19) :
  let red_cards := (red_fraction * total_cards : ℚ).toNat,
      non_red_cards := total_cards - red_cards,
      black_cards := (black_fraction * non_red_cards : ℚ).toNat,
      green_cards := total_cards - (red_cards + black_cards)
  in green_cards = 379 :=
by
  sorry

end num_green_cards_l92_92301


namespace trigonometric_identity_tan_22_5_l92_92373

theorem trigonometric_identity_tan_22_5 :
  let θ := real.pi / 8 in
  (real.tan θ) / (1 - (real.tan θ)^2) = 1 / 2 :=
by
  sorry

end trigonometric_identity_tan_22_5_l92_92373


namespace yield_percentage_l92_92885

theorem yield_percentage (d : ℝ) (q : ℝ) (f : ℝ) : d = 12 → q = 150 → f = 100 → (d * f / q) * 100 = 8 :=
by
  intros h_d h_q h_f
  rw [h_d, h_q, h_f]
  sorry

end yield_percentage_l92_92885


namespace ellipse_foci_distance_l92_92966

theorem ellipse_foci_distance (x y : ℝ) (h : 9 * x^2 + y^2 = 36) : 
  let a := 6
      b := 2
      c := Real.sqrt (a^2 - b^2)
  in 2 * c = 8 * Real.sqrt 2 :=
by
  sorry

end ellipse_foci_distance_l92_92966


namespace BF_parallel_AC_l92_92723

variables {A B C D K L E F : Point}
variables [parallelogram ABCD]
variables (angle_B_obtuse : obtuse_angle ∠ABC)
variables (AD_gt_AB : AD > AB)
variables (on_line_AC_K : on_line AC K)
variables (on_line_AC_L : on_line AC L)
variables (K_between_A_and_L : between_points A K L)
variables (angle_ADL_eq_angle_KBA : ∠ADL = ∠KBA)
variables (BK_intersects_omega_at_B_and_E : intersects_at_two_points (line_through B K) ω B E)
variables (EL_intersects_omega_at_E_and_F : intersects_at_two_points (line_through E L) ω E F)

theorem BF_parallel_AC :
  parallel (line_through B F) (line_through A C) :=
sorry

end BF_parallel_AC_l92_92723


namespace find_x_l92_92706

-- Define the condition variables
variables (y z x : ℝ) (Y Z X : ℝ)
-- Primary conditions given in the problem
variable (h_y : y = 7)
variable (h_z : z = 6)
variable (h_cosYZ : Real.cos (Y - Z) = 15 / 16)

-- The main theorem to prove
theorem find_x (h_y : y = 7) (h_z : z = 6) (h_cosYZ : Real.cos (Y - Z) = 15 / 16) :
  x = Real.sqrt 22 :=
sorry

end find_x_l92_92706


namespace maximum_fly_path_length_l92_92897

theorem maximum_fly_path_length (side_length : ℝ) (h : side_length = 1) : 
  ∃ L, L = 4 * Real.sqrt 3 + 4 * Real.sqrt 2 :=
by
  use 4 * Real.sqrt 3 + 4 * Real.sqrt 2
  exact h.symm ▸ rfl

end maximum_fly_path_length_l92_92897


namespace sin_90_eq_one_l92_92411

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92411


namespace best_play_wins_majority_l92_92128

/-- Probability that the best play wins with a majority of the votes given the conditions -/
theorem best_play_wins_majority (n : ℕ) :
  let p := 1 - (1 / 2)^n
  in p > (1 - (1 / 2)^n) ∧ p ≤ 1 :=
sorry

end best_play_wins_majority_l92_92128


namespace solve_inequality_l92_92010

open Set

def g (x : ℝ) : ℝ := (3 * x - 9) * (2 * x - 5) / (2 * x + 4)

theorem solve_inequality : 
  ∀ x : ℝ, g x ≥ 0 ↔ x ∈ (Iio (-2) ∪ Ioc (-2) (5 / 2) ∪ Ici 3) :=
by
  sorry

end solve_inequality_l92_92010


namespace correct_fourth_number_correct_eighth_number_l92_92841

-- Condition: Initial number on the board and sequence of medians
def initial_board : List ℝ := [1]
def medians : List ℝ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- The number written fourth is 2
def fourth_number_written (board : List ℝ) : ℝ := 2

-- The number written eighth is also 2
def eighth_number_written (board : List ℝ) : ℝ := 2

-- Formalizing the conditions and assertions
theorem correct_fourth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    fourth_number_written board = 2 := 
sorry

theorem correct_eighth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    eighth_number_written board = 2 := 
sorry

-- Function to calculate medians from the board (to be implemented)
noncomputable def calculate_medians_from_board (board : List ℝ) : List ℝ := sorry

end correct_fourth_number_correct_eighth_number_l92_92841


namespace sin_90_eq_one_l92_92425

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92425


namespace inequality_proof_l92_92727

theorem inequality_proof (n : ℕ) (x : ℕ → ℝ) 
  (hn : n ≥ 2) 
  (hx_pos : ∀ j, j < n → x j > -1) 
  (hx_sum : ∑ j in Finset.range n, x j = n) : 
  (∑ j in Finset.range n, (1 / (1 + x j))) ≥ (∑ j in Finset.range n, (x j / (1 + (x j)^2))) :=
begin
  sorry
end

end inequality_proof_l92_92727


namespace trace_coincide_l92_92371

variables (S2II S2I S2IV : Type) [Point S2II] [Point S2I] [Point S2IV]
variables (s1 s4 : Line) (x14 : Line) (alpha1 : Angle)

-- Conditions
axiom dist_eq : dist S2II S2I = dist S2IV S2I
axiom S2IV_on_s1 : S2IV ∈ s1
axiom x14_perpendicular : perpendicular x14 (segment S2IV S2I)

-- Theorem statement
theorem trace_coincide : 
  if alpha1 > 45 then ∃ x14, x14 ⊥ (segment S2IV S2I) ∧ s4 = s1 ∧ 2 solutions
  else if alpha1 = 45 then ∃! x14, x14 ⊥ (segment S2IV S2I) ∧ s4 = s1 ∧ 1 solution
  else ¬ (∃ x14, x14 ⊥ (segment S2IV S2I) ∧ s4 = s1) :=
sorry

end trace_coincide_l92_92371


namespace smallest_composite_no_prime_factors_less_than_20_l92_92983

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92983


namespace find_standard_eq_find_min_MP_l92_92044

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

theorem find_standard_eq (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ellipse_eq (sqrt 2) 1)
  (h4 : ∀ (x : ℝ), x^2 / 4 + 1 / 2 = 1 → a^2 = 2 * b^2) :
  ellipse_eq x y :=
begin
  sorry
end

theorem find_min_MP (p x y : ℝ) (hx : |x| ≤ 2)
  (h1 : ellipse_eq x y) :

  (|p| ≤ 1 → ∃ (x_val y_val : ℝ), (x_val, y_val) ∈ ellipse_eq ∧ min_val = sqrt (2 - p^2) ∧ x = 2 * p) ∧
  (p > 1 → ∃ (x_val y_val : ℝ), (x_val, y_val) ∈ ellipse_eq ∧ min_val = |p - 2| ∧ x = 2) ∧
  (p < -1 → ∃ (x_val y_val : ℝ), (x_val, y_val) ∈ ellipse_eq ∧ min_val = |p + 2| ∧ x = -2) :=
begin
  sorry
end

end find_standard_eq_find_min_MP_l92_92044


namespace problem_part1_problem_part2_l92_92600

def f (x : ℝ) : ℝ := abs (x - 2) - abs (x - 4)

theorem problem_part1 (x : ℝ) : f x < 0 ↔ x < 3 := 
by sorry

theorem problem_part2 (m : ℝ) (h : g (m, f, ℝ) ≠ 0) :
    (∃ x, g x = 1 / (m - f x)) ↔ m ∈ set.Ioo (-∞ : ℝ) (-2 : ℝ) ∪ set.Ioo (2 : ℝ) (∞ : ℝ) := 
by sorry

end problem_part1_problem_part2_l92_92600


namespace symmetric_midpoint_l92_92550

theorem symmetric_midpoint (m n : ℝ) :
  (n = (4 + 6) / 2) ∧ (-3 = (m + (-9)) / 2) → m = 3 ∧ n = 5 :=
by
  intro h
  cases h with hn hm
  rw [← hn, ← hm]
  -- Calculation steps should use basic arithmetic libraries
  have hn : n = 5 := by norm_num
  have hm : m = 3 := by
    linarith
  exact ⟨hm, hn⟩

end symmetric_midpoint_l92_92550


namespace least_area_of_prime_dim_l92_92909

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_area_of_prime_dim (l w : ℕ) (h_perimeter : 2 * (l + w) = 120)
    (h_integer_dims : l > 0 ∧ w > 0) (h_prime_dim : is_prime l ∨ is_prime w) :
    l * w = 116 :=
sorry

end least_area_of_prime_dim_l92_92909


namespace logw_u_value_l92_92066

noncomputable def logw_u (u v w : ℝ) : ℝ :=
  log w u

theorem logw_u_value (u v w : ℝ) (h1 : u ≠ 1) (h2 : v ≠ 1) (h3 : w ≠ 1)
  (h4 : u > 0) (h5 : v > 0) (h6 : w > 0)
  (h7 : log u (v * w) + log v w = 5)
  (h8 : log v u + log w v = 3) :
  logw_u u v w = 4 / 5 := sorry

end logw_u_value_l92_92066


namespace quadratic_roots_l92_92802

theorem quadratic_roots (d : ℝ) (h : ∀ x, x^2 + 9*x + d = 0 → x = (-9 + real.sqrt d) / 2 ∨ x = (-9 - real.sqrt d) / 2) :
  d = 16.2 :=
  sorry

end quadratic_roots_l92_92802


namespace sin_90_eq_1_l92_92447

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92447


namespace M_minus_m_l92_92176

variable {x y : ℂ} (hx : x ≠ 0) (hy : y ≠ 0)

theorem M_minus_m : 
  let expr := complex.abs (x + y) / (complex.abs x + complex.abs y)
  let M := 1
  let m := 0
  M - m = 1 :=
by
  let expr := complex.abs (x + y) / (complex.abs x + complex.abs y)
  have M : expr ≤ 1 := by sorry  -- upper bound proof here
  have m : expr ≥ 0 := by sorry  -- lower bound proof here
  show 1 - 0 = 1 from by simp
  exactly one_sub_zero

end M_minus_m_l92_92176


namespace smallest_composite_no_prime_factors_less_than_20_l92_92989

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92989


namespace max_cables_191_l92_92928

/-- 
  There are 30 employees: 20 with brand A computers and 10 with brand B computers.
  Cables can only connect a brand A computer to a brand B computer.
  Employees can communicate with each other if their computers are directly connected by a cable 
  or by relaying messages through a series of connected computers.
  The maximum possible number of cables used to ensure every employee can communicate with each other
  is 191.
-/
theorem max_cables_191 (A B : ℕ) (hA : A = 20) (hB : B = 10) : 
  ∃ (max_cables : ℕ), max_cables = 191 ∧ 
  (∀ (i j : ℕ), (i ≤ A ∧ j ≤ B) → (i = A ∨ j = B) → i * j ≤ max_cables) := 
sorry

end max_cables_191_l92_92928


namespace goal1_goal2_l92_92084

variables (a b : ℝ)

-- Define A and B
def A : ℝ := 2 * a^2 + a * b - 2 * b - 1
def B : ℝ := -a^2 + a * b - 2

-- Goal 1: Prove that 3A - (2A - 2B) = 3ab - 2b - 5
theorem goal1 : 3 * A - (2 * A - 2 * B) = 3 * a * b - 2 * b - 5 := 
by 
  sorry

-- Goal 2: Prove that if A + 2B is a constant for any b, then a = 2/3
theorem goal2 
  (h : ∀ b : ℝ, A + 2 * B = some_constant) : a = 2 / 3 := 
by 
  sorry

end goal1_goal2_l92_92084


namespace determine_m_l92_92635

theorem determine_m (a m : ℝ) (h : a > 0) (h2 : (m, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = -a * p.1 ^ 2 + 2 * a * p.1 + 3)) (h3 : m ≠ 0) : m = 2 :=
sorry

end determine_m_l92_92635


namespace time_to_pass_tree_l92_92326

noncomputable def length_of_train : ℝ := 630 -- Length of the train in meters
noncomputable def speed_of_train_km_hr : ℝ := 63 -- Speed of the train in km/hr
noncomputable def speed_of_train_m_s : ℝ := speed_of_train_km_hr * (1000 / 3600) -- Convert speed to m/s

theorem time_to_pass_tree : (length_of_train / speed_of_train_m_s) = 36 := 
by 
  have speed_conv: speed_of_train_m_s = 17.5 := 
    by norm_num1
  rw speed_conv
  norm_num1 -- This performs the calculation 630 / 17.5 = 36
  sorry -- Skip the actual proof steps for conversion

end time_to_pass_tree_l92_92326


namespace sin_90_eq_one_l92_92426

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92426


namespace min_distance_run_l92_92684

-- Define constants for positions and distances
def pointA : ℝ := 0
def pointB : ℝ := 1000
def wallLength : ℝ := 1000
def distanceFromAToWallStart : ℝ := 400
def distanceFromWallEndToB : ℝ := 600

-- Define the point where runner touches the wall
variable (C : ℝ) (hC : 0 ≤ C ∧ C ≤ wallLength)

-- Reflect point B across the wall to get B'
def pointBP : ℝ := 1000 + wallLength

-- Pythagorean theorem to calculate the distance
def distance : ℝ := Real.sqrt ( (distanceFromAToWallStart + distanceFromWallEndToB) ^ 2 + wallLength ^ 2 )

-- Statement to prove the minimum distance is 1414 meters
theorem min_distance_run : Real.ceil distance = 1414 := by
  sorry

end min_distance_run_l92_92684


namespace sin_90_eq_one_l92_92413

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92413


namespace area_of_quadrilateral_integer_l92_92694

theorem area_of_quadrilateral_integer (AB CD : ℕ) (hAB_perp_BC : AB ⊥ BC) (hBC_perp_CD : BC ⊥ CD) 
(hBC_tangent_circle : is_tangent BC (circle O (AD / 2)))
(choices : (AB, CD) ∈ [(4, 2), (8, 2), (12, 3), (16, 4), (20, 5)]) :
  (∃ (x : ℕ), (AB * CD = x^2)) :=
sorry

end area_of_quadrilateral_integer_l92_92694


namespace smallest_composite_no_prime_factors_less_than_20_l92_92990

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92990


namespace pennyWiseCheaperBy500Cents_l92_92832

noncomputable def cameraPrice : ℝ := 49.99
noncomputable def lensPrice : ℝ := 64.99
noncomputable def superSaversCameraDiscount : ℝ := 0.18
noncomputable def superSaversLensDiscount : ℝ := 0.20
noncomputable def pennyWiseCameraDiscount : ℝ := 12
noncomputable def pennyWiseLensDiscount : ℝ := 15

def superSaversCameraPrice := (1 - superSaversCameraDiscount) * cameraPrice
def pennyWiseCameraPrice := cameraPrice - pennyWiseCameraDiscount
def cameraPriceDifference := superSaversCameraPrice - pennyWiseCameraPrice

def superSaversLensPrice := (1 - superSaversLensDiscount) * lensPrice
def pennyWiseLensPrice := lensPrice - pennyWiseLensDiscount
def lensPriceDifference := superSaversLensPrice - pennyWiseLensPrice

def totalPriceDifference := 100 * (cameraPriceDifference + lensPriceDifference)

theorem pennyWiseCheaperBy500Cents :
  totalPriceDifference = 500 := by
  sorry

end pennyWiseCheaperBy500Cents_l92_92832


namespace sin_90_degrees_l92_92485

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92485


namespace tangent_line_eq_l92_92229

theorem tangent_line_eq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) : 
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_eq_l92_92229


namespace exponent_identity_l92_92090

theorem exponent_identity (x : ℕ) (h : x = 243) : x^6 = 3^12 * 3^18 :=
by {
  rw h,
  sorry
}

end exponent_identity_l92_92090


namespace find_root_and_coefficient_l92_92574

theorem find_root_and_coefficient (m: ℝ) (x: ℝ) (h₁: x ^ 2 - m * x - 6 = 0) (h₂: x = 3) :
  (x = 3 ∧ -2 = -6 / 3 ∨ m = 1) :=
by
  sorry

end find_root_and_coefficient_l92_92574


namespace best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92150

-- Part (a)
theorem best_play_wins_majority_two_classes (n : ℕ) :
  let prob_tie := (1 / 2) ^ n in
  1 - prob_tie = 1 - (1 / 2) ^ n :=
sorry

-- Part (b)
theorem best_play_wins_majority_multiple_classes (n s : ℕ) :
  let prob_tie := (1 / 2) ^ ((s - 1) * n) in
  1 - prob_tie = 1 - (1 / 2) ^ ((s - 1) * n) :=
sorry

end best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92150


namespace find_m_eq_2_l92_92615

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l92_92615


namespace fourth_number_is_2_eighth_number_is_2_l92_92859

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end fourth_number_is_2_eighth_number_is_2_l92_92859


namespace simplify_and_evaluate_expression_l92_92780

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 4) :
  (1 / (x + 2) + 1) / ((x^2 + 6 * x + 9) / (x^2 - 4)) = 2 / 7 :=
by
  sorry

end simplify_and_evaluate_expression_l92_92780


namespace sqrt_inequality_l92_92547

theorem sqrt_inequality (x : ℝ) : abs ((x^2 - 9) / 3) < 3 ↔ -Real.sqrt 18 < x ∧ x < Real.sqrt 18 :=
by
  sorry

end sqrt_inequality_l92_92547


namespace banana_orange_equivalence_l92_92219

/-- Given that 3/4 of 12 bananas are worth 9 oranges,
    prove that 1/3 of 9 bananas are worth 3 oranges. -/
theorem banana_orange_equivalence :
  (3 / 4) * 12 = 9 → (1 / 3) * 9 = 3 :=
by
  intro h
  have h1 : (9 : ℝ) = 9 := by sorry -- This is from the provided condition
  have h2 : 1 * 9 = 1 * 9 := by sorry -- Deducing from h1: 9 = 9
  have h3 : 9 = 9 := by sorry -- concluding 9 bananas = 9 oranges
  have h4 : (1 / 3) * 9 = 3 := by sorry -- 1/3 of 9
  exact h4

end banana_orange_equivalence_l92_92219


namespace set_contains_negative_l92_92577

theorem set_contains_negative (A : set ℝ) (n : ℕ) (h1 : A.finite) (h2 : A.card < n) (h3 : n ≥ 2)
  (h4 : ∀ k ∈ list.range n, ∃ l : list ℝ, (∀ x ∈ l, x ∈ A ∧ x ≠ y ∈ l → x ≠ y) ∧ (l.sum = 2^k) ) : 
  ∃ x ∈ A, x < 0 :=
begin
  sorry
end

end set_contains_negative_l92_92577


namespace sin_90_eq_one_l92_92421

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92421


namespace construct_K2_arrows_from_K_l92_92339

-- We define the states and transitions for a given chain K
constant State : Type
constant E1 E2 : State
constant arrow : State → State → Prop

-- Given transition arrows as per Figure 76
axiom trans1 : arrow E1 E1
axiom trans2 : arrow E1 E2
axiom trans3 : arrow E2 E1
axiom trans4 : arrow E2 E2

-- Define the product chain K^2 as pairs of states in K
def StateK2 := State × State

-- Define the transition arrows for K^2
def arrowK2 (s1 s2 : StateK2) : Prop :=
  arrow s1.1 s2.1 ∧ arrow s1.2 s2.2

-- Main theorem statement for part (a)
theorem construct_K2_arrows_from_K :
  ∀ s1 s2 : StateK2, 
  (arrow s1.1 s2.1 ∧ arrow s1.2 s2.2) ↔ arrowK2 s1 s2 :=
by {
  intros,
  simp [arrowK2],
  sorry
}

end construct_K2_arrows_from_K_l92_92339


namespace best_play_majority_win_probability_l92_92136

theorem best_play_majority_win_probability (n : ℕ) :
  (1 - (1 / 2) ^ n) = probability_best_play_wins_majority n :=
sorry

end best_play_majority_win_probability_l92_92136


namespace gratuities_correct_l92_92916

def cost_of_striploin : ℝ := 80
def cost_of_wine : ℝ := 10
def sales_tax_rate : ℝ := 0.10
def total_bill_with_gratuities : ℝ := 140

def total_bill_before_tax : ℝ := cost_of_striploin + cost_of_wine := by sorry

def sales_tax : ℝ := sales_tax_rate * total_bill_before_tax := by sorry

def total_bill_with_tax : ℝ := total_bill_before_tax + sales_tax := by sorry

def gratuities : ℝ := total_bill_with_gratuities - total_bill_with_tax := by sorry

theorem gratuities_correct : gratuities = 41 := by sorry

end gratuities_correct_l92_92916


namespace orthographic_projection_area_l92_92046

-- Define the original side length and area of the equilateral triangle
def side_length : ℝ := 1
def area_of_equilateral_triangle (a : ℝ) : ℝ := (sqrt 3 / 4) * (a ^ 2)

-- Area of the orthographic projection according to the given condition
def area_of_projection (S : ℝ) : ℝ := (sqrt 2 / 4) * S

theorem orthographic_projection_area :
  area_of_projection (area_of_equilateral_triangle side_length) = sqrt 6 / 16 :=
by
  sorry

end orthographic_projection_area_l92_92046


namespace no_solutions_l92_92220

def f (x : ℕ) (y : ℕ) : ℤ := 5 * x + 60 * (y - 1970) - 4

theorem no_solutions : ∀ (x : ℕ) (y : ℕ), 
  1 ≤ x ∧ x ≤ 12 ∧ 1970 ≤ y ∧ y ≤ 1989 → f(x, y) ≠ y :=
by 
  sorry

end no_solutions_l92_92220


namespace sum_of_distances_l92_92695

def point := ℝ × ℝ

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def A : point := (15, 0)
noncomputable def B : point := (0, 3)
noncomputable def D : point := (9, 7)

noncomputable def AD := distance A D
noncomputable def BD := distance B D

theorem sum_of_distances : AD + BD = real.sqrt 97 + real.sqrt 85 :=
by
  sorry

end sum_of_distances_l92_92695


namespace vector_dot_cross_product_l92_92734

open Matrix


def a : Fin 3 → ℝ := ![2, -4, 1]
def b : Fin 3 → ℝ := ![3, 0, 2]
def c : Fin 3 → ℝ := ![-1, 3, 2]
def d : Fin 3 → ℝ := ![4, -1, 0]

theorem vector_dot_cross_product :
  let ab := ∀ i, a i - b i
  let bc := ∀ i, b i - c i
  let cd := ∀ i, c i - d i
  (ab 0, ab 1, ab 2) • ![bc 0, bc 1, bc 2] ⨯ ![cd 0, cd 1, cd 2] = 45 :=
by
  -- Proof goes here
  sorry

end vector_dot_cross_product_l92_92734


namespace meeting_point_distance_l92_92877

noncomputable def departure_time_zj := 8.0 -- 8:00 AM in hours
noncomputable def departure_time_w := 9.0 -- 9:00 AM in hours
noncomputable def arrival_time := 12.0 -- 12:00 PM in hours
noncomputable def speed_zj := 60 -- speed of Xiao Zhang in km/h
noncomputable def speed_w := 40 -- speed of Xiao Wang in km/h
noncomputable def distance_ab := 120 -- distance between point A and point B in km

noncomputable def meeting_distance : ℝ := 96 -- final result to be proven

theorem meeting_point_distance :
  let time_w_meet := (12.0 - 9.0) * 60 * 40/(60 + 40) in
  (9.0 - 8.0) * speed_zj + time_w_meet = meeting_distance :=
by sorry

end meeting_point_distance_l92_92877


namespace find_four_numbers_l92_92009

open Real

noncomputable def solution_set := (1/4, 1/4, 1/4, 1/4)

theorem find_four_numbers 
  (a b c d : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0)
  (h_sum : a + b + c + d = 1) 
  (h_cond : max (a^2 / b) (b^2 / a) * max (c^2 / d) (d^2 / c) = (min (a + b) (c + d))^4) : 
  (a, b, c, d) = solution_set :=
by
  sorry

end find_four_numbers_l92_92009


namespace find_m_for_root_l92_92022

-- Define the fractional equation to find m
def fractional_equation (x m : ℝ) : Prop :=
  (x + 2) / (x - 1) = m / (1 - x)

-- State the theorem that we need to prove
theorem find_m_for_root : ∃ m : ℝ, (∃ x : ℝ, fractional_equation x m) ∧ m = -3 :=
by
  sorry

end find_m_for_root_l92_92022


namespace concrete_order_amount_l92_92367

def feet_to_yards (x : ℝ) : ℝ := x / 3
def inches_to_yards (x : ℝ) : ℝ := x / 36

-- Given dimensions and slope conditions
def width_feet := 4
def length_feet := 80
def thickness_inches := 4
def slope_increase_per_20_feet := 1

-- Converted dimensions in yards
def width_yards := feet_to_yards width_feet
def length_yards := feet_to_yards length_feet
def thickness_yards := inches_to_yards thickness_inches

-- Additional thickness due to slope in yards
def additional_thickness_due_to_slope := inches_to_yards (slope_increase_per_20_feet * length_feet / 20)
def total_effective_thickness_yards := thickness_yards + additional_thickness_due_to_slope

-- Calculate volume in cubic yards
def volume_cubic_yards := width_yards * length_yards * total_effective_thickness_yards

-- Final required concrete amount, rounded up
def required_concrete_cubic_yards := Float.ceil (volume_cubic_yards.toReal)

-- The proof statement
theorem concrete_order_amount :
  required_concrete_cubic_yards = 8 := 
by
  -- skipping the proof with sorry
  sorry

end concrete_order_amount_l92_92367


namespace distance_between_foci_of_ellipse_l92_92974

theorem distance_between_foci_of_ellipse (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 36 → 2 * Real.sqrt (b^2 - a^2) = 8 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end distance_between_foci_of_ellipse_l92_92974


namespace systematic_sampling_sequence_l92_92819

theorem systematic_sampling_sequence :
  ∃ S : List ℕ, S = [5, 17, 29, 41, 53] ∧
  (∀ k, 1 ≤ k ∧ k ≤ 4 → S.get? k - S.get? (k-1) = some 12) ∧
  S.length = 5 :=
by
  exists [5, 17, 29, 41, 53]
  repeat
    split
  case a_1 => rfl
  case a_2 =>
    intros k hk
    cases hk with h1 h2
    cases k with k0
    simp
    split_ifs
    cases k0
    case zero
      simp
    case succ k0
      cases k0
  case a_3 => simp
  sorry

end systematic_sampling_sequence_l92_92819


namespace smallest_composite_no_prime_factors_less_than_20_l92_92988

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92988


namespace limit_n_a_n_to_zero_l92_92165

open Filter Real

theorem limit_n_a_n_to_zero 
  (a : ℕ → ℝ)
  (h_bound : ∀ n k, n ≤ k ∧ k ≤ 2 * n → 0 ≤ a k ∧ a k ≤ 100 * a n)
  (h_series : Summable a) : 
  Tendsto (λ n, n * a n) atTop (𝓝 0) :=
sorry

end limit_n_a_n_to_zero_l92_92165


namespace pyramid_height_is_6_l92_92011

-- Define the conditions for the problem
def square_side_length : ℝ := 18
def pyramid_base_side_length (s : ℝ) : Prop := s * s = (square_side_length / 2) * (square_side_length / 2)
def pyramid_slant_height (s l : ℝ) : Prop := 2 * s * l = square_side_length * square_side_length

-- State the main theorem
theorem pyramid_height_is_6 (s l h : ℝ) (hs : pyramid_base_side_length s) (hl : pyramid_slant_height s l) : h = 6 := 
sorry

end pyramid_height_is_6_l92_92011


namespace fourth_number_is_two_eighth_number_is_two_l92_92850

-- Conditions:
-- 1. Initial number on the board is 1
-- 2. Sequence of medians observed by Mitya

def initial_number : ℕ := 1
def medians : list ℚ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- Required proof statements:

-- a) The fourth number written on the board is 2
theorem fourth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 3 = some 2 :=
sorry

-- b) The eighth number written on the board is 2
theorem eighth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 7 = some 2 :=
sorry

end fourth_number_is_two_eighth_number_is_two_l92_92850


namespace compute_sin_90_l92_92513

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92513


namespace sin_90_eq_one_l92_92408

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92408


namespace solution_set_l92_92007

theorem solution_set (x : ℝ) (h : x ≠ 5) :
  (∃ y, y = (x * (x + 3) / (x - 5) ^ 2) ∧ y ≥ 15) ↔
  x ≤ 52 / 14 ∨ x ≥ 101 / 14 :=
begin
  sorry
end

end solution_set_l92_92007


namespace range_of_f_l92_92976

noncomputable def f (x : ℝ) : ℝ := 2 * cos ( (π / 4) * sin ( sqrt (x - 3) + 2 * x + 2 ) )

theorem range_of_f : ∀ (x : ℝ), x ≥ 3 → ∃ y, y = f(x) ∧ y ∈ set.Icc (real.sqrt 2) 2 :=
by
  intro x hx
  have hdom : x - 3 ≥ 0 := by linarith
  sorry -- Fill in the proof steps.

end range_of_f_l92_92976


namespace number_of_integers_between_25_and_36_l92_92261

theorem number_of_integers_between_25_and_36 :
  {n : ℕ | 25 < n ∧ n < 36}.card = 10 :=
by
  sorry

end number_of_integers_between_25_and_36_l92_92261


namespace sin_90_eq_1_l92_92461

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92461


namespace fourth_number_is_two_eighth_number_is_two_l92_92835

theorem fourth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 4 ∧ board !! 3 = some 2 :=
by
  sorry

theorem eighth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 8 ∧ board !! 7 = some 2 :=
by
  sorry

end fourth_number_is_two_eighth_number_is_two_l92_92835


namespace find_a_l92_92568

theorem find_a (a : ℝ) : 
  (f : ℝ → ℝ) = (λ x, a * x^3 + 9 * x^2 + 6 * x - 7) →
  (deriv f (-1) = 4) →
  a = 16 / 3 := by
  sorry

end find_a_l92_92568


namespace pinecones_left_l92_92298

theorem pinecones_left 
  (total_pinecones : ℕ)
  (pct_eaten_by_reindeer pct_collected_for_fires : ℝ)
  (total_eaten_by_reindeer : ℕ := (pct_eaten_by_reindeer * total_pinecones).to_nat)
  (total_eaten_by_squirrels : ℕ := 2 * total_eaten_by_reindeer)
  (total_eaten : ℕ := total_eaten_by_reindeer + total_eaten_by_squirrels)
  (pinecones_after_eating : ℕ := total_pinecones - total_eaten)
  (total_collected_for_fires : ℕ := (pct_collected_for_fires * pinecones_after_eating).to_nat)
  (remaining_pinecones : ℕ := pinecones_after_eating - total_collected_for_fires)
  (2000_pinecones : total_pinecones = 2000)
  (20_percent_eaten_by_reindeer : pct_eaten_by_reindeer = 0.20)
  (25_percent_collected_for_fires : pct_collected_for_fires = 0.25) : 
  remaining_pinecones = 600 := 
by
  sorry

end pinecones_left_l92_92298


namespace tshirts_per_package_l92_92759

-- Definitions based on the conditions
def total_tshirts : ℕ := 70
def num_packages : ℕ := 14

-- Theorem to prove the number of t-shirts per package
theorem tshirts_per_package : total_tshirts / num_packages = 5 := by
  -- The proof is omitted, only the statement is provided as required.
  sorry

end tshirts_per_package_l92_92759


namespace mike_arcade_minutes_l92_92758

theorem mike_arcade_minutes (P : ℕ) (H : P = 100) (h_spends_half : ∀ P, spends_half P = P / 2)
  (h_spends_on_food : spends_on_food = 10)
  (h_spends_remainder_on_tokens : ∀ P, spends_remainder_on_tokens P = (P / 2) - 10)
  (h_minutes_per_dollar : minutes_per_dollar = 60 / 8) :
  total_minutes_can_play P = 300 :=
by
  unfold spends_half spends_on_food spends_remainder_on_tokens minutes_per_dollar total_minutes_can_play
  simp [H]
  sorry

-- Definitions for the conditions
def spends_half (P : ℕ) : ℕ := P / 2

def spends_on_food : ℕ := 10

def spends_remainder_on_tokens (P : ℕ) : ℕ := (P / 2) - 10

def minutes_per_dollar : ℕ := 60 / 8

def total_minutes_can_play (P : ℕ) : ℕ :=
  (spends_remainder_on_tokens P) * minutes_per_dollar

end mike_arcade_minutes_l92_92758


namespace teacher_already_graded_worksheets_l92_92921

-- Define the conditions
def num_worksheets : ℕ := 9
def problems_per_worksheet : ℕ := 4
def remaining_problems : ℕ := 16
def total_problems := num_worksheets * problems_per_worksheet

-- Define the required proof
theorem teacher_already_graded_worksheets :
  (total_problems - remaining_problems) / problems_per_worksheet = 5 :=
by sorry

end teacher_already_graded_worksheets_l92_92921


namespace positive_difference_zero_l92_92161

noncomputable def sum_jo := (50 * (50 + 1)) / 2

noncomputable def round_to_nearest_5 (n : ℕ) : ℕ :=
if n % 5 < 3 then (n / 5) * 5 else (n / 5 + 1) * 5

noncomputable def sum_kate : ℕ :=
(1..50).map round_to_nearest_5 |> List.sum

theorem positive_difference_zero : |sum_jo - sum_kate| = 0 := by
  sorry

end positive_difference_zero_l92_92161


namespace Q_points_coplanar_l92_92179

-- Define the basic structure of our problem
variables {R : Type*} [unordered_comm_ring R]

structure Point (R : Type*) := 
  (x : R) 
  (y : R) 
  (z : R)

def midpoint (A B : Point R) : Point R :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2, (A.z + B.z) / 2⟩

noncomputable def plane (p1 p2 p3 : Point R) : set (Point R) :=
  {p : Point R | ∃α β ∈ ℝ, α + β ≤ 1 ∧ p.x = α * p1.x + β * p2.x + (1 - α - β) * p3.x ∧
                              p.y = α * p1.y + β * p2.y + (1 - α - β) * p3.y ∧ 
                              p.z = α * p1.z + β * p2.z + (1 - α - β) * p3.z}

variables (A1 A2 A3 A4 A5 : Point R)
variables (B1 B2 B3 B4 B5 Q1 Q2 Q3 Q4 Q5 : Point R)

axiom H1 : B1 = midpoint A3 A4
axiom H2 : B2 = midpoint A4 A5
axiom H3 : B3 = midpoint A5 A1
axiom H4 : B4 = midpoint A1 A2
axiom H5 : B5 = midpoint A2 A3

theorem Q_points_coplanar :
  ∃ (p1 p2 p3 : Point R), Q1 ∈ plane p1 p2 p3 ∧ Q2 ∈ plane p1 p2 p3 ∧ Q3 ∈ plane p1 p2 p3 ∧ Q4 ∈ plane p1 p2 p3 ∧ Q5 ∈ plane p1 p2 p3 :=
sorry

end Q_points_coplanar_l92_92179


namespace sin_90_eq_one_l92_92410

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92410


namespace no_other_conjugate_numbers_l92_92020

theorem no_other_conjugate_numbers
  {a b : ℝ} {f : ℝ → ℝ}
  (quadratic : ∃ (r s t : ℝ), ∀ x, f(x) = r*x^2 + s*x + t)
  (h1 : f(a) = b)
  (h2 : f(b) = a)
  (h_distinct : a ≠ b) :
   ∀ x y, (f(x) = y) ∧ (f(y) = x) → (x = a ∧ y = b) ∨ (x = b ∧ y = a) :=
by
  sorry

end no_other_conjugate_numbers_l92_92020


namespace area_of_ABCD_l92_92121

open Real

noncomputable def rectangle_area (ABCD : Type) [rectangle ABCD]
  (C_trisected_CF_CE : ∃ F E, (F ∈ line AD) ∧ (E ∈ line AB) ∧ (angle C = 3 * angle FCE))
  (BE : ℝ) (AF : ℝ) : ℝ :=
  192 * sqrt 3 - 96

theorem area_of_ABCD (ABCD : Type) [rectangle ABCD]
  (C_trisected_CF_CE : ∃ F E, (F ∈ line AD) ∧ (E ∈ line AB) ∧ (angle C = 3 * angle FCE))
  (BE_eq_8 : BE = 8)
  (AF_eq_4 : AF = 4) :
  rectangle_area ABCD C_trisected_CF_CE 8 4 = 192 * sqrt 3 - 96 :=
by sorry

end area_of_ABCD_l92_92121


namespace simplify_and_evaluate_l92_92212

def a : ℝ := -4
def b : ℝ := 1/2

theorem simplify_and_evaluate : b * (a + b) + (-a + b) * (-a - b) - a^2 = -2 := by
  sorry

end simplify_and_evaluate_l92_92212


namespace number_of_integers_between_25_and_36_l92_92266

theorem number_of_integers_between_25_and_36 :
  {n : ℕ | 25 < n ∧ n < 36}.card = 10 :=
by
  sorry

end number_of_integers_between_25_and_36_l92_92266


namespace trajectory_of_moving_circle_l92_92642

def circle1 (x y : ℝ) := (x + 4) ^ 2 + y ^ 2 = 2
def circle2 (x y : ℝ) := (x - 4) ^ 2 + y ^ 2 = 2

theorem trajectory_of_moving_circle (x y : ℝ) : 
  (x = 0) ∨ (x ^ 2 / 2 - y ^ 2 / 14 = 1) := 
  sorry

end trajectory_of_moving_circle_l92_92642


namespace find_AC_and_B_l92_92565

-- Define points, lines, and conditions
def A : ℝ × ℝ := (5, 1)

def median_CM (x y : ℝ) : Prop := 2 * x - y - 5 = 0
def altitude_BH (x y : ℝ) : Prop := x - 2 * y - 5 = 0

-- Define the points and lines used in the proof
def B (x y : ℝ) : Prop :=
  let midpoint_AB : ℝ × ℝ := ((x + 5) / 2, (y + 1) / 2)
  ∧ altitude_BH x y
  ∧ median_CM midpoint_AB.1 midpoint_AB.2

theorem find_AC_and_B :
  ∃ (C_eq : ℝ × ℝ → Prop) (B_eq : ℝ × ℝ),
    (∀ t, C_eq = (λ p, 2 * p.1 + p.2 + t = 0))
    ∧ (C_eq = (λ p : ℝ×ℝ, 2 * p.1 + p.2 - 11 = 0))
    ∧ B_eq = (-1, -3) :=
by
  sorry

end find_AC_and_B_l92_92565


namespace cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l92_92183

theorem cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7 (p q r : ℝ)
  (h_roots : 3*p^3 - 4*p^2 + 220*p - 7 = 0 ∧ 3*q^3 - 4*q^2 + 220*q - 7 = 0 ∧ 3*r^3 - 4*r^2 + 220*r - 7 = 0)
  (h_vieta : p + q + r = 4 / 3) :
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 64.556 :=
sorry

end cubic_polynomial_roots_3x3_minus_4x2_plus_220x_minus_7_l92_92183


namespace probability_within_0_80_l92_92683

noncomputable def normal_distribution (mean variance : ℝ) := sorry

variables (η : ℝ) (δ : ℝ)
hypothesis h1 : δ > 0
hypothesis h2 : normal_distribution 100 (δ^2)
hypothesis h3 : ∫ (x : ℝ) in 80..120, pdf (normal_distribution 100 (δ^2)) x = 0.6

theorem probability_within_0_80 : ∫ (x : ℝ) in 0..80, pdf (normal_distribution 100 (δ^2)) x = 0.2 := 
sorry

end probability_within_0_80_l92_92683


namespace sin_90_deg_l92_92449

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92449


namespace smallest_composite_no_prime_factors_less_than_20_l92_92985

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92985


namespace books_sold_l92_92202

theorem books_sold (initial_books new_books current_books books_sold : ℝ)
  (h_initial : initial_books = 4.5)
  (h_new : new_books = 175.3)
  (h_current : current_books = 62.8)
  (h_total : books_sold = (initial_books + new_books) - current_books) :
  books_sold = 117 :=
by
  rw [h_initial, h_new, h_current] at h_total
  exact h_total
sorry

end books_sold_l92_92202


namespace part_I_part_II_l92_92573

-- Part I: Existence of k
theorem part_I (A : ℕ) (dec_exp : ∀ n, A = ∑ i in finset.range (n + 1), (10^i) * (A / 10 ^ i % 10)) : ∃ k : ℕ, let f (A : ℕ) := (∑ i in finset.range (nat.log 10 A + 1), 2^(nat.log 10 A - i) * (A / 10^i % 10)) in 
  let A_seq := λ n, nat.rec_on n A (λ n a_n, f a_n) in A_seq (k + 1) = A_seq k :=
sorry

-- Part II: Finding A_k for 19^86
theorem part_II (A : ℕ) (hA : A = 19^86) :
  let f (A : ℕ) := (∑ i in finset.range (nat.log 10 A + 1), 2^(nat.log 10 A - i) * (A / 10^i % 10)) in 
  let A_seq := λ n, nat.rec_on n A (λ n a_n, f a_n) in A_seq (nat.find (λ k, A_seq (k + 1) = A_seq k)) = 19 :=
sorry

end part_I_part_II_l92_92573


namespace smallest_k_for_simple_sum_l92_92037

def is_simple (n : ℕ) : Prop :=
∀ d ∈ n.digits 10, d = 0 ∨ d = 1

theorem smallest_k_for_simple_sum :
  ∃ k : ℕ, (∀ n : ℕ, ∃ (a : list ℕ), ∀ (i : ℕ), i < k → a.nth i ≠ none ∧ is_simple (a.nth_le i (by linarith)) ∧ n = a.sum) ∧ k = 9 :=
sorry

end smallest_k_for_simple_sum_l92_92037


namespace tan_double_angle_l92_92585

theorem tan_double_angle (α : ℝ) (h : sin α - 2 * cos α = 0) : 
  tan (2 * α) = -4 / 3 :=
begin
  sorry
end

end tan_double_angle_l92_92585


namespace binomial_25_2_eq_300_l92_92393

-- Define the factorial function
def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * factorial n

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Statement of the problem
theorem binomial_25_2_eq_300 : binomial 25 2 = 300 := 
by
  sorry

end binomial_25_2_eq_300_l92_92393


namespace incorrect_relationships_l92_92656

noncomputable def probability (A B : set Ω) : ℝ := sorry

variables {Ω : Type} {P : set Ω → ℝ} (A B : set Ω)

axiom P_intersection : P (A ∩ B) = 1 / 9
axiom P_complement_A : P (Aᶜ) = 2 / 3
axiom P_B : P B = 1 / 3

-- Definition for a mutually exclusive property
def mutually_exclusive (A B : set Ω) : Prop := P (A ∩ B) = 0

-- Definition for a complementary property
def complementary (A B : set Ω) : Prop := A ∩ B = ∅ ∧ A ∪ B = set.univ

-- Definition for an independent property
def independent (A B : set Ω) : Prop := P (A ∩ B) = P A * P B

-- The main theorem with given conditions
theorem incorrect_relationships : mutually_exclusive A B ∨ complementary A B ∨ (mutually_exclusive A B ∧ independent A B) :=
sorry

end incorrect_relationships_l92_92656


namespace common_root_equation_l92_92557

theorem common_root_equation {m : ℝ} (x : ℝ) (h1 : m * x - 1000 = 1001) (h2 : 1001 * x = m - 1000 * x) : m = 2001 ∨ m = -2001 :=
by
  -- Skipping the proof details
  sorry

end common_root_equation_l92_92557


namespace average_speed_of_water_current_l92_92345

theorem average_speed_of_water_current
  (upstream_times : List ℕ) (downstream_times : List ℕ)
  (upstream_distance : ℕ) (downstream_distance : ℕ) :
  upstream_times = [40, 45, 50] →
  downstream_times = [12, 15] →
  upstream_distance = 3 →
  downstream_distance = 2 →
  let total_time := (upstream_times.sum + downstream_times.sum : ℕ) / 60.0,
      total_distance := (upstream_distance + downstream_distance : ℕ),
      average_speed := total_distance / total_time in
  average_speed ≈ 1.85 :=
by
  sorry

end average_speed_of_water_current_l92_92345


namespace question_range_of_k_l92_92073

noncomputable theory

open Real

def has_two_real_solutions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x1 x2 ∈ Set.Icc a b, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2

def range_of_k := Set.Ico (1 / exp 2) (1 / (2 * exp 1))

theorem question_range_of_k
  (f g : ℝ → ℝ)
  (h_f : ∀ x, f x = (x : ℝ) * k)
  (h_g : ∀ x, g x = log x / x)
  (h : has_two_real_solutions f g (1 / exp 1) (exp 1)) :
  k ∈ range_of_k := sorry

end question_range_of_k_l92_92073


namespace school_team_profit_is_333_l92_92363

noncomputable def candy_profit (total_bars : ℕ) (price_800_bars : ℕ) (price_400_bars : ℕ) (sold_600_bars_price : ℕ) (remaining_600_bars_price : ℕ) : ℚ :=
  let cost_800_bars := 800 / 3
  let cost_400_bars := 400 / 4
  let total_cost := cost_800_bars + cost_400_bars
  let revenue_sold_600_bars := 600 / 2
  let revenue_remaining_600_bars := (600 * 2) / 3
  let total_revenue := revenue_sold_600_bars + revenue_remaining_600_bars
  total_revenue - total_cost

theorem school_team_profit_is_333 :
  candy_profit 1200 3 4 2 2 = 333 := by
  sorry

end school_team_profit_is_333_l92_92363


namespace probability_of_sum_being_odd_l92_92309

noncomputable def probability_sum_odd : ℚ :=
  let p_heads := (1/2 : ℚ) in
  let p_tails := 1 - p_heads in
  let p_one_head := (fintype.card {i : Fin 3 // i < 1} : ℕ) * p_heads * p_tails^2 in
  let p_two_heads := (fintype.card {i : Fin 3 // i < 2} * (p_heads^2) * p_tails) in
  let p_three_heads := p_heads^3 in
  let p_sum_odd_one_die := 1 / 2 in
  let p_one_die_sum_odd := p_one_head * p_sum_odd_one_die in
  let p_two_dice_sum_odd := p_two_heads * (2 * ((1/2) * (1/2))) in
  let p_three_dice_sum_odd := p_three_heads * (fintype.card {i : Fin 3 // i = 1} * (1/2)^3) in
  p_one_die_sum_odd + p_two_dice_sum_odd + p_three_dice_sum_odd

theorem probability_of_sum_being_odd : probability_sum_odd = (7 / 16 : ℚ) := 
sorry

end probability_of_sum_being_odd_l92_92309


namespace total_cost_to_fill_large_bucket_l92_92922

-- Defining the conditions related to the problem
def small_bucket_volume : ℕ := 120
def large_bucket_volume : ℕ := 800
def small_bucket_cost : ℕ := 3

-- The main hypothesis
theorem total_cost_to_fill_large_bucket :
  let small_buckets_needed := (large_bucket_volume + small_bucket_volume - 1) / small_bucket_volume in
  let total_cost := small_buckets_needed * small_bucket_cost in
  total_cost = 21 :=
by
  let small_buckets_needed := (large_bucket_volume + small_bucket_volume - 1) / small_bucket_volume
  let total_cost := small_buckets_needed * small_bucket_cost
  have h_needed : small_buckets_needed = 7 := sorry
  have h_cost : total_cost = 21 := sorry
  exact h_cost

end total_cost_to_fill_large_bucket_l92_92922


namespace area_DEFG_l92_92359

-- Define points and the properties of the rectangle ABCD
variable (A B C D E G F : Type)
variables (area_ABCD : ℝ) (Eg_parallel_AB_CD Df_parallel_AD_BC : Prop)
variable (E_position_AD : ℝ) (G_position_CD : ℝ) (F_midpoint_BC : Prop)
variables (length_abcd width_abcd : ℝ)

-- Assumptions based on given conditions
axiom h1 : area_ABCD = 150
axiom h2 : E_position_AD = 1 / 3
axiom h3 : G_position_CD = 1 / 3
axiom h4 : Eg_parallel_AB_CD
axiom h5 : Df_parallel_AD_BC
axiom h6 : F_midpoint_BC

-- Theorem to prove the area of DEFG
theorem area_DEFG : length_abcd * width_abcd / 3 = 50 :=
    sorry

end area_DEFG_l92_92359


namespace value_of_m_l92_92621

theorem value_of_m (a m : ℝ) (h : a > 0) (hm : m ≠ 0) :
  (P : ℝ × ℝ) (P = (m, 3))
  (H : ∀ x : ℝ, -a * x^2 + 2 * a * x + 3 = 3 → x = 0 ∨ x = 2) :
  m = 2 :=
by
  sorry

end value_of_m_l92_92621


namespace min_elements_in_A_l92_92047

theorem min_elements_in_A (n : ℕ) (h : n ≥ 2) (A : set ℕ) (a : ℕ)
  (h1 : 1 ∈ A) (h2 : a ∈ A)
  (h3 : ∀ x ∈ A, x ≠ 1 → ∃ s t p ∈ A, x = s + t + p)
  (h4 : 7 * 3^n < a)
  (h5 : a < 3^(n+2))
  : set.card A = n + 4 :=
by
  sorry

end min_elements_in_A_l92_92047


namespace find_m_l92_92627

theorem find_m (a m : ℝ) (h_pos : a > 0) (h_points : (m, 3) ∈ set_of (λ x : ℝ × ℝ, ∃ x_val : ℝ, x.snd = -a * (x_val)^2 + 2 * a * x_val + 3)) (h_non_zero : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92627


namespace locus_of_points_eq_l92_92947

-- Define the ellipse
def ellipse_eq (x y : ℝ) := (x^2 / 8) + (y^2 / 4) = 1

-- Define the fixed line
def fixed_line (x : ℝ) := x = 2

-- Define the distance formula for the left focus of the ellipse
def distance_from_focus (x y : ℝ) := real.sqrt ((x + 2)^2 + y^2)

-- Define the distance formula from the fixed line
def distance_from_line (x : ℝ) := real.abs (x - 2)

-- Prove that the locus of points is given by the equation y^2 = -8x
theorem locus_of_points_eq : ∀ (x y : ℝ), distance_from_focus x y = distance_from_line x → y^2 = -8 * x := by
  sorry

end locus_of_points_eq_l92_92947


namespace evaluate_product_roots_of_unity_l92_92001

theorem evaluate_product_roots_of_unity :
  let w := Complex.exp (2 * Real.pi * Complex.I / 13)
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) =
  (3^12 + 3^11 + 3^10 + 3^9 + 3^8 + 3^7 + 3^6 + 3^5 + 3^4 + 3^3 + 3^2 + 3 + 1) :=
by
  sorry

end evaluate_product_roots_of_unity_l92_92001


namespace triangle_C_squared_eq_b_a_plus_b_l92_92050

variables {A B C a b : ℝ}

theorem triangle_C_squared_eq_b_a_plus_b
  (h1 : C = 2 * B)
  (h2 : A ≠ B) :
  C^2 = b * (a + b) :=
sorry

end triangle_C_squared_eq_b_a_plus_b_l92_92050


namespace geom_progression_sum_not_equal_term_l92_92035

theorem geom_progression_sum_not_equal_term 
  (a q: ℤ) (k: ℕ → ℕ) (m: ℕ):
  q ≠ 0 ∧ q ≠ -1 → 
  (∀ i j, i ≠ j → k i ≠ k j) → 
  ∀ (l: ℕ), ∑ i in finset.range m, a * q ^ k i ≠ a * q ^ l := 
by 
  sorry

end geom_progression_sum_not_equal_term_l92_92035


namespace distance_from_origin_l92_92906

theorem distance_from_origin :
  ∃ (m : ℝ), m = Real.sqrt (108 + 8 * Real.sqrt 10) ∧
              (∃ (x y : ℝ), y = 8 ∧ 
                            (x - 2)^2 + (y - 5)^2 = 49 ∧ 
                            x = 2 + 2 * Real.sqrt 10 ∧ 
                            m = Real.sqrt ((x^2) + (y^2))) :=
by
  sorry

end distance_from_origin_l92_92906


namespace maryann_rescue_friends_l92_92190

theorem maryann_rescue_friends
    (cheap_time : ℕ)
    (expensive_time : ℕ)
    (total_time : ℕ)
    (F : ℕ) :
  (∀ F, cheap_time = 6 ∧ expensive_time = 8 ∧ total_time = 42 → 6 + 8 = 14 → 14 * F = 42 → F = 3) :=
  by
    intros F H1 H2 H3 H4
    cases H1 with h1 h1'
    cases h1' with h2 h3
    rw [← H2] at H3
    rw [← H4] at H3
    rw [h1, h2, h3]
    sorry

end maryann_rescue_friends_l92_92190


namespace count_integers_between_25_and_36_l92_92282

theorem count_integers_between_25_and_36 :
  {x : ℤ | 25 < x ∧ x < 36}.finite.card = 10 :=
by
  sorry

end count_integers_between_25_and_36_l92_92282


namespace functional_equation_solution_l92_92522

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y) →
  (f = id ∨ f = abs) :=
by sorry

end functional_equation_solution_l92_92522


namespace prime_dates_in_2008_l92_92085

-- Definitions used in the problem statement
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_days := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def days_in_month_2008 (month : ℕ) : ℕ :=
  if month = 2 then 29 else 
  if month = 3 ∨ month = 5 ∨ month = 7 ∨ month = 11 then 31 else 
  if month = 11 then 30 else 0

def prime_months : list ℕ := [2, 3, 5, 7, 11]

-- Helper function to count prime days in a given month
def prime_days_in_month (days : ℕ) : ℕ :=
  prime_days.countp (λ d, d ≤ days)

-- Main theorem statement
theorem prime_dates_in_2008 : 
  ∑ m in prime_months, prime_days_in_month (days_in_month_2008 m) = 53 :=
by sorry

end prime_dates_in_2008_l92_92085


namespace range_of_m_eq_l92_92645

noncomputable def range_of_m (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / x + 4 / y = 1) : Set ℝ :=
  {m : ℝ | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 4 / y = 1 ∧ x + y / 4 < m^2 - 3 * m}
    
theorem range_of_m_eq :
  ∀ x y : ℝ, x > 0 → y > 0 → 1 / x + 4 / y = 1 → 
  range_of_m x y (by sorry) (by sorry) (by sorry) = {m : ℝ | m < -1 ∨ m > 4} :=
begin
  intros x y h1 h2 h3,
  sorry
end

end range_of_m_eq_l92_92645


namespace each_sister_got_five_bars_l92_92650

-- Defining the initial number of granola bars.
def initial_bars : ℕ := 20

-- Defining the number of bars set aside for each day of the week.
def bars_set_aside : ℕ := 7

-- Defining the number of bars traded to friend Pete.
def bars_traded : ℕ := 3

-- Defining the number of sisters.
def number_of_sisters : ℕ := 2

-- Proving that each sister received 5 granola bars.
theorem each_sister_got_five_bars :
  let remaining_bars_after_week := initial_bars - bars_set_aside,
      remaining_bars_after_trade := remaining_bars_after_week - bars_traded,
      bars_per_sister := remaining_bars_after_trade / number_of_sisters
  in bars_per_sister = 5 := by
  sorry

end each_sister_got_five_bars_l92_92650


namespace number_of_integers_between_25_and_36_l92_92264

theorem number_of_integers_between_25_and_36 :
  {n : ℕ | 25 < n ∧ n < 36}.card = 10 :=
by
  sorry

end number_of_integers_between_25_and_36_l92_92264


namespace incorrect_derivative_formula_l92_92875

theorem incorrect_derivative_formula :
  ¬ ((deriv (sin)) = (λ x, -cos x)) :=
by {
  -- Proof omitted
  sorry
}

end incorrect_derivative_formula_l92_92875


namespace decimal_to_binary_25_l92_92237

theorem decimal_to_binary_25: (1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0) = 25 :=
by 
  sorry

end decimal_to_binary_25_l92_92237


namespace nancy_siblings_l92_92246

-- Define the eye color types
inductive EyeColor : Type
| green : EyeColor
| blue : EyeColor
| brown : EyeColor

-- Define the hair color types
inductive HairColor : Type
| red : HairColor
| black : HairColor
| blonde : HairColor

-- Define a child with eye color and hair color
structure Child : Type :=
(name : String)
(eyeColor : EyeColor)
(hairColor : HairColor)

-- List of children
def children : List Child :=
[
  { name := "Lucas", eyeColor := EyeColor.green, hairColor := HairColor.black },
  { name := "Nancy", eyeColor := EyeColor.brown, hairColor := HairColor.blonde },
  { name := "Olivia", eyeColor := EyeColor.brown, hairColor := HairColor.black },
  { name := "Ethan", eyeColor := EyeColor.blue, hairColor := HairColor.blonde },
  { name := "Mia", eyeColor := EyeColor.green, hairColor := HairColor.red },
  { name := "Noah", eyeColor := EyeColor.blue, hairColor := HairColor.blonde },
  { name := "Emma", eyeColor := EyeColor.green, hairColor := HairColor.blonde }
]

-- Define a predicate to indicate siblings with at least one common characteristic
def siblings (c1 c2 c3 : Child) : Prop :=
(c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor) ∧
(c2.eyeColor = c3.eyeColor ∨ c2.hairColor = c3.hairColor) ∧
(c1.eyeColor = c3.eyeColor ∨ c1.hairColor = c3.hairColor)

-- Proof statement
theorem nancy_siblings : ∃ c1 c2, c1.name = "Nancy" ∧ c1.name ≠ c2.name ∧ c1.name ≠ "Nancy" ∧ c2.name ≠ "Nancy" ∧ 
    siblings c1 c2 
:= sorry

end nancy_siblings_l92_92246


namespace sin_90_deg_l92_92458

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92458


namespace executive_vacation_days_l92_92888

theorem executive_vacation_days (n : ℕ) :
  (∑ k in Finset.range n, 16 * (1 / 2) ^ k = 30) → 
  n = 4 :=
by
  simp at *
  sorry

end executive_vacation_days_l92_92888


namespace best_play_wins_majority_l92_92140

variables (n : ℕ)

-- Conditions
def students_in_play_A : ℕ := n
def students_in_play_B : ℕ := n
def mothers : ℕ := 2 * n

-- Question
theorem best_play_wins_majority : 
  (probability_fin_votes_wins_majority (students_in_play_A n) (students_in_play_B n) (mothers n)) = 1 - (1/2)^n :=
sorry

end best_play_wins_majority_l92_92140


namespace find_real_values_for_inequality_l92_92005

theorem find_real_values_for_inequality {x : ℝ} : 
  (x ≠ 5) → ( (x ∈ [3, 5) ∪ (Set.Icc (5 : ℝ) (125 / 14))) ↔ (x * (x + 3) / (x - 5) ^ 2 ≥ 15)) :=
begin
  sorry
end

end find_real_values_for_inequality_l92_92005


namespace sin_ninety_degrees_l92_92398

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92398


namespace leonel_has_more_dogs_l92_92376

theorem leonel_has_more_dogs (total_animals_anthony : ℕ) 
  (cats_fraction_anthony : ℚ) 
  (leonel_cat_ratio : ℚ) 
  (total_combined_animals : ℕ) 
  (cats_ratio_anthony : total_animals_anthony * cats_fraction_anthony = 8)
  (leonel_total_animals : total_combined_animals - total_animals_anthony = 15) :
  let dogs_anthony := total_animals_anthony - 8
      cats_leonel := 8 * leonel_cat_ratio
      dogs_leonel := 15 - cats_leonel
  in dogs_leonel - dogs_anthony = 7 :=
sorry

end leonel_has_more_dogs_l92_92376


namespace paint_cost_per_quart_l92_92094

-- Definitions of conditions
def edge_length (cube_edge_length : ℝ) : Prop := cube_edge_length = 10
def surface_area (s_area : ℝ) : Prop := s_area = 6 * (10^2)
def coverage_per_quart (coverage : ℝ) : Prop := coverage = 120
def total_cost (cost : ℝ) : Prop := cost = 16
def required_quarts (quarts : ℝ) : Prop := quarts = 600 / 120
def cost_per_quart (cost : ℝ) (quarts : ℝ) (price_per_quart : ℝ) : Prop := price_per_quart = cost / quarts

-- Main theorem statement translating the problem into Lean
theorem paint_cost_per_quart {cube_edge_length s_area coverage cost quarts price_per_quart : ℝ} :
  edge_length cube_edge_length →
  surface_area s_area →
  coverage_per_quart coverage →
  total_cost cost →
  required_quarts quarts →
  quarts = s_area / coverage →
  cost_per_quart cost quarts 3.20 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- proof will go here
  sorry

end paint_cost_per_quart_l92_92094


namespace weigh_1_to_10_kg_l92_92867

theorem weigh_1_to_10_kg (n : ℕ) : 1 ≤ n ∧ n ≤ 10 →
  ∃ (a b c : ℤ), 
    (abs a ≤ 1 ∧ abs b ≤ 1 ∧ abs c ≤ 1 ∧
    (n = a * 3 + b * 4 + c * 9)) :=
by sorry

end weigh_1_to_10_kg_l92_92867


namespace f_minimum_value_no_vertical_tangent_l92_92060

noncomputable def f (a x : ℝ) : ℝ := a / x + Real.log x - 1
noncomputable def g (x : ℝ) : ℝ := (Real.log x - 1) * Real.exp x + x

theorem f_minimum_value (a : ℝ) :
  (a ≤ 0 → ¬∃ m, ∀ x ∈ set.Ioo 0 Real.exp, f a x ≥ m) ∧ 
  (0 < a ∧ a < Real.exp → ∃ m, ∀ x ∈ set.Ioo 0 Real.exp, f a x ≥ m ∧ f a a = m) ∧ 
  (a ≥ Real.exp → ∃ m, ∀ x ∈ set.Ioo 0 Real.exp, f a x ≥ m ∧ f a Real.exp = m) :=
sorry

theorem no_vertical_tangent (x_0 : ℝ) :
  x_0 ∈ set.Ioo 0 Real.exp → ¬∃ x, has_deriv_at g x_0 0 :=
sorry

end f_minimum_value_no_vertical_tangent_l92_92060


namespace distance_between_foci_of_ellipse_l92_92972

theorem distance_between_foci_of_ellipse (a b : ℝ) (ha : a = 2) (hb : b = 6) :
  ∀ (x y : ℝ), 9 * x^2 + y^2 = 36 → 2 * Real.sqrt (b^2 - a^2) = 8 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end distance_between_foci_of_ellipse_l92_92972


namespace sin_90_eq_1_l92_92442

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92442


namespace sin_90_eq_one_l92_92424

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92424


namespace problem_statement_l92_92957

noncomputable def i : ℂ := complex.I

theorem problem_statement :
  i^23 + i^52 + i^103 = 1 - 2 * i :=
by
  sorry

end problem_statement_l92_92957


namespace angle_AhatD_C_l92_92127

theorem angle_AhatD_C (A B C D : Type) 
  [Geometry A] [Triangle A B C] 
  (hB : angle B = 50)
  (hA_bisector : is_bisector D A)
  (hC_bisector : is_bisector D C) :
  angle A D C = 115 :=
sorry

end angle_AhatD_C_l92_92127


namespace trigonometric_theorem_holds_l92_92813

variable {a b c : ℝ}
variable {α : ℝ}

theorem trigonometric_theorem_holds (h_ac : 0 < α ∧ α < π / 2) (h_ob : π / 2 < α ∧ α < π) :
  (a = b + c) ∧ (cos α > 0 → a^2 = b^2 + c^2 - 2 * b * c * cos α) ∧ (cos α < 0 → a^2 = b^2 + c^2 - 2 * b * c * cos α) :=
by
  sorry

end trigonometric_theorem_holds_l92_92813


namespace problem_l92_92954

theorem problem (h k : ℤ) 
  (h1 : 5 * 3 ^ 4 - h * 3 ^ 2 + k = 0)
  (h2 : 5 * (-1) ^ 4 - h * (-1) ^ 2 + k = 0)
  (h3 : 5 * 2 ^ 4 - h * 2 ^ 2 + k = 0) :
  |5 * h - 4 * k| = 70 := 
sorry

end problem_l92_92954


namespace parity_of_f_monotonicity_of_f_extreme_values_of_f_l92_92064

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (x y : ℝ) : f(x + y) = f(x) + f(y)
axiom less_than_zero (x : ℝ) (hx : 0 < x) : f(x) < 0
axiom f_at_3 : f(3) = -2

theorem parity_of_f : ∀ x : ℝ, f(-x) = -f(x) :=
by sorry

theorem monotonicity_of_f : ∀ x₁ x₂ : ℝ, (x₁ < x₂) → f(x₂) < f(x₁) :=
by sorry

theorem extreme_values_of_f : 
  ∃ (maxV minV : ℝ), 
    (∀ x ∈ set.Icc (-12 : ℝ) 12, f(x) ≤ maxV) ∧ 
    (∀ x ∈ set.Icc (-12 : ℝ) 12, minV ≤ f(x)) ∧
    (maxV = f(-12) ∧ minV = f(12)) :=
by sorry

end parity_of_f_monotonicity_of_f_extreme_values_of_f_l92_92064


namespace line_ellipse_intersection_length_l92_92122

theorem line_ellipse_intersection_length :
  let line_l (t : ℝ) := (1 + (1 / 2) * t, (sqrt 3 / 2) * t)
  let ellipse_C (θ : ℝ) := (cos θ, 2 * sin θ)
  let points := {P : ℝ × ℝ | ∃ t θ, P = line_l t ∧ P = ellipse_C θ}
  let A := (1, 0)
  let B := (-1/7, -8 * sqrt 3 / 7)
  |(A, B)| = sqrt ((1 + 1/7)^2 + (0 + 8 * sqrt 3 / 7)^2)
in
  ∑ x ∈ points, x = (1, 0) ∨ x = (-1/7, -8 * sqrt 3 / 7) ∧ |(A, B)| = 16/7 :=
by
  sorry

end line_ellipse_intersection_length_l92_92122


namespace pinecones_left_l92_92299

theorem pinecones_left (initial_pinecones : ℕ)
    (percent_eaten_by_reindeer : ℝ)
    (percent_collected_for_fires : ℝ)
    (twice_eaten_by_squirrels : ℕ → ℕ)
    (eaten_by_reindeer : ℕ → ℝ → ℕ)
    (collected_for_fires : ℕ → ℝ → ℕ)
    (h_initial : initial_pinecones = 2000)
    (h_percent_reindeer : percent_eaten_by_reindeer = 0.20)
    (h_twice_squirrels : ∀ n, twice_eaten_by_squirrels n = 2 * n)
    (h_percent_fires : percent_collected_for_fires = 0.25)
    (h_eaten_reindeer : ∀ n p, eaten_by_reindeer n p = n * p)
    (h_collected_fires : ∀ n p, collected_for_fires n p = n * p) :
  let reindeer_eat := eaten_by_reindeer initial_pinecones percent_eaten_by_reindeer
  let squirrel_eat := twice_eaten_by_squirrels reindeer_eat
  let after_eaten := initial_pinecones - reindeer_eat - squirrel_eat
  let fire_collect := collected_for_fires after_eaten percent_collected_for_fires
  let final_pinecones := after_eaten - fire_collect
  final_pinecones = 600 :=
by sorry

end pinecones_left_l92_92299


namespace second_hand_distance_l92_92805

-- Define the conditions
def radius : ℝ := 9
def minutes : ℕ := 45
def rotations_per_minute : ℕ := 1
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def total_distance (circumference : ℝ) (minutes : ℕ) (rotations_per_minute : ℕ) : ℝ :=
  circumference * (minutes * rotations_per_minute)

-- Define the theorem to prove the distance traveled is 810π
theorem second_hand_distance :
  total_distance (circumference radius) minutes rotations_per_minute = 810 * Real.pi := by
  sorry

end second_hand_distance_l92_92805


namespace spinner_final_direction_west_l92_92768

theorem spinner_final_direction_west :
  let initial_direction := "north"
  let clockwise_revolutions := 3 + 1/2 : ℚ
  let counterclockwise_revolutions := 1 + 3/4 : ℚ
  let net_clockwise_revolutions := clockwise_revolutions - counterclockwise_revolutions
  let reduced_net_revolutions := net_clockwise_revolutions % 1
  let final_direction := if reduced_net_revolutions == 0 then "north"
                         else if reduced_net_revolutions == 1/4 then "east"
                         else if reduced_net_revolutions == 1/2 then "south"
                         else if reduced_net_revolutions == 3/4 then "west"
                         else "undefined"
  final_direction = "west" := by sorry

end spinner_final_direction_west_l92_92768


namespace range_b_sq_e_n_minus_2_e_b_l92_92581

theorem range_b_sq_e_n_minus_2_e_b (a b : ℝ) (h : b * (Real.exp a - 1) + a = Real.exp b - Real.log b) :
  Set.range (λ n, b^2 * Real.exp n - 2 * Real.exp b) = Set.Ici (-Real.exp 1) :=
sorry

end range_b_sq_e_n_minus_2_e_b_l92_92581


namespace price_increase_problem_l92_92801

variable (P P' x : ℝ)

theorem price_increase_problem
  (h1 : P' = P * (1 + x / 100))
  (h2 : P = P' * (1 - 23.076923076923077 / 100)) :
  x = 30 :=
by
  sorry

end price_increase_problem_l92_92801


namespace equal_costs_at_60_guests_l92_92217

def caesars_cost (x : ℕ) : ℕ := 800 + 30 * x
def venus_cost (x : ℕ) : ℕ := 500 + 35 * x

theorem equal_costs_at_60_guests : 
  ∃ x : ℕ, caesars_cost x = venus_cost x ∧ x = 60 := 
by
  existsi 60
  unfold caesars_cost venus_cost
  split
  . sorry
  . refl

end equal_costs_at_60_guests_l92_92217


namespace friends_trio_exists_l92_92000

variables {Student : Type} {School : Type}

-- Define schools A, B, C each with n students
variables (A B C : set Student) (n : ℕ)
variables (a : Student) (b : Student) (c : Student)

-- Predicates for students belonging to schools
variables (in_A : Student → Prop) (in_B : Student → Prop) (in_C : Student → Prop)
variables (is_friend : Student → Student → Prop) -- Friendship predicate

-- Conditions
axiom disjoint_schools : ∀ x, (in_A x → ¬ in_B x ∧ ¬ in_C x) ∧ (in_B x → ¬ in_A x ∧ ¬ in_C x) ∧ (in_C x → ¬ in_A x ∧ ¬ in_B x)
axiom friends_symm : ∀ x y, is_friend x y → is_friend y x
axiom n_students : cardinal.mk { x // in_A x } = n ∧ cardinal.mk { x // in_B x } = n ∧ cardinal.mk { x // in_C x } = n
axiom friends_count : ∀ x, (in_A x ∨ in_B x ∨ in_C x) → (cardinal.mk { y // (in_A y ∨ in_B y ∨ in_C y) ∧ ¬ (in_A x ∧ in_B x ∧ in_C x) ∧ is_friend x y }) ≥ n + 1

theorem friends_trio_exists :
  ∃ a b c, in_A a ∧ in_B b ∧ in_C c ∧ is_friend a b ∧ is_friend b c ∧ is_friend c a :=
sorry

end friends_trio_exists_l92_92000


namespace evaluate_expression_l92_92532

noncomputable def math_expr (x c : ℝ) : ℝ := (x^2 + c)^2 - (x^2 - c)^2

theorem evaluate_expression (x : ℝ) (c : ℝ) (hc : 0 < c) : 
  math_expr x c = 4 * x^2 * c :=
by sorry

end evaluate_expression_l92_92532


namespace fourth_number_is_two_eighth_number_is_two_l92_92852

-- Conditions:
-- 1. Initial number on the board is 1
-- 2. Sequence of medians observed by Mitya

def initial_number : ℕ := 1
def medians : list ℚ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- Required proof statements:

-- a) The fourth number written on the board is 2
theorem fourth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 3 = some 2 :=
sorry

-- b) The eighth number written on the board is 2
theorem eighth_number_is_two (numbers : list ℕ) (h_initial : numbers.head = initial_number)
  (h_medians : ∀ k, medians.nth k = some (list.median (numbers.take (k + 1)))) :
  numbers.nth 7 = some 2 :=
sorry

end fourth_number_is_two_eighth_number_is_two_l92_92852


namespace dot_product_perpendicular_vectors_l92_92657

variables (a b : ℝ^3) (h1 : inner a b = 0) (ha : ‖a‖ = 5) (hb : ‖b‖ = 2)

theorem dot_product_perpendicular_vectors :
  inner (a + b) (a - b) = 21 :=
sorry

end dot_product_perpendicular_vectors_l92_92657


namespace tan_double_angle_l92_92025

theorem tan_double_angle (θ : ℝ) (h_θ1 : sin θ = 1 / 3) (h_θ2 : 0 < θ ∧ θ < π / 2) :
  tan (2 * θ) = 4 * Real.sqrt 2 / 7 :=
sorry

end tan_double_angle_l92_92025


namespace find_certain_number_l92_92236

theorem find_certain_number (x certain_number : ℤ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (certain_number + 62 + 98 + 124 + x) / 5 = 78) : 
  certain_number = 106 := 
by 
  sorry

end find_certain_number_l92_92236


namespace sum_of_reciprocal_squares_lt_inequality_l92_92194

theorem sum_of_reciprocal_squares_lt_inequality (n : ℕ) (hn : n ≥ 2) :
  1 + ∑ i in Finset.range (n - 1), 1 / ((i + 2)^2 : ℝ) < (2 * (n : ℝ) - 1) / (n : ℝ) :=
by
  sorry

end sum_of_reciprocal_squares_lt_inequality_l92_92194


namespace equal_sum_of_distances_l92_92082

-- Define the types for Points and Triangles
def Point := ℝ × ℝ

structure Triangle :=
  (A B C : Point)

-- Define the common circumcircle and incircle condition
def common_circumcircle (T1 T2 : Triangle) : Prop :=
  ∃ (O : Point) (R : ℝ), -- There exists a common circumcenter O and radius R
    distance O T1.A = R ∧ distance O T1.B = R ∧ distance O T1.C = R ∧
    distance O T2.A = R ∧ distance O T2.B = R ∧ distance O T2.C = R

def common_incircle (T1 T2 : Triangle) : Prop :=
  ∃ (I : Point) (r : ℝ), -- There exists a common incenter I and radius r
    distance I (orthogonal_projection I T1.AB) = r ∧ 
    distance I (orthogonal_projection I T1.BC) = r ∧ 
    distance I (orthogonal_projection I T1.CA) = r ∧
    distance I (orthogonal_projection I T2.AB) = r ∧ 
    distance I (orthogonal_projection I T2.BC) = r ∧ 
    distance I (orthogonal_projection I T2.CA) = r

-- Define the function to calculate the sum of distances from a point to the sides of a triangle
def sum_of_distances (P : Point) (T : Triangle) : ℝ :=
  distance P (orthogonal_projection P T.AB) +
  distance P (orthogonal_projection P T.BC) +
  distance P (orthogonal_projection P T.CA)

-- Define the main theorem
theorem equal_sum_of_distances (T1 T2 : Triangle) (P : Point)
  (h1 : common_circumcircle T1 T2)
  (h2 : common_incircle T1 T2)
  (h3 : P.inside T1 ∧ P.inside T2) :
  sum_of_distances P T1 = sum_of_distances P T2 :=
sorry

end equal_sum_of_distances_l92_92082


namespace sequence_formula_l92_92231

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0     => 0  -- We adjust for Lean's ℕ which starts at 0
  | 1     => 1
  | 2     => 3
  | 3     => 7
  | 4     => 15
  | 5     => 31
  | (n+1) => 2^(n + 1) - 1

theorem sequence_formula (n : ℕ) :
  sequence n = 2^n - 1 := 
sorry

end sequence_formula_l92_92231


namespace trisection_triangle_l92_92313

theorem trisection_triangle (A B C D M : Point)
  (h_triangle: is_triangle A B C)
  (h_median: is_median A D B C)
  (h_midpoint_M: is_midpoint M A D)
  : divides AB at M (1, 2) :=
sorry

end trisection_triangle_l92_92313


namespace arithmetic_sequence_sum_l92_92125

variable (a : ℕ → ℝ)
variable (d : ℝ)

noncomputable def arithmetic_sequence := ∀ n : ℕ, a n = a 0 + n * d

theorem arithmetic_sequence_sum (h₁ : a 1 + a 2 = 3) (h₂ : a 3 + a 4 = 5) :
  a 7 + a 8 = 9 :=
by
  sorry

end arithmetic_sequence_sum_l92_92125


namespace number_of_elements_in_union_l92_92582

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

theorem number_of_elements_in_union : ncard (A ∪ B) = 4 :=
by
  sorry

end number_of_elements_in_union_l92_92582


namespace sum_of_real_roots_eq_three_l92_92319

theorem sum_of_real_roots_eq_three : 
  (∑ root in (Polynomial.roots (Polynomial.mk [1, -4, 5, -4, 1])).toFinset, root.re) = 3 := 
sorry

end sum_of_real_roots_eq_three_l92_92319


namespace largest_n_sum_exceeds_2008_l92_92786

theorem largest_n_sum_exceeds_2008 :
  ∃ (n : ℕ), (∑ i in finset.range (n + 1), i) > 2008 :=
sorry

end largest_n_sum_exceeds_2008_l92_92786


namespace sin_90_degrees_l92_92486

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92486


namespace sin_90_deg_l92_92455

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92455


namespace expression_nonnegative_l92_92523

theorem expression_nonnegative (x : ℝ) :
  0 <= x ∧ x < 3 → (2*x - 6*x^2 + 9*x^3) / (9 - x^3) ≥ 0 := 
by
  sorry

end expression_nonnegative_l92_92523


namespace integer_values_count_l92_92250

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l92_92250


namespace product_of_ab_l92_92794

theorem product_of_ab (u v a b : ℂ) (c : ℝ) :
  u = -3 + 2 * complex.i ∧ v = 2 + 2 * complex.i ∧ 
  (a = 1 - complex.i ∧ b = 1 + complex.i ∧ 
  (∃ z : ℂ, az + b*z.conj = c)) → 
  a * b = 2 :=
by 
  sorry

end product_of_ab_l92_92794


namespace simplify_and_evaluate_l92_92210

theorem simplify_and_evaluate :
  let x := √2 + 1 in
  (1 - 1 / x) / ((x^2 - 2*x + 1) / x^2) = 1 + √2 / 2 :=
by
  sorry

end simplify_and_evaluate_l92_92210


namespace total_slices_served_today_l92_92362

theorem total_slices_served_today (slices_lunch slices_dinner : Nat) (h_lunch : slices_lunch = 7) (h_dinner : slices_dinner = 5) : 
  slices_lunch + slices_dinner = 12 :=
by {
  rw [h_lunch, h_dinner],
  exact rfl
}

end total_slices_served_today_l92_92362


namespace sin_90_eq_one_l92_92416

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92416


namespace triangle_area_l92_92646

noncomputable def vector_OP (x : ℝ) : ℝ × ℝ := (2 * Real.cos (Real.pi / 2 + x), -1)
noncomputable def vector_OQ (x : ℝ) : ℝ × ℝ := (-Real.sin (Real.pi / 2 - x), Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (vector_OP x).1 * (vector_OQ x).1 + (vector_OP x).2 * (vector_OQ x).2

def area_of_triangle (a b c : ℝ) (A : ℝ) : ℝ := 0.5 * b * c * Real.sin A

theorem triangle_area
  (A : ℝ)
  (b c A : ℝ)   
  (h_f : f A = 1)
  (h_bc : b + c = 5 + 3 * Real.sqrt 2)
  (h_a : a = Real.sqrt 13)
  (h_acute : 0 < A ∧ A < Real.pi / 2) :
  area_of_triangle a b c (Real.pi / 4) = 15 / 2 :=
begin
  sorry
end

end triangle_area_l92_92646


namespace bulbs_incandescent_on_l92_92932

variables (I F : ℕ)

noncomputable def percent_incandescent_on : ℝ :=
  let total_bulbs_on := 0.30 * I + 0.80 * F in
  let total_bulbs := I + F in
  100 * (0.30 * I / total_bulbs_on)

theorem bulbs_incandescent_on (h : 0.30 * I + 0.80 * F = 0.70 * (I + F)) :
  percent_incandescent_on I F = 8.57 := 
sorry

end bulbs_incandescent_on_l92_92932


namespace common_root_of_two_equations_l92_92558

theorem common_root_of_two_equations (m x : ℝ) :
  (m * x - 1000 = 1001) ∧ (1001 * x = m - 1000 * x) → (m = 2001 ∨ m = -2001) :=
by
  sorry

end common_root_of_two_equations_l92_92558


namespace remaining_number_is_6218_l92_92197

theorem remaining_number_is_6218 :
  let candidates := { n : ℕ | n ∈ Set.Icc 3 223 ∧ n % 4 = 3 },
      initial_sum := candidates.sum,
      steps := candidates.card - 1
  in initial_sum - 2 * steps = 6218 :=
by
  let candidates := { n : ℕ | n ∈ Set.Icc 3 223 ∧ n % 4 = 3 }
  let initial_sum := candidates.sum
  let steps := candidates.card - 1
  have h_initial_sum : initial_sum = 6328 := sorry
  have h_steps : steps = 55 := sorry
  calc
    initial_sum - 2 * steps
        = 6328 - 2 * 55 : by rw [h_initial_sum, h_steps]
    ... = 6218 : by norm_num

end remaining_number_is_6218_l92_92197


namespace find_m_l92_92608

variable (a m x : ℝ)

noncomputable def quadratic_function : ℝ → ℝ := λ x, -a * x^2 + 2 * a * x + 3

theorem find_m (h1 : a > 0) (h2 : quadratic_function a m = 3) (h3 : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92608


namespace sin_90_eq_one_l92_92406

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92406


namespace sum_of_squares_is_18_l92_92953

open Complex

-- Define the equation x^9 - 27^3 = 0
def equation (x : ℂ) : Prop := x^9 = 27^3

-- Define the sum of squares for the real solutions of the equation
def sum_of_squares_of_real_solutions : ℂ :=
  let sols := Real.roots_of_polynomial_has_solutions 
  sols.sum (λ x, x^2)

-- The theorem to be proven
theorem sum_of_squares_is_18 : sum_of_squares_of_real_solutions = 18 := by sorry

end sum_of_squares_is_18_l92_92953


namespace mary_investment_is_600_l92_92754

-- Define the conditions.
def mary_investment {M : ℕ} (total_profit : ℕ) (mike_investment : ℕ) (mary_share_more : ℕ) (mary_total_profit : ℕ) (mike_total_profit : ℕ) : Prop :=
  (total_profit / 3 / 2) + (M * (total_profit - total_profit / 3) / (M + mike_investment)) = mary_total_profit ∧
  (total_profit / 3 / 2) + (mike_investment * (total_profit - total_profit / 3) / (M + mike_investment)) = mike_total_profit ∧
  mary_total_profit = mike_total_profit + mary_share_more

-- Construct the theorem to prove Mary invested $600.
theorem mary_investment_is_600 : ∃ M, mary_investment 7500 400 1000 4250 3250 ∧ M = 600 := 
by
  use 600
  sorry

end mary_investment_is_600_l92_92754


namespace fourth_number_is_two_eighth_number_is_two_l92_92839

theorem fourth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 4 ∧ board !! 3 = some 2 :=
by
  sorry

theorem eighth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 8 ∧ board !! 7 = some 2 :=
by
  sorry

end fourth_number_is_two_eighth_number_is_two_l92_92839


namespace basketball_team_selection_l92_92769

-- Definitions and conditions
def num_members : Nat := 15
def num_captains : Nat := 2
def num_starting_positions : Nat := 5
def remaining_members := num_members - num_captains
def binom (n k : ℕ) := nat.choose n k

-- The proof problem statement
theorem basketball_team_selection :
  (binom num_members num_captains) * (remaining_members.perm num_starting_positions) = 16201200 := 
by sorry

end basketball_team_selection_l92_92769


namespace divides_three_and_eleven_l92_92004

theorem divides_three_and_eleven (n : ℕ) (h : n ≥ 1) : (n ∣ 3^n + 1 ∧ n ∣ 11^n + 1) ↔ (n = 1 ∨ n = 2) := by
  sorry

end divides_three_and_eleven_l92_92004


namespace trig_identity_proof_l92_92816

theorem trig_identity_proof :
  let sin240 := - (Real.sin (120 * Real.pi / 180))
  let tan240 := Real.tan (240 * Real.pi / 180)
  Real.sin (600 * Real.pi / 180) + tan240 = Real.sqrt 3 / 2 :=
by
  sorry

end trig_identity_proof_l92_92816


namespace number_of_distinguishable_large_triangles_l92_92351

theorem number_of_distinguishable_large_triangles (colors : Fin 8) :
  ∃(large_triangles : Fin 960), true :=
by
  sorry

end number_of_distinguishable_large_triangles_l92_92351


namespace fourth_number_on_board_eighth_number_on_board_l92_92847

theorem fourth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 4 ∧ median numbers[3] = 2 :=
sorry

theorem eighth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 8 ∧ median numbers[7] = 2 :=
sorry

end fourth_number_on_board_eighth_number_on_board_l92_92847


namespace constant_expression_l92_92638

noncomputable def a (n : ℕ) : ℚ :=
  if n > 0 then (2^n) / (2^(2^n) + 1) else 0

noncomputable def A (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n+1), a i

noncomputable def B (n : ℕ) : ℚ :=
  ∏ i in Finset.range (n+1), a i

theorem constant_expression (n : ℕ) (h : n > 0) : 3 * A n + B n * 2^((1 + n) * (2 - n) / 2) = 2 :=
  sorry

end constant_expression_l92_92638


namespace time_to_pass_bridge_correct_l92_92327

noncomputable def length_of_train : ℝ := 327 -- meters
noncomputable def speed_of_train_kmh : ℝ := 40 -- km/h
noncomputable def length_of_bridge : ℝ := 122 -- meters

noncomputable def convert_kmh_to_mps (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

noncomputable def total_distance_to_cover : ℝ :=
  length_of_train + length_of_bridge

noncomputable def speed_of_train_mps : ℝ :=
  convert_kmh_to_mps speed_of_train_kmh

noncomputable def time_to_pass_bridge : ℝ :=
  total_distance_to_cover / speed_of_train_mps

theorem time_to_pass_bridge_correct :
  time_to_pass_bridge ≈ 40.4 := 
sorry

end time_to_pass_bridge_correct_l92_92327


namespace mode_not_change_if_one_removed_l92_92365

def data_set : List ℕ := [5, 6, 8, 8, 8, 1, 4]

def mode_removed (lst : List ℕ) : ℕ :=
(lst.erase 8).maximum' sorry

theorem mode_not_change_if_one_removed :
  mode_removed data_set = 8 :=
sorry

end mode_not_change_if_one_removed_l92_92365


namespace trig_identity_l92_92815

theorem trig_identity (α : ℝ) :
  (Real.cos (α - 35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180 + α) +
   Real.sin (α - 35 * Real.pi / 180) * Real.sin (25 * Real.pi / 180 + α)) = 1 / 2 :=
by
  sorry

end trig_identity_l92_92815


namespace mutually_exclusive_event_l92_92904

theorem mutually_exclusive_event (A B C D: Prop) 
  (h_A: ¬ (A ∧ (¬D)) ∧ ¬ ¬ D)
  (h_B: ¬ (B ∧ (¬D)) ∧ ¬ ¬ D)
  (h_C: ¬ (C ∧ (¬D)) ∧ ¬ ¬ D)
  (h_D: ¬ (D ∧ (¬D)) ∧ ¬ ¬ D) :
  D :=
sorry

end mutually_exclusive_event_l92_92904


namespace inequality_solution_l92_92655

theorem inequality_solution (x y : ℝ) (h : 5 * x > -5 * y) : x + y > 0 :=
sorry

end inequality_solution_l92_92655


namespace larger_number_l92_92093

theorem larger_number (a b : ℕ) (h1 : 5 * b = 7 * a) (h2 : b - a = 10) : b = 35 :=
sorry

end larger_number_l92_92093


namespace remaining_players_lives_l92_92308

theorem remaining_players_lives :
  ∀ (initial_players quit_players total_lives : ℕ),
    initial_players = 13 →
    quit_players = 8 →
    total_lives = 30 →
    (initial_players - quit_players) = 5 →
    total_lives / (initial_players - quit_players) = 6 :=
by
  intros initial_players quit_players total_lives
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end remaining_players_lives_l92_92308


namespace max_m_value_l92_92602

def f (x : ℝ) : ℝ := x^2 - 5 * x + 7

theorem max_m_value (n : ℕ) (h_n : 0 < n) : 
  ∃ (m : ℕ) (a : Fin (m + 1) → ℝ), 
    (∀ i, 1 ≤ a i ∧ a i ≤ (n : ℝ) + 5 / n) ∧ 
    m = 6 ∧ 
    f (a 0) > ∑ i in Finset.range m, f (a (i + 1)) := sorry

end max_m_value_l92_92602


namespace three_digit_numbers_with_three_digits_count_l92_92654

theorem three_digit_numbers_with_three_digits_count : 
  {n : ℕ // n < 1000 ∧ n > 0 ∧ (∀ d1 d2 d3 : ℕ, (n = d1 * 100 + d2 * 10 + d3) → 
  (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3))}.card = 648 :=
by
  sorry

end three_digit_numbers_with_three_digits_count_l92_92654


namespace probability_two_defective_phones_l92_92366

theorem probability_two_defective_phones (total_smartphones : ℕ) 
  (typeA amountA defectiveA : ℕ) 
  (typeB amountB defectiveB : ℕ)
  (typeC amountC defectiveC : ℕ)
  (hA : amountA = 100) (hB : amountB = 80) (hC : amountC = 70)
  (hDA : defectiveA = 30) (hDB : defectiveB = 25) (hDC : defectiveC = 21)
  (hTotal : total_smartphones = 250) :
  let P_first_pick := (defectiveA + defectiveB + defectiveC) / total_smartphones,
      P_second_pick := (defectiveA + defectiveB + defectiveC - 1) / (total_smartphones - 1)
  in P_first_pick * P_second_pick ≈ 0.0916 :=
by
  -- Definitions and setup
  let total_defective := defectiveA + defectiveB + defectiveC
  have hTotal_defective: total_defective = 76 := by 
    rw [hDA, hDB, hDC]; norm_num
    
  -- Probabilities
  let P_first_pick := (total_defective : ℚ) / total_smartphones
  have hP_first_pick: P_first_pick = 76 / 250 := by rw [hTotal_defective, hTotal]; norm_num
  
  let P_second_pick := (total_defective - 1 : ℚ) / (total_smartphones - 1)
  
  -- Final probability calculation
  let P_both_defective := P_first_pick * P_second_pick
  have approx_P_both_defective: P_both_defective ≈ 0.0916 := 
    by 
      have eq1: (76 : ℚ) / 250 = 0.304 := by norm_num
      have eq2: (75 : ℚ) / 249 = 0.3012 := by norm_num
      rw [hP_first_pick, eq2]
      have eq3: 0.304 * 0.3012 = 0.0915632 := by norm_num
      convert eq3
  exact approx_P_both_defective

end probability_two_defective_phones_l92_92366


namespace seq_eleven_l92_92039

noncomputable def seq (n : ℕ) : ℤ := sorry

axiom seq_add (p q : ℕ) (hp : 0 < p) (hq : 0 < q) : seq (p + q) = seq p + seq q
axiom seq_two : seq 2 = -6

theorem seq_eleven : seq 11 = -33 := by
  sorry

end seq_eleven_l92_92039


namespace hockey_season_total_games_l92_92116

theorem hockey_season_total_games :
  ∃ (total_games : Nat),
    let top5_vs_top5 := (4 * 12 * 5) / 2,
    let top5_vs_other10 := 10 * 8 * 5,
    let mid5_vs_mid5 := (4 * 10 * 5) / 2,
    let mid5_vs_bottom5 := 5 * 6 * 5,
    let bottom5_vs_bottom5 := (4 * 8 * 5) / 2,
    total_games = top5_vs_top5 + top5_vs_other10 + mid5_vs_mid5 + mid5_vs_bottom5 + bottom5_vs_bottom5 ∧
    total_games = 850 :=
by
  let top5_vs_top5 := (4 * 12 * 5) / 2
  let top5_vs_other10 := 10 * 8 * 5
  let mid5_vs_mid5 := (4 * 10 * 5) / 2
  let mid5_vs_bottom5 := 5 * 6 * 5
  let bottom5_vs_bottom5 := (4 * 8 * 5) / 2
  let total_games := top5_vs_top5 + top5_vs_other10 + mid5_vs_mid5 + mid5_vs_bottom5 + bottom5_vs_bottom5
  existsi total_games
  simp [top5_vs_top5, top5_vs_other10, mid5_vs_mid5, mid5_vs_bottom5, bottom5_vs_bottom5, -(Nat.add_comm)]
  refine ⟨rfl, by rfl⟩
  sorry

end hockey_season_total_games_l92_92116


namespace students_behind_Minyoung_l92_92871

def total_students : ℕ := 35
def students_in_front_of_Minyoung : ℕ := 27

theorem students_behind_Minyoung (N F : ℕ) (hN : N = total_students) (hF : F = students_in_front_of_Minyoung) : N - (F + 1) = 7 := 
by
  rw [hN, hF]
  simp
  sorry

end students_behind_Minyoung_l92_92871


namespace sin_90_eq_1_l92_92468

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92468


namespace piecewise_function_evaluation_l92_92070

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / (5 - x) else Real.log (x) / Real.log 4

theorem piecewise_function_evaluation : f (f (-3)) = -3 / 2 :=
by sorry

end piecewise_function_evaluation_l92_92070


namespace best_play_majority_win_probability_l92_92135

theorem best_play_majority_win_probability (n : ℕ) :
  (1 - (1 / 2) ^ n) = probability_best_play_wins_majority n :=
sorry

end best_play_majority_win_probability_l92_92135


namespace fish_to_rice_value_l92_92667

variable (f l r : ℝ)

theorem fish_to_rice_value (h1 : 5 * f = 3 * l) (h2 : 2 * l = 7 * r) : f = 2.1 * r :=
by
  sorry

end fish_to_rice_value_l92_92667


namespace distance_to_line_from_focus_l92_92793

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

def line_equation (x y : ℝ) : Prop := y = sqrt(3) * x

def right_focus : (ℝ × ℝ) := (1, 0)

theorem distance_to_line_from_focus :
  ∀ (d : ℝ), d = (abs (-sqrt(3))) / sqrt ((1:ℝ)^2 + (sqrt(3))^2) → d = sqrt(3) / 2 :=
by
  sorry

end distance_to_line_from_focus_l92_92793


namespace find_X_for_mean_increase_l92_92798

theorem find_X_for_mean_increase :
  ∃ (X : ℝ), let F := [-4, -1, X, 6, 9] in
  let F_new := [2, 3, X, 6, 9] in
  (F.sum / 5) * 2 = F_new.sum / 5 ∧ X = 0 :=
by
  sorry

end find_X_for_mean_increase_l92_92798


namespace problem_statement_l92_92926

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

def is_increasing_on_domain (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem problem_statement :
  (∃ f : ℝ → ℝ, (f = (λ x, x + x^3)) ∧ is_odd f ∧ is_increasing_on_domain f) :=
begin
  existsi (λ x : ℝ, x + x^3),
  split,
  { refl },
  split,
  { intros x,
    simp,
    ring },
  { intros x y hxy,
    simp,
    linarith [pow_pos (pow_two_pos_iff.2 hxy), pow_two_nonneg x, add_pos, zero_lt_one] }
end

end problem_statement_l92_92926


namespace determine_m_l92_92637

theorem determine_m (a m : ℝ) (h : a > 0) (h2 : (m, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = -a * p.1 ^ 2 + 2 * a * p.1 + 3)) (h3 : m ≠ 0) : m = 2 :=
sorry

end determine_m_l92_92637


namespace expected_value_of_coins_is_45_point_5_cents_l92_92902

-- Define the values of the coins
def penny : ℝ := 1
def nickel : ℝ := 5
def dime : ℝ := 10
def quarter : ℝ := 25
def half_dollar : ℝ := 50

-- Define the probability each coin comes up heads
def prob_heads : ℝ := 1 / 2

-- Calculate expected values for each coin
def expected_penny := prob_heads * penny
def expected_nickel := prob_heads * nickel
def expected_dime := prob_heads * dime
def expected_quarter := prob_heads * quarter
def expected_half_dollar := prob_heads * half_dollar

-- The expected value of the total amount when all coins are flipped
def expected_value_total := expected_penny + expected_nickel + expected_dime + expected_quarter + expected_half_dollar

-- Proof statement
theorem expected_value_of_coins_is_45_point_5_cents :
  expected_value_total = 45.5 := sorry

end expected_value_of_coins_is_45_point_5_cents_l92_92902


namespace sin_90_eq_1_l92_92441

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92441


namespace quarters_spent_l92_92778

theorem quarters_spent (original : ℕ) (remaining : ℕ) (q : ℕ) 
  (h1 : original = 760) 
  (h2 : remaining = 342) 
  (h3 : q = original - remaining) : q = 418 := 
by
  sorry

end quarters_spent_l92_92778


namespace quadratic_function_m_value_l92_92672

theorem quadratic_function_m_value (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - 3 = 0 → x^2 = x^{m-1}) → m = 3 := by sorry

end quadratic_function_m_value_l92_92672


namespace tank_fill_time_with_hole_l92_92767

def pipe_filling_rate := 1 / 15.0
def hole_emptying_rate := 1 / 60.000000000000014

theorem tank_fill_time_with_hole : 
  let net_filling_rate := pipe_filling_rate - hole_emptying_rate in
  (1 / net_filling_rate) = 20.000000000000001 :=
by
  let Rp := pipe_filling_rate
  let Rh := hole_emptying_rate
  let Rnet := Rp - Rh
  have H1 : Rp = 1 / 15.0 := rfl
  have H2 : Rh = 1 / 60.000000000000014 := rfl
  have H3 : 1 / Rnet = 20.000000000000001 := sorry
  exact H3

end tank_fill_time_with_hole_l92_92767


namespace parabola_translation_l92_92689

theorem parabola_translation :
  ∀ x y : ℝ, (y = (1/2) * (x - 4)^2 + 2) ↔ (∃ y0 : ℝ, ∃ x0 : ℝ, y0 = (1/2) * x0^2 - 1 ∧ y = y0 + 3 ∧ x = x0 + 4) :=
begin
  sorry
end

end parabola_translation_l92_92689


namespace target_statement_l92_92692

variable {a : ℕ → ℝ} -- Let a represent the arithmetic sequence
variable (d : ℝ) -- Let d be the common difference of the arithmetic sequence 

-- Defining arithmetic sequences
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n + d

-- Given condition
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 3 + a 4 + a 12 + a 13 + a 14 = 8

-- Target statement to be proved
theorem target_statement (a : ℕ → ℝ) (d : ℝ) [arith_seq: arithmetic_sequence a d ] (cond: given_condition a) :
    5 * a 6 - 2 * a 3 = 4 :=
sorry

end target_statement_l92_92692


namespace parabola_distance_proof_l92_92056

noncomputable def parabola_focus_distance : ℝ := sorry

theorem parabola_distance_proof :
  let P := (1 : ℝ, 2 : ℝ),
      F := (1 : ℝ, 0 : ℝ) in
  |(F.2 - P.2)| = 2 :=
by
  sorry

end parabola_distance_proof_l92_92056


namespace pinecones_left_l92_92300

theorem pinecones_left (initial_pinecones : ℕ)
    (percent_eaten_by_reindeer : ℝ)
    (percent_collected_for_fires : ℝ)
    (twice_eaten_by_squirrels : ℕ → ℕ)
    (eaten_by_reindeer : ℕ → ℝ → ℕ)
    (collected_for_fires : ℕ → ℝ → ℕ)
    (h_initial : initial_pinecones = 2000)
    (h_percent_reindeer : percent_eaten_by_reindeer = 0.20)
    (h_twice_squirrels : ∀ n, twice_eaten_by_squirrels n = 2 * n)
    (h_percent_fires : percent_collected_for_fires = 0.25)
    (h_eaten_reindeer : ∀ n p, eaten_by_reindeer n p = n * p)
    (h_collected_fires : ∀ n p, collected_for_fires n p = n * p) :
  let reindeer_eat := eaten_by_reindeer initial_pinecones percent_eaten_by_reindeer
  let squirrel_eat := twice_eaten_by_squirrels reindeer_eat
  let after_eaten := initial_pinecones - reindeer_eat - squirrel_eat
  let fire_collect := collected_for_fires after_eaten percent_collected_for_fires
  let final_pinecones := after_eaten - fire_collect
  final_pinecones = 600 :=
by sorry

end pinecones_left_l92_92300


namespace distance_covered_at_40kmph_l92_92324

def total_distance : ℝ := 250
def speed_40 : ℝ := 40
def speed_60 : ℝ := 60
def total_time : ℝ := 5.2

theorem distance_covered_at_40kmph :
  ∃ (x : ℝ), (x / speed_40 + (total_distance - x) / speed_60 = total_time) ∧ x = 124 :=
  sorry

end distance_covered_at_40kmph_l92_92324


namespace integer_values_count_l92_92252

theorem integer_values_count (x : ℕ) (h1 : 5 < Real.sqrt x) (h2 : Real.sqrt x < 6) : 
  ∃ count : ℕ, count = 10 := 
by 
  sorry

end integer_values_count_l92_92252


namespace sin_90_eq_1_l92_92460

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92460


namespace each_sister_got_five_bars_l92_92649

-- Defining the initial number of granola bars.
def initial_bars : ℕ := 20

-- Defining the number of bars set aside for each day of the week.
def bars_set_aside : ℕ := 7

-- Defining the number of bars traded to friend Pete.
def bars_traded : ℕ := 3

-- Defining the number of sisters.
def number_of_sisters : ℕ := 2

-- Proving that each sister received 5 granola bars.
theorem each_sister_got_five_bars :
  let remaining_bars_after_week := initial_bars - bars_set_aside,
      remaining_bars_after_trade := remaining_bars_after_week - bars_traded,
      bars_per_sister := remaining_bars_after_trade / number_of_sisters
  in bars_per_sister = 5 := by
  sorry

end each_sister_got_five_bars_l92_92649


namespace min_cans_needed_l92_92368

theorem min_cans_needed (C : ℕ → ℕ) (H : C 1 = 15) : ∃ n, C n * n >= 64 ∧ ∀ m, m < n → C 1 * m < 64 :=
by
  sorry

end min_cans_needed_l92_92368


namespace determine_integers_with_equal_divisors_l92_92945

def is_positive_integer (n : ℕ) : Prop := n > 0

def has_equal_divisors (n : ℕ) : Prop :=
  let factors := n.factors
  let x := factors.count 2
  let y := factors.count 3
  let z := (factors.erase_all 2).erase_all 3
  x ≥ 1 ∧ y ≥ 1 ∧ 2 * (x * y * z.length) = (x + 1) * (y + 1) * z.length

theorem determine_integers_with_equal_divisors :
  ∀ n : ℕ, is_positive_integer n → has_equal_divisors n →
    ∃ k, n = 72 * k ∨ n = 108 * k ∧ (∀ (d : ℕ), k % d ≠ 0) :=
by
  sorry

end determine_integers_with_equal_divisors_l92_92945


namespace ellipse_foci_distance_l92_92967

theorem ellipse_foci_distance (x y : ℝ) (h : 9 * x^2 + y^2 = 36) : 
  let a := 6
      b := 2
      c := Real.sqrt (a^2 - b^2)
  in 2 * c = 8 * Real.sqrt 2 :=
by
  sorry

end ellipse_foci_distance_l92_92967


namespace parallel_lines_distance_l92_92644

noncomputable def l1 (x y : ℝ) : ℝ := x - y + 6
noncomputable def l2 (x y : ℝ) : ℝ := x - y + (2 / 3)

def distance_between_lines : ℝ := (|6 - (2 / 3)|) / Real.sqrt (1 + (-1)^2)

theorem parallel_lines_distance :
  ∀ (A B : ℝ), A = 1 ∧ B = -1 ∧ (l1 0 0) = 6 ∧ (l2 0 0) = (2 / 3) → distance_between_lines = 8 * Real.sqrt 2 / 3 :=
by
  intros A B h
  simp [distance_between_lines]
  sorry

end parallel_lines_distance_l92_92644


namespace common_root_of_two_equations_l92_92559

theorem common_root_of_two_equations (m x : ℝ) :
  (m * x - 1000 = 1001) ∧ (1001 * x = m - 1000 * x) → (m = 2001 ∨ m = -2001) :=
by
  sorry

end common_root_of_two_equations_l92_92559


namespace sin_90_eq_1_l92_92478

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92478


namespace locus_of_points_perpendicular_to_AB_l92_92079

theorem locus_of_points_perpendicular_to_AB
  (A B M : Point) (k : ℝ) :
  ∃ D : Point, ortho_proj A B M D ∧ (dist_sq A M - dist_sq M B) = k ↔ is_perpendicular_to (line_through A B) (locus M) :=
sorry

end locus_of_points_perpendicular_to_AB_l92_92079


namespace int_values_satisfy_condition_l92_92260

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l92_92260


namespace water_volume_correct_l92_92087

-- Define the conditions
def ratio_water_juice : ℕ := 5
def ratio_juice_water : ℕ := 3
def total_punch_volume : ℚ := 3  -- in liters

-- Define the question and the correct answer
def volume_of_water (ratio_water_juice ratio_juice_water : ℕ) (total_punch_volume : ℚ) : ℚ :=
  (ratio_water_juice * total_punch_volume) / (ratio_water_juice + ratio_juice_water)

-- The proof problem
theorem water_volume_correct : volume_of_water ratio_water_juice ratio_juice_water total_punch_volume = 15 / 8 :=
by
  sorry

end water_volume_correct_l92_92087


namespace sin_90_eq_1_l92_92474

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92474


namespace product_of_last_two_digits_l92_92665

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 8 = 0) : A * B = 32 :=
by
  sorry

end product_of_last_two_digits_l92_92665


namespace hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l92_92919

namespace CatchUpProblem

-- Part (a)
theorem hieu_catches_up_beatrice_in_5_minutes :
  ∀ (d_b_walked : ℕ) (relative_speed : ℕ) (catch_up_time : ℕ),
  d_b_walked = 5 / 6 ∧ relative_speed = 10 ∧ catch_up_time = 5 :=
sorry

-- Part (b)(i)
theorem probability_beatrice_hieu_same_place :
  ∀ (total_pairs : ℕ) (valid_pairs : ℕ) (probability : Rat),
  total_pairs = 3600 ∧ valid_pairs = 884 ∧ probability = 221 / 900 :=
sorry

-- Part (b)(ii)
theorem range_of_x_for_meeting_probability :
  ∀ (probability : Rat) (valid_pairs : ℕ) (total_pairs : ℕ) (lower_bound : ℕ) (upper_bound : ℕ),
  probability = 13 / 200 ∧ valid_pairs = 234 ∧ total_pairs = 3600 ∧ 
  lower_bound = 10 ∧ upper_bound = 120 / 11 :=
sorry

end CatchUpProblem

end hieu_catches_up_beatrice_in_5_minutes_probability_beatrice_hieu_same_place_range_of_x_for_meeting_probability_l92_92919


namespace smallest_composite_no_prime_factors_less_than_20_l92_92997

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92997


namespace cost_per_pound_correct_l92_92389

def flat_fee : ℝ := 5
def weight : ℝ := 5
def total_cost : ℝ := 9
def cost_per_pound : ℝ := (total_cost - flat_fee) / weight

theorem cost_per_pound_correct : cost_per_pound = 0.8 := 
by
  sorry

end cost_per_pound_correct_l92_92389


namespace robin_can_make_baggies_l92_92207

def robin_cookies_problem : Prop :=
  let chocolate_chip_cookies := 23
  let oatmeal_cookies := 25
  let cookies_per_baggie := 6
  let total_cookies := chocolate_chip_cookies + oatmeal_cookies
  (total_cookies // cookies_per_baggie) = 8

theorem robin_can_make_baggies : robin_cookies_problem :=
by
  -- Proof goes here
  sorry

end robin_can_make_baggies_l92_92207


namespace inverse_of_f_l92_92012

-- Define the original function
def f (x : ℝ) : ℝ := Real.log x + 1

-- Define the proposed inverse function
def f_inv (y : ℝ) : ℝ := Real.exp (y - 1)

-- State the theorem to prove the inverse relationship
theorem inverse_of_f : ∀ y, y ∈ ℝ → f (f_inv y) = y :=
by
  sorry

end inverse_of_f_l92_92012


namespace find_m_l92_92613

variable (a m x : ℝ)

noncomputable def quadratic_function : ℝ → ℝ := λ x, -a * x^2 + 2 * a * x + 3

theorem find_m (h1 : a > 0) (h2 : quadratic_function a m = 3) (h3 : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92613


namespace impossible_arrangement_of_natural_numbers_l92_92709

open Nat

theorem impossible_arrangement_of_natural_numbers :
  ¬ ∃ (a : Fin 1995 → ℕ), (∀ i, ∃ p, Prime p ∧ ((a i < a (i + 1) → a (i + 1) = p * a i) ∧ (a i > a (i + 1) → a i = p * a (i + 1)))) :=
sorry

end impossible_arrangement_of_natural_numbers_l92_92709


namespace num_of_integers_satisfying_sqrt_condition_l92_92277

theorem num_of_integers_satisfying_sqrt_condition : 
  let S := { x : ℤ | 5 < Real.sqrt x ∧ x < 36 }
  in (S.card = 10) :=
begin
  let S := { x : ℤ | 25 < x ∧ x < 36 },
  sorry
end

end num_of_integers_satisfying_sqrt_condition_l92_92277


namespace count_integers_between_25_and_36_l92_92279

theorem count_integers_between_25_and_36 :
  {x : ℤ | 25 < x ∧ x < 36}.finite.card = 10 :=
by
  sorry

end count_integers_between_25_and_36_l92_92279


namespace sum_of_angles_in_segments_outside_pentagon_l92_92903

theorem sum_of_angles_in_segments_outside_pentagon 
  (α β γ δ ε : ℝ) 
  (hα : α = 0.5 * (360 - arc_BCDE))
  (hβ : β = 0.5 * (360 - arc_CDEA))
  (hγ : γ = 0.5 * (360 - arc_DEAB))
  (hδ : δ = 0.5 * (360 - arc_EABC))
  (hε : ε = 0.5 * (360 - arc_ABCD)) 
  (arc_BCDE arc_CDEA arc_DEAB arc_EABC arc_ABCD : ℝ) :
  α + β + γ + δ + ε = 720 := 
by 
  sorry

end sum_of_angles_in_segments_outside_pentagon_l92_92903


namespace p_eq_q_l92_92019

def p (n : ℕ) : ℕ := (x y z w : ℕ) → (x + y + 2 * z + 3 * w = n - 1)

def q (n : ℕ) : ℕ := (a b c d : ℕ) → (a + b + c + d = n) ∧ (a ≥ b) ∧ (c ≥ d) ∧ (a ≥ d) ∧ (b < c)

theorem p_eq_q (n : ℕ) : p n = q n := sorry

end p_eq_q_l92_92019


namespace find_remainder_l92_92978

-- Definitions of the given polynomials
def f (x : ℝ) : ℝ := x^4 - x^3 + 1
def g (x : ℝ) : ℝ := x^2 - 4x + 6

-- Definition of the remainder
def r (x : ℝ) : ℝ := 6x - 35

-- The theorem to prove
theorem find_remainder :
  ∀ x : ℝ, ∃ q : ℝ → ℝ, f(x) = g(x) * q(x) + r(x) :=
by
  -- Polynomial division proof here (Skipped)
  sorry

end find_remainder_l92_92978


namespace elf_can_equalize_milk_l92_92024

def avg (a b : ℝ) : ℝ := (a + b) / 2

def elf_operation (cups : Fin 30 → ℝ) (i j : Fin 30) : Fin 30 → ℝ :=
  fun k => if k = i ∨ k = j then avg (cups i) (cups j) else cups k

theorem elf_can_equalize_milk (initial_milk : Fin 30 → ℝ) :
  ∃ (seq : List (Fin 30 × Fin 30)), 
    (seq.foldl (fun cups op => elf_operation cups op.1 op.2) initial_milk) = (fun _ => sorry) :=
sorry

end elf_can_equalize_milk_l92_92024


namespace complex_power_identity_l92_92002

theorem complex_power_identity (i : ℂ) (h1: i^4 = 1) (h2: i^2 = -1) : i^6 + i^{16} + i^{-26} = -1 := 
by sorry

end complex_power_identity_l92_92002


namespace design_with_rotational_symmetry_l92_92924

def has_90_degree_rotational_symmetry (design : Type) :=
  ∀ (rotate : ℤ → design → design), rotate 90 design = design

inductive SquareDesign
| horizontal_lines
| vertical_lines
| plus_sign
| diagonal_lines
| central_dot

open SquareDesign

theorem design_with_rotational_symmetry :
  has_90_degree_rotational_symmetry plus_sign :=
  sorry

end design_with_rotational_symmetry_l92_92924


namespace final_number_is_6218_l92_92199

theorem final_number_is_6218 :
  ∃ n : ℕ, (n = 6218) ∧ (∀ k, 3 ≤ k ∧ k ≤ 223 ∧ k % 4 = 3) ∧
  (∀ S : List ℕ, (∀ x ∈ S, 3 ≤ x ∧ x ≤ 223 ∧ x % 4 = 3) → list.sum S - 2 * (S.length - 1) = 6218) :=
begin
  sorry
end

end final_number_is_6218_l92_92199


namespace fourth_number_on_board_eighth_number_on_board_l92_92845

theorem fourth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 4 ∧ median numbers[3] = 2 :=
sorry

theorem eighth_number_on_board (medians : List ℚ) (hmed : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (numbers : List ℚ), numbers.length ≥ 8 ∧ median numbers[7] = 2 :=
sorry

end fourth_number_on_board_eighth_number_on_board_l92_92845


namespace shanghai_masters_total_matches_l92_92690

theorem shanghai_masters_total_matches {players groups : Type} 
  [fintype players] [fintype groups] (h1 : fintype.card players = 8) (h2 : fintype.card groups = 2) 
  (h3 : Π g : groups, fintype.card {p : players // p ∈ g} = 4) :
  count_matches_masters  = 16 := by sorry

-- Definitions required for the theorem
def count_matches_masters : ℕ :=
  let group_matches := 2 * 6 in     -- Matches in the round-robin stage
  let knockout_matches := 2 in      -- Matches in the knockout stage
  let final_matches := 2 in         -- Matches in the final stages
  group_matches + knockout_matches + final_matches

end shanghai_masters_total_matches_l92_92690


namespace equation_of_ellipse_area_of_triangle_l92_92583

noncomputable def a_gt_b_gt_0 (a b : ℝ) : Prop := a > b ∧ b > 0

noncomputable def point_in_ellipse (a b x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def midpoint (x1 y1 x2 y2 mx my : ℝ) : Prop := 
  mx = (x1 + x2) / 2 ∧ my = (y1 + y2) / 2

noncomputable def right_angle (angle : ℝ) : Prop := angle = Real.pi / 2

-- Given conditions
def conditions (a b : ℝ) (F1 F2 Q M P : ℝ × ℝ) : Prop :=
  a_gt_b_gt_0 a b ∧ 
  point_in_ellipse a b (Q.1) (Q.2) ∧ 
  midpoint Q.1 Q.2 F2.1 F2.2 M.1 M.2 ∧ 
  F1 = (- sqrt 2, 0) ∧ 
  F2 = (sqrt 2, 0) ∧
  point_in_ellipse a b P.1 P.2 ∧ 
  right_angle (Real.pi / 2)

-- Proof statement for the equation of the ellipse
theorem equation_of_ellipse (a b : ℝ) (F1 F2 Q M P : ℝ × ℝ) (h : conditions a b F1 F2 Q M P) :
  (a = 2) ∧ (b = sqrt 2) ∧ ((x:ℝ)⁻²/(2:ℝ)⁻² + (y:ℝ)⁻²/(sqrt 2)^(-2) = 1) :=
by
  sorry

-- Proof statement for the area of triangle F1PF2
theorem area_of_triangle (a b : ℝ) (F1 F2 Q M P : ℝ × ℝ) (h : conditions a b F1 F2 Q M P) :
  right_angle (Real.pi / 2) ∧ 
  P = (-1, 1) ∧ 
  (1/2 * 2 * 2 = 2) :=
by
  sorry

end equation_of_ellipse_area_of_triangle_l92_92583


namespace route_y_slower_than_route_x_l92_92762

theorem route_y_slower_than_route_x :
  let t_X := (12 / 45) * 60,
      t_Y1 := (9 / 50) * 60,
      t_Y2 := (1 / 10) * 60,
      t_Y := t_Y1 + t_Y2 in
  t_X - t_Y = -0.8 :=
by {
  let t_X := (12 / 45) * 60,
  let t_Y1 := (9 / 50) * 60,
  let t_Y2 := (1 / 10) * 60,
  let t_Y := t_Y1 + t_Y2,
  calc
  t_X - t_Y = 16 - 16.8 : by sorry
           ... = -0.8 : by sorry
}

end route_y_slower_than_route_x_l92_92762


namespace digit_five_occurrences_l92_92192

/-- 
  Define that a 24-hour digital clock display shows times containing at least one 
  occurrence of the digit '5' a total of 450 times in a 24-hour period.
--/
def contains_digit_five (n : Nat) : Prop := 
  n / 10 = 5 ∨ n % 10 = 5

def count_times_with_digit_five : Nat :=
  let hours_with_five := 2 * 60  -- 05:00-05:59 and 15:00-15:59, each hour has 60 minutes
  let remaining_hours := 22 * 15 -- 22 hours, each hour has 15 minutes
  hours_with_five + remaining_hours

theorem digit_five_occurrences : count_times_with_digit_five = 450 := by
  sorry

end digit_five_occurrences_l92_92192


namespace minimum_value_x3_plus_y3_minus_5xy_l92_92032

theorem minimum_value_x3_plus_y3_minus_5xy (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  x^3 + y^3 - 5 * x * y ≥ -125 / 27 := 
sorry

end minimum_value_x3_plus_y3_minus_5xy_l92_92032


namespace sin_90_deg_l92_92457

theorem sin_90_deg : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_90_deg_l92_92457


namespace horizontal_asymptote_l92_92664

def numerator : ℚ[X] := 15 * X ^ 4 + 6 * X ^ 3 + 5 * X ^ 2 + 4 * X + 2
def denominator : ℚ[X] := 5 * X ^ 4 + 3 * X ^ 3 + 9 * X ^ 2 + 2 * X + 1

theorem horizontal_asymptote :
  ∀ x : ℚ, (∃ y : ℚ, y = numerator.coeff 4 / denominator.coeff 4 ∧ y = 3) :=
by
  sorry

end horizontal_asymptote_l92_92664


namespace number_of_integers_between_25_and_36_l92_92262

theorem number_of_integers_between_25_and_36 :
  {n : ℕ | 25 < n ∧ n < 36}.card = 10 :=
by
  sorry

end number_of_integers_between_25_and_36_l92_92262


namespace num_of_integers_satisfying_sqrt_condition_l92_92273

theorem num_of_integers_satisfying_sqrt_condition : 
  let S := { x : ℤ | 5 < Real.sqrt x ∧ x < 36 }
  in (S.card = 10) :=
begin
  let S := { x : ℤ | 25 < x ∧ x < 36 },
  sorry
end

end num_of_integers_satisfying_sqrt_condition_l92_92273


namespace log_expression_value_l92_92383

noncomputable def log := Real.log10

theorem log_expression_value :
  (log 8 + log 125 - log 2 - log 5) / (log (Real.sqrt 10) * log 0.1) = -4 :=
by
  have h1 : log 8 = 3 * log 2 := sorry
  have h2 : log 125 = 3 * log 5 := sorry
  have h3 : log 0.1 = log (1 / 10) := sorry
  have h4 : log (Real.sqrt 10) = 0.5 * log 10 := sorry
  sorry

end log_expression_value_l92_92383


namespace fourth_number_is_two_eighth_number_is_two_l92_92838

theorem fourth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 4 ∧ board !! 3 = some 2 :=
by
  sorry

theorem eighth_number_is_two
  (notebook : List ℚ)
  (h_notebook : notebook = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]) :
  ∃ (board : List ℚ), board.length ≥ 8 ∧ board !! 7 = some 2 :=
by
  sorry

end fourth_number_is_two_eighth_number_is_two_l92_92838


namespace similar_triangles_and_angle_sum_l92_92744

variable (ABCD : Type) [rectangle ABCD]
variable (A B C D E P F G : ABCD) 
variable (h1 : BC = 2 * AB)
variable (h2 : E = midpoint B C)
variable (h3 : P ∈ AD)
variable (h4 : is_perpendicular (F, A) (B, P))
variable (h5 : is_perpendicular (G, D) (C, P))
variable (h6 : angle B P C = 85)

theorem similar_triangles_and_angle_sum :
  (similar (triangle B E F) (triangle B E P)) ∧ (angle B E F + angle C E G = 85) :=
by
  sorry

end similar_triangles_and_angle_sum_l92_92744


namespace best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92151

-- Part (a)
theorem best_play_wins_majority_two_classes (n : ℕ) :
  let prob_tie := (1 / 2) ^ n in
  1 - prob_tie = 1 - (1 / 2) ^ n :=
sorry

-- Part (b)
theorem best_play_wins_majority_multiple_classes (n s : ℕ) :
  let prob_tie := (1 / 2) ^ ((s - 1) * n) in
  1 - prob_tie = 1 - (1 / 2) ^ ((s - 1) * n) :=
sorry

end best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92151


namespace sin_90_eq_one_l92_92415

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92415


namespace find_x_from_dot_product_l92_92059

theorem find_x_from_dot_product :
  let a := (-3 : ℤ, 2 : ℤ, 5 : ℤ)
  let b := (1 : ℤ, x : ℤ, -1 : ℤ)
  (a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 2) → x = 5 := by
  intros a b h
  sorry

end find_x_from_dot_product_l92_92059


namespace temperature_drop_l92_92288

-- Define the initial temperature and the drop in temperature
def initial_temperature : ℤ := -6
def drop : ℤ := 5

-- Define the resulting temperature after the drop
def resulting_temperature : ℤ := initial_temperature - drop

-- The theorem to be proved
theorem temperature_drop : resulting_temperature = -11 :=
by
  sorry

end temperature_drop_l92_92288


namespace bucket_proof_l92_92380

variable (CA : ℚ) -- capacity of Bucket A
variable (CB : ℚ) -- capacity of Bucket B
variable (SA_init : ℚ) -- initial amount of sand in Bucket A
variable (SB_init : ℚ) -- initial amount of sand in Bucket B

def bucket_conditions : Prop := 
  CB = (1 / 2) * CA ∧
  SA_init = (1 / 4) * CA ∧
  SB_init = (3 / 8) * CB

theorem bucket_proof (h : bucket_conditions CA CB SA_init SB_init) : 
  (SA_init + SB_init) / CA = 7 / 16 := 
  by sorry

end bucket_proof_l92_92380


namespace teacup_lids_arrangement_l92_92821

theorem teacup_lids_arrangement :
  let n := 6 in
  ∑ k in (finset.range (n + 1)).filter (λ k, k = 2),
    nat.choose n k *
    (nat.factorial (n - k)) *
    ((derangements (n - k))) = 135 :=
by
  sorry

#check teacup_lids_arrangement

end teacup_lids_arrangement_l92_92821


namespace general_term_T_value_l92_92604

-- Given conditions
def f (x : ℝ) : ℝ := (2 * x + 3) / (3 * x)

def a : ℕ → ℝ
| 0       := 1
| (n + 1) := f (1 / (a n))

-- General term of sequence aₙ
theorem general_term (n : ℕ) : a n = (2 * n + 1) / 3 :=
  sorry

-- Sequence Tₙ
def T (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, (a (2 * k + 1) * a (2 * k) - a (2 * k + 2) * a (2 * k + 1))

-- Value for Tₙ
theorem T_value (n : ℕ) : T n = - ((2 * n^2 + 3 * n) / 3) :=
  sorry

end general_term_T_value_l92_92604


namespace angle_product_condition_l92_92751
noncomputable theory

variables {A B C D P : Point}
variables {α β γ δ : ℝ}

def convex_quadrilateral (ABCD : Quadrilateral) : Prop := convex ABCD

def intersect_at_P (s1 s2 : Line) (P : Point) : Prop := intersects s1 s2 P

def internal_angles (α β γ δ : ℝ) 
  (α_angle : Angle A P D = α) (β_angle : Angle B P C = β) 
  (γ_angle : Angle D P A = γ) (δ_angle : Angle C P B = δ) : Prop := 
  true

theorem angle_product_condition 
  (h_convex : convex_quadrilateral ABCD)
  (h_intersect : intersect_at_P (line_through A B) (line_through C D) P)
  (h_angles : internal_angles α β γ δ (Angle A P D) (Angle B P C) (Angle D P A) (Angle C P B)) :
  α + γ = β + δ ↔ (distance P A * distance P B = distance P C * distance P D) :=
by sorry

end angle_product_condition_l92_92751


namespace total_items_l92_92304

theorem total_items (B M C : ℕ) 
  (h1 : B = 58) 
  (h2 : B = M + 18) 
  (h3 : B = C - 27) : 
  B + M + C = 183 :=
by 
  sorry

end total_items_l92_92304


namespace polyhedra_overlap_l92_92180

-- Definitions for the polyhedron and translations
universe u
variables {α : Type u} [EuclideanSpace α]
variables (P1 : Polyhedron α) 
variable [h_convex : ConvexPolyhedron P1]
variables (A : Fin 9 → α)
variable (P : Fin 9 → Polyhedron α)

-- Condition: P1 is the convex polyhedron with vertices A_1, A_2, ..., A_9
def polyhedron_with_vertices (P1 : Polyhedron α) (A : Fin 9 → α) : Prop :=
  P1.vertices = Set.range A

-- Condition: Pi obtained from P1 by translating A_1 to A_i
def translated_polyhedron (P1 : Polyhedron α) (A : Fin 9 → α) (i : Fin 9) : Polyhedron α :=
  P1.translate (A i - A 0)

-- Theorem to prove: at least two of the polyhedra P1, P2, ..., P9 have an interior point in common
theorem polyhedra_overlap
  (hP1 : polyhedron_with_vertices P1 A)
  (hP : ∀ i, P i = translated_polyhedron P1 A i) :
  ∃ i j, i ≠ j ∧ (P i).interior ∩ (P j).interior ≠ ∅ :=
sorry

end polyhedra_overlap_l92_92180


namespace probability_within_d_l92_92907

noncomputable def d : ℝ := 0.3

theorem probability_within_d (d : ℝ) : 0 < d ∧ ∃ (square : set (ℝ × ℝ)), 
  (0, 0) ∈ square ∧ (3030, 0) ∈ square ∧ (3030, 3030) ∈ square ∧ (0, 3030) ∈ square ∧
  (∀ p ∈ square, (probability (within_d_of_lattice_point p) = 1 / 3)) :=
by
  sorry

end probability_within_d_l92_92907


namespace probability_at_least_two_black_balls_l92_92111

theorem probability_at_least_two_black_balls
  (white_balls : ℕ) (black_balls : ℕ) (drawn_balls : ℕ) 
  (total_balls := white_balls + black_balls)
  (p2_black := (choose black_balls 2) * (choose white_balls 1) + (choose black_balls 3))
  (total_ways := choose total_balls drawn_balls):
  white_balls = 5 → 
  black_balls = 3 → 
  drawn_balls = 3 → 
  (p2_black / total_ways) = 2 / 7 :=
by
  sorry

end probability_at_least_two_black_balls_l92_92111


namespace problem_sums_greater_than_one_l92_92321

theorem problem_sums_greater_than_one : 
  let A := (1 / 4, 2 / 8, 3 / 4)
  let B := (3, -1.5, -0.5)
  let C := (0.25, 0.75, 0.05)
  let D := (3 / 2, -3 / 4, 1 / 4)
  let E := (1.5, 1.5, -2)
  (A.1 + A.2 + A.3 > 1) ∧ (C.1 + C.2 + C.3 > 1) := 
  by
  sorry

end problem_sums_greater_than_one_l92_92321


namespace incorrect_statement_about_factors_multiples_l92_92934

theorem incorrect_statement_about_factors_multiples : ¬(56 ÷ 7 = 8 → (∃ m n : ℕ, 56 = m * n ∧ 7 = n) ∧ (∃ m n : ℕ, 7 = m * n ∧ 56 = n)) := 
by
  sorry

end incorrect_statement_about_factors_multiples_l92_92934


namespace sin_90_eq_1_l92_92480

theorem sin_90_eq_1 : Real.sin (Float.pi / 2) = 1 := by
  sorry

end sin_90_eq_1_l92_92480


namespace plane_distance_last_10_seconds_l92_92787

theorem plane_distance_last_10_seconds (s : ℝ → ℝ) (h : ∀ t, s t = 60 * t - 1.5 * t^2) : 
  s 20 - s 10 = 150 := 
by 
  sorry

end plane_distance_last_10_seconds_l92_92787


namespace cube_placement_l92_92114

theorem cube_placement (n : ℕ) (points : Set (ℝ × ℝ × ℝ)) (h1 : n = 13) (h2 : points.finite) (h3 : points.to_finset.card = 1956) :
  ∃ (small_cube : Set (ℝ × ℝ × ℝ)), (∀ (x y z : ℝ), small_cube = { p | p.1 = x ∧ p.2 = y ∧ p.3 = z ∧ x >= 0 ∧ x < 1 ∧ y >= 0 ∧ y < 1 ∧ z >= 0 ∧ z < 1 }) ∧ ∀ p ∈ points, p ∉ small_cube :=
begin
  sorry
end

end cube_placement_l92_92114


namespace sin_90_degree_l92_92498

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92498


namespace power_of_two_exists_multiple_digits_1_2_l92_92774

noncomputable def sequence (k : ℕ) : ℕ :=
if k = 1 then 2 else
let a_k := sequence (k - 1) in
let x := 10^(k-1) + a_k in
let y := 2 * 10^(k-1) + a_k in
if (x % 2^k = 0) then x else y

theorem power_of_two_exists_multiple_digits_1_2 :
  ∀ (k : ℕ), ∃ (a_k : ℕ), (a_k.digits 10).foldr (λ d acc, d = 1 ∨ d = 2 ∧ acc) True ∧ a_k % 2^k = 0 :=
by
  intro k
  use sequence k
  induction k with n hn
  case zero =>
    use 2
    split
    case left => trivial
    case right => trivial
  case succ =>
    split
    case left =>
      sorry
    case right =>
      sorry

end power_of_two_exists_multiple_digits_1_2_l92_92774


namespace probability_sum_is_4_l92_92814

open Real

def rounding (x : ℝ) : ℤ :=
  if x - floor x < 0.5 then floor x else ceil x

theorem probability_sum_is_4 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 3.5)
  (hx : 0.5 ≤ x ∧ x < 1.5 ∨ 1.5 ≤ x ∧ x < 2) :
  let rounded_sum := rounding x + rounding (3.5 - x) 
  in (rounded_sum = 4) → (3 / 7 : ℝ) :=
sorry

end probability_sum_is_4_l92_92814


namespace binary_arith_proof_l92_92382

theorem binary_arith_proof :
  let a := 0b1101110  -- binary representation of 1101110_2
  let b := 0b101010   -- binary representation of 101010_2
  let c := 0b100      -- binary representation of 100_2
  (a * b / c) = 0b11001000010 :=  -- binary representation of the final result
by
  sorry

end binary_arith_proof_l92_92382


namespace prob_xi_neg2_to_0_l92_92750

noncomputable def xi : ℝ → ℝ := sorry -- Placeholder for the normal distribution random variable definition
noncomputable def μ : ℝ := sorry -- Placeholder for the mean of the normal distribution
noncomputable def σ : ℝ := sorry -- Placeholder for the standard deviation of the normal distribution

axiom xi_normal_distribution : xi ~ Normal μ σ^2

axiom symmetry_condition : (prob (λ x, x < -1) xi) = (prob (λ x, x > 1) xi)
axiom prob_xi_gt_2 : (prob (λ x, x > 2) xi) = 0.3

theorem prob_xi_neg2_to_0 : (prob (λ x, -2 < x ∧ x < 0) xi) = 0.2 :=
by
  sorry

end prob_xi_neg2_to_0_l92_92750


namespace no_true_propositions_l92_92572

-- Definitions of lines and planes
variables (a b : Line) (alpha : Plane)

-- Definitions of propositions
def prop1 := (a ∥ b) ∧ (b ⊂ alpha) → a ∥ alpha
def prop2 := (a ∥ b) ∧ (b ∥ alpha) → a ∥ alpha
def prop3 := (a ∥ alpha) ∧ (b ∥ alpha) → a ∥ b

-- Main statement
theorem no_true_propositions : (¬ prop1) ∧ (¬ prop2) ∧ (¬ prop3) :=
by sorry

end no_true_propositions_l92_92572


namespace election_votes_l92_92120

theorem election_votes (V : ℕ) (h : 0.70 * V - 0.30 * V = 320) : V = 800 :=
by
  sorry

end election_votes_l92_92120


namespace monomial_sum_mn_l92_92097

-- Define the conditions as Lean definitions
def is_monomial_sum (x y : ℕ) (m n : ℕ) : Prop :=
  ∃ k : ℕ, (x ^ 2) * (y ^ m) + (x ^ n) * (y ^ 3) = x ^ k

-- State our main theorem
theorem monomial_sum_mn (x y : ℕ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
sorry  -- Completion of the proof is not required

end monomial_sum_mn_l92_92097


namespace weight_of_replaced_person_l92_92224

theorem weight_of_replaced_person (avg_increase : ℝ) (num_people : ℕ) 
(new_person : ℝ) (weight_increase : ℝ) :
  (avg_increase = 2.5) →
  (num_people = 8) →
  (new_person = 70) →
  (weight_increase = num_people * avg_increase) →
  (weight_of_replaced_person = new_person - weight_increase) →
  weight_of_replaced_person = 50 :=
by {
  intros,
  sorry
}

end weight_of_replaced_person_l92_92224


namespace subset_cardinality_l92_92721

theorem subset_cardinality {m n : ℕ} (hm : 0 < m) (hn : 0 < n) (A : Finset (Fin (n+1))) (B : Finset (Fin (m+1))) (S : Finset (Fin (n+1) × Fin (m+1))) :
  (∀ (a b x y : (Fin (n+1) × Fin (m+1))), (a, b) ∈ S → (x, y) ∈ S → (a - x) * (b - y) ≤ 0) →
  S.card ≤ n + m - 1 :=
by
  sorry

end subset_cardinality_l92_92721


namespace count_satisfying_polynomials_l92_92653

open Polynomial

def polynomial_integer_coeff_deg_leq_5 (P : Polynomial ℤ) : Prop := 
  degree P ≤ 5

def polynomial_range_conditions (P : Polynomial ℤ) : Prop := 
  ∀ x ∈ ({0, 1, 2, 3, 4, 5} : Finset ℕ), 0 ≤ P.eval x ∧ P.eval x < 120

theorem count_satisfying_polynomials :
  let S := { P : Polynomial ℤ | polynomial_integer_coeff_deg_leq_5 P ∧ polynomial_range_conditions P } 
  in S.card = 86400000 :=
sorry

end count_satisfying_polynomials_l92_92653


namespace max_sum_sin2_eq_l92_92031

-- Defining the conditions given in the problem
variables (n : ℕ) (a : ℝ)
variable (x : ℝ^n)
variable (h1 : 0 ≤ a ∧ a ≤ n)
variable (h2 : ∑ i in finset.range n, (real.sin (x i))^2 = a)

-- Statement: The maximum value of ∑ (sin 2x_i) is 2 * sqrt(a * (n - a))
theorem max_sum_sin2_eq : 
  | ∑ i in finset.range n, real.sin (2 * x i) | ≤ 2 * real.sqrt (a * (n - a)) :=
sorry

end max_sum_sin2_eq_l92_92031


namespace smallest_composite_no_prime_factors_less_than_20_l92_92998

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92998


namespace smallest_composite_no_prime_factors_less_than_20_l92_92982

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92982


namespace smallest_prime_divides_sum_l92_92662

theorem smallest_prime_divides_sum :
  ∃ a, Prime a ∧ a ∣ (3 ^ 11 + 5 ^ 13) ∧
       ∀ b, Prime b → b ∣ (3 ^ 11 + 5 ^ 13) → a ≤ b :=
sorry

end smallest_prime_divides_sum_l92_92662


namespace smallest_composite_no_prime_factors_less_than_20_l92_92996

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92996


namespace kelseys_sister_age_in_2021_l92_92717

-- Definitions based on given conditions
def kelsey_birth_year : ℕ := 1999 - 25
def sister_birth_year : ℕ := kelsey_birth_year - 3

-- Prove that Kelsey's older sister is 50 years old in 2021
theorem kelseys_sister_age_in_2021 : (2021 - sister_birth_year) = 50 :=
by
  -- Add proof here
  sorry

end kelseys_sister_age_in_2021_l92_92717


namespace same_cost_number_of_guests_l92_92215

theorem same_cost_number_of_guests (x : ℕ) : 
  (800 + 30 * x = 500 + 35 * x) ↔ (x = 60) :=
by {
  sorry
}

end same_cost_number_of_guests_l92_92215


namespace reflection_through_plane_l92_92168

def normal_vector := !![2, -1, 1]
def reflection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  !![
    [-10 / 6, 16 / 6, 8 / 6],
    [  8 / 6,  4 / 6, 14 / 6],
    [  8 / 6, 10 / 6,  4 / 6]
  ]

theorem reflection_through_plane (u : Matrix (Fin 3) (Fin 1) ℚ) :
    let plane_point := !![1, 1, 1]
    let n := normal_vector
    let proj := (2 * u 0 0 - u 1 0 + u 2 0) / (2 * 2 + -1 * -1 + 1 * 1)  -- (u ⋅ n) / (n ⋅ n)
    let q := u - proj * !![n]  -- projection of u onto plane Q
    let reflection_u := 2 * q - u  -- reflection of u through plane
  in reflection_matrix * u = reflection_u :=
by
  -- Proof omitted
  sorry

end reflection_through_plane_l92_92168


namespace volume_ratio_l92_92937

-- Given constants representing the container volumes and the conditions
variables (A B : ℝ)

-- Define the conditions as given in the problem:
def first_container_full : ℝ := (2/3) * A
def second_container_after_pour : ℝ := (5/8) * B

-- Problem statement
theorem volume_ratio : first_container_full A = second_container_after_pour B → (A / B) = (15 / 16) :=
by
  intro h
  have : (2 / 3) * A = (5 / 8) * B := h
  sorry

end volume_ratio_l92_92937


namespace find_m_l92_92609

variable (a m x : ℝ)

noncomputable def quadratic_function : ℝ → ℝ := λ x, -a * x^2 + 2 * a * x + 3

theorem find_m (h1 : a > 0) (h2 : quadratic_function a m = 3) (h3 : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92609


namespace perimeter_of_piece_divided_by_a_l92_92285

theorem perimeter_of_piece_divided_by_a (a : ℝ) :
  let intersection_right := (a, (2 * a) / 3)
      intersection_left := (-a, (-2 * a) / 3)
      vertical_length := (2 * a) - (-(2 * a)) / 3
      horizontal_length := 2 * a
      hypotenuse_length := real.sqrt ((2 * a)^2 + ((4 * a) / 3)^2) / 3
      diagonal_length := a * real.sqrt 2 in
  vertical_length + horizontal_length + hypotenuse_length + diagonal_length / a = 6 + ((2 * real.sqrt 13 + 3 * real.sqrt 2) / 3) :=
sorry

end perimeter_of_piece_divided_by_a_l92_92285


namespace count_valid_A_l92_92086

theorem count_valid_A : 
  ∃! (count : ℕ), count = 4 ∧ ∀ A : ℕ, (1 ≤ A ∧ A ≤ 9) → 
  (∃ x1 x2 : ℕ, x1 + x2 = 2 * A + 1 ∧ x1 * x2 = 2 * A ∧ x1 > 0 ∧ x2 > 0) → A = 1 ∨ A = 2 ∨ A = 3 ∨ A = 4 :=
sorry

end count_valid_A_l92_92086


namespace probability_odd_sum_l92_92714

noncomputable def X : ℕ → Prop := λ n, n >= 1 ∧ n <= 4
noncomputable def Y : ℕ → Prop := λ n, n >= 1 ∧ n <= 3
noncomputable def Z : ℕ → Prop := λ n, n >= 1 ∧ n <= 5

theorem probability_odd_sum :
  (∃ x ∈ {1, 2, 3, 4}, ∃ y ∈ {1, 2, 3}, ∃ z ∈ {1, 2, 3, 4, 5}, 
   is_even (x + y + z) = false) / (4 * 3 * 5) = 2 / 5 :=
sorry

end probability_odd_sum_l92_92714


namespace KarenEggRolls_l92_92195

-- Definitions based on conditions
def OmarEggRolls : ℕ := 219
def TotalEggRolls : ℕ := 448

-- The statement to be proved
theorem KarenEggRolls : (TotalEggRolls - OmarEggRolls = 229) :=
by {
    -- Proof step goes here
    sorry
}

end KarenEggRolls_l92_92195


namespace midpoint_of_interception_l92_92543

theorem midpoint_of_interception (x1 x2 y1 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2) 
  (h3 : y1 = x1 - 1) 
  (h4 : y2 = x2 - 1) : 
  ( (x1 + x2) / 2, (y1 + y2) / 2 ) = (3, 2) :=
by 
  sorry

end midpoint_of_interception_l92_92543


namespace trains_cross_time_approx_9_secs_l92_92341

-- Definitions of the conditions
def train1_length : ℝ := 230
def train1_speed_km_per_hr : ℝ := 120

def train2_length : ℝ := 270.04
def train2_speed_km_per_hr : ℝ := 80

-- Conversion factor from km/hr to m/s
def km_per_hr_to_m_per_s (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Speeds in m/s
def train1_speed_m_per_s := km_per_hr_to_m_per_s train1_speed_km_per_hr
def train2_speed_m_per_s := km_per_hr_to_m_per_s train2_speed_km_per_hr

-- Relative speed considering opposite directions
def relative_speed := train1_speed_m_per_s + train2_speed_m_per_s

-- Total distance to be covered
def total_distance := train1_length + train2_length

-- Time taken for the trains to cross each other
def time_to_cross := total_distance / relative_speed

-- The theorem to be proved
theorem trains_cross_time_approx_9_secs : abs (time_to_cross - 9) < 0.1 := by
  sorry

end trains_cross_time_approx_9_secs_l92_92341


namespace max_elements_in_union_l92_92588

theorem max_elements_in_union (A B : Finset ℕ) (h_cond1: A ⊆ (Finset.range 100).map (λ x, x + 1)) 
  (h_cond2: B ⊆ (Finset.range 100).map (λ x, x + 1)) 
  (h_size: A.card = B.card) 
  (h_disjoint: A ∩ B = ∅) 
  (h_imp: ∀ x, x ∈ A → 2 * x + 2 ∈ B) : 
  (A ∪ B).card ≤ 66 :=
sorry

end max_elements_in_union_l92_92588


namespace sum_of_last_three_coefficients_l92_92870

open scoped BigOperators

/-- The sum of the last three coefficients of the polynomial expansion of (1 - 2/x)^7 is 29. -/
theorem sum_of_last_three_coefficients : 
  let f := λ (x : ℚ), (1 - 2 / x)
  let coeffs := (∑ k in Finset.range 8, (choose 7 k) * (-2:ℚ) ^ k * x ^ (7 - k))
  let last_three := (coeffs.coeff 0) + (coeffs.coeff 1) + (coeffs.coeff 2)
  last_three = (29:ℚ) :=
by 
  sorry

end sum_of_last_three_coefficients_l92_92870


namespace area_of_triangle_cef_l92_92328

theorem area_of_triangle_cef :
  ∀ (abcd : Type) [metric_space abcd] (a b c d e f : abcd),
  -- conditions
  (is_square abcd a b c d) → 
  (dist a b = 8) →
  (midpoint a b = f) →
  (midpoint a d = e) →
  -- question
  (triangle_area c e f = 16) :=
by
  sorry

end area_of_triangle_cef_l92_92328


namespace nth_equation_l92_92764

-- Define the product of a list of integers
def prod_list (lst : List ℕ) : ℕ :=
  lst.foldl (· * ·) 1

-- Define the product of first n odd numbers
def prod_odds (n : ℕ) : ℕ :=
  prod_list (List.map (λ i => 2 * i - 1) (List.range n))

-- Define the product of the range from n+1 to 2n
def prod_range (n : ℕ) : ℕ :=
  prod_list (List.range' (n + 1) n)

-- The theorem to prove
theorem nth_equation (n : ℕ) (hn : 0 < n) : prod_range n = 2^n * prod_odds n := 
  sorry

end nth_equation_l92_92764


namespace min_F_value_range_of_m_l92_92735

def f (x : ℝ) := x * Real.exp x
def g (x : ℝ) := 1/2 * x^2 + x
def F (x : ℝ) := f x + g x

theorem min_F_value : ∃ x, F x = -1 - 1 / Real.exp 1 :=
sorry

theorem range_of_m (m : ℝ) : (∀ x1 x2 : ℝ, x1 ∈ Ici (-1 : ℝ) → x2 ∈ Ici (-1 : ℝ) → x1 > x2 → m * (f x1 - f x2) > g x1 - g x2) ↔ m ≥ Real.exp 1 :=
sorry

end min_F_value_range_of_m_l92_92735


namespace sqrt_equation_solution_l92_92549

theorem sqrt_equation_solution (x : ℝ) (h : sqrt (2 * x + 9) = 11) : x = 56 := 
by 
  sorry

end sqrt_equation_solution_l92_92549


namespace best_play_wins_majority_l92_92132

/-- Probability that the best play wins with a majority of the votes given the conditions -/
theorem best_play_wins_majority (n : ℕ) :
  let p := 1 - (1 / 2)^n
  in p > (1 - (1 / 2)^n) ∧ p ≤ 1 :=
sorry

end best_play_wins_majority_l92_92132


namespace trigonometric_identity_sum_to_product_l92_92795

theorem trigonometric_identity_sum_to_product :
  ∃ (a b c d : ℕ), 
    (∀ x, cos (2 * x) + cos (6 * x) + cos (10 * x) + cos (14 * x) = (a * cos (b * x) * cos (c * x) * cos (d * x)))
    ∧ a + b + c + d = 18 :=
sorry

end trigonometric_identity_sum_to_product_l92_92795


namespace angle_FHG_is_120_degrees_l92_92830

theorem angle_FHG_is_120_degrees
  (P Q F G H : Point)
  (circle1 circle2 : Circle)
  (h1 : ∀ {C C' : Circle}, congruent C C' → C.center ∈ C' ∧ C'.center ∈ C)
  (h2 : line_through P Q ∧ line_through P Q contains_point F ∧ line_through P Q contains_point G)
  (h3 : ¬(set_inter (circle1.points) (circle2.points) = {H}))
  (h4 : PQ = QH ∧ QH = PH)
  (h5 : PF = FQ) :
  angle F H G = 120 :=
begin
  sorry
end

end angle_FHG_is_120_degrees_l92_92830


namespace best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92148

-- Part (a)
theorem best_play_wins_majority_two_classes (n : ℕ) :
  let prob_tie := (1 / 2) ^ n in
  1 - prob_tie = 1 - (1 / 2) ^ n :=
sorry

-- Part (b)
theorem best_play_wins_majority_multiple_classes (n s : ℕ) :
  let prob_tie := (1 / 2) ^ ((s - 1) * n) in
  1 - prob_tie = 1 - (1 / 2) ^ ((s - 1) * n) :=
sorry

end best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92148


namespace max_prime_sequence_l92_92964

open Nat

def sequence_max_prime (k : ℕ) : ℕ :=
  (finset.range 100).countp (λ i, prime (k + i))

theorem max_prime_sequence : ∀ k ≥ 1, sequence_max_prime 2 ≥ sequence_max_prime k :=
begin
  sorry
end

end max_prime_sequence_l92_92964


namespace sin_90_eq_1_l92_92445

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92445


namespace meghal_game_max_score_l92_92757

theorem meghal_game_max_score :
  (∑ n in Finset.range 2016.succ, if n % 2 = 0 then n + 2 else n + 1) = 1019088 :=
sorry

end meghal_game_max_score_l92_92757


namespace time_after_2405_minutes_from_midnight_l92_92289

-- Define the initial condition: time is midnight
def midnight : Nat := 0

-- Define the function that calculates time after a certain number of minutes from midnight
noncomputable def time_after_minutes (m : Nat) : String :=
  let total_minutes := m % 1440  -- Calculate minutes in a 24-hour period (1440 minutes)
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  let period := if hours < 12 then "a.m." else "p.m."
  let adjusted_hours := if hours == 0 then 12 else if hours <= 12 then hours else hours - 12
  s!"{adjusted_hours}:{minutes:02d} {period}"

-- Define the theorem to prove the time after 2405 minutes from midnight
theorem time_after_2405_minutes_from_midnight : 
  time_after_minutes 2405 = "4:05 p.m." :=
by
  -- Skip the full proof for now
  sorry

end time_after_2405_minutes_from_midnight_l92_92289


namespace cosine_alpha_l92_92055

-- Conditions
def alpha (α : ℝ) : Prop := 0 < α ∧ α < π / 2
def cos_sum (α : ℝ) : Prop := cos (π / 3 + α) = 1 / 3

-- Statement to be proved
theorem cosine_alpha (α : ℝ) (h1 : alpha α) (h2 : cos_sum α) : cos α = (2 * sqrt 6 + 1) / 6 :=
sorry

end cosine_alpha_l92_92055


namespace max_perimeter_is_49_l92_92203

def triangle := {a b c : ℕ // a + b > c ∧ b + c > a ∧ c + a > b}

def perimeter (t : triangle) : ℕ := t.val.a + t.val.b + t.val.c

noncomputable def max_perimeter (t1 t2 t3 : triangle) : ℕ :=
  let initial_perimeter := perimeter t1 + perimeter t2 + perimeter t3
  let joinable_sides := [(t1.val.a, t2.val.a), (t1.val.a, t3.val.a), (t2.val.a, t3.val.a),
                         (t1.val.b, t2.val.b), (t1.val.b, t3.val.b), (t2.val.b, t3.val.b),
                         (t1.val.c, t2.val.c), (t1.val.c, t3.val.c), (t2.val.c, t3.val.c)]
  let shared_side_subtraction := joinable_sides.foldl (fun acc (x, y) => if x = y then acc + 2 * x else acc) 0
  initial_perimeter - shared_side_subtraction

def triangle1 : triangle := ⟨(5,8,10), by simp [triangle]; linarith⟩
def triangle2 : triangle := ⟨(5,10,12), by simp [triangle]; linarith⟩
def triangle3 : triangle := ⟨(5,8,12), by simp [triangle]; linarith⟩

theorem max_perimeter_is_49 : max_perimeter triangle1 triangle2 triangle3 = 49 :=
by
  sorry

end max_perimeter_is_49_l92_92203


namespace average_discount_rate_correct_l92_92895

-- Define the marked and sold prices
def marked_prices : List ℝ := [80, 120, 150, 40, 50]
def sold_prices : List ℝ := [68, 96, 135, 32, 45]

-- Calculate the individual discounts
def discounts : List ℝ := List.zipWith (λ mp sp => mp - sp) marked_prices sold_prices

-- Calculate the total discount
def total_discount : ℝ := discounts.sum

-- Calculate the total marked price
def total_marked_price : ℝ := marked_prices.sum

-- Calculate the average discount rate
def average_discount_rate : ℝ := (total_discount / total_marked_price) * 100

theorem average_discount_rate_correct : average_discount_rate ≈ 14.55 :=
by
  sorry

end average_discount_rate_correct_l92_92895


namespace number_of_ways_to_choose_four_knowing_each_other_l92_92681

-- Define a type for people
constant Person : Type

-- Define a set of 8 people (finite)
constant G : Finset Person
axiom card_G : G.card = 8

-- Define a symmetric relation "knows" on G
constant knows : Person → Person → Prop
axiom knows_symmetric : ∀ a b : Person, knows a b → knows b a
axiom knows_irreflexive : ∀ a : Person, ¬ knows a a

-- Each person knows exactly 6 others
axiom knows_degree : ∀ a : Person, (G.filter (knows a)).card = 6

-- The target proof: the number of ways to choose four people such that every pair among them knows each other is 16
theorem number_of_ways_to_choose_four_knowing_each_other :
  (∃ S : Finset (Finset Person), 
  S.card = 16 ∧ ∀ t ∈ S, t.card = 4 ∧ ∀ x y ∈ t, knows x y) := sorry

end number_of_ways_to_choose_four_knowing_each_other_l92_92681


namespace find_x_l92_92790

theorem find_x (x : ℝ) :
  (1 / 3) * ((2 * x + 8) + (7 * x + 3) + (3 * x + 9)) = 5 * x^2 - 8 * x + 2 ↔ 
  x = (36 + Real.sqrt 2136) / 30 ∨ x = (36 - Real.sqrt 2136) / 30 := 
sorry

end find_x_l92_92790


namespace first_year_after_2020_with_sum_15_l92_92810

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2020_with_sum_15 :
  ∀ n, n > 2020 → (sum_of_digits n = 15 ↔ n = 2058) := by
  sorry

end first_year_after_2020_with_sum_15_l92_92810


namespace arithmetic_sequence_properties_geometric_sequence_properties_l92_92042

noncomputable def a_sequence (n : ℕ) : ℕ := 2 * n + 1

noncomputable def S_n (n : ℕ) : ℕ := n^2 + 2 * n

noncomputable def b_sequence (n : ℕ) : ℕ := a_sequence n + (3^(n-1))

noncomputable def T_n (n : ℕ) : ℕ := S_n n + (3^n - 1) / 2

theorem arithmetic_sequence_properties :
  (∀ n, a_sequence n = 2 * n + 1) ∧
  (∀ n, S_n n = n^2 + 2 * n)
:=
by
  -- conditions given
  have condition1 : a_sequence 2 = 5 := rfl
  have condition2 : a_sequence 5 + a_sequence 7 = 26 := rfl
  -- prove the results
  sorry

theorem geometric_sequence_properties :
  (∀ n, b_sequence n = a_sequence n + 3^(n-1)) ∧
  (∀ n, T_n n = n^2 + 2 * n + (3^n - 1) / 2)
:=
by
  -- conditions given
  have condition3 : ∀ n, b_sequence n - a_sequence n = 3^(n-1) := rfl
  -- prove the results assuming previous theorem
  sorry

end arithmetic_sequence_properties_geometric_sequence_properties_l92_92042


namespace basketball_team_mode_mean_l92_92223

def ages := [18, 18, 18, 18, 18, 19, 19, 19, 19, 20, 21, 21]

noncomputable def mode (l : List ℕ) : ℕ :=
  l.groupBy id
   .toList
   .map (λ x => (x.snd.length, x.fst))
   .max
   .getOrElse (0, 0)
   .snd

noncomputable def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem basketball_team_mode_mean :
  mode ages = 18 ∧ mean ages = 19 :=
by
  sorry

end basketball_team_mode_mean_l92_92223


namespace sqrt_of_25_l92_92387

theorem sqrt_of_25 : ∃ x : ℝ, x^2 = 25 ∧ (x = 5 ∨ x = -5) :=
by {
  sorry
}

end sqrt_of_25_l92_92387


namespace smallest_composite_no_prime_factors_less_than_20_l92_92980

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92980


namespace best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92152

-- Part (a)
theorem best_play_wins_majority_two_classes (n : ℕ) :
  let prob_tie := (1 / 2) ^ n in
  1 - prob_tie = 1 - (1 / 2) ^ n :=
sorry

-- Part (b)
theorem best_play_wins_majority_multiple_classes (n s : ℕ) :
  let prob_tie := (1 / 2) ^ ((s - 1) * n) in
  1 - prob_tie = 1 - (1 / 2) ^ ((s - 1) * n) :=
sorry

end best_play_wins_majority_two_classes_best_play_wins_majority_multiple_classes_l92_92152


namespace more_girls_than_boys_l92_92108

theorem more_girls_than_boys
  (b g : ℕ)
  (ratio : b / g = 3 / 4)
  (total : b + g = 42) :
  g - b = 6 :=
sorry

end more_girls_than_boys_l92_92108


namespace problem_statement_l92_92174

-- Define the function f
def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

-- Auxiliary definition for the monotonicity condition
def f_monotonic_increasing (ω : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → x2 < Real.pi / 5 → f ω x1 < f ω x2

-- Each of the following definitions checks the separate conclusions
def A (ω : ℝ) : Prop :=
  ω = 1

def B (ω : ℝ) : Prop :=
  f ω (3 * Real.pi / 10) > 1 / 2

def C (ω : ℝ) : Prop :=
  f ω Real.pi < 0 → 
    ∃! x : ℝ, 0 < x ∧ x < 101/100 * Real.pi ∧ f ω x = 0

def D (ω : ℝ) : Prop :=
  f ω Real.pi < 0 →
    ∀ x1 x2 : ℝ, 2 * Real.pi / 5 ≤ x1 → x1 < x2 → x2 ≤ Real.pi → f ω x1 > f ω x2

-- The final statement to check if A, B, and C are true and D is false
theorem problem_statement (ω : ℝ) :
  f_monotonic_increasing ω →
  (A ω ∧ B ω ∧ C ω ∧ ¬ D ω) :=
by sorry

end problem_statement_l92_92174


namespace remainder_seven_pow_two_thousand_mod_thirteen_l92_92977

theorem remainder_seven_pow_two_thousand_mod_thirteen :
  7^2000 % 13 = 1 := by
  sorry

end remainder_seven_pow_two_thousand_mod_thirteen_l92_92977


namespace perfect_square_461_l92_92872

theorem perfect_square_461 (x : ℤ) (y : ℤ) (hx : 5 ∣ x) (hy : 5 ∣ y) 
  (h : x^2 + 461 = y^2) : x^2 = 52900 :=
  sorry

end perfect_square_461_l92_92872


namespace sin_90_eq_1_l92_92464

theorem sin_90_eq_1 :
  let θ := 90 : ℝ in
  let cos_θ := real.cos θ in
  let sin_θ := real.sin θ in 
  let rotation_matrix := ![![cos_θ, -sin_θ], ![sin_θ, cos_θ]] in
  let point := ![1, 0] in
  let rotated_point := matrix.mul_vec rotation_matrix point in
  rotated_point = ![0, 1] → 
  sin_θ = 1 :=
by
  sorry

end sin_90_eq_1_l92_92464


namespace sin_ninety_deg_l92_92434

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92434


namespace length_PF_eq_16_3_l92_92826

-- Definitions for the problem conditions
def parabola : (ℝ × ℝ) → Prop := λ P, P.2 ^ 2 = 8 * (P.1 + 2)
def focus : ℝ × ℝ := (0, 0)
def line (θ : ℝ) : (ℝ × ℝ) → Prop := λ P, P.2 = tan θ * P.1
def is_intersection (P : ℝ × ℝ) : Prop := parabola P ∧ line (π / 3) P
def is_perpendicular_bisector (M : ℝ × ℝ) (l : ((ℝ × ℝ) → Prop)) : Prop := 
  ∀ P, l P → P.2 = M.2 → P.1 = M.1

-- Main theorem statement
theorem length_PF_eq_16_3 : 
  ∀ (P A B: ℝ × ℝ), 
    is_intersection A ∧
    is_intersection B ∧
    (P = (16 / 3, 0)) ∧
    is_perpendicular_bisector ((A.1 + B.1) / 2, (A.2 + B.2) / 2) (line (θ := atan (-(1 / (tan (π / 3)))))) →
  (dist (focus) P) = 16 / 3 := 
by sorry

end length_PF_eq_16_3_l92_92826


namespace min_distance_between_parallel_lines_l92_92595

open Real

-- Define line l1 and line l2
def line1 (x y : ℝ) : Prop := 3 * x - 4 * y + 1 = 0
def line2 (x y : ℝ) : Prop := 6 * x - 8 * y + 4 = 0

-- Define the distance formula between two parallel lines
def distance_between_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / sqrt (A^2 + B^2)

-- Prove that the minimum distance between the lines l1 and l2 is 1/5
theorem min_distance_between_parallel_lines :
  distance_between_lines 6 (-8) 2 4 = 1 / 5 :=
by
  sorry

end min_distance_between_parallel_lines_l92_92595


namespace unique_singleton_function_l92_92963

theorem unique_singleton_function (f : ℕ → ℝ) :
  (∀ a b c : ℕ, f (a * c) + f (b * c) - f c * f (a * b) ≥ 1) →
  (∀ x : ℕ, f x = 1) :=
begin
  sorry
end

end unique_singleton_function_l92_92963


namespace extra_flowers_correct_l92_92936

variable (pickedTulips : ℕ) (pickedRoses : ℕ) (usedFlowers : ℕ)

def totalFlowers : ℕ := pickedTulips + pickedRoses
def extraFlowers : ℕ := totalFlowers pickedTulips pickedRoses - usedFlowers

theorem extra_flowers_correct : 
  pickedTulips = 39 → pickedRoses = 49 → usedFlowers = 81 → extraFlowers pickedTulips pickedRoses usedFlowers = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end extra_flowers_correct_l92_92936


namespace intersection_points_count_l92_92101

variables {R : Type*} [LinearOrderedField R]

def line1 (x y : R) : Prop := 3 * y - 2 * x = 1
def line2 (x y : R) : Prop := x + 2 * y = 2
def line3 (x y : R) : Prop := 4 * x - 6 * y = 5

theorem intersection_points_count : 
  ∃ p1 p2 : R × R, 
   (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
   (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
   p1 ≠ p2 ∧ 
   (∀ p : R × R, (line1 p.1 p.2 ∧ line3 p.1 p.2) → False) := 
sorry

end intersection_points_count_l92_92101


namespace irrational_sqrt3_is_only_irrational_l92_92322

theorem irrational_sqrt3_is_only_irrational :
  ((¬ ∃ (a b : ℤ), b ≠ 0 ∧ (0 : ℚ) = a / b) → False) ∧
  ((¬ ∃ (a b : ℤ), b ≠ 0 ∧ (-2 : ℚ) = a / b) → False) ∧
  ((¬ ∃ (a b : ℤ), b ≠ 0 ∧ (3 / 7 : ℚ) = a / b) → False) ∧
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (\sqrt{3}: ℝ) = a / b) :=
sorry

end irrational_sqrt3_is_only_irrational_l92_92322


namespace gratuities_charged_l92_92915

-- Define the conditions in the problem
def total_bill : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def ny_striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Calculate the total cost before tax and gratuities
def subtotal : ℝ := ny_striploin_cost + wine_cost

-- Calculate the taxes paid
def tax : ℝ := subtotal * sales_tax_rate

-- Calculate the total bill before gratuities
def total_before_gratuities : ℝ := subtotal + tax

-- Goal: Prove that gratuities charged is 41
theorem gratuities_charged : (total_bill - total_before_gratuities) = 41 := by sorry

end gratuities_charged_l92_92915


namespace largest_cube_surface_area_l92_92894

theorem largest_cube_surface_area (width length height : ℝ) 
    (h_width : width = 12)
    (h_length : length = 16)
    (h_height : height = 14) :
    ∃ (side : ℝ), side = 12 ∧ 6 * side^2 = 864 :=
by 
    use 12
    split
    · exact rfl
    · norm_num

end largest_cube_surface_area_l92_92894


namespace angle_A1C1O_30_l92_92157

noncomputable def triangleABC (A B C O A1 C1 : Type) [triangle ABC] : Prop :=
  ∃ (A1 B1 C1 : Type), 
    (angle_eq ABC 120) ∧ 
    (angle_bisectors_intersect A1 B1 C1 O) ∧ 
    (angle_eq_cyclic A1 C1 O 30)

theorem angle_A1C1O_30 (A B C O A1 B1 C1 : Type) [triangle ABC] :
  triangleABC A B C O A1 C1 → 
  (angle_eq A1 C1 O 30) :=
sorry

end angle_A1C1O_30_l92_92157


namespace sin_90_eq_one_l92_92414

noncomputable theory
open Real

/--
The sine of an angle in the unit circle is the y-coordinate of the point at that angle from the positive x-axis.
Rotating the point (1,0) by 90 degrees counterclockwise about the origin results in the point (0,1).
Prove that \(\sin 90^\circ = 1\).
-/
theorem sin_90_eq_one : sin (90 * (real.pi / 180)) = 1 :=
by
  -- Definitions and conditions for the unit circle and sine function
  let angle := 90 * (real.pi / 180)
  have h1 : (cos angle, sin angle) = (0, 1),
  { sorry },
  -- Desired conclusion
  exact h1.2

end sin_90_eq_one_l92_92414


namespace sum_first_60_terms_sequence_l92_92245

theorem sum_first_60_terms_sequence :
  (∑ i in Finset.range 60, (a : ℕ → ℝ) i) = 1830 :=
sorry

end sum_first_60_terms_sequence_l92_92245


namespace sin_ninety_degrees_l92_92400

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92400


namespace best_play_wins_majority_l92_92141

variables (n : ℕ)

-- Conditions
def students_in_play_A : ℕ := n
def students_in_play_B : ℕ := n
def mothers : ℕ := 2 * n

-- Question
theorem best_play_wins_majority : 
  (probability_fin_votes_wins_majority (students_in_play_A n) (students_in_play_B n) (mothers n)) = 1 - (1/2)^n :=
sorry

end best_play_wins_majority_l92_92141


namespace david_marks_in_biology_l92_92520

theorem david_marks_in_biology (marks_english marks_math marks_physics marks_chemistry : ℕ)
  (average_marks num_subjects total_marks_known : ℕ)
  (h1 : marks_english = 76)
  (h2 : marks_math = 65)
  (h3 : marks_physics = 82)
  (h4 : marks_chemistry = 67)
  (h5 : average_marks = 75)
  (h6 : num_subjects = 5)
  (h7 : total_marks_known = marks_english + marks_math + marks_physics + marks_chemistry)
  (h8 : total_marks_known = 290)
  : ∃ biology_marks : ℕ, biology_marks = 85 ∧ biology_marks = (average_marks * num_subjects) - total_marks_known :=
by
  -- placeholder for proof
  sorry

end david_marks_in_biology_l92_92520


namespace int_values_satisfy_condition_l92_92258

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l92_92258


namespace derivative_f_at_2_l92_92597

noncomputable def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

theorem derivative_f_at_2 : (deriv f 2) = 4 := by
  sorry

end derivative_f_at_2_l92_92597


namespace ritsumeikan_2011_q5_l92_92726

noncomputable def f (x : ℝ) : ℝ := ∫ t in 0..x, 1 / (1 + t^2)

theorem ritsumeikan_2011_q5
  (x : ℝ)
  (hx : -1 ≤ x)
  (hx' : x < 1) :
  cos (2 * f (sqrt ((1 + x) / (1 - x)))) = -x :=
by
  sorry

end ritsumeikan_2011_q5_l92_92726


namespace count_integers_between_25_and_36_l92_92284

theorem count_integers_between_25_and_36 :
  {x : ℤ | 25 < x ∧ x < 36}.finite.card = 10 :=
by
  sorry

end count_integers_between_25_and_36_l92_92284


namespace true_propositions_l92_92049

noncomputable def z : ℂ := 2 / (1 + I)

def p1 : Prop := complex.abs z = 2
def p2 : Prop := z^2 = 2 * I
def p3 : Prop := complex.conj z = 1 + I
def p4 : Prop := 
  let x := z.re in
  let y := z.im in
  x > 0 ∧ y < 0 

theorem true_propositions : { p3, p4 } = { true } :=
sorry

end true_propositions_l92_92049


namespace percentage_difference_B_A_l92_92372

theorem percentage_difference_B_A :
  ∀ (A B C : ℕ),
  A = 1184 →
  B + C = A →
  148 + B = C →
  (B = C) →
  ((B - 0).natAbs * 100) / (A) = 50 :=
by
  sorry

end percentage_difference_B_A_l92_92372


namespace only_c_eq_one_exists_infinite_sequence_l92_92554

def f (n : ℕ) : ℕ :=
  ∑ i in (Finset.range (n + 1)).filter (λ d => d ∣ n), i

theorem only_c_eq_one_exists_infinite_sequence (c : ℕ) :
  (∃ a : ℕ → ℕ, (∀ i : ℕ, f (a i) - a i = c) ∧ StrictMono a) ↔ c = 1 :=
by
  sorry

end only_c_eq_one_exists_infinite_sequence_l92_92554


namespace fourth_number_is_2_eighth_number_is_2_l92_92855

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end fourth_number_is_2_eighth_number_is_2_l92_92855


namespace sequence_general_formula_l92_92576

theorem sequence_general_formula (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 2 = 4 →
  S 4 = 30 →
  (∀ n, n ≥ 2 → a (n + 1) + a (n - 1) = 2 * (a n + 1)) →
  ∀ n, a n = n^2 :=
by
  intros h1 h2 h3
  sorry

end sequence_general_formula_l92_92576


namespace difference_of_numbers_l92_92812

theorem difference_of_numbers (x y : ℕ) (h1 : x + y = 64) (h2 : y = 26) : x - y = 12 :=
sorry

end difference_of_numbers_l92_92812


namespace polygon_diagonals_separate_triangles_l92_92355

-- Given definitions for the problem
variables (n : ℕ) (k_0 k_1 k_2 : ℕ)

-- Conditions based on the problem statement
def total_triangles : Prop := k_0 + k_1 + k_2 = n - 2
def edges_sum : Prop := k_1 + 2 * k_2 = n

-- The theorem to prove
theorem polygon_diagonals_separate_triangles
  (h1 : total_triangles n k_0 k_1 k_2)
  (h2 : edges_sum n k_0 k_1 k_2) : k_2 ≥ 2 :=
sorry

end polygon_diagonals_separate_triangles_l92_92355


namespace probability_not_safe_trip_is_correct_l92_92186

noncomputable def probability_not_safe_trip (p : ℝ) (n : ℕ) : ℝ :=
  1 - (1 - p)^n

-- Setting specific values for p and n
def p : ℝ := 0.001
def n : ℕ := 775

def probability_not_safe_trip_775km_approx : ℝ := 0.53947

-- Theorem statement
theorem probability_not_safe_trip_is_correct :
  probability_not_safe_trip p n ≈ probability_not_safe_trip_775km_approx :=
sorry

end probability_not_safe_trip_is_correct_l92_92186


namespace sin_90_eq_one_l92_92422

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92422


namespace period_of_tan_transformation_l92_92317

theorem period_of_tan_transformation :
  ∀ x : ℝ, ( ∃ k : ℤ, x = k * (2 * π / 3) ) ↔ y = tan (3 * x / 2) = y :=
sorry

end period_of_tan_transformation_l92_92317


namespace smallest_composite_no_prime_factors_less_than_20_l92_92992

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92992


namespace expression_value_l92_92553

def floor : ℝ → ℤ := Int.floor

theorem expression_value :
  (floor 6.5 : ℝ) * (floor (2 / 3) : ℝ) + (floor 2 : ℝ) * 7.2 + (floor 8.4 : ℝ) - 6.2 = 16.2 :=
by
  sorry

end expression_value_l92_92553


namespace sin_90_eq_one_l92_92418

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92418


namespace remaining_number_is_6218_l92_92198

theorem remaining_number_is_6218 :
  let candidates := { n : ℕ | n ∈ Set.Icc 3 223 ∧ n % 4 = 3 },
      initial_sum := candidates.sum,
      steps := candidates.card - 1
  in initial_sum - 2 * steps = 6218 :=
by
  let candidates := { n : ℕ | n ∈ Set.Icc 3 223 ∧ n % 4 = 3 }
  let initial_sum := candidates.sum
  let steps := candidates.card - 1
  have h_initial_sum : initial_sum = 6328 := sorry
  have h_steps : steps = 55 := sorry
  calc
    initial_sum - 2 * steps
        = 6328 - 2 * 55 : by rw [h_initial_sum, h_steps]
    ... = 6218 : by norm_num

end remaining_number_is_6218_l92_92198


namespace solve_quadratic_l92_92089

theorem solve_quadratic (x : ℝ) (h : x^2 - 6*x + 8 = 0) : x = 2 ∨ x = 4 :=
sorry

end solve_quadratic_l92_92089


namespace count_congruent_to_2_mod_7_l92_92652

theorem count_congruent_to_2_mod_7 (n : ℕ) : 
  n = 150 → (finset.card (finset.filter (λ k, k % 7 = 2) (finset.range n))) = 22 :=
by
  sorry

end count_congruent_to_2_mod_7_l92_92652


namespace integral_e_x_plus_x_eval_l92_92530

noncomputable def integral_e_x_plus_x : ℝ :=
  ∫ x in 0..1, (Real.exp x + x)

theorem integral_e_x_plus_x_eval : integral_e_x_plus_x = Real.exp 1 - (1/2 : ℝ) :=
  by
  sorry

end integral_e_x_plus_x_eval_l92_92530


namespace sum_of_perimeters_l92_92927

theorem sum_of_perimeters (a : ℕ → ℝ) (h₁ : a 0 = 180) (h₂ : ∀ n, a (n + 1) = 1 / 2 * a n) :
  (∑' n, a n) = 360 :=
by
  sorry

end sum_of_perimeters_l92_92927


namespace sin_ninety_deg_l92_92427

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92427


namespace positive_difference_of_solutions_l92_92318

theorem positive_difference_of_solutions :
  ∀ r : ℝ, r ≠ -4 → (r^2 - 5*r - 14) / (r + 4) = 2*r + 9 →
    (∀ r1 r2, r1^2 + 22*r1 + 50 = 0 → r2^2 + 22*r2 + 50 = 0 →
    |r1 - r2| = 2) :=
by
  sorry

end positive_difference_of_solutions_l92_92318


namespace kia_vehicles_count_l92_92752

theorem kia_vehicles_count (total_vehicles : ℕ) (dodge_fraction : ℚ) (hyundai_to_dodge_ratio : ℚ) 
  (total_vehicles_eq : total_vehicles = 400)
  (dodge_fraction_eq : dodge_fraction = 1 / 2)
  (hyundai_to_dodge_ratio_eq : hyundai_to_dodge_ratio = 1 / 2) :
  let dodge_vehicles := (dodge_fraction * total_vehicles : ℚ).to_nat in
  let hyundai_vehicles := (hyundai_to_dodge_ratio * dodge_vehicles : ℚ).to_nat in
  let kia_vehicles := total_vehicles - (dodge_vehicles + hyundai_vehicles) in
  kia_vehicles = 100 :=
by
  sorry

end kia_vehicles_count_l92_92752


namespace smallest_composite_no_prime_factors_less_than_20_l92_92987

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92987


namespace ellipse_M1_equation_range_lambda1_lambda2_l92_92666

theorem ellipse_M1_equation (a b : ℝ) (m : ℝ) (h : m > 0) (x y : ℝ) :
  (M1_similar_M2 : is_similar_ellipse a b) 
  (M2_eq : x^2 + 2 * y^2 = 1) 
  (pt_on_M1 : (1, (Math.sqrt 2) / 2)) : 
  (x / Math.sqrt 2)^2 / 2 + y^2 / 1 = 1 := sorry

theorem range_lambda1_lambda2 (a b k : ℝ) (m : ℝ) (h : m > 0) (x1 x2 : ℝ) : 
  (line_intersects_ellipse : line_intersects_ellipse k P (-2, 0)) 
  (AF_eq : λf1, -y1 = λ1 * y2) 
  (BF_eq : λf2, -y2 = λ2 * y1) 
  (M1_focus : focus_a1_b1 := 1) : 
  (6 < (λ1 + λ2) < 10) := sorry

end ellipse_M1_equation_range_lambda1_lambda2_l92_92666


namespace trigonometric_propositions_l92_92374

def proposition1 := ¬ (∃ p : ℝ, ∀ x : ℝ, sin (|x|) = sin (|x + p|))
def proposition2 := ¬ (∀ x : ℝ, tan x < tan (x + 0.00001))
def proposition3 := ¬ (∃ k : ℝ, ∀ x : ℝ, abs (cos (2 * x) + 1 / 2) = abs (cos (2 * (x + k)) + 1 / 2))
def proposition4 := ∀ x, sin (x + 5 * pi / 2) = - sin (- (x + 5 * pi / 2))
def proposition5 := ¬ (∀ x, sin (2 * x + pi / 3) = 0)

theorem trigonometric_propositions : proposition1 ∧ ¬ proposition2 ∧ ¬ proposition3 ∧ proposition4 ∧ ¬ proposition5 := by
  sorry

end trigonometric_propositions_l92_92374


namespace sandy_shopping_amount_l92_92208

theorem sandy_shopping_amount (remaining_money : ℝ) (spent_percentage : ℝ) (original_amount : ℝ) :
  remaining_money = (1 - spent_percentage) * original_amount → original_amount = 320 :=
by
  intro h
  have : original_amount = remaining_money / (1 - spent_percentage),
  { sorry }
  rw [this]
  norm_num

end sandy_shopping_amount_l92_92208


namespace common_difference_l92_92725

variable {a : ℕ → ℝ}
variables {x y z d : ℝ} (n : ℕ)

-- a is defined as an arithmetic progression with common difference d
def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a (k + 1) = a k + d

-- Conditions
axiom condition1 : arithmetic_progression a d
axiom condition2 : (∑ i in Finset.range n, a (2 * i + 1) ^ 2) = x
axiom condition3 : (∑ i in Finset.range n, a (2 * i + 2) ^ 2) = y
axiom condition4 : a n + a (n + 1) = z

-- The result to be proven
theorem common_difference (n_pos : 0 < n) : d = (y - x) / (n * z) := sorry

end common_difference_l92_92725


namespace steve_waist_measurement_in_cm_l92_92782

theorem steve_waist_measurement_in_cm : 
  ∀ (waist_in_inches : ℤ) (inches_per_foot : ℤ) (cm_per_foot : ℝ), 
  waist_in_inches = 39 → inches_per_foot = 12 → cm_per_foot = 30.48 →
  let waist_in_feet := (waist_in_inches : ℝ) / (inches_per_foot : ℝ)
  let waist_in_cm := waist_in_feet * cm_per_foot in
  waist_in_cm.round = 99 := 
by 
  intros waist_in_inches inches_per_foot cm_per_foot h1 h2 h3 
  let waist_in_feet := (waist_in_inches : ℝ) / (inches_per_foot : ℝ)
  let waist_in_cm := waist_in_feet * cm_per_foot
  sorry

end steve_waist_measurement_in_cm_l92_92782


namespace minimum_value_of_f_l92_92663

noncomputable def f (x : ℝ) := 2 * x + 18 / x

theorem minimum_value_of_f :
  ∃ x > 0, f x = 12 ∧ ∀ y > 0, f y ≥ 12 :=
by
  sorry

end minimum_value_of_f_l92_92663


namespace mono_sum_eq_five_l92_92100

-- Conditions
def term1 (x y : ℝ) (m : ℕ) : ℝ := x^2 * y^m
def term2 (x y : ℝ) (n : ℕ) : ℝ := x^n * y^3

def is_monomial_sum (x y : ℝ) (m n : ℕ) : Prop :=
  term1 x y m + term2 x y n = x^(2:ℕ) * y^(3:ℕ)

-- Theorem stating the result
theorem mono_sum_eq_five (x y : ℝ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
by
  sorry

end mono_sum_eq_five_l92_92100


namespace sin_x_squared_not_periodic_l92_92349

theorem sin_x_squared_not_periodic (p : ℝ) (hp : 0 < p) : 
  ¬ ∀ x : ℝ, sin ((x + p)^2) = sin (x^2) :=
by
  sorry

end sin_x_squared_not_periodic_l92_92349


namespace find_a_plus_b_l92_92603

noncomputable def f (a x b : ℝ) : ℝ := log a x + b

theorem find_a_plus_b (a b : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1)
  (h₃ : ∀ x, 1 ≤ x ∧ x ≤ 2 → 1 ≤ f a x b ∧ f a x b ≤ 2) :
  (0 < a ∧ a < 1 → a + b = 5 / 2) ∧ (a > 1 → a + b = 3) :=
by
  sorry

end find_a_plus_b_l92_92603


namespace integer_solutions_count_l92_92270

theorem integer_solutions_count :
  (finset.filter (λ (x : ℤ), 5 < real.sqrt (x : ℝ) ∧ real.sqrt (x : ℝ) < 6) 
  (finset.Icc 26 35)).card = 10 :=
by
  sorry

end integer_solutions_count_l92_92270


namespace big_joe_height_is_8_l92_92379

variable (Pepe_height Frank_height Larry_height Ben_height BigJoe_height : ℝ)

axiom Pepe_height_def : Pepe_height = 4.5
axiom Frank_height_def : Frank_height = Pepe_height + 0.5
axiom Larry_height_def : Larry_height = Frank_height + 1
axiom Ben_height_def : Ben_height = Larry_height + 1
axiom BigJoe_height_def : BigJoe_height = Ben_height + 1

theorem big_joe_height_is_8 :
  BigJoe_height = 8 :=
sorry

end big_joe_height_is_8_l92_92379


namespace problem_f_eq_2_l92_92562

noncomputable def f (x : ℝ) : ℝ := log (sqrt (x^2 + 1) - x) + 1

theorem problem_f_eq_2 :
  f 2015 + f (-2015) = 2 := by
  sorry

end problem_f_eq_2_l92_92562


namespace cost_of_pears_l92_92931

theorem cost_of_pears (P : ℕ)
  (apples_cost : ℕ := 40)
  (dozens : ℕ := 14)
  (total_cost : ℕ := 1260)
  (h_p : dozens * P + dozens * apples_cost = total_cost) :
  P = 50 :=
by
  sorry

end cost_of_pears_l92_92931


namespace smallest_composite_no_prime_factors_less_than_20_l92_92984

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92984


namespace smallest_composite_no_prime_factors_less_than_20_l92_92993

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def all_prime_factors_at_least (n k : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ k

theorem smallest_composite_no_prime_factors_less_than_20 :
  ∃ n : ℕ, is_composite n ∧ all_prime_factors_at_least n 23 ∧
           ∀ m : ℕ, is_composite m ∧ all_prime_factors_at_least m 23 → n ≤ m :=
sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92993


namespace sphere_volume_l92_92067

theorem sphere_volume (S : ℝ) (hS : S = 4 * π) : ∃ V : ℝ, V = (4 / 3) * π := 
by
  sorry

end sphere_volume_l92_92067


namespace correct_fourth_number_correct_eighth_number_l92_92840

-- Condition: Initial number on the board and sequence of medians
def initial_board : List ℝ := [1]
def medians : List ℝ := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

-- The number written fourth is 2
def fourth_number_written (board : List ℝ) : ℝ := 2

-- The number written eighth is also 2
def eighth_number_written (board : List ℝ) : ℝ := 2

-- Formalizing the conditions and assertions
theorem correct_fourth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    fourth_number_written board = 2 := 
sorry

theorem correct_eighth_number :
  ∃ board : List ℝ, 
    board.head = 1 ∧ 
    -- Assume the sequence of medians can be calculated from the board
    (calculate_medians_from_board board = medians) ∧
    eighth_number_written board = 2 := 
sorry

-- Function to calculate medians from the board (to be implemented)
noncomputable def calculate_medians_from_board (board : List ℝ) : List ℝ := sorry

end correct_fourth_number_correct_eighth_number_l92_92840


namespace determine_m_l92_92636

theorem determine_m (a m : ℝ) (h : a > 0) (h2 : (m, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = -a * p.1 ^ 2 + 2 * a * p.1 + 3)) (h3 : m ≠ 0) : m = 2 :=
sorry

end determine_m_l92_92636


namespace ball_hits_ground_in_5_seconds_l92_92933

-- Problem statement in Lean 4
theorem ball_hits_ground_in_5_seconds :
  ∀ t : ℝ, (h t = -16 * t^2 - 20 * t + 200) → (h 0 = 200) → (h 5 = 0) :=
by
  intro t h_eq h_initial
  sorry

end ball_hits_ground_in_5_seconds_l92_92933


namespace fourth_number_is_2_eighth_number_is_2_l92_92858

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end fourth_number_is_2_eighth_number_is_2_l92_92858


namespace last_four_digits_of_5_pow_2018_l92_92193

theorem last_four_digits_of_5_pow_2018 : 
  (5^2018) % 10000 = 5625 :=
by {
  sorry
}

end last_four_digits_of_5_pow_2018_l92_92193


namespace cid_earnings_l92_92390

theorem cid_earnings :
  let model_a_oil_change_cost := 20
  let model_a_repair_cost := 30
  let model_a_wash_cost := 5
  let model_b_oil_change_cost := 25
  let model_b_repair_cost := 40
  let model_b_wash_cost := 8
  let model_c_oil_change_cost := 30
  let model_c_repair_cost := 50
  let model_c_wash_cost := 10

  let model_a_oil_changes := 5
  let model_a_repairs := 10
  let model_a_washes := 15
  let model_b_oil_changes := 3
  let model_b_repairs := 4
  let model_b_washes := 10
  let model_c_oil_changes := 2
  let model_c_repairs := 6
  let model_c_washes := 5

  let total_earnings := 
      (model_a_oil_change_cost * model_a_oil_changes) +
      (model_a_repair_cost * model_a_repairs) +
      (model_a_wash_cost * model_a_washes) +
      (model_b_oil_change_cost * model_b_oil_changes) +
      (model_b_repair_cost * model_b_repairs) +
      (model_b_wash_cost * model_b_washes) +
      (model_c_oil_change_cost * model_c_oil_changes) +
      (model_c_repair_cost * model_c_repairs) +
      (model_c_wash_cost * model_c_washes)

  total_earnings = 1200 := by
  sorry

end cid_earnings_l92_92390


namespace least_positive_integer_condition_l92_92863

theorem least_positive_integer_condition :
  ∃ b : ℕ, b > 0 ∧
    b % 3 = 2 ∧
    b % 5 = 4 ∧
    b % 6 = 5 ∧
    b % 7 = 6 ∧
    ∀ n : ℕ, (n > 0 ∧ n % 3 = 2 ∧ n % 5 = 4 ∧ n % 6 = 5 ∧ n % 7 = 6) → n ≥ b :=
    ∃ b : ℕ, b = 209 := sorry

end least_positive_integer_condition_l92_92863


namespace range_of_a_l92_92599

def f (a x : ℝ) : ℝ := a^2 * x - 2 * a + 1

theorem range_of_a (a : ℝ) : (∃ x ∈ Icc (0 : ℝ) 1, f a x ≤ 0) ↔ a ∈ Icc (1 / 2) ∞ := 
sorry

end range_of_a_l92_92599


namespace machine_x_widgets_per_hour_l92_92753

-- Definitions of the variables and conditions
variable (Wx Wy Tx Ty: ℝ)
variable (h1: Tx = Ty + 60)
variable (h2: Wy = 1.20 * Wx)
variable (h3: Wx * Tx = 1080)
variable (h4: Wy * Ty = 1080)

-- Statement of the problem to prove
theorem machine_x_widgets_per_hour : Wx = 3 := by
  sorry

end machine_x_widgets_per_hour_l92_92753


namespace cost_per_bag_l92_92716

-- Definitions and variables based on the conditions
def sandbox_length : ℝ := 3  -- Sandbox length in feet
def sandbox_width : ℝ := 3   -- Sandbox width in feet
def bag_area : ℝ := 3        -- Area of one bag of sand in square feet
def total_cost : ℝ := 12     -- Total cost to fill up the sandbox in dollars

-- Statement to prove
theorem cost_per_bag : (total_cost / (sandbox_length * sandbox_width / bag_area)) = 4 :=
by
  sorry

end cost_per_bag_l92_92716


namespace correct_statement_about_CH3COOK_l92_92320

def molar_mass_CH3COOK : ℝ := 98  -- in g/mol

def avogadro_number : ℝ := 6.02 * 10^23  -- molecules per mole

def hydrogen_atoms_in_CH3COOK (mol_CH3COOK : ℝ) : ℝ :=
  3 * mol_CH3COOK * avogadro_number

theorem correct_statement_about_CH3COOK (mol_CH3COOK : ℝ) (h: mol_CH3COOK = 1) :
  hydrogen_atoms_in_CH3COOK mol_CH3COOK = 3 * avogadro_number :=
by
  sorry

end correct_statement_about_CH3COOK_l92_92320


namespace sum_largest_smallest_g_l92_92736

noncomputable def g (x : ℝ) : ℝ := (|x - 3|) + (|x - 5|) - (|2 * x - 8|) + 1

theorem sum_largest_smallest_g :
  let g_vals := (Set.image g (Set.Icc 3 7)) in
  (Set.max g_vals) + (Set.min g_vals) = 4 :=
by
  sorry

end sum_largest_smallest_g_l92_92736


namespace fraction_product_is_one_l92_92315

theorem fraction_product_is_one : 
  (1 / 4) * (1 / 5) * (1 / 6) * 120 = 1 :=
by 
  sorry

end fraction_product_is_one_l92_92315


namespace count_integers_between_25_and_36_l92_92283

theorem count_integers_between_25_and_36 :
  {x : ℤ | 25 < x ∧ x < 36}.finite.card = 10 :=
by
  sorry

end count_integers_between_25_and_36_l92_92283


namespace find_m_l92_92610

variable (a m x : ℝ)

noncomputable def quadratic_function : ℝ → ℝ := λ x, -a * x^2 + 2 * a * x + 3

theorem find_m (h1 : a > 0) (h2 : quadratic_function a m = 3) (h3 : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92610


namespace sin_90_degrees_l92_92489

theorem sin_90_degrees : Real.sin (Float.pi / 2) = 1 :=
by
  sorry

end sin_90_degrees_l92_92489


namespace tom_books_relation_l92_92828

variables (p1 p2 : ℝ)

-- Conditions of the original problem
def tom_books_originally : ℕ := 5
def tom_books_sold : ℕ := 4
def tom_books_left : ℕ := tom_books_originally - tom_books_sold
def money_earned_from_selling := 4 * p1
def new_books_bought : ℕ := 38
def cost_per_new_book := p2
def total_cost_of_new_books := new_books_bought * cost_per_new_book

-- Prove the relationship between p1 and p2 and total number of books
theorem tom_books_relation :
  4 * p1 = 38 * p2 ∧ (tom_books_left + new_books_bought = 39) :=
begin
  sorry
end

end tom_books_relation_l92_92828


namespace find_m_eq_2_l92_92618

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l92_92618


namespace mask_assignment_l92_92792

def digit_square_ends_in_different_digit (d : Nat) : Prop :=
  let product := d * d
  product / 10 > 0 ∧ product % 10 ≠ d

def is_mask_assignment_valid (elephant mouse pig panda : Nat) : Prop :=
  digit_square_ends_in_different_digit elephant ∧
  digit_square_ends_in_different_digit mouse ∧
  digit_square_ends_in_different_digit pig ∧
  digit_square_ends_in_different_digit panda ∧
  (mouse * mouse) % 10 = elephant ∧
  (pig * pig) % 10 = elephant ∧
  List.nodup [elephant, mouse, pig, panda]

theorem mask_assignment :
  ∃ (elephant mouse pig panda : Nat), 
  is_mask_assignment_valid elephant mouse pig panda ∧
  elephant = 6 ∧ mouse = 4 ∧ pig = 8 ∧ panda = 1 := by
  sorry

end mask_assignment_l92_92792


namespace additional_emails_per_day_l92_92160

theorem additional_emails_per_day
  (emails_per_day_before : ℕ)
  (half_days : ℕ)
  (total_days : ℕ)
  (total_emails : ℕ)
  (emails_received_first_half : ℕ := emails_per_day_before * half_days)
  (emails_received_second_half : ℕ := total_emails - emails_received_first_half)
  (emails_per_day_after : ℕ := emails_received_second_half / half_days) :
  emails_per_day_before = 20 → half_days = 15 → total_days = 30 → total_emails = 675 → (emails_per_day_after - emails_per_day_before = 5) :=
by
  intros
  sorry

end additional_emails_per_day_l92_92160


namespace part1_part2_l92_92749

def p (a : ℝ) : Prop := a^2 - 5*a - 6 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + 1 = 0 → x < 0

theorem part1 (a : ℝ) (hp : p a) : a ∈ Set.Iio (-1) ∪ Set.Ioi 6 :=
sorry

theorem part2 (a : ℝ) (h_or : p a ∨ q a) (h_and : ¬ (p a ∧ q a)) : a ∈ Set.Iio (-1) ∪ Set.Ioc 2 6 :=
sorry

end part1_part2_l92_92749


namespace sin_90_degree_l92_92503

-- Definitions based on conditions
def unit_circle_point (angle : ℝ) : ℝ × ℝ :=
  if angle = 90 * (π / 180) then (0, 1) else sorry

def sin_usual (angle : ℝ) : ℝ :=
  (unit_circle_point angle).snd

-- The main theorem as per the question and conditions
theorem sin_90_degree : sin_usual (90 * (π / 180)) = 1 :=
by
  sorry

end sin_90_degree_l92_92503


namespace sin_90_eq_1_l92_92446

-- Define the unit circle
def unit_circle (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)

-- Define the sine of 90 degrees using radians
def sin_90_degrees : ℝ := unit_circle (Real.pi / 2).snd

-- State the theorem
theorem sin_90_eq_1 : sin_90_degrees = 1 :=
by
  sorry

end sin_90_eq_1_l92_92446


namespace height_of_picture_frame_l92_92761

-- Definitions of lengths and perimeter
def length : ℕ := 10
def perimeter : ℕ := 44

-- Perimeter formula for a rectangle
def rectangle_perimeter (L H : ℕ) : ℕ := 2 * (L + H)

-- Theorem statement: Proving the height is 12 inches based on given conditions
theorem height_of_picture_frame : ∃ H : ℕ, rectangle_perimeter length H = perimeter ∧ H = 12 := by
  sorry

end height_of_picture_frame_l92_92761


namespace radius_difference_proof_l92_92803

noncomputable def radius_difference (r : ℝ) : ℝ := 
  let R := r * Real.sqrt (5 / 2) in 
  R - r

theorem radius_difference_proof (r : ℝ) : radius_difference r = 0.58 * r :=
by 
  have h : Real.sqrt (5 / 2) ≈ 1.581 := sorry
  have R := r * Real.sqrt (5 / 2)
  calc
    radius_difference r 
      = (r * Real.sqrt (5 / 2)) - r : by rfl
  ... = (1.581 * r) - r          : by rw [h]
  ... = 0.58 * r                  : by ring

end radius_difference_proof_l92_92803


namespace sin_ninety_deg_l92_92431

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92431


namespace common_root_equation_l92_92556

theorem common_root_equation {m : ℝ} (x : ℝ) (h1 : m * x - 1000 = 1001) (h2 : 1001 * x = m - 1000 * x) : m = 2001 ∨ m = -2001 :=
by
  -- Skipping the proof details
  sorry

end common_root_equation_l92_92556


namespace final_number_is_6218_l92_92200

theorem final_number_is_6218 :
  ∃ n : ℕ, (n = 6218) ∧ (∀ k, 3 ≤ k ∧ k ≤ 223 ∧ k % 4 = 3) ∧
  (∀ S : List ℕ, (∀ x ∈ S, 3 ≤ x ∧ x ≤ 223 ∧ x % 4 = 3) → list.sum S - 2 * (S.length - 1) = 6218) :=
begin
  sorry
end

end final_number_is_6218_l92_92200


namespace compute_sin_90_l92_92510

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92510


namespace find_lambda_l92_92052

variable (a b : EuclideanSpace ℝ) (λ : ℝ)

-- Conditions
axiom nonzero_a : a ≠ 0
axiom nonzero_b : b ≠ 0
axiom angle_ab : real.angle a b = real.pi / 3
axiom magnitude_eq : ‖a‖ = ‖b‖
axiom perp : ∀ λ, a ⬝ (λ • a - b) = 0

-- Statement to prove
theorem find_lambda : λ = 1/2 :=
by
  sorry

end find_lambda_l92_92052


namespace floor_ceil_multiplication_l92_92956

theorem floor_ceil_multiplication :
  (⟦ 0.998 ⟧ : ℤ) * (⟧ 1.999 ⟦ : ℤ) = 0 := 
by
  have h1: ⟦ 0.998 ⟧ = 0 := by sorry
  have h2: ⟧ 1.999 ⟦ = 2 := by sorry
  exact calc
    (⟦ 0.998 ⟧ : ℤ) * (⟧ 1.999 ⟦ : ℤ) = 0 * 2 : by rw [h1, h2]
                          ... = 0     : by rfl

end floor_ceil_multiplication_l92_92956


namespace expand_polynomial_l92_92536

theorem expand_polynomial (z : ℂ) :
  (3 * z^3 + 4 * z^2 - 5 * z + 1) * (2 * z^2 - 3 * z + 4) * (z - 1) = 3 * z^6 - 3 * z^5 + 5 * z^4 + 2 * z^3 - 5 * z^2 + 4 * z - 16 :=
by sorry

end expand_polynomial_l92_92536


namespace evaluate_expression_l92_92720

noncomputable def a : ℝ := 2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def b : ℝ := -2 * Real.sqrt 2 + 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def c : ℝ := 2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6
noncomputable def d : ℝ := -2 * Real.sqrt 2 - 3 * Real.sqrt 3 + 4 * Real.sqrt 6

theorem evaluate_expression : (1/a + 1/b + 1/c + 1/d)^2 = 952576 / 70225 := by
  sorry

end evaluate_expression_l92_92720


namespace angle_AC1_plane_BCD_l92_92038

noncomputable def tetrahedron_angle : ℝ :=
  let R := 1 in -- Assume radius R = 1 for simplicity, as it cancels out in calculations
  let C1 := (R, 0, 0) in
  let C := (R/2, R / (2 * Real.sqrt 3), R * Real.sqrt 2 / 3) in -- vertices of a regular tetrahedron assuming centered at origin
  let B := (R/2, - R / (2 * Real.sqrt 3), R * Real.sqrt 2 / 3) in
  let D := (-R, 0, 0) in
  let O := (0, 0, 0) in
  let AC1 := C1 - C in
  let n_BCD := (B - C) × (D - C) in -- normal of the plane BCD
  let cos_theta := (AC1 ∙ n_BCD) / (‖AC1‖ * ‖n_BCD‖) in
  Real.arctan (Real.sqrt 2 / 2)

theorem angle_AC1_plane_BCD {A B C D C1 : Point} (h_tetra : regular_tetrahedron A B C D)
  (h_sphere : circumscribed_sphere A B C D)
  (h_diameter : diameter O C1) :
  angle (line AC1) (plane BCD) = Real.arctan (Real.sqrt 2 / 2) := by
  sorry

end angle_AC1_plane_BCD_l92_92038


namespace num_of_integers_satisfying_sqrt_condition_l92_92278

theorem num_of_integers_satisfying_sqrt_condition : 
  let S := { x : ℤ | 5 < Real.sqrt x ∧ x < 36 }
  in (S.card = 10) :=
begin
  let S := { x : ℤ | 25 < x ∧ x < 36 },
  sorry
end

end num_of_integers_satisfying_sqrt_condition_l92_92278


namespace inverse_f_at_5_l92_92671

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Assume the domain and range conditions for the inverse to exist
lemma inv_fun_exists : ∃ g : ℝ → ℝ, ∀ x : ℝ, x ≤ 1 → x^2 + 1 = g (x^2 + 1) :=
begin
  use λ y, real.sqrt (y - 1), -- The inverse of x^2 + 1 within the domain
  intros x hx,
  rw [real.sqrt_sq (sub_pos.2 (by linarith [hx]))],
  linarith,
end

-- Prove that the inverse function at 5 is -2
theorem inverse_f_at_5 : f⁻¹ 5 = -2 :=
by {
  have h : f (-2) = 5 := by norm_num,
  have h_inv := inv_fun_exists,
  cases h_inv with g hg,
  have hf_g : g (f (-2)) = -2 := by simp [hg],
  rw hf_g at h,
  exact h.symm,
}

end inverse_f_at_5_l92_92671


namespace number_of_children_in_group_l92_92797

theorem number_of_children_in_group (n : ℕ) 
    (h : (12 * 8) / n = 6) : 
    n = 16 := 
by {
  have h1 : 96 / n = 6 := h,
  have h2 : 96 = 6 * n,
  linarith,
  exact eq.symm (nat.div_eq_of_eq_mul_right (by norm_num) h2),
  sorry
}

end number_of_children_in_group_l92_92797


namespace original_total_price_of_candy_and_soda_l92_92329

noncomputable def price_before_increase := 
by
  sorry

theorem original_total_price_of_candy_and_soda :
  (∃ original_price_candy original_price_soda : ℝ,
    price_before_increase original_price_candy 1.25 = 10 ∧ 
    price_before_increase original_price_soda 1.5 = 6 ∧ 
    original_price_candy + original_price_soda = 16) :=
by
  sorry

end original_total_price_of_candy_and_soda_l92_92329


namespace skateboard_cost_correct_l92_92715

variable (cost_toy_cars : ℝ) (cost_toy_trucks : ℝ) (total_spent_toys : ℝ) (cost_skateboard : ℝ)

-- Conditions from the problem
axiom toy_cars_cost : cost_toy_cars = 14.88
axiom toy_trucks_cost : cost_toy_trucks = 5.86
axiom total_spent_on_toys : total_spent_toys = 25.62

-- Define the calculation for the skateboard cost
def calculated_skateboard_cost : ℝ :=
  total_spent_toys - (cost_toy_cars + cost_toy_trucks)

-- Prove that the calculated skateboard cost is $4.88
theorem skateboard_cost_correct : calculated_skateboard_cost = 4.88 :=
by
  unfold calculated_skateboard_cost
  rw [toy_cars_cost, toy_trucks_cost, total_spent_on_toys]
  norm_num
  sorry

end skateboard_cost_correct_l92_92715


namespace square_perimeter_division_l92_92204

theorem square_perimeter_division
  (a : ℝ)
  (M divides AC : ℝ)
  (area_ratio : ℝ)
  (line_through_M : ℝ)
  (perimeter_ratio : ℝ)
  (h1 : M divides AC = 1 / 5)
  (h2 : area_ratio = 1 / 11)
  (h3 : perimeter_ratio = 5 / 19) :
  (line_through_M divides perimeter of the square) = perimeter_ratio :=
sorry

end square_perimeter_division_l92_92204


namespace height_increase_per_year_l92_92879

-- Variables and definitions
variable (h : ℝ)
axiom initial_height : ℝ := 4 -- Initial height of the tree in feet

-- Condition: The height of the tree at the end of the 6th year is 1/4 taller than at the end of the 4th year
axiom cond : initial_height + 6*h = (initial_height + 4*h) + 1/4* (initial_height + 4*h)

-- Statement to be proved:
theorem height_increase_per_year : h = 1 :=
by
  sorry -- Proof is omitted

end height_increase_per_year_l92_92879


namespace sets_without_perfect_squares_l92_92181

def Si (i : ℕ) : set ℤ := { n : ℤ | 200 * i ≤ n ∧ n < 200 * (i + 1) }

lemma perfect_square_bounds (i : ℕ) (n : ℤ) (h : n ∈ Si i) : 
  any_square_is_outside_range (S: set ℤ) (n \in Si i) (n = k^2 \forall k: ℤ \):
  ((∀ m: ℤ, square: ℤ  (n - m) / m == k^2)) := sorry

theorem sets_without_perfect_squares : 
  (cardinality_found_sets (S_0, S_1, ..., S_499) == 259) (S_0: ..., ..., S_499: set ℤ):=

begin
  sorry
end

end sets_without_perfect_squares_l92_92181


namespace triangle_cosA_and_c_l92_92107

theorem triangle_cosA_and_c (a b : ℝ) (A B : ℝ) (c : ℝ)
  (ha : a = 3) 
  (hb : b = 2 * sqrt 6) 
  (hB_A : B = 2 * A) : 
  cos A = sqrt 6 / 3 ∧ c = 5 :=
by
  sorry

end triangle_cosA_and_c_l92_92107


namespace sin_ninety_deg_l92_92437

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92437


namespace ellipse_equation_l92_92061

theorem ellipse_equation
  (P : ℝ × ℝ)
  (a b c : ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : 2 * a = 5 + 3)
  (h3 : (2 * c) ^ 2 = 5 ^ 2 - 3 ^ 2)
  (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1 ∨ P.2 ^ 2 / a ^ 2 + P.1 ^ 2 / b ^ 2 = 1)
  : ((a = 4) ∧ (c = 2) ∧ (b ^ 2 = 12) ∧
    (P.1 ^ 2 / 16 + P.2 ^ 2 / 12 = 1) ∨
    (P.2 ^ 2 / 16 + P.1 ^ 2 / 12 = 1)) :=
sorry

end ellipse_equation_l92_92061


namespace solve_for_x_l92_92213

theorem solve_for_x : ∃ x : ℚ, 6 * (2 * x + 3) - 4 = -3 * (2 - 5 * x) + 3 * x ∧ x = 10 / 3 := by
  sorry

end solve_for_x_l92_92213


namespace intersection_points_count_l92_92102

variables {R : Type*} [LinearOrderedField R]

def line1 (x y : R) : Prop := 3 * y - 2 * x = 1
def line2 (x y : R) : Prop := x + 2 * y = 2
def line3 (x y : R) : Prop := 4 * x - 6 * y = 5

theorem intersection_points_count : 
  ∃ p1 p2 : R × R, 
   (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
   (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
   p1 ≠ p2 ∧ 
   (∀ p : R × R, (line1 p.1 p.2 ∧ line3 p.1 p.2) → False) := 
sorry

end intersection_points_count_l92_92102


namespace determine_m_l92_92632

theorem determine_m (a m : ℝ) (h : a > 0) (h2 : (m, 3) ∈ set_of (λ p : ℝ × ℝ, p.2 = -a * p.1 ^ 2 + 2 * a * p.1 + 3)) (h3 : m ≠ 0) : m = 2 :=
sorry

end determine_m_l92_92632


namespace find_number_l92_92900

-- Definitions of the fractions involved
def frac_2_15 : ℚ := 2 / 15
def frac_1_5 : ℚ := 1 / 5
def frac_1_2 : ℚ := 1 / 2

-- Condition that the number is greater than the sum of frac_2_15 and frac_1_5 by frac_1_2 
def number : ℚ := frac_2_15 + frac_1_5 + frac_1_2

-- Theorem statement matching the math proof problem
theorem find_number : number = 5 / 6 :=
by
  sorry

end find_number_l92_92900


namespace number_of_integers_between_25_and_36_l92_92265

theorem number_of_integers_between_25_and_36 :
  {n : ℕ | 25 < n ∧ n < 36}.card = 10 :=
by
  sorry

end number_of_integers_between_25_and_36_l92_92265


namespace smallest_composite_no_prime_factors_less_than_20_l92_92981

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92981


namespace product_lcm_gcd_l92_92546

def a : ℕ := 6
def b : ℕ := 8

theorem product_lcm_gcd : Nat.lcm a b * Nat.gcd a b = 48 := by
  sorry

end product_lcm_gcd_l92_92546


namespace annual_decrease_in_budget_V_l92_92675

-- Define the initial budgets for projects Q and V in 1990
def budget_Q_1990 : ℝ := 540000
def budget_V_1990 : ℝ := 780000

-- Define the annual increase in the budget for project Q
def annual_increase_Q : ℝ := 30000

-- Define an unknown annual decrease in the budget for project V
def D : ℝ

-- Define the budgets for projects Q and V in 1994
def budget_Q_1994 := budget_Q_1990 + 4 * annual_increase_Q
def budget_V_1994 := budget_V_1990 - 4 * D

-- Define the given condition that the budgets were equal in 1994
def budgets_equal_in_1994 := budget_Q_1994 = budget_V_1994

-- The theorem to prove
theorem annual_decrease_in_budget_V : budgets_equal_in_1994 → D = 30000 := by
  sorry

end annual_decrease_in_budget_V_l92_92675


namespace fraction_value_l92_92293

theorem fraction_value : (10 + 20 + 30 + 40) / 10 = 10 := by
  sorry

end fraction_value_l92_92293


namespace at_least_one_not_greater_than_one_l92_92566

theorem at_least_one_not_greater_than_one (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) : 
  ∃ x ∈ {a / b, b / c, c / a}, x ≤ 1 :=
by
  sorry

end at_least_one_not_greater_than_one_l92_92566


namespace complex_number_in_second_quadrant_l92_92962

open Complex

theorem complex_number_in_second_quadrant (z : ℂ) :
  (Complex.abs z = Real.sqrt 7) →
  (z.re < 0 ∧ z.im > 0) →
  z = -2 + Real.sqrt 3 * Complex.I :=
by
  intros h1 h2
  sorry

end complex_number_in_second_quadrant_l92_92962


namespace problem1_problem2_problem3_l92_92158

-- Problem 1: Standard Equation of Circle C
theorem problem1 {x y : ℝ} (h : ∃ (a : ℝ), a > 0 ∧ (x, y) = (2*a, a) ∧ r = 2*a ∧ (2*sqrt(3))^2 = a^2 + (sqrt(3))^2 - (2*a)^2) :
  (x - 2)^2 + (y - 1)^2 = 4 :=
by
  sorry

-- Problem 2: Value of Real Number b
theorem problem2 {x y b : ℝ} 
  (h1 : ∃ (a : ℝ), a > 0 ∧ (x, y) = (2*a, a) ∧ r = 2*a ∧ (2*sqrt(3))^2 = a^2 + (sqrt(3))^2 - (2*a)^2) 
  (h2 : ∃ (A B : ℝ), A ≠ B ∧ y = -2*x + b ∧ ( (A, -2*A + b), (B, -2*B + b) ) := A ∧ B ∧ (x-2)^2 + (y-1)^2 = 4) :
  b = (5 + sqrt(15)) / 2 ∨ b = (5 - sqrt(15)) / 2 :=
by
  sorry

-- Problem 3: Range of Values for y-coordinate of Center C
theorem problem3 {x y : ℝ} (h1 : ∃ (a : ℝ), a > 0 ∧ a <= 2 ∧ (x, y) = (2*a, a) ∧ r = 3)
  (h2 : (0, 3) ∈ set.PointsOnCircle (x, y) 3 ∧ ((x-2*a)^2 + (y-a)^2 = 9) ∧ ∃ (M : ℝ), (M (y+1))/2 = 4 ∧ 0 <= a ∧ a <= 2) :
  0 < y ∧ y <= 2 :=
by
  sorry

end problem1_problem2_problem3_l92_92158


namespace unique_solution_eq_l92_92951

theorem unique_solution_eq (a : ℝ) : 
  (∀ x : ℝ, a * 3^x + 3^(-x) = 3) ↔ a ∈ (set.Icc (-∞) 0) ∪ {9 / 4} :=
sorry

end unique_solution_eq_l92_92951


namespace delete_edges_maintain_degree_l92_92533

-- Define a simple graph with vertices and edges
structure SimpleGraph (V : Type) :=
  (adj : V → V → Prop)
  (symm : ∀ {x y : V}, adj x y → adj y x)
  (loopless : ∀ {x : V}, ¬adj x x)

-- Define the condition for the original graph G
variables {V : Type} {G : SimpleGraph V} {v e : ℕ}
  (hv : Fintype.card V = v)
  (he : (∑ x, Fintype.card {y // G.adj x y}) = 2 * e)

-- Definition of the half degree condition
def half_degree_condition (G : SimpleGraph V) (S : set (V × V)) : Prop :=
  ∀ x : V, Fintype.card {y // G.adj x y ∧ (x, y) ∉ S} ≥ (Fintype.card {y // G.adj x y}) / 2

-- Define the claim in Lean as a theorem
theorem delete_edges_maintain_degree :
  ∃ S : set (V × V), |S| ≥ (e - v + 1) / 2 ∧ half_degree_condition G S :=
sorry

end delete_edges_maintain_degree_l92_92533


namespace male_employees_count_l92_92112

theorem male_employees_count (x : ℕ) (h1 : 7 * x / (8 * x)) (h2 : (7 * x + 3) / (8 * x) = 8 / 9) :
  7 * x = 189 :=
by
  sorry

end male_employees_count_l92_92112


namespace loss_percentage_l92_92330

theorem loss_percentage (C : ℝ) (h : 40 * C = 100 * C) : 
  ∃ L : ℝ, L = 60 := 
sorry

end loss_percentage_l92_92330


namespace always_possible_to_form_triangle_l92_92569

/-
Given six sticks with distinctly different lengths, it is always possible
to form a triangle such that the sides of the triangle consist of 1, 2, and 3 of the original sticks respectively.
-/

theorem always_possible_to_form_triangle (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
(h₁ : a₁ > 0) (h₂ : a₂ > 0) (h₃ : a₃ > 0) (h₄ : a₄ > 0) (h₅ : a₅ > 0) (h₆ : a₆ > 0)
(h₇ : a₁ ≠ a₂) (h₈ : a₂ ≠ a₃) (h₉ : a₃ ≠ a₄) (h₁₀ : a₄ ≠ a₅) (h₁₁ : a₅ ≠ a₆) (h₁₂ : a₁ > a₂)
(h₁₃ : a₂ > a₃) (h₁₄ : a₃ > a₄) (h₁₅ : a₄ > a₅) (h₁₆ : a₅ > a₆) :
∃ b₁ b₂ b₃ : ℝ,
  {b₁, b₂, b₃} ⊆ {a₁, a₂, a₃, a₄, a₅, a₆} ∧
  b₁ + b₂ > b₃ ∧ b₁ + b₃ > b₂ ∧ b₂ + b₃ > b₁ :=
sorry

end always_possible_to_form_triangle_l92_92569


namespace sin_90_eq_one_l92_92420

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92420


namespace ineq_sum_fraction_l92_92187

noncomputable def sequenceSum (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  Finset.sum (Finset.range n) a

theorem ineq_sum_fraction (n : ℕ) (a : ℕ → ℝ) (a_pos : ∀ i, 0 < a i) :
  let s := sequenceSum n a in
  s = ∑ i in Finset.range n, a i →
  2 ≥ ∑ i in Finset.range n, a i / (s - a i) ∧ ∑ i in Finset.range n, a i / (s - a i) ≥ n / (n - 1) :=
by
  sorry

end ineq_sum_fraction_l92_92187


namespace kabadi_players_l92_92214

-- Define the conditions
def kho_only : Nat := 40
def both_games : Nat := 5
def total_players : Nat := 50

-- Define kabadi_only based on the given conditions
def kabadi_only : Nat := total_players - kho_only - both_games

-- Prove the total number of people who play kabadi
theorem kabadi_players : kabadi_only + both_games = 15 := by
  -- kabadi_only = 10 from the given conditions
  have kabadi_only_val : kabadi_only = 10 := by rfl
  -- total kabadi players
  rw [kabadi_only_val]
  rfl

end kabadi_players_l92_92214


namespace sin_ninety_degrees_l92_92394

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92394


namespace fraction_of_apples_consumed_l92_92935

theorem fraction_of_apples_consumed (f : ℚ) 
  (bella_eats_per_day : ℚ := 6) 
  (days_per_week : ℕ := 7) 
  (grace_remaining_apples : ℚ := 504) 
  (weeks_passed : ℕ := 6) 
  (total_apples_picked : ℚ := 42 / f) :
  (total_apples_picked - (bella_eats_per_day * days_per_week * weeks_passed) = grace_remaining_apples) 
  → f = 1 / 18 :=
by
  intro h
  sorry

end fraction_of_apples_consumed_l92_92935


namespace price_after_six_months_l92_92233

theorem price_after_six_months (initial_price : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) (num_changes : ℕ)
  (increases : ℕ) (decreases : ℕ) (final_price : ℝ) :
  initial_price = 64 →
  increase_rate = 3/2 →
  decrease_rate = 1/2 →
  num_changes = 6 →
  increases = 3 →
  decreases = 3 →
  final_price = initial_price * (increase_rate ^ increases) * (decrease_rate ^ decreases) →
  final_price = 27 :=
by
  intros h_init h_inc_rate h_dec_rate h_num_changes h_incs h_decs h_final_eq
  rw [h_init, h_inc_rate, h_dec_rate, h_num_changes, h_incs, h_decs, h_final_eq]
  sorry

end price_after_six_months_l92_92233


namespace real_estate_commission_l92_92358

theorem real_estate_commission (r : ℝ) (P : ℝ) (C : ℝ) (h : r = 0.06) (hp : P = 148000) : C = P * r :=
by
  -- Definitions and proof steps will go here.
  sorry

end real_estate_commission_l92_92358


namespace pi_estimation_based_on_geometric_probability_l92_92899

theorem pi_estimation_based_on_geometric_probability :
  let side_length := 1
  let total_beans := 5120
  let beans_in_circle := 4009
  let radius := side_length / 2
  let area_square := side_length ^ 2
  let area_circle := (Real.pi * radius ^ 2)

  let proportion := (beans_in_circle:ℝ) / (total_beans:ℝ)
  let estimated_pi := 4 * proportion
  Real.floor (estimated_pi * 100) = 313 :=
by sorry

end pi_estimation_based_on_geometric_probability_l92_92899


namespace compute_sin_90_l92_92512

noncomputable def sin_90_eq_one : Prop :=
  let angle_0_point := (1, 0) in
  let angle_90_point := (0, 1) in
  (angle_90_point.y = 1)  ∧ ∀ θ : ℝ, θ = 90 → Real.sin (θ * (Real.pi / 180)) = 1

theorem compute_sin_90 : sin_90_eq_one := 
by 
  -- the proof steps go here
  sorry

end compute_sin_90_l92_92512


namespace latest_ant_falloff_time_l92_92742

-- Definition for the problem conditions
variables {m : ℕ} (hm : 0 < m)
def checkerboard := fin m × fin m
def ant_trajectory (p : checkerboard m) (dir : fin 4) : ℕ → checkerboard m :=
  λ t, match dir with
      | ⟨0, _⟩ => (⟨(p.1 + t) % m, sorry⟩)  -- move east
      | ⟨1, _⟩ => (⟨(p.1 + t) % m, sorry⟩)  -- move north
      | ⟨2, _⟩ => (⟨(p.1 + m - t % m), sorry⟩)  -- move west
      | ⟨3, _⟩ => (⟨(p.1 + m - t % m), sorry⟩)  -- move south
      | _ => p  -- default case (impossible due to fin 4)
  end

-- Problem statement in Lean
theorem latest_ant_falloff_time : ∀ {m : ℕ}, 0 < m → (∃ t : ℚ, ∀ ants : list (checkerboard m), 
  let paths := ants.map (λ p, ant_trajectory p sorry) in
  ∀ (t' : ℚ), t' ≥ t → ¬ ∃ a, ∃ t'', t'' ≤ t' ∧ paths.any (λ path, path t'' = some a))
  → t = (3 * m) / 2 - 1 :=
sorry

end latest_ant_falloff_time_l92_92742


namespace find_m_l92_92629

theorem find_m (a m : ℝ) (h_pos : a > 0) (h_points : (m, 3) ∈ set_of (λ x : ℝ × ℝ, ∃ x_val : ℝ, x.snd = -a * (x_val)^2 + 2 * a * x_val + 3)) (h_non_zero : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92629


namespace property_damage_worth_40000_l92_92388

-- Definitions based on conditions in a)
def medical_bills : ℝ := 70000
def insurance_rate : ℝ := 0.80
def carl_payment : ℝ := 22000
def carl_rate : ℝ := 0.20

theorem property_damage_worth_40000 :
  ∃ P : ℝ, P = 40000 ∧ 
    (carl_payment = carl_rate * (P + medical_bills)) :=
by
  sorry

end property_damage_worth_40000_l92_92388


namespace polynomials_in_H_count_l92_92732

-- Define the set H of polynomials
def is_polynomial_in_H (Q : polynomial ℚ) : Prop :=
  ∃ (n : ℕ) (c : fin n → ℤ),
    Q = polynomial.monomial n 1 + 
        ∑ i : fin n, polynomial.monomial i (c i) + 
        polynomial.C 75 ∧
        (∀ (a b : ℤ), (polynomial.has_root Q (a + b * complex.I) → ∃ (a b : ℤ), b ≠ 0))

-- Main theorem statement
theorem polynomials_in_H_count : ∃ X : ℕ, 
  (set_of (λ Q, is_polynomial_in_H Q)).finite.card = X :=
sorry

end polynomials_in_H_count_l92_92732


namespace triangle_area_144_l92_92354

noncomputable def area_of_triangle_abc (t1 t2 t3 : ℝ) (p : Point) (A B C : Triangle) : ℝ :=
  if conditions_held then 144 else 0

-- The main theorem we wish to prove:
theorem triangle_area_144 {A B C : Triangle} {P : Point}
    (t1_area : ℝ) (t2_area : ℝ) (t3_area : ℝ)
    (h1 : t1_area = 4)
    (h2 : t2_area = 9)
    (h3 : t3_area = 49)
    (H : PointInTriangle P A B C)
    (ParallelLinesThroughP : ∃ t1 t2 t3, 
      TriangleSimilar t1 A B C ∧ TriangleSimilar t2 A B C ∧ TriangleSimilar t3 A B C ∧
      Area t1 = t1_area ∧ Area t2 = t2_area ∧ Area t3 = t3_area) :
  area_of_triangle_abc t1_area t2_area t3_area P A B C = 144 := 
by sorry

end triangle_area_144_l92_92354


namespace sin_90_eq_one_l92_92417

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l92_92417


namespace triangle_BC_length_l92_92106

/-- In a triangle ABC with AB = 91 and AC = 106, 
and a circle centered at A with radius 91 intersects BC at points B and X, 
with BX and CX having integer lengths, prove that BC = 85. -/
theorem triangle_BC_length (AB AC radius BC BX CX : ℝ)
  (h1 : AB = 91) (h2 : AC = 106) (h3 : radius = 91)
  (h4 : ∃ (BX CX : ℕ), BC = BX + CX) 
  (h5 : XY: ∃ B X : ℝ, radius = X):BC= 85 := 
  sorry

end triangle_BC_length_l92_92106


namespace sunday_price_correct_l92_92222

def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.60
def second_discount_rate : ℝ := 0.25
def discounted_price : ℝ := original_price * (1 - first_discount_rate)
def sunday_price : ℝ := discounted_price * (1 - second_discount_rate)

theorem sunday_price_correct :
  sunday_price = 75 := by
  sorry

end sunday_price_correct_l92_92222


namespace identify_random_events_l92_92925

def is_random_event (event : String) : Prop :=
  event = "Riding a bike to a crossroad and encountering a red light" ∨
  event = "Someone winning a lottery ticket"

theorem identify_random_events :
  ∀ (events : List String),
    events = ["For any real number x, x^2 < 0",
              "The sum of the interior angles of a triangle is 180°",
              "Riding a bike to a crossroad and encountering a red light",
              "Someone winning a lottery ticket"] →
    List.filter is_random_event events = 
      ["Riding a bike to a crossroad and encountering a red light",
       "Someone winning a lottery ticket"] :=
by
  intros events h
  rw h
  simp [is_random_event]
  sorry

end identify_random_events_l92_92925


namespace smallest_composite_no_prime_factors_less_than_20_l92_92986

/-- A composite number is a number that is the product of two or more natural numbers, each greater than 1. -/
def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- A number has no prime factors less than 20 if all its prime factors are at least 20. -/
def no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  ∀ p : ℕ, prime p → p ∣ n → p ≥ 20

/-- Prove that 529 is the smallest composite number that has no prime factors less than 20. -/
theorem smallest_composite_no_prime_factors_less_than_20 : 
  is_composite 529 ∧ no_prime_factors_less_than_20 529 ∧ 
  ∀ n : ℕ, is_composite n ∧ no_prime_factors_less_than_20 n → n ≥ 529 :=
by sorry

end smallest_composite_no_prime_factors_less_than_20_l92_92986


namespace solution_set_inequality_range_of_a_l92_92072

-- Define the function f(x) = |x - a| - |x + 1|
def f (a x : ℝ) : ℝ := abs (x - a) - abs (x + 1)

-- Question (1): Given a = 3, find the solution set of the inequality f(x) ≥ 2x + 3
theorem solution_set_inequality (x : ℝ) : 
  (∃ a = 3, f a x ≥ 2 * x + 3) ↔ x ≤ -1/4 :=
by
  sorry

-- Question (2): If there exists an x in the interval [1, 2] such that f(x) ≤ |x - 5| holds true,
-- find the range of possible values for a
theorem range_of_a (a : ℝ) : 
  (∃ x ∈ set.Icc 1 2, f a x ≤ abs (x - 5)) ↔ -4 ≤ a ∧ a ≤ 7 :=
by
  sorry

end solution_set_inequality_range_of_a_l92_92072


namespace complex_number_expression_l92_92175

noncomputable theory
open Complex

theorem complex_number_expression (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = -6 :=
by {
  -- The proof needs to be filled in here.
  sorry
}

end complex_number_expression_l92_92175


namespace fourth_number_is_2_eighth_number_is_2_l92_92857

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end fourth_number_is_2_eighth_number_is_2_l92_92857


namespace coloring_scheme_count_l92_92332

variable (m : ℕ) (n : ℕ)
variable (m_ge_2 : m ≥ 2) (n_ge_4 : n ≥ 4)

noncomputable def count_coloring_schemes : ℕ :=
  if even n then
    m * ((m-1) ^ (n/2) + (-1 : ℤ) ^ (n/2) * (m-1)) ^ 2
  else
    m * ((m-1) ^ n + (-1 : ℤ) ^ n * (m-1))

theorem coloring_scheme_count :
  ∀ (m : ℕ) (n : ℕ),
  m ≥ 2 → n ≥ 4 →
  (a_n = if even n then
           m * ((m-1) ^ (n / 2) + (-1 : ℤ) ^ (n / 2) * (m-1)) ^ 2
         else
           m * ((m-1) ^ n + (-1 : ℤ) ^ n * (m-1))) :=
by sorry

end coloring_scheme_count_l92_92332


namespace function_increasing_no_negative_roots_l92_92601

noncomputable def f (a x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem function_increasing (a : ℝ) (h : a > 1) : 
  ∀ (x1 x2 : ℝ), (-1 < x1) → (x1 < x2) → (f a x1 < f a x2) := 
by
  -- placeholder proof
  sorry

theorem no_negative_roots (a : ℝ) (h : a > 1) : 
  ∀ (x : ℝ), (x < 0) → (f a x ≠ 0) := 
by
  -- placeholder proof
  sorry

end function_increasing_no_negative_roots_l92_92601


namespace fraction_of_students_saying_not_enjoy_but_actually_enjoys_l92_92377

open Nat

-- Setup definitions based on the given conditions
def total_students : ℕ := 100
def percentage_enjoy_chess : ℚ := 7 / 10
def percentage_not_enjoy_chess : ℚ := 3 / 10
def percentage_admit_enjoy_chess : ℚ := 3 / 4
def percentage_false_claim_not_enjoy_chess : ℚ := 1 / 4
def percentage_honest_not_enjoy_chess : ℚ := 4 / 5
def percentage_false_claim_enjoy_chess : ℚ := 1 / 5

-- Define the proof statement
theorem fraction_of_students_saying_not_enjoy_but_actually_enjoys :
  let students_enjoy := percentage_enjoy_chess * total_students,
      students_not_enjoy := percentage_not_enjoy_chess * total_students,
      students_false_claim_not_enjoy := percentage_false_claim_not_enjoy_chess * students_enjoy,
      students_honestly_not_enjoy := percentage_honest_not_enjoy_chess * students_not_enjoy,
      total_say_not_enjoy := students_false_claim_not_enjoy + students_honestly_not_enjoy,
      fraction := students_false_claim_not_enjoy / total_say_not_enjoy
  in fraction = 35 / 83 := by { sorry }

end fraction_of_students_saying_not_enjoy_but_actually_enjoys_l92_92377


namespace must_swap_gold_and_silver_l92_92860

-- Define the types for coins and colors
inductive CoinType
| gold
| silver

-- Define the structure for coins
structure Coin where
  weight : ℕ
  ctype : CoinType

-- Define the initial condition
def initial_condition (coins : list Coin) : Prop :=
  ∀ i j, i < j → coins[i].weight ≠ coins[j].weight ∧ coins.last.weight > coins[i].weight

-- Define the problem statement
theorem must_swap_gold_and_silver (coins : list Coin) (hc : initial_condition coins) :
  ∃ i j, i ≠ j ∧ coins[i].ctype ≠ coins[j].ctype ∧
  sorted (weight ∘ list.enum i j coins → list.swap i j):
  sorry

end must_swap_gold_and_silver_l92_92860


namespace angle_is_45_degrees_l92_92083

def vector_a : ℝ × ℝ := (-1, 2)
def vector_b : ℝ × ℝ := (2, 1)

def angle_between (u v : ℝ × ℝ) : ℝ :=
    let dot_product := (u.1 * v.1 + u.2 * v.2)
    let norm_u := Real.sqrt (u.1 * u.1 + u.2 * u.2)
    let norm_v := Real.sqrt (v.1 * v.1 + v.2 * v.2)
    let cos_theta := dot_product / (norm_u * norm_v)
    Real.acos cos_theta

theorem angle_is_45_degrees : 
      angle_between (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) vector_b = Real.pi / 4 :=
by
  sorry

end angle_is_45_degrees_l92_92083


namespace int_values_satisfy_condition_l92_92257

theorem int_values_satisfy_condition :
  ∃ (count : ℕ), count = 10 ∧ ∀ (x : ℤ), 6 > Real.sqrt x ∧ Real.sqrt x > 5 ↔ (x ≥ 26 ∧ x ≤ 35) := by
  sorry

end int_values_satisfy_condition_l92_92257


namespace sin_ninety_degrees_l92_92402

theorem sin_ninety_degrees : Real.sin (90 * Real.pi / 180) = 1 := 
by
  sorry

end sin_ninety_degrees_l92_92402


namespace num_bijective_functions_on_three_elements_l92_92748

theorem num_bijective_functions_on_three_elements :
  let f : ({1, 2, 3} → {1, 2, 3}) := sorry in
  (∀ x1 x2 : {1, 2, 3}, x1 ≠ x2 → f x1 ≠ f x2) →
  ∃ l : List ({1, 2, 3} → {1, 2, 3}), (l.length = 6) ∧ (∀ f ∈ l, ∀ x1 x2 : {1, 2, 3}, x1 ≠ x2 → f x1 ≠ f x2) :=
sorry

end num_bijective_functions_on_three_elements_l92_92748


namespace exists_lambda_divisibility_l92_92743
open Nat 

-- Define conditions
def coprime_ints_gt_one (p q : ℕ) : Prop := p > 1 ∧ q > 1 ∧ Nat.coprime p q

-- Define the sum of 2019th powers of all bad numbers for (p, q)
def sum_of_bad_numbers (p q : ℕ) : ℕ := sorry  -- Placeholder for the actual sum definition

-- Define the main theorem
theorem exists_lambda_divisibility (p q : ℕ) (hpq : coprime_ints_gt_one p q) :
  ∃ λ : ℕ, (p - 1) * (q - 1) ∣ λ * sum_of_bad_numbers p q := sorry

end exists_lambda_divisibility_l92_92743


namespace best_play_wins_majority_l92_92129

/-- Probability that the best play wins with a majority of the votes given the conditions -/
theorem best_play_wins_majority (n : ℕ) :
  let p := 1 - (1 / 2)^n
  in p > (1 - (1 / 2)^n) ∧ p ≤ 1 :=
sorry

end best_play_wins_majority_l92_92129


namespace smallest_n_rearranged_twice_is_r_l92_92015

def digit_rearrangement_condition (n r : ℕ) : Prop :=
  -- Define the condition that rearranging digits of n gives r
  -- This function captures the core idea behind the digit manipulation
  let digits_n := to_digits n in
  let digits_r := to_digits r in
  rearrange_digits digits_n = digits_r

theorem smallest_n_rearranged_twice_is_r :
  ∃ n : ℕ, (∀ m : ℕ, digit_rearrangement_condition m (2 * m) → m ≥ 263157894736842105) ∧ 
           digit_rearrangement_condition 263157894736842105 (2 * 263157894736842105) :=
sorry

end smallest_n_rearranged_twice_is_r_l92_92015


namespace shop_discount_l92_92807

/--
Given:
  - original_price: ℝ := 746.67
  - sale_price: ℝ := 560
Prove:
  - percent_discount == 25
--/
theorem shop_discount
  (original_price : ℝ) (sale_price : ℝ)
  (h : original_price = 746.67) (h2 : sale_price = 560) :
  let discount_amount := original_price - sale_price,
      percent_discount := (discount_amount / original_price) * 100
  in percent_discount ≈ 25 :=
by sorry

end shop_discount_l92_92807


namespace compute_f_expression_l92_92783

theorem compute_f_expression
  (f : ℝ → ℝ)
  (h_func_eq : ∀ x y : ℝ, f(x) * f(y) = f(Real.sqrt (x^2 + y^2)))
  (h_non_zero : ∃ x, f x ≠ 0) :
  f 1 - f 0 - f (-1) = -1 := 
sorry

end compute_f_expression_l92_92783


namespace smallest_positive_period_l92_92548

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi := 
begin
  sorry
end

end smallest_positive_period_l92_92548


namespace sin_ninety_deg_l92_92430

theorem sin_ninety_deg : Real.sin (Float.pi / 2) = 1 := 
by sorry

end sin_ninety_deg_l92_92430


namespace find_segment_O1O2_length_points_M_K_N_collinear_l92_92201

variable {R : Type} [RealField R]

noncomputable def radius (r : R) : R := r

structure Point :=
(x : R)
(y : R)

structure Circle :=
(center : Point)
(radius : R)

structure LineSegment :=
(start : Point)
(end : Point)

-- Definitions of the circles with centers O1 and O2 and equal radii.
def circle1 (O1 : Point) (r : R) : Circle := {
  center := O1,
  radius := r
}

def circle2 (O2 : Point) (r : R) : Circle :=
  center := O2,
  radius := r
}

-- Definition of the line segment O1O2
def segment_O1O2 (O1 O2 : Point) : LineSegment := {
  start := O1,
  end := O2
}

-- Hypotheses
variable (O1 O2 : Point)
variable (r : R)
variable (t : LineSegment)
variable (s : LineSegment)

-- Definitions of points
variable (K L M P N : Point)

-- Conditions
axiom O1_eq_Circle1_center : (circle1 O1 r).center = O1
axiom O2_eq_Circle2_center : (circle2 O2 r).center = O2
axiom O1O2_eq_segment_O1O2 : O1.x = 0 → O1.y = 0 → O2.x = 2 * r → O2.y = 0
axiom tangent_point_t : true -- Assume an appropriate definition for tangency and tangent points exist

-- Proof goals
theorem find_segment_O1O2_length : O1O2_eq_segment_O1O2 O1 O2 r → 
  segment_O1O2 O1 O2 r = 2 * radius r := 
begin
  sorry
end

theorem points_M_K_N_collinear : 
  is_collinear M K N := -- Assume an appropriate definition for collinearity exists
begin
  sorry
end

end find_segment_O1O2_length_points_M_K_N_collinear_l92_92201


namespace min_value_expression_l92_92014

theorem min_value_expression (x y : ℝ) : ∃ (c : ℝ), c = 0 ∧ (x^2 + 6 * x * y + 9 * y^2) = c :=
by
  use 0
  split
  -- Proof skipped
  sorry

end min_value_expression_l92_92014


namespace sufficient_but_not_necessary_condition_l92_92658

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  (a > b → a^3 + b^3 > a^2 * b + a * b^2) ∧ 
  (a^3 + b^3 > a^2 * b + a * b^2 → ¬ (a > b)) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l92_92658


namespace find_m_l92_92631

theorem find_m (a m : ℝ) (h_pos : a > 0) (h_points : (m, 3) ∈ set_of (λ x : ℝ × ℝ, ∃ x_val : ℝ, x.snd = -a * (x_val)^2 + 2 * a * x_val + 3)) (h_non_zero : m ≠ 0) : m = 2 := 
sorry

end find_m_l92_92631


namespace distance_circle_center_to_line_l92_92975

open Real

theorem distance_circle_center_to_line :
  let center := (-1 : ℝ, 0 : ℝ)
  let A := -2
  let B := 1
  let C := -3
  let distance := |A * center.fst + B * center.snd + C| / sqrt (A^2 + B^2)
  distance = sqrt 5 / 5 := sorry

end distance_circle_center_to_line_l92_92975


namespace sequences_equal_l92_92080

theorem sequences_equal (a b : ℕ → ℝ)
  (h1 : ∀ i j k r s t, a i + a j + a k = a r + a s + a t → (i, j, k) ∈ [(r, s, t).permutations])
  (h2 : ∀ x, 0 < x → (∃ i j, x = a j - a i) ↔ (∃ m n, x = b m - b n))
  (ha : ∀ n, a n < a (n + 1))
  (hb : ∀ n, b n < b (n + 1))
  (ha0 : 0 < a 1)
  (hb0 : b 0 = 0) :
  ∀ k, a k = b k :=
sorry

end sequences_equal_l92_92080


namespace Isabel_afternoon_runs_l92_92710

theorem Isabel_afternoon_runs (circuit_length morning_runs weekly_distance afternoon_runs : ℕ)
  (h_circuit_length : circuit_length = 365)
  (h_morning_runs : morning_runs = 7)
  (h_weekly_distance : weekly_distance = 25550)
  (h_afternoon_runs : weekly_distance = morning_runs * circuit_length * 7 + afternoon_runs * circuit_length) :
  afternoon_runs = 21 :=
by
  -- The actual proof goes here
  sorry

end Isabel_afternoon_runs_l92_92710


namespace square_position_l92_92238

def starting_position := "EFGH"
def rotated_position := "GHEF"
def reflected_position := "EFHG"
def rotated_reflected_position := "HGEF"

theorem square_position (n : ℕ) : n % 4 = 2 → seq_position n = "GHEF" :=
by
  sorry

end square_position_l92_92238


namespace find_m_eq_2_l92_92614

theorem find_m_eq_2 (a m : ℝ) (h1 : a > 0) (h2 : -a * m^2 + 2 * a * m + 3 = 3) (h3 : m ≠ 0) : m = 2 :=
by
  sorry

end find_m_eq_2_l92_92614


namespace product_less_two_l92_92771

theorem product_less_two (n : ℕ) : (∏ k in (range n).map (λ k, 1 + 1 / ((k + 1) * (k + 3)))) < 2 := by
  sorry

end product_less_two_l92_92771


namespace smallest_positive_root_floor_eq_three_l92_92740

noncomputable def g_func (x : ℝ) : ℝ := Real.cos x + 3 * Real.sin x + 2 * Real.cot x

def smallest_satisfying (P : ℝ → Prop) : ℝ := Inf {x : ℝ | 0 < x ∧ P x}

theorem smallest_positive_root_floor_eq_three :
  ∃ s > 0, g_func s = 0 ∧ ⌊s⌋ = 3 :=
by
  sorry

end smallest_positive_root_floor_eq_three_l92_92740


namespace tan_double_angle_l92_92057

-- Given conditions
variable (α : ℝ)
variable (h1 : ∀ n : ℤ, α ∈ (π * (2 * n + 1) : ℝ) .. π * (2 * n + 2))
variable (h2 : Real.cos (α + π) = 4 / 5)

-- The goal
theorem tan_double_angle : Real.tan (2 * α) = 24 / 7 := by
  sorry

end tan_double_angle_l92_92057


namespace part1_part2_l92_92882

/-
Part (1)
- Given conditions:
  - P: (0, 1)
  - l_1: 2x + y - 8 = 0
  - l_2: x - 3y + 10 = 0
- Prove: The equation of line l is x + 4y - 4 = 0.
-/

def point := (ℝ, ℝ)
def line (a b c : ℝ) : Prop := ∀ p : point, let (x, y) := p in a * x + b * y + c = 0

def line_through (p : point) (l : line) := l p

def bisected_by (p : point) (l : line) (l1 : line) (l2 : line) : Prop :=
  ∃ a : ℝ, 
  let (px, py) := p in
  let A := (a, 8 - 2 * a) in
  let B := (-a, 2 * a - 6) in
  l1 A ∧ l2 B ∧ px = 0 ∧ py = 1

theorem part1 (P : point) (l1 : line 2 1 (-8)) (l2 : line 1 (-3) 10) : 
  line_through P (line 1 4 (-4)) →
  bisected_by P (line 1 4 (-4)) l1 l2 :=
sorry

/-
Part (2)
- Given conditions:
  - l_1: x - 2y + 5 = 0
  - l: 3x - 2y + 7 = 0
- Prove: The equation of the reflected ray's line is 29x - 2y + 33 = 0.
-/

def reflection (l1 : line) (l : line) (reflected : line) : Prop :=
  ∃ M N K : point, 
  let (Mx, My) := M in 
  let (Nx, Ny) := N in 
  let (Kx, Ky) := K in
  M = (-1, 2) ∧ N = (-5, 0) ∧
  let K := (-17/13, -32/13) in
  l1 M ∧ l K ∧ reflected M ∧ reflected K

theorem part2 (l1 : line 1 (-2) 5) (l : line 3 (-2) 7) : 
  reflection l1 l (line 29 (-2) 33) :=
sorry

end part1_part2_l92_92882


namespace Mona_Sona_meet_first_time_l92_92760

-- Define speeds in terms of meters per second
def mona_speed_m_s : ℝ := 18 * (1000 / 3600)
def sona_speed_m_s : ℝ := 36 * (1000 / 3600)

-- Define the length of the track
def track_length : ℝ := 400

-- State the theorem: Mona and Sona will meet for the first time at the starting point after 80 seconds
theorem Mona_Sona_meet_first_time : ∃ t : ℝ, t = 80 ∧
  (sona_speed_m_s * t) % track_length = (mona_speed_m_s * t) % track_length :=
  sorry

end Mona_Sona_meet_first_time_l92_92760


namespace ratio_boys_to_girls_l92_92678

variables (B G : ℤ)

def boys_count : ℤ := 50
def girls_count (B : ℤ) : ℤ := B + 80

theorem ratio_boys_to_girls : 
  (B = boys_count) → 
  (G = girls_count B) → 
  ((B : ℚ) / (G : ℚ) = 5 / 13) :=
by
  sorry

end ratio_boys_to_girls_l92_92678


namespace magic_square_S_divisible_by_3_l92_92687

-- Definitions of the 3x3 magic square conditions
def is_magic_square (a : ℕ → ℕ → ℤ) (S : ℤ) : Prop :=
  (a 0 0 + a 0 1 + a 0 2 = S) ∧
  (a 1 0 + a 1 1 + a 1 2 = S) ∧
  (a 2 0 + a 2 1 + a 2 2 = S) ∧
  (a 0 0 + a 1 0 + a 2 0 = S) ∧
  (a 0 1 + a 1 1 + a 2 1 = S) ∧
  (a 0 2 + a 1 2 + a 2 2 = S) ∧
  (a 0 0 + a 1 1 + a 2 2 = S) ∧
  (a 0 2 + a 1 1 + a 2 0 = S)

-- Main theorem statement
theorem magic_square_S_divisible_by_3 :
  ∀ (a : ℕ → ℕ → ℤ) (S : ℤ),
    is_magic_square a S →
    S % 3 = 0 :=
by
  -- Here we assume the existence of the proof
  sorry

end magic_square_S_divisible_by_3_l92_92687
