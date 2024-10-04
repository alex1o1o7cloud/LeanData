import Mathlib

namespace part_one_part_two_l394_394797

/-- Problem Part 1: Given a quadratic function f(x) = x^2 - 2x - 3,
    prove g(x) = f(x) + a does not intersect the x-axis if and only if a > 4. -/
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, f x + a ≠ 0) ↔ a > 4 :=
begin
  sorry
end

/-- Problem Part 2: Given f(x) = x^2 - 2x - 3 and h(x) = x + 25 / x + b,
    prove that for any x1 in [1, 4], there exists an x2 in (1, 5] 
    such that f(x1) = h(x2) if and only if -21 < b ≤ -14. -/
theorem part_two (b : ℝ) : 
  (∀ x1 ∈ set.Icc (1 : ℝ) 4, ∃ x2 ∈ set.Ioc (1 : ℝ) 5, f x1 = h x2) ↔ -21 < b ∧ b ≤ -14 :=
begin
  sorry
end

/-- Definitions for function f and g in the theorem part_one. -/
def f (x : ℝ) : ℝ := x^2 - 2 * x - 3
def g (x : ℝ) (a : ℝ) : ℝ := f x + a

/-- Definitions for functions f and h in the theorem part_two. -/
def h (x : ℝ) (b : ℝ) : ℝ := x + 25 / x + b

end part_one_part_two_l394_394797


namespace total_books_in_week_l394_394949

def books_read (n : ℕ) : ℕ :=
  if n = 0 then 2 -- day 1 (indexed by 0)
  else if n = 1 then 2 -- day 2
  else 2 + n -- starting from day 3 (indexed by 2)

-- Summing the books read from day 1 to day 7 (indexed from 0 to 6)
theorem total_books_in_week : (List.sum (List.map books_read [0, 1, 2, 3, 4, 5, 6])) = 29 := by
  sorry

end total_books_in_week_l394_394949


namespace min_value_of_expression_l394_394604

theorem min_value_of_expression
  (x y : ℝ)
  (h1 : x - y ≥ 0)
  (h2 : x + y - 2 ≥ 0)
  (h3 : x ≤ 2) :
  ∃ min_val : ℝ, min_val = -1 / 2 ∧
    ∀ (x y : ℝ), (x - y ≥ 0) → (x + y - 2 ≥ 0) → (x ≤ 2) → x^2 + y^2 - 2 * x ≥ min_val :=
begin
  sorry
end

end min_value_of_expression_l394_394604


namespace find_a_l394_394795

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x + 3 else 4 / x

theorem find_a (a : ℝ) (h : f a = 2) : a = -1 ∨ a = 2 :=
sorry

end find_a_l394_394795


namespace circle_line_chord_length_l394_394444

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 4 = 0
noncomputable def line_equation (x y : ℝ) : Prop := x + y + 2 = 0
noncomputable def chord_length (L : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y ∧ line_equation x y ∧ 
              let r := Real.sqrt 6 in
              let d := Real.sqrt 2 in
              L = 2 * Real.sqrt (r^2 - d^2)

theorem circle_line_chord_length : chord_length 4 :=
sorry

end circle_line_chord_length_l394_394444


namespace cos_equivalent_l394_394480

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom mag_a : ∥a∥ = 1
axiom mag_b : ∥b∥ = 1
axiom mag_c : ∥c∥ = real.sqrt 2
axiom abc_zero : a + b + c = 0

-- Proof statement
theorem cos_equivalent : 
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
by sorry

end cos_equivalent_l394_394480


namespace magnitude_of_difference_l394_394839

-- Given definitions
def vec_a : ℝ × ℝ := (6, -2)
def vec_b (m : ℝ) : ℝ × ℝ := (3, m)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Target statement
theorem magnitude_of_difference (m : ℝ) (h_parallel : parallel vec_a (vec_b m)) :
  (let diff := (vec_a.1 - (vec_b m).1, vec_a.2 - (vec_b m).2) in
  real.sqrt (diff.1^2 + diff.2^2)) = real.sqrt 10 :=
by sorry

end magnitude_of_difference_l394_394839


namespace find_cos_Y_l394_394881

variable (XYZ : Triangle)
variable (angleZ : XYZ.angle = 90)
variable (XY XZ : ℝ)
variable (hXY : XY = 7)
variable (hXZ : XZ = 24)

noncomputable def lenghtYZ := Math.sqrt (XY ^ 2 + XZ ^ 2)

theorem find_cos_Y (hYZ : lenghtYZ = 25) : cos (XYZ.angle Y) = 24 / 25 := by
  sorry

end find_cos_Y_l394_394881


namespace f_prime_at_2_l394_394441

noncomputable def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := f(x) / x

-- Condition 1: The point (2, 1) lies on y = f(x) / x which gives f(2) = 2 
def condition1 : Prop := f(2) = 2

-- Condition 2: The lines tangent to y = f(x) at (0,0) and y = f(x) / x at (2,1) have the same slope == 1/2
def tangent_slope : ℝ := 1/2
def g' := λ x, (x * (Deriv f x) - f x) / (x^2)
def condition2 : Prop := g'(2) = tangent_slope

-- Proving f'(2) = 2
theorem f_prime_at_2 : condition1 ∧ condition2 → (Deriv f 2) = 2 :=
by
  sorry

end f_prime_at_2_l394_394441


namespace equation_1_solution_equation_2_solution_l394_394600

section EquationSolutions

variables {x : ℝ}

/-- Equation 1: Solve x^2 - 4x - 1 = 0 --/
theorem equation_1_solution : x^2 - 4x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 :=
by
  sorry

/-- Equation 2: Solve (x + 3)^2 = x + 3 --/
theorem equation_2_solution : (x + 3)^2 = x + 3 ↔ x = -3 ∨ x = -2 :=
by
  sorry

end EquationSolutions

end equation_1_solution_equation_2_solution_l394_394600


namespace jim_cards_l394_394913

theorem jim_cards
  (total_cards : ℕ)
  (cards_per_set : ℕ)
  (sister_sets : ℕ)
  (friend_sets : ℕ)
  (total_given : ℕ)
  (sister_cards : ℕ)
  (friend_cards : ℕ)
  (remaining_cards : ℕ)
  (brother_sets : ℕ)
  (h1 : total_cards = 365)
  (h2 : cards_per_set = 13)
  (h3 : sister_sets = 5)
  (h4 : friend_sets = 2)
  (h5 : total_given = 195)
  (h6 : sister_cards = sister_sets * cards_per_set)
  (h7 : friend_cards = friend_sets * cards_per_set)
  (h8 : remaining_cards = total_given - (sister_cards + friend_cards))
  (h9 : brother_sets = remaining_cards / cards_per_set) :
  brother_sets = 8 := 
begin
  sorry,
end

end jim_cards_l394_394913


namespace smaller_of_two_numbers_l394_394217

theorem smaller_of_two_numbers 
  (a b d : ℝ) (h : 0 < a ∧ a < b) (u v : ℝ) 
  (huv : u / v = b / a) (sum_uv : u + v = d) : 
  min u v = (a * d) / (a + b) :=
by
  sorry

end smaller_of_two_numbers_l394_394217


namespace find_a_for_monotonic_f_l394_394811

theorem find_a_for_monotonic_f :
  ∀ (a : ℝ), (∀ x₁ x₂, 3 ≤ x₁ → x₁ ≤ x₂ → |2 * x₁ + a| ≤ |2 * x₂ + a|) → a = -6 :=
begin
  intros a H,
  -- proof to be done
  sorry
end

end find_a_for_monotonic_f_l394_394811


namespace integer_solution_unique_l394_394770

variable (x y : ℤ)

def nested_sqrt_1964_times (x : ℤ) : ℤ := 
  sorry -- (This should define the function for nested sqrt 1964 times, but we'll use sorry to skip the proof)

theorem integer_solution_unique : 
  nested_sqrt_1964_times x = y → x = 0 ∧ y = 0 :=
by
  intros h
  sorry -- Proof of the theorem goes here

end integer_solution_unique_l394_394770


namespace range_of_a_l394_394852

theorem range_of_a (x a : ℝ) (α : x ≤ -1 ∨ x > 3) (β : a - 1 ≤ x ∧ x < a + 2) :
  (∀ x, β x → α x) ∧ ∃ x, α x ∧ ¬ β x ↔ a ≤ -3 ∨ a > 4 :=
by
  sorry

end range_of_a_l394_394852


namespace interior_angle_of_regular_hexagon_is_120_degrees_l394_394250

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l394_394250


namespace problem1_monotonic_increase_problem2_max_area_l394_394825

-- Given conditions and definitions
def f (x : ℝ) : ℝ := (1 / 2) * sin (2 * x) - cos (x + (real.pi / 4)) ^ 2

-- Problem 1: Prove interval(s) for monotonic increase
theorem problem1_monotonic_increase : 
  (∀ x ∈ Ioo 0 (real.pi / 4), has_deriv_at f x (cos 2 * x) x ∧ cos (2 * x) > 0) ∧ 
  (∀ x ∈ Ico (real.pi / 4) (3 * real.pi / 4), has_deriv_at f x (cos 2 * x) x ∧ cos (2 * x) < 0) :=
sorry

-- Problem 2: Prove the maximum area
theorem problem2_max_area (A B C : ℝ) (a b c : ℝ) 
  (acute_triangle : angle A < π / 2 ∧ angle B < π / 2 ∧ angle C < π / 2) 
  (b_eq_one : b = 1) 
  (f_B2_eq_zero : f (B / 2) = 0) :
  (∃ a c, a * c ≤ 2 + sqrt 3 ∧ triangle_area a b c = (2 + sqrt 3) / 4) :=
sorry

end problem1_monotonic_increase_problem2_max_area_l394_394825


namespace regular_hexagon_interior_angle_l394_394257

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l394_394257


namespace angle_between_a_b_is_zero_l394_394552

variables {a b c : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
axiom a_unit_vector : ‖a‖ = 1
axiom b_unit_vector : ‖b‖ = 1
axiom c_unit_vector : ‖c‖ = 1
axiom sum_zero : a + b + 2 • c = 0

-- Theorem statement
theorem angle_between_a_b_is_zero :
  real.angle a b = 0 :=
by sorry

end angle_between_a_b_is_zero_l394_394552


namespace range_of_k_for_triangle_function_l394_394505

def triangle_function (f : ℝ → ℝ) (D : set ℝ) : Prop :=
  ∀ (a b c : ℝ), a ∈ D → b ∈ D → c ∈ D → 
  let fa := f a in let fb := f b in let fc := f c 
  in fa + fb > fc ∧ fa + fc > fb ∧ fb + fc > fa

theorem range_of_k_for_triangle_function :
  let f (x : ℝ) := λ k, k * x + 2 
  let D := {x : ℝ | 1 ≤ x ∧ x ≤ 4}
  ∃ k : ℝ, triangle_function (f k) D ↔ k ∈ Ioo (-2/7) 1 :=
sorry

end range_of_k_for_triangle_function_l394_394505


namespace ceil_minus_floor_eq_one_implies_ceil_minus_y_l394_394929

noncomputable def fractional_part (y : ℝ) : ℝ := y - ⌊y⌋

theorem ceil_minus_floor_eq_one_implies_ceil_minus_y (y : ℝ) (h : ⌈y⌉ - ⌊y⌋ = 1) : ⌈y⌉ - y = 1 - fractional_part y :=
by
  sorry

end ceil_minus_floor_eq_one_implies_ceil_minus_y_l394_394929


namespace harmonic_sum_inequality_l394_394160

theorem harmonic_sum_inequality (n : ℕ) (hn : 0 < n) :
  ∑ k in Finset.range n, (1 / (n + k)) ≥ n * (Real.sqrt_n (2 : ℕ) n - 1) :=
sorry

end harmonic_sum_inequality_l394_394160


namespace savings_after_expense_increase_l394_394322

-- Define constants and initial conditions
def salary : ℝ := 7272.727272727273
def savings_rate : ℝ := 0.10
def expense_increase_rate : ℝ := 0.05

-- Define initial savings, expenses, and new expenses
def initial_savings : ℝ := savings_rate * salary
def initial_expenses : ℝ := salary - initial_savings
def new_expenses : ℝ := initial_expenses * (1 + expense_increase_rate)
def new_savings : ℝ := salary - new_expenses

-- The theorem statement
theorem savings_after_expense_increase : new_savings = 400 := by
  sorry

end savings_after_expense_increase_l394_394322


namespace no_solution_for_inequality_l394_394374

theorem no_solution_for_inequality (x : ℝ) : ¬ (3 * x^2 + 9 * x ≤ -12) :=
by
  sorry

end no_solution_for_inequality_l394_394374


namespace proof_problem_l394_394830

-- Define the given parabola and conditions
def parabola (p : ℕ) : Prop := ∃ (x y : ℝ), x^2 = 2 * p * y ∧ p > 0
def focus_distance : ℝ := 4
def midpoint_R (P Q R : ℝ × ℝ) : Prop := R = (4, 6) ∧ R = (1/2 * (P.1 + Q.1), 1/2 * (P.2 + Q.2))
def slope_is_one (P Q : ℝ × ℝ) : Prop := (Q.2 - P.2) / (Q.1 - P.1) = 1
def tangents_intersect (P Q A : ℝ × ℝ) : Prop := ∃ (tangent : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ), 
  tangent P = A ∧ tangent Q = A

def distance (A F : ℝ × ℝ) : ℝ := (dist A F)
def AF_distance (A F : ℝ × ℝ) : Prop := distance A F = 2 * √13

-- Combining all definitions into a single theorem statement
theorem proof_problem (P Q R A F : ℝ × ℝ) :
  parabola 4 ∧
  midpoint_R P Q R ∧
  slope_is_one P Q ∧
  tangents_intersect P Q A ∧
  AF_distance A (0, 4) :=
begin
  sorry
end

end proof_problem_l394_394830


namespace simplify_expression_l394_394161

variable (y : ℝ)

theorem simplify_expression : (5 * y + 6 * y + 7 * y + 2) = (18 * y + 2) := 
by
  sorry

end simplify_expression_l394_394161


namespace projection_incenter_or_excenter_of_triangle_l394_394622

theorem projection_incenter_or_excenter_of_triangle 
  {M A B C : Point} 
  (h_eq_dist : equidistant_from_lines M (line A B) (line B C) (line A C)) :
  let M_1 := orthogonal_projection M (plane A B C)
  in is_incenter M_1 (triangle A B C) ∨ is_excenter M_1 (triangle A B C) :=
sorry

end projection_incenter_or_excenter_of_triangle_l394_394622


namespace probability_meeting_semifinals_probability_meeting_final_l394_394882

-- Definitions used in the problem
variable {teams : Fin 8 → Prop}

-- Placeholder for main theorem - Part (a)
theorem probability_meeting_semifinals (A B : Fin 8) : 
  (probability (λ s, teams s) (λ s, s = A ∧ s ≠ B ∧ sorry)) = 1 / 14 :=
sorry

-- Placeholder for main theorem - Part (b)
theorem probability_meeting_final (A B : Fin 8) : 
  (probability (λ s, teams s) (λ s, s = A ∧ s ≠ B ∧ sorry)) = 1 / 28 :=
sorry

end probability_meeting_semifinals_probability_meeting_final_l394_394882


namespace Rebecca_tent_stakes_l394_394595

theorem Rebecca_tent_stakes : 
  ∃ T D W : ℕ, 
    D = 3 * T ∧ 
    W = T + 2 ∧ 
    T + D + W = 22 ∧ 
    T = 4 := 
by
  sorry

end Rebecca_tent_stakes_l394_394595


namespace angle_RST_measure_l394_394892

variables {R P Q S T : Type} [Point.real_point_class P] [Point.real_point_class Q] [Point.real_point_class R] [Point.real_point_class S] [Point.real_point_class T]

def isosceles_triangle (a b c : Point) (ab ac : Point) :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ ab = ac

def vertically_opposite_angles (a b : Angle) := a = b

def angle_sum_triple (a b c : ℝ) := a + b + c = 180

theorem angle_RST_measure
  (h₁: intersection_point R P T Q S)
  (h₂: isosceles_triangle P Q R PQ PR)
  (h₃: isosceles_triangle R S T RS RT)
  (h₄: ∠PQR = 2 * x) : ∠RST = 90 - x :=
by
  sorry

end angle_RST_measure_l394_394892


namespace coefficient_x2y2_l394_394999

theorem coefficient_x2y2 : 
  let expr1 := (1 + x) ^ 3
  let expr2 := (1 + y) ^ 4
  let C3_2 := Nat.choose 3 2
  let C4_2 := Nat.choose 4 2
  (C3_2 * C4_2 = 18) := by
    sorry

end coefficient_x2y2_l394_394999


namespace trapezoid_leg_length_l394_394610

theorem trapezoid_leg_length (S : ℝ) (h₁ : S > 0) : 
  ∃ x : ℝ, x = Real.sqrt (2 * S) ∧ x > 0 :=
by
  sorry

end trapezoid_leg_length_l394_394610


namespace polynomial_value_l394_394816

variable (x : ℝ)

theorem polynomial_value (h : x^2 - 2*x + 6 = 9) : 2*x^2 - 4*x + 6 = 12 :=
by
  sorry

end polynomial_value_l394_394816


namespace simplify_expression_l394_394988

-- Define the conditions
variable (x : ℝ)
variable (hx : sin x ≠ 0)

-- Define the statement
theorem simplify_expression :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by sorry

end simplify_expression_l394_394988


namespace ellipse_major_axis_length_l394_394183

theorem ellipse_major_axis_length : 
  ∀ (x y : ℝ), x^2 + 2 * y^2 = 2 → 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
by
  sorry

end ellipse_major_axis_length_l394_394183


namespace smallest_number_l394_394301

theorem smallest_number (a b c d : ℝ) (h₁ : a = -1) (h₂ : b = 0) (h₃ : c = 1) (h₄ : d = real.sqrt 3) : 
  a < b ∧ a < c ∧ a < d :=
by
  rw [h₁, h₂, h₃, h₄]
  exact ⟨by norm_num, by norm_num, by norm_num⟩

end smallest_number_l394_394301


namespace simplify_expression_l394_394986

-- Define the conditions
variable (x : ℝ)
variable (hx : sin x ≠ 0)

-- Define the statement
theorem simplify_expression :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by sorry

end simplify_expression_l394_394986


namespace cylinder_cube_volume_ratio_l394_394687

theorem cylinder_cube_volume_ratio (s : ℝ) (hs : 0 < s) :
  let r := s / 2 in
  let V_cylinder := π * r^2 * s in
  let V_cube := s^3 in
  V_cylinder / V_cube = π / 4 :=
by
  dsimp
  sorry

end cylinder_cube_volume_ratio_l394_394687


namespace interior_angle_of_regular_hexagon_is_120_degrees_l394_394254

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l394_394254


namespace isosceles_triangle_perimeter_l394_394191

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_eq_triangle : a + b + c = 60) (h_eq_sides : a = b) 
  (isosceles_base : c = 15) (isosceles_side1_eq : a = 20) : a + b + c = 55 :=
by
  sorry

end isosceles_triangle_perimeter_l394_394191


namespace median_of_data_mode_of_data_l394_394371

def data : List ℕ := [15, 17, 14, 10, 15, 17, 17, 16, 14, 12]

noncomputable def median (l : List ℕ) : ℝ := if even (l.length) then (l.get! (l.length / 2 - 1) + l.get! (l.length / 2)) / 2 else l.get! (l.length / 2)

noncomputable def mode (l : List ℕ) : ℕ := l.foldl (λ m x, if l.count x > l.count m then x else m) (l.head!)

theorem median_of_data : median data = 14.5 := sorry

theorem mode_of_data : mode data = 17 := sorry

end median_of_data_mode_of_data_l394_394371


namespace magnitude_of_complex_solution_l394_394603

theorem magnitude_of_complex_solution (w : ℂ) (h : w^2 + 2 * w = 11 - 16 * complex.I) : 
  complex.abs w = 17 ∨ complex.abs w = real.sqrt 89 :=
sorry

end magnitude_of_complex_solution_l394_394603


namespace yolanda_speed_proof_l394_394304

-- Define the conditions
def yolanda_speed (v : ℝ) : Prop := 
  let yolanda_time := 15.0 / 60.0 -- Yolanda bikes for 15/60 hours before husband starts
  let husband_time := 15.0 / 60.0 -- Husband drives for 15/60 hours to catch up
  let yolanda_distance := v * yolanda_time
  let husband_distance := 40 * husband_time
  yolanda_distance = husband_distance

-- Prove that Yolanda's speed is 40 miles per hour
theorem yolanda_speed_proof : ∃ v : ℝ, yolanda_speed v ∧ v = 40 :=
by
  exists 40
  unfold yolanda_speed
  simp
  sorry

end yolanda_speed_proof_l394_394304


namespace dog_max_distance_from_origin_l394_394955

noncomputable def greatest_distance_from_origin : ℝ :=
  let post := (6 : ℝ, 8 : ℝ)
  let rope_length : ℝ := 15
  let wall_y : ℝ := 5
  let origin := (0 : ℝ, 0 : ℝ)
  let center_to_origin := real.sqrt ((post.1 - origin.1) ^ 2 + (post.2 - origin.2) ^ 2)
  let max_point := (post.1, post.2 + rope_length)
  real.sqrt ((max_point.1 - origin.1) ^ 2 + (max_point.2 - origin.2) ^ 2)

theorem dog_max_distance_from_origin : 
  greatest_distance_from_origin = real.sqrt 565 :=
by
  sorry

end dog_max_distance_from_origin_l394_394955


namespace estimate_more_white_balls_l394_394868

theorem estimate_more_white_balls (total_draws : ℕ) (white_draws : ℕ)
  (h_total_draws : total_draws = 10)
  (h_white_draws : white_draws = 9)
  (h_large_difference : ∀ w b : ℕ, w ≠ b → w ≫ b ∨ b ≫ w)
  : ∃ w b : ℕ, w > b :=
by 
  -- Sorry is used to skip the proof.
  sorry

end estimate_more_white_balls_l394_394868


namespace roots_are_real_l394_394426

-- Defining the conditions
def eq1 (m n : ℝ) : Prop := 18 * m + 3 * n + 2 = 0
def eq2 (m n : ℝ) : Prop := 2 * m ≠ 0

-- The discriminant is non-negative
def discriminant_non_negative (m n : ℝ) : Prop :=
  let Δ := n ^ 2 - 4 * (2 * m) * 2 in
  Δ ≥ 0

-- The proof statement
theorem roots_are_real (m n : ℝ) (h1 : eq1 m n) (h2 : eq2 m n) : discriminant_non_negative m n :=
sorry

end roots_are_real_l394_394426


namespace quart_cost_l394_394325

def cost_per_quart_of_paint : ℝ :=
  let side_length_feet : ℝ := 10
  let surface_area_cube : ℝ := 6 * (side_length_feet ^ 2)
  let coverage_per_quart : ℝ := 60
  let total_cost : ℝ := 32
  let total_quarts_needed : ℝ := surface_area_cube / coverage_per_quart
  total_cost / total_quarts_needed

theorem quart_cost :
  let side_length_feet : ℝ := 10
  let surface_area_cube : ℝ := 6 * (side_length_feet ^ 2)
  let coverage_per_quart : ℝ := 60
  let total_cost : ℝ := 32
  let total_quarts_needed : ℝ := surface_area_cube / coverage_per_quart
  total_cost / total_quarts_needed = 3.20 :=
by
  let side_length_feet : ℝ := 10
  let surface_area_cube : ℝ := 6 * (side_length_feet ^ 2)
  let coverage_per_quart : ℝ := 60
  let total_cost : ℝ := 32
  let total_quarts_needed : ℝ := surface_area_cube / coverage_per_quart
  show total_cost / total_quarts_needed = 3.20 from sorry

end quart_cost_l394_394325


namespace table_seating_problem_l394_394139

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l394_394139


namespace interior_angle_of_regular_hexagon_l394_394223

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l394_394223


namespace find_integer_mod_condition_l394_394395

theorem find_integer_mod_condition (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 4) (h3 : n ≡ -998 [ZMOD 5]) : n = 2 :=
sorry

end find_integer_mod_condition_l394_394395


namespace find_x_minus_y_l394_394863

theorem find_x_minus_y (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 :=
by
  -- Proof omitted
  sorry

end find_x_minus_y_l394_394863


namespace simplify_trig_expression_l394_394982

variable {x : ℝ}

theorem simplify_trig_expression (h : 1 + cos x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l394_394982


namespace ab_minus_a_plus_b_eq_two_l394_394809

theorem ab_minus_a_plus_b_eq_two
  (a b : ℝ)
  (h1 : a + 1 ≠ 0)
  (h2 : b - 1 ≠ 0)
  (h3 : a + (1 / (a + 1)) = b + (1 / (b - 1)) - 2)
  (h4 : a - b + 2 ≠ 0)
: ab - a + b = 2 :=
sorry

end ab_minus_a_plus_b_eq_two_l394_394809


namespace translate_quadratic_vertex_right_l394_394640

theorem translate_quadratic_vertex_right : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = 2 * (x - 4)^2 - 3) ∧ 
  (∃ (g : ℝ → ℝ), (∀ x, g x = 2 * ((x - 1) - 3)^2 - 3))) → 
  (∃ v : ℝ × ℝ, v = (4, -3)) :=
sorry

end translate_quadratic_vertex_right_l394_394640


namespace point_in_third_quadrant_l394_394897

theorem point_in_third_quadrant
  (a b : ℝ)
  (hne : a ≠ 0)
  (y_increase : ∀ x1 x2, x1 < x2 → -5 * a * x1 + b < -5 * a * x2 + b)
  (ab_pos : a * b > 0) : 
  a < 0 ∧ b < 0 :=
by
  sorry

end point_in_third_quadrant_l394_394897


namespace shooter_random_event_l394_394300

def eventA := "The sun rises from the east"
def eventB := "A coin thrown up from the ground will fall down"
def eventC := "A shooter hits the target with 10 points in one shot"
def eventD := "Xiao Ming runs at a speed of 30 meters per second"

def is_random_event (event : String) := event = eventC

theorem shooter_random_event : is_random_event eventC := 
by
  sorry

end shooter_random_event_l394_394300


namespace interior_angle_regular_hexagon_l394_394243

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l394_394243


namespace range_of_a_l394_394616

theorem range_of_a {a : ℝ} : (∀ x y : ℝ, x <= 4 → y <= 4 → x < y → (x^2 - 2 * (1 - a) * x + 2 > y^2 - 2 * (1 - a) * y + 2)) → a ≤ -3 :=
by {
  -- Here we assume the correctness of the conditions and bounded range
  -- that ensures the function is decreasing.
  intros,
  sorry
}

end range_of_a_l394_394616


namespace regular_tetrahedron_inscription_l394_394960

variables {a b : ℝ}

def circumradius (a : ℝ) : ℝ :=
  (a * Real.sqrt 6) / 4

def inradius (b : ℝ) : ℝ :=
  (b * Real.sqrt 6) / 12

theorem regular_tetrahedron_inscription (h : circumradius a ≥ inradius b) : 3 * a ≥ b :=
  by
  -- Use the definitions of circumradius and inradius
  have h_circum : circumradius a = (a * Real.sqrt 6) / 4 := rfl
  have h_inrad : inradius b = (b * Real.sqrt 6) / 12 := rfl
  -- Substitute into the hypothesis
  rw [h_circum, h_inrad] at h
  -- Cancel the common factor of sqrt(6)
  have h_cancel_sqrt : (a * Real.sqrt 6) / 4 ≥ (b * Real.sqrt 6) / 12 := h
  rw [Real.div_div_eq_div_mul, Real.div_div_eq_div_mul] at h_cancel_sqrt
  -- Simplify the fractions
  have h_simplified : a / 4 ≥ b / 12 := by { exact h_cancel_sqrt }
  -- Multiply both sides of the inequality by 12
  linarith

end regular_tetrahedron_inscription_l394_394960


namespace union_eq_l394_394940

open Set

variable {α : Type*} [LinearOrder α]

def M : Set α := {x | 0 < x ∧ x < 1}
def N : Set α := {x | -3 < x ∧ x < 3}

theorem union_eq :
  M ∪ N = N :=
by
  sorry

end union_eq_l394_394940


namespace distance_from_center_to_plane_ABC_l394_394417

noncomputable def radius_of_sphere : ℝ := Real.sqrt 3

structure PointsOnSphere where
  P A B C : EuclideanSpace ℝ (Fin 3)
  on_sphere : ∀ p ∈ {P, A, B, C}, ∥p∥ = radius_of_sphere

theorem distance_from_center_to_plane_ABC
  (points : PointsOnSphere)
  (PA_perp_PB : InnerProductSpace.inner points.P points.A = 0)
  (PB_perp_PC : InnerProductSpace.inner points.P points.B = 0)
  (PC_perp_PA : InnerProductSpace.inner points.P points.C = 0) :
  ∃ dist : ℝ, dist = Real.sqrt 3 / 3 := 
sorry

end distance_from_center_to_plane_ABC_l394_394417


namespace cone_spheres_radius_l394_394531

theorem cone_spheres_radius (R H : ℝ) (hR : R = 4) (hH : H = 15) 
    (r : ℝ) (h : r ≠ 0):
  let s := Real.sqrt (R^2 + H^2),
      x := 14 - r,
      y := (2 * Real.sqrt 3 / 3) * r,
      equation := Real.sqrt 241 - r = Real.sqrt (x^2 + y^2)
  in equation → r = 1.5 :=
by
  intros
  sorry

end cone_spheres_radius_l394_394531


namespace smallest_n_for_divisibility_l394_394651

theorem smallest_n_for_divisibility (n : ℕ) (h : 2 ∣ 3^(2*n) - 1) (k : ℕ) : n = 2^(2007) := by
  sorry

end smallest_n_for_divisibility_l394_394651


namespace vector_sum_zero_l394_394464

variables {α : Type*} [add_comm_group α]
variables (a b c : α)

-- Definitions of conditions
def not_collinear (u v : α) : Prop :=
  ¬ ∃ (k : ℝ), u = k • v

def collinear (u v : α) : Prop :=
  ∃ (k : ℝ), u = k • v

axiom h_not_collinear_ab : not_collinear a b
axiom h_not_collinear_ac : not_collinear a c
axiom h_not_collinear_bc : not_collinear b c
axiom h_collinear_ab_c : collinear (a + b) c
axiom h_collinear_bc_a : collinear (b + c) a

-- The theorem to prove the answer
theorem vector_sum_zero : a + b + c = (0 : α) :=
by sorry

end vector_sum_zero_l394_394464


namespace no_such_n_exists_l394_394763

noncomputable def is_partitionable (s : Finset ℕ) : Prop :=
  ∃ (A B : Finset ℕ), A ∪ B = s ∧ A ∩ B = ∅ ∧ (A.prod id = B.prod id)

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, n > 0 ∧ is_partitionable {n, n+1, n+2, n+3, n+4, n+5} :=
by
  sorry

end no_such_n_exists_l394_394763


namespace mrs_hilt_hot_dogs_l394_394948

theorem mrs_hilt_hot_dogs (cost_per_hotdog total_cost : ℕ) (h1 : cost_per_hotdog = 50) (h2 : total_cost = 300) :
  total_cost / cost_per_hotdog = 6 := by
  sorry

end mrs_hilt_hot_dogs_l394_394948


namespace trigonometric_identity_proof_l394_394994

theorem trigonometric_identity_proof (α : ℝ) :
  (sin (2 * α) + sin (5 * α) - sin (3 * α)) / (cos α + 1 - 2 * (sin (2 * α))^2) = 2 * sin α :=
  sorry

end trigonometric_identity_proof_l394_394994


namespace correct_statements_l394_394302

-- Definitions derived from conditions
def ray (A B : Point) : Prop := true -- You can further define properties specific to rays as needed

def line (A B : Point) : Prop := true -- You can further define properties specific to lines as needed

-- Fundamental principles from conditions
axiom unique_line : ∀ (A B : Point), A ≠ B → ∃! l, line A B
axiom three_points_three_lines : ∀ (A B C : Point), A ≠ B ∧ B ≠ C ∧ A ≠ C →  ∃! l1 l2 l3, (line A B) ∧ (line B C) ∧ (line A C)

-- Translated proof problem statement
theorem correct_statements (A B C : Point) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) :
  (three_points_three_lines A B C (and.intro h1 (and.intro h2 h3))) ∧ (unique_line A B h1) := 
sorry

end correct_statements_l394_394302


namespace solve_MQ_above_A_l394_394891

-- Definitions of the given conditions
def ABCD_side := 8
def MNPQ_length := 16
def MNPQ_width := 8
def area_outer_inner_ratio := 1 / 3

-- Definition to prove
def length_MQ_above_A := 8 / 3

-- The area calculations
def area_MNPQ := MNPQ_length * MNPQ_width
def area_ABCD := ABCD_side * ABCD_side
def area_outer := (area_outer_inner_ratio * area_MNPQ)
def MQ_above_A_calculated := area_outer / MNPQ_length

theorem solve_MQ_above_A :
  MQ_above_A_calculated = length_MQ_above_A := by sorry

end solve_MQ_above_A_l394_394891


namespace gain_amount_is_75_l394_394324

-- Definitions based on conditions
def selling_price : ℝ := 225
def gain_percentage : ℝ := 50 / 100

-- The gain amount
def gain_amount (cp sp : ℝ) : ℝ := sp - cp

-- Prove the final gain amount is $75
theorem gain_amount_is_75 (cp : ℝ) (h1 : selling_price = 1.5 * cp) :
  gain_amount cp selling_price = 75 :=
sorry

end gain_amount_is_75_l394_394324


namespace distance_between_X_and_Y_l394_394582

theorem distance_between_X_and_Y (yolanda_rate : ℝ) (bob_rate : ℝ) (bob_distance : ℝ) (yolanda_head_start : ℝ) : 
  yolanda_rate = 3 ∧ bob_rate = 4 ∧ bob_distance = 16 ∧ yolanda_head_start = 1 → 
  ∃ D : ℝ, D = 31 :=
by
  intros h
  cases h with hy hb
  cases hb with hd hh
  sorry

end distance_between_X_and_Y_l394_394582


namespace odd_four_digit_strictly_decreasing_count_l394_394492

theorem odd_four_digit_strictly_decreasing_count : 
  ∃ count : ℕ, count = 105 ∧ 
  count = (finset.univ.filter (λ x : finset (fin (10)), ∃ (d1 d2 d3 d4 : ℕ),
    d1 ∈ finset.range 10 ∧ d2 ∈ finset.range 10 ∧ d3 ∈ finset.range 10 ∧ d4 ∈ finset.range 10 ∧
    d1 > d2 ∧ d2 > d3 ∧ d3 > d4 ∧ d4 % 2 = 1 ∧ 
    1000 * d1 + 100 * d2 + 10 * d3 + d4 < 10000 ∧ 
    1000 * d1 + 100 * d2 + 10 * d3 + d4 ≥ 1000)).card :=
begin
  -- Proof is skipped
  sorry
end

end odd_four_digit_strictly_decreasing_count_l394_394492


namespace soccer_team_lineups_l394_394951

-- Define the number of players in the team
def numPlayers : Nat := 16

-- Define the number of regular players to choose (excluding the goalie)
def numRegularPlayers : Nat := 10

-- Define the total number of starting lineups, considering the goalie and the combination of regular players
def totalStartingLineups : Nat :=
  numPlayers * Nat.choose (numPlayers - 1) numRegularPlayers

-- The theorem to prove
theorem soccer_team_lineups : totalStartingLineups = 48048 := by
  sorry

end soccer_team_lineups_l394_394951


namespace people_at_table_l394_394123

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l394_394123


namespace simplify_trig_expression_l394_394972

theorem simplify_trig_expression (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := by
  sorry

end simplify_trig_expression_l394_394972


namespace time_to_complete_race_l394_394670

theorem time_to_complete_race (v t : ℝ) 
    (h1 : v * (t + 10) = 900) 
    (h2 : v * t = 1000) : 
    t = 100 := 
begin
  -- Proof omitted 
  sorry
end

end time_to_complete_race_l394_394670


namespace marsha_remainder_l394_394945

noncomputable def remainder_a_b_mod_30 (a b : ℕ) : ℕ :=
  let a_mod_60 := 58
  let b_mod_90 := 84
  if (a % 60 = a_mod_60) ∧ (b % 90 = b_mod_90) then
    (a + b) % 30
  else
    0 -- Invalid inputs, cannot determine the correct answer

theorem marsha_remainder (a b : ℕ)
  (h1 : a % 60 = 58)
  (h2 : b % 90 = 84) :
  remainder_a_b_mod_30 a b = 22 :=
begin
  sorry
end

end marsha_remainder_l394_394945


namespace tangent_line_at_point_l394_394769

noncomputable def curve : ℝ → ℝ := λ x, sin x / x
noncomputable def point_M : ℝ × ℝ := (2 * Real.pi, 0)
noncomputable def tangent_line_eqn (x y : ℝ) : Prop := x - 2 * Real.pi * y - 2 * Real.pi = 0

theorem tangent_line_at_point (x y : ℝ) (h_curve : curve (2 * Real.pi) = 0) 
  (h_tangent : y = (1 / (2 * Real.pi)) * (x - 2 * Real.pi)) :
  tangent_line_eqn x y :=
begin
  sorry
end

end tangent_line_at_point_l394_394769


namespace exchange_points_configurations_possible_l394_394956

/-- 
Given a city divided into parts by roads, with one existing currency exchange point,
prove that it is possible to construct up to four different configurations 
such that each part of the city contains exactly two exchange points
by adding three additional exchange points.
--/
theorem exchange_points_configurations_possible : 
  ∃ (configs : list (set (set point))), 
  length configs ≤ 4 ∧ ∀ config ∈ configs, 
  ∀ partition : set point, partition ∈ partitions city → count_of_points_in partition = 2 :=
sorry

end exchange_points_configurations_possible_l394_394956


namespace angle_between_a_and_a_plus_3b_l394_394551

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_a_and_a_plus_3b (a b : V) 
  (h : ∥a + 2 • b∥ = ∥b∥) : angle (a + 3 • b) a = π / 2 :=
begin
  sorry
end

end angle_between_a_and_a_plus_3b_l394_394551


namespace divisible_by_120_l394_394564

theorem divisible_by_120 (n : ℕ) (hn_pos : n > 0) : 120 ∣ n * (n^2 - 1) * (n^2 - 5 * n + 26) := 
by
  sorry

end divisible_by_120_l394_394564


namespace intersection_height_is_20_l394_394216

noncomputable def pole_intersection_height (h1 h2 d : ℝ) : ℝ :=
  let line1 := λ x, - (h1 / d) * x + h1
  let line2 := λ x, (h2 / d) * x
  let x_intersect := d * h1 / (h1 + h2)
  line2 x_intersect

theorem intersection_height_is_20 (h1 h2 d : ℝ) (h1_eq : h1 = 30) (h2_eq : h2 = 60) (d_eq : d = 120) :
  pole_intersection_height h1 h2 d = 20 :=
by
  rw [h1_eq, h2_eq, d_eq]
  simp [pole_intersection_height, h1_eq, h2_eq, d_eq]
  sorry

end intersection_height_is_20_l394_394216


namespace more_cans_on_tuesday_l394_394101

-- Condition: Mike collected 71 cans on Monday
def cansOnMonday : ℕ := 71

-- Condition: Mike collected a total of 98 cans altogether
def totalCans : ℕ := 98

-- Theorem to prove the number of cans collected on Tuesday minus
-- the number of cans collected on Monday, taken absolute value, is 44.
theorem more_cans_on_tuesday : abs (totalCans - cansOnMonday - cansOnMonday) = 44 := 
sorry

end more_cans_on_tuesday_l394_394101


namespace christopher_more_than_karen_l394_394543

-- Define the number of quarters Karen and Christopher have
def karen_quarters : ℕ := 32
def christopher_quarters : ℕ := 64

-- Define the value of a quarter in dollars
def value_of_quarter : ℚ := 0.25

-- Define the amount of money Christopher has more than Karen in dollars
def christopher_more_money : ℚ := (christopher_quarters - karen_quarters) * value_of_quarter

-- Theorem to prove that Christopher has $8.00 more than Karen
theorem christopher_more_than_karen : christopher_more_money = 8 := by
  sorry

end christopher_more_than_karen_l394_394543


namespace no_complete_path_l394_394704

def Chessboard : Type := fin 6 × fin 6

def valid_jump (start end_ : Chessboard) : Prop :=
  let (sx, sy) := start
  let (ex, ey) := end_
  (sx = ex ∧ abs (sy.1 - ey.1) = 1) ∨ -- move 1 square vertically
  (sy = ey ∧ abs (sx.1 - ex.1) = 1) ∨ -- move 1 square horizontally
  (sx = ex ∧ abs (sy.1 - ey.1) = 2) ∨ -- move 2 squares vertically
  (sy = ey ∧ abs (sx.1 - ex.1) = 2)   -- move 2 squares horizontally

def valid_sequence (squares : list Chessboard) : Prop :=
  ∀ (i : ℕ), i < 34 → (i % 2 = 0 → ∃ sq, sq ∈ squares ∧ valid_jump (squares.nth i).get (squares.nth (i + 1)).get) ∧
                          (i % 2 = 1 → ∃ sq, sq ∈ squares ∧ valid_jump (squares.nth i).get (squares.nth (i + 1)).get)

theorem no_complete_path : ¬ ∃ (start : Chessboard) (squares : list Chessboard),
  squares.length = 36 ∧ squares.head = start ∧ valid_sequence squares :=
sorry

end no_complete_path_l394_394704


namespace tangent_line_ellipse_l394_394386

noncomputable def ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : Prop :=
  (a^2 = 5) ∧ (b^2 = 4)

theorem tangent_line_ellipse :
  (∃ A B : ℝ × ℝ, 
    let P : ℝ × ℝ := (1, 1/2)
    ∧ ∃ k : ℝ, 
        (tangent_to_circle (λ x y : ℝ, x^2 + y^2 = 1) P k)
        ∧ (line_through A B = line_through_foci_and_vertex_of_ellipse 5 4))
  → ellipse_equation 5 4 :=
sorry

end tangent_line_ellipse_l394_394386


namespace triangles_share_excenter_l394_394806

open EuclideanGeometry

variables {O P A B Q C D : Point}

-- Geometric Hypotheses
axiom tangent1 : Tangent O P A
axiom tangent2 : Tangent O P B
axiom intersection : Intersects OP (Line A B) Q
axiom chord : Chord O C D

-- Symmetric properties
axiom symmetric_intersection_point : ∃ Q : Point, Midpoint Q A B ∧ Intersects OP (Line A B) Q

-- Main Statement: Proof that the two triangles share an excenter
theorem triangles_share_excenter
  (h1 : Tangent O P A)
  (h2 : Tangent O P B)
  (h3 : Intersects OP (Line A B) Q)
  (h4 : Chord O C D)
  (h5 : SymmetricIntersectionPoint OP A B Q)
  : ExistsExcenter (Triangle P A B) (Triangle P C D) :=
  sorry

end triangles_share_excenter_l394_394806


namespace sin_cos_identity_l394_394674

theorem sin_cos_identity (a b : ℝ) :
  sin 160 * cos 10 + cos 20 * sin 10 = 1 / 2 :=
by
-- Use the sum of angles identity
have h1 : sin 160 = sin (180 - 20), by sorry,
-- Simplify sin(180 - θ) = sin(θ)
have h2 : sin (180 - 20) = sin 20, by sorry,
-- Apply the sum of angles identity
have h3 : sin 20 * cos 10 + cos 20 * sin 10 = sin (20 + 10), by sorry,
-- Simplify sin(30) = 1/2
have h4 : sin (20 + 10) = sin 30, by sorry,
have h5 : sin 30 = 1 / 2, by sorry,
show sin 160 * cos 10 + cos 20 * sin 10 = 1 / 2, from by rw [h1, h2, h3, h4, h5]

end sin_cos_identity_l394_394674


namespace midpoints_parallel_to_base_l394_394963

variable {α : Type*} [LinearOrderedField α]

-- Define the points and assumptions
variables (A B C M N : α × α) 

-- Define the midpoints M and N
def isMidpoint (X Y Z : α × α) : Prop :=
  Y = (X + Z) / 2

-- Define the isosceles triangle
def isIsoscelesTriangle (A B C : α × α) : Prop :=
  dist A B = dist A C

-- Statement of the problem
theorem midpoints_parallel_to_base 
  (h_mid_M : isMidpoint A M B)
  (h_mid_N : isMidpoint A N C)
  (h_iso : isIsoscelesTriangle A B C) :
  ∃ K : α × α, isMidpoint K M N ∧ K = (M + N) / 2 ∧ (vector_to_segment (M, N)) ∥ (vector_to_segment (B, C)) :=
sorry

end midpoints_parallel_to_base_l394_394963


namespace polygon_sum_13th_position_l394_394710

theorem polygon_sum_13th_position :
  let sum_n : ℕ := (100 * 101) / 2;
  2 * sum_n = 10100 :=
by
  sorry

end polygon_sum_13th_position_l394_394710


namespace value_of_f_15_l394_394858

def f (n : ℕ) : ℕ := n^2 + 2*n + 19

theorem value_of_f_15 : f 15 = 274 := 
by 
  -- Add proof here
  sorry

end value_of_f_15_l394_394858


namespace prob_normal_interval_l394_394416

noncomputable theory
open MeasureTheory ProbabilityTheory Real

def xi_dist : ProbabilityDistribution ℝ := ProbabilityDistribution.normal 0 σ^2
def prob_greater_than_3 : ℝ := xi_dist.measure {(x : ℝ) | x > 3}
def prob_between_neg3_and_3 : ℝ := xi_dist.measure {x : ℝ | -3 ≤ x ∧ x ≤ 3}

axiom prob_greater_than_3_spec : prob_greater_than_3 = 0.023

theorem prob_normal_interval : prob_between_neg3_and_3 = 0.954 := by
  have symmetry := {-- derive the symmetry property based on normal distribution
    sorry
  }
  have prob_less_than_neg3 := {-- from symmetry and given conditions
    sorry
  }
  -- use total probability with symmetry to calculate the interval probability
  sorry

end prob_normal_interval_l394_394416


namespace keith_pears_picked_l394_394074

theorem keith_pears_picked (jason_picked : ℕ) (mike_picked : ℕ) (total_picked : ℕ) :
  jason_picked = 46 → mike_picked = 12 → total_picked = 105 → 
  let keith_picked := total_picked - (jason_picked + mike_picked) in
  keith_picked = 47 :=
by {
  intros h1 h2 h3,
  simp [h1, h2, h3]
}

end keith_pears_picked_l394_394074


namespace value_of_abc_l394_394859

noncomputable def f (a b c x : ℝ) := a * x^2 + b * x + c
noncomputable def f_inv (a b c x : ℝ) := c * x^2 + b * x + a

-- The main theorem statement
theorem value_of_abc (a b c : ℝ) (h : ∀ x : ℝ, f a b c (f_inv a b c x) = x) : a + b + c = 1 :=
sorry

end value_of_abc_l394_394859


namespace b7_in_form_l394_394091

theorem b7_in_form (a : ℕ → ℚ) (b : ℕ → ℚ) : 
  a 0 = 3 → 
  b 0 = 5 → 
  (∀ n : ℕ, a (n + 1) = (a n)^2 / (b n)) → 
  (∀ n : ℕ, b (n + 1) = (b n)^2 / (a n)) → 
  b 7 = (5^50 : ℚ) / (3^41 : ℚ) := 
by 
  intros h1 h2 h3 h4 
  sorry

end b7_in_form_l394_394091


namespace num_unique_values_expression_l394_394893

theorem num_unique_values_expression : 
  ∃ S : Finset ℤ, 
    (∀ {a b c d : ℕ}, 
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
      a ∈ {1, 2, 3, 4} ∧ b ∈ {1, 2, 3, 4} ∧ c ∈ {1, 2, 3, 4} ∧ d ∈ {1, 2, 3, 4} → 
      (a * b - c * d) ∈ S) ∧
    S.card = 9 := 
sorry

end num_unique_values_expression_l394_394893


namespace find_f_neg_one_l394_394004

-- Definition: f(x) is an odd function and for x >= 0, f(x) = x * (1 + x)
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def f (x : ℝ) : ℝ := if x ≥ 0 then x * (1 + x) else -(x * (1 + -x))

theorem find_f_neg_one (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_def : ∀ x, x ≥ 0 → f x = x * (1 + x)) : f (-1) = -2 :=
by
  sorry

end find_f_neg_one_l394_394004


namespace equiangular_iff_rectangle_l394_394699

theorem equiangular_iff_rectangle (Q : Type) [quadrilateral Q] : 
  (∀ (a b c d : angle), a = b ∧ b = c ∧ c = d ∧ d = 90) ↔ rectangle Q :=
sorry

end equiangular_iff_rectangle_l394_394699


namespace calc_julia_payment_l394_394679

theorem calc_julia_payment :
  let base_cost := 30.0
  let text_cost := 0.04
  let extra_minute_cost := 0.12
  let email_cost := 0.20
  let num_texts := 150
  let hours_used := 26.5
  let included_hours := 25.0
  let num_emails := 25
  let cost : ℝ :=
    base_cost +
    (text_cost * num_texts) +
    (extra_minute_cost * (hours_used - included_hours) * 60) +
    (email_cost * num_emails)
  in cost = 51.8 :=
by
  let base_cost := 30.0
  let text_cost := 0.04
  let extra_minute_cost := 0.12
  let email_cost := 0.20
  let num_texts := 150
  let hours_used := 26.5
  let included_hours := 25.0
  let num_emails := 25
  let cost : ℝ :=
    base_cost +
    (text_cost * num_texts) +
    (extra_minute_cost * (hours_used - included_hours) * 60) +
    (email_cost * num_emails)
  show cost = 51.8
  sorry

end calc_julia_payment_l394_394679


namespace ounces_per_serving_l394_394968

theorem ounces_per_serving (x : ℕ) : x = 8 :=
  let daily_water := 64
  let current_serving_size := 16
  let current_servings := daily_water / current_serving_size
  let previous_servings := daily_water / x
  have h1 : current_servings = 4 := by sorry
  have h2 : previous_servings = current_servings + 4 := by sorry
  have h3 : daily_water / x = 8 := by rw [h2, h1]; sorry
  eq_of_mul_eq_mul_right (by norm_num) (by rw [mul_comm, ← nat.mul_div_cancel', h3]; norm_num)
  sorry

end ounces_per_serving_l394_394968


namespace jake_spent_on_motorcycle_l394_394534

theorem jake_spent_on_motorcycle :
  ∀ (initial money spent concert ticket loss remaining motorcycle : ℝ),
  initial = 5000 →
  remaining = 825 →
  loss = (remaining / 3) →
  concert ticket = (2 * (remaining + loss)) →
  spent = (initial - concert ticket) →
  motorcycle = spent →
  motorcycle = 2800 :=
by
  intros initial money spent concert ticket loss remaining motorcycle
  assume h1 : initial = 5000
  assume h2 : remaining = 825
  assume h3 : loss = (remaining / 3)
  assume h4 : concert ticket = (2 * (remaining + loss))
  assume h5 : spent = (initial - concert ticket)
  assume h6 : motorcycle = spent
  sorry

end jake_spent_on_motorcycle_l394_394534


namespace length_AC_l394_394589

theorem length_AC {A B C D E : Point}
  (hD_on_AC : D ∈ line_segment A C)
  (hE_on_BC : E ∈ line_segment B C)
  (AD_eq_EC : distance A D = distance E C)
  (BD_eq_ED : distance B D = distance E D)
  (angle_BDC_eq_DEB : angle B D C = angle D E B)
  (AB_eq_7 : distance A B = 7)
  (BE_eq_2 : distance B E = 2) :
  distance A C = 12 :=
sorry

end length_AC_l394_394589


namespace degree_measure_of_regular_hexagon_interior_angle_l394_394283

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l394_394283


namespace bounded_complex_sequence_l394_394567

theorem bounded_complex_sequence 
  (a_0 b_0 c_0 : ℂ) 
  (a : ℕ → ℂ) 
  (b : ℕ → ℂ) 
  (c : ℕ → ℂ) 
  (h_init : a 0 = a_0 ∧ b 0 = b_0 ∧ c 0 = c_0)
  (h_rec : ∀ n, a (n + 1) = (a n)^2 + 2 * (b n) * (c n) 
                 ∧ b (n + 1) = (b n)^2 + 2 * (c n) * (a n) 
                 ∧ c (n + 1) = (c n)^2 + 2 * (a n) * (b n))
  (h_bounded : ∀ n, max (|a n|) (max (|b n|) (|c n|)) ≤ 2022) :
  |a_0|^2 + |b_0|^2 + |c_0|^2 ≤ 1 :=
sorry

end bounded_complex_sequence_l394_394567


namespace cube_coloring_schemes_l394_394414

-- Define the problem conditions
-- A cube with six faces, colors already chosen for three faces meeting at vertex A.
def cube_faces := {1, 2, 3, 4, 5, 6}
def available_colors := {a, b, c, d, e}
def chosen_colors := {a, b, c}  -- Colors for faces 1, 2, and 3

-- Define the statement to be proved
theorem cube_coloring_schemes : 
  ∃ color_schemes : ℕ, color_schemes = 13 :=
by 
  sorry

end cube_coloring_schemes_l394_394414


namespace inequality_a_c_b_l394_394087

noncomputable def a : ℝ := Real.logBase 2 (1 / 3)
noncomputable def b : ℝ := 2 ^ 1.1
noncomputable def c : ℝ := 0.8 ^ 2.3

theorem inequality_a_c_b : a < c ∧ c < b :=
by {
  have h_a : a = Real.logBase 2 (1 / 3), by sorry,
  have h_b : b = 2 ^ 1.1, by sorry,
  have h_c : c = 0.8 ^ 2.3, by sorry,
  have h_a_neg : a < 0, by sorry,
  have h_b_pos : b > 1, by sorry,
  have h_c_range : 0 < c ∧ c < 1, by sorry,
  exact ⟨lt_of_lt_of_le h_a_neg h_c_range.right, lt_of_le_of_lt h_c_range.right h_b_pos⟩
}

end inequality_a_c_b_l394_394087


namespace billboard_color_schemes_l394_394954

theorem billboard_color_schemes : 
  let num_billboards := 4,
  let colors := ["red", "blue"],
  ∀ color_scheme : (Fin num_billboards → String), 
  (∀ i : Fin (num_billboards - 1), color_scheme i = "red" → color_scheme (i + 1) ≠ "red") →
  ∃ (valid_schemes : Finset (Fin num_billboards → String)), 
  valid_schemes.count = 8 :=
by
  sorry

end billboard_color_schemes_l394_394954


namespace probability_less_equal_2_l394_394876

variable {σ : ℝ} (hσ : σ > 0) -- σ > 0
variable (ξ : ℝ → ℝ) -- random variable ξ

-- ξ follows normal distribution
axiom normal_distribution : ∀ x : ℝ, ξ x = (1 / (σ * sqrt (2 * π))) * exp (- (x - 1)^2 / (2 * σ^2))

-- Probability of ξ taking values in interval (0, 2) is 0.8
axiom probability_interval_0_2 : (∫ x in 0..2, (1 / (σ * sqrt (2 * π))) * exp (- (x - 1)^2 / (2 * σ^2))) = 0.8

-- Conclude probability of ξ taking values in the interval (-∞, 2] is 0.9
theorem probability_less_equal_2 : (∫ x in -∞..2, (1 / (σ * sqrt (2 * π))) * exp (- (x - 1)^2 / (2 * σ^2))) = 0.9 :=
  sorry

end probability_less_equal_2_l394_394876


namespace uniq_increasing_seq_l394_394764

noncomputable def a (n : ℕ) : ℕ := n -- The correct sequence a_n = n

theorem uniq_increasing_seq (a : ℕ → ℕ)
  (h1 : a 2 = 2)
  (h2 : ∀ n m : ℕ, a (n * m) = a n * a m)
  (h_inc : ∀ n m : ℕ, n < m → a n < a m) : ∀ n : ℕ, a n = n := by
  -- Here we would place the proof, skipping it for now with sorry
  sorry

end uniq_increasing_seq_l394_394764


namespace interior_angle_of_regular_hexagon_is_120_degrees_l394_394247

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l394_394247


namespace sqrt_div_five_l394_394366

theorem sqrt_div_five (a b : ℕ) (h : 4^2 * 5^3 = 2000) : 
  (real.sqrt (4^2 * 5^3)) / 5 = 4 * real.sqrt 5 :=
by {
  rw h,
  sorry,
}

end sqrt_div_five_l394_394366


namespace interior_angle_of_regular_hexagon_is_120_degrees_l394_394249

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l394_394249


namespace greatest_difference_smaller_number_l394_394636

theorem greatest_difference_smaller_number :
  ∀ (d1 d2 d3 : ℕ), (d1 = 5 ∧ d2 = 9 ∧ d3 = 2) →
  let largest := d2 * 100 + d1 * 10 + d3,
      smallest := d3 * 100 + d1 * 10 + d2 in
  largest - smallest = 693 ∧ smallest = 259 :=
begin
  intros d1 d2 d3 h,
  rcases h with ⟨h1, h2, h3⟩,
  let largest := d2 * 100 + d1 * 10 + d3,
  let smallest := d3 * 100 + d1 * 10 + d2,
  have h_largest : largest = 952, by {
    simp [h1, h2, h3],
  },
  have h_smallest : smallest = 259, by {
    simp [h1, h2, h3],
  },
  have h_diff : largest - smallest = 693, by {
    simp [h_largest, h_smallest],
  },
  tauto,
end

end greatest_difference_smaller_number_l394_394636


namespace travel_time_in_minutes_l394_394313

def bird_speed : ℝ := 8 -- Speed of the bird in miles per hour
def distance_to_travel : ℝ := 3 -- Distance to be traveled in miles

theorem travel_time_in_minutes : (distance_to_travel / bird_speed) * 60 = 22.5 :=
by
  sorry

end travel_time_in_minutes_l394_394313


namespace overall_profit_percentage_correct_l394_394321

def firstBookCP : ℕ := 50
def firstBookSP : ℕ := 80
def secondBookCP : ℕ := 100
def secondBookSP : ℕ := 130
def thirdBookCP : ℕ := 150
def thirdBookSP : ℕ := 190

def totalCostPrice : ℕ := firstBookCP + secondBookCP + thirdBookCP
def totalSellingPrice : ℕ := firstBookSP + secondBookSP + thirdBookSP
def totalProfit : ℕ := totalSellingPrice - totalCostPrice

def profitPercentage : ℚ := (totalProfit.toRat / totalCostPrice.toRat) * 100

theorem overall_profit_percentage_correct : profitPercentage = 33.33 := 
by sorry

end overall_profit_percentage_correct_l394_394321


namespace count_multiples_in_range_l394_394495

open Nat

theorem count_multiples_in_range :
  let lcm_value := lcm 11 8 in
  let multiples := (range 301).filter (λ n, n ≥ 100 ∧ n % lcm_value = 0) in
  multiples.length = 2 :=
by
  -- define the lcm_value as the least common multiple of 11 and 8
  let lcm_value := lcm 11 8
  -- find all multiples of lcm_value in the range [100, 300]
  let multiples := (range 301).filter (λ n, n ≥ 100 ∧ n % lcm_value = 0)
  -- check the length of the multiples list
  have h : multiples.length = 2 := sorry
  exact h

end count_multiples_in_range_l394_394495


namespace correct_fraction_simplification_l394_394855

theorem correct_fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (∀ (c d : ℝ), (c ≠ d) → (a+2 = c → b+2 = d → (a+2)/d ≠ a/b))
  ∧ (∀ (e f : ℝ), (e ≠ f) → (a-2 = e → b-2 = f → (a-2)/f ≠ a/b))
  ∧ (∀ (g h : ℝ), (g ≠ h) → (a^2 = g → b^2 = h → a^2/h ≠ a/b))
  ∧ (a / b = ( (1/2)*a / (1/2)*b )) := 
sorry

end correct_fraction_simplification_l394_394855


namespace hiram_roommate_sheets_l394_394841

theorem hiram_roommate_sheets (sheets_removed : ℕ) :
  let pages : ℕ := 60
  let total_sheets : ℕ := 30
  let avg_page_number := 21
  (∃ b c, sheets_removed = c ∧ 
          let sum_before := b * (2 * b + 1)
          let pages_after := (2 * (b + c) + 1 + 60) / 2 * (60 - 2 * c - 2 * b) / 2
          ((sum_before + pages_after) / (pages - 2 * c)).1 = avg_page_number) →
  (sheets_removed = 10 ∨ sheets_removed = 12 ∨ sheets_removed = 15) := 
by
  intro sheets_removed pages total_sheets avg_page_number H
  sorry

end hiram_roommate_sheets_l394_394841


namespace seated_people_count_l394_394130

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l394_394130


namespace verify_statements_l394_394040

theorem verify_statements (a b c x0 : ℝ) (h1 : f x = x^3 + a * x^2 + b * x + c) (h2 : set_of(x = -1 ∨ x = 1 ∨ x = x0) ⊆ set_of(f x = 0)) (h3 : 2 < x0 ∧ x0 < 3) :
  (a + c = 0) ∧ (c ∈ Ioo 2 3) ∧ (4 * a + 2 * b + c < -8) :=
begin
  sorry
end

end verify_statements_l394_394040


namespace football_total_points_l394_394894

theorem football_total_points :
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  Zach_points + Ben_points + Sarah_points + Emily_points = 109.0 :=
by
  let Zach_points := 42.0
  let Ben_points := 21.0
  let Sarah_points := 18.5
  let Emily_points := 27.5
  have h : Zach_points + Ben_points + Sarah_points + Emily_points = 42.0 + 21.0 + 18.5 + 27.5 := by rfl
  have total_points := 42.0 + 21.0 + 18.5 + 27.5
  have result := 109.0
  sorry

end football_total_points_l394_394894


namespace corridor_width_l394_394879

theorem corridor_width
  (a k h w : ℝ)
  (angle_q : ℝ := 60)
  (angle_r : ℝ := 70)
  (cos_eq_one_sub_cos_square_half_pi_sub_angle : ∀ x : ℝ, real.cos x = 1 - real.sin(real.pi/2 - x)^2)
  (cot_eq_cos_div_sin : ∀ x : ℝ, real.cot x = real.cos x / real.sin x) :
  w = h * real.cot (angle_r) + k * real.cot (angle_q) :=
by
  sorry

end corridor_width_l394_394879


namespace tangent_line_eq_at_point_l394_394615

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def a : ℝ := 1 / Real.exp 1
noncomputable def b : ℝ := f a
noncomputable def tangent_eq (x : ℝ) : ℝ := -1 / Real.exp 1

theorem tangent_line_eq_at_point : ∀ x : ℝ, f' a = 0 → f a = b → tangent_eq x = -1 / Real.exp 1 :=
by
  intros x h_deriv h_val
  sorry

end tangent_line_eq_at_point_l394_394615


namespace find_values_of_n_l394_394794

theorem find_values_of_n:
  (∀ a : ℕ → ℕ, (∀ x : ℝ, (1 - x - x^2 - x^3) * (∑ n in (nat.range ∞), a n * x ^ n) = 1) →
  ∀ n : ℕ, (a (n-1) = n^2) → (n = 1 ∨ n = 9)) :=
by
  sorry

end find_values_of_n_l394_394794


namespace correct_quotient_is_58_l394_394309

def incorrect_divisor : ℕ := 87
def incorrect_quotient : ℕ := 24
def correct_divisor : ℕ := 36

theorem correct_quotient_is_58 : (incorrect_divisor * incorrect_quotient) / correct_divisor = 58 := by
  calc 
  incorrect_divisor * incorrect_quotient = 2088 := by norm_num
  2088 / correct_divisor = 58 := by norm_num

end correct_quotient_is_58_l394_394309


namespace triangle_C_is_60_triangle_side_c_l394_394026

noncomputable def vectors_and_angles (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) : Prop :=
  m = (Real.sin A, Real.cos A) ∧
  n = (Real.cos B, Real.sin B) ∧
  m.1 * n.1 + m.2 * n.2 = Real.sin (2 * C) ∧
  A + B + C = 180 ∧
  Real.sin A + Real.sin B = 2 * Real.sin C

theorem triangle_C_is_60 
    (A B C a b c : ℝ) 
    (h: vectors_and_angles A B C a b c (\sin A, \cos A) (\cos B, \sin B)) : 
    C = 60 :=
  by {
   sorry
  }

noncomputable def angles_and_dot_product (A B C : ℝ) (a b c x : ℝ) : Prop :=
  Real.sin A + Real.sin B + Real.sin C = 3 * Real.sin C ∧
  Real.sin C = (√3) / 2 ∧ 
  x = 18 ∧ 
  (Real.sin C ^ 2 = (Real.sin A * Real.sin B))

theorem triangle_side_c 
    (A B C a b c x : ℝ)
    (h: angles_and_dot_product A B C a b c x) :
    c = 6 :=
  by {
   sorry
  }

end triangle_C_is_60_triangle_side_c_l394_394026


namespace simplify_trig_expression_l394_394970

theorem simplify_trig_expression (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := by
  sorry

end simplify_trig_expression_l394_394970


namespace calculate_expression_l394_394733

theorem calculate_expression :
  (sqrt 2 * sqrt 6 + abs (2 - sqrt 3) - (1 / 2) ^ (-2)) = sqrt 3 - 2 :=
by
  sorry

end calculate_expression_l394_394733


namespace find_a_l394_394822

noncomputable def f (a x : ℝ) : ℝ := Real.log x (a + 1/2)
def g (x : ℝ) : ℝ := x^2 + 4 * x - 2

def h (a x : ℝ) : ℝ :=
  if f a x ≥ g x then f a x else g x

theorem find_a : ∃ a : ℝ, (∀ x : ℝ, h a x ≥ -2) := 
sorry

end find_a_l394_394822


namespace function_solution_l394_394820

theorem function_solution (f : ℝ → ℝ) (a : ℝ) : 
  (∀ x : ℝ, f x = sorry) → f a = sorry → (a = 1 ∨ a = -1) :=
by
  intros hfa hfb
  sorry

end function_solution_l394_394820


namespace evaluate_expression_l394_394760

theorem evaluate_expression : 
  let a := 45
  let b := 15
  (a + b)^2 - (a^2 + b^2 + 2 * a * 5) = 900 :=
by
  let a := 45
  let b := 15
  sorry

end evaluate_expression_l394_394760


namespace machine_b_performs_better_l394_394944

-- Defining the defect data for both machines
def machine_a_data : List ℕ := [0, 1, 0, 2, 2, 0, 3, 1, 2, 4]
def machine_b_data : List ℕ := [2, 3, 1, 1, 0, 2, 1, 1, 0, 1]

-- Function to calculate the average of a list of natural numbers
def average (l : List ℕ) : ℚ :=
  (l.toNat.sum l).toRat / l.length.toRat

-- Function to calculate the variance of a list of natural numbers
def variance (l : List ℕ) : ℚ :=
  let avg := average l
  (l.map (λ x => (x.toRat - avg) ^ 2)).sum / l.length.toRat

-- The main theorem to prove that machine B performs better than machine A
theorem machine_b_performs_better :
  let avg_a := average machine_a_data
  let var_a := variance machine_a_data
  let avg_b := average machine_b_data
  let var_b := variance machine_b_data
  avg_a > avg_b ∧ var_a > var_b → true :=
sorry

end machine_b_performs_better_l394_394944


namespace valid_values_for_D_l394_394888

-- Definitions for the distinct digits and the non-zero condition
def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9
def distinct_nonzero_digits (A B C D : ℕ) : Prop :=
  is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Condition for the carry situation
def carry_in_addition (A B C D : ℕ) : Prop :=
  ∃ carry1 carry2 carry3 carry4 : ℕ,
  (A + B + carry1) % 10 = D ∧ (B + C + carry2) % 10 = A ∧
  (C + C + carry3) % 10 = B ∧ (A + B + carry4) % 10 = C ∧
  (carry1 = 1 ∨ carry2 = 1 ∨ carry3 = 1 ∨ carry4 = 1)

-- Main statement
theorem valid_values_for_D (A B C D : ℕ) :
  distinct_nonzero_digits A B C D →
  carry_in_addition A B C D →
  ∃ n, n = 5 :=
sorry

end valid_values_for_D_l394_394888


namespace total_people_seated_l394_394152

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l394_394152


namespace intersection_of_set_M_with_complement_of_set_N_l394_394084

theorem intersection_of_set_M_with_complement_of_set_N (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 4, 5}) (hN : N = {1, 3}) : M ∩ (U \ N) = {4, 5} :=
by
  sorry

end intersection_of_set_M_with_complement_of_set_N_l394_394084


namespace perfect_square_factors_l394_394031

theorem perfect_square_factors (a b c : ℕ) (hA : a = 12) (hB : b = 10) (hC : c = 14) :
  let factors2 := finset.range (a / 2 + 1) * 2,
  let factors3 := finset.range (b / 2 + 1) * 2,
  let factors7 := finset.range (c / 2 + 1) * 2,
  (factors2.card * factors3.card * factors7.card) = 336 :=
by
  sorry

end perfect_square_factors_l394_394031


namespace reduce_to_one_in_11_operations_l394_394421

theorem reduce_to_one_in_11_operations : 
  ∃ n : ℕ, n = 81 ∧ (∃ k : ℕ, reduce_to_one_with_operations n k ∧ k = 11) :=
sorry

-- Hypothetical definition of the operation to reduce number of tiles
def reduce_to_one_with_operations (n : ℕ) (k : ℕ) : Prop :=
  ∀ (m : ℕ), m ≤ k →
    (m % 2 = 1 → isNotPerfectSquare (f (n % m))) ∧
    (m % 2 = 0 → isNotPrime (f (n % m)))

def isNotPerfectSquare (n : ℕ) : Prop :=
  ¬ ∃ m : ℕ, m * m = n

def isNotPrime (n : ℕ) : Prop := 
  ¬ nat.prime n

def f (n : ℕ) : ℕ :=
  -- Renumbering function, to be defined as per problem's context
  sorry

end reduce_to_one_in_11_operations_l394_394421


namespace find_k_b_and_angle_ACB_l394_394037

theorem find_k_b_and_angle_ACB :
  ∃ (k b : ℝ) (∠ACB : ℝ),
    (∀ (x y : ℝ),
      x^2 + y^2 + 8 * x - 4 * y = 0 → 
      x + 4 = 0 ∧ y - 2 = 0 ∧ k = 2 ∧ b = 5) ∧
    (∠ACB = 120) :=
  sorry

end find_k_b_and_angle_ACB_l394_394037


namespace remove_two_terms_sum_one_l394_394338

theorem remove_two_terms_sum_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (60/120) + (30/120) + (20/120) + (15/120) + (12/120) + (10/120)
  let target_sum := 1
  ∃ (a b : ℚ), a ∈ fractions ∧ b ∈ fractions ∧ 
    (a ≠ b) ∧ 
    (total_sum - (a + b) = target_sum) 
    ∧ ((a = 1/8 ∧ b = 1/10) ∨ (a = 1/10 ∧ b = 1/8)) := 
by
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let fractions_set := fractions.to_finset
  let total_sum := 147 / 120
  let target := 120 / 120
  use [1/8, 1/10]
  simp [Fractions, fractions_set, total_sum, target]
  split
  { exact sorry }
  { split
    { rw [1/8, 1/10, total_sum, target], norm_num }
    { simp, norm_num }}
  sorry

end remove_two_terms_sum_one_l394_394338


namespace playdough_cost_l394_394916

-- Definitions of the costs and quantities
def lego_cost := 250
def sword_cost := 120
def playdough_quantity := 10
def total_paid := 1940

-- Variables representing the quantities bought
def lego_quantity := 3
def sword_quantity := 7

-- Function to calculate the total cost for lego and sword
def total_lego_cost := lego_quantity * lego_cost
def total_sword_cost := sword_quantity * sword_cost

-- Variable representing the cost of playdough
variable (P : ℝ)

-- The main statement to prove
theorem playdough_cost :
  total_lego_cost + total_sword_cost + playdough_quantity * P = total_paid → P = 35 :=
by
  sorry

end playdough_cost_l394_394916


namespace regular_hexagon_interior_angle_l394_394269

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l394_394269


namespace positive_multiples_of_12_two_digit_count_l394_394845

open Nat

theorem positive_multiples_of_12_two_digit_count : 
  {k : ℕ // 10 ≤ 12 * k ∧ 12 * k ≤ 99}.card = 8 := by
  sorry

end positive_multiples_of_12_two_digit_count_l394_394845


namespace sum_of_units_digits_eq_0_l394_394774

-- Units digit function definition
def units_digit (n : ℕ) : ℕ :=
  n % 10

-- Problem statement in Lean 
theorem sum_of_units_digits_eq_0 :
  units_digit (units_digit (17 * 34) + units_digit (19 * 28)) = 0 :=
by
  sorry

end sum_of_units_digits_eq_0_l394_394774


namespace cos_angle_sub_vectors_l394_394483

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (H1 : ∥a∥ = 1) (H2 : ∥b∥ = 1) (H3 : ∥c∥ = real.sqrt 2)
variables (H4 : a + b + c = 0)

theorem cos_angle_sub_vectors :
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry

end cos_angle_sub_vectors_l394_394483


namespace count_primes_with_digit_three_l394_394847

def is_digit_three (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := Prime n

def primes_with_digit_three_count (lim : ℕ) (count : ℕ) : Prop :=
  ∀ n < lim, is_digit_three n → is_prime n → count = 9

theorem count_primes_with_digit_three (lim : ℕ) (count : ℕ) :
  primes_with_digit_three_count 150 9 := 
by
  sorry

end count_primes_with_digit_three_l394_394847


namespace max_neg_integers_l394_394669

-- Definitions for the conditions
def areIntegers (a b c d e f : Int) : Prop := True
def sumOfProductsNeg (a b c d e f : Int) : Prop := (a * b + c * d * e * f) < 0

-- The theorem to prove
theorem max_neg_integers (a b c d e f : Int) (h1 : areIntegers a b c d e f) (h2 : sumOfProductsNeg a b c d e f) : 
  ∃ s : Nat, s = 4 := 
sorry

end max_neg_integers_l394_394669


namespace sum_of_angles_in_triangle_measure_of_angle_C_l394_394515

/-- Sum of internal angles in a triangle is 180 degrees -/
theorem sum_of_angles_in_triangle (A B C : ℝ) (h : A + B = 80) : A + B + C = 180 := sorry

/-- Measure of angle C in a triangle given angle A and B sum to 80 degrees -/
theorem measure_of_angle_C (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 :=
by 
  rw [<- add_assoc] at h2
  have h : C = 180 - (A + B) by linarith,
  rw h,
  rw h1,
  norm_num,
  assumption

end sum_of_angles_in_triangle_measure_of_angle_C_l394_394515


namespace find_f_prime_2_l394_394435

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f x / x

axiom tangent_coincide_at_points : ∀ f : ℝ → ℝ,
  (deriv f 0 = 1 / 2) ∧ (deriv (λ x, f x / x) 2 = 1 / 2)

theorem find_f_prime_2 :
  ∀ f : ℝ → ℝ, (f 2 = 2) ∧ ((∀ x, f 0 = 0)) ∧ tangent_coincide_at_points f → deriv f 2 = 2 :=
begin
  sorry
end

end find_f_prime_2_l394_394435


namespace square_perimeter_is_64_l394_394332

-- Given conditions
variables (s : ℕ)
def is_square_divided_into_four_congruent_rectangles : Prop :=
  ∀ (r : ℕ), r = 4 → (∀ (p : ℕ), p = (5 * s) / 2 → p = 40)

-- Lean 4 statement for the proof problem
theorem square_perimeter_is_64 
  (h : is_square_divided_into_four_congruent_rectangles s) 
  (hs : (5 * s) / 2 = 40) : 
  4 * s = 64 :=
by
  sorry

end square_perimeter_is_64_l394_394332


namespace fraction_simplifies_correctly_l394_394857

variable (a b : ℕ)

theorem fraction_simplifies_correctly (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := 
by sorry

end fraction_simplifies_correctly_l394_394857


namespace regular_hexagon_interior_angle_l394_394265

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l394_394265


namespace prime_factor_sum_1540_l394_394653

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p : ℕ, is_prime p ∧ p ∣ n then
    Classical.choose h
  else
    n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p : ℕ, is_prime p ∧ p ∣ n then
    Classical.choose (Exists.choose_spec h)
  else
    n

theorem prime_factor_sum_1540 :
  let spf := smallest_prime_factor 1540,
      lpf := largest_prime_factor 1540
  in 2 ≤ spf ∧ spf ≤ lpf ∧ lpf ≤ 1540 ∧ spf + lpf = 13 :=
by
  sorry

end prime_factor_sum_1540_l394_394653


namespace most_least_and_average_sorted_l394_394606

-- Define the daily sorting deviations
def monday_sort : ℤ := 6
def tuesday_sort : ℤ := 4
def wednesday_sort : ℤ := -6
def thursday_sort : ℤ := 8
def friday_sort : ℤ := -1
def saturday_sort : ℤ := 7
def sunday_sort : ℤ := -4

-- Define the planned average sorting volume per day (in 10,000s)
def planned_daily_sort : ℤ := 20

-- The list of sorting deviations for each day
def sorting_deviations : List ℤ := [monday_sort, tuesday_sort, wednesday_sort, thursday_sort, friday_sort, saturday_sort, sunday_sort]

-- Define the conditions for each requirement
theorem most_least_and_average_sorted :
  sorting_deviations.nth 3 = some 8 ∧
  sorting_deviations.nth 2 = some -6 ∧
  8 - (-6) = 14 ∧
  (sorting_deviations.sum + 7 * planned_daily_sort) / 7 = 22 :=
by
  sorry

end most_least_and_average_sorted_l394_394606


namespace perfect_square_a_value_l394_394864

theorem perfect_square_a_value (x y a : ℝ) :
  (∃ k : ℝ, x^2 + 2 * x * y + y^2 - a * (x + y) + 25 = k^2) →
  a = 10 ∨ a = -10 :=
sorry

end perfect_square_a_value_l394_394864


namespace range_of_f_l394_394011

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin (x - Real.pi / 4) * Real.sin (x + Real.pi / 4)

theorem range_of_f : ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), 
  -Real.sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end range_of_f_l394_394011


namespace Jean_average_speed_correct_l394_394734

/-!
Chantal starts walking at 5 miles per hour. Two-thirds of the way to the tower,
the trail becomes steep, and she slows down to 3 miles per hour.
Chantal descends the steep part of the trail at 4 miles per hour.
She meets Jean two-thirds of the way to the tower.
What was Jean's average speed, in miles per hour, until they meet?
-/

noncomputable def Jean_average_speed : ℝ :=
  let d := 1 in  -- We can assume d = 1 for simplicity since it scales linearly.
  let t1 := (2 * d) / 5 in -- Time for first segment by Chantal
  let t2 := d / 3 in -- Time for second segment by Chantal
  let t3 := (d / 3) / 4 in -- Time for descent by Chantal
  let T := t1 + t2 - t3 in -- Total time taken by Chantal to meet Jean
  (2 * d) / T -- Average speed of Jean

theorem Jean_average_speed_correct : Jean_average_speed = 1.6 :=
by
  rw [Jean_average_speed]
  have h : let d := 1 in (2 * d / (2 * d / 5 + d / 3 - (d / 3 / 4))) = 1.6, by norm_num
  exact h

end Jean_average_speed_correct_l394_394734


namespace min_value_l394_394775

variable {α : Type*} [LinearOrderedField α]

-- Define a geometric sequence with strictly positive terms
def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ (q : α), q > 0 ∧ ∀ n, a (n + 1) = a n * q

-- Given conditions
variables (a : ℕ → α) (S : ℕ → α)
variables (h_geom : is_geometric_sequence a)
variables (h_pos : ∀ n, a n > 0)
variables (h_a23 : a 2 * a 6 = 4) (h_a3 : a 3 = 1)

-- Sum of the first n terms of a geometric sequence
def sum_first_n (a : ℕ → α) (n : ℕ) : α :=
  if n = 0 then 0
  else a 0 * ((1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0)))

-- Statement of the theorem
theorem min_value (a : ℕ → α) (S : ℕ → α) 
  (h_geom : is_geometric_sequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a23 : a 2 * a 6 = 4)
  (h_a3 : a 3 = 1)
  (h_Sn : ∀ n, S n = sum_first_n a n) :
  ∃ n, n = 3 ∧ (S n + 9 / 4) ^ 2 / (2 * a n) = 8 :=
sorry

end min_value_l394_394775


namespace john_saves_7680_per_year_l394_394541

-- Define old rent cost
def oldRent := 1200

-- Define the increase percentage for new apartment costs
def increasePercent := 0.40

-- Define the new rent cost
def newRent := oldRent + oldRent * increasePercent

-- Define the number of people sharing the cost
def numPeople := 3

-- Define John's share of the new apartment cost
def johnsShare := newRent / numPeople

-- Define monthly savings
def monthlySavings := oldRent - johnsShare

-- Define yearly savings
def yearlySavings := monthlySavings * 12

-- Theorem statement
theorem john_saves_7680_per_year : yearlySavings = 7680 := 
by {
  -- Placeholder for actual proof,
  -- but statement builds successfully.
  sorry
}

end john_saves_7680_per_year_l394_394541


namespace part1_part2_l394_394784

-- Definitions and conditions
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := (finset.range n).sum (λ i, a i)
def geometric_seq (a : ℕ → ℝ) : Prop := ∃ q, ∀ n, a (n + 1) = q * a n
def arithmetic_seq (s : ℕ → ℝ) : Prop := ∀ n, s (n + 1) = 2 * s n - s (n - 1)

-- Given conditions
variables {a : ℕ → ℝ} (q : ℝ)
hypothesis h1 : geometric_seq a
hypothesis h2 : arithmetic_seq (λ n, S n a)

-- Goal for part 1
theorem part1 (h1 : geometric_seq a) (h2 : arithmetic_seq (λ n, S n a))
: (2 : ℝ) * a 5 = a 2 + a 8 := sorry

-- Given conditions for part 2
variables {b : ℕ → ℝ}
hypothesis h3 : b 1 = a 2
hypothesis h4 : b 3 = a 5

-- Goal for part 2
theorem part2 (h1 : geometric_seq a) 
              (h3 : b 1 = a 2)
              (h4 : b 3 = a 5):
  (∀ n, b n = -3/4 * n + 7/4) → 
  ∃ (T : ℕ → ℝ), T n = -5/3 + ((5 - 3 * n)/3) * ((-1/2) ^ n) := sorry

end part1_part2_l394_394784


namespace subset_with_three_colors_exists_l394_394900

noncomputable def complete_graph (V : Type) : Type :=
  { e : V × V // e.1 ≠ e.2 }

theorem subset_with_three_colors_exists :
  ∀ (V : Type) [Fintype V] [DecidableEq V], Fintype.card V = 2004 →
  (E : complete_graph V → fin 4) →
  ∃ (S : set (complete_graph V)), set.card S >= 3 ∧ 
  (∃ c1 c2 c3 : fin 4, c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧ 
    (∀ e ∈ S, E e = c1 ∨ E e = c2 ∨ E e = c3)) :=
begin
  intros V instFintype instDecEq h_card E,
  sorry
end

end subset_with_three_colors_exists_l394_394900


namespace interior_triangles_from_chords_l394_394105

theorem interior_triangles_from_chords (h₁ : ∀ p₁ p₂ p₃ : Prop, ¬(p₁ ∧ p₂ ∧ p₃)) : 
  ∀ (nine_points_on_circle : Finset ℝ) (h₂ : nine_points_on_circle.card = 9), 
    ∃ (triangles : ℕ), triangles = 210 := 
by 
  sorry

end interior_triangles_from_chords_l394_394105


namespace infinite_monochromatic_subgraph_l394_394089

-- Define the infinite complete graph with k colors.
constant K_infty : Type
constant edge_color : K_infty → K_infty → Fin k

-- The main theorem to prove the existence of an infinite monochromatic subgraph.
theorem infinite_monochromatic_subgraph {k : ℕ} (h : ∀ (v1 v2 : K_infty), edge_color v1 v2 < k) :
  ∃ (G : set K_infty), (G.infinite ∧ (∀ (v1 v2 : K_infty), v1 ≠ v2 → v1 ∈ G → v2 ∈ G → edge_color v1 v2 = edge_color v2 v1)) :=
sorry

end infinite_monochromatic_subgraph_l394_394089


namespace number_of_people_seated_l394_394145

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l394_394145


namespace cosine_of_subtracted_vectors_l394_394472

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Conditions
def condition1 := ∥a∥ = 1
def condition2 := ∥b∥ = 1
def condition3 := ∥c∥ = real.sqrt 2
def condition4 := a + b + c = 0

-- Proof statement
theorem cosine_of_subtracted_vectors 
  (h1 : condition1 a) 
  (h2 : condition2 b) 
  (h3 : condition3 c) 
  (h4 : condition4 a b c) : 
  inner_product_space.cos (a - c) (b - c) = 4 / 5 :=
sorry

end cosine_of_subtracted_vectors_l394_394472


namespace tangent_line_circle_l394_394500

theorem tangent_line_circle (r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = r^2 → x + y = 2 * r) ↔ r = 2 + Real.sqrt 2 :=
by
  sorry

end tangent_line_circle_l394_394500


namespace part_I_part_II_l394_394448

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

theorem part_I :
  ∀ (x : ℝ), x = Real.pi / 12 ∨ x = 7 * Real.pi / 12 → f x = 3 ∨ f x = -3 :=
by
  intro x hx
  cases hx
  { rw [hx, f]
    norm_num
    rw Real.sin_add
    norm_num
    iterate 2 { rw Real.sin_pi_div_three }
    rw [neg_div Int.cast_pos (by norm_num), neg_one_mul, to_real_neg_one_gte_coeff],
  { 
    rw hx 
    rw f
    norm_num
    rw Real.sin_add
    repeat {rw [unenf]}
    sorry

theorem part_II :
  ∀ (x : ℝ), -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 6 → 
  -3 * Real.sqrt 3 / 2 ≤ f x ∧ f x ≤ 3 :=
by
  intro x hx
  cases hx
  { rw [Real.sin_add, multiple_coeff]
    rw [result from one_alpha(tree_x)]
    rw second_ev(suppression) sorry

end part_I_part_II_l394_394448


namespace line_parallel_or_on_plane_l394_394024

noncomputable def equidistant (A B C : Point) (α : Plane) : Prop :=
equidistant_from_plane A α ∧ equidistant_from_plane B α ∧ equidistant_from_plane C α

axiom three_distinct_points (A B C : Point) : A ≠ B ∧ B ≠ C ∧ A ≠ C

theorem line_parallel_or_on_plane
  (l : Line) (α : Plane)
  (A B C : Point)
  (hA : A ∈ l) (hB : B ∈ l) (hC : C ∈ l)
  (d_eqd : equidistant A B C α) :
  l ∥ α ∨ l ⊆ α :=
sorry

end line_parallel_or_on_plane_l394_394024


namespace teacher_problems_remaining_l394_394673

theorem teacher_problems_remaining (problems_per_worksheet : Nat) 
                                   (total_worksheets : Nat) 
                                   (graded_worksheets : Nat) 
                                   (remaining_problems : Nat)
  (h1 : problems_per_worksheet = 4)
  (h2 : total_worksheets = 9)
  (h3 : graded_worksheets = 5)
  (h4 : remaining_problems = total_worksheets * problems_per_worksheet - graded_worksheets * problems_per_worksheet) :
  remaining_problems = 16 :=
sorry

end teacher_problems_remaining_l394_394673


namespace find_sum_lent_l394_394705

theorem find_sum_lent (r t : ℝ) (I : ℝ) (P : ℝ) (h1: r = 0.06) (h2 : t = 8) (h3 : I = P - 520) (h4: I = P * r * t) : P = 1000 := by
  sorry

end find_sum_lent_l394_394705


namespace sequence_integral_l394_394331

noncomputable def sequence (n : ℕ) : ℕ
| 0 => 0
| 1 => 2
| (n + 1) => (2 * (2 * (n + 1) - 1) * sequence n) / (n + 1)

theorem sequence_integral : ∀ n : ℕ, ∃ m : ℤ, sequence n = m :=
by
  intro n
  sorry

end sequence_integral_l394_394331


namespace perpendicular_vectors_x_value_l394_394098

theorem perpendicular_vectors_x_value
  (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (1, -2)) (hb : b = (-3, x))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  x = -3 / 2 := by
  sorry

end perpendicular_vectors_x_value_l394_394098


namespace Jenny_reading_days_l394_394537

theorem Jenny_reading_days :
  let words_per_hour := 100
  let book1_words := 200
  let book2_words := 400
  let book3_words := 300
  let total_words := book1_words + book2_words + book3_words
  let total_hours := total_words / words_per_hour
  let minutes_per_day := 54
  let hours_per_day := minutes_per_day / 60
  total_hours / hours_per_day = 10 :=
by
  sorry

end Jenny_reading_days_l394_394537


namespace tan_alpha_l394_394000

theorem tan_alpha (α β : ℝ)
  (h1 : Real.tan (α + β) = 3 / 5)
  (h2 : Real.tan β = 1 / 3) :
  Real.tan α = 2 / 9 := by
  sorry

end tan_alpha_l394_394000


namespace interior_angle_regular_hexagon_l394_394242

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l394_394242


namespace interest_rate_l394_394950

theorem interest_rate (P1 P2 I T1 T2 total_amount : ℝ) (r : ℝ) :
  P1 = 10000 →
  P2 = 22000 →
  T1 = 2 →
  T2 = 3 →
  total_amount = 27160 →
  (I = P1 * r * T1 / 100 + P2 * r * T2 / 100) →
  P1 + P2 = 22000 →
  (P1 + I = total_amount) →
  r = 6 :=
by
  intros hP1 hP2 hT1 hT2 htotal_amount hI hP_total hP1_I_total
  -- Actual proof would go here
  sorry

end interest_rate_l394_394950


namespace prime_iff_binomial_mod_l394_394565

noncomputable def binomial (n k : ℕ) : ℕ :=
if h : k ≤ n then
  (n.choose k)
else
  0

theorem prime_iff_binomial_mod (n : ℕ) (h1 : 1 < n) :
  (∀ k ∈ set.Ico 1 n, binomial (n - 1) k ≡ (-1 : ℕ)^k [MOD n]) ↔ nat.prime n :=
by sorry

end prime_iff_binomial_mod_l394_394565


namespace number_of_people_seated_l394_394147

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l394_394147


namespace angle_BAD_is_30_deg_l394_394643

theorem angle_BAD_is_30_deg (A B C D : Type) [EuclideanGeometry A] [EuclideanGeometry B] 
[EucideGeometory C] [EuclideanGeometry D]
(h_isosceles_ABC : IsIsoscelesTriangle A B C (eqOfDegrees 50))
(h_isosceles_ADC : IsIsoscelesTriangle A D C (eqOfDegrees 110))
(h_inside_D : Inside A B C D) 
(h_eq_AB_BC : Dist A B = Dist B C) 
(h_eq_AD_DC : Dist A D = Dist D C) : 
Angle A B D = 30 := 
by 
  sorry

end angle_BAD_is_30_deg_l394_394643


namespace orthogonal_vectors_magnitude_l394_394465

def vector_a : ℝ × ℝ := (1, -2)
def vector_b (m : ℝ) : ℝ × ℝ := (2, m)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)

theorem orthogonal_vectors_magnitude :
  ∀ m : ℝ, dot_product vector_a (vector_b m) = 0 → magnitude (vector_b m) = Real.sqrt 5 :=
by
  intro m h
  have h1 : m = 1 := sorry  -- Solve for m 
  rw [h1]
  exact sorry  -- Prove the magnitude

end orthogonal_vectors_magnitude_l394_394465


namespace mass_percentage_O_in_N2O5_l394_394398

noncomputable def atomic_mass_N : ℝ := 14.01
noncomputable def atomic_mass_O : ℝ := 16.00

def molar_mass_N2O5 : ℝ := (2 * atomic_mass_N) + (5 * atomic_mass_O)
def mass_O_in_N2O5 : ℝ := 5 * atomic_mass_O

theorem mass_percentage_O_in_N2O5 :
  (mass_O_in_N2O5 / molar_mass_N2O5) * 100 ≈ 74.06 := 
sorry

end mass_percentage_O_in_N2O5_l394_394398


namespace smallest_among_given_numbers_l394_394722

theorem smallest_among_given_numbers :
  let a := abs (-3)
  let b := -2
  let c := 0
  let d := Real.pi
  b < a ∧ b < c ∧ b < d := by
  sorry

end smallest_among_given_numbers_l394_394722


namespace limit_of_a2n_l394_394831

open Filter
open_locale TopologicalSpace

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n ∈ Set.Ici 1, a (n + 1) + a n = (1 / 3) ^ n

theorem limit_of_a2n (a : ℕ → ℝ) (h : sequence a) :
  tendsto (λ n, a (2 * n)) at_top (𝓝 (-3 / 4)) :=
sorry

end limit_of_a2n_l394_394831


namespace original_amount_of_rice_l394_394175

theorem original_amount_of_rice
  (x : ℕ) -- the total amount of rice in kilograms
  (h1 : x = 10 * 500) -- statement that needs to be proven
  (h2 : 210 = x * (21 / 50)) -- remaining rice condition after given fractions are consumed
  (consume_day_one : x - (3 / 10) * x  = (7 / 10) * x) -- after the first day's consumption
  (consume_day_two : ((7 / 10) * x) - ((2 / 5) * ((7 / 10) * x)) = 210) -- after the second day's consumption
  : x = 500 :=
by
  sorry

end original_amount_of_rice_l394_394175


namespace cos_equivalent_l394_394476

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom mag_a : ∥a∥ = 1
axiom mag_b : ∥b∥ = 1
axiom mag_c : ∥c∥ = real.sqrt 2
axiom abc_zero : a + b + c = 0

-- Proof statement
theorem cos_equivalent : 
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
by sorry

end cos_equivalent_l394_394476


namespace students_other_communities_is_369_l394_394883

-- Define the total number of students.
def total_students : ℕ := 2500

-- Define the percentages for each religion.
def percent_muslims : ℝ := 29.55 / 100
def percent_hindus : ℝ := 27.25 / 100
def percent_sikhs : ℝ := 13.7 / 100
def percent_christians : ℝ := 8.5 / 100
def percent_buddhists : ℝ := 6.25 / 100

-- Calculate the total percentage not in other communities.
def total_percent_known : ℝ := percent_muslims + percent_hindus + percent_sikhs + percent_christians + percent_buddhists

-- Calculate the percentage of students belonging to other communities.
def percent_other_communities : ℝ := 1 - total_percent_known

-- Calculate the number of students belonging to other communities.
def students_other_communities : ℕ := (percent_other_communities * total_students).round.toNat

-- The theorem we want to prove.
theorem students_other_communities_is_369 
  : students_other_communities = 369 := by
  sorry

end students_other_communities_is_369_l394_394883


namespace sum_p_q_r_eq_2_l394_394069

noncomputable def sequence (n : ℕ) : ℕ :=
if n < 3 then 1
else if 3 ≤ n ∧ n < 8 then 3
else if 8 ≤ n ∧ n < 15 then 5
else sorry -- complete sequence definition as needed

theorem sum_p_q_r_eq_2 
  (p q r : ℤ)
  (h : ∀ (n : ℕ), sequence n = p * int.floor (real.sqrt (n + q)) + r)
  (h_sequence : ∀ (n : ℕ), sequence n = 
    if n < 3 then 1
    else if 3 ≤ n ∧ n < 8 then 3
    else if 8 ≤ n ∧ n < 15 then 5
    else sorry): 
  p + q + r = 2 := sorry

end sum_p_q_r_eq_2_l394_394069


namespace bob_expected_rolls_leap_year_l394_394731

open Probability

theorem bob_expected_rolls_leap_year : 
  let P_composite := 4 / 10,
      P_prime := 4 / 10,
      P_roll_again := 1 / 10,
      E := 1 -- Derivation from P_composite, P_prime, P_roll_again and resolver equation
  in E * 366 = 366 := 
by
  -- Define the probabilities
  let P_composite := (4: ℚ) / 10
  let P_prime := (4: ℚ) / 10
  let P_roll_again := (1: ℚ) / 10

  -- Define the expected number of rolls per day without re-roll (E)
  have E_eq : E = 1 := sorry

  -- Expected number of rolls in a leap year
  let rolls_leap_year := E * 366

  -- Conclude proof
  show E * 366 = 366, by
    rw [E_eq]
    norm_num

end bob_expected_rolls_leap_year_l394_394731


namespace quadratic_root_iff_l394_394676

theorem quadratic_root_iff (a b c : ℝ) :
  (∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0) ↔ (a + b + c = 0) :=
by
  sorry

end quadratic_root_iff_l394_394676


namespace simplify_trig_expression_l394_394969

theorem simplify_trig_expression (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := by
  sorry

end simplify_trig_expression_l394_394969


namespace g_satisfies_equation_l394_394745

def g (n : Int) : Int := n + 2

theorem g_satisfies_equation : ∀ a b : Int, g (a + b) + g (ab + 1) = g a * g b + 2 := by
  intros a b
  unfold g
  calc
    (a + b + 2) + ((ab + 1) + 2) = a + b + 2 + ab + 1 + 2 : by simp
    _ = a + b + ab + 5 : by ring
    _ = (a + 2) * (b + 2) + 2 : by ring
    _ = g a * g b + 2 : by unfold g
    sorry

end g_satisfies_equation_l394_394745


namespace complement_union_l394_394836

open Set

universe u

variable (U : Type u) [TopologicalSpace U] [OrderTopology U] [LinearOrder U]
variable (A B : Set U) (R : Set U)

def A := {x : U | x < 0}
def B := {x : U | x ≥ 2}

theorem complement_union (U : Set ℝ) (A B : Set ℝ) :
  U = (univ : Set ℝ) → (A = {x : ℝ | x < 0}) → (B = {x : ℝ | x ≥ 2}) →
  compl (A ∪ B) = {x : ℝ | 0 ≤ x ∧ x < 2} :=
begin
  intro U_univ,
  intro A_def,
  intro B_def,
  sorry,
end

end complement_union_l394_394836


namespace hotel_r_charge_percentage_l394_394672

-- Let P, R, and G be the charges for a single room at Hotels P, R, and G respectively
variables (P R G : ℝ)

-- Given conditions:
-- 1. The charge for a single room at Hotel P is 55% less than the charge for a single room at Hotel R.
-- 2. The charge for a single room at Hotel P is 10% less than the charge for a single room at Hotel G.
axiom h1 : P = 0.45 * R
axiom h2 : P = 0.90 * G

-- The charge for a single room at Hotel R is what percent greater than the charge for a single room at Hotel G.
theorem hotel_r_charge_percentage : (R - G) / G * 100 = 100 :=
sorry

end hotel_r_charge_percentage_l394_394672


namespace quadratic_real_roots_m_l394_394873

theorem quadratic_real_roots_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 + 4 * x1 + m = 0 ∧ x2 * x2 + 4 * x2 + m = 0) →
  m ≤ 4 :=
by
  sorry

end quadratic_real_roots_m_l394_394873


namespace nelly_bid_l394_394952

theorem nelly_bid (joe_bid sarah_bid : ℕ) (h1 : joe_bid = 160000) (h2 : sarah_bid = 50000)
  (h3 : ∀ nelly_bid, nelly_bid = 3 * joe_bid + 2000) (h4 : ∀ nelly_bid, nelly_bid = 4 * sarah_bid + 1500) :
  ∃ nelly_bid, nelly_bid = 482000 :=
by
  -- Skipping the proof with sorry
  sorry

end nelly_bid_l394_394952


namespace price_per_dozen_in_April_l394_394627

theorem price_per_dozen_in_April 
  (M : ℝ)  -- number of dozens sold in May
  (A : ℝ)  -- the price per dozen in April
  (may_price: ℝ := 1.20)  -- $1.20 in May
  (june_price: ℝ := 3.00) -- $3.00 in June
  (may_sold: ℝ := M)  -- M dozens sold in May
  (april_sold: ℝ := 2/3 * M)  -- 2/3 as many dozen were sold in April as in May
  (june_sold: ℝ := 4/3 * M)  -- Twice as many were sold in June as in April
  (average_price: ℝ := 2)  -- The average price per dozen over the 3-month period is $2
: (may_sold * may_price + april_sold * A + june_sold * june_price) / (may_sold + april_sold + june_sold) = average_price → 
    A = 1.20 :=
begin
  sorry
end

end price_per_dozen_in_April_l394_394627


namespace triangle_area_calculations_l394_394904

theorem triangle_area_calculations
  {A B C O P R Q : Type}
  [has_angle A B C O P R Q]
  (h1 : altitude_intersection A B C O P R Q)
  (h2 : parallel RP AC)
  (h3 : length AC = 4)
  (h4 : sin_angle ABC = 24 / 25) :
  (angle ABC < 90 → area ABC = 16 / 3 ∧ area POC = 21 / 25) ∧
  (angle ABC > 90 → area ABC = 3 ∧ area POC = 112 / 75) :=
sorry

end triangle_area_calculations_l394_394904


namespace complete_the_square_b_26_l394_394298

theorem complete_the_square_b_26 :
  ∃ (a b : ℝ), (∀ x : ℝ, x^2 + 10 * x - 1 = 0 ↔ (x + a)^2 = b) ∧ b = 26 :=
sorry

end complete_the_square_b_26_l394_394298


namespace log_product_in_interval_l394_394865

theorem log_product_in_interval :
  let y := (log 6 / log 5) * (log 7 / log 6) * (log 8 / log 7) * (log 9 / log 8) * (log 10 / log 9) in
  1 < y ∧ y < 2 := sorry

end log_product_in_interval_l394_394865


namespace cos_angle_sub_vectors_l394_394482

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (H1 : ∥a∥ = 1) (H2 : ∥b∥ = 1) (H3 : ∥c∥ = real.sqrt 2)
variables (H4 : a + b + c = 0)

theorem cos_angle_sub_vectors :
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry

end cos_angle_sub_vectors_l394_394482


namespace problem_solution_l394_394628

noncomputable def quadratic_function_vertex_and_max (f g : ℝ → ℝ) : Prop :=
  (f (1 : ℝ) = (16 : ℝ)) ∧
  (f (-3 : ℝ) = (0 : ℝ)) ∧
  (f (5 : ℝ) = (0 : ℝ)) ∧
  (∀ x : ℝ, f x = -(x^2) + 2 * x + 15) ∧
  (∀ x ∈ set.Icc (0 : ℝ) (2 : ℝ), g x = 7)

theorem problem_solution : 
  ∃ (f g : ℝ → ℝ), quadratic_function_vertex_and_max f g :=
begin
  existsi (λ (x : ℝ), -(x^2) + 2 * x + 15),
  existsi (λ (x : ℝ), -(x^2) + 2),
  sorry
end

end problem_solution_l394_394628


namespace find_a_l394_394788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + 3 * x - 9

theorem find_a (a : ℝ) (h : ∃ (x : ℝ), x = -3 ∧ (f a x).derivative = 0) : a = 4 :=
sorry

end find_a_l394_394788


namespace travel_time_between_resorts_l394_394632

theorem travel_time_between_resorts
  (num_cars : ℕ)
  (car_interval : ℕ)
  (opposing_encounter_time : ℕ)
  (travel_time : ℕ) :
  num_cars = 80 →
  car_interval = 15 →
  (opposing_encounter_time * 2 * car_interval / travel_time) = num_cars →
  travel_time = 20 :=
by
  sorry

end travel_time_between_resorts_l394_394632


namespace bc_plus_dc_eq_2de_l394_394532

/-- Isosceles triangle with AB = AC inscribed in circle ω -/
structure IsoscelesTriangle (P Q R : Type) :=
(AB : P = Q)
(AC : P = R)

/-- Circle ω -/
structure Circle (P : Type) := 
(ω : P)

/-- Prove that BC + DC = 2DE given conditions -/
theorem bc_plus_dc_eq_2de {P Q R S T U : Type}
  (tri : IsoscelesTriangle P Q R)
  (circle : Circle P)
  (D_on_arc : S)
  (E : T)
  (foot_perpendicular : E)
  : BC + DC = 2DE :=
sorry

end bc_plus_dc_eq_2de_l394_394532


namespace interior_angle_of_regular_hexagon_l394_394227

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l394_394227


namespace triangle_area_l394_394714

-- Given conditions of the problem
def square_side : ℝ := 8
def PA_length : ℝ := 5
def PB_length : ℝ := 5
def PC_length : ℝ := 5
def height_APB : ℝ := 3

theorem triangle_area :
    let base := 8 in
    let height := 3 in
    (1 / 2) * base * height = 12 := 
by
    let base := 8
    let height := 3
    exact ((1:ℝ) / 2) * base * height = 12

end triangle_area_l394_394714


namespace part_I_part_II_l394_394019

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x + 1|
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := 2 * |x| + a

-- Theorem for part (I)
theorem part_I (x : ℝ) : (f x) ≥ (g x 0) → x ∈ (set.Icc (-1/3 : ℚ) 1) := sorry

-- Theorem for part (II)
theorem part_II (a : ℝ) : (∃ x : ℝ, (f x) ≥ (g x a)) → a ∈ (set.Iic (1 : ℚ)) := sorry

end part_I_part_II_l394_394019


namespace smallest_x_satisfying_equation_l394_394749

theorem smallest_x_satisfying_equation (k : ℝ) (h₁ : x + 2 ≤ 3) (h₂ : k = 2) :
  ∃ x, x |x| = 3x + k ∧ x + 2 ≤ 3 ∧ x = -2 := by
  sorry

end smallest_x_satisfying_equation_l394_394749


namespace prod_of_real_roots_equation_l394_394192

theorem prod_of_real_roots_equation :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0 ∨ x = -(4 / 7) → (0 * (-(4 / 7)) = 0) :=
by sorry

end prod_of_real_roots_equation_l394_394192


namespace max_score_sum_l394_394054

theorem max_score_sum (m n : ℕ) (h_m : 2 ≤ m) (h_n : 2 ≤ n) (scores : Fin n → Fin m → ℕ) :
  let p := (λ i : Fin n, ∑ j, (let x := n - (∑ i, if scores i j = 0 then 1 else 0) in if scores i j > 0 then x else 0)) in
  (∀ j : Fin m, 0 ≤ (∑ i, if scores i j = 0 then 1 else 0) ∧ (∑ i, if scores i j = 0 then 1 else 0) ≤ n) →
  (∀ i : Fin n, 0 ≤ p i) →
  let sorted_scores := Multiset.sort (≤) (Finset.univ.image p).val in 
  (p 0 + p (n-1)) ≤ m * (n - 1) := 
sorry

end max_score_sum_l394_394054


namespace evaluate_expression_l394_394758

theorem evaluate_expression (b : ℝ) (h : b = 2) : 
  (4 * b^(-2) + b^(-1) / 3) / b^(-1) = 25 / 6 := 
  by
  sorry

end evaluate_expression_l394_394758


namespace fx_le_1_l394_394824

-- Statement
theorem fx_le_1 (x : ℝ) (h : x > 0) : (1 + Real.log x) / x ≤ 1 := 
sorry

end fx_le_1_l394_394824


namespace imaginary_part_of_1_minus_2i_l394_394927

def i := Complex.I

theorem imaginary_part_of_1_minus_2i : Complex.im (1 - 2 * i) = -2 :=
by
  sorry

end imaginary_part_of_1_minus_2i_l394_394927


namespace sequence_well_defined_and_decreasing_l394_394085

noncomputable def u : ℕ → ℝ
| 0       := 3
| (n + 1) := real.sqrt (2 + u n)

theorem sequence_well_defined_and_decreasing :
  (∀ n, 0 ≤ u n) ∧ (∀ n, u n > u (n + 1)) :=
by
  sorry

end sequence_well_defined_and_decreasing_l394_394085


namespace point_on_graph_of_inverse_proportion_l394_394345

theorem point_on_graph_of_inverse_proportion :
  ∃ x y : ℝ, (x = 2 ∧ y = 4) ∧ y = 8 / x :=
by
  sorry

end point_on_graph_of_inverse_proportion_l394_394345


namespace degree_measure_of_regular_hexagon_interior_angle_l394_394279

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l394_394279


namespace total_people_seated_l394_394154

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l394_394154


namespace greg_attendance_probability_l394_394840

theorem greg_attendance_probability :
  let P_Rain := 0.5
  let P_Attend_if_Rain := 0.3
  let P_Sunny := 1 - P_Rain
  let P_Attend_if_Sunny := 0.9
  let P_Attend := P_Rain * P_Attend_if_Rain + P_Sunny * P_Attend_if_Sunny
  in P_Attend = 0.6 :=
by
  sorry

end greg_attendance_probability_l394_394840


namespace leadership_arrangements_l394_394630

theorem leadership_arrangements :
  let teachers := 5
  let groups := 3
  ∃ (arrangements : ℕ), 
    arrangements = 54 
    ∧ (∃ (A B : ℕ), A = teachers ∧ B = groups 
            ∧ ∃ f : (fin teachers) → option (fin groups × fin groups), 
                (∀ t : fin teachers, ∃ g h : option (fin groups), 
                    f t = some (g.val, h.val) ∧ (g.val ≠ none) ∧ (h.val ≠ none))
                ∧ ∀ i j : fin teachers, i ≠ j → f i ≠ f j)
:= sorry

end leadership_arrangements_l394_394630


namespace find_value_of_m_l394_394190

-- Define the quadratic function and the values in the given table
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
variables (a b c m : ℝ)
variables (h1 : quadratic_function a b c (-1) = m)
variables (h2 : quadratic_function a b c 0 = 2)
variables (h3 : quadratic_function a b c 1 = 1)
variables (h4 : quadratic_function a b c 2 = 2)
variables (h5 : quadratic_function a b c 3 = 5)
variables (h6 : quadratic_function a b c 4 = 10)

-- Theorem stating that the value of m is 5
theorem find_value_of_m : m = 5 :=
by
  sorry

end find_value_of_m_l394_394190


namespace paintings_correct_l394_394065

def octagon_paintings : Prop :=
  let num_disks := 8
  let disks := Fin num_disks
  let colors := {blue := 4, red := 2, green := 2}
  let symmetries := -- Represent rotations and reflections of the octagon
    [(0 : Int), 45, 90, 135, 180, 225, 270, 315, -- rotations
     "ref_v1", "ref_v2", "ref_v3", "ref_v4", -- reflections through vertices
     "ref_m1", "ref_m2", "ref_m3", "ref_m4"] -- reflections through midpoints
  ∀ (paintings : {arrangement : disks → Fin 3 // 
                    arrangement.parity colors -- Each color used the required number of times
                    }), 
    let count_fixed := Burnside.fixed_count symmetries paintings
    count_fixed = 34

theorem paintings_correct : octagon_paintings :=
  sorry

end paintings_correct_l394_394065


namespace find_x2_times_sum_roots_l394_394094

noncomputable def sqrt2015 := Real.sqrt 2015

theorem find_x2_times_sum_roots
  (x1 x2 x3 : ℝ)
  (h_eq : ∀ x : ℝ, sqrt2015 * x^3 - 4030 * x^2 + 2 = 0 → x = x1 ∨ x = x2 ∨ x = x3)
  (h_ineq : x1 < x2 ∧ x2 < x3) :
  x2 * (x1 + x3) = 2 := by
  sorry

end find_x2_times_sum_roots_l394_394094


namespace find_f_and_g_l394_394934

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

lemma even_function (x : ℝ) : f(-x) = f(x) := sorry
lemma odd_function (x : ℝ) : g(-x) = -g(x) := sorry
lemma func_equation (x : ℝ) (h : x ∈ Ioo (-1 : ℝ) 1) : f(x) + g(x) = -2 * log 10 (1 + x) := sorry

theorem find_f_and_g (x : ℝ) (h : x ∈ Ioo (-1 : ℝ) 1) : 
  10 ^ (f x) = 1 / (1 - x ^ 2) ∧ 10 ^ (g x) = (1 - x) / (1 + x) := 
begin
  sorry
end

end find_f_and_g_l394_394934


namespace interior_angle_of_regular_hexagon_l394_394226

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l394_394226


namespace cos_angle_sub_vectors_l394_394485

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (H1 : ∥a∥ = 1) (H2 : ∥b∥ = 1) (H3 : ∥c∥ = real.sqrt 2)
variables (H4 : a + b + c = 0)

theorem cos_angle_sub_vectors :
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry

end cos_angle_sub_vectors_l394_394485


namespace seated_people_count_l394_394135

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l394_394135


namespace randy_initial_money_l394_394594

theorem randy_initial_money (X : ℕ) (h : X + 200 - 1200 = 2000) : X = 3000 :=
by {
  sorry
}

end randy_initial_money_l394_394594


namespace sin_60_equiv_l394_394381

theorem sin_60_equiv : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := 
by
  sorry

end sin_60_equiv_l394_394381


namespace count_primes_with_digit_three_l394_394846

def is_digit_three (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := Prime n

def primes_with_digit_three_count (lim : ℕ) (count : ℕ) : Prop :=
  ∀ n < lim, is_digit_three n → is_prime n → count = 9

theorem count_primes_with_digit_three (lim : ℕ) (count : ℕ) :
  primes_with_digit_three_count 150 9 := 
by
  sorry

end count_primes_with_digit_three_l394_394846


namespace cos_equivalent_l394_394479

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom mag_a : ∥a∥ = 1
axiom mag_b : ∥b∥ = 1
axiom mag_c : ∥c∥ = real.sqrt 2
axiom abc_zero : a + b + c = 0

-- Proof statement
theorem cos_equivalent : 
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
by sorry

end cos_equivalent_l394_394479


namespace width_of_rectangle_l394_394327

theorem width_of_rectangle (w l : ℝ) (h1 : l = 2 * w) (h2 : l * w = 1) : w = Real.sqrt 2 / 2 :=
sorry

end width_of_rectangle_l394_394327


namespace complement_event_prob_l394_394007

variable {Ω : Type*} [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}

theorem complement_event_prob (A : Set Ω) (hA : P A = 0.5) : P (Aᶜ) = 0.5 := by
  rw [ProbabilityMeasure.compl_eq_one_sub]
  rw [hA]
  norm_num
  -- This theorem states that, given the probability of event A is 0.5,
  -- the probability of its complementary event must also be 0.5.

end complement_event_prob_l394_394007


namespace compute_expression_l394_394737

theorem compute_expression :
  (Real.exp (Real.log 3)) + Real.log' 25 (Real.sqrt 5) + (0.125 ^ (-2 / 3 : ℝ)) = 11 :=
by
  sorry

end compute_expression_l394_394737


namespace total_people_seated_l394_394157

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l394_394157


namespace interior_angle_of_regular_hexagon_is_120_degrees_l394_394248

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l394_394248


namespace sin_alpha_minus_two_thirds_l394_394785

theorem sin_alpha_minus_two_thirds : 
  ∀ α : ℝ, 
  cos (π / 6 - α) = 2 / 3 → sin (α - 2 * π / 3) = - 2 / 3 :=
by
  intros α h
  sorry

end sin_alpha_minus_two_thirds_l394_394785


namespace find_AC_len_l394_394100

variables {A B C P Q : Type} [MetricSpace P]

-- Definitions for points and conditions
variable {G : P} -- The centroid G
variable (AP BQ : P)
variable (AP_len : ℝ) (BQ_len : ℝ)
variable (hAP_len : AP_len = 15) (hBQ_len : BQ_len = 20)
variable (h_perp : AP ⊥ BQ)

-- Define the lengths AG and BG according to the centroid property
def AG : ℝ := (2 / 3) * AP_len
def BG : ℝ := (1 / 3) * BQ_len

-- The proof to be written
theorem find_AC_len (hAP_len : AP_len = 15) (hBQ_len : BQ_len = 20) (h_perp : AP ⊥ BQ) :
  let AG := (2 / 3) * AP_len,
      BG := (1 / 3) * BQ_len in
  AC = (20 * Real.sqrt 13) / 3 := 
sorry

end find_AC_len_l394_394100


namespace quadratic_root_four_times_another_l394_394750

theorem quadratic_root_four_times_another (a : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + a * x + 2 * a = 0 ∧ x2 = 4 * x1) → a = 25 / 2 :=
by
  sorry

end quadratic_root_four_times_another_l394_394750


namespace exists_segment_with_color_conditions_l394_394106

variables {n : ℕ} (h₁ : n ≥ 4) 
variables (A : Fin n → ℝ) (color : Fin n → Fin 4)

theorem exists_segment_with_color_conditions :
  ∃ (i j : Fin n), i < j ∧
    (∃ (c1 c2 : Fin 4), c1 ≠ c2 ∧ 
    ((∀ k, k ∈ Ico i j → (color k = c1 ∨ color k = c2)) ∧ 
    ∃ k1 k2, k1 ∈ Ico i j ∧ k2 ∈ Ico i j ∧ color k1 ≠ color k2)) :=
sorry

end exists_segment_with_color_conditions_l394_394106


namespace rhombus_inside_circle_square_outside_all_l394_394211

theorem rhombus_inside_circle_square_outside_all (x y : ℝ) :
  2 * |x| + 2 * |y| ≥ x^2 + y^2 ∧ x^2 + y^2 ≥ max (|x|^2) (|y|^2) →
  -- Rhombus inside circle
  (∀ x y, 0 < (2 * |x| + 2 * |y|) → 0 < x^2 + y^2) ∧
  -- Square outside both circle and rhombus
  (∀ x y, x^2 + y^2 ≤ max (|x|^2) (|y|^2) → (|x| < 2) ∧ (|y| < 2)) :=
by
  sorry

end rhombus_inside_circle_square_outside_all_l394_394211


namespace sum_of_squares_bounds_l394_394202

theorem sum_of_squares_bounds (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 10) : 
  (x^2 + y^2 ≤ 100) ∧ (x^2 + y^2 ≥ 50) :=
by 
  sorry

end sum_of_squares_bounds_l394_394202


namespace divisor_is_seventeen_l394_394635

theorem divisor_is_seventeen (D x : ℕ) (h1 : D = 7 * x) (h2 : D + x = 136) : x = 17 :=
by
  sorry

end divisor_is_seventeen_l394_394635


namespace conj_poly_eq_l394_394654

def conj_poly (a b : ℂ) : ℂ[X] :=
    X ^ 2 - (a - b) * X + (a + b)

theorem conj_poly_eq :
  ∃ (p : ℂ[X]), p = conj_poly (6 : ℂ) (2 * complex.I : ℂ) ∧ 
    (∀ x, x^2 - (6 + 2 * complex.I) * x + (11 + 10 * complex.I) = 2 * x * complex.I - 10 * complex.I → 
      x ∈ (roots p)) :=
begin
  sorry
end

end conj_poly_eq_l394_394654


namespace max_visible_unit_cubes_from_corner_l394_394348

theorem max_visible_unit_cubes_from_corner :
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  faces_visible - edges_shared + corner_cube = 331 := by
  let n := 11
  let faces_visible := 3 * n^2
  let edges_shared := 3 * (n - 1)
  let corner_cube := 1
  have result : faces_visible - edges_shared + corner_cube = 331 := by
    sorry
  exact result

end max_visible_unit_cubes_from_corner_l394_394348


namespace equilateral_triangle_average_distance_l394_394591

noncomputable def F (Y : Finset (ℝ × ℝ)) (points : Finset (ℝ × ℝ)) : ℝ :=
  (Finset.sum points (λ X, Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2))) / points.card

theorem equilateral_triangle_average_distance (points : Finset (ℝ × ℝ)) 
  (Y : Finset (ℝ × ℝ)) (h : ∀ p ∈ points, ∃! q ∈ Y, q = p) :
  ∃ t : ℝ, ∀ (n : ℕ) (X : Fin (n+1) → (ℝ × ℝ)),
  ∃ Y ∈ Y, F Y (Finset.univ.image X) = t :=
sorry

end equilateral_triangle_average_distance_l394_394591


namespace select_five_markers_l394_394209

theorem select_five_markers :
  ∃ k : ℕ, k = nat.choose 15 5 ∧ k = 3003 :=
begin
  use nat.choose 15 5,
  split,
  { refl },
  { sorry }
end

end select_five_markers_l394_394209


namespace simplify_trig_expression_l394_394983

variable {x : ℝ}

theorem simplify_trig_expression (h : 1 + cos x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l394_394983


namespace election_winner_votes_l394_394638

theorem election_winner_votes (V W L : ℕ) (h1 : W = 0.62 * V) (h2 : W = 992) (h3 : V = W / 0.62) :
  W - ((1 - 0.62) * V) = 384 :=
by
  sorry

end election_winner_votes_l394_394638


namespace geometric_sequence_fifth_term_l394_394895

theorem geometric_sequence_fifth_term (α : ℕ → ℝ) (h : α 4 * α 5 * α 6 = 27) : α 5 = 3 :=
sorry

end geometric_sequence_fifth_term_l394_394895


namespace interior_angle_regular_hexagon_l394_394246

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l394_394246


namespace ben_more_dogs_than_teddy_l394_394607

theorem ben_more_dogs_than_teddy
  (teddy_dogs : ℕ) (teddy_cats : ℕ)
  (ben_dogs : ℕ) (dave_dogs : ℕ) (dave_cats : ℕ) (total_pets : ℕ) :
  teddy_dogs = 7 →
  teddy_cats = 8 →
  dave_cats = teddy_cats + 13 →
  dave_dogs = teddy_dogs - 5 →
  total_pets = teddy_dogs + teddy_cats + ben_dogs + dave_dogs + dave_cats →
  total_pets = 54 →
  ben_dogs = 16 →
  ben_dogs - teddy_dogs = 9 :=
by
  intros h_teddy_dogs h_teddy_cats h_dave_cats h_dave_dogs h_total_pets h_total_pets_value h_ben_dogs
  rw [h_teddy_dogs, h_ben_dogs]
  norm_num
  exact eq.refl 9

end ben_more_dogs_than_teddy_l394_394607


namespace product_of_solutions_eq_zero_l394_394195

theorem product_of_solutions_eq_zero :
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) →
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → (x = 0 ∨ x = -4/7)) →
  (0 * (-4/7) = 0) :=
by
  sorry

end product_of_solutions_eq_zero_l394_394195


namespace a_n_formula_l394_394920

-- Define the initial set A_0
def A_0 : Finset ℕ := {1, 2}

-- Define the recursive formation of A_n
def A (n : ℕ) : Finset ℕ :=
  if h : n = 0 then A_0
  else
    let m := 2^n - 1 in -- the largest element in A_{n-1}
    (A (n - 1)) ∪ (Finset.range (m + 1) \ {0})

-- Define a_n as the size of the set A_n
def a_n (n : ℕ) : ℕ := (A n).card

-- Theorem to prove the relationship
theorem a_n_formula (n : ℕ) : a_n n = 2^n + 1 :=
by
  induction n with n ih
  case zero =>
    -- Base case when n = 0
    show a_n 0 = 2^0 + 1
    sorry
  case succ n =>
    -- Inductive step
    have : a_n (n + 1) = 2 * a_n n - 1 := sorry
    show a_n (n + 1) = 2^(n + 1) + 1
    from calc
      a_n (n + 1) = 2 * a_n n - 1     : by assumption
      ... = 2 * (2^n + 1) - 1         : by rw [ih]
      ... = 2^(n + 1) + 2 - 1         : by ring
      ... = 2^(n + 1) + 1             : by ring

end a_n_formula_l394_394920


namespace cosine_identity_l394_394469

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) 

theorem cosine_identity 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (hc : ∥c∥ = real.sqrt 2)
  (habc : a + b + c = 0) :
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
sorry

end cosine_identity_l394_394469


namespace cos_angle_sub_vectors_l394_394488

-- Definitions of the conditions
variables (a b c : ℝ^3) (ha : ‖ a ‖ = 1) (hb : ‖ b ‖ = 1) (hc : ‖ c ‖ = √2)
variable (h0 : a + b + c = 0)

-- The theorem statement
theorem cos_angle_sub_vectors : 
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry -- Proof omitted

end cos_angle_sub_vectors_l394_394488


namespace convex_iff_m_eq_f_l394_394420
open Finset

noncomputable def m (S : Set (Fin n)) (a : Fin n → ℕ) : ℕ :=
a.sum id

def is_convex_polygon (S : Set (Fin n)) : Prop := sorry

def f (n : ℕ) : ℕ :=
2 * nat.choose n 4

theorem convex_iff_m_eq_f (S : Set (Fin n))
  (h1 : 3 < n)
  (h2 : ∀ P1 P2 P3 ∈ S, ¬ collinear P1 P2 P3)
  (h3 : ∀ P1 P2 P3 P4 ∈ S, ¬ concyclic P1 P2 P3 P4)
  (a : Fin n → ℕ)
  (h4 : m S a = a.sum id) :
  is_convex_polygon S ↔ m S a = f n := sorry

end convex_iff_m_eq_f_l394_394420


namespace aitana_fraction_more_than_jayda_l394_394719

theorem aitana_fraction_more_than_jayda (jayda_spending total_spending aitana_spending : ℚ)
  (h_jayda : jayda_spending = 400)
  (h_total : total_spending = 960)
  (h_aitana : aitana_spending = total_spending - jayda_spending) :
  (aitana_spending - jayda_spending) / jayda_spending = 2 / 5 := by
  sorry

end aitana_fraction_more_than_jayda_l394_394719


namespace percentage_puppies_l394_394047

theorem percentage_puppies (total_students : ℕ) (H1 : total_students = 40)
  (percent_puppies_with_parrots : ℕ) (H2 : percent_puppies_with_parrots = 25)
  (both_puppies_parrots : ℕ) (H3 : both_puppies_parrots = 8) :
  let puppies_percent := 80 in puppies_percent = 80 :=
by {
  sorry
}

end percentage_puppies_l394_394047


namespace symmetric_point_l394_394902

-- Definitions based on conditions
def point_3d := ℝ × ℝ × ℝ
def symm_point_xOy_plane (P : point_3d) : point_3d :=
  (P.1, P.2, -P.3)

-- Statement of the theorem
theorem symmetric_point (P : point_3d) (h : P = (1, 2, 3)) : symm_point_xOy_plane P = (1, 2, -3) :=
by
  sorry

end symmetric_point_l394_394902


namespace intersection_eq_l394_394023

def setM : Set ℝ := {x | Real.log10 (x - 1) < 0}
def setN : Set ℝ := {x | 2 * x ^ 2 - 3 * x ≤ 0}
def setM_inter_N : Set ℝ := {x | 1 < x ∧ x ≤ 3 / 2}

theorem intersection_eq :
  (setM ∩ setN) = setM_inter_N :=
by {
  sorry
}

end intersection_eq_l394_394023


namespace combined_population_three_years_ago_l394_394639

theorem combined_population_three_years_ago :
  ∀ (p : ℕ), 
    let Pirajussaraí_three_years_ago := p,
    let Tucupira_current := 1.5 * p,
    let Pirajussaraí_current := p,
    let combined_current_population := Tucupira_current + Pirajussaraí_current in
    combined_current_population = 9000 →
    2 * p = 7200 :=
by 
    intro p,
    let Pirajussaraí_three_years_ago := p,
    let Tucupira_current := 1.5 * p,
    let Pirajussaraí_current := p,
    let combined_current_population := Tucupira_current + Pirajussaraí_current,
    assume h : combined_current_population = 9000,
    sorry

end combined_population_three_years_ago_l394_394639


namespace locus_line_segment_l394_394071

-- Definitions of the parameters of the problem
structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- A regular tetrahedron with one vertex at the origin and others at unit distance
noncomputable def regular_tetrahedron (P A B C : Point) :=
  (dist P A = dist P B ∧ dist P B = dist P C ∧ dist A B = dist A C ∧ dist B C = dist P A ∧
   dist P A = 1) ∧ (P.z = 0 ∧ A.z = 0 ∧ B.z = 0 ∧ C.z = 0)

-- Point M within or on boundary of triangle ABC
def point_in_triangle (M A B C : Point) :=
  (M.x = a * A.x + b * B.x + c * C.x ∧ M.y = a * A.y + b * B.y + c * C.y ∧ M.z = a * A.z + b * B.z + c * C.z) ∧
  (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 1)

-- Distances from M to faces PAB, PBC, PCA
def distances_in_arithmetic_sequence (M P A B C : Point) : Prop :=
  let d1 := distance_from_point_to_plane M P A B in
  let d2 := distance_from_point_to_plane M P B C in
  let d3 := distance_from_point_to_plane M P C A in
  (2 * d2 = d1 + d3)

-- Locus of point M being a line segment
theorem locus_line_segment (P A B C M : Point)
  (h_tetra : regular_tetrahedron P A B C)
  (h_point : point_in_triangle M A B C)
  (h_distances : distances_in_arithmetic_sequence M P A B C) :
  ∃ l, is_line_segment l ∧ M ∈ l := 
sorry

end locus_line_segment_l394_394071


namespace num_planes_determined_by_four_points_l394_394789

theorem num_planes_determined_by_four_points (A B C D : Type) 
  [Point A] [Point B] [Point C] [Point D] (h_distinct: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D): 
  (num_planes_determined A B C D = 1 ∨ num_planes_determined A B C D = 4) :=
sorry

end num_planes_determined_by_four_points_l394_394789


namespace odd_function_f_of_f_neg_two_l394_394428

def f (x : ℝ) : ℝ := 
  if x > 0 then 2*x - 1 
  else if x = 0 then 0 
  else 2*x + 1

theorem odd_function (x : ℝ) : f(-x) = -f(x) := sorry

theorem f_of_f_neg_two : f(f(-2)) = -5 := 
begin
  -- We skip the proof and assume the result for illustration.
  sorry
end

end odd_function_f_of_f_neg_two_l394_394428


namespace sum_of_angles_l394_394407

theorem sum_of_angles (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 360) (hx_deg : floor x = x) :
  (∑ y in {y | ∃ (hy : 0 ≤ y ∧ y ≤ 360), floor y = y ∧ (Real.sin y * Real.sin y * Real.sin y + Real.cos y * Real.cos y * Real.cos y) = (Real.sin y + Real.cos y)}, y) = 900 := 
sorry

end sum_of_angles_l394_394407


namespace luke_made_18_dollars_weed_eating_l394_394574

   variable (money_mowing money_weed_eating spend_per_week weeks : ℕ)

   -- Conditions
   def condition1 := money_mowing = 9
   def condition2 := spend_per_week = 3
   def condition3 := weeks = 9

   -- Question and Answer in terms of proof
   theorem luke_made_18_dollars_weed_eating (h1 : condition1) (h2 : condition2) (h3 : condition3) :
     money_weed_eating = 18 :=
   by
     sorry
   
end luke_made_18_dollars_weed_eating_l394_394574


namespace football_tournament_pigeonhole_l394_394880

open Finite

theorem football_tournament_pigeonhole
  (teams : Finset ℕ)
  (matches : ℕ)
  (points : ℕ → ℕ → ℕ)
  (wins : ℕ → ℕ)
  (losses : ℕ → ℕ)
  (h_teams : teams.card = 16)
  (h_matches : ∀ t ∈ teams, matches = 15)
  (h_points : ∀ t ∈ teams, points t = 3 * (wins t) + 1 * (matches - (wins t + losses t)))
  (h_wins : ∀ t ∈ teams, 5 ≤ wins t)
  (h_losses : ∀ t ∈ teams, 5 ≤ losses t) :
  ∃ t1 t2 ∈ teams, t1 ≠ t2 ∧ points t1 = points t2 :=
by
  sorry

end football_tournament_pigeonhole_l394_394880


namespace Bernoulli_inequality_l394_394158

theorem Bernoulli_inequality (n : ℕ) (a : ℝ) (h : a > -1) : (1 + a)^n ≥ n * a + 1 := 
sorry

end Bernoulli_inequality_l394_394158


namespace people_at_table_l394_394126

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l394_394126


namespace points_on_circle_at_distance_sqrt2_l394_394187

theorem points_on_circle_at_distance_sqrt2 {x y : ℝ} 
  (h1 : x^2 + 2 * x + y^2 + 4 * y - 3 = 0)
  (h2 : ∀ (x y : ℝ), real.dist ((-1, -2), (x, y)) = √2)
  : (∃ p, p ∈ ({(x, y) : set (ℝ × ℝ) | h1 x y}) ∧ ∀ q ∈ p, real.dist q ({x, y | x + y + 1 = 0}) = √2) → 3 :=
by sorry

end points_on_circle_at_distance_sqrt2_l394_394187


namespace triangle_angle_distance_l394_394046

noncomputable def triangle_properties (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) : Prop :=
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R

theorem triangle_angle_distance (ABC P Q R: Type) (angle : ABC → ABC → ABC → ℝ) (dist : ABC → ABC → ℝ) :
  triangle_properties ABC P Q R angle dist →
  ∀ (A B C P Q R : ABC),
    angle B P C = 45 ∧
    angle Q A C = 45 ∧
    angle B C P = 30 ∧
    angle A C Q = 30 ∧
    angle A B R = 15 ∧
    angle B A R = 15 →
    angle P R Q = 90 ∧
    dist Q R = dist P R :=
by intros; sorry

end triangle_angle_distance_l394_394046


namespace find_parabola_focus_l394_394392

theorem find_parabola_focus : 
  ∀ (x y : ℝ), (y = 2 * x ^ 2 + 4 * x - 1) → (∃ p q : ℝ, p = -1 ∧ q = -(23:ℝ) / 8 ∧ (y = 2 * x ^ 2 + 4 * x - 1) → (x, y) = (p, q)) :=
by
  sorry

end find_parabola_focus_l394_394392


namespace interior_angle_of_regular_hexagon_l394_394273

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l394_394273


namespace table_seating_problem_l394_394142

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l394_394142


namespace triangle_PR_length_l394_394903

/-- In triangle \( PQR \), where the side \( PQ = 7 \), \( QR = 10 \), and the length of the median \( PM \) from \( P \) to \( QR \) is \( 5\), prove that the length of \( PR \) is \( \sqrt{149} \). -/
theorem triangle_PR_length :
  ∀ (P Q R M : Type)
    [inner_product_space ℝ P]
    (p q r m : P)
    (hpq : dist p q = 7)
    (hqr : dist q r = 10)
    (hpm : dist p m = 5)
    (hM : m = midpoint ℝ q r),
  dist p r = Real.sqrt 149 :=
by
  sorry

end triangle_PR_length_l394_394903


namespace simplify_expression_l394_394931

noncomputable def p (a b c x k : ℝ) := 
  k * (((x + a) ^ 2 / ((a - b) * (a - c))) +
       ((x + b) ^ 2 / ((b - a) * (b - c))) +
       ((x + c) ^ 2 / ((c - a) * (c - b))))

theorem simplify_expression (a b c k : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : b ≠ c) (h₃ : k ≠ 0) :
  p a b c x k = k :=
sorry

end simplify_expression_l394_394931


namespace river_flow_rate_l394_394329

theorem river_flow_rate
  (h : ℝ) (h_eq : h = 3)
  (w : ℝ) (w_eq : w = 36)
  (V : ℝ) (V_eq : V = 3600)
  (conversion_factor : ℝ) (conversion_factor_eq : conversion_factor = 3.6) :
  (60 / (w * h)) * conversion_factor = 2 := by
  sorry

end river_flow_rate_l394_394329


namespace cos_angle_sub_vectors_l394_394490

-- Definitions of the conditions
variables (a b c : ℝ^3) (ha : ‖ a ‖ = 1) (hb : ‖ b ‖ = 1) (hc : ‖ c ‖ = √2)
variable (h0 : a + b + c = 0)

-- The theorem statement
theorem cos_angle_sub_vectors : 
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry -- Proof omitted

end cos_angle_sub_vectors_l394_394490


namespace min_value_f_l394_394791

theorem min_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : 
  ∃ z : ℝ, z = x + y + x * y ∧ z = -9/8 :=
by 
  sorry

end min_value_f_l394_394791


namespace construct_sphere_diameter_l394_394647

open EuclideanGeometry

noncomputable def sphere_diameter (S : Sphere ℝ) (P Q : Point ℝ) : ℝ :=
  2 * S.radius

theorem construct_sphere_diameter
  (S : Sphere ℝ)
  (P : Point ℝ)
  (hP : P ∈ S.surface)
  (Q : Point ℝ)
  (hQ : Q ∈ S.surface)
  (dPQ : dist P Q = 2 * S.radius) :
  (diameter S = dPQ) :=
begin
  sorry
end

end construct_sphere_diameter_l394_394647


namespace value_of_y_l394_394802

/-- Given an acute triangle where two of its altitudes divide the sides into segments of lengths
  6, a, 4 and y units respectively, and the area of the triangle formed by one of these altitudes
  and its sectioned sides is 12 square units, the value of y is 10. -/
theorem value_of_y (a y : ℝ) (h1 : y > 0) (h2 : ∀ A B C : Type,[metric_space A])
  (h3 : ∃ (triangle : triangle A), acute_triangle (triangle) 
  ∧ divides_sides triangle 6 a 4 y 
  ∧ divides_area triangle 12) : 
  y = 10 :=
sorry

end value_of_y_l394_394802


namespace total_people_seated_l394_394156

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l394_394156


namespace repeating_decimal_denominator_l394_394176

theorem repeating_decimal_denominator : 
  let S := 142857 / 999999 in
  ∀ (a b : ℚ), S = a / b → a / b = 1 / 7 → b = 7 :=
by
  intros S a b h₁ h₂
  sorry

end repeating_decimal_denominator_l394_394176


namespace find_a_l394_394412

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Solution steps go here
  sorry

end find_a_l394_394412


namespace imaginary_part_of_conjugate_z_mul_i_l394_394807

variable (i : ℂ) 

theorem imaginary_part_of_conjugate_z_mul_i (z : ℂ) (hz : z = 2 + i) : 
  Complex.imag (Complex.conj z * i) = 2 :=
by
  sorry

end imaginary_part_of_conjugate_z_mul_i_l394_394807


namespace function_neither_odd_nor_even_and_monotonic_l394_394344

def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def monotonic_increasing (f : ℝ → ℝ) (domain : Set ℝ) : Prop := 
  ∀ x y ∈ domain, x ≤ y → f x ≤ f y

theorem function_neither_odd_nor_even_and_monotonic {f : ℝ → ℝ} 
  (hA : f = λ x, x^2 + 1)
  (hB : f = λ x, (x + 1) / x)
  (hC : f = λ x, |x + 1|)
  (hD : f = λ x, 2^x - 2^(-x)) :
  ¬ is_odd (λ x, |x + 1|) ∧ ¬ is_even (λ x, |x + 1|) ∧ monotonic_increasing (λ x, |x + 1|) {x | 0 < x} :=
sorry

end function_neither_odd_nor_even_and_monotonic_l394_394344


namespace simplify_expression_l394_394990

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
    (Real.sqrt (4 + ( (x^3 - 2) / (3 * x) ) ^ 2)) = 
    (Real.sqrt (x^6 - 4 * x^3 + 36 * x^2 + 4) / (3 * x)) :=
by sorry

end simplify_expression_l394_394990


namespace sum_reciprocal_transformation_l394_394533

theorem sum_reciprocal_transformation 
  (a b c d S : ℝ) 
  (h1 : a + b + c + d = S)
  (h2 : 1 / a + 1 / b + 1 / c + 1 / d = S)
  (h3 : a ≠ 0 ∧ a ≠ 1)
  (h4 : b ≠ 0 ∧ b ≠ 1)
  (h5 : c ≠ 0 ∧ c ≠ 1)
  (h6 : d ≠ 0 ∧ d ≠ 1) :
  S = -2 :=
by
  sorry

end sum_reciprocal_transformation_l394_394533


namespace minimum_distance_sq_l394_394423

noncomputable def b_of_a (a : ℝ) := - (1 / 2) * a^2 + 3 * real.log a

def point_on_line (m n : ℝ) := n = 2 * m + 1 / 2

def distance_sq (a m b n : ℝ) := (a - m)^2 + (b - n)^2

theorem minimum_distance_sq (a m n : ℝ) (ha : a > 0) (hmn : point_on_line m n) :
  distance_sq a m (b_of_a a) n = 9 / 5 :=
begin
  sorry
end

end minimum_distance_sq_l394_394423


namespace circles_intersecting_l394_394747

-- Definition of the circles
def circle_C1 : (ℝ × ℝ) → Prop :=
  λ p, (p.1)^2 + (p.2)^2 = 9

def circle_C2 : (ℝ × ℝ) → Prop :=
  λ p, (p.1)^2 + (p.2)^2 - 8*p.1 + 6*p.2 + 9 = 0

-- Centers and radii
def center_C1 : ℝ × ℝ := (0, 0)
def radius_C1 : ℝ := 3

def center_C2 : ℝ × ℝ := (4, -3)
def radius_C2 : ℝ := 4

-- Distance between centers
def distance_centers : ℝ := Real.sqrt ((center_C2.1 - center_C1.1)^2 + (center_C2.2 - center_C1.2)^2)

-- Positional relationship
def positional_relationship (C1 C2 : (ℝ × ℝ) → Prop) : Prop :=
  abs (radius_C1 - radius_C2) < distance_centers ∧ distance_centers < radius_C1 + radius_C2

-- Statement to prove
theorem circles_intersecting : positional_relationship circle_C1 circle_C2 :=
by
  sorry

end circles_intersecting_l394_394747


namespace inequality_proof_problem_equality_case_l394_394159

theorem inequality_proof_problem (x y : ℝ) :
  5 * x^2 + y^2 + 1 ≥ 4 * x * y + 2 * x :=
by
  sorry

theorem equality_case :
  (1 : ℝ, 2 : ℝ) :=
by
  sorry

end inequality_proof_problem_equality_case_l394_394159


namespace pascal_ratio_l394_394664

/-- Define Pascal's Triangle -/
def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1
  else pascal (n - 1) (k - 1) + pascal (n - 1) k

/-- Number of 1's in the first n rows -/
def ones_in_rows (n : ℕ) : ℕ :=
  2 * n - 1

/-- Total number of elements in the first n rows -/
def total_elements (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Number of elements that are not 1's -/
def non_ones_in_rows (n : ℕ) : ℕ :=
  (n * n - 3 * n + 2) / 2

/-- Prove that the quotient of the numbers which are not 1's and the number of 1's equals to the correct answer -/
theorem pascal_ratio (n : ℕ) (h : n > 0) : 
  (non_ones_in_rows n) / (ones_in_rows n) = (n^2 - 3*n + 2) / (4*n - 2) :=
by
  sorry

end pascal_ratio_l394_394664


namespace dietitian_calorie_excess_l394_394694

theorem dietitian_calorie_excess
  (lunch_fraction : ℚ)
  (total_calories : ℚ)
  (recommended_calories : ℚ)
  (lunch_fraction_eq : lunch_fraction = 3 / 4)
  (total_calories_eq : total_calories = 40)
  (recommended_calories_eq : recommended_calories = 25)
  : (lunch_fraction * total_calories - recommended_calories) = 5 :=
by
  rw [lunch_fraction_eq, total_calories_eq, recommended_calories_eq]
  norm_num
  sorry

end dietitian_calorie_excess_l394_394694


namespace distance_from_P_to_plane_OAB_l394_394526

def normal_vector := (2, -2, 1)
def point_P := (-1, 3, 2)

-- Define the function to calculate the distance from a point to a plane
def distance_point_to_plane (n : ℝ × ℝ × ℝ) (P : ℝ × ℝ × ℝ) : ℝ :=
  let (a, b, c) := n
  let (x₀, y₀, z₀) := P
  |(a * x₀ + b * y₀ + c * z₀) / (Real.sqrt (a^2 + b^2 + c^2))

-- The proof statement
theorem distance_from_P_to_plane_OAB : distance_point_to_plane normal_vector point_P = 2 := by
  sorry

end distance_from_P_to_plane_OAB_l394_394526


namespace max_third_side_length_l394_394996

theorem max_third_side_length 
  {A B C : ℝ} 
  (h1 : ∃ (a b c : ℝ), ∀ (α β γ : ℝ),
          cos (2 * α) + cos (2 * β) + cos (2 * γ) = 1 ∧ α = π / 2) 
  (h2 : True) 
  (h3 : ∃ (a : ℝ), a = 12 ∧ ∃ (b : ℝ), b = 15) : 
  ∃ (c : ℝ), c = Real.sqrt 369 :=
sorry

end max_third_side_length_l394_394996


namespace regular_hexagon_interior_angle_l394_394255

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l394_394255


namespace cos_equivalent_l394_394477

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom mag_a : ∥a∥ = 1
axiom mag_b : ∥b∥ = 1
axiom mag_c : ∥c∥ = real.sqrt 2
axiom abc_zero : a + b + c = 0

-- Proof statement
theorem cos_equivalent : 
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
by sorry

end cos_equivalent_l394_394477


namespace mark_one_piece_per_row_and_column_l394_394580

variable {P : Type} {piece : P}

-- Conditions
def row_has_at_least_one_piece (board : Fin 8 → Fin 8 → Option P) : Prop :=
  ∀ (i : Fin 8), ∃ (j : Fin 8), board i j = some piece

def different_rows_have_different_numbers_of_pieces (board : Fin 8 → Fin 8 → Option P) : Prop :=
  ∀ (i1 i2 : Fin 8), i1 ≠ i2 → (∃ n1, (∑ j, if board i1 j = some piece then 1 else 0) = n1) ∧
  (∃ n2, (∑ j, if board i2 j = some piece then 1 else 0) = n2) → n1 ≠ n2

-- The statement to be proven
theorem mark_one_piece_per_row_and_column (board : Fin 8 → Fin 8 → Option P)
  (h1 : row_has_at_least_one_piece board)
  (h2 : different_rows_have_different_numbers_of_pieces board) :
  ∃ (marking : Fin 8 → Fin 8), 
    function.injective marking ∧ ∀ i, board i (marking i) = some piece :=
  sorry

end mark_one_piece_per_row_and_column_l394_394580


namespace oranges_in_box_l394_394614

theorem oranges_in_box (O : ℕ) (h : 0.7 * (14 + O - 20) = 14) : O = 26 :=
sorry

end oranges_in_box_l394_394614


namespace regular_hexagon_interior_angle_l394_394267

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l394_394267


namespace sum_of_ages_of_henrys_brothers_l394_394028

theorem sum_of_ages_of_henrys_brothers (a b c : ℕ) : 
  a = 2 * b → 
  b = c ^ 2 →
  a ≠ b ∧ a ≠ c ∧ b ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  a + b + c = 14 :=
by
  intro h₁ h₂ h₃ h₄
  sorry

end sum_of_ages_of_henrys_brothers_l394_394028


namespace regular_hexagon_interior_angle_measure_l394_394237

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l394_394237


namespace no_consecutive_squares_l394_394633

theorem no_consecutive_squares (a b : ℤ) : ¬ (a^2 = b^2 + 1) :=
begin
  sorry
end

end no_consecutive_squares_l394_394633


namespace percentage_income_spent_on_transport_l394_394577

theorem percentage_income_spent_on_transport (income : ℝ) (remaining : ℝ) (spent : ℝ) (percent : ℝ) 
  (h_income : income = 2000) 
  (h_remaining : remaining = 1900) 
  (h_spent : spent = income - remaining) 
  (h_percent : percent = (spent / income) * 100) : 
  percent = 5 :=
by 
  rw [h_income, h_remaining, h_spent, h_percent]
  norm_num
  rw [div_eq_mul_inv, inv_eq_one_div]
  norm_num


end percentage_income_spent_on_transport_l394_394577


namespace find_n_l394_394503

theorem find_n (n : ℕ) (h : n > 0) (k : ℕ) (h2 : k = 2021^2) :
  (√((n * k) / n) = k) → n = k :=
by
  sorry

end find_n_l394_394503


namespace count_positive_integers_within_square_bounds_l394_394778

theorem count_positive_integers_within_square_bounds : 
  {x : ℕ | 121 ≤ x * x ∧ x * x ≤ 225}.finite.to_finset.card = 5 := by
  sorry

end count_positive_integers_within_square_bounds_l394_394778


namespace sum_of_reciprocals_l394_394626

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  (1 / x) + (1 / y) = 3 / 8 := 
sorry

end sum_of_reciprocals_l394_394626


namespace max_sin_A_l394_394044

-- Definitions for vectors and inner products
variables {V : Type*} [inner_product_space ℝ V]

-- Given Definitions
def vec_m (CB AC : V) : V := CB - 3 • AC
def vec_n (CB : V) : V := CB

-- Given the perpendicular condition
axiom perp_condition (CB AC : V) : ⟪vec_m CB AC, vec_n CB⟫ = 0

-- theorem to prove the maximum value of sin A
theorem max_sin_A (A B C : V) (AB AC : V) (hBC : B - C = vec_n (B - C)) (hAB : AB = B - A) (hAC : AC = C - A) :
  ∥AB∥ > 0 → ∥AC∥ > 0 → (⟪vec_m (B - C) AC, vec_n (B - C)⟫ = 0) →
  ∃ Amax : ℝ, Amax = 3 / 5 :=
sorry

end max_sin_A_l394_394044


namespace total_students_in_class_l394_394048

-- Definitions based on the conditions
def num_girls : ℕ := 140
def num_boys_absent : ℕ := 40
def num_boys_present := num_girls / 2
def num_boys := num_boys_present + num_boys_absent
def total_students := num_girls + num_boys

-- Theorem to be proved
theorem total_students_in_class : total_students = 250 :=
by
  sorry

end total_students_in_class_l394_394048


namespace probability_of_neither_is_correct_l394_394305

-- Definitions of the given conditions
def total_buyers : ℕ := 100
def cake_buyers : ℕ := 50
def muffin_buyers : ℕ := 40
def both_cake_and_muffin_buyers : ℕ := 19

-- Define the probability calculation function
def probability_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  let buyers_neither := total - (cake + muffin - both)
  (buyers_neither : ℚ) / (total : ℚ)

-- State the main theorem to ensure it is equivalent to our mathematical problem
theorem probability_of_neither_is_correct :
  probability_neither total_buyers cake_buyers muffin_buyers both_cake_and_muffin_buyers = 0.29 := 
sorry

end probability_of_neither_is_correct_l394_394305


namespace interior_angle_of_regular_hexagon_l394_394278

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l394_394278


namespace ab_cd_zero_l394_394829

theorem ab_cd_zero {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) (h3 : ac + bd = 0) : ab + cd = 0 :=
sorry

end ab_cd_zero_l394_394829


namespace T_2017_l394_394198

noncomputable def a_n : ℕ → ℚ
| 1     := 4
| 2     := 3 / 4
| (n+2) := if h: n > 0 then 1 - 1 / a_n (n+1) else 0 -- Derived from a_n - a_n * a_{n+1} - 1 = 0

def T_n (n : ℕ) : ℚ :=
  (List.prod (List.map a_n (List.range n)))

theorem T_2017 : T_n 2017 = 4 :=
sorry

end T_2017_l394_394198


namespace longest_side_is_six_l394_394715

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (4, 7)
def C : ℝ × ℝ := (7, 2)

-- Define a function to calculate the Euclidean distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Calculate the distances between each pair of points
def dist_AB : ℝ := distance A B
def dist_AC : ℝ := distance A C
def dist_BC : ℝ := distance B C

-- Define the length of the longest side of the triangle
def longest_side_length : ℝ := max dist_AB (max dist_AC dist_BC)

-- Prove that the longest side length is equal to 6
theorem longest_side_is_six : longest_side_length = 6 :=
by {
  -- Directly establishing conditions derived from earlier calculation steps as facts
  have h1 : dist_AB = Real.sqrt 34 := by simp [dist_AB, distance, A, B],
  have h2 : dist_AC = 6 := by simp [dist_AC, distance, A, C],
  have h3 : dist_BC = Real.sqrt 34 := by simp [dist_BC, distance, B, C],
  simp [longest_side_length, h1, h2, h3, Real.sqrt_pos],
  sorry
}

end longest_side_is_six_l394_394715


namespace solve_triangle_problem_l394_394045

noncomputable def triangle_problem 
  (a b c A B C : ℝ) 
  (hTriangle : c * sin A = a * cos C) 
  (hAngleSum : A + B + C = π)
  (hAnglesPos : 0 < A ∧ 0 < B ∧ 0 < C)
  (hAnglesRange : A < π ∧ B < π ∧ C < π)
  : Prop :=
  (C = π/4) ∧
  ((sqrt 3 * sin A - cos (B + π/4) = 2) ∧ A = π/3 ∧ B = (5 * π)/12)

-- The statement without the proof
theorem solve_triangle_problem : 
  ∃ (a b c A B C : ℝ), 
  triangle_problem a b c A B C :=
  sorry

end solve_triangle_problem_l394_394045


namespace minimum_value_f_l394_394793

theorem minimum_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : (∃ x y : ℝ, (x + y + x * y) ≥ - (9 / 8) ∧ 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :=
sorry

end minimum_value_f_l394_394793


namespace evaluate_expression_l394_394777

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem evaluate_expression (y : ℝ) (hy : y = 2 / 3) :
  floor 6.5 * floor y + floor 2 * 7.2 + floor 8.4 - 6.2 = 16.2 :=
by
  have floor_y : floor y = 0 := by linarith [floor_nonneg real_nonneg_iff]
  have : floor 6.5 = 6 := by norm_num
  have : floor 2 = 2 := by norm_num
  have : floor 8.4 = 8 := by norm_num
  calc
    floor 6.5 * floor y + floor 2 * 7.2 + floor 8.4 - 6.2
        = 6 * 0 + 2 * 7.2 + 8 - 6.2   : by rw [floor_y, this_floor_6_5, this_floor_2, this_floor_8_4]
    ... = 0 + 14.4 + 8 - 6.2         : by norm_num
    ... = 22.4 - 6.2                 : by norm_num
    ... = 16.2                       : by norm_num

end evaluate_expression_l394_394777


namespace vector_linear_combination_l394_394365

open Matrix

theorem vector_linear_combination :
  let v1 := ![3, -9]
  let v2 := ![2, -8]
  let v3 := ![1, -6]
  4 • v1 - 3 • v2 + 2 • v3 = ![8, -24] :=
by sorry

end vector_linear_combination_l394_394365


namespace part1_no_second_quadrant_part2_triangle_area_l394_394020

theorem part1_no_second_quadrant (k : ℝ) (line : ∀ x, ℝ) : 
  (∀ x, line x = k * x - 2 * k + 1) → ¬(∃ x, y, x < 0 ∧ y > 0 ∧ y = line x) → k ∈ Set.Ici (1/2) :=
sorry

theorem part2_triangle_area (k : ℝ) (line : ∀ x, ℝ) :
  (∀ x, line x = k * x - 2 * k + 1) →
  (∃ x y : ℝ, y = 0 ∧ x = 2 - 1/k ∧ line x = y)
  ∧ (∃ x y : ℝ, x = 0 ∧ y = 1 - 2 * k ∧ line y = y)
  ∧ (1/2 * |2 - 1/k| * |1 - 2 * k| = 9/2) →
  (line = (λ x, -x + 3) ∨ line = (λ x, -1/4 * x + 3/2)) :=
sorry

end part1_no_second_quadrant_part2_triangle_area_l394_394020


namespace f_prime_at_2_l394_394439

noncomputable def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := f(x) / x

-- Condition 1: The point (2, 1) lies on y = f(x) / x which gives f(2) = 2 
def condition1 : Prop := f(2) = 2

-- Condition 2: The lines tangent to y = f(x) at (0,0) and y = f(x) / x at (2,1) have the same slope == 1/2
def tangent_slope : ℝ := 1/2
def g' := λ x, (x * (Deriv f x) - f x) / (x^2)
def condition2 : Prop := g'(2) = tangent_slope

-- Proving f'(2) = 2
theorem f_prime_at_2 : condition1 ∧ condition2 → (Deriv f 2) = 2 :=
by
  sorry

end f_prime_at_2_l394_394439


namespace correct_quadratic_equation_l394_394656

-- The main statement to prove.
theorem correct_quadratic_equation :
  (∀ (x y a : ℝ), (3 * x + 2 * y - 1 ≠ 0) ∧ (5 * x^2 - 6 * y - 3 ≠ 0) ∧ (a * x^2 - x + 2 ≠ 0) ∧ (x^2 - 1 = 0) → (x^2 - 1 = 0)) :=
by
  sorry

end correct_quadratic_equation_l394_394656


namespace average_speed_x_to_z_l394_394590

theorem average_speed_x_to_z 
  (d : ℝ)
  (h1 : d > 0)
  (distance_xy : ℝ := 2 * d)
  (distance_yz : ℝ := d)
  (speed_xy : ℝ := 100)
  (speed_yz : ℝ := 75)
  (total_distance : ℝ := distance_xy + distance_yz)
  (time_xy : ℝ := distance_xy / speed_xy)
  (time_yz : ℝ := distance_yz / speed_yz)
  (total_time : ℝ := time_xy + time_yz) :
  total_distance / total_time = 90 :=
by
  sorry

end average_speed_x_to_z_l394_394590


namespace probability_of_selecting_meiqi_l394_394680

def four_red_bases : List String := ["Meiqi", "Wangcunkou", "Zhulong", "Xiaoshun"]

theorem probability_of_selecting_meiqi :
  (1 / 4 : ℝ) = 1 / (four_red_bases.length : ℝ) :=
  by sorry

end probability_of_selecting_meiqi_l394_394680


namespace cosine_of_subtracted_vectors_l394_394471

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Conditions
def condition1 := ∥a∥ = 1
def condition2 := ∥b∥ = 1
def condition3 := ∥c∥ = real.sqrt 2
def condition4 := a + b + c = 0

-- Proof statement
theorem cosine_of_subtracted_vectors 
  (h1 : condition1 a) 
  (h2 : condition2 b) 
  (h3 : condition3 c) 
  (h4 : condition4 a b c) : 
  inner_product_space.cos (a - c) (b - c) = 4 / 5 :=
sorry

end cosine_of_subtracted_vectors_l394_394471


namespace solve_for_x_l394_394165

-- Assumptions and conditions of the problem
def a : ℚ := 4 / 7
def b : ℚ := 1 / 5
def c : ℚ := 12
def d : ℚ := 105

-- The statement of the problem
theorem solve_for_x (x : ℚ) (h : a * b * x = c) : x = d :=
by sorry

end solve_for_x_l394_394165


namespace number_of_equilateral_triangles_l394_394180

-- Define the problem with conditions and expected result
theorem number_of_equilateral_triangles :
  ∀ (k : ℤ), 
  (-8 ≤ k ∧ k ≤ 8) →
  (∃ (number_of_triangles : ℕ),
     number_of_triangles = 480) :=
begin
  intros k hk,
  use 480,
  sorry -- Proof to be constructed
end

end number_of_equilateral_triangles_l394_394180


namespace additional_charge_is_correct_l394_394914

noncomputable def additional_charge_per_segment (initial_fee : ℝ) (total_distance : ℝ) (total_charge : ℝ) (segment_length : ℝ) : ℝ :=
  let segments := total_distance / segment_length
  let charge_for_distance := total_charge - initial_fee
  charge_for_distance / segments

theorem additional_charge_is_correct :
  additional_charge_per_segment 2.0 3.6 5.15 (2/5) = 0.35 :=
by
  sorry

end additional_charge_is_correct_l394_394914


namespace problem_statement_l394_394554

noncomputable def a : ℝ := (Real.tan 23) / (1 - (Real.tan 23) ^ 2)
noncomputable def b : ℝ := 2 * Real.sin 13 * Real.cos 13
noncomputable def c : ℝ := Real.sqrt ((1 - Real.cos 50) / 2)

theorem problem_statement : c < b ∧ b < a :=
by
  -- Proof omitted
  sorry

end problem_statement_l394_394554


namespace correct_equation_l394_394299

theorem correct_equation (x y a b : ℝ) :
  ¬ (-(x - 6) = -x - 6) ∧
  ¬ (-y^2 - y^2 = 0) ∧
  ¬ (9 * a^2 * b - 9 * a * b^2 = 0) ∧
  (-9 * y^2 + 16 * y^2 = 7 * y^2) :=
by
  sorry

end correct_equation_l394_394299


namespace monotonicity_of_f_range_of_a_l394_394804

noncomputable def f (a x : ℝ) : ℝ := -2 * x^3 + 6 * a * x^2 - 1
noncomputable def g (a x : ℝ) : ℝ := Real.exp x - 2 * a * x - 1

theorem monotonicity_of_f (a : ℝ) :
  (a < 0 → (∀ x, f a x ≥ f a (2 * a) → x ≤ 0 ∨ x ≤ 2 * a)) ∧
  (a = 0 → ∀ x y, x ≤ y → f a x ≥ f a y) ∧
  (a > 0 → (∀ x, f a x ≤ f a 0 → x ≤ 0) ∧
           (∀ x, 0 < x ∧ x < 2 * a → f a x ≥ f a 2 * a) ∧
           (∀ x, 2 * a < x → f a x ≤ f a (2 * a))) :=
sorry

theorem range_of_a :
  ∀ a : ℝ, a ≥ 1 / 2 →
  ∃ x1 : ℝ, x1 > 0 ∧ ∃ x2 : ℝ, f a x1 ≥ g a x2 :=
sorry

end monotonicity_of_f_range_of_a_l394_394804


namespace minimum_distance_to_line_l394_394446

noncomputable def C1 (t : Real) : Real × Real :=
  (-4 + Real.cos t, 3 + Real.sin t)

noncomputable def C2 (θ : Real) : Real × Real :=
  (8 * Real.cos θ, 3 * Real.sin θ)

def P : Real × Real := C1 (Real.pi / 2)

def line_C3 (x y : Real) : Prop := x - 2 * y - 7 = 0

def midpoint (P Q : Real × Real) : Real × Real :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def distance_to_line (M : Real × Real) : Real :=
  Real.abs (4 * M.1 - 3 * M.2 - 13) / 5

theorem minimum_distance_to_line :
  ∃ θ : Real, distance_to_line (midpoint P (C2 θ)) = (8 * Real.sqrt 5) / 5 :=
sorry

end minimum_distance_to_line_l394_394446


namespace angle_between_p_and_q_is_zero_degrees_l394_394924

variables {V : Type*} [inner_product_space ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

theorem angle_between_p_and_q_is_zero_degrees
  (p q r : V)
  (hp : is_unit_vector p)
  (hq : is_unit_vector q)
  (hr : is_unit_vector r)
  (h : p + q + 2 • r = 0) :
  real.angle p q = 0 :=
sorry

end angle_between_p_and_q_is_zero_degrees_l394_394924


namespace equal_mass_after_transfer_l394_394683

variable (V1 V2 D1 D2 x : ℕ)

def mass1_initial := V1 * D1
def mass2_initial := V2 * D2

def mass2_after_mix := mass2_initial + D1 * x
def volume2_after_mix := V2 + x

def density2_after_mix := mass2_after_mix / volume2_after_mix

def mass_each := (mass1_initial + mass2_initial) / 2

theorem equal_mass_after_transfer :
  V1 = 80 → V2 = 100 → D1 = 6 / 5 → D2 = 4 / 5 → 
  (density2_after_mix = 44 / 50) → x = 25 :=
by
  intros
  sorry

end equal_mass_after_transfer_l394_394683


namespace probability_sum_of_10_dice_equals_50_l394_394712

open Set

noncomputable def probability_sum_50 (n_dice : ℕ) (n_faces : ℕ) (desired_sum : ℕ) : ℚ :=
  let total_outcomes := n_faces^n_dice
  let favorable_outcomes := Nat.choose ((desired_sum - n_dice) + (n_dice - 1)) (n_dice - 1)
  favorable_outcomes / total_outcomes

theorem probability_sum_of_10_dice_equals_50 :
  probability_sum_50 10 8 50 ≈ 0.763 := by
  sorry

end probability_sum_of_10_dice_equals_50_l394_394712


namespace well_diameter_is_two_meters_l394_394681

noncomputable def volume_of_well (diameter : ℝ) (height : ℝ) :=
  π * (diameter / 2) ^ 2 * height

theorem well_diameter_is_two_meters :
  ∃ diameter : ℝ, 
  (∀ volume height : ℝ, (volume = 31.41592653589793) ∧ (height = 10) →
  volume_of_well diameter height = volume) ∧ 
  diameter = 2 :=
begin
  use 2,
  intros volume height h,
  cases h with h_volume h_height,
  rw [h_volume, h_height, volume_of_well],
  norm_num,
  sorry
end

end well_diameter_is_two_meters_l394_394681


namespace triangle_side_relationship_l394_394424

theorem triangle_side_relationship
  (a b c : ℝ)
  (habc : a < b + c)
  (ha_pos : a > 0) :
  a^2 < a * b + a * c :=
by
  sorry

end triangle_side_relationship_l394_394424


namespace algebraic_expression_value_l394_394870

theorem algebraic_expression_value (a b : ℝ) (h1 : a ≠ b) (h2 : a^2 - 5 * a + 2 = 0) (h3 : b^2 - 5 * b + 2 = 0) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -13 / 2 := by
  sorry

end algebraic_expression_value_l394_394870


namespace first_nonzero_digit_fraction_l394_394287

theorem first_nonzero_digit_fraction (n d : ℕ) (h1 : d = 149) (h2 : n = 1) :
  let fractional_part := (n:ℝ) / (d:ℝ) - real.floor ((n:ℝ) / (d:ℝ)) in
  (real.floor (fractional_part * 10)) = 7 :=
by
  sorry

end first_nonzero_digit_fraction_l394_394287


namespace unit_circle_root_l394_394005

noncomputable def z_eq (n : ℕ) (a : ℝ) (z : ℂ) : Prop := z^(n+1) - a * z^n + a * z - 1 = 0

theorem unit_circle_root (n : ℕ) (a : ℝ) (z : ℂ)
  (h1 : 2 ≤ n)
  (h2 : 0 < a)
  (h3 : a < (n+1) / (n-1))
  (h4 : z_eq n a z) :
  ∥z∥ = 1 := sorry

end unit_circle_root_l394_394005


namespace ratio_of_potatoes_l394_394911

def total_potatoes : ℕ := 24
def number_of_people : ℕ := 3
def potatoes_per_person : ℕ := 8
def total_each_person : ℕ := potatoes_per_person * number_of_people

theorem ratio_of_potatoes :
  total_potatoes = total_each_person → (potatoes_per_person : ℚ) / (potatoes_per_person : ℚ) = 1 :=
by
  sorry

end ratio_of_potatoes_l394_394911


namespace exists_n_sum_1990_consecutive_l394_394618

theorem exists_n_sum_1990_consecutive :
  ∃ (n : ℕ),
    (∃ (x : ℕ), n = 1990 * x + 1989 * 995) ∧
    (∃ k, (∀ i : ℕ, 1 ≤ i ∧ i ≤ k → ∃ (x : ℕ), n = i * x + i * (i + 1) / 2) ∧ k = 1990) ∧
    (n = nat.pow 5 9 * nat.pow 199 198 ∨ n = nat.pow 5 198 * nat.pow 199 9) :=
begin
  sorry
end

end exists_n_sum_1990_consecutive_l394_394618


namespace equivalent_trigonometric_identity_l394_394866

variable (α : ℝ)

theorem equivalent_trigonometric_identity
  (h1 : α ∈ Set.Ioo (-(Real.pi/2)) 0)
  (h2 : Real.sin (α + (Real.pi/4)) = -1/3) :
  (Real.sin (2*α) / Real.cos ((Real.pi/4) - α)) = 7/3 := 
by
  sorry

end equivalent_trigonometric_identity_l394_394866


namespace count_qualifying_rows_l394_394375

-- Define Pascal's triangle as a function of binomial coefficients.
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- Condition: Rows must range from 0 to 29.
def in_range (n : ℕ) : Prop := n ≤ 29

-- Condition: Exclude the first and last entries in Pascal's triangle, only consider internal entries.
def internal_entries (n : ℕ) (k : ℕ) : Prop := 1 ≤ k ∧ k < n

-- Condition: Internal entries must all be even.
def all_even (n : ℕ) : Prop :=
  ∀ k, internal_entries n k → pascal n k % 2 = 0

-- Condition: No two consecutive internal entries are equal.
def no_consecutive_equal (n : ℕ) : Prop :=
  ∀ k, internal_entries n k → internal_entries n (k+1) → pascal n k ≠ pascal n (k+1)

-- Define the exact property of rows we are looking for.
def qualifying_row (n : ℕ) : Prop :=
  in_range n ∧ all_even n ∧ no_consecutive_equal n

-- Define the proof problem to count such rows.
theorem count_qualifying_rows : ∃ count, count = 3 ∧ count = (Finset.card (Finset.filter qualifying_row (Finset.range 30))) :=
by
  sorry

end count_qualifying_rows_l394_394375


namespace find_f_neg_one_l394_394001

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (1 + x) else - f (-x)

theorem find_f_neg_one :
    f(-1) = -2 :=
by
  -- Proof to be filled
  sorry

end find_f_neg_one_l394_394001


namespace Carlotta_final_stage_performance_l394_394410

theorem Carlotta_final_stage_performance :
  ∀ (x : ℕ), (3 * x + 5 * x + x = 54) → x = 6 :=
by
  intros x h
  have h_eq : 9 * x = 54 := by
    rw [← add_mul, ← add_mul] at h
    cases h
  exact Eq.trans h_eq (Nat.div_eq_of_eq_mul_right 6 h_eq)


end Carlotta_final_stage_performance_l394_394410


namespace marilyn_total_caps_l394_394575

def marilyn_initial_caps : ℝ := 51.0
def nancy_gives_caps : ℝ := 36.0
def total_caps (initial: ℝ) (given: ℝ) : ℝ := initial + given

theorem marilyn_total_caps : total_caps marilyn_initial_caps nancy_gives_caps = 87.0 :=
by
  sorry

end marilyn_total_caps_l394_394575


namespace regular_hexagon_interior_angle_l394_394259

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l394_394259


namespace side_length_S2_l394_394121

def square_side_length 
  (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : Prop :=
  s = 650

theorem side_length_S2 (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : square_side_length w h R1 R2 S1 S2 S3 r s combined_rectangle cond1 cond2 cond3 cond4 cond5 cond6 cond7 cond8 :=
sorry

end side_length_S2_l394_394121


namespace arrangement_of_volunteers_l394_394351

theorem arrangement_of_volunteers :
  let volunteers := {A, B, C, D, E, F}
  let elders := {甲, 乙, 丙}
  let pairs := { (s : elders × fin 2 → volunteers) //
    ∀ (a : elders), ∃ (v1 v2 : volunteers), v1 ≠ v2 ∧ (s a 0 = v1 ∧ s a 1 = v2)}
  -- Given conditions
  (∀ s : pairs, s 甲 0 ≠ A ∧ s 甲 1 ≠ A) ∧
  (∀ s : pairs, s 乙 0 ≠ B ∧ s 乙 1 ≠ B)
  -- Total arrangements
  → finset.card pairs = 42 := 
sorry

end arrangement_of_volunteers_l394_394351


namespace exists_five_positive_integers_sum_20_product_420_l394_394942
-- Import the entirety of Mathlib to ensure all necessary definitions are available

-- Lean statement for the proof problem
theorem exists_five_positive_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ a + b + c + d + e = 20 ∧ a * b * c * d * e = 420 :=
sorry

end exists_five_positive_integers_sum_20_product_420_l394_394942


namespace value_of_A_l394_394967

-- Define the variables and the expression for A
variables {a b c : ℝ}

theorem value_of_A (a b c : ℝ) :
  (a ≠ b) → (a ≠ c) → (b ≠ c) →
  ( ( (b - c)^2 / ( (a - b) * (a - c) ) ) + ( (c - a)^2 / ( (b - c) * (b - a) ) ) + ( (a - b)^2 / ( (c - a) * (c - b) ) ) ) = 3 :=
begin
  sorry
end

end value_of_A_l394_394967


namespace anne_winning_strategy_l394_394350

def anne_bills_game (n : ℕ) (N0 : ℕ) : Prop :=
  ∃ k : ℕ, N0 - n^k = 0 ∨ ∀ j : ℕ, N0 - n^j > 0

def f (n : ℕ) : ℕ :=
  (finset.range 5001).filter (λ N0, ∃ k : ℕ, 1 ≤ N0 ∧ (N0 ≤ 5000) ∧ (N0 - n^k = 0 ∨ ∀ j : ℕ, N0 - n^j > 0)).card

theorem anne_winning_strategy :
  ∃ (count : ℕ), count = 63 ∧ (finset.range 1001).filter (λ n, f n ≥ 2520).card = count :=
by 
  sorry

end anne_winning_strategy_l394_394350


namespace john_saving_yearly_l394_394540

def old_monthly_cost : ℕ := 1200
def increase_percentage : ℕ := 40
def split_count : ℕ := 3

def old_annual_cost (monthly_cost : ℕ) := monthly_cost * 12
def new_monthly_cost (monthly_cost : ℕ) (percentage : ℕ) := monthly_cost * (100 + percentage) / 100
def new_monthly_share (new_cost : ℕ) (split : ℕ) := new_cost / split
def new_annual_cost (monthly_share : ℕ) := monthly_share * 12
def annual_savings (old_annual : ℕ) (new_annual : ℕ) := old_annual - new_annual

theorem john_saving_yearly 
  (old_cost : ℕ := old_monthly_cost)
  (increase : ℕ := increase_percentage)
  (split : ℕ := split_count) :
  annual_savings (old_annual_cost old_cost) 
                 (new_annual_cost (new_monthly_share (new_monthly_cost old_cost increase) split)) 
  = 7680 :=
by
  sorry

end john_saving_yearly_l394_394540


namespace part1_part2_l394_394018

noncomputable def f (x : ℝ) : ℝ := abs (x - 3 / 4) + abs (x + 5 / 4)

theorem part1:
  {a : ℝ} (h : ∃ x, f x ≤ a) → 2 ≤ a :=
sorry

theorem part2 
  (m n : ℝ) 
  (hm : 0 < m) 
  (hn : 0 < n) 
  (h_sum : m + 2 * n = 2) :
  ∀ x, sqrt (m + 1) + sqrt (2 * n + 1) ≤ 2 * sqrt (f x) :=
sorry

end part1_part2_l394_394018


namespace g_range_l394_394088

noncomputable def g (x y z : ℝ) : ℝ := 
  (x^2 / (x^2 + y^2)) + (y^2 / (y^2 + z^2)) + (z^2 / (z^2 + x^2))

theorem g_range (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 / 2 ≤ g x y z ∧ g x y z ≤ 2 :=
sorry

end g_range_l394_394088


namespace joey_needs_figures_to_cover_cost_l394_394915

-- Definitions based on conditions
def cost_sneakers : ℕ := 92
def earnings_per_lawn : ℕ := 8
def lawns : ℕ := 3
def earnings_per_hour : ℕ := 5
def work_hours : ℕ := 10
def price_per_figure : ℕ := 9

-- Total earnings from mowing lawns
def earnings_lawns := lawns * earnings_per_lawn
-- Total earnings from job
def earnings_job := work_hours * earnings_per_hour
-- Total earnings from both
def total_earnings := earnings_lawns + earnings_job
-- Remaining amount to cover the cost
def remaining_amount := cost_sneakers - total_earnings

-- Correct answer based on the problem statement
def collectible_figures_needed := remaining_amount / price_per_figure

-- Lean 4 statement to prove the requirement
theorem joey_needs_figures_to_cover_cost :
  collectible_figures_needed = 2 := by
  sorry

end joey_needs_figures_to_cover_cost_l394_394915


namespace regular_hexagon_interior_angle_l394_394268

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l394_394268


namespace cost_of_weed_eater_string_l394_394702

-- Definitions
def num_blades := 4
def cost_per_blade := 8
def total_spent := 39
def total_cost_of_blades := num_blades * cost_per_blade
def cost_of_string := total_spent - total_cost_of_blades

-- The theorem statement
theorem cost_of_weed_eater_string : cost_of_string = 7 :=
by {
  -- The proof would go here
  sorry
}

end cost_of_weed_eater_string_l394_394702


namespace cosine_of_subtracted_vectors_l394_394474

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Conditions
def condition1 := ∥a∥ = 1
def condition2 := ∥b∥ = 1
def condition3 := ∥c∥ = real.sqrt 2
def condition4 := a + b + c = 0

-- Proof statement
theorem cosine_of_subtracted_vectors 
  (h1 : condition1 a) 
  (h2 : condition2 b) 
  (h3 : condition3 c) 
  (h4 : condition4 a b c) : 
  inner_product_space.cos (a - c) (b - c) = 4 / 5 :=
sorry

end cosine_of_subtracted_vectors_l394_394474


namespace people_at_table_l394_394124

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l394_394124


namespace min_value_of_f_l394_394399

noncomputable def f (x : ℝ) : ℝ := 3 * real.sqrt (2 * x) + 4 / x

theorem min_value_of_f : ∀ x : ℝ, x > 0 → ∀ y : ℝ, (∀ z : ℝ, z > 0 → f(z) ≥ y) → y = 8 :=
by sorry

end min_value_of_f_l394_394399


namespace Mrs_Hilt_walks_to_fountain_l394_394103

theorem Mrs_Hilt_walks_to_fountain :
  ∀ (distance trips : ℕ), distance = 30 → trips = 4 → distance * trips = 120 :=
by
  intros distance trips h_distance h_trips
  sorry

end Mrs_Hilt_walks_to_fountain_l394_394103


namespace geometric_sequence_sum_l394_394083

noncomputable def a_n (a₁ q n : ℕ) : ℝ := a₁ * (q ^ (n - 1))

noncomputable def S (a₁ q n : ℕ) : ℝ := (a₁ * (1 - q ^ n)) / (1 - q)

theorem geometric_sequence_sum (a₁ q : ℝ) (n : ℕ)
  (h1 : a₁ > 0) 
  (h2 : q > 0)
  (h3 : 4 * a₁ - a₁ * (q ^ 2) = 0) :
  (S a₁ q 3) / a₁ = 7 :=
by
  sorry

end geometric_sequence_sum_l394_394083


namespace base_b_addition_not_divisible_by_five_l394_394781

def not_divisible_by_five (n : ℕ) : Prop := ¬ (5 ∣ n)

theorem base_b_addition_not_divisible_by_five :
  ∀ b : ℕ,
    b ∈ {3, 4, 6, 7, 8} →
    not_divisible_by_five (3 * b^3 + 2 * b^2 + 6 * b + 3) ↔ (b = 4 ∨ b = 6) :=
by
  sorry

end base_b_addition_not_divisible_by_five_l394_394781


namespace volume_of_112_ounces_l394_394204

variable (k : ℚ)
variable (V W : ℚ)

theorem volume_of_112_ounces (h1 : V = k * W) (h2 : 27 = k * 63) : V = 48 :=
by
  have : k = 3 / 7 := by
    sorry
  have vol_112 := (3/7) * 112
  show V = vol_112
  sorry

end volume_of_112_ounces_l394_394204


namespace find_x_l394_394039

theorem find_x (x : ℝ) (h : x^29 * 4^15 = 2 * 10^29) : x = 5 := 
by 
  sorry

end find_x_l394_394039


namespace rightmost_three_digits_of_3_pow_1987_l394_394648

theorem rightmost_three_digits_of_3_pow_1987 :
  3^1987 % 2000 = 187 :=
by sorry

end rightmost_three_digits_of_3_pow_1987_l394_394648


namespace son_l394_394703

-- Define the context of the problem with conditions
variables (S M : ℕ)

-- Condition 1: The man is 28 years older than his son
def condition1 : Prop := M = S + 28

-- Condition 2: In two years, the man's age will be twice the son's age
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- The final statement to prove the son's present age
theorem son's_age (h1 : condition1 S M) (h2 : condition2 S M) : S = 26 :=
by
  sorry

end son_l394_394703


namespace real_roots_iff_a_in_interval_l394_394405

theorem real_roots_iff_a_in_interval (a : ℝ) :
  (∃ x : ℝ, x^2 - 4 * x + |a| + |a - 3| = 0) ↔ (a ∈ set.Icc (-(1:ℝ)/2) (7/2)) :=
by sorry

end real_roots_iff_a_in_interval_l394_394405


namespace tan_theta_minus_reciprocal_l394_394786

noncomputable def theta : ℝ := sorry -- θ is some number in (0, π)

-- Assume θ is within the specified interval
def theta_in_interval : 0 < theta ∧ theta < Real.pi := sorry

-- Assume sinθ and cosθ are the roots of the given quadratic equation
def sin_cos_roots : ∃ (sin_theta cos_theta : ℝ), 
  25 * sin_theta ^ 2 - 5 * sin_theta - 12 = 0 ∧ 
  25 * cos_theta ^ 2 - 5 * cos_theta - 12 = 0 ∧ 
  sin_theta = Real.sin theta ∧ 
  cos_theta = Real.cos theta := sorry

-- Define the main theorem/question
theorem tan_theta_minus_reciprocal (htheta : theta_in_interval) (hsin_cos : sin_cos_roots) : 
  Real.tan theta - (1 / Real.tan theta) = -7 / 12 := 
  sorry

end tan_theta_minus_reciprocal_l394_394786


namespace exists_root_in_interval_l394_394619

noncomputable def f (x : ℝ) : ℝ := x^2 + log x - 4

theorem exists_root_in_interval : ∃ α ∈ Ioo (2 : ℝ) (3 : ℝ), f α = 0 := sorry

end exists_root_in_interval_l394_394619


namespace monitor_height_l394_394206

theorem monitor_height (width circumference : ℕ) (h_width : width = 12) (h_circumference : circumference = 38) :
  2 * (width + 7) = circumference :=
by
  sorry

end monitor_height_l394_394206


namespace x_intercept_of_line_through_10_3_neg2_neg3_l394_394671

noncomputable def line_through_points_x_intercept (p1 p2 : ℝ × ℝ) : ℝ :=
let m := (p2.2 - p1.2) / (p2.1 - p1.1) in
let b := p1.2 - m * p1.1 in
- b / m

theorem x_intercept_of_line_through_10_3_neg2_neg3 :
  line_through_points_x_intercept (10, 3) (-2, -3) = 4 :=
by
  -- Proof omitted.
  sorry

end x_intercept_of_line_through_10_3_neg2_neg3_l394_394671


namespace least_num_square_tiles_l394_394307

theorem least_num_square_tiles :
  ∀ (length width : ℕ), length = 624 → width = 432 → (length * width) / (Nat.gcd length width)^2 = 117 :=
begin
  sorry
end

end least_num_square_tiles_l394_394307


namespace university_language_students_l394_394754

theorem university_language_students :
  let G_min := ⌈0.75 * 2500⌉
  let G_max := ⌊0.80 * 2500⌋
  let R_min := ⌈0.35 * 2500⌉
  let R_max := ⌊0.45 * 2500⌋
  G_min ≤ G ∧ G ≤ G_max →
  R_min ≤ R ∧ R ≤ R_max →
  (G + R ≥ 2500) →
  let m' := G + R - 2500
  let M' := 2500 - (G + R - 2500)
  M' - m' = 625 :=
begin
  sorry
end

end university_language_students_l394_394754


namespace simplify_trig_expression_l394_394973

theorem simplify_trig_expression (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := by
  sorry

end simplify_trig_expression_l394_394973


namespace solve_quadratic_inequality_l394_394995

noncomputable def is_solution_interval (a b : ℝ) : set ℝ := {x : ℝ | a ≤ x ∧ x ≤ b}

noncomputable def quadratic_inequality (x : ℝ) : ℝ := -x^2 - 2*x + 3

theorem solve_quadratic_inequality :
    ∀ (x : ℝ), x ∈ is_solution_interval (-1 - Real.sqrt 2) (-1 + Real.sqrt 2) →
    quadratic_inequality x ≥ 0 :=
by
  sorry

end solve_quadratic_inequality_l394_394995


namespace fixed_point_of_tangents_l394_394443

theorem fixed_point_of_tangents 
  (P : ℝ × ℝ) (x y : ℝ) 
  (hP : ∃ m : ℝ, P = (2 * m + 3, m) ∧ P.1 - 2 * P.2 - 3 = 0) 
  (hO : x^2 + y^2 = 1) 
  (tangent_condition : ∀ A B : ℝ × ℝ, tangent O P A ∧ tangent O P B → line_through A B (1 / 3, -2 / 3)) 
  : line_passes_through_fixed_point P (1 / 3, -2 / 3) :=
begin
  sorry
end


end fixed_point_of_tangents_l394_394443


namespace count_special_5digit_numbers_correct_l394_394923

noncomputable def count_special_5digit_numbers : Nat :=
  let digits : List Nat := [0, 1, 2, 3, 4, 6, 7, 8, 9]
  let ten_thousands_place_choices := digits.erase 0
  let ten_thousands_place_choices := ten_thousands_place_choices.erase 5
  let count := 0

  for d₁ in ten_thousands_place_choices do
    let remaining_digits₁ := digits.erase d₁
    let remaining_digits₁ := remaining_digits₁.erase 5

    for d₂ in remaining_digits₁ do
      let remaining_digits₂ := remaining_digits₁.erase d₂

      for d₃ in remaining_digits₂ do
        let remaining_digits₃ := remaining_digits₂.erase d₃

        for d₄ in remaining_digits₃ do
          let remaining_digits₄ := remaining_digits₃.erase d₄

          for d₅ in remaining_digits₄ do
            let count := count + 1
  
  count

theorem count_special_5digit_numbers_correct :
  count_special_5digit_numbers = 24192 :=
by
  sorry

end count_special_5digit_numbers_correct_l394_394923


namespace twelfth_term_l394_394409

noncomputable def a (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else (n * (n + 2)) - ((n - 1) * (n + 1))

theorem twelfth_term : a 12 = 25 :=
by sorry

end twelfth_term_l394_394409


namespace intervals_of_monotonicity_of_g_decreasing_f_implies_min_a_range_of_a_for_condition_l394_394017

noncomputable def g (x : ℝ) := x / (Real.log x)
noncomputable def f (x a : ℝ) := g x - a * x

theorem intervals_of_monotonicity_of_g :
  (∀ x ∈ (0, 1), deriv g x < 0) ∧
  (∀ x ∈ (1, e), deriv g x < 0) ∧
  (∀ x ∈ (e, +∞), deriv g x > 0) := sorry

theorem decreasing_f_implies_min_a (a : ℝ) :
  (∀ x ∈ (1, +∞), deriv (λ x, f x a) x ≤ 0) → a ≥ 1/4 := sorry

theorem range_of_a_for_condition (a : ℝ) :
  (∃ x1 x2 ∈ ([e : ℝ, Real.exp 2]), f x1 a ≤ deriv (λ x, f x a) x2 + a)
  → a ≥ (1/2) - (1/(4 * Real.exp 2)) := sorry

end intervals_of_monotonicity_of_g_decreasing_f_implies_min_a_range_of_a_for_condition_l394_394017


namespace planes_share_common_point_l394_394061

noncomputable def point_symmetric (S: Point) (M: Point) (L: Line) (A: Point) : Prop :=
  symmetric L S A

variable {A B C S A' B' C': Point}

axiom no_two_edges_equal_length : (AB ≠ AC) ∧ (AB ≠ BC) ∧ (AC ≠ BC) ∧ (AS ≠ BS) ∧ (AS ≠ CS) ∧ (BS ≠ CS)

axiom A'_symmetric_to_S : point_symmetric S (midpoint B C) (perpendicular_bisector B C) A'
axiom B'_symmetric_to_S : point_symmetric S (midpoint A C) (perpendicular_bisector A C) B'
axiom C'_symmetric_to_S : point_symmetric S (midpoint A B) (perpendicular_bisector A B) C'

theorem planes_share_common_point :
  ∃ P : Point, is_on P (plane ABC) ∧ is_on P (plane AB'C') ∧ is_on P (plane A'BC') ∧ is_on P (plane A'B'C) :=
sorry

end planes_share_common_point_l394_394061


namespace minimum_f_value_range_a_sequences_exist_l394_394014

noncomputable def f (x : ℝ) : ℝ := real.e^x - x

-- Statement for the first proof problem: finding the minimum value
theorem minimum_f : ∃ x : ℝ, f(x) = 1 ∧ ∀ y, f(y) ≥ 1 := sorry

-- Statement for the second proof problem: range of values for a
theorem value_range_a (a : ℝ) : (∃ x : ℝ, (1/2 ≤ x ∧ x ≤ 2) ∧ f(x) > a * x) ↔ a < real.e^2 / 2 - 1 := sorry

-- Statement for the third proof problem: existence of sequences {a_n} and {b_n}
theorem sequences_exist (n : ℕ) (n_pos : 0 < n) :
  ∃ (a b : ℕ → ℝ),
    (a 1 + b 1 = ∫ t in 0..1, f t) ∧
    (∀ k, 1 ≤ k → a k + b k = (∫ t in 0..k, f t) - (∫ t in 0..k-1, f t)) ∧
    (∃ d : ℝ, ∀ m, 1 ≤ m → a (m + 1) = a m + d) ∧
    (∃ c r : ℝ, 0 < r ∧ (b 1 = f 1) ∧ ∀ m, 1 ≤ m → b (m + 1) = b m * r) := sorry

end minimum_f_value_range_a_sequences_exist_l394_394014


namespace dietitian_calorie_intake_l394_394691

variable (food_calories : ℕ) (fraction_eaten : ℚ) (recommended_calories : ℕ)

-- Conditions
def conditions := food_calories = 40 ∧ fraction_eaten = 3/4 ∧ recommended_calories = 25

-- Derived result
def result := (fraction_eaten * food_calories : ℚ) - recommended_calories = 5

-- The statement to prove
theorem dietitian_calorie_intake (h : conditions food_calories fraction_eaten recommended_calories) :
  result food_calories fraction_eaten recommended_calories :=
begin
  sorry
end

end dietitian_calorie_intake_l394_394691


namespace total_time_before_main_game_l394_394909

-- Define the time spent on each activity according to the conditions
def download_time := 10
def install_time := download_time / 2
def update_time := 2 * download_time
def account_time := 5
def internet_issues_time := 15
def discussion_time := 20
def video_time := 8

-- Define the total preparation time
def preparation_time := download_time + install_time + update_time + account_time + internet_issues_time + discussion_time + video_time

-- Define the in-game tutorial time
def tutorial_time := 3 * preparation_time

-- Prove that the total time before playing the main game is 332 minutes
theorem total_time_before_main_game : preparation_time + tutorial_time = 332 := by
  -- Provide a detailed proof here
  sorry

end total_time_before_main_game_l394_394909


namespace false_proposition_l394_394025

variable (x : ℝ)

def p : Prop := (|x| = x) ↔ (x > 0)
def q : Prop := (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0)

theorem false_proposition : ¬ (p ∧ q) :=
sorry

end false_proposition_l394_394025


namespace teena_speed_l394_394171

theorem teena_speed (T : ℝ) : 
  (∀ (d₀ d_poe d_ahead : ℝ), 
    d₀ = 7.5 ∧ d_poe = 40 * 1.5 ∧ d_ahead = 15 →
    T = (d₀ + d_poe + d_ahead) / 1.5) → 
  T = 55 :=
by
  intros
  sorry

end teena_speed_l394_394171


namespace wenlock_olympian_games_first_held_year_difference_l394_394109

theorem wenlock_olympian_games_first_held_year_difference :
  2012 - 1850 = 162 :=
sorry

end wenlock_olympian_games_first_held_year_difference_l394_394109


namespace general_formula_l394_394455

noncomputable def a_seq : ℕ → ℤ
| 1     := 2
| 2     := -6
| 3     := 12
| n + 1 := (-1)^(n + 1 + 1) * (n + 1) * (n + 2)

theorem general_formula (n : ℕ) (hn : 0 < n) : 
  a_seq n = (-1)^(n + 1) * n * (n + 1) :=
by
  sorry

end general_formula_l394_394455


namespace find_initial_amount_l394_394099

-- Let x be the initial amount Mark paid for the Magic card
variable {x : ℝ}

-- Condition 1: The card triples in value, resulting in 3x
-- Condition 2: Mark makes a profit of 200
def initial_amount (x : ℝ) : Prop := (3 * x - x = 200)

-- Theorem: Prove that the initial amount x equals 100 given the conditions
theorem find_initial_amount (h : initial_amount x) : x = 100 := by
  sorry

end find_initial_amount_l394_394099


namespace interior_angle_of_regular_hexagon_is_120_degrees_l394_394251

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l394_394251


namespace regular_hexagon_interior_angle_measure_l394_394236

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l394_394236


namespace last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l394_394289

noncomputable def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_1989_1989:
  last_digit (1989 ^ 1989) = 9 := 
sorry

theorem last_digit_1989_1992:
  last_digit (1989 ^ 1992) = 1 := 
sorry

theorem last_digit_1992_1989:
  last_digit (1992 ^ 1989) = 2 := 
sorry

theorem last_digit_1992_1992:
  last_digit (1992 ^ 1992) = 6 := 
sorry

end last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l394_394289


namespace number_of_people_purchased_only_book_A_l394_394620

def A_cap_B := 500
def B_only := 250
def A_total := 1500

theorem number_of_people_purchased_only_book_A : ∃ A_only : ℕ, A_only = A_total - A_cap_B ∧ A_only = 1000 :=
by
  use 1000
  unfold A_total A_cap_B
  split
  rfl
  rfl

end number_of_people_purchased_only_book_A_l394_394620


namespace coeff_x2_term_expansion_l394_394376

noncomputable def coeff_x2_in_expansion (f : ℚ[X]) : ℚ :=
  (f.coeff 2 : ℚ)

theorem coeff_x2_term_expansion :
  coeff_x2_in_expansion ((1 - polynomial.C (1 / polynomial.X)) * (1 + polynomial.X)^4) = 2 :=
by
  sorry

end coeff_x2_term_expansion_l394_394376


namespace problem_statement_l394_394835

open Finset

variable (E : Finset ℕ) (G : Finset ℕ)

theorem problem_statement (hE : E = Finset.range 200 \ {0})
  (hG : ∀ x ∈ G, x ∈ E) 
  (h_size : G.card = 100)
  (h_sum : ∑ i in G, i = 10080)
  (h_pair_sum : ∀ i j ∈ G, i ≠ j → i + j ≠ 201) :
  (∑ i in G, (i^2) = 2686700) ∧ (G.filter (λ x, odd x)).card % 4 = 0 := sorry

end problem_statement_l394_394835


namespace interior_angle_of_regular_hexagon_l394_394225

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l394_394225


namespace heat_dissipation_resistor_l394_394724

variables (R C ℰ r : ℝ)

def heat_released_in_resistor (R C ℰ r : ℝ) : ℝ :=
  (C * ℰ^2 * R) / (2 * (R + r))

theorem heat_dissipation_resistor :
  heat_released_in_resistor R C ℰ r = (C * ℰ^2 * R) / (2 * (R + r)) :=
by sorry

end heat_dissipation_resistor_l394_394724


namespace div_by_frac_eq_mult_multiply_12_by_4_equals_48_l394_394358

theorem div_by_frac_eq_mult (a b : ℝ) (hb : b ≠ 0) : a / (1 / b) = a * b :=
by
  sorry

theorem multiply_12_by_4_equals_48 : 12 * 4 = 48 :=
by
  have h : 12 / (1 / 4) = 12 * 4 := div_by_frac_eq_mult 12 4 (by norm_num)
  exact h.trans (by norm_num)

end div_by_frac_eq_mult_multiply_12_by_4_equals_48_l394_394358


namespace participant_advances_with_median_l394_394518

theorem participant_advances_with_median 
  (scores : List ℝ) (h_len : scores.length = 19) (h_unique : scores.nodup)
  (h_top9 : ∀ x ∈ scores.take 9, x ≥ scores.nth_le 9 sorry) :
  (∃ x ∈ scores, x > scores.nth_le 9 sorry) ↔ ∃ x ∈ scores.take 9, x 
by
  sorry

end participant_advances_with_median_l394_394518


namespace people_at_table_l394_394129

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l394_394129


namespace find_m_l394_394451

theorem find_m 
  (h : ∀ x, (0 < x ∧ x < 2) ↔ ( - (1 / 2) * x^2 + 2 * x > m * x )) :
  m = 1 :=
sorry

end find_m_l394_394451


namespace length_MN_l394_394645

noncomputable def calculate_MN 
  (K A B C D M N : Point)
  (AC BD a b : ℝ)
  (circle1 circle2 : Circle)
  (ext_tangent : Tangent circle1 circle2 K)
  (tangent1 : TangentLine circle1 circle2 A B)
  (tangent2 : TangentLine circle1 circle2 C D)
  (common_tangent : TangentLine circle1 circle2 M N)
  (AC_eq_a : AC = a)
  (BD_eq_b : BD = b)
  : ℝ :=
\begin
  have key1 : MN = (a + b) / 2,
  { rw [AC_eq_a, BD_eq_b], sorry },
  exact key1
\end

-- Theorem statement
theorem length_MN (K A B C D M N : Point) 
  (AC BD a b : ℝ)
  (circle1 circle2 : Circle)
  (ext_tangent : Tangent circle1 circle2 K)
  (tangent1 : TangentLine circle1 circle2 A B)
  (tangent2 : TangentLine circle1 circle2 C D)
  (common_tangent : TangentLine circle1 circle2 M N)
  (AC_eq_a : AC = a)
  (BD_eq_b : BD = b):
  calculate_MN K A B C D M N AC BD a b circle1 circle2 ext_tangent tangent1 tangent2 common_tangent AC_eq_a BD_eq_b = (a + b) / 2 := 
by sorry

end length_MN_l394_394645


namespace percentage_cd_only_l394_394112

noncomputable def percentage_power_windows : ℝ := 0.60
noncomputable def percentage_anti_lock_brakes : ℝ := 0.40
noncomputable def percentage_cd_player : ℝ := 0.75
noncomputable def percentage_gps_system : ℝ := 0.50
noncomputable def percentage_pw_and_abs : ℝ := 0.10
noncomputable def percentage_abs_and_cd : ℝ := 0.15
noncomputable def percentage_pw_and_cd : ℝ := 0.20
noncomputable def percentage_gps_and_abs : ℝ := 0.12
noncomputable def percentage_gps_and_cd : ℝ := 0.18
noncomputable def percentage_pw_and_gps : ℝ := 0.25

theorem percentage_cd_only : 
  percentage_cd_player - (percentage_abs_and_cd + percentage_pw_and_cd + percentage_gps_and_cd) = 0.22 := 
by
  sorry

end percentage_cd_only_l394_394112


namespace correct_product_of_a_and_b_l394_394059

-- Define reversal function for two-digit numbers
def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  units * 10 + tens

-- State the main problem
theorem correct_product_of_a_and_b (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 0 < b) 
  (h : (reverse_digits a) * b = 284) : a * b = 68 :=
sorry

end correct_product_of_a_and_b_l394_394059


namespace find_pq_l394_394885

theorem find_pq :
    ∃ (p q : ℕ), 
        (280 + q : ℚ) / (400 + p + q) = 4 / 7 ∧
        (p : ℚ) / (p + 120) = 3 / 5 ∧
        p = 180 ∧ 
        q = 120 :=
by
    sorry

end find_pq_l394_394885


namespace math_proof_problem_l394_394544

theorem math_proof_problem
  (n m k l : ℕ)
  (hpos_n : n > 0)
  (hpos_m : m > 0)
  (hpos_k : k > 0)
  (hpos_l : l > 0)
  (hneq_n : n ≠ 1)
  (hdiv : n^k + m*n^l + 1 ∣ n^(k+l) - 1) :
  (m = 1 ∧ l = 2*k) ∨ (l ∣ k ∧ m = (n^(k-l) - 1) / (n^l - 1)) :=
by 
  sorry

end math_proof_problem_l394_394544


namespace find_d_plus_f_l394_394740

variables (a b c d e f : ℂ)

-- Conditions given in the problem
def condition1 := b = 2
def condition2 := c = -a - 2 * e
def condition3 := a + b * complex.I + c + d * complex.I + (3 * e + f * complex.I) = 2 * complex.I

-- Theorem to prove
theorem find_d_plus_f : condition1 ∧ condition2 ∧ condition3 → d + f = 0 :=
by
  intro h,
  sorry

end find_d_plus_f_l394_394740


namespace fraction_of_students_l394_394878

theorem fraction_of_students {G B T : ℕ} (h1 : B = 2 * G) (h2 : T = G + B) (h3 : (1 / 2) * (G : ℝ) = (x : ℝ) * (T : ℝ)) : x = (1 / 6) :=
by sorry

end fraction_of_students_l394_394878


namespace min_value_of_expr_l394_394860

theorem min_value_of_expr (x : ℝ) (h : x > 2) : ∃ y, (y = x + 4 / (x - 2)) ∧ y ≥ 6 :=
by
  sorry

end min_value_of_expr_l394_394860


namespace inequality_proof_l394_394867

theorem inequality_proof (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = x^2 + 3 * x + 2) →
  a > 0 →
  b > 0 →
  b ≤ a / 7 →
  (∀ x : ℝ, |x + 2| < b → |f x + 4| < a) :=
by
  sorry

end inequality_proof_l394_394867


namespace prove_conditions_l394_394662

theorem prove_conditions :
  (a ∈ {a, b, c}) ∧ ({0, 1} ⊆ (set_of nat)) :=
by
  apply and.intro
  {
    exact set.mem_insert a {b, c}
  }
  {
    apply set.subset.trans (set.singleton_subset_iff.mpr (nat.cast_id ..))
    sorry -- Requires proof that {0, 1} ⊆ ℕ
  }

end prove_conditions_l394_394662


namespace obtuse_triangle_k_values_l394_394625

theorem obtuse_triangle_k_values (k : ℕ) (h : k > 0) :
  (∃ k, (5 < k ∧ k ≤ 12) ∨ (21 ≤ k ∧ k < 29)) → ∃ n : ℕ, n = 15 :=
by
  sorry

end obtuse_triangle_k_values_l394_394625


namespace common_tangent_exists_l394_394548

theorem common_tangent_exists:
  ∃ (a b c : ℕ), (a + b + c = 11) ∧
  ( ∀ (x y : ℝ),
      (y = x^2 + 12/5) ∧ 
      (x = y^2 + 99/10) ∧ 
      (a*x + b*y = c) ∧ 
      0 < a ∧ 0 < b ∧ 0 < c ∧ 
      Int.gcd (Int.gcd a b) c = 1
  ) := 
by
  sorry

end common_tangent_exists_l394_394548


namespace div120_l394_394115

theorem div120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
sorry

end div120_l394_394115


namespace product_of_solutions_eq_zero_l394_394194

theorem product_of_solutions_eq_zero :
  (∃ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) →
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → (x = 0 ∨ x = -4/7)) →
  (0 * (-4/7) = 0) :=
by
  sorry

end product_of_solutions_eq_zero_l394_394194


namespace dietitian_calorie_intake_l394_394692

variable (food_calories : ℕ) (fraction_eaten : ℚ) (recommended_calories : ℕ)

-- Conditions
def conditions := food_calories = 40 ∧ fraction_eaten = 3/4 ∧ recommended_calories = 25

-- Derived result
def result := (fraction_eaten * food_calories : ℚ) - recommended_calories = 5

-- The statement to prove
theorem dietitian_calorie_intake (h : conditions food_calories fraction_eaten recommended_calories) :
  result food_calories fraction_eaten recommended_calories :=
begin
  sorry
end

end dietitian_calorie_intake_l394_394692


namespace intersection_M_N_l394_394456

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | (x ∈ U) ∧ ¬(x ∈ complement_U_N)}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
sorry

end intersection_M_N_l394_394456


namespace min_sum_is_11_over_28_l394_394922

-- Definition of the problem
def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the minimum sum problem
def min_sum (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A ∈ digits ∧ B ∈ digits ∧ C ∈ digits ∧ D ∈ digits →
  ((A : ℚ) / B + (C : ℚ) / D) = (11 : ℚ) / 28

-- The theorem statement
theorem min_sum_is_11_over_28 :
  ∃ A B C D : ℕ, min_sum A B C D :=
sorry

end min_sum_is_11_over_28_l394_394922


namespace number_of_sets_leq_bound_l394_394779

theorem number_of_sets_leq_bound (n : ℕ) (x : Fin n → ℝ) (λ : ℝ) (h1 : 2 ≤ n) (h2 : (∑ i, x i) = 0) (h3 : (∑ i, (x i)^2) = 1) (hλ : 0 < λ):
  {A : Finset (Fin n) | ∑ i in A, x i ≥ λ}.card ≤ 2^(n-3) / λ^2 :=
sorry

end number_of_sets_leq_bound_l394_394779


namespace maximal_k_value_l394_394377

theorem maximal_k_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∃ k : ℝ, (∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 → (k + x/y) * (k + y/z) * (k + z/x) ≤ (x/y + y/z + z/x) * (y/x + z/y + x/z)) 
  ∧ k = real.cbrt 9 - 1) :=
by {
  use real.cbrt 9 - 1,
  intros x y z hx hy hz,
  sorry -- Proof would go here.
}

end maximal_k_value_l394_394377


namespace electricity_usage_in_september_l394_394169

noncomputable def SunnyElementarySchoolUsageInSeptember : ℕ :=
  let OctoberUsage : ℕ := 1400
  let SavingsFraction : ℚ := 0.7
  let SeptemberUsage : ℚ := OctoberUsage / SavingsFraction
  SeptemberUsage.toNat

theorem electricity_usage_in_september :
  let OctoberUsage : ℕ := 1400
  let SavingsFraction : ℚ := 0.7
  ∀ S : ℚ, (SavingsFraction * S = OctoberUsage) → S = 2000 :=
by
  assume OctoberUsage SavingsFraction S h
  -- Skipping the proof
  sorry

end electricity_usage_in_september_l394_394169


namespace degree_measure_of_regular_hexagon_interior_angle_l394_394284

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l394_394284


namespace log_satisfies_condition_l394_394372

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem log_satisfies_condition :
  ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → f(x₁ * x₂) = f(x₁) + f(x₂) := 
  by
  intros x₁ x₂ hx₁ hx₂
  sorry

end log_satisfies_condition_l394_394372


namespace Billy_age_l394_394729

-- Defining the ages of Billy, Joe, and Sam
variable (B J S : ℕ)

-- Conditions given in the problem
axiom Billy_twice_Joe : B = 2 * J
axiom sum_BJ_three_times_S : B + J = 3 * S
axiom Sam_age : S = 27

-- Statement to prove
theorem Billy_age : B = 54 :=
by
  sorry

end Billy_age_l394_394729


namespace correct_total_distance_l394_394387

noncomputable def total_distance_traveled (radius : ℝ) (n : ℕ) : ℝ :=
  let adjacent_dist := radius
  let non_adjacent_dist := radius * real.sqrt 2
  let individual_distance := 3 * non_adjacent_dist + 2 * adjacent_dist
  n * individual_distance

theorem correct_total_distance :
  total_distance_traveled 50 8 = 1200 * real.sqrt 2 + 800 :=
by
  -- The proof is omitted.
  sorry

end correct_total_distance_l394_394387


namespace total_length_of_figure_2_segments_l394_394717

-- Definitions based on conditions
def rectangle_length : ℕ := 10
def rectangle_breadth : ℕ := 6
def square_side : ℕ := 4
def interior_segment : ℕ := rectangle_breadth / 2

-- Summing up the lengths of segments in Figure 2
def total_length_of_segments : ℕ :=
  square_side + 2 * rectangle_length + interior_segment

-- Mathematical proof problem statement
theorem total_length_of_figure_2_segments :
  total_length_of_segments = 27 :=
sorry

end total_length_of_figure_2_segments_l394_394717


namespace relay_race_time_l394_394965

-- Define the time it takes for each runner.
def Rhonda_time : ℕ := 24
def Sally_time : ℕ := Rhonda_time + 2
def Diane_time : ℕ := Rhonda_time - 3

-- Define the total time for the relay race.
def total_relay_time : ℕ := Rhonda_time + Sally_time + Diane_time

-- State the theorem we want to prove: the total relay time is 71 seconds.
theorem relay_race_time : total_relay_time = 71 := 
by 
  -- The following "sorry" indicates a step where the proof would be completed.
  sorry

end relay_race_time_l394_394965


namespace tangent_lines_coincide_l394_394436

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f x / x

theorem tangent_lines_coincide : 
  f(2) = 2 ∧ 
  (∀ x, g x = f x / x) ∧ 
  (2 * deriv f 2 - f 2) / 4 = 1/2 → 
  deriv f 2 = 2 :=
by
  intros h,
  cases' h with h1 h2,
  cases' h2 with h2 h3,
  have : f 2 = 2 := h1,
  have : (2 * deriv f 2 - f 2) / 4 = 1 / 2 := h3,
  sorry

end tangent_lines_coincide_l394_394436


namespace integral_equation_solution_l394_394391

noncomputable def varphi (x : ℝ) : ℝ := 1

theorem integral_equation_solution :
  ∀ x : ℝ, varphi(x) + ∫ t in 0..1, x * (exp(x * t) - 1) * varphi(t) = exp(x) - x :=
by
  intro x
  -- Definitions and conditions can be stated here as needed
  have h : ∀ t : ℝ, varphi(t) = 1, from fun t => rfl,
  rw [integral_congr h],
  sorry

end integral_equation_solution_l394_394391


namespace sand_weight_proof_l394_394695

-- Definitions for the given conditions
def side_length : ℕ := 40
def bag_weight : ℕ := 30
def area_per_bag : ℕ := 80

-- Total area of the sandbox
def total_area := side_length * side_length

-- Number of bags needed
def number_of_bags := total_area / area_per_bag

-- Total weight of sand needed
def total_weight := number_of_bags * bag_weight

-- The proof statement
theorem sand_weight_proof :
  total_weight = 600 :=
by
  sorry

end sand_weight_proof_l394_394695


namespace distinct_numbers_in_list_l394_394400

theorem distinct_numbers_in_list : 
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 1000) → 
  (finset.card (finset.image (λ n, int.floor (n^2 / 2000 : ℚ)) (finset.range 1000)) = 501) :=
by
  assume n hn,
  contradiction


end distinct_numbers_in_list_l394_394400


namespace table_seating_problem_l394_394138

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l394_394138


namespace sequence_sum_l394_394360

theorem sequence_sum : 
  ∑ n in Finset.range' 3 12, n * (1 - 1 / n) = 65 := 
by
  sorry

end sequence_sum_l394_394360


namespace sum_of_coordinates_l394_394997

-- Given a function f, Point (2, 3) is on the graph of y = f(x)/2
-- Proving the sum of the coordinates of the unique point on the graph of y = f⁻¹(x)/2 equals 6.5

noncomputable def f (x : ℝ) : ℝ := sorry  -- Define the function f (details unknown)

noncomputable def g (x : ℝ) : ℝ := sorry  -- Define the inverse function f⁻¹ (details unknown)

theorem sum_of_coordinates (h : 3 = f 2 / 2) : 
  let x := 6 in 
  let y := 0.5 in 
  x + y = 6.5 :=
by
  let f_inv := g
  let y_point := f_inv 6 / 2
  let sum_coords := 6 + 0.5
  exact sorry

end sum_of_coordinates_l394_394997


namespace table_seating_problem_l394_394140

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l394_394140


namespace result_l394_394935

noncomputable def alpha (n : ℕ) (h : 0 < n) : ℝ :=
  classical.some (exists_real_root (n * x ^ 3 + 2 * x - n))

noncomputable def beta (n : ℕ) (h : 2 ≤ n) : ℕ :=
  ⌊ (n+1) * alpha n (lt_of_le_of_lt (zero_le_one) h) ⌋

theorem result : (1 / 1006 : ℝ) * (∑ k in finset.range 2012, β (k + 2) (nat.le_add_left 2 k)) = 2015 := sorry

end result_l394_394935


namespace length_OR_coordinates_Q_area_OPQR_8_p_value_l394_394184

noncomputable def point_R : (ℝ × ℝ) := (0, 4)

noncomputable def OR_distance : ℝ := 0 - 4 -- the vertical distance from O to R

theorem length_OR : OR_distance = 4 := sorry

noncomputable def point_Q (p : ℝ) : (ℝ × ℝ) := (p, 2 * p + 4)

theorem coordinates_Q (p : ℝ) : point_Q p = (p, 2 * p + 4) := sorry

noncomputable def area_OPQR (p : ℝ) : ℝ := 
  let OR : ℝ := 4
  let PQ : ℝ := 2 * p + 4
  let OP : ℝ := p
  1 / 2 * (OR + PQ) * OP

theorem area_OPQR_8 : area_OPQR 8 = 96 := sorry

theorem p_value (h : area_OPQR p = 77) : p = 7 := sorry

end length_OR_coordinates_Q_area_OPQR_8_p_value_l394_394184


namespace PA_distance_l394_394558

noncomputable section

def curve (x y : ℝ) : Prop := 2 * x = sqrt (4 + y^2)

def A : ℝ × ℝ := (-sqrt 5, 0)
def B : ℝ × ℝ := (sqrt 5, 0)

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def isOnCurve (P : ℝ × ℝ) : Prop :=
  curve P.1 P.2

def pointB : ℝ × ℝ := (sqrt 5, 0)
def pointA : ℝ × ℝ := (-sqrt 5, 0)

def PB_distance (P : ℝ × ℝ) : Prop :=
  distance P pointB = 2

theorem PA_distance (P : ℝ × ℝ) (h1 : isOnCurve P) (h2 : PB_distance P) : distance P pointA = 4 :=
sorry

end PA_distance_l394_394558


namespace vector_c_expression_l394_394460

-- Define the vectors a, b, c
def vector_a : ℤ × ℤ := (1, 2)
def vector_b : ℤ × ℤ := (-1, 1)
def vector_c : ℤ × ℤ := (1, 5)

-- Define the addition of vectors in ℤ × ℤ
def vec_add (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the scalar multiplication of vectors in ℤ × ℤ
def scalar_mul (k : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (k * v.1, k * v.2)

-- Given the conditions
def condition1 := vector_a = (1, 2)
def condition2 := vec_add vector_a vector_b = (0, 3)

-- The goal is to prove that vector_c = 2 * vector_a + vector_b
theorem vector_c_expression : vec_add (scalar_mul 2 vector_a) vector_b = vector_c := by
  sorry

end vector_c_expression_l394_394460


namespace sum_gcd_lcm_eq_4851_l394_394297

theorem sum_gcd_lcm_eq_4851 (a b : ℕ) (ha : a = 231) (hb : b = 4620) :
  Nat.gcd a b + Nat.lcm a b = 4851 :=
by
  rw [ha, hb]
  sorry

end sum_gcd_lcm_eq_4851_l394_394297


namespace B_catches_up_with_A_l394_394336

-- Definitions based on conditions:
def speedA : ℝ := 10 -- A's speed in kmph
def timeBeforeB : ℝ := 4 -- Time before B starts in hours
def speedB : ℝ := 20 -- B's speed in kmph

-- Proven that the catch-up distance from the start point:
theorem B_catches_up_with_A : ∀ (distanceA_travelled timeToCatchUp distanceFromStart : ℝ),
  -- Distance A has traveled before B starts cycling
  distanceA_travelled = speedA * timeBeforeB →
  -- Relative speed of B with respect to A
  let relative_speed := speedB - speedA in
  -- Time for B to catch up with A
  timeToCatchUp = distanceA_travelled / relative_speed →
  -- Distance from the start where B catches A
  distanceFromStart = speedB * timeToCatchUp →
  distanceFromStart = 80 :=
  by
    intros distanceA_travelled timeToCatchUp distanceFromStart h1 h2 h3
    rw [h1, h2] 
    sorry

end B_catches_up_with_A_l394_394336


namespace max_ab_bc_cd_da_l394_394828

theorem max_ab_bc_cd_da (a b c d : ℕ)
  (h_perms : List.perm [a, b, c, d] [1, 2, 4, 5]) :
  ab + bc + cd + da ≤ 36 :=
by
  -- Proof uses conditions only
  sorry

end max_ab_bc_cd_da_l394_394828


namespace log_sum_of_sequence_l394_394524

theorem log_sum_of_sequence {a : ℕ → ℝ} (h1 : ∀ n, a n > 0)
  (hg : ∃ r a₁, ∀ n, a (n + 1) = a n * r) 
  (h2 : a 8 * a 10 = 4) : 
  ∑ n in finset.range 19, real.logb (1/2) (a n) = -19 := 
sorry

end log_sum_of_sequence_l394_394524


namespace cosine_identity_l394_394467

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) 

theorem cosine_identity 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (hc : ∥c∥ = real.sqrt 2)
  (habc : a + b + c = 0) :
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
sorry

end cosine_identity_l394_394467


namespace sin_sum_bound_l394_394116

theorem sin_sum_bound (x y z : ℝ) (h : x + y + z = 0) : 
  sin x + sin y + sin z ≤ 3 * Real.sqrt 3 / 2 := 
sorry

end sin_sum_bound_l394_394116


namespace roots_difference_l394_394353

theorem roots_difference :
  let a := 2 
  let b := 5 
  let c := -12
  let disc := b*b - 4*a*c
  let root1 := (-b + Real.sqrt disc) / (2 * a)
  let root2 := (-b - Real.sqrt disc) / (2 * a)
  let larger_root := max root1 root2
  let smaller_root := min root1 root2
  larger_root - smaller_root = 5.5 := by
  sorry

end roots_difference_l394_394353


namespace general_formula_lambda_range_l394_394925

-- Given Sequence Definitions and Conditions
def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^n
def sum_first_n_terms (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * (1 - q^(n+1)) / (1 - q)
def arithmetic_sequence (b c d : ℝ) := (b + d) / 2 = c
def b_n (a₃ₙ₊₁ : ℕ → ℝ) (n : ℕ) : ℝ := real.log (a₃ₙ₊₁ n)
def T_n (b : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, b i

-- Conditions
axiom condition1 : ∀ (a₁ q : ℝ), q > 1 → sum_first_n_terms a₁ q 2 = 7
axiom condition2 : ∀ (a₁ a₂ a₃ : ℝ), arithmetic_sequence (a₁ + 3) (3 * a₂) (a₃ + 4)
axiom condition3 : ∀ (a₃ₙ₊₁ : ℕ → ℝ), ∀ n, a₃ₙ₊₁ (3 * n + 1) = 2^(3*n)
axiom condition4 : ∀ (Tₙ : ℕ → ℝ) (n : ℕ), (∑ i in finset.range n, 1 / Tₙ i) < λ

-- Proof tasks
theorem general_formula (q : ℝ) (n : ℕ) : q > 1 → ∀ (a₁ a₂ a₃ : ℝ), 
  condition1 a₁ q 2 →
  condition2 a₁ a₂ a₃ →
  a₂ = 2 / q → 
  a₁ = 2^(-1) → 
  ∀ n, geometric_sequence (2^(-1)) q n = 2^(n-1) := sorry

theorem lambda_range (λ : ℝ) : ∀ Tₙ, 
  (∀ n, Tₙ n = 3 * n * (n + 1) / 2 * real.log 2) →
  (λ ≥ (2 / (3 * real.log 2))) := sorry

end general_formula_lambda_range_l394_394925


namespace equal_height_division_of_hemisphere_l394_394319

noncomputable def height_dividing_hemisphere_volume : ℝ :=
let r := 1 in
let V_hemisphere := (2 / 3) * Real.pi * r^3 in
let V_cap (x : ℝ) := (Real.pi * x^2 / 3) * (3 * r - x) in
have h1 : V_hemisphere = (4 / 3) * Real.pi / 2, from by sorry,
have h2 : ∃ x, V_cap x = V_hemisphere / 2, from by sorry,
classical.some h2

theorem equal_height_division_of_hemisphere : height_dividing_hemisphere_volume = 0.6527 := 
by
  sorry

end equal_height_division_of_hemisphere_l394_394319


namespace alice_correct_percentage_l394_394363

/-
  Carlos and Alice are students in Mr. Alan's math class. Last evening, they both addressed half
  of the problems in their homework individually and then attempted the remaining half collectively.
  Carlos successfully solved 70% of the problems he tackled alone, and also achieved an overall
  performance of 82% correctness in his answers. Alice solved 85% of her individual problems correctly.
  During their group work, it was noted that 80% of those problems were solved correctly. 

  We aim to show Alice's overall percentage of correct answers is 82.5%.
-/

variables (t : ℕ) -- total number of problems in the assignment
variables (x : ℕ) -- number of problems solved correctly together

-- Define the conditions
def carlos_correct_alone : ℕ := 0.70 * (t / 2)
def carlos_correct_total : ℕ := 0.82 * t
def group_correct : ℕ := 0.80 * (t / 2)
def alice_correct_alone : ℕ := 0.85 * (t / 2)

-- Prove Alice's overall correct answer percentage is 82.5%
theorem alice_correct_percentage (t : ℕ) :
  (alice_correct_alone t + group_correct t) / t * 100 = 82.5 := 
sorry

end alice_correct_percentage_l394_394363


namespace arithmetic_sequence_sum_positive_l394_394429

-- We define f to be a monotonically increasing and odd function.
def isMonotonicIncreasing (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x ≤ y → f x ≤ f y
def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- We define the arithmetic sequence {a_n} with the first term a1 and common difference d.
def isArithmeticSequence (a : ℕ → ℝ) : Prop := ∃ a1 d : ℝ, a 1 = a1 ∧ a (n : ℕ) = a1 + d * n

-- The main theorem statement.
theorem arithmetic_sequence_sum_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h1 : isMonotonicIncreasing f)
  (h2 : isOddFunction f)
  (h3 : isArithmeticSequence a)
  (h4 : a 1 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 := sorry

end arithmetic_sequence_sum_positive_l394_394429


namespace C_plus_D_l394_394751

theorem C_plus_D (D C : ℚ) (h1 : ∀ x : ℚ, (Dx - 17) / ((x - 2) * (x - 4)) = C / (x - 2) + 4 / (x - 4))
  (h2 : ∀ x : ℚ, (x - 2) * (x - 4) = x^2 - 6 * x + 8) :
  C + D = 8.5 := sorry

end C_plus_D_l394_394751


namespace range_of_omega_l394_394573

noncomputable def is_periodic_fn (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem range_of_omega 
  {ω : ℝ} 
  (h1 : 0 < ω)
  (f : ℝ → ℝ := λ x, sqrt 3 * sin (ω * x) + cos (ω * x)) 
  (period : ℝ := 2 * π / ω)
  (h2 : ∃ x, (π/6 < x ∧ x < π/3) ∧ (∃ k : ℤ, x = π/(3 * ω) + k * π / ω))
  (h3 : π < period) : 
  (1 < ω) ∧ (ω < 2) :=
sorry

end range_of_omega_l394_394573


namespace balloon_rearrangements_l394_394842

-- Define the letters involved: vowels and consonants
def vowels := ['A', 'O', 'O', 'O']
def consonants := ['B', 'L', 'L', 'N']

-- State the problem in Lean 4:
theorem balloon_rearrangements : 
  ∃ n : ℕ, 
  (∀ (vowels := ['A', 'O', 'O', 'O']) 
     (consonants := ['B', 'L', 'L', 'N']), 
     n = 32) := sorry  -- we state that the number of rearrangements is 32 but do not provide the proof itself.

end balloon_rearrangements_l394_394842


namespace john_saves_7680_per_year_l394_394542

-- Define old rent cost
def oldRent := 1200

-- Define the increase percentage for new apartment costs
def increasePercent := 0.40

-- Define the new rent cost
def newRent := oldRent + oldRent * increasePercent

-- Define the number of people sharing the cost
def numPeople := 3

-- Define John's share of the new apartment cost
def johnsShare := newRent / numPeople

-- Define monthly savings
def monthlySavings := oldRent - johnsShare

-- Define yearly savings
def yearlySavings := monthlySavings * 12

-- Theorem statement
theorem john_saves_7680_per_year : yearlySavings = 7680 := 
by {
  -- Placeholder for actual proof,
  -- but statement builds successfully.
  sorry
}

end john_saves_7680_per_year_l394_394542


namespace prod_of_real_roots_equation_l394_394193

theorem prod_of_real_roots_equation :
  (∀ x : ℝ, (x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4)) → x = 0 ∨ x = -(4 / 7) → (0 * (-(4 / 7)) = 0) :=
by sorry

end prod_of_real_roots_equation_l394_394193


namespace regular_hexagon_interior_angle_l394_394264

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l394_394264


namespace satisfy_equation_l394_394762

open Int

theorem satisfy_equation (k : ℕ) : 
  let x := k^2 - 2,
      y1 := k * (k^2 - 3),
      y2 := -k * (k^2 - 3)
  in y1^2 = x^3 - 3*x + 2 ∧ y2^2 = x^3 - 3*x + 2 :=
by 
  let x := k ^ 2 - 2
  let y1 := k * (k ^ 2 - 3)
  let y2 := -k * (k ^ 2 - 3)
  show y1 ^ 2 = x ^ 3 - 3 * x + 2 ∧ y2 ^ 2 = x ^ 3 - 3 * x + 2
  sorry

end satisfy_equation_l394_394762


namespace max_value_of_x_l394_394928

theorem max_value_of_x (x y z : ℝ) (h1 : x + y + z = 9) (h2 : xy + xz + yz = 20) : x ≤ (18 + real.sqrt 312) / 6 ∧ (18 - real.sqrt 312) / 6 ≤ x :=
sorry

end max_value_of_x_l394_394928


namespace area_BCD_is_correct_l394_394066

def triangle_area (b h : ℕ) : ℚ := (b * h) / 2

variables (A B C D : Type)
variables [Real.metric_space A] [Real.metric_space B] [Real.metric_space C] [Real.metric_space D]
variables (BC : ℕ) (CD : ℕ) (area_ABC : ℕ) (height_B_to_AC : ℚ)

-- Given values from the problem
axiom BC_equal_7 : BC = 7
axiom CD_equal_30 : CD = 30
axiom area_ABC_equal_36 : area_ABC = 36
axiom height_B_to_AC_calculated : height_B_to_AC = 72 / 7

-- The statement to be proven
theorem area_BCD_is_correct:
  triangle_area CD height_B_to_AC = 1080 / 7 :=
  sorry

end area_BCD_is_correct_l394_394066


namespace seated_people_count_l394_394134

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l394_394134


namespace regular_hexagon_interior_angle_measure_l394_394238

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l394_394238


namespace expected_value_binomial_l394_394432

theorem expected_value_binomial :
  (∃ ξ : ℕ → ℝ, (∀ n p. ξ n p = n * p) → ξ 6 (1/3) = 2) :=
by
  sorry

end expected_value_binomial_l394_394432


namespace relationship_abc_l394_394813

-- Definitions/Assumptions
def f : ℝ → ℝ := sorry  -- placeholder for the function f(x)

axiom even_function (x : ℝ) : f(x + 2) = f(-(x + 2))
axiom decreasing_on_interval {x1 x2 : ℝ} (h1 : x1 ∈ set.Ici 2) (h2 : x2 ∈ set.Ici 2) : 
  (f x1 - f x2) * (x1 - x2) < 0

-- Definitions for specific variables a, b, c
def a := f 1
def b := f (5 / 2)
def c := f (-1 / 2)

-- Prove the relationship between a, b, and c
theorem relationship_abc : c < a ∧ a < b :=
by sorry

end relationship_abc_l394_394813


namespace binomial_problem_l394_394861

-- Conditions
def is_real (x : ℝ) : Prop := True
def is_nonnegative_integer (k : ℕ) : Prop := True

-- Definition of binomial coefficient
noncomputable def binomial_coeff (x : ℝ) (k : ℕ) : ℝ :=
  x * (x - 1) * (x - 2) * ... * (x - k + 1) / k.factorial

-- Given problem statement
theorem binomial_problem (x : ℝ) (k : ℕ) (h1 : is_real x) (h2 : is_nonnegative_integer k) :
  (binomial_coeff (-1/2) 1007 * 3^1007) / binomial_coeff 2014 1007 = - (3^1007) / (2^2015) :=
sorry

end binomial_problem_l394_394861


namespace number_of_new_students_l394_394884

theorem number_of_new_students (initial_students left_students final_students new_students : ℕ) 
  (h_initial : initial_students = 4) 
  (h_left : left_students = 3) 
  (h_final : final_students = 43) : 
  new_students = final_students - (initial_students - left_students) :=
by 
  sorry

end number_of_new_students_l394_394884


namespace variance_scaled_xi_l394_394311

-- Definition of a normal distribution for realism, but simplified for our use case
def normal_dist (mean variance : ℝ) := { x : ℝ // x = mean + variance }

-- Given condition
def xi : normal_dist 2 (2^2) := ⟨2 + 2^2⟩

-- Proof problem statement
theorem variance_scaled_xi : 
  D((1/4) * xi) = 1/4 := by
  sorry

end variance_scaled_xi_l394_394311


namespace sequences_are_infinitesimal_l394_394962

noncomputable theory

open Filter

-- Define sequences
def x₁ (n : ℕ) : ℝ := -1 / n
def x₂ (n : ℕ) : ℝ := (-1)^(n-1) / n
def β (n : ℕ) : ℝ := 1 / (2 * n - 1)

-- Infinitesimal sequences definition
def is_infinitesimal (a : ℕ → ℝ) := ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n| < ε

-- Lean statement for the problem
theorem sequences_are_infinitesimal :
  is_infinitesimal x₁ ∧ is_infinitesimal x₂ ∧ is_infinitesimal β :=
begin
  sorry
end

end sequences_are_infinitesimal_l394_394962


namespace probability_lfloor_log2_eq_zero_l394_394557

noncomputable def probability_floor_log2_eq (x : ℝ) : ℝ :=
if hx : 0 < x ∧ x < 1 then
  let log3x := Real.log 3 * Real.log x / Real.log 2 in
  let logx := Real.log x / Real.log 2 in
  if Real.floor (log3x) = Real.floor (logx) then 1 else 0
else 0

theorem probability_lfloor_log2_eq_zero :
  (interval_integral (λ x, probability_floor_log2_eq x) 0 1) = 5 / 7 :=
sorry

end probability_lfloor_log2_eq_zero_l394_394557


namespace regular_hexagon_interior_angle_measure_l394_394231

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l394_394231


namespace smallest_x_is_odd_l394_394545

theorem smallest_x_is_odd :
  let x := Nat.find (λ x : ℕ, ∀ a b c d : ℕ, (2^a - 2^b) % (2^c - 2^d) ≠ x) in
  x % 2 = 1 :=
by
  sorry

end smallest_x_is_odd_l394_394545


namespace slope_line3_l394_394941

open Real Nat

def line1 (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def point_A : ℝ × ℝ := (3, 0)
def line2 (y : ℝ) : Prop := y = 2
def area_ABC (a b c : ℝ × ℝ) : ℝ := 1/2 * (abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2)))
def point_B : ℝ × ℝ := (0, 2)
def point_C : ℝ × ℝ := (9, 2)

theorem slope_line3 : ∀ (m : ℝ),
  line1 point_A.1 point_A.2 →
  line2 point_B.2 →
  area_ABC point_A point_B point_C = 6 →
  line2 point_C.2 →
  point_C.1 = 9 →
  m = (point_C.2 - point_A.2) / (point_C.1 - point_A.1) :=
by
  intros m line1_A line2_B area_eq line2_C pointC12
  have slope := (point_C.2 - point_A.2) / (point_C.1 - point_A.1)
  sorry

end slope_line3_l394_394941


namespace negation_of_existence_l394_394186

theorem negation_of_existence (h : ¬ (∃ x : ℝ, x^2 - x - 1 > 0)) : ∀ x : ℝ, x^2 - x - 1 ≤ 0 :=
sorry

end negation_of_existence_l394_394186


namespace point_on_graph_and_sum_of_coordinates_l394_394431

noncomputable def f : ℝ → ℝ := sorry

theorem point_on_graph_and_sum_of_coordinates :
  (f 5 = 9) →
  (∃ y : ℝ, (3 * y = 4 * f (5 * 1) + 6) ∧ y = 14) ∧ (1 + 14 = 15) :=
begin
  intros h,
  use 14,
  split,
  { calc 3 * 14 = 42 : by norm_num
           ... = 4 * f (5 * 1) + 6 : by { rw h, norm_num } },
  { norm_num }
end

end point_on_graph_and_sum_of_coordinates_l394_394431


namespace factorial_divisibility_l394_394936

theorem factorial_divisibility (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (a! + b!) ∣ (a! * b!)) : 3 * a ≥ 2 * b + 2 :=
by
  sorry

end factorial_divisibility_l394_394936


namespace sum_q_t_at_12_l394_394550

-- Define the set T and polynomial q_t as per conditions
def T : Set (Fin 11 → Fin 2) :=
  {t | ∀ (n : Fin 11), t n = 0 ∨ t n = 1}

def q_t (t : Fin 11 → Fin 2) : Polynomial ℝ :=
  Polynomial.interp (λ n, if t n = 1 then 1 else 0) -- Interpolating a polynomial q_t from the tuple t

-- Define the polynomial q as the sum of all q_t
def q : Polynomial ℝ :=
  ∑ t in T, q_t t

-- The final theorem stating the result
theorem sum_q_t_at_12 : q.eval 12 = 2048 :=
  by
    sorry

end sum_q_t_at_12_l394_394550


namespace z_square_is_pure_imaginary_l394_394009

noncomputable def z : ℂ := 2 / (1 + complex.I)

theorem z_square_is_pure_imaginary : ∃ (a : ℝ), z^2 = 0 + (a * complex.I) :=
by
  sorry

end z_square_is_pure_imaginary_l394_394009


namespace basketball_surface_area_l394_394677

theorem basketball_surface_area (C : ℝ) (r : ℝ) (A : ℝ) (π : ℝ) 
  (h1 : C = 30) 
  (h2 : C = 2 * π * r) 
  (h3 : A = 4 * π * r^2) 
  : A = 900 / π := by
  sorry

end basketball_surface_area_l394_394677


namespace bridge_length_correct_l394_394182

def length_of_bridge (L_T : ℕ) (S_T : ℝ) (T : ℝ) : ℝ :=
  let S_T_m_s := S_T * (1000 / 3600)
  let total_distance := S_T_m_s * T
  total_distance - L_T

theorem bridge_length_correct :
  length_of_bridge 160 45 30 = 215 :=
by
  sorry

end bridge_length_correct_l394_394182


namespace find_angle_C_l394_394043

open Real

theorem find_angle_C (a b C A B : ℝ) 
  (h1 : a^2 + b^2 = 6 * a * b * cos C)
  (h2 : sin C ^ 2 = 2 * sin A * sin B) :
  C = π / 3 := 
  sorry

end find_angle_C_l394_394043


namespace seashells_total_l394_394598

def seashells :=
  let sam_seashells := 18
  let mary_seashells := 47
  sam_seashells + mary_seashells

theorem seashells_total : seashells = 65 := by
  sorry

end seashells_total_l394_394598


namespace degree_measure_of_regular_hexagon_interior_angle_l394_394286

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l394_394286


namespace dietitian_calorie_excess_l394_394693

theorem dietitian_calorie_excess
  (lunch_fraction : ℚ)
  (total_calories : ℚ)
  (recommended_calories : ℚ)
  (lunch_fraction_eq : lunch_fraction = 3 / 4)
  (total_calories_eq : total_calories = 40)
  (recommended_calories_eq : recommended_calories = 25)
  : (lunch_fraction * total_calories - recommended_calories) = 5 :=
by
  rw [lunch_fraction_eq, total_calories_eq, recommended_calories_eq]
  norm_num
  sorry

end dietitian_calorie_excess_l394_394693


namespace num_sequences_length_20_l394_394367

noncomputable def a : ℕ → ℕ 
| 1     := 0
| 2     := 1
| n + 1 := if (n + 1) % 2 = 1 then a (n - 1) + b (n - 1) else a (n - 1) + b (n - 1)

noncomputable def b : ℕ → ℕ 
| 1     := 1
| 2     := 0
| n + 1 := if (n + 1) % 2 = 1 then a n + b (n - 1) else a n + b (n - 1)

theorem num_sequences_length_20 : (a 20) + b 20 = 1874 :=
sorry

end num_sequences_length_20_l394_394367


namespace interior_angle_regular_hexagon_l394_394241

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l394_394241


namespace correct_fraction_simplification_l394_394854

theorem correct_fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (∀ (c d : ℝ), (c ≠ d) → (a+2 = c → b+2 = d → (a+2)/d ≠ a/b))
  ∧ (∀ (e f : ℝ), (e ≠ f) → (a-2 = e → b-2 = f → (a-2)/f ≠ a/b))
  ∧ (∀ (g h : ℝ), (g ≠ h) → (a^2 = g → b^2 = h → a^2/h ≠ a/b))
  ∧ (a / b = ( (1/2)*a / (1/2)*b )) := 
sorry

end correct_fraction_simplification_l394_394854


namespace calculate_PS_l394_394641

noncomputable def Triangle (A B C : Type*) := A ≠ B ∧ B ≠ C ∧ A ≠ C

def side_lengths (M N P : ℝ) : Triangle ℝ :=
  (M ≠ N) ∧ (N ≠ P) ∧ (M ≠ P)

def angle_bisector_theorem (MN NP MP : ℝ) (NQ QP : ℝ) : Prop :=
  NQ / QP = NP / MP ∧ NQ + QP = NP

theorem calculate_PS (M N P Q R S : ℝ)
  (h_triang: side_lengths 13 26 24)
  (h_bis : angle_bisector_theorem 13 26 24 (26 * (13 / 25)) (24 * (13 / 25)))
  (h_cos : cos (angle M N P) = 361 / 416)
  (h_sin : sin (angle M N P) = 55 / 416)
  (h_in_circumcircle : S ≠ M ∧ S ∈ circumcircle S M R Q) :
  distance P S = sqrt 455 :=
sorry

end calculate_PS_l394_394641


namespace correct_propositions_l394_394661

theorem correct_propositions :
  (∀ (A : set ℕ), Pr(A) = 1) ∧
  (∀ (A : set ℕ), Pr(A) ≠ 1.1) ∧
  (¬ (∀ (A B : set ℕ), mutually_exclusive(A, B) → complementary(A, B))) ∧
  (∀ (A B : set ℕ), complementary(A, B) → mutually_exclusive(A, B)) ∧
  (is_classical_probability_model(roll_die_observe_number)) :=
sorry

end correct_propositions_l394_394661


namespace find_a_l394_394010

noncomputable theory

open Real

theorem find_a : ∃ a : ℝ, 
  (∀ x : ℝ, f x = a * (x + 1) - exp x) ∧
  f' 0 = -2 → a = -1 :=
by
  sorry

def f (a : ℝ) (x : ℝ) : ℝ := a * (x + 1) - exp x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := (deriv (f a)) x

end find_a_l394_394010


namespace no_positive_integer_solutions_l394_394621

theorem no_positive_integer_solutions : ¬∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ^ 4004 + y ^ 4004 = z ^ 2002 :=
by
  sorry

end no_positive_integer_solutions_l394_394621


namespace number_of_algorithms_is_three_l394_394188

/-- 
  This code aims to prove that the number of algorithms from the given statements is 3
  given the conditions.
-/

def statement1 : Prop :=
  ∃ (step1 step2 : String), step1 = "take a train from Jinan to Beijing" ∧ step2 = "fly to Paris"

def statement2 : Prop :=
  ∃ (story : String), story = "boiling water to make tea involves a clear set of rules and steps."

def statement3 : Prop :=
  ∃ (measure : String), measure = "measuring the height of a tree without clear standard criteria."

def statement4 : Prop :=
  ∃ (triangle : String), triangle = "use trigonometric formulas to find sides and angles, and use area formula."

def is_algorithm (stmt : Prop) : Prop :=
  match stmt with
  | statement1 => True
  | statement2 => True
  | statement3 => False
  | statement4 => True
  | _ => False

theorem number_of_algorithms_is_three : 
  (is_algorithm statement1 ∧ is_algorithm statement2 ∧ is_algorithm statement4) ∧ ¬ is_algorithm statement3 → 
  (∃ n, n = 3) :=
by
  sorry

end number_of_algorithms_is_three_l394_394188


namespace value_v2_at_x_10_l394_394219

noncomputable def evaluate_polynomial_horner (f : ℕ → ℕ) (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0     => f n
  | (n+1) => evaluate_polynomial_horner f x n * x + f (n+1)

def horner_test_value (x : ℕ) : ℕ :=
  let f : ℕ → ℕ := λ k, match k with
                        | 4 => 3
                        | 2 => 1
                        | 1 => 2
                        | 0 => 4
                        | _ => 0
                        end
  evaluate_polynomial_horner f x 4

theorem value_v2_at_x_10 : horner_test_value 10 = 310 :=
by
  -- Proof omitted (sorry)
  sorry

end value_v2_at_x_10_l394_394219


namespace cosine_of_subtracted_vectors_l394_394475

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Conditions
def condition1 := ∥a∥ = 1
def condition2 := ∥b∥ = 1
def condition3 := ∥c∥ = real.sqrt 2
def condition4 := a + b + c = 0

-- Proof statement
theorem cosine_of_subtracted_vectors 
  (h1 : condition1 a) 
  (h2 : condition2 b) 
  (h3 : condition3 c) 
  (h4 : condition4 a b c) : 
  inner_product_space.cos (a - c) (b - c) = 4 / 5 :=
sorry

end cosine_of_subtracted_vectors_l394_394475


namespace equation_D_is_quadratic_l394_394658

def is_quadratic_in_one_variable (eq : ℕ → Prop) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ eq = λ x, a * x ^ 2 + b * x + c

def equation_D : ℕ → Prop := λ x, x ^ 2 - 1 = 0

theorem equation_D_is_quadratic :
  is_quadratic_in_one_variable equation_D :=
sorry

end equation_D_is_quadratic_l394_394658


namespace nine_points_on_ellipse_l394_394801

variable {Point : Type}

@[class] def Triangle (P : Type) := (A B C : P)
def Midpoint (P : Type) [AffineSpace ℝ P] (a b : P) := (1/2) • (a + b)
noncomputable def LineIntersection (P : Type) [affine_space ℝ P] (l1 l2 : Set P) := S ∈ intersection(l1, l2)

theorem nine_points_on_ellipse
  {P : Type} [affine_space ℝ P] 
  {A B C P A1 B1 C1 A2 B2 C2 C3 A3 B3 : P}
  (h_triangle_ABC : Triangle P A B C)
  (h_P_inside : P ∈ interior (triangle P A B C))
  (h_A1 : A1 = LineIntersection P (line P A P) (line P B C))
  (h_B1 : B1 = LineIntersection P (line P B P) (line P A C))
  (h_C1 : C1 = LineIntersection P (line P C P) (line P A B))
  (h_A2 : A2 = Midpoint P A P)
  (h_B2 : B2 = Midpoint P B P)
  (h_C2 : C2 = Midpoint P C P)
  (h_C3 : C3 = Midpoint P A B)
  (h_A3 : A3 = Midpoint P B C)
  (h_B3 : B3 = Midpoint P C A) :
  ∃ E : Ellipse P, 
    (A1 ∈ E) ∧ (A2 ∈ E) ∧ (A3 ∈ E) ∧ 
    (B1 ∈ E) ∧ (B2 ∈ E) ∧ (B3 ∈ E) ∧ 
    (C1 ∈ E) ∧ (C2 ∈ E) ∧ (C3 ∈ E) :=
sorry

end nine_points_on_ellipse_l394_394801


namespace rectangle_length_equal_four_l394_394308

theorem rectangle_length_equal_four (area_eq : ∀ (a b : ℝ), a = b → a = b) :
  (∀ (side width length : ℝ), side = 4 → width = 4 → side * side = width * length → length = 4) := 
by
  intro side width length h_side h_width h_area_eq
  have side_sq : side * side = 16 := by
    rw h_side
    norm_num
  have width_eq_four : width = 4 := by
    exact h_width
  have area_eq_side : 16 = width * length := by
    rw h_area_eq

  have lt := 16
  have lw := width * length
  
  sorry

end rectangle_length_equal_four_l394_394308


namespace subcommittee_count_l394_394682

open Nat

def choose (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem subcommittee_count :
  let Republicans := 10
  let Democrats := 7
  choose Republicans 4 * choose Democrats 3 = 7350 :=
by
  sorry

end subcommittee_count_l394_394682


namespace total_weight_of_5_moles_of_cai2_l394_394290

-- Definitions based on the conditions
def weight_of_calcium : Real := 40.08
def weight_of_iodine : Real := 126.90
def iodine_atoms_in_cai2 : Nat := 2
def moles_of_calcium_iodide : Nat := 5

-- Lean 4 statement for the proof problem
theorem total_weight_of_5_moles_of_cai2 :
  (weight_of_calcium + (iodine_atoms_in_cai2 * weight_of_iodine)) * moles_of_calcium_iodide = 1469.4 := by
  sorry

end total_weight_of_5_moles_of_cai2_l394_394290


namespace interior_angle_of_regular_hexagon_l394_394275

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l394_394275


namespace cards_given_to_John_l394_394578

theorem cards_given_to_John :
  ∀ (initial to_Jeff left : ℕ),
    initial = 573 →
    to_Jeff  = 168 →
    left     = 210 →
    (initial - left - to_Jeff = 195) :=
by {
  intros initial to_Jeff left h_initial h_to_Jeff h_left,
  rw [h_initial, h_to_Jeff, h_left],
  norm_num,  -- This simplifies the arithmetic expression 573 - 210 - 168 = 195
  sorry
}

end cards_given_to_John_l394_394578


namespace find_b_l394_394036

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 35 * b) : b = 63 := 
by 
  sorry

end find_b_l394_394036


namespace closest_point_on_line_is_correct_l394_394771

theorem closest_point_on_line_is_correct :
  ∃ (p : ℝ × ℝ), p = (-0.04, -0.28) ∧
  ∃ x : ℝ, p = (x, (3 * x - 1) / 4) ∧
  ∀ q : ℝ × ℝ, (q = (x, (3 * x - 1) / 4) → 
  (dist (2, -3) p) ≤ (dist (2, -3) q)) :=
sorry

end closest_point_on_line_is_correct_l394_394771


namespace range_of_a_l394_394509

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, 0 < x → log x - a * x ≤ 2 * a^2 - 3) : 1 ≤ a :=
sorry

end range_of_a_l394_394509


namespace intersection_point_correct_l394_394452

noncomputable def intersection_points : ℝ × ℝ := (1, (2 * Real.sqrt 5) / 5)

-- Parametric equations for the first curve
def curve1 (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < π) : ℝ × ℝ :=
  (Real.sqrt 5 * Real.cos θ, Real.sin θ)

-- Parametric equations for the second curve
def curve2 (t : ℝ) : ℝ × ℝ :=
  ((5 / 4) * t^2, t)

theorem intersection_point_correct :
  ∃ θ (hθ : 0 ≤ θ ∧ θ < π) t,
    curve1 θ hθ = curve2 t ∧
    curve1 θ hθ = intersection_points :=
sorry

end intersection_point_correct_l394_394452


namespace problem_solution_l394_394027

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 2 - 1) * ((Real.log x / Real.log 2) - 2)

theorem problem_solution (x : ℝ) (h1 : 2 ^ x ≤ 256) (h2 : Real.log x / Real.log 2 ≥ 1 / 2) :
  (√2 ≤ x ∧ x ≤ 8) ∧ 
  (∀ x, (√2 ≤ x ∧ x ≤ 8) → (f x ≥ -1 / 4) ∧ (f x ≤ 2)) :=
by
  sorry

end problem_solution_l394_394027


namespace interior_angle_of_regular_hexagon_l394_394230

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l394_394230


namespace barn_size_correct_l394_394623

def barn_size (price_per_sq_ft house_size property_value : ℕ) : ℕ :=
  let house_value := price_per_sq_ft * house_size
  let barn_value := property_value - house_value
  barn_value / price_per_sq_ft

theorem barn_size_correct :
  barn_size 98 2400 333200 = 1000 :=
by
  unfold barn_size
  simp [Nat.mul_sub, Nat.div_eq_of_lt]
  sorry

end barn_size_correct_l394_394623


namespace remainder_of_sum_of_ns_l394_394082

theorem remainder_of_sum_of_ns (S : ℕ) :
  (∃ (ns : List ℕ), (∀ n ∈ ns, ∃ m : ℕ, n^2 + 12*n - 1997 = m^2) ∧ S = ns.sum) →
  S % 1000 = 154 :=
by
  sorry

end remainder_of_sum_of_ns_l394_394082


namespace jean_pages_written_l394_394912

theorem jean_pages_written:
  (∀ d : ℕ, 150 * d = 900 → d * 2 = 12) :=
by
  sorry

end jean_pages_written_l394_394912


namespace no_cut_off_raisin_center_l394_394064

def raisin_in_center_of_square : Prop := ∀ (square : ℝ) (raisin : ℝ), 
  let center_pos := (square / 2, square / 2) in
  raisin = center_pos

def valid_cut_intersects_adjacent_sides (cut : ℝ × ℝ → ℝ × ℝ) 
      (square : ℝ) : Prop := 
  ∀ (p1 p2 : ℝ × ℝ), 
  (p1.1 ≠ 0 ∧ p1.1 ≠ square ∧ p2.1 ≠ 0 ∧ p2.1 ≠ square ∧
   p1.2 ≠ 0 ∧ p1.2 ≠ square ∧ p2.2 ≠ 0 ∧ p2.2 ≠ square)  -- points are not vertices
  → (p1 = cut (0, p1.2) ∧ p2 = cut (p2.1, 0)) -- cut intersects adjacent sides

theorem no_cut_off_raisin_center (square raisin : ℝ) 
      (cut : ℝ × ℝ → ℝ × ℝ) 
      (h_center : raisin_in_center_of_square square raisin) 
      (h_cut : valid_cut_intersects_adjacent_sides cut square) : 
      ¬ ∃ (triangle_cut : ℝ × ℝ → ℝ × ℝ), 
         raisin ∈ (triangle_cut (0, raisin.2) ∪ triangle_cut (raisin.1, 0)) :=
by
  sorry

end no_cut_off_raisin_center_l394_394064


namespace heat_dissipation_resistor_l394_394723

variables (R C ℰ r : ℝ)

def heat_released_in_resistor (R C ℰ r : ℝ) : ℝ :=
  (C * ℰ^2 * R) / (2 * (R + r))

theorem heat_dissipation_resistor :
  heat_released_in_resistor R C ℰ r = (C * ℰ^2 * R) / (2 * (R + r)) :=
by sorry

end heat_dissipation_resistor_l394_394723


namespace g_of_3_eq_3_l394_394812

theorem g_of_3_eq_3
  (f g : ℝ → ℝ)
  (h_odd : ∀ x, f(-x) = -f(x))
  (h_even : ∀ x, g(-x) = g(x))
  (h1 : f(-3) + g(3) = 2)
  (h2 : f(3) + g(-3) = 4)
  : g(3) = 3 :=
sorry

end g_of_3_eq_3_l394_394812


namespace figure4_total_length_l394_394718

-- Define the conditions
def top_segments_sum := 3 + 1 + 1  -- Sum of top segments in Figure 3
def bottom_segment := top_segments_sum -- Bottom segment length in Figure 3
def vertical_segment1 := 10  -- First vertical segment length
def vertical_segment2 := 9  -- Second vertical segment length
def remaining_segment := 1  -- The remaining horizontal segment

-- Total length of remaining segments in Figure 4
theorem figure4_total_length : 
  bottom_segment + vertical_segment1 + vertical_segment2 + remaining_segment = 25 := by
  sorry

end figure4_total_length_l394_394718


namespace table_seating_problem_l394_394137

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l394_394137


namespace sum_of_factors_of_30_is_72_l394_394294

-- Condition: given the number 30
def number := 30

-- Define the positive factors of 30
def factors : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- Statement to prove the sum of the positive factors
theorem sum_of_factors_of_30_is_72 : (factors.sum) = 72 := 
by
  sorry

end sum_of_factors_of_30_is_72_l394_394294


namespace solve_fx_eq_six_l394_394569

def f (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solve_fx_eq_six (x : ℝ) :
  f(x) = 6 ↔ x = -1/2 ∨ x = 7 :=
by
  sorry

end solve_fx_eq_six_l394_394569


namespace exists_cut_divides_grid_l394_394741

-- Definitions for the structure of the problem
structure Grid (m n : ℕ) :=
(width : ℕ)
(height : ℕ)
(circles : ℕ)
(stars : ℕ)
(positions : fin m → fin n → Prop) -- A function to determine if a circle or star is at a given position

-- Definition for the specific given example in the problem
def six_by_six_grid : Grid 6 6 :=
{ width := 6,
  height := 6,
  circles := 4,
  stars := 4,
  positions := λ i j, (i.val = 0 ∧ j.val = 0) ∨
                    (i.val = 0 ∧ j.val = 3) ∨
                    (i.val = 3 ∧ j.val = 0) ∨
                    (i.val = 3 ∧ j.val = 3) ∨
                    (i.val = 5 ∧ j.val = 5)
  -- Here positions can be customized to match the exact stated configuration, simplified for brevity
}

-- Statement of the proof problem
theorem exists_cut_divides_grid 
  (G : Grid 6 6) : 
  ∃ (cut_row cut_col : ℕ), 
  (cut_row = 3 ∧ cut_col = 3 ∧
   (Π i j, (
      (0 ≤ i < 3 ∧ 0 ≤ j < 3) →
      G.positions i j → count (G.positions i j) = 1 ∙   ∧
      ∀ (0 ≤ i < 3 ∧ 3 ≤ j < 6) →
      G.positions i (j - 3) → count (G.positions i j) = 1 ∙ 
    ))
sorry

end exists_cut_divides_grid_l394_394741


namespace cube_diagonal_correct_l394_394685

noncomputable 
def cube_diagonal (surface_area : ℝ) : ℝ := 
  (√(surface_area / 6)) * √3

theorem cube_diagonal_correct :
  cube_diagonal 294 = 7 * √3 :=
by
  sorry

end cube_diagonal_correct_l394_394685


namespace sequence_sum_l394_394832

theorem sequence_sum (a : ℕ → ℚ) (h : ∀ n, (finset.range n).sum (λ k, 3^k * a (k + 1)) = (n + 1) / 3) :
  ∀ n, a n = if n = 1 then 2 / 3 else 1 / 3^n := 
by 
  sorry  -- Proof to be filled in later

end sequence_sum_l394_394832


namespace fifth_scroll_age_l394_394320

def scroll_age (n : ℕ) : ℕ :=
  match n with
  | 1 => 4080
  | k + 1 => scroll_age k + (scroll_age k / 2)

theorem fifth_scroll_age : scroll_age 5 = 20655 := by
  sorry

end fifth_scroll_age_l394_394320


namespace john_saving_yearly_l394_394539

def old_monthly_cost : ℕ := 1200
def increase_percentage : ℕ := 40
def split_count : ℕ := 3

def old_annual_cost (monthly_cost : ℕ) := monthly_cost * 12
def new_monthly_cost (monthly_cost : ℕ) (percentage : ℕ) := monthly_cost * (100 + percentage) / 100
def new_monthly_share (new_cost : ℕ) (split : ℕ) := new_cost / split
def new_annual_cost (monthly_share : ℕ) := monthly_share * 12
def annual_savings (old_annual : ℕ) (new_annual : ℕ) := old_annual - new_annual

theorem john_saving_yearly 
  (old_cost : ℕ := old_monthly_cost)
  (increase : ℕ := increase_percentage)
  (split : ℕ := split_count) :
  annual_savings (old_annual_cost old_cost) 
                 (new_annual_cost (new_monthly_share (new_monthly_cost old_cost increase) split)) 
  = 7680 :=
by
  sorry

end john_saving_yearly_l394_394539


namespace fractional_inequality_solution_l394_394177

theorem fractional_inequality_solution :
  ∃ (m n : ℕ), n = m^2 - 1 ∧ 
               (m + 2) / (n + 2 : ℝ) > 1 / 3 ∧ 
               (m - 3) / (n - 3 : ℝ) < 1 / 10 ∧ 
               1 ≤ m ∧ m ≤ 9 ∧ 1 ≤ n ∧ n ≤ 9 ∧ 
               (m = 3) ∧ (n = 8) := 
by
  sorry

end fractional_inequality_solution_l394_394177


namespace simplify_expression_l394_394987

-- Define the conditions
variable (x : ℝ)
variable (hx : sin x ≠ 0)

-- Define the statement
theorem simplify_expression :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by sorry

end simplify_expression_l394_394987


namespace correct_propositions_l394_394462

-- Definitions based on conditions
variables (m n : Line) (α β : Plane)

-- Propositions
def prop1 : Prop := (m ∥ α ∧ n ∥ α) → m ∥ n
def prop2 : Prop := (m ∥ α ∧ n ⊥ α) → m ⊥ n
def prop3 : Prop := (m ⊥ α ∧ m ∥ β) → α ⊥ β
def prop4 : Prop := (m ⊥ α ∧ α ⊥ β) → m ∥ β

-- Proving correctness of propositions
theorem correct_propositions : ¬ prop1 ∧ prop2 ∧ prop3 ∧ ¬ prop4 :=
by
  sorry

end correct_propositions_l394_394462


namespace intersection_complement_eq_singleton_l394_394837

universe u
variable {U A B : Set.{u} ℕ}

def setU : Set ℕ := {1, 2, 3, 4}
def setA : Set ℕ := {1, 4}
def setB : Set ℕ := {2, 4}

theorem intersection_complement_eq_singleton :
  (setA ∩ (setU \ setB)) = {1} :=
sorry

end intersection_complement_eq_singleton_l394_394837


namespace distance_between_consecutive_trees_l394_394342

noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) (obstacle_pos : ℝ) (obstacle_gap : ℝ) : ℝ :=
  let planting_distance := yard_length - obstacle_gap
  let num_gaps := num_trees - 1
  planting_distance / num_gaps

theorem distance_between_consecutive_trees :
  distance_between_trees 600 36 250 10 = 16.857 := by
  sorry

end distance_between_consecutive_trees_l394_394342


namespace collinear_C_l394_394798

noncomputable theory

variables (A1 A2 A3 A4 P : Type)
[inhabited A1] [inhabited A2] [inhabited A3] [inhabited A4] [inhabited P]
[line_segment : ∀ {X Y : Type} [inhabited X] [inhabited Y], Prop]

-- Given a quadrilateral A1, A2, A3, A4 inscribed in a circle
axiom quadrilateral_inscribed_in_circle : 
  ∀ {A1 A2 A3 A4 : Type} [inhabited A1] [inhabited A2] [inhabited A3] [inhabited A4], Prop

-- Let P be an arbitrary point on the circle
axiom point_on_circle :
  ∀ {A1 A2 A3 A4 P : Type} [inhabited A1] [inhabited A2] [inhabited A3] [inhabited A4] [inhabited P], Prop

-- The projections of P onto the lines A1A2, A2A3, A3A4, A4A1 are B1, B2, B3, B4
axiom projections_B :
  ∀ {A1 A2 A3 A4 P B1 B2 B3 B4 : Type} 
  [inhabited A1] [inhabited A2] [inhabited A3] [inhabited A4] [inhabited P]
  [inhabited B1] [inhabited B2] [inhabited B3] [inhabited B4], Prop

-- Projections of P onto lines B1B2, B2B3, B3B4, B4B1 are C1, C2, C3, C4
axiom projections_C :
  ∀ {B1 B2 B3 B4 P C1 C2 C3 C4 : Type} 
  [inhabited B1] [inhabited B2] [inhabited B3] [inhabited B4] [inhabited P]
  [inhabited C1] [inhabited C2] [inhabited C3] [inhabited C4], Prop

-- Prove C1, C2, C3, C4 are collinear
theorem collinear_C : 
  ∀ {A1 A2 A3 A4 P B1 B2 B3 B4 C1 C2 C3 C4 : Type}
  [inhabited A1] [inhabited A2] [inhabited A3] [inhabited A4] [inhabited P]
  [inhabited B1] [inhabited B2] [inhabited B3] [inhabited B4]
  [inhabited C1] [inhabited C2] [inhabited C3] [inhabited C4], 
  quadrilateral_inscribed_in_circle →
  point_on_circle →
  projections_B →
  projections_C →
  line_segment C1 C2 ∧ line_segment C2 C3 ∧ line_segment C3 C4 :=
sorry

end collinear_C_l394_394798


namespace cubesWithTwoColoredFaces_l394_394730

structure CuboidDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

def numberOfSmallerCubes (d : CuboidDimensions) : ℕ :=
  d.length * d.width * d.height

def numberOfCubesWithTwoColoredFaces (d : CuboidDimensions) : ℕ :=
  2 * (d.length - 2) * 2 + 2 * (d.width - 2) * 2 + 2 * (d.height - 2) * 2

theorem cubesWithTwoColoredFaces :
  numberOfCubesWithTwoColoredFaces { length := 4, width := 3, height := 3 } = 16 := by
  sorry

end cubesWithTwoColoredFaces_l394_394730


namespace number_of_people_seated_l394_394146

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l394_394146


namespace parallel_ZN_AD_BC_l394_394917

variables {A B C D M X Y Z N : Type*}
variables [linear_ordered_field ℝ]

-- Define points in the context of an isosceles trapezoid
variables (A B C D M X Y Z N : ℝ)
variables {trapezoid : Type*}
variables (AD_parallel_BC : A * D = B * C)
variables (intersect_M : A * C + B * D = 2 * M)
variables (AX_AM : A + X = 2 * M)
variables (BY_BM : B + Y = 2 * M)
variables (Z_midpoint : Z = (X + Y) / 2)
variables (N_intersection : N = intersect_of(X, D, Y, C))

-- The key theorem to prove
theorem parallel_ZN_AD_BC : Parallel Z N A D ∧ Parallel Z N B C :=
sorry

end parallel_ZN_AD_BC_l394_394917


namespace largest_base_not_sum_of_digits_evict_2_4_l394_394655

theorem largest_base_not_sum_of_digits_evict_2_4 :
  ∃ b : ℕ, (∀ base : ℕ, base ≤ b → ( ∑ d in (↑((base + 1) ^ 4)).digits b, d ) ≠ 2^4) ∧ b = 6 := 
begin
  use 6,
  split,
  {
    intros base h,
    by_cases base = 6,
    {
      simp [h],
      norm_num,
    },
    {
      sorry, -- Proof for all bases considering different restrictions
    }
  },
  {
    refl,
  },
end

end largest_base_not_sum_of_digits_evict_2_4_l394_394655


namespace probability_x_squared_lt_y_correct_l394_394707

noncomputable def probability_x_squared_lt_y : ℝ :=
  let area_under_curve := ∫ x in 0..1, x^2
  let area_rectangle := 5
  area_under_curve / area_rectangle

theorem probability_x_squared_lt_y_correct :
  probability_x_squared_lt_y = 1 / 15 :=
by
  -- Proof will be provided here.
  sorry

end probability_x_squared_lt_y_correct_l394_394707


namespace common_root_uniqueness_l394_394907

noncomputable theory

variable {p q : ℝ}

theorem common_root_uniqueness (h₁ : ∃ (x₀ : ℝ), 2017 * x₀^2 + p * x₀ + q = 0 ∧ p * x₀^2 + q * x₀ + 2017 = 0) :
  ∃ (x₀ : ℝ), x₀ = 1 → p + q = -2017 :=
by
  sorry

end common_root_uniqueness_l394_394907


namespace remove_two_fractions_sum_is_one_l394_394339

theorem remove_two_fractions_sum_is_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (fractions.sum : ℚ)
  let remaining_sum := total_sum - (1/8 + 1/10)
  remaining_sum = 1 := by
    sorry

end remove_two_fractions_sum_is_one_l394_394339


namespace seventh_number_fifth_row_is_272_l394_394178

-- Define the initial sequence and recursive rule
def initial_row (n : ℕ) : list ℕ :=
  list.map (λ i, 2*i - 1) (list.range n)

def next_row (prev_row : list ℕ) : list ℕ :=
  match prev_row with
  | [] => []
  | [_] => []
  | (a :: b :: rest) => (a + b) :: next_row (b :: rest)

-- Recursively build the triangle
def build_triangle : ℕ → list (list ℕ)
  | 0 => []
  | 1 => [initial_row 1]
  | (n+1) => let prev_triangle := build_triangle n in
    prev_triangle ++ [next_row (prev_triangle.last'.get_or_else [])]

-- Define the problem statement
theorem seventh_number_fifth_row_is_272 :
  let triangle := build_triangle 5 in
  triangle.nth 4 = some [80, 112, 144, 176, 208, 240, 272] ∧ (triangle.nth 4).get_or_else [].nth 6 = some 272 :=
sorry

end seventh_number_fifth_row_is_272_l394_394178


namespace hyperbola_eccentricity_l394_394369

noncomputable def right_focus (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ℝ := real.sqrt (a^2 + b^2)

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h₁ : a > 0) 
  (h₂ : b > 0)
  (h₃ : c = right_focus a b h₁ h₂) 
  (h₄ : e = c / a) : e = real.sqrt 2 := 
sorry

end hyperbola_eccentricity_l394_394369


namespace ellipse_eccentricity_correct_l394_394430

noncomputable def ellipse_eccentricity 
  (a b : ℝ) 
  (h_eq : b^2 = 2) 
  (h_major_axis : 2 * a = 6) : ℝ :=
let c := Real.sqrt (a^2 - b^2) in
c / a

theorem ellipse_eccentricity_correct :
  ∀ (a b : ℝ), 
    b^2 = 2 → 
    2 * a = 6 → 
    ellipse_eccentricity a b (by sorry) (by sorry) = Real.sqrt 7 / 3 :=
by
  intros a b h_eq h_major_axis
  unfold ellipse_eccentricity
  sorry

end ellipse_eccentricity_correct_l394_394430


namespace total_money_spent_l394_394075

theorem total_money_spent {s j : ℝ} (hs : s = 14.28) (hj : j = 4.74) : s + j = 19.02 :=
by
  sorry

end total_money_spent_l394_394075


namespace f_prime_at_2_l394_394440

noncomputable def f (x : ℝ) : ℝ := sorry
def g (x : ℝ) : ℝ := f(x) / x

-- Condition 1: The point (2, 1) lies on y = f(x) / x which gives f(2) = 2 
def condition1 : Prop := f(2) = 2

-- Condition 2: The lines tangent to y = f(x) at (0,0) and y = f(x) / x at (2,1) have the same slope == 1/2
def tangent_slope : ℝ := 1/2
def g' := λ x, (x * (Deriv f x) - f x) / (x^2)
def condition2 : Prop := g'(2) = tangent_slope

-- Proving f'(2) = 2
theorem f_prime_at_2 : condition1 ∧ condition2 → (Deriv f 2) = 2 :=
by
  sorry

end f_prime_at_2_l394_394440


namespace equilateral_triangle_side_length_l394_394709

theorem equilateral_triangle_side_length (perimeter : ℕ) (h_perimeter : perimeter = 69) : 
  ∃ (side_length : ℕ), side_length = perimeter / 3 := 
by
  sorry

end equilateral_triangle_side_length_l394_394709


namespace range_of_f_x1_x2_l394_394015

noncomputable def f (a x : ℝ) : ℝ := 2 * a * (Real.log x - x) + x^2

def range_of_extremes (a : ℝ) : Set ℝ :=
  let x₁ := a / 2 - (Real.sqrt (a^2 - 4 * a)) / 2
  let x₂ := a / 2 + (Real.sqrt (a^2 - 4 * a)) / 2
  {f a x₁ + f a x₂}

theorem range_of_f_x1_x2 (a : ℝ) (h : a > 4) :
  ∃ y, y ∈ range_of_extremes a ↔ f a x₁ + f a x₂ ∈ Iio (16 * Real.log 2 - 24) := 
sorry

end range_of_f_x1_x2_l394_394015


namespace T_mod_2027_l394_394559

def T : ℤ :=
  (List.sum (List.map (λ (i : ℕ), if i % 4 < 2 then i.succ else -(i.succ)) (List.range 2026)))

theorem T_mod_2027 :
  T % 2027 = 2026 :=
by
  sorry

end T_mod_2027_l394_394559


namespace remove_two_fractions_sum_is_one_l394_394340

theorem remove_two_fractions_sum_is_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (fractions.sum : ℚ)
  let remaining_sum := total_sum - (1/8 + 1/10)
  remaining_sum = 1 := by
    sorry

end remove_two_fractions_sum_is_one_l394_394340


namespace f_2013_l394_394743

noncomputable def f (x : ℝ) : ℝ := 
  if hx : -2 < x ∧ x < 0 then (Real.sqrt 2 + 1) ^ x
  else sorry  -- Placeholder as full definition is based on periodicity

lemma f_periodic (x : ℝ) : f (x + 2) = f x :=
sorry

lemma f_neg_one : f (-1) = Real.sqrt 2 - 1 :=
  begin
    -- Proof using the given condition when x ∈ (-2,0)
    unfold f,
    rw if_pos,
    { rw [pow_neg, ←div_eq_inv_mul],
      have h : Real.sqrt 2 - 1 = (1 - 1 / ( Real.sqrt 2+1)) * (Real.sqrt 2 + 1) := by sorry,
      exact h.symm },
    split,
    norm_num,
    norm_num,
  end

theorem f_2013 : f 2013 = Real.sqrt 2 - 1 :=
by
  have h1 : 2013 = 2 * 1006 + 1 := by norm_num,
  rw [←f_periodic 1, h1], 
  exact f_neg_one

end f_2013_l394_394743


namespace find_q_l394_394034

theorem find_q (p q : ℝ) (h : ∀ x : ℝ, (x^2 + p * x + q) ≥ 1) : q = 1 + (p^2 / 4) :=
sorry

end find_q_l394_394034


namespace tangent_line_eq_extreme_points_a_eq_1_func_inequality_x_ge_1_l394_394821

-- Define function f(x) = ln x - a*(x - 1)
def f (x : ℝ) (a : ℝ) : ℝ := log x - a * (x - 1)

-- (I) Equation of the tangent line to f(x) at (1, f(1)) is y = (1 - a)(x - 1)
theorem tangent_line_eq (a : ℝ) : 
  (∃ y : ℝ, y = (1 - a) * (x - 1)) :=
    sorry

-- (II) For a = 1, extreme points and values of f(x)
theorem extreme_points_a_eq_1 : 
  (∀ x : ℝ, f(x) 1 = log x - x + 1) → 
  (∃ x : ℝ, is_local_max (λ x, f(x) 1) 1) ∧ 
  (¬(∃ x : ℝ, is_local_min (λ x, f(x) 1) x)) :=
    sorry

-- (III) For x ≥ 1, f(x) ≤ ln x / (x + 1), find range of a
theorem func_inequality_x_ge_1 (a : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → f(x) a ≤ log x / (x + 1)) → 
  a ∈ Ioi (1/2) :=
    sorry

end tangent_line_eq_extreme_points_a_eq_1_func_inequality_x_ge_1_l394_394821


namespace regular_hexagon_interior_angle_measure_l394_394233

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l394_394233


namespace September_1st_is_Wednesday_l394_394310

theorem September_1st_is_Wednesday (skip_missed : ℕ → ℕ) (total_missed : ℕ) :
  (∀ n, (n % 7 = 1 → skip_missed n = 1) ∧ 
        (n % 7 = 2 → skip_missed n = 2) ∧ 
        (n % 7 = 3 → skip_missed n = 3) ∧ 
        (n % 7 = 4 → skip_missed n = 4) ∧ 
        (n % 7 = 5 → skip_missed n = 5) ∧ 
        (n % 7 = 0 ∨ n % 7 = 6 → skip_missed n = 0)) →
  total_missed = 64 →
  (skip_missed 1 = 0) →
  (∀ n, 1 ≤ n ∧ n ≤ 30 → skip_missed n) % 7 = 2 :=
by {
  sorry
}

end September_1st_is_Wednesday_l394_394310


namespace equation_of_line_m_l394_394384

theorem equation_of_line_m 
  (Q : ℝ × ℝ) 
  (Q'' : ℝ × ℝ) 
  (origin : ℝ × ℝ)
  (line_ell : ℝ → ℝ → Prop) 
  (line_intersects_origin : ∃ x y : ℝ, line_ell x y ∧ (x, y) = origin)
  (Q''_coords : Q'' = (5, 2)) :

  line_ell 3 (-1) ∧ Q = (-2, 5) ∧ origin = (0, 0) ∧ 
  (∀ Q' : ℝ × ℝ, reflect_about_line Q line_ell = Q' → reflect_about_line Q' line_m = Q'') →
  ∃ (a b : ℝ), a * x - b * y = 0 ∧ 
  (∃ t : ℝ, line_m x y = t * (x - 2 * y)) := 
sorry

end equation_of_line_m_l394_394384


namespace compute_ggggg4_l394_394568

def g (x : ℝ) : ℝ :=
if x ≥ 3 then -x^2 + 1 else x + 10

theorem compute_ggggg4 : g (g (g (g (g 4)))) = -14 :=
by
  sorry

end compute_ggggg4_l394_394568


namespace cost_of_each_book_l394_394538

noncomputable def cost_of_book (money_given money_left notebook_cost notebook_count book_count : ℕ) : ℕ :=
  (money_given - money_left - (notebook_count * notebook_cost)) / book_count

-- Conditions
def money_given : ℕ := 56
def money_left : ℕ := 14
def notebook_cost : ℕ := 4
def notebook_count : ℕ := 7
def book_count : ℕ := 2

-- Theorem stating that the cost of each book is $7 under given conditions
theorem cost_of_each_book : cost_of_book money_given money_left notebook_cost notebook_count book_count = 7 := by
  sorry

end cost_of_each_book_l394_394538


namespace range_of_a1_of_arithmetic_sequence_l394_394419

theorem range_of_a1_of_arithmetic_sequence
  {a : ℕ → ℝ} (S : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum: ∀ n, S n = (n + 1) * (a 0 + a n) / 2)
  (h_min: ∀ n > 0, S n ≥ S 0)
  (h_S1: S 0 = 10) :
  -30 < a 0 ∧ a 0 < -27 := 
sorry

end range_of_a1_of_arithmetic_sequence_l394_394419


namespace intersection_is_as_expected_l394_394022

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x | 2 * x^2 - 3 * x - 2 ≤ 0 }

noncomputable def logarithmic_condition : Set ℝ :=
  { x | x > 0 ∧ x ≠ 1 }

noncomputable def intersection_of_sets : Set ℝ :=
  (quadratic_inequality_solution ∩ logarithmic_condition)

theorem intersection_is_as_expected :
  intersection_of_sets = { x | (0 < x ∧ x < 1) ∨ (1 < x ∧ x ≤ 2) } :=
by
  sorry

end intersection_is_as_expected_l394_394022


namespace cross_section_area_parallel_l394_394800

-- Definitions
def regular_tetrahedron (S A B C : Point) (SO height : ℝ) (BC length : ℝ) : Prop :=
  (SO = 3) ∧ (BC = 6) -- This captures the height and the base side length conditions

def perpendicular (A S B C O' : Point) (segment_length: (A - O').length): Prop :=
  (S.is_perpendicular_to_plane (A, segment_length))

-- Given constants for the problem
noncomputable def S : Point := sorry
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def O' : Point := sorry
noncomputable def P : Point := sorry

-- Representation of the length ratio
def length_ratio (AP PO' : ℝ) : Prop :=
  (AP / PO' = 8)

-- Main theorem stating the area of the cross-section parallel to base through P
theorem cross_section_area_parallel (S A B C O' P : Point) (height side_length : ℝ) (AP PO' : ℝ)
  (h1 : regular_tetrahedron S A B C height side_length)
  (h2 : perpendicular A S B C O' AP)
  (h3 : length_ratio AP PO') :
  cross_section_area_parallel_to_base_through (S A B C O' P) = real.sqrt 3 := 
  sorry

end cross_section_area_parallel_l394_394800


namespace solve_trig_eq_l394_394599

theorem solve_trig_eq {x : ℝ} (hx : cos x ≠ 0) :
  (sin (2 * x))^2 - 4 * (sin x)^2) / ((sin (2 * x))^2 + 4 * (sin x)^2 - 4) + 1 = 2 * (tan x)^2 ↔
  ∃ m : ℤ, x = π / 4 * (2 * m + 1) :=
sorry

end solve_trig_eq_l394_394599


namespace interior_angle_of_regular_hexagon_l394_394277

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l394_394277


namespace height_of_cylinder_is_2_25_l394_394905

noncomputable def original_height_of_cylinder (h : ℝ) : Prop :=
  let r := 3
  let V_original := π * r^2 * h
  let V_radius_incremented := π * (r + 4)^2 * h
  let V_height_incremented := π * r^2 * (h + 10)
  (V_radius_incremented - V_original = V_height_incremented - V_original) →
  (V_radius_incremented - V_original = 90 * π) →
  h = 2.25

theorem height_of_cylinder_is_2_25 (h : ℝ) (z : ℝ) : original_height_of_cylinder h :=
by
  sorry

end height_of_cylinder_is_2_25_l394_394905


namespace inverse_sum_is_minus_two_l394_394572

variable (f : ℝ → ℝ)
variable (h_injective : Function.Injective f)
variable (h_surjective : Function.Surjective f)
variable (h_eq : ∀ x : ℝ, f (x + 1) + f (-x - 3) = 2)

theorem inverse_sum_is_minus_two (x : ℝ) : f⁻¹ (2009 - x) + f⁻¹ (x - 2007) = -2 := 
  sorry

end inverse_sum_is_minus_two_l394_394572


namespace absolute_value_lower_bound_squares_roots_l394_394323

theorem absolute_value_lower_bound_squares_roots 
  (n : ℕ)
  (r : Fin n → ℝ)
  (a : ℕ → ℝ)
  (poly_monic : a n = 1)
  (a_n_minus_1_eq_2a_n_minus_2 : a (n - 1) = 2 * a (n - 2))
  (sum_roots_eq_neg_an_minus_1 : ∑ i in Finset.range n, r i = -a (n - 1))
  (sum_product_roots_eq_an_minus_2 : ∑ i in Finset.range n, ∑ j in Finset.range i, r i * r j  = a (n - 2)) :
  |-(∑ i in Finset.range n, r i ^ 2 + 2 * a (n - 2)) / 2| = 1 / 16 := 
by 
  sorry

end absolute_value_lower_bound_squares_roots_l394_394323


namespace number_of_rows_is_ten_l394_394700

-- Definition of the arithmetic sequence
def arithmetic_sequence_sum (n : ℕ) : ℕ :=
  n * (3 * n + 1) / 2

-- The main theorem to prove
theorem number_of_rows_is_ten :
  (∃ n : ℕ, arithmetic_sequence_sum n = 145) ↔ n = 10 :=
by
  sorry

end number_of_rows_is_ten_l394_394700


namespace number_of_ones_l394_394092

def sequence_condition (s : Fin 2015 → ℤ) : Prop :=
  (∀ i, s i ∈ {-1, 0, 1}) ∧
  (∑ i, s i = 5) ∧
  (∑ i, (s i + 1)^2 = 3040)

theorem number_of_ones (s : Fin 2015 → ℤ) (h : sequence_condition s) :
  (∑ i, if s i = 1 then 1 else 0) = 510 :=
sorry

end number_of_ones_l394_394092


namespace find_D_from_distance_l394_394890

-- Defining the conditions
variable (C : ℝ)
noncomputable def distance_point_to_line : ℝ :=
  abs ((1 * (-C) + (-1) * 0 + 0) / real.sqrt (1^2 + (-1)^2))

-- Defining the given condition
axiom given_distance (h : distance_point_to_line C = real.sqrt 8)

-- The theorem we want to prove
theorem find_D_from_distance :
  ∃ D, distance_point_to_line C = real.sqrt D ∧ D = 8 :=
sorry

end find_D_from_distance_l394_394890


namespace students_tried_out_l394_394634

theorem students_tried_out (x : ℕ) (h1 : 8 * (x - 17) = 384) : x = 65 := 
by
  sorry

end students_tried_out_l394_394634


namespace simplify_trig_expression_l394_394980

variable {x : ℝ}

theorem simplify_trig_expression (h : 1 + cos x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l394_394980


namespace layer_sum_eq_2014_implies_sum_of_digits_l394_394035

theorem layer_sum_eq_2014_implies_sum_of_digits :
  ∀ (w x y z : ℕ), 
    w ≠ 0 ∧ 1 ≤ w ∧ w ≤ 9 ∧ 
    0 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 ∧ 
    0 ≤ z ∧ z ≤ 9 ∧ 
    1000 * w + 100 * x + 10 * y + z + 
    100 * x + 10 * y + z + 
    10 * y + z + 
    z = 2014 -> w + x + y + z = 13 :=
by {
  intros,
  sorry
}

end layer_sum_eq_2014_implies_sum_of_digits_l394_394035


namespace lambda_plus_mu_eq_one_l394_394422

def z1 := Complex.mk (-1) 2
def z2 := Complex.mk 1 (-1)
def z3 := Complex.mk 3 (-4)

def A := (z1.re, z1.im)
def B := (z2.re, z2.im)
def C := (z3.re, z3.im)

variables (λ μ : ℝ)

theorem lambda_plus_mu_eq_one (h : (C.1, C.2) = (λ * A.1 + μ * B.1, λ * A.2 + μ * B.2)) : λ + μ = 1 :=
sorry

end lambda_plus_mu_eq_one_l394_394422


namespace geometric_sequence_sum_range_l394_394796

noncomputable def geometric_sum (q : ℚ) : ℚ :=
  (2 / q) + 2 + (2 * q)

theorem geometric_sequence_sum_range (q : ℚ) (hq : q ≠ 0) :
  geometric_sum q ∈ Iic (-2) ∪ Ici 6 :=
by
  sorry

end geometric_sequence_sum_range_l394_394796


namespace evaluate_expression_l394_394759

def expression_evaluation (a b : ℝ) (h : a ≠ b) : ℝ :=
  (a ^ (-6) - b ^ (-6)) / (a ^ (-3) - b ^ (-3))

theorem evaluate_expression (a b : ℝ) (h : a ≠ b) :
  expression_evaluation a b h = a ^ (-6) + a ^ (-3) * b ^ (-3) + b ^ (-6) := by
  sorry

end evaluate_expression_l394_394759


namespace number_of_five_digit_palindromic_numbers_l394_394494

theorem number_of_five_digit_palindromic_numbers : 
  ∃ n : ℕ, (∀ (n_str : String), n_str.length = 5 ∧ (n_str.to_list = n_str.to_list.reverse) → True) ∧ n = 900 :=
sorry

end number_of_five_digit_palindromic_numbers_l394_394494


namespace total_students_trip_l394_394579

theorem total_students_trip:
  let buses          := 7
  let students_per_bus := 53
  let students_in_cars := 4
  let students_total   := buses * students_per_bus + students_in_cars
  in students_total = 375 := 
by 
  sorry

end total_students_trip_l394_394579


namespace find_n_and_p_l394_394454

-- Define the random variable η with Binomial distribution B(n, p)
variable {η : Type} [ProbabilityTheory.RandomVariable (ProbabilityTheory.Binomial n p) η]

-- Define the conditions given in the problem
def expectation_condition (n p : ℕ) (hnp : 0 ≤ p ∧ p ≤ 1) : Prop :=
  2 * @ProbabilityTheory.Expectation _ _ η = 8

def variance_condition (n p : ℝ) (hnp : 0 ≤ p ∧ p ≤ 1) : Prop :=
  16 * (@ProbabilityTheory.Variance _ _ η) = 32

-- State the proof problem
theorem find_n_and_p (n : ℕ) (p : ℝ) (hnp : 0 ≤ p ∧ p ≤ 1) :
  expectation_condition n p hnp ∧ variance_condition n p hnp → n = 8 ∧ p = 0.5 :=
begin
  sorry
end

end find_n_and_p_l394_394454


namespace probability_of_different_colors_is_3_over_5_l394_394516

def total_ways_to_draw_two_balls (total_balls : ℕ) : ℕ :=
  total_balls.choose 2

def ways_to_draw_different_colors (red_balls white_balls : ℕ) : ℕ :=
  red_balls.choose 1 * white_balls.choose 1

def probability_of_different_colors (red_balls white_balls : ℕ) (total_balls_drawn : ℕ) : ℚ :=
  let total_ways := total_ways_to_draw_two_balls (red_balls + white_balls)
  let ways_diff_colors := ways_to_draw_different_colors red_balls white_balls
  ways_diff_colors / total_ways

theorem probability_of_different_colors_is_3_over_5 :
  probability_of_different_colors 2 3 2 = 3 / 5 :=
by
  unfold probability_of_different_colors
  unfold total_ways_to_draw_two_balls
  have h1 : 5.choose 2 = 10 := by rfl
  have h2 : 2.choose 1 * 3.choose 1 = 6 := by rfl
  rw [h1, h2]
  norm_num

end probability_of_different_colors_is_3_over_5_l394_394516


namespace sum_of_digits_of_special_two_digit_number_l394_394203

theorem sum_of_digits_of_special_two_digit_number (x : ℕ) (h1 : 1 ≤ x ∧ x < 10) 
  (h2 : ∃ (n : ℕ), n = 11 * x + 30) 
  (h3 : ∃ (sum_digits : ℕ), sum_digits = (x + 3) + x) 
  (h4 : (11 * x + 30) % ((x + 3) + x) = 3)
  (h5 : (11 * x + 30) / ((x + 3) + x) = 7) :
  (x + 3) + x = 7 := 
by 
  sorry

end sum_of_digits_of_special_two_digit_number_l394_394203


namespace number_of_primes_under_150_with_ones_digit_3_l394_394849

noncomputable def primes_under_150_with_ones_digit_3 : Finset ℕ :=
  Finset.filter (λ n, Nat.Prime n) (Finset.filter (λ n, n % 10 = 3) (Finset.range 150))

theorem number_of_primes_under_150_with_ones_digit_3 :
  Finset.card primes_under_150_with_ones_digit_3 = 9 :=
by
  sorry

end number_of_primes_under_150_with_ones_digit_3_l394_394849


namespace minimize_A_l394_394817

noncomputable def a (n : ℕ) : ℚ := (64 - 4 * n) / 5

noncomputable def A (n : ℕ) : ℚ := abs (∑ i in finset.range 13, a (n + i))

theorem minimize_A : {n : ℕ // A n = A (10)} :=
begin
  use 10,
  -- Proof will be filled here.
  sorry
end

end minimize_A_l394_394817


namespace interior_angle_regular_hexagon_l394_394239

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l394_394239


namespace sum_of_factors_of_30_l394_394295

theorem sum_of_factors_of_30 : 
  let n := 30 in sum (filter (λ d, n % d = 0) (list.range (n + 1))) = 72 :=
by 
  let n := 30
  sorry

end sum_of_factors_of_30_l394_394295


namespace incorrect_increasing_interval_l394_394450

noncomputable def f : ℝ → ℝ := λ x, Real.sin (2 * x + Real.pi / 6)

theorem incorrect_increasing_interval :
  ¬ (∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → ∃ y ≥ -Real.pi / 4 ∧ y ≤ Real.pi / 4, f y < f x) := 
sorry

end incorrect_increasing_interval_l394_394450


namespace ball_arrangement_l394_394111

theorem ball_arrangement :
  ∃ (f : list ℕ → list ℕ → Prop), (∀ w b : list ℕ, length w = 4 → length b = 5 →
    (∀ i : ℕ, i ∈ [0,1,2] → (w.nth i).isSome ∧ (b.nth i).isSome) →
    (∀ l : list ℕ, l ∈ [w, b] → l.sum ≥ 2 * 3) →
    (f w b ↔ 3 * 6 = 18)) :=
by
  sorry

end ball_arrangement_l394_394111


namespace height_of_scale_model_l394_394349

theorem height_of_scale_model (scale_ratio actual_height : ℕ) (h1 : scale_ratio = 25) (h2 : actual_height = 1454) : (actual_height / scale_ratio : ℝ).round = 58 := 
by {
  rw [h1, h2],
  sorry -- Proof omitted
}

end height_of_scale_model_l394_394349


namespace sqrt3_identity_l394_394041

noncomputable def sqrt3 : ℝ := real.sqrt 3

def integer_part (r : ℝ) : ℝ := real.floor r
def decimal_part (r : ℝ) : ℝ := r - real.floor r

theorem sqrt3_identity :
  let x := integer_part sqrt3
  let y := decimal_part sqrt3
  sqrt3 * x - y = 1 :=
by
  sorry

end sqrt3_identity_l394_394041


namespace rectangular_prism_parallel_edges_l394_394844

theorem rectangular_prism_parallel_edges (length width height : ℕ) (h1 : length ≠ width) (h2 : width ≠ height) (h3 : length ≠ height) : 
  ∃ pairs : ℕ, pairs = 6 := by
  sorry

end rectangular_prism_parallel_edges_l394_394844


namespace monica_problem_l394_394946

open Real

noncomputable def completingSquare : Prop :=
  ∃ (b c : ℤ), (∀ x : ℝ, (x - 4) ^ 2 = x^2 - 8 * x + 16) ∧ b = -4 ∧ c = 8 ∧ (b + c = 4)

theorem monica_problem : completingSquare := by
  sorry

end monica_problem_l394_394946


namespace remaining_card_is_diamonds_3_l394_394690

-- Define the structure and initial arrangement of the deck.
def deck : List String := ["Joker", "small Joker"] ++
                        ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"].bind (λ rank =>
                        ["Spades", "Hearts", "Diamonds", "Clubs"].map (λ suit =>
                        suit ++ " " ++ rank))

-- Define the discard procedure
noncomputable def discardProcedure (cards : List String) : String :=
  let rec discard (remaining : List String) (step : Nat) : String :=
    match remaining with
    | []     => "No cards left"
    | [x]    => x
    | x :: xs => discard (xs.rotateLeft 1) (step + 1)
  discard cards 0

-- The theorem to prove the last remaining card.
theorem remaining_card_is_diamonds_3 : discardProcedure deck = "Diamonds 3" :=
sorry

end remaining_card_is_diamonds_3_l394_394690


namespace find_total_covered_area_l394_394215

-- Define the side length
def side_length : ℝ := 12

-- Define the area of a single square
def square_area (side : ℝ) : ℝ := side * side

-- The total area if there were no overlap
def total_area_without_overlap (side : ℝ) : ℝ := 2 * square_area side

-- The area of the overlap
def overlap_area (side : ℝ) : ℝ := 2 * (square_area side / 4)

-- Total area covered by these two squares
def total_covered_area (side : ℝ) : ℝ := total_area_without_overlap side - overlap_area side

-- Prove that the total covered area is 216 sq units
theorem find_total_covered_area : total_covered_area side_length = 216 :=
by
  -- proof will be placed here
  sorry

end find_total_covered_area_l394_394215


namespace boar_sausages_left_l394_394947

def sausages_left (s0: ℕ) (monday_eaten: ℕ) (tuesday_eaten: ℕ) (friday_eaten: ℕ) : ℕ :=
  s0 - monday_eaten - tuesday_eaten - friday_eaten

theorem boar_sausages_left : ∀ (s0: ℕ), 
  let s0 := 600 in
  let monday_eaten := 2 * s0 / 5 in
  let s1 := s0 - monday_eaten in
  let tuesday_eaten := s1 / 2 in
  let s2 := s1 - tuesday_eaten in
  let friday_eaten := 3 * s2 / 4 in
  sausages_left s0 monday_eaten tuesday_eaten friday_eaten = 45 := 
by
  intros
  sorry

end boar_sausages_left_l394_394947


namespace derivative_of_exp_x_plus_x_l394_394013

-- Statement of the problem
theorem derivative_of_exp_x_plus_x (x : ℝ) :
  deriv (λ x, Real.exp x + x) x = Real.exp x + 1 :=
by
  sorry

end derivative_of_exp_x_plus_x_l394_394013


namespace not_sufficient_nor_necessary_geometric_seq_l394_394068

theorem not_sufficient_nor_necessary_geometric_seq {a : ℕ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q) :
    (a 1 < a 3) ↔ (¬(a 2 < a 4) ∨ ¬(a 4 < a 2)) :=
by
  sorry

end not_sufficient_nor_necessary_geometric_seq_l394_394068


namespace x_intercept_perpendicular_l394_394650

noncomputable def line1 := (3 : ℝ) * (λ x, x) - (2 : ℝ) * (λ y, y) = 6

noncomputable def perp_slope (m : ℝ) : ℝ := -1 / m

noncomputable def line2 (y_intercept : ℝ) :=
  (λ x, - (2 / 3 : ℝ) * x + y_intercept)

theorem x_intercept_perpendicular (y_intercept : ℝ) (x_intercept : ℝ) :
  (∃ (x : ℝ), line2 y_intercept x = 0) ∧ y_intercept = 5 ∧ perp_slope (3 / 2) = -(2 / 3) → x_intercept = 7.5 :=
begin
  sorry
end

end x_intercept_perpendicular_l394_394650


namespace P_subset_Q_l394_394547

def P (x : ℝ) := abs x < 2
def Q (x : ℝ) := x < 2

theorem P_subset_Q : ∀ x : ℝ, P x → Q x := by
  sorry

end P_subset_Q_l394_394547


namespace initial_wage_illiterate_l394_394055

variable (I : ℕ) -- initial daily average wage of illiterate employees

theorem initial_wage_illiterate (h1 : 20 * I - 20 * 10 = 300) : I = 25 :=
by
  simp at h1
  sorry

end initial_wage_illiterate_l394_394055


namespace minimum_value_side_c_l394_394514

open Real

noncomputable def minimum_side_c (a b c : ℝ) (B : ℝ) (S : ℝ) : ℝ := c

theorem minimum_value_side_c (a b c B : ℝ) (h1 : c * cos B = a + 1 / 2 * b)
  (h2 : S = sqrt 3 / 12 * c) :
  minimum_side_c a b c B S >= 1 :=
by
  -- Precise translation of mathematical conditions and required proof. 
  -- The actual steps to prove the theorem would be here.
  sorry

end minimum_value_side_c_l394_394514


namespace range_of_a_l394_394449

def f (x a : ℝ) : ℝ := (x^2 + 2 * x + a) / x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x → f(x, a) > 0) → a ∈ set.Ioi (-3) :=
  by 
  sorry

end range_of_a_l394_394449


namespace distance_first_hour_l394_394201

theorem distance_first_hour (x : ℕ) 
  (h1 : ∀ n : ℕ, n > 0 → distance n = sum (i in range n, (x + 2 * (i - 1))) )
  (h2 : distance 12 = 552) : 
  x = 35 := 
  sorry

end distance_first_hour_l394_394201


namespace length_EQ_is_4_l394_394052

-- Define the problem with the given conditions
theorem length_EQ_is_4 :
  ∀ (EFGH IJKL : Type) (s : ℕ) (a b : ℕ) (EH IJ : EFGH → IJKL → Prop) (shaded_area_ratio : ℚ),
  (∀ (x : EFGH), s = 8) →               -- EFGH is a square with side length 8
  (∀ (x : IJKL), a = 12 ∧ b = 8) →      -- IJKL is a rectangle with JL = 12 and IK = 8
  (∀ (e h i j : EFGH) (eh : EH e h) (ij : IJ i j), (eh ∧ ij → ⊥)) →  -- EH and IJ are perpendicular
  shaded_area_ratio = (1 / 3) →         -- shaded_area_ratio is one-third
  EQ = 4 :=                             -- conclusion: length of EQ is 4
by
  intros _ _ _ _ _ _ h_square h_rectangle h_perpendicular h_ratio,
  sorry

end length_EQ_is_4_l394_394052


namespace sum_first_nine_terms_eq_zero_l394_394520

noncomputable def a_n (a₁ d : ℝ) : ℕ → ℝ
| 0       => a₁
| (n + 1) => a_n a₁ d n + d

def arithmetic_seq_condition (a₁ d : ℝ) : Prop :=
a_n a₁ d 5 = a_n a₁ d 2 + a_n a₁ d 7

def S_n (a₁ d : ℝ) (n : ℕ) : ℝ :=
n * a₁ + (n * (n - 1) / 2) * d

theorem sum_first_nine_terms_eq_zero (a₁ d : ℝ) (h : arithmetic_seq_condition a₁ d) : S_n a₁ d 9 = 0 :=
by
  sorry

end sum_first_nine_terms_eq_zero_l394_394520


namespace simplify_expression_l394_394989

theorem simplify_expression (x : ℝ) (hx : 0 ≤ x) :
  (Real.sqrt (50 * x) * Real.sqrt (18 * x) * Real.sqrt (98 * x) * Real.root 3 (250 * x^2)) =
  525 * x^2 * Real.sqrt (8 * x) * Real.root 3 (2 * x) :=
by
  sorry

end simplify_expression_l394_394989


namespace twenty_twenty_third_term_l394_394953

def sequence_denominator (n : ℕ) : ℕ :=
  2 * n - 1

def sequence_numerator_pos (n : ℕ) : ℕ :=
  (n + 1) / 2

def sequence_numerator_neg (n : ℕ) : ℤ :=
  -((n + 1) / 2 : ℤ)

def sequence_term (n : ℕ) : ℚ :=
  if n % 2 = 1 then 
    (sequence_numerator_pos n) / (sequence_denominator n) 
  else 
    (sequence_numerator_neg n : ℚ) / (sequence_denominator n)

theorem twenty_twenty_third_term :
  sequence_term 2023 = 1012 / 4045 := 
sorry

end twenty_twenty_third_term_l394_394953


namespace river_width_l394_394328

noncomputable def convertFlowRate (flow_rate_kmph : ℝ) : ℝ :=
  flow_rate_kmph * (1000 / 60)

theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) :
  depth = 2 → flow_rate_kmph = 5 → volume_per_minute = 7500 → 
  let flow_rate_m_per_min := convertFlowRate flow_rate_kmph in 
  let width := volume_per_minute / (depth * flow_rate_m_per_min) in
  width = 45 :=
by
  intros h_depth h_flow_rate h_volume_per_minute
  simp [convertFlowRate, h_depth, h_flow_rate, h_volume_per_minute]
  sorry

end river_width_l394_394328


namespace cosine_of_subtracted_vectors_l394_394473

variables {V : Type*} [inner_product_space ℝ V] (a b c : V)

-- Conditions
def condition1 := ∥a∥ = 1
def condition2 := ∥b∥ = 1
def condition3 := ∥c∥ = real.sqrt 2
def condition4 := a + b + c = 0

-- Proof statement
theorem cosine_of_subtracted_vectors 
  (h1 : condition1 a) 
  (h2 : condition2 b) 
  (h3 : condition3 c) 
  (h4 : condition4 a b c) : 
  inner_product_space.cos (a - c) (b - c) = 4 / 5 :=
sorry

end cosine_of_subtracted_vectors_l394_394473


namespace third_square_placed_is_G_l394_394757

-- Define the problem conditions and the proof goal
theorem third_square_placed_is_G 
  (A B C D E F G H : Prop)
  (formation : list Prop) 
  (E_is_last : formation.nth 7 = some E)
  (distinct_squares : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧
                      B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧
                      C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧
                      D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧
                      E ≠ F ∧ E ≠ G ∧ E ≠ H ∧
                      F ≠ G ∧ F ≠ H ∧
                      G ≠ H) :
  formation.nth 2 = some G :=
sorry

end third_square_placed_is_G_l394_394757


namespace min_distance_circle_to_line_l394_394810

def circle : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def line (p : ℝ × ℝ) : ℝ := 3 * p.1 + 4 * p.2 + 15

theorem min_distance_circle_to_line :
  ∃ (d : ℝ), (∀ (p : ℝ × ℝ), p ∈ circle → dist p (3, 4, 15) = d) ∧ d = 2 :=
sorry

end min_distance_circle_to_line_l394_394810


namespace Q_2_plus_Q_neg_2_l394_394090

noncomputable def cubic_polynomial (a b c k : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + k

theorem Q_2_plus_Q_neg_2 (a b c k : ℝ) 
  (h0 : cubic_polynomial a b c k 0 = k)
  (h1 : cubic_polynomial a b c k 1 = 3 * k)
  (hneg1 : cubic_polynomial a b c k (-1) = 4 * k) :
  cubic_polynomial a b c k 2 + cubic_polynomial a b c k (-2) = 22 * k :=
sorry

end Q_2_plus_Q_neg_2_l394_394090


namespace distinct_count_floor_squares_div_2000_l394_394403

theorem distinct_count_floor_squares_div_2000 : 
  ∀ n : ℕ, n ≤ 1000 → 
    (finset.range (1000 + 1)).map (λ n, ⌊(n : ℕ)^2 / 2000⌋).to_finset.card = 501 :=
by
  sorry

end distinct_count_floor_squares_div_2000_l394_394403


namespace intersection_of_M_and_N_l394_394038

open Set

theorem intersection_of_M_and_N :
  let M := {x : ℝ | log 2 x < 1}
  let M' := {x : ℝ | 0 < x ∧ x < 2}
  let N := {x : ℝ | x^2 - 1 ≤ 0}
  let N' := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
  M = M' ∧ N = N' →
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  intros h
  sorry

end intersection_of_M_and_N_l394_394038


namespace simplify_trig_expression_l394_394981

variable {x : ℝ}

theorem simplify_trig_expression (h : 1 + cos x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l394_394981


namespace rectangle_area_l394_394708

theorem rectangle_area (y : ℝ) (w : ℝ) : 
  (3 * w) ^ 2 + w ^ 2 = y ^ 2 → 
  3 * w * w = (3 / 10) * y ^ 2 :=
by
  intro h
  sorry

end rectangle_area_l394_394708


namespace common_difference_of_consecutive_multiples_l394_394199

/-- The sides of a rectangular prism are consecutive multiples of a certain number n. The base area is 450.
    Prove that the common difference between the consecutive multiples is 15. -/
theorem common_difference_of_consecutive_multiples (n d : ℕ) (h₁ : n * (n + d) = 450) : d = 15 :=
sorry

end common_difference_of_consecutive_multiples_l394_394199


namespace simplify_expression_l394_394991

theorem simplify_expression (x : ℝ) : 
  x - 2 * (1 + x) + 3 * (1 - x) - 4 * (1 + 2 * x) = -12 * x - 3 := 
by 
  -- Proof goes here
  sorry

end simplify_expression_l394_394991


namespace john_investment_in_bankA_l394_394077

-- Definitions to set up the conditions
def total_investment : ℝ := 1500
def bankA_rate : ℝ := 0.04
def bankB_rate : ℝ := 0.06
def final_amount : ℝ := 1575

-- Definition of the question to be proved
theorem john_investment_in_bankA (x : ℝ) (h : 0 ≤ x ∧ x ≤ total_investment) :
  (x * (1 + bankA_rate) + (total_investment - x) * (1 + bankB_rate) = final_amount) -> x = 750 := sorry


end john_investment_in_bankA_l394_394077


namespace cupcake_cost_l394_394390

def initialMoney : ℝ := 20
def moneyFromMother : ℝ := 2 * initialMoney
def totalMoney : ℝ := initialMoney + moneyFromMother
def costPerBoxOfCookies : ℝ := 3
def numberOfBoxesOfCookies : ℝ := 5
def costOfCookies : ℝ := costPerBoxOfCookies * numberOfBoxesOfCookies
def moneyAfterCookies : ℝ := totalMoney - costOfCookies
def moneyLeftAfterCupcakes : ℝ := 30
def numberOfCupcakes : ℝ := 10

noncomputable def costPerCupcake : ℝ := 
  (moneyAfterCookies - moneyLeftAfterCupcakes) / numberOfCupcakes

theorem cupcake_cost :
  costPerCupcake = 1.50 :=
by 
  sorry

end cupcake_cost_l394_394390


namespace pure_imaginary_solutions_l394_394748

noncomputable def polynomial := fun (x : ℂ) => x^4 - 4*x^3 + 10*x^2 - 40*x - 100

theorem pure_imaginary_solutions :
  {z : ℂ | polynomial z = 0 ∧ ∃ a : ℝ, z = a * complex.I} =
  {sqrt 10 * complex.I, -sqrt 10 * complex.I} :=
by
  sorry

end pure_imaginary_solutions_l394_394748


namespace possible_values_of_a_l394_394933

theorem possible_values_of_a (x y a : ℝ) (h1 : x + y = a) (h2 : x^3 + y^3 = a) (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 :=
by sorry

end possible_values_of_a_l394_394933


namespace cuboid_edge_sum_l394_394205

-- Define the properties of a cuboid
structure Cuboid (α : Type) [LinearOrderedField α] where
  length : α
  width : α
  height : α

-- Define the volume of a cuboid
def volume {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  c.length * c.width * c.height

-- Define the surface area of a cuboid
def surface_area {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

-- Define the sum of all edges of a cuboid
def edge_sum {α : Type} [LinearOrderedField α] (c : Cuboid α) : α :=
  4 * (c.length + c.width + c.height)

-- Given a geometric progression property
def gp_property {α : Type} [LinearOrderedField α] (c : Cuboid α) (q a : α) : Prop :=
  c.length = q * a ∧ c.width = a ∧ c.height = a / q

-- The main problem to be stated in Lean
theorem cuboid_edge_sum (α : Type) [LinearOrderedField α] (c : Cuboid α) (a q : α)
  (h1 : volume c = 8)
  (h2 : surface_area c = 32)
  (h3 : gp_property c q a) :
  edge_sum c = 32 := by
    sorry

end cuboid_edge_sum_l394_394205


namespace milk_production_days_l394_394499

variable (x : ℕ)
def cows := 2 * x
def cans := 2 * x + 2
def days := 2 * x + 1
def total_cows := 2 * x + 4
def required_cans := 2 * x + 10

theorem milk_production_days :
  (total_cows * required_cans) = ((2 * x) * (2 * x + 1) * required_cans) / ((2 * x + 2) * (2 * x + 4)) :=
sorry

end milk_production_days_l394_394499


namespace increase_area_is_correct_l394_394312

-- Define the dimensions of the rectangular garden
def length_rect : ℝ := 50
def width_rect : ℝ := 20

-- Define the perimeter (circumference of the new circular garden)
def perimeter : ℝ := 2 * (length_rect + width_rect)

-- Define the radius of the circular garden
def radius : ℝ := perimeter / (2 * Real.pi)

-- Define the area of the circular garden
def area_circle : ℝ := Real.pi * (radius ^ 2)

-- Define the area of the rectangular garden
def area_rect : ℝ := length_rect * width_rect

-- Define the increase in area
def increase_area : ℝ := area_circle - area_rect

-- The proof problem statement with the expected result
theorem increase_area_is_correct : increase_area = 560 := by
  sorry

end increase_area_is_correct_l394_394312


namespace bn_is_arithmetic_sequence_sum_first_n_terms_l394_394525

-- Given conditions
def a : ℕ → ℕ
def f (n : ℕ) : ℕ := a n
axiom a_1 : a 1 = 1
axiom recurrence (n : ℕ) : a (n + 1) - 2 * a n = 2^n

-- Define b_n in terms of a_n
def b (n : ℕ) : ℕ := a n / 2^(n - 1)

-- Define the sum S_n of the first n terms of a_n
def S (n : ℕ) : ℕ := ∑ k in Finset.range n, a (k + 1)

-- Prove that the sequence {b_n} is arithmetic
theorem bn_is_arithmetic_sequence : 
  ∀ n m : ℕ, (m ≠ 0) →  b (n + m) = b n + m :=
sorry

-- Prove the sum of first n terms
theorem sum_first_n_terms : 
  ∀ n : ℕ, (n ≠ 0) → S n = (n - 1) * 2^n + 1 :=
sorry

end bn_is_arithmetic_sequence_sum_first_n_terms_l394_394525


namespace cube_root_of_8_l394_394992

theorem cube_root_of_8 : (∛ 8 = 2) :=
by {
  -- Definition that 2 to the power of 3 equals 8 is implicit in the calculation of the cube root of 8
  sorry
}

end cube_root_of_8_l394_394992


namespace checkerboard_black_squares_l394_394364

theorem checkerboard_black_squares : 
  let n := 29 in
  (∃ f : ℕ × ℕ → Prop, 
    (∀ i j : ℕ, i < n → j < n → (i + j) % 2 = 0 ↔ f (i, j)) ∧ 
    (f (0, 0) ∧ f (0, 28) ∧ f (28, 0) ∧ f (28, 28))) →
  (∃ k : ℕ, k = 422 ∧ 
    (∀ f : ℕ × ℕ → Prop, 
      ((∀ i j : ℕ, i < 28 → j < 28 → (i + j) % 2 = 0 ↔ f (i, j)) → 
      ∃ num_black_squares : ℕ, 
      (num_black_squares = (392 + 15 + 15)) ∧ k = num_black_squares)) :=
  sorry

end checkerboard_black_squares_l394_394364


namespace number_of_arrangements_l394_394493

theorem number_of_arrangements : 
  let available_letters := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] in
  let first_letter := 'C' in
  let second_letter := 'D' in
  let remaining_letters := ['A', 'B', 'E', 'F', 'G', 'H'] in
  let third_letter_choices := 6 in
  let fourth_letter_choices := 5 in
  third_letter_choices * fourth_letter_choices = 30 
:= by sorry

end number_of_arrangements_l394_394493


namespace twelve_div_one_fourth_eq_48_l394_394355

theorem twelve_div_one_fourth_eq_48 : 12 / (1 / 4) = 48 := by
  -- We know that dividing by a fraction is equivalent to multiplying by its reciprocal
  sorry

end twelve_div_one_fourth_eq_48_l394_394355


namespace solution_set_log_ineq_l394_394200

theorem solution_set_log_ineq (x : ℝ) : 
  (log 2 ((x - 1) / x) >= 1) ↔ (-1 ≤ x ∧ x < 0) :=
begin
  sorry -- The proof is not provided here
end

end solution_set_log_ineq_l394_394200


namespace symmetric_scanning_codes_count_l394_394330

-- Definition of a symmetric 8x8 scanning code grid under given conditions
def is_symmetric_code (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∀ i j : Fin 8, grid i j = grid (7 - i) (7 - j) ∧ grid i j = grid j i

def at_least_one_each_color (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∃ i j k l : Fin 8, grid i j = true ∧ grid k l = false

def total_symmetric_scanning_codes : Nat :=
  1022

theorem symmetric_scanning_codes_count :
  ∀ (grid : Fin 8 → Fin 8 → Bool), is_symmetric_code grid ∧ at_least_one_each_color grid → 
  1022 = total_symmetric_scanning_codes :=
by
  sorry

end symmetric_scanning_codes_count_l394_394330


namespace sin_double_angle_l394_394411

theorem sin_double_angle (θ : ℝ) (hθ1 : |θ| < π / 2) (hθ2 : (cos θ / sin θ) = (5 / 3) * cos (2 * π - θ)) :
  sin (2 * θ) = 24 / 25 :=
by
  sorry

end sin_double_angle_l394_394411


namespace nine_div_zero_point_three_repeat_l394_394222

theorem nine_div_zero_point_three_repeat :
  ∀ (q : ℝ), q = (1 / 3) → (9 / q = 27) := by
  intros q hq
  rw [hq]
  norm_num
  sorry

end nine_div_zero_point_three_repeat_l394_394222


namespace eccentricity_of_hyperbola_is_sqrt5_l394_394805

noncomputable def hyperbola_eccentricity (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) : Prop :=
  let c := sqrt (a^2 + b^2) in
  let F := (c, 0) in
  let asymptote := λ x : ℝ, b / a * x in
  let F' := (b^2 - a^2) / c, -2 * a * b / c in
  ∃ e : ℝ, e = sqrt 5 ∧ (let x := F'.1, y := F'.2 in (x^2 / a^2 - y^2 / b^2 = 1))

theorem eccentricity_of_hyperbola_is_sqrt5 {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) :
  hyperbola_eccentricity a b a_pos b_pos :=
sorry

end eccentricity_of_hyperbola_is_sqrt5_l394_394805


namespace total_people_seated_l394_394151

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l394_394151


namespace interior_angle_of_regular_hexagon_is_120_degrees_l394_394253

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l394_394253


namespace problem_solution_l394_394787

theorem problem_solution (a b : ℤ) (h : {1, a, b / a} = {0, a^2, a + b}) : a ^ 2005 + b ^ 2005 = -1 := 
by 
  sorry

end problem_solution_l394_394787


namespace cubic_polynomial_q_l394_394617

/-- Given the rational function 1 / q(x) with vertical asymptotes at x = -1, x = 1, and x = 3,
and the cubic polynomial q(x) such that q(2) = 12, prove that q(x) = -4x^3 + 12x^2 + 4x - 12. -/
theorem cubic_polynomial_q (q : ℝ → ℝ) 
  (h₀ : ∀ x, q(x) = 0 → (x = -1 ∨ x = 1 ∨ x = 3)) 
  (h₁ : ∃ a, q(x) = a*(x + 1)*(x - 1)*(x - 3)) 
  (h₂ : q(2) = 12) : 
  q(x) = -4*x^3 + 12*x^2 + 4*x - 12 := 
begin
  sorry
end

end cubic_polynomial_q_l394_394617


namespace count_diff_squares_plus_three_l394_394030

theorem count_diff_squares_plus_three:
  (finset.filter (λ n, ∃ a b : ℕ, n = (a + b) ^ 2 - b ^ 2 + 3)
    (finset.range 501)).card = 373 :=
by {
  sorry
}

end count_diff_squares_plus_three_l394_394030


namespace sum_of_b_l394_394896

-- Defining the geometric sequence {a_n} with given initial condition and common ratio
def a₁ : ℕ := 2
def q : ℕ := 2
def a (n : ℕ) : ℕ := 2^n

-- Defining the sum of first n terms S_n
def S (n : ℕ) : ℕ := 2 * (2^n - 1)

-- Defining the sequence b_n
def b (n : ℕ) : ℚ := (a (n + 1) : ℚ) / ((S n) * (S (n + 1)))

-- Defining the sum T_n of the first n terms of b_n
def T (n : ℕ) : ℚ := (Finset.range n).sum (λ k, b k)

-- The main theorem we are required to prove
theorem sum_of_b (n : ℕ) : T n = (1 / 2) - (1 / (2 ^ (n + 2) - 2)) :=
by sorry

end sum_of_b_l394_394896


namespace min_integer_value_expr_l394_394862

theorem min_integer_value_expr (x : ℝ) : 
  ∃ (y : ℤ), y = -15 ∧ (∀ z : ℤ, z ≤ y → f(x) ≤ f(z)) 
where
  f(x) := (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 3) :=
begin
  sorry,
end

end min_integer_value_expr_l394_394862


namespace angle_PCD_l394_394517

/-- In parallelogram ABCD, if ∠BAD = 76°, side AD has midpoint P, and ∠PBA = 52°, 
    then ∠PCD = 38°. -/
theorem angle_PCD (A B C D P : Type) [Parallelogram A B C D] 
  (h1 : angle A B = 76) 
  (h2 : is_midpoint P A D)
  (h3 : angle P B A = 52) :
  angle P C D = 38 :=
sorry

end angle_PCD_l394_394517


namespace randy_blocks_left_l394_394118

theorem randy_blocks_left 
  (initial_blocks : ℕ := 78)
  (used_blocks : ℕ := 19)
  (given_blocks : ℕ := 25)
  (bought_blocks : ℕ := 36)
  (sets_from_sister : ℕ := 3)
  (blocks_per_set : ℕ := 12) :
  (initial_blocks - used_blocks - given_blocks + bought_blocks + (sets_from_sister * blocks_per_set)) / 2 = 53 := 
by
  sorry

end randy_blocks_left_l394_394118


namespace top_600_minimum_income_l394_394179

noncomputable def minimum_income (N: ℝ) (b: ℝ) (x: ℝ) : Prop :=
  N = b * x ^ (-2)

theorem top_600_minimum_income :
  ∃ x: ℝ, minimum_income 600 (6 * 10 ^ 9) x ∧ x = 10 ^ (3.5) :=
by
  sorry

end top_600_minimum_income_l394_394179


namespace smallest_seating_l394_394316

theorem smallest_seating (N : ℕ) (h: ∀ (chairs : ℕ) (occupants : ℕ), 
  chairs = 100 ∧ occupants = 25 → 
  ∃ (adjacent_occupied: ℕ), adjacent_occupied > 0 ∧ adjacent_occupied < chairs ∧
  adjacent_occupied ≠ occupants) : 
  N = 25 :=
sorry

end smallest_seating_l394_394316


namespace find_alpha_beta_sum_l394_394570

noncomputable def alpha_beta (α β : ℝ) : Prop :=
  0 < α ∧ α < (π / 2) ∧ 0 < β ∧ β < (π / 2) ∧ 
  tan α = 1 / 7 ∧ tan β = 1 / 3

theorem find_alpha_beta_sum (α β : ℝ) (h : alpha_beta α β) : α + 2 * β = π / 4 :=
sorry

end find_alpha_beta_sum_l394_394570


namespace player2_winning_strategy_l394_394335

-- Definitions of the game setup
def initial_position_player1 := (1, 1)
def initial_position_player2 := (998, 1998)

def adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 - 1 ∨ p1.2 = p2.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 - 1 ∨ p1.1 = p2.1 + 1))

-- A function defining the winning condition for Player 2
def player2_wins (p1 p2 : ℕ × ℕ) : Prop :=
  p1 = p2 ∨ p1.1 = (initial_position_player2.1)

-- Theorem stating the pair (998, 1998) guarantees a win for Player 2
theorem player2_winning_strategy : player2_wins (998, 0) (998, 1998) :=
sorry

end player2_winning_strategy_l394_394335


namespace find_k_l394_394459

/--
Given a system of linear equations:
1) x + 2 * y = -a + 1
2) x - 3 * y = 4 * a + 6
If the expression k * x - y remains unchanged regardless of the value of the constant a, 
show that k = -1.
-/
theorem find_k 
  (a x y k : ℝ) 
  (h1 : x + 2 * y = -a + 1) 
  (h2 : x - 3 * y = 4 * a + 6)
  (h3 : ∀ a₁ a₂ x₁ x₂ y₁ y₂, (x₁ + 2 * y₁ = -a₁ + 1) → (x₁ - 3 * y₁ = 4 * a₁ + 6) → 
                               (x₂ + 2 * y₂ = -a₂ + 1) → (x₂ - 3 * y₂ = 4 * a₂ + 6) → 
                               (k * x₁ - y₁ = k * x₂ - y₂)) : 
  k = -1 :=
  sorry

end find_k_l394_394459


namespace smallest_positive_period_of_f_triangle_side_range_l394_394818

noncomputable def f (x : ℝ) : ℝ := cos x * (sqrt 3 * sin x - cos x)

theorem smallest_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
by
  use π
  sorry

variables {A B C a b c : ℝ}

theorem triangle_side_range (hB : f B = 1/2) (ha_c : a + c = 1) :
  b ∈ Set.Ico (1/2:ℝ) 1 :=
by
  have hB_pi3 : B = π / 3 := sorry
  have h_cosB : cos B = 1 / 2 := sorry
  have h_b2 : b^2 = 3 * (a - 1 / 2)^2 + 1 / 4 := sorry
  have h_range : ∀ a, 0 < a → a < 1 → (1 / 4 ≤ 3 * (a - 1 / 2)^2 + 1 / 4 ∧ 3 * (a - 1 / 2)^2 + 1 / 4 < 1) := sorry
  exact sorry

end smallest_positive_period_of_f_triangle_side_range_l394_394818


namespace correct_proposition_l394_394660

-- Definitions of the propositions as boolean valued statements
def proposition_A : Prop :=
  ∃ (points : set (euclidean_space 3)), points.card = 3 ∧ (∀ p₁ p₂ p₃ ∈ points, collinear ℝ {p₁, p₂, p₃} → ¬plane_of_points ℝ {p₁, p₂, p₃})

def proposition_B : Prop :=
  ∀ (l₁ l₂ : euclidean_space 3), is_skew ℝ l₁ l₂ → 0 < angle ℝ l₁ l₂ ∧ angle ℝ l₁ l₂ ≤ π / 2

def proposition_C : Prop :=
  ∀ (angle₁ angle₂ : angle ℝ), (∀ (a₁ b₁ : ℝ), a₁ ≠ 0 ∧ b₁ ≠ 0 → ∥angle₁.a∥ = ∥angle₂.a∥ ∧ ∥angle₁.b∥ = ∥angle₂.b∥ → angle₁ = angle₂)

def proposition_D : Prop :=
  ∀ (l : line (euclidean_space 3)) (α : plane (euclidean_space 3)),
    (∀ l' ∈ (lines_in_plane α), parallel ℝ l l') → parallel ℝ l α

-- Definition of the proof that proposition B is correct
theorem correct_proposition : proposition_B :=
by sorry  -- Proof


end correct_proposition_l394_394660


namespace photographer_max_photos_l394_394334

-- The initial number of birds of each species
def total_birds : ℕ := 20
def starlings : ℕ := 8
def wagtails : ℕ := 7
def woodpeckers : ℕ := 5

-- Define a function to count the remaining birds of each species after n photos
def remaining_birds (n : ℕ) (species : ℕ) : ℕ := species - (if species ≤ n then species else n)

-- Define the main theorem we want to prove
theorem photographer_max_photos (n : ℕ) (h1 : remaining_birds n starlings ≥ 4) (h2 : remaining_birds n wagtails ≥ 3) : 
  n ≤ 7 :=
by
  sorry

end photographer_max_photos_l394_394334


namespace interval_width_and_count_l394_394303

def average_income_intervals := [3000, 4000, 5000, 6000, 7000]
def frequencies := [5, 9, 4, 2]

theorem interval_width_and_count:
  (average_income_intervals[1] - average_income_intervals[0] = 1000) ∧
  (frequencies.length = 4) :=
by
  sorry

end interval_width_and_count_l394_394303


namespace range_of_a_l394_394447

def f (a x : ℝ) : ℝ := (Real.log x - 2 * a * x) / x

def g (x : ℝ) : ℝ := Real.log x / x

theorem range_of_a : 
  ∃! k : ℤ, f a k > 1 → (1 / 4 * Real.log 2 - 1 / 2 ≤ a ∧ a < 1 / 6 * Real.log 3 - 1 / 2) :=
sorry

end range_of_a_l394_394447


namespace algorithm_must_have_sequential_structure_l394_394174

-- Definitions for types of structures used in algorithm definitions.
inductive Structure
| Logical
| Selection
| Loop
| Sequential

-- Predicate indicating whether a given Structure is necessary for any algorithm.
def necessary (s : Structure) : Prop :=
  match s with
  | Structure.Logical => False
  | Structure.Selection => False
  | Structure.Loop => False
  | Structure.Sequential => True

-- The theorem statement to prove that the sequential structure is necessary for any algorithm.
theorem algorithm_must_have_sequential_structure :
  necessary Structure.Sequential :=
by
  sorry

end algorithm_must_have_sequential_structure_l394_394174


namespace minimize_sum_AB_AC_l394_394738

-- Definitions and conditions
variables {E D F B C A G: Point}
-- Assume points and angle exist with the appropriate properties

-- Suppose G is the reflection of F in (ED)
-- Reflect C to get C'
noncomputable def reflect_point (P Q R: Point) : Point := sorry

noncomputable def optimal_position : Prop :=
  let C' := reflect_point C E D in
  let BC' := line_through B C' in
  A ∈ (ED) ∧ A ∈ BC'

-- Main theorem
theorem minimize_sum_AB_AC {A : Point} :
  ∃ G C', is_reflection G F (ED) ∧ is_reflection C' C (ED) ∧
  optimal_position A :=
by {
  sorry
}

end minimize_sum_AB_AC_l394_394738


namespace flip_ratio_l394_394536

theorem flip_ratio (jen_triple_flips tyler_double_flips : ℕ)
  (hjen : jen_triple_flips = 16)
  (htyler : tyler_double_flips = 12)
  : 2 * tyler_double_flips / 3 * jen_triple_flips = 1 / 2 := 
by
  rw [hjen, htyler]
  norm_num
  sorry

end flip_ratio_l394_394536


namespace domain_of_f_l394_394767

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 + 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f y = x} = {x : ℝ | x ∉ {-3, -1}} :=
by
  sorry

end domain_of_f_l394_394767


namespace degree_measure_of_regular_hexagon_interior_angle_l394_394285

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l394_394285


namespace age_of_son_l394_394666

theorem age_of_son (S M : ℕ) (h1 : M = S + 28) (h2 : M + 2 = 2 * (S + 2)) : S = 26 := by
  sorry

end age_of_son_l394_394666


namespace Mike_cards_before_birthday_l394_394102

theorem Mike_cards_before_birthday (total_cards_now cards_received : ℕ) (h1 : total_cards_now = 82) (h2 : cards_received = 18) :
  total_cards_now - cards_received = 64 :=
by
  rw [h1, h2]
  exact rfl

end Mike_cards_before_birthday_l394_394102


namespace area_of_closed_region_l394_394899

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * Real.sin (a * x) + Real.cos (a * x)
def g (a : ℝ) : ℝ := Real.sqrt (a^2 + 1)

-- Define the conditions and the period
def period (a : ℝ) : ℝ := 2 * Real.pi / a

-- Main theorem statement
theorem area_of_closed_region (a : ℝ) (h : a > 0) :
  ∫ x in 0..period a, g a - f a x = (2 * Real.pi * Real.sqrt (a^2 + 1)) / a :=
by
  sorry

end area_of_closed_region_l394_394899


namespace cotangent_identity_l394_394530

theorem cotangent_identity
  (a b c : ℝ)
  (h : a^2 + b^2 = 2019 * c^2)
  (triangle_ineq : ∀ x y z : ℝ, x + y > z ∧ x + z > y ∧ y + z > x) : 
  ∀ A B C : ℝ, 
  (triangle_ineq a b c) →
  (triangle_ineq c a b) →
  (triangle_ineq b c a) →
  (A = real.angle a b c) →
  (B = real.angle b c a) →
  (C = real.angle c a b) →
  (tan A ≠ 0) →
  (tan B ≠ 0) →
  (tan C ≠ 0) →
  (cot C / (cot A + cot B) = 1009) := sorry

end cotangent_identity_l394_394530


namespace tetrahedron_four_points_cover_tetrahedron_three_points_not_cover_l394_394799

theorem tetrahedron_four_points_cover (T : Tetrahedron) (h : T.is_regular) (h_edges : ∀ e ∈ T.edges, e.length = 1) :
  ∃ (A B C D : T.Point), ∀ (p : T.Point) (h_mem : p ∈ T.surface), ∃ q ∈ {A, B, C, D}, T.distance p q ≤ 0.5 := sorry

theorem tetrahedron_three_points_not_cover (T : Tetrahedron) (h : T.is_regular) (h_edges : ∀ e ∈ T.edges, e.length = 1) :
  ¬ (∃ (A B C : T.Point), ∀ (p : T.Point) (h_mem : p ∈ T.surface), ∃ q ∈ {A, B, C}, T.distance p q ≤ 0.5) := sorry

end tetrahedron_four_points_cover_tetrahedron_three_points_not_cover_l394_394799


namespace regular_hexagon_interior_angle_l394_394263

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l394_394263


namespace segment_ratio_proof_l394_394107

variable {A B C P P1 Q Q1 D : Point}
variable 
  (hPab : SameLine A B P)
  (hP1ab : SameLine A B P1)
  (hQac : SameLine A Q C)
  (hQ1ac : SameLine A Q1 C)
  (hD : Collinear A (Line.intersection (Line.mk P Q) (Line.mk P1 Q1)) D)
  (collinear_D_BC : Collinear D B C)

theorem segment_ratio_proof
  (hBD : line_intersect (segment B D) (segment D C)) :
  ratio (BD, CD) = 
  (ratio (BP, PA) - ratio (BP1, P1A)) / 
  (ratio (CQ, QA) - ratio (CQ1, Q1A)) :=
sorry

end segment_ratio_proof_l394_394107


namespace triangles_congruent_l394_394212

variables {O1 O2 O3 A1 A2 A3 K : Type}
variables [circle O1] [circle O2] [circle O3]
variables (h1 : intersect O1 O2 O3 = K)
variables (h2 : other_points_of_intersection O1 O2 O3 = (A1, A2, A3))

theorem triangles_congruent :
  congruent (triangle O1 O2 O3) (triangle A1 A2 A3) :=
sorry

end triangles_congruent_l394_394212


namespace correct_quadratic_equation_l394_394657

-- The main statement to prove.
theorem correct_quadratic_equation :
  (∀ (x y a : ℝ), (3 * x + 2 * y - 1 ≠ 0) ∧ (5 * x^2 - 6 * y - 3 ≠ 0) ∧ (a * x^2 - x + 2 ≠ 0) ∧ (x^2 - 1 = 0) → (x^2 - 1 = 0)) :=
by
  sorry

end correct_quadratic_equation_l394_394657


namespace x_lt_y_l394_394566

theorem x_lt_y (n : ℕ) (h_n : n > 2) (x y : ℝ) (h_x_pos : 0 < x) (h_y_pos : 0 < y) (h_x : x ^ n = x + 1) (h_y : y ^ (n + 1) = y ^ 3 + 1) : x < y :=
sorry

end x_lt_y_l394_394566


namespace area_of_triangle_OAB_l394_394427

theorem area_of_triangle_OAB {a : Real} (h₁ : a ≠ 0) (h₂ : a > 0) :
  let center := (a, (2 / a))
  let r := Real.sqrt (a^2 + (2 / a)^2)
  let A := (a, 0)
  let B := (0, 4 / a)
  let area := 1/2 * Real.abs a * Real.abs (4 / a)
  area = 4 :=
sorry

end area_of_triangle_OAB_l394_394427


namespace total_students_l394_394053

variable (A B AB : ℕ)

-- Conditions
axiom h1 : AB = (1 / 5) * (A + AB)
axiom h2 : AB = (1 / 4) * (B + AB)
axiom h3 : A - B = 75

-- Proof problem
theorem total_students : A + B + AB = 600 :=
by
  sorry

end total_students_l394_394053


namespace people_at_table_l394_394128

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l394_394128


namespace regular_hexagon_interior_angle_l394_394256

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l394_394256


namespace cos_angle_sub_vectors_l394_394481

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (H1 : ∥a∥ = 1) (H2 : ∥b∥ = 1) (H3 : ∥c∥ = real.sqrt 2)
variables (H4 : a + b + c = 0)

theorem cos_angle_sub_vectors :
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry

end cos_angle_sub_vectors_l394_394481


namespace count_two_digit_numbers_with_seven_l394_394496

theorem count_two_digit_numbers_with_seven : 
  ∃ count : ℕ, count = 18 ∧ 
  (count = (set.count (λ n, (10 ≤ n ∧ n ≤ 99) ∧ 
    (n % 10 = 7 ∨ (n / 10) % 10 = 7)) (set.Icc 10 99))) :=
by
  sorry

end count_two_digit_numbers_with_seven_l394_394496


namespace regular_hexagon_interior_angle_measure_l394_394232

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l394_394232


namespace minimum_distance_from_P_to_Q_l394_394006

-- Definitions based on the given problem’s conditions
def P_on_curve (P : ℝ × ℝ) : Prop := P.1 = 2

def Q : ℝ × ℝ := (1/2, real.sqrt 3 / 2)

-- Statement of the proof problem
theorem minimum_distance_from_P_to_Q :
  ∀ P : ℝ × ℝ, P_on_curve P → dist P Q = 3 / 2 :=
by sorry

end minimum_distance_from_P_to_Q_l394_394006


namespace div_by_frac_eq_mult_multiply_12_by_4_equals_48_l394_394357

theorem div_by_frac_eq_mult (a b : ℝ) (hb : b ≠ 0) : a / (1 / b) = a * b :=
by
  sorry

theorem multiply_12_by_4_equals_48 : 12 * 4 = 48 :=
by
  have h : 12 / (1 / 4) = 12 * 4 := div_by_frac_eq_mult 12 4 (by norm_num)
  exact h.trans (by norm_num)

end div_by_frac_eq_mult_multiply_12_by_4_equals_48_l394_394357


namespace sector_area_is_correct_l394_394359

/-- The radius of a circle with a diameter of 10 meters. -/
def radius : ℝ := 5

/-- The central angle in radians of 60 degrees. -/
def theta_rad : ℝ := (60 * Real.pi) / 180

/-- The area of the circle with the given radius. -/
def area_circle : ℝ := Real.pi * (radius ^ 2)

/-- The expected area of the sector. -/
def expected_area_sector : ℝ := (theta_rad / (2 * Real.pi)) * area_circle

/-- Prove the area of a sector of a circle with a diameter of 10 meters and a central angle of 60 degrees is equal to 25π/6. -/
theorem sector_area_is_correct : expected_area_sector = 25 * Real.pi / 6 :=
by
  sorry

end sector_area_is_correct_l394_394359


namespace alpha_third_quadrant_l394_394853

theorem alpha_third_quadrant (α : ℝ) :
  ∀ α, sin α < 0 ∧ tan α > 0 → (∃ k : ℤ, π + k * 2 * π < α ∧ α < 3 * π / 2 + k * 2 * π) :=
begin
  sorry
end

end alpha_third_quadrant_l394_394853


namespace total_amount_paid_is_correct_l394_394783

-- Define the initial conditions
def tireA_price : ℕ := 75
def tireA_discount : ℕ := 20
def tireB_price : ℕ := 90
def tireB_discount : ℕ := 30
def tireC_price : ℕ := 120
def tireC_discount : ℕ := 45
def tireD_price : ℕ := 150
def tireD_discount : ℕ := 60
def installation_fee : ℕ := 15
def disposal_fee : ℕ := 5

-- Calculate the total amount paid
def total_paid : ℕ :=
  let tireA_total := (tireA_price - tireA_discount) + installation_fee + disposal_fee
  let tireB_total := (tireB_price - tireB_discount) + installation_fee + disposal_fee
  let tireC_total := (tireC_price - tireC_discount) + installation_fee + disposal_fee
  let tireD_total := (tireD_price - tireD_discount) + installation_fee + disposal_fee
  tireA_total + tireB_total + tireC_total + tireD_total

-- Statement of the theorem
theorem total_amount_paid_is_correct :
  total_paid = 360 :=
by
  -- proof goes here
  sorry

end total_amount_paid_is_correct_l394_394783


namespace find_q_r_s_l394_394080

noncomputable def is_valid_geometry 
  (AD : ℝ) (AL : ℝ) (AM : ℝ) (AN : ℝ) (q : ℕ) (r : ℕ) (s : ℕ) : Prop :=
  AD = 10 ∧ AL = 3 ∧ AM = 3 ∧ AN = 3 ∧ ¬(∃ p : ℕ, p^2 ∣ s)

theorem find_q_r_s : ∃ (q r s : ℕ), is_valid_geometry 10 3 3 3 q r s ∧ q + r + s = 711 :=
by
  sorry

end find_q_r_s_l394_394080


namespace area_of_extended_triangle_l394_394601

noncomputable def area_of_triangle_A'B'C' (ABC : Type) [triangle ABC] 
  (equilateral : Equilateral ABC)
  (side_length : 2)
  (A B C : Point)
  (ABKL B C MN CA OP : Square)
  (lines_extended : ∃ A' B' C' : Point, 
    is_extended (KL, MN, OP) (A', B', C')) : ℝ :=
  12 + 13 * Real.sqrt 3

theorem area_of_extended_triangle (ABC : Type) [triangle ABC]
  (equilateral : Equilateral ABC)
  (side_length : eq side_length 2)
  (squares_drawn : DrawnExternally ABC ABKL B C MN CA OP)
  (KL MN OP : LineSegment) (lines_extended : Extends KL MN OP) : 
  area_of_triangle_A'B'C' ABC equilateral side_length KL MN OP = 12 + 13 * Real.sqrt 3 := 
sorry

end area_of_extended_triangle_l394_394601


namespace area_of_quad_with_circles_l394_394581

variables {r a b : ℝ}

-- Definitions of the conditions provided in the problem statement
def convex_quad_with_circles (ABCD : quadrilateral) :=
  ∃ (A B C D : Point),
    is_convex ABCD ∧
    (tangent (circle_with_rad r (center_on_diag A C)) (side_AB ABCD)) ∧
    (tangent (circle_with_rad r (center_on_diag A C)) (side_AD ABCD)) ∧
    (tangent (circle_with_rad r (center_on_diag A C)) (side_BC ABCD)) ∧
    (tangent (circle_with_rad r (center_on_diag B D)) (side_BC ABCD)) ∧
    (tangent (circle_with_rad r (center_on_diag B D)) (side_CD ABCD)) ∧
    (tangent (circle_with_rad r (center_on_diag B D)) (side_AD ABCD)) ∧
    (externally_tangent (circle_with_rad r (center_on_diag A C)) (circle_with_rad r (center_on_diag B D)))

-- The actual theorem to be proved
theorem area_of_quad_with_circles (ABCD : quadrilateral) 
  (h : convex_quad_with_circles ABCD) : 
  (area ABCD) = 4 * r^2 * (sqrt 2 + 1) :=
sorry -- Proof not required

end area_of_quad_with_circles_l394_394581


namespace integral_ln_sin_eq_l394_394220

theorem integral_ln_sin_eq :
  ∫ x in 0..(Real.pi / 2), Real.log (Real.sin x) = - (Real.pi / 2) * Real.log 2 := by
sory

end integral_ln_sin_eq_l394_394220


namespace max_radius_sum_l394_394214

theorem max_radius_sum (r : ℝ) :
  let r_sq := (r^2)
  (3 : ℝ) = real.sqrt 9 ∧
  (8 : ℝ) = real.sqrt 64 ∧ 
  (∀ (x : ℝ), x > r → false) →
  r_sq = (225/73) →
  225 + 73 = 298 :=
by
  intros r_sq h₁ h₂ r_max r_eq
  have h_congruent := h₁.1
  have h_height := h₁.2
  have r_in_cones := h₂ r
  rw r_eq at r_sq
  exact r_impl
  sorry

end max_radius_sum_l394_394214


namespace jane_jill_dolls_l394_394910

theorem jane_jill_dolls : 
  let jane_dolls := 13 in
  let jill_dolls := jane_dolls + 6 in
  jane_dolls + jill_dolls = 32 :=
by
  let jane_dolls := 13
  let jill_dolls := jane_dolls + 6
  show jane_dolls + jill_dolls = 32
  sorry

end jane_jill_dolls_l394_394910


namespace sum_of_factors_of_30_l394_394296

theorem sum_of_factors_of_30 : 
  let n := 30 in sum (filter (λ d, n % d = 0) (list.range (n + 1))) = 72 :=
by 
  let n := 30
  sorry

end sum_of_factors_of_30_l394_394296


namespace minimum_cans_required_l394_394756

theorem minimum_cans_required (liters_oz : ℝ) (can_oz : ℕ) (liters_needed : ℝ) : ℕ :=
  let total_oz_needed := liters_needed * liters_oz in
  let cans_needed := total_oz_needed / can_oz in
  if cans_needed > (total_oz_needed / can_oz).floor then cans_needed.floor.to_nat + 1 else cans_needed.floor.to_nat

example : minimum_cans_required 33.814 15 3.8 = 9 := sorry

end minimum_cans_required_l394_394756


namespace sequence_property_l394_394418

variable {n : ℕ}

def sequence (a : ℕ → ℕ) :=
  ∀ (S_n : ℕ),
  (2 * S_n = a (n + 1) - 2^(n+1) + 1) → 
  (a 1 = 1) →
  (a 2 = 5) →
  (a 1, a 2 + 5, a 3 G : ∃ k, a 1 =k a 2 + 5 k a 3)→ 
  a n = 3^n - 2^n

theorem sequence_property :
  sequence (λ n : ℕ, if n = 1 then 1 else if n = 2 then 5 
              else 3^(n) - 2^(n)) := 
by {
  -- Proof skipped with sorry.
  sorry
}

end sequence_property_l394_394418


namespace cookies_not_eaten_l394_394958

def totalCookies : ℕ := 200
def wifePercentage : ℕ := 30
def daughterTakes : ℕ := 40

theorem cookies_not_eaten :
  let wifeTakes := (wifePercentage * totalCookies) / 100 in
  let remainingAfterWife := totalCookies - wifeTakes in
  let remainingAfterDaughter := remainingAfterWife - daughterTakes in
  let javierEats := remainingAfterDaughter / 2 in
  remainingAfterDaughter - javierEats = 50 := by
  sorry

end cookies_not_eaten_l394_394958


namespace expenditure_increase_36_percent_l394_394317

theorem expenditure_increase_36_percent
  (m : ℝ) -- mass of the bread
  (p_bread : ℝ) -- price of the bread
  (p_crust : ℝ) -- price of the crust
  (h1 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h2 : p_crust = 1.2 * p_bread) -- condition: crust is 20% more expensive
  (h3 : ∃ (m_crust : ℝ), m_crust = 0.75 * m) -- condition: crust is 25% lighter in weight
  (h4 : ∃ (m_consumed_bread : ℝ), m_consumed_bread = 0.85 * m) -- condition: 15% of bread dries out
  (h5 : ∃ (m_consumed_crust : ℝ), m_consumed_crust = 0.75 * m) -- condition: crust is consumed completely
  : (17 / 15) * (1.2 : ℝ) = 1.36 := 
by sorry

end expenditure_increase_36_percent_l394_394317


namespace smallest_prime_6_less_than_perfect_square_l394_394292

/-
The problem statement:
Prove that the smallest positive number that is prime and 6 less than a perfect square is 3.
-/

theorem smallest_prime_6_less_than_perfect_square : 
  ∃ (n : ℕ), nat.prime n ∧ ∃ (k : ℕ), n = k^2 - 6 ∧ n = 3 :=
by
  sorry

end smallest_prime_6_less_than_perfect_square_l394_394292


namespace trapezoid_angles_and_area_l394_394588

-- Define the conditions in Lean 4

variables (O K : Point) (A B C D M Q : Point)
variables (r : ℝ)
variables (alpha : Real) -- Angle in radians
variables (AD DK AO OK AM MD PD OP BQ AB S_ABCD : ℝ)

-- Given conditions: Problem setup
axiom is_center_of_incircle (O : Point) (ABCD : IsoscelesTrapezoid) : CircleCenteredAt(O).InscribedInTrapezoid(ABCD)
axiom parallelBCAD (BC AD : Line) : BC ∥ AD
axiom AO_intersects_CD_at_K (AO : Line) (K CD : Point) : AO.IntersectsAt(K)
axiom AO_OK_values : AO = 5 ∧ OK = 3

-- Define the problem statement in Lean 4
theorem trapezoid_angles_and_area :
  (α = arccos (1 / 3)) ∧ (S_ABCD = 25 * sqrt 2) :=
by
  -- The proof is omitted
  sorry

end trapezoid_angles_and_area_l394_394588


namespace P7_eq_P1_l394_394957

theorem P7_eq_P1
  (A B C A1 B1 C1 P1 : Type)
  [Point A] [Point B] [Point C]
  [Line BC AB CA P1B1 P2A1 P3C1]
  [On A1 BC] [On B1 CA] [On C1 AB]
  [On P1 BC]
  (P2 : Type) [Intersection P1B1 AB P2]
  (P3 : Type) [Intersection P2A1 CA P3]
  (P4 : Type) [Intersection P3C1 BC P4]
  (P5 : Type) [Intersection P4B1 AB P5]
  (P6 : Type) [Intersection P5A1 CA P6]
  (P7 : Type) [Intersection P6C1 BC P7] :
  P7 = P1 :=
sorry

end P7_eq_P1_l394_394957


namespace order_of_abc_l394_394413

noncomputable def a : ℝ := 4 ^ real.log 4.1 / real.log 3
noncomputable def b : ℝ := 4 ^ real.log 2.7 / real.log 3
noncomputable def c : ℝ := (1 / 2) ^ real.log 0.1 / real.log 3

theorem order_of_abc : a > c ∧ c > b := by
  have ha : a = 2 ^ (real.log 4.1 ^ 2 / real.log 3) := by sorry
  have hb : b = 2 ^ (real.log 2.7 ^ 2 / real.log 3) := by sorry
  have hc : c = 2 ^ (real.log 10 / real.log 3) := by sorry
  have h1 : real.log 4.1 ^ 2 > real.log 10 := by sorry
  have h2 : real.log 10 > real.log 2.7 ^ 2 := by sorry
  exact ⟨ha.trans_lt (exp_base_increasing h1), hc.trans_lt (exp_base_increasing h2)⟩

end order_of_abc_l394_394413


namespace cosine_identity_l394_394468

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) 

theorem cosine_identity 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (hc : ∥c∥ = real.sqrt 2)
  (habc : a + b + c = 0) :
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
sorry

end cosine_identity_l394_394468


namespace exists_five_positive_integers_sum_20_product_420_l394_394943
-- Import the entirety of Mathlib to ensure all necessary definitions are available

-- Lean statement for the proof problem
theorem exists_five_positive_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ a + b + c + d + e = 20 ∧ a * b * c * d * e = 420 :=
sorry

end exists_five_positive_integers_sum_20_product_420_l394_394943


namespace sand_weight_proof_l394_394696

-- Definitions for the given conditions
def side_length : ℕ := 40
def bag_weight : ℕ := 30
def area_per_bag : ℕ := 80

-- Total area of the sandbox
def total_area := side_length * side_length

-- Number of bags needed
def number_of_bags := total_area / area_per_bag

-- Total weight of sand needed
def total_weight := number_of_bags * bag_weight

-- The proof statement
theorem sand_weight_proof :
  total_weight = 600 :=
by
  sorry

end sand_weight_proof_l394_394696


namespace negation_equivalence_l394_394185

-- Define the propositions
def proposition (a b : ℝ) : Prop := a > b → a + 1 > b

def negation_proposition (a b : ℝ) : Prop := a ≤ b → a + 1 ≤ b

-- Statement to prove
theorem negation_equivalence (a b : ℝ) : ¬(proposition a b) ↔ negation_proposition a b := 
sorry

end negation_equivalence_l394_394185


namespace range_of_x_in_fourth_quadrant_l394_394887

theorem range_of_x_in_fourth_quadrant :
  ∀ x : ℝ, (2 * x + 6 > 0 ∧ 5 * x < 0) ↔ (-3 < x ∧ x < 0) :=
by
  intros x
  split
  { intro h
    cases h with h1 h2
    split
    { linarith }
    { linarith } }
  { intro h
    cases h with h1 h2
    split
    { linarith }
    { linarith } }

end range_of_x_in_fourth_quadrant_l394_394887


namespace roots_of_polynomial_l394_394404

theorem roots_of_polynomial :
  (Polynomial.roots (Polynomial.C 8 * Polynomial.X ^ 4 + Polynomial.C 26 * Polynomial.X ^ 3 
    - Polynomial.C 66 * Polynomial.X ^ 2 + Polynomial.C 24 * Polynomial.X)).to_finset = 
  {0, 1/2, 3/2, -4} :=
sorry

end roots_of_polynomial_l394_394404


namespace non_negative_solution_count_positive_solution_count_l394_394739

open Nat

-- Problem 1: Non-negative integer solutions
theorem non_negative_solution_count (N n : ℕ) (hN : N ≥ 1) (hn : n ≥ 1) :
    (∑ (i : Fin N), n.toNat) ≤ n → (Nat.choose (n + N) n) = (Nat.coe choose n n hN hn) :=
sorry

-- Problem 2: Positive integer solutions
theorem positive_solution_count (N n : ℕ) (hN : N ≥ 1) (hn : n ≥ 1) :
    (∑ (i : Fin N), (n.toNat + 1)) ≤ n → (Nat.choose (n - 1) (N - 1)) = (Nat.coe choose n n hn hN) :=
sorry

end non_negative_solution_count_positive_solution_count_l394_394739


namespace decreasing_function_range_l394_394379

theorem decreasing_function_range (k : ℝ) : (∀ x : ℝ, k + 2 < 0) ↔ k < -2 :=
by
  sorry

end decreasing_function_range_l394_394379


namespace max_f_of_polynomial_l394_394561

theorem max_f_of_polynomial (f : ℝ → ℝ)
    (hf_nonneg : ∀ x, 0 ≤ f x)
    (h_poly : ∃ p : Polynomial ℝ, ∀ x, f x = Polynomial.eval x p)
    (h1 : f 4 = 16)
    (h2 : f 16 = 512) :
    f 8 ≤ 64 :=
by
  sorry

end max_f_of_polynomial_l394_394561


namespace PE_perpendicular_BC_l394_394415

-- Define the necessary entities and conditions
variables {A B C D P Q E : Type} [EuclideanGeometry]

-- Given Conditions
variables {cyclic_quadrilateral : CyclicQuadrilateral A B C D}
variables {midpoint_E : Midpoint E A B}
variables {intersecting_diagonals : IntersectingDiagonals A B C D P}
variables {intersecting_extensions : IntersectingExtensions D A C B Q}
variables {PQ_perpendicular_AC : PQ ⊥ AC}

-- Theorem to prove
theorem PE_perpendicular_BC (cyclic_quadrilateral : CyclicQuadrilateral A B C D)
  (intersecting_diagonals : IntersectingDiagonals A B C D P)
  (intersecting_extensions : IntersectingExtensions D A C B Q)
  (midpoint_E : Midpoint E A B)
  (PQ_perpendicular_AC : IsPerpendicular PQ AC) :
  IsPerpendicular PE BC := by
  sorry

end PE_perpendicular_BC_l394_394415


namespace capacity_of_new_vessel_is_10_l394_394716

-- Define the conditions
def first_vessel_capacity : ℕ := 2
def first_vessel_concentration : ℚ := 0.25
def second_vessel_capacity : ℕ := 6
def second_vessel_concentration : ℚ := 0.40
def total_liquid_combined : ℕ := 8
def new_mixture_concentration : ℚ := 0.29
def total_alcohol_content : ℚ := (first_vessel_capacity * first_vessel_concentration) + (second_vessel_capacity * second_vessel_concentration)
def desired_vessel_capacity : ℚ := total_alcohol_content / new_mixture_concentration

-- The theorem we want to prove
theorem capacity_of_new_vessel_is_10 : desired_vessel_capacity = 10 := by
  sorry

end capacity_of_new_vessel_is_10_l394_394716


namespace proportionality_intersect_calculation_l394_394042

variables {x1 x2 y1 y2 : ℝ}

/-- Proof that (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15,
    given specific conditions on x1, x2, y1, and y2. -/
theorem proportionality_intersect_calculation
  (h1 : y1 = 5 / x1) 
  (h2 : y2 = 5 / x2)
  (h3 : x1 * y1 = 5)
  (h4 : x2 * y2 = 5)
  (h5 : x1 = -x2)
  (h6 : y1 = -y2) :
  (x1 - 2 * x2) * (3 * y1 + 4 * y2) = -15 := 
sorry

end proportionality_intersect_calculation_l394_394042


namespace ratio_volume_cylinder_cube_l394_394688

-- Define the volume of the cube
def volume_cube (s : ℝ) : ℝ :=
  s ^ 3

-- Define the radius of the inscribed cylinder
def radius_cylinder (s : ℝ) : ℝ :=
  s / 2

-- Define the height of the inscribed cylinder
def height_cylinder (s : ℝ) : ℝ :=
  s

-- Define the volume of the inscribed cylinder
def volume_cylinder (s : ℝ) : ℝ :=
  π * (radius_cylinder s)^2 * (height_cylinder s)

-- State the theorem about the ratio of volumes
theorem ratio_volume_cylinder_cube (s : ℝ) (h : s > 0) : 
  (volume_cylinder s) / (volume_cube s) = π / 4 := by
  sorry

end ratio_volume_cylinder_cube_l394_394688


namespace no_equilateral_triangle_on_lattice_points_l394_394385

noncomputable def is_lattice_point (p : ℤ × ℤ) : Prop := 
  true

noncomputable def is_equilateral_triangle (A B C : ℤ × ℤ) : Prop := 
  (∥A.1 - B.1∥ + ∥A.2 - B.2∥ = ∥B.1 - C.1∥ + ∥B.2 - C.2∥) ∧ 
  (∥B.1 - C.1∥ + ∥B.2 - C.2∥ = ∥C.1 - A.1∥ + ∥C.2 - A.2∥) ∧ 
  (∥C.1 - A.1∥ + ∥C.2 - A.2∥ = ∥A.1 - B.1∥ + ∥A.2 - B.2∥)

theorem no_equilateral_triangle_on_lattice_points : ¬ ∃ (A B C : ℤ × ℤ), 
  (is_lattice_point A) ∧ (is_lattice_point B) ∧ (is_lattice_point C) ∧ 
  (is_equilateral_triangle A B C) :=
by
  sorry

end no_equilateral_triangle_on_lattice_points_l394_394385


namespace cost_of_six_dozen_l394_394596

variable (cost_of_four_dozen : ℕ)
variable (dozens_to_purchase : ℕ)

theorem cost_of_six_dozen :
  cost_of_four_dozen = 24 →
  dozens_to_purchase = 6 →
  (dozens_to_purchase * (cost_of_four_dozen / 4)) = 36 :=
by
  intros h1 h2
  sorry

end cost_of_six_dozen_l394_394596


namespace bus_routes_have_8_stops_l394_394877

theorem bus_routes_have_8_stops (n : ℕ) 
    (h₁ : ∀ (s₁ s₂ : ℕ), s₁ ≠ s₂ → ∃ route, route ∋ s₁ ∧ route ∋ s₂)
    (h₂ : ∀ (r₁ r₂ : ℕ), r₁ ≠ r₂ → ∃! stop, stop ∈ r₁ ∧ stop ∈ r₂)
    (h₃ : ∀ (route : ℕ), ∃ (stops : ℕ), stops ≥ 3 ∧ route = stops)
    (H : 57 = n * (n - 1) + 1) :
  n = 8 :=
by
  sorry

end bus_routes_have_8_stops_l394_394877


namespace emily_jumping_game_l394_394388

def tiles_number (n : ℕ) : Prop :=
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 5 = 2

theorem emily_jumping_game : tiles_number 47 :=
by
  unfold tiles_number
  sorry

end emily_jumping_game_l394_394388


namespace cos_angle_sub_vectors_l394_394484

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (H1 : ∥a∥ = 1) (H2 : ∥b∥ = 1) (H3 : ∥c∥ = real.sqrt 2)
variables (H4 : a + b + c = 0)

theorem cos_angle_sub_vectors :
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry

end cos_angle_sub_vectors_l394_394484


namespace value_of_expression_l394_394926

theorem value_of_expression {a b : ℝ} (h1 : 2 * a^2 + 6 * a - 14 = 0) (h2 : 2 * b^2 + 6 * b - 14 = 0) :
  (2 * a - 3) * (4 * b - 6) = -2 :=
by
  sorry

end value_of_expression_l394_394926


namespace triangle_area_ABC_l394_394050

theorem triangle_area_ABC :
  let A : (ℝ × ℝ) := (0, 0)
  let B : (ℝ × ℝ) := (1, 2)
  let C : (ℝ × ℝ) := (2, 0)
  (1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))) = 2 :=
by {
  let A : (ℝ × ℝ) := (0, 0)
  let B : (ℝ × ℝ) := (1, 2)
  let C : (ℝ × ℝ) := (2, 0)
  calc
    (1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))
      = (1 / 2 * abs (0 * (2 - 0) + 1 * (0 - 0) + 2 * (0 - 2))) : by rw [A, B, C]
  ... = (1 / 2 * abs (0 + 0 - 4)) : by norm_num
  ... = (1 / 2 * 4) : by norm_num
  ... = 2 : by norm_num
}

end triangle_area_ABC_l394_394050


namespace max_sqrt_sum_l394_394086

variables {a b c : ℝ}

theorem max_sqrt_sum (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + 9 * c^2 = 1) : 
  sqrt a + sqrt b + sqrt 3 * c ≤ sqrt 21 / 3 :=
sorry

end max_sqrt_sum_l394_394086


namespace geometric_progression_general_term_l394_394051

noncomputable def a_n (n : ℕ) : ℝ := 2^(n-1)

theorem geometric_progression_general_term :
  (∀ n : ℕ, n ≥ 1 → a_n n > 0) ∧
  a_n 1 = 1 ∧
  a_n 2 + a_n 3 = 6 →
  ∀ n, a_n n = 2^(n-1) :=
by
  intros h
  sorry

end geometric_progression_general_term_l394_394051


namespace probability_of_event_A_correct_l394_394097

structure Vector :=
  (x : ℤ)
  (y : ℤ)

def a_m (m : ℤ) : Vector :=
  ⟨m, 1⟩

def b_n (n : ℤ) : Vector :=
  ⟨2, n⟩

def ortho (u v : Vector) : Prop :=
  u.x * v.x + u.y * v.y = 0

def a_minus_b (m n : ℤ) : Vector :=
  ⟨m - 2, 1 - n⟩

def event_A (m n : ℤ) : Prop :=
  ortho (a_m m) (a_minus_b m n)

noncomputable def probability_event_A : ℚ :=
  if 2 ≤ 16 then 2 / 16 else 0

theorem probability_of_event_A_correct :
  probability_event_A = 1 / 8 := by
  sorry

end probability_of_event_A_correct_l394_394097


namespace equation_D_is_quadratic_l394_394659

def is_quadratic_in_one_variable (eq : ℕ → Prop) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ eq = λ x, a * x ^ 2 + b * x + c

def equation_D : ℕ → Prop := λ x, x ^ 2 - 1 = 0

theorem equation_D_is_quadratic :
  is_quadratic_in_one_variable equation_D :=
sorry

end equation_D_is_quadratic_l394_394659


namespace value_of_n_seventh_term_of_expansion_l394_394442

-- Problem 1: Proving the value of n given the sum of binomial coefficients.
theorem value_of_n (n : ℕ) (h_sum : ∑ k in finset.range 3, nat.choose n k = 56) : n = 10 :=
sorry

-- Problem 2: Proving the seventh term of the expansion.
theorem seventh_term_of_expansion (T_7 : ℝ → ℝ) (x : ℝ) (h_T_7 : T_7 x = nat.choose 10 6 * (x^2)^4 * (1 / (2 * real.sqrt x))^6) :
  T_7 x = (105 / 32) * x^5 :=
sorry

end value_of_n_seventh_term_of_expansion_l394_394442


namespace regular_hexagon_interior_angle_measure_l394_394235

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l394_394235


namespace initial_wage_of_illiterate_l394_394057

-- Definitions from the conditions
def illiterate_employees : ℕ := 20
def literate_employees : ℕ := 10
def total_employees := illiterate_employees + literate_employees

-- Given that the daily average wages of illiterate employees decreased to Rs. 10
def daily_wages_after_decrease : ℝ := 10
-- The total decrease in the average salary of all employees by Rs. 10 per day
def decrease_in_avg_wage : ℝ := 10

-- To be proved: the initial daily average wage of the illiterate employees was Rs. 25.
theorem initial_wage_of_illiterate (I : ℝ) :
  (illiterate_employees * I - illiterate_employees * daily_wages_after_decrease = total_employees * decrease_in_avg_wage) → 
  I = 25 := 
by
  sorry

end initial_wage_of_illiterate_l394_394057


namespace remove_two_terms_sum_one_l394_394337

theorem remove_two_terms_sum_one :
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let total_sum := (60/120) + (30/120) + (20/120) + (15/120) + (12/120) + (10/120)
  let target_sum := 1
  ∃ (a b : ℚ), a ∈ fractions ∧ b ∈ fractions ∧ 
    (a ≠ b) ∧ 
    (total_sum - (a + b) = target_sum) 
    ∧ ((a = 1/8 ∧ b = 1/10) ∨ (a = 1/10 ∧ b = 1/8)) := 
by
  let fractions := [1/2, 1/4, 1/6, 1/8, 1/10, 1/12]
  let fractions_set := fractions.to_finset
  let total_sum := 147 / 120
  let target := 120 / 120
  use [1/8, 1/10]
  simp [Fractions, fractions_set, total_sum, target]
  split
  { exact sorry }
  { split
    { rw [1/8, 1/10, total_sum, target], norm_num }
    { simp, norm_num }}
  sorry

end remove_two_terms_sum_one_l394_394337


namespace solution_of_inequality_l394_394851

theorem solution_of_inequality (a x : ℝ) (h₀ : 0 < a) (h₁ : a < 1) :
  (x - a) * (x - a⁻¹) < 0 ↔ a < x ∧ x < a⁻¹ :=
by sorry

end solution_of_inequality_l394_394851


namespace prob_green_ball_is_7_over_15_l394_394370

-- Define the conditions
def ContainerA := (red: 5, green: 5)
def ContainerB := (red: 8, green: 2)
def ContainerC := (red: 3, green: 7)

def totalBalls (c : (red : Nat, green : Nat)) : Nat := c.red + c.green
def probGreen (c : (red : Nat, green : Nat)) : ℚ := c.green / totalBalls c
def probContainer : ℚ := 1 / 3

-- Define the combined probabilities
def combinedProbGreenA : ℚ := probContainer * probGreen ContainerA
def combinedProbGreenB : ℚ := probContainer * probGreen ContainerB
def combinedProbGreenC : ℚ := probContainer * probGreen ContainerC

-- Define the total probability of selecting a green ball
def totalProbGreen : ℚ := combinedProbGreenA + combinedProbGreenB + combinedProbGreenC

-- The main statement to prove
theorem prob_green_ball_is_7_over_15 : totalProbGreen = 7 / 15 :=
by 
  unfold ContainerA ContainerB ContainerC
  unfold totalBalls probGreen probContainer combinedProbGreenA combinedProbGreenB combinedProbGreenC totalProbGreen
  sorry

end prob_green_ball_is_7_over_15_l394_394370


namespace solve_for_x_l394_394966

theorem solve_for_x (z : ℂ) (x : ℂ) (h₁ : 7 * x - z = 15000) (h₂ : z = 10 + 180 * complex.I) : 
  x = 2144 + (2 / 7) * complex.I :=
  sorry

end solve_for_x_l394_394966


namespace people_at_table_l394_394127

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l394_394127


namespace num_true_propositions_l394_394368

def prop1 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = x^3 → ¬ (∃ c, c ∈ set.Icc (-1:ℝ) 1 ∧ has_deriv_at f 0 c)

def prop2 (a b c d : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ),
  (f x = a * x^3 + b * x^2 + c * x + d) →
  (∃ x, deriv f x = 0) ↔ b^2 - 3 * a * c > 0

def prop3 (m n : ℝ) : Prop :=
  ∀ (f : ℝ → ℝ),
  (f x = m * x^3 + (m - 1) * x^2 + 48 * (m - 2) * x + n) →
  (∀ x ∈ set.Icc (-4:ℝ) 4, deriv f x < 0)

def prop4 (g : ℝ → ℝ) : Prop :=
  g x = (finset.range 2011).prod (λ n, x - n) →
  deriv g 2010 = nat.fact 2009

theorem num_true_propositions :
  let p1 := prop1 (λ x, x^3)
  let p2 := prop2 (λ x, 1, 1, 1, 1) -- example constants to form a cubic
  let p3 := prop3 (λ x, 1, 0, 0, 0) -- example constants to form an odd cubic
  let p4 := prop4 (λ x, (finset.range 2011).prod (λ n, x - n))
  (p2 ∧ p3 ∧ p4 ∧ ¬ p1) ↔ true := sorry

end num_true_propositions_l394_394368


namespace range_area_triangle_l394_394096

noncomputable theory
open Real

def f (x : ℝ) : ℝ := if (0 < x ∧ x < 1) then -log x else log x

def tangent_line (f : ℝ → ℝ) (x : ℝ) : ℝ → ℝ :=
λ t, (deriv f x) * (t - x) + f x

theorem range_area_triangle (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < 1) 
  (h₃ : 1 < x₂)
  (h₄ : x₁ * x₂ = 1) :
  let l₁ := tangent_line f x₁,
      l₂ := tangent_line f x₂,
      A := (0, l₁ 0),
      B := (0, l₂ 0),
      P := (2 * x₁ * x₂ / (x₁ + x₂), l₁ (2 * x₁ * x₂ / (x₁ + x₂))) in
  0 < (1 / 2) * 2 * (2 * x₁ * x₂ / (x₁ + x₂)) ∧
  (1 / 2) * 2 * (2 * x₁ * x₂ / (x₁ + x₂)) < 1 := sorry

end range_area_triangle_l394_394096


namespace S3_minus_S2_S4_minus_S3_Sn_plus_1_minus_Sn_sum_50_terms_l394_394119

section Squares
variable {a : ℝ} {n : ℕ} {b : ℝ}
-- Given conditions for Part 1
def S (a : ℝ) (x : ℝ) := (a + x)^2

-- Proof for: S_3 - S_2 = 2\sqrt{3} + 9 when a = 1, b = 3
theorem S3_minus_S2 (a b : ℝ) (h₀ : a = 1) (h₁ : b = 3) : 
  S a (2 * Real.sqrt b) - S a (Real.sqrt b) = 2 * Real.sqrt (3) + 9 :=
by
  sorry

-- Proof for: S_4 - S_3 = 2\sqrt{3} + 15 when a = 1, b = 3
theorem S4_minus_S3 (a b : ℝ) (h₀ : a = 1) (h₁ : b = 3) : 
  S a (3 * Real.sqrt b) - S a (2 * Real.sqrt b) = 2 * Real.sqrt (3) + 15 :=
by
  sorry

-- General formula: ∀ n ∈ ℕ, S_{n+1} - S_{n} = 6n - 3 + 2 *sqrt{3}
theorem Sn_plus_1_minus_Sn (a b : ℝ) (n : ℕ) (h₀ : a = 1) (h₁ : b = 3) : 
  S a ((n + 1) * Real.sqrt b) - S a (n * Real.sqrt b) = 6 * n - 3 + 2 * Real.sqrt (3) :=
by
  sorry

-- Sum proof: T = t_1 + t_2 + t_3 + ... + t_{50} equals 7500 + 100√(3)
theorem sum_50_terms (a b : ℝ) (h₀ : a = 1) (h₁ : b = 3) : 
  (∑ k in (Finset.range 50), S a ((k + 1) * Real.sqrt b) - S a (k * Real.sqrt b)) = 7500 + 100 * Real.sqrt 3 :=
by
  sorry

end Squares

end S3_minus_S2_S4_minus_S3_Sn_plus_1_minus_Sn_sum_50_terms_l394_394119


namespace smaller_solution_of_quadratic_eq_l394_394406

noncomputable def smaller_solution (a b c : ℝ) : ℝ :=
  if a ≠ 0 then min ((-b + Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a))
              ((-b - Real.sqrt (b ^ 2 - 4 * a * c)) / (2 * a))
  else if b ≠ 0 then -c / b else 0 

theorem smaller_solution_of_quadratic_eq :
  smaller_solution 1 (-13) (-30) = -2 := 
by
  sorry

end smaller_solution_of_quadratic_eq_l394_394406


namespace interior_angle_of_regular_hexagon_is_120_degrees_l394_394252

theorem interior_angle_of_regular_hexagon_is_120_degrees :
  ∀ (n : ℕ), n = 6 → (n - 2) * 180 / n = 120 :=
by
  intros n h
  rw [h]
  norm_num
  sorry

end interior_angle_of_regular_hexagon_is_120_degrees_l394_394252


namespace interior_angle_of_regular_hexagon_l394_394271

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l394_394271


namespace regular_hexagon_interior_angle_l394_394261

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l394_394261


namespace sequence_a_n_term_l394_394901

theorem sequence_a_n_term :
  ∃ a : ℕ → ℕ, 
  a 1 = 1 ∧
  (∀ n : ℕ, a (n+1) = 2 * a n + 1) ∧
  a 10 = 1023 := by
  sorry

end sequence_a_n_term_l394_394901


namespace root_k_value_l394_394506

theorem root_k_value
  (k : ℝ)
  (h : Polynomial.eval 4 (Polynomial.C 2 * Polynomial.X^2 + Polynomial.C 3 * Polynomial.X - Polynomial.C k) = 0) :
  k = 44 :=
sorry

end root_k_value_l394_394506


namespace interior_angle_regular_hexagon_l394_394244

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l394_394244


namespace relationship_and_probability_l394_394333

variables (a b c d n : ℕ)

def completed_table : Prop :=
  a = 70 ∧ b = 10 ∧ c = 80 ∧ d = 40 ∧ n = 200

def chi_squared (a b c d n : ℕ) : ℝ :=
  let numerator := n * ((a * d - b * c) ^ 2)
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  numerator / denominator

def stratified_sampling_probability (male : ℕ) : ℝ :=
  if male ≥ 3 then (4 / 10 : ℝ) else 0

theorem relationship_and_probability :
  completed_table a b c d n →
  chi_squared a b c d n > 10.828 ∧
  stratified_sampling_probability 3 = 2 / 5 :=
by
  intros _ _ _
  sorry

end relationship_and_probability_l394_394333


namespace intersecting_plane_product_one_l394_394706

variables {Point : Type*} [MetricSpace Point]

noncomputable def ratio (P1 Q P2 : Point) : ℝ := dist P1 Q / dist Q P2

def closed_polygon (P : ℕ → Point) (n : ℕ) : Prop := P n = P 0

def intersects_internally (P Q : Point → Point) (n : ℕ) : Prop := 
∀ i < n, dist (P i) (Q i) + dist (Q i) (P (i + 1)) = dist (P i) (P (i + 1))

theorem intersecting_plane_product_one
  {P : ℕ → Point} {Q : Point → Point}
  (n : ℕ) (hn : 0 < n)
  (h_closed : closed_polygon P n)
  (h_internal : intersects_internally P Q n) :
  ∏ i in finset.range n, ratio (P i) (Q i) (P (i + 1) % n) = 1 :=
sorry

end intersecting_plane_product_one_l394_394706


namespace number_of_borrowing_schemes_l394_394168

-- Define the problem conditions
def students : Type := {A, B, C, D, E}
def categories : Type := {A, B, C, D}
def studentA := A -- Here we assume A is a predefined identifier

-- Define the conditions as hypotheses
def borrows_from_category (s : students) (c : categories) : Prop := sorry

def condition1 (s : students) : Prop :=
  ∃ c : categories, borrows_from_category s c

def condition2 (c : categories) : Prop :=
  ∃ s : students, borrows_from_category s c

def condition3 : students :=
  students.A -- Student A

def condition3_hypothesis : borrows_from_category condition3 categories.A := sorry
  
-- Define the theorem to be proved
theorem number_of_borrowing_schemes : 
  condition1 ∧ condition2 ∧ (∀ s ∈ students, condition1 s) ∧ condition3_hypothesis →  -- checking all conditions
  (Σ n : ℕ, n = 60) := 
by
  sorry -- to indicate the proof is omitted

end number_of_borrowing_schemes_l394_394168


namespace solve_for_y_l394_394164

noncomputable def log5 (x : ℝ) : ℝ := (Real.log x) / (Real.log 5)

theorem solve_for_y (y : ℝ) (h₀ : log5 ((2 * y + 10) / (3 * y - 6)) + log5 ((3 * y - 6) / (y - 4)) = 3) : 
  y = 170 / 41 :=
sorry

end solve_for_y_l394_394164


namespace no_blue_points_in_red_triangle_l394_394755

/-
Each point in the plane with integer coordinates is colored red or blue such that the following two properties hold:
1. For any two red points, the line segment joining them does not contain any blue points.
2. For any two blue points that are distance 2 apart, the midpoint of the line segment joining them is blue.

Prove that if three red points are the vertices of a triangle, then the interior of the triangle does not contain any blue points.
-/

-- Define a point on the 2D integer coordinate plane
structure Point := (x : ℤ) (y : ℤ)

-- Predicate for whether a point is red
def is_red (p : Point) : Prop := sorry
-- Predicate for whether a point is blue
def is_blue (p : Point) : Prop := sorry

-- Condition 1: For any two red points, the line segment joining them does not contain any blue points
axiom condition1 : ∀ {p1 p2 : Point}, is_red p1 → is_red p2 → ∀ (p : Point), p ≠ p1 → p ≠ p2 → (p.x - p1.x) * (p2.y - p1.y) = (p.y - p1.y) * (p2.x - p1.x) → ¬ is_blue p

-- Condition 2: For any two blue points that are distance 2 apart, the midpoint of the line segment joining them is blue
axiom condition2 : ∀ {p1 p2 : Point}, is_blue p1 → is_blue p2 → (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = 4 → is_blue ⟨(p1.x + p2.x)/2, (p1.y + p2.y)/2⟩

-- Definition of a triangle's interior
def in_triangle (p A B C : Point) : Prop := 
  let area := (λ (P Q R : Point), (Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y)) in
  let a := area A B p,
      b := area B C p,
      c := area C A p in
  (a = 0 ∨ b = 0 ∨ c = 0 ∨ (a > 0 ∧ b > 0 ∧ c > 0) ∨ (a < 0 ∧ b < 0 ∧ c < 0))

-- Main theorem
theorem no_blue_points_in_red_triangle (A B C : Point) (hA : is_red A) (hB : is_red B) (hC : is_red C) :
  ∀ (p : Point), in_triangle p A B C → ¬ is_blue p := 
sorry

end no_blue_points_in_red_triangle_l394_394755


namespace total_investment_by_Q_is_correct_l394_394583

noncomputable def main : ℝ :=
  let P1_investment := 50000 : ℝ
  let P1_share := 3 : ℝ
  let Q1_share := 4 : ℝ
  let P2_investment := 30000 : ℝ
  let P2_share := 2 : ℝ
  let Q2_share := 3 : ℝ

  let Q1_investment := (P1_investment * Q1_share) / P1_share
  let Q2_investment := (P2_investment * Q2_share) / P2_share

  Q1_investment + Q2_investment

theorem total_investment_by_Q_is_correct :
  main = 111666.67 := 
sorry

end total_investment_by_Q_is_correct_l394_394583


namespace perfect_square_k_l394_394504

theorem perfect_square_k (a b k : ℝ) (h : ∃ c : ℝ, a^2 + 2*(k-3)*a*b + 9*b^2 = (a + c*b)^2) : 
  k = 6 ∨ k = 0 := 
sorry

end perfect_square_k_l394_394504


namespace trapezoid_EFGH_p_q_sum_l394_394213

noncomputable def p_q_sum : ℕ :=
  let EF := 68
  let FH := 21
  let JK := 7
  let FG := 34
  let GH := 34
  let EH := Nat.sqrt (EF * EF - FH * FH)
  let p := 9
  let q := 463
  p + q

theorem trapezoid_EFGH_p_q_sum :
  (EFGH_trapezoid : (EF parallel GH) ∧ (FG = 34) ∧ (GH = 34) ∧ (EH ⊥ FH) ∧ (J_intersection : ∃ J, J ∈ EG ∧ J ∈ FH) ∧ (K_midpoint F H K) ∧ JK = 7) →
  p_q_sum = 472 :=
by
  sorry

end trapezoid_EFGH_p_q_sum_l394_394213


namespace symmetric_point_of_A_l394_394519

-- Define the point A
def pointA : ℝ × ℝ × ℝ := (1, 1, 2)

-- Define a function that returns the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, -p.2, -p.3)

-- Statement to prove
theorem symmetric_point_of_A :
  symmetric_point pointA = (-1, -1, -2) :=
by sorry

end symmetric_point_of_A_l394_394519


namespace moles_of_Na2SO4_formed_l394_394378

/-- 
Given the following conditions:
1. 1 mole of H2SO4 reacts with 2 moles of NaOH.
2. In the presence of 0.5 moles of HCl and 0.5 moles of KOH.
3. At a temperature of 25°C and a pressure of 1 atm.
Prove that the moles of Na2SO4 formed is 1 mole.
-/

theorem moles_of_Na2SO4_formed
  (H2SO4 : ℝ) -- moles of H2SO4
  (NaOH : ℝ) -- moles of NaOH
  (HCl : ℝ) -- moles of HCl
  (KOH : ℝ) -- moles of KOH
  (T : ℝ) -- temperature in °C
  (P : ℝ) -- pressure in atm
  : H2SO4 = 1 ∧ NaOH = 2 ∧ HCl = 0.5 ∧ KOH = 0.5 ∧ T = 25 ∧ P = 1 → 
  ∃ Na2SO4 : ℝ, Na2SO4 = 1 :=
by
  sorry

end moles_of_Na2SO4_formed_l394_394378


namespace algebraic_expression_value_l394_394838

open Real

theorem algebraic_expression_value
  (θ : ℝ)
  (a := (cos θ, sin θ))
  (b := (1, -2))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (2 * sin θ - cos θ) / (sin θ + cos θ) = 5 :=
by
  sorry

end algebraic_expression_value_l394_394838


namespace sum_f_geq_zero_l394_394093

def f (n k a x : ℕ) : ℤ := 
  (⌊(n + k + x : ℕ) / a⌋ : ℤ) - (⌊(n + x : ℕ) / a⌋ : ℤ) - (⌊(k + x : ℕ) / a⌋ : ℤ) + (⌊(x : ℕ) / a⌋ : ℤ)

theorem sum_f_geq_zero (n k a m : ℕ) (a_pos : 0 < a) : 
  ∑ x in finset.range (m + 1), f n k a x ≥ 0 :=
sorry

end sum_f_geq_zero_l394_394093


namespace sum_of_factors_of_30_is_72_l394_394293

-- Condition: given the number 30
def number := 30

-- Define the positive factors of 30
def factors : List ℕ := [1, 2, 3, 5, 6, 10, 15, 30]

-- Statement to prove the sum of the positive factors
theorem sum_of_factors_of_30_is_72 : (factors.sum) = 72 := 
by
  sorry

end sum_of_factors_of_30_is_72_l394_394293


namespace complex_expression_equiv_l394_394761

-- Define the expressions involved
def z1 : ℂ := 7 - 3 * complex.i
def z2 : ℂ := 2 + 5 * complex.i
def z3 : ℂ := 1 - 18 * complex.i

-- The theorem stating the equivalence 
theorem complex_expression_equiv : z1 - 3 * z2 = z3 :=
by
  -- Proof can be filled in later
  sorry

end complex_expression_equiv_l394_394761


namespace number_of_people_seated_l394_394150

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l394_394150


namespace sandbox_sand_weight_l394_394698

theorem sandbox_sand_weight
  (side_len : ℝ) 
  (side_len_eq : side_len = 40)
  (bag_weight : ℝ)
  (bag_weight_eq : bag_weight = 30)
  (bag_coverage : ℝ)
  (bag_coverage_eq : bag_coverage = 80) :
  let area := side_len * side_len in
  let num_bags := area / bag_coverage in
  let total_weight := num_bags * bag_weight in
  total_weight = 600 := by
  sorry

end sandbox_sand_weight_l394_394698


namespace find_a_l394_394033

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x

noncomputable def f (a x : ℝ) : ℝ := (1 / (2^x - 1)) + a

theorem find_a (a : ℝ) : 
  is_odd_function (f a) → a = 1 / 2 :=
by
  sorry

end find_a_l394_394033


namespace find_y_from_x_squared_l394_394501

theorem find_y_from_x_squared (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = 6) : y = 29 :=
by
  sorry

end find_y_from_x_squared_l394_394501


namespace Greg_older_than_Marcia_l394_394735

theorem Greg_older_than_Marcia:
  (Cindy Jan Marcia Greg : ℕ) 
  (h1 : Cindy = 5)
  (h2 : Jan = Cindy + 2)
  (h3 : Marcia = 2 * Jan)
  (h4 : Greg = 16)
  : Greg - Marcia = 2 := 
by
  sorry

end Greg_older_than_Marcia_l394_394735


namespace parallel_vectors_l394_394553

theorem parallel_vectors (m n : ℝ) :
  let a := (m, -1, 2)
  let b := (3, -4, n)
  a ∥ b → m = 3 / 4 ∧ n = 8 :=
by
  sorry

end parallel_vectors_l394_394553


namespace heat_dissipated_in_resistor_l394_394726

theorem heat_dissipated_in_resistor
  (R C r : ℝ)
  (ε : ℝ): 
  C > 0 → ε > 0 → r > 0 → R > 0 →
  ∃ (Q_R : ℝ), Q_R = C * ε^2 * R / (2 * (R + r)) :=
by
  intro hC hε hr hR
  use C * ε^2 * R / (2 * (R + r))
  sorry

end heat_dissipated_in_resistor_l394_394726


namespace quadratic_inequality_l394_394511

theorem quadratic_inequality (a b c : ℝ)
  (h1 : a < 0) 
  (h2 : b = -2 * a)
  (h3 : c = -8 * a)
  (h4 : ∀ x, ax^2 + b * x + c < 0 ↔ (x < -2 ∨ x > 4)) :
  (ax^2 - 2 * a * x - 8 * a) evaluated at x=5 < (ax^2 - 2 * a * x - 8 * a) evaluated at x=2 < (ax^2 - 2 * a * x - 8 * a) evaluated at x=1 :=
sorry

end quadratic_inequality_l394_394511


namespace number_of_triangles_in_polygon_l394_394850

theorem number_of_triangles_in_polygon {n : ℕ} (h : n > 0) :
  let vertices := (2 * n + 1)
  ∃ triangles_containing_center : ℕ, triangles_containing_center = (n * (n + 1) * (2 * n + 1)) / 6 :=
sorry

end number_of_triangles_in_polygon_l394_394850


namespace cuckoo_sounds_seven_hours_l394_394318

def cuckoo_sounds (hour : ℕ) : ℕ :=
  if hour = 0 then 12 else if hour > 12 then hour - 12 else hour

theorem cuckoo_sounds_seven_hours (h : ℕ) :
  let times := [10, 11, 12, 1, 2, 3, 4] in
  (∑ x in times, cuckoo_sounds x) = 43 :=
by sorry

end cuckoo_sounds_seven_hours_l394_394318


namespace angle_CDE_l394_394523

theorem angle_CDE (angle_A angle_B angle_C angle_AEB angle_BED : ℝ)
  (isosceles_ADE : ∀ (x : ℝ), angle_AED = x ∧ angle_ADE = x)
  (hA : angle_A = 90)
  (hB : angle_B = 90)
  (hC : angle_C = 90)
  (hAEB : angle_AEB = 50)
  (hBED : angle_BED = 45) :
  angle_CDE = 112.5 :=
sorry

end angle_CDE_l394_394523


namespace simplify_trig_expression_l394_394979

variable {x : ℝ}

theorem simplify_trig_expression (h : 1 + cos x ≠ 0) : 
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by
  sorry

end simplify_trig_expression_l394_394979


namespace arc_length_greater_than_diameter_l394_394587

noncomputable def arc_length {k k' : Real} (A B : Real) (circumference_area_bisect : Bool) : Prop :=
  ∀ (A B : Point) (r d : Real) (C: Circle) (A ∈ Circumference C) (B ∈ Circumference C) 
    (arc_k' : Arc) (T : Set) (h1 : bisects_area T C) (h2: subtends_arc A B arc_k'),
    arc.length k' > Circle.diameter C

/-- Given points A and B on a circle k connected by an arc k' which bisects the area of circle k,
proves that the length of the arc k' is greater than the diameter of circle k. -/
theorem arc_length_greater_than_diameter (C : Circle) (A B : Point)
  (arc_k' : Arc) (r d : Real) (circumference_area_bisect : Bool)
  (h1 : A ∈ Circumference C) (h2 : B ∈ Circumference C)
  (h3 : bisects_area arc_k' C) (h4: subtends_arc A B arc_k')
  (arc_length_theorem : arc_length C k' A B circumference_area_bisect) :
  arc.length arc_k' > Circle.diameter C :=
sorry

end arc_length_greater_than_diameter_l394_394587


namespace jason_borrowed_198_l394_394535

def earnings (i : ℕ) : ℕ :=
  match i % 6 with
  | 1 => 2
  | 2 => 3
  | 3 => 4
  | 4 => 5
  | 5 => 6
  | _ => 7

theorem jason_borrowed_198 :
  (∑ i in (Finset.range 45).map (Function.Embedding.coeFnL (Finset.Ico.coeFinSetHom)) earnings) = 198 :=
by
  sorry

end jason_borrowed_198_l394_394535


namespace ratio_pq_f₁f₂_l394_394727

variable (P Q R F₁ F₂ : ℝ × ℝ)

noncomputable def ellipse_eq : Prop := 
  ∀ (x y : ℝ), y^2 * 16 + x^2 * 4 = 64

noncomputable def is_vertex_q : Prop := Q = (0, 2)

noncomputable def is_parallel_x_axis : Prop := P.2 = 0 ∧ R.2 = 0

noncomputable def side_pr_longer_pq : Prop := dist P R > dist P Q

noncomputable def foci_on_sides : Prop := F₁.1 < Q.1 ∧ F₂.1 > Q.1 ∧ dist Q R ≥ dist Q F₁ ∧ dist Q P ≥ dist Q F₂

noncomputable def pq_over_f₁f₂ := dist P Q / dist F₁ F₂

theorem ratio_pq_f₁f₂ :
  ellipse_eq P Q R ∧ 
  is_vertex_q Q ∧
  is_parallel_x_axis P R ∧
  side_pr_longer_pq P Q R ∧
  foci_on_sides F₁ F₂ P Q R
  → pq_over_f₁f₂ P Q F₁ F₂ = Real.sqrt 15 / 6 :=
by
  sorry

end ratio_pq_f₁f₂_l394_394727


namespace no_functions_periodic_modular_cond_l394_394095

open Nat

theorem no_functions_periodic_modular_cond :
  ∃ (f : ℕ → Fin 17), 
    (∀ x, f (x + 17) = f x) ∧ (∀ x, f (x^2) = f x * f x + 15) :=
  false := sorry

end no_functions_periodic_modular_cond_l394_394095


namespace initial_wage_of_illiterate_l394_394058

-- Definitions from the conditions
def illiterate_employees : ℕ := 20
def literate_employees : ℕ := 10
def total_employees := illiterate_employees + literate_employees

-- Given that the daily average wages of illiterate employees decreased to Rs. 10
def daily_wages_after_decrease : ℝ := 10
-- The total decrease in the average salary of all employees by Rs. 10 per day
def decrease_in_avg_wage : ℝ := 10

-- To be proved: the initial daily average wage of the illiterate employees was Rs. 25.
theorem initial_wage_of_illiterate (I : ℝ) :
  (illiterate_employees * I - illiterate_employees * daily_wages_after_decrease = total_employees * decrease_in_avg_wage) → 
  I = 25 := 
by
  sorry

end initial_wage_of_illiterate_l394_394058


namespace min_max_fn_floor_l394_394563

theorem min_max_fn_floor (m : ℕ) (r : Fin m → ℚ) (hr_sum : Finset.sum Finset.univ r = 1)
  (h_pos_m : 0 < m) :
  ∃ (n : ℕ), ∀ n ∈ {n : ℕ | 0 < n}, 0 ≤ n - ∑ i : Fin m, ⌊r i * n⌋ ∧ n - ∑ i : Fin m, ⌊r i * n⌋ ≤ m - 1 := 
sorry

end min_max_fn_floor_l394_394563


namespace find_ratio_af_fb_l394_394072

-- Define points in triangle
variables {A B C D E F P Q : Type} [point_space A B C D E F P Q]

-- Define conditions
variables (d_on_bc : ∃ λ : ℝ, D = λ • B + (1 - λ) • C)
variables (f_on_ab : ∃ μ : ℝ, F = μ • A + (1 - μ) • B)
variables (ad_cf_intersect_at_p : line_intersect (A, D) (C, F) P)
variables (e_on_ac : ∃ ν : ℝ, E = ν • A + (1 - ν) • C)
variables (be_cf_intersect_at_q : line_intersect (B, E) (C, F) Q)
variables (ap_pd_ratio : ∃ k : ℝ, k = 3/4 ∧ P = k • A + (1 - k) • D)
variables (fq_qc_ratio : ∃ m : ℝ, m = 3/5 ∧ Q = m • F + (1 - m) • C)

-- The goal
theorem find_ratio_af_fb (A B C D E F P Q : point_space) :
  d_on_bc → 
  f_on_ab → 
  ad_cf_intersect_at_p → 
  e_on_ac → 
  be_cf_intersect_at_q → 
  ap_pd_ratio → 
  fq_qc_ratio →
  ∃ x : ℝ, x = ... → 
  ratio_af_fb (result : ℝ) := 
sorry

end find_ratio_af_fb_l394_394072


namespace interior_angle_of_regular_hexagon_l394_394274

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l394_394274


namespace num_possible_diagonal_lengths_eq_17_l394_394780

def quadrilateral_diagonal_lengths : ℕ :=
  let x := set.Ioc 3 21 in
  (set.image (λ n : ℕ, n) x).to_finset.card

theorem num_possible_diagonal_lengths_eq_17 :
  quadrilateral_diagonal_lengths = 17 :=
by
  sorry

end num_possible_diagonal_lengths_eq_17_l394_394780


namespace cos_angle_sub_vectors_l394_394487

-- Definitions of the conditions
variables (a b c : ℝ^3) (ha : ‖ a ‖ = 1) (hb : ‖ b ‖ = 1) (hc : ‖ c ‖ = √2)
variable (h0 : a + b + c = 0)

-- The theorem statement
theorem cos_angle_sub_vectors : 
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry -- Proof omitted

end cos_angle_sub_vectors_l394_394487


namespace dog_age_difference_proof_l394_394497

def avg (a b : ℕ) := (a + b) / 2

theorem dog_age_difference_proof :
  ∀ (D1 D2 D3 D4 D5 : ℕ), 
    D1 = 10 → 
    D2 = D1 - 2 →
    D3 = D2 + 4 →
    D4 = D3 / 2 →
    D5 = D4 + 20 →
    avg D1 D5 = 18 → 
    D1 - D2 = 2 :=
begin
  intros,  -- Introduce variables and hypotheses
  sorry,   -- Placeholder for the proof
end

end dog_age_difference_proof_l394_394497


namespace extreme_points_of_f_range_of_a_l394_394939

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x ≥ -1 then Real.log (x + 1) + a * (x^2 - x) 
  else 0

theorem extreme_points_of_f (a : ℝ) :
  (a < 0 → ∃ x, f a x = 0) ∧
  (0 ≤ a ∧ a ≤ 8/9 → ∃! x, f a x = 0) ∧
  (a > 8/9 → ∃ x₁ x₂, x₁ < x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

end extreme_points_of_f_range_of_a_l394_394939


namespace cos_equivalent_l394_394478

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)

-- Conditions
axiom mag_a : ∥a∥ = 1
axiom mag_b : ∥b∥ = 1
axiom mag_c : ∥c∥ = real.sqrt 2
axiom abc_zero : a + b + c = 0

-- Proof statement
theorem cos_equivalent : 
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
by sorry

end cos_equivalent_l394_394478


namespace find_f_prime_2_l394_394433

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f x / x

axiom tangent_coincide_at_points : ∀ f : ℝ → ℝ,
  (deriv f 0 = 1 / 2) ∧ (deriv (λ x, f x / x) 2 = 1 / 2)

theorem find_f_prime_2 :
  ∀ f : ℝ → ℝ, (f 2 = 2) ∧ ((∀ x, f 0 = 0)) ∧ tangent_coincide_at_points f → deriv f 2 = 2 :=
begin
  sorry
end

end find_f_prime_2_l394_394433


namespace proposition_1_proposition_4_l394_394346

theorem proposition_1 (x : ℝ) : x^2 ∈ ℝ := sorry

theorem proposition_4 (x1 x2 y1 y2 : ℂ) : x1 = x2 → y1 = y2 → x1 + y1 * complex.i = x2 + y2 * complex.i := sorry

end proposition_1_proposition_4_l394_394346


namespace determinant_of_A_l394_394736

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 0, -2],
  ![8, 5, -4],
  ![3, 3, 6]
]

theorem determinant_of_A : A.det = 108 := by
  sorry

end determinant_of_A_l394_394736


namespace probability_pairing_l394_394049

theorem probability_pairing (students : Finset ℕ) (h32 : students.card = 32) (Margo Irma Joe Pete : ℕ)
  (hM : Margo ∈ students) (hI : Irma ∈ students) (hJ : Joe ∈ students) (hP : Pete ∈ students)
  (h_distinct : Margo ≠ Irma ∧ Joe ≠ Pete ∧ Margo ≠ Joe ∧ Margo ≠ Pete ∧ Irma ≠ Joe ∧ Irma ≠ Pete) :
  let pairs := (students.erase Margo).erase Irma in
  let pairs_remaining := (pairs.erase Joe).erase Pete in
  (1 / (pairs.card : ℚ)) * (1 / (pairs_remaining.card : ℚ)) = 1 / 899 :=
by 
  sorry

end probability_pairing_l394_394049


namespace perimeter_triangle_XPQ_l394_394528

theorem perimeter_triangle_XPQ :
  ∀ (X Y Z G P Q : Type)
  (XY YZ XZ : ℝ)
  (G_is_centroid : is_centroid_triangle XYZ G)
  (line_parallel_YZ : line_parallel_through_points G YZ XY P XZ Q),
  XY = 15 → YZ = 30 → XZ = 25 →
  ∃ (XP XQ PQ : ℝ),
    XP = (2/3) * XY ∧ XQ = (2/3) * XZ ∧ PQ = (2/3) * YZ ∧
    XP + XQ + PQ = 140 / 3 := 
begin
  intro X Y Z G P Q,
  intros XY YZ XZ G_is_centroid line_parallel_YZ,
  intros h1 h2 h3,
  existsi (2 / 3 * XY),
  existsi (2 / 3 * XZ),
  existsi (2 / 3 * YZ),
  repeat { split },
  { sorry }, -- XP = (2/3) * XY
  { sorry }, -- XQ = (2/3) * XZ
  { sorry }, -- PQ = (2/3) * YZ
  { sorry }, -- XP + XQ + PQ = 140 / 3
end

end perimeter_triangle_XPQ_l394_394528


namespace part_I_part_II_part_III_l394_394546

noncomputable def interval_coverage_with_conditions (N : ℕ) (a : ℕ → ℝ) :=
  (∀ x ∈ set.Icc (0 : ℝ) 100, ∃ k, k ∈ finset.range N ∧ x ∈ set.Icc (a k) (a k + 1)) ∧
  (∀ k ∈ finset.range N, ∃ x ∈ set.Icc (0 : ℝ) 100, ∀ i ∈ finset.range N, i ≠ k → ¬(x ∈ set.Icc (a i) (a i + 1)))

theorem part_I (N : ℕ) :
  interval_coverage_with_conditions N (λ k => k - 1) ∧ ¬interval_coverage_with_conditions N (λ k => k / 2 - 1) :=
sorry

theorem part_II :
  ∃ N, interval_coverage_with_conditions N (λ k => k - 1) ∧ N = 100 :=
sorry

theorem part_III :
  ∃ N, interval_coverage_with_conditions N (λ k => -1 / 2 + (100 / 199) * (k - 1)) ∧ N = 200 :=
sorry

end part_I_part_II_part_III_l394_394546


namespace intersection_P_T_l394_394834

def P : Set ℝ := {x | x^2 - x - 2 = 0}
def T : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def intersection := {2}

theorem intersection_P_T :
  P ∩ T = intersection :=
sorry

end intersection_P_T_l394_394834


namespace second_train_start_time_l394_394218

-- Define the conditions as hypotheses
def station_distance : ℝ := 200
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def meet_time : ℝ := 12 - 7 -- Time they meet after the first train starts, in hours.

-- The theorem statement corresponding to the proof problem
theorem second_train_start_time :
  ∃ T : ℝ, 0 <= T ∧ T <= 5 ∧ (5 * speed_train_A) + ((5 - T) * speed_train_B) = station_distance → T = 1 :=
by
  -- Placeholder for actual proof
  sorry

end second_train_start_time_l394_394218


namespace evaluate_expression1_evaluate_expression2_l394_394362

theorem evaluate_expression1 : (0.25)^(1 / 2) - [-2 * (3 / 7)^0]^2 * [(-2)^3]^(4 / 3) + (Real.sqrt 2 - 1)^(-1) - 2^(1 / 2) = -125 / 2 := 
by
  sorry

theorem evaluate_expression2 : (log 5)^2 + log 2 * log 50 = 1 :=
by
  sorry

end evaluate_expression1_evaluate_expression2_l394_394362


namespace area_of_square_BDEF_l394_394728

variables (AB BC EH : ℝ)
variable (area_of_BDEF : ℝ)

-- Given conditions
def is_right_triangle (a b : ℝ) : Prop :=
  ∃ c : ℝ, c * c = a * a + b * b

def square_in_right_triangle (AB BC EH : ℝ) (side : ℝ) : Prop :=
  ∃ BD : ℝ, BD = side ∧ BD * BD = area_of_BDEF

theorem area_of_square_BDEF :
  is_right_triangle AB BC →
  AB = 15 →
  BC = 20 →
  EH = 2 →
  square_in_right_triangle AB BC EH 10 →
  area_of_BDEF = 100 :=
by
  sorry

end area_of_square_BDEF_l394_394728


namespace cosine_identity_l394_394470

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) 

theorem cosine_identity 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (hc : ∥c∥ = real.sqrt 2)
  (habc : a + b + c = 0) :
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
sorry

end cosine_identity_l394_394470


namespace find_roots_of_parabola_l394_394510

-- Define the conditions given in the problem
variables (a b c : ℝ)
variable (a_nonzero : a ≠ 0)
variable (passes_through_1_0 : a * 1^2 + b * 1 + c = 0)
variable (axis_of_symmetry : -b / (2 * a) = -2)

-- Lean theorem statement
theorem find_roots_of_parabola (a b c : ℝ) (a_nonzero : a ≠ 0)
(passes_through_1_0 : a * 1^2 + b * 1 + c = 0) (axis_of_symmetry : -b / (2 * a) = -2) :
  (a * (-5)^2 + b * (-5) + c = 0) ∧ (a * 1^2 + b * 1 + c = 0) :=
by
  -- Placeholder for the proof
  sorry

end find_roots_of_parabola_l394_394510


namespace find_XY_length_l394_394458

variables (a b c : ℝ) -- sides of triangle ABC
variables (s : ℝ) -- semi-perimeter s = (a + b + c) / 2

-- Definition of similar triangles and perimeter condition
noncomputable def XY_length
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ) 
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) : ℝ :=
  s * a / (b + c) -- by the given solution

-- The theorem statement
theorem find_XY_length
  (a b c : ℝ) (s : ℝ) -- given sides and semi-perimeter
  (AX : ℝ) (XY : ℝ) (AY : ℝ) (BX : ℝ)
  (BC : ℝ) (CY : ℝ) 
  (h1 : AX + AY + XY = BX + BC + CY)
  (h2 : AX = c * XY / a) 
  (h3 : AY = b * XY / a) :
  XY = s * a / (b + c) :=
sorry -- proof


end find_XY_length_l394_394458


namespace price_of_feed_corn_l394_394326

theorem price_of_feed_corn :
  ∀ (num_sheep : ℕ) (num_cows : ℕ) (grass_per_cow : ℕ) (grass_per_sheep : ℕ)
    (feed_corn_duration_cow : ℕ) (feed_corn_duration_sheep : ℕ)
    (total_grass : ℕ) (total_expenditure : ℕ) (months_in_year : ℕ),
  num_sheep = 8 →
  num_cows = 5 →
  grass_per_cow = 2 →
  grass_per_sheep = 1 →
  feed_corn_duration_cow = 1 →
  feed_corn_duration_sheep = 2 →
  total_grass = 144 →
  total_expenditure = 360 →
  months_in_year = 12 →
  ((total_expenditure : ℝ) / (((num_cows * feed_corn_duration_cow * 4) + (num_sheep * (4 / feed_corn_duration_sheep))) : ℝ)) = 10 :=
by
  intros
  sorry

end price_of_feed_corn_l394_394326


namespace parts_production_equation_l394_394314

theorem parts_production_equation (x : ℝ) : 
  let apr := 50
  let may := 50 * (1 + x)
  let jun := 50 * (1 + x) * (1 + x)
  (apr + may + jun = 182) :=
sorry

end parts_production_equation_l394_394314


namespace emily_necklaces_l394_394389

theorem emily_necklaces (total_beads : ℤ) (beads_per_necklace : ℤ) 
(h_total_beads : total_beads = 16) (h_beads_per_necklace : beads_per_necklace = 8) : 
  total_beads / beads_per_necklace = 2 := 
by
  sorry

end emily_necklaces_l394_394389


namespace regular_hexagon_interior_angle_l394_394260

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l394_394260


namespace prob_3_draws_to_exceed_sum_7_l394_394678

noncomputable def prob_exceeds_sum_7_in_3_draws
  (chips : Finset ℕ) (draws : List ℕ) : ℚ :=
  if chips = {1, 2, 3, 4, 5} 
     ∧ draws.nodup
     ∧ draws.length = 3
     ∧ draws.sum > 7 
     ∧ (∀ d : List ℕ, d.nodup → d.length = 2 → d.sum ≤ 7 → draws.starts_with d) 
  then 13 / 20 
  else 0

theorem prob_3_draws_to_exceed_sum_7 : 
  prob_exceeds_sum_7_in_3_draws {1, 2, 3, 4, 5} [1, 2, 5] = 13 / 20 := 
by 
  sorry

end prob_3_draws_to_exceed_sum_7_l394_394678


namespace correct_average_is_39_4_l394_394173

variable (nums : List ℝ)
variable (wrong_nums: List ℝ)

def initial_avg : ℝ := 40.2
def num_values : ℕ := 12
def wrong_sum : ℝ := initial_avg * num_values

def correction (nums wrong_nums : List ℝ) : ℝ :=
  (wrong_nums[0] - nums[0]) + (nums[1] - wrong_nums[1]) + (wrong_nums[2] - nums[2]) + (nums[3] - wrong_nums[3])

def correct_sum (nums wrong_nums : List ℝ) : ℝ :=
  wrong_sum + correction nums wrong_nums

def correct_avg (nums wrong_nums : List ℝ) : ℝ :=
  correct_sum nums wrong_nums / num_values

theorem correct_average_is_39_4 
  (h1 : wrong_nums = [nums[0] + 19, 13, 45, nums[3] - 11]) 
  : Float.round(1 (correct_avg nums wrong_nums)) = 39.4 :=
  by
    sorry

end correct_average_is_39_4_l394_394173


namespace interior_angle_of_regular_hexagon_l394_394229

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l394_394229


namespace notepad_duration_l394_394721

theorem notepad_duration (a8_papers_per_a4 : ℕ)
  (a4_papers : ℕ)
  (notes_per_day : ℕ)
  (notes_per_side : ℕ) :
  a8_papers_per_a4 = 16 →
  a4_papers = 8 →
  notes_per_day = 15 →
  notes_per_side = 2 →
  (a4_papers * a8_papers_per_a4 * notes_per_side) / notes_per_day = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end notepad_duration_l394_394721


namespace triangle_is_obtuse_l394_394197

-- Define the conditions of the problem
def angles (x : ℝ) : Prop :=
  2 * x + 3 * x + 6 * x = 180

def obtuse_angle (x : ℝ) : Prop :=
  6 * x > 90

-- State the theorem
theorem triangle_is_obtuse (x : ℝ) (hx : angles x) : obtuse_angle x :=
sorry

end triangle_is_obtuse_l394_394197


namespace simplify_expression_l394_394985

-- Define the conditions
variable (x : ℝ)
variable (hx : sin x ≠ 0)

-- Define the statement
theorem simplify_expression :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by sorry

end simplify_expression_l394_394985


namespace f_explicit_formula_and_monotonicity_range_of_a_l394_394012

-- Definitions for f and g based on given conditions
def f (f'1 : ℝ) : ℝ → ℝ := λ x, (f'1 / real.exp 1) * real.exp x - (f'1 / real.exp 1) * x + (1/2) * x^2
def g (a : ℝ) : ℝ → ℝ := λ x, (1/2) * x^2 + a

-- Problem 1: Explicit formula for f(x) and intervals of monotonicity
theorem f_explicit_formula_and_monotonicity (f'1 : ℝ) (h_f'1 : f'1 = real.exp 1) :
  (∀ x, f f'1 x = real.exp x - x + (1/2) * x^2) ∧ 
  (∀ x, x < 0 → 0 < f f'1 x) ∧
  (∀ x, 0 < x → f f'1 x < 0) :=
sorry

-- Problem 2: Range of a such that g(x) has exactly two distinct points of intersection with f(x)
theorem range_of_a (a : ℝ) :
  (∀ x, f (real.exp 1) x = real.exp x - x + (1/2) * x^2) →
  (∃ x y : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ -1 ≤ y ∧ y ≤ 2 ∧ x ≠ y ∧ g a x = f (real.exp 1) x ∧ g a y = f (real.exp 1) y) ↔
  (1 < a ∧ a ≤ 1 + 1 / real.exp 1) :=
sorry

end f_explicit_formula_and_monotonicity_range_of_a_l394_394012


namespace weight_of_new_person_l394_394668

theorem weight_of_new_person (avg_increase : ℝ) (n : ℕ) (weight_replaced : ℝ) (weight_new: ℝ) (h : avg_increase = 2.5) (hna : n = 8) (hw : weight_replaced = 65) : weight_new = 85 :=
by
  have hw_increase : weight_new - weight_replaced = n * avg_increase,
  sorry


end weight_of_new_person_l394_394668


namespace inequality_always_holds_l394_394772

theorem inequality_always_holds (x : ℝ) (a : ℝ) (h₀ : |a| ≤ 1) :
  x^2 + (a - 6) * x + (9 - 3 * a) > 0 ↔ x ∈ Set.Ioo (-∞) 2 ∪ Set.Ioo 4 ∞ := sorry

end inequality_always_holds_l394_394772


namespace seating_arrangements_three_families_l394_394210

noncomputable def number_of_seating_arrangements (n k m : ℕ) : ℕ :=
  (k!) ^ n * n!

theorem seating_arrangements_three_families :
  number_of_seating_arrangements 3 3 1 = (3!) ^ 4 :=
by
  -- this theorem will prove that the computed number of seating arrangements 
  -- is equal to (3!)^4 when considering the problem constraints.
  sorry

end seating_arrangements_three_families_l394_394210


namespace arg_range_l394_394445

theorem arg_range (z : ℂ) (a b : ℝ) (h₁ : a ∈ ℝ⁺) (h₂ : a > b) (h₃ : b > 0) 
    (h₄ : complex.arg (z + a + a * complex.I) = real.pi / 4) 
    (h₅ : complex.arg (z - a - a * complex.I) = 5 * real.pi / 4) : 
    complex.arg (z - b + b * complex.I) ∈ set.Icc (real.arctan ((a + b) / (a - b))) 
                                                    (real.arctan ((a - b) / (a + b)) + real.pi) :=
sorry

end arg_range_l394_394445


namespace interior_angle_of_regular_hexagon_l394_394224

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l394_394224


namespace sum_of_a_with_integer_roots_is_16_l394_394207

theorem sum_of_a_with_integer_roots_is_16 (a : ℤ) (p q : ℤ) (h1 : (∃ (x : ℤ), x * x - a * x + 2 * a = 0)) (h2 : p + q = a) (h3 : p * q = 2 * a) :
  let possible_values := {a | (∃ p q : ℤ, (x * x - a * x + 2 * a = 0) ∧ (p + q = a) ∧ (p * q = 2 * a)) } in 
  ∑ (a in possible_values.to_finset) = 16 :=
by 
  sorry

end sum_of_a_with_integer_roots_is_16_l394_394207


namespace simplify_trig_expression_l394_394977

open_locale real

theorem simplify_trig_expression (x : ℝ) (h : 1 + real.cos x ≠ 0) : 
  (real.sin x / (1 + real.cos x) + (1 + real.cos x) / real.sin x) = 2 * real.csc x :=
sorry

end simplify_trig_expression_l394_394977


namespace one_over_seq_eq_l394_394555

noncomputable theory

open_locale classical

def seq_a : ℕ → ℝ
def seq_b : ℕ → ℝ

def seq_a 0 := 3
def seq_b 0 := -1

def seq_a (n : ℕ) : ℝ := 
  seq_a n + seq_b n + real.sqrt ((seq_a n)^2 + (seq_b n)^2)

def seq_b (n : ℕ) : ℝ := 
  seq_a n + seq_b n - real.sqrt ((seq_a n)^2 + (seq_b n)^2)

theorem one_over_seq_eq : 
  (1 / seq_a 100 + 1 / seq_b 100) = -2 / 3 :=
sorry

end one_over_seq_eq_l394_394555


namespace distinct_count_floor_squares_div_2000_l394_394402

theorem distinct_count_floor_squares_div_2000 : 
  ∀ n : ℕ, n ≤ 1000 → 
    (finset.range (1000 + 1)).map (λ n, ⌊(n : ℕ)^2 / 2000⌋).to_finset.card = 501 :=
by
  sorry

end distinct_count_floor_squares_div_2000_l394_394402


namespace intersection_of_A_and_B_l394_394457

open Set

def A : Set ℕ := {1, 2, 3}
def B : Set ℤ := {x | x^2 < 9}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l394_394457


namespace total_fencing_cost_l394_394869

theorem total_fencing_cost
  (park_is_square : true)
  (cost_per_side : ℕ)
  (h1 : cost_per_side = 43) :
  4 * cost_per_side = 172 :=
by
  sorry

end total_fencing_cost_l394_394869


namespace intersection_locus_ellipse_and_min_perimeter_rectangle_l394_394752

noncomputable def fixed_base {α : Type*} [field α] (A B : α × α) := ∃ d : α, ∀ C D : α × α, AC = d ∧ BD = d
noncomputable def isosceles_trapezoid {α : Type*} [field α] (AB CD : α × α) := AB = CD

theorem intersection_locus_ellipse_and_min_perimeter_rectangle {α : Type*} 
  [field α] (A B C D : α × α) (d : α) (O : α × α) 
  (h_fixed_base : fixed_base A B)
  (h_isosceles : isosceles_trapezoid AB CD)
  (h_equal_diagonals : (AC = d) ∧ (BD = d)) :
  (locus_intersection O A B C D O → ellipse O) ∧ (min_perimeter A B C D A B C D → rectangle A B C D) :=
sorry

end intersection_locus_ellipse_and_min_perimeter_rectangle_l394_394752


namespace degree_measure_of_regular_hexagon_interior_angle_l394_394282

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l394_394282


namespace min_value_f_l394_394790

theorem min_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : 
  ∃ z : ℝ, z = x + y + x * y ∧ z = -9/8 :=
by 
  sorry

end min_value_f_l394_394790


namespace correct_propositions_l394_394461

variable (a b c : Type) -- Representing three distinct lines
variable (γ : Type) -- Representing a plane

-- Definitions and Propositions
def parallel (x y : Type) : Prop := sorry -- definition for parallel lines
def perpendicular (x y : Type) : Prop := sorry -- definition for perpendicular lines
def coplanar (x y z : Type) : Prop := sorry -- definition for coplanar lines

-- Propositions based on the conditions
def Prop1 := parallel a b ∧ parallel b c → parallel a c
def Prop4 := perpendicular a γ ∧ perpendicular b γ → parallel a b

-- The Lean theorem proving the correctness of identified propositions
theorem correct_propositions : Prop1 a b c ∧ Prop4 a b c γ :=
by {
  sorry -- The actual proof steps are omitted
}

end correct_propositions_l394_394461


namespace problem_expression_value_l394_394560

theorem problem_expression_value {a b c k1 k2 : ℂ} 
  (h_root : ∀ x, x^3 - k1 * x - k2 = 0 → x = a ∨ x = b ∨ x = c) 
  (h_condition : k1 + k2 ≠ 1)
  (h_vieta1 : a + b + c = 0)
  (h_vieta2 : a * b + b * c + c * a = -k1)
  (h_vieta3 : a * b * c = k2) :
  (1 + a)/(1 - a) + (1 + b)/(1 - b) + (1 + c)/(1 - c) = 
  (3 + k1 + 3 * k2)/(1 - k1 - k2) :=
by
  sorry

end problem_expression_value_l394_394560


namespace percy_bound_longer_martha_step_l394_394576

theorem percy_bound_longer_martha_step (steps_per_gap_martha: ℕ) (bounds_per_gap_percy: ℕ)
  (gaps: ℕ) (total_distance: ℕ) 
  (step_length_martha: ℝ) (bound_length_percy: ℝ) :
  steps_per_gap_martha = 50 →
  bounds_per_gap_percy = 15 →
  gaps = 50 →
  total_distance = 10560 →
  step_length_martha = total_distance / (steps_per_gap_martha * gaps) →
  bound_length_percy = total_distance / (bounds_per_gap_percy * gaps) →
  (bound_length_percy - step_length_martha) = 10 :=
by
  sorry

end percy_bound_longer_martha_step_l394_394576


namespace new_average_is_ten_l394_394612

-- Define the initial conditions
def initial_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : Prop :=
  x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ + x₈ + x₉ = 9 * 7

-- Define the transformation on the nine numbers
def transformed_sum (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) : ℝ :=
  (x₁ - 3) + (x₂ - 3) + (x₃ - 3) +
  (x₄ + 5) + (x₅ + 5) + (x₆ + 5) +
  (2 * x₇) + (2 * x₈) + (2 * x₉)

-- The theorem to prove the new average is 10
theorem new_average_is_ten (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ : ℝ) 
  (h : initial_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉) :
  transformed_sum x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ / 9 = 10 :=
by 
  sorry

end new_average_is_ten_l394_394612


namespace intersection_of_A_and_B_range_of_k_l394_394938

noncomputable def f (x : ℝ) : ℝ := real.sqrt (-x^2 + 2 * x + 3)
noncomputable def g (x : ℝ) : ℝ := x^2 - x + 1

def A : set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B : set ℝ := {y | 3/4 ≤ y}

theorem intersection_of_A_and_B : A ∩ B = {x | 3/4 ≤ x ∧ x ≤ 3} :=
sorry

theorem range_of_k (k : ℝ) : (∀ x > 0, g x ≥ k * x) → k ≤ 1 :=
sorry

end intersection_of_A_and_B_range_of_k_l394_394938


namespace incircle_radius_of_right_triangle_l394_394642

theorem incircle_radius_of_right_triangle 
  (D E F : Type) 
  [IsTriangle D E F] 
  [RightAngleAtF : RightAngle At F] 
  [AngleAtD : Measure.A D = 60] 
  (DF : Measure DF = 15) :
  RadiusOfIncircle = 5 * sqrt 3 := 
sorry

end incircle_radius_of_right_triangle_l394_394642


namespace regular_hexagon_interior_angle_l394_394258

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l394_394258


namespace fraction_of_smooth_integers_divisible_by_7_is_one_fourth_l394_394742

def is_even (n : ℕ) : Prop := n % 2 = 0

def digits_sum_to_10 (n : ℕ) : Prop := 
  (n / 10) + (n % 10) = 10

def is_smooth (n : ℕ) : Prop := 
  10 < n ∧ n < 100 ∧ is_even n ∧ digits_sum_to_10 n

def is_divisible_by_7 (n : ℕ) : Prop := n % 7 = 0

theorem fraction_of_smooth_integers_divisible_by_7_is_one_fourth :
  (∃ s : finset ℕ, ∃ t : finset ℕ, 
    (∀ n ∈ s, is_smooth n) ∧ 
    (∀ n ∈ t, is_smooth n ∧ is_divisible_by_7 n) ∧ 
    t.card = s.card / 4) := 
sorry

end fraction_of_smooth_integers_divisible_by_7_is_one_fourth_l394_394742


namespace interior_angle_of_regular_hexagon_l394_394272

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l394_394272


namespace nick_candy_bars_needed_l394_394104

-- Definitions based on conditions
def candy_bar_price : ℝ := 5
def chocolate_orange_price : ℝ := 10
def goal_amount : ℝ := 1000
def chocolate_oranges_sold : ℝ := 20
def chocolate_oranges_income : ℝ := chocolate_oranges_sold * chocolate_orange_price
def remaining_amount : ℝ := goal_amount - chocolate_oranges_income
def required_candy_bars : ℝ := remaining_amount / candy_bar_price

-- Proof statement
theorem nick_candy_bars_needed : required_candy_bars = 160 :=
sorry

end nick_candy_bars_needed_l394_394104


namespace onions_left_on_shelf_l394_394637

def initial_onions : ℕ := 98
def sold_onions : ℕ := 65
def remaining_onions : ℕ := initial_onions - sold_onions

theorem onions_left_on_shelf : remaining_onions = 33 :=
by 
  -- Proof would go here
  sorry

end onions_left_on_shelf_l394_394637


namespace number_of_people_seated_l394_394149

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l394_394149


namespace distinct_numbers_in_list_l394_394401

theorem distinct_numbers_in_list : 
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 1000) → 
  (finset.card (finset.image (λ n, int.floor (n^2 / 2000 : ℚ)) (finset.range 1000)) = 501) :=
by
  assume n hn,
  contradiction


end distinct_numbers_in_list_l394_394401


namespace simplify_trig_expression_l394_394971

theorem simplify_trig_expression (x : ℝ) (h : sin x ^ 2 + cos x ^ 2 = 1) :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * csc x := by
  sorry

end simplify_trig_expression_l394_394971


namespace sum_factorial_mod_21_l394_394512

open BigOperators
open Nat

theorem sum_factorial_mod_21 : 
  (∑ n in Finset.range 21, n.factorial) % 21 = 12 := 
sorry

end sum_factorial_mod_21_l394_394512


namespace inverse_proposition_l394_394181

   theorem inverse_proposition (x a b : ℝ) :
     (x ≥ a^2 + b^2 → x ≥ 2 * a * b) →
     (x ≥ 2 * a * b → x ≥ a^2 + b^2) :=
   sorry
   
end inverse_proposition_l394_394181


namespace smallest_area_of_triangle_l394_394079

open Real
open Vec3

def point := ℝ × ℝ × ℝ

def A : point := (-2, 3, 1)
def B : point := (2, 4, 2)
def C (s : ℝ) : point := (s, s, 2)

def cross_product (u v : point) : point :=
  (u.2.2 * v.2.1 - u.2.1 * v.2.2, u.2 * v.1 - u.1 * v.2, u.1 * v.2.1 - u.2.1 * v.1)

def vector_magnitude (v : point) : ℝ :=
  sqrt(v.1^2 + v.2.1^2 + v.2.2^2)

def area_of_triangle (A B C : point) : ℝ :=
  (1 / 2) * vector_magnitude (cross_product (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2) (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2))

theorem smallest_area_of_triangle : ∃ s : ℝ, area_of_triangle A B (C s) = sqrt 17 :=
sorry

end smallest_area_of_triangle_l394_394079


namespace degree_measure_of_regular_hexagon_interior_angle_l394_394281

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l394_394281


namespace central_number_l394_394491

theorem central_number (C : ℕ) (verts : Finset ℕ) (h : verts = {1, 2, 7, 8, 9, 13, 14}) :
  (∀ T ∈ {t | ∃ a b c, (a + b + c) % 3 = 0 ∧ a ∈ verts ∧ b ∈ verts ∧ c ∈ verts}, (T + C) % 3 = 0) →
  C = 9 :=
by
  sorry

end central_number_l394_394491


namespace projective_iff_fractional_linear_l394_394114

def projective_transformation (P : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))

theorem projective_iff_fractional_linear (P : ℝ → ℝ) : 
  projective_transformation P ↔ ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d)) :=
by 
  sorry

end projective_iff_fractional_linear_l394_394114


namespace tan_A_in_right_triangle_l394_394886

noncomputable def side_AC : ℕ := (34^2 - 30^2).sqrt -- Calculate AC using the Pythagorean theorem

theorem tan_A_in_right_triangle (A B C : Type) [RightTriangle A B C (angle_BAC := 90)] (AB BC : ℕ)
  (h1 : AB = 30) (h2 : BC = 34) :
  (tan_A : ℚ) = 8 / 15 :=
by
  -- Proof is omitted
  sorry

end tan_A_in_right_triangle_l394_394886


namespace simplify_trig_expression_l394_394974

open_locale real

theorem simplify_trig_expression (x : ℝ) (h : 1 + real.cos x ≠ 0) : 
  (real.sin x / (1 + real.cos x) + (1 + real.cos x) / real.sin x) = 2 * real.csc x :=
sorry

end simplify_trig_expression_l394_394974


namespace initial_wage_illiterate_l394_394056

variable (I : ℕ) -- initial daily average wage of illiterate employees

theorem initial_wage_illiterate (h1 : 20 * I - 20 * 10 = 300) : I = 25 :=
by
  simp at h1
  sorry

end initial_wage_illiterate_l394_394056


namespace ratio_P_K_is_2_l394_394585

theorem ratio_P_K_is_2 (P K M : ℝ) (r : ℝ)
  (h1: P + K + M = 153)
  (h2: P = r * K)
  (h3: P = (1/3) * M)
  (h4: M = K + 85) : r = 2 :=
  sorry

end ratio_P_K_is_2_l394_394585


namespace part_1_part_2_part_3_l394_394373

def f (x a b : ℝ) : ℝ := a*x^2 + (b+1)*x + b - 1

theorem part_1 (x : ℝ) : 
    f x 1 3 = x ↔ x = -2 ∨ x = -1 :=
by { unfold f, sorry }

theorem part_2 (a : ℝ) : 
    (∀ b : ℝ, ∃ x1 x2 : ℝ, f x1 a b = x1 ∧ f x2 a b = x2 ∧ x1 ≠ x2) ↔ 0 < a ∧ a < 1 :=
by { unfold f, sorry }

theorem part_3 (a : ℝ) (b : ℝ) (x1 x2 : ℝ) :
    0 < a ∧ a < 1 ∧ f x1 a b = x1 ∧ f x2 a b = x2 ∧ x1 ≠ x2 ∧
    (let C := (x1 + x2) / 2 in -C + (2 * a) / (5 * a ^ 2 - 4 * a + 1) = C) → b = -2 :=
by { unfold f, sorry }

end part_1_part_2_part_3_l394_394373


namespace term_containing_x3_l394_394605

-- Define the problem statement in Lean 4
theorem term_containing_x3 (a : ℝ) (x : ℝ) (hx : x ≠ 0) 
(h_sum_coeff : (2 + a) ^ 5 = 0) :
  (2 * x + a / x) ^ 5 = -160 * x ^ 3 :=
sorry

end term_containing_x3_l394_394605


namespace sum_of_numbers_is_919_l394_394959

-- Problem Conditions
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def is_three_digit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999
def satisfies_equation (x y : ℕ) : Prop := 1000 * x + y = 11 * x * y

-- Main Statement
theorem sum_of_numbers_is_919 (x y : ℕ) 
  (h1 : is_two_digit x) 
  (h2 : is_three_digit y) 
  (h3 : satisfies_equation x y) : 
  x + y = 919 := 
sorry

end sum_of_numbers_is_919_l394_394959


namespace cos_angle_sub_vectors_l394_394486

-- Definitions of the conditions
variables (a b c : ℝ^3) (ha : ‖ a ‖ = 1) (hb : ‖ b ‖ = 1) (hc : ‖ c ‖ = √2)
variable (h0 : a + b + c = 0)

-- The theorem statement
theorem cos_angle_sub_vectors : 
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry -- Proof omitted

end cos_angle_sub_vectors_l394_394486


namespace unique_distribution_of_darts_l394_394720

theorem unique_distribution_of_darts :
  ∃ l : List ℕ, (∀ x ∈ l, 1 ≤ x ∧ x ≤ 6) ∧ l.sorted (· ≥ ·) ∧ l.length = 5 ∧ Multiset.card l.to_multiset = 6 → l = [2, 1, 1, 1, 1] :=
by
  sorry

end unique_distribution_of_darts_l394_394720


namespace number_of_pairs_sold_is_65_l394_394667

-- Given conditions
def total_amount : ℝ := 637
def average_price_per_pair : ℝ := 9.8

-- Define the number of pairs
def number_of_pairs : ℝ := total_amount / average_price_per_pair

-- Prove that the number of pairs sold is 65
theorem number_of_pairs_sold_is_65 : number_of_pairs = 65 := by
  sorry

end number_of_pairs_sold_is_65_l394_394667


namespace count_polynomials_with_h_3_l394_394453

-- Definitions
def is_valid_polynomial (n : ℕ) (a_0 : ℕ) (a : Fin n → ℤ) : Prop :=
  a_0 > 0 ∧ h = n + a_0 + ∑ i, |a i|

def number_of_polynomials (h : ℕ) : ℕ :=
  ∑ n : ℕ, ∑ a_0 : ℕ, ∑ a : Fin n → ℤ, if is_valid_polynomial n a_0 a then 1 else 0

-- Main statement
theorem count_polynomials_with_h_3 : number_of_polynomials 3 = 5 :=
by
  sorry

end count_polynomials_with_h_3_l394_394453


namespace calculate_total_money_made_l394_394609

def original_price : ℕ := 51
def discount : ℕ := 8
def num_tshirts_sold : ℕ := 130
def discounted_price : ℕ := original_price - discount
def total_money_made : ℕ := discounted_price * num_tshirts_sold

theorem calculate_total_money_made :
  total_money_made = 5590 := 
sorry

end calculate_total_money_made_l394_394609


namespace table_seating_problem_l394_394141

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l394_394141


namespace probability_at_least_two_same_color_l394_394208

theorem probability_at_least_two_same_color :
  let white := 3
  let black := 4
  let red := 5
  let total_balls := white + black + red
  let ways_to_choose_three := Nat.choose total_balls 3
  let ways_no_two_same_color := white * black * red
  let probability_no_two_same_color := (ways_no_two_same_color : ℚ) / ways_to_choose_three
  let probability_at_least_two_same_color := 1 - probability_no_two_same_color
  probability_at_least_two_same_color = 8 / 11 :=
by
  have white_pos : white > 0 := by decide
  have black_pos : black > 0 := by decide
  have red_pos : red > 0 := by decide
  let total_balls := white + black + red
  have ways_to_choose_three_eq : ways_to_choose_three = 220 := by decide
  have ways_no_two_same_color_eq : ways_no_two_same_color = 60 := by decide
  have probability_no_two_same_color_eq : probability_no_two_same_color = (60 : ℚ) / 220 := by decide
  have probability_at_least_two_same_eq : probability_at_least_two_same_color = 1 - (60 : ℚ) / 220 := by decide
  simp [probability_at_least_two_same_eq, div_eq_mul_inv, ←nat_cast_eq_coe_nat, inv_mul_eq_div, show 1 - (60 : ℚ) / 220 = 8 / 11 by norm_num]
  sorry

end probability_at_least_two_same_color_l394_394208


namespace number_of_zeros_of_odd_function_l394_394189

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Statement of the proof problem
theorem number_of_zeros_of_odd_function
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_zeros_nonneg : ∃ a b : ℝ, 0 ≤ a ∧ 0 < b ∧ f(a) = 0 ∧ f(b) = 0 ∧ a ≠ b) :
  (∃ c d e : ℝ, c < d ∧ d < e ∧ f(c) = 0 ∧ f(d) = 0 ∧ f(e) = 0 ∧ c ≠ d ∧ d ≠ e ∧ c ≠ e) :=
sorry

end number_of_zeros_of_odd_function_l394_394189


namespace increasing_log_function_l394_394827

noncomputable def g (a : ℝ) (x : ℝ) := a * x^2 - x

noncomputable def is_increasing (a : ℝ) (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

noncomputable def log_base_a (a x : ℝ) := real.log x / real.log a

theorem increasing_log_function (a : ℝ) :
  (∃ (I : set ℝ), I = set.Icc 2 4 ∧ is_increasing a (λ x, log_base_a a (g a x)) I) ↔ a ∈ set.Ioi 1 :=
begin
  sorry
end

end increasing_log_function_l394_394827


namespace simplify_trig_expression_l394_394975

open_locale real

theorem simplify_trig_expression (x : ℝ) (h : 1 + real.cos x ≠ 0) : 
  (real.sin x / (1 + real.cos x) + (1 + real.cos x) / real.sin x) = 2 * real.csc x :=
sorry

end simplify_trig_expression_l394_394975


namespace determine_final_round_l394_394629

noncomputable def can_enter_final_round (student_score : ℕ) (scores : list ℕ) : Prop :=
  let sorted_scores := list.sorted scores in
  (sorted_scores.nth_le 9 sorry) <= student_score

theorem determine_final_round (student_score : ℕ) (scores : list ℕ) :
  list.length scores = 19 ∧ list.nodup scores →
  can_enter_final_round student_score scores :=
begin
  sorry -- Skipping proof as required
end

end determine_final_round_l394_394629


namespace isosceles_right_triangle_inscribed_center_lines_l394_394753

noncomputable def inscribed_circle_center (A B C : Point) (ABC : Triangle A B C) : Point := sorry

theorem isosceles_right_triangle_inscribed_center_lines 
  (A B C Aa D1 D2 : Point)
  (hTriangle : Triangle A B C)
  (hAltitudeAa : Altitude A Aa (Line B C))
  (hPerpendicularD1 : Perpendicular Aa D1 (Line A B))
  (hPerpendicularD2 : Perpendicular Aa D2 (Line A C)) :
  let O1 := inscribed_circle_center Aa B D1 (Triangle.build Aa B D1)
  let O2 := inscribed_circle_center A Aa D1 (Triangle.build A Aa D1)
  let O3 := inscribed_circle_center A Aa D2 (Triangle.build A Aa D2)
  let O4 := inscribed_circle_center Aa C D2 (Triangle.build Aa C D2)
  (Line O1 O2).same_side (Line A Aa) ∧ 
  (Line O3 O4).same_side (Line A Aa) ∧ 
  IsoscelesRightTriangle (Triangle.build O1 O2 O3) ∧ 
  HypotenuseOnBC (Line O1 O3) (Line B C) :=
sorry

end isosceles_right_triangle_inscribed_center_lines_l394_394753


namespace two_digit_numbers_of_power_of_two_l394_394032

theorem two_digit_numbers_of_power_of_two : 
  {n : ℕ | 10 ≤ 2^n ∧ 2^n ≤ 99}.to_finset.card = 3 :=
by
  sorry

end two_digit_numbers_of_power_of_two_l394_394032


namespace interior_angle_of_regular_hexagon_l394_394276

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ sumInteriorAngles : ℕ, sumInteriorAngles = (n - 2) * 180) →
  ∀ (interiorAngle : ℕ), (∃ sumInteriorAngles : ℕ, sumInteriorAngles = 720) → 
  interiorAngle = sumInteriorAngles / 6 →
  interiorAngle = 120 :=
by
  sorry

end interior_angle_of_regular_hexagon_l394_394276


namespace sandbox_sand_weight_l394_394697

theorem sandbox_sand_weight
  (side_len : ℝ) 
  (side_len_eq : side_len = 40)
  (bag_weight : ℝ)
  (bag_weight_eq : bag_weight = 30)
  (bag_coverage : ℝ)
  (bag_coverage_eq : bag_coverage = 80) :
  let area := side_len * side_len in
  let num_bags := area / bag_coverage in
  let total_weight := num_bags * bag_weight in
  total_weight = 600 := by
  sorry

end sandbox_sand_weight_l394_394697


namespace regular_hexagon_interior_angle_measure_l394_394234

theorem regular_hexagon_interior_angle_measure :
  let n := 6
  let sum_of_angles := (n - 2) * 180
  let measure_of_each_angle := sum_of_angles / n
  measure_of_each_angle = 120 :=
by
  sorry

end regular_hexagon_interior_angle_measure_l394_394234


namespace tangent_lines_coincide_l394_394437

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f x / x

theorem tangent_lines_coincide : 
  f(2) = 2 ∧ 
  (∀ x, g x = f x / x) ∧ 
  (2 * deriv f 2 - f 2) / 4 = 1/2 → 
  deriv f 2 = 2 :=
by
  intros h,
  cases' h with h1 h2,
  cases' h2 with h2 h3,
  have : f 2 = 2 := h1,
  have : (2 * deriv f 2 - f 2) / 4 = 1 / 2 := h3,
  sorry

end tangent_lines_coincide_l394_394437


namespace calc_val_l394_394361

theorem calc_val : 
  (3 + 5 + 7) / (2 + 4 + 6) * (4 + 8 + 12) / (1 + 3 + 5) = 10 / 3 :=
by 
  -- Calculation proof
  sorry

end calc_val_l394_394361


namespace compute_expression_l394_394663

theorem compute_expression :
  (1 / 36) / ((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) + 
  (((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) / (1 / 36)) = -10 / 3 :=
by
  sorry

end compute_expression_l394_394663


namespace add_salt_solution_l394_394843

theorem add_salt_solution
  (initial_amount : ℕ) (added_concentration : ℕ) (desired_concentration : ℕ)
  (initial_concentration : ℝ) :
  initial_amount = 50 ∧ initial_concentration = 0.4 ∧ added_concentration = 10 ∧ desired_concentration = 25 →
  (∃ (x : ℕ), x = 50 ∧ 
    (initial_concentration * initial_amount + 0.1 * x) / (initial_amount + x) = 0.25) :=
by
  sorry

end add_salt_solution_l394_394843


namespace cylinder_cube_volume_ratio_l394_394686

theorem cylinder_cube_volume_ratio (s : ℝ) (hs : 0 < s) :
  let r := s / 2 in
  let V_cylinder := π * r^2 * s in
  let V_cube := s^3 in
  V_cylinder / V_cube = π / 4 :=
by
  dsimp
  sorry

end cylinder_cube_volume_ratio_l394_394686


namespace find_value_l394_394611

-- Define the mean, standard deviation, and the number of standard deviations
def mean : ℝ := 17.5
def std_dev : ℝ := 2.5
def num_std_dev : ℝ := 2.7

-- The theorem to prove that the value is exactly 10.75
theorem find_value : mean - (num_std_dev * std_dev) = 10.75 := 
by
  sorry

end find_value_l394_394611


namespace find_a_l394_394819

-- Define the function f(x)
def f (a x : ℝ) : ℝ := Math.log (x^2 - 2 * a * x + 3) / Math.log (1/2)

-- State the condition that f(x) is an even function
def is_even_function (a : ℝ) : Prop := ∀ x : ℝ, f a x = f a (-x)

-- The main theorem we want to prove
theorem find_a (a : ℝ) (h_even : is_even_function a) : a = 0 := 
by
  sorry

end find_a_l394_394819


namespace cos_angle_sub_vectors_l394_394489

-- Definitions of the conditions
variables (a b c : ℝ^3) (ha : ‖ a ‖ = 1) (hb : ‖ b ‖ = 1) (hc : ‖ c ‖ = √2)
variable (h0 : a + b + c = 0)

-- The theorem statement
theorem cos_angle_sub_vectors : 
  real.cos ⟪a - c, b - c⟫ = 4 / 5 :=
sorry -- Proof omitted

end cos_angle_sub_vectors_l394_394489


namespace total_people_seated_l394_394153

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l394_394153


namespace triangle_inequality_l394_394918

open Real

variables {a b c S : ℝ}

-- Assuming a, b, c are the sides of a triangle
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
-- Assuming S is the area of the triangle
axiom Herons_area : S = sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_inequality : 
  a^2 + b^2 + c^2 ≥ 4 * S * sqrt 3 ∧ (a^2 + b^2 + c^2 = 4 * S * sqrt 3 ↔ a = b ∧ b = c) := sorry

end triangle_inequality_l394_394918


namespace percent_a_in_solution_y_l394_394993

-- Define variables and parameters
variables (A : ℝ) -- percentage of chemical a in solution y
constants (x_a : ℝ) (x_b : ℝ) (mix_a : ℝ) (frac_x : ℝ) (frac_y : ℝ)
constants (soln_y : ℝ)

-- Define the given conditions
axiom condition1 : x_a = 0.4 -- solution x is 40% a
axiom condition2 : x_b = 0.6 -- solution x is 60% b
axiom condition3 : mix_a = 0.47 -- mixture is 47% a
axiom condition4 : frac_x = 0.30 -- 30% of the mixture is solution x
axiom condition5 : frac_y = 0.70 -- 70% of the mixture is solution y
axiom condition6 : soln_y = A -- solution y is A% a and A% b

-- State the theorem
theorem percent_a_in_solution_y : A = 0.5 :=
by
  -- Here would be the detailed proof, which is omitted
  sorry

end percent_a_in_solution_y_l394_394993


namespace sequence_divisible_by_11_l394_394833

theorem sequence_divisible_by_11 
  (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
  (∀ n, n = 4 ∨ n = 8 ∨ n ≥ 10 → 11 ∣ a n) := sorry

end sequence_divisible_by_11_l394_394833


namespace cos_B_of_geometric_sequence_l394_394513

theorem cos_B_of_geometric_sequence
  {a b c : ℝ} (h_geom_seq : b^2 = a * c) (h_eqn : 2 * c - 4 * a = 0) :
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c)
  in cos_B = 3 / 4 := sorry

end cos_B_of_geometric_sequence_l394_394513


namespace number_of_people_seated_l394_394148

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l394_394148


namespace simplify_expression_l394_394984

-- Define the conditions
variable (x : ℝ)
variable (hx : sin x ≠ 0)

-- Define the statement
theorem simplify_expression :
  (sin x / (1 + cos x) + (1 + cos x) / sin x) = 2 * (1 / sin x) :=
by sorry

end simplify_expression_l394_394984


namespace son_l394_394306

theorem son's_age (S M : ℕ) 
  (h1 : M = S + 24) 
  (h2 : M + 2 = 2 * (S + 2)) : S = 22 := 
by 
  sorry

end son_l394_394306


namespace possible_triples_l394_394498

theorem possible_triples (x y z : ℤ) :
  (xyz + 4 * (x + y + z) = 2 * (xy + xz + yz) + 7) ↔
  (x = 1 ∧ y = 1 ∧ z = 1) ∨
  (x = 3 ∧ y = 3 ∧ z = 1) ∨
  (x = 3 ∧ y = 1 ∧ z = 3) ∨
  (x = 1 ∧ y = 3 ∧ z = 3) :=
begin
  sorry
end

end possible_triples_l394_394498


namespace Paco_has_salty_cookies_left_l394_394110

-- Given conditions:
def initial_salty_cookies := 26
def salty_cookies_shared_with_Ana := 11
def salty_cookies_shared_with_Juan := 3

-- Salty cookies left calculation:
def salty_cookies_left := initial_salty_cookies - (salty_cookies_shared_with_Ana + salty_cookies_shared_with_Juan)

theorem Paco_has_salty_cookies_left : salty_cookies_left = 12 := by
  -- Expanding the definition to alleviate the condition
  have total_shared : salty_cookies_shared_with_Ana + salty_cookies_shared_with_Juan = 14 := by
    rfl

  -- Proving the final amount
  show initial_salty_cookies - 14 = 12
  calc
    initial_salty_cookies - 14
    = 26 - 14 : by rfl
    ... = 12 : by norm_num

end Paco_has_salty_cookies_left_l394_394110


namespace b2009_value_l394_394556

noncomputable def b (n : ℕ) : ℝ := sorry

axiom b_recursion (n : ℕ) (hn : 2 ≤ n) : b n = b (n - 1) * b (n + 1)

axiom b1_value : b 1 = 2 + Real.sqrt 3
axiom b1776_value : b 1776 = 10 + Real.sqrt 3

theorem b2009_value : b 2009 = -4 + 8 * Real.sqrt 3 := 
by sorry

end b2009_value_l394_394556


namespace six_dice_divisible_by_seven_l394_394162

-- Definitions and conditions
def Die : Type := {n : Nat // n > 0 ∧ n < 7}
def opposite_faces_sum_seven (d : Die) : Prop := d.1 + (7 - d.1) = 7

-- Proof problem: Given conditions
theorem six_dice_divisible_by_seven :
  ∃ (d1 d2 d3 d4 d5 d6 : Die),
    (opposite_faces_sum_seven d1) ∧
    (opposite_faces_sum_seven d2) ∧
    (opposite_faces_sum_seven d3) ∧
    (opposite_faces_sum_seven d4) ∧
    (opposite_faces_sum_seven d5) ∧
    (opposite_faces_sum_seven d6) ∧
    (let n := d1.1 * 10^5 + d2.1 * 10^4 + d3.1 * 10^3 + d4.1 * 10^2 + d5.1 * 10 + d6.1
     in n % 7 = 0) :=
sorry

end six_dice_divisible_by_seven_l394_394162


namespace lateral_surface_area_is_12_l394_394711

noncomputable def lateral_surface_area_of_pyramid
  (V : ℝ) (a : ℝ) : ℝ :=
  let base_area := 6 * Real.sqrt 3 in
  let h := V * 3 / base_area in
  let s := Real.sqrt (h^2 + (a / Real.sqrt 3)^2) in
  6 * (a * s / 2)

theorem lateral_surface_area_is_12 :
  ∀ (V : ℝ) (a : ℝ), V = 2 * Real.sqrt 3 → a = 2 → 
  lateral_surface_area_of_pyramid V a = 12 := 
by
  intros V a hV ha
  rw [hV, ha]
  simp
  sorry

end lateral_surface_area_is_12_l394_394711


namespace find_f_neg_one_l394_394002

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (1 + x) else - f (-x)

theorem find_f_neg_one :
    f(-1) = -2 :=
by
  -- Proof to be filled
  sorry

end find_f_neg_one_l394_394002


namespace problem_proof_l394_394108

theorem problem_proof :
  (1 == "Promote") ∧
  (2 == "Good") ∧
  (3 == "Leave") ∧
  (4 == "Wonder/Surprise") ∧
  (5 == "Difficult") ∧
  (6 == "Eagerly") ∧
  (7 == "Presence") ∧
  (8 == "Empty") ∧
  (9 == "Realize") ∧
  (10 == "It was then that") ∧
  (11 == "Balance") ∧
  (12 == "Face") ∧
  (13 == "Positive") ∧
  (14 == "Remind") ∧
  (15 == "Manage") ∧
  (16 == "Independent") ∧
  (17 == "Goal") ∧
  (18 == "Experience") ∧
  (19 == "Get rewarded") ∧
  (20 == "Blessing") :=
  by sorry

end problem_proof_l394_394108


namespace people_at_table_l394_394125

theorem people_at_table (n : ℕ)
  (h1 : ∃ (d : ℕ), d > 0 ∧ forall i : ℕ, 1 ≤ i ∧ i < n → (i + d) % n ≠ (31 % n))
  (h2 : ((31 - 7) % n) = ((31 - 14) % n)) :
  n = 41 := 
sorry

end people_at_table_l394_394125


namespace common_noninteger_root_eq_coeffs_l394_394961

theorem common_noninteger_root_eq_coeffs (p1 p2 q1 q2 : ℤ) (α : ℝ) :
  (α^2 + (p1: ℝ) * α + (q1: ℝ) = 0) ∧ (α^2 + (p2: ℝ) * α + (q2: ℝ) = 0) ∧ ¬(∃ (k : ℤ), α = k) → p1 = p2 ∧ q1 = q2 :=
by {
  sorry
}

end common_noninteger_root_eq_coeffs_l394_394961


namespace largest_area_of_G1G2G3_l394_394919

namespace Geometry

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def triangle_centroid (A B C : Point) : Point :=
{ x := (A.x + B.x + C.x) / 3,
  y := (A.y + B.y + C.y) / 3 }

noncomputable def triangle_area (A B C : Point) : ℝ :=
0.5 * abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))

variables (A1 B1 C1 A2 B2 C2 A3 B3 C3 : Point)

def D1 := { x := (B1.x + C1.x) / 2, y := (B1.y + C1.y) / 2 }
def E1 := { x := (A1.x + C1.x) / 2, y := (A1.y + C1.y) / 2 }
def F1 := { x := (A1.x + B1.x) / 2, y := (A1.y + B1.y) / 2 }
def D2 := { x := (B2.x + C2.x) / 2, y := (B2.y + C2.y) / 2 }
def E2 := { x := (A2.x + C2.x) / 2, y := (A2.y + C2.y) / 2 }
def F2 := { x := (A2.x + B2.x) / 2, y := (A2.y + B2.y) / 2 }
def D3 := { x := (B3.x + C3.x) / 2, y := (B3.y + C3.y) / 2 }
def E3 := { x := (A3.x + C3.x) / 2, y := (A3.y + C3.y) / 2 }
def F3 := { x := (A3.x + B3.x) / 2, y := (A3.y + B3.y) / 2 }

def G1 := triangle_centroid A1 B1 C1
def G2 := triangle_centroid A2 B2 C2
def G3 := triangle_centroid A3 B3 C3

axiom area_A1A2A3 : triangle_area A1 A2 A3 = 2
axiom area_B1B2B3 : triangle_area B1 B2 B3 = 3
axiom area_C1C2C3 : triangle_area C1 C2 C3 = 4
axiom area_D1D2D3 : triangle_area D1 D2 D3 = 20
axiom area_E1E2E3 : triangle_area E1 E2 E3 = 21
axiom area_F1F2F3 : triangle_area F1 F2 F3 = 2020

noncomputable def G1G2G3_area : ℝ :=
triangle_area G1 G2 G3

theorem largest_area_of_G1G2G3 : G1G2G3_area A1 B1 C1 A2 B2 C2 A3 B3 C3 = 917 :=
sorry

end Geometry

end largest_area_of_G1G2G3_l394_394919


namespace max_value_of_sin_cos_sum_l394_394826

theorem max_value_of_sin_cos_sum :
  ∀ x : ℝ, ∃ y ≤ 2, y = sin (x / 2) + sqrt 3 * cos (x / 2) :=
by
  sorry

end max_value_of_sin_cos_sum_l394_394826


namespace minimum_h25_l394_394571

def is_tenuous (h : ℕ → ℕ) : Prop :=
  ∀ (x y : ℕ), x > 0 → y > 0 → h(x) + h(y) > (x + y)^2

noncomputable def sum_h (h : ℕ → ℕ) : ℕ :=
  ∑ i in Finset.range 30, h (i + 1)

theorem minimum_h25 (h : ℕ → ℕ) (h_tenuous: is_tenuous h)
  (h_min_sum: sum_h h = 15376) : h 25 = 480 :=
sorry

end minimum_h25_l394_394571


namespace ten_sided_polygon_diagonals_l394_394684

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem ten_sided_polygon_diagonals :
  number_of_diagonals 10 = 35 :=
by sorry

end ten_sided_polygon_diagonals_l394_394684


namespace intersection_of_lines_l394_394396

theorem intersection_of_lines :
  ∃ x y : ℚ, 12 * x - 5 * y = 8 ∧ 10 * x + 2 * y = 20 ∧ x = 58 / 37 ∧ y = 667 / 370 :=
by
  sorry

end intersection_of_lines_l394_394396


namespace greatest_root_of_g_l394_394393

noncomputable def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem greatest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y, g y = 0 → y ≤ x := 
begin
  let x := sqrt (3 / 7),
  use x,
  split,
  { sorry },
  { intros y hy,
    sorry },
end

end greatest_root_of_g_l394_394393


namespace fraction_product_l394_394649

theorem fraction_product : (1 / 4 : ℚ) * (2 / 5) * (3 / 6) = (1 / 20) := by
  sorry

end fraction_product_l394_394649


namespace number_of_primes_under_150_with_ones_digit_3_l394_394848

noncomputable def primes_under_150_with_ones_digit_3 : Finset ℕ :=
  Finset.filter (λ n, Nat.Prime n) (Finset.filter (λ n, n % 10 = 3) (Finset.range 150))

theorem number_of_primes_under_150_with_ones_digit_3 :
  Finset.card primes_under_150_with_ones_digit_3 = 9 :=
by
  sorry

end number_of_primes_under_150_with_ones_digit_3_l394_394848


namespace not_in_eighth_row_l394_394067

theorem not_in_eighth_row (a b c : ℤ) (h_distinct: a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ¬ ∃ row_i, row_i > 7 ∧ 855 ∈ row_elems (next_rows (list.replicate 1 [a, b, c]) row_i) := sorry

end not_in_eighth_row_l394_394067


namespace sum_first_n_terms_b_l394_394803

-- Defining the arithmetic sequence and its properties
def a (n : ℕ) : ℕ := 2 * n - 1

-- Define the sum of first n terms of the sequence a_n
def S (n : ℕ) : ℕ := n^2

-- Define the sequence b_n
def b (n : ℕ) : ℝ := (a (n + 2) : ℝ) / (2 ^ n * a n * a (n + 1))

-- Sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℝ := (∑ k in Finset.range n, b k)

theorem sum_first_n_terms_b (n : ℕ) : T n = 1 - (1 / (2 ^ n * (2 * n + 1))) :=
by sorry

end sum_first_n_terms_b_l394_394803


namespace tangent_triangle_area_eq_e_div_4_l394_394172

-- Define the curve 
def f (x : ℝ) : ℝ := x * Real.exp x

-- Define the point of tangency
def P : ℝ × ℝ := (1, Real.exp 1)

-- State the theorem
theorem tangent_triangle_area_eq_e_div_4 :
  let x_intercept := 1 / 2
  let y_intercept := -Real.exp 1
  let base := (1 : ℝ) / 2
  let height := abs y_intercept
  let area := (1 / 2) * base * height
  area = Real.exp 1 / 4 :=
by
  sorry

end tangent_triangle_area_eq_e_div_4_l394_394172


namespace value_of_k_l394_394508

def equation (x k : ℝ) : Prop := 2 / (x - 1) - k / (1 - x) = 1 

theorem value_of_k (k : ℝ) : (∃ x : ℝ, x > 0 ∧ 2 / (x - 1) - k / (1 - x) = 1) → k = -2 := 
by 
  assume h : ∃ x : ℝ, x > 0 ∧ 2 / (x - 1) - k / (1 - x) = 1 
  sorry 

end value_of_k_l394_394508


namespace area_of_EPHQ_l394_394120

variable {P Q E F G H : EuclideanGeometry.Point}
variable (EF FG GH HE : ℝ)
variable (rect : EuclideanGeometry.Rectangle E F G H)
variable (midpoint_FG: EuclideanGeometry.midpoint F G P)
variable (midpoint_GH: EuclideanGeometry.midpoint G H Q)

theorem area_of_EPHQ (hEF : EF = 6) (hFG_GH : FG = GH = 10) (hArea: EF * FG = 60) :
  EuclideanGeometry.area (EuclideanGeometry.Quadrilateral E P H Q) == 30 :=
by
  sorry

end area_of_EPHQ_l394_394120


namespace convert_and_compute_l394_394354

noncomputable def base4_to_base10 (n : ℕ) : ℕ :=
  if n = 231 then 2 * 4^2 + 3 * 4^1 + 1 * 4^0
  else if n = 21 then 2 * 4^1 + 1 * 4^0
  else if n = 3 then 3
  else 0

noncomputable def base10_to_base4 (n : ℕ) : ℕ :=
  if n = 135 then 2 * 4^2 + 1 * 4^1 + 3 * 4^0
  else 0

theorem convert_and_compute :
  base10_to_base4 ((base4_to_base10 231 / base4_to_base10 3) * base4_to_base10 21) = 213 :=
by {
  sorry
}

end convert_and_compute_l394_394354


namespace exactly_two_correct_propositions_l394_394347

def isosceles_triangle_with_exterior_angle_120_is_equilateral : Prop :=
  ∀ (T : Triangle), T.is_isosceles ∧ T.exterior_angle = 120 → T.is_equilateral

def two_isosceles_triangles_with_equal_exterior_angles_are_equilateral : Prop :=
  ∀ (T1 T2 : Triangle), T1.is_isosceles ∧ T2.is_isosceles ∧ T1.exterior_angle = T2.exterior_angle → 
  T1.is_equilateral ∧ T2.is_equilateral

def triangle_with_height_as_median_is_equilateral : Prop :=
  ∀ (T : Triangle), (∀ (A B C : Point), T.height_on_side_is_median A B C) → T.is_equilateral

def triangle_with_equal_exterior_angles_is_equilateral : Prop :=
  ∀ (T : Triangle), (∃ d, ∀ (A B C : Angle), T.exterior_angle A = d ∧ T.exterior_angle B = d ∧ T.exterior_angle C = d) → T.is_equilateral

theorem exactly_two_correct_propositions : 
  isosceles_triangle_with_exterior_angle_120_is_equilateral ∧ 
  two_isosceles_triangles_with_equal_exterior_angles_are_equilateral ∧ 
  triangle_with_height_as_median_is_equilateral ∧ 
  triangle_with_equal_exterior_angles_is_equilateral →
  num_of_correct_propositions = 2 :=
sorry

end exactly_two_correct_propositions_l394_394347


namespace fraction_simplifies_correctly_l394_394856

variable (a b : ℕ)

theorem fraction_simplifies_correctly (h : a ≠ b) : (1/2 * a) / (1/2 * b) = a / b := 
by sorry

end fraction_simplifies_correctly_l394_394856


namespace rocky_total_miles_l394_394060

theorem rocky_total_miles :
  let d1 := 4 in
  let d2 := 2 * d1 in
  let d3 := 3 * d2 in
  d1 + d2 + d3 = 36 :=
by
  let d1 := 4
  let d2 := 2 * d1
  let d3 := 3 * d2
  sorry

end rocky_total_miles_l394_394060


namespace hash_non_distributive_l394_394921

def hash (a b : ℝ) : ℝ := a + 2*b

theorem hash_non_distributive (x y z : ℝ) :
  ¬ (∀ x y z, x # (y + z) = (x # y) + (x # z)) ∧ 
  ¬ (∀ x y z, x + (y # z) = (x + y) # (x + z)) ∧
  ¬ (∀ x y z, x # (y # z) = (x # y) # (x # z)) :=
by
  sorry

end hash_non_distributive_l394_394921


namespace pedro_plums_l394_394586

theorem pedro_plums :
  ∃ P Q : ℕ, P + Q = 32 ∧ 2 * P + Q = 52 ∧ P = 20 :=
by
  sorry

end pedro_plums_l394_394586


namespace smallest_k_divides_ab_l394_394930

theorem smallest_k_divides_ab (S : Finset ℕ) (hS : S = Finset.range 51)
  (k : ℕ) : (∀ T : Finset ℕ, T ⊆ S → T.card = k → ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b)) ↔ k = 39 :=
by
  sorry

end smallest_k_divides_ab_l394_394930


namespace tangent_lines_coincide_l394_394438

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f x / x

theorem tangent_lines_coincide : 
  f(2) = 2 ∧ 
  (∀ x, g x = f x / x) ∧ 
  (2 * deriv f 2 - f 2) / 4 = 1/2 → 
  deriv f 2 = 2 :=
by
  intros h,
  cases' h with h1 h2,
  cases' h2 with h2 h3,
  have : f 2 = 2 := h1,
  have : (2 * deriv f 2 - f 2) / 4 = 1 / 2 := h3,
  sorry

end tangent_lines_coincide_l394_394438


namespace scientific_notation_of_876000_l394_394713

theorem scientific_notation_of_876000 :
  ∃ a n, (1 ≤ |a| ∧ |a| < 10) ∧ (Int n) ∧ (876000 = a * 10^n) ∧ (a = 8.76) ∧ (n = 5) := 
by
  use 8.76
  use 5
  split
  sorry -- proof that 1 ≤ |a| < 10
  split
  sorry -- proof that n is an integer
  split
  sorry -- proof that 876000 = a * 10^n
  split
  sorry -- proof that a = 8.76
  sorry -- proof that n = 5

end scientific_notation_of_876000_l394_394713


namespace log_eq_imp_x_eq_b_l394_394425

theorem log_eq_imp_x_eq_b {b x : ℝ} (hb_pos : b > 0) (hb_ne_one : b ≠ 1) (hx_ne_one : x ≠ 1)
  (h : log (b^2) x + log (x^2) b = 1) : x = b :=
sorry

end log_eq_imp_x_eq_b_l394_394425


namespace S_max_min_sum_l394_394964

-- Define the variables and conditions
variables (x y : ℝ)

-- Given condition
def condition := 4 * x^2 - 5 * x * y + 4 * y^2 = 5

-- Define S
def S := x^2 + y^2

-- Prove the main statement
theorem S_max_min_sum (h : condition x y) : 
  (1 / (max S S)) + (1 / (min S S)) = 8 / 5 := by
  sorry

end S_max_min_sum_l394_394964


namespace num_paths_l394_394167

theorem num_paths (X Y : ℕ) (P : ℕ) :
    (∀ (x y : ℕ), P!x y (x + 1, y) ∨ P!x y (x + 1, y + 3))
    → X = 9
    → Y = 9
    → number_of_paths (0, 0) (X, Y) = 84 :=
begin
  sorry
end

end num_paths_l394_394167


namespace proof_problem_l394_394592

def problem : Prop :=
  ∃ (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004

theorem proof_problem : 
  problem → 
  ∃! (x y z t : ℤ), 
    0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ t ∧ 
    x^2 + y^2 + z^2 + t^2 = 2^2004 :=
sorry

end proof_problem_l394_394592


namespace possible_values_of_b_over_c_l394_394529

theorem possible_values_of_b_over_c 
  (A B C a b c : ℝ)
  (hA : A = π / 6)
  (h1 : a + b * cos C = cos B + 4 * cos C)
  (h_triangle_abc : a^2 = b^2 + c^2 - 2 * b * c * cos A) : 
  ∃ x, x = b / c ∧ (x = sqrt 3 / 2 ∨ x = 2) :=
sorry

end possible_values_of_b_over_c_l394_394529


namespace domain_of_f_l394_394768

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x^2 + 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f y = x} = {x : ℝ | x ∉ {-3, -1}} :=
by
  sorry

end domain_of_f_l394_394768


namespace find_a_l394_394016

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sin x - 2 * x - a

theorem find_a (a : ℝ) :
  (∀ x ∈ Icc 0 Real.pi, f x a ≤ f 0 a) ∧ f 0 a = -1 →
  a = 1 :=
by
  intro h
  have h1 : f 0 a = -a := by simp [f]
  rw [h1] at h
  have h2 := h.2
  rw [← h2]
  linarith
  sorry

end find_a_l394_394016


namespace ruda_received_clock_on_correct_date_l394_394122

/-- Ruda's clock problem -/
def ruda_clock_problem : Prop :=
  ∃ receive_date : ℕ → ℕ × ℕ × ℕ, -- A function mapping the number of presses to a date (Year, Month, Day)
  (∀ days_after_received, 
    receive_date days_after_received = 
    if days_after_received <= 45 then (2022, 10, 27 - (45 - days_after_received)) -- Calculating the receive date.
    else receive_date 45)
  ∧
  receive_date 45 = (2022, 12, 11) -- The day he checked the clock has to be December 11th

-- We want to prove that:
theorem ruda_received_clock_on_correct_date : ruda_clock_problem :=
by
  sorry

end ruda_received_clock_on_correct_date_l394_394122


namespace find_q_l394_394502

def quadratic_min_value (p q : ℝ) : Prop :=
  let y := λ (x : ℝ), x^2 - p*x + q
  inf (set.range y) = 1

theorem find_q (p q : ℝ) (h : quadratic_min_value p q) : q = 1 + p^2 / 4 :=
by sorry

end find_q_l394_394502


namespace reinforcement_calculation_l394_394665

theorem reinforcement_calculation :
  let men := 2000 in
  let days_initial := 54 in
  let days_used := 21 in
  let days_remaining := 20 in
  let provisions_initial := men * days_initial in
  let provisions_used := men * days_used in
  let provisions_remaining := provisions_initial - provisions_used in
  let total_men_after_reinforcement := men + 1300 in
  provisions_remaining = total_men_after_reinforcement * days_remaining :=
by
  sorry

end reinforcement_calculation_l394_394665


namespace quadratic_function_inequality_l394_394823

theorem quadratic_function_inequality
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : a + b + c = 0) :
  ∀ x ∈ Ioo 0 1, (λ x : ℝ, a * x^2 + b * x + c) x < 0 := sorry

end quadratic_function_inequality_l394_394823


namespace exists_one_tangent_circle_logarithmic_inequality_counter_subset_counterexample_l394_394382

-- Statement 1
theorem exists_one_tangent_circle (P Q : Point) (l : Line) (hPQ : P ≠ Q) (hPQ_same_side : same_side P Q l) :
  ¬(∃ c1 c2 : Circle, c1 ≠ c2 ∧ tangent c1 l ∧ tangent c2 l ∧ passes_through P c1 ∧ passes_through Q c1 ∧ passes_through P c2 ∧ passes_through Q c2) :=
by
  sorry

-- Statement 2
theorem logarithmic_inequality_counter (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne1 : a ≠ 1) (hne2 : b ≠ 1) :
  ∃ a b : ℝ, log a b + log b a < 2 :=
by
  sorry

-- Statement 3
theorem subset_counterexample (A B : set (ℝ × ℝ)) (C_r : ℝ → set (ℝ × ℝ)) (h_Cr : ∀ r ≥ 0, C_r r = {p | p.1^2 + p.2^2 ≤ r^2})
    (cond : ∀ r ≥ 0, C_r r ∪ A ⊆ C_r r ∪ B) : ¬ (A ⊆ B) :=
by
  sorry

end exists_one_tangent_circle_logarithmic_inequality_counter_subset_counterexample_l394_394382


namespace cube_side_length_l394_394602

-- Definitions of vertices with given coordinates
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def A0 : Point := { x := 0, y := 0, z := 0 }
def A1 : Point := { x := 5, y := 0, z := 0 }
def A2 : Point := { x := 0, y := 12, z := 0 }
def A3 : Point := { x := 0, y := 0, z := 9 }

-- Define the conditions in the problem
def right_angle (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x) * (p1.x - p3.x) + (p1.y - p2.y) * (p1.y - p3.y) + (p1.z - p2.z) * (p1.z - p3.z) = 0

-- The theorem we need to prove
theorem cube_side_length :
  right_angle A1 A0 A3 ∧ right_angle A2 A0 A1 ∧ right_angle A3 A0 A2 ∧
  ∃ s : ℝ, 
    (∃ B0 D0 E0 G0 : Point, B0.x = s ∧ B0 ∈ segment A0 A1 ∧ 
                            D0.y = s ∧ D0 ∈ segment A0 A2 ∧
                            E0.z = s ∧ E0 ∈ segment A0 A3 ∧
                            G0 = {x := s, y := s, z := s} ∧
                            on_plane G0 A1 A2 A3) →
    s = 180 / 71 := 
by { sorry }

-- Additional auxiliary definitions
def segment (p1 p2 : Point) : set Point :=
  { p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = t * p1.x + (1 - t) * p2.x ∧
                                 p.y = t * p1.y + (1 - t) * p2.y ∧
                                 p.z = t * p1.z + (1 - t) * p2.z }

def on_plane (p : Point) (p1 p2 p3 : Point) : Prop :=
  let d := 1 in -- normalizing d to 1
  (1 / 5) * p.x + (1 / 12) * p.y + (1 / 9) * p.z = d

end cube_side_length_l394_394602


namespace difference_in_perimeters_l394_394898

open Classical

noncomputable def side_length : ℝ := Real.sqrt 3

structure Octagon :=
  (CO OM MP PU UT TE : ℝ)
  (angle : ℕ → ℝ) -- Angles are either 90 or 270 degrees

def COMPUTER : Octagon := 
  { CO := side_length,
    OM := side_length,
    MP := side_length,
    PU := side_length,
    UT := side_length,
    TE := side_length,
    angle := λ i, if i % 2 = 0 then 90 else 270 }

theorem difference_in_perimeters (CO OM MP PU UT TE : ℝ)
  (h1 : CO = side_length)
  (h2 : OM = side_length)
  (h3 : MP = side_length)
  (h4 : PU = side_length)
  (h5 : UT = side_length)
  (h6 : TE = side_length)
  (equal_area : ∃ A B : ℝ, true) : 
  ∃ diff : ℝ, diff = 2 * side_length :=
sorry

end difference_in_perimeters_l394_394898


namespace remove_toothpicks_l394_394782

-- Definitions based on problem conditions
def toothpicks := 40
def triangles := 40
def initial_triangulation := True
def additional_condition := True

-- Statement to be proved
theorem remove_toothpicks :
  initial_triangulation ∧ additional_condition ∧ (triangles > 40) → ∃ (t: ℕ), t = 15 :=
by
  sorry

end remove_toothpicks_l394_394782


namespace twelve_div_one_fourth_eq_48_l394_394356

theorem twelve_div_one_fourth_eq_48 : 12 / (1 / 4) = 48 := by
  -- We know that dividing by a fraction is equivalent to multiplying by its reciprocal
  sorry

end twelve_div_one_fourth_eq_48_l394_394356


namespace interior_angle_of_regular_hexagon_l394_394228

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l394_394228


namespace number_of_teams_l394_394875

theorem number_of_teams (n : ℕ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → (games_played : ℕ) = 4) 
  (h2 : ∀ (i j : ℕ), i ≠ j → (count : ℕ) = 760) : 
  n = 20 := 
by 
  sorry

end number_of_teams_l394_394875


namespace find_x_l394_394463

-- Define the vectors a and b
def a (x : ℝ) : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (-1, 2)

-- Define the condition that a is perpendicular to b
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- State the theorem that we want to prove
theorem find_x (x : ℝ) :
  perpendicular (a x) b → x = 1 := 
begin
  sorry
end

end find_x_l394_394463


namespace interior_angle_regular_hexagon_l394_394245

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l394_394245


namespace multiplication_as_sum_of_squares_l394_394078

theorem multiplication_as_sum_of_squares :
  85 * 135 = 85^2 + 50^2 + 35^2 + 15^2 + 15^2 + 5^2 + 5^2 + 5^2 := by
  sorry

end multiplication_as_sum_of_squares_l394_394078


namespace scholars_number_l394_394998

theorem scholars_number (n : ℕ) : n < 600 ∧ n % 15 = 14 ∧ n % 19 = 13 → n = 509 :=
by
  intro h
  sorry

end scholars_number_l394_394998


namespace trapezoid_geometry_l394_394527

/-- In trapezoid ABCD with bases AD and BC, point F lies on the extension of AD. BF intersects diagonal AC at E, and side DC extends to G such that FG is parallel to BC.
Moreover, EF = 40 and GF = 30. Prove that BE = 30. --/
theorem trapezoid_geometry (A B C D F G E : Point)
    (h1 : trapezoid A B C D AD BC)
    (h2 : isExtension F AD)
    (h3 : intersect BF AC = E)
    (h4 : parallel FG BC)
    (h5 : EF = 40)
    (h6 : GF = 30) :
    BE = 30 :=
by
  sorry

end trapezoid_geometry_l394_394527


namespace common_chord_and_min_area_circle_l394_394872

theorem common_chord_and_min_area_circle :
  let C1 := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 3 * p.1 - 3 * p.2 + 3 = 0}
  let C2 := {p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 2 * p.1 - 2 * p.2 = 0}
  ∃ A B : ℝ × ℝ,
  (A ∈ C1 ∧ A ∈ C2) ∧ (B ∈ C1 ∧ B ∈ C2) ∧
  (∀ p : ℝ × ℝ, (p ∈ C1 ∧ p ∈ C2) → ((p.1 + p.2) = 3)) ∧
  (∀ circle : set (ℝ × ℝ), (∀ p : ℝ × ℝ, p ∈ circle → p ∈ C1 ∨ p ∈ C2) →
    ∃ center : ℝ × ℝ, ∃ radius : ℝ,
    circle = {q : ℝ × ℝ | (q.1 - center.1)^2 + (q.2 - center.2)^2 = radius^2} ∧
    (radius = sqrt (3 / 2)) → circle = C1)
:= sorry

end common_chord_and_min_area_circle_l394_394872


namespace smallest_n_is_589_l394_394170

noncomputable def smallest_n (a b c : ℕ) : ℕ :=
  let f : ℕ → ℕ := λ x, x / 5 + x / 25 + x / 125 + x / 625 + x / 3125 + x / 15625 + x / 78125
  in 2 * f(a) + f(2 * a) + f(c)

theorem smallest_n_is_589 (a b c : ℕ) (h1 : a + b + c = 2010) (h2 : b = 2 * a) :
  smallest_n a b (2010 - 3 * a) = 589 := sorry

end smallest_n_is_589_l394_394170


namespace simplify_trig_expression_l394_394976

open_locale real

theorem simplify_trig_expression (x : ℝ) (h : 1 + real.cos x ≠ 0) : 
  (real.sin x / (1 + real.cos x) + (1 + real.cos x) / real.sin x) = 2 * real.csc x :=
sorry

end simplify_trig_expression_l394_394976


namespace max_acute_angles_in_convex_polygon_l394_394288

def is_convex_polygon (P : Polygon) : Prop :=
  ∀ (θ : Angle), θ ∈ P.interior_angles → θ < 180

def is_acute_angle (θ : Angle) : Prop := θ < 90

noncomputable def sum_exterior_angles : ℝ := 360

noncomputable def relate_angles (θ φ : Angle) : Prop := θ + φ = 180

theorem max_acute_angles_in_convex_polygon (P : Polygon) :
  is_convex_polygon P →
  (∃ θs, θs ⊆ P.interior_angles ∧ θs.length = 3 ∧ θs.forall is_acute_angle ∧
  ∀ θs', θs' ⊆ P.interior_angles ∧ (∀ θ ∈ θs', is_acute_angle θ) ∧ θs'.length > 3 → false) :=
sorry

end max_acute_angles_in_convex_polygon_l394_394288


namespace area_of_triangle_DBC_is_12_l394_394522

-- Define points
def A := (0, 8)
def B := (0, 0)
def C := (12, 0)

-- Define segments
def AB := 8
def AD := (3 / 4) * AB
def BC := 12
def BE := (1 / 4) * BC
def D := (0, 8 - AD)
def E := (12 - BE, 0)

-- The base and height of triangle DBC
def base := BC
def height := 8 - AD

-- The area of triangle DBC
def area_DBC := (1 / 2) * base * height

theorem area_of_triangle_DBC_is_12 : area_DBC = 12 := by
  -- We will complete this proof following the conditions and calculations from above.
  sorry

end area_of_triangle_DBC_is_12_l394_394522


namespace solve_abs_eq_l394_394562

noncomputable def solve_eq (m : ℝ) : set ℝ :=
  if m ≥ 3 then
    { x | (x = (-m + sqrt (m^2 + 40)) / 4) ∨ (x = (-m - sqrt (m^2 + 40)) / 4) }
  else if 3 / 2 ≤ m ∧ m < 3 then
    { x | (x = (m + sqrt (m^2 + 40)) / 4) ∨ (x = 3 / m) }
  else if m ≤ -3 then
    { x | (x = (-m + sqrt (m^2 + 40)) / 4) ∨ (x = (-m - sqrt (m^2 + 40)) / 4) }
  else if -3 / 2 ≥ m ∧ m > -3 then
    { x | (x = (m - sqrt (m^2 + 40)) / 4) ∨ (x = 3 / m) }
  else
    ∅ 

theorem solve_abs_eq (m : ℝ) :
  ∃ x : ℝ, |x^2 - 1| + |x^2 - 4| = m * x ↔ x ∈ solve_eq m :=
by
  sorry

end solve_abs_eq_l394_394562


namespace cube_inscribed_sphere_volume_l394_394008

noncomputable def cubeSurfaceArea (a : ℝ) : ℝ := 6 * a^2
noncomputable def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3
noncomputable def inscribedSphereRadius (a : ℝ) : ℝ := a / 2

theorem cube_inscribed_sphere_volume :
  ∀ (a : ℝ), cubeSurfaceArea a = 24 → sphereVolume (inscribedSphereRadius a) = (4 / 3) * Real.pi := 
by 
  intros a h₁
  sorry

end cube_inscribed_sphere_volume_l394_394008


namespace parallelogram_diagonal_length_l394_394521

theorem parallelogram_diagonal_length (A B C D : ℂ) 
  (hA : A = complex.i) 
  (hB : B = 1) 
  (hC : C = 4 + 2 * complex.i) 
  (hParallelogram : A + C = B + D) :
  D = 4 + 3 * complex.i ∧ complex.abs (D - B) = 3 * real.sqrt 2 :=
by
  sorry

end parallelogram_diagonal_length_l394_394521


namespace find_f_prime_2_l394_394434

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f x / x

axiom tangent_coincide_at_points : ∀ f : ℝ → ℝ,
  (deriv f 0 = 1 / 2) ∧ (deriv (λ x, f x / x) 2 = 1 / 2)

theorem find_f_prime_2 :
  ∀ f : ℝ → ℝ, (f 2 = 2) ∧ ((∀ x, f 0 = 0)) ∧ tangent_coincide_at_points f → deriv f 2 = 2 :=
begin
  sorry
end

end find_f_prime_2_l394_394434


namespace value_of_leftover_coins_is_13_50_l394_394908

def quarters_james := 120
def dimes_james := 200
def half_dollars_james := 90

def quarters_lindsay := 150
def dimes_lindsay := 310
def half_dollars_lindsay := 160

def roll_quarters := 40
def roll_dimes := 50
def roll_half_dollars := 20

def total_quarters := quarters_james + quarters_lindsay
def total_dimes := dimes_james + dimes_lindsay
def total_half_dollars := half_dollars_james + half_dollars_lindsay

def leftover_quarters := total_quarters % roll_quarters
def leftover_dimes := total_dimes % roll_dimes
def leftover_half_dollars := total_half_dollars % roll_half_dollars

def value_leftover_quarters := leftover_quarters * 0.25
def value_leftover_dimes := leftover_dimes * 0.10
def value_leftover_half_dollars := leftover_half_dollars * 0.50

def total_value_leftover_coins := value_leftover_quarters + value_leftover_dimes + value_leftover_half_dollars

theorem value_of_leftover_coins_is_13_50 :
  total_value_leftover_coins = 13.50 :=
by
  sorry

end value_of_leftover_coins_is_13_50_l394_394908


namespace bisect_BC_l394_394081

section

variable {α : Type*} [EuclideanGeometry α]

-- Introducing points A, B, and C forming triangle ABC and H being the orthocenter
variables (A B C H : α)

-- Introducing circumcircle and A' as the antipodal point of A
variables (circumcircle : Circle α) (A' : α)

-- Given conditions
-- H is the orthocenter of triangle ABC
def isOrthocenter : Prop := Orthocenter A B C H

-- A' is the antipodal point of A in the circumcircle of triangle ABC
def isAntipodal : Prop := AntipodalPoint circumcircle A A'

-- Defining the main theorem to prove
theorem bisect_BC (h_orthocenter : isOrthocenter A B C H) (h_antipodal : isAntipodal circumcircle A A') : 
  SegmentBisector A' H B C := 
sorry

end

end bisect_BC_l394_394081


namespace interior_angle_regular_hexagon_l394_394240

theorem interior_angle_regular_hexagon : 
  let n := 6 in
  (n - 2) * 180 / n = 120 := 
by
  let n := 6
  sorry

end interior_angle_regular_hexagon_l394_394240


namespace positive_integers_in_S_l394_394549

-- Define the number 10^6 - 1
def num := 10^6 - 1

-- Problem statement
theorem positive_integers_in_S : ∃ S : Finset ℕ, S.card = 191 ∧ ∀ n ∈ S, n > 1 ∧ n ∣ num :=
by
  sorry

end positive_integers_in_S_l394_394549


namespace correct_neighbors_of_Nate_l394_394073

def Person := {Jack, Kelly, Lan, Mihai, Nate}

-- Assuming the seating is a list (cyclically considered) of persons around the table
def circular_seating (s : List Person) := true  -- Placeholder

def beside (a b : Person) (s : List Person) := 
  ∃ i, s[(i + 1) % s.length] = a ∧ s[i % s.length] = b ∨ s[(i + 1) % s.length] = b ∧ s[i % s.length] = a

def not_beside (a b : Person) (s : List Person) := 
  ¬ beside a b s

def is_correct_seating (s : List Person) :=
  beside Lan Mihai s ∧ not_beside Jack Kelly s

def neighbors (a : Person) (s : List Person) :=
  ∀ i, s[i % s.length] = a → 
       s[(i + 1) % s.length] = Jack ∧ s[(i - 1 + s.length) % s.length] = Kelly ∨
       s[(i + 1) % s.length] = Kelly ∧ s[(i - 1 + s.length) % s.length] = Jack

theorem correct_neighbors_of_Nate (s : List Person) (h : is_correct_seating s) :
  neighbors Nate s := 
  sorry

end correct_neighbors_of_Nate_l394_394073


namespace prime_factor_sum_1540_l394_394652

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p : ℕ, is_prime p ∧ p ∣ n then
    Classical.choose h
  else
    n

def largest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p : ℕ, is_prime p ∧ p ∣ n then
    Classical.choose (Exists.choose_spec h)
  else
    n

theorem prime_factor_sum_1540 :
  let spf := smallest_prime_factor 1540,
      lpf := largest_prime_factor 1540
  in 2 ≤ spf ∧ spf ≤ lpf ∧ lpf ≤ 1540 ∧ spf + lpf = 13 :=
by
  sorry

end prime_factor_sum_1540_l394_394652


namespace problem_equation_and_min_OM_l394_394063

noncomputable theory

-- Definitions per the given conditions
def ellipse_center : (Float × Float) := (0, 0)
def f1 : (Float × Float) := (0, -Float.sqrt 3)
def f2 : (Float × Float) := (0, Float.sqrt 3)
def eccentricity : Float := Float.sqrt 3 / 2
def curveC := { p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 ^ 2 + (p.2 ^ 2 / 4) = 1}
def tangent_eq (x₀ y₀ x : ℝ) := -4 * x₀ / y₀ * (x - x₀) + y₀
def intersection_A (x₀ y₀ : ℝ) := (1 / x₀, 0)
def intersection_B (x₀ y₀ : ℝ) := (0, 4 / y₀)
def O := (0, 0)
def M (A B : ℝ × ℝ) := (A.1 + B.1, A.2 + B.2)

-- Main problem in Lean 4 statement
theorem problem_equation_and_min_OM
    (M_on_traj : ∀ (x y : ℝ), (x > 1 ∧ y > 2) → (x=1 / x ∧ y=4 / y → ((1 / x ^ 2) + (4 / y ^ 2) = 1)))
    (min_OM : ∀ (x y : ℝ), (x > 1 ∧ y > 2) → ((x^2 - 1 + (4 / (x^2 - 1)) + 5 ≥ 9))) :
    ∃ traj_eq : (ℝ × ℝ ⟨contains the set constraint for the equation⟩ ),
    ∃ min_OM_val : ℝ, ∀ x y, ((x,y) ∈ traj_eq) ∧ min_OM_val = 3 := 
sorry

end problem_equation_and_min_OM_l394_394063


namespace simplify_trig_expression_l394_394978

open_locale real

theorem simplify_trig_expression (x : ℝ) (h : 1 + real.cos x ≠ 0) : 
  (real.sin x / (1 + real.cos x) + (1 + real.cos x) / real.sin x) = 2 * real.csc x :=
sorry

end simplify_trig_expression_l394_394978


namespace seated_people_count_l394_394131

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l394_394131


namespace infinitely_many_primes_l394_394117

theorem infinitely_many_primes : ∀ (p : ℕ) (h_prime : Nat.Prime p), ∃ (q : ℕ), Nat.Prime q ∧ q > p :=
by
  sorry

end infinitely_many_primes_l394_394117


namespace cone_volume_proof_l394_394507

noncomputable def volume_of_cone (r l : ℝ) : ℝ :=
  let h := real.sqrt (l^2 - r^2)
  in (1/3) * real.pi * r^2 * h

theorem cone_volume_proof :
  volume_of_cone 1 (2 * real.sqrt(7)) = real.sqrt(3) * real.pi :=
by
  sorry

end cone_volume_proof_l394_394507


namespace pyramid_ratio_l394_394070

theorem pyramid_ratio (A B C D P Q: ℝ^3) :
  (∥B - A∥ = 4) ∧ (∥D - A∥ = 2 * real.sqrt 2) ∧ (∥C - D∥ = 2) ∧
  (∥P - A∥ = 4) ∧ (A.1 = 0 ∧ A.2 = 0 ∧ A.3 = 0) ∧
  (B.2 = 0 ∧ B.3 = 0) ∧ (D.1 = 0 ∧ D.3 = 0) ∧ 
  (Q ∈ segment ℝ P B) ∧ 
  (vector.angle (Q - C) (plane_normal P A C) = real.arcsin (real.sqrt 3 / 3)) 
  → ∥Q - P∥ / ∥B - P∥ = 1 / 2 :=
sorry

end pyramid_ratio_l394_394070


namespace ticket_distribution_l394_394343

theorem ticket_distribution : 
  let people := {1, 2, 3, 4, 5}
  let tickets := 4
  (∀ person ∈ people, person receives at most one ticket) ∧ (all tickets are distributed) → 
  ∃! (S : set people), |S| = tickets := 
by
  sorry

end ticket_distribution_l394_394343


namespace total_unique_handshakes_l394_394631

def num_couples := 8
def num_individuals := num_couples * 2
def potential_handshakes_per_person := num_individuals - 1 - 1
def total_handshakes := num_individuals * potential_handshakes_per_person / 2

theorem total_unique_handshakes : total_handshakes = 112 := sorry

end total_unique_handshakes_l394_394631


namespace total_people_seated_l394_394155

-- Define the setting
def seated_around_round_table (n : ℕ) : Prop :=
  ∀ a b, 1 ≤ a ∧ a ≤ n ∧ 1 ≤ b ∧ b ≤ n

-- Define the card assignment condition
def assigned_card_numbers (n : ℕ) : Prop :=
  ∀ k, 1 ≤ k ∧ k ≤ n → k = (k % n) + 1

-- Define the condition of equal distances
def equal_distance_condition (n : ℕ) (p1 p2 p3 : ℕ) : Prop :=
  p1 = 31 ∧ p2 = 7 ∧ p3 = 14 ∧
  ((p1 - p2 + n) % n = (p1 - p3 + n) % n ∨
   (p2 - p1 + n) % n = (p3 - p1 + n) % n)

-- Statement of the theorem
theorem total_people_seated (n : ℕ) :
  seated_around_round_table n →
  assigned_card_numbers n →
  equal_distance_condition n 31 7 14 →
  n = 41 :=
by
  sorry

end total_people_seated_l394_394155


namespace total_investment_by_Q_is_correct_l394_394584

noncomputable def main : ℝ :=
  let P1_investment := 50000 : ℝ
  let P1_share := 3 : ℝ
  let Q1_share := 4 : ℝ
  let P2_investment := 30000 : ℝ
  let P2_share := 2 : ℝ
  let Q2_share := 3 : ℝ

  let Q1_investment := (P1_investment * Q1_share) / P1_share
  let Q2_investment := (P2_investment * Q2_share) / P2_share

  Q1_investment + Q2_investment

theorem total_investment_by_Q_is_correct :
  main = 111666.67 := 
sorry

end total_investment_by_Q_is_correct_l394_394584


namespace smallest_integer_with_odd_even_divisors_l394_394380

theorem smallest_integer_with_odd_even_divisors :
  ∃ (n : ℕ), (∃ m : ℕ, n = 2^m * 3^2 * 5) ∧
    (8 = ∏ p in (factors n).to_finset.filter (λ p, ¬ p.is_even), (factorization n p) + 1) ∧
    (16 = ∏ p in (factors n).to_finset.filter (λ p, p.is_even), (factorization n p) + 1) ∧
    (n = 360) :=
by
  sorry

end smallest_integer_with_odd_even_divisors_l394_394380


namespace polygon_sides_eq_four_l394_394874

theorem polygon_sides_eq_four (n : ℕ) 
  (h1 : ∀ n, sum_of_interior_angles n = (n - 2) * 180)
  (h2 : sum_of_exterior_angles = 360)
  (h3 : (n - 2) * 180 = 360) : n = 4 := 
sorry

end polygon_sides_eq_four_l394_394874


namespace problem_1_problem_2_l394_394675

-- Problem 1: Prove that the solution set of the inequality is as specified.
theorem problem_1 (x : ℝ) : (2 * x) / (x - 2) ≤ 1 ↔ -2 ≤ x ∧ x < 2 := 
sorry

-- Problem 2: Prove that the solution set of the inequality ax^2 + 2x + 1 > 0 for a > 0 is as specified.
theorem problem_2 (a x : ℝ) (h : 0 < a) : ax^2 + 2x + 1 > 0 ↔ 
  if a = 1 then x ≠ -1 else 
  if a > 1 then true else  
  (-∞ < x ∧ x < (-1 - sqrt (1 - a)) / a) ∨ ((-1 + sqrt (1 - a)) / a < x ∧ x < ∞) :=
sorry

end problem_1_problem_2_l394_394675


namespace ratio_volume_cylinder_cube_l394_394689

-- Define the volume of the cube
def volume_cube (s : ℝ) : ℝ :=
  s ^ 3

-- Define the radius of the inscribed cylinder
def radius_cylinder (s : ℝ) : ℝ :=
  s / 2

-- Define the height of the inscribed cylinder
def height_cylinder (s : ℝ) : ℝ :=
  s

-- Define the volume of the inscribed cylinder
def volume_cylinder (s : ℝ) : ℝ :=
  π * (radius_cylinder s)^2 * (height_cylinder s)

-- State the theorem about the ratio of volumes
theorem ratio_volume_cylinder_cube (s : ℝ) (h : s > 0) : 
  (volume_cylinder s) / (volume_cube s) = π / 4 := by
  sorry

end ratio_volume_cylinder_cube_l394_394689


namespace combinatorial_expression_identity_l394_394408

theorem combinatorial_expression_identity (n : ℕ) : 
  ∑ k in Finset.range (n + 1), (-2)^k * Nat.choose n k = (-1)^n := 
by 
  sorry

end combinatorial_expression_identity_l394_394408


namespace seated_people_count_l394_394136

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l394_394136


namespace find_complex_roots_l394_394766
noncomputable def roots_of_polynomial (z : ℂ) : Prop :=
  z^4 - 6 * z^2 + 8 = 0

theorem find_complex_roots :
  {z : ℂ | roots_of_polynomial z} = {(-2 : ℂ), -complex.sqrt 2, complex.sqrt 2, 2} :=
by sorry

end find_complex_roots_l394_394766


namespace minimum_value_proof_l394_394808

noncomputable def minValue (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : ℝ := 
  (x + 8 * y) / (x * y)

theorem minimum_value_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : 
  minValue x y hx hy h = 9 := 
by
  sorry

end minimum_value_proof_l394_394808


namespace collinear_intersections_of_parallelograms_l394_394383

noncomputable def parallelogram (A B C D : Type) [hilbert_space A B C D]

theorem collinear_intersections_of_parallelograms
    (A B C D K L : Type) 
    [parallelogram A B C D] 
    [parallelogram A K L D]
    (H1 : collinear_points (set.univ A B C D) (set.univ A K L D)) -- BC // KL
    (H2 : ¬parallel_lines A C K D) : -- AC not parallel to KD
    collinear_points ((AK_intersect_DC A K C D) , (AB_intersect_DL A B D L), (AC_intersect_KD A C K D)) :=
sorry

end collinear_intersections_of_parallelograms_l394_394383


namespace degree_measure_of_regular_hexagon_interior_angle_l394_394280

theorem degree_measure_of_regular_hexagon_interior_angle : 
  ∀ (n : ℕ), n = 6 → ∀ (interior_angle : ℕ), interior_angle = (n - 2) * 180 / n → interior_angle = 120 :=
by
  sorry

end degree_measure_of_regular_hexagon_interior_angle_l394_394280


namespace influenza_treatment_sum_l394_394352

-- Definitions for the sequence and conditions
def a : ℕ → ℕ 
| 0     := 0       -- This is a_0, which is not used but required for initial index.
| 1     := 1       -- a_1 = 1
| 2     := 2       -- a_2 = 2
| (n+3) := a (n + 1) + 1 + (-1)^(n + 1)  -- Recurrence relation shifted by index for Lean's zero-indexed natural numbers

-- Statement to prove that the sum of the first 30 terms of the sequence is 255
theorem influenza_treatment_sum : (∑ i in Finset.range 30, a (i + 1)) = 255 := by
  sorry

end influenza_treatment_sum_l394_394352


namespace sum_of_coefficients_l394_394773

theorem sum_of_coefficients : 
  let P := (λ x : ℝ, (4 * x^2 - 4 * x + 3)^4 * (4 + 3 * x - 3 * x^2)^2) in 
  P 1 = 1296 := 
by 
  let P := (λ x : ℝ, (4 * x^2 - 4 * x + 3)^4 * (4 + 3 * x - 3 * x^2)^2)
  exact calc
    P 1 = ((4 * 1^2 - 4 * 1 + 3)^4) * ((4 + 3 * 1 - 3 * 1^2)^2) : by rw [P]
      ... = (3^4) * (4^2) : by rw [calc_1]
      ... = 81 * 16 : by rw [calc_2]
      ... = 1296 : by rw [calc_3]

end sum_of_coefficients_l394_394773


namespace bryan_bought_5_tshirts_l394_394732

theorem bryan_bought_5_tshirts (total_cost : ℤ) (tshirt_cost : ℤ) (pants_cost : ℤ) (pants_qty : ℤ) (tshirt_qty : ℤ) :
  total_cost = 1500 →
  tshirt_cost = 100 →
  pants_cost = 250 →
  pants_qty = 4 →
  (total_cost = (tshirt_qty * tshirt_cost + pants_qty * pants_cost)) →
  tshirt_qty = 5 :=
begin
  assume h1 h2 h3 h4 h5,
  sorry
end

end bryan_bought_5_tshirts_l394_394732


namespace mode_scores_is_90_l394_394062

theorem mode_scores_is_90 :
  let scores := [90, 85, 90, 80, 95] in
  (∀ x ∈ scores, x = 90 → x ∈ [90]) → 
  90 ∈ [90] ∧
  (∀ y ≠ 90, ¬ (y = mode scores)) :=
by
  let scores := [90, 85, 90, 80, 95] in
  sorry

end mode_scores_is_90_l394_394062


namespace proportion_first_number_l394_394871

theorem proportion_first_number (x : ℝ) (h : x / 10 = 8 / 0.6) : x = 133.333... :=
by
  sorry

end proportion_first_number_l394_394871


namespace seated_people_count_l394_394133

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l394_394133


namespace rhombus_area_is_correct_l394_394644

noncomputable def rhombus_area (side length angle: ℝ) : ℝ :=
  (side length ^ 2) * (Real.sin angle)

theorem rhombus_area_is_correct :
  rhombus_area 4 (Real.pi / 4) = 8 * Real.sqrt 2 := by
  sorry

end rhombus_area_is_correct_l394_394644


namespace price_Ramesh_paid_l394_394593

-- Define the conditions
def labelled_price_sold (P : ℝ) := 1.10 * P
def discount_price_paid (P : ℝ) := 0.80 * P
def additional_costs := 125 + 250
def total_cost (P : ℝ) := discount_price_paid P + additional_costs

-- The main theorem stating that given the conditions,
-- the price Ramesh paid for the refrigerator is Rs. 13175.
theorem price_Ramesh_paid (P : ℝ) (H : labelled_price_sold P = 17600) :
  total_cost P = 13175 :=
by
  -- Providing a placeholder, as we do not need to provide the proof steps in the problem formulation
  sorry

end price_Ramesh_paid_l394_394593


namespace regular_hexagon_interior_angle_l394_394270

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l394_394270


namespace p_necessary_but_not_sufficient_for_q_l394_394113

variables {x : ℝ}

def p := x^2 > 4
def q := x > 2

theorem p_necessary_but_not_sufficient_for_q : (q → p) ∧ ¬(p → q) :=
by
  sorry

end p_necessary_but_not_sufficient_for_q_l394_394113


namespace number_of_people_seated_l394_394144

theorem number_of_people_seated (n : ℕ) :
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → 1 ≤ ((i + k) % n) ∧ ((i + k) % n) ≤ n) →
  (1 ≤ 31 ∧ 31 ≤ n) ∧ 
  ((31 + 7) % n = ((31 + 14) % n) →
  n = 41 :=
sorry

end number_of_people_seated_l394_394144


namespace heat_dissipated_in_resistor_l394_394725

theorem heat_dissipated_in_resistor
  (R C r : ℝ)
  (ε : ℝ): 
  C > 0 → ε > 0 → r > 0 → R > 0 →
  ∃ (Q_R : ℝ), Q_R = C * ε^2 * R / (2 * (R + r)) :=
by
  intro hC hε hr hR
  use C * ε^2 * R / (2 * (R + r))
  sorry

end heat_dissipated_in_resistor_l394_394725


namespace find_AB_find_angle_C_l394_394815

-- Define the type of triangle and its properties
universe u
variables {α : Type u} [linear_ordered_field α]

structure triangle (α : Type u) :=
  (A B C : α) -- vertices
  (a b c : α) -- side lengths opposite to vertices A, B, and C respectively
  (sin_A sin_B sin_C : α) -- sin values of angles A, B, and C respectively
  (perimeter : α)
  (area : α)

-- Hypotheses based on the problem conditions
def triangle_conditions (t : triangle α) : Prop :=
  t.perimeter = sqrt 2 + 1 ∧
  t.sin_A + t.sin_B = sqrt 2 * t.sin_C ∧
  t.area = 1 / 6 * t.sin_C

-- Problem 1: Prove that AB = 1
theorem find_AB (t : triangle α) (h : triangle_conditions t) : 
  t.a = 1 :=
sorry

-- Problem 2: Prove that angle C = 60 degrees
theorem find_angle_C (t : triangle α) (h : triangle_conditions t) : 
  t.sin_C = 1 / 2 ∧ t.C = 60 :=
sorry

end find_AB_find_angle_C_l394_394815


namespace regular_hexagon_interior_angle_l394_394266

-- Define what it means to be a regular hexagon
def regular_hexagon (sides : ℕ) : Prop :=
  sides = 6

-- Define the degree measure of an interior angle of a regular hexagon
def interior_angle (sides : ℕ) : ℝ :=
  ((sides - 2) * 180) / sides

-- The theorem we want to prove
theorem regular_hexagon_interior_angle (sides : ℕ) (h : regular_hexagon sides) : 
  interior_angle sides = 120 := 
by
  rw [regular_hexagon, interior_angle] at h
  simp at h
  rw h
  sorry

end regular_hexagon_interior_angle_l394_394266


namespace table_seating_problem_l394_394143

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l394_394143


namespace construct_trapezoid_l394_394646

-- Definitions of the given lengths of bases and diagonals
variables (a b c d : ℝ)

-- Define points and lines involved in the construction
variables (A B C D K : Type)

-- Define the existence of the bases and diagonals
axioms (AD BC AC BD : ℝ)
axiom h_ad : AD = a
axiom h_bc : BC = b
axiom h_ac : AC = c
axiom h_bd : BD = d

-- Define the relationships and coordinates of points based on the conditions
noncomputable def exists_trapezoid (a b c d : ℝ) : Prop :=
∃ (A B C D K : Type),
  -- the construction criteria in terms of distances and parallelism
  (AD = a) ∧ (BC = b) ∧ (AC = c) ∧ (BD = d) ∧
  -- Additionally, parallelism and point relationships
  (∃ (l : Type), parallel (line (point C) (point K)) (line (point A) (point K))) ∧
  (point D lies_on line (point K) (point A)) ∧
  sorry -- Further relationships not fully formalized here

-- Statement of the theorem
theorem construct_trapezoid : exists_trapezoid a b c d :=
sorry

end construct_trapezoid_l394_394646


namespace triangle_perimeter_correct_l394_394814

noncomputable def triangle_perimeter (a b x : ℕ) : ℕ := a + b + x

theorem triangle_perimeter_correct :
  ∀ (x : ℕ), (2 + 4 + x = 10) → 2 < x → x < 6 → (∀ k : ℕ, k = x → k % 2 = 0) → triangle_perimeter 2 4 x = 10 :=
by
  intros x h1 h2 h3
  rw [triangle_perimeter, h1]
  sorry

end triangle_perimeter_correct_l394_394814


namespace exists_v_satisfying_equation_l394_394776

noncomputable def custom_operation (v : ℝ) : ℝ :=
  v - (v / 3) + Real.sin v

theorem exists_v_satisfying_equation :
  ∃ v : ℝ, custom_operation (custom_operation v) = 24 := 
sorry

end exists_v_satisfying_equation_l394_394776


namespace largest_d_bound_l394_394397

theorem largest_d_bound (y : Fin 51 → ℝ) (h_sum_zero : ∑ i, y i = 0)
    (N := y 25) :
    (∑ i, (y i)^2) ≥ (1276 / 25) * N^2 := 
begin
  sorry
end

end largest_d_bound_l394_394397


namespace ratio_of_a_to_b_l394_394196

theorem ratio_of_a_to_b 
  (b c a : ℝ)
  (h1 : b / c = 1 / 5) 
  (h2 : a / c = 1 / 7.5) : 
  a / b = 2 / 3 :=
by
  sorry

end ratio_of_a_to_b_l394_394196


namespace prove_math_problem_l394_394163

noncomputable def function_is_non_negative (x : ℝ) : Prop := 
  ∀ y : ℝ, y^2 = (x^2 * (x + 1)) / (x - 1) → y^2 ≥ 0

noncomputable def function_is_symmetric := 
  ∀ (x : ℝ), (y : ℝ), (y^2 = (x^2 * (x + 1)) / (x - 1)) -> (y = -sqrt((x^2 * (x + 1)) / (x - 1)) ∨ y = sqrt((x^2 * (x + 1)) / (x - 1)))

noncomputable def behavior_near_vasymptote := 
  ∀ ε > 0, ∀ δ > 0, ∀ x : ℝ, (1 < x ∧ x < 1 + δ) → ((x * x * (x + 1)) / (x - 1)) > 1 / ε

noncomputable def behavior_large_x :=
  ∀ x : ℝ, (x > 1 → (abs (((x^2 * (x + 1)) / (x - 1)) - x^2) < ε))

noncomputable def stationary_points :=
  ∀ x : ℝ, (d dx ((x^2 * (x + 1)) / (x - 1)) = 0)

theorem prove_math_problem :
  ∀ x : ℝ, x ≠ 1 → function_is_non_negative x → function_is_symmetric → behavior_near_vasymptote → behavior_large_x → stationary_points :=
by
  sorry

end prove_math_problem_l394_394163


namespace highest_qualification_number_l394_394608

theorem highest_qualification_number (participants : ℕ) (qual_num : ℕ → ℕ) :
  participants = 256 →
  (∀ i j, i ≠ j → qual_num i ≠ qual_num j) →
  (∀ i j, |qual_num i - qual_num j| > 2 → qual_num i < qual_num j) →
  (∃ w, w ≤ participants ∧ w = 16) :=
begin
  intros,
  sorry
end

end highest_qualification_number_l394_394608


namespace find_integer_n_l394_394394

theorem find_integer_n : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ -3125 [MOD 16] ∧ n = 11 :=
by
  use 11
  split
  norm_num
  split
  norm_num
  split
  norm_num
  sorry

end find_integer_n_l394_394394


namespace angle_P_is_96_degrees_l394_394029

-- Prove the measure of angle P in the pentagon is 96 degrees
theorem angle_P_is_96_degrees (a b c d : ℝ) (h1 : a = 128) (h2 : b = 92) (h3 : c = 113) (h4 : d = 111) :
  ∃ P : ℝ, P = 96 :=
by
  -- Given conditions
  let θ1 := a
  let θ2 := b
  let θ3 := c
  let θ4 := d
  
  have θ_sum : θ1 + θ2 + θ3 + θ4 = 128 + 92 + 113 + 111 := by
    rw [h1, h2, h3, h4]
  
  have pentagon_total : θ1 + θ2 + θ3 + θ4 + ?m_1 = 540 := by
    rw [θ_sum]
    linarith
  
  use ?m_1
  sorry

end angle_P_is_96_degrees_l394_394029


namespace exists_polynomials_eq_Q_l394_394021

def Q (x1 x2 x3 x4 x5 x6 x7 : ℝ) : ℝ :=
  (x1 + x2 + x3 + x4 + x5 + x6 + x7)^2 + 2 * (x1^2 + x2^2 + x3^2 + x4^2 + x5^2 + x6^2 + x7^2)

theorem exists_polynomials_eq_Q :
  ∃ (P1 P2 P3 P4 P5 P6 P7 : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ),
    (∀ x1 x2 x3 x4 x5 x6 x7, (0 <= P1 x1 x2 x3 x4 x5 x6 x7) ∧ (0 <= P2 x1 x2 x3 x4 x5 x6 x7) ∧
                                   (0 <= P3 x1 x2 x3 x4 x5 x6 x7) ∧ (0 <= P4 x1 x2 x3 x4 x5 x6 x7) ∧
                                   (0 <= P5 x1 x2 x3 x4 x5 x6 x7) ∧ (0 <= P6 x1 x2 x3 x4 x5 x6 x7) ∧
                                   (0 <= P7 x1 x2 x3 x4 x5 x6 x7)) ∧
    (∀ x1 x2 x3 x4 x5 x6 x7, 
       Q x1 x2 x3 x4 x5 x6 x7 = 
        (P1 x1 x2 x3 x4 x5 x6 x7)^2 + 
        (P2 x1 x2 x3 x4 x5 x6 x7)^2 + 
        (P3 x1 x2 x3 x4 x5 x6 x7)^2 + 
        (P4 x1 x2 x3 x4 x5 x6 x7)^2 + 
        (P5 x1 x2 x3 x4 x5 x6 x7)^2 + 
        (P6 x1 x2 x3 x4 x5 x6 x7)^2 + 
        (P7 x1 x2 x3 x4 x5 x6 x7)^2) :=
by sorry

end exists_polynomials_eq_Q_l394_394021


namespace min_expr_value_l394_394746

noncomputable def expr_min_value : ℝ := 
  (@Real.sin (Real.div Real.pi 4) + (1 / @Real.sin (Real.div Real.pi 4)))^3 +
  (@Real.cos (Real.div Real.pi 4) + (1 / @Real.cos (Real.div Real.pi 4)))^3

theorem min_expr_value : expr_min_value = (729 * Real.sqrt 2) / 16 :=
by
  sorry

end min_expr_value_l394_394746


namespace average_apples_per_guest_l394_394076

theorem average_apples_per_guest
  (servings_per_pie : ℕ)
  (pies : ℕ)
  (apples_per_serving : ℚ)
  (total_guests : ℕ)
  (red_delicious_proportion : ℚ)
  (granny_smith_proportion : ℚ)
  (total_servings := pies * servings_per_pie)
  (total_apples := total_servings * apples_per_serving)
  (total_red_delicious := (red_delicious_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (total_granny_smith := (granny_smith_proportion / (red_delicious_proportion + granny_smith_proportion)) * total_apples)
  (average_apples_per_guest := total_apples / total_guests) :
  servings_per_pie = 8 →
  pies = 3 →
  apples_per_serving = 1.5 →
  total_guests = 12 →
  red_delicious_proportion = 2 →
  granny_smith_proportion = 1 →
  average_apples_per_guest = 3 :=
by
  intros;
  sorry

end average_apples_per_guest_l394_394076


namespace minimum_value_f_l394_394792

theorem minimum_value_f (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) : (∃ x y : ℝ, (x + y + x * y) ≥ - (9 / 8) ∧ 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :=
sorry

end minimum_value_f_l394_394792


namespace apples_per_pie_correct_l394_394613

def total_apples : Nat := 51
def handed_out_apples : Nat := 41
def remaining_apples : Nat := total_apples - handed_out_apples
def number_of_pies : Nat := 2
def apples_per_pie (remaining_apples : Nat) (number_of_pies : Nat) := remaining_apples / number_of_pies

theorem apples_per_pie_correct : 
  apples_per_pie remaining_apples number_of_pies = 5 :=
by
  have rem_apples_calc : remaining_apples = 51 - 41 := rfl
  have num_pies_calc : number_of_pies = 2 := rfl
  rw [rem_apples_calc, num_pies_calc]
  exact Nat.div_eq_of_eq_mul (by norm_num)

end apples_per_pie_correct_l394_394613


namespace problem_l394_394932

theorem problem
  (x y: ℝ)
  (hx: 1 < x)
  (hy: 1 < y)
  (h: (Real.log x / Real.log 2)^5 + (Real.log y / Real.log 3)^5 + 32 = 16 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)):
  x^2 + y^2 = (2: ℝ)^(2 * Real.root 16 5) + (3: ℝ)^(2 * Real.root 16 5) := by
  sorry

end problem_l394_394932


namespace probability_of_drawing_red_or_green_l394_394291

def red_marbles : ℕ := 4
def green_marbles : ℕ := 3
def yellow_marbles : ℕ := 6

def total_marbles : ℕ := red_marbles + green_marbles + yellow_marbles
def favorable_marbles : ℕ := red_marbles + green_marbles
def probability_of_red_or_green : ℚ := favorable_marbles / total_marbles

theorem probability_of_drawing_red_or_green :
  probability_of_red_or_green = 7 / 13 := by
  sorry

end probability_of_drawing_red_or_green_l394_394291


namespace percent_absent_l394_394597

-- Given conditions
def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def absent_boys_fraction : ℚ := 1 / 8
def absent_girls_fraction : ℚ := 1 / 4

-- Theorem to prove
theorem percent_absent : 100 * ((absent_boys_fraction * boys + absent_girls_fraction * girls) / total_students) = 17.5 := 
sorry

end percent_absent_l394_394597


namespace triangle_inequality_l394_394744

theorem triangle_inequality (a b c : ℝ) (h : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
sorry

end triangle_inequality_l394_394744


namespace system_of_equations_solution_l394_394166

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 : ℝ), 
    (x1 + 2 * x2 = 10) ∧
    (3 * x1 + 2 * x2 + x3 = 23) ∧
    (x2 + 2 * x3 = 13) ∧
    (x1 = 4) ∧
    (x2 = 3) ∧
    (x3 = 5) :=
sorry

end system_of_equations_solution_l394_394166


namespace transform_table_to_non_negative_sums_l394_394889

theorem transform_table_to_non_negative_sums (m n : ℕ) (table : matrix (fin m) (fin n) ℤ) :
  ∃ (table' : matrix (fin m) (fin n) ℤ),
    (∀ i : fin m, 0 ≤ ∑ j : fin n, table' i j) ∧ (∀ j : fin n, 0 ≤ ∑ i : fin m, table' i j) :=
sorry

end transform_table_to_non_negative_sums_l394_394889


namespace seated_people_count_l394_394132

theorem seated_people_count (n : ℕ) :
  (∀ (i : ℕ), i > 0 → i ≤ n) ∧
  (∀ (k : ℕ), k > 0 → k ≤ n → ∃ (p q : ℕ), 
         p = 31 ∧ q = 7 ∧ (p < n) ∧ (q < n) ∧
         p + 16 + 1 = q ∨ 
         p = 31 ∧ q = 14 ∧ (p < n) ∧ (q < n) ∧ 
         p - (n - q) + 1 = 16) → 
  n = 41 := 
by 
  sorry

end seated_people_count_l394_394132


namespace data_plan_comparison_l394_394341

theorem data_plan_comparison : ∃ (m : ℕ), 500 < m :=
by
  let cost_plan_x (m : ℕ) : ℕ := 15 * m
  let cost_plan_y (m : ℕ) : ℕ := 2500 + 10 * m
  use 501
  have h : 500 < 501 := by norm_num
  exact h

end data_plan_comparison_l394_394341


namespace axes_of_symmetry_intersect_inside_l394_394937

theorem axes_of_symmetry_intersect_inside (P : Polygon) (h : has_two_axes_of_symmetry P) : 
  ∃ (A B : Point), is_axis_of_symmetry P A ∧ is_axis_of_symmetry P B ∧ A ≠ B ∧ intersects_inside P A B :=
sorry

end axes_of_symmetry_intersect_inside_l394_394937


namespace number_of_ways_l394_394221

-- Definition of the problem conditions
def valid_numbers (A B C D : ℕ) (digits : Finset ℕ) : Prop :=
  digits = {1, 3, 4, 5, 6, 7, 8, 9} ∧
  (A / 100 < 10 ∧ A / 100 > 0) ∧ -- A is a three-digit number
  (B / 10 < 10 ∧ B / 10 > 0) ∧  -- B is a two-digit number
  (C / 10 < 10 ∧ C / 10 > 0) ∧  -- C is a two-digit number
  (D < 10) ∧                    -- D is a one-digit number
  (B < C) ∧                     -- B < C
  (A + D = 143) ∧
  (B + C = 143) ∧
  (digits.ssubset {A/100, (A mod 100)/10, A mod 10, 
                   B/10, B mod 10, 
                   C/10, C mod 10, 
                   D} = ∅)   -- Ensure all digits are used exactly once

-- The theorem statement ensuring that the correct answer is deduced
theorem number_of_ways : 
  ∃ (A B C D : ℕ) (digits : Finset ℕ), valid_numbers A B C D digits → 24 :=
sorry

end number_of_ways_l394_394221


namespace find_f_neg_one_l394_394003

-- Definition: f(x) is an odd function and for x >= 0, f(x) = x * (1 + x)
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def f (x : ℝ) : ℝ := if x ≥ 0 then x * (1 + x) else -(x * (1 + -x))

theorem find_f_neg_one (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_def : ∀ x, x ≥ 0 → f x = x * (1 + x)) : f (-1) = -2 :=
by
  sorry

end find_f_neg_one_l394_394003


namespace correct_option_is_D_l394_394701

theorem correct_option_is_D
  (students_total : ℕ) (students_selected : ℕ)
  (h1 : students_total = 310)
  (h2 : students_selected = 31) :
  (D : string) = "The sample size is 31" :=
by
  -- Insert proof here
  sorry

end correct_option_is_D_l394_394701


namespace regular_hexagon_interior_angle_l394_394262

-- Definitions for the conditions
def is_regular_hexagon (sides : ℕ) (angles : list ℝ) : Prop :=
  sides = 6 ∧ angles.length = 6 ∧ ∀ angle ∈ angles, angle = (720.0 / 6)

-- The theorem statement
theorem regular_hexagon_interior_angle :
  ∀ (sides : ℕ) (angles : list ℝ), is_regular_hexagon(sides)(angles) → (angles.head = 120.0) :=
by
  -- skip the proof
  sorry

end regular_hexagon_interior_angle_l394_394262


namespace cosine_identity_l394_394466

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) 

theorem cosine_identity 
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (hc : ∥c∥ = real.sqrt 2)
  (habc : a + b + c = 0) :
  real.cos (inner (a - c) (b - c)) = 4 / 5 :=
sorry

end cosine_identity_l394_394466


namespace trash_can_purchase_count_ways_to_purchase_trash_cans_l394_394315

theorem trash_can_purchase (k : ℕ) :
  k + (8 - k) = 8 ∧ 150 * k + 225 * (8 - k) ≤ 1500 → 4 ≤ k ∧ k ≤ 8 :=
by
  sorry

theorem count_ways_to_purchase_trash_cans :
  ∃ n : ℕ, n = 5 ∧ ∀ k : ℕ, (k + (8 - k) = 8 ∧ 150 * k + 225 * (8 - k) ≤ 1500 → 4 ≤ k ∧ k ≤ 8) → (4 ≤ k ∧ k ≤ 8) :=
by
  use 5
  intro k h
  have : 4 ≤ k ∧ k ≤ 8 := trash_can_purchase k h
  exact this

end trash_can_purchase_count_ways_to_purchase_trash_cans_l394_394315


namespace arithmetic_sequence_sum_of_bn_l394_394624

theorem arithmetic_sequence (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = (2^(n+1) * a n) / (a n + 2^n)) :
  ∃ d : ℝ, ∀ n : ℕ, n > 0 → (2^(n+1) / a (n + 1)) - (2^n / a n) = d :=
by
  sorry

theorem sum_of_bn (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = (2^(n+1) * a n) / (a n + 2^n))
  (h3 : ∀ n : ℕ, b n = 2^(n+1) / a n + 3)
  (Sn_def : ∀ n : ℕ, S n = ∑ i in finset.range n, b (i + 1)) : 
  ∀ n : ℕ, S n = n^2 + 6*n :=
by
  sorry

end arithmetic_sequence_sum_of_bn_l394_394624


namespace exists_sequence_with_unique_differences_l394_394906

-- Define the main existence theorem
theorem exists_sequence_with_unique_differences (k : ℕ) :
  ∃ S : set ℕ, 
    (∀ x y ∈ S, x ≠ y → |x - y| ∉ S) ∧ 
    (∀ n ∈ set.Icc 1 k, ∃ x y ∈ S, n = |x - y|) ∧ 
    (¬ ∃ x y ∈ S, k + 1 = |x - y|) := 
sorry

end exists_sequence_with_unique_differences_l394_394906


namespace find_solutions_l394_394765

theorem find_solutions (x : ℝ) :
  (real.cbrt (18 * x - 2) + real.cbrt (16 * x + 2) + real.cbrt (-72 * x) = 
  6 * real.cbrt x) ↔ (x = 0 ∨ x = 1/9 ∨ x = -1/8) :=
sorry

end find_solutions_l394_394765
