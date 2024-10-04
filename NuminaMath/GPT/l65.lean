import Mathlib

namespace surface_area_of_sphere_l65_65099

noncomputable theory

-- Define a tetrahedron with given properties and prove the surface area of the sphere.
theorem surface_area_of_sphere (A B C D O : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]
  (AB AC : ℝ) (BC : ℝ) (AD_perpendicular : ∀ (x : ℝ), x = 0)
  (G_centroid : ℝ) (tangent_DG_ABC : ℝ)
  (on_sphere : ∀ (x : Type), x = O) (sphere_radius : ℝ)
  : AB = 5 ∧ AC = 5 ∧ BC = 8 ∧ AD_perpendicular 1 ∧ G_centroid = 2 ∧ tangent_DG_ABC = 1/2 →
    (4 * π * sphere_radius^2 = 634 * π / 9)
sorry

end surface_area_of_sphere_l65_65099


namespace solution_one_solution_two_solution_three_l65_65581

open Real

noncomputable def problem_one (a b : ℝ) (cosA : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 then 1 else 0

theorem solution_one (a b : ℝ) (cosA : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → problem_one a b cosA = 1 := by
  intros ha hb hcos
  unfold problem_one
  simp [ha, hb, hcos]

noncomputable def problem_two (a b : ℝ) (cosA sinB : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 ∧ sinB = sqrt 10 / 4 then sqrt 10 / 4 else 0

theorem solution_two (a b : ℝ) (cosA sinB : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → sinB = sqrt 10 / 4 → problem_two a b cosA sinB = sqrt 10 / 4 := by
  intros ha hb hcos hsinB
  unfold problem_two
  simp [ha, hb, hcos, hsinB]

noncomputable def problem_three (a b : ℝ) (cosA sinB sin2AminusB : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 ∧ sinB = sqrt 10 / 4 ∧ sin2AminusB = sqrt 10 / 8 then sqrt 10 / 8 else 0

theorem solution_three (a b : ℝ) (cosA sinB sin2AminusB : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → sinB = sqrt 10 / 4 → sin2AminusB = sqrt 10 / 8 → problem_three a b cosA sinB sin2AminusB = sqrt 10 / 8 := by
  intros ha hb hcos hsinB hsin2AminusB
  unfold problem_three
  simp [ha, hb, hcos, hsinB, hsin2AminusB]

end solution_one_solution_two_solution_three_l65_65581


namespace total_share_proof_l65_65266

variable (P R : ℕ) -- Parker's share and Richie's share
variable (total_share : ℕ) -- Total share

-- Define conditions
def ratio_condition : Prop := (P : ℕ) / 2 = (R : ℕ) / 3
def parker_share_condition : Prop := P = 50

-- Prove the total share is 125
theorem total_share_proof 
  (h1 : ratio_condition P R)
  (h2 : parker_share_condition P) : 
  total_share = 125 :=
sorry

end total_share_proof_l65_65266


namespace postage_cost_correct_l65_65318

-- Conditions
def base_rate : ℕ := 35
def additional_rate_per_ounce : ℕ := 25
def weight_in_ounces : ℚ := 5.25
def first_ounce : ℚ := 1
def fraction_weight : ℚ := weight_in_ounces - first_ounce
def num_additional_charges : ℕ := Nat.ceil (fraction_weight)

-- Question and correct answer
def total_postage_cost : ℕ := base_rate + (num_additional_charges * additional_rate_per_ounce)
def answer_in_cents : ℕ := 160

theorem postage_cost_correct : total_postage_cost = answer_in_cents := by sorry

end postage_cost_correct_l65_65318


namespace number_of_correct_propositions_l65_65750

theorem number_of_correct_propositions : 
  let P1 := ¬(∀ l1 l2 : Line, l1 ∥ l2 → ∀ π : Plane, proj(π, l1) ∥ proj(π, l2)) ∧
             ∀ α β : Plane, (α ∥ β → ∀ m : Line, m ⊂ α → m ∥ β) ∧
             ¬(∀ α β : Plane, ∀ m : Line, (m = α ∩ β → ∀ n : Line, n ⊂ α → n ⊥ m → n ⊥ β)) ∧
             ∀ P : Point, ∀ A B C : Point, (equidistant(P, A, B, C) → is_circumcenter(proj(plane_of_triangle(A, B, C), P), A, B, C))
  in count_true [P1.1, P1.2, P1.3, P1.4] = 2 :=
by
  sorry

end number_of_correct_propositions_l65_65750


namespace sum_of_solutions_l65_65033

def f (x : ℝ) : ℝ :=
  if x < -3 then 3 * x + 9 else -x^2 - 2 * x + 2

theorem sum_of_solutions :
  (∃ x < -3, f x = 3) → False ∧ 
  (∀ x ≥ -3, (f x = 3 → (x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2))) → 
  (-1 + Real.sqrt 2) + (-1 - Real.sqrt 2) = -2 :=
by
  sorry

end sum_of_solutions_l65_65033


namespace incorrect_x3_x4_l65_65120

noncomputable def f : ℝ → ℝ :=
  λ x, if (0 < x ∧ x ≤ 10) then abs (Real.log x) else 
       if (10 < x ∧ x < 20) then abs (Real.log (20 - x)) else 0

theorem incorrect_x3_x4 (t : ℝ) (x1 x2 x3 x4 : ℝ)
  (h_f_x1 : f x1 = t) (h_f_x2 : f x2 = t) (h_f_x3 : f x3 = t) (h_f_x4 : f x4 = t)
  (h_order : x1 < x2 ∧ x2 < x3 ∧ x3 < x4) :
  x3 * x4 ≠ 361 := 
sorry

end incorrect_x3_x4_l65_65120


namespace angle_ADB_correct_l65_65183

noncomputable def measure_angle_ADB : real :=
  135
  
theorem angle_ADB_correct {A B C D : Type} [right_triangle A B C]
  (hA : angle A = 45) (hB : angle B = 45)
  (hD : D = angle_bisector_intersection A B) : 
  measure_angle (angle A D B) = 135 :=
sorry

end angle_ADB_correct_l65_65183


namespace find_a2_l65_65237

variable {α : Type*} [linear_ordered_field α]

def arithmetic_sequence (a : α) (d : α) (n : ℕ) : α :=
  a + d * (n - 1)

def sum_first_n_terms (a : α) (d : α) (n : ℕ) : α :=
  n * a + (n * (n - 1)) / 2 * d

theorem find_a2 (a d : ℝ) (h : sum_first_n_terms a d 4 = arithmetic_sequence a d 4 + 3) : 
  arithmetic_sequence a d 2 = 1 :=
by
  sorry

end find_a2_l65_65237


namespace white_checker_cannot_capture_both_black_l65_65622

-- Definitions relevant to the problem:
-- Infinite chessboard, positions encoded as (x, y) tuples.
structure Position where
  x : Int
  y : Int

-- Two initial black checkers placed on adjacent black squares diagonally.
def black_checker_1 : Position := ⟨0, 0⟩
def black_checker_2 : Position := ⟨1, 1⟩

-- A function to determine the new position of the white checker after a capture move.
def move_white_checker (p : Position) (bc : Position) : Position :=
  ⟨p.x + 2 * (bc.x - p.x), p.y + 2 * (bc.y - p.y)⟩

-- The main theorem to prove that it is impossible for the white checker
-- to capture both black checkers.
theorem white_checker_cannot_capture_both_black (white_checker : Position)
    (move_white_checker : Position → Position → Position)
    (p₁ p₂ : Position)
    (h₁ : p₁ = black_checker_1)
    (h₂ : p₂ = black_checker_2) :
    ∀ n : Int, white_checker = move_white_checker
        (move_white_checker white_checker p₁) p₂ →
        ⊢ false :=
by
  sorry

end white_checker_cannot_capture_both_black_l65_65622


namespace problem1_problem2_l65_65479

open BigOperators

-- Conditions
def b_sequence (n : ℕ) : ℝ := 2^n
def a_sequence (n : ℕ) : ℝ := Real.logb 2 (b_sequence n) + 2
def c_sequence (n : ℕ) : ℝ := 1 / (a_sequence n * a_sequence (n + 1))

-- Questions translated to Lean 4 statement
theorem problem1 (n : ℕ) : ∃ d : ℕ, ∀ m : ℕ, a_sequence (m + 1) = a_sequence m + d := 
  by sorry

theorem problem2 (n : ℕ) : ∑ k in Finset.range n, c_sequence k = n / (3 * n + 9) :=
  by sorry

end problem1_problem2_l65_65479


namespace waste_in_scientific_notation_l65_65159

def water_waste_per_person : ℝ := 0.32
def number_of_people : ℝ := 10^6

def total_daily_waste : ℝ := water_waste_per_person * number_of_people

def scientific_notation (x : ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

theorem waste_in_scientific_notation :
  scientific_notation total_daily_waste ∧ total_daily_waste = 3.2 * 10^5 :=
by
  sorry

end waste_in_scientific_notation_l65_65159


namespace arithmetic_sequence_x_value_l65_65053

theorem arithmetic_sequence_x_value :
  ∃ x : ℚ, (x - 2) - (3/4) = (5 * x) - (x - 2) ∧ x = -19/12 :=
by
  -- Definitions arising from conditions
  have h1 : ∀ (x : ℚ), (x - 2) - (3/4) = (5 * x) - (x - 2) → x = -19/12
  intros x h
  -- Insert further steps and simplifications within sorry block if desired
  sorry
  use (-19/12)
  split
  repeat sorry -- Placeholder for actual proof steps

end arithmetic_sequence_x_value_l65_65053


namespace fg_of_neg2_l65_65873

def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x + 5

theorem fg_of_neg2 : f (g (-2)) = 1 := by
  sorry

end fg_of_neg2_l65_65873


namespace lars_total_breads_per_day_l65_65208

def loaves_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def hours_per_day : ℕ := 6

theorem lars_total_breads_per_day :
  (loaves_per_hour * hours_per_day) + ((hours_per_day / 2) * baguettes_per_two_hours) = 150 :=
  by 
  sorry

end lars_total_breads_per_day_l65_65208


namespace linear_function_solution_l65_65874

open Function

theorem linear_function_solution (f : ℝ → ℝ)
  (h_lin : ∃ k b, k ≠ 0 ∧ ∀ x, f x = k * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x - 1) :
  (∀ x, f x = 2 * x - 1 / 3) ∨ (∀ x, f x = -2 * x + 1) :=
by
  sorry

end linear_function_solution_l65_65874


namespace trailing_zeros_in_square_l65_65759

-- Define x as given in the conditions
def x : ℕ := 10^12 - 4

-- State the theorem which asserts that the number of trailing zeros in x^2 is 11
theorem trailing_zeros_in_square : 
  ∃ n : ℕ, n = 11 ∧ x^2 % 10^12 = 0 :=
by
  -- Placeholder for the proof
  sorry

end trailing_zeros_in_square_l65_65759


namespace tree_height_l65_65280

theorem tree_height (D h : ℝ) (angle_A angle_B : ℝ) (move_distance : ℝ)
  (h1 : angle_A = 15) (h2 : angle_B = 30) (h3 : move_distance = 40)
  (h4 : h = (D + move_distance) * tan (angle_A * (Real.pi / 180)))
  (h5 : h = D * tan (angle_B * (Real.pi / 180))) : 
h = 20 :=
by
  sorry

end tree_height_l65_65280


namespace four_digit_number_in_grid_l65_65271
-- Import the math library

-- Define the proof problem statement
theorem four_digit_number_in_grid (grid : Matrix ℕ 3 3) (arrows : Matrix ℕ 3 3) 
  (arrow_condition: ∀ i j : Fin 3, valid_arrow grid[i, j] arrows[i, j]) : 
  grid[1, 0] = 1 ∧ grid[1, 1] = 2 ∧ grid[1, 2] = 1 ∧ grid[1, 3] = 2 :=
by
  sorry

end four_digit_number_in_grid_l65_65271


namespace find_lambda_l65_65109

noncomputable def angle_between_vectors (a b : ℝ^3) := (π / 3)

theorem find_lambda
  (a b : ℝ^3)
  (λ : ℝ)
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = 2)
  (h3 : (λ • a + b) ⬝ (2 • a - λ • b) = 0)
  (h4 : angle_between_vectors a b = π / 3) :
  λ = -1 + real.sqrt 3 ∨ λ = -1 - real.sqrt 3 :=
sorry

end find_lambda_l65_65109


namespace problem_part1_problem_part2_l65_65764

-- Part 1
theorem problem_part1 : (1 / 2) ^ (-1 : ℤ) - (sqrt 2019 - 1) ^ 0 = 1 := 
by
  sorry

-- Part 2
theorem problem_part2 (x y : ℝ) : (x - y) ^ 2 - (x + 2 * y) * (x - 2 * y) = -2 * x * y + 5 * y ^ 2 := 
by
  sorry

end problem_part1_problem_part2_l65_65764


namespace statement_A_statement_B_statement_C_statement_D_l65_65928

def set_A : set ℝ := {x | x^2 + 2*x - 3 > 0}
def set_B (a : ℝ) : set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0}

theorem statement_A (a : ℝ) (h : a > 0) :
  (∃ n : ℤ, ↑n ∈ set_A ∩ set_B a ∧ ∀ m : ℤ, (↑m ∈ set_A ∩ set_B a) → m = n) →
  a ∈ set.Ico (3/4 : ℝ) (4/3 : ℝ) :=
sorry

theorem statement_B (a : ℝ) :
  set_A ∩ set_B a = ∅ ↔ a ∈ set.Icc (-4/3 : ℝ) 0 :=
sorry

theorem statement_C (a : ℝ) :
  (set_A ∪ set_B a = set.univ) → ¬ (a ∈ set.Ioo (-4/3 : ℝ) 0) :=
sorry

theorem statement_D (a : ℝ) :
  ¬ (set_A ∩ set_B a = set.univ) :=
sorry

end statement_A_statement_B_statement_C_statement_D_l65_65928


namespace measure_of_obtuse_angle_ADB_l65_65182

-- Definitions of our conditions for the right triangle ABC
variables (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables (angle_A angle_B : ℝ)

-- Given conditions
def right_triangle_ABC : Prop :=
  angle_A = 45 ∧ angle_B = 45 ∧ angle_A + angle_B = 90

-- Definition of the angle bisectors intersection at point D
def bisectors_meet_at_D : Prop :=
  ∃ D, true

-- Angle splitting property due to bisectors
def angle_bisectors_split : Prop :=
  angle_A / 2 = 22.5 ∧ angle_B / 2 = 22.5

-- Main theorem statement
theorem measure_of_obtuse_angle_ADB :
  right_triangle_ABC A B C ∧ bisectors_meet_at_D A B C D ∧ angle_bisectors_split A B →
  ∃ obtuse_angle_ADB, obtuse_angle_ADB = 270 :=
begin
  sorry
end

end measure_of_obtuse_angle_ADB_l65_65182


namespace problem_solution_l65_65598

theorem problem_solution
  (a b : ℝ)
  (h_eqn : ∃ (a b : ℝ), 3 * a * a + 9 * a - 21 = 0 ∧ 3 * b * b + 9 * b - 21 = 0 )
  (h_vieta_sum : a + b = -3)
  (h_vieta_prod : a * b = -7) :
  (2 * a - 5) * (3 * b - 4) = 47 := 
by
  sorry

end problem_solution_l65_65598


namespace Tim_has_7_times_more_l65_65771

-- Define the number of Dan's violet balloons
def Dan_violet_balloons : ℕ := 29

-- Define the number of Tim's violet balloons
def Tim_violet_balloons : ℕ := 203

-- Prove that the ratio of Tim's balloons to Dan's balloons is 7
theorem Tim_has_7_times_more (h : Tim_violet_balloons = 7 * Dan_violet_balloons) : 
  Tim_violet_balloons = 7 * Dan_violet_balloons := 
by {
  sorry
}

end Tim_has_7_times_more_l65_65771


namespace rocks_spit_out_l65_65429

def initial_rocks : ℕ := 10
def fish_ate_half (r : ℕ) : ℕ := r / 2
def remaining_rocks : ℕ := 7

theorem rocks_spit_out : ∀ (r : ℕ), r = initial_rocks → remaining_rocks = initial_rocks - (fish_ate_half initial_rocks) + r - 5 → r = 2 := 
by
  intro r,
  intro hr,
  intro hp,
  sorry

end rocks_spit_out_l65_65429


namespace find_m_l65_65861

theorem find_m (m : ℝ) (A B U : set ℝ) (hA : A = {2, m}) (hB : B = {1, m^2}) (hU : A ∪ B = {1, 2, 3, 9}) : m = 3 := 
by 
  sorry

end find_m_l65_65861


namespace sum_of_n_with_unformable_postage_120_equals_43_l65_65065

theorem sum_of_n_with_unformable_postage_120_equals_43 :
  ∃ n1 n2 : ℕ, n1 = 21 ∧ n2 = 22 ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n1 * b + (n1 + 1) * c) ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n2 * b + (n2 + 1) * c) ∧ 
  (120 = 7 * a + n1 * b + (n1 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (120 = 7 * a + n2 * b + (n2 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (n1 + n2 = 43) :=
by
  sorry

end sum_of_n_with_unformable_postage_120_equals_43_l65_65065


namespace grey_cubes_at_end_of_day_two_l65_65306

-- Define the cube structure
structure Cube :=
  (side_len : ℕ)
  (initial_grey : ℕ)

-- Define the neighbours relationship
def is_neighbour (a b : ℕ × ℕ × ℕ) : Prop :=
  (abs (a.1 - b.1) + abs (a.2 - b.2) + abs (a.3 - b.3) = 1)

-- Define the initial state of the cube and the conditions of the problem
noncomputable def initial_state := set.univ : set (ℕ × ℕ × ℕ)

def grey_at_day (n : ℕ) (state : set (ℕ × ℕ × ℕ)) : set (ℕ × ℕ × ℕ) :=
  match n with
  | 0 => { ⟨2, 2, 2⟩ } -- Assume the initial grey cube is (2, 2, 2)
  | n + 1 =>
    state ∪ { c | ∃ g ∈ state, is_neighbour g c ∧ c ∉ state }
  end

-- Prove the number of grey cubes at the end of the second day
theorem grey_cubes_at_end_of_day_two :
  (grey_at_day 2 initial_state).card = 17 :=
sorry

end grey_cubes_at_end_of_day_two_l65_65306


namespace comic_stack_ways_l65_65618

-- Define the factorial function for convenience
noncomputable def factorial : ℕ → ℕ
  | 0       => 1
  | (n + 1) => (n + 1) * factorial n

-- Conditions: Define the number of each type of comic book
def batman_comics := 7
def superman_comics := 4
def wonder_woman_comics := 5
def flash_comics := 3

-- The total number of comic books
def total_comics := batman_comics + superman_comics + wonder_woman_comics + flash_comics

-- Proof problem: The number of ways to stack the comics
theorem comic_stack_ways :
  (factorial batman_comics) * (factorial superman_comics) * (factorial wonder_woman_comics) * (factorial flash_comics) * (factorial 4) = 1102489600 := sorry

end comic_stack_ways_l65_65618


namespace smallest_n_for_quadratic_representation_l65_65449

theorem smallest_n_for_quadratic_representation :
  ∃ n : ℕ, (∀ n' : ℕ, (n' < n) → 
  ¬ ∃ (a_i b_i : Fin n' → ℚ), 
  (∀ x : ℚ, x^2 + x + 4 = ∑ i, (a_i i * x + b_i i)^2)) 
  ∧ ∃ (a_i b_i : Fin n → ℚ), 
  (∀ x : ℚ, x^2 + x + 4 = ∑ i, (a_i i * x + b_i i)^2) 
  := 
  ∃ (n = 5, (∀ n' : ℕ, (n' < 5) → 
  ¬ ∃ (a_i b_i : Fin n' → ℚ),
  (∀ x : ℚ, x^2 + x + 4 = ∑ i, (a_i i * x + b_i i)^2)) 
  ∧ ∃ (a_i b_i : Fin 5 → ℚ), 
  (∀ x : ℚ, x^2 + x + 4 = ∑ i, (a_i i * x + b_i i)^2)).

end smallest_n_for_quadratic_representation_l65_65449


namespace range_of_expression_l65_65185

theorem range_of_expression:
  ∀ (θ : ℝ), -4 ≤ (√3 * (2 * Real.cos θ) + (1/2) * (4 * Real.sin θ)) ∧ 
             (√3 * (2 * Real.cos θ) + (1/2) * (4 * Real.sin θ)) ≤ 4 :=
by
  sorry

end range_of_expression_l65_65185


namespace range_of_k_find_value_of_k_l65_65843

-- Definitions based on the conditions
def is_ellipse (k : ℝ) : Prop := (9 - k > 0) ∧ (k - 1 > 0) ∧ (9 - k ≠ k - 1)

def eccentricity_condition (k : ℝ) (e : ℝ) : Prop :=
  (e = √(6 / 7)) ∧
  (
    let a := if (9 - k > k - 1) then √(9 - k) else √(k - 1) in
    let b := if (9 - k > k - 1) then √(k - 1) else √(9 - k) in
    let c := if (9 - k > k - 1) then √(10 - 2 * k) else √(2 * k - 10) in
    e = c / a
  )

-- The Lean statements for both parts of the problem
theorem range_of_k : ∀ (k : ℝ), is_ellipse k → 1 < k ∧ k < 9 ∧ k ≠ 5 :=
by
  sorry

theorem find_value_of_k (e : ℝ) (k : ℝ) : is_ellipse k → eccentricity_condition k e → k = 2 ∨ k = 8 := 
by
  sorry

end range_of_k_find_value_of_k_l65_65843


namespace compare_abc_l65_65088

-- Define the given constants
def a : ℝ := 2 ^ 0.3
def b : ℝ := 3 ^ 2
def c : ℝ := 2 ^ (-0.3)

-- State the theorem representing the problem
theorem compare_abc : c < a ∧ a < b := 
by {
  -- Proof omitted
  sorry
}

end compare_abc_l65_65088


namespace midpoint_sum_correct_l65_65345

def midpoint_sum (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  let mz := (z1 + z2) / 2
  mx + my + mz

theorem midpoint_sum_correct : 
  midpoint_sum 8 (-4) 10 (-2) 6 (-2) = 8 :=
by 
  unfold midpoint_sum 
  simp 
  norm_num
  done

end midpoint_sum_correct_l65_65345


namespace bart_trees_needed_l65_65786

-- Define the constants and conditions given
def firewood_per_tree : Nat := 75
def logs_burned_per_day : Nat := 5
def days_in_november : Nat := 30
def days_in_december : Nat := 31
def days_in_january : Nat := 31
def days_in_february : Nat := 28

-- Calculate the total number of days from November 1 through February 28
def total_days : Nat := days_in_november + days_in_december + days_in_january + days_in_february

-- Calculate the total number of pieces of firewood needed
def total_firewood_needed : Nat := total_days * logs_burned_per_day

-- Calculate the number of trees needed
def trees_needed : Nat := total_firewood_needed / firewood_per_tree

-- The proof statement
theorem bart_trees_needed : trees_needed = 8 := 
by
  -- Placeholder for the proof
  sorry

end bart_trees_needed_l65_65786


namespace georgia_test_problems_l65_65470

/-- 
If Georgia completed 10 problems in the first 20 minutes,
twice as many problems in the next 20 minutes, 
and has 45 problems left to solve, 
then the total number of problems on the test is 75.
-/
theorem georgia_test_problems
  (completed_first_20 : ℕ)
  (completed_next_20 : ℕ)
  (problems_left : ℕ)
  (total_completed : completed_first_20 + completed_next_20 = 30)
  (completed_next_20_eq : completed_next_20 = 2 * completed_first_20)
  (total_problems : total_completed + problems_left = 75)
  : total_problems = 75 :=
sorry

end georgia_test_problems_l65_65470


namespace train_cross_time_platform_l65_65017

def speed := 36 -- in kmph
def time_for_pole := 12 -- in seconds
def time_for_platform := 44.99736021118311 -- in seconds

theorem train_cross_time_platform :
  time_for_platform = 44.99736021118311 :=
by
  sorry

end train_cross_time_platform_l65_65017


namespace prove_circle_trajectory_prove_line_intersection_through_fixed_point_l65_65198

noncomputable def circle_trajectory_through_point_and_tangent_line (p : ℝ) : Prop :=
  ∀ M : ℝ × ℝ,
    (M = (0, 1)) ∧ (M.2 = -1 - ((p - (-1)) / 2)) →
    (M.1^2 = 4 * M.2)

noncomputable def line_passing_fixed_point_and_intersecting_parabola (k : ℝ) : Prop :=
  ∀ A B C : ℝ × ℝ,
    (A ∈ {(x, y) | y = k * x - 2} ∧ B ∈ {(x, y) | y = k * x - 2} ∧ C = (-B.1, B.2)) →
    (A.1 + B.1 = 0 ∧ A.2 + C.2 = 4) →  -- Points symmetric about y-axis and line equation conditions
    (A.1 = 0 ∧ A.2 = 2)                -- Conclusion that AC passes through (0, 2)

theorem prove_circle_trajectory :
  circle_trajectory_through_point_and_tangent_line 2 := by
  sorry

theorem prove_line_intersection_through_fixed_point :
  line_passing_fixed_point_and_intersecting_parabola (sqrt 2) := by
  sorry

end prove_circle_trajectory_prove_line_intersection_through_fixed_point_l65_65198


namespace zero_intervals_l65_65808

noncomputable def f : ℝ → ℝ := λ x,  if x >= -5 then 2^x - 3 else 2^(-10 - x) - 3

theorem zero_intervals 
  (k : ℤ)
  (h_symmetry : ∀ x, f (-x - 10) = f x)
  (h_definition : ∀ x, x >= -5 → f x = 2^x - 3)
  (h_zero : ∃ y ∈ ((k.to_real) + 1), f y = 0) :
  k = 1 ∨ k = -12 :=
  sorry

end zero_intervals_l65_65808


namespace find_curve_and_area_l65_65316

theorem find_curve_and_area :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → (∃ y : ℝ, y = x + 4 - 4 * (Real.sqrt x))) ∧ 
  (∫ x in 0..4, x + 4 - 4 * (Real.sqrt x)) = 8 / 3 :=
by
  sorry

end find_curve_and_area_l65_65316


namespace d_r_difference_l65_65540

noncomputable def find_d_r_diff : ℕ := 
  let d := 19 in
  let r := 9 in
  d - r

theorem d_r_difference : find_d_r_diff = 10 := 
by
  sorry

end d_r_difference_l65_65540


namespace lateral_surface_area_cone_l65_65837

-- Given definitions (conditions)
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Question transformed into a theorem statement
theorem lateral_surface_area_cone : 
  ∀ (r l : ℝ), r = radius → l = slant_height → (1 / 2) * (2 * real.pi * r) * l = 15 * real.pi := 
by 
  intros r l hr hl
  rw [hr, hl]
  sorry

end lateral_surface_area_cone_l65_65837


namespace a_n_formula_b_n_formula_S_n_formula_l65_65849

open Nat

def f (x: ℕ) : ℕ := 2 * x + 1

def a (n: ℕ) : ℕ :=
if h : n ≠ 0 then f n
else 0

def b : ℕ → ℕ
| 0     := 2
| n + 1 := 2 * b n

def c (n: ℕ) : ℚ := (a n : ℚ) / (b n : ℚ)

def S (n: ℕ) : ℚ :=
if n = 1 then 1 / 2 else 3 - (2 * n + 5) / (2 ^ n : ℚ)

theorem a_n_formula (n : ℕ) (h : n ≠ 0) : a n = 2 * n + 1 := by
  sorry

theorem b_n_formula (n : ℕ) : b n = 2 ^ n := by
  sorry

theorem S_n_formula (n : ℕ) : 
  (Σ i in range (n + 1), c (i + 1)) = S (n + 1) := by
  sorry

end a_n_formula_b_n_formula_S_n_formula_l65_65849


namespace disks_inequality_l65_65912

variable {n : ℕ}
variable {R : Fin n → ℝ}
variable {O : ℝ × ℝ}
variable {P : Fin n → ℝ × ℝ}
variable {D : Fin n → Set (ℝ × ℝ)}

noncomputable def disk (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2 }

-- Conditions
variable (cond1 : n ≥ 6)
variable (cond2 : ∀ i j : Fin n, i ≠ j → D i ∩ D j = ∅)
variable (cond3 : ∀ i : Fin n, D i = disk (D i).center (R i))
variable (cond4 : ∀ i : Fin n, P i ∈ D i)
variable (cond5 : ∀ i j : Fin n, i ≤ j → R i ≥ R j)

-- Statement
theorem disks_inequality :
  OP_sum : (Σ i : Fin n, dist O (P i))
  (R_sum : (Σ i in (Finset.range n).filter (λ i, i ≥ 6), R i))
  (OP_sum >= R_sum) :=
begin
  sorry
end

end disks_inequality_l65_65912


namespace jeremy_goal_product_l65_65163

theorem jeremy_goal_product 
  (g1 g2 g3 g4 g5 : ℕ) 
  (total5 : g1 + g2 + g3 + g4 + g5 = 13)
  (g6 g7 : ℕ) 
  (h6 : g6 < 10) 
  (h7 : g7 < 10) 
  (avg6 : (13 + g6) % 6 = 0) 
  (avg7 : (13 + g6 + g7) % 7 = 0) :
  g6 * g7 = 15 := 
sorry

end jeremy_goal_product_l65_65163


namespace cumulative_vibrations_is_correct_l65_65250

-- Define the initial conditions
def initial_vibrations : ℕ := 1600

def percentage_increase (setting : ℕ) : ℕ :=
  match setting with
  | 2 => 10
  | 3 => 22
  | 4 => 35
  | 5 => 50
  | _ => 0

def time_in_seconds (setting : ℕ) : ℕ :=
  match setting with
  | 1 => (3 * 60) + 15
  | 2 => (5 * 60) + 45
  | 3 => (4 * 60) + 30
  | 4 => (6 * 60) + 10
  | 5 => (8 * 60) + 20
  | _ => 0

-- Calculate number of vibrations per second for each setting
def vibrations_per_second (setting : ℕ) : ℕ :=
  initial_vibrations + (initial_vibrations * percentage_increase(setting) / 100)

-- Calculate total vibrations for each setting
def total_vibrations (setting : ℕ) : ℕ :=
  vibrations_per_second(setting) * time_in_seconds(setting)

-- Calculate cumulative vibrations
def cumulative_vibrations : ℕ :=
  total_vibrations(1) + total_vibrations(2) + total_vibrations(3) + total_vibrations(4) + total_vibrations(5)

theorem cumulative_vibrations_is_correct : cumulative_vibrations = 3445440 :=
  by sorry

end cumulative_vibrations_is_correct_l65_65250


namespace remainder_is_15_l65_65990

-- Definitions based on conditions
def S : ℕ := 476
def L : ℕ := S + 2395
def quotient : ℕ := 6

-- The proof statement
theorem remainder_is_15 : ∃ R : ℕ, L = quotient * S + R ∧ R = 15 := by
  sorry

end remainder_is_15_l65_65990


namespace ellipse_foci_x_axis_l65_65653

theorem ellipse_foci_x_axis (m n : ℝ) (h_eq : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1)
  (h_foci : ∃ (c : ℝ), c = 0 ∧ (c^2 = 1 - n/m)) : n > m ∧ m > 0 ∧ n > 0 :=
sorry

end ellipse_foci_x_axis_l65_65653


namespace max_principals_in_10_years_l65_65046

theorem max_principals_in_10_years (term_length : ℕ) (total_years : ℕ) (h_term : term_length = 4) (h_years : total_years = 10) :
  ∃ max_principals : ℕ, max_principals = 4 :=
by
  use 4
  sorry

end max_principals_in_10_years_l65_65046


namespace g_recursion_l65_65592

noncomputable def g (n : ℕ) : ℚ√7 := 
  (7 + 4 * √7) / 14 * ((1 + √7) / 2)^n + (7 - 4 * √7) / 14 * ((1 - √7) / 2)^n 

theorem g_recursion (n : ℕ) : g (n + 2) - g n = g n := 
  sorry

end g_recursion_l65_65592


namespace repayment_amount_l65_65211

theorem repayment_amount (borrowed amount : ℝ) (increase_percentage : ℝ) (final_amount : ℝ) 
  (h1 : borrowed_amount = 100) 
  (h2 : increase_percentage = 0.10) :
  final_amount = borrowed_amount * (1 + increase_percentage) :=
by 
  rw [h1, h2]
  norm_num
  exact eq.refl 110


end repayment_amount_l65_65211


namespace area_of_triangle_l65_65553

theorem area_of_triangle (c : ℝ) (tanA : ℝ) (cosC : ℝ) (h1 : c = 4) (h2 : tanA = 3) (h3 : cosC = (√5) / 5) : 
  let sinC := 2 * √5 / 5
  let sinA := 3 * √10 / 10
  let b := √10
  let area := (1 / 2) * b * c * sinA in
  area = 6 :=
sorry

end area_of_triangle_l65_65553


namespace methane_needed_l65_65446

theorem methane_needed (total_benzene_g : ℝ) (molar_mass_benzene : ℝ) (toluene_moles : ℝ) : 
  total_benzene_g = 156 ∧ molar_mass_benzene = 78 ∧ toluene_moles = 2 → 
  toluene_moles = total_benzene_g / molar_mass_benzene := 
by
  intros
  sorry

end methane_needed_l65_65446


namespace supporters_received_all_items_l65_65405

def lcm (a b : Nat) : Nat := Nat.lcm a b

theorem supporters_received_all_items (n : Nat) :
  (∀ k, k ≥ 1 → (k % 25 = 0 → k ≤ n) → 
  (k % 40 = 0 → k ≤ n) →
  (k % 90 = 0 → k ≤ n) →
  n = 5000) →
  let l := lcm 25 (lcm 40 90) in
  n / l = 2 :=
by
  sorry

end supporters_received_all_items_l65_65405


namespace Andy_hours_worked_l65_65752

noncomputable def hourly_rate := 9
noncomputable def pay_per_racquet := 15
noncomputable def pay_per_grommet := 10
noncomputable def pay_per_stencil := 1
noncomputable def total_earnings := 202
noncomputable def num_racquets := 7
noncomputable def num_grommets := 2
noncomputable def num_stencils := 5

theorem Andy_hours_worked :
  let earnings_from_racquets := num_racquets * pay_per_racquet,
      earnings_from_grommets := num_grommets * pay_per_grommet,
      earnings_from_stencils := num_stencils * pay_per_stencil,
      total_service_earnings := earnings_from_racquets + earnings_from_grommets + earnings_from_stencils,
      hourly_wage_earnings := total_earnings - total_service_earnings,
      hours_worked := hourly_wage_earnings / hourly_rate
  in hours_worked = 8 :=
by
  -- Proof goes here
  sorry

end Andy_hours_worked_l65_65752


namespace LocusOnSegmentAB_LocusInsideTriangle_l65_65186

section LocusOrthocenter

variables (O A B M P Q H : Type)
variable h : HasSmallerThan (angle A O B) 90
variable hM : ∀ M : Point, M ∈ LineSegment A B ∨ M ∈ TriangleInterior O A B
variable hP : ∀ P : Point, IsPerpendicular (Line M P) (Line O A)
variable hQ : ∀ Q : Point, IsPerpendicular (Line M Q) (Line O B)
variable hH : IsOrthocenter H (Triangle O P Q)

-- Locus of H when M is on segment AB
theorem LocusOnSegmentAB (M : Point) (hMseg : M ∈ LineSegment A B) : 
  H ∈ LineSegment (Orthocenter A) (Orthocenter B) :=
by
  sorry

-- Locus of H when M is in the interior of the triangle
theorem LocusInsideTriangle (M : Point) (hMtri : M ∈ TriangleInterior O A B) : 
  H ∈ TriangleInterior O (Orthocenter A) (Orthocenter B) :=
by
  sorry

end LocusOrthocenter

end LocusOnSegmentAB_LocusInsideTriangle_l65_65186


namespace eval_expression_l65_65702

theorem eval_expression : 
  (520 * 0.43 / 0.26 - 217 * (2 + 3/7)) - (31.5 / (12 + 3/5) + 114 * (2 + 1/3) + (61 + 1/2)) = 0.5 := 
by
  sorry

end eval_expression_l65_65702


namespace carolyn_stickers_l65_65412

open Nat

theorem carolyn_stickers :
  ∀ (belle_stickers : Nat) (difference : Nat),
    belle_stickers = 97 →
    difference = 18 →
    belle_stickers - difference = 79 :=
by
  intros belle_stickers difference h_belle h_diff
  rw [h_belle, h_diff]
  exact rfl

end carolyn_stickers_l65_65412


namespace minimum_time_to_finish_food_l65_65401

-- Define the constants involved in the problem
def carrots_total : ℕ := 1000
def muffins_total : ℕ := 1000
def amy_carrots_rate : ℝ := 40 -- carrots per minute
def amy_muffins_rate : ℝ := 70 -- muffins per minute
def ben_carrots_rate : ℝ := 60 -- carrots per minute
def ben_muffins_rate : ℝ := 30 -- muffins per minute

-- Proof statement
theorem minimum_time_to_finish_food : 
  ∃ T : ℝ, 
  (∀ c : ℝ, c = 5 → 
  (∀ T_1 : ℝ, T_1 = (carrots_total / (amy_carrots_rate + ben_carrots_rate)) → 
  (∀ T_2 : ℝ, T_2 = ((muffins_total + (amy_muffins_rate * c)) / (amy_muffins_rate + ben_muffins_rate)) +
  (muffins_total / ben_muffins_rate) - T_1 - c →
  T = T_1 + T_2) ∧
  T = 23.5 )) :=
sorry

end minimum_time_to_finish_food_l65_65401


namespace symmetrical_hexagon_exists_l65_65370

-- Define the regular pentagon base and its properties
structure RegularPentagonBase where
  A B C D E : Point 
  unit_length : ∀ (P Q : Point) [Side (P Q) ∈ {A B, B C, C D, D E, E A}], distance P Q = 1
  angles_108_deg : ∀ (P ∈ {A, B, C, D, E}), angle P = 108

-- Define the pyramid with the regular pentagon base and its vertex
structure RegularPentagonalPyramid where
  base : RegularPentagonBase
  F : Point

open RegularPentagonBase RegularPentagonalPyramid

theorem symmetrical_hexagon_exists (pyramid : RegularPentagonalPyramid) : 
  ∃ (plane : Plane), is_hexagon (plane ∩ pyramid) ∧ is_symmetrical (plane ∩ pyramid) := 
sorry

end symmetrical_hexagon_exists_l65_65370


namespace intersection_M_N_l65_65090

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1, 2}
def N : Set ℤ := {x | ∃ y ∈ M, |y| = x}

-- The main theorem to prove M ∩ N = {0, 1, 2}
theorem intersection_M_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_M_N_l65_65090


namespace original_height_of_tree_l65_65925

theorem original_height_of_tree
  (current_height_in_inches : ℕ)
  (percent_taller : ℕ)
  (current_height_is_V := 180)
  (percent_taller_is_50 := 50) :
  (current_height_in_inches * 100) / (percent_taller + 100) / 12 = 10 := sorry

end original_height_of_tree_l65_65925


namespace monotonic_quadratic_range_l65_65523

-- Define a quadratic function
noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- The theorem
theorem monotonic_quadratic_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → quadratic a x₁ ≤ quadratic a x₂) ∨
  (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → quadratic a x₁ ≥ quadratic a x₂) →
  (a ≤ 2 ∨ 3 ≤ a) :=
sorry

end monotonic_quadratic_range_l65_65523


namespace find_ellipse_and_range_of_AB_l65_65828

noncomputable def ellipse_equation (a b : ℝ) (h1 : a^2 = b^2 + 1) (h2 : 1/a^2 + 1/(2*b^2) = 1) : set (ℝ × ℝ) :=
  { p | p.1^2 / a^2 + p.2^2 / b^2 = 1 }

noncomputable def intersection_line_ellipse (k : ℝ) (x y : ℝ) (C : set (ℝ × ℝ)) : Prop :=
  (y = k * (x - 2)) ∧ ((x^2 / 2) + y^2 = 1)

theorem find_ellipse_and_range_of_AB :
  ∃ C : set (ℝ × ℝ),
    (∀ x y, (C (1, sqrt 2 / 2) ∧ (∀ t, (t ∈ (2 * sqrt 6 / 3, 2) → (|AB| ∈ (0, 2 * sqrt 5 / 3))))))) ∧
    (C = { p | p.1^2 / 2 + p.2^2 = 1 }) :=
begin
  sorry
end

end find_ellipse_and_range_of_AB_l65_65828


namespace num_words_with_consonant_l65_65139

-- Definitions
def letters : List Char := ['A', 'B', 'C', 'D', 'E']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D']

-- Total number of 4-letter words without restrictions
def total_words : Nat := 5 ^ 4

-- Number of 4-letter words with only vowels
def vowels_only_words : Nat := 2 ^ 4

-- Number of 4-letter words with at least one consonant
def words_with_consonant : Nat := total_words - vowels_only_words

theorem num_words_with_consonant : words_with_consonant = 609 := by
  -- Add proof steps
  sorry

end num_words_with_consonant_l65_65139


namespace intersection_complement_l65_65930

def U := {-2, -1, 0, 1, 2}
def A := {-1, 1}
def B := {0, 1, 2}
def C_U_B := U \ B

theorem intersection_complement :
  A ∩ C_U_B = {-1} := by
  sorry

end intersection_complement_l65_65930


namespace total_tickets_sold_l65_65742

-- We start by defining our variables and conditions
variables {x y : ℕ}

-- The condition for the total cost of tickets
def total_cost := 12 * x + 8 * y = 3320

-- The condition for the number of tickets sold for seats in the balcony
def balcony_condition := y = x + 115

-- The final theorem stating the total number of tickets
theorem total_tickets_sold : total_cost → balcony_condition → x + y = 355 :=
by
  intros h1 h2
  sorry

end total_tickets_sold_l65_65742


namespace austin_needs_six_weeks_l65_65408

theorem austin_needs_six_weeks
  (work_rate: ℕ) (hours_monday hours_wednesday hours_friday: ℕ) (bicycle_cost: ℕ) 
  (weekly_hours: ℕ := hours_monday + hours_wednesday + hours_friday) 
  (weekly_earnings: ℕ := weekly_hours * work_rate) 
  (weeks_needed: ℕ := bicycle_cost / weekly_earnings):
  work_rate = 5 ∧ hours_monday = 2 ∧ hours_wednesday = 1 ∧ hours_friday = 3 ∧ bicycle_cost = 180 ∧ weeks_needed = 6 :=
by {
  sorry
}

end austin_needs_six_weeks_l65_65408


namespace coprime_in_ten_consecutive_l65_65285

theorem coprime_in_ten_consecutive (k : ℤ) :
  ∃ a ∈ (finset.range 10).map (λ i, k + i), ∀ b ∈ (finset.range 10).map (λ i, k + i), a ≠ b → Int.gcd a b = 1 :=
sorry

end coprime_in_ten_consecutive_l65_65285


namespace village_population_l65_65744

noncomputable def number_of_people_in_village
  (vampire_drains_per_week : ℕ)
  (werewolf_eats_per_week : ℕ)
  (weeks : ℕ) : ℕ :=
  let drained := vampire_drains_per_week * weeks
  let eaten := werewolf_eats_per_week * weeks
  drained + eaten

theorem village_population :
  number_of_people_in_village 3 5 9 = 72 := by
  sorry

end village_population_l65_65744


namespace point_in_second_quadrant_l65_65319

/--
Point P lies in the second quadrant if its x-coordinate is negative and its y-coordinate is positive.
-/
theorem point_in_second_quadrant {x y : ℝ} (h1 : x = -2) (h2 : y = 3) : x < 0 ∧ y > 0 :=
by {
  rw [h1, h2],
  split;
  sorry
}

end point_in_second_quadrant_l65_65319


namespace find_n_divisible_by_highest_power_of_2_l65_65072

def a_n (n : ℕ) : ℕ :=
  10^n * 999 + 488

theorem find_n_divisible_by_highest_power_of_2:
  ∀ n : ℕ, (n > 0) → (a_n n = 10^n * 999 + 488) → (∃ k : ℕ, 2^(k + 9) ∣ a_n 6) := sorry

end find_n_divisible_by_highest_power_of_2_l65_65072


namespace length_of_train_is_600_meters_l65_65709

def length_of_train_equals_length_of_platform (l : ℕ) : Prop :=
l = l  -- Dummy condition representing the equal length of train and platform

theorem length_of_train_is_600_meters
  (speed_kmph : ℕ) (time_seconds : ℕ) : Prop :=
  speed_kmph = 72 → time_seconds = 60 →
  let speed_mps := speed_kmph * 1000 / 3600 in
  let distance := speed_mps * time_seconds in
  let total_length := distance / 2 in
  total_length = 600

#reduce length_of_train_equals_length_of_platform
#reduce length_of_train_is_600_meters

/- 
We define the conditions and the proof statement. 
  condition 1: The length of the train and platform are set internally as equal.
  condition 2: The speed of the train is provided in km/hr.
  condition 3: The train crosses the platform in a specified time.
-/

end length_of_train_is_600_meters_l65_65709


namespace ChipsEquivalence_l65_65456

theorem ChipsEquivalence
  (x y : ℕ)
  (h1 : y = x - 2)
  (h2 : 3 * x - 3 = 4 * y - 4) :
  3 * x - 3 = 24 :=
by
  sorry

end ChipsEquivalence_l65_65456


namespace unique_equal_real_roots_l65_65353

theorem unique_equal_real_roots :
  let eqA := (λ x: ℝ, x^2 + x + 1)
  let eqB := (λ x: ℝ, 4 * x^2 + 2 * x + 1)
  let eqC := (λ x: ℝ, x^2 + 12 * x + 36)
  let eqD := (λ x: ℝ, x^2 + x - 2)
  (∃ a b c : ℝ, ∀ x : ℝ, (a * x^2 + b * x + c = 0) ∧ (b^2 - 4 * a * c = 0) ↔ ∃ x : ℝ, (eqC x = 0)) :=
by {
  sorry
}

end unique_equal_real_roots_l65_65353


namespace lark_combination_count_l65_65590

-- Definitions for the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- Condition definitions
def valid_first_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 30 ∧ is_odd n
def valid_second_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 30 ∧ is_even n
def valid_third_number (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 40 ∧ is_multiple_of_5 n

-- Main theorem statement
theorem lark_combination_count : 
  let first_count := ∑ i in finset.range 31, if valid_first_number i then 1 else 0,
      second_count := ∑ i in finset.range 31, if valid_second_number i then 1 else 0,
      third_count := ∑ i in finset.range 41, if valid_third_number i then 1 else 0 in
  first_count * second_count * third_count = 1800 :=
by 
  sorry

end lark_combination_count_l65_65590


namespace power_of_hundred_expansion_l65_65349

theorem power_of_hundred_expansion :
  ∀ (n : ℕ), (100 = 10^2) → (100^50 = 10^100) :=
by
  intros n h
  rw [h]
  exact pow_mul 10 2 50
# You can replace pow_mul with the relevant lemmata if it conflicts.

end power_of_hundred_expansion_l65_65349


namespace sqrt7_minus_3_lt_sqrt5_minus_2_l65_65420

theorem sqrt7_minus_3_lt_sqrt5_minus_2:
  (2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3) ∧ (2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) -> 
  Real.sqrt 7 - 3 < Real.sqrt 5 - 2 := by
  sorry

end sqrt7_minus_3_lt_sqrt5_minus_2_l65_65420


namespace prove_curve_and_distance_l65_65508

-- Define curve E in polar coordinates
def curve_polar (ρ θ : ℝ) : Prop :=
  4 * (ρ^2 - 4) * (sin θ)^2 = (16 - ρ^2) * (cos θ)^2

-- Define point P in polar coordinates
def point_P (α : ℝ) : ℝ × ℝ :=
  (4 * cos α, 2 * sin α)

-- Define point M as the midpoint of segment OP
def point_M (α : ℝ) : ℝ × ℝ :=
  (2 * cos α, sin α)

-- Parameterized line l
def param_line_l (t : ℝ) : ℝ × ℝ :=
  (-sqrt 2 + (2 * sqrt 5 / 5) * t, sqrt 2 + (sqrt 5 / 5) * t)

-- General equation of line l
def line_l (x y : ℝ) : Prop :=
  x - 2 * y + 3 * sqrt 2 = 0

-- Maximum distance from point M to line l
def max_distance_M_to_l (M : ℝ × ℝ) : ℝ :=
  ((2 * M.1 - 2 * M.2 + 3 * sqrt 2).abs) / sqrt 5

-- The formalized math proof problem
theorem prove_curve_and_distance :
  (∀ (ρ θ : ℝ), curve_polar ρ θ →
    ∃ (x y : ℝ), x^2 + 4 * y^2 = 16) ∧
  (∀ (α : ℝ), ∃ (d : ℝ), max_distance_M_to_l (point_M α) ≤ sqrt 10) :=
by
  sorry

end prove_curve_and_distance_l65_65508


namespace dorothy_age_proof_l65_65432

noncomputable def ticket_cost (age : ℕ) : ℕ := if age ≤ 18 then 7 else 10

def total_cost (dorothy_age : ℕ) : ℕ :=
  2 * (ticket_cost dorothy_age) + 3 * 10

theorem dorothy_age_proof (dorothy_age : ℕ) : total_cost dorothy_age = 44 → dorothy_age ≤ 18 :=
by 
  intro h,
  have ht : ticket_cost dorothy_age = 7 :=
    by {
      by_cases dorothy_age ≤ 18,
      { unfold ticket_cost, simp [h] },
      { sorry }
    },
  unfold total_cost at h,
  rw [ht] at h,
  simp at h,
  sorry

end dorothy_age_proof_l65_65432


namespace mushrooms_used_by_Karla_correct_l65_65082

-- Given conditions
def mushrooms_cut_each_mushroom : ℕ := 4
def mushrooms_cut_total : ℕ := 22 * mushrooms_cut_each_mushroom
def mushrooms_used_by_Kenny : ℕ := 38
def mushrooms_remaining : ℕ := 8
def mushrooms_total_used_by_Kenny_and_remaining : ℕ := mushrooms_used_by_Kenny + mushrooms_remaining
def mushrooms_used_by_Karla : ℕ := mushrooms_cut_total - mushrooms_total_used_by_Kenny_and_remaining

-- Statement to prove
theorem mushrooms_used_by_Karla_correct :
  mushrooms_used_by_Karla = 42 :=
by
  sorry

end mushrooms_used_by_Karla_correct_l65_65082


namespace remainder_of_385857_div_6_is_3_l65_65680

theorem remainder_of_385857_div_6_is_3:
  let n := 385857 in
  n % 2 ≠ 0 ∧ n % 3 = 0 → n % 6 = 3 :=
by
  sorry

end remainder_of_385857_div_6_is_3_l65_65680


namespace number_of_pairs_eq_two_l65_65534

theorem number_of_pairs_eq_two :
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x > 0 ∧ y > 0 ∧ x^2 - y^2 = 91}.toFinset.card = 2 :=
sorry

end number_of_pairs_eq_two_l65_65534


namespace hyperbola_eccentricity_l65_65448

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_eq1 : ∀ x y : ℝ, abs (2 * x - y) < ε → abs x < a → abs y < b) -- Approximation of asymptote 2x-y=0
  (h_eq2 : ∀ x y : ℝ, abs (2 * x + y) < ε → abs x < a → abs y < b) -- Approximation of asymptote 2x+y=0
  : (∃ k : ℝ, k > 0 ∧ b = 2 * k ∧ a = k) → (e : ℝ, e = sqrt 5) := sorry

end hyperbola_eccentricity_l65_65448


namespace maximilian_annual_revenue_l65_65254

-- Define the number of units in the building
def total_units : ℕ := 100

-- Define the occupancy rate
def occupancy_rate : ℚ := 3 / 4

-- Define the monthly rent per unit
def monthly_rent : ℚ := 400

-- Calculate the number of occupied units
def occupied_units : ℕ := (occupancy_rate * total_units : ℚ).natAbs

-- Calculate the monthly rent revenue
def monthly_revenue : ℚ := occupied_units * monthly_rent

-- Calculate the annual rent revenue
def annual_revenue : ℚ := monthly_revenue * 12

-- Prove that the annual revenue is $360,000
theorem maximilian_annual_revenue : annual_revenue = 360000 := by
  sorry

end maximilian_annual_revenue_l65_65254


namespace polynomial_divisibility_l65_65964

-- Definitions
def f (k l m n : ℕ) (x : ℂ) : ℂ :=
  x^(4 * k) + x^(4 * l + 1) + x^(4 * m + 2) + x^(4 * n + 3)

def g (x : ℂ) : ℂ :=
  x^3 + x^2 + x + 1

-- Theorem statement
theorem polynomial_divisibility (k l m n : ℕ) : ∀ x : ℂ, g x ∣ f k l m n x :=
  sorry

end polynomial_divisibility_l65_65964


namespace tan_of_negative_7pi_over_4_l65_65795

theorem tan_of_negative_7pi_over_4 : Real.tan (-7 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_of_negative_7pi_over_4_l65_65795


namespace point_B_outside_circle_l65_65909

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem point_B_outside_circle : 
  let center := (0, 1)
  let radius := 5
  let pointA := (3, 4)
  let pointB := (4, 5)
  let pointC := (5, 1)
  let pointD := (1, 5)
  (distance pointB center > radius) := 
sorry

end point_B_outside_circle_l65_65909


namespace train_passes_jogger_in_45_seconds_l65_65389

-- Definitions based on the conditions
def jogger_speed_kmh : ℝ := 12
def train_speed_kmh : ℝ := 60
def wind_speed_kmh : ℝ := 4
def initial_distance_m : ℝ := 300
def train_length_m : ℝ := 300

-- Effective speeds considering wind
def effective_jogger_speed_kmh : ℝ := jogger_speed_kmh - wind_speed_kmh
def effective_train_speed_kmh : ℝ := train_speed_kmh - wind_speed_kmh

-- Conversion factor from km/hr to m/s
def kmh_to_mps : ℝ := 1000 / 3600

-- Relative speed of train with respect to jogger in m/s
def relative_speed_mps : ℝ := (effective_train_speed_kmh - effective_jogger_speed_kmh) * kmh_to_mps

-- Total distance the train needs to cover in meters
def total_distance_m : ℝ := initial_distance_m + train_length_m

-- Time to pass the jogger in seconds
def time_to_pass_s : ℝ := total_distance_m / relative_speed_mps

-- Theorem statement
theorem train_passes_jogger_in_45_seconds : time_to_pass_s ≈ 45 := 
by sorry

end train_passes_jogger_in_45_seconds_l65_65389


namespace S_greater_T_l65_65095

def sequence_a (n : ℕ) : ℤ :=
  if h1 : 1 ≤ n ∧ n ≤ 729 then
    (2 * n - 1) * (-1)^(n + 1)
  else
    0

def sequence_b (k : ℕ) : ℤ :=
  if h2 : ∃ n, (n ≥ 1 ∧ n ≤ 729) ∧ k = mapped_to_b n then
    -3 * (-3)^(k-1)
  else
    0

noncomputable def S : ℤ :=
  ∑ n in Finset.range 730, sequence_a n

noncomputable def T : ℤ :=
  ∑ k in Finset.range 7, sequence_b k

theorem S_greater_T : S > T := by
  sorry

end S_greater_T_l65_65095


namespace shirt_cost_l65_65542

theorem shirt_cost
  (J S B : ℝ)
  (h1 : 3 * J + 2 * S = 69)
  (h2 : 2 * J + 3 * S = 61)
  (h3 : 3 * J + 3 * S + 2 * B = 90) :
  S = 9 := 
by
  sorry

end shirt_cost_l65_65542


namespace trigonometric_identity_l65_65146

theorem trigonometric_identity (α : ℝ) 
  (h : Real.cos (5 * Real.pi / 12 - α) = Real.sqrt 2 / 3) :
  Real.sqrt 3 * Real.cos (2 * α) - Real.sin (2 * α) = 10 / 9 := sorry

end trigonometric_identity_l65_65146


namespace sum_of_elements_in_A_l65_65222

theorem sum_of_elements_in_A (a : ℝ) :
  let A := {x | x^2 + (a + 2)x + a + 1 = 0} in
  (if a = 0 then ∑ x in A, x = -1 else ∑ x in A, x = -(a + 2)) :=
by
  sorry

end sum_of_elements_in_A_l65_65222


namespace intersection_point_l65_65954

theorem intersection_point (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ d) :
  let x := (d - c) / (2 * b)
  let y := (a * (d - c)^2) / (4 * b^2) + (d + c) / 2
  (ax^2 + bx + c = y) ∧ (ax^2 - bx + d = y) :=
by
  sorry

end intersection_point_l65_65954


namespace product_mnp_l65_65776

theorem product_mnp (m n p : ℕ) (b x z c : ℂ) (h1 : b^8 * x * z - b^7 * z - b^6 * x = b^5 * (c^5 - 1)) 
  (h2 : (b^m * x - b^n) * (b^p * z - b^3) = b^5 * c^5) : m * n * p = 30 :=
sorry

end product_mnp_l65_65776


namespace hall_width_l65_65171

theorem hall_width 
  (L H cost total_expenditure : ℕ)
  (W : ℕ)
  (h1 : L = 20)
  (h2 : H = 5)
  (h3 : cost = 20)
  (h4 : total_expenditure = 19000)
  (h5 : total_expenditure = (L * W + 2 * (H * L) + 2 * (H * W)) * cost) :
  W = 25 := 
sorry

end hall_width_l65_65171


namespace distance_from_pole_l65_65913

-- Define the structure for polar coordinates.
structure PolarCoordinates where
  r : ℝ
  θ : ℝ

-- Define point A with its polar coordinates.
def A : PolarCoordinates := { r := 3, θ := -4 }

-- State the problem to prove that the distance |OA| is 3.
theorem distance_from_pole (A : PolarCoordinates) : A.r = 3 :=
by {
  sorry
}

end distance_from_pole_l65_65913


namespace trains_clear_each_other_in_24_seconds_l65_65337

def length_first_train : ℝ := 160
def length_second_train : ℝ := 320
def speed_first_train_kmph : ℝ := 42
def speed_second_train_kmph : ℝ := 30

def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

def speed_first_train_mps : ℝ := kmph_to_mps speed_first_train_kmph
def speed_second_train_mps : ℝ := kmph_to_mps speed_second_train_kmph
def relative_speed_mps : ℝ := speed_first_train_mps + speed_second_train_mps
def total_length : ℝ := length_first_train + length_second_train

def time_clear_each_other : ℝ := total_length / relative_speed_mps

theorem trains_clear_each_other_in_24_seconds : time_clear_each_other = 24 := by
  sorry

end trains_clear_each_other_in_24_seconds_l65_65337


namespace value_of_2a2_minus_a4_l65_65069

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (ha : ∀ n, a n = a 0 * q ^ n) : Prop := 
  ∀ n, S n = a 0 * (1 - q ^ n) / (1 - q)

theorem value_of_2a2_minus_a4 (a : ℕ → ℝ) (q : ℝ) (a0 : ℝ) 
  (S : ℕ → ℝ) (h0 : q ≠ 1)
  (h1 : geometric_sequence a q h0)
  (h2 : S 4 / S 2 = 3) : 
  2 * a 1 - a 3 = 0 := 
sorry

end value_of_2a2_minus_a4_l65_65069


namespace cycloid_to_cartesian_l65_65770

variable (a θ x y : ℝ)

-- Conditions as definitions
def parametric1 (a θ : ℝ) : ℝ := a * (θ - Real.sin θ)
def parametric2 (a θ : ℝ) : ℝ := a * (1 - Real.cos θ)

def range_theta (θ : ℝ) : Prop := 0 ≤ θ ∧ θ ≤ Real.pi
def a_positive (a : ℝ) : Prop := 0 < a

-- Theorem statement
theorem cycloid_to_cartesian :
  (∃ θ, x = parametric1 a θ ∧ y = parametric2 a θ ∧ range_theta θ) ∧ a_positive a →
  (x - a * Real.arccos ((a - y) / a))^2 + (y - a)^2 = a^2 := 
by
  sorry

end cycloid_to_cartesian_l65_65770


namespace lcm_220_504_l65_65341

/-- The least common multiple of 220 and 504 is 27720. -/
theorem lcm_220_504 : Nat.lcm 220 504 = 27720 :=
by
  -- This is the final statement of the theorem. The proof is not provided and marked with 'sorry'.
  sorry

end lcm_220_504_l65_65341


namespace solve_for_x_l65_65036

def f (x : ℝ) : ℝ := 3 * x - 4

noncomputable def f_inv (x : ℝ) : ℝ := (x + 4) / 3

theorem solve_for_x : ∃ x : ℝ, f x = f_inv x ∧ x = 2 := by
  sorry

end solve_for_x_l65_65036


namespace alex_age_thrice_ben_in_n_years_l65_65613

-- Definitions based on the problem's conditions
def Ben_current_age := 4
def Alex_current_age := Ben_current_age + 30

-- The main problem defined as a theorem to be proven
theorem alex_age_thrice_ben_in_n_years :
  ∃ n : ℕ, Alex_current_age + n = 3 * (Ben_current_age + n) ∧ n = 11 :=
by
  sorry

end alex_age_thrice_ben_in_n_years_l65_65613


namespace actual_time_is_5_22_pm_l65_65985

open Nat Rat Real

-- Definitions based on conditions
axiom car_clock_gain_constant (k : ℝ) : k > 1 -- car clock gains time at a constant rate greater than 1
axiom initial_sync : ∀ (t : ℕ), (t = 8 * 60) → (car_clock_time t) = t  -- both clocks indicated 8:00 AM at the same moment
axiom time_difference : (t_actual t_car : ℕ) → (t_actual = 8 * 60 + 30) /\ (t_car = 8 * 60 + 40) 
axiom later_time_car_clock : (t_car_later : ℕ) → t_car_later = (12 * 60 + 30)

-- Theorem to prove the correct actual time
theorem actual_time_is_5_22_pm (t_actual_later : ℕ) : 
  (car_clock_time t_actual_later) = later_time_car_clock -> t_actual_later = (17 * 60 + 22) :=
sorry

end actual_time_is_5_22_pm_l65_65985


namespace at_least_one_vowel_l65_65138

-- Define the set of letters
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'I'}

-- Define the vowels within the set of letters
def vowels : Finset Char := {'A', 'E', 'I'}

-- Define the consonants within the set of letters
def consonants : Finset Char := {'B', 'C', 'D', 'F'}

-- Function to count the total number of 3-letter words from a given set
def count_words (s : Finset Char) (length : Nat) : Nat :=
  s.card ^ length

-- Define the statement of the problem
theorem at_least_one_vowel : count_words letters 3 - count_words consonants 3 = 279 :=
by
  sorry

end at_least_one_vowel_l65_65138


namespace exists_y_square_divisible_by_five_btw_50_and_120_l65_65794

theorem exists_y_square_divisible_by_five_btw_50_and_120 : ∃ y : ℕ, (∃ k : ℕ, y = k^2) ∧ (y % 5 = 0) ∧ (50 ≤ y ∧ y ≤ 120) ∧ y = 100 :=
by
  sorry

end exists_y_square_divisible_by_five_btw_50_and_120_l65_65794


namespace value_g2_l65_65234

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (g (x - y)) = g x * g y - g x + g y - x^3 * y^3

theorem value_g2 : g 2 = 8 :=
by sorry

end value_g2_l65_65234


namespace find_f_prime_zero_l65_65243

noncomputable def differentiable_and_monotonic (f : ℝ → ℝ) : Prop :=
differentiable ℝ f ∧ (∀ x y, x ≤ y → f x ≤ f y ) 

theorem find_f_prime_zero
  (f : ℝ → ℝ)
  (λ : ℝ)
  (h_diff : differentiable_and_monotonic f)
  (h_f0 : f 0 = 0)
  (h_ineq : ∀ x, f (f x) ≥ (λ + 1) * f x - λ * x) :
  f' 0 = λ ∨ f' 0 = 1 :=
sorry

end find_f_prime_zero_l65_65243


namespace problem1_problem2_problem3_l65_65851

variables (a : ℝ) (f : ℝ → ℝ)
-- Define the function f(x) = e^x - ax
noncomputable def f (x : ℝ) := Real.exp x - a * x

-- Problem 1: Prove that if the tangent line of f(x) at x = 0 passes through the point (1, 0), then a = 2
theorem problem1 (h : tangentLineThroughPoint (f 0) (1, 0)) : a = 2 := sorry

-- Problem 2: Prove that if f(x) has no zeros on (-1, +∞), then a ∈ [-1/e, e]
theorem problem2 (h : ∀ x > -1, f x ≠ 0) : -1 / Real.exp 1 ≤ a ∧ a < Real.exp 1 := sorry

-- Problem 3: Prove that f(x) ≥ (1 + x) / (f(x) + x) for all x ∈ ℝ when a = 1
theorem problem3 (ha : a = 1) (x : ℝ) : f x ≥ (1 + x) / (f x + x) := sorry

end problem1_problem2_problem3_l65_65851


namespace evaluate_expression_l65_65049

theorem evaluate_expression : 
  (⌈(23 / 11) - ⌈31 / 19⌉⌉ / ⌈(35 / 9) + ⌈(9 * 19) / 35⌉⌉) = 1 / 9 := 
by sorry

end evaluate_expression_l65_65049


namespace race_time_l65_65172

theorem race_time 
    (v_A v_B t_A t_B : ℝ)
    (h1 : v_A = 1000 / t_A) 
    (h2 : v_B = 940 / t_A)
    (h3 : v_B = 1000 / (t_A + 15)) 
    (h4 : t_B = t_A + 15) :
    t_A = 235 := 
  by
    sorry

end race_time_l65_65172


namespace concyclic_points_BCQP_l65_65561

open EuclideanGeometry

noncomputable def isosceles_triangle (A B C : Point) : Prop := 
Isosceles ∧ (AB = AC > BC)

noncomputable def special_point_D (A B C D : Point) : Prop := 
D ∈ triangle_interior A B C ∧ DA = DB + DC

noncomputable def perpendicular_bisector_ab {A B : Point} : Line := 
mid_axis A B -- Assuming 'mid_axis' gives the perpendicular bisector

noncomputable def perpendicular_bisector_ac {A C : Point} : Line := 
mid_axis A C -- Assuming 'mid_axis' gives the perpendicular bisector

noncomputable def external_angle_bisector_adb (A B D : Point) : Line :=
angle_bisector (∠ADB) true -- Assuming 'angle_bisector' with true denotes external

noncomputable def external_angle_bisector_adc (A C D : Point) : Line :=
angle_bisector (∠ADC) true -- Assuming 'angle_bisector' with true denotes external

noncomputable def point_P (A B D : Point) : Point :=
(perpendicular_bisector_ab A B) ∩ (external_angle_bisector_adb A B D)

noncomputable def point_Q (A C D : Point) : Point :=
(perpendicular_bisector_ac A C) ∩ (external_angle_bisector_adc A C D)

theorem concyclic_points_BCQP (A B C D P Q : Point) 
(h_isosceles : isosceles_triangle A B C)
(h_special : special_point_D A B C D)
(hP : P = point_P A B D)
(hQ : Q = point_Q A C D) : cyclic B C P Q :=
sorry

end concyclic_points_BCQP_l65_65561


namespace measure_15_minutes_l65_65330

/-- Given a timer setup with a 7-minute hourglass and an 11-minute hourglass, show that we can measure exactly 15 minutes. -/
theorem measure_15_minutes (h7 : ∃ t : ℕ, t = 7) (h11 : ∃ t : ℕ, t = 11) : ∃ t : ℕ, t = 15 := 
  by 
    sorry

end measure_15_minutes_l65_65330


namespace ellipse_problem_solution_l65_65482

noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def line_through_focus (x y k : ℝ) : Prop :=
  y = k * (x - 1)

noncomputable def solve_for_intersections (a b k : ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ),
    ellipse_equation x1 y1 a b ∧
    ellipse_equation x2 y2 a b ∧
    line_through_focus x1 y1 k ∧
    line_through_focus x2 y2 k

/--
Given an ellipse passing through points (√2, 0) and (0, 1), with its right focus being F,
if a line passing through point F intersects the ellipse at points A and B such that 
$\overrightarrow{AF}=3\overrightarrow{FB}$,
then the value of $| \overrightarrow{OA}+ \overrightarrow{OB}|$ is $ \frac{2\sqrt{5}}{3}$.
-/
theorem ellipse_problem_solution :
  ∀ (a b k : ℝ),
    a^2 = 2 ∧ 
    b^2 = 1 ∧ 
    solve_for_intersections a b k →
    |(1/(2k^2 + 1) - 3/(2k^2 + 1) + 1, 2/(2k^2 + 1) - 1 + 3/(2k^2 + 1) + 1)| = (2 * sqrt 5) / 3 :=
begin
  sorry
end

end ellipse_problem_solution_l65_65482


namespace interval_monotonic_decrease_l65_65478

noncomputable def f (x : ℝ) : ℝ := Real.sin x - (1/2) * x

theorem interval_monotonic_decrease : 
  ∃ I : Set ℝ, I = Set.Ioo (Real.pi / 3) Real.pi ∧ 
  ∀ x ∈ I, f' x < 0 :=
by
  sorry

end interval_monotonic_decrease_l65_65478


namespace intersection_of_A_and_B_range_of_m_l65_65083

-- Problem 1: Intersection of A and B when m=5
theorem intersection_of_A_and_B (x : ℝ) :
  let A := { x | x ^ 2 - x - 12 ≤ 0 } in
  let B := { x | x ^ 2 - 6 * x + 5 ≤ 0 } in
  (x ∈ A ∧ x ∈ B) ↔ (1 ≤ x ∧ x ≤ 4) :=
sorry

-- Problem 2: Range of real numbers m
theorem range_of_m (m : ℝ) :
  let A := { x | x ^ 2 - x - 12 ≤ 0 } in
  let B := { x | x ^ 2 - (1 + m) * x + m ≤ 0 } in
  (∀ x, x ∈ A → x ∈ B) ∧ ¬(∀ x, x ∈ B → x ∈ A) ↔ -3 ≤ m ∧ m ≤ 4 :=
sorry

end intersection_of_A_and_B_range_of_m_l65_65083


namespace amber_pieces_correct_l65_65204

def jerry_sweeps_up (green_pieces amber_pieces clear_pieces total_pieces : ℕ) : Prop :=
  green_pieces = 35 ∧
  clear_pieces = 85 ∧
  green_pieces * 4 = total_pieces ∧
  amber_pieces = total_pieces - (green_pieces + clear_pieces)

theorem amber_pieces_correct : ∃ (amber_pieces : ℕ),
  jerry_sweeps_up 35 amber_pieces 85 (35 * 4) ∧ amber_pieces = 20 :=
by {
  use 20,
  simp [jerry_sweeps_up],
  sorry
}

end amber_pieces_correct_l65_65204


namespace number_of_zeros_of_f_l65_65520

-- Definitions for conditions
def increasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y

def f (b c : ℝ) (x : ℝ) : ℝ := x^3 + b*x + c

-- Conditions
variable (b c : ℝ)
variable h1 : increasing_function (f b c)
variable h2 : -1 ≤ 1
variable h3 : (f b c (1/2)) * (f b c (-1/2)) < 0

-- Prove the number of zeros of f is 1
theorem number_of_zeros_of_f : ∃! x : ℝ, f b c x = 0 :=
sorry  -- Proof goes here

end number_of_zeros_of_f_l65_65520


namespace compare_M_N_l65_65929

variables (a : ℝ)

-- Definitions based on given conditions
def M : ℝ := 2 * a * (a - 2) + 3
def N : ℝ := (a - 1) * (a - 3)

theorem compare_M_N : M a ≥ N a := 
by {
  sorry
}

end compare_M_N_l65_65929


namespace locus_of_points_l65_65371

-- Defining the setup for the problem
variables (P Q R A B C D E F S S₀ : Point)
variables (PQ QR RP : ℝ)
variables (area : Triangle → ℝ)
variables (SAB SCD SEF : Triangle)

-- Conditions given in the problem
variable h : (AB_length / PQ) = (CD_length / QR) = (EF_length / RP)

-- Defining the triangles involved
def triangle_PQR := Triangle P Q R
def triangle_S₀AB := Triangle S₀ A B
def triangle_S₀CD := Triangle S₀ C D
def triangle_S₀EF := Triangle S₀ E F
def triangle_SAB := Triangle S A B
def triangle_SCD := Triangle S C D
def triangle_SEF := Triangle S E F

-- Statement to be proven
theorem locus_of_points (S : Point) :
  (area (triangle_SAB) + area (triangle_SCD) + area (triangle_SEF) = area (triangle_S₀AB) + area (triangle_S₀CD) + area (triangle_S₀EF)) →
  ( (∃ L : Line, ∀ S, S ∈ L ∧ L ∥ segment D' E' ∧ L passes_through S₀) ∨ (S ∈ triangle_PQR)) :=
sorry

end locus_of_points_l65_65371


namespace find_f_of_13_l65_65715

def f : ℤ → ℤ := sorry  -- We define f as a function from integers to integers

theorem find_f_of_13 : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x k : ℤ, f (x + 4 * k) = f x) ∧ 
  (f (-1) = 2) → 
  f 13 = -2 := 
by 
  sorry

end find_f_of_13_l65_65715


namespace total_amount_shared_l65_65262

theorem total_amount_shared 
  (Parker_share : ℤ)
  (ratio_2 : ℤ)
  (ratio_3 : ℤ)
  (total_parts : ℤ)
  (part_value : ℤ)
  (total_amount : ℤ) :
  Parker_share = 50 →
  ratio_2 = 2 →
  ratio_3 = 3 →
  total_parts = ratio_2 + ratio_3 →
  part_value = Parker_share / ratio_2 →
  total_amount = total_parts * part_value →
  total_amount = 125 :=
by 
  intros hParker_share hratio_2 hratio_3 htotal_parts hpart_value htotal_amount
  rw [hParker_share, hratio_2, hratio_3, htotal_parts, hpart_value, htotal_amount]
  sorry

end total_amount_shared_l65_65262


namespace add_sub_time_complexity_mul_div_time_complexity_l65_65633

-- Assuming necessary imports for complexity and logarithm definitions
noncomputable theory

open Complex

-- Definitions for integers and time complexity
variables (m n : ℕ)

def time_complexity_add_sub (m n : ℕ) : BigO := BigO (log m + log n)
def time_complexity_mul_div (m n : ℕ) : BigO := BigO (log m * log n)

-- Theorem statements based on the problem conditions and answers
theorem add_sub_time_complexity (h : m ≤ n) : time_complexity_add_sub m n = BigO (log m + log n) :=
by sorry

theorem mul_div_time_complexity (h : m ≤ n) : time_complexity_mul_div m n = BigO (log m * log n) :=
by sorry

end add_sub_time_complexity_mul_div_time_complexity_l65_65633


namespace rationalize_cube_root_sum_l65_65634

theorem rationalize_cube_root_sum :
  let a := real.cbrt 4
  let b := real.cbrt 3
  let denom := a - b
  let numer := real.cbrt 16 + real.cbrt 12 + real.cbrt 9
  denom * numer = 1
  ∧ (16 + 12 + 9 + 1 = 38) :=
by
  sorry

end rationalize_cube_root_sum_l65_65634


namespace part1_real_roots_part2_root_greater_than_6_l65_65074

theorem part1_real_roots (m : ℝ) : 
  let a := 1
  let b := -(m - 1)
  let c := m - 2
  let discriminant := b^2 - 4 * a * c in
  discriminant ≥ 0 :=
by 
  sorry

theorem part2_root_greater_than_6 (m : ℝ) : 
  (∃ x : ℝ, x > 6 ∧ (x^2 - (m - 1) * x + (m - 2) = 0)) → m > 8 :=
by 
  sorry

end part1_real_roots_part2_root_greater_than_6_l65_65074


namespace vitamin_c_total_l65_65625

theorem vitamin_c_total :
  ∀ (A O : ℕ), 
    (103 = A) ∧ (2 * 103 + 3 * O = 452) → (A + O = 185) :=
by
  intros A O h 
  cases h with hA h2
  rw hA 
  rw hA at h2
  sorry

end vitamin_c_total_l65_65625


namespace abs_eq_cases_l65_65147

theorem abs_eq_cases (x : ℝ) (h : x < 2) : 
  ( -2 ≤ x ∧ x < 2 → |x - 2| + |2 + x| = 4 ) ∧ 
  ( x < -2 → |x - 2| + |2 + x| = -2 * x ) :=
begin
  split,
  {
    intro h1,
    cases h1 with h2 h3,
    rw [abs_of_neg (sub_neg_of_lt h3), abs_of_nonneg (add_nonneg_of_le_of_nonneg (le_of_lt h3) (le_of_lt h))],
    linarith,
  },
  {
    intro h4,
    rw [abs_of_neg (sub_neg_of_lt h4), abs_of_neg (add_neg_of_neg_of_nonpos h4 (by linarith))],
    linarith,
  }
end

end abs_eq_cases_l65_65147


namespace find_cos_alpha_beta_l65_65472

noncomputable def trig_cos_alpha_beta (α β : ℝ) : Prop :=
  0 < α ∧ α < π ∧
  0 < β ∧ β < (π / 2) ∧
  cos (α - (β / 2)) = - (1 / 9) ∧ 
  sin ((α / 2) - β) = 2 / 3

theorem find_cos_alpha_beta (α β : ℝ) (h : trig_cos_alpha_beta α β) :
  cos (α + β) = - (239 / 729) :=
sorry

end find_cos_alpha_beta_l65_65472


namespace hyperbola_eccentricity_l65_65519

-- Define hyperbola and circle conditions
def hyperbola_asymptote_tangent_to_circle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∀ (x y : ℝ), 
    (x + a)^2 + y^2 = (1/4) * a^2 →
    (b * x + a * y = 0 ∨ b * x - a * y = 0)

-- Define eccentricity calculation
def eccentricity (a b : ℝ) : ℝ := (a^2 + b^2).sqrt / a

-- The theorem to prove
theorem hyperbola_eccentricity 
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : hyperbola_asymptote_tangent_to_circle a b ha hb) : 
  eccentricity a b = 2 * (3).sqrt / 3 :=
sorry

end hyperbola_eccentricity_l65_65519


namespace sum_of_digits_of_likable_numbers_l65_65001

theorem sum_of_digits_of_likable_numbers (n : ℕ) : n = 145800 :=
  let digits := {1, 2, 4, 5, 7, 8}
  let num_count := 5 * 6^4
  let sum_of_digits := (1 + 2 + 4 + 5 + 7 + 8) * (num_count / 6) * 5
  n = sum_of_digits
  sorry

end sum_of_digits_of_likable_numbers_l65_65001


namespace factorize_expression_l65_65441

theorem factorize_expression (a b x y : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a + 2 * b) * (3 * a - 2 * b) :=
by
  sorry

end factorize_expression_l65_65441


namespace find_w_value_l65_65825

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem find_w_value
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : sqrt x / sqrt y - sqrt y / sqrt x = 7 / 12)
  (h2 : x - y = 7) :
  x + y = 25 := 
by
  sorry

end find_w_value_l65_65825


namespace math_problem_l65_65503

variables {a : ℕ → ℤ} {d : ℤ}

-- Condition: The sequence {a_n} is an arithmetic progression with a non-zero common difference d.
def arithmetic_progression (a : ℕ → ℤ) (d : ℤ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n + d

-- Condition: S_n represents the sum of the first n terms.
def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  nat.foldl (+) 0 (list.range n).map a

-- Condition: a_3, a_4, and a_8 form a geometric progression.
def geometric_progression (a3 a4 a8 : ℤ) : Prop :=
  a4 ^ 2 = a3 * a8

theorem math_problem (h_ap : arithmetic_progression a d) (h_gp : geometric_progression (a 3) (a 4) (a 8)) (h_d_nonzero : d ≠ 0) :
  (a 0 * d < 0) ∧ (d * (S a 4) < 0) ∧ ((a 4 / a 3) = 4) ∧ ¬((S a 8) = (-20 * (S a 4))) :=
sorry

end math_problem_l65_65503


namespace max_profit_price_l65_65352

def profit (x : ℝ) : ℝ := (10 + x) * (400 - 20 * x)

theorem max_profit_price : 
  (∃ x, ∀ y, profit y ≤ profit x) ∧ (90 + x = 95) :=
begin
  let x := 5,
  use x,
  intros y,
  have h1 : profit y = -20 * y ^ 2 + 200 * y + 4000,
  {
    unfold profit,
    ring,
  },
  have h2 : profit x = -20 * x ^ 2 + 200 * x + 4000,
  {
    unfold profit,
    ring,
  },
  have h3 : -20 * (y - 5) ^ 2 ≤ 0,
  {
    nlinarith,
  },
  rw [h1, h2],
  nlinarith,
end

end max_profit_price_l65_65352


namespace tile_ratio_l65_65557

-- Definitions corresponding to the conditions in the problem
def orig_grid_size : ℕ := 6
def orig_black_tiles : ℕ := 12
def orig_white_tiles : ℕ := 24
def border_size : ℕ := 1

-- The combined problem statement
theorem tile_ratio (orig_grid_size orig_black_tiles orig_white_tiles border_size : ℕ) :
  let new_grid_size := orig_grid_size + 2 * border_size
  let new_tiles := new_grid_size^2
  let added_tiles := new_tiles - orig_grid_size^2
  let total_white_tiles := orig_white_tiles + added_tiles
  let black_to_white_ratio := orig_black_tiles / total_white_tiles
  black_to_white_ratio = (3 : ℕ) / 13 :=
by {
  sorry
}

end tile_ratio_l65_65557


namespace ellipse_properties_l65_65815

open Real

noncomputable def ellipse_eq (a b : ℝ) : Prop :=
  ∃ e : ℝ, e = 1 / 2 ∧ a = 4 ∧ a^2 = b^2 + (a * e)^2 ∧
  (b^2 = 12 ∧ ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def max_area_ABPQ (P Q : ℝ × ℝ) (ell_eq : Prop) : ℝ :=
  if P = (2, 3) ∧ Q = (2, -3) then
    let A B : ℝ × ℝ := sorry -- assume A and B exist with the given conditions
    let t : ℝ := 0 in
    3 * Real.sqrt (48 - 3 * t^2)
  else 0

noncomputable def slope_AB_constant (A B P Q : ℝ × ℝ) (slope_AB : ℝ) : Prop :=
  ∃ k : ℝ, A = sorry ∧ B = sorry ∧ P = (2, 3) ∧ Q = (2, -3) ∧ slope_AB = 1 / 2

theorem ellipse_properties :
  (∃ a b : ℝ, ellipse_eq a b) ∧
  (ellipse_eq 4 2 → max_area_ABPQ (2, 3) (2, -3) true = 12 * Real.sqrt 3) ∧
  (ellipse_eq 4 2 → ∀ A B P Q, slope_AB_constant A B P Q (1 / 2)) := sorry

end ellipse_properties_l65_65815


namespace arrangement_count_of_1_to_6_divisible_by_7_l65_65898

theorem arrangement_count_of_1_to_6_divisible_by_7 :
  {s : List ℕ // s.perm [1, 2, 3, 4, 5, 6] ∧ ∀ a b c, List.pairs s (List.pairs s.tail s.tail^.tail = a :: b :: c :: _) → (a * c - b^2) % 7 = 0 } → 12 :=
sorry

end arrangement_count_of_1_to_6_divisible_by_7_l65_65898


namespace range_of_dot_product_theorem_l65_65862

noncomputable section

variables (a b : ℝ^2)

def magnitude (v : ℝ^2) : ℝ := real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def magnitude_sq (v : ℝ^2) : ℝ := v.1 ^ 2 + v.2 ^ 2

def range_of_dot_product (a b : ℝ^2) : Prop := 
  let dot_product := a.1 * b.1 + a.2 * b.2 in
  -14 ≤ dot_product ∧ dot_product ≤ 34

theorem range_of_dot_product_theorem
  (ha : 2 ≤ magnitude a ∧ magnitude a ≤ 6) 
  (hb : 2 ≤ magnitude b ∧ magnitude b ≤ 6) 
  (hab_minus : 2 ≤ magnitude (a - b) ∧ magnitude (a - b) ≤ 6)
  : range_of_dot_product a b :=
sorry

end range_of_dot_product_theorem_l65_65862


namespace Petya_chips_l65_65454

theorem Petya_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) :
  ∃ T : ℕ, T = 24 :=
by {
  let T_triangle := 3 * x - 3,
  let T_square := 4 * y - 4,
  -- The conditions ensure T_triangle = T_square
  have h3 : T_triangle = T_square, from h2,
  -- substituting y = x - 2 into T_square
  have h4 : T_square = 4 * (x - 2) - 4, from calc
    T_square = 4 * y - 4 : by rfl
    ... = 4 * (x - 2) - 4 : by rw h1,
  -- simplify to find x,
  have h5 : 3 * x - 3 = 4 * (x - 2) - 4, from h2,
  have h6 : 3 * x - 3 = 4 * x - 8 - 4, from h5,
  have h7 : 3 * x - 3 = 4 * x - 12, from by simp at h6,
  have h8 : -3 = x - 12, from by linarith,
  have h9 : x = 9, from by linarith,
  -- Find the total number of chips
  let T := 3 * x - 3,
  have h10 : T = 24, from calc
    T = 3 * 9 - 3 : by rw h9
    ... = 24 : by simp,
  exact ⟨24, h10⟩
}

end Petya_chips_l65_65454


namespace moles_of_CH4_combined_l65_65798

theorem moles_of_CH4_combined
  (moles_CH4 : ℝ)
  (moles_Br2 : ℝ)
  (moles_HBr : ℝ)
  (H_eq : moles_HBr = 1)
  (reaction_eq : ∀ (moles_CH4 moles_Br2 : ℝ), CH4 + Br2 → CH3Br + HBr → (moles_CH4 = moles_Br2))
  : moles_CH4 = 1 :=
sorry

end moles_of_CH4_combined_l65_65798


namespace sp_contains_at_least_three_elements_l65_65824

theorem sp_contains_at_least_three_elements (p : ℕ) (h_prime : Prime p) (h_ge_5 : p ≥ 5) :
  ∃ n1 n2 n3, n1 ∈ Z+ ∧ n2 ∈ Z+ ∧ n3 ∈ Z+ ∧
    n1 < n2 ∧ n2 < n3 ∧
    n1 ∈ S_p p ∧ n2 ∈ S_p p ∧ n3 ∈ S_p p :=
by {
  -- Definitions and conditions
  let S_p := λ (p : ℕ), {n : ℕ | n > 0 ∧ (p ∣ (a_n n))},
  -- Proof goal
  sorry
}

end sp_contains_at_least_three_elements_l65_65824


namespace sandy_has_4_times_more_marbles_l65_65917

open Nat

-- Define the context
def Jessica_marbles : ℕ := 3 * 12
def Sandy_marbles : ℕ := 144

-- State the problem
theorem sandy_has_4_times_more_marbles (jessica_marbles_eq : Jessica_marbles = 3 * 12)
  (sandy_marbles_eq : Sandy_marbles = 144) : Sandy_marbles / Jessica_marbles = 4 :=
by
  rw [← jessica_marbles_eq, ← sandy_marbles_eq]
  sorry

end sandy_has_4_times_more_marbles_l65_65917


namespace geometric_progression_fourth_term_l65_65546

theorem geometric_progression_fourth_term :
  let a1 := 2^(1/2)
  let a2 := 2^(1/4)
  let a3 := 2^(1/8)
  a4 = 2^(1/16) :=
by
  sorry

end geometric_progression_fourth_term_l65_65546


namespace functions_increasing_in_interval_l65_65855

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x

theorem functions_increasing_in_interval :
  ∀ x, -Real.pi / 4 < x → x < Real.pi / 4 →
  (f x < f (x + 1e-6)) ∧ (g x < g (x + 1e-6)) :=
sorry

end functions_increasing_in_interval_l65_65855


namespace valid_n_values_count_l65_65071

open Nat

def f (n : ℕ) : ℕ := 6 + 7 * n + 4 * n^2 + 2 * n^3 + 5 * n^4 + 3 * n^5

theorem valid_n_values_count : (finset.filter (fun n => f n % 11 = 0) (finset.Icc 2 100)).card = 7 :=
by
  sorry

end valid_n_values_count_l65_65071


namespace longest_side_proof_l65_65710

noncomputable def longest_side_of_triangle (a b c : ℕ) : ℕ :=
if h : a ≥ b ∧ a ≥ c then a
else if h : b ≥ a ∧ b ≥ c then b
else c

theorem longest_side_proof (x : ℕ) :
  let a := 6 * x,
      b := 4 * x,
      c := 3 * x,
      P := 104 in
  a + b + c = P → x = 8 → longest_side_of_triangle a b c = 48 := 
by 
  intros,
  subst h_1,
  simp [a, b, c, longest_side_of_triangle],
  split_ifs,
  all_goals { refl },
  sorry

end longest_side_proof_l65_65710


namespace sequence_x_value_l65_65893

theorem sequence_x_value (p q r x : ℕ) 
  (h1 : 13 = 5 + p + q) 
  (h2 : r = p + q + 13) 
  (h3 : x = 13 + r + 40) : 
  x = 74 := 
by 
  sorry

end sequence_x_value_l65_65893


namespace probability_is_real_l65_65638

-- Define the intervals and conditions
def interval := {x : ℚ // 0 ≤ x ∧ x < 2 ∧ ∃ n d : ℤ, n / d = x ∧ 1 ≤ d ∧ d ≤ 5}

-- Define the set of rational numbers within the specified interval for a and b
def valid_rat (a : interval) : Prop := 0 ≤ a.val ∧ a.val < 2 ∧ (∃ n d : ℤ, n / d = a.val ∧ 1 ≤ d ∧ d ≤ 5)

-- Define the property that the expression (cos (a * π) + i * sin (b * π))^4 is real
def is_real (a b : interval) : Prop :=
  let x := Real.cos (a.val * Real.pi)
  let y := Real.sin (b.val * Real.pi)
  let z := Complex.mk x y
  (z^4).im = 0

-- The main theorem to prove
theorem probability_is_real :
  (∑ a in interval, ∑ b in interval, if is_real a b then 1 else 0 : ℚ) / (↑(interval.card) * ↑(interval.card)) = 6 / 25 := sorry

end probability_is_real_l65_65638


namespace number_of_valid_arrangements_l65_65904

def is_valid_triplet (a b c : ℕ) : Prop :=
  (a * c - b ^ 2) % 7 = 0

def is_valid_sequence (s : List ℕ) : Prop :=
  ∀ (i : ℕ), i + 2 < s.length → is_valid_triplet (s.get i) (s.get (i + 1)) (s.get (i + 2))

theorem number_of_valid_arrangements : 
  (List.permutations [1, 2, 3, 4, 5, 6]).countp is_valid_sequence = 12 := 
sorry

end number_of_valid_arrangements_l65_65904


namespace inequality_proof_l65_65942

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom abc_eq_one : a * b * c = 1

theorem inequality_proof :
  (1 + a * b) / (1 + a) + (1 + b * c) / (1 + b) + (1 + c * a) / (1 + c) ≥ 3 :=
by
  sorry

end inequality_proof_l65_65942


namespace sum_of_coeffs_correct_l65_65499

-- Define the condition that the coefficient of second term in the expansion of (x + 2y)^n is 8.
def coeff_second_term (n : ℕ) : Prop :=
  2 * n = 8

-- Define the sum of the coefficients of all terms in the expansion of (1 + x) + (1 + x)^2 + ... + (1 + x)^n with x = 1.
def sum_of_coeffs (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).sum (λ k, (2 ^ k))

-- State the theorem we aim to prove.
theorem sum_of_coeffs_correct (n : ℕ) (h : coeff_second_term n) : sum_of_coeffs n = 30 :=
by
  -- In actual proof, Lean would require steps of proof here.
  sorry

end sum_of_coeffs_correct_l65_65499


namespace exp_inequality_iff_l65_65491

theorem exp_inequality_iff (a b : ℝ) : a > b ↔ 2^a > 2^b := 
by {
  sorry
}

end exp_inequality_iff_l65_65491


namespace sum_of_cubes_1812_l65_65635

theorem sum_of_cubes_1812 :
  ∃ a b c d : ℤ, a = 303 ∧ b = 301 ∧ c = -302 ∧ d = -302 ∧ a^3 + b^3 + c^3 + d^3 = 1812 :=
by
  use 303, 301, -302, -302
  simp
  sorry

end sum_of_cubes_1812_l65_65635


namespace six_digit_number_divisible_by_504_l65_65451

theorem six_digit_number_divisible_by_504 : 
  ∃ a b c : ℕ, (523 * 1000 + 100 * a + 10 * b + c) % 504 = 0 := by 
sorry

end six_digit_number_divisible_by_504_l65_65451


namespace factorial_expression_identity_l65_65028

open Nat

theorem factorial_expression_identity : 7! - 6 * 6! - 7! = 0 := by
  sorry

end factorial_expression_identity_l65_65028


namespace extreme_value_of_f_range_of_a_l65_65854

noncomputable def f (x a : ℝ) := Real.exp x - a * x - 1

theorem extreme_value_of_f (a : ℝ) (ha : 0 < a) : ∃ x, f x a = a - a * Real.log a - 1 :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ 2 ∧ 0 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 a = f x2 a ∧ abs (x1 - x2) ≥ 1 ) →
  (e - 1 < a ∧ a < Real.exp 2 - Real.exp 1) :=
sorry

end extreme_value_of_f_range_of_a_l65_65854


namespace smallest_n_l65_65979

noncomputable def count_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 + n / 15625 + n / 78125 + n / 390625

theorem smallest_n (a b c : ℕ) (m n : ℕ) (h : a + b + c = 2010)
  (hc : c = 710) (hmn : a.factorial * b.factorial * c.factorial = m * 10^n) (hmdiv : ¬ 10 ∣ m) :
  n = 500 :=
by
  -- Placeholder for the proof
  sorry

end smallest_n_l65_65979


namespace fill_in_blank_with_warning_l65_65792

-- Definitions corresponding to conditions
def is_noun (word : String) : Prop :=
  -- definition of being a noun
  sorry

def corresponds_to_chinese_hint (word : String) (hint : String) : Prop :=
  -- definition of corresponding to a Chinese hint
  sorry

-- The theorem we want to prove
theorem fill_in_blank_with_warning : ∀ word : String, 
  (is_noun word ∧ corresponds_to_chinese_hint word "警告") → word = "warning" :=
by {
  sorry
}

end fill_in_blank_with_warning_l65_65792


namespace jean_calls_on_monday_l65_65203

theorem jean_calls_on_monday :
  ∀ (calls_tue calls_wed calls_thu calls_fri avg_calls_per_day total_days: ℕ),
  calls_tue = 46 →
  calls_wed = 27 →
  calls_thu = 61 →
  calls_fri = 31 →
  avg_calls_per_day = 40 →
  total_days = 5 →
  let total_calls_week := avg_calls_per_day * total_days in
  let total_calls_tue_to_fri := calls_tue + calls_wed + calls_thu + calls_fri in
  let calls_mon := total_calls_week - total_calls_tue_to_fri in
  calls_mon = 35 :=
by
  intros calls_tue calls_wed calls_thu calls_fri avg_calls_per_day total_days
         h_calls_tue h_calls_wed h_calls_thu h_calls_fri h_avg_calls_per_day h_total_days
  simp only [h_calls_tue, h_calls_wed, h_calls_thu, h_calls_fri, h_avg_calls_per_day, h_total_days]
  let total_calls_week := avg_calls_per_day * total_days
  let total_calls_tue_to_fri := calls_tue + calls_wed + calls_thu + calls_fri
  let calls_mon := total_calls_week - total_calls_tue_to_fri
  exact rfl

end jean_calls_on_monday_l65_65203


namespace constant_term_in_expansion_l65_65986

theorem constant_term_in_expansion :
  let expr := (x + 1) * (2 * x^2 - 1 / x)^6
  term := some (n : Int) in
  term == 60 :=
by
  sorry

end constant_term_in_expansion_l65_65986


namespace number_of_scores_4_is_two_l65_65080

theorem number_of_scores_4_is_two
  (x1 x2 x3 x4 : ℕ)
  (h1 : x1 * x2 * x3 * x4 = 72)
  (h2 : x1 ≤ 10 ∧ x2 ≤ 10 ∧ x3 ≤ 10 ∧ x4 ≤ 10)
  (h3 : ∃ s : ℕ, s = x1 + x2 + x3 + x4 ∧ (s - 1) + (s - 2) + (s - 3) = 3 * s - 6) :
  ({x1, x2, x3, x4}.count 4 = 2) :=
sorry

end number_of_scores_4_is_two_l65_65080


namespace alternate_interior_angles_imply_parallel_l65_65697

theorem alternate_interior_angles_imply_parallel (l1 l2 : Line) (t : Transversal) 
  (h : alternate_interior_angles_equal l1 l2 t) : are_parallel l1 l2 :=
sorry

end alternate_interior_angles_imply_parallel_l65_65697


namespace return_amount_is_correct_l65_65213

-- Define the borrowed amount and the interest rate
def borrowed_amount : ℝ := 100
def interest_rate : ℝ := 10 / 100

-- Define the condition of the increased amount
def increased_amount : ℝ := borrowed_amount * interest_rate

-- Define the total amount to be returned
def total_amount : ℝ := borrowed_amount + increased_amount

-- Lean 4 statement to prove
theorem return_amount_is_correct : total_amount = 110 := by
  -- Borrowing amount definition
  have h1 : borrowed_amount = 100 := rfl
  -- Interest rate definition
  have h2 : interest_rate = 10 / 100 := rfl
  -- Increased amount calculation
  have h3 : increased_amount = borrowed_amount * interest_rate := rfl
  -- Expanded calculation of increased_amount
  have h4 : increased_amount = 100 * (10 / 100) := by rw [h1, h2]
  -- Simplify the increased_amount
  have h5 : increased_amount = 10 := by norm_num [h4]
  -- Total amount calculation
  have h6 : total_amount = borrowed_amount + increased_amount := rfl
  -- Expanded calculation of total_amount
  have h7 : total_amount = 100 + 10 := by rw [h1, h5]
  -- Simplify the total_amount
  show 100 + 10 = 110 from rfl
  sorry

end return_amount_is_correct_l65_65213


namespace sum_of_divisors_330_l65_65687

theorem sum_of_divisors_330 : 
  (∑ k in (Finset.filter (λ d, 330 % d = 0) (Finset.range (331))), k) = 864 :=
by
  have prime_factorization : ∃ (a b c d : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 11 ∧ 330 = a * b * c * d :=
    ⟨2, 3, 5, 11, rfl, rfl, rfl, rfl, rfl⟩
  sorry

end sum_of_divisors_330_l65_65687


namespace sum_of_primitive_roots_mod_11_l65_65042

theorem sum_of_primitive_roots_mod_11 :
  ∑ i in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.filter (λ a => (∀ b : ℕ, 0 < b ∧ b < 11 → (a ^ b % 11 ≠ 1))) = 23 :=
by
  sorry

end sum_of_primitive_roots_mod_11_l65_65042


namespace range_g_l65_65934

theorem range_g (f : ℝ → ℝ) (g : ℝ → ℝ) (b : ℝ)
  (h1 : ∀ x, f x = 3 ^ x + b)
  (h2 : b < -1) :
  set.range g = set.Ioo 0 (2 / 9) :=
sorry

end range_g_l65_65934


namespace line_through_B_l65_65301

open EuclideanGeometry

variables (A B M1 M2 : Point) 
variables (S1 S2 : Circle) 
variable (P : rotational_homothety)

-- Given conditions
variable (h1 : S1.intersects S2)  -- S1 and S2 intersect at points A and B
variable (h2 : P.center = A)        -- P is a homothety centered at A
variable (h3 : P.maps_to S1 S2)     -- P maps circle S1 to circle S2
variable (h4 : P.maps_to_point M1 = M2)  -- P maps point M1 on S1 to M2 on S2

-- Proof problem: Prove that line M1 M2 passes through point B
theorem line_through_B :
  (line_through (line M1 M2) B) :=
sorry

end line_through_B_l65_65301


namespace line_mn_fixed_point_max_area_triangle_fmn_l65_65842

-- Define the conditions for the ellipse and points
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1
def focus : ℝ × ℝ := (1, 0)
def is_midpoint (p a b : ℝ × ℝ) : Prop := p.1 = (a.1 + b.1) / 2 ∧ p.2 = (a.2 + b.2) / 2
def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Proof Problem 1: Line MN passes through fixed point
theorem line_mn_fixed_point (A B C D M N : ℝ × ℝ)
  (h_ellipse_A : ellipse A.1 A.2) (h_ellipse_B : ellipse B.1 B.2)
  (h_ellipse_C : ellipse C.1 C.2) (h_ellipse_D : ellipse D.1 D.2)
  (h_focus : focus = (1, 0))
  (h_midpoint_M : is_midpoint M A B) (h_midpoint_N : is_midpoint N C D)
  (h_perpendicular : is_perpendicular ((B.2 - A.2) / (B.1 - A.1)) ((D.2 - C.2) / (D.1 - C.1))) :
  ∃ E : ℝ × ℝ, E = (3/5, 0) ∧ line_through_MN M N E := sorry

-- Proof Problem 2: Maximum area of triangle FMN is achieved and is 4/25
theorem max_area_triangle_fmn (M N F : ℝ × ℝ)
  (h_ellipse_M : ellipse M.1 M.2) (h_ellipse_N : ellipse N.1 N.2)
  (h_focus : F = (1, 0))
  (exists_slope_AB : ∃ m : ℝ, m = (M.2 - F.2) / (M.1 - F.1))
  (exists_slope_CD : ∃ m : ℝ, m = (N.2 - F.2) / (N.1 - F.1))
  (h_perpendicular : is_perpendicular ((M.2 - F.2) / (M.1 - F.1)) ((N.2 - F.2) / (N.1 - F.1))) :
  ∃ S : ℝ, S = 4 / 25 ∧ max_area_FMN F M N S := sorry

-- Definitions or theorems for line_through_MN and max_area_FMN 
-- should be provided as required.


end line_mn_fixed_point_max_area_triangle_fmn_l65_65842


namespace John_and_Rose_work_together_l65_65205

theorem John_and_Rose_work_together (John_days : ℕ) (Rose_days : ℕ)
  (H1 : John_days = 320) (H2 : Rose_days = 480) : 
  (John_days * Rose_days) / (John_days + Rose_days) = 192 := by
  unfold John_days
  unfold Rose_days
  rw [H1, H2]
  sorry

end John_and_Rose_work_together_l65_65205


namespace length_of_diagonal_l65_65076

/--
Given a quadrilateral with sides 9, 11, 17, and 13,
the number of different whole numbers that could be the length of the diagonal
represented by the dashed line is 15.
-/
theorem length_of_diagonal (x : ℕ) :
  (∀ x, 5 ≤ x ∧ x ≤ 19 → x ∈ {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}) →
  card {x | 5 ≤ x ∧ x ≤ 19} = 15 :=
by
  sorry

end length_of_diagonal_l65_65076


namespace interior_triangle_area_l65_65840

theorem interior_triangle_area (a1 a2 a3 a4 : ℕ) (h1 : a1 = 36) (h2 : a2 = 64) (h3 : a3 = 100) (h4 : a4 = 121) : 
  let side1 := Nat.sqrt a1,
      side2 := Nat.sqrt a2,
      side3 := Nat.sqrt a3 in
  let area := (side1 * side2) / 2 in
  area = 24 := by
  sorry

end interior_triangle_area_l65_65840


namespace min_value_x1_x2_frac1_x1x2_l65_65521

theorem min_value_x1_x2_frac1_x1x2 (a x1 x2 : ℝ) (ha : a > 2) (h_sum : x1 + x2 = a) (h_prod : x1 * x2 = a - 2) :
  x1 + x2 + 1 / (x1 * x2) ≥ 4 :=
sorry

end min_value_x1_x2_frac1_x1x2_l65_65521


namespace sequence_properties_l65_65480

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = a n - (1/3) * (a n)^2

theorem sequence_properties (a : ℕ → ℝ) (h : sequence a) :
  (∀ n : ℕ, n > 0 → a (n + 1) < a n) ∧  -- A: monotonically decreasing
  (∀ n : ℕ, n > 0 → a n < 2 * a (n + 1)) ∧  -- B: a_n < 2a_{n+1}
  5/2 < 100 * (a 100) ∧ 100 * (a 100) < 3 -- D: 5/2 < 100a_{100} < 3
  :=
sorry

end sequence_properties_l65_65480


namespace investment_ratio_l65_65876

theorem investment_ratio (a_profit b_profit : ℕ) (same_period_rate : Prop) 
  (h_a_profit : a_profit = 60000) (h_b_profit : b_profit = 6000) 
  (h_same_period_rate : same_period_rate):
  a_profit / b_profit = 10 := 
by
  have gcd_60000_6000 : Int.gcd 60000 6000 = 6000 := sorry
  rw [h_a_profit, h_b_profit]
  exact sorry

end investment_ratio_l65_65876


namespace sqrt_difference_approximation_l65_65696

theorem sqrt_difference_approximation : abs (sqrt 144 - sqrt 140 - 0.17) < 0.02 := 
sorry

end sqrt_difference_approximation_l65_65696


namespace fraction_of_sy_not_declared_major_l65_65949

-- Conditions
variables (T : ℝ) -- Total number of students
variables (first_year : ℝ) -- Fraction of first-year students
variables (second_year : ℝ) -- Fraction of second-year students
variables (decl_fy_major : ℝ) -- Fraction of first-year students who have declared a major
variables (decl_sy_major : ℝ) -- Fraction of second-year students who have declared a major

-- Definitions from conditions
def fraction_first_year_students := 1 / 2
def fraction_second_year_students := 1 / 2
def fraction_fy_declared_major := 1 / 5
def fraction_sy_declared_major := 4 * fraction_fy_declared_major

-- Hollow statement
theorem fraction_of_sy_not_declared_major :
  first_year = fraction_first_year_students →
  second_year = fraction_second_year_students →
  decl_fy_major = fraction_fy_declared_major →
  decl_sy_major = fraction_sy_declared_major →
  (1 - decl_sy_major) * second_year = 1 / 10 :=
by
  sorry

end fraction_of_sy_not_declared_major_l65_65949


namespace range_of_fraction_l65_65577

def is_differentiable (f : ℝ → ℝ) : Prop := ∀ x, ∃ df_dx, is_deriv f df_dx x

variables {a b c : ℝ}

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x^3 + (1 / 2) * a * x^2 + 2 * b * x + c

noncomputable def f' (x : ℝ) : ℝ :=
  x^2 + a * x + 2 * b

axiom max_in_interval : ∃ (x : ℝ), (0 < x ∧ x < 1 ∧ ∀ y, f(y) ≤ f(x))

axiom min_in_interval : ∃ (x : ℝ), (1 < x ∧ x < 2 ∧ ∀ y, f(y) ≥ f(x))

theorem range_of_fraction : (∃ (b : ℝ) (a : ℝ), 0 < b ∧ b < 1 ∧ -3 < a ∧ a < -1) → 
  (∀ b a, 0 < b → b < 1 → -3 < a → a < -1 → 
  (1 / 4 < (b - 2) / (a - 1) ∧ (b - 2) / (a - 1) < 1)) :=
by sorry

end range_of_fraction_l65_65577


namespace find_y_l65_65010

variable (y : ℕ)

-- Definitions based on conditions:
def vertices := (1, y) ∧ (9, y) ∧ (1, 5) ∧ (9, 5)
def area_64 := 8 * (y - 5) = 64
def y_positive := 0 < y

-- The statement to be proven:
theorem find_y (h1 : vertices) (h2 : area_64) (h3 : y_positive) : y = 13 := 
sorry

end find_y_l65_65010


namespace measure_15_minutes_l65_65329

/-- Given a timer setup with a 7-minute hourglass and an 11-minute hourglass, show that we can measure exactly 15 minutes. -/
theorem measure_15_minutes (h7 : ∃ t : ℕ, t = 7) (h11 : ∃ t : ℕ, t = 11) : ∃ t : ℕ, t = 15 := 
  by 
    sorry

end measure_15_minutes_l65_65329


namespace tangent_line_circle_l65_65562

-- Definitions of the conditions
def circle_equation (θ : Real) : Real := 2 * Real.cos θ
def line_equation (ρ θ a : Real) : Real := 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a

-- Statement of the problem
theorem tangent_line_circle (a : Real) :
  (∃ θ, line_equation (circle_equation θ) θ a = 0) ↔ (a = -8 ∨ a = 2) :=
by
  sorry

end tangent_line_circle_l65_65562


namespace arithmetic_sequence_l65_65100

noncomputable def a_n (n : ℕ) : ℕ := 4 * n - 3
noncomputable def S_n (n : ℕ) : ℕ := 2 * n^2 - n
noncomputable def b_n (n : ℕ) : ℕ := S_n n / (n - 1/2)
noncomputable def inv_b_prod (n : ℕ) : ℚ := 1 / (b_n n * b_n (n + 1))

theorem arithmetic_sequence (d : ℕ) (d_pos : d > 0) (h1 : (a_n 2) * (a_n 3) = 45) (h2 : (a_n 1) + (a_n 4) = 14) :
  (∀ n, a_n n = 4n - 3) ∧ (∀ n, S_n n = 2 * n^2 - n) ∧
  (∀ n, b_n n = 2 * n) ∧ (∀ n, Σ i in range n, inv_b_prod i = n / (4 * n + 4)) ∧
  (tendsto (λ n, Σ i in range n, inv_b_prod i) atTop (𝓝 (1 / 4))) :=
sorry

end arithmetic_sequence_l65_65100


namespace appropriate_line_chart_for_temperature_l65_65387

-- Define the assumption that line charts are effective in displaying changes in data over time
axiom effective_line_chart_display (changes_over_time : Prop) : Prop

-- Define the statement to be proved, using the assumption above
theorem appropriate_line_chart_for_temperature (changes_over_time : Prop) 
  (line_charts_effective : effective_line_chart_display changes_over_time) : Prop :=
  sorry

end appropriate_line_chart_for_temperature_l65_65387


namespace scientific_notation_86400_l65_65712

theorem scientific_notation_86400 : 86400 = 8.64 * 10^4 :=
by
  sorry

end scientific_notation_86400_l65_65712


namespace pure_imaginary_implies_x_eq_1_l65_65537

theorem pure_imaginary_implies_x_eq_1 (x : ℝ) (h : (x^2 - 1) + (x + 1) * complex.I = (x + 1) * complex.I) : x = 1 :=
by
  sorry

end pure_imaginary_implies_x_eq_1_l65_65537


namespace Q_plus_partition_exists_l65_65368

def Q_plus := {q : ℚ // q > 0}

section partition
variables (A B C : set Q_plus)

def BA := {q : Q_plus | ∃ (b ∈ B) (a ∈ A), q = ⟨b.val * a.val, mul_pos b.2 a.2⟩}
def B_squared := {q : Q_plus | ∃ (b1 b2 ∈ B), q = ⟨b1.val * b2.val, mul_pos b1.2 b2.2⟩}
def BC := {q : Q_plus | ∃ (b ∈ B) (c ∈ C), q = ⟨b.val * c.val, mul_pos b.2 c.2⟩}

noncomputable def is_partition (A B C : set Q_plus) :=
  (Q_plus = A ∪ B ∪ C) ∧ disjoint A B ∧ disjoint A C ∧ disjoint B C

theorem Q_plus_partition_exists (A B C : set Q_plus) 
  (hA : BA = B) (hB : B_squared = C) (hC : BC = A) 
  (hD : ∀ q : Q_plus, q.val ^ 3 ∈ A) 
  (hE : ∃ A B C, is_partition A B C ∧ ∀ n <= 34, n ∉ A ∨ n + 1 ∉ A) : 
  ∃ A B C, is_partition A B C ∧ hA ∧ hB ∧ hC ∧ hD ∧ hE :=
sorry

end Q_plus_partition_exists_l65_65368


namespace find_xyz_l65_65526

variables (A B C B₁ A₁ C₁ : Type)
variables [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B] [AddCommGroup C] [Module ℝ C]

def AC1 (AB BC CC₁ : A) (x y z : ℝ) : A :=
  x • AB + 2 • y • BC + 3 • z • CC₁

theorem find_xyz (AB BC CC₁ AC1 : A)
  (h1 : AC1 = AB + BC + CC₁)
  (h2 : AC1 = x • AB + 2 • y • BC + 3 • z • CC₁) :
  x + y + z = 11 / 6 :=
sorry

end find_xyz_l65_65526


namespace hazel_made_56_cups_l65_65137

-- Definitions based on problem conditions:
def sold_to_kids (sold: ℕ) := sold = 18
def gave_away (sold: ℕ) (gave: ℕ) := gave = sold / 2
def drank (drank: ℕ) := drank = 1
def half_total (total: ℕ) (sum_sold_gave_drank: ℕ) := sum_sold_gave_drank = total / 2

-- Main statement that needs to be proved:
theorem hazel_made_56_cups : ∃ (total: ℕ), 
  ∀ (sold gave drank sum_sold_gave_drank: ℕ), 
    sold_to_kids sold → 
    gave_away sold gave → 
    drank drank → 
    half_total total (sold + gave + drank) → 
    total = 56 := 
by sorry

end hazel_made_56_cups_l65_65137


namespace bamboo_pole_is_10_l65_65957

noncomputable def bamboo_pole_length (x : ℕ) : Prop :=
  (x - 4)^2 + (x - 2)^2 = x^2

theorem bamboo_pole_is_10 : bamboo_pole_length 10 :=
by
  -- The proof is not provided
  sorry

end bamboo_pole_is_10_l65_65957


namespace find_a_and_b_l65_65233

noncomputable def f (x: ℝ) (b: ℝ): ℝ := x^2 + 5*x + b
noncomputable def g (x: ℝ) (b: ℝ): ℝ := 2*b*x + 3

theorem find_a_and_b (a b: ℝ):
  (∀ x: ℝ, f (g x b) b = a * x^2 + 30 * x + 24) →
  a = 900 / 121 ∧ b = 15 / 11 :=
by
  intro H
  -- Proof is omitted as requested
  sorry

end find_a_and_b_l65_65233


namespace arithmetic_sequence_general_form_sum_first_n_terms_l65_65839

-- Define the arithmetic sequence conditions
noncomputable def arithmetic_sequence_conditions (a : ℕ → ℤ) (d : ℤ) := a 2 + a 5 = 19 ∧ a 3 + a 6 = 25

-- Define the geometric sequence conditions
noncomputable def geometric_sequence_conditions (a b : ℕ → ℤ) := b 1 = a 1 - 2  ∧ ∀ n : ℕ, b (n + 1) = 2 * b n

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) := ∃ a₁ d : ℤ, ∀ n, a n = a₁ + n * d

-- Problem 1: Prove the given arithmetic sequence
theorem arithmetic_sequence_general_form (a : ℕ → ℤ) (h : arithmetic_sequence_conditions a (3 : ℤ)) :
  ∃ a₁ d, ∀ n, a n = 3 * n - 1 := sorry

-- Problem 2: Prove the sum of the first n terms for sequence {b_n}
theorem sum_first_n_terms (a b : ℕ → ℤ) (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence_conditions a b) :
  ∀ n, (∑ k in Finset.range n, b k) = (3 * n^2 + n + 4) / 2 - 2^(n + 1) := sorry

end arithmetic_sequence_general_form_sum_first_n_terms_l65_65839


namespace student_marks_l65_65397

theorem student_marks (total_marks : ℕ) (percent_pass : ℝ) (failed_by : ℕ) (marks_obtained : ℕ) :
  total_marks = 500 → percent_pass = 0.4 → failed_by = 50 → marks_obtained = 150 → 
  marks_obtained = ((percent_pass * total_marks.to_real).to_nat) - failed_by := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  norm_num
  sorry

end student_marks_l65_65397


namespace necessary_and_sufficient_condition_x_eq_1_l65_65713

theorem necessary_and_sufficient_condition_x_eq_1
    (x : ℝ) :
    (x = 1 ↔ x^2 - 2 * x + 1 = 0) :=
sorry

end necessary_and_sufficient_condition_x_eq_1_l65_65713


namespace bob_pennies_l65_65152

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l65_65152


namespace simplify_and_evaluate_l65_65288

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  (1 + 1 / (m - 2)) / ((m^2 - m) / (m - 2)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_and_evaluate_l65_65288


namespace sum_of_divisors_330_l65_65688

theorem sum_of_divisors_330 : 
  (∑ k in (Finset.filter (λ d, 330 % d = 0) (Finset.range (331))), k) = 864 :=
by
  have prime_factorization : ∃ (a b c d : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 11 ∧ 330 = a * b * c * d :=
    ⟨2, 3, 5, 11, rfl, rfl, rfl, rfl, rfl⟩
  sorry

end sum_of_divisors_330_l65_65688


namespace initial_number_of_persons_l65_65649

theorem initial_number_of_persons (N : ℕ) (average_weight_increase : ℝ) (weight_of_left_person : ℝ) (weight_of_new_person : ℝ) 
  (h1 : average_weight_increase = 4.5) (h2 : weight_of_left_person = 65) (h3 : weight_of_new_person = 74) :
  N = 2 :=
by
  have total_weight_increase : ℝ := average_weight_increase * N
  have weight_difference : ℝ := weight_of_new_person - weight_of_left_person
  have equation : total_weight_increase = weight_difference
  rw [h1, h2, h3] at equation
  sorry

end initial_number_of_persons_l65_65649


namespace chang_e_5_descent_rate_l65_65620

theorem chang_e_5_descent_rate :
  ∀ (d_i d_f : ℝ) (t : ℝ) (v_i v_f : ℝ), 
  d_i = 1800 → d_f = 0 → t = 12 * 60 → v_i = 1800 → v_f = 0 →
  let v := (d_f - d_i) / t in
  let a := (v_f - v_i) / t in
  v = -5 / 2 ∧ a = -5 / 2 :=
by {
  intros d_i d_f t v_i v_f h_d_i h_d_f h_t h_v_i h_v_f,
  simp [h_d_i, h_d_f, h_t, h_v_i, h_v_f],
  let v := (0 - 1800) / (12 * 60),
  let a := (0 - 1800) / (12 * 60),
  split;
  simp [v, a]; sorry
}

end chang_e_5_descent_rate_l65_65620


namespace exists_set_no_three_ap_l65_65193

theorem exists_set_no_three_ap (n : ℕ) (k : ℕ) :
  (n ≥ 1983) →
  (k ≤ 100000) →
  ∃ S : Finset ℕ,
    S.card = n ∧
    (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a < b → b < c → b ≠ (a + c) / 2) :=
sorry

end exists_set_no_three_ap_l65_65193


namespace value_of_P2_plus_Pneg2_l65_65223

noncomputable def P (x : ℝ) : ℝ := sorry

axiom P_0 : P 0 = k
axiom P_1 : P 1 = 3 * k
axiom P_neg1 : P (-1) = 4 * k

theorem value_of_P2_plus_Pneg2 (P : ℝ → ℝ) (k : ℝ) 
  (h0 : P 0 = k) (h1 : P 1 = 3 * k) (h_neg1 : P (-1) = 4 * k) : 
  P 2 + P (-2) = 82 * k := sorry

end value_of_P2_plus_Pneg2_l65_65223


namespace length_of_CD_l65_65571

-- Definitions and conditions translated from the problem:
def AB : ℝ := 6
def BC : ℝ := 12
def r : ℝ := BC / 2
def perpendicular_distance_from_center_to_AB : ℝ := r
noncomputable def perpendicular_distance_from_center_to_CD : ℝ :=
  sqrt (r^2 - (AB / 2)^2)

-- Length of CD
def CD : ℝ := 2 * sqrt (r^2 - (perpendicular_distance_from_center_to_CD^2))

-- Lean 4 statement proving the length of the chord CD
theorem length_of_CD (h1 : AB = 6) (h2 : BC = 12) : CD = 6 := by
  sorry

end length_of_CD_l65_65571


namespace minimize_expression_l65_65060

variable (a b c : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : a ≠ 0)

theorem minimize_expression : 
  (a > b) → (b > c) → (a ≠ 0) → 
  ∃ x : ℝ, x = 4 ∧ ∀ y, y = (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 → x ≤ y := sorry

end minimize_expression_l65_65060


namespace lines_concurrent_l65_65239

open EuclideanGeometry

variables (C1 C2 : Circle)
variables (O1 O2 : Point)
variables (A1 A2 B1 B2 : Point)

-- Conditions
axiom centers_of_C1_C2 : C1.center = O1 ∧ C2.center = O2
axiom extern_tangents : external_tangent C1 C2 A1 A2
axiom intern_tangents : internal_tangent C1 C2 B1 B2

-- The theorem to prove
theorem lines_concurrent 
  (C1 C2 : Circle) (O1 O2 : Point) 
  (A1 A2 B1 B2 : Point) 
  (h1 : C1.center = O1) 
  (h2 : C2.center = O2) 
  (h3 : external_tangent C1 C2 A1 A2) 
  (h4 : internal_tangent C1 C2 B1 B2) : 
  concurrent (line_through A1 B1) (line_through A2 B2) (line_through O1 O2) :=
sorry

end lines_concurrent_l65_65239


namespace find_BC_length_l65_65624

variables {α : Real} (AO : Real) (BC : Real)
hypothesis (r : AO = 15)
hypothesis (h_sin_alpha : sin α = sqrt(21) / 5)
hypothesis (h1 : ∃ (M B C : Point),
  OnCircle M B C AO ∧ ∠ AMB = α ∧ ∠ OMC = α)
  
theorem find_BC_length : BC = 12 :=
by sorry

end find_BC_length_l65_65624


namespace evaluate_expression_eq_zero_l65_65437

theorem evaluate_expression_eq_zero (a b c d : ℝ) 
  (h1 : d = c + 1)
  (h2 : c = b - 8)
  (h3 : b = a + 4)
  (h4 : a = 7) 
  (h5 : a + 3 ≠ 0) 
  (h6 : b - 3 ≠ 0) 
  (h7 : c + 10 ≠ 0) 
  (h8 : d + 1 ≠ 0) : 
  ((a + 5) / (a + 3)) * ((b - 2) / (b - 3)) * ((c + 7) / (c + 10)) * ((d - 4) / (d + 1)) = 0 :=
by 
  have h9 : a = 7 := h4,
  have h10 : b = a + 4 := h3,
  have h11 : c = b - 8 := h2,
  have h12 : d = c + 1 := h1,
  sorry

end evaluate_expression_eq_zero_l65_65437


namespace perimeter_of_modified_quadrilateral_l65_65746

def original_vertices : List (ℝ × ℝ) := [(0,0), (5,0), (5,8), (0,8)]

def diagonal_length : ℝ := Real.sqrt ((5 - 0)^2 + (8 - 0)^2)

def left_segment_length : ℝ := 8 - 2
def right_segment_retained_length : ℝ := 3
def top_segment_length : ℝ := 5

def total_perimeter_length : ℝ := left_segment_length + diagonal_length + top_segment_length + right_segment_retained_length

theorem perimeter_of_modified_quadrilateral :
  total_perimeter_length = 14 + diagonal_length :=
by
  unfold left_segment_length
  unfold diagonal_length
  unfold top_segment_length
  unfold right_segment_retained_length
  unfold total_perimeter_length
  have h1 : 8 - 2 = 6 := rfl
  have h2 : 5 = 5 := rfl
  have h3 : 3 = 3 := rfl
  calc
    total_perimeter_length
        = 6 + Real.sqrt 89 + 5 + 3 : by rw [h1, rfl, h2, h3]
    ... = 14 + Real.sqrt 89 : by linarith

#eval total_perimeter_length    -- 14 + sqrt 89

end perimeter_of_modified_quadrilateral_l65_65746


namespace number_of_integers_in_T_l65_65224

theorem number_of_integers_in_T :
  let T := {n : ℤ | n > 1 ∧ ∃ m : ℕ, (n : ℚ) = 1 / (0.m1m2m3m4m5m6m7m8 : ℚ) ∧ (∀ i, mi = mi+8)} in
  T.card = 728 :=
sorry

end number_of_integers_in_T_l65_65224


namespace fraction_of_married_women_is_23_over_29_l65_65365

noncomputable def fraction_of_married_women (w : ℕ) (me : ℕ) (mm : ℕ) : ℚ :=
  let mw := me - mm in
  (mw : ℚ) / w

theorem fraction_of_married_women_is_23_over_29 :
  let total_employees := 100
  let w := 58
  let me := 60
  let m := total_employees - w
  let mm := (1 / 3 : ℚ) * m
  let mw := me - mm
  fraction_of_married_women w me mm = (23 / 29 : ℚ) :=
by
  sorry

end fraction_of_married_women_is_23_over_29_l65_65365


namespace area_preserved_l65_65632

variables (A B C D : ℝ × ℝ) (v : ℝ × ℝ)

def vector_add (p q : ℝ × ℝ) : ℝ × ℝ := (p.1 + q.1, p.2 + q.2)

def cross_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.2 - u.2 * v.1

def area_of_quadrilateral (P Q R S : ℝ × ℝ) : ℝ :=
  (cross_product (vector_add R (-P)) (vector_add S (-Q))) / 2.0

theorem area_preserved (A B C D : ℝ × ℝ) (v : ℝ × ℝ) :
  area_of_quadrilateral A B C D = area_of_quadrilateral (vector_add A v) B (vector_add C v) D :=
by sorry

end area_preserved_l65_65632


namespace find_schnauzers_l65_65462

theorem find_schnauzers (D S : ℕ) (h : 3 * D - 5 + (D - S) = 90) (hD : D = 20) : S = 45 :=
by
  sorry

end find_schnauzers_l65_65462


namespace repayment_amount_l65_65210

theorem repayment_amount (borrowed amount : ℝ) (increase_percentage : ℝ) (final_amount : ℝ) 
  (h1 : borrowed_amount = 100) 
  (h2 : increase_percentage = 0.10) :
  final_amount = borrowed_amount * (1 + increase_percentage) :=
by 
  rw [h1, h2]
  norm_num
  exact eq.refl 110


end repayment_amount_l65_65210


namespace solution_count_l65_65141

theorem solution_count : 
  {x : ℤ | (x - 3) ^ (16 - x ^ 2) = 1}.finite.card = 3 :=
by
  sorry

end solution_count_l65_65141


namespace value_of_a_range_of_f_l65_65116

noncomputable def f (a x: ℝ) : ℝ := a^(x - 1)

theorem value_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f a 2 = 1/2) : a = 1/2 :=
  sorry

theorem range_of_f (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x, x ≥ 0 → 
    if ha : 0 < a ∧ a < 1 then f a x ∈ (0, a^{-1}] 
    else if ha : 1 < a then f a x ∈ [a^{-1}, +∞) 
    else false) :=
  sorry

end value_of_a_range_of_f_l65_65116


namespace sufficient_and_necessary_condition_l65_65596

variables {E : Type*} [inner_product_space ℝ E]

theorem sufficient_and_necessary_condition (a b : E) :
  (∥a + b∥ > ∥a - b∥) ↔ (inner a b > 0) :=
by
  sorry

end sufficient_and_necessary_condition_l65_65596


namespace number_of_correct_statements_l65_65877

def new_op (a b : ℝ) : ℝ :=
if a < b then a + b - 3 else a - b + 3

theorem number_of_correct_statements : 
  let op := new_op; 
  (op (-1) (-2) = 4) ∧
  (∀ x, op x (x + 2) = 5 → x = 3) ∧
  (∀ x, op x (2 * x) = 3 → (x = 2 ∨ x = 0)) ∧
  (¬ ∃ x, op ((x^2 + 1) 1) = 0) →
  (2 correct_statements) :=
by
  sorry

end number_of_correct_statements_l65_65877


namespace slope_range_l65_65188

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A B C : Point)

def isOrthocenter (F : Point) (Δ : Triangle) : Prop :=
  -- Orthocenter condition (To be defined properly, simplified here)
  sorry

def isCentroid (G : Point) (Δ : Triangle) : Prop :=
  -- Centroid condition (To be defined properly, simplified here)
  sorry

noncomputable def slope (P Q : Point) : ℝ :=
  (Q.y - P.y) / (Q.x - P.x)

def liesOnParabola (P : Point) (p : ℝ) : Prop :=
  P.y^2 = 2 * p * P.x

theorem slope_range (p : ℝ) (Δ : Triangle) (F G : Point)
  (hp : p > 0)
  (hA : liesOnParabola Δ.A p)
  (hB : liesOnParabola Δ.B p)
  (hC : liesOnParabola Δ.C p)
  (hF : isOrthocenter F Δ)
  (hG : isCentroid G Δ) :
  slope F G ∈ Set.Icc (-Real.sqrt 7 / 7) (Real.sqrt 7 / 7) :=
sorry

end slope_range_l65_65188


namespace find_schnauzers_l65_65461

theorem find_schnauzers (D S : ℕ) (h : 3 * D - 5 + (D - S) = 90) (hD : D = 20) : S = 45 :=
by
  sorry

end find_schnauzers_l65_65461


namespace solve_for_y_l65_65538

theorem solve_for_y (y : ℝ) (h : -3 * y - 9 = 6 * y + 3) : y = -4 / 3 :=
by
  sorry

end solve_for_y_l65_65538


namespace initial_people_count_l65_65756

theorem initial_people_count (x : ℕ) (h : (x - 2) + 2 = 10) : x = 10 :=
by
  sorry

end initial_people_count_l65_65756


namespace polynomial_divides_factorial_plus_two_l65_65773

theorem polynomial_divides_factorial_plus_two (P : Polynomial ℤ) :
  (∀ n : ℕ, 0 < n → P.eval n ∣ n.factorial + 2) → (P = Polynomial.C 1 ∨ P = Polynomial.C (-1)) :=
by sorry

end polynomial_divides_factorial_plus_two_l65_65773


namespace even_number_of_faces_with_all_three_colors_l65_65943

-- Definitions based on conditions
def Polyhedron (V : Type) [DecidableEq V] := 
  { P : Set (Set (Set V → Set V → Set V)) // ∀ face ∈ P, face.card = 3 }

def is_colored {V : Type} (colors : Set V) (vertices : Set V) : Prop :=
  ∀ v ∈ vertices, v ∈ colors

-- Theorem statement:
theorem even_number_of_faces_with_all_three_colors
  (V : Type) [DecidableEq V]
  (colors vertices : Set V)
  (p : Polyhedron V) :
  is_colored colors vertices → 
  ∃ even_faces_count : ℕ,
  even_faces_count % 2 = 0 :=
sorry

end even_number_of_faces_with_all_three_colors_l65_65943


namespace no_solutions_rebus_l65_65969

theorem no_solutions_rebus : ∀ (K U S Y : ℕ), 
  (K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y) →
  (∀ d, d < 10) → 
  let KUSY := 1000 * K + 100 * U + 10 * S + Y in
  let UKSY := 1000 * U + 100 * K + 10 * S + Y in
  let result := 10000 * U + 1000 * K + 100 * S + 10 * Y + S in
  KUSY + UKSY ≠ result :=
begin
  sorry
end

end no_solutions_rebus_l65_65969


namespace measure_time_with_hourglasses_l65_65331

def hourglass7 : ℕ := 7
def hourglass11 : ℕ := 11
def target_time : ℕ := 15

theorem measure_time_with_hourglasses :
  ∃ (time_elapsed : ℕ), time_elapsed = target_time :=
by
  use 15
  sorry

end measure_time_with_hourglasses_l65_65331


namespace cubical_storage_unit_blocks_l65_65612

theorem cubical_storage_unit_blocks :
  let side_length := 8
  let thickness := 1
  let total_volume := side_length ^ 3
  let interior_side_length := side_length - 2 * thickness
  let interior_volume := interior_side_length ^ 3
  let blocks_required := total_volume - interior_volume
  blocks_required = 296 := by
    sorry

end cubical_storage_unit_blocks_l65_65612


namespace Petya_has_24_chips_l65_65458

noncomputable def PetyaChips (x y : ℕ) : ℕ := 3 * x - 3

theorem Petya_has_24_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) : PetyaChips x y = 24 :=
by
  sorry

end Petya_has_24_chips_l65_65458


namespace locus_is_ellipse_l65_65999

-- Define the complex number z in terms of real numbers x and y
variable {x y : ℝ}

-- Define the complex numbers z₁ and z₂
def z₁ := complex.mk 2 (-1)
def z₂ := complex.mk (-3) 1

-- Define the equation to be satisfied
def equation := (complex.abs (complex.mk x y - z₁) + complex.abs (complex.mk x y - z₂) = 6)

-- Prove that the locus defined by the equation is an ellipse
theorem locus_is_ellipse (h : equation) : 
  ∃ a b c d : ℝ, a ≠ b ∧ equation ↔ x^2 / a^2 + y^2 / b^2 = 1 := 
sorry

end locus_is_ellipse_l65_65999


namespace graph_with_vertices_degree_ge_two_contains_cycle_graph_with_n_vertices_and_n_edges_contains_cycle_l65_65974

-- Part One
theorem graph_with_vertices_degree_ge_two_contains_cycle {G : SimpleGraph V} 
  [Fintype V] (h1 : ∀ v : V, 2 ≤ G.degree v) : 
  G.has_cycle := sorry

-- Part Two
theorem graph_with_n_vertices_and_n_edges_contains_cycle {G : SimpleGraph V}
  [Fintype V] (h1 : Fintype.card V = n) (h2 : n ≤ G.edge_finset.card) :
  G.has_cycle := sorry

end graph_with_vertices_degree_ge_two_contains_cycle_graph_with_n_vertices_and_n_edges_contains_cycle_l65_65974


namespace determine_b_for_smallest_positive_a_real_roots_l65_65778

theorem determine_b_for_smallest_positive_a_real_roots :
  ∃ (a b : ℝ), 0 < a ∧ (∀ x : ℝ, (polynomial.eval x (polynomial.C 1 * polynomial.X ^ 3 - polynomial.C (2 * a) * polynomial.X ^ 2 + polynomial.C b * polynomial.X - polynomial.C (2 * a))).is_root) ∧ b = 81 / 4 :=
by sorry

end determine_b_for_smallest_positive_a_real_roots_l65_65778


namespace find_angle_B_max_area_triangle_l65_65162

-- Step 1: Prove B = π / 6
theorem find_angle_B (a b A B : ℝ) (hb : b = 2) (h : a * sin (2 * B) = sqrt 3 * b * sin A) :
  B = π / 6 :=
by sorry

-- Step 2: Prove the maximum area of triangle
theorem max_area_triangle (a b c B : ℝ) (hb : b = 2) (hgeo : b * b = a * c) :
  1 / 2 * a * c * sin B ≤ sqrt 3 :=
by sorry

end find_angle_B_max_area_triangle_l65_65162


namespace scientific_notation_280000000_l65_65780

theorem scientific_notation_280000000 :
  (∃ n : ℕ, 280000000 = (2.8 * 10^n)) ∧ (1 ≤ 2.8) ∧ (2.8 < 10) ∧ (n = 8) :=
by
  have h : 280000000 = 2.8 * 10^8 := sorry
  exact ⟨⟨8, h⟩, sorry, sorry, sorry⟩

end scientific_notation_280000000_l65_65780


namespace sum_of_other_endpoint_coordinates_l65_65958

-- Definitions for the conditions
structure Point where
  x : ℝ
  y : ℝ

def endpoint1 : Point := { x := 6, y := -2 }
def midpoint : Point := { x := 5, y := 4 }

-- Goal: Prove that the sum of the coordinates of the other endpoint is 14
theorem sum_of_other_endpoint_coordinates :
  ∃ (endpoint2 : Point), 
    (midpoint.x = (endpoint1.x + endpoint2.x) / 2) ∧ 
    (midpoint.y = (endpoint1.y + endpoint2.y) / 2) ∧ 
    (endpoint2.x + endpoint2.y = 14) :=
by
  sorry

end sum_of_other_endpoint_coordinates_l65_65958


namespace quadruplet_babies_l65_65563

variable (a b c : ℕ)
variable (h1 : b = 3 * c)
variable (h2 : a = 5 * b)
variable (h3 : 2 * a + 3 * b + 4 * c = 1500)

theorem quadruplet_babies : 4 * c = 136 := by
  sorry

end quadruplet_babies_l65_65563


namespace point_b_value_l65_65963

theorem point_b_value :
  let A_initial := -2 in
  let A_after_right_8 := A_initial + 8 in
  let B := A_after_right_8 - 4 in
  B = 2 :=
by
  let A_initial := -2
  let A_after_right_8 := A_initial + 8
  let B := A_after_right_8 - 4
  show B = 2
  sorry

end point_b_value_l65_65963


namespace sum_of_divisors_330_l65_65689

theorem sum_of_divisors_330 : 
  (∑ k in (Finset.filter (λ d, 330 % d = 0) (Finset.range (331))), k) = 864 :=
by
  have prime_factorization : ∃ (a b c d : ℕ), a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 11 ∧ 330 = a * b * c * d :=
    ⟨2, 3, 5, 11, rfl, rfl, rfl, rfl, rfl⟩
  sorry

end sum_of_divisors_330_l65_65689


namespace measure_time_with_hourglasses_l65_65332

def hourglass7 : ℕ := 7
def hourglass11 : ℕ := 11
def target_time : ℕ := 15

theorem measure_time_with_hourglasses :
  ∃ (time_elapsed : ℕ), time_elapsed = target_time :=
by
  use 15
  sorry

end measure_time_with_hourglasses_l65_65332


namespace length_of_AB_l65_65743

-- Definitions of conditions
def right_angle_triangle (A B C : Type) (right_angle_at : Prop) := 
  right_angle_at 

def inscribed_circle_radius (radius : ℝ) := 
  radius 

def angle_A (angle : ℝ) := 
  angle 

-- The main theorem we need to prove
theorem length_of_AB 
  (A B C : Type)
  (right_angle_at_C : right_angle_triangle A B C (true))
  (inscribed_circle : inscribed_circle_radius 4)
  (angle_at_A : angle_A 45) 
  : ∃ (length_of_AB : ℝ), length_of_AB = 8 * real.sqrt 2 := 
sorry

end length_of_AB_l65_65743


namespace lateral_surface_area_cone_l65_65838

-- Given definitions (conditions)
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Question transformed into a theorem statement
theorem lateral_surface_area_cone : 
  ∀ (r l : ℝ), r = radius → l = slant_height → (1 / 2) * (2 * real.pi * r) * l = 15 * real.pi := 
by 
  intros r l hr hl
  rw [hr, hl]
  sorry

end lateral_surface_area_cone_l65_65838


namespace cone_lateral_surface_area_l65_65834

-- Definitions and conditions
def radius (r : ℝ) := r = 3
def slant_height (l : ℝ) := l = 5
def lateral_surface_area (A : ℝ) (C : ℝ) (l : ℝ) := A = 0.5 * C * l
def circumference (C : ℝ) (r : ℝ) := C = 2 * Real.pi * r

-- Proof (statement only)
theorem cone_lateral_surface_area :
  ∀ (r l C A : ℝ), 
    radius r → 
    slant_height l → 
    circumference C r → 
    lateral_surface_area A C l → 
    A = 15 * Real.pi := 
by intros; sorry

end cone_lateral_surface_area_l65_65834


namespace problem_y_eq_l65_65606

theorem problem_y_eq (y : ℝ) (h : y^3 - 3*y = 9) : y^5 - 10*y^2 = -y^2 + 9*y + 27 := by
  sorry

end problem_y_eq_l65_65606


namespace average_percentage_reduction_equation_l65_65724

theorem average_percentage_reduction_equation (x : ℝ) : 200 * (1 - x)^2 = 162 :=
by 
  sorry

end average_percentage_reduction_equation_l65_65724


namespace program_outputs_63_l65_65477

def loop_program_output : ℕ :=
  let rec loop (s i : ℕ) : ℕ :=
    if i > 30 then i
    else loop (s + i) (2 * i + 1)
  in loop 0 1

theorem program_outputs_63 :
  loop_program_output = 63 := by
  sorry

end program_outputs_63_l65_65477


namespace Petya_chips_l65_65453

theorem Petya_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) :
  ∃ T : ℕ, T = 24 :=
by {
  let T_triangle := 3 * x - 3,
  let T_square := 4 * y - 4,
  -- The conditions ensure T_triangle = T_square
  have h3 : T_triangle = T_square, from h2,
  -- substituting y = x - 2 into T_square
  have h4 : T_square = 4 * (x - 2) - 4, from calc
    T_square = 4 * y - 4 : by rfl
    ... = 4 * (x - 2) - 4 : by rw h1,
  -- simplify to find x,
  have h5 : 3 * x - 3 = 4 * (x - 2) - 4, from h2,
  have h6 : 3 * x - 3 = 4 * x - 8 - 4, from h5,
  have h7 : 3 * x - 3 = 4 * x - 12, from by simp at h6,
  have h8 : -3 = x - 12, from by linarith,
  have h9 : x = 9, from by linarith,
  -- Find the total number of chips
  let T := 3 * x - 3,
  have h10 : T = 24, from calc
    T = 3 * 9 - 3 : by rw h9
    ... = 24 : by simp,
  exact ⟨24, h10⟩
}

end Petya_chips_l65_65453


namespace hexagonal_pyramid_volume_l65_65068

open Real -- Open the Real namespace

theorem hexagonal_pyramid_volume (a R : ℝ) (h : R ≥ a) :
  let S := (3 * Real.sqrt 3 / 2) * a^2,
      h := R + Real.sqrt (R^2 - a^2)
  in (1 / 3) * S * h = (Real.sqrt 3 / 2) * a^2 * (R + Real.sqrt (R^2 - a^2)) := by
  sorry

end hexagonal_pyramid_volume_l65_65068


namespace triangle_identity_l65_65554

theorem triangle_identity (A B C : Real) (h_triangle: A + B + C = π) :
  (cot A + cot B) / (tan A + tan B) + 
  (cot B + cot C) / (tan B + tan C) + 
  (cot C + cot A) / (tan C + tan A) = 1 :=
by
  sorry

end triangle_identity_l65_65554


namespace total_share_proof_l65_65264

variable (P R : ℕ) -- Parker's share and Richie's share
variable (total_share : ℕ) -- Total share

-- Define conditions
def ratio_condition : Prop := (P : ℕ) / 2 = (R : ℕ) / 3
def parker_share_condition : Prop := P = 50

-- Prove the total share is 125
theorem total_share_proof 
  (h1 : ratio_condition P R)
  (h2 : parker_share_condition P) : 
  total_share = 125 :=
sorry

end total_share_proof_l65_65264


namespace mn_value_l65_65096

-- Defining the sequence conditions
variable {a : ℕ → ℝ}
variable {q : ℝ}

axiom geom_seq (a : ℕ → ℝ) (q : ℝ) : (∀ n, a(n + 1) = a n * q)

axiom positive_seq (a : ℕ → ℝ) : (∀ n, a n > 0)

axiom condition_one (a : ℕ → ℝ) : 2 * a 5 = a 3 - a 4

axiom condition_two (a : ℕ → ℝ) (n m : ℕ) : a 1 = 4 * real.sqrt (a n * a m)

theorem mn_value : ∃ n m, n + m = 6 := by
  sorry

end mn_value_l65_65096


namespace find_b_value_l65_65245

def inversely_proportional (a b : ℕ) : Prop :=
  ∃ k, a^3 * b^(1/4) = k

theorem find_b_value
  (a b : ℝ)
  (h1 : inversely_proportional a b)
  (h2 : a = 3)
  (h3 : b = 16)
  (h4 : a^2 * b = 54) :
  b = 54^(2/5) :=
by {
  sorry -- ← This is where the proof would go.
}

end find_b_value_l65_65245


namespace bridge_length_is_132_l65_65016

-- Definitions of variables given in the conditions
def train_length := 140 -- Length of the train in meters
def train_speed_km_per_hr := 72 -- Speed of the train in km/hr
def time_to_cross_bridge := 13.598912087033037 -- Time to cross the bridge in seconds

-- Conversion of speed from km/hr to m/s
def train_speed_m_per_s := train_speed_km_per_hr * (1000 / 3600)

-- Total distance covered (train length + bridge length)
def total_distance := train_speed_m_per_s * time_to_cross_bridge

-- Length of the bridge calculation
def bridge_length := total_distance - train_length

-- Proof that the length of the bridge is 132 meters
theorem bridge_length_is_132 : bridge_length = 132 :=
by
  -- Since we need a proof here, we are using sorry to skip it for now.
  sorry

end bridge_length_is_132_l65_65016


namespace day_of_week_proof_l65_65200

/-- 
January 1, 1978, is a Sunday in the Gregorian calendar.
What day of the week is January 1, 2000, in the Gregorian calendar?
-/
def day_of_week_2000 := "Saturday"

theorem day_of_week_proof :
  let initial_year := 1978
  let target_year := 2000
  let initial_weekday := "Sunday"
  let years_between := target_year - initial_year -- 22 years
  let normal_days := years_between * 365 -- Normal days in these years
  let leap_years := 5 -- Number of leap years in the range
  let total_days := normal_days + leap_years -- Total days considering leap years
  let remainder_days := total_days % 7 -- days modulo 7
  initial_weekday = "Sunday" → remainder_days = 6 → 
  day_of_week_2000 = "Saturday" :=
by
  sorry

end day_of_week_proof_l65_65200


namespace return_amount_is_correct_l65_65212

-- Define the borrowed amount and the interest rate
def borrowed_amount : ℝ := 100
def interest_rate : ℝ := 10 / 100

-- Define the condition of the increased amount
def increased_amount : ℝ := borrowed_amount * interest_rate

-- Define the total amount to be returned
def total_amount : ℝ := borrowed_amount + increased_amount

-- Lean 4 statement to prove
theorem return_amount_is_correct : total_amount = 110 := by
  -- Borrowing amount definition
  have h1 : borrowed_amount = 100 := rfl
  -- Interest rate definition
  have h2 : interest_rate = 10 / 100 := rfl
  -- Increased amount calculation
  have h3 : increased_amount = borrowed_amount * interest_rate := rfl
  -- Expanded calculation of increased_amount
  have h4 : increased_amount = 100 * (10 / 100) := by rw [h1, h2]
  -- Simplify the increased_amount
  have h5 : increased_amount = 10 := by norm_num [h4]
  -- Total amount calculation
  have h6 : total_amount = borrowed_amount + increased_amount := rfl
  -- Expanded calculation of total_amount
  have h7 : total_amount = 100 + 10 := by rw [h1, h5]
  -- Simplify the total_amount
  show 100 + 10 = 110 from rfl
  sorry

end return_amount_is_correct_l65_65212


namespace trigonometric_identity_l65_65474

theorem trigonometric_identity (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : cos α = 3 / 5) :
  (sin (α + π / 6 + π / 12)) = 7 * sqrt 2 / 10 :=
by
  sorry

end trigonometric_identity_l65_65474


namespace leibniz_proof_infinite_series_proof_l65_65703

noncomputable def leibniz_formula : Prop := 
  (1 - (1 / 3) + (1 / 5) - (1 / 7) + ∙∙∙ = (π / 4))

noncomputable def sum_infinite_series : Prop := 
  (1 + (1 / 3^2) + (1 / 5^2) + (1 / 7^2) + ∙∙∙ = (π^2 / 8))

theorem leibniz_proof (h₁ : Σ, ℕ → ℝ  := sorry
  :
  leibniz_formula :=
sorry

theorem infinite_series_proof (h₂ : Σ, ℕ → ℝ  := sorry 
  :
  sum_infinite_series :=
sorry

end leibniz_proof_infinite_series_proof_l65_65703


namespace regression_coefficients_l65_65123

theorem regression_coefficients {x y : ℝ} (b a : ℝ) :
  (∀ x y : ℝ, correlation x y < 0) →
  (∀ x : ℝ, x = 0 → y > 4.0) →
  (a > 0 ∧ b < 0) :=
by
  intros h1 h2
  sorry

end regression_coefficients_l65_65123


namespace cone_lateral_surface_area_l65_65832

-- Define the radius and slant height as given constants
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Definition of the lateral surface area
def lateral_surface_area (r l : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * r in
  (circumference * l) / 2

-- The proof problem statement in Lean 4
theorem cone_lateral_surface_area : lateral_surface_area radius slant_height = 15 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l65_65832


namespace infinitely_many_noncongruent_triangles_l65_65286

def relatively_prime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd c a = 1

def integer_area (a b c : ℕ) : Prop :=
  ∃ (S : ℕ), let p := (a + b + c) / 2 in S = Nat.sqrt (p * (p - a) * (p - b) * (p - c))

def non_integer_altitudes (a b c : ℕ) : Prop :=
  let S := Nat.sqrt ((a + (b + c) / 2) * (a - (b - c) / 2)) in
  ¬((2 * S) / a).isNat ∧ ¬((2 * S) / b).isNat ∧ ¬((2 * S) / c).isNat

theorem infinitely_many_noncongruent_triangles :
  ∃ (t : ℕ) (a b c : ℕ), (t > 0) ∧
  a = 9 * t * (5 * t + 2) + 5 ∧
  b = 16 * t * (5 * t + 2) + 5 ∧
  c = 25 * t * (5 * t + 2) ∧
  t % 210 = 106 ∧
  relatively_prime a b c ∧
  integer_area a b c ∧
  non_integer_altitudes a b c :=
  sorry

end infinitely_many_noncongruent_triangles_l65_65286


namespace sin_of_angle_through_P_l65_65908

theorem sin_of_angle_through_P : 
  let α := angle.mkCoordOrigin
  let P := (-Real.sqrt 3, -1)
  ∃ (α : ℝ), α = atan2 (snd P) (fst P) ∧ sin α = -1 / 2 :=
by
  let α := angle.mkCoordOrigin
  let P := (-Real.sqrt 3, -1)
  use (atan2 (snd P) (fst P))
  split
  {
    exact rfl
  }
  {
    sorry -- Proof of sin α = -1/2
  }

end sin_of_angle_through_P_l65_65908


namespace max_area_of_triangle_l65_65192

theorem max_area_of_triangle (a c : ℝ)
    (h1 : a^2 + c^2 = 16 + a * c) : 
    ∃ s : ℝ, s = 4 * Real.sqrt 3 := by
  sorry

end max_area_of_triangle_l65_65192


namespace quadratic_eq_has_distinct_real_roots_l65_65548

theorem quadratic_eq_has_distinct_real_roots (c : ℝ) (h : c = 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 ^ 2 - 3 * x1 + c = 0) ∧ (x2 ^ 2 - 3 * x2 + c = 0)) :=
by {
  sorry
}

end quadratic_eq_has_distinct_real_roots_l65_65548


namespace trip_time_80_minutes_l65_65952

noncomputable def v : ℝ := 1 / 2
noncomputable def speed_highway := 4 * v -- 4 times speed on the highway
noncomputable def time_mountain : ℝ := 20 / v -- Distance on mountain road divided by speed on mountain road
noncomputable def time_highway : ℝ := 80 / speed_highway -- Distance on highway divided by speed on highway
noncomputable def total_time := time_mountain + time_highway

theorem trip_time_80_minutes : total_time = 80 :=
by sorry

end trip_time_80_minutes_l65_65952


namespace graph_is_ellipse_l65_65431

theorem graph_is_ellipse (x y : ℝ) :
    x^2 + 2*y^2 - 6*x - 8*y + 9 = 0 →
    ∃ a b c d e f: ℝ, a = 1 ∧ b = 2 ∧ c = -6 ∧ d = -8 ∧ e = 9 ∧ f = 0 
    ∧ (a * ((x - 3)^2 / 8) + b * ((y - 2)^2 / 4) = 1 ∧ f = 0) ∧ 
    (a / 8 = 1 ∧ b / 4 = 1/2 ) :=
begin
  sorry
end

end graph_is_ellipse_l65_65431


namespace most_cost_effective_years_for_machine_cost_effective_years_8_l65_65800

theorem most_cost_effective_years_for_machine (x : ℕ) (h : 1 <= x) :
  10.47 - 1.3 * x ≥ 0 → x ≤ 8 :=
  by sorry

theorem cost_effective_years_8 :
  10.47 - 1.3 * 8 ≥ 0 :=
  by sorry

end most_cost_effective_years_for_machine_cost_effective_years_8_l65_65800


namespace bart_trees_needed_l65_65788

noncomputable def calculate_trees (firewood_per_tree : ℕ) (logs_per_day : ℕ) (days_in_period : ℕ) : ℕ :=
  (days_in_period * logs_per_day) / firewood_per_tree

theorem bart_trees_needed :
  let firewood_per_tree := 75 in
  let logs_per_day := 5 in
  let days_in_november := 30 in
  let days_in_december := 31 in
  let days_in_january := 31 in
  let days_in_february := 28 in
  let total_days := days_in_november + days_in_december + days_in_january + days_in_february in
  calculate_trees firewood_per_tree logs_per_day total_days = 8 :=
by
  sorry

end bart_trees_needed_l65_65788


namespace original_price_before_discounts_l65_65747

theorem original_price_before_discounts (P : ℝ) (h : 0.684 * P = 6840) : P = 10000 :=
by
  sorry

end original_price_before_discounts_l65_65747


namespace per_minute_charge_after_6_minutes_l65_65725

noncomputable def cost_plan_a (x : ℝ) (t : ℝ) : ℝ :=
  if t <= 6 then 0.60 else 0.60 + (t - 6) * x

noncomputable def cost_plan_b (t : ℝ) : ℝ :=
  t * 0.08

theorem per_minute_charge_after_6_minutes :
  ∃ (x : ℝ), cost_plan_a x 12 = cost_plan_b 12 ∧ x = 0.06 :=
by
  use 0.06
  simp [cost_plan_a, cost_plan_b]
  sorry

end per_minute_charge_after_6_minutes_l65_65725


namespace translation_symmetry_l65_65995

theorem translation_symmetry :
  ∃ m > 0, (∀ x : ℝ, y = sin x - sqrt 3 * cos x -> y = function.translate (λ x : ℝ, 2 * sin (x + m - pi / 3))) ∧ 
  (∀ y : ℝ, y = 2 * sin (x + m - pi / 3) -> y = 2 * sin (-x + m - pi / 3)) ∧ 
  (∀ m > 0, ∃ k : ℤ, m = k * pi + 5 * pi / 6) :=
begin
  sorry
end

end translation_symmetry_l65_65995


namespace initial_cost_of_smartphone_l65_65249

theorem initial_cost_of_smartphone 
(C : ℝ) 
(h : 0.85 * C = 255) : 
C = 300 := 
sorry

end initial_cost_of_smartphone_l65_65249


namespace original_height_of_tree_l65_65924

theorem original_height_of_tree
  (current_height_in_inches : ℕ)
  (percent_taller : ℕ)
  (current_height_is_V := 180)
  (percent_taller_is_50 := 50) :
  (current_height_in_inches * 100) / (percent_taller + 100) / 12 = 10 := sorry

end original_height_of_tree_l65_65924


namespace find_point_M_l65_65324

/-- Define the function f(x) = x^3 + x - 2. -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- Define the derivative of the function, f'(x). -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

/-- Define the condition that the slope of the tangent line is perpendicular to y = -1/4x - 1. -/
def slope_perpendicular_condition (m : ℝ) : Prop := m = 4

/-- Main theorem: The coordinates of the point M are (1, 0) and (-1, -4). -/
theorem find_point_M : 
  ∃ (x₀ y₀ : ℝ), f x₀ = y₀ ∧ slope_perpendicular_condition (f' x₀) ∧ 
  ((x₀ = 1 ∧ y₀ = 0) ∨ (x₀ = -1 ∧ y₀ = -4)) := 
sorry

end find_point_M_l65_65324


namespace y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l65_65119

variable (x y : ℝ)

-- Condition: y is defined as a function of x
def y_def := y = 2 * x + 5

-- Theorem: y > 0 if and only if x > -5/2
theorem y_positive_if_and_only_if_x_greater_than_negative_five_over_two 
  (h : y_def x y) : y > 0 ↔ x > -5 / 2 := by sorry

end y_positive_if_and_only_if_x_greater_than_negative_five_over_two_l65_65119


namespace triangle_sides_length_a_triangle_perimeter_l65_65227

theorem triangle_sides_length_a (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) :
  a = Real.sqrt 3 :=
sorry

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) 
  (h2 : (b * c * Real.sin (π / 3)) / 2 = Real.sqrt 3 / 2) :
  a + b + c = 3 + Real.sqrt 3 :=
sorry

end triangle_sides_length_a_triangle_perimeter_l65_65227


namespace odd_and_increasing_on_interval_l65_65022

theorem odd_and_increasing_on_interval (f : ℝ → ℝ) (h1 : ∀ x, f (-x) = - f x) (h2 : ∀ x y, 1 < x → x < y → f x < f y) :
  f = (λ x, Real.exp x - Real.exp (-x)) :=
sorry

end odd_and_increasing_on_interval_l65_65022


namespace trucks_sold_l65_65044

-- Definitions for conditions
def cars_and_trucks_total (T C : Nat) : Prop :=
  T + C = 69

def cars_more_than_trucks (T C : Nat) : Prop :=
  C = T + 27

-- Theorem statement
theorem trucks_sold (T C : Nat) (h1 : cars_and_trucks_total T C) (h2 : cars_more_than_trucks T C) : T = 21 :=
by
  -- This will be replaced by the proof
  sorry

end trucks_sold_l65_65044


namespace periodic_sequence_sum_l65_65576

noncomputable def x (n : ℕ) : ℕ → ℝ
| 0     := 0
| 1     := 1
| 2     := a
| (n+3) := |x(n+2) - x(n+1)|

theorem periodic_sequence_sum (a : ℝ) (ha : a ≤ 1) (ha_ne_zero : a ≠ 0) (h_period : ∀ n, x (n + 3) = x n) :
  ∑ i in range 2013, x i = 1342 := sorry

end periodic_sequence_sum_l65_65576


namespace vertex_of_quadratic_l65_65987

noncomputable def f (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem vertex_of_quadratic :
  ∃ (h k : ℝ), (∀ x : ℝ, f x = (x - h)^2 + k) ∧ (h = 1) ∧ (k = -2) :=
by
  sorry

end vertex_of_quadratic_l65_65987


namespace find_x_when_y_is_18_l65_65326

-- We assert that y varies inversely as x^2
def varies_inversely_as (y x : ℝ) (k : ℝ) : Prop :=
  y * x^2 = k

-- Given conditions
def given_conditions (x y k : ℝ) : Prop :=
  varies_inversely_as 2 3 k ∧ varies_inversely_as y x k

-- The final theorem we need to prove
theorem find_x_when_y_is_18 (k : ℝ) :
  given_conditions 3 2 k → given_conditions 1 18 k :=
begin
  sorry
end

end find_x_when_y_is_18_l65_65326


namespace sum_convergence_l65_65221

noncomputable def a : ℕ → ℝ
| 0     := 1  -- We are assuming a_0 is a placeholder for the starting value
| (n+1) := sorry -- Define a_{n+1} in terms of b_n and condition b_{n+1} = a_{n+1} b_n - 2

noncomputable def b : ℕ → ℝ
| 0     := 1  -- We are assuming b_0 is a placeholder for the starting value
| (n+1) := a (n+1) * b n - 2

theorem sum_convergence :
  (∀ n > 0, b n = a n * b (n-1) - 2) → ∀ n, n > 0 → ∑ n, ∑ (i : ℕ) in finset.range n, (1 / (∏ j in finset.range (i + 1), a j)) = 3 / 2 :=
by 
  -- Proof goes here
  sorry

end sum_convergence_l65_65221


namespace perpendicular_lines_l65_65043

variable {b : ℝ}

def direction_vector1 : ℝ × ℝ × ℝ := (1, b, 3)
def direction_vector2 : ℝ × ℝ × ℝ := (-2, 5, 1)

theorem perpendicular_lines : (direction_vector1.1 * direction_vector2.1 + 
                              direction_vector1.2 * direction_vector2.2 + 
                              direction_vector1.3 * direction_vector2.3 = 0) → 
                             b = -1 / 5 := by
  intro h
  sorry

end perpendicular_lines_l65_65043


namespace equation_of_line_passing_through_points_l65_65039

-- Definition of the points
def point1 : ℝ × ℝ := (-2, -3)
def point2 : ℝ × ℝ := (4, 7)

-- The statement to prove
theorem equation_of_line_passing_through_points :
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (forall (x y : ℝ), 
  y + 3 = (5 / 3) * (x + 2) → 3 * y - 5 * x = 1) := sorry

end equation_of_line_passing_through_points_l65_65039


namespace find_1997th_digit_in_decimal_expansion_of_1_div_7_l65_65340

theorem find_1997th_digit_in_decimal_expansion_of_1_div_7 :
  let repeating_block : String := "142857"
  let n : Nat := 1997
  (repeating_block[(n % 6) - 1] = '5') :=
by
  let repeating_block : String := "142857"
  let n : Nat := 1997
  have h : n % 6 = 5 := by decide
  have hn : n % 6 - 1 = 4 := by decide
  show repeating_block[hn] = '5' from by decide

end find_1997th_digit_in_decimal_expansion_of_1_div_7_l65_65340


namespace find_x_l65_65793

theorem find_x (x : ℝ) : (x + 3 * x + 1000 + 3000) / 4 = 2018 → x = 1018 :=
by 
  intro h
  sorry

end find_x_l65_65793


namespace arithmetic_mean_of_sequence_beginning_at_5_l65_65760

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

def sequence_sum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def arithmetic_mean (a d n : ℕ) : ℚ :=
  sequence_sum a d n / n

theorem arithmetic_mean_of_sequence_beginning_at_5 : 
  arithmetic_mean 5 1 60 = 34.5 :=
by
  sorry

end arithmetic_mean_of_sequence_beginning_at_5_l65_65760


namespace a_3_eq_5_a_4_eq_12_a_5_eq_29_recursive_relation_sum_of_squares_in_sequence_l65_65428

-- Definitions for sequence and conditions
def a : ℕ → ℝ
def det (a b c d : ℝ) : ℝ := a * d - b * c

axiom a_1 : a 1 = 1
axiom a_2 : a 2 = 2
axiom determinant_condition : ∀ (n : ℕ), n > 0 → det (a (n+2)) (a (n+1)) (a (n+1)) (a n) = (-1)^(n+1)

-- 1. Prove specific values of the sequence
theorem a_3_eq_5 : a 3 = 5 := sorry
theorem a_4_eq_12 : a 4 = 12 := sorry
theorem a_5_eq_29 : a 5 = 29 := sorry

-- 2. Prove the recursive relation
theorem recursive_relation (n : ℕ) (hn : n > 0) : a (n+2) = 2 * a (n+1) + a n := sorry

-- 3. Prove the sum of the squares of any two consecutive terms of the sequence is in the sequence
theorem sum_of_squares_in_sequence (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, a (n + 1) ^ 2 + a n ^ 2 = a k := sorry

end a_3_eq_5_a_4_eq_12_a_5_eq_29_recursive_relation_sum_of_squares_in_sequence_l65_65428


namespace bracelet_cost_l65_65734

theorem bracelet_cost (B : ℝ)
  (H1 : 5 = 5)
  (H2 : 3 = 3)
  (H3 : 2 * B + 5 + B + 3 = 20) : B = 4 :=
by
  sorry

end bracelet_cost_l65_65734


namespace penelope_starbursts_l65_65960

theorem penelope_starbursts (candies_ratio_mnm : ℕ) (candies_ratio_sb : ℕ) (candies_total_mnm : ℕ) (candies_total_sb : ℕ) :
  candies_ratio_mnm = 5 ∧
  candies_ratio_sb = 3 ∧
  candies_total_mnm = 25 →
  candies_total_sb = 15 :=
begin
  sorry
end

end penelope_starbursts_l65_65960


namespace sum_slope_y_intercept_eq_l65_65910

-- Definitions of points and midpoints
def point := (ℝ × ℝ)
def A : point := (0, 8)
def B : point := (0, 0)
def C : point := (10, 0)
def midpoint (P Q : point) : point := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
def D : point := midpoint A C

-- Definition of slope given two points
def slope (P Q : point) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

-- Line equation in point-slope form
def line (P Q : point) : point → ℝ := λ x y, y - P.2 = slope P Q * (x - P.1)

-- Slope of the line passing through C and D
def m : ℝ := slope C D

-- y-intercept of the line passing through C and D
def y_intercept : ℝ := (0, -m * C.1 + C.2).2

-- The sum of the slope and the y-intercept
def sum_slope_y_intercept : ℝ := m + y_intercept

-- Proof that the sum of the slope and the y-intercept is 36 / 5
theorem sum_slope_y_intercept_eq : sum_slope_y_intercept = 36 / 5 := by
  sorry

end sum_slope_y_intercept_eq_l65_65910


namespace laurie_possible_pairs_l65_65218

theorem laurie_possible_pairs :
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, let (x, y) := p in ∃ (A B C D : ℕ), 
      x = 10 * A + B ∧ y = 10 * C + D ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ 
      (10A + B) * (10C + D) = (10B + A) * (10D + C)) ∧ 
    pairs.length = 28 := sorry

end laurie_possible_pairs_l65_65218


namespace find_a_l65_65994

theorem find_a (a : ℝ) : 
  (∀ (f : ℝ → ℝ), (f = λ x, a * log x + x) → 
  (∃ (x : ℝ), f x = x ∧ f' x = 0)) → 
  a = -1 :=
by
  assume h
  have f_def : (λ x, a * log x + x) := sorry
  have f_deriv : (λ x, a / x + 1) := sorry
  have zero_at_one : (a / 1 + 1 = 0) := sorry
  show a = -1, from sorry

end find_a_l65_65994


namespace find_t_l65_65494

-- Define the problem conditions and the function
def f (x : ℝ) (t : ℝ) := (x^2 - 4) * (x - t)

-- Define the statement we want to prove
theorem find_t (t : ℝ) (h : deriv (λ x, f x t) (-1) = 0) : t = 1/2 := by
  sorry

end find_t_l65_65494


namespace hyperbola_t_square_l65_65003

theorem hyperbola_t_square (t : ℝ)
  (h1 : ∃ a : ℝ, ∀ (x y : ℝ), (y^2 / 4) - (5 * x^2 / 64) = 1 ↔ ((x, y) = (2, t) ∨ (x, y) = (4, -3) ∨ (x, y) = (0, -2))) :
  t^2 = 21 / 4 :=
by
  -- We need to prove t² = 21/4 given the conditions
  sorry

end hyperbola_t_square_l65_65003


namespace find_principal_sum_l65_65303

def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem find_principal_sum (CI SI : ℝ) (t : ℕ)
  (h1 : CI = 11730) 
  (h2 : SI = 10200) 
  (h3 : t = 2) :
  ∃ P r, P = 17000 ∧
  compound_interest P r t = CI ∧
  simple_interest P r t = SI :=
by
  sorry

end find_principal_sum_l65_65303


namespace rulers_left_in_drawer_l65_65668

theorem rulers_left_in_drawer (initial_rulers taken_rulers : ℕ) (h1 : initial_rulers = 46) (h2 : taken_rulers = 25) :
  initial_rulers - taken_rulers = 21 :=
by
  sorry

end rulers_left_in_drawer_l65_65668


namespace total_capacity_iv_bottle_l65_65716

-- Definitions of the conditions
def initial_volume : ℝ := 100 -- milliliters
def rate_of_flow : ℝ := 2.5 -- milliliters per minute
def observation_time : ℝ := 12 -- minutes
def empty_space_at_12_min : ℝ := 80 -- milliliters

-- Definition of the problem statement in Lean 4
theorem total_capacity_iv_bottle :
  initial_volume + rate_of_flow * observation_time + empty_space_at_12_min = 150 := 
by
  sorry

end total_capacity_iv_bottle_l65_65716


namespace rebus_no_solution_l65_65966

open Nat
open DigitFin

theorem rebus_no_solution (K U S Y : Fin 10) (h1 : K ≠ U) (h2 : K ≠ S) (h3 : K ≠ Y) (h4 : U ≠ S) (h5 : U ≠ Y) (h6 : S ≠ Y) :
  let KUSY := K.val * 1000 + U.val * 100 + S.val * 10 + Y.val
  let UKSY := U.val * 1000 + K.val * 100 + S.val * 10 + Y.val
  let UKSUS := U.val * 100000 + K.val * 10000 + S.val * 1000 + U.val * 100 + S.val * 10 + S.val
  KUSY + UKSY ≠ UKSUS := by
sorry

end rebus_no_solution_l65_65966


namespace ChipsEquivalence_l65_65457

theorem ChipsEquivalence
  (x y : ℕ)
  (h1 : y = x - 2)
  (h2 : 3 * x - 3 = 4 * y - 4) :
  3 * x - 3 = 24 :=
by
  sorry

end ChipsEquivalence_l65_65457


namespace bob_pennies_l65_65154

theorem bob_pennies (a b : ℕ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  have h3 : 4 * a - b = 5, from sorry,
  have h4 : b - 3 * a = 4, from sorry,
  have h5 : 4 * a - 3 * a = 9, from sorry,
  have h6 : a = 9, from sorry,
  have h7 : b + 1 = 36 - 4, from sorry,
  have h8 : b + 1 = 32, from sorry,
  have h9 : b = 31, from sorry,
  exact h9

end bob_pennies_l65_65154


namespace sixes_more_than_nines_l65_65782

theorem sixes_more_than_nines : 
  let count_digit (d n : ℕ) := ((List.range (n + 1)).map (λ x, (x.digits 10).count d)).sum 
  count_digit 6 625 - count_digit 9 625 = 27 :=
by
  sorry

end sixes_more_than_nines_l65_65782


namespace radius_of_fifth_sphere_l65_65165

noncomputable def cone_height : ℝ := 7
noncomputable def base_radius : ℝ := 7

def identical_spheres (r : ℝ) : Prop := 
  r > 0 ∧
  ∀ (O₁ O₂ O₃ O₄ O₅ : ℝ×ℝ×ℝ),
  let r₁ : ℝ := (O₁.1*O₁.1 + O₁.2*O₁.2 + O₁.3*O₁.3)^(1/2),
      r₂ : ℝ := (O₂.1*O₂.1 + O₂.2*O₂.2 + O₂.3*O₂.3)^(1/2),
      r₃ : ℝ := (O₃.1*O₃.1 + O₃.2*O₃.2 + O₃.3*O₃.3)^(1/2),
      r₄ : ℝ := (O₄.1*O₄.1 + O₄.2*O₄.2 + O₄.3*O₄.3)^(1/2),
      r₅ : ℝ := (O₅.1*O₅.1 + O₅.2*O₅.2 + O₅.3*O₅.3)^(1/2) in
  r₁ = r ∧ r₂ = r ∧ r₃ = r ∧ r₄ = r ∧ r₅ = 2 * r * (2^(1/2) - 1/2) + r 

-- Function to check the radius of the fifth sphere
theorem radius_of_fifth_sphere : 
  ∀ (x r : ℝ), cone_height = 7 ∧ base_radius = 7 ∧ identical_spheres r → 
  x = 2 * sqrt(2) - 1 :=
sorry

end radius_of_fifth_sphere_l65_65165


namespace raul_money_left_l65_65279

def initial_money : ℝ := 87
def comic_cost : ℝ := 4
def num_comics : ℕ := 8
def novel_cost : ℝ := 7
def num_novels : ℕ := 3
def magazine_cost : ℝ := 5.50
def num_magazines : ℕ := 2

def total_comic_cost : ℝ := num_comics * comic_cost
def total_novel_cost : ℝ := num_novels * novel_cost
def total_magazine_cost : ℝ := num_magazines * magazine_cost

def total_cost : ℝ := total_comic_cost + total_novel_cost + total_magazine_cost

def money_left : ℝ := initial_money - total_cost

theorem raul_money_left : money_left = 23 := by
  unfold money_left total_cost total_comic_cost total_novel_cost total_magazine_cost
  unfold initial_money comic_cost num_comics novel_cost num_novels magazine_cost num_magazines
  norm_num
  sorry

end raul_money_left_l65_65279


namespace dormitory_inequalities_l65_65328

theorem dormitory_inequalities (x : ℕ) :
  let total_students := 4 * x + 19 in
  let occupied_students := 6 * (x - 1) in
  let last_dormitory_students := total_students - occupied_students in
  1 ≤ last_dormitory_students ∧ last_dormitory_students ≤ 5 :=
by
  sorry

end dormitory_inequalities_l65_65328


namespace number_of_pairs_eq_two_l65_65533

theorem number_of_pairs_eq_two :
  {p : ℕ × ℕ | let x := p.1, y := p.2 in x > 0 ∧ y > 0 ∧ x^2 - y^2 = 91}.toFinset.card = 2 :=
sorry

end number_of_pairs_eq_two_l65_65533


namespace hall_length_l65_65733

theorem hall_length (L h : ℝ) (width volume : ℝ) 
  (h_width : width = 6) 
  (h_volume : L * width * h = 108) 
  (h_area : 12 * L = 2 * L * h + 12 * h) : 
  L = 6 := 
  sorry

end hall_length_l65_65733


namespace range_of_a_l65_65320

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, (x^2 + (a^2 + 1) * x + a - 2 = 0 ∧ y^2 + (a^2 + 1) * y + a - 2 = 0)
    ∧ x > 1 ∧ y < -1) ↔ (-1 < a ∧ a < 0) := sorry

end range_of_a_l65_65320


namespace solve_eq_l65_65644

theorem solve_eq : ∀ x : ℝ, 81 = 3 * (27)^(x-1) → x = 2 :=
by
  intro x h
  sorry

end solve_eq_l65_65644


namespace sqrt_eq_seven_l65_65293

variable (y : ℝ)

theorem sqrt_eq_seven (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) : 
  sqrt (64 - y^2) + sqrt (36 - y^2) = 7 :=
sorry

end sqrt_eq_seven_l65_65293


namespace max_square_side_length_l65_65667

-- Given: distances between consecutive lines in L and P
def distances_L : List ℕ := [2, 4, 6, 2, 4, 6, 2, 4, 6, 2, 4, 6, 2]
def distances_P : List ℕ := [3, 1, 2, 6, 3, 1, 2, 6, 3, 1, 2, 6, 3, 1]

-- Theorem: Maximum possible side length of a square with sides on lines L and P
theorem max_square_side_length : ∀ (L P : List ℕ), L = distances_L → P = distances_P → ∃ s, s = 40 :=
by
  intros L P hL hP
  sorry

end max_square_side_length_l65_65667


namespace total_amount_shared_l65_65261

theorem total_amount_shared 
  (Parker_share : ℤ)
  (ratio_2 : ℤ)
  (ratio_3 : ℤ)
  (total_parts : ℤ)
  (part_value : ℤ)
  (total_amount : ℤ) :
  Parker_share = 50 →
  ratio_2 = 2 →
  ratio_3 = 3 →
  total_parts = ratio_2 + ratio_3 →
  part_value = Parker_share / ratio_2 →
  total_amount = total_parts * part_value →
  total_amount = 125 :=
by 
  intros hParker_share hratio_2 hratio_3 htotal_parts hpart_value htotal_amount
  rw [hParker_share, hratio_2, hratio_3, htotal_parts, hpart_value, htotal_amount]
  sorry

end total_amount_shared_l65_65261


namespace quadrilateral_area_l65_65101

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ e = 1 / 2 ∧ (∃ x y : ℝ, x = -1 ∧ y = 3 / 2 
  ∧ (x^2 / a^2) + (y^2 / b^2) = 1) ∧ (4 = a^2) ∧ (3 = b^2) ∧ (x^2 / 4) + (y^2 / 3) = 1

theorem quadrilateral_area : Prop :=
  ∃ l m : ℝ, ∃ F₁ F₂ : ℝ × ℝ, F₁ = (-1, 0) ∧ F₂ = (1, 0) ∧ (y = x + m)
  ∧ (7 = m^2) ∧ ⟦|m| = √7
  ∧ (F₁M ⊥ l) ∧ (F₂N ⊥ l)
  -- The area of the quadrilateral F₁MNF₂
  ∧ ∃ (F₁M F₂M F₁N F₂N : ℝ), F₁M = |(F₁M)| ∧ F₂N = |(F₂N)| 
  ∧ S = 1 / 2 |F₁M^2 - F₂N^2| = √7


end quadrilateral_area_l65_65101


namespace correct_propositions_l65_65308

def Plane (α : Type) (β : Type) := α → β → Prop
def Line (α : Type) := α → Prop

variables (α β : Type) (a : α)

-- C1: A plane is perpendicular to another plane if there is a perpendicular line from one plane to the other.
def PerpendicularPlanes (p : Plane α β) (q : Plane α β) :=
  ∃ l : Line α, l ⊆ p ∧ l ⊆ q

-- C2: If two planes are parallel, there are infinitely many lines in one plane parallel to the other plane.
def ParallelPlanes (p : Plane α β) (q : Plane α β) :=
  ∃ l : Line α, ∀ x, x ∈ l → x ∈ p ∧ x ∈ q

-- C3: If there are infinitely many lines in one plane parallel to another plane, it does not necessarily mean the two planes are parallel.
def InfinitelyManyParallelLines (p : Plane α β) (q : Plane α β) :=
  ∃ l : Line α, l ⊆ p ∧ l ⊆ q ∧ ¬ ParallelPlanes p q

-- C4: A line a is parallel to plane α if there is a line in plane α parallel to line a.
def ParallelLinePlane (a : Line α) (p : Plane α β) :=
  ∃ l : Line α, l ⊆ p ∧ a = l

-- C5: If two lines are parallel, their projections in the same plane could be two points, which does not guarantee the projections are parallel.
def ParallelLinesProjections (l1 l2 : Line α) :=
  ∃ p : Plane α β, l1 ⊆ p ∧ l2 ⊆ p


-- The problem statement
theorem correct_propositions : 
  (PerpendicularPlanes α β → Q1 α β) ∧
  (PerpendicularPlanes α β → Q2 α β) ∧
  (PerpendicularPlanes α β → ¬ Q3 α β) ∧
  (PerpendicularPlanes α β → Q4 α β) :=
sorry

end correct_propositions_l65_65308


namespace wall_length_l65_65364

theorem wall_length (side_length_mirror : ℝ) (width_wall : ℝ) (area_wall : ℝ) :
  side_length_mirror = 34 → width_wall = 54 → 
  2 * (side_length_mirror * side_length_mirror) = area_wall →
  area_wall / width_wall = 43 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  sorry

end wall_length_l65_65364


namespace arithmetic_sequence_proof_l65_65860

noncomputable def a_n : ℕ → ℝ := sorry  -- Definition of sequence {a_n}
noncomputable def b_n : ℕ → ℝ := sorry  -- Definition of sequence {b_n}

def S_n (n : ℕ) : ℝ := (n / 2) * (a_n 1 + a_n n)  -- Sum of the first n terms of {a_n}
def T_n (n : ℕ) : ℝ := (n / 2) * (b_n 1 + b_n n)  -- Sum of the first n terms of {b_n}

axiom ratio_sn_tn (n : ℕ) (h : 0 < n) : S_n n / T_n n = (2 * n - 3) / (4 * n - 3)

theorem arithmetic_sequence_proof :
  (a_n 2 / (b_n 3 + b_n 13)) + (a_n 14 / (b_n 5 + b_n 11)) = 9 / 19 :=
begin
  -- proof will go here
  sorry
end

end arithmetic_sequence_proof_l65_65860


namespace distinct_integer_roots_l65_65235

-- Definitions of m and the polynomial equation.
def poly (m : ℤ) (x : ℤ) : Prop :=
  x^2 - 2 * (2 * m - 3) * x + 4 * m^2 - 14 * m + 8 = 0

-- Theorem stating that for m = 12 and m = 24, the polynomial has specific roots.
theorem distinct_integer_roots (m x : ℤ) (h1 : 4 < m) (h2 : m < 40) :
  (m = 12 ∨ m = 24) ∧ 
  ((m = 12 ∧ (x = 26 ∨ x = 16) ∧ poly m x) ∨
   (m = 24 ∧ (x = 52 ∨ x = 38) ∧ poly m x)) :=
by
  sorry

end distinct_integer_roots_l65_65235


namespace polynomial_division_l65_65935

def f (x : ℝ) := 3 * x^4 + 9 * x^3 - 7 * x^2 + 2 * x + 7
def d (x : ℝ) := x^2 + 2 * x - 3

theorem polynomial_division :
  ∃ q r : ℝ → ℝ, (f = λ x, q x * d x + r x) ∧ (∀ x, degree r < degree d) ∧ (q 2 + r (-2) = 72) :=
by
  sorry

end polynomial_division_l65_65935


namespace integers_with_inverses_mod_17_integers_without_inverses_mod_17_l65_65142

theorem integers_with_inverses_mod_17 :
  (∃ (n m : ℕ), n = 21 ∧ m = 2 ∧ p nat.card {x : ℕ | x ≤ 20 ∧ nat.gcd x 17 = 1} = n - m) :=
by
  sorry

theorem integers_without_inverses_mod_17 :
  (∃ (n m : ℕ), n = 21 ∧ m = 2 ∧ p nat.card {x : ℕ | x ≤ 20 ∧ nat.gcd x 17 ≠ 1} = m) :=
by 
  sorry

end integers_with_inverses_mod_17_integers_without_inverses_mod_17_l65_65142


namespace calculate_race_distances_l65_65175

variable {a b c t1 t2 : ℝ}
variable {d1 d2 : ℝ}

-- Conditions for the first race over distance d1
def first_race_conditions :=
  (d1 / a = (d1 - 30) / b) ∧
  (d1 / b = (d1 - 15) / c) ∧
  (d1 / a = (d1 - 42) / c)

-- Conditions for the second race over distance d2
def second_race_conditions :=
  (d2 / a = (d2 - 25) / b) ∧
  (d2 / b = (d2 - 20) / c) ∧
  (d2 / a = (d2 - 40) / c)

-- Proof problem statement: Given the conditions, d1 should be 150 and d2 should be 120
theorem calculate_race_distances (h1 : first_race_conditions) (h2 : second_race_conditions) : d1 = 150 ∧ d2 = 120 :=
by
  sorry

end calculate_race_distances_l65_65175


namespace fraction_girls_is_half_l65_65260

variable (A : ℝ) (hA : A > 0)

-- Condition: one sixth of the audience are adults
def fraction_adults : ℝ := 1 / 6

-- Condition: two fifths of the children are boys
def fraction_boys : ℝ := 2 / 5

-- Calculate the fraction of children in the audience
def fraction_children : ℝ := 1 - fraction_adults

-- Calculate the fraction of girls among the children
def fraction_girls_among_children : ℝ := 1 - fraction_boys

-- Calculate the fraction of girls in the total audience
def fraction_girls_in_audience : ℝ := fraction_girls_among_children * fraction_children

-- The theorem stating that the fraction of girls in the total audience is 1/2
theorem fraction_girls_is_half : fraction_girls_in_audience = 1 / 2 :=
by
  sorry

end fraction_girls_is_half_l65_65260


namespace half_angle_quadrant_l65_65543

theorem half_angle_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) :
    (∃ n : ℤ, (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)) :=
sorry

end half_angle_quadrant_l65_65543


namespace maximilian_annual_revenue_l65_65256

-- Define the number of units in the building
def total_units : ℕ := 100

-- Define the occupancy rate
def occupancy_rate : ℚ := 3 / 4

-- Define the monthly rent per unit
def monthly_rent : ℚ := 400

-- Calculate the number of occupied units
def occupied_units : ℕ := (occupancy_rate * total_units : ℚ).natAbs

-- Calculate the monthly rent revenue
def monthly_revenue : ℚ := occupied_units * monthly_rent

-- Calculate the annual rent revenue
def annual_revenue : ℚ := monthly_revenue * 12

-- Prove that the annual revenue is $360,000
theorem maximilian_annual_revenue : annual_revenue = 360000 := by
  sorry

end maximilian_annual_revenue_l65_65256


namespace BF_value_l65_65856

-- Define the conditions of the problem
variables {p : ℝ} {x y : ℝ}

noncomputable def parabola_condition := y^2 = 2 * p * x
noncomputable def focus_value := p / 2
noncomputable def point_A := (4 : ℝ, 0 : ℝ)
noncomputable def point_B := (p : ℝ, ℝ.sqrt 2 * p)

-- Point P on C
def P_on_C (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ x ≥ 0

-- Distance definition
noncomputable def distance_P_A (x y : ℝ) : ℝ := Real.sqrt ((x - 4)^2 + y^2)
noncomputable def distance_P_A_min (x y : ℝ) : Prop := distance_P_A x y = Real.sqrt 15

-- Main statement
theorem BF_value (h₁: 0 < p) (h₂: p < 4) (h₃: ∃ (x y : ℝ), P_on_C x y ∧ distance_P_A_min x y) :
  Real.sqrt (p^2 + (ℝ.sqrt 2 * p - p/2)^2) = 9 / 2 :=
sorry

end BF_value_l65_65856


namespace no_savings_by_buying_kit_l65_65384

-- Define the prices of individual filters
def price_filter1 := 12.45
def price_filter2 := 14.05
def price_filter3 := 11.50

-- Define the quantities of each type of filter
def qty_filter1 := 2
def qty_filter2 := 2
def qty_filter3 := 1

-- Calculate the total price of the individual filters
def total_price_individual := (price_filter1 * qty_filter1) + (price_filter2 * qty_filter2) + (price_filter3 * qty_filter3)

-- Define the price of the kit
def kit_price := 72.50

-- Calculate the amount saved by purchasing the kit
def amount_saved := total_price_individual - kit_price

theorem no_savings_by_buying_kit : amount_saved = -8 := by
  -- Just restate the goal
  sorry

end no_savings_by_buying_kit_l65_65384


namespace relative_position_of_I_M_O_l65_65339

-- Definition of the binary operation as given in the problem
def star (a b : ℂ) (z : ℂ) : ℂ :=
  z * (b - a) + a

-- Given conditions and question in Lean 4
theorem relative_position_of_I_M_O (I M O : ℂ) (ζ : ℂ) (hζ : ζ = complex.exp (complex.I * real.pi / 3)) :
  star I (star M O ζ) ζ = star (star O I ζ) M ζ →
  ∃ (i m o : ℂ) (hI : I = i) (hM : M = m) (hO : O = o),
    o = 0 ∧
    i = -ζ * m ∧
    ∀ (θ : ℝ), θ = (2 * real.pi) / 3 →
      ∃ (A B C : ℂ),
        A = I ∧ B = M ∧ C = O ∧
        (∥A - C∥ = ∥B - C∥) ∧
        (arg (B - C) - arg (A - C) = θ) :=
by sorry

end relative_position_of_I_M_O_l65_65339


namespace total_amount_shared_l65_65263

theorem total_amount_shared 
  (Parker_share : ℤ)
  (ratio_2 : ℤ)
  (ratio_3 : ℤ)
  (total_parts : ℤ)
  (part_value : ℤ)
  (total_amount : ℤ) :
  Parker_share = 50 →
  ratio_2 = 2 →
  ratio_3 = 3 →
  total_parts = ratio_2 + ratio_3 →
  part_value = Parker_share / ratio_2 →
  total_amount = total_parts * part_value →
  total_amount = 125 :=
by 
  intros hParker_share hratio_2 hratio_3 htotal_parts hpart_value htotal_amount
  rw [hParker_share, hratio_2, hratio_3, htotal_parts, hpart_value, htotal_amount]
  sorry

end total_amount_shared_l65_65263


namespace exists_n_for_each_k_l65_65073

noncomputable def f2 (n : ℕ) : ℕ :=
  (n.divisors.filter fun d => is_square d).card

noncomputable def f3 (n : ℕ) : ℕ :=
  (n.divisors.filter fun d => is_cube d).card

theorem exists_n_for_each_k (k : ℕ) (hk : k > 0) : ∃ (n : ℕ), f2 n / f3 n = k :=
begin
  sorry
end

end exists_n_for_each_k_l65_65073


namespace number_of_trees_in_park_l65_65008

-- Given conditions
def length_rect : ℝ := 1500
def width_rect : ℝ := 980.5
def diameter_semi_circle : ℝ := width_rect
def area_per_tree : ℝ := 23.5

-- Calculations based on the problem's conditions
def area_rect : ℝ := length_rect * width_rect
def radius_semi_circle : ℝ := diameter_semi_circle / 2
def area_semi_circle : ℝ := (real.pi * radius_semi_circle^2) / 2
def total_area : ℝ := area_rect + area_semi_circle
def num_trees : ℝ := total_area / area_per_tree

-- The proof problem statement
theorem number_of_trees_in_park : round num_trees = 78637 := 
by simp [num_trees, total_area, area_semi_circle, radius_semi_circle, area_rect, length_rect, width_rect, area_per_tree]; sorry

end number_of_trees_in_park_l65_65008


namespace average_nums_correct_l65_65298

def nums : List ℕ := [55, 48, 507, 2, 684, 42]

theorem average_nums_correct :
  (List.sum nums) / (nums.length) = 223 := by
  sorry

end average_nums_correct_l65_65298


namespace squares_not_all_congruent_l65_65360

theorem squares_not_all_congruent :
  ∃ S₁ S₂ : Square,
    (S₁ ≠ S₂) ∧ 
    (∀ S : Square, is_rectangle S ∧ θ S = 90 ∧ is_regular_polygon S ∧ is_rhombus S) → 
    ¬(∀ S₁ S₂ : Square, S₁ ≅ S₂) :=
by
  sorry

end squares_not_all_congruent_l65_65360


namespace hazel_lemonade_total_l65_65134

theorem hazel_lemonade_total 
  (total_lemonade: ℕ)
  (sold_construction: ℕ := total_lemonade / 2) 
  (sold_kids: ℕ := 18) 
  (gave_friends: ℕ := sold_kids / 2) 
  (drank_herself: ℕ := 1) :
  total_lemonade = 56 :=
  sorry

end hazel_lemonade_total_l65_65134


namespace cone_lateral_surface_area_l65_65831

-- Define the radius and slant height as given constants
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Definition of the lateral surface area
def lateral_surface_area (r l : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * r in
  (circumference * l) / 2

-- The proof problem statement in Lean 4
theorem cone_lateral_surface_area : lateral_surface_area radius slant_height = 15 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l65_65831


namespace walter_age_at_2003_l65_65407

theorem walter_age_at_2003 :
  ∀ (w : ℕ),
  (1998 - w) + (1998 - 3 * w) = 3860 → 
  w + 5 = 39 :=
by
  intros w h
  sorry

end walter_age_at_2003_l65_65407


namespace smallest_n_l65_65333

theorem smallest_n (n : ℕ) : 
  (25 * n = (Nat.lcm 10 (Nat.lcm 16 18)) → n = 29) :=
by sorry

end smallest_n_l65_65333


namespace interest_rate_of_first_account_l65_65006

theorem interest_rate_of_first_account (r : ℝ) 
  (h_inv1: 1000 * r) 
  (h_inv2: 1800 * 0.04) 
  (h_total: 1000 * r + 1800 * 0.04 = 92) :
  r = 0.02 :=
sorry

end interest_rate_of_first_account_l65_65006


namespace penelope_starbursts_l65_65959

theorem penelope_starbursts (candies_ratio_mnm : ℕ) (candies_ratio_sb : ℕ) (candies_total_mnm : ℕ) (candies_total_sb : ℕ) :
  candies_ratio_mnm = 5 ∧
  candies_ratio_sb = 3 ∧
  candies_total_mnm = 25 →
  candies_total_sb = 15 :=
begin
  sorry
end

end penelope_starbursts_l65_65959


namespace Petya_has_24_chips_l65_65459

noncomputable def PetyaChips (x y : ℕ) : ℕ := 3 * x - 3

theorem Petya_has_24_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) : PetyaChips x y = 24 :=
by
  sorry

end Petya_has_24_chips_l65_65459


namespace complement_A_in_U_l65_65524

def U : Set ℝ := {x : ℝ | x > 0}
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}
def AC : Set ℝ := {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (2 ≤ x)}

theorem complement_A_in_U : U \ A = AC := 
by 
  sorry

end complement_A_in_U_l65_65524


namespace segment_length_l65_65102

variables (a xM1 xM2 y yM)

def point_M := (xM1, yM)
def point_N := (-1, -2)

theorem segment_length (h1 : point_M = (a+3, a-4))
  (h2 : point_N = (-1, -2))
  (h3 : yM = -2) :
  abs ((a + 3) - (-1)) = 6 :=
by
  sorry

end segment_length_l65_65102


namespace angle_EGQ_l65_65282

-- Definitions of the conditions
def segment_EF_has_midpoint_G (EF G : Point) : Prop := exists F, midpoint F G EF
def segment_FG_has_midpoint_H (FG H : Point) : Prop := exists G, midpoint G H FG
def circles_with_diameters (EF FG : Segment) : Prop := true -- assuming existence of circles

-- The proof problem
theorem angle_EGQ (T GQ : Segment) (angle : ℝ) (EF G E Q : Point)
  (h1 : segment_EF_has_midpoint_G EF G)
  (h2 : segment_FG_has_midpoint_H G Q)
  (h3 : circles_with_diameters EF G)
  (h4 : splits_combined_area_in_two_equal_parts GQ EF G : Prop) :
  angle == 225.0 := 
sorry -- this statement is sufficient, and the proof is omitted.

end angle_EGQ_l65_65282


namespace g_is_odd_function_l65_65774

def g (x : ℝ) : ℝ := 1 / (3^x - 1) + 1 / 3

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g x := 
by 
  sorry

end g_is_odd_function_l65_65774


namespace perimeter_triangle_MNF_l65_65997

noncomputable def ellipse_perimeter := 
  let a := 1 / 2 in
  let fm := 2 * a in
  let fn := 2 * a in
  fm + fn

theorem perimeter_triangle_MNF :
  let a := 1 / 2 in
  let fm := 2 * a in
  let fn := 2 * a in
  4 * a = 2 :=
by
  sorry

end perimeter_triangle_MNF_l65_65997


namespace third_place_prize_correct_l65_65207

-- Define the conditions and formulate the problem
def total_amount_in_pot : ℝ := 210
def third_place_percentage : ℝ := 0.15
def third_place_prize (P : ℝ) : ℝ := third_place_percentage * P

-- The theorem to be proved
theorem third_place_prize_correct : 
  third_place_prize total_amount_in_pot = 31.5 := 
by
  sorry

end third_place_prize_correct_l65_65207


namespace polynomial_coeff_substitution_l65_65593

theorem polynomial_coeff_substitution :
  let a_0 := (2*(-1)-1)^5,
      a_1 := 5 * (2*(-1)-1)^4 * 2,
      a_2 := 10 * (2*(-1)-1)^3 * (2)^2,
      a_3 := 10 * (2*(-1)-1)^2 * (2)^3,
      a_4 := 5 * (2*(-1)-1) * (2)^4,
      a_5 := 1 * (2)^5 in
  a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = -243 :=
by
  let a_0 := (2*(-1)-1)^5
  let a_1 := 5 * (2*(-1)-1)^4 * 2
  let a_2 := 10 * (2*(-1)-1)^3 * (2)^2
  let a_3 := 10 * (2*(-1)-1)^2 * (2)^3
  let a_4 := 5 * (2*(-1)-1) * (2)^4
  let a_5 := 1 * (2)^5
  sorry

end polynomial_coeff_substitution_l65_65593


namespace min_period_monotonic_decreasing_l65_65749

theorem min_period_monotonic_decreasing :
  ∀ (x : ℝ), ∀ y : ℝ, 
  (y = abs (sin x) ∨ y = cos x ∨ y = tan x ∨ y = cos (x / 2)) →
  (∀ x, y = abs (sin x) → ∃ k : ℤ, y x = y (x + k * π)) ∧
  (∀ x ∈ Ioo (π / 2) π, y = abs (sin x) → ∀ x₁ x₂, x₁ < x₂ → y x₁ > y x₂) :=
begin
  sorry
end

end min_period_monotonic_decreasing_l65_65749


namespace complementary_angles_difference_l65_65314

def complementary_angles (θ1 θ2 : ℝ) : Prop :=
  θ1 + θ2 = 90

theorem complementary_angles_difference:
  ∀ (θ1 θ2 : ℝ), 
  (θ1 / θ2 = 4 / 5) → 
  complementary_angles θ1 θ2 → 
  abs (θ2 - θ1) = 10 :=
by
  sorry

end complementary_angles_difference_l65_65314


namespace ik_parallel_al_l65_65366

open EuclideanGeometry

theorem ik_parallel_al
  {ABC A1 B1 C1 M N T P Q L K I : Point}
  (h_incircle : is_incircle_of ABC I A1 B1 C1)
  (h_midpoints : midpoint M A B1 ∧ midpoint N A C1)
  (h_MN_meets_A1C1_at_T : collinear_points I M N T ∧ lies_on_line T A1 C1)
  (h_tangents : tangent_at_point TP T I ∧ tangent_at_point TQ T I)
  (h_PQ_meets_MN_at_L : collinear_points T P Q L ∧ collinear_points L M N)
  (h_B1C1_meets_PQ_at_K : collinear_points K B1 C1 ∧ collinear_points K P Q)
  : parallel (line_through_points I K) (line_through_points A L) :=
sorry

end ik_parallel_al_l65_65366


namespace negation_of_p_l65_65488

-- Definition of the proposition p
def p : Prop := ∃ x_0 : ℝ, x_0 ∈ set.Ioo 0 3 ∧ x_0 - 2 < real.log x_0

-- Negation of p
def neg_p : Prop := ∀ x : ℝ, x ∈ set.Ioo 0 3 → x - 2 ≥ real.log x

-- Statement to prove that neg_p is the negation of p
theorem negation_of_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_l65_65488


namespace delta_value_l65_65870

theorem delta_value : ∃ Δ : ℤ, 4 * (-3) = Δ + 5 ∧ Δ = -17 := 
by
  use -17
  sorry

end delta_value_l65_65870


namespace bart_trees_needed_l65_65785

-- Define the constants and conditions given
def firewood_per_tree : Nat := 75
def logs_burned_per_day : Nat := 5
def days_in_november : Nat := 30
def days_in_december : Nat := 31
def days_in_january : Nat := 31
def days_in_february : Nat := 28

-- Calculate the total number of days from November 1 through February 28
def total_days : Nat := days_in_november + days_in_december + days_in_january + days_in_february

-- Calculate the total number of pieces of firewood needed
def total_firewood_needed : Nat := total_days * logs_burned_per_day

-- Calculate the number of trees needed
def trees_needed : Nat := total_firewood_needed / firewood_per_tree

-- The proof statement
theorem bart_trees_needed : trees_needed = 8 := 
by
  -- Placeholder for the proof
  sorry

end bart_trees_needed_l65_65785


namespace complex_number_property_l65_65157

theorem complex_number_property (i : ℂ) (h : i^2 = -1) : (1 + i)^(20) - (1 - i)^(20) = 0 :=
by {
  sorry
}

end complex_number_property_l65_65157


namespace hyperbola_eccentricity_range_l65_65087

variable (a : ℝ) (e : ℝ)

theorem hyperbola_eccentricity_range (h : a > 1)
  (he : e^2 = 1 + ((a + 1) ^ 2) / (a ^ 2)) :
  sqrt 2 < e ∧ e < sqrt 5 :=
sorry

end hyperbola_eccentricity_range_l65_65087


namespace right_triangle_angle_l65_65892

open Real

theorem right_triangle_angle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h2 : c^2 = 2 * a * b) : 
  ∃ θ : ℝ, θ = 45 ∧ tan θ = a / b := 
by sorry

end right_triangle_angle_l65_65892


namespace molecular_weight_constant_l65_65343

-- Given condition
def molecular_weight (compound : Type) : ℝ := 260

-- Proof problem statement (no proof yet)
theorem molecular_weight_constant (compound : Type) : molecular_weight compound = 260 :=
by
  sorry

end molecular_weight_constant_l65_65343


namespace meeting_probability_l65_65392

-- Define the conditions as Lean definitions
def in_interval (t : ℝ) : Prop := 0 ≤ t ∧ t ≤ 2

def arrived_late (x y : ℝ) : Prop := abs (x - y) ≤ 0.5

-- Define the problem statement
theorem meeting_probability :
  ∫ z in 0..2, (∫ x in 0..(z - 0.5), ∫ y in 0..(z - 0.5), ∫ w in 0..(z - 0.5),
     if arrived_late x y ∧ arrived_late x w ∧ arrived_late y w then 1 else 0) / 16 = 1 / 192 :=
sorry

end meeting_probability_l65_65392


namespace negation_proposition_l65_65660

theorem negation_proposition:
  (¬ (∀ x : ℝ, x > 1 → sin x - 2^x < 0)) ↔ (∃ x : ℝ, x > 1 ∧ sin x - 2^x ≥ 0) := 
by
  sorry

end negation_proposition_l65_65660


namespace remainder_when_dividing_n_by_d_l65_65148

def n : ℕ := 25197638
def d : ℕ := 4
def r : ℕ := 2

theorem remainder_when_dividing_n_by_d :
  n % d = r :=
by
  sorry

end remainder_when_dividing_n_by_d_l65_65148


namespace parallel_line_passing_through_point_l65_65678

theorem parallel_line_passing_through_point :
  ∃ m b : ℝ, (∀ x y : ℝ, 4 * x + 2 * y = 8 → y = -2 * x + 4) ∧ b = 1 ∧ m = -2 ∧ b = 1 := by
  sorry

end parallel_line_passing_through_point_l65_65678


namespace problem1_problem2_l65_65133

-- Definitions for Problem 1
def m1 : ℝ × ℝ := (1/2, -sqrt 3 / 2)
def n1 (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

-- Problem 1 Proof Statement
theorem problem1 (x : ℝ) (h1 : m1.1 = (n1 x).1 * 2 ∧ m1.2 = (n1 x).2 * -2) :
  Real.tan x = -sqrt 3 := by
  sorry
  
-- Definitions for Problem 2
def m2 : ℝ × ℝ := (1/2, -sqrt 3 / 2)
def n2 (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

-- Problem 2 Proof Statement
theorem problem2 (x : ℝ) (h2 : m2.1 * (n2 x).1 + m2.2 * (n2 x).2 = 1/3)
  (h3 : 0 < x ∧ x < Real.pi / 2) :
  Real.cos x = (1 + 2 * sqrt 6) / 6 := by
  sorry

end problem1_problem2_l65_65133


namespace hazel_lemonade_total_l65_65135

theorem hazel_lemonade_total 
  (total_lemonade: ℕ)
  (sold_construction: ℕ := total_lemonade / 2) 
  (sold_kids: ℕ := 18) 
  (gave_friends: ℕ := sold_kids / 2) 
  (drank_herself: ℕ := 1) :
  total_lemonade = 56 :=
  sorry

end hazel_lemonade_total_l65_65135


namespace objective_function_range_l65_65486

noncomputable def feasible_region (A B C : ℝ × ℝ) := 
  let (x, y) := A
  let (x1, y1) := B 
  let (x2, y2) := C 
  {p : ℝ × ℝ | True} -- The exact feasible region description is not specified

theorem objective_function_range
  (A B C: ℝ × ℝ)
  (a b : ℝ)
  (x y : ℝ)
  (hA : A = (x, y))
  (hB : B = (1, 1))
  (hC : C = (5, 2))
  (h1 : a + b = 3)
  (h2 : 5 * a + 2 * b = 12) :
  let z := a * x + b * y
  3 ≤ z ∧ z ≤ 12 :=
by
  sorry

end objective_function_range_l65_65486


namespace murtha_pebbles_after_20_days_l65_65948

/- Define the sequence function for the pebbles collected each day -/
def pebbles_collected_day (n : ℕ) : ℕ :=
  if (n = 0) then 0 else 1 + pebbles_collected_day (n - 1)

/- Define the total pebbles collected by the nth day -/
def total_pebbles_collected (n : ℕ) : ℕ :=
  (n * (pebbles_collected_day n)) / 2

/- Define the total pebbles given away by the nth day -/
def pebbles_given_away (n : ℕ) : ℕ :=
  (n / 5) * 3

/- Define the net total of pebbles Murtha has on the nth day -/
def pebbles_net (n : ℕ) : ℕ :=
  total_pebbles_collected (n + 1) - pebbles_given_away (n + 1)

/- The main theorem about the pebbles Murtha has after the 20th day -/
theorem murtha_pebbles_after_20_days : pebbles_net 19 = 218 := 
  by sorry

end murtha_pebbles_after_20_days_l65_65948


namespace f_odd_and_solution_set_l65_65826

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2
  else if x < 0 then - log (-x) / log 2
  else 0

theorem f_odd_and_solution_set :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ 
  {x : ℝ | f (x) ≤ 1 / 2} = {x : ℝ | x ≤ -sqrt 2 / 2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ sqrt 2} :=
by 
  -- Proof goes here
  sorry

end f_odd_and_solution_set_l65_65826


namespace expand_product_l65_65789

theorem expand_product :
  (3 * x + 4) * (x - 2) * (x + 6) = 3 * x^3 + 16 * x^2 - 20 * x - 48 :=
by
  sorry

end expand_product_l65_65789


namespace measure_of_obtuse_angle_ADB_l65_65181

-- Definitions of our conditions for the right triangle ABC
variables (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables (angle_A angle_B : ℝ)

-- Given conditions
def right_triangle_ABC : Prop :=
  angle_A = 45 ∧ angle_B = 45 ∧ angle_A + angle_B = 90

-- Definition of the angle bisectors intersection at point D
def bisectors_meet_at_D : Prop :=
  ∃ D, true

-- Angle splitting property due to bisectors
def angle_bisectors_split : Prop :=
  angle_A / 2 = 22.5 ∧ angle_B / 2 = 22.5

-- Main theorem statement
theorem measure_of_obtuse_angle_ADB :
  right_triangle_ABC A B C ∧ bisectors_meet_at_D A B C D ∧ angle_bisectors_split A B →
  ∃ obtuse_angle_ADB, obtuse_angle_ADB = 270 :=
begin
  sorry
end

end measure_of_obtuse_angle_ADB_l65_65181


namespace calculate_annual_rent_l65_65257

-- Defining the conditions
def num_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4
def monthly_rent : ℚ := 400

-- Defining the target annual rent
def annual_rent (units : ℕ) (occupancy : ℚ) (rent : ℚ) : ℚ :=
  let occupied_units := occupancy * units
  let monthly_revenue := occupied_units * rent
  monthly_revenue * 12

-- Proof problem statement
theorem calculate_annual_rent :
  annual_rent num_units occupancy_rate monthly_rent = 360000 := by
  sorry

end calculate_annual_rent_l65_65257


namespace vertex_coloring_exists_l65_65011

-- Define the graph with given conditions
variables (V : Type) [Fintype V] [DecidableEq V]
variables (G : SimpleGraph V)
variables (h_card : Fintype.card V = 2004)
variables (h_deg : ∀ v : V, G.degree v ≤ 5)

-- The theorem statement
theorem vertex_coloring_exists :
  ∃ (A B : Finset V), A ∩ B = ∅ ∧ A ∪ B = Finset.univ ∧ (∃ E_cross, E_cross ⊆ G.edgeSet ∧ |E_cross| ≥ 3 / 5 * |G.edgeSet| ∧ ∀ e ∈ E_cross, (e.1 ∈ A ∧ e.2 ∈ B) ∨ (e.1 ∈ B ∧ e.2 ∈ A)) :=
sorry

end vertex_coloring_exists_l65_65011


namespace correct_proposition_is_B_l65_65510

-- Definitions of propositions
def propositionA (p q : Prop) : Prop := ¬(p ∧ q) → ¬p ∧ ¬q
def propositionB (a b : ℝ) : Prop := ¬(a > b → 2^a > 2^b - 1) = (a ≤ b → 2^a ≤ 2^b - 1)
def propositionC : Prop := ¬(∀ x : ℝ, x^2 + 1 ≥ 1) = ∃ x : ℝ, x^2 + 1 < 1
def propositionD (A B : ℝ) (α β R : ℝ) : Prop := (A > B → sin α > sin β) ∧ ¬(sin α > sin β → A > B)

-- Problem statement
theorem correct_proposition_is_B (p q : Prop) (a b A B α β R : ℝ) :
  propositionB a b :=
sorry

end correct_proposition_is_B_l65_65510


namespace statement_I_statement_II_statement_III_not_true_l65_65464

section
variable {x y : ℝ}

-- Definition of floor function
def floor_real (x : ℝ) : ℤ := Int.floor x

-- Statement I: Prove ⌊x + 2⌋ = ⌊x⌋ + 2 for all x
theorem statement_I (x : ℝ) : floor_real (x + 2) = floor_real x + 2 :=
sorry

-- Statement II: Prove ⌊x + y⌋ ≤ ⌊x⌋ + ⌊y⌋ for all x, y
theorem statement_II (x y : ℝ) : floor_real (x + y) ≤ floor_real x + floor_real y :=
sorry

-- Statement III: Disprove ⌊x^2⌋ = (⌊x⌋)^2 for all x
theorem statement_III_not_true (x : ℝ) : ¬ (floor_real (x^2) = (floor_real x)^2) :=
sorry

end

end statement_I_statement_II_statement_III_not_true_l65_65464


namespace part1_part2_l65_65610

variables {a b x : ℝ}

-- Define the function f(x)
def f (x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x ^ 2 - b * x

-- Condition that a ≠ 1
axiom a_ne_one : a ≠ 1

-- Derivative of function f
def f' (x : ℝ) : ℝ := (a / x) + (1 - a) * x - b

-- Proof statement for part 1
theorem part1 (h : f' 1 = 0) : b = 1 := by
  sorry

-- Proof statement for part 2
theorem part2 (hx0 : ∃ x0 ≥ 1, f x0 < a / (a - 1)) :
  a ∈ Set.Ioo (-Real.sqrt 2 - 1) (Real.sqrt 2 - 1) ∪ Set.Ioi 1 := by
  sorry

end part1_part2_l65_65610


namespace frustum_lateral_area_l65_65388

noncomputable def lateral_surface_area (r₁ r₂ h : ℝ) : ℝ :=
  let s := Real.sqrt (h^2 + (r₂ - r₁)^2)
  π * (r₁ + r₂) * s

theorem frustum_lateral_area :
  lateral_surface_area 5 8 9 = 39 * π * Real.sqrt 10 :=
by
  sorry

end frustum_lateral_area_l65_65388


namespace sum_of_6_consecutive_primes_product_l65_65111

-- Define the given number
def p : ℤ := 64312311692944269609355712372657

-- Define the condition that p is the product of 6 consecutive primes
def is_product_of_6_consecutive_primes (n : ℤ) : Prop :=
  ∃ a b c d e f : ℤ, Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧ Nat.Prime e ∧ Nat.Prime f ∧ 
  a = b + 2 ∧ b = c + 2 ∧ c = d + 2 ∧ d = e + 2 ∧ e = f + 2 ∧ n = a * b * c * d * e * f

-- Define the main theorem statement
theorem sum_of_6_consecutive_primes_product (h : is_product_of_6_consecutive_primes p) : 
  ∃ a b c d e f : ℤ, 
    Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧ Nat.Prime e ∧ Nat.Prime f ∧ 
    a = b + 2 ∧ b = c + 2 ∧ c = d + 2 ∧ d = e + 2 ∧ e = f + 2 ∧ p = a * b * c * d * e * f ∧ 
    a + b + c + d + e + f = 1200974 
:= sorry

end sum_of_6_consecutive_primes_product_l65_65111


namespace evaluate_at_x_value_minimum_value_l65_65853

noncomputable def f (x : ℝ) : ℝ :=
  Real.log (x / 8) / Real.log 2 * (Real.log (x / 2) / Real.log 4) + 1 / 2

theorem evaluate_at_x_value (x : ℝ) (m : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 2^m) (hm : 1 < m) :
  x = 4^(2 / 3) → f x = 5 / 6 :=
sorry

theorem minimum_value (x : ℝ) (m : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 2^m) (hm : 1 < m) :
  (∀ (y : ℝ), y = f x → 1 < m ≤ 2 → y = 1 / 2 * m^2 - 2 * m + 2) ∧
  (∀ (y : ℝ), y = f x → m > 2 → y = 0) :=
sorry

end evaluate_at_x_value_minimum_value_l65_65853


namespace A_in_second_quadrant_l65_65818

-- Define the coordinates of point A
def A_x : ℝ := -2
def A_y : ℝ := 3

-- Define the condition that point A lies in the second quadrant
def is_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State the theorem
theorem A_in_second_quadrant : is_second_quadrant A_x A_y :=
by
  -- The proof will be provided here.
  sorry

end A_in_second_quadrant_l65_65818


namespace woods_width_l65_65666

theorem woods_width (Area Length Width : ℝ) (hArea : Area = 24) (hLength : Length = 3) : 
  Width = 8 := 
by
  sorry

end woods_width_l65_65666


namespace seashells_total_l65_65637

theorem seashells_total :
    let Sam := 35
    let Joan := 18
    let Alex := 27
    Sam + Joan + Alex = 80 :=
by
    sorry

end seashells_total_l65_65637


namespace bob_pennies_l65_65156

theorem bob_pennies (a b : ℕ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  have h3 : 4 * a - b = 5, from sorry,
  have h4 : b - 3 * a = 4, from sorry,
  have h5 : 4 * a - 3 * a = 9, from sorry,
  have h6 : a = 9, from sorry,
  have h7 : b + 1 = 36 - 4, from sorry,
  have h8 : b + 1 = 32, from sorry,
  have h9 : b = 31, from sorry,
  exact h9

end bob_pennies_l65_65156


namespace average_rainfall_correct_l65_65886

-- Definitions based on given conditions
def total_rainfall : ℚ := 420 -- inches
def days_in_august : ℕ := 31
def hours_in_a_day : ℕ := 24

-- Defining total hours in August
def total_hours_in_august : ℕ := days_in_august * hours_in_a_day

-- The average rainfall in inches per hour
def average_rainfall_per_hour : ℚ := total_rainfall / total_hours_in_august

-- The statement to prove
theorem average_rainfall_correct :
  average_rainfall_per_hour = 420 / 744 :=
by
  sorry

end average_rainfall_correct_l65_65886


namespace range_of_x_range_of_a_l65_65244

-- Definitions based on Problem (I) and conditions
def p (x a : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - x - 6 <= 0) ∧ (x^2 + 3 * x - 10 > 0)

-- (I) Prove the range of x if a = 1 and p ∧ q
theorem range_of_x (x : ℝ) : p x 1 ∧ q x → 2 < x ∧ x < 3 :=
by sorry

-- Definitions based on Problem (II) and conditions
def sufficiently_not_necessary (p q : ℝ → Prop) : Prop := ∀ x, q x → p x ∧ ¬ (p x → q x)

-- (II) Prove the range of a if q is sufficient but not necessary for p
theorem range_of_a (a : ℝ) : sufficiently_not_necessary (p x a) q → 1 < a ∧ a ≤ 2 :=
by sorry

end range_of_x_range_of_a_l65_65244


namespace norm_of_vector_a_l65_65131

-- Given Conditions
variables (x : ℝ)
def a : ℝ × ℝ := (x, real.sqrt 3)
def b : ℝ × ℝ := (x, -real.sqrt 3)

-- Problem Statement
theorem norm_of_vector_a (h : (2 * a + b).fst * b.fst + (2 * a + b).snd * b.snd = 0) : 
  real.sqrt (a.fst ^ 2 + a.snd ^ 2) = 2 :=
sorry

end norm_of_vector_a_l65_65131


namespace total_painted_surface_area_l65_65751

-- Defining the conditions
def num_cubes := 19
def top_layer := 1
def middle_layer := 5
def bottom_layer := 13
def exposed_faces_top_layer := 5
def exposed_faces_middle_corner := 3
def exposed_faces_middle_center := 1
def exposed_faces_bottom_layer := 1

-- Question: How many square meters are painted?
theorem total_painted_surface_area : 
  let top_layer_area := top_layer * exposed_faces_top_layer
  let middle_layer_area := (4 * exposed_faces_middle_corner) + exposed_faces_middle_center
  let bottom_layer_area := bottom_layer * exposed_faces_bottom_layer
  top_layer_area + middle_layer_area + bottom_layer_area = 31 :=
by
  sorry

end total_painted_surface_area_l65_65751


namespace abs_diff_mn_sqrt_eight_l65_65236

theorem abs_diff_mn_sqrt_eight {m n p : ℝ} (h1 : m * n = 6) (h2 : m + n + p = 7) (h3 : p = 1) :
  |m - n| = 2 * Real.sqrt 3 :=
by
  sorry

end abs_diff_mn_sqrt_eight_l65_65236


namespace sum_of_divisors_330_l65_65684

theorem sum_of_divisors_330 : (∑ d in (finset.filter (λ d, 330 % d = 0) (finset.range (330 + 1))), d) = 864 :=
by {
  sorry
}

end sum_of_divisors_330_l65_65684


namespace number_of_dominoes_l65_65202

theorem number_of_dominoes (players : ℕ) (dominoes_per_player : ℕ) : players = 4 → dominoes_per_player = 7 → players * dominoes_per_player = 28 :=
by
  intros h_players h_dominoes_per_player
  rw [h_players, h_dominoes_per_player]
  norm_num
  sorry

end number_of_dominoes_l65_65202


namespace ratio_length_to_width_is_3_l65_65657

-- Define the conditions given in the problem
def area_of_garden : ℕ := 768
def width_of_garden : ℕ := 16

-- Define the length calculated from the area and width
def length_of_garden := area_of_garden / width_of_garden

-- Define the ratio to be proven
def ratio_of_length_to_width := length_of_garden / width_of_garden

-- Prove that the ratio is 3:1
theorem ratio_length_to_width_is_3 :
  ratio_of_length_to_width = 3 := by
  sorry

end ratio_length_to_width_is_3_l65_65657


namespace min_n_exist_sum_eq_30_l65_65394

theorem min_n_exist_sum_eq_30 :
  (∃ (n : ℕ) (a : ℕ → ℕ), 
  (∑ i in finset.range n, a i) = 2007 ∧ (∀ i < n, 0 < a i) ∧ 
  (∀ s t, s ≤ t → (∑ i in finset.range (t + 1), a i) - (∑ i in finset.range s, a i) ≠ 30)
  → 1018 < n) :=
by sorry

end min_n_exist_sum_eq_30_l65_65394


namespace total_interest_obtained_l65_65735

-- Define the interest rates and face values
def interest_16 := 0.16 * 100
def interest_12 := 0.12 * 100
def interest_20 := 0.20 * 100

-- State the theorem to be proved
theorem total_interest_obtained : 
  interest_16 + interest_12 + interest_20 = 48 :=
by
  sorry

end total_interest_obtained_l65_65735


namespace geometric_sequence_sum_l65_65572

noncomputable def sum_geometric_sequence (a: ℕ → ℤ) (q: ℚ) (n: ℕ) : ℚ :=
  a 0 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum :
  ∃ (a: ℕ → ℤ) (q: ℚ), 
    (a 7 = 1 ∧ q = 1/2) →
    sum_geometric_sequence a q 8 = 255 :=
begin
  sorry
end

end geometric_sequence_sum_l65_65572


namespace percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l65_65918

theorem percentage_increase_first_job :
  let old_salary := 65
  let new_salary := 70
  (new_salary - old_salary) / old_salary * 100 = 7.69 := by
  sorry

theorem percentage_increase_second_job :
  let old_salary := 120
  let new_salary := 138
  (new_salary - old_salary) / old_salary * 100 = 15 := by
  sorry

theorem percentage_increase_third_job :
  let old_salary := 200
  let new_salary := 220
  (new_salary - old_salary) / old_salary * 100 = 10 := by
  sorry

end percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l65_65918


namespace fifth_sphere_radius_l65_65167

noncomputable def cone_height : ℝ := 7
noncomputable def cone_base_radius : ℝ := 7

axiom r1 (r : ℝ) : 
  (r * (2 * real.sqrt 2 + 1) = cone_height) → 
  r = (2 * real.sqrt 2 - 1)

theorem fifth_sphere_radius (h : ℝ) (r : ℝ) (r2 : ℝ) :
  h = cone_height → r = (2 * real.sqrt 2 - 1) → r2 = r → r2 = (2 * real.sqrt 2 - 1) :=
by
  intros h_eq r_eq r2_eq
  rw [h_eq, r_eq, r2_eq]
  sorry

end fifth_sphere_radius_l65_65167


namespace positive_t_is_correct_l65_65079

def find_positive_t (t : ℝ) : Prop :=
  |3 + t * complex.I| = 7

theorem positive_t_is_correct : ∃ t > 0, find_positive_t t ∧ t = 2 * real.sqrt 10 :=
  sorry

end positive_t_is_correct_l65_65079


namespace isosceles_triangle_AFG_l65_65484

theorem isosceles_triangle_AFG
  (A B C D E F G : Point)
  (ABCD_isosceles_trapezoid : ABCD.is_isosceles_trapezoid A B C D)
  (incircle_BCD_touches_CD_at_E : (IncircleBCD A B C D).Touches E CD)
  (F_on_angle_bisector_DAC : F.on_angle_bisector (∠ DAC))
  (EF_perp_CD : EF.Perpendicular CD)
  (circumcircle_ACF_intersects_CD_at_G : (Circumcircle A C F).Intersects G CD) :
  AF = FG :=
by
  sorry

end isosceles_triangle_AFG_l65_65484


namespace calculate_expression_l65_65758

theorem calculate_expression : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end calculate_expression_l65_65758


namespace probability_walk_320_l65_65427

structure Condition where
  gates : Nat
  distance_between_gates : Nat
  max_distance : Nat

def probability_of_walking_320 (c : Condition) : ℚ := 
  let num_possible_situations := c.gates * (c.gates - 1)
  let num_valid_situations := 105 -- (calculated in steps)
  num_valid_situations / num_possible_situations

theorem probability_walk_320 (c : Condition) (h : c.gates = 15 ∧ c.distance_between_gates = 80 ∧ c.max_distance = 320) : 
  let p := probability_of_walking_320 c in
  let m := p.num in
  let n := p.denom in
  m = 1 ∧ n = 2 ∧ m + n = 3 :=
by
  sorry

end probability_walk_320_l65_65427


namespace new_concentration_of_mixture_l65_65745

theorem new_concentration_of_mixture :
  let v1 := 2
  let c1 := 0.25
  let v2 := 6
  let c2 := 0.40
  let V := 10
  let alcohol_amount_v1 := v1 * c1
  let alcohol_amount_v2 := v2 * c2
  let total_alcohol := alcohol_amount_v1 + alcohol_amount_v2
  let new_concentration := (total_alcohol / V) * 100
  new_concentration = 29 := 
by
  sorry

end new_concentration_of_mixture_l65_65745


namespace ChipsEquivalence_l65_65455

theorem ChipsEquivalence
  (x y : ℕ)
  (h1 : y = x - 2)
  (h2 : 3 * x - 3 = 4 * y - 4) :
  3 * x - 3 = 24 :=
by
  sorry

end ChipsEquivalence_l65_65455


namespace limit_seq_l65_65772

noncomputable def sequence : ℕ → ℝ
| 0       => sqrt 5
| (n + 1) => (sequence n) ^ 2 - 2

theorem limit_seq : 
  tendsto (λ n, (∏ i in finset.range (n+1), sequence i) / (sequence (n + 1))) at_top (𝓝 1) :=
by
  sorry

end limit_seq_l65_65772


namespace sum_first_three_coefficients_expansion_l65_65693

theorem sum_first_three_coefficients_expansion : 
  (∑ k in finset.range 3, nat.choose 7 k) = 29 :=
by sorry

end sum_first_three_coefficients_expansion_l65_65693


namespace combined_age_in_years_l65_65020

theorem combined_age_in_years (years : ℕ) (adam_age : ℕ) (tom_age : ℕ) (target_age : ℕ) :
  adam_age = 8 → tom_age = 12 → target_age = 44 → (adam_age + tom_age) + 2 * years = target_age → years = 12 :=
by
  intros h_adam h_tom h_target h_combined
  rw [h_adam, h_tom, h_target] at h_combined
  linarith

end combined_age_in_years_l65_65020


namespace Phil_earns_50_percent_less_than_Mike_l65_65615

open Real

def Mike_earnings : ℝ := 12
def Phil_earnings : ℝ := 6
def earnings_difference := Mike_earnings - Phil_earnings
def percentage_difference := (earnings_difference / Mike_earnings) * 100

theorem Phil_earns_50_percent_less_than_Mike : percentage_difference = 50 := by
  sorry

end Phil_earns_50_percent_less_than_Mike_l65_65615


namespace collinear_iff_lambda_mu_eq_one_l65_65506

variables {R : Type*} [Field R]
variables {a b : R} {λ μ : R}

-- Non-collinearity of vectors
axiom vector_non_collinear (a b : R) : a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b

-- Definition of vectors AB and AC
def vec_AB (a b λ : R) : R := λ * a + b
def vec_AC (a b μ : R) : R := a + μ * b

-- Definition of collinearity
def collinear {R : Type*} [Field R] (u v : R) : Prop :=
  ∃ k : R, u = k * v

-- Prove that points A, B, and C are collinear iff λμ = 1
theorem collinear_iff_lambda_mu_eq_one (h_collinear : collinear (vec_AB a b λ) (vec_AC a b μ)) : λ * μ = 1 :=
sorry

end collinear_iff_lambda_mu_eq_one_l65_65506


namespace compute_ratio_l65_65567

noncomputable def areas_ratio (P Q R S T F E : Type)
  [HasSub R] [HasAdd R] [HasMul R] [HasDiv R] [Zero R] [One R] [LinearOrder R] [Field R] 
  (PQ PR PS RT : R)
  (h1 : PQ = 130)
  (h2 : PR = 130)
  (h3 : PS = 45)
  (h4 : RT = 90)
  (RTF_area SBE_area : R) : Prop :=
  RTF_area / SBE_area = 9 / 44

theorem compute_ratio (P Q R S T F E : Type)
  [HasSub R] [HasAdd R] [HasMul R] [HasDiv R] [Zero R] [One R] [LinearOrder R] [Field R] 
  (PQ PR PS RT : R)
  (h1 : PQ = 130)
  (h2 : PR = 130)
  (h3 : PS = 45)
  (h4 : RT = 90)
  (RTF_area SBE_area : R) : 
  areas_ratio P Q R S T F E PQ PR PS RT h1 h2 h3 h4 RTF_area SBE_area := 
sorry

end compute_ratio_l65_65567


namespace inequality_proof_l65_65278

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) :
  ∑ i in Finset.range n, (i + 1 : ℕ) * a i ≤ Nat.choose n 2 + ∑ i in Finset.range n, a i ^ (i + 1) :=
  sorry

end inequality_proof_l65_65278


namespace eval_expression_l65_65424

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

theorem eval_expression : 2 * f 3 + 3 * f (-3) = 147 := by
  sorry

end eval_expression_l65_65424


namespace colton_burger_shares_l65_65418

theorem colton_burger_shares (foot_in_inches : ℝ) (share_fraction : ℝ):
  foot_in_inches = 12 → share_fraction = 2 / 5 → 
  let brother_share := share_fraction * foot_in_inches in
  let colton_share := foot_in_inches - brother_share in
  brother_share = 4.8 ∧ colton_share = 7.2 := 
by
  intros h1 h2
  let brother_share := share_fraction * foot_in_inches
  let colton_share := foot_in_inches - brother_share
  have brother_share_val : brother_share = 4.8 := by
    calc
      brother_share = (2 / 5) * 12 : by rw [h1, h2]
      ... = 4.8 : by norm_num
  have colton_share_val : colton_share = 7.2 := by
    calc 
      colton_share = 12 - 4.8 : by rw [h1, brother_share_val]
      ... = 7.2 : by norm_num
  show brother_share = 4.8 ∧ colton_share = 7.2 from ⟨brother_share_val, colton_share_val⟩

end colton_burger_shares_l65_65418


namespace MrMartin_bagels_l65_65617

-- Define the conditions
def C (cost_of_coffee : Real) (cost_of_bagel : Real) (total_cost : Real) := 3 * cost_of_coffee + 2 * cost_of_bagel = total_cost
def cost_Mrs_Martin := 12.75
def cost_of_bagel := 1.5
def B (cost_of_coffee : Real) (cost_of_bagel : Real) (bagels : Nat) := 2 * cost_of_coffee + bagels * cost_of_bagel = 14.00

-- Prove that Mr. Martin bought 5 bagels
theorem MrMartin_bagels:
  ∃ (x : Nat) (C_value : Real), C C_value cost_of_bagel cost_Mrs_Martin ∧ B C_value cost_of_bagel x ∧ x = 5 :=
by sorry

end MrMartin_bagels_l65_65617


namespace non_visible_dots_l65_65675

-- Define the configuration of the dice
def total_dots_on_one_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
def total_dots_on_two_dice : ℕ := 2 * total_dots_on_one_die
def visible_dots : ℕ := 2 + 3 + 5

-- The statement to prove
theorem non_visible_dots : total_dots_on_two_dice - visible_dots = 32 := by sorry

end non_visible_dots_l65_65675


namespace arc_length_ln_x_l65_65761

-- Define the function y = ln x
def f (x : ℝ) : ℝ := Real.log x

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 1 / x

-- Define the integral expression for arc length L
noncomputable def arc_length (a b : ℝ) : ℝ :=
  ∫ x in a..b, (Real.sqrt (1 + (f' x)^2))

-- The main theorem statement
theorem arc_length_ln_x :
  arc_length (Real.sqrt 3) (Real.sqrt 15) = (1 / 2) * Real.log (9 / 5) + 2 :=
sorry

end arc_length_ln_x_l65_65761


namespace sum_f_2008_l65_65373

noncomputable def f : ℕ × ℤ → ℤ
| (0, 0) := 1
| (0, 1) := 1
| (0, k) := if k ≠ 0 ∧ k ≠ 1 then 0 else 1
| (n + 1, k) := f (n, k) + f (n, k - 2 * (n + 1))

theorem sum_f_2008 :
  ∑ k in finset.range (nat.choose 2009 2 + 1), f (2008, k) = 2 ^ 2008 :=
sorry

end sum_f_2008_l65_65373


namespace classA_classC_ratio_l65_65416

-- Defining the sizes of classes B and C as given in conditions
def classB_size : ℕ := 20
def classC_size : ℕ := 120

-- Defining the size of class A based on the condition that it is twice as big as class B
def classA_size : ℕ := 2 * classB_size

-- Theorem to prove that the ratio of the size of class A to class C is 1:3
theorem classA_classC_ratio : classA_size / classC_size = 1 / 3 := 
sorry

end classA_classC_ratio_l65_65416


namespace can_reach_any_city_l65_65730

-- Define the parameters given in the problem
def number_of_cities : ℕ := 53
def total_roads : ℕ := 312
def max_degree : ℕ := 12
def toll_per_road : ℕ := 10
def budget : ℕ := 120

-- Define the statement to be proven
theorem can_reach_any_city (G : SimpleGraph (Fin number_of_cities))
  (h1 : Finite (Fin number_of_cities))
  (h2 : G.IsConnected)
  (h3 : G.EdgeCount = total_roads)
  (h4 : ∀ (v : Fin number_of_cities), G.degree v ≤ max_degree) :
  ∀ (u v : Fin number_of_cities), ∃ (p : G.Walk u v), p.length * toll_per_road ≤ budget :=
sorry


end can_reach_any_city_l65_65730


namespace area_of_region_correct_l65_65827

noncomputable def area_of_region (P : Point) (α : Plane) (h : distance_from_point_to_plane P α = sqrt 3)
    (Q : Point) : real :=
  if (Q ∈ α ∧ 30 <= angle_between_line_and_plane (line PQ) α ∧ angle_between_line_and_plane (line PQ) α <= 60)
  then 8 * π
  else 0

theorem area_of_region_correct (P : Point) (α : Plane) (h : distance_from_point_to_plane P α = sqrt 3) :
  ∀ Q : Point, (Q ∈ α ∧ 30 <= angle_between_line_and_plane (line PQ) α ∧ angle_between_line_and_plane (line PQ) α <= 60) →
              area_of_region P α h Q = 8 * π :=
sorry

end area_of_region_correct_l65_65827


namespace maximilian_annual_revenue_l65_65255

-- Define the number of units in the building
def total_units : ℕ := 100

-- Define the occupancy rate
def occupancy_rate : ℚ := 3 / 4

-- Define the monthly rent per unit
def monthly_rent : ℚ := 400

-- Calculate the number of occupied units
def occupied_units : ℕ := (occupancy_rate * total_units : ℚ).natAbs

-- Calculate the monthly rent revenue
def monthly_revenue : ℚ := occupied_units * monthly_rent

-- Calculate the annual rent revenue
def annual_revenue : ℚ := monthly_revenue * 12

-- Prove that the annual revenue is $360,000
theorem maximilian_annual_revenue : annual_revenue = 360000 := by
  sorry

end maximilian_annual_revenue_l65_65255


namespace f_28_l65_65094

def f1 (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)
def f : ℕ → (ℝ → ℝ)
| 0     := id
| (n+1) := f1 ∘ f n

theorem f_28 (x : ℝ) : f 28 x = 1 / (1 - x) := by
  sorry

end f_28_l65_65094


namespace similar_parabolas_l65_65276

theorem similar_parabolas :
  (∀ (a b : ℝ), b = 2 * a^2 → ∃ (x y : ℝ), y = x^2 ∧ x = 2 * a ∧ y = 2 * b) :=
by
  assume a b : ℝ,
  assume h : b = 2 * a^2,
  existsi (2 * a),
  existsi (2 * b),
  split,
  {
    calc
      2 * b = 2 * (2 * a^2) : by rw [h]
      ...   = (2 * a)^2 : by ring,
  },
  split,
  { refl },
  { exact h }

end similar_parabolas_l65_65276


namespace symmetric_graphs_implies_range_a_l65_65160

def f (x a : ℝ) : ℝ := x^2 + Real.log (x + a)
def g (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2

theorem symmetric_graphs_implies_range_a (a : ℝ) :
  (∀ x < 0, g x = f (-x) a) → a ∈ Set.Iio (Real.sqrt Real.exp 1) :=
by
  intro h
  sorry

end symmetric_graphs_implies_range_a_l65_65160


namespace race_track_width_l65_65312

noncomputable def width_of_race_track (C_inner : ℝ) (r_outer : ℝ) : ℝ :=
  let r_inner := C_inner / (2 * Real.pi)
  r_outer - r_inner

theorem race_track_width : 
  width_of_race_track 880 165.0563499208679 = 25.0492072460867 :=
by
  sorry

end race_track_width_l65_65312


namespace geometric_sequence_fourth_term_l65_65108

theorem geometric_sequence_fourth_term (x : ℝ) (h1 : (2 * x + 2) ^ 2 = x * (3 * x + 3))
  (h2 : x ≠ -1) : (3*x + 3) * (3/2) = -27/2 :=
by
  sorry

end geometric_sequence_fourth_term_l65_65108


namespace number_of_valid_arrangements_l65_65903

def is_valid_triplet (a b c : ℕ) : Prop :=
  (a * c - b ^ 2) % 7 = 0

def is_valid_sequence (s : List ℕ) : Prop :=
  ∀ (i : ℕ), i + 2 < s.length → is_valid_triplet (s.get i) (s.get (i + 1)) (s.get (i + 2))

theorem number_of_valid_arrangements : 
  (List.permutations [1, 2, 3, 4, 5, 6]).countp is_valid_sequence = 12 := 
sorry

end number_of_valid_arrangements_l65_65903


namespace analytical_expression_range_of_a_l65_65515

noncomputable def f (x φ : Real) := sqrt 3 * sin (2*x + φ) + cos(2*x + φ)

theorem analytical_expression (φ : Real) (hφ : |φ| < π/2) :
  f x φ = 2 * sin (2 * x - π / 6) := sorry

noncomputable def g (x : Real) := 2 * sin (2*x - π/6)

theorem range_of_a {a : Real} :
  (∃ x₁ x₂ : Real, (f x₁ (-π/3) = a) ∧ (f x₂ (-π/3) = a) ∧ x₁ ≠ x₂ ∧ 
                    x₁ ∈ [π/6, 5π/12] ∧ x₂ ∈ [π/6, 5π/12]) → 
  a ∈ [sqrt 3, 2) := sorry

end analytical_expression_range_of_a_l65_65515


namespace intersect_count_l65_65980

noncomputable def f (x : ℝ) : ℝ := sorry  -- Function f defined for all real x.
noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Inverse function of f.

theorem intersect_count : 
  (∃ a b : ℝ, a ≠ b ∧ f (a^2) = f (a^3) ∧ f (b^2) = f (b^3)) :=
by sorry

end intersect_count_l65_65980


namespace monotonicity_f_n_2_f_le_x_minus_1_l65_65117

-- Define the function f(x) and the specific case when n=2
def f (x : ℝ) (n : ℕ) (a : ℝ) : ℝ :=
  (1 / (1 - x) ^ n) + a * real.log (x - 1)

-- Prove the monotonicity for the case when n = 2
theorem monotonicity_f_n_2 (a : ℝ) :
  ∀ x: ℝ, x > 1 → 
  ((a ≤ 0 → ∀ x > 1, deriv (λ x, f x 2 a) x < 0) ∧
   (a > 0 → (∀ x ∈ Ioo 1 (1+sqrt(2/a)), deriv (λ x, f x 2 a) x < 0) ∧ 
             (∀ x ∈ Ioo (1+sqrt(2/a)) ⊤, deriv (λ x, f x 2 a) x > 0 ))) :=
sorry

-- Prove f(x) <= x-1 for a = 1 and ∀ n ∈ ℕ*, x ≥ 2
theorem f_le_x_minus_1 (n : ℕ) (hn : n > 0) :
  ∀ x : ℝ, x ≥ 2 → f x n 1 ≤ x - 1 :=
sorry

end monotonicity_f_n_2_f_le_x_minus_1_l65_65117


namespace parallelogram_perimeter_l65_65705

-- Define the necessary conditions based on the problem statement
variables {A B C D : Type*} [metric_space A]

structure parallelogram (A B C D : A) : Prop :=
(diagonal_eq : dist B D = 2)
(equilateral_triangle_BCD : dist B C = dist B D ∧ dist B D = dist C D)

noncomputable def perimeter_of_parallelogram (A B C D : A) [parallelogram A B C D] : ℝ :=
dist A B + dist B C + dist C D + dist D A

theorem parallelogram_perimeter (A B C D : A)
  [h : parallelogram A B C D] :
  perimeter_of_parallelogram A B C D = 8 :=
sorry

end parallelogram_perimeter_l65_65705


namespace side_length_of_square_l65_65701

theorem side_length_of_square (A : ℝ) (hA : A = real.sqrt 900) : real.sqrt A = 30 := 
by 
  sorry

end side_length_of_square_l65_65701


namespace correct_statements_are_two_three_four_l65_65400

theorem correct_statements_are_two_three_four :
  (∀ f x₀, (∃ x₀, has_extremum_at f x₀ → deriv f x₀ = 0)) ∧
  (∀ specific general, (inductive_reasoning specific general ∧ deductive_reasoning general specific)) ∧
  (∀ proof_problem, (synthetic_method proof_problem cause_to_effect ∧ analytic_method proof_problem effect_to_cause))
:=
sorry

-- Definitions to complete this proof
def has_extremum_at (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ (y : ℝ), (f(x) ≥ y → x = x₀ to y ≠ f(x₀)) ∨ (f(x) ≤ y → x = x₀ to y ≠ f(x₀))

def inductive_reasoning (specific : α) (general : α → Prop) : Prop := 
  specific → general specific

def deductive_reasoning (general : α → Prop) (specific : α) : Prop := 
  general specific → specific

def synthetic_method (proof_problem : Prop) (from : Prop) (to : Prop) : Prop := from → to

def analytic_method (proof_problem : Prop) (from : Prop) (to : Prop) : Prop := to → from

end correct_statements_are_two_three_four_l65_65400


namespace weeks_to_buy_bicycle_l65_65411

-- Definitions based on problem conditions
def hourly_wage : Int := 5
def hours_monday : Int := 2
def hours_wednesday : Int := 1
def hours_friday : Int := 3
def weekly_hours : Int := hours_monday + hours_wednesday + hours_friday
def weekly_earnings : Int := weekly_hours * hourly_wage
def bicycle_cost : Int := 180

-- Statement of the theorem to prove
theorem weeks_to_buy_bicycle : ∃ w : Nat, w * weekly_earnings = bicycle_cost :=
by
  -- Since this is a statement only, the proof is omitted
  sorry

end weeks_to_buy_bicycle_l65_65411


namespace max_sides_of_convex_polygon_with_three_obtuse_angles_l65_65729

theorem max_sides_of_convex_polygon_with_three_obtuse_angles (n : ℕ) :
  (∃ (A B C : ℝ), 90 < A ∧ A < 180 ∧ 90 < B ∧ B < 180 ∧ 90 < C ∧ C < 180 ∧
  let sum_obtuse := A + B + C in 
  (n > 3 ∧ (n-2) * 180 < 540 + (n-3) * 90)) → n ≤ 6 :=
by 
  sorry

end max_sides_of_convex_polygon_with_three_obtuse_angles_l65_65729


namespace return_amount_is_correct_l65_65214

-- Define the borrowed amount and the interest rate
def borrowed_amount : ℝ := 100
def interest_rate : ℝ := 10 / 100

-- Define the condition of the increased amount
def increased_amount : ℝ := borrowed_amount * interest_rate

-- Define the total amount to be returned
def total_amount : ℝ := borrowed_amount + increased_amount

-- Lean 4 statement to prove
theorem return_amount_is_correct : total_amount = 110 := by
  -- Borrowing amount definition
  have h1 : borrowed_amount = 100 := rfl
  -- Interest rate definition
  have h2 : interest_rate = 10 / 100 := rfl
  -- Increased amount calculation
  have h3 : increased_amount = borrowed_amount * interest_rate := rfl
  -- Expanded calculation of increased_amount
  have h4 : increased_amount = 100 * (10 / 100) := by rw [h1, h2]
  -- Simplify the increased_amount
  have h5 : increased_amount = 10 := by norm_num [h4]
  -- Total amount calculation
  have h6 : total_amount = borrowed_amount + increased_amount := rfl
  -- Expanded calculation of total_amount
  have h7 : total_amount = 100 + 10 := by rw [h1, h5]
  -- Simplify the total_amount
  show 100 + 10 = 110 from rfl
  sorry

end return_amount_is_correct_l65_65214


namespace common_root_is_1_neg1_i_negi_l65_65294

open Complex

theorem common_root_is_1_neg1_i_negi (a b c d k : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (a * k^3 + b * k^2 + c * k + d = 0) → (b * k^3 + c * k^2 + d * k + a = 0) →
  k = 1 ∨ k = -1 ∨ k = Complex.i ∨ k = -Complex.i :=
by
  sorry

end common_root_is_1_neg1_i_negi_l65_65294


namespace math_problem_l65_65907

-- Definitions related to Planes, Points, Lines, and Distances
variables {Plane Point Line : Type}
variable (α β : Plane)
variable (P1 P2 A B C : Point)
variable (l m : Line)
variable (dist : Point → Plane → Real)
variable (collinear : Point → Point → Point → Prop)
variable (proj : Point → Line → Line)
variable (skew : Line → Line → Prop)

-- Conditions translated to Lean
def condition1 := P1 ∉ α → P2 ∉ α → ¬(∃! β : Plane, (P1 ∈ β ∧ P2 ∈ β ∧ β ⊥ α))
def condition2 := A ∈ β → B ∈ β → C ∈ β → ¬ collinear A B C → dist A α = dist B α → dist C α → α ∥ β
def condition3 := (∀ k : Line, k ⊥ α ∧ k ⊥ l) → l ⊥ α
def condition4 := skew l m → (∀ p ∈ α, proj(p, l) ∥ proj(p, m)) → False

-- Statement of the problem
theorem math_problem : condition1 α P1 P2 ∧ condition2 α β A B C dist collinear ∧ condition3 α l ∧ condition4 α l m proj skew → num_true_propositions = 0 :=
by
  sorry

end math_problem_l65_65907


namespace total_profit_l65_65019

/-- Given the investments of A, B, and C, and the share of A in the profit,
    prove that the total profit is Rs. 14200. -/
theorem total_profit (A_investment B_investment C_investment A_share total_profit : ℕ)
  (hA_investment : A_investment = 6300)
  (hB_investment : B_investment = 4200)
  (hC_investment : C_investment = 10500)
  (hA_share : A_share = 4260)
  (h_ratio : (A_investment + B_investment + C_investment = 21000)) 
  : total_profit = 14200 :=
begin
  sorry
end

end total_profit_l65_65019


namespace length_PQ_correct_l65_65566

noncomputable def parametric_circle : (ℝ → ℝ × ℝ) :=
  λ φ, (1 + Real.cos φ, Real.sin φ)

noncomputable def polar_line : (ℝ → ℝ × ℝ) :=
  λ θ, (3 / (2 * Real.sin (θ + Real.pi / 3)), θ)

noncomputable def rho_P := 1
noncomputable def theta_P := Real.pi / 3

noncomputable def rho_Q := 3
noncomputable def theta_Q := Real.pi / 3

def length_PQ : ℝ :=
  abs (rho_P - rho_Q)

theorem length_PQ_correct :
  length_PQ = 2 :=
  by
    unfold length_PQ rho_P theta_P rho_Q theta_Q ;
    simp ;
    linarith ;
    sorry -- leave the detailed proof steps for further elaboration if needed.

end length_PQ_correct_l65_65566


namespace hours_per_day_l65_65728

theorem hours_per_day (initial_employees : ℕ) (hourly_rate : ℕ) (work_days_per_week : ℕ) 
  (weeks_per_month : ℕ) (additional_employees : ℕ) (total_monthly_pay_August : ℕ) 
  (total_employees_August : ℕ) : ℕ :=
  let h := 10 in
  initial_employees = 500 ∧
  hourly_rate = 12 ∧
  work_days_per_week = 5 ∧
  weeks_per_month = 4 ∧
  additional_employees = 200 ∧
  total_employees_August = 700 ∧
  total_monthly_pay_August = 1680000 → 
  (total_monthly_pay_August = total_employees_August * hourly_rate * h * 
  work_days_per_week * weeks_per_month) → h = 10 := 
by {
  intros _ _ _ _ _ _ _,
  sorry
}

end hours_per_day_l65_65728


namespace tree_original_height_l65_65922

theorem tree_original_height (current_height_in: ℝ) (growth_percentage: ℝ)
  (h1: current_height_in = 180) (h2: growth_percentage = 0.50) :
  ∃ (original_height_ft: ℝ), original_height_ft = 10 :=
by
  have original_height_in := current_height_in / (1 + growth_percentage)
  have original_height_ft := original_height_in / 12
  use original_height_ft
  sorry

end tree_original_height_l65_65922


namespace drawing_red_ball_is_certain_l65_65354

def certain_event (balls : List String) : Prop :=
  ∀ ball ∈ balls, ball = "red"

theorem drawing_red_ball_is_certain:
  certain_event ["red", "red", "red", "red", "red"] :=
by
  sorry

end drawing_red_ball_is_certain_l65_65354


namespace volume_correct_l65_65361

-- Given conditions
def width : ℕ := 9
def length : ℕ := 4
def height : ℕ := 7

-- Define the volume of the box
def volume : ℕ := width * length * height

-- Statement of the proposition
theorem volume_correct : volume = 252 := by
  sorry

end volume_correct_l65_65361


namespace AX_is_correct_length_l65_65573

noncomputable def AX_length : ℝ :=
  let angle_BAC := 12 * Real.pi / 180
  let angle_BXC := 36 * Real.pi / 180
  let angle_ABX := 24 * Real.pi / 180
  let sin_36 := Real.sin angle_BXC
  let cos_12_sin_24 := Real.cos angle_BAC * Real.sin angle_ABX
  cos_12_sin_24 * Real.csc angle_BXC

theorem AX_is_correct_length (A B C D X : Point) (r : ℝ)
  (h_circle_radius : r = 1/2)
  (h_points_on_circle : OnCircle A r ∧ OnCircle B r ∧ OnCircle C r ∧ OnCircle D r)
  (h_X_on_diameter : OnDiameter X A D)
  (h_BX_eq_CX : dist B X = dist C X)
  (h_angle_BAC : angle B A C = 12 * Real.pi / 180)
  (h_angle_BXC : 3 * angle B A C = angle B X C) :
  dist A X = AX_length := 
  sorry

end AX_is_correct_length_l65_65573


namespace time_between_stops_approx_l65_65647

/-- The average speed of the bus in km/h -/
def speed : Float := 60.0

/-- The distance from Yahya's house to the Pinedale mall in km -/
def distance : Float := 40.0

/-- The number of stops from Yahya's house to the Pinedale mall -/
def stops : Nat := 8 

/-- Time between stops in minutes -/
theorem time_between_stops_approx :
  let total_time_in_hours := distance / speed
  let total_time_in_minutes := total_time_in_hours * 60.0
  let intervals := stops - 1
  let time_between_stops := total_time_in_minutes / intervals
  time_between_stops ≈ 5.71 :=
by
  sorry

end time_between_stops_approx_l65_65647


namespace geometric_sequence_thm_proof_l65_65662

noncomputable def geometric_sequence_thm (a : ℕ → ℤ) : Prop :=
  (∃ r : ℤ, ∃ a₀ : ℤ, ∀ n : ℕ, a n = a₀ * r ^ n) ∧
  (a 2) * (a 10) = 4 ∧
  (a 2) + (a 10) > 0 →
  (a 6) = 2

theorem geometric_sequence_thm_proof (a : ℕ → ℤ) :
  geometric_sequence_thm a :=
  by
  sorry

end geometric_sequence_thm_proof_l65_65662


namespace num_arrangements_l65_65804

-- Define the problem conditions
def athletes : Finset ℕ := {0, 1, 2, 3, 4, 5}
def A : ℕ := 0
def B : ℕ := 1

-- Define the constraint that athlete A cannot run the first leg and athlete B cannot run the fourth leg
def valid_arrangements (sequence : Fin 4 → ℕ) : Prop :=
  sequence 0 ≠ A ∧ sequence 3 ≠ B

-- Main theorem statement: There are 252 valid arrangements
theorem num_arrangements : (Fin 4 → ℕ) → ℕ :=
  sorry

end num_arrangements_l65_65804


namespace option_b_correct_option_c_correct_option_d_correct_l65_65820

theorem option_b_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : Real.log a * Real.log b = 1) :
  Real.log 2 / Real.log a < Real.log 2 / Real.log b := 
sorry

theorem option_c_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : Real.log a * Real.log b = 1) :
  (1 / 2)^(a*b + 1) < (1 / 2)^(a + b) :=
sorry

theorem option_d_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : Real.log a * Real.log b = 1) :
  a^a * b^b > a^b * b^a :=
sorry

end option_b_correct_option_c_correct_option_d_correct_l65_65820


namespace principal_sum_correct_l65_65304

noncomputable def principal_sum (CI SI : ℝ) (t : ℕ) : ℝ :=
  let P := ((SI * t) / t) in
  let x := 5100 / P in
  26010000 / ((CI - SI) / t)

theorem principal_sum_correct :
  principal_sum 11730 10200 2 ≈ 16993.46 :=
by
  simp only [principal_sum]
  sorry

end principal_sum_correct_l65_65304


namespace oil_per_cylinder_l65_65469

theorem oil_per_cylinder (cylinders : ℕ) (oil_added : ℕ) (oil_additional : ℕ)
  (h1 : cylinders = 6)
  (h2 : oil_added = 16)
  (h3 : oil_additional = 32) :
  (oil_added + oil_additional) / cylinders = 8 :=
by
  rw [h1, h2, h3]
  norm_num

end oil_per_cylinder_l65_65469


namespace inequality_holds_and_equality_occurs_l65_65467

theorem inequality_holds_and_equality_occurs (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) :
  (1 / (x + 3) + 1 / (y + 3) ≤ 2 / 5) ∧ (x = 2 ∧ y = 2 → 1 / (x + 3) + 1 / (y + 3) = 2 / 5) :=
by
  sorry

end inequality_holds_and_equality_occurs_l65_65467


namespace sine_addition_formula_l65_65287

theorem sine_addition_formula (α β : ℝ) :
  sin (α - β) * cos β + cos (α - β) * sin β = sin α :=
by
  sorry

end sine_addition_formula_l65_65287


namespace sum_a6_a7_a8_a9_l65_65813

variable {a : ℕ → ℕ}
axiom h : ∀ n, (finset.range (n + 1)).sum a = n^3

theorem sum_a6_a7_a8_a9 : a 6 + a 7 + a 8 + a 9 = 604 := by
  have S_9 := h 9
  have S_5 := h 5
  have h1 : (9:ℕ)^3 = 729 := by norm_num
  have h2 : (5:ℕ)^3 = 125 := by norm_num
  simp at S_9 S_5
  rw [←S_9, ←S_5]
  norm_num
  sorry

end sum_a6_a7_a8_a9_l65_65813


namespace Penelope_Candies_l65_65962

variable (M : ℕ) (S : ℕ)
variable (h1 : 5 * S = 3 * M)
variable (h2 : M = 25)

theorem Penelope_Candies : S = 15 := by
  sorry

end Penelope_Candies_l65_65962


namespace fixed_point_planes_of_tetrahedron_fixed_line_planes_of_tetrahedron_l65_65623

open Affine

-- Definition and lean statement for part (a)
theorem fixed_point_planes_of_tetrahedron (α β γ : ℝ) {A B C D K L M : Point V} (hA: A ≠ B) (h1 : B = A +ᵥ (α • (B -ᵥ A))) (h2 : C = A +ᵥ (β • (C -ᵥ A))) (h3 : D = A +ᵥ (γ • (D -ᵥ A))) (hγ: γ = α + β + 1) :
  ∃ X : Point V, ∀ T: Triangle, {K, L, M} ⊆ T.points ∧ T ⊆ (Plane.mk X A B) := sorry

-- Definition and lean statement for part (b)
theorem fixed_line_planes_of_tetrahedron (α β γ : ℝ) {A B C D K L M : Point V} (hA: A ≠ B) (h1 : B = A +ᵥ (α • (B -ᵥ A))) (h2 : C = A +ᵥ (β • (C -ᵥ A))) (h3 : D = A +ᵥ (γ • (D -ᵥ A))) (hβ: β = α + 1) (hγ: γ = β + 1) :
  ∃ L : Line, ∀ T: Triangle, {K, L, M} ⊆ T.points ∧ T ⊆ (Plane.mk (A +ᵥ (vector_between B C)) A B) := sorry

end fixed_point_planes_of_tetrahedron_fixed_line_planes_of_tetrahedron_l65_65623


namespace math_problem_l65_65089

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 3 + b * x + 2

theorem math_problem (a b : ℝ) (h : f a b (-12) = 3) : f a b 12 = 1 :=
by
  have eq1 : 8 * a + 2 * b = -1, sorry
  have eq2 : f a b 12 = 8 * a + 2 * b + 2, sorry
  rw eq1 at eq2
  exact eq2

end math_problem_l65_65089


namespace tan_eight_pi_over_three_l65_65691

theorem tan_eight_pi_over_three : tan (8 * Real.pi / 3) = -Real.sqrt 3 :=
by sorry

end tan_eight_pi_over_three_l65_65691


namespace sought_line_eq_l65_65991

-- Definitions used in the conditions
def circle_eq (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def line_perpendicular (x y : ℝ) : Prop := x + y = 0
def center_of_circle : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem sought_line_eq (x y : ℝ) :
  (circle_eq x y ∧ line_perpendicular x y ∧ (x, y) = center_of_circle) →
  (x + y + 1 = 0) :=
by
  sorry

end sought_line_eq_l65_65991


namespace general_term_correct_inequality_proof_l65_65124

-- Define the sequence using the given recurrence relations
def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 2 * sequence (n - 1) + 1

-- Define the general term formula for the sequence
def general_term (n : ℕ) : ℕ :=
  2^n - 1

-- Prove that the sequence defined by the recurrence relation has the general term formula
theorem general_term_correct (n : ℕ) : sequence n = general_term n :=
sorry

-- Prove the inequality involving the ratios of successive terms
theorem inequality_proof (n : ℕ) : (∑ k in Finset.range (n + 1).filter (λ k, k ≠ 0), (sequence k) / (sequence (k+1))) < n / 2 :=
sorry

end general_term_correct_inequality_proof_l65_65124


namespace find_x_value_l65_65067

theorem find_x_value : ∃ x : ℝ, 25^(-3) = (5^(60/x)) / (5^(36/x) * 25^(21/x)) ∧ x = 3 :=
by
  sorry

end find_x_value_l65_65067


namespace factorial_equation_l65_65539

theorem factorial_equation (n : ℕ) : 5! * 3! = n! → n = 6 :=
by
  intro h
  sorry

end factorial_equation_l65_65539


namespace correct_option_d_l65_65525

variables {l m n : Line} {a β : Plane}

-- Conditions
axiom line_perpendicular (x y : Line) : Prop   -- x ⊥ y
axiom line_contained_in_plane (x : Line) (p : Plane) : Prop -- x ⊆ p
axiom planes_parallel (p q : Plane) : Prop -- p ∥ q
axiom points_equidistant_from_plane (points : set Point) (p : Plane) : Prop -- points equidistant from p
axiom planes_intersect (p q : Plane) : Prop -- p and q intersect

-- Correct Answer (Option D)
theorem correct_option_d (m n : Line) (a : Plane) :
  (line_parallel m n) ∧ (line_perpendicular n a) → (line_perpendicular m a) :=
sorry

end correct_option_d_l65_65525


namespace librarian_books_arrangement_l65_65004

theorem librarian_books_arrangement (n m : ℕ) (h_n : n = 5) (h_m : m = 3) :
  (nat.choose (n + m) n) = 56 :=
by
  rw [h_n, h_m]
  simp
  sorry

end librarian_books_arrangement_l65_65004


namespace average_percentage_reduction_equation_l65_65723

theorem average_percentage_reduction_equation (x : ℝ) : 200 * (1 - x)^2 = 162 :=
by 
  sorry

end average_percentage_reduction_equation_l65_65723


namespace product_cos_tends_to_limit_l65_65241

open Real

theorem product_cos_tends_to_limit (t : ℝ) :
  tendsto (λ n, ∏ k in finset.range n, (2 * cos (t / 2^k) - 1)) at_top (nhds ((2 * cos t + 1) / 3)) :=
sorry

end product_cos_tends_to_limit_l65_65241


namespace quadratic_negative_root_l65_65466

theorem quadratic_negative_root (m : ℝ) : (∃ x : ℝ, (m * x^2 + 2 * x + 1 = 0 ∧ x < 0)) ↔ (m ≤ 1) :=
by
  sorry

end quadratic_negative_root_l65_65466


namespace johns_new_total_is_correct_l65_65206

theorem johns_new_total_is_correct:
  let initial_squat := 700
  let initial_bench := 400
  let initial_deadlift := 800
  let squat_loss_percentage := 0.30
  let deadlift_loss := 200
  let new_squat := initial_squat - squat_loss_percentage * initial_squat
  let new_bench := initial_bench
  let new_deadlift := initial_deadlift - deadlift_loss
  let new_total := new_squat + new_bench + new_deadlift
  new_total = 1490 :=
by
  simp [initial_squat, initial_bench, initial_deadlift, 
        squat_loss_percentage, deadlift_loss, 
        new_squat, new_bench, new_deadlift, new_total]
  norm_num
  sorry

end johns_new_total_is_correct_l65_65206


namespace smallest_palindrome_not_five_digit_l65_65062

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.toDigits 10
  s = s.reverse

theorem smallest_palindrome_not_five_digit (n : ℕ) :
  (∃ n, is_palindrome n ∧ 100 ≤ n ∧ n < 1000 ∧ ¬is_palindrome (102 * n)) → n = 101 := by
  sorry

end smallest_palindrome_not_five_digit_l65_65062


namespace min_value_b1_b2_l65_65768

noncomputable def seq (b : ℕ → ℕ) : Prop :=
  ∀ n ≥ 1, b (n + 2) = (b n + 2017) / (1 + b (n + 1))

theorem min_value_b1_b2 (b : ℕ → ℕ)
  (h_pos : ∀ n, b n > 0)
  (h_seq : seq b) :
  b 1 + b 2 = 2018 := sorry

end min_value_b1_b2_l65_65768


namespace count_valid_permutations_l65_65906

def is_valid_permutation (perm : List ℕ) : Prop :=
  (perm.length = 6) ∧ (perm = [1,2,3,4,5,6].permute) ∧
  (∀ i, i < 4 → ¬(perm.nth_le i sorry < perm.nth_le (i+1) sorry ∧ perm.nth_le (i+1) sorry < perm.nth_le (i+2) sorry) ∧
           ¬(perm.nth_le i sorry > perm.nth_le (i+1) sorry ∧ perm.nth_le (i+1) sorry > perm.nth_le (i+2) sorry)) ∧
  (∃ i, 1 ≤ i ∧ i < 6 ∧ perm.nth_le i sorry = 1 ∧ perm.nth_le (i - 1) sorry = 2 ∨ perm.nth_le (i - 1) sorry = 1 ∧ perm.nth_le i sorry = 2)

theorem count_valid_permutations : 
  ∃ (l : List (List ℕ)), List.Forall is_valid_permutation l ∧ l.length = 96 :=
sorry

end count_valid_permutations_l65_65906


namespace problem_l65_65518

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x

theorem problem (k x₁ x₂ : ℝ) 
  (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) 
  (h : g x₁ / k ≤ f x₂ / (k + 1)) : 
  k ≥ 1 / (2 * Real.exp 1 - 1) := sorry

end problem_l65_65518


namespace trigonometric_identity_proof_l65_65421

/-- 
Prove that:
a) arccos (sin (-π / 7)) = 9π / 14
b) arcsin (cos (33 * π / 5)) = -π / 10
-/
theorem trigonometric_identity_proof :
  arccos (sin (-π / 7)) = 9 * π / 14 ∧ arcsin (cos (33 * π / 5)) = -π / 10 :=
by
  sorry

end trigonometric_identity_proof_l65_65421


namespace exists_3x3_grid_l65_65269

theorem exists_3x3_grid : 
  ∃ (a₁₂ a₂₁ a₂₃ a₃₂ : ℕ), 
  a₁₂ ≠ a₂₁ ∧ a₁₂ ≠ a₂₃ ∧ a₁₂ ≠ a₃₂ ∧ 
  a₂₁ ≠ a₂₃ ∧ a₂₁ ≠ a₃₂ ∧ 
  a₂₃ ≠ a₃₂ ∧ 
  a₁₂ ≤ 25 ∧ a₂₁ ≤ 25 ∧ a₂₃ ≤ 25 ∧ a₃₂ ≤ 25 ∧ 
  a₁₂ > 0 ∧ a₂₁ > 0 ∧ a₂₃ > 0 ∧ a₃₂ > 0 ∧
  (∃ (a₁₁ a₁₃ a₃₁ a₃₃ a₂₂ : ℕ),
  a₁₁ ≤ 25 ∧ a₁₃ ≤ 25 ∧ a₃₁ ≤ 25 ∧ a₃₃ ≤ 25 ∧ a₂₂ ≤ 25 ∧
  a₁₁ > 0 ∧ a₁₃ > 0 ∧ a₃₁ > 0 ∧ a₃₃ > 0 ∧ a₂₂ > 0 ∧
  a₁₁ ≠ a₁₂ ∧ a₁₁ ≠ a₂₁ ∧ a₁₁ ≠ a₁₃ ∧ a₁₁ ≠ a₃₁ ∧ 
  a₁₃ ≠ a₃₃ ∧ a₁₃ ≠ a₂₃ ∧ a₂₁ ≠ a₃₁ ∧ a₃₁ ≠ a₃₂ ∧ 
  a₃₃ ≠ a₂₂ ∧ a₃₃ ≠ a₃₂ ∧ a₂₂ = 1 ∧
  (a₁₂ % a₂₂ = 0 ∨ a₂₂ % a₁₂ = 0) ∧
  (a₂₁ % a₂₂ = 0 ∨ a₂₂ % a₂₁ = 0) ∧
  (a₂₃ % a₂₂ = 0 ∨ a₂₂ % a₂₃ = 0) ∧
  (a₃₂ % a₂₂ = 0 ∨ a₂₂ % a₃₂ = 0) ∧
  (a₁₁ % a₁₂ = 0 ∨ a₁₂ % a₁₁ = 0) ∧
  (a₁₁ % a₂₁ = 0 ∨ a₂₁ % a₁₁ = 0) ∧
  (a₁₃ % a₁₂ = 0 ∨ a₁₂ % a₁₃ = 0) ∧
  (a₁₃ % a₂₃ = 0 ∨ a₂₃ % a₁₃ = 0) ∧
  (a₃₁ % a₂₁ = 0 ∨ a₂₁ % a₃₁ = 0) ∧
  (a₃₁ % a₃₂ = 0 ∨ a₃₂ % a₃₁ = 0) ∧
  (a₃₃ % a₂₃ = 0 ∨ a₂₃ % a₃₃ = 0) ∧
  (a₃₃ % a₃₂ = 0 ∨ a₃₂ % a₃₃ = 0)) 
  :=
sorry

end exists_3x3_grid_l65_65269


namespace reflection_matrix_l65_65313

theorem reflection_matrix (a b : ℚ) :
  let R := matrix.of ![![a, b], ![-(4/5 : ℚ), (3/5 : ℚ)]] in
  R ⬝ R = 1 →  a = -(3/5 : ℚ) ∧ b = -(4/5 : ℚ) :=
sorry

end reflection_matrix_l65_65313


namespace sum_of_intersections_is_minus_twelve_l65_65630

theorem sum_of_intersections_is_minus_twelve :
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ x : ℚ, ax + 7 = 0 ∧ 2x + b = 0) →
   (a = 1 ∨ a = 2 ∨ a = 7 ∨ a = 14) →
   let pairs := [(1, 14), (2, 7), (7, 2), (14, 1)] in
   (∑ (p : ℕ × ℕ) in pairs.to_finset, - \((7 : ℚ) / (p.1 : ℚ))) = -12) :=
by
  sorry

end sum_of_intersections_is_minus_twelve_l65_65630


namespace proof_part_a_proof_part_b_l65_65325

variables {A B C X M : Type}
variables [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty X] [Nonempty M]

-- Given conditions: (Assuming A, B, C are points of an acute-angled triangle)
def is_acute_triangle (A B C : Type) : Prop := sorry

-- Tangents at B and C to the circumcircle of triangle ABC meet at X
def tangents_meet (A B C X : Type) : Prop := sorry

-- M is the midpoint of BC
def is_midpoint (B M C : Type) : Prop := sorry

-- Angle BAM
def angle_BAM (A B M : Type) : A := sorry

-- Angle CAX
def angle_CAX (A C X : Type) : A := sorry

-- Computers if two angles are equal
def angles_equal (α β : Type) : Prop := sorry

-- AM
def length_AM (A M : Type) : ℝ := sorry

-- AX
def length_AX (A X : Type) : ℝ := sorry

-- Cosine of angle BAC
def cosine_angle_BAC (A B C : Type) : ℝ := sorry

theorem proof_part_a (A B C X M : Type) 
  (h1 : is_acute_triangle A B C)
  (h2 : tangents_meet A B C X) 
  (h3 : is_midpoint B M C) :
  angles_equal (angle_BAM A B M) (angle_CAX A C X) := 
by sorry

theorem proof_part_b (A B C X M : Type) 
  (h1 : is_acute_triangle A B C)
  (h2 : tangents_meet A B C X) 
  (h3 : is_midpoint B M C) :
  length_AM A M / length_AX A X = cosine_angle_BAC A B C := 
by sorry

end proof_part_a_proof_part_b_l65_65325


namespace sum_cos_eq_l65_65034

noncomputable def i := complex.I

-- Define the function representing the sum.
def sum_expr : ℂ :=
  ∑ n in finset.range 31, i^n * complex.cos (real.to_radians (60 + 90 * n))

-- Provide the condition given in the problem.
axiom i_squared : i^2 = -1

-- Define the main statement to prove.
theorem sum_cos_eq : sum_expr = 7.5 - 7 * i :=
by 
  have h : ∀ n, ∃ k, n = 4 * k + n % 4 := nat.exists_eq_mul_add_mod
  sorry

end sum_cos_eq_l65_65034


namespace greatest_possible_points_l65_65895

-- Define the number of teams and game results
constant num_teams : ℕ := 8
constant points_win : ℕ := 3
constant points_draw : ℕ := 1
constant points_loss : ℕ := 0

-- Definition of the maximum points that each of the top three teams can earn
def max_points_each_top_team : ℕ :=
  (num_teams - 3) * 2 * points_win + 3 * (2 * points_draw)

theorem greatest_possible_points :
  max_points_each_top_team = 36 :=
by
  sorry

end greatest_possible_points_l65_65895


namespace common_chord_length_common_chord_diameter_eq_circle_l65_65129

/-
Given two circles C1: x^2 + y^2 - 2x + 10y - 24 = 0 and C2: x^2 + y^2 + 2x + 2y - 8 = 0,
prove that 
1. The length of the common chord is 2 * sqrt(5).
2. The equation of the circle that has the common chord as its diameter is (x + 8/5)^2 + (y - 6/5)^2 = 36/5.
-/

-- Define the first circle C1
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 10 * y - 24 = 0

-- Define the second circle C2
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

-- Prove the length of the common chord
theorem common_chord_length : ∃ d : ℝ, d = 2 * Real.sqrt 5 :=
sorry

-- Prove the equation of the circle that has the common chord as its diameter
theorem common_chord_diameter_eq_circle : ∃ (x y : ℝ → ℝ), (x + 8/5)^2 + (y - 6/5)^2 = 36/5 :=
sorry

end common_chord_length_common_chord_diameter_eq_circle_l65_65129


namespace arrangement_count_l65_65901

theorem arrangement_count (s : Finset ℕ) (h1 : s = {1, 2, 3, 4, 5, 6}) :
  ∃ f : ℕ → ℕ, 
  (∀ (i : ℕ), i ∈ Finset.range 4 → f i ∈ s) ∧
  (∀ (i : ℕ), i ∈ Finset.range 4 → ((f i) * (f (i + 2)) - (f (i + 1))^2) % 7 = 0) ∧
  (Finset.card (Finset.image f (Finset.range 6))) = 12 :=
sorry

end arrangement_count_l65_65901


namespace jars_needed_l65_65199

-- Definitions based on the given conditions
def total_cherry_tomatoes : ℕ := 56
def cherry_tomatoes_per_jar : ℕ := 8

-- Lean theorem to prove the question
theorem jars_needed (total_cherry_tomatoes cherry_tomatoes_per_jar : ℕ) (h1 : total_cherry_tomatoes = 56) (h2 : cherry_tomatoes_per_jar = 8) : (total_cherry_tomatoes / cherry_tomatoes_per_jar) = 7 := by
  -- Proof omitted
  sorry

end jars_needed_l65_65199


namespace complex_calculation_l65_65091

def i : ℂ := complex.I

theorem complex_calculation :
  2 * i * (1 - i) = 2 + 2 * i :=
by
  sorry

end complex_calculation_l65_65091


namespace total_revenue_correct_l65_65616

variables (milk_yesterday_am milk_yesterday_pm milk_today_am milk_left storage_fee delivery_fee milk_price_am milk_price_pm : ℕ)
variables (total_milk total_milk_sold : ℕ)
variables (revenue_from_milk total_fees total_revenue expected_revenue : ℕ)

def milk_yesterday_am := 68
def milk_yesterday_pm := 82
def milk_today_am := milk_yesterday_am - 18
def total_milk := milk_yesterday_am + milk_yesterday_pm + milk_today_am
def milk_left := 24
def storage_fee := 10
def delivery_fee := 20
def milk_price_am := 3500 -- Convert price to cents to avoid decimals
def milk_price_pm := 4000
def total_milk_sold := total_milk - milk_left

def revenue_from_milk := (milk_price_am * (milk_yesterday_am + milk_today_am) + milk_price_pm * milk_yesterday_pm) / 100 * (total_milk_sold / total_milk)
def total_fees := (2 * delivery_fee) + (milk_left * storage_fee)
def total_revenue := revenue_from_milk - total_fees
def expected_revenue := 380

theorem total_revenue_correct : total_revenue = expected_revenue := by
  sorry

end total_revenue_correct_l65_65616


namespace hyperbola_equation_l65_65775

theorem hyperbola_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 - 2 → (∃ k : ℝ, k ≠ 0 ∧ x * y = k) := 
by
  intros h
  sorry

end hyperbola_equation_l65_65775


namespace intersect_circumcircle_at_most_one_l65_65926

variables {A B C D E O : Type*} [EuclideanGeometry O A B C D E]

-- Given conditions
def circumcenter_A_B_C (O : O) (A B C : Type*) : Prop := -- defining O as circumcenter of \triangle ABC
  is_circumcenter O A B C

def intersect_perpendicular_bisectors (O : O) (A B C D E : Type*) : Prop := 
  -- perpendicular bisectors of \overline{OB} and \overline{OC} intersect lines \overline{AB} and \overline{AC} at D≠A and E≠A respectively
  is_perpendicular_bisector (line O B) (line A B) D ∧
  is_perpendicular_bisector (line O C) (line A C) E ∧
  D ≠ A ∧ E ≠ A

def max_intersection_points (A B C D E : Type*) : Prop :=
  ∀ P, P ∈ (circumcircle_triangle_ADE A D E) → points_on_line P B C → ∃! P

-- Proving the maximum intersection points
theorem intersect_circumcircle_at_most_one (O A B C D E : Type*) :
  circumcenter_A_B_C O A B C →
  intersect_perpendicular_bisectors O A B C D E →
  max_intersection_points A B C D E := 
by
  sorry

end intersect_circumcircle_at_most_one_l65_65926


namespace necessary_but_not_sufficient_range_m_l65_65819

namespace problem

variable (m x y : ℝ)

/-- Propositions for m -/
def P := (1 < m ∧ m < 4) 
def Q := (2 < m ∧ m < 3) ∨ (3 < m ∧  m < 4)

/-- Statements that P => Q is necessary but not sufficient -/
theorem necessary_but_not_sufficient (hP : 1 < m ∧ m < 4) : 
  ((m-1) * (m-4) < 0) ∧ (Q m) :=
by 
  sorry

theorem range_m (h1 : ¬ (P m ∧ Q m)) (h2 : P m ∨ Q m) : 
  1 < m ∧ m ≤ 2 ∨ m = 3 :=
by
  sorry

end problem

end necessary_but_not_sufficient_range_m_l65_65819


namespace range_of_a_l65_65232

noncomputable def f (x : ℝ) (a : ℝ) (n : ℕ) : ℝ :=
  real.log ((1 + ∑ k in finset.range (n - 1), (k+1)^x + a * n^x) / n)

theorem range_of_a (a : ℝ) (n : ℕ) (hx : n ≥ 2) :
  (∀ x : ℝ, x ≤ 1 → 0 < (1 + ∑ k in finset.range (n-1), (k+1)^x + a * n^x) / n) ↔ a > -((n - 1) / 2) :=
sorry

end range_of_a_l65_65232


namespace probability_quadratic_inequality_l65_65056

theorem probability_quadratic_inequality :
  (set_integral volume (set.Icc (0: ℝ) 1) (λ x, set_integral volume (set.Icc (0: ℝ) 1) (λ y, if (x^2 - 3*x*y + 2*y^2 > 0) then (1 : ℝ) else 0)) = 3 / 4) :=
sorry

end probability_quadratic_inequality_l65_65056


namespace books_sold_in_march_l65_65383

-- Define the conditions
def books_sold_in_january : ℕ := 15
def books_sold_in_february : ℕ := 16
def average_books_sold_per_month : ℕ := 16
def total_months : ℕ := 3

-- Statement to prove
theorem books_sold_in_march :
  books_sold_in_january + books_sold_in_february + ?books_march = average_books_sold_per_month * total_months →
  ?books_march = 17 :=
by
  sorry

end books_sold_in_march_l65_65383


namespace domain_of_shifted_linear_function_l65_65549

theorem domain_of_shifted_linear_function (f : ℝ → ℝ) :
  (∀ x, f x = x + 1) → (set.Ioc (2 : ℝ) 3 = {y | 2 < y ∧ y ≤ 3}) → 
  (set.Ioc (1 : ℝ) 2 = {x | 1 < x ∧ x ≤ 2}) := 
by
 sorry

end domain_of_shifted_linear_function_l65_65549


namespace point_inside_polygon_iff_odd_marked_vertices_l65_65811

variable (Polygon : Type)
variable (l : Type)
variable (P : Type)

-- Definitions based on the conditions:
-- Assume a set of vertices and edges and intersections
variable (Vertices : Set Polygon)
variable (Edges : Set (Polygon × Polygon))
variable (Intersection : Edges → l)

-- Marking condition where vertices are marked based on their emanating sides intersecting l on opposite sides of P
def MarkedVertices (v : Vertices) (P : l) (Intersection : Edges → l) : Prop :=
  ∃ e1 e2 : Edges, e1 ∈ Edges ∧ e2 ∈ Edges ∧ Intersection e1 ≠ Intersection e2 ∧
  ((Intersection e1 < P ∧ Intersection e2 > P) ∨ (Intersection e1 > P ∧ Intersection e2 < P))

-- Statement of the problem
theorem point_inside_polygon_iff_odd_marked_vertices :
  (∃ n : ℕ, MarkedVertices Vertices P Intersection = 2 * n + 1 ∧ VertexCondition Vertices) ↔ 
  (PointInsidePolygon P Vertices Edges) :=
sorry

end point_inside_polygon_iff_odd_marked_vertices_l65_65811


namespace convert_base_5_to_binary_l65_65426

theorem convert_base_5_to_binary (n : ℕ) (h₁ : n = 4 * 5^1 + 4) : 
  Nat.toDigits 2 24 = [1, 1, 0, 0, 0] :=
by
  have h₂ : n = 24 :=
    by rw [← h₁, show 4 * 5 + 4 = 24 from rfl]
  exact Eq.symm (Nat.toDigits_eq _ _)

end convert_base_5_to_binary_l65_65426


namespace vector_at_t_4_l65_65810

theorem vector_at_t_4 
  (vec_neg1 : ℝ × ℝ × ℝ := (2, 5, 11)) 
  (vec_0 : ℝ × ℝ × ℝ := (3, 7, 15)) 
  (vec_1 : ℝ × ℝ × ℝ := (1, 2, 4)) : 
  (ℝ × ℝ × ℝ) :=
  ∃ (a d : ℝ × ℝ × ℝ),
    vec_neg1 = (a.1 - d.1, a.2 - d.2, a.3 - d.3) ∧
    vec_1 = (a.1 + d.1, a.2 + d.2, a.3 + d.3) ∧
    vec_0 = a ∧
    (4 * d.1 + a.1, 4 * d.2 + a.2, 4 * d.3 + a.3) = (-5, -13, -29)

end vector_at_t_4_l65_65810


namespace perimeter_quadrilateral_div_b_l65_65396

noncomputable def intersect_point1 (b : ℝ) : ℝ × ℝ :=
(b, b / 3)

noncomputable def intersect_point2 (b : ℝ) : ℝ × ℝ :=
(-b, -b / 3)

noncomputable def perimeter_div_b (b : ℝ) : ℝ :=
let p1 := intersect_point1 b in
let p2 := intersect_point2 b in
2 * (b + p1.snd) + 2 * (b - p2.fst) + real.sqrt (b^2 + (2 * b)^2) / b

theorem perimeter_quadrilateral_div_b (b : ℝ) (hb : b ≠ 0) :
  perimeter_div_b b = (10 / 3 : ℝ) + real.sqrt 5 :=
by
  sorry

end perimeter_quadrilateral_div_b_l65_65396


namespace exists_multicolored_cycle_exists_tricolored_triangle_l65_65769

noncomputable def complete_graph {n : ℕ} (V : Type) [Fintype V] [DecidableEq V] [Fintype (Sym2 V)] : 
  SimpleGraph V := SimpleGraph.completeGraph V

variables {n : ℕ} (G : SimpleGraph (Fin n)) [DecidableRel G.adj]

-- Condition: The complete graph on n vertices
noncomputable def K_n : SimpleGraph (Fin n) := complete_graph (Fin n)

-- Condition: The edges of the complete graph are colored with n different colors, ensuring each color is used
variable (coloring : Sym2 (Fin n) → Fin n) -- Edge coloring

-- 1. Prove that there exists a multicolored cycle
theorem exists_multicolored_cycle :
  ∃ (cycle : List (Sym2 (Fin n))), (∀ e ∈ cycle, ∃ c ∈ Fin n, coloring e = c) ∧ (List.Nodup (List.map coloring cycle)) ∧ SimpleGraph.cycle K_n cycle :=
sorry

-- 2. Prove that there exists a tricolored triangle
theorem exists_tricolored_triangle (h : ∃ (cycle : List (Sym2 (Fin n))), (∀ e ∈ cycle, ∃ c ∈ Fin n, coloring e = c) ∧ (List.Nodup (List.map coloring cycle)) ∧ SimpleGraph.cycle K_n cycle) :
  ∃ (triangle : List (Sym2 (Fin n))), (List.length triangle = 3) ∧ (∀ e ∈ triangle, ∃ c ∈ Fin n, coloring e = c) ∧ (List.Nodup (List.map coloring triangle)) ∧ SimpleGraph.cycle K_n triangle :=
sorry

end exists_multicolored_cycle_exists_tricolored_triangle_l65_65769


namespace find_lambda_l65_65864

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (2, 4)

def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem find_lambda (λ : ℝ) (h : (dot_product (a.1 - λ * b.1, a.2 - λ * b.2) b) = 0):
  λ = 7 / 10 :=
sorry

end find_lambda_l65_65864


namespace fabric_purchase_l65_65338

theorem fabric_purchase (c_s p_s s_s total_s : ℝ) (c_y p_y s_y d_y : ℝ)
    (price_c price_p price_s price_d discount : ℝ) :
    c_s = 75 →
    p_s = 45 →
    s_s = 63 →
    total_s = 250 →
    price_c = 7.5 →
    price_p = 6 →
    price_s = 9 →
    price_d = 4.5 →
    discount = 0.1 →
    c_y = c_s / price_c →
    p_y = p_s / price_p →
    s_y = s_s / price_s →
    total_s = c_s + p_s + s_s + (d_y * price_d * (1 - discount)) →
    c_y = 10 ∧ p_y = 7.5 ∧ s_y = 7 ∧ d_y ≈ 16.54 ∧ d_y * price_d * (1 - discount) = 67 :=
by
  intros
  sorry

end fabric_purchase_l65_65338


namespace sum_of_divisors_330_l65_65681

def is_sum_of_divisors (n : ℕ) (sum : ℕ) :=
  sum = ∑ d in divisors n, d

theorem sum_of_divisors_330 : is_sum_of_divisors 330 864 :=
by {
  -- sorry
}

end sum_of_divisors_330_l65_65681


namespace hazel_made_56_cups_l65_65136

-- Definitions based on problem conditions:
def sold_to_kids (sold: ℕ) := sold = 18
def gave_away (sold: ℕ) (gave: ℕ) := gave = sold / 2
def drank (drank: ℕ) := drank = 1
def half_total (total: ℕ) (sum_sold_gave_drank: ℕ) := sum_sold_gave_drank = total / 2

-- Main statement that needs to be proved:
theorem hazel_made_56_cups : ∃ (total: ℕ), 
  ∀ (sold gave drank sum_sold_gave_drank: ℕ), 
    sold_to_kids sold → 
    gave_away sold gave → 
    drank drank → 
    half_total total (sold + gave + drank) → 
    total = 56 := 
by sorry

end hazel_made_56_cups_l65_65136


namespace smallest_integer_side_4_l65_65777

def triangle_sides_valid (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def smallest_integer_side (a b : ℝ) : ℕ :=
  if h : ∃ s : ℕ, s > (b - a) ∧ s < (a + b) then
    @Nat.find (λ s, s > (b - a) ∧ s < (a + b)) h
  else 0

theorem smallest_integer_side_4 : smallest_integer_side 7.8 11 = 4 :=
by
  unfold smallest_integer_side
  split
  sorry

end smallest_integer_side_4_l65_65777


namespace problem_statement_l65_65103

theorem problem_statement :
  (¬ (∀ x : ℝ, 2 * x < 3 * x)) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := 
sorry

end problem_statement_l65_65103


namespace sqrt_fraction_addition_l65_65413

theorem sqrt_fraction_addition : 
    sqrt ((9 / 49) + (16 / 9)) = (sqrt 865) / 21 := 
sorry

end sqrt_fraction_addition_l65_65413


namespace circle_has_area_24pi_l65_65027

-- Definitions of geometric elements and given conditions
variables (O A B C D E F : Point)
variable (circle : Circle)
variable [Diameter : circle.Diameter AB]
variable [Diameter : circle.Diameter CD]
variable [Perpendicular : Perpendicular AB CD]
variable [Chord : Chord DF circle]
variable [Intersection : IntersectAt DF AB E]
variable (DE EF : Length)

-- Given distances DE = 6 and EF = 2
axiom DE_length : DE = 6
axiom EF_length : EF = 2

-- Statement of the problem
theorem circle_has_area_24pi : area circle = 24 * Real.pi :=
by
  -- Proof will go here
  sorry

end circle_has_area_24pi_l65_65027


namespace books_sold_in_march_l65_65382

-- Define the conditions
def books_sold_in_january : ℕ := 15
def books_sold_in_february : ℕ := 16
def average_books_sold_per_month : ℕ := 16
def total_months : ℕ := 3

-- Statement to prove
theorem books_sold_in_march :
  books_sold_in_january + books_sold_in_february + ?books_march = average_books_sold_per_month * total_months →
  ?books_march = 17 :=
by
  sorry

end books_sold_in_march_l65_65382


namespace contrapositive_proposition_l65_65522

theorem contrapositive_proposition (x : ℝ) : (x > 10 → x > 1) ↔ (x ≤ 1 → x ≤ 10) :=
by
  sorry

end contrapositive_proposition_l65_65522


namespace locus_intersection_pc_l65_65238

noncomputable theory

open Classical

variables {A B C A' B' C' P Q : Point}

-- Assumptions about the triangle and midpoints
variables (ABC : Triangle A B C) 
          (A' : A'.is_midpoint (B, C))
          (B' : B'.is_midpoint (A, C))
          (C' : C'.is_midpoint (A, B))

-- Line passing through point A
variable (ℓ : Line)
assumption hₗ : A ∈ ℓ

-- Perpendiculars dropped from B and C to line ℓ
variables (P : Point)
variables (Q : Point)
assumption hP : P.foot_of_perpendicular (B, ℓ)
assumption hQ : Q.foot_of_perpendicular (C, ℓ)

-- Variables for lines PC' and QB'
variable line_PC' : Line
variable line_QB' : Line
assumption h_PC' : C' ∈ line_PC' ∧ P ∈ line_PC'
assumption h_QB' : B' ∈ line_QB' ∧ Q ∈ line_QB'

-- The location of the intersection point M of lines PC' and QB'
variable M : Point
assumption hM : M ∈ line_PC' ∧ M ∈ line_QB'

-- The Feuerbach Circle
variable FeuerbachCircle : Circle
assumption hFeuerbach : FeuerbachCircle ∈ FeuerbachCircle_def (ABC)

-- Problem: Prove that M lies on the circle
theorem locus_intersection_pc'_qb' :
  M ∈ FeuerbachCircle :=
by sorry

end locus_intersection_pc_l65_65238


namespace find_circle_center_l65_65447

def circle_center_eq : Prop :=
  ∃ (x y : ℝ), (x^2 - 6 * x + y^2 + 2 * y - 12 = 0) ∧ (x = 3) ∧ (y = -1)

theorem find_circle_center : circle_center_eq :=
sorry

end find_circle_center_l65_65447


namespace sum_sequence_2023_l65_65512

def f (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

def sequence (n : ℕ) : ℝ := 
  if n % 3 = 0 then 1 else 
  if n % 3 = 1 then 0 else -1

theorem sum_sequence_2023 :
  (∑ i in Finset.range 2023, sequence i.succ) = 1 :=
by
  have h1 : sequence 1 = 1 := rfl
  have h2 : ∀ n, sequence (n + 3) = sequence n,
      from λ n => by simp [sequence, Nat.add_mod]
  have h3 : f (sequence 1) + f (sequence 2 + sequence 3) = 0,
      from calc
        f (sequence 1) = f 1 := rfl
        ... = 1 - (2 / (3^1 + 1)) := rfl
        ... = 1 - (2 / 4) := by norm_num
        ... = 0.5 := by norm_num
        ... (-f (sequence 2+sequence 3))  = (-f (0 + (-1))) := 
        rfl
     
  sorry

end sum_sequence_2023_l65_65512


namespace triangle_inequality_example_l65_65358

/-- Triangle inequality theorem application --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_inequality_example : can_form_triangle 8 8 15 :=
by {
  unfold can_form_triangle,
  split,
  calc
    8 + 8 > 15 : by linarith,
  split,
  calc
    8 + 15 > 8 : by linarith,
  calc
    8 + 15 > 8 : by linarith,
}

end triangle_inequality_example_l65_65358


namespace transformed_function_correct_l65_65671

noncomputable theory

def original_function (x : ℝ) : ℝ := Real.sin (2 * x)

def shifted_function (x : ℝ) : ℝ := Real.sin (2 * (x + (Real.pi / 8)))

theorem transformed_function_correct : 
  ∀ x : ℝ, shifted_function x = Real.sin (2 * x + (Real.pi / 4)) :=
by 
  intro x 
  unfold shifted_function 
  rw [mul_add, add_mul, ←Real.add_assoc, (show 2 * (Real.pi / 8) = Real.pi / 4, by ring)]
  sorry

end transformed_function_correct_l65_65671


namespace range_of_m_l65_65609

noncomputable def f (x : ℝ) : ℝ := x^2 + 3

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f(x) + m^2 * f(x) ≥ f(x - 1) + 3 * f(m)) ↔ (m ≤ -1 ∨ 0 ≤ m) :=
by
  sorry

end range_of_m_l65_65609


namespace one_fourth_of_12_point8_eq_fractions_l65_65445

theorem one_fourth_of_12_point8_eq_fractions :
  let x := 12.8 / 4 in
  x = 16 / 5 ∧ x = 3 + 1 / 5 := by
  sorry

end one_fourth_of_12_point8_eq_fractions_l65_65445


namespace pure_imaginary_complex_number_l65_65545

variable (a : ℝ)

def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem pure_imaginary_complex_number (h : pure_imaginary ((3 - complex.i) * (a + 2 * complex.i))) :
  a = -2 / 3 :=
by
  sorry

end pure_imaginary_complex_number_l65_65545


namespace work_completion_days_l65_65363

theorem work_completion_days (A B : Type) (A_work_rate B_work_rate : ℝ) :
  (1 / 16 : ℝ) = (1 / 20) + A_work_rate → B_work_rate = (1 / 80) := by
  sorry

end work_completion_days_l65_65363


namespace squared_length_of_n_l65_65290

def p (x : ℝ) := -x + 1
def q (x : ℝ) := x + 1
def r (x : ℝ) := (3 : ℝ)

def n (x : ℝ) := min (p x) (min (q x) (r x))

theorem squared_length_of_n :
  ((sqrt (8) + 4 + sqrt (8))^2 = 48 + 16 * sqrt (8)) :=
by 
suffices h : sqrt (8) = 2 * sqrt (2), by {
  rw [←h, mul_assoc, add_assoc, ←mul_assoc 2, ←mul_add, add_comm],
  norm_num,
  rw [sqrt_mul, sqrt_eq_orig_sqrt 2],
  norm_num,
 },
 have aux : (2 : ℝ) ≥ 0 := le_of_eq rfl,
 rw [sqrt_eq_orig_sqrt, sqrt_mul],
 norm_num,
 have : sqrt (4 : ℝ) = 2,
  by rw sqrt_eq_orig_sqrt; norm_num,
 rw this,
 norm_num
 sorry
 
end squared_length_of_n_l65_65290


namespace evaluate_nested_function_l65_65846

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 3 * x - 1 / 2 else 2 ^ x

theorem evaluate_nested_function : f (f (1 / 2)) = 2 := 
by
  sorry

end evaluate_nested_function_l65_65846


namespace weeks_to_buy_bicycle_l65_65410

-- Definitions based on problem conditions
def hourly_wage : Int := 5
def hours_monday : Int := 2
def hours_wednesday : Int := 1
def hours_friday : Int := 3
def weekly_hours : Int := hours_monday + hours_wednesday + hours_friday
def weekly_earnings : Int := weekly_hours * hourly_wage
def bicycle_cost : Int := 180

-- Statement of the theorem to prove
theorem weeks_to_buy_bicycle : ∃ w : Nat, w * weekly_earnings = bicycle_cost :=
by
  -- Since this is a statement only, the proof is omitted
  sorry

end weeks_to_buy_bicycle_l65_65410


namespace eval_expression_l65_65030

-- Define the expression to be evaluated
def expression := 4 * Real.log 2 + 3 * Real.log 5 - Real.log (1 / 5)

-- Theorem statement
theorem eval_expression : expression = 4 := by
  -- Proof of this theorem would go here.
  sorry

end eval_expression_l65_65030


namespace find_polar_circle_equation_l65_65564

variable (α θ ρ : ℝ)

def parametric_circle (α : ℝ) : Prop := 
  ∃ (x y : ℝ), x = 5 * cos α ∧ y = -6 + 5 * sin α

def polar_line (α_0 : ℝ) : Prop := 
  ∃ (θ : ℝ), θ = α_0 ∧ tan α_0 = sqrt 5 / 2

def polar_circle_equation (ρ θ : ℝ) : Prop := 
  ρ^2 + 12 * ρ * sin θ + 11 = 0

theorem find_polar_circle_equation (α θ ρ : ℝ) :
  (parametric_circle α) →
  (polar_circle_equation ρ θ) := sorry

lemma line_intersects_circle (α_0 : ℝ) :
  tan α_0 = (sqrt 5) / 2 →
  ∃ (ρ_1 ρ_2 : ℝ),
  (polar_circle_equation ρ_1 α_0 ∧ polar_circle_equation ρ_2 α_0) ∧
  abs (ρ_1 - ρ_2) = 6 := sorry

end find_polar_circle_equation_l65_65564


namespace angle_difference_l65_65225

theorem angle_difference (P Q R S : Type) [Angle P Q R = 40] [Angle P Q S = 28] :
  Angle S Q R = 12 :=
sorry

end angle_difference_l65_65225


namespace john_ate_12_ounces_of_steak_l65_65919

-- Conditions
def original_weight : ℝ := 30
def burned_fraction : ℝ := 0.5
def eaten_fraction : ℝ := 0.8

-- Theorem statement
theorem john_ate_12_ounces_of_steak :
  (original_weight * (1 - burned_fraction) * eaten_fraction) = 12 := by
  sorry

end john_ate_12_ounces_of_steak_l65_65919


namespace minimum_YP_PQ_QZ_l65_65885

def triangle (A B C : Type) [metric_space A B C] := ∃ (u v w : A B C), u ≠ v ∧ v ≠ w ∧ w ≠ u

variables (X Y Z : ℝ) -- assuming real coordinates for simplicity
variable (P : ℝ)
variable (Q : ℝ)

-- Conditions
def angle_XYZ := 50 -- angle in degrees, needs conversion if we use radians
def XY := 8
def XZ := 5
variables {YP PQ QZ : ℝ} -- lengths of the segments

theorem minimum_YP_PQ_QZ : 
  ∃ (P Q : ℝ), P ∈ XY ∧ Q ∈ XZ ∧ 
  (∀ (Ys Zq : ℝ), Ys ∈ XY → Zq ∈ XZ → YP + PQ + QZ = 6.53) := 
sorry

end minimum_YP_PQ_QZ_l65_65885


namespace geometric_sequence_sum_l65_65483

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) 
    (h1 : a 1 * q + a 1 * q^2 = 2) 
    (h2 : a 1 * q^2 + a 1 * q^3 = 1) 
    (S : ℕ → ℝ := λ n, a 1 * (1 - q^(n + 1)) / (1 - q)): 
    lim (λ n, S n) at_top = 16 / 3 :=
by
  sorry

end geometric_sequence_sum_l65_65483


namespace slope_of_dividing_line_l65_65423

noncomputable def parallelogram_vertices : list (ℝ × ℝ) := [(15,60), (15,144), (36,192), (36,108)]

theorem slope_of_dividing_line (m n : ℤ) (h : Int.gcd m n = 1) :
  let slope := (128 : ℚ) / 17 in
  slope = m / n → m + n = 145 :=
by
  intros
  sorry

end slope_of_dividing_line_l65_65423


namespace number_of_valid_arrangements_l65_65905

def is_valid_triplet (a b c : ℕ) : Prop :=
  (a * c - b ^ 2) % 7 = 0

def is_valid_sequence (s : List ℕ) : Prop :=
  ∀ (i : ℕ), i + 2 < s.length → is_valid_triplet (s.get i) (s.get (i + 1)) (s.get (i + 2))

theorem number_of_valid_arrangements : 
  (List.permutations [1, 2, 3, 4, 5, 6]).countp is_valid_sequence = 12 := 
sorry

end number_of_valid_arrangements_l65_65905


namespace domain_of_f_l65_65038

noncomputable def domain (f : ℝ → ℝ) (S : set ℝ) : Prop :=
∀ x, x ∈ S ↔ ∃ y, f y = x

theorem domain_of_f :
  ∀ x : ℝ, x ∉ {1, 2} ↔ ∃ y : ℝ, f y = x :=
by
  let f := λ x, 1 / (x^2 - 3 * x + 2)
  have h : ∀ x, x^2 - 3 * x + 2 = (x - 1) * (x - 2), from by
    sorry
  sorry

end domain_of_f_l65_65038


namespace triangle_med_slope_area_l65_65672

noncomputable def triangleYXZ (X Y : ℝ × ℝ) (A : ℝ) := (X = (10, 15)) ∧ (Y = (20, 17)) ∧ (A = 50)

theorem triangle_med_slope_area (r s : ℝ) : triangleYXZ (10, 15) (20, 17) 50 → 
  (∃ Z : ℝ × ℝ, Z = (r, s) ∧ (∃ m : ℝ, m = (-3) ∧ 
  ∃ M : ℝ × ℝ, M = ((X.1 + Y.1)/2, (X.2 + Y.2)/2) ∧ 
  2 * 50 = abs (r * (17 - 15) + 10 * (s - 17) + 20 * (15 - s))) → 
  r + s ≤ 29.26 :=
  sorry

end triangle_med_slope_area_l65_65672


namespace min_a10_l65_65676

noncomputable def sigma (S : Set ℕ) : ℕ := S.sum id

theorem min_a10 (A : Set ℕ)
  (h1 : A = {a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10, a_11})
  (h2 : a_1 < a_2) (h3 : a_2 < a_3) (h4 : a_3 < a_4) (h5 : a_4 < a_5)
  (h6 : a_5 < a_6) (h7 : a_6 < a_7) (h8 : a_7 < a_8) (h9 : a_8 < a_9)
  (h10 : a_9 < a_10) (h11 : a_10 < a_11)
  (hSum : ∀ n, n ≤ 1500 → ∃ S ⊆ A, sigma S = n) :
  a_10 = 248 :=
sorry

end min_a10_l65_65676


namespace organization_members_count_l65_65555

theorem organization_members_count (num_committees : ℕ) (pair_membership : ℕ → ℕ → ℕ) :
  num_committees = 5 →
  (∀ i j k l : ℕ, i ≠ j → k ≠ l → pair_membership i j = pair_membership k l → i = k ∧ j = l ∨ i = l ∧ j = k) →
  ∃ (num_members : ℕ), num_members = 10 :=
by
  sorry

end organization_members_count_l65_65555


namespace exist_a_sequence_l65_65485

theorem exist_a_sequence (n : ℕ) (h : n ≥ 2) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i ∧ x i ≤ 1) :
  ∃ (a : Fin (n+1) → ℝ), (a 0 + a n = 0) ∧ (∀ i, |a i| ≤ 1) ∧ (∀ i : Fin n, |a i.succ - a i| = x i) :=
by
  sorry

end exist_a_sequence_l65_65485


namespace EC_length_proof_l65_65493

theorem EC_length_proof :
  ∀ (A B C D E : Type) 
  (m : Type) 
  [angle A 45] 
  [segment BC (8 * sqrt 2)] 
  [perpendicular BD AC] 
  [perpendicular CE AB] 
  [angle_eq (angle DBC) (2 * angle ECB)] 
  [segment_length EC 8] 
  (a b : ℕ),
  b = 1 → a = 8 → a + b = 9 :=
begin
  intros,
  sorry
end

end EC_length_proof_l65_65493


namespace ticket_divisors_count_l65_65015

theorem ticket_divisors_count :
  let x_values := {x : ℕ | x ∣ 60 ∧ x ∣ 90}
  finset.card x_values = 8 :=
by
  sorry

end ticket_divisors_count_l65_65015


namespace cube_volume_l65_65706

theorem cube_volume (length width : ℝ) (h_length : length = 48) (h_width : width = 72) :
  let area := length * width
  let side_length_in_inches := Real.sqrt (area / 6)
  let side_length_in_feet := side_length_in_inches / 12
  let volume := side_length_in_feet ^ 3
  volume = 8 :=
by
  sorry

end cube_volume_l65_65706


namespace find_height_of_smaller_cuboid_l65_65530

theorem find_height_of_smaller_cuboid :
  ∃ h : ℝ, (h > 0 ∧ 32 * (5 * 4 * h) = 16 * 10 * 12) := 
begin
  use 3,
  split,
  { norm_num },
  { norm_num }
end

end find_height_of_smaller_cuboid_l65_65530


namespace find_t_of_symmetric_domain_l65_65817

noncomputable def even_function {f : ℝ → ℝ} (hf : ∀ x, f x = f (-x)) 
  (domain : Set ℝ) (h_domain : domain = Set.Icc (2-4 : ℝ) (2 : ℝ)) : Prop :=
  domain = Set.Icc (-2 : ℝ) (2 : ℝ)

theorem find_t_of_symmetric_domain (f : ℝ → ℝ) 
  (h_even : ∀ x, f x = f (-x)) 
  (domain : Set ℝ) 
  (h_domain : domain = Set.Icc (t-4) t) : t = 2 :=
begin
  sorry
end

end find_t_of_symmetric_domain_l65_65817


namespace smallest_value_of_expression_l65_65058

variable (a b c : ℝ)
variable (hab : a > b)
variable (hbc : b > c)
variable (ha_nonzero : a ≠ 0)

theorem smallest_value_of_expression :
  ∃ (x : ℝ), x = 6 ∧ 
  (∀ a b c : ℝ, a > b → b > c → a ≠ 0 →
  (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 ≥ x) :=
begin
  sorry
end

end smallest_value_of_expression_l65_65058


namespace perpendicular_KL_HM_l65_65476

variable {A B C D K L M H : Point}

-- Define the conditions as Lean hypotheses
axiom hAB_eq_BC : dist A B = dist B C
axiom hAD_eq_DC : dist A D = dist D C
axiom midpoint_K : midpoint K A B
axiom midpoint_L : midpoint L C D
axiom midpoint_M : midpoint M A C
axiom perp_A_to_BC : ∃ S, ∀ P, perpendicular (line_through A S) (line_through B C)
axiom perp_C_to_AD : ∃ P, ∀ Q, perpendicular (line_through C P) (line_through A D) ∧ intersection (line_through A P) (line_through C Q) = H

-- Proof goal: KL and HM are perpendicular
theorem perpendicular_KL_HM : perpendicular (line_through K L) (line_through H M) := sorry

end perpendicular_KL_HM_l65_65476


namespace problem1_problem2_l65_65031

section

-- Problem 1
theorem problem1 :
  (1 * (2.25)^(1/2) - (-9.6)^0 - (27/8)^(-(2/3)) + (1.5)^(-2) = 1/2) :=
by
  sorry

-- Problem 2
theorem problem2 :
  (1/2 * log 10 25 + log 10 2 - log 10 (sqrt 0.1) - (log 2 9 * log 3 2) = -1/2) :=
by
  sorry

end

end problem1_problem2_l65_65031


namespace Eva_arts_marks_difference_l65_65047

noncomputable def marks_difference_in_arts : ℕ := 
  let M1 := 90
  let A2 := 90
  let S1 := 60
  let M2 := 80
  let A1 := A2 - 75
  let S2 := 90
  A2 - A1

theorem Eva_arts_marks_difference : marks_difference_in_arts = 75 := by
  sorry

end Eva_arts_marks_difference_l65_65047


namespace max_NF_value_AN_slope_value_l65_65816

-- Let us define the conditions of the problem first:
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def major_axis_relation (a b : ℝ) : Prop := 2 * a = (3 * real.sqrt 5 / 5) * 2 * b
def point_on_ellipse (a b : ℝ) (x y : ℝ) : Prop := ellipse a b x y
def vertex_left (a b : ℝ) : Prop := a > b > 0

-- Problem Ⅰ: Maximum value of |NF|
noncomputable def right_focus (a b : ℝ) : ℝ × ℝ := (real.sqrt (a^2 - b^2), 0)
def max_NF (a b : ℝ) : ℝ := a + real.sqrt (a^2 - b^2)

theorem max_NF_value
  (a b : ℝ)
  (h1 : vertex_left a b)
  (h2 : major_axis_relation a b)
  (h3 : point_on_ellipse a b (-1) (2 * real.sqrt 10 / 3)) :
  max_NF a b = 5 := sorry

-- Problem Ⅱ: Slope of the line AN
def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)
def AN_slope (a b : ℝ) : ℝ := line_slope (-a) 0 (3 / 4) (5 * real.sqrt 3 / 4)

theorem AN_slope_value
  (a b : ℝ)
  (h1 : vertex_left a b)
  (h2 : major_axis_relation a b)
  (h3 : point_on_ellipse a b (-1) (2 * real.sqrt 10 / 3)) :
  AN_slope a b = 5 * real.sqrt 3 / 3 := sorry

end max_NF_value_AN_slope_value_l65_65816


namespace min_folds_to_exceed_thickness_l65_65463

def initial_thickness : ℝ := 0.1
def desired_thickness : ℝ := 12

theorem min_folds_to_exceed_thickness : ∃ (n : ℕ), initial_thickness * 2^n > desired_thickness ∧ ∀ m < n, initial_thickness * 2^m ≤ desired_thickness := by
  sorry

end min_folds_to_exceed_thickness_l65_65463


namespace quadratic_root_m_value_l65_65879

theorem quadratic_root_m_value (m : ℝ) : (x^2 + (m+2)*x - 2 = 0) → (1:ℝ) is root ↔ m = -1 := sorry

end quadratic_root_m_value_l65_65879


namespace lateral_surface_area_cone_l65_65836

-- Given definitions (conditions)
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Question transformed into a theorem statement
theorem lateral_surface_area_cone : 
  ∀ (r l : ℝ), r = radius → l = slant_height → (1 / 2) * (2 * real.pi * r) * l = 15 * real.pi := 
by 
  intros r l hr hl
  rw [hr, hl]
  sorry

end lateral_surface_area_cone_l65_65836


namespace repay_loan_with_interest_l65_65217

theorem repay_loan_with_interest (amount_borrowed : ℝ) (interest_rate : ℝ) (total_payment : ℝ) 
  (h1 : amount_borrowed = 100) (h2 : interest_rate = 0.10) :
  total_payment = amount_borrowed + (amount_borrowed * interest_rate) :=
by sorry

end repay_loan_with_interest_l65_65217


namespace seating_arrangements_l65_65669

theorem seating_arrangements {n : ℕ} :
  n = 7 → (number of ways to arrange persons A and B with at least one empty seat between them) = 30 :=
by sorry

end seating_arrangements_l65_65669


namespace find_angle_θ_l65_65283

-- Define the given constants and conditions
constants (a : ℝ) -- distance between Ships A and B in nautical miles
constants (θ : ℝ) -- the angle east of north Ship A must head towards
constants (speed_B : ℝ) -- speed of Ship B (nautical miles per hour)
constants (speed_A : ℝ := sqrt 3 * speed_B) -- speed of Ship A

-- Directions and angles
axiom angle_B_direction : 60 -- direction of Ship B in degrees east of north from Ship A
axiom angle_CAB : angle_B_direction - θ -- ∠CAB

-- Applying trigonometric equations based on the given information and conditions
axiom angle_B : 120 -- ∠B
axiom sin_angle_B : Real.sin 120 = Real.sin (180 - 120) = Real.sin 60

-- Law of Sines relationship
axiom law_of_sines (x : ℝ) : 
  x / Real.sin (angle_B_direction - θ) = sqrt 3 * x / Real.sin 120

-- Simplification and conclusion
axiom simplify_eq : 
  Real.sin (60 - θ) = 1/2 -- derived from the equation in the solution steps

-- The expected angle θ to catch Ship B as quickly as possible
theorem find_angle_θ : θ = 30 :=
sorry

end find_angle_θ_l65_65283


namespace polygon_parallel_sides_l65_65037

noncomputable def is_inscribed_in_circle (polygon : List (ℝ × ℝ)) : Prop := sorry  -- Define the function to check if all vertices are on a common circle

noncomputable def are_n_minus_1_opposite_pairs_parallel (polygon : List (ℝ × ℝ)) (n : ℕ) : Prop := sorry  -- Define the function to check parallel pairs

theorem polygon_parallel_sides (n : ℕ) (h1 : n % 2 = 1)  -- n is odd
  (polygon : List (ℝ × ℝ))
  (h2 : polygon.length = 2 * n)  -- polygon has 2n sides
  (h3 : is_inscribed_in_circle polygon)  -- polygon is inscribed in a circle
  (h4 : are_n_minus_1_opposite_pairs_parallel polygon n)  -- n-1 pairs of opposite sides are parallel
  : are_parallel (polygon.get ⟨n-1, sorry⟩)  -- A_nA_{n+1} 
                (polygon.get ⟨2*n-1, sorry⟩) :=  -- A_{2n}A_1
sorry

end polygon_parallel_sides_l65_65037


namespace part_I_part_II_l65_65517

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (2 * x^2 - 3 * x)

theorem part_I : ∀ x : ℝ, f x > 0 ↔ (x < 0 ∨ x > 3 / 2) :=
by
  intro x
  have hx : Real.exp x > 0 := Real.exp_pos x
  simp [f, mul_pos_iff, hx]
  sorry

theorem part_II :
  ∃ min_val max_val : ℝ, 
    (∀ x ∈ Icc 0 2, f x ≥ min_val ∧ f x ≤ max_val) ∧
    (min_val = -Real.exp 1) ∧ 
    (max_val = 2 * Real.exp 2) :=
by
  let critical_points := {0, 1, 2}
  have eval_zero : f 0 = 0 := by simp [f]
  have eval_one : f 1 = -Real.exp 1 := by simp [f, Real.exp]
  have eval_two : f 2 = 2 * Real.exp 2 := by simp [f, Real.exp]
  let min_val := min (min (f 0) (f 1)) (f 2)
  let max_val := max (max (f 0) (f 1)) (f 2)
  use min_val, max_val
  split
  {
    intros x hx,
    split,
    { apply Real.monotonic? f min_val hx },
    { apply Real.monotonic? f max_val hx }
  },
  {
    have : min_val = -Real.exp 1 := by simp [min_val, eval_zero, eval_one, eval_two]
    exact this
  },
  {
    have : max_val = 2 * Real.exp 2 := by simp [max_val, eval_zero, eval_one, eval_two]
    exact this
  }
  sorry

end part_I_part_II_l65_65517


namespace proposition_relationship_l65_65883
-- Import library

-- Statement of the problem
theorem proposition_relationship (p q : Prop) (hpq : p ∨ q) (hnp : ¬p) : ¬p ∧ q :=
  by
  sorry

end proposition_relationship_l65_65883


namespace problem1_problem2_problem3_problem4_problem5_l65_65763

noncomputable theory

-- Problem 1
theorem problem1 : -3 * 1 / 4 - (-1 / 9) + (-3 / 4) + 1 * 8 / 9 = -2 := by
  sorry

-- Problem 2
theorem problem2 : (-50) * (2 * 1 / 5 - 3 * 3 / 10 + 1 * 7 / 25) = -9 := by
  sorry

-- Problem 3
theorem problem3 : (3 / 5 - 1 / 2 - 7 / 12) * (60 * 3 / 7 - 60 * 1 / 7 + 60 * 5 / 7) = -29 := by
  sorry

-- Problem 4
theorem problem4 : (-3 / 2) ^ 3 * (-3 / 5) ^ 2 - 2 * 5 / 19 * 19 / 43 * (-1 * 1 / 2) ^ 3 + (4 / 5) ^ 2 * (-3 / 2) ^ 3 = 0 := by
  sorry

-- Problem 5
theorem problem5 : (∑ i in finset.range 49, 1 / ((i+1) * (i+2))) = 49 / 50 := by
  sorry

end problem1_problem2_problem3_problem4_problem5_l65_65763


namespace graph_behavior_of_g_l65_65311

theorem graph_behavior_of_g :
  ∀ x : ℝ, g(x) = x^2 - 2x - 8 → 
  ( ∃ a b c : ℝ, a = 1 ∧ b = -2 ∧ c = -8 ∧ a > 0 ) → 
  ( ∀ x : ℝ, (g (x) → ∞ → g (x) = x^2 - 2x - 8) ∧ (g (x) → -∞ → g (x) = x^2 - 2x - 8) ) → 
  (g(x) = x^2 -2x - 8 → x ∈ ℝ → (up_to_the_right_and_left: Prop), sorry) :=
begin
  sorry
end

end graph_behavior_of_g_l65_65311


namespace find_apron_cost_l65_65050

-- Definitions used in the conditions
variables (hand_mitts cost small_knife utensils apron : ℝ)
variables (nieces : ℕ)
variables (total_cost_before_discount total_cost_after_discount : ℝ)

-- Conditions given
def conditions := 
  hand_mitts = 14 ∧ 
  utensils = 10 ∧ 
  small_knife = 2 * utensils ∧
  (total_cost_before_discount : ℝ) = (3 * hand_mitts + 3 * utensils + 3 * small_knife + 3 * apron) ∧
  (total_cost_after_discount : ℝ) = 135 ∧
  total_cost_before_discount * 0.75 = total_cost_after_discount ∧
  nieces = 3

-- Theorem statement (proof problem)
theorem find_apron_cost (h : conditions hand_mitts utensils small_knife apron nieces total_cost_before_discount total_cost_after_discount) : 
  apron = 16 :=
by 
  sorry

end find_apron_cost_l65_65050


namespace triangle_sides_length_a_triangle_perimeter_l65_65228

theorem triangle_sides_length_a (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) :
  a = Real.sqrt 3 :=
sorry

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) 
  (hA : A = π / 3) (h1 : (b + c) / (Real.sin B + Real.sin C) = 2) 
  (h2 : (b * c * Real.sin (π / 3)) / 2 = Real.sqrt 3 / 2) :
  a + b + c = 3 + Real.sqrt 3 :=
sorry

end triangle_sides_length_a_triangle_perimeter_l65_65228


namespace no_solutions_rebus_l65_65971

theorem no_solutions_rebus : ∀ (K U S Y : ℕ), 
  (K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y) →
  (∀ d, d < 10) → 
  let KUSY := 1000 * K + 100 * U + 10 * S + Y in
  let UKSY := 1000 * U + 100 * K + 10 * S + Y in
  let result := 10000 * U + 1000 * K + 100 * S + 10 * Y + S in
  KUSY + UKSY ≠ result :=
begin
  sorry
end

end no_solutions_rebus_l65_65971


namespace volume_of_space_not_occupied_by_cones_l65_65731

noncomputable def cylinder_height : ℝ := 36
noncomputable def cylinder_radius : ℝ := 10
noncomputable def cone_height : ℝ := 18
noncomputable def cone_radius : ℝ := 10

theorem volume_of_space_not_occupied_by_cones :
  let V_cylinder := π * cylinder_radius^2 * cylinder_height,
      V_cone := (1 / 3) * π * cone_radius^2 * cone_height
  in
  V_cylinder - 3 * V_cone = 1800 * π :=
by
  sorry

end volume_of_space_not_occupied_by_cones_l65_65731


namespace intersect_A_B_complement_l65_65608

-- Define the sets A and B
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | x > 1}

-- Find the complement of B in ℝ
def B_complement := {x : ℝ | x ≤ 1}

-- Prove that the intersection of A and the complement of B is equal to (-1, 1]
theorem intersect_A_B_complement : A ∩ B_complement = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by
  -- Proof is to be provided
  sorry

end intersect_A_B_complement_l65_65608


namespace parabolas_similar_l65_65274

theorem parabolas_similar :
  ∀ x y : ℝ, 
  (¬((y = x^2) ∧ (y = 2 * x^2)) → ∀ a : ℝ, 2 * (a, 2 * a^2) ∈ set_of (λ p : ℝ × ℝ, p.snd = p.fst ^ 2)) := 
  sorry

end parabolas_similar_l65_65274


namespace number_of_integer_points_l65_65953

theorem number_of_integer_points (T : ℕ) :
  {p : ℤ × ℤ // p.1 ^ 2 + p.2 ^ 2 < 10}.card = 29 :=
sorry

end number_of_integer_points_l65_65953


namespace sphere_surface_area_diameter_4_l65_65500

noncomputable def sphere_surface_area (d : ℝ) : ℝ :=
  4 * Real.pi * (d / 2) ^ 2

theorem sphere_surface_area_diameter_4 :
  sphere_surface_area 4 = 16 * Real.pi :=
by
  sorry

end sphere_surface_area_diameter_4_l65_65500


namespace balloons_difference_l65_65021

-- Define the balloons each person brought
def Allan_red := 150
def Allan_blue_total := 75
def Allan_forgotten_blue := 25
def Allan_green := 30

def Jake_red := 100
def Jake_blue := 50
def Jake_green := 45

-- Calculate the actual balloons Allan brought to the park
def Allan_blue := Allan_blue_total - Allan_forgotten_blue
def Allan_total := Allan_red + Allan_blue + Allan_green

-- Calculate the total number of balloons Jake brought
def Jake_total := Jake_red + Jake_blue + Jake_green

-- State the problem: Prove Allan distributed 35 more balloons than Jake
theorem balloons_difference : Allan_total - Jake_total = 35 := 
by
  sorry

end balloons_difference_l65_65021


namespace number_of_toys_sold_l65_65391

theorem number_of_toys_sold (SP gain CP : ℕ) 
    (h_SP : SP = 21000) 
    (h_gain : gain = CP * 3) 
    (h_CP : CP = 1000) 
    : SP - gain = 18 * CP :=
by
  have h1 : gain = 3 * 1000 := by rw [h_CP, mul_comm, h_gain]
  have h2 : 21000 - 3000 = 18 * 1000 := by norm_num
  rw [h1, h2]
  sorry

end number_of_toys_sold_l65_65391


namespace bob_pennies_l65_65153

theorem bob_pennies (a b : ℕ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  have h3 : 4 * a - b = 5, from sorry,
  have h4 : b - 3 * a = 4, from sorry,
  have h5 : 4 * a - 3 * a = 9, from sorry,
  have h6 : a = 9, from sorry,
  have h7 : b + 1 = 36 - 4, from sorry,
  have h8 : b + 1 = 32, from sorry,
  have h9 : b = 31, from sorry,
  exact h9

end bob_pennies_l65_65153


namespace book_total_pages_l65_65144

theorem book_total_pages (x : ℝ) 
  (h1 : ∀ d1 : ℝ, d1 = x * (1/6) + 10)
  (h2 : ∀ remaining1 : ℝ, remaining1 = x - d1)
  (h3 : ∀ d2 : ℝ, d2 = remaining1 * (1/5) + 12)
  (h4 : ∀ remaining2 : ℝ, remaining2 = remaining1 - d2)
  (h5 : ∀ d3 : ℝ, d3 = remaining2 * (1/4) + 14)
  (h6 : ∀ remaining3 : ℝ, remaining3 = remaining2 - d3)
  (h7 : remaining3 = 52) : x = 169 := sorry

end book_total_pages_l65_65144


namespace delta_value_l65_65869

theorem delta_value : ∃ Δ : ℤ, 4 * (-3) = Δ + 5 ∧ Δ = -17 := 
by
  use -17
  sorry

end delta_value_l65_65869


namespace distance_from_A_to_B_l65_65651

theorem distance_from_A_to_B (D : ℝ) :
  (∃ D, (∀ tC, tC = D / 30) 
      ∧ (∀ tD, tD = D / 48 ∧ tD < (D / 30 - 1.5))
      ∧ D = 120) :=
by
  sorry

end distance_from_A_to_B_l65_65651


namespace Petya_chips_l65_65452

theorem Petya_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) :
  ∃ T : ℕ, T = 24 :=
by {
  let T_triangle := 3 * x - 3,
  let T_square := 4 * y - 4,
  -- The conditions ensure T_triangle = T_square
  have h3 : T_triangle = T_square, from h2,
  -- substituting y = x - 2 into T_square
  have h4 : T_square = 4 * (x - 2) - 4, from calc
    T_square = 4 * y - 4 : by rfl
    ... = 4 * (x - 2) - 4 : by rw h1,
  -- simplify to find x,
  have h5 : 3 * x - 3 = 4 * (x - 2) - 4, from h2,
  have h6 : 3 * x - 3 = 4 * x - 8 - 4, from h5,
  have h7 : 3 * x - 3 = 4 * x - 12, from by simp at h6,
  have h8 : -3 = x - 12, from by linarith,
  have h9 : x = 9, from by linarith,
  -- Find the total number of chips
  let T := 3 * x - 3,
  have h10 : T = 24, from calc
    T = 3 * 9 - 3 : by rw h9
    ... = 24 : by simp,
  exact ⟨24, h10⟩
}

end Petya_chips_l65_65452


namespace probability_dmitry_before_anatoly_l65_65915

theorem probability_dmitry_before_anatoly (m : ℝ) (non_neg_m : 0 < m) :
  let volume_prism := (m^3) / 2
  let volume_tetrahedron := (m^3) / 3
  let probability := volume_tetrahedron / volume_prism
  probability = (2 : ℝ) / 3 :=
by
  sorry

end probability_dmitry_before_anatoly_l65_65915


namespace proof_problem_l65_65197

variables (x_A x_B m : ℝ)

-- Condition 1:
def cost_relation : Prop := x_B = x_A + 20

-- Condition 2:
def quantity_relation : Prop := 540 / x_A = 780 / x_B

-- Condition 3:
def total_books := 70
def total_cost := 3550
def min_books_relation : Prop := 45 * m + 65 * (total_books - m) ≤ total_cost 

-- Part 1:
def cost_price_A (x : ℝ) : Prop := x = 45
def cost_price_B (x : ℝ) : Prop := x = 65

-- Part 2:
def min_books_A (m : ℝ) : Prop := m ≥ 50

-- Define the proof problem
theorem proof_problem :
  (cost_relation x_A x_B) ∧ 
  (quantity_relation x_A x_B) →
  (cost_price_A x_A) ∧ 
  (cost_price_B x_B) ∧ 
  (min_books_relation x_A x_B m) → 
  (min_books_A m) :=
by sorry

end proof_problem_l65_65197


namespace sum_valid_numbers_eq_468_l65_65064

def is_valid_number (n : ℕ) : Prop :=
  n % 4 = 0 ∧ n % 3 = 1 ∧ 1 ≤ n ∧ n ≤ 100

def valid_numbers : List ℕ :=
  List.filter is_valid_number (List.range' 1 100)

theorem sum_valid_numbers_eq_468 :
  (valid_numbers.sum = 468) :=
by
  sorry

end sum_valid_numbers_eq_468_l65_65064


namespace number_of_possible_b_l65_65430

theorem number_of_possible_b : 
  ∃ b : ℤ, (finset.card (finset.filter (λ x : ℤ, x^2 + b * x + 1 ≤ 0) (finset.Icc (-5) 5)) = 4) ∧ 
  (finset.card (finset.filter (λ b : ℤ, (finset.card (finset.filter (λ x : ℤ, x^2 + b * x + 1 ≤ 0) (finset.Icc (-5) 5)) = 4)) (finset.Icc (-5) 5)) = 2) := sorry

end number_of_possible_b_l65_65430


namespace number_of_technicians_l65_65558

/-- 
In a workshop, the average salary of all the workers is Rs. 8000. 
The average salary of some technicians is Rs. 12000 and the average salary of the rest is Rs. 6000. 
The total number of workers in the workshop is 24.
Prove that there are 8 technicians in the workshop.
-/
theorem number_of_technicians 
  (total_workers : ℕ) 
  (avg_salary_all : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) 
  (num_technicians rest_workers : ℕ) 
  (h_total : total_workers = num_technicians + rest_workers)
  (h_avg_salary : (num_technicians * avg_salary_technicians + rest_workers * avg_salary_rest) = total_workers * avg_salary_all)
  (h1 : total_workers = 24)
  (h2 : avg_salary_all = 8000)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_rest = 6000) :
  num_technicians = 8 :=
by
  sorry

end number_of_technicians_l65_65558


namespace total_pages_in_book_l65_65246

theorem total_pages_in_book :
  ∃ x : ℝ, (x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20)
           - (1/2 * ((x - (1/6 * x + 10) - (1/3 * (x - (1/6 * x + 10)) + 20))) + 25) = 120) ∧
           x = 552 :=
by
  sorry

end total_pages_in_book_l65_65246


namespace factorization_correct_l65_65790

theorem factorization_correct : ∃ a b : ℤ, (5*y + a)*(y + b) = 5*y^2 + 17*y + 6 ∧ a - b = -1 := by
  sorry

end factorization_correct_l65_65790


namespace radius_touches_segments_and_arc_l65_65661

noncomputable def radius_of_tangent_circle
  (r a : ℝ) (α : ℝ) (h1 : α < 90) : ℝ :=
  let cot_half_alpha := cot (α / 2)
  in (cot_half_alpha ^ 2) * ((sqrt (r ^ 2 + a * r * sin α)) / (sin (α / 2)) - a * cot_half_alpha - r)

theorem radius_touches_segments_and_arc
  {r a α : ℝ} (h1 : α < 90) :
  ∃ x : ℝ, x = radius_of_tangent_circle r a α h1 :=
sorry

end radius_touches_segments_and_arc_l65_65661


namespace john_can_buy_notebooks_l65_65586

theorem john_can_buy_notebooks (john_money : ℕ) (notebook_cost : ℕ) (h_john_money : john_money = 4575) (h_notebook_cost : notebook_cost = 325) : 
  ∃ n : ℕ, 325 * n ≤ 4575 ∧ (∀ m : ℕ, 325 * m ≤ 4575 → m ≤ n) ∧ n = 14 :=
by
  exists 14
  split
  · norm_num, simp [h_john_money, h_notebook_cost] at * {contextual := tt}, sorry
  split
  · intros m h, sorry
  · refl

end john_can_buy_notebooks_l65_65586


namespace slope_DD_l65_65335

open Real

def point (x y : ℝ) := (x, y)
def reflect_over_y_eq_2x (pt : ℝ × ℝ) : ℝ × ℝ := (pt.2 / 2, pt.1 / 2)

def D := point 3 2
def E := point 5 4
def F := point 2 6

def D' := reflect_over_y_eq_2x D
def E' := reflect_over_y_eq_2x E
def F' := reflect_over_y_eq_2x F

def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

theorem slope_DD'_neq_neg_half : slope D D' ≠ -1 / 2 := by
  sorry

end slope_DD_l65_65335


namespace line_through_midpoints_passes_through_incircle_center_l65_65551

open EuclideanGeometry Triangle

variables {A B C X I : Point} -- vertices A, B, C, point of tangency X, and incircle center I
variables (triangle_ABC : Triangle A B C)
variables (inscribed_circle : InscribedCircle triangle_ABC)
variables (X_on_BC : IsTangentAt inscribed_circle X B C)

-- define midpoints M and L
def M := midpoint B C
def L := midpoint A X

-- statement of the problem
theorem line_through_midpoints_passes_through_incircle_center :
  joins (midpoint (segment A X)) (midpoint (segment B C)) I :=
sorry

end line_through_midpoints_passes_through_incircle_center_l65_65551


namespace Morgan_mean_score_l65_65081

def scores := [78, 82, 90, 95, 98, 102, 105]
def Alex_scores := [78, 82, 90, 95]
def Morgan_scores := [98, 102, 105]

def mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length : ℝ)

def Alex_mean := 91.5

theorem Morgan_mean_score : mean Morgan_scores = 94.67 :=
  by
    have Alex_sum : (Alex_scores.sum : ℝ) = 4 * Alex_mean :=
      by
        sorry
    have total_sum : (scores.sum : ℝ) = 650 :=
      by
        sorry
    have Morgan_sum : (Morgan_scores.sum : ℝ) = total_sum - Alex_sum :=
      by
        sorry
    show mean Morgan_scores = 94.67
    by
      sorry

end Morgan_mean_score_l65_65081


namespace circle_radius_l65_65803

theorem circle_radius {r : ℝ} : 
  (∀ x: ℝ, (x+sqrt(r))^2 - r = x^2 + 2*x*sqrt(r)) ∧ 
  (∃ t: ℝ, t^2 + 2*t*sqrt(r) = t) ∧ 
  (x: ℝ, x(x + 2*sqrt(r) - 1) = 0) →
  r = 1/4 :=
by
  sorry

end circle_radius_l65_65803


namespace minimize_b_plus_4c_l65_65191

noncomputable def triangle := Type

variable {ABC : triangle}
variable (a b c : ℝ) -- sides of the triangle
variable (BAC : ℝ) -- angle BAC
variable (D : triangle → ℝ) -- angle bisector intersecting BC at D
variable (AD : ℝ) -- length of AD
variable (min_bc : ℝ) -- minimum value of b + 4c

-- Conditions
variable (h1 : BAC = 120)
variable (h2 : D ABC = 1)
variable (h3 : AD = 1)

-- Proof statement
theorem minimize_b_plus_4c (h1 : BAC = 120) (h2 : D ABC = 1) (h3 : AD = 1) : min_bc = 9 := 
sorry

end minimize_b_plus_4c_l65_65191


namespace extreme_value_a_range_l65_65547

theorem extreme_value_a_range (a : ℝ) :
  (∃ x : ℝ, (1 / Real.exp 1 < x ∧ x < Real.exp 1 ∧ x + a * Real.log x + 1 + a / x = 0)) →
  -Real.exp 1 < a ∧ a < -1 / Real.exp 1 :=
by sorry

end extreme_value_a_range_l65_65547


namespace locus_of_points_l65_65600

noncomputable def A_circle_of_Apollonius_locus (A B C M : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] :=
  M ∈ circumcircle (internal_bisector_intersection A B C) (external_bisector_intersection A B C) A

theorem locus_of_points (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] :
  (locus_points : Type*) -> ∀ M : locus_points, (dist M B / dist M C) = (dist A B / dist A C) ↔ M ∈ A_circle_of_Apollonius_locus A B C :=
begin
  sorry
end

end locus_of_points_l65_65600


namespace symmetric_lines_intersect_at_point_l65_65973

open EuclideanGeometry

variables {A B C H H₁ H₂ H₃ : Point}
variables {l : Line}
variables (P₁ P₂ P₃ : Point)

def is_orthocenter (A B C H : Point) : Prop :=
-- H is the orthocenter of triangle ABC
sorry

def is_reflection (P₁ P₂ : Point) (l : Line) : Prop :=
-- P₂ is the reflection of P₁ with respect to line l
sorry

def on_circumcircle (A B C P : Point) : Prop :=
-- P lies on the circumcircle of triangle ABC
sorry

theorem symmetric_lines_intersect_at_point :
  is_orthocenter A B C H →
  is_reflection H H₁ (line_through B C) →
  is_reflection H H₂ (line_through C A) →
  is_reflection H H₃ (line_through A B) →
  on_circumcircle A B C H₁ →
  on_circumcircle A B C H₂ →
  on_circumcircle A B C H₃ →
  (∃ P : Point, ∀ (l : Line), intersects_at_single_point (symmetric_line H l (line_through B C)) 
                                           (symmetric_line H l (line_through C A)) 
                                           (symmetric_line H l (line_through A B)) P) :=
by
  sorry

end symmetric_lines_intersect_at_point_l65_65973


namespace possible_diagonal_lengths_l65_65078

theorem possible_diagonal_lengths (y : ℕ) :
  (y > 4 ∧ y < 20) ↔ 15 := by
  sorry

end possible_diagonal_lengths_l65_65078


namespace smallest_n_mod_1000_l65_65070

noncomputable def f (n : ℕ) : ℕ := (n.digits 5).sum

noncomputable def g (n : ℕ) : ℕ := (f n).digits 9.sum

theorem smallest_n_mod_1000 : ∃ n : ℕ, (g n).digits 17.any (λ d, d > 9) ∧ n % 1000 = 791 := by 
  sorry

end smallest_n_mod_1000_l65_65070


namespace geometric_sum_first_4_terms_l65_65664

theorem geometric_sum_first_4_terms :
  let a1 := 1
  let r := 2
  let n := 4
  S_4 = (a1 * (1 - r ^ n)) / (1 - r)
  implies S_4 = 15 :=
by
  let a1 := 1
  let r := 2
  let n := 4
  let S_4 := (a1 * (1 - r ^ n)) / (1 - r)
  have : S_4 = 15 := sorry
  exact this

end geometric_sum_first_4_terms_l65_65664


namespace discounted_price_per_bag_l65_65433

theorem discounted_price_per_bag
  (cost_per_bag : ℝ)
  (num_bags : ℕ)
  (initial_price : ℝ)
  (num_sold_initial : ℕ)
  (net_profit : ℝ)
  (discounted_revenue : ℝ)
  (discounted_price : ℝ) :
  cost_per_bag = 3.0 →
  num_bags = 20 →
  initial_price = 6.0 →
  num_sold_initial = 15 →
  net_profit = 50 →
  discounted_revenue = (net_profit + (num_bags * cost_per_bag) - (num_sold_initial * initial_price) ) →
  discounted_price = (discounted_revenue / (num_bags - num_sold_initial)) →
  discounted_price = 4.0 :=
by
  sorry

end discounted_price_per_bag_l65_65433


namespace additional_time_5_l65_65296

variables {M : ℝ}
variables {time_AB : ℝ} {D_AB : ℝ} {D_BC : ℝ} {D_AC : ℝ}

-- Assume the given conditions
def conditions : Prop :=
  time_AB = 7 ∧
  D_AB = D_BC + M ∧
  D_AC = 6 * M

-- Define the proof problem statement
theorem additional_time_5 (h : conditions) : 
  let v := D_AB / time_AB in
  let time_BC := D_BC / v in
  time_BC = 5 :=
begin
  sorry
end

end additional_time_5_l65_65296


namespace initial_pennies_indeterminate_l65_65281

-- Conditions
def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def mom_nickels : ℕ := 2
def total_nickels_now : ℕ := 18

-- Proof problem statement
theorem initial_pennies_indeterminate :
  ∀ (initial_nickels dad_nickels mom_nickels total_nickels_now : ℕ), 
  initial_nickels = 7 → dad_nickels = 9 → mom_nickels = 2 → total_nickels_now = 18 → 
  (∃ (initial_pennies : ℕ), true) → false :=
by
  sorry

end initial_pennies_indeterminate_l65_65281


namespace trigonometric_identity_l65_65642

variable (x y z : ℝ)

theorem trigonometric_identity :
  cos (x + y) * sin z + sin (x + y) * cos z = sin (x + y + z) :=
by
  sorry

end trigonometric_identity_l65_65642


namespace cos_double_angle_l65_65871

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 :=
by 
  sorry

end cos_double_angle_l65_65871


namespace group_division_count_l65_65779

theorem group_division_count (n : ℕ) :
  (number_of_ways_to_form_groups (2 * n) n) = (fact (2 * n)) / ((2 ^ n) * (fact n)) :=
sorry

end group_division_count_l65_65779


namespace calculate_moles_of_Be2C_necessary_l65_65114

def balanced_reaction1 : Prop :=
  ∀ (Be2C H2O BeOH2 CH4 : ℕ),
    Be2C + 4 * H2O → 2 * BeOH2 + CH4

def balanced_reaction2 : Prop :=
  ∀ (Be2C O2 BeO CO2 : ℕ),
    3 * Be2C + 8 * O2 → 6 * BeO + 4 * CO2

def stoichiometry_reaction1 : Prop :=
  ∀ (Be2C H2O : ℕ),
    H2O / Be2C = 4

def total_moles_of_Be2C (H2O O2 Be2C_required : ℕ) : Prop :=
  stoichiometry_reaction1 →
  H2O = 50 →
  O2 = 25 →
  Be2C_required = 12.5

theorem calculate_moles_of_Be2C_necessary 
  (H2O O2 Be2C_required : ℕ) : 
  total_moles_of_Be2C H2O O2 Be2C_required → 
  Be2C_required = 12.5 := 
sorry

end calculate_moles_of_Be2C_necessary_l65_65114


namespace consecutive_torchbearers_probability_l65_65578

theorem consecutive_torchbearers_probability :
  let torchbearers: Finset ℕ := {1, 2, 3, 4, 5}
  let total_ways := (torchbearers.card.choose 2)
  let favorable_ways := (torchbearers.filter (λ x y, abs (x - y) = 1)).card
  (favorable_ways.toNat / total_ways.toNat) = 2 / 5 := sorry

end consecutive_torchbearers_probability_l65_65578


namespace range_of_a_l65_65118

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) (h : ∀ x : ℝ, f x = -x^3 + a*x^2 - x - 1) :
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ↔ (-real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3) :=
by
  sorry

end range_of_a_l65_65118


namespace verify_asymptotes_l65_65187

def numerator (x : ℝ) : ℝ := x^2 - 2*x - 8
def denominator (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

theorem verify_asymptotes :
  let f (x) := numerator x / denominator x in 
  let a := 0 in   -- holes
  let b := 3 in   -- vertical asymptotes
  let c := 1 in   -- horizontal asymptotes
  let d := 0 in   -- oblique asymptotes
  a + 2*b + 3*c + 4*d = 9 :=
by 
  sorry

end verify_asymptotes_l65_65187


namespace larger_solution_quadratic_l65_65797

theorem larger_solution_quadratic : 
  ∀ x1 x2 : ℝ, (x^2 - 13 * x - 48 = 0) → x1 ≠ x2 → (x1 = 16 ∨ x2 = 16) → max x1 x2 = 16 :=
by
  sorry

end larger_solution_quadratic_l65_65797


namespace largest_n_for_positive_sum_l65_65105

/-- Given that {a_n} is an arithmetic sequence with the following conditions:
    1. a_1 > 0
    2. a_2016 + a_2017 > 0
    3. a_2016 * a_2017 < 0
Prove that the largest natural number n for which the sum of the first n terms S_n > 0 is 4032. -/
theorem largest_n_for_positive_sum 
  (a : ℕ → ℝ) 
  (h_arith : ∃ d, ∀ n, a (n+1) = a n + d)
  (h_a1 : a 1 > 0) 
  (h_cond1 : a 2016 + a 2017 > 0)
  (h_cond2 : a 2016 * a 2017 < 0) : 
  ∃ n, n = 4032 ∧ (∑ i in Finset.range (n + 1), a i) > 0 := 
sorry

end largest_n_for_positive_sum_l65_65105


namespace find_c_of_parabola_l65_65381

theorem find_c_of_parabola (a b c : ℝ) (h_vertex : ∀ x, y = a * (x - 3)^2 - 5)
                           (h_point : ∀ x y, (x = 1) → (y = -3) → y = a * (x - 3)^2 - 5)
                           (h_standard_form : ∀ x, y = a * x^2 + b * x + c) :
  c = -0.5 :=
sorry

end find_c_of_parabola_l65_65381


namespace parabola_intersects_line_at_given_point_l65_65122

theorem parabola_intersects_line_at_given_point (p x0 : ℝ) (h₀ : p > 0) (h₁ : (sqrt 3)^2 = 2 * p * x0)
  (h₂ : sqrt 3 = sqrt 3 * (x0 - p / 2)) : p = 1 := 
sorry

end parabola_intersects_line_at_given_point_l65_65122


namespace a0_equals_n_a3_equals_35_l65_65084

variable (x : ℝ)

-- Define the polynomial expansion
def poly_sum (n : ℕ) := (Finset.range (n+1)).sum (λ k, (1 + x)^k)

-- Define the list of coefficients being equal to the polynomial expansion
def coefficients (n : ℕ) : List ℕ := 
  List.range (n + 1)

-- Define the conditions for the problem
def given_conditions (n : ℕ) : Prop := 
  poly_sum x n = (coefficients n).sum

-- The hypothesis given that a₃ = 35
axiom h₃ : ∀ (a₃ : ℕ), a₃ = 35 → poly_sum x (n : ℕ) = a₃ 

-- The first theorem to prove a₀ = n
theorem a0_equals_n (n : ℕ) :
  given_conditions n →
  (coefficients n).head = n := 
by
  -- Given the conditions and required proof
  sorry

-- The second theorem to prove the value of n when a₃ = 35
theorem a3_equals_35 (n : ℕ) (a₃ : ℕ) :
  given_conditions n →
  h₃ a₃ →
  n = 6 := 
by
  -- Given the conditions and required proof
  sorry

end a0_equals_n_a3_equals_35_l65_65084


namespace bob_pennies_l65_65149

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l65_65149


namespace total_time_for_5_smoothies_l65_65584

-- Definitions for the conditions
def freeze_time : ℕ := 40
def blend_time_per_smoothie : ℕ := 3
def chop_time_apples_per_smoothie : ℕ := 2
def chop_time_bananas_per_smoothie : ℕ := 3
def chop_time_strawberries_per_smoothie : ℕ := 4
def chop_time_mangoes_per_smoothie : ℕ := 5
def chop_time_pineapples_per_smoothie : ℕ := 6
def number_of_smoothies : ℕ := 5

-- Total chopping time per smoothie
def chop_time_per_smoothie : ℕ := chop_time_apples_per_smoothie + 
                                  chop_time_bananas_per_smoothie + 
                                  chop_time_strawberries_per_smoothie + 
                                  chop_time_mangoes_per_smoothie + 
                                  chop_time_pineapples_per_smoothie

-- Total chopping time for 5 smoothies
def total_chop_time : ℕ := chop_time_per_smoothie * number_of_smoothies

-- Total blending time for 5 smoothies
def total_blend_time : ℕ := blend_time_per_smoothie * number_of_smoothies

-- Total time to make 5 smoothies
def total_time : ℕ := total_chop_time + total_blend_time

-- Theorem statement
theorem total_time_for_5_smoothies : total_time = 115 := by
  sorry

end total_time_for_5_smoothies_l65_65584


namespace total_share_proof_l65_65265

variable (P R : ℕ) -- Parker's share and Richie's share
variable (total_share : ℕ) -- Total share

-- Define conditions
def ratio_condition : Prop := (P : ℕ) / 2 = (R : ℕ) / 3
def parker_share_condition : Prop := P = 50

-- Prove the total share is 125
theorem total_share_proof 
  (h1 : ratio_condition P R)
  (h2 : parker_share_condition P) : 
  total_share = 125 :=
sorry

end total_share_proof_l65_65265


namespace kevin_bucket_size_l65_65589

def rate_of_leakage (r : ℝ) : Prop := r = 1.5
def time_away (t : ℝ) : Prop := t = 12
def bucket_size (b : ℝ) (r t : ℝ) : Prop := b = 2 * r * t

theorem kevin_bucket_size
  (r t b : ℝ)
  (H1 : rate_of_leakage r)
  (H2 : time_away t) :
  bucket_size b r t :=
by
  simp [rate_of_leakage, time_away, bucket_size] at *
  sorry

end kevin_bucket_size_l65_65589


namespace chess_piece_max_visitable_squares_l65_65727

-- Define initial board properties and movement constraints
structure ChessBoard :=
  (rows : ℕ)
  (columns : ℕ)
  (movement : ℕ)
  (board_size : rows * columns = 225)

-- Define condition for unique visitation
def can_visit (movement : ℕ) (board_size : ℕ) : Prop :=
  ∃ (max_squares : ℕ), (max_squares ≤ board_size) ∧ (max_squares = 196)

-- Main theorem statement 
theorem chess_piece_max_visitable_squares (cb : ChessBoard) : 
  can_visit 196 225 :=
by sorry

end chess_piece_max_visitable_squares_l65_65727


namespace chain_of_tangent_circles_iff_l65_65631

-- Define the circles, their centers, and the conditions
structure Circle := 
  (center : ℝ × ℝ) 
  (radius : ℝ)

structure TangentData :=
  (circle1 : Circle)
  (circle2 : Circle)
  (angle : ℝ)

-- Non-overlapping condition
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let dist := (x2 - x1)^2 + (y2 - y1)^2
  dist > (c1.radius + c2.radius)^2

-- Existence of tangent circles condition
def exists_chain_of_tangent_circles (c1 c2 : Circle) (n : ℕ) : Prop :=
  ∃ (tangent_circle : Circle), tangent_circle.radius = c1.radius ∨ tangent_circle.radius = c2.radius

-- Angle condition
def angle_condition (ang : ℝ) (n : ℕ) : Prop :=
  ∃ (k : ℤ), ang = k * (360 / n)

-- Final theorem to prove
theorem chain_of_tangent_circles_iff (c1 c2 : Circle) (t : TangentData) (n : ℕ) 
  (h1 : non_overlapping c1 c2) 
  (h2 : t.circle1 = c1 ∧ t.circle2 = c2) 
  : exists_chain_of_tangent_circles c1 c2 n ↔ angle_condition t.angle n := 
  sorry

end chain_of_tangent_circles_iff_l65_65631


namespace lcm_is_2310_l65_65656

def a : ℕ := 210
def b : ℕ := 605
def hcf : ℕ := 55

theorem lcm_is_2310 (lcm : ℕ) : Nat.lcm a b = 2310 :=
by 
  have h : a * b = lcm * hcf := by sorry
  sorry

end lcm_is_2310_l65_65656


namespace solution_set_of_inequality_l65_65663

theorem solution_set_of_inequality (a : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) :
  (a > 1 → {x : ℝ | x > 3 ∨ x < -1} = {x : ℝ | a ^ (x^2 - 3) > a ^ (2x)}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | -1 < x ∧ x < 3} = {x : ℝ | a ^ (x^2 - 3) > a ^ (2x)}) :=
sorry

end solution_set_of_inequality_l65_65663


namespace peter_money_l65_65629

theorem peter_money (cost_per_ounce : ℝ) (amount_bought : ℝ) (leftover_money : ℝ) (total_money : ℝ) :
  cost_per_ounce = 0.25 ∧ amount_bought = 6 ∧ leftover_money = 0.50 → total_money = 2 :=
by
  intros h
  let h1 := h.1
  let h2 := h.2.1
  let h3 := h.2.2
  sorry

end peter_money_l65_65629


namespace variance_of_xi_l65_65509

noncomputable def Eξ : ℝ := (Eξ == -1 * (1/3) + 0 * a + 1 * b = 1/4)
noncomputable def sum_ab : ℝ := (1/3 + a + b == 1)
noncomputable def Dξ : ℝ := (
  λ ξ : ℝ, 
  let a := (a ≠ 0), 
  let b := (b ≠ 0), 
  (Eξ == Dξ)
)

theorem variance_of_xi :
  (Eξ == 1/4) ∧ (sum_ab == 1) → Dξ = 41 / 48 := 
begin
  sorry
end

end variance_of_xi_l65_65509


namespace reciprocal_of_sum_correct_l65_65440

variable (x y : ℝ)

-- Define the reciprocal_of_sum function
def reciprocal_of_sum (x y : ℝ) : ℝ := 1 / (x + y)

-- Theorem stating the algebraic expression for the reciprocal of the sum of x and y
theorem reciprocal_of_sum_correct : reciprocal_of_sum x y = 1 / (x + y) := by 
  sorry

end reciprocal_of_sum_correct_l65_65440


namespace percentage_of_men_correct_l65_65911

def number_of_women : ℕ := 180
def number_of_men : ℕ := 420
def total_students : ℕ := number_of_women + number_of_men := by
  sorry

def percentage_of_men (number_of_men : ℕ) (total_students : ℕ) : ℚ :=
(number_of_men : ℚ) / (total_students : ℚ) * 100

theorem percentage_of_men_correct :
  percentage_of_men number_of_men total_students = 70 := by
  sorry

end percentage_of_men_correct_l65_65911


namespace dimensions_increased_three_times_l65_65988

variables (L B H k : ℝ) (n : ℝ)
 
-- Given conditions
axiom cost_initial : 350 = k * 2 * (L + B) * H
axiom cost_increased : 3150 = k * 2 * n^2 * (L + B) * H

-- Proof statement
theorem dimensions_increased_three_times : n = 3 :=
by
  sorry

end dimensions_increased_three_times_l65_65988


namespace problem_statement_l65_65219

noncomputable def λ_floor_is_perfect_square (λ : ℝ) (n : ℕ) : Prop :=
  (∀ m, n + 1 ≤ m → m ≤ 4 * n → ∃ k : ℕ, k * k = ⌊ λ ^ m ⌋) →
  ∃ k : ℕ, k * k = ⌊ λ ⌋

theorem problem_statement {λ : ℝ} (hλ : λ ≥ 1) {n : ℕ} (hn : n > 0) :
  λ_floor_is_perfect_square λ n :=
sorry

end problem_statement_l65_65219


namespace monotonic_increasing_interval_l65_65315

def f (x : ℝ) : ℝ := abs (x + 2)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, x ∈ Icc (-2 : ℝ) (⊤ : ℝ) → (f x = abs (x + 2) ∧ ∀ x₁ x₂ : ℝ, -2 ≤ x₁ → x₁ ≤ x₂ → x₂ → f x₁ ≤ f x₂) :=
by
  sorry

end monotonic_increasing_interval_l65_65315


namespace bus_driver_worked_hours_l65_65720

theorem bus_driver_worked_hours :
  ∃ (R OT: ℕ), OT = 256 / 31.50 ∧ R = 40 ∧ (R * 18) + (OT * 31.50) = 976 ∧ R + OT = 48 :=
begin
  sorry
end

end bus_driver_worked_hours_l65_65720


namespace angle_greater_than_135_l65_65497

theorem angle_greater_than_135
    (ABC : Triangle)
    (H1 : ∀ side : ABC.Sides, ¬∃ alt bis med : Segment, FormTriangleFromSegments alt bis med side) :
    ∃ angle : ABC.Angles, angle > 135 := 
sorry

end angle_greater_than_135_l65_65497


namespace integer_values_m_l65_65127

theorem integer_values_m (m x y : ℤ) (h1 : x - 2 * y = m) (h2 : 2 * x + 3 * y = 2 * m - 3)
    (h3 : 3 * x + y ≥ 0) (h4 : x + 5 * y < 0) : m = 1 ∨ m = 2 :=
by
  sorry

end integer_values_m_l65_65127


namespace paint_coverage_l65_65916

-- Define the conditions
def cost_per_gallon : ℝ := 45
def total_area : ℝ := 1600
def number_of_coats : ℝ := 2
def total_contribution : ℝ := 180 + 180

-- Define the target statement to prove
theorem paint_coverage (H : total_contribution = 360) : 
  let cost_per_gallon := 45 
  let number_of_gallons := total_contribution / cost_per_gallon
  let total_coverage := total_area * number_of_coats
  let coverage_per_gallon := total_coverage / number_of_gallons
  coverage_per_gallon = 400 :=
by
  sorry

end paint_coverage_l65_65916


namespace gcd_of_powers_l65_65679

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2007 - 1) : 
  Nat.gcd m n = 131071 :=
by
  sorry

end gcd_of_powers_l65_65679


namespace hyperbola_standard_eq_l65_65829

theorem hyperbola_standard_eq (λ : ℝ) :
  (∃ λ : ℝ, ∀ x y : ℝ, y^2 - (1/4) * x^2 = λ 
  → (4, sqrt 3) = (x, y) 
  → y = (1/2) * x 
  → y = -(1/2) * x) 
  ∧ λ = -1 :=
  sorry

end hyperbola_standard_eq_l65_65829


namespace average_of_remaining_two_l65_65984

theorem average_of_remaining_two (a b c d e f : ℝ) 
  (h_avg6 : (a + b + c + d + e + f) / 6 = 3.95)
  (h_avg2_1 : (a + b) / 2 = 3.8) 
  (h_avg2_2 : (c + d) / 2 = 3.85) : 
  (e + f) / 2 = 4.2 :=
begin
  sorry
end

end average_of_remaining_two_l65_65984


namespace minimum_n_for_moves_l65_65220

theorem minimum_n_for_moves (n : ℕ) :
  (∀ l : ℕ, ∃ m : ℕ, m ≥ l ∧ ∃ cards : fin 100 → ℕ, (∀ i : fin 100, cards i < m ∧ cards i % n = 0) ∧ (∀ k : ℕ, 0 < k → (finset.univ.sum (λ i : fin 100, cards i + k - 1) = k))) → n = 10000 :=
sorry

end minimum_n_for_moves_l65_65220


namespace maximum_radius_of_sphere_l65_65989

noncomputable def maxRadius (r : ℝ) : Prop := r ≤ 3 / 2 ^ (1 / 3 : ℝ)

theorem maximum_radius_of_sphere:
  ∀ r : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ r → r - real.sqrt (r ^ 2 - x ^ 2) ≥ x ^ 4) → maxRadius r :=
by 
  intros r h,
  sorry

end maximum_radius_of_sphere_l65_65989


namespace cauchy_convergence_l65_65273

open Filter

theorem cauchy_convergence {x : ℕ → ℝ} {a : ℝ} 
  (h : Tendsto x atTop (𝓝 a)) : 
  Tendsto (fun n => (∑ i in Finset.range n, x i) / n) atTop (𝓝 a) := 
sorry

end cauchy_convergence_l65_65273


namespace Penelope_Candies_l65_65961

variable (M : ℕ) (S : ℕ)
variable (h1 : 5 * S = 3 * M)
variable (h2 : M = 25)

theorem Penelope_Candies : S = 15 := by
  sorry

end Penelope_Candies_l65_65961


namespace relationship_inequality_l65_65375

variable {a b c d : ℝ}

-- Define the conditions
def is_largest (a b c : ℝ) : Prop := a > b ∧ a > c
def positive_numbers (a b c d : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
def ratio_condition (a b c d : ℝ) : Prop := a / b = c / d

-- The theorem statement
theorem relationship_inequality 
  (h_largest : is_largest a b c)
  (h_positive : positive_numbers a b c d)
  (h_ratio : ratio_condition a b c d) :
  a + d > b + c :=
sorry

end relationship_inequality_l65_65375


namespace students_remaining_after_four_stops_l65_65878

theorem students_remaining_after_four_stops
  (S : ℕ)
  (S0 : S = 72)
  (recurrence : ∀ n, S (n + 1) = nat.floor ((2:ℚ) / 3 * S n))
  : S 4 = 14 :=
by
  sorry

end students_remaining_after_four_stops_l65_65878


namespace max_T_n_at_2_l65_65291

noncomputable def geom_seq (a n : ℕ) : ℕ :=
  a * 2 ^ n

noncomputable def S_n (a n : ℕ) : ℕ :=
  a * (2 ^ n - 1)

noncomputable def T_n (a n : ℕ) : ℕ :=
  (17 * S_n a n - S_n a (2 * n)) / geom_seq a n

theorem max_T_n_at_2 (a : ℕ) : (∀ n > 0, T_n a n ≤ T_n a 2) :=
by
  -- proof omitted
  sorry

end max_T_n_at_2_l65_65291


namespace triangle_area_l65_65390

theorem triangle_area (P : ℝ × ℝ) (Q R : ℝ × ℝ) 
  (hP : P = (1, 6))
  (hQ : Q = (-5, 0))
  (hR : R = (-2, 0))
  (line1 : ∀ x, (x, x + 5) ∈ set.range (λ x, (x, x + 5)))
  (line2 : ∀ x, (x, 2 * x + 4) ∈ set.range (λ x, (x, 2 * x + 4)))
  : (1/2:ℝ) * (float.ofReal (abs (Q.fst - R.fst))) * (abs (P.snd)) = 9 :=
by
  -- Skip the proof
  sorry

end triangle_area_l65_65390


namespace find_a_l65_65931

variables {a b c : ℂ}

-- Given conditions
variables (h1 : a + b + c = 5) 
variables (h2 : a * b + b * c + c * a = 5) 
variables (h3 : a * b * c = 5)
variables (h4 : a.im = 0) -- a is real

theorem find_a : a = 4 :=
by
  sorry

end find_a_l65_65931


namespace exists_minimal_sequence_l65_65937

noncomputable def sequence_in_Fp {p : ℕ} (hp : p > 1) (a : ℕ → ℤ) : Prop :=
  ∃ x y : ℤ, (a 0 = x ∧ a 1 = y ∧ ∀ n ≥ 1, a (n + 1) = (p + 1) * a n - p * a (n - 1)) ∧ (a ≠ (λ n, x))

noncomputable def special_sequence {p : ℕ} (hp : p > 1) : (ℕ → ℤ) :=
  λ n, (p^n - 1) / (p - 1)

theorem exists_minimal_sequence {p : ℕ} (hp : p > 1) :
  let a := special_sequence hp in
  sequence_in_Fp hp a ∧ ∀ b : ℕ → ℤ, sequence_in_Fp hp b → ∀ n : ℕ, a n ≤ b n :=
by
  intro a
  use sorry
  use sorry
  intro b hb
  intro n
  apply sorry

end exists_minimal_sequence_l65_65937


namespace clock_angle_8_30_l65_65665

theorem clock_angle_8_30 :
  let total_degrees := 360
  let degrees_per_hour := total_degrees / 12
  let minute_hand_angle := 6 * degrees_per_hour
  let hour_hand_angle := 8 * degrees_per_hour + degrees_per_hour / 2
  abs (hour_hand_angle - minute_hand_angle) = 75 :=
by
  let total_degrees := 360
  let degrees_per_hour := total_degrees / 12
  let minute_hand_angle := 6 * degrees_per_hour
  let hour_hand_angle := 8 * degrees_per_hour + degrees_per_hour / 2
  show abs (hour_hand_angle - minute_hand_angle) = 75
  sorry

end clock_angle_8_30_l65_65665


namespace slope_angle_at_one_l65_65322

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1)

theorem slope_angle_at_one : 
  let df := deriv f 1 in
  tan ((df / (1 + df^2)).atan) = Real.pi / 4 :=
by
  sorry

end slope_angle_at_one_l65_65322


namespace quadratic_sum_of_roots_l65_65121

theorem quadratic_sum_of_roots (a b : ℝ)
  (h1: ∀ x: ℝ, x^2 + b * x - a < 0 ↔ 3 < x ∧ x < 4):
  a + b = -19 :=
sorry

end quadratic_sum_of_roots_l65_65121


namespace kelly_sony_games_solution_l65_65921

def kelly_sony_games_left (n g : Nat) : Nat :=
  n - g

theorem kelly_sony_games_solution (initial : Nat) (given_away : Nat) 
  (h_initial : initial = 132)
  (h_given_away : given_away = 101) :
  kelly_sony_games_left initial given_away = 31 :=
by
  rw [h_initial, h_given_away]
  unfold kelly_sony_games_left
  norm_num

end kelly_sony_games_solution_l65_65921


namespace find_smallest_a_l65_65292
open Real

noncomputable def a_min := 2 / 9

theorem find_smallest_a (a b c : ℝ)
  (h1 : (1/4, -9/8) = (1/4, a * (1/4) * (1/4) - 9/8))
  (h2 : ∃ n : ℤ, a + b + c = n)
  (h3 : a > 0)
  (h4 : b = - a / 2)
  (h5 : c = a / 16 - 9 / 8): 
  a = a_min :=
by {
  -- Lean code equivalent to the provided mathematical proof will be placed here.
  sorry
}

end find_smallest_a_l65_65292


namespace Sn_two_n_minus_one_constant_l65_65104

-- An arithmetic sequence is defined
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n m : ℕ, a(n + 1) - a(n) = d

-- The sequence aₙ is a constant
def is_constant (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a n = a m

-- Define Sₙ as the sum of the first n terms of sequence a
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

-- Prove that S_{2n-1} is a constant when aₙ is a constant
theorem Sn_two_n_minus_one_constant
  (a : ℕ → ℝ) (h_arith : is_arithmetic_sequence a) (h_const : is_constant a) :
  ∃ k : ℝ, S a (2*n - 1) = k :=
sorry

end Sn_two_n_minus_one_constant_l65_65104


namespace value_of_expression_l65_65847

noncomputable def f : ℝ → ℝ
| x => if x > 0 then -1 else if x < 0 then 1 else 0

theorem value_of_expression (a b : ℝ) (h : a ≠ b) :
  (a + b + (a - b) * f (a - b)) / 2 = min a b := 
sorry

end value_of_expression_l65_65847


namespace patricia_candies_final_l65_65628

def initial_candies : ℕ := 764
def taken_candies : ℕ := 53
def back_candies_per_7_taken : ℕ := 19

theorem patricia_candies_final :
  let given_back_times := taken_candies / 7
  let total_given_back := given_back_times * back_candies_per_7_taken
  let final_candies := initial_candies - taken_candies + total_given_back
  final_candies = 844 :=
by
  sorry

end patricia_candies_final_l65_65628


namespace lens_circumference_l65_65614

def circumference (d: ℝ) : ℝ := Real.pi * d

theorem lens_circumference :
  circumference 10 = 31.42 := 
  sorry

end lens_circumference_l65_65614


namespace cubic_roots_squared_roots_l65_65945

open Polynomial

theorem cubic_roots_squared_roots (a b c : ℂ) :
  let p (x : ℂ) := x^3 + a * x^2 + b * x + c in
  ∃ α β γ : ℂ, p α = 0 ∧ p β = 0 ∧ p γ = 0 ∧ p (α^2) = 0 ∧ p (β^2) = 0 ∧ p (γ^2) = 0 ↔
  (a, b, c) ∈ {((0,0,0)), ((-1,0,0)), ((-2,1,0)), ((0,-1,0)), ((1,1,0)), ((-3,3,-1)), ((-1,-1,1)), ((1,-1,-1)), ((0,0,-1)), ((1-i*√7)/2, (-1-i*√7)/2,-1), ((1+i*√7)/2, (-1+i*√7)/2,-1)} := sorry

end cubic_roots_squared_roots_l65_65945


namespace intersection_A_B_l65_65126

noncomputable def A := { p : ℝ × ℝ | p.2 = log (p.1 - 2) }
noncomputable def B := { p : ℝ × ℝ | p.2 = 2 ^ p.1 }

theorem intersection_A_B :
  ∃ (p : ℝ × ℝ), p ∈ A ∧ p ∈ B :=
sorry

end intersection_A_B_l65_65126


namespace bus_travel_distance_l65_65719

theorem bus_travel_distance :
  let s := (100 * (5 + 2 / 3)) / (1 + (5 + 2 / 3)) in
  let t := 5 + 2 / 3 in
  (t = 17 / 3) → (s = 85) → 
  (∃ V : ℝ, V = s / t ∧ V = 100 - s ∧ s = 85) :=
  by
    sorry

end bus_travel_distance_l65_65719


namespace cos_double_angle_l65_65505

theorem cos_double_angle (θ : ℝ) (h₁ : tan θ = 2) : cos (2 * θ) = -3 / 5 :=
by sorry

end cos_double_angle_l65_65505


namespace isosceles_triangle_perpendicular_l65_65552

universe u

variables {α : Type u} 
variables {A B C D : α} [euclidean_domain α] [decidable_eq α]
variables (AB AC BC BD CD AD : α)
variables (h_eq1 : AB = AC) (h_eq2 : BD = CD)

-- Proof: AD is perpendicular to BC given AB = AC and BD = CD
theorem isosceles_triangle_perpendicular (h_eq1 : AB = AC) (h_eq2 : BD = CD) : AD ⊥ BC :=
sorry

end isosceles_triangle_perpendicular_l65_65552


namespace find_a_5_l65_65502

variable (a : ℕ → ℕ)

def is_geom_seq {r : ℕ} := ∀ n, 1 + a (n + 1) = (1 + a n) * r

theorem find_a_5 (h: is_geom_seq a 2) (ha1: a 1 = 1) : a 5 = 31 :=
by
  sorry

end find_a_5_l65_65502


namespace work_together_zero_days_l65_65362

theorem work_together_zero_days (a b : ℝ) (ha : a = 1/18) (hb : b = 1/9) (x : ℝ) (hx : 1 - x * a = 2/3) : x = 6 →
  (a - a) * (b - b) = 0 := by
  sorry

end work_together_zero_days_l65_65362


namespace part_1_monotonicity_part_2_range_of_a_l65_65852

-- Definition of the function f
def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.exp (2 * x) - a * (2 * x + 1) * Real.log (2 * x + 1)

-- Part (1): Proving monotonicity of f when a = 2
theorem part_1_monotonicity (x : ℝ) (h : x > -1/2) : let a := 2 in 
  Monotone (λ x, f x a) :=
sorry

-- Part (2): Proving the range of values for a such that the inequality holds
theorem part_2_range_of_a (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ Real.pi / 2) :
  ∀ a, f x a ≥ 2 * Real.cos x ^ 2 - 2 * (a - 2) * x ↔ a ≤ 3 :=
sorry

end part_1_monotonicity_part_2_range_of_a_l65_65852


namespace power_of_hundred_expansion_l65_65350

theorem power_of_hundred_expansion :
  ∀ (n : ℕ), (100 = 10^2) → (100^50 = 10^100) :=
by
  intros n h
  rw [h]
  exact pow_mul 10 2 50
# You can replace pow_mul with the relevant lemmata if it conflicts.

end power_of_hundred_expansion_l65_65350


namespace steve_driving_speed_back_l65_65708

noncomputable def steve_speed_way_to_work (v : ℝ) : Prop :=
let t_drive := 20 / v in
let t_cycle := (20 / v) + 0.5 in
let t_back_drive := (20 / (2 * v)) + 0.333 in
(t_drive + 10 / t_cycle + t_back_drive) = 6

noncomputable def steve_speed_back (v : ℝ) : ℝ := 2 * v

theorem steve_driving_speed_back (v : ℝ) (h : steve_speed_way_to_work v) : steve_speed_back v = 11.61 := sorry

end steve_driving_speed_back_l65_65708


namespace initial_spiders_l65_65544

def spiders_make_webs (S : ℕ) : Prop :=
  S spiders make S webs in S days

def one_spider_one_web_seven_days : Prop :=
  1 spider makes 1 web in 7 days

theorem initial_spiders (S : ℕ) (h1 : spiders_make_webs S) (h2 : one_spider_one_web_seven_days) : S = 7 :=
sorry

end initial_spiders_l65_65544


namespace cone_lateral_surface_area_l65_65830

-- Define the radius and slant height as given constants
def radius : ℝ := 3
def slant_height : ℝ := 5

-- Definition of the lateral surface area
def lateral_surface_area (r l : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * r in
  (circumference * l) / 2

-- The proof problem statement in Lean 4
theorem cone_lateral_surface_area : lateral_surface_area radius slant_height = 15 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l65_65830


namespace ceil_sqrt_200_eq_15_l65_65436

theorem ceil_sqrt_200_eq_15 : ⌈Real.sqrt 200⌉ = 15 := 
sorry

end ceil_sqrt_200_eq_15_l65_65436


namespace clever_calculation_part1_clever_calculation_part2_clever_calculation_part3_l65_65417

-- Prove that 46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 equals 56.056
theorem clever_calculation_part1 : 46.3 * 0.56 + 5.37 * 5.6 + 1 * 0.056 = 56.056 :=
by
sorry

-- Prove that 101 * 92 - 92 equals 9200
theorem clever_calculation_part2 : 101 * 92 - 92 = 9200 :=
by
sorry

-- Prove that 36000 / 125 / 8 equals 36
theorem clever_calculation_part3 : 36000 / 125 / 8 = 36 :=
by
sorry

end clever_calculation_part1_clever_calculation_part2_clever_calculation_part3_l65_65417


namespace simplify_part1_simplify_part2_evaluation_part2_l65_65976

variable (a x y : ℝ)

theorem simplify_part1 : 3 * (a^2 - 2 * a) + (-2 * a^2 + 5 * a) = a^2 - a := 
by sorry

theorem simplify_part2 : -3 * x * y^2 - 2 * (x * y - (3 / 2) * x^2 * y) + (2 * x * y^2 - 3 * x^2 * y) = 
   - x * y^2 - 2 * x * y :=
by sorry

theorem evaluation_part2 : 
    let x := -4; let y := 1 / 2 
  in -x * y^2 - 2 * x * y = 5 := 
by sorry

end simplify_part1_simplify_part2_evaluation_part2_l65_65976


namespace tangent_line_tangent_to_curve_l65_65161

theorem tangent_line_tangent_to_curve (k : ℝ) :
  (∃ x y : ℝ, y = exp x ∧ k * x - y - k = 0 ∧ ∃ m : ℝ, y - exp m = exp m * (x - m) ∧ exp m = k) → k = exp 2 :=
begin
  sorry
end

end tangent_line_tangent_to_curve_l65_65161


namespace similar_parabolas_l65_65277

theorem similar_parabolas :
  (∀ (a b : ℝ), b = 2 * a^2 → ∃ (x y : ℝ), y = x^2 ∧ x = 2 * a ∧ y = 2 * b) :=
by
  assume a b : ℝ,
  assume h : b = 2 * a^2,
  existsi (2 * a),
  existsi (2 * b),
  split,
  {
    calc
      2 * b = 2 * (2 * a^2) : by rw [h]
      ...   = (2 * a)^2 : by ring,
  },
  split,
  { refl },
  { exact h }

end similar_parabolas_l65_65277


namespace unique_function_sum_l65_65641

theorem unique_function_sum :
  ∃! (f : ℕ+ → ℕ+), (∀ m n : ℕ+, f (m + f n) = n + f (m + 95)) ∧ (finset.sum (finset.range 19) (fun n => f n) = 1995) :=
sorry

end unique_function_sum_l65_65641


namespace incorrect_average_l65_65297

theorem incorrect_average (S : ℕ) (A_correct : ℕ) (A_incorrect : ℕ) (S_correct : ℕ) 
  (h1 : S = 135)
  (h2 : A_correct = 19)
  (h3 : A_incorrect = (S + 25) / 10)
  (h4 : S_correct = (S + 55) / 10)
  (h5 : S_correct = A_correct) :
  A_incorrect = 16 :=
by
  -- The proof will go here, which is skipped with a 'sorry'
  sorry

end incorrect_average_l65_65297


namespace simplify_expression_l65_65468

theorem simplify_expression : 
  ((3 + 4 + 5 + 6 + 7) / 3 + (3 * 6 + 9)^2 / 9) = 268 / 3 := 
by 
  sorry

end simplify_expression_l65_65468


namespace shaded_area_in_hexagon_l65_65568

theorem shaded_area_in_hexagon (hex_area : ℝ) (triangle_count : ℕ) (shaded_count : ℕ) (A : ℝ) :
  hex_area = 216 → 
  triangle_count = 18 → 
  shaded_count = 6 → 
  A = (shaded_count / triangle_count) * hex_area → 
  A = 72 := 
by 
  intros h1 h2 h3 h4 
  rw [h1, h2, h3, h4]
  sorry

end shaded_area_in_hexagon_l65_65568


namespace quadratic_has_integer_solutions_l65_65465

theorem quadratic_has_integer_solutions : 
  ∃ (s : Finset ℕ), ∀ a : ℕ, a ∈ s ↔ (1 ≤ a ∧ a ≤ 50 ∧ ((∃ n : ℕ, 4 * a + 1 = n^2))) ∧ s.card = 6 := 
  sorry

end quadratic_has_integer_solutions_l65_65465


namespace range_of_m_l65_65882

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ 4) → 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := 
by 
  sorry

end range_of_m_l65_65882


namespace sufficiency_s_for_q_l65_65107

variables {q r s : Prop}

theorem sufficiency_s_for_q (h₁ : r → q) (h₂ : ¬(q → r)) (h₃ : r ↔ s) : s → q ∧ ¬(q → s) :=
by
  sorry

end sufficiency_s_for_q_l65_65107


namespace perimeter_triangle_APR_l65_65336

-- Definitions based on conditions
variables {P Q R A B C : Type}
variable (AB AC PQ QR AP PR AR : ℝ)
variable (h1 : AB = 24)
variable (h2 : AC = 24)
variable (h3 : AP = 12)
variable (h4 : PR = 24)
variable (h5 : AR = 12)

-- The statement of the theorem
theorem perimeter_triangle_APR : AP + PR + AR = 48 :=
by {
  have h6 : AP = 12, from h3,
  have h7 : PR = 24, from h4,
  have h8 : AR = 12, from h5,
  calc
    AP + PR + AR = 12 + 24 + 12 : by rw [h6, h7, h8]
              ... = 48 : by linarith
}

end perimeter_triangle_APR_l65_65336


namespace service_cost_per_vehicle_l65_65169

-- Defining the problem setup
def mini_van_tank_capacity : ℕ := 65
def truck_tank_capacity : ℕ := mini_van_tank_capacity + Nat.ofNat (0.2 * 65)
def fuel_cost_per_liter : ℝ := 0.70
def num_mini_vans : ℕ := 4
def num_trucks : ℕ := 2
def total_cost : ℝ := 396.00

-- Statement to prove the service cost per vehicle
theorem service_cost_per_vehicle :
  let total_fuel_cost := (num_mini_vans * mini_van_tank_capacity + 
                          num_trucks * truck_tank_capacity) * 
                          fuel_cost_per_liter,
      total_service_cost := total_cost - total_fuel_cost,
      total_vehicles := num_mini_vans + num_trucks
  in total_service_cost / total_vehicles = 2.30 := by
  sorry

end service_cost_per_vehicle_l65_65169


namespace molecular_weight_dinitrogen_trioxide_l65_65342

theorem molecular_weight_dinitrogen_trioxide:
  let molecular_weight_N2O3 := (2 * 14.01) + (3 * 16.00)
  ∧ let weight_3_moles := 3 * molecular_weight_N2O3
  in weight_3_moles = 228.06 :=
by
  sorry

end molecular_weight_dinitrogen_trioxide_l65_65342


namespace find_common_ratio_l65_65170

noncomputable def geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (pos : ∀ n, 0 < a n)
  (h1 : a 1 * a 3 = 4)
  (h2 : a 2 + a 4 = 10) : ℝ :=
classical.some (exists_sqrt (eq.symm (div_eq_iff (ne_of_gt $ pos 2)).mpr
  (by { rw [← mul_assoc, h1], linarith })))

theorem find_common_ratio (a : ℕ → ℝ) 
  (pos : ∀ n, 0 < a n)
  (h1 : a 1 * a 3 = 4)
  (h2 : a 2 + a 4 = 10) : geometric_sequence_common_ratio a pos h1 h2 = 2 :=
sorry

end find_common_ratio_l65_65170


namespace calculation_l65_65762

theorem calculation : (3 * 4 * 5) * ((1 / 3 : ℚ) + (1 / 4 : ℚ) - (1 / 5 : ℚ)) = 23 := by
  sorry

end calculation_l65_65762


namespace maximal_good_set_cardinality_l65_65740

def is_good_set (S : Finset ℕ) : Prop :=
  ∀ x ∈ S, ¬ x ∣ S.sum - x

def A := (Finset.range 64).erase 0  -- {1, 2, ..., 63} but index shifted by 1

theorem maximal_good_set_cardinality :
  ∃ S ⊆ A, is_good_set S ∧ S.card = 59 := 
sorry

end maximal_good_set_cardinality_l65_65740


namespace find_g0_l65_65601

-- Define the polynomials f, g, h and their properties.
variables {R : Type*} [CommRing R]
variables (f g h : R[X])

-- Conditions
def condition1 := h = f * g
def condition2 := (f.coeff 0 = -6)
def condition3 := (h.coeff 0 = 12)

-- Theorem statement
theorem find_g0 (H1 : condition1 f g h) (H2 : condition2 f) (H3 : condition3 h) : 
  g.coeff 0 = -2 :=
sorry

end find_g0_l65_65601


namespace kevin_bucket_size_l65_65588

def rate_of_leakage (r : ℝ) : Prop := r = 1.5
def time_away (t : ℝ) : Prop := t = 12
def bucket_size (b : ℝ) (r t : ℝ) : Prop := b = 2 * r * t

theorem kevin_bucket_size
  (r t b : ℝ)
  (H1 : rate_of_leakage r)
  (H2 : time_away t) :
  bucket_size b r t :=
by
  simp [rate_of_leakage, time_away, bucket_size] at *
  sorry

end kevin_bucket_size_l65_65588


namespace Lesha_received_11_gifts_l65_65372

theorem Lesha_received_11_gifts (x : ℕ) 
    (h1 : x < 100) 
    (h2 : x % 2 = 0) 
    (h3 : x % 5 = 0) 
    (h4 : x % 7 = 0) :
    x - (x / 2 + x / 5 + x / 7) = 11 :=
by {
    sorry
}

end Lesha_received_11_gifts_l65_65372


namespace solution_set_of_inequality_l65_65977

noncomputable def solve_inequality (x : ℝ) : Prop :=
  abs (sqrt (3 * x - 2) - 3) > 1

theorem solution_set_of_inequality :
  {x : ℝ | solve_inequality x} = {x | x > 6 ∨ (2 / 3 ≤ x ∧ x < 2)} :=
by sorry

end solution_set_of_inequality_l65_65977


namespace rebus_no_solution_l65_65968

open Nat
open DigitFin

theorem rebus_no_solution (K U S Y : Fin 10) (h1 : K ≠ U) (h2 : K ≠ S) (h3 : K ≠ Y) (h4 : U ≠ S) (h5 : U ≠ Y) (h6 : S ≠ Y) :
  let KUSY := K.val * 1000 + U.val * 100 + S.val * 10 + Y.val
  let UKSY := U.val * 1000 + K.val * 100 + S.val * 10 + Y.val
  let UKSUS := U.val * 100000 + K.val * 10000 + S.val * 1000 + U.val * 100 + S.val * 10 + S.val
  KUSY + UKSY ≠ UKSUS := by
sorry

end rebus_no_solution_l65_65968


namespace fuel_consumption_40_kmh_minimum_fuel_consumption_minimum_fuel_speed_80_kmh_l65_65741

-- Define the fuel consumption function y
def fuel_consumption_per_hour (x : ℝ) : ℝ :=
  (1 / 128000) * x^3 - (3 / 80) * x + 8

-- Problem 1: Fuel consumption at 40 km/h
theorem fuel_consumption_40_kmh :
  (100 / 40) * (fuel_consumption_per_hour 40) = 17.5 :=
by sorry

-- Problem 2: Minimum fuel consumption
def total_fuel_consumption (x : ℝ) : ℝ :=
  (fuel_consumption_per_hour x) * (100 / x)

theorem minimum_fuel_consumption :
  ∃ x, 0 < x ∧ x ≤ 120 ∧ total_fuel_consumption x = 11.25 :=
by sorry

noncomputable def minimum_fuel_speed :=
  classical.some minimum_fuel_consumption

theorem minimum_fuel_speed_80_kmh :
  minimum_fuel_speed = 80 :=
by sorry

end fuel_consumption_40_kmh_minimum_fuel_consumption_minimum_fuel_speed_80_kmh_l65_65741


namespace inequality_solution_intervals_l65_65442

theorem inequality_solution_intervals (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1) + (x + 3) / (2 * x) ≥ 4) ↔ (0 < x ∧ x < 1) := 
sorry

end inequality_solution_intervals_l65_65442


namespace smallest_x_ensures_prime_l65_65344

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ k : ℕ, k > 1 → k ∣ n → k = n

def quadratic (x : ℤ) := 8 * x^2 - 53 * x + 21

theorem smallest_x_ensures_prime :
  ∃ (x : ℤ), (∀ y : ℤ, |quadratic y|.natAbs.is_prime → y ≥ 8) ∧ |quadratic x|.natAbs.is_prime ∧ x = 8 :=
by sorry

end smallest_x_ensures_prime_l65_65344


namespace arithmetic_sequence_x_value_l65_65867

theorem arithmetic_sequence_x_value (x : ℝ) (a2 a1 d : ℝ)
  (h1 : a1 = 1 / 3)
  (h2 : a2 = x - 2)
  (h3 : d = 4 * x + 1 - a2)
  (h2_eq_d_a1 : a2 - a1 = d) : x = - (8 / 3) :=
by
  -- Proof yet to be completed
  sorry

end arithmetic_sequence_x_value_l65_65867


namespace astronomy_club_officers_count_l65_65982

def num_ways_choose_officers 
  (total_members : ℕ) 
  (officers : ℕ) 
  (special_pairs : List (ℕ × ℕ)) : ℕ :=
let remaining_without_pairs := total_members - 4 in
let nothing_special :=
  remaining_without_pairs * (remaining_without_pairs - 1) * (remaining_without_pairs - 2) 
let one_pair_special :=
  2 * 1 * (remaining_without_pairs - 2)
let both_pairs_special :=  2 * 2 * 2 * 1 
in nothing_special + 2 * one_pair_special + both_pairs_special

theorem astronomy_club_officers_count : 
  num_ways_choose_officers 25 3 [(1, 2), (3, 4)] = 8072 :=
  by
  -- We can leave the proof as sorry for now.
  sorry

end astronomy_club_officers_count_l65_65982


namespace paint_for_320_statues_l65_65541

def paint_needed_for_statue (height : ℝ) (paint : ℝ) (new_height : ℝ) : ℝ :=
  paint * (new_height / height)^3

def total_paint_needed (height : ℝ) (paint : ℝ) (new_height : ℝ) (num_statues : ℕ) : ℝ :=
  num_statues * paint_needed_for_statue height paint new_height

theorem paint_for_320_statues :
  total_paint_needed 8 2 2 320 = 10 :=
by
  sorry

end paint_for_320_statues_l65_65541


namespace evaluate_expression_l65_65438

theorem evaluate_expression : 
  (4 * 6 / (12 * 16)) * (8 * 12 * 16 / (4 * 6 * 8)) = 1 :=
by
  sorry

end evaluate_expression_l65_65438


namespace quadratic_inverse_sum_roots_l65_65098

theorem quadratic_inverse_sum_roots (x1 x2 : ℝ) (h1 : x1^2 - 2023 * x1 + 1 = 0) (h2 : x2^2 - 2023 * x2 + 1 = 0) : 
  (1/x1 + 1/x2) = 2023 :=
by
  -- We outline the proof steps that should be accomplished.
  -- These will be placeholders and not part of the actual statement.
  -- sorry allows us to skip the proof.
  sorry

end quadratic_inverse_sum_roots_l65_65098


namespace count_valid_subsets_is_11_l65_65806

open Set

def numbers : Set ℕ := {23, 65, 35, 96, 18, 82, 70}

def subsets_with_sum_multiple_of_11 (s : Set ℕ) : Set (Set ℕ) :=
  {s' ∈ (powerset s) | (s'.sum id) % 11 = 0}

theorem count_valid_subsets_is_11 :
  (subsets_with_sum_multiple_of_11 numbers).size = 11 :=
by sorry

end count_valid_subsets_is_11_l65_65806


namespace distance_focus_directrix_parabola_l65_65054

theorem distance_focus_directrix_parabola :
  let p := 1,
      focus := (0, p / 2),
      directrix := -(p / 2)
  in abs ((p / 2) - (- (p / 2))) = 2 :=
by
  let p := 1
  let focus := (0, p / 2)
  let directrix := -(p / 2)
  show abs ((p / 2) - (- (p / 2))) = 2
  sorry

end distance_focus_directrix_parabola_l65_65054


namespace shortest_trip_on_octahedron_l65_65402

-- Definitions for the problem
def regular_octahedron_edges_length := 2
def midpoint (x : ℝ) : ℝ := x / 2
def non_adjacent (e1 e2 : ℝ) : Prop := e1 ≠ e2

-- The theorem to be proven
theorem shortest_trip_on_octahedron :
  ∀ m1 m2 : ℝ, m1 = midpoint regular_octahedron_edges_length 
               → m2 = midpoint regular_octahedron_edges_length 
               → non_adjacent m1 m2 
               → m1 + m2 = 2 :=
by
  -- Proof not required
  sorry

end shortest_trip_on_octahedron_l65_65402


namespace sum_of_divisors_330_l65_65682

def is_sum_of_divisors (n : ℕ) (sum : ℕ) :=
  sum = ∑ d in divisors n, d

theorem sum_of_divisors_330 : is_sum_of_divisors 330 864 :=
by {
  -- sorry
}

end sum_of_divisors_330_l65_65682


namespace cone_lateral_surface_area_l65_65833

-- Definitions and conditions
def radius (r : ℝ) := r = 3
def slant_height (l : ℝ) := l = 5
def lateral_surface_area (A : ℝ) (C : ℝ) (l : ℝ) := A = 0.5 * C * l
def circumference (C : ℝ) (r : ℝ) := C = 2 * Real.pi * r

-- Proof (statement only)
theorem cone_lateral_surface_area :
  ∀ (r l C A : ℝ), 
    radius r → 
    slant_height l → 
    circumference C r → 
    lateral_surface_area A C l → 
    A = 15 * Real.pi := 
by intros; sorry

end cone_lateral_surface_area_l65_65833


namespace valid_integer_values_n_l65_65802

def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

theorem valid_integer_values_n : ∃ (n_values : ℕ), n_values = 3 ∧
  ∀ n : ℤ, is_integer (3200 * (2 / 5) ^ (2 * n)) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end valid_integer_values_n_l65_65802


namespace coefficient_x3_in_expansion_l65_65569

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3_in_expansion :
  let x := 1
  let a := x
  let b := 2
  let n := 50
  let k := 47
  let coefficient := binom n (n - k) * b^k
  coefficient = 19600 * 2^47 := by
  sorry

end coefficient_x3_in_expansion_l65_65569


namespace number_of_solutions_l65_65536

theorem number_of_solutions : ∃! (xy : ℕ × ℕ), (xy.1 ^ 2 - xy.2 ^ 2 = 91 ∧ xy.1 > 0 ∧ xy.2 > 0) := sorry

end number_of_solutions_l65_65536


namespace price_of_tray_l65_65317

noncomputable def price_per_egg : ℕ := 50
noncomputable def tray_eggs : ℕ := 30
noncomputable def discount_per_egg : ℕ := 10

theorem price_of_tray : (price_per_egg - discount_per_egg) * tray_eggs / 100 = 12 :=
by
  sorry

end price_of_tray_l65_65317


namespace complex_division_evaluation_l65_65784

open Complex

theorem complex_division_evaluation :
  (2 : ℂ) / (I * (3 - I)) = (1 / 5 : ℂ) - (3 / 5) * I :=
by
  sorry

end complex_division_evaluation_l65_65784


namespace max_value_f_when_a_eq_1_l65_65115

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ a then x^3 - 3*x else -2*x

theorem max_value_f_when_a_eq_1 : ∃ x : ℝ, f x 1 = 2 ∧ ∀ y : ℝ, f y 1 ≤ 2 := 
by 
  -- Proof goes here
  sorry

end max_value_f_when_a_eq_1_l65_65115


namespace evaluate_expression_l65_65435

theorem evaluate_expression : 81^(-1/4 : ℝ) + 16^(-3/4 : ℝ) = 11 / 24 := 
by
  sorry

end evaluate_expression_l65_65435


namespace max_possible_value_l65_65492

theorem max_possible_value (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∀ (z : ℝ), (z = (x + y + 1) / x) → z ≤ -0.2 :=
by sorry

end max_possible_value_l65_65492


namespace inscribable_in_cone_if_opposite_dihedral_angles_equal_l65_65965

-- Define the setup for the convex pyramidal corner
structure PyramidalCorner :=
  (apex : Point)
  (vertices : Point × Point × Point × Point)  -- points A, B, C, D
  (edges   : Line × Line × Line × Line)       -- lines SA, SB, SC, SD

-- Define the condition: sums of opposite dihedral angles are equal
def oppositeDihedralAnglesEqual (pc : PyramidalCorner) : Prop :=
  sorry  -- This would normally be your precise mathematical definition

-- The theorem to prove:
theorem inscribable_in_cone_if_opposite_dihedral_angles_equal (pc : PyramidalCorner) :
  oppositeDihedralAnglesEqual(pc) ↔ canBeInscribedInCone(pc) :=
  sorry

structure Point
structure Line

def canBeInscribedInCone (pc: PyramidalCorner) : Prop :=
    sorry

end inscribable_in_cone_if_opposite_dihedral_angles_equal_l65_65965


namespace primitive_root_mod_p_of_mod_p_alpha_l65_65603

theorem primitive_root_mod_p_of_mod_p_alpha {p x α : ℕ} (hp : p.prime) (hα : 0 < α) 
  (h : is_primitive_root x (p ^ α)) : is_primitive_root x p :=
sorry

end primitive_root_mod_p_of_mod_p_alpha_l65_65603


namespace invalid_votes_l65_65755

theorem invalid_votes (T V : ℕ) (A B : ℕ) (h1 : T = 90083) (h2 : V = 90000) (h3 : A = 0.45 * V) (h4 : B = 0.55 * V) (h5 : B - A = 9000) :
  (T - V = 83) :=
by
  sorry

end invalid_votes_l65_65755


namespace problem_I_problem_II_problem_III_problem_IV_l65_65692

/-- Problem I: Given: (2x - y)^2 = 1, Prove: y = 2x - 1 ∨ y = 2x + 1 --/
theorem problem_I (x y : ℝ) : (2 * x - y) ^ 2 = 1 → (y = 2 * x - 1) ∨ (y = 2 * x + 1) := 
sorry

/-- Problem II: Given: 16x^4 - 8x^2y^2 + y^4 - 8x^2 - 2y^2 + 1 = 0, Prove: y = 2x - 1 ∨ y = -2x - 1 ∨ y = 2x + 1 ∨ y = -2x + 1 --/
theorem problem_II (x y : ℝ) : 16 * x^4 - 8 * x^2 * y^2 + y^4 - 8 * x^2 - 2 * y^2 + 1 = 0 ↔ 
    (y = 2 * x - 1) ∨ (y = -2 * x - 1) ∨ (y = 2 * x + 1) ∨ (y = -2 * x + 1) := 
sorry

/-- Problem III: Given: x^2 * (1 - |y| / y) + y^2 + y * |y| = 8, Prove: (y = 2 ∧ y > 0) ∨ ((x = 2 ∨ x = -2) ∧ y < 0) --/
theorem problem_III (x y : ℝ) (hy : y ≠ 0) : x^2 * (1 - abs y / y) + y^2 + y * abs y = 8 →
    (y = 2 ∧ y > 0) ∨ ((x = 2 ∨ x = -2) ∧ y < 0) := 
sorry

/-- Problem IV: Given: x^2 + x * |x| + y^2 + (|x| * y^2 / x) = 8, Prove: x^2 + y^2 = 4 ∧ x > 0 --/
theorem problem_IV (x y : ℝ) (hx : x ≠ 0) : x^2 + x * abs x + y^2 + (abs x * y^2 / x) = 8 →
    (x^2 + y^2 = 4 ∧ x > 0) := 
sorry

end problem_I_problem_II_problem_III_problem_IV_l65_65692


namespace cards_problem_l65_65875

theorem cards_problem : 
  ∀ (cards people : ℕ),
  cards = 60 →
  people = 8 →
  ∃ fewer_people : ℕ,
  (∀ p: ℕ, p < people → (p < fewer_people → cards/people < 8)) ∧ 
  fewer_people = 4 := 
by 
  intros cards people h_cards h_people
  use 4
  sorry

end cards_problem_l65_65875


namespace triangles_are_right_angled_l65_65737

/-- Given a point inside a convex pentagon is connected to its vertices,
    dividing the pentagon into five equal non-isosceles triangles.
    Prove that these five triangles are right-angled. -/
theorem triangles_are_right_angled 
  (A B C D E O : Point)
  (h_convex : convex_pentagon A B C D E)
  (h_point_inside : point_inside_pentagon O A B C D E)
  (h_equal_areas : equal_areas (triangle O A B) (triangle O B C) (triangle O C D) (triangle O D E) (triangle O E A))
  (h_non_isosceles : non_isosceles (triangle O A B) (triangle O B C) (triangle O C D) (triangle O D E) (triangle O E A)) : 
  right_angled_triangles (triangle O A B) (triangle O B C) (triangle O C D) (triangle O D E) (triangle O E A) := 
sorry

end triangles_are_right_angled_l65_65737


namespace smallest_sum_abcd_equals_683_l65_65655

noncomputable def smallest_sum_abcd : ℕ :=
  Nat.find_greatest (λ s, ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ a * b * c * d = nat.factorial 12 ∧ a + b + c + d = s) 683

theorem smallest_sum_abcd_equals_683 :
  smallest_sum_abcd = 683 :=
begin
  sorry
end

end smallest_sum_abcd_equals_683_l65_65655


namespace y_value_l65_65041

noncomputable def y_length (AB AD AC : ℝ) (angle_ACD : ℝ) : ℝ :=
  5 * sqrt (3 * (1 - real.cos angle_ACD))

theorem y_value :
  let y := y_length 10 (10 * real.sqrt 3) (10 * real.sqrt 3) (50 * real.pi / 180)
  in y = 5 * real.sqrt (3 * (1 - real.cos (50 * real.pi / 180))) :=
by
  sorry

end y_value_l65_65041


namespace correct_statement_l65_65092

noncomputable def proposition : Prop :=
  ∀ (m n : Line) (α β : Plane),
    (m ≠ n) →
    (α ≠ β) →
    (m ∥ n) →
    (m ⊥ β) →
    (n ⊥ β)

-- Lean statement to create the hypothesis and prove the proposition
theorem correct_statement :=
  proposition

end correct_statement_l65_65092


namespace sum_of_angles_l65_65385

noncomputable def angle_x (arc_angle : ℝ) : ℝ := 3 * arc_angle / 2
noncomputable def angle_y (arc_angle : ℝ) : ℝ := 5 * arc_angle / 2

theorem sum_of_angles 
  (arc_angle : ℝ) 
  (h : arc_angle = 360 / 16) 
  : angle_x arc_angle + angle_y arc_angle = 90 :=
by
  have h_arc_angle : arc_angle = 22.5 := by sorry -- Calculation from condition
  rw [angle_x, angle_y]
  simp [h_arc_angle]
  norm_num
  exact eq.refl 90

end sum_of_angles_l65_65385


namespace max_a2_plus_b2_l65_65490

theorem max_a2_plus_b2 (a b : ℝ) 
  (h : abs (a - 1) + abs (a - 6) + abs (b + 3) + abs (b - 2) = 10) : 
  (a^2 + b^2) ≤ 45 :=
sorry

end max_a2_plus_b2_l65_65490


namespace find_polynomials_l65_65801

-- Define S(k) as the sum of the digits of k
def S (k : ℕ) : ℕ := k.digits 10 |>.sum

-- Define the main theorem
theorem find_polynomials (P : ℕ → ℕ → ℕ) (h1 : ∀ n, n ≥ 2016 → P n > 0)
  (h2 : ∀ n, S (P n) = P (S n)) : 
  (∃ c : ℕ, 1 ≤ c ∧ c ≤ 9 ∧ ∀ n, P n = c) ∨ (∀ n, P n = n) :=
sorry

end find_polynomials_l65_65801


namespace bob_has_winning_strategy_l65_65748

-- Define the conditions of the game
variables (n : ℕ) (c : ℕ) (board : Fin n → ℕ)

-- Conditions on n and c
def valid_n_and_c := n > 2 ∧ 0 < c ∧ c < n

-- Definition of Bob's winning condition
def bob_wins (a : Fin n → ℕ) : Prop :=
  let indices := Finset.univ : Finset (Fin n) in
  let diff := λ (i : Fin n), (a i) - (a (⟨(i + 1) % n⟩ : Fin n)) in
  (indices.product indices).fold
    (λ (acc : ℕ) (pair : Fin n × Fin n), acc * (diff pair.1)) 1 % n = 0 ∨
  (indices.product indices).fold
    (λ (acc : ℕ) (pair : Fin n × Fin n), acc * (diff pair.1)) 1 % n = c

-- The main theorem statement
theorem bob_has_winning_strategy (h_n_c : valid_n_and_c n c) : ∃ a : Fin n → ℕ, bob_wins n c a := sorry

end bob_has_winning_strategy_l65_65748


namespace incorrect_statements_l65_65698

/- Defining the conditions -/
def domain_f (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 2 → (f x).domain
def domain_fx (f : ℝ → ℝ) := ∀ x, 0 ≤ x ∧ x ≤ 1 → (f (2 * x)).domain

def statementB (f : ℝ → ℝ) : Prop :=
  ∀ x, 1 ≤ x → f (1 + sqrt x) = 2 * x + 1 → f x = 2 * x ^ 2 + 4 * x + 3

def statementC (f : ℝ → ℝ) : Prop :=
  ∀ y, y = 4^x + 2^x + 1 → y ≥ -1 / 4

def statementD (f : ℝ → ℝ) : Prop :=
  (∀ x, x ≤ 1 → f x = -x^2 - a * x - 5) ∧
  (∀ x, x > 1 → f x = a / x) ∧
  (∀ x, f x ≤ f (x + 1))

theorem incorrect_statements (f : ℝ → ℝ) :
  domain_f f →
  domain_fx f →
  (statementB f = false) ∧
  (statementC f = false) ∧
  statementD f :=
by 
  sorry

end incorrect_statements_l65_65698


namespace triangle_concurrence_product_l65_65579

theorem triangle_concurrence_product
  (A B C D E F P : Type)
  (h_concurrent : concurrent (Line.mk A D) (Line.mk B E) (Line.mk F C) P)
  (h_on_sides :
    (D ∈ Segment B C) ∧
    (E ∈ Segment C A) ∧
    (F ∈ Segment A B))
  (h_sum_ratios : (AP / PD) + (BP / PE) + (CP / PF) = 88)
  (h_mult_ratios : (AP / PD) * (BP / PE) = 32) :
  (AP / PD) * (BP / PE) * (CP / PF) = 1792 := sorry

end triangle_concurrence_product_l65_65579


namespace polynomial_not_factorable_l65_65927

theorem polynomial_not_factorable (m n : ℕ) (h₁ : m > 0) (h₂ : n > 0) :
  ¬ ∃ (Q R : Polynomial ℤ), Q.degree > 0 ∧ R.degree > 0 ∧ (Q * R = Polynomial.Coeff ℤ m * (Polynomial.X ^ 2 - 100) ^ n - 11) :=
by
  sorry

end polynomial_not_factorable_l65_65927


namespace product_of_inverses_l65_65029

theorem product_of_inverses : 
  ((1 - 1 / (3^2)) * (1 - 1 / (5^2)) * (1 - 1 / (7^2)) * (1 - 1 / (11^2)) * (1 - 1 / (13^2)) * (1 - 1 / (17^2))) = 210 / 221 := 
by {
  sorry
}

end product_of_inverses_l65_65029


namespace miguel_book_total_pages_l65_65252

theorem miguel_book_total_pages :
  let pages_first_4_days := 4 * 48,
      pages_next_5_days := 5 * 35,
      pages_subsequent_4_days := 4 * 28,
      pages_first_13_days := pages_first_4_days + pages_next_5_days + pages_subsequent_4_days,
      pages_last_day := 19,
      total_pages := pages_first_13_days + pages_last_day
  in total_pages = 498 :=
by
  sorry

end miguel_book_total_pages_l65_65252


namespace max_term_in_sequence_l65_65177

theorem max_term_in_sequence (a : ℕ → ℝ) (d : ℝ) (n : ℕ) (h : d ≠ 0) (h_pos : ∀ i j, 0 < a i * a j) :
  n = 1990 →
  (∀ i, a i = a 1 + (i-1) * d) →
  let b := λ k, a k * a (1991 - k) in
  ∃ k, b k = max (b 995) (b 996) :=
by
  intros
  sorry

end max_term_in_sequence_l65_65177


namespace part1_part2_l65_65489

variables {x m : ℝ}

-- Proposition p
def p : Prop := abs (x - 3) < 1

-- Proposition q
def q : Prop := (m - 2 < x) ∧ (x < m + 1)

-- Theorem Part (1)
theorem part1 : ¬p → (x ≤ 2 ∨ x ≥ 4) := by 
  unfold p
  sorry

-- Theorem Part (2)
theorem part2 (h : p → q) : 3 ≤ m ∧ m ≤ 4 := by 
  unfold p q
  sorry

end part1_part2_l65_65489


namespace paul_erasers_l65_65267

theorem paul_erasers (E : ℕ) :
  ∀ (crayons_birthday : ℕ) (crayons_end : ℕ) (diff_crayons_erasers : ℕ), 
  crayons_birthday = 617 → crayons_end = 523 → diff_crayons_erasers = 66 → 
  (E = crayons_end - diff_crayons_erasers) → 
  E = 457 := by
  intros crayons_birthday crayons_end diff_crayons_erasers hb he hd hE
  rw [he, hd] at hE
  exact hE
  sorry

end paul_erasers_l65_65267


namespace arrangement_count_of_1_to_6_divisible_by_7_l65_65899

theorem arrangement_count_of_1_to_6_divisible_by_7 :
  {s : List ℕ // s.perm [1, 2, 3, 4, 5, 6] ∧ ∀ a b c, List.pairs s (List.pairs s.tail s.tail^.tail = a :: b :: c :: _) → (a * c - b^2) % 7 = 0 } → 12 :=
sorry

end arrangement_count_of_1_to_6_divisible_by_7_l65_65899


namespace hyperbola_equation_l65_65809

def is_hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 - (y^2 / b^2) = 1

theorem hyperbola_equation
  (O : ℝ × ℝ)
  (H_origin_center : O = (0, 0))
  (point : ℝ × ℝ)
  (H_pass_through_point : point = (Real.sqrt 5, 4))
  (H_focus_parabola : (∀ x y : ℝ, y^2 = 4 * x → (x, y) = (1,0)))
  : ∃ b : ℝ, is_hyperbola_equation 1 b Real.sqrt(5) 4 ∧ is_hyperbola_equation 1 b 1 0 :=
sorry

end hyperbola_equation_l65_65809


namespace total_selection_schemes_l65_65045

theorem total_selection_schemes (buses: Finset ℕ) (spots: Finset ℕ) (A B: ℕ) (Wulan_Butong: ℕ) 
  (h1 : buses = {0, 1, 2, 3, 4, 5}) -- Representing buses A, B, C, D, E, F as 0, 1, 2, 3, 4, 5
  (h2 : spots = {0, 1, 2, 3}) -- Representing spots as 0, 1, 2, 3
  (h3 : ¬ Wulan_Butong ∈ {0, 1}) -- Representing that A and B (i.e., 0 and 1) do not go to Wulan Butong
  : buses.card = 6 ∧ 
    spots.card = 4 →
    (∑ s in spots.powerset, if ∃ b ∈ s, b = Wulan_Butong then (s.erase Wulan_Butong).card! else (s.card - 1)!) = 240 := 
by {
  sorry -- This is where the proof should go
}

end total_selection_schemes_l65_65045


namespace price_reduction_equation_l65_65722

theorem price_reduction_equation (x : ℝ) : 200 * (1 - x) ^ 2 = 162 :=
by
  sorry

end price_reduction_equation_l65_65722


namespace length_of_diagonal_l65_65075

/--
Given a quadrilateral with sides 9, 11, 17, and 13,
the number of different whole numbers that could be the length of the diagonal
represented by the dashed line is 15.
-/
theorem length_of_diagonal (x : ℕ) :
  (∀ x, 5 ≤ x ∧ x ≤ 19 → x ∈ {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}) →
  card {x | 5 ≤ x ∧ x ≤ 19} = 15 :=
by
  sorry

end length_of_diagonal_l65_65075


namespace no_positive_reals_satisfy_equations_l65_65972

theorem no_positive_reals_satisfy_equations :
  ¬ ∃ (a b c d : ℝ), (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (0 < d) ∧
  (a / b + b / c + c / d + d / a = 6) ∧ (b / a + c / b + d / c + a / d = 32) :=
by sorry

end no_positive_reals_satisfy_equations_l65_65972


namespace sum_of_consecutive_integers_log5_50_l65_65450

noncomputable def log5 (x : ℝ) : ℝ :=
  Real.log x / Real.log 5

theorem sum_of_consecutive_integers_log5_50 :
  ∃ (c d : ℤ), c < log5 50 ∧ log5 50 < d ∧ d = c + 1 ∧ c + d = 5 := by
  sorry

end sum_of_consecutive_integers_log5_50_l65_65450


namespace proj_length_eq_2_5_l65_65527

variables (u z : ℝ^3)  -- Assuming ℝ^3 as the vector space for u and z.

-- Conditions
axiom norm_u : ∥u∥ = 5
axiom norm_z : ∥z∥ = 8
axiom dot_u_z : u ⬝ z = 20

-- The proof problem statement
theorem proj_length_eq_2_5 : ∥((u ⬝ z) / ∥z∥^2) • z∥ = 2.5 := by
  sorry

end proj_length_eq_2_5_l65_65527


namespace country_X_tax_l65_65887

theorem country_X_tax (I T x : ℝ) (hI : I = 51999.99) (hT : T = 8000) (h : T = 0.14 * x + 0.20 * (I - x)) : 
  x = 39999.97 := sorry

end country_X_tax_l65_65887


namespace sequence_result_l65_65174

theorem sequence_result (a b c d e f g h i j k : ℕ)
  (h1 : a + b = c) (h2 : c^2 + 1 = d)
  (h3 : e + f = g) (h4 : g^2 + 1 = h)
  (h5 : i + j = k) (h6 : k^2 + 1 = l) :
--- Given the conditions
  (a = 1) (b = 2) (c = 3) (d = 10)
  (e = 2) (f = 3) (g = 5) (h = 26)
  (i = 3) (j = 4) (k = 7) (l = 50) :
  (4 + 5)^2 + 1 = 82 :=
begin
  -- Assuming the pattern holds true as given and needs to be proven.
  sorry
end

end sequence_result_l65_65174


namespace ratio_of_areas_l65_65978

open Real

noncomputable def radius_of_larger_circle (s_1 : ℝ) : ℝ := s_1 / (2 * sqrt 2)
noncomputable def radius_of_smaller_circle (R : ℝ) : ℝ := R / 2
noncomputable def side_length_of_IJKL (r : ℝ) : ℝ := r * sqrt 2
noncomputable def area_of_square (s : ℝ) : ℝ := s ^ 2

theorem ratio_of_areas :
  let s_1 := 4 in
  let R := radius_of_larger_circle s_1 in
  let r := radius_of_smaller_circle R in
  let s_2 := side_length_of_IJKL r in
  area_of_square s_2 / area_of_square s_1 = 1 / 4 :=
by
  sorry

end ratio_of_areas_l65_65978


namespace rebus_no_solution_l65_65967

open Nat
open DigitFin

theorem rebus_no_solution (K U S Y : Fin 10) (h1 : K ≠ U) (h2 : K ≠ S) (h3 : K ≠ Y) (h4 : U ≠ S) (h5 : U ≠ Y) (h6 : S ≠ Y) :
  let KUSY := K.val * 1000 + U.val * 100 + S.val * 10 + Y.val
  let UKSY := U.val * 1000 + K.val * 100 + S.val * 10 + Y.val
  let UKSUS := U.val * 100000 + K.val * 10000 + S.val * 1000 + U.val * 100 + S.val * 10 + S.val
  KUSY + UKSY ≠ UKSUS := by
sorry

end rebus_no_solution_l65_65967


namespace power_sum_divisible_by_five_l65_65582

theorem power_sum_divisible_by_five : 
  (3^444 + 4^333) % 5 = 0 := 
by 
  sorry

end power_sum_divisible_by_five_l65_65582


namespace consecutive_integers_equality_l65_65639

theorem consecutive_integers_equality (n : ℕ) (h_eq : (n - 3) + (n - 2) + (n - 1) + n = (n + 1) + (n + 2) + (n + 3)) : n = 12 :=
by {
  sorry
}

end consecutive_integers_equality_l65_65639


namespace solve_system_l65_65602

def system_of_equations (n : ℕ) (x : ℕ → ℕ) : Prop :=
  (n > 5) ∧ 
  (x 1 + x 2 + x 3 + ∑ i in finset.range (n-2), x i.succ.succ.succ = n + 2) ∧ 
  (x 1 + 2 * x 2 + 3 * x 3 + ∑ i in finset.range (n-2), (i+4) * x i.succ.succ.succ = 2 * n + 2) ∧ 
  (x 1 + 4 * x 2 + 9 * x 3 + ∑ i in finset.range (n-2), (i+4)^2 * x i.succ.succ.succ = n^2 + n + 4) ∧ 
  (x 1 + 8 * x 2 + 27 * x 3 + ∑ i in finset.range (n-2), (i+4)^3 * x i.succ.succ.succ = n^3 + n + 8)

theorem solve_system (n : ℕ) (x : ℕ → ℕ) (hn : n > 5) :
  system_of_equations n x → 
  (x 1 = n) ∧ (x 2 = 1) ∧ (∀ i, 3 ≤ i ∧ i ≤ n → x i = 0) ∧ (x n = 1) :=
by
  sorry

end solve_system_l65_65602


namespace system_of_equations_l65_65374

/-- Given A and B holding some coins such that:
    1. A gets half of B's money and his total is 48 coins.
    2. B gets two-thirds of A's money and his total is 48 coins.
    Prove that the system of equations is:
    \[
    \left\{
      \begin{array}{l}
        x + \frac{1}{2}y = 48 \\
        y + \frac{2}{3}x = 48
      \end{array}
    \right.
    \]
-/
theorem system_of_equations (x y : ℝ) :
  (x + (1 / 2) * y = 48) ∧ (y + (2 / 3) * x = 48) :=
begin
  sorry -- proof steps would go here
end

end system_of_equations_l65_65374


namespace possible_diagonal_lengths_l65_65077

theorem possible_diagonal_lengths (y : ℕ) :
  (y > 4 ∧ y < 20) ↔ 15 := by
  sorry

end possible_diagonal_lengths_l65_65077


namespace general_formula_a_n_maximum_value_T_n_l65_65848

def f (x : ℝ) := -3 * x^2 + 6 * x

def S (n : ℕ) : ℝ := -3 * (n:ℝ)^2 + 6 * (n:ℝ)

def a (n : ℕ) : ℝ := if n = 1 then S 1 else S n - S (n - 1)

def b (n : ℕ) : ℝ := (1 / 2) ^ (n - 1)

def c (n : ℕ) : ℝ := (a n * b n) / 6

def T (n : ℕ) : ℝ := (Finset.range n).sum (λ i => c (i + 1))

-- The problem to prove the equivalence
theorem general_formula_a_n (n : ℕ) : a n = 9 - 6 * n := sorry

theorem maximum_value_T_n : T 1 = 1 / 2 := sorry

end general_formula_a_n_maximum_value_T_n_l65_65848


namespace angle_ABC_is_60_degrees_l65_65242

open Real

def point3d := ℝ × ℝ × ℝ

def A : point3d := (-3, 1, 5)
def B : point3d := (-4, 0, 1)
def C : point3d := (-5, 0, 2)

def dist (p1 p2 : point3d) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

def AB := dist A B
def AC := dist A C
def BC := dist B C

def cos_angle_ABC :=
  (AB^2 + BC^2 - AC^2) / (2 * AB * BC)

noncomputable def angle_ABC : ℝ :=
  real.arccos cos_angle_ABC

theorem angle_ABC_is_60_degrees : angle_ABC = 60 :=
begin
  sorry
end

end angle_ABC_is_60_degrees_l65_65242


namespace experiment_arrangements_l65_65754

/-- 
We are given 6 experiments labeled 0, 1, 2, 3, 4, 5. 
The experiment labeled 0 cannot be first. 
The label of the last experiment must be smaller than the one immediately preceding it.
Given these conditions, prove that the total number of valid arrangements of the experiments is 300.
-/
theorem experiment_arrangements :
  let experiments : List Nat := [0, 1, 2, 3, 4, 5] in
  let valid_arrangements := 
    {arr | arr ∈ experiments.permutations ∧ 
           arr.head ≠ 0 ∧ 
           arr.reverse.head < arr.reverse.tail.head} in
  valid_arrangements.card = 300 := by
  sorry

end experiment_arrangements_l65_65754


namespace arc_length_EF_is_9_l65_65650

variable (D : Type) [geometric_circle D]
variable (circumference_D : ℝ) (arc_EF_angle : ℝ)
variable (arc_length_EF : ℝ)

axiom circumference_of_D : circumference_D = 72
axiom angle_EF : arc_EF_angle = 45

theorem arc_length_EF_is_9 : arc_length_EF = 9 :=
by
  sorry

end arc_length_EF_is_9_l65_65650


namespace trig_expression_equality_l65_65415

theorem trig_expression_equality :
  (Real.tan (60 * Real.pi / 180) + 2 * Real.sin (45 * Real.pi / 180) - 2 * Real.cos (30 * Real.pi / 180)) 
  = Real.sqrt 2 :=
by
  have h1 : Real.tan (60 * Real.pi / 180) = Real.sqrt 3 := by sorry
  have h2 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  sorry

end trig_expression_equality_l65_65415


namespace divisors_of_9_fac_multiple_of_10_l65_65140

theorem divisors_of_9_fac_multiple_of_10 :
  ∀ d ∈ (finset.Ico 1 (9!)).filter (λ e, 10 ∣ e), d ∈ (finset.range 71) :=
begin
  sorry
end

end divisors_of_9_fac_multiple_of_10_l65_65140


namespace arithmetic_sequences_l65_65334

theorem arithmetic_sequences (a d : ℝ) 
  (h1 : (a - d)^2 + a^2 = 100) 
  (h2 : (a + d)^2 + a^2 = 164) :
  (a, a - d, a + d) = (8, 6, 10) ∨ (a, a - d, a + d) = (-8, -10, -6) ∨
  (a, a - d, a + d) = (sqrt 2, -sqrt 2, 9 * sqrt 2) ∨
  (a, a - d, a + d) = (-sqrt 2, -9 * sqrt 2, sqrt 2) :=
sorry

end arithmetic_sequences_l65_65334


namespace monotonic_decreasing_interval_l65_65659

theorem monotonic_decreasing_interval (k : ℤ) :
  ∃ a b : ℝ, (∀ x, a ≤ x ∧ x ≤ b → ¬strict_mono_incr_on (λ x, Real.sin (2 * x - π / 3)) {x | a ≤ x ∧ x ≤ b}) ∧ 
  a = k * π - π / 12 ∧ b = k * π + 5 * π / 12 :=
sorry

end monotonic_decreasing_interval_l65_65659


namespace power_expansion_l65_65348

theorem power_expansion (a b : ℕ) (h : 100 = 10^2) : 100^50 = 10^100 := 
by
  rw h
  sorry

end power_expansion_l65_65348


namespace sum_ineq_l65_65597

-- Definitions based on given problem
namespace Sequence

def is_arithmetic {R : Type} [OrderedAddCommGroup R] (a : ℕ → R) : Prop :=
∀ n m : ℕ, a (n + m) = a n + a m

def is_geometric {R : Type} [LinearOrderedField R] (b : ℕ → R) : Prop :=
∀ n m : ℕ, b (n + m) = b n * b m

variables {R : Type} [LinearOrderedField R]

def a (n : ℕ) : R := 2 * n - 1
def b (n : ℕ) : R := 2 ^ (n - 1)

def S_n (n : ℕ) : R :=
∑ i in finset.range n, a i.succ / b i.succ

-- Conditions of the problem
variables (a_1_eq : a 1 = (1 : R))
variables (b_1_eq : b 1 = (1 : R))
variables (cond1 : a 3 + b 5 = 21)
variables (cond2 : a 5 + b 3 = 13)

-- Main statement to prove
theorem sum_ineq (n : ℕ) : S_n n < 6 :=
sorry

end Sequence

end sum_ineq_l65_65597


namespace rotten_fruits_without_smell_is_correct_l65_65180

def total_apples : ℕ := 200
def total_oranges : ℕ := 150
def total_pears : ℕ := 100

def rotten_apples_rate : ℚ := 0.40
def rotten_oranges_rate : ℚ := 0.25
def rotten_pears_rate : ℚ := 0.35

def smelly_rotten_apples_rate : ℚ := 0.70
def smelly_rotten_oranges_rate : ℚ := 0.50
def smelly_rotten_pears_rate : ℚ := 0.80

def rotten_fruits_without_smell : ℕ := 50

theorem rotten_fruits_without_smell_is_correct :
  let rotten_apples := (total_apples : ℚ) * rotten_apples_rate,
      rotten_oranges := (total_oranges : ℚ) * rotten_oranges_rate,
      rotten_pears := (total_pears : ℚ) * rotten_pears_rate,
      
      smelly_rotten_apples := rotten_apples * smelly_rotten_apples_rate,
      smelly_rotten_oranges := rotten_oranges * smelly_rotten_oranges_rate,
      smelly_rotten_pears := rotten_pears * smelly_rotten_pears_rate,
      
      non_smelly_rotten_apples := rotten_apples - smelly_rotten_apples,
      non_smelly_rotten_oranges := rotten_oranges - smelly_rotten_oranges,
      non_smelly_rotten_pears := rotten_pears - smelly_rotten_pears,
      
      total_non_smelly_rotten_fruits := non_smelly_rotten_apples + non_smelly_rotten_oranges + non_smelly_rotten_pears
  in total_non_smelly_rotten_fruits = rotten_fruits_without_smell :=
sorry

end rotten_fruits_without_smell_is_correct_l65_65180


namespace parabolas_similar_l65_65275

theorem parabolas_similar :
  ∀ x y : ℝ, 
  (¬((y = x^2) ∧ (y = 2 * x^2)) → ∀ a : ℝ, 2 * (a, 2 * a^2) ∈ set_of (λ p : ℝ × ℝ, p.snd = p.fst ^ 2)) := 
  sorry

end parabolas_similar_l65_65275


namespace max_k_value_l65_65781

def maximum_k (k : ℕ) : ℕ := 2

theorem max_k_value
  (k : ℕ)
  (h1 : 2 * k + 1 ≤ 20)  -- Condition implicitly implied by having subsets of a 20-element set
  (h2 : ∀ (s t : Finset (Fin 20)), s.card = 7 → t.card = 7 → s ≠ t → (s ∩ t).card = k) : k ≤ maximum_k k := 
by {
  sorry
}

end max_k_value_l65_65781


namespace domain_of_f_x_plus_2_l65_65880

theorem domain_of_f_x_plus_2 (f : ℝ → ℝ) (dom_f_x_minus_1 : ∀ x, 1 ≤ x ∧ x ≤ 2 → 0 ≤ x-1 ∧ x-1 ≤ 1) :
  ∀ y, 0 ≤ y ∧ y ≤ 1 ↔ -2 ≤ y-2 ∧ y-2 ≤ -1 :=
by
  sorry

end domain_of_f_x_plus_2_l65_65880


namespace difference_of_squares_l65_65414

theorem difference_of_squares : (540^2 - 460^2 = 80000) :=
by
  have a := 540
  have b := 460
  have identity := (a + b) * (a - b)
  sorry

end difference_of_squares_l65_65414


namespace find_m_l65_65841

-- Given conditions
def z₁ : ℂ := 2 - complex.i
def z₂ (m : ℝ) : ℂ := complex.of_real m + complex.i

-- Define being purely imaginary for a complex number
def purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem find_m : 
  ∃ m : ℝ, purely_imaginary (z₁ * z₂ m) ↔ m = -1/2 := 
by
  sorry

end find_m_l65_65841


namespace water_pouring_fraction_l65_65002

theorem water_pouring_fraction (k : ℕ) (hk : k = 7) :
  (∏ i in Finset.range(7 + 1), (2 * i - 1) / (2 * i + 1)) = (1 / 15) :=
by
  rw hk
  sorry

end water_pouring_fraction_l65_65002


namespace Petya_has_24_chips_l65_65460

noncomputable def PetyaChips (x y : ℕ) : ℕ := 3 * x - 3

theorem Petya_has_24_chips (x y : ℕ) (h1 : y = x - 2) (h2 : 3 * x - 3 = 4 * y - 4) : PetyaChips x y = 24 :=
by
  sorry

end Petya_has_24_chips_l65_65460


namespace quadratic_representation_and_integrality_l65_65604

theorem quadratic_representation_and_integrality
  (a b c : ℝ) :
  ∃ (d e f : ℝ), (∀ x : ℝ, ax^2 + bx + c = d/2 * x * (x - 1) + e * x + f) ∧
  (∀ n : ℤ, (ax^2 + bx + c).eval ↑n ∈ ℤ ↔ d ∈ ℤ ∧ e ∈ ℤ ∧ f ∈ ℤ) :=
by
  let p (x : ℝ) := ax^2 + bx + c
  let d := 2 * a
  let e := a + b
  let f := c
  use [d, e, f]
  split
  · intro x
    sorry -- This part needs to show that the polynomials match
  · intro n
    sorry -- This part needs to show the integer condition

end quadratic_representation_and_integrality_l65_65604


namespace find_f1_l65_65607

def f (x : ℝ) (t : ℝ) : ℝ :=
  if x < 2 then 2 * t ^ x else Real.log (x^2 - 1) / Real.log t

theorem find_f1 (t : ℝ) (h : f 2 t = 1) : f 1 t = 6 :=
begin
  sorry
end

end find_f1_l65_65607


namespace average_of_all_results_l65_65367

theorem average_of_all_results (n₁ n₂ : ℕ) (a₁ a₂ : ℕ) (h₁ : n₁ = 40) (h₂ : a₁ = 30) 
  (h₃ : n₂ = 30) (h₄ : a₂ = 40) : 
  (n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 34.2857 := 
by
  sorry

end average_of_all_results_l65_65367


namespace problem_b_problem_c_problem_d_l65_65507

noncomputable def circle (x y : ℝ) := x^2 + y^2 - 4*x = 0
noncomputable def line (k x y : ℝ) := k * x - y + 1 - 2 * k = 0

theorem problem_b (k : ℝ) (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) :
  k = 1 →
  (∃ x y : ℝ, C x y ∧ ∃ d : ℝ, d = dist (2 : ℝ, 0) l x + 2 ∧ d = 2 + sqrt 2 / 2) ∧
  (∃ P : ℝ × ℝ, P = (2, 1) ∧ l k P.1 P.2 ∧ ∀ x y : ℝ, C x y → ∃n: ℕ, n = 3) :=
sorry

theorem problem_c (k : ℝ) (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) :
  (∃ P : ℝ × ℝ, P = (2,0) ∧ ∀ x y : ℝ, k^2 = 0 ∧ dist P (x, y) = 1 ∧ C x y) → k = 0 :=
sorry

theorem problem_d (k : ℝ) (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → ℝ → Prop) :
  (∃ M N : ℝ × ℝ, C M.1 M.2 ∧ C N.1 N.2 ∧ l k M.1 M.2 ∧ l k N.1 N.2) →
  (∃ Q : ℝ × ℝ, ∃ c r : ℝ, ∀ x y : ℝ, Q = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧ C Q.1 Q.2) :=
sorry

end problem_b_problem_c_problem_d_l65_65507


namespace range_of_x_no_such_m_l65_65376

-- Define two main propositions to be proved.

-- First problem: Range of x for given inequality conditions
theorem range_of_x (x : ℝ) : 
  (∀ m : ℝ, (-2 ≤ m ∧ m ≤ 2) → 2 * x - 1 > m * (x^2 - 1)) → 
  (sqrt 7 - 1) / 2 < x ∧ x < (sqrt 3 + 1) / 2 :=
sorry

-- Second problem: Non-existence of m for given x range conditions
theorem no_such_m : 
  ¬(∃ m : ℝ, ∀ x : ℝ, (-2 ≤ x ∧ x ≤ 2) → 2 * x - 1 > m * (x^2 - 1)) :=
sorry

end range_of_x_no_such_m_l65_65376


namespace ratio_of_areas_l65_65595

open Function

def point_in_triangle (P A B C : AffineSpace ℝ) := 
  ∃ (k₁ k₂ k₃ : ℝ), k₁ + k₂ + k₃ = 1 ∧ 0 < k₁ ∧ 0 < k₂ ∧ 0 < k₃ ∧ 
  k₁ • (A : ℝ) + k₂ • (B : ℝ) + k₃ • (C : ℝ) = (P : ℝ)

theorem ratio_of_areas (A B C P : AffineSpace ℝ)
  (h1 : point_in_triangle P A B C)
  (h2 : vector (P -ᵥ A) + 3 • vector (P -ᵥ B) + 5 • vector (P -ᵥ C) = 0) :
  (area_triangle A B C) / (area_triangle A P C) = 4 := sorry

end ratio_of_areas_l65_65595


namespace sides_not_proportional_l65_65113

theorem sides_not_proportional
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  ¬ (∃ k : ℝ, (log10 a = k * a) ∧ (log10 b = k * b) ∧ (log10 c = k * c)) :=
sorry

end sides_not_proportional_l65_65113


namespace range_fraction_a_b_l65_65559

theorem range_fraction_a_b (A B C a b : ℝ) (h_acute : A + B + C = 180) 
  (h_angle_A : 0 < A ∧ A < 90)
  (h_angle_B : 0 < B ∧ B < 90)
  (h_angle_C : 0 < C ∧ C < 90)
  (h_A_eq_2B : A = 2 * B)
  (h_sides : (a : ℝ) / (b : ℝ) = 2 * Real.cos B) : 
  sqrt 2 < (a / b) ∧ (a / b) < sqrt 3 := 
by sorry

end range_fraction_a_b_l65_65559


namespace cos_alpha_plus_pi_over_4_l65_65807

open Real

theorem cos_alpha_plus_pi_over_4 
  (α : ℝ) (h1 : π < α) (h2 : α < (3 / 2) * π) (h3 : tan (α + π) = 4 / 3) : 
  cos (α + π / 4) = sqrt(2) / 10 :=
sorry

end cos_alpha_plus_pi_over_4_l65_65807


namespace sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1_sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1_l65_65704

open Real

-- Problem (a)
theorem sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1 (n k : Nat) :
  (sqrt 2 - 1)^n = sqrt k - sqrt (k - 1) :=
sorry

-- Problem (b)
theorem sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1 (m n k : Nat) :
  (sqrt m - sqrt (m - 1))^n = sqrt k - sqrt (k - 1) :=
sorry

end sqrt2_minus_1_eq_sqrtk_sqrtk_minus_1_sqrtm_minus_sqrtm_minus_1_eq_sqrtk_sqrtk_minus_1_l65_65704


namespace additional_income_needed_to_meet_goal_l65_65395

def monthly_current_income : ℤ := 4000
def annual_goal : ℤ := 60000
def additional_amount_per_month (monthly_current_income annual_goal : ℤ) : ℤ :=
  (annual_goal - (monthly_current_income * 12)) / 12

theorem additional_income_needed_to_meet_goal :
  additional_amount_per_month monthly_current_income annual_goal = 1000 :=
by
  sorry

end additional_income_needed_to_meet_goal_l65_65395


namespace delivery_boxes_l65_65386

-- Define the conditions
def stops : ℕ := 3
def boxes_per_stop : ℕ := 9

-- Define the total number of boxes
def total_boxes : ℕ := stops * boxes_per_stop

-- State the theorem
theorem delivery_boxes : total_boxes = 27 := by
  sorry

end delivery_boxes_l65_65386


namespace proof_problem_1_proof_problem_2_l65_65032
noncomputable def problem_1 : Real :=
  Real.sqrt 9 + Real.abs (-3) + (Real.cbrt (-27)) - ((-1:ℝ) ^ 2019)

noncomputable def problem_2 : Real :=
  Real.sqrt ((-6) ^ 2) + Real.abs (1 - Real.sqrt 2) - Real.cbrt 8

theorem proof_problem_1 : problem_1 = 4 := by
  sorry

theorem proof_problem_2 : problem_2 = 3 + Real.sqrt 2 := by
  sorry

end proof_problem_1_proof_problem_2_l65_65032


namespace number_of_solutions_l65_65865

theorem number_of_solutions :
  {p : ℤ × ℤ // p.1 ^ 2020 + p.2 ^ 2 = 2 * p.2}.to_finset.card = 4 := 
sorry

end number_of_solutions_l65_65865


namespace sum_of_divisors_330_l65_65683

def is_sum_of_divisors (n : ℕ) (sum : ℕ) :=
  sum = ∑ d in divisors n, d

theorem sum_of_divisors_330 : is_sum_of_divisors 330 864 :=
by {
  -- sorry
}

end sum_of_divisors_330_l65_65683


namespace matrix_addition_l65_65766

variable (A B : Matrix (Fin 2) (Fin 2) ℤ) -- Define matrices with integer entries

-- Define the specific matrices used in the problem
def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![ ![2, 3], ![-1, 4] ]

def matrix_B : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![-1, 8], ![-3, 0] ]

-- Define the result matrix
def result_matrix : Matrix (Fin 2) (Fin 2) ℤ := 
  ![ ![3, 14], ![-5, 8] ]

-- The theorem to prove
theorem matrix_addition : 2 • matrix_A + matrix_B = result_matrix := by
  sorry -- Proof omitted

end matrix_addition_l65_65766


namespace filtration_minimum_l65_65726

noncomputable def lg : ℝ → ℝ := sorry

theorem filtration_minimum (x : ℕ) (lg2 : ℝ) (lg3 : ℝ) (h1 : lg2 = 0.3010) (h2 : lg3 = 0.4771) :
  (2 / 3 : ℝ) ^ x ≤ 1 / 20 → x ≥ 8 :=
sorry

end filtration_minimum_l65_65726


namespace matrix_vector_equation_l65_65226

variables {α : Type*} [CommRing α] (M : Matrix (Fin 2) (Fin 2) α) 
v w u : Vector (Fin 2) α

theorem matrix_vector_equation 
  (Mv : M.mulVec v = ![2, -3])
  (Mw : M.mulVec w = ![-1, 4])
  (Mu : M.mulVec u = ![3, 0]) :
  M.mulVec (3 • v - 4 • w + u) = 
    ![13, -25] := 
    sorry

end matrix_vector_equation_l65_65226


namespace wheel_velocity_upper_lower_l65_65583

variable (v r : ℝ) (ω : ℝ := v / r)

theorem wheel_velocity_upper_lower :
  (∀ p, (p = 0 ∨ p = 2 * v) :=
  by
    -- The point of contact has zero speed relative to the ground
    sorry
  
    -- The uppermost point has twice the speed of the translational speed
    sorry

end wheel_velocity_upper_lower_l65_65583


namespace limit_problem_1_l65_65378

theorem limit_problem_1 : 
  limit (λ x, (5 * x^2 + 3 * x - 4) / (x^2 - 1)) 2 = 22 / 3 :=
sorry

end limit_problem_1_l65_65378


namespace verify_pairs_l65_65933

-- Noncomputable definition is used here due to the transformation from a mathematical problem.
noncomputable def arithmetic_geometric_pairs : set (ℕ × ℤ) :=
  {p | ∃ (a_1 d : ℤ) (n : ℕ), n = p.1 ∧ d ≠ 0 ∧ n ≥ 4 ∧ (p.2 = a_1 / d) ∧
    (∀ k: ℕ, k < n → ∃ (a_i : ℤ), a_i = a_1 + k * d) ∧
    (∃ t: ℕ, t < n ∧ 
      (∀ i j: ℕ, i < t ∧ j < t ∧ i ≠ j → (a_1 + i * d) * (a_1 + t * d) = (a_1 + ((i + j - t) % n) * d)))}

theorem verify_pairs : arithmetic_geometric_pairs = {(4, -4), (4, 1)} :=
sorry

end verify_pairs_l65_65933


namespace fewest_removal_proof_l65_65791

noncomputable def fewest_toothpicks_to_remove 
  (total_toothpicks : ℕ)
  (num_triangles : ℕ)
  (num_squares : ℕ)
  (toothpicks_per_triangle : ℕ)
  (toothpicks_per_square : ℕ)
  (shared_toothpicks : ℕ)
  : ℕ :=
  begin
    sorry
  end

theorem fewest_removal_proof :
  fewest_toothpicks_to_remove 50 10 4 3 4 4 = 10 :=
by sorry

end fewest_removal_proof_l65_65791


namespace find_n_l65_65380

theorem find_n (A_seq B_seq : ℕ → ℕ) (n : ℕ) 
  (hA_seq : ∀ k, A_seq k = 1 + 2 * k) 
  (hB_seq : ∀ k, B_seq k = n - 2 * k) 
  (A_counts_to : A_seq 9 = 19) 
  (B_counts_to : B_seq 9 = 89) : 
  n = 107 :=
by 
  have hA9 : A_seq 9 = 1 + 2 * 9, from hA_seq 9,
  have hB9 : B_seq 9 = n - 2 * 9, from hB_seq 9,
  rw [A_counts_to, hA9] at hB9,
  rw [B_counts_to] at hB9,
  exact sorry

end find_n_l65_65380


namespace problem_solution_l65_65195

-- Definitions and assumptions
variables (priceA priceB : ℕ)
variables (numBooksA numBooksB totalBooks : ℕ)
variables (costPriceA : priceA = 45)
variables (costPriceB : priceB = 65)
variables (totalCost : priceA * numBooksA + priceB * numBooksB ≤ 3550)
variables (totalBooksEq : numBooksA + numBooksB = 70)

-- Proof problem
theorem problem_solution :
  priceA = 45 ∧ priceB = 65 ∧ ∃ (numBooksA : ℕ), numBooksA ≥ 50 :=
by
  sorry

end problem_solution_l65_65195


namespace slope_AD_l65_65393

-- Define the problem conditions
def ellipse_eq : (ℝ × ℝ) → Prop := λ p, (p.1^2 / 4 + p.2^2 / 2 = 1)

structure ParallelogramInscribed (A B C D : ℝ × ℝ) : Prop :=
(ins_ellipse : ellipse_eq A ∧ ellipse_eq B ∧ ellipse_eq C ∧ ellipse_eq D)
(slope_AB_1 : ∃ t : ℝ, B.2 = B.1 + t ∧ A.2 = A.1 + t ∧ t ≠ 0)
(sym_points : D.1 = -B.1 ∧ D.2 = -B.2)

-- Proof problem statement
theorem slope_AD (A B C D : ℝ × ℝ) (h : ParallelogramInscribed A B C D) : 
  ∃ k_2 : ℝ, k_2 = -1/2 :=
sorry

end slope_AD_l65_65393


namespace angle_ADB_correct_l65_65184

noncomputable def measure_angle_ADB : real :=
  135
  
theorem angle_ADB_correct {A B C D : Type} [right_triangle A B C]
  (hA : angle A = 45) (hB : angle B = 45)
  (hD : D = angle_bisector_intersection A B) : 
  measure_angle (angle A D B) = 135 :=
sorry

end angle_ADB_correct_l65_65184


namespace pascal_third_number_l65_65690

theorem pascal_third_number (n : ℕ) (hn : n = 100) : nat.choose n 2 = 4950 :=
by
  rw hn
  simp
  sorry

end pascal_third_number_l65_65690


namespace decryption_proof_l65_65619

def encryption_table : List ℕ := 
  [15, 10, 13, 6, 14, 1, 3, 12, 9, 8, 16, 4, 5, 7, 11, 2]

def encrypted_message : String := "тинаийпмтногмееокбпоучвлнлшеюуао"

def original_last_word : String := "палочку"

theorem decryption_proof :
  (apply_encryption encryption_table encrypted_message 2014).last_part = original_last_word :=
by
  sorry

end decryption_proof_l65_65619


namespace common_factor_l65_65351

theorem common_factor (x y : ℝ) : 
  ∃ c : ℝ, c * (3 * x * y^2 - 4 * x^2 * y) = 6 * x^2 * y - 8 * x * y^2 ∧ c = 2 * x * y := 
by 
  sorry

end common_factor_l65_65351


namespace sum_of_extreme_values_of_a_l65_65932

theorem sum_of_extreme_values_of_a (a b c : ℝ) (h₁ : a + b + c = 5) (h₂ : a^2 + b^2 + c^2 = 8) :
  let min_a := 1 in
  let max_a := 3 in
  min_a + max_a = 4 :=
by
  -- Proof steps go here
  sorry

end sum_of_extreme_values_of_a_l65_65932


namespace minimum_value_reached_at_x_7_l65_65346

noncomputable def quadratic_function : ℝ → ℝ := λ x, x^2 - 14 * x + 40

theorem minimum_value_reached_at_x_7 :
  quadratic_function 7 = -9 :=
by
  sorry

end minimum_value_reached_at_x_7_l65_65346


namespace tan_alpha_eq_neg_4_div_3_l65_65528

theorem tan_alpha_eq_neg_4_div_3 
  (α : ℝ)
  (ha : Vector2D ℝ (5, -3))
  (hb : Vector2D ℝ (9, -6 - Real.cos α))
  (hcond : Vector2D ℝ (1, Real.cos α) = 2 • ha - hb)
  (hparallel : 2 • ha - hb = k • ha)
  (h_quad : π / 2 < α ∧ α < π) :
  Real.tan α = -4 / 3 := by
  sorry

end tan_alpha_eq_neg_4_div_3_l65_65528


namespace find_y_l65_65132

def a : (ℝ × ℝ × ℝ) := (1, 2, 6)
def b (y : ℝ) : (ℝ × ℝ × ℝ) := (2, y, -1)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3
def orthogonal (u v : ℝ × ℝ × ℝ) : Prop := dot_product u v = 0

theorem find_y (y : ℝ) : orthogonal a (b y) → y = 2 := by
  sorry

end find_y_l65_65132


namespace min_value_fraction_subtraction_l65_65498

theorem min_value_fraction_subtraction
  (a b : ℝ)
  (ha : 0 < a ∧ a ≤ 3 / 4)
  (hb : 0 < b ∧ b ≤ 3 - a)
  (hineq : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → a * x + b - 3 ≤ 0) :
  ∃ a b, (0 < a ∧ a ≤ 3 / 4) ∧ (0 < b ∧ b ≤ 3 - a) ∧ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → a * x + b - 3 ≤ 0) ∧ (1 / a - b = 1) :=
by 
  sorry

end min_value_fraction_subtraction_l65_65498


namespace handrail_length_correct_l65_65013

/-- Define the conditions of the spiral staircase problem -/
def radius : ℝ := 3
def rise : ℝ := 12
def turn_in_radians : ℝ := 2 * Real.pi -- Equivalent to 360 degrees in radians

/-- Compute circumference with the given radius -/
def circumference : ℝ := 2 * Real.pi * radius

/-- Compute length of the handrail using Pythagorean theorem -/
noncomputable def handrail_length : ℝ := Real.sqrt (rise^2 + circumference^2)

/-- State the final proof goal -/
theorem handrail_length_correct : Real.round (handrail_length * 10) / 10 = 22.3 := by
  simp only [radius, rise, turn_in_radians, circumference, handrail_length]
  sorry

end handrail_length_correct_l65_65013


namespace average_ABC_is_3_l65_65844

theorem average_ABC_is_3
  (A B C : ℝ)
  (h1 : 2003 * C - 4004 * A = 8008)
  (h2 : 2003 * B + 6006 * A = 10010)
  (h3 : B = 2 * A - 6) : 
  (A + B + C) / 3 = 3 :=
sorry

end average_ABC_is_3_l65_65844


namespace interval_of_monotonic_decrease_maximum_value_of_k_l65_65516

-- Given conditions and definitions.
def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 - 3 * x
def f' (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Conditions: local minimum at x = 3, f(3) = -9
def local_minimum_at_3 : Prop := f 3 = -9
def derivative_at_3 : Prop := f' 3 = 0

-- Questions transformed into Lean statements
theorem interval_of_monotonic_decrease : local_minimum_at_3 ∧ derivative_at_3 → ∀ x : ℝ, f'(x) < 0 ↔ -1 < x ∧ x < 3 :=
by sorry

theorem maximum_value_of_k (k : ℕ) : ∀ x : ℝ, x > 0 → f'(x) > k * (x * real.log x - 1) - 6 * x - 4 → k ≤ 6 :=
by sorry

end interval_of_monotonic_decrease_maximum_value_of_k_l65_65516


namespace definite_integral_abs_poly_l65_65048

theorem definite_integral_abs_poly :
  ∫ x in (-2 : ℝ)..(2 : ℝ), |x^2 - 2*x| = 8 :=
by
  sorry

end definite_integral_abs_poly_l65_65048


namespace f_inequality_l65_65035

noncomputable def f : ℝ → ℝ := sorry

theorem f_inequality
  (h1 : ∀ x, 0 < x ∧ x < π / 2 → has_deriv_at f (derivative (derivative f)) x)
  (h2 : ∀ x, 0 < x ∧ x < π / 2 → f x * tan x + derivative (derivative f) x < 0) :
  sqrt 3 * f (π / 3) < f (π / 6) := 
sorry

end f_inequality_l65_65035


namespace arrangement_count_l65_65900

theorem arrangement_count (s : Finset ℕ) (h1 : s = {1, 2, 3, 4, 5, 6}) :
  ∃ f : ℕ → ℕ, 
  (∀ (i : ℕ), i ∈ Finset.range 4 → f i ∈ s) ∧
  (∀ (i : ℕ), i ∈ Finset.range 4 → ((f i) * (f (i + 2)) - (f (i + 1))^2) % 7 = 0) ∧
  (Finset.card (Finset.image f (Finset.range 6))) = 12 :=
sorry

end arrangement_count_l65_65900


namespace tangent_line_equation_tangent_line_at_point_1_2_l65_65992

noncomputable def curve (x : ℝ) : ℝ := -x^3 + 3 * x^2

def point : ℝ × ℝ := (1, 2)

def derivative_curve (x : ℝ) : ℝ := -3 * x^2 + 6 * x

theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) = point → ∃ m b : ℝ, y = m * x + b ∧ m = derivative_curve 1 ∧ b = 2 - m :=
begin
  intros x y h,
  existsi 3,
  existsi -1,
  simp [point, h],
  split,
  { linarith },
  split,
  { unfold derivative_curve, linarith },
  { unfold derivative_curve, linarith }
end

theorem tangent_line_at_point_1_2 :
  ∃ m b : ℝ, (curve 1 - 2 = m * (1 - 1) → curve 1 = m * 1 + b) :=
begin
  existsi 3,
  existsi -1,
  linarith
end

end tangent_line_equation_tangent_line_at_point_1_2_l65_65992


namespace cards_average_2021_l65_65000

theorem cards_average_2021 (n : ℕ) (h1 : (∑ i in Finset.range (n+1), i) = n * (n + 1) / 2)
                           (h2 : (∑ i in Finset.range (n+1), i^2) = n * (n + 1) * (2 * n + 1) / 6)
                           (h3 : (n * (n + 1) * (2 * n + 1) / 6) / (n * (n + 1) / 2) = 2021) :
  n = 3031 :=
by
  sorry

end cards_average_2021_l65_65000


namespace number_of_elements_unchanged_l65_65648

theorem number_of_elements_unchanged (n : ℕ) (xs : Fin n → ℝ)
  (h_avg : (∑ i, xs i) / n = 7)
  (h_mult_avg : (∑ i, 10 * xs i) / n = 70) :
  n = n :=
by
  sorry

end number_of_elements_unchanged_l65_65648


namespace probability_satisfies_condition_l65_65240

open Nat

def S : Finset ℕ := (finset.range (6001)).filter (λ x, 6000 % x = 0)

def satisfies_condition (a b c d : ℕ) : Prop := 
  lcm (gcd a b) (gcd c d) = gcd (lcm a b) (lcm c d)

theorem probability_satisfies_condition :
  (Finset.card (S.product (S.product (S.product S))).filter 
  (λ quadruple, satisfies_condition quadruple.1 quadruple.2.1 quadruple.2.2.1 quadruple.2.2.2)).to_real / 
  (S.card ^ 4).to_real = 41 / 512 := 
sorry

end probability_satisfies_condition_l65_65240


namespace compare_power_sizes_l65_65419

theorem compare_power_sizes : 2^100 < 3^75 := 
by presumably {
  sorry
}

end compare_power_sizes_l65_65419


namespace shaded_area_of_grid_l65_65767

theorem shaded_area_of_grid : 
  let total_area := 27 in
  let unshaded_triangle_1_area := 22.5 in
  let unshaded_triangle_2_area := 2.5 in
  let total_unshaded_area := unshaded_triangle_1_area + unshaded_triangle_2_area in
  total_area - total_unshaded_area = 2 :=
by
  let total_area := 27
  let unshaded_triangle_1_area := 22.5
  let unshaded_triangle_2_area := 2.5
  let total_unshaded_area := unshaded_triangle_1_area + unshaded_triangle_2_area
  show total_area - total_unshaded_area = 2
  sorry

end shaded_area_of_grid_l65_65767


namespace radius_of_fifth_sphere_l65_65166

noncomputable def cone_height : ℝ := 7
noncomputable def base_radius : ℝ := 7

def identical_spheres (r : ℝ) : Prop := 
  r > 0 ∧
  ∀ (O₁ O₂ O₃ O₄ O₅ : ℝ×ℝ×ℝ),
  let r₁ : ℝ := (O₁.1*O₁.1 + O₁.2*O₁.2 + O₁.3*O₁.3)^(1/2),
      r₂ : ℝ := (O₂.1*O₂.1 + O₂.2*O₂.2 + O₂.3*O₂.3)^(1/2),
      r₃ : ℝ := (O₃.1*O₃.1 + O₃.2*O₃.2 + O₃.3*O₃.3)^(1/2),
      r₄ : ℝ := (O₄.1*O₄.1 + O₄.2*O₄.2 + O₄.3*O₄.3)^(1/2),
      r₅ : ℝ := (O₅.1*O₅.1 + O₅.2*O₅.2 + O₅.3*O₅.3)^(1/2) in
  r₁ = r ∧ r₂ = r ∧ r₃ = r ∧ r₄ = r ∧ r₅ = 2 * r * (2^(1/2) - 1/2) + r 

-- Function to check the radius of the fifth sphere
theorem radius_of_fifth_sphere : 
  ∀ (x r : ℝ), cone_height = 7 ∧ base_radius = 7 ∧ identical_spheres r → 
  x = 2 * sqrt(2) - 1 :=
sorry

end radius_of_fifth_sphere_l65_65166


namespace polar_line_equation_l65_65799

theorem polar_line_equation (ρ θ: ℝ) (h1: ρ = √2) (h2: θ = π / 4):
  ∃ ρ θ, (ρ * sin θ = 1) ∧ (ρ = √2) ∧ (θ = π / 4) := 
by 
  sorry

end polar_line_equation_l65_65799


namespace b2_b7_product_l65_65230

variable {b : ℕ → ℤ}

-- Define the conditions: b is an arithmetic sequence and b_4 * b_5 = 15
def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

axiom increasing_arithmetic_sequence : is_arithmetic_sequence b
axiom b4_b5_product : b 4 * b 5 = 15

-- The target theorem to prove
theorem b2_b7_product : b 2 * b 7 = -9 :=
sorry

end b2_b7_product_l65_65230


namespace principal_sum_correct_l65_65305

noncomputable def principal_sum (CI SI : ℝ) (t : ℕ) : ℝ :=
  let P := ((SI * t) / t) in
  let x := 5100 / P in
  26010000 / ((CI - SI) / t)

theorem principal_sum_correct :
  principal_sum 11730 10200 2 ≈ 16993.46 :=
by
  simp only [principal_sum]
  sorry

end principal_sum_correct_l65_65305


namespace probability_of_union_l65_65636

theorem probability_of_union :
  let Ω := {1, 2, 3, 4, 5, 6}
  let A := {3}
  let B := {2, 4, 6}
  let P := λ s, (s.card / Ω.card : ℚ)
  P (A ∪ B) = 2 / 3 :=
by
  let Ω := {1, 2, 3, 4, 5, 6}
  let A := {3}
  let B := {2, 4, 6}
  let P := λ s: Ω, (s.card / Ω.card : ℚ)
  have P_A : P A = 1 / 6 := by sorry
  have P_B : P B = 1 / 2 := by sorry
  have A_B_disjoint : A ∩ B = ∅ := by sorry
  have P_A_union_B : P (A ∪ B) = P A + P B := by sorry
  show P (A ∪ B) = 2 / 3 from
    calc P (A ∪ B) = P A + P B : P_A_union_B
                 ... = 1 / 6 + 1 / 2 : by rw [P_A, P_B]
                 ... = 2 / 3 : by sorry

end probability_of_union_l65_65636


namespace angle_EDF_eq_60_l65_65190

noncomputable def isosceles_triangle (A B C : Type) [MetricSpace A] (AB AC : ℝ) :=
  AB = AC 

noncomputable def point_on_side (A B C : Type) [MetricSpace A] (D E F : A) :=
  D ∈ segment B C ∧ E ∈ segment A C ∧ F ∈ segment A B
  
noncomputable def equilateral_triangle (D E F : Type) [MetricSpace D] (DE EF FD : ℝ) :=
  DE = EF ∧ EF = FD

theorem angle_EDF_eq_60 (A B C D E F : Type) [MetricSpace A] 
  (h1 : isosceles_triangle A B C (dist A B) (dist A C))
  (h2 : ∠A = 100°)
  (h3 : point_on_side A B C D E F)
  (h4 : equilateral_triangle D E F (dist D E) (dist E F) (dist F D))
  : ∠D E F = 60° :=
sorry

end angle_EDF_eq_60_l65_65190


namespace paintable_area_l65_65251

theorem paintable_area (n : ℕ) (length width height window_area : ℕ) 
  (total_bedrooms : n = 4) 
  (length_bedroom : length = 15) 
  (width_bedroom : width = 12) 
  (height_bedroom : height = 9) 
  (door_window_area : window_area = 80) :
  let wall_area := 2 * (length * height) + 2 * (width * height),
      paintable_area := wall_area - window_area
  in (paintable_area * n) = 1624 := 
by {
  sorry 
}

end paintable_area_l65_65251


namespace bob_pennies_l65_65155

theorem bob_pennies (a b : ℕ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  have h3 : 4 * a - b = 5, from sorry,
  have h4 : b - 3 * a = 4, from sorry,
  have h5 : 4 * a - 3 * a = 9, from sorry,
  have h6 : a = 9, from sorry,
  have h7 : b + 1 = 36 - 4, from sorry,
  have h8 : b + 1 = 32, from sorry,
  have h9 : b = 31, from sorry,
  exact h9

end bob_pennies_l65_65155


namespace arrangement_count_l65_65902

theorem arrangement_count (s : Finset ℕ) (h1 : s = {1, 2, 3, 4, 5, 6}) :
  ∃ f : ℕ → ℕ, 
  (∀ (i : ℕ), i ∈ Finset.range 4 → f i ∈ s) ∧
  (∀ (i : ℕ), i ∈ Finset.range 4 → ((f i) * (f (i + 2)) - (f (i + 1))^2) % 7 = 0) ∧
  (Finset.card (Finset.image f (Finset.range 6))) = 12 :=
sorry

end arrangement_count_l65_65902


namespace ratio_of_rectangles_l65_65307

theorem ratio_of_rectangles (p q : ℝ) (h1 : q ≠ 0) 
    (h2 : q^2 = 1/4 * (2 * p * q  - q^2)) : p / q = 5 / 2 := 
sorry

end ratio_of_rectangles_l65_65307


namespace segments_form_triangle_l65_65357

theorem segments_form_triangle :
  ∀ (a b c : ℝ), a = 8 ∧ b = 8 ∧ c = 15 → 
    (a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  intros a b c h
  rw [← h.1, ← h.2.1, ← h.2.2]
  split
  apply lt_add_of_pos_of_le
  linarith
  linarith
  split
  apply lt_add_of_pos_of_le
  linarith
  linarith
  apply lt_add_of_pos_of_le
  linarith
  linarith

end segments_form_triangle_l65_65357


namespace bond_selling_price_l65_65591

theorem bond_selling_price
    (face_value : ℝ)
    (interest_rate_face : ℝ)
    (interest_rate_selling : ℝ)
    (interest : ℝ)
    (selling_price : ℝ)
    (h1 : face_value = 5000)
    (h2 : interest_rate_face = 0.07)
    (h3 : interest_rate_selling = 0.065)
    (h4 : interest = face_value * interest_rate_face)
    (h5 : interest = selling_price * interest_rate_selling) :
  selling_price = 5384.62 :=
sorry

end bond_selling_price_l65_65591


namespace part_a_part_b_general_question_l65_65128

-- Define the automates as functions
def first_automate (a b : ℕ) : ℕ × ℕ := (a + 1, b + 1)

def second_automate : (ℕ × ℕ) → option (ℕ × ℕ)
| (a, b) := if a % 2 = 0 ∧ b % 2 = 0 then some (a / 2, b / 2) else none

def third_automate (a b c : ℕ) : ℕ × ℕ := (a, c)

-- Initial condition
def initial_card : ℕ × ℕ := (5, 19)

-- Part (a): Prove it is possible to obtain (1, 50)
theorem part_a : ∃ (f : ℕ × ℕ → ℕ × ℕ), f initial_card = (1, 50) := sorry

-- Part (b): Prove it is not possible to obtain (1, 100)
theorem part_b : ¬ ∃ (f : ℕ × ℕ → ℕ × ℕ), f initial_card = (1, 100) := sorry

-- General question: Prove the conditions under which (1, n) can be obtained
theorem general_question (a b : ℕ) (h : a < b) (n : ℕ) : 
  ∃ (f : ℕ × ℕ → ℕ × ℕ), f (a, b) = (1, n) ↔ ∃ (d : ℕ), d ∣ (a - b) ∧ d ∣ (1 - n) := sorry

end part_a_part_b_general_question_l65_65128


namespace cube_edge_length_l65_65495

theorem cube_edge_length (V : ℝ) (a : ℝ)
  (hV : V = (4 / 3) * Real.pi * (Real.sqrt 3 * a / 2) ^ 3)
  (hVolume : V = (9 * Real.pi) / 2) :
  a = Real.sqrt 3 :=
by
  sorry

end cube_edge_length_l65_65495


namespace find_x_l65_65550

-- Define the vectors
def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, 4)

-- Define the dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- The theorem we want to prove
theorem find_x (x : ℝ) : dot_product a (b x) = 0 → x = 2 :=
by {
  -- Placeholder for the proof
  sorry
}

end find_x_l65_65550


namespace problem_statement_l65_65106

variable (f : ℕ → ℝ)

theorem problem_statement (hf : ∀ k : ℕ, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)
  (h : f 4 = 25) : ∀ k : ℕ, k ≥ 4 → f k ≥ k^2 := 
by
  sorry

end problem_statement_l65_65106


namespace mutually_exclusive_events_l65_65295

-- Types for Students and Balls
inductive Student | A | B | C | D | E
inductive Ball | Red | Blue | Green | Yellow | Orange

-- Function representing the distribution of balls to students
def distribution (s : Student) : Ball

-- Definitions based on the problem conditions
def receives_red_ball (s : Student) : Prop := distribution s = Ball.Red
def mutually_exclusive (P Q : Prop) : Prop := P → ¬Q

-- Proving the events are mutually exclusive but not contradictory
theorem mutually_exclusive_events :
  (mutually_exclusive (receives_red_ball Student.A) (receives_red_ball Student.B)) :=
by
  sorry

end mutually_exclusive_events_l65_65295


namespace find_a_perpendicular_lines_l65_65130

theorem find_a_perpendicular_lines (a : ℝ) :
  (∀ (x y : ℝ),
    a * x + 2 * y + 6 = 0 → 
    x + (a - 1) * y + a^2 - 1 = 0 → (a * 1 + 2 * (a - 1) = 0)) → 
  a = 2/3 :=
by
  intros h
  sorry

end find_a_perpendicular_lines_l65_65130


namespace find_BP_l65_65714

-- Definitions
variables {A B C D P : Type} [Point A] [Point B] [Point C] [Point D] [Point P]
variable (circle : Circle A B C D) -- A, B, C, D on a circle
variable (intersect_AC_BD : Intersect (A, C) (B, D) P) -- Line AC and BD intersect at P
variable (AP : Length A P = 6)
variable (PC : Length P C = 2)
variable (BD : Length B D = 9)
variable (BP_gt_DP : Length B P > Length P D)

-- Equivalent proof statement
theorem find_BP : Length B P = 6 :=
by
  sorry

end find_BP_l65_65714


namespace C1_cartesian_eq_C2_cartesian_eq_min_distance_PQ_l65_65425

-- Define the parametric equations for C1
def parametric_C1 (t : ℝ) : ℝ × ℝ :=
  (-2 + 2 * Real.cos t, 1 + 2 * Real.sin t)

-- Define the polar equation for C2
def polar_C2 (ρ θ : ℝ) : Prop :=
  4 * ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0

-- Prove that the Cartesian equation of C1 is (x + 2)² + (y - 1)² = 4
theorem C1_cartesian_eq :
  ∃ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 4 ↔ 
    ∃ t : ℝ, (x, y) = parametric_C1 t := sorry

-- Prove that the Cartesian equation of C2 is 4x - y + 1 = 0
theorem C2_cartesian_eq :
  ∃ x y : ℝ, 4 * x - y + 1 = 0 ↔ 
    ∃ ρ θ : ℝ, (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ 
               polar_C2 ρ θ := sorry

-- Prove the minimum distance |PQ| between a point P on C1 and a point Q on C2 is 0
theorem min_distance_PQ :
  ∀ P ∈ (λ t : ℝ, parametric_C1 t),
  ∀ Q ∈ (λ p : ℝ × ℝ, (∃ ρ θ : ℝ, p = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ polar_C2 ρ θ)),
  min (dist P Q) = 0 := sorry

end C1_cartesian_eq_C2_cartesian_eq_min_distance_PQ_l65_65425


namespace odd_and_symmetric_f_l65_65694

open Real

noncomputable def f (A ϕ : ℝ) (x : ℝ) := A * sin (x + ϕ)

theorem odd_and_symmetric_f (A ϕ : ℝ) (hA : A > 0) (hmin : f A ϕ (π / 4) = -1) : 
  ∃ g : ℝ → ℝ, g x = -A * sin x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, g (π / 2 - x) = g (π / 2 + x)) :=
sorry

end odd_and_symmetric_f_l65_65694


namespace find_average_l65_65605

theorem find_average (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : log y x * log x y = 4) (h4 : x * y = 64) : (x + y) / 2 = 10 := 
  sorry

end find_average_l65_65605


namespace volume_correct_surface_area_cube_correct_surface_area_sphere_correct_l65_65881

/- Define the edge length -/
def edge_length : ℝ := 2 * Real.sqrt 2

/- Define volume of the cube -/
def volume_of_cube : ℝ := edge_length ^ 3

/- Define surface area of the cube -/
def surface_area_of_cube : ℝ := 6 * edge_length ^ 2

/- Define surface area of the inscribed sphere -/
def radius_of_inscribed_sphere : ℝ := edge_length / 2
def surface_area_of_inscribed_sphere : ℝ := 4 * Real.pi * (radius_of_inscribed_sphere ^ 2)

/- Proofs for the required statements -/
theorem volume_correct : volume_of_cube = 16 * Real.sqrt 2 := by sorry
theorem surface_area_cube_correct : surface_area_of_cube = 48 := by sorry
theorem surface_area_sphere_correct : surface_area_of_inscribed_sphere = 8 * Real.pi := by sorry

end volume_correct_surface_area_cube_correct_surface_area_sphere_correct_l65_65881


namespace intersection_of_sets_l65_65125

def setA (x : ℝ) : Prop := x^2 - 4 * x - 5 > 0

def setB (x : ℝ) : Prop := 4 - x^2 > 0

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end intersection_of_sets_l65_65125


namespace num_possible_values_of_s_l65_65700

theorem num_possible_values_of_s (n p q r s : ℕ) (P : 100 < p ∧ p < q ∧ q < r ∧ r < s) 
  (at_least_three_consecutive : ∃ a b c, a + 1 = b ∧ b + 1 = c ∧ (p = a ∨ p = b ∨ p = c ∨ 
  q = a ∨ q = b ∨ q = c ∨ r = a ∨ r = b ∨ r = c ∨ s = a ∨ s = b ∨ s = c))
  (avg_remaining : (Finset.range (n+1)).sum (λ x, x) - (p + q + r + s) = 89.5625 * (n - 4)) :
  ∃! k, k = 22 := sorry

end num_possible_values_of_s_l65_65700


namespace triangle_perimeter_l65_65884

theorem triangle_perimeter
  (a b : ℝ)
  (x := 6)
  (D E F : Type*)
  [metric_space D] [metric_space E] [metric_space F]
  (DE : ℝ := 10)
  (EF : ℝ := x)
  (DF : ℝ := real.sqrt 64) -- √64 because x = 6 and 10^2 - 6^2 = 64
  (V U W X : D)
  (h_triangle : ∠DEF = 90)
  (h_rect1 : rect DEUV)
  (h_rect2 : rect EFWX)
  (h_cyclic : ∀ (U V W X : D), cyclic_quadrilateral U V W X) :
  (DE + EF + DF) = 10 + 6 + 8 := by sorry

end triangle_perimeter_l65_65884


namespace gold_distribution_l65_65565

/-- 
There are ten ranked individuals, the highest-ranking official among ten people being awarded gold by their rank,
decreasing accordingly. The top three individuals received four jin of gold each;
the last four individuals received three jin of gold each; the three individuals in the middle,
will receive gold according to their ranks. 
The total gold the three individuals who have not arrived yet should receive is 83/26 jin.
-/
theorem gold_distribution 
  (a : ℕ → ℚ) (d : ℚ) 
  (h_seq : ∀ n, a (n+1) = a n + d)  
  (h_top : ∑ i in finset.range 3, a (8 + i) = 4) 
  (h_low : ∑ i in finset.range 4, a (4 + i) = 3):
  ∑ i in finset.range 3, a (5 + i) = 83 / 26 := 
sorry

end gold_distribution_l65_65565


namespace chess_tournament_points_l65_65164

/--
In a chess tournament with 29 participants, where:
  - A win in a match gives 1 point,
  - A draw gives 0.5 points,
  - A loss gives 0 points,
  - A player with no opponent receives 1 point (max once per tournament),
  - Players with the same points play against each other in each round,
  - 9 total rounds are played,
  - No draws recorded,

Prove that it is possible for two players to each have 8 points before the final round.
-/
theorem chess_tournament_points (players playing 9 rounds : ℕ)
  (points : ℕ → ℕ → ℝ)
  (play : (ℕ → ℕ) → Prop)
  (no_draws : Prop)
  (final_round : Prop) :
  players = 29 →
  (∀ (w l : ℕ), w ≠ l → points w l = 1 ∧ points l w = 0) →
  (∀ (p : ℕ), (p != 0) → points p 0 = 0.5) →
  (∀ (p : ℕ), points p p = 1) →
  (∀ (p : ℕ), points p p ≤ 1) →
  play (λ p, points 0 p + 1) →
  (∃ x y, x ≠ y ∧ points x 8 = 8 ∧ points y 8 = 8) →
  (final_round → no_draws) →
  (x, y ∈ (play final_round) → (points x 9 = 8 ∧ points y 9 = 8)) → sorry

end chess_tournament_points_l65_65164


namespace simplify_log_expression_l65_65289

theorem simplify_log_expression :
  (\lg 2) ^ 2 + \lg 2 * \lg 5 + \lg 5 = 1 := sorry

end simplify_log_expression_l65_65289


namespace simplify_and_evaluate_expression_l65_65643

theorem simplify_and_evaluate_expression :
  let a := 2 * Real.sin (Real.pi / 3) + 3
  (a + 1) / (a - 3) - (a - 3) / (a + 2) / ((a^2 - 6 * a + 9) / (a^2 - 4)) = Real.sqrt 3 := by
  sorry

end simplify_and_evaluate_expression_l65_65643


namespace smallest_value_of_expression_l65_65059

variable (a b c : ℝ)
variable (hab : a > b)
variable (hbc : b > c)
variable (ha_nonzero : a ≠ 0)

theorem smallest_value_of_expression :
  ∃ (x : ℝ), x = 6 ∧ 
  (∀ a b c : ℝ, a > b → b > c → a ≠ 0 →
  (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 ≥ x) :=
begin
  sorry
end

end smallest_value_of_expression_l65_65059


namespace polynomials_equality_l65_65627

noncomputable def P (x : ℝ) : ℝ := sorry
noncomputable def Q (x : ℝ) : ℝ := sorry

theorem polynomials_equality 
  (degP : ∃ n : ℕ, P(x) and P is of degree n ∧ n > 0)
  (degQ : ∃ m : ℕ, Q(x) and Q is of degree m ∧ m > 0)
  (h1 : ∀ x : ℝ, P(P(x)) = Q(Q(x)))
  (h2 : ∀ x : ℝ, P(P(P(x))) = Q(Q(Q(x)))) :
  ∀ x : ℝ, P(x) = Q(x) := 
sorry

end polynomials_equality_l65_65627


namespace sum_a_n_l65_65513

def f (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

noncomputable def sequence (a : ℕ → ℝ) : ℕ → ℝ
| 0 := 1
| (n + 1) := sequence (n % 3)

theorem sum_a_n (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 3) = a n)
  (h3 : f (a 1) + f (a 2 + a 3) = 0) :
  (Finset.range 2023).sum a = 1 :=
sorry

end sum_a_n_l65_65513


namespace bob_pennies_l65_65150

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l65_65150


namespace intersection_of_A_and_B_l65_65859

def setA : Set ℤ := {x | abs x < 4}
def setB : Set ℤ := {x | x - 1 ≥ 0}
def setIntersection : Set ℤ := {1, 2, 3}

theorem intersection_of_A_and_B : setA ∩ setB = setIntersection :=
by
  sorry

end intersection_of_A_and_B_l65_65859


namespace supplies_left_after_purchase_and_loss_l65_65587

theorem supplies_left_after_purchase_and_loss :
  ∀ (students : ℕ) (construction_paper_needed_per_student : ℕ) (bottles_of_glue : ℕ) (fraction_lost : ℚ) (additional_paper : ℕ),
  students = 15 →
  construction_paper_needed_per_student = 5 →
  bottles_of_glue = 10 →
  fraction_lost = 2/3 →
  additional_paper = 15 →
  let total_supplies := (students * construction_paper_needed_per_student) + bottles_of_glue in
  let supplies_lost := (fraction_lost * total_supplies).to_nat in
  let supplies_left_after_loss := total_supplies - supplies_lost in
  supplies_left_after_loss + additional_paper = 44 :=
by
  intros students construction_paper_needed_per_student bottles_of_glue fraction_lost additional_paper
  intros h_students h_construction_paper_needed_per_student h_bottles_of_glue h_fraction_lost h_additional_paper
  simp only
  sorry

end supplies_left_after_purchase_and_loss_l65_65587


namespace division_expression_is_7_l65_65270

noncomputable def evaluate_expression : ℝ :=
  1 / 2 / 3 / 4 / 5 / (6 / 7 / 8 / 9 / 10)

theorem division_expression_is_7 : evaluate_expression = 7 :=
by
  sorry

end division_expression_is_7_l65_65270


namespace eggs_volume_correct_l65_65248

def raw_spinach_volume : ℕ := 40
def cooking_reduction_ratio : ℚ := 0.20
def cream_cheese_volume : ℕ := 6
def total_quiche_volume : ℕ := 18
def cooked_spinach_volume := (raw_spinach_volume : ℚ) * cooking_reduction_ratio
def combined_spinach_and_cream_cheese_volume := cooked_spinach_volume + (cream_cheese_volume : ℚ)
def eggs_volume := (total_quiche_volume : ℚ) - combined_spinach_and_cream_cheese_volume

theorem eggs_volume_correct : eggs_volume = 4 := by
  sorry

end eggs_volume_correct_l65_65248


namespace expectation_X_variance_X_expectation_p_X_l65_65939

variables (α β : ℝ) (hα : 0 < α) (hβ : 0 < β)

def beta_pdf (x : ℝ) (α β : ℝ) : ℝ :=
  if (0 < x) ∧ (x < 1) then x^(α-1) * (1-x)^(β-1) / real.betaFunc α β else 0

theorem expectation_X :
  ∀ X:𝓟(MeasureTheory.MeasureSpace.measure_space ℝ),
  (∀ x, X(x) = beta_pdf x α β) →
  MeasureTheory.Integral.integral X (λ x, x) = α / (α + β) := sorry

theorem variance_X :
  ∀ X:𝓟(MeasureTheory.MeasureSpace.measure_space ℝ),
  (∀ x, X(x) = beta_pdf x α β) →
  (MeasureTheory.Integral.integral X (λ x, x^2) - (MeasureTheory.Integral.integral X (λ x, x))^2) = α * β / ((α + β + 1) * (α + β)^2) := sorry

theorem expectation_p_X (p : ℝ) (hp : p > -α):
  ∀ X:𝓟(MeasureTheory.MeasureSpace.measure_space ℝ),
  (∀ x, X(x) = beta_pdf x α β) →
  MeasureTheory.Integral.integral X (λ x, x^p) = real.betaFunc (α + p) β / real.betaFunc α β := sorry

end expectation_X_variance_X_expectation_p_X_l65_65939


namespace count_values_not_dividing_g_l65_65936

-- Define the function g(n) as the product of the proper positive integer divisors of n.
def product_of_proper_divisors (n : ℕ) : ℕ :=
  if h : n > 1 then
    (Finset.filter (λ d, d > 1 ∧ d < n ∧ n % d = 0) (Finset.range n)).prod id
  else
    1

-- Define the problem of counting values of n in the range [2,100] where n does not divide g(n)
theorem count_values_not_dividing_g :
  let g := product_of_proper_divisors in
  (Finset.filter (λ n, n ≤ 100 ∧ 2 ≤ n ∧ ¬ (n ∣ g n)) (Finset.range 101)).card = 30 :=
sorry

end count_values_not_dividing_g_l65_65936


namespace min_sum_distances_l65_65189

theorem min_sum_distances (a h : ℝ) (AB AC : ℝ)
  (h1 : AB = 2 * a) (h2 : AC = h) (h3 : is_right_angled_parallelogram ABCD)
  (EF_parallel_to_AC : EF ∥ AC) :
  ∃ M : Point, 
    (minimum
      (λ M, distance(A, M) + distance(M, B) + distance(M, F))
      ((M ∈ EF) ∧ divides_parallelogram_evenly(ABCD, M)) = h + a * sqrt(3)) :=
sorry

end min_sum_distances_l65_65189


namespace green_can_not_tell_truth_correct_leg_allocation_l65_65711

variable (Green Blue Red : ℕ)

-- Given conditions as functions
def always_lies (legs : ℕ) : Prop := legs = 7
def always_truth (legs : ℕ) : Prop := legs = 8
def green_statement := Green + Blue + Red = 21
def blue_statement : Prop := always_lies Green
def red_statement : Prop := always_lies Green ∧ always_lies Blue

-- Proof for the Green Octopus truth question
theorem green_can_not_tell_truth :
  green_statement → green_statement ∧ always_truth Green → False := by
  unfold green_statement always_truth
  sorry

-- Proof for the allocation of legs correctly
theorem correct_leg_allocation :
  Green = 7 ∧ Blue = 8 ∧ Red = 7 :=
begin
  sorry
end

end green_can_not_tell_truth_correct_leg_allocation_l65_65711


namespace length_RW_l65_65914

-- Define the conditions
variables (P Q R S T U V W: Type) 
variable [linear_ordered_field P]
variable [linear_ordered_field Q]
variable [linear_ordered_field R]
variable [linear_ordered_field S]
variable [linear_ordered_field T]
variable [linear_ordered_field U]
variable [linear_ordered_field V]
variable [linear_ordered_field W]

-- Lengths of sides
abbreviation length (a b : P) : Q := real.sqrt ((b - a) * (b - a))
variable (PQ STU VWX RW: Q)
hypothesis hPQ : PQ = 10
hypothesis hSTU : STU = 6
hypothesis hVWX : VWX = 4

-- Similar triangles due to parallel lines
hypothesis h_sim1 : similar P Q R S T U
hypothesis h_sim2 : similar P Q R V W X
hypothesis h_sim3 : similar S T U V W X

-- Proportional relationship in the triangles
hypothesis h_proportional : (length P Q / length V W) = (length Q R / RW)

-- Conclusion: length of RW
theorem length_RW : RW = 8 / 3 := by
  sorry

end length_RW_l65_65914


namespace cos_theta_minus_pi_six_l65_65086

theorem cos_theta_minus_pi_six (θ : ℝ) (h : Real.sin (θ + π / 3) = 2 / 3) : 
  Real.cos (θ - π / 6) = 2 / 3 :=
sorry

end cos_theta_minus_pi_six_l65_65086


namespace bear_pies_l65_65946

-- Lean definitions model:

variables (v_M v_B u_M u_B : ℝ)
variables (M_raspberries B_raspberries : ℝ)
variables (P_M P_B : ℝ)

-- Given conditions
axiom v_B_eq_6v_M : v_B = 6 * v_M
axiom u_B_eq_3u_M : u_B = 3 * u_M
axiom B_raspberries_eq_2M_raspberries : B_raspberries = 2 * M_raspberries
axiom P_sum : P_B + P_M = 60
axiom P_B_eq_9P_M : P_B = 9 * P_M

-- The theorem to prove
theorem bear_pies : P_B = 54 :=
sorry

end bear_pies_l65_65946


namespace seconds_in_part_of_day_l65_65866

theorem seconds_in_part_of_day : (1 / 4) * (1 / 6) * (1 / 8) * 24 * 60 * 60 = 450 := by
  sorry

end seconds_in_part_of_day_l65_65866


namespace kids_have_equal_eyes_l65_65201

theorem kids_have_equal_eyes (mom_eyes dad_eyes kids_num total_eyes kids_eyes : ℕ) 
  (h_mom_eyes : mom_eyes = 1) 
  (h_dad_eyes : dad_eyes = 3) 
  (h_kids_num : kids_num = 3) 
  (h_total_eyes : total_eyes = 16) 
  (h_family_eyes : mom_eyes + dad_eyes + kids_num * kids_eyes = total_eyes) :
  kids_eyes = 4 :=
by
  sorry

end kids_have_equal_eyes_l65_65201


namespace find_A_minus_C_l65_65670

/-- There are three different natural numbers A, B, and C. 
    When A + B = 84, B + C = 60, and A = 6B, find the value of A - C. -/
theorem find_A_minus_C (A B C : ℕ) 
  (h1 : A + B = 84) 
  (h2 : B + C = 60) 
  (h3 : A = 6 * B) 
  (h4 : A ≠ B) 
  (h5 : A ≠ C) 
  (h6 : B ≠ C) :
  A - C = 24 :=
sorry

end find_A_minus_C_l65_65670


namespace part1_part2_l65_65404

-- Define a cyclic quadrilateral and the related points and intersections in circle O.
variables {O : Type*} [incircle O] {A B C D E F H M G : O}

-- Conditions:
-- 1. ABCD is an inscribed quadrilateral in circle O.
axiom h1 : cyclic_quadrilateral A B C D
-- 2. BC and AD intersect at point F; AB and CD intersect at point E.
axiom h2a : intersecting_lines B C A D F
axiom h2b : intersecting_lines A B C D E
-- 3. The circumcircle of triangle ECF intersects circle O again at point H.
axiom h3 : second_intersection_circumcircle E C F O H
-- 4. Segment AH intersects EF at point M, and segment MC intersects circle O at point G.
axiom h4a : line_segment_intersection A H E F M
axiom h4b : line_segment_intersection M C O G

-- Proving that M is the midpoint of EF.
theorem part1 : midpoint M E F :=
by sorry

-- Proving that points A, G, E, and F are concyclic.
theorem part2 : concyclic_points A G E F :=
by sorry

end part1_part2_l65_65404


namespace convergence_equiv_subseq_a_s_l65_65941

section
variables {d : ℕ} -- Dimension d
variables {ξ : ℕ → ℝ^d} -- Random vectors in ℝ^d
variables {ξ₀ : ℝ^d} -- The vector to which we converge

def convergence_in_probability (ξ ξ₀ : ℕ → ℝ^d) :=
  ∀ ε > 0, ∀ δ > 0, ∃ N, ∀ n ≥ N, |ξₙ - ξ₀| < ε

theorem convergence_equiv_subseq_a_s (ξ : ℕ → ℝ^d) (ξ₀ : ℝ^d) :
  (convergence_in_probability ξ₀ ξ) ↔
  (∀ (n' : ℕ → ℕ) (hn' : ∀ i, n' i < n' (i+1)), 
    ∃ n'' : ℕ → ℕ, (∀ i, n'' i < n'' (i+1) ∧ n' (n'' i) < n' (n'' (i+1)) ∧ 
    (ξ (n' ∘ n'') i →ᵣ ξ₀))) :=
sorry
end

end convergence_equiv_subseq_a_s_l65_65941


namespace radius_of_wider_can_l65_65674

theorem radius_of_wider_can (h : ℝ) (x : ℝ) (π_ne_zero : π ≠ 0) (h_ne_zero : h ≠ 0) :
  (x^2 = 432) ↔ (x = 12 * real.sqrt 3) :=
by
  have := eq_of_smul_eq_smul (3 * h) : 12^2 * 3 * π = x^2 * h * π → (144 * 3 * h = x^2 * h) :=
    λ h : 144 * 3 * h = x^2 * h, div_eq_of_eq_mul' (pi_ne_zero.pi_ne_zero) h
  rw [mul_assoc, mul_comm 3, mul_assoc π, mul_comm h]
  simp [mul_eq_iff_eq' (3 * π * h_ne_zero) (real.sqrt_mul' 144 3).eq.symm]
  sorry

end radius_of_wider_can_l65_65674


namespace number_of_solutions_l65_65535

theorem number_of_solutions : ∃! (xy : ℕ × ℕ), (xy.1 ^ 2 - xy.2 ^ 2 = 91 ∧ xy.1 > 0 ∧ xy.2 > 0) := sorry

end number_of_solutions_l65_65535


namespace bob_pennies_l65_65151

theorem bob_pennies (a b : ℤ) (h1 : b + 1 = 4 * (a - 1)) (h2 : b - 1 = 3 * (a + 1)) : b = 31 :=
by
  sorry

end bob_pennies_l65_65151


namespace smallest_divisible_by_6_l65_65947

theorem smallest_divisible_by_6 : 
  ∃ n : ℕ, (50000 ≤ n ∧ n < 60000) ∧ 
  (∀ d, d ∈ [1, 2, 3, 7, 8] → d ∈ digits 10 n) ∧ 
  (∀ d, d ∈ digits 10 n → d ∈ [1, 2, 3, 7, 8]) ∧ 
  (n % 6 = 0) ∧ 
  (n = 13782) := 
sorry

end smallest_divisible_by_6_l65_65947


namespace distance_origin_to_line_l65_65652

theorem distance_origin_to_line : 
  let A := 1
  let B := Real.sqrt 3
  let C := -2
  let x1 := 0
  let y1 := 0
  let distance := |A*x1 + B*y1 + C| / Real.sqrt (A^2 + B^2)
  distance = 1 :=
by 
  let A := 1
  let B := Real.sqrt 3
  let C := -2
  let x1 := 0
  let y1 := 0
  let distance := |A*x1 + B*y1 + C| / Real.sqrt (A^2 + B^2)
  sorry

end distance_origin_to_line_l65_65652


namespace hexagon_area_of_equilateral_triangle_with_squares_l65_65956

theorem hexagon_area_of_equilateral_triangle_with_squares
  (a : ℝ) :
  ∃ (S : ℝ), S = a^2 * (3 + Real.sqrt 3) :=
by
  use a^2 * (3 + Real.sqrt 3)
  sorry

end hexagon_area_of_equilateral_triangle_with_squares_l65_65956


namespace shifted_function_is_equivalent_l65_65850

def f (x : ℝ) (φ : ℝ) := Real.sin ((π / 3) * x + φ)

theorem shifted_function_is_equivalent 
  (φ : ℝ) (h₁ : |φ| < π / 2) 
  (h₂ : ∀ x, f x φ = f (2 - x) φ) : 
  ∀ x, f (x - 3) φ = Real.sin ((π / 3) * x - 5 * π / 6) := 
by 
  -- the proof goes here
  sorry

end shifted_function_is_equivalent_l65_65850


namespace thirteenth_number_sum_12_l65_65399

def digits_sum_to_12 (n : ℕ) : Prop :=
  (n.digits 10).sum = 12

theorem thirteenth_number_sum_12 : ∃ n : ℕ, digits_sum_to_12 n ∧ (list.filter digits_sum_to_12 (list.range 1000)).nth 12 = some n := 
sorry

end thirteenth_number_sum_12_l65_65399


namespace range_of_f_l65_65321

-- Define the function
def f (x : ℝ) : ℝ := 2 * x / (x - 1)

-- Statement of the theorem about the range of the function
theorem range_of_f :
  { y : ℝ | ∃ x : ℝ, f x = y } = { y : ℝ | y ≠ 2 } :=
sorry

end range_of_f_l65_65321


namespace closest_to_613_div_0_307_l65_65434

def closest_to_fraction : ℝ := 3000

theorem closest_to_613_div_0_307 (x : ℝ) (h₁ : x = 613) (h₂ : x = 0.307) :
  ∃ y, y = closest_to_fraction :=
begin
  use closest_to_fraction,
  sorry
end

end closest_to_613_div_0_307_l65_65434


namespace binom_16_15_eq_16_l65_65765

theorem binom_16_15_eq_16 : binomial 16 15 = 16 := by
  sorry

end binom_16_15_eq_16_l65_65765


namespace factor_difference_of_squares_l65_65051

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_difference_of_squares_l65_65051


namespace power_expansion_l65_65347

theorem power_expansion (a b : ℕ) (h : 100 = 10^2) : 100^50 = 10^100 := 
by
  rw h
  sorry

end power_expansion_l65_65347


namespace color_conversion_l65_65950

noncomputable def final_state_possible (n : ℕ) : Prop :=
  ∃ (reach_final : (n % 2 = 0)), ∀ {c : ℕ}, 3 ≤ n → -- Only even n can reach a final state of uniform color.
  ∃ (initial_state : Fin n → ℕ), 
  (∀ i, initial_state i ∈ {1, 2}) → 
  ∃ (sequence_of_steps : list (Fin n)), 
  (forall step in sequence_of_steps, (∃ i, initial_state i = initial_state (i + 1) + 1 ∨ initial_state (i + 1) = initial_state i + 1)) ∧ 
  (∃ final_state : Fin n → ℕ, ∀ i, final_state i = c)

theorem color_conversion (n : ℕ) (h1 : n ≥ 3) : final_state_possible n ↔ n % 2 = 0 := 
by
  sorry

end color_conversion_l65_65950


namespace bart_trees_needed_l65_65787

noncomputable def calculate_trees (firewood_per_tree : ℕ) (logs_per_day : ℕ) (days_in_period : ℕ) : ℕ :=
  (days_in_period * logs_per_day) / firewood_per_tree

theorem bart_trees_needed :
  let firewood_per_tree := 75 in
  let logs_per_day := 5 in
  let days_in_november := 30 in
  let days_in_december := 31 in
  let days_in_january := 31 in
  let days_in_february := 28 in
  let total_days := days_in_november + days_in_december + days_in_january + days_in_february in
  calculate_trees firewood_per_tree logs_per_day total_days = 8 :=
by
  sorry

end bart_trees_needed_l65_65787


namespace fish_problem_l65_65757

theorem fish_problem : 
  ∀ (B T S : ℕ), 
    B = 10 → 
    T = 3 * B → 
    S = 35 → 
    B + T + S + 2 * S = 145 → 
    S - T = 5 :=
by sorry

end fish_problem_l65_65757


namespace all_permissible_triangles_present_l65_65097

theorem all_permissible_triangles_present (p : ℕ) (hp : Nat.Prime p) :
  ∃ (tr : Set (ℕ × ℕ × ℕ)), 
  (∀ i j k, i + j + k = p → (i, j, k) ∈ tr) ∧
  (∀ (tri : ℕ × ℕ × ℕ), tri ∈ tr → permissible_triangle tri p) ∧
  ∀ i j k, i + j + k = p → (∃ tri₁ tri₂ ∈ tr, (tri₁ ≠ tri₂ ∧ can_be_divided_into tri₁ tri₂ tri p)) := 
sorry

-- Definitions
def permissible_triangle (t : ℕ × ℕ × ℕ) (p : ℕ) : Prop :=
  let (i, j, k) := t
  i + j + k = p

def can_be_divided_into (tri₁ tri₂ : ℕ × ℕ × ℕ) (tri : ℕ × ℕ × ℕ) (p : ℕ) : Prop :=
  let (i, j, k) := tri
  let (a, b, c) := tri₁
  let (d, e, f) := tri₂
  tri₁ ≠ tri₂ ∧ (i = a + d ∧ j = b + e ∧ k = c + f ∧ permissible_triangle (a, b, c) p ∧ permissible_triangle (d, e, f) p)

end all_permissible_triangles_present_l65_65097


namespace min_a_plus_b_l65_65753

noncomputable def min_ab_value (a b : ℝ) :=
  a + b

theorem min_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a^2 ≥ 12 * b) (h2 : 9 * b^2 ≥ 4 * a) :
  min_ab_value a b ≥ 16 + (4 / 3) * real.cbrt 3 :=
sorry

end min_a_plus_b_l65_65753


namespace arrangement_count_of_1_to_6_divisible_by_7_l65_65897

theorem arrangement_count_of_1_to_6_divisible_by_7 :
  {s : List ℕ // s.perm [1, 2, 3, 4, 5, 6] ∧ ∀ a b c, List.pairs s (List.pairs s.tail s.tail^.tail = a :: b :: c :: _) → (a * c - b^2) % 7 = 0 } → 12 :=
sorry

end arrangement_count_of_1_to_6_divisible_by_7_l65_65897


namespace visitors_saturday_l65_65621

def friday_visitors : ℕ := 3575
def saturday_visitors : ℕ := 5 * friday_visitors

theorem visitors_saturday : saturday_visitors = 17875 := by
  -- proof details would go here
  sorry

end visitors_saturday_l65_65621


namespace right_triangle_hypotenuse_log_l65_65998

theorem right_triangle_hypotenuse_log (h : ℝ) :
  let a := log 125 / log 3,
      b := log 32 / log 5
  in a^2 + b^2 = h^2 →
  3^h = 5^13 :=
by
  intros a b hab
  rw [a, b]
  sorry

end right_triangle_hypotenuse_log_l65_65998


namespace area_triangle_BMN_l65_65560

noncomputable def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

structure Quadrilateral := (A B C D : Point) (angle_D_right : angle D A B = 90)
structure Segment := (p1 p2 : Point)

def AB := Segment ⟨A, B⟩
def BC := Segment ⟨B, C⟩
def CD := Segment ⟨C, D⟩
def DA := Segment ⟨D, A⟩
def midpoint_BC := midpoint B C
def midpoint_DA := midpoint D A
def BM := Segment ⟨B, midpoint_BC⟩
def MN := Segment ⟨midpoint_DA, midpoint_BC⟩

theorem area_triangle_BMN
  (quad : Quadrilateral A B C D)
  (h1 : ∥A - B∥ = 13)
  (h2 : ∥B - C∥ = 13)
  (h3 : ∥C - D∥ = 24)
  (h4 : ∥D - A∥ = 24)
  (h5 : quad.angle_D_right)
  (M : Point := midpoint(B, C))
  (N : Point := midpoint(D, A)) :
  area (triangle B M N) = 78 :=
sorry

end area_triangle_BMN_l65_65560


namespace major_axis_length_is_eight_l65_65025

noncomputable def length_of_major_axis : ℝ :=
  let foci1 := (3 : ℝ, -4 + 2 * Real.sqrt 3)
  let foci2 := (3 : ℝ, -4 - 2 * Real.sqrt 3)
  if (true : Prop) then -- this condition ensures that the foci and tangency to both the x and y axes are considered
    sorry -- proof not implemented
  else
    0 -- this default value ensures the noncomputable def

theorem major_axis_length_is_eight :
  length_of_major_axis = 8 := sorry

end major_axis_length_is_eight_l65_65025


namespace sum_of_simple_numbers_l65_65812

theorem sum_of_simple_numbers : 
  let simple_numbers := {k : ℕ | ∃ n : ℕ, 2 ≤ n ∧ k = 2^n - 1 ∧ 3 ≤ k ∧ k ≤ 2013} in
  ∑ k in simple_numbers, k = 2035 :=
by
  sorry

end sum_of_simple_numbers_l65_65812


namespace sum_of_divisors_330_l65_65686

theorem sum_of_divisors_330 : (∑ d in (finset.filter (λ d, 330 % d = 0) (finset.range (330 + 1))), d) = 864 :=
by {
  sorry
}

end sum_of_divisors_330_l65_65686


namespace a9_value_in_polynomial_l65_65158

theorem a9_value_in_polynomial :
  let f := λ x : ℝ, x^2 + x^10 in
  let g := λ x : ℝ, a_0 + a_1*(x+1) + a_2*(x+1)^2 + a_3*(x+1)^3 + a_4*(x+1)^4 +
                   a_5*(x+1)^5 + a_6*(x+1)^6 + a_7*(x+1)^7 + a_8*(x+1)^8 +
                   a_9*(x+1)^9 + a_{10}*(x+1)^10 in
  (x^2 + x^{10} = a_0 + a_1*(x+1) + a_2*(x+1)^2 + a_3*(x+1)^3 + a_4*(x+1)^4 +
                  a_5*(x+1)^5 + a_6*(x+1)^6 + a_7*(x+1)^7 + a_8*(x+1)^8 +
                  a_9*(x+1)^9 + a_{10}*(x+1)^10) →
  a_10 = 1 →
  (a_9 + 10 = 0) →
  a_9 = -10 :=
by
  intros
  sorry

end a9_value_in_polynomial_l65_65158


namespace repayment_amount_l65_65209

theorem repayment_amount (borrowed amount : ℝ) (increase_percentage : ℝ) (final_amount : ℝ) 
  (h1 : borrowed_amount = 100) 
  (h2 : increase_percentage = 0.10) :
  final_amount = borrowed_amount * (1 + increase_percentage) :=
by 
  rw [h1, h2]
  norm_num
  exact eq.refl 110


end repayment_amount_l65_65209


namespace find_f_2015_l65_65732

def f (x : ℝ) := 2 * x - 1 

theorem find_f_2015 (f : ℝ → ℝ)
  (H1 : ∀ a b : ℝ, f ((2 * a + b) / 3) = (2 * f a + f b) / 3)
  (H2 : f 1 = 1)
  (H3 : f 4 = 7) :
  f 2015 = 4029 := by 
  sorry

end find_f_2015_l65_65732


namespace senior_employee_bonus_l65_65300

theorem senior_employee_bonus (J S : ℝ) 
  (h1 : S = J + 1200)
  (h2 : J + S = 5000) : 
  S = 3100 :=
sorry

end senior_employee_bonus_l65_65300


namespace evaluate_expression_l65_65783

theorem evaluate_expression :
  (⟨⟨18 / 8⟩ - ⟨28 / 18⟩⟩) / (⟨28 / 8⟩ + ⟨(8 * 18) / 28⟩) = 1 / 10 := 
sorry

end evaluate_expression_l65_65783


namespace gary_hours_worked_l65_65005

/-- Prove that Gary worked 52 hours given the conditions. --/
theorem gary_hours_worked (wage : ℝ) (total_earnings : ℝ) (regular_hours : ℝ) (overtime_rate : ℝ) : 
  40 = regular_hours ∧
  12 = wage ∧
  696 = total_earnings ∧
  1.5 = overtime_rate →
  let RegularEarnings := regular_hours * wage,
      OvertimeEarnings := total_earnings - RegularEarnings,
      OvertimeHours := OvertimeEarnings / (wage * overtime_rate),
      TotalHours := regular_hours + OvertimeHours
  in TotalHours = 52 := 
by {
  sorry
}

end gary_hours_worked_l65_65005


namespace minimum_value_of_AF_plus_4BF_l65_65658

-- Define the parabola C : y² = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the focus F of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the absolute value of the distance between two points
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The main theorem to prove
theorem minimum_value_of_AF_plus_4BF : 
  ∃ (A B : ℝ × ℝ), parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
                   A ≠ B ∧ 
                   (minimum (λ A B, distance A focus + 4 * distance B focus) = 9) :=
begin
  sorry
end

end minimum_value_of_AF_plus_4BF_l65_65658


namespace final_position_correct_total_distance_correct_l65_65699

def movements : List Int := [15, -25, 20, -35]

-- Final Position: 
def final_position (moves : List Int) : Int := moves.sum

-- Total Distance Traveled calculated by taking the absolutes and summing:
def total_distance (moves : List Int) : Nat :=
  moves.map (λ x => Int.natAbs x) |>.sum

theorem final_position_correct : final_position movements = -25 :=
by
  sorry

theorem total_distance_correct : total_distance movements = 95 :=
by
  sorry

end final_position_correct_total_distance_correct_l65_65699


namespace sum_sequence_2023_l65_65511

def f (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

def sequence (n : ℕ) : ℝ := 
  if n % 3 = 0 then 1 else 
  if n % 3 = 1 then 0 else -1

theorem sum_sequence_2023 :
  (∑ i in Finset.range 2023, sequence i.succ) = 1 :=
by
  have h1 : sequence 1 = 1 := rfl
  have h2 : ∀ n, sequence (n + 3) = sequence n,
      from λ n => by simp [sequence, Nat.add_mod]
  have h3 : f (sequence 1) + f (sequence 2 + sequence 3) = 0,
      from calc
        f (sequence 1) = f 1 := rfl
        ... = 1 - (2 / (3^1 + 1)) := rfl
        ... = 1 - (2 / 4) := by norm_num
        ... = 0.5 := by norm_num
        ... (-f (sequence 2+sequence 3))  = (-f (0 + (-1))) := 
        rfl
     
  sorry

end sum_sequence_2023_l65_65511


namespace combinations_of_A_and_B_5AB0_divisible_by_4_and_5_l65_65993

theorem combinations_of_A_and_B_5AB0_divisible_by_4_and_5 :
  let valid_values_for_B := {b | b ∈ {0, 2, 4, 6, 8}} in
  let count_A := 10 in
  let valid_combinations := count_A * valid_values_for_B.card in
  valid_combinations = 50 :=
by
  let valid_values_for_B := {0, 2, 4, 6, 8}
  let valid_combinations := 10 * valid_values_for_B.to_finset.card
  sorry

end combinations_of_A_and_B_5AB0_divisible_by_4_and_5_l65_65993


namespace perpendicular_bisector_correct_vertex_C_correct_l65_65501

-- Define the vertices A, B, and the coordinates of the angle bisector line
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := -1, y := -1 }

-- The angle bisector CD equation
def angle_bisector_CD (p : Point) : Prop :=
  p.x + p.y - 1 = 0

-- The perpendicular bisector equation of side AB
def perpendicular_bisector_AB (p : Point) : Prop :=
  4 * p.x + 6 * p.y - 3 = 0

-- Coordinates of vertex C
def C_coordinates (c : Point) : Prop :=
  c.x = -1 ∧ c.y = 2

theorem perpendicular_bisector_correct :
  ∀ (M : Point), M.x = 0 ∧ M.y = 1/2 →
  ∀ (p : Point), perpendicular_bisector_AB p :=
sorry

theorem vertex_C_correct :
  ∃ (C : Point), angle_bisector_CD C ∧ (C : Point) = { x := -1, y := 2 } :=
sorry

end perpendicular_bisector_correct_vertex_C_correct_l65_65501


namespace inequality_proof_l65_65110

variable {R : Type*} [LinearOrderedField R]

-- Define our variables and conditions
variables (n : ℕ) (λ η : R) (x : Fin n → R)

-- Conditions: n > 1, λ > 0, η ≥ λ^2, and sum of x's equals to λ
variables (h1 : n > 1) (h2 : λ > 0) (h3 : η ≥ λ^2) (h4 : ∑ i, x i = λ)
variable (pos_x : ∀ i, 0 < x i)

-- Statement of the theorem
theorem inequality_proof :
  (∑ i, x i / (η * x ((i + 1) % n) - (x ((i + 1) % n))^3)) 
    ≥ n^3 / (η * n^2 - λ^2) :=
by
  sorry

end inequality_proof_l65_65110


namespace probability_of_winning_l65_65896

/-- 
Given the conditions:
1. The game consists of 8 rounds.
2. The probability of Alex winning any round is 1/2.
3. Mel's chance of winning any round is twice that of Chelsea.
4. The outcomes of the rounds are independent.
Prove that the probability that Alex wins 4 rounds, Mel wins 3 rounds, and Chelsea wins 1 round is 35/324.
-/
theorem probability_of_winning (p_A p_M p_C : ℚ) (n_A n_M n_C : ℕ) (total_rounds : ℕ)
  (h_rounds : total_rounds = 8)
  (h_pA : p_A = 1/2)
  (h_pM : p_M = 2 * p_C)
  (h_total_p : p_A + p_M + p_C = 1)
  (h_nA : n_A = 4)
  (h_nM : n_M = 3)
  (h_nC : n_C = 1) :
  (nat.choose 8 4 * nat.choose (8-4) (3) * nat.choose (8-4-3) (1)) 
  * (p_A^4 * p_M^3 * p_C^1) = 35 / 324 :=
by {
  sorry
}

end probability_of_winning_l65_65896


namespace sin_double_alpha_l65_65093

theorem sin_double_alpha {α : ℝ} (h0 : 0 < α) (h1 : α < π / 2)
  (h2 : tan (α + π / 4) = 3 * cos (2 * α)) :
  sin (2 * α) = 2 / 3 :=
by
  sorry

end sin_double_alpha_l65_65093


namespace total_marbles_l65_65888

variable (w o p : ℝ)

-- Conditions as hypothesis
axiom h1 : o + p = 10
axiom h2 : w + p = 12
axiom h3 : w + o = 5

theorem total_marbles : w + o + p = 13.5 :=
by
  sorry

end total_marbles_l65_65888


namespace min_sum_third_col_6x6_grid_l65_65052

theorem min_sum_third_col_6x6_grid :
  ∀ (grid : Matrix (Fin 6) (Fin 6) ℕ),
    (∀ i : Fin 6, ∀ j : Fin 5, grid i j < grid i (j + 1)) →
    (∃ (nums : Finset ℕ), nums = Finset.image (λ i : Fin 6, grid i 2) Finset.univ ∧ nums = {3, 6, 9, 12, 15, 18}) →
    (Finset.sum nums id = 63) :=
by
sorry

end min_sum_third_col_6x6_grid_l65_65052


namespace probability_no_two_same_color_consecutive_l65_65717

noncomputable def probability_no_same_color_consecutive (total_arrangements : ℕ) (successful_arrangements : ℕ) : ℝ := 
  successful_arrangements / total_arrangements

theorem probability_no_two_same_color_consecutive :
  let total_arrangements := Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3)
  let successful_arrangements := 3 * 2^8
  probability_no_same_color_consecutive total_arrangements successful_arrangements ≈ 48 / 105 :=
sorry

end probability_no_two_same_color_consecutive_l65_65717


namespace evaluate_expression_l65_65739

def seq (i : ℕ) : ℕ :=
  if 1 ≤ i ∧ i ≤ 5 then 2 * i else seq (i - 1) * seq (i - 2) + 1

def prod_seq (n : ℕ) : ℕ :=
  (List.range (n + 1)).map seq |>.prod

def sum_sq_seq (n : ℕ) : ℕ :=
  (List.range (n + 1)).map (λ i, seq i ^ 2) |>.sum

theorem evaluate_expression :
  prod_seq 10 - sum_sq_seq 10 = 
  sorry

end evaluate_expression_l65_65739


namespace sum_y_values_eq_zero_l65_65599

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x + 5

theorem sum_y_values_eq_zero :
  let S := { y : ℝ | 4096 * y^3 - 32 * y - 6 = 0 } in ∑ y in S, y = 0 :=
by
  let S := {y : ℝ | 4096 * y^3 - 32 * y - 6 = 0}
  sorry

end sum_y_values_eq_zero_l65_65599


namespace triangle_angle_A_sin_sum_range_l65_65823

-- Problem 1: Prove A = 2π/3 given the conditions.
theorem triangle_angle_A (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : (sin B + sin C) ^ 2 - sin A ^ 2 = sin B * sin C) :
  A = 2 * π / 3 :=
sorry

-- Problem 2: Prove the range for sin B + sin C given A = 2π/3.
theorem sin_sum_range (B C : ℝ)
  (h1 : B + C = π / 3)
  (h2 : 0 < B)
  (h3 : B < π / 3) :
  ∃ x, x ∈ (Set.Ioo (sqrt 3 / 2) 1) ∧ sin B + sin C = x :=
sorry

end triangle_angle_A_sin_sum_range_l65_65823


namespace dot_product_value_l65_65863

variable {V : Type*} [inner_product_space ℝ V]

theorem dot_product_value
  (a b : V)
  (h1 : ∥a + b∥ = sqrt 10)
  (h2 : ∥a - b∥ = sqrt 6) :
  a ⬝ b = 1 :=
by
  sorry

end dot_product_value_l65_65863


namespace total_runtime_combined_is_26430_l65_65176

theorem total_runtime_combined_is_26430 :
  let first_show_seasons := [
    (5, 18, 45),  -- 5 seasons with 18 episodes of 45 minutes
    (3, 22, 60),  -- 3 seasons with 22 episodes of 60 minutes
    (4, 15, 50),  -- 4 seasons with 15 episodes of 50 minutes
    (2, 12, 40)   -- 2 seasons with 12 episodes of 40 minutes
  ]
  let second_show_seasons := [
    (4, 16, 60),  -- 4 seasons with 16 episodes of 60 minutes
    (6, 24, 45),  -- 6 seasons with 24 episodes of 45 minutes
    (3, 20, 55),  -- 3 seasons with 20 episodes of 55 minutes
    (1, 28, 30)   -- 1 season with 28 episodes of 30 minutes
  ]
  let runtime (seasons : List (ℕ × ℕ × ℕ)) : ℕ :=
    seasons.foldl (fun total (s : ℕ × ℕ × ℕ) => total + s.1 * s.2 * s.3) 0
  runtime first_show_seasons + runtime second_show_seasons = 26430 := by
  sorry

end total_runtime_combined_is_26430_l65_65176


namespace problem1_problem2_real_problem2_complex_problem3_l65_65938

-- Problem 1: Prove that if 2 ∈ A, then {-1, 1/2} ⊆ A
theorem problem1 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : 2 ∈ A) : -1 ∈ A ∧ (1/2) ∈ A := sorry

-- Problem 2: Prove that A cannot be a singleton set for real numbers, but can for complex numbers.
theorem problem2_real (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : ¬(∃ a, A = {a}) := sorry

theorem problem2_complex (A : Set ℂ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (h2 : ∃ a ∈ A, a ≠ 1) : (∃ a, A = {a}) := sorry

-- Problem 3: Prove that 1 - 1/a ∈ A given a ∈ A
theorem problem3 (A : Set ℝ) (h1 : ∀ a ∈ A, (1/(1-a)) ∈ A) (a : ℝ) (ha : a ∈ A) : (1 - 1/a) ∈ A := sorry

end problem1_problem2_real_problem2_complex_problem3_l65_65938


namespace T_n_less_than_one_fourth_l65_65504

theorem T_n_less_than_one_fourth (a : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : ∀ n, 4 * S n = (a n) ^ 2 + 2 * a n)
  (h2 : ∀ n, a n = 2 * n)
  (h3 : ∀ n, b n = 1 / (a n * (a n + 2)))
  (h4 : ∀ n, T n = ∑ k in finset.range (n + 1), b k) :
  ∀ n, T n < 1 / 4 :=
by
  sorry

end T_n_less_than_one_fourth_l65_65504


namespace tsunami_added_sand_l65_65268

noncomputable def dig_rate : ℝ := 8 / 4 -- feet per hour
noncomputable def sand_after_storm : ℝ := 8 / 2 -- feet
noncomputable def time_to_dig_up_treasure : ℝ := 3 -- hours
noncomputable def total_sand_dug_up : ℝ := dig_rate * time_to_dig_up_treasure -- feet

theorem tsunami_added_sand :
  total_sand_dug_up - sand_after_storm = 2 :=
by
  sorry

end tsunami_added_sand_l65_65268


namespace chord_length_l65_65574

noncomputable def polar_equation_circle (ρ θ : ℝ) : Prop :=
  ρ = Real.sqrt 2 * Real.cos (θ + Real.pi / 4)

def parametric_equation_line (t : ℝ) : ℝ × ℝ :=
  (1 + (4 / 5) * t, -1 - (3 / 5) * t)

theorem chord_length :
  (∃ (ρ θ : ℝ), polar_equation_circle ρ θ) →
  (∃ (t : ℝ), parametric_equation_line t = (1 + (4 / 5) * t, -1 - (3 / 5) * t)) →
  ∃ (L : ℝ), L = 7 / 5 :=
by
  sorry

end chord_length_l65_65574


namespace proof_problem_l65_65196

variables (x_A x_B m : ℝ)

-- Condition 1:
def cost_relation : Prop := x_B = x_A + 20

-- Condition 2:
def quantity_relation : Prop := 540 / x_A = 780 / x_B

-- Condition 3:
def total_books := 70
def total_cost := 3550
def min_books_relation : Prop := 45 * m + 65 * (total_books - m) ≤ total_cost 

-- Part 1:
def cost_price_A (x : ℝ) : Prop := x = 45
def cost_price_B (x : ℝ) : Prop := x = 65

-- Part 2:
def min_books_A (m : ℝ) : Prop := m ≥ 50

-- Define the proof problem
theorem proof_problem :
  (cost_relation x_A x_B) ∧ 
  (quantity_relation x_A x_B) →
  (cost_price_A x_A) ∧ 
  (cost_price_B x_B) ∧ 
  (min_books_relation x_A x_B m) → 
  (min_books_A m) :=
by sorry

end proof_problem_l65_65196


namespace ellipse_equation_and_triangle_area_l65_65481

theorem ellipse_equation_and_triangle_area :
  let center := (0, 0)
  let focus := (0, Real.sqrt 2)
  let e := Real.sqrt 2 / 2
  let a := 2
  let b := Real.sqrt 2
  let ellipse_eq := ∀ x y : ℝ, (x^2 / 2 + y^2 / 4 = 1)
  let P := (1, Real.sqrt 2)
  let eq_area (A B : ℝ × ℝ) : ℝ := 
    let d (p1 p2 : ℝ × ℝ) := Real.sqrt (((p1.1 - p2.1) ^ 2) + ((p1.2 - p2.2) ^ 2))
    (1 / 2) * Real.sqrt 2 * Real.abs ((P.2 - A.2) * B.1 - (P.1 - A.1) * B.2) / b
  Exists A B : ℝ × ℝ, (∃ k : ℝ, slope_k (1, sqrt 2) A = k ∧ slope_k (1, sqrt 2) B = -k ∧ k ≠ 0 ∧ ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2) ∧ eq_area P A B = Real.sqrt 2
:=
by
  sorry

end ellipse_equation_and_triangle_area_l65_65481


namespace airplane_seat_difference_l65_65023

theorem airplane_seat_difference (F C X : ℕ) 
    (h1 : 387 = F + 310) 
    (h2 : C = 310) 
    (h3 : C = 4 * F + X) :
    X = 2 :=
by
    sorry

end airplane_seat_difference_l65_65023


namespace option_A_is_correct_l65_65845

variable {θ : ℝ}
def tan (θ : ℝ) := Real.sin θ / Real.cos θ

theorem option_A_is_correct :
  let a := Real.cos (2 * θ)
  let b := Real.sin (2 * θ)
  (
    tan θ = (1 - a) / b ∨
    tan θ = (1 + a) / b ∨
    tan θ = b / (1 + a) ∨
    tan θ = b / (1 - a)
  ) ∧
  (
    ∃ e1 e2 e3 e4 : ℝ, 
    (e1 = (1 - a) / b ∨ e1 = (1 + a) / b ∨ e1 = b / (1 + a) ∨ e1 = b / (1 - a)) ∧ 
    (e2 = (1 - a) / b ∨ e2 = (1 + a) / b ∨ e2 = b / (1 + a) ∨ e2 = b / (1 - a)) ∧ 
    e1 = tan θ ∧ 
    e2 = tan θ ∧ 
    e1 ≠ e2
  ) 
:= sorry

end option_A_is_correct_l65_65845


namespace no_solutions_for_a_gt_1_l65_65444

theorem no_solutions_for_a_gt_1 (a b : ℝ) (h_a_gt_1 : 1 < a) :
  ¬∃ x : ℝ, a^(2-2*x^2) + (b+4) * a^(1-x^2) + 3*b + 4 = 0 ↔ 0 < b ∧ b < 4 :=
by
  sorry

end no_solutions_for_a_gt_1_l65_65444


namespace dot_product_range_l65_65872

variables {ℝ : Type*} [OrderedSemiring ℝ] [NormedGroup ℝ]

variables (c d : ℝ) 

noncomputable def norm (v : ℝ) : ℝ := |v|

def cos (θ : ℝ) : ℝ := sorry  -- Assume cosine function is defined

theorem dot_product_range (θ : ℝ) (h1 : norm c = 8) (h2 : norm d = 13) :
  -104 ≤ c * d * cos θ ∧ c * d * cos θ ≤ 104 :=
sorry

end dot_product_range_l65_65872


namespace maximum_value_A_B_l65_65575

noncomputable def line_l (t α : ℝ) : ℝ × ℝ :=
(x * cos α, y * sin α)

def circle_C (x y : ℝ) : Prop :=
(x - 1)^2 + (y - 2)^2 = 4

theorem maximum_value_A_B (t1 t2 α : ℝ) (Hα : 0 < α ∧ α < π / 2) :
  let l_x := t1 * cos α
  let l_y := t1 * sin α
  let C := circle_C l_x l_y
  C → (t1 + t2 = 2 * cos α + 4 * sin α) ∧ (t1 * t2 = 1) →
  (frac_1_OA_plus_OB := (1 / abs t1) + (1 / abs t2)) := 2 * sqrt 5 :=
sorry

end maximum_value_A_B_l65_65575


namespace rahul_share_proof_l65_65707

variable (total_payment : ℝ) (rahul_days : ℝ) (rajesh_days : ℝ) (rahul_share : ℝ)

def rahul_payment (total_payment : ℝ) (rahul_days : ℝ) (rajesh_days : ℝ) : ℝ :=
  total_payment * (1 / rahul_days) / ((1 / rahul_days) + (1 / rajesh_days))

theorem rahul_share_proof :
  rahul_payment 2250 3 2 = 900 := by
  sorry

end rahul_share_proof_l65_65707


namespace maximize_volume_l65_65299

variables {α β : ℝ} {MD : ℝ} [Fact (MD = 3)]

-- We will add some assumptions to define our rectangle base and height relations
def base_is_rectangle (A B C D : ℝ × ℝ) :=
  A.1 = 0 ∧ A.2 = 0 ∧
  B.1 > 0 ∧ B.2 = 0 ∧
  C.1 > 0 ∧ C.2 > 0 ∧
  D.1 = 0 ∧ D.2 > 0

def perpend_faces_abm_bcm (A B C D M : ℝ × ℝ × ℝ) :=
  M.1 = D.1 ∧ M.2 = D.2 ∧ M.3 = 3

def volume_pyramid (V : ℝ) (A B C D M : ℝ × ℝ × ℝ) :=
  V = (1 / 3) * dist A B * dist B C * dist_point_to_plane M A B C

def optimal_angles (α β : ℝ) :=
  α = Real.pi / 4 ∧ β = Real.pi / 4

theorem maximize_volume :
  ∀ (A B C D M : ℝ × ℝ × ℝ), 
    base_is_rectangle A.1 A.2 ∧
    base_is_rectangle B.1 B.2 ∧
    base_is_rectangle C.1 C.2 ∧
    base_is_rectangle D.1 D.2 ∧
    perpend_faces_abm_bcm A B C D M →
    ∃ V : ℝ, 
      volume_pyramid V A B C D M ∧
      optimal_angles α β :=
by
  sorry

end maximize_volume_l65_65299


namespace number_of_ducks_l65_65736

-- Definitions of variables and conditions
variable (C D T B : ℕ)

-- Given conditions
def condition1 := C = 56
def condition2 := D = T / 12
def condition3 := D = B / 4

-- We aim to prove that D = 7 given these conditions
theorem number_of_ducks (h1 : condition1) (h2 : condition2) (h3 : condition3) : D = 7 :=
sorry

end number_of_ducks_l65_65736


namespace count_diff_squares_l65_65532

theorem count_diff_squares (n : ℕ) (h : 1 ≤ n ∧ n ≤ 2000) :
  1000 = (∑ k in (finset.range 2000).filter (λ x, (n = (2 * k + 1) ∨ n % 4 = 0) ∧ n ≤ 2000), 1) - 
         (∑ k in (finset.range 2000).filter (λ x, n % 4 = 2), 1) :=
by
  sorry

end count_diff_squares_l65_65532


namespace sequence_sum_l65_65611

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℕ := n + 1

-- Define the geometric sequence {b_n}
def b_n (n : ℕ) : ℕ := 2^(n - 1)

-- State the theorem
theorem sequence_sum : (b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) + b_n (a_n 5) + b_n (a_n 6)) = 126 := by
  sorry

end sequence_sum_l65_65611


namespace cheryl_probability_at_least_two_same_color_l65_65718

theorem cheryl_probability_at_least_two_same_color :
  let total_marbles := 10
  let red_marbles := 4
  let green_marbles := 3
  let yellow_marbles := 3
  let carol_draws := 3
  let claudia_draws := 3
  let cheryl_draws := total_marbles - carol_draws - claudia_draws
  total_marbles = red_marbles + green_marbles + yellow_marbles →
  cheryl_draws = 4 →
  (probability (Cheryl_draws_at_least_two_same_color total_marbles red_marbles green_marbles yellow_marbles carol_draws claudia_draws cheryl_draws) = 1) :=
begin
  sorry
end

end cheryl_probability_at_least_two_same_color_l65_65718


namespace max_value_is_63_l65_65981

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^2 + 3*x*y + 4*y^2

theorem max_value_is_63 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (cond : x^2 - 3*x*y + 4*y^2 = 9) :
  max_value x y ≤ 63 :=
by
  sorry

end max_value_is_63_l65_65981


namespace solution_set_l65_65231

noncomputable def f : ℝ → ℝ := sorry

-- Given conditions
axiom f_cond_1 : f 1 = 1
axiom f_cond_2 : ∀ x, f'' x < 1/2

-- Translate to Lean statement
theorem solution_set :
  ∀ x, (x ∈ (set.Ioo 0 (1/10)) ∪ set.Ioi 10) → 
       f (real.log x ^ 2) < (real.log x ^ 2) / 2 + 1/2 :=
by
  sorry

end solution_set_l65_65231


namespace tree_original_height_l65_65923

theorem tree_original_height (current_height_in: ℝ) (growth_percentage: ℝ)
  (h1: current_height_in = 180) (h2: growth_percentage = 0.50) :
  ∃ (original_height_ft: ℝ), original_height_ft = 10 :=
by
  have original_height_in := current_height_in / (1 + growth_percentage)
  have original_height_ft := original_height_in / 12
  use original_height_ft
  sorry

end tree_original_height_l65_65923


namespace product_of_xy_l65_65814

theorem product_of_xy (x y : ℝ) : 
  (1 / 5 * (x + y + 4 + 5 + 6) = 5) ∧ 
  (1 / 5 * ((x - 5) ^ 2 + (y - 5) ^ 2 + (4 - 5) ^ 2 + (5 - 5) ^ 2 + (6 - 5) ^ 2) = 2) 
  → x * y = 21 :=
by sorry

end product_of_xy_l65_65814


namespace relationship_among_abc_l65_65229

theorem relationship_among_abc (x : ℝ) (hx : 1 < x) :
  let a := 2^0.3
  let b := 0.3^2
  let c := log x (x^2 + 0.3)
  c > a ∧ a > b := 
by
  sorry

end relationship_among_abc_l65_65229


namespace change_percentage_difference_l65_65406

theorem change_percentage_difference 
  (initial_yes : ℚ) (initial_no : ℚ) (initial_undecided : ℚ)
  (final_yes : ℚ) (final_no : ℚ) (final_undecided : ℚ)
  (h_initial : initial_yes = 0.4 ∧ initial_no = 0.3 ∧ initial_undecided = 0.3)
  (h_final : final_yes = 0.6 ∧ final_no = 0.1 ∧ final_undecided = 0.3) :
  (final_yes - initial_yes + initial_no - final_no) = 0.2 := by
sorry

end change_percentage_difference_l65_65406


namespace austin_needs_six_weeks_l65_65409

theorem austin_needs_six_weeks
  (work_rate: ℕ) (hours_monday hours_wednesday hours_friday: ℕ) (bicycle_cost: ℕ) 
  (weekly_hours: ℕ := hours_monday + hours_wednesday + hours_friday) 
  (weekly_earnings: ℕ := weekly_hours * work_rate) 
  (weeks_needed: ℕ := bicycle_cost / weekly_earnings):
  work_rate = 5 ∧ hours_monday = 2 ∧ hours_wednesday = 1 ∧ hours_friday = 3 ∧ bicycle_cost = 180 ∧ weeks_needed = 6 :=
by {
  sorry
}

end austin_needs_six_weeks_l65_65409


namespace rhombus_area_l65_65673

-- Given conditions
def rhombus_side_length : ℝ := 5
def rhombus_angle : ℝ := 60 * Real.pi / 180 -- converting degrees to radians

-- Prove the area of the rhombus
theorem rhombus_area :
  let height := rhombus_side_length * Real.sin (Real.pi / 3),
      base := rhombus_side_length
  in (base * height) = 12.5 * Real.sqrt 3 :=
by sorry

end rhombus_area_l65_65673


namespace fifth_sphere_radius_l65_65168

noncomputable def cone_height : ℝ := 7
noncomputable def cone_base_radius : ℝ := 7

axiom r1 (r : ℝ) : 
  (r * (2 * real.sqrt 2 + 1) = cone_height) → 
  r = (2 * real.sqrt 2 - 1)

theorem fifth_sphere_radius (h : ℝ) (r : ℝ) (r2 : ℝ) :
  h = cone_height → r = (2 * real.sqrt 2 - 1) → r2 = r → r2 = (2 * real.sqrt 2 - 1) :=
by
  intros h_eq r_eq r2_eq
  rw [h_eq, r_eq, r2_eq]
  sorry

end fifth_sphere_radius_l65_65168


namespace team_lineup_count_l65_65626

theorem team_lineup_count (total_members specialized_kickers remaining_players : ℕ) 
  (captain_assignments : specialized_kickers = 2) 
  (available_members : total_members = 20) 
  (choose_players : remaining_players = 8) : 
  (2 * (Nat.choose 19 remaining_players)) = 151164 := 
by
  sorry

end team_lineup_count_l65_65626


namespace counting_integers_satisfying_inequality_l65_65531

theorem counting_integers_satisfying_inequality : 
  {n : ℤ | n ≠ 0 ∧ (1 / (|n| : ℝ)) > 1 / 5}.to_finset.card = 8 :=
by 
  sorry

end counting_integers_satisfying_inequality_l65_65531


namespace repay_loan_with_interest_l65_65216

theorem repay_loan_with_interest (amount_borrowed : ℝ) (interest_rate : ℝ) (total_payment : ℝ) 
  (h1 : amount_borrowed = 100) (h2 : interest_rate = 0.10) :
  total_payment = amount_borrowed + (amount_borrowed * interest_rate) :=
by sorry

end repay_loan_with_interest_l65_65216


namespace carHouseholdsCorrect_l65_65173

-- Definitions (conditions from part a)
variable (households : ℕ) -- total number of households
variable (neitherCarNorBike : ℕ) -- households with neither car nor bike
variable (bothCarAndBike : ℕ) -- households with both car and bike
variable (bikeOnly : ℕ) -- households with only a bike

-- Given conditions
def totalHouseholds : Prop := households = 90
def neitherCarNorBikeCondition : Prop := neitherCarNorBike = 11
def bothCarAndBikeCondition : Prop := bothCarAndBike = 20
def bikeOnlyCondition : Prop := bikeOnly = 35

-- Conclusion to prove
def hasCarCondition : Prop :=
  let totalWithAtLeastOne := households - neitherCarNorBike
  let carOnly := totalWithAtLeastOne - (bikeOnly + bothCarAndBike)
  let totalWithCar := carOnly + bothCarAndBike
  totalWithCar = 44

theorem carHouseholdsCorrect:
  totalHouseholds ∧ neitherCarNorBikeCondition ∧ bothCarAndBikeCondition ∧ bikeOnlyCondition → hasCarCondition :=
  by
    intro h
    sorry

end carHouseholdsCorrect_l65_65173


namespace is_monotonically_increasing_l65_65040

def function_y (x : ℝ) : ℝ := (3 - x^2) * Real.exp x

lemma derivative_y (x : ℝ) : deriv (function_y) x = (3 - x^2 - 2*x) * Real.exp x :=
by
  -- Use derivation rules and properties of exponential function
  sorry

theorem is_monotonically_increasing : ∀ x : ℝ, x > -3 ∧ x < 1 → deriv (function_y) x > 0 :=
by
  intros x hx
  rw derivative_y
  -- Simplify (3 - x^2 - 2*x)
  -- Note: This should be solved to complete the proof
  -- Proof of positivity of (3 - x^2 - 2*x)
  sorry

end is_monotonically_increasing_l65_65040


namespace calculate_annual_rent_l65_65259

-- Defining the conditions
def num_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4
def monthly_rent : ℚ := 400

-- Defining the target annual rent
def annual_rent (units : ℕ) (occupancy : ℚ) (rent : ℚ) : ℚ :=
  let occupied_units := occupancy * units
  let monthly_revenue := occupied_units * rent
  monthly_revenue * 12

-- Proof problem statement
theorem calculate_annual_rent :
  annual_rent num_units occupancy_rate monthly_rent = 360000 := by
  sorry

end calculate_annual_rent_l65_65259


namespace probability_of_girls_under_18_l65_65556

theorem probability_of_girls_under_18
  (total_members : ℕ)
  (girls : ℕ)
  (boys : ℕ)
  (underaged_girls : ℕ)
  (two_members_chosen : ℕ)
  (total_ways_to_choose_two : ℕ)
  (ways_to_choose_two_girls : ℕ)
  (ways_to_choose_at_least_one_underaged : ℕ)
  (prob : ℚ)
  : 
  total_members = 15 →
  girls = 8 →
  boys = 7 →
  underaged_girls = 3 →
  two_members_chosen = 2 →
  total_ways_to_choose_two = (Nat.choose total_members two_members_chosen) →
  ways_to_choose_two_girls = (Nat.choose girls two_members_chosen) →
  ways_to_choose_at_least_one_underaged = 
    (Nat.choose underaged_girls 1 * Nat.choose (girls - underaged_girls) 1 + Nat.choose underaged_girls 2) →
  prob = (ways_to_choose_at_least_one_underaged : ℚ) / (total_ways_to_choose_two : ℚ) →
  prob = 6 / 35 :=
by
  intros
  sorry

end probability_of_girls_under_18_l65_65556


namespace solve_correct_problems_l65_65890

theorem solve_correct_problems (x : ℕ) (h1 : 3 * x + x = 120) : x = 30 :=
by
  sorry

end solve_correct_problems_l65_65890


namespace purely_imaginary_z_l65_65112

noncomputable def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem purely_imaginary_z (z : ℂ) (h₁ : is_purely_imaginary z) (h₂ : is_purely_imaginary ((z + 2) ^ 2 - 8 * complex.I)) :
  z = -2 * complex.I :=
by
  sorry

end purely_imaginary_z_l65_65112


namespace union_of_sets_l65_65858

theorem union_of_sets (A B : Set ℕ) (hA : A = {1, 2, 6}) (hB : B = {2, 3, 6}) :
  A ∪ B = {1, 2, 3, 6} :=
by
  rw [hA, hB]
  ext x
  simp [Set.union]
  sorry

end union_of_sets_l65_65858


namespace price_of_dynaco_stock_l65_65026

theorem price_of_dynaco_stock 
  (price_microtron : ℕ) (total_shares : ℕ) (average_price : ℕ) (shares_dynaco : ℕ) 
  (revenue_microtron : ℤ) (revenue_total : ℤ) 
  (h₀ : price_microtron = 36)
  (h₁ : total_shares = 300)
  (h₂ : average_price = 40)
  (h₃ : shares_dynaco = 150)
  (h₄ : revenue_microtron = ↑(shares_dynaco * price_microtron))
  (h₅ : revenue_total = ↑(total_shares * average_price)) :
  ∃ (price_dynaco : ℤ), price_dynaco = 44 :=
by
  use 44
  sorry

end price_of_dynaco_stock_l65_65026


namespace find_principal_sum_l65_65302

def compound_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

theorem find_principal_sum (CI SI : ℝ) (t : ℕ)
  (h1 : CI = 11730) 
  (h2 : SI = 10200) 
  (h3 : t = 2) :
  ∃ P r, P = 17000 ∧
  compound_interest P r t = CI ∧
  simple_interest P r t = SI :=
by
  sorry

end find_principal_sum_l65_65302


namespace coefficient_x3_in_expansion_l65_65570

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem coefficient_x3_in_expansion :
  let x := 1
  let a := x
  let b := 2
  let n := 50
  let k := 47
  let coefficient := binom n (n - k) * b^k
  coefficient = 19600 * 2^47 := by
  sorry

end coefficient_x3_in_expansion_l65_65570


namespace distance_between_foci_hyperbola_l65_65796

open Real

theorem distance_between_foci_hyperbola 
  (x y : ℝ) 
  (h : 9 * x^2 - 18 * x - 16 * y^2 - 32 * y = 144) : 
  2 * sqrt(137 / 9 + 137 / 16) = sqrt(3047) / 6 := 
  sorry

end distance_between_foci_hyperbola_l65_65796


namespace induction_product_equality_l65_65272

open Nat

theorem induction_product_equality :
  ∀ (n : ℕ), n > 0 → ( ∏ i in (finset.range n).map (λ x, x + n + 1), i ) = 2^n * ∏ i in (finset.range (2 * n)), i + 1 :=
by
  sorry

end induction_product_equality_l65_65272


namespace find_x_l65_65085

-- Defining vectors a and b
def vector_a : ℝ × ℝ × ℝ := (3, 2, -1)
def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (x, -4, 2)

-- Definition of orthogonality of two vectors
def orthogonal (a b : ℝ × ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3 = 0

-- The proof statement
theorem find_x (x : ℝ) (h : orthogonal (3, 2, -1) (x, -4, 2)) : x = 10 / 3 :=
sorry

end find_x_l65_65085


namespace percentage_decrease_in_spring_l65_65018

-- Given Conditions
variables (initial_members : ℕ) (increased_percent : ℝ) (total_decrease_percent : ℝ)
-- population changes
variables (fall_members : ℝ) (spring_members : ℝ)

-- The initial conditions given by the problem
axiom initial_membership : initial_members = 100
axiom fall_increase : increased_percent = 6
axiom total_decrease : total_decrease_percent = 14.14

-- Derived values based on conditions
axiom fall_members_calculated : fall_members = initial_members * (1 + increased_percent / 100)
axiom spring_members_calculated : spring_members = initial_members * (1 - total_decrease_percent / 100)

-- The correct answer which we need to prove
theorem percentage_decrease_in_spring : 
  ((fall_members - spring_members) / fall_members) * 100 = 19 := by
  sorry

end percentage_decrease_in_spring_l65_65018


namespace john_ate_12_ounces_of_steak_l65_65920

-- Conditions
def original_weight : ℝ := 30
def burned_fraction : ℝ := 0.5
def eaten_fraction : ℝ := 0.8

-- Theorem statement
theorem john_ate_12_ounces_of_steak :
  (original_weight * (1 - burned_fraction) * eaten_fraction) = 12 := by
  sorry

end john_ate_12_ounces_of_steak_l65_65920


namespace diameter_tube_is_correct_l65_65379

noncomputable def diameter_of_tube (length_tape : ℝ) (thickness_tape : ℝ) (diameter_roll : ℝ) : ℝ :=
  10 * real.sqrt ((real.pi - 1) / real.pi)

theorem diameter_tube_is_correct :
  ∀ (length_tape thickness_tape diameter_roll: ℝ),
  length_tape = 25 →
  thickness_tape = 0.01 →
  diameter_roll = 10 →
  diameter_of_tube length_tape thickness_tape diameter_roll = 10 * real.sqrt ((real.pi - 1) / real.pi) :=
by 
  intros length_tape thickness_tape diameter_roll h1 h2 h3
  rw [h1, h2, h3]
  sorry

end diameter_tube_is_correct_l65_65379


namespace ellipse_standard_form_range_m_l65_65944

--(I) Given the ellipse equation and focal distance
theorem ellipse_standard_form (a : ℝ) (h_pos : a > 0) (hfocal : 2 * sqrt(a^2 - (8 - a^2)) = 4) :
  (a^2 = 6) →
  (∀ (x y : ℝ), (x^2 / 6 + y^2 / 2 = 1) ↔ (x^2 / a^2 + y^2 / (8 - a^2) = 1)) :=
by {
  intros h_ellipse,
  sorry
}

--(II) Given the line inclination, point M and condition related to the foci and chord
theorem range_m (m : ℝ) (a : ℝ) :
  a = sqrt(6) →
  (m > a) →
  (m^2 < 12) →
  (∀ (x1 x2 y1 y2 : ℝ), (x1 + x2 = m) ∧ (x1 * x2 = (m^2 - 6) / 2) →
    (4 * (m^2 - 6) / 2 - (m + 6) * m + m^2 + 12 < 0) →
    (0 < m) ∧ (m < 3) →
    (m ∈ (sqrt(6), 3))) :=
by {
  intros h_a m_gt_a m_lt_12 h_conditions,
  sorry
}

end ellipse_standard_form_range_m_l65_65944


namespace problem_solution_l65_65857

theorem problem_solution (a b : ℝ) (h : {1, a, b / 2} = {0, a^2, a + b}) : a ^ 2013 + b ^ 2014 = -1 :=
by
  -- This is where the proof would go
  sorry

end problem_solution_l65_65857


namespace exists_positive_integer_k_l65_65975

def satisfies_conditions (k : ℕ) : Prop :=
  gcd 2012 2020 = gcd (2012 + k) 2020 ∧
  gcd 2012 2020 = gcd 2012 (2020 + k) ∧
  gcd 2012 2020 = gcd (2012 + k) (2020 + k)

theorem exists_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ satisfies_conditions k ∧
  ∃ n : ℕ, k = 8 * n ∧ gcd n (5 * 101 * 503) = 1 :=
begin
  sorry
end

end exists_positive_integer_k_l65_65975


namespace spider_legs_is_multiple_of_human_legs_l65_65012

def human_legs : ℕ := 2
def spider_legs : ℕ := 8

theorem spider_legs_is_multiple_of_human_legs :
  spider_legs = 4 * human_legs :=
by 
  sorry

end spider_legs_is_multiple_of_human_legs_l65_65012


namespace class_avg_GPA_l65_65309

theorem class_avg_GPA (n : ℕ) (h1 : n > 0) : 
  ((1 / 4 : ℝ) * 92 + (3 / 4 : ℝ) * 76 = 80) :=
sorry

end class_avg_GPA_l65_65309


namespace find_values_l65_65529

-- Definitions of the vectors
def a (λ : ℝ) : ℝ × ℝ × ℝ := (λ + 1, 0, 2)
def b (λ μ : ℝ) : ℝ × ℝ × ℝ := (6, 2*μ - 1, 2*λ)

-- Definition of parallelism for vectors
def vectors_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, a = (t * b.1, t * b.2, t * b.3)

-- Main theorem to prove
theorem find_values (λ μ : ℝ) :
  vectors_parallel (a λ) (b λ μ) ↔ (λ = 2 ∧ μ = 0.5) ∨ (λ = -3 ∧ μ = 0.5) :=
by
  sorry

end find_values_l65_65529


namespace positive_integers_less_than_2000_with_property_l65_65143

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem positive_integers_less_than_2000_with_property :
  {n : ℕ // n > 0 ∧ n < 2000 ∧ n = 5 * (sum_of_digits n)}.card = 2 :=
by
  sorry

end positive_integers_less_than_2000_with_property_l65_65143


namespace sixty_th_number_is_16_l65_65654

theorem sixty_th_number_is_16 :
  let cumulative : ℕ → ℕ
      cumulative 0 := 0
      cumulative (n+1) := cumulative n + 2 * (n + 1)
  in cumulative 7 < 60 ∧ 60 ≤ cumulative 8 → 60 * 16 = 960 := 
by
  sorry

end sixty_th_number_is_16_l65_65654


namespace arnold_monthly_mileage_l65_65403

-- Define the conditions
def car1_mpg := 50
def car2_mpg := 10
def car3_mpg := 15
def gas_cost_per_gallon := 2
def monthly_gas_expense := 56

-- Translate the question to a Lean theorem statement
theorem arnold_monthly_mileage : 
  ∃ (M : ℝ), 
  (M / 3 * (1 / car1_mpg) * gas_cost_per_gallon + 
   M / 3 * (1 / car2_mpg) * gas_cost_per_gallon + 
   M / 3 * (1 / car3_mpg) * gas_cost_per_gallon = monthly_gas_expense) ∧
  M ≈ 586.05 :=
begin
  sorry
end

end arnold_monthly_mileage_l65_65403


namespace triangle_inequality_example_l65_65359

/-- Triangle inequality theorem application --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_inequality_example : can_form_triangle 8 8 15 :=
by {
  unfold can_form_triangle,
  split,
  calc
    8 + 8 > 15 : by linarith,
  split,
  calc
    8 + 15 > 8 : by linarith,
  calc
    8 + 15 > 8 : by linarith,
}

end triangle_inequality_example_l65_65359


namespace repay_loan_with_interest_l65_65215

theorem repay_loan_with_interest (amount_borrowed : ℝ) (interest_rate : ℝ) (total_payment : ℝ) 
  (h1 : amount_borrowed = 100) (h2 : interest_rate = 0.10) :
  total_payment = amount_borrowed + (amount_borrowed * interest_rate) :=
by sorry

end repay_loan_with_interest_l65_65215


namespace quadrilateral_inequality_l65_65475

theorem quadrilateral_inequality
  (A B C D O I K H : Point)
  (convex_ABCD : ConvexQuadrilateral A B C D)
  (intersection_O : IntersectingDiagonals A C B D O)
  (I_foot : FootOfPerpendicular B A D I)
  (K_foot : FootOfPerpendicular O A D K)
  (H_foot : FootOfPerpendicular C A D H) :
  distance A D * distance B I * distance C H ≤ distance A C * distance B D * distance O K := sorry


end quadrilateral_inequality_l65_65475


namespace rectangle_perimeter_l65_65891

def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

def satisfies_relations (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  b1 + b2 = b3 ∧
  b1 + b3 = b4 ∧
  b3 + b4 = b5 ∧
  b4 + b5 = b6 ∧
  b2 + b5 = b7

def non_overlapping_squares (b1 b2 b3 b4 b5 b6 b7 : ℕ) : Prop :=
  -- Placeholder for expressing that the squares are non-overlapping.
  true -- This is assumed as given in the problem.

theorem rectangle_perimeter (b1 b2 b3 b4 b5 b6 b7 : ℕ)
  (h1 : b1 = 1) (h2 : b2 = 2)
  (h_relations : satisfies_relations b1 b2 b3 b4 b5 b6 b7)
  (h_non_overlapping : non_overlapping_squares b1 b2 b3 b4 b5 b6 b7)
  (h_rel_prime : relatively_prime b6 b7) :
  2 * (b6 + b7) = 46 := by
  sorry

end rectangle_perimeter_l65_65891


namespace barbed_wire_cost_l65_65983

noncomputable def total_cost_barbed_wire (area : ℕ) (cost_per_meter : ℝ) (gate_width : ℕ) : ℝ :=
  let s := Real.sqrt area
  let perimeter := 4 * s - 2 * gate_width
  perimeter * cost_per_meter

theorem barbed_wire_cost :
  total_cost_barbed_wire 3136 3.5 1 = 777 := by
  sorry

end barbed_wire_cost_l65_65983


namespace correct_calculated_value_l65_65868

theorem correct_calculated_value (x : ℝ) (h : 3 * x - 5 = 103) : x / 3 - 5 = 7 := 
by 
  sorry

end correct_calculated_value_l65_65868


namespace arithmetic_sequence_divisible_by_2005_l65_65178

-- Problem Statement
theorem arithmetic_sequence_divisible_by_2005
  (a : ℕ → ℕ) -- Define the arithmetic sequence
  (d : ℕ) -- Common difference
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) -- Arithmetic sequence condition
  (h_product_div_2005 : ∀ n, 2005 ∣ (a n) * (a (n + 31))) -- Given condition on product divisibility
  : ∀ n, 2005 ∣ a n := 
sorry

end arithmetic_sequence_divisible_by_2005_l65_65178


namespace min_integers_to_ensure_double_l65_65805

theorem min_integers_to_ensure_double (s : Finset ℕ) (h : s = Finset.range 17 \ {0}) : 
  ∃ (A : Finset ℕ), A ⊆ s ∧ A.card = 12 ∧ ∃ a b ∈ A, a = 2 * b := 
by
  sorry

end min_integers_to_ensure_double_l65_65805


namespace cannot_form_set_l65_65355

-- Definitions based on conditions in a)
def male_students_class_20_grade_1_fogang_middle_school : Set := sorry
def parents_of_students_fogang_middle_school : Set := sorry
def family_members_li_ming : Set := sorry
def good_friends_wang_ming : Type := sorry

-- The theorem statement we aim to prove
theorem cannot_form_set {A B C : Set} {D : Type} :
  (A == male_students_class_20_grade_1_fogang_middle_school) →
  (B == parents_of_students_fogang_middle_school) →
  (C == family_members_li_ming) →
  (D == good_friends_wang_ming) →
  ¬(is_set D) :=
by
  intros hA hB hC hD
  sorry

end cannot_form_set_l65_65355


namespace minimize_expression_l65_65061

variable (a b c : ℝ)
variable (h1 : a > b)
variable (h2 : b > c)
variable (h3 : a ≠ 0)

theorem minimize_expression : 
  (a > b) → (b > c) → (a ≠ 0) → 
  ∃ x : ℝ, x = 4 ∧ ∀ y, y = (a+b)^2 + (b+c)^2 + (c+a)^2 / a^2 → x ≤ y := sorry

end minimize_expression_l65_65061


namespace sum_a_n_l65_65514

def f (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

noncomputable def sequence (a : ℕ → ℝ) : ℕ → ℝ
| 0 := 1
| (n + 1) := sequence (n % 3)

theorem sum_a_n (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 3) = a n)
  (h3 : f (a 1) + f (a 2 + a 3) = 0) :
  (Finset.range 2023).sum a = 1 :=
sorry

end sum_a_n_l65_65514


namespace find_ABC_l65_65889

-- Define the angles as real numbers in degrees
variables (ABC CBD DBC DBE ABE : ℝ)

-- Assert the given conditions
axiom horz_angle: CBD = 90
axiom DBC_ABC_relation : DBC = ABC + 30
axiom straight_angle: DBE = 180
axiom measure_abe: ABE = 145

-- State the proof problem
theorem find_ABC : ABC = 30 :=
by
  -- Include all steps required to derive the conclusion in the proof
  sorry

end find_ABC_l65_65889


namespace log_comparison_l65_65473

theorem log_comparison :
  let a := Real.logBase 3 2
  let b := Real.log 2
  let c := 5 ^ (-1 / 2 : ℝ)
  in c < a ∧ a < b :=
by 
  sorry

end log_comparison_l65_65473


namespace hot_sauce_container_size_l65_65585

theorem hot_sauce_container_size :
  let serving_size := 0.5
  let servings_per_day := 3
  let days := 20
  let total_consumed := servings_per_day * serving_size * days
  let one_quart := 32
  one_quart - total_consumed = 2 :=
by
  sorry

end hot_sauce_container_size_l65_65585


namespace last_integer_in_sequence_div3_l65_65377

theorem last_integer_in_sequence_div3 (a0 : ℤ) (sequence : ℕ → ℤ)
  (h0 : a0 = 1000000000)
  (h_seq : ∀ n, sequence n = a0 / (3^n)) :
  ∃ k, sequence k = 2 ∧ ∀ m, sequence m < 2 → sequence m < 1 := 
sorry

end last_integer_in_sequence_div3_l65_65377


namespace homothety_maps_circles_l65_65640

-- Defining the circumcircle function
def circumcircle (t : Triangle) : Circle := sorry

-- Declaring the transform 'h_G_m2' for the homothety
def h_G_m2 (G : Point) (r : ℝ) (p : Point) : Point := sorry 

-- Defining points and transformations
variables (G : Point) (A B C I_A I_B I_C : Point)
variable h_G_m2_ratio : ℝ
variable (t₁ : Triangle) [circumcircle t₁ = Circle] 
variable (t₂ : Triangle) (r = -1/2) 

-- Main proof statement
theorem homothety_maps_circles :
  h_G_m2 G r (circumcircle t₁) = circumcircle t₂ :=
sorry

end homothety_maps_circles_l65_65640


namespace miles_total_instruments_l65_65253

theorem miles_total_instruments :
  let fingers := 10
  let hands := 2
  let heads := 1
  let trumpets := fingers - 3
  let guitars := hands + 2
  let trombones := heads + 2
  let french_horns := guitars - 1
  (trumpets + guitars + trombones + french_horns) = 17 :=
by
  sorry

end miles_total_instruments_l65_65253


namespace divisible_by_8640_l65_65284

theorem divisible_by_8640 (x : ℤ) : 8640 ∣ (x^9 - 6 * x^7 + 9 * x^5 - 4 * x^3) :=
  sorry

end divisible_by_8640_l65_65284


namespace systematic_sampling_first_group_l65_65677

theorem systematic_sampling_first_group
  (a : ℕ → ℕ)
  (d : ℕ)
  (n : ℕ)
  (a₁ : ℕ)
  (a₁₆ : ℕ)
  (h₁ : d = 8)
  (h₂ : a 16 = a₁₆)
  (h₃ : a₁₆ = 125)
  (h₄ : a n = a₁ + (n - 1) * d) :
  a 1 = 5 :=
by
  sorry

end systematic_sampling_first_group_l65_65677


namespace problem_solution_l65_65194

-- Definitions and assumptions
variables (priceA priceB : ℕ)
variables (numBooksA numBooksB totalBooks : ℕ)
variables (costPriceA : priceA = 45)
variables (costPriceB : priceB = 65)
variables (totalCost : priceA * numBooksA + priceB * numBooksB ≤ 3550)
variables (totalBooksEq : numBooksA + numBooksB = 70)

-- Proof problem
theorem problem_solution :
  priceA = 45 ∧ priceB = 65 ∧ ∃ (numBooksA : ℕ), numBooksA ≥ 50 :=
by
  sorry

end problem_solution_l65_65194


namespace max_value_of_expression_l65_65646

noncomputable def problem_statement (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x^2 - x * y + 2 * y^2 = 8

theorem max_value_of_expression (x y : ℝ) (h : problem_statement x y) :
  ∃ a b c d : ℕ, a + b + c + d = 113 ∧
                  (x^2 + x * y + 2 * y^2) = (72 + 32 * real.sqrt 2) / 7 :=
sorry

end max_value_of_expression_l65_65646


namespace find_bottle_price_l65_65951

theorem find_bottle_price 
  (x : ℝ) 
  (promotion_free_bottles : ℝ := 3)
  (discount_per_bottle : ℝ := 0.6)
  (box_price : ℝ := 26)
  (box_bottles : ℝ := 4) :
  ∃ x : ℝ, (box_price / (x - discount_per_bottle)) - (box_price / x) = promotion_free_bottles :=
sorry

end find_bottle_price_l65_65951


namespace price_reduction_equation_l65_65721

theorem price_reduction_equation (x : ℝ) : 200 * (1 - x) ^ 2 = 162 :=
by
  sorry

end price_reduction_equation_l65_65721


namespace majority_of_votes_l65_65179

theorem majority_of_votes (total_valid_votes : ℝ) (percentage_winning : ℝ) (majority : ℝ) 
  (h1 : total_valid_votes = 430)
  (h2 : percentage_winning = 0.70)
  (h3 : majority = 172) : 
  ∃ W L : ℝ, 
    W = percentage_winning * total_valid_votes ∧
    L = (1 - percentage_winning) * total_valid_votes ∧
    W - L = majority :=
by
  obtain ⟨W, hW⟩ : ∃ W, W = percentage_winning * total_valid_votes := ⟨percentage_winning * total_valid_votes, rfl⟩
  obtain ⟨L, hL⟩ : ∃ L, L = (1 - percentage_winning) * total_valid_votes := ⟨(1 - percentage_winning) * total_valid_votes, rfl⟩
  use W, L
  split; try {assumption}
  calc
    W - L = percentage_winning * total_valid_votes - (1 - percentage_winning) * total_valid_votes : by rw [hW, hL]
    ... = (percentage_winning - (1 - percentage_winning)) * total_valid_votes : by ring
    ... = (0.7 - 0.3) * 430 : by rw [h1, h2]
    ... = 172 : by norm_num

end majority_of_votes_l65_65179


namespace standard_equation_of_hyperbola_l65_65063

theorem standard_equation_of_hyperbola :
  ∃ a b c : ℝ, (c / a = sqrt (5 : ℝ) / 2) ∧ ((9 / a^2) - (2 / b^2) = 1) ∧ (a^2 + b^2 = c^2) 
  ∧ (3 / a)^2 - ((-sqrt 2) / b)^2 = 1 
  ∧ (a = 1) ∧ (b = 1 / 2) → (∀ x y, (x^2 / 1) - (y^2 / (1 / 4)) = 1) := sorry

end standard_equation_of_hyperbola_l65_65063


namespace sum_first_six_terms_l65_65066

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_six_terms :
  geometric_series_sum (1/4) (1/4) 6 = 4095 / 12288 :=
by 
  sorry

end sum_first_six_terms_l65_65066


namespace angle_between_vectors_l65_65822

open Real

variables {V : Type*} [InnerProductSpace ℝ V] (a b : V)

theorem angle_between_vectors 
  (h1 : ∥a + b∥ = ∥a - b∥) (h2 : ∥a + b∥ = 2 * ∥a∥) :
  angle (a - b) b = 5 * π / 6 :=
sorry

end angle_between_vectors_l65_65822


namespace range_of_m_l65_65940

theorem range_of_m {m : ℝ} (h₀ : 0 < m) :
  (∀ (x y : ℝ), 0 < x → 0 < y → 
    (x + y + sqrt (x^2 + y^2 + xy) > m * sqrt (xy)) ∧
    (x + y + m * sqrt (xy) > sqrt (x^2 + y^2 + xy)) ∧
    (sqrt (x^2 + y^2 + xy) + m * sqrt (xy) > x + y)) →
  2 - Real.sqrt 3 < m ∧ m < 2 + Real.sqrt 3 :=
by
  sorry

end range_of_m_l65_65940


namespace mold_growth_half_surface_l65_65996

-- Problem Statement: 
-- The growth rate of a certain mold doubles every day.
-- The mold can cover the entire surface of a tank after 14 days.
-- Prove that it takes 13 days to cover half of the tank's surface.

theorem mold_growth_half_surface (n : ℕ) (A : ℝ) (h_growth : ∀ k, A k * 2 = A (k + 1)) 
    (h_full_cover : A 14 = 1) : 
    ∃ d, d = 13 ∧ A d = 1 / 2 :=
by
  sorry

end mold_growth_half_surface_l65_65996


namespace ryan_chinese_learning_hours_l65_65439

variable (hours_english : ℕ)
variable (days : ℕ)
variable (total_hours : ℕ)

theorem ryan_chinese_learning_hours (h1 : hours_english = 6) 
                                    (h2 : days = 5) 
                                    (h3 : total_hours = 65) : 
                                    total_hours - (hours_english * days) / days = 7 := by
  sorry

end ryan_chinese_learning_hours_l65_65439


namespace cone_lateral_surface_area_l65_65835

-- Definitions and conditions
def radius (r : ℝ) := r = 3
def slant_height (l : ℝ) := l = 5
def lateral_surface_area (A : ℝ) (C : ℝ) (l : ℝ) := A = 0.5 * C * l
def circumference (C : ℝ) (r : ℝ) := C = 2 * Real.pi * r

-- Proof (statement only)
theorem cone_lateral_surface_area :
  ∀ (r l C A : ℝ), 
    radius r → 
    slant_height l → 
    circumference C r → 
    lateral_surface_area A C l → 
    A = 15 * Real.pi := 
by intros; sorry

end cone_lateral_surface_area_l65_65835


namespace mailman_junk_mail_l65_65327

/-- 
  Given:
    - n = 640 : total number of pieces of junk mail for the block
    - h = 20 : number of houses in the block
  
  Prove:
    - The number of pieces of junk mail given to each house equals 32, when the total number of pieces of junk mail is divided by the number of houses.
--/
theorem mailman_junk_mail (n h : ℕ) (h_total : n = 640) (h_houses : h = 20) :
  n / h = 32 :=
by
  sorry

end mailman_junk_mail_l65_65327


namespace flagpole_perpendicular_to_ground_l65_65955

def flagpole_ground_relationship (flagpole : ℝ^3) (ground_plane : set ℝ^3) : Prop :=
  ∃ p : ℝ^3, ∀ x : ℝ^3, x ∈ ground_plane → p ≠ x ∧ (flagpole - p) ⬝ (x - p) = 0

axiom flagpole_properties (flagpole : ℝ^3) (ground_plane : set ℝ^3) 
  (h1 : flagpole ≠ (0 : ℝ^3)) : 
  ∀ p : ℝ^3, p ∈ ground_plane → (flagpole ⬝ (p - ⟨0, 0, 0⟩)) = 0
  
theorem flagpole_perpendicular_to_ground (flagpole : ℝ^3) (ground_plane : set ℝ^3) 
  (h1 : flagpole ≠ (0 : ℝ^3)) (h2 : ∀ p : ℝ^3, p ∈ ground_plane → (flagpole ⬝ (p - ⟨0, 0, 0⟩)) = 0) :
  flagpole_ground_relationship flagpole ground_plane :=
by
  sorry

end flagpole_perpendicular_to_ground_l65_65955


namespace no_solutions_rebus_l65_65970

theorem no_solutions_rebus : ∀ (K U S Y : ℕ), 
  (K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y) →
  (∀ d, d < 10) → 
  let KUSY := 1000 * K + 100 * U + 10 * S + Y in
  let UKSY := 1000 * U + 100 * K + 10 * S + Y in
  let result := 10000 * U + 1000 * K + 100 * S + 10 * Y + S in
  KUSY + UKSY ≠ result :=
begin
  sorry
end

end no_solutions_rebus_l65_65970


namespace twenty_five_percent_M_eq_thirty_five_percent_1504_l65_65145

theorem twenty_five_percent_M_eq_thirty_five_percent_1504 (M : ℝ) : 
  0.25 * M = 0.35 * 1504 → M = 2105.6 :=
by
  sorry

end twenty_five_percent_M_eq_thirty_five_percent_1504_l65_65145


namespace prove_correct_options_l65_65487

theorem prove_correct_options (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 2) :
  (min (((1 : ℝ) / x) + (1 / y)) = 2) ∧
  (max (x * y) = 1) ∧
  (min (x^2 + y^2) = 2) ∧
  (max (x * (y + 1)) = (9 / 4)) :=
by
  sorry

end prove_correct_options_l65_65487


namespace exists_monotonic_increasing_divergent_sequence_with_non_zero_limit_l65_65471

theorem exists_monotonic_increasing_divergent_sequence_with_non_zero_limit :
  ∃ (a : ℕ → ℝ), 
    (∀ n, a n < a (n + 1)) ∧ 
    tendsto (λ n, a n) atTop atTop ∧
    tendsto (λ n, (list.prod (list.map a (list.range (n + 1))) / 2^(a n) : ℝ)) atTop (nhds (1 / 2)) :=
sorry

end exists_monotonic_increasing_divergent_sequence_with_non_zero_limit_l65_65471


namespace remainder_when_2x_divided_by_7_l65_65695

theorem remainder_when_2x_divided_by_7 (x y r : ℤ) (h1 : x = 10 * y + 3)
    (h2 : 2 * x = 7 * (3 * y) + r) (h3 : 11 * y - x = 2) : r = 1 := by
  sorry

end remainder_when_2x_divided_by_7_l65_65695


namespace lattice_points_on_line_segment_l65_65422

theorem lattice_points_on_line_segment :
  ∀ (x1 y1 x2 y2 : ℤ), 
    x1 = 4 → y1 = 19 → x2 = 39 → y2 = 239 → 
    let dx := x2 - x1 in
    let dy := y2 - y1 in
    let g := Int.gcd dx dy in
    ∃ (t : ℕ), (0 ≤ t) ∧ (t ≤ g) ∧ (x1 + t * (dx / g) ∈ (0..=dx)) ∧ (y1 + t * (dy / g) ∈ (0..=dy)) → t = 6 :=
begin
  intros x1 y1 x2 y2 h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  let dx := 39 - 4,
  let dy := 239 - 19,
  let g := Int.gcd dx dy,
  use 6, 
  sorry
end

end lattice_points_on_line_segment_l65_65422


namespace ellipse_other_x_intercept_l65_65024

-- Define the ellipse with given conditions
def is_ellipse (F1 F2 : (ℝ × ℝ)) (P : ℝ × ℝ) (length : ℝ) : Prop :=
  let d1 := Real.sqrt ((F1.1 - P.1)^2 + (F1.2 - P.2)^2)
  let d2 := Real.sqrt ((F2.1 - P.1)^2 + (F2.2 - P.2)^2)
  d1 + d2 = length

-- Given conditions for the ellipse
def F1 := (0, 3) : ℝ × ℝ
def F2 := (4, 0) : ℝ × ℝ
def P1 := (0, 0) : ℝ × ℝ
def length := Real.sqrt ((F1.1 - P1.1)^2 + (F1.2 - P1.2)^2) + Real.sqrt ((F2.1 - P1.1)^2 + (F2.2 - P1.2)^2)

-- Prove that the other x-intercept is (56/11, 0)
theorem ellipse_other_x_intercept : 
  is_ellipse F1 F2 (56 / 11, 0) length :=
sorry

end ellipse_other_x_intercept_l65_65024


namespace ordered_pair_arith_progression_l65_65645

/-- 
Suppose (a, b) is an ordered pair of integers such that the three numbers a, b, and ab 
form an arithmetic progression, in that order. Prove the sum of all possible values of a is 8.
-/
theorem ordered_pair_arith_progression (a b : ℤ) (h : ∃ (a b : ℤ), (b - a = ab - b)) : 
  (a = 0 ∨ a = 1 ∨ a = 3 ∨ a = 4) → a + (if a = 0 then 1 else 0) + 
  (if a = 1 then 1 else 0) + (if a = 3 then 3 else 0) + (if a = 4 then 4 else 0) = 8 :=
by
  sorry

end ordered_pair_arith_progression_l65_65645


namespace square_triangle_area_ratio_l65_65014

theorem square_triangle_area_ratio (s t : ℝ) 
  (h1 : s^2 = (t^2 * real.sqrt 3) / 4) : 
  s / t = real.sqrt (real.sqrt 3) / 2 :=
by sorry

end square_triangle_area_ratio_l65_65014


namespace find_f_x_l65_65496

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x - 1) = x^2 - x) : ∀ x : ℝ, f x = (1/4) * (x^2 - 1) := 
sorry

end find_f_x_l65_65496


namespace find_ratio_l65_65580

-- Define the points A, B, C, M, E, F, G as elements in ℝ^2 or ℝ^3 assuming a 2D or 3D plane.
variables {A B C M E F G : Type*}
variables [add_comm_group A] [module ℝ A]

-- Define midpoints and segment relationships as per the problem.
def is_midpoint (M B C : A) : Prop := 2 • M = B + C
def same_segment (A X B : A) : Prop := ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = (1-t) • A + t • B

-- Conditions based on the problem statement
axiom h1 : is_midpoint M B C
axiom h2 : ∃ (A B : A), dist A B = 15
axiom h3 : ∃ (A C : A), dist A C = 21
axiom h4 : same_segment A E C
axiom h5 : same_segment A F B
axiom h6 : ∃ (G : A), ∃ (FE : set A), FE = {X | same_segment E X F} ∧ G ∈ FE ∧ G ∈ {X | A -ᵥ X = a • M for some a : ℝ}
axiom h7 : ∃ (x : ℝ), E = (1-3x) • A + 3x • F 

-- Question in Lean
theorem find_ratio (A B C M E F G : A) [add_comm_group A] [module ℝ A]
    (h1 : is_midpoint M B C)
    (h2 : ∃ (A B : A), dist A B = 15)
    (h3 : ∃ (A C : A), dist A C = 21)
    (h4 : same_segment A E C)
    (h5 : same_segment A F B)
    (h6 : ∃ (G : A), ∃ (FE : set A), FE = {X | same_segment E X F} ∧ G ∈ FE ∧ G ∈ {X | A -ᵥ X = a • M for some a : ℝ})
    (h7 : ∃ (x : ℝ), E = (1-3x) • A + 3x • F): 
    ∃ (r : ℝ), r = 7/2 ∧ r = dist E G / dist G F :=
sorry

end find_ratio_l65_65580


namespace garden_area_difference_l65_65398
-- Import the entire Mathlib

-- Lean Statement
theorem garden_area_difference :
  let length_Alice := 15
  let width_Alice := 30
  let length_Bob := 18
  let width_Bob := 28
  let area_Alice := length_Alice * width_Alice
  let area_Bob := length_Bob * width_Bob
  let difference := area_Bob - area_Alice
  difference = 54 :=
by
  sorry

end garden_area_difference_l65_65398


namespace minimum_value_of_f_l65_65369

def f (x y : ℝ) : ℝ := x^2 + 6y^2 - 2 * x * y - 14 * x - 6 * y + 72

theorem minimum_value_of_f : ∃ (x y : ℝ), f x y = 3 ∧ ∀ (a b : ℝ), f a b ≥ 3 := 
by 
  sorry

end minimum_value_of_f_l65_65369


namespace calculate_annual_rent_l65_65258

-- Defining the conditions
def num_units : ℕ := 100
def occupancy_rate : ℚ := 3 / 4
def monthly_rent : ℚ := 400

-- Defining the target annual rent
def annual_rent (units : ℕ) (occupancy : ℚ) (rent : ℚ) : ℚ :=
  let occupied_units := occupancy * units
  let monthly_revenue := occupied_units * rent
  monthly_revenue * 12

-- Proof problem statement
theorem calculate_annual_rent :
  annual_rent num_units occupancy_rate monthly_rent = 360000 := by
  sorry

end calculate_annual_rent_l65_65258


namespace exists_j_half_for_all_j_l65_65007

def is_j_half (n j : ℕ) : Prop := 
  ∃ (q : ℕ), n = (2 * j + 1) * q + j

theorem exists_j_half_for_all_j (k : ℕ) : 
  ∃ n : ℕ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ k → is_j_half n j :=
by
  sorry

end exists_j_half_for_all_j_l65_65007


namespace probability_of_one_radio_operator_per_group_l65_65009

def total_ways_to_assign_soldiers_to_groups : ℕ := 27720
def ways_to_assign_radio_operators_to_groups : ℕ := 7560

theorem probability_of_one_radio_operator_per_group :
  (ways_to_assign_radio_operators_to_groups : ℚ) / (total_ways_to_assign_soldiers_to_groups : ℚ) = 3 / 11 := 
sorry

end probability_of_one_radio_operator_per_group_l65_65009


namespace integer_root_of_P_l65_65443

def P (x : ℤ) : ℤ := x^3 - 4 * x^2 - 8 * x + 24 

theorem integer_root_of_P :
  (∃ x : ℤ, P x = 0) ∧ (∀ x : ℤ, P x = 0 → x = 2) :=
sorry

end integer_root_of_P_l65_65443


namespace solution_count_l65_65055

noncomputable def count_solutions (θ : ℝ) : ℕ :=
if (θ > 0 ∧ θ < 2 * Real.pi) ∧ (Real.tan (5 * Real.pi * Real.cos θ) = (1 / Real.tan (5 * Real.pi * Real.sin θ)))
then 28 else 0

theorem solution_count :
  ∃ θ : ℝ, (θ ∈ (0, 2 * Real.pi)) ∧ (Real.tan (5 * Real.pi * Real.cos θ) = (1 / Real.tan (5 * Real.pi * Real.sin θ))) → 
  count_solutions θ = 28 :=
sorry

end solution_count_l65_65055


namespace sum_of_divisors_330_l65_65685

theorem sum_of_divisors_330 : (∑ d in (finset.filter (λ d, 330 % d = 0) (finset.range (330 + 1))), d) = 864 :=
by {
  sorry
}

end sum_of_divisors_330_l65_65685


namespace parabola_coeff_sum_l65_65738

theorem parabola_coeff_sum
  (a b c : ℝ)
  (h_vertex : ∀ x, g x = a * (x + 2)^2 + 6)
  (h_point : g 0 = 2):
  a + 2 * b + c = -7 :=
sorry

end parabola_coeff_sum_l65_65738


namespace roots_of_polynomial_equation_l65_65057

theorem roots_of_polynomial_equation (x : ℝ) :
  4 * x ^ 4 - 21 * x ^ 3 + 34 * x ^ 2 - 21 * x + 4 = 0 ↔ x = 4 ∨ x = 1 / 4 ∨ x = 1 :=
by
  sorry

end roots_of_polynomial_equation_l65_65057


namespace number_of_possible_values_l65_65323

noncomputable def sum_arith_progression (n a d : ℕ) : ℕ :=
  n * (a + (n - 1) * d / 2)

theorem number_of_possible_values (a : ℤ) (n : ℕ) (h1 : sum_arith_progression n a 2 = 153) (h2 : n > 1) : 
  (nat.factors 153).count n = 5 :=
sorry

end number_of_possible_values_l65_65323


namespace graph_neg_g_neg_x_l65_65310
-- Import the necessary library

-- Define the piecewise function g
def g (x : ℝ) : ℝ :=
  if h₁ : -4 ≤ x ∧ x ≤ -1 then x + 4
  else if h₂ : -1 ≤ x ∧ x ≤ 1 then -x^2 + 1
  else if h₃ : 1 ≤ x ∧ x ≤ 4 then x - 1
  else 0 -- assuming g(x) is defined as 0 outside the given intervals for completeness

-- Define the function -g(-x)
def h (x : ℝ) : ℝ :=
  -g (-x)

-- Prove that h correctly represents the piecewise function
theorem graph_neg_g_neg_x :
  ∀ (x : ℝ), 
    (4 ≤ x ∧ x ≤ 1 → h(x) = x - 4) ∧
    (-1 ≤ x ∧ x ≤ 1 → h(x) = x^2 - 1) ∧
    (-1 ≤ x ∧ x ≤ -4 → h(x) = x + 1) :=
sorry

end graph_neg_g_neg_x_l65_65310


namespace union_A_B_inter_complement_A_B_disjoint_A_C_l65_65821

variables (a : ℝ)

def A := { x : ℝ | 4 ≤ x ∧ x < 8 }
def B := { x : ℝ | 6 < x ∧ x < 9 }
def C := { x : ℝ | x > a }

theorem union_A_B : A ∪ B = { x : ℝ | 4 ≤ x ∧ x < 9 } := 
by sorry

theorem inter_complement_A_B : (set.compl A) ∩ B = { x : ℝ | 8 ≤ x ∧ x < 9 } := 
by sorry

theorem disjoint_A_C : (A ∩ C = ∅) → a ≥ 8 :=
by sorry

end union_A_B_inter_complement_A_B_disjoint_A_C_l65_65821


namespace sharpshooter_target_orders_l65_65894

theorem sharpshooter_target_orders : 
  let targets := [ ("A", 2), ("B", 3), ("C", 2), ("D", 1) ] in
  let total_targets := 8 in
  let A_targets := 2 in
  let B_targets := 3 in
  let C_targets := 2 in
  let D_targets := 1 in
  (total_targets.factorial / (A_targets.factorial * B_targets.factorial * C_targets.factorial * D_targets.factorial)) = 1680 := 
by 
  sorry

end sharpshooter_target_orders_l65_65894


namespace segments_form_triangle_l65_65356

theorem segments_form_triangle :
  ∀ (a b c : ℝ), a = 8 ∧ b = 8 ∧ c = 15 → 
    (a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  intros a b c h
  rw [← h.1, ← h.2.1, ← h.2.2]
  split
  apply lt_add_of_pos_of_le
  linarith
  linarith
  split
  apply lt_add_of_pos_of_le
  linarith
  linarith
  apply lt_add_of_pos_of_le
  linarith
  linarith

end segments_form_triangle_l65_65356


namespace product_multiple_of_4_probability_l65_65247

-- Condition representations
def maria_rolls : ℕ := 10
def karim_rolls : ℕ := 6

-- Event that the product of their rolls is a multiple of 4
def is_multiple_of_4 (x y : ℕ) : Prop := (x * y) % 4 = 0

-- Probability calculation as per conditions
def probability {α : Type} (S : Set α) (event : Set α) : ℚ :=
  (event.toFinite.card : ℚ) / (S.toFinite.card : ℚ)

def die_rolls (n : ℕ) : Set ℕ := {k | 1 ≤ k ∧ k ≤ n}

-- Define the desired probability
def desired_probability : ℚ := 11 / 30

-- Proof statement
theorem product_multiple_of_4_probability :
  probability (die_rolls maria_rolls ×ˢ die_rolls karim_rolls) {x | is_multiple_of_4 (x.fst) (x.snd)} = desired_probability :=
sorry

end product_multiple_of_4_probability_l65_65247


namespace P_symmetric_to_C_minor_axis_l65_65594

open Set

variable {F1 F2 O A B : Point}
variable {C : Point} -- Point on the ellipse
variable {CD : Line} -- Chord perpendicular to AB
variable {ellipse : Ellipse}

def is_major_axis (AB : Line) (ellipse : Ellipse) : Prop :=
  -- Definition for major axis (usually: the line passing through the foci and the center)
  ellipse.major_axis = AB

def is_foci (F1 F2 : Point) (ellipse : Ellipse) : Prop :=
  ellipse.foci = (F1, F2)

def center (ellipse : Ellipse) : Point :=
  ellipse.center

def is_on_ellipse (C : Point) (ellipse : Ellipse) : Prop :=
  ellipse.contains C

def is_perpendicular (CD : Line) (AB : Line) : Prop :=
  CD.is_perpendicular_to AB

def angle_bisector (O C D P : Point) : Prop :=
  -- Definition to capture that P is on the bisector of angle OCD
  ∃ (bisector : Line), bisector.bisects_angle O C D ∧ ellipse.contains P ∧ bisector.contains P

theorem P_symmetric_to_C_minor_axis 
  (h_major_axis : is_major_axis AB ellipse)
  (h_foci : is_foci F1 F2 ellipse)
  (h_center : O = center ellipse)
  (h_on_ellipse : is_on_ellipse C ellipse)
  (h_perpendicular : is_perpendicular CD AB) :
  ∀ P, angle_bisector O C D P → is_symmetric_about_minor_axis C P ellipse :=
sorry

end P_symmetric_to_C_minor_axis_l65_65594
