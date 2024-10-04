import Mathlib

namespace soccer_team_students_l118_118656

theorem soccer_team_students :
  ∀ (n p b m : ℕ),
    n = 25 →
    p = 10 →
    b = 6 →
    n - (p - b) = m →
    m = 21 :=
by
  intros n p b m h_n h_p h_b h_trivial
  sorry

end soccer_team_students_l118_118656


namespace tianji_horse_winning_probability_l118_118104

-- Define the setup for the horses
def horse_ordering (tianji_top tianji_middle tianji_bottom kingqi_top kingqi_middle kingqi_bottom : Type*) : Prop :=
  (tianji_top > kingqi_middle ∧ tianji_top < kingqi_top) ∧
  (tianji_middle > kingqi_bottom ∧ tianji_middle < kingqi_middle) ∧
  (tianji_bottom < kingqi_bottom)

-- Define the main theorem to prove
theorem tianji_horse_winning_probability {T : Type*} (tianji_top tianji_middle tianji_bottom kingqi_top kingqi_middle kingqi_bottom : T) :
  horse_ordering tianji_top tianji_middle tianji_bottom kingqi_top kingqi_middle kingqi_bottom →
  (3 / 9 : ℚ) = 1 / 3 :=
by
  intro h
  -- proof should be here but we omit it with sorry
  sorry

end tianji_horse_winning_probability_l118_118104


namespace sin_dihedral_angle_l118_118634

noncomputable def dihedral_sin_eq (a b c a1 b1 c1 p : ℝ) (α : ℝ) : Prop :=
  ( -- Conditions
    (a = b ∧ b = c ∧ c = a1 ∧ a1 = b1 ∧ b1 = c1) ∧ -- All edge lengths of prism are equal
    (p = (c + c1) / 2) -- Point P is the midpoint of CC1
  ) →
  ( -- Result
    sin α = sqrt 10 / 4
  )

-- Define the statement for the given proof problem in Lean 4
theorem sin_dihedral_angle (a b c a1 b1 c1 p α : ℝ) :
  dihedral_sin_eq a b c a1 b1 c1 p α :=
sorry -- Implementation of proof is omitted

end sin_dihedral_angle_l118_118634


namespace perpendicular_lambda_parallel_lambda_l118_118034

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 3)
def b : ℝ × ℝ := (-1, 2)

-- Define the vector m as a function of lambda
def m (λ : ℝ) : ℝ × ℝ := (a.1 - λ * b.1, a.2 - λ * b.2)

-- Define the vector n
def n : ℝ × ℝ := (2 * a.1 + b.1, 2 * a.2 + b.2)

-- Function to compute the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the perpendicular condition
def is_perpendicular (λ : ℝ) : Prop :=
  dot_product (m λ) n = 0

-- Define the parallel condition
def is_parallel (λ : ℝ) : Prop :=
  n.2 * (m λ).1 - n.1 * (m λ).2 = 0

-- Proofs
theorem perpendicular_lambda :
  is_perpendicular (52 / 9) :=
by
  sorry

theorem parallel_lambda :
  is_parallel (-1 / 2) :=
by
  sorry

end perpendicular_lambda_parallel_lambda_l118_118034


namespace not_harmonious_set_example_harmonious_set_odd_minimum_harmonious_set_elements_l118_118708

def harmonious_set (A : Set ℕ) : Prop :=
  ∀ a ∈ A, ∃ (S T : Set ℕ), S ∩ T = ∅ ∧ S ∪ T = A \ {a} ∧ S.sum = T.sum

-- Problem 1: Prove that {1, 2, 3, 4, 5} is not a "harmonious set".
theorem not_harmonious_set_example : ¬harmonious_set {1, 2, 3, 4, 5} :=
  sorry

-- Problem 2: Prove that if A is a "harmonious set", then the number of elements in A is odd.
theorem harmonious_set_odd (A : Set ℕ) (hA : harmonious_set A) : ∃ n, Set.card A = 2 * n + 1 :=
  sorry

-- Problem 3: Prove that the minimum number of elements in a "harmonious set" is 7.
theorem minimum_harmonious_set_elements : ∃ A : Set ℕ, harmonious_set A ∧ Set.card A = 7 :=
  sorry

end not_harmonious_set_example_harmonious_set_odd_minimum_harmonious_set_elements_l118_118708


namespace am_gm_inequality_l118_118398

theorem am_gm_inequality (a : ℕ → ℝ) (n : ℕ) (h : ∀ i, 0 < a i) (prod_eq_one : (∏ i in finset.range n, a i = 1)) :
  ∏ i in finset.range n, (2 + a i) ≥ 3^n :=
by
  sorry

end am_gm_inequality_l118_118398


namespace segment_length_l118_118319

noncomputable def distance : ℝ × ℝ → ℝ × ℝ → ℝ
| (x1, y1), (x2, y2) := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem segment_length : distance (1, 2) (8, 6) = Real.sqrt 65 :=
by
  sorry

end segment_length_l118_118319


namespace find_m_plus_n_l118_118417

-- Definitions of the given vectors and their parallel relationship
def a : ℝ × ℝ × ℝ := (2, -1, 2)
def b (m n : ℝ) : ℝ × ℝ × ℝ := (1, m, n)
def parallel (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ λ : ℝ, u = (λ * v.1, λ * v.2, λ * v.3)

-- The theorem to prove the value of m + n given the parallelism condition
theorem find_m_plus_n (m n : ℝ) (h : parallel a (b m n)) : m + n = 1 / 2 :=
by
  sorry

end find_m_plus_n_l118_118417


namespace necessary_but_not_sufficient_condition_is_purely_imaginary_l118_118735

noncomputable def is_purely_imaginary (z : ℂ) : Prop :=
  ∃ (b : ℝ), z = ⟨0, b⟩

theorem necessary_but_not_sufficient_condition_is_purely_imaginary (a b : ℝ) (h_imaginary : is_purely_imaginary (⟨a, b⟩)) : 
  (a = 0) ∧ (b ≠ 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_is_purely_imaginary_l118_118735


namespace square_diff_problem_l118_118414

theorem square_diff_problem
  (x : ℤ)
  (h : x^2 = 9801) :
  (x + 3) * (x - 3) = 9792 := 
by
  -- proof would go here
  sorry

end square_diff_problem_l118_118414


namespace hyperbola_equation_l118_118399

noncomputable def hyperbola_focus : ℝ × ℝ := (5, 0)

noncomputable def hyperbola_asymptotes : ℝ → ℝ := λ x, (4 / 3) * x

theorem hyperbola_equation :
  ∃ λ : ℝ,
    λ ≠ 0 ∧
    (∀ (x y : ℝ), (4 / 3) * x = y ∨ (4 / 3) * x = -y → 
    (x^2 / (9 * λ) - y^2 / (16 * λ) = 1)) ∧
    ((5, 0) = hyperbola_focus) →
  (∃ (a b : ℝ),
    a^2 = 9 ∧
    b^2 = 16 ∧
    a^2 + b^2 = 25 ∧
    (λ = 1) ∧
    (x^2 / 9 - y^2 / 16 = 1)) :=
begin
  sorry
end

end hyperbola_equation_l118_118399


namespace octagon_mass_l118_118713

theorem octagon_mass :
  let side_length := 1 -- side length of the original square (meters)
  let thickness := 0.3 -- thickness of the sheet (cm)
  let density := 7.8 -- density of steel (g/cm^3)
  let x := 50 * (2 - Real.sqrt 2) -- side length of the triangles (cm)
  let octagon_area := 20000 * (Real.sqrt 2 - 1) -- area of the octagon (cm^2)
  let volume := octagon_area * thickness -- volume of the octagon (cm^3)
  let mass := volume * density / 1000 -- mass of the octagon (kg), converted from g to kg
  mass = 19 :=
by
  sorry

end octagon_mass_l118_118713


namespace curve_is_circle_l118_118676

theorem curve_is_circle (r θ : ℝ) (h : r = 3 * (Real.sin θ) * (Real.cot θ)) : 
  ∃ (a b: ℝ), (a, b) = (3 / 2, 0) ∧ 
  (∃ (R: ℝ), R = 3 / 2 ∧ (∀ x y : ℝ, x^2 + y^2 = 3 * x → (x - a)^2 + y^2 = R^2)) :=
sorry

end curve_is_circle_l118_118676


namespace cone_height_l118_118012

theorem cone_height (base_area slant_height : ℝ) (h : ℝ)
  (h_base_area : base_area = Real.pi)
  (h_slant_height : slant_height = 2) :
  let r := 1 in
  (slant_height^2 = r^2 + h^2) →
  h = Real.sqrt 3 :=
by
  intro h_r_law
  sorry

end cone_height_l118_118012


namespace pyarelal_loss_l118_118316

/-
Problem statement:
Given the following conditions:
1. Ashok's capital is 1/9 of Pyarelal's.
2. Ashok experienced a loss of 12% on his investment.
3. Pyarelal's loss was 9% of his investment.
4. Their total combined loss is Rs. 2,100.

Prove that the loss incurred by Pyarelal is Rs. 1,829.32.
-/

theorem pyarelal_loss (P : ℝ) (total_loss : ℝ) (ashok_ratio : ℝ) (ashok_loss_percent : ℝ) (pyarelal_loss_percent : ℝ)
  (h1 : ashok_ratio = (1 : ℝ) / 9)
  (h2 : ashok_loss_percent = 0.12)
  (h3 : pyarelal_loss_percent = 0.09)
  (h4 : total_loss = 2100)
  (h5 : total_loss = ashok_loss_percent * (P * ashok_ratio) + pyarelal_loss_percent * P) :
  pyarelal_loss_percent * P = 1829.32 :=
by
  sorry

end pyarelal_loss_l118_118316


namespace problem_statement_l118_118737

theorem problem_statement (x y : ℝ) (h1 : 2 ^ x = 3) (h2 : log 4 (8 / 3) = y) : 
  x + 2 * y = 3 := 
sorry

end problem_statement_l118_118737


namespace triangle_area_45_45_90_l118_118633

-- Define the properties of the right triangle
structure RightTriangle :=
  (hypotenuse : ℝ)
  (one_angle : ℝ)
  (other_angle : ℝ := 90 - one_angle)
  (leg_length : ℝ := hypotenuse / Real.sqrt 2)

-- Define the area of the triangle
def TriangleArea (triangle : RightTriangle) : ℝ :=
  1 / 2 * triangle.leg_length * triangle.leg_length

-- Given statement for the problem
theorem triangle_area_45_45_90 :
  let triangle := { hypotenuse := 14, one_angle := 45 } in
  TriangleArea triangle = 49 :=
by
  let triangle := { hypotenuse := 14, one_angle := 45 }
  sorry

end triangle_area_45_45_90_l118_118633


namespace kolya_correct_valya_incorrect_l118_118306

variable (x : ℝ)

def p : ℝ := 1 / x
def r : ℝ := 1 / (x + 1)
def q : ℝ := (x - 1) / x
def s : ℝ := x / (x + 1)

theorem kolya_correct : r / (1 - s * q) = 1 / 2 := by
  sorry

theorem valya_incorrect : (q * r + p * r) / (1 - s * q) = 1 / 2 := by
  sorry

end kolya_correct_valya_incorrect_l118_118306


namespace arithmetic_sequence_sum_equals_product_l118_118694

theorem arithmetic_sequence_sum_equals_product :
  ∃ (a_1 a_2 a_3 : ℤ), (a_2 = a_1 + d) ∧ (a_3 = a_1 + 2 * d) ∧ 
    a_1 ≠ 0 ∧ (a_1 + a_2 + a_3 = a_1 * a_2 * a_3) ∧ 
    (∃ d x : ℤ, x ≠ 0 ∧ d ≠ 0 ∧ 
    ((x = 1 ∧ d = 1) ∨ (x = -3 ∧ d = 1) ∨ (x = 3 ∧ d = -1) ∨ (x = -1 ∧ d = -1))) :=
sorry

end arithmetic_sequence_sum_equals_product_l118_118694


namespace problem_statement_l118_118813

noncomputable def a : ℕ := 44
noncomputable def b : ℕ := 36
noncomputable def c : ℕ := 33

theorem problem_statement : \( \sqrt{3}+\frac{1}{\sqrt{3}} + \sqrt{11} + \frac{1}{\sqrt{11}} = \dfrac{44\sqrt{3} + 36\sqrt{11}}{33} \) ∧ a + b + c = 113 :=
by
    sorry

end problem_statement_l118_118813


namespace intersection_of_KP_are_feet_of_altitudes_l118_118497

variables {A B C H K P : Type} [point A] [point B] [point C] [point H] [point K] [point P]

-- Suppose we have a triangle ABC with height BH.
-- Points K and P are symmetric to the foot H of the altitude BH relative to sides AB and BC respectively.

theorem intersection_of_KP_are_feet_of_altitudes
    (triangle_ABC : is_triangle A B C)
    (H_on_BH : is_foot_of_altitude H B A C)
    (K_symm_H_AB : is_symmetric K H A B)
    (P_symm_H_BC : is_symmetric P H B C)
    (KP_inter_AB : intersect KP AB = some E)
    (KP_inter_BC : intersect KP BC = some F) :
    is_foot_of_altitude E A B C ∧ is_foot_of_altitude F B C A :=
sorry

end intersection_of_KP_are_feet_of_altitudes_l118_118497


namespace sum_greater_than_product_l118_118909

theorem sum_greater_than_product (S : ℕ) :
  S = (Finset.range 2015).sum ∧ ∀ a b ∈ Finset.range 2015, a + b ≥ gcd a b + lcm a b → 
  S > 2014 * Nat.gcd 2014! (2014 - 1)! := 
sorry

end sum_greater_than_product_l118_118909


namespace sqrt_k_squared_minus_pk_is_integer_l118_118482

theorem sqrt_k_squared_minus_pk_is_integer (p : ℕ) (hp : Nat.Prime p) (odd_p : p % 2 = 1) :
  ∃ k : ℕ, (√(k^2 - p * k) : ℕ) = (k^2 - p * k) ∧ k = (p + 1) ∕ 2 ^ 2 :=
begin
  -- To be proven
  sorry
end

end sqrt_k_squared_minus_pk_is_integer_l118_118482


namespace find_larger_number_l118_118557

theorem find_larger_number 
  (x y : ℤ)
  (h1 : x + y = 37)
  (h2 : x - y = 5) : max x y = 21 := 
sorry

end find_larger_number_l118_118557


namespace medial_triangle_area_l118_118452

structure Triangle (α : Type*) :=
  (A B C D E F : α)
  (midpoint_D : B + C = 2 * D)
  (midpoint_E : C + A = 2 * E)
  (midpoint_F : A + B = 2 * F)
  (area_ABC : ℝ)

noncomputable def area_medial_triangle {α : Type*} [add_group α] [module ℝ α] 
  (T : Triangle α) (h : T.area_ABC = 48) : ℝ :=
1 / 4 * T.area_ABC

theorem medial_triangle_area {α : Type*} [add_group α] [module ℝ α]
  (T : Triangle α) (h : T.area_ABC = 48) :
  area_medial_triangle T h = 12 :=
by
  sorry

end medial_triangle_area_l118_118452


namespace find_c_l118_118529

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let num := u.1 * v.1 + u.2 * v.2
  let den := v.1 * v.1 + v.2 * v.2
  let scalar := num / den
  (scalar * v.1, scalar * v.2)

theorem find_c : 
  ∃ c : ℝ, projection (-3, c) (3, 2) = (-4 * 3, -4 * 2) ∧ c = -43/2 :=
by
  exists (-43 / 2)
  simp [projection]
  -- expand the projection definition with given vectors
  have projection_eq : projection (-3, -43 / 2) (3, 2) = (-12, -8) := by
    rw [projection, prod.mk.eta]
    simp [( * ), ( + ), ( / ), Int.cast_bit1, Int.cast_mul, ofNat_one, pow_two]

  split
  · exact projection_eq
  · simp

end find_c_l118_118529


namespace blue_pairs_count_l118_118653

-- Define the problem and conditions
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def sum9_pairs : Finset (ℕ × ℕ) := { (1, 8), (2, 7), (3, 6), (4, 5), (8, 1), (7, 2), (6, 3), (5, 4) }

-- Definition for counting valid pairs excluding pairs summing to 9
noncomputable def count_valid_pairs : ℕ := 
  (faces.card * (faces.card - 2)) / 2

-- Theorem statement proving the number of valid pairs
theorem blue_pairs_count : count_valid_pairs = 24 := 
by
  sorry

end blue_pairs_count_l118_118653


namespace tank_full_capacity_l118_118046

variable (T : ℝ) -- Define T as a real number representing the total capacity of the tank.

-- The main condition: (3 / 4) * T + 5 = (7 / 8) * T
axiom condition : (3 / 4) * T + 5 = (7 / 8) * T

-- Proof statement: Prove that T = 40
theorem tank_full_capacity : T = 40 :=
by
  -- Using the given condition to derive that T = 40.
  sorry

end tank_full_capacity_l118_118046


namespace mean_of_observations_decreased_l118_118525

noncomputable def original_mean : ℕ := 200

theorem mean_of_observations_decreased (S' : ℕ) (M' : ℕ) (n : ℕ) (d : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : M' = 185)
  (h4 : S' = M' * n)
  : original_mean = (S' + d * n) / n :=
by
  rw [original_mean]
  sorry

end mean_of_observations_decreased_l118_118525


namespace tan_of_obtuse_angle_l118_118374

theorem tan_of_obtuse_angle (α : ℝ) (h_cos : Real.cos α = -1/2) (h_obtuse : π/2 < α ∧ α < π) :
  Real.tan α = -Real.sqrt 3 :=
sorry

end tan_of_obtuse_angle_l118_118374


namespace not_solvable_det_three_times_l118_118373

theorem not_solvable_det_three_times (a b c d : ℝ) (h : a * d - b * c = 5) :
  ¬∃ (x : ℝ), (3 * a + 1) * (3 * d + 1) - (3 * b + 1) * (3 * c + 1) = x :=
by {
  -- This is where the proof would go, but the problem states that it's not solvable with the given information.
  sorry
}

end not_solvable_det_three_times_l118_118373


namespace area_unpainted_region_l118_118978

theorem area_unpainted_region
  (width_board_1 : ℝ)
  (width_board_2 : ℝ)
  (cross_angle_degrees : ℝ)
  (unpainted_area : ℝ)
  (h1 : width_board_1 = 5)
  (h2 : width_board_2 = 7)
  (h3 : cross_angle_degrees = 45)
  (h4 : unpainted_area = (49 * Real.sqrt 2) / 2) : 
  unpainted_area = (width_board_2 * ((width_board_1 * Real.sqrt 2) / 2)) / 2 :=
sorry

end area_unpainted_region_l118_118978


namespace area_bounded_by_curves_is_correct_l118_118672

noncomputable def area_bounded_by_curves : ℝ :=
  ∫ x in 1..Real.exp 1, 6 / x

theorem area_bounded_by_curves_is_correct : area_bounded_by_curves = 6 :=
by
  sorry

end area_bounded_by_curves_is_correct_l118_118672


namespace axis_of_symmetry_value_at_theta_minus_pi_over_4_l118_118035

-- Definition of f(x)
def f (x : ℝ) := sin x + cos x

-- Prove the equation of the axis of symmetry for the function f(x).
theorem axis_of_symmetry (k : ℤ) : 
  ∃ k : ℤ, ∀ x : ℝ, f(x) = f(k * π + π / 4) :=
sorry

-- Given the conditions, find the value of f(θ - π / 4)
theorem value_at_theta_minus_pi_over_4 (θ : ℝ) (hθ: 0 < θ ∧ θ < π / 2) 
  (h : f(θ + π / 4) = sqrt(2) / 3) : f(θ - π / 4) = 4 / 3 :=
sorry

end axis_of_symmetry_value_at_theta_minus_pi_over_4_l118_118035


namespace leading_coefficient_of_g_l118_118168

theorem leading_coefficient_of_g (g : ℤ → ℤ) (h : ∀ x : ℤ, g(x + 1) - g(x) = 8 * x + 6) : ∃ d : ℤ, leading_coeff 4 (λ x, 4 * x * x + 4 * x + d) :=
by sorry

end leading_coefficient_of_g_l118_118168


namespace combined_loading_time_l118_118191

theorem combined_loading_time (rA rB rC : ℝ) (hA : rA = 1 / 6) (hB : rB = 1 / 8) (hC : rC = 1 / 10) :
  1 / (rA + rB + rC) = 120 / 47 := by
  sorry

end combined_loading_time_l118_118191


namespace twenty_million_in_scientific_notation_l118_118932

/-- Prove that 20 million in scientific notation is 2 * 10^7 --/
theorem twenty_million_in_scientific_notation : 20000000 = 2 * 10^7 :=
by
  sorry

end twenty_million_in_scientific_notation_l118_118932


namespace count_rectangles_on_grid_l118_118983

/-- On a 4x4 grid where points are spaced equally at 1 unit apart both horizontally and vertically,
the number of rectangles whose four vertices are points on this grid is 101. -/
theorem count_rectangles_on_grid : 
  let n := 4 in
  let points := [(i, j) | i in List.range n, j in List.range n] in
  let rectangles := { ((x1, y1), (x2, y2)) |
                      (x1, y1) ∈ points ∧ (x2, y2) ∈ points ∧ x1 < x2 ∧ y1 < y2 } in
  rectangles.count = 101 :=
by
  /- To prove that the number of rectangles is 101,
     the detailed proof would be provided here (proof omitted in this statement). -/
  sorry

end count_rectangles_on_grid_l118_118983


namespace angle_D_l118_118383

-- Define the points in the 3D space
def point := (ℝ × ℝ × ℝ)

-- Given points
def O' : point := (0, 0, 1)
def O  : point := (0, 0, 0)
def A  : point := (1, 0, 1)
def D' : point := (sqrt 3, 0, 0)

-- Function to calculate vector from two points
def vector (p1 p2 : point) : (ℝ × ℝ × ℝ) :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

-- Vectors from the given points
def vec_O'A := vector O' A
def vec_D'O := vector D' O

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def angle_between_vectors (u v : ℝ × ℝ × ℝ) : ℝ :=
  let cos_theta := (dot_product u v) / (magnitude u * magnitude v) in
  real.arccos cos_theta

-- Prove that the angle between the line D'O and O'A is 45 degrees
theorem angle_D'O_O'A :
  angle_between_vectors vec_D'O vec_O'A = real.pi / 4 :=
sorry

end angle_D_l118_118383


namespace GCF_LCM_15_21_14_20_l118_118118

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_15_21_14_20 :
  GCF (LCM 15 21) (LCM 14 20) = 35 :=
by
  sorry

end GCF_LCM_15_21_14_20_l118_118118


namespace cylinder_radius_and_volume_l118_118258

noncomputable def radius (S : ℝ) (h : ℝ) : ℝ := 
  let s : ℝ := S / (2 * π) 
  in (-(8 : ℝ) + Real.sqrt ((8 : ℝ) ^ 2 + 4 * s))/ 2

noncomputable def volume (r : ℝ) (h : ℝ) : ℝ := 
  π * r^2 * h

theorem cylinder_radius_and_volume (S : ℝ) (h : ℝ) 
  (H_S : S = 130 * π) (H_h : h = 8) : 
  radius S h = 5 ∧ volume (radius S h) h = 200 * π :=
by
  rw [H_S, H_h]
  have r_eq : radius (130 * π) 8 = 5 := 
  by
    sorry
  
  have vol_eq : volume 5 8 = 200 * π :=
  by
    sorry
  
  exact And.intro r_eq vol_eq

end cylinder_radius_and_volume_l118_118258


namespace gcd_50403_40302_l118_118674

theorem gcd_50403_40302 : Nat.gcd 50403 40302 = 1 :=
by
  sorry

end gcd_50403_40302_l118_118674


namespace total_distance_of_journey_l118_118281

variables (x v : ℝ)
variable (d : ℝ := 600)  -- d is the total distance given by the solution to be 600 miles

-- Define the conditions stated in the problem
def condition_1 := (x = 10 * v)  -- x = 10 * v (from first part of the solution)
def condition_2 := (3 * v * d - 90 * v = -28.5 * 3 * v)  -- 2nd condition translated from second part

theorem total_distance_of_journey : 
  ∀ (x v : ℝ), condition_1 x v ∧ condition_2 x v -> x = d :=
sorry

end total_distance_of_journey_l118_118281


namespace convex_polygon_diagonals_30_sides_l118_118662

theorem convex_polygon_diagonals_30_sides :
  ∀ (n : ℕ), n = 30 → ∀ (sides : ℕ), sides = n →
  let total_segments := (n * (n - 1)) / 2 in
  let diagonals := total_segments - n in
  diagonals = 405 :=
by
  intro n hn sides hs
  simp only [hn, hs]
  let total_segments := (30 * 29) / 2
  have h_total_segments : total_segments = 435 := by sorry
  let diagonals := total_segments - 30
  have h_diagonals : diagonals = 405 := by sorry
  exact h_diagonals

end convex_polygon_diagonals_30_sides_l118_118662


namespace isosceles_triangles_l118_118683

variables {Point : Type*} [MetricSpace Point]
variables (A B C D K : Point)

# Conditions
variable (h1 : dist A B = dist C D) -- Equal segments AB and CD
variable (h2 : collinear A K C) -- A, K, and C are collinear
variable (h3 : collinear B K D) -- B, K, and D are collinear
variable (h4 : parallel (line_through A C) (line_through B D)) -- AC parallel to BD

-- Goal: Prove triangles AKC and BKD are isosceles
theorem isosceles_triangles :
  is_isosceles_triangle A K C ∧ is_isosceles_triangle B K D :=
sorry

end isosceles_triangles_l118_118683


namespace expand_expression_l118_118351

variable (x y z : ℝ)

theorem expand_expression : (x + 5) * (3 * y + 2 * z + 15) = 3 * x * y + 2 * x * z + 15 * x + 15 * y + 10 * z + 75 := by
  sorry

end expand_expression_l118_118351


namespace problem_1_problem_2_l118_118377

def p (x : ℝ) : Prop := -x^2 + 6*x + 16 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 4*x + 4 - m^2 ≤ 0 ∧ m > 0

theorem problem_1 (x : ℝ) : p x → -2 ≤ x ∧ x ≤ 8 :=
by
  -- Proof goes here
  sorry

theorem problem_2 (m : ℝ) : (∀ x, p x → q x m) ∧ (∃ x, ¬ p x ∧ q x m) → m ≥ 6 :=
by
  -- Proof goes here
  sorry

end problem_1_problem_2_l118_118377


namespace determine_A_l118_118889

variable {A B : ℝ}

def f (x : ℝ) := A * x^2 - 3 * B^3
def g (x : ℝ) := B * x

theorem determine_A (hB : B ≠ 0) (h : f (g 2) = 0) : A = 3 * B / 4 := 
by sorry

end determine_A_l118_118889


namespace sqrt_multiplication_l118_118577

theorem sqrt_multiplication : (real.sqrt 2) * (real.sqrt 3) = real.sqrt 6 :=
by
  sorry

end sqrt_multiplication_l118_118577


namespace intersection_of_A_and_B_l118_118030

-- Define the set A
def A : Set ℝ := {-1, 0, 1}

-- Define the set B based on the given conditions
def B : Set ℝ := {y | ∃ x ∈ A, y = Real.cos (Real.pi * x)}

-- The main theorem to prove that A ∩ B is {-1, 1}
theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by
  sorry

end intersection_of_A_and_B_l118_118030


namespace count_3_digit_numbers_multiple_30_not_75_l118_118789

theorem count_3_digit_numbers_multiple_30_not_75 : 
  (finset.filter (λ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 30 = 0 ∧ n % 75 ≠ 0) (finset.range 1000)).card = 21 := sorry

end count_3_digit_numbers_multiple_30_not_75_l118_118789


namespace eval_f_at_10_l118_118429

def f (x : ℚ) : ℚ := (6 * x + 3) / (x - 2)

theorem eval_f_at_10 : f 10 = 63 / 8 :=
by
  sorry

end eval_f_at_10_l118_118429


namespace proportion_of_segments_l118_118739

theorem proportion_of_segments
  (a b c d : ℝ)
  (h1 : b = 3)
  (h2 : c = 4)
  (h3 : d = 6)
  (h4 : a / b = c / d) :
  a = 2 :=
by
  sorry

end proportion_of_segments_l118_118739


namespace digits_of_2_pow_100_l118_118421

theorem digits_of_2_pow_100 : 
  let n := (2 : ℝ) ^ (100 : ℝ)
  let d := ⌊Real.log10 n⌋ + 1 
  d = 31 := by
  -- Definitions are stated
  let n := (2 : ℝ) ^ (100 : ℝ)
  let d := ⌊Real.log10 n⌋ + 1
  sorry -- Proof is omitted

end digits_of_2_pow_100_l118_118421


namespace wiper_application_is_line_to_surface_l118_118934

/-- Let L represent the action from the wiper (considered as a line) -/
def L : Type := Line

/-- Let S represent the surface of the windshield -/
def S : Type := Surface

/-- The application of a car's windshield wiper cleaning the rainwater on the glass -/
def windshield_wiper_application : L → S → Prop := λ l s, l ⊆ s

theorem wiper_application_is_line_to_surface 
  (application: L → S → Prop)
  (l: L)
  (s: S)
  (h: application l s):
  application l s = (λ (l: L) (s: S), l ⊆ s) :=
by 
  sorry -- Proof is not required as per the instructions


end wiper_application_is_line_to_surface_l118_118934


namespace new_area_after_increasing_length_and_width_l118_118511

theorem new_area_after_increasing_length_and_width
  (L W : ℝ)
  (hA : L * W = 450)
  (hL' : 1.2 * L = L')
  (hW' : 1.3 * W = W') :
  (1.2 * L) * (1.3 * W) = 702 :=
by sorry

end new_area_after_increasing_length_and_width_l118_118511


namespace greatest_divisor_consistent_remainder_l118_118691

noncomputable def gcd_of_differences : ℕ :=
  Nat.gcd (Nat.gcd 1050 28770) 71670

theorem greatest_divisor_consistent_remainder :
  gcd_of_differences = 30 :=
by
  -- The proof can be filled in here...
  sorry

end greatest_divisor_consistent_remainder_l118_118691


namespace sum_a_b_l118_118808

noncomputable def product_sequence (a b : ℕ) : Prop :=
  (∏ i in finset.range (a - 1), (i + 2 + 1) / (i + 2)) = 32 

theorem sum_a_b (a b : ℕ) (h : product_sequence a b) : a + b = 127 := 
sorry

end sum_a_b_l118_118808


namespace maximum_ab_l118_118801

noncomputable def max_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x, f x = 4 * x^3 - a * x^2 - 2 * b * x + 2) (h4 : has_extreme_at f 2) : ℝ :=
if c : 2 * a + b = 24 then (2 * a * b) / 2 else 0

theorem maximum_ab {a b : ℝ} (h1 : a > 0) (h2 : b > 0) 
    (h3 : ∀ x, f x = 4 * x^3 - a * x^2 - 2 * b * x + 2) 
    (h4 : has_extreme_at f 2) : max_ab a b h1 h2 h3 h4 = 72 := sorry

end maximum_ab_l118_118801


namespace find_S13_l118_118880

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {d : ℕ}

-- Given definitions:
def arithmetic_seq (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sn_sum_first_n (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n - 1))) / 2

-- Given conditions
axiom cond1 : ∀ a d, arithmetic_seq a d
axiom cond2 : a 1 + a 11 = 18
axiom seq : a = λ n, a 0 + n * d

-- The statement we need to prove
theorem find_S13 (h_seq : arithmetic_seq a d) (h_sum : sn_sum_first_n S a) : S 13 = 117 := by 
  sorry

end find_S13_l118_118880


namespace Kolya_is_correct_Valya_is_incorrect_l118_118290

noncomputable def Kolya_probability (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

noncomputable def Valya_probability_not_losing (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

theorem Kolya_is_correct (x : ℝ) (hx : x > 0) : Kolya_probability x = 1 / 2 :=
by
  sorry

theorem Valya_is_incorrect (x : ℝ) (hx : x > 0) : Valya_probability_not_losing x = 1 / 2 :=
by
  sorry

end Kolya_is_correct_Valya_is_incorrect_l118_118290


namespace impossible_partition_l118_118095

open Finset

theorem impossible_partition : ¬(∃ (P : Finset (Finset ℕ)), 
  (∀ A ∈ P, A.nonempty ∧ A ⊆ (range 22).toFinset ∧ 
            (A.sup id = A.erase (A.sup id)).sum id) ∧ 
  (P.bUnion id = (range 22).toFinset)) :=
sorry

end impossible_partition_l118_118095


namespace problem1_problem2_l118_118085

variables (A B C D : Type) [Real.uniform_space A]
variables {a b c BD AD DC : Real}
variables (angle_ABC angle_ACB : Real)

def proof1 (h1 : b^2 = a * c) (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) : Prop :=
  BD = b

def proof2 (h3 : AD = 2 * DC) (h1 : b^2 = a * c) : Prop :=
  Real.cos angle_ABC = 7 / 12

theorem problem1 {A B C D : Type} [Real.uniform_space A]
  {a b c BD : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) :
  proof1 a b c BD angle_ABC angle_ACB h1 h2 :=
sorry

theorem problem2 {A B C D : Type} [Real.uniform_space A]
  {a b c BD AD DC : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h3 : AD = 2 * DC) :
  proof2 a b c BD angle_ABC h1 h3 :=
sorry

end problem1_problem2_l118_118085


namespace max_value_expression_l118_118216

theorem max_value_expression (s : ℝ) : 
  ∃ M, M = -3 * s^2 + 36 * s + 7 ∧ (∀ t : ℝ, -3 * t^2 + 36 * t + 7 ≤ M) :=
by
  use 115
  sorry

end max_value_expression_l118_118216


namespace math_problem_l118_118528

variables {x y : ℝ}

theorem math_problem (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x < y) : 
  (2 * y - x) = 24 - (4 * Real.sqrt 3) / 3 :=
by sorry

end math_problem_l118_118528


namespace function_range_y_eq_1_div_x_minus_2_l118_118076

theorem function_range_y_eq_1_div_x_minus_2 (x : ℝ) : (∀ y : ℝ, y = 1 / (x - 2) ↔ x ∈ {x : ℝ | x ≠ 2}) :=
sorry

end function_range_y_eq_1_div_x_minus_2_l118_118076


namespace sum_abc_eq_113_l118_118817

theorem sum_abc_eq_113 :
  (∃ (a b c : ℕ), c > 0 ∧
    (c * (√3 + 1/√3 + √11 + 1/√11) = a * √3 + b * √11) ∧ 
    (∀ c', c' > 0 ∧ (c' * (√3 + 1/√3 + √11 + 1/√11) = a * √3 + b * √11) → c' ≥ c) ∧ 
    a + b + c = 113) :=
begin
  sorry
end

end sum_abc_eq_113_l118_118817


namespace min_distance_race_l118_118178

-- Define the basic conditions and distances
def A_to_first_wall : ℝ := 400
def first_wall_to_second_wall : ℝ := 1500
def second_wall_to_B : ℝ := 600

-- Define the function to calculate the minimum running distance
def min_running_distance : ℝ :=
  let total_vertical_distance := A_to_first_wall + first_wall_to_second_wall + second_wall_to_B
  let horizontal_distance := 1500
  real.sqrt (horizontal_distance^2 + total_vertical_distance^2)

theorem min_distance_race : min_running_distance = 2915 :=
by
  have h1 : total_vertical_distance = 2500 := by 
    simp [A_to_first_wall, first_wall_to_second_wall, second_wall_to_B]
  have h2 : min_running_distance = real.sqrt (1500^2 + 2500^2) := by 
    simp [min_running_distance, h1]
  calc min_running_distance 
       = real.sqrt (1500^2 + 2500^2) : by rw [h2]
    ... = real.sqrt (2250000 + 6250000) : by norm_num
    ... = real.sqrt 8500000 : by norm_num
    ... = 2915 : by norm_num

end min_distance_race_l118_118178


namespace distance_P_to_line_l118_118939

-- Define the point P and the line equation
def P : ℝ × ℝ := (1, -4)

def line (x y : ℝ) : Prop := 4 * x + 3 * y - 2 = 0

-- Define the point-to-line distance formula
def distance_point_to_line (A B C x1 y1 : ℝ) : ℝ :=
  abs (A * x1 + B * y1 + C) / sqrt (A^2 + B^2)

-- Define the proof problem
theorem distance_P_to_line : distance_point_to_line 4 3 (-2) 1 (-4) = 2 := by
  sorry

end distance_P_to_line_l118_118939


namespace kolya_correct_valya_incorrect_l118_118312

-- Definitions from the problem conditions:
def hitting_probability_valya (x : ℝ) : ℝ := 1 / (x + 1)
def hitting_probability_kolya (x : ℝ) : ℝ := 1 / x
def missing_probability_kolya (x : ℝ) : ℝ := 1 - (1 / x)
def missing_probability_valya (x : ℝ) : ℝ := 1 - (1 / (x + 1))

-- Proof for Kolya's assertion
theorem kolya_correct (x : ℝ) (hx : x > 0) : 
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2) :=
sorry

-- Proof for Valya's assertion
theorem valya_incorrect (x : ℝ) (hx : x > 0) :
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2 ∧ (r + p - r * p) / (1 - s * q) = 1 / 2) :=
sorry

end kolya_correct_valya_incorrect_l118_118312


namespace juan_original_number_l118_118101

theorem juan_original_number (x : ℝ) (h : (3 * (x + 3) - 4) / 2 = 10) : x = 5 :=
by
  sorry

end juan_original_number_l118_118101


namespace find_f_inv_64_l118_118802

noncomputable def f : ℝ → ℝ :=
  sorry  -- We don't know the exact form of f.

axiom f_property_1 : f 5 = 2

axiom f_property_2 : ∀ x : ℝ, f (2 * x) = 2 * f x

def f_inv (y : ℝ) : ℝ :=
  sorry  -- We define the inverse function in terms of y.

theorem find_f_inv_64 : f_inv 64 = 160 :=
by {
  -- Main statement to be proved.
  sorry
}

end find_f_inv_64_l118_118802


namespace height_difference_l118_118979

noncomputable def crateA_height : ℝ := 20 * 12

noncomputable def crateB_height : ℝ := 
  let d := 6 * Real.sqrt 3
  6 + 20 * d + 6

theorem height_difference : crateA_height - crateB_height = 228 - 120 * Real.sqrt 3 := by
  have hA : crateA_height = 20 * 12 := rfl
  have hB : crateB_height = 12 + 20 * (6 * Real.sqrt 3) := rfl
  rw [hA, hB]
  ring_nf
  sorry

end height_difference_l118_118979


namespace distinct_real_number_count_l118_118117

def f (x: ℝ) : ℝ := x^2 - 2 * x

theorem distinct_real_number_count (n : ℕ) :
  { c : ℝ | f(f(f(f(c)))) = 2 }.to_finset.card = n :=
sorry

end distinct_real_number_count_l118_118117


namespace linear_transformation_normal_l118_118275

noncomputable def isNormal (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  ∃ f, ∀ x, f(x) = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (- ((x - μ) ^ 2) / (2 * σ ^ 2))

theorem linear_transformation_normal (X : ℝ → ℝ) (a b A B : ℝ)
  (hXnormal : isNormal X a b) :
  isNormal (λ x, A * X x + B) (A * a + B) (|A| * b) :=
sorry

end linear_transformation_normal_l118_118275


namespace optimal_play_results_in_draw_l118_118323

-- Define the concept of an optimal player, and a game state in Tic-Tac-Toe
structure Game :=
(board : Fin 3 × Fin 3 → Option Bool) -- Option Bool represents empty, O, or X
(turn : Bool) -- False for O's turn, True for X's turn

def draw (g : Game) : Bool :=
-- Implementation of checking for a draw will go here
sorry

noncomputable def optimal_move (g : Game) : Game :=
-- Implementation of finding the optimal move for the current player
sorry

theorem optimal_play_results_in_draw :
  ∀ (g : Game) (h : ∀ g, optimal_move g = g),
    draw (optimal_move g) = true :=
by
  -- The proof will be provided here
  sorry

end optimal_play_results_in_draw_l118_118323


namespace centroid_projection_sum_l118_118451

theorem centroid_projection_sum 
  (A B C G P Q R : Point)
  (h₁ : dist A B = 4) 
  (h₂ : dist A C = 5) 
  (h₃ : dist B C = 3)
  (G_is_centroid : centroid A B C = G)
  (P_on_BC : orthogonal_projection B C G = P)
  (Q_on_AC : orthogonal_projection A C G = Q)
  (R_on_AB : orthogonal_projection A B G = R)
  : dist G P + dist G Q + dist G R = 47 / 15 :=
sorry

end centroid_projection_sum_l118_118451


namespace polynomial_root_sum_l118_118142

theorem polynomial_root_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) (h3 : Polynomial.root (Polynomial.C 4 + Polynomial.X * Polynomial.C (-4) + Polynomial.X^2 * Polynomial.C (-1) + Polynomial.X^3) x)
(h4 : Polynomial.root (Polynomial.C 4 + Polynomial.X * Polynomial.C (-4) + Polynomial.X^2 * Polynomial.C (-1) + Polynomial.X^3) y)
: x + y + (x^3 / y^2) + (y^3 / x^2) = 174 := by sorry

end polynomial_root_sum_l118_118142


namespace part1_prove_BD_eq_b_part2_prove_cos_ABC_l118_118090

-- Definition of the problem setup
variables {a b c : ℝ}
variables {A B C : ℝ}    -- angles
variables {D : ℝ}        -- point on side AC

-- Conditions
axiom b_squared_eq_ac : b^2 = a * c
axiom BD_sin_ABC_eq_a_sin_C : D * Real.sin B = a * Real.sin C
axiom AD_eq_2DC : A = 2 * C

-- Part (1)
theorem part1_prove_BD_eq_b : D = b :=
by
  sorry

-- Part (2)
theorem part2_prove_cos_ABC :
  Real.cos B = 7 / 12 :=
by
  sorry

end part1_prove_BD_eq_b_part2_prove_cos_ABC_l118_118090


namespace full_spots_in_garage_186_l118_118267

def parking_garage : Type := 
{ stories : ℕ
, spots_per_level : ℕ
, open_spots_first : ℕ
, diff_second_first : ℕ
, diff_third_second : ℕ
, open_spots_fourth : ℕ
}

def garage : parking_garage := 
{ stories := 4
, spots_per_level := 100
, open_spots_first := 58
, diff_second_first := 2
, diff_third_second := 5
, open_spots_fourth := 31
}

def calc_full_spots (g : parking_garage) : ℕ := 
let open_spots_first := g.open_spots_first in
let open_spots_second := open_spots_first + g.diff_second_first in
let open_spots_third := open_spots_second + g.diff_third_second in
let open_spots_fourth := g.open_spots_fourth in
let total_open_spots := open_spots_first + open_spots_second + open_spots_third + open_spots_fourth in
let total_spots := g.stories * g.spots_per_level in
total_spots - total_open_spots

theorem full_spots_in_garage_186 : calc_full_spots garage = 186 := by
  sorry

end full_spots_in_garage_186_l118_118267


namespace soccer_balls_inflated_l118_118495

theorem soccer_balls_inflated :
  let n := 200 in
  let p1 := 0.60 in
  let p2 := 0.30 in
  let remaining_balls := n * (1 - p1) in
  remaining_balls * (1 - p2) = 56 := by 
  sorry

end soccer_balls_inflated_l118_118495


namespace probability_heads_9_of_12_flips_l118_118999

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l118_118999


namespace area_of_PQR_l118_118078

theorem area_of_PQR (X Y Z P Q R : Type) 
  [add_comm_group X] [module ℝ X] 
  [add_comm_group Y] [module ℝ Y] 
  [add_comm_group Z] [module ℝ Z] 
  [add_comm_group P] [module ℝ P] 
  [add_comm_group Q] [module ℝ Q] 
  [add_comm_group R] [module ℝ R] 
  [is_midpoint P X Z] 
  [is_midpoint Q X Y] 
  [is_midpoint R Y Z] 
  (hXYZ : area_of_triangle X Y Z = 30) : 
  area_of_triangle P Q R = 7.5 := 
sorry

end area_of_PQR_l118_118078


namespace leading_coefficient_four_l118_118173

noncomputable def poly_lead_coefficient (g : ℕ → ℕ → ℕ) : Prop :=
  ∀ x : ℕ, g(x + 1) - g(x) = 8 * x + 6

theorem leading_coefficient_four {g : ℕ → ℕ} 
  (h : poly_lead_coefficient g) : 
  ∃ a b c : ℕ, ∀ x : ℕ, g x = 4 * x^2 + 2 * x + c :=
begin
  sorry
end

end leading_coefficient_four_l118_118173


namespace probability_of_a_l118_118740

theorem probability_of_a
  (p : set ℕ → ℝ)
  (a b : set ℕ)
  (h_independent : ∀ s1 s2, p (s1 ∩ s2) = p s1 * p s2)
  (p_b : p b = 2 / 5)
  (p_intersection : p (a ∩ b) = 0.22857142857142856) : 
  p a = 0.5714285714285714 := by
  sorry

end probability_of_a_l118_118740


namespace geometric_mean_condition_l118_118065

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}

theorem geometric_mean_condition
  (h_arith : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) / 6 = (a 3 + a 4) / 2)
  (h_geom_pos : ∀ n, 0 < b n) :
  Real.sqrt (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) = Real.sqrt (b 3 * b 4) :=
sorry

end geometric_mean_condition_l118_118065


namespace solution_correctness_l118_118870

noncomputable def problem_statement : ℕ :=
  let conditions := {z : ℂ // |z| = 1 ∧ (z ^ 720 - z ^ 120).im = 0}
  let N := conditions.count_fun (λ z, 1)
  N % 1000

theorem solution_correctness : problem_statement = 440 :=
by {
  -- proof omitted
  sorry
}

end solution_correctness_l118_118870


namespace valid_permutation_example_l118_118354

def is_permutation (l1 l2 : List ℕ) : Prop := 
  l1 ~ l2

theorem valid_permutation_example 
  (a : ℕ → ℕ)
  (ha₀: (a 1) = 2)
  (ha₁: (a 2) = 4)
  (ha₂: (a 3) = 9)
  (ha₃: (a 4) = 5)
  (ha₄: (a 5) = 1)
  (ha₅: (a 6) = 6)
  (ha₆: (a 7) = 8)
  (ha₇: (a 8) = 3)
  (ha₈: (a 9) = 7)
  : 
  is_permutation [a 1, a 2, a 3, a 4, a 5, a 6, a 7, a 8, a 9] [1, 2, 3, 4, 5, 6, 7, 8, 9]
  ∧ (a 1 + a 2 + a 3 + a 4 = a 4 + a 5 + a 6 + a 7)
  ∧ (a 4 + a 5 + a 6 + a 7 = a 7 + a 8 + a 9 + a 1)
  ∧ (a 1^2 + a 2^2 + a 3^2 + a 4^2 = a 4^2 + a 5^2 + a 6^2 + a 7^2)
  ∧ (a 4^2 + a 5^2 + a 6^2 + a 7^2 = a 7^2 + a 8^2 + a 9^2 + a 1^2) :=
begin
  sorry
end

end valid_permutation_example_l118_118354


namespace perfect_squares_m_l118_118803

theorem perfect_squares_m (m : ℕ) (hm_pos : m > 0) (hm_min4_square : ∃ a : ℕ, m - 4 = a^2) (hm_plus5_square : ∃ b : ℕ, m + 5 = b^2) : m = 20 ∨ m = 4 :=
by
  sorry

end perfect_squares_m_l118_118803


namespace find_gx_l118_118686

noncomputable def g (x : ℝ) : ℝ :=
  -4 * x^4 + 5 * x^3 - 10 * x^2 + 11 * x - 4

theorem find_gx (x : ℝ) :
  4 * x^4 + 2 * x^2 - 7 * x + 3 + g(x) = 5 * x^3 - 8 * x^2 + 4 * x - 1 :=
by
  sorry

end find_gx_l118_118686


namespace cinematic_academy_members_l118_118192

theorem cinematic_academy_members (h1 : (1/4 : ℝ) * x ≥ 198.75) : x = 795 :=
by
  noncomputable
  sorry

end cinematic_academy_members_l118_118192


namespace coefficient_x4_of_polynomial_l118_118673

def polynomial : ℤ → ℤ :=
  2 * (λ x, x^4 - 2 * x^3) + 3 * (λ x, 2 * x^2 - 3 * x^4 + x^6) - (λ x, 5 * x^6 - 2 * x^4)

theorem coefficient_x4_of_polynomial :
  (∃ c : ℤ, ∀ x : ℤ, polynomial x = c * x^4 + ...) -- other terms not need to be listed
  ∧ c = -5 :=
by
  sorry

end coefficient_x4_of_polynomial_l118_118673


namespace log_inequality_l118_118805

theorem log_inequality {x : ℝ} (h : log 7 (log 3 (log 2 x)) = 0) : x^(-1/2) = sqrt 2 / 4 :=
by
  sorry

end log_inequality_l118_118805


namespace triangle_problem_l118_118825

open Real

theorem triangle_problem (a b S : ℝ) (A B : ℝ) (hA_cos : cos A = (sqrt 6) / 3) (hA_val : a = 3) (hB_val : B = A + π / 2):
  b = 3 * sqrt 2 ∧
  S = (3 * sqrt 2) / 2 :=
by
  sorry

end triangle_problem_l118_118825


namespace general_equation_of_line_cartesian_equation_of_curve_minimum_AB_l118_118380

open Real

-- Definitions for the problem
variable (t : ℝ) (φ θ : ℝ) (x y P : ℝ)

-- Conditions
def line_parametric := x = t * sin φ ∧ y = 1 + t * cos φ
def curve_polar := P * (cos θ)^2 = 4 * sin θ
def curve_cartesian := x^2 = 4 * y
def line_general := x * cos φ - y * sin φ + sin φ = 0

-- Proof problem statements

-- 1. Prove the general equation of line l
theorem general_equation_of_line (h : line_parametric t φ x y) : line_general φ x y :=
sorry

-- 2. Prove the cartesian coordinate equation of curve C
theorem cartesian_equation_of_curve (h : curve_polar P θ) : curve_cartesian x y :=
sorry

-- 3. Prove the minimum |AB| where line l intersects curve C
theorem minimum_AB (h_line : line_parametric t φ x y) (h_curve : curve_cartesian x y) : ∃ (min_ab : ℝ), min_ab = 4 :=
sorry

end general_equation_of_line_cartesian_equation_of_curve_minimum_AB_l118_118380


namespace max_value_inequality_l118_118903

theorem max_value_inequality : 
  ∀ (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) (m : ℝ),
  (∀ n, S_n n = (n * a_n 1 + (1 / 2) * n * (n - 1) * d) ∧
  (∀ n, a_n n ^ 2 + (S_n n ^ 2 / n ^ 2) >= m * (a_n 1) ^ 2)) → 
  m ≤ 1 / 5 := 
sorry

end max_value_inequality_l118_118903


namespace count_pos_3digit_multiples_of_30_not_75_l118_118795

theorem count_pos_3digit_multiples_of_30_not_75 : 
  let multiples_of_30 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 30 = 0}
  let multiples_of_75 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 75 = 0}
  (multiples_of_30 \ multiples_of_75).size = 24 :=
by
  sorry

end count_pos_3digit_multiples_of_30_not_75_l118_118795


namespace chocolate_cost_l118_118249

theorem chocolate_cost (cost_per_box: ℕ) (candies_per_box: ℕ) (total_candies: ℕ) (h1: cost_per_box = 8) (h2: candies_per_box = 40) (h3: total_candies = 360) : (total_candies / candies_per_box) * cost_per_box = 72 :=
by
  rw [h1, h2, h3]
  norm_num

end chocolate_cost_l118_118249


namespace stratified_sampling_selection_l118_118440

theorem stratified_sampling_selection :
  ∀ (total_schools universities middle_schools primary_schools sample_schools : ℕ),
    total_schools = 500 → 
    universities = 10 →
    middle_schools = 200 →
    primary_schools = 290 →
    sample_schools = 50 →
    let proportion_universities := (universities : ℚ) / total_schools,
        proportion_middle_schools := (middle_schools : ℚ) / total_schools,
        proportion_primary_schools := (primary_schools : ℚ) / total_schools,
        selected_universities := proportion_universities * sample_schools,
        selected_middle_schools := proportion_middle_schools * sample_schools,
        selected_primary_schools := proportion_primary_schools * sample_schools
    in selected_universities = 1 ∧ selected_middle_schools = 20 ∧ selected_primary_schools = 29 :=
by
  intros;
  sorry

end stratified_sampling_selection_l118_118440


namespace count_pos_3digit_multiples_of_30_not_75_l118_118792

theorem count_pos_3digit_multiples_of_30_not_75 : 
  let multiples_of_30 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 30 = 0}
  let multiples_of_75 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 75 = 0}
  (multiples_of_30 \ multiples_of_75).size = 24 :=
by
  sorry

end count_pos_3digit_multiples_of_30_not_75_l118_118792


namespace probability_nine_heads_l118_118984

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118984


namespace volume_of_water_flowing_per_minute_l118_118232

variable (d w r : ℝ) (V : ℝ)

theorem volume_of_water_flowing_per_minute (h1 : d = 3) 
                                           (h2 : w = 32) 
                                           (h3 : r = 33.33) : 
  V = 3199.68 :=
by
  sorry

end volume_of_water_flowing_per_minute_l118_118232


namespace fractions_integer_or_product_of_25_integer_l118_118955

theorem fractions_integer_or_product_of_25_integer :
  ∀ (f : Fin 48 → ℚ), 
    (∀ i, ∃ n d : ℤ, 2 ≤ n ∧ n ≤ 49 ∧ 2 ≤ d ∧ d ≤ 49 ∧ f i = n / d ∧ 
    ∀ j, f i ≠ f j → i ≠ j) →
    (∃ i, ∃ n : ℤ, f i = (n : ℚ) ∧ n ∈ Set.Icc 2 49) ∨ 
    (∃ S : Finset (Fin 48), S.card ≤ 25 ∧ 
    (∏ i in S, f i).denom = 1) :=
sorry

end fractions_integer_or_product_of_25_integer_l118_118955


namespace complex_z_solution_l118_118746

theorem complex_z_solution (z : ℂ) (i : ℂ) (h : i * z = 1 - i) (hi : i * i = -1) : z = -1 - i :=
by sorry

end complex_z_solution_l118_118746


namespace find_a_l118_118730

noncomputable def circle_center_a (a : ℝ) : ℝ := a
def circle_center_y : ℝ := 2
def circle_radius : ℝ := 2
def line_l (x y : ℝ) : Prop := x - y + 3 = 0
def chord_length : ℝ := 2 * real.sqrt 3

theorem find_a (a : ℝ) (h : 0 < a)
  (h_chord : (∀ (x y : ℝ), line_l x y → ((x - circle_center_a a) ^ 2 + (y - circle_center_y) ^ 2 = circle_radius ^ 2))) : 
  a = real.sqrt 2 - 1 :=
  sorry

end find_a_l118_118730


namespace circumference_of_smaller_circle_l118_118544

theorem circumference_of_smaller_circle (r R : ℝ)
  (h1 : 4 * R^2 = 784) 
  (h2 : R = (7/3) * r) :
  2 * Real.pi * r = 12 * Real.pi := 
by {
  sorry
}

end circumference_of_smaller_circle_l118_118544


namespace kavi_sold_on_wednesday_l118_118467

noncomputable def total_stock : ℕ := 600
noncomputable def sold_mon : ℕ := 25
noncomputable def sold_tue : ℕ := 70
noncomputable def sold_thu : ℕ := 110
noncomputable def sold_fri : ℕ := 145
noncomputable def unsold_percent : ℕ := 25

theorem kavi_sold_on_wednesday : 
  let sold_total : ℕ := (75 * total_stock) / 100 in
  let sold_excluding_wed : ℕ := sold_mon + sold_tue + sold_thu + sold_fri in
  sold_total - sold_excluding_wed = 100 :=
by 
  sorry

end kavi_sold_on_wednesday_l118_118467


namespace pinwheel_area_l118_118827

-- Definitions based on conditions
def GridCenter : (ℝ × ℝ) := (5, 5)
def GridSize : ℝ := 10

-- Theorem that aligns with the proof problem
theorem pinwheel_area
  (kite_center : ℝ × ℝ = GridCenter)
  (kite_vertices : List (ℝ × ℝ) := [(5, 5), (5,4), (6, 5), (6, 4)]) 
  (kites_count : ℕ := 4)
  (area_of_kite : ℝ := 1.5) : 
  kites_count * area_of_kite = 6 := by
  sorry

end pinwheel_area_l118_118827


namespace largest_value_among_expressions_l118_118734

theorem largest_value_among_expressions 
  (a1 a2 b1 b2 : ℝ)
  (h1 : 0 < a1) (h2 : a1 < a2) (h3 : a2 < 1)
  (h4 : 0 < b1) (h5 : b1 < b2) (h6 : b2 < 1)
  (ha : a1 + a2 = 1) (hb : b1 + b2 = 1) :
  a1 * b1 + a2 * b2 > a1 * a2 + b1 * b2 ∧ 
  a1 * b1 + a2 * b2 > a1 * b2 + a2 * b1 := 
sorry

end largest_value_among_expressions_l118_118734


namespace no_integer_roots_of_P_l118_118481

-- Define the polynomial and its conditions
def polynomial (a : List Int) (n : Nat) (x : Int) : Int :=
  if h : a.length >= n + 1 then
    List.sum (List.map (λ i, a.get ⟨i, Nat.lt_of_lt_of_le i.succ_pos h⟩ * (x ^ (n - i))) (List.range (n + 1)))
  else
    0

-- Conditions extracted from the problem statement
def P_odd_at_0 (a : List Int) (n : Nat) : Prop :=
  if h : a.length >= n + 1 then
    let P0 := polynomial a n 0
    P0 % 2 = 1
  else
    false

def P_odd_at_1 (a : List Int) (n : Nat) : Prop :=
  if h : a.length >= n + 1 then
    let P1 := polynomial a n 1
    P1 % 2 = 1
  else
    false

-- The main theorem to be proved
theorem no_integer_roots_of_P (a : List Int) (n : Nat) (h1 : P_odd_at_0 a n) (h2 : P_odd_at_1 a n) :
  ∀ x : Int, polynomial a n x ≠ 0 := sorry

end no_integer_roots_of_P_l118_118481


namespace solve_for_p_l118_118146

-- Conditions
def C1 (n : ℕ) : Prop := (3 : ℚ) / 4 = n / 48
def C2 (m n : ℕ) : Prop := (3 : ℚ) / 4 = (m + n) / 96
def C3 (p m : ℕ) : Prop := (3 : ℚ) / 4 = (p - m) / 160

-- Theorem to prove
theorem solve_for_p (n m p : ℕ) (h1 : C1 n) (h2 : C2 m n) (h3 : C3 p m) : p = 156 := 
by 
    sorry

end solve_for_p_l118_118146


namespace f_2015_plus_f_2016_l118_118391

noncomputable def f : ℝ → ℝ
def odd (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def f_periodic_6 (f : ℝ → ℝ) : Prop := ∀ x, f(x + 6) = f(x) + f(3)

theorem f_2015_plus_f_2016 (h_odd : odd f) (h_periodic : f_periodic_6 f) (h_f1 : f(1) = 1) : f(2015) + f(2016) = -1 :=
by
  sorry

end f_2015_plus_f_2016_l118_118391


namespace subtraction_like_terms_l118_118320

variable (a : ℝ)

theorem subtraction_like_terms : 3 * a ^ 2 - 2 * a ^ 2 = a ^ 2 :=
by
  sorry

end subtraction_like_terms_l118_118320


namespace number_of_zeros_l118_118166

noncomputable def f (x : ℝ) := log x + 2 * x - 1

theorem number_of_zeros :
  ∃! x : ℝ, 0 < x ∧ f x = 0 
:= sorry

end number_of_zeros_l118_118166


namespace johns_family_total_l118_118462

theorem johns_family_total (father_side : ℕ) (mother_side_factor : ℝ) (total : ℕ)
  (h_father : father_side = 20)
  (h_mother_factor : mother_side_factor = 0.50)
  (h_total : total = father_side + (father_side + nat.ceil (mother_side_factor * father_side))) :
  total = 50 :=
by
  rw [h_father, h_mother_factor] at h_total
  sorry

end johns_family_total_l118_118462


namespace sum_of_digits_of_k_l118_118219

noncomputable def k : ℕ := 10^30 - 54

theorem sum_of_digits_of_k : (list.sum (nat.digits 10 k) = 11) :=
by
  sorry

end sum_of_digits_of_k_l118_118219


namespace sum_distances_regular_polygon_l118_118501

theorem sum_distances_regular_polygon (n : ℕ) (a h : ℝ) (M : ℝ × ℝ)
  (h_1 h_2 ... h_n : ℝ)
  (poly_is_regular : ∀ i j, distance_to_side M i = h_i ∧ distance_to_side M j = h_j)
  (polygon_apothem : ℝ) : 
  Σi, h_i = n * h :=
sorry

end sum_distances_regular_polygon_l118_118501


namespace function_range_y_eq_1_div_x_minus_2_l118_118075

theorem function_range_y_eq_1_div_x_minus_2 (x : ℝ) : (∀ y : ℝ, y = 1 / (x - 2) ↔ x ∈ {x : ℝ | x ≠ 2}) :=
sorry

end function_range_y_eq_1_div_x_minus_2_l118_118075


namespace part_1_part_2_l118_118091

variables {A B C D : Type}
variables {a b c : ℝ} -- Side lengths of the triangle
variables {A_angle B_angle C_angle : ℝ} -- Angles of the triangle
variables {R : ℝ} -- Circumradius of the triangle

-- Assuming the given conditions:
axiom b_squared_eq_ac : b^2 = a * c
axiom bd_sin_eq_a_sin_C : ∀ {BD : ℝ}, BD * sin B_angle = a * sin C_angle
axiom ad_eq_2dc : ∀ {AD DC : ℝ}, AD = 2 * DC

-- Theorems to prove:
theorem part_1 (BD : ℝ) : BD * sin B_angle = a * sin C_angle → BD = b := by
  intros h
  sorry

theorem part_2 (AD DC : ℝ) (H : AD = 2 * DC) : cos B_angle = 7 / 12 := by
  intros h
  sorry

end part_1_part_2_l118_118091


namespace joshua_additional_cents_needed_l118_118466

def cost_of_pen_cents : ℕ := 600
def money_joshua_has_cents : ℕ := 500
def money_borrowed_cents : ℕ := 68

def additional_cents_needed (cost money has borrowed : ℕ) : ℕ :=
  cost - (has + borrowed)

theorem joshua_additional_cents_needed :
  additional_cents_needed cost_of_pen_cents money_joshua_has_cents money_borrowed_cents = 32 :=
by
  sorry

end joshua_additional_cents_needed_l118_118466


namespace junior_score_correct_l118_118056

noncomputable def junior_score (total_students : ℕ) (junior_percentage : ℝ) (average_class_score : ℝ) (average_senior_score : ℝ) : ℝ :=
  let juniors := (junior_percentage * total_students).to_nat in
  let seniors := total_students - juniors in
  let total_class_score := average_class_score * total_students in
  let total_senior_score := average_senior_score * seniors in
  (total_class_score - total_senior_score) / juniors

theorem junior_score_correct :
  ∀ (total_students : ℕ) (junior_percentage : ℝ) (average_class_score : ℝ) (average_senior_score : ℝ),
  total_students = 20 → junior_percentage = 0.3 → average_class_score = 78 →
  average_senior_score = 75 →
  junior_score total_students junior_percentage average_class_score average_senior_score = 85 :=
by
  intros total_students junior_percentage average_class_score average_senior_score h₁ h₂ h₃ h₄
  sorry

end junior_score_correct_l118_118056


namespace seq_general_term_lambda_range_l118_118759

def seq : ℕ → ℝ
| 1       := sqrt 2 + 1
| (n + 1) := (λ a_n_minus1, sqrt (n + 2) + sqrt (n + 1)) (seq n)

def b (n : ℕ) : ℝ :=
1 / (seq n)

def S (n : ℕ) : ℝ :=
∑ i in (Finset.range n).map Finset.succ, b i

theorem seq_general_term (n : ℕ) : seq (n + 1) = sqrt (n + 2) + sqrt (n + 1) :=
sorry

theorem lambda_range (λ : ℝ) :
  (∀ (n : ℕ), n ≠ 0 → sqrt (n + 2) ≤ λ * (S n + 1) ∧ λ * (S n + 1) ≤ n + 101) →
  (λ ≥ sqrt (6) / 2 ∧ λ ≤ 20) :=
sorry

end seq_general_term_lambda_range_l118_118759


namespace number_of_zeros_of_f_l118_118163

def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

theorem number_of_zeros_of_f : ∃! x : ℝ, f x = 0 := 
sorry

end number_of_zeros_of_f_l118_118163


namespace number_of_teams_l118_118450

theorem number_of_teams (x : ℕ) (h : x * (x - 1) = 90) : x = 10 :=
sorry

end number_of_teams_l118_118450


namespace find_norm_b_l118_118764

variables {ℝ : Type} [NormedAddCommGroup ℝ] [InnerProductSpace ℝ ℝ]

theorem find_norm_b (a b : ℝ) (hab : ∥a∥ = 1) (ha2b : ∥a - 2 • b∥ = real.sqrt 21)
(angle_ab : real.angle a b = real.pi / 1.5) : ∥b∥ = 2 :=
by sorry

end find_norm_b_l118_118764


namespace polynomial_condition_satisfied_l118_118698

-- Definitions as per conditions:
def p (x : ℝ) : ℝ := x^2 + 1

-- Conditions:
axiom cond1 : p 3 = 10
axiom cond2 : ∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2

-- Theorem to prove:
theorem polynomial_condition_satisfied : (p 3 = 10) ∧ (∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2) :=
by
  apply And.intro cond1
  apply cond2

end polynomial_condition_satisfied_l118_118698


namespace purchasing_plans_count_l118_118649

theorem purchasing_plans_count :
  (∃ (x y : ℕ), 15 * x + 20 * y = 360) ∧ ∀ (x y : ℕ), 15 * x + 20 * y = 360 → (x % 4 = 0) ∧ (y = 18 - (3 / 4) * x) := sorry

end purchasing_plans_count_l118_118649


namespace Kolya_correct_Valya_incorrect_l118_118301

-- Kolya's Claim (Part a)
theorem Kolya_correct (x : ℝ) (p r : ℝ) (hpr : r = 1/(x+1) ∧ p = 1/x) : 
  (p / (1 - (1 - r) * (1 - p))) = (r / (1 - (1 - r) * (1 - p))) :=
sorry

-- Valya's Claim (Part b)
theorem Valya_incorrect (x : ℝ) (p r : ℝ) (q s : ℝ) (hprs : r = 1/(x+1) ∧ p = 1/x ∧ q = 1 - p ∧ s = 1 - r) : 
  ((q * r / (1 - s * q)) + (p * r / (1 - s * q))) = 1/2 :=
sorry

end Kolya_correct_Valya_incorrect_l118_118301


namespace sin_2alpha_minus_cos_2alpha_alpha_plus_beta_l118_118394

-- Problem 1
theorem sin_2alpha_minus_cos_2alpha (α : ℝ) (h1 : Real.sin α = 2 - 4 * Real.sin (α / 2) ^ 2) :
  Real.sin (2 * α) - Real.cos (2 * α) = 7 / 5 := sorry

-- Problem 2
theorem alpha_plus_beta (α β : ℝ) (h2 : α ∈ Ioo 0 Real.pi) (h3 : β ∈ Ioo (Real.pi / 2) Real.pi)
  (h4 : 3 * Real.tan β ^ 2 - 5 * Real.tan β - 2 = 0) :
  α + β = 5 * Real.pi / 4 := sorry

end sin_2alpha_minus_cos_2alpha_alpha_plus_beta_l118_118394


namespace value_of_k_l118_118262

theorem value_of_k 
  (k : ℚ)
  (h1 : ∃ k, 3 * x + 4 * y = 12)
  (h2 : ∀ k, 3 * x + 4 * y = -8 ∧ 3 * x + 4 * y = 21)
: k = -107 / 3 :=
sorry

end value_of_k_l118_118262


namespace number_of_girls_l118_118531

theorem number_of_girls (k : ℕ) (h1 : 4 * k + 3 * k = 56) : 4 * k = 32 :=
by
  have h2 : 7 * k = 56 := by
    linarith
  have h3 : k = 8 := by
    linarith
  rw [h3]
  linarith

end number_of_girls_l118_118531


namespace no_roots_f_f_x_l118_118898

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b*x + c

theorem no_roots_f_f_x (b c : ℝ) (h : (b - 1)^2 - 4 * c < 0) :
  ∀ x : ℝ, f b c (f b c x) ≠ x :=
begin
  assume x,
  sorry
end

end no_roots_f_f_x_l118_118898


namespace ratio_area_BPD_to_ABC_l118_118473

variables {A B C M P D : Type*}
variables (AB BC AP BP AM MB : ℝ)
variables [h_M_midpoint : M = (A + B) / 2] [h_P_between : P ∈ (M + B)] [h_MD_parallel_PC : parallel M P D C]

theorem ratio_area_BPD_to_ABC 
  (h_MP_mid : A + B = 2 * M) 
  (h_PM : M < P) 
  (h_PB : P < B) 
  (h_parallel : parallel (M, D) (P, C)) : 
  let x := AP in 
  let BP := x - (1 / 2) * AB in
  let r := (BP / AB)^2 in
  r = ((x - (1 / 2) * AB) / AB)^2 :=
by
  sorry

end ratio_area_BPD_to_ABC_l118_118473


namespace cone_volume_from_half_sector_l118_118619

theorem cone_volume_from_half_sector (r : ℝ) (l : ℝ) (h : ℝ) (V : ℝ) :
  r = 3 → l = 6 → h = 3 * Real.sqrt 3 → V = (1 / 3) * Real.pi * r^2 * h → V = 9 * Real.pi * Real.sqrt 3 :=
by
  intros hr hl hh hv
  rw [hr, hl, hh, hv]
  sorry

end cone_volume_from_half_sector_l118_118619


namespace students_at_end_l118_118598

def initial_students := 11
def students_left := 6
def new_students := 42

theorem students_at_end (init : ℕ := initial_students) (left : ℕ := students_left) (new : ℕ := new_students) :
    (init - left + new) = 47 := 
by
  sorry

end students_at_end_l118_118598


namespace smallest_y_value_l118_118571

theorem smallest_y_value :
  ∃ y : ℝ, (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) ∧ ∀ z : ℝ, (3 * z ^ 2 + 33 * z - 90 = z * (z + 16)) → y ≤ z :=
sorry

end smallest_y_value_l118_118571


namespace part1_part2_l118_118407

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1 / 2| + |2 * x - 1 / 2|

theorem part1 : {x : ℝ | f x < 3} = Icc (-3/4 : ℝ) (3/4 : ℝ) := sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : (2 * a) / (a + 2) + b / (b + 1) = 1) :
  a + b ≥ 3 / 2 := sorry

end part1_part2_l118_118407


namespace tangent_perpendicular_and_monotonicity_intervals_k_value_l118_118751

noncomputable def f (x : ℝ) : ℝ := (2 * x) / real.log x

theorem tangent_perpendicular_and_monotonicity_intervals :
  ∀ (x : ℝ), (0 < x ∧ x < 1 ∨ 1 < x ∧ x < exp 1) → deriv f x < 0 :=
begin
  -- proof here
  sorry
end

theorem k_value :
  ∃ (k : ℝ), (∀ (x : ℝ), (0 < x) → f x > k / real.log x + 2 * real.sqrt x) ∧ k = 2 :=
begin
  -- proof here
  sorry
end

end tangent_perpendicular_and_monotonicity_intervals_k_value_l118_118751


namespace larger_of_two_numbers_l118_118559

  theorem larger_of_two_numbers (x y : ℕ) 
    (h₁ : x + y = 37) 
    (h₂ : x - y = 5) 
    : x = 21 :=
  sorry
  
end larger_of_two_numbers_l118_118559


namespace intersection_of_M_and_N_l118_118413

-- Define the sets M and N
def M : set ℝ := {x | -3 < x ∧ x < 2}
def N : set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

-- The statement to be proved
theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} := sorry

end intersection_of_M_and_N_l118_118413


namespace find_angle_EDC_l118_118822

variable (A B C D E : Type)
variable [LinearOrder A] [LinearOrder B] [LinearOrder C] [LinearOrder D] [LinearOrder E]

open Angle

theorem find_angle_EDC 
  (h1: ∠A = 100)
  (h2: ∠B = 50)
  (h3: ∠C = 30)
  (h4: is_angle_bisector AD ∠A)
  (h5: is_median BE AC)
  : ∠EDC = 30 := 
sorry

end find_angle_EDC_l118_118822


namespace coincide_centroids_triangles_l118_118444

variables {A B C D E F : Type} [AddCommGroup A] [VectorSpace ℝ A]

structure Hexagon (A B C D E F : Point) :=
  (midpoint_ab : Point)
  (midpoint_bc : Point)
  (midpoint_cd : Point)
  (midpoint_de : Point)
  (midpoint_ef : Point)
  (midpoint_fa : Point)

theorem coincide_centroids_triangles (hexagon : Hexagon A B C D E F)
  (AM E CN F : Point)
  (MA ME : linesegment A E)
  (M_mid : midpoint MA)
  (N_mid : midpoint ME)
  (NA NE : linesegment C F)
  (N_mid' : midpoint NA)
  (N_mid'' : midpoint NE) :
  centroid (triangle A hexagon.midpoint_fa hexagon.midpoint_de) = centroid (triangle C hexagon.midpoint_fa hexagon.midpoint_de) :=
sorry

end coincide_centroids_triangles_l118_118444


namespace james_payment_l118_118861

theorem james_payment (cost_first_100_channels : ℕ)
  (cost_next_100_channels : ℕ)
  (total_cost : ℕ)
  (james_payment : ℕ) : cost_first_100_channels = 100 →
  cost_next_100_channels = cost_first_100_channels / 2 →
  total_cost = cost_first_100_channels + cost_next_100_channels →
  james_payment = total_cost / 2 →
  james_payment = 75 := 
by
  intros h1 h2 h3 h4
  rw [h1] at h2
  rw [h2, h1] at h3
  rw [h3] at h4
  assumption
  sorry

end james_payment_l118_118861


namespace sum_due_years_l118_118935

def TD (BD BG : ℝ) : ℝ := BD - BG
def FV (BD TD : ℝ) : ℝ := BD + TD
def t (TD FV r : ℝ) : ℝ := (TD / (FV * r)) / (1 - (TD / FV))

theorem sum_due_years (BG BD : ℝ) (r : ℝ) (hBG : BG = 360) (hBD : BD = 1360) (hr : r = 0.12) :
  t (TD BD BG) (FV BD (TD BD BG)) r ≈ 6.125 := 
by 
  sorry

end sum_due_years_l118_118935


namespace max_pencils_thrown_out_l118_118250

theorem max_pencils_thrown_out (n : ℕ) : (n % 7 ≤ 6) :=
by
  sorry

end max_pencils_thrown_out_l118_118250


namespace min_max_values_l118_118941

noncomputable def f (x : ℝ) : ℝ := 1 + 3 * x - x^3

theorem min_max_values : 
  (∃ x : ℝ, f x = -1) ∧ (∃ x : ℝ, f x = 3) :=
by
  sorry

end min_max_values_l118_118941


namespace average_score_l118_118231

theorem average_score (a : Fin 8 → ℕ) (h : ∀ i, a i = i + 1) :
  1 / 28 * (Finset.sum (Finset.pairs (Finset.univ : Finset (Fin 8)))
    (λ (ij : Fin 8 × Fin 8), a ij.1 + a ij.2)) = 9 := by
  sorry

end average_score_l118_118231


namespace sum_fractions_l118_118020

def f (x : ℝ) : ℝ := (4^x + 1) / (4^(x - (1/2)) + 1)

theorem sum_fractions :
    (∑ k in finset.range 2013, f (k.succ / 2014)) = 3019.5 := by
    sorry

end sum_fractions_l118_118020


namespace purification_layers_required_l118_118826

theorem purification_layers_required :
  (∃ n : ℕ, (n ≥ 5) ∧ (a : ℝ) (h : a > 0),
    let impurity_after_n_layers := a * (4 / 5) ^ n 
    in impurity_after_n_layers ≤ a / 3) :=
sorry

end purification_layers_required_l118_118826


namespace max_consecutive_good_numbers_l118_118705

def sum_of_divisors (n : ℕ) : ℕ := nat.divisors n |>.sum

def is_good (n : ℕ) : Prop := nat.gcd n (sum_of_divisors n) = 1

theorem max_consecutive_good_numbers : ∀ (n : ℕ), n > 0 → n ≥ 2 → is_good n → is_good (n + 1) → is_good (n + 2) → is_good (n + 3) → is_good (n + 4) → is_good (n + 5) → False :=
by
  sorry

end max_consecutive_good_numbers_l118_118705


namespace relationship_of_numbers_l118_118534

theorem relationship_of_numbers :
  let a := Real.sqrt 2
  let b := (1 / 2) ^ 2
  let c := Real.log 2 (1 / 2)
  c < b ∧ b < a :=
by
  sorry

end relationship_of_numbers_l118_118534


namespace reflected_light_ray_line_eq_l118_118621

theorem reflected_light_ray_line_eq 
  (M : Point) (M_coords : M = ⟨-2, 3⟩)
  (P : Point) (P_coords : P = ⟨1, 0⟩) :
  ∃ L : Line, L.equation = "x + y - 1 = 0" := 
begin
  sorry
end

end reflected_light_ray_line_eq_l118_118621


namespace older_sibling_age_l118_118264

def total_bill : ℝ := 9.85
def mother_charge : ℝ := 4.95
def charge_per_year : ℝ := 0.35
def children_charge : ℝ := total_bill - mother_charge
def total_years : ℝ := children_charge / charge_per_year

def is_valid_age (age : ℝ) : Prop :=
  ∃ (twin_age : ℝ), twin_age > 0 ∧ age = total_years - 2 * twin_age ∧ age ∈ {4, 6}

theorem older_sibling_age :
  (is_valid_age 4) ∨ (is_valid_age 6) :=
sorry

end older_sibling_age_l118_118264


namespace bamboo_poles_probability_l118_118538

theorem bamboo_poles_probability :
  let lengths := [2.2, 2.3, 2.4, 2.5, 2.6],
      favorable_pairs := [(2.2, 2.4), (2.3, 2.5), (2.4, 2.6)],
      total_draws := (lengths.length * (lengths.length - 1)) / 2
  in (favorable_pairs.length / total_draws.toFloat) = 3 / 10 := sorry

end bamboo_poles_probability_l118_118538


namespace guarantee_min_points_l118_118504

-- Define points for positions
def points_for_position (pos : ℕ) : ℕ :=
  if pos = 1 then 6
  else if pos = 2 then 4
  else if pos = 3 then 2
  else 0

-- Define the maximum points
def max_points_per_race := 6
def races := 4
def max_points := max_points_per_race * races

-- Define the condition of no ties
def no_ties := true

-- Define the problem statement
theorem guarantee_min_points (no_ties: true) (h1: points_for_position 1 = 6)
  (h2: points_for_position 2 = 4) (h3: points_for_position 3 = 2)
  (h4: max_points = 24) : 
  ∃ min_points, (min_points = 22) ∧ (∀ points, (points < min_points) → (∃ another_points, (another_points > points))) :=
  sorry

end guarantee_min_points_l118_118504


namespace part_1_part_2_l118_118092

variables {A B C D : Type}
variables {a b c : ℝ} -- Side lengths of the triangle
variables {A_angle B_angle C_angle : ℝ} -- Angles of the triangle
variables {R : ℝ} -- Circumradius of the triangle

-- Assuming the given conditions:
axiom b_squared_eq_ac : b^2 = a * c
axiom bd_sin_eq_a_sin_C : ∀ {BD : ℝ}, BD * sin B_angle = a * sin C_angle
axiom ad_eq_2dc : ∀ {AD DC : ℝ}, AD = 2 * DC

-- Theorems to prove:
theorem part_1 (BD : ℝ) : BD * sin B_angle = a * sin C_angle → BD = b := by
  intros h
  sorry

theorem part_2 (AD DC : ℝ) (H : AD = 2 * DC) : cos B_angle = 7 / 12 := by
  intros h
  sorry

end part_1_part_2_l118_118092


namespace equal_sum_of_inradii_l118_118239

def triangle (a b c : Point) : Prop := 
  collinear a b c = false

variables {U A P V X Y : Point}

-- Define r_1, r_2, r_3 as the inradii of the triangles
noncomputable def r1 : ℝ := inradius (triangle U D X)
noncomputable def r2 : ℝ := inradius (triangle U A P)
noncomputable def r3 : ℝ := inradius (triangle P V Y)

theorem equal_sum_of_inradii (sq : Square) 
  (h1: triangle U D X) 
  (h2: triangle U A P)
  (h3: triangle P V Y):
  r1 = r2 + r3 := sorry

end equal_sum_of_inradii_l118_118239


namespace complement_union_A_B_is_correct_l118_118010

-- Define the set of real numbers R
def R : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | ∃ (y : ℝ), y = Real.log (x + 3) }

-- Simplified definition for A to reflect x > -3
def A_simplified : Set ℝ := { x | x > -3 }

-- Define set B
def B : Set ℝ := { x | x ≥ 2 }

-- Define the union of A and B
def union_A_B : Set ℝ := A_simplified ∪ B

-- Define the complement of the union in R
def complement_R_union_A_B : Set ℝ := R \ union_A_B

-- State the theorem
theorem complement_union_A_B_is_correct :
  complement_R_union_A_B = { x | x ≤ -3 } := by
  sorry

end complement_union_A_B_is_correct_l118_118010


namespace greatest_possible_value_of_product_of_slopes_l118_118200

theorem greatest_possible_value_of_product_of_slopes 
    (theta : ℝ) (m₁ m₂ : ℝ) (h₁ : theta = real.arctan m₁)
    (h₂ : theta = real.arctan m₂) (h₃ : m₂ = 3 * m₁) :
  (1 + m₁ * m₂ ≠ 0) → abs ((m₂ - m₁) / (1 + m₁ * m₂)) = real.sqrt 3 → 
  abs ((m₂ - m₁) / (1 + m₁ * m₂)) = real.sqrt 3 → m₁ * m₂ = 1 / 3 := 
sorry

end greatest_possible_value_of_product_of_slopes_l118_118200


namespace min_z_l118_118693

def z (x y : ℝ) : ℝ := 3 * x^2 + 5 * y^2 + 6 * x - 4 * y + 3 * x^3 + 15

theorem min_z : ∃ x y, z x y = 8.2 ∧ ∀ a b, z a b ≥ z x y :=
begin
  use [-1, 2/5],
  split,
  { sorry },
  { sorry }
end

end min_z_l118_118693


namespace number_of_pages_in_contract_l118_118920

theorem number_of_pages_in_contract (total_pages_copied : ℕ) (copies_per_person : ℕ) (number_of_people : ℕ)
  (h1 : total_pages_copied = 360) (h2 : copies_per_person = 2) (h3 : number_of_people = 9) :
  total_pages_copied / (copies_per_person * number_of_people) = 20 :=
by
  sorry

end number_of_pages_in_contract_l118_118920


namespace bank_transfer_amount_l118_118915

/-- Paul made two bank transfers. A service charge of 2% was added to each transaction.
The second transaction was reversed without the service charge. His account balance is now $307 if 
it was $400 before he made any transfers. Prove that the amount of the first bank transfer was 
$91.18. -/
theorem bank_transfer_amount (x : ℝ) (initial_balance final_balance : ℝ) (service_charge_rate : ℝ) 
  (second_transaction_reversed : Prop)
  (h_initial : initial_balance = 400)
  (h_final : final_balance = 307)
  (h_charge : service_charge_rate = 0.02)
  (h_reversal : second_transaction_reversed):
  initial_balance - (1 + service_charge_rate) * x = final_balance ↔
  x = 91.18 := 
by
  sorry

end bank_transfer_amount_l118_118915


namespace explicit_formula_for_sequence_l118_118722

theorem explicit_formula_for_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (hSn : ∀ n, S n = 2 * a n + 1) : 
  ∀ n, a n = (-2) ^ (n - 1) := 
by
  sorry

end explicit_formula_for_sequence_l118_118722


namespace Kolya_is_correct_Valya_is_incorrect_l118_118292

noncomputable def Kolya_probability (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

noncomputable def Valya_probability_not_losing (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

theorem Kolya_is_correct (x : ℝ) (hx : x > 0) : Kolya_probability x = 1 / 2 :=
by
  sorry

theorem Valya_is_incorrect (x : ℝ) (hx : x > 0) : Valya_probability_not_losing x = 1 / 2 :=
by
  sorry

end Kolya_is_correct_Valya_is_incorrect_l118_118292


namespace ferry_journey_time_difference_l118_118715

/-
  Problem statement:
  Prove that the journey of ferry Q is 1 hour longer than the journey of ferry P,
  given the following conditions:
  1. Ferry P travels for 3 hours at 6 kilometers per hour.
  2. Ferry Q takes a route that is two times longer than ferry P.
  3. Ferry P is slower than ferry Q by 3 kilometers per hour.
-/

theorem ferry_journey_time_difference :
  let speed_P := 6
  let time_P := 3
  let distance_P := speed_P * time_P
  let distance_Q := 2 * distance_P
  let speed_diff := 3
  let speed_Q := speed_P + speed_diff
  let time_Q := distance_Q / speed_Q
  time_Q - time_P = 1 :=
by
  sorry

end ferry_journey_time_difference_l118_118715


namespace number_of_valid_n_l118_118770

theorem number_of_valid_n (n : ℕ) (hn : -80 < 4^n ∧ 4^n < 80) : ({n | -80 < 4^n ∧ 4^n < 80}.toFinset.card = 4) :=
by
  sorry

end number_of_valid_n_l118_118770


namespace average_matches_is_correct_l118_118623

def matches := [1, 2, 3, 4, 5]
def members := [4, 3, 6, 2, 4]
def total_matches_played := 4 * 1 + 3 * 2 + 6 * 3 + 2 * 4 + 4 * 5
def total_members := 4 + 3 + 6 + 2 + 4
def average_matches_per_member := (total_matches_played: ℝ) / (total_members: ℝ)

theorem average_matches_is_correct : Float.round average_matches_per_member = 3 := 
by
  sorry

end average_matches_is_correct_l118_118623


namespace no_complete_divisibility_l118_118950

-- Definition of non-divisibility
def not_divides (m n : ℕ) := ¬ (m ∣ n)

theorem no_complete_divisibility (a b c d : ℕ) (h : a * d - b * c > 1) : 
  not_divides (a * d - b * c) a ∨ not_divides (a * d - b * c) b ∨ not_divides (a * d - b * c) c ∨ not_divides (a * d - b * c) d :=
by 
  sorry

end no_complete_divisibility_l118_118950


namespace sum_possible_g_values_l118_118667

theorem sum_possible_g_values :
  (∃ (b c d f g h : ℕ), b > 0 ∧ c > 0 ∧ d > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0 ∧ 
    (60 * b * c = 1080) ∧ 
    (d * 6 * f = 1080) ∧ 
    (g * h * 3 = 1080) ∧ 
    List.sum (List.filter (λ g, (∃ h, g * h * 3 = 1080)) (List.range 11)) = 48)
:=
  sorry

end sum_possible_g_values_l118_118667


namespace distance_between_points_on_number_line_l118_118513

theorem distance_between_points_on_number_line : 
  ∀ (a b : ℝ), a = -9 → b = 9 → abs(a - b) = 18 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end distance_between_points_on_number_line_l118_118513


namespace parabola_intersection_points_l118_118666

open Finset

-- Defining the conditions given in the problem
def focus := (0 : ℝ, 0 : ℝ)

def a_values := {-3, -2, -1, 0, 1, 2, 3}
def b_values := {-4, -3, -2, -1, 1, 2, 3, 4}

def directrices (a b : ℤ) := { p | ∃ x : ℝ, p = (x, ((a : ℝ) * x + (b : ℝ))) }

-- Non-intersecting parallel directrices count
def non_intersecting_parallel_count := 84

/-- Proving the number of intersection points for the given parabolas conditions. -/
theorem parabola_intersection_points : 
  2 * (choose 40 2 - non_intersecting_parallel_count) = 1392 := 
by 
  sorry

end parabola_intersection_points_l118_118666


namespace range_of_a_l118_118390

noncomputable def even_function (f : ℝ → ℝ) : Prop := 
∀ x : ℝ, f(x) = f(-x)

noncomputable def monotonically_decreasing_on_nonneg (f : ℝ → ℝ) : Prop := 
∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f(x) ≥ f(y)

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) : 
  even_function f → 
  monotonically_decreasing_on_nonneg f →
  (∀ x : ℝ, 0 ≤ x → x ≤ 1 → f(x^3 - x^2 + a) + f(-x^3 + x^2 - a) ≥ 2 * f(1)) →
  -23 / 27 ≤ a ∧ a ≤ 1 :=
by
  intro h_even h_monotone h_ineq
  sorry

end range_of_a_l118_118390


namespace g_at_3_eq_19_l118_118573

def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem g_at_3_eq_19 : g 3 = 19 := by
  sorry

end g_at_3_eq_19_l118_118573


namespace smallest_multiple_of_9_0_1_smallest_multiple_of_9_1_2_l118_118702

-- Definitions for part (a)
def is_multiple_of_9 (n : ℕ) : Prop :=
  (n % 9 = 0)

def only_uses_digits (n : ℕ) (digits : ℕ → Prop) : Prop :=
  ∀ d ∈ n.digits, digits d

def digits_0_1 (d : ℕ) : Prop :=
  d = 0 ∨ d = 1

-- Proof statement for part (a)
theorem smallest_multiple_of_9_0_1 :
  is_multiple_of_9 111111111 ∧ only_uses_digits 111111111 digits_0_1 :=
by
  sorry

-- Definitions for part (b)
def digits_1_2 (d : ℕ) : Prop :=
  d = 1 ∨ d = 2

-- Proof statement for part (b)
theorem smallest_multiple_of_9_1_2 :
  is_multiple_of_9 12222 ∧ only_uses_digits 12222 digits_1_2 :=
by
  sorry

end smallest_multiple_of_9_0_1_smallest_multiple_of_9_1_2_l118_118702


namespace angle_AMH_l118_118001

theorem angle_AMH {A B C D H M : Point} (h1 : parallelogram A B C D) 
  (h2 : angle B = 111) (h3 : dist B C = dist B D)
  (h4 : angle B H D = 90) (h5 : midpoint A B M) : 
  angle A M H = 132 :=
sorry

end angle_AMH_l118_118001


namespace binary_palindromes_upto_1988_l118_118431

def is_binary_palindrome (n : ℕ) : Prop :=
  let binary_str := toDigits 2 n;
  binary_str = binary_str.reverse

def count_binary_palindromes (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter is_binary_palindrome |>.card

theorem binary_palindromes_upto_1988 : count_binary_palindromes 1988 = 92 := by
  sorry

end binary_palindromes_upto_1988_l118_118431


namespace peter_walked_distance_l118_118916

variable (total_distance : ℝ)
variable (time_per_mile : ℝ)
variable (remaining_time : ℝ)

theorem peter_walked_distance 
  (h1 : total_distance = 2.5)
  (h2 : time_per_mile = 20)
  (h3 : remaining_time = 30) :
  let distance_per_minute := 1 / time_per_mile in
  let remaining_distance := distance_per_minute * remaining_time in
  let walked_distance := total_distance - remaining_distance in
  walked_distance = 1 :=
by
  sorry

end peter_walked_distance_l118_118916


namespace systematic_sampling_first_group_l118_118211

/-- 
    In a systematic sampling of size 20 from 160 students,
    where students are divided into 20 groups evenly,
    if the number drawn from the 15th group is 116,
    then the number drawn from the first group is 4.
-/
theorem systematic_sampling_first_group (groups : ℕ) (students : ℕ) (interval : ℕ)
  (number_from_15th : ℕ) (number_from_first : ℕ) :
  groups = 20 →
  students = 160 →
  interval = 8 →
  number_from_15th = 116 →
  number_from_first = number_from_15th - interval * 14 →
  number_from_first = 4 :=
by
  intros hgroups hstudents hinterval hnumber_from_15th hequation
  sorry

end systematic_sampling_first_group_l118_118211


namespace smallest_n_divisible_by_4_l118_118108

noncomputable def sqrt_7 : Real := Real.sqrt 7

noncomputable def a_seq : Nat → Real
| 0       => sqrt_7
| (n + 1) => 1 / (a_seq n - ⌊a_seq n⌋)

noncomputable def b_seq (n : Nat) : Int := Int.floor (a_seq n)

lemma b_seq_periodicity (n : Nat) (h : n ≥ 5) : b_seq n = b_seq (n - 4) := sorry

theorem smallest_n_divisible_by_4 :
  ∃ n > 2004, b_seq n % 4 = 0 ∧ 
  ∀ m, m > 2004 → m < n → b_seq m % 4 ≠ 0 :=
begin
  use 2005,
  split,
  { exact Nat.lt_succ_self 2004, },
  split,
  { unfold b_seq a_seq,
    -- Proof that b_seq 2005 % 4 = 0 is omitted
    sorry,
  },
  { intros m hm1 hm2,
    -- Proof by contradiction to show no smaller m satisfying conditions
    sorry,
  }
end

end smallest_n_divisible_by_4_l118_118108


namespace count_3_digit_numbers_multiple_30_not_75_l118_118788

theorem count_3_digit_numbers_multiple_30_not_75 : 
  (finset.filter (λ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 30 = 0 ∧ n % 75 ≠ 0) (finset.range 1000)).card = 21 := sorry

end count_3_digit_numbers_multiple_30_not_75_l118_118788


namespace encoded_value_is_correct_l118_118614

theorem encoded_value_is_correct : 
  ∀ A B C D E F G : ℕ,
  ∀ (hG : G = 6) (hF : F = 5) (hC: C = 6) (hB: B = 2) (hA: A = 3) (hD: D = 0), 
  let code := 6 * 7^2 + 5 * 7^1 + 0 * 7^0 in
  (code = 329) :=
by
  intros A B C D E F G hG hF hC hB hA hD
  let code := 6 * 7^2 + 5 * 7^1 + 0 * 7^0
  exact eq.refl 329

end encoded_value_is_correct_l118_118614


namespace birds_percentage_hawks_l118_118061

-- Define the conditions and the main proof problem
theorem birds_percentage_hawks (H : ℝ) :
  (0.4 * (1 - H) + 0.25 * 0.4 * (1 - H) + H = 0.65) → (H = 0.3) :=
by
  intro h
  sorry

end birds_percentage_hawks_l118_118061


namespace find_a_l118_118434

noncomputable def log_a (a: ℝ) (x: ℝ) : ℝ := Real.log x / Real.log a

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : log_a a 2 - log_a a 4 = 2) :
  a = Real.sqrt 2 / 2 :=
sorry

end find_a_l118_118434


namespace sufficient_condition_m_ge_4_range_of_x_for_m5_l118_118396

variable (x m : ℝ)

-- Problem (1)
theorem sufficient_condition_m_ge_4 (h : m > 0)
  (hpq : ∀ x, ((x + 2) * (x - 6) ≤ 0) → (2 - m ≤ x ∧ x ≤ 2 + m)) : m ≥ 4 := by
  sorry

-- Problem (2)
theorem range_of_x_for_m5 (h : m = 5)
  (hp_or_q : ∀ x, ((x + 2) * (x - 6) ≤ 0 ∨ (-3 ≤ x ∧ x ≤ 7)) )
  (hp_and_not_q : ∀ x, ¬(((x + 2) * (x - 6) ≤ 0) ∧ (-3 ≤ x ∧ x ≤ 7))):
  ∀ x, x ∈ Set.Ico (-3) (-2) ∨ x ∈ Set.Ioc (6) (7) := by
  sorry

end sufficient_condition_m_ge_4_range_of_x_for_m5_l118_118396


namespace geometric_series_sum_S20_l118_118385

-- Given conditions of the problem
def series {a : ℕ → ℝ} (a1 : a 1 = 1) (S : ℕ → ℝ) := 
∀ n : ℕ, n ≥ 1 → S n = (a n + 1)^2 / 4

-- The main theorem we need to prove
theorem geometric_series_sum_S20 (a : ℕ → ℝ) (S : ℕ → ℝ) (h : series a (λ n, (a n + 1)^2 / 4)) :
  S 20 = 400 :=
sorry

end geometric_series_sum_S20_l118_118385


namespace cot_alpha_l118_118738

-- Definitions for conditions in the problem
def angle (α : ℝ) : Prop := 180 < α ∧ α < 270

def cos_value (α : ℝ) : Prop := real.cos α = -99 / 101

-- The statement to be proven
theorem cot_alpha (α : ℝ) (h1 : angle α) (h2 : cos_value α) :
  real.cot α = 99 / 20 :=
sorry

end cot_alpha_l118_118738


namespace set_op_equivalence_l118_118881

variables (U : Type) [universal : set U]
variables (X Y Z : set U)

def complement (s : set U) : set U := universal \ s

def op (X Y : set U) : set U := (complement X) ∪ Y

theorem set_op_equivalence :
  op X (op Y Z) = ((complement X) ∪ (complement Y)) ∪ Z :=
by sorry

end set_op_equivalence_l118_118881


namespace range_of_a_l118_118890

-- Definitions of the function and conditions
def f (a x : ℝ) : ℝ := (Real.exp x) / (1 + a * x^2)

-- The proof statement
theorem range_of_a (a : ℝ) (h : 0 < a ) (mono : ∀ x : ℝ, 0 ≤ (Derivative fun x => f a x)) : 0 < a ∧ a ≤ 1 :=
sorry

end range_of_a_l118_118890


namespace three_digit_multiples_l118_118783

theorem three_digit_multiples : 
  let three_digit_range := {x : ℕ | 100 ≤ x ∧ x < 1000}
  let multiples_of_30 := {x ∈ three_digit_range | x % 30 = 0}
  let multiples_of_75 := {x ∈ three_digit_range | x % 75 = 0}
  let count_multiples_of_30 := Set.card multiples_of_30
  let count_multiples_of_75 := Set.card multiples_of_75
  let count_common_multiples := Set.card (multiples_of_30 ∩ multiples_of_75)
  count_multiples_of_30 - count_common_multiples = 24 :=
by
  sorry

end three_digit_multiples_l118_118783


namespace exists_divisible_sk_l118_118876

noncomputable def sequence_of_integers (c : ℕ) (a : ℕ → ℕ) :=
  ∀ n : ℕ, 0 < n → a n < a (n + 1) ∧ a (n + 1) < a n + c

noncomputable def infinite_string (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (10 ^ n) * (a (n + 1)) + a n

noncomputable def sk (s : ℕ) (k : ℕ) : ℕ :=
  (s % (10 ^ k))

theorem exists_divisible_sk (a : ℕ → ℕ) (c m : ℕ)
  (h : sequence_of_integers c a) :
  ∀ m : ℕ, ∃ k : ℕ, m > 0 → (sk (infinite_string a k) k) % m = 0 := by
  sorry

end exists_divisible_sk_l118_118876


namespace find_larger_number_l118_118558

theorem find_larger_number 
  (x y : ℤ)
  (h1 : x + y = 37)
  (h2 : x - y = 5) : max x y = 21 := 
sorry

end find_larger_number_l118_118558


namespace quadratic_has_two_distinct_real_roots_for_any_k_find_k_for_isosceles_triangle_with_given_bc_l118_118002

noncomputable section

-- Define the quadratic equation
def quadratic_equation (k : ℝ) : Polynomial ℝ :=
  Polynomial.coeff 2 1 + Polynomial.coeff 1 (-(2 * k + 1)) + Polynomial.coeff 0 (k^2 + k)

-- Define discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  let b := -(2 * k + 1)
  let c := k^2 + k
  b^2 - 4 * 1 * c

-- Define the isosceles triangle condition
def is_isosceles_triangle (a b : ℝ) (c : ℝ) : Prop :=
  a = c ∨ b = c

theorem quadratic_has_two_distinct_real_roots_for_any_k (k : ℝ) : discriminant k > 0 := by
  sorry

theorem find_k_for_isosceles_triangle_with_given_bc (k : ℝ) :
  let roots := quadratic_equation k
  (is_isosceles_triangle roots (Polynomial.root roots) 5) ↔ k = 4 ∨ k = 5 := by
  sorry

end quadratic_has_two_distinct_real_roots_for_any_k_find_k_for_isosceles_triangle_with_given_bc_l118_118002


namespace sum_fourth_powers_roots_l118_118246

open Real

-- Define polynomial P
noncomputable def P (x : ℝ) : ℝ := x^2 + 2 * x + 3

-- State the problem as a formal statement in Lean 4
theorem sum_fourth_powers_roots : 
  let r_1 := -1 + sqrt (1 - 3)
  let r_2 := -1 - sqrt (1 - 3)
  r_1^4 + r_2^4 = -14 := 
begin
  sorry
end

end sum_fourth_powers_roots_l118_118246


namespace vacation_animals_total_l118_118136

noncomputable def lisa := 40
noncomputable def alex := lisa / 2
noncomputable def jane := alex + 10
noncomputable def rick := 3 * jane
noncomputable def tim := 2 * rick
noncomputable def you := 5 * tim
noncomputable def total_animals := lisa + alex + jane + rick + tim + you

theorem vacation_animals_total : total_animals = 1260 := by
  sorry

end vacation_animals_total_l118_118136


namespace solve_fiftieth_term_l118_118961

variable (a₇ a₂₁ : ℤ) (d : ℚ)

-- The conditions stated in the problem
def seventh_term : a₇ = 10 := by sorry
def twenty_first_term : a₂₁ = 34 := by sorry

-- The fifty term calculation assuming the common difference d
def fiftieth_term_is_fraction (d : ℚ) : ℚ := 10 + 43 * d

-- Translate the condition a₂₁ = a₇ + 14 * d
theorem solve_fiftieth_term : a₂₁ = a₇ + 14 * d → 
                              fiftieth_term_is_fraction d = 682 / 7 := by sorry


end solve_fiftieth_term_l118_118961


namespace subset_implies_range_of_a_l118_118109

theorem subset_implies_range_of_a (a : ℝ) : 
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 5 → x > a) → a < -2 :=
by
  intro h
  sorry

end subset_implies_range_of_a_l118_118109


namespace Mason_fathers_age_indeterminate_l118_118907

theorem Mason_fathers_age_indeterminate
  (Mason_age : ℕ) (Sydney_age Mason_father_age D : ℕ)
  (hM : Mason_age = 20)
  (hS_M : Mason_age = Sydney_age / 3)
  (hS_F : Mason_father_age - D = Sydney_age) :
  ¬ ∃ F, Mason_father_age = F :=
by {
  sorry
}

end Mason_fathers_age_indeterminate_l118_118907


namespace rectangle_is_possible_l118_118540

def possibleToFormRectangle (stick_lengths : List ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (a + b) * 2 = List.sum stick_lengths

noncomputable def sticks : List ℕ := List.range' 1 99

theorem rectangle_is_possible : possibleToFormRectangle sticks :=
sorry

end rectangle_is_possible_l118_118540


namespace no_real_roots_f_of_f_eq_x_l118_118896

-- Definitions and conditions as provided above
variable {b c : ℝ}
def f (x : ℝ) : ℝ := x^2 + b * x + c

-- Define the condition about no real roots for f(x) = x
def no_real_roots_f_eq_x : Prop := (b - 1)^2 - 4 * c < 0

-- Problem statement: Prove that f(f(x)) = x has no real roots
theorem no_real_roots_f_of_f_eq_x (h : no_real_roots_f_eq_x) :
  ∀ x : ℝ, f(f(x)) ≠ x := by
  sorry

end no_real_roots_f_of_f_eq_x_l118_118896


namespace smallest_n_for_fn_eq_2n_l118_118657

def move_last_digit_to_front (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let remaining_digits := n / 10
  last_digit * 10^(remaining_digits.digits.length) + remaining_digits

theorem smallest_n_for_fn_eq_2n :
  ∃ n : ℕ, move_last_digit_to_front n = 2 * n ∧ n = 105263157894736842 :=
by
  sorry

end smallest_n_for_fn_eq_2n_l118_118657


namespace part_a_proof_part_b_proof_l118_118238

noncomputable def part_a_inequality (a b c d : ℝ) (h : a + b + c + d = 0) : Prop :=
  max a b + max a c + max a d + max b c + max b d + max c d ≥ 0

noncomputable def part_b_max_k : ℝ := 2

theorem part_a_proof (a b c d : ℝ) (h : a + b + c + d = 0) : part_a_inequality a b c d h :=
sorry

theorem part_b_proof : part_b_max_k = 2 :=
sorry

end part_a_proof_part_b_proof_l118_118238


namespace find_seventh_term_l118_118886

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- Define arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Define sum of the first n terms of the sequence
def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n * a 0) + (d * (n * (n - 1)) / 2)

-- Now state the theorem
theorem find_seventh_term
  (h_arith_seq : arithmetic_sequence a d)
  (h_nonzero_d : d ≠ 0)
  (h_sum_five : S 5 = 5)
  (h_squares_eq : a 0 ^ 2 + a 1 ^ 2 = a 2 ^ 2 + a 3 ^ 2) :
  a 6 = 9 :=
sorry

end find_seventh_term_l118_118886


namespace find_positive_integer_solutions_l118_118355

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  3 ^ x + 4 ^ y = 5 ^ z → x = 2 ∧ y = 2 ∧ z = 2 :=
by
  sorry

end find_positive_integer_solutions_l118_118355


namespace b_income_percentage_increase_l118_118161

theorem b_income_percentage_increase (A_m B_m C_m : ℕ) (annual_income_A : ℕ)
  (C_income : C_m = 15000)
  (annual_income_A_cond : annual_income_A = 504000)
  (ratio_cond : A_m / B_m = 5 / 2)
  (A_m_cond : A_m = annual_income_A / 12) :
  ((B_m - C_m) * 100 / C_m) = 12 :=
by
  sorry

end b_income_percentage_increase_l118_118161


namespace first_term_exceeds_10000_l118_118515

theorem first_term_exceeds_10000 :
  ∃ n : ℕ, n ≥ 2 ∧ (∑ i in range (n - 1), 2^i) > 10000 :=
by
  sorry

end first_term_exceeds_10000_l118_118515


namespace part1_proof_part2_proof_l118_118081

-- Definitions corresponding to the conditions in a)
variables (a b c BD : ℝ) (A B C : RealAngle)
variables (D : Point) (AD DC : ℝ)

-- Replace the conditions with the necessary hypotheses
hypothesis h1 : b^2 = a * c
hypothesis h2 : BD * sin B = a * sin C
hypothesis h3 : AD = 2 * DC

noncomputable def Part1 : Prop := BD = b

theorem part1_proof : Part1 a b c BD A B C do
  sorry

noncomputable def Part2 : Prop := cos B = 7 / 12

theorem part2_proof (hADDC : AD = 2 * DC) : Part2 A B C.
  sorry

end part1_proof_part2_proof_l118_118081


namespace function_is_x_cubed_l118_118384

variable (f : ℝ → ℝ)

def domain_condition : Prop := ∀ x1 x2 : ℝ, True

def odd_function_condition : Prop := ∀ x1 x2 : ℝ, (x1 + x2 ≠ 0) → (f x1 + f x2 = 0)

def monotonic_increasing_condition : Prop := ∀ x t : ℝ, (t > 0) → (f (x + t) > f x)

theorem function_is_x_cubed :
    domain_condition f ∧ odd_function_condition f ∧ monotonic_increasing_condition f →
    f = λ x, x^3 := 
by
  sorry

end function_is_x_cubed_l118_118384


namespace maximum_value_of_f_l118_118016

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem maximum_value_of_f : ∀ x ∈ set.Icc (1 / 2 : ℝ) 2, f x ≤ f (1 / 2) :=
by
  sorry

end maximum_value_of_f_l118_118016


namespace angle_MAN_65_l118_118240

-- Define the problem conditions
variables {A B C M N L : Type}
variable (BAC_50 : 50 = 50)  -- Denotes that ∠BAC is 50 degrees
variable (angle_bisector_AL : L ∈ segment BC)
variable (MA_eq_ML : dist M A = dist M L)
variable (NA_eq_NL : dist N A = dist N L)

-- Define the theorem to prove that ∠MAN = 65 degrees
theorem angle_MAN_65 
  (BAC_50 : ∠ BAC = 50)
  (angle_bisector_AL : is_angle_bisector A L B C)
  (MA_eq_ML : dist M A = dist M L)
  (NA_eq_NL : dist N A = dist N L):
  ∠ MAN = 65 :=
sorry  -- Proof omitted

end angle_MAN_65_l118_118240


namespace ratio_of_incomes_l118_118959

variable {I1 I2 E1 E2 S1 S2 : ℝ}

theorem ratio_of_incomes
  (h1 : I1 = 4000)
  (h2 : E1 / E2 = 3 / 2)
  (h3 : S1 = 1600)
  (h4 : S2 = 1600)
  (h5 : S1 = I1 - E1)
  (h6 : S2 = I2 - E2) :
  I1 / I2 = 5 / 4 :=
by
  sorry

end ratio_of_incomes_l118_118959


namespace necessary_but_not_sufficient_condition_l118_118627

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ a, a > 2 → a ∈ set.Ici 2) ∧ ¬(∃ a, a ∈ set.Ici 2 ∧ a ≤ 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l118_118627


namespace abc_le_one_eighth_l118_118872

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : 
  a * b * c ≤ 1 / 8 := 
by
  sorry

end abc_le_one_eighth_l118_118872


namespace rhombus_perimeter_l118_118949

-- Conditions
def shorter_diagonal : ℝ := 20
def longer_diagonal : ℝ := 1.3 * shorter_diagonal
def half_shorter_diagonal : ℝ := shorter_diagonal / 2
def half_longer_diagonal : ℝ := longer_diagonal / 2
def side_length : ℝ := Real.sqrt ((half_shorter_diagonal ^ 2) + (half_longer_diagonal ^ 2))

-- Statement to prove
theorem rhombus_perimeter : (4 * side_length) = 4 * Real.sqrt 269 :=
by
  sorry

end rhombus_perimeter_l118_118949


namespace num_int_triples_satisfying_eqs_l118_118357

theorem num_int_triples_satisfying_eqs : 
  ∃ (s : Set (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ s ↔ x * y = 6 ∧ y * z = 15) ∧ 
    Finset.card ((s.toFinset : Finset (ℕ × ℕ × ℕ))) = 2 :=
begin
  sorry
end

end num_int_triples_satisfying_eqs_l118_118357


namespace probability_reaching_3_1_in_5_steps_m_plus_n_l118_118148

def valid_paths_5_steps (s: List (ℕ × ℕ)) : ℕ := 
  let count_steps := λ p, ((p.count &1.fst - p.count_map &2.fst) = 3) ∧ 
                          ((p.count_map &2.snd - p.count_map p.fst) = 1) in 
  s.filter count_steps).length

theorem probability_reaching_3_1_in_5_steps : 
  (valid_paths_5_steps [((0, 0), (3, 1))].length) = 40 ∧
  40 * ((1/4)^5) = 5 / 128 :=
begin
  sorry,
end

theorem m_plus_n : 5 + 128 = 133 :=
begin
  sorry,
end

end probability_reaching_3_1_in_5_steps_m_plus_n_l118_118148


namespace probability_nine_heads_l118_118990

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118990


namespace angle_KLM_is_90_degrees_l118_118926

theorem angle_KLM_is_90_degrees (x : ℝ) (h_parallel : ∀ (JK LM : set 𝓡) (hJK_nm : ∀ (P Q : 𝓡³), P ∈ JK → Q ∈ JK → P ≠ Q) (hLM_nm : ∀ (P Q : 𝓡³), P ∈ LM → Q ∈ LM → P ≠ Q) (h_parallel : ∀ (P Q R S : 𝓡³), P ∈ JK → Q ∈ JK → R ∈ LM → S ∈ LM → slope_between P Q = slope_between R S), 
  angle_OML : ℝ, angle_OJK : ℝ, angle_KLM : ℝ)
  (h_angle_OML : angle_OML = x)
  (h_angle_OJK : angle_OJK = 3 * x)
  (h_angle_KLM : angle_KLM = 2 * x) :
  angle_KLM = 90 := by
  sorry

end angle_KLM_is_90_degrees_l118_118926


namespace ratio_second_to_first_l118_118911

noncomputable def ratio_of_second_to_first (x y z : ℕ) (k : ℕ) : ℕ := sorry

theorem ratio_second_to_first
    (x y z : ℕ)
    (h1 : z = 2 * y)
    (h2 : y = k * x)
    (h3 : (x + y + z) / 3 = 78)
    (h4 : x = 18)
    (k_val : k = 4):
  ratio_of_second_to_first x y z k = 4 := sorry

end ratio_second_to_first_l118_118911


namespace lcm_of_9_12_15_l118_118215

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end lcm_of_9_12_15_l118_118215


namespace verandah_area_correct_l118_118940

-- Definitions and conditions from the problem
def room_length : ℝ := 15
def room_width : ℝ := 12
def verandah_width_longer_sides : ℝ := 2
def verandah_width_one_shorter_side : ℝ := 3
def verandah_width_other_shorter_side : ℝ := 1
def semicircle_radius : ℝ := 2

-- Derived values based on the conditions
def verandah_area_longer_sides : ℝ := 2 * verandah_width_longer_sides * room_length
def verandah_area_one_shorter_side : ℝ := (room_width + 2 * verandah_width_longer_sides) * verandah_width_one_shorter_side
def verandah_area_other_shorter_side : ℝ := (room_width + verandah_width_longer_sides + verandah_width_other_shorter_side) * verandah_width_other_shorter_side
def verandah_area_semi_circle : ℝ := (1 / 2) * Mathlib.pi * semicircle_radius^2

-- Total verandah area
def total_verandah_area : ℝ := verandah_area_longer_sides + verandah_area_one_shorter_side + verandah_area_other_shorter_side + verandah_area_semi_circle

-- The target proof statement
theorem verandah_area_correct : total_verandah_area ≈ 129.28 :=
by
  sorry

end verandah_area_correct_l118_118940


namespace circle_center_and_radius_l118_118408

theorem circle_center_and_radius :
  ∃ C : ℝ × ℝ, ∃ r : ℝ, (∀ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 3 = 0 ↔ 
    (x - C.1)^2 + (y - C.2)^2 = r^2) ∧ C = (1, -2) ∧ r = Real.sqrt 2 :=
by 
  sorry

end circle_center_and_radius_l118_118408


namespace cost_of_french_bread_is_correct_l118_118929

noncomputable def cost_of_sandwiches := 2 * 7.75
noncomputable def cost_of_salami := 4.00
noncomputable def cost_of_brie := 3 * cost_of_salami
noncomputable def cost_of_olives := 10.00 * (1/4)
noncomputable def cost_of_feta := 8.00 * (1/2)
noncomputable def total_cost_of_items := cost_of_sandwiches + cost_of_salami + cost_of_brie + cost_of_olives + cost_of_feta
noncomputable def total_spent := 40.00
noncomputable def cost_of_french_bread := total_spent - total_cost_of_items

theorem cost_of_french_bread_is_correct :
  cost_of_french_bread = 2.00 :=
by
  sorry

end cost_of_french_bread_is_correct_l118_118929


namespace simplify_polynomial_l118_118586

theorem simplify_polynomial :
  (λ x : ℝ, (2 * x^2 + 3 * x + 7) * (x - 2) - (x - 2) * (x^2 - 4 * x + 9) + (4 * x^2 - 3 * x + 1) * (x - 2) * (x - 5))
  = (λ x : ℝ, 5 * x^3 - 26 * x^2 + 35 * x - 6) :=
by 
sory

end simplify_polynomial_l118_118586


namespace find_tasty_21_moves_find_tasty_20_moves_l118_118185

-- Definition of the problem conditions

def candies :=
  set ℕ -- Candy set represented as a set of natural numbers

variable (tasty : candies) -- Define the subset of candies that are tasty

-- Definition of the problem's clauses
def num_candies : ℕ := 28

def find_all_tasty_candies (m : ℕ) : Prop :=
  ∃ strategy : finset (set ℕ), 
    (∀ s ∈ strategy, s.card ≤ num_candies ∧ s ⊆ set.range num_candies) ∧ 
    (strategy.card ≤ m) ∧ 
    (∀ c ∈ set.range num_candies, c ∈ tasty → c ∈ ⋃ (s ∈ strategy), s) ∧
    (∀ c ∈ set.range num_candies, ¬(c ∈ tasty) → c ∉ ⋃ (s ∈ strategy), s)

-- The statement of the theorem splits the question into two parts:
theorem find_tasty_21_moves : find_all_tasty_candies tasty 21 := sorry

theorem find_tasty_20_moves : find_all_tasty_candies tasty 20 := sorry

end find_tasty_21_moves_find_tasty_20_moves_l118_118185


namespace calculate_f_of_f_of_f_l118_118752

def f (x : ℤ) : ℤ := 5 * x - 4

theorem calculate_f_of_f_of_f (h : f (f (f 3)) = 251) : f (f (f 3)) = 251 := 
by sorry

end calculate_f_of_f_of_f_l118_118752


namespace complex_power_conjugate_l118_118433

theorem complex_power_conjugate (z : ℂ) (hz : z = (1 + complex.i) / (1 - complex.i)) : 
  (conj z) ^ 2017 = - complex.i :=
by
  have h1 : z = complex.i := by sorry  -- proof to simplify z will be placed here
  have h2 : conj z = - complex.i := by sorry  -- proof to find conjugate of z will be placed here
  show (conj z) ^ 2017 = - complex.i from by sorry

end complex_power_conjugate_l118_118433


namespace work_done_by_force_l118_118269

theorem work_done_by_force :
  ∫ x in 0..1, (1 + Real.exp x) = Real.exp 1 :=
by
  sorry

end work_done_by_force_l118_118269


namespace increasing_ratio_of_f_l118_118741

variable {f : ℝ → ℝ}

theorem increasing_ratio_of_f (h_domain : ∀ x > 0, f x ≠ 0) 
  (h_deriv : ∀ x > 0, f' x > f x / x) :
  ∀ {x y : ℝ}, 0 < x → x < y → (f x / x) < (f y / y) :=
by
  sorry

end increasing_ratio_of_f_l118_118741


namespace leading_coefficient_four_l118_118175

noncomputable def poly_lead_coefficient (g : ℕ → ℕ → ℕ) : Prop :=
  ∀ x : ℕ, g(x + 1) - g(x) = 8 * x + 6

theorem leading_coefficient_four {g : ℕ → ℕ} 
  (h : poly_lead_coefficient g) : 
  ∃ a b c : ℕ, ∀ x : ℕ, g x = 4 * x^2 + 2 * x + c :=
begin
  sorry
end

end leading_coefficient_four_l118_118175


namespace Q_2_plus_Q_neg2_l118_118111

variable (m : ℝ)
variable (Q : ℝ → ℝ)

-- Given Conditions
axiom Q_0 : Q 0 = m
axiom Q_1 : Q 1 = 2m
axiom Q_neg1 : Q (-1) = 4m
axiom Q_2 : Q 2 = 5m
axiom quartic : ∃ (a b c d : ℝ), Q = λ x, a*x^4 + b*x^3 + c*x^2 + d*x + m

-- We aim to prove that Q(2) + Q(-2) = 66m
theorem Q_2_plus_Q_neg2 : Q 2 + Q (-2) = 66m :=
by
  sorry

end Q_2_plus_Q_neg2_l118_118111


namespace line_equation_l118_118514

-- Definitions based on conditions
def point : ℝ × ℝ := (-1, 2)
def slope : ℝ := 1

-- Main proof statement
theorem line_equation (x y : ℝ) (H : (x + 1) = slope * (y - 2)) : x - y + 3 = 0 :=
sorry

end line_equation_l118_118514


namespace cats_sold_during_sale_l118_118273

-- Definitions based on conditions in a)
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def cats_left : ℕ := 8
def total_cats := siamese_cats + house_cats

-- Proof statement
theorem cats_sold_during_sale : total_cats - cats_left = 10 := by
  sorry

end cats_sold_during_sale_l118_118273


namespace average_people_moving_l118_118852

theorem average_people_moving (days : ℕ) (total_people : ℕ) 
    (h_days : days = 5) (h_total_people : total_people = 3500) : 
    (total_people / days) = 700 :=
by
  sorry

end average_people_moving_l118_118852


namespace string_cuts_l118_118974

theorem string_cuts (L S : ℕ) (h_diff : L - S = 48) (h_sum : L + S = 64) : 
  (L / S) = 7 :=
by
  sorry

end string_cuts_l118_118974


namespace snake_body_length_l118_118639

theorem snake_body_length (L : ℝ) (H : ℝ) (h1 : H = L / 10) (h2 : L = 10) : L - H = 9 :=
by
  sorry

end snake_body_length_l118_118639


namespace members_removed_l118_118284

-- Definitions and problem conditions
def messages_per_day : ℕ := 50
def initial_members : ℕ := 150
def remaining_messages_per_week : ℕ := 45500
def days_per_week : ℕ := 7

-- Our theorem statement to prove the number of members removed
theorem members_removed : 
  ∃ (x : ℕ), 
  let messages_per_week := messages_per_day * days_per_week in
  let initial_messages := initial_members * messages_per_week in
  let remaining_members := initial_members - x in
  remaining_members * messages_per_week = remaining_messages_per_week ∧ 
  x = 20 :=
begin
  sorry
end

end members_removed_l118_118284


namespace cylinder_height_calculation_l118_118530

noncomputable def radius : ℝ := 12
noncomputable def lateral_surface_area : ℝ := 1583.3626974092558
noncomputable def expected_height : ℝ := 21.01

theorem cylinder_height_calculation :
  let h := lateral_surface_area / (2 * Real.pi * radius) in
  h ≈ expected_height := sorry

end cylinder_height_calculation_l118_118530


namespace area_triangle_MOI_l118_118052

-- Definitions of the vertices of triangle ABC (assuming it lies on a 2D plane)
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (0, 15)

-- Definition of the circumcenter O
def O : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Definition of the incenter I
def I : ℝ × ℝ := 
  ((15 * B.1 + 8 * (B.1 + C.1) + 17 * C.1) / (15 + 8 + 17), 
   (15 * B.2 + 8 * (B.2 + C.2) + 17 * C.2) / (15 + 8 + 17))

-- The circle centered at M is assumed to be tangent to lines AC, BC, and the circumcircle.
-- Defining the center M based on geometric assumptions and constraints.
-- Note: Simplified assumption used in the problem; the precise determination of M may differ.
def M : ℝ × ℝ := (4, 4)

-- Proving that the area of triangle MOI is equal to 3
theorem area_triangle_MOI : 
  let area := 1 / 2 * abs (M.1 * (O.2 - I.2) + O.1 * (I.2 - M.2) + I.1 * (M.2 - O.2)) in
  area = 3 :=
by
  sorry

end area_triangle_MOI_l118_118052


namespace tuesday_miles_l118_118868

/--   Lennon is reimbursed $0.36 per mile.
  On Monday he drove 18 miles.
  On Wednesday and Thursday he drove 20 miles each day.
  On Friday he drove 16 miles.
  He will be reimbursed a total of $36.
  Prove that the mileage driven by Lennon on Tuesday is 26 miles.
--/

theorem tuesday_miles (reimbursement_rate : ℝ)
                      (monday_miles : ℕ)
                      (wednesday_miles : ℕ)
                      (thursday_miles : ℕ)
                      (friday_miles : ℕ)
                      (total_reimbursement : ℝ)
                      (tuesday_miles := ((total_reimbursement - (monday_miles + wednesday_miles + thursday_miles + friday_miles) * reimbursement_rate) / reimbursement_rate).to_nat) :
                      reimbursement_rate = 0.36 ∧ 
                      monday_miles = 18 ∧ 
                      wednesday_miles = 20 ∧ 
                      thursday_miles = 20 ∧ 
                      friday_miles = 16 ∧ 
                      total_reimbursement = 36 →
                      tuesday_miles = 26 :=
by
  sorry

end tuesday_miles_l118_118868


namespace percentage_first_division_l118_118840

-- Given definitions
def total_students : ℕ := 300
def second_division_percentage : ℕ := 54
def just_passed_students : ℕ := 48
def failed_students : ℕ := 0

-- Proof statement
theorem percentage_first_division :
  (λ (total_students second_division_percentage just_passed_students failed_students, 
    let just_passed_percentage := (just_passed_students * 100) / total_students in
    100 = just_passed_percentage + second_division_percentage + 30) :=
sorry

end percentage_first_division_l118_118840


namespace range_of_m_l118_118765

def vector_a : ℝ × ℝ := (-3, m)
def vector_b : ℝ × ℝ := (4, 3)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2
def slopes_not_equal (v1 v2 : ℝ × ℝ) : Prop := v1.1 / v1.2 ≠ v2.1 / v2.2

theorem range_of_m (m : ℝ) (h_dot : dot_product vector_a vector_b < 0)
    (h_slope : slopes_not_equal vector_a vector_b) : 
    m < 4 ∧ m ≠ -9 / 4 :=
sorry

end range_of_m_l118_118765


namespace functional_equation_l118_118687

variables (R : Type) [AddCommGroup R] [MulAction ℝ R] [DistribMulAction ℝ R] [TopologicalSpace R]
noncomputable def f : R → R := sorry
noncomputable def g : R → R := sorry
noncomputable def h : R → R := sorry

theorem functional_equation (x y : R) (f g h : R → R)
  (h_eq : ∀ (x y : R), f(x + y^3) + g(x^3 + y) = h(x*y)) :
  (∃ c : R, ∀ x, f(x) = c) ∧
  (∃ d : R, ∀ x, g(x) = d) ∧
  (∃ e : R, ∀ x, h(x) = e) ∧
  ∃ c d e : R, c + d = e ∧ d = e - c :=
begin
  sorry
end

end functional_equation_l118_118687


namespace kolya_correct_valya_incorrect_l118_118307

variable (x : ℝ)

def p : ℝ := 1 / x
def r : ℝ := 1 / (x + 1)
def q : ℝ := (x - 1) / x
def s : ℝ := x / (x + 1)

theorem kolya_correct : r / (1 - s * q) = 1 / 2 := by
  sorry

theorem valya_incorrect : (q * r + p * r) / (1 - s * q) = 1 / 2 := by
  sorry

end kolya_correct_valya_incorrect_l118_118307


namespace problem_l118_118123

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem problem (a : ℝ) (x : ℝ) (hx : x ∈ Set.Ici (-5)) (ha : a = 1) : 
  f x a + x + 5 ≥ -6 / Real.exp 5 := 
sorry

end problem_l118_118123


namespace problem1_problem2_problem3_problem4_l118_118318

open Real

/-- The area of the figure bounded by the coordinate axes, the line x = 3, and the parabola y = x^2 + 1 is 12. -/
theorem problem1 :
  ∫ x in 0..3, x^2 + 1 = 12 := sorry

/-- The area of the figure bounded by the y-axis, the lines y = -2, y = 3, and the parabola x = (1/2) y^2 is 35/6. -/
theorem problem2 :
  (1/2:ℝ) * ∫ y in -2..3, y^2 = 35/6 := sorry

/-- The area of the figure bounded by the parabolas y = x^2 and x = y^2 is 1/3. -/
theorem problem3 :
  ∫ x in 0..1, sqrt x - x^2 = (1/3:ℝ) := sorry

/-- The area of the figure bounded by the parabolas y = x^2 + 1, y = (1/2) x^2, and the line y = 5 is 4/3 (5 sqrt 10 - 8). -/
theorem problem4 :
  (4/3) * (5 * sqrt 10 - 8) = ∫ y in 0..5, sqrt (2 * y) - ∫ y in 1..5, sqrt (y - 1) := sorry

end problem1_problem2_problem3_problem4_l118_118318


namespace sum_fraction_pattern_l118_118750

theorem sum_fraction_pattern {n : ℕ} (h : n > 0) : 
  ∑ i in Finset.range n, (i + 3) / (i * (i + 1)) * (1 / (2^i : ℝ)) = 1 - 1 / ((n+1) * 2^n : ℝ) := 
sorry

end sum_fraction_pattern_l118_118750


namespace arithmetic_sequence_properties_summation_inequality_l118_118066

noncomputable def a_n (n : ℕ) : ℕ := 3 + 2 * (n - 1)
noncomputable def b_n (n : ℕ) : ℕ := 8 ^ (n - 1)
noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2)

theorem arithmetic_sequence_properties (d q : ℕ) (h₁ : a_1 = 3) (h₂ : b_1 = 1)
    (h₃ : b_2 * S_2 = 64) (h₄ : b_3 * S_3 = 960) :
    a_n = (λ n, 2 * n + 1) ∧ b_n = (λ n, 8 ^ (n - 1)) :=
by
  sorry

theorem summation_inequality (n : ℕ) :
    ∑ k in Finset.range n, (1 / S_n k) < 3 / 4 :=
by
  sorry

end arithmetic_sequence_properties_summation_inequality_l118_118066


namespace seeds_never_grew_l118_118491

def total_seeds := 23
def uneaten_plants := 9
def final_plants := 4

theorem seeds_never_grew (total_seeds uneaten_plants : ℕ) : ℕ :=
  let eaten_by_squirrels_and_rabbits := 1/3
  let strangled_by_weeds := 1/3
  let remaining_plants := total_seeds - final_plants
  final_plants = remaining_plants := by sorry

# Check correct answer
example : seeds_never_grew 23 9 = 4 := by sorry

end seeds_never_grew_l118_118491


namespace carter_total_additional_cakes_l118_118546

def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet_cakes : ℕ := 8

def increased_cheesecakes : ℕ := (usual_cheesecakes * 1.5).natAbs
def increased_muffins : ℕ := (usual_muffins * 1.2).natAbs
def increased_red_velvet_cakes : ℕ := (usual_red_velvet_cakes * 1.8).natAbs
def increased_chocolate_moist_cakes : ℕ := (increased_red_velvet_cakes * 0.5).natAbs
def increased_fruitcakes : ℕ := (increased_muffins * (2/3)).natAbs
def increased_carrot_cakes : ℕ := (0 * 1.25).natAbs

def additional_cheesecakes : ℕ := increased_cheesecakes - usual_cheesecakes
def additional_muffins : ℕ := increased_muffins - usual_muffins
def additional_red_velvet_cakes : ℕ := increased_red_velvet_cakes - usual_red_velvet_cakes
def additional_chocolate_moist_cakes : ℕ := increased_chocolate_moist_cakes - 0
def additional_fruitcakes : ℕ := increased_fruitcakes - 0
def additional_carrot_cakes : ℕ := increased_carrot_cakes - 0

def total_additional_cakes : ℕ :=
  additional_cheesecakes + 
  additional_muffins + 
  additional_red_velvet_cakes +
  additional_chocolate_moist_cakes +
  additional_fruitcakes +
  additional_carrot_cakes

theorem carter_total_additional_cakes : total_additional_cakes = 21 :=
by
  -- proof goes here
  sorry

end carter_total_additional_cakes_l118_118546


namespace num_bicycles_eq_20_l118_118973

-- Definitions based on conditions
def num_cars : ℕ := 10
def num_motorcycles : ℕ := 5
def total_wheels : ℕ := 90
def wheels_per_bicycle : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_motorcycle : ℕ := 2

-- Statement to prove
theorem num_bicycles_eq_20 (B : ℕ) 
  (h_wheels_from_bicycles : wheels_per_bicycle * B = 2 * B)
  (h_wheels_from_cars : num_cars * wheels_per_car = 40)
  (h_wheels_from_motorcycles : num_motorcycles * wheels_per_motorcycle = 10)
  (h_total_wheels : wheels_per_bicycle * B + 40 + 10 = total_wheels) :
  B = 20 :=
sorry

end num_bicycles_eq_20_l118_118973


namespace email_sending_ways_l118_118271

theorem email_sending_ways (n k : ℕ) (hn : n = 3) (hk : k = 5) : n^k = 243 := 
by
  sorry

end email_sending_ways_l118_118271


namespace perimeter_is_22_l118_118447

-- Definitions based on the conditions
def side_lengths : List ℕ := [2, 3, 2, 6, 2, 4, 3]

-- Statement of the problem
theorem perimeter_is_22 : side_lengths.sum = 22 := 
  sorry

end perimeter_is_22_l118_118447


namespace angle_y_equals_90_l118_118695

/-- In a geometric configuration, if ∠CBD = 120° and ∠ABE = 30°, 
    then the measure of angle y is 90°. -/
theorem angle_y_equals_90 (angle_CBD angle_ABE : ℝ) 
  (h1 : angle_CBD = 120) 
  (h2 : angle_ABE = 30) : 
  ∃ y : ℝ, y = 90 := 
by
  sorry

end angle_y_equals_90_l118_118695


namespace boats_left_l118_118126

theorem boats_left (initial_boats : ℕ) (percent_eaten : ℝ) (shot_boats : ℕ) (eaten := (percent_eaten * initial_boats.to_real).to_nat) 
(remaining_boats := initial_boats - eaten - shot_boats) :
  initial_boats = 30 → percent_eaten = 0.2 → shot_boats = 2 → remaining_boats = 22 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

end boats_left_l118_118126


namespace remainder_calculation_l118_118217

theorem remainder_calculation :
  ((2367 * 1023) % 500) = 41 := by
  sorry

end remainder_calculation_l118_118217


namespace num_colors_l118_118129

def total_balls := 350
def balls_per_color := 35

theorem num_colors :
  total_balls / balls_per_color = 10 := 
by
  sorry

end num_colors_l118_118129


namespace cosine_to_sine_shift_l118_118976

theorem cosine_to_sine_shift :
    ∀ x : ℝ, (∀ x, cos (2*x + π/6) = sin (2*x + 2*π/3)) →
            sin (2*(x - π/3) + 2*π/3) = sin (2*x) := 
by
  intros x h
  sorry

end cosine_to_sine_shift_l118_118976


namespace part_I_part_II_l118_118488

variable {x s t : ℝ}

noncomputable def f (x a : ℝ) := abs (x - a)

theorem part_I (h₁ : ∀ x, f x 2 ≥ 6 - abs (2 * x - 5)) : 
  (set.Iio (1 / 3) ∪ set.Ici (13 / 3)) = {x | f x 2 ≥ 6 - abs (2 * x - 5)} :=
sorry

theorem part_II (h₂ : ∀ x, f x 3 ≤ 4 → x ∈ set.Icc (-1) 7) 
  (s_pos : 0 < s) (t_pos : 0 < t) (hyp : 2 * s + t = 3) : (1 / s + 8 / t) ≥ 6 :=
sorry

end part_I_part_II_l118_118488


namespace area_section_regular_hexagon_l118_118120

-- Definitions of points on the cube
structure Cube := (A B C D A1 B1 C1 D1 : ℝ³)
def side_length := 1 : ℝ
def midpoint (P Q : ℝ³) : ℝ³ := (P + Q) / 2

-- Conditions
def M (c : Cube) : ℝ³ := midpoint c.A1 c.A
def F (c : Cube) : ℝ³ := midpoint c.A c.D
def N (c : Cube) : ℝ³ := midpoint c.C c.C1
def L (c : Cube) : ℝ³ := midpoint c.D c.C
def O (c : Cube) : ℝ³ := sorry -- Intersection of FM with D1A1
def M1 (c : Cube) : ℝ³ := sorry -- Point on C1B1 based on line from N parallel to FM
def P (c : Cube) : ℝ³ := midpoint c.A1 c.B1

-- Theorem to be proved
theorem area_section_regular_hexagon (c : Cube) : 
  let s := (1 / Real.sqrt 2 : ℝ) in
  let area := (3 * Real.sqrt 3) / 4 in
  sorry

end area_section_regular_hexagon_l118_118120


namespace polynomial_roots_distinct_and_expression_is_integer_l118_118423

-- Defining the conditions and the main theorem
theorem polynomial_roots_distinct_and_expression_is_integer (a b c : ℂ) :
  (a^3 - a^2 - a - 1 = 0) → (b^3 - b^2 - b - 1 = 0) → (c^3 - c^2 - c - 1 = 0) → 
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ k : ℤ, ((a^(1982) - b^(1982)) / (a - b) + (b^(1982) - c^(1982)) / (b - c) + (c^(1982) - a^(1982)) / (c - a) = k) :=
by
  intros h1 h2 h3
  -- Proof omitted
  sorry

end polynomial_roots_distinct_and_expression_is_integer_l118_118423


namespace BF_bisects_ED_l118_118878

section GeometryProof

variables {A B C D E F O : Type}

/-- Assuming point C is on circle Omega with diameter AB and center O. -/
axiom C_on_Omega : ∃ (O : Type) (A B C : Type), Circle O (Diameter A B) ∧ PointOnCircle C O

/-- Point D is the intersection of the perpendicular bisector of AC with circle Omega. -/
axiom D_intersection : ∃ (D : Type), PerpendicularBisectorIntersection AC D ∧ PointOnCircle D O

/-- Point E is the projection of D onto BC. -/
axiom E_projection : ∃ (E : Type), Projection D BCE

/-- Point F is the intersection of line AE with circle Omega. -/
axiom F_intersection : ∃ (F : Type), LineIntersection AE FOmega

/-- To prove that line BF bisects segment ED. -/
theorem BF_bisects_ED :
  ∀ (B F E D : Type) (circleOmega : Circle ℝ), IsMidpoint (Intersection (LineBF B F) (Segment E D)) (Segment E D) :=
by
  sorry

end GeometryProof

end BF_bisects_ED_l118_118878


namespace kevin_needs_one_more_l118_118062

theorem kevin_needs_one_more (total_questions : ℕ := 100)
    (physics_questions : ℕ := 20) (chemistry_questions : ℕ := 40) (biology_questions : ℕ := 40)
    (physics_correct_pct : ℝ := 0.80) (chemistry_correct_pct : ℝ := 0.50) (biology_correct_pct : ℝ := 0.70)
    (passing_pct : ℝ := 0.65) :
    let physics_correct := physics_correct_pct * physics_questions,
        chemistry_correct := chemistry_correct_pct * chemistry_questions,
        biology_correct := biology_correct_pct * biology_questions,
        total_correct := physics_correct + chemistry_correct + biology_correct,
        passing_mark := passing_pct * total_questions in total_correct < passing_mark ∧ passing_mark - total_correct = 1 :=
by
  sorry

end kevin_needs_one_more_l118_118062


namespace difference_in_squares_l118_118610

noncomputable def radius_of_circle (x y h R : ℝ) : Prop :=
  5 * x^2 - 4 * x * h + h^2 = R^2 ∧ 5 * y^2 + 4 * y * h + h^2 = R^2

theorem difference_in_squares (x y h R : ℝ) (h_radius : radius_of_circle x y h R) :
  2 * x - 2 * y = (8/5 : ℝ) * h :=
by
  sorry

end difference_in_squares_l118_118610


namespace conditional_probability_l118_118830

-- Definitions based on the conditions
def P (event: Type) (prob: ℚ) : Prop := sorry

def EastWindApril := Type
def RainApril := Type

axiom prob_east_wind_april : P EastWindApril (8 / 30)
axiom prob_east_wind_and_rain : P (EastWindApril × RainApril) (7 / 30)

-- The theorem statement for the mathematically equivalent proof problem
theorem conditional_probability (A B : Type) (PA PAB : ℚ)
  (hPA : P A PA) (hPAB : P (A × B) PAB) : P (B | A) (7 / 8) :=
by
  -- The proof is omitted and replaced with sorry
  sorry

end conditional_probability_l118_118830


namespace three_digit_multiples_of_30_not_75_l118_118778

theorem three_digit_multiples_of_30_not_75 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ 30 ∣ n ∧ ¬ (75 ∣ n)}.card = 21 :=
by sorry

end three_digit_multiples_of_30_not_75_l118_118778


namespace max_cookies_one_member_l118_118254

theorem max_cookies_one_member (cookies members : ℕ) (h1 : members = 40) (h2 : cookies = 200) (h3 : ∀ m : ℕ, m < members → 1 ≤ dist m → 1 ≤ cookies) : 
  ∃ m : ℕ, m ≤ cookies ∧ m = 161 := 
begin
  sorry
end

end max_cookies_one_member_l118_118254


namespace measure_minor_arc_LB_l118_118849

-- Define the conditions about the circle and angles
def circle_C (O L B : Type) [metric_space O] [metric_space L] [metric_space B] : Prop :=
  ∃ (C : set O), C = metric.sphere L (dist L O)

def angle_LOB_eq_60_deg (O L B : Type) [metric_space O] [metric_space L] [metric_space B] 
  (angle : O → L → B → ℝ) : Prop :=
  angle O L B = 60

-- Define the theorem statement
theorem measure_minor_arc_LB 
  (O L B : Type) [metric_space O] [metric_space L] [metric_space B]
  (C : set O) (angle : O → L → B → ℝ)
  (hC : circle_C O L B) (hAngle : angle_LOB_eq_60_deg O L B angle) : 
  measure_arc_LB C L B = 60 := 
sorry

end measure_minor_arc_LB_l118_118849


namespace average_age_of_children_l118_118441

theorem average_age_of_children 
  (num_participants : ℕ) (total_avg_age : ℕ)
  (num_women : ℕ) (num_men : ℕ) (num_children : ℕ)
  (women_avg_age : ℕ) (men_avg_age : ℕ) :
  num_participants = 50 →
  total_avg_age = 20 →
  num_women = 30 →
  num_men = 10 →
  num_children = 10 →
  women_avg_age = 22 →
  men_avg_age = 25 →
  (90 / 10 = 9) :=
begin
  intros,
  sorry
end

end average_age_of_children_l118_118441


namespace fixed_points_corresponding_l118_118499

-- Definitions based on conditions
noncomputable def similar_figures (F1 F2 F3 : Type) := sorry
noncomputable def similarity_transformation_center (F2 F3 : Type) := sorry
noncomputable def fixed_points (F : Type) := sorry
noncomputable def auxiliary_point (F : Type) := sorry
noncomputable def similarity_coefficient (F2 F3 : Type) := sorry

-- Assumptions from the problem
variables (F1 F2 F3 : Type) [similar_figures F1 F2 F3]
variables (O1 : Type) [similarity_transformation_center F2 F3]
variables (J2 J3 : Type) [fixed_points F2] [fixed_points F3]
variables (W : Type) [auxiliary_point F2]
variables (k1 : ℝ) [similarity_coefficient F2 F3]

-- Statement of the theorem
theorem fixed_points_corresponding :
  (∀ F1 F2 F3, similar_figures F1 F2 F3) →
  (∀ F2 F3, similarity_transformation_center F2 F3) →
  (∀ J2 J3, fixed_points F2 → fixed_points F3) →
  (∀ W, auxiliary_point F2) →
  (∀ k1, similarity_coefficient F2 F3) →
  (∀ J2 J3 W, angle J2 O1 O1 J3 = angle J2 W W J3) →
  (∀ O1 J2 J3, dist O1 J2 / dist O1 J3 = k1) →
  (∀ F1 F2 F3, fixed_points F1 = fixed_points F2 = fixed_points F3) :=
sorry

end fixed_points_corresponding_l118_118499


namespace smallest_positive_period_max_value_on_interval_l118_118116

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

theorem smallest_positive_period (x : ℝ) :
  (∃ T > 0, ∀ x, ([f (x + π / 2)] ^ 2) = ([f (x + π / 2 + T)] ^ 2)) ∧
  (∀ T' > 0, T' < π → ∃ x, ([f (x + π / 2)] ^ 2) ≠ ([f (x + π / 2 + T')] ^ 2)) :=
sorry

theorem max_value_on_interval (x : ℝ) :
  (0 ≤ x ∧ x ≤ π / 2 → f x * f (x - π / 4) ≤ 1 + Real.sqrt 2 / 2) ∧
  (∃ x', 0 ≤ x' ∧ x' ≤ π / 2 ∧ f x' * f (x' - π / 4) = 1 + Real.sqrt 2 / 2) :=
sorry

end smallest_positive_period_max_value_on_interval_l118_118116


namespace problem_10_order_l118_118600

theorem problem_10_order (a b c : ℝ) (h1 : a = Real.sin (17 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) + Real.cos (17 * Real.pi / 180) * Real.sin (45 * Real.pi / 180))
    (h2 : b = 2 * (Real.cos (13 * Real.pi / 180))^2 - 1)
    (h3 : c = Real.sqrt 3 / 2) :
    c < a ∧ a < b :=
sorry

end problem_10_order_l118_118600


namespace translate_function_3_units_right_l118_118518

theorem translate_function_3_units_right :
  ∀ x : ℝ, (λ x : ℝ, 2 * x) (x - 3) = 2 * x - 6 := 
by
  intros x
  sorry

end translate_function_3_units_right_l118_118518


namespace three_digit_multiples_l118_118786

theorem three_digit_multiples : 
  let three_digit_range := {x : ℕ | 100 ≤ x ∧ x < 1000}
  let multiples_of_30 := {x ∈ three_digit_range | x % 30 = 0}
  let multiples_of_75 := {x ∈ three_digit_range | x % 75 = 0}
  let count_multiples_of_30 := Set.card multiples_of_30
  let count_multiples_of_75 := Set.card multiples_of_75
  let count_common_multiples := Set.card (multiples_of_30 ∩ multiples_of_75)
  count_multiples_of_30 - count_common_multiples = 24 :=
by
  sorry

end three_digit_multiples_l118_118786


namespace money_left_after_spending_l118_118460

theorem money_left_after_spending (initial_amount spent_amount remaining_amount : ℝ) 
  (h_initial : initial_amount = 100.0)
  (h_spent : spent_amount = 15.0) :
  remaining_amount = initial_amount - spent_amount :=
by {
  rw [h_initial, h_spent],
  norm_num,
  exact rfl,
}

end money_left_after_spending_l118_118460


namespace sin_equation_proof_l118_118397

theorem sin_equation_proof (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) : 
  Real.sin (5 * π / 6 - x) + Real.sin (π / 3 - x) ^ 2 = 19 / 16 :=
by
  sorry

end sin_equation_proof_l118_118397


namespace hayley_friends_l118_118769

theorem hayley_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (h1 : total_stickers = 72) (h2 : stickers_per_friend = 8) : (total_stickers / stickers_per_friend) = 9 :=
by
  sorry

end hayley_friends_l118_118769


namespace no_p_n_eq_5_l118_118471

noncomputable def sequence_of_primes : ℕ → ℕ
| 0 := 2
| (n+1) := Nat.factors (sequence_of_primes 0 * sequence_of_primes 1 * ... * sequence_of_primes n + 1).last

theorem no_p_n_eq_5 (n : ℕ) : sequence_of_primes n ≠ 5 :=
sorry

end no_p_n_eq_5_l118_118471


namespace negation_of_proposition_l118_118007
-- Add the necessary import

-- Define the conditions and the statement using Lean 4
theorem negation_of_proposition (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : a ≠ c) :
  (∀ (p : ℝ → ℝ), 
     (p = (λ x, a * x^2 + 2 * b * x + c) ∨ 
      p = (λ x, b * x^2 + 2 * c * x + a) ∨ 
      p = (λ x, c * x^2 + 2 * a * x + c)) → ¬((∃ r1 r2 : ℝ, r1 ≠ r2 ∧ p r1 = 0 ∧ p r2 = 0))) ↔ 
  (∀ (p : ℝ → ℝ), 
     (p = (λ x, a * x^2 + 2 * b * x + c) ∨ 
      p = (λ x, b * x^2 + 2 * c * x + a) ∨ 
      p = (λ x, c * x^2 + 2 * a * x + c)) → (¬(∃ r1 r2 : ℝ, r1 ≠ r2 ∧ p r1 = 0 ∧ p r2 = 0))) := by sorry

end negation_of_proposition_l118_118007


namespace smallest_y_value_l118_118570

theorem smallest_y_value :
  ∃ y : ℝ, (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) ∧ ∀ z : ℝ, (3 * z ^ 2 + 33 * z - 90 = z * (z + 16)) → y ≤ z :=
sorry

end smallest_y_value_l118_118570


namespace value_range_of_f_l118_118184

noncomputable def f (x : ℝ) : ℝ := log (4 ^ x - 2 ^ (x + 1) + 3) / log 2

theorem value_range_of_f :
  ∀ x : ℝ, f(x) ≥ 1 := 
by
  sorry

end value_range_of_f_l118_118184


namespace range_of_m_range_of_f_l118_118754

noncomputable def y (m : ℝ) (x : ℝ) := real.sqrt (m * x^2 - 6 * m * x + m + 8)

theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, m * x^2 - 6 * m * x + m + 8 ≥ 0) ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

noncomputable def f (m : ℝ) := real.sqrt (8 - 8 * m)

theorem range_of_f :
  ∀ m : ℝ, (0 ≤ m ∧ m ≤ 1) → (0 ≤ f(m) ∧ f(m) ≤ 2 * real.sqrt 2) :=
sorry

end range_of_m_range_of_f_l118_118754


namespace largest_possible_perimeter_l118_118644

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 15) : 
  (7 + 8 + x) ≤ 29 := 
sorry

end largest_possible_perimeter_l118_118644


namespace min_positive_value_sum_50_l118_118346

theorem min_positive_value_sum_50 :
  (∃ a : Fin 50 → ℤ,
    (∀ i : Fin 50, a i = 1 ∨ a i = -1) ∧ 
    (∑ i : Fin 50, a i) % 2 = 0 ∧
    (let S := ∑ i in finset.range 50, ∑ j in finset.range i, a i * a j in
     S = 7)) :=
sorry

end min_positive_value_sum_50_l118_118346


namespace find_sin_beta_l118_118376

variable (α β : ℝ)

-- Conditions
axiom sin_alpha_eq : Math.sin α = 4 / 5
axiom cos_alpha_plus_beta_eq : Math.cos (α + β) = -3 / 5
axiom alpha_in_first_quadrant : 0 < α ∧ α < Math.pi / 2
axiom beta_in_first_quadrant : 0 < β ∧ β < Math.pi / 2

theorem find_sin_beta : Math.sin β = 24 / 25 := by
  sorry

end find_sin_beta_l118_118376


namespace result_l118_118295

noncomputable def Kolya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  P_A = (1: ℝ) / 2

def Valya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  let P_B := (r * p) / (1 - s * q)
  let P_AorB := P_A + P_B
  P_AorB > (1: ℝ) / 2

theorem result (p r : ℝ) (h1 : Kolya_right p r) (h2 : ¬Valya_right p r) :
  Kolya_right p r ∧ ¬Valya_right p r :=
  ⟨h1, h2⟩

end result_l118_118295


namespace no_valid_arrangement_exists_l118_118454

def arrangement_property (arr : List ℕ) : Prop :=
  ∀ i, arr[i] = Nat.gcd arr[(i-2).mod 1400] arr[(i-1).mod 1400] + Nat.gcd arr[(i+1).mod 1400] arr[(i+2).mod 1400]

theorem no_valid_arrangement_exists (arr : List ℕ) (h_len : arr.length = 1400) (h_2021 : 2021 ∈ arr) :
  ¬ arrangement_property arr :=
sorry

end no_valid_arrangement_exists_l118_118454


namespace how_many_converses_true_l118_118401

theorem how_many_converses_true :
  (iff (∀ (a b : ℝ), a > 0 ∧ b > 0 → a + b > 0) (∀ (a b : ℝ), a + b > 0 → a > 0 ∧ b > 0)) ↔ false ∧
  (iff (∀ (sq : Type) [is_square sq], get_diagonals sq = get_perpendicular_bisectors sq)
        (∀ (quad : Type) [is_quad quad], get_perpendicular_bisectors quad = get_diagonals quad)) ↔ false ∧
  (iff (∀ (rt : Type) [is_right_triangle rt], get_median_to_hypotenuse rt = get_half_hypotenuse rt)
        (∀ (tri : Type) [is_triangle tri], get_half_hypotenuse tri = get_median_to_hypotenuse tri)) ↔ true ∧
  (iff (∀ (rh : Type) [is_rhombus rh], all_sides_equal rh)
        (∀ (quad : Type) [is_quad quad], all_sides_equal quad)) ↔ true →
  2 := 
by
  sorry

end how_many_converses_true_l118_118401


namespace percentage_loss_is_25_l118_118263

def cost_price := 1400
def selling_price := 1050
def loss := cost_price - selling_price
def percentage_loss := (loss / cost_price) * 100

theorem percentage_loss_is_25 : percentage_loss = 25 := by
  sorry

end percentage_loss_is_25_l118_118263


namespace line_segment_value_of_x_l118_118622

theorem line_segment_value_of_x (x : ℝ) (h1 : (1 - 4)^2 + (3 - x)^2 = 25) (h2 : x > 0) : x = 7 :=
sorry

end line_segment_value_of_x_l118_118622


namespace ratio_of_votes_l118_118338

theorem ratio_of_votes (votes_A votes_B total_votes : ℕ) (hA : votes_A = 14) (hTotal : votes_A + votes_B = 21) : votes_A / Nat.gcd votes_A votes_B = 2 ∧ votes_B / Nat.gcd votes_A votes_B = 1 := 
by
  sorry

end ratio_of_votes_l118_118338


namespace infinitely_many_colorings_l118_118138

def colorings_exist (clr : ℕ → Prop) : Prop :=
  ∀ a b : ℕ, (clr a = clr b) ∧ (0 < a - 10 * b) → clr (a - 10 * b) = clr a

theorem infinitely_many_colorings : ∃ (clr : ℕ → Prop), colorings_exist clr :=
sorry

end infinitely_many_colorings_l118_118138


namespace kolya_correct_valya_incorrect_l118_118308

-- Definitions from the problem conditions:
def hitting_probability_valya (x : ℝ) : ℝ := 1 / (x + 1)
def hitting_probability_kolya (x : ℝ) : ℝ := 1 / x
def missing_probability_kolya (x : ℝ) : ℝ := 1 - (1 / x)
def missing_probability_valya (x : ℝ) : ℝ := 1 - (1 / (x + 1))

-- Proof for Kolya's assertion
theorem kolya_correct (x : ℝ) (hx : x > 0) : 
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2) :=
sorry

-- Proof for Valya's assertion
theorem valya_incorrect (x : ℝ) (hx : x > 0) :
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2 ∧ (r + p - r * p) / (1 - s * q) = 1 / 2) :=
sorry

end kolya_correct_valya_incorrect_l118_118308


namespace card_cannot_fit_l118_118187

-- Define the areas and the ratio
variable (area_card area_envelope : ℝ)
variable (ratio_length_width : ℝ)

-- Specify the conditions
def square_side_length (area : ℝ) := real.sqrt area
def envelope_dimensions (area : ℝ) (ratio : ℝ × ℝ) := 
  let x := real.sqrt (area / (ratio.1 * ratio.2)) in (ratio.1 * x, ratio.2 * x)

-- State the theorem
theorem card_cannot_fit (area_card area_envelope : ℝ) (ratio : ℝ × ℝ)
  (h1 : area_card = 144) 
  (h2 : area_envelope = 180) 
  (h3 : ratio = (4, 3)) : 
  let card_side := square_side_length area_card in 
  let (envelope_length, envelope_width) := envelope_dimensions area_envelope ratio in 
  card_side > max envelope_length envelope_width := 
by 
  -- Placeholder proof
  sorry

end card_cannot_fit_l118_118187


namespace part1_part2_l118_118036

def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- The first part of the problem
theorem part1 (a : ℝ) (h : A a ∩ B = ∅) : -1 ≤ a ∧ a ≤ 2 := 
by
  sorry

-- The second part of the problem
theorem part2 (a : ℝ) (h : A a ∪ B = B) : a ∈ (Set.Ioo Float.negInf (-4) ∪ Set.Ioo 5 Float.posInf) :=
by
  sorry

end part1_part2_l118_118036


namespace last_three_digits_of_7_pow_1992_l118_118684

open BigOperators

theorem last_three_digits_of_7_pow_1992 :
  ∀ (k : ℕ), k = 1992 → (7 ^ k % 1000 = 201) :=
by
  assume k,
  assume hk : k = 1992,
  -- Placeholder for proof
  sorry

end last_three_digits_of_7_pow_1992_l118_118684


namespace eggs_per_basket_l118_118905

-- Lucas places a total of 30 blue Easter eggs in several yellow baskets
-- Lucas places a total of 42 green Easter eggs in some purple baskets
-- Each basket contains the same number of eggs
-- There are at least 5 eggs in each basket

theorem eggs_per_basket (n : ℕ) (h1 : n ∣ 30) (h2 : n ∣ 42) (h3 : n ≥ 5) : n = 6 :=
by
  sorry

end eggs_per_basket_l118_118905


namespace proof_problem_l118_118392

theorem proof_problem (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y * z)) + (y / (1 + z * x)) + (z / (1 + x * y)) ∧ 
  (x / (1 + y * z)) + (y / (1 + z * x)) + (z / (1 + x * y)) ≤ Real.sqrt 2 :=
by
  sorry

end proof_problem_l118_118392


namespace min_positive_value_of_sum_l118_118342

open nat

theorem min_positive_value_of_sum :
  ∃ (a : fin 50 → ℤ), 
  (∀ i, a i = 1 ∨ a i = -1) ∧ 
  (∀ (i j : fin 50), i < j → 0 < ∑ i in finset.range 50, ∑ j in finset.Ico (i + 1) 50, (a i * a j)) ∧
  (∑ i in finset.range 50, ∑ j in finset.Ico (i + 1) 50, (a i * a j) = 7) :=
by {
  -- Proof goes here
  sorry
}

end min_positive_value_of_sum_l118_118342


namespace sum_third_largest_and_smallest_l118_118516

def third_largest_and_smallest_sum (digits : List ℕ) : ℕ :=
  let sorted_digits := digits.reverse.quickSort
  let third_largest := [sorted_digits[0], sorted_digits[1], sorted_digits[3], sorted_digits[2]].foldl (λ acc d, 10 * acc + d) 0
  let sorted_digits := digits.quickSort
  let third_smallest := [sorted_digits[0], sorted_digits[1], sorted_digits[3], sorted_digits[2]].foldl (λ acc d, 10 * acc + d) 0
  third_largest + third_smallest

theorem sum_third_largest_and_smallest :
  third_largest_and_smallest_sum [7, 6, 8, 5] = 14443 :=
  by sorry

end sum_third_largest_and_smallest_l118_118516


namespace probability_of_prime_roll_l118_118928

def is_prime (n : Nat) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def num_successful_outcomes := List.countp is_prime [1, 2, 3, 4, 5, 6, 7, 8]

def total_outcomes := 8

def probability_prime := num_successful_outcomes / total_outcomes

theorem probability_of_prime_roll :
  probability_prime = 1 / 2 :=
by
  sorry

end probability_of_prime_roll_l118_118928


namespace monthly_rent_calculation_l118_118282

noncomputable def monthly_rent (purchase_cost : ℕ) (maintenance_pct : ℝ) (annual_taxes : ℕ) (target_roi : ℝ) : ℝ :=
  let annual_return := target_roi * (purchase_cost : ℝ)
  let total_annual_requirement := annual_return + (annual_taxes : ℝ)
  let monthly_requirement := total_annual_requirement / 12
  let actual_rent := monthly_requirement / (1 - maintenance_pct)
  actual_rent

theorem monthly_rent_calculation :
  monthly_rent 12000 0.15 400 0.06 = 109.80 :=
by
  sorry

end monthly_rent_calculation_l118_118282


namespace min_value_l118_118367

-- Define the circles C1 and C2
def circle_C1 (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 9

def circle_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 1

-- Define the points M, N, and P
def point_M (x y : ℝ) : Prop :=
  circle_C1 x y

def point_N (x y : ℝ) : Prop :=
  circle_C2 x y

def point_P (x : ℝ) : Prop :=
  True

-- Define the line y = -1
def line_y_neg_1 (x y : ℝ) : Prop :=
  y = -1

-- Define the distance function
def distance (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

-- Define the minimum value condition
theorem min_value (xM yM xN yN xP : ℝ)
  (hM : point_M xM yM) 
  (hN : point_N xN yN)
  (hP : point_P xP)
  (h_line : line_y_neg_1 xP (-1)) :
  distance (xP, -1) (xM, yM) + distance (xP, -1) (xN, yN) = 5 * real.sqrt 2 - 4 :=
sorry

end min_value_l118_118367


namespace exist_integers_in_S_l118_118157

theorem exist_integers_in_S (n : ℤ) (S : finset ℤ) (h₁ : 1 < n) (h₂ : (3/4 : ℤ) * n < S.card) :
  ∃ (a b c : ℤ),
    a % n ∈ S ∧
    b % n ∈ S ∧
    c % n ∈ S ∧
    (a + b) % n ∈ S ∧
    (b + c) % n ∈ S ∧
    (c + a) % n ∈ S ∧ 
    (a + b + c) % n ∈ S := by
  sorry

end exist_integers_in_S_l118_118157


namespace count_3_digit_numbers_multiple_30_not_75_l118_118791

theorem count_3_digit_numbers_multiple_30_not_75 : 
  (finset.filter (λ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 30 = 0 ∧ n % 75 ≠ 0) (finset.range 1000)).card = 21 := sorry

end count_3_digit_numbers_multiple_30_not_75_l118_118791


namespace rhombus_area_l118_118324

-- Definition of square ABCD with side length 1 and midpoints E, F, G, H.
structure Square :=
  (A B C D E F G H : Point)
  (side_length : ℝ)
  (is_square : is_square ABCD)
  (midpoints : midpoints ABCD [E, F, G, H])

-- Intersection points M and N
structure Intersections (AF EC AG CH : Line) :=
  (M : Point)
  (N : Point)
  (intersect_AF_EC : intersects AF EC M)
  (intersect_AG_CH : intersects AG CH N)

-- The proof problem statement
theorem rhombus_area (sq : Square) (ints : Intersections (AF sq) (EC sq) (AG sq) (CH sq)) :
  is_rhombus (AMCN sq ints)
  ∧ area (AMCN sq ints) = 1 / 3 :=
sorry

end rhombus_area_l118_118324


namespace part1_proof_part2_proof_l118_118079

-- Definitions corresponding to the conditions in a)
variables (a b c BD : ℝ) (A B C : RealAngle)
variables (D : Point) (AD DC : ℝ)

-- Replace the conditions with the necessary hypotheses
hypothesis h1 : b^2 = a * c
hypothesis h2 : BD * sin B = a * sin C
hypothesis h3 : AD = 2 * DC

noncomputable def Part1 : Prop := BD = b

theorem part1_proof : Part1 a b c BD A B C do
  sorry

noncomputable def Part2 : Prop := cos B = 7 / 12

theorem part2_proof (hADDC : AD = 2 * DC) : Part2 A B C.
  sorry

end part1_proof_part2_proof_l118_118079


namespace complex_number_equality_l118_118745

noncomputable def z : ℂ := 1 - 2 * complex.I

theorem complex_number_equality : z = -complex.I := by
  -- omitted proof steps
  sorry

end complex_number_equality_l118_118745


namespace result_l118_118293

noncomputable def Kolya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  P_A = (1: ℝ) / 2

def Valya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  let P_B := (r * p) / (1 - s * q)
  let P_AorB := P_A + P_B
  P_AorB > (1: ℝ) / 2

theorem result (p r : ℝ) (h1 : Kolya_right p r) (h2 : ¬Valya_right p r) :
  Kolya_right p r ∧ ¬Valya_right p r :=
  ⟨h1, h2⟩

end result_l118_118293


namespace leading_coefficient_of_g_l118_118172

-- Let g be a polynomial that satisfies the functional equation
variable (g : ℕ → ℝ)
variable (h : ∀ x : ℕ, g (x + 1) - g x = 8 * (x : ℝ) + 6)

-- Theorem stating the leading coefficient of g is 4
theorem leading_coefficient_of_g : leading_coeff (polynomial.of_finsupp (λ x, g x)) = 4 :=
sorry

end leading_coefficient_of_g_l118_118172


namespace cricket_team_throwers_l118_118912

def cricket_equation (T N : ℕ) := 
  (2 * N / 3 = 51 - T) ∧ (T + N = 58)

theorem cricket_team_throwers : 
  ∃ T : ℕ, ∃ N : ℕ, cricket_equation T N ∧ T = 37 :=
by
  sorry

end cricket_team_throwers_l118_118912


namespace y_value_l118_118045

theorem y_value (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -6) : y = 29 :=
by
  sorry

end y_value_l118_118045


namespace equiv_or_neg_equiv_l118_118589

theorem equiv_or_neg_equiv (x y : ℤ) (h : (x^2) % 239 = (y^2) % 239) :
  (x % 239 = y % 239) ∨ (x % 239 = (-y) % 239) :=
by
  sorry

end equiv_or_neg_equiv_l118_118589


namespace solution_to_symmetry_problem_l118_118732

noncomputable def symmetric_points_problem (x y z : ℝ) : Prop :=
  let A := (x^2 + 4, 4 - y, 1 + 2z)
  let B := (-4x, 9, 7 - z)
  in A.1 = -B.1 ∧ (A.2, A.3) = (B.2, B.3)

theorem solution_to_symmetry_problem : ∃ (x y z : ℝ), symmetric_points_problem x y z ∧ x = 2 ∧ y = -5 ∧ z = -8 :=
by {
  use [2, -5, -8],
  unfold symmetric_points_problem,
  simp only [eq_self_iff_true, and_self, eq_neg_iff_add_eq_zero, true_and],
  split,
  { ring },
  split,
  { refl },
  exact smul_left_cancel_iff.1 rfl,
  sorry
}

end solution_to_symmetry_problem_l118_118732


namespace nitrogen_fixation_and_colony_size_l118_118223

variables (Rhizobia : Type)
variables (N2_air sterile_air : Rhizobia)
variables (fixation_size reproduction_rate : Rhizobia -> ℤ)
variables (aerobic_respiration : Rhizobia -> Prop)

-- Conditions
def increased_nitrogen_concentration (r : Rhizobia) : Prop := r = N2_air
def inhibited_aerobic_respiration (r : Rhizobia) : Prop := 
  increased_nitrogen_concentration r ∧ ¬ aerobic_respiration r
def strong_fixation_and_reproduction (r : Rhizobia) : Prop := 
  r = sterile_air ∧ aerobic_respiration r

-- Proof problem statement
theorem nitrogen_fixation_and_colony_size : 
  increased_nitrogen_concentration N2_air →
  inhibited_aerobic_respiration N2_air →
  strong_fixation_and_reproduction sterile_air →
  (fixation_size N2_air < fixation_size sterile_air) ∧
  (reproduction_rate N2_air < reproduction_rate sterile_air) :=
sorry

end nitrogen_fixation_and_colony_size_l118_118223


namespace find_y_l118_118115
-- Define the operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - a - 2

-- The theorem we want to prove
theorem find_y : ∃ y : ℝ, star 3 y = 25 ∧ y = 5 :=
begin
  have h1: star 3 y = 3 * y + 3 * y - 3 - 2, by {
    unfold star,
    sorry,
  },
  have h2: star 3 y = 6 * y - 5, by {
    sorry,
  },
  have h3: 6 * y - 5 = 25, by {
    sorry,
  },
  have h4: 6 * y = 30, by {
    sorry,
  },
  have h5: y = 5, by {
    sorry,
  },
  use 5,
  split,
  exact h3,
  exact h5,
end

end find_y_l118_118115


namespace min_positive_value_sum_50_l118_118348

theorem min_positive_value_sum_50 :
  (∃ a : Fin 50 → ℤ,
    (∀ i : Fin 50, a i = 1 ∨ a i = -1) ∧ 
    (∑ i : Fin 50, a i) % 2 = 0 ∧
    (let S := ∑ i in finset.range 50, ∑ j in finset.range i, a i * a j in
     S = 7)) :=
sorry

end min_positive_value_sum_50_l118_118348


namespace MrsHiltReadTotalChapters_l118_118599

-- Define the number of books and chapters per book
def numberOfBooks : ℕ := 4
def chaptersPerBook : ℕ := 17

-- Define the total number of chapters Mrs. Hilt read
def totalChapters (books : ℕ) (chapters : ℕ) : ℕ := books * chapters

-- The main statement to be proved
theorem MrsHiltReadTotalChapters : totalChapters numberOfBooks chaptersPerBook = 68 := by
  sorry

end MrsHiltReadTotalChapters_l118_118599


namespace circle_center_line_condition_l118_118749

theorem circle_center_line_condition (a : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 4 * y - 6 = 0 → (a, -2) = (x, y) → x + 2 * y + 1 = 0) → a = 3 :=
by
  sorry

end circle_center_line_condition_l118_118749


namespace flowers_wilted_l118_118456

theorem flowers_wilted (total_flowers : ℕ) (flowers_per_bouquet : ℕ) (bouquets : ℕ) (wilted : ℕ) (h1 : total_flowers = 66) (h2 : flowers_per_bouquet = 8) (h3 : bouquets = 7) : wilted = 10 := 
by
  let used_flowers := bouquets * flowers_per_bouquet
  have h4 : used_flowers = 56 := by rw [h2, h3]; norm_num
  let wilted := total_flowers - used_flowers
  have h5 : wilted = 10 := by rw [h1, h4]; norm_num
  exact h5

end flowers_wilted_l118_118456


namespace functional_equation_satisfied_for_continuous_f_l118_118119

variables {H : Type*} [inner_product_space ℝ H] [complete_space H]
variables {a : ℝ} (b : H → ℝ) (c : H →ₗ[ℝ] H)
variables [is_self_adjoint c]

noncomputable def f (z : H) : ℝ := a + b z + ⟪c z, z⟫

theorem functional_equation_satisfied_for_continuous_f :
  (∀ x y z : H, f (x + y + real.pi • z) + f (x + real.sqrt 2 • z) + 
    f (y + real.sqrt 2 • z) + f (real.pi • z)
    = f (x + y + real.sqrt 2 • z) + f (x + real.pi • z) + 
      f (y + real.pi • z) + f (real.sqrt 2 • z)) :=
sorry

end functional_equation_satisfied_for_continuous_f_l118_118119


namespace number_of_shorts_jimmy_picks_l118_118863

def cost_shorts := 15
def cost_shirts := 17
def discount_rate := 0.10
def total_money := 117
def number_of_shirts := 5

theorem number_of_shorts_jimmy_picks :
  ∃ n : ℕ, total_money - (number_of_shirts * cost_shirts * (1 - discount_rate)) = n * cost_shorts ∧ n = 2 :=
begin
  sorry
end

end number_of_shorts_jimmy_picks_l118_118863


namespace find_number_l118_118235

theorem find_number (x : ℝ) :
  let mean1 := (20 + 40 + 60) / 3 in
  let mean2 := (10 + 70 + x) / 3 in
  mean1 = mean2 + 4 →
  x = 28 :=
by
  assume h : (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4
  sorry

end find_number_l118_118235


namespace bucket_capacity_l118_118458

theorem bucket_capacity (jack_buckets_per_trip : ℕ)
                        (jill_buckets_per_trip : ℕ)
                        (jack_trip_ratio : ℝ)
                        (jill_trips : ℕ)
                        (tank_capacity : ℝ)
                        (bucket_capacity : ℝ)
                        (h1 : jack_buckets_per_trip = 2)
                        (h2 : jill_buckets_per_trip = 1)
                        (h3 : jack_trip_ratio = 3 / 2)
                        (h4 : jill_trips = 30)
                        (h5 : tank_capacity = 600) :
  bucket_capacity = 5 :=
by 
  sorry

end bucket_capacity_l118_118458


namespace cone_volume_correct_l118_118158

variable (π : ℝ) (r h : ℝ)
variables (cone_lateral_surface_area : ℝ := 2 / 3 * π) (slant_height : ℝ := 1)

noncomputable def cone_volume_formula (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

theorem cone_volume_correct : 
  (2 * π * r = 4 / 3 * π) → 
  (2 * π * r * slant_height / 2 = cone_lateral_surface_area) → 
  (h = sqrt (slant_height^2 - r^2)) →
  cone_volume_formula r h = (4 * sqrt 5 / 81) * π :=
by
  intros
  sorry

end cone_volume_correct_l118_118158


namespace monotonicity_and_inequality_l118_118027

noncomputable def f (x : ℝ) := 2 * Real.exp x
noncomputable def g (a : ℝ) (x : ℝ) := a * x + 2
noncomputable def F (a : ℝ) (x : ℝ) := f x - g a x

theorem monotonicity_and_inequality (a : ℝ) (x₁ x₂ : ℝ) (hF_nonneg : ∀ x, F a x ≥ 0) (h_lt : x₁ < x₂) :
  (F a x₂ - F a x₁) / (x₂ - x₁) > 2 * (Real.exp x₁ - 1) :=
sorry

end monotonicity_and_inequality_l118_118027


namespace probability_nine_heads_l118_118996

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118996


namespace floor_1000_cos_E_convex_quadrilateral_l118_118067

theorem floor_1000_cos_E_convex_quadrilateral 
  (EFGH : Quadrilateral)
  (convex_EFGH : EFGH.Convex)
  (eq_angle_E_G : EFGH.AngleE = EFGH.AngleG)
  (eq_side_EF_GH : EFGH.SideEF = 200 ∧ EFGH.SideGH = 200)
  (neq_side_EG_FH : EFGH.SideEG ≠ EFGH.SideFH)
  (perimeter_EFGH : EFGH.Perimeter = 720) :
  (floor (1000 * (cos EFGH.AngleE))) = 400 := 
sorry

end floor_1000_cos_E_convex_quadrilateral_l118_118067


namespace relationship_among_a_b_c_l118_118517

noncomputable def f : ℝ → ℝ := sorry

theorem relationship_among_a_b_c :
  (∀ x : ℝ, f(x) = f(2 - x)) ∧
  (∀ x : ℝ, x ≠ 1 → (x - 1) * (deriv f x) < 0) →
  f(4 / 3) > f(0.5) ∧ f(0.5) > f(3) :=
by {
  intro h,
  sorry
}

end relationship_among_a_b_c_l118_118517


namespace sin2_sub_cos2_range_l118_118885

variable (x y r : ℝ) (θ : ℝ)
hypothesis (h1 : sin θ = y / r)
hypothesis (h2 : cos θ = x / r)

theorem sin2_sub_cos2_range : -1 ≤ sin θ ^ 2 - cos θ ^ 2 ∧ sin θ ^ 2 - cos θ ^ 2 ≤ 1 :=
by
  sorry

end sin2_sub_cos2_range_l118_118885


namespace binom_sub_floor_divisible_by_prime_l118_118500

theorem binom_sub_floor_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - (n / p)) :=
sorry

end binom_sub_floor_divisible_by_prime_l118_118500


namespace part1_proof_part2_proof_l118_118082

-- Definitions corresponding to the conditions in a)
variables (a b c BD : ℝ) (A B C : RealAngle)
variables (D : Point) (AD DC : ℝ)

-- Replace the conditions with the necessary hypotheses
hypothesis h1 : b^2 = a * c
hypothesis h2 : BD * sin B = a * sin C
hypothesis h3 : AD = 2 * DC

noncomputable def Part1 : Prop := BD = b

theorem part1_proof : Part1 a b c BD A B C do
  sorry

noncomputable def Part2 : Prop := cos B = 7 / 12

theorem part2_proof (hADDC : AD = 2 * DC) : Part2 A B C.
  sorry

end part1_proof_part2_proof_l118_118082


namespace pyramid_bottom_right_value_l118_118936

theorem pyramid_bottom_right_value (a x y z b : ℕ) (h1 : 18 = (21 + x) / 2)
  (h2 : 14 = (21 + y) / 2) (h3 : 16 = (15 + z) / 2) (h4 : b = (21 + y) / 2) :
  a = 6 := 
sorry

end pyramid_bottom_right_value_l118_118936


namespace new_student_info_l118_118285

-- Definitions of the information pieces provided by each classmate.
structure StudentInfo where
  last_name : String
  gender : String
  total_score : Nat
  specialty : String

def student_A : StudentInfo := {
  last_name := "Ji",
  gender := "Male",
  total_score := 260,
  specialty := "Singing"
}

def student_B : StudentInfo := {
  last_name := "Zhang",
  gender := "Female",
  total_score := 220,
  specialty := "Dancing"
}

def student_C : StudentInfo := {
  last_name := "Chen",
  gender := "Male",
  total_score := 260,
  specialty := "Singing"
}

def student_D : StudentInfo := {
  last_name := "Huang",
  gender := "Female",
  total_score := 220,
  specialty := "Drawing"
}

def student_E : StudentInfo := {
  last_name := "Zhang",
  gender := "Female",
  total_score := 240,
  specialty := "Singing"
}

-- The theorem we need to prove based on the given conditions.
theorem new_student_info :
  ∃ info : StudentInfo,
    info.last_name = "Huang" ∧
    info.gender = "Male" ∧
    info.total_score = 240 ∧
    info.specialty = "Dancing" :=
  sorry

end new_student_info_l118_118285


namespace proof_vertices_and_circumcircle_l118_118400

noncomputable def vertices_of_triangle
  (C : (ℝ × ℝ))
  (line_AB : ℝ → ℝ)
  (line_BH : (ℝ × ℝ) → ℝ)
  (A : (ℝ × ℝ))
  (B : (ℝ × ℝ)) :=
  C = (2, -8) ∧
  (∀ x, line_AB x = -2 * x + 11) ∧
  (∀ p : ℝ × ℝ, line_BH p = p.1 + 3 * p.2 + 2) ∧
  A = (5, 1) ∧
  B = (7, -3)

noncomputable def circumcircle_of_triangle
  (A B C : (ℝ × ℝ))
  (circle : ℝ → ℝ → ℝ) :=
  A = (5, 1) ∧
  B = (7, -3) ∧
  C = (2, -8) ∧
  (circle = λ x y, x^2 + y^2 - 4 * x + 6 * y - 12)

theorem proof_vertices_and_circumcircle :
  ∃ A B : (ℝ × ℝ), ∃ circle : ℝ → ℝ → ℝ,
    vertices_of_triangle
      (2, -8)
      (λ x, -2 * x + 11)
      (λ p, p.1 + 3 * p.2 + 2)
      A
      B ∧
    circumcircle_of_triangle
      A
      B
      (2, -8)
      circle :=
  sorry

end proof_vertices_and_circumcircle_l118_118400


namespace num_both_sports_l118_118833

def num_people := 310
def num_tennis := 138
def num_baseball := 255
def num_no_sport := 11

theorem num_both_sports : (num_tennis + num_baseball - (num_people - num_no_sport)) = 94 :=
by 
-- leave the proof out for now
sorry

end num_both_sports_l118_118833


namespace triangle_is_right_l118_118807

-- Define the side lengths of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- Define a predicate to check if a triangle is right using Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- The proof problem statement
theorem triangle_is_right : is_right_triangle a b c :=
sorry

end triangle_is_right_l118_118807


namespace sequence_50th_term_is_111_l118_118960

-- Define the sequence condition
def is_in_sequence (n : ℕ) : Prop :=
  ∃ (s : List ℕ), (∀ x ∈ s, ∃ k : ℕ, x = 3 ^ k) ∧ (n = s.sum)

-- Define the 50th term of the sequence
def sequence_50th_term : ℕ :=
  Nat.find (ExistsInSequence 50)

theorem sequence_50th_term_is_111 : sequence_50th_term = 111 :=
by
  -- Proof omitted
  sorry

#check sequence_50th_term_is_111

end sequence_50th_term_is_111_l118_118960


namespace result_l118_118297

noncomputable def Kolya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  P_A = (1: ℝ) / 2

def Valya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  let P_B := (r * p) / (1 - s * q)
  let P_AorB := P_A + P_B
  P_AorB > (1: ℝ) / 2

theorem result (p r : ℝ) (h1 : Kolya_right p r) (h2 : ¬Valya_right p r) :
  Kolya_right p r ∧ ¬Valya_right p r :=
  ⟨h1, h2⟩

end result_l118_118297


namespace eq1_solution_eq2_no_solution_l118_118506

-- For Equation (1)
theorem eq1_solution (x : ℝ) (h : (3 / (2 * x - 2)) + (1 / (1 - x)) = 3) : 
  x = 7 / 6 :=
by sorry

-- For Equation (2)
theorem eq2_no_solution (y : ℝ) : ¬((y / (y - 1)) - (2 / (y^2 - 1)) = 1) :=
by sorry

end eq1_solution_eq2_no_solution_l118_118506


namespace soda_difference_l118_118102

-- Definitions for the amounts of soda each person has based on the conditions.

def soda_amount_julio : ℝ := 
  let orange_soda := 4 * 2 in
  let grape_soda := 7 * 2 in
  orange_soda + grape_soda

def soda_amount_mateo : ℝ := 
  let orange_soda := 1 * 2 in
  let grape_soda := 3 * 2 in
  orange_soda + grape_soda

def soda_amount_sophia : ℝ := 
  let orange_soda := 6 * 1.5 in
  let strawberry_soda := 5 * 2.5 in
  orange_soda + strawberry_soda

-- Theorem to prove the difference between the maximum and minimum soda amounts.
theorem soda_difference : 
  let max_soda := max soda_amount_julio (max soda_amount_mateo soda_amount_sophia) in
  let min_soda := min soda_amount_julio (min soda_amount_mateo soda_amount_sophia) in
  max_soda - min_soda = 14 := by {
  sorry
}

end soda_difference_l118_118102


namespace second_smallest_palindromic_prime_l118_118952

-- Three digit number definition
def three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Palindromic number definition
def is_palindromic (n : ℕ) : Prop := 
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds = ones 

-- Prime number definition
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Second-smallest three-digit palindromic prime
theorem second_smallest_palindromic_prime :
  ∃ n : ℕ, three_digit_number n ∧ is_palindromic n ∧ is_prime n ∧ 
  ∃ m : ℕ, three_digit_number m ∧ is_palindromic m ∧ is_prime m ∧ m > 101 ∧ m < n ∧ 
  n = 131 := 
by
  sorry

end second_smallest_palindromic_prime_l118_118952


namespace second_derivative_parametric_l118_118360

variable (t : ℝ)

noncomputable def x (t : ℝ) := Real.log t
noncomputable def y (t : ℝ) := t^3 + 2 * t + 4

theorem second_derivative_parametric :
  let dxdt := (deriv x t),
      dydt := (deriv y t),
      dy_dx := dydt / dxdt,
      d2y_dt2 := deriv (λ t, dy_dx) t,
      d2y_dx2 := d2y_dt2 / dxdt
  in d2y_dx2 = 9 * t ^ 3 + 2 * t :=
by
  sorry

end second_derivative_parametric_l118_118360


namespace min_positive_sum_of_products_l118_118343

-- Definitions based on the problem conditions
def a : Fin 50 → ℤ := sorry  -- Let a_i be integers (either 1 or -1) defined over the finite set {1, 2, ..., 50}

-- Sum over pairs (i, j) such that 1 ≤ i < j ≤ 50
def sum_of_products (a : Fin 50 → ℤ) : ℤ :=
  ∑ i in Finset.range 50, ∑ j in Finset.Ico (i+1) 50, a i * a j

-- The theorem we need to prove
theorem min_positive_sum_of_products : 
  (∀ i, a i = 1 ∨ a i = -1) → 0 < sum_of_products a → sum_of_products a = 7 :=
sorry

end min_positive_sum_of_products_l118_118343


namespace circle_colored_l118_118133

noncomputable def circle_coloring (n : ℕ) : Prop :=
  ∃ (S : Finset ℕ) (colors : Fin n → Bool),
    0 ∉ S ∧
    S ⊆ Finset.range n ∧
    2 ≤ S.card ∧
    (∀ x, colors x = colors ((x + 1) % n)) →
    ∃ d, 1 < d ∧ d < n ∧
          (∀ x y, (x ≠ y ∧ x % d ≠ 0 ∧ y % d ≠ 0 ∧ (x - y) % d = 0) → colors x = colors y)

theorem circle_colored (n : ℕ) (n_pos : 0 < n) :
  circle_coloring n := sorry

end circle_colored_l118_118133


namespace find_x_y_l118_118744

variables (x y : ℝ)

def a : ℝ × ℝ := (5, 6)
def b : ℝ × ℝ := (x, 3)
def c : ℝ × ℝ := (2, y)

-- Vector dot product
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Vectors are perpendicular if their dot product is zero
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Vectors are parallel if they are scalar multiples of each other
def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (v.1 = k * u.1) ∧ (v.2 = k * u.2)

theorem find_x_y : 
  is_perpendicular a b ∧ is_parallel a c → x = -18/5 ∧ y = 12/5 := by
  sorry

end find_x_y_l118_118744


namespace sum_possible_distances_l118_118416

theorem sum_possible_distances {A B : ℝ} (hAB : |A - B| = 2) (hA : |A| = 3) : 
  (if A = 3 then |B + 2| + |B - 2| else |B + 4| + |B - 4|) = 12 :=
by
  sorry

end sum_possible_distances_l118_118416


namespace solve_2x2_minus1_eq_3x_l118_118147
noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let root1 := (-b + sqrt_discriminant) / (2 * a)
  let root2 := (-b - sqrt_discriminant) / (2 * a)
  (root1, root2)

theorem solve_2x2_minus1_eq_3x :
  solve_quadratic 2 (-3) (-1) = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4 ) :=
by
  let roots := solve_quadratic 2 (-3) (-1)
  have : roots = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4) := by sorry
  exact this

end solve_2x2_minus1_eq_3x_l118_118147


namespace count_pos_3digit_multiples_of_30_not_75_l118_118796

theorem count_pos_3digit_multiples_of_30_not_75 : 
  let multiples_of_30 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 30 = 0}
  let multiples_of_75 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 75 = 0}
  (multiples_of_30 \ multiples_of_75).size = 24 :=
by
  sorry

end count_pos_3digit_multiples_of_30_not_75_l118_118796


namespace Nickel_ate_3_chocolates_l118_118502

-- Definitions of the conditions
def Robert_chocolates : ℕ := 12
def extra_chocolates : ℕ := 9
def Nickel_chocolates (N : ℕ) : Prop := Robert_chocolates = N + extra_chocolates

-- The proof goal
theorem Nickel_ate_3_chocolates : ∃ N : ℕ, Nickel_chocolates N ∧ N = 3 :=
by
  sorry

end Nickel_ate_3_chocolates_l118_118502


namespace problem1_problem2_l118_118083

variables (A B C D : Type) [Real.uniform_space A]
variables {a b c BD AD DC : Real}
variables (angle_ABC angle_ACB : Real)

def proof1 (h1 : b^2 = a * c) (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) : Prop :=
  BD = b

def proof2 (h3 : AD = 2 * DC) (h1 : b^2 = a * c) : Prop :=
  Real.cos angle_ABC = 7 / 12

theorem problem1 {A B C D : Type} [Real.uniform_space A]
  {a b c BD : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) :
  proof1 a b c BD angle_ABC angle_ACB h1 h2 :=
sorry

theorem problem2 {A B C D : Type} [Real.uniform_space A]
  {a b c BD AD DC : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h3 : AD = 2 * DC) :
  proof2 a b c BD angle_ABC h1 h3 :=
sorry

end problem1_problem2_l118_118083


namespace ratio_proof_l118_118799

theorem ratio_proof (a b c : ℝ) (ha : b / a = 3) (hb : c / b = 4) :
    (a + 2 * b) / (b + 2 * c) = 7 / 27 := by
  sorry

end ratio_proof_l118_118799


namespace permutation_7_2_is_42_l118_118243

theorem permutation_7_2_is_42 :
  ∀ (n k : ℕ), n = 7 → k = 2 → (n! / (n - k)!) = 42 :=
by
intros n k hn hk
rw [hn, hk]
exact 42

end permutation_7_2_is_42_l118_118243


namespace middle_number_l118_118537

theorem middle_number {a b c : ℚ} 
  (h1 : a + b = 15) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) 
  (h4 : c = 2 * a) : 
  b = 25 / 3 := 
by 
  sorry

end middle_number_l118_118537


namespace min_S_value_l118_118721

open BigOperators

def S (x : Fin n → ℤ) : ℤ :=
  (∑ i in Finset.range n, ∑ j in Finset.Ico i.succ n, x i * x j)

-- Main statement
theorem min_S_value (x : Fin n → ℤ)
  (hx : ∀ i, x i = 0 ∨ x i = 1 ∨ x i = -1) :
  S x = if even n then -n / 2 else -((n - 1) / 2) :=
sorry

end min_S_value_l118_118721


namespace find_symmetric_line_l118_118757

variables (x y : ℝ)

def l : Prop := x - y - 1 = 0
def l₁ : Prop := 2 * x - y - 2 = 0
def symmetric_line (l l₁ : Prop) : Prop :=
  ∃ l₂ : Prop, symmetric l₁ l l₂ ∧ l₂ = (x - 2 * y - 1) = 0

theorem find_symmetric_line :
  ∃ (l₂ : Prop), symmetric_line l l₁ l₂ :=
by
  sorry

end find_symmetric_line_l118_118757


namespace complex_abs_inequality_l118_118366

noncomputable def main : Prop :=
  ∀ (x y : ℝ), let z := x + y * complex.I in
    complex.abs z ≤ abs x + abs y

theorem complex_abs_inequality :
  main := 
by
  sorry

end complex_abs_inequality_l118_118366


namespace total_pages_is_1200_l118_118585

theorem total_pages_is_1200 (A B : ℕ) (h1 : 24 * (A + B) = 60 * A) (h2 : B = A + 10) : (60 * A) = 1200 := by
  sorry

end total_pages_is_1200_l118_118585


namespace max_value_of_function_l118_118524

-- Define the function y = 2 - x - 4 / x for x > 0
def func (x : ℝ) (h : x > 0) : ℝ := 2 - x - 4 / x

-- State and prove that the maximum value of the function is -2
theorem max_value_of_function : ∃ (c : ℝ), (∀ (x : ℝ), x > 0 → func x (by simp [x_pos]) ≤ c) ∧ c = -2 := by
  sorry

end max_value_of_function_l118_118524


namespace ellipse_standard_equation_l118_118727

noncomputable def semi_major_axis (major_axis_length : ℕ) : ℕ :=
  major_axis_length / 2

noncomputable def semi_focal_distance (semi_major_axis : ℕ) (eccentricity : ℚ) : ℕ :=
  semi_major_axis * eccentricity.toNat  -- Assuming eccentricity can be safely converted to natural number

noncomputable def semi_minor_axis_squared (semi_major_axis : ℕ) (semi_focal_distance : ℕ) : ℕ :=
  semi_major_axis^2 - semi_focal_distance^2

theorem ellipse_standard_equation : 
  ∀ (major_axis_length : ℕ) (eccentricity : ℚ),
    major_axis_length = 12 ∧ 
    eccentricity = 1/3 → 
    (∀ (x y : ℝ), 
      x^2/(semi_major_axis major_axis_length)^2 + y^2/(semi_minor_axis_squared (semi_major_axis major_axis_length) (semi_focal_distance (semi_major_axis major_axis_length) eccentricity)) = 1) :=
by
  intros major_axis_length eccentricity h x y
  let a := semi_major_axis major_axis_length
  let c := semi_focal_distance a eccentricity
  let b_sq := semi_minor_axis_squared a c
  have h1 : a = 6 := sorry
  have h2 : c = 2 := sorry
  have h3 : b_sq = 32 := sorry
  rw [h1, h2]
  sorry

end ellipse_standard_equation_l118_118727


namespace value_of_x_plus_y_div_y_l118_118438

variable (w x y : ℝ)
variable (hx : w / x = 1 / 6)
variable (hy : w / y = 1 / 5)

theorem value_of_x_plus_y_div_y : (x + y) / y = 11 / 5 :=
by
  sorry

end value_of_x_plus_y_div_y_l118_118438


namespace percentage_salt_l118_118641

-- Variables
variables {S1 S2 R : ℝ}

-- Conditions
def first_solution := S1
def second_solution := (25 / 100) * 19.000000000000007
def resulting_solution := 16

theorem percentage_salt (S1 S2 : ℝ) (H1: S2 = 19.000000000000007) 
(H2: (75 / 100) * S1 + (25 / 100) * S2 = 16) : 
S1 = 15 :=
by
    rw [H1] at H2
    sorry

end percentage_salt_l118_118641


namespace probability_nine_heads_l118_118991

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118991


namespace center_of_rectangle_on_segment_MN_l118_118869

theorem center_of_rectangle_on_segment_MN
  (A B C D P M N: Point ℝ)
  (h_rect : Rectangle A B C D)
  (hP_on_AD : P ∈ Segment A D)
  (hBPC_angle : ∠ B P C = 90)
  (hM_perp_A_BP : Perpendicular_From_Point_At_Segment A BP M)
  (hN_perp_D_CP : Perpendicular_From_Point_At_Segment D CP N) :
  Center A B C D ∈ Segment M N :=
sorry

end center_of_rectangle_on_segment_MN_l118_118869


namespace unique_integer_solution_l118_118692

theorem unique_integer_solution (x y : ℤ) : 
  x^4 + y^4 = 3 * x^3 * y → x = 0 ∧ y = 0 :=
by
  -- This is where the proof would go
  sorry

end unique_integer_solution_l118_118692


namespace dot_product_of_vectors_l118_118812

variables (a b : EuclideanSpace ℝ (Fin 2))

def vector_norm (v : EuclideanSpace ℝ (Fin 2)) : ℝ := Real.sqrt (v.inner v)

noncomputable def angle := 60 * Real.pi / 180

theorem dot_product_of_vectors
  (ha : vector_norm a = 2)
  (hb : vector_norm b = 1)
  (angle_ab : ∀x y : EuclideanSpace ℝ (Fin 2), 
                vector_angle x y = angle) : 
  a ⬝ b = 1 := sorry

end dot_product_of_vectors_l118_118812


namespace zero_exists_in_interval_l118_118967

noncomputable def f (x : ℝ) : ℝ := 2^x - x^3

theorem zero_exists_in_interval : ∃ x ∈ (set.Ioo 1 2), f x = 0 :=
by sorry

end zero_exists_in_interval_l118_118967


namespace angle_double_l118_118483

-- Define the lengths of the legs in the first triangle
def longer_leg_first_triangle : ℝ := 1
def shorter_leg_first_triangle (t : ℝ) : ℝ := t

-- Define the legs of the second triangle
def leg1_second_triangle (t : ℝ) : ℝ := 2 * t
def leg2_second_triangle (t : ℝ) : ℝ := 1 - t^2

-- Prove the angle relationship
theorem angle_double (t : ℝ) (h1 : t > 0) (h2 : t < 1) :
  let alpha := real.arctan t in
  let beta := real.arctan (2 * t / (1 - t^2)) in
  beta = 2 * alpha :=
by
  sorry

end angle_double_l118_118483


namespace min_positive_value_of_sum_l118_118341

open nat

theorem min_positive_value_of_sum :
  ∃ (a : fin 50 → ℤ), 
  (∀ i, a i = 1 ∨ a i = -1) ∧ 
  (∀ (i j : fin 50), i < j → 0 < ∑ i in finset.range 50, ∑ j in finset.Ico (i + 1) 50, (a i * a j)) ∧
  (∑ i in finset.range 50, ∑ j in finset.Ico (i + 1) 50, (a i * a j) = 7) :=
by {
  -- Proof goes here
  sorry
}

end min_positive_value_of_sum_l118_118341


namespace binom_not_relatively_prime_l118_118498

theorem binom_not_relatively_prime (n k l : ℕ) (hn : 1 ≤ n) (hk : 1 ≤ k) (hl : 1 ≤ l) (hkn : k < n) (hln : l < n) :
  ¬ nat.coprime (nat.choose n k) (nat.choose n l) :=
sorry

end binom_not_relatively_prime_l118_118498


namespace leading_coefficient_of_g_l118_118167

theorem leading_coefficient_of_g (g : ℤ → ℤ) (h : ∀ x : ℤ, g(x + 1) - g(x) = 8 * x + 6) : ∃ d : ℤ, leading_coeff 4 (λ x, 4 * x * x + 4 * x + d) :=
by sorry

end leading_coefficient_of_g_l118_118167


namespace total_boundary_length_of_bolded_figure_l118_118277

theorem total_boundary_length_of_bolded_figure 
  (area_square : ℝ) (points_per_side : ℕ) (side_length : ℝ) 
  (segments_per_side : ℕ) (segment_length : ℝ) 
  (diameter : ℝ) (radius : ℝ) (arc_length : ℝ) 
  (total_length : ℝ) :
  area_square = 64 ∧ 
  side_length = real.sqrt 64 ∧ 
  points_per_side = 2 ∧ 
  segments_per_side = 4 ∧ 
  segment_length = side_length / segments_per_side ∧ 
  diameter = segment_length ∧ 
  radius = diameter / 2 ∧ 
  arc_length = real.pi * radius ∧ 
  total_length = (4 * segment_length + 4 * arc_length) ∧ 
  abs (total_length - 20.6) < 0.1 :=
by sorry

end total_boundary_length_of_bolded_figure_l118_118277


namespace tank_capacity_is_900_l118_118279

noncomputable def pipe_filling_capacity : Type :=
  ∃ (capacity : ℕ),  
    (∀ A_rate B_rate C_rate : ℕ, 
      A_rate = 40 ∧ B_rate = 30 ∧ C_rate = 20 →
      ∀ cycle_minutes : ℕ, 
        cycle_minutes = 54 →
        (∃ cycles : ℕ, cycles = cycle_minutes / 3 ∧
          (cycles * (A_rate + B_rate - C_rate) = capacity)))

theorem tank_capacity_is_900 : pipe_filling_capacity :=
  ⟨900, λ A_rate B_rate C_rate rates_eq cycle_minutes cycle_minutes_eq, 
    begin
      obtain ⟨cycles_eq, capacity_eq⟩ : 
        cycle_minutes / 3 = 18 /\
        18 * (A_rate + B_rate - C_rate) = 900,
      {
        sorry
      }
    end⟩

end tank_capacity_is_900_l118_118279


namespace jason_watermelons_l118_118461

theorem jason_watermelons (total_wm sandy_wm jason_wm : ℕ) 
  (h_total : total_wm = 48)
  (h_sandy : sandy_wm = 11)
  (h_sum : total_wm = sandy_wm + jason_wm) : 
  jason_wm = 37 := 
by 
  have h1 : 48 = 11 + jason_wm, from eq.subst h_total h_sum,
  have h2 : 48 - 11 = jason_wm, by exact eq.subst h_sandy (nat.sub_eq_of_eq_add (eq.symm h1)),
  exact eq.symm h2

end jason_watermelons_l118_118461


namespace probability_nine_heads_l118_118989

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118989


namespace stars_substitution_correct_l118_118144

-- Define x and y with given conditions
def ends_in_5 (n : ℕ) : Prop := n % 10 = 5
def product_ends_in_25 (x y : ℕ) : Prop := (x * y) % 100 = 25
def tens_digit_even (n : ℕ) : Prop := (n / 10) % 2 = 0
def valid_tens_digit (n : ℕ) : Prop := (n / 10) % 10 ≤ 3

theorem stars_substitution_correct :
  ∃ (x y : ℕ), ends_in_5 x ∧ ends_in_5 y ∧ product_ends_in_25 x y ∧ tens_digit_even x ∧ valid_tens_digit y ∧ x * y = 9125 :=
sorry

end stars_substitution_correct_l118_118144


namespace no_valid_n_l118_118368

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem no_valid_n : ∀ n : ℕ, n + sum_of_digits n + sum_of_digits (sum_of_digits n) ≠ 2023 :=
by
  assume n
  sorry

end no_valid_n_l118_118368


namespace determine_right_triangle_l118_118228

-- Definitions based on conditions
def condition_A (A B C : ℝ) : Prop := A^2 + B^2 = C^2
def condition_B (A B C : ℝ) : Prop := A^2 - B^2 = C^2
def condition_C (A B C : ℝ) : Prop := A + B = C
def condition_D (A B C : ℝ) : Prop := A / B = 3 / 4 ∧ B / C = 4 / 5

-- Problem statement: D cannot determine that triangle ABC is a right triangle
theorem determine_right_triangle (A B C : ℝ) : ¬ condition_D A B C :=
by sorry

end determine_right_triangle_l118_118228


namespace solve_for_X_l118_118704

theorem solve_for_X : ∃ X : ℝ, 1.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1200.0000000000002 ∧ X = 0.5 := by
  use 0.5
  split
  . sorry -- Proof for the mathematical equality
  . sorry -- Proof that X = 0.5 satisfies the equation

end solve_for_X_l118_118704


namespace count_valid_points_l118_118855

def is_valid_point (a b : ℕ) : Prop :=
  a ≠ b ∧ a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} ∧ (a^2 + b^2 ≥ 25)

theorem count_valid_points : (finset.univ.product finset.univ).filter (λ p : ℕ × ℕ, is_valid_point p.1 p.2).card = 20 :=
by sorry

end count_valid_points_l118_118855


namespace survey_representative_l118_118548

universe u

inductive SurveyOption : Type u
| A : SurveyOption  -- Selecting a class of students
| B : SurveyOption  -- Selecting 50 male students
| C : SurveyOption  -- Selecting 50 female students
| D : SurveyOption  -- Randomly selecting 50 eighth-grade students

def most_appropriate_survey : SurveyOption := SurveyOption.D

theorem survey_representative : most_appropriate_survey = SurveyOption.D := 
by sorry

end survey_representative_l118_118548


namespace sqrt_mul_eq_l118_118575

theorem sqrt_mul_eq {a b: ℝ} (ha: 0 ≤ a) (hb: 0 ≤ b): sqrt(a * b) = sqrt(a) * sqrt(b) :=
  by sorry

example : sqrt(2) * sqrt(3) = sqrt(6) :=
  by apply sqrt_mul_eq; norm_num

end sqrt_mul_eq_l118_118575


namespace part1_part2_l118_118760

section part1

variable (x : ℕ → ℝ)
variable h1 : x 1 = 1 / 2
variable h2 : ∀ n : ℕ, x (n + 2) = x (n + 1) / (2 - x (n + 1))

theorem part1 : ∀ n : ℕ, 0 < x (n + 1) ∧ x (n + 1) < 1 :=
begin
  sorry
end

end part1

section part2

variable (x : ℕ → ℝ)
variable (a : ℕ → ℝ)
variable h1 : x 1 = 1 / 2
variable h2 : ∀ n : ℕ, x (n + 2) = x (n + 1) / (2 - x (n + 1))
variable h3 : ∀ n : ℕ, a (n + 1) = 1 / x (n + 1)

theorem part2 : ∀ n : ℕ, a (n + 1) = 2 ^ n + 1 :=
begin
  sorry
end

end part2

end part1_part2_l118_118760


namespace area_of_transformed_triangle_l118_118113

noncomputable def matrix90 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

def vec_a := ![3, 2]
def vec_b := ![-1, 5]
def transformed_a := matrix90.mul_vec vec_a
def transformed_b := matrix90.mul_vec vec_b

def parallelogram_area (v w : Fin 2 → ℝ) : ℝ :=
  v 0 * w 1 - v 1 * w 0

def triangle_area (v w : Fin 2 → ℝ) : ℝ :=
  parallelogram_area v w / 2

theorem area_of_transformed_triangle :
  triangle_area transformed_a transformed_b = 17 / 2 := by
  sorry

end area_of_transformed_triangle_l118_118113


namespace coeff_x_in_expansion_l118_118937

-- Define binomial coefficients and general term formula
def binom_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Expansion term for (2 - x)^4 with general term C_4^r * 2^(4-r) * (-x)^r
def expansion_term (n : ℕ) (r : ℕ) (x : ℝ) : ℝ := binom_coefficient n r * 2^(n - r) * (-x)^r

-- Question: Prove that the coefficient of x in (1 + x)(2 - x)^4 is -16
theorem coeff_x_in_expansion : 
  let x := 1 in 
  ∑ r in Finset.range 5, (expansion_term 4 r x) 
  + ∑ r in Finset.range 4, (expansion_term 4 r x * x) 
  = (-16 : ℝ) := 
by 
  sorry

end coeff_x_in_expansion_l118_118937


namespace sum_of_transformed_parabolas_is_non_horizontal_line_l118_118630

theorem sum_of_transformed_parabolas_is_non_horizontal_line
    (a b c : ℝ)
    (f g : ℝ → ℝ)
    (hf : ∀ x, f x = a * (x - 8)^2 + b * (x - 8) + c)
    (hg : ∀ x, g x = -a * (x + 8)^2 - b * (x + 8) - (c - 3)) :
    ∃ m q : ℝ, ∀ x : ℝ, (f x + g x) = m * x + q ∧ m ≠ 0 :=
by sorry

end sum_of_transformed_parabolas_is_non_horizontal_line_l118_118630


namespace perpendicular_line_distance_l118_118436

noncomputable def distance_from_point_to_line
    (a b c x1 y1 : ℝ) : ℝ :=
    |a * x1 + b * y1 + c| / Real.sqrt (a^2 + b^2)

theorem perpendicular_line_distance :
    ∀ (m : ℝ), (2 * (-2) * -1 / m = -1) →
    distance_from_point_to_line 1 1 3 (-2) (-2) = Real.sqrt 2 / 2 :=
begin
    sorry
end

end perpendicular_line_distance_l118_118436


namespace kolya_correct_valya_incorrect_l118_118310

-- Definitions from the problem conditions:
def hitting_probability_valya (x : ℝ) : ℝ := 1 / (x + 1)
def hitting_probability_kolya (x : ℝ) : ℝ := 1 / x
def missing_probability_kolya (x : ℝ) : ℝ := 1 - (1 / x)
def missing_probability_valya (x : ℝ) : ℝ := 1 - (1 / (x + 1))

-- Proof for Kolya's assertion
theorem kolya_correct (x : ℝ) (hx : x > 0) : 
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2) :=
sorry

-- Proof for Valya's assertion
theorem valya_incorrect (x : ℝ) (hx : x > 0) :
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2 ∧ (r + p - r * p) / (1 - s * q) = 1 / 2) :=
sorry

end kolya_correct_valya_incorrect_l118_118310


namespace distance_covered_l118_118591

-- Define the given values
def time_in_hours : ℝ := 24 / 60  -- time converted to hours
def speed : ℝ := 10  -- speed in km/hr

-- Define the theorem for distance
theorem distance_covered (t : ℝ) (s : ℝ) : t = time_in_hours → s = speed → s * t = 4 :=
by
  intros ht hs
  rw [ht, hs]
  sorry

end distance_covered_l118_118591


namespace bee_loss_rate_l118_118255

theorem bee_loss_rate (initial_bees : ℕ) (days : ℕ) (remaining_bees : ℕ) :
  initial_bees = 80000 → 
  days = 50 → 
  remaining_bees = initial_bees / 4 → 
  (initial_bees - remaining_bees) / days = 1200 :=
by
  intros h₁ h₂ h₃
  sorry

end bee_loss_rate_l118_118255


namespace consecutive_no_carry_pairs_count_l118_118706

-- Define the problem conditions
def isHundredsDigitFiveOrSix (n : ℕ) : Prop :=
  (n / 100) % 10 = 5 ∨ (n / 100) % 10 = 6

def noCarryingInAddition (n : ℕ) : Prop :=
  n % 10 ≠ 9

-- The main statement to be proved:
theorem consecutive_no_carry_pairs_count : 
  (Finset.range 1001).filter (λ n, isHundredsDigitFiveOrSix (1500 + n) ∧ noCarryingInAddition (1500 + n)).card = 200 :=
by
  sorry

end consecutive_no_carry_pairs_count_l118_118706


namespace poly_ineq_solution_l118_118370

-- Define the inequality conversion
def poly_ineq (x : ℝ) : Prop :=
  x^2 + 2 * x ≤ -1

-- Formalize the set notation for the solution
def solution_set : Set ℝ :=
  { x | x = -1 }

-- State the theorem
theorem poly_ineq_solution : {x : ℝ | poly_ineq x} = solution_set :=
by
  sorry

end poly_ineq_solution_l118_118370


namespace prime_sum_not_always_composite_l118_118202

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_prime (p : ℕ) : Prop := nat.prime p

-- The predicate that indicates if a number is composite
def is_composite (n : ℕ) : Prop := ∃ m (h : 1 < m) (h2 : m < n), m ∣ n

theorem prime_sum_not_always_composite :
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ ¬ is_prime (p + q) ∧ ¬ is_composite (p + q) := sorry

end prime_sum_not_always_composite_l118_118202


namespace length_DC_eq_15_l118_118352

open EuclideanGeometry

-- Setting up the conditions and variables
variable (A B C D F E : Point)
variable (line_AB : Line)
variable (line_DC : Line)

-- Define the conditions in the problem
axiom trapezoid_ABCD : trapezoid A B C D
axiom AB_parallel_DC : parallel line_AB line_DC
axiom length_AB : distance A B = 6
axiom length_BC : distance B C = 3 * Real.sqrt 3
axiom angle_BCD : angle B C D = 60
axiom angle_CDA : angle C D A = 60

-- The theorem to be proven
theorem length_DC_eq_15 : distance D C = 15 :=
by 
  sorry

end length_DC_eq_15_l118_118352


namespace max_distance_circle_line_l118_118631

def circle (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

def line (t : ℝ) : ℝ × ℝ := (t, 1 + t)

def distance_point_line (point : ℝ × ℝ) (A B C : ℝ) : ℝ :=
  abs (A * point.1 + B * point.2 + C) / Real.sqrt (A^2 + B^2)

theorem max_distance_circle_line : 
  let distance_center_to_line := distance_point_line (1, 0) 1 (-1) 1
  distance_center_to_line = Real.sqrt 2 →
  ∀ α : ℝ, ∃ t : ℝ, abs (sqrt ((1 + Real.cos α - t)^2 + (Real.sin α - (1 + t))^2)) ≤ Real.sqrt 2 + 1 :=
by intros
   exact sorry

end max_distance_circle_line_l118_118631


namespace laura_park_time_percentage_l118_118105

theorem laura_park_time_percentage (num_trips: ℕ) (time_in_park: ℝ) (walking_time: ℝ) 
    (total_percentage_in_park: ℝ) 
    (h1: num_trips = 6) 
    (h2: time_in_park = 2) 
    (h3: walking_time = 0.5) 
    (h4: total_percentage_in_park = 80) : 
    (time_in_park * num_trips) / ((time_in_park + walking_time) * num_trips) * 100 = total_percentage_in_park :=
by
  sorry

end laura_park_time_percentage_l118_118105


namespace a_parallel_c_l118_118762

variables {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]


def is_perpendicular (u v : V) : Prop := ⟪u, v⟫ = 0
def is_parallel (u v : V) : Prop := ∃ k : ℝ, k ≠ 0 ∧ u = k • v

-- Given three lines as vectors
variables (a b c : V)

-- Conditions
axiom a_perp_b : is_perpendicular a b
axiom b_perp_c : is_perpendicular b c

-- Theorem to prove
theorem a_parallel_c {a b c : V} (a_perp_b : is_perpendicular a b)
  (b_perp_c : is_perpendicular b c) : is_parallel a c :=
sorry

end a_parallel_c_l118_118762


namespace find_line_equation_l118_118071

noncomputable def line_equation (k b: ℝ) : affine_plane ℝ :=
λ P : ℝ × ℝ, P.snd = k * P.fst + b

theorem find_line_equation : 
  ∃ k b : ℝ, (∀ x y : ℝ, (x, y) ∈ line_equation k b ↔ y = k * x + b) ∧
    (∀ x y : ℝ, (x, y) ∈ line_equation k b →
      ∃ x' y' : ℝ, (x', y') ∈ line_equation k (5 - 3 * k + b) ∧
      x' = x + 3 ∧ y' = y + 5) ∧
    (∀ x y : ℝ, (x, y) ∈ line_equation k (5 - 3 * k + b) →
      ∃ x' y' : ℝ, (x', y') ∈ line_equation k (3 - 4 * k + b) ∧
      x' = x + 1 ∧ y' = y - 2) ∧
    (∀ x y : ℝ, (x, y) ∈ line_equation k b →
      ∃ x' y' : ℝ, (x', y') ∈ line_equation k (3.5 - 3 * k) ∧
      (x', y') = (4 - x, 6 - y)) ∧
    k = 3/4 ∧ b = 1/8 :=
sorry

end find_line_equation_l118_118071


namespace leading_coefficient_of_g_l118_118170

-- Let g be a polynomial that satisfies the functional equation
variable (g : ℕ → ℝ)
variable (h : ∀ x : ℕ, g (x + 1) - g x = 8 * (x : ℝ) + 6)

-- Theorem stating the leading coefficient of g is 4
theorem leading_coefficient_of_g : leading_coeff (polynomial.of_finsupp (λ x, g x)) = 4 :=
sorry

end leading_coefficient_of_g_l118_118170


namespace angle_A_in_triangle_ABC_range_of_cosB_minus_sqrt3_sinC_l118_118824

-- Part I
theorem angle_A_in_triangle_ABC (a b c : ℝ) (A B C : ℝ) (h₁ : ∠A = asin(a/b) ∨ ∠A = π - asin(a/b))(h₂ : 2 * a * cos A = b * cos C + c * cos B) :
  A = π / 3 :=
by {
  sorry,
}

-- Part II
theorem range_of_cosB_minus_sqrt3_sinC (B : ℝ) (C : ℝ) (h₁ : C = 2 * π / 3 - B) (h₂ : ∀ B, 0 < B ∧ B < 2 * π / 3):
  ∀ x, (x = cos B - √3 * sin C) → -1 ≤ x ∧ x < -1 / 2 :=
by {
  sorry,
}

end angle_A_in_triangle_ABC_range_of_cosB_minus_sqrt3_sinC_l118_118824


namespace basketball_football_difference_l118_118832

def U := Finset.univ
def A := {x // x = (1 : ℕ) ∨ x = 2 ∨ x = 3} -- representing 23 basketball enthusiasts
def B := {x // x = (4 : ℕ) ∨ x = 5 ∨ x = 6 ∨ x = 7} -- representing 29 football enthusiasts
def m := 23
def n := 23 + 29 - 46

theorem basketball_football_difference :
  m - n = 17 := by
  sorry

end basketball_football_difference_l118_118832


namespace usual_time_is_120_l118_118563

variable (S T : ℕ) (h1 : 0 < S) (h2 : 0 < T)
variable (h3 : (4 : ℚ) / 3 = 1 + (40 : ℚ) / T)

theorem usual_time_is_120 : T = 120 := by
  sorry

end usual_time_is_120_l118_118563


namespace cubic_root_bound_l118_118875

theorem cubic_root_bound (a b c λ : ℂ) 
  (h₀ : 1 ≥ a) 
  (h₁ : a ≥ b) 
  (h₂ : b ≥ c) 
  (h₃ : c ≥ 0)
  (h₄ : λ^3 + a * λ^2 + b * λ + c = 0) : 
  |λ| ≤ 1 := 
sorry

end cubic_root_bound_l118_118875


namespace total_students_l118_118057

-- Given definitions
def basketball_count : ℕ := 7
def cricket_count : ℕ := 5
def both_count : ℕ := 3

-- The goal to prove
theorem total_students : basketball_count + cricket_count - both_count = 9 :=
by
  sorry

end total_students_l118_118057


namespace reese_spent_in_april_l118_118143

theorem reese_spent_in_april :
  ∀ (initial_savings spent_in_february spent_in_march remaining_savings spent_in_april : ℝ),
  initial_savings = 11000 →
  spent_in_february = 0.2 * initial_savings →
  spent_in_march = 0.4 * initial_savings →
  remaining_savings = 2900 →
  initial_savings - spent_in_february - spent_in_march - spent_in_april = remaining_savings →
  spent_in_april = 1500 :=
by
  intros initial_savings spent_in_february spent_in_march remaining_savings spent_in_april
  assume h_initial_savings h_spent_in_february h_spent_in_march h_remaining_savings h_final_equation
  sorry

end reese_spent_in_april_l118_118143


namespace solve_for_r_l118_118668

-- Define the vectors a and b
def a : ℝ × ℝ × ℝ := (3, 1, -2)
def b : ℝ × ℝ × ℝ := (1, 2, -2)

-- Define the target vector
def c : ℝ × ℝ × ℝ := (5, 3, -7)

-- Compute the cross product of a and b
def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

-- a × b
def a_cross_b : ℝ × ℝ × ℝ := cross_product a b

-- Prove r = -13/45 under the given conditions
theorem solve_for_r (p q r : ℝ) :
  let p_a_q_b_r_cross := (λ (p q r : ℝ), (p * a.fst + q * b.fst + r * a_cross_b.fst,
                                            p * a.snd + q * b.snd + r * a_cross_b.snd,
                                            p * a.thd + q * b.thd + r * a_cross_b.thd)) in
  p_a_q_b_r_cross p q r = c → r = -13 / 45 := by
  -- Proof skipped
  sorry

end solve_for_r_l118_118668


namespace leading_coefficient_four_l118_118174

noncomputable def poly_lead_coefficient (g : ℕ → ℕ → ℕ) : Prop :=
  ∀ x : ℕ, g(x + 1) - g(x) = 8 * x + 6

theorem leading_coefficient_four {g : ℕ → ℕ} 
  (h : poly_lead_coefficient g) : 
  ∃ a b c : ℕ, ∀ x : ℕ, g x = 4 * x^2 + 2 * x + c :=
begin
  sorry
end

end leading_coefficient_four_l118_118174


namespace kolya_correct_valya_incorrect_l118_118303

variable (x : ℝ)

def p : ℝ := 1 / x
def r : ℝ := 1 / (x + 1)
def q : ℝ := (x - 1) / x
def s : ℝ := x / (x + 1)

theorem kolya_correct : r / (1 - s * q) = 1 / 2 := by
  sorry

theorem valya_incorrect : (q * r + p * r) / (1 - s * q) = 1 / 2 := by
  sorry

end kolya_correct_valya_incorrect_l118_118303


namespace roots_calculation_l118_118477

theorem roots_calculation (c d : ℝ) (h : c^2 - 5*c + 6 = 0) (h' : d^2 - 5*d + 6 = 0) :
  c^3 + c^4 * d^2 + c^2 * d^4 + d^3 = 503 := by
  sorry

end roots_calculation_l118_118477


namespace survivor_quitting_probability_l118_118533

noncomputable def probability_all_quitters_same_tribe : ℚ :=
  let total_contestants := 20
  let tribe_size := 10
  let total_quitters := 3
  let total_ways := (Nat.choose total_contestants total_quitters)
  let tribe_quitters_ways := (Nat.choose tribe_size total_quitters)
  (tribe_quitters_ways + tribe_quitters_ways) / total_ways

theorem survivor_quitting_probability :
  probability_all_quitters_same_tribe = 4 / 19 :=
by
  sorry

end survivor_quitting_probability_l118_118533


namespace scientific_notation_of_twenty_million_l118_118930

-- Define the number 20 million
def twenty_million : ℂ :=
  20000000

-- Define the scientific notation to be proved correct
def scientific_notation : ℂ :=
  2 * 10 ^ 7

-- The theorem to prove the equivalence
theorem scientific_notation_of_twenty_million : twenty_million = scientific_notation :=
  sorry

end scientific_notation_of_twenty_million_l118_118930


namespace parking_garage_full_spots_l118_118265

theorem parking_garage_full_spots :
  let total_spots := 4 * 100,
      open_first_level := 58,
      open_second_level := open_first_level + 2,
      open_third_level := open_second_level + 5,
      open_fourth_level := 31,
      total_open_spots := open_first_level + open_second_level + open_third_level + open_fourth_level in
  total_spots - total_open_spots = 186 :=
by
  let total_spots := 4 * 100
  let open_first_level := 58
  let open_second_level := open_first_level + 2
  let open_third_level := open_second_level + 5
  let open_fourth_level := 31
  let total_open_spots := open_first_level + open_second_level + open_third_level + open_fourth_level
  show total_spots - total_open_spots = 186
  sorry

end parking_garage_full_spots_l118_118265


namespace triangle_QDC_isosceles_l118_118474

-- Definitions of the circles, points and lines.
variables (Γ1 Γ2 : Set Point)
variable {P Q A B C D : Point}
variable (h_intersect : P ∈ Γ1 ∧ P ∈ Γ2 ∧ Q ∈ Γ1 ∧ Q ∈ Γ2)
variable (hA : A ∈ Γ1)
variable (hB : B ∈ Γ2 ∧ collinear A P B ∧ ¬(B = A))
variable (hC : C ∈ Γ2 ∧ collinear A Q C ∧ ¬(C = A))
variable (h_tangent : is_tangent_at Γ1 Q D ∧ intersect_with_line D B C)

-- The goal is to show that triangle QDC is isosceles at D, i.e., angle QDC = angle QCD.
theorem triangle_QDC_isosceles :
  ∀ {Γ1 Γ2 : Set Point} {P Q A B C D : Point},
    P ∈ Γ1 ∧ P ∈ Γ2 ∧ Q ∈ Γ1 ∧ Q ∈ Γ2 →
    A ∈ Γ1 →
    B ∈ Γ2 ∧ collinear A P B ∧ ¬(B = A) →
    C ∈ Γ2 ∧ collinear A Q C ∧ ¬(C = A) →
    is_tangent_at Γ1 Q D ∧ intersect_with_line D B C →
    is_isosceles Q D C :=
by sorry -- skipped proof initialization

end triangle_QDC_isosceles_l118_118474


namespace solve_for_x_l118_118221

theorem solve_for_x : ∃ x : ℝ, (2010 + x)^3 = -x^3 ∧ x = -1005 := 
by
  use -1005
  sorry

end solve_for_x_l118_118221


namespace problem_statement_l118_118072

-- Define the points A, B, C
structure Point where
  x : ℝ
  y : ℝ

-- Define the vectors AB, AC, and OC
def vector (p1 p2 : Point) : Point :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

-- Dot product of two vectors
def dot_product (v1 v2 : Point) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

-- Magnitude of a vector
def magnitude (v : Point) : ℝ :=
  Real.sqrt (v.x * v.x + v.y * v.y)

-- Test conditions
def A := Point.mk 1 4
def B := Point.mk (-2) 3
def C := Point.mk 2 (-1)
def O := Point.mk 0 0

def AB := vector A B
def AC := vector A C
def OC := vector O C
def AB_plus_AC := vector A B + vector A C

theorem problem_statement :
  (dot_product AB AC = 2) ∧ 
  (magnitude AB_plus_AC = 2 * Real.sqrt 10) ∧
  ∃ t : ℝ, ((dot_product (Point.mk (AB.x - t * OC.x) (AB.y - t * OC.y)) OC = 0) → (t = -1)) :=
by
  sorry

end problem_statement_l118_118072


namespace train_cross_poles_time_l118_118856

-- Defining the given conditions
def train_length : ℝ := 300.0              -- Length of the train in meters
def train_speed_kmh : ℝ := 200.0            -- Speed of the train in km/h
def distances_between_poles : List ℝ := [150.0, 250.0, 400.0, 500.0, 700.0]  -- Distances between poles in meters

-- Other necessary computations
def total_distance : ℝ := distances_between_poles.sum + train_length           -- Total distance the train will cover
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)                      -- Speed of the train in m/s

-- The theorem to be proven
theorem train_cross_poles_time:
  (total_distance / train_speed_ms) ≈ 46.78 :=
sorry

end train_cross_poles_time_l118_118856


namespace edge_coloring_of_convex_polyhedron_l118_118457

theorem edge_coloring_of_convex_polyhedron 
  (P : Type) [convex_polyhedron P]
  (each_vertex_belongs_to_three_faces : ∀ v : vertex P, ∃ (S : finset (face P)), S.card = 3 ∧ ∀ f ∈ S, v ∈ f)
  (each_vertex_dyed_black_white : ∀ (f p : vertex P), edge P f p → (∃ b : bool, f.color = b) ∧ (∃ b : bool, p.color ≠ b)) :
  (∃ (c : edge P → color), 
    (∀ v : vertex P, ∀ e₁ e₂ e₃ : edge P, e₁ ∈ edges v → e₂ ∈ edges v → e₃ ∈ edges v → 
      (e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₂ ≠ e₃) → 
      (c e₁ ≠ c e₂ ∧ c e₁ ≠ c e₃ ∧ c e₂ ≠ c e₃)) ∧
    (∀ f : face P, ∃ c₁ c₂ : color, c₁ ≠ c₂ ∧ (∃ (e₁ e₂ : edge P), e₁ ∈ edges f ∧ e₂ ∈ edges f ∧ c e₁ = c₁ ∧ c e₂ = c₂)))
 :=
sorry

end edge_coloring_of_convex_polyhedron_l118_118457


namespace graph_symmetric_center_l118_118041

-- Define the function f and the condition given in the problem
variable (f : ℝ → ℝ)
variable (H : ∀ x, f(x+2) = -f(2 - x))

-- The theorem statement which needs to be proved
-- The point (2, 0) is the center of symmetry for the graph y = f(x)
theorem graph_symmetric_center : 
  (∀ x, f(x) + f(4 - x) = 0) → 
  ∃ a b, (∀ x, f(x) + f(2 * a - x) = 2 * b) := 
by
  sorry

end graph_symmetric_center_l118_118041


namespace distance_from_P_to_AD_l118_118925

theorem distance_from_P_to_AD :
  let A := (0, 5)
  let D := (0, 0) 
  let B := (5, 5)
  let C := (5, 0)
  let M := (2.5, 0)
  let radius_M := 2.5
  let radius_A := 5
  let EqCircleM : (ℝ × ℝ) → ℝ := λ p, (p.1 - 2.5) ^ 2 + p.2 ^ 2
  let EqCircleA : (ℝ × ℝ) → ℝ := λ p, p.1 ^ 2 + (p.2 - 5) ^ 2
  ∃ P : ℝ × ℝ,
    EqCircleM P = radius_M ^ 2 ∧
    EqCircleA P = radius_A ^ 2 ∧
    P.2 = 4 + 2 * Real.sqrt 5 ∧
    P.1 = 0.5 + 4 * Real.sqrt 5 ∧
    ∀ q : ℝ × ℝ, q = P → q.1 = 0.5 + 4 * Real.sqrt 5 :=
by
  sorry

end distance_from_P_to_AD_l118_118925


namespace problem_statement_l118_118321

noncomputable def expression : ℝ :=
  ((-5 / 6) ^ 2022) * ((6 / 5) ^ 2023) + 
  (-5) ^ (-1) - 
  (π - 3.14) ^ 0 + 
  (1 / 3) ^ (-2)

theorem problem_statement : expression = 9 :=
by
  sorry

end problem_statement_l118_118321


namespace factorial_log_sum_l118_118806

noncomputable def factorial (n : ℕ) : ℝ :=
  if n = 0 then 1 else ∏ i in finset.range (n + 1), (if i = 0 then 1 else i)

theorem factorial_log_sum (n : ℕ) (h1 : n > 0) :
  (∑ k in finset.range (n - 1) + 2, 1 / real.log (factorial n) / real.log (k + 2)) = 1 := by
sorry

end factorial_log_sum_l118_118806


namespace sum_mod_1000_l118_118112

-- Define the conditions for n
def is_valid_n (n : ℕ) : Prop :=
  ∃ m : ℕ, n^2 + 12 * n - 1981 = m^2

-- The main theorem stating what needs to be proven
theorem sum_mod_1000 :
  let T := ∑ n in Finset.filter is_valid_n (Finset.range 2000), n
  T % 1000 = 3 :=
sorry

end sum_mod_1000_l118_118112


namespace count_kelvin_liked_5_digit_numbers_l118_118468

-- Define the condition that a digit list must satisfy
def strictly_decreasing (l : List ℕ) : Prop :=
  ∀ (i : ℕ), i < l.length - 1 → l.get ⟨i, sorry⟩ > l.get ⟨i + 1, sorry⟩

def at_most_one_violation (l : List ℕ) : Prop :=
  ∃ (k : ℕ), k < l.length - 1 ∧ ∀ (i : ℕ), (i < k ∨ i > k + 1) 
    → l.get ⟨i, sorry⟩ > l.get ⟨i + 1, sorry⟩

-- Define a predicate that a number's digit list is liked by Kelvin
def liked_by_kelvin (l : List ℕ) : Prop :=
  l.length = 5 ∧ (strictly_decreasing l ∨ at_most_one_violation l)

-- Define the list of digits from 0 to 9
def digit_list : List ℕ := List.range 10

-- Define the theorem
theorem count_kelvin_liked_5_digit_numbers : 
  (Finset.filter liked_by_kelvin (Finset.powersetLen 5 (Finset.univ.filter_map (λ x, digit_list.nth x)))).card = 6678 :=
by sorry

end count_kelvin_liked_5_digit_numbers_l118_118468


namespace cord_lengths_l118_118189

noncomputable def cordLengthFirstDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthSecondDog (distance : ℝ) : ℝ :=
  distance / 2

noncomputable def cordLengthThirdDog (radius : ℝ) : ℝ :=
  radius

theorem cord_lengths (d1 d2 r : ℝ) (h1 : d1 = 30) (h2 : d2 = 40) (h3 : r = 20) :
  cordLengthFirstDog d1 = 15 ∧ cordLengthSecondDog d2 = 20 ∧ cordLengthThirdDog r = 20 := by
  sorry

end cord_lengths_l118_118189


namespace correct_option_is_C_condition3_l118_118651

theorem correct_option_is_C_condition3 :
  ∀ (A B : Set ℝ), (∃ (x : ℝ), x + 1/x < 2) → ¬(∀ (T1 T2 : Triangle), congruent T1 T2 → area T1 = area T2) :=
by
  sorry

end correct_option_is_C_condition3_l118_118651


namespace sum_of_coefficients_l118_118821

theorem sum_of_coefficients :
  ∃ a b c : ℕ, (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (sqrt 3 + 1 / sqrt 3 + sqrt 11 + 1 / sqrt 11 = (a * sqrt 3 + b * sqrt 11) / c) ∧
  (∀ d : ℕ, d ≠ 0 → (sqrt 3 + 1 / sqrt 3 + sqrt 11 + 1 / sqrt 11 = (a * sqrt 3 + b * sqrt 11) / d → c ≤ d)) ∧
  (a + b + c = 113) :=
sorry

end sum_of_coefficients_l118_118821


namespace work_completion_days_l118_118590

theorem work_completion_days (a b : ℕ) (h1 : a + b = 6) (h2 : a + b = 15 / 4) : a = 6 :=
by
  sorry

end work_completion_days_l118_118590


namespace at_most_one_solution_l118_118894

theorem at_most_one_solution (a b c : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (hcpos : 0 < c) :
  ∃! x : ℝ, a * x + b * ⌊x⌋ - c = 0 :=
sorry

end at_most_one_solution_l118_118894


namespace meter_to_leap_l118_118150

theorem meter_to_leap
  (strides leaps bounds meters : ℝ)
  (h1 : 3 * strides = 4 * leaps)
  (h2 : 5 * bounds = 7 * strides)
  (h3 : 2 * bounds = 9 * meters) :
  1 * meters = (56 / 135) * leaps :=
by
  sorry

end meter_to_leap_l118_118150


namespace no_solution_not_externally_tangent_l118_118050

theorem no_solution_not_externally_tangent {C₁ C₂ : Type} (h : ∀ (x y : ℝ), ¬ ((x, y) ∈ C₁ ∧ (x, y) ∈ C₂)) :
  ¬ (∃ (x y : ℝ), (x, y) ∈ (C₁ ∩ C₂)) :=
by
  sorry

end no_solution_not_externally_tangent_l118_118050


namespace badminton_team_ways_l118_118494

def num_combinations (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem badminton_team_ways : num_combinations 6 2 * num_combinations 4 2 / 2 = 45 := by
  sorry

end badminton_team_ways_l118_118494


namespace mean_home_runs_l118_118156

theorem mean_home_runs :
  let hits_by_players := [2, 3, 2, 1, 1]
  let home_runs := [5, 6, 8, 10, 12]
  let total_players := List.sum hits_by_players
  let total_home_runs := List.sum (List.zipWith (· * ·) hits_by_players home_runs)
  (total_players = 9) → (total_home_runs = 66) → total_home_runs / total_players = 66 / 9 :=
by
  intros h_total_players h_total_home_runs
  rw [h_total_players, h_total_home_runs]
  norm_num
  sorry

end mean_home_runs_l118_118156


namespace original_budget_calculation_l118_118214

noncomputable def original_budget (X : ℝ) : ℝ :=
  let clothes := 0.9 * 0.25 * X
  let groceries := 1.05 * 0.15 * X
  let electronics := 0.10 * X * 1.2
  let dining := 1.12 * 0.05 * X
  X - (clothes + groceries + electronics + dining)

theorem original_budget_calculation (X : ℝ) :
  original_budget X = 13500 → X ≈ 30579.16 :=
by
  intro h
  -- Introduce an approximation tolerance
  have tolerance : ℝ := 0.01
  -- Compute the approximate value
  let approx_X := 13500 / 0.4415
  show X ≈ 30579.16
  sorry

end original_budget_calculation_l118_118214


namespace unique_zero_value_of_a_l118_118025

def f (x a : ℝ) : ℝ := x^2 - 2 * x + a * (exp(x - 1) + exp(-x + 1))

theorem unique_zero_value_of_a (a : ℝ) :
  (∃! x, f x a = 0) → a = 1 / 2 := by
  sorry

end unique_zero_value_of_a_l118_118025


namespace digit_in_725th_place_l118_118225

theorem digit_in_725th_place : 
  let seq := "24137931034482758620689655172" in
  let repeat_len := String.length seq in
  (725 % repeat_len = 21) →
  seq.get 20 = '6' :=
by
  sorry

end digit_in_725th_place_l118_118225


namespace min_positive_sum_of_products_l118_118345

-- Definitions based on the problem conditions
def a : Fin 50 → ℤ := sorry  -- Let a_i be integers (either 1 or -1) defined over the finite set {1, 2, ..., 50}

-- Sum over pairs (i, j) such that 1 ≤ i < j ≤ 50
def sum_of_products (a : Fin 50 → ℤ) : ℤ :=
  ∑ i in Finset.range 50, ∑ j in Finset.Ico (i+1) 50, a i * a j

-- The theorem we need to prove
theorem min_positive_sum_of_products : 
  (∀ i, a i = 1 ∨ a i = -1) → 0 < sum_of_products a → sum_of_products a = 7 :=
sorry

end min_positive_sum_of_products_l118_118345


namespace inradius_of_triangle_l118_118724

theorem inradius_of_triangle (a b c : ℕ) (h_rt : a^2 + b^2 = c^2) (h_a : a = 13) (h_b : b = 84) (h_c : c = 85) : 
  let s := (a + b + c) / 2 in
  let A := (a * b) / 2 in
  let r := A / s in
  r = 6 :=
by
  sorry

end inradius_of_triangle_l118_118724


namespace max_sum_of_triplet_product_60_l118_118956

theorem max_sum_of_triplet_product_60 : 
  ∃ a b c : ℕ, a * b * c = 60 ∧ a + b + c = 62 :=
sorry

end max_sum_of_triplet_product_60_l118_118956


namespace find_x_l118_118222

theorem find_x (x : ℝ) (h : 70 + 60 / (x / 3) = 71) : x = 180 :=
sorry

end find_x_l118_118222


namespace count_sets_B_l118_118484

open Set

def A : Set ℕ := {1, 2}

theorem count_sets_B (B : Set ℕ) (h1 : A ∪ B = {1, 2, 3}) : 
  (∃ Bs : Finset (Set ℕ), ∀ b ∈ Bs, A ∪ b = {1, 2, 3} ∧ Bs.card = 4) := sorry

end count_sets_B_l118_118484


namespace gumball_ratio_l118_118260

-- Define the conditions
variables (B G R : ℕ)
variables (h_total_gumballs : R + G + B = 56)
variables (h_red_gumballs : R = 16)
variables (h_green_to_blue_ratio : G = 4 * B)

-- Statement of the theorem
theorem gumball_ratio (h_total_gumballs : R + G + B = 56)
                      (h_red_gumballs : R = 16)
                      (h_green_to_blue_ratio : G = 4 * B) :
                      B : R = 1 : 2 :=
by
  sorry

end gumball_ratio_l118_118260


namespace molecular_weight_AlI3_correct_percentage_Al_correct_percentage_I_correct_l118_118675

/-- Data for atomic weights -/
def atomic_weight_Al : Float := 26.98
def atomic_weight_I : Float := 126.90

/-- Data for the elements in the compound -/
def count_Al : Nat := 1
def count_I : Nat := 3

/-- Formula for molecular weight calculation -/
def molecular_weight_AlI3 : Float := 
  (count_Al * atomic_weight_Al) + (count_I * atomic_weight_I)

/-- Proof statement for the molecular weight -/
theorem molecular_weight_AlI3_correct : molecular_weight_AlI3 = 407.68 := by
  sorry

/-- Formula for percentage composition -/
def percentage_Al : Float := (atomic_weight_Al / molecular_weight_AlI3) * 100
def percentage_I : Float := ((count_I * atomic_weight_I) / molecular_weight_AlI3) * 100

/-- Proof statements for percentage composition -/
theorem percentage_Al_correct : percentage_Al = 6.62 := by
  sorry

theorem percentage_I_correct : percentage_I = 93.4 := by
  sorry

end molecular_weight_AlI3_correct_percentage_Al_correct_percentage_I_correct_l118_118675


namespace remainder_sum_div_6_l118_118426

theorem remainder_sum_div_6 (n : ℤ) : ((5 - n) + (n + 4)) % 6 = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end remainder_sum_div_6_l118_118426


namespace milk_level_lowered_l118_118527

variable (length width : ℝ)
variable (volume_gallons : ℝ)
variable (conversion_factor : ℝ)
variable (height_lowered_ft height_lowered_in : ℝ)

-- Given conditions
def given_conditions : Prop :=
  length = 50 ∧
  width = 25 ∧
  volume_gallons = 4687.5 ∧
  conversion_factor = 7.5

-- Theorem to prove how many inches the milk level should be lowered
theorem milk_level_lowered (h : given_conditions) :
  height_lowered_in = 6 :=
by
  sorry

end milk_level_lowered_l118_118527


namespace max_xy_l118_118242

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 3 * y = 6) : xy ≤ 3 / 2 := sorry

end max_xy_l118_118242


namespace isogonal_conjugates_OH_l118_118329

section isogonal_conjugates

variables {A B C : Type} -- Type declaration for points
variables [EuclideanGeometry A B C] -- Assuming Euclidean geometry context
variables (triangleABC : Triangle A B C) -- instance of triangle

-- Definitions of circumcenter and orthocenter for a triangle
def circumcenter (Δ : Triangle A B C) : A := Δ.circumcenter
def orthocenter (Δ : Triangle A B C) : A := Δ.orthocenter

-- Isogonal conjugation predicate
def isogonal_conjugates (P Q : A) (Δ : Triangle A B C) := 
  isogonal_conjugate P Q Δ

-- Theorem statement
theorem isogonal_conjugates_OH :
  isogonal_conjugates (circumcenter triangleABC) (orthocenter triangleABC)
                      triangleABC :=
sorry 

end isogonal_conjugates

end isogonal_conjugates_OH_l118_118329


namespace exist_distinct_real_numbers_l118_118107

noncomputable def f : ℝ → ℝ := sorry

theorem exist_distinct_real_numbers (f : ℝ → ℝ)
  (h : ∀ x, f (f x) = Real.floor x) :
  ∃ a b : ℝ, a ≠ b ∧ |f a - f b| ≥ |a - b| := 
sorry

end exist_distinct_real_numbers_l118_118107


namespace triangle_perimeter_l118_118690

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem triangle_perimeter {A B O : ℝ × ℝ}
  (hA : A = (3, 7))
  (hB : B = (5, 2))
  (hO : O = (0, 0)) :
  let d : ℝ := distance A.1 A.2 B.1 B.2,
      d1 : ℝ := distance O.1 O.2 A.1 A.2,
      d2 : ℝ := distance O.1 O.2 B.1 B.2,
      P  : ℝ := d + d1 + d2
  in P = 2 * real.sqrt 29 + real.sqrt 58 :=
by
  sorry

end triangle_perimeter_l118_118690


namespace digit_in_725th_place_l118_118224

theorem digit_in_725th_place : 
  let seq := "24137931034482758620689655172" in
  let repeat_len := String.length seq in
  (725 % repeat_len = 21) →
  seq.get 20 = '6' :=
by
  sorry

end digit_in_725th_place_l118_118224


namespace second_train_speed_l118_118197

def speed_of_second_train (length1 length2 speed1 relative_distance: ℝ) (clearing_time: ℝ) : ℝ :=
  let relative_speed := (relative_distance / 1000) / (clearing_time / 3600)
  relative_speed - speed1

theorem second_train_speed :
  speed_of_second_train 120 280 42 400 20 = 30 :=
by
  sorry

end second_train_speed_l118_118197


namespace scientific_notation_of_twenty_million_l118_118931

-- Define the number 20 million
def twenty_million : ℂ :=
  20000000

-- Define the scientific notation to be proved correct
def scientific_notation : ℂ :=
  2 * 10 ^ 7

-- The theorem to prove the equivalence
theorem scientific_notation_of_twenty_million : twenty_million = scientific_notation :=
  sorry

end scientific_notation_of_twenty_million_l118_118931


namespace cost_of_tshirt_before_pictures_l118_118509

-- Given definitions of parameters
def initial_investment : ℝ := 1500
def sale_price_per_tshirt : ℝ := 20
def break_even_tshirts : ℝ := 83
def total_revenue := break_even_tshirts * sale_price_per_tshirt
def total_cost_of_tshirts (C : ℝ) := break_even_tshirts * C

-- Theorem to prove the cost per T-shirt before adding the pictures
theorem cost_of_tshirt_before_pictures :
  ∃ (C : ℝ), total_cost_of_tshirts C + initial_investment = total_revenue :=
by
  -- Let C be the cost of each T-shirt
  use 160 / 83
  -- Substitute the calculated values
  simp [total_revenue, total_cost_of_tshirts, initial_investment]
  norm_num
  finish

end cost_of_tshirt_before_pictures_l118_118509


namespace probability_nine_heads_l118_118986

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118986


namespace colored_triangle_has_diverse_smaller_triangle_l118_118314

def colored_vertex (V : Type) := V → Prop

noncomputable def large_triangle_subdivision_with_colored_vertices 
  (V : Type) (A B C : V) (triangle : set V) (triangles : set (set V)) (coloring : V → Prop) : Prop :=
  (∀ T ∈ triangles, ∃ (v₁ v₂ v₃ : V),
    v₁ ∈ T ∧ v₂ ∈ T ∧ v₃ ∈ T ∧
    coloring v₁ ≠ coloring v₂ ∧ coloring v₂ ≠ coloring v₃ ∧ coloring v₃ ≠ coloring v₁)

def exists_smaller_triangle_with_different_colors 
  {V : Type} 
  (A B C : V) 
  (triangles : set (set V)) 
  (coloring : V → Prop) : Prop := 
  ∃ T ∈ triangles, 
    ∃ (v₁ v₂ v₃ : V),
      v₁ ∈ T ∧ v₂ ∈ T ∧ v₃ ∈ T ∧ 
      coloring v₁ ≠ coloring v₂ ∧ coloring v₂ ≠ coloring v₃ ∧ coloring v₃ ≠ coloring v₁

theorem colored_triangle_has_diverse_smaller_triangle 
  {V : Type} 
  (A B C : V) 
  (triangles : set (set V)) 
  (coloring : V → Prop) 
  (h : large_triangle_subdivision_with_colored_vertices V A B C {A, B, C} triangles coloring) :
  exists_smaller_triangle_with_different_colors A B C triangles coloring :=
sorry

end colored_triangle_has_diverse_smaller_triangle_l118_118314


namespace find_x_parallel_vectors_l118_118032

theorem find_x_parallel_vectors :
  ∀ x : ℝ, (∃ k : ℝ, (1, 2) = (k * (2 * x), k * (-3))) → x = -3 / 4 :=
by
  sorry

end find_x_parallel_vectors_l118_118032


namespace tim_same_age_tina_l118_118194

-- Define the ages of Tim and Tina
variables (x y : ℤ)

-- Given conditions
def condition_tim := x + 2 = 2 * (x - 2)
def condition_tina := y + 3 = 3 * (y - 3)

-- The goal is to prove that Tim is the same age as Tina
theorem tim_same_age_tina (htim : condition_tim x) (htina : condition_tina y) : x = y :=
by 
  sorry

end tim_same_age_tina_l118_118194


namespace students_play_neither_l118_118831

-- Define the conditions
def total_students : ℕ := 39
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def both_players : ℕ := 17

-- Define a theorem that states the equivalent proof problem
theorem students_play_neither : 
  total_students - (football_players + long_tennis_players - both_players) = 10 := by
  sorry

end students_play_neither_l118_118831


namespace count_3_digit_numbers_multiple_30_not_75_l118_118787

theorem count_3_digit_numbers_multiple_30_not_75 : 
  (finset.filter (λ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 30 = 0 ∧ n % 75 ≠ 0) (finset.range 1000)).card = 21 := sorry

end count_3_digit_numbers_multiple_30_not_75_l118_118787


namespace horizontal_distance_P_Q_l118_118605

-- Definitions for the given conditions
def curve (x : ℝ) : ℝ := x^2 + 2 * x - 3

-- Define the points P and Q on the curve
def P_x : Set ℝ := {x | curve x = 8}
def Q_x : Set ℝ := {x | curve x = -1}

-- State the theorem to prove horizontal distance is 3sqrt3
theorem horizontal_distance_P_Q : 
  ∃ (x₁ x₂ : ℝ), x₁ ∈ P_x ∧ x₂ ∈ Q_x ∧ |x₁ - x₂| = 3 * Real.sqrt 3 :=
sorry

end horizontal_distance_P_Q_l118_118605


namespace complex_number_count_l118_118420

noncomputable def number_of_complex_numbers_satisfying_conditions : ℤ :=
  10

theorem complex_number_count (z : ℂ) (h : abs z < 30) : 
  (exp z = (z - 1) / (z + 1)) → (number_of_complex_numbers_satisfying_conditions = 10) := by
  sorry

end complex_number_count_l118_118420


namespace student_correct_answers_l118_118841

theorem student_correct_answers (C W : ℕ) 
  (h1 : 4 * C - W = 130) 
  (h2 : C + W = 80) : 
  C = 42 := by
  sorry

end student_correct_answers_l118_118841


namespace area_triangle_MOI_l118_118053

-- Definitions of the vertices of triangle ABC (assuming it lies on a 2D plane)
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 0)
def C : ℝ × ℝ := (0, 15)

-- Definition of the circumcenter O
def O : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Definition of the incenter I
def I : ℝ × ℝ := 
  ((15 * B.1 + 8 * (B.1 + C.1) + 17 * C.1) / (15 + 8 + 17), 
   (15 * B.2 + 8 * (B.2 + C.2) + 17 * C.2) / (15 + 8 + 17))

-- The circle centered at M is assumed to be tangent to lines AC, BC, and the circumcircle.
-- Defining the center M based on geometric assumptions and constraints.
-- Note: Simplified assumption used in the problem; the precise determination of M may differ.
def M : ℝ × ℝ := (4, 4)

-- Proving that the area of triangle MOI is equal to 3
theorem area_triangle_MOI : 
  let area := 1 / 2 * abs (M.1 * (O.2 - I.2) + O.1 * (I.2 - M.2) + I.1 * (M.2 - O.2)) in
  area = 3 :=
by
  sorry

end area_triangle_MOI_l118_118053


namespace product_of_g_values_roots_of_f_eq_neg59_l118_118480

noncomputable def polynomial_f : Polynomial ℂ := Polynomial.Coeff 5 1 + Polynomial.Coeff 3 (-1) + Polynomial.Coeff 2 2 + Polynomial.Coeff 0 1
noncomputable def polynomial_g (x : ℂ) : ℂ := x^2 - 3

theorem product_of_g_values_roots_of_f_eq_neg59 (x1 x2 x3 x4 x5 : ℂ) (hx : polynomial_f.eval x1 = 0) (hy : polynomial_f.eval x2 = 0) (hz : polynomial_f.eval x3 = 0) (hw : polynomial_f.eval x4 = 0) (hv : polynomial_f.eval x5 = 0) :
  polynomial_g x1 * polynomial_g x2 * polynomial_g x3 * polynomial_g x4 * polynomial_g x5 = -59 :=
by
  sorry

end product_of_g_values_roots_of_f_eq_neg59_l118_118480


namespace seq_limit_eq_one_l118_118009

open Filter Real

/-- Define the sequence recursively. -/
def seq : ℕ → ℝ
| 0       := 2
| (n + 1) := (2 * seq n - 1) / (seq n)

/-- Prove that the sequence converges to 1. -/
theorem seq_limit_eq_one : tendsto seq atTop (𝓝 1) := 
sorry

end seq_limit_eq_one_l118_118009


namespace math_problem_l118_118286

-- Proposition 1
def prop1_statement (a : ℝ) : Prop := 
  (∃ (x y : ℝ), a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0 ∧ 
    ∃ r : ℝ, r > 0 ∧ a^2x^2 + (a+2)y^2 + 2ax + a = r^2) ↔ a = -1

-- Proposition 2 is found incorrect
def prop2_statement : Prop := False

-- Proposition 3
structure Cube :=
  (A B C D A₁ B₁ C₁ D₁ : Type*)
  (E F : Type*)
  (is_midpoint_E : ∃ M N : Type*, M ≠ N ∧ E = (M + N) / 2)
  (is_midpoint_F : ∃ Q P : Type*, Q ≠ P ∧ F = (Q + P) / 2)

def prop3_statement (cube : Cube): Prop := 
  ∃ (P : Type*), 
    (∃ (line_CE : P), P ∈ line_CE) ∧
    (∃ (line_D₁F: P), P ∈ line_D₁F) ∧
    (∀ (DA : P), P ∈ DA)

-- Proposition 4
def prop4_statement (f : ℝ → ℝ) [is_power_function : ∃ c > 0, ∃ n ∈ ℤ, f x = c * x^n] : Prop := 
  ∀ x, x < 0 → f x ≥ 0

theorem math_problem (a : ℝ) (cube : Cube) (f : ℝ → ℝ) [is_power_function : ∃ c > 0, ∃ n ∈ ℤ, f x = c * x^n] :
  prop1_statement a ∧ prop3_statement cube ∧ prop4_statement f := by sorry

end math_problem_l118_118286


namespace floor_function_properties_l118_118154

noncomputable def floor_function (x : ℝ) : ℤ := Int.floor x

theorem floor_function_properties (x : ℝ) (h₁ : x ∈ Set.Icc (-2 : ℝ) 3) :
  ¬ (∀ x : ℝ, floor_function (-x) = floor_function x) ∧
  ¬ (∀ x : ℝ, floor_function (-x) = -floor_function x) ∧ 
  Set.range (λ x : ℝ, floor_function x) = {-2, -1, 0, 1, 2, 3} := by
  sorry

end floor_function_properties_l118_118154


namespace no_roots_f_f_x_l118_118899

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ :=
  x^2 + b*x + c

theorem no_roots_f_f_x (b c : ℝ) (h : (b - 1)^2 - 4 * c < 0) :
  ∀ x : ℝ, f b c (f b c x) ≠ x :=
begin
  assume x,
  sorry
end

end no_roots_f_f_x_l118_118899


namespace passed_in_both_subjects_l118_118445

theorem passed_in_both_subjects (A B C : ℝ)
  (hA : A = 0.25)
  (hB : B = 0.48)
  (hC : C = 0.27) :
  1 - (A + B - C) = 0.54 := by
  sorry

end passed_in_both_subjects_l118_118445


namespace points_coincide_l118_118195

open EuclideanGeometry

theorem points_coincide 
  (A B C : Point) 
  (h : Triangle ABC)
  (circ : circumcircle h)
  (A1 B1 C1 : Point)
  (A1_diam : diametrically_opposite circ A A1)
  (B1_diam : diametrically_opposite circ B B1)
  (C1_diam : diametrically_opposite circ C C1)
  (A0 B0 C0 : Point)
  (A0_mid : midpoint A0 B C)
  (B0_mid : midpoint B0 A C)
  (C0_mid : midpoint C0 A B)
  (A2 B2 C2 : Point)
  (A2_symm : symmetric A1 A0 A2)
  (B2_symm : symmetric B1 B0 B2)
  (C2_symm : symmetric C1 C0 C2) 
: A2 = B2 ∧ B2 = C2 :=
sorry

end points_coincide_l118_118195


namespace octagon_mass_l118_118714

theorem octagon_mass :
  let side_length := 1 -- side length of the original square (meters)
  let thickness := 0.3 -- thickness of the sheet (cm)
  let density := 7.8 -- density of steel (g/cm^3)
  let x := 50 * (2 - Real.sqrt 2) -- side length of the triangles (cm)
  let octagon_area := 20000 * (Real.sqrt 2 - 1) -- area of the octagon (cm^2)
  let volume := octagon_area * thickness -- volume of the octagon (cm^3)
  let mass := volume * density / 1000 -- mass of the octagon (kg), converted from g to kg
  mass = 19 :=
by
  sorry

end octagon_mass_l118_118714


namespace polygon_diagonals_30_l118_118659

-- Define the properties and conditions of the problem
def sides := 30

-- Define the number of diagonals calculation function
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement to check the number of diagonals in a 30-sided convex polygon
theorem polygon_diagonals_30 : num_diagonals sides = 375 := by
  sorry

end polygon_diagonals_30_l118_118659


namespace product_probability_multiple_of_45_l118_118594

-- Conditions
def single_digit_multiples_of_3 : Set ℕ := {3, 6, 9}
def primes_less_than_20 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Correct answer
def probability_of_multiple_of_45 (multiples : Set ℕ) (primes : Set ℕ) : ℚ :=
  let total_outcomes := multiplicity p ∣ multiplicity n in
  let favorable_outcomes := multiplicity 45 p ∣ multiplicity (9, 5) in
  favorable_outcomes / total_outcomes

theorem product_probability_multiple_of_45 :
  probability_of_multiple_of_45 single_digit_multiples_of_3 primes_less_than_20 = 1 / 24 := 
sorry

end product_probability_multiple_of_45_l118_118594


namespace a_b_cannot_interchange_m_n_value_l118_118670

-- Definition of conditions for Problem 1
def f (x : ℤ) (b c : ℤ) : ℤ := x^2 + b * x + c
def a := sin
def θ_range (θ : ℝ) : Prop := θ ∈ set.Icc (-real.pi / 2) (real.pi / 2)

-- Question 1: Show a and b cannot be interchanged
theorem a_b_cannot_interchange {b c : ℤ} (θ : ℝ) (hθ : θ_range θ) : (a θ) ≠ b := sorry

-- Definition of functions for Problem 2
def f_ratio (x k : ℝ) : ℝ := (x^2 + k * x + 1) / (x^2 + x + 1)
def g (x : ℝ) : ℝ := 2^x - 3 / 2
def g_range : set.Icc (-1 / 2) 4 

-- Question 2: Prove m + n = log2(11 / 2)
theorem m_n_value {k m n : ℝ} (hk : g_range) (hg_k : ∀ x ∈ set.Icc k k, k = g x) : m + n = real.log (11 / 2) / real.log 2 :=  sorry

end a_b_cannot_interchange_m_n_value_l118_118670


namespace f_strictly_increasing_intervals_value_of_a_l118_118768

noncomputable def vector_m (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def vector_n (x : ℝ) : ℝ × ℝ := (6 * Real.sin x + Real.cos x, 7 * Real.sin x - 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (vector_m x).fst * (vector_n x).fst + (vector_m x).snd * (vector_n x).snd

-- Prove 1
theorem f_strictly_increasing_intervals :
  ∀ (k : ℤ), ∀ (x : ℝ), k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 -> f x = 
    4 * Real.sqrt 2 * Real.sin (2 * x - Real.pi / 4) + 2 by
  sorry

-- Define and prove additional parts for problem 2
variables (A b c : ℝ)
variables (side_eq : b + c = 2 + 3 * Real.sqrt 2) (fA_eq : f A = 6) (area_eq : 3 = b * c * Real.sqrt 2 / 4)

-- Prove 2
theorem value_of_a :
  let a := Real.sqrt (10) in
  area_eq ∧ side_eq ∧ fA_eq -> a = Real.sqrt (10) by
  sorry

end f_strictly_increasing_intervals_value_of_a_l118_118768


namespace find_y_l118_118766

theorem find_y 
  (a : (ℝ × ℝ))
  (b : (ℝ × ℝ))
  (h₁ : a = (1, 1))
  (h₂ : ∃ y : ℝ, b = (2, y) ∧ (real.sqrt ((a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2) = a.1 * b.1 + a.2 * b.2)) :
  ∃ y, b = (2, y) ∧ y = 3 := sorry

end find_y_l118_118766


namespace intersection_lies_on_arc_l118_118728

variable (A B C D E : Point)
variable (a : Length)
variable [h_eq_tri : EquilateralTriangle A B C (side_length := a*1)] -- Equilateral Triangle
variable [extension_BD : ∃ x : Length, ∃ BD : Segment, extension_and_length BD A B D x] -- D is on extension of AB beyond B
variable [extension_CE : ∃ y : Length, ∃ CE : Segment, extension_and_length CE A C E y] -- E is on extension of AC beyond C
variable [product : x * y = a^2]

theorem intersection_lies_on_arc : ∃ P : Point, (∃ P_D_C : Segment DC D P C, ∃ P_B_E : Segment BE B P E /\ P ≠ B /\ P ≠ C) → 
  P ∈ arc BC (circumcircle_of_equilateral_triangle ABC) := 
sorry

end intersection_lies_on_arc_l118_118728


namespace general_term_sum_first_n_terms_l118_118725

variable {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}
variable (d : ℝ) (h1 : d ≠ 0)
variable (a10 : a 10 = 19)
variable (geo_seq : ∀ {x y z}, x * z = y ^ 2 → x = 1 → y = a 2 → z = a 5)
variable (arith_seq : ∀ n, a n = a 1 + (n - 1) * d)

-- General term of the arithmetic sequence
theorem general_term (a_1 : ℝ) (h1 : a 1 = a_1) : a n = 2 * n - 1 :=
sorry

-- Sum of the first n terms of the sequence b_n
theorem sum_first_n_terms (n : ℕ) : S n = (2 * n - 3) * 2^(n + 1) + 6 :=
sorry

end general_term_sum_first_n_terms_l118_118725


namespace Kolya_correct_Valya_incorrect_l118_118298

-- Kolya's Claim (Part a)
theorem Kolya_correct (x : ℝ) (p r : ℝ) (hpr : r = 1/(x+1) ∧ p = 1/x) : 
  (p / (1 - (1 - r) * (1 - p))) = (r / (1 - (1 - r) * (1 - p))) :=
sorry

-- Valya's Claim (Part b)
theorem Valya_incorrect (x : ℝ) (p r : ℝ) (q s : ℝ) (hprs : r = 1/(x+1) ∧ p = 1/x ∧ q = 1 - p ∧ s = 1 - r) : 
  ((q * r / (1 - s * q)) + (p * r / (1 - s * q))) = 1/2 :=
sorry

end Kolya_correct_Valya_incorrect_l118_118298


namespace intersection_M_N_l118_118412

def M : set ℤ := {x | x * (x - 1) ≤ 0}
def N : set ℤ := {x | ∃ n : ℕ, x = 2 * n}

theorem intersection_M_N : M ∩ N = {0} :=
by
  sorry

end intersection_M_N_l118_118412


namespace leftover_coverage_l118_118678

theorem leftover_coverage 
  (coverage_per_bag : ℕ)
  (length : ℕ)
  (width : ℕ)
  (num_bags : ℕ) :
  coverage_per_bag = 250 →
  length = 22 →
  width = 36 →
  num_bags = 4 →
  let lawn_area := length * width,
      total_coverage := coverage_per_bag * num_bags,
      leftover_coverage := total_coverage - lawn_area
  in leftover_coverage = 208 := 
by
  intros h1 h2 h3 h4
  let lawn_area := 22 * 36
  let total_coverage := 250 * 4
  let leftover_coverage := total_coverage - lawn_area
  have : lawn_area = 792 := by norm_num
  have : total_coverage = 1000 := by norm_num
  have : leftover_coverage = total_coverage - lawn_area := rfl
  show leftover_coverage = 208, from by
    rw [this, this, this]
    norm_num
  sorry

end leftover_coverage_l118_118678


namespace maximum_composite_pairwise_coprime_set_cardinality_l118_118496

/-- A number is composite if it is not prime and greater than 1 -/
def is_composite (n : ℕ) : Prop :=
  ¬ (Nat.Prime n) ∧ n > 1

/-- Two numbers are coprime if their gcd is 1 -/
def coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- A list of numbers is pairwise coprime if every pair of distinct elements from the list is coprime -/
def pairwise_coprime (l : List ℕ) : Prop :=
  ∀ (x ∈ l) (y ∈ l), x ≠ y → coprime x y 

/-- A list of numbers are all composite and pairwise coprime -/
def valid_set (l : List ℕ) : Prop :=
  (∀ x ∈ l, 10 ≤ x ∧ x ≤ 99 ∧ is_composite x) ∧ pairwise_coprime l

/-- Theorem: The maximum number of two-digit composite pairwise coprime numbers is 4 -/
theorem maximum_composite_pairwise_coprime_set_cardinality : 
  ∃ l : List ℕ, valid_set l ∧ l.length = 4 :=
sorry

end maximum_composite_pairwise_coprime_set_cardinality_l118_118496


namespace snake_body_length_l118_118637

theorem snake_body_length (l h : ℝ) (h_head: h = l / 10) (h_length: l = 10) : l - h = 9 := 
by 
  rw [h_length, h_head] 
  norm_num
  sorry

end snake_body_length_l118_118637


namespace small_cubes_with_two_faces_painted_l118_118315

-- Statement of the problem
theorem small_cubes_with_two_faces_painted
  (remaining_cubes : ℕ)
  (edges_with_two_painted_faces : ℕ)
  (number_of_edges : ℕ) :
  remaining_cubes = 60 → edges_with_two_painted_faces = 2 → number_of_edges = 12 →
  (remaining_cubes - (4 * (edges_with_two_painted_faces - 1) * (number_of_edges))) = 28 :=
by
  sorry

end small_cubes_with_two_faces_painted_l118_118315


namespace correct_calculation_l118_118581

theorem correct_calculation : (∀ x y : ℝ, sqrt x + sqrt y ≠ sqrt (x + y)) ∧
                             (∀ x y : ℝ, (2 * sqrt y) - sqrt y ≠ 2) ∧
                             (∀ x : ℝ, (sqrt 12) / 3 ≠ 2) →
                             (sqrt 2 * sqrt 3 = sqrt 6) := by
  intros h1 h2 h3
  sorry

end correct_calculation_l118_118581


namespace jeff_scores_point_in_5_minutes_l118_118862

theorem jeff_scores_point_in_5_minutes
  (h_play_time : 2 * 60 = 120)
  (h_points_per_game : ∀ (g : ℕ), g = 8)
  (h_won_games : 3)
  (total_points : 3 * 8 = 24) :
  120 / 24 = 5 :=
by
  sorry

end jeff_scores_point_in_5_minutes_l118_118862


namespace cyclic_quadrilateral_diameter_l118_118536

theorem cyclic_quadrilateral_diameter
  (AB BC CD DA : ℝ)
  (h1 : AB = 25)
  (h2 : BC = 39)
  (h3 : CD = 52)
  (h4 : DA = 60) : 
  ∃ D : ℝ, D = 65 :=
by
  sorry

end cyclic_quadrilateral_diameter_l118_118536


namespace cyclic_quadrilateral_iff_eq_dist_l118_118834

theorem cyclic_quadrilateral_iff_eq_dist {A B C D P : Type*} 
  [euclidean_space A] [euclidean_space B] [euclidean_space C] [euclidean_space D] [euclidean_space P] 
  (h_convex : convex_quadrilateral A B C D) 
  (h_diagonal : ¬ (bisects_angle BD ABC ∧ bisects_angle BD CDA))
  (h_P : inside_quadrilateral P A B C D)
  (h_angles1 : ∠ P B C = ∠ D B A)
  (h_angles2 : ∠ P D C = ∠ B D A) :
  cyclic_quadrilateral A B C D ↔ dist A P = dist C P :=
sorry

end cyclic_quadrilateral_iff_eq_dist_l118_118834


namespace henry_apple_weeks_l118_118038

theorem henry_apple_weeks (apples_per_box : ℕ) (boxes : ℕ) (people : ℕ) (apples_per_day : ℕ) (days_per_week : ℕ) :
  apples_per_box = 14 → boxes = 3 → people = 2 → apples_per_day = 1 → days_per_week = 7 →
  (apples_per_box * boxes) / (people * apples_per_day * days_per_week) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end henry_apple_weeks_l118_118038


namespace exists_circle_with_1982_lattice_points_l118_118134

theorem exists_circle_with_1982_lattice_points :
 ∃ (A : ℝ × ℝ) (R : ℝ), (∃ (r1 r2 : ℝ) (pts : ℕ), (r1 < r2 ∧ 
 ∃ P : fin pts → ℤ × ℤ, ∀ i, r1 < dist A (P i) ∧ dist (P i) < r2)) :=
sorry

end exists_circle_with_1982_lattice_points_l118_118134


namespace intersection_product_l118_118758

-- Define the polar equation and conditions
def C1_polar_eqn := ∀ (ρ θ : ℝ), ρ = 4 * real.cos θ

-- Define the parametric equations of C2
def x_C2 (t : ℝ) := 3 - 1/2 * t
def y_C2 (t : ℝ) := (real.sqrt 3 / 2) * t

-- Define point A
def A := (3 : ℝ, 0 : ℝ)

-- Define the Cartesian equation derived from C1_polar_eqn
def C1_cart_eqn := ∀ (x y : ℝ), x^2 + y^2 = 4 * x

-- Define the general equation of C2 derived from parametric equations
def C2_general_eqn := ∀ (x y : ℝ), (real.sqrt 3) * x + y - 3 * (real.sqrt 3) = 0

-- Main theorem statement
theorem intersection_product : 
  ∀ (P Q : (ℝ × ℝ)), 
    (∀ (t : ℝ), 
      (P = (x_C2 t, y_C2 t) ∨ Q = (x_C2 t, y_C2 t)) /\ 
      C1_cart_eqn (x_C2 t) (y_C2 t)) → 
    |(real.norm ((fst P) - (fst A), (snd P) - (snd A)))| * 
    |(real.norm ((fst Q) - (fst A), (snd Q) - (snd A)))| = 3 :=
sorry

end intersection_product_l118_118758


namespace benny_eggs_l118_118658

def dozen := 12

def total_eggs (n: Nat) := n * dozen

theorem benny_eggs:
  total_eggs 7 = 84 := 
by 
  sorry

end benny_eggs_l118_118658


namespace european_math_school_gathering_l118_118152

theorem european_math_school_gathering :
  ∃ n : ℕ, n < 400 ∧ n % 17 = 16 ∧ n % 19 = 12 ∧ n = 288 :=
by
  sorry

end european_math_school_gathering_l118_118152


namespace area_sum_four_smaller_circles_equals_area_of_large_circle_l118_118835

theorem area_sum_four_smaller_circles_equals_area_of_large_circle (R : ℝ) :
  let radius_large := R
  let radius_small := R / 2
  let area_large := π * radius_large^2
  let area_small := π * radius_small^2
  let total_area_small := 4 * area_small
  area_large = total_area_small :=
by
  sorry

end area_sum_four_smaller_circles_equals_area_of_large_circle_l118_118835


namespace consecutive_integers_sum_l118_118395

theorem consecutive_integers_sum (a b : ℤ) (h1 : b = a + 1) (h2 : (a : ℝ) < real.sqrt 7) (h3 : real.sqrt 7 < (b : ℝ)) : a + b = 5 := 
by sorry

end consecutive_integers_sum_l118_118395


namespace three_digit_multiples_of_30_not_75_l118_118781

theorem three_digit_multiples_of_30_not_75 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ 30 ∣ n ∧ ¬ (75 ∣ n)}.card = 21 :=
by sorry

end three_digit_multiples_of_30_not_75_l118_118781


namespace polynomial_coeff_sum_l118_118049

theorem polynomial_coeff_sum :
  let p := (2 : ℝ) * (x^2) - (4 : ℝ) * x + 5
  let q := (8 : ℝ) - (3 : ℝ) * x
  let result := p * q
  let a := result.coeff 3
  let b := result.coeff 2
  let c := result.coeff 1
  let d := result.coeff 0
  9 * a + 3 * b + 2 * c + d = -24 :=
begin
  -- Definitions of p, q, and result
  let p := 2 * (x^2) - 4 * x + 5,
  let q := 8 - 3 * x,
  let result := p * q,

  -- Coefficients of the resulting polynomial
  let a := result.coeff 3,
  let b := result.coeff 2,
  let c := result.coeff 1,
  let d := result.coeff 0,

  -- Expected outcome
  show 9 * a + 3 * b + 2 * c + d = -24,
  exact sorry
end

end polynomial_coeff_sum_l118_118049


namespace john_weight_loss_percentage_l118_118465

def john_initial_weight := 220
def john_final_weight_after_gain := 200
def weight_gain := 2

theorem john_weight_loss_percentage : 
  ∃ P : ℝ, (john_initial_weight - (P / 100) * john_initial_weight + weight_gain = john_final_weight_after_gain) ∧ P = 10 :=
sorry

end john_weight_loss_percentage_l118_118465


namespace probability_nine_heads_l118_118993

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118993


namespace probability_three_draws_exceed_ten_l118_118607

open Finset

-- Definitions based on problem conditions
def chips := range 1 9 -- Chips numbered from 1 to 8

/-- Defining pairs that sum to 8, exclude (4, 4) because chips are unique -/
def valid_pairs : Finset (ℕ × ℕ) := 
  {(1, 7), (2, 6), (3, 5), (5, 3), (6, 2), (7, 1)}.toFinset

/-- Calculate the total probability of valid pairs which sum to 8 -/
def valid_pairs_probability : ℚ := valid_pairs.card / (8 * 7)

-- Defining the probability of third chip drawn exceeding 10
def third_chip_probability : ℚ := 4 / 6

-- Combined final probability
def final_probability : ℚ := valid_pairs_probability * third_chip_probability

theorem probability_three_draws_exceed_ten : 
  final_probability = 1 / 14 :=
by sorry

end probability_three_draws_exceed_ten_l118_118607


namespace three_digit_multiples_of_30_not_75_l118_118779

theorem three_digit_multiples_of_30_not_75 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ 30 ∣ n ∧ ¬ (75 ∣ n)}.card = 21 :=
by sorry

end three_digit_multiples_of_30_not_75_l118_118779


namespace general_term_sum_of_terms_l118_118388

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions provided
axiom h1 : ∀ n : ℕ, a n = 3 * S n - 2 
axiom h2 : ∀ n : ℕ, S n = (finset.range n).sum a

-- Question (1): Finding the general term of the sequence a_n
theorem general_term : ∀ n : ℕ, n > 0 → a n = (-1 / 2) ^ (n - 1) := 
by
  sorry

-- Question (2): Finding the sum of the first n terms of the sequence {n * a_n}
def seq_b (n : ℕ) := (n : ℝ) * a n

axiom h3 : ∀ n : ℕ, T n = (finset.range n).sum seq_b 

theorem sum_of_terms : ∀ n : ℕ, T (n + 1) = 4 / 9 - (2 / 3 * (n + 1) + 4 / 9) * (-1 / 2) ^ (n + 1) := 
by
  sorry

end general_term_sum_of_terms_l118_118388


namespace Kolya_is_correct_Valya_is_incorrect_l118_118288

noncomputable def Kolya_probability (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

noncomputable def Valya_probability_not_losing (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

theorem Kolya_is_correct (x : ℝ) (hx : x > 0) : Kolya_probability x = 1 / 2 :=
by
  sorry

theorem Valya_is_incorrect (x : ℝ) (hx : x > 0) : Valya_probability_not_losing x = 1 / 2 :=
by
  sorry

end Kolya_is_correct_Valya_is_incorrect_l118_118288


namespace three_digit_multiples_l118_118784

theorem three_digit_multiples : 
  let three_digit_range := {x : ℕ | 100 ≤ x ∧ x < 1000}
  let multiples_of_30 := {x ∈ three_digit_range | x % 30 = 0}
  let multiples_of_75 := {x ∈ three_digit_range | x % 75 = 0}
  let count_multiples_of_30 := Set.card multiples_of_30
  let count_multiples_of_75 := Set.card multiples_of_75
  let count_common_multiples := Set.card (multiples_of_30 ∩ multiples_of_75)
  count_multiples_of_30 - count_common_multiples = 24 :=
by
  sorry

end three_digit_multiples_l118_118784


namespace number_of_satisfying_sets_l118_118487

-- Let A be the set {1, 2}
def A : Set ℕ := {1, 2}

-- Define a predicate for sets B that satisfy A ∪ B = {1, 2, 3}
def satisfiesCondition (B : Set ℕ) : Prop :=
  (A ∪ B = {1, 2, 3})

-- The theorem statement asserting there are 4 sets B satisfying the condition
theorem number_of_satisfying_sets : (Finset.filter satisfiesCondition (Finset.powerset {1, 2, 3})).card = 4 :=
by sorry

end number_of_satisfying_sets_l118_118487


namespace trig_identity_example_l118_118696

theorem trig_identity_example :
  (cos (36 * Real.pi / 180) * sin (24 * Real.pi / 180) + sin (144 * Real.pi / 180) * sin (84 * Real.pi / 180)) /
  (cos (44 * Real.pi / 180) * sin (16 * Real.pi / 180) + sin (136 * Real.pi / 180) * sin (76 * Real.pi / 180)) = 1 := by
  sorry

end trig_identity_example_l118_118696


namespace three_digit_multiples_l118_118782

theorem three_digit_multiples : 
  let three_digit_range := {x : ℕ | 100 ≤ x ∧ x < 1000}
  let multiples_of_30 := {x ∈ three_digit_range | x % 30 = 0}
  let multiples_of_75 := {x ∈ three_digit_range | x % 75 = 0}
  let count_multiples_of_30 := Set.card multiples_of_30
  let count_multiples_of_75 := Set.card multiples_of_75
  let count_common_multiples := Set.card (multiples_of_30 ∩ multiples_of_75)
  count_multiples_of_30 - count_common_multiples = 24 :=
by
  sorry

end three_digit_multiples_l118_118782


namespace solution_set_inequality_l118_118964

theorem solution_set_inequality (x : ℝ) : 
  x * (x - 1) ≥ x ↔ x ≤ 0 ∨ x ≥ 2 := 
sorry

end solution_set_inequality_l118_118964


namespace money_left_for_expenses_l118_118463

def d : ℕ → ℕ := sorry
def S_b : ℕ := 5555 -- in base 8
def T_d : ℕ := 1500

lemma decimal_conversion_of_S_b : d S_b = 2925 := 
  sorry

theorem money_left_for_expenses : 
  (d S_b) - T_d = 1425 := 
by
  rw decimal_conversion_of_S_b
  norm_num

end money_left_for_expenses_l118_118463


namespace least_three_digit_base8_divisible_by_7_l118_118566

-- Define the base 8 number system
def base8_to_nat (d2 d1 d0 : Nat) : Nat := d2 * 8^2 + d1 * 8 + d0

-- Define the condition of being a 3-digit base 8 number
def is_three_digit_base8 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 777

-- Define the condition of being divisible by 7
def divisible_by_7 (n : Nat) : Prop :=
  n % 7 = 0

-- State the problem as a theorem
theorem least_three_digit_base8_divisible_by_7 :
    ∃ n, is_three_digit_base8 n ∧ divisible_by_7 n ∧ 
    ∀ m, is_three_digit_base8 m ∧ divisible_by_7 m → n ≤ m :=
  -- The number 70 in base 10
  ∃ n, base8_to_nat 1 0 6 = n ∧ is_three_digit_base8 n ∧ divisible_by_7 n ∧ 
    ( ∀ m, base8_to_nat ((m / 64) % 8) ((m / 8) % 8) (m % 8) = m → is_three_digit_base8 m ∧ divisible_by_7 m → n ≤ m ) :=
begin
  -- Proof will be filled in here
  sorry
end

end least_three_digit_base8_divisible_by_7_l118_118566


namespace tangent_line_eqn_m_range_l118_118026

-- Part (1) tanget line problem
theorem tangent_line_eqn (x : ℝ) (h₁ : 1 ≤ x) (m : ℝ) (h₂ : m = 2) :
  let f (x : ℝ) := real.log (x^m) + 2 * real.exp (x - 1) - 2 * x + m in
  tangent_eqn : tangent_line (f, 1, (f 1), (f' 1)) = λ x, 2 * x :=
sorry

-- Part (2) inequality range problem
theorem m_range (x : ℝ) (h₁ : 1 ≤ x) (m : ℝ)
  (h₂ : ∀ x ≠ 1, let f (x : ℝ) := real.log (x^m) + 2 * real.exp (x - 1) - 2 * x + m in f x ≥ m * x) :
  m ≤ 2 :=
sorry

end tangent_line_eqn_m_range_l118_118026


namespace arithmetic_sequence_diff_l118_118927

theorem arithmetic_sequence_diff (b : ℕ → ℚ) (h1 : ∀ n : ℕ, b (n + 1) - b n = b 1 - b 0)
  (h2 : (Finset.range 150).sum b = 150)
  (h3 : (Finset.Ico 150 300).sum b = 300) : b 2 - b 1 = 1 / 150 :=
by
  sorry

end arithmetic_sequence_diff_l118_118927


namespace problem_l118_118043

theorem problem (x : ℝ) (h : x + 2 / x = 4) : - (5 * x) / (x^2 + 2) = -5 / 4 := 
sorry

end problem_l118_118043


namespace sufficient_condition_for_magnitude_equality_l118_118731

noncomputable def vec_mag (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem sufficient_condition_for_magnitude_equality
  (a b : ℝ × ℝ × ℝ)
  (ha : a ≠ (0, 0, 0))
  (hb : b ≠ (0, 0, 0))
  (h : a = (-2) • b) :
  vec_mag a - vec_mag b = vec_mag (a + b) :=
by
  sorry

end sufficient_condition_for_magnitude_equality_l118_118731


namespace sum_of_x_satisfying_equation_l118_118900

noncomputable def g (x : ℝ) : ℝ := 10 * x + 5
noncomputable def g_inv (y : ℝ) : ℝ := (y - 5) / 10
noncomputable def f (x : ℝ) : ℝ := g ((3 * x)⁻²)

theorem sum_of_x_satisfying_equation :
  let equation := (g_inv x = f x) in
  (x₁ + x₂ + x₃) where x₁, x₂, x₃ are roots of the equation = 50 :=
by
  -- Specify that the roots should satisfy the equation
  have roots_satisfy_equation : ∀ x, g_inv x = f x → is_subset [x₁, x₂, x₃] equation
  sorry
  -- Use Vieta's formulas or structure of cubic equation to show sum of roots
  have sum_of_roots := -(-450) / 9
  exact sum_of_roots = 50

end sum_of_x_satisfying_equation_l118_118900


namespace min_distinct_integers_needed_l118_118918

-- Defining the number of terms in sequence
def n : ℕ := 2006

-- Define the function to count distinct ratios for given distinct integers
def distinct_ratios (num_distinct : ℕ) : ℕ :=
  num_distinct * (num_distinct - 1)

-- Theorem stating that the minimum number of distinct integers required is 46
theorem min_distinct_integers_needed (a : ℕ → ℕ) (h_pos : ∀ i, 1 ≤ a i) :
  (∀ n : ℕ, (distinct_ratios n < 2005) → n < 46) :=
begin
  sorry
end

end min_distinct_integers_needed_l118_118918


namespace length_of_cube_l118_118492

theorem length_of_cube
  (density : ℕ := 19)
  (cost_per_gram : ℕ := 60)
  (sale_multiplier : ℝ := 1.5)
  (profit : ℕ := 123120) :
  ∃ (L : ℝ), L^3 = 216 ∧ L = 6 :=
begin
  use 6,
  split,
  { norm_num },
  { refl }
end

end length_of_cube_l118_118492


namespace hexagon_area_twice_triangle_area_l118_118472

-- Definitions based on the given problem
variables (A B C O A' B' C' : Point)
variable [circles γ]

-- Conditions provided
def is_acute_triangle (A B C : Point) : Prop := 
  ∀ <terms involving angles> -- specify within the allowed Lean syntax

def is_circumcenter (O : Point) (A B C : Point) : Prop := 
  <terms specifying O as the circumcenter of triangle ABC>

def intersects_at_circumcircle (γ : Circle) (P Q : Point) : Point :=
  <terms specifying the second intersection point>

-- Translate the problem into solution proof
theorem hexagon_area_twice_triangle_area 
  (h_acute : is_acute_triangle A B C)
  (h_circumcenter : is_circumcenter O A B C)
  (h_intersect_A : intersects_at_circumcircle γ (A O) = A')
  (h_intersect_B : intersects_at_circumcircle γ (B O) = B')
  (h_intersect_C : intersects_at_circumcircle γ (C O) = C') :
  area_of_hexagon A C' B A' C B' = 2 * area_of_triangle A B C :=
sorry

end hexagon_area_twice_triangle_area_l118_118472


namespace collinearity_of_D_E_F_l118_118448

variables {A B C H P H1 H2 H3 D E F : Type}

-- Conditions
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited H] [Inhabited P] [Inhabited H1] [Inhabited H2] [Inhabited H3] [Inhabited D] [Inhabited E] [Inhabited F]

-- Definitions:
variable (circumcircle : ∀ (Δ : Type), Δ → Δ → Δ → Set Δ)
variable (orthocenter : ∀ (Δ : Type), Δ → Δ → Δ → Δ)
variable (reflect : ∀ {Δ : Type}, Δ → Δ → Δ)
variable (on_circumcircle : ∀ {Δ : Type}, Δ → Δ → Δ → Δ → Prop)
variable (perpendicular : ∀ {Δ : Type}, Δ → Δ → Δ → Δ → Prop)
variable (intersection : ∀ {Δ : Type}, Δ → Δ → Δ → Δ → Δ)

-- Let P be an arbitrary point on the circumcircle
variable (arbitrary_point : ∀ Δ, Set Δ)

-- Definition of inscribed triangle Δ
def Δ {A B C : Type} := A → B → C → Prop 

-- Proof statement
theorem collinearity_of_D_E_F
  (h_inscribed : Δ A B C)
  (h_orthocenter : H = orthocenter A B C)
  (h_reflect1 : H1 = reflect H (segment A B))
  (h_reflect2 : H2 = reflect H (segment B C))
  (h_reflect3 : H3 = reflect H (segment C A))
  (h_P_on_circumcircle : on_circumcircle P A B C)
  (h_intersect1 : D = intersection P H1 A B)
  (h_intersect2 : E = intersection P H2 B C)
  (h_intersect3 : F = intersection P H3 C A)
  : collinear D E F :=
sorry

end collinearity_of_D_E_F_l118_118448


namespace EF_parallel_BC_l118_118005

variables {α : Type*} [plane α]

noncomputable def is_parallel {P Q X Y : α.pts} :=
  ∃ l₁ l₂, line colinear P Q l₁ ∧ line colinear X Y l₂ ∧ parallel l₁ l₂

theorem EF_parallel_BC
  (ABC : Triangle α)
  (AB_eq_AC : ABC.AB = ABC.AC)
  (Ω : Circle ABC)
  (D : α.pts)
  (D_on_minor_arc_AB : arc AB Ω)
  (E : α.pts)
  (extended_AD : colinear A D E∆)
  (A_E_same_side_of_BC : same_side A E BC)
  (ω : Circle BDE)
  (F : α.pts)
  (F_on_AB_and_ω : colinear F AB ∧ colinear F ω) :
  is_parallel EF BC :=
sorry

end EF_parallel_BC_l118_118005


namespace count_sets_B_l118_118485

open Set

def A : Set ℕ := {1, 2}

theorem count_sets_B (B : Set ℕ) (h1 : A ∪ B = {1, 2, 3}) : 
  (∃ Bs : Finset (Set ℕ), ∀ b ∈ Bs, A ∪ b = {1, 2, 3} ∧ Bs.card = 4) := sorry

end count_sets_B_l118_118485


namespace problem_statement_l118_118814

noncomputable def a : ℕ := 44
noncomputable def b : ℕ := 36
noncomputable def c : ℕ := 33

theorem problem_statement : \( \sqrt{3}+\frac{1}{\sqrt{3}} + \sqrt{11} + \frac{1}{\sqrt{11}} = \dfrac{44\sqrt{3} + 36\sqrt{11}}{33} \) ∧ a + b + c = 113 :=
by
    sorry

end problem_statement_l118_118814


namespace max_c_l118_118013

theorem max_c (c : ℝ) : 
  (∀ x y : ℝ, x > y ∧ y > 0 → x^2 - 2 * y^2 ≤ c * x * (y - x)) 
  → c ≤ 2 * Real.sqrt 2 - 4 := 
by
  sorry

end max_c_l118_118013


namespace apple_selling_price_l118_118635

theorem apple_selling_price (CP SP Loss : ℝ) (h₀ : CP = 18) (h₁ : Loss = (1/6) * CP) (h₂ : SP = CP - Loss) : SP = 15 :=
  sorry

end apple_selling_price_l118_118635


namespace sum_of_first_10_terms_of_geometric_sequence_l118_118743

theorem sum_of_first_10_terms_of_geometric_sequence :
  let b : ℕ → ℝ := λ n => (2 : ℝ)^(n-1) in  -- Define the geometric sequence {b_n}
  ∑ i in finset.range 10, b (2 * i + 1) = (1 / 3) * (4^10 - 1) := 
by
  sorry  -- Proof is omitted

end sum_of_first_10_terms_of_geometric_sequence_l118_118743


namespace sum_of_calculators_is_307_l118_118837

def initial_values : ℕ × ℕ × ℕ := (2, 1, 0)

def operations (x y z : ℕ) : ℕ × ℕ × ℕ :=
  (x^3, y^2, z + 1)

def after_one_round (values : ℕ × ℕ × ℕ) (n : ℕ) : ℕ × ℕ × ℕ :=
  Nat.iterate n (λ (vals : ℕ × ℕ × ℕ), operations vals.1 vals.2 vals.3) values

def final_values : ℕ × ℕ × ℕ :=
  after_one_round initial_values 50

def final_sum (values : ℕ × ℕ × ℕ) : ℕ :=
  values.1 + values.2 + values.3

theorem sum_of_calculators_is_307 : final_sum final_values = 307 := by
  sorry

end sum_of_calculators_is_307_l118_118837


namespace polynomial_divisible_by_square_l118_118139

noncomputable def polynomial := λ (a₁ a₂ a₃ a₄ x : ℝ), x^4 + a₁ * x^3 + a₂ * x^2 + a₃ * x + a₄

theorem polynomial_divisible_by_square (a₁ a₂ a₃ a₄ x₀ : ℝ)
    (h₁ : polynomial a₁ a₂ a₃ a₄ x₀ = 0)
    (h₂ : polynomial a₁ a₂ a₃ a₄'.derivative x₀ = 0) : 
    ∃ g : ℝ → ℝ, polynomial a₁ a₂ a₃ a₄ = (λ x, (x - x₀)^2 * g x) :=
sorry

end polynomial_divisible_by_square_l118_118139


namespace leftover_coverage_l118_118680

variable (bagCoverage lawnLength lawnWidth bagsPurchased : ℕ)

def area_of_lawn (length width : ℕ) : ℕ :=
  length * width

def total_coverage (bagCoverage bags : ℕ) : ℕ :=
  bags * bagCoverage

theorem leftover_coverage :
  let lawnLength := 22
  let lawnWidth := 36
  let bagCoverage := 250
  let bagsPurchased := 4
  let lawnArea := area_of_lawn lawnLength lawnWidth
  let totalSeedCoverage := total_coverage bagCoverage bagsPurchased
  totalSeedCoverage - lawnArea = 208 := by
  sorry

end leftover_coverage_l118_118680


namespace profit_calculation_l118_118613

noncomputable def profit_amount (SP : ℝ) (profit_percentage : ℝ) := 
  (profit_percentage / 100) * (SP / (1 + profit_percentage / 100))

theorem profit_calculation : profit_amount 850 31.782945736434108 ≈ 204.84 :=
  by
    sorry

end profit_calculation_l118_118613


namespace polar_equation_of_line_segment_l118_118048

theorem polar_equation_of_line_segment :
  ∀ (ρ θ : ℝ), (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → (x = ρ * cos θ ∧ (1 - x) = ρ * sin θ)) → 
  (0 ≤ θ ∧ θ ≤ π/2) → 
 ρ = 1 / (cos θ + sin θ) :=
by
  intros ρ θ h line_segment_θ_range
  sorry

end polar_equation_of_line_segment_l118_118048


namespace sum_of_x_i_lt_16n_over_33_l118_118902

theorem sum_of_x_i_lt_16n_over_33 (n : ℕ) (x : Fin n → ℝ) (h1 : n ≥ 3)
  (h2 : ∀ i, x i ∈ Set.Ici (-1))
  (h3 : ∑ i, (x i) ^ 5 = 0) :
  ∑ i, x i < 16 * n / 33 := 
sorry

end sum_of_x_i_lt_16n_over_33_l118_118902


namespace abc_le_one_eighth_l118_118871

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : 
  a * b * c ≤ 1 / 8 := 
by
  sorry

end abc_le_one_eighth_l118_118871


namespace circle_radius_zero_l118_118701

theorem circle_radius_zero (x y : ℝ) : 2*x^2 - 8*x + 2*y^2 + 4*y + 10 = 0 → (x - 2)^2 + (y + 1)^2 = 0 :=
by
  intro h
  sorry

end circle_radius_zero_l118_118701


namespace min_period_f_interval_monotonic_decrease_min_value_on_interval_l118_118023

noncomputable def f (x : ℝ) : ℝ := sin (-2 * x) + cos (-2 * x)

theorem min_period_f : ∀ s, s > 0 ∧ (∀ x ∈ ℝ, f (x + s) = f x) → s >= π := by
  sorry

theorem interval_monotonic_decrease : 
  ∃ k : ℤ, ∀ x ∈ set.Icc (-(π / 8) + k * π) ((3 * π / 8) + k * π), 
  ∀ x₁ x₂, x₁ < x₂ → f x₁ >= f x₂ := by
  sorry

theorem min_value_on_interval : 
  ∃ x ∈ set.Icc 0 (π / 2), f x = -Real.sqrt 2 ∧ x = (3 * π / 8) := by
  sorry

end min_period_f_interval_monotonic_decrease_min_value_on_interval_l118_118023


namespace no_real_roots_f_of_f_eq_x_l118_118897

-- Definitions and conditions as provided above
variable {b c : ℝ}
def f (x : ℝ) : ℝ := x^2 + b * x + c

-- Define the condition about no real roots for f(x) = x
def no_real_roots_f_eq_x : Prop := (b - 1)^2 - 4 * c < 0

-- Problem statement: Prove that f(f(x)) = x has no real roots
theorem no_real_roots_f_of_f_eq_x (h : no_real_roots_f_eq_x) :
  ∀ x : ℝ, f(f(x)) ≠ x := by
  sorry

end no_real_roots_f_of_f_eq_x_l118_118897


namespace larger_number_l118_118554

theorem larger_number (a b : ℤ) (h1 : a - b = 5) (h2 : a + b = 37) : a = 21 :=
sorry

end larger_number_l118_118554


namespace reachable_FromCapitalToDalniy_l118_118829

noncomputable def kingdom_graph (V : Type) [Finite V] (capital Dalniy : V) : SimpleGraph V :=
  { adjacency := λ u v, (u = capital ∧ v ∈ city_neighbors capital) ∨
                        (u = Dalniy ∧ v ∈ city_neighbors Dalniy) ∨
                        (u ≠ capital ∧ u ≠ Dalniy ∧ v ∈ city_neighbors u),
    sym := sorry }  -- Define adjacency symmetrically

def city_neighbors (v : V) : Set V :=
  if v = capital then { x | adj_capital x }    -- 21 neighbors
  else if v = Dalniy then { x | adj_Dalniy x } -- 1 neighbor
  else { x | adj_other x }                     -- 20 neighbors

axiom adj_capital : V → Prop                   -- Define 21 neighbors of capital
axiom adj_Dalniy : V → Prop                    -- Define 1 neighbor of Distant
axiom adj_other : V → Prop                     -- Define 20 neighbors of other cities

theorem reachable_FromCapitalToDalniy
  (G : SimpleGraph V) [Fintype V]
  {capital Dalniy : V} (hc : G.degree capital = 21) (hd : G.degree Dalniy = 1)
  (ho : ∀ v, v ≠ capital ∧ v ≠ Dalniy → G.degree v = 20) :
  ∃ p : Path G capital Dalniy, True :=
sorry

end reachable_FromCapitalToDalniy_l118_118829


namespace solution_set_l118_118755

open Real

noncomputable def f : ℝ → ℝ := sorry -- The function f is abstractly defined
axiom f_point : f 1 = 0 -- f passes through (1, 0)
axiom f_deriv_pos : ∀ (x : ℝ), x > 0 → x * (deriv f x) > 1 -- xf'(x) > 1 for x > 0

theorem solution_set (x : ℝ) : f x ≤ log x ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end solution_set_l118_118755


namespace part1_part2_part3_l118_118406

noncomputable def f (x a : ℝ) := log x - a + a / x

theorem part1 (h : f' = λ x a, (x - a) / x^2) (h_tangent : f' 1 a = 0) :
  a = 1 := by
  sorry

theorem part2 (h : f' = λ x a, (x - a) / x^2) (h_zero_conditions : 0 < a → (a ≤ 1 ∨ a ≥ e / (e - 1))) : 
  ((∀ x ∈ Set.Ioo 1 e, f x a ≠ 0) ∧ (1 < a ∧ a < e/(e - 1) → ∃ x ∈ Set.Ioo 1 e, f x a = 0)) := by
  sorry

theorem part3 (x1 x2 : ℝ) (h_interval : x1 ∈ Set.Ioo 1 e ∧ x2 ∈ Set.Ioo 1 e) 
  (h_condition : (x1 - x2) * (abs (f x1 a) - abs (f x2 a)) > 0)
  (h_values_a : 0 < a ∧ (a ≤ 1 ∨ a ≥ e)) :
  True := by
  sorry

end part1_part2_part3_l118_118406


namespace smallest_square_length_proof_l118_118218

-- Define square side length required properties
noncomputable def smallest_square_side_length (rect_w rect_h min_side : ℝ) : ℝ :=
  if h : min_side^2 % (rect_w * rect_h) = 0 then min_side 
  else if h : (min_side + 1)^2 % (rect_w * rect_h) = 0 then min_side + 1
  else if h : (min_side + 2)^2 % (rect_w * rect_h) = 0 then min_side + 2
  else if h : (min_side + 3)^2 % (rect_w * rect_h) = 0 then min_side + 3
  else if h : (min_side + 4)^2 % (rect_w * rect_h) = 0 then min_side + 4
  else if h : (min_side + 5)^2 % (rect_w * rect_h) = 0 then min_side + 5
  else if h : (min_side + 6)^2 % (rect_w * rect_h) = 0 then min_side + 6
  else if h : (min_side + 7)^2 % (rect_w * rect_h) = 0 then min_side + 7
  else if h : (min_side + 8)^2 % (rect_w * rect_h) = 0 then min_side + 8
  else if h : (min_side + 9)^2 % (rect_w * rect_h) = 0 then min_side + 9
  else min_side + 2 -- ensuring it can't be less than min_side

-- State the theorem
theorem smallest_square_length_proof : smallest_square_side_length 2 3 10 = 12 :=
by 
  unfold smallest_square_side_length
  norm_num
  sorry

end smallest_square_length_proof_l118_118218


namespace part1_proof_part2_proof_l118_118080

-- Definitions corresponding to the conditions in a)
variables (a b c BD : ℝ) (A B C : RealAngle)
variables (D : Point) (AD DC : ℝ)

-- Replace the conditions with the necessary hypotheses
hypothesis h1 : b^2 = a * c
hypothesis h2 : BD * sin B = a * sin C
hypothesis h3 : AD = 2 * DC

noncomputable def Part1 : Prop := BD = b

theorem part1_proof : Part1 a b c BD A B C do
  sorry

noncomputable def Part2 : Prop := cos B = 7 / 12

theorem part2_proof (hADDC : AD = 2 * DC) : Part2 A B C.
  sorry

end part1_proof_part2_proof_l118_118080


namespace divide_L_shape_into_four_congruent_parts_l118_118103

theorem divide_L_shape_into_four_congruent_parts
  (L : Type) (is_L_shaped : L → Prop) (L_region : L)
  (condition : is_L_shaped L_region) :
  ∃ cuts : list (L → L),
  (∀ i < 4, cuts i = cuts (4 - i - 1)) ∧
  ∀ i j < 4, ∃ (P : L → Prop), P (cuts i L_region) ∧ P (cuts j L_region) :=
by
  sorry

end divide_L_shape_into_four_congruent_parts_l118_118103


namespace first_player_win_with_perfect_play_l118_118201

-- Define the set of numbers used in the game
def number_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the 2x4 board
structure Board :=
  (top_row : Fin 4 → Option ℕ)
  (bottom_row : Fin 4 → Option ℕ)

-- Condition: Players take turns placing numbers into an initially empty 2x4 array 
def initial_board : Board := 
  { top_row := λ _, none, 
    bottom_row := λ _, none }

-- Define the win conditions
def first_player_wins (b : Board) : Prop :=
  (∏ i, b.top_row i.getD 1) > (∏ i, b.bottom_row i.getD 1)

def second_player_wins (b : Board) : Prop :=
  (∏ i, b.bottom_row i.getD 1) > (∏ i, b.top_row i.getD 1)

-- Define the turn structure and perfect play
def perfect_play : Prop := sorry -- this represents perfect strategy

-- Prove that the first player wins with a perfect strategy
theorem first_player_win_with_perfect_play :
  perfect_play → ∀ b, first_player_wins b :=
sorry

end first_player_win_with_perfect_play_l118_118201


namespace jan_storage_fraction_l118_118098

def jan_sections (total_feet : ℕ) (section_feet : ℕ) : ℕ :=
  total_feet / section_feet

def friend_sections (total_sections : ℕ) (fraction : ℚ) : ℕ :=
  (fraction * total_sections).toNat

def remaining_sections (total_sections given_sections : ℕ) : ℕ :=
  total_sections - given_sections

def storage_sections (remaining kept_on_hand : ℕ) : ℕ :=
  remaining - kept_on_hand

def storage_fraction (storage remaining : ℕ) : ℚ :=
  storage / remaining

theorem jan_storage_fraction (total_feet : ℕ) (section_feet : ℕ)
  (fraction_to_friend : ℚ) (kept_on_hand : ℕ) :
  storage_fraction (storage_sections
    (remaining_sections 
      (jan_sections total_feet section_feet) 
      (friend_sections 
        (jan_sections total_feet section_feet) 
        fraction_to_friend)
    ) 
    kept_on_hand
  ) 
  (remaining_sections
    (jan_sections total_feet section_feet) 
    (friend_sections 
      (jan_sections total_feet section_feet) 
      fraction_to_friend)
  ) = 1 / 2 :=
by
  sorry

end jan_storage_fraction_l118_118098


namespace circle_center_sum_l118_118361

theorem circle_center_sum (a b : ℝ) :
    (∀ x y : ℝ, x^2 + y^2 + 6 * x - 4 * y - 12 = 0 → (x + 3) ^ 2 + (y - 2) ^ 2 = 25) →
    a = -3 →
    b = 2 →
    a + b = -1 :=
by
  intros h ha hb
  rw [ha, hb]
  norm_num
  sorry

end circle_center_sum_l118_118361


namespace probability_nine_heads_l118_118988

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118988


namespace sequences_properties_sum_first_2n_terms_l118_118850

def geometric_sequence (a_n : ℕ → ℤ) (q : ℤ) := ∀ n, a_n (n + 1) = a_n n * q
def arithmetic_sequence (b_n : ℕ → ℤ) (d : ℤ) := ∀ n, b_n (n + 1) = b_n n + d

noncomputable def a_n : ℕ → ℤ := λ n, 3 ^ n
noncomputable def b_n : ℕ → ℤ := λ n, 2 * n + 1

def c_n (n : ℕ) : ℤ := (-1) ^ n * b_n n + a_n n

theorem sequences_properties (q : ℤ) (h_q : q ≠ 1) :
  geometric_sequence a_n q ∧ arithmetic_sequence b_n 2 ∧
  b_n 1 = a_n 1 ∧ b_n 4 = a_n 2 ∧ b_n 13 = a_n 3 :=
by
  sorry

theorem sum_first_2n_terms (n : ℕ) :
  let S_2n := ∑ i in range (2 * n), c_n i
  S_2n = (3 ^ n - 3) / 2 + 2 * n :=
by
  sorry

end sequences_properties_sum_first_2n_terms_l118_118850


namespace find_length_AB_l118_118415

noncomputable def AB_length (m : ℝ) : ℝ :=
  if 0 < |m| ∧ |m| < 2 * Real.sqrt 5 ∧ m ^ 2 = 10
  then Real.sqrt 10
  else 0

theorem find_length_AB (m : ℝ) (h : 0 < |m| ∧ |m| < 2 * Real.sqrt 5 ∧ m ^ 2 = 10) :
  AB_length m = Real.sqrt 10 := by
  unfold AB_length
  simp [h]
  sorry

end find_length_AB_l118_118415


namespace p_correct_l118_118699

noncomputable def p : ℝ → ℝ := sorry

axiom p_at_3 : p 3 = 10

axiom p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2

theorem p_correct : ∀ x, p x = x^2 + 1 :=
sorry

end p_correct_l118_118699


namespace tilly_bag_cost_l118_118975

noncomputable def cost_per_bag (n s P τ F : ℕ) : ℕ :=
  let revenue := n * s
  let total_sales_tax := n * (s * τ / 100)
  let total_additional_expenses := total_sales_tax + F
  (revenue - (P + total_additional_expenses)) / n

theorem tilly_bag_cost :
  let n := 100
  let s := 10
  let P := 300
  let τ := 5
  let F := 50
  cost_per_bag n s P τ F = 6 :=
  by
    let n := 100
    let s := 10
    let P := 300
    let τ := 5
    let F := 50
    have : cost_per_bag n s P τ F = 6 := sorry
    exact this

end tilly_bag_cost_l118_118975


namespace condition_I_condition_II_condition_III_l118_118410

theorem condition_I (m : ℝ) : 
  (A = ∅) → (m > 1) :=
sorry

theorem condition_II (m : ℝ) : 
  (A.has_exactly_two_subsets) → (m = 0 ∨ m = 1) :=
sorry

theorem condition_III (m : ℝ) : 
  (∃ x ∈ (1/2 : ℝ, 2 : ℝ), x ∈ A) → (0 < m ∧ m ≤ 1) :=
sorry

end condition_I_condition_II_condition_III_l118_118410


namespace complement_U_A_l118_118811

def U : Set ℝ := { x | x^2 ≤ 4 }
def A : Set ℝ := { x | abs (x + 1) ≤ 1 }

theorem complement_U_A :
  (U \ A) = { x | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end complement_U_A_l118_118811


namespace polygon_diagonals_30_l118_118660

-- Define the properties and conditions of the problem
def sides := 30

-- Define the number of diagonals calculation function
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement to check the number of diagonals in a 30-sided convex polygon
theorem polygon_diagonals_30 : num_diagonals sides = 375 := by
  sorry

end polygon_diagonals_30_l118_118660


namespace distinguishable_colorings_l118_118682

-- Define types or properties based on the problem conditions
def Cube := {faces : List (Fin 6) // faces.length = 6} -- A cube with 6 faces

-- Assume functions for coloring and rotations
def isIndistinguishableUnderRotation (c1 c2 : Cube) : Prop := sorry -- This function checks if two colorings are the same up to rotation

def allColorings := List Cube -- List of all possible colorings without considering rotations

-- Problem statement: Prove the number of distinguishable colorings, considering rotations
theorem distinguishable_colorings (colorings : List Cube) (h : colorings = allColorings) : ∃ (n : Nat), n = 3 :=
sorry

end distinguishable_colorings_l118_118682


namespace y_relationship_income_not_decrease_maximize_income_at_2_5_l118_118647

-- Definitions of the conditions
def price (x : ℝ) : ℝ := 1 + x / 100
def pieces_sold (x : ℝ) : ℝ := 1 - (x / 100) * (2 / 3)
def operating_income_ratio (x : ℝ) : ℝ := price x * pieces_sold x

-- Question (1)
theorem y_relationship (x : ℝ) : 
  operating_income_ratio x =  1 - (1 / 1500) * x^2 + (1 / 300) * x + 1 :=
sorry

-- Question (2)
theorem income_not_decrease (x : ℝ) : 
  0 ≤ x ∧ x ≤ 5 → operating_income_ratio x ≥ 1 :=
by
  sorry

-- Question (3)
theorem maximize_income_at_2_5 :
  ∀ x : ℝ, operating_income_ratio x ≤ operating_income_ratio 2.5 :=
by
  sorry

end y_relationship_income_not_decrease_maximize_income_at_2_5_l118_118647


namespace tangent_line_to_circle_l118_118449

noncomputable def problem_statement : Prop :=
  ∃ (θ : ℝ) (a : ℝ), 
    (∀ θ, ∃ ρ, ρ = 2 * sin θ) ∧
    (∀ θ, ∃ ρ, ρ * sin (θ + π / 3) = a) ∧
    (a = 3 / 2 ∨ a = -1 / 2)

theorem tangent_line_to_circle : problem_statement := sorry

end tangent_line_to_circle_l118_118449


namespace angle_A_is_2pi_over_3_l118_118823

variable {a b c : ℝ} -- Define the sides as real numbers.
variable {A B C : ℝ} -- Define the angles as real numbers.

noncomputable def m : ℝ × ℝ := (a + c, -b)
noncomputable def n : ℝ × ℝ := (a - c, b)

theorem angle_A_is_2pi_over_3
  (h : m.1 * n.1 + m.2 * n.2 = b * c) -- Dot product condition
  (ha2 : a^2 = b^2 + c^2 + b * c) -- Derived from the previous condition
  (ha_cos : cos A = (b^2 + c^2 - a^2) / (2 * b * c)) -- Cosine rule
  (hangle : 0 < A ∧ A < π) -- Angle in range
  : A = 2 * π / 3 := sorry

end angle_A_is_2pi_over_3_l118_118823


namespace probability_nine_heads_l118_118995

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118995


namespace sum_abc_eq_113_l118_118816

theorem sum_abc_eq_113 :
  (∃ (a b c : ℕ), c > 0 ∧
    (c * (√3 + 1/√3 + √11 + 1/√11) = a * √3 + b * √11) ∧ 
    (∀ c', c' > 0 ∧ (c' * (√3 + 1/√3 + √11 + 1/√11) = a * √3 + b * √11) → c' ≥ c) ∧ 
    a + b + c = 113) :=
begin
  sorry
end

end sum_abc_eq_113_l118_118816


namespace probability_heads_9_of_12_flips_l118_118998

theorem probability_heads_9_of_12_flips : (nat.choose 12 9) / (2^12) = 55 / 1024 :=
by
  sorry

end probability_heads_9_of_12_flips_l118_118998


namespace second_discount_rate_l118_118602

theorem second_discount_rate (P S : ℝ) (d₁ p : ℝ) :
  P = 124.67 →
  S = 89.38 →
  d₁ = 0.185 →
  p ≈ 0.1203 :=
by
  intros
  sorry

end second_discount_rate_l118_118602


namespace amanda_lighter_savings_l118_118490

theorem amanda_lighter_savings :
  let gas_station_cost_per_lighter := 1.75
  let amazon_pack_cost := 5.00
  let lighters_needed := 24
  let pack_size := 12 in
  let total_cost_gas_station := lighters_needed * gas_station_cost_per_lighter in
  let packs_needed := lighters_needed / pack_size in
  let total_cost_amazon := packs_needed * amazon_pack_cost in
  let savings := total_cost_gas_station - total_cost_amazon in
  savings = 32 :=
by
  sorry

end amanda_lighter_savings_l118_118490


namespace find_value_of_a_squared_plus_b_squared_l118_118430

-- Defining the problem's conditions as Lean definitions
variable {a b : ℕ}
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a^2 + 2 * a * b - 3 * b^2 - 41 = 0)

-- Theorem statement to prove
theorem find_value_of_a_squared_plus_b_squared (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + 2 * a * b - 3 * b^2 - 41 = 0) : 
  a^2 + b^2 = 221 :=
begin
  sorry
end

end find_value_of_a_squared_plus_b_squared_l118_118430


namespace probability_divisible_by_4_l118_118208

theorem probability_divisible_by_4 :
  (probability (product_of_dice_divisible_by_4 (list_of_rolls dice_rolls 6))) = 61 / 64 :=
sorry

-- Definitions relevant to the problem

def dice_faces : list ℕ := [1, 2, 3, 4, 5, 6]

def dice_rolls (n : ℕ) : list (list ℕ) :=
(list.replicate n dice_faces).sequence

def product (l : list ℕ) : ℕ := l.foldl (*) 1

def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def product_of_dice_divisible_by_4 (rolls : list (list ℕ)) : ℕ → ℕ :=
  λ n, rolls.count (λ l, divisible_by_4 (product l))

-- Probability function
noncomputable def probability (event_count : ℕ) : ℚ :=
event_count / (6 ^ 6)

end probability_divisible_by_4_l118_118208


namespace three_digit_multiples_l118_118785

theorem three_digit_multiples : 
  let three_digit_range := {x : ℕ | 100 ≤ x ∧ x < 1000}
  let multiples_of_30 := {x ∈ three_digit_range | x % 30 = 0}
  let multiples_of_75 := {x ∈ three_digit_range | x % 75 = 0}
  let count_multiples_of_30 := Set.card multiples_of_30
  let count_multiples_of_75 := Set.card multiples_of_75
  let count_common_multiples := Set.card (multiples_of_30 ∩ multiples_of_75)
  count_multiples_of_30 - count_common_multiples = 24 :=
by
  sorry

end three_digit_multiples_l118_118785


namespace village_population_rate_l118_118213

theorem village_population_rate (R : ℕ) :
  (76000 - 17 * R = 42000 + 17 * 800) → R = 1200 :=
by
  intro h
  -- The actual proof is omitted.
  sorry

end village_population_rate_l118_118213


namespace most_cost_effective_way_cost_is_860_l118_118968

-- Definitions based on the problem conditions
def adult_cost := 150
def child_cost := 60
def group_cost_per_person := 100
def group_min_size := 5

-- Number of adults and children
def num_adults := 4
def num_children := 7

-- Calculate the total cost for the most cost-effective way
noncomputable def most_cost_effective_way_cost :=
  let group_tickets_count := 5  -- 4 adults + 1 child
  let remaining_children := num_children - 1
  group_tickets_count * group_cost_per_person + remaining_children * child_cost

-- Theorem to state the cost for the most cost-effective way
theorem most_cost_effective_way_cost_is_860 : most_cost_effective_way_cost = 860 := by
  sorry

end most_cost_effective_way_cost_is_860_l118_118968


namespace train_length_correct_l118_118280

noncomputable def train_length (time : ℝ) (platform_length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time
  total_distance - platform_length

theorem train_length_correct :
  train_length 17.998560115190784 200 90 = 249.9640028797696 :=
by
  sorry

end train_length_correct_l118_118280


namespace total_weekly_pay_proof_l118_118552

-- Define the weekly pay for employees X and Y
def weekly_pay_employee_y : ℝ := 260
def weekly_pay_employee_x : ℝ := 1.2 * weekly_pay_employee_y

-- Definition of total weekly pay
def total_weekly_pay : ℝ := weekly_pay_employee_x + weekly_pay_employee_y

-- Theorem stating the total weekly pay equals 572
theorem total_weekly_pay_proof : total_weekly_pay = 572 := by
  sorry

end total_weekly_pay_proof_l118_118552


namespace boats_left_l118_118125

theorem boats_left (initial_boats : ℕ) (percent_eaten : ℝ) (shot_boats : ℕ) (eaten := (percent_eaten * initial_boats.to_real).to_nat) 
(remaining_boats := initial_boats - eaten - shot_boats) :
  initial_boats = 30 → percent_eaten = 0.2 → shot_boats = 2 → remaining_boats = 22 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

end boats_left_l118_118125


namespace sum_of_distances_condition_l118_118244

theorem sum_of_distances_condition (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| < a) → a > 4 :=
sorry

end sum_of_distances_condition_l118_118244


namespace find_PR_l118_118069

-- Define the right triangle PQR and the given conditions
variables {P Q R : Type} [normed_add_torsor ℝ P]

-- lengths of the sides of the triangle
variables (QR PR : ℝ)

-- angle Q of the triangle
variables (θ : ℝ)

-- Given conditions
def is_right_triangle (P Q R : Type) [normed_add_torsor ℝ P] : Prop := sorry
def sin_Q (θ : ℝ) : Prop := θ = real.arcsin ((7 * real.sqrt 53) / 53)
def length_QR_eq_8 (QR : ℝ) : Prop := QR = 8

-- The statement to prove
theorem find_PR (P Q R : Type) [normed_add_torsor ℝ P] (QR : ℝ) (θ : ℝ)
  (h1 : is_right_triangle P Q R)
  (h2 : sin_Q θ)
  (h3 : length_QR_eq_8 QR) :
  PR = (56 * real.sqrt 53) / 53 :=
sorry

end find_PR_l118_118069


namespace mean_and_variance_of_dataset_l118_118723

theorem mean_and_variance_of_dataset :
  let data := [-1, 0, 4, 6, 7, 14]
  (list.median data = 5) →
  (list.mean data = 5) ∧ (list.variance data = 74 / 3) :=
by
  sorry

-- Note: list.median, list.mean, and list.variance are placeholders for appropriate functions.

end mean_and_variance_of_dataset_l118_118723


namespace part_whole_ratio_l118_118628

theorem part_whole_ratio (N x : ℕ) (hN : N = 160) (hx : x + 4 = N / 4 - 4) :
  x / N = 1 / 5 :=
  sorry

end part_whole_ratio_l118_118628


namespace savings_is_200_l118_118278

-- Define the constant costs and conditions
def window_cost : ℕ := 100
def free_windows (n : ℕ) : ℕ := (n / 10) * 2 -- 2 free windows per 10 bought
def alice_needs : ℕ := 9
def bob_needs : ℕ := 11

-- Calculate individual costs given their needs without the discount
def cost_alice := alice_needs * window_cost
def cost_bob := bob_needs * window_cost

-- Calculate the total cost if bought separately
def total_separate_cost := cost_alice + cost_bob

-- Calculate the cost when purchased together
def joint_needs := alice_needs + bob_needs
def cost_joint := (joint_needs - free_windows(joint_needs)) * window_cost

-- Calculate the savings when purchasing together
def savings := total_separate_cost - cost_joint

theorem savings_is_200 : savings = 200 := by
  sorry

end savings_is_200_l118_118278


namespace book_area_correct_l118_118947

/-- Converts inches to centimeters -/
def inch_to_cm (inches : ℚ) : ℚ :=
  inches * 2.54

/-- The length of the book given a parameter x -/
def book_length (x : ℚ) : ℚ :=
  3 * x - 4

/-- The width of the book in inches -/
def book_width_in_inches : ℚ :=
  5 / 2

/-- The width of the book in centimeters -/
def book_width : ℚ :=
  inch_to_cm book_width_in_inches

/-- The area of the book given a parameter x -/
def book_area (x : ℚ) : ℚ :=
  book_length x * book_width

/-- Proof that the area of the book with x = 5 is 69.85 cm² -/
theorem book_area_correct : book_area 5 = 69.85 := by
  sorry

end book_area_correct_l118_118947


namespace total_cookies_eaten_l118_118562

theorem total_cookies_eaten :
  let charlie := 15
  let father := 10
  let mother := 5
  let grandmother := 12 / 2
  let dog := 3 * 0.75
  charlie + father + mother + grandmother + dog = 38.25 :=
by
  sorry

end total_cookies_eaten_l118_118562


namespace john_average_speed_l118_118464

noncomputable def average_speed (distance1 speed1 distance2 speed2 : ℝ) : ℝ :=
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  total_distance / total_time

theorem john_average_speed :
  average_speed 50 15 25 45 ≈ 19.29 :=
by
  sorry

end john_average_speed_l118_118464


namespace A_times_B_correct_l118_118877

noncomputable def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {y | y > 1}
noncomputable def A_times_B : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem A_times_B_correct : A_times_B = {x | (0 ≤ x ∧ x ≤ 1) ∨ x > 2} := 
sorry

end A_times_B_correct_l118_118877


namespace part1_part2_increasing_part2_decreasing_part3_l118_118753

-- Define the function f
def f (x : ℝ) (k : ℝ) : ℝ := (Real.log x + k) / Real.exp x

-- Define the derivative of f
def f' (x : ℝ) (k : ℝ) : ℝ := (1 / x - Real.log x - k) / Real.exp x

-- Define g(x) = x * f'(x)
def g (x : ℝ) (k : ℝ) : ℝ := x * f' x k

-- Prove Part (Ⅰ)
theorem part1 {k : ℝ} (h : f' 1 k = 0) : k = 1 :=
by 
  sorry

-- Prove Part (Ⅱ)
theorem part2_increasing : ∀ x : ℝ, 0 < x ∧ x < 1 → f' x 1 > 0 :=
by 
  sorry

theorem part2_decreasing : ∀ x : ℝ, 1 < x → f' x 1 < 0 :=
by 
  sorry

-- Prove Part (Ⅲ)
theorem part3 : ∀ x : ℝ, 0 < x → g x 1 < 1 + Real.exp (-2) :=
by 
  sorry

end part1_part2_increasing_part2_decreasing_part3_l118_118753


namespace cosine_sum_eq_neg_half_l118_118141

theorem cosine_sum_eq_neg_half (n : ℕ) : 
  (∑ k in Finset.range n, cos ((2 * (k+1) * Real.pi) / (2 * n + 1))) = -1/2 :=
by
  sorry

end cosine_sum_eq_neg_half_l118_118141


namespace min_value_fraction_l118_118733

theorem min_value_fraction (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y ≤ 2) :
  (∃ t : ℝ, t = x - y ∧ t > 0 ∧ ∀ (z : ℝ), z = x - y → z = 4 * (sqrt 2) - 4 ∧ (2 / (x + 3 * y) + 1 / (x - y) ≥ (3 + 2 * sqrt 2) / 4)) :=
begin
  sorry
end

end min_value_fraction_l118_118733


namespace graph_fixed_point_l118_118943

theorem graph_fixed_point (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∃ (x y : ℝ), (x = 2) ∧ (y = 2) ∧ (y = a^(x-2) + 1) :=
by {
  use [2, 2],
  split,
  { refl },
  split,
  { refl },
  { sorry }
}

end graph_fixed_point_l118_118943


namespace arithmetic_sequence_sum_l118_118180

noncomputable def Sn (a d n : ℕ) : ℕ :=
n * a + (n * (n - 1) * d) / 2

theorem arithmetic_sequence_sum (a d : ℕ) (h1 : a = 3 * d) (h2 : Sn a d 5 = 50) : Sn a d 8 = 104 :=
by
/-
  From the given conditions:
  - \(a_4\) is the geometric mean of \(a_2\) and \(a_7\) implies \(a = 3d\)
  - Sum of first 5 terms is 50 implies \(S_5 = 50\)
  We need to prove \(S_8 = 104\)
-/
  sorry

end arithmetic_sequence_sum_l118_118180


namespace num_subsets_inclusive_l118_118954

open Set

theorem num_subsets_inclusive:
  {X : Set ℕ} → X ⊆ {1, 2, 3, 4, 5, 6} → {1, 2, 3} ⊆ X → 
  ({Y : Set ℕ // {1, 2, 3} ⊆ Y ∧ Y ⊆ {1, 2, 3, 4, 5, 6}}).finite.to_finset.card = 8
  := by 
  sorry

end num_subsets_inclusive_l118_118954


namespace find_725th_digit_l118_118227

noncomputable def decimal_expansion : ℕ → ℕ
| 0 := 2
| 1 := 4
| 2 := 1
| 3 := 3
| 4 := 7
| 5 := 9
| 6 := 3
| 7 := 1
| 8 := 0
| 9 := 3
| 10 := 4
| 11 := 4
| 12 := 8
| 13 := 2
| 14 := 7
| 15 := 5
| 16 := 8
| 17 := 6
| 18 := 2
| 19 := 0
| 20 := 6
| 21 := 8
| 22 := 9
| 23 := 5
| 24 := 5
| 25 := 1
| 26 := 7
| 27 := 8
| 28 := 6
| (n + 29) := decimal_expansion n

theorem find_725th_digit :
  decimal_expansion 20 = 8 :=
by
  sorry

end find_725th_digit_l118_118227


namespace altitude_inequality_l118_118962

-- Defining the problem parameters
variables {a b c ma mb mc : ℝ}
-- Setting conditions
variables (h_ma : ma > 0) (h_mb : mb > 0) (h_mc : mc > 0)
-- Proving the inequality
theorem altitude_inequality
  (h1 : ma * mb + mb * mc + mc * ma)
  (h2 : a^2 + b^2 + c^2) :
  ma * mb + mb * mc + mc * ma ≤ (3 / 4) * (a^2 + b^2 + c^2) :=
sorry

end altitude_inequality_l118_118962


namespace contradiction_divisible_by_2_l118_118209

open Nat

theorem contradiction_divisible_by_2 (a b : ℕ) (h : (a * b) % 2 = 0) : a % 2 = 0 ∨ b % 2 = 0 :=
by
  sorry

end contradiction_divisible_by_2_l118_118209


namespace hexagon_perimeter_l118_118842

-- Define the hexagon with given conditions
structure Hexagon :=
  (A B C D E F : ℝ)
  (angle_A angle_C angle_E : Real.Angle := Real.Angle.ofDegrees 60)
  (angle_B angle_D angle_F : Real.Angle := Real.Angle.ofDegrees 120)
  (sides_equal : A = B ∧ B = C ∧ C = D ∧ D = E ∧ E = F)
  (area : ℝ := 18)

-- Define the problem statement
theorem hexagon_perimeter (h : Hexagon) : 
  6 * (A h) = 12 * (real.root 4 (3 : ℝ)) :=
sorry

end hexagon_perimeter_l118_118842


namespace find_angle_P_l118_118064

-- Define the angles as real numbers
variables (P Q R : ℝ)

-- Given conditions
def angle_R := 18
def angle_Q := 3 * angle_R
def sum_of_angles_in_triangle := P + Q + R = 180

-- The theorem to prove
theorem find_angle_P (hR : R = angle_R) (hQ : Q = angle_Q) (hSum : sum_of_angles_in_triangle) : P = 108 :=
by 
  unfold angle_R at hR
  unfold angle_Q at hQ
  unfold sum_of_angles_in_triangle at hSum
  sorry

end find_angle_P_l118_118064


namespace winning_votes_l118_118237

variable (V : ℕ) -- total votes

-- Given conditions
def total_votes := V > 0
def winner_percentage := 0.62 * V
def loser_percentage := 0.38 * V
def winning_margin := winner_percentage - loser_percentage = 300

-- Goal to prove
theorem winning_votes :
  total_votes →
  winner_percentage →
  loser_percentage →
  winning_margin →
  0.62 * V = 775 :=
sorry

end winning_votes_l118_118237


namespace categorize_numbers_l118_118685
-- Lean 4 statement for the proof problem


def numbers : List Float := [-4, -abs (-4 / 3), 0, 22 / 7, -3.14, 2006, -(+5), +1.88]

def positive_set : Set Float := {22 / 7, 2006, 1.88}
def negative_set : Set Float := {-4, -4 / 3, -3.14, -5}
def non_negative_integers_set : Set Float := {0, 2006}
def fractions_set : Set Float := {-4 / 3, 22 / 7, -3.14, 1.88}

theorem categorize_numbers:
  (positive_set = {x | x ∈ numbers ∧ x > 0}) ∧
  (negative_set = {x | x ∈ numbers ∧ x < 0}) ∧
  (non_negative_integers_set = {x | x ∈ numbers ∧ x ≥ 0 ∧ ∃ z : ℕ, x = z}) ∧
  (fractions_set = {x | x ∈ numbers ∧ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)}) :=
by {
  sorry
}

end categorize_numbers_l118_118685


namespace solve_tan_theta₂_l118_118077

section ProofProblem

variables {P A B C O M : Type} 
variables (α : Set (Set P)) (θ₁ θ₂ : Real)

-- Definitions of the given conditions
def is_triangular_prism (P A B C : Type) : Prop := 
sorry

def orthogonal_projection_to_circumcenter 
(P A B C O : Type) : Prop := 
sorry

def midpoint_is_M (P O M : Type) : Prop := 
sorry

def cross_section_AM_parallel_BC 
(α AM BC : Set (Set P)) : Prop := 
sorry

def angle_PAM_as_θ₁
(P A M : Type) (θ₁ : Real) : Prop := 
sorry

def acute_dihedral_angle_between_planes
(α ABC : Set (Set P)) (θ₂ : Real) : Prop := 
sorry

-- The theorem we want to prove
theorem solve_tan_theta₂
(P A B C O M : Type)
(α : Set (Set P)) (θ₁ θ₂ : Real) 
(h₁ : is_triangular_prism P A B C)
(h₂ : orthogonal_projection_to_circumcenter P A B C O)
(h₃ : midpoint_is_M P O M)
(h₄ : cross_section_AM_parallel_BC α A M)
(h₅ : angle_PAM_as_θ₁ P A M θ₁)
(h₆ : acute_dihedral_angle_between_planes α (Set B) θ₂) 
(maximize_θ₁ : ∀ θ₁, θ₁ ≤ θ₁.max)
: tan θ₂ = sqrt 2 / 2 :=
sorry

end ProofProblem

end solve_tan_theta₂_l118_118077


namespace mary_euros_more_than_total_l118_118908

theorem mary_euros_more_than_total {USD_eur : ℝ} (hUSD_eur : USD_eur = 0.85) :
  let Michelle_USD := 30
  let Alice_USD := 18
  let Marco_eur := 24
  let Mary_initial_eur := 15

  -- After Marco gives half to Mary
  let Marco_final_eur := Marco_eur / 2
  let Mary_after_transfer := Mary_initial_eur + Marco_final_eur

  -- Mary spends 5 euros
  let Mary_final_eur := Mary_after_transfer - 5

  -- Michelle gives Alice 40% of her money
  let Alice_after_transfer_USD := Alice_USD + 0.4 * Michelle_USD

  -- Alice converts $10 to euros
  let Alice_eur := 10 * USD_eur

  -- Total euros for Marco and Alice
  let total_euros := Marco_final_eur + Alice_eur
    
  -- Mary has more euros compared to the total euros of Marco and Alice
  Mary_final_eur - total_euros = 1.5 :=
by
  -- Definitions
  let Michelle_USD := 30
  let Alice_USD := 18
  let Marco_eur := 24
  let Mary_initial_eur := 15

  let Marco_final_eur := Marco_eur / 2
  let Mary_after_transfer := Mary_initial_eur + Marco_final_eur

  let Mary_final_eur := Mary_after_transfer - 5

  let Alice_after_transfer_USD := Alice_USD + 0.4 * Michelle_USD

  let Alice_eur := 10 * USD_eur

  let total_euros := Marco_final_eur + Alice_eur

  -- Goal
  have : Mary_final_eur - total_euros = 1.5 := sorry
  assumption 

end mary_euros_more_than_total_l118_118908


namespace minimum_sum_distances_l118_118371

variables {A B P : Point} {l : Line} (hx1 : A = ⟨2, 0⟩) (hx2 : B = ⟨-2, -4⟩) (hl : l = {x | x 1 - 2 * x 2 + 8 = 0})

theorem minimum_sum_distances :
  (∃ P : Point, (P ∈ l ∧ (distance P A + distance P B) = 12)) :=
by
  sorry

end minimum_sum_distances_l118_118371


namespace find_pairs_l118_118469

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, -Real.sqrt 5], [2 * Real.sqrt 5, -3]]

def conditions (m n : ℤ) : Prop :=
  n ≥ 1 ∧ abs m ≤ n

def all_entries_integer (n : ℕ) (m : ℤ) : Prop :=
  ∀ i j, ∃ k : ℤ, (matrix.iterate (λ M, matrix_A * M) n 1 i j - (m + n^2) • matrix_A 1 i j : ℝ) = k

theorem find_pairs :
  ∃ (m n : ℤ), conditions m n ∧ all_entries_integer (n.to_nat) m ∧ ((m = 0 ∧ n = 1) ∨ (m = -6 ∧ n = 7)) :=
  sorry

end find_pairs_l118_118469


namespace area_between_two_chords_is_correct_l118_118199

noncomputable def area_between_chords 
  (r: ℝ) (d: ℝ) (chord_dist: ℝ) 
  (h_radius: r = 8) 
  (h_chord_dist: chord_dist = 8)
  : ℝ :=
  32 * Real.sqrt 3 + (64 * Real.pi / 3)

theorem area_between_two_chords_is_correct :
  area_between_chords 8 8 8 = 32 * Real.sqrt 3 + (64 * Real.pi / 3) :=
sorry

end area_between_two_chords_is_correct_l118_118199


namespace proportional_segments_in_equilateral_triangle_l118_118470

def equilateral_triangle (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def on_segment (P : Type*) (A B : Type*) [metric_space P] [metric_space A] [metric_space B] : Prop :=
  dist A B = dist A P + dist P B

theorem proportional_segments_in_equilateral_triangle 
  {A B C A1 A2 B1 B2 C1 C2 : Type*} [metric_space A] [metric_space B] [metric_space C]
  [metric_space A1] [metric_space A2] [metric_space B1] [metric_space B2] [metric_space C1] [metric_space C2]
  (h_equilateral : equilateral_triangle A B C)
  (h_A1_A2_on_BC : on_segment A1 B C) (h_B1_B2_on_CA : on_segment B1 C A) (h_C1_C2_on_AB : on_segment C1 A B)
  (h_eq_segments : dist A1 A2 = dist B1 B2 ∧ dist B1 B2 = dist C1 C2) :
  are_proportional (dist B2 C1) (dist C2 A1) ∧ are_proportional (dist C2 A1) (dist A2 B1) :=
sorry

end proportional_segments_in_equilateral_triangle_l118_118470


namespace car_catches_up_truck_l118_118645

-- Definitions of the problem's conditions
def truck_speed : ℝ := 45
def car_speed : ℝ := 60
def start_delay : ℝ := 1

-- Define the function to represent the distance difference
def distance_difference (t : ℝ) : ℝ :=
  car_speed * t - (truck_speed * (t + start_delay) + truck_speed * start_delay)

-- Prove that the distance difference is zero
theorem car_catches_up_truck : ∃ t, distance_difference t = 0 :=
  ∃ (t = 6), sorry

end car_catches_up_truck_l118_118645


namespace AC_eq_200_l118_118646

theorem AC_eq_200 (A B C : ℕ) (h1 : A + B + C = 500) (h2 : B + C = 330) (h3 : C = 30) : A + C = 200 := by
  sorry

end AC_eq_200_l118_118646


namespace triangle_MOI_area_is_3_4_l118_118054

variables {A B C M O I : Type}
variables (AB AC BC : ℝ) (I O M : Type)
variables [Euclidean_space O] [Euclidean_space I] [Euclidean_space M]
variables
  (coords_A : O::coord)
  (coords_B : O::coord)
  (coords_C : O::coord)
  (coords_I : I::coord)
  (coords_O : O::coord)
  (coords_M : M::coord)

-- Definitions based on the problem
def pts
:= [coords_A, coords_B, coords_C] : list 𝕂

def circumcenter
:= coords_O = ((C - A) / 2)

def incenter
:= coords_I = ((B-C) / 2)

def MOI_area
:= (1 / 2) abs ( MOI ((O - I) - (O - M)) + ((M + (I - O))) - ((I - M)))

-- Statement of the theorem
theorem triangle_MOI_area_is_3_4 :=
  begin
    let AB := 15, AC := 8, BC := 17,
    let coords_A := (0,0),
    let coords_B := (8,0),
    let coords_C := (0,17),
    let coords_I := (3.4,3.4),
    let coords_O := (4,8.5),
    let coords_M := (5,5),
    -- Prove the area is 3.4
    sorry
  end

end triangle_MOI_area_is_3_4_l118_118054


namespace area_of_overlapping_squares_l118_118190

/-- The area covered by three identical square pieces with overlapping centers
    and a total perimeter of 48 cm -/
theorem area_of_overlapping_squares
    (n : ℕ)
    (side_length : ℝ)
    (perimeter : ℝ)
    (h1 : n = 3)
    (h2 : perimeter = 48)
    (h3 : perimeter = n * 4 * side_length)
    : (side_length ^ 2 * n) - (side_length ^ 2 : ℝ / 2) = 90 := sorry

end area_of_overlapping_squares_l118_118190


namespace hyperbola_asymptote_perpendicular_to_line_l118_118846

variable {a : ℝ}

theorem hyperbola_asymptote_perpendicular_to_line (h : a > 0)
  (C : ∀ x y : ℝ, x^2 / a^2 - y^2 = 1)
  (l : ∀ x y : ℝ, 2 * x - y + 1 = 0) :
  a = 2 :=
by
  sorry

end hyperbola_asymptote_perpendicular_to_line_l118_118846


namespace logarithms_proven_l118_118800

open Real

noncomputable def logarithms_proof (y : ℝ) : Prop :=
  log 5 (log 2 (log 3 y)) = 2 → y^(1/3) = 3^(2^25 / 3)

-- Prove the statement using Lean
theorem logarithms_proven (y : ℝ) (h : log 5 (log 2 (log 3 y)) = 2) : y^(1/3) = 3^(2^25 / 3) :=
  sorry

end logarithms_proven_l118_118800


namespace snake_body_length_l118_118640

theorem snake_body_length (L : ℝ) (H : ℝ) (h1 : H = L / 10) (h2 : L = 10) : L - H = 9 :=
by
  sorry

end snake_body_length_l118_118640


namespace special_prize_winner_l118_118921

def students : Type := {1, 2, 3, 4, 5, 6}

def prediction_A (s : students) : Prop := s = 1 ∨ s = 2
def prediction_B (s : students) : Prop := s ≠ 3
def prediction_C (s : students) : Prop := s ≠ 4 ∧ s ≠ 5 ∧ s ≠ 6
def prediction_D (s : students) : Prop := s = 4 ∨ s = 5 ∨ s = 6

def exactly_one_correct (s : students) : Prop :=
  let A := prediction_A s
  let B := prediction_B s
  let C := prediction_C s
  let D := prediction_D s
  (A ∧ ¬ B ∧ ¬ C ∧ ¬ D) ∨
  (¬ A ∧ B ∧ ¬ C ∧ ¬ D) ∨
  (¬ A ∧ ¬ B ∧ C ∧ ¬ D) ∨
  (¬ A ∧ ¬ B ∧ ¬ C ∧ D)

theorem special_prize_winner : ∃ s : students, s = 3 ∧ exactly_one_correct s :=
sorry

end special_prize_winner_l118_118921


namespace dark_more_than_light_l118_118642

-- Define the board size
def board_size : ℕ := 9

-- Define the number of dark squares in odd rows
def dark_in_odd_row : ℕ := 5

-- Define the number of light squares in odd rows
def light_in_odd_row : ℕ := 4

-- Define the number of dark squares in even rows
def dark_in_even_row : ℕ := 4

-- Define the number of light squares in even rows
def light_in_even_row : ℕ := 5

-- Calculate the total number of dark squares
def total_dark_squares : ℕ := (dark_in_odd_row * ((board_size + 1) / 2)) + (dark_in_even_row * (board_size / 2))

-- Calculate the total number of light squares
def total_light_squares : ℕ := (light_in_odd_row * ((board_size + 1) / 2)) + (light_in_even_row * (board_size / 2))

-- Define the main theorem
theorem dark_more_than_light : total_dark_squares - total_light_squares = 1 := by
  sorry

end dark_more_than_light_l118_118642


namespace profit_percent_l118_118595

-- Definitions of cost price and selling price
variable (SP : ℝ) (CP : ℝ)

-- Given condition: CP is 82% of SP
axiom cost_price_condition (h : CP = 0.82 * SP) : True

-- Statement to prove the profit percent
theorem profit_percent (h : CP = 0.82 * SP) : ((SP - CP) / CP) * 100 ≈ 21.95 :=
by
  sorry

end profit_percent_l118_118595


namespace period_increasing_intervals_range_g_l118_118719

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - Real.sin (2 * x)

theorem period_increasing_intervals :
    (∀ x : ℝ, f (x + π) = f x) ∧ 
    (∀ k : ℤ, ∀ x : ℝ, (k * π - 5 * π / 8 ≤ x ∧ x ≤ k * π - π / 8) → 
                    (f'' x > 0)) := 
sorry

noncomputable def g (x : ℝ) : ℝ := -sqrt 2 * Real.sin (2 * x)

theorem range_g :
    (∀ x : ℝ, (-π / 6 ≤ x ∧ x ≤ π / 3) → 
              (g x ≥ -sqrt 2 ∧ g x ≤ sqrt 6 / 2)) := 
sorry

end period_increasing_intervals_range_g_l118_118719


namespace shaded_area_correct_l118_118442

-- Definition of the problem conditions
def radius_larger_circle : ℝ := 10
def radius_smaller_circle : ℝ := 5

-- Define areas of the circles
def area_circle (r : ℝ) : ℝ := π * r^2
def area_larger_circle := area_circle radius_larger_circle
def area_smaller_circle := area_circle radius_smaller_circle

-- Total area of the two smaller circles
def total_area_smaller_circles := 2 * area_smaller_circle

-- Shaded area between the larger circle and the two smaller circles
def shaded_area := area_larger_circle - total_area_smaller_circles

-- Proof statement (to be proved)
theorem shaded_area_correct : shaded_area = 50 * π := by
  sorry

end shaded_area_correct_l118_118442


namespace decreasing_interval_f_ratio_a_b_l118_118402

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + Real.cos (π / 3 - 2 * x)

theorem decreasing_interval_f :
  ∃ a b, a = π / 3 ∧ b = 5 * π / 6 ∧ ∀ x, x ∈ Icc a b → ∃ c, f c > f x :=
sorry

variables {A B C a b c : ℝ}
variable (m : ℝ × ℝ := (1, 2)) (n : ℝ × ℝ := (Real.sin B, Real.sin C))

theorem ratio_a_b :
  f A = 2 →
  m = (1, 2) →
  n = (Real.sin B, Real.sin C) →
  m.1 * n.2 - m.2 * n.1 = 0 →
  c = 2 * b →
  Real.cos A = 1 / 2 →
  a / b = Real.sqrt 3 :=
sorry

end decreasing_interval_f_ratio_a_b_l118_118402


namespace larger_number_l118_118555

theorem larger_number (a b : ℤ) (h1 : a - b = 5) (h2 : a + b = 37) : a = 21 :=
sorry

end larger_number_l118_118555


namespace min_positive_period_f_l118_118160

/-- Define the function f(x) as given in the conditions -/
def f (x : ℝ) : ℝ := 2 * sin x * cos x * cos (2 * x)

/-- Prove that the minimum positive period of f(x) is π/2 -/
theorem min_positive_period_f : ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = π / 2 :=
sorry

end min_positive_period_f_l118_118160


namespace area_of_ADFE_l118_118763

namespace Geometry

open Classical

noncomputable def area_triangle (A B C : Type) [Field A] (area_DBF area_BFC area_FCE : A) : A :=
  let total_area := area_DBF + area_BFC + area_FCE
  let area := (105 : A) / 4
  total_area + area

theorem area_of_ADFE (A B C D E F : Type) [Field A] 
  (area_DBF : A) (area_BFC : A) (area_FCE : A) : 
  area_DBF = 4 → area_BFC = 6 → area_FCE = 5 → 
  area_triangle A B C area_DBF area_BFC area_FCE = (15 : A) + (105 : A) / 4 := 
by 
  intros 
  sorry

end area_of_ADFE_l118_118763


namespace calc_sqrt3ab_l118_118409

theorem calc_sqrt3ab (a b : ℝ) :
  (∃ M : ℝ × ℝ, M = (sqrt 3, -1)) →
  (∃ A B : ℝ × ℝ, let O : ℝ × ℝ := (0,0) in 
    let d := 4 in 
    let circle_O := ∃ (x y : ℝ), x^2 + y^2 = d ∧ 
      ((A.1 * A.1 + A.2 * A.2 = d) ∧ 
      (B.1 * B.1 + B.2 * B.2 = d) ∧  
      (a * A.1 + A.2 + b = 0) ∧ 
      (a * B.1 + B.2 + b = 0)) ∧ 
    let l := ax + y + b = 0 in 
    let OM := sqrt(3)*O.1 - O.2 in 
      (l intersects circle_O at A and B) ∧ 
      (OM is a vector) ∧ 
      (O, A, B are collinear) ∧  
      (O is the center of the circle)) →
  abs((a^2 + 1)^(-1/2) * b) = 2/3 →
  a = -sqrt 3 →
  b > O →
  sqrt(3) * a * b = -4 :=
begin
  sorry
end

end calc_sqrt3ab_l118_118409


namespace kolya_correct_valya_incorrect_l118_118305

variable (x : ℝ)

def p : ℝ := 1 / x
def r : ℝ := 1 / (x + 1)
def q : ℝ := (x - 1) / x
def s : ℝ := x / (x + 1)

theorem kolya_correct : r / (1 - s * q) = 1 / 2 := by
  sorry

theorem valya_incorrect : (q * r + p * r) / (1 - s * q) = 1 / 2 := by
  sorry

end kolya_correct_valya_incorrect_l118_118305


namespace problem_a2_b_c_in_M_l118_118328

def P : Set ℤ := {x | ∃ k : ℤ, x = 3 * k + 1}
def Q : Set ℤ := {x | ∃ k : ℤ, x = 3 * k - 1}
def M : Set ℤ := {x | ∃ k : ℤ, x = 3 * k}

theorem problem_a2_b_c_in_M (a b c : ℤ) (ha : a ∈ P) (hb : b ∈ Q) (hc : c ∈ M) : 
  a^2 + b - c ∈ M :=
sorry

end problem_a2_b_c_in_M_l118_118328


namespace equal_incircle_radii_of_original_triangles_l118_118203
-- Import the necessary library

-- Define the properties and conditions of the problem
variables {α : Type*} [linear_ordered_field α]
variables (A B C D E F : α) -- Representing the vertices of the hexagon
variables (r : α) -- Radius of the incircle of the six small triangles
variables (R_ACE R_BDF : α) -- Radius of the incircle of the original triangles

-- Define the condition stating that intersected triangles form a common hexagon and have equal inradii for small triangles.
def intersected_triangle_hexagon (A B C D E F : α) : Prop := 
  ∃ (r : α), 
    (∃ (R_ACE R_BDF : α), 
      (∀ (P Q R : α), 
        incircle_radius (triangle_of_points P Q R) = r) ∧
      (R_ACE = radius_of_incircle (triangle_of_points A C E)) ∧
      (R_BDF = radius_of_incircle (triangle_of_points B D F)))

-- State the problem as a theorem to be proved
theorem equal_incircle_radii_of_original_triangles
  (A B C D E F : α)
  (h : intersected_triangle_hexagon A B C D E F) :
  R_ACE = R_BDF :=
sorry

end equal_incircle_radii_of_original_triangles_l118_118203


namespace point_in_fourth_quadrant_l118_118372

open Complex

-- Define the given complex number Z
def Z : ℂ := 2 * I / (1 + I)

-- Define the conjugate of Z
def Z_conjugate : ℂ := conj Z

-- The point corresponding to Z_conjugate in the complex plane
def point_conjugate : ℝ × ℝ := (Z_conjugate.re, Z_conjugate.im)

-- Prove that the point corresponding to the conjugate of Z is in the fourth quadrant
theorem point_in_fourth_quadrant : point_conjugate.1 > 0 ∧ point_conjugate.2 < 0 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end point_in_fourth_quadrant_l118_118372


namespace twenty_two_percent_of_three_hundred_l118_118565

theorem twenty_two_percent_of_three_hundred : 
  (22 / 100) * 300 = 66 :=
by
  sorry

end twenty_two_percent_of_three_hundred_l118_118565


namespace possible_values_of_n_for_dividing_convex_circumscribed_ngon_l118_118611

-- Definitions for n > 3, convex n-gon, equal triangles, non-intersecting diagonals, and circumscribed polygon.
def convex_ngon (n : ℕ) (ngon : polygon) : Prop :=
  n > 3 ∧ ngon.isConvex ∧ ngon.isNGon n

def divided_into_equal_triangles_by_non_intersecting_diagonals (ngon : polygon) (triangles : list triangle) : Prop :=
  ∀ (t : triangle), t ∈ triangles → (t.area = (ngon.area / (ngon.numSides - 2))) ∧ (∀ (d1 d2 : line), d1 ∈ t.diagonals ∧ d2 ∈ t.diagonals → d1 ≠ d2 → d1 ∩ d2 = ∅)

def is_circumscribed (ngon : polygon) : Prop :=
  ∃ (circumcircle : circle), (∀ (v : vertex), v ∈ ngon.vertices → v ∈ circumcircle.points)

-- Main proof statement
theorem possible_values_of_n_for_dividing_convex_circumscribed_ngon :
  ∀ (n : ℕ) (ngon : polygon) (triangles : list triangle),
    convex_ngon n ngon ∧ divided_into_equal_triangles_by_non_intersecting_diagonals ngon triangles ∧ is_circumscribed ngon →
    n = 4 :=
by sorry

end possible_values_of_n_for_dividing_convex_circumscribed_ngon_l118_118611


namespace cube_modulo_values_l118_118669

-- Define the main theorem to be proven.
theorem cube_modulo_values : ∀ (a : ℕ) (n : ℕ),
  (a = 2 → (2^3 % n = (if n = 7 then 1 else if n = 11 then 8 else if n = 13 then 8 else if n = 17 then 8 else sorry)))
  ∧ (a = 3 → (3^3 % n = (if n = 7 then 6 else if n = 11 then 5 else if n = 13 then 1 else if n = 17 then 10 else sorry)))
  ∧ (a = 4 → (4^3 % n = (if n = 7 then 1 else if n = 11 then 9 else if n = 13 then 12 else if n = 17 then 13 else sorry))) :=
begin
  sorry -- This is the part where the proof would typically go but is omitted as instructed.
end

end cube_modulo_values_l118_118669


namespace tan_problem_l118_118703

noncomputable def problem : ℝ :=
  (Real.tan (20 * Real.pi / 180) + Real.tan (40 * Real.pi / 180) + Real.tan (120 * Real.pi / 180)) / 
  (Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180))

theorem tan_problem : problem = -Real.sqrt 3 := by
  sorry

end tan_problem_l118_118703


namespace perimeter_after_growth_operations_perimeter_after_four_growth_operations_l118_118364

theorem perimeter_after_growth_operations (initial_perimeter : ℝ) (growth_factor : ℝ) (growth_steps : ℕ):
  initial_perimeter = 27 ∧ growth_factor = 4/3 ∧ growth_steps = 2 → 
    initial_perimeter * growth_factor^growth_steps = 48 :=
by
  sorry

theorem perimeter_after_four_growth_operations (initial_perimeter : ℝ) (growth_factor : ℝ) (growth_steps : ℕ):
  initial_perimeter = 27 ∧ growth_factor = 4/3 ∧ growth_steps = 4 → 
    initial_perimeter * growth_factor^growth_steps = 256/3 :=
by
  sorry

end perimeter_after_growth_operations_perimeter_after_four_growth_operations_l118_118364


namespace rearrangement_count_is_two_l118_118422

def is_adjacent (c1 c2 : Char) : Bool :=
  (c1 = 'a' ∧ c2 = 'b') ∨
  (c1 = 'b' ∧ c2 = 'c') ∨
  (c1 = 'c' ∧ c2 = 'd') ∨
  (c1 = 'd' ∧ c2 = 'e') ∨
  (c1 = 'b' ∧ c2 = 'a') ∨
  (c1 = 'c' ∧ c2 = 'b') ∨
  (c1 = 'd' ∧ c2 = 'c') ∨
  (c1 = 'e' ∧ c2 = 'd')

def no_adjacent_letters (s : List Char) : Bool :=
  match s with
  | [] => true
  | [_] => true
  | c1 :: c2 :: cs => 
    ¬ is_adjacent c1 c2 ∧ no_adjacent_letters (c2 :: cs)

def valid_rearrangements_count : Nat :=
  let perms := List.permutations ['a', 'b', 'c', 'd', 'e']
  perms.filter no_adjacent_letters |>.length

theorem rearrangement_count_is_two :
  valid_rearrangements_count = 2 :=
by sorry

end rearrangement_count_is_two_l118_118422


namespace find_cows_l118_118060

theorem find_cows :
  ∃ (D C : ℕ), (2 * D + 4 * C = 2 * (D + C) + 30) → C = 15 := 
sorry

end find_cows_l118_118060


namespace determine_linear_function_l118_118427

variables {α β : Type}
variables [LinearOrderedField α] [LinearOrderedField β]

noncomputable def f : α → α := sorry

theorem determine_linear_function (h_lin : ∃ a b : α, f = λ x, a * x + b ∧ a ≠ 0)
  (h_composition : ∀ x : α, f (f x) = x - 2) :
  f = λ x, x - 1 :=
by sorry

end determine_linear_function_l118_118427


namespace log_power_comparison_l118_118717

theorem log_power_comparison :
  let a := Real.log 0.3 / Real.log 3
  let b := 3 ^ 0.3
  let c := 0.3 ^ 0.3 in
  a < c ∧ c < b :=
by
  sorry

end log_power_comparison_l118_118717


namespace crates_of_mangoes_sold_l118_118865

def total_crates_sold := 50
def crates_grapes_sold := 13
def crates_passion_fruits_sold := 17

theorem crates_of_mangoes_sold : 
  (total_crates_sold - (crates_grapes_sold + crates_passion_fruits_sold) = 20) :=
by 
  sorry

end crates_of_mangoes_sold_l118_118865


namespace sum_c_d_eq_neg3_l118_118892

theorem sum_c_d_eq_neg3 (c d : ℤ) (h : ∀ x : ℝ, x - 2 = 0 ∨ x + 3 = 0 → x^2 + c * x + d = 0) : c + d = -3 :=
by
  have h1 := h 2 (or.inl rfl)
  have h2 := h (-3) (or.inr rfl)
  have eq1 : 4 + 2 * c + d = 0 := by simp [h1]
  have eq2 : 9 - 3 * c + d = 0 := by simp [h2]
  have eq_comb := eq_add_of_eq_eq_sub eq1 eq2
  simp at eq_comb
  cases eq_comb with hc hd
  exact hd

end sum_c_d_eq_neg3_l118_118892


namespace exists_m_with_prime_factors_diff_l118_118114

def σ (n : ℕ) : ℕ := ∑ i in (n.divisors : Finset ℕ), i

def D (n : ℕ) : Finset ℕ := n.factorization.support.toFinset

theorem exists_m_with_prime_factors_diff (k : ℕ) : ∃ m : ℕ, m > 0 ∧ (D (σ m) \ D m).card = k := 
  sorry

end exists_m_with_prime_factors_diff_l118_118114


namespace unit_digit_3_pow_58_l118_118220

theorem unit_digit_3_pow_58 : (3 ^ 58) % 10 = 9 := by
  -- Conditions based on the cycle of the unit digits of powers of 3
  have h1 : (3 ^ 1) % 10 = 3 := by norm_num
  have h2 : (3 ^ 2) % 10 = 9 := by norm_num
  have h3 : (3 ^ 3) % 10 = 7 := by norm_num
  have h4 : (3 ^ 4) % 10 = 1 := by norm_num
  -- Since the pattern repeats every 4 steps:
  have cycle : ∀ n, (3 ^ (4 * n + 1)) % 10 = 3 := by sorry
  have cycle : ∀ n, (3 ^ (4 * n + 2)) % 10 = 9 := by sorry
  have cycle : ∀ n, (3 ^ (4 * n + 3)) % 10 = 7 := by sorry
  have cycle : ∀ n, (3 ^ (4 * n + 4)) % 10 = 1 := by sorry
  -- Now, finding 3^58
  have : 58 = 4 * 14 + 2 := by norm_num
  -- Therefore, (3 ^ 58) % 10 = (3 ^ (4 * 14 + 2)) % 10 = 9
  show (3 ^ 58) % 10 = 9 from cycle 14

end unit_digit_3_pow_58_l118_118220


namespace snake_body_length_l118_118638

theorem snake_body_length (l h : ℝ) (h_head: h = l / 10) (h_length: l = 10) : l - h = 9 := 
by 
  rw [h_length, h_head] 
  norm_num
  sorry

end snake_body_length_l118_118638


namespace smallest_n_value_l118_118031

def A : Set ℕ := {x | ∃ n : ℕ, n > 0 ∧ x = 2 * n - 1}
def B : Set ℕ := {x | ∃ n : ℕ, n > 0 ∧ x = 2^n}

def sequence (A B : Set ℕ) : ℕ → ℕ
| 0 => 0 -- assume a₀ is 0
| (n+1) => (A ∪ B).min (n+1)  -- Definition passed as recursive sequence

-- The sum of the first n elements of the sequence
def S : ℕ → ℕ
| 0 => 0
| (n+1) => S n + sequence A B n

theorem smallest_n_value (n : ℕ) :
  (∀ k, k < 27 → S k ≤ 12 * sequence A B (k + 1)) ∧ S 27 > 12 * sequence A B 28 :=
sorry

end smallest_n_value_l118_118031


namespace sqrt_multiplication_l118_118579

theorem sqrt_multiplication : (real.sqrt 2) * (real.sqrt 3) = real.sqrt 6 :=
by
  sorry

end sqrt_multiplication_l118_118579


namespace triangle_range_values_l118_118720

def acute_triangle (Δ : Triangle) : Prop := 
  Δ.A < 90 ∧ Δ.B < 90 ∧ Δ.C < 90

def b_eq_2c {ℝ : Type*} [LinearOrderedField ℝ] (b c : ℝ) : Prop :=
  b = 2 * c

def trigonometric_identity {ℝ : Type*} [LinearOrderedField ℝ] (A B C : ℝ) [Real.sin B - Real.sin (A + B) = 2 * Real.sin C * Real.cos A ] : Prop :=
  Real.sin B - Real.sin (A + B) = 2 * Real.sin C * Real.cos A

theorem triangle_range_values (Δ : Triangle) (b c A B C : ℝ) [acute_triangle Δ] (hb : b_eq_2c b c):
  (trigonometric_identity A B C) →
  ((Real.cos B + Real.sin B) ^ 2) + Real.sin (2 * C) = 1 + Real.sqrt 3 / 2 :=
sorry

end triangle_range_values_l118_118720


namespace angle_XRS_eq_135_l118_118074

open Real

noncomputable def point := (ℝ × ℝ)

-- Define the square WXYZ with side length 6
def W : point := (0, 0)
def X : point := (6, 0)
def Y : point := (6, -6)
def Z : point := (0, -6)

-- Define that WX = WY = 6 (isosceles triangle conditions)
def isosceles_triangle : Prop := dist W X = 6 ∧ dist W Y = 6

-- Define the intersection point R of line segments XY and WZ
def R : point := (3, -3)

-- Define point S on line segment YZ such that RS is perpendicular to YZ
def S : point := (6, -3)
def RS_perpendicular_YZ : Prop := (S.1 = Y.1 ∧ S.2 = R.2) ∧ (R.1 - S.1) * (Y.1 - Z.1) + (R.2 - S.2) * (Y.2 - Z.2) = 0

-- The goal is to prove the measure of angle XRS is 135 degrees
theorem angle_XRS_eq_135 :
  isosceles_triangle →
  RS_perpendicular_YZ →
  angle (R.1 - X.1, R.2 - X.2) (S.1 - R.1, S.2 - R.2) = pi * 3 / 4 :=
by
  sorry

end angle_XRS_eq_135_l118_118074


namespace remaining_charge_time_135_l118_118131

def cell_initial_empty : Prop := 
  ∀ t : ℕ, t = 0 → charge t = 0

def charge_45_min_for_25_percent (charge : ℕ → ℕ) : Prop := 
  charge 45 = 25

theorem remaining_charge_time_135 (charge : ℕ → ℕ) (h1 : cell_initial_empty) (h2 : charge_45_min_for_25_percent charge) : 
  ∃ additional_time : ℕ, additional_time = 135 :=
sorry

end remaining_charge_time_135_l118_118131


namespace cube_letter_shaded_face_l118_118617

/-- Given a cube with different letters on each face:
    - In the first position: face T is adjacent to faces E and A.
    - In the second position: face X is adjacent to faces P and F.
    - In the third position: face E is adjacent to faces A and V.
   Prove that the letter on a specific shaded face of the cube is V. -/
theorem cube_letter_shaded_face 
  (T E A X P F V : Type) 
  (adj1 : T = E → A ∧ A ≠ T) 
  (adj2 : X = P → F ∧ F ≠ X) 
  (adj3 : E = A → V ∧ V ≠ E) :
  T = V :=
sorry

end cube_letter_shaded_face_l118_118617


namespace necessary_but_not_sufficient_l118_118624

variable (a : ℝ)

theorem necessary_but_not_sufficient (h : a ≥ 2) : (a = 2 ∨ a > 2) ∧ ¬(a > 2 → a ≥ 2) := by
  sorry

end necessary_but_not_sufficient_l118_118624


namespace convex_polygon_diagonals_30_sides_l118_118661

theorem convex_polygon_diagonals_30_sides :
  ∀ (n : ℕ), n = 30 → ∀ (sides : ℕ), sides = n →
  let total_segments := (n * (n - 1)) / 2 in
  let diagonals := total_segments - n in
  diagonals = 405 :=
by
  intro n hn sides hs
  simp only [hn, hs]
  let total_segments := (30 * 29) / 2
  have h_total_segments : total_segments = 435 := by sorry
  let diagonals := total_segments - 30
  have h_diagonals : diagonals = 405 := by sorry
  exact h_diagonals

end convex_polygon_diagonals_30_sides_l118_118661


namespace diagonals_from_vertex_l118_118810

theorem diagonals_from_vertex (sum_interior_angles : ℕ) (h : sum_interior_angles = 1800) : 
  let x := (sum_interior_angles / 180) + 2 in 
  x - 3 = 9 :=
by 
  let x := (sum_interior_angles / 180) + 2 
  have hx : x = 12 := by sorry
  show x - 3 = 9 from calc
    x - 3 = 12 - 3 : by rw hx
        ... = 9   : by norm_num

end diagonals_from_vertex_l118_118810


namespace A_squared_plus_B_squared_eq_one_l118_118888

theorem A_squared_plus_B_squared_eq_one
  (A B : ℝ) (h1 : A ≠ B)
  (h2 : ∀ x : ℝ, (A * (B * x ^ 2 + A) ^ 2 + B - (B * (A * x ^ 2 + B) ^ 2 + A)) = B ^ 2 - A ^ 2) :
  A ^ 2 + B ^ 2 = 1 :=
sorry

end A_squared_plus_B_squared_eq_one_l118_118888


namespace union_A_B_eq_A_l118_118006

def A : set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B : set ℝ := {-1, 0, 1, 2, 3}

theorem union_A_B_eq_A : A ∪ B = A :=
by
  -- Added proof to ensure the correctness
  sorry

end union_A_B_eq_A_l118_118006


namespace Kolya_is_correct_Valya_is_incorrect_l118_118289

noncomputable def Kolya_probability (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

noncomputable def Valya_probability_not_losing (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

theorem Kolya_is_correct (x : ℝ) (hx : x > 0) : Kolya_probability x = 1 / 2 :=
by
  sorry

theorem Valya_is_incorrect (x : ℝ) (hx : x > 0) : Valya_probability_not_losing x = 1 / 2 :=
by
  sorry

end Kolya_is_correct_Valya_is_incorrect_l118_118289


namespace unique_complex_root_l118_118331

theorem unique_complex_root (k : ℂ) : 
  (∃ x : ℂ, x ≠ 0 ∧ (x^2 / (x+1) + x^2 / (x+2) = k * x^2)) ↔ (k = 2 * complex.I ∨ k = -2 * complex.I) :=
by sorry

end unique_complex_root_l118_118331


namespace sum_of_coefficients_l118_118819

theorem sum_of_coefficients :
  ∃ a b c : ℕ, (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (sqrt 3 + 1 / sqrt 3 + sqrt 11 + 1 / sqrt 11 = (a * sqrt 3 + b * sqrt 11) / c) ∧
  (∀ d : ℕ, d ≠ 0 → (sqrt 3 + 1 / sqrt 3 + sqrt 11 + 1 / sqrt 11 = (a * sqrt 3 + b * sqrt 11) / d → c ≤ d)) ∧
  (a + b + c = 113) :=
sorry

end sum_of_coefficients_l118_118819


namespace paths_from_A_to_B_via_C_l118_118913

open Classical

-- Definitions based on conditions
variables (lattice : Type) [PartialOrder lattice]
variables (A B C : lattice)
variables (first_red first_blue second_red second_blue first_green second_green orange : lattice)

-- Conditions encoded as hypotheses
def direction_changes : Prop :=
  -- Arrow from first green to orange is now one way from orange to green
  ∀ x : lattice, x = first_green → orange < x ∧ ¬ (x < orange) ∧
  -- Additional stop at point C located directly after the first blue arrows
  (C < first_blue ∨ first_blue < C)

-- Now stating the proof problem
theorem paths_from_A_to_B_via_C :
  direction_changes lattice first_green orange first_blue C →
  -- Total number of paths from A to B via C is 12
  (2 + 2) * 3 * 1 = 12 :=
by
  sorry

end paths_from_A_to_B_via_C_l118_118913


namespace product_pass_rate_l118_118632

variable {a b : ℝ} (h_a : 0 ≤ a ∧ a ≤ 1) (h_b : 0 ≤ b ∧ b ≤ 1) (h_indep : true)

theorem product_pass_rate : (1 - a) * (1 - b) = 
((1 - a) * (1 - b)) :=
by
  sorry

end product_pass_rate_l118_118632


namespace number_of_solutions_l118_118358

noncomputable def count_solutions : ℕ :=
  Finset.card { n : ℕ | 2 ≤ n ∧ (n % 2 = 1) ∧ n ≤ 100 }

theorem number_of_solutions : count_solutions = 25 :=
  sorry

end number_of_solutions_l118_118358


namespace right_triangle_48_55_l118_118443

def right_triangle_properties (a b : ℕ) (ha : a = 48) (hb : b = 55) : Prop :=
  let area := 1 / 2 * a * b
  let hypotenuse := Real.sqrt (a ^ 2 + b ^ 2)
  area = 1320 ∧ hypotenuse = 73

theorem right_triangle_48_55 : right_triangle_properties 48 55 (by rfl) (by rfl) :=
  sorry

end right_triangle_48_55_l118_118443


namespace average_annual_growth_rate_l118_118177

-- Define the conditions
def revenue_current_year : ℝ := 280
def revenue_planned_two_years : ℝ := 403.2

-- Define the growth equation
def growth_equation (x : ℝ) : Prop :=
  revenue_current_year * (1 + x)^2 = revenue_planned_two_years

-- State the theorem
theorem average_annual_growth_rate : ∃ x : ℝ, growth_equation x ∧ x = 0.2 := by
  sorry

end average_annual_growth_rate_l118_118177


namespace eccentricity_of_ellipse_exists_k_for_BN_eq_2BM_l118_118110

-- Part I: Prove the eccentricity of the ellipse
theorem eccentricity_of_ellipse (a b c: ℝ) (h1: a > b ∧ b > 0) (h2: 2 * c^2 - 5 * a^2 * c + 2 * a^4 = 0) :
  c / a = Real.sqrt 2 / 2 :=
suffices e: c = a * (Real.sqrt 2 / 2), sorry,
(eq_div_iff (ne_of_gt h1.1)).mpr e.symm

-- Part II: Prove the existence of k ∈ [1/4, 1/2] such that |BN| = 2|BM|
theorem exists_k_for_BN_eq_2BM (a b c: ℝ) (h1: a^2 = b^2 + c^2) (h2: c = a * (Real.sqrt 2 / 2))
  (BM BN: ℝ → ℝ) (h3: ∀ k, BM k * BN (-1 / k) = 0) :
  ∃ k ∈ Icc (1/4: ℝ) (1/2: ℝ), BN k = 2 * BM k :=
sorry

end eccentricity_of_ellipse_exists_k_for_BN_eq_2BM_l118_118110


namespace leftover_coverage_l118_118679

variable (bagCoverage lawnLength lawnWidth bagsPurchased : ℕ)

def area_of_lawn (length width : ℕ) : ℕ :=
  length * width

def total_coverage (bagCoverage bags : ℕ) : ℕ :=
  bags * bagCoverage

theorem leftover_coverage :
  let lawnLength := 22
  let lawnWidth := 36
  let bagCoverage := 250
  let bagsPurchased := 4
  let lawnArea := area_of_lawn lawnLength lawnWidth
  let totalSeedCoverage := total_coverage bagCoverage bagsPurchased
  totalSeedCoverage - lawnArea = 208 := by
  sorry

end leftover_coverage_l118_118679


namespace lottery_game_probability_l118_118063

noncomputable def winning_ticket_probability : ℕ :=
if (∀ s : Finset ℕ, (s.card = 6) → (∀ (n ∈ s), n ∈ {1, 2, 4, 8, 16, 32}) → (∏ x in s, x | (2^15)))
then 1 else 0

theorem lottery_game_probability :
  winning_ticket_probability = 1 :=
sorry

end lottery_game_probability_l118_118063


namespace sum_of_nu_lcm_15_eq_90_l118_118572

open Nat

theorem sum_of_nu_lcm_15_eq_90 :
  ∑ ν in (Finset.filter (λ ν : ℕ => lcm ν 15 = 90) (Finset.range 91)), ν = 12 :=
by
  sorry

end sum_of_nu_lcm_15_eq_90_l118_118572


namespace three_digit_multiples_of_30_not_75_l118_118776

theorem three_digit_multiples_of_30_not_75 : 
  let multiples_of_30 := list.range' 120 (990 - 120 + 30) 30
  let multiples_of_75 := list.range' 150 (900 - 150 + 150) 150
  let non_multiples_of_75 := multiples_of_30.filter (λ x => x % 75 ≠ 0)
  non_multiples_of_75.length = 24 := 
by 
  sorry

end three_digit_multiples_of_30_not_75_l118_118776


namespace count_letters_with_dot_only_l118_118828

theorem count_letters_with_dot_only (A B C : ℕ) (h1 : A = 10)
                                      (h2 : B = 24)
                                      (h3 : C = 40)
                                      (h4 : A + B = 34) :
  ∃ x : ℕ, x + A + B = C ∧ x = 6 :=
by {
  use 6,
  split,
  { simp [h1, h2, h3, h4] },
  { refl }
}

end count_letters_with_dot_only_l118_118828


namespace sum_of_x_coordinates_above_line_eq_18_l118_118132

-- We define the points as given in the problem
def points : List (ℝ × ℝ) := [(2, 9), (5, 15), (10, 25), (15, 30), (18, 55)]

-- Define the line equation y = 2x + 5
def above_line (p : ℝ × ℝ) : Prop := p.2 > 2 * p.1 + 5

-- Define the sum of x-coordinates of points above the line
def sum_of_x_above_line : ℝ :=
  points.filter above_line |>.sum_by (λ p, p.1)

-- The theorem we need to prove
theorem sum_of_x_coordinates_above_line_eq_18 :
  sum_of_x_above_line = 18 :=
by
  sorry

end sum_of_x_coordinates_above_line_eq_18_l118_118132


namespace ratio_of_products_l118_118476

theorem ratio_of_products (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  ((a - c) * (b - d)) / ((a - b) * (c - d)) = -4 / 3 :=
by 
  sorry

end ratio_of_products_l118_118476


namespace total_amount_due_l118_118906

theorem total_amount_due 
  (price_A : ℝ) (quantity_A : ℕ) (price_B : ℝ) (quantity_B : ℕ) (price_online : ℝ) (quantity_online : ℕ) (discount : ℝ) :
  price_A = 15 → quantity_A = 8 →
  price_B = 12 → quantity_B = 12 →
  price_online = 16.99 → quantity_online = 5 →
  discount = 0.15 →
  ((quantity_A * price_A) + (quantity_B * price_B)) * (1 - discount) + (quantity_online * price_online) = 309.35 :=
begin
  intros h1A h2A h1B h2B h3_online h4_online h_discount,
  sorry,
end

end total_amount_due_l118_118906


namespace total_distance_correct_savings_correct_l118_118844

-- Define the data for distances and energy consumption
def daily_adjustments : List Int := [-8, -12, -16, 0, 22, 31, 33]
def base_distance_per_day : Int := 50
def num_days : Int := 7

def gasoline_consumption_per_100km : Float := 5.5
def gasoline_price_per_liter : Float := 8.2

def electric_consumption_per_100km : Float := 15
def electric_price_per_kwh : Float := 0.56

-- Define the total distance calculation
def total_distance : Int := base_distance_per_day * num_days + daily_adjustments.sum

-- Define the cost for gasoline car
def gasoline_car_cost : Float := (total_distance.toFloat / 100) * gasoline_consumption_per_100km * gasoline_price_per_liter

-- Define the cost for electric car
def electric_car_cost : Float := (total_distance.toFloat / 100) * electric_consumption_per_100km * electric_price_per_kwh

-- Define the savings calculation
def savings : Float := gasoline_car_cost - electric_car_cost

-- Theorem stating the total distance traveled
theorem total_distance_correct : total_distance = 400 := by
    sorry

-- Theorem stating the savings calculation is correct
theorem savings_correct : savings = 146.8 := by
    sorry

end total_distance_correct_savings_correct_l118_118844


namespace diff_function_gt_1_l118_118435

variable {R : Type} [Real R]

theorem diff_function_gt_1 (f : R → R) (h1 : ∀ x : R, differentiable (f x)) (h2 : ∀ x : R, (derivative (f x)) > 1) :
  f 3 > f 1 + 2 := by
  sorry

end diff_function_gt_1_l118_118435


namespace find_c_find_area_find_sin_2A_C_l118_118051

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 7
noncomputable def B : ℝ := Real.pi / 3 -- 60 degrees in radians

axiom cos_rule (c : ℝ) : c^2 - 2*c - 3 = 0
axiom sine_a_eq (sinA : ℝ) : (Real.sqrt 21) / 7 = sinA

theorem find_c : ∃ c : ℝ, c = 3 :=
by
  use 3
  exact cos_rule 3

theorem find_area : ∃ S : ℝ, S = (3 * Real.sqrt 3) / 2 :=
by
  use (3 * Real.sqrt 3) / 2
  sorry

theorem find_sin_2A_C :
  ∃ sin_2A_C : ℝ, sin_2A_C = (Real.sqrt 21) / 14 :=
by
  have sinA : ℝ := (Real.sqrt 21) / 7
  have cosA : ℝ := (2 * Real.sqrt 7) / 7
  use (Real.sqrt 21) / 14
  sorry

end find_c_find_area_find_sin_2A_C_l118_118051


namespace solution_set_of_inequality_l118_118021

noncomputable def f (x : ℝ) : ℝ := 2 * x + Real.sin x

theorem solution_set_of_inequality :
  {m : ℝ | f (m ^ 2) + f (2 * m - 3) < 0} = set.Ioo (-3 : ℝ) 1 :=
by
  -- The proof goes here
  sorry

end solution_set_of_inequality_l118_118021


namespace perpendicular_lines_probability_l118_118251

-- Definitions and Conditions
def square_vertices := {A B C D : Type} 

def total_lines := 6

def basic_events := 36

def perpendicular_basic_events := 10

-- Lean 4 statement
theorem perpendicular_lines_probability :
  (perpendicular_basic_events : ℚ) / basic_events = (5 : ℚ) / 18 :=
by sorry

end perpendicular_lines_probability_l118_118251


namespace necessary_but_not_sufficient_condition_l118_118626

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (∀ a, a > 2 → a ∈ set.Ici 2) ∧ ¬(∃ a, a ∈ set.Ici 2 ∧ a ≤ 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l118_118626


namespace pies_in_each_row_l118_118493

theorem pies_in_each_row (pecan_pies apple_pies rows : Nat) (hpecan : pecan_pies = 16) (happle : apple_pies = 14) (hrows : rows = 30) :
  (pecan_pies + apple_pies) / rows = 1 :=
by
  sorry

end pies_in_each_row_l118_118493


namespace not_both_periodic_l118_118322

theorem not_both_periodic 
  (W_1 W_2 : String)
  (n : ℕ)
  (h_length : W_1.length = n)
  (h_length2 : W_2.length = n)
  (h_diff_first : W_1.head ≠ W_2.head)
  (h_rest_eq : W_1.drop 1 = W_2.drop 1) :
  ¬ (periodic W_1 ∧ periodic W_2) :=
by
  sorry

end not_both_periodic_l118_118322


namespace four_digit_numbers_count_l118_118709

theorem four_digit_numbers_count : 
  let valid_numbers := {x : ℕ | x ≥ 1000 ∧ x < 10000 ∧ ∀ d ∈ x.digits 10, d = 2 ∨ d = 3 ∧ (∃ d2 ∈ x.digits 10, d2 = 2) ∧ (∃ d3 ∈ x.digits 10, d3 = 3)} in
  valid_numbers.card = 14 :=
by
  sorry

end four_digit_numbers_count_l118_118709


namespace sum_of_coefficients_l118_118820

theorem sum_of_coefficients :
  ∃ a b c : ℕ, (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (sqrt 3 + 1 / sqrt 3 + sqrt 11 + 1 / sqrt 11 = (a * sqrt 3 + b * sqrt 11) / c) ∧
  (∀ d : ℕ, d ≠ 0 → (sqrt 3 + 1 / sqrt 3 + sqrt 11 + 1 / sqrt 11 = (a * sqrt 3 + b * sqrt 11) / d → c ≤ d)) ∧
  (a + b + c = 113) :=
sorry

end sum_of_coefficients_l118_118820


namespace median_moons_l118_118567

def number_of_moons : List ℕ := [0, 0, 1, 2, 2, 5, 10, 15, 18, 24]

noncomputable def median (l : List ℕ) : ℚ :=
  if hl : l.length % 2 = 1 then
    l.sort l.nth_le ((l.length / 2)) sorry
  else
    let mid := l.length / 2
    (l.sort l.nth_le (mid - 1) sorry + l.sort l.nth_le mid sorry) / 2

theorem median_moons : median number_of_moons = 3.5 := sorry

end median_moons_l118_118567


namespace selling_price_correct_l118_118845

-- Define the parameters
def stamp_duty_rate : ℝ := 0.002
def commission_rate : ℝ := 0.0035
def bought_shares : ℝ := 3000
def buying_price_per_share : ℝ := 12
def profit : ℝ := 5967

-- Define the selling price per share
noncomputable def selling_price_per_share (x : ℝ) : ℝ :=
  bought_shares * x - bought_shares * buying_price_per_share -
  bought_shares * x * (stamp_duty_rate + commission_rate) - 
  bought_shares * buying_price_per_share * (stamp_duty_rate + commission_rate)

-- The target selling price per share
def target_selling_price_per_share : ℝ := 14.14

-- Statement of the problem
theorem selling_price_correct (x : ℝ) : selling_price_per_share x = profit → x = target_selling_price_per_share := by
  sorry

end selling_price_correct_l118_118845


namespace max_projection_value_l118_118809

variables (e1 e2 : ℝ^3)
axiom abs_e1 : ‖e1‖ = 3
axiom abs_2e1_plus_e2 : ‖2 • e1 + e2‖ = 3

noncomputable def projection_in_direction (u v : ℝ^3) : ℝ :=
  (u • v) / ‖v‖

theorem max_projection_value :
  projection_in_direction e1 e2 = -3 * real.sqrt 3 / 2 :=
sorry

end max_projection_value_l118_118809


namespace a_left_5_days_before_completion_l118_118608

theorem a_left_5_days_before_completion :
  ∀ (a_rate b_rate : ℝ) (t_total : ℝ),
    a_rate = 1 / 10 →
    b_rate = 1 / 20 →
    t_total = 10 →
    let x := 5 in
    (t_total - x) * (a_rate + b_rate) + x * b_rate = 1 :=
by
  intros a_rate b_rate t_total h1 h2 h3 x
  simp [h1, h2, h3]
  sorry

end a_left_5_days_before_completion_l118_118608


namespace exists_polynomial_with_10000_roots_l118_118096

theorem exists_polynomial_with_10000_roots :
  ∃ (p : ℝ[X]), degree p = 100 ∧ ∃ (p_p : ℝ[X]), p_p = p.comp(p) ∧ root_set p_p ℝ = 10000 :=
sorry

end exists_polynomial_with_10000_roots_l118_118096


namespace shortest_cowboy_trip_l118_118612

noncomputable def shortest_trip_distance 
  (C : ℝ × ℝ) (B : ℝ × ℝ) (T : ℝ × ℝ) : ℝ :=
  let C_reflected := (C.1, -C.2) in
  let d_CB := Real.sqrt ((B.1 - C_reflected.1)^2 + (B.2 - C_reflected.2)^2) in
  let d_BT := Real.sqrt ((T.1 - B.1)^2 + (T.2 - B.2)^2) in
  6 + d_CB + 2 * d_BT

theorem shortest_cowboy_trip 
  (C B T : ℝ × ℝ)
  (hC : C = (0, 6))
  (hB : B = (-12, 11))
  (hT : T = (-9, 9)) : 
  shortest_trip_distance C B T = 6 + Real.sqrt(433) + 2 * Real.sqrt(13) :=
by
  sorry

end shortest_cowboy_trip_l118_118612


namespace correct_calculation_l118_118580

theorem correct_calculation : (∀ x y : ℝ, sqrt x + sqrt y ≠ sqrt (x + y)) ∧
                             (∀ x y : ℝ, (2 * sqrt y) - sqrt y ≠ 2) ∧
                             (∀ x : ℝ, (sqrt 12) / 3 ≠ 2) →
                             (sqrt 2 * sqrt 3 = sqrt 6) := by
  intros h1 h2 h3
  sorry

end correct_calculation_l118_118580


namespace hyperbola_eccentricity_correct_l118_118756

noncomputable def hyperbola_eccentricity (a b : ℝ) (c : ℝ) : ℝ :=
  c / a

theorem hyperbola_eccentricity_correct (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (c : ℝ) (hc : c = Real.sqrt (a^2 + b^2))
  (h_dist : Real.abs (b * c / Real.sqrt (a^2 + b^2)) = (Real.sqrt 2 / 3) * c) :
  hyperbola_eccentricity a b c = (3 * Real.sqrt 7) / 7 :=
by
  sorry

end hyperbola_eccentricity_correct_l118_118756


namespace three_digit_multiples_of_30_not_75_l118_118772

theorem three_digit_multiples_of_30_not_75 : 
  let multiples_of_30 := list.range' 120 (990 - 120 + 30) 30
  let multiples_of_75 := list.range' 150 (900 - 150 + 150) 150
  let non_multiples_of_75 := multiples_of_30.filter (λ x => x % 75 ≠ 0)
  non_multiples_of_75.length = 24 := 
by 
  sorry

end three_digit_multiples_of_30_not_75_l118_118772


namespace quadratic_root_property_l118_118736

theorem quadratic_root_property (m n : ℝ)
  (hmn : m^2 + m - 2021 = 0)
  (hn : n^2 + n - 2021 = 0) :
  m^2 + 2 * m + n = 2020 :=
by sorry

end quadratic_root_property_l118_118736


namespace circumcircle_eq_of_triangle_vertices_l118_118747

theorem circumcircle_eq_of_triangle_vertices (A B C: ℝ × ℝ) (hA : A = (0, 4)) (hB : B = (0, 0)) (hC : C = (3, 0)) :
  ∃ D E F : ℝ,
    x^2 + y^2 + D*x + E*y + F = 0 ∧
    (x - 3/2)^2 + (y - 2)^2 = 25/4 :=
by 
  sorry

end circumcircle_eq_of_triangle_vertices_l118_118747


namespace smallest_percentage_of_both_soda_and_milk_l118_118149

variable {α : Type} [fintype α]

noncomputable def percentage_of_soda_drinkers : ℝ := 0.9
noncomputable def percentage_of_milk_drinkers : ℝ := 0.8

theorem smallest_percentage_of_both_soda_and_milk :
  ∃ p : ℝ, p ≥ 0 ∧ p = 0.7 ∧ percentage_of_soda_drinkers + percentage_of_milk_drinkers - 1 = p :=
by
  sorry

end smallest_percentage_of_both_soda_and_milk_l118_118149


namespace length_of_plot_l118_118522

noncomputable def cost_per_meter : ℝ := 26.50
noncomputable def total_cost : ℝ := 5300
def breadth (x : ℝ) : Prop := x > 0
def length (x : ℝ) : ℝ := x + 50
def perimeter (x : ℝ) : ℝ := 2 * (x + 50) + 2 * x
def cost_of_fencing (x : ℝ) : ℝ := total_cost / cost_per_meter

theorem length_of_plot (x : ℝ) (h1 : breadth x) (h2 : cost_of_fencing x = perimeter x) :
  length x = 75 :=
sorry

end length_of_plot_l118_118522


namespace kolya_correct_valya_incorrect_l118_118304

variable (x : ℝ)

def p : ℝ := 1 / x
def r : ℝ := 1 / (x + 1)
def q : ℝ := (x - 1) / x
def s : ℝ := x / (x + 1)

theorem kolya_correct : r / (1 - s * q) = 1 / 2 := by
  sorry

theorem valya_incorrect : (q * r + p * r) / (1 - s * q) = 1 / 2 := by
  sorry

end kolya_correct_valya_incorrect_l118_118304


namespace circle_and_tangent_lines_l118_118381

open Real

noncomputable def circle_eq (x y a : ℝ) := (x - a) ^ 2 + (y - 2*a) ^ 2 = 5

theorem circle_and_tangent_lines
  (A B C : Point)
  (A_x A_y B_x B_y P_x P_y : ℝ)
  (hA : A.x = 3 ∧ A.y = 2)
  (hB : B.x = 1 ∧ B.y = 6)
  (hC : C.y = 2 * C.x)
  (hP : P_x = -1 ∧ P_y = 3) :
  (∃ r, circle_eq A.x A.y C.x ∧ circle_eq B.x B.y C.x ∧ (C.x = 2 ∧ r = sqrt 5)) ∧
  (∃ k, line_eq k P_x P_y C ∧ tangent_to_circle k C)
:=
sorry

end circle_and_tangent_lines_l118_118381


namespace john_new_salary_after_raise_l118_118100

theorem john_new_salary_after_raise (original_salary : ℝ) (percentage_increase : ℝ) (h1 : original_salary = 60) (h2 : percentage_increase = 0.8333333333333334) : 
  original_salary * (1 + percentage_increase) = 110 := 
sorry

end john_new_salary_after_raise_l118_118100


namespace prove_ellipse_and_sum_constant_l118_118854

-- Define the ellipse properties
def ellipse_center_origin (a b : ℝ) : Prop :=
  a = 4 ∧ b^2 = 12

-- Standard equation of the ellipse
def ellipse_standard_eqn (x y : ℝ) : Prop :=
  (x^2 / 16) + (y^2 / 12) = 1

-- Define the conditions for m and n given point M(1, 3)
def condition_m_n (m n : ℝ) (x0 : ℝ) : Prop :=
  (9 * m^2 + 96 * m + 48 - (13/4) * x0^2 = 0) ∧ (9 * n^2 + 96 * n + 48 - (13/4) * x0^2 = 0)

-- Prove the standard equation of the ellipse and m+n constant properties
theorem prove_ellipse_and_sum_constant (a b x y m n x0 : ℝ) 
  (h1 : ellipse_center_origin a b)
  (h2 : ellipse_standard_eqn x y)
  (h3 : condition_m_n m n x0) :
  m + n = -32/3 := 
sorry

end prove_ellipse_and_sum_constant_l118_118854


namespace count_irrational_numbers_l118_118652

open Real

theorem count_irrational_numbers : 
  let numbers := [ -π / 2, 1 / 3, abs (-3), sqrt 4, cbrt (-8), sqrt 7 ]
  in (count is_irrational numbers) = 2 
  where
    is_irrational (x : ℝ) : Prop := ¬ x.is_rational :=
  by sorry

end count_irrational_numbers_l118_118652


namespace part_1_part_2_l118_118094

variables {A B C D : Type}
variables {a b c : ℝ} -- Side lengths of the triangle
variables {A_angle B_angle C_angle : ℝ} -- Angles of the triangle
variables {R : ℝ} -- Circumradius of the triangle

-- Assuming the given conditions:
axiom b_squared_eq_ac : b^2 = a * c
axiom bd_sin_eq_a_sin_C : ∀ {BD : ℝ}, BD * sin B_angle = a * sin C_angle
axiom ad_eq_2dc : ∀ {AD DC : ℝ}, AD = 2 * DC

-- Theorems to prove:
theorem part_1 (BD : ℝ) : BD * sin B_angle = a * sin C_angle → BD = b := by
  intros h
  sorry

theorem part_2 (AD DC : ℝ) (H : AD = 2 * DC) : cos B_angle = 7 / 12 := by
  intros h
  sorry

end part_1_part_2_l118_118094


namespace problem_statement_l118_118815

noncomputable def a : ℕ := 44
noncomputable def b : ℕ := 36
noncomputable def c : ℕ := 33

theorem problem_statement : \( \sqrt{3}+\frac{1}{\sqrt{3}} + \sqrt{11} + \frac{1}{\sqrt{11}} = \dfrac{44\sqrt{3} + 36\sqrt{11}}{33} \) ∧ a + b + c = 113 :=
by
    sorry

end problem_statement_l118_118815


namespace boat_speed_in_still_water_l118_118247

variable (x : ℝ) -- Speed of the boat in still water
variable (r : ℝ) -- Rate of the stream
variable (d : ℝ) -- Distance covered downstream
variable (t : ℝ) -- Time taken downstream

theorem boat_speed_in_still_water (h_rate : r = 5) (h_distance : d = 168) (h_time : t = 8) :
  x = 16 :=
by
  -- Substitute conditions into the equation.
  -- Calculate the effective speed downstream.
  -- Solve x from the resulting equation.
  sorry

end boat_speed_in_still_water_l118_118247


namespace average_selections_per_car_l118_118636

-- Definitions based on conditions
def num_cars : ℕ := 12
def num_clients : ℕ := 9
def selections_per_client : ℕ := 4

-- Theorem to prove
theorem average_selections_per_car :
  (num_clients * selections_per_client) / num_cars = 3 :=
by
  -- Placeholder for the proof
  sorry

end average_selections_per_car_l118_118636


namespace flu_indefinite_spread_if_initial_immunity_flu_ceases_no_initial_immunity_l118_118843

theorem flu_indefinite_spread_if_initial_immunity
  (people : Type)
  (friends : people → set people)
  (immune healthy infected : people → Prop)
  (visit : ∀ p : people,  friends p ≠ ∅)
  (imfection_rule : ∀ p q : people, infected q → q ∈ (friends p) → ¬ immune p → healthy p → infected p)
  (recovery_rule : ∀ p : people, infected p → immune p ∧ ¬ infected p)
  (immunity_period : ∀ p : people, immune p → healthy p)
  (initial_immunity : ∃ p : people, immune p):
  (∀ d : ℕ, ∃ p : people, infected p) :=
sorry

theorem flu_ceases_no_initial_immunity
  (people : Type)
  (friends : people → set people)
  (immune healthy infected : people → Prop)
  (visit : ∀ p : people,  friends p ≠ ∅)
  (imfection_rule : ∀ p q : people, infected q → q ∈ (friends p) → ¬ immune p → healthy p → infected p)
  (recovery_rule : ∀ p : people, infected p → immune p ∧ ¬ infected p)
  (immunity_period : ∀ p : people, immune p → healthy p)
  (no_initial_immunity : ∀ p : people, ¬ immune p):
  ∃ d : ℕ, ∀ p : people, ¬ infected p :=
sorry

end flu_indefinite_spread_if_initial_immunity_flu_ceases_no_initial_immunity_l118_118843


namespace angle_is_90_degrees_l118_118884

def a : ℝ^3 := ⟨2, -3, -4⟩
def b : ℝ^3 := ⟨real.sqrt 3, 5, -2⟩
def c : ℝ^3 := ⟨7, -2, 8⟩

def ac_dot : ℝ := 2 * 7 + (-3) * (-2) + (-4) * 8
def ab_dot : ℝ := 2 * real.sqrt 3 + (-3) * 5 + (-4) * (-2)

def new_vec : ℝ^3 :=
  (ac_dot * b) - (ab_dot • c)

theorem angle_is_90_degrees :
  ∃ θ : ℝ, θ = 90 ∧ (a ⬝ new_vec) = 0 := 
sorry

end angle_is_90_degrees_l118_118884


namespace min_positive_value_sum_50_l118_118347

theorem min_positive_value_sum_50 :
  (∃ a : Fin 50 → ℤ,
    (∀ i : Fin 50, a i = 1 ∨ a i = -1) ∧ 
    (∑ i : Fin 50, a i) % 2 = 0 ∧
    (let S := ∑ i in finset.range 50, ∑ j in finset.range i, a i * a j in
     S = 7)) :=
sorry

end min_positive_value_sum_50_l118_118347


namespace problem_statement_l118_118330

noncomputable def complex_exp_eq (a b : ℝ) : Prop := (a + (b * complex.I))^4 = (a - (b * complex.I))^4

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : complex_exp_eq a b) : b / a = 1 := 
by sorry

end problem_statement_l118_118330


namespace calculate_new_average_weight_l118_118972

noncomputable def new_average_weight (original_team_weight : ℕ) (num_original_players : ℕ) 
 (new_player1_weight : ℕ) (new_player2_weight : ℕ) (num_new_players : ℕ) : ℕ :=
 (original_team_weight + new_player1_weight + new_player2_weight) / (num_original_players + num_new_players)

theorem calculate_new_average_weight : 
  new_average_weight 847 7 110 60 2 = 113 := 
by 
sorry

end calculate_new_average_weight_l118_118972


namespace probability_divisor_of_10_on_8_sided_die_l118_118618

theorem probability_divisor_of_10_on_8_sided_die :
  let outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
  let divisors_of_10 := {1, 2, 5, 10}
  let favorable_outcomes := outcomes ∩ divisors_of_10
  (favorable_outcomes.card : ℚ) / outcomes.card = 3 / 8 :=
by
  sorry

end probability_divisor_of_10_on_8_sided_die_l118_118618


namespace least_number_subtracted_l118_118592

theorem least_number_subtracted (n : ℕ) (h : n = 964807) : ∃ x : ℕ, x = 7 ∧ (n - x) % 8 = 0 :=
by
  use 7
  split
  · rfl
  · rw h
    norm_num
    sorry

end least_number_subtracted_l118_118592


namespace sum_consecutive_integers_product_1080_l118_118176

theorem sum_consecutive_integers_product_1080 :
  ∃ n : ℕ, n * (n + 1) = 1080 ∧ n + (n + 1) = 65 :=
by
  sorry

end sum_consecutive_integers_product_1080_l118_118176


namespace number_of_guys_l118_118186

theorem number_of_guys (n : ℕ) (h : n ≥ 1) :
  let bullets_original := 25
  let bullets_shot := 4
  let bullets_remaining_per_guy := bullets_original - bullets_shot
  let total_remaining_bullets := n * bullets_remaining_per_guy
  let total_shots := n * bullets_shot
  let total_bullets_original := n * bullets_original
  in total_bullets_original - total_shots = total_remaining_bullets :=
by
  sorry

end number_of_guys_l118_118186


namespace g_eq_one_l118_118891

theorem g_eq_one (g : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), g (x - y) = g x * g y) 
  (h2 : ∀ (x : ℝ), g x ≠ 0) : 
  g 5 = 1 :=
by
  sorry

end g_eq_one_l118_118891


namespace log_base2_result_l118_118362

theorem log_base2_result :
  ∑ n in (finset.range 21).filter (λ n, n > 0), (2 * (n ^ 2) + n) = 5950 :=
by
  sorry

end log_base2_result_l118_118362


namespace possible_values_of_g_l118_118887

theorem possible_values_of_g (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ x ∈ set.Icc (0 : ℝ) (real.infinity), 
    x = (a / (a^2 + b) + b / (b^2 + c) + c / (c^2 + a)) :=
begin
  sorry,
end

end possible_values_of_g_l118_118887


namespace jebb_expense_l118_118099

-- Define the costs
def seafood_platter := 45.0
def rib_eye_steak := 38.0
def vintage_wine_glass := 18.0
def chocolate_dessert := 12.0

-- Define the rules and discounts
def discount_percentage := 0.10
def service_fee_12 := 0.12
def service_fee_15 := 0.15
def tip_percentage := 0.20

-- Total food and wine cost
def total_food_and_wine_cost := 
  seafood_platter + rib_eye_steak + (2 * vintage_wine_glass) + chocolate_dessert

-- Total food cost excluding wine
def food_cost_excluding_wine := 
  seafood_platter + rib_eye_steak + chocolate_dessert

-- 10% discount on food cost excluding wine
def discount_amount := discount_percentage * food_cost_excluding_wine
def reduced_food_cost := food_cost_excluding_wine - discount_amount

-- New total cost before applying the service fee
def total_cost_before_service_fee := reduced_food_cost + (2 * vintage_wine_glass)

-- Determine the service fee based on cost
def service_fee := 
  if total_cost_before_service_fee > 80.0 then 
    service_fee_15 * total_cost_before_service_fee 
  else if total_cost_before_service_fee >= 50.0 then 
    service_fee_12 * total_cost_before_service_fee 
  else 
    0.0

-- Total cost after discount and service fee
def total_cost_after_service_fee := total_cost_before_service_fee + service_fee

-- Tip amount (20% of total cost after discount and service fee)
def tip_amount := tip_percentage * total_cost_after_service_fee

-- Total amount Jebb spent
def total_amount_spent := total_cost_after_service_fee + tip_amount

-- Lean theorem statement
theorem jebb_expense :
  total_amount_spent = 167.67 :=
by
  -- prove the theorem here
  sorry

end jebb_expense_l118_118099


namespace center_of_symmetry_l118_118405

theorem center_of_symmetry :
  ∃ c : ℝ, (c, 0) = (π / 4, 0) ∧ 
    (∀ (x y : ℝ), (x, y) ∈ (fun (x : ℝ) => g(x)) → (π / 12 - x, y) ∈ (fun (x : ℝ) => f(x))) ∧ 
    (f(x) = (sin (4 * x + π / 3)) / (sin (2 * x + 2 * π / 3))) ∧ 
    (x = π / 12) :=
sorry

end center_of_symmetry_l118_118405


namespace max_fitting_rectangles_l118_118039

def large_rectangle_area : ℝ := 50 * 90
def small_rectangle_area : ℝ := 1 * 10 * Real.sqrt 2

-- Constants representing dimensions
def large_width : ℝ := 50
def large_height : ℝ := 90
def small_width : ℝ := 1
def small_height : ℝ := 10 * Real.sqrt 2

theorem max_fitting_rectangles (w_L h_L w_S h_S : ℝ) 
  (h_large: w_L = 50) (h_large2: h_L = 90) 
  (h_small: w_S = 1) (h_small2: h_S = 10 * Real.sqrt 2) :
  ∃ n, n = 300 :=
by
  use 300
  sorry

end max_fitting_rectangles_l118_118039


namespace distance_of_intersections_l118_118015

theorem distance_of_intersections : 
  let C₁ := {p : ℝ × ℝ | (p.1 + 2)^2 + (p.2 - 1)^2 = 1}
  let C₂ := {p : ℝ × ℝ | p.1^2 / 16 + p.2^2 / 9 = 1}
  let line := {p : ℝ × ℝ | ∃ s : ℝ, p.1 = -4 + (s / sqrt 2) ∧ p.2 = s / sqrt 2}
  let intersections := C₁ ∩ line
  let points : set (ℝ × ℝ) := {(x1, y1) | ∃ q ∈ intersections, q = (x1, y1)}
  ∃ A B : ℝ × ℝ, A ∈ points ∧ B ∈ points ∧ A ≠ B ∧ 
    real.dist A B = sqrt 2 :=
by 
  sorry

end distance_of_intersections_l118_118015


namespace arrange_balls_l118_118977

/-- Given 4 yellow balls and 3 red balls, we want to prove that there are 35 different ways to arrange these balls in a row. -/
theorem arrange_balls : (Nat.choose 7 4) = 35 := by
  sorry

end arrange_balls_l118_118977


namespace correct_calculation_l118_118582

theorem correct_calculation : (∀ x y : ℝ, sqrt x + sqrt y ≠ sqrt (x + y)) ∧
                             (∀ x y : ℝ, (2 * sqrt y) - sqrt y ≠ 2) ∧
                             (∀ x : ℝ, (sqrt 12) / 3 ≠ 2) →
                             (sqrt 2 * sqrt 3 = sqrt 6) := by
  intros h1 h2 h3
  sorry

end correct_calculation_l118_118582


namespace perimeter_of_square_is_160_cm_l118_118153

noncomputable def area_of_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_of_square (area_of_rectangle : ℝ) : ℝ := 5 * area_of_rectangle

noncomputable def side_length_of_square (area_of_square : ℝ) : ℝ := Real.sqrt area_of_square

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ := 4 * side_length

theorem perimeter_of_square_is_160_cm :
  perimeter_of_square (side_length_of_square (area_of_square (area_of_rectangle 32 10))) = 160 :=
by
  sorry

end perimeter_of_square_is_160_cm_l118_118153


namespace incorrect_option_B_l118_118503

def round_nearest (x : ℚ) (d : ℚ) : ℚ :=
  let multiple := x / d
  d * (round multiple).toRat

def x := 0.05019
def A := round_nearest x 0.1 = 0.1
def B := round_nearest x 0.01 = 0.10
def C := round_nearest x 0.001 = 0.050
def D := round_nearest x 0.0001 = 0.0502

theorem incorrect_option_B : B → False :=
by
  have b_nearest := round_nearest x 0.01
  have b_correct := 0.05
  have h : b_nearest ≠ 0.10 := by sorry -- Calculation to show that round_nearest x 0.01 = 0.05
  exact h

#eval incorrect_option_B

end incorrect_option_B_l118_118503


namespace area_of_circle_l118_118137

--Define the points A and B
def A : ℝ × ℝ := (5, 12)
def B : ℝ × ℝ := (14, 8)

--Define a theorem to prove the area of the circle given the conditions
theorem area_of_circle (C : ℝ × ℝ)
  (hA_on_ω : ∃ O : ℝ × ℝ, ∃ r : ℝ, dist A O = r ∧ dist B O = r)
  (h_tangents_intersect : C.2 = 0) :
  ∃ r : ℝ, r^2 = (121975 / 1961) ∧ real.pi * r^2 = (121975 * real.pi) / 1961 := 
by
  sorry

end area_of_circle_l118_118137


namespace larger_number_l118_118553

theorem larger_number (a b : ℤ) (h1 : a - b = 5) (h2 : a + b = 37) : a = 21 :=
sorry

end larger_number_l118_118553


namespace factorization_check_l118_118229

theorem factorization_check 
  (A : 4 - x^2 + 3 * x ≠ (2 - x) * (2 + x) + 3)
  (B : -x^2 + 3 * x + 4 ≠ -(x + 4) * (x - 1))
  (D : x^2 * y - x * y + x^3 * y ≠ x * (x * y - y + x^2 * y)) :
  1 - 2 * x + x^2 = (1 - x) ^ 2 :=
by
  sorry

end factorization_check_l118_118229


namespace solve_green_balls_l118_118609

variables (total_balls white_balls yellow_balls red_balls purple_balls green_balls : ℕ)
variable (prob_neither_red_nor_purple : ℚ)

-- Given conditions
def conditions : Prop :=
  total_balls = 60 ∧
  white_balls = 22 ∧
  yellow_balls = 8 ∧
  red_balls = 5 ∧
  purple_balls = 7 ∧
  prob_neither_red_nor_purple = 0.8

-- Question to be proved
def proof_problem : Prop :=
  green_balls = 18

-- Lean theorem statement
theorem solve_green_balls (h : conditions total_balls white_balls yellow_balls red_balls purple_balls green_balls prob_neither_red_nor_purple) :
  proof_problem total_balls white_balls yellow_balls red_balls purple_balls green_balls :=
  sorry

end solve_green_balls_l118_118609


namespace A_is_linear_l118_118479

variables {X_3 : Type*} [AddCommGroup X_3] [Module ℝ X_3]

noncomputable def A (x : X_3) : X_3 :=
{x1 - x2, 2 * x1 + x3, 3 * x1}

theorem A_is_linear (x y : X_3) (α : ℝ) :
  A (x + y) = A x + A y ∧
  A (α • x) = α • A x := 
by 
  sorry

end A_is_linear_l118_118479


namespace min_pos_diff_geom_arith_l118_118145

def geom_sequence (a r n : ℕ) : ℕ := a * r ^ n
def arith_sequence (a d n : ℕ) : ℕ := a + d * n

theorem min_pos_diff_geom_arith (A B : ℕ → ℕ) (hA : ∀ n, A n = geom_sequence 3 2 n)
                                 (hB : ∀ n, B n = arith_sequence 15 15 n) :
                                 ∃ k, k > 0 ∧ k = min { | A i - B j | | i j : ℕ, A i ≤ 300 ∧ B j ≤ 300 } := by
  sorry

end min_pos_diff_geom_arith_l118_118145


namespace no_integer_solutions_l118_118505

theorem no_integer_solutions (x y z : ℤ) : ¬ (x^2 + y^2 = 3 * z^2) :=
sorry

end no_integer_solutions_l118_118505


namespace find_principal_l118_118233

theorem find_principal (SI : ℝ) (R : ℝ) (T : ℝ) (hSI : SI = 4025.25) (hR : R = 9) (hT : T = 5) :
    let P := SI / (R * T / 100)
    P = 8950 :=
by
  -- we will put proof steps here
  sorry

end find_principal_l118_118233


namespace count_3_digit_numbers_multiple_30_not_75_l118_118790

theorem count_3_digit_numbers_multiple_30_not_75 : 
  (finset.filter (λ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 30 = 0 ∧ n % 75 ≠ 0) (finset.range 1000)).card = 21 := sorry

end count_3_digit_numbers_multiple_30_not_75_l118_118790


namespace part1_part2_part3_l118_118011

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≥ 0 then a^x - 1 else -a^(-x) + 1

theorem part1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  f 2 a + f (-2) a = 0 := 
sorry

theorem part2 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∀ x : ℝ, f x a = if x ≥ 0 then a^x - 1 else -a^(-x) + 1 := 
sorry

theorem part3 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (if a > 1 then (1 - log a 2, 1 + log a 5) else set.univ : set ℝ) = 
  { x | -1 < f (x - 1) a ∧ f (x - 1) a < 4 } := 
sorry

end part1_part2_part3_l118_118011


namespace part1_part2_l118_118028

-- Definitions of functions f and g
def f (x : ℝ) : ℝ := Real.log x
def g (x m : ℝ) : ℝ := x + m
def F (x m : ℝ) : ℝ := f x - g x m

-- Condition for part (1)
def part1_condition (x m : ℝ) : Prop := f x ≤ g x m

-- Question for part (1)
theorem part1 (m : ℝ) (h : ∀ x : ℝ, x > 0 → part1_condition x m) : m ≥ -1 := sorry

-- Conditions for part (2)
def part2_roots_condition (x1 x2 m : ℝ) : Prop :=
  x1 < x2 ∧ F x1 m = 0 ∧ F x2 m = 0

-- Question for part (2)
theorem part2 (x1 x2 m : ℝ) (h1 : part2_roots_condition x1 x2 m) (h2 : m < -1) (h3 : 0 < x1) (h4 : x2 > 1) : x1 * x2 < 1 := sorry

end part1_part2_l118_118028


namespace required_blue_balls_to_remove_l118_118261

-- Define the constants according to conditions
def total_balls : ℕ := 120
def red_balls : ℕ := 54
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℚ := 0.75 -- ℚ is the type for rational numbers

-- Lean theorem statement
theorem required_blue_balls_to_remove (x : ℕ) : 
    (red_balls:ℚ) / (total_balls - x : ℚ) = desired_percentage_red → x = 48 :=
by
  sorry

end required_blue_balls_to_remove_l118_118261


namespace kolya_correct_valya_incorrect_l118_118309

-- Definitions from the problem conditions:
def hitting_probability_valya (x : ℝ) : ℝ := 1 / (x + 1)
def hitting_probability_kolya (x : ℝ) : ℝ := 1 / x
def missing_probability_kolya (x : ℝ) : ℝ := 1 - (1 / x)
def missing_probability_valya (x : ℝ) : ℝ := 1 - (1 / (x + 1))

-- Proof for Kolya's assertion
theorem kolya_correct (x : ℝ) (hx : x > 0) : 
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2) :=
sorry

-- Proof for Valya's assertion
theorem valya_incorrect (x : ℝ) (hx : x > 0) :
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2 ∧ (r + p - r * p) / (1 - s * q) = 1 / 2) :=
sorry

end kolya_correct_valya_incorrect_l118_118309


namespace purchasing_power_decrease_l118_118710

theorem purchasing_power_decrease (orig_price final_price unchanged_wage : ℕ)
  (price_increase_rate : ℚ)
  (price_increase : final_price = orig_price * (1 + price_increase_rate)) :
  unchanged_wage / final_price = (1 / (1 + price_increase_rate)) →
  (1 - (unchanged_wage / final_price)) * 100 = 60 :=
by
  assume orig_price final_price unchanged_wage : ℕ
  assume price_increase_rate : ℚ
  assume h1 : final_price = orig_price * (1 + price_increase_rate)
  assume h2 : unchanged_wage / final_price = (1 / (1 + price_increase_rate))
  exact sorry

end purchasing_power_decrease_l118_118710


namespace find_x_l118_118958

theorem find_x (x : ℝ) (data_set : set ℝ) (h1 : data_set = {3, -1, 0, 2, x}) (h2 : (sup data_set) - (inf data_set) = 5) :
  x = -2 ∨ x = 4 :=
  sorry

end find_x_l118_118958


namespace planes_parallel_to_line_l118_118601

theorem planes_parallel_to_line (l : Line) (P Q : Point) (hP : P ∉ l) (hQ : Q ∉ l) : 
    ∃ (n : ℕ), n = 0 ∨ n = 1 ∨ n = ∞ :=
by 
  -- Since this is only a statement with sorry, no proof is provided. 
  sorry

end planes_parallel_to_line_l118_118601


namespace hyperbola_condition_l118_118966

theorem hyperbola_condition (k : ℝ) : 
  (-1 < k ∧ k < 1) ↔ (∃ x y : ℝ, (x^2 / (k-1) + y^2 / (k+1)) = 1) := 
sorry

end hyperbola_condition_l118_118966


namespace possible_values_of_a_l118_118798

variables {a b c d : ℝ} (h1 : (b * d ≠ 0)) (h2 : (a / b < -c / d))

theorem possible_values_of_a : (a: ℝ) :=
begin
  -- This theorem should prove that 'a' can be positive, negative, or zero.
  -- Since we abstract the proof details, we just assert the required statement.
  sorry
end

end possible_values_of_a_l118_118798


namespace three_digit_multiples_of_30_not_75_l118_118773

theorem three_digit_multiples_of_30_not_75 : 
  let multiples_of_30 := list.range' 120 (990 - 120 + 30) 30
  let multiples_of_75 := list.range' 150 (900 - 150 + 150) 150
  let non_multiples_of_75 := multiples_of_30.filter (λ x => x % 75 ≠ 0)
  non_multiples_of_75.length = 24 := 
by 
  sorry

end three_digit_multiples_of_30_not_75_l118_118773


namespace fraction_equality_l118_118350

theorem fraction_equality (a b c : ℝ) (h1 : b + c + a ≠ 0) (h2 : b + c ≠ a) : 
  (b^2 + a^2 - c^2 + 2*b*c) / (b^2 + c^2 - a^2 + 2*b*c) = 1 := 
by 
  sorry

end fraction_equality_l118_118350


namespace find_number_l118_118648

theorem find_number :
  ∃ x : ℤ, 27 * (x + 143) = 9693 ∧ x = 216 :=
by 
  existsi 216
  sorry

end find_number_l118_118648


namespace find_increasing_intervals_find_range_CO_dot_CA_CB_l118_118022

-- Problem 1
def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + 2 * sin (x - π / 4) * sin (x + π / 4)

theorem find_increasing_intervals :
  set_of (λ x, ∃ k : ℤ, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3) = {x | ∀ k : ℤ, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3} :=
by sorry

-- Problem 2
def f_C (C : ℝ) : ℝ := sin (2 * C - π / 6)

theorem find_range_CO_dot_CA_CB (C : ℝ) (a b c : ℝ) (O : point3d) [c = 2 * sqrt 3]
  (f_C_eq : f_C C = 1)
  (O_eq: | O.A | = | O.B | = | O.C |) :
  6 ≤ CO ⬝ (CA + CB) ∧ CO ⬝ (CA + CB) ≤ 12 :=
by sorry

end find_increasing_intervals_find_range_CO_dot_CA_CB_l118_118022


namespace part1_prove_BD_eq_b_part2_prove_cos_ABC_l118_118087

-- Definition of the problem setup
variables {a b c : ℝ}
variables {A B C : ℝ}    -- angles
variables {D : ℝ}        -- point on side AC

-- Conditions
axiom b_squared_eq_ac : b^2 = a * c
axiom BD_sin_ABC_eq_a_sin_C : D * Real.sin B = a * Real.sin C
axiom AD_eq_2DC : A = 2 * C

-- Part (1)
theorem part1_prove_BD_eq_b : D = b :=
by
  sorry

-- Part (2)
theorem part2_prove_cos_ABC :
  Real.cos B = 7 / 12 :=
by
  sorry

end part1_prove_BD_eq_b_part2_prove_cos_ABC_l118_118087


namespace find_b_plus_m_l118_118106

def matrix_C (b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 3, b],
    ![0, 1, 5],
    ![0, 0, 1]
  ]

def matrix_RHS : Matrix (Fin 3) (Fin 3) ℕ :=
  ![
    ![1, 27, 3003],
    ![0, 1, 45],
    ![0, 0, 1]
  ]

theorem find_b_plus_m (b m : ℕ) (h : matrix_C b ^ m = matrix_RHS) : b + m = 306 := 
  sorry

end find_b_plus_m_l118_118106


namespace polynomial_is_integer_l118_118196

def f (x : ℤ) : ℚ := 1/5 * x^5 + 1/3 * x^3 + 7/15 * x

theorem polynomial_is_integer (x : ℤ) : f(x) ∈ ℤ := sorry

end polynomial_is_integer_l118_118196


namespace number_of_children_l118_118339

theorem number_of_children (pencils_per_child : ℕ) (total_pencils : ℕ) (h1 : pencils_per_child = 4) (h2 : total_pencils = 32) : (total_pencils / pencils_per_child) = 8 :=
by
  rw [h1, h2]
  norm_num

end number_of_children_l118_118339


namespace surface_area_of_sphere_l118_118047

theorem surface_area_of_sphere (h : ℝ) (v : ℝ) (s : ℝ) (side : ℝ):
  h = 4 ∧ v = 16 ∧ side^2 * h = v ∧ s = 2 * real.sqrt(2 * side^2 + h^2)
  → 4 * real.pi * (s / 2)^2 = 24 * real.pi :=
by
  sorry

end surface_area_of_sphere_l118_118047


namespace parking_garage_full_spots_l118_118266

theorem parking_garage_full_spots :
  let total_spots := 4 * 100,
      open_first_level := 58,
      open_second_level := open_first_level + 2,
      open_third_level := open_second_level + 5,
      open_fourth_level := 31,
      total_open_spots := open_first_level + open_second_level + open_third_level + open_fourth_level in
  total_spots - total_open_spots = 186 :=
by
  let total_spots := 4 * 100
  let open_first_level := 58
  let open_second_level := open_first_level + 2
  let open_third_level := open_second_level + 5
  let open_fourth_level := 31
  let total_open_spots := open_first_level + open_second_level + open_third_level + open_fourth_level
  show total_spots - total_open_spots = 186
  sorry

end parking_garage_full_spots_l118_118266


namespace monic_poly_degree4_l118_118478

theorem monic_poly_degree4 (p : ℝ → ℝ)
  (h_monic : ∀ c, leading_coeff p = 1)
  (h_deg : degree p = 4)
  (h_p1 : p 1 = 17)
  (h_p2 : p 2 = 38)
  (h_p3 : p 3 = 63) :
  p 0 + p 4 = 68 := 
sorry

end monic_poly_degree4_l118_118478


namespace find_f_neg_2_l118_118008

def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 4

variable (a b : ℝ)

theorem find_f_neg_2 (h1 : f a b 2 = 6) : f a b (-2) = -14 :=
by
  sorry

end find_f_neg_2_l118_118008


namespace odd_nat_numbers_with_friendly_integer_l118_118274

def is_friendly (n : ℕ) : Prop :=
  ∀ i, (0 < i) ∧ (i < nat.digits 10 n).length → (nat.digits 10 n).nth i - (nat.digits 10 n).nth (i - 1) = 1

theorem odd_nat_numbers_with_friendly_integer (m : ℕ) (hm1 : odd m) : 
  (∃ n : ℕ, is_friendly n ∧ 64 * m ∣ n) ↔ (nat.gcd m 5 = 1) :=
sorry

end odd_nat_numbers_with_friendly_integer_l118_118274


namespace probability_nine_heads_l118_118994

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118994


namespace cost_of_fencing_irregular_pentagon_l118_118689

noncomputable def total_cost_fencing (AB BC CD DE AE : ℝ) (cost_per_meter : ℝ) : ℝ := 
  (AB + BC + CD + DE + AE) * cost_per_meter

theorem cost_of_fencing_irregular_pentagon :
  total_cost_fencing 20 25 30 35 40 2 = 300 := 
by
  sorry

end cost_of_fencing_irregular_pentagon_l118_118689


namespace cos_alpha_pi_over_4_eq_neg_56_over_65_l118_118379

theorem cos_alpha_pi_over_4_eq_neg_56_over_65
  (α β : ℝ)
  (hα : α ∈ Ioo (3*π/4) π)
  (hβ : β ∈ Ioo (3*π/4) π)
  (h1 : sin (α + β) = -3/5)
  (h2 : sin (β - π/4) = 12/13)
  (h3 : α + β ∈ Ioo (3*π/2) 2*π)
  (h4 : β - π/4 ∈ Ioo (π/2) (3*π/4)) :
  cos (α + π/4) = -56/65 := sorry

end cos_alpha_pi_over_4_eq_neg_56_over_65_l118_118379


namespace black_equals_sum_of_white_l118_118904

theorem black_equals_sum_of_white :
  ∃ (a b c d : ℤ) (a_neq_zero : a ≠ 0) (b_neq_zero : b ≠ 0) (c_neq_zero : c ≠ 0) (d_neq_zero : d ≠ 0),
    (c + d * Real.sqrt 7 = (Real.sqrt (a + b * Real.sqrt 2) + Real.sqrt (a - b * Real.sqrt 2))^2) :=
by
  sorry

end black_equals_sum_of_white_l118_118904


namespace initial_pieces_l118_118455

-- Define the conditions
def pieces_used : ℕ := 156
def pieces_left : ℕ := 744

-- Define the total number of pieces of paper Isabel bought initially
def total_pieces : ℕ := pieces_used + pieces_left

-- State the theorem that we need to prove
theorem initial_pieces (h1 : pieces_used = 156) (h2 : pieces_left = 744) : total_pieces = 900 :=
by
  sorry

end initial_pieces_l118_118455


namespace area_is_correct_l118_118847

open Real

-- Definitions given in the problem statement
def side_length := 1
def angle_FAB := real.pi * 75 / 180
def angle_BCD := real.pi * 75 / 180

-- Calculations given in the problem statement
def area_of_figure : ℝ := 
  2 * ((1 / 2) * (sin (real.pi * 75 / 180)))

-- The proof statement
theorem area_is_correct :
  area_of_figure = (sqrt 6 + sqrt 2) / 2 :=
by 
  unfold area_of_figure
  rw [sin_add, sin_pi_div_two, cos_pi_div_two, sin_pi_div_three, cos_pi_div_three]
  sorry

end area_is_correct_l118_118847


namespace larger_of_two_numbers_l118_118561

  theorem larger_of_two_numbers (x y : ℕ) 
    (h₁ : x + y = 37) 
    (h₂ : x - y = 5) 
    : x = 21 :=
  sorry
  
end larger_of_two_numbers_l118_118561


namespace relationship_money_masks_l118_118230

-- Definitions based on the conditions
def initial_money : ℕ := 60
def cost_per_mask : ℕ := 2

-- Lean statement to prove the relationship
theorem relationship_money_masks (x y : ℕ) (h : y = initial_money - cost_per_mask * x) : y = 60 - 2 * x :=
by
  rw [initial_money, cost_per_mask]
  exact h


end relationship_money_masks_l118_118230


namespace result_l118_118294

noncomputable def Kolya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  P_A = (1: ℝ) / 2

def Valya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  let P_B := (r * p) / (1 - s * q)
  let P_AorB := P_A + P_B
  P_AorB > (1: ℝ) / 2

theorem result (p r : ℝ) (h1 : Kolya_right p r) (h2 : ¬Valya_right p r) :
  Kolya_right p r ∧ ¬Valya_right p r :=
  ⟨h1, h2⟩

end result_l118_118294


namespace tangent_circle_circumference_l118_118508

theorem tangent_circle_circumference (α β γ : Point) (r : ℝ) 
  (h1 : ∠ α β γ = 120) 
  (h2 : arc_length α β γ = 18) :
  ∃ r2 : ℝ, tangent_circle_circumference r2 α β γ = 18 := 
begin
  sorry
end

end tangent_circle_circumference_l118_118508


namespace three_digit_multiples_of_30_not_75_l118_118777

theorem three_digit_multiples_of_30_not_75 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ 30 ∣ n ∧ ¬ (75 ∣ n)}.card = 21 :=
by sorry

end three_digit_multiples_of_30_not_75_l118_118777


namespace probability_nine_heads_l118_118992

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118992


namespace find_hyperbola_eq_l118_118386

-- Definitions based on conditions
def hyperbola_eq (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1
def right_focus (c : ℝ) := (c, 0)
def vertex_B (b : ℝ) := (0, b)
def point_A (a b c : ℝ) := (2*c/3, b/3)
def dist_BF (F B : ℝ × ℝ) : ℝ := real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2)
def vec_BA_eq_2vec_AF (A B F : ℝ × ℝ) : Prop := (A.1 - B.1, A.2 - B.2) = 2 * ((F.1 - A.1), 0)

-- The final lean statement
theorem find_hyperbola_eq (a b c : ℝ) (A B F : ℝ × ℝ)
  (h1 : 0 < a) (h2: 0 < b)
  (h3 : B = vertex_B b)
  (h4 : F = right_focus c)
  (h5 : A = point_A a b c)
  (h6 : vec_BA_eq_2vec_AF A B F)
  (h7 : dist_BF F B = 4) :
  hyperbola_eq 2 (real.sqrt 6) 2  1 :=
sorry

end find_hyperbola_eq_l118_118386


namespace range_of_expression_l118_118155

theorem range_of_expression
  (b c : ℝ)
  (h1 : -4 < b)
  (h2 : b < 0)
  (h3 : 0 < c)
  (h4 : c < 4)
  (h5 : b^2 - 4 * c > 0) :
  ∃ y (y_range : 0 < y ∧ y < 1), y = c^2 + 2 * b * c + 4 * c :=
by
  sorry 

end range_of_expression_l118_118155


namespace almost_perfect_numbers_l118_118895

def num_divisors (n : ℕ) : ℕ := 
  if n = 0 then 0 else 
    (List.range (n + 1)).count (λ d, d > 0 ∧ n % d = 0)

def f (n : ℕ) : ℕ := 
  (List.filter (λ d, d > 0 ∧ n % d = 0) (List.range (n + 1))).map num_divisors |>.sum

theorem almost_perfect_numbers (n : ℕ) (h : n > 1) : 
  f n = n ↔ (n = 3 ∨ n = 18 ∨ n = 36) := sorry

end almost_perfect_numbers_l118_118895


namespace tom_sells_games_for_225_42_usd_l118_118549

theorem tom_sells_games_for_225_42_usd :
  let initial_usd := 200
  let usd_to_eur := 0.85
  let tripled_usd := initial_usd * 3
  let eur_value := tripled_usd * usd_to_eur
  let eur_to_jpy := 130
  let jpy_value := eur_value * eur_to_jpy
  let percent_sold := 0.40
  let sold_jpy_value := jpy_value * percent_sold
  let jpy_to_usd := 0.0085
  let sold_usd_value := sold_jpy_value * jpy_to_usd
  sold_usd_value = 225.42 :=
by
  sorry

end tom_sells_games_for_225_42_usd_l118_118549


namespace hexagon_perimeter_l118_118523

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 5) (h2 : num_sides = 6) : 
  num_sides * side_length = 30 := by
  sorry

end hexagon_perimeter_l118_118523


namespace sum_of_inverses_mod_11_l118_118564

/-- Statement: What is 3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶ mod 11? -/
theorem sum_of_inverses_mod_11 :
  (3⁻¹ + 3⁻² + 3⁻³ + 3⁻⁴ + 3⁻⁵ + 3⁻⁶) % 11 = 4 := (
    sorry
  )

end sum_of_inverses_mod_11_l118_118564


namespace solve_for_x_l118_118181

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (4 * x + 28)) : 
  x = -17 / 5 := 
by 
  sorry

end solve_for_x_l118_118181


namespace sqrt_multiplication_l118_118578

theorem sqrt_multiplication : (real.sqrt 2) * (real.sqrt 3) = real.sqrt 6 :=
by
  sorry

end sqrt_multiplication_l118_118578


namespace moon_speed_kmh_l118_118596

theorem moon_speed_kmh (speed_kms : ℝ) (h : speed_kms = 0.9) : speed_kms * 3600 = 3240 :=
by
  rw [h]
  norm_num

end moon_speed_kmh_l118_118596


namespace starting_time_calculation_l118_118948

def glowing_light_start_time (max_glows : ℚ) (interval : ℚ) (end_time : Time) : Time :=
  let total_glows := max_glows.floor
  let total_seconds := total_glows * interval
  let total_hours := (total_seconds / 3600).toInt
  let remaining_seconds := total_seconds % 3600
  let total_minutes := (remaining_seconds / 60).toInt
  let remaining_final_seconds := remaining_seconds % 60
  let end_hours := end_time.hour - total_hours
  let end_minutes := (end_time.min - total_minutes + 60) % 60
  let end_seconds := end_time.sec - remaining_final_seconds
  Time.mk end_hours end_minutes end_seconds

theorem starting_time_calculation :
  (glowing_light_start_time 236.61904761904762 21 (Time.mk 3 20 47)) = Time.mk 1 58 11 :=
by
  sorry

end starting_time_calculation_l118_118948


namespace exists_infinitely_many_natural_numbers_roots_perfect_squares_l118_118140

theorem exists_infinitely_many_natural_numbers_roots_perfect_squares :
  ∃ᶠ n : ℕ in at_top, 
    let a := 2 - 3 * n ^ 2,
        b := (n ^ 2 - 1) ^ 2, 
        Δ := a ^ 2 - 4 * b,
        r1 := (a + Δ.sqrt) / 2,
        r2 := (a - Δ.sqrt) / 2
    in is_integer r1 ^ 2 ∧ is_integer r2 ^ 2 := 
sorry

end exists_infinitely_many_natural_numbers_roots_perfect_squares_l118_118140


namespace union_sets_l118_118761

noncomputable def M : Set ℤ := {1, 2, 3}
noncomputable def N : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem union_sets : M ∪ N = {0, 1, 2, 3} := by
  sorry

end union_sets_l118_118761


namespace last_two_nonzero_digits_75_factorial_l118_118953

theorem last_two_nonzero_digits_75_factorial : Nat :=
  (75.factorial / 10^18) % 100 = 32 :=
by
  sorry

end last_two_nonzero_digits_75_factorial_l118_118953


namespace larger_of_two_numbers_l118_118560

  theorem larger_of_two_numbers (x y : ℕ) 
    (h₁ : x + y = 37) 
    (h₂ : x - y = 5) 
    : x = 21 :=
  sorry
  
end larger_of_two_numbers_l118_118560


namespace dozens_in_each_box_l118_118543

theorem dozens_in_each_box (boxes total_mangoes : ℕ) (h1 : boxes = 36) (h2 : total_mangoes = 4320) :
  (total_mangoes / 12) / boxes = 10 :=
by
  -- The proof will go here.
  sorry

end dozens_in_each_box_l118_118543


namespace number_of_real_solutions_l118_118359

theorem number_of_real_solutions (f : ℝ → ℝ) : set.countable {x : ℝ | f x = 0} :=
begin
  /-
  Let's define the equation in the problem.
  -/
  let equation := λ x, (2:ℝ)^(2*x + 1) * (8:ℝ)^(2*x + 3) - (16:ℝ)^(3*x + 2) = 0,

  /-
  We now need to find the number of real solutions to this equation
  -/
  sorry
end

end number_of_real_solutions_l118_118359


namespace number_of_crowns_l118_118369

-- Define the conditions
def feathers_per_crown : ℕ := 7
def total_feathers : ℕ := 6538

-- Theorem statement
theorem number_of_crowns : total_feathers / feathers_per_crown = 934 :=
by {
  sorry  -- proof omitted
}

end number_of_crowns_l118_118369


namespace probability_highest_card_is_4_l118_118541

/-- There are five cards in a box, labeled 1, 2, 3, 4, and 5. If three cards are drawn from the box without replacement, the probability that the highest card drawn is 4 is 3/10. -/
theorem probability_highest_card_is_4 :
  let total_ways := Nat.choose 5 3,
      favorable_ways := Nat.choose 3 2 in
  (favorable_ways : ℚ) / total_ways = 3 / 10 :=
by
  let total_ways := Nat.choose 5 3
  let favorable_ways := Nat.choose 3 2
  show (favorable_ways : ℚ) / total_ways = 3 / 10
  sorry

end probability_highest_card_is_4_l118_118541


namespace count_pos_3digit_multiples_of_30_not_75_l118_118794

theorem count_pos_3digit_multiples_of_30_not_75 : 
  let multiples_of_30 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 30 = 0}
  let multiples_of_75 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 75 = 0}
  (multiples_of_30 \ multiples_of_75).size = 24 :=
by
  sorry

end count_pos_3digit_multiples_of_30_not_75_l118_118794


namespace selling_price_is_correct_l118_118257

def cost_price : ℝ := 900
def profit_percentage : ℝ := 60
def selling_price : ℝ := cost_price + (profit_percentage / 100 * cost_price)

theorem selling_price_is_correct : selling_price = 1440 := 
by
  have profit : ℝ := profit_percentage / 100 * cost_price
  have sp : ℝ := cost_price + profit
  show sp = 1440
  -- Proof steps
  sorry

end selling_price_is_correct_l118_118257


namespace problem1_problem2_l118_118883

open Real

variables {α β γ : ℝ}

theorem problem1 (α β : ℝ) :
  abs (cos (α + β)) ≤ abs (cos α) + abs (sin β) ∧
  abs (sin (α + β)) ≤ abs (cos α) + abs (cos β) :=
sorry

theorem problem2 (h : α + β + γ = 0) :
  abs (cos α) + abs (cos β) + abs (cos γ) ≥ 1 :=
sorry

end problem1_problem2_l118_118883


namespace find_a_extremum_and_monotonicity_condition_l118_118019

noncomputable theory 
open Classical

-- Define the function
def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x - 1

-- Define the derivative of the function
def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Define the proof statement
theorem find_a_extremum_and_monotonicity_condition :
  (∃ a : ℝ, f_prime 1 a = 0 ∧ a = 3) ∧ (∀ a : ℝ, (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x a ≤ 0) ↔ a ≥ 3) :=
by {
  { -- Part 1: Prove that a = 3 if x = 1 is an extremum point
    have h1 : ∃ a : ℝ, f_prime 1 a = 0,
    { use 3, rw [f_prime, pow_two], norm_num },
    exact ⟨h1, rfl⟩ },
  { -- Part 2: Prove that a >= 3 ensures f is monotonically decreasing on (-1, 1)
    intro a,
    split,
    { -- (←) direction
      intro h,
      specialize h (-1),
      specialize h 0,
      specialize h 1,
      linarith },
    { -- (→) direction
      intro ha,
      intros x hx,
      specialize ha,
      calc
        f_prime x a = 3 * x^2 - a : by { rw f_prime }
               ... ≤ 3 - a : by { linarith [hx.left, hx.right, pow_two_nonneg x] }
               ... ≤ 0 : by exact sub_nonpos.mpr ha }
  }
}.Sorry   

end find_a_extremum_and_monotonicity_condition_l118_118019


namespace revenue_fall_percentage_is_correct_l118_118236

-- Define the old and new revenues
def old_revenue : ℝ := 72.0
def new_revenue : ℝ := 48.0

-- Define the percentage decrease calculation
def calculate_percentage_decrease (old new : ℝ) : ℝ :=
  ((old - new) / old) * 100

-- The theorem statement
theorem revenue_fall_percentage_is_correct : calculate_percentage_decrease old_revenue new_revenue = 33.33 := 
by
  -- proof is omitted
  sorry

end revenue_fall_percentage_is_correct_l118_118236


namespace find_inverse_l118_118804

theorem find_inverse :
  (∃ x : ℝ, (3 * x^3 + 6 = 87)) ∧ (∀ y : ℝ, (g y = 87 → y = 3)) := 
by 
  -- Define g(x)
  let g : ℝ → ℝ := λ x, 3 * x^3 + 6
  -- Let x be the inverse of g at 87
  have h : ∃ x : ℝ, g x = 87 := 
    by 
      use 3
      simp
      norm_num

  show (∃ x : ℝ, g x = 87),
  from h

-- Sorry as placeholder for proof
sorry

end find_inverse_l118_118804


namespace rocket_travel_time_l118_118283

/-- The rocket's distance formula as an arithmetic series sum.
    We need to prove that the rocket reaches 240 km after 15 seconds
    given the conditions in the problem. -/
theorem rocket_travel_time :
  ∃ n : ℕ, (2 * n + (n * (n - 1))) / 2 = 240 ∧ n = 15 :=
by
  sorry

end rocket_travel_time_l118_118283


namespace sum_tan_inverse_squared_l118_118378

theorem sum_tan_inverse_squared : 
  let x k := (Real.tan (k * Real.pi / 17))
  in (Finset.sum (Finset.range 16) (λ k => 1 / (1 + (x (k+1))^2))) = 15 / 2 := sorry

end sum_tan_inverse_squared_l118_118378


namespace smaller_rectangle_dimensions_l118_118643

def side_length : ℝ := 10
def total_area : ℝ := side_length * side_length
def area_smaller_rect : ℝ := total_area / 3
def dimensions_smaller_rect (x y : ℝ) : Prop := x * y = area_smaller_rect ∧ y = side_length

theorem smaller_rectangle_dimensions : 
  ∃ x y : ℝ, dimensions_smaller_rect x y ∧ x = 10 / 3 ∧ y = 10 :=
by
  sorry

end smaller_rectangle_dimensions_l118_118643


namespace item_A_percentage_gain_item_B_percentage_gain_item_C_percentage_gain_l118_118276

def percentage_increase(P : ℝ) (r : ℝ) := P + (r / 100) * P
def percentage_discount(P : ℝ) (r : ℝ) := P - (r / 100) * P
def percentage_tax(P : ℝ) (r : ℝ) := P + (r / 100) * P
def overall_percentage_gain(original : ℝ) (final : ℝ) := ((final - original) / original) * 100

noncomputable def item_A_final_price :=
  let P := 100
  let increased_price := percentage_increase P 31.5
  let first_discounted_price := percentage_discount increased_price 12.5
  let second_discounted_price := percentage_discount first_discounted_price 17.3
  percentage_tax second_discounted_price 7.5

noncomputable def item_B_final_price :=
  let P := 100
  let increased_price := percentage_increase P 27.8
  let first_discounted_price := percentage_discount increased_price 9.6
  let second_discounted_price := percentage_discount first_discounted_price 14.7
  percentage_tax second_discounted_price 7.5

noncomputable def item_C_final_price :=
  let P := 100
  let increased_price := percentage_increase P 33.1
  let first_discounted_price := percentage_discount increased_price 11.7
  let second_discounted_price := percentage_discount first_discounted_price 13.9
  percentage_tax second_discounted_price 7.5

theorem item_A_percentage_gain :
  overall_percentage_gain 100 item_A_final_price = 2.2821 := 
  by
    sorry

theorem item_B_percentage_gain :
  overall_percentage_gain 100 item_B_final_price = 5.939 := 
  by
    sorry

theorem item_C_percentage_gain :
  overall_percentage_gain 100 item_C_final_price = 8.7696 := 
  by
    sorry

end item_A_percentage_gain_item_B_percentage_gain_item_C_percentage_gain_l118_118276


namespace part1_part2_l118_118393

variables {x m : ℝ}

-- Conditions
def p (x : ℝ) : Prop := x^2 - 4 * x - 5 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def m_gt_zero (m : ℝ) : Prop := m > 0

-- Equivalent proof problem:
theorem part1 (h_m_gt_zero : m_gt_zero m) (h_subset : ∀ x, p x → q x m) : m ∈ set.Ici 4 :=
sorry

theorem part2 (h_m_eq_five : m = 5) (h_p_or_q : ∃ x, p x ∨ q x m) (h_not_p_and_q : ∀ x, ¬ (p x ∧ q x m)) : 
  ∃ x, x ∈ set.Icc (-4) (-1) ∪ set.Icc 5 6 :=
sorry

end part1_part2_l118_118393


namespace find_initial_numbers_l118_118198

def initialNumbers (a b : ℕ) : Prop :=
  a > b ∧ a = 1580 ∧ ∃ k : ℕ, 1580 - b = 1024 * k

theorem find_initial_numbers :
  ∀ a b : ℕ, initialNumbers a b →
  ∀ n : ℕ, n = 10 →
  b = 556 :=
by
  intros a b h n hn
  cases h with ha hb
  rw hb at *
  sorry

end find_initial_numbers_l118_118198


namespace final_value_after_operations_l118_118428

theorem final_value_after_operations (initial_value : ℝ) (increase_rate decrease_rate : ℝ) 
    (h_initial : initial_value = 1500) (h_increase : increase_rate = 0.20) (h_decrease : decrease_rate = 0.40) :
    let increased_value := initial_value * (1 + increase_rate)
    let final_value := increased_value * (1 - decrease_rate)
    final_value = 1080 := 
by 
  let increased_value := initial_value * (1 + increase_rate)
  let final_value := increased_value * (1 - decrease_rate)
  rw [h_initial, h_increase, h_decrease]
  norm_num
  sorry

end final_value_after_operations_l118_118428


namespace planes_through_three_points_l118_118969

def point := ℝ × ℝ × ℝ -- Definition of a point in 3D space.

theorem planes_through_three_points (A B C : point) : 
  ¬ collinear A B C → ∃! P : set point, is_plane P ∧ A ∈ P ∧ B ∈ P ∧ C ∈ P
  ∧ collinear A B C → ∃(S : set (set point)), infinite S ∧ ∀ P ∈ S, is_plane P ∧ A ∈ P ∧ B ∈ P ∧ C ∈ P :=
sorry

-- Definitions required for the theorem
def collinear (A B C : point) : Prop := 
  ∃ (λ : ℝ), ∃ (μ : ℝ), ∃ (ν : ℝ), A = (λ * B.1 + μ * C.1, λ * B.2 + μ * C.2, λ * B.3 + μ * C.3)

def is_plane (P : set point) : Prop := 
  ∃ (n : point) (c : ℝ), ∀ (x : point), x ∈ P ↔ n.1 * x.1 + n.2 * x.2 + n.3 * x.3 = c

end planes_through_three_points_l118_118969


namespace share_of_C_l118_118603

theorem share_of_C
  (total_payment : ℝ)
  (A_work : ℝ)
  (B_work : ℝ)
  (combined_with_C : ℝ)
  (A_share : ℝ)
  (B_share : ℝ) :
  (total_payment = 600) →
  (A_work = 6) →
  (B_work = 8) →
  (combined_with_C = 3) →
  (A_share = 300) →
  (B_share = 225) →
  (total_payment - (A_share + B_share) = 75) :=
begin
  intros h1 h2 h3 h4 h5 h6,
  -- additional steps showing the proof, skipping here
  sorry
end

end share_of_C_l118_118603


namespace highest_coal_cost_point_at_equal_cost_l118_118980

structure Location where
  distance : ℕ -- distance in kilometers
  price : ℝ -- price in korona

structure Transport where
  cost_per_km_per_ton : ℝ -- cost per kilometer per ton in korona

noncomputable def coal_cost (location : Location) (transport : Transport) (distance_from_location : ℝ) : ℝ :=
  location.price + transport.cost_per_km_per_ton * distance_from_location

noncomputable def equal_coal_cost_point (A B : Location) (transport : Transport) : ℝ :=
  let total_distance := B.distance
  let x := (B.price - A.price + transport.cost_per_km_per_ton * total_distance) / (2 * transport.cost_per_km_per_ton)
  x

theorem highest_coal_cost_point_at_equal_cost (A B : Location) (transport : Transport) :
  A.distance = 0 → B.distance = 225 → A.price = 3.75 → B.price = 4.25 → transport.cost_per_km_per_ton = 0.008 →
  let x := equal_coal_cost_point A B transport in
  coal_cost A transport x = coal_cost B transport (B.distance - x) ∧ 
  (∀ y, y ≠ x → coal_cost A transport y < coal_cost A transport x) :=
by
  assume hA hB hPA hPB hT
  sorry

end highest_coal_cost_point_at_equal_cost_l118_118980


namespace theta_in_second_quadrant_l118_118375

theorem theta_in_second_quadrant (θ : ℝ) : 
  sin (π / 2 + θ) < 0 ∧ tan (π - θ) > 0 → θ ∈ set.Ioo (π / 2) π :=
by
  sorry

end theta_in_second_quadrant_l118_118375


namespace circle_radius_tangent_lines_l118_118252

theorem circle_radius_tangent_lines {k : ℝ} (h : k > 8) :
  (∃ r : ℝ, 
    (∀ x : ℝ, (x, x) ∈ {(x, y) | y = x} → dist (0, k) (x, x) = r * real.sqrt 2) ∧ 
    (∀ x : ℝ, (x, -x) ∈ {(x, y) | y = -x} → dist (0, k) (x, -x) = r * real.sqrt 2) ∧ 
    dist (0, k) (0, 8) = r + (k - 8)) :=
  r = k * real.sqrt 2 + k - 8 * real.sqrt 2 - 8 :=
sorry

end circle_radius_tangent_lines_l118_118252


namespace find_725th_digit_l118_118226

noncomputable def decimal_expansion : ℕ → ℕ
| 0 := 2
| 1 := 4
| 2 := 1
| 3 := 3
| 4 := 7
| 5 := 9
| 6 := 3
| 7 := 1
| 8 := 0
| 9 := 3
| 10 := 4
| 11 := 4
| 12 := 8
| 13 := 2
| 14 := 7
| 15 := 5
| 16 := 8
| 17 := 6
| 18 := 2
| 19 := 0
| 20 := 6
| 21 := 8
| 22 := 9
| 23 := 5
| 24 := 5
| 25 := 1
| 26 := 7
| 27 := 8
| 28 := 6
| (n + 29) := decimal_expansion n

theorem find_725th_digit :
  decimal_expansion 20 = 8 :=
by
  sorry

end find_725th_digit_l118_118226


namespace right_angles_in_2_days_l118_118593

-- Definitions
def hands_right_angle_twice_a_day (n : ℕ) : Prop :=
  n = 22

def right_angle_12_hour_frequency : Nat := 22
def hours_per_day : Nat := 24
def days : Nat := 2

-- Theorem to prove
theorem right_angles_in_2_days :
  hands_right_angle_twice_a_day right_angle_12_hour_frequency →
  right_angle_12_hour_frequency * (hours_per_day / 12) * days = 88 :=
by
  unfold hands_right_angle_twice_a_day
  intros 
  sorry

end right_angles_in_2_days_l118_118593


namespace find_a_b_sum_l118_118707

theorem find_a_b_sum (a b : ℝ) :
  (∀ x : ℝ, f x = 
    if x < 3 then a * x + b 
    else 7 - 2 * x) ∧ 
  (∀ x : ℝ, f (f x) = x) →
  a + b = 4 := 
sorry

end find_a_b_sum_l118_118707


namespace solve_linear_system_l118_118965

theorem solve_linear_system :
  ∃ (x y : ℝ), (x + 3 * y = -1) ∧ (2 * x + y = 3) ∧ (x = 2) ∧ (y = -1) :=
  sorry

end solve_linear_system_l118_118965


namespace steve_halfway_time_longer_than_danny_l118_118327

theorem steve_halfway_time_longer_than_danny 
  (T_D : ℝ) (T_S : ℝ)
  (h1 : T_D = 33) 
  (h2 : T_S = 2 * T_D):
  (T_S / 2) - (T_D / 2) = 16.5 :=
by sorry

end steve_halfway_time_longer_than_danny_l118_118327


namespace probability_more_wins_l118_118663

theorem probability_more_wins (m n : ℕ) (rel_prime : Nat.gcd 103 486 = 1) (103 + 486 = 589) :
  let total_matches := 5
  let probability_each_outcome := 1/3
  let wins := λ k : ℕ, probability_each_outcome ^ k
  let losses := λ k : ℕ, probability_each_outcome ^ k
  let ties := λ k : ℕ, probability_each_outcome ^ k
  ∑ _ in Finset.range total_matches, wins k + losses k + ties k = m / n ∧ Nat.gcd m n = 1 ∧ m + n = 589 :=
sorry

end probability_more_wins_l118_118663


namespace product_of_sums_l118_118333

theorem product_of_sums (n : ℕ) : 
  ((∑ k in Finset.range (n + 1), (2 * k + 1)) * (∑ j in Finset.range (n), 2 * (j + 1))) = (n + 1)^3 * n :=
by
  -- The proof goes here.
  sorry

end product_of_sums_l118_118333


namespace probability_nine_heads_l118_118997

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118997


namespace find_vet_fees_for_dogs_l118_118287

-- Conditions
def vet_fees_dogs : ℕ := 8 * D
def vet_fees_cats : ℕ := 3 * 13
def total_vet_fees := vet_fees_dogs + vet_fees_cats
def vet_donated : ℕ := 53
def total_donated := 3 * vet_donated

-- Stating the problem
theorem find_vet_fees_for_dogs (D : ℕ) (h : total_vet_fees = total_donated) :
  D = 15 :=
by
  have eq1 : total_vet_fees = 8 * D + 3 * 13 := rfl
  have eq2 : total_donated = 3 * 53 := rfl
  rw [eq1, eq2, ←add_assoc] at h
  linarith


end find_vet_fees_for_dogs_l118_118287


namespace p_correct_l118_118700

noncomputable def p : ℝ → ℝ := sorry

axiom p_at_3 : p 3 = 10

axiom p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2

theorem p_correct : ∀ x, p x = x^2 + 1 :=
sorry

end p_correct_l118_118700


namespace limit_fx_x_squared_equals_f_prime_0_l118_118024

open Filter

noncomputable def f (x : ℝ) : ℝ := x^2

theorem limit_fx_x_squared_equals_f_prime_0 :
  tendsto (λ (Δx : ℝ), (f Δx - f 0) / Δx) (𝓝 0) (𝓝 0) :=
sorry

end limit_fx_x_squared_equals_f_prime_0_l118_118024


namespace boats_left_l118_118128

theorem boats_left (total_boats : ℕ) (percentage_eaten : ℝ) (boats_shot : ℕ) 
  (h1 : total_boats = 30) (h2 : percentage_eaten = 0.20) (h3 : boats_shot = 2) : 
  total_boats - (total_boats * percentage_eaten).toNat - boats_shot = 22 :=
by
  sorry

end boats_left_l118_118128


namespace Kolya_correct_Valya_incorrect_l118_118299

-- Kolya's Claim (Part a)
theorem Kolya_correct (x : ℝ) (p r : ℝ) (hpr : r = 1/(x+1) ∧ p = 1/x) : 
  (p / (1 - (1 - r) * (1 - p))) = (r / (1 - (1 - r) * (1 - p))) :=
sorry

-- Valya's Claim (Part b)
theorem Valya_incorrect (x : ℝ) (p r : ℝ) (q s : ℝ) (hprs : r = 1/(x+1) ∧ p = 1/x ∧ q = 1 - p ∧ s = 1 - r) : 
  ((q * r / (1 - s * q)) + (p * r / (1 - s * q))) = 1/2 :=
sorry

end Kolya_correct_Valya_incorrect_l118_118299


namespace octagon_mass_is_19kg_l118_118712

-- Define the parameters given in the problem
def side_length_square_sheet := 1  -- side length in meters
def thickness_sheet := 0.3  -- thickness in cm (3 mm)
def density_steel := 7.8  -- density in g/cm³

-- Given the geometric transformations and constants, prove the mass of the octagon
theorem octagon_mass_is_19kg :
  ∃ mass : ℝ, (mass = 19) :=
by
  -- Placeholder for the proof.
  -- The detailed steps would include geometrical transformations and volume calculations,
  -- which have been rigorously defined in the problem and derived in the solution.
  sorry

end octagon_mass_is_19kg_l118_118712


namespace sqrt_uniform_continuous_sin_uniform_continuous_l118_118893

-- Definition of uniform continuity
def uniform_continuous {α β : Type*} [MetricSpace α] [MetricSpace β] (f : α → β) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₁ x₂ : α, dist x₁ x₂ < δ → dist (f x₁) (f x₂) < ε

-- Define the real interval [1, +∞)
def interval1 : Set ℝ := {x | 1 ≤ x}

-- Prove that f(x) = sqrt(x) is uniformly continuous on [1, +∞)
theorem sqrt_uniform_continuous : uniform_continuous (λ x : ℝ, Real.sqrt x) :=
sorry

-- Prove that f(x) = sin(x) is uniformly continuous on (-∞, +∞)
theorem sin_uniform_continuous : uniform_continuous Real.sin :=
sorry

end sqrt_uniform_continuous_sin_uniform_continuous_l118_118893


namespace part_1_part_2_l118_118093

variables {A B C D : Type}
variables {a b c : ℝ} -- Side lengths of the triangle
variables {A_angle B_angle C_angle : ℝ} -- Angles of the triangle
variables {R : ℝ} -- Circumradius of the triangle

-- Assuming the given conditions:
axiom b_squared_eq_ac : b^2 = a * c
axiom bd_sin_eq_a_sin_C : ∀ {BD : ℝ}, BD * sin B_angle = a * sin C_angle
axiom ad_eq_2dc : ∀ {AD DC : ℝ}, AD = 2 * DC

-- Theorems to prove:
theorem part_1 (BD : ℝ) : BD * sin B_angle = a * sin C_angle → BD = b := by
  intros h
  sorry

theorem part_2 (AD DC : ℝ) (H : AD = 2 * DC) : cos B_angle = 7 / 12 := by
  intros h
  sorry

end part_1_part_2_l118_118093


namespace rectangle_area_of_congruent_circles_l118_118547

-- Given conditions
def congruent_circles (P Q R : Point) (C QD : Circle) : Prop := 
  (∃ d : ℝ, d = 6) ∧ (QD.center = Q) ∧ (QD.diameter = 6) ∧ 
  tangent_to_sides P Q R

-- Definition of tangent_to_sides to express that the circles are tangent to the sides of the rectangle
def tangent_to_sides (P Q R : Point) : Prop := 
  (tangent_to_side P) ∧ (tangent_to_side Q) ∧ (tangent_to_side R)

variables {A B C D P Q R : Point}
variables {rect : Rectangle A B C D}

-- Problem statement in Lean
theorem rectangle_area_of_congruent_circles :
  congruent_circles P Q R (circle Q) →
  tangent_to_sides P Q R →
  rect.height = 6 →
  rect.width  = 12 →
  rect.area = 72 :=
by
  sorry

end rectangle_area_of_congruent_circles_l118_118547


namespace three_digit_multiples_of_30_not_75_l118_118775

theorem three_digit_multiples_of_30_not_75 : 
  let multiples_of_30 := list.range' 120 (990 - 120 + 30) 30
  let multiples_of_75 := list.range' 150 (900 - 150 + 150) 150
  let non_multiples_of_75 := multiples_of_30.filter (λ x => x % 75 ≠ 0)
  non_multiples_of_75.length = 24 := 
by 
  sorry

end three_digit_multiples_of_30_not_75_l118_118775


namespace boats_left_l118_118127

theorem boats_left (total_boats : ℕ) (percentage_eaten : ℝ) (boats_shot : ℕ) 
  (h1 : total_boats = 30) (h2 : percentage_eaten = 0.20) (h3 : boats_shot = 2) : 
  total_boats - (total_boats * percentage_eaten).toNat - boats_shot = 22 :=
by
  sorry

end boats_left_l118_118127


namespace possible_values_f_zero_l118_118325

theorem possible_values_f_zero (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = 2 * f x * f y) :
    f 0 = 0 ∨ f 0 = 1 / 2 := 
sorry

end possible_values_f_zero_l118_118325


namespace collinearity_of_MNP_l118_118851

-- Define the geometrical setup
variables {A B C D M N P : Point}
variables (AD BC BD : Line)
variables (h1 : Perpendicular AB AD)
variables (h2 : Perpendicular BC CD)
variables (h3 : OnLine M AD)
variables (h4 : OnLine N BC)
variables (h5 : Perpendicular CM BD)
variables (h6 : Perpendicular AN BD)
variables (h7 : Intersection AC BD P)

-- The problem statement
theorem collinearity_of_MNP :
  Collinear M P N :=
sorry

end collinearity_of_MNP_l118_118851


namespace distance_from_P_to_AD_l118_118922

theorem distance_from_P_to_AD : 
  let A := (0, 5)
  let D := (0, 0)
  let M := (2.5, 0)
  ∃ P : ℝ × ℝ, (x - 2.5)^2 + y^2 = 6.25 ∧ x^2 + (y - 5)^2 = 25 ∧ 
               (P = (25/4, (0.5 * (25/4)) - 0.625)) → 
               dist (P : ℝ) 0 = 25/4 :=
by
  sorry

end distance_from_P_to_AD_l118_118922


namespace distance_from_P_to_AD_l118_118923

theorem distance_from_P_to_AD : 
  let A := (0, 5)
  let D := (0, 0)
  let M := (2.5, 0)
  ∃ P : ℝ × ℝ, (x - 2.5)^2 + y^2 = 6.25 ∧ x^2 + (y - 5)^2 = 25 ∧ 
               (P = (25/4, (0.5 * (25/4)) - 0.625)) → 
               dist (P : ℝ) 0 = 25/4 :=
by
  sorry

end distance_from_P_to_AD_l118_118923


namespace solution_set_of_inequality_l118_118742

noncomputable theory
open Real

def strictly_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f y < f x

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (hf_domain : ∀ x, f x ≠ 0)
  (hf_decreasing : strictly_decreasing_on f (Ioi 2))
  (hf_even : even_function (λ x, f (x + 2)))
  (x : ℝ) :
  (f (2 * x - 1) - f (x + 1) > 0) ↔ (4/3 < x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l118_118742


namespace smallest_y_l118_118568

theorem smallest_y (y : ℝ) :
  (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) → y = -10 :=
sorry

end smallest_y_l118_118568


namespace stream_speed_calculation_l118_118248

/-- The speed of the boat in still water is 13 km/hr -/
def boat_speed_still_water : ℝ := 13

/-- The distance travelled downstream is 69 km -/
def distance_downstream : ℝ := 69

/-- The time taken to travel downstream is 3.6315789473684212 hours -/
def time_downstream : ℝ := 3.6315789473684212

/-- The speed of the boat downstream is the sum of its speed in still water
    and the speed of the stream -/
def speed_downstream (stream_speed : ℝ) : ℝ :=
  boat_speed_still_water + stream_speed

/-- The downstream speed can be calculated from distance and time -/
def calc_speed_downstream : ℝ :=
  distance_downstream / time_downstream

theorem stream_speed_calculation : 
  calc_speed_downstream - boat_speed_still_water = 6 :=
by
  sorry

end stream_speed_calculation_l118_118248


namespace pages_for_35_dollars_l118_118859

theorem pages_for_35_dollars (cost_per_pages : ℕ) (pages_per_cost : ℕ) (total_cost : ℕ) :
  cost_per_pages = 7 →  
  pages_per_cost = 5 → 
  total_cost = 3500 →
  total_cost * pages_per_cost = cost_per_pages * 2500 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end pages_for_35_dollars_l118_118859


namespace probability_nine_heads_l118_118985

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118985


namespace sqrt_mul_eq_l118_118576

theorem sqrt_mul_eq {a b: ℝ} (ha: 0 ≤ a) (hb: 0 ≤ b): sqrt(a * b) = sqrt(a) * sqrt(b) :=
  by sorry

example : sqrt(2) * sqrt(3) = sqrt(6) :=
  by apply sqrt_mul_eq; norm_num

end sqrt_mul_eq_l118_118576


namespace total_students_is_100_l118_118838

-- Definitions of the conditions
def largest_class_students : Nat := 24
def decrement : Nat := 2

-- Let n be the number of classes, which is given by 5
def num_classes : Nat := 5

-- The number of students in each class
def students_in_class (n : Nat) : Nat := 
  if n = 1 then largest_class_students
  else largest_class_students - decrement * (n - 1)

-- Total number of students in the school
def total_students : Nat :=
  List.sum (List.map students_in_class (List.range num_classes))

-- Theorem to prove that total_students equals 100
theorem total_students_is_100 : total_students = 100 := by
  sorry

end total_students_is_100_l118_118838


namespace find_angle_B_in_right_triangle_l118_118068

theorem find_angle_B_in_right_triangle (A B C : ℝ) (hC : C = 90) (hA : A = 35) :
  B = 55 :=
by
  -- Assuming A, B, and C represent the three angles of a triangle ABC
  -- where C = 90 degrees and A = 35 degrees, we need to prove B = 55 degrees.
  sorry

end find_angle_B_in_right_triangle_l118_118068


namespace problem_1_problem_2_problem_3_l118_118003

-- Define the sequence and its properties
def seq (a : ℕ) (n : ℕ) : ℕ :=
  if n = 1 then a else
  if seq a (n - 1) <= 18 then 2 * seq a (n - 1) else 2 * seq a (n - 1) - 36

-- Define the set M
def M (a : ℕ) : set ℕ := {b | ∃ n, b = seq a n}

-- Problem (1)
theorem problem_1 (a1 : ℕ) (h1 : a1 = 6) : M a1 = {6, 12, 24} := sorry

-- Problem (2)
theorem problem_2 (a1 : ℕ) (h1 : a1 ∈ ℕ) :
  (∃ x ∈ M a1, x % 3 = 0) → ∀ x ∈ M a1, x % 3 = 0 := sorry

-- Problem (3)
theorem problem_3 (a1 : ℕ) (h1 : a1 ∈ ℕ) (h2 : a1 ≤ 36) :
  ∃ max_elements : ℕ, max_elements = 8 := sorry

end problem_1_problem_2_problem_3_l118_118003


namespace angles_of_tangency_triangle_l118_118551

theorem angles_of_tangency_triangle 
  (A B C : ℝ) 
  (ha : A = 40)
  (hb : B = 80)
  (hc : C = 180 - A - B)
  (a1 b1 c1 : ℝ)
  (ha1 : a1 = (1/2) * (180 - A))
  (hb1 : b1 = (1/2) * (180 - B))
  (hc1 : c1 = 180 - a1 - b1) :
  (a1 = 70 ∧ b1 = 50 ∧ c1 = 60) :=
by sorry

end angles_of_tangency_triangle_l118_118551


namespace slower_whale_speed_is_15_l118_118204

/-- Defining problem conditions -/
variables (v : ℝ) (time_cross : ℝ := 15) (length_slower : ℝ := 45) (speed_faster : ℝ := 18)

-- Define relative speed of the faster whale with respect to the slower whale
def relative_speed (v : ℝ) : ℝ := speed_faster - v

-- The main theorem stating that the speed of the slower whale is 15 mps under given conditions
theorem slower_whale_speed_is_15 (h : length_slower = relative_speed v * time_cross) : v = 15 :=
sorry

end slower_whale_speed_is_15_l118_118204


namespace proportion_of_students_in_height_range_needs_frequency_distribution_l118_118193

theorem proportion_of_students_in_height_range_needs_frequency_distribution 
  (students : Type) [fintype students] (height : students → ℝ) :
  (∃ freq_dist : ℝ → ℕ, 
  ∀ range : set ℝ, proportion_within_range (height '' (set.univ : set students)) range = freq_dist range) →
  (∀ range : set ℝ, necessary_info range = frequency_distribution) :=
by
  -- the proof goes here
  sorry

end proportion_of_students_in_height_range_needs_frequency_distribution_l118_118193


namespace length_PF_is_8_l118_118029
noncomputable def find_length_PF : ℝ :=
  let parabola := ∀ x y, y^2 = 8 * x in
  let F := (2, 0) in
  let directrix := -2 in
  let P := (6, 4 * Real.sqrt 3) in
  let A := (directrix, 4 * Real.sqrt 3) in
  let slope_AF := -Real.sqrt 3 in
  Real.sqrt ((6 - 2)^2 + (4 * Real.sqrt 3 - 0)^2)

theorem length_PF_is_8 : find_length_PF = 8 :=
sorry

end length_PF_is_8_l118_118029


namespace quadratic_roots_distinct_real_l118_118963

theorem quadratic_roots_distinct_real 
  (a b c: ℝ) 
  (h_eq_a: a = 1) (h_eq_b: b = 2023) (h_eq_c: c = 2035) 
  (h_eq_q: ∀ x, a * x^2 + b * x + c = 0):
  let Δ := b^2 - 4 * a * c in Δ > 0 :=
by
  sorry

end quadratic_roots_distinct_real_l118_118963


namespace prob_no_distinct_roots_l118_118475

-- Definition of integers a, b, c between -7 and 7
def valid_range (n : Int) : Prop := -7 ≤ n ∧ n ≤ 7

-- Definition of the discriminant condition for non-distinct real roots
def no_distinct_real_roots (a b c : Int) : Prop := b * b - 4 * a * c ≤ 0

-- Counting total triplets (a, b, c) with valid range
def total_triplets : Int := 15 * 15 * 15

-- Counting valid triplets with no distinct real roots
def valid_triplets : Int := 225 + (3150 / 2) -- 225 when a = 0 and estimation for a ≠ 0

theorem prob_no_distinct_roots : 
  let P := valid_triplets / total_triplets 
  P = (604 / 1125 : Rat) := 
by
  sorry

end prob_no_distinct_roots_l118_118475


namespace three_digit_multiples_of_30_not_75_l118_118780

theorem three_digit_multiples_of_30_not_75 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ 30 ∣ n ∧ ¬ (75 ∣ n)}.card = 21 :=
by sorry

end three_digit_multiples_of_30_not_75_l118_118780


namespace common_tangent_l118_118882

-- Definitions as per problem conditions
variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α] 
variables (A B C D : α) (Y : set α) (Y₁ Y₂ Y₃ : set (set α))

-- Semicircular arc Y with diameter AB, point C on Y distinct from A and B
def is_semi_circular_arc (Y : set α) (A B : α) : Prop := 
  -- (Y being defined as a semicircular arc with diameter AB, needs further geometric constraints)

def on_arc (C : α) (Y : set α) (A B : α) : Prop := 
  is_semi_circular_arc Y A B ∧ C ∈ Y ∧ C ≠ A ∧ C ≠ B

-- D is the foot of the perpendicular from C to AB
def is_perpendicular_foot (C D : α) (A B : α) : Prop := 
  -- (D ensuring perpendicularity from C to line segment AB, needs further geometric constraints incorporating orthogonality)

-- Y₁ is the incircle of triangle ABC, Y₂, Y₃ tangent to CD, the semicircular arc Y, and AB
def is_tangent_incircle (Y₁ : set (set α)) (A B C : α) : Prop := 
  -- (Y₁ being defined as the incircle of ΔABC, needs further geometric constraints)

def is_tangent_to_common_elements (Y₂ Y₃ : set (set α)) (CD Y AB : set α) : Prop := 
  -- (Y₂, Y₃ being defined as tangent to CD, semicircle arc Y, and AB, needs further geometric constraints)

-- The actual theorem to be proven
theorem common_tangent {A B C D : α} {Y : set α} {Y₁ Y₂ Y₃ : set (set α)}
  (hY : is_semi_circular_arc Y A B)
  (hC : on_arc C Y A B)
  (hD : is_perpendicular_foot C D A B)
  (hY₁ : is_tangent_incircle Y₁ A B C)
  (hY₂Y₃ : is_tangent_to_common_elements Y₂ Y₃ {x | x ∈ (line_through C D)} Y (line_through A B)) :
  ∃ T : set (set α), T ∈ {x | x ∈ Y₁ ∧ x ∈ Y₂ ∧ x ∈ Y₃} :=
sorry

end common_tangent_l118_118882


namespace serial_number_with_smallest_magnitude_is_1001_or_1002_l118_118389

noncomputable def find_serial_number_with_smallest_magnitude : Nat :=
  let sequence := λ (n : Nat), (n - 2015, n + 12)
  let magnitude := λ (n : Nat), Real.sqrt ((n - 2015)^2 + (n + 12)^2)
  if magnitude 1001 ≤ magnitude 1002 then 1001 else 1002

theorem serial_number_with_smallest_magnitude_is_1001_or_1002 :
  find_serial_number_with_smallest_magnitude = 1001 ∨ find_serial_number_with_smallest_magnitude = 1002 :=
  by
    sorry

end serial_number_with_smallest_magnitude_is_1001_or_1002_l118_118389


namespace find_value_of_expression_l118_118588

theorem find_value_of_expression (a b c d : ℤ) (h₁ : a = -1) (h₂ : b + c = 0) (h₃ : abs d = 2) :
  4 * a + (b + c) - abs (3 * d) = -10 := by
  sorry

end find_value_of_expression_l118_118588


namespace total_matches_l118_118097

-- Definitions from the problem conditions
def boxes : Nat := 5 * 12
def matchesPerBox : Nat := 20

-- Statement to prove the total matches
theorem total_matches : (boxes = 5 * 12) ∧ (matchesPerBox = 20) → boxes * matchesPerBox = 1200 :=
by
  intros h
  rw [h.left, h.right]
  sorry

end total_matches_l118_118097


namespace find_larger_number_l118_118556

theorem find_larger_number 
  (x y : ℤ)
  (h1 : x + y = 37)
  (h2 : x - y = 5) : max x y = 21 := 
sorry

end find_larger_number_l118_118556


namespace all_triangles_similar_and_smallest_angle_15_l118_118365

open_locale classical

variables {A B C X Y Z : Type}
variable (T : triangle A B C)
variables (AX AB CX CY YX YZ CA : ℝ)
variable (X_on_AB : X ∈ segment A B)
variable (Y_on_CX : Y ∈ segment C X)
variable (Z_on_CA : Z ∈ ray C A)
variable (angle_XYZ_45 : ∠ XYZ = 45)
variable (AX_ratio : AX / AB = 4 / 5)
variable (CY_ratio : CY / YX = 2)
variable (angle_CXZ_ABC : ∠ CXZ = 180 - ∠ ABC)

theorem all_triangles_similar_and_smallest_angle_15 :
  (∀ T ∈ Σ, similar T) ∧ (smallest_angle T = 15) :=
begin
  sorry
end

end all_triangles_similar_and_smallest_angle_15_l118_118365


namespace kolya_correct_valya_incorrect_l118_118311

-- Definitions from the problem conditions:
def hitting_probability_valya (x : ℝ) : ℝ := 1 / (x + 1)
def hitting_probability_kolya (x : ℝ) : ℝ := 1 / x
def missing_probability_kolya (x : ℝ) : ℝ := 1 - (1 / x)
def missing_probability_valya (x : ℝ) : ℝ := 1 - (1 / (x + 1))

-- Proof for Kolya's assertion
theorem kolya_correct (x : ℝ) (hx : x > 0) : 
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2) :=
sorry

-- Proof for Valya's assertion
theorem valya_incorrect (x : ℝ) (hx : x > 0) :
  let r := hitting_probability_valya x,
      p := hitting_probability_kolya x,
      s := missing_probability_valya x,
      q := missing_probability_kolya x in
  (r / (1 - s * q) = 1 / 2 ∧ (r + p - r * p) / (1 - s * q) = 1 / 2) :=
sorry

end kolya_correct_valya_incorrect_l118_118311


namespace swan_percentage_not_ducks_l118_118866

theorem swan_percentage_not_ducks (total_birds geese swans herons ducks : ℝ)
  (h_total : total_birds = 100)
  (h_geese : geese = 0.30 * total_birds)
  (h_swans : swans = 0.20 * total_birds)
  (h_herons : herons = 0.20 * total_birds)
  (h_ducks : ducks = 0.30 * total_birds) :
  (swans / (total_birds - ducks) * 100) = 28.57 :=
by
  sorry

end swan_percentage_not_ducks_l118_118866


namespace while_loop_final_value_l118_118919

theorem while_loop_final_value :
  let S := 0
  let i := 0
  let ⟨S, i⟩ := 
    Nat.iterate 4 
    (λ ⟨S, i⟩, ⟨S + i, i^2 + 1⟩) 
    (S, i)
  S = 8 := by
  sorry

end while_loop_final_value_l118_118919


namespace arithmetic_seq_max_sum_l118_118726

noncomputable def max_arith_seq_sum_lemma (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_seq_max_sum :
  ∀ (a1 d : ℤ),
    (3 * a1 + 6 * d = 9) →
    (a1 + 5 * d = -9) →
    max_arith_seq_sum_lemma a1 d 3 = 21 :=
by
  sorry

end arithmetic_seq_max_sum_l118_118726


namespace count_six_digit_even_numbers_is_312_count_greater_than_102345_is_599_l118_118212

-- Lean definitions for first problem
noncomputable def count_six_digit_even_numbers : ℕ :=
  let digits := {0, 1, 2, 3, 4, 5}
  let case1 := Nat.factorial 5      -- Last digit is 0
  let case2 := 2 * 4 * Nat.factorial 4 -- Last digit is 2 or 4
  case1 + case2

-- Lean theorem statement for the first problem
theorem count_six_digit_even_numbers_is_312 : count_six_digit_even_numbers = 312 :=
  sorry

-- Lean definitions for second problem
noncomputable def count_greater_than_102345 : ℕ :=
  let total_permutations := 5 * Nat.factorial 5
  total_permutations - 1

-- Lean theorem statement for the second problem
theorem count_greater_than_102345_is_599 : count_greater_than_102345 = 599 :=
  sorry

end count_six_digit_even_numbers_is_312_count_greater_than_102345_is_599_l118_118212


namespace min_positive_value_of_sum_l118_118340

open nat

theorem min_positive_value_of_sum :
  ∃ (a : fin 50 → ℤ), 
  (∀ i, a i = 1 ∨ a i = -1) ∧ 
  (∀ (i j : fin 50), i < j → 0 < ∑ i in finset.range 50, ∑ j in finset.Ico (i + 1) 50, (a i * a j)) ∧
  (∑ i in finset.range 50, ∑ j in finset.Ico (i + 1) 50, (a i * a j) = 7) :=
by {
  -- Proof goes here
  sorry
}

end min_positive_value_of_sum_l118_118340


namespace sum_distinct_values_of_squares_l118_118507

theorem sum_distinct_values_of_squares (x y z : ℕ)
    (hx : x + y + z = 27)
    (hg : Int.gcd x y + Int.gcd y z + Int.gcd z x = 7) :
    (x^2 + y^2 + z^2 = 574) :=
sorry

end sum_distinct_values_of_squares_l118_118507


namespace urn_problem_probability_l118_118654

theorem urn_problem_probability :
  let initial_red := 2,
      initial_blue := 2,
      total_iterations := 5,
      final_red := 3,
      final_blue := 6,
      total_final_balls := initial_red + initial_blue + total_iterations in
  ∀ (draw_replace_operation : (ℕ × ℕ × ℕ) → (ℕ × ℕ × ℕ) → ℕ → (ℕ × ℕ)),
    -- Assume a drawing operation which takes current counts of red, blue in urn and ball,
    -- and returns the updated urn counts after drawing and replacing a matching ball.
    (draw_replace_operation (initial_red, initial_blue, total_iterations) 
                             (final_red, final_blue, total_final_balls) total_iterations
    = 1 / 7) := 
sorry

end urn_problem_probability_l118_118654


namespace arithmetic_mean_of_geometric_sequence_l118_118665

theorem arithmetic_mean_of_geometric_sequence (a r : ℕ) (h_a : a = 4) (h_r : r = 3) :
    ((a) + (a * r) + (a * r^2)) / 3 = (52 / 3) :=
by
  sorry

end arithmetic_mean_of_geometric_sequence_l118_118665


namespace proof_correct_operation_l118_118584

/-
Given conditions:
1. x^a * x^b = x^(a + b)
2. x^a / x^b = x^(a - b)
3. (x^a)^b = x^(a * b)
4. (a * b)^n = a^n * b^n

Prove that the correct operation among the following is:

1. x^3 * x^4 = x^12
2. x^4 / x = x^3
3. (x^3)^4 = x^7
4. (x^3 * y)^3 = x^6 * y^3

The correct operation is x^4 / x = x^3.
-/

theorem proof_correct_operation :
  let x, y : ℕ := sorry
  (x^3 * x^4 ≠ x^12)
  ∧ (x^4 / x = x^3)
  ∧ ((x^3)^4 ≠ x^7)
  ∧ ((x^3 * y)^3 ≠ x^6 * y^3) :=
by 
  sorry

end proof_correct_operation_l118_118584


namespace total_notes_in_week_l118_118253

-- Define the conditions for day hours ring pattern
def day_notes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 15 then 2
  else if minute = 30 then 4
  else if minute = 45 then 6
  else if minute = 0 then 
    8 + (if hour % 2 = 0 then hour else hour / 2)
  else 0

-- Define the conditions for night hours ring pattern
def night_notes (hour : ℕ) (minute : ℕ) : ℕ :=
  if minute = 15 then 3
  else if minute = 30 then 5
  else if minute = 45 then 7
  else if minute = 0 then 
    9 + (if hour % 2 = 1 then hour else hour / 2)
  else 0

-- Define total notes over day period
def total_day_notes : ℕ := 
  (day_notes 6 0 + day_notes 7 0 + day_notes 8 0 + day_notes 9 0 + day_notes 10 0 + day_notes 11 0
 + day_notes 12 0 + day_notes 1 0 + day_notes 2 0 + day_notes 3 0 + day_notes 4 0 + day_notes 5 0)
 +
 (2 * 12 + 4 * 12 + 6 * 12)

-- Define total notes over night period
def total_night_notes : ℕ := 
  (night_notes 6 0 + night_notes 7 0 + night_notes 8 0 + night_notes 9 0 + night_notes 10 0 + night_notes 11 0
 + night_notes 12 0 + night_notes 1 0 + night_notes 2 0 + night_notes 3 0 + night_notes 4 0 + night_notes 5 0)
 +
 (3 * 12 + 5 * 12 + 7 * 12)

-- Define the total number of notes the clock will ring in a full week
def total_week_notes : ℕ :=
  7 * (total_day_notes + total_night_notes)

theorem total_notes_in_week : 
  total_week_notes = 3297 := 
  by 
  sorry

end total_notes_in_week_l118_118253


namespace kelvin_wins_strategy_l118_118650

theorem kelvin_wins_strategy (n : ℕ) (h : n > 0) : (n % 3 ≠ 0) ↔ Kelvin_wins n :=
sorry

end kelvin_wins_strategy_l118_118650


namespace min_positive_sum_of_products_l118_118344

-- Definitions based on the problem conditions
def a : Fin 50 → ℤ := sorry  -- Let a_i be integers (either 1 or -1) defined over the finite set {1, 2, ..., 50}

-- Sum over pairs (i, j) such that 1 ≤ i < j ≤ 50
def sum_of_products (a : Fin 50 → ℤ) : ℤ :=
  ∑ i in Finset.range 50, ∑ j in Finset.Ico (i+1) 50, a i * a j

-- The theorem we need to prove
theorem min_positive_sum_of_products : 
  (∀ i, a i = 1 ∨ a i = -1) → 0 < sum_of_products a → sum_of_products a = 7 :=
sorry

end min_positive_sum_of_products_l118_118344


namespace cricketer_average_after_19_innings_l118_118419

theorem cricketer_average_after_19_innings
  (runs_19th_inning : ℕ)
  (increase_in_average : ℤ)
  (initial_average : ℤ)
  (new_average : ℤ)
  (h1 : runs_19th_inning = 95)
  (h2 : increase_in_average = 4)
  (eq1 : 18 * initial_average + 95 = 19 * (initial_average + increase_in_average))
  (eq2 : new_average = initial_average + increase_in_average) :
  new_average = 23 :=
by sorry

end cricketer_average_after_19_innings_l118_118419


namespace problem1_problem2_l118_118086

variables (A B C D : Type) [Real.uniform_space A]
variables {a b c BD AD DC : Real}
variables (angle_ABC angle_ACB : Real)

def proof1 (h1 : b^2 = a * c) (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) : Prop :=
  BD = b

def proof2 (h3 : AD = 2 * DC) (h1 : b^2 = a * c) : Prop :=
  Real.cos angle_ABC = 7 / 12

theorem problem1 {A B C D : Type} [Real.uniform_space A]
  {a b c BD : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) :
  proof1 a b c BD angle_ABC angle_ACB h1 h2 :=
sorry

theorem problem2 {A B C D : Type} [Real.uniform_space A]
  {a b c BD AD DC : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h3 : AD = 2 * DC) :
  proof2 a b c BD angle_ABC h1 h3 :=
sorry

end problem1_problem2_l118_118086


namespace concurrency_of_AP_BQ_CR_l118_118121

theorem concurrency_of_AP_BQ_CR
  (ABC : Triangle)
  (P Q R : Point)
  (hPQ : tangent_point ABC.incircle BC P)
  (hQR : tangent_point ABC.incircle CA Q)
  (hRP : tangent_point ABC.incircle AB R):
  concurrent ABC.A P ABC.B Q ABC.C R := 
sorry

end concurrency_of_AP_BQ_CR_l118_118121


namespace necessary_but_not_sufficient_l118_118625

variable (a : ℝ)

theorem necessary_but_not_sufficient (h : a ≥ 2) : (a = 2 ∨ a > 2) ∧ ¬(a > 2 → a ≥ 2) := by
  sorry

end necessary_but_not_sufficient_l118_118625


namespace equal_row_column_sums_sum_of_all_sums_l118_118188

variables (A : Fin 20 → ℕ) (B : Fin 15 → ℕ)
noncomputable def X : ℕ := sorry -- the common sum value, potentially 0

theorem equal_row_column_sums :
  (∀ i, A i = X) ∧ (∀ j, B j = X) → X = 0 :=
by
  sorry

theorem sum_of_all_sums :
  (∀ i, A i = 0) ∧ (∀ j, B j = 0) →
  (∑ i, A i + ∑ j, B j) = 0 :=
by
  sorry

end equal_row_column_sums_sum_of_all_sums_l118_118188


namespace exists_product_of_s_distinct_primes_l118_118210

theorem exists_product_of_s_distinct_primes (s : ℕ) (h_s : 0 < s) :
  ∃ N : ℕ, ∀ n : ℕ, n > N → ∃ m : ℕ, n < m ∧ m < 2 * n ∧ (∃ (primes : Fin s → ℕ), (∀ i, Nat.Prime (primes i)) ∧ m = (∏ i, primes i)) :=
by
  sorry

end exists_product_of_s_distinct_primes_l118_118210


namespace largest_four_digit_by_two_moves_l118_118597

def moves (n : Nat) (d1 d2 d3 d4 : Nat) : Prop :=
  ∃ x y : ℕ, d1 = x → d2 = y → n = 1405 → (x ≤ 2 ∧ y ≤ 2)

theorem largest_four_digit_by_two_moves :
  ∃ n : ℕ, moves 1405 1 4 0 5 ∧ n = 7705 :=
by
  sorry

end largest_four_digit_by_two_moves_l118_118597


namespace bloodPressureFriday_l118_118270

def bloodPressureSunday : ℕ := 120
def bpChangeMonday : ℤ := 20
def bpChangeTuesday : ℤ := -30
def bpChangeWednesday : ℤ := -25
def bpChangeThursday : ℤ := 15
def bpChangeFriday : ℤ := 30

theorem bloodPressureFriday : bloodPressureSunday + bpChangeMonday + bpChangeTuesday + bpChangeWednesday + bpChangeThursday + bpChangeFriday = 130 := by {
  -- Placeholder for the proof
  sorry
}

end bloodPressureFriday_l118_118270


namespace max_area_triangle_l118_118004

theorem max_area_triangle 
  (A B C : Real) (a b c : Real)
  (h1 : angles_in_triangle A B C = true)
  (h2 : sides_correspond A B C a b c = true)
  (h3 : 2 * a * (cos C)^2 + 2 * c * cos A * cos C + b = 0) 
  (h4 : B = 4 * sin B) :
  (C = 2 * π / 3) ∧ (max_area a b c = sqrt 3) :=
by
  sorry

end max_area_triangle_l118_118004


namespace sqrt_expression_value_l118_118363

-- Defines the sequence and the initial condition
def is_arithmetic_progression (x : ℕ → ℝ) : Prop :=
  ∀ n ≥ 2, x n = (x (n - 1) + 98 * x n + x (n + 1)) / 100

theorem sqrt_expression_value (x : ℕ → ℝ) (h : is_arithmetic_progression x) :
  sqrt (((x 2023 - x 1) / 2022) * (2021 / (x 2023 - x 2))) + 2021 = 2022 :=
  sorry

end sqrt_expression_value_l118_118363


namespace person_half_age_in_10_years_l118_118272

def mother's_present_age := 50
def person's_present_age : ℕ := (2 / 5 : ℚ) * mother's_present_age

theorem person_half_age_in_10_years :
  ∃ (Y : ℕ), Y = 10 ∧ (person's_present_age + Y = (1 / 2 : ℚ) * (mother's_present_age + Y)) :=
sorry

end person_half_age_in_10_years_l118_118272


namespace complement_of_union_is_empty_l118_118489

-- Define the universal set U
def U : Set α := {a, b, c, d, e}

-- Define the set N
def N : Set α := {b, d, e}

-- Define the set M
def M : Set α := {a, c, d}

-- Define the complement of the union of M and N with respect to U
noncomputable def complement_of_union : Set α := U \ (M ∪ N)

-- State the theorem to be proven
theorem complement_of_union_is_empty : complement_of_union = ∅ :=
by sorry

end complement_of_union_is_empty_l118_118489


namespace all_numbers_same_color_l118_118901

theorem all_numbers_same_color 
  (n k : ℕ) 
  (h_gcd : Nat.gcd k n = 1) 
  (h_k_lt_n : 0 < k ∧ k < n) 
  (M : Finset ℕ := Finset.range (n - 1).succ) 
  (color : ℕ → Prop) 
  (h_color_a : ∀ i ∈ M, color i = color (n - i)) 
  (h_color_b : ∀ i ∈ M, i ≠ k → color i = color (Nat.abs (k - i))) : 
  ∀ i ∈ M, color i = color 1 := sorry

end all_numbers_same_color_l118_118901


namespace juggler_path_radius_l118_118620

theorem juggler_path_radius : 
  (∃ (center : ℝ × ℝ) (r : ℝ), ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = r^2 ∧
  (x^2 + y^2 + 5 = 2x + 4y) → r = 0) :=
sorry

end juggler_path_radius_l118_118620


namespace problem_4x_32y_l118_118797

theorem problem_4x_32y (x y : ℝ) (h : 2 * x + 5 * y = 4) : 4^x * 32^y = 16 :=
by sorry

end problem_4x_32y_l118_118797


namespace sum_abc_eq_113_l118_118818

theorem sum_abc_eq_113 :
  (∃ (a b c : ℕ), c > 0 ∧
    (c * (√3 + 1/√3 + √11 + 1/√11) = a * √3 + b * √11) ∧ 
    (∀ c', c' > 0 ∧ (c' * (√3 + 1/√3 + √11 + 1/√11) = a * √3 + b * √11) → c' ≥ c) ∧ 
    a + b + c = 113) :=
begin
  sorry
end

end sum_abc_eq_113_l118_118818


namespace part_a_l118_118241

theorem part_a (A B C A1 B1 C1 : Point) (hA1 : A1 ∈ line_segment B C) (hB1 : B1 ∈ line_segment C A) (hC1 : C1 ∈ line_segment A B) :
  ∃ (T : Triangle), (T = triangle AB1 C1 ∨ T = triangle A1 B C1 ∨ T = triangle A1 B1 C) ∧ Area T ≤ Area (triangle A B C) / 4 := 
sorry

end part_a_l118_118241


namespace angle_between_tangents_l118_118981

theorem angle_between_tangents (r R d : ℝ) (α β : ℝ) (hr : r < R) 
  (h1 : sin (α / 2) = (R - r) / d)
  (h2 : sin (β / 2) = (R + r) / d) :
  let φ := 2 * arcsin ((sin (β / 2) - sin (α / 2)) / 2)
  in φ = 2 * arcsin ((sin (β / 2) - sin (α / 2)) / 2) :=
by
  sorry

end angle_between_tangents_l118_118981


namespace net_profit_positive_max_average_net_profit_l118_118259

def initial_investment : ℕ := 720000
def first_year_expense : ℕ := 120000
def annual_expense_increase : ℕ := 40000
def annual_sales : ℕ := 500000

def net_profit (n : ℕ) : ℕ := annual_sales - (first_year_expense + (n-1) * annual_expense_increase)
def average_net_profit (y n : ℕ) : ℕ := y / n

theorem net_profit_positive (n : ℕ) : net_profit n > 0 :=
sorry -- prove when net profit is positive

theorem max_average_net_profit (n : ℕ) : 
∀ m, average_net_profit (net_profit m) m ≤ average_net_profit (net_profit n) n :=
sorry -- prove when the average net profit is maximized

end net_profit_positive_max_average_net_profit_l118_118259


namespace leftover_coverage_l118_118677

theorem leftover_coverage 
  (coverage_per_bag : ℕ)
  (length : ℕ)
  (width : ℕ)
  (num_bags : ℕ) :
  coverage_per_bag = 250 →
  length = 22 →
  width = 36 →
  num_bags = 4 →
  let lawn_area := length * width,
      total_coverage := coverage_per_bag * num_bags,
      leftover_coverage := total_coverage - lawn_area
  in leftover_coverage = 208 := 
by
  intros h1 h2 h3 h4
  let lawn_area := 22 * 36
  let total_coverage := 250 * 4
  let leftover_coverage := total_coverage - lawn_area
  have : lawn_area = 792 := by norm_num
  have : total_coverage = 1000 := by norm_num
  have : leftover_coverage = total_coverage - lawn_area := rfl
  show leftover_coverage = 208, from by
    rw [this, this, this]
    norm_num
  sorry

end leftover_coverage_l118_118677


namespace function_passes_through_fixed_point_l118_118945

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  ∃ k : ℝ, k = (2, 2) :=
by
  use (2, 2)
  sorry

end function_passes_through_fixed_point_l118_118945


namespace combined_resistance_parallel_l118_118839

def reciprocal (x : ℝ) := 1 / x

theorem combined_resistance_parallel (R1 R2 R3 R4 : ℝ) (h1 : R1 = 8) (h2 : R2 = 9) (h3 : R3 = 6) (h4 : R4 = 12) :
  let R := 1 / (reciprocal R1 + reciprocal R2 + reciprocal R3 + reciprocal R4) in
  R = 72 / 35 :=
by
  sorry

end combined_resistance_parallel_l118_118839


namespace transformed_area_l118_118879

theorem transformed_area (R : Set (ℝ × ℝ)) (area_R : ℝ) 
    (h_area : area_R = 15) :
    let A := ![![3, 4], ![6, -2]] in
    let det_A := A.det in
    abs det_A * area_R = 450 :=
by
  let A := ![![3, 4], ![6, -2]]
  let det_A := A.det
  have h_det : det_A = -30 := by sorry
  have h_abs_det : abs det_A = 30 := by sorry
  rw [h_area, h_abs_det]
  have h_transformed_area : 30 * 15 = 450 := by norm_num
  exact h_transformed_area

end transformed_area_l118_118879


namespace dice_product_prob_divisible_by_4_l118_118205

/-- 
  Given that Una rolls 6 standard 6-sided dice simultaneously, 
  prove that the probability that the product of the 6 numbers obtained is divisible by 4 
  is 61/64.
-/
theorem dice_product_prob_divisible_by_4 :
  (fin 6 → fin 6 → ℚ) := sorry

end dice_product_prob_divisible_by_4_l118_118205


namespace least_pqrs_is_14_l118_118432
noncomputable def least_pqrs : ℕ :=
  let p := 3
  let q := 3
  let r := 4
  let s := 4
  p + q + r + s

theorem least_pqrs_is_14 :
  ∃ p q r s : ℕ, p > 1 ∧ q > 1 ∧ r > 1 ∧ s > 1 ∧
    31 * (p + 1) = 37 * (q + 1) ∧
    41 * (r + 1) = 43 * (s + 1) ∧
    p + q + r + s = 14 :=
begin
  existsi (3:ℕ),
  existsi (3:ℕ),
  existsi (4:ℕ),
  existsi (4:ℕ),
  split,
  linarith,
  split,
  linarith,
  split,
  linarith,
  split,
  linarith,
  split,
  linarith,
  split,
  { linarith, },
  { linarith, },
  { linarith, },
  split,
  { show 31 * (3 + 1) = 37 * (3 + 1), by norm_num },
  split,
  { show 41 * (4 + 1) = 43 * (4 + 1), by norm_num },
  { show 3 + 3 + 4 + 4 = 14, by norm_num }
end

end least_pqrs_is_14_l118_118432


namespace valid_starting_lineups_count_l118_118510

theorem valid_starting_lineups_count :
  let players := 15
  let select_count := 5
  let tim_mike_sam := 3
  let total_combinations := ∑ (choose (12, 3)) * 3 + (choose (12, 5)) + (choose (12, 4)) * 3
  ∃ num_lineups, num_lineups = 2937 :=
begin
  let players := 15,
  let select_count := 5,
  let tim_mike_sam := 3,
  let total_combinations := 
    3 * (Nat.choose 12 3) + 
    (Nat.choose 12 5) + 
    3 * (Nat.choose 12 4),
  use total_combinations,
  exact 2937
sorry

end valid_starting_lineups_count_l118_118510


namespace expected_value_of_die_l118_118418

noncomputable def expected_value : ℚ :=
  (1/14) * 1 + (1/14) * 2 + (1/14) * 3 + (1/14) * 4 + (1/14) * 5 + (1/14) * 6 + (1/14) * 7 + (3/8) * 8

theorem expected_value_of_die : expected_value = 5 :=
by
  sorry

end expected_value_of_die_l118_118418


namespace range_of_m_l118_118957

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Icc 0 (π / 4), tan x ≤ m) ∨
  (∀ x1 ∈ Icc (-1 : ℝ) 3, ∃ x2 ∈ Icc 0 2, x1^2 + m ≥ ((1 / 2) ^ x2 - m)) ∧
  ¬((∀ x ∈ Icc 0 (π / 4), tan x ≤ m) ∧ 
  (∀ x1 ∈ Icc (-1 : ℝ) 3, ∃ x2 ∈ Icc 0 2, x1^2 + m ≥ ((1 / 2) ^ x2 - m)))
  ↔ (m ∈ Icc (1 / 8 : ℝ) 1) :=
by {
  sorry
}

end range_of_m_l118_118957


namespace interval_length_of_possible_m_values_l118_118122

theorem interval_length_of_possible_m_values : 
  let T := {(x, y) | 1 ≤ x ∧ x ≤ 40 ∧ 1 ≤ y ∧ y ≤ 40}
  in (∃ (m : ℚ), {p ∈ T | p.2 ≤ m * p.1}.card = 600) ∧
     (∀ (a b : ℕ), (∀ a b, Nat.coprime a b → a = 1 ∧ b = 6) →
     (interval_length : ℚ := 1 / 6) →
     a + b = 7) :=
by sorry

end interval_length_of_possible_m_values_l118_118122


namespace xiao_wang_conjecture_incorrect_l118_118587

theorem xiao_wang_conjecture_incorrect : ∃ n : ℕ, n > 0 ∧ (n^2 - 8 * n + 7 > 0) := by
  sorry

end xiao_wang_conjecture_incorrect_l118_118587


namespace stratified_sampling_l118_118970

theorem stratified_sampling (total_students_A total_students_B total_students_C sample_size : ℕ)
    (hA : total_students_A = 3600)
    (hB : total_students_B = 5400)
    (hC : total_students_C = 1800)
    (h_sample_size : sample_size = 90) :
    ∃ (n_A n_B n_C : ℕ),
    n_A = 30 ∧ n_B = 45 ∧ n_C = 15 :=
by
  let total_students := total_students_A + total_students_B + total_students_C
  have h_total : total_students = 10800 := by
    rw [hA, hB, hC]
    norm_num
  let p_A := total_students_A / total_students
  let p_B := total_students_B / total_students
  let p_C := total_students_C / total_students
  have h_pA : p_A = 1 / 3 := by
    rw [hA]
    norm_num
  have h_pB : p_B = 1 / 2 := by
    rw [hB]
    norm_num
  have h_pC : p_C = 1 / 6 := by
    rw [hC]
    norm_num
  let n_A := sample_size * p_A
  let n_B := sample_size * p_B
  let n_C := sample_size * p_C
  use n_A, n_B, n_C
  have h_nA : n_A = 30 := by
    rw [h_sample_size, h_pA]
    norm_num
  have h_nB : n_B = 45 := by
    rw [h_sample_size, h_pB]
    norm_num
  have h_nC : n_C = 15 := by
    rw [h_sample_size, h_pC]
    norm_num
  exact ⟨h_nA, h_nB, h_nC⟩


end stratified_sampling_l118_118970


namespace probability_divisible_by_4_l118_118207

theorem probability_divisible_by_4 :
  (probability (product_of_dice_divisible_by_4 (list_of_rolls dice_rolls 6))) = 61 / 64 :=
sorry

-- Definitions relevant to the problem

def dice_faces : list ℕ := [1, 2, 3, 4, 5, 6]

def dice_rolls (n : ℕ) : list (list ℕ) :=
(list.replicate n dice_faces).sequence

def product (l : list ℕ) : ℕ := l.foldl (*) 1

def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def product_of_dice_divisible_by_4 (rolls : list (list ℕ)) : ℕ → ℕ :=
  λ n, rolls.count (λ l, divisible_by_4 (product l))

-- Probability function
noncomputable def probability (event_count : ℕ) : ℚ :=
event_count / (6 ^ 6)

end probability_divisible_by_4_l118_118207


namespace product_of_invertible_labels_l118_118519

def function2 (x : ℝ) : ℝ := x^3 - 3 * x
def function3 (x : ℝ) : Option ℝ :=
  if x = -6 then some 5
  else if x = -4 then some 3
  else if x = -2 then some 1
  else if x = 0 then some (-1)
  else if x = 2 then some (-3)
  else if x = 4 then some (-5)
  else if x = 6 then some (-7)
  else none

def function4 (x : ℝ) : Option ℝ :=
  if x = -6 then some (Real.sin (-6))
  else if x = -4 then some (Real.sin (-4))
  else if x = -2 then some (Real.sin (-2))
  else if x = 0 then some (Real.sin 0)
  else if x = 2 then some (Real.sin 2)
  else if x = 4 then some (Real.sin 4)
  else if x = 6 then some (Real.sin 6)
  else none

def function5 (x : ℝ) : Option ℝ :=
  if x = 0 then none else some (1 / x)

def is_invertible (f : ℝ → Option ℝ) : Prop :=
  ∀ y1 y2 x1 x2, f x1 = some y1 → f x2 = some y2 → y1 = y2 → x1 = x2

theorem product_of_invertible_labels : 2 * 3 * 5 = 30 := by
  sorry

end product_of_invertible_labels_l118_118519


namespace four_digit_integers_correct_five_digit_integers_correct_l118_118982

-- Definition for the four-digit integers problem
def num_four_digit_integers := ∃ digits : Finset (Fin 5), 4 * 24 = 96

theorem four_digit_integers_correct : num_four_digit_integers := 
by
  sorry

-- Definition for the five-digit integers problem without repetition and greater than 21000
def num_five_digit_integers := ∃ digits : Finset (Fin 5), 48 + 18 = 66

theorem five_digit_integers_correct : num_five_digit_integers := 
by
  sorry

end four_digit_integers_correct_five_digit_integers_correct_l118_118982


namespace triangle_fold_crease_length_l118_118629

noncomputable def length_of_crease (A B C : ℝ) (AB AC BC : ℝ) : ℝ :=
  let M := BC / 2   -- Midpoint of BC
  in real.sqrt (AB^2 - M^2)

theorem triangle_fold_crease_length:
  ∀ (A B C : ℝ), (AB AC BC : ℝ),
  AB = 4 → AC = 4 → BC = 6 → length_of_crease A B C AB AC BC = real.sqrt 7 :=
begin
  intros A B C AB AC BC hAB hAC hBC,
  simp [hAB, hBC, length_of_crease],
  linarith,
end

end triangle_fold_crease_length_l118_118629


namespace finite_sequence_primes_l118_118335

theorem finite_sequence_primes:
  ∃ (c : Fin 5 → ℤ), let a := [0, 2] in
  (∀ i, c i = [3, 5, 29, 101, 107][i]) ∧
  ∀ a ∈ (↑a : Multiset ℤ), ∀ i < 5, Prime (a + c i) := 
begin
  sorry
end

end finite_sequence_primes_l118_118335


namespace parallel_condition_l118_118951

variable (ℝ : Type) [field ℝ]
variable (Point Line Plane : Type)
variable (intersect : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define a helper predicate for not intersecting with any line
def does_not_intersect_any_line (ℓ : Line) (π : Plane) : Prop :=
  ∀ (l₁ : Line), intersect l₁ π → ¬ intersect ℓ l₁

-- Main theorem statement
theorem parallel_condition (ℓ : Line) (π : Plane) :
  parallel ℓ π ↔ does_not_intersect_any_line ℓ π :=
by
  sorry

end parallel_condition_l118_118951


namespace max_students_l118_118159

theorem max_students : ∃ n : ℕ, (∀ x, x > n) ∧ 
  ∀ p, prime p → 
    1080 % p = 0 → 
    920 % p = 0 → 
    680 % p = 0 → 
    p ∣ 920 → 
    p ≠ 23 ∧ p ≠ 1 ∧ p = 5 ∧ n = 184 := by sorry

end max_students_l118_118159


namespace abc_le_one_eighth_l118_118874

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end abc_le_one_eighth_l118_118874


namespace trapezoid_circles_tangent_l118_118938

theorem trapezoid_circles_tangent
  (A B C D O : Point)
  (trapezoid : is_trapezoid A D B C)
  (intersect : intersects_at_diagonals A D B C O) :
  tangent (circumcircle △AOD) (circumcircle △BOC) O :=
sorry

end trapezoid_circles_tangent_l118_118938


namespace find_a_l118_118044

theorem find_a (a : ℝ) : (∃ k : ℝ, (x - 2) * (x + k) = x^2 + a * x - 5) ↔ a = 1 / 2 :=
by
  sorry

end find_a_l118_118044


namespace problem1_problem2_l118_118084

variables (A B C D : Type) [Real.uniform_space A]
variables {a b c BD AD DC : Real}
variables (angle_ABC angle_ACB : Real)

def proof1 (h1 : b^2 = a * c) (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) : Prop :=
  BD = b

def proof2 (h3 : AD = 2 * DC) (h1 : b^2 = a * c) : Prop :=
  Real.cos angle_ABC = 7 / 12

theorem problem1 {A B C D : Type} [Real.uniform_space A]
  {a b c BD : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h2 : BD * Real.sin angle_ABC = a * Real.sin angle_ACB) :
  proof1 a b c BD angle_ABC angle_ACB h1 h2 :=
sorry

theorem problem2 {A B C D : Type} [Real.uniform_space A]
  {a b c BD AD DC : Real}
  {angle_ABC angle_ACB : Real}
  (h1 : b^2 = a * c)
  (h3 : AD = 2 * DC) :
  proof2 a b c BD angle_ABC h1 h3 :=
sorry

end problem1_problem2_l118_118084


namespace count_pos_3digit_multiples_of_30_not_75_l118_118793

theorem count_pos_3digit_multiples_of_30_not_75 : 
  let multiples_of_30 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 30 = 0}
  let multiples_of_75 := {n : ℕ | (100 ≤ n ∧ n < 1000) ∧ n % 75 = 0}
  (multiples_of_30 \ multiples_of_75).size = 24 :=
by
  sorry

end count_pos_3digit_multiples_of_30_not_75_l118_118793


namespace distance_from_P_to_AD_l118_118924

theorem distance_from_P_to_AD :
  let A := (0, 5)
  let D := (0, 0) 
  let B := (5, 5)
  let C := (5, 0)
  let M := (2.5, 0)
  let radius_M := 2.5
  let radius_A := 5
  let EqCircleM : (ℝ × ℝ) → ℝ := λ p, (p.1 - 2.5) ^ 2 + p.2 ^ 2
  let EqCircleA : (ℝ × ℝ) → ℝ := λ p, p.1 ^ 2 + (p.2 - 5) ^ 2
  ∃ P : ℝ × ℝ,
    EqCircleM P = radius_M ^ 2 ∧
    EqCircleA P = radius_A ^ 2 ∧
    P.2 = 4 + 2 * Real.sqrt 5 ∧
    P.1 = 0.5 + 4 * Real.sqrt 5 ∧
    ∀ q : ℝ × ℝ, q = P → q.1 = 0.5 + 4 * Real.sqrt 5 :=
by
  sorry

end distance_from_P_to_AD_l118_118924


namespace initial_total_balls_l118_118542

theorem initial_total_balls (B T : Nat) (h1 : B = 9) (h2 : ∀ (n : Nat), (T - 5) * 1/5 = 4) :
  T = 25 := sorry

end initial_total_balls_l118_118542


namespace empty_solution_set_implies_a_range_l118_118018

def f (a x: ℝ) := x^2 + (1 - a) * x - a

theorem empty_solution_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬ (f a (f a x) < 0)) → -3 ≤ a ∧ a ≤ 2 * Real.sqrt 2 - 3 :=
by
  sorry

end empty_solution_set_implies_a_range_l118_118018


namespace friends_attended_reception_l118_118256

theorem friends_attended_reception :
  (total_people bride_invited groom_invited : ℕ) 
  (bride_couples groom_couples : ℕ)
  (h1 : total_people = 180)
  (h2 : bride_invited = bride_couples * 2)
  (h3 : groom_invited = groom_couples * 2)
  (h4 : bride_couples = 20)
  (h5 : groom_couples = 20)
  : (total_people - (bride_invited + groom_invited) = 100) := by
  sorry

end friends_attended_reception_l118_118256


namespace find_n_l118_118655

noncomputable
def equilateral_triangle_area_ratio (n : ℕ) (h : n > 4) : Prop :=
  let ratio := (2 : ℚ) / (n - 2 : ℚ)
  let area_PQR := (1 / 7 : ℚ)
  let menelaus_ap_pd := (n * (n - 2) : ℚ) / 4
  let area_triangle_ABP := (2 * (n - 2) : ℚ) / (n * (n - 2) + 4)
  let area_sum := 3 * area_triangle_ABP
  (area_sum * 7 = 6 * (n * (n - 2) + 4))

theorem find_n (n : ℕ) (h : n > 4) : 
  (equilateral_triangle_area_ratio n h) → n = 6 := sorry

end find_n_l118_118655


namespace sum_abs_a1_to_a10_l118_118070

def S (n : ℕ) : ℤ := n^2 - 4 * n + 2
def a (n : ℕ) : ℤ := if n = 1 then S 1 else S n - S (n - 1)

theorem sum_abs_a1_to_a10 : (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 66) := 
by
  sorry

end sum_abs_a1_to_a10_l118_118070


namespace leading_coefficient_of_g_l118_118171

-- Let g be a polynomial that satisfies the functional equation
variable (g : ℕ → ℝ)
variable (h : ∀ x : ℕ, g (x + 1) - g x = 8 * (x : ℝ) + 6)

-- Theorem stating the leading coefficient of g is 4
theorem leading_coefficient_of_g : leading_coeff (polynomial.of_finsupp (λ x, g x)) = 4 :=
sorry

end leading_coefficient_of_g_l118_118171


namespace cos_half_diff_minus_sin_half_B_eq_zero_l118_118453

theorem cos_half_diff_minus_sin_half_B_eq_zero (A B C : ℝ) 
  (h : sin A * cos (C / 2) ^ 2 + sin C * cos (A / 2) ^ 2 = (3 / 2) * sin B) : 
  cos ((A - C) / 2) - 2 * sin (B / 2) = 0 := 
sorry

end cos_half_diff_minus_sin_half_B_eq_zero_l118_118453


namespace probability_odd_sum_of_drawn_balls_l118_118606

/-- 
  In a set of 13 balls numbered from 1 to 13, 
  prove that the probability of drawing 7 balls such that their sum is odd is 121/246.
-/
theorem probability_odd_sum_of_drawn_balls : 
  (∃ balls : set ℕ, balls = {x | 1 ≤ x ∧ x ≤ 13} ∧ ∀ s : finset ℕ, s ⊆ balls ∧ s.card = 7 → 
  (∃ odd_draws : ℕ, odd_draws = (s.filter (λ n, odd n)).card ∧ (odd_draws % 2 = 1 → 
  (↑((finset.card ((s.filter (λ n, odd n)))).factorial * finset.card ((s.filter (λ n, even n)))).factorial) / 
  (((finset.card ((s.filter (λ n, odd n)))).factorial * (finset.card ((s.filter (λ n, even n)))).factorial) = 121 / 246))) sorry

end probability_odd_sum_of_drawn_balls_l118_118606


namespace range_of_a_l118_118014

noncomputable def complex_number (a : ℝ) := (2 + complex.i) * (a - 2 * complex.i)

theorem range_of_a (a : ℝ) : 
  let z := complex_number a in
  let x := complex.re z in
  let y := complex.im z in
  y < 0 ∧ x > 0 → -1 < a ∧ a < 4 :=
by {
  intro h,
  sorry
}

end range_of_a_l118_118014


namespace S6_values_l118_118000

noncomputable def a (n : ℕ) : ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := sorry

axiom geo_seq (q : ℝ) :
  ∀ n : ℕ, a n = a 0 * q ^ n

variable (a3_eq_4 : a 2 = 4) 
variable (S3_eq_7 : S 3 = 7)

theorem S6_values : S 6 = 63 ∨ S 6 = 133 / 27 := sorry

end S6_values_l118_118000


namespace find_a1_l118_118124

-- Given an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

-- Arithmetic sequence is monotonically increasing
def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

-- First condition: sum of first three terms
def sum_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 12

-- Second condition: product of first three terms
def product_first_three_terms (a : ℕ → ℝ) : Prop :=
  a 0 * a 1 * a 2 = 48

-- Proving that a_1 = 2 given the conditions
theorem find_a1 (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : is_monotonically_increasing a)
  (h3 : sum_first_three_terms a) (h4 : product_first_three_terms a) : a 0 = 2 :=
by
  -- Proof will be filled in here
  sorry

end find_a1_l118_118124


namespace price_of_skateboard_l118_118130

-- Given condition (0.20 * p = 300)
variable (p : ℝ)
axiom upfront_payment : 0.20 * p = 300

-- Theorem statement to prove the price of the skateboard
theorem price_of_skateboard : p = 1500 := by
  sorry

end price_of_skateboard_l118_118130


namespace triangular_pyramid_has_parallelogram_triangular_pyramid_has_rhombus_triangular_pyramid_has_rectangle_l118_118334

/-- Define a triangular pyramid ABCD and points E, F, G on edges AB, AC, and CD -/
variables {A B C D E F G H : Type} [Plane Geometry]

/-- Define conditions: E ∈ AB, F ∈ AC, G ∈ CD, plane σ determined by E, F, G intersects BD at H to form a four-sided figure EFGH -/
def triangular_pyramid (A B C D E F G H : Point) (σ : Plane) : Prop :=
  E ∈ line_segment A B ∧
  F ∈ line_segment A C ∧
  G ∈ line_segment C D ∧
  σ = Plane.determine E F G ∧
  H ∈ line_segment B D ∧
  H = Plane.intersection σ (line_segment B D) ∧
  CrossSectionFourSided A B C D E F G H

/-- Proof statement: A triangular pyramid can have a cross-section that is a parallelogram -/
theorem triangular_pyramid_has_parallelogram (A B C D E F G H : Point) (σ : Plane) :
  triangular_pyramid A B C D E F G H σ →
  ∃ E F G H, Parallelogram E F G H :=
by
  intros h_tp
  sorry

/-- Proof statement: A triangular pyramid can have a cross-section that is a rhombus -/
theorem triangular_pyramid_has_rhombus (A B C D E F G H : Point) (σ : Plane) :
  triangular_pyramid A B C D E F G H σ →
  ∃ E F G H, Rhombus E F G H :=
by
  intros h_tp
  sorry

/-- Proof statement: A triangular pyramid can have a cross-section that is a rectangle if BC ⊥ AD -/
theorem triangular_pyramid_has_rectangle (A B C D E F G H : Point) (σ : Plane) :
  triangular_pyramid A B C D E F G H σ →
  Perpendicular (line_segment B C) (line_segment A D) →
  ∃ E F G H, Rectangle E F G H :=
by
  intros h_tp h_perpendicular
  sorry

end triangular_pyramid_has_parallelogram_triangular_pyramid_has_rhombus_triangular_pyramid_has_rectangle_l118_118334


namespace karen_speed_proof_l118_118864

-- Define constants and their values
def Karen_late_minutes : ℕ := 4
def Karen_late_hours : ℚ := 1 / 15
def Tom_speed_mph : ℚ := 45
def Tom_distance_before_loss : ℚ := 24
def distance_difference_from_start : ℚ := 4

-- Define Karen's distance and time calculation.
def Karen_speed_mph : ℚ := 78.75

-- Define the proof statement.
theorem karen_speed_proof :
  let time_Tom := Tom_distance_before_loss / Tom_speed_mph in 
  let time_Karen := time_Tom - Karen_late_hours in 
  let distance_Karen := Tom_distance_before_loss + distance_difference_from_start in
  distance_Karen = Karen_speed_mph * time_Karen :=
by
  sorry -- Proof to be filled in.

end karen_speed_proof_l118_118864


namespace max_value_c_exists_l118_118853

open Real

-- Definitions for points, circle, and distances
def Point := ℝ × ℝ
def Circle := Point × ℝ

def B : Point := (2, 0)
def A : Point := (-3, 0)
def circleA : Circle := (A, 5)

def is_outside (C : Circle) (P : Point) := 
  let (center, radius) := C
  dist center P > radius

noncomputable
def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  
axiom O : Point

theorem max_value_c_exists :
  ∃ (c : ℝ), c = (Real.sqrt 115 - 10) / 5 ∧ 
  (∀ (X : Point), is_outside circleA X →
  (dist O X - 2 ≥ c * min (dist B X) ((dist B X)^2))) :=
sorry

end max_value_c_exists_l118_118853


namespace smallest_common_term_larger_than_2023_l118_118535

noncomputable def a_seq (n : ℕ) : ℤ :=
  3 * n - 2

noncomputable def b_seq (m : ℕ) : ℤ :=
  10 * m - 8

theorem smallest_common_term_larger_than_2023 :
  ∃ (n m : ℕ), a_seq n = b_seq m ∧ a_seq n > 2023 ∧ a_seq n = 2032 :=
by {
  sorry
}

end smallest_common_term_larger_than_2023_l118_118535


namespace part1_prove_BD_eq_b_part2_prove_cos_ABC_l118_118088

-- Definition of the problem setup
variables {a b c : ℝ}
variables {A B C : ℝ}    -- angles
variables {D : ℝ}        -- point on side AC

-- Conditions
axiom b_squared_eq_ac : b^2 = a * c
axiom BD_sin_ABC_eq_a_sin_C : D * Real.sin B = a * Real.sin C
axiom AD_eq_2DC : A = 2 * C

-- Part (1)
theorem part1_prove_BD_eq_b : D = b :=
by
  sorry

-- Part (2)
theorem part2_prove_cos_ABC :
  Real.cos B = 7 / 12 :=
by
  sorry

end part1_prove_BD_eq_b_part2_prove_cos_ABC_l118_118088


namespace mean_of_numbers_l118_118664

theorem mean_of_numbers :
  let numbers := [13, 8, 13, 21, 7, 23]
  let sum := 85
  let count := 6
  Float.round (sum / count) 2 = 14.17 :=
by
  let numbers := [13, 8, 13, 21, 7, 23]
  let sum := 85
  let count := 6
  exact sorry

end mean_of_numbers_l118_118664


namespace constant_term_expansion_l118_118425

open Real

theorem constant_term_expansion {α : ℝ} (h : sin (π - α) = 2 * cos α) :
  let t := tan α in
  let exp := (λ (x : ℝ) => (x + t / x)^6) in
  (coeff_zero_term exp) = 160 :=
by 
  let t := tan α
  sorry

noncomputable def coeff_zero_term (f : ℝ → ℝ) : ℝ :=
  sorry

end constant_term_expansion_l118_118425


namespace min_value_g_l118_118332

def g (x : ℝ) : ℝ := x + (x / (x^2 + 2)) + (x * (x + 3) / (x^2 + 3)) + (3 * (x + 1) / (x * (x^2 + 3)))

theorem min_value_g (x : ℝ) (hx : x > 0) : ∃ m : ℝ, m = 3.568 ∧ ∀ y : ℝ, y > 0 → g y ≥ m :=
by
  -- Proof goes here
  sorry

end min_value_g_l118_118332


namespace cos_squared_identity_l118_118439

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * real.sqrt ((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c)))

theorem cos_squared_identity (A B C a b c R : ℝ) (hA : A + B + C = π) (h_cosA : real.cos A = (b^2 + c^2 - a^2) / (2 * b * c)) (h_cosB : real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)) (h_cosC : real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) (h_R : R = circumradius a b c) :
  (real.cos A)^2 + (real.cos B)^2 + (real.cos C)^2 = 1 ↔ a^2 + b^2 + c^2 = 8 * R^2 ↔ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) :=
  sorry

end cos_squared_identity_l118_118439


namespace problem1_problem2_l118_118042

-- Problem 1: Given x < -2, prove |1 - |1 + x|| = -2 - x
theorem problem1 (x : ℝ) (h : x < -2) : |1 - |1 + x|| = -2 - x := 
by sorry

-- Problem 2: Given |a| = -a, prove |a-1| - |a-2| = -1
theorem problem2 (a : ℝ) (h : |a| = -a) : |a - 1| - |a - 2| = -1 := 
by sorry

end problem1_problem2_l118_118042


namespace relationship_among_a_b_c_l118_118718

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.1 * Real.log 2)
noncomputable def c : ℝ := Real.exp (1.3 * Real.log 0.2)

theorem relationship_among_a_b_c : a < c ∧ c < b :=
by
  have a_neg : a < 0 :=
    by sorry
  have b_pos : b > 1 :=
    by sorry
  have c_pos : c < 1 :=
    by sorry
  sorry

end relationship_among_a_b_c_l118_118718


namespace paint_cost_is_300_l118_118521

-- Define the conditions
def length : ℝ := 13.416407864998739
def breadth : ℝ := length / 3
def cost_per_sq_meter : ℝ := 5
def area_floor : ℝ := length * breadth
def total_cost : ℝ := area_floor * cost_per_sq_meter

-- State the theorem
theorem paint_cost_is_300 : total_cost = 300 := 
by 
  -- proof skipped
  sorry

end paint_cost_is_300_l118_118521


namespace processing_box_is_assignment_calculation_l118_118942

-- Define the types of boxes in a flowchart
inductive FlowchartBox
| processing    : FlowchartBox
| decision      : FlowchartBox
| terminal      : FlowchartBox
| input_output  : FlowchartBox

-- Define the functions of each type of box
def boxFunction : FlowchartBox → String
| FlowchartBox.processing := "Processes data, assigns values, computes"
| FlowchartBox.decision := "Determines program's execution based on conditions"
| FlowchartBox.terminal := "Represents start and end of the program"
| FlowchartBox.input_output := "Handles data input and output"

-- Define the possible answers
inductive FlowchartFunction
| assignment_calculation : FlowchartFunction
| input_information      : FlowchartFunction
| output_information     : FlowchartFunction
| start_end_algorithm    : FlowchartFunction

-- Map the box corresponding to each function
def correctAnswer : FlowchartBox → FlowchartFunction
| FlowchartBox.processing := FlowchartFunction.assignment_calculation
| FlowchartBox.decision := FlowchartFunction.start_end_algorithm
| FlowchartBox.terminal := FlowchartFunction.start_end_algorithm
| FlowchartBox.input_output := FlowchartFunction.input_information

-- The proof statement
theorem processing_box_is_assignment_calculation :
  correctAnswer FlowchartBox.processing = FlowchartFunction.assignment_calculation :=
by {
  -- Prove that by definition, processing box corresponds to "assignment, calculation".
  sorry
}

end processing_box_is_assignment_calculation_l118_118942


namespace Lauryn_employs_80_men_l118_118867

theorem Lauryn_employs_80_men (W M : ℕ) 
  (h1 : M = W - 20) 
  (h2 : M + W = 180) : 
  M = 80 := 
by 
  sorry

end Lauryn_employs_80_men_l118_118867


namespace percentage_reduction_consistency_l118_118437

theorem percentage_reduction_consistency 
  (initial_price new_price : ℝ) (X Y : ℝ)
  (h1 : initial_price = 3) (h2 : new_price = 5)
  (equal_expenditure : initial_price * X = new_price * Y) :
  ((X - Y) / X) * 100 = 40 := by
  -- Proof will go here
  sorry

end percentage_reduction_consistency_l118_118437


namespace dice_product_prob_divisible_by_4_l118_118206

/-- 
  Given that Una rolls 6 standard 6-sided dice simultaneously, 
  prove that the probability that the product of the 6 numbers obtained is divisible by 4 
  is 61/64.
-/
theorem dice_product_prob_divisible_by_4 :
  (fin 6 → fin 6 → ℚ) := sorry

end dice_product_prob_divisible_by_4_l118_118206


namespace probability_two_white_balls_same_color_l118_118971

theorem probability_two_white_balls_same_color :
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  (total_combinations_white + total_combinations_black > 0) →
  (total_combinations_white / total_combinations_same_color) = (3 / 4) :=
by
  let num_white := 3
  let num_black := 2
  let total_combinations_white := num_white.choose 2
  let total_combinations_black := num_black.choose 2
  let total_combinations_same_color := total_combinations_white + total_combinations_black
  intro h
  sorry

end probability_two_white_balls_same_color_l118_118971


namespace solve_system_eqs_l118_118688

theorem solve_system_eqs (x y : ℝ) (h1 : (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7)
                            (h2 : (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7) :
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) :=
sorry

end solve_system_eqs_l118_118688


namespace number_of_zeros_of_f_l118_118164

def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

theorem number_of_zeros_of_f : ∃! x : ℝ, f x = 0 := 
sorry

end number_of_zeros_of_f_l118_118164


namespace functional_equation_solution_l118_118353

theorem functional_equation_solution :
  ∀ (f : ℚ → ℝ), (∀ x y : ℚ, f (x + y) = f x + f y + 2 * x * y) →
  ∃ k : ℝ, ∀ x : ℚ, f x = x^2 + k * x :=
by
  sorry

end functional_equation_solution_l118_118353


namespace three_digit_multiples_of_30_not_75_l118_118774

theorem three_digit_multiples_of_30_not_75 : 
  let multiples_of_30 := list.range' 120 (990 - 120 + 30) 30
  let multiples_of_75 := list.range' 150 (900 - 150 + 150) 150
  let non_multiples_of_75 := multiples_of_30.filter (λ x => x % 75 ≠ 0)
  non_multiples_of_75.length = 24 := 
by 
  sorry

end three_digit_multiples_of_30_not_75_l118_118774


namespace num_nonnegative_integers_l118_118771

theorem num_nonnegative_integers : 
  ∃ (s : Finset ℤ), (∀ x ∈ s, ∃ (b : Fin 7 → Fin 5), x = ∑ i : Fin 7, (b i - 2) * 4^i) ∧ 
  (∀ x ∈ s, 0 ≤ x) ∧ 
  s.card = 10923 :=
begin
  sorry
end

end num_nonnegative_integers_l118_118771


namespace product_count_not_divisible_by_10_l118_118040

theorem product_count_not_divisible_by_10 : 
  let S := {2, 3, 5, 7, 13}
  in (∑ k in finset.powerset_len S.to_finset 2 S.to_finset, 
      if ¬ ((∏ x in k, x) % 10 == 0) then 1 else 0) 
   = 18
:= sorry

end product_count_not_divisible_by_10_l118_118040


namespace oxen_b_is_12_l118_118234

variable (oxen_b : ℕ)

def share (oxen months : ℕ) : ℕ := oxen * months

def total_share (oxen_a oxen_b oxen_c months_a months_b months_c : ℕ) : ℕ :=
  share oxen_a months_a + share oxen_b months_b + share oxen_c months_c

def proportion (rent_c rent total_share_c total_share : ℕ) : Prop :=
  rent_c * total_share = rent * total_share_c

theorem oxen_b_is_12 : oxen_b = 12 := by
  let oxen_a := 10
  let oxen_c := 15
  let months_a := 7
  let months_b := 5
  let months_c := 3
  let rent := 210
  let rent_c := 54
  let share_a := share oxen_a months_a
  let share_c := share oxen_c months_c
  let total_share_val := total_share oxen_a oxen_b oxen_c months_a months_b months_c
  let total_rent := share_a + 5 * oxen_b + share_c
  have h1 : proportion rent_c rent share_c total_rent := by sorry
  rw [proportion] at h1
  sorry

end oxen_b_is_12_l118_118234


namespace find_m_value_l118_118411

theorem find_m_value (m : ℝ) : (∃ M, M = {3, m + 1} ∧ 4 ∈ M) → m = 3 :=
by
  sorry

end find_m_value_l118_118411


namespace distinct_bead_arrangements_l118_118446

theorem distinct_bead_arrangements :
  let beads := 8
  let factorial (n : ℕ) : ℕ := (List.range n).product
  20160 = (factorial beads) / 2 :=
by
  sorry

end distinct_bead_arrangements_l118_118446


namespace determine_self_intersections_l118_118135

noncomputable def max_self_intersections (α : ℝ) : ℕ :=
  Nat.floor (180 / α)

theorem determine_self_intersections (α : ℝ) (hα1 : α > 0) (hα2 : α < 180) :
  ∃ n : ℕ, n = max_self_intersections α ∧ n * α < 180 :=
begin
  use max_self_intersections α,
  split,
  { refl },
  { sorry } -- proof of n * α < 180
end

end determine_self_intersections_l118_118135


namespace general_formula_sum_series_l118_118387

noncomputable def S : ℕ → ℕ
| 0       := 0
| (n + 1) := S n + 2 * (a n) + 1

noncomputable def a : ℕ → ℕ
| 0       := 3
| (n + 1) := 2 * a n + 1

theorem general_formula (n : ℕ) : a n = 2^(n + 1) - 1 := 
sorry

theorem sum_series (n : ℕ) : 
  (∑ i in finset.range n, 1 / (a (i + 1) + 1) : ℝ) < 1/2 :=
sorry

end general_formula_sum_series_l118_118387


namespace ketchup_bottles_count_l118_118317

def ratio_ketchup_mustard_mayo : Nat × Nat × Nat := (3, 3, 2)
def num_mayo_bottles : Nat := 4

theorem ketchup_bottles_count 
  (r : Nat × Nat × Nat)
  (m : Nat)
  (h : r = ratio_ketchup_mustard_mayo)
  (h2 : m = num_mayo_bottles) :
  ∃ k : Nat, k = 6 := by
sorry

end ketchup_bottles_count_l118_118317


namespace octagon_mass_is_19kg_l118_118711

-- Define the parameters given in the problem
def side_length_square_sheet := 1  -- side length in meters
def thickness_sheet := 0.3  -- thickness in cm (3 mm)
def density_steel := 7.8  -- density in g/cm³

-- Given the geometric transformations and constants, prove the mass of the octagon
theorem octagon_mass_is_19kg :
  ∃ mass : ℝ, (mass = 19) :=
by
  -- Placeholder for the proof.
  -- The detailed steps would include geometrical transformations and volume calculations,
  -- which have been rigorously defined in the problem and derived in the solution.
  sorry

end octagon_mass_is_19kg_l118_118711


namespace centers_covered_by_circle_l118_118917

variable {Point : Type} [metric_space Point]

theorem centers_covered_by_circle (X A B C : Point) (r : ℝ) (hr: r = 1):
  dist X A < r ∧ dist X B < r ∧ dist X C < r →
  ∃ (D : Point), ∀ (P : Point), (P = A ∨ P = B ∨ P = C) → dist D P ≤ r :=
by
  intro h
  use X
  intro P hp
  cases hp
  { rw hp
    apply (h.1).le }
  { cases hp
    { rw hp
      apply (h.1.right).le }
    { rw hp
      apply (h.2).le } }
  done

end centers_covered_by_circle_l118_118917


namespace omega_cannot_be_half_l118_118404

theorem omega_cannot_be_half: 
  ∀ (ω : ℝ), 0 < ω → (∀ x, -π/2 < x ∧ x < π/2 → derivative (λ x, cos (ω * x) - sin (ω * x)) x < 0) → 
  ω ≠ 1/2 := 
by 
  intros ω hω hDeriv 
  sorry

end omega_cannot_be_half_l118_118404


namespace value_of_fraction_is_four_l118_118182

theorem value_of_fraction_is_four : (2^8) / (8^2) = 4 := by
  have h1 : 8 = 2^3 := by sorry
  have h2 : 8^2 = (2^3)^2 := by rw [h1]
  calc
    (2^8) / (8^2) = (2^8) / ((2^3)^2) : by rw [h2]
              ... = (2^8) / (2^6)     : by rw [pow_mul]
              ... = 2^(8-6)          : by rw [pow_sub]
              ... = 2^2              : by sorry
              ... = 4                : by sorry

end value_of_fraction_is_four_l118_118182


namespace g_is_even_l118_118858

def g (x : ℝ) : ℝ := 4 / (3 * x^4 - 7)

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  intros x
  unfold g
  have hx : (-x)^4 = x^4 := by norm_num
  rw hx
  rfl

end g_is_even_l118_118858


namespace max_m_value_l118_118424

theorem max_m_value {m : ℝ} : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0 → x < m)) ∧ ¬(∀ x : ℝ, (x^2 - 2 * x - 8 > 0 ↔ x < m)) → m ≤ -2 :=
sorry

end max_m_value_l118_118424


namespace Kolya_is_correct_Valya_is_incorrect_l118_118291

noncomputable def Kolya_probability (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

noncomputable def Valya_probability_not_losing (x : ℝ) : ℝ :=
  let r := 1 / (x + 1)
  let p := 1 / x
  let s := x / (x + 1)
  let q := (x - 1) / x
  r / (1 - s * q)

theorem Kolya_is_correct (x : ℝ) (hx : x > 0) : Kolya_probability x = 1 / 2 :=
by
  sorry

theorem Valya_is_incorrect (x : ℝ) (hx : x > 0) : Valya_probability_not_losing x = 1 / 2 :=
by
  sorry

end Kolya_is_correct_Valya_is_incorrect_l118_118291


namespace cube_face_shading_l118_118520

theorem cube_face_shading {n : ℕ} (h : n = 4) :
  let total_cubes := n * n * n,
      face_shaded_cubes := 2 * (4 + 4),
      corners := 8 in
  total_cubes = 64 ∧ total_cubes = (h * h * h) ∧ face_shaded_cubes = 2 * 8  →
  8 + (6 * 4 - 8) = 24 :=
by
  intro h1 h2
  sorry

end cube_face_shading_l118_118520


namespace part1_prove_BD_eq_b_part2_prove_cos_ABC_l118_118089

-- Definition of the problem setup
variables {a b c : ℝ}
variables {A B C : ℝ}    -- angles
variables {D : ℝ}        -- point on side AC

-- Conditions
axiom b_squared_eq_ac : b^2 = a * c
axiom BD_sin_ABC_eq_a_sin_C : D * Real.sin B = a * Real.sin C
axiom AD_eq_2DC : A = 2 * C

-- Part (1)
theorem part1_prove_BD_eq_b : D = b :=
by
  sorry

-- Part (2)
theorem part2_prove_cos_ABC :
  Real.cos B = 7 / 12 :=
by
  sorry

end part1_prove_BD_eq_b_part2_prove_cos_ABC_l118_118089


namespace range_of_a_l118_118748

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (a / x - 4 / x^2 < 1)) → a < 4 := 
by
  sorry

end range_of_a_l118_118748


namespace isosceles_triangle_AC_length_l118_118914

-- Define the problem conditions
def isosceles_triangle (A B C M : Type) [Group A] [Group B] [Group C] [Group M]
  (AB BC AC : Real) (AM MB : Real) (angle_BMC : Real) := 
  AC = 2 * AM + 2 * (MB / 2 * (sqrt(3)/2)) -- This calculates AH as AM + 1.5 and AC as 2 * AH

-- Lean statement with conditions and expected answer AC = 17
theorem isosceles_triangle_AC_length
  (A B C M : Type) [Group A] [Group B] [Group C] [Group M]
  (AB BC : A ≃ B) (AC : A ≃ C) (AM : A ≃ M) (MB : M ≃ B) (angle_BMC : Real) :
  isosceles_triangle A B C M 1 7 3 (pi / 3) → AC = 17 :=
by 
  sorry

end isosceles_triangle_AC_length_l118_118914


namespace sum_of_squares_ge_const_term_squares_l118_118729

noncomputable def norm_squared (R : Polynomial ℝ) : ℝ :=
  R.coeffs.sum (λ c, c^2)

variable (P Q : Polynomial ℝ)

theorem sum_of_squares_ge_const_term_squares
    (hP_deg : P.natDegree > 0)
    (hQ_deg : Q.natDegree > 0)
    (hP_leading : P.leadingCoeff = 1)
    (hQ_leading : Q.leadingCoeff = 1) :
    norm_squared (P * Q) ≥ P.coeff 0 ^ 2 + Q.coeff 0 ^ 2 := 
  sorry

end sum_of_squares_ge_const_term_squares_l118_118729


namespace cookie_sheet_perimeter_l118_118604

theorem cookie_sheet_perimeter :
  let width_in_inches := 15.2
  let length_in_inches := 3.7
  let conversion_factor := 2.54
  let width_in_cm := width_in_inches * conversion_factor
  let length_in_cm := length_in_inches * conversion_factor
  2 * (width_in_cm + length_in_cm) = 96.012 :=
by
  sorry

end cookie_sheet_perimeter_l118_118604


namespace octal_multiplication_l118_118848

theorem octal_multiplication :
  let to_decimal (n : ℕ) (b : ℕ) : ℕ :=
    (n / 10) * b + (n % 10) in
  to_decimal 53 8 * to_decimal 26 8 = to_decimal 1662 8 :=
begin
  sorry
end

end octal_multiplication_l118_118848


namespace find_magnitude_l118_118033

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions as definitions
def unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop := ∥v∥ = 1
def condition : Prop := ∥a + 2 • b∥ = Real.sqrt 3

-- The theorem to prove
theorem find_magnitude (h₁ : unit_vector a) (h₂ : unit_vector b) (h₃ : condition) : 
  ∥a - 2 • b∥ = Real.sqrt 7 :=
sorry

end find_magnitude_l118_118033


namespace twenty_million_in_scientific_notation_l118_118933

/-- Prove that 20 million in scientific notation is 2 * 10^7 --/
theorem twenty_million_in_scientific_notation : 20000000 = 2 * 10^7 :=
by
  sorry

end twenty_million_in_scientific_notation_l118_118933


namespace sum_diff_probabilities_equal_l118_118716

def set := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def selected_twice (a b : ℕ) : Prop :=
  a ∈ set ∧ b ∈ set

noncomputable def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem sum_diff_probabilities_equal :
  (∃ a b, selected_twice a b ∧ divisible_by_3 (a + b)) =
  (∃ a b, selected_twice a b ∧ divisible_by_3 (a - b)) :=
begin
  sorry
end

end sum_diff_probabilities_equal_l118_118716


namespace geometric_sequence_log_sum_correct_l118_118059

noncomputable def geometric_sequence_log_sum (n : ℕ) (b : ℕ → ℝ) : ℝ :=
  if b 7 * b 8 = 9 then
    ∑ i in finset.range 14, log 3 (b (i + 1))
  else
    0

theorem geometric_sequence_log_sum_correct :
  ∀ (b : ℕ → ℝ), (∀ n, b n > 0) →
  (∃ r, ∀ n, b (n + 1) = b n * r) → 
  (b 7 * b 8 = 9) → 
  geometric_sequence_log_sum 14 b = 14 :=
by
  intros b pos_seq geometric_seq cond_b7b8
  simp [geometric_sequence_log_sum, cond_b7b8]
  sorry

end geometric_sequence_log_sum_correct_l118_118059


namespace full_spots_in_garage_186_l118_118268

def parking_garage : Type := 
{ stories : ℕ
, spots_per_level : ℕ
, open_spots_first : ℕ
, diff_second_first : ℕ
, diff_third_second : ℕ
, open_spots_fourth : ℕ
}

def garage : parking_garage := 
{ stories := 4
, spots_per_level := 100
, open_spots_first := 58
, diff_second_first := 2
, diff_third_second := 5
, open_spots_fourth := 31
}

def calc_full_spots (g : parking_garage) : ℕ := 
let open_spots_first := g.open_spots_first in
let open_spots_second := open_spots_first + g.diff_second_first in
let open_spots_third := open_spots_second + g.diff_third_second in
let open_spots_fourth := g.open_spots_fourth in
let total_open_spots := open_spots_first + open_spots_second + open_spots_third + open_spots_fourth in
let total_spots := g.stories * g.spots_per_level in
total_spots - total_open_spots

theorem full_spots_in_garage_186 : calc_full_spots garage = 186 := by
  sorry

end full_spots_in_garage_186_l118_118268


namespace solve_problem1_solve_problem2_l118_118767

def problem1 : Prop :=
  ∀ θ : ℝ, (0 < θ ∧ θ < π/2) →
  ∃ (sin_theta cos_theta : ℝ), 
    (sin_theta = 2 * cos_theta) ∧
    (sin_theta^2 + cos_theta^2 = 1) ∧
    (sin_theta = 2 * sqrt 5 / 5) ∧
    (cos_theta = sqrt 5 / 5)

def problem2 : Prop :=
  ∀ (θ φ : ℝ), (0 < θ ∧ θ < π/2) ∧ (0 < φ ∧ φ < π/2) →
  (sin (θ - φ) = sqrt 10 / 10) →
  (cos θ = sqrt 5 / 5) ∧ (sin θ = 2 * sqrt 5 / 5) →
  cos φ = sqrt 2 / 2

theorem solve_problem1 : problem1 := by
  sorry

theorem solve_problem2 : problem2 := by
  sorry

end solve_problem1_solve_problem2_l118_118767


namespace a_b_sum_of_powers_l118_118910

variable (a b : ℝ)

-- Conditions
def condition1 := a + b = 1
def condition2 := a^2 + b^2 = 3
def condition3 := a^3 + b^3 = 4
def condition4 := a^4 + b^4 = 7
def condition5 := a^5 + b^5 = 11

-- Theorem statement
theorem a_b_sum_of_powers (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b) 
  (h4 : condition4 a b) (h5 : condition5 a b) : a^10 + b^10 = 123 :=
sorry

end a_b_sum_of_powers_l118_118910


namespace final_score_l118_118681

theorem final_score (questions_first_half questions_second_half : Nat)
  (points_correct points_incorrect : Int)
  (correct_first_half incorrect_first_half correct_second_half incorrect_second_half : Nat) :
  questions_first_half = 10 →
  questions_second_half = 15 →
  points_correct = 3 →
  points_incorrect = -1 →
  correct_first_half = 6 →
  incorrect_first_half = 4 →
  correct_second_half = 10 →
  incorrect_second_half = 5 →
  (points_correct * correct_first_half + points_incorrect * incorrect_first_half 
   + points_correct * correct_second_half + points_incorrect * incorrect_second_half) = 39 := 
by
  intros
  sorry

end final_score_l118_118681


namespace area_of_triangle_abe_l118_118313

theorem area_of_triangle_abe
    (A B C D E : Type)
    [triangle : is_right_triangle A B C]
    (angle_C : angle C = 90)
    (len_AC : length AC = 2)
    (len_BC : length BC = 1)
    (on_line_AC : point_on_line D AC)
    (folding : after_folding triangle A B D)
    (perpendicular : is_perpendicular AD DE) :
  area A B E = 1.5 :=
by sorry

end area_of_triangle_abe_l118_118313


namespace find_dividend_l118_118058

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 38) (h_quotient : quotient = 19) (h_remainder : remainder = 7) :
  divisor * quotient + remainder = 729 := by
  sorry

end find_dividend_l118_118058


namespace sqrt_mul_eq_l118_118574

theorem sqrt_mul_eq {a b: ℝ} (ha: 0 ≤ a) (hb: 0 ≤ b): sqrt(a * b) = sqrt(a) * sqrt(b) :=
  by sorry

example : sqrt(2) * sqrt(3) = sqrt(6) :=
  by apply sqrt_mul_eq; norm_num

end sqrt_mul_eq_l118_118574


namespace distinguishable_large_triangles_l118_118545

theorem distinguishable_large_triangles (colors : ℕ) (corner_colors : ℕ) (total_combinations : ℕ) :
    colors = 6 →
    corner_colors = 6 + (6 * 5) + (nat.choose 6 3) →
    total_combinations = corner_colors * colors →
    total_combinations = 336 :=
by
  intros hcolors hcorner_colors htotal_combinations
  sorry


end distinguishable_large_triangles_l118_118545


namespace total_germs_calculation_l118_118073

def number_of_dishes : ℕ := 10800
def germs_per_dish : ℕ := 500
def total_germs : ℕ := 5400000

theorem total_germs_calculation : germs_per_ddish * number_of_idshessh = total_germs :=
by sorry

end total_germs_calculation_l118_118073


namespace number_of_satisfying_sets_l118_118486

-- Let A be the set {1, 2}
def A : Set ℕ := {1, 2}

-- Define a predicate for sets B that satisfy A ∪ B = {1, 2, 3}
def satisfiesCondition (B : Set ℕ) : Prop :=
  (A ∪ B = {1, 2, 3})

-- The theorem statement asserting there are 4 sets B satisfying the condition
theorem number_of_satisfying_sets : (Finset.filter satisfiesCondition (Finset.powerset {1, 2, 3})).card = 4 :=
by sorry

end number_of_satisfying_sets_l118_118486


namespace fractional_sum_values_l118_118151

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem fractional_sum_values (x y z t : ℝ)
  (h1 : fractional_part (x + y + z) = 1 / 4)
  (h2 : fractional_part (y + z + t) = 1 / 4)
  (h3 : fractional_part (z + t + x) = 1 / 4)
  (h4 : fractional_part (t + x + y) = 1 / 4) :
  fractional_part (x + y + z + t) ∈ {0, 1 / 3, 2 / 3} :=
sorry

end fractional_sum_values_l118_118151


namespace range_of_f_condition_l118_118403

noncomputable def f : ℝ → ℝ
| x := if -1 ≤ x ∧ x ≤ 1 then 3 * (1 - 2)^x / (2^x + 1)
       else -(1/4) * (x^3 + 3*x)

theorem range_of_f_condition :
  (∀ x : ℝ, (∀ m : ℝ, -3 ≤ m ∧ m ≤ 2 → f (m * x - 1) + f x > 0)
     ↔ (-1 / 2 < x ∧ x < 1 / 3)) :=
by {
   -- Proof omitted
   sorry
}

end range_of_f_condition_l118_118403


namespace device_works_probability_l118_118616

noncomputable def device_working_probability : ℝ :=
let p_damaged := 0.1 in
let p_not_damaged := 1 - p_damaged in
p_not_damaged * p_not_damaged

theorem device_works_probability (p_damaged : ℝ) (h0 : p_damaged = 0.1) :
  device_working_probability = 0.81 := by
  rw [←h0]
  dsimp [device_working_probability]
  norm_num
  sorry

end device_works_probability_l118_118616


namespace probability_nine_heads_l118_118987

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n + 1, k + 1 => binom n k + binom n (k + 1)

def flips : ℕ := 12
def heads : ℕ := 9
def total_outcomes : ℕ := 2^flips
def favorable_outcomes : ℕ := binom flips heads
def probability : Rat := favorable_outcomes / total_outcomes.toRat

theorem probability_nine_heads :
  probability = (220/4096) := by
  sorry

end probability_nine_heads_l118_118987


namespace polynomial_condition_satisfied_l118_118697

-- Definitions as per conditions:
def p (x : ℝ) : ℝ := x^2 + 1

-- Conditions:
axiom cond1 : p 3 = 10
axiom cond2 : ∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2

-- Theorem to prove:
theorem polynomial_condition_satisfied : (p 3 = 10) ∧ (∀ (x y : ℝ), p x * p y = p x + p y + p (x * y) - 2) :=
by
  apply And.intro cond1
  apply cond2

end polynomial_condition_satisfied_l118_118697


namespace graph_fixed_point_l118_118944

theorem graph_fixed_point (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) : 
  ∃ (x y : ℝ), (x = 2) ∧ (y = 2) ∧ (y = a^(x-2) + 1) :=
by {
  use [2, 2],
  split,
  { refl },
  split,
  { refl },
  { sorry }
}

end graph_fixed_point_l118_118944


namespace find_k_l118_118356

noncomputable def k_solutions : set ℝ :=
{k : ℝ | ∥k • ⟨3, -4, 1⟩ - ⟨6, 9, -2⟩∥ = 3 * real.sqrt 26}

theorem find_k (k : ℝ) : k ∈ k_solutions ↔ k = 1.478 ∨ k = -3.016 :=
by sorry

end find_k_l118_118356


namespace Kolya_correct_Valya_incorrect_l118_118302

-- Kolya's Claim (Part a)
theorem Kolya_correct (x : ℝ) (p r : ℝ) (hpr : r = 1/(x+1) ∧ p = 1/x) : 
  (p / (1 - (1 - r) * (1 - p))) = (r / (1 - (1 - r) * (1 - p))) :=
sorry

-- Valya's Claim (Part b)
theorem Valya_incorrect (x : ℝ) (p r : ℝ) (q s : ℝ) (hprs : r = 1/(x+1) ∧ p = 1/x ∧ q = 1 - p ∧ s = 1 - r) : 
  ((q * r / (1 - s * q)) + (p * r / (1 - s * q))) = 1/2 :=
sorry

end Kolya_correct_Valya_incorrect_l118_118302


namespace multiplication_of_exponents_l118_118583

theorem multiplication_of_exponents (x : ℝ) : (x ^ 4) * (x ^ 2) = x ^ 6 := 
by
  sorry

end multiplication_of_exponents_l118_118583


namespace distance_between_skew_lines_l118_118382

open Real EuclideanGeometry

noncomputable def cube_distance (a : ℝ) : ℝ := a * sqrt 3 / 3

theorem distance_between_skew_lines (a : ℝ) (h : 0 < a) :
  let AC1 := a * sqrt 3 in
  let EF := AC1 / 3 in
  EF = cube_distance a :=
by sorry

end distance_between_skew_lines_l118_118382


namespace Kolya_correct_Valya_incorrect_l118_118300

-- Kolya's Claim (Part a)
theorem Kolya_correct (x : ℝ) (p r : ℝ) (hpr : r = 1/(x+1) ∧ p = 1/x) : 
  (p / (1 - (1 - r) * (1 - p))) = (r / (1 - (1 - r) * (1 - p))) :=
sorry

-- Valya's Claim (Part b)
theorem Valya_incorrect (x : ℝ) (p r : ℝ) (q s : ℝ) (hprs : r = 1/(x+1) ∧ p = 1/x ∧ q = 1 - p ∧ s = 1 - r) : 
  ((q * r / (1 - s * q)) + (p * r / (1 - s * q))) = 1/2 :=
sorry

end Kolya_correct_Valya_incorrect_l118_118300


namespace rectangle_area_ratio_l118_118532

theorem rectangle_area_ratio (length width diagonal : ℝ) (h_ratio : length / width = 5 / 2) (h_diagonal : diagonal = 13) :
    ∃ k : ℝ, (length * width) = k * diagonal^2 ∧ k = 10 / 29 :=
by
  sorry

end rectangle_area_ratio_l118_118532


namespace smallest_y_l118_118569

theorem smallest_y (y : ℝ) :
  (3 * y ^ 2 + 33 * y - 90 = y * (y + 16)) → y = -10 :=
sorry

end smallest_y_l118_118569


namespace probability_of_odd_score_l118_118615

-- Definitions of various conditions based on the problem
def inner_circle_radius := 4
def outer_circle_radius := 8

def point_values_inner_regions := [3, 4, 4]
def point_values_outer_regions := [5, 3, 3]

def doubled_point_values_outer_regions := point_values_outer_regions.map (λ x, 2 * x)

-- The hypothesis that the final probability of an odd score when two darts are thrown is 5/12
theorem probability_of_odd_score :
  let inner_circle_area := π * inner_circle_radius^2
  let outer_circle_area := π * outer_circle_radius^2 - inner_circle_area
  let total_area := outer_circle_area + inner_circle_area

  let point_area (points : List ℤ) (area : ℝ) : ℝ :=
    area / (points.length : ℝ)
  
  let inner_area := point_area point_values_inner_regions inner_circle_area
  let outer_area := point_area point_values_outer_regions outer_circle_area
  let total_area_of_odds := (inner_area * (point_values_inner_regions.count (λ x, x % 2 = 1))) 
                      + (outer_area * (point_values_outer_regions.map (λ x, 2 * x).count (λ x, x % 2 = 1)))
  let total_area_of_evens := total_area - total_area_of_odds
  
  let probability_odd := total_area_of_odds / total_area
  let probability_even := total_area_of_evens / total_area
  
  (2 * (probability_odd * probability_even)) = (5 / 12) :=
by
  sorry

end probability_of_odd_score_l118_118615


namespace magnitude_product_l118_118349

theorem magnitude_product : (|Complex.mk 3 (-5)| * |Complex.mk 3 5| = 34) := 
by
  sorry

end magnitude_product_l118_118349


namespace wheel_revolutions_l118_118162

theorem wheel_revolutions (diameter : ℝ) (distance_miles : ℝ) (feet_per_mile : ℝ) (radius : ℝ) (circumference : ℝ) (distance_feet : ℝ) (N : ℝ) :
  diameter = 8 →
  distance_miles = 3 →
  feet_per_mile = 5280 →
  radius = diameter / 2 →
  circumference = 2 * Real.pi * radius →
  distance_feet = distance_miles * feet_per_mile →
  N = distance_feet / circumference →
  N = 1980 / Real.pi :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3] at *
  rw h4 at h5
  rw h6 at *
  rw h5 at h7
  sorry

end wheel_revolutions_l118_118162


namespace P_on_bisector_angle_AOD_l118_118512

-- Define our points and quadrilateral
variables {A B C D O P : Type}

-- Conditions
-- The diagonals of the convex quadrilateral ABCD are equal
axiom diagonals_equal : dist A C = dist B D
-- The diagonals intersect at point O
axiom diagonals_intersect : segment A C ∩ segment B D = {O}
-- Point P inside triangle AOD
axiom P_in_AOD : ∃ (t s : ℝ), 0 < t ∧ 0 < s ∧ t + s < 1 ∧ P = (1 - t - s) • A + t • O + s • D
-- CD ∥ BP and AB ∥ CP
axiom CD_parallel_BP : parallel (line CD) (line BP)
axiom AB_parallel_CP : parallel (line AB) (line CP)

-- Prove that P lies on the bisector of ∠AOD
theorem P_on_bisector_angle_AOD :
  lies_on_bisector P (angle A O D) :=
sorry

end P_on_bisector_angle_AOD_l118_118512


namespace coins_cover_percentage_l118_118245

theorem coins_cover_percentage (r : ℝ) (h_pos : r > 0) :
  let area_triangle := (sqrt 3 / 4) * (2 * r)^2 in
  let area_coins := 3 * π * r^2 in
  (area_coins / area_triangle) * 100 = (50 / sqrt 3) * π := sorry

end coins_cover_percentage_l118_118245


namespace S_calculation_T_calculation_l118_118671

def S (a b : ℕ) : ℕ := 4 * a + 6 * b
def T (a b : ℕ) : ℕ := 5 * a + 3 * b

theorem S_calculation : S 6 3 = 42 :=
by sorry

theorem T_calculation : T 6 3 = 39 :=
by sorry

end S_calculation_T_calculation_l118_118671


namespace exists_sequence_with_properties_l118_118336

theorem exists_sequence_with_properties :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ, (∃ m : ℕ, m ≤ n ∧ a m = n) ∧
  (∀ k : ℕ, 0 < k → ∑ i in Finset.range (k + 1), a i % (k + 1) = 0)) :=
by
  sorry

end exists_sequence_with_properties_l118_118336


namespace leading_coefficient_of_g_l118_118169

theorem leading_coefficient_of_g (g : ℤ → ℤ) (h : ∀ x : ℤ, g(x + 1) - g(x) = 8 * x + 6) : ∃ d : ℤ, leading_coeff 4 (λ x, 4 * x * x + 4 * x + d) :=
by sorry

end leading_coefficient_of_g_l118_118169


namespace number_of_zeros_l118_118165

noncomputable def f (x : ℝ) := log x + 2 * x - 1

theorem number_of_zeros :
  ∃! x : ℝ, 0 < x ∧ f x = 0 
:= sorry

end number_of_zeros_l118_118165


namespace choir_members_l118_118526

theorem choir_members (n : ℕ) : 
  (∃ k m : ℤ, n + 4 = 10 * k ∧ n + 5 = 11 * m) ∧ 200 < n ∧ n < 300 → n = 226 :=
by 
  sorry

end choir_members_l118_118526


namespace find_x0_l118_118017

noncomputable def f (x : ℝ) : ℝ := Math.log x + x

theorem find_x0 :
  ∃ x0 : ℝ, x0 = 1 / 2 ∧
    let tangent_line := (λ x : ℝ, f x0 + (f' x0) * (x - x0)) in
    let given_line := (λ x : ℝ, 3 * x - 1) in
    (∀ x, tangent_line x = x - 3 = given_line x) :=
sorry

end find_x0_l118_118017


namespace function_passes_through_fixed_point_l118_118946

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  ∃ k : ℝ, k = (2, 2) :=
by
  use (2, 2)
  sorry

end function_passes_through_fixed_point_l118_118946


namespace triangle_MOI_area_is_3_4_l118_118055

variables {A B C M O I : Type}
variables (AB AC BC : ℝ) (I O M : Type)
variables [Euclidean_space O] [Euclidean_space I] [Euclidean_space M]
variables
  (coords_A : O::coord)
  (coords_B : O::coord)
  (coords_C : O::coord)
  (coords_I : I::coord)
  (coords_O : O::coord)
  (coords_M : M::coord)

-- Definitions based on the problem
def pts
:= [coords_A, coords_B, coords_C] : list 𝕂

def circumcenter
:= coords_O = ((C - A) / 2)

def incenter
:= coords_I = ((B-C) / 2)

def MOI_area
:= (1 / 2) abs ( MOI ((O - I) - (O - M)) + ((M + (I - O))) - ((I - M)))

-- Statement of the theorem
theorem triangle_MOI_area_is_3_4 :=
  begin
    let AB := 15, AC := 8, BC := 17,
    let coords_A := (0,0),
    let coords_B := (8,0),
    let coords_C := (0,17),
    let coords_I := (3.4,3.4),
    let coords_O := (4,8.5),
    let coords_M := (5,5),
    -- Prove the area is 3.4
    sorry
  end

end triangle_MOI_area_is_3_4_l118_118055


namespace movie_theater_seating_l118_118836

theorem movie_theater_seating :
  ∃ (ways : ℕ), ways = 40 :=
by
  -- Given Conditions
  let seats : ℕ := 10
  let people : ℕ := 3
  let gaps : ℕ := 7
  let middle_gaps : ℕ := 6

  -- Number of ways to choose 3 gaps out of 6 where A will sit in the middle
  let choose_gaps := nat.choose middle_gaps people

  -- Number of ways to arrange the remaining two people
  let arrange_remaining := 2 * 1 -- i.e., A_2^2

  -- Total number of ways
  let ways : ℕ := choose_gaps * arrange_remaining
  -- Checking the answer
  have : ways = 40 := sorry
  
  exact ⟨ways, this⟩

end movie_theater_seating_l118_118836


namespace toms_dog_age_in_six_years_l118_118550

-- Define the conditions as hypotheses
variables (B T D : ℕ)

-- Conditions
axiom h1 : B = 4 * D
axiom h2 : T = B - 3
axiom h3 : B + 6 = 30

-- The proof goal: Tom's dog's age in six years
theorem toms_dog_age_in_six_years : D + 6 = 12 :=
  sorry -- Proof is omitted based on the instructions

end toms_dog_age_in_six_years_l118_118550


namespace min_pieces_per_orange_l118_118459

theorem min_pieces_per_orange (oranges : ℕ) (calories_per_orange : ℕ) (people : ℕ) (calories_per_person : ℕ) (pieces_per_orange : ℕ) :
  oranges = 5 →
  calories_per_orange = 80 →
  people = 4 →
  calories_per_person = 100 →
  pieces_per_orange ≥ 4 :=
by
  intro h_oranges h_calories_per_orange h_people h_calories_per_person
  sorry

end min_pieces_per_orange_l118_118459


namespace abc_le_one_eighth_l118_118873

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end abc_le_one_eighth_l118_118873


namespace value_of_x_l118_118183

theorem value_of_x (x : ℕ) (h : x + (10 * x + x) = 12) : x = 1 := by
  sorry

end value_of_x_l118_118183


namespace total_players_l118_118539

noncomputable def women (men : ℝ) : ℝ := men + 6
noncomputable def men_to_women_ratio (men women : ℝ) : ℝ := men / women

theorem total_players : ∃ (M W : ℝ), W = M + 6 ∧ M / W = 0.45454545454545453 ∧ M + W = 16 :=
by
  have h1 : ∀ M W, W = M + 6 → M / W = 0.45454545454545453 → M + W = 16 := sorry
  exact ⟨5, 11, rfl, rfl, by exact h1 5 11 rfl rfl⟩

end total_players_l118_118539


namespace henry_apple_weeks_l118_118037

theorem henry_apple_weeks (apples_per_box : ℕ) (boxes : ℕ) (people : ℕ) (apples_per_day : ℕ) (days_per_week : ℕ) :
  apples_per_box = 14 → boxes = 3 → people = 2 → apples_per_day = 1 → days_per_week = 7 →
  (apples_per_box * boxes) / (people * apples_per_day * days_per_week) = 3 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end henry_apple_weeks_l118_118037


namespace characterized_rectangle_region_l118_118326

/-- Characterize the region determined by points in the Cartesian coordinate system such
that the first coordinate is the perimeter of a rectangle and the second coordinate is
the area of the same rectangle. -/
theorem characterized_rectangle_region (k t : ℝ) (h₁ : 0 < k) (h₂ : 0 < t) :
  4 * t ≤ k^2 ↔ t ≤ (k / 4)^2 :=
begin
  sorry
end

end characterized_rectangle_region_l118_118326


namespace line_tangents_correct_l118_118337

-- Define the circle equation
def circle_1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the point (3, 1)
def point_P : (ℝ × ℝ) := (3, 1)

-- Define line equation which we need to prove is the correct line joining points of tangency
def line_AB (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Define existence of points A and B on the circle which are the points of tangency
axiom points_A_B (A B : ℝ × ℝ) (A_on_circle : circle_1 A.1 A.2) (B_on_circle : circle_1 B.1 B.2)
  (tangent_A : (A.2 - 1) * (A.1 - 1) + 1 = 0) 
  (tangent_B : (B.2 - 1) * (B.1 - 1) + 1 = 0) 
  (line_A_B_through_tangent_points : line_AB A.1 A.2 ∧ line_AB B.1 B.2) :
  True

-- Purpose proof statement
theorem line_tangents_correct : 
  ∃ A B : ℝ × ℝ, circle_1 A.1 A.2 ∧ circle_1 B.1 B.2 ∧
  (A.2 - 1) * (A.1 - 1) + 1 = 0 ∧ (B.2 - 1) * (B.1 - 1) + 1 = 0 ∧
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 :=
begin
  sorry
end

end line_tangents_correct_l118_118337


namespace james_payment_l118_118860

theorem james_payment (cost_first_100_channels : ℕ)
  (cost_next_100_channels : ℕ)
  (total_cost : ℕ)
  (james_payment : ℕ) : cost_first_100_channels = 100 →
  cost_next_100_channels = cost_first_100_channels / 2 →
  total_cost = cost_first_100_channels + cost_next_100_channels →
  james_payment = total_cost / 2 →
  james_payment = 75 := 
by
  intros h1 h2 h3 h4
  rw [h1] at h2
  rw [h2, h1] at h3
  rw [h3] at h4
  assumption
  sorry

end james_payment_l118_118860


namespace sufficient_but_not_necessary_l118_118179

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬(∀ y : ℝ, (x < -1 ∨ y > 1) → (y < -1)) :=
by
  -- This means we would prove that if x < -1, then x < -1 ∨ x > 1 holds (sufficient),
  -- and show that there is a case (x > 1) where x < -1 is not necessary for x < -1 ∨ x > 1. 
  sorry

end sufficient_but_not_necessary_l118_118179


namespace option_e_forms_cube_l118_118857

-- Define each option as a type
inductive Shape
| T
| Irregular
| L
| StrangeCombination
| Cross

def option_a := (Shape.T, Shape.T)
def option_b := (Shape.T, Shape.Irregular)
def option_c := (Shape.L, Shape.L)
def option_d := Shape.StrangeCombination
def option_e := (Shape.Cross, Shape.Cross)

-- Predicate to check if a given option can form a cube
def can_form_cube : (Shape × Shape) → Prop
| (Shape.Cross, Shape.Cross) := true
| _ := false

-- The theorem we need to prove
theorem option_e_forms_cube : can_form_cube option_e = true :=
by sorry

end option_e_forms_cube_l118_118857


namespace result_l118_118296

noncomputable def Kolya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  P_A = (1: ℝ) / 2

def Valya_right (p r : ℝ) := 
  let q := 1 - p
  let s := 1 - r
  let P_A := r / (1 - s * q)
  let P_B := (r * p) / (1 - s * q)
  let P_AorB := P_A + P_B
  P_AorB > (1: ℝ) / 2

theorem result (p r : ℝ) (h1 : Kolya_right p r) (h2 : ¬Valya_right p r) :
  Kolya_right p r ∧ ¬Valya_right p r :=
  ⟨h1, h2⟩

end result_l118_118296
