import Mathlib

namespace eighth_number_is_148_l752_752869

-- Define the digit sum function
def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

-- Define the predicate for our number
def meets_conditions (n : ℕ) : Prop :=
  0 < n ∧ digit_sum n = 13

-- Define the desired property
theorem eighth_number_is_148 :
  ∃ (s : List ℕ), s.length = 8 ∧ 
    (∀ (i : ℕ), i < 8 → meets_conditions (s.nth_le i (by linarith))) ∧
    List.nth s 7 = some 148 :=
by sorry

end eighth_number_is_148_l752_752869


namespace problem_inequality_l752_752935

variable {α : Type*} [LinearOrder α]

def M (x y : α) : α := max x y
def m (x y : α) : α := min x y

theorem problem_inequality (a b c d e : α) (h : a < b) (h1 : b < c) (h2 : c < d) (h3 : d < e) : 
  M (M a (m b c)) (m d (m a e)) = b := sorry

end problem_inequality_l752_752935


namespace enclosed_area_tangents_arc_l752_752835

variables (R α : ℝ)

def tangent_point_area (R α : ℝ) : ℝ :=
  R^2 * (Real.cot (α / 2) - (1 / 2) * (Real.pi - α))

theorem enclosed_area_tangents_arc (R α : ℝ) :
  tangent_point_area R α =
  R^2 * (Real.cot (α / 2) - (1 / 2) * (Real.pi - α)) :=
sorry

end enclosed_area_tangents_arc_l752_752835


namespace non_negative_integral_values_satisfy_condition_l752_752649

theorem non_negative_integral_values_satisfy_condition :
  { x : ℕ | ⌊x / 5⌋ = ⌊x / 7⌋ } = {0, 1, 2, 3, 4, 7, 8, 9, 14} := by
  sorry

end non_negative_integral_values_satisfy_condition_l752_752649


namespace intersection_is_line_l752_752635

-- Definitions of the cylindrical coordinates
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

-- The problem and conditions
def θ1 : ℝ := arbitrary
def θ2 : ℝ := arbitrary
axiom θ1_ne_θ2 : θ1 ≠ θ2

-- The goal is to prove that the intersection of planes θ = θ1 and θ = θ2 is a line.
theorem intersection_is_line :
  {p : CylindricalCoord | p.θ = θ1} ∩ {p : CylindricalCoord | p.θ = θ2} = {p : CylindricalCoord | p.r = 0 ∧ p.θ = θ1 ∧ p.θ = θ2 ∧ ∃ z, p.z = z} :=
by
  sorry

end intersection_is_line_l752_752635


namespace find_circle_radius_l752_752210

-- Definitions based on the given conditions
def circle_eq (x y : ℝ) : Prop := (x^2 - 8*x + y^2 - 10*y + 34 = 0)

-- Problem statement
theorem find_circle_radius (x y : ℝ) : circle_eq x y → ∃ r : ℝ, r = Real.sqrt 7 :=
by
  sorry

end find_circle_radius_l752_752210


namespace sum_coefficients_rational_terms_l752_752064

theorem sum_coefficients_rational_terms 
  (x : ℝ) : (1 + real.sqrt x) ^ 6 = 32 :=
by
  sorry

end sum_coefficients_rational_terms_l752_752064


namespace coordinates_of_P_l752_752502

-- Definitions of conditions
def inFourthQuadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0
def absEqSeven (x : ℝ) : Prop := |x| = 7
def ysquareEqNine (y : ℝ) : Prop := y^2 = 9

-- Main theorem
theorem coordinates_of_P (x y : ℝ) (hx : absEqSeven x) (hy : ysquareEqNine y) (hq : inFourthQuadrant x y) :
  (x, y) = (7, -3) :=
  sorry

end coordinates_of_P_l752_752502


namespace unique_a_for_three_distinct_real_solutions_l752_752928

theorem unique_a_for_three_distinct_real_solutions (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 - 2 * x + 1 - 3 * |x|) ∧
  ((∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) ∧
  (∀ x4 : ℝ, f x4 = 0 → (x4 = x1 ∨ x4 = x2 ∨ x4 = x3) )) ) ↔
  a = 1 / 4 :=
sorry

end unique_a_for_three_distinct_real_solutions_l752_752928


namespace area_of_regular_hexagon_l752_752235

def side_length : ℝ := 8
def area_of_hexagon (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2
  
theorem area_of_regular_hexagon : area_of_hexagon side_length = 96 * Real.sqrt 3 := by
  sorry

end area_of_regular_hexagon_l752_752235


namespace subset_A_B_l752_752372

theorem subset_A_B (a : ℝ) (A : Set ℝ := {0, -a}) (B : Set ℝ := {1, a-2, 2a-2}) 
  (h : A ⊆ B) : a = 1 := by
  sorry

end subset_A_B_l752_752372


namespace interest_problem_l752_752866

open Real

noncomputable def interest_cents_credited (T : ℝ) (A : ℝ) (f : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  let Aprime := A + f
  let P := Aprime / (1 + r * t)
  let Interest := Aprime - P
  Interest - Interest.toInt

theorem interest_problem
  (A : ℝ) -- Total amount after interest and fee deduction
  (f : ℝ) -- Fee deducted
  (r : ℝ) -- Annual interest rate
  (t : ℝ) -- Time period in years
  (cents_credited : ℝ) -- Cents part of interest credited
  (hA : A = 317.44) 
  (hf : f = 5) 
  (hr : r = 0.08) 
  (ht : t = 3/12)
  (hcents : cents_credited = 32) :
  interest_cents_credited 317.44 317.44 5 0.08 (3/12) = 0.32 := sorry

end interest_problem_l752_752866


namespace max_non_overlapping_areas_l752_752486

theorem max_non_overlapping_areas (n : ℕ) : 
  let areas := 2*n + 2 + (2*n + 2*n + 1) in
  areas = 3*n + 3 :=
by
  sorry

end max_non_overlapping_areas_l752_752486


namespace Kelly_initial_games_eq_106_l752_752695

def initial_games (given_away : ℕ) (left : ℕ) := given_away + left

theorem Kelly_initial_games_eq_106 : initial_games 64 42 = 106 :=
by
  rw [initial_games]
  exact Nat.add_comm 64 42
  simp
  exact Nat.add_comm 42 64
  sorry

end Kelly_initial_games_eq_106_l752_752695


namespace average_of_seven_starting_with_d_l752_752394

theorem average_of_seven_starting_with_d (c d : ℕ) (h : d = (c + 3)) : 
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6 :=
by
  sorry

end average_of_seven_starting_with_d_l752_752394


namespace inequality_solution_set_compare_mn_and_2m_plus_2n_l752_752259

def f (x : ℝ) : ℝ := |x| + |x - 3|

theorem inequality_solution_set :
  {x : ℝ | f x - 5 ≥ x} = { x : ℝ | x ≤ -2 / 3 } ∪ { x : ℝ | x ≥ 8 } :=
sorry

theorem compare_mn_and_2m_plus_2n (m n : ℝ) (hm : ∃ x, m = f x) (hn : ∃ x, n = f x) :
  2 * (m + n) < m * n + 4 :=
sorry

end inequality_solution_set_compare_mn_and_2m_plus_2n_l752_752259


namespace find_accomplice_profession_l752_752156

universe u

-- Define the roles
inductive Role
| thief
| accomplice
| falsely_accused

-- Define professions as strings (for simplicity)
def Painter : String := "Painter"
def PianoTuner : String := "PianoTuner"
def Decorator : String := "Decorator"
def Doctor : String := "Doctor"
def InsuranceAgent : String := "InsuranceAgent"

-- Define the suspects
inductive Suspect
| Bertrand
| Alfred
| Charles

-- Define statements within type indicating who stated it and what they claimed
structure Statement where
  suspect : Suspect
  assertion : Suspect → String

-- Given statements in the conditions
def Bertrand_statements : List Statement :=
[
  { suspect := Suspect.Bertrand, assertion := λ s => if s = Suspect.Bertrand then Painter else
                                                        if s = Suspect.Alfred then PianoTuner else
                                                        if s = Suspect.Charles then Decorator else "" }
]

def Alfred_statements : List Statement :=
[
  { suspect := Suspect.Alfred, assertion := λ s => if s = Suspect.Alfred then Doctor else
                                                     if s = Suspect.Charles then InsuranceAgent else
                                                     if s = Suspect.Bertrand then Painter else "" }
]

def Charles_statements : List Statement :=
[
  { suspect := Suspect.Charles, assertion := λ s => if s = Suspect.Alfred then PianoTuner else
                                                     if s = Suspect.Bertrand then Decorator else
                                                     if s = Suspect.Charles then InsuranceAgent else "" }
]

-- Define properties of roles
def always_lies (s : List Statement) : Prop := ∀ stmt ∈ s, not stmt.assertion stmt.suspect = Painter
def sometimes_lies (s : List Statement) : Prop := ∃ stmt ∈ s, ¬stmt.assertion stmt.suspect = Painter ∧ ∃ stmt' ∈ s, stmt'.assertion stmt'.suspect = Painter
def never_lies (s : List Statement) : Prop := ∀ stmt ∈ s, stmt.assertion stmt.suspect = Painter

open Suspect

-- Theorem to prove the profession of the accomplice (Alfred)
theorem find_accomplice_profession : ∃ r : Role, r = Role.accomplice → (Alfred_statements.any (λ stmt => stmt.assertion Suspect.Alfred) = Doctor) :=
by
  sorry

end find_accomplice_profession_l752_752156


namespace total_potatoes_l752_752141

open Nat

theorem total_potatoes (P T R : ℕ) (h1 : P = 5) (h2 : T = 6) (h3 : R = 48) : P + (R / T) = 13 := by
  sorry

end total_potatoes_l752_752141


namespace solution_set_for_f_l752_752244

noncomputable def f : ℝ → ℝ := sorry -- f(x) defined somewhere

theorem solution_set_for_f (f_increasing : ∀ x y : ℝ, x < y → f x < f y)
  (f_at_0 : f 0 = -1)
  (f_at_3 : f 3 = 1) :
  {x : ℝ | |f x| < 1} = set.Ioo 0 3 :=
by 
  sorry

end solution_set_for_f_l752_752244


namespace domain_of_sqrt_2cosx_plus_1_l752_752206

noncomputable def domain_sqrt_2cosx_plus_1 (x : ℝ) : Prop :=
  ∃ (k : ℤ), (2 * k * Real.pi - 2 * Real.pi / 3) ≤ x ∧ x ≤ (2 * k * Real.pi + 2 * Real.pi / 3)

theorem domain_of_sqrt_2cosx_plus_1 :
  (∀ (x: ℝ), 0 ≤ 2 * Real.cos x + 1 ↔ domain_sqrt_2cosx_plus_1 x) :=
by
  sorry

end domain_of_sqrt_2cosx_plus_1_l752_752206


namespace third_chapter_is_24_pages_l752_752827

-- Define the total number of pages in the book
def total_pages : ℕ := 125

-- Define the number of pages in the first chapter
def first_chapter_pages : ℕ := 66

-- Define the number of pages in the second chapter
def second_chapter_pages : ℕ := 35

-- Define the number of pages in the third chapter
def third_chapter_pages : ℕ := total_pages - (first_chapter_pages + second_chapter_pages)

-- Prove that the number of pages in the third chapter is 24
theorem third_chapter_is_24_pages : third_chapter_pages = 24 := by
  sorry

end third_chapter_is_24_pages_l752_752827


namespace find_price_of_turban_l752_752123

-- Define the main variables and conditions
def price_of_turban (T : ℝ) : Prop :=
  ((3 / 4) * 90 + T = 60 + T) → T = 30

-- State the theorem with the given conditions and aim to find T
theorem find_price_of_turban (T : ℝ) (h1 : 90 + T = 120) :  price_of_turban T :=
by
  intros
  sorry


end find_price_of_turban_l752_752123


namespace part_1_part_2_l752_752590

noncomputable def f (x ω : ℝ) : ℝ :=
  4 * sin (ω * x / 2) * cos (ω * x / 2) - 4 * sqrt 3 * cos (ω * x / 2) ^ 2 + 2 * sqrt 3

theorem part_1 {ω : ℝ} (h : ω > 0) (h_dist : ∀ x y, f x ω = f y ω → |y - x| = π / 2) :
  ∀ x, f x ω = 4 * sin (2 * x - π / 3) := sorry

theorem part_2 {ω : ℝ} (h : ω > 0) (hf : symmetric (f x ω) (π / 3, 0))
  (h_monotone : monotone_on (f x ω) (Icc 0 (π / 4))) :
  ω = 1 := sorry

end part_1_part_2_l752_752590


namespace solution_l752_752333

noncomputable def problem :=
  let A : (ℝ × ℝ × ℝ) := (a, 0, 0) in
  let B : (ℝ × ℝ × ℝ) := (0, b, 0) in
  let C : (ℝ × ℝ × ℝ) := (0, 0, c) in
  let plane_eq (x y z a b c : ℝ) : Prop := (x / a + y / b + z / c = 1) in
  let distance_eq (a b c : ℝ) : Prop :=
    (1 / (Real.sqrt ((1 / (a^2)) + 1 / (b^2) + 1 / (c^2)))) = 2 in
  let centroid (a b c : ℝ) : ℝ × ℝ × ℝ := (a / 3, b / 3, c / 3) in
  let (p, q, r) := centroid a b c in
  (plane_eq a 0 0 a b c) ∧
  (plane_eq 0 b 0 a b c) ∧
  (plane_eq 0 0 c a b c) ∧
  (distance_eq a b c) → (1 / (p^2) + 1 / (q^2) + 1 / (r^2)) = 2.25

theorem solution : problem :=
sorry

end solution_l752_752333


namespace probability_sum_of_digits_eq_10_l752_752795

theorem probability_sum_of_digits_eq_10 (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1): 
  let P := m / n
  let valid_numbers := 120
  let total_numbers := 2020
  (P = valid_numbers / total_numbers) → (m = 6) → (n = 101) → (m + n = 107) :=
by 
  sorry

end probability_sum_of_digits_eq_10_l752_752795


namespace angle_problems_l752_752203

noncomputable def smallest_positive_angle (θ : ℤ) : ℤ :=
  let base := (θ % 360 + 360) % 360
  in if base = 0 then 360 else base

noncomputable def largest_negative_angle (θ : ℤ) : ℤ :=
  let base := (θ % 360 + 360) % 360
  in if base = 0 then -360 else base - 360

noncomputable def angles_in_range (θ lower upper : ℤ) : List ℤ :=
  let base := (θ % 360 + 360) % 360
  List.filter (fun x => lower ≤ x ∧ x ≤ upper) [base + k * 360 | k <- [-2, -1, 0, 1]]

theorem angle_problems (θ : ℤ) :
  θ = -2010 →
  smallest_positive_angle θ = 150 ∧
  largest_negative_angle θ = -210 ∧
  angles_in_range θ -720 720 = [-570, -210, 150, 510] := by
  intros hθ
  rw [hθ]
  sorry

end angle_problems_l752_752203


namespace exists_pos_int_such_sqrt_not_int_l752_752390

theorem exists_pos_int_such_sqrt_not_int (a b c : ℤ) : ∃ n : ℕ, 0 < n ∧ ¬∃ k : ℤ, k * k = n^3 + a * n^2 + b * n + c :=
by
  sorry

end exists_pos_int_such_sqrt_not_int_l752_752390


namespace merchant_marked_price_l752_752498

variable (L C M S : ℝ)

theorem merchant_marked_price :
  (C = 0.8 * L) → (C = 0.8 * S) → (S = 0.8 * M) → (M = 1.25 * L) :=
by
  sorry

end merchant_marked_price_l752_752498


namespace c1_c2_not_collinear_l752_752522

-- Definitions of vectors a, b, c1, c2
def a : ℝ × ℝ × ℝ := (1, -2, 3)
def b : ℝ × ℝ × ℝ := (3, 0, -1)
def c1 : ℝ × ℝ × ℝ := (2 * a.1 + 4 * b.1, 2 * a.2 + 4 * b.2, 2 * a.3 + 4 * b.3)
def c2 : ℝ × ℝ × ℝ := (3 * b.1 - a.1, 3 * b.2 - a.2, 3 * b.3 - a.3)

-- Proving that c1 and c2 are not collinear
theorem c1_c2_not_collinear : ¬ ∃ (γ : ℝ), c1 = (γ * c2.1, γ * c2.2, γ * c2.3) := 
by
  sorry

end c1_c2_not_collinear_l752_752522


namespace probability_of_x_gt_3y_is_correct_l752_752735

noncomputable def probability_x_gt_3y : ℚ :=
  let rectangle_width := 2016
  let rectangle_height := 2017
  let triangle_height := 672 -- 2016 / 3
  let triangle_area := 1 / 2 * rectangle_width * triangle_height
  let rectangle_area := rectangle_width * rectangle_height
  triangle_area / rectangle_area

theorem probability_of_x_gt_3y_is_correct :
  probability_x_gt_3y = 336 / 2017 :=
by
  -- Proof will be filled in later
  sorry

end probability_of_x_gt_3y_is_correct_l752_752735


namespace number_of_such_integers_l752_752349

open Nat

def satisfies_conditions (n : ℕ) : Prop := 
  (n > 1) ∧ (∃ (d : ℕ → ℕ), (∀ i, d i = d (i + 8)) ∧ (let k := d 1 * 1000 + d 2 * 100 + d 3 * 10 + d 4 in 
  (k % 10 = d 4) ∧ (div (k % 100) 10 = d 3) ∧ (div (k % 1000) 100 = d 2)))

theorem number_of_such_integers : (n : ℕ) (satisfies_conditions n) : (8 : ℕ) :=
sorry

end number_of_such_integers_l752_752349


namespace six_by_six_tile_eight_by_eight_tile_l752_752823

theorem six_by_six_tile:
  ∃ (A B: set (fin 6 × fin 6)),
  (A ∪ B = set.univ) ∧ (A ∩ B = ∅) ∧
  (∀ (d : fin 6 × fin 6), d ∈ A ∨ d ∈ B) ∧
  (∀ (u v: fin 2), ¬((u, v) ∈ A ∧ (u, v.succ) ∈ B ∨ (u, v) ∈ B ∧ (u, v.succ) ∈ A)) :=
sorry

theorem eight_by_eight_tile:
  ∃ (configuration: set (fin 8 × fin 8)),
  (∀ (x y : fin 8), (x, y) ∈ configuration) ∧ 
  ¬(∃ (A B: set (fin 8 × fin 8)),
  (A ∪ B = configuration) ∧
  (A ∩ B = ∅) ∧
  (∀ (d : fin 8 × fin 8), d ∈ A ∨ d ∈ B) ∧
  (∀ (u v: fin 2), ¬ ((u, v) ∈ A ∧ (u, v.succ) ∈ B ∨ (u, v) ∈ B ∧ (u, v.succ) ∈ A))) :=
sorry

end six_by_six_tile_eight_by_eight_tile_l752_752823


namespace no_integer_solution_for_unique_root_l752_752200

theorem no_integer_solution_for_unique_root :
  ∀ m : ℤ, ¬(∃ x : ℝ, 36 * x^2 - (m : ℝ) * x - 4 = 0 ∧ discriminant_eq_zero m) :=
begin
  sorry
end

def discriminant_eq_zero (m : ℤ) : Prop :=
  let discriminant := ((m : ℝ)^2) + 576 in
  discriminant = 0

end no_integer_solution_for_unique_root_l752_752200


namespace trigonometric_identity_proof_l752_752538

noncomputable def theta : ℝ := real.pi * 40 / 180

theorem trigonometric_identity_proof :
  (tan theta)^2 - (cos theta)^2 / ((tan theta)^2 * (cos theta)^2) = 2 * (sin theta)^2 - (sin theta)^4 :=
by
  sorry

end trigonometric_identity_proof_l752_752538


namespace trapezoidal_region_area_of_regular_hexagon_l752_752429

-- Define a regular hexagon with a given side length
def regular_hexagon (s : ℝ) : Prop := ∀ (i j : ℕ), (i ≠ j) → dist (point i) (point j) = if abs (i - j) = 1 ∨ abs (i - j) = 5 then s else if abs (i - j) = 3 then s * √3 else 2 * s * √3 / 2

-- Define the midpoints of alternate sides
def midpoint (i j : ℕ) : point := (point i + point j) / 2

-- Prove the area of the trapezoidal region formed
theorem trapezoidal_region_area_of_regular_hexagon :
  regular_hexagon 12 →
  area (trapezoidal_region (midpoint 0 2) (midpoint 2 4) (midpoint 4 0) (midpoint 1 3)) = 144 * √3 :=
  by sorry

end trapezoidal_region_area_of_regular_hexagon_l752_752429


namespace lcm_multiple_not_2008_l752_752536

theorem lcm_multiple_not_2008 (n m k l : ℕ) (h1 : 2^k ≤ m) (h2 : m < 2^(k + 1)) 
  (h3 : 3^l ≤ m) (h4 : m < 3^(l + 1)) :
  (nat.lcm (list.range n.succ)) ≠ 2008 * (nat.lcm (list.range m.succ)) :=
by
  sorry

end lcm_multiple_not_2008_l752_752536


namespace number_of_oxygen_atoms_l752_752573

theorem number_of_oxygen_atoms 
  (M_weight : ℝ)
  (H_weight : ℝ)
  (Cl_weight : ℝ)
  (O_weight : ℝ)
  (MW_formula : M_weight = H_weight + Cl_weight + n * O_weight)
  (M_weight_eq : M_weight = 68)
  (H_weight_eq : H_weight = 1)
  (Cl_weight_eq : Cl_weight = 35.5)
  (O_weight_eq : O_weight = 16)
  : n = 2 := 
  by sorry

end number_of_oxygen_atoms_l752_752573


namespace range_of_a_l752_752984

noncomputable def f (x a : ℝ) : ℝ :=
  x ^ 2 + a * Real.log x + 2 / x

theorem range_of_a (a : ℝ) (h : ∀ x ∈ Set.Ioo (1 : ℝ) 4, differentiable_in ℝ (λ x, x ^ 2 + a * Real.log x + 2 / x) x ∧ deriv (λ x, x ^ 2 + a * Real.log x + 2 / x) x ≤ 0) :
  a ≤ -63 / 2 :=
by
  sorry

end range_of_a_l752_752984


namespace parabola_directrix_l752_752207

theorem parabola_directrix (x y : ℝ) :
  (∃ a b c : ℝ, y = (a * x^2 + b * x + c) / 12 ∧ a = 1 ∧ b = -6 ∧ c = 5) →
  y = -10 / 3 :=
by
  sorry

end parabola_directrix_l752_752207


namespace bobby_candy_left_l752_752166

theorem bobby_candy_left (initial_candies := 21) (first_eaten := 5) (second_eaten := 9) : 
  initial_candies - first_eaten - second_eaten = 7 :=
by
  -- Proof goes here
  sorry

end bobby_candy_left_l752_752166


namespace beaver_hid_36_carrots_l752_752580

variable (x y : ℕ)

-- Conditions
def beaverCarrots := 4 * x
def bunnyCarrots := 6 * y

-- Given that both animals hid the same total number of carrots
def totalCarrotsEqual := beaverCarrots x = bunnyCarrots y

-- Bunny used 3 fewer burrows than the beaver
def bunnyBurrows := y = x - 3

-- The goal is to show the beaver hid 36 carrots
theorem beaver_hid_36_carrots (H1 : totalCarrotsEqual x y) (H2 : bunnyBurrows x y) : beaverCarrots x = 36 := by
  sorry

end beaver_hid_36_carrots_l752_752580


namespace largest_n_binom_l752_752454

theorem largest_n_binom (n : ℕ) (h1: finset.sum (finset.range 9) (λ k, if k = 3 then nat.choose 8 3 else if k = 4 then nat.choose 8 4 else 0) = nat.choose 9 n) : n = 5 :=
sorry

end largest_n_binom_l752_752454


namespace gcd_A_B_41_l752_752518

theorem gcd_A_B_41 :
  ∀ (A B : ℕ), (∃ (A B C : ℕ), A + B = 81 * 41 ∧ (∀ a b : ℕ, a ∈ finset.range 82 → b ∈ finset.range 82 → a ≠ b → A + B = 81 * 41 → true) ∧ (∀ a b : ℕ, a ∈ finset.range 82 → b ∈ finset.range 82 → gcd(A, 81 * 41) = gcd(A, B))) → gcd(A, B) = 41 :=
by
  sorry

end gcd_A_B_41_l752_752518


namespace fourth_root_cube_root_approx_l752_752887

theorem fourth_root_cube_root_approx : Real.toRat ((0.008)^(1/3)^(1/4)) = 0.67 :=
begin
  sorry
end

end fourth_root_cube_root_approx_l752_752887


namespace palic_function_differentiable_l752_752717

-- Define the real numbers and properties of a, b, c
variables (a b c : ℝ)
axiom sum_eq_one : a + b + c = 1
axiom sum_sq_eq_one : a^2 + b^2 + c^2 = 1
axiom sum_cube_neq_one : a^3 + b^3 + c^3 ≠ 1

-- Define the Palic function property
def is_palic_function (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, 
    f x + f y + f z = f (a * x + b * y + c * z) + f (b * x + c * y + a * z) + f (c * x + a * y + b * z)

-- Prove that any Palic function is infinitely differentiable and of a specific form
theorem palic_function_differentiable (f : ℝ → ℝ) (hf : is_palic_function a b c f) : 
  (∃ p q r : ℝ, ∀ x : ℝ, f x = p * x^2 + q * x + r) ∧ (∀ n : ℕ, differentiable ℝ^[n] f) :=
sorry

end palic_function_differentiable_l752_752717


namespace factorialExpressionMinDiff_l752_752416

theorem factorialExpressionMinDiff (a1 a2 am b1 b2 bn : ℕ) 
  (h1 : a1 ≥ a2 ∧ a2 ≥ am)
  (h2 : b1 ≥ b2 ∧ b2 ≥ bn)
  (h3: a1! * a2! * am! = 1638 * (b1! * b2! * bn!)) :
  |a1 - b1| = 1 :=
begin
  sorry
end

end factorialExpressionMinDiff_l752_752416


namespace eulers_formula_l752_752385

-- Definitions related to simply connected polyhedra
def SimplyConnectedPolyhedron (V E F : ℕ) : Prop := true  -- Genus 0 implies it is simply connected

-- Euler's characteristic property for simply connected polyhedra
theorem eulers_formula (V E F : ℕ) (h : SimplyConnectedPolyhedron V E F) : V - E + F = 2 := 
by
  sorry

end eulers_formula_l752_752385


namespace problem_l752_752001

-- Define the values in the grid
def grid : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ := (4, 3, 1, 1, 6, 2, 3)

-- Define the variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def condition_1 := (A = 3) ∧ (B = 2) ∧ (C = 4)
def condition_2 := (4 + A + 1 + B + C + 3 = 9)
def condition_3 := (A + 1 + 6 = 9)
def condition_4 := (1 + A + 6 = 9)
def condition_5 := (B + 2 + C + 5 = 9)

-- Define that the sum of the red cells is equal to any row
def sum_of_red_cells := (A + B + C = 9)

-- The final goal to prove
theorem problem : condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 ∧ sum_of_red_cells := 
by {
  refine ⟨_, _, _, _, _, _⟩;
  sorry   -- proofs for each condition
}

end problem_l752_752001


namespace proof_problem_l752_752752

noncomputable def g : ℕ → ℕ := sorry
noncomputable def g_inv : ℕ → ℕ := sorry

-- Given conditions
axiom h1 : g 4 = 7
axiom h2 : g 6 = 2
axiom h3 : g 3 = 8

theorem proof_problem : g_inv (g_inv 8 + g_inv 7) = 4 :=
by
  -- We could use the following commands if formal verification was required
  have h4 : g_inv 8 = 3, from sorry
  have h5 : g_inv 7 = 4, from sorry
  rw [h4, h5]
  -- The steps will include adding these known values to the proof environment
  sorry

end proof_problem_l752_752752


namespace principal_amount_borrowed_l752_752809

theorem principal_amount_borrowed (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (R_eq : R = 12) (T_eq : T = 3) (SI_eq : SI = 7200) :
  (SI = (P * R * T) / 100) → P = 20000 :=
by sorry

end principal_amount_borrowed_l752_752809


namespace ratio_of_b_plus_e_over_c_plus_f_l752_752658

theorem ratio_of_b_plus_e_over_c_plus_f 
  (a b c d e f : ℝ)
  (h1 : a + b = 2 * a + c)
  (h2 : a - 2 * b = 4 * c)
  (h3 : a + b + c = 21)
  (h4 : d + e = 3 * d + f)
  (h5 : d - 2 * e = 5 * f)
  (h6 : d + e + f = 32) :
  (b + e) / (c + f) = -3.99 :=
sorry

end ratio_of_b_plus_e_over_c_plus_f_l752_752658


namespace area_of_intersecting_lines_l752_752446

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))|

theorem area_of_intersecting_lines : 
  let A := (3, 3) in
  let B := (4.5, 7.5) in
  let C := (7.5, 4.5) in
  area_of_triangle A B C = 8.625 :=
by 
  let A := (3, 3) 
  let B := (4.5, 7.5)
  let C := (7.5, 4.5)
  sorry

end area_of_intersecting_lines_l752_752446


namespace simplify_polynomial_expression_l752_752396

variable {R : Type*} [CommRing R]

theorem simplify_polynomial_expression (x : R) :
  (2 * x^6 + 3 * x^5 + 4 * x^4 + x^3 + x^2 + x + 20) - (x^6 + 4 * x^5 + 2 * x^4 - x^3 + 2 * x^2 + 5) =
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - x^2 + 15 := 
by
  sorry

end simplify_polynomial_expression_l752_752396


namespace subset_condition_l752_752365

variable (a : ℝ)

def A : Set ℝ := {0, -a}
def B : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_condition : A a ⊆ B a ↔ a = 1 := by 
  sorry

end subset_condition_l752_752365


namespace geometric_series_sum_l752_752905

theorem geometric_series_sum :
  ∃ S, let a := 1 in let r := 2 in let n := 12 in
  S = (a * (r^n - 1)) / (r-1) ∧ S = 4095 :=
by
  sorry

end geometric_series_sum_l752_752905


namespace area_of_smallest_region_l752_752569

def ellipse (x y : ℝ) : Prop := (x^2 / 4 + y^2 = 9)
def abs_line (x y : ℝ) : Prop := (y = |x|)

theorem area_of_smallest_region :
  ∃ (area : ℝ), area = 3 * real.pi ∧ 
    ∀(x y : ℝ), abs_line x y → ellipse x y →
      let A := { p : ℝ × ℝ | abs_line p.1 p.2 ∧ ellipse p.1 p.2 } in
      let region := set.prod {-6*|5|/5 ≤ x ≤ 6*|5|/5} {-6*|5|/5 ≤ y ≤ 6*|5|/5} in
      region ⊆ A := 
sorry

end area_of_smallest_region_l752_752569


namespace find_m_l752_752683

-- Given definitions and conditions
def is_ellipse (x y m : ℝ) := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (m : ℝ) := Real.sqrt ((m - 4) / m) = 1 / 2

-- Prove that m = 16 / 3 given the conditions
theorem find_m (m : ℝ) (cond1 : is_ellipse 1 1 m) (cond2 : eccentricity m) (cond3 : m > 4) : m = 16 / 3 :=
by
  sorry

end find_m_l752_752683


namespace solution_l752_752335

noncomputable def problem :=
  let A : (ℝ × ℝ × ℝ) := (a, 0, 0) in
  let B : (ℝ × ℝ × ℝ) := (0, b, 0) in
  let C : (ℝ × ℝ × ℝ) := (0, 0, c) in
  let plane_eq (x y z a b c : ℝ) : Prop := (x / a + y / b + z / c = 1) in
  let distance_eq (a b c : ℝ) : Prop :=
    (1 / (Real.sqrt ((1 / (a^2)) + 1 / (b^2) + 1 / (c^2)))) = 2 in
  let centroid (a b c : ℝ) : ℝ × ℝ × ℝ := (a / 3, b / 3, c / 3) in
  let (p, q, r) := centroid a b c in
  (plane_eq a 0 0 a b c) ∧
  (plane_eq 0 b 0 a b c) ∧
  (plane_eq 0 0 c a b c) ∧
  (distance_eq a b c) → (1 / (p^2) + 1 / (q^2) + 1 / (r^2)) = 2.25

theorem solution : problem :=
sorry

end solution_l752_752335


namespace boys_meet_once_excluding_start_finish_l752_752444

theorem boys_meet_once_excluding_start_finish 
    (d : ℕ) 
    (h1 : 0 < d) 
    (boy1_speed : ℕ) (boy2_speed : ℕ) 
    (h2 : boy1_speed = 6) (h3 : boy2_speed = 10)
    (relative_speed : ℕ) (h4 : relative_speed = boy1_speed + boy2_speed) 
    (time_to_meet_A_again : ℕ) (h5 : time_to_meet_A_again = d / relative_speed) 
    (boy1_laps_per_sec boy2_laps_per_sec : ℕ) 
    (h6 : boy1_laps_per_sec = boy1_speed / d) 
    (h7 : boy2_laps_per_sec = boy2_speed / d)
    (lcm_laps : ℕ) (h8 : lcm_laps = Nat.lcm 6 10)
    (meetings_per_lap : ℕ) (h9 : meetings_per_lap = lcm_laps / d)
    (total_meetings : ℕ) (h10 : total_meetings = meetings_per_lap * time_to_meet_A_again)
  : total_meetings = 1 := by
  sorry

end boys_meet_once_excluding_start_finish_l752_752444


namespace complex_conjugate_of_z_l752_752972

theorem complex_conjugate_of_z
  (z : ℂ)
  (h : z * (1 + complex.I) = 2) :
  complex.conj z = 1 + complex.I :=
sorry

end complex_conjugate_of_z_l752_752972


namespace sqrt_expression_l752_752169

theorem sqrt_expression : 
  (Real.sqrt 75 + Real.sqrt 8 - Real.sqrt 18 - Real.sqrt 6 * Real.sqrt 2 = 3 * Real.sqrt 3 - Real.sqrt 2) := 
by 
  sorry

end sqrt_expression_l752_752169


namespace christina_speed_l752_752319

theorem christina_speed
  (d v_j v_l t : ℝ)
  (D_l : ℝ)
  (h_d : d = 360)
  (h_v_j : v_j = 5)
  (h_v_l : v_l = 12)
  (h_D_l : D_l = 360)
  (h_t : t = D_l / v_l)
  (h_distance : d = v_j * t + c * t) :
  c = 7 :=
by
  sorry

end christina_speed_l752_752319


namespace isosceles_right_triangle_area_l752_752049

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = 6 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 18 := by 
  sorry

end isosceles_right_triangle_area_l752_752049


namespace diff_of_squares_not_2018_l752_752171

theorem diff_of_squares_not_2018 (a b : ℕ) (h : a > b) : ¬(a^2 - b^2 = 2018) :=
by {
  -- proof goes here
  sorry
}

end diff_of_squares_not_2018_l752_752171


namespace length_of_chord_l752_752607

theorem length_of_chord (r AB : ℝ) (h1 : r = 6) (h2 : 0 < AB) (h3 : AB <= 2 * r) : AB ≠ 14 :=
by
  sorry

end length_of_chord_l752_752607


namespace find_digits_l752_752005

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 7
noncomputable def C : ℕ := 1

def row_sums_equal (A B C : ℕ) : Prop :=
  (4 + 2 + 3 = 9) ∧
  (A + 6 + 1 = 9) ∧
  (1 + 2 + 6 = 9) ∧
  (B + 2 + C = 9)

def column_sums_equal (A B C : ℕ) : Prop :=
  (4 + 1 + B = 12) ∧
  (2 + A + 2 = 6 + 3 + 1) ∧
  (3 + 1 + 6 + C = 12)

def red_cells_sum_equals_row_sum (A B C : ℕ) : Prop :=
  (A + B + C = 9)

theorem find_digits :
  ∃ (A B C : ℕ),
    row_sums_equal A B C ∧
    column_sums_equal A B C ∧
    red_cells_sum_equals_row_sum A B C ∧
    100 * A + 10 * B + C = 371 :=
by
  exact ⟨3, 7, 1, ⟨rfl, rfl, rfl, rfl⟩⟩

end find_digits_l752_752005


namespace complement_A_l752_752375

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := { x : ℝ | x < 2 }

theorem complement_A :
  (U \ A) = { x : ℝ | x >= 2 } :=
by
  sorry

end complement_A_l752_752375


namespace sin_sum_inequality_l752_752388

theorem sin_sum_inequality (n : ℕ) : 
  ((finset.range (3 * n + 1)).sum (λ k => |Real.sin (k + 1)|)) > (8 / 5) * n :=
sorry

end sin_sum_inequality_l752_752388


namespace globe_division_l752_752845

theorem globe_division (parallels meridians : ℕ)
  (h_parallels : parallels = 17)
  (h_meridians : meridians = 24) :
  let slices_per_sector := parallels + 1
  let sectors := meridians
  let total_parts := slices_per_sector * sectors
  total_parts = 432 := by
  sorry

end globe_division_l752_752845


namespace wire_length_l752_752499

theorem wire_length (r_sphere r_cylinder : ℝ) (V_sphere_eq_V_cylinder : (4/3) * π * r_sphere^3 = π * r_cylinder^2 * 144) :
  r_sphere = 12 → r_cylinder = 4 → 144 = 144 := sorry

end wire_length_l752_752499


namespace virus_diameter_scientific_notation_l752_752016

theorem virus_diameter_scientific_notation:
  (0.00045 : ℝ) = 4.5 * (10:ℝ)^(-4) :=
sorry

end virus_diameter_scientific_notation_l752_752016


namespace find_digits_l752_752006

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 7
noncomputable def C : ℕ := 1

def row_sums_equal (A B C : ℕ) : Prop :=
  (4 + 2 + 3 = 9) ∧
  (A + 6 + 1 = 9) ∧
  (1 + 2 + 6 = 9) ∧
  (B + 2 + C = 9)

def column_sums_equal (A B C : ℕ) : Prop :=
  (4 + 1 + B = 12) ∧
  (2 + A + 2 = 6 + 3 + 1) ∧
  (3 + 1 + 6 + C = 12)

def red_cells_sum_equals_row_sum (A B C : ℕ) : Prop :=
  (A + B + C = 9)

theorem find_digits :
  ∃ (A B C : ℕ),
    row_sums_equal A B C ∧
    column_sums_equal A B C ∧
    red_cells_sum_equals_row_sum A B C ∧
    100 * A + 10 * B + C = 371 :=
by
  exact ⟨3, 7, 1, ⟨rfl, rfl, rfl, rfl⟩⟩

end find_digits_l752_752006


namespace problem_D_l752_752623

variable {a b c : ℝ}
variable (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : b ≠ 1) (h₄ : c ≠ 1)

def op ⊕ (a b : ℝ) : ℝ := a ^ (Real.log a / Real.log b)

theorem problem_D :
  (op ⊕ a b) ^ c = op ⊕ (a ^ c) b :=
by
  sorry

end problem_D_l752_752623


namespace fraction_students_looking_up_is_three_fourths_l752_752435

noncomputable def students := 200 : ℕ
noncomputable def eyes_seeing_plane := 300 : ℕ
noncomputable def students_looking_up : ℕ := eyes_seeing_plane / 2
noncomputable def fraction_students_looking_up : ℚ := students_looking_up / students

theorem fraction_students_looking_up_is_three_fourths :
  fraction_students_looking_up = 3 / 4 :=
by
  sorry

end fraction_students_looking_up_is_three_fourths_l752_752435


namespace point_in_fourth_quadrant_l752_752419

theorem point_in_fourth_quadrant (A : ℝ) (B : ℝ) :
  A = tan 35 ∧ B = cos 215 ∧ A > 0 ∧ B < 0 → 
  (∃ x y : ℝ, x = A ∧ y = B ∧ x > 0 ∧ y < 0) :=
by
  sorry

end point_in_fourth_quadrant_l752_752419


namespace pasha_wins_l752_752293

-- Definitions for the game's conditions

def vertex (n : ℕ) := n ∈ {1, 2, 3, 4, 5, 6}

def is_diametrically_opposite (a b : ℕ) : Prop :=
  (a, b) = (1, 4) ∨ (a, b) = (2, 5) ∨ (a, b) = (3, 6) ∨ 
  (a, b) = (4, 1) ∨ (a, b) = (5, 2) ∨ (a, b) = (6, 3)

variables {u v w x y z : ℕ}

noncomputable def edge_product (a b : ℕ) : ℕ :=
  a * b

-- Game setup
variable (a b c : ℕ)

-- Pasha's strategy implies that product of vertices on opposite edges are equal and repeated twice.
def sum_of_edge_products (a b c : ℕ) : ℕ :=
  2 * (a * a + b * b + c * c)

-- Proof that sum is even ensuring Pasha wins when he follows the strategy.
theorem pasha_wins : ∀ (a b c : ℕ), Even (sum_of_edge_products a b c) :=
by
  intro a b c
  show Even (2 * (a * a + b * b + c * c))
  exact even_of_mul_left _ _ (by norm_num)

end pasha_wins_l752_752293


namespace ratio_AE_AO_l752_752153

noncomputable theory

open Real

variables {s : ℝ} (A B C D O : EuclideanGeometry.Point ℝ) 

/-- Given a square ABCD with side length s inscribed in a circle centered at O. 
    There is a smaller circle with radius s/2 tangent to the larger circle at B.
    If the line AO intersects the smaller circle at E, 
    then the ratio AE/AO equals (2 - sqrt(2)) / 2. -/
theorem ratio_AE_AO (H1 : EuclideanGeometry.is_square A B C D s)
  (H2 : EuclideanGeometry.is_inscribed A B C D O) 
  (H3 : tangent_at_point (EuclideanGeometry.Circle O (s * sqrt 2 / 2)) 
                         (EuclideanGeometry.Circle B (s / 2)) B)
  (H4 : ∃ E, EuclideanGeometry.intersects_line_segment A O E 
            ∧ EuclideanGeometry.on_circle E (EuclideanGeometry.Circle B (s / 2)))
  : ∃ E, EuclideanGeometry.ratio AE AO = (2 - sqrt 2) / 2 := 
sorry

end ratio_AE_AO_l752_752153


namespace proof_problem_l752_752947

def geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def arithmetic_sequence (a2 a3_1 a4 : ℕ) : Prop :=
  2 * a3_1 = a2 + a4

def a_n (n : ℕ) : ℕ := 2^(n-1)

def b_n (n : ℕ) : ℕ := a_n n + n

theorem proof_problem :
  (∃ a : ℕ → ℕ, geometric_sequence a 2 ∧ arithmetic_sequence (a 2) ((a 3) + 1) (a 4)) →
  a_n 1 = 1 ∧ a_n = λ n, 2^(n-1) ∧ (∑ i in finset.range 5, b_n (i + 1)) = 46 :=
by
  intro h
  sorry

end proof_problem_l752_752947


namespace train_cross_time_l752_752810

noncomputable def time_to_cross_pole (length: ℝ) (speed_kmh: ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  length / speed_ms

theorem train_cross_time :
  let length := 100
  let speed := 126
  abs (time_to_cross_pole length speed - 2.8571) < 0.0001 :=
by
  let length := 100
  let speed := 126
  have h1 : abs (time_to_cross_pole length speed - 2.8571) < 0.0001
  sorry
  exact h1

end train_cross_time_l752_752810


namespace problem_solution_l752_752583

open Int

def ceiling_sum_not_divisible_by_five_count : ℕ := 
  List.length (List.filter (λ n, 
    ((Int.ceil (1001 / n.toRat) + 
      Int.ceil (1002 / n.toRat) + 
      Int.ceil (1003 / n.toRat)) % 5 ≠ 0)) 
  (List.range' 1 1000))

theorem problem_solution :
  ceiling_sum_not_divisible_by_five_count = 23 := 
sorry

end problem_solution_l752_752583


namespace total_distance_eq_l752_752694

def distance_traveled_by_bus : ℝ := 2.6
def distance_traveled_by_subway : ℝ := 5.98
def total_distance_traveled : ℝ := distance_traveled_by_bus + distance_traveled_by_subway

theorem total_distance_eq : total_distance_traveled = 8.58 := by
  sorry

end total_distance_eq_l752_752694


namespace greatest_x_is_8_l752_752089

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def polynomial (x : ℤ) : ℤ := 4 * x^2 - 39 * x + 21

noncomputable def greatest_integer_x : ℤ :=
  let candidates := {x : ℤ | polynomial x > 0 ∧ is_prime (Int.natAbs (polynomial x))};
  Finset.max' (candidates.to_finset) (by sorry)

theorem greatest_x_is_8 : greatest_integer_x = 8 := sorry

end greatest_x_is_8_l752_752089


namespace certain_event_is_A_l752_752119

def isCertainEvent (event : Prop) : Prop := event

axiom event_A : Prop
axiom event_B : Prop
axiom event_C : Prop
axiom event_D : Prop

axiom event_A_is_certain : isCertainEvent event_A
axiom event_B_is_not_certain : ¬ isCertainEvent event_B
axiom event_C_is_impossible : ¬ event_C
axiom event_D_is_not_certain : ¬ isCertainEvent event_D

theorem certain_event_is_A : isCertainEvent event_A := by
  exact event_A_is_certain

end certain_event_is_A_l752_752119


namespace card_game_digit_d_l752_752676

theorem card_game_digit_d :
  let n := 60
  let k := 13
  ∃ D : ℕ, (nat.choose n k = 7446680 * 10^7 + 480) ∧ D = 4 :=
begin
  let n := 60,
  let k := 13,
  use 4,
  sorry
end

end card_game_digit_d_l752_752676


namespace range_of_m_l752_752841

variable {R : Type} [LinearOrderedField R]

noncomputable def f (x : R) : R := sorry

theorem range_of_m (m : R) : 
  (∀ x : R, f (-x) = -f x) ∧ 
  (∀ x : R, f (3 / 4 + x) = f (3 / 4 - x)) ∧ 
  f 4 > -2 ∧ 
  f 2 = m - 3 / m → 
  m < -1 ∨ (0 < m ∧ m < 3) :=
by
  intros h
  cases h with h_odd h1
  cases h1 with h_sym h2
  cases h2 with h4 h2_eq
  sorry

end range_of_m_l752_752841


namespace greatest_possible_sum_example_sum_case_l752_752786

/-- For integers x and y such that x^2 + y^2 = 50, the greatest possible value of x + y is 10. -/
theorem greatest_possible_sum (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 10 :=
sorry

-- Auxiliary theorem to state that 10 can be achieved
theorem example_sum_case : ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ x + y = 10 :=
sorry

end greatest_possible_sum_example_sum_case_l752_752786


namespace multiple_choice_test_probability_l752_752742

noncomputable def probability_at_least_half_correct (n : ℕ) (prob_correct : ℚ) : ℚ :=
  ∑ k in finset.range (n + 1), if k ≥ n / 2 then (nat.choose n k : ℚ) * prob_correct^k * (1 - prob_correct)^(n - k) else 0

theorem multiple_choice_test_probability :
  probability_at_least_half_correct 16 (1 / 4) = 1 / 32 :=
sorry

end multiple_choice_test_probability_l752_752742


namespace parameter_conditions_l752_752550

theorem parameter_conditions (p x y : ℝ) :
  (x - p)^2 = 16 * (y - 3 + p) →
  y^2 + ((x - 3) / (|x| - 3))^2 = 1 →
  |x| ≠ 3 →
  p > 3 ∧ 
  ((p ≤ 4 ∨ p ≥ 12) ∧ (p < 19 ∨ 19 < p)) :=
sorry

end parameter_conditions_l752_752550


namespace length_2a_sub_b_eq_2_find_k_for_orthogonality_l752_752970

variables (a b : ℝ) (k : ℝ)
variables (angle_ab : ℝ) (norm_a : ℝ) (norm_b : ℝ)

-- Given conditions
variable (h_angle : angle_ab = π / 3)
variable (h_norm_a : norm_a = 1)
variable (h_norm_b : norm_b = 2)

-- Define vectors
variables (vec_a vec_b : EuclideanSpace ℝ (Fin 3))
variable (h_norm_vec_a : ‖vec_a‖ = norm_a)
variable (h_norm_vec_b : ‖vec_b‖ = norm_b)
variable (h_angle_vec_a_b : real.angle_between vec_a vec_b = angle_ab)

-- Question 1: Show that |2a - b| = 2
theorem length_2a_sub_b_eq_2 :
  ‖2 • vec_a - vec_b‖ = 2 := by
  sorry

-- Question 2: Find that k such that (a + b) and (a + k • b) are orthogonal
theorem find_k_for_orthogonality (h_ortho : inner (vec_a + vec_b) (vec_a + k • vec_b) = 0) :
  k = -2 / 5 := by
  sorry

end length_2a_sub_b_eq_2_find_k_for_orthogonality_l752_752970


namespace increase_by_1_or_prime_l752_752838

theorem increase_by_1_or_prime (a : ℕ → ℕ) :
  a 0 = 6 →
  (∀ n, a (n + 1) = a n + Nat.gcd (a n) (n + 1)) →
  ∀ n, n < 1000000 → (∃ p, p = 1 ∨ Nat.Prime p ∧ a (n + 1) = a n + p) :=
by
  intro ha0 ha_step
  -- Proof omitted
  sorry

end increase_by_1_or_prime_l752_752838


namespace solution_l752_752347

noncomputable def problem_statement : Prop :=
  let O := (0, 0, 0)
  let A := (α : ℝ, 0, 0)
  let B := (0, β : ℝ, 0)
  let C := (0, 0, γ : ℝ)
  let plane_equation := λ x y z, (x / α + y / β + z / γ = 1)
  let distance_from_origin := 2
  let centroid := ((α / 3), (β / 3), (γ / 3))
  let p := centroid.1
  let q := centroid.2
  let r := centroid.3
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 2.25

theorem solution : problem_statement := 
by sorry

end solution_l752_752347


namespace simple_interest_years_l752_752288

theorem simple_interest_years (P : ℝ) (hP : P > 0) (R : ℝ := 2.5) (SI : ℝ := P / 5) : 
  ∃ T : ℝ, P * R * T / 100 = SI ∧ T = 8 :=
by
  sorry

end simple_interest_years_l752_752288


namespace square_perimeter_l752_752759

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by 
  have h1 : s = 1 := 
  begin
    -- Considering non-zero s, divide both sides by s
    by_contradiction hs,
    have hs' : s ≠ 0 := λ h0, hs (by simp [h0] at h),
    exact hs (by simpa [h, hs'] using h),
  end,
  rw h1,
  norm_num

end square_perimeter_l752_752759


namespace max_chord_length_l752_752257

noncomputable def length_of_chord (θ : ℝ) : ℝ :=
  let t := (8 * Real.sin θ + Real.cos θ + 1) / (2 * Real.sin θ - Real.cos θ + 3)
  in  Real.abs (t * Real.sqrt 5)

theorem max_chord_length : ∃ θ : ℝ, length_of_chord θ = 8 * Real.sqrt 5 :=
sorry

end max_chord_length_l752_752257


namespace concyclic_ECMD_l752_752967

noncomputable def circle {α : Type*} [MetricSpace α] (O : α) (r : ℝ) := 
  { P : α // dist P O = r }

variables {α : Type*} [MetricSpace α] {O A B C D E F M : α}
variables (circle : circle O) [diameter : dist A B = 2 * dist A O]
variables (on_circle : circle O C) (on_circle' : circle O D)
variables (tangent_C : ∀ P, P ∈ tangent_line circle C → dist P O = dist C O)
variables (tangent_D : ∀ P, P ∈ tangent_line circle D → dist P O = dist D O)
variables (E : intersection (tangent_line circle C) (tangent_line circle D))
variables (F : intersection (line_through A D) (line_through B C))
variables (M : intersection (line_through E F) (line_through A B))

theorem concyclic_ECMD : concyclic {E, C, M, D} :=
  sorry

end concyclic_ECMD_l752_752967


namespace solution_range_l752_752883

-- Given conditions from the table
variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom h₁ : f a b c 1.1 = -0.59
axiom h₂ : f a b c 1.2 = 0.84
axiom h₃ : f a b c 1.3 = 2.29
axiom h₄ : f a b c 1.4 = 3.76

theorem solution_range (a b c : ℝ) : 
  ∃ x : ℝ, 1.3 < x ∧ x < 1.4 ∧ f a b c x = 3 :=
sorry

end solution_range_l752_752883


namespace arithmetic_sequence_inequality_l752_752216

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality
  (a : ℕ → α) (d : α)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_d_pos : d > 0)
  (n : ℕ)
  (h_n_gt_1 : n > 1) :
  a 1 * a (n + 1) < a 2 * a n := 
sorry

end arithmetic_sequence_inequality_l752_752216


namespace functions_are_equal_l752_752517

def f (x : ℝ) : ℝ := abs (x + 1)
def g (x : ℝ) : ℝ := if x >= -1 then x + 1 else -x - 1

theorem functions_are_equal : ∀ x : ℝ, f x = g x :=
by
  sorry

end functions_are_equal_l752_752517


namespace sharpen_knives_cost_l752_752381

theorem sharpen_knives_cost :
  let first_knife_cost := 5.00
  let next_three_knives_cost := 4.00
  let remaining_knives_cost := 3.00
  let total_knives := 9
  total_knives = 9 → 
  ∑ k in {1, 2, 3, 4, 5, 6, 7, 8, 9}, if k = 1 then first_knife_cost else if k > 1 ∧ k ≤ 4 then next_three_knives_cost else remaining_knives_cost = 32.00 :=
by
  sorry

end sharpen_knives_cost_l752_752381


namespace area_of_yard_proof_l752_752219

def area_of_yard (L W : ℕ) : ℕ :=
  L * W

theorem area_of_yard_proof (L W : ℕ) (hL : L = 40) (hFence : 2 * W + L = 52) : 
  area_of_yard L W = 240 := 
by 
  sorry

end area_of_yard_proof_l752_752219


namespace find_x_l752_752576

theorem find_x (x : ℚ) (h : (real.sqrt (7 * x) / real.sqrt (2 * (x - 2))) = 3) : x = 36 / 11 :=
by
  sorry

end find_x_l752_752576


namespace probability_of_meeting_l752_752079

noncomputable def meeting_probability : ℝ :=
  let total_area := 60 * 60 in
  let valid_area := total_area - (40 * 40) in
  valid_area / total_area

theorem probability_of_meeting : meeting_probability = (5 / 9) := by
  sorry

end probability_of_meeting_l752_752079


namespace arithmetic_progression_ratio_l752_752178

variable {α : Type*} [LinearOrder α] [Field α]

theorem arithmetic_progression_ratio (a d : α) (h : 15 * a + 105 * d = 3 * (8 * a + 28 * d)) : a / d = 7 / 3 := 
by sorry

end arithmetic_progression_ratio_l752_752178


namespace floor_expression_eq_eight_l752_752542

theorem floor_expression_eq_eight :
  let n := 2022 in
  (Int.floor ((↑(n + 1) ^ 3 / ((↑n - 1) * ↑n : ℝ)) - (↑(n - 1) ^ 3 / (↑n * (↑n + 1) : ℝ)))) = 8 :=
by
  sorry

end floor_expression_eq_eight_l752_752542


namespace average_of_numbers_l752_752027

theorem average_of_numbers (a b c d e : ℕ) (h₁ : a = 8) (h₂ : b = 9) (h₃ : c = 10) (h₄ : d = 11) (h₅ : e = 12) :
  (a + b + c + d + e) / 5 = 10 :=
by
  sorry

end average_of_numbers_l752_752027


namespace sum_of_cubes_l752_752094

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l752_752094


namespace set_intersection_l752_752992

def A := {x : ℝ | -5 < x ∧ x < 2}
def B := {x : ℝ | |x| < 3}

theorem set_intersection : {x : ℝ | x ∈ A ∧ x ∈ B} = {x : ℝ | -3 < x ∧ x < 2} :=
by
  sorry

end set_intersection_l752_752992


namespace contrapositive_of_a_gt_1_then_a_sq_gt_1_l752_752408

theorem contrapositive_of_a_gt_1_then_a_sq_gt_1 : 
  (∀ a : ℝ, a > 1 → a^2 > 1) → (∀ a : ℝ, a^2 ≤ 1 → a ≤ 1) :=
by 
  sorry

end contrapositive_of_a_gt_1_then_a_sq_gt_1_l752_752408


namespace increasing_interval_l752_752791

section

def translated_func (x : ℝ) : ℝ :=
  3 * Real.sin (2 * x - 2 * Real.pi / 3)

theorem increasing_interval :
  ∀ x, (Real.pi / 12 ≤ x ∧ x ≤ 7 * Real.pi / 12) →
       MonotoneOn translated_func (Set.Icc (Real.pi / 12) (7 * Real.pi / 12)) :=
by
  intro x h
  sorry

end

end increasing_interval_l752_752791


namespace count_primes_between_40_and_80_with_prime_remainder_l752_752281

-- Definition of prime numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of prime numbers within a specific range
def primes_in_range (a b : ℕ) : List ℕ :=
  (List.range' a (b - a + 1)).filter is_prime

-- Checking if a number is in a set of primes
def is_prime_remainder (n : ℕ) : Prop :=
  is_prime (n % 12)

-- Counting primes in range with prime remainders
def count_prime_remainders_in_range (a b : ℕ) : ℕ :=
  (primes_in_range a b).filter is_prime_remainder |>.length

theorem count_primes_between_40_and_80_with_prime_remainder : 
  count_prime_remainders_in_range 40 80 = 10 :=
by
  sorry

end count_primes_between_40_and_80_with_prime_remainder_l752_752281


namespace tangent_half_angle_squared_l752_752132

variable (a b c d α s : ℝ)

-- Given conditions
def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2
def cos_alpha (a b c d α : ℝ) : ℝ := (a^2 + d^2 - b^2 - c^2) / (2 * a * d + 2 * b * c)

-- Proof that the formula holds
theorem tangent_half_angle_squared :
  α = angle_between_sides a d →
  2 * s = a + b + c + d →
  (Real.tan (α / 2))^2 = (s - a) * (s - d) / ((s - b) * (s - c)) :=
by
  sorry

end tangent_half_angle_squared_l752_752132


namespace combined_length_platforms_l752_752080

def length_train_A := 500  -- meters
def time_train_A_platform_1 := 75  -- seconds
def time_train_A_signal := 25  -- seconds
def length_train_B := 400  -- meters
def time_train_B_platform_2 := 60  -- seconds
def time_train_B_signal := 20  -- seconds

theorem combined_length_platforms :
  let speed_train_A := length_train_A / time_train_A_signal in
  let length_platform_1 := (speed_train_A * time_train_A_platform_1) - length_train_A in
  let speed_train_B := length_train_B / time_train_B_signal in
  let length_platform_2 := (speed_train_B * time_train_B_platform_2) - length_train_B in
  length_platform_1 + length_platform_2 = 1800 := by
  sorry

end combined_length_platforms_l752_752080


namespace number_of_cheaters_l752_752438

noncomputable def number_of_students : ℕ := 2000
noncomputable def total_yes_responses : ℕ := 510
noncomputable def approx_half_students : ℕ := number_of_students / 2
noncomputable def estimated_cheaters_in_sample : ℕ := total_yes_responses - approx_half_students
noncomputable def estimated_cheaters : ℕ := 2 * estimated_cheaters_in_sample

theorem number_of_cheaters : estimated_cheaters = 20 :=
by
  have h1 : approx_half_students = 1000 := rfl
  have h2 : estimated_cheaters_in_sample = 10 :=
    by
      rw h1
      exact Nat.sub_self 500
  have h3 : estimated_cheaters = 2 * 10 :=
    by
      rw h2
      exact rfl
  exact Nat.mul 2 10

end number_of_cheaters_l752_752438


namespace triangle_area_50_l752_752186

theorem triangle_area_50 :
  let A := (0, 0)
  let B := (0, 10)
  let C := (-10, 0)
  let base := 10
  let height := 10
  0 + base * height / 2 = 50 := by
sorry

end triangle_area_50_l752_752186


namespace distance_between_parallel_lines_l752_752410

-- Definitions
def line_eq1 (x y : ℝ) : Prop := 3 * x - 4 * y - 12 = 0
def line_eq2 (x y : ℝ) : Prop := 6 * x - 8 * y + 11 = 0

-- Statement of the problem
theorem distance_between_parallel_lines :
  (∀ x y : ℝ, line_eq1 x y ↔ line_eq2 x y) →
  (∃ d : ℝ, d = 7 / 2) :=
by
  sorry

end distance_between_parallel_lines_l752_752410


namespace fair_coin_heads_probability_l752_752392

def fair_coin_probability : ℝ :=
  let total_outcomes := 2
  let favorable_outcomes := 1
  favorable_outcomes / total_outcomes

theorem fair_coin_heads_probability : fair_coin_probability = 1 / 2 := 
by
  intro
  sorry

end fair_coin_heads_probability_l752_752392


namespace isosceles_right_triangle_area_l752_752045

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (A : ℝ)
  (h_def : h = 6 * Real.sqrt 2)
  (rel_leg_hypotenuse : h = Real.sqrt 2 * l)
  (area_def : A = 1 / 2 * l * l) :
  A = 18 :=
by
  sorry

end isosceles_right_triangle_area_l752_752045


namespace restaurant_total_tables_l752_752146

theorem restaurant_total_tables (N O : ℕ) (h1 : 6 * N + 4 * O = 212) (h2 : N = O + 12) : N + O = 40 :=
sorry

end restaurant_total_tables_l752_752146


namespace moments_of_inertia_relation_l752_752709

variable (m : ℝ) -- Total mass of the system
variable (O X : ℝ) -- Points O (center of mass) and X (arbitrary point)
variable {ι : Type*} -- Type for indices of masses
variable (m_i : ι → ℝ) -- Masses
variable (x_i : ι → ℝ) -- Position vectors from O to each mass

-- Center of mass condition
axiom center_of_mass (h_sum_xi : ∑ i, m_i i * x_i i = 0)

-- Position vector from X to O
noncomputable def a := X - O

-- Moments of inertia definitions
noncomputable def I_O := ∑ i, m_i i * (x_i i)^2
noncomputable def I_X := ∑ i, m_i i * (x_i i + (a))^2

-- The theorem to be proven
theorem moments_of_inertia_relation : I_X = I_O + m * (X - O)^2 := sorry

end moments_of_inertia_relation_l752_752709


namespace triangle_cookie_cutters_count_l752_752923

theorem triangle_cookie_cutters_count :
  ∃ T : ℕ, (3 * T + 4 * 4 + 2 * 6 = 46) ∧ T = 6 :=
by
  use 6
  split
  sorry

end triangle_cookie_cutters_count_l752_752923


namespace annie_total_miles_l752_752161

theorem annie_total_miles (initial_gallons : ℕ) (miles_per_gallon : ℕ)
  (initial_trip_miles : ℕ) (purchased_gallons : ℕ) (final_gallons : ℕ)
  (total_miles : ℕ) :
  initial_gallons = 12 →
  miles_per_gallon = 28 →
  initial_trip_miles = 280 →
  purchased_gallons = 6 →
  final_gallons = 5 →
  total_miles = 364 := by
  sorry

end annie_total_miles_l752_752161


namespace trigonometric_identity_l752_752899

theorem trigonometric_identity
  (h1 : cos (70 * (Real.pi / 180)) ≠ 0)
  (h2 : sin (70 * (Real.pi / 180)) ≠ 0) :
  (1 / cos (70 * (Real.pi / 180)) - 2 / sin (70 * (Real.pi / 180))) = 
  (4 * sin (10 * (Real.pi / 180))) / sin (40 * (Real.pi / 180)) :=
  sorry

end trigonometric_identity_l752_752899


namespace zane_picked_up_62_pounds_l752_752913

variable (daliah : ℝ) (dewei : ℝ) (zane : ℝ)

def daliah_garbage : ℝ := 17.5
def dewei_garbage : ℝ := daliah - 2
def zane_garbage : ℝ := dewei * 4

theorem zane_picked_up_62_pounds (h_daliah : daliah = daliah_garbage) 
                                  (h_dewei : dewei = dewei_garbage) 
                                  (h_zane : zane = zane_garbage) : 
  zane = 62 :=
by 
  sorry

end zane_picked_up_62_pounds_l752_752913


namespace sum_of_squares_leq_six_fifths_l752_752083

theorem sum_of_squares_leq_six_fifths
  (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h_sum : a_1 + a_2 + a_3 + a_4 + a_5 = 0)
  (h_abs_diff : ∀ i j, i ∈ {1, 2, 3, 4, 5} → j ∈ {1, 2, 3, 4, 5} → |(if i = 1 then a_1 else if i = 2 then a_2 else if i = 3 then a_3 else if i = 4 then a_4 else a_5) 
                        - (if j = 1 then a_1 else if j = 2 then a_2 else if j = 3 then a_3 else if j = 4 then a_4 else a_5)| ≤ 1) :
  a_1^2 + a_2^2 + a_3^2 + a_4^2 + a_5^2 ≤ 6 / 5 := 
by
  sorry

end sum_of_squares_leq_six_fifths_l752_752083


namespace midpoint_fraction_l752_752796

theorem midpoint_fraction (a b : ℚ) (h1 : a = 3/4) (h2 : b = 5/6) : (a + b) / 2 = 19/24 := by
  sorry

end midpoint_fraction_l752_752796


namespace magnitude_e1_minus_2e2_max_value_x_over_a_l752_752993

noncomputable def e1 : ℝ^3 := sorry  -- Assume as unit vector
noncomputable def e2 : ℝ^3 := sorry  -- Assume as unit vector

theorem magnitude_e1_minus_2e2 : 
  (∥(e1 - (2 : ℝ) • e2)∥ = real.sqrt 3) :=
begin
  sorry
end

theorem max_value_x_over_a :
  (∀ (x y : ℝ), 
    (∠ e1 e2 = real.pi / 4) → 
    (x ≠ 0) → 
    let a := x • e1 + y • e2 
    in (∥x∥ / ∥a∥) ≤ real.sqrt 2) :=
begin
  sorry
end

end magnitude_e1_minus_2e2_max_value_x_over_a_l752_752993


namespace rectangle_fence_perimeter_l752_752071

theorem rectangle_fence_perimeter 
    (num_posts : ℕ) (post_width : ℝ) (spacing : ℝ) (width_ratio : ℝ) 
    (h_num_posts : num_posts = 36)
    (h_post_width : post_width = 0.5)
    (h_spacing : spacing = 3)
    (h_width_ratio : width_ratio = 2) :
    let width_posts := num_posts / (2 * (width_ratio + 1)),
        length_posts := width_ratio * width_posts,
        width_gap := width_posts - 1,
        length_gap := length_posts - 1,
        total_width := width_gap * spacing + width_posts * post_width,
        total_length := length_gap * spacing + length_posts * post_width
    in 2 * (total_width + total_length) = 177 := by
  sorry

end rectangle_fence_perimeter_l752_752071


namespace audrey_dreaming_fraction_l752_752165

theorem audrey_dreaming_fraction
  (total_asleep_time : ℕ) 
  (not_dreaming_time : ℕ)
  (dreaming_time : ℕ)
  (fraction_dreaming : ℚ)
  (h_total_asleep : total_asleep_time = 10)
  (h_not_dreaming : not_dreaming_time = 6)
  (h_dreaming : dreaming_time = total_asleep_time - not_dreaming_time)
  (h_fraction : fraction_dreaming = dreaming_time / total_asleep_time) :
  fraction_dreaming = 2 / 5 := 
by {
  sorry
}

end audrey_dreaming_fraction_l752_752165


namespace sarah_books_check_out_l752_752014

theorem sarah_books_check_out
  (words_per_minute : ℕ)
  (words_per_page : ℕ)
  (pages_per_book : ℕ)
  (reading_hours : ℕ)
  (number_of_books : ℕ) :
  words_per_minute = 40 →
  words_per_page = 100 →
  pages_per_book = 80 →
  reading_hours = 20 →
  number_of_books = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sarah_books_check_out_l752_752014


namespace TriangleProblem_l752_752243

theorem TriangleProblem
  (a b c A B C : ℝ)
  (area : ℝ)
  (h_triangle : a = b * cos C + (real.sqrt 3 / 3) * c * sin B)
  (h_area : (1/2) * a * c * (real.sqrt 3 / 2) = real.sqrt 3)
  (h_b : b = 2)
  (h_angle : B = π / 3) :
  a = 2 ∧ c = 2 :=
by
  sorry

end TriangleProblem_l752_752243


namespace distribute_balls_ways_l752_752557

theorem distribute_balls_ways :
  let n := 7
  let total_ways := 2^n
  let one_person_none := 2
  let one_person_one := 2 * n
  ∃ ways : ℕ, ways = total_ways - one_person_none - one_person_one ∧ ways = 112 :=
by 
  let n := 7
  let total_ways := 2^n
  let one_person_none := 2
  let one_person_one := 2 * n
  let ways := total_ways - one_person_none - one_person_one
  use ways
  split
  . rfl
  . rfl

end distribute_balls_ways_l752_752557


namespace find_b_sq_sum_l752_752067

-- Problem Statement and Conditions
theorem find_b_sq_sum :
  ∃ (b1 b2 b3 : ℝ), (∀ θ : ℝ, cos θ ^ 3 = b1 * cos θ + b2 * cos (2 * θ) + b3 * cos (3 * θ)) 
  ∧ (b1 = 3/4 ∧ b2 = 0 ∧ b3 = 1/4)
  ∧ (b1^2 + b2^2 + b3^2 = 5/8) :=
sorry

end find_b_sq_sum_l752_752067


namespace maximum_value_x_add_inv_x_l752_752782

noncomputable def maximum_x_add_inv_x (x : ℝ) (xs : List ℝ) : ℝ :=
  max (x + 1/x) (xs.map (λ xi => xi + 1/xi)).maximum

theorem maximum_value_x_add_inv_x
  (x₁ x₂ x₃ ... x₂₀₂₃ : ℝ)
  (h₁ : (x₁ + x₂ + x₃ + ... + x₂₀₂₃) = 2024)
  (h₂ : (1/x₁ + 1/x₂ + 1/x₃ + ... + 1/x₂₀₂₃) = 2024) :
  maximum_x_add_inv_x x₁ [x₂, x₃, ..., x₂₀₂₃] = 4049 / 2024 := sorry

end maximum_value_x_add_inv_x_l752_752782


namespace geometric_sequence_problem_l752_752686

variable {a : ℕ → ℝ} (q : ℝ)
variable {a3 a10 : ℝ}

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a(n+1) = a(n) * q
def condition1 (a : ℕ → ℝ) := a 5 * a 8 = 6
def condition2 (a : ℕ → ℝ) := a 3 + a 10 = 5

-- Question translated into a Lean theorem statement
theorem geometric_sequence_problem (h1 : geometric_sequence a q) (h2 : condition1 a) (h3 : condition2 a) :
  (a 20) / (a 13) = 3 / 2 ∨ (a 20) / (a 13) = 2 / 3 := sorry

end geometric_sequence_problem_l752_752686


namespace trajectory_of_moving_circle_l752_752949

-- Definitions for the given circles C1 and C2
def Circle1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def Circle2 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

-- Prove the trajectory of the center of the moving circle M
theorem trajectory_of_moving_circle (x y : ℝ) :
  ((∃ x_center y_center : ℝ, Circle1 x_center y_center ∧ Circle2 x_center y_center ∧ 
  -- Tangency conditions for Circle M
  (x - x_center)^2 + y^2 = (x_center - 2)^2 + y^2 ∧ (x - x_center)^2 + y^2 = (x_center + 2)^2 + y^2)) →
  (x = 0 ∨ x^2 - y^2 / 3 = 1) := 
sorry

end trajectory_of_moving_circle_l752_752949


namespace zane_picked_up_62_pounds_l752_752914

variable (daliah : ℝ) (dewei : ℝ) (zane : ℝ)

def daliah_garbage : ℝ := 17.5
def dewei_garbage : ℝ := daliah - 2
def zane_garbage : ℝ := dewei * 4

theorem zane_picked_up_62_pounds (h_daliah : daliah = daliah_garbage) 
                                  (h_dewei : dewei = dewei_garbage) 
                                  (h_zane : zane = zane_garbage) : 
  zane = 62 :=
by 
  sorry

end zane_picked_up_62_pounds_l752_752914


namespace BD_perpendicular_BC_l752_752942

-- Definitions to set the conditions and proof requirement
variables {A B C E M D : Type}

-- Open metric and analytic geometry spaces to define points and lines
open_locale classical
noncomputable theory

-- Our points and their properties as variables
variables (a c : ℝ) -- Representing coordinates

-- Definition of Points in Cartesian Plane
def pointA : ℝ × ℝ := (0, a)
def pointB : ℝ × ℝ := (-c, 0)
def pointC : ℝ × ℝ := (c, 0)
def pointE : ℝ × ℝ := (-c, 0)
def pointM : ℝ × ℝ := (-c, 0)
def pointD : ℝ × ℝ := (-c, 0)

-- Definitions of specific lines
def lineCx : ℝ → ℝ := λ x, 0
def lineBE : ℝ → ℝ := λ x, -c

-- Condition: Triangle ABC is isosceles at A
axiom isosceles_triangle_ABC : (pointA a c).fst = (pointB a c).fst ∨ (pointA a c).fst = (pointC a c).fst

-- Conditions for perpendicular lines (Cx ⊥ CA and BE ⊥ Cx)
axiom perpendicular_Cx_CA : ⊥ (lineCx c) (pointC a c)
axiom perpendicular_BE_Cx : ⊥ (lineBE c) (lineCx c)

-- Midpoint condition and Intersection point
axiom midpoint_MB : pointM a c = (pointB a c + pointE a c) / 2
axiom intersection_pointD : ∃ t, pointD a c = pointA a c + t * (pointM a c - pointA a c) ∧ pointD a c ∈ lineCx

-- The statement to prove
theorem BD_perpendicular_BC : (⊥ (lineBE c) (λ x, lineCx (pointC a c).snd)) := 
sorry

end BD_perpendicular_BC_l752_752942


namespace locus_is_circle_l752_752359

noncomputable def locus_of_tangent_circumcircles 
  (A B C D : Point) (circle : Circumference A B C D)
  : Set Point :=
  { X | let P := some_fixed_point; -- any fixed point where lines and tangents are concurrent
        let PA := distance P A;
        let PB := distance P B;
        PX = sqrt (PA * PB) -- leveraging power of a point theorem
      }

theorem locus_is_circle
  (A B C D : Point) (circle : Circumference A B C D) :
  ∃ P : Point, ∃ r : ℝ, r = sqrt (distance P A * distance P B) ∧
  ∀ X : Point, (X ∈ locus_of_tangent_circumcircles A B C D circle) ↔
                (distance X P = r) :=
by
  sorry

end locus_is_circle_l752_752359


namespace hyperbola_eccentricity_l752_752961

-- Conditions from the problem
variables {a b c : ℝ} {P F1 F2 : E} [normed_group E]

-- Define the hyperbola and foci conditions
def is_hyperbola (a b : ℝ) (P : E) : Prop :=
(P.x / a)^2 - (P.y / b)^2 = 1

-- Define vectors involved in the conditions
def focal_condition (O P F2 : E) : Prop :=
(P + F2) • (P - F2) = 0

-- Define the distance conditions between P, F1, and F2
def distance_condition (P F1 F2 : E) : Prop :=
dist P F1 = sqrt 3 * dist P F2

-- Eccentricity

def eccentricity (a : ℝ) : ℝ :=
c / a

-- The theorem stating the desired proof
theorem hyperbola_eccentricity (a b c : ℝ) (P F1 F2 : E) [normed_group E]
  (h1 : a > 0) (h2 : b > 0)
  (hyperbola : is_hyperbola a b P)
  (focal_cond : focal_condition (0 : E) P F2)
  (dist_cond : distance_condition P F1 F2) :
  eccentricity a = sqrt 3 + 1 :=
sorry

end hyperbola_eccentricity_l752_752961


namespace a_general_form_T_sum_general_lambda_range_l752_752599

-- Definition of sequence a_n and given conditions
def S (n : ℕ) (h : n > 0) : ℤ := 2 * a n h - 1
def a : ℕ → ℤ := λ n, if h : (n > 0) then 2^(n-1) else 0

theorem a_general_form (n : ℕ) (h : n > 0) : 2 * a n h - 1 = S n h := sorry

-- Definition of sequence b_n and its conditions
def b (n : ℕ) : ℤ := 2 * n * a n (by linarith)
def T (n : ℕ) : ℤ := ∑ i in range n, b i

theorem T_sum_general (n : ℕ) : T n = (n-1) * 2^(n+1) + 2 := sorry

-- Definition of sequence c_n and condition on λ
def c (n : ℕ) (λ : ℤ) : ℤ := 3^n + 2 * (-1)^(n-1) * λ * a n (by linarith)

theorem lambda_range (λ : ℤ) : (-3/2 : ℚ) < λ ∧ λ < (1 : ℚ) → (∀ n : ℕ, n > 0 → c (n+1) λ > c n λ) := sorry

end a_general_form_T_sum_general_lambda_range_l752_752599


namespace sum_of_grid_numbers_l752_752922

theorem sum_of_grid_numbers (A E: ℕ) (S: ℕ) 
    (hA: A = 2) 
    (hE: E = 3)
    (h1: ∃ B : ℕ, 2 + B = S ∧ 3 + B = S)
    (h2: ∃ D : ℕ, 2 + D = S ∧ D + 3 = S)
    (h3: ∃ F : ℕ, 3 + F = S ∧ F + 3 = S)
    (h4: ∃ G H I: ℕ, 
         2 + G = S ∧ G + H = S ∧ H + C = S ∧ 
         3 + H = S ∧ E + I = S ∧ H + I = S):
  A + B + C + D + E + F + G + H + I = 22 := 
by 
  sorry

end sum_of_grid_numbers_l752_752922


namespace smallest_possible_value_of_n_l752_752020

theorem smallest_possible_value_of_n 
  {a b c m n : ℕ} 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (hc_pos : c > 0) 
  (h_ordering : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c = 3010) 
  (h_factorial : a.factorial * b.factorial * c.factorial = m * 10^n) 
  (h_m_not_div_10 : ¬ (10 ∣ m)) 
  : n = 746 := 
sorry

end smallest_possible_value_of_n_l752_752020


namespace max_quarters_in_wallet_l752_752030

theorem max_quarters_in_wallet:
  ∃ (q n : ℕ), 
    (30 * n) + 50 = 31 * (n + 1) ∧ 
    q = 22 :=
by
  sorry

end max_quarters_in_wallet_l752_752030


namespace modulus_of_complex_number_l752_752977

theorem modulus_of_complex_number (z : ℂ) (h : z + 3/z = 0) : |z| = Real.sqrt 3 := by
  sorry

end modulus_of_complex_number_l752_752977


namespace find_length_BF_l752_752464

-- Defining the problem conditions
variables (A B C D E F : Type) [has_coe_to_fun A B]
variables [linear_ordered_field C] [add_comm_group D] [module C D]
variables {AE : C} {DE : C} {CE : C}
variable [normal : AE = 3]
variable [distDE : DE = 5]
variable [distCE : CE = 7]
variable [right_angle_ABC : ∠ A B C = π/2]
variable [right_angle_DEF : ∠ D E F = π/2]
variable [points_on_AC_E : E ∈ line(A, C)]
variable [points_on_AC_F : F ∈ line(A, C)]
variable [perpendicular_DE_AC : is_perpendicular (line(D, E)) (line(A, C))]
variable [perpendicular_BF_AC : is_perpendicular (line(B, F)) (line(A, C))]

-- Stating the proof goal
theorem find_length_BF : 
  BF = 4.2 :=
sorry -- proof goes here

end find_length_BF_l752_752464


namespace evaluate_f_at_2_l752_752588

def f (x : ℝ) : ℝ := 2^x + 1

theorem evaluate_f_at_2 : f 2 = 5 := by
  sorry

end evaluate_f_at_2_l752_752588


namespace goldfish_equal_in_seven_months_l752_752868

/-- Define the growth of Alice's goldfish: they triple every month. -/
def alice_goldfish (n : ℕ) : ℕ := 3 * 3 ^ n

/-- Define the growth of Bob's goldfish: they quadruple every month. -/
def bob_goldfish (n : ℕ) : ℕ := 256 * 4 ^ n

/-- The main theorem we want to prove: For Alice and Bob's goldfish count to be equal,
    it takes 7 months. -/
theorem goldfish_equal_in_seven_months : ∃ n : ℕ, alice_goldfish n = bob_goldfish n ∧ n = 7 := 
by
  sorry

end goldfish_equal_in_seven_months_l752_752868


namespace parallelogram_square_l752_752803

-- Definitions required for the problem
variable (Q : Type) [Quadrilateral Q]

/-- A parallelogram with perpendicular and equal diagonals is a square. -/
theorem parallelogram_square (P : Q) (hParallelogram : isParallelogram P)
  (hPerpendicular : diagonalsPerpendicular P) (hEqual : diagonalsEqual P) : isSquare P :=
by sorry

end parallelogram_square_l752_752803


namespace first_discount_is_20_percent_l752_752424

-- Define the problem parameters
def original_price : ℝ := 200
def final_price : ℝ := 152
def second_discount : ℝ := 0.05

-- Define the function to compute the price after two discounts
def price_after_discounts (first_discount : ℝ) : ℝ := 
  original_price * (1 - first_discount) * (1 - second_discount)

-- Define the statement that we need to prove
theorem first_discount_is_20_percent : 
  ∃ (first_discount : ℝ), price_after_discounts first_discount = final_price ∧ first_discount = 0.20 :=
by
  sorry

end first_discount_is_20_percent_l752_752424


namespace rectangle_to_total_height_ratio_l752_752876

theorem rectangle_to_total_height_ratio 
  (total_area : ℕ)
  (width : ℕ)
  (area_per_side : ℕ)
  (height : ℕ)
  (triangle_base : ℕ)
  (triangle_area : ℕ)
  (rect_area : ℕ)
  (total_height : ℕ)
  (ratio : ℚ)
  (h_eqn : 3 * height = area_per_side)
  (h_value : height = total_area / (2 * 3))
  (total_height_eqn : total_height = 2 * height)
  (ratio_eqn : ratio = height / total_height) :
  total_area = 12 → width = 3 → area_per_side = 6 → triangle_base = 3 →
  triangle_area = triangle_base * height / 2 → rect_area = width * height →
  rect_area = area_per_side → ratio = 1 / 2 :=
by
  intros
  sorry

end rectangle_to_total_height_ratio_l752_752876


namespace Shara_shells_total_l752_752018

def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

theorem Shara_shells_total : (initial_shells + (shells_per_day * days) + shells_fourth_day) = 41 := by
  sorry

end Shara_shells_total_l752_752018


namespace sum_of_cubes_l752_752100

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752100


namespace h_at_3_l752_752182

-- Define the function h(x) based on the given condition
noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^2^3 + 1) * ... * (x^2^2008 + 1) - 1) / (x^(2^2009 - 1) - 1)

-- Prove that h(3) equals 3
theorem h_at_3 : h 3 = 3 := by
  sorry

end h_at_3_l752_752182


namespace tile_problem_result_l752_752793

-- We define the number of tiles
def num_tiles := 12
def player_tiles := 4
def first_sum_odd := 1
def others_sum_even := 2 ∨ 4 ∨ 0

-- Given m and n represent the probability simplified as m/n
def m := 192
def n := 99

-- The sum m+n
def result := m + n

theorem tile_problem_result :
  num_tiles = 12 ∧ player_tiles = 4 ∧ first_sum_odd ∧ others_sum_even → result = 291 :=
by
  sorry

end tile_problem_result_l752_752793


namespace side_length_of_base_l752_752026

-- Given conditions
def lateral_face_area := 90 -- Area of one lateral face in square meters
def slant_height := 20 -- Slant height in meters

-- The theorem statement
theorem side_length_of_base 
  (s : ℝ)
  (h : ℝ := slant_height)
  (a : ℝ := lateral_face_area)
  (h_area : 2 * a = s * h) :
  s = 9 := 
sorry

end side_length_of_base_l752_752026


namespace assigned_digit_matches_first_digit_l752_752176

-- Define the digit assignment function and the specific conditions
def d (N : Vector ℕ 7) : ℕ := sorry

-- Specific 7-digit numbers involved
def N1 := Vector.ofFn (λ _ => 1)
def N2 := Vector.ofFn (λ _ => 2)
def N3 := Vector.ofFn (λ _ => 3)
def N4 := Vector.ofFn (λ i => if i == 0 then 1 else 2)

-- Conditions given in the problem
axiom condition1 : d N1 = 1
axiom condition2 : d N2 = 2
axiom condition3 : d N3 = 3
axiom condition4 : d N4 = 1

-- The main theorem we need to prove
theorem assigned_digit_matches_first_digit (N : Vector ℕ 7) : d N = N.head :=
by sorry

end assigned_digit_matches_first_digit_l752_752176


namespace initial_price_of_product_l752_752060

variable (r s : ℝ)

theorem initial_price_of_product (h1 : 0 < r) (h2 : 0 < s) : 
  let x := 1 / (1 + (r - s) / 100 - (r * s) / 10000) in
  x = 10000 / (10000 + 100 * (r - s) - r * s) :=
by 
  sorry

end initial_price_of_product_l752_752060


namespace exists_four_mutually_acquainted_l752_752675

-- Definition of acquaintanceship relation
def Acquaintance (X : Type) := X → X → Prop

-- Given conditions
section
variables {M : Type} [Fintype M] [DecidableRel (Acquaintance M)]
variable h_card : Fintype.card M = 9

-- Condition: For any three men, at least two are mutually acquainted
axiom acquaintance_condition : ∀ (s : Finset M), s.card = 3 → ∃ (x y : M), x ∈ s ∧ y ∈ s ∧ Acquaintance M x y

-- To prove: There exists a subset of four men who are all mutually acquainted
theorem exists_four_mutually_acquainted :
  ∃ (s : Finset M), s.card = 4 ∧ ∀ (x y : M), x ∈ s → y ∈ s → x ≠ y → Acquaintance M x y := sorry
end

end exists_four_mutually_acquainted_l752_752675


namespace sum_of_solutions_l752_752043

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋₊

def satisfies_equation (x : ℝ) : Prop :=
  x - floor x = 1 / (floor x ^ 2)

def smallest_positive_solutions : List ℝ :=
  [2.25, 3.11111111111111, 4.0625]

def sum_smallest_solutions : ℚ :=
  1357 / 144

theorem sum_of_solutions :
  ∑ x in smallest_positive_solutions, x = 9 + 73 / 144 :=
by sorry

end sum_of_solutions_l752_752043


namespace triangle_interior_points_l752_752007

theorem triangle_interior_points (A B C M N : Point)
  (h1 : interior A B C M N)
  (h2 : ∠ M A B = ∠ N A C)
  (h3 : ∠ M B A = ∠ N B C) :
  (M.dist_to A * N.dist_to A / (A.dist_to B * A.dist_to C)) + 
  (M.dist_to B * N.dist_to B / (B.dist_to A * B.dist_to C)) + 
  (M.dist_to C * N.dist_to C / (C.dist_to A * C.dist_to B)) = 1 :=
sorry

end triangle_interior_points_l752_752007


namespace no_nat_number_with_digits_product_1560_l752_752920

def digits (n : ℕ) : List ℕ :=
  toDigits 10 n -- Helper function for simplicity; assumes toDigits is provided or defined elsewhere

def digits_product (n : ℕ) : ℕ :=
  (digits n).prod

theorem no_nat_number_with_digits_product_1560 :
  ¬ ∃ (n : ℕ), digits_product n = 1560 :=
by
  sorry

end no_nat_number_with_digits_product_1560_l752_752920


namespace find_sum_of_paintable_numbers_l752_752644
noncomputable def isPaintable (h t u: ℕ) : Prop :=
  ∀ n: ℕ, (n mod h = 1) ∨ (n mod t = 3) ∨ (n mod u = 5)

noncomputable def isValidTriple (h t u : ℕ) : Prop :=
  isPaintable h t u ∧ (t % 2 ≠ 0) ∧ Prime u

noncomputable def paintableNumber : ℕ :=
  let triple := (4, 3, 5)
  if h, t, u ∈ triple then isValidTriple h t u then 100 * t + 10 * u + h else 0

theorem find_sum_of_paintable_numbers (h t u : ℕ) : 
  ∑ i in (finset.range 1000), if isValidTriple h t u then paintableNumber h t u else 0 = 354 :=
by sorry

end find_sum_of_paintable_numbers_l752_752644


namespace twenty_five_percent_of_2004_l752_752456

theorem twenty_five_percent_of_2004 : (1 / 4 : ℝ) * 2004 = 501 := by
  sorry

end twenty_five_percent_of_2004_l752_752456


namespace length_ab_l752_752126

-- Define the conditions of the problem
variables (x y : ℝ)
axiom right_triangle : ∠B = 90
axiom length_ac : sqrt (x^2 + y^2) = 100
axiom slope_ac : y / x = 4 / 3

-- Define the theorem statement
theorem length_ab (y x : ℝ) (right_triangle : ∠B = 90) (length_ac : sqrt (x^2 + y^2) = 100) (slope_ac : y / x = 4 / 3) : y = 80 :=
sorry

end length_ab_l752_752126


namespace infinite_squares_and_circles_difference_l752_752908

theorem infinite_squares_and_circles_difference 
  (side_length : ℝ)
  (h₁ : side_length = 1)
  (square_area_sum : ℝ)
  (circle_area_sum : ℝ)
  (h_square_area : square_area_sum = (∑' n : ℕ, (side_length / 2^n)^2))
  (h_circle_area : circle_area_sum = (∑' n : ℕ, π * (side_length / 2^(n+1))^2 ))
  : square_area_sum - circle_area_sum = 2 - (π / 2) :=
by 
  sorry 

end infinite_squares_and_circles_difference_l752_752908


namespace point_not_in_square_l752_752938

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def is_square (points : list (ℝ × ℝ)) : Prop :=
  match points with
  | [a, b, c, d] =>
    let d1 := distance a b,
        d2 := distance a c,
        d3 := distance a d,
        d4 := distance b c,
        d5 := distance b d,
        d6 := distance c d in
    d1 = d2 ∧ d1 = d3 ∧ d1 = d4 ∧ d5 = d6 ∧
    d1^2 + d1^2 = d5^2
  | _ => false

theorem point_not_in_square :
  ¬is_square [(-1, 3), (0, -4), (-2, -1), (1, 1), (3, -2)] ∧
  is_square [(0, -4), (-2, -1), (1, 1), (3, -2)] :=
by
  sorry

end point_not_in_square_l752_752938


namespace ab_cd_sum_l752_752023

noncomputable def g : ℕ → ℕ :=
  λ x, if x = 1 then 6 else if x = 2 then 4 else if x = 4 then 1 else 0

theorem ab_cd_sum :
  let a := 2,
      b := g (g(2)),
      c := 4,
      d := g (g(4))
  in a * b + c * d = 26 :=
by
  let a := 2
  let b := g (g(2))
  let c := 4
  let d := g (g(4))
  have hb : b = 1 := by
    unfold g
    simp
  have hd : d = 6 := by
    unfold g
    simp
  rw [hb, hd]
  calc
    a * b + c * d = 2 * 1 + 4 * 6 := by refl
                ... = 2 + 24 := by norm_num
                ... = 26 := by norm_num

end ab_cd_sum_l752_752023


namespace total_tickets_l752_752870

theorem total_tickets (tickets_first_day tickets_second_day tickets_third_day : ℕ) 
  (h1 : tickets_first_day = 5 * 4) 
  (h2 : tickets_second_day = 32)
  (h3 : tickets_third_day = 28) :
  tickets_first_day + tickets_second_day + tickets_third_day = 80 := by
  sorry

end total_tickets_l752_752870


namespace find_angle_l752_752989

variables (a b : EuclideanSpace ℝ (Fin 2))

def norm_a : ℝ := ‖a‖
def norm_b : ℝ := ‖b‖
def dot_prod : ℝ := a ⬝ b
def angle_between : ℝ := Real.angleBetween a b

theorem find_angle
  (h1 : norm_a = 1)
  (h2 : norm_b = Real.sqrt 2)
  (h3 : dot_prod = 1) :
  angle_between = π / 4 :=
sorry

end find_angle_l752_752989


namespace exists_four_mutually_acquainted_l752_752674

-- Definition of acquaintanceship relation
def Acquaintance (X : Type) := X → X → Prop

-- Given conditions
section
variables {M : Type} [Fintype M] [DecidableRel (Acquaintance M)]
variable h_card : Fintype.card M = 9

-- Condition: For any three men, at least two are mutually acquainted
axiom acquaintance_condition : ∀ (s : Finset M), s.card = 3 → ∃ (x y : M), x ∈ s ∧ y ∈ s ∧ Acquaintance M x y

-- To prove: There exists a subset of four men who are all mutually acquainted
theorem exists_four_mutually_acquainted :
  ∃ (s : Finset M), s.card = 4 ∧ ∀ (x y : M), x ∈ s → y ∈ s → x ≠ y → Acquaintance M x y := sorry
end

end exists_four_mutually_acquainted_l752_752674


namespace lily_sees_leo_l752_752376

theorem lily_sees_leo : 
  ∀ (d₁ d₂ v₁ v₂ : ℝ), 
  d₁ = 0.75 → 
  d₂ = 0.75 → 
  v₁ = 15 → 
  v₂ = 9 → 
  (d₁ + d₂) / (v₁ - v₂) * 60 = 15 :=
by 
  intros d₁ d₂ v₁ v₂ h₁ h₂ h₃ h₄
  -- skipping the proof with sorry
  sorry

end lily_sees_leo_l752_752376


namespace tan_of_angle_in_second_quadrant_l752_752962

-- Define the problem statement in Lean 4
theorem tan_of_angle_in_second_quadrant (α : ℝ) (h1 : real.cos α = -4/5) (h2 : π/2 < α ∧ α < π) : 
  real.tan α = -3/4 := 
sorry

end tan_of_angle_in_second_quadrant_l752_752962


namespace autumn_pencils_l752_752526

theorem autumn_pencils : 
  ∀ (start misplaced broken found bought : ℕ),
  start = 20 →
  misplaced = 7 → 
  broken = 3 → 
  found = 4 → 
  bought = 2 → 
  start - misplaced - broken + found + bought = 16 :=
by 
  intros start misplaced broken found bought h_start h_misplaced h_broken h_found h_bought,
  rw [h_start, h_misplaced, h_broken, h_found, h_bought],
  linarith

end autumn_pencils_l752_752526


namespace cost_price_calculation_l752_752858

noncomputable def CP : ℝ := 100 / 1.20

theorem cost_price_calculation (SP : ℝ) (profit_rate : ℝ) (h1 : SP = 100) (h2 : profit_rate = 0.20) : CP = 83.33 :=
by
  have h3 : SP = CP * (1 + profit_rate),
  {
    rw [h1, h2],
    exact (100 = CP * 1.20)
  },
  rw h3,
  exact 83.33

#check cost_price_calculation

end cost_price_calculation_l752_752858


namespace sin_cos_sum_identity_l752_752816

theorem sin_cos_sum_identity :
  sin (20 * π / 180) * cos (40 * π / 180) + cos (20 * π / 180) * sin (140 * π / 180) = (√3 / 2) :=
by
  sorry

end sin_cos_sum_identity_l752_752816


namespace repeating_decimal_to_fraction_l752_752564

theorem repeating_decimal_to_fraction : (0.3636363636...) = (4 / 11) := 
sorry

end repeating_decimal_to_fraction_l752_752564


namespace trig_identity_l752_752894

theorem trig_identity :
  (1 / real.cos (70 * real.pi / 180) - 2 / real.sin (70 * real.pi / 180)) = (2 * (real.sin (50 * real.pi / 180) - 1) / real.sin (40 * real.pi / 180)) :=
  sorry

end trig_identity_l752_752894


namespace find_f_cos30_l752_752965

noncomputable def f (t : ℝ) : ℝ := (t^2 - 1) / 2

theorem find_f_cos30 :
  let alpha := real.pi / 6 in
  f (real.cos (30 * real.pi / 180)) = -1 / 8 :=
by
  -- Placeholder for the detailed proof, not part of the task
  sorry

end find_f_cos30_l752_752965


namespace commercial_break_duration_l752_752559

def long_commercials := [5, 6, 7]
def short_commercials := list.repeat 2 11
def second_long_interruption := 3
def sixth_short_interruption := 0.5

def total_long_commercial_time := long_commercials.sum
def total_short_commercial_time := short_commercials.sum
def total_commercial_time := total_long_commercial_time + total_short_commercial_time

def total_interruption_time := second_long_interruption + sixth_short_interruption

def total_commercial_break_time := total_commercial_time + total_interruption_time

theorem commercial_break_duration :
  total_commercial_break_time = 43.5 :=
by 
  -- Calculate the total time for long and short commercials
  have h1 : total_long_commercial_time = 5 + 6 + 7 := by simp [long_commercials, list.sum]
  have h2 : total_short_commercial_time = 11 * 2 := by simp [short_commercials, list.sum]
  have h3 : total_commercial_time = 40 := by simp [total_commercial_time, h1, h2]
  -- Calculate the total interruption time
  have h4 : total_interruption_time = 3 + 0.5 := by simp [total_interruption_time, second_long_interruption, sixth_short_interruption]
  -- The total duration of commercial break
  simp [total_commercial_break_time, h3, h4]
  show 40 + 3.5 = 43.5
  norm_num

end commercial_break_duration_l752_752559


namespace eighteen_spies_placement_l752_752152

-- Define the size of the board
def board_size : Nat := 6

-- A spy's vision includes the two cells in front and one cell to the left and right
def spy_vision_range (board : Array (Array (Option String))) (x y : Nat) : Set (Nat × Nat) :=
  let vision_cells := 
    {(i, j) | (i, j) ∈ {(x+1, y), (x+2, y), (x, y-1), (x, y+1)} ∧ 
                   i < board_size ∧ j < board_size ∧ i ≥ 0 ∧ j ≥ 0}
  vision_cells

-- Define a valid placement of spies
def valid_placement (board : Array (Array (Option String))) (n : Nat) : Prop :=
  ∀ i j, (board[i][j] = some "spy") → 
    ∀ id jd, (id, jd) ∈ spy_vision_range board i j → (board[id][jd] ≠ some "spy")

-- Define the board with spies
def board_with_spies : Array (Array (Option String)) :=
  #[ #[some "spy", none, none, none, none, some "spy"]
   ,#[none, some "spy", none, some "spy", none, none]
   ,#[none, none, none, none, some "spy", none]
   ,#[none, some "spy", none, some "spy", none, none]
   ,#[none, some "spy", none, none, none, some "spy"]
   ,#[some "spy", none, none, none, none, some "spy"] ]

theorem eighteen_spies_placement : valid_placement board_with_spies 18 :=
sorry

end eighteen_spies_placement_l752_752152


namespace good_function_inequality_l752_752449

def is_good_function (f : ℚ+ → ℚ+) : Prop :=
  ∀ x y : ℚ+, f(x) + f(y) ≥ 4 * f(x + y)

theorem good_function_inequality
  (f : ℚ+ → ℚ+)
  (h_good : is_good_function f)
  (x y z : ℚ+) :
  f(x) + f(y) + f(z) ≥ 8 * f(x + y + z) :=
sorry

end good_function_inequality_l752_752449


namespace determine_base_l752_752918

theorem determine_base (x b : ℝ) (h1 : 7^(x+7) = 9^x) (h2 : x = Real.logb b (7^7)) : b = (9/7) := 
by {
    sorry
}

end determine_base_l752_752918


namespace imaginary_part_of_z_eq_l752_752973

-- Define the context: z is a complex number, z satisfies the equation
variables {z : ℂ}

-- Define the condition
def condition (z : ℂ) := z * (3 - 4 * complex.I) = 5

-- State the theorem: Prove that the imaginary part of z is 5/7
theorem imaginary_part_of_z_eq (hz : condition z) : z.im = 5 / 7 :=
sorry

-- Ensure the statement builds successfully
#check imaginary_part_of_z_eq

end imaginary_part_of_z_eq_l752_752973


namespace triangle_property_l752_752252

theorem triangle_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
    (h_perimeter : a + b + c = 12) (h_inradius : 2 * (a + b + c) = 24) :
    ¬((a^2 + b^2 = c^2) ∨ (a^2 + b^2 > c^2) ∨ (c^2 > a^2 + b^2)) := 
sorry

end triangle_property_l752_752252


namespace constant_dot_product_l752_752602

-- Definition of the ellipse with the given conditions
def ellipse_c (x y a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def focal_length (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 ∧ 2 * c = 2 * Real.sqrt 3

def point_on_ellipse (x y a b : ℝ) : Prop :=
  x = Real.sqrt 2 ∧ y = -(Real.sqrt 2) / 2

-- Definition of the dot product and conditions for proving the result
def dot_product_condition (D A B : ℝ × ℝ) : Prop :=
  let DA := (A.1 - D.1, A.2 - D.2)
  let DB := (B.1 - D.1, B.2 - D.2)
  DA.1 * DB.1 + DA.2 * DB.2 = 3

-- The theorem to prove our desired result
theorem constant_dot_product :
  ∀ (a b x y : ℝ) (D A B : ℝ × ℝ),
    ellipse_c x y a b →
    focal_length a b (Real.sqrt 3) →
    point_on_ellipse x y a b →
    dot_product_condition D (0, 1) (0, -1) ∨
    dot_product_condition D ((1 / Real.sqrt ((1 + A.2^2) / A.2)), A.2 / Real.sqrt ((1 + A.2^2) / A.2)) ((-1 / Real.sqrt ((1 + A.2^2) / A.2)), -A.2 / Real.sqrt ((1 + A.2^2) / A.2))
    sorry

end constant_dot_product_l752_752602


namespace autumn_pencils_l752_752529

-- Define the conditions of the problem.
def initial_pencils := 20
def misplaced_pencils := 7
def broken_pencils := 3
def found_pencils := 4
def bought_pencils := 2

-- Define the number of pencils lost and gained.
def pencils_lost := misplaced_pencils + broken_pencils
def pencils_gained := found_pencils + bought_pencils

-- Define the final number of pencils.
def final_pencils := initial_pencils - pencils_lost + pencils_gained

-- The theorem we want to prove.
theorem autumn_pencils : final_pencils = 16 := by
  sorry

end autumn_pencils_l752_752529


namespace average_minutes_55_l752_752859

def average_minutes_heard (total_people : ℕ) (talk_minutes : ℕ) 
  (percentage_entire : ℕ) (percentage_miss : ℕ) : ℚ :=
let remainder := total_people * (100 - percentage_entire - percentage_miss) / 100,
    half_listens := remainder / 4,
    quarter_listens := remainder / 4,
    three_quarters_listens := remainder / 2,
    total_minutes_heard := 
      (total_people * percentage_entire * talk_minutes / 100) + 
      (half_listens * (talk_minutes / 2)) + 
      (quarter_listens * (talk_minutes / 4)) + 
      (three_quarters_listens * (3 * talk_minutes / 4)) in
total_minutes_heard / total_people

theorem average_minutes_55 (total_people : ℕ) (talk_minutes : ℕ) 
  (percentage_entire : ℕ) (percentage_miss : ℕ)
  (h_talk : talk_minutes = 90)
  (h_percentage_entire : percentage_entire = 30)
  (h_percentage_miss : percentage_miss = 15) : 
  average_minutes_heard total_people talk_minutes percentage_entire percentage_miss = 55 := sorry

end average_minutes_55_l752_752859


namespace find_sum_x_y_l752_752927

theorem find_sum_x_y :
  (∃ x y : ℤ, x * Real.log 27 * Real.logb 13 Real.exp = 27 * Real.logb 13 y ∧ y > 70 ∧ 3 ^ x = y ^ 9 ∧ x + y = 117) :=
by {
  use 36,
  use 81,
  have h₁ : 36 * Real.log 27 * (1 / (Real.log 13)) = 27 * (1 / (Real.log 13)) * Real.log 81 := by sorry,
  have h₂ : 81 > 70 := by norm_num,
  have h₃ : 3 ^ 36 = 81 ^ 9 := by sorry,
  have h₄ : 36 + 81 = 117 := by norm_num,
  exact ⟨h₁, h₂, h₃, h₄⟩,
}

end find_sum_x_y_l752_752927


namespace problem_l752_752258

noncomputable def f (ω x : ℝ) := sqrt 3 * cos (ω * x / 2) * sin (ω * x / 2) + cos (ω * x / 2) ^ 2 + cos (ω * x)

theorem problem
  (ω : ℝ) 
  (h_ω_pos : ω > 0)
  (h_passes : f ω (π / 3) = 1 / 2)
  (h_distance : (1 / 4) * (2 * π / ω) > π / 10) :
  (ω = 2 ∧ ∀ x : ℝ, f ω x = sqrt 3 * sin (2 * x + π / 3) + 1 / 2) := 
sorry

end problem_l752_752258


namespace range_of_a_l752_752964

noncomputable def f (x a : ℝ) := (x^2 - 2 * a * x) * Real.exp x

theorem range_of_a {a : ℝ} (h_nonneg : a ≥ 0)
    (h_mono_decreasing : ∀ x ∈ Set.Icc (-1 : ℝ) 1, deriv (λ x, f x a) x ≤ 0) :
  a ≥ 3 / 4 :=
by
  sorry

end range_of_a_l752_752964


namespace integral_f_eq_three_l752_752134

noncomputable def f (x : ℝ) : ℝ := 2 - |1 - x|

theorem integral_f_eq_three :
  ∫ x in 0..2, f x = 3 :=
by 
  sorry

end integral_f_eq_three_l752_752134


namespace range_of_x_l752_752222

theorem range_of_x (a b c x : ℝ) (h1 : a^2 + 2 * b^2 + 3 * c^2 = 6) (h2 : a + 2 * b + 3 * c > |x + 1|) : -7 < x ∧ x < 5 :=
by
  sorry

end range_of_x_l752_752222


namespace geometric_sequence_a2_l752_752753

noncomputable def geometric_sequence_sum (n : ℕ) (a : ℝ) : ℝ :=
  a * (3^n) - 2

theorem geometric_sequence_a2 (a : ℝ) : (∃ a1 a2 a3 : ℝ, 
  a1 = geometric_sequence_sum 1 a ∧ 
  a1 + a2 = geometric_sequence_sum 2 a ∧ 
  a1 + a2 + a3 = geometric_sequence_sum 3 a ∧ 
  a2 = 6 * a ∧ 
  a3 = 18 * a ∧ 
  (6 * a)^2 = (a1) * (a3) ∧ 
  a = 2) →
  a2 = 12 :=
by
  intros h
  sorry

end geometric_sequence_a2_l752_752753


namespace coin_order_l752_752066

def covers (x y : String) := (x, y)

theorem coin_order :
  ∀ (F E D C B A : String),
  covers F E ∧ ¬ covers F D ∧
  covers E B ∧ covers E C ∧
  covers D B ∧
  covers C A ∧
  (∀ z, z ≠ A → ∃ w, covers w z) →
  [F, D, E, C, B, A] = ["F", "D", "E", "C", "B", "A"] :=
by
  intros F E D C B A h
  -- additional reasoning required to complete the proof but skipped here
  sorry

end coin_order_l752_752066


namespace polar_cartesian_equiv_max_distance_l752_752990

noncomputable def polar_to_cartesian (ρ θ : ℝ) : (ℝ × ℝ) :=
  (ρ * real.cos θ, ρ * real.sin θ)

theorem polar_cartesian_equiv (ρ θ : ℝ) (h : ρ = 2 * real.sin θ) :
    let (x, y) := polar_to_cartesian ρ θ in x^2 + y^2 - 2 * y = 0 :=
by
  cases (polar_to_cartesian ρ θ) with x y
  simp only at *
  sorry

theorem max_distance 
  (t : ℝ) 
  (hLine : ∀ t, (x, y) = (-3/5 * t + 2, 4/5 * t)) 
  (rho : ℝ) (theta : ℝ) (hCircle : rho = 2 * real.sin theta) 
  (hMNMax : ∀ M N, M = (2, 0) → (x, y) = polar_to_cartesian rho θ → dist M N ≤ real.sqrt 5 + 1 ) :
  true :=
by
  -- prove that the maximum value of MN is sqrt(5) + 1
  sorry

end polar_cartesian_equiv_max_distance_l752_752990


namespace third_number_in_decomposition_of_5_pow_4_l752_752213

theorem third_number_in_decomposition_of_5_pow_4 :
  ∀ (m n : ℕ), m ≥ 2 ∧ n = 4 ∧ m = 5 → (∃ seq : ℕ → ℕ, seq 3 = 125) :=
by
  intros m n h
  cases h with hm hn
  cases hn with hn1 hn2
  sorry

end third_number_in_decomposition_of_5_pow_4_l752_752213


namespace square_perimeter_l752_752763

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by
  sorry

end square_perimeter_l752_752763


namespace calculate_neg2_add3_l752_752168

theorem calculate_neg2_add3 : (-2) + 3 = 1 :=
  sorry

end calculate_neg2_add3_l752_752168


namespace angle_C_is_pi_over_3_l752_752817

theorem angle_C_is_pi_over_3
  (a b c : ℝ)
  (h_parallel: (a + c, b) = (k * (b - a), k * (c - a)) for some k : ℝ) :
  angle_C = (π / 3) :=
begin
  -- Let \(\overrightarrow{P} = (a + c, b)\) and \(\overrightarrow{q} = (b - a, c - a)\) such that \(\overrightarrow{P} \parallel \overrightarrow{q}\),
  -- implying there exists a scalar \( k \) such that \( (a + c, b) = k \cdot (b - a, c - a) \),
  -- hence, it follows that \( C = \frac{\pi}{3} \).
  sorry
end

end angle_C_is_pi_over_3_l752_752817


namespace calculate_star_l752_752581

def star (a b : ℝ) (h : a ≠ b) : ℝ := (a + b) / (a - b)

theorem calculate_star : ∀ (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hac : c ≠ (star a b hab)),
  star (star a b hab) c hac = 1 / 5 :=
by
  sorry

-- Specific case for given values
example : ∃ (h₁ : 4 ≠ 5) (h₂ : -9 ≠ 6), star (star 4 5 h₁) 6 h₂ = 1 / 5 := 
by
  use by norm_num, by norm_num 
  exact calculate_star 4 5 6 by norm_num by norm_num by norm_num

end calculate_star_l752_752581


namespace no_real_solutions_iff_range_of_k_l752_752224

theorem no_real_solutions_iff_range_of_k (k : ℝ) :
  (∀ x : ℝ, k * x ^ 2 + sqrt 2 * k * x + 2 < 0) ↔ k ∈ Icc 0 4 :=
sorry

end no_real_solutions_iff_range_of_k_l752_752224


namespace charles_paints_l752_752510

-- Define the ratio and total work conditions
def ratio_a_to_c (a c : ℕ) := a * 6 = c * 2

def total_work (total : ℕ) := total = 320

-- Define the question, i.e., the amount of work Charles does
theorem charles_paints (a c total : ℕ) (h_ratio : ratio_a_to_c a c) (h_total : total_work total) : 
  (total / (a + c)) * c = 240 :=
by 
  -- We include sorry to indicate the need for proof here
  sorry

end charles_paints_l752_752510


namespace ratio_square_outside_circle_to_total_area_l752_752834

noncomputable theory

open_locale classical

-- Definitions for areas
def area_circle : ℝ := 2 * x
def area_square : ℝ := 2 * area_circle

def overlapping_area : ℝ := 0.5 * area_circle

-- Total area of the figure (union of circle and square)
def total_area : ℝ := area_circle + area_square - overlapping_area

-- Area of square outside the circle
def area_square_outside_circle : ℝ := area_square - overlapping_area

-- Prove that the ratio of the area of the square outside the circle to the total area is 3/5
theorem ratio_square_outside_circle_to_total_area (x : ℝ) : 
  (area_square_outside_circle / total_area) = (3 / 5) :=
by sorry

end ratio_square_outside_circle_to_total_area_l752_752834


namespace find_y_l752_752766

-- Define the problem conditions
def avg_condition (y : ℝ) : Prop := (15 + 25 + y) / 3 = 23

-- Prove that the value of 'y' satisfying the condition is 29
theorem find_y (y : ℝ) (h : avg_condition y) : y = 29 :=
sorry

end find_y_l752_752766


namespace autumn_pencils_l752_752528

-- Define the conditions of the problem.
def initial_pencils := 20
def misplaced_pencils := 7
def broken_pencils := 3
def found_pencils := 4
def bought_pencils := 2

-- Define the number of pencils lost and gained.
def pencils_lost := misplaced_pencils + broken_pencils
def pencils_gained := found_pencils + bought_pencils

-- Define the final number of pencils.
def final_pencils := initial_pencils - pencils_lost + pencils_gained

-- The theorem we want to prove.
theorem autumn_pencils : final_pencils = 16 := by
  sorry

end autumn_pencils_l752_752528


namespace polar_coordinates_of_point_l752_752547

theorem polar_coordinates_of_point :
  ∀ (x y : ℝ), x = -2 → y = sqrt 2 → 
  (∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = sqrt (x^2 + y^2) ∧ θ = 3 * π / 4) :=
begin
  intros x y hx hy,
  use [sqrt (x^2 + y^2), 3 * π / 4],
  split, 
  { show sqrt (x^2 + y^2) > 0, 
    rw [hx, hy],
    norm_num,
    norm_num },
  split,
  { show 0 ≤ 3 * π / 4, 
    norm_num },
  split,
  { show 3 * π / 4 < 2 * π, 
    norm_num },
  split,
  { show sqrt (x^2 + y^2) = sqrt 6,
    rw [hx, hy],
    norm_num },
  show 3 * π / 4 = 3 * π / 4,
  refl
end

end polar_coordinates_of_point_l752_752547


namespace circle_equation_from_diameter_l752_752292

theorem circle_equation_from_diameter (A B : ℝ × ℝ)
  (hA : A = (-1, 3)) (hB : B = (5, -5)) :
  ∃ (C : ℝ × ℝ) (r : ℝ), (C = ((2 : ℝ), (-1 : ℝ))) ∧ (r = 5) ∧
    (∀ (x y : ℝ), ((x - 2) ^ 2 + (y + 1) ^ 2 = 5 ^ 2) ↔ (x^2 + y^2 - 4x + 2y - 20 = 0)) :=
by
  sorry

end circle_equation_from_diameter_l752_752292


namespace equation_infinitely_many_solutions_iff_b_eq_neg9_l752_752919

theorem equation_infinitely_many_solutions_iff_b_eq_neg9 (b : ℤ) :
  (∀ x : ℤ, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  sorry

end equation_infinitely_many_solutions_iff_b_eq_neg9_l752_752919


namespace black_area_fraction_after_three_changes_l752_752154

theorem black_area_fraction_after_three_changes
  (initial_black_area : ℚ)
  (change_factor : ℚ)
  (h1 : initial_black_area = 1)
  (h2 : change_factor = 2 / 3)
  : (change_factor ^ 3) * initial_black_area = 8 / 27 := 
by
  sorry

end black_area_fraction_after_three_changes_l752_752154


namespace exists_composite_number_triplet_replacement_exists_1997_digit_composite_number_triplet_replacement_l752_752468

-- Proof Problem 1: There exists a natural number that remains composite regardless of any triplet replacement.
theorem exists_composite_number_triplet_replacement : 
  ∃ N : ℕ, ∀ (f : ℕ → ℕ) (i : ℕ), (N' = replace_triplet N f i) → ¬ is_prime N' :=
sorry

-- Proof Problem 2: There exists a 1997-digit natural number that remains composite regardless of any triplet replacement.
theorem exists_1997_digit_composite_number_triplet_replacement : 
  ∃ N : ℕ, (N_digits N = 1997) ∧ ∀ (f : ℕ → ℕ) (i : ℕ), (N' = replace_triplet N f i) → ¬ is_prime N' :=
sorry

end exists_composite_number_triplet_replacement_exists_1997_digit_composite_number_triplet_replacement_l752_752468


namespace part1_part2_l752_752595

noncomputable def f (x a : ℝ) : ℝ := -x^2 + (a-1)*x + a

theorem part1 (a : ℝ) :
  (∀ x ∈ set.Icc (-1:ℝ) 2, has_deriv_at (λ x, f x a) (-2*x + (a-1)) x)
  → ((a ≥ 5) ∨ (a ≤ -1)) :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, f x a ≤ 0) ↔
  if a > -1 then ∀ x, (x ≤ -1 ∨ x ≥ a)
  else if a = -1 then ∀ x, true
  else if a < -1 then ∀ x, (x ≤ a ∨ x ≥ -1) :=
sorry

end part1_part2_l752_752595


namespace erased_number_sum_l752_752404

theorem erased_number_sum
    (x : ℕ) (y : ℕ)
    (h1 : 0 ≤ y) (h2 : y ≤ 9)
    (h3 : 10 * x + 45 - (x + y) = 2002) :
    {218, 219, 220, 221, 222, 224, 225, 226, 227} = {x, x+1, x+2, x+3, x+4, x+6, x+7, x+8, x+9} :=
by
  sorry

end erased_number_sum_l752_752404


namespace triangle_area_is_correct_l752_752050

-- Define the given conditions
def is_isosceles_right_triangle (h : ℝ) (l : ℝ) : Prop :=
  h = l * sqrt 2

def triangle_hypotenuse := 6 * sqrt 2  -- Given hypotenuse

-- Prove that the area of the triangle is 18 square units
theorem triangle_area_is_correct : 
  ∃ (l : ℝ), is_isosceles_right_triangle triangle_hypotenuse l ∧ (1/2) * l^2 = 18 :=
by
  sorry

end triangle_area_is_correct_l752_752050


namespace cos_C_of_triangle_l752_752666

theorem cos_C_of_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hc : c = 4)
  (h_sine_relation : 3 * Real.sin A = 2 * Real.sin B)
  (h_cosine_law : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  Real.cos C = -1/4 :=
by
  sorry

end cos_C_of_triangle_l752_752666


namespace geometric_sequence_general_formula_maximum_value_m_l752_752232

theorem geometric_sequence_general_formula 
  (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 = 1)
  (hq : q > 0)
  (h_sum : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q))
  (h_arith_seq : ∀ S1 S2 S3 a1 a2 a3, 2 * (S3 + a3) = (S1 + a1) + (S2 + a2))
  (ha1 : ∀ n, a n = a 1 * q ^ (n - 1)) :
  ∀ n, a n = (1 / 2) ^ (n - 1) := 
sorry

theorem maximum_value_m 
  (a b : ℕ → ℝ) (T : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = (1 / 2) ^ (a n * b n))
  (hT : ∀ n, T n = ∑ k in finset.range n, b k)
  (hT_m : ∀ n, T n ≥ 1) :
  ∀ m, m ≤ 1 :=
sorry

end geometric_sequence_general_formula_maximum_value_m_l752_752232


namespace necessary_but_not_sufficient_l752_752979

noncomputable def isEllipseWithFociX (a b : ℝ) : Prop :=
  ∃ (C : ℝ → ℝ → Prop), (∀ (x y : ℝ), C x y ↔ (x^2 / a + y^2 / b = 1)) ∧ (a > b ∧ a > 0 ∧ b > 0)

theorem necessary_but_not_sufficient (a b : ℝ) :
  (∀ (C : ℝ → ℝ → Prop), (∀ x y : ℝ, C x y ↔ (x^2 / a + y^2 / b = 1))
    → ((a > b ∧ a > 0 ∧ b > 0) → isEllipseWithFociX a b))
  ∧ ¬ (a > b → ∀ (C : ℝ → ℝ → Prop), (∀ x y : ℝ, C x y ↔ (x^2 / a + y^2 / b = 1)) → isEllipseWithFociX a b) :=
sorry

end necessary_but_not_sufficient_l752_752979


namespace tetrahedron_surface_area_is_4sqrt3_l752_752616

-- Defining the conditions 
def edge_length (a : ℝ) : Prop := a = 2

def equilateral_triangle (A B C : ℝ × ℝ × ℝ) : Prop := 
  (dist A B = dist B C) ∧ (dist B C = dist A C)

-- Defining an equilateral face 
noncomputable def face_area (a : ℝ) : ℝ := 
  (sqrt 3 / 4) * a^2

-- Defining the total surface area of the tetrahedron
noncomputable def tetrahedron_surface_area (a : ℝ) : ℝ := 
  4 * face_area a

-- The theorem to be proved
theorem tetrahedron_surface_area_is_4sqrt3 (a : ℝ) (hf : edge_length a) : 
  tetrahedron_surface_area a = 4 * sqrt 3 :=
by
  unfold edge_length at hf
  unfold tetrahedron_surface_area face_area
  rw hf
  ring
  sorry  -- Placeholder for completing the detailed proof

end tetrahedron_surface_area_is_4sqrt3_l752_752616


namespace flag_height_l752_752778

-- Define the problem conditions
variable (h : ℝ) -- height of the flag
variable (area_shaded : ℝ) -- total area of four shaded regions

-- State the theorem
theorem flag_height (h_pos : 0 < h) (h_length : 2 * h) (total_shaded : 4 * 2 * h^2 / 7 = 1400) :
  h = 35 := by
  sorry

end flag_height_l752_752778


namespace negative_reciprocal_opposite_of_negative_abs_neg_three_l752_752415

def abs_val (x : ℤ) : ℤ := if x < 0 then -x else x

def opposite_number (x : ℤ) : ℤ := -x

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem negative_reciprocal_opposite_of_negative_abs_neg_three : 
  reciprocal (opposite_number (abs_val (-3))) = -1 / 3 :=
by
  sorry

end negative_reciprocal_opposite_of_negative_abs_neg_three_l752_752415


namespace notebook_and_pencil_cost_l752_752082

theorem notebook_and_pencil_cost :
  ∃ (x y : ℝ), 6 * x + 4 * y = 9.2 ∧ 3 * x + y = 3.8 ∧ x + y = 1.8 :=
by
  sorry

end notebook_and_pencil_cost_l752_752082


namespace multiply_121_54_l752_752577

theorem multiply_121_54 : ∃ x : ℕ, 121 * 54 = x ∧ x = 6534 := by
  use 6534
  constructor
  · exact rfl
  · sorry

end multiply_121_54_l752_752577


namespace midpoint_of_RQ_is_diagonal_intersection_l752_752710

variable {P Q R A B C D : Point}
variable (intersects_PAB_PCD : ∃ Q : Point, Q ≠ P ∧ circleThrough P A B ∩ circleThrough P C D = {P, Q})
variable (intersects_PAD_PBC : ∃ R : Point, R ≠ P ∧ circleThrough P A D ∩ circleThrough P B C = {P, R})

-- Assuming a definition of the intersection of diagonals for an arbitrary quadrilateral
def intersection_of_diagonals (A B C D : Point) : Point := sorry

-- Main theorem statement
theorem midpoint_of_RQ_is_diagonal_intersection (h1 : ∃ Q : Point, intersects_PAB_PCD Q)
  (h2 : ∃ R : Point, intersects_PAD_PBC R) :
  is_midpoint (midpoint R Q) (intersection_of_diagonals A B C D) :=
sorry

end midpoint_of_RQ_is_diagonal_intersection_l752_752710


namespace count_5_digit_palindromes_l752_752997

def is_digit (x : ℕ) : Prop := 0 ≤ x ∧ x ≤ 9

def is_a_digit (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

def is_palindrome_5_digit (a b c : ℕ) : Prop := 
  is_a_digit a ∧ is_digit b ∧ is_digit c

theorem count_5_digit_palindromes : 
  (∃ (a b c : ℕ), is_palindrome_5_digit a b c) →
  (finset.univ.filter (λ t : ℕ, ∃ (a b c : ℕ), (is_palindrome_5_digit a b c ∧ t = 10001 * a + 1010 * b + 100 * c)) ).card = 900 :=
by sorry

end count_5_digit_palindromes_l752_752997


namespace radius_of_each_sphere_l752_752792

-- Define the side length of the cube
def cube_side_length : ℝ := 3

-- Define the radius of each sphere
noncomputable def sphere_radius (r : ℝ) : Prop :=
  let center_distance := 3 / 2 - r in
  sqrt 3 * center_distance = sqrt 3

-- State the theorem
theorem radius_of_each_sphere : ∃ r : ℝ, sphere_radius r ∧ r = 1 / 4 :=
by
  use 1 / 4
  sorry

end radius_of_each_sphere_l752_752792


namespace probability_sum_equals_robert_age_l752_752012

def Coin := {15, 25}  -- The outcomes of the coin flip

noncomputable def Dice := Finset.range 1 7  -- The outcomes of a standard 6-sided die

def isFair (s : Finset ℕ) : Prop := ∀ x ∈ s, 1 / (s.card : ℝ) = 1 / (card Finset.range 1 7 : ℝ)

def robertAge := 16  -- Robert's age

theorem probability_sum_equals_robert_age :
  (1 / 2 : ℝ) * (1 / 6 : ℝ) = 1 / 12 :=
by
  sorry

end probability_sum_equals_robert_age_l752_752012


namespace base_length_of_parallelogram_l752_752069

theorem base_length_of_parallelogram (A h : ℝ) (hA : A = 44) (hh : h = 11) :
  ∃ b : ℝ, b = 4 ∧ A = b * h :=
by
  sorry

end base_length_of_parallelogram_l752_752069


namespace deductive_reasoning_l752_752304

theorem deductive_reasoning (
  deductive_reasoning_form : Prop
): ¬(deductive_reasoning_form → true → correct_conclusion) :=
by sorry

end deductive_reasoning_l752_752304


namespace jar_of_sauce_costs_2_l752_752744

noncomputable def cost_of_pasta : ℝ := 1.0
noncomputable def cost_of_meatballs : ℝ := 5.0
noncomputable def total_servings : ℝ := 8.0
noncomputable def cost_per_serving : ℝ := 1.0

theorem jar_of_sauce_costs_2 :
  let total_cost := total_servings * cost_per_serving,
      known_costs := cost_of_pasta + cost_of_meatballs
  in total_cost - known_costs = 2.0 :=
by
  -- Proof to be filled in
  sorry

end jar_of_sauce_costs_2_l752_752744


namespace necklaces_made_l752_752924

theorem necklaces_made (total_beads : ℕ) (beads_per_necklace : ℕ) (h1 : total_beads = 18) (h2 : beads_per_necklace = 3) : total_beads / beads_per_necklace = 6 := 
by {
  sorry
}

end necklaces_made_l752_752924


namespace positive_number_set_correct_negative_fraction_set_correct_rational_number_set_correct_l752_752198

def is_positive (x : ℝ) : Prop := x > 0
def is_negative_fraction (x : ℝ) : Prop := x < 0 ∧ ∃ p q: ℤ, q ≠ 0 ∧ x = p / q
def is_rational (x : ℝ) : Prop := ∃ p q: ℤ, q ≠ 0 ∧ x = p / q

def numbers := 
  [(-9.3 : ℝ), (3 / 100 : ℝ), (-20 : ℝ), (0 : ℝ), (0.01 : ℝ), (-1 : ℝ), (-7 / 2 : ℝ), (3.14 : ℝ), (3.3 : ℝ), (Real.pi)]

def positive_numbers := [(3 / 100 : ℝ), (0.01 : ℝ), (3.14 : ℝ), (3.3 : ℝ), (Real.pi)]
def negative_fraction_numbers := [(-7 / 2 : ℝ)]
def rational_numbers := 
  [(-9.3 : ℝ), (3 / 100 : ℝ), (-20 : ℝ), (0 : ℝ), (0.01 : ℝ), (-1 : ℝ), (-7 / 2 : ℝ), (3.14 : ℝ), (3.3 : ℝ)]

theorem positive_number_set_correct : 
  {x | x ∈ numbers ∧ is_positive x} = set.from_list positive_numbers :=
by sorry

theorem negative_fraction_set_correct : 
  {x | x ∈ numbers ∧ is_negative_fraction x} = set.from_list negative_fraction_numbers :=
by sorry

theorem rational_number_set_correct : 
  {x | x ∈ numbers ∧ is_rational x} = set.from_list rational_numbers :=
by sorry

end positive_number_set_correct_negative_fraction_set_correct_rational_number_set_correct_l752_752198


namespace problem_solution_l752_752943

theorem problem_solution (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = a * x^2 + (b - 3) * x + 3) →
  (∀ x : ℝ, f x = f (-x)) →
  (a^2 - 2 = -a) →
  a + b = 4 :=
by
  intros h1 h2 h3
  sorry

end problem_solution_l752_752943


namespace average_service_proof_l752_752671

variable (k : ℕ) (ha : 8) (hb : 6)
variable (a_ratio : 7) (b_ratio : 5)

-- Define the total number of employees as ratios 7k for department A and 5k for department B
def total_employees := a_ratio * k + b_ratio * k

-- Define the total years of service for both departments
def total_years_of_service := ha * (a_ratio * k) + hb * (b_ratio * k)

-- Define the average years of service in the company
def average_years_of_service := (total_years_of_service / total_employees : ℚ)

theorem average_service_proof :
  average_years_of_service k ha hb a_ratio b_ratio = 7 + 1/6 :=
by sorry

end average_service_proof_l752_752671


namespace problem_trajectory_of_moving_circle_problem_slopes_condition_l752_752948

open Real

theorem problem_trajectory_of_moving_circle (C₁ : ℝ → ℝ → Prop)
  (l : ℝ → Prop) (trajectory : ℝ → ℝ → Prop) :
  (∀ (x y : ℝ), C₁ x y ↔ (x - 2) ^ 2 + y ^ 2 = 1) ∧
  (∀ (x : ℝ), l x ↔ x = -1) ∧
  (∀ (x y : ℝ), trajectory x y ↔ y ^ 2 = 8 * x)
  → (∀ (x y : ℝ), y ^ 2 = 8 * x ↔ (trajectory x y)) := sorry

theorem problem_slopes_condition (M P : ℝ × ℝ) (A B : ℝ × ℝ) :
  (P = (1, 0)) →
  (∀ t, M = (-1, t) →
    ∀ m : ℝ, (A.1 = m * A.2 + 1) ∧ (B.1 = m * B.2 + 1) ∧
    (A.2 ^ 2 = 8 * A.1) ∧ (B.2 ^ 2 = 8 * B.1) →
    (let k_MP := (P.2 - M.2) / (P.1 - M.1),
         k_MA := (A.2 - M.2) / (A.1 + 1),
         k_MB := (B.2 - M.2) / (B.1 + 1)
     in 2 * k_MP = k_MA + k_MB)) := sorry

end problem_trajectory_of_moving_circle_problem_slopes_condition_l752_752948


namespace trigonometric_identity_l752_752895

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - 2 / Real.sin (70 * Real.pi / 180)) = 4 * Real.cot (40 * Real.pi / 180) :=
by
  -- The proof will be skipped with sorry
  sorry

end trigonometric_identity_l752_752895


namespace find_unknown_rate_l752_752851

def total_cost (n1 n2 : ℕ) (p1 p2 : ℝ) (r : ℝ) := n1 * p1 + n2 * p2 + 2 * r

def average_price (total_cost : ℝ) (total_number : ℕ) := total_cost / total_number

theorem find_unknown_rate (r : ℝ) 
  (h1 : total_cost 3 6 100 150 r = 1650) 
  (h2 : average_price 1650 11 = 150) : r = 225 :=
by
  sorry

end find_unknown_rate_l752_752851


namespace determine_parabola_equation_l752_752833

-- Given conditions
variable (p : ℝ) (h_p : p > 0)
variable (x1 x2 : ℝ)
variable (AF BF : ℝ)
variable (h_AF : AF = x1 + p / 2)
variable (h_BF : BF = x2 + p / 2)
variable (h_AF_value : AF = 2)
variable (h_BF_value : BF = 3)

-- Prove the equation of the parabola
theorem determine_parabola_equation (h1 : x1 + x2 = 5 - p)
(h2 : x1 * x2 = p^2 / 4)
(h3 : AF * BF = 6) :
  y^2 = (24/5 : ℝ) * x := 
sorry

end determine_parabola_equation_l752_752833


namespace peter_final_erasers_l752_752382

variable (P : ℕ) (B : ℕ)

def peter_initial_erasers := P = 8
def bridget_additional_erasers := B = 3

theorem peter_final_erasers : P + B = 11 :=
by
  -- Assuming the initial conditions
  have h1 : P = 8 := peter_initial_erasers
  have h2 : B = 3 := bridget_additional_erasers
  sorry

end peter_final_erasers_l752_752382


namespace find_m_l752_752290

variable {m : ℝ}

def vector_a : ℝ × ℝ × ℝ := (1, -1, 0)
def vector_b : ℝ × ℝ × ℝ := (-1, 2, 1)
def vector_c : ℝ × ℝ × ℝ := (2, 1, m)

def coplanar_condition (λ μ : ℝ) : Prop := 
  vector_c = (λ * vector_a.1 + μ * vector_b.1, λ * vector_a.2 + μ * vector_b.2, λ * vector_a.3 + μ * vector_b.3)

theorem find_m (h : ∃ (λ μ : ℝ), coplanar_condition λ μ) : m = 3 := 
sorry

end find_m_l752_752290


namespace bob_age_is_eleven_l752_752402

/-- 
Susan, Arthur, Tom, and Bob are siblings. Arthur is 2 years older than Susan, 
Tom is 3 years younger than Bob. Susan is 15 years old, 
and the total age of all four family members is 51 years. 
This theorem states that Bob is 11 years old.
-/

theorem bob_age_is_eleven
  (S A T B : ℕ)
  (h1 : A = S + 2)
  (h2 : T = B - 3)
  (h3 : S = 15)
  (h4 : S + A + T + B = 51) : 
  B = 11 :=
  sorry

end bob_age_is_eleven_l752_752402


namespace lanterns_at_top_of_tower_l752_752874

theorem lanterns_at_top_of_tower :
  ∀ (a_1 : ℕ), (∃ (n : ℕ) (a : ℕ → ℕ), n = 7 ∧ a 1 = a_1 ∧ (∀ i, i ≤ n → a (i+1) = 2 * a i) ∧ (finset.range n).sum (λ i, a (i+1)) = 381) → a_1 = 3 :=
by
  intros a_1 h
  obtain ⟨n, a, hn, ha1, hrec, hsum⟩ := h
  sorry

end lanterns_at_top_of_tower_l752_752874


namespace correct_answer_l752_752286

-- Given conditions:
variables {f : ℝ → ℝ}

-- Condition 1: Symmetry about x = -2 for f(x+2), which implies f is even.
def is_symmetric_about_neg2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x + 2) = f(-x + 2)

-- Condition 2: Monotonically decreasing on the interval [0, 3].
def is_monotonic_decreasing_on_0_3 (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x < y → y ≤ 3 → f x > f y

-- The proof problem: Proving the correct answer given the conditions.
theorem correct_answer (h1 : is_symmetric_about_neg2 f) (h2 : is_monotonic_decreasing_on_0_3 f) :
  f (-1) > f 2 ∧ f 2 > f 3 :=
by sorry

end correct_answer_l752_752286


namespace problem_equivalent_lean_l752_752629

-- Define the conditions
def f (x m : ℝ) := sqrt (abs (x + 1) + abs (x - 3) - m)
def g (a b : ℝ) := 7 * a + 4 * b

-- Main theorem stating the problem
theorem problem_equivalent_lean (a b m : ℝ) (h₁ : ∀ x : ℝ, abs (x + 1) + abs (x - 3) - m ≥ 0) (h₂ : (2 / (3 * a + b)) + (1 / (a + 2 * b)) = 4) :
  m ≤ 4 ∧ g a b = 9 / 4 :=
begin
  sorry,
end

end problem_equivalent_lean_l752_752629


namespace degree_of_g_l752_752705

open Polynomial

theorem degree_of_g (f g : Polynomial ℂ) (h1 : f = -3 * X^5 + 4 * X^4 - X^2 + C 2) (h2 : degree (f + g) = 2) : degree g = 5 :=
sorry

end degree_of_g_l752_752705


namespace tangent_line_eqn_and_decreasing_interval_range_of_a_l752_752631

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

theorem tangent_line_eqn_and_decreasing_interval (a : ℝ) (h : a = 1) :
  let f_a (x : ℝ) := f x a in
  ∀ x, (x = 1 → f_a x = 0) ∧
  (0 < x ∧ x < 1 → (f_a x).derivative < 0) :=
sorry

theorem range_of_a (a : ℝ) (h : a > 0)
  (h_increasing : ∀ x ≥ 2, (f x a).derivative ≥ 0) :
  a ≥ 1 / 2 :=
sorry

end tangent_line_eqn_and_decreasing_interval_range_of_a_l752_752631


namespace cot_difference_in_triangle_l752_752688

noncomputable def cot (θ : ℝ) := 1 / Real.tan θ

theorem cot_difference_in_triangle 
  (A B C D : Type) [IsTriangle A B C]
  (hAD_median : IsMedian A D B C)
  (hAD_angle : ∠ (D A) (A C) = π / 6):
  abs (cot (angle B A C) - cot (angle C A B)) = 2 :=
by
  sorry

end cot_difference_in_triangle_l752_752688


namespace distinct_configurations_l752_752177

/-- 
Define m, n, and the binomial coefficient function.
conditions:
  - integer grid dimensions m and n with m >= 1, n >= 1.
  - initially (m-1)(n-1) coins in the subgrid of size (m-1) x (n-1).
  - legal move conditions for coins.
question:
  - Prove the number of distinct configurations of coins equals the binomial coefficient.
-/
def number_of_distinct_configurations (m n : ℕ) : ℕ :=
  Nat.choose (m + n - 2) (m - 1)

theorem distinct_configurations (m n : ℕ) (h_m : 1 ≤ m) (h_n : 1 ≤ n) :
  number_of_distinct_configurations m n = Nat.choose (m + n - 2) (m - 1) :=
sorry

end distinct_configurations_l752_752177


namespace sum_of_cubes_l752_752104

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l752_752104


namespace subset_A_B_l752_752371

theorem subset_A_B (a : ℝ) (A : Set ℝ := {0, -a}) (B : Set ℝ := {1, a-2, 2a-2}) 
  (h : A ⊆ B) : a = 1 := by
  sorry

end subset_A_B_l752_752371


namespace difference_between_largest_and_smallest_l752_752805

def largest_number (digits : List ℕ) : ℕ :=
  digits.foldl (λ n d => 10 * n + d) 0

def smallest_number (digits : List ℕ) : ℕ :=
  digits.foldr (λ d n => 10 * n + d) 0

theorem difference_between_largest_and_smallest :
  let digits := [0, 3, 4, 8]
  ∃ (largest smallest : ℕ),
    largest = largest_number (List.sort (λ x y => x > y) digits) ∧
    smallest = smallest_number (List.sort (λ x y => x < y) (List.filter (≠ 0) digits) ++ [0]) ∧
    (largest - smallest) = 5382 :=
by
  sorry

end difference_between_largest_and_smallest_l752_752805


namespace ratio_AD_EC_l752_752314

variable (A B C D E : Type)
variable [Triangle A B C]
variable [Line A C] [Line B D] [Line B E]
variable [Meet B D A C D] [Meet B E A C E]

-- Definitions specifying that BD segments angle ABC in a 1:2 ratio and BE bisects angle DBC
def segments (BD : Line A C) (BE : Line A C) :=
  SegmentAngle ABC ABD (AngleRatio.mk 1 2) ∧
  Bisect DBC BE

-- The goal to prove given the above conditions
theorem ratio_AD_EC (h1 : segments BD BE) :
  ∃ r : Ratio, r = Ratio.mk AD EC ∧ r = Ratio.mk (AB * (BC - BD)) (2 * BD * BC) := 
by
  sorry

end ratio_AD_EC_l752_752314


namespace triangle_area_is_correct_l752_752052

-- Define the given conditions
def is_isosceles_right_triangle (h : ℝ) (l : ℝ) : Prop :=
  h = l * sqrt 2

def triangle_hypotenuse := 6 * sqrt 2  -- Given hypotenuse

-- Prove that the area of the triangle is 18 square units
theorem triangle_area_is_correct : 
  ∃ (l : ℝ), is_isosceles_right_triangle triangle_hypotenuse l ∧ (1/2) * l^2 = 18 :=
by
  sorry

end triangle_area_is_correct_l752_752052


namespace ethanol_in_tank_l752_752875

theorem ethanol_in_tank (capacity fuel_a fuel_b : ℝ)
  (ethanol_a ethanol_b : ℝ)
  (h1 : capacity = 218)
  (h2 : fuel_a = 122)
  (h3 : fuel_b = capacity - fuel_a)
  (h4 : ethanol_a = 0.12)
  (h5 : ethanol_b = 0.16) :
  fuel_a * ethanol_a + fuel_b * ethanol_b = 30 := 
by {
  sorry
}

end ethanol_in_tank_l752_752875


namespace polynomial_root_product_bound_l752_752420

theorem polynomial_root_product_bound
  {n : ℕ} (a : Fin n → ℝ) 
  (α : Fin n → ℝ)
  (hα : ∀ i, α i ∈ Ioo 0 1)
  (hP1 : ∏ i, (1 - α i) = |∏ i, α i|) : 
  ∏ i, α i ≤ 1 / 2^n :=
sorry

end polynomial_root_product_bound_l752_752420


namespace percentage_non_silver_new_shipment_l752_752480

def total_cars_initial : ℕ := 40
def percentage_silver_initial : ℝ := 20 / 100
def total_cars_new_shipment : ℕ := 80
def percentage_silver_total : ℝ := 40 / 100

theorem percentage_non_silver_new_shipment :
  let silver_initial := percentage_silver_initial * total_cars_initial
  let total_cars := total_cars_initial + total_cars_new_shipment
  let silver_total := percentage_silver_total * total_cars
  let silver_new_shipment := silver_total - silver_initial
  let non_silver_new_shipment := total_cars_new_shipment - silver_new_shipment
  (non_silver_new_shipment / total_cars_new_shipment) * 100 = 50 := 
by
  sorry

end percentage_non_silver_new_shipment_l752_752480


namespace wheel_distance_l752_752864

theorem wheel_distance (r : ℝ) (revolutions : ℕ) (h_r : r = 2) (h_revolutions : revolutions = 3) : 
  let C := 2 * real.pi * r
  in revolutions * C = 12 * real.pi :=
by
  have h_C : C = 2 * real.pi * r := rfl
  rw [h_r, h_revolutions, h_C]
  sorry

end wheel_distance_l752_752864


namespace total_cost_proof_l752_752078

def uber_cost : ℤ := 22
def lyft_cost : ℤ := uber_cost - 3
def taxi_cost : ℤ := lyft_cost - 4
def tip : ℤ := (taxi_cost * 20) / 100
def total_cost : ℤ := taxi_cost + tip

theorem total_cost_proof :
  total_cost = 18 :=
by
  sorry

end total_cost_proof_l752_752078


namespace centroid_plane_distance_l752_752339

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end centroid_plane_distance_l752_752339


namespace coin_toss_probability_l752_752128

theorem coin_toss_probability :
  (∀ (coin_toss : ℕ → Bool), (∀ n, coin_toss n = tt ∨ coin_toss n = ff) ∧ 
  (∀ n, probability (coin_toss n = tt) = 1/2) → 
  probability (∀ n < 5, coin_toss n = coin_toss 0) = 1/32) :=
sorry

end coin_toss_probability_l752_752128


namespace range_of_x_l752_752249

noncomputable def f (x : ℝ) : ℝ := sorry

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f(x) = f(-x)

def is_monotone_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x ≤ y → f(x) ≤ f(y)

theorem range_of_x (f : ℝ → ℝ)
  (even_f : is_even f)
  (monotone_f : is_monotone_increasing f {x | 0 ≤ x}) :
  {x : ℝ | f(2 * x - 1) < f(1 / 3)} = Ioo (1 / 3 : ℝ) (2 / 3) := 
sorry

end range_of_x_l752_752249


namespace parallelogram_sides_l752_752315

theorem parallelogram_sides:
  ∀ (A B C K L M: Point)
  (AB BC: ℝ)
  (BK LM BM KL: ℝ)
  (hAB: AB = 18)
  (hBC: BC = 12)
  (hAreaRatio: parallelogram_area BKLM = (4 / 9) * triangle_area ABC)
  , (BM = KL) → (BK = LM) 
  , BM * BK = ((4 : ℝ) / 9) * (triangle_area ABC)
  , ((BK = 6 ∧ KL = 8) ∨ (BK = 12 ∧ KL = 4)) :=
begin
  sorry
end

-- Definitions of triangle_area and parallelogram_area should be filled in the actual proof context

end parallelogram_sides_l752_752315


namespace bug_position_after_2012_jumps_l752_752730

def circle_points := {1, 2, 3, 4, 5}

def next_point (pos : ℕ) : ℕ :=
  if pos % 2 = 0 then (pos + 3) % 5
  else (pos + 2) % 5

def point_after_jumps (start : ℕ) (jumps : ℕ) : ℕ :=
  Nat.iterate next_point jumps start

theorem bug_position_after_2012_jumps : point_after_jumps 5 2012 = 2 := sorry

end bug_position_after_2012_jumps_l752_752730


namespace base3_composite_numbers_l752_752451

theorem base3_composite_numbers:
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 12002110 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 2210121012 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 121212 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 102102 = a * b) ∧ 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 1001 * AB = a * b) :=
by {
  sorry
}

end base3_composite_numbers_l752_752451


namespace term_addition_k_to_kplus1_l752_752448

theorem term_addition_k_to_kplus1 (k : ℕ) : 
  (2 * k + 2) + (2 * k + 3) = 4 * k + 5 := 
sorry

end term_addition_k_to_kplus1_l752_752448


namespace last_three_digits_of_power_l752_752806

theorem last_three_digits_of_power (h : 7^500 ≡ 1 [MOD 1250]) : 7^10000 ≡ 1 [MOD 1250] :=
by
  sorry

end last_three_digits_of_power_l752_752806


namespace cos2theta_collinear_l752_752994

variables (θ : ℝ) 

def AB : ℝ × ℝ := (-1, -3)
def BC : ℝ × ℝ := (2 * Real.sin θ, 2)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

theorem cos2theta_collinear :
  collinear AB BC →
  Real.cos (2 * θ) = 7 / 9 :=
by
  intro h
  sorry

end cos2theta_collinear_l752_752994


namespace globe_division_l752_752844

theorem globe_division (parallels meridians : ℕ)
  (h_parallels : parallels = 17)
  (h_meridians : meridians = 24) :
  let slices_per_sector := parallels + 1
  let sectors := meridians
  let total_parts := slices_per_sector * sectors
  total_parts = 432 := by
  sorry

end globe_division_l752_752844


namespace find_x_l752_752423

theorem find_x :
  let a := 0.15
  let b := 0.06
  let c := 0.003375
  let d := 0.000216
  let e := 0.0225
  let f := 0.0036
  let g := 0.08999999999999998
  ∃ x, c - (d / e) + x + f = g →
  x = 0.092625 :=
by
  sorry

end find_x_l752_752423


namespace magnitude_of_b_l752_752274

variables (a b : EuclideanSpace ℝ (Fin 3)) (θ : ℝ)

-- Defining the conditions
def vector_a_magnitude : Prop := ‖a‖ = 1
def vector_angle_condition : Prop := θ = Real.pi / 3
def linear_combination_magnitude : Prop := ‖2 • a - b‖ = 2 * Real.sqrt 3
def b_magnitude : Prop := ‖b‖ = 4

-- The statement we want to prove
theorem magnitude_of_b (h1 : vector_a_magnitude a) (h2 : vector_angle_condition θ) (h3 : linear_combination_magnitude a b) : b_magnitude b :=
sorry

end magnitude_of_b_l752_752274


namespace math_club_team_selection_l752_752377

open Nat

-- Lean statement of the problem
theorem math_club_team_selection : 
  (choose 7 3) * (choose 9 3) = 2940 :=
by 
  sorry

end math_club_team_selection_l752_752377


namespace regular_polygon_sides_l752_752150

theorem regular_polygon_sides (n : ℕ) (h : ∀ n, (n > 2) → (360 / n = 20)) : n = 18 := sorry

end regular_polygon_sides_l752_752150


namespace range_of_a_l752_752628

noncomputable def f (a x : ℝ) : ℝ :=
if x > 0 then x^2 - 2 else -3 * abs (x + a) + a

theorem range_of_a :
  (∃ (a : ℝ), (∀ x > 0, f a x = x^2 - 2) ∧ 
               (∀ x < 0, f a x = -3 * abs (x + a) + a) ∧
               ∃ (x : ℝ), (x < 0 ∧ 2 - x^2 = -3 * abs (x + a) + a) ∧ 
                           (x^2 - 3 * x - 2 * a - 2 = 0 ∧ discr_eq_pos (x^2 + 3 * x + 4 * a - 2) > 0) ∧ 
                           (1 < a) ∧ (a < 17 / 16))
→ (1 < a ∧ a < 17 / 16) :=
sorry

end range_of_a_l752_752628


namespace square_perimeter_l752_752756

theorem square_perimeter (s : ℝ) (h : s^2 = s) (h₀ : s ≠ 0) : 4 * s = 4 :=
by {
  have s_eq_1 : s = 1 := by {
    field_simp [h],
    exact h₀,
    linarith,
  },
  rw s_eq_1,
  ring,
}

end square_perimeter_l752_752756


namespace periodic_even_l752_752917

noncomputable def f : ℝ → ℝ := sorry  -- We assume the existence of such a function.

variables {α β : ℝ}  -- acute angles of a right triangle

-- Function properties
theorem periodic_even (h_periodic: ∀ x: ℝ, f (x + 2) = f x)
  (h_even: ∀ x: ℝ, f (-x) = f x)
  (h_decreasing: ∀ x y: ℝ, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f x > f y)
  (h_inc_interval_0_1: ∀ x y: ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y)
  (ha: 0 < α ∧ α < π / 2)
  (hb: 0 < β ∧ β < π / 2)
  (h_sum_right_triangle: α + β = π / 2): f (Real.sin α) > f (Real.cos β) :=
sorry

end periodic_even_l752_752917


namespace transform_circle_to_ellipse_polar_to_cartesian_intersection_product_cs_inequality_range_of_m_l752_752789

-- Elective 4-2: Matrix and Transformation
noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![(2 : ℝ), (0 : ℝ); (0 : ℝ), (4 : ℝ)]

theorem transform_circle_to_ellipse : 
  ∀ x y: ℝ, (x^2 + y^2 = 1) → ((x / 2)^2 + (y / 4)^2 = 1) := by
  sorry

-- Elective 4-4: Coordinate Systems and Parametric Equations
theorem polar_to_cartesian (ρ θ : ℝ) : 
  5*ρ^2 - 3*ρ^2*(cos θ ^ 2 - sin θ ^ 2) - 8 = 0 ↔ (ρ * cos θ)^2 / 4 + (ρ * sin θ)^2 = 1 := by
  sorry

theorem intersection_product :
  ∀ t : ℝ, (1 - sqrt(3)*t)^2 / 4 + (t)^2 = 1 → let x := 1 - sqrt(3)*t, y := t in 
  |1 - x| * |0 - y| = (3 / 7) := by
  sorry

-- Elective 4-5: Inequality Seminar
variables (a b c m : ℝ)
theorem cs_inequality :
  a^2 + (1/4)*b^2 + (1/9)*c^2 >= 
  (a + b + c)^2 / 14 := by
  sorry

theorem range_of_m :
  (a + b + c + 2 - 2*m = 0) ∧ (a^2 + (1/4)*b^2 + (1/9)*c^2 + m - 1 = 0) → 
  -5/2 ≤ m ∧ m ≤ 1 := by
  sorry

end transform_circle_to_ellipse_polar_to_cartesian_intersection_product_cs_inequality_range_of_m_l752_752789


namespace sin_value_of_angle_l752_752255

variable {α : Real}

-- Given conditions: The point (-√3/2, -1/2) lies on the unit circle 
-- and is the terminal side of angle α.
def point_on_unit_circle (α : Real) : Prop :=
  ∃ (x y : Real), x = -√3 / 2 ∧ y = -1 / 2 ∧ cos α = x ∧ sin α = y

-- The statement we are going to prove
theorem sin_value_of_angle (h : point_on_unit_circle α) : sin α = -1 / 2 :=
by
  sorry

end sin_value_of_angle_l752_752255


namespace sum_reciprocal_dist_leq_l752_752713

open Real

-- Define the conditions of the problem
variables {n : ℕ}
variable (O : Fin n → ℝ × ℝ)
variables (h₁ : n ≥ 3)
variables (h₂ : ∀ i j k l : Fin n, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l →
                (O i, O j).2^2 + (O i, O k).2^2 + (O i, O l).2^2 ≠ (O j, O k).2^2 + (O j, O l).2^2 + (O k, O l).2^2)
noncomputable def dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem sum_reciprocal_dist_leq (h : ∀ l : ℝ × ℝ, ∑ i j : Fin n, i < j → (dist (O i) l = 1 ∨ dist (O j) l = 1)) :
  ∑ i in Finset.range n, ∑ j in filter (λ j, i < j) (Finset.range n), (1 / dist (O ↑i) (O ↑j)) ≤ (n - 1) * π / 4 := sorry

end sum_reciprocal_dist_leq_l752_752713


namespace compute_fraction_power_l752_752904

theorem compute_fraction_power :
  8 * (2 / 7)^4 = 128 / 2401 :=
by
  sorry

end compute_fraction_power_l752_752904


namespace Alfred_gain_percentage_l752_752158

section AlfredGain

def scooterA_purchase_price : ℝ := 4700
def scooterA_repair_cost : ℝ := 600
def scooterA_selling_price : ℝ := 5800

def scooterB_purchase_price : ℝ := 3500
def scooterB_repair_cost : ℝ := 800
def scooterB_selling_price : ℝ := 4800

def scooterC_purchase_price : ℝ := 5400
def scooterC_repair_cost : ℝ := 1000
def scooterC_selling_price : ℝ := 7000

def total_cost : ℝ := scooterA_purchase_price + scooterA_repair_cost +
                      scooterB_purchase_price + scooterB_repair_cost +
                      scooterC_purchase_price + scooterC_repair_cost

def total_selling_price : ℝ := scooterA_selling_price + scooterB_selling_price + scooterC_selling_price

def total_gain : ℝ := total_selling_price - total_cost

def gain_percentage : ℝ := (total_gain / total_cost) * 100

theorem Alfred_gain_percentage : gain_percentage = 10 :=
by
  -- Placeholder for proof
  sorry

end AlfredGain

end Alfred_gain_percentage_l752_752158


namespace vertical_coordinate_of_Q_l752_752610

theorem vertical_coordinate_of_Q (α : ℝ) (hα : 0 < α ∧ α < π/2) (hP : cos α = 1 / 3) :
  ∃ y : ℝ, (y = (2 * sqrt 2 + sqrt 3) / 6) :=
by 
  let sin_alpha := sqrt (1 - (1 / 3) ^ 2)
  let sin_alpha := 2 * sqrt 2 / 3
  let y := (1 / 2) * sin_alpha + (sqrt 3 / 2) * (1 / 3)
  use y
  sorry

end vertical_coordinate_of_Q_l752_752610


namespace smallest_four_digit_sum_27_l752_752798

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10) = sum

theorem smallest_four_digit_sum_27 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ digits_sum_to n 27 ∧ (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ digits_sum_to m 27 → n ≤ m) :=
  ∃ n, n = 1899 ∧ 1000 ≤ n ∧ n < 10000 ∧ digits_sum_to n 27 ∧ (∀ m, 1000 ≤ m ∧ m < 10000 ∧ digits_sum_to m 27 → n ≤ m)
  sorry

end smallest_four_digit_sum_27_l752_752798


namespace rational_in_set_l752_752512

theorem rational_in_set (x : ℝ) (h : x ∈ ({-Real.pi, 0, Real.sqrt 3, Real.sqrt 2} : set ℝ)) : x = 0 → ∃ p q : ℤ, q ≠ 0 ∧ (x = (p / q : ℝ)) :=
by
  intros _
  sorry

end rational_in_set_l752_752512


namespace sum_of_excluded_values_l752_752205

theorem sum_of_excluded_values (x : ℝ) :
  (3 : ℝ) := 
by 
  let roots := {1, 2}
  let sum_of_roots := 3
  exact sum_of_roots

end sum_of_excluded_values_l752_752205


namespace average_temperature_MTWT_l752_752029

theorem average_temperature_MTWT (T_TWTF : ℝ) (T_M : ℝ) (T_F : ℝ) (T_MTWT : ℝ) :
    T_TWTF = 40 →
    T_M = 42 →
    T_F = 10 →
    T_MTWT = ((4 * T_TWTF - T_F + T_M) / 4) →
    T_MTWT = 48 := 
by
  intros hT_TWTF hT_M hT_F hT_MTWT
  rw [hT_TWTF, hT_M, hT_F] at hT_MTWT
  norm_num at hT_MTWT
  exact hT_MTWT

end average_temperature_MTWT_l752_752029


namespace distance_from_B_to_center_is_74_l752_752485

noncomputable def circle_radius := 10
noncomputable def B_distance (a b : ℝ) := a^2 + b^2

theorem distance_from_B_to_center_is_74 
  (a b : ℝ)
  (hA : a^2 + (b + 6)^2 = 100)
  (hC : (a + 4)^2 + b^2 = 100) :
  B_distance a b = 74 :=
sorry

end distance_from_B_to_center_is_74_l752_752485


namespace leo_weight_proof_l752_752284

def Leo_s_current_weight (L K : ℝ) := 
  L + 10 = 1.5 * K ∧ L + K = 170 → L = 98

theorem leo_weight_proof : ∀ (L K : ℝ), L + 10 = 1.5 * K ∧ L + K = 170 → L = 98 := 
by 
  intros L K h
  sorry

end leo_weight_proof_l752_752284


namespace total_money_is_305_l752_752848

-- Define the worth of each gold coin, silver coin, and the quantity of each type of coin and cash.
def worth_of_gold_coin := 50
def worth_of_silver_coin := 25
def number_of_gold_coins := 3
def number_of_silver_coins := 5
def cash_in_dollars := 30

-- Define the total money calculation based on given conditions.
def total_gold_value := number_of_gold_coins * worth_of_gold_coin
def total_silver_value := number_of_silver_coins * worth_of_silver_coin
def total_value := total_gold_value + total_silver_value + cash_in_dollars

-- The proof statement asserting the total value.
theorem total_money_is_305 : total_value = 305 := by
  -- Proof omitted for brevity.
  sorry

end total_money_is_305_l752_752848


namespace sum_of_cubes_l752_752108

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752108


namespace diamonds_in_G15_l752_752543

theorem diamonds_in_G15 (G : ℕ → ℕ) 
  (h₁ : G 1 = 3)
  (h₂ : ∀ n, n ≥ 2 → G (n + 1) = 3 * (2 * (n - 1) + 3) - 3 ) :
  G 15 = 90 := sorry

end diamonds_in_G15_l752_752543


namespace least_distance_traveled_by_six_boys_l752_752750

theorem least_distance_traveled_by_six_boys 
  (radius : ℝ) (boys : fin 6 → ℂ) 
  (h_radius : radius = 40) 
  (h_boys_spaced : ∀ i : fin 6, norm (boys i) = radius)
  (h_boys_angles : ∀ i j : fin 6, i ≠ j → arg (boys i) - arg (boys j) = (i.1 - j.1) * (2 * π / 6)) :
  let distance_traveled := 6 * (80 + 80*real.sqrt 3) in
  distance_traveled = 480 + 480*real.sqrt 3 :=
sorry

end least_distance_traveled_by_six_boys_l752_752750


namespace positive_divisors_3k1_ge_3k_minus_1_l752_752746

theorem positive_divisors_3k1_ge_3k_minus_1 (n : ℕ) (h : 0 < n) :
  (∃ k : ℕ, (3 * k + 1) ∣ n) → (∃ k : ℕ, ¬ (3 * k - 1) ∣ n) :=
  sorry

end positive_divisors_3k1_ge_3k_minus_1_l752_752746


namespace thirteen_lines_twelve_lines_no_connected_four_l752_752708

-- Defining the problem conditions and theorem
def Points := Fin 6  -- Represents the 6 points A₁, A₂, A₃, A₄, A₅, A₆

-- Condition: No three points are collinear
def no_three_collinear (p: Finset Points) : Prop := 
  ∀ (a b c: Points), a ≠ b → b ≠ c → a ≠ c → (¬ collinear a b c)

-- Define a function to determine if points are connected by a line segment
def is_connected (lines : Finset (Points × Points)) : Points → Points → Prop :=
  λ a b, (a, b) ∈ lines ∨ (b, a) ∈ lines

-- Theorem for the first question
theorem thirteen_lines (lines: Finset (Points × Points)) 
  (h_collinear: no_three_collinear Points)
  (h_size: lines.card = 13) :
  ∃ (p : Finset Points), p.card = 4 ∧
  ∀ (a b ∈ p), is_connected lines a b  :=
begin
  sorry
end

-- For the second question, you can either visualize or construct an example
-- to show the conclusion does not hold with 12 line segments:
noncomputable def twelve_lines_counterexample : Finset (Points × Points) := 
{ sorry }

-- Theorem for the second question
theorem twelve_lines_no_connected_four  :
  ∃ (lines: Finset (Points × Points)),
    lines.card = 12 ∧
    no_three_collinear Points ∧
    (∀ (p : Finset Points), p.card = 4 → 
      ∃ (a b ∈ p), ¬ is_connected lines a b) :=
begin
  use twelve_lines_counterexample,
  split,
  { -- Proof that the constructed counterexample has exactly 12 lines
    sorry, },
  split,
  { -- Proof that the points are no three collinear
    sorry, },
  { -- Proof that for any four points, there exists a pair not connected
    sorry }
end

end thirteen_lines_twelve_lines_no_connected_four_l752_752708


namespace inequality_proof_l752_752612

variables {A B C : Point}
variable {L : Line}
variables {u v w : ℝ} -- The perpendicular distances
variable {S : ℝ} -- The area of triangle ABC
noncomputable def tan (angle : ℝ) : ℝ := sorry -- Assume a definition for tangent

variable [acuteABC : AcuteTriangle A B C] -- Condition for acute-angled triangle
variable [perpendicularDistances : PerpendicularDistances A B C L u v w] -- Condition for perpendicular distances
variable [triangleArea : TriangleArea A B C S] -- Area condition

theorem inequality_proof :
    u^2 * (tan ∡A) + v^2 * (tan ∡B) + w^2 * (tan ∡C) ≥ 2 * S :=
sorry

end inequality_proof_l752_752612


namespace exists_indices_l752_752719

theorem exists_indices {a b : ℕ → ℕ} (h1 : ∀ n, Nat) :
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q :=
sorry

end exists_indices_l752_752719


namespace problem_pt1_problem_pt2_l752_752544
-- Import necessary libraries

-- Define the function f(x) with conditions a > 0 and a ≠ 1
noncomputable def f (a x: ℝ) : ℝ := real.log (3 - a * x)

-- Define the function g(x)
noncomputable def g (x: ℝ) : ℝ := real.log 3 - 3 * x - real.log 3 + 3 * x

-- State the problem in Lean 4
theorem problem_pt1 (a : ℝ) (ha_pos : 0 < a) (ha_one : a ≠ 1) :
  (f 3 x).domain = (set.Iio 1) ∧ 
  (∀ x, g (-x) = - g x ∧ (g x).domain = set.Ioo (-1) 1) :=
sorry

theorem problem_pt2 :
  ∃ a > 0, a ≠ 1 ∧ 
  ∀ x ∈ set.Icc 2 3, 
    (f a x = f a 3 ∧ f a 3 = 1)
 :=
  ∃ a > 0, a = 3 / 4 ∧  ∀ x ∈ set.Icc 2 3, 
   (3 <= f a x ∧ f a 3 = 1)
 :=
sorry

end problem_pt1_problem_pt2_l752_752544


namespace roots_within_unit_disk_l752_752326

open Complex

theorem roots_within_unit_disk (z1 z2 z3 w1 w2 : ℂ)
  (h_mod_z1 : abs z1 ≤ 1)
  (h_mod_z2 : abs z2 ≤ 1)
  (h_mod_z3 : abs z3 ≤ 1)
  (h_roots : (z - z1) * (z - z2) + (z - z2) * (z - z3) + (z - z3) * (z - z1) = 0)
  (h_w1 : w1 = (2 / 3 * (z1 + z2 + z3) + complex.sqrt ((2 / 3 * (z1 + z2 + z3))^2 - 4 * (1 / 3 * (z1 * z2 + z2 * z3 + z3 * z1)))) / 2)
  (h_w2 : w2 = (2 / 3 * (z1 + z2 + z3) - complex.sqrt ((2 / 3 * (z1 + z2 + z3))^2 - 4 * (1 / 3 * (z1 * z2 + z2 * z3 + z3 * z1)))) / 2) :
  ∀ j, j ∈ {1, 2, 3} → min (abs (zj j - w1)) (abs (zj j - w2)) ≤ 1 :=
by {
  sorry
}

end roots_within_unit_disk_l752_752326


namespace difference_between_means_is_1125_l752_752406

def sum_excluding_max_income (incomes : List ℕ) : ℕ :=
  incomes.foldr (λ x acc, if x ≠ (incomes.maximum (by sorry)).get_or_else 0 then acc + x else acc) 0

theorem difference_between_means_is_1125 (incomes : List ℕ) (h_length : incomes.length = 1200)
  (h_max : incomes.maximum (by sorry) = some 150000) (incorrect_max : ℕ = 1500000) :
  let S := sum_excluding_max_income incomes in
  (S + incorrect_max) / 1200 - (S + 150000) / 1200 = 1125 := 
by
  sorry

end difference_between_means_is_1125_l752_752406


namespace tara_spends_total_l752_752403

noncomputable def day1_cost : ℝ :=
  let gallons_used := 315 / 30 in
  let cost_first_station := 12 * 3 in
  let cost_second_station := gallons_used * 3.5 in
  cost_first_station + cost_second_station

noncomputable def day2_cost : ℝ :=
  let gallons_used := 400 / 25 in
  let cost_first_station := 15 * 4 in
  let cost_second_station := gallons_used * 4.5 in
  cost_first_station + cost_second_station

noncomputable def total_trip_cost : ℝ := day1_cost + day2_cost

#eval total_trip_cost  -- This should evaluate to 204.75

theorem tara_spends_total := total_trip_cost = 204.75

end tara_spends_total_l752_752403


namespace sum_of_cubes_l752_752106

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l752_752106


namespace projection_correct_l752_752253

theorem projection_correct 
  (w : ℝ × ℝ)
  (h_proj1 : ∃ c : ℝ, c * w = (12/13, -5/13)) :
  let v1 := (1, 4) in
  let v2 := (3, 2) in
  let dot_product := v2.1 * w.1 + v2.2 * w.2 in
  let norm_squared := w.1 * w.1 + w.2 * w.2 in
  let k := dot_product / norm_squared in
  k * w = (312/169, -130/169) := 
by
  let w:= (12, -5) -- from the projection given
  sorry

end projection_correct_l752_752253


namespace w_share_l752_752808

variable (k : ℝ) (w x y z : ℝ)

-- Conditions
def w_def := w = k
def x_def := x = 6 * k
def y_def := y = 2 * k
def z_def := z = 4 * k
def condition := x = y + 1500

-- Theorem to prove
theorem w_share : condition → w = 375 :=
by
  intro h
  rw [x_def, y_def] at h
  have k_val : k = 375 := sorry
  rw [w_def, k_val]
  exact sorry

end w_share_l752_752808


namespace bankers_discount_calculation_l752_752407

noncomputable def banker's_discount (BG : ℝ) (T : ℝ) (R : ℝ) : ℝ :=
  let FV := (BG * (1 + (R * T) / 100)) / (R * T / 100 + 1)
  let BD := (FV * R * T) / 100
  in BD

theorem bankers_discount_calculation :
  banker's_discount 500 5 15 = 785.71 :=
by
  sorry

end bankers_discount_calculation_l752_752407


namespace initial_average_runs_l752_752028

theorem initial_average_runs (A : ℝ) (h : 10 * A + 65 = 11 * (A + 3)) : A = 32 :=
  by sorry

end initial_average_runs_l752_752028


namespace solution_part1_solution_part2_l752_752985

def f (x a : ℝ) : ℝ := |x + 1 - 2 * a| + |x - a^2|
def g (x : ℝ) : ℝ := x^2 - 2 * x - 4 + 4 / (x - 1)^2

theorem solution_part1 (a : ℝ) : f (2 * a^2 - 1) a > 4 * |a - 1| → a < -5/3 ∨ a > 1 :=
sorry

theorem solution_part2 (a : ℝ) (x y : ℝ) (h : f x a + g y ≤ 0) : 0 ≤ a ∧ a ≤ 2 :=
sorry

end solution_part1_solution_part2_l752_752985


namespace number_of_equilateral_triangles_l752_752042

def eq_lines (k: ℤ) : ℝ → ℝ → Prop := λ x y, y = k ∨ y = sqrt 2 * x + k ∨ y = -sqrt 2 * x + k

def valid_ks : set ℤ := {k | k ≥ -5 ∧ k ≤ 5}

theorem number_of_equilateral_triangles : ∑ k in (finset.range 11).map (λ i, (-5 : ℤ) + i), 
  ∑ m in (finset.range 11).map (λ i, (-5 : ℤ) + i), 
  ∑ x, eq_lines k x x ∧ eq_lines m x x → 
  ∃ n, ∑ i in (finset.range n), 1 = 180 :=
by
  sorry

end number_of_equilateral_triangles_l752_752042


namespace expressway_lengths_l752_752878

theorem expressway_lengths (x y : ℕ) (h1 : x + y = 519) (h2 : x = 2 * y - 45) : x = 331 ∧ y = 188 :=
by
  -- Proof omitted
  sorry

end expressway_lengths_l752_752878


namespace third_number_in_100th_group_l752_752189

def sequence (n : ℕ) : ℕ := 2 * n - 1

def group_start (n : ℕ) : ℕ := n^2 - n + 1

noncomputable def third_number_in_group (n : ℕ) : ℕ := group_start n + 2

theorem third_number_in_100th_group : third_number_in_group 100 = 9905 := 
by 
  sorry

end third_number_in_100th_group_l752_752189


namespace solve_for_y_l752_752117

theorem solve_for_y (y : ℝ) (h : (30 / 50 : ℝ) = sqrt(y / 50)) : y = 18 := by
  sorry

end solve_for_y_l752_752117


namespace product_of_other_endpoint_l752_752059

theorem product_of_other_endpoint (x y : ℤ) : 
  (4, -3) = (x + 10) / 2 ∧ (y + 7) / 2 →
  x * y = 26 :=
by
  intro h
  cases h with h1 h2
  sorry

end product_of_other_endpoint_l752_752059


namespace original_selling_price_is_990_l752_752531

theorem original_selling_price_is_990 
( P : ℝ ) -- original purchase price
( SP_1 : ℝ := 1.10 * P ) -- original selling price
( P_new : ℝ := 0.90 * P ) -- new purchase price
( SP_2 : ℝ := 1.17 * P ) -- new selling price
( h : SP_2 - SP_1 = 63 ) : SP_1 = 990 :=
by {
  -- This is just the statement, proof is not provided
  sorry
}

end original_selling_price_is_990_l752_752531


namespace average_pencil_cost_l752_752519

theorem average_pencil_cost :
  (let cost_pencils := 15.00
       shipping_cost := 7.50
       discount := 1.50
       num_pencils := 150
       total_cost := cost_pencils + shipping_cost - discount
       total_cost_in_cents := 100 * total_cost
       average_cost := total_cost_in_cents / num_pencils in
   average_cost) = 14 := by
  sorry

end average_pencil_cost_l752_752519


namespace flies_needed_per_frog_l752_752190

theorem flies_needed_per_frog (flies_per_frog : ℕ) : 
  (∀ (fish : ℕ) (frogs : ℕ) (flights : ℕ),
    fish = 8 * frogs → 
    flights = 15 * 9 →
    frogs = 32_400 / flights → 
    flies_per_frog = 30 
  ) := 
  sorry

end flies_needed_per_frog_l752_752190


namespace largest_domain_of_g_l752_752412

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_domain_condition (x : ℝ) : x ∈ {y | ∃ x, y = 1 / x^2} 
axiom g_functional_condition (x : ℝ) : g(x) + g(1 / x^2) = x^2

theorem largest_domain_of_g : ∀ x, x ∈ domain g ↔ x = 1 ∨ x = -1 := by
  sorry

end largest_domain_of_g_l752_752412


namespace find_f_at_one_l752_752944

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 2 * x - 8

theorem find_f_at_one (h_cond : f a b (-1) = 10) : f a b (1) = 14 := by
  sorry

end find_f_at_one_l752_752944


namespace speed_of_person_l752_752122

theorem speed_of_person (distance_meters : ℝ) (time_minutes : ℝ) 
  (h1 : distance_meters = 1000) (h2 : time_minutes = 10) : 
  let distance_km := distance_meters / 1000,
      time_hours := time_minutes / 60,
      speed := distance_km / time_hours in
  speed = 6 :=
by
  -- distance_km: Real
  -- time_hours: Real
  -- speed: Real
  sorry

end speed_of_person_l752_752122


namespace union_A_B_equals_x_lt_3_l752_752697

theorem union_A_B_equals_x_lt_3 :
  let A := { x : ℝ | 3 - x > 0 ∧ x + 2 > 0 }
  let B := { x : ℝ | 3 > 2*x - 1 }
  A ∪ B = { x : ℝ | x < 3 } :=
by
  sorry

end union_A_B_equals_x_lt_3_l752_752697


namespace book_pages_l752_752939

theorem book_pages (pages_per_day : ℕ) (days : ℕ) : pages_per_day = 8 → days = 72 → pages_per_day * days = 576 := 
by
  intros h_pages_per_day h_days
  rw [h_pages_per_day, h_days]
  norm_num
  sorry

end book_pages_l752_752939


namespace trailing_zeroes_of_60_factorial_plus_120_factorial_l752_752124

def trailing_zeroes (n : ℕ) : ℕ :=
  (Nat.digits 10 n).takeWhile (· = 0).length

theorem trailing_zeroes_of_60_factorial_plus_120_factorial :
  trailing_zeroes (Nat.factorial 60 + Nat.factorial 120) = 14 :=
sorry

end trailing_zeroes_of_60_factorial_plus_120_factorial_l752_752124


namespace train_speed_l752_752508

theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) 
  (h_train_len : train_length = 110) 
  (h_bridge_len : bridge_length = 132) 
  (h_time : time = 12.099) : 
  let total_distance := train_length + bridge_length in
  let speed_m_per_s := total_distance / time in
  let speed_km_per_h := speed_m_per_s * 3.6 in
  speed_km_per_h = 72 :=
by
  sorry

end train_speed_l752_752508


namespace contrapositive_mul_non_zero_l752_752032

variables (a b : ℝ)

theorem contrapositive_mul_non_zero (h : a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :
  (a = 0 ∨ b = 0) → a * b = 0 :=
by
  sorry

end contrapositive_mul_non_zero_l752_752032


namespace maximum_intersections_l752_752180

noncomputable def max_intersections (m p : ℕ) (r s : ℝ) (Q1 Q2 : Type)
  [polygon Q1] [regular_polygon Q1 m] [polygon Q2] [convex Q2] [sided Q2 p] : ℕ :=
if r < s ∧ m ≤ p then
  m * p
else 
  0

theorem maximum_intersections (m p : ℕ) (r s : ℝ) (Q1 Q2 : Type)
  [polygon Q1] [regular_polygon Q1 m] [polygon Q2] [convex Q2] [sided Q2 p] 
  (h1 : r < s) (h2 : m ≤ p) : 
  max_intersections m p r s Q1 Q2 = m * p := 
by
  unfold max_intersections
  simp [h1, h2]
  sorry

end maximum_intersections_l752_752180


namespace smallest_unrepresentable_integer_l752_752931

theorem smallest_unrepresentable_integer :
  ∃ n : ℕ, (∀ a b c d : ℕ, (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → 
  n ≠ (2^a - 2^b) / (2^c - 2^d)) ∧ n = 11 :=
by
  sorry

end smallest_unrepresentable_integer_l752_752931


namespace min_value_of_inverse_proportional_function_l752_752266

theorem min_value_of_inverse_proportional_function 
  (x y : ℝ) (k : ℝ) 
  (h1 : y = k / x) 
  (h2 : ∀ x, -2 ≤ x ∧ x ≤ -1 → y ≤ 4) :
  (∀ x, x ≥ 8 → y = -1 / 2) :=
by
  sorry

end min_value_of_inverse_proportional_function_l752_752266


namespace find_power_l752_752819

-- Define the problem with given constants and power x
def problem (x : ℝ) : Prop :=
  (5^x) * (12^0.25) * (60^0.75) = 300

-- Assert the problem solution with x roughly equal to 1.886
theorem find_power : ∃ x, problem x ∧ x ≈ 1.886 :=
  by
    sorry

end find_power_l752_752819


namespace identify_incorrect_options_l752_752801

theorem identify_incorrect_options :
  ¬ (∀ (x1 y1 x2 y2 : ℝ), (y2 - y1) ≠ 0 ∧ (x2 - x1) ≠ 0 → (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1)) ∧
  (
    let A := (3, 1),
        B := (2, 3),
        y_axis := (0, 1),
        PA := λ P : ℝ × ℝ, real.dist P A,
        PB := λ P : ℝ × ℝ, real.dist P B in
    ∀ P : ℝ × ℝ,
      (P.1 = 0) → PA P + PB P < real.sqrt 29
  ) ∧
  (
    let distance_between_lines := λ (a b c a' b' c' : ℝ),
      real.abs (c' - c) / real.sqrt (a ^ 2 + b ^ 2) in
    distance_between_lines 1 (-2) (-2) 2 (-4) 1 = real.sqrt 5 / 2
  ) ∧
  (
    let A := (-3, 4),
        B := (3, 2),
        P := (1, 0),
        slope := λ (P1 P2 : ℝ × ℝ), (P2.2 - P1.2) / (P2.1 - P1.1) in
    ∀ l : ℝ × ℝ, 
      slope P A <= -1 ∨ slope P A >= 1 ∧
      slope P B <= -1 ∨ slope P B >= 1
  ) :=
  sorry

end identify_incorrect_options_l752_752801


namespace find_log2_sum_l752_752598

axiom arithmetic_sequence (a : ℕ → ℤ) (h_arith : ∀ n ≥ 2, a (n + 1) + a (n - 1) = a n ^ 2) : Prop

axiom geometric_sequence (b : ℕ → ℤ) (h_geom : ∀ n ≥ 2, b (n + 1) * b (n - 1) = 2 * b n) : Prop

theorem find_log2_sum (a : ℕ → ℤ) (b : ℕ → ℤ)
  (h_arith : ∀ n ≥ 2, a (n + 1) + a (n - 1) = a n ^ 2)
  (h_geom : ∀ n ≥ 2, b (n + 1) * b (n - 1) = 2 * b n)
  (ha_pos : ∀ n, 0 < a n)
  (hb_pos : ∀ n, 0 < b n) :
  Real.log2 (a 2 + b 2) = 2 := by 
  sorry

end find_log2_sum_l752_752598


namespace length_of_CD_l752_752779

-- Given a region within 4 units of line segment CD
-- Includes a cylinder with radius 4, height h, 
-- a hemisphere of radius 4 capping one end,
-- and a cone with base radius 4 capping the other end.
-- Total volume of the region is 352π.
-- Prove the length of the segment CD is 22 units.

theorem length_of_CD (h r: ℝ) (V: ℝ) (total_volume: V = 352 * Real.pi)
  (radius: r = 4) : h + r = 22 :=
by
  -- conditions
  have hemisphere_volume : Real := (1/2) * (4/3) * Real.pi * (4^3),
  have cone_volume : Real := (1/3) * Real.pi * (4^2) * 4,
  have caps_volume : Real := hemisphere_volume + cone_volume,
  have cylinder_volume : Real := V - caps_volume,
  have height_cylinder := cylinder_volume / (Real.pi * (r^2)),
  have total_length := height_cylinder + r,
  -- proof
  sorry

end length_of_CD_l752_752779


namespace cost_of_later_purchase_l752_752031

-- Define the costs of bats and balls as constants.
def cost_of_bat : ℕ := 500
def cost_of_ball : ℕ := 100

-- Define the quantities involved in the later purchase.
def bats_purchased_later : ℕ := 3
def balls_purchased_later : ℕ := 5

-- Define the expected total cost for the later purchase.
def expected_total_cost_later : ℕ := 2000

-- The theorem to be proved: the cost of the later purchase of bats and balls is $2000.
theorem cost_of_later_purchase :
  bats_purchased_later * cost_of_bat + balls_purchased_later * cost_of_ball = expected_total_cost_later :=
sorry

end cost_of_later_purchase_l752_752031


namespace selected_students_in_interval_l752_752145

-- Define the problem parameters
def num_students : ℕ := 1221
def samples_selected : ℕ := 37
def interval_start : ℕ := 496
def interval_end : ℕ := 825
def interval_length := interval_end - interval_start + 1

-- Define the systematic sampling function
def systematic_sampling (total : ℕ) (selected : ℕ) : ℕ := 
  total / selected

-- Define the condition for systematic sampling
def sampling_ratio : ℕ := systematic_sampling num_students samples_selected

-- Define the proof problem
theorem selected_students_in_interval :
  let num_selected_in_interval := interval_length / sampling_ratio in
  num_selected_in_interval = 10 :=
by
  sorry

end selected_students_in_interval_l752_752145


namespace cos_alpha_minus_beta_l752_752969

theorem cos_alpha_minus_beta {α β : ℝ} 
  (h1 : Real.tan ((α + β) / 2) = Real.sqrt 6 / 2)
  (h2 : Real.cot α * Real.cot β = 7 / 13) : 
  Real.cos (α - β) = 2 / 3 := by
  sorry

end cos_alpha_minus_beta_l752_752969


namespace odd_function_value_neg2_l752_752618

-- Define the function f(x) and the given properties
def f (x : ℝ) : ℝ :=
if x > 0 then 10^x else -10^x

-- Formal statement of the problem
theorem odd_function_value_neg2 :
  f (-2) = -100 := sorry

end odd_function_value_neg2_l752_752618


namespace perimeter_of_rearranged_figure_l752_752860

theorem perimeter_of_rearranged_figure (side_length : ℕ) (h : side_length = 100) :
  let rectangle_short_side := side_length / 2;
  let rectangle_long_side := side_length;
  let three_long_sides := 3 * rectangle_long_side;
  let four_short_sides := 4 * rectangle_short_side;
  three_long_sides + four_short_sides = 500 := 
by
  -- definitions based on the conditions
  let rectangle_short_side := side_length / 2;
  let rectangle_long_side := side_length;
  let three_long_sides := 3 * rectangle_long_side;
  let four_short_sides := 4 * rectangle_short_side;

  -- the goal
  have total_length := three_long_sides + four_short_sides;
  calc
    total_length = 3 * side_length + 4 * (side_length / 2) : by sorry -- missing legit substitutions and arithmetic simplification
           ... = 3 * 100 + 4 * 50 : by rw [h]
           ... = 300 + 200 : by rw [add_mul, mul_assoc, nat.div_eq_of_lt];
           ... = 500 : rfl

end perimeter_of_rearranged_figure_l752_752860


namespace find_a_l752_752772

theorem find_a : ∃ a : ℝ, (a * 4 + (a + 1) * (-8) = a + 2) ∧ a = -2 :=
by
  use -2
  split
  . have h : 4 * (-2) + ((-2) + 1) * (-8) = -2 + 2 := by
      simp only [mul_neg_eq_neg_mul_symm, add_assoc (4 * (-2)) (-2 * (-8))]
      norm_num
    exact h
  . rfl

end find_a_l752_752772


namespace number_of_non_congruent_squares_in_6_by_6_grid_l752_752646

theorem number_of_non_congruent_squares_in_6_by_6_grid :
  let total_standard_squares := 25 + 16 + 9 + 4 + 1 in
  let total_alternative_squares := 25 + 9 + 40 + 24 in
  total_standard_squares + total_alternative_squares = 153 := 
by
  sorry

end number_of_non_congruent_squares_in_6_by_6_grid_l752_752646


namespace trigonometric_identity_l752_752900

theorem trigonometric_identity
  (h1 : cos (70 * (Real.pi / 180)) ≠ 0)
  (h2 : sin (70 * (Real.pi / 180)) ≠ 0) :
  (1 / cos (70 * (Real.pi / 180)) - 2 / sin (70 * (Real.pi / 180))) = 
  (4 * sin (10 * (Real.pi / 180))) / sin (40 * (Real.pi / 180)) :=
  sorry

end trigonometric_identity_l752_752900


namespace power_function_decreasing_m_l752_752267

theorem power_function_decreasing_m :
  ∀ (m : ℝ), (m^2 - 5*m - 5) * (2*m + 1) < 0 → m = -1 :=
by
  sorry

end power_function_decreasing_m_l752_752267


namespace minimum_value_f_at_1_k_bounds_l752_752986

noncomputable def f (x : ℝ) : ℝ := (x - (1 / x)) * Real.log x
noncomputable def g (x k : ℝ) : ℝ := x - (k / x)

theorem minimum_value_f_at_1 :
  (∃ x > 0, ∀ y > 0, f(y) ≥ f(x)) ∧ (f(1) = 0) :=
sorry

theorem k_bounds (k : ℝ) (h : ∃ x ∈ Set.Ici 1, f(x) - g(x, k) = 0) :
  1 ≤ k ∧ k < 17 / 8 :=
sorry

end minimum_value_f_at_1_k_bounds_l752_752986


namespace sum_of_cubes_l752_752102

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l752_752102


namespace total_cost_correct_l752_752075

-- Define the conditions
def uber_cost : ℤ := 22
def lyft_additional_cost : ℤ := 3
def taxi_additional_cost : ℤ := 4
def tip_percentage : ℚ := 0.20

-- Define the variables for cost of Lyft and Taxi based on the problem
def lyft_cost : ℤ := uber_cost - lyft_additional_cost
def taxi_cost : ℤ := lyft_cost - taxi_additional_cost

-- Calculate the tip
def tip : ℚ := taxi_cost * tip_percentage

-- Final total cost including the tip
def total_cost : ℚ := taxi_cost + tip

-- The theorem to prove
theorem total_cost_correct :
  total_cost = 18 := by
  sorry

end total_cost_correct_l752_752075


namespace parabola_equation_parabola_equation_min_value_dot_product_l752_752596

/-- The conditions defined for the parabola and points -/
def parabola (p : ℝ) (h : p > 0) : Prop :=
  ∃ F : ℝ × ℝ, F = (0, p / 2) ∧
  ∃ M N : ℝ × ℝ, 
    (M.1 * M.1 = 2 * p * M.2) ∧ 
    (N.1 * N.1 = 2 * p * N.2) ∧ 
    (M.2 = N.2 + 1) ∧
    (|M.1 - N.1| = 8) 

/-- The problem's first property: equation of the parabola -/
theorem parabola_equation : ∀ p : ℝ, p > 0 → parabola p → (x: ℝ) := M.1 := (x^2 = 2 *p * y ) -- correctness as given will yield wrong output
-- to paraphrase properly altering the problem's first property is necessary:
theorem parabola_equation : ∀ p : ℝ, p = 2 → ∃ y. parabola p y → ∀ x, x^2 = 4y:= sorry

axiom min_value_of_dot_product (l: ℝ × ℝ) (l2: ℝ × ℝ) : parabola p = lx2:

/-- The second problem part: minimum dot product value -/
theorem min_value_dot_product : ∀ p : ℝ, p = 2 → parabola p → 
  ∀ (P M N : ℝ × ℝ), 
    parallel l (M ⊙ (l2))  
    tangent equiv :
  ∃ v : ℝ, v = -32 :=sorry

end parabola_equation_parabola_equation_min_value_dot_product_l752_752596


namespace find_x_value_l752_752653

theorem find_x_value (x : ℝ) (h1 : Real.sin (π / 2 - x) = -Real.sqrt 3 / 2) (h2 : π < x ∧ x < 2 * π) : x = 7 * π / 6 :=
sorry

end find_x_value_l752_752653


namespace prime_of_factorial_plus_one_l752_752389

theorem prime_of_factorial_plus_one (n : ℕ) (h : (n! + 1) % (n + 1) = 0) : Nat.prime (n + 1) :=
sorry

end prime_of_factorial_plus_one_l752_752389


namespace min_cardinality_A_l752_752959

theorem min_cardinality_A (m a b : ℕ) (h : Nat.gcd a b = 1)
    (A : Set ℕ) (hA : ∀ n, (a * n ∈ A ∨ b * n ∈ A)) :
    ∃ k, k = ∑ i in Finset.range (m + 1), (-1 : ℤ) ^ (i + 1) * (m / a ^ i) ∧ k = card (A ∩ {n | n ≤ m}) :=
by
  sorry

end min_cardinality_A_l752_752959


namespace vertical_asymptote_of_f_l752_752579

noncomputable def asymptote_x_value : ℚ := 4 / 3

theorem vertical_asymptote_of_f : ∀ (x : ℚ), (6 * x - 8 = 0) → (x = asymptote_x_value) := 
by
  intros x h,
  sorry

end vertical_asymptote_of_f_l752_752579


namespace catch_up_time_l752_752584

theorem catch_up_time (S : ℝ) (t1 t2 : ℝ) (delay : ℝ) (v_y v_o d_y v_r : ℝ)
  (h1 : t1 = 12)
  (h2 : t2 = 20)
  (h3 : delay = 5)
  (h4 : v_y = S / t2)
  (h5 : v_o = S / t1)
  (h6 : d_y = v_y * delay)
  (h7 : v_r = v_o - v_y)
  (h8 : ∀ t : ℝ, d_y = v_r * t → t = 7.5) :
  5 + 7.5 = 12.5 := 
by 
  calc
  5 + 7.5 = 12.5 : by norm_num

end catch_up_time_l752_752584


namespace solution_set_ineq_l752_752430

theorem solution_set_ineq (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x < -1 ∨ x > 3 / 2 :=
by
  split
  { intros h
    sorry },
  { intros h
    sorry }

end solution_set_ineq_l752_752430


namespace solution_l752_752344

noncomputable def problem_statement : Prop :=
  let O := (0, 0, 0)
  let A := (α : ℝ, 0, 0)
  let B := (0, β : ℝ, 0)
  let C := (0, 0, γ : ℝ)
  let plane_equation := λ x y z, (x / α + y / β + z / γ = 1)
  let distance_from_origin := 2
  let centroid := ((α / 3), (β / 3), (γ / 3))
  let p := centroid.1
  let q := centroid.2
  let r := centroid.3
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 2.25

theorem solution : problem_statement := 
by sorry

end solution_l752_752344


namespace overall_percentage_change_l752_752391

theorem overall_percentage_change (S : ℝ) :
  let S1 := S * 0.6 in       -- After 40% decrease
  let S2 := S1 * 1.2 in      -- After 20% increase
  let final_salary := S2 * 0.9 in  -- After 10% decrease
  let overall_change := (final_salary - S) / S * 100 in
  overall_change = -35.2 :=
by
  sorry

end overall_percentage_change_l752_752391


namespace remaining_amount_is_9_l752_752906

-- Define the original prices of the books
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00

-- Define the discount rate for the first two books
def discount_rate : ℝ := 0.25

-- Define the total cost without discount
def total_cost_without_discount := book1_price + book2_price + book3_price + book4_price

-- Calculate the discounts for the first two books
def book1_discount := book1_price * discount_rate
def book2_discount := book2_price * discount_rate

-- Calculate the discounted prices for the first two books
def discounted_book1_price := book1_price - book1_discount
def discounted_book2_price := book2_price - book2_discount

-- Calculate the total cost of the books with discounts applied
def total_cost_with_discount := discounted_book1_price + discounted_book2_price + book3_price + book4_price

-- Define the free shipping threshold
def free_shipping_threshold : ℝ := 50.00

-- Calculate the remaining amount Connor needs to spend
def remaining_amount_to_spend := free_shipping_threshold - total_cost_with_discount

-- State the theorem
theorem remaining_amount_is_9 : remaining_amount_to_spend = 9.00 := by
  -- we would provide the proof here
  sorry

end remaining_amount_is_9_l752_752906


namespace sum_of_cubes_l752_752093

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l752_752093


namespace slower_train_speed_correct_l752_752794

noncomputable def speed_of_slower_train : ℝ := sorry

theorem slower_train_speed_correct :
  (∃ v : ℝ,
    (135.0108 = (v + 45) * ((5:ℝ) / 18) * 6) ∧
    v = speed_of_slower_train) →
  speed_of_slower_train = 36.00648 :=
by
  intros h
  cases h with v hv
  cases hv with eq1 eq2
  rw eq2
  exact sorry

end slower_train_speed_correct_l752_752794


namespace quadratic_root_expression_value_l752_752283

theorem quadratic_root_expression_value (a : ℝ) 
  (h : a^2 - 2 * a - 3 = 0) : 2 * a^2 - 4 * a + 1 = 7 :=
by
  sorry

end quadratic_root_expression_value_l752_752283


namespace exp_gt_x_plus_one_l752_752081

theorem exp_gt_x_plus_one (x : ℝ) (hx: x > 0) : exp x > x + 1 := 
sorry

end exp_gt_x_plus_one_l752_752081


namespace reasoning_wrong_form_l752_752434

theorem reasoning_wrong_form {c : ℂ} 
  (h1 : ∃ x : ℂ, x ∈ ℝ) 
  (h2 : c ∈ ℂ) : ¬ (c ∈ ℝ) :=
by sorry

end reasoning_wrong_form_l752_752434


namespace general_term_formula_l752_752952

/-- Given a sequence defined by S_n = 2n^2 - n + 1, we want to prove the general term formula. -/
theorem general_term_formula (n : ℕ) : 
  let S : ℕ → ℕ := λ n, 2*n^2 - n + 1
  in a_n n =
  (if n = 1 then 2 else 4*n - 3) :=
sorry

end general_term_formula_l752_752952


namespace compute_result_l752_752902

noncomputable def compute_expr : ℚ :=
  8 * (2 / 7)^4

theorem compute_result : compute_expr = 128 / 2401 := 
by 
  sorry

end compute_result_l752_752902


namespace find_some_number_l752_752659

-- Definitions and conditions from the problem
def q (x : ℝ) (c : ℝ) : ℝ := (x - 5)^2 + (x + 1)^2 - c

-- Lean statement: Proving the number that should replace "some number" to make the equation true when y is least at x = 2
theorem find_some_number (y : ℝ) (h : y = q 2 c) : c = 18 :=
by
  -- Since q is not explicitly defined, we assume the least value is zero
  have : q 2 c = 0,
  sorry,
  -- Solve for c
  have : (2 - 5)^2 + (2 + 1)^2 - c = 0,
  sorry,
  -- Simplify to get c = 18
  have : 18 - c = 0,
  sorry,
  exact eq.symm (eq_of_sub_eq_zero this)

end find_some_number_l752_752659


namespace ellipse_trajectory_l752_752233

theorem ellipse_trajectory (P : ℝ × ℝ) :
  let F1 := (1 : ℝ, 0) in
  let F2 := (-1 : ℝ, 0) in
  (dist P F1 + dist P F2 = 4) → (P.1^2 / 4 + P.2^2 / 3 = 1) :=
begin
  sorry
end

end ellipse_trajectory_l752_752233


namespace sin_double_angle_l752_752609

-- Given Conditions
variable {α : ℝ}
variable (h1 : 0 < α ∧ α < π / 2) -- α is in the first quadrant
variable (h2 : Real.sin α = 3 / 5) -- sin(α) = 3/5

-- Theorem statement
theorem sin_double_angle (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 := 
sorry

end sin_double_angle_l752_752609


namespace no_inf_set_nat_abc_perfect_square_l752_752170

theorem no_inf_set_nat_abc_perfect_square :
  ¬∃ (S : Set ℕ), (Set.Infinite S) ∧
  (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → c ≠ a → ∃ (k : ℕ), abc + 1 = k^2) :=
by
  sorry

end no_inf_set_nat_abc_perfect_square_l752_752170


namespace area_ratio_correct_l752_752311

noncomputable def area_ratio (circ : Circle) (A B C E F : Point) (β : Angle) (h1 : is_diameter A B circ) (h2 : forms_angle_with_diameter CF A (cos β)) (h3 : intersects_at AC BF E) (h4 : angle_at_point AED β) : Real := 
  let AE := distance A E
  let CE := distance C E
  (tan β) ^ 2

theorem area_ratio_correct (circ : Circle) (A B C E F : Point) (β : Angle) (h1 : is_diameter A B circ) (h2 : forms_angle_with_diameter CF A (cos β)) (h3 : intersects_at AC BF E) (h4 : angle_at_point AED β) :
  area_ratio circ A B C E F β h1 h2 h3 h4 = (Real.tan β) ^ 2 :=
sorry

end area_ratio_correct_l752_752311


namespace evaluate_statements_l752_752301

-- Define the conditions of the problem
def study (smoking lung_cancer : Prop) : Prop :=
  ∃ (data : Type) (analyzed : data → Prop), 
  (∀ d : data, analyzed d) ∧ (∀ d : data, analyzed d → smoking ∧ lung_cancer) ∧ 
  (probability smoking lung_cancer > 0.99)

-- Define each statement
def statement1 (ρ : Prop) : Prop := 
  ∀ (n : ℕ), (n = 100 → ∃ m : ℕ, (m = 99 ∧ ρ))

def statement2 (ρ : Prop) : Prop := 
  ∀ x, (ρ x → probabilistic x lung_cancer (0.99))

def statement3 (ρ : Prop) : Prop := 
  ∀ (n : ℕ), (n = 100 → ∃ x, ∀ y, ρ x y lung_cancer)

def statement4 (ρ : Prop) : Prop := 
  ∀ (n : ℕ), (n = 100 → ¬ ∃ x, ∀ y, ρ x y lung_cancer)

-- Define the theorem to prove that the only correct statement is statement4
theorem evaluate_statements (ρ : Prop) (smoking lung_cancer : ρ -> Prop) (S1 S2 S3 S4 : Prop) :
  study smoking lung_cancer →
  S1 ↔ statement1 smoking →
  S2 ↔ statement2 smoking →
  S3 ↔ statement3 smoking →
  S4 ↔ statement4 smoking →
  S1 = false ∧ S2 = false ∧ S3 = false ∧ S4 = true :=
by 
  sorry 

end evaluate_statements_l752_752301


namespace combined_rate_mpg_900_over_41_l752_752879

-- Declare the variables and conditions
variables {d : ℕ} (h_d_pos : d > 0)

def combined_mpg (d : ℕ) : ℚ :=
  let anna_car_gasoline := (d : ℚ) / 50
  let ben_car_gasoline  := (d : ℚ) / 20
  let carl_car_gasoline := (d : ℚ) / 15
  let total_gasoline    := anna_car_gasoline + ben_car_gasoline + carl_car_gasoline
  ((3 : ℚ) * d) / total_gasoline

-- Define the theorem statement
theorem combined_rate_mpg_900_over_41 :
  ∀ d : ℕ, d > 0 → combined_mpg d = 900 / 41 :=
by
  intros d h_d_pos
  rw [combined_mpg]
  -- Steps following the solution
  sorry -- proof omitted

end combined_rate_mpg_900_over_41_l752_752879


namespace triangle_perimeter_upper_bound_l752_752863

theorem triangle_perimeter_upper_bound (a b : ℕ) (s : ℕ) (h₁ : a = 7) (h₂ : b = 23) 
  (h₃ : 16 < s) (h₄ : s < 30) : 
  ∃ n : ℕ, n = 60 ∧ n > a + b + s := 
by
  sorry

end triangle_perimeter_upper_bound_l752_752863


namespace average_speed_of_bus_trip_l752_752830

theorem average_speed_of_bus_trip 
  (v d : ℝ) 
  (h1 : d = 560)
  (h2 : ∀ v > 0, ∀ Δv > 0, (d / v) - (d / (v + Δv)) = 2)
  (h3 : Δv = 10): 
  v = 50 := 
by 
  sorry

end average_speed_of_bus_trip_l752_752830


namespace silver_nitrate_mass_fraction_l752_752167

variable (n : ℝ) (M : ℝ) (m_total : ℝ)
variable (m_agno3 : ℝ) (omega_agno3 : ℝ)

theorem silver_nitrate_mass_fraction 
  (h1 : n = 0.12) 
  (h2 : M = 170) 
  (h3 : m_total = 255)
  (h4 : m_agno3 = n * M) 
  (h5 : omega_agno3 = (m_agno3 * 100) / m_total) : 
  m_agno3 = 20.4 ∧ omega_agno3 = 8 :=
by
  -- insert proof here eventually 
  sorry

end silver_nitrate_mass_fraction_l752_752167


namespace probability_intersection_three_elements_l752_752271

theorem probability_intersection_three_elements (U : Finset ℕ) (hU : U = {1, 2, 3, 4, 5}) : 
  ∃ (p : ℚ), p = 5 / 62 :=
by
  sorry

end probability_intersection_three_elements_l752_752271


namespace math_problem_l752_752250

/-- Definition of the function -/
def f (a b x : ℝ) : ℝ := a * (1/4)^(abs x) + b

/-- Main problem condition: the function passes through the origin (0, 0). -/
def passes_through_origin (a b : ℝ) : Prop :=
  f a b 0 = 0

/-- Main problem condition: the function approaches y=2 but doesn't intersect. -/
def approaches_line_y_eq_2 (a b : ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x, abs x > M -> abs (f a b x - 2) < ε

/-- Correct answer confirmation -/
theorem math_problem (a b : ℝ) (ha : a = -2) (hb : b = 2) :
  passes_through_origin a b ∧
  approaches_line_y_eq_2 a b ∧
  (a = -2 ∧ b = 2) ∧
  (∀ x, f a b x > 0 ∧ f a b x ≤ 2) ∧
  (∀ x y, x < y ∧ y < 0 → f a b x > f a b y) ∧
  (∀ x y, f a b x = f a b y ∧ x ≠ y → x + y = 0) :=
sorry

end math_problem_l752_752250


namespace a2_a3_a4_a_n_generalgeneral_formula_sum_b_n_l752_752425

-- Define the sequences and conditions
def a (n : ℕ) : ℚ :=
  if n = 1 then 1 else n * a (n - 1) / (a (n - 1) + 2 * n - 2)

def b (n : ℕ) : ℚ :=
  (1 - 1 / 2^n) * a n

-- Define S_n
def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, b (k + 1))

-- The theorem statements
theorem a2 : a 2 = 2 / 3 := sorry
theorem a3 : a 3 = 3 / 7 := sorry
theorem a4 : a 4 = 4 / 15 := sorry
theorem a_n_generalgeneral_formula (n : ℕ) (hn : n > 0) : a n = n / (2^n - 1) := sorry
theorem sum_b_n (n : ℕ) : S n = 2 - (2 + n) / 2^n := sorry

end a2_a3_a4_a_n_generalgeneral_formula_sum_b_n_l752_752425


namespace prob_2_lt_X_le_4_l752_752256

-- Define the PMF of the random variable X
noncomputable def pmf_X (k : ℕ) : ℝ :=
  if h : k ≥ 1 then 1 / (2 ^ k) else 0

-- Define the probability that X lies in the range (2, 4]
noncomputable def P_2_lt_X_le_4 : ℝ :=
  pmf_X 3 + pmf_X 4

-- Theorem stating the probability of x lying in (2, 4) is 3/16.
theorem prob_2_lt_X_le_4 : P_2_lt_X_le_4 = 3 / 16 := 
by
  -- Provide proof here
  sorry

end prob_2_lt_X_le_4_l752_752256


namespace prove_evaluation_l752_752925

open Int
open Real

def evaluate_expr : ℝ :=
  let a := (12 : ℝ) / 5
  let b := a^2
  let c := ceil b
  let d := c + (11 : ℝ) / 3
  floor d

theorem prove_evaluation : evaluate_expr = 9 := 
by
  sorry

end prove_evaluation_l752_752925


namespace relation_among_a_b_c_l752_752352

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 3
noncomputable def c : ℝ := Real.sqrt 6 - Real.sqrt 2

theorem relation_among_a_b_c : a > c ∧ c > b :=
by {
  sorry
}

end relation_among_a_b_c_l752_752352


namespace find_real_x_interval_l752_752929

theorem find_real_x_interval :
    {x : ℝ // ∃ x ∈ Ioo (-2 : ℝ) (0 : ℝ), (1 / (x^2 + 1) > 3 / x + 13 / 10)} := 
by
  -- Proof skipped.
  sorry

end find_real_x_interval_l752_752929


namespace centroid_plane_distance_l752_752338

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end centroid_plane_distance_l752_752338


namespace pyramid_coloring_l752_752832

theorem pyramid_coloring (vertices : Fin 5 → ℕ → ℕ) :
  let colors := {1, 2, 3, 4},
  (∀ v, v ∈ vertices → v < 5 ∧ v ∈ colors) →
  (∀ i ≠ j, vertices i ≠ vertices j) →
  (∃! (v1 v2 v3 v4 : ℕ), v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧ v1 ∈ colors ∧ v2 ∈ colors ∧ v3 ∈ colors ∧ v4 ∈ colors) :=
by
  sorry

end pyramid_coloring_l752_752832


namespace silvia_shorter_percentage_l752_752853

theorem silvia_shorter_percentage {a b : ℕ} (h₁ : a = 4) (h₂ : b = 3) :
  let j := a + b,
      s := Real.sqrt (a^2 + b^2),
      percentage := ((j - s) / j) * 100
  in percentage ≈ 30 := by
  sorry

end silvia_shorter_percentage_l752_752853


namespace problem1_problem2_problem3_l752_752231

noncomputable def f (b a : ℝ) (x : ℝ) := (b - 2^x) / (2^x + a) 

-- (1) Prove values of a and b
theorem problem1 (a b : ℝ) : 
  (f b a 0 = 0) ∧ (f b a (-1) = -f b a 1) → (a = 1 ∧ b = 1) :=
sorry

-- (2) Prove f is decreasing function
theorem problem2 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f b a x₁ - f b a x₂ > 0 :=
sorry

-- (3) Find range of k such that inequality always holds
theorem problem3 (a b : ℝ) (h_a1 : a = 1) (h_b1 : b = 1) (k : ℝ) : 
  (∀ t : ℝ, f b a (t^2 - 2*t) + f b a (2*t^2 - k) < 0) → k < -(1/3) :=
sorry

end problem1_problem2_problem3_l752_752231


namespace hcf_of_48_and_64_is_16_l752_752771

theorem hcf_of_48_and_64_is_16
  (lcm_value : Nat)
  (hcf_value : Nat)
  (a : Nat)
  (b : Nat)
  (h_lcm : lcm_value = Nat.lcm a b)
  (hcf_def : hcf_value = Nat.gcd a b)
  (h_lcm_value : lcm_value = 192)
  (h_a : a = 48)
  (h_b : b = 64)
  : hcf_value = 16 := by
  sorry

end hcf_of_48_and_64_is_16_l752_752771


namespace roots_equilateral_triangle_l752_752358

noncomputable def omega : ℂ := complex.exp (2 * real.pi * complex.I / 3)

theorem roots_equilateral_triangle (a b z1 z2 : ℂ) (h1 : z2 = omega * z1) (h2 : z1^2 + a * z1 + b = 0) (h3 : z2^2 + a * z2 + b = 0) : a^2 / b = 0 :=
sorry

end roots_equilateral_triangle_l752_752358


namespace number_of_bottle_caps_put_inside_l752_752665

-- Definitions according to the conditions
def initial_bottle_caps : ℕ := 7
def final_bottle_caps : ℕ := 14
def additional_bottle_caps (initial final : ℕ) := final - initial

-- The main theorem to prove
theorem number_of_bottle_caps_put_inside : additional_bottle_caps initial_bottle_caps final_bottle_caps = 7 :=
by
  sorry

end number_of_bottle_caps_put_inside_l752_752665


namespace trigonometric_values_l752_752953

noncomputable def point_on_terminal_side (x : ℝ) : Prop := 
  ∃ α : ℝ, x ≠ 0 ∧ P = (x, 3) ∧ cos α = (sqrt 10 / 10) * x

theorem trigonometric_values (x : ℝ) (α : ℝ) (h₁ : x ≠ 0) (h₂ : cos α = (sqrt 10 / 10) * x) :
  point_on_terminal_side x →
  sin α = (3 * sqrt 10) / 10 ∧ tan α = if x = 1 then 3 else -3 :=
by
  -- Proof follows from the given conditions and solving system of equations.
  sorry

end trigonometric_values_l752_752953


namespace domain_of_v_l752_752086

noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

theorem domain_of_v :
  {x : ℝ | v x ∈ ℝ} = {x : ℝ | x > 1} :=
by
  sorry

end domain_of_v_l752_752086


namespace concurrence_of_lines_l752_752351

-- Define the points of intersection according to the problem conditions
def intersection {α : Type} (a b : set α) : set α := {p | p ∈ a ∧ p ∈ b}

-- Define concurrency for lines
def concurrent {α : Type} (l1 l2 l3 : set α) : Prop :=
  ∃ p, p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3

-- Problem statement in Lean 4
theorem concurrence_of_lines
  {α : Type}
  (a b c a' b' c' : set α)
  (h1 : concurrent a b c)
  (h2 : concurrent a' b' c')
  (xab' : set α := intersection a b')
  (xba' : set α := intersection b a')
  (xac' : set α := intersection a c')
  (xca' : set α := intersection c a')
  (xbc' : set α := intersection b c')
  (xcb' : set α := intersection c b'):
  concurrent (intersection xab' xba') (intersection xac' xca') (intersection xbc' xcb') :=
sorry

end concurrence_of_lines_l752_752351


namespace centers_of_circumcircles_on_line_l752_752229

theorem centers_of_circumcircles_on_line
  (ω : Circle) 
  (A : Point) 
  (hA : A ∈ interior ω)
  (B : Point) 
  (hAB : A ≠ B) :
  ∀ (X Y : Point), chord_passing_through A ω X Y →
    center (circumcircle (triangle B X Y)) ∈ line_perpendicular_bisector B C := sorry

end centers_of_circumcircles_on_line_l752_752229


namespace right_triangle_hypotenuse_l752_752503

theorem right_triangle_hypotenuse (a b c : ℝ) 
  (h1 : a + b + c = 60) 
  (h2 : 0.5 * a * b = 120) 
  (h3 : a^2 + b^2 = c^2) : 
  c = 26 :=
by {
  sorry
}

end right_triangle_hypotenuse_l752_752503


namespace symmetry_xOz_A_l752_752313

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetry_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y , z := p.z }

theorem symmetry_xOz_A :
  let A := Point3D.mk 2 (-3) 1
  symmetry_xOz A = Point3D.mk 2 3 1 :=
by
  sorry

end symmetry_xOz_A_l752_752313


namespace exponent_arithmetic_proof_l752_752532

theorem exponent_arithmetic_proof :
  ( (6 ^ 6 / 6 ^ 5) ^ 3 * 8 ^ 3 / 4 ^ 3) = 1728 := by
  sorry

end exponent_arithmetic_proof_l752_752532


namespace centroid_plane_distance_l752_752337

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end centroid_plane_distance_l752_752337


namespace number_of_valid_four_digit_integers_l752_752998

-- Definition that the set of allowed digits is {2, 5, 7}
def allowed_digits := {2, 5, 7}

-- Definition of a four-digit positive integer using digits from the allowed set
def valid_four_digit_integers : Nat := 
  let choices_per_digit := 3 -- 2, 5, or 7
  let digit_count := 4
  choices_per_digit ^ digit_count

-- The theorem stating the number of such integers is 81
theorem number_of_valid_four_digit_integers : valid_four_digit_integers = 81 := by
  sorry

end number_of_valid_four_digit_integers_l752_752998


namespace find_p_l752_752469

variable (m n p : ℚ)

theorem find_p (h1 : m = 8 * n + 5) (h2 : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by
  sorry

end find_p_l752_752469


namespace percent_daffodils_is_57_l752_752492

-- Condition 1: Four-sevenths of the flowers are yellow
def fraction_yellow : ℚ := 4 / 7

-- Condition 2: Two-thirds of the red flowers are daffodils
def fraction_red_daffodils_given_red : ℚ := 2 / 3

-- Condition 3: Half of the yellow flowers are tulips
def fraction_yellow_tulips_given_yellow : ℚ := 1 / 2

-- Calculate fractions of yellow and red flowers
def fraction_red : ℚ := 1 - fraction_yellow

-- Calculate fractions of daffodils
def fraction_yellow_daffodils : ℚ := fraction_yellow * (1 - fraction_yellow_tulips_given_yellow)
def fraction_red_daffodils : ℚ := fraction_red * fraction_red_daffodils_given_red

-- Total fraction of daffodils
def fraction_daffodils : ℚ := fraction_yellow_daffodils + fraction_red_daffodils

-- Proof statement
theorem percent_daffodils_is_57 :
  fraction_daffodils * 100 = 57 := by
  sorry

end percent_daffodils_is_57_l752_752492


namespace largest_n_is_9_l752_752208

open Finset Nat

noncomputable def max_n_satisfying_conditions : ℕ :=
  max₀ { n | ∃ (x : Fin n → ℕ) (hx : ∀ i j, i ≠ j → x i ≠ x j),
                       ∀ (a : Fin n → ℤ) (ha : a ≠ 0), 
                         n^3 ∣ (Finset.univ.sum (λ i, a i * x i)) → false }

theorem largest_n_is_9 : max_n_satisfying_conditions = 9 := 
by
  /- Proof omitted -/
  sorry

end largest_n_is_9_l752_752208


namespace purple_marble_probability_l752_752828

theorem purple_marble_probability (P_blue P_green P_purple : ℝ) (h1 : P_blue = 0.35) (h2 : P_green = 0.45) (h3 : P_blue + P_green + P_purple = 1) :
  P_purple = 0.2 := 
by sorry

end purple_marble_probability_l752_752828


namespace find_monotonic_increasing_interval_l752_752775

/-- 
Given the function f(x) = log_(1/2)(x^2 - 2x - 3), prove that the monotonic 
increasing interval of f is (-∞, -1), considering its domain x < -1 or x > 3.
-/
def monotonic_increasing_interval : Prop :=
  let f (x : ℝ) : ℝ := Real.logBase (1/2) (x^2 - 2 * x - 3)
  let domain (x : ℝ) : Prop := x < -1 ∨ x > 3
  ∀ x y : ℝ, domain x → domain y → x < y → f x ≤ f y

theorem find_monotonic_increasing_interval : monotonic_increasing_interval :=
sorry

end find_monotonic_increasing_interval_l752_752775


namespace volume_tetrahedron_OMNB1_l752_752593

noncomputable def cube := {A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ}

noncomputable def is_center (O : ℝ × ℝ × ℝ) (A B C D : ℝ × ℝ × ℝ) : Prop :=
  O = ((A.1 + C.1) / 2, (A.2 + C.2) / 2, A.3)

noncomputable def is_midpoint (M : ℝ × ℝ × ℝ) (A1 D1 : ℝ × ℝ × ℝ) : Prop :=
  M = ((A1.1 + D1.1) / 2, (A1.2 + D1.2) / 2, (A1.3 + D1.3) / 2)

noncomputable def is_midpoint2 (N : ℝ × ℝ × ℝ) (C C1 : ℝ × ℝ × ℝ) : Prop :=
  N = ((C.1 + C1.1) / 2, (C.2 + C1.2) / 2, (C.3 + C1.3) / 2)

noncomputable def edge_length_one (A B : ℝ × ℝ × ℝ) : Prop :=
  dist A B = 1

theorem volume_tetrahedron_OMNB1 {A B C D A1 B1 C1 D1 : ℝ × ℝ × ℝ} (edge_length_one : ∀ p1 p2 ∈ {A, B, C, D, A1, B1, C1, D1}, dist p1 p2 = 1)
  (O : ℝ × ℝ × ℝ) (hO : is_center O A B C D)
  (M : ℝ × ℝ × ℝ) (hM : is_midpoint M A1 D1)
  (N : ℝ × ℝ × ℝ) (hN : is_midpoint2 N C C1) :
  volume_tetrahedron O M N B1 = 7 / 48 :=
sorry

end volume_tetrahedron_OMNB1_l752_752593


namespace part1_part2_l752_752626

-- Part (1)
theorem part1 (x : ℝ) (m : ℝ) (h : x = 2) : 
  (x / (x - 3) + m / (3 - x) = 3) → m = 5 :=
sorry

-- Part (2)
theorem part2 (x : ℝ) (m : ℝ) :
  (x / (x - 3) + m / (3 - x) = 3) → (x > 0) → (m < 9) ∧ (m ≠ 3) :=
sorry

end part1_part2_l752_752626


namespace exists_polyhedron_with_property_l752_752854

structure Vertex :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def is_equilateral_triangle (A B C : Vertex) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A

def polyhedron_predicate (V : List Vertex) : Prop :=
  ∀ (A B : Vertex) (hA_in : A ∈ V) (hB_in : B ∈ V) (h_A_ne_B : A ≠ B),
    ∃ C : Vertex, C ∈ V ∧ is_equilateral_triangle A B C

theorem exists_polyhedron_with_property :
  ∃ V : List Vertex, V.length = 5 ∧ polyhedron_predicate V :=
by
  sorry

end exists_polyhedron_with_property_l752_752854


namespace numbers_not_difference_of_squares_l752_752872

theorem numbers_not_difference_of_squares : 
  (∀ n ∈ (2001..2010), ∃ k, n = 4 * k + 2) ↔ (n = 2002 ∨ n = 2006 ∨ n = 2010) := 
begin
  sorry
end

end numbers_not_difference_of_squares_l752_752872


namespace area_second_square_in_triangle_l752_752068

theorem area_second_square_in_triangle (A B C : ℝ) 
  (isosceles_right_triangle : triangle A B C)
  (s₁ := 21) (area_first_square : s₁^2 = 441)
  (fraction_second_square := 4 / 9) :
  ∃ s₂ : ℝ, s₂^2 = 392 :=
begin
  sorry
end

end area_second_square_in_triangle_l752_752068


namespace initial_distance_l752_752443

-- Define the constants based on the conditions
def speed1 := 5 -- miles per hour
def speed2 := 21 -- miles per hour
def distance_one_min_before := 0.43333333333333335 -- miles

-- Relative speed in miles per minute
def relative_speed_per_minute := (speed1 + speed2) / 60

-- Prove that the initial distance between the two boats is the sum of distance_one_min_before and distance covered in one minute
theorem initial_distance :
  let D := distance_one_min_before + relative_speed_per_minute
  D = 0.8666666666666666 := by
    sorry

end initial_distance_l752_752443


namespace jeopardy_episode_length_l752_752321

-- Definitions based on the conditions
def num_episodes_jeopardy : ℕ := 2
def num_episodes_wheel : ℕ := 2
def wheel_twice_jeopardy (J : ℝ) : ℝ := 2 * J
def total_time_watched : ℝ := 120 -- in minutes

-- Condition stating the total time watched in terms of J
def total_watching_time_formula (J : ℝ) : ℝ :=
  num_episodes_jeopardy * J + num_episodes_wheel * (wheel_twice_jeopardy J)

theorem jeopardy_episode_length : ∃ J : ℝ, total_watching_time_formula J = total_time_watched ∧ J = 20 :=
by
  use 20
  simp [total_watching_time_formula, wheel_twice_jeopardy, num_episodes_jeopardy, num_episodes_wheel, total_time_watched]
  sorry

end jeopardy_episode_length_l752_752321


namespace factor_polynomial_l752_752038

def p (x y z : ℝ) : ℝ := x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2)

theorem factor_polynomial (x y z : ℝ) : 
  p x y z = (x - y) * (y - z) * (z - x) * -(x * y + x * z + y * z) :=
by 
  simp [p]
  sorry

end factor_polynomial_l752_752038


namespace reciprocal_sum_of_roots_l752_752723

theorem reciprocal_sum_of_roots :
  let p := 7 * X^2 - 6 * X + 8 in
  let c := (λ (x : ℚ), is_root p x) in
  let d := (λ (y : ℚ), is_root p y) in
  (∃ c d : ℚ, c + d = 6 / 7 ∧ c * d = 8 / 7 ∧ ∃ α β : ℚ, α = 1 / c ∧ β = 1 / d) →
  α + β = 3 / 4 :=
begin
  sorry
end

end reciprocal_sum_of_roots_l752_752723


namespace sum_of_possible_radii_l752_752484

theorem sum_of_possible_radii : 
  (∃ r : ℝ, (r > 0 ∧ (∀ ε > 0, ∀ x y, ((x - r)^2 + (y - r)^2 = r^2 → 
              ((x - 5)^2 + y^2 = 2^2) → r - 5)^2 + r^2 = (r + 2)^2)) ∧ 
  ∀ r₁ r₂, (r₁ = 7 + 2 * Real.sqrt 7 ∨ r₁ = 7 - 2 * Real.sqrt 7) ∧ 
           (r₂ = 7 + 2 * Real.sqrt 7 ∨ r₂ = 7 - 2 * Real.sqrt 7) → 
           r₁ + r₂ = 14)
:= sorry

end sum_of_possible_radii_l752_752484


namespace total_cost_of_panels_l752_752693

theorem total_cost_of_panels
    (sidewall_width : ℝ)
    (sidewall_height : ℝ)
    (triangle_base : ℝ)
    (triangle_height : ℝ)
    (panel_width : ℝ)
    (panel_height : ℝ)
    (panel_cost : ℝ)
    (total_cost : ℝ)
    (h_sidewall : sidewall_width = 9)
    (h_sidewall_height : sidewall_height = 7)
    (h_triangle_base : triangle_base = 9)
    (h_triangle_height : triangle_height = 6)
    (h_panel_width : panel_width = 10)
    (h_panel_height : panel_height = 15)
    (h_panel_cost : panel_cost = 32)
    (h_total_cost : total_cost = 32) :
    total_cost = panel_cost :=
by
  sorry

end total_cost_of_panels_l752_752693


namespace sqrt_rational_of_sum_sqrt_rational_l752_752318

theorem sqrt_rational_of_sum_sqrt_rational (A B : ℚ) (h1 : ∃ r : ℚ, sqrt A + sqrt B = r) :
  ∃ rA rB: ℚ, sqrt A = rA ∧ sqrt B = rB := 
by
  sorry

end sqrt_rational_of_sum_sqrt_rational_l752_752318


namespace sum_of_all_different_possible_areas_of_cool_rectangles_l752_752148

-- Define the concept of a cool rectangle
def is_cool_rectangle (a b : ℕ) : Prop :=
  a * b = 2 * (2 * a + 2 * b)

-- Define the function to calculate the area of a rectangle
def area (a b : ℕ) : ℕ := a * b

-- Define the set of pairs (a, b) that satisfy the cool rectangle condition
def cool_rectangle_pairs : List (ℕ × ℕ) :=
  [(5, 20), (6, 12), (8, 8)]

-- Calculate the sum of all different possible areas of cool rectangles
def sum_of_cool_rectangle_areas : ℕ :=
  List.sum (cool_rectangle_pairs.map (λ p => area p.fst p.snd))

-- Theorem statement
theorem sum_of_all_different_possible_areas_of_cool_rectangles :
  sum_of_cool_rectangle_areas = 236 :=
by
  -- This is where the proof would go based on the given solution.
  sorry

end sum_of_all_different_possible_areas_of_cool_rectangles_l752_752148


namespace pseudo_ultrafilters_count_l752_752327

def N : ℕ := factorial 12

def X : finset ℕ := (nat.divisors N).erase 1

structure PseudoUltrafilter (X : finset ℕ) (U : finset ℕ) :=
  (nonempty : U.nonempty)
  (closure_divisors : ∀ {a b : ℕ}, a ∣ b → a ∈ U → b ∈ U)
  (closure_gcd : ∀ {a b : ℕ}, a ∈ U → b ∈ U → a.gcd b ∈ U)
  (closure_lcm : ∀ {a b : ℕ}, a ∉ U → b ∉ U → a.lcm b ∉ U)

def count_pseudo_ultrafilters : ℕ :=
  19

theorem pseudo_ultrafilters_count :
  ∃ (U : finset (finset ℕ)), U.card = count_pseudo_ultrafilters ∧ 
  ∀ u ∈ U, PseudoUltrafilter X u :=
sorry

end pseudo_ultrafilters_count_l752_752327


namespace ratio_ties_to_losses_l752_752024

def total_games : ℕ := 56
def losses : ℕ := 12
def wins : ℕ := 38

def ties : ℕ := total_games - (losses + wins)

theorem ratio_ties_to_losses (total_games = 56) (losses = 12) (wins = 38) : ties / losses = 1 / 2 :=
by 
  sorry

end ratio_ties_to_losses_l752_752024


namespace matrix_identity_implies_sum_squares_equals_one_l752_752055

theorem matrix_identity_implies_sum_squares_equals_one (x y z : ℝ)
  (hM : 
    let M := ![
      ![x, 0, z],
      ![y, 0, -z],
      ![x, 0, z]
    ];
    Mᵀ ⬝ M = 1) :
  x^2 + y^2 + z^2 = 1 :=
begin
  sorry,
end

end matrix_identity_implies_sum_squares_equals_one_l752_752055


namespace profit_increase_l752_752061

theorem profit_increase (P : ℝ) : 
  let April_profits := 1.50 * P
      May_profits := 1.20 * P
      June_profits := 1.80 * P
      increase := June_profits - P
      percent_increase := (increase / P) * 100
  in percent_increase = 80 := by
  sorry

end profit_increase_l752_752061


namespace converse_not_always_true_l752_752804

theorem converse_not_always_true :
  ¬ (∀ (T1 T2 : Triangle), (∀ (a b : Angle), corresponding_angles_equal T1 T2 a b → congruent T1 T2)) :=
sorry

end converse_not_always_true_l752_752804


namespace smallest_n_l752_752951

def abs_diff_nearest_square (k : ℕ) : ℕ :=
  let nearest_square := round (sqrt k : ℝ)^2
  abs (k - nearest_square)

theorem smallest_n : ∃ n : ℕ, (n > 0) ∧ 
  ( (∑ i in range (n + 1), abs_diff_nearest_square i) / n = 100 ) ∧ 
  (n = 89800) :=
by {
  sorry
}

end smallest_n_l752_752951


namespace elena_tulips_l752_752562

-- Define the conditions
def lilies := 8
def petals_per_lily := 6
def petals_per_tulip := 3
def total_petals := 63

-- Define the statement to be proved
theorem elena_tulips : ∃ n : ℕ, n = 5 ∧ (lilies * petals_per_lily + n * petals_per_tulip = total_petals) := 
by 
  existsi 5
  split
  · rfl
  · simp [lilies, petals_per_lily, petals_per_tulip, total_petals]
  sorry

end elena_tulips_l752_752562


namespace Zach_scored_more_points_l752_752807

theorem Zach_scored_more_points : 
  ∀ (Zach_points Ben_points : ℕ), Zach_points = 42 → Ben_points = 21 → Zach_points - Ben_points = 21 :=
by
  intros Zach_points Ben_points hZach hBen
  rw [hZach, hBen]
  norm_num
  sorry

end Zach_scored_more_points_l752_752807


namespace fraction_of_repeating_decimal_l752_752571

theorem fraction_of_repeating_decimal:
  let a := (4 / 10 : ℝ)
  let r := (1 / 10 : ℝ)
  (∑' n:ℕ, a * r^n) = (4 / 9 : ℝ) := by
  sorry

end fraction_of_repeating_decimal_l752_752571


namespace geometric_sequence_sum_l752_752212

theorem geometric_sequence_sum (a : ℕ → ℝ) (a1 : a 1 = 1) (q : ℝ) 
  (q_pos : q > 0) (geo_seq : ∀ n, a (n+1) = a n * q) 
  (arith_seq : -a 1 + 3 / 4 * a 2 + a 3 = a 3 - 3 / 4 * a 2 - (-a 1)) :
  ∑ i in finset.range 4, a i = 15 := 
by
  sorry

end geometric_sequence_sum_l752_752212


namespace shaded_area_of_floor_l752_752873

def area_of_one_tile : ℝ := 1

def radius_of_quarter_circle : ℝ := 1 / 2

def area_of_one_quarter_circle (r : ℝ) : ℝ := (1 / 4) * π * r^2

def total_area_of_white_regions_on_one_tile (r : ℝ) : ℝ := 4 * area_of_one_quarter_circle r

def shaded_area_on_one_tile (area_tile : ℝ) (total_white_area : ℝ) : ℝ := area_tile - total_white_area

def total_number_of_tiles (length : ℝ) (width : ℝ) (tile_size : ℝ) : ℕ := (length / tile_size) * (width / tile_size)

def total_shaded_area (num_tiles : ℕ) (shaded_area_per_tile : ℝ) : ℝ := num_tiles * shaded_area_per_tile

theorem shaded_area_of_floor :
  let length := 8
  let width := 10
  let tile_size := 1
  let r := 1 / 2
  let area_tile := area_of_one_tile
  let total_white_area := total_area_of_white_regions_on_one_tile r
  let shaded_area_per_tile := shaded_area_on_one_tile area_tile total_white_area
  let num_tiles := total_number_of_tiles length width tile_size
  total_shaded_area num_tiles shaded_area_per_tile = 80 - 20 * π :=
by
  sorry

end shaded_area_of_floor_l752_752873


namespace problem_1_problem_2_l752_752670

-- Definitions based on the problem conditions
def A (a b c: ℝ) : ℝ := sorry  -- This will represent angle A in radians
def area (a b c: ℝ) : ℝ := sorry  -- This will represent the area of the triangle

noncomputable def given_conditions (a b c A: ℝ) : Prop :=
  (B = π / 4) ∧
  (a * sin B = sqrt(3) * b * cos A) ∧
  (b = 4)

-- Prove question == answer given conditions
theorem problem_1:
  ∀ a b c,
  given_conditions a b c (π / 3) →
  A a b c = π / 3 := by
  sorry

theorem problem_2:
  ∀ a b c,
  given_conditions a b c (π / 3) →
  area a b c = (1 + sqrt 3) * sqrt 2 / 2 - (sqrt 2 + sqrt 6) / 2 := by
  sorry

end problem_1_problem_2_l752_752670


namespace fraction_to_terminating_decimal_l752_752197

theorem fraction_to_terminating_decimal :
  (45 / (2^2 * 5^3) : ℚ) = 0.09 :=
by sorry

end fraction_to_terminating_decimal_l752_752197


namespace prime_property_l752_752272

theorem prime_property (a b c : ℕ) (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c)
 (h_eq : a + b + c + a * b * c = 99) : |a - b| + |b - c| + |c - a| = 34 := sorry

end prime_property_l752_752272


namespace quadratic_real_roots_range_l752_752287

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + m = 0) → m ≤ 9 / 4 :=
by
  sorry

end quadratic_real_roots_range_l752_752287


namespace sum_of_cubes_l752_752105

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l752_752105


namespace minimum_sum_distances_inside_rhombus_l752_752164

theorem minimum_sum_distances_inside_rhombus 
  (A B C D M : Point) 
  (h_rhombus : is_rhombus A B C D)
  (h_AB : dist A B = 6) 
  (h_angle_ABC : angle A B C = 60) : 
  ∃ M : Point, is_inside_rhombus M A B C D ∧ dist M A + dist M B + dist M C = 6 * sqrt 3 :=
  sorry

end minimum_sum_distances_inside_rhombus_l752_752164


namespace exists_x_y_not_divisible_by_3_l752_752361

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (hk : 0 < k) :
  ∃ (x y : ℤ), (x^2 + 2 * y^2 = 3^k) ∧ (¬ (x % 3 = 0)) ∧ (¬ (y % 3 = 0)) :=
sorry

end exists_x_y_not_divisible_by_3_l752_752361


namespace agent_007_can_encrypt_agent_013_can_encrypt_l752_752467

theorem agent_007_can_encrypt :
  ∃ (m n : ℕ), 0.07 = (1 / m) + (1 / n) :=
by
  use [20, 50]
  sorry
  
theorem agent_013_can_encrypt :
  ∃ (m n : ℕ), 0.13 = (1 / m) + (1 / n) :=
by
  use [8, 200]
  sorry

end agent_007_can_encrypt_agent_013_can_encrypt_l752_752467


namespace subset_of_sets_l752_752367

theorem subset_of_sets (a : ℝ) : 
  {0, -a} ⊆ {1, a - 2, 2 * a - 2} → a = 1 :=
by
  intros h
  sorry

end subset_of_sets_l752_752367


namespace largest_invertible_interval_l752_752220

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 5

theorem largest_invertible_interval : 
  ∃ (I : set ℝ), (∀ x ∈ I, ∀ y ∈ I, f x = f y → x = y) ∧ (0 : ℝ) ∈ I ∧ I = {x | x <= 1} :=
sorry

end largest_invertible_interval_l752_752220


namespace candy_distribution_l752_752861

def totalCandy := 344
def reservedCandy := 56
def numberOfStudents := 43
def remainingCandy := totalCandy - reservedCandy
def piecesPerStudent := remainingCandy / numberOfStudents

theorem candy_distribution :⌜ remainingCandy = totalCandy - reservedCandy ⌝ ∧⌜ piecesPerStudent = (remainingCandy / numberOfStudents)⌝ ∧⌜ piecesPerStudent.floor = 6 := by
  sorry

end candy_distribution_l752_752861


namespace sum_of_first_7_terms_l752_752120

theorem sum_of_first_7_terms (a d : ℤ) (h3 : a + 2 * d = 2) (h4 : a + 3 * d = 5) (h5 : a + 4 * d = 8) :
  (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d) + (a + 6 * d)) = 35 :=
by
  sorry

end sum_of_first_7_terms_l752_752120


namespace binary111011_is_59_l752_752546

def binary_to_decimal (b : ℕ) : ℕ :=
  -- function to convert a binary number to its decimal representation
  sum (List.mapWithIndex (λ i bit, bit * 2^i) (b.digits 2).reverse)

def binary111011 : ℕ := 0b111011   -- 0b prefix represents binary literals in Lean

theorem binary111011_is_59 : binary_to_decimal binary111011 = 59 :=
sorry

end binary111011_is_59_l752_752546


namespace distribute_balls_l752_752554

theorem distribute_balls (balls : Finset (Fin 7)) (P1 P2 : Finset (Fin 7)) :
  (P1.card ≥ 2) ∧ (P2.card ≥ 2) ∧ (balls.card = 7) ∧ (P1 ∪ P2 = balls) ∧ (P1 ∩ P2 = ∅) →
  card {d : balls → {1, 2} // (∀ b, d b = 1 ↔ b ∈ P1) ∧ (∀ b, d b = 2 ↔ b ∈ P2)} = 112 :=
sorry

end distribute_balls_l752_752554


namespace solution_l752_752346

noncomputable def problem_statement : Prop :=
  let O := (0, 0, 0)
  let A := (α : ℝ, 0, 0)
  let B := (0, β : ℝ, 0)
  let C := (0, 0, γ : ℝ)
  let plane_equation := λ x y z, (x / α + y / β + z / γ = 1)
  let distance_from_origin := 2
  let centroid := ((α / 3), (β / 3), (γ / 3))
  let p := centroid.1
  let q := centroid.2
  let r := centroid.3
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 2.25

theorem solution : problem_statement := 
by sorry

end solution_l752_752346


namespace find_a_l752_752035

open Real

theorem find_a :
  ∃ a : ℝ, (1/5) * (0.5 + a + 1 + 1.4 + 1.5) = 0.28 * 3 + 0.16 := by
  use 0.6
  sorry

end find_a_l752_752035


namespace initial_amount_is_825_l752_752482

theorem initial_amount_is_825 (P R : ℝ) 
    (h1 : 956 = P * (1 + 3 * R / 100))
    (h2 : 1055 = P * (1 + 3 * (R + 4) / 100)) : 
    P = 825 := 
by 
  sorry

end initial_amount_is_825_l752_752482


namespace equal_heights_of_cylinder_and_cone_l752_752144

theorem equal_heights_of_cylinder_and_cone
  (r h : ℝ)
  (hc : h > 0)
  (hr : r > 0)
  (V_cylinder V_cone : ℝ)
  (V_cylinder_eq : V_cylinder = π * r ^ 2 * h)
  (V_cone_eq : V_cone = 1/3 * π * r ^ 2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
h = h := -- Since we are given that the heights are initially the same
sorry

end equal_heights_of_cylinder_and_cone_l752_752144


namespace sara_disproves_tom_l752_752578

-- Define the type and predicate of cards
inductive Card
| K
| M
| card5
| card7
| card8

open Card

-- Define the conditions
def is_consonant : Card → Prop
| K => true
| M => true
| _ => false

def is_odd : Card → Prop
| card5 => true
| card7 => true
| _ => false

def is_even : Card → Prop
| card8 => true
| _ => false

-- Tom's statement
def toms_statement : Prop :=
  ∀ c, is_consonant c → is_odd c

-- The card Sara turns over (card8) to disprove Tom's statement
theorem sara_disproves_tom : is_even card8 ∧ is_consonant card8 → ¬toms_statement :=
by
  sorry

end sara_disproves_tom_l752_752578


namespace convert_157_base_10_to_base_7_l752_752184

-- Given
def base_10_to_base_7(n : ℕ) : String := "313"

-- Prove
theorem convert_157_base_10_to_base_7 : base_10_to_base_7 157 = "313" := by
  sorry

end convert_157_base_10_to_base_7_l752_752184


namespace find_extrema_l752_752199

noncomputable def f (x : ℝ) : ℝ := (2 / 3) * Real.cos (3 * x - Real.pi / 6)

theorem find_extrema :
  ∃ c₁ c₂ : ℝ, 
  (0 < c₁ ∧ c₁ < Real.pi / 2) ∧ 
  (0 < c₂ ∧ c₂ < Real.pi / 2) ∧ 
  (f c₁ = 2 / 3) ∧ 
  (f c₂ = - (2 / 3)) ∧ 
  (∀ x, (0 < x ∧ x < Real.pi / 2) → f' x = 0 → f x ≤ f c₁ ∨ f x ≥ f c₂) := sorry

end find_extrema_l752_752199


namespace num_incorrect_statements_l752_752011

-- Definitions for the conditions
def statement1 : Prop := ∀ (s : Set ℝ), ∃ a b, a ≠ b ∧ a ∈ s ∧ b ∈ s ∧ (∀ x ∈ s, x = a ∨ x = b)
def statement2 : Prop := ∀ (s : List ℝ) (c : ℝ), (s.map (λ x => x - c)).variance = s.variance
def statement3 : Prop := ∀ (n : ℕ), ∀ (rows : List (List ℕ)), 
  (∀ r ∈ rows, r.length = n) → stratified_sample rows ≠ simple_random_sample rows
def statement4 : Prop := ∀ (s : List ℝ), s.variance ≥ 0

-- Main theorem
theorem num_incorrect_statements : 
  (¬ statement1) ∧ statement2 ∧ (¬ statement3) ∧ (¬ statement4) → 3 = 3 := 
by 
  sorry

end num_incorrect_statements_l752_752011


namespace unique_seatings_around_table_l752_752679

-- Define the factorial function
def fact : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * fact n 

-- Problem statement in Lean
theorem unique_seatings_around_table : 
  let n := 6 in 
  (fact n) / n = 120 := 
by sorry

end unique_seatings_around_table_l752_752679


namespace total_eggs_examined_l752_752530

/-- Ben was given 7 trays of eggs to examine for a research study. 
Each tray holds 10 eggs. However, he was instructed 
to examine different percentages of the eggs on each tray as follows: 
Tray 1: 80% of eggs, Tray 2: 90% of eggs, Tray 3: 70% of eggs, 
Tray 4: 60% of eggs, Tray 5: 50% of eggs, Tray 6: 40% of eggs, Tray 7: 30% of eggs. -/
def trays : ℕ := 7
def eggs_per_tray : ℕ := 10
def percentages : List ℝ := [0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3]

theorem total_eggs_examined : trays = 7 → eggs_per_tray = 10 → percentages = [0.8, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3] → 
  (List.map (λ p, p * eggs_per_tray) percentages).sum = 42 := by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end total_eggs_examined_l752_752530


namespace min_side_length_of_square_l752_752605

theorem min_side_length_of_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∃ s : ℝ, s = 
    if a < (Real.sqrt 2 + 1) * b then 
      a 
    else 
      (Real.sqrt 2 / 2) * (a + b) := 
sorry

end min_side_length_of_square_l752_752605


namespace problem_l752_752000

-- Define the values in the grid
def grid : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ := (4, 3, 1, 1, 6, 2, 3)

-- Define the variables A, B, and C
variables (A B C : ℕ)

-- Define the conditions
def condition_1 := (A = 3) ∧ (B = 2) ∧ (C = 4)
def condition_2 := (4 + A + 1 + B + C + 3 = 9)
def condition_3 := (A + 1 + 6 = 9)
def condition_4 := (1 + A + 6 = 9)
def condition_5 := (B + 2 + C + 5 = 9)

-- Define that the sum of the red cells is equal to any row
def sum_of_red_cells := (A + B + C = 9)

-- The final goal to prove
theorem problem : condition_1 ∧ condition_2 ∧ condition_3 ∧ condition_4 ∧ condition_5 ∧ sum_of_red_cells := 
by {
  refine ⟨_, _, _, _, _, _⟩;
  sorry   -- proofs for each condition
}

end problem_l752_752000


namespace subset_A_B_l752_752370

theorem subset_A_B (a : ℝ) (A : Set ℝ := {0, -a}) (B : Set ℝ := {1, a-2, 2a-2}) 
  (h : A ⊆ B) : a = 1 := by
  sorry

end subset_A_B_l752_752370


namespace rooms_in_house_l752_752724

-- define the number of paintings
def total_paintings : ℕ := 32

-- define the number of paintings per room
def paintings_per_room : ℕ := 8

-- define the number of rooms
def number_of_rooms (total_paintings : ℕ) (paintings_per_room : ℕ) : ℕ := total_paintings / paintings_per_room

-- state the theorem
theorem rooms_in_house : number_of_rooms total_paintings paintings_per_room = 4 :=
by sorry

end rooms_in_house_l752_752724


namespace calculate_expression_l752_752533

theorem calculate_expression :
  (Real.sqrt 12 - 2 * Real.cos (Float.pi / 6) + abs (Real.sqrt 3 - 2) + 2⁻¹) = 5 / 2 :=
by
  sorry

end calculate_expression_l752_752533


namespace minimum_kinder_surprises_needed_l752_752561

theorem minimum_kinder_surprises_needed (S : Finset (Finset ℕ)) :
  (∀ s ∈ S, s.card = 3) → 
  (∀ (a b ∈ S), a ≠ b → ∃ x ∈ a, x ∉ b) → 
  (∀ x ∈ (Finset.range 11).sat, ∃ s ∈ S, x ∈ s) →
  S.card ≥ 121 :=
sorry

end minimum_kinder_surprises_needed_l752_752561


namespace number_of_new_numbers_l752_752181

theorem number_of_new_numbers : 
  let M := 9876543210 in
  ∑ k in (Finset.range (5 + 1)).filter (λ k, 1 ≤ k), Nat.choose (10 - k) k = 88 :=
by
  let M := 9876543210
  sorry

end number_of_new_numbers_l752_752181


namespace greg_total_earnings_l752_752643

-- Define the charges and walking times as given
def charge_per_dog : ℕ := 20
def charge_per_minute : ℕ := 1
def one_dog_minutes : ℕ := 10
def two_dogs_minutes : ℕ := 7
def three_dogs_minutes : ℕ := 9
def total_dogs_one : ℕ := 1
def total_dogs_two : ℕ := 2
def total_dogs_three : ℕ := 3

-- Total earnings computation
def earnings_one_dog : ℕ := charge_per_dog + charge_per_minute * one_dog_minutes
def earnings_two_dogs : ℕ := total_dogs_two * charge_per_dog + total_dogs_two * charge_per_minute * two_dogs_minutes
def earnings_three_dogs : ℕ := total_dogs_three * charge_per_dog + total_dogs_three * charge_per_minute * three_dogs_minutes
def total_earnings : ℕ := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

-- The proof: Greg's total earnings should be $171
theorem greg_total_earnings : total_earnings = 171 := by
  -- Placeholder for the proof (not required as per the instructions)
  sorry

end greg_total_earnings_l752_752643


namespace perpendicular_lines_l752_752362

def Line := Type
def Plane := Type

variable (a b : Line)
variable (α β : Plane)

-- Conditions
axiom condition1 : a ⊂ α
axiom condition2 : b ⊥ β
axiom condition3 : α ≠ β  -- Ensuring that alpha and beta are not parallel

-- Goal
theorem perpendicular_lines : a ⊥ b :=
by sorry

end perpendicular_lines_l752_752362


namespace area_of_square_EFGH_l752_752163

theorem area_of_square_EFGH :
  let original_square_side := 6 in
  let semicircle_radius := original_square_side / 2 in
  let distance_from_square := 1 in
  let efgh_side := original_square_side + 2 * distance_from_square in
  efgh_side * efgh_side = 36 :=
by
  let original_square_side := 6
  let semicircle_radius := original_square_side / 2
  let distance_from_square := 1
  let efgh_side := original_square_side + 2 * distance_from_square
  show efgh_side * efgh_side = 36
  sorry

end area_of_square_EFGH_l752_752163


namespace closest_distance_l752_752307

noncomputable def set_A := {z : ℂ | z ^ 3 = 27}
noncomputable def set_B := {z : ℂ | z ^ 3 - 9 * z ^ 2 - 27 * z + 243 = 0}

theorem closest_distance {zA zB : ℂ} (hA : zA ∈ set_A) (hB : zB ∈ set_B)
  (minA : ∀ w ∈ set_A, complex.abs w ≥ complex.abs zA)
  (minB : ∀ w ∈ set_B, complex.abs w ≥ complex.abs zB) :
  complex.abs (zA - zB) = 3 * (real.sqrt 3 - 1) :=
sorry

end closest_distance_l752_752307


namespace conjugate_of_z_l752_752592

theorem conjugate_of_z : 
  ∀ z : ℂ, ((1 + complex.I) / (1 - complex.I)) * z = (3 + 4 * complex.I) → conj z = (4 + 3 * complex.I) :=
by
  sorry

end conjugate_of_z_l752_752592


namespace arrangements_A_B_next_to_each_other_l752_752784

/--
There are 5 people, and we want to find the number of different arrangements
where person A and person B are standing next to each other.
-/
theorem arrangements_A_B_next_to_each_other (A B C D E : Type) :
  ∃ n, n = 2 * 4! :=
sorry

end arrangements_A_B_next_to_each_other_l752_752784


namespace centroid_plane_distance_l752_752336

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end centroid_plane_distance_l752_752336


namespace find_a_value_l752_752034

-- Given values
def month_code : List ℝ := [1, 2, 3, 4, 5]
def prices (a : ℝ) : List ℝ := [0.5, a, 1, 1.4, 1.5]

-- Linear regression equation parameters
def lin_reg_slope : ℝ := 0.28
def lin_reg_intercept : ℝ := 0.16

-- Average function
def average (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Proof statement
theorem find_a_value (a : ℝ) (h : (average (prices a)) = lin_reg_slope * (average month_code) + lin_reg_intercept) : a = 0.6 :=
  sorry

end find_a_value_l752_752034


namespace sum_of_roots_eq_4140_l752_752211

open Complex

noncomputable def sum_of_roots : ℝ :=
  let θ0 := 270 / 5;
  let θ1 := (270 + 360) / 5;
  let θ2 := (270 + 2 * 360) / 5;
  let θ3 := (270 + 3 * 360) / 5;
  let θ4 := (270 + 4 * 360) / 5;
  θ0 + θ1 + θ2 + θ3 + θ4

theorem sum_of_roots_eq_4140 : sum_of_roots = 4140 := by
  sorry

end sum_of_roots_eq_4140_l752_752211


namespace sampling_correct_probability_of_event_a_l752_752436

noncomputable def number_of_athletes_from_a : ℕ := 27
noncomputable def number_of_athletes_from_b : ℕ := 9
noncomputable def number_of_athletes_from_c : ℕ := 18
noncomputable def total_athletes_selected : ℕ := 6

-- Calculating the number of athletes to be sampled from each association
def selected_from_a : ℕ := number_of_athletes_from_a * total_athletes_selected / (number_of_athletes_from_a + number_of_athletes_from_b + number_of_athletes_from_c)
def selected_from_b : ℕ := number_of_athletes_from_b * total_athletes_selected / (number_of_athletes_from_a + number_of_athletes_from_b + number_of_athletes_from_c)
def selected_from_c : ℕ := number_of_athletes_from_c * total_athletes_selected / (number_of_athletes_from_a + number_of_athletes_from_b + number_of_athletes_from_c)

theorem sampling_correct :
  selected_from_a = 3 ∧ selected_from_b = 1 ∧ selected_from_c = 2 := by
  sorry

-- Defining the number of combinations
def total_combinations (n k : ℕ) : ℕ := nat.choose n k

def favorable_combinations : ℕ := 9
def total_combinations_sampling : ℕ := total_combinations total_athletes_selected 2

theorem probability_of_event_a :
  favorable_combinations / total_combinations_sampling = (3 / 5 : ℝ) := by
  sorry

end sampling_correct_probability_of_event_a_l752_752436


namespace blood_type_combinations_l752_752865

theorem blood_type_combinations : 
    (∀ p ∈ ["A", "B", "O", "AB"], ∀ c ∈ ["A", "B", "O", "AB"], 
    (p = "AB" ∨ c = "AB") → "O" ∉ ["A", "B", "O", "AB"]) →
    count_possible_combinations(["A", "B", "O"]) = 9 :=
by sorry

end blood_type_combinations_l752_752865


namespace article_A_profit_l752_752303

theorem article_A_profit {x y CP_A : ℝ} (hx : x = 1.6 * CP_A)
  (hy : 1.05 * y = 0.9 * x) : 
  let final_selling_price_A := 0.972 * x,
      profit := final_selling_price_A - CP_A,
      profit_percent := (profit / CP_A) * 100
  in profit_percent = 55.52 := 
sorry

end article_A_profit_l752_752303


namespace largest_sum_is_seven_tenths_l752_752890

theorem largest_sum_is_seven_tenths :
  let a := 1/5 + 1/2,
      b := 1/5 + 1/6,
      c := 1/5 + 1/4,
      d := 1/5 + 1/8,
      e := 1/5 + 1/9 in
  max a (max b (max c (max d e))) = 7/10 := 
  by
    sorry

end largest_sum_is_seven_tenths_l752_752890


namespace f_f_of_2_l752_752360

def f (x : ℤ) : ℤ := 4 * x ^ 3 - 3 * x + 1

theorem f_f_of_2 : f (f 2) = 78652 := 
by
  sorry

end f_f_of_2_l752_752360


namespace find_sin_alpha_find_sin_beta_l752_752239

-- Define the problem conditions
variables {α β θ : ℝ}
variable h_alpha : α ∈ Ioo (-π/2) 0
variable h1 : cos α - sin θ + sin α * sin θ = 0
variable h2 : cos α + cos θ - sin α * cos θ = 0
variable h_cos_ab : cos (α - β) = 2 / 3
variables h_ab_lower : -π / 2 < α - β
variables h_ab_upper : α - β < 0

-- Prove part (I): sin α = -1/3
theorem find_sin_alpha (h1 : cos α - sin θ + sin α * sin θ = 0)
                       (h2 : cos α + cos θ - sin α * cos θ = 0)
                       (h_alpha : α ∈ Ioo (-π/2) 0) :
  sin α = -1/3 :=
by
  sorry

-- Prove part (II): sin β = (2 * sqrt 10 - 2) / 9, given the previous results
theorem find_sin_beta (h_alpha : α ∈ Ioo (-π/2) 0)
                      (h1 : cos α - sin θ + sin α * sin θ = 0)
                      (h2 : cos α + cos θ - sin α * cos θ = 0)
                      (h_cos_ab : cos (α - β) = 2 / 3)
                      (h_ab_lower : -π / 2 < α - β)
                      (h_ab_upper : α - β < 0)
                      (h_sin_alpha : sin α = -1/3) :
  sin β = (2 * sqrt 10 - 2) / 9 :=
by
  sorry

end find_sin_alpha_find_sin_beta_l752_752239


namespace tan_identity_l752_752247

variable (α β : ℝ)

theorem tan_identity (h₁ : 0 < α ∧ α < π / 2) (h₂ : 0 < β ∧ β < π / 2) (h₃ : Real.sin (2 * α) = 2 * Real.sin (2 * β)) : 
  Real.tan (α + β) = 3 * Real.tan (α - β) := 
by 
  sorry

end tan_identity_l752_752247


namespace translation_three_units_left_l752_752440

-- Define the original function
def original_function (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the transformed function after shifting 3 units to the left
def transformed_function (x : ℝ) : ℝ := 3 * x^2 + 20 * x + 28

theorem translation_three_units_left :
  (∀ x : ℝ, transformed_function x = original_function (x + 3)) →
  let a := 3; let b := 20; let c := 28 in
  a + b + c = 51 :=
by
  intros h
  have : ∀ x : ℝ, 3 * x^2 + 20 * x + 28 = 3 * (x + 3) ^ 2 + 2 * (x + 3) - 5 := 
    by
      intro x
      calc
        3 * x^2 + 20 * x + 28
            = 3 * x^2 + 18 * x + 27 + 2 * x + 6 - 5 : by sorry
        ... = 3 * (x + 3) ^ 2 + 2 * (x + 3) - 5 : by sorry
  have : original_function (x + 3) = transformed_function x := sorry
  sorry

end translation_three_units_left_l752_752440


namespace number_of_valid_three_digit_numbers_l752_752940

theorem number_of_valid_three_digit_numbers : 
  (∃ odd_numbers even_numbers, 
    (set.to_finset {1, 3, 5, 7, 9} = odd_numbers) ∧ 
    (set.to_finset {2, 4, 6, 8} = even_numbers) ∧ 
    ∀ digits, 
    (digits ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
    (set.card digits = 3) ∧ 
    (set.card (digits ∩ odd_numbers) ≥ 1) ∧ 
    (set.card (digits ∩ even_numbers) ≥ 1) →
    set.card (set.filter 
      (λ n, 
        let d := n / 100 % 10, t := n / 10 % 10, u := n % 10 
        in 
        n < 1000 ∧ 
        d ≠ t ∧ d ≠ u ∧ t ≠ u ∧ 
        (set.to_finset ([d, t, u].to_list) ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
        (∃ d' t' u', {d', t', u'} = set.to_finset ([d, t, u].to_list) ∧ 
         (d' ∈ odd_numbers ∨ t' ∈ odd_numbers ∨ u' ∈ odd_numbers) ∧ 
         (d' ∈ even_numbers ∨ t' ∈ even_numbers ∨ u' ∈ even_numbers)) ) 
      {n : ℕ | n < 1000} ) = 420
  ) sorry

end number_of_valid_three_digit_numbers_l752_752940


namespace smallest_number_l752_752513

theorem smallest_number : ∀ (a b c d : ℝ),
  a = 0 → b = -2 → c = 1 → d = -real.sqrt 3 →
  (b < a) → (b < c) → (b < d) → 
  b = -2 :=
by
  intros a b c d ha hb hc hd hba hbc hbd
  sorry

end smallest_number_l752_752513


namespace centroid_sum_of_squares_l752_752341

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end centroid_sum_of_squares_l752_752341


namespace z_in_second_quadrant_l752_752285

-- Define the complex number z and the given condition
def z : ℂ := -1 / 2 + (3 / 2) * I

-- The given condition in the problem
def condition : Prop := (1 + 2*I) / z = 1 - I

-- The goal is to show that z lies in the second quadrant
theorem z_in_second_quadrant (h : condition) : -1 / 2 + 3 / 2 * I.im > 0 ∧ -1 / 2 < 0 :=
by
  sorry

end z_in_second_quadrant_l752_752285


namespace find_k_plus_l_l752_752418

-- Definitions
variable (B N C O : Type) [MetricSpace B] [MetricSpace N] [MetricSpace C] [MetricSpace O]
variable (triangle_BNC : Triangle B N C)
variable (angle_NBC : Angle N B C) (perimeter_BNC : ℕ)
variable (center_O : MetricSpace O) (radius_circle : ℕ)

-- Conditions
axiom perimeter_triangle_BNC : perimeter_BNC = 180
axiom right_angle_NBC : is_right_angle angle_NBC
axiom circle_conditions : (∃ O : MetricSpace O, center_O = O ∧ radius_circle = 23 ∧ tangent circ center_O radius_circle ⟷ (tangent (line_segment B N) (line_segment B C) O radius_circle) ∧ tangent circ center_O radius_circle ⟷ (tangent (line_segment N C) (line_segment B C) O radius_circle))

-- The target statement to prove 
theorem find_k_plus_l : 
  let k := 113, l := 3 in
  relatively_prime k l → perimeter_BNC = 180 → is_right_angle angle_NBC → 
  ∃ (OB : ℚ), OB = k / l ∧ k + l = 116 := 
sorry

end find_k_plus_l_l752_752418


namespace dot_product_of_a_and_c_is_4_l752_752275

def vector := (ℝ × ℝ)

def a : vector := (1, -2)
def b : vector := (-3, 2)

def three_a : vector := (3 * 1, 3 * -2)
def two_b_minus_a : vector := (2 * -3 - 1, 2 * 2 - -2)

def c : vector := (-(-three_a.fst + two_b_minus_a.fst), -(-three_a.snd + two_b_minus_a.snd))

def dot_product (u v : vector) : ℝ := u.fst * v.fst + u.snd * v.snd

theorem dot_product_of_a_and_c_is_4 : dot_product a c = 4 := 
by
  sorry

end dot_product_of_a_and_c_is_4_l752_752275


namespace Ginger_water_usage_l752_752941

-- Define constants and initial conditions
def hours := 8
def water_bottle := 2 -- Ginger's water bottle capacity in cups

-- Define the drinking pattern
def drinking_pattern : Fin hours → ℝ
| 0 => 1
| 1 => 1.5
| 2 => 2
| n+3 => drinking_pattern n + 0.5

-- Define water requirements for each type of plant
def water_per_plant : Fin 3 → ℝ
| 0 => 3
| 1 => 4
| 2 => 5

-- Define the number of plants for each type
def num_plants : Fin 3 → ℕ
| 0 => 2
| 1 => 3
| 2 => 4

-- SumN is a helper function to calculate the sum
def sumN {α : Type _} [AddMonoid α] (f : Fin n → α) : α :=
  (Finset.univ : Finset (Fin n)).sum f

-- Total water Ginger drank in 8 hours
def total_water_drunk : ℝ :=
  sumN drinking_pattern

-- Total water used for watering plants
def total_water_plants : ℝ :=
  sumN (λ i => water_per_plant i * num_plants i)

-- Total water used
def total_water_used : ℝ :=
  total_water_drunk + total_water_plants

-- The proof statement
theorem Ginger_water_usage : total_water_used = 60 := by
  sorry

end Ginger_water_usage_l752_752941


namespace correct_propositions_l752_752871

section proposition_proof

variables {z : ℂ} {a : ℝ} {f : ℝ → ℝ} {g : ℝ → ℝ → ℝ} {F : ℝ → ℝ → ℝ}
  {a_n : ℕ → ℝ} (p q : ℝ)

-- Proposition (1)
def proposition_1 : Prop := ∃ z : ℂ, abs (z - 2) - abs (z + 2) = 1

-- Proposition (2)
def proposition_2 : Prop := ∀ a : ℝ, (Complex.ofReal (a^2) + Complex.i * a)

-- Proposition (3)
def proposition_3 : Prop := (∀ n m : ℕ, n < m → a_n ≤ a_m) → (∀ x1 x2 : ℝ, x1 < x2 → f x1 ≤ f x2)

-- Proposition (4)
def proposition_4 : Prop := ∀ (x y : ℝ), g (x-1) (y-2) = 0 ↔ g x y = 0

-- Proposition (5)
def proposition_5 : Prop := ∀ (F : ℝ → ℝ → ℝ), (F = (fun x y => x^2/a^2 + y^2/b^2 - 1)) → (∃ p q : ℝ, F (p*x) (q*y) = 0)

theorem correct_propositions : ¬ proposition_1 ∧ proposition_2 ∧ proposition_3 ∧ proposition_4 ∧ proposition_5 :=
by sorry

end proposition_proof

end correct_propositions_l752_752871


namespace value_of_x_l752_752291

theorem value_of_x
  (x : ℝ)
  (h1 : x = 0)
  (h2 : x^2 - 1 ≠ 0) :
  (x = 0) ↔ (x ^ 2 - 1 ≠ 0) :=
by
  sorry

end value_of_x_l752_752291


namespace sum_of_cubes_l752_752109

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752109


namespace gcd_f_x_l752_752246

-- Define that x is a multiple of 23478
def is_multiple_of (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

-- Define the function f(x)
noncomputable def f (x : ℕ) : ℕ := (2 * x + 3) * (7 * x + 2) * (13 * x + 7) * (x + 13)

-- Assert the proof problem
theorem gcd_f_x (x : ℕ) (h : is_multiple_of x 23478) : Nat.gcd (f x) x = 546 :=
by 
  sorry

end gcd_f_x_l752_752246


namespace max_value_of_f_l752_752265

noncomputable def roots (a b : ℝ) :=
  1 + 2 = a ∧ 1 * 2 = b

noncomputable def f (a b x : ℝ) :=
  (a - 1) * Real.sqrt (x - 3) + (b - 1) * Real.sqrt (4 - x)

theorem max_value_of_f (a b : ℝ) (h : roots a b) :
  ∃ x ∈ set.Icc 3 4, f a b x = Real.sqrt 5 := sorry

end max_value_of_f_l752_752265


namespace least_number_remainder_l752_752091

theorem least_number_remainder (n : ℕ) :
  (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4) → n = 256 :=
by
  sorry

end least_number_remainder_l752_752091


namespace maximum_value_ratio_l752_752663

theorem maximum_value_ratio (a b : ℝ) (h1 : a + b - 2 ≥ 0) (h2 : b - a - 1 ≤ 0) (h3 : a ≤ 1) :
  ∃ x, x = (a + 2 * b) / (2 * a + b) ∧ x ≤ 7/5 := sorry

end maximum_value_ratio_l752_752663


namespace line_intersects_circle_l752_752383

variable (x₀ y₀ r : Real)

theorem line_intersects_circle (h : x₀^2 + y₀^2 > r^2) : 
  ∃ p : ℝ × ℝ, (p.1^2 + p.2^2 = r^2) ∧ (x₀ * p.1 + y₀ * p.2 = r^2) := by
  sorry

end line_intersects_circle_l752_752383


namespace triangle_problem_l752_752702

-- Define the conditions for the triangle and the given properties
variables {A B C a b c : ℝ} (h1 : c = 3) (h2 : sin (C - π / 6) * cos C = 1 / 4)
           (h3 : sin B = 2 * sin A)

-- Problem to prove
theorem triangle_problem (h1 : c = 3) 
                         (h2 : sin (C - π / 6) * cos C = 1 / 4) 
                         (h3 : sin B = 2 * sin A) :
  (C = π / 3) ∧ (a = sqrt 3 ∧ b = 2 * sqrt 3 ∧ a + b + c = 3 + 3 * sqrt 3) :=
by sorry

end triangle_problem_l752_752702


namespace locus_of_vertex_C_is_ellipse_with_focus_A_l752_752157

-- Define the basic elements of the problem
def A : ℝ := 0
def B : ℝ := 2
def AB : ℝ := B - A  -- Length of side AB
def median_length : ℝ := 3 / 2

-- The theorem that we need to prove
theorem locus_of_vertex_C_is_ellipse_with_focus_A :
  ∀ C : ℝ, ∃ D : ℝ, (D - A)^2 + (D - A)^2 = median_length^2 ∧ 
             (C - D)^2 = (AB / 2)^2 ∧ 
             true := -- Additional conditions for an ellipse

begin
  -- Sorry is used as a placeholder to indicate that the proof is omitted.
  sorry
end

end locus_of_vertex_C_is_ellipse_with_focus_A_l752_752157


namespace variance_of_2ξ_plus_3_l752_752254

-- Given that the variance of ξ is 2
def variance_ξ : ℝ := 2

-- Define the random variable ξ (details about ξ itself are abstracted since only variance matters)
axiom ξ : Type

-- Define the variance function D for a random variable
def D (x : Type) : ℝ

-- Given the property of variance: D(a*ξ + b) = a^2 * D(ξ)
axiom variance_property (a b : ℝ) (x : Type) : D (a • x + b) = a^2 * D x

-- The proof problem: Given that D(ξ) = 2, prove that D(2 • ξ + 3) = 8
theorem variance_of_2ξ_plus_3 (h : D ξ = variance_ξ) : D (2 • ξ + 3) = 8 :=
by
  sorry

end variance_of_2ξ_plus_3_l752_752254


namespace largest_in_set_l752_752021

theorem largest_in_set (a : ℤ) (h : a = -3) : 
  let s := {-2 * a, 5 * a, 36 / a, a ^ 3, 2} in 
  6 ∈ s ∧ ∀ x ∈ s, x ≤ 6 := 
begin
  sorry
end

end largest_in_set_l752_752021


namespace smallest_number_l752_752514

theorem smallest_number : ∀ (a b c d : ℝ),
  a = 0 → b = -2 → c = 1 → d = -real.sqrt 3 →
  (b < a) → (b < c) → (b < d) → 
  b = -2 :=
by
  intros a b c d ha hb hc hd hba hbc hbd
  sorry

end smallest_number_l752_752514


namespace four_mutually_acquainted_l752_752673

theorem four_mutually_acquainted (G : SimpleGraph (Fin 9)) 
  (h : ∀ (s : Finset (Fin 9)), s.card = 3 → ∃ (u v : Fin 9), u ∈ s ∧ v ∈ s ∧ G.Adj u v) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (u v : Fin 9), u ∈ s → v ∈ s → G.Adj u v :=
by
  sorry

end four_mutually_acquainted_l752_752673


namespace simon_stamps_received_l752_752019

theorem simon_stamps_received (initial_stamps total_stamps received_stamps : ℕ) (h1 : initial_stamps = 34) (h2 : total_stamps = 61) : received_stamps = 27 :=
by
  sorry

end simon_stamps_received_l752_752019


namespace eggs_in_each_basket_l752_752691

theorem eggs_in_each_basket :
  ∃ n, n ≥ 5 ∧ n ∣ 30 ∧ n ∣ 45 ∧ (n = 5 ∨ n = 15) :=
by {
  use 15,
  split,
  { exact le_of_eq rfl },
  split,
  { norm_num },
  split,
  { norm_num },
  right,
  refl,
}

end eggs_in_each_basket_l752_752691


namespace find_a_value_l752_752033

-- Given values
def month_code : List ℝ := [1, 2, 3, 4, 5]
def prices (a : ℝ) : List ℝ := [0.5, a, 1, 1.4, 1.5]

-- Linear regression equation parameters
def lin_reg_slope : ℝ := 0.28
def lin_reg_intercept : ℝ := 0.16

-- Average function
def average (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Proof statement
theorem find_a_value (a : ℝ) (h : (average (prices a)) = lin_reg_slope * (average month_code) + lin_reg_intercept) : a = 0.6 :=
  sorry

end find_a_value_l752_752033


namespace complex_problem_find_a_b_l752_752641

theorem complex_problem (i : ℂ) (h_imag : i * i = -1) :
  let z := (1 + i) * (1 + i) + (2 * i) / (1 + i) in
  z = 1 + 3 * i ∧ abs z = real.sqrt 10 :=
by {
  let z := (1 + i) * (1 + i) + (2 * i) / (1 + i),
  have : z = 1 + 3 * i,
  { sorry },
  have : abs z = real.sqrt 10,
  { sorry },
  exact ⟨this, this_1⟩
}

theorem find_a_b (i : ℂ) (h_imag : i * i = -1) :
  let z := 1 + 3 * i in
  ∃ a b : ℝ, (z * z + a * conj z + b = 2 + 3 * i) ∧ a = 1 ∧ b = 9 :=
by {
  let z := 1 + 3 * i,
  have : ∃ a b : ℝ, (z * z + a * conj z + b = 2 + 3 * i) ∧ a = 1 ∧ b = 9,
  { sorry },
  exact this
}

end complex_problem_find_a_b_l752_752641


namespace triangle_problem_l752_752668

-- Define a triangle with given parameters and properties
variables {A B C : ℝ}
variables {a b c : ℝ} (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C) 
variables (h_b2a : b = 2 * a)
variables (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3)

-- Prove the required angles and side length
theorem triangle_problem 
    (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
    (h_b2a : b = 2 * a)
    (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) :

    Real.cos C = -1/2 ∧ C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := 
by 
  sorry

end triangle_problem_l752_752668


namespace intersection_A_B_l752_752268

def A : Set ℝ := { x | -2 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = { x | 1 < x ∧ x < 2 } := by
  sorry

end intersection_A_B_l752_752268


namespace train_length_correct_l752_752862

noncomputable def length_of_train (train_passes_man: Nat → Nat) (train_crosses_platform: Nat → Nat) : Nat := by
  sorry

theorem train_length_correct (L passes_man in 8) (passes_platform in 20) (platform_length in 270) : 
  L = 180 :=
  sorry

end train_length_correct_l752_752862


namespace mabel_marble_ratio_l752_752511

variable (A K M : ℕ)

-- Conditions
def condition1 : Prop := A + 12 = 2 * K
def condition2 : Prop := M = 85
def condition3 : Prop := M = A + 63

-- The main statement to prove
theorem mabel_marble_ratio (h1 : condition1 A K) (h2 : condition2 M) (h3 : condition3 A M) : M / K = 5 :=
by
  sorry

end mabel_marble_ratio_l752_752511


namespace functional_equation_solution_l752_752926

noncomputable def f : ℚ → ℚ := sorry

theorem functional_equation_solution :
  (∀ x y : ℚ, f (f x + x * f y) = x + f x * y) →
  (∀ x : ℚ, f x = x) :=
by
  intro h
  sorry

end functional_equation_solution_l752_752926


namespace min_value_expression_l752_752707

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ m : ℝ, (m = 4 + 6 * Real.sqrt 2) ∧ 
  ∀ a b : ℝ, (0 < a) → (0 < b) → m ≤ (Real.sqrt ((a^2 + b^2) * (2*a^2 + 4*b^2))) / (a * b) :=
by sorry

end min_value_expression_l752_752707


namespace subset_of_sets_l752_752369

theorem subset_of_sets (a : ℝ) : 
  {0, -a} ⊆ {1, a - 2, 2 * a - 2} → a = 1 :=
by
  intros h
  sorry

end subset_of_sets_l752_752369


namespace final_silver_tokens_l752_752867

structure TokenCounts :=
  (red : ℕ)
  (blue : ℕ)

def initial_tokens : TokenCounts := { red := 100, blue := 50 }

def exchange_booth1 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red - 3, blue := tokens.blue + 2 }

def exchange_booth2 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red + 1, blue := tokens.blue - 3 }

noncomputable def max_exchanges (initial : TokenCounts) : ℕ × ℕ :=
  let x := 48
  let y := 47
  (x, y)

noncomputable def silver_tokens (x y : ℕ) : ℕ := x + y

theorem final_silver_tokens (x y : ℕ) (tokens : TokenCounts) 
  (hx : tokens.red = initial_tokens.red - 3 * x + y)
  (hy : tokens.blue = initial_tokens.blue + 2 * x - 3 * y) 
  (hx_le : tokens.red >= 3 → false)
  (hy_le : tokens.blue >= 3 → false) : 
  silver_tokens x y = 95 :=
by {
  sorry
}

end final_silver_tokens_l752_752867


namespace trigonometric_identity_proof_l752_752537

noncomputable def theta : ℝ := real.pi * 40 / 180

theorem trigonometric_identity_proof :
  (tan theta)^2 - (cos theta)^2 / ((tan theta)^2 * (cos theta)^2) = 2 * (sin theta)^2 - (sin theta)^4 :=
by
  sorry

end trigonometric_identity_proof_l752_752537


namespace intersection_non_empty_l752_752331

theorem intersection_non_empty (n : ℕ) (h : n > 3) (A : Fin n → Set (Fin 2))
  (hA : ∀ i j : Fin n, i ≠ j → (A i ∩ A j).card = 1) :
  ∃ x, ∀ i, x ∈ A i :=
sorry

end intersection_non_empty_l752_752331


namespace cost_of_second_type_of_rice_is_22_l752_752822

noncomputable def cost_second_type_of_rice (c1 : ℝ) (w1 : ℝ) (w2 : ℝ) (avg : ℝ) (total_weight : ℝ) : ℝ :=
  ((total_weight * avg) - (w1 * c1)) / w2

theorem cost_of_second_type_of_rice_is_22 :
  cost_second_type_of_rice 16 8 4 18 12 = 22 :=
by
  sorry

end cost_of_second_type_of_rice_is_22_l752_752822


namespace subset_condition_l752_752366

variable (a : ℝ)

def A : Set ℝ := {0, -a}
def B : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_condition : A a ⊆ B a ↔ a = 1 := by 
  sorry

end subset_condition_l752_752366


namespace inradius_of_AP_triangle_l752_752009

noncomputable theory

-- Let a, a + d, and a + 2d be the sides of the triangle
variables (a d : ℝ)

-- Let h be the altitude to the side a + d
variables (h : ℝ)

-- Let r be the radius of the inscribed circle
variables (r : ℝ)

-- Conditions: sides form an arithmetic progression
def sides_arithmetic_progression : Prop := (a + d) ≠ 0

-- Conclusion: the radius of the inscribed circle is 1/3 of the altitude
theorem inradius_of_AP_triangle (h_ne_zero : h ≠ 0) 
  (ps : sides_arithmetic_progression a d) :
  r = h / 3 :=
sorry

end inradius_of_AP_triangle_l752_752009


namespace ratio_turkeys_to_ducks_l752_752380

theorem ratio_turkeys_to_ducks (chickens ducks turkeys total_birds : ℕ)
  (h1 : chickens = 200)
  (h2 : ducks = 2 * chickens)
  (h3 : total_birds = 1800)
  (h4 : total_birds = chickens + ducks + turkeys) :
  (turkeys : ℚ) / ducks = 3 := by
sorry

end ratio_turkeys_to_ducks_l752_752380


namespace time_to_produce_one_item_l752_752495

theorem time_to_produce_one_item :
  (200 : ℕ) / (2 * 60 : ℕ) = 0.6 := 
sorry

end time_to_produce_one_item_l752_752495


namespace Rachel_homework_diff_l752_752740

theorem Rachel_homework_diff (math_pages reading_pages : ℕ) (h_math : math_pages = 9) (h_reading : reading_pages = 2) :
  math_pages - reading_pages = 7 := 
by
  rw [h_math, h_reading]
  norm_num
-- Use [sorry] here if needed

end Rachel_homework_diff_l752_752740


namespace constant_term_expansion_eq_160_l752_752660

theorem constant_term_expansion_eq_160 :
  let expr := (λ x : ℤ, (1/x + 2*x)^6) in 
  ∃ c : ℤ, c = 160 ∧ (∃ r : ℕ, 2 * r - 6 = 0 ∧ (∏ (i : ℕ) in finset.range 7, if i = r then 2 else if i = 0 then 1/x else x) = c) :=
sorry

end constant_term_expansion_eq_160_l752_752660


namespace fraction_product_equals_12_l752_752799

theorem fraction_product_equals_12 :
  (1 / 3) * (9 / 2) * (1 / 27) * (54 / 1) * (1 / 81) * (162 / 1) * (1 / 243) * (486 / 1) = 12 := 
by
  sorry

end fraction_product_equals_12_l752_752799


namespace isosceles_right_triangle_area_l752_752048

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = 6 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 18 := by 
  sorry

end isosceles_right_triangle_area_l752_752048


namespace common_root_and_arithmetic_reciprocal_l752_752620

variable {α : Type*} [LinearOrderedField α]

structure arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  (is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d)
  (nonzero : ∀ n : ℕ, a n ≠ 0)
  (d_nonzero : d ≠ 0)

theorem common_root_and_arithmetic_reciprocal (a : ℕ → α) (d : α) 
  (h_arith : arithmetic_sequence a d) :
  (∀ i : ℕ, ∃ p : α, p = -1 ∧ a i * p ^ 2 + 2 * a (i + 1) * p + a (i + 2) = 0) ∧
  (∃ (mi : ℕ → α) (c : α), (∀ i : ℕ, a i * (mi i) ^ 2 + 2 * a (i + 1) * mi i + a (i + 2) = 0) ∧
  (forall (i : ℕ), (mi i) ≠ -1) ∧
  (is_arithmetic (λ i, (1 / (mi i + 1))) (- (1 / 2)))) :=
by
  sorry

end common_root_and_arithmetic_reciprocal_l752_752620


namespace sum_of_cubes_l752_752097

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752097


namespace monitor_width_32_inches_l752_752143

noncomputable def monitor_width (diagonal : ℝ) (width_ratio : ℝ) (height_ratio : ℝ) : ℝ :=
  width_ratio * diagonal / Real.sqrt (width_ratio^2 + height_ratio^2)

theorem monitor_width_32_inches :
  monitor_width 32 16 9 ≈ 27.85 := 
by
  sorry

end monitor_width_32_inches_l752_752143


namespace autumn_pencils_l752_752525

theorem autumn_pencils : 
  ∀ (start misplaced broken found bought : ℕ),
  start = 20 →
  misplaced = 7 → 
  broken = 3 → 
  found = 4 → 
  bought = 2 → 
  start - misplaced - broken + found + bought = 16 :=
by 
  intros start misplaced broken found bought h_start h_misplaced h_broken h_found h_bought,
  rw [h_start, h_misplaced, h_broken, h_found, h_bought],
  linarith

end autumn_pencils_l752_752525


namespace num_solutions_abs_quadratic_eq_l752_752706

theorem num_solutions_abs_quadratic_eq (k : ℝ) : 
  (k < -5 / 4 ∨ -1 < k ∧ k < 1 ∨ k > 5 / 4) → 
  (∀ x : ℝ, ¬ (|x^2 - 1| = x + k)) ∧ 
  ((k = -5 / 4 ∨ k = 5 / 4) → ∃! x : ℝ, |x^2 - 1| = x + k) ∧ 
  ((-5 / 4 < k ∧ k ≤ -1) ∨ (1 ≤ k ∧ k < 5 / 4) → 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (|x1^2 - 1| = x1 + k ∧ |x2^2 - 1| = x2 + k)) := 
begin
  sorry
end

end num_solutions_abs_quadratic_eq_l752_752706


namespace triangle_construction_solutions_l752_752230

-- Definitions for the problem conditions
variables {P : Type} [euclidean_plane P] -- assume a Euclidean plane setting
variables {circle_k center_K : P} (radius_k : ℝ) -- circle k with center K and radius radius_k
variables {d_length : ℝ} -- segment length d
variables {P : P} -- point P

noncomputable def construct_triangle_ABC (A B C : P) : Prop :=
 ∃ (AB_eq_d : segment (A, B) = d_length)
   (AB_parallel_d : parallel (line (A, B)) (line_through d_length))
   (angle_bisector_ACB_through_P : angle_bisector (∠ A C B) P), 
   true

theorem triangle_construction_solutions :
  (d_length < 2 * radius_k → ∃ (A B C : P), construct_triangle_ABC A B C ∧ card (triangle_solutions A B C) ≤ 4) ∧
  (d_length = 2 * radius_k → ∃ (A B C : P), construct_triangle_ABC A B C ∧ card (triangle_solutions A B C) ≤ 2) ∧
  (d_length > 2 * radius_k → ¬ ∃ (A B C : P), construct_triangle_ABC A B C) :=
begin
  -- Proof will go here
  sorry
end

end triangle_construction_solutions_l752_752230


namespace extreme_value_range_of_a_ln_inequality_l752_752261

-- Define the function f as given in the problem
def f (a x : ℝ) : ℝ := 2 * a * real.log x + (a + 1) * x^2 + 1

-- Define the conditions and the problem statements
theorem extreme_value (a : ℝ) (h : a = -1/2) : ∃ x : ℝ, f a x = 3/2 := sorry

theorem range_of_a (h : ∀ x1 x2 : ℝ, x1 > x2 → 0 < (f 1 x1 - f 1 x2) / (x1 - x2) - (x1 + x2 + 4)) : 
  ∀ a : ℝ, a ≥ 1 := sorry

theorem ln_inequality (n : ℕ) (h1 : 1 < n) : real.log (n + 1) > ∑ i in finset.range n, 1/(i+2) := sorry

end extreme_value_range_of_a_ln_inequality_l752_752261


namespace integer_part_of_2011th_term_is_91_l752_752070

def sequence (n : ℕ) : ℝ :=
  if n = 1 then 105 else if n = 2 then 85 else (sequence (n - 1) + sequence (n - 2)) / 2

theorem integer_part_of_2011th_term_is_91 :
  ⌊sequence 2011⌋ = 91 :=
by
  sorry

end integer_part_of_2011th_term_is_91_l752_752070


namespace purely_imaginary_z_l752_752971

open Complex

theorem purely_imaginary_z (z : ℂ) (h1 : ∀ z, Im z = z ∧ Re z = 0) 
(h2 : ∀ z, Im ((z + 2)^2 + 8 * I) = ((z + 2)^2 + 8 * I) ∧ Re ((z + 2)^2 + 8 * I) = 0) : 
  z = 2 * I := 
sorry

end purely_imaginary_z_l752_752971


namespace magazines_sold_l752_752324

theorem magazines_sold (total_sold : Float) (newspapers_sold : Float) (magazines_sold : Float)
  (h1 : total_sold = 425.0)
  (h2 : newspapers_sold = 275.0) :
  magazines_sold = total_sold - newspapers_sold :=
by
  sorry

#check magazines_sold

end magazines_sold_l752_752324


namespace circle_diameter_PQ_fixed_point_l752_752954

section ellipse_problem

variables (a b c : ℝ)
variables (h1 : a > b)
variables (h2 : b > 0)
variables (e : ℝ := sqrt 2 / 2)
variables (h3 : c = a * e)
variables (h4 : a = sqrt (b^2 + c^2))
variables (h5 : b * c = 1)
variables (h6 : ∀ (T : ℝ × ℝ), T.1^2 / a^2 + T.2^2 / b^2 = 1 → 
  let F1 : ℝ × ℝ := (-c, 0)
  let F2 : ℝ × ℝ := (c, 0)
  in max_area T F1 F2)

lemma ellipse_standard_eq :
  a = sqrt 2 ∧ b = 1 ∧ c = 1 :=
  sorry

variables (M N : ℝ × ℝ)
variables (h7 : M.1^2 / a^2 + M.2^2 / b^2 = 1)
variables (h8 : N.1^2 / a^2 + N.2^2 / b^2 = 1)
variables (P Q : ℝ × ℝ)
variables (h9 : ∃ (k : ℝ), M.2 = k * M.1 + (1/2) ∧ N.2 = k * N.1 + (1/2))
variables (h10 : ∃ (xA : ℝ), A.1 = 0 ∧ A.2 = 1)
variables (A : ℝ × ℝ)
variables (h11 : ∃ (xM xN : ℝ), P = (xM, 0) ∧ Q = (xN, 0) ∧ M = (xM, 1) ∧ N = (xN, 1))

theorem circle_diameter_PQ_fixed_point (P Q : ℝ × ℝ) :
  ∃ (k M N : ℝ × ℝ), M.1^2 / a^2 + M.2^2 / b^2 = 1 ∧ N.1^2 / a^2 + N.2^2 / b^2 = 1 ∧
  circle (P, Q).diameter contains (0, sqrt 6) ∧ circle (P, Q).diameter contains (0, -sqrt 6) :=
sorry

end ellipse_problem

end circle_diameter_PQ_fixed_point_l752_752954


namespace Family_A_saved_Family_B_saved_Family_C_saved_Family_D_saved_l752_752294

-- Define percentage saved function
def percentage_saved (passengers : ℕ) (planned_spending cost_per_orange : ℝ) : ℝ :=
  let total_cost := passengers * cost_per_orange
  (total_cost / planned_spending) * 100

-- Define theorems for each family
theorem Family_A_saved :
  percentage_saved 4 20 1.5 = 30 := by
  -- proving logic here
  sorry

theorem Family_B_saved :
  percentage_saved 6 22 2 = 54.54545454545455 := by
  -- proving logic here 
  sorry

theorem Family_C_saved :
  percentage_saved 5 18 1.8 = 50 := by
  -- proving logic here 
  sorry

theorem Family_D_saved :
  percentage_saved 3 12 2.2 = 55 := by
  -- proving logic here 
  sorry

end Family_A_saved_Family_B_saved_Family_C_saved_Family_D_saved_l752_752294


namespace smallest_cube_edge_length_l752_752072

noncomputable def edge_length_smallest_cube (s₁ : ℝ) : ℝ :=
  let s₂ := s₁ / Real.sqrt 3 in
  let s₃ := s₂ / Real.sqrt 3 in
  let s₄ := s₃ / Real.sqrt 3 in
  let s₅ := s₄ / Real.sqrt 3 in
  s₅

theorem smallest_cube_edge_length (s₁ : ℝ) :
  edge_length_smallest_cube s₁ = s₁ / 9 :=
by sorry

end smallest_cube_edge_length_l752_752072


namespace fraction_simplification_l752_752175

theorem fraction_simplification (x : ℝ) (h: x ≠ 1) : (5 * x / (x - 1) - 5 / (x - 1)) = 5 := 
sorry

end fraction_simplification_l752_752175


namespace zero_matrix_transformation_l752_752677

open Matrix

theorem zero_matrix_transformation (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ)
  (h_row : ∀ i : Fin n, (∑ j, A i j) = 0)
  (h_col : ∀ j : Fin n, (∑ i, A i j) = 0) :
  ∃ (k : ℕ) (operations : Fin k → (Fin n) × (Fin n)), 
  (let apply_operation := 
    λ (A : Matrix (Fin n) (Fin n) ℝ) (op : (Fin n) × (Fin n)), 
    let i := op.1, j := op.2 in
    (λ A, A.update_column j (λ v, v + A i) - A.update_column j (λ v, v  - A i)) in
    (nat.iterate apply_operation k A = 0)) := 
begin
  sorry
end

end zero_matrix_transformation_l752_752677


namespace largest_n_l752_752523

noncomputable def a (n : ℕ) (x : ℤ) : ℤ := 2 + (n - 1) * x
noncomputable def b (n : ℕ) (y : ℤ) : ℤ := 3 + (n - 1) * y

theorem largest_n {n : ℕ} (x y : ℤ) :
  a 1 x = 2 ∧ b 1 y = 3 ∧ 3 * a 2 x < 2 * b 2 y ∧ a n x * b n y = 4032 →
  n = 367 :=
sorry

end largest_n_l752_752523


namespace xy_expr_value_l752_752225

variable (x y : ℝ)

-- Conditions
def cond1 : Prop := x - y = 2
def cond2 : Prop := x * y = 3

-- Statement to prove
theorem xy_expr_value (h1 : cond1 x y) (h2 : cond2 x y) : x * y^2 - x^2 * y = -6 :=
by
  sorry

end xy_expr_value_l752_752225


namespace complex_problem_find_a_b_l752_752640

theorem complex_problem (i : ℂ) (h_imag : i * i = -1) :
  let z := (1 + i) * (1 + i) + (2 * i) / (1 + i) in
  z = 1 + 3 * i ∧ abs z = real.sqrt 10 :=
by {
  let z := (1 + i) * (1 + i) + (2 * i) / (1 + i),
  have : z = 1 + 3 * i,
  { sorry },
  have : abs z = real.sqrt 10,
  { sorry },
  exact ⟨this, this_1⟩
}

theorem find_a_b (i : ℂ) (h_imag : i * i = -1) :
  let z := 1 + 3 * i in
  ∃ a b : ℝ, (z * z + a * conj z + b = 2 + 3 * i) ∧ a = 1 ∧ b = 9 :=
by {
  let z := 1 + 3 * i,
  have : ∃ a b : ℝ, (z * z + a * conj z + b = 2 + 3 * i) ∧ a = 1 ∧ b = 9,
  { sorry },
  exact this
}

end complex_problem_find_a_b_l752_752640


namespace autumn_pencils_l752_752524

theorem autumn_pencils : 
  ∀ (start misplaced broken found bought : ℕ),
  start = 20 →
  misplaced = 7 → 
  broken = 3 → 
  found = 4 → 
  bought = 2 → 
  start - misplaced - broken + found + bought = 16 :=
by 
  intros start misplaced broken found bought h_start h_misplaced h_broken h_found h_bought,
  rw [h_start, h_misplaced, h_broken, h_found, h_bought],
  linarith

end autumn_pencils_l752_752524


namespace age_weight_not_proportional_l752_752025

theorem age_weight_not_proportional (age weight : ℕ) : ¬(∃ k, ∀ (a w : ℕ), w = k * a → age / weight = k) :=
by
  sorry

end age_weight_not_proportional_l752_752025


namespace minimum_maximum_product_l752_752057

def numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define all possible ways to divide the set into three groups of three.
def possible_groups_3x3 := sorry -- Placeholder for the actual definition

-- Define the maximum product of each grouping.
def max_product_of_grouping (g : possible_groups_3x3) : Nat := sorry -- Placeholder for the actual definition

theorem minimum_maximum_product :
  ∃ g : possible_groups_3x3, max_product_of_grouping g = 72 := sorry

end minimum_maximum_product_l752_752057


namespace marble_ratio_l752_752585

theorem marble_ratio (total_marbles red_marbles dark_blue_marbles : ℕ) (h_total : total_marbles = 63) (h_red : red_marbles = 38) (h_blue : dark_blue_marbles = 6) :
  (total_marbles - red_marbles - dark_blue_marbles) / red_marbles = 1 / 2 := by
  sorry

end marble_ratio_l752_752585


namespace square_perimeter_l752_752757

theorem square_perimeter (s : ℝ) (h : s^2 = s) (h₀ : s ≠ 0) : 4 * s = 4 :=
by {
  have s_eq_1 : s = 1 := by {
    field_simp [h],
    exact h₀,
    linarith,
  },
  rw s_eq_1,
  ring,
}

end square_perimeter_l752_752757


namespace retailer_profit_percentage_l752_752520

theorem retailer_profit_percentage 
  (CP MP SP : ℝ)
  (hCP : CP = 100)
  (hMP : MP = CP + 0.65 * CP)
  (hSP : SP = MP - 0.25 * MP)
  : ((SP - CP) / CP) * 100 = 23.75 := 
sorry

end retailer_profit_percentage_l752_752520


namespace neg_proposition_l752_752776

theorem neg_proposition :
  (¬(∀ x : ℕ, x^3 > x^2)) ↔ (∃ x : ℕ, x^3 ≤ x^2) := 
sorry

end neg_proposition_l752_752776


namespace unit_vectors_dot_product_zero_l752_752637

variables (a b : ℝ^3)
variables (a_unit : ∥a∥ = 1)
variables (b_unit : ∥b∥ = 1)
variables (cos_theta : a.dot b = 1/2)

theorem unit_vectors_dot_product_zero :
  (2 * a - b).dot b = 0 :=
sorry

end unit_vectors_dot_product_zero_l752_752637


namespace bertha_family_no_daughters_l752_752886

/-- Definition of the family situation. -/
structure BerthaFamily where
  daughters : ℕ -- Number of Bertha's daughters
  granddaughters : ℕ -- Number of Bertha's granddaughters
  total_daughters_and_granddaughters : ℕ -- Total number of Bertha's daughters and granddaughters
  no_great_granddaughters : bool -- Indicates Bertha has no great granddaughters

noncomputable def BerthaFamilyExample : BerthaFamily := {
  daughters := 5,
  granddaughters := 20, -- 25 - 5
  total_daughters_and_granddaughters := 25,
  no_great_granddaughters := tt
}

/-- Proof that the number of Bertha's daughters and granddaughters with no daughters is 21. -/
theorem bertha_family_no_daughters (family : BerthaFamily) (conditions : 
  family.daughters = 5 ∧ 
  family.total_daughters_and_granddaughters = 25 ∧ 
  family.no_great_granddaughters = tt) : 
  family.total_daughters_and_granddaughters - (family.granddaughters / family.daughters) = 21 := 
by
  sorry

end bertha_family_no_daughters_l752_752886


namespace minimum_value_of_sum_l752_752241

noncomputable def left_focus (a b c : ℝ) : ℝ := -c 

noncomputable def right_focus (a b c : ℝ) : ℝ := c

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
magnitude (q.1 - p.1, q.2 - p.2)

def hyperbola_eq (x y : ℝ) : Prop :=
  (x^2 / 4) - (y^2 / 12) = 1

def P_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola_eq P.1 P.2

theorem minimum_value_of_sum (P : ℝ × ℝ) (A : ℝ × ℝ) (F F' : ℝ × ℝ) (a b c : ℝ)
  (h1 : F = (-c, 0)) (h2 : F' = (c, 0)) (h3 : A = (1, 4)) (h4 : 2 * a = 4)
  (h5 : c^2 = a^2 + b^2) (h6 : P_on_hyperbola P) :
  (|distance P F| + |distance P A|) ≥ 9 :=
sorry

end minimum_value_of_sum_l752_752241


namespace number_of_subsets_of_M_sum_of_elements_in_all_subsets_of_M_l752_752991

theorem number_of_subsets_of_M (M : Finset ℕ) (hM : M = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
    M.card = 10 → M.powerset.card = 2^10 := 
by
    intro hM_card
    rw hM
    rw Finset.card_eq_sum_ones (Finset.univ : Finset M)
    sorry

theorem sum_of_elements_in_all_subsets_of_M (M : Finset ℕ) (hM : M = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
    ∑ (S : Finset ℕ) in M.powerset, (S.sum id) = 55 * 2^9 :=
by
    rw hM
    sorry

end number_of_subsets_of_M_sum_of_elements_in_all_subsets_of_M_l752_752991


namespace spencer_sessions_per_day_l752_752457

theorem spencer_sessions_per_day :
  let jumps_per_minute := 4
  let minutes_per_session := 10
  let jumps_per_session := jumps_per_minute * minutes_per_session
  let total_jumps := 400
  let days := 5
  let jumps_per_day := total_jumps / days
  let sessions_per_day := jumps_per_day / jumps_per_session
  sessions_per_day = 2 :=
by
  sorry

end spencer_sessions_per_day_l752_752457


namespace exists_root_interval_l752_752770

noncomputable def f (x : ℝ) : ℝ := x - 2 + Real.log x

theorem exists_root_interval :
  (∀ x y : ℝ, x < y → f x < f y) → f 1 < 0 → f 2 > 0 → ∃ c ∈ set.Ioo 1 2, f c = 0 :=
by
  intros h_mono h_f1 h_f2
  sorry

end exists_root_interval_l752_752770


namespace max_area_sheep_pen_l752_752857

theorem max_area_sheep_pen : 
  (∃ (l w : ℝ), l + 2 * w = 30 ∧ l * w = 112.5) :=
by {
  sorry,
}

end max_area_sheep_pen_l752_752857


namespace ones_digit_of_9_pow_27_l752_752797

-- Definitions representing the cyclical pattern
def ones_digit_of_9_power (n : ℕ) : ℕ :=
  if n % 2 = 1 then 9 else 1

-- The problem statement to be proven
theorem ones_digit_of_9_pow_27 : ones_digit_of_9_power 27 = 9 := 
by
  -- the detailed proof steps are omitted
  sorry

end ones_digit_of_9_pow_27_l752_752797


namespace probability_of_selecting_storybook_l752_752664

theorem probability_of_selecting_storybook (reference_books storybooks picture_books : ℕ) 
  (h1 : reference_books = 5) (h2 : storybooks = 3) (h3 : picture_books = 2) :
  (storybooks : ℚ) / (reference_books + storybooks + picture_books) = 3 / 10 :=
by {
  sorry
}

end probability_of_selecting_storybook_l752_752664


namespace z_squared_in_second_quadrant_l752_752937

def z : ℂ := Complex.ofReal (Real.cos (75 * Real.pi / 180)) + Complex.i * Complex.ofReal (Real.sin (75 * Real.pi / 180))

theorem z_squared_in_second_quadrant : 
  let z2 := z * z in Im z2 > 0 ∧ Re z2 < 0 :=
sorry

end z_squared_in_second_quadrant_l752_752937


namespace inequality_proof_l752_752711

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
by
  sorry

end inequality_proof_l752_752711


namespace polynomial_P_at_zero_l752_752712

variable (b c : ℝ)

noncomputable def P : ℝ → ℝ :=
  λ x, x^2 + b * x + c

theorem polynomial_P_at_zero :
  (P b c (0:ℝ) = -(3/2)) :=
  sorry

end polynomial_P_at_zero_l752_752712


namespace domain_of_f_3x_l752_752662

variable (f : ℝ → ℝ)
variable (dom_f : Set.Ioc (1 / 3) 1)

theorem domain_of_f_3x (x : ℝ) : (-1 < x ∧ x ≤ 0) ↔ (f (3 ^ x)) ∈ dom_f :=
sorry

end domain_of_f_3x_l752_752662


namespace simplify_to_linear_binomial_l752_752215

theorem simplify_to_linear_binomial (k : ℝ) (x : ℝ) : 
  (-3 * k * x^2 + x - 1) + (9 * x^2 - 4 * k * x + 3 * k) = 
  (1 - 4 * k) * x + (3 * k - 1) → 
  k = 3 := by
  sorry

end simplify_to_linear_binomial_l752_752215


namespace true_proposition_l752_752960

-- Definition of proposition p
def p : Prop := ∀ (x : ℝ), 2^x > 0

-- Definition of proposition q
def q : Prop := (∀ (x : ℝ), x > 1 → x > 2) ∧ (∃ (x : ℝ), x > 2 ∧ x ≤ 1)

-- The theorem to prove 
theorem true_proposition : p ∧ ¬ q := 
by 
  sorry

end true_proposition_l752_752960


namespace natural_number_inequality_l752_752729

theorem natural_number_inequality
  (m n a b k l : ℕ)
  (h1 : (m : ℝ) / n < (a : ℝ) / b)
  (h2 : (a : ℝ) / b < (k : ℝ) / l)
  (h3 : |m * l - k * n| = 1) :
  b ≥ n + l := by
  sorry

end natural_number_inequality_l752_752729


namespace composite_numbers_infinitely_many_l752_752387

theorem composite_numbers_infinitely_many (k : ℤ) (hk : k ≠ 1) : ∃ᶠ n in at_top, ¬nat.prime (2^(2^n) + k) :=
sorry

end composite_numbers_infinitely_many_l752_752387


namespace min_value_of_a1_plus_a7_l752_752298

variable {a : ℕ → ℝ}
variable {a3 a5 : ℝ}

-- Conditions
def is_positive_geometric_sequence (a : ℕ → ℝ) := 
  ∀ n, a n > 0 ∧ (∃ r, ∀ i, a (i + 1) = a i * r)

def condition (a : ℕ → ℝ) (a3 a5 : ℝ) :=
  a 3 = a3 ∧ a 5 = a5 ∧ a3 * a5 = 64

-- Prove that the minimum value of a1 + a7 is 16
theorem min_value_of_a1_plus_a7
  (h1 : is_positive_geometric_sequence a)
  (h2 : condition a a3 a5) :
  ∃ a1 a7, a 1 = a1 ∧ a 7 = a7 ∧ (∃ (min_sum : ℝ), min_sum = 16 ∧ ∀ sum, sum = a1 + a7 → sum ≥ min_sum) :=
sorry

end min_value_of_a1_plus_a7_l752_752298


namespace tan_ratio_proof_l752_752591

theorem tan_ratio_proof (α : ℝ) (h : 5 * Real.sin (2 * α) = Real.sin 2) : 
  Real.tan (α + 1 * Real.pi / 180) / Real.tan (α - 1 * Real.pi / 180) = - 3 / 2 := 
sorry

end tan_ratio_proof_l752_752591


namespace evaluate_log_expression_l752_752196

theorem evaluate_log_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx_ne_one : x ≠ 1) (hy_ne_one : y ≠ 1) :
  log x (y^2) * log y (x^3) = 6 := 
by
  sorry

end evaluate_log_expression_l752_752196


namespace simplify_expression_l752_752748

theorem simplify_expression : (4 + 3) + (8 - 3 - 1) = 11 := by
  sorry

end simplify_expression_l752_752748


namespace minimal_isosceles_base_l752_752736

theorem minimal_isosceles_base 
  (α q : ℝ) 
  (a b c : ℝ)
  (hab_sum : a + b = q)
  (cos_law : c^2 = a^2 + b^2 - 2 * a * b * cos α) : 
  c ≥ q * sqrt (1 - cos α) / sqrt 2 :=
sorry

end minimal_isosceles_base_l752_752736


namespace decimal_125th_place_l752_752282

theorem decimal_125th_place (n : ℕ) (h : n = 125) : 
  (Nat.mod n 6 = 5) → (∃ s, s = "571428".to_list) → (s.get! 4 = '2') :=
by 
  intros h125 hs
  have hseq := hs h
  apply List.get_eq _ _ _ _ _
  · sorry

end decimal_125th_place_l752_752282


namespace percentage_of_b_l752_752475

variable (a b c p : ℝ)

theorem percentage_of_b (h1 : 0.06 * a = 12) (h2 : p * b = 6) (h3 : c = b / a) : 
  p = 6 / (200 * c) := by
  sorry

end percentage_of_b_l752_752475


namespace square_pentagon_intersections_l752_752751

-- Definitions based on the conditions
def square_inscribed_in_circle (CASH : Finset Point) (circle : Circle) : Prop := sorry
def regular_pentagon_inscribed_in_circle (MONEY : Finset Point) (circle : Circle) : Prop := sorry
def no_shared_vertices (CASH MONEY : Finset Point) : Prop := sorry

-- The theorem to prove
theorem square_pentagon_intersections
  (CASH : Finset Point) (MONEY : Finset Point) (circle : Circle) 
  (h1 : square_inscribed_in_circle CASH circle)
  (h2 : regular_pentagon_inscribed_in_circle MONEY circle)
  (h3 : no_shared_vertices CASH MONEY) :
  intersections CASH MONEY = 8 :=
sorry

end square_pentagon_intersections_l752_752751


namespace initial_action_figures_l752_752322

theorem initial_action_figures (x : ℕ) (h1 : x + 2 = 10) : x = 8 := 
by sorry

end initial_action_figures_l752_752322


namespace three_angles_difference_is_2pi_over_3_l752_752226

theorem three_angles_difference_is_2pi_over_3 (α β γ : ℝ) 
    (h1 : 0 ≤ α) (h2 : α ≤ β) (h3 : β < γ) (h4 : γ ≤ 2 * Real.pi)
    (h5 : Real.cos α + Real.cos β + Real.cos γ = 0)
    (h6 : Real.sin α + Real.sin β + Real.sin γ = 0) :
    β - α = 2 * Real.pi / 3 :=
sorry

end three_angles_difference_is_2pi_over_3_l752_752226


namespace min_value_expression_l752_752703

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (min (a^4 + b^4 + (1 / (a + b)^4)) (a' b' : ℝ) : ((0 < a') ∧ (0 < b'))) = (√2 / 2) :=
sorry

end min_value_expression_l752_752703


namespace domain_of_f_range_of_f_decreasing_on_interval_l752_752980

noncomputable def f (x : ℝ) : ℝ := 1 / x - 2

theorem domain_of_f : SetOf (λ x, f x) = Set.Union (Set.Ioo Float.minf 0) (Set.Ioo 0 Float.inf) := sorry

theorem range_of_f : SetOf (λ y, ∃ x, f x = y) = Set.Union (Set.Ioo Float.minf (-2)) (Set.Ioo (-2) Float.inf) := sorry

theorem decreasing_on_interval (x₁ x₂ : ℝ) (h₀ : 0 < x₁) (hx₁₂ : x₁ < x₂) : f x₁ > f x₂ := sorry

end domain_of_f_range_of_f_decreasing_on_interval_l752_752980


namespace max_unit_squares_with_20_sticks_l752_752379

-- Define the conditions
def unit_length_sticks : ℕ := 20
def first_row_individual_squares : Prop := true
def subsequent_rows_form_rectangles : Prop := true

-- The theorem statement
theorem max_unit_squares_with_20_sticks (h1 : first_row_individual_squares) (h2 : subsequent_rows_form_rectangles) : 
  ∃ n : ℕ, n = 7 ∧ max_unit_squares unit_length_sticks = n :=
begin
  -- Use the given conditions and show the maximum number of unit squares
  sorry
end

end max_unit_squares_with_20_sticks_l752_752379


namespace vegetable_price_l752_752442

theorem vegetable_price (v : ℝ) 
  (beef_cost : ∀ (b : ℝ), b = 3 * v)
  (total_cost : 4 * (3 * v) + 6 * v = 36) : 
  v = 2 :=
by {
  -- The proof would go here.
  sorry
}

end vegetable_price_l752_752442


namespace darrel_pennies_l752_752185

theorem darrel_pennies (quarters dimes nickels : ℕ) (fee : ℝ) (received : ℝ) (penny_value : ℝ) (pennies : ℕ) :
  (quarters = 76) →
  (dimes = 85) →
  (nickels = 20) →
  (fee = 0.10) →
  (received = 27) →
  (penny_value = 0.01) →
  let total_value := 0.25 * quarters + 0.10 * dimes + 0.05 * nickels in
  let x := received / (1 - fee) in
  (x = 30) →
  (x - total_value = 1.50) →
  (pennies = (1.50 / penny_value : ℕ)) →
  pennies = 150 :=
by
  sorry

end darrel_pennies_l752_752185


namespace distribute_balls_ways_l752_752556

theorem distribute_balls_ways :
  let n := 7
  let total_ways := 2^n
  let one_person_none := 2
  let one_person_one := 2 * n
  ∃ ways : ℕ, ways = total_ways - one_person_none - one_person_one ∧ ways = 112 :=
by 
  let n := 7
  let total_ways := 2^n
  let one_person_none := 2
  let one_person_one := 2 * n
  let ways := total_ways - one_person_none - one_person_one
  use ways
  split
  . rfl
  . rfl

end distribute_balls_ways_l752_752556


namespace probability_all_same_color_is_correct_l752_752476

-- Definitions of quantities
def yellow_marbles := 3
def green_marbles := 7
def purple_marbles := 5
def total_marbles := yellow_marbles + green_marbles + purple_marbles

-- Calculation of drawing 4 marbles all the same color
def probability_all_yellow : ℚ := (yellow_marbles / total_marbles) * ((yellow_marbles - 1) / (total_marbles - 1)) * ((yellow_marbles - 2) / (total_marbles - 2)) * ((yellow_marbles - 3) / (total_marbles - 3))
def probability_all_green : ℚ := (green_marbles / total_marbles) * ((green_marbles - 1) / (total_marbles - 1)) * ((green_marbles - 2) / (total_marbles - 2)) * ((green_marbles - 3) / (total_marbles - 3))
def probability_all_purple : ℚ := (purple_marbles / total_marbles) * ((purple_marbles - 1) / (total_marbles - 1)) * ((purple_marbles - 2) / (total_marbles - 2)) * ((purple_marbles - 3) / (total_marbles - 3))

-- Total probability of drawing 4 marbles all the same color
def total_probability_same_color : ℚ := probability_all_yellow + probability_all_green + probability_all_purple

-- Theorem statement
theorem probability_all_same_color_is_correct : total_probability_same_color = 532 / 4095 :=
by
  sorry

end probability_all_same_color_is_correct_l752_752476


namespace building_floors_l752_752829

-- Define the properties of the staircases
def staircaseA_steps : Nat := 104
def staircaseB_steps : Nat := 117
def staircaseC_steps : Nat := 156

-- The problem asks us to show the number of floors, which is the gcd of the steps of all staircases 
theorem building_floors :
  Nat.gcd (Nat.gcd staircaseA_steps staircaseB_steps) staircaseC_steps = 13 :=
by
  sorry

end building_floors_l752_752829


namespace no_unique_solution_for_any_a_l752_752202

theorem no_unique_solution_for_any_a :
  ¬ ∃ a x y, (x^2 + y^2 = 2) ∧ (|y| - x = a) ∧ 
  (∀ x' y', (x'^2 + y'^2 = 2 ∧ |y'| - x' = a) → (x' = x ∧ y' = y)) :=
  sorry

end no_unique_solution_for_any_a_l752_752202


namespace circumcircle_radius_l752_752975

theorem circumcircle_radius (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) :
  let s₁ := a^2 + b^2
  let s₂ := c^2
  s₁ = s₂ → 
  (c / 2) = 6.5 :=
by
  sorry

end circumcircle_radius_l752_752975


namespace simplify_vectors_l752_752749

variables {Point : Type} [AddGroup Point] (A B C D : Point)

def vector (P Q : Point) : Point := Q - P

theorem simplify_vectors :
  vector A B + vector B C - vector A D = vector D C :=
by
  sorry

end simplify_vectors_l752_752749


namespace _l752_752357

noncomputable def main_theorem :=
  ∀ (x y z : ℝ), 
  (cos x + cos y + cos z = 1) → 
  (sin x + sin y + sin z = 0) →
  cos (2 * x) + cos (2 * y) + cos (2 * z) = -1

end _l752_752357


namespace integer_satisfaction_l752_752452

theorem integer_satisfaction (x : ℤ) : 
  (x + 15 ≥ 16 ∧ -3 * x ≥ -15) ↔ (1 ≤ x ∧ x ≤ 5) :=
by 
  sorry

end integer_satisfaction_l752_752452


namespace ellipse_equation_intersection_fixed_point_l752_752608

-- Definitions for the ellipse characteristics.
def a : ℝ := sqrt 4
def b : ℝ := sqrt 3
def e : ℝ := 1 / 2
def c : ℝ := e * a

-- Definitions for points on the ellipse.
def E : ℝ → ℝ → Prop := λ x y, x^2 / a^2 + y^2 / b^2 = 1
def M : ℝ × ℝ := (0, b)
def N : ℝ × ℝ := (0, -b)
def F2 : ℝ × ℝ := (c, 0)

-- Condition vectors for dot product.
def vec_MF2 : ℝ × ℝ := (c, -b)
def vec_NF2 : ℝ × ℝ := (c, b)

-- Dot product condition.
def dot_product_condition : Prop := (c^2 - b^2 = -2)

-- The condition for slopes.
def slope_condition (k : ℝ) (m : ℝ) (x1 : ℝ) (x2 : ℝ) : Prop :=
  (((4 * k^2 - 1) * x1 * x2 + 4 * k * (m - sqrt 3) * (x1 + x2) + 4 * (m - sqrt 3)^2 * (3 + 4 * k^2) = 0) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0))

-- First part: Equation confirmation.
theorem ellipse_equation :
  dot_product_condition →
  (a = 2 * c) →
  (a^2 - b^2 = c^2) →
  (E x y ↔ (x^2 / 4 + y^2 / 3 = 1)) := by
  sorry

-- Second part: Fixed point confirmation.
theorem intersection_fixed_point (k m x1 x2 : ℝ) :
  slope_condition k m x1 x2 →
  ∃ (p : ℝ × ℝ), p = (0, 2 * sqrt 3) :=
  by sorry

end ellipse_equation_intersection_fixed_point_l752_752608


namespace cover_points_with_circles_l752_752945

open EuclideanGeometry

-- Definitions and conditions from the problem
def total_diameter (circles : List (Circle ℝ)) : ℝ :=
  circles.map Circle.diameter |>.sum

def min_distance (circles : List (Circle ℝ)) : ℝ :=
  circles.combinations 2 |>.map (λ ⟨C₁, C₂⟩, Circle.distance C₁ C₂) |>.foldr min (circles.head.radius)

def construct_circles (points : Finₑ 100 → Point ℝ) : List (Circle ℝ) :=
  sorry -- Construction process as described in solutions steps (not needed for the specification of the problem)

axiom distance_spec (C₁ C₂ : Circle ℝ) : Circle.distance C₁ C₂ = max 0 (Circle.center_distance C₁ C₂ - C₁.radius - C₂.radius)

theorem cover_points_with_circles (points : Finₑ 100 → Point ℝ) :
  ∃ circles : List (Circle ℝ), 
    total_diameter circles < 100 ∧ 
    (∀ i : Finₑ 100, ∃ C : Circle ℝ, C.contains (points i)) ∧ 
    min_distance circles > 1 :=
begin
  -- Proof steps will go here
  sorry
end

end cover_points_with_circles_l752_752945


namespace parabola_line_intersect_l752_752722

theorem parabola_line_intersect (a : ℝ) 
  (P Q : ℝ × ℝ) 
  (M := (a, 0))
  (constant : ℝ)
  (h_line : ∃ α (t: ℝ), P = (a + t * Real.cos α, t * Real.sin α) ∧ Q = (a - t * Real.cos α, -t * Real.sin α))
  (h_parabola : ∀ (x y : ℝ), y^2 = 4 * x → (x, y) = P ∨ (x, y) = Q)
  (h_constant : 1 / Real.norm (P.1 - M.1, P.2 - M.2)^2 + 1 / Real.norm (Q.1 - M.1, Q.2 - M.2)^2 = constant) :
  a = 2 :=
sorry

end parabola_line_intersect_l752_752722


namespace total_profit_percent_is_correct_l752_752151

def cloth_base_prices : Type := (cloth_A_base : ℝ, cloth_B_base : ℝ, cloth_C_base : ℝ)
def cloth_market_adjustments : Type := (cloth_A_adj : ℝ, cloth_B_adj : ℝ, cloth_C_adj : ℝ)
def cloth_quantities_sold : Type := (cloth_A_qty : ℕ, cloth_B_qty : ℕ, cloth_C_qty : ℕ)
def cloth_cost_prices : Type := (cloth_A_cost : ℝ, cloth_B_cost : ℝ, cloth_C_cost : ℝ)
def sales_tax : ℝ := 0.05
def discount_rate : ℕ → ℝ := λ qty, if qty > 10 then 0.15 else 0.05

def calculate_profit_percent (base_prices : cloth_base_prices) 
                             (adjustments : cloth_market_adjustments)
                             (quantities : cloth_quantities_sold)
                             (cost_prices : cloth_cost_prices) : ℝ :=
by
  let (cloth_A_base, cloth_B_base, cloth_C_base) := base_prices
  let (cloth_A_adj, cloth_B_adj, cloth_C_adj) := adjustments
  let (cloth_A_qty, cloth_B_qty, cloth_C_qty) := quantities
  let (cloth_A_cost, cloth_B_cost, cloth_C_cost) := cost_prices

  -- New Selling Prices
  let cloth_A_new := cloth_A_base * (1 + cloth_A_adj)
  let cloth_B_new := cloth_B_base * (1 - cloth_B_adj)
  let cloth_C_new := cloth_C_base

  -- Total Sales before Tax and Discount
  let total_sales := (cloth_A_new * cloth_A_qty) + (cloth_B_new * cloth_B_qty) + (cloth_C_new * cloth_C_qty)

  -- Apply Discount on Cloth B
  let cloth_B_sales := cloth_B_new * cloth_B_qty
  let cloth_B_discount := cloth_B_sales * discount_rate cloth_B_qty
  let total_sales_after_discount := total_sales - cloth_B_discount

  -- Apply Sales Tax
  let total_sales_after_tax := total_sales_after_discount * (1 + sales_tax)

  -- Cost Prices
  let total_cost := (cloth_A_cost * cloth_A_qty) + (cloth_B_cost * cloth_B_qty) + (cloth_C_cost * cloth_C_qty)

  -- Total Profit
  let total_profit := total_sales_after_tax - total_cost

  -- Profit Percent
  (total_profit / total_cost) * 100

-- Constants
def base_prices : cloth_base_prices := (15, 20, 25)
def market_adjustments : cloth_market_adjustments := (0.08, 0.06, 0)
def quantities_sold : cloth_quantities_sold := (10, 15, 5)
def cost_prices : cloth_cost_prices := (10, 15, 20)

theorem total_profit_percent_is_correct : 
  calculate_profit_percent base_prices market_adjustments quantities_sold cost_prices ≈ 30.13% := 
sorry

end total_profit_percent_is_correct_l752_752151


namespace angle_in_fourth_quadrant_l752_752135

theorem angle_in_fourth_quadrant (θ : ℝ) (hθ : θ = 300) : 270 < θ ∧ θ < 360 :=
by
  -- theta equals 300
  have h1 : θ = 300 := hθ
  -- check that 300 degrees lies between 270 and 360
  sorry

end angle_in_fourth_quadrant_l752_752135


namespace clara_gave_10_stickers_l752_752891

-- Defining the conditions
def initial_stickers : ℕ := 100
def remaining_after_boy (B : ℕ) : ℕ := initial_stickers - B
def remaining_after_friends (B : ℕ) : ℕ := (remaining_after_boy B) / 2

-- Theorem stating that Clara gave 10 stickers to the boy
theorem clara_gave_10_stickers (B : ℕ) (h : remaining_after_friends B = 45) : B = 10 :=
by
  sorry

end clara_gave_10_stickers_l752_752891


namespace vector_b_satisfies_conditions_l752_752699

noncomputable section

def a : ℝ^3 := ![3, 2, 4]

def b : ℝ^3 := ![1, 16, 3]

def dotProduct (u v : ℝ^3) : ℝ := u 0 * v 0 + u 1 * v 1 + u 2 * v 2

def crossProduct (u v : ℝ^3) : ℝ^3 :=
  ![u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0]

theorem vector_b_satisfies_conditions :
  dotProduct a b = 20 ∧ crossProduct a b = ![-20, 5, 2] := by
  sorry

end vector_b_satisfies_conditions_l752_752699


namespace square_perimeter_l752_752762

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by
  sorry

end square_perimeter_l752_752762


namespace quadrilateral_theorem_l752_752131

variables {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

-- Given conditions in context of quadrilateral,
def convex_quadrilateral (A B C D : Type*) :=
  ∃ (angle_ABC angle_CDA : ℝ), angle_ABC + angle_CDA = 300 ∧ ∃ (AB CD BC AD : ℝ), AB * CD = BC * AD

-- The main theorem:
theorem quadrilateral_theorem (A B C D : Type*) [metric_space A] 
    [metric_space B] [metric_space C] [metric_space D] 
    (convex_quad : convex_quadrilateral A B C D) :
  ∃ (AB CD AC BD : ℝ), AB * CD = AC * BD :=
by
  sorry

end quadrilateral_theorem_l752_752131


namespace solution_set_of_inequality_l752_752932

theorem solution_set_of_inequality :
  {x : ℝ | (x-1)*(2-x) ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l752_752932


namespace even_function_monotonicity_l752_752179

theorem even_function_monotonicity {f : ℝ → ℝ} (hf_even : ∀ x, f x = f (-x))
  (hf_domain : ∀ x, x ≠ 0 → x ∈ (Set.Ioo -∞ 0 ∪ Set.Ioo 0 ∞))
  (hf_derivative : ∀ x, x * (f' x) > 0) (x1 x2 : ℝ) (hx1x2 : x1 > x2) (hx1x2_sum : x1 + x2 > 0) :
  f x1 > f x2 :=
sorry

end even_function_monotonicity_l752_752179


namespace four_mutually_acquainted_l752_752672

theorem four_mutually_acquainted (G : SimpleGraph (Fin 9)) 
  (h : ∀ (s : Finset (Fin 9)), s.card = 3 → ∃ (u v : Fin 9), u ∈ s ∧ v ∈ s ∧ G.Adj u v) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (u v : Fin 9), u ∈ s → v ∈ s → G.Adj u v :=
by
  sorry

end four_mutually_acquainted_l752_752672


namespace solution_set_correct_l752_752957

noncomputable def f (x : ℝ) : ℝ := if x ≤ 0 then 2^(-x) - 4 else 2^(x) - 4

theorem solution_set_correct : 
  (∀ x, f x = f |x|) → 
  (∀ x, f x = 2^(-x) - 4 ∨ f x = 2^(x) - 4) → 
  { x | f (x - 2) > 0 } = { x | x < 0 ∨ x > 4 } :=
by
  intro h1 h2
  sorry

end solution_set_correct_l752_752957


namespace problem1_problem2_problem3_problem4_l752_752373

open Set

def M : Set ℝ := { x | x > 3 / 2 }
def N : Set ℝ := { x | x < 1 ∨ x > 3 }
def R := {x : ℝ | 1 ≤ x ∧ x ≤ 3 / 2}

theorem problem1 : M = { x | 2 * x - 3 > 0 } := sorry
theorem problem2 : N = { x | (x - 3) * (x - 1) > 0 } := sorry
theorem problem3 : M ∩ N = { x | x > 3 } := sorry
theorem problem4 : (M ∪ N)ᶜ = R := sorry

end problem1_problem2_problem3_problem4_l752_752373


namespace gcd_12a_18b_l752_752655

theorem gcd_12a_18b (a b : ℕ) (h : Nat.gcd a b = 12) : Nat.gcd (12 * a) (18 * b) = 72 :=
sorry

end gcd_12a_18b_l752_752655


namespace sandy_money_l752_752811

theorem sandy_money (x : ℝ) (h : 0.70 * x = 210) : x = 300 := by
sorry

end sandy_money_l752_752811


namespace imaginary_part_of_expression_l752_752614

noncomputable def imaginary_unit := complex.I

theorem imaginary_part_of_expression :
  complex.im ((1 + imaginary_unit) / (1 - imaginary_unit)) = 1 :=
by
  sorry

end imaginary_part_of_expression_l752_752614


namespace square_perimeter_l752_752758

theorem square_perimeter (s : ℝ) (h : s^2 = s) (h₀ : s ≠ 0) : 4 * s = 4 :=
by {
  have s_eq_1 : s = 1 := by {
    field_simp [h],
    exact h₀,
    linarith,
  },
  rw s_eq_1,
  ring,
}

end square_perimeter_l752_752758


namespace sum_of_cubes_l752_752101

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752101


namespace cost_of_one_dozen_pens_l752_752409

theorem cost_of_one_dozen_pens
  (p q : ℕ)
  (h1 : 3 * p + 5 * q = 240)
  (h2 : p = 5 * q) :
  12 * p = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l752_752409


namespace matrix_not_invertible_x_y_z_l752_752214

theorem matrix_not_invertible_x_y_z (x y z : ℝ) 
  (h1 : (matrix.det ![
    ![x, 2 * y, 2 * z],
    ![2 * y, z, x],
    ![2 * z, x, y]]) = 0) 
  (h2 : x^2 + y^2 + z^2 = 0) : 
  (x / (2*y + 2*z) + y / (x + z) + z / (x + y)) = 0 :=
sorry

end matrix_not_invertible_x_y_z_l752_752214


namespace arrange_BA1L1L2O1O2N1_with_Ls_before_Os_l752_752651

/--
Consider the word composed of letters "B", "A₁", "L₁", "L₂", "O₁", "O₂", "N₁".
Each letter with a subscript is considered a distinct character.
We need to prove the total number of permutations adhering to the condition that all 'L's
must appear in order before any 'O's is 210.
-/
theorem arrange_BA1L1L2O1O2N1_with_Ls_before_Os : 
  let letters := ["B", "A₁", "L₁", "L₂", "O₁", "O₂", "N₁"] in
  let distinct_letters := letters.nodup in
  let num_permutations := 210 in
  number_of_permutations letters distinct_letters (λ (s : list string), (s.indexes("L₁") < s.indexes("O₁")) ∧ (s.indexes("L₂") < s.indexes("O₂"))) = num_permutations :=
begin
  sorry  -- proof omitted
end

end arrange_BA1L1L2O1O2N1_with_Ls_before_Os_l752_752651


namespace overall_weighted_average_correct_median_correct_mode_correct_l752_752297

-- Definitions for the given conditions
def number_of_students : ℕ := 50

def sectionA_weightage : ℝ := 0.7
def sectionB_weightage : ℝ := 0.3

def group1_sectionA_score : ℕ := 95
def group1_sectionB_score : ℕ := 85
def group1_students : ℕ := 6

def group2_sectionA_score : ℕ := 0
def group2_sectionB_score : ℕ := 0
def group2_students : ℕ := 4

def group3_sectionA_score : ℕ := 80
def group3_sectionB_score : ℕ := 90
def group3_students : ℕ := 10

def group4_sectionA_score : ℕ := 60
def group4_sectionB_score : ℕ := 70
def group4_students : ℕ := 30

def weighted_score (sectionA_score sectionB_score : ℕ) : ℝ :=
    (sectionA_score : ℝ) * sectionA_weightage + (sectionB_score : ℝ) * sectionB_weightage

def overall_weighted_average : ℝ :=
  (group1_students * weighted_score group1_sectionA_score group1_sectionB_score +
   group2_students * weighted_score group2_sectionA_score group2_sectionB_score +
   group3_students * weighted_score group3_sectionA_score group3_sectionB_score +
   group4_students * weighted_score group4_sectionA_score group4_sectionB_score) /
  number_of_students

def scores : List ℝ :=
  List.replicate group1_students (weighted_score group1_sectionA_score group1_sectionB_score) ++
  List.replicate group2_students (weighted_score group2_sectionA_score group2_sectionB_score) ++
  List.replicate group3_students (weighted_score group3_sectionA_score group3_sectionB_score) ++
  List.replicate group4_students (weighted_score group4_sectionA_score group4_sectionB_score)

def median : ℝ :=
  let sorted_scores := scores.qsort (· < ·)
  (sorted_scores.nth! (number_of_students / 2 - 1) + sorted_scores.nth! (number_of_students / 2)) / 2

def mode : ℝ :=
  scores.foldl (λ (mode_count : ℝ × ℕ) (score : ℝ) =>
    let count := scores.count (· == score)
    if count > mode_count.2 then (score, count) else mode_count) (0, 0).1

-- Mathematical equivalent proof problem
theorem overall_weighted_average_correct :
  overall_weighted_average = 65.44 := sorry

theorem median_correct : 
  median = 63 := sorry

theorem mode_correct : 
  mode = 63 := sorry

end overall_weighted_average_correct_median_correct_mode_correct_l752_752297


namespace extreme_points_count_range_of_a_for_non_negative_f_l752_752721

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log (x + 1) + a * (x^2 - x)
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 1 / (x + 1) + 2 * a * x - a

def number_of_extreme_points (a : ℝ) : ℕ :=
if a < 0 then
  1
else if 0 ≤ a ∧ a ≤ 8 / 9 then
  0
else
  2

theorem extreme_points_count (a : ℝ) : 
  (a < 0 → number_of_extreme_points a = 1) ∧
  (0 ≤ a ∧ a ≤ 8 / 9 → number_of_extreme_points a = 0) ∧
  (a > 8 / 9 → number_of_extreme_points a = 2) :=
sorry

theorem range_of_a_for_non_negative_f : 
  (∀ x > 0, ∀ a ∈ [0, 1], f x a ≥ 0) :=
sorry

end extreme_points_count_range_of_a_for_non_negative_f_l752_752721


namespace num_pairs_divisible_7_l752_752650

theorem num_pairs_divisible_7 (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 1000) (hy : 1 ≤ y ∧ y ≤ 1000)
  (divisible : (x^2 + y^2) % 7 = 0) : 
  (∃ k : ℕ, k = 20164) :=
sorry

end num_pairs_divisible_7_l752_752650


namespace parking_monthly_charge_l752_752852

theorem parking_monthly_charge :
  ∀ (M : ℕ), (52 * 10 - 12 * M = 100) → M = 35 :=
by
  intro M h
  sorry

end parking_monthly_charge_l752_752852


namespace smallest_number_neg2_l752_752515

theorem smallest_number_neg2 : 
  ∀ (a b c d : ℝ), 
    (a = 0) → 
    (b = -2) → 
    (c = 1) → 
    (d = -real.sqrt 3) →
    b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by 
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  split
  { exact le_refl b }
  split
  { exact le_of_lt (by norm_num) }
  { exact le_trans (by norm_num) (real.sqrt_pos.mpr (by norm_num)) }

#align smallest_number_neg2 smallest_number_neg2

end smallest_number_neg2_l752_752515


namespace quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l752_752251

-- Case 1
theorem quadratic_function_expression 
  (a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = 3) : 
  by {exact (a = -2 ∧ b = 3)} := sorry

theorem quadratic_function_range 
  (x : ℝ) 
  (h : -1 ≤ x ∧ x ≤ 2) : 
  (-3 ≤ -2*x^2 + 3*x + 2 ∧ -2*x^2 + 3*x + 2 ≤ 25/8) := sorry

-- Case 2
theorem quadratic_function_m_range 
  (m a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = m) 
  (h₃ : a > 0) : 
  m < 1 := sorry

end quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l752_752251


namespace length_of_wire_l752_752138

-- Conditions
def volume_dm3 : ℝ := 2.2
def diameter_cm : ℝ := 0.50
def volume_cm3 : ℝ := volume_dm3 * 1000 -- converting dm^3 to cm^3

-- Radius in cm
def radius_cm := diameter_cm / 2

-- Volume formula for the cylinder in cm³
def cylinder_volume (r h : ℝ) := Real.pi * (r ^ 2) * h

-- The proof that the length of the wire is approximately 112.09 meters
theorem length_of_wire :
  ∃ (h : ℝ), h ≈ 112.09 ∧ cylinder_volume radius_cm h = volume_cm3 * 1000 := by
  sorry

end length_of_wire_l752_752138


namespace arctan_sum_eq_pi_over_4_l752_752689

theorem arctan_sum_eq_pi_over_4 
  (a b c : ℝ) 
  (h1 : b^2 + c^2 = a^2) :
  (arctan (b / (c + a)) + arctan (c / (b + a)) = π / 4) :=
begin
  sorry
end

end arctan_sum_eq_pi_over_4_l752_752689


namespace isosceles_triangle_perimeter_l752_752604

variable (a b c : ℝ)

-- Definition that side lengths are given
def is_triangle (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c)

def is_valid_triangle := is_triangle a a b ∧ b = 2 ∧ a = 5

theorem isosceles_triangle_perimeter : ∀ (a b : ℝ), is_valid_triangle → 2 * a + b = 12 :=
by
  intros a b h
  sorry

end isosceles_triangle_perimeter_l752_752604


namespace range_of_a_l752_752621

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) → (-2 < a ∧ a ≤ 6/5) :=
by
  sorry

end range_of_a_l752_752621


namespace hypotenuse_length_l752_752441

-- Define the geometric setup
variables {A B C X Y : Type}

-- Define lengths of segments
variable (AB AC BX CY : ℝ)

-- Define ratios for segments
variable (AX XB AY YC : ℝ)

-- Hypotheses
def conditions :=
  right_triangle A B C ∧
  on_line_segment X A B ∧
  on_line_segment Y A C ∧
  (AX/XB = 2/3) ∧
  (AY/YC = 1/3) ∧
  (BY = 18) ∧
  (CX = 40)

-- Theorem to prove the hypotenuse length
theorem hypotenuse_length (h : conditions) : BC = 13 * sqrt 26 := 
  sorry

end hypotenuse_length_l752_752441


namespace total_money_l752_752847

theorem total_money (gold_value silver_value cash : ℕ) (num_gold num_silver : ℕ)
  (h_gold : gold_value = 50)
  (h_silver : silver_value = 25)
  (h_num_gold : num_gold = 3)
  (h_num_silver : num_silver = 5)
  (h_cash : cash = 30) :
  gold_value * num_gold + silver_value * num_silver + cash = 305 :=
by
  sorry

end total_money_l752_752847


namespace num_of_rational_pairs_l752_752328

theorem num_of_rational_pairs (a b c d k : ℤ) (h1 : Int.gcd a b = 1) (h2 : Int.gcd c d = 1) (h3 : ad - bc = k) 
(h4 : k > 0) : ∃ (S : Finset (ℚ × ℚ)), (∀ (x1 x2 : ℚ), 0 ≤ x1 ∧ x1 < 1 ∧ 0 ≤ x2 ∧ x2 < 1 ∧ x1 ∈ S ∧ x2 ∈ S → ∃ (n m : ℤ), a * n + b * m ∧ c * n + d * m ∧ Finset.card S = k).
sorry

end num_of_rational_pairs_l752_752328


namespace ellipse_equation_fixed_point_sum_l752_752955

-- Definitions based on given problem conditions

def ellipse (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

def point_on_ellipse (x y a b : ℝ) (P : ℝ × ℝ) := ellipse P.1 P.2 a b

noncomputable def eccentricity (a c : ℝ) := c / a

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def area_triangle (A B O : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - O.2) + B.1 * (O.2 - A.2) + O.1 * (A.2 - B.2))

-- Part I: Proving the equation of the ellipse
theorem ellipse_equation (a b : ℝ) (P : ℝ × ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : eccentricity a (a / 2) = 1 / 2) 
  (h4 : point_on_ellipse 1 (3/2) a b) : 
  ellipse 1 (3/2) 2 sqrt(3) := 
sorry

-- Part II: Proving the constant value |GM| + |HM| 
theorem fixed_point_sum (A B O G H: ℝ × ℝ) (M : ℝ × ℝ) 
  (h1 : ellipse 1 (3/2) 2 sqrt(3))
  (h2 : midpoint A B = M)
  (h3 : ∀ M′, midpoint A B = M′ → area_triangle A B O ≤ area_triangle A B O)
  : ∃ G H : ℝ × ℝ, |G.1 - M.1| + |H.1 - M.1| = 2 * sqrt(2) :=
sorry

end ellipse_equation_fixed_point_sum_l752_752955


namespace collinearity_of_E_O_K_l752_752312

noncomputable def rightTriangle (A B C : Point) : Prop :=
  isTriangle A B C ∧ rightAngle C

noncomputable def bisectsAngle (P A B C : Point) : Prop :=
  isOnCircumcircle P A B C ∧ angleBisect P C A B

noncomputable def incircleCenter (O A B C : Point) : Prop :=
  isIncenter O A B C

noncomputable def tangentPoint (E A C : Point) : Prop :=
  isOnIncircle E A C ∧ tangent E A C

noncomputable def collinearPoints (E O K : Point) : Prop :=
  collinear E O K

theorem collinearity_of_E_O_K 
  (A B C P Q K O E : Point)
  (h1 : rightTriangle A B C)
  (h2 : bisectsAngle P A B C)
  (h3 : bisectsAngle Q A B C)
  (h4 : intersection K PQ AB)
  (h5 : incircleCenter O A B C)
  (h6 : tangentPoint E A C)
  : collinearPoints E O K :=
sorry

end collinearity_of_E_O_K_l752_752312


namespace centroid_sum_of_squares_l752_752343

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end centroid_sum_of_squares_l752_752343


namespace two_triangles_from_tetrahedron_l752_752814

-- Let a tetrahedron have vertices A, B, C, D, and edges AB, AC, AD, BC, BD, CD.
structure Tetrahedron (α : Type) [LinearOrder α] :=
  (A B C D : α)
  (AB AC AD BC BD CD : α)
  (AB_ge_AC : AB ≥ AC)
  (AC_ge_BD : AC ≥ BD)

-- We need to prove that two distinct triangles can be formed from the edges of the tetrahedron.
theorem two_triangles_from_tetrahedron {α : Type} [LinearOrder α] (T : Tetrahedron α) : 
  ∃ (triangle1 triangle2 : set α), 
  (triangle1 = {T.BC, T.CD, T.BD}) ∧ 
  (triangle2 = {T.AC, T.AD, T.CD}) :=
by
  sorry

end two_triangles_from_tetrahedron_l752_752814


namespace sequence_50th_term_l752_752400

theorem sequence_50th_term : 
  ∀ n, (∃ m, (m = -48 + 2 * (n - 1)) ∧ (n = 50)) → m = 50 :=
by
  intros n h
  obtain ⟨m, hm1, hm2⟩ := h
  rw hm2 at hm1
  rw hm1
  sorry

end sequence_50th_term_l752_752400


namespace range_of_a_l752_752589

noncomputable def f (a x : ℝ) := a * x^3
noncomputable def g (x : ℝ) := 9 * x^2 + 3 * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ Icc 1 2 → f a x ≥ g x) → a ≥ 11 :=
by
  -- proof to be filled in
  sorry

end range_of_a_l752_752589


namespace find_sum_of_money_l752_752139

theorem find_sum_of_money:
  ∃ (P : ℝ), 
  let SI1 := P * (22/100) * 3 in
  let SI2 := P * (15/100) * 4 in
  SI1 = SI2 + 1200 ∧ P = 20000 :=
by
  sorry

end find_sum_of_money_l752_752139


namespace subset_of_sets_l752_752368

theorem subset_of_sets (a : ℝ) : 
  {0, -a} ⊆ {1, a - 2, 2 * a - 2} → a = 1 :=
by
  intros h
  sorry

end subset_of_sets_l752_752368


namespace problem1_problem2_l752_752535

theorem problem1 :
  |-2| + (1/3)^(-1:ℤ) - (sqrt 3 - 2021)^0 - sqrt 3 * Real.tan (Real.pi / 3) = 1 :=
by
  sorry

theorem problem2 :
  4 * Real.sin (Real.pi / 6) - sqrt 2 * Real.cos (Real.pi / 4) - sqrt 3 * Real.tan (Real.pi / 6) + 2 * Real.sin (Real.pi / 3) = sqrt 3 :=
by
  sorry

end problem1_problem2_l752_752535


namespace solution_l752_752332

noncomputable def problem :=
  let A : (ℝ × ℝ × ℝ) := (a, 0, 0) in
  let B : (ℝ × ℝ × ℝ) := (0, b, 0) in
  let C : (ℝ × ℝ × ℝ) := (0, 0, c) in
  let plane_eq (x y z a b c : ℝ) : Prop := (x / a + y / b + z / c = 1) in
  let distance_eq (a b c : ℝ) : Prop :=
    (1 / (Real.sqrt ((1 / (a^2)) + 1 / (b^2) + 1 / (c^2)))) = 2 in
  let centroid (a b c : ℝ) : ℝ × ℝ × ℝ := (a / 3, b / 3, c / 3) in
  let (p, q, r) := centroid a b c in
  (plane_eq a 0 0 a b c) ∧
  (plane_eq 0 b 0 a b c) ∧
  (plane_eq 0 0 c a b c) ∧
  (distance_eq a b c) → (1 / (p^2) + 1 / (q^2) + 1 / (r^2)) = 2.25

theorem solution : problem :=
sorry

end solution_l752_752332


namespace smallest_multiple_of_18_with_digits_8_9_0_l752_752054

theorem smallest_multiple_of_18_with_digits_8_9_0 :
  ∃ (m : ℕ), (∀ d ∈ (m.digits 10), d = 8 ∨ d = 9 ∨ d = 0) ∧ (m % 18 = 0) ∧ (m = 890) ∧ (m / 18 = 49) :=
by
  -- Existence of m such that digits of m are only 8, 9, 0
  use 890
  split
  -- All digits of m are 8, 9, 0
  { left; exact Nat.digits_of_digits 10 890 }
  split
  -- m is divisible by 18
  { exact Nat.mod_eq_zero_of_dvd (Nat.dvd_of_digits_if_lte10 10 890 18) }
  split
  -- m is equal to 890
  { refl }
  -- m / 18 = 49
  { rw Nat.div_eq_iff_eq_mul_left _ (succ_pos' 17),
    exact Nat.dvd_trans _ }

end smallest_multiple_of_18_with_digits_8_9_0_l752_752054


namespace slope_y_intercept_indeterminate_l752_752384

noncomputable def point (x y : ℝ) := (x, y)

def A : ℝ × ℝ := point 5 10
def B : ℝ × ℝ := point 5 20

theorem slope_y_intercept_indeterminate (A B : ℝ × ℝ) (hA : A = (5, 10)) (hB : B = (5, 20)) :
  let m := (B.2 - A.2) / (B.1 - A.1) in let y_intercept := if B.1 - A.1 = 0 then none else some (A.2 - m * A.1) in
  (B.1 = A.1) ↔ m * (y_intercept.get_or_else 0) = ⊤ :=
by {
  sorry
}

end slope_y_intercept_indeterminate_l752_752384


namespace tan_theta_neg_four_thirds_l752_752242

theorem tan_theta_neg_four_thirds 
  (θ : ℝ) 
  (h1 : sin θ + cos θ = 1/5)
  (h2 : 0 < θ ∧ θ < π) : tan θ = -4/3 := sorry

end tan_theta_neg_four_thirds_l752_752242


namespace physicist_can_destroy_all_imons_l752_752813

-- Define a type for immediate operations allowed by the physicist
inductive Operation
| destroy (i : ℕ)
| double

-- Define an edge as a relationship between two vertices
structure Edge where
  src : ℕ
  dst : ℕ

-- Define a simple graph with vertices and edges
structure Graph where
  vertices : Finset ℕ
  edges : Finset Edge

-- Function to determine the number of edges connected to a given vertex
def degree (G : Graph) (v : ℕ) : ℕ :=
  (G.edges.filter (λ e => e.src = v ∨ e.dst = v)).card

-- Function to perform the destroy operation
def destroy (G : Graph) (v : ℕ) : Graph :=
  { G with
    vertices := G.vertices.erase v,
    edges := G.edges.filter (λ e => e.src ≠ v ∧ e.dst ≠ v) }

-- Function to perform the doubling operation
def double (G : Graph) : Graph :=
  let V' := G.vertices.map (λ v => 2 * v + 1)
  let E' := G.edges.map (λ e => ⟨2 * e.src + 1, 2 * e.dst + 1⟩)
  let new_edges := G.vertices.map (λ v => ⟨v, 2 * v + 1⟩)
  { vertices := G.vertices ∪ V', edges := G.edges ∪ E' ∪ new_edges }

-- The main theorem where we assert that all imons can be "destroyed"
theorem physicist_can_destroy_all_imons (initial_graph : Graph) :
  ∃ sequence_of_operations : list Operation,
  let final_graph := sequence_of_operations.foldl (λ G op =>
    match op with
    | Operation.destroy vertex => destroy G vertex
    | Operation.double => double G) initial_graph in
  final_graph.edges = ∅ :=
by
  sorry

end physicist_can_destroy_all_imons_l752_752813


namespace sphere_diameter_l752_752767

theorem sphere_diameter (V1 V2 r1 r2 : ℝ) (c d : ℕ) (h1 : V1 = (4 / 3) * Real.pi * (r1^3))
  (h2 : r1 = 6) (h3 : V2 = 3 * V1) (h4 : V2 = (4 / 3) * Real.pi * (r2^3)) 
  (h5 : r2 = 6 * Real.cbrt 3) (h6 : (12 : ℝ) = c) (h7 : (3 : ℝ) = d) : c + d = 15 := by 
  sorry

end sphere_diameter_l752_752767


namespace min_cubes_cover_snaps_l752_752147

-- Defining the conditions
def modified_cube (cubes: ℕ) : Prop := 
  ∀ (n: ℕ), n < cubes -> (snaps_covered n) ∧ (receptacle_visible n)

-- The proof problem statement
theorem min_cubes_cover_snaps : ∃ n, modified_cube n ∧ n = 6 :=
sorry

end min_cubes_cover_snaps_l752_752147


namespace find_real_a_l752_752221

theorem find_real_a (a : ℝ) : 
  (a ^ 2 + 2 * a - 15 = 0) ∧ (a ^ 2 + 4 * a - 5 ≠ 0) → a = 3 :=
by 
  sorry

end find_real_a_l752_752221


namespace students_not_examined_l752_752433

theorem students_not_examined (boys girls examined : ℕ) (h1 : boys = 121) (h2 : girls = 83) (h3 : examined = 150) : 
  (boys + girls - examined = 54) := by
  sorry

end students_not_examined_l752_752433


namespace water_depth_l752_752489

-- Define the parameters of the problem
def tank_length : ℝ := 15
def diameter : ℝ := 4
def radius : ℝ := diameter / 2
def surface_area : ℝ := 12
def chord_length : ℝ := surface_area / tank_length

-- The quadratic equation derived from the problem
def quadratic_eq (h : ℝ) : Prop :=
  h^2 - 4 * h + 0.16 = 0

-- The main theorem statement, proving the water depth satisfies the quadratic equation
theorem water_depth (h : ℝ) (hl : h = 3.96 ∨ h = 0.04) : quadratic_eq h :=
by { 
  rcases hl with (rfl | rfl);
  unfold quadratic_eq;
  ring;
  norm_num }

end water_depth_l752_752489


namespace remainder_even_coefficients_div_3_l752_752652

theorem remainder_even_coefficients_div_3 :
  let poly := λ x : ℤ, (2 * x + 4)^2010
  let a_n := λ (n : ℕ), a_n
  let even_coeff_sum := ∑ i in (finset.range (2010 // 2)).map_even, a_n(2*i)
  even_coeff_sum % 3 = 0 := 
sorry

end remainder_even_coefficients_div_3_l752_752652


namespace total_pages_is_360_l752_752129

-- Definitions from conditions
variable (A B : ℕ) -- Rates of printer A and printer B in pages per minute.
variable (total_pages : ℕ) -- Total number of pages of the task.

-- Given conditions
axiom h1 : 24 * (A + B) = total_pages -- Condition from both printers working together.
axiom h2 : 60 * A = total_pages -- Condition from printer A alone.
axiom h3 : B = A + 3 -- Condition of printer B printing 3 more pages per minute.

-- Goal: Prove the total number of pages is 360
theorem total_pages_is_360 : total_pages = 360 := 
by 
  sorry

end total_pages_is_360_l752_752129


namespace angle_A_CB_l752_752690

theorem angle_A_CB (A B C O : Type) [is_triangle A B C] :
    is_angle_bisector (∠ A C O) (∠ B C O) ∧ 
    is_angle_bisector (∠ C A O) (∠ B A O) ∧
    AC + AO = BC ∧
    ∠ ABC = 25 :=
    ∠ ACB = 105 :=
sorry

end angle_A_CB_l752_752690


namespace line_intersects_xz_plane_at_8_0_17_l752_752209

theorem line_intersects_xz_plane_at_8_0_17:
  ∃ t: ℝ, (8, 0, 17) = (2 + 2 * t, 0, 5 + 4 * t) := 
by
  existsi 3
  simp
  split
  { norm_num }
  { 
    split
    { norm_num }
    { norm_num }
  }

end line_intersects_xz_plane_at_8_0_17_l752_752209


namespace concurrency_of_tangents_l752_752053

-- Define the problem in terms of Lean expressions and assumptions
theorem concurrency_of_tangents
  (ΔABC : Type)
  (incircle_I : Point → Point → Point → Set Point)
  (K L M : Point)
  (tangent_points : (Point → Point → Set Point) → Point)
  (tangents_A : (Set Point → (Point → Point → Set Point)) → Set Point)
  (tangents_B : (Set Point → (Point → Point → Set Point)) → Set Point)
  (tangents_C : (Set Point → (Point → Point → Set Point)) → Set Point) :
  let ⟨A, B, C⟩ := ΔABC in
  ∃ I_0 : Point, tangent_points (tangents_A (incircle_I A B K)) = I_0 ∧
  tangent_points (tangents_B (incircle_I B C L)) = I_0 ∧
  tangent_points (tangents_C (incircle_I C A M)) = I_0 :=
by
  -- This is a placeholder for the proof
  sorry

end concurrency_of_tangents_l752_752053


namespace range_of_a_if_slope_is_obtuse_l752_752289

theorem range_of_a_if_slope_is_obtuse : 
  ∀ a : ℝ, (a^2 + 2 * a < 0) → -2 < a ∧ a < 0 :=
by
  intro a
  intro h
  sorry

end range_of_a_if_slope_is_obtuse_l752_752289


namespace smallest_number_neg2_l752_752516

theorem smallest_number_neg2 : 
  ∀ (a b c d : ℝ), 
    (a = 0) → 
    (b = -2) → 
    (c = 1) → 
    (d = -real.sqrt 3) →
    b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by 
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  split
  { exact le_refl b }
  split
  { exact le_of_lt (by norm_num) }
  { exact le_trans (by norm_num) (real.sqrt_pos.mpr (by norm_num)) }

#align smallest_number_neg2 smallest_number_neg2

end smallest_number_neg2_l752_752516


namespace square_perimeter_l752_752761

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by 
  have h1 : s = 1 := 
  begin
    -- Considering non-zero s, divide both sides by s
    by_contradiction hs,
    have hs' : s ≠ 0 := λ h0, hs (by simp [h0] at h),
    exact hs (by simpa [h, hs'] using h),
  end,
  rw h1,
  norm_num

end square_perimeter_l752_752761


namespace non_congruent_squares_on_6x6_grid_l752_752647

theorem non_congruent_squares_on_6x6_grid :
  let a1 := 5 * 5,
      a2 := 4 * 4,
      a3 := 3 * 3,
      a4 := 2 * 2,
      a5 := 1 * 1,
      d1 := 5 * 5,
      d2 := 4 * 4,
      d3 := 5 * 4 + 4 * 5,
      d4 := 5 * 3 + 3 * 5
  in a1 + a2 + a3 + a4 + a5 + d1 + d2 + d3 + d4 = 166 :=
by 
  let a1 := 5 * 5
  let a2 := 4 * 4
  let a3 := 3 * 3
  let a4 := 2 * 2
  let a5 := 1 * 1
  let d1 := 5 * 5
  let d2 := 4 * 4
  let d3 := 5 * 4 + 4 * 5
  let d4 := 5 * 3 + 3 * 5
  show a1 + a2 + a3 + a4 + a5 + d1 + d2 + d3 + d4 = 166
  sorry

end non_congruent_squares_on_6x6_grid_l752_752647


namespace trigonometric_identity_l752_752896

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - 2 / Real.sin (70 * Real.pi / 180)) = 4 * Real.cot (40 * Real.pi / 180) :=
by
  -- The proof will be skipped with sorry
  sorry

end trigonometric_identity_l752_752896


namespace ticket_cost_l752_752882

open Real

-- Variables for ticket prices
variable (A C S : ℝ)

-- Given conditions
def cost_condition : Prop :=
  C = A / 2 ∧ S = A - 1.50 ∧ 6 * A + 5 * C + 3 * S = 40.50

-- The goal is to prove that the total cost for 10 adult tickets, 8 child tickets,
-- and 4 senior tickets is 64.38
theorem ticket_cost (h : cost_condition A C S) : 10 * A + 8 * C + 4 * S = 64.38 :=
by
  -- Implementation of the proof would go here
  sorry

end ticket_cost_l752_752882


namespace pyramid_volume_l752_752421

-- Definitions based on the problem conditions
def AB : ℝ := 14 * Real.sqrt 3
def BC : ℝ := 15 * Real.sqrt 3
def midpoint (A B : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2, (A.3 + B.3) / 2)

-- Coordinates for vertices assuming Cartesian system
def A : ℝ × ℝ × ℝ := (0, Real.sqrt (14^2 * 3 + 15^2 * 3), 0)
def B : ℝ × ℝ × ℝ := (2 * 14 * Real.sqrt 3, 0, 0)
def C : ℝ × ℝ × ℝ := (14 * Real.sqrt 3, 0, 0)
def D : ℝ × ℝ × ℝ := (-14 * Real.sqrt 3, 0, 0)
def P : ℝ × ℝ × ℝ := midpoint A C  -- Midpoint of diagonal AC

-- Helper function for distance
def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

-- Calculations for volume of the pyramid
def area_of_triangle (a b c : ℝ) : ℝ := 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

def volume_of_pyramid (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- Finally, the statement of our theorem
theorem pyramid_volume : 
  let base_area := area_of_triangle (distance A B) (distance A C) (distance B C) in
  volume_of_pyramid base_area (distance A P) = 735 :=
by {
  sorry  -- to be elaborated with the proof
}

end pyramid_volume_l752_752421


namespace find_m_l752_752393

-- Given conditions:
-- Regular hexagon inscribed in a circle of unit area
-- Point Q lies inside the circle fulfilling specific area requirements
def hexagon_area : ℝ := 1  -- unit circle, area = 1
def region1_area : ℝ := 1/12  -- Area for region bounded by Q, B1, B2, and minor arc
def region2_area : ℝ := 1/15  -- Area for region bounded by Q, B3, B4, and minor arc
def region3_area (m : ℝ) : ℝ := 1/18 - real.sqrt 3 / m  -- Area for region bounded by Q, B5, B6, and minor arc

-- The problem statement, mathematically equivalent proof problem
theorem find_m : ∃ (m : ℝ), region3_area m = 20 * real.sqrt 3 := sorry

end find_m_l752_752393


namespace mutual_exclusive_and_complementary_l752_752800

def Event : Type := Set Unit

variable (A B C D : Event)
variable (hit_both_times miss_both_times hit_once hit_at_least_once : Event)
variable (mutually_exclusive complementary : Set (Event × Event))

axiom A_def : A = hit_both_times
axiom B_def : B = miss_both_times
axiom C_def : C = hit_once
axiom D_def : D = hit_at_least_once

def is_mutually_exclusive (E1 E2 : Event) : Prop :=
  E1 ∩ E2 = ∅

def is_complementary (E1 E2 : Event) : Prop :=
  E1 ∪ E2 = Univ ∧ E1 ∩ E2 = ∅

theorem mutual_exclusive_and_complementary :
  (mutually_exclusive = {(A, B), (A, C), (B, D)} ∧ 
   complementary = {(B, D)}) := by
  sorry

end mutual_exclusive_and_complementary_l752_752800


namespace interval_monotonic_decrease_axis_of_symmetry_cos_2alpha_l752_752260

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + sin x ^ 2 - cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := sqrt 2 * sin (4 * x)

-- (1) Prove the interval of monotonic decrease for f(x)
theorem interval_monotonic_decrease (k : ℤ) :
  ∀ x, (f x = 2 * sin x * cos x + sin x ^ 2 - cos x ^ 2) -> 
  (sqrt 2 * sin (2 * x - π / 4) < 0) -> 
  (3 * π / 8 + ↑k * π ≤ x ∧ x ≤ 7 * π / 8 + ↑k * π) :=
sorry

-- (2) Prove the axis of symmetry for g(x)
theorem axis_of_symmetry (k : ℤ) :
  ∀ x, (g x = sqrt 2 * sin (4 * x)) -> 
  (4 * x = π / 2 + ↑k * π) -> 
  x = π / 8 + ↑k * π / 4 :=
sorry

-- (3) Prove if f(-α/2) = -√3/3 for α ∈ (0, π), then cos 2α = -√5/3
theorem cos_2alpha (α : ℝ) :
  (0 < α ∧ α < π) ∧ (f (-α / 2) = -sqrt 3 / 3) -> 
  cos (2 * α) = -sqrt 5 / 3 :=
sorry

end interval_monotonic_decrease_axis_of_symmetry_cos_2alpha_l752_752260


namespace sequence_solution_l752_752566

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), (a 0 = 1) ∧ (∀ k < 4, (a (k + 1) - 1) * a (k - 1) ≥ a k ^ 2 * (a k - 1)) ∧
  ( ∑ i in Finset.range 4, a i / a (i + 1) = 99 / 100 ) ∧
  (a 1 = 2 ∧ a 2 = 5 ∧ a 3 = 56 ∧ a 4 = 78400)
:= by
  sorry

end sequence_solution_l752_752566


namespace math_equivalent_proof_problem_l752_752277

noncomputable def f (x m : ℝ) : ℝ := Real.exp x - m * x

def is_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ x y, x ∈ I → y ∈ I → x < y → f x < f y
def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ x y, x ∈ I → y ∈ I → x < y → f y < f x

theorem math_equivalent_proof_problem :
  (∀ x ∈ Set.Ioi 0, f x m).has_deriv_at (Real.exp x - m) → 
  ∀ I, (is_increasing (λ x, f x m) I → m ≤ 1) ∧
  (¬is_decreasing (λ x, f x m) I ∨ m ≤ 1) ∧
  (¬(m ≤ 1) ∨ ¬is_increasing (λ x, f x m) I) := sorry

end math_equivalent_proof_problem_l752_752277


namespace maximum_value_of_expr_l752_752958

noncomputable def max_value_expr (a b c : Fin 2022 → ℝ) : ℝ :=
  (Finset.range 2022).sum (λ i =>
    a i * if (i + 1) < 2022 then b (i + 1) else 0 * if (i + 2) < 2022 then c (i + 2) else 0 +
    if (i - 2) < 2022 then a i * b (i - 1) * c (i - 2) else 0)

theorem maximum_value_of_expr :
  ∀ (a b c : Fin 2022 → ℝ),
    (∀ i, 0 ≤ a i) → 
    (∀ i, 0 ≤ b i) → 
    (∀ i, 0 ≤ c i) → 
    (Finset.range 2022).sum a = 1 →
    (Finset.range 2022).sum (λ i, b i ^ 2) = 2 →
    (Finset.range 2022).sum (λ i, c i ^ 3) = 3 →
    max_value_expr a b c ≤ real.cbrt 12 := by
  sorry

end maximum_value_of_expr_l752_752958


namespace original_avg_sequence_l752_752504

theorem original_avg_sequence (x : ℝ)
  (h1 : let original_sum := x + (x+1) + (x+2) + (x+3) + (x+4) + (x+5) + (x+6) + (x+7) + (x+8) + (x+9))
        original_sum = 10 * x + 45)
  (h2 : let modified_sum := (x-9) + (x-8) + (x-7) + (x-6) + (x-5) + (x-4) + (x-3) + (x-2) + (x-1) + x
        modified_sum = 10 * x - 45)
  (h3 : (modified_sum / 10) = 15.5) :
  let original_avg := original_sum / 10 
  original_avg = 24.5 :=
by
  sorry

end original_avg_sequence_l752_752504


namespace least_number_of_cans_l752_752840
open Nat

-- Defining the quantities of each drink
def Maaza : ℕ := 200
def Pepsi : ℕ := 288
def Sprite : ℕ := 736
def CocaCola : ℕ := 450
def Fanta : ℕ := 625

-- Conditions to ensure that GCD must be checked amongst all values
theorem least_number_of_cans (hMaaza : Maaza = 200) 
                            (hPepsi : Pepsi = 288) 
                            (hSprite : Sprite = 736) 
                            (hCocaCola : CocaCola = 450) 
                            (hFanta : Fanta = 625) :
  let gcd_value := gcd (gcd (gcd (gcd Maaza Pepsi) Sprite) CocaCola) Fanta in
  gcd_value = 1 →
  let total_cans := Maaza / gcd_value + Pepsi / gcd_value + Sprite / gcd_value + CocaCola / gcd_value + Fanta / gcd_value in
  total_cans = 2299 :=
begin
  intros,
  rw hMaaza at *,
  rw hPepsi at *,
  rw hSprite at *,
  rw hCocaCola at *,
  rw hFanta at *,
  sorry
end

end least_number_of_cans_l752_752840


namespace number_of_true_subsets_A_l752_752269

def is_triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a

def A : Finset (ℕ × ℕ) :=
  {(x, y) | ∃ (z : ℕ), z = 7 ∧ x + y ≥ 8 ∧ 1 ≤ x ∧ x ≤ y ∧ y ≤ 7 ∧ is_triangle x y z}.toFinset

theorem number_of_true_subsets_A : A.card = 16 → 2 ^ 16 - 1 = 65535 :=
by
  intro h
  rw h
  norm_num

end number_of_true_subsets_A_l752_752269


namespace centroid_sum_of_squares_l752_752340

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end centroid_sum_of_squares_l752_752340


namespace initial_fund_is_890_l752_752773

-- Given Conditions
def initial_fund (n : ℕ) : ℝ := 60 * n - 10
def bonus_given (n : ℕ) : ℝ := 50 * n
def remaining_fund (initial : ℝ) (bonus : ℝ) : ℝ := initial - bonus

-- Proof problem: Prove that the initial amount equals $890 under the given constraints
theorem initial_fund_is_890 :
  ∃ n : ℕ, 
    initial_fund n = 890 ∧ 
    initial_fund n - bonus_given n = 140 :=
by
  sorry

end initial_fund_is_890_l752_752773


namespace workers_finish_job_l752_752821

theorem workers_finish_job :
  ∀ (job_total_work_days remaining_work_days additional_workers total_workers : ℕ),
  job_total_work_days = 6 * 8 → 
  remaining_work_days = job_total_work_days - 6 * 3 →
  additional_workers = 4 →
  total_workers = 6 + additional_workers →
  remaining_work_days / total_workers = 3 :=
by {
  -- Given
  intros,
  rw total_workers,
  rw remaining_work_days,
  sorry,
}

end workers_finish_job_l752_752821


namespace sum_of_coordinates_of_center_of_circle_l752_752777

theorem sum_of_coordinates_of_center_of_circle :
  ∃ (cx cy : ℝ), (cx, cy) = ((7 + -5) / 2, (-3 + 2) / 2) ∧ cx + cy = 0.5 :=
by
  use ((7 + -5) / 2)
  use ((-3 + 2) / 2)
  split
  -- Proof that the center is at the given coordinates (skipped, use sorry)
  sorry
  -- Proof that the sum of the coordinates is 0.5 (skipped, use sorry)
  sorry

end sum_of_coordinates_of_center_of_circle_l752_752777


namespace additional_toothpicks_for_6_steps_l752_752521

/-- Given a staircase with 4 steps that uses 28 toothpicks, and an observed pattern of 
increase in toothpicks needed to extend the staircase, prove that extending to a 6-step 
staircase requires a total of 26 additional toothpicks. -/
theorem additional_toothpicks_for_6_steps (initial_steps : ℕ) 
(initial_toothpicks : ℕ) (toothpicks_step_4_to_5 : ℕ) 
(toothpicks_step_5_to_6 : ℕ) : 
  initial_steps = 4 → initial_toothpicks = 28 →
  toothpicks_step_4_to_5 = 12 → toothpicks_step_5_to_6 = 14 →
  let additional_toothpicks := toothpicks_step_4_to_5 + toothpicks_step_5_to_6 in
  additional_toothpicks = 26 :=
begin
  intros h1 h2 h3 h4,
  let additional_toothpicks := toothpicks_step_4_to_5 + toothpicks_step_5_to_6,
  have h5 : additional_toothpicks = 26,
  { rw [h3, h4],
    exact rfl },
  exact h5
end

end additional_toothpicks_for_6_steps_l752_752521


namespace shuaiFenRatioAndBonus_correct_l752_752133

open Real

noncomputable def shuaiFenRatioAndBonus : ℝ × ℝ :=
  let a₁ := 1   -- Bonus A
  let r := 0.9 -- Common ratio
  let a₂ := a₁ * r -- Bonus B
  let a₃ := a₁ * r^2 -- Bonus C
  let a₄ := a₁ * r^3 -- Bonus D
  let totalBonus := a₁ + a₂ + a₃ + a₄
  let acBonus := a₁ + a₃
  if totalBonus = 68780 ∧ acBonus = 36200 then (0.1, a₄) else (0, 0)

theorem shuaiFenRatioAndBonus_correct :
  let (ratio, dBonus) := shuaiFenRatioAndBonus
  ratio = 0.1 ∧ dBonus = 14580 := by
  let a₁ : ℝ := 1   -- Bonus A
  let r : ℝ := 0.9 -- Common ratio
  let a₂ : ℝ := a₁ * r -- Bonus B
  let a₃ : ℝ := a₁ * r^2 -- Bonus C
  let a₄ : ℝ := a₁ * r^3 -- Bonus D
  have h1 : a₁ + a₂ + a₃ + a₄ = 1 + 0.9 + 0.9^2 + 0.9^3 := by
    sorry
  have h2 : a₁ + a₃ = 1 + 0.9^2 := by
    sorry
  let totalBonus : ℝ := (1 + 0.9 + 0.9^2 + 0.9^3) * 1000 * r.prod_cent  else 0
  let acBonus : ℝ := 3.6 * 10000
  have h3 : totalBonus = 68780 := by
    sorry
  have h4 : acBonus = 36200 := by
    sorry
  have h5 : let total_bonus := 68780 - 36200 in
    let mut d_bonus :=  money.totalBonus - acBonus 
    ((0.9^-2)*total.d_bonus)= 14580):= by
    dc_bonus sorry  
  from h1 h2 h3 h4 h5 this:= 
    sorry
 }

end shuaiFenRatioAndBonus_correct_l752_752133


namespace centroid_sum_of_squares_l752_752342

theorem centroid_sum_of_squares (a b c : ℝ) 
  (h : 1 / Real.sqrt (1 / a^2 + 1 / b^2 + 1 / c^2) = 2) : 
  1 / (a / 3) ^ 2 + 1 / (b / 3) ^ 2 + 1 / (c / 3) ^ 2 = 9 / 4 :=
by
  sorry

end centroid_sum_of_squares_l752_752342


namespace cumulative_weighted_gpa_l752_752507

def weighted_gpa (grades_credit_hours : List (ℕ × ℕ)) : ℚ :=
  (grades_credit_hours.map (λ gc, (gc.1 * gc.2 : ℚ)).sum / (grades_credit_hours.map (λ gc, (gc.2 : ℚ)).sum))

def round_to_tenth (x : ℚ) : ℚ :=
  (Float.ofInt x.nat_num).round (Float.mk 10 0) / 10

theorem cumulative_weighted_gpa : 
  let sophomore_year := (95, 30)
  let freshman_year := (86, 32) in
  round_to_tenth (weighted_gpa [sophomore_year, freshman_year]) = 90.4 := 
by
  sorry

end cumulative_weighted_gpa_l752_752507


namespace inequality_proof_l752_752587

variable {n : ℕ}
variable (a : Fin n → ℝ)

def nonneg_real_sum (a : Fin n → ℝ) : Prop :=
  (∀ i, 0 ≤ a i) ∧ (∑ i, a i = n)

theorem inequality_proof (h : nonneg_real_sum a) :
  ∑ i, (a i)^2 / (1 + (a i)^4) ≤ ∑ i, 1 / (1 + a i) :=
sorry

end inequality_proof_l752_752587


namespace domain_of_v_l752_752087

noncomputable def v (x : ℝ) : ℝ := 1 / real.sqrt (x - 1)

theorem domain_of_v :
  {x : ℝ | ∃ y : ℝ, v(x) = y} = set.Ioi 1 := 
by
  sorry

end domain_of_v_l752_752087


namespace expected_value_of_fair_6_sided_die_l752_752453

noncomputable def fair_die_expected_value : ℝ :=
  (1/6) * 1 + (1/6) * 2 + (1/6) * 3 + (1/6) * 4 + (1/6) * 5 + (1/6) * 6

theorem expected_value_of_fair_6_sided_die : fair_die_expected_value = 3.5 := by
  sorry

end expected_value_of_fair_6_sided_die_l752_752453


namespace sum_of_intersections_l752_752946

-- Define f with the given condition
axiom f : ℝ → ℝ
axiom h_f_symmetry : ∀ x : ℝ, f (-x) = 2 - f x

-- Define intersection points
axiom intersections : list (ℝ × ℝ)
axiom h_intersections : ∀ p ∈ intersections, 
  let x := p.fst, y := p.snd in y = (x + 1) / x ∧ y = f x

-- Define main theorem to be proved
theorem sum_of_intersections (m : ℕ) (h_m : list.length intersections = m) :
  ∑ i in intersections.map (λ p, p.fst + p.snd), id = m :=
sorry

end sum_of_intersections_l752_752946


namespace autumn_pencils_l752_752527

-- Define the conditions of the problem.
def initial_pencils := 20
def misplaced_pencils := 7
def broken_pencils := 3
def found_pencils := 4
def bought_pencils := 2

-- Define the number of pencils lost and gained.
def pencils_lost := misplaced_pencils + broken_pencils
def pencils_gained := found_pencils + bought_pencils

-- Define the final number of pencils.
def final_pencils := initial_pencils - pencils_lost + pencils_gained

-- The theorem we want to prove.
theorem autumn_pencils : final_pencils = 16 := by
  sorry

end autumn_pencils_l752_752527


namespace compute_expression_l752_752541

theorem compute_expression : 16 * (125 / 2 + 25 / 4 + 9 / 16 + 1) = 1125 := by
  linarith

end compute_expression_l752_752541


namespace probability_of_selecting_red_star_shines_over_china_l752_752460

theorem probability_of_selecting_red_star_shines_over_china :
  let books := ["The Red Star Shines Over China", "Red Rock", "The Long March", "How Steel is Made"],
      favorable_outcomes := 1,
      total_outcomes := books.length
  in (favorable_outcomes : ℚ) / total_outcomes = 1 / 4 :=
by
  sorry

end probability_of_selecting_red_star_shines_over_china_l752_752460


namespace multiple_optimal_solutions_for_z_l752_752734

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 0 2
def B := Point.mk (-2) (-2)
def C := Point.mk 2 0

def z (a : ℝ) (P : Point) : ℝ := P.y - a * P.x

def maxz_mult_opt_solutions (a : ℝ) : Prop :=
  z a A = z a B ∨ z a A = z a C ∨ z a B = z a C

theorem multiple_optimal_solutions_for_z :
  (maxz_mult_opt_solutions (-1)) ∧ (maxz_mult_opt_solutions 2) :=
by
  sorry

end multiple_optimal_solutions_for_z_l752_752734


namespace cannot_fill_13_l752_752549

theorem cannot_fill_13 : ∀ S i, S = 1 → i = 3 → 
  (while i < 13 do 
     S := S * i; 
     i := i + 2) → 
  S ≠ 1 * 3 * 5 * 7 * 9 * 11 * 13 :=
by
  intros S i hS hi hwhile
  sorry

end cannot_fill_13_l752_752549


namespace number_of_choices_l752_752474

theorem number_of_choices (students lectures : ℕ) (h_students : students = 5) (h_lectures : lectures = 4) : 
  let choices := lectures ^ students in choices = 4 ^ 5 :=
by {
  sorry
}

end number_of_choices_l752_752474


namespace log_diff_lt_zero_l752_752982

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

theorem log_diff_lt_zero (b c m n : ℝ) 
  (h_symm : ∀ x, f (x) b c = f (1 - x) b c) 
  (h_pos : f 0 b c > 0) 
  (h_zeros : f m b c = 0 ∧ f n b c = 0)
  (h_mn : m ≠ n ∧ m > 0 ∧ n > 0) : 
  log 3 m - log (1 / 3) n < 0 := 
by 
  sorry

end log_diff_lt_zero_l752_752982


namespace number_of_non_congruent_squares_in_6_by_6_grid_l752_752645

theorem number_of_non_congruent_squares_in_6_by_6_grid :
  let total_standard_squares := 25 + 16 + 9 + 4 + 1 in
  let total_alternative_squares := 25 + 9 + 40 + 24 in
  total_standard_squares + total_alternative_squares = 153 := 
by
  sorry

end number_of_non_congruent_squares_in_6_by_6_grid_l752_752645


namespace cost_to_fill_half_of_CanB_l752_752889

theorem cost_to_fill_half_of_CanB (r h : ℝ) (C_cost : ℝ) (VC VB : ℝ) 
(h1 : VC = 2 * VB) 
(h2 : VB = Real.pi * r^2 * h) 
(h3 : VC = Real.pi * (2 * r)^2 * (h / 2)) 
(h4 : C_cost = 16):
  C_cost / 4 = 4 :=
by
  sorry

end cost_to_fill_half_of_CanB_l752_752889


namespace wall_width_l752_752127

theorem wall_width (w h l : ℝ)
  (h_eq_6w : h = 6 * w)
  (l_eq_7h : l = 7 * h)
  (V_eq : w * h * l = 86436) :
  w = 7 :=
by
  sorry

end wall_width_l752_752127


namespace number_of_valid_arrangements_l752_752745

-- Define the total number of boys and girls
def boys : ℕ := 4
def girls : ℕ := 3

-- Define the number of people to be selected
def selection_count : ℕ := 3

-- Define the total number of permutations selecting 3 out of 7 people (4 boys + 3 girls)
def total_permutations : ℕ := number_of_permutations 7 selection_count

-- Define the number of permutations selecting 3 out of 4 boys
def boys_only_permutations : ℕ := number_of_permutations boys selection_count

-- Define the number of valid permutations with at least one girl
def valid_permutations_with_at_least_one_girl : ℕ := total_permutations - boys_only_permutations

-- The theorem we need to prove
theorem number_of_valid_arrangements : valid_permutations_with_at_least_one_girl = 186 :=
sorry

-- Helper function for permutations (P(n, k) = n! / (n - k)!)
def number_of_permutations (n k : ℕ) : ℕ :=
(n.factorial) / ((n - k).factorial)

end number_of_valid_arrangements_l752_752745


namespace spherical_to_rectangular_coords_l752_752910

theorem spherical_to_rectangular_coords : 
  ∀ (ρ θ φ : ℝ), 
  ρ = 3 → θ = π / 4 → φ = π / 6 → 
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ) = 
  (3 * Real.sin (π / 6) * Real.cos (π / 4), 3 * Real.sin (π / 6) * Real.sin (π / 4), 3 * Real.cos (π / 6)) :=
by
  intro ρ θ φ hρ hθ hφ
  simp [hρ,hθ,hφ]
  sorry

end spherical_to_rectangular_coords_l752_752910


namespace cos_value_given_sin_l752_752611

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) : 
  Real.cos (π / 3 - α) = 3 / 5 :=
sorry

end cos_value_given_sin_l752_752611


namespace remainder_xyz_mod7_condition_l752_752657

-- Define variables and conditions
variables (x y z : ℕ)
theorem remainder_xyz_mod7_condition (hx : x < 7) (hy : y < 7) (hz : z < 7)
  (h1 : x + 3 * y + 2 * z ≡ 0 [MOD 7])
  (h2 : 3 * x + 2 * y + z ≡ 2 [MOD 7])
  (h3 : 2 * x + y + 3 * z ≡ 3 [MOD 7]) :
  (x * y * z % 7) ≡ 1 [MOD 7] := sorry

end remainder_xyz_mod7_condition_l752_752657


namespace least_positive_integer_divisible_l752_752455

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem least_positive_integer_divisible (n : ℕ) (h₁ : ∀ i ∈ (List.range 10).map (λ n, n + 1), i ∣ n) (h₂ : n > 1000) : n = 2520 :=
by
  sorry

end least_positive_integer_divisible_l752_752455


namespace triangle_inequalities_l752_752386

variables {α : Type} [RealField α]

-- Define points A, B, C, M and functions to represent the distances and projections
variables {A B C M : α} 
variables (R_a R_b R_c d_a d_b d_c : α)

-- Assume M is inside the triangle ABC
axiom M_inside_triangle : True

-- The Lean theorem statement equivalent to the given problem
theorem triangle_inequalities 
  (h1 : R_a R_b R_c ≥ 8 * d_a * d_b * d_c)
  (h2 : R_a + R_b + R_c ≥ 2 * (d_a + d_b + d_c))
  (h3 : 1 / R_a + 1 / R_b + 1 / R_c ≤ 1 / 2 * (1 / d_a + 1 / d_b + 1 / d_c)) : 
  True := sorry

end triangle_inequalities_l752_752386


namespace sum_of_cubes_l752_752103

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by 
  sorry

end sum_of_cubes_l752_752103


namespace cubes_sum_l752_752112

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l752_752112


namespace log_power_sum_l752_752654

theorem log_power_sum (a b : Real) (ha : a = log 25) (hb : b = log 49) : 
  5^(a/b) + 7^(b/a) = 12 :=
by
  sorry

end log_power_sum_l752_752654


namespace find_digits_l752_752004

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 7
noncomputable def C : ℕ := 1

def row_sums_equal (A B C : ℕ) : Prop :=
  (4 + 2 + 3 = 9) ∧
  (A + 6 + 1 = 9) ∧
  (1 + 2 + 6 = 9) ∧
  (B + 2 + C = 9)

def column_sums_equal (A B C : ℕ) : Prop :=
  (4 + 1 + B = 12) ∧
  (2 + A + 2 = 6 + 3 + 1) ∧
  (3 + 1 + 6 + C = 12)

def red_cells_sum_equals_row_sum (A B C : ℕ) : Prop :=
  (A + B + C = 9)

theorem find_digits :
  ∃ (A B C : ℕ),
    row_sums_equal A B C ∧
    column_sums_equal A B C ∧
    red_cells_sum_equals_row_sum A B C ∧
    100 * A + 10 * B + C = 371 :=
by
  exact ⟨3, 7, 1, ⟨rfl, rfl, rfl, rfl⟩⟩

end find_digits_l752_752004


namespace total_cost_correct_l752_752076

-- Define the conditions
def uber_cost : ℤ := 22
def lyft_additional_cost : ℤ := 3
def taxi_additional_cost : ℤ := 4
def tip_percentage : ℚ := 0.20

-- Define the variables for cost of Lyft and Taxi based on the problem
def lyft_cost : ℤ := uber_cost - lyft_additional_cost
def taxi_cost : ℤ := lyft_cost - taxi_additional_cost

-- Calculate the tip
def tip : ℚ := taxi_cost * tip_percentage

-- Final total cost including the tip
def total_cost : ℚ := taxi_cost + tip

-- The theorem to prove
theorem total_cost_correct :
  total_cost = 18 := by
  sorry

end total_cost_correct_l752_752076


namespace monotonic_intervals_when_a_is_1_minimum_value_of_2f_x1_minus_f_x2_l752_752630

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*a*x + Real.log x

-- First part: monotonicity when a = 1
theorem monotonic_intervals_when_a_is_1 :
  ∀ x : ℝ, x > 0 → (f x 1)' > 0 := sorry

-- Second part: minimum value of 2f(x1) - f(x2)
theorem minimum_value_of_2f_x1_minus_f_x2 (a x1 x2 : ℝ) (h1 : 2*x1^2 - 2*a*x1 + 1 = 0)
  (h2 : 2*x2^2 - 2*a*x2 + 1 = 0) (h3 : x1 < x2) (h4 : a > Real.sqrt 2) :
  2 * f x1 a - f x2 a = - (1 + 4 * Real.log 2) / 2 := sorry

end monotonic_intervals_when_a_is_1_minimum_value_of_2f_x1_minus_f_x2_l752_752630


namespace cubic_root_solution_l752_752398

theorem cubic_root_solution :
  ∀ x : ℂ, (x^3 + 2 * x^2 + 4 * x + 2 = 0) ↔ (x = -1 ∨ x = -1 + complex.I ∨ x = -1 - complex.I) :=
by
  intro x
  apply iff.intro
  -- Proof steps would go here
  sorry

end cubic_root_solution_l752_752398


namespace average_percentage_l752_752125

theorem average_percentage (num_students1 num_students2 : Nat) (avg1 avg2 avg : Nat) :
  num_students1 = 15 ->
  avg1 = 73 ->
  num_students2 = 10 ->
  avg2 = 88 ->
  (num_students1 * avg1 + num_students2 * avg2) / (num_students1 + num_students2) = avg ->
  avg = 79 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_percentage_l752_752125


namespace sum_of_cubes_l752_752098

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752098


namespace employees_surveyed_correct_l752_752302

noncomputable def total_employees_surveyed : ℝ :=
  let uninsured := 104
  let part_time := 54
  let p_neither := 0.5671641791044776
  -- Calculating the number of uninsured employees who work part time
  let uninsured_and_part_time := 0.125 * uninsured
  -- Defining the formula for neither part time nor uninsured
  (part_time - uninsured - part_time + uninsured_and_part_time ) / ( sorry )

-- The theorem to prove
theorem employees_surveyed_correct : E = 305 :=
by {
  have h1 : (E - 104 - 54 + 13) / E = 0.5671641791044776 := sorry,
  have h2 : (E - 132) / E = 0.5671641791044776 := sorry,
  have h3 : E * 0.4328358208955224 = 132 := by { rw ←h2, norm_num1, },
  have h4 : E = 305 := by { simp[h3], },
  exact E = 305
}


end employees_surveyed_correct_l752_752302


namespace range_of_x_in_function_l752_752684

theorem range_of_x_in_function (x : ℝ) : (y = 1/(x + 3) → x ≠ -3) :=
sorry

end range_of_x_in_function_l752_752684


namespace n_value_l752_752121

-- Condition definitions
variables {R : Type*} [linear_ordered_field R]

-- A function representing trigonometric sine in radians
noncomputable def sine (x : R) : R := sin x

-- Helper lemma: reflection of the given problem's setup
lemma distance_n_gon (R n : ℕ) (A_1 A_2 A_3 A_4 : ℕ) :
  A_2 = A_1 + 1 ∧ 
  A_3 = A_1 + 2 ∧ 
  A_4 = A_1 + 3 ∧ 
  A_1 A_2 = 2 * R * sine (π / n) ∧ 
  A_1 A_3 = 2 * R * sine (2 * π / n) ∧ 
  A_1 A_4 = 2 * R * sine (3 * π / n) :=
sorry

-- Main theorem representing the problem
theorem n_value (A_1 A_2 A_3 A_4 : ℕ) (R n : ℕ) 
  (h1 : A_1A_2 = 2 * R * sine (π / n))
  (h2 : A_1A_3 = 2 * R * sine (2 * π / n))
  (h3 : A_1A_4 = 2 * R * sine (3 * π / n)) 
  (h4 : (1 : R) / (2 * R * sine (π / n)) = (1 / (2 * R * sine (2 * π / n)) + 1 / (2 * R * sine (3 * π / n)))) :
  n = 7 :=
sorry

end n_value_l752_752121


namespace find_b_c_d_l752_752907

theorem find_b_c_d :
  ∃ b c d : ℤ, 
    (∀ n : ℕ, n > 0 → a n = b * int.floor (real.sqrt_real (real.sqrt_real ((n : ℝ) + (c : ℝ)))) + d) ∧ 
    b + c + d = 1 :=
begin
  -- Definitions for a sequence where each integer k appears k^2 times
  -- We would define a function 'a' here based on the problem's conditions.
  sorry
end

end find_b_c_d_l752_752907


namespace square_perimeter_l752_752760

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by 
  have h1 : s = 1 := 
  begin
    -- Considering non-zero s, divide both sides by s
    by_contradiction hs,
    have hs' : s ≠ 0 := λ h0, hs (by simp [h0] at h),
    exact hs (by simpa [h, hs'] using h),
  end,
  rw h1,
  norm_num

end square_perimeter_l752_752760


namespace find_lambda_l752_752700

section
variables (a b : ℝ × ℝ) (λ : ℝ)
def vector_a := (1, 2)
def vector_b := (2, 4)
def vector_c := (λ * vector_a.1 + vector_b.1, λ * vector_a.2 + vector_b.2)
def is_perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem find_lambda (λ : ℝ) 
  (ha : a = vector_a) 
  (hb : b = vector_b) 
  (hc : vector_c λ = λ * a + b) 
  (h_perp : is_perpendicular (vector_c λ) a) : 
  λ = -2 := 
sorry
end

end find_lambda_l752_752700


namespace john_drinks_amount_l752_752437

noncomputable def initial_amount : ℚ := 3 / 4

noncomputable def maria_fraction : ℚ := 1 / 2

noncomputable def john_fraction : ℚ := 1 / 3

theorem john_drinks_amount :
  let maria_drinks := maria_fraction * initial_amount
      remaining := initial_amount - maria_drinks
      john_drinks := john_fraction * remaining
  in john_drinks = 1 / 8 :=
by
  sorry

end john_drinks_amount_l752_752437


namespace number_of_students_chose_banana_l752_752308

theorem number_of_students_chose_banana (total_students : ℕ) (percentage_chose_banana : ℕ) :
  total_students = 100 → percentage_chose_banana = 20 → (total_students * percentage_chose_banana / 100) = 20 :=
by
  intros h1 h2
  rw [h1, h2]  -- Simplify using the given conditions
  norm_num  -- Compute the arithmetic expression
  sorry  -- Placeholder for proof

end number_of_students_chose_banana_l752_752308


namespace largest_common_number_in_arithmetic_sequences_l752_752183

theorem largest_common_number_in_arithmetic_sequences (n : ℕ) :
  (∃ a1 a2 : ℕ, a1 = 5 + 8 * n ∧ a2 = 3 + 9 * n ∧ a1 = a2 ∧ 1 ≤ a1 ∧ a1 ≤ 150) →
  (a1 = 93) :=
by
  sorry

end largest_common_number_in_arithmetic_sequences_l752_752183


namespace non_congruent_squares_on_6x6_grid_l752_752648

theorem non_congruent_squares_on_6x6_grid :
  let a1 := 5 * 5,
      a2 := 4 * 4,
      a3 := 3 * 3,
      a4 := 2 * 2,
      a5 := 1 * 1,
      d1 := 5 * 5,
      d2 := 4 * 4,
      d3 := 5 * 4 + 4 * 5,
      d4 := 5 * 3 + 3 * 5
  in a1 + a2 + a3 + a4 + a5 + d1 + d2 + d3 + d4 = 166 :=
by 
  let a1 := 5 * 5
  let a2 := 4 * 4
  let a3 := 3 * 3
  let a4 := 2 * 2
  let a5 := 1 * 1
  let d1 := 5 * 5
  let d2 := 4 * 4
  let d3 := 5 * 4 + 4 * 5
  let d4 := 5 * 3 + 3 * 5
  show a1 + a2 + a3 + a4 + a5 + d1 + d2 + d3 + d4 = 166
  sorry

end non_congruent_squares_on_6x6_grid_l752_752648


namespace shaded_area_l752_752310

theorem shaded_area (r : ℝ) (π : ℝ) (shaded_area : ℝ) (h_r : r = 4) (h_π : π = 3) : shaded_area = 32.5 :=
by
  sorry

end shaded_area_l752_752310


namespace cubes_sum_l752_752114

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l752_752114


namespace eval_expression_at_neg_one_l752_752118

variable (x : ℤ)

theorem eval_expression_at_neg_one : x = -1 → 3 * x ^ 2 + 2 * x - 1 = 0 := by
  intro h
  rw [h]
  show 3 * (-1) ^ 2 + 2 * (-1) - 1 = 0
  sorry

end eval_expression_at_neg_one_l752_752118


namespace domain_of_v_l752_752085

noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

theorem domain_of_v :
  {x : ℝ | v x ∈ ℝ} = {x : ℝ | x > 1} :=
by
  sorry

end domain_of_v_l752_752085


namespace hyperbola_eccentricity_l752_752264

theorem hyperbola_eccentricity (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : b = sqrt 3 * a) (h₄ : c^2 = a^2 + b^2) : 
  (c / a) = 2 :=
by
  sorry

end hyperbola_eccentricity_l752_752264


namespace sum_of_series_gt_f_half_l752_752696

theorem sum_of_series_gt_f_half 
  (f : ℝ → ℝ)
  (h0 : f 0 = 0)
  (h1 : ∀ x y ∈ (-∞, -1) ∪ (1, ∞), f (1 / x) + f (1 / y) = f ((x + y) / (1 + x * y)))
  (h2 : ∀ x ∈ (-1:ℝ, 0:ℝ), f x > 0) :
  ∑' n, f (1 / (n^2 + 7*n + 11)) > f (1/2) :=
sorry

end sum_of_series_gt_f_half_l752_752696


namespace shaded_region_area_l752_752428

-- Define the conditions
def num_congruent_squares : ℕ := 25
def PQ : ℝ := 10
def num_squares_in_large_square : ℕ := 16

-- Calculate the area of the large square having diagonal PQ
def area_large_square : ℝ := (PQ^2) / 2

-- Calculate the area of one small congruent square
def area_one_small_square : ℝ := area_large_square / num_squares_in_large_square

-- The theorem which we need to prove
theorem shaded_region_area :
  num_congruent_squares * area_one_small_square = 78.125 := by
  sorry

end shaded_region_area_l752_752428


namespace man_son_age_ratio_is_two_to_one_l752_752497

-- Define the present age of the son
def son_present_age := 33

-- Define the present age of the man
def man_present_age := son_present_age + 35

-- Define the son's age in two years
def son_age_in_two_years := son_present_age + 2

-- Define the man's age in two years
def man_age_in_two_years := man_present_age + 2

-- Define the expected ratio of the man's age to son's age in two years
def ratio := man_age_in_two_years / son_age_in_two_years

-- Theorem statement verifying the ratio
theorem man_son_age_ratio_is_two_to_one : ratio = 2 := by
  -- Note: Proof not required, so we use sorry to denote the missing proof
  sorry

end man_son_age_ratio_is_two_to_one_l752_752497


namespace melanie_total_dimes_l752_752725

/-- Melanie had 7 dimes in her bank. Her dad gave her 8 dimes. Her mother gave her 4 dimes. -/
def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

/-- How many dimes does Melanie have now? -/
theorem melanie_total_dimes : initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end melanie_total_dimes_l752_752725


namespace customer_paid_amount_l752_752058

theorem customer_paid_amount (cost_price : ℕ) (percentage_increase : ℕ) (additional_amount : ℕ) (total_paid : ℕ) 
    (h1 : cost_price = 6925) 
    (h2 : percentage_increase = 24) 
    (h3 : additional_amount = (percentage_increase * cost_price) / 100) : 
    total_paid = 8587 := 
by
    have h4 : additional_amount = 1662, by sorry
    have h5 : total_paid = cost_price + additional_amount, by sorry
    rw [h1, h4] at h5
    exact h5

end customer_paid_amount_l752_752058


namespace intersection_result_l752_752634

open Set

variable {R : Type*} [LinearOrderedField R]

def M : Set R := { x | log 3 x < 3 }
def N : Set R := { x | x^2 - 4 * x - 5 > 0 }
def complement_N : Set R := { x | -1 ≤ x ∧ x ≤ 5 }
def intersection := { x | 0 < x ∧ x ≤ 5 }

theorem intersection_result :
  M ∩ complement_N = intersection := by sorry

end intersection_result_l752_752634


namespace number_of_false_propositions_l752_752720

-- Define the polynomial and its roots
def polynomial := λ (x : ℝ), x^2 + 3 * x - 1

-- Proposition p: The two roots of the equation x^2 + 3x - 1 = 0 have opposite signs
def p := ∃ (r s : ℝ), polynomial r = 0 ∧ polynomial s = 0 ∧ r * s < 0

-- Proposition q: The sum of the two roots of the equation x^2 + 3x - 1 = 0 is 3
def q := ∃ (r s : ℝ), polynomial r = 0 ∧ polynomial s = 0 ∧ r + s = 3

-- Proving the number of false propositions among ¬p, ¬q, p∧q, and p∨q is 2
theorem number_of_false_propositions : p ∧ ¬q → (nat.count (¬p :: ¬q :: (p ∧ q) :: (p ∨ q) :: []) false) = 2 :=
by
  sorry

end number_of_false_propositions_l752_752720


namespace proof_problem_l752_752586

variable {n : ℕ}
variable {r s t u v : ℕ → ℝ}
variable (h_r : ∀ i, 1 ≤ r i)
variable (h_s : ∀ i, 1 ≤ s i)
variable (h_t : ∀ i, 1 ≤ t i)
variable (h_u : ∀ i, 1 ≤ u i)
variable (h_v : ∀ i, 1 ≤ v i)

noncomputable def R := (1 / (n : ℝ)) * (∑ i in Finset.range n, r i)
noncomputable def S := (1 / (n : ℝ)) * (∑ i in Finset.range n, s i)
noncomputable def T := (1 / (n : ℝ)) * (∑ i in Finset.range n, t i)
noncomputable def U := (1 / (n : ℝ)) * (∑ i in Finset.range n, u i)
noncomputable def V := (1 / (n : ℝ)) * (∑ i in Finset.range n, v i)

theorem proof_problem :
  (∏ i in Finset.range n, (r i * s i * t i * u i * v i + 1) / (r i * s i * t i * u i * v i - 1)) 
  ≥ ((R * S * T * U * V + 1) / (R * S * T * U * V - 1)) ^ n
:= by
  sorry

end proof_problem_l752_752586


namespace parts_of_second_liquid_l752_752731

variable (x : ℝ)

def first_liquid_water_percentage := 0.20
def second_liquid_water_percentage := 0.35
def water_in_new_mixture_percentage := 0.24285714285714285
def parts_first_liquid := 10

-- Equation based on problem statement
def water_in_first_liquid := first_liquid_water_percentage * parts_first_liquid
def water_in_second_liquid := second_liquid_water_percentage * x
def total_liquid_parts := parts_first_liquid + x
def water_in_new_mixture := water_in_new_mixture_percentage * total_liquid_parts

theorem parts_of_second_liquid:
  water_in_first_liquid + water_in_second_liquid = water_in_new_mixture → x = 4 :=
by
  sorry

end parts_of_second_liquid_l752_752731


namespace pentagon_tiles_plane_l752_752234

-- Define the properties of the pentagon
structure Pentagon :=
(AB BC CD DE : ℝ)
(angleB angleD : ℝ)
(equalSides : AB = BC ∧ BC = CD ∧ CD = DE)
(rightAngles : angleB = 90 ∧ angleD = 90)

-- The theorem that such pentagons can tile the plane
theorem pentagon_tiles_plane (P : Pentagon) : 
  P.equalSides → P.rightAngles → ∃ tiling : set (set Pentagon), true :=
by
  intros h1 h2
  apply Exists.intro (λ _, true)
  sorry

end pentagon_tiles_plane_l752_752234


namespace total_number_of_triangles_is_13_l752_752309

-- Define the number of small triangles in the first large triangle
def small_triangles_first_triangle : Nat := 3

-- Define the additional triangles in the first large triangle
def additional_triangles_first_triangle : Nat := 1

-- Define the number of small triangles in the second large triangle
def small_triangles_second_triangle : Nat := 3

-- Define the number of medium triangles in the second large triangle
def medium_triangles_second_triangle : Nat := 4

-- Define the entire single large triangle counts in each large triangle
def one_large_triangle : Int := 1

-- Define the total number of triangles in the first large triangle
def total_triangles_first_triangle : Nat := 
  small_triangles_first_triangle + additional_triangles_first_triangle + one_large_triangle

-- Define the total number of triangles in the second large triangle
def total_triangles_second_triangle : Nat :=
  small_triangles_second_triangle + medium_triangles_second_triangle + one_large_triangle

-- Final theorem to prove total number of triangles in both triangles
theorem total_number_of_triangles_is_13 :
  total_triangles_first_triangle + total_triangles_second_triangle = 13 :=
by
  sorry

end total_number_of_triangles_is_13_l752_752309


namespace Punta_position_l752_752494

theorem Punta_position (N x y p : ℕ) (h1 : N = 36) (h2 : x = y / 4) (h3 : x + y = 35) : p = 8 := by
  sorry

end Punta_position_l752_752494


namespace compounding_frequency_l752_752411

variable (i : ℝ) (EAR : ℝ)

/-- Given the nominal annual rate (i = 6%) and the effective annual rate (EAR = 6.09%), 
    prove that the frequency of payment (n) is 4. -/
theorem compounding_frequency (h1 : i = 0.06) (h2 : EAR = 0.0609) : 
  ∃ n : ℕ, (1 + i / n)^n - 1 = EAR ∧ n = 4 := sorry

end compounding_frequency_l752_752411


namespace part1_part2_l752_752192

-- Definitions based on the given conditions
def daily_sales (x : ℝ) : ℝ := 800 * x + 400
def profit_per_zongzi (x : ℝ) : ℝ := 2 - x
def total_profit (x : ℝ) : ℝ := daily_sales(x) * profit_per_zongzi(x)

-- Theorem 1: For a price reduction of 0.2 yuan
theorem part1 (h₀ : 0.2 ∈ ℝ) :
  daily_sales h₀ = 560 ∧ total_profit h₀ = 1008 :=
sorry

-- Theorem 2: For a total profit of 1200 yuan
theorem part2 :
  ∃ x ∈ ℝ, total_profit x = 1200 ∧ daily_sales x ≤ 1100 := 
sorry

end part1_part2_l752_752192


namespace parabola_equation_dist_midpoint_directrix_l752_752136

-- Problem 1: Parabola through point with vertex at origin and axes of symmetry
theorem parabola_equation (x y: ℝ) (p : ℝ) (h1 : (0, 0) = (0, 0)) 
   (h2 : y ^ 2 = -2 * p * x) (h3 : (-2, -4) ∈ set_of (λ (p : ℝ × ℝ), y ^ 2 = -2 * p * x)) :
   (y ^ 2 = -8 * x) ∨ (x ^ 2 = -y) :=
sorry

-- Problem 2: Distance from midpoint to directrix
theorem dist_midpoint_directrix (p : ℝ) (h1 : 0 < p) (A : ℝ × ℝ) (F : ℝ × ℝ) (B : ℝ × ℝ) 
   (h2 : A = (0, 2)) (h3 : F = (p / 2, 0)) (h4 : B = ((p / 4, 1))) (h5 : B ∈ set_of (λ (p : ℝ × ℝ), y ^ 2 = 2 * p * x)) :
   dist B (set_of (λ (x : ℝ), x = -p / 2)) = 3 * sqrt 2 / 4 :=
sorry

end parabola_equation_dist_midpoint_directrix_l752_752136


namespace part1_part2_l752_752305

def pointA : (ℝ × ℝ) := (1, 2)
def pointB : (ℝ × ℝ) := (-2, 3)
def pointC : (ℝ × ℝ) := (8, -5)

-- Definitions of the vectors
def OA : (ℝ × ℝ) := pointA
def OB : (ℝ × ℝ) := pointB
def OC : (ℝ × ℝ) := pointC
def AB : (ℝ × ℝ) := (pointB.1 - pointA.1, pointB.2 - pointA.2)

-- Part 1: Proving the values of x and y
theorem part1 : ∃ (x y : ℝ), OC = (x * OA.1 + y * OB.1, x * OA.2 + y * OB.2) ∧ x = 2 ∧ y = -3 :=
by
  sorry

-- Part 2: Proving the value of m when vectors are parallel
theorem part2 : ∃ (m : ℝ), ∃ k : ℝ, AB = (k * (m + 8), k * (2 * m - 5)) ∧ m = 1 :=
by
  sorry

end part1_part2_l752_752305


namespace radius_of_tangent_circle_l752_752483

noncomputable def circle_radius {r : ℝ} (D E F : Point) : Prop :=
  ∃ O : Point,
    (triangle DEF ∧
     is_30_60_90_triangle DEF ∧
     length DE = 2 ∧
     is_tangent_to_coordinate_axes O r ∧
     is_tangent_to_hypotenuse O r (hypotenuse DEF) ∧
     r = (5 + sqrt(3)) / 2)

theorem radius_of_tangent_circle :
  ∃ r, circle_radius DEF :=
begin
  sorry
end

end radius_of_tangent_circle_l752_752483


namespace wall_width_l752_752769

theorem wall_width (V h l w : ℝ) (h_cond : h = 6 * w) (l_cond : l = 42 * w) (vol_cond : 252 * w^3 = 129024) : w = 8 := 
by
  -- Proof is omitted; required to produce lean statement only
  sorry

end wall_width_l752_752769


namespace child_ticket_cost_l752_752490

def cost_of_adult_ticket : ℕ := 22
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 2
def total_family_cost : ℕ := 58
def cost_of_child_ticket : ℕ := 7

theorem child_ticket_cost :
  2 * cost_of_adult_ticket + number_of_children * cost_of_child_ticket = total_family_cost :=
by
  sorry

end child_ticket_cost_l752_752490


namespace coffee_last_days_l752_752877

theorem coffee_last_days (coffee_weight : ℕ) (cups_per_lb : ℕ) (angie_daily : ℕ) (bob_daily : ℕ) (carol_daily : ℕ) 
  (angie_coffee_weight : coffee_weight = 3) (cups_brewing_rate : cups_per_lb = 40)
  (angie_consumption : angie_daily = 3) (bob_consumption : bob_daily = 2) (carol_consumption : carol_daily = 4) : 
  ((coffee_weight * cups_per_lb) / (angie_daily + bob_daily + carol_daily) = 13) := by
  sorry

end coffee_last_days_l752_752877


namespace ceil_sqrt_169_eq_13_l752_752195

theorem ceil_sqrt_169_eq_13 : Int.ceil (Real.sqrt 169) = 13 := by
  sorry

end ceil_sqrt_169_eq_13_l752_752195


namespace find_digits_l752_752003

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 7
noncomputable def C : ℕ := 1

def row_sums_equal (A B C : ℕ) : Prop :=
  (4 + 2 + 3 = 9) ∧
  (A + 6 + 1 = 9) ∧
  (1 + 2 + 6 = 9) ∧
  (B + 2 + C = 9)

def column_sums_equal (A B C : ℕ) : Prop :=
  (4 + 1 + B = 12) ∧
  (2 + A + 2 = 6 + 3 + 1) ∧
  (3 + 1 + 6 + C = 12)

def red_cells_sum_equals_row_sum (A B C : ℕ) : Prop :=
  (A + B + C = 9)

theorem find_digits :
  ∃ (A B C : ℕ),
    row_sums_equal A B C ∧
    column_sums_equal A B C ∧
    red_cells_sum_equals_row_sum A B C ∧
    100 * A + 10 * B + C = 371 :=
by
  exact ⟨3, 7, 1, ⟨rfl, rfl, rfl, rfl⟩⟩

end find_digits_l752_752003


namespace longest_train_clearance_time_correct_l752_752073

noncomputable def time_to_clear 
  (length1 : ℝ) (length2 : ℝ) (length3 : ℝ)
  (speed1_kmph : ℝ) (speed2_kmph : ℝ) (speed3_kmph : ℝ)
  (initial_dist : ℝ) (dist_increment : ℝ) (max_dist : ℝ)
  (curv_angle : ℝ) : ℝ :=
let speed1 := speed1_kmph * (1000 / 3600)
let speed2 := speed2_kmph * (1000 / 3600)
let speed3 := speed3_kmph * (1000 / 3600)
let relative_speed1 := speed1 + speed3
let relative_speed2 := speed2 + speed3
let longest_train := length3
let chosen_relative_speed := if relative_speed1 > relative_speed2 then relative_speed2 else relative_speed1
in longest_train / chosen_relative_speed

theorem longest_train_clearance_time_correct :
  time_to_clear 180 240 300 50 35 45 250 100 450 10 = 13.50 := 
sorry

end longest_train_clearance_time_correct_l752_752073


namespace domain_of_v_l752_752088

noncomputable def v (x : ℝ) : ℝ := 1 / real.sqrt (x - 1)

theorem domain_of_v :
  {x : ℝ | ∃ y : ℝ, v(x) = y} = set.Ioi 1 := 
by
  sorry

end domain_of_v_l752_752088


namespace clara_wheel_replacement_l752_752173

theorem clara_wheel_replacement :
  ∀ n, n = 18 → (7 * (n - 1)) % 12 = 11 :=
by
  intro n hn
  rw [hn]
  sorry

end clara_wheel_replacement_l752_752173


namespace proof_x_plus_y_geq_0_l752_752968

variable {x y : ℝ}

def log2_3 := Real.log 3 / Real.log 2
def log5_3 := Real.log 3 / Real.log 5

theorem proof_x_plus_y_geq_0 (hx : 0 < log2_3) (hy1 : 0 < log5_3) (hy2 : log5_3 < 1) 
(h : (log2_3^x - log5_3^x) ≥ (log2_3^(-y) - log5_3^(-y))) : 
x + y ≥ 0 :=
sorry

end proof_x_plus_y_geq_0_l752_752968


namespace sum_of_cubes_l752_752095

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l752_752095


namespace lawn_area_l752_752743

theorem lawn_area (s l : ℕ) (hs: 5 * s = 10) (hl: 5 * l = 50) (hposts: 2 * (s + l) = 24) (hlen: l + 1 = 3 * (s + 1)) :
  s * l = 500 :=
by {
  sorry
}

end lawn_area_l752_752743


namespace students_facing_coach_sixty_sixth_student_number_l752_752783

universe u

def students := Finset ℕ
def multiples (n m : ℕ) := {x : ℕ // x ≤ n ∧ x % m = 0}
def not_multiples (n : ℕ) (m : ℕ) := {x : ℕ // x ≤ n ∧ x % m ≠ 0}

def A : students := Finset.range 241 -- [1, 2, ..., 240]

def A_3 := Finset.filter (λ x, x % 3 == 0) A
def A_5 := Finset.filter (λ x, x % 5 == 0) A
def A_7 := Finset.filter (λ x, x % 7 == 0) A

def A_3_or_5_or_7 := A_3 ∪ A_5 ∪ A_7

def facing_students := A.filter (λ x, x ∉ A_3_or_5_or_7) -- Students not turning

def number_of_facing_students := (facing_students.card : ℕ)

def facing_students_seq := (facing_students.sort)

-- Find the 66th student in the sorted sequence
def sixty_sixth_student := facing_students_seq.nth_le 65 (by sorry)

theorem students_facing_coach : number_of_facing_students = 109 :=
by sorry

theorem sixty_sixth_student_number : (sixty_sixth_student : ℕ) = 159 :=
by sorry

end students_facing_coach_sixty_sixth_student_number_l752_752783


namespace distribute_balls_l752_752555

theorem distribute_balls (balls : Finset (Fin 7)) (P1 P2 : Finset (Fin 7)) :
  (P1.card ≥ 2) ∧ (P2.card ≥ 2) ∧ (balls.card = 7) ∧ (P1 ∪ P2 = balls) ∧ (P1 ∩ P2 = ∅) →
  card {d : balls → {1, 2} // (∀ b, d b = 1 ↔ b ∈ P1) ∧ (∀ b, d b = 2 ↔ b ∈ P2)} = 112 :=
sorry

end distribute_balls_l752_752555


namespace largest_digit_divisible_by_6_l752_752090

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N = 8 ∧ (45670 + N) % 6 = 0 :=
sorry

end largest_digit_divisible_by_6_l752_752090


namespace solveProblem_l752_752718

noncomputable def problemStatement : Prop :=
  ∃ (p q r A B C : ℝ),
    (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
      1 / (s^3 - 14 * s^2 + 49 * s - 24) = A / (s - p) + B / (s - q) + C / (s - r) ) ∧
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (A + B + C = 0) ∧
    (A * (q + r) + B * (p + r) + C * (p + q) = -49) ∧
    (A * q * r + B * p * r + C * p * q = 24) ∧
    (p + q + r = 14) ∧
    (p * q + p * r + q * r = 49) ∧
    (p * q * r = 24) ∧
    (1 / A + 1 / B + 1 / C = 123)
    
theorem solveProblem : problemStatement :=
begin
  sorry
end

end solveProblem_l752_752718


namespace conjugate_of_product_l752_752223

noncomputable def i : ℂ := Complex.i
noncomputable def z : ℂ := √3 - i
noncomputable def w : ℂ := 1 + √3 * i

theorem conjugate_of_product : conj (z * w) = 2 * √3 - 2 * i :=
by
  sorry

end conjugate_of_product_l752_752223


namespace min_value_on_interval_l752_752981

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

theorem min_value_on_interval (a : ℝ) (h : f 2 a = 20) :
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, f x a = -7 :=
begin
  sorry
end

end min_value_on_interval_l752_752981


namespace total_cost_proof_l752_752077

def uber_cost : ℤ := 22
def lyft_cost : ℤ := uber_cost - 3
def taxi_cost : ℤ := lyft_cost - 4
def tip : ℤ := (taxi_cost * 20) / 100
def total_cost : ℤ := taxi_cost + tip

theorem total_cost_proof :
  total_cost = 18 :=
by
  sorry

end total_cost_proof_l752_752077


namespace range_of_f_l752_752062

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 2^x - 5 else 3 * Real.sin x

-- State the theorem about the range of f(x)
theorem range_of_f :
  set.range f = set.Icc (-5 : ℝ) 3 :=
sorry

end range_of_f_l752_752062


namespace find_remainder_2500th_term_l752_752545

theorem find_remainder_2500th_term : 
    let seq_position (n : ℕ) := n * (n + 1) / 2 
    let n := ((1 + Int.ofNat 20000).natAbs.sqrt + 1) / 2
    let term_2500 := if seq_position n < 2500 then n + 1 else n
    (term_2500 % 7) = 1 := by 
    sorry

end find_remainder_2500th_term_l752_752545


namespace shaded_region_area_l752_752010

variables (a b : ℕ) 
variable (A : Type) 

def AD := 5
def CD := 2
def semi_major_axis := 6
def semi_minor_axis := 4

noncomputable def area_ellipse := Real.pi * semi_major_axis * semi_minor_axis
noncomputable def area_rectangle := AD * CD
noncomputable def area_shaded_region := area_ellipse - area_rectangle

theorem shaded_region_area : area_shaded_region = 24 * Real.pi - 10 :=
by {
  sorry
}

end shaded_region_area_l752_752010


namespace exists_way_to_E_from_S_l752_752479

structure Graph (α : Type) :=
(vertices : set α)
(edges : set (α × α))

variables {α : Type} (G : Graph α) (S E : α)

-- Condition 1: The graph is connected
def is_connected (G : Graph α) : Prop :=
∀ u v ∈ G.vertices, ∃ p : list α, ∀ i ∈ p, i ∈ G.vertices ∧ (u, v) ∈ (list.tails p).zip (list.tails p).tail

-- Condition 2: Room S has exactly one door (edge)
def one_door (G : Graph α) (S : α) : Prop :=
∃ unique e : (α × α), e ∈ G.edges ∧ (S = e.fst ∨ S = e.snd)

-- Condition 3: Room E is another node in the graph
def room_in_graph (G : Graph α) (E : α) : Prop := E ∈ G.vertices

-- Define the "way" P as an infinite sequence of L and R instructions
def way (P : ℕ → bool) : Prop := true

-- Main statement
theorem exists_way_to_E_from_S (G : Graph α) (S E : α)
  (h1 : is_connected G)
  (h2 : one_door G S)
  (h3 : room_in_graph G E) :
  ∃ P : ℕ → bool, way P ∧ (∃ p : list α, p.head = S ∧ p.tail = list.init p.tail ∧ p.tail.last = E) :=
sorry

end exists_way_to_E_from_S_l752_752479


namespace math_problem_l752_752414

noncomputable def U : ℝ := -2.5
noncomputable def V : ℝ := 3.4
noncomputable def W : ℝ := -1.0
noncomputable def X : ℝ := 0.05
noncomputable def Y : ℝ := 2.1

theorem math_problem :
  U < 0 ∧ W < 0 ∧ V > 0 ∧ Y > 0 ∧ X ≈ 0.05 →
  (U - V < 0) ∧
  (U * V < 0) ∧
  ((X / V) * W < 0) ∧
  ((X + Y) / W < 0) :=
by
  sorry

end math_problem_l752_752414


namespace find_a_l752_752036

open Real

theorem find_a :
  ∃ a : ℝ, (1/5) * (0.5 + a + 1 + 1.4 + 1.5) = 0.28 * 3 + 0.16 := by
  use 0.6
  sorry

end find_a_l752_752036


namespace tan_theta_parallel_vectors_l752_752638

variables {θ : ℝ}

-- Definitions for the conditions
def a : ℝ × ℝ := (2, Real.sin θ)
def b : ℝ × ℝ := (1, Real.cos θ)
def are_parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

-- The main theorem statement
theorem tan_theta_parallel_vectors :
  are_parallel a b → Real.tan θ = 2 :=
sorry

end tan_theta_parallel_vectors_l752_752638


namespace polynomial_divisibility_l752_752737

theorem polynomial_divisibility (n : ℕ) (α : ℝ) (hn : n ≠ 1) (hα : sin α ≠ 0) :
  ∃ R(x) : polynomial ℝ, (x^n * sin α - x * sin (n * α) + sin ((n - 1) * α)) = (x^2 - 2 * x * cos α + 1) * R(x) :=
by
  sorry

end polynomial_divisibility_l752_752737


namespace atomic_weight_of_nitrogen_l752_752056

-- Definitions from conditions
def molecular_weight := 53.0
def hydrogen_weight := 1.008
def chlorine_weight := 35.45
def hydrogen_atoms := 4
def chlorine_atoms := 1

-- The proof goal
theorem atomic_weight_of_nitrogen : 
  53.0 - (4.0 * 1.008) - 35.45 = 13.518 :=
by
  sorry

end atomic_weight_of_nitrogen_l752_752056


namespace half_plus_six_of_ten_eq_eleven_l752_752279

theorem half_plus_six_of_ten_eq_eleven :
  ∃ (n : ℕ), n = 10 ∧ (n / 2) + 6 = 11 :=
by
  use 10
  split
  · refl
  · sorry

end half_plus_six_of_ten_eq_eleven_l752_752279


namespace total_employees_l752_752837

theorem total_employees (x : Nat) (h1 : x < 13) : 13 + 6 * x = 85 :=
by
  sorry

end total_employees_l752_752837


namespace inversion_preserves_angle_l752_752738

-- Define the main properties and assumptions used
variable {P O : Point} -- Points P (intersection/tangency) and O (center of inversion)
variable {C₁ C₂ : Circle} -- Circles
variable {l₁ l₂ : Line} -- Lines

-- Necessary assumptions
axiom tangent_preservation (C : Circle) (l : Line) (P : Point) :
    tangent_at C l P → tangent_at (invert_circle C O) (invert_line l O) (invert_point P O)

-- The main theorem
theorem inversion_preserves_angle {P O : Point} (C₁ C₂ : Circle) (l₁ l₂ : Line):
  angle_between C₁ C₂ P = angle_between (invert_circle C₁ O) (invert_circle C₂ O) (invert_point P O) ∧
  angle_between C₁ l₁ P = angle_between (invert_circle C₁ O) (invert_line l₁ O) (invert_point P O) ∧
  angle_between l₁ l₂ P = angle_between (invert_line l₁ O) (invert_line l₂ O) (invert_point P O) :=
by
  sorry


end inversion_preserves_angle_l752_752738


namespace num_combinations_of_4_choose_2_l752_752755

theorem num_combinations_of_4_choose_2 : 
  nat.choose 4 2 = 6 :=
by
  sorry

end num_combinations_of_4_choose_2_l752_752755


namespace possible_value_of_phi_l752_752768

theorem possible_value_of_phi :
  ∃ φ > 0, (∀ x, cos (2 * x + 2 * φ + π / 6) = cos (2 * (π / 6 - x) + 2 * φ + π / 6)) ↔ φ = π / 4 := by
  sorry

end possible_value_of_phi_l752_752768


namespace prob_two_absent_one_present_l752_752295

theorem prob_two_absent_one_present :
  let P_A := 1 / 20
  let P_P := 19 / 20
  (P_A * P_A * P_P + P_A * P_P * P_A + P_P * P_A * P_A) = 7125 / 1000000 :=
by
  let P_A := 1 / 20
  let P_P := 19 / 20
  have h : (P_A * P_A * P_P + P_A * P_P * P_A + P_P * P_A * P_A) = (0.002375 + 0.002375 + 0.002375) := by sorry
  have h_eq : (0.002375 + 0.002375 + 0.002375) = 0.007125 := by sorry
  have frac_eq : 0.007125 = 7125 / 1000000 := by sorry
  rw [h, h_eq, frac_eq]
  sorry

end prob_two_absent_one_present_l752_752295


namespace inequality_proof_l752_752656

noncomputable def a := (1.01: ℝ) ^ (0.5: ℝ)
noncomputable def b := (1.01: ℝ) ^ (0.6: ℝ)
noncomputable def c := (0.6: ℝ) ^ (0.5: ℝ)

theorem inequality_proof : b > a ∧ a > c := 
by
  sorry

end inequality_proof_l752_752656


namespace number_of_valid_four_digit_integers_l752_752999

-- Definition that the set of allowed digits is {2, 5, 7}
def allowed_digits := {2, 5, 7}

-- Definition of a four-digit positive integer using digits from the allowed set
def valid_four_digit_integers : Nat := 
  let choices_per_digit := 3 -- 2, 5, or 7
  let digit_count := 4
  choices_per_digit ^ digit_count

-- The theorem stating the number of such integers is 81
theorem number_of_valid_four_digit_integers : valid_four_digit_integers = 81 := by
  sorry

end number_of_valid_four_digit_integers_l752_752999


namespace cheese_remaining_after_10_customers_l752_752472

theorem cheese_remaining_after_10_customers :
  (let s_k := λ k : ℕ, 20 / (k + 10) in
   let total_sold := ∑ i in (finset.range 10), s_k i in
   let remaining_cheese := 20 - total_sold in
   remaining_cheese = 10) :=
by
  have s_k_nonneg : ∀ k : ℕ, k < 10 → s_k k ≥ 0 := 
    λ k hk, by
      dsimp [s_k]
      apply div_nonneg; norm_num; linarith
  sorry  -- Proof goes here

end cheese_remaining_after_10_customers_l752_752472


namespace inequality_of_trig_function_l752_752916

theorem inequality_of_trig_function 
  (a b A B : ℝ) 
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end inequality_of_trig_function_l752_752916


namespace quadratic_coefficient_not_zero_l752_752217

theorem quadratic_coefficient_not_zero (m : ℝ) (x : ℝ) :
  (m-2) * x^2 + 5 * x + m^2 - 2 * m = 0 → m ≠ 2 :=
by
  intro h
  have h₁ : m - 2 ≠ 0, from sorry
  contradiction

end quadratic_coefficient_not_zero_l752_752217


namespace avg_speed_second_part_l752_752488

theorem avg_speed_second_part (v : ℝ) (h : 0 < v) :
  (7 / 10 + 10 / v = 17 / 7.99) → v ≈ 7 :=
by
  sorry

end avg_speed_second_part_l752_752488


namespace data_median_is_neg_half_l752_752600

noncomputable def median_of_data (data : List ℚ) : ℚ :=
  let sorted_data := data.sort
  if sorted_data.length % 2 = 1 then
    sorted_data.get (sorted_data.length / 2)
  else
    (sorted_data.get (sorted_data.length / 2 - 1) + sorted_data.get (sorted_data.length / 2)) / 2

theorem data_median_is_neg_half :
  ∀ x : ℚ, (1 * ((-3 - 3 + 4 - 3 + x + 2) / 6) = 1) → median_of_data [-3, -3, 4, -3, x, 2] = -0.5 :=
by
  intros x h_avg
  sorry

end data_median_is_neg_half_l752_752600


namespace parallel_lines_l752_752597

variables {A B C P A' B' C' : Type*}

-- Assume these points are given and that the necessary constructions are made:
variable [circumcircle : has_circumcircle (triangle A B C)] -- P is on the circumcircle of ΔABC
variable [perpendicular1 : is_perpendicular P A' (line B C)] -- PA' ⊥ BC
variable [perpendicular2 : is_perpendicular P B' (line A C)] -- PB' ⊥ AC
variable [perpendicular3 : is_perpendicular P C' (line A B)] -- PC' ⊥ AB

theorem parallel_lines :
  all_parallel (AA' : line A A') (BB' : line B B') (CC' : line C C') := 
sorry  -- Proof goes here

end parallel_lines_l752_752597


namespace max_value_of_f_l752_752572

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (1 + sin x) + sqrt (1 - sin x) +
  sqrt (2 + sin x) + sqrt (2 - sin x) +
  sqrt (3 + sin x) + sqrt (3 - sin x)

theorem max_value_of_f : ∃ (x : ℝ), f(x) = 2 + 2 * sqrt 2 + 2 * sqrt 3 :=
by
  sorry

end max_value_of_f_l752_752572


namespace subset_condition_l752_752364

variable (a : ℝ)

def A : Set ℝ := {0, -a}
def B : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_condition : A a ⊆ B a ↔ a = 1 := by 
  sorry

end subset_condition_l752_752364


namespace num_carnations_l752_752785

-- Define the conditions
def num_roses : ℕ := 5
def total_flowers : ℕ := 10

-- Define the statement we want to prove
theorem num_carnations : total_flowers - num_roses = 5 :=
by {
  -- The proof itself is not required, so we use 'sorry' to indicate incomplete proof
  sorry
}

end num_carnations_l752_752785


namespace David_total_swim_time_l752_752915

theorem David_total_swim_time :
  let t_freestyle := 48
  let t_backstroke := t_freestyle + 4
  let t_butterfly := t_backstroke + 3
  let t_breaststroke := t_butterfly + 2
  t_freestyle + t_backstroke + t_butterfly + t_breaststroke = 212 :=
by
  sorry

end David_total_swim_time_l752_752915


namespace percentage_paid_to_A_l752_752445

theorem percentage_paid_to_A (A B : ℝ) (h1 : A + B = 550) (h2 : B = 220) : (A / B) * 100 = 150 := by
  -- Proof omitted
  sorry

end percentage_paid_to_A_l752_752445


namespace depth_of_well_l752_752505

theorem depth_of_well 
  (t1 t2 : ℝ) 
  (d : ℝ) 
  (h1: t1 + t2 = 8) 
  (h2: d = 32 * t1^2) 
  (h3: t2 = d / 1100) 
  : d = 1348 := 
  sorry

end depth_of_well_l752_752505


namespace rowing_upstream_speed_l752_752496

theorem rowing_upstream_speed (V_m V_down V_up V_s : ℝ) 
  (hVm : V_m = 40) 
  (hVdown : V_down = 60) 
  (hVdown_eq : V_down = V_m + V_s) 
  (hVup_eq : V_up = V_m - V_s) : 
  V_up = 20 := 
by
  sorry

end rowing_upstream_speed_l752_752496


namespace function_properties_l752_752041

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

noncomputable def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x : ℝ, f (x + T) = f (x)

def y (x : ℝ) : ℝ := cos (2 * (x + π / 4))

theorem function_properties :
  is_odd_function y ∧ period y π :=
by
  sorry

end function_properties_l752_752041


namespace trig_identity_l752_752892

theorem trig_identity :
  (1 / real.cos (70 * real.pi / 180) - 2 / real.sin (70 * real.pi / 180)) = (2 * (real.sin (50 * real.pi / 180) - 1) / real.sin (40 * real.pi / 180)) :=
  sorry

end trig_identity_l752_752892


namespace tan_theta_eq_neg_sqrt_l752_752701

noncomputable def tan_theta (x : ℝ) (θ : ℝ) : ℝ :=
  if h : 0 < θ ∧ θ < π / 2 ∧ cos (θ / 2) = sqrt ((x - 1) / (2 * x)) then
    -sqrt (x^2 - 1)
  else
    0

theorem tan_theta_eq_neg_sqrt (x θ : ℝ) (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : cos (θ / 2) = sqrt ((x - 1) / (2 * x))) :
  tan θ = tan_theta x θ := by
  sorry

end tan_theta_eq_neg_sqrt_l752_752701


namespace tan_cos_identity_l752_752539

theorem tan_cos_identity :
  (∃ θ : ℝ, θ = 40 ∧ 
    (tan θ ^ 2 - cos θ ^ 2) / (tan θ ^ 2 * cos θ ^ 2) = 1 / (cos θ ^ 2) + 1) :=
begin
  use 40.0,
  split,
  { refl },
  {
    sorry -- Proof goes here
  }
end

end tan_cos_identity_l752_752539


namespace problem_1_problem_2_l752_752601

noncomputable def arith_geo_seq_general_term (a : ℕ → ℤ) : Prop :=
∀ (n : ℕ), n > 0 → a n = 2 * n - 1

noncomputable def seq_sum (a b : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
∀ (n : ℕ), n > 0 → S n = n^2 + 2^(n+1) - 2

theorem problem_1 (a : ℕ → ℤ) (h1 : a 1 = 1)
                  (h2 : ∃ d, d ≠ 0 ∧ ∀ (n : ℕ), a (n+1) = a n + d)
                  (h3 : (a 2)^2 = a 1 * a 5) :
  arith_geo_seq_general_term a := sorry

theorem problem_2 (a b : ℕ → ℤ) (S : ℕ → ℤ)
                  (h4 : b = (λ n, 2^n))
                  (h5 : arith_geo_seq_general_term a) :
  seq_sum a b S := sorry

end problem_1_problem_2_l752_752601


namespace determine_m_l752_752263

def hyperbola_eccentricity_condition (m : ℝ) (h : m > 0) : Prop :=
  let a := 2 in
  let e := Real.sqrt 3 in
  e = Real.sqrt (a^2 + m^2) / a

theorem determine_m (m : ℝ) (h : m > 0) :
  hyperbola_eccentricity_condition m h → m = 2 * Real.sqrt 2 :=
by { sorry }

end determine_m_l752_752263


namespace cubes_sum_l752_752116

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l752_752116


namespace complex_problem_find_a_b_l752_752642

theorem complex_problem (i : ℂ) (h_imag : i * i = -1) :
  let z := (1 + i) * (1 + i) + (2 * i) / (1 + i) in
  z = 1 + 3 * i ∧ abs z = real.sqrt 10 :=
by {
  let z := (1 + i) * (1 + i) + (2 * i) / (1 + i),
  have : z = 1 + 3 * i,
  { sorry },
  have : abs z = real.sqrt 10,
  { sorry },
  exact ⟨this, this_1⟩
}

theorem find_a_b (i : ℂ) (h_imag : i * i = -1) :
  let z := 1 + 3 * i in
  ∃ a b : ℝ, (z * z + a * conj z + b = 2 + 3 * i) ∧ a = 1 ∧ b = 9 :=
by {
  let z := 1 + 3 * i,
  have : ∃ a b : ℝ, (z * z + a * conj z + b = 2 + 3 * i) ∧ a = 1 ∧ b = 9,
  { sorry },
  exact this
}

end complex_problem_find_a_b_l752_752642


namespace g_10_5_l752_752619

noncomputable def f (x : ℝ) : ℝ := 
if 0 ≤ x ∧ x ≤ 2 then g (x + 2) else -g (-x + 2)

noncomputable def g (x : ℝ) : ℝ := x - 2

theorem g_10_5 : g 10.5 = 0.5 := by
  sorry

end g_10_5_l752_752619


namespace proof_correct_option_C_l752_752615

def line := Type
def plane := Type
def perp (m : line) (α : plane) : Prop := sorry
def parallel (n : line) (α : plane) : Prop := sorry
def perpnal (m n: line): Prop := sorry 

variables (m n : line) (α β γ : plane)

theorem proof_correct_option_C : perp m α → parallel n α → perpnal m n := sorry

end proof_correct_option_C_l752_752615


namespace total_money_l752_752846

theorem total_money (gold_value silver_value cash : ℕ) (num_gold num_silver : ℕ)
  (h_gold : gold_value = 50)
  (h_silver : silver_value = 25)
  (h_num_gold : num_gold = 3)
  (h_num_silver : num_silver = 5)
  (h_cash : cash = 30) :
  gold_value * num_gold + silver_value * num_silver + cash = 305 :=
by
  sorry

end total_money_l752_752846


namespace f_value_at_3_l752_752040

noncomputable def f : ℝ → ℝ := sorry

theorem f_value_at_3 (h : ∀ x : ℝ, f x + 2 * f (1 - x) = 4 * x^2) : f 3 = -4 / 3 :=
by
  sorry

end f_value_at_3_l752_752040


namespace valid_operation_l752_752802

variable {a b : ℝ}

theorem valid_operation :
  (a^2 + b^3 ≠ 2 * a^5) ∧ 
  ((-a^2)^3 = -a^6) ∧ 
  (a^2 * a^4 ≠ a^8) ∧ 
  (a^4 / a ≠ a^4) :=
by
  sorry

end valid_operation_l752_752802


namespace probability_good_graph_l752_752493

def is_good_graph (G : simple_graph (fin 6)) : Prop :=
  ∀ v : fin 6, (∃ w, (v ≠ w ∧ G.adj v w)) →
  ∃ p : G.walk v v, G.is_eulerian_circuit p

def count_good_graphs : ℕ :=
  ((finset.univ : finset (simple_graph (fin 6))).filter (λ G, is_good_graph G)).card

theorem probability_good_graph : 
  (count_good_graphs : ℚ) / ((2 : ℕ) ^ 15) = 507 / 16384 :=
sorry

end probability_good_graph_l752_752493


namespace total_votes_cast_l752_752682

def votes_witch : ℕ := 7
def votes_unicorn : ℕ := 3 * votes_witch
def votes_dragon : ℕ := votes_witch + 25
def votes_total : ℕ := votes_witch + votes_unicorn + votes_dragon

theorem total_votes_cast : votes_total = 60 := by
  sorry

end total_votes_cast_l752_752682


namespace transform_any_initial_arrangement_l752_752329

variables (m n N : ℕ) (Z : Type) [fintype Z] [add_group Z]

def coprime (a b : ℕ) : Prop := ∀ x : ℕ, x ≠ 0 → x ∣ a → x ∣ b → x = 1

def move_possible (m n N : ℕ) [fact (N > 0)]
  (Z : Type) [fintype Z] [add_group Z] :=
  ∀ (A B : matrix (fin m) (fin n) Z),
  let move := λ g (ci : fin m × fin n) (M : matrix (fin m) (fin n) Z),
    M.update_row ci.1 (λ j, (M ci.1 j + g) %% N)
    .update_col ci.2 (λ i, (M i ci.2 + g) %% N) in
  (∀ g ci, is_valid g ci → move g ci A = B) →
    ∃ k : ℕ, ∃ L : vector (ℕ × fin m × fin n) k,
      (finset.range k).foldl (λ M ⟨g, r, c⟩, move g (r, c) M) A = B

theorem transform_any_initial_arrangement
  (hc1: coprime N (m - 1))
  (hc2: coprime N (n - 1))
  (hc3: coprime N (m + n - 1)) :
  move_possible m n N ℤ := 
sorry

end transform_any_initial_arrangement_l752_752329


namespace largest_m_for_game_with_2022_grids_l752_752432

variables (n : ℕ) (f : ℕ → ℕ)

/- Definitions using conditions given -/

/-- Definition of the game and the marking process -/
def game (n : ℕ) : ℕ := 
  if n % 4 = 0 then n / 2 + 1
  else if n % 4 = 2 then n / 2 + 1
  else 0

/-- Main theorem statement -/
theorem largest_m_for_game_with_2022_grids : game 2022 = 1011 :=
by sorry

end largest_m_for_game_with_2022_grids_l752_752432


namespace solution_l752_752345

noncomputable def problem_statement : Prop :=
  let O := (0, 0, 0)
  let A := (α : ℝ, 0, 0)
  let B := (0, β : ℝ, 0)
  let C := (0, 0, γ : ℝ)
  let plane_equation := λ x y z, (x / α + y / β + z / γ = 1)
  let distance_from_origin := 2
  let centroid := ((α / 3), (β / 3), (γ / 3))
  let p := centroid.1
  let q := centroid.2
  let r := centroid.3
  (1 / p^2) + (1 / q^2) + (1 / r^2) = 2.25

theorem solution : problem_statement := 
by sorry

end solution_l752_752345


namespace sequence_23rd_term_is_45_l752_752692

def sequence_game (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * n - 1 else 2 * n + 1

theorem sequence_23rd_term_is_45 :
  sequence_game 23 = 45 :=
by
  -- Proving the 23rd term in the sequence as given by the game rules
  sorry

end sequence_23rd_term_is_45_l752_752692


namespace points_for_win_l752_752680

theorem points_for_win :
  ∃ (W : ℕ), (∀ (n : ℕ), (n = 6) → 
  let total_games := (n * (n - 1)) / 2 in 
  let max_points := 30 * W in 
  let min_points := total_games in 
  max_points - min_points = 15 → W = 1) :=
sorry

end points_for_win_l752_752680


namespace anya_needs_to_eat_minimal_squares_l752_752471

theorem anya_needs_to_eat_minimal_squares :
  ∃ (min_squares_to_eat : ℕ), 
  min_squares_to_eat = 5 ∧ 
  ∀ (bar : matrix (fin 5) (fin 6) ℕ), 
  ∃ (nut_rect : fin 2 × fin 3), 
  (∀ (ate_squares : fin 30), ... -- condition detailing which squares are eaten
    ∃ (i : fin 5), ∃ (j : fin 6), 
    bar i j = nut_rect.1 ∧ bar i j = nut_rect.2
  ) sorry

end anya_needs_to_eat_minimal_squares_l752_752471


namespace min_value_at_zero_max_value_a_l752_752353

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - (a * x / (x + 1))

-- Part (I)
theorem min_value_at_zero {a : ℝ} (h : ∀ x, f x a ≥ f 0 a) : a = 1 :=
sorry

-- Part (II)
theorem max_value_a (h : ∀ x > 0, f x a > 0) : a ≤ 1 :=
sorry

end min_value_at_zero_max_value_a_l752_752353


namespace relationship_between_f_b_minus_2_and_f_a_plus_1_l752_752881

noncomputable def f (a : ℝ) (x : ℝ) := Real.log a (abs x)
def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_monotonic_on_interval (f : ℝ → ℝ) (interval : Set ℝ) := 
  (∀ x y ∈ interval, x < y → f x ≤ f y) ∨ (∀ x y ∈ interval, x < y → f y ≤ f x)

theorem relationship_between_f_b_minus_2_and_f_a_plus_1
  (a : ℝ) (b : ℝ)
  (h1 : b = 0)
  (h2 : f a (b-2) < f a (a+1))
  (h3 : is_even_function (f a))
  (h4 : is_monotonic_on_interval (f a) (Set.Ioi 0)) : 
  f a (b-2) < f a (a+1) := sorry

end relationship_between_f_b_minus_2_and_f_a_plus_1_l752_752881


namespace machine_X_produces_in_18_days_l752_752741

variables (W X Y Z : ℝ)

-- Defining the given conditions
def condition1 := W / X = W / Y + 2
def condition2 := 3 * (X + Y) = 5 * W / 4
def condition3 := Z = 18 * X

-- The statement to prove
theorem machine_X_produces_in_18_days (h1 : condition1) (h2 : condition2) (h3 : condition3) :
  Z = (15 * W^2 / 4) / (W - 1) :=
sorry

end machine_X_produces_in_18_days_l752_752741


namespace correct_answer_l752_752978

-- Define planes α and β with a dihedral angle of 50 degrees.
variable (α β : Plane) (h_dihedral : Plane.dihedralAngle α β = 50)

-- Define skew lines b and c.
variable (b c : Line)

-- Conditions for option A
def optionA : Prop := (b ∥ α) ∧ (c ∥ β)

-- Conditions for option B
def optionB : Prop := (b ∥ α) ∧ (c ⊥ β)

-- Conditions for option C
def optionC : Prop := (b ⊥ α) ∧ (c ⊥ β)

-- Conditions for option D
def optionD : Prop := (b ⊥ α) ∧ (c ∥ β)

-- The theorem stating that if b ⊥ α and c ⊥ β, then the angle between b and c is 50 degrees.
theorem correct_answer : optionC α β b c h_dihedral → Line.angle b c = 50 := sorry

end correct_answer_l752_752978


namespace number_of_distinct_tables_l752_752130

def table := matrix (fin 7) (fin 7) ℕ 
def original_table : table := 
  ![[1, 2, 3, 4, 5, 6, 7], 
    [7, 1, 2, 3, 4, 5, 6], 
    [6, 7, 1, 2, 3, 4, 5], 
    [5, 6, 7, 1, 2, 3, 4], 
    [4, 5, 6, 7, 1, 2, 3], 
    [3, 4, 5, 6, 7, 1, 2], 
    [2, 3, 4, 5, 6, 7, 1]]

theorem number_of_distinct_tables :
  (finset.univ.permutes original_table).card = factorial 7 * factorial 7 :=
sorry

end number_of_distinct_tables_l752_752130


namespace problem_statement_l752_752350

noncomputable def geom_seq (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q^(n - 1)

noncomputable def geom_sum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a₁ * n else a₁ * (1 - q^n) / (1 - q)

theorem problem_statement (a₁ q : ℝ) (h : geom_seq a₁ q 6 = 8 * geom_seq a₁ q 3) :
  geom_sum a₁ q 6 / geom_sum a₁ q 3 = 9 :=
by
  -- proof goes here
  sorry

end problem_statement_l752_752350


namespace curve_is_circle_l752_752551

theorem curve_is_circle (r θ : ℝ) (x y : ℝ) (h1 : r = 6 * sin θ * csc θ)
    (h2 : r = real.sqrt (x^2 + y^2)) : x^2 + y^2 = 36 :=
by
  sorry

end curve_is_circle_l752_752551


namespace find_pairs_l752_752201

namespace SolutionProof

open Classical

def is_solution_pair (x y : ℕ) : Prop :=
  (1 / (x : ℝ) + 1 / (y : ℝ) + 1 / (Real.ofRat (Nat.lcm x y)) + 1 / (Real.ofRat (Nat.gcd x y)) = 1 / 2)

theorem find_pairs : 
  {p | is_solution_pair p.1 p.2} = {(5, 20), (6, 12), (8, 8), (8, 12), (9, 24), (12, 15)} := 
  sorry

end SolutionProof

end find_pairs_l752_752201


namespace tangent_parallel_x_axis_decreasing_function_l752_752983

/-- Definition of the function f(x) = 2ln(x) + a/x^2 --/
def f (x a : ℝ) : ℝ := 2 * Real.log x + a / x^2

/-- Definition of the derivative of f --/
def f_prime (x a : ℝ) : ℝ := (2 / x) - (2 * a / x^3)

/-- Problem 1: Proving the value of 'a' such that the tangent at (1, f(1)) is parallel to the x-axis --/
theorem tangent_parallel_x_axis (a : ℝ) : f_prime 1 a = 0 ↔ a = 1 := by
  sorry

/-- Problem 2: Proving the range of 'a' for which f(x) is decreasing on [1, 3] --/
theorem decreasing_function (a : ℝ) : (∀ x ∈ Set.Icc (1 : ℝ) (3 : ℝ), f_prime x a ≤ 0) ↔ a ∈ Set.Ici (9 : ℝ) := by
  sorry

end tangent_parallel_x_axis_decreasing_function_l752_752983


namespace no_positive_sequence_exists_l752_752921

theorem no_positive_sequence_exists:
  ¬ (∃ (b : ℕ → ℝ), (∀ n, b n > 0) ∧ (∀ m : ℕ, (∑' k, b ((k + 1) * m)) = (1 / m))) :=
by
  sorry

end no_positive_sequence_exists_l752_752921


namespace fg_plus_gf_at_2_l752_752354

noncomputable def f (x : ℝ) : ℝ := (4 * x^3 + 2 * x^2 + x + 10) / (2 * x^2 - 3 * x + 5)
noncomputable def g (x : ℝ) : ℝ := 2 * x - 2

theorem fg_plus_gf_at_2 :
  let x := 2 in (f (g x)) + (g (f x)) = 142 / 7 := sorry

end fg_plus_gf_at_2_l752_752354


namespace equal_segments_in_ac_contact_l752_752681

-- Definition of a point in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Definition of a triangle in the plane
structure Triangle :=
  (A B C : Point)

-- Definition of the midpoint of a segment
def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Definition of an acute-angled triangle
def is_acute (Δ : Triangle) : Prop := sorry

-- Given conditions
variables (A B C A0 B0 C0 C1 A1 B1 : Point)
variables (Δ : Triangle)

-- Definitions and conditions
def h_A0 := sorry -- Definition for A0 as the foot of altitude from A.
def h_B0 := sorry -- Definition for B0 as the foot of altitude from B.
def h_C0 := sorry -- Definition for C0 as the foot of altitude from C.
def C1 := midpoint A0 B0 -- Midpoint of segment A0 B0.
def A1 := midpoint B0 C0 -- Midpoint of segment B0 C0.
def B1 := midpoint C0 A0 -- Midpoint of segment C0 A0.

-- Theorem
theorem equal_segments_in_ac_contact (h_acute : is_acute Δ) : 
  Δ.A = A ∧ Δ.B = B ∧ Δ.C = C ∧ 
  A1 = midpoint B0 C0 ∧ B1 = midpoint C0 A0 ∧ C1 = midpoint A0 B0 →
  distance A1 B1 = distance B1 C1 ∧ distance B1 C1 = distance C1 A1 :=
sorry

end equal_segments_in_ac_contact_l752_752681


namespace order_of_scores_l752_752193

variables (E L T N : ℝ)

-- Conditions
axiom Lana_condition_1 : L ≠ T
axiom Lana_condition_2 : L ≠ N
axiom Lana_condition_3 : T ≠ N

axiom Tom_condition : ∃ L' E', L' ≠ T ∧ E' > L' ∧ E' ≠ T ∧ E' ≠ L' 

axiom Nina_condition : N < E

-- Proof statement
theorem order_of_scores :
  N < L ∧ L < T :=
sorry

end order_of_scores_l752_752193


namespace morgan_change_l752_752726

variable hamburger_cost : ℕ := 4
variable onion_rings_cost : ℕ := 2
variable smoothie_cost : ℕ := 3
variable bill : ℕ := 20

theorem morgan_change : hamburger_cost + onion_rings_cost + smoothie_cost = 9 → bill - 9 = 11 :=
by sorry

end morgan_change_l752_752726


namespace club_truncator_more_wins_than_losses_l752_752174

noncomputable def probability_of_more_wins_than_losses : ℚ := 299 / 729

theorem club_truncator_more_wins_than_losses :
  let m := 299
  let n := 729
  m + n = 1028 ∧
  ((let outcomes := 2187 in
    let equal_wins_losses := 393 in
    let prob_equal_wins_losses := equal_wins_losses / outcomes in
    let prob_not_equal_wins_losses := 1 - prob_equal_wins_losses in
    let prob_more_wins_than_losses := prob_not_equal_wins_losses / 2 in
    prob_more_wins_than_losses = 299 / 729)) := 
  begin
    split,
    { refl },
    { sorry }
  end

end club_truncator_more_wins_than_losses_l752_752174


namespace total_money_is_305_l752_752849

-- Define the worth of each gold coin, silver coin, and the quantity of each type of coin and cash.
def worth_of_gold_coin := 50
def worth_of_silver_coin := 25
def number_of_gold_coins := 3
def number_of_silver_coins := 5
def cash_in_dollars := 30

-- Define the total money calculation based on given conditions.
def total_gold_value := number_of_gold_coins * worth_of_gold_coin
def total_silver_value := number_of_silver_coins * worth_of_silver_coin
def total_value := total_gold_value + total_silver_value + cash_in_dollars

-- The proof statement asserting the total value.
theorem total_money_is_305 : total_value = 305 := by
  -- Proof omitted for brevity.
  sorry

end total_money_is_305_l752_752849


namespace find_digits_l752_752002

noncomputable def A : ℕ := 3
noncomputable def B : ℕ := 7
noncomputable def C : ℕ := 1

def row_sums_equal (A B C : ℕ) : Prop :=
  (4 + 2 + 3 = 9) ∧
  (A + 6 + 1 = 9) ∧
  (1 + 2 + 6 = 9) ∧
  (B + 2 + C = 9)

def column_sums_equal (A B C : ℕ) : Prop :=
  (4 + 1 + B = 12) ∧
  (2 + A + 2 = 6 + 3 + 1) ∧
  (3 + 1 + 6 + C = 12)

def red_cells_sum_equals_row_sum (A B C : ℕ) : Prop :=
  (A + B + C = 9)

theorem find_digits :
  ∃ (A B C : ℕ),
    row_sums_equal A B C ∧
    column_sums_equal A B C ∧
    red_cells_sum_equals_row_sum A B C ∧
    100 * A + 10 * B + C = 371 :=
by
  exact ⟨3, 7, 1, ⟨rfl, rfl, rfl, rfl⟩⟩

end find_digits_l752_752002


namespace trigonometric_identity_l752_752898

theorem trigonometric_identity
  (h1 : cos (70 * (Real.pi / 180)) ≠ 0)
  (h2 : sin (70 * (Real.pi / 180)) ≠ 0) :
  (1 / cos (70 * (Real.pi / 180)) - 2 / sin (70 * (Real.pi / 180))) = 
  (4 * sin (10 * (Real.pi / 180))) / sin (40 * (Real.pi / 180)) :=
  sorry

end trigonometric_identity_l752_752898


namespace total_amount_after_7_years_l752_752155

theorem total_amount_after_7_years (P A1: ℕ) (hP: P = 500) (hA1: A1 = 590) : 
    let r := (A1 - P) / (P * 2) in
    let t := 2 + 5 in
    let A2 := P + P * r * t in
    A2 = 815 := 
by
  -- Proof goes here
  sorry

end total_amount_after_7_years_l752_752155


namespace dot_product_of_perpendicular_and_magnitude_l752_752976

variables {V : Type*} [inner_product_space ℝ V]
variables (OA OB : V)

theorem dot_product_of_perpendicular_and_magnitude (h1 : inner_product_space.is_perp OA (OB - OA))
(h2 : ∥OA∥ = 3) : inner_product_space.ℝ_dot OA OB = 9 :=
by sorry

end dot_product_of_perpendicular_and_magnitude_l752_752976


namespace part1_part2_l752_752534

-- Definition for part 1
def expr1 (a b : ℝ) : ℝ := (a^(1/3) * b^(1/2)) * (-3 * a^(1/2) * b^(1/3)) / (1/3 * a^(1/6) * b^(5/6))

-- Theorem statement for part 1
theorem part1 (a b : ℝ) : expr1 a b = -9 * a^(2/3) := 
by sorry

-- Definition for part 2
def expr2 : ℝ := (0.064)^(-1/3) - (-7/8)^0 + (81/16)^(1/4) + |-0.01|^(1/2)

-- Theorem statement for part 2
theorem part2 : expr2 = 2.5 :=
by sorry

end part1_part2_l752_752534


namespace solve_trig_eq_l752_752399

noncomputable def rad (d : ℝ) := d * (Real.pi / 180)

theorem solve_trig_eq (z : ℝ) (k : ℤ) :
  (7 * Real.cos (z) ^ 3 - 6 * Real.cos (z) = 3 * Real.cos (3 * z)) ↔
  (z = rad 90 + k * rad 180 ∨
   z = rad 39.2333 + k * rad 180 ∨
   z = rad 140.7667 + k * rad 180) :=
sorry

end solve_trig_eq_l752_752399


namespace solve_sin_cos_eq_l752_752465

noncomputable def sin_cos_eq (t : ℝ) : Prop :=
  sin(2 * t)^6 + cos(2 * t)^6 = (3 / 2) * (sin(2 * t)^4 + cos(2 * t)^4) + (1 / 2) * (sin t + cos t)

theorem solve_sin_cos_eq : 
  ∀ (t : ℝ), (sin_cos_eq t) ↔ (∃ k : ℤ, t = π * (2 * k + 1)) ∨ (∃ n : ℤ, t = π / 2 * (4 * n - 1)) := 
by {
  sorry
}

end solve_sin_cos_eq_l752_752465


namespace area_of_triangle_l752_752667

theorem area_of_triangle {A B C : ℝ} {a b c : ℝ}
  (h1 : b = 2) (h2 : c = 2 * Real.sqrt 2) (h3 : C = Real.pi / 4) :
  1 / 2 * b * c * Real.sin (Real.pi - C - (1 / 2 * Real.pi / 3)) = Real.sqrt 3 + 1 :=
by
  sorry

end area_of_triangle_l752_752667


namespace grocer_purchased_72_pounds_l752_752466

noncomputable def grocer_problem : Bool :=
  let CP_per_pound := 0.50 / 3
  let SP_per_pound := 1.00 / 4
  let Profit_per_pound := SP_per_pound - CP_per_pound
  let Total_profit := 6.00
  let Total_pounds := Total_profit / Profit_per_pound
  Total_pounds = 72

theorem grocer_purchased_72_pounds :
  grocer_problem = True :=
sorry

end grocer_purchased_72_pounds_l752_752466


namespace functions_equal_l752_752553

def f (x : ℝ) : ℝ := exp (x + 1) * exp (x - 1)
def g (x : ℝ) : ℝ := exp (2 * x)

theorem functions_equal : ∀ x : ℝ, f x = g x :=
by
  unfold f g
  sorry

end functions_equal_l752_752553


namespace tan_cos_identity_l752_752540

theorem tan_cos_identity :
  (∃ θ : ℝ, θ = 40 ∧ 
    (tan θ ^ 2 - cos θ ^ 2) / (tan θ ^ 2 * cos θ ^ 2) = 1 / (cos θ ^ 2) + 1) :=
begin
  use 40.0,
  split,
  { refl },
  {
    sorry -- Proof goes here
  }
end

end tan_cos_identity_l752_752540


namespace points_lie_on_single_plane_l752_752594

theorem points_lie_on_single_plane
  (points : Finset Point)
  (H_finite : points.Finite)
  (H_no_three_collinear : ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → ¬Collinear p1 p2 p3)
  (H_plane_contains_other_point : ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → ∃ p4 : Point, p4 ∈ points ∧ PlaneContains p4 (PlaneThrough p1 p2 p3)) :
  ∃ (plane : Plane), ∀ (p : Point), p ∈ points → PlaneContains p plane := 
by 
  sorry

end points_lie_on_single_plane_l752_752594


namespace distance_between_P_and_Q_l752_752348

def quadratic_vertex(a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  (h, a * h^2 + b * h + c)

def distance (P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_P_and_Q :
  let P := quadratic_vertex 1 (-4) 5 in
  let Q := quadratic_vertex 1 6 20 in
  distance P Q = 5 * Real.sqrt 5 :=
by
  sorry

end distance_between_P_and_Q_l752_752348


namespace oil_tank_depth_l752_752855

theorem oil_tank_depth
  (height : ℝ) (diameter : ℝ) (oil_depth_horizontal : ℝ) (volume_oil : ℝ)
  (h_height : height = 20)
  (h_diameter : diameter = 6)
  (h_oil_depth_horizontal : oil_depth_horizontal = 4)
  (h_volume : volume_oil = 62.9) :
  ∃ (depth_upright : ℝ), depth_upright ≈ 2.2 :=
by
  -- We assume accurate calculations based on the provided conditions
  let radius := diameter / 2
  let volume := volume_oil
  let height_upright := volume / (π * radius ^ 2)
  use height_upright
  -- Here we assume the calculations and rounding have been done accurately
  have h_approx : height_upright ≈ 2.2 := sorry
  exact h_approx

end oil_tank_depth_l752_752855


namespace cistern_fill_time_l752_752836

-- Capacity of the cistern
variable (C : ℝ)

-- Rates of filling or emptying the cistern
def rateA := C / 4
def rateB := -C / 7
def rateC := C / 5

-- Combined rate for all taps open simultaneously
def combinedRate := rateA + rateB + rateC

-- Time required to fill the cistern
def timeToFill := C / combinedRate

-- The proof problem: the time to fill the cistern is approximately 3.2558 hours
theorem cistern_fill_time : timeToFill C = 140 / 43 := by
  unfold timeToFill combinedRate rateA rateB rateC
  field_simp
  norm_num
  sorry

end cistern_fill_time_l752_752836


namespace values_of_t_l752_752567

theorem values_of_t (x y z t : ℝ) 
  (h1 : 3 * x^2 + 3 * x * z + z^2 = 1)
  (h2 : 3 * y^2 + 3 * y * z + z^2 = 4)
  (h3 : x^2 - x * y + y^2 = t) : 
  t ≤ 10 :=
sorry

end values_of_t_l752_752567


namespace num_red_balls_l752_752678

theorem num_red_balls (x : ℕ) (h : 4 / (4 + x) = 1 / 5) : x = 16 :=
by
  sorry

end num_red_balls_l752_752678


namespace nebraska_more_plates_than_georgia_l752_752401

theorem nebraska_more_plates_than_georgia :
  (26 ^ 2 * 10 ^ 5) - (26 ^ 4 * 10 ^ 2) = 21902400 :=
by
  sorry

end nebraska_more_plates_than_georgia_l752_752401


namespace lifting_weights_l752_752754

theorem lifting_weights (n : ℕ) : 2 * 25 * 10 = 2 * 18 * n → n = 14 := by
  intro h
  have h1 : 2 * 25 * 10 = 500 := by norm_num
  rw h1 at h
  have h2 : 2 * 18 = 36 := by norm_num
  rw h2 at h
  have h3 : 36 * n = 500 := by exact h
  have h4 : n = 500 / 36 := by linarith
  have h5 : (500 / 36 : ℝ) ≈ 13.89 := by norm_num
  have h6 : 14 = (500 / 36).ceil := by norm_num
  exact h
  sorry

end lifting_weights_l752_752754


namespace intersection_points_of_line_and_conic_l752_752417

theorem intersection_points_of_line_and_conic
  (hL : ∀ x, y = x + 3)
  (hC : ∀ y x, (y^2 / 9) - (x * |x| / 4) = 1) :
  ∃ (count : ℕ), count = 3 :=
by
  sorry

end intersection_points_of_line_and_conic_l752_752417


namespace solution_set_l752_752238

open Real Set

variable (f : ℝ → ℝ)

-- Conditions
def odd_function : Prop := ∀ x, f (-x) = -f x
def increasing_on_positive : Prop := ∀ x y, 0 < x → x < y → f x < f y
def f_neg_half_zero : Prop := f (-1 / 2) = 0

-- Theorem
theorem solution_set (h1 : odd_function f) (h2 : increasing_on_positive f) (h3 : f_neg_half_zero f) :
  { x : ℝ | f (log 4 x) > 0 } = Ioo (1 / 2) 1 ∪ Ioi 2 :=
  sorry

end solution_set_l752_752238


namespace paper_cut_pieces_l752_752015

theorem paper_cut_pieces (k : ℕ) : 
  ∃ k : ℕ, 7 + 6 * k = 331 := 
begin
  use 54,
  sorry
end

end paper_cut_pieces_l752_752015


namespace emily_mean_seventh_score_l752_752194

theorem emily_mean_seventh_score :
  let a1 := 85
  let a2 := 88
  let a3 := 90
  let a4 := 94
  let a5 := 96
  let a6 := 92
  (a1 + a2 + a3 + a4 + a5 + a6 + a7) / 7 = 91 → a7 = 92 :=
by
  intros
  sorry

end emily_mean_seventh_score_l752_752194


namespace students_speaking_both_l752_752850

noncomputable def total_students := 499.99999999999994
noncomputable def percentage_not_speaking_french := 0.86
noncomputable def number_not_speaking_english := 40

noncomputable def number_who_speak_french (total: ℝ) (percentage_not_french: ℝ) : ℝ :=
  total * (1 - percentage_not_french)

noncomputable def number_who_speak_both (total: ℝ) (percentage_not_french: ℝ) (number_not_english: ℝ) : ℝ :=
  number_who_speak_french(total, percentage_not_french) - number_not_english

theorem students_speaking_both :
  number_who_speak_both total_students percentage_not_speaking_french number_not_speaking_english = 30 :=
by
  sorry

end students_speaking_both_l752_752850


namespace hexagon_properties_proof_l752_752065

open EuclideanGeometry

-- Define the problem in Lean context
variables {Ω : Type} [MetricSpace Ω]

theorem hexagon_properties_proof
  (hexagon : ConvexHexagon Ω)
  (circumcircle : Circle Ω)
  (P : Ω)
  (vertices : (A B C A' B' C' : Ω)) 
  (on_circle : ∀ (V ∈ {A, B, C, A', B', C'}), import Circle V circumcircle)
  (diameters : ∀ (z : Ω), z ∈ {AA', BB', CC'} → z ∈ Diam circumcircle) 
  (P_different : ¬(P ∈ {A, B, C, A', B', C'}))
  (Qfeet : ∀ (Q1 Q2 Q3 Q4 Q5 Q6 : Ω), 
            feetPerpendicular P (A * B, A * C, C * A', A' * B', B' * C', C' * A)) : 
  -- Prove the two claims
  (∀ (i : Fin 6), perpendicular (side hexagon (Qfeet!((i + 1) % 6))) (side hexagon (Qfeet (i % 6)))) ∧
  Collinear {midpoint (Q1, Q4), midpoint (Q2, Q5), midpoint (Q3, Q6), P} := 
sorry

end hexagon_properties_proof_l752_752065


namespace trig_identity_l752_752893

theorem trig_identity :
  (1 / real.cos (70 * real.pi / 180) - 2 / real.sin (70 * real.pi / 180)) = (2 * (real.sin (50 * real.pi / 180) - 1) / real.sin (40 * real.pi / 180)) :=
  sorry

end trig_identity_l752_752893


namespace symbol_definitions_l752_752473

theorem symbol_definitions :
  (∀ (x : Type) (s : set x), "Belongs to" is represented by $\in$) ∧
  (∀ (x : Type) (s1 s2 : set x), "Is a subset of" is represented by $\subseteq$) ∧
  ("The empty set" is represented by ∅) ∧
  ("The set of real numbers" is represented by ℝ) := 
begin
  sorry
end

end symbol_definitions_l752_752473


namespace square_perimeter_l752_752764

theorem square_perimeter (s : ℝ) (h : s^2 = s) : 4 * s = 4 :=
by
  sorry

end square_perimeter_l752_752764


namespace average_beef_sales_l752_752378

def monday_sales : ℝ := 198.5
def tuesday_sales : ℝ := 276.2
def wednesday_sales : ℝ := 150.7
def thursday_sales : ℝ := 210.0
def friday_sales : ℝ := 420.0
def saturday_sales : ℝ := 150.0
def sunday_sales : ℝ := 324.6

def total_sales : ℝ := 
  monday_sales + tuesday_sales + wednesday_sales +
  thursday_sales + friday_sales + saturday_sales + sunday_sales

def average_sales_per_day : ℝ := total_sales / 7

theorem average_beef_sales :
  average_sales_per_day = 247.14 := by
  sorry

end average_beef_sales_l752_752378


namespace cost_formula_l752_752037

-- Definitions based on conditions
def base_cost : ℕ := 15
def additional_cost_per_pound : ℕ := 5
def environmental_fee : ℕ := 2

-- Definition of cost function
def cost (P : ℕ) : ℕ := base_cost + additional_cost_per_pound * (P - 1) + environmental_fee

-- Theorem stating the formula for the cost C
theorem cost_formula (P : ℕ) (h : 1 ≤ P) : cost P = 12 + 5 * P :=
by
  -- Proof would go here
  sorry

end cost_formula_l752_752037


namespace quadratic_roots_equal_integral_l752_752909

theorem quadratic_roots_equal_integral (c : ℝ) (h : (6^2 - 4 * 3 * c) = 0) : 
  ∃ x : ℝ, (3 * x^2 - 6 * x + c = 0) ∧ (x = 1) := 
by sorry

end quadratic_roots_equal_integral_l752_752909


namespace line_passes_through_quadrants_l752_752227

variables (a b c p : ℝ)

-- Given conditions
def conditions :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  (a + b) / c = p ∧ 
  (b + c) / a = p ∧ 
  (c + a) / b = p

-- Goal statement
theorem line_passes_through_quadrants : conditions a b c p → 
  (∃ x : ℝ, x > 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p > 0) ∧
  (∃ x : ℝ, x < 0 ∧ px + p < 0) :=
sorry

end line_passes_through_quadrants_l752_752227


namespace icosahedron_red_neighbors_l752_752450

theorem icosahedron_red_neighbors 
    (V : Finset ℕ) 
    (E : Finset (ℕ × ℕ)) 
    (adj : ℕ → ℕ → Prop)
    (red_vertices : Finset ℕ) 
    (h_V : V.card = 12) 
    (h_E : E.card = 30)
    (h_adj : ∀ v ∈ V, (∃ w, (w ≠ v ∧ adj v w))) 
    (h_deg : ∀ v ∈ V, ((Finset.filter (adj v) V).card = 5))
    (h_red : red_vertices.card = 3) 
    (h_red_subset : red_vertices ⊆ V) : 
    ∃ v ∈ V, (2 ≤ (Finset.filter (λ w, adj v w ∧ w ∈ red_vertices) V).card) := 
by 
  sorry

end icosahedron_red_neighbors_l752_752450


namespace find_a1_l752_752426

theorem find_a1 (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a (n + 2) + (-1) ^ n * a n = 3 * n - 1)
  (h_sum : ∑ i in Finset.range 12, a i = 243) : 
  a 1 = 7 := 
sorry

end find_a1_l752_752426


namespace cubes_sum_l752_752115

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l752_752115


namespace compute_fraction_power_l752_752903

theorem compute_fraction_power :
  8 * (2 / 7)^4 = 128 / 2401 :=
by
  sorry

end compute_fraction_power_l752_752903


namespace tank_dimension_l752_752149

theorem tank_dimension 
  (x : ℝ) 
  (A : ℝ) 
  (cost_per_sqft : ℝ = 20) 
  (total_cost : ℝ = 1440)
  (dim1 : ℝ = 3) 
  (dim2 : ℝ = 6) :
  18 * x + 36 = 72 → x = 2 :=
by
  sorry

end tank_dimension_l752_752149


namespace num_distinct_prime_factors_30_fact_l752_752188

def is_prime (n : ℕ) : Prop := (n > 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def prime_factors (n : ℕ) : Finset ℕ :=
  (Finset.range (n+1)).filter (λ p, is_prime p ∧ p ∣ factorial n)

theorem num_distinct_prime_factors_30_fact : (prime_factors 30).card = 10 :=
by 
  sorry

end num_distinct_prime_factors_30_fact_l752_752188


namespace correct_answer_is_D_l752_752636

-- Definition of planes and lines
variables {Point : Type} [Geometry Point]
variables {Plane : Type} [line : Geometry.line Plane] [plane : Geometry.plane Point]

-- Given variables
variables (α β : Plane) (a b : line)

-- Condition definitions
axiom two_different_planes : α ≠ β
axiom two_non_coincident_lines : ¬ (∃ p : Point, p ∈ a ∧ p ∈ b)

-- Proposition D
axiom α_parallel_β : Geometry.parallel α β
axiom a_not_in_α : ¬ (∀ p : Point, p ∈ a → p ∈ α)
axiom a_not_in_β : ¬ (∀ p : Point, p ∈ a → p ∈ β)
axiom a_parallel_α : Geometry.parallel a α

-- Lean statement for the given problem
theorem correct_answer_is_D : Geometry.parallel a β :=
sorry

end correct_answer_is_D_l752_752636


namespace hexagon_vertices_to_zero_l752_752300

theorem hexagon_vertices_to_zero 
  (a b c d e f : ℕ) 
  (h_sum : a + b + c + d + e + f = 2003) :
  ∃ steps : list (ℕ → ℕ), 
    (∀ v : ℕ, v ∈ steps → 
      v = |(steps.head % steps.nth 4.head )| ) →
    ∀ x y z w u v : ℕ, 
      x = y ∧ y = z ∧ z = w ∧ w = u ∧ u = v ∧ v = 0 :=
sorry

end hexagon_vertices_to_zero_l752_752300


namespace midpoint_locus_distance_to_AB_l752_752987

section hyperbola_problem

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 4 = 1

-- Define the equation of the locus of the midpoint P
def locus_of_midpoint (x y : ℝ) : Prop :=
  3 * (4 * x^2 - y^2)^2 = 4 * (16 * x^2 + y^2)

-- Define the distance from the center O to the line AB
def distance_from_O_to_AB (d : ℝ) : Prop :=
  d = 2 * real.sqrt 3 / 3

-- Theorem statements
theorem midpoint_locus (x y : ℝ) (h1 : hyperbola (x - m) (y - n)) (h2 : hyperbola (x + m) (y + n)) (h3 : (x - m) * (x + m) + (y - n) * (y + n) = 0) : 
  locus_of_midpoint x y :=
sorry

theorem distance_to_AB (x y : ℝ) (d : ℝ) (h1 : hyperbola (x - m) (y - n)) (h2 : hyperbola (x + m) (y + n)) (h3 : (x - m) * (x + m) + (y - n) * (y + n) = 0) :
  distance_from_O_to_AB d :=
sorry

end hyperbola_problem

end midpoint_locus_distance_to_AB_l752_752987


namespace smallest_N_such_that_fn_geq_fn_for_all_n_l752_752704

noncomputable def d (n : ℕ) : ℕ := (Nat.divisors n).length

noncomputable def f (n : ℕ) : ℚ :=
  (d n : ℚ) / Real.toRat (Real.root 4 n.toReal)

theorem smallest_N_such_that_fn_geq_fn_for_all_n (N : ℕ) :
  (∀ n, n ≠ N → ¬(N ∣ n) → f N ≥ f n) ↔ N = 13824 := by
  sorry

end smallest_N_such_that_fn_geq_fn_for_all_n_l752_752704


namespace Zhang_Hai_average_daily_delivery_is_37_l752_752839

theorem Zhang_Hai_average_daily_delivery_is_37
  (d1_packages : ℕ) (d1_count : ℕ)
  (d2_packages : ℕ) (d2_count : ℕ)
  (d3_packages : ℕ) (d3_count : ℕ)
  (total_days : ℕ) 
  (h1 : d1_packages = 41) (h2 : d1_count = 1)
  (h3 : d2_packages = 35) (h4 : d2_count = 2)
  (h5 : d3_packages = 37) (h6 : d3_count = 4)
  (h7 : total_days = 7) :
  (d1_count * d1_packages + d2_count * d2_packages + d3_count * d3_packages) / total_days = 37 := 
by sorry

end Zhang_Hai_average_daily_delivery_is_37_l752_752839


namespace sum_T_eq_correct_l752_752685

-- Definition and conditions for the sequence {a_n}
def a (n : ℕ) : ℕ := 8 * 4^(n - 1)

-- Sequence {b_n} defined by the logarithm of sequence {a_n}
def b (n : ℕ) : ℕ := 2 * n + 1

-- Sum of the first n terms of the sequence {b_n}
def S (n : ℕ) : ℕ := n^2 + 2 * n

-- Sum of the first n terms of the sequence {1/S_n}
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, (1 : ℝ) / (i * (i + 2))

-- Theorem statement that verifies the correct answer for T_n
theorem sum_T_eq_correct (n : ℕ) : 
    T n = (3 : ℝ) / 4 - (2 * n + 3) / (2 * (n + 1) * (n + 2)) :=
sorry  -- Proof of this theorem is omitted

end sum_T_eq_correct_l752_752685


namespace answer_choices_l752_752500

theorem answer_choices (n : ℕ) (h : (n + 1) ^ 4 = 625) : n = 4 :=
by {
  sorry
}

end answer_choices_l752_752500


namespace calculate_C_over_D_is_five_over_two_l752_752299

noncomputable def calculate_ratio (matrix : (Fin 36) → (Fin 90) → ℝ) : ℝ :=
  let row_sums := λ i, ∑ j, matrix i j
  let col_sums := λ j, ∑ i, matrix i j
  let C := (1 / 36) * ∑ i, row_sums i
  let D := (1 / 90) * ∑ j, col_sums j
  C / D

theorem calculate_C_over_D_is_five_over_two (matrix : (Fin 36) → (Fin 90) → ℝ) : calculate_ratio matrix = 5 / 2 := by
  sorry

end calculate_C_over_D_is_five_over_two_l752_752299


namespace problem_conditions_l752_752356

variables {V : Type*} [inner_product_space ℝ V]
variables (m n : V) (α β : submodule ℝ V)

def perpendicular (x y : V) : Prop := ⟪x, y⟫ = 0
def parallel (x y : V) : Prop := ∃ k : ℝ, x = k • y

theorem problem_conditions :
  (α.perpendicular β ∧ perpendicular m α → ¬ intersect m β) ∧
  (perpendicular m n ∧ perpendicular m α → ¬ intersect n α) ∧
  (parallel m α ∧ parallel n α → ¬ parallel m n) ∧
  (perpendicular m β ∧ perpendicular n α → ¬ α.perpendicular β) := sorry

end problem_conditions_l752_752356


namespace FM_perp_EN_l752_752325

variables {A B C D O E F M N : Point}

-- Definitions of the geometric conditions
def is_rectangle (A B C D : Point) : Prop := ∃ (O : Point), (AB ≠ BC) ∧ (O is center of ABCD)

def perpendicular_from_O_to_BD_cuts_AB_BC (O E F : Point) (AB BC BD : Line) : Prop :=
  ∃ (perp : Line), perp ∋ O ∧ perp ⊥ BD ∧ perp ∋ E ∧ perp ∋ F ∧ E ∈ AB ∧ F ∈ BC

def midpoints (M N : Point) (CD AD : Segment) : Prop :=
  midpoint M CD ∧ midpoint N AD

-- The theorem to prove
theorem FM_perp_EN 
  (H_rect : is_rectangle A B C D)
  (H_perp : perpendicular_from_O_to_BD_cuts_AB_BC O E F AB BC BD)
  (H_mid : midpoints M N CD AD) :
  FM ⊥ EN :=
sorry

end FM_perp_EN_l752_752325


namespace find_p_l752_752950

def isosceles_right_triangle_area (a b p : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (p > 0) ∧ 
  let θ := a / b in
  let area := (1 / 2) * (2 * a / b) * (p / 2) in
  let isotropic_cond := θ = p / 2 in
  area = 1 ∧ isotropic_cond

theorem find_p (a b : ℝ) : ∃ p : ℝ, isosceles_right_triangle_area a b p → p = 2 :=
sorry

end find_p_l752_752950


namespace routeY_is_quicker_l752_752727

noncomputable def timeRouteX : ℝ := 
  8 / 40 

noncomputable def timeRouteY1 : ℝ := 
  6.5 / 50 

noncomputable def timeRouteY2 : ℝ := 
  0.5 / 10

noncomputable def timeRouteY : ℝ := 
  timeRouteY1 + timeRouteY2  

noncomputable def timeDifference : ℝ := 
  (timeRouteX - timeRouteY) * 60 

theorem routeY_is_quicker : 
  timeDifference = 1.2 :=
by
  sorry

end routeY_is_quicker_l752_752727


namespace even_function_proof_l752_752159

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (x) = f (-x)

def problem_conditions (f : ℝ → ℝ) : Prop :=
  (f = λ x, sin ((2015 * real.pi / 2) + x)) ∨
  (f = λ x, cos ((2015 * real.pi / 2) + x)) ∨
  (f = λ x, tan ((2015 * real.pi / 2) + x)) ∨
  (f = λ x, sin ((2014 * real.pi / 2) + x))

theorem even_function_proof :
  ∃ f, problem_conditions f ∧ is_even_function f :=
begin
  use (λ x, sin ((2015 * real.pi / 2) + x)),
  split,
  { left, refl },
  { intros x, simp }
end

end even_function_proof_l752_752159


namespace length_of_EG_l752_752739

theorem length_of_EG 
  (inscribed : ∀ (E F G H : Type), IsInscribedInCircle (E F G H))
  (angle_EFG : angle E F G = 80)
  (angle_EHG : angle E H G = 50)
  (length_EH : dist E H = 5)
  (length_FG : dist F G = 7) :
  dist E G = 7 := 
sorry

end length_of_EG_l752_752739


namespace sum_of_cubes_l752_752110

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752110


namespace product_of_common_divisors_l752_752574

/-- Definition for 180_factored -/
def factored_180 := [1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 9, -9, 10, -10, 12, -12, 15, -15, 18, -18, 20, -20, 30, -30, 36, -36, 45, -45, 60, -60, 90, -90, 180, -180]

/-- Definition for 45_factored -/
def factored_45 := [1, -1, 3, -3, 5, -5, 9, -9, 15, -15, 45, -45]

/-- Definition for common divisors between 180 and 45 -/
def common_divisors := [1, -1, 3, -3, 5, -5, 9, -9, 15, -15, 45, -45]

/-- The product of all integer divisors of 180 that also divide 45 is 8305845625 -/
theorem product_of_common_divisors : (common_divisors.foldl (*) 1) = 8305845625 := by
  sorry

end product_of_common_divisors_l752_752574


namespace basketball_committee_l752_752478

theorem basketball_committee (total_players guards : ℕ) (choose_committee choose_guard : ℕ) :
  total_players = 12 → guards = 4 → choose_committee = 3 → choose_guard = 1 →
  (guards * ((total_players - guards).choose (choose_committee - choose_guard)) = 112) :=
by
  intros h_tp h_g h_cc h_cg
  rw [h_tp, h_g, h_cc, h_cg]
  simp
  norm_num
  sorry

end basketball_committee_l752_752478


namespace number_of_triples_l752_752974

def count_triples (a b c : ℕ) : Prop :=
  (2017 ≥ 10 * a ∧ 10 * a ≥ 100 * b ∧ 100 * b ≥ 1000 * c ∧ a > 0 ∧ b > 0 ∧ c > 0)

theorem number_of_triples :
  (∑ (c : ℕ) in {1, 2}, ∑ (b : ℕ) in {(ceil (c / 1000 * 100):ℕ) .. min (2017 / 100):ℕ}, 
  (b * 100 / 10 .. min 201 (b * 100 / 10)) = 574 :=
begin
  sorry
end

end number_of_triples_l752_752974


namespace transaction_gain_per_year_l752_752501

noncomputable def principal : ℝ := 9000
noncomputable def time : ℝ := 2
noncomputable def rate_lending : ℝ := 6
noncomputable def rate_borrowing : ℝ := 4

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def total_interest_earned := simple_interest principal rate_lending time
noncomputable def total_interest_paid := simple_interest principal rate_borrowing time

noncomputable def total_gain := total_interest_earned - total_interest_paid
noncomputable def gain_per_year := total_gain / 2

theorem transaction_gain_per_year : gain_per_year = 180 :=
by
  sorry

end transaction_gain_per_year_l752_752501


namespace isosceles_right_triangle_area_l752_752047

theorem isosceles_right_triangle_area (h : ℝ) (h_eq : h = 6 * Real.sqrt 2) : 
  ∃ A : ℝ, A = 18 := by 
  sorry

end isosceles_right_triangle_area_l752_752047


namespace smallest_x_y_z_sum_l752_752714

theorem smallest_x_y_z_sum :
  ∃ x y z : ℝ, x + 3*y + 6*z = 1 ∧ x*y + 2*x*z + 6*y*z = -8 ∧ x*y*z = 2 ∧ x + y + z = -(8/3) := 
sorry

end smallest_x_y_z_sum_l752_752714


namespace inverse_of_exp2_inverse_function_of_exp2_l752_752930

namespace InverseFunction

noncomputable def inverse_function_example : (x : ℝ) → ℝ :=
λ x, Real.log x / Real.log 2

theorem inverse_of_exp2 (x : ℝ) (h : 0 < x) : 2 ^ (Real.log x / Real.log 2) = x :=
by sorry

theorem inverse_function_of_exp2 :
  function.bijective (λ x : {x // 0 < x}, Real.log x / Real.log 2) ∧ (∀ x ∈ set.univ, (Real.log 2) * Real.log (2 ^ x) = x) :=
by sorry

end InverseFunction

end inverse_of_exp2_inverse_function_of_exp2_l752_752930


namespace calculate_difference_l752_752191

def total_students : ℕ := 2001
def S_lower_bound : ℕ := 1601
def S_upper_bound : ℕ := 1700
def F_lower_bound : ℕ := 601
def F_upper_bound : ℕ := 800

theorem calculate_difference :
  ∃ (m M : ℕ), 
  (∀ S F : ℕ, (S_lower_bound ≤ S ∧ S ≤ S_upper_bound) ∧ (F_lower_bound ≤ F ∧ F ≤ F_upper_bound) → 
  let B := S + F - total_students in
  m = S_lower_bound + F_lower_bound - total_students ∧
  M = S_upper_bound + F_upper_bound - total_students ∧
  M - m = 298) :=
sorry

end calculate_difference_l752_752191


namespace no_positive_integer_makes_expression_integer_l752_752565

theorem no_positive_integer_makes_expression_integer : 
  ∀ n : ℕ, n > 0 → ¬ ∃ k : ℤ, (n^(3 * n - 2) - 3 * n + 1) = k * (3 * n - 2) := 
by 
  intro n hn
  sorry

end no_positive_integer_makes_expression_integer_l752_752565


namespace head_start_proofs_l752_752296

def HeadStartAtoB : ℕ := 150
def HeadStartAtoC : ℕ := 310
def HeadStartAtoD : ℕ := 400

def HeadStartBtoC : ℕ := HeadStartAtoC - HeadStartAtoB
def HeadStartCtoD : ℕ := HeadStartAtoD - HeadStartAtoC
def HeadStartBtoD : ℕ := HeadStartAtoD - HeadStartAtoB

theorem head_start_proofs :
  (HeadStartBtoC = 160) ∧
  (HeadStartCtoD = 90) ∧
  (HeadStartBtoD = 250) :=
by
  sorry

end head_start_proofs_l752_752296


namespace garbage_collection_l752_752912

theorem garbage_collection (Daliah Dewei Zane : ℝ) 
(h1 : Daliah = 17.5)
(h2 : Dewei = Daliah - 2)
(h3 : Zane = 4 * Dewei) :
Zane = 62 :=
sorry

end garbage_collection_l752_752912


namespace no_positive_integers_xy_l752_752330

theorem no_positive_integers_xy 
  (n : ℕ) (hn : n > 0) : 
  ¬ ∃ (x y : ℕ), (x > 0) ∧ (y > 0) ∧ (sqrt n + sqrt (n+1) < sqrt x + sqrt y) ∧ (sqrt x + sqrt y < sqrt (4*n + 2)) :=
by
  sorry

end no_positive_integers_xy_l752_752330


namespace proof_C1_cartesian_proof_C1_polar_proof_C2_cartesian_proof_C2_polar_length_PQ_l752_752622

noncomputable def C1_cartesian (x y : ℝ) : Prop :=
  (x - sqrt 3) ^ 2 + (y + 1) ^ 2 = 9

noncomputable def C1_polar (ρ θ : ℝ) : Prop :=
  ρ ^ 2 - 2 * sqrt 3 * ρ * cos θ + 2 * ρ * sin θ - 5 = 0

noncomputable def C2_cartesian (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 2 * x

noncomputable def C2_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * cos θ

theorem proof_C1_cartesian (x y : ℝ) :
  C1_cartesian x y ↔ x ^ 2 + y ^ 2 - 2 * sqrt 3 * x + 2 * y - 5 = 0 := sorry

theorem proof_C1_polar (ρ θ : ℝ) :
  C1_polar ρ θ ↔ ρ ^ 2 - 2 * sqrt 3 * ρ * cos θ + 2 * ρ * sin θ - 5 = 0 := sorry

theorem proof_C2_cartesian (x y : ℝ) :
  C2_cartesian x y ↔ x ^ 2 + y ^ 2 = 2 * x := sorry

theorem proof_C2_polar (ρ θ : ℝ) :
  C2_polar ρ θ ↔ ρ = 2 * cos θ := sorry

theorem length_PQ (ρ1 ρ2 : ℝ) :
  ρ1 = 1 + sqrt 6 → ρ2 = 1 - sqrt 6 → |ρ1 - ρ2| = 2 * sqrt 6 := sorry

end proof_C1_cartesian_proof_C1_polar_proof_C2_cartesian_proof_C2_polar_length_PQ_l752_752622


namespace equalize_cheese_pieces_l752_752790

-- Defining the initial masses of the three pieces of cheese
def cheese1 : ℕ := 5
def cheese2 : ℕ := 8
def cheese3 : ℕ := 11

-- State that the fox can cut 1g simultaneously from any two pieces
def can_equalize_masses (cut_action : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ n1 n2 n3 _ : ℕ,
    cut_action cheese1 cheese2 cheese3 ∧
    (n1 = 0 ∧ n2 = 0 ∧ n3 = 0)

-- Introducing the fox's cut action
def cut_action (a b c : ℕ) : Prop :=
  (∃ x : ℕ, x ≥ 0 ∧ a - x ≥ 0 ∧ b - x ≥ 0 ∧ c ≤ cheese3) ∧
  (∃ y : ℕ, y ≥ 0 ∧ a - y ≥ 0 ∧ b ≤ cheese2 ∧ c - y ≥ 0) ∧
  (∃ z : ℕ, z ≥ 0 ∧ a ≤ cheese1 ∧ b - z ≥ 0 ∧ c - z ≥ 0) 

-- The theorem that proves it's possible to equalize the masses
theorem equalize_cheese_pieces : can_equalize_masses cut_action :=
by
  sorry

end equalize_cheese_pieces_l752_752790


namespace digit_to_make_divisible_by_7_l752_752084

theorem digit_to_make_divisible_by_7 :
  ∃ (x : ℕ), (x < 10) ∧ (let num := (50 : ℕ).repeat 8 ++ [x] ++ (50 : ℕ).repeat 9;
                             nat_of_digit_list num) % 7 = 0 :=
  sorry

end digit_to_make_divisible_by_7_l752_752084


namespace min_distance_PQ_is_3sqrt10_10_l752_752617
open Real

noncomputable def curve (x : ℝ) : ℝ := 2 * exp x + x
noncomputable def line (x : ℝ) : ℝ := 3 * x - 1

theorem min_distance_PQ_is_3sqrt10_10 :
  let P := (0, curve 0)
      Q := (Qx : ℝ, line Qx)
  in min (dist P Q) = 3 * sqrt 10 / 10 :=
by
  sorry

end min_distance_PQ_is_3sqrt10_10_l752_752617


namespace Shara_shells_total_l752_752017

def initial_shells : ℕ := 20
def shells_per_day : ℕ := 5
def days : ℕ := 3
def shells_fourth_day : ℕ := 6

theorem Shara_shells_total : (initial_shells + (shells_per_day * days) + shells_fourth_day) = 41 := by
  sorry

end Shara_shells_total_l752_752017


namespace extreme_values_of_cubic_function_l752_752880

-- Assuming f is a cubic function and f' is its derivative.
variables {f : ℝ → ℝ} (hf : ∃ a b c d : ℝ, ∀ x : ℝ, f x = a * x^3 + b * x^2 + c * x + d)

-- Condition on the derivative of f
variable (hf' : ∀ x : ℝ, f' x = deriv f x)

-- Additional Conditions derived from the graph of y = x * f'(x)
variable (h1 : ∀ x < -2, f' x > 0)
variable (h2 : ∀ x > -2 ∧ x < 2, f' x ≤ 0)
variable (h3 : ∀ x > 2, f' x > 0)

theorem extreme_values_of_cubic_function :
  (∀ x : ℝ, f x ≤ f (-2)) ∧ (∀ x : ℝ, f 2 ≤ f x) :=
sorry

end extreme_values_of_cubic_function_l752_752880


namespace distance_from_point_to_line_l752_752204

noncomputable def distance_point_to_line : ℝ :=
  let a : ℝ × ℝ × ℝ := (2, -2, 1)
  let b : ℝ × ℝ × ℝ := (1, 1, 0)
  let c : ℝ × ℝ × ℝ := (3, -2, 4)
  let line_direction : ℝ × ℝ × ℝ := (c.1 - b.1, c.2 - b.2, c.3 - b.3)
  let v : ℝ × ℝ × ℝ := (b.1 + line_direction.1, b.2 + line_direction.2, b.3 + line_direction.3)
  let vector_from_a_to_v := (v.1 - a.1, v.2 - a.2, v.3 - a.3)
  let t := (-(vector_from_a_to_v.1 * line_direction.1 + vector_from_a_to_v.2 * line_direction.2 + vector_from_a_to_v.3 * line_direction.3)) / (line_direction.1^2 + line_direction.2^2 + line_direction.3^2)
  let closest_point_on_line : ℝ × ℝ × ℝ := (b.1 + t * line_direction.1, b.2 + t * line_direction.2, b.3 + t * line_direction.3)
  let distance_squared := (a.1 - closest_point_on_line.1)^2 + (a.2 - closest_point_on_line.2)^2 + (a.3 - closest_point_on_line.3)^2
  sqrt distance_squared

theorem distance_from_point_to_line : distance_point_to_line = sqrt 10 := by
  sorry

end distance_from_point_to_line_l752_752204


namespace magnitude_difference_l752_752639

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (norm_a : ‖a‖ = 2) (norm_b : ‖b‖ = 1) (norm_a_plus_b : ‖a + b‖ = Real.sqrt 3)

theorem magnitude_difference :
  ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end magnitude_difference_l752_752639


namespace globe_division_l752_752843

theorem globe_division (parallels meridians : ℕ) (h_parallels : parallels = 17) (h_meridians : meridians = 24) : 
  (meridians * (parallels + 1)) = 432 :=
by
  rw [h_parallels, h_meridians]
  simp
  sorry

end globe_division_l752_752843


namespace smallest_positive_value_l752_752552

noncomputable def sqrt7 : ℝ := Real.sqrt 7
noncomputable def sqrt19 : ℝ := Real.sqrt 19
noncomputable def sqrt17 : ℝ := Real.sqrt 17

def optionA : ℝ := 12 - 4 * sqrt7
def optionB : ℝ := 4 * sqrt7 - 12
def optionC : ℝ := 25 - 6 * sqrt19
def optionD : ℝ := 65 - 15 * sqrt17
def optionE : ℝ := 15 * sqrt17 - 65

theorem smallest_positive_value :
  optionA = minPositiveIfInSet
  (λ x, x = optionA ∨ x = optionB ∨ x = optionC ∨ x = optionD ∨ x = optionE)
  (λ x, 0 < x)
  := sorry

end smallest_positive_value_l752_752552


namespace problem_statement_l752_752687

def class_of_rem (k : ℕ) : Set ℤ := {n | ∃ m : ℤ, n = 4 * m + k}

theorem problem_statement : (2013 ∈ class_of_rem 1) ∧ 
                            (-2 ∈ class_of_rem 2) ∧ 
                            (∀ x : ℤ, x ∈ class_of_rem 0 ∨ x ∈ class_of_rem 1 ∨ x ∈ class_of_rem 2 ∨ x ∈ class_of_rem 3) ∧ 
                            (∀ a b : ℤ, (∃ k : ℕ, (a ∈ class_of_rem k ∧ b ∈ class_of_rem k)) ↔ (a - b) ∈ class_of_rem 0) :=
by
  -- each of the statements should hold true
  sorry

end problem_statement_l752_752687


namespace cos_sin_equation_solution_l752_752397

noncomputable def solve_cos_sin_equation (x : ℝ) (n : ℤ) : Prop :=
  let lhs := (Real.cos x) / (Real.sqrt 3)
  let rhs := Real.sqrt ((1 - (Real.cos (2*x)) - 2 * (Real.sin x)^3) / (6 * Real.sin x - 2))
  (lhs = rhs) ∧ (Real.cos x ≥ 0)

theorem cos_sin_equation_solution:
  (∃ (x : ℝ) (n : ℤ), solve_cos_sin_equation x n) ↔ 
  ∃ (n : ℤ), (x = (π / 2) + 2 * π * n) ∨ (x = (π / 6) + 2 * π * n) :=
by
  sorry

end cos_sin_equation_solution_l752_752397


namespace officers_selection_l752_752487

theorem officers_selection (members : List String) (alice bob : String)
  (h_mem_alice : alice ∈ members) (h_mem_bob : bob ∈ members) (h_length : members.length = 30)
  (h_alice_bob_condition : ∀ p v s t : String, ((alice ∈ [p, v, s, t]) ↔ (bob ∈ [p, v, s, t]))) :
  let total_ways :=
    -- Case 1: Alice and Bob are not officers
    (28 * 27 * 26 * 25) +
    -- Case 2: Alice and Bob are officers
    (Nat.choose 4 2 * 28 * 27) 
  in total_ways = 495936 := by
  sorry

end officers_selection_l752_752487


namespace find_x_l752_752815

-- Define the vectors m and n
def vector_m : ℝ × ℝ := (3, -2)
def vector_n (x : ℝ) : ℝ × ℝ := (1, x)

-- Define the condition that vector_m is perpendicular to (vector_m + vector_n)
def is_perpendicular (v w : ℝ × ℝ) : Prop := 
  v.1 * w.1 + v.2 * w.2 = 0

-- Prove that x must equal 8 under these conditions
theorem find_x (x : ℝ) : 
  is_perpendicular vector_m (vector_m.1 + vector_n x.1, vector_m.2 + vector_n x.2) -> 
  x = 8 := 
sorry

end find_x_l752_752815


namespace find_angles_l752_752306

open Real

-- Define the triangle KLM as an acute triangle
def is_acute_triangle (K L M : Point) : Prop :=
  let angles : Array ℝ := [angle K L M, angle L M K, angle M K L]
  angles.all (fun α => α < π/2) ∧
  angles.sum = π

-- Define V as the orthocenter of the triangle
def is_orthocenter (V K L M : Point) : Prop :=
  let altitudes : Array Line := [altitude V K L, altitude V L M, altitude V M K]
  altitudes.all (fun l => meets_at_perpendiculars l (vertices_opposite V K L M))

-- Define X as the foot of the altitude from V to side KL
def is_foot_of_altitude (V X K L : Point) : Prop :=
  altitude V K L ∧ point_on_line X (altitude_line V K L) ∧ 
  angle X K L = π/2

-- Define angle bisector condition
def bisector_parallel_to_LM (V X L M : Point) : Prop := 
  let bisector := angle_bisector (angle_between_points X V L)
  is_parallel bisector (line_through L M)

-- Define the given angle condition
def angle_mkl_equals_seventy (K L M : Point) : Prop :=
  angle M K L = 70 * π / 180 -- converting 70 degrees to radians

-- Define main theorem
theorem find_angles {K L M V X : Point} 
  (h_acute : is_acute_triangle K L M) 
  (h_orthocenter : is_orthocenter V K L M) 
  (h_foot : is_foot_of_altitude V X K L) 
  (h_bisector : bisector_parallel_to_LM V X L M) 
  (h_angle_mkl : angle_mkl_equals_seventy K L M) : 
  angle K L M = 55 * π / 180 ∧ angle K M L = 55 * π / 180 := 
sorry

end find_angles_l752_752306


namespace sum_of_cubes_l752_752092

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l752_752092


namespace mixed_solution_concentration_l752_752462

-- Defining the conditions as given in the question
def weight1 : ℕ := 200
def concentration1 : ℕ := 25
def saltInFirstSolution : ℕ := (concentration1 * weight1) / 100

def weight2 : ℕ := 300
def saltInSecondSolution : ℕ := 60

def totalSalt : ℕ := saltInFirstSolution + saltInSecondSolution
def totalWeight : ℕ := weight1 + weight2

-- Statement of the proof
theorem mixed_solution_concentration :
  ((totalSalt : ℚ) / (totalWeight : ℚ)) * 100 = 22 :=
by
  sorry

end mixed_solution_concentration_l752_752462


namespace number_of_odd_exponents_l752_752633

-- Define the set of exponents
def exponents : Set ℚ := {-2, -1, -1/2, 1/3, 1/2, 1, 2, 3}

-- Define the predicate for checking if y = x^a is an odd function
def is_odd_function (a : ℚ) : Prop := ∀ x : ℝ, (x ≠ 0 → x ^ a = (-x) ^ a)

-- Extract the subset of exponents that make the function y = x^a odd
def odd_exponents : Set ℚ := {a ∈ exponents | is_odd_function a}

-- Prove that there are exactly 4 odd exponents
theorem number_of_odd_exponents : odd_exponents.card = 4 := by
  sorry

end number_of_odd_exponents_l752_752633


namespace largest_value_of_expressions_l752_752715

-- Definition of y based on the problem conditions
def y : ℝ := 10 ^ (-2023)

-- Proof statement
theorem largest_value_of_expressions :
  let A := 5 + y in
  let B := 5 - y in
  let C := 5 * y in
  let D := 5 / y in
  let E := y / 5 in
  D > A ∧ D > B ∧ D > C ∧ D > E :=
sorry

end largest_value_of_expressions_l752_752715


namespace sarah_books_check_out_l752_752013

theorem sarah_books_check_out
  (words_per_minute : ℕ)
  (words_per_page : ℕ)
  (pages_per_book : ℕ)
  (reading_hours : ℕ)
  (number_of_books : ℕ) :
  words_per_minute = 40 →
  words_per_page = 100 →
  pages_per_book = 80 →
  reading_hours = 20 →
  number_of_books = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end sarah_books_check_out_l752_752013


namespace cover_by_strip_l752_752506

theorem cover_by_strip {n : ℕ} (S : Finset (ℝ × ℝ)) (h_n : 3 ≤ n)
  (h_points : ∀ (A B C : ℝ × ℝ), A ∈ S → B ∈ S → C ∈ S → A ≠ B → B ≠ C → A ≠ C →
    ∃ (l₁ l₂ : ℝ → (ℝ × ℝ) → Prop), (∀ p ∈ [A, B, C]], p ∈ [l₁, l₂] ∧ parallel_line_distance l₁ l₂ = 1)) :
  ∃ (l₁ l₂ : ℝ → (ℝ × ℝ) → Prop), (∀ p ∈ S, p ∈ [l₁, l₂]) ∧ parallel_line_distance l₁ l₂ = 2 :=
sorry

end cover_by_strip_l752_752506


namespace monotonicity_S_l752_752627

def S (t : ℝ) : ℝ :=
if 0 < t ∧ t < 1/2 then 2 * (1 - t + t^2 - t^3)
else if t ≥ 1/2 then 1/2 * (t + 1/t)
else 0

theorem monotonicity_S : 
  (∀ t ∈ Ioo (0 : ℝ) 1, differentiable_at ℝ S t ∧ S' t < 0) ∧
  (∀ t ∈ Icc (1 : ℝ) ∞, differentiable_at ℝ S t ∧ S' t > 0) :=
by
  sorry

end monotonicity_S_l752_752627


namespace quarters_count_l752_752463

noncomputable def num_coins := 12
noncomputable def total_value := 166 -- in cents
noncomputable def min_value := 1 + 5 + 10 + 25 + 50 -- minimum value from one of each type
noncomputable def remaining_value := total_value - min_value
noncomputable def remaining_coins := num_coins - 5

theorem quarters_count :
  ∀ (p n d q h : ℕ), 
  p + n + d + q + h = num_coins ∧
  p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1 ∧ h ≥ 1 ∧
  (p + 5*n + 10*d + 25*q + 50*h = total_value) → 
  q = 3 := 
by 
  sorry

end quarters_count_l752_752463


namespace median_of_throws_is_15_5_l752_752477

def throws : List ℕ := [5, 17, 16, 14, 20, 11, 20, 15, 18, 10]

theorem median_of_throws_is_15_5 :
  let sorted_throws := throws.qsort (· <= ·)
  let n := sorted_throws.length
  let mid := n / 2
  ∀ median, median = ((sorted_throws.get (mid - 1) + sorted_throws.get mid).toDouble) / 2 →
  median = 15.5 :=
by
  let sorted_throws := throws.qsort (· <= ·)
  let n := sorted_throws.length
  let mid := n / 2
  let median := ((sorted_throws.get (mid - 1) + sorted_throws.get mid).toDouble) / 2
  show median = 15.5
  sorry

end median_of_throws_is_15_5_l752_752477


namespace optionC_correct_l752_752613

variables (a b : Line) (α β : Plane)
hypothesis h1 : a ⊂ α
hypothesis h2 : b ⊥ β
hypothesis h3 : α ∥ β

theorem optionC_correct : a ⊥ b :=
sorry

end optionC_correct_l752_752613


namespace geometric_sum_squares_l752_752781

theorem geometric_sum_squares (n : ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n, (finset.range (n+1)).sum a = 2^n - 1) :
  (finset.range n).sum (λ k, (a k)^2) = (4^n - 1) / 3 := 
sorry

end geometric_sum_squares_l752_752781


namespace isosceles_triangle_perimeter_l752_752237

theorem isosceles_triangle_perimeter (a b : ℕ) (h_eq : a = 5 ∨ a = 9) (h_side : b = 9 ∨ b = 5) (h_neq : a ≠ b) : 
  (a + a + b = 19 ∨ a + a + b = 23) :=
by
  sorry

end isosceles_triangle_perimeter_l752_752237


namespace find_parameter_a_l752_752218

-- Define the polynomial coefficients and the resulting value of 'a' for the roots in GP
noncomputable def a : ℝ := 60

def is_geometric_progression (x1 x2 x3 : ℝ) : Prop :=
  ∃ b q : ℝ, x1 = b ∧ x2 = b * q ∧ x3 = b * q^2

def distinct_real_roots {a b c d : ℝ} (p : Polynomial ℝ) : Prop :=
  p.natDegree = 3 ∧ ∀ (x y : ℝ), Polynomial.eval x p = 0 → Polynomial.eval y p = 0 → x = y → false

def polynomial (a : ℝ) : Polynomial ℝ := X^3 - 15 * X^2 + a * X - 64

theorem find_parameter_a :
  ∃ (a : ℝ), distinct_real_roots (polynomial a) ∧ 
              ∀ r1 r2 r3 : ℝ, Polynomial.eval r1 (polynomial a) = 0 → 
                              Polynomial.eval r2 (polynomial a) = 0 → 
                              Polynomial.eval r3 (polynomial a) = 0 →
                              is_geometric_progression r1 r2 r3 → 
                              a = 60 :=
begin
  sorry
end

end find_parameter_a_l752_752218


namespace gcd_21_n_eq_7_count_l752_752936

theorem gcd_21_n_eq_7_count :
  {n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ Nat.gcd 21 n = 7}.card = 19 := 
by sorry

end gcd_21_n_eq_7_count_l752_752936


namespace inverse_of_square_positive_is_negative_l752_752413

variable {x : ℝ}

-- Original proposition: ∀ x, x < 0 → x^2 > 0
def original_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 > 0

-- Inverse proposition to be proven: ∀ x, x^2 > 0 → x < 0
def inverse_proposition (x : ℝ) : Prop :=
  x^2 > 0 → x < 0

theorem inverse_of_square_positive_is_negative :
  (∀ x : ℝ, x < 0 → x^2 > 0) → (∀ x : ℝ, x^2 > 0 → x < 0) :=
  sorry

end inverse_of_square_positive_is_negative_l752_752413


namespace price_per_pound_of_apples_l752_752316

-- Conditions
def half_apple_per_day : ℝ := 1 / 2
def weight_of_one_apple : ℝ := 1 / 4
def cost_for_2_weeks : ℝ := 7
def days_in_2_weeks : ℝ := 14

-- Derived information
def apples_consumed_in_2_weeks (days: ℝ) (half_apples: ℝ) : ℝ :=
  days * half_apples

def total_weight_of_apples (apples: ℝ) (weight_per_apple: ℝ) : ℝ :=
  apples * weight_per_apple

def price_per_pound (total_cost: ℝ) (total_weight: ℝ) : ℝ :=
  total_cost / total_weight

-- Proof statement
theorem price_per_pound_of_apples : 
  price_per_pound cost_for_2_weeks (total_weight_of_apples (apples_consumed_in_2_weeks days_in_2_weeks half_apple_per_day) weight_of_one_apple) = 4 :=
by
  sorry

end price_per_pound_of_apples_l752_752316


namespace part_one_part_two_part_three_l752_752988

-- Definitions for Part (1)
def line_eq (k : ℝ) (x : ℝ) := k * x + k - 1
def fixed_point := (-1 : ℝ, -1 : ℝ)

-- Part (1): Prove the line passes through the fixed point
theorem part_one (k : ℝ) : line_eq k (-1) = -1 :=
by sorry

-- Definitions for Part (2)
def below_x_axis (k : ℝ) (x : ℝ) : Prop := line_eq k x ≤ 0

-- Part (2): Prove the range of k
theorem part_two (k : ℝ) (h1: ∀ x, -4 < x → x < 4 → below_x_axis k x) :
  -1/3 ≤ k ∧ k ≤ 1/5 :=
by sorry

-- Definitions for Part (3)
def triangle_area (k : ℝ) : ℝ := 
  let x_intercept := (-(k - 1)) / k in
  let height := k - 1 in
  (1 / 2) * x_intercept * height

-- Part (3): Prove the equations of line l with area 1
theorem part_three (k : ℝ) (h : triangle_area k = 1) :
  k = 2 + Real.sqrt 3 ∨ k = 2 - Real.sqrt 3 :=
by sorry

end part_one_part_two_part_three_l752_752988


namespace trig_identity_evaluation_l752_752888

theorem trig_identity_evaluation :
  sin (Real.pi * 68 / 180) * sin (Real.pi * 67 / 180) - sin (Real.pi * 23 / 180) * cos (Real.pi * 68 / 180)
  = (Real.sqrt 2) / 2 :=
by sorry

end trig_identity_evaluation_l752_752888


namespace marbles_problem_l752_752856

theorem marbles_problem (n : ℕ) :
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0) → 
  n - 10 = 830 :=
sorry

end marbles_problem_l752_752856


namespace sum_of_cubes_l752_752111

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752111


namespace trigonometric_identity_l752_752897

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) - 2 / Real.sin (70 * Real.pi / 180)) = 4 * Real.cot (40 * Real.pi / 180) :=
by
  -- The proof will be skipped with sorry
  sorry

end trigonometric_identity_l752_752897


namespace one_time_purchase_600_purchase_payment_160_actual_payment_x_actual_payment_x_gt_500_actual_payment_900_two_days_discount_when_a_250_l752_752140

-- Condition Definitions
def discount_price (x : ℕ) : ℕ :=
  if x < 200 then x
  else if x < 500 then (8 * x) / 10
  else (4 * 500) + (7 * (x - 500)) / 10

def total_shopping_payment (a : ℕ) : ℕ :=
  if 200 < a ∧ a < 300 then 8 * a / 10 + (8 * 500 / 10 + 7 * (900 - a - 500) / 10) else 0

-- Proof Problems
theorem one_time_purchase_600 : discount_price 600 = 470 := by sorry

theorem purchase_payment_160 (x : ℕ) : 
  x < 200 ∧ 160 = x ∨ (200 ≤ x ∧ x < 500) ∧ 160 = 8 * x / 10 :=
by sorry

theorem actual_payment_x (x : ℕ) (h₁ : 200 ≤ x ∧ x < 500) : discount_price x = 8 * x / 10 :=
by sorry

theorem actual_payment_x_gt_500 (x : ℕ) (h₂ : x ≥ 500) : discount_price x = 7 * x / 10 + 50 :=
by sorry

theorem actual_payment_900_two_days (a : ℕ) (h₃ : 200 < a ∧ a < 300) : total_shopping_payment a = 0.1 * a + 680 :=
by sorry

theorem discount_when_a_250 : total_shopping_payment 250 = 195 :=
by sorry

end one_time_purchase_600_purchase_payment_160_actual_payment_x_actual_payment_x_gt_500_actual_payment_900_two_days_discount_when_a_250_l752_752140


namespace sqrt_expr_domain_l752_752074

theorem sqrt_expr_domain (x : ℝ) : (x - 2 ≥ 0) ↔ (x ≥ 2) :=
begin
  sorry
end

end sqrt_expr_domain_l752_752074


namespace main_theorem_l752_752548

-- Define central distance for a function f
noncomputable def central_distance (f : ℝ → ℝ) : ℝ :=
  infi (λ x, real.sqrt (x^2 + (f x)^2))

-- Propositional statements
def prop1 : Prop :=
  central_distance (λ x : ℝ, 1 / x) > 1

def prop2 : Prop :=
  central_distance (λ x : ℝ, real.sqrt (-x^2 - 4*x + 5)) > 1

-- Assuming equal central distance, the subtraction function must have at least one zero
def prop3 (f g : ℝ → ℝ) (h : central_distance f = central_distance g) : Prop :=
  ∃ x, f x - g x = 0

-- Main theorem combining the propositions
theorem main_theorem : prop1 ∧ prop2 ∧ ∃ f g, (central_distance f = central_distance g) ∧ prop3 f g :=
by {
  sorry  
}

end main_theorem_l752_752548


namespace max_value_of_combined_function_l752_752022

-- Define real functions f and g with their respective ranges
variable {f g : ℝ → ℝ}
variable (hf : ∀ x, -3 ≤ f x ∧ f x ≤ 5)
variable (hg : ∀ x, -4 ≤ g x ∧ g x ≤ 2)

-- Define the range of the combined function 2f(x)g(x) + f(x)
def combined_range : Set ℝ := {y | ∃ x, y = 2 * f x * g x + f x}

-- Theorem stating the maximum value of d
theorem max_value_of_combined_function : ∃ d, d = 45 ∧ ∀ y ∈ combined_range hf hg, y ≤ d := by
  sorry

end max_value_of_combined_function_l752_752022


namespace sum_abs_roots_l752_752575

theorem sum_abs_roots :
  let p : Polynomial ℝ := Polynomial.C 1 * (Polynomial.X ^ 4)
                    - Polynomial.C 6 * (Polynomial.X ^ 3)
                    + Polynomial.C 9 * (Polynomial.X ^ 2)
                    + Polynomial.C 24 * Polynomial.X
                    - Polynomial.C 36 in
  (Polynomial.roots p).map Real.abs.sum = 3 + 4 * Real.sqrt 3 :=
by
  sorry

end sum_abs_roots_l752_752575


namespace a_perpendicular_to_a_minus_b_l752_752276

def vector := ℝ × ℝ

def dot_product (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2

def a : vector := (-2, 1)
def b : vector := (-1, 3)

def a_minus_b : vector := (a.1 - b.1, a.2 - b.2) 

theorem a_perpendicular_to_a_minus_b : dot_product a a_minus_b = 0 := by
  sorry

end a_perpendicular_to_a_minus_b_l752_752276


namespace round_81_739_to_nearest_whole_l752_752162

def round_to_nearest_whole (x : ℝ) : ℤ :=
  if x - x.floor < 0.5 then x.floor.to_int else (x.floor + 1).to_int

theorem round_81_739_to_nearest_whole :
  round_to_nearest_whole 81.739 = 82 := 
by
  sorry

end round_81_739_to_nearest_whole_l752_752162


namespace distinct_positive_least_sum_seven_integers_prod_2016_l752_752568

theorem distinct_positive_least_sum_seven_integers_prod_2016 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    n1 < n2 ∧ n2 < n3 ∧ n3 < n4 ∧ n4 < n5 ∧ n5 < n6 ∧ n6 < n7 ∧
    (n1 * n2 * n3 * n4 * n5 * n6 * n7) % 2016 = 0 ∧
    n1 + n2 + n3 + n4 + n5 + n6 + n7 = 31 :=
sorry

end distinct_positive_least_sum_seven_integers_prod_2016_l752_752568


namespace number_of_cycles_odd_l752_752933

/-- The main theorem stating the equivalence of the conditions -/
theorem number_of_cycles_odd (m : ℕ) (h1 : m > 1) (h2 : ¬ (3 ∣ m)) :
    (odd (f(m))) ↔ (m % 12 ∈ {2, 5, 7, 10}) := 
sorry

end number_of_cycles_odd_l752_752933


namespace compute_result_l752_752901

noncomputable def compute_expr : ℚ :=
  8 * (2 / 7)^4

theorem compute_result : compute_expr = 128 / 2401 := 
by 
  sorry

end compute_result_l752_752901


namespace symmetric_about_y_axis_l752_752422

theorem symmetric_about_y_axis (f : ℝ → ℝ) : 
  ∀ x : ℝ, f(-x) = f(x) → (x, f x) = (-x, f (-x)) :=
by
  sorry

end symmetric_about_y_axis_l752_752422


namespace poly_not_factorable_l752_752603

def notDivisibleByFive (k : ℤ) : Prop :=
  ¬ (5 ∣ k)

theorem poly_not_factorable (k : ℤ) (h : notDivisibleByFive k) : ¬ ∃ (p q : polynomial ℤ), p.degree < 5 ∧ q.degree < 5 ∧ p * q = polynomial.C k + polynomial.X ^ 5 - polynomial.X :=
sorry

end poly_not_factorable_l752_752603


namespace leftmost_three_digits_of_num_four_ring_arrangements_l752_752606

noncomputable def num_four_ring_arrangements (n k : Nat) : ℕ :=
  Nat.choose n k * Nat.factorial k * Nat.choose n (k - 1)

def leftmost_three_nonzero_digits (n : ℕ) : ℕ :=
  let s := toString n
  let digits := s.filter (fun c => c ≠ '0')
  digits.take 3 |> String.toInt!

theorem leftmost_three_digits_of_num_four_ring_arrangements :
  leftmost_three_nonzero_digits (num_four_ring_arrangements 7 4) = 294 := by
  sorry

end leftmost_three_digits_of_num_four_ring_arrangements_l752_752606


namespace find_f_half_l752_752245

variable {R : Type*} [LinearOrderedField R]

def odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

def f_condition (f : R → R) : Prop := ∀ x : R, x < 0 → f x = 1 / (x + 1)

theorem find_f_half (f : R → R) (h_odd : odd_function f) (h_condition : f_condition f) : f (1 / 2) = -2 := by
  sorry

end find_f_half_l752_752245


namespace initial_number_of_cells_is_9_l752_752788

-- Definitions for conditions
def cell_death_rate : ℕ := 2
def cell_split_factor : ℕ := 2
def cells_at_hour_5 : ℕ := 164

-- The theorem we want to prove
theorem initial_number_of_cells_is_9 : 
  ∃ (n₀ : ℕ), (∀ h : ℕ, h ≤ 5, let n := (∃ k, (h = 5 - k) ∧ (n:(ℕ)) = cells_at_hour_5 →  
  (n = cells_at_hour_5)) ↔ 
  (h = 5 ∧ n = 164) ∨
  (h = 4 ∧ n = (n + 2) * 2 ∨
  (h = 3 ∧ n = (n + 2) * 2 ∨ 
  (h = 2 ∧ n = (n÷ 2) + 2 ∨ 
  (h = 1 ∧ n = (n÷ 2) + 2 ∨ 
  (h = 0 ∧ n = (n÷ 2) + 2 ∨
) := 
  n = 9
 := 
sorry

end initial_number_of_cells_is_9_l752_752788


namespace max_profit_at_l752_752826

def deposit_amount (k x : ℝ) : ℝ := k * x
def profit (k x : ℝ) : ℝ := 0.048 * k * x - k * x * x

theorem max_profit_at (k : ℝ) (hk : k > 0) : 
  ∃ x, 0 < x ∧ x < 0.048 ∧ profit k x = (λ x, 0.048 * k * x - k * x * x) 0.024 :=
by
  sorry

end max_profit_at_l752_752826


namespace f_at_23_pi_over_6_l752_752374

noncomputable
def f (x : ℝ) : ℝ := sorry

theorem f_at_23_pi_over_6 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f(x + Real.pi) = f(x) + Real.sin x)
  (h2 : ∀ x : ℝ, 0 ≤ x ∧ x < Real.pi → f(x) = 0)
  : f (23 * Real.pi / 6) = 1 / 2 :=
sorry

end f_at_23_pi_over_6_l752_752374


namespace max_value_unit_vectors_l752_752963

-- We start by defining vectors and their properties in the context of Lean.
variables {α : Type*} [normed_group α] [normed_space ℝ α]

-- a and b are unit vectors, meaning their norms are 1.
variables (a b : α) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1)

-- We need to prove that the maximum value of ‖a + b‖ + ‖a - b‖ is 2√2.
theorem max_value_unit_vectors (a b : α) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) :
  (‖a + b‖ + ‖a - b‖) ≤ 2 * real.sqrt 2 :=
sorry

end max_value_unit_vectors_l752_752963


namespace tank_capacity_percentage_l752_752812

noncomputable def Volume (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem tank_capacity_percentage {hA hB : ℝ} {circumferenceA circumferenceB : ℝ}
  (hA_eq : hA = 10) (circumferenceA_eq : circumferenceA = 11)
  (hB_eq : hB = 11) (circumferenceB_eq : circumferenceB = 10) :
  let rA := circumferenceA / (2 * π) in
  let rB := circumferenceB / (2 * π) in
  let VA := Volume rA hA in
  let VB := Volume rB hB in
  (VA / VB) * 100 = 44 :=
by {
  sorry
}

end tank_capacity_percentage_l752_752812


namespace simplify_complex_fraction_l752_752395

theorem simplify_complex_fraction :
  (∀ (z : ℂ), ((2 + 3 * complex.I) / (3 - 2 * complex.I)) ^ 1500 = 1) :=
by
  have h1 : (2 + 3 * complex.I) / (3 - 2 * complex.I) = complex.I := by
    calc
      (2 + 3 * complex.I) / (3 - 2 * complex.I) = ((2 + 3 * complex.I) * (3 + 2 * complex.I)) / ((3 - 2 * complex.I) * (3 + 2 * complex.I)) : by field_simp
      ... = (6 + 4 * complex.I + 9 * complex.I + 6 * (complex.I ^ 2)) / (9 - 4 * (complex.I ^ 2)) : by rw [←complex.sq, ←complex.I_mul_I]
      ... = (6 + 13 * complex.I - 6) / (9 + 4) : by rw [←complex.sq_eq_neg_one, ←complex.sq_eq_neg_one]
      ... = complex.I : by field_simp
  have h2 : complex.I ^ 4 = 1 := by
    rw [←complex.sq, ←complex.I_mul_I, complex.sq_eq_neg_one, complex.sq_eq_one]
  rw [←complex.I_mul_I] at h1 h2
  sorry

end simplify_complex_fraction_l752_752395


namespace max_area_triangle_PAB_l752_752240

-- Definitions of points and circle
def A := (2, 0)
def B := (0, -1)
def circle := {P : ℝ × ℝ | (P.1)^2 + (P.2 - 1)^2 = 1}

-- Line equation from points A and B
def line_eq (P : ℝ × ℝ) : Prop := P.1 - 2 * P.2 = 0

-- Maximum area of triangle PAB when P is on the circle
theorem max_area_triangle_PAB :
  ∀ P : ℝ × ℝ, P ∈ circle → ∀ d : ℝ, 
  (d = (abs (0 - 2)) / (sqrt 5)) →
  (1 + (sqrt 5) / 2) = 
  (1 / 2 * (sqrt 5) * ((d + 1))) := 
by
  intro P hP d hd,
  sorry

end max_area_triangle_PAB_l752_752240


namespace calculation1_calculation2_calculation3_calculation4_l752_752732

theorem calculation1 : 72 * 54 + 28 * 54 = 5400 := 
by sorry

theorem calculation2 : 60 * 25 * 8 = 12000 := 
by sorry

theorem calculation3 : 2790 / (250 * 12 - 2910) = 31 := 
by sorry

theorem calculation4 : (100 - 1456 / 26) * 78 = 3432 := 
by sorry

end calculation1_calculation2_calculation3_calculation4_l752_752732


namespace number_of_valid_pairs_l752_752934

theorem number_of_valid_pairs (a b : ℝ) :
  (∃ x y : ℤ, a * (x : ℝ) + b * (y : ℝ) = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) →
  ∃! pairs_count : ℕ, pairs_count = 72 :=
by
  sorry

end number_of_valid_pairs_l752_752934


namespace profit_equation_correct_l752_752831

theorem profit_equation_correct (x : ℝ) : 
  let original_selling_price := 36
  let purchase_price := 20
  let original_sales_volume := 200
  let price_increase_effect := 5
  let desired_profit := 1200
  let original_profit_per_unit := original_selling_price - purchase_price
  let new_selling_price := original_selling_price + x
  let new_sales_volume := original_sales_volume - price_increase_effect * x
  (original_profit_per_unit + x) * new_sales_volume = desired_profit :=
sorry

end profit_equation_correct_l752_752831


namespace isosceles_right_triangle_area_l752_752044

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (A : ℝ)
  (h_def : h = 6 * Real.sqrt 2)
  (rel_leg_hypotenuse : h = Real.sqrt 2 * l)
  (area_def : A = 1 / 2 * l * l) :
  A = 18 :=
by
  sorry

end isosceles_right_triangle_area_l752_752044


namespace complement_computation_l752_752818

open Set

theorem complement_computation (U A : Set ℕ) :
  U = {1, 2, 3, 4, 5, 6, 7} → A = {2, 4, 5} →
  U \ A = {1, 3, 6, 7} :=
by
  intros hU hA
  rw [hU, hA]
  ext
  simp
  sorry

end complement_computation_l752_752818


namespace city_rentals_cost_per_mile_l752_752172

theorem city_rentals_cost_per_mile (x : ℝ)
  (h₁ : 38.95 + 150 * x = 41.95 + 150 * 0.29) :
  x = 0.31 :=
by sorry

end city_rentals_cost_per_mile_l752_752172


namespace find_percent_increase_decrease_l752_752063

variable (R : ℝ) (p : ℝ)

-- Define conditions
def revenue_1996 := R + (p / 100) * R
def revenue_1997 := revenue_1996 - (p / 100) * revenue_1996
def target_revenue_1997 := R * (1 - 4 / 100)

-- Define the theorem to prove
theorem find_percent_increase_decrease : revenue_1997 R p = target_revenue_1997 R → p = 20 := by
  sorry

end find_percent_increase_decrease_l752_752063


namespace probability_sum_k_terms_l752_752966

variable {f g : ℝ → ℝ}
variable (a : ℝ)

-- Defining the hypotheses
axiom h1 : ∀ x, g x ≠ 0
axiom h2 : ∀ x, f'' x * g x < f x * g'' x
axiom h3 : ∀ x, f x = a^x * g x
axiom h4 : f 1 / g 1 + f (-1) / g (-1) = 5 / 2

-- Main theorem
theorem probability_sum_k_terms 
  (k : ℕ) (k_range : 1 ≤ k ∧ k ≤ 10)
  : ∑ i in Finset.range k, f i / g i > 15 / 16 ↔ k > 4 := sorry

end probability_sum_k_terms_l752_752966


namespace problem1_problem2_l752_752624

theorem problem1 (x : ℝ) : 
  let n := 14 in
  let expr := (1 / 2 + 2 * x) ^ n in
  let max_coeff := binomial n 7 * (1 / 2) ^ 7 * 2 ^ 7 in
  max_coeff = 3432 :=
by
  sorry

theorem problem2 (x : ℝ) : 
  let n := 12 in
  let sum_first_three_coeffs := binomial n 0 + binomial n 1 + binomial n 2 in
  sum_first_three_coeffs = 79 →
  let max_term := binomial n 10 * (1 / 2) ^ 12 * 4 ^ 10 * x ^ 10 in
  max_term = 16896 * x ^ 10 :=
by
  sorry

end problem1_problem2_l752_752624


namespace profit_from_third_year_max_annual_avg_net_profit_and_total_profit_l752_752160

-- Define the initial conditions
def initial_investment : ℝ := 720000
def first_year_expenditure : ℝ := 120000
def annual_increase_expenditure : ℝ := 40000
def annual_sales_income : ℝ := 500000

-- Define the function f(n): total net profit for the first n years
def f (n : ℕ) : ℝ := -2 * n^2 + 40 * n - 72

-- Prove that the factory starts making a profit from the third year (Part I)
theorem profit_from_third_year :
  ∃ (n : ℕ), n ≥ 3 ∧ f n > 0 := 
sorry

-- Define the annual average net profit function
def annual_avg_net_profit (n : ℕ) : ℝ := f n / n

-- Prove that the annual average net profit reaches its maximum at 6 years and total profit (Part II)
theorem max_annual_avg_net_profit_and_total_profit :
  (annual_avg_net_profit 6 = (40 - 2 * (6 + (36 / 6)))) ∧ (f 6 = 1440000) := 
sorry

end profit_from_third_year_max_annual_avg_net_profit_and_total_profit_l752_752160


namespace grace_sequence_positives_l752_752278

-- Grace's sequence is defined by the initial term and common difference
def sequence (n : ℕ) : ℤ := 47 - 4 * (n + 1)

-- Define the target property: the count of positive numbers in the sequence should be 11.
theorem grace_sequence_positives :
  (finset.range 20).count (λ n, sequence n > 0) = 11 := sorry

end grace_sequence_positives_l752_752278


namespace mr_li_age_l752_752461

theorem mr_li_age (xiaofang_age : ℕ) (h1 : xiaofang_age = 5)
  (h2 : ∀ t : ℕ, (t = 3) → ∀ mr_li_age_in_3_years : ℕ, (mr_li_age_in_3_years = xiaofang_age + t + 20)) :
  ∃ mr_li_age : ℕ, mr_li_age = 25 :=
by
  sorry

end mr_li_age_l752_752461


namespace problem_statement_l752_752995

noncomputable def a (x : ℝ): (ℝ × ℝ) := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
noncomputable def b (x : ℝ): (ℝ × ℝ) := (Real.cos (x / 2), -Real.sin (x / 2))
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def norm (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1^2 + u.2^2)
noncomputable def f (x : ℝ): ℝ := (dot_product (a x) (b x)) - norm (a x + b x)

theorem problem_statement (x : ℝ) (hx : -Real.pi / 3 ≤ x ∧ x ≤ Real.pi / 4) :
  dot_product (a x) (b x) = Real.cos (2 * x) ∧ 
  norm (a x + b x) = 2 * Real.cos x ∧ 
  ∀ (y : ℝ), y ∈ (set.Icc (-Real.pi / 3) (Real.pi / 4)) → f y ≤ -1 ∧ f y ≥ -3 / 2 := sorry

end problem_statement_l752_752995


namespace triangle_area_is_correct_l752_752051

-- Define the given conditions
def is_isosceles_right_triangle (h : ℝ) (l : ℝ) : Prop :=
  h = l * sqrt 2

def triangle_hypotenuse := 6 * sqrt 2  -- Given hypotenuse

-- Prove that the area of the triangle is 18 square units
theorem triangle_area_is_correct : 
  ∃ (l : ℝ), is_isosceles_right_triangle triangle_hypotenuse l ∧ (1/2) * l^2 = 18 :=
by
  sorry

end triangle_area_is_correct_l752_752051


namespace placemat_length_l752_752142

theorem placemat_length :
  let R := 5
  let w := 1
  let theta := Real.pi / 4
  let chord_length := 2 * R * Real.sin (theta / 2)
  let cos_value := (Real.sqrt 2) / 2
  let sin_pi8 := Real.sqrt ((1 - cos_value) / 2)
  let val := Real.sqrt 24.75 - 5 * Real.sqrt((1 + cos_value) / 2) + 1 / 2
  (sin_pi8 = Real.sin (Real.pi / 8)) →
  (cos_value = Real.cos (Real.pi / 4)) →
  (val = Real.sqrt 24.75 - 5 * Real.sqrt((1 + (Real.sqrt 2) / 2) / 2) + 1 / 2) →
  x = val :=
sorry

end placemat_length_l752_752142


namespace sum_of_cubes_l752_752096

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l752_752096


namespace trig_eq_solutions_l752_752582

theorem trig_eq_solutions (k m n : ℤ) (x : ℝ) :
  2 + cos x^2 + cos (4 * x) + cos (2 * x) + 2 * sin (3 * x) * sin (7 * x) + sin (7 * x)^2 = cos (π * k / 2022)^2 ↔
  (k = 2022 * m) ∧ (x = π / 4 + n * π / 2) :=
sorry

end trig_eq_solutions_l752_752582


namespace sum_of_f10_values_l752_752039

noncomputable def f : ℕ → ℝ := sorry

axiom f_cond1 : f 1 = 4

axiom f_cond2 : ∀ (m n : ℕ), m ≥ n → f (m + n) + f (m - n) = (f (2 * m) + f (2 * n)) / 2

theorem sum_of_f10_values : f 10 = 400 :=
sorry

end sum_of_f10_values_l752_752039


namespace solution_l752_752334

noncomputable def problem :=
  let A : (ℝ × ℝ × ℝ) := (a, 0, 0) in
  let B : (ℝ × ℝ × ℝ) := (0, b, 0) in
  let C : (ℝ × ℝ × ℝ) := (0, 0, c) in
  let plane_eq (x y z a b c : ℝ) : Prop := (x / a + y / b + z / c = 1) in
  let distance_eq (a b c : ℝ) : Prop :=
    (1 / (Real.sqrt ((1 / (a^2)) + 1 / (b^2) + 1 / (c^2)))) = 2 in
  let centroid (a b c : ℝ) : ℝ × ℝ × ℝ := (a / 3, b / 3, c / 3) in
  let (p, q, r) := centroid a b c in
  (plane_eq a 0 0 a b c) ∧
  (plane_eq 0 b 0 a b c) ∧
  (plane_eq 0 0 c a b c) ∧
  (distance_eq a b c) → (1 / (p^2) + 1 / (q^2) + 1 / (r^2)) = 2.25

theorem solution : problem :=
sorry

end solution_l752_752334


namespace least_perimeter_of_triangle_DEF_l752_752669

-- Conditions
variables {D E F d e f : ℕ}
variables {cos_D : ℝ} {cos_E : ℝ} {cos_F : ℝ}

hypothesis h_cos_D : cos_D = 3/5
hypothesis h_cos_E : cos_E = 9/10
hypothesis h_cos_F : cos_F = -1/3
hypothesis h_integer_sides : 0 < d ∧ 0 < e ∧ 0 < f
hypothesis h_angle_sum : D + E + F = π

noncomputable def sin_D := Real.sqrt (1 - cos_D^2)
noncomputable def sin_E := Real.sqrt (1 - cos_E^2)
noncomputable def sin_F := Real.sqrt (1 - cos_F^2)

-- Law of Sines
axiom law_of_sines : d / sin_D = e / sin_E ∧ e / sin_E = f / sin_F

theorem least_perimeter_of_triangle_DEF : d + e + f = 50 :=
by
   -- Here, we put the steps to prove the given perimeter
   sorry

end least_perimeter_of_triangle_DEF_l752_752669


namespace find_k_l752_752273

variables {a b : ℝ^3} (k : ℝ)

-- Given: |a| = |b| = 1
def magnitude_a : real := 1
def magnitude_b : real := 1

-- Given: a is perpendicular to b
def perpendicular_a_b : Prop := a ⬝ b = 0

-- Given: (2a + 3b) is perpendicular to (k*a - 4b)
def perpendicular_combination : Prop := (2 * a + 3 * b) ⬝ (k * a - 4 * b) = 0

-- Prove: The value of k is 6.
theorem find_k (h1 : magnitude_a = 1) 
                (h2 : magnitude_b = 1)
                (h3 : perpendicular_a_b)
                (h4 : perpendicular_combination) : k = 6 :=
sorry

end find_k_l752_752273


namespace terminal_side_angles_l752_752459

theorem terminal_side_angles (k : ℤ) (β : ℝ) :
  β = (Real.pi / 3) + 2 * k * Real.pi → -2 * Real.pi ≤ β ∧ β < 4 * Real.pi :=
by
  sorry

end terminal_side_angles_l752_752459


namespace probability_both_red_l752_752824

-- Definitions for the problem conditions
def total_balls := 16
def red_balls := 7
def blue_balls := 5
def green_balls := 4
def first_red_prob := (red_balls : ℚ) / total_balls
def second_red_given_first_red_prob := (red_balls - 1 : ℚ) / (total_balls - 1)

-- The statement to be proved
theorem probability_both_red : (first_red_prob * second_red_given_first_red_prob) = (7 : ℚ) / 40 :=
by 
  -- Proof goes here
  sorry

end probability_both_red_l752_752824


namespace range_of_a_l752_752270

def P (x : ℝ) : Prop := x^2 ≤ 1

def M (a : ℝ) : Set ℝ := {a}

theorem range_of_a (a : ℝ) (h : ∀ x, (P x ∨ x = a) ↔ P x) : P a :=
by
  sorry

end range_of_a_l752_752270


namespace isosceles_right_triangle_area_l752_752046

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (A : ℝ)
  (h_def : h = 6 * Real.sqrt 2)
  (rel_leg_hypotenuse : h = Real.sqrt 2 * l)
  (area_def : A = 1 / 2 * l * l) :
  A = 18 :=
by
  sorry

end isosceles_right_triangle_area_l752_752046


namespace spend_money_l752_752447

theorem spend_money (n : ℕ) (h : n > 7) : ∃ a b : ℕ, 3 * a + 5 * b = n :=
by
  sorry

end spend_money_l752_752447


namespace maize_storage_l752_752509

theorem maize_storage (x : ℝ)
  (h1 : 24 * x - 5 + 8 = 27) : x = 1 :=
  sorry

end maize_storage_l752_752509


namespace globe_division_l752_752842

theorem globe_division (parallels meridians : ℕ) (h_parallels : parallels = 17) (h_meridians : meridians = 24) : 
  (meridians * (parallels + 1)) = 432 :=
by
  rw [h_parallels, h_meridians]
  simp
  sorry

end globe_division_l752_752842


namespace odd_function_at_origin_zero_l752_752491

variables {ℝ : Type*} [linear_ordered_field ℝ]

def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem odd_function_at_origin_zero (f : ℝ → ℝ) (h : odd_function f) : f 0 = 0 :=
sorry

end odd_function_at_origin_zero_l752_752491


namespace ordered_quadruple_solution_exists_l752_752280

theorem ordered_quadruple_solution_exists (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : 
  a^2 * b = c ∧ b * c^2 = a ∧ c * a^2 = b ∧ a + b + c = d → (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3) :=
by
  sorry

end ordered_quadruple_solution_exists_l752_752280


namespace combined_tax_rate_l752_752884

-- Definitions and conditions
def tax_rate_mork : ℝ := 0.45
def tax_rate_mindy : ℝ := 0.20
def income_ratio_mindy_to_mork : ℝ := 4

-- Theorem statement
theorem combined_tax_rate :
  ∀ (M : ℝ), (tax_rate_mork * M + tax_rate_mindy * (income_ratio_mindy_to_mork * M)) / (M + income_ratio_mindy_to_mork * M) = 0.25 :=
by
  intros M
  sorry

end combined_tax_rate_l752_752884


namespace original_number_value_l752_752470

theorem original_number_value (t : ℝ) 
  (h1 : t * 1.125 - t * 0.75 = 30) : 
  t = 80 := 
begin
  sorry,
end

end original_number_value_l752_752470


namespace sum_of_cubes_l752_752099

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752099


namespace unique_handshakes_l752_752820

-- Define the circular arrangement and handshakes conditions
def num_people := 30
def handshakes_per_person := 2

theorem unique_handshakes : 
  (num_people * handshakes_per_person) / 2 = 30 :=
by
  -- Sorry is used here as a placeholder for the proof
  sorry

end unique_handshakes_l752_752820


namespace cubes_sum_l752_752113

theorem cubes_sum (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end cubes_sum_l752_752113


namespace pete_walked_blocks_l752_752733

theorem pete_walked_blocks
  (bus_distance : ℕ)
  (total_distance : ℕ)
  (blocks_traveled_by_bus : bus_distance = 20)
  (total_blocks_traveled : total_distance = 50)
  : ∃ x : ℕ, 2 * x + 2 * bus_distance = total_distance ∧ x = 5 := 
by {
  use 5,
  split,
  {
    rw [blocks_traveled_by_bus, total_blocks_traveled],
    norm_num,
  },
  {
    norm_num,
  }
}

end pete_walked_blocks_l752_752733


namespace coefficient_of_x_squared_is_2_l752_752570

-- Define the polynomial expression
def polynomial : polynomial ℚ :=
  5 * (monomial 1 1 - monomial 4 1)
  - 4 * (monomial 2 1 - 2 * monomial 4 1 + monomial 6 1)
  + 3 * (2 * monomial 2 1 - monomial 8 1)

-- Define the coefficient extraction function
def coefficient_of_x_squared := polynomial.coeff 2

-- State the theorem to prove
theorem coefficient_of_x_squared_is_2 : coefficient_of_x_squared = 2 :=
  sorry

end coefficient_of_x_squared_is_2_l752_752570


namespace eval_g_231_l752_752632

def g (a b c : ℤ) : ℚ :=
  (c ^ 2 + a ^ 2) / (c - b)

theorem eval_g_231 : g 2 (-3) 1 = 5 / 4 :=
by
  sorry

end eval_g_231_l752_752632


namespace expression_interval_l752_752716

noncomputable def expression (x y z w : ℝ) : ℝ :=
  (Real.sqrt(x^2 + (1 - y)^2) +
   Real.sqrt(y^2 + (1 - z)^2) +
   Real.sqrt(z^2 + (1 - w)^2) +
   Real.sqrt(w^2 + (1 - x)^2))

theorem expression_interval (x y z w : ℝ)
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hw : 0 ≤ w)
  (hx1 : x ≤ 1) (hy1 : y ≤ 1) (hz1 : z ≤ 1) (hw1 : w ≤ 1) :
  2 * Real.sqrt 2 ≤ expression x y z w ∧ expression x y z w ≤ 4 :=
by
  sorry

end expression_interval_l752_752716


namespace arithmetic_problem_l752_752431

theorem arithmetic_problem : 987 + 113 - 1000 = 100 :=
by
  sorry

end arithmetic_problem_l752_752431


namespace garbage_collection_l752_752911

theorem garbage_collection (Daliah Dewei Zane : ℝ) 
(h1 : Daliah = 17.5)
(h2 : Dewei = Daliah - 2)
(h3 : Zane = 4 * Dewei) :
Zane = 62 :=
sorry

end garbage_collection_l752_752911


namespace intersection_point_of_lines_l752_752187

theorem intersection_point_of_lines : 
  (∃ x y : ℚ, (8 * x - 3 * y = 5) ∧ (5 * x + 2 * y = 20)) ↔ (x = 70 / 31 ∧ y = 135 / 31) :=
sorry

end intersection_point_of_lines_l752_752187


namespace sum_of_cubes_l752_752107

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 :=
by
  sorry

end sum_of_cubes_l752_752107


namespace sixth_number_in_sequence_l752_752427

def sequence : List ℕ := [11, 23, 47, 83, 131, 191, 263, 347, 443, 551, 671]

theorem sixth_number_in_sequence : sequence.get? 5 = some 191 :=
by
  sorry

end sixth_number_in_sequence_l752_752427


namespace polynomial_factor_implies_a_minus_b_l752_752248

theorem polynomial_factor_implies_a_minus_b (a b : ℝ) :
  (∀ x y : ℝ, (x + y - 2) ∣ (x^2 + a * x * y + b * y^2 - 5 * x + y + 6))
  → a - b = 1 :=
by
  intro h
  -- Proof needs to be filled in
  sorry

end polynomial_factor_implies_a_minus_b_l752_752248


namespace angle_between_hands_l752_752317

theorem angle_between_hands (h m : ℕ) (Hh : h = 9) (Hm : m = 15) : 
  let total_degrees := 360
  let minutes_in_circle := 60
  let degrees_per_minute := total_degrees / minutes_in_circle
  let hours_in_circle := 12
  let degrees_per_hour := total_degrees / hours_in_circle
  let minute_angle := m * degrees_per_minute
  let hour_fraction := m / 60
  let hour_angle := (h * degrees_per_hour) + (hour_fraction * degrees_per_hour)
  let angle_difference := abs (minute_angle - hour_angle)
  let smallest_angle := min angle_difference (total_degrees - angle_difference)
  smallest_angle = total_degrees / 2 + 15 :=
sorry

end angle_between_hands_l752_752317


namespace no_ingredient_pies_max_l752_752563

theorem no_ingredient_pies_max :
  ∃ (total apple blueberry cream chocolate no_ingredient : ℕ),
    total = 48 ∧
    apple = 24 ∧
    blueberry = 16 ∧
    cream = 18 ∧
    chocolate = 12 ∧
    no_ingredient = total - (apple + blueberry + chocolate - min apple blueberry - min apple chocolate - min blueberry chocolate) - cream ∧
    no_ingredient = 10 := sorry

end no_ingredient_pies_max_l752_752563


namespace hallie_reads_pages_l752_752996

variable (P : ℕ) -- Number of pages read on the first day.

theorem hallie_reads_pages :
  ∃ P : ℕ, let second_day := 2 * P,
               third_day := 2 * P + 10,
               fourth_day := 29,
               total_pages := 354
           in P + second_day + third_day + fourth_day = total_pages ∧ P = 63 := 
by
  sorry

end hallie_reads_pages_l752_752996


namespace arrangements_of_masters_and_apprentices_l752_752787

theorem arrangements_of_masters_and_apprentices : 
  ∃ n : ℕ, n = 48 ∧ 
     let pairs := 3 
     let ways_to_arrange_pairs := pairs.factorial 
     let ways_to_arrange_within_pairs := 2 ^ pairs 
     ways_to_arrange_pairs * ways_to_arrange_within_pairs = n := 
sorry

end arrangements_of_masters_and_apprentices_l752_752787


namespace cashier_adjustment_l752_752481

-- Define the conditions
variables {y : ℝ}

-- Error calculation given the conditions
def half_dollar_error (y : ℝ) : ℝ := 0.50 * y
def five_dollar_error (y : ℝ) : ℝ := 5 * y
def total_error (y : ℝ) : ℝ := half_dollar_error y + five_dollar_error y

-- Theorem statement
theorem cashier_adjustment (y : ℝ) : total_error y = 5.50 * y :=
sorry

end cashier_adjustment_l752_752481


namespace problem_statement_l752_752262

noncomputable def f (x : ℝ) : ℝ := 2 / (2019^x + 1) + Real.sin x

noncomputable def f' (x : ℝ) := (deriv f) x

theorem problem_statement :
  f 2018 + f (-2018) + f' 2019 - f' (-2019) = 2 :=
by {
  sorry
}

end problem_statement_l752_752262


namespace evaluate_expression_at_minus_one_l752_752747

theorem evaluate_expression_at_minus_one :
  ((-1 + 1) * (-1 - 2) + 2 * (-1 + 4) * (-1 - 4)) = -30 := by
  sorry

end evaluate_expression_at_minus_one_l752_752747


namespace simplify_tan_alpha_l752_752137

noncomputable def f (α : ℝ) : ℝ :=
(Real.sin (Real.pi / 2 + α) + Real.sin (-Real.pi - α)) /
  (3 * Real.cos (2 * Real.pi - α) + Real.cos (3 * Real.pi / 2 - α))

theorem simplify_tan_alpha (α : ℝ) (h : Real.tan α = 1) : f α = 1 := by
  sorry

end simplify_tan_alpha_l752_752137


namespace angle_CAD_is_15_degrees_l752_752363

theorem angle_CAD_is_15_degrees
    (A B C D F : Type)
    (is_eq_triangle : equilateral_triangle A B C)
    (is_rectangle : rectangle B C D F)
    (F_on_extension : lies_on_extension F B C) :
  measure_of_angle CAD = 15 :=
sorry

end angle_CAD_is_15_degrees_l752_752363


namespace tangent_equation_line_AB_fixed_point_range_OE_OF_l752_752228

-- Problem (1): Tangent Equation
theorem tangent_equation (P : ℝ × ℝ) (C : ℝ × ℝ × ℝ) (x y : ℝ) : 
  P = (-1, 1) →
  C = (2, 0, 1) → -- Center (2, 0) with radius 1
  (x - 2)^2 + y^2 = 1 →
  (y - 1 = 0 ∨ 3x + 4y - 1 = 0) :=
by
  sorry

-- Problem (2): Fixed Point of AB
theorem line_AB_fixed_point (C : ℝ × ℝ × ℝ) (P : ℝ × ℝ) (x y : ℝ) : 
  ∀ t : ℝ, P = (t, -t) →
    C = (2, 0, 1) → -- Center (2, 0) with radius 1
    line_AB (t, -t) passes through (3/2, -1/2) :=
by
  sorry

-- Problem (3): Range of OE · OF
theorem range_OE_OF (C : ℝ × ℝ × ℝ) (m : ℝ) : 
  C = (2, 0, 1) → -- Center (2, 0) with radius 1
  -2 - sqrt 2 < m ∧ m < -2 + sqrt 2 →
  (let x E F := origin coordinates) in
    (OE dot OF) ∈ Icc 2 (5 + 2*sqrt 2) :=
by
  sorry

end tangent_equation_line_AB_fixed_point_range_OE_OF_l752_752228


namespace trapezoid_sides_l752_752765

-- Definitions
def is_isosceles_trapezoid (a b c d : ℝ) : Prop :=
  a = b ∧ c = d ∧ a ≠ c ∧ a ≠ d

def circumscribed_circle (a b c d : ℝ) : Prop :=
  a + c = b + d

def area_of_trapezoid (a b c d h : ℝ) : ℝ :=
  (a + c) * h / 2

-- Given conditions
def area := 32.0
def angle_A := 30.0
def height := 4.0
def base_AB := 8.0
def base_CD := 8.0
def side_BC := 8 - 4 * Real.sqrt 3
def side_AD := 8 + 4 * Real.sqrt 3

theorem trapezoid_sides :
  ∀ (S h : ℝ), S = 32 ∧ h = 4 →
  ∃ (a b c d : ℝ),
  is_isosceles_trapezoid a b c d ∧
  circumscribed_circle a b c d ∧
  area_of_trapezoid b c a d h = S ∧
  (a, b, c, d) = (base_AB, base_CD, side_BC, side_AD) :=
by
  intros S h hS
  use base_AB, base_CD, side_BC, side_AD
  sorry

end trapezoid_sides_l752_752765


namespace initial_solution_amount_l752_752560

theorem initial_solution_amount (x : ℝ) (h1 : x - 200 + 1000 = 2000) : x = 1200 := by
  sorry

end initial_solution_amount_l752_752560


namespace tom_gave_fred_balloons_l752_752439

theorem tom_gave_fred_balloons (initial_balloons : ℕ) (current_balloons : ℕ) (given_balloons : ℕ) :
  initial_balloons = 30 → current_balloons = 14 → given_balloons = initial_balloons - current_balloons → given_balloons = 16 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3.symm

end tom_gave_fred_balloons_l752_752439


namespace quantile_60_percent_l752_752405

def data_set : list ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def quantile (p : ℝ) (data : list ℕ) : ℕ :=
  let n := data.length
  let pos := p * (n : ℝ) in
  if h : pos = pos.to_nat then data.nth_le (pos.to_nat - 1) sorry
  else data.nth_le pos.ceil.pred sorry

theorem quantile_60_percent : quantile 0.6 data_set = 5 := 
by 
  -- The actual proof would go here, but is not required
  sorry

end quantile_60_percent_l752_752405


namespace a4_is_5_l752_752236

-- Definitions based on the given conditions in the problem
def sum_arith_seq (n a1 d : ℤ) : ℤ := n * a1 + (n * (n-1)) / 2 * d

def S6 : ℤ := 24
def S9 : ℤ := 63

-- The proof problem: we need to prove that a4 = 5 given the conditions
theorem a4_is_5 (a1 d : ℤ) (h_S6 : sum_arith_seq 6 a1 d = S6) (h_S9 : sum_arith_seq 9 a1 d = S9) : 
  a1 + 3 * d = 5 :=
sorry

end a4_is_5_l752_752236


namespace complex_repair_cost_l752_752323

theorem complex_repair_cost
  (charge_tire : ℕ)
  (cost_part_tire : ℕ)
  (num_tires : ℕ)
  (charge_complex : ℕ)
  (num_complex : ℕ)
  (profit_retail : ℕ)
  (fixed_expenses : ℕ)
  (total_profit : ℕ)
  (profit_tire : ℕ := charge_tire - cost_part_tire)
  (total_profit_tire : ℕ := num_tires * profit_tire)
  (total_revenue_complex : ℕ := num_complex * charge_complex)
  (initial_profit : ℕ :=
    total_profit_tire + profit_retail - fixed_expenses)
  (needed_profit_complex : ℕ := total_profit - initial_profit) :
  needed_profit_complex = 100 / num_complex :=
by
  sorry

end complex_repair_cost_l752_752323


namespace perimeter_sum_l752_752698

structure Point :=
  (x : ℤ)
  (y : ℤ)

def distance (P Q : Point) : ℝ := 
  real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

noncomputable def perimeter : ℝ :=
  let A : Point := ⟨1, 1⟩;
  let B : Point := ⟨4, 5⟩;
  let C : Point := ⟨6, 3⟩;
  let D : Point := ⟨9, 1⟩
  in distance A B + distance B C + distance C D + distance D A

theorem perimeter_sum : 
  ∃ (c d e f : ℤ), 
  c * real.sqrt d + e * real.sqrt f = perimeter ∧ 
  c + d + e + f = 18 := 
by {
  sorry
}

end perimeter_sum_l752_752698


namespace cyclic_quadrilateral_partition_l752_752008

theorem cyclic_quadrilateral_partition 
  (n : ℕ) 
  (odd_n : n % 2 = 1) 
  (n_ge_3 : n ≥ 3)
  (ABCD : Type) 
  (is_partitioned_into_cyclic_quads : (quadrilateral: Type) → (partition : set quadrilateral) → Prop )
  (h : is_partitioned_into_cyclic_quads ABCD n) : 
  is_cyclic ABCD :=
sorry

end cyclic_quadrilateral_partition_l752_752008


namespace term_with_largest_binomial_and_coefficient_l752_752625

theorem term_with_largest_binomial_and_coefficient :
  ∃ (n : ℕ) (T₃ T₄ T₅ : ℝ), 
    (n = 5) ∧
    ((sqrt[3] x^2 + 3 * x^2)^n) = 992 + (sum_i^n C_n^i) ∧
    T₃ = 90 * x^6 ∧
    T₄ = 270 * x^(22/3) ∧
    T₅ = 405 * x^(26/3) :=
by
  let n := 5
  let T₃ := 90 * x^6
  let T₄ := 270 * x^(22/3)
  let T₅ := 405 * x^(26/3)
  exists n, T₃, T₄, T₅
  split
  { exact rfl }
  split
  { sorry } -- prove sum of coefficients condition
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }

end term_with_largest_binomial_and_coefficient_l752_752625


namespace circle_standard_equation_l752_752780

theorem circle_standard_equation (x y : ℝ) :
  let center_x := 2
  let center_y := -1
  let radius := 3
  (center_x = 2) ∧ (center_y = -1) ∧ (radius = 3) → (x - center_x) ^ 2 + (y - center_y) ^ 2 = radius ^ 2 :=
by
  intros
  sorry

end circle_standard_equation_l752_752780


namespace charlie_and_delta_can_purchase_l752_752825

def num_cookies : ℕ := 7
def num_cupcakes : ℕ := 4
def total_items : ℕ := 4

def ways_purchase (c d : ℕ) : ℕ :=
  if d > c then 0 else
    (Finset.card (Finset.powersetLen c (Finset.range (num_cookies + num_cupcakes)))
    * (if d = 0 then 1
       else (Finset.card (Finset.powersetLen d (Finset.range num_cookies)) + (num_cookies * (num_cookies - 1)) / 2)))

def total_ways : ℕ :=
  ways_purchase 4 0 + ways_purchase 3 1 + ways_purchase 2 2 + ways_purchase 1 3 + ways_purchase 0 4

theorem charlie_and_delta_can_purchase : total_ways = 4054 :=
  by sorry

end charlie_and_delta_can_purchase_l752_752825


namespace min_value_of_f_l752_752774

noncomputable def f (x : ℝ) : ℝ := x^3 - 12 * x

theorem min_value_of_f :
  ∃ x ∈ set.Icc (-4 : ℝ) (4 : ℝ), ∀ y ∈ set.Icc (-4 : ℝ) (4 : ℝ), f x ≤ f y ∧ f x = -16 :=
begin
  sorry
end

end min_value_of_f_l752_752774


namespace jack_kids_solution_l752_752320

def jack_kids (k : ℕ) : Prop :=
  7 * 3 * k = 63

theorem jack_kids_solution : jack_kids 3 :=
by
  sorry

end jack_kids_solution_l752_752320


namespace ellipse_standard_equation_l752_752956

theorem ellipse_standard_equation
    (h1 : ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ ((x - 2) = 0 ∨ (x + 2) = 0 ∨ y = 0)) 
    (h2 : (- sqrt 3, 0) ∈ {(x, 0) | x^2 / 4 + 0 = 1}):
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 :=
by
  sorry

end ellipse_standard_equation_l752_752956


namespace find_f_3_l752_752558

def quadratic_function (b c : ℝ) (x : ℝ) : ℝ :=
- x^2 + b * x + c

theorem find_f_3 (b c : ℝ) (h1 : quadratic_function b c 2 + quadratic_function b c 4 = 12138)
                       (h2 : 3*b + c = 6079) :
  quadratic_function b c 3 = 6070 := 
by
  sorry

end find_f_3_l752_752558


namespace sum_of_inversion_counts_of_all_permutations_l752_752355

noncomputable def sum_of_inversion_counts (n : ℕ) (fixed_val : ℕ) (fixed_pos : ℕ) : ℕ :=
  if n = 6 ∧ fixed_val = 4 ∧ fixed_pos = 3 then 120 else 0

theorem sum_of_inversion_counts_of_all_permutations :
  sum_of_inversion_counts 6 4 3 = 120 :=
by
  sorry

end sum_of_inversion_counts_of_all_permutations_l752_752355


namespace equivalent_range_of_x_l752_752458

theorem equivalent_range_of_x (b : Fin 20 → { x // x = 0 ∨ x = 3 }) : 
    let x : ℝ := ∑ i in Finset.range 20, (b ⟨i, Finset.mem_range.2 (Nat.lt_succ_self _)⟩ * 4^-(i+1))
    in (0 ≤ x ∧ x < 1/4) ∨ (3/4 ≤ x ∧ x < 1) :=
by
  sorry

end equivalent_range_of_x_l752_752458


namespace cousin_reads_book_time_l752_752728

theorem cousin_reads_book_time
  (my_reading_time_hours : ℕ)
  (conversion_factor_hours_to_minutes : ℕ)
  (speed_ratio : ℕ)
  (my_reading_time_minutes : ℕ)
  (cousin_time_minutes : ℕ) :
  my_reading_time_hours = 3 →
  conversion_factor_hours_to_minutes = 60 →
  speed_ratio = 4 →
  my_reading_time_minutes = my_reading_time_hours * conversion_factor_hours_to_minutes →
  cousin_time_minutes = my_reading_time_minutes / speed_ratio →
  cousin_time_minutes = 45 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3] at h5
  have h6 : my_reading_time_minutes = 180 := by rw [h1, h2]; exact Nat.mul_comm 3 60
  rw h6 at h5
  exact h5
  sorry

end cousin_reads_book_time_l752_752728


namespace gain_percent_l752_752661

theorem gain_percent (C S S_d : ℝ) 
  (h1 : 50 * C = 20 * S) 
  (h2 : S_d = S * (1 - 0.15)) : 
  ((S_d - C) / C) * 100 = 112.5 := 
by 
  sorry

end gain_percent_l752_752661


namespace game_winning_strategy_l752_752885

theorem game_winning_strategy (n : ℕ) (h : n > 6) :
  (∃ k, n = 3 * k) ↔ (∀ strategy : ℝ → ℝ, strategy_winner strategy = Jenn) :=
sorry

end game_winning_strategy_l752_752885
